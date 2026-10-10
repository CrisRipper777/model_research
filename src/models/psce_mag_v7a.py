from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import remove_self_loops

from src.data.semantic_candidates import load_semantic_candidate_cache
from src.models.mvcge_mag_v21 import _ExpertTransform, _IntrinsicProjector


_MODALITIES = ("text", "visual")
_VARIANTS = ("a0_raw", "a1_static_dual", "a2_conditional_dual")


class ChunkedWeightedSpMM(Function):
    """Sparse weighted source-to-target sum with edge-chunk recomputation.

    The custom backward gathers source states again, so autograd does not keep
    an E-by-hidden message tensor for every edge.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor, chunk_size: int):
        src, dst = edge_index
        output = torch.zeros_like(x)
        chunk_size = int(chunk_size)
        for start in range(0, src.numel(), chunk_size):
            end = min(start + chunk_size, src.numel())
            source = x.index_select(0, src[start:end])
            message = source * weight[start:end].unsqueeze(-1)
            output.index_add_(0, dst[start:end], message)
        ctx.save_for_backward(x, edge_index, weight)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, edge_index, weight = ctx.saved_tensors
        src, dst = edge_index
        grad_x = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[2] else None
        for start in range(0, src.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, src.numel())
            source_ids = src[start:end]
            target_ids = dst[start:end]
            grad_target = grad_output.index_select(0, target_ids)
            source = x.index_select(0, source_ids)
            if grad_x is not None:
                grad_x.index_add_(
                    0,
                    source_ids,
                    grad_target * weight[start:end].unsqueeze(-1),
                )
            if grad_weight is not None:
                grad_weight[start:end] = (grad_target * source).sum(dim=-1)
        return grad_x, None, grad_weight, None


def chunked_weighted_spmm(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 32768,
) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("semantic edge_index must have shape [2,E]")
    if edge_index.size(1) != weight.numel():
        raise ValueError("semantic edge and weight lengths differ")
    return ChunkedWeightedSpMM.apply(x, edge_index, weight, int(chunk_size))


class Model(nn.Module):
    """Physical-Semantic Collaborative Experts for full-graph MAG NC."""

    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    supports_link_prediction = False
    record_run_telemetry = True
    collect_v7_diagnostics = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("psce_mag_v7a requires positive Text and Visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError("psce_mag_v7a requires exact [Text, Visual] feature layout")

        self.variant = str(model_cfg.get("variant", "a2_conditional_dual"))
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        self.requires_global_semantic_candidates = self.variant != "a0_raw"
        self.num_nodes = int(data_info.get("num_nodes", 0))
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.physical_hops = int(model_cfg.get("physical_hops", 4))
        self.semantic_hops = int(model_cfg.get("semantic_hops", 2))
        self.num_experts = int(model_cfg.get("num_experts", 4))
        self.top_k = int(model_cfg.get("top_k", 2))
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.expert_feature_scale = float(model_cfg.get("expert_feature_scale", 0.1))
        self.router_dim = int(model_cfg.get("router_dim", 64))
        self.router_hidden_dim = int(model_cfg.get("router_hidden_dim", 128))
        self.modality_embed_dim = int(model_cfg.get("modality_embed_dim", 16))
        self.relation_dim = int(model_cfg.get("relation_dim", 32))
        self.relation_hidden_dim = int(model_cfg.get("relation_hidden_dim", 64))
        if int(model_cfg.get("text_knn_k", 8)) != 8 or int(model_cfg.get("visual_knn_k", 8)) != 8:
            raise ValueError("V7A freezes both modality KNN searches at K=8")
        self.semantic_edge_weight_init = float(
            model_cfg.get("semantic_edge_weight_init", 0.8)
        )
        self.semantic_strength_init = float(model_cfg.get("semantic_strength_init", 0.2))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.balance_weight = float(model_cfg.get("balance_weight", 0.01))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.relation_chunk_size = int(model_cfg.get("relation_chunk_size", 32768))
        self.propagation_chunk_size = int(model_cfg.get("propagation_chunk_size", 32768))
        if (self.hidden_dim, self.physical_hops, self.semantic_hops, self.num_experts, self.top_k) != (
            256,
            4,
            2,
            4,
            2,
        ):
            raise ValueError("V7A freezes hidden=256, physical_hops=4, semantic_hops=2, experts=4, Top-2")
        if (self.expert_bottleneck, self.relation_dim, self.relation_hidden_dim) != (64, 32, 64):
            raise ValueError("V7A freezes expert/relation dimensions to 64/32/64")
        if not 0.0 < self.semantic_edge_weight_init < 1.0:
            raise ValueError("semantic_edge_weight_init must be in (0,1)")
        if not 0.0 < self.semantic_strength_init < 1.0:
            raise ValueError("semantic_strength_init must be in (0,1)")
        if self.eps <= 0 or self.relation_chunk_size < 1 or self.propagation_chunk_size < 1:
            raise ValueError("eps and chunk sizes must be positive")
        self.out_dim = self.hidden_dim

        # This common tree/order maps directly to the V4A R0 Raw trunk.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
                "visual": _IntrinsicProjector(
                    self.visual_dim, self.hidden_dim, self.dropout_p
                ),
            }
        )
        hadamard = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.alpha_raw = nn.Parameter(0.5 * hadamard)
        self.experts = nn.ModuleList(
            [
                _ExpertTransform(self.hidden_dim, self.expert_bottleneck, self.dropout_p)
                for _ in range(self.num_experts)
            ]
        )
        self.router_projectors = nn.ModuleDict(
            {m: nn.Linear(self.hidden_dim, self.router_dim) for m in _MODALITIES}
        )
        self.router_norms = nn.ModuleDict(
            {m: nn.LayerNorm(self.router_dim) for m in _MODALITIES}
        )
        self.modality_embedding = nn.Embedding(2, self.modality_embed_dim)
        physical_router_input_dim = 4 * self.router_dim + self.modality_embed_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(physical_router_input_dim, self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.selection_head = nn.Linear(self.router_hidden_dim, self.num_experts)
        self.strength_head = nn.Linear(self.router_hidden_dim, 1)
        nn.init.zeros_(self.strength_head.weight)
        nn.init.constant_(self.strength_head.bias, -1.15)
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # Semantic-only initialization is isolated so the legacy Raw trunk and
        # the NC classifier consume the same RNG stream as V4A R0.
        with torch.random.fork_rng(devices=[]):
            self.relation_projections = nn.ModuleDict(
                {
                    "text": nn.Linear(self.hidden_dim, self.relation_dim, bias=False),
                    "visual": nn.Linear(self.hidden_dim, self.relation_dim, bias=False),
                }
            )
            self.relation_scorer = nn.Sequential(
                nn.Linear(4 * self.relation_dim, self.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(self.relation_hidden_dim, 1),
            )
            nn.init.normal_(self.relation_scorer[-1].weight, mean=0.0, std=0.005)
            edge_logit = math.log(
                self.semantic_edge_weight_init / (1.0 - self.semantic_edge_weight_init)
            )
            nn.init.constant_(self.relation_scorer[-1].bias, edge_logit)
            self.semantic_coefficients_raw = nn.Parameter(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)],
                        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
                    ],
                    dtype=torch.float32,
                )
            )
            self.semantic_context_norms = nn.ModuleDict(
                {
                    "intrinsic": nn.LayerNorm(self.hidden_dim),
                    "physical": nn.LayerNorm(self.hidden_dim),
                    "semantic": nn.LayerNorm(self.hidden_dim),
                }
            )
            semantic_router_input_dim = 3 * self.hidden_dim + self.modality_embed_dim + 1
            self.semantic_context_encoder = nn.Sequential(
                nn.Linear(semantic_router_input_dim, self.router_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
            )
            self.semantic_selection_head = nn.Linear(
                self.router_hidden_dim, self.num_experts
            )
            self.semantic_strength_head = nn.Linear(self.router_hidden_dim, 1)
            nn.init.zeros_(self.semantic_strength_head.weight)
            nn.init.constant_(
                self.semantic_strength_head.bias,
                math.log(self.semantic_strength_init / (1.0 - self.semantic_strength_init)),
            )

        active_semantic = self.variant != "a0_raw"
        for module in (
            self.relation_projections,
            self.relation_scorer,
            self.semantic_context_norms,
            self.semantic_context_encoder,
            self.semantic_selection_head,
            self.semantic_strength_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(active_semantic)
        self.semantic_coefficients_raw.requires_grad_(active_semantic)

        self.register_buffer(
            "semantic_candidate_edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "semantic_unique_edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "semantic_candidate_degree", torch.empty((0,), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "semantic_fingerprint_state", torch.zeros((32,), dtype=torch.uint8)
        )
        self.register_buffer(
            "semantic_feature_fingerprint_state", torch.zeros((32,), dtype=torch.uint8)
        )
        self.semantic_cache_metadata: dict[str, Any] | None = None
        self.semantic_candidate_fingerprint: str | None = None
        self.semantic_feature_fingerprint: str | None = None
        self._semantic_enabled = self.variant != "a0_raw"
        self._gradient_diagnostics: dict[str, Any] | None = None
        self.profile_forward = False
        self._last_timing: dict[str, float] = {}

    def load_semantic_cache(
        self,
        path: str,
        *,
        expected_dataset: str | None = None,
        expected_feature_fingerprint: str | None = None,
        expected_candidate_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        cache = load_semantic_candidate_cache(
            path,
            expected_dataset=expected_dataset,
            expected_feature_fingerprint=expected_feature_fingerprint,
            expected_candidate_fingerprint=expected_candidate_fingerprint,
        )
        metadata = cache["metadata"]
        if self.num_nodes and self.num_nodes != int(metadata["num_nodes"]):
            raise ValueError("semantic cache node count does not match the NC graph")
        if int(metadata["text_dim"]) != self.text_dim or int(metadata["visual_dim"]) != self.visual_dim:
            raise ValueError("semantic cache modality dimensions do not match model features")
        fingerprint = str(metadata["candidate_fingerprint"])
        fingerprint_bytes = torch.tensor(list(bytes.fromhex(fingerprint)), dtype=torch.uint8)
        feature_fingerprint = str(metadata["feature_fingerprint"])
        feature_fingerprint_bytes = torch.tensor(
            list(bytes.fromhex(feature_fingerprint)), dtype=torch.uint8
        )
        self.semantic_candidate_edge_index = cache["edge_index"]
        self.semantic_unique_edge_index = cache["unique_edge_index"]
        self.semantic_candidate_degree = cache["degree"]
        self.semantic_fingerprint_state.copy_(fingerprint_bytes)
        self.semantic_feature_fingerprint_state.copy_(feature_fingerprint_bytes)
        self.semantic_candidate_fingerprint = fingerprint
        self.semantic_feature_fingerprint = feature_fingerprint
        self.semantic_cache_metadata = metadata
        if self.num_nodes == 0:
            self.num_nodes = int(metadata["num_nodes"])
        return metadata

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        for name, expected_buffer, description in (
            (
                "semantic_fingerprint_state",
                self.semantic_fingerprint_state,
                "candidate graph",
            ),
            (
                "semantic_feature_fingerprint_state",
                self.semantic_feature_fingerprint_state,
                "input features",
            ),
        ):
            saved_fingerprint = state_dict.get(name)
            if saved_fingerprint is not None:
                expected = expected_buffer.detach().cpu()
                saved = saved_fingerprint.detach().cpu()
                if not torch.equal(saved, expected):
                    raise RuntimeError(
                        f"checkpoint semantic {description} fingerprint differs from the attached cache"
                    )
            elif self.semantic_candidate_fingerprint is not None and strict:
                raise RuntimeError(f"checkpoint is missing its semantic {description} fingerprint")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_semantic_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and self.variant == "a0_raw":
            raise ValueError("a0_raw has no semantic path")
        if enabled and self.semantic_candidate_fingerprint is None:
            raise RuntimeError("cannot enable semantic propagation without a validated cache")
        self._semantic_enabled = enabled

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [nodes,{self.input_dim}], got {tuple(x.shape)}")
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    @staticmethod
    def _normalized_operator(
        edge_index: torch.Tensor | None, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if edge_index is None:
            raise ValueError("psce_mag_v7a requires the global physical edge_index")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("physical edge_index must have shape [2,E]")
        edge_index = edge_index.to(dtype=torch.long)
        input_loops = int((edge_index[0] == edge_index[1]).sum().item())
        edge_index, _ = remove_self_loops(edge_index)
        src, dst = edge_index
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=dtype, device=edge_index.device))
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        norm = inv_sqrt[src] * inv_sqrt[dst]
        return src, dst, norm, degree > 0, input_loops

    @staticmethod
    def _propagate(
        state: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(state).index_add(0, dst, state[src] * norm.unsqueeze(-1))

    def _active_rms_norm(self, state: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(state)
        if bool(active.any()):
            selected = state[active]
            rms = torch.sqrt(selected.square().mean(dim=0) + self.eps)
            normalized[active] = selected / rms
        return normalized

    @staticmethod
    def _effective_alpha(raw: torch.Tensor, eps: float) -> torch.Tensor:
        return raw / (raw.norm(p=2, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _top2(logits: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
        dense_probs = torch.softmax(logits, dim=-1)
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        route_weights = torch.zeros_like(dense_probs).scatter(1, top_indices, top_weights)
        selected_mask = torch.zeros_like(dense_probs, dtype=torch.bool).scatter(
            1, top_indices, True
        )
        return {
            "dense_probs": dense_probs,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "route_weights": route_weights,
            "selected_mask": selected_mask,
        }

    def _route_balance(
        self, route: dict[str, torch.Tensor], active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(active.any()):
            importance = route["dense_probs"][active].mean(dim=0)
            selection_share = route["selected_mask"][active].float().mean(dim=0) / self.top_k
            balance = self.num_experts * (importance * selection_share).sum()
        else:
            importance = route["dense_probs"].sum(dim=0) * 0.0
            selection_share = importance
            balance = route["dense_probs"].sum() * 0.0
        return importance, selection_share, balance

    def _physical_route(
        self, modality_index: int, prior: torch.Tensor, active: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        modality = _MODALITIES[modality_index]
        router_state = self.router_norms[modality](
            self.router_projectors[modality](prior)
        )
        if bool(active.any()):
            global_context = router_state[active].mean(dim=0, keepdim=True)
        else:
            global_context = router_state.new_zeros((1, self.router_dim))
        embedding = self.modality_embedding.weight[modality_index].unsqueeze(0)
        zero = torch.zeros_like(global_context)
        static_input = torch.cat([zero, zero, global_context, zero, embedding], dim=-1)
        encoded = self.context_encoder(static_input)
        logits_single = self.selection_head(encoded)
        route_single = self._top2(logits_single, self.top_k)
        strength_single = torch.sigmoid(self.strength_head(encoded).squeeze(-1))
        route = {
            key: value.expand(prior.size(0), *value.shape[1:])
            for key, value in route_single.items()
        }
        route["strength"] = strength_single.expand(prior.size(0))
        route["router_state"] = router_state
        route["static_input"] = static_input
        route["encoded"] = encoded
        importance, selection_share, balance = self._route_balance(route, active)
        route["importance"] = importance
        route["selection_share"] = selection_share
        route["balance"] = balance
        return route

    def _score_semantic_edges(
        self,
        projected: dict[str, torch.Tensor],
        unique_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if unique_edge_index.numel() == 0:
            return projected["text"].new_empty((0,))
        source, target = unique_edge_index
        pieces: list[torch.Tensor] = []
        for start in range(0, source.numel(), self.relation_chunk_size):
            end = min(start + self.relation_chunk_size, source.numel())
            edge_chunk = unique_edge_index[:, start:end]

            def score_chunk(text_projection, visual_projection, edges):
                left, right = edges
                text_left = text_projection.index_select(0, left)
                text_right = text_projection.index_select(0, right)
                visual_left = visual_projection.index_select(0, left)
                visual_right = visual_projection.index_select(0, right)
                relation_features = torch.cat(
                    [
                        text_left * text_right,
                        (text_left - text_right).abs(),
                        visual_left * visual_right,
                        (visual_left - visual_right).abs(),
                    ],
                    dim=-1,
                )
                return self.relation_scorer(relation_features).squeeze(-1)

            if self.training and torch.is_grad_enabled():
                score = checkpoint(
                    score_chunk,
                    projected["text"],
                    projected["visual"],
                    edge_chunk,
                    use_reentrant=False,
                )
            else:
                score = score_chunk(projected["text"], projected["visual"], edge_chunk)
            pieces.append(score)
        return torch.cat(pieces, dim=0)

    def _semantic_propagate(
        self,
        state: torch.Tensor,
        edge_scores: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("semantic model requires an attached, validated candidate cache")
        if self.semantic_candidate_degree.numel() != state.size(0):
            raise RuntimeError(
                "semantic candidate cache uses global node IDs and cannot be applied to a sampled LP subgraph"
            )
        unique = self.semantic_unique_edge_index
        if unique.size(1) != edge_scores.numel():
            raise RuntimeError("semantic score count does not match cached candidate edges")
        source, target = unique
        degree = self.semantic_candidate_degree.to(dtype=state.dtype)
        denominator = (degree[source] * degree[target]).clamp_min(1.0).rsqrt()
        semantic_weight = torch.sigmoid(edge_scores)
        normalized_unique_weight = semantic_weight * denominator
        directed_weight = torch.cat(
            [normalized_unique_weight, normalized_unique_weight], dim=0
        )
        result = chunked_weighted_spmm(
            state,
            self.semantic_candidate_edge_index,
            directed_weight,
            self.propagation_chunk_size,
        )
        if return_aux:
            return result, semantic_weight, normalized_unique_weight
        return result

    def _semantic_active_masks(
        self,
        physical_source: torch.Tensor,
        physical_target: torch.Tensor,
        semantic_state1: torch.Tensor,
        physical_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_active1 = self.semantic_candidate_degree > 0
        has_semantic_source = torch.zeros(
            physical_active.shape, dtype=torch.long, device=physical_active.device
        )
        if physical_source.numel():
            has_semantic_source.index_add_(
                0,
                physical_target,
                semantic_active1[physical_source].to(dtype=torch.long),
            )
        semantic_active2 = physical_active & (has_semantic_source > 0)
        # A topological mask preserves the first semantic hop for physical
        # isolates while preventing empty second-hop rows from being normalized.
        del semantic_state1
        return semantic_active1, semantic_active2

    def _calibrate_semantic(
        self,
        semantic_state: torch.Tensor,
        raw_reference: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(active.any()):
            selected_semantic = semantic_state[active]
            selected_raw = raw_reference[active]
            semantic_rms = torch.sqrt(selected_semantic.square().mean() + self.eps)
            raw_rms = torch.sqrt(selected_raw.square().mean() + self.eps)
            scale = torch.minimum(
                torch.ones_like(raw_rms),
                raw_rms.detach() / (semantic_rms.detach() + self.eps),
            )
            output = torch.zeros_like(semantic_state)
            output[active] = selected_semantic * scale
        else:
            semantic_rms = semantic_state.sum() * 0.0
            raw_rms = raw_reference.sum() * 0.0
            scale = semantic_rms.new_ones(())
            output = torch.zeros_like(semantic_state)
        return output, scale, semantic_rms, raw_rms

    def _semantic_route(
        self,
        modality_index: int,
        prior: torch.Tensor,
        physical_first: torch.Tensor,
        semantic_first: torch.Tensor,
        raw_first: torch.Tensor,
        semantic_active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        modality = _MODALITIES[modality_index]
        amplitude = torch.sqrt(semantic_first.square().mean(dim=-1, keepdim=True) + self.eps)
        raw_amplitude = torch.sqrt(raw_first.square().mean(dim=-1, keepdim=True) + self.eps)
        amplitude_ratio = amplitude / raw_amplitude
        context_blocks = [
            self.semantic_context_norms["intrinsic"](prior),
            self.semantic_context_norms["physical"](physical_first),
            self.semantic_context_norms["semantic"](semantic_first),
        ]
        embedding = self.modality_embedding.weight[modality_index].view(1, -1)
        if self.variant == "a1_static_dual":
            if bool(semantic_active.any()):
                pooled = [block[semantic_active].mean(dim=0, keepdim=True) for block in context_blocks]
                pooled_amplitude = amplitude_ratio[semantic_active].mean(dim=0, keepdim=True)
            else:
                pooled = [prior.new_zeros((1, self.hidden_dim)) for _ in context_blocks]
                pooled_amplitude = prior.new_zeros((1, 1))
            static_context = torch.cat([*pooled, embedding, pooled_amplitude], dim=-1)
            encoded = self.semantic_context_encoder(static_context)
            logits = self.semantic_selection_head(encoded)
            strength = torch.sigmoid(self.semantic_strength_head(encoded).squeeze(-1))
        else:
            embedding_nodes = embedding.expand(prior.size(0), -1)
            node_context = torch.cat(
                [*context_blocks, embedding_nodes, amplitude_ratio], dim=-1
            )
            encoded = self.semantic_context_encoder(node_context)
            logits = self.semantic_selection_head(encoded)
            strength = torch.sigmoid(self.semantic_strength_head(encoded).squeeze(-1))
        route = self._top2(logits, self.top_k)
        route = {
            key: value.expand(prior.size(0), *value.shape[1:])
            for key, value in route.items()
        }
        route["strength"] = strength.expand(prior.size(0))
        route["encoded"] = encoded
        importance, selection_share, balance = self._route_balance(route, semantic_active)
        route["importance"] = importance
        route["selection_share"] = selection_share
        route["balance"] = balance
        route["amplitude_ratio"] = amplitude_ratio
        return route

    def _expert_profiles(
        self,
        coefficients: torch.Tensor,
        states: torch.Tensor,
        experts: nn.ModuleList,
        expert_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profiles = torch.einsum("mk,knd->mnd", coefficients, states)
        transformed_rows = []
        for expert_id, expert in enumerate(experts):
            profile = profiles[expert_id]
            if self.training and torch.is_grad_enabled():
                transformed_rows.append(
                    checkpoint(expert, profile, use_reentrant=False)
                )
            else:
                transformed_rows.append(expert(profile))
        transformed = torch.stack(transformed_rows, dim=0)
        return profiles, profiles + float(expert_scale) * transformed

    @staticmethod
    def _rms(value: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        if active is not None:
            value = value[active]
        if value.numel() == 0:
            return value.new_zeros(())
        return torch.sqrt(value.float().square().mean())

    @staticmethod
    def _cosine_mean(left: torch.Tensor, right: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if not bool(active.any()):
            return left.sum() * 0.0
        l = left[active].reshape(-1).float()
        r = right[active].reshape(-1).float()
        denom = l.norm() * r.norm()
        return torch.where(denom > 0, torch.dot(l, r) / denom.clamp_min(1.0e-12), denom * 0.0)

    @staticmethod
    def _quantile_summary(values: torch.Tensor) -> dict[str, float]:
        detached = values.detach().float()
        if detached.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
        quantiles = torch.quantile(
            detached,
            torch.tensor([0.10, 0.50, 0.90], device=detached.device),
        )
        return {
            "mean": float(detached.mean().cpu()),
            "std": float(detached.std(unbiased=False).cpu()),
            "p10": float(quantiles[0].cpu()),
            "p50": float(quantiles[1].cpu()),
            "p90": float(quantiles[2].cpu()),
        }

    def _details_summary(
        self,
        *,
        semantic_weight: torch.Tensor | None,
        semantic_states: dict[str, dict[str, torch.Tensor]],
        routes: dict[str, dict[str, dict[str, torch.Tensor]]],
        physical_corrections: dict[str, torch.Tensor],
        semantic_corrections: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"gradient_diagnostics": self._gradient_diagnostics}
        if semantic_weight is not None:
            result["edge_weight"] = self._quantile_summary(semantic_weight)
        modalities: dict[str, Any] = {}
        for modality in _MODALITIES:
            physical_route = routes["physical"][modality]
            physical_active = physical_route["active"]
            record: dict[str, Any] = {
                "physical_expert_selection_share": physical_route["selection_share"].detach().cpu().tolist(),
                "physical_strength_mean": float(physical_route["strength"].detach().mean().cpu()),
                "physical_contribution_rms": float(
                    self._rms(physical_corrections[modality], physical_active).cpu()
                ),
            }
            if modality in semantic_states:
                states = semantic_states[modality]
                semantic_route = routes["semantic"][modality]
                semantic_active = states["active1"] | states["active2"]
                sem_rms = [self._rms(states[f"state{k}"], states[f"active{k}"]) for k in (1, 2)]
                raw_rms = [self._rms(states[f"raw{k}"], states[f"active{k}"]) for k in (1, 2)]
                record.update(
                    {
                        "semantic_expert_selection_share": semantic_route["selection_share"].detach().cpu().tolist(),
                        "semantic_strength": self._quantile_summary(semantic_route["strength"]),
                        "semantic_hop_rms": [float(value.cpu()) for value in sem_rms],
                        "raw_reference_hop_rms": [float(value.cpu()) for value in raw_rms],
                        "semantic_raw_cosine": [
                            float(self._cosine_mean(states[f"state{k}"], states[f"raw{k}"], states[f"active{k}"]).cpu())
                            for k in (1, 2)
                        ],
                        "semantic_contribution_rms": float(
                            self._rms(semantic_corrections[modality], semantic_active).cpu()
                        ),
                        "semantic_to_physical_contribution_rms": float(
                            self._rms(semantic_corrections[modality], semantic_active).cpu()
                            / self._rms(physical_corrections[modality], physical_active).clamp_min(1.0e-12).cpu()
                        ),
                        "semantic_active_nodes_hop1": int(states["active1"].sum().item()),
                        "semantic_active_nodes_hop2": int(states["active2"].sum().item()),
                    }
                )
            modalities[modality] = record
        result["modalities"] = modalities
        return result

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        if x.size(0) != self.num_nodes and self.semantic_candidate_fingerprint is not None:
            raise RuntimeError(
                "PSCE-MAG V7A is full-graph NC only: the global-ID semantic cache cannot be used with a sampled LP subgraph"
            )
        inputs = self._split_modalities(x)
        priors = {modality: self.projectors[modality](inputs[modality]) for modality in _MODALITIES}
        source, target, physical_norm, physical_active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )

        physical_bases: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            states: list[torch.Tensor] = []
            state = priors[modality]
            for _ in range(self.physical_hops):
                state = self._propagate(state, source, target, physical_norm)
                state = state * physical_active.to(state.dtype).unsqueeze(-1)
                states.append(state)
            physical_bases[modality] = torch.stack(
                [self._active_rms_norm(item, physical_active) for item in states], dim=0
            )
            del states, state

        alpha_physical = self._effective_alpha(self.alpha_raw, self.eps)
        outputs: dict[str, torch.Tensor] = {}
        physical_routes: dict[str, dict[str, torch.Tensor]] = {}
        physical_corrections: dict[str, torch.Tensor] = {}
        physical_balances: dict[str, torch.Tensor] = {}
        semantic_routes: dict[str, dict[str, torch.Tensor]] = {}
        semantic_corrections: dict[str, torch.Tensor] = {}
        semantic_states: dict[str, dict[str, torch.Tensor]] = {}
        semantic_weights: torch.Tensor | None = None

        semantic_active_run = self.variant != "a0_raw" and self._semantic_enabled
        edge_scores: torch.Tensor | None = None
        semantic_weight: torch.Tensor | None = None
        directed_weights: torch.Tensor | None = None
        projected_relations: dict[str, torch.Tensor] = {}
        if self.profile_forward and x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        relation_started = time.perf_counter()
        if semantic_active_run:
            if self.semantic_candidate_fingerprint is None:
                raise RuntimeError("V7A A1/A2 requires a validated semantic candidate cache")
            if self.semantic_candidate_degree.numel() != x.size(0):
                raise RuntimeError(
                    "global semantic candidate cache cannot be applied to a sampled LP subgraph"
                )
            projected_relations = {
                modality: self.relation_projections[modality](priors[modality])
                for modality in _MODALITIES
            }
            edge_scores = self._score_semantic_edges(
                projected_relations, self.semantic_unique_edge_index
            )
            if self.profile_forward and x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            semantic_weight = torch.sigmoid(edge_scores)
            unique = self.semantic_unique_edge_index
            degree = self.semantic_candidate_degree.to(dtype=priors["text"].dtype)
            normalized_unique = semantic_weight * (
                degree[unique[0]] * degree[unique[1]]
            ).clamp_min(1.0).rsqrt()
            directed_weights = torch.cat([normalized_unique, normalized_unique], dim=0)
            if self.profile_forward:
                self._last_timing["relation_scoring_seconds"] = (
                    time.perf_counter() - relation_started
                )
        elif self.profile_forward:
            self._last_timing["relation_scoring_seconds"] = 0.0
        if self.profile_forward:
            self._last_timing["semantic_propagation_seconds"] = 0.0

        for modality_index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            _, physical_expert_values = self._expert_profiles(
                alpha_physical,
                physical_bases[modality],
                self.experts,
                self.expert_feature_scale,
            )
            route = self._physical_route(modality_index, prior, physical_active)
            route["active"] = physical_active
            physical_routes[modality] = route
            physical_mixture = torch.einsum(
                "nm,mnd->nd", route["route_weights"], physical_expert_values
            )
            physical_correction = route["strength"].unsqueeze(-1) * physical_mixture
            physical_corrections[modality] = physical_correction
            physical_balances[modality] = route["balance"]
            outputs[modality] = prior + physical_correction

            if semantic_active_run:
                assert edge_scores is not None
                assert semantic_weight is not None and directed_weights is not None
                if modality_index == 0:
                    # Scores and degree normalization are shared by Text/Visual.
                    semantic_weights = semantic_weight
                semantic_started = time.perf_counter()
                if self.profile_forward and prior.device.type == "cuda":
                    torch.cuda.synchronize(prior.device)
                sem_state1_raw = chunked_weighted_spmm(
                    prior,
                    self.semantic_candidate_edge_index,
                    directed_weights,
                    self.propagation_chunk_size,
                )
                sem_active1, sem_active2 = self._semantic_active_masks(
                    source, target, sem_state1_raw, physical_active
                )
                sem_state1_raw = sem_state1_raw * sem_active1.to(prior.dtype).unsqueeze(-1)
                sem_state1, sem_scale1, sem_rms1, raw_rms1 = self._calibrate_semantic(
                    sem_state1_raw, physical_bases[modality][0], sem_active1
                )
                sem_state2_raw = self._propagate(sem_state1, source, target, physical_norm)
                sem_state2_raw = sem_state2_raw * sem_active2.to(prior.dtype).unsqueeze(-1)
                sem_state2, sem_scale2, sem_rms2, raw_rms2 = self._calibrate_semantic(
                    sem_state2_raw, physical_bases[modality][1], sem_active2
                )
                if self.profile_forward and prior.device.type == "cuda":
                    torch.cuda.synchronize(prior.device)
                if self.profile_forward:
                    self._last_timing["semantic_propagation_seconds"] += (
                        time.perf_counter() - semantic_started
                    )
                sem_states_for_profiles = torch.stack([sem_state1, sem_state2], dim=0)
                alpha_semantic = self._effective_alpha(
                    self.semantic_coefficients_raw, self.eps
                )
                _, sem_expert_values = self._expert_profiles(
                    alpha_semantic,
                    sem_states_for_profiles,
                    self.experts,
                    self.expert_feature_scale,
                )
                sem_route = self._semantic_route(
                    modality_index,
                    prior,
                    physical_bases[modality][0],
                    sem_state1,
                    physical_bases[modality][0],
                    sem_active1,
                )
                semantic_routes[modality] = sem_route
                sem_mixture = torch.einsum(
                    "nm,mnd->nd", sem_route["route_weights"], sem_expert_values
                )
                sem_correction = sem_route["strength"].unsqueeze(-1) * sem_mixture
                semantic_corrections[modality] = sem_correction
                outputs[modality] = outputs[modality] + sem_correction
                semantic_states[modality] = {
                    "state1": sem_state1,
                    "state2": sem_state2,
                    "raw1": physical_bases[modality][0],
                    "raw2": physical_bases[modality][1],
                    "active1": sem_active1,
                    "active2": sem_active2,
                    "scale1": sem_scale1,
                    "scale2": sem_scale2,
                    "rms1": sem_rms1,
                    "rms2": sem_rms2,
                    "raw_rms1": raw_rms1,
                    "raw_rms2": raw_rms2,
                }

        physical_balance = 0.5 * (physical_balances["text"] + physical_balances["visual"])
        if semantic_active_run:
            semantic_balance = 0.5 * (
                semantic_routes["text"]["balance"] + semantic_routes["visual"]["balance"]
            )
            balance_loss = 0.5 * (physical_balance + semantic_balance)
        else:
            semantic_balance = physical_balance.new_zeros(())
            balance_loss = physical_balance
        aux_loss = self.balance_weight * balance_loss
        fused = torch.cat([outputs["text"], outputs["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )
        info: dict[str, Any] = {
            "variant": self.variant,
            "active_nodes": physical_active.detach(),
            "input_self_loops_removed": input_self_loops,
            "balance_loss": balance_loss.detach(),
            "physical_balance_loss": physical_balance.detach(),
            "semantic_balance_loss": semantic_balance.detach(),
            "semantic_enabled": semantic_active_run,
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "modalities": {
                modality: {
                    "physical_strength_mean": physical_routes[modality]["strength"].detach().mean(),
                    **(
                        {"semantic_strength_mean": semantic_routes[modality]["strength"].detach().mean()}
                        if semantic_active_run
                        else {}
                    ),
                }
                for modality in _MODALITIES
            },
        }
        if return_details:
            info["details"] = self._details_summary(
                semantic_weight=semantic_weights,
                semantic_states=semantic_states,
                routes={"physical": physical_routes, "semantic": semantic_routes},
                physical_corrections=physical_corrections,
                semantic_corrections=semantic_corrections,
            )
        return z, None, None, aux_loss, info

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        selected_names = [
            "alpha_raw",
            "experts.0.linear1.weight",
            "selection_head.weight",
            "strength_head.bias",
            "semantic_coefficients_raw",
            "relation_projections.text.weight",
            "relation_projections.visual.weight",
            "relation_scorer.0.weight",
            "relation_scorer.2.weight",
            "semantic_selection_head.weight",
            "semantic_strength_head.bias",
        ]
        parameters = dict(self.named_parameters())
        diagnostics: dict[str, Any] = {}
        for name in selected_names:
            parameter = parameters[name]
            gradient = parameter.grad
            norm = float(gradient.detach().float().norm().cpu()) if gradient is not None else 0.0
            diagnostics[name] = {
                "gradient_present": gradient is not None,
                "gradient_finite": bool(gradient is None or torch.isfinite(gradient).all()),
                "gradient_norm": norm,
                "gradient_nonzero": bool(norm > 0.0),
            }
        self._gradient_diagnostics = diagnostics
        return diagnostics
