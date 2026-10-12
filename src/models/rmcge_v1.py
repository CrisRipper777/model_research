"""RM-CGE V1: relation-modality collaborative experts for full-graph NC."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.data.semantic_candidates import _array_fingerprint
from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b


_MODALITIES = ("text", "visual")
_RELATIONS = ("physical", "semantic")
_SUPPORTS = ("physical", "text", "visual")


def _empty_sparse(
    num_nodes: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0,), dtype=dtype, device=device),
        (num_nodes, num_nodes),
        device=device,
    ).coalesce()


def canonical_undirected_edges(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Return sorted unique non-loop pairs with the canonical orientation u < v."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edges = edge_index.to(dtype=torch.long)
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edges.device)
    if int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes):
        raise ValueError("edge_index contains node IDs outside [0,num_nodes)")
    left = torch.minimum(edges[0], edges[1])
    right = torch.maximum(edges[0], edges[1])
    keep = left != right
    codes = torch.unique(left[keep] * int(num_nodes) + right[keep], sorted=True)
    return torch.stack((codes // int(num_nodes), codes % int(num_nodes)), dim=0)


def _row_normalized_operator(
    unique_edges: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make a symmetric unit-edge COO adjacency and normalize each row by degree."""
    if unique_edges.ndim != 2 or unique_edges.size(0) != 2:
        raise ValueError("unique_edges must have shape [2,E]")
    device = unique_edges.device
    if unique_edges.size(1) == 0:
        operator = _empty_sparse(num_nodes, device=device, dtype=dtype)
        return operator, torch.zeros(num_nodes, dtype=torch.long, device=device)
    directed = torch.cat((unique_edges, unique_edges.flip(0)), dim=1)
    degree = torch.bincount(directed[0], minlength=num_nodes)
    values = degree[directed[0]].to(dtype=dtype).reciprocal()
    operator = torch.sparse_coo_tensor(
        directed, values, (num_nodes, num_nodes), device=device
    ).coalesce()
    return operator, degree


def build_physical_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the loop-free, deduplicated, undirected physical row-mean operator."""
    unique = canonical_undirected_edges(edge_index, num_nodes)
    return _row_normalized_operator(unique, num_nodes, dtype=dtype)


def build_semantic_operators(
    unique_edge_index: torch.Tensor,
    source_flags: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Build independent row-normalized Text and Visual supports from V7B flags."""
    unique = unique_edge_index.to(dtype=torch.long)
    if unique.ndim != 2 or unique.size(0) != 2:
        raise ValueError("unique_edge_index must have shape [2,E]")
    source_flags = source_flags.to(device=unique.device)
    if source_flags.dtype != torch.uint8 or tuple(source_flags.shape) != (
        unique.size(1),
        2,
    ):
        raise ValueError("source_flags must be uint8 with shape [unique_edges,2]")
    if source_flags.numel() and (
        bool((source_flags > 2).any()) or bool((source_flags.sum(dim=1) == 0).any())
    ):
        raise ValueError("V7B source counts must be in {0,1,2} and each edge needs a source")
    if unique.numel():
        if int(unique.min()) < 0 or int(unique.max()) >= int(num_nodes):
            raise ValueError("semantic edge node ID is out of range")
        if bool((unique[0] >= unique[1]).any()):
            raise ValueError("semantic unique edges must be canonical u<v non-loop pairs")
        codes = unique[0] * int(num_nodes) + unique[1]
        if codes.numel() > 1 and not bool((codes[1:] > codes[:-1]).all()):
            raise ValueError("semantic edges must be sorted and duplicate-free")

    result: dict[str, torch.Tensor] = {}
    for column, name in ((0, "text"), (1, "visual")):
        selected = unique[:, source_flags[:, column] > 0]
        operator, degree = _row_normalized_operator(selected, num_nodes, dtype=dtype)
        result[f"p_{name}"] = operator
        result[f"degree_{name}"] = degree
    return result


def joint_masked_softmax(
    logits: torch.Tensor, relation_valid: torch.Tensor
) -> torch.Tensor:
    """Normalize jointly across [relation, expert] while safely handling empty rows."""
    if logits.ndim != 3 or logits.size(1) != 2:
        raise ValueError("logits must have shape [N,2,num_experts]")
    if relation_valid.shape != logits.shape[:2]:
        raise ValueError("relation_valid must have shape [N,2]")
    pair_valid = relation_valid.unsqueeze(-1).expand_as(logits)
    any_valid = relation_valid.any(dim=1, keepdim=True).unsqueeze(-1)
    masked = logits.masked_fill(~pair_valid, -torch.inf)
    # An all-invalid row uses finite placeholders before softmax and is masked to zero after.
    masked = torch.where(any_valid, masked, torch.zeros_like(masked))
    probabilities = torch.softmax(masked.flatten(1), dim=-1).view_as(logits)
    return probabilities * pair_valid.to(probabilities.dtype)


class ModalityProjector(nn.Module):
    """Independent two-layer residual projector for one input modality."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.shortcut = nn.Linear(input_dim, hidden_dim)
        self.first = nn.Linear(input_dim, hidden_dim)
        self.second = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(value)
        update = self.dropout(F.gelu(self.first(value)))
        update = self.dropout(self.second(update))
        return self.norm(residual + update)


class CrossModalBottleneck(nn.Module):
    """Direction-specific nonlinear conversion applied after relation propagation."""

    def __init__(self, hidden_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(hidden_dim, bottleneck)
        self.second = nn.Linear(bottleneck, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.dropout(F.gelu(self.first(value))))


class GraphExpert(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(4 * hidden_dim, bottleneck)
        self.second = nn.Linear(bottleneck, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.dropout(F.gelu(self.first(value))))


class RelationExpertLayer(nn.Module):
    """One shared expert bank and one joint router for both target modalities."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_experts: int,
        expert_bottleneck: int,
        router_hidden_dim: int,
        relation_embedding_dim: int,
        modality_embedding_dim: int,
        cross_modal_bottleneck: int,
        dropout: float,
        router_correction_init_std: float,
        physical_prior: float,
        semantic_prior: float,
        expert_chunk_size: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_chunk_size = expert_chunk_size
        self.state_norms = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in _MODALITIES}
        )
        # One parameter-free block scale calibrator is reused on all four blocks.
        self.block_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_modal = nn.ModuleDict(
            {
                "visual_to_text": CrossModalBottleneck(
                    hidden_dim, cross_modal_bottleneck, dropout
                ),
                "text_to_visual": CrossModalBottleneck(
                    hidden_dim, cross_modal_bottleneck, dropout
                ),
            }
        )
        self.experts = nn.ModuleList(
            GraphExpert(hidden_dim, expert_bottleneck, dropout)
            for _ in range(num_experts)
        )
        router_input_dim = (
            3 * hidden_dim + relation_embedding_dim + modality_embedding_dim + 1
        )
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_experts),
        )
        self.relation_embedding = nn.Embedding(2, relation_embedding_dim)
        self.modality_embedding = nn.Embedding(2, modality_embedding_dim)
        nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)
        final = self.router[-1]
        nn.init.normal_(
            final.weight, mean=0.0, std=float(router_correction_init_std)
        )
        nn.init.zeros_(final.bias)
        relation_prior = torch.tensor(
            [physical_prior, semantic_prior], dtype=torch.float32
        )
        if bool((relation_prior <= 0).any()) or not torch.isclose(
            relation_prior.sum(), torch.tensor(1.0)
        ):
            raise ValueError("relation prior probabilities must be positive and sum to one")
        self.router_log_prior = nn.Parameter(
            relation_prior.log().unsqueeze(-1).expand(2, num_experts).clone()
            - math.log(num_experts)
        )
        self.update_dropout = nn.Dropout(dropout)

    def router_logits(
        self,
        node_state: torch.Tensor,
        same_context: torch.Tensor,
        cross_context: torch.Tensor,
        neighbor_count: torch.Tensor,
        *,
        relation_id: int,
        modality_id: int,
    ) -> torch.Tensor:
        n = node_state.size(0)
        relation = self.relation_embedding.weight[relation_id].expand(n, -1)
        modality = self.modality_embedding.weight[modality_id].expand(n, -1)
        features = torch.cat(
            (
                node_state,
                same_context,
                cross_context,
                relation,
                modality,
                torch.log1p(neighbor_count.to(node_state.dtype)).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.router(features) + self.router_log_prior[relation_id]

    def mix_experts(
        self,
        block: torch.Tensor,
        probabilities: torch.Tensor,
        valid: torch.Tensor,
        *,
        collect_details: bool,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        n = block.size(0)
        delta_chunks: list[torch.Tensor] = []
        square_sums = block.new_zeros(self.num_experts, dtype=torch.float64)
        cosine_sums: dict[tuple[int, int], torch.Tensor] = {}
        valid_count = 0
        chunk_size = self.expert_chunk_size
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            block_chunk = block[start:stop]
            probability_chunk = probabilities[start:stop]
            valid_chunk = valid[start:stop]
            outputs = [
                self.experts[index](block_chunk) for index in range(self.num_experts)
            ]
            weighted = torch.zeros_like(outputs[0])
            for index, output in enumerate(outputs):
                weighted = weighted + output * probability_chunk[:, index : index + 1]
                if collect_details and bool(valid_chunk.any()):
                    selected = output[valid_chunk].float()
                    square_sums[index] += selected.double().square().sum()
            if collect_details and bool(valid_chunk.any()):
                selected_outputs = [output[valid_chunk].float() for output in outputs]
                valid_count += int(valid_chunk.sum().item())
                for left in range(self.num_experts):
                    left_norm = F.normalize(selected_outputs[left], dim=-1, eps=1e-12)
                    for right in range(left + 1, self.num_experts):
                        right_norm = F.normalize(
                            selected_outputs[right], dim=-1, eps=1e-12
                        )
                        key = (left, right)
                        value = (left_norm * right_norm).sum(dim=-1).sum().double()
                        cosine_sums[key] = cosine_sums.get(key, value.new_zeros(())) + value
            delta_chunks.append(weighted)

        delta = torch.cat(delta_chunks, dim=0) if delta_chunks else block.new_zeros((0, self.hidden_dim))
        if not collect_details:
            return delta, None
        denom = max(valid_count * self.hidden_dim, 1)
        expert_rms = [
            float((square_sums[index] / denom).sqrt().cpu())
            for index in range(self.num_experts)
        ]
        expert_cosine = [
            {
                "experts": [left, right],
                "mean": float((value / max(valid_count, 1)).cpu()),
            }
            for (left, right), value in sorted(cosine_sums.items())
        ]
        return delta, {
            "expert_output_rms": expert_rms,
            "expert_pairwise_cosine": expert_cosine,
        }


class Model(nn.Module):
    """Validated global-ID full-graph NC model; no sampled-LP support."""

    requires_global_semantic_candidates = True
    requires_full_graph_training = True
    supports_link_prediction = False
    full_graph_training = True
    record_run_telemetry = True
    record_epoch_history = True
    out_dim = 128

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.num_nodes = int(data_info["num_nodes"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.num_layers = int(model_cfg.get("num_layers", 2))
        self.dropout = float(model_cfg.get("dropout", 0.2))
        self.num_experts = int(model_cfg.get("num_experts", 3))
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.expert_chunk_size = int(model_cfg.get("expert_chunk_size", 32768))
        self.activation_checkpointing = bool(
            model_cfg.get("activation_checkpointing", True)
        )
        self.router_hidden_dim = int(model_cfg.get("router_hidden_dim", 64))
        self.relation_embedding_dim = int(
            model_cfg.get("relation_embedding_dim", 8)
        )
        self.modality_embedding_dim = int(
            model_cfg.get("modality_embedding_dim", 8)
        )
        self.cross_modal_bottleneck = int(
            model_cfg.get("cross_modal_bottleneck", 64)
        )
        if self.hidden_dim != 128 or self.num_layers != 2 or self.num_experts != 3:
            raise ValueError("RM-CGE V1 freezes hidden_dim=128, num_layers=2, num_experts=3")
        if self.expert_bottleneck != 64 or not math.isclose(self.dropout, 0.2):
            raise ValueError("RM-CGE V1 freezes expert_bottleneck=64 and dropout=0.2")
        if self.text_dim < 1 or self.visual_dim < 1:
            raise ValueError("RM-CGE requires positive Text and Visual feature dimensions")
        if self.expert_chunk_size < 1:
            raise ValueError("expert_chunk_size must be positive")
        if not math.isclose(float(model_cfg.get("residual_scale", 0.5)), 0.5):
            raise ValueError("RM-CGE V1 freezes residual_scale=0.5")

        self.projectors = nn.ModuleDict(
            {
                "text": ModalityProjector(self.text_dim, self.hidden_dim, self.dropout),
                "visual": ModalityProjector(
                    self.visual_dim, self.hidden_dim, self.dropout
                ),
            }
        )
        common_layer_args = {
            "num_experts": self.num_experts,
            "expert_bottleneck": self.expert_bottleneck,
            "router_hidden_dim": self.router_hidden_dim,
            "relation_embedding_dim": self.relation_embedding_dim,
            "modality_embedding_dim": self.modality_embedding_dim,
            "cross_modal_bottleneck": self.cross_modal_bottleneck,
            "dropout": self.dropout,
            "router_correction_init_std": float(
                model_cfg.get("router_correction_init_std", 0.01)
            ),
            "physical_prior": float(model_cfg.get("physical_prior", 0.6)),
            "semantic_prior": float(model_cfg.get("semantic_prior", 0.4)),
            "expert_chunk_size": self.expert_chunk_size,
        }
        self.encoder_layers = nn.ModuleList(
            RelationExpertLayer(self.hidden_dim, **common_layer_args)
            for _ in range(self.num_layers)
        )
        self.fusion = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        self.semantic_cache_path: str | None = None
        self.semantic_dataset: str | None = None
        self.semantic_candidate_fingerprint: str | None = None
        self.semantic_feature_fingerprint: str | None = None
        self.semantic_v7b_fingerprint: str | None = None
        self.semantic_metadata: dict[str, Any] | None = None
        self._validated_feature_hashes: tuple[str, str] | None = None
        self._physical_edge_reference: torch.Tensor | None = None
        self._physical_edge_key: tuple[Any, ...] | None = None
        self._gradient_diagnostics: dict[str, Any] | None = None
        self.register_buffer("semantic_pt", None, persistent=False)
        self.register_buffer("semantic_pv", None, persistent=False)
        self.register_buffer("semantic_degree_text", None, persistent=False)
        self.register_buffer("semantic_degree_visual", None, persistent=False)
        self.register_buffer("physical_operator", None, persistent=False)
        self.register_buffer("physical_degree", None, persistent=False)

    @property
    def semantic_cache_loaded(self) -> bool:
        return all(
            value is not None
            for value in (
                self.semantic_pt,
                self.semantic_pv,
                self.semantic_degree_text,
                self.semantic_degree_visual,
            )
        )

    def load_semantic_cache(
        self,
        path: str | Path,
        *,
        expected_dataset: str | None = None,
        expected_feature_fingerprint: str | None = None,
        expected_candidate_fingerprint: str | None = None,
        expected_v7b_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        cache = load_semantic_candidate_cache_v7b(
            path,
            expected_dataset=expected_dataset,
            expected_feature_fingerprint=expected_feature_fingerprint,
            expected_candidate_fingerprint=expected_candidate_fingerprint,
            expected_v7b_fingerprint=expected_v7b_fingerprint,
        )
        metadata = cache["metadata"]
        if int(metadata.get("num_nodes", -1)) != self.num_nodes:
            raise ValueError("semantic cache node count does not match the NC graph")
        if int(metadata.get("top_k", -1)) != 8:
            raise ValueError("RM-CGE V1 requires the frozen V7B K=8 cache")
        views = build_semantic_operators(
            cache["unique_edge_index"],
            cache["source_flags"],
            self.num_nodes,
        )
        self.semantic_pt = views["p_text"]
        self.semantic_pv = views["p_visual"]
        self.semantic_degree_text = views["degree_text"]
        self.semantic_degree_visual = views["degree_visual"]
        self.semantic_cache_path = str(Path(path).resolve())
        self.semantic_dataset = str(metadata.get("dataset", ""))
        self.semantic_candidate_fingerprint = str(
            metadata.get("candidate_fingerprint", "")
        )
        self.semantic_feature_fingerprint = str(
            metadata.get("feature_fingerprint", "")
        )
        self.semantic_v7b_fingerprint = str(metadata.get("v7b_fingerprint", ""))
        self.semantic_metadata = metadata
        return metadata

    def _validate_input_features_once(self, x: torch.Tensor) -> None:
        if self.semantic_metadata is None or self._validated_feature_hashes is not None:
            return
        identity = self.semantic_metadata.get("feature_identity", {})
        expected_text = identity.get("text", {}).get("array_sha256")
        expected_visual = identity.get("visual", {}).get("array_sha256")
        if expected_text is None or expected_visual is None:
            raise ValueError("V7B cache is missing modality feature array fingerprints")
        text = np.ascontiguousarray(
            x[:, : self.text_dim].detach().to(device="cpu", dtype=torch.float32).numpy()
        )
        visual = np.ascontiguousarray(
            x[
                :, self.text_dim : self.text_dim + self.visual_dim
            ].detach().to(device="cpu", dtype=torch.float32).numpy()
        )
        observed = (_array_fingerprint(text), _array_fingerprint(visual))
        if observed != (expected_text, expected_visual):
            raise ValueError("model input modalities do not match the V7B feature fingerprint")
        self._validated_feature_hashes = observed

    def _get_physical_operator(
        self, edge_index: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("RM-CGE full-graph NC requires physical edge_index")
        key = (
            edge_index.device,
            edge_index.data_ptr(),
            int(getattr(edge_index, "_version", 0)),
            dtype,
        )
        if self.physical_operator is None or self._physical_edge_key != key:
            same_graph = (
                self._physical_edge_reference is not None
                and self._physical_edge_reference.device == edge_index.device
                and self.physical_operator.dtype == dtype
                and torch.equal(self._physical_edge_reference, edge_index)
            )
            if same_graph:
                self._physical_edge_key = key
                return self.physical_operator, self.physical_degree
            operator, degree = build_physical_operator(
                edge_index, self.num_nodes, dtype=dtype
            )
            self.physical_operator = operator
            self.physical_degree = degree
            self._physical_edge_reference = edge_index.detach().clone()
            self._physical_edge_key = key
        return self.physical_operator, self.physical_degree

    @staticmethod
    def _apply_mask(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return value * valid.to(value.dtype).unsqueeze(-1)

    def _execute_layer(
        self,
        layer_index: int,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        operators: Mapping[str, torch.Tensor],
        degrees: Mapping[str, torch.Tensor],
        *,
        intervention: Mapping[str, Any],
        collect_details: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        layer = self.encoder_layers[layer_index]
        current = {"text": h_text, "visual": h_visual}
        normalized = {
            name: layer.state_norms[name](current[name]) for name in _MODALITIES
        }
        propagated = {
            support: {
                modality: torch.sparse.mm(operators[support], normalized[modality])
                for modality in _MODALITIES
            }
            for support in _SUPPORTS
        }
        disabled_relations = set(intervention.get("disable_relations", ()))
        cross_enabled = not bool(intervention.get("disable_cross_modal", False))
        target_supports = {
            "text": ("physical", "text"),
            "visual": ("physical", "visual"),
        }
        contexts: dict[str, list[dict[str, Any]]] = {}
        router_logits: dict[str, list[torch.Tensor]] = {}
        valid_masks: dict[str, list[torch.Tensor]] = {}
        blocks: dict[str, list[torch.Tensor]] = {}
        for modality_id, modality in enumerate(_MODALITIES):
            direction = (
                "visual_to_text" if modality == "text" else "text_to_visual"
            )
            contexts[modality] = []
            router_logits[modality] = []
            valid_masks[modality] = []
            blocks[modality] = []
            for relation_id, (relation, support) in enumerate(
                zip(_RELATIONS, target_supports[modality])
            ):
                same = propagated[support][modality]
                degree = degrees[support]
                valid = degree > 0
                if relation in disabled_relations:
                    valid = torch.zeros_like(valid)
                if cross_enabled:
                    cross = layer.cross_modal[direction](
                        propagated[support][
                            "visual" if modality == "text" else "text"
                        ]
                    )
                    cross = self._apply_mask(cross, valid)
                else:
                    cross = torch.zeros_like(same)
                state_block = layer.block_norm(current[modality])
                same_block = layer.block_norm(same)
                diff_block = layer.block_norm(current[modality] - same)
                cross_block = layer.block_norm(cross)
                block = torch.cat(
                    (state_block, same_block, diff_block, cross_block), dim=-1
                )
                logits = layer.router_logits(
                    state_block,
                    same_block,
                    cross_block,
                    degree,
                    relation_id=relation_id,
                    modality_id=modality_id,
                )
                contexts[modality].append(
                    {
                        "relation": relation,
                        "support": support,
                        "same": same,
                        "cross": cross,
                        "degree": degree,
                        "valid": valid,
                        "block": block,
                    }
                )
                blocks[modality].append(block)
                router_logits[modality].append(logits)
                valid_masks[modality].append(valid)

        next_states: dict[str, torch.Tensor] = {}
        layer_details: dict[str, Any] = {
            "layer": layer_index + 1,
            "modalities": {},
        } if collect_details else {}
        for modality_id, modality in enumerate(_MODALITIES):
            relation_valid = torch.stack(valid_masks[modality], dim=1)
            logits = torch.stack(router_logits[modality], dim=1)
            probabilities = joint_masked_softmax(logits, relation_valid)
            delta = torch.zeros_like(current[modality])
            modality_detail: dict[str, Any] = {"relations": {}}
            for relation_id, context in enumerate(contexts[modality]):
                response, expert_detail = layer.mix_experts(
                    blocks[modality][relation_id],
                    probabilities[:, relation_id, :],
                    context["valid"],
                    collect_details=collect_details,
                )
                delta = delta + response
                if collect_details:
                    active = context["valid"]
                    relation_prob = probabilities[:, relation_id, :]
                    if bool(active.any()):
                        mean_route = relation_prob[active].mean(dim=0)
                        mean_neighbors = context["degree"][active].float().mean()
                    else:
                        mean_route = relation_prob.new_zeros(self.num_experts)
                        mean_neighbors = relation_prob.new_zeros(())
                    modality_detail["relations"][context["relation"]] = {
                        "support": context["support"],
                        "valid_nodes": int(active.sum().item()),
                        "valid_node_fraction": float(active.float().mean().cpu()),
                        "mean_neighbors_when_valid": float(mean_neighbors.cpu()),
                        "same_context_rms": self._masked_rms(context["same"], active),
                        "cross_context_rms": self._masked_rms(context["cross"], active),
                        "cross_to_same_context_rms": self._rms_ratio(
                            context["cross"], context["same"], active
                        ),
                        "mean_joint_relation_mass": float(mean_route.sum().cpu()),
                        "mean_joint_expert_probabilities": [
                            float(value.cpu()) for value in mean_route
                        ],
                        **(expert_detail or {}),
                    }
            update = 0.5 * layer.update_dropout(delta)
            next_states[modality] = current[modality] + update
            if collect_details:
                participating = probabilities.sum(dim=(1, 2)) > 0
                if bool(participating.any()):
                    mean_joint = probabilities[participating].mean(dim=0)
                    entropy = -(
                        probabilities[participating]
                        * probabilities[participating].clamp_min(1.0e-12).log()
                    ).sum(dim=(1, 2)).mean()
                    route_sum = probabilities[participating].sum(dim=(1, 2))
                else:
                    mean_joint = probabilities.new_zeros((2, self.num_experts))
                    entropy = probabilities.new_zeros(())
                    route_sum = probabilities.new_zeros((0,))
                modality_detail.update(
                    {
                        "participating_nodes": int(participating.sum().item()),
                        "mean_joint_relation_marginal": {
                            relation: float(mean_joint[index].sum().cpu())
                            for index, relation in enumerate(_RELATIONS)
                        },
                        "mean_joint_expert_use": [
                            float(value.cpu())
                            for value in mean_joint.sum(dim=0)
                        ],
                        "joint_route_entropy_nats": float(entropy.cpu()),
                        "route_probability_sum_min": (
                            float(route_sum.min().cpu()) if route_sum.numel() else None
                        ),
                        "route_probability_sum_max": (
                            float(route_sum.max().cpu()) if route_sum.numel() else None
                        ),
                        "router_log_prior": layer.router_log_prior.detach()
                        .exp()
                        .cpu()
                        .tolist(),
                        "delta_rms": self._masked_rms(delta, participating),
                        "input_state_rms": self._masked_rms(
                            current[modality], participating
                        ),
                        "relative_update_rms": self._rms_ratio(
                            update, current[modality], participating
                        ),
                        "updated_state_rms": self._masked_rms(
                            next_states[modality], None
                        ),
                    }
                )
                layer_details["modalities"][modality] = modality_detail
        return next_states["text"], next_states["visual"], layer_details

    @staticmethod
    def _masked_rms(value: torch.Tensor, mask: torch.Tensor | None) -> float:
        if mask is not None:
            if not bool(mask.any()):
                return 0.0
            value = value[mask]
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().square().mean().sqrt().cpu())

    @classmethod
    def _rms_ratio(
        cls,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> float:
        denom = cls._masked_rms(denominator, mask)
        return cls._masked_rms(numerator, mask) / max(denom, 1.0e-12)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
        intervention: Mapping[str, Any] | None = None,
    ):
        if x.ndim != 2 or x.size(0) != self.num_nodes:
            raise RuntimeError("RM-CGE V1 requires the complete global-ID node matrix")
        if x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError("input width does not match configured Text/Visual dimensions")
        if x.dtype != torch.float32:
            raise ValueError("RM-CGE V1 expects the frozen feature cache dtype float32")
        if not self.semantic_cache_loaded:
            raise RuntimeError("RM-CGE requires a validated V7B semantic cache before forward")
        if edge_index is None:
            raise ValueError("RM-CGE full-graph NC requires physical edge_index")
        intervention = dict(intervention or {})
        allowed = {
            "disable_cross_modal",
            "disable_relations",
            "disable_intrinsic_skip",
        }
        unknown = set(intervention) - allowed
        if unknown:
            raise ValueError(f"unknown frozen intervention keys: {sorted(unknown)}")
        disabled_relations = set(intervention.get("disable_relations", ()))
        if disabled_relations - set(_RELATIONS):
            raise ValueError(f"disable_relations must be drawn from {_RELATIONS}")
        self._validate_input_features_once(x)
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        physical, degree_physical = self._get_physical_operator(edge_index, x.dtype)
        operators = {
            "physical": physical,
            "text": self.semantic_pt,
            "visual": self.semantic_pv,
        }
        degrees = {
            "physical": degree_physical,
            "text": self.semantic_degree_text,
            "visual": self.semantic_degree_visual,
        }
        for name, operator in operators.items():
            if operator.device != x.device:
                raise RuntimeError(f"{name} operator and input features are on different devices")

        raw = {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }
        initial = {name: self.projectors[name](raw[name]) for name in _MODALITIES}
        current = dict(initial)
        graph_details: list[dict[str, Any]] = []
        for layer_index in range(self.num_layers):
            layer_intervention = {
                **intervention,
                "disable_relations": disabled_relations,
            }
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and not return_details
            ):
                def run_layer(text_state, visual_state, index=layer_index):
                    text_next, visual_next, _ = self._execute_layer(
                        index,
                        text_state,
                        visual_state,
                        operators,
                        degrees,
                        intervention=layer_intervention,
                        collect_details=False,
                    )
                    return text_next, visual_next

                text_next, visual_next = checkpoint(
                    run_layer,
                    current["text"],
                    current["visual"],
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                layer_detail = None
            else:
                text_next, visual_next, layer_detail = self._execute_layer(
                    layer_index,
                    current["text"],
                    current["visual"],
                    operators,
                    degrees,
                    intervention=layer_intervention,
                    collect_details=return_details,
                )
            current = {"text": text_next, "visual": visual_next}
            if return_details and layer_detail is not None:
                graph_details.append(layer_detail)

        fused = torch.cat(
            (
                current["text"],
                current["visual"],
                (current["text"] - current["visual"]).abs(),
                current["text"] * current["visual"],
            ),
            dim=-1,
        )
        graph_term = self.fusion(fused)
        skip_term = self.skip(torch.cat((initial["text"], initial["visual"]), dim=-1))
        if bool(intervention.get("disable_intrinsic_skip", False)):
            skip_term = torch.zeros_like(skip_term)
        z = self.fusion_norm(graph_term + skip_term)
        aux_loss = x.new_zeros(())
        info: dict[str, Any] = {"model": "rmcge_v1"}
        if return_details:
            info["details"] = {
                "layers": graph_details,
                "fusion": {
                    "graph_term_rms": self._masked_rms(graph_term, None),
                    "intrinsic_skip_rms": self._masked_rms(skip_term, None),
                    "graph_to_skip_rms": self._rms_ratio(
                        graph_term, skip_term, None
                    ),
                    "final_embedding_rms": self._masked_rms(z, None),
                    "final_embedding_finite": bool(torch.isfinite(z).all()),
                },
            }
        return z, None, None, aux_loss, info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        # RM-CGE is a full-graph encoder; batch_size is accepted by the shared API only.
        del batch_size
        was_training = self.training
        self.eval()
        try:
            z = self.forward(
                x.to(device),
                edge_index.to(device) if edge_index is not None else None,
            )[0]
        finally:
            if was_training:
                self.train()
        return z.detach().cpu()

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        for name, parameter in self.named_parameters():
            if name.startswith("projectors."):
                group = "projectors"
            elif ".cross_modal." in name:
                group = "cross_modal"
            elif ".experts." in name:
                group = "experts"
            elif any(
                key in name
                for key in ("router.", "relation_embedding.", "modality_embedding.", "router_log_prior")
            ):
                group = "router"
            elif name.startswith(("fusion.", "skip.", "fusion_norm.")):
                group = "fusion"
            else:
                group = "graph_normalization"
            row = diagnostics.setdefault(
                group,
                {
                    "parameter_tensors": 0,
                    "parameter_count": 0,
                    "gradient_present": 0,
                    "gradient_finite": True,
                    "gradient_nonzero": 0,
                    "nonzero_tensors": 0,
                    "tensors_with_grad": 0,
                    "gradient_l2_squared": 0.0,
                },
            )
            row["parameter_tensors"] += 1
            row["parameter_count"] += int(parameter.numel())
            grad = parameter.grad
            if grad is None:
                continue
            row["tensors_with_grad"] += 1
            row["gradient_present"] += int(parameter.numel())
            finite = bool(torch.isfinite(grad).all())
            row["gradient_finite"] = row["gradient_finite"] and finite
            nonzero = int(torch.count_nonzero(grad).item())
            row["gradient_nonzero"] += nonzero
            row["nonzero_tensors"] += int(nonzero > 0)
            norm = float(grad.detach().float().norm().cpu())
            row["gradient_l2_squared"] += norm * norm
        for row in diagnostics.values():
            row["gradient_l2_norm"] = math.sqrt(row.pop("gradient_l2_squared"))
            row["all_parameter_tensors_have_finite_nonzero_gradients"] = (
                row["tensors_with_grad"] == row["parameter_tensors"]
                and row["gradient_finite"]
                and row["nonzero_tensors"] == row["tensors_with_grad"]
            )
        self._gradient_diagnostics = diagnostics
        return diagnostics
