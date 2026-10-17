"""CG-CGE V1: complementary graph experts with receiver-guided completion.

The physical and same-source encoders are computed independently for Text and
Visual. Cross-modal messages are constructed only after both unimodal states
have been finalized. The sparse receiver attention custom autograd function
keeps only edge scalars between forward and backward; edge-by-hidden-width
values are materialized a chunk at a time.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.psce_mag_v7a import _MODALITIES, chunked_weighted_spmm
from src.models.rmcge_v2 import Model as RMCGEV2Model


CONDITIONS = ("B1_complementary", "B2_conditioned_experts")
_ROLES = ("semantic", "complementary")


class _ReceiverSparseAttention(torch.autograd.Function):
    """Segmented softmax aggregation with bounded E x D temporary memory."""

    @staticmethod
    def forward(ctx, query, key, value, src, dst, chunk_size):
        if query.ndim != 2 or key.shape != query.shape:
            raise ValueError("query/key must have matching [N, attention_dim] shapes")
        if value.ndim != 2 or value.size(0) != query.size(0):
            raise ValueError("value must have shape [N, value_dim]")
        if src.ndim != 1 or dst.shape != src.shape:
            raise ValueError("src and dst must be matching edge vectors")
        if src.device.type == "cpu" and src.numel() and (
            int(src.min()) < 0 or int(src.max()) >= query.size(0)
            or int(dst.min()) < 0 or int(dst.max()) >= query.size(0)
        ):
            raise ValueError("attention edge contains an out-of-range global node ID")
        if int(chunk_size) < 1:
            raise ValueError("attention chunk size must be positive")

        n, edge_count = query.size(0), src.numel()
        scale = float(query.size(1)) ** -0.5
        q32, k32, v32 = query.float(), key.float(), value.float()
        if edge_count:
            scores = q32.new_empty((edge_count,))
            for start in range(0, edge_count, int(chunk_size)):
                end = min(start + int(chunk_size), edge_count)
                scores[start:end] = (
                    q32.index_select(0, dst[start:end])
                    * k32.index_select(0, src[start:end])
                ).sum(-1) * scale
            row_max = scores.new_full((n,), -torch.inf)
            row_max.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
            exp_scores = torch.exp(scores - row_max.index_select(0, dst))
            row_sum = scores.new_zeros((n,))
            row_sum.index_add_(0, dst, exp_scores)
            attention = exp_scores / row_sum.index_select(0, dst).clamp_min(1.0e-30)
        else:
            attention = q32.new_empty((0,))

        output = v32.new_zeros((n, value.size(1)))
        for start in range(0, edge_count, int(chunk_size)):
            end = min(start + int(chunk_size), edge_count)
            src_chunk, dst_chunk = src[start:end], dst[start:end]
            weighted = v32.index_select(0, src_chunk) * attention[start:end, None]
            output.index_add_(0, dst_chunk, weighted)

        ctx.save_for_backward(query, key, value, src, dst, attention)
        ctx.chunk_size = int(chunk_size)
        ctx.scale = scale
        return output.to(value.dtype), attention

    @staticmethod
    def backward(ctx, grad_output, grad_attention_output):
        query, key, value, src, dst, attention = ctx.saved_tensors
        n, edge_count = query.size(0), src.numel()
        q32, k32, v32 = query.float(), key.float(), value.float()
        grad_out = (
            torch.zeros_like(value, dtype=torch.float32)
            if grad_output is None
            else grad_output.float()
        )
        grad_att_direct = (
            torch.zeros_like(attention)
            if grad_attention_output is None
            else grad_attention_output.float()
        )
        grad_value = torch.zeros_like(v32)
        grad_att = torch.zeros_like(attention, dtype=torch.float32)

        # First pass computes dL/da and dL/dV without retaining edge vectors
        # wider than a single chunk.
        for start in range(0, edge_count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, edge_count)
            src_chunk, dst_chunk = src[start:end], dst[start:end]
            go = grad_out.index_select(0, dst_chunk)
            v = v32.index_select(0, src_chunk)
            alpha = attention[start:end].float()
            grad_att[start:end] = (go * v).sum(-1) + grad_att_direct[start:end]
            grad_value.index_add_(0, src_chunk, go * alpha[:, None])

        row_inner = grad_att.new_zeros((n,))
        if edge_count:
            row_inner.index_add_(0, dst, attention.float() * grad_att)
        grad_score = attention.float() * (grad_att - row_inner.index_select(0, dst))
        grad_query = torch.zeros_like(q32)
        grad_key = torch.zeros_like(k32)
        for start in range(0, edge_count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, edge_count)
            src_chunk, dst_chunk = src[start:end], dst[start:end]
            scaled = grad_score[start:end, None] * ctx.scale
            grad_query.index_add_(
                0, dst_chunk, scaled * k32.index_select(0, src_chunk)
            )
            grad_key.index_add_(
                0, src_chunk, scaled * q32.index_select(0, dst_chunk)
            )

        return (
            grad_query.to(query.dtype),
            grad_key.to(key.dtype),
            grad_value.to(value.dtype),
            None,
            None,
            None,
        )


def receiver_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    edge_index: torch.Tensor,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate only listed ``src -> dst`` edges with per-receiver softmax."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    return _ReceiverSparseAttention.apply(
        query, key, value, edge_index[0], edge_index[1], int(chunk_size)
    )


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class Model(RMCGEV2Model):
    """B1/B2 conditions share one V7A expert bank and a common initialization."""

    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    supports_link_prediction = False
    supports_unimodal_aux_loss = True
    record_run_telemetry = True
    record_epoch_history = True
    collect_v7_diagnostics = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        condition = str(cfg.model.get("condition", CONDITIONS[0]))
        if condition not in CONDITIONS:
            raise ValueError(f"cg_cge_v1 condition must be one of {CONDITIONS}")
        base_cfg = copy.deepcopy(cfg)
        # V1 creates the established directional projectors and shared expert
        # trunk; its added module initialization is isolated from the NC head.
        base_cfg.model.variant = "V1"
        super().__init__(base_cfg, data_info)
        self.study_condition = condition
        self.condition = condition
        self.variant = condition

        # These V2 modules are unused by CG-CGE. Keep directional Phi and the
        # inherited physical node router; remove old edge gates/coefficient
        # trees so they cannot be mistaken for trainable CG-CGE mechanisms.
        for name in (
            "delta_coefficients", "cross_coefficients", "knn_coefficients",
            "edge_projections", "edge_modality_embedding", "edge_scorer",
            "relation_projections", "relation_scorer", "semantic_coefficients_raw",
            "semantic_context_norms", "semantic_context_encoder",
            "semantic_selection_head", "semantic_strength_head",
        ):
            if hasattr(self, name):
                delattr(self, name)

        self.attention_dim = int(cfg.model.get("attention_dim", 32))
        self.cross_edge_chunk_size = int(cfg.model.get("cross_edge_chunk_size", 8192))
        self.view_edge_chunk_size = int(cfg.model.get("view_edge_chunk_size", self.propagation_chunk_size))
        self.view_router_hidden = int(cfg.model.get("view_router_hidden", 64))
        self.channel_gate_hidden = int(cfg.model.get("channel_gate_hidden", 128))
        self.interaction_hidden = int(cfg.model.get("interaction_hidden", 256))
        self.scale_floor = float(cfg.model.get("scale_floor", 1.0e-3))
        self.cross_lambda_max = float(cfg.model.get("cross_lambda_max", 0.5))
        self.complementary_eta = float(cfg.model.get("complementary_eta", 0.25))
        if (self.hidden_dim, self.physical_hops, self.num_experts, self.top_k,
            self.expert_bottleneck, self.attention_dim) != (256, 4, 4, 2, 64, 32):
            raise ValueError("CG-CGE freezes hidden=256, hops=4, experts=4, Top-2, bottleneck=64, attention=32")
        if min(self.cross_edge_chunk_size, self.view_edge_chunk_size, self.view_router_hidden,
               self.channel_gate_hidden, self.interaction_hidden) < 1 or self.scale_floor <= 0:
            raise ValueError("CG-CGE chunk sizes, hidden sizes, and scale floor must be positive")
        if not 0.0 < self.cross_lambda_max <= 1.0 or self.complementary_eta < 0:
            raise ValueError("invalid bounded cross residual or complementary expert scale")

        # Added modules are identical for B1/B2 and do not advance the RNG that
        # initializes the external classifier head.
        with torch.random.fork_rng(devices=[]):
            self.view_type_embedding = nn.Embedding(2, 8)
            self.view_router = nn.Sequential(
                nn.Linear(2 * self.hidden_dim + self.modality_embed_dim + 8, self.view_router_hidden),
                nn.GELU(), nn.Linear(self.view_router_hidden, self.num_experts),
            )
            nn.init.zeros_(self.view_router[-1].weight)
            nn.init.zeros_(self.view_router[-1].bias)

            self.query_projections = nn.ModuleDict({
                name: nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
                for name in _MODALITIES
            })
            self.key_projections = nn.ModuleDict({
                name: nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
                for name in _MODALITIES
            })
            for projection in list(self.query_projections.values()) + list(self.key_projections.values()):
                nn.init.normal_(projection.weight, mean=0.0, std=0.02)

            source_dim = 3 * self.hidden_dim + 5
            self.source_selector = nn.Sequential(
                nn.Linear(source_dim, 128), nn.GELU(), nn.Linear(128, 3),
            )
            nn.init.zeros_(self.source_selector[-1].weight)
            with torch.no_grad():
                self.source_selector[-1].bias.copy_(torch.log(torch.tensor([0.45, 0.35, 0.20])))

            gate_dim = 4 * self.hidden_dim
            self.channel_gate = nn.Sequential(
                nn.Linear(gate_dim, self.channel_gate_hidden), nn.GELU(),
                nn.Linear(self.channel_gate_hidden, self.hidden_dim),
            )
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.constant_(self.channel_gate[-1].bias, math.log(0.73 / 0.27))

            self.interaction = nn.Sequential(
                nn.Linear(gate_dim, self.interaction_hidden), nn.GELU(),
                nn.Linear(self.interaction_hidden, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
            )
            self.comp_modality_embedding = nn.Embedding(2, 8)
            self.comp_router = nn.Sequential(
                nn.Linear(self.hidden_dim + 8, 64), nn.GELU(),
                nn.Dropout(self.dropout_p), nn.Linear(64, self.num_experts),
            )
            nn.init.zeros_(self.comp_router[-1].weight)
            nn.init.zeros_(self.comp_router[-1].bias)

            self.role_film_scale = nn.Parameter(torch.zeros(self.num_experts, 2, 2, self.hidden_dim))
            self.role_film_shift = nn.Parameter(torch.zeros(self.num_experts, 2, 2, self.hidden_dim))
            self.cross_lambda_raw = nn.Parameter(torch.zeros(2))
            self.gamma_s_raw = nn.Parameter(torch.full((2,), _inverse_softplus(0.08)))

            self.unimodal_heads = nn.ModuleDict({
                name: nn.Linear(self.hidden_dim, int(data_info["num_classes"]))
                for name in _MODALITIES
            })

        self.use_role_film = condition == "B2_conditioned_experts"
        if not self.use_role_film:
            self.role_film_scale.requires_grad_(False)
            self.role_film_shift.requires_grad_(False)
        self._last_cg_diagnostics: dict[str, Any] = {}

    @property
    def gamma_s(self) -> torch.Tensor:
        return F.softplus(self.gamma_s_raw)

    @property
    def cross_lambda(self) -> torch.Tensor:
        return self.cross_lambda_max * torch.sigmoid(self.cross_lambda_raw)

    def _support_mask(self, edge_index: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        result = torch.zeros(n, dtype=torch.bool, device=device)
        if edge_index.numel():
            result[edge_index[1]] = True
        return result

    def _active_channel_scale(self, raw_first_hop: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if bool(active.any()):
            rms = torch.sqrt(raw_first_hop[active].float().square().mean(dim=0) + self.eps)
            return rms.to(raw_first_hop.dtype).detach().clamp_min(self.scale_floor)
        return raw_first_hop.new_full((self.hidden_dim,), self.scale_floor)

    @staticmethod
    def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if value.numel() == 0:
            return value.new_zeros((), dtype=torch.float32)
        selected = value[:, mask, :] if value.ndim == 3 else value[mask]
        if selected.numel() == 0:
            return value.new_zeros((), dtype=torch.float32)
        return selected.detach().float().square().mean().sqrt()

    def _cap_candidate(
        self, value: torch.Tensor, mask: torch.Tensor, reference_rms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_rms = self._masked_rms(value, mask)
        # The floor stabilizes the ratio but never scales weak candidates up.
        cap = torch.minimum(torch.ones_like(raw_rms), reference_rms.detach().clamp_min(self.scale_floor) /
                            raw_rms.detach().clamp_min(self.scale_floor))
        output = value * cap.to(value.dtype)
        output = output * mask.to(value.dtype).unsqueeze(-1)
        return output, raw_rms, self._masked_rms(output, mask)

    def _expert_input(self, expert_id: int, role_index: int, modality_index: int, value: torch.Tensor) -> torch.Tensor:
        if not self.use_role_film:
            return value
        normalized = F.layer_norm(value, (self.hidden_dim,))
        return value + self.role_film_scale[expert_id, role_index, modality_index] * normalized + self.role_film_shift[expert_id, role_index, modality_index]

    def _expert_mixture(
        self, value: torch.Tensor, route_weights: torch.Tensor, *, role_index: int | None,
        modality_index: int, physical_residual: bool = False,
    ) -> torch.Tensor:
        result = torch.zeros_like(value[0] if physical_residual else value)
        for expert_id, expert in enumerate(self.experts):
            source = value[expert_id] if physical_residual else value
            if role_index is not None and self.use_role_film:
                scale = self.role_film_scale[expert_id, role_index, modality_index]
                shift = self.role_film_shift[expert_id, role_index, modality_index]
                def apply(x, a, b, expert=expert):
                    adapted = x + a * F.layer_norm(x, (self.hidden_dim,)) + b
                    return expert(adapted)
                transformed = checkpoint(apply, source, scale, shift, use_reentrant=False) if self.training and torch.is_grad_enabled() else apply(source, scale, shift)
            else:
                transformed = checkpoint(expert, source, use_reentrant=False) if self.training and torch.is_grad_enabled() else expert(source)
            candidate = source + self.expert_feature_scale * transformed if physical_residual else transformed
            result = result + route_weights[:, expert_id:expert_id + 1] * candidate
        return result

    def _single_modality_state(
        self, modality: str, modality_index: int, prior: torch.Tensor,
        src: torch.Tensor, dst: torch.Tensor, physical: torch.Tensor,
        active: torch.Tensor, collect_details: bool,
    ) -> tuple[torch.Tensor, dict[str, Any], list[torch.Tensor]]:
        states = []
        state = prior
        for _ in range(self.physical_hops):
            state = self._propagate(state, src, dst, physical)
            state = state * active.to(state.dtype).unsqueeze(-1)
            states.append(state)
        scale = self._active_channel_scale(states[0], active)
        normalized = torch.stack([self._active_rms_norm(item, active) for item in states], dim=0)
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        physical_profiles = torch.einsum("kl,lnd->knd", alpha, normalized)

        base_route = self._physical_route(modality_index, prior, active)
        zero = torch.zeros_like(states[0])
        correction = self._node_correction(prior, states[0], zero, zero, modality_index)
        base_logits = self.selection_head(base_route["encoded"]).expand(prior.size(0), -1)
        physical_route = self._top2(base_logits + correction, self.top_k)
        physical_route["strength"] = base_route["strength"]
        _, physical_share, physical_balance = self._route_balance(physical_route, active)
        physical_message = self._expert_mixture(
            physical_profiles, physical_route["route_weights"], role_index=None,
            modality_index=modality_index, physical_residual=True,
        )
        physical_message = physical_message * active.to(prior.dtype).unsqueeze(-1)
        physical_response = physical_route["strength"].unsqueeze(-1) * physical_message

        semantic_edges = getattr(self, f"semantic_{modality}_edge_index_v2").to(prior.device)
        semantic_weight = getattr(self, f"semantic_{modality}_weight_v2").to(device=prior.device, dtype=prior.dtype)
        semantic_raw = chunked_weighted_spmm(prior, semantic_edges, semantic_weight, self.view_edge_chunk_size)
        semantic_mask = self._support_mask(semantic_edges, prior.size(0), prior.device)
        semantic_context = semantic_raw / scale
        semantic_context = semantic_context * semantic_mask.to(prior.dtype).unsqueeze(-1)
        role = self.view_type_embedding.weight[0].expand(prior.size(0), -1)
        modality_embedding = self.modality_embedding.weight[modality_index].expand(prior.size(0), -1)
        semantic_logits = self.view_router(torch.cat((prior, semantic_context, modality_embedding, role), dim=-1))
        semantic_route = self._top2(semantic_logits, self.top_k)
        _, semantic_share, semantic_balance = self._route_balance(semantic_route, semantic_mask)
        semantic_expert = self._expert_mixture(
            semantic_context, semantic_route["route_weights"], role_index=0,
            modality_index=modality_index,
        )
        semantic_response = (semantic_context + semantic_expert) * semantic_mask.to(prior.dtype).unsqueeze(-1)
        semantic_contribution = self.gamma_s[modality_index] * semantic_response

        unimodal = prior + physical_response + semantic_contribution
        balance = 0.5 * (physical_balance + semantic_balance)
        diagnostics = {}
        if collect_details:
            diagnostics = {
                "physical_support_nodes": int(active.sum().detach().cpu()),
                "same_semantic_support_nodes": int(semantic_mask.sum().detach().cpu()),
                "physical_profile_rms": float(self._masked_rms(physical_profiles, active).cpu()),
                "physical_response_rms": float(self._masked_rms(physical_response, active).cpu()),
                "same_semantic_raw_rms": float(self._masked_rms(semantic_raw, semantic_mask).cpu()),
                "same_semantic_response_rms": float(self._masked_rms(semantic_response, semantic_mask).cpu()),
                "same_semantic_weighted_rms": float(self._masked_rms(semantic_contribution, semantic_mask).cpu()),
                "gamma_s": float(self.gamma_s[modality_index].detach().cpu()),
                "physical_expert_selection_share": physical_share.detach().cpu().tolist(),
                "same_expert_selection_share": semantic_share.detach().cpu().tolist(),
                "unimodal_rms": float(self._masked_rms(unimodal, torch.ones_like(active)).cpu()),
            }
        route_balances = [physical_balance, semantic_balance]
        return unimodal, diagnostics, route_balances

    def _cross_completion(
        self, target: str, target_index: int, source: str,
        u_target: torch.Tensor, u_source: torch.Tensor,
        h_source: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
        physical: torch.Tensor, active: torch.Tensor, collect_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
        projection_name = "visual_to_text" if source == "visual" else "text_to_visual"
        candidate_source = 0.7 * h_source + 0.3 * u_source
        sender_value = self.cross_projections[projection_name](candidate_source)
        physical_raw = chunked_weighted_spmm(
            sender_value, torch.stack((src, dst), dim=0), physical, self.propagation_chunk_size
        )
        physical_mask = active

        support_name = source
        semantic_edges = getattr(self, f"semantic_{support_name}_edge_index_v2").to(u_target.device)
        semantic_mask = self._support_mask(semantic_edges, u_target.size(0), u_target.device)
        query = self.query_projections[target](F.layer_norm(u_target, (self.hidden_dim,)))
        key = self.key_projections[source](F.layer_norm(u_source, (self.hidden_dim,)))
        semantic_raw, attention = receiver_sparse_attention(
            query, key, sender_value, semantic_edges, self.cross_edge_chunk_size
        )
        semantic_raw = semantic_raw * semantic_mask.to(semantic_raw.dtype).unsqueeze(-1)

        physical_reference = self._masked_rms(physical_raw, physical_mask).clamp_min(self.scale_floor)
        physical_cal, physical_rms, physical_cal_rms = self._cap_candidate(
            physical_raw, physical_mask, physical_reference
        )
        semantic_cal, semantic_rms, semantic_cal_rms = self._cap_candidate(
            semantic_raw, semantic_mask, physical_reference
        )
        any_support = physical_mask | semantic_mask
        consistency = F.cosine_similarity(physical_cal.detach(), semantic_cal.detach(), dim=-1, eps=1.0e-8)
        p_rel = torch.tanh(self._masked_rms(physical_cal, physical_mask) /
                           self._masked_rms(u_target, torch.ones_like(physical_mask)).clamp_min(1.0e-6))
        s_rel = torch.tanh(self._masked_rms(semantic_cal, semantic_mask) /
                           self._masked_rms(u_target, torch.ones_like(semantic_mask)).clamp_min(1.0e-6))
        selector_features = torch.cat((
            u_target, physical_cal, semantic_cal,
            consistency[:, None], p_rel.expand(u_target.size(0), 1),
            s_rel.expand(u_target.size(0), 1),
            physical_mask.to(u_target.dtype)[:, None], semantic_mask.to(u_target.dtype)[:, None],
        ), dim=-1)
        logits = self.source_selector(selector_features)
        logits = logits.clone()
        logits[~physical_mask, 0] = -torch.inf
        logits[~semantic_mask, 1] = -torch.inf
        source_weights = torch.softmax(logits, dim=-1)
        empty_weights = source_weights.new_zeros(source_weights.shape)
        empty_weights[:, 2] = 1.0
        source_weights = torch.where(any_support[:, None], source_weights, empty_weights)
        p_accept = source_weights[:, 0] + source_weights[:, 1]
        c_bar = (source_weights[:, 0:1] * physical_cal + source_weights[:, 1:2] * semantic_cal) / p_accept.clamp_min(1.0e-6)[:, None]
        c_bar = c_bar * any_support.to(c_bar.dtype).unsqueeze(-1)

        interaction_features = torch.cat((
            u_target, c_bar, u_target * c_bar, (u_target - c_bar).abs(),
        ), dim=-1)
        gate = torch.sigmoid(self.channel_gate(interaction_features))
        interaction = self.interaction(interaction_features)
        modality_embed = self.comp_modality_embedding.weight[target_index].expand(u_target.size(0), -1)
        comp_route = self._top2(self.comp_router(torch.cat((interaction, modality_embed), dim=-1)), self.top_k)
        _, comp_share, comp_balance = self._route_balance(comp_route, any_support)
        comp_expert = self._expert_mixture(
            interaction, comp_route["route_weights"], role_index=1,
            modality_index=target_index,
        )
        delta = self.cross_lambda[target_index] * p_accept[:, None] * gate * (
            c_bar + self.complementary_eta * comp_expert
        )
        delta = delta * any_support.to(delta.dtype).unsqueeze(-1)

        if collect_details and attention.numel():
            entropy_edge = -(attention.detach().clamp_min(1.0e-30) * attention.detach().clamp_min(1.0e-30).log())
            entropy_by_receiver = entropy_edge.new_zeros(u_target.size(0))
            entropy_by_receiver.index_add_(0, semantic_edges[1], entropy_edge)
            semantic_count = torch.bincount(semantic_edges[1], minlength=u_target.size(0))
            entropy_value = float(entropy_by_receiver[semantic_mask].mean().cpu()) if bool(semantic_mask.any()) else 0.0
            multi_neighbor = semantic_count > 1
            entropy_multi = float(entropy_by_receiver[multi_neighbor].mean().cpu()) if bool(multi_neighbor.any()) else 0.0
        elif collect_details:
            entropy_value, entropy_multi = 0.0, 0.0
        diagnostics = {}
        if collect_details:
            diagnostics = {
                "physical_cross_raw_rms": float(physical_rms.cpu()),
                "physical_cross_calibrated_rms": float(physical_cal_rms.cpu()),
                "semantic_cross_raw_rms": float(semantic_rms.cpu()),
                "semantic_cross_calibrated_rms": float(semantic_cal_rms.cpu()),
                "physical_support_nodes": int(physical_mask.sum().detach().cpu()),
                "semantic_support_nodes": int(semantic_mask.sum().detach().cpu()),
                "physical_source_weight": float(source_weights[:, 0].detach().mean().cpu()),
                "semantic_source_weight": float(source_weights[:, 1].detach().mean().cpu()),
                "none_source_weight": float(source_weights[:, 2].detach().mean().cpu()),
                "accepted_source_fraction": float((p_accept.detach() > 0).float().mean().cpu()),
                "channel_gate_mean": float(gate.detach().mean().cpu()),
                "channel_gate_saturation_fraction": float(((gate.detach() < 0.05) | (gate.detach() > 0.95)).float().mean().cpu()),
                "complementary_expert_rms": float(self._masked_rms(comp_expert, any_support).cpu()),
                "cross_residual_rms": float(self._masked_rms(delta, any_support).cpu()),
                "cross_to_unimodal_rms": float((self._masked_rms(delta, any_support) /
                    self._masked_rms(u_target, torch.ones_like(any_support)).clamp_min(1.0e-8)).cpu()),
                "complementary_expert_selection_share": comp_share.detach().cpu().tolist(),
                "semantic_attention_entropy": entropy_value,
                "semantic_attention_entropy_multi_neighbor": entropy_multi,
                "query_rms": float(query.detach().float().square().mean().sqrt().cpu()),
                "key_rms": float(key.detach().float().square().mean().sqrt().cpu()),
                "role_film_enabled": self.use_role_film,
            }
        return u_target + delta, delta, source_weights, diagnostics, comp_balance

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if edge_index is None:
            raise ValueError("CG-CGE V1 requires the global physical edge_index; link prediction is unsupported")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("CG-CGE semantic caches use global node IDs; sampled/local node IDs are unsupported")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("CG-CGE V1 requires a validated V7B.1 semantic cache")

        inputs = self._split_modalities(x)
        priors = {modality: self.projectors[modality](inputs[modality]) for modality in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        unimodal: dict[str, torch.Tensor] = {}
        modality_diagnostics: dict[str, Any] = {}
        balance_terms: list[torch.Tensor] = []
        for index, modality in enumerate(_MODALITIES):
            unimodal[modality], modality_diagnostics[modality], local_balance = self._single_modality_state(
                modality, index, priors[modality], src, dst, physical, active, return_details
            )
            balance_terms.extend(local_balance)

        completed: dict[str, torch.Tensor] = {}
        cross_diagnostics: dict[str, Any] = {}
        for target_index, target in enumerate(_MODALITIES):
            source = "visual" if target == "text" else "text"
            completed[target], delta, weights, cross_diagnostics[target], comp_balance = self._cross_completion(
                target, target_index, source, unimodal[target], unimodal[source], priors[source],
                src, dst, physical, active, return_details,
            )
            modality_diagnostics[target]["cross_completion"] = cross_diagnostics[target]
            balance_terms.append(comp_balance)

        balance = torch.stack(balance_terms).mean() if balance_terms else x.new_zeros(())
        aux_loss = self.balance_weight * balance
        fused_input = torch.cat((completed["text"], completed["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused_input)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused_input))))
        )
        info: dict[str, Any] = {
            "variant": self.study_condition,
            "condition": self.study_condition,
            "active_nodes": active.detach(),
            "input_self_loops_removed": loops,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "role_film_enabled": self.use_role_film,
        }
        if self.training:
            info["unimodal_logits"] = {
                modality: self.unimodal_heads[modality](unimodal[modality])
                for modality in _MODALITIES
            }
        if return_details:
            info["details"] = {
                "modalities": modality_diagnostics,
                "fusion_input_rms": float(fused_input.detach().float().square().mean().sqrt().cpu()),
                "fused_output_rms": float(z.detach().float().square().mean().sqrt().cpu()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "role_film_parameters": sum(p.numel() for p in (self.role_film_scale, self.role_film_shift)),
                "role_film_trainable_parameters": sum(p.numel() for p in (self.role_film_scale, self.role_film_shift) if p.requires_grad),
            }
        return z, None, None, aux_loss, info

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        parameters = dict(self.named_parameters())
        names = (
            "gamma_s_raw", "view_router.2.weight", "cross_lambda_raw",
            "query_projections.text.weight", "query_projections.visual.weight",
            "key_projections.text.weight", "key_projections.visual.weight",
            "cross_projections.visual_to_text.0.weight",
            "cross_projections.text_to_visual.0.weight",
            "source_selector.2.weight", "channel_gate.2.weight",
            "comp_router.3.weight", "unimodal_heads.text.weight",
            "unimodal_heads.visual.weight", "role_film_scale", "role_film_shift",
        )
        result: dict[str, Any] = {}
        for name in names:
            parameter = parameters[name]
            gradient = parameter.grad
            norm = float(gradient.detach().float().norm().cpu()) if gradient is not None else 0.0
            result[name] = {
                "gradient_present": gradient is not None,
                "gradient_finite": bool(gradient is None or torch.isfinite(gradient).all()),
                "gradient_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        # The NC trainer serializes this attribute after the first backward
        # pass. Keep the local name for model diagnostics and mirror the
        # trainer's telemetry contract.
        self._last_gradient_diagnostics = result
        self._gradient_diagnostics = result
        return result


__all__ = ["Model", "CONDITIONS", "receiver_sparse_attention"]
