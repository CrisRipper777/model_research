"""RM-CGE V2.1: message-purpose conditioned selection on the fixed MAG."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.psce_mag_v7a import chunked_weighted_spmm
from src.models.rmcge_v2 import Model as RMCGEV2Model, _MODALITIES


CONDITIONS = ("C0", "C1", "C2")
PURPOSES = ("intra", "cross")


def conserve_receiver_mass(
    physical_weight: torch.Tensor,
    scores: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Reweight directed edges while preserving each receiver's physical mass."""
    if physical_weight.ndim != 1 or scores.shape != physical_weight.shape:
        raise ValueError("physical_weight and scores must be matching edge vectors")
    if dst.shape != physical_weight.shape:
        raise ValueError("dst must align with edge weights")
    if dst.numel() and (int(dst.min()) < 0 or int(dst.max()) >= num_nodes):
        raise ValueError("dst contains a node outside [0, num_nodes)")
    p = physical_weight.float()
    s = scores.float()
    # The model bounds scores with tanh. Keep the helper safe for external use too.
    if not torch.isfinite(p).all() or not torch.isfinite(s).all():
        raise ValueError("physical weights and scores must be finite")
    mass = torch.zeros(num_nodes, dtype=torch.float32, device=p.device)
    denominator = torch.zeros_like(mass)
    mass.index_add_(0, dst, p)
    unnormalized = torch.exp(s)
    denominator.index_add_(0, dst, p * unnormalized)
    valid = denominator[dst] > 0 if dst.numel() else dst.new_empty((0,), dtype=torch.bool)
    safe_denom = denominator[dst].clamp_min(torch.finfo(torch.float32).tiny)
    selected = p * unnormalized * (mass[dst] / safe_denom)
    selected = torch.where(valid, selected, torch.zeros_like(selected))
    return selected.to(dtype=physical_weight.dtype)


class Model(RMCGEV2Model):
    """C0 delegates to the frozen V2/V1 forward; C1 and C2 add two residual paths."""

    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    supports_link_prediction = False
    record_run_telemetry = True
    record_epoch_history = True
    collect_v7_diagnostics = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        requested = str(cfg.model.get("condition", cfg.model.get("variant", "C0"))).upper()
        if requested not in CONDITIONS:
            raise ValueError(f"rmcge_v21 condition must be one of {CONDITIONS}")
        # All three conditions begin with the exact V2/V1 public parameter tree.
        base_cfg = copy.deepcopy(cfg)
        base_cfg.model.variant = "V1"
        super().__init__(base_cfg, data_info)
        self.condition = requested
        self.study_variant = "V1"
        self._v21_intervention = {
            "intra_selection_off": False,
            "cross_selection_off": False,
            "cross_off": False,
            "knn_off": False,
            "node_router_off": False,
        }
        self._intervention.update(
            {"force_unit_edges": False, "cross_off": False, "knn_off": False, "node_router_off": False}
        )
        self.purpose_edge_chunk_size = int(cfg.model.get("purpose_edge_chunk_size", self.edge_chunk_size))
        if self.purpose_edge_chunk_size < 1:
            raise ValueError("purpose_edge_chunk_size must be positive")
        if requested != "C0":
            # These modules must not advance the RNG used later for the paired head.
            with torch.random.fork_rng(devices=[]):
                self.query_projection = nn.Linear(self.hidden_dim, 32, bias=False)
                self.key_projection = nn.Linear(self.hidden_dim, 32, bias=False)
                self.edge_target_modality_embedding = nn.Embedding(2, 8)
                self.edge_purpose_embedding = nn.Embedding(2, 8)
                self.purpose_edge_scorer = nn.Sequential(
                    nn.Linear(147, 64), nn.GELU(), nn.Linear(64, 1)
                )
                nn.init.zeros_(self.purpose_edge_scorer[-1].weight)
                nn.init.zeros_(self.purpose_edge_scorer[-1].bias)
                self.cross_delta_coefficients = nn.Parameter(torch.full((self.num_experts,), 0.20))

    def set_interventions(
        self,
        *,
        force_unit_edges: bool = False,
        intra_selection_off: bool = False,
        cross_selection_off: bool = False,
        cross_off: bool = False,
        knn_off: bool = False,
        node_router_off: bool = False,
    ) -> None:
        # Retain the legacy keyword for checkpoint/audit tooling; V2.1 never
        # uses V2's global edge multiplier.
        del force_unit_edges
        self._v21_intervention = {
            "intra_selection_off": bool(intra_selection_off),
            "cross_selection_off": bool(cross_selection_off),
            "cross_off": bool(cross_off),
            "knn_off": bool(knn_off),
            "node_router_off": bool(node_router_off),
        }
        self._intervention.update(
            {
                "force_unit_edges": False,
                "cross_off": bool(cross_off),
                "knn_off": bool(knn_off),
                "node_router_off": bool(node_router_off),
            }
        )

    def _score_message_edges(
        self,
        receiver: torch.Tensor,
        sender_message: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        degree: torch.Tensor,
        modality_index: int,
        purpose_index: int,
    ) -> torch.Tensor:
        if src.numel() == 0:
            return receiver.new_empty((0,), dtype=torch.float32)
        log_degree = torch.log1p(degree.float())
        modality = self.edge_target_modality_embedding.weight[modality_index]
        purpose = self.edge_purpose_embedding.weight[purpose_index]
        receiver_unit = F.normalize(receiver.float(), dim=-1, eps=1.0e-8)
        sender_unit = F.normalize(sender_message.float(), dim=-1, eps=1.0e-8)
        chunks: list[torch.Tensor] = []
        for start in range(0, src.numel(), self.purpose_edge_chunk_size):
            end = min(start + self.purpose_edge_chunk_size, src.numel())
            src_chunk, dst_chunk = src[start:end], dst[start:end]

            def score_chunk(recv, send, src_ids, dst_ids, logdeg, mod, role):
                q = self.query_projection(recv[dst_ids])
                k = self.key_projection(send[src_ids])
                cosine = (receiver_unit[dst_ids] * sender_unit[src_ids]).sum(-1, keepdim=True)
                features = torch.cat(
                    (
                        q,
                        k,
                        q * k,
                        (q - k).abs(),
                        cosine,
                        logdeg[dst_ids, None],
                        logdeg[src_ids, None],
                        mod.expand(src_ids.numel(), -1),
                        role.expand(src_ids.numel(), -1),
                    ),
                    dim=-1,
                )
                return torch.tanh(self.purpose_edge_scorer(features).squeeze(-1))

            if self.training and torch.is_grad_enabled():
                values = checkpoint(
                    score_chunk,
                    receiver,
                    sender_message,
                    src_chunk,
                    dst_chunk,
                    log_degree,
                    modality,
                    purpose,
                    use_reentrant=False,
                )
            else:
                values = score_chunk(
                    receiver, sender_message, src_chunk, dst_chunk, log_degree, modality, purpose
                )
            chunks.append(values)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _selection_path_diagnostics(
        physical: torch.Tensor,
        selected: torch.Tensor,
        scores: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        degree: torch.Tensor,
    ) -> dict[str, Any]:
        p = physical.detach().float()
        chosen = selected.detach().float()
        score = scores.detach().float()
        n = int(degree.numel())
        rows_p = torch.zeros(n, dtype=torch.float32, device=p.device)
        rows_q = torch.zeros_like(rows_p)
        rows_p.index_add_(0, dst, p)
        rows_q.index_add_(0, dst, chosen)
        row_error = (rows_q - rows_p).abs()
        degree = degree.detach().long()
        eligible = degree >= 2
        single = degree == 1
        p_prob = p / rows_p[dst].clamp_min(1.0e-30)
        q_prob = chosen / rows_q[dst].clamp_min(1.0e-30)
        midpoint = 0.5 * (p_prob + q_prob)
        edge_entropy_p = -p_prob.clamp_min(1.0e-30).log() * p_prob
        edge_entropy_q = -q_prob.clamp_min(1.0e-30).log() * q_prob
        entropy_p = torch.zeros_like(rows_p)
        entropy_q = torch.zeros_like(rows_q)
        entropy_p.index_add_(0, dst, edge_entropy_p)
        entropy_q.index_add_(0, dst, edge_entropy_q)
        ess_terms = q_prob.square()
        ess_inv = torch.zeros_like(rows_q)
        ess_inv.index_add_(0, dst, ess_terms)
        kl_edge = q_prob * (q_prob.clamp_min(1.0e-30).log() - p_prob.clamp_min(1.0e-30).log())
        kl_rows = torch.zeros_like(rows_q)
        kl_rows.index_add_(0, dst, kl_edge)
        js_p = p_prob * (p_prob.clamp_min(1.0e-30).log() - midpoint.clamp_min(1.0e-30).log())
        js_q = q_prob * (q_prob.clamp_min(1.0e-30).log() - midpoint.clamp_min(1.0e-30).log())
        js_rows = torch.zeros_like(rows_q)
        js_rows.index_add_(0, dst, 0.5 * (js_p + js_q))
        count = torch.zeros_like(rows_q)
        within_sq = torch.zeros_like(rows_q)
        ratio = chosen / p.clamp_min(1.0e-30)
        ratio_mean = torch.zeros_like(rows_q)
        ratio_mean.index_add_(0, dst, ratio)
        count.index_add_(0, dst, torch.ones_like(ratio))
        ratio_mean = ratio_mean / count.clamp_min(1.0)
        within_sq.index_add_(0, dst, (ratio - ratio_mean[dst]).square())
        within_std = (within_sq / count.clamp_min(1.0)).sqrt()

        def mean_on(values: torch.Tensor, mask: torch.Tensor) -> float | None:
            return float(values[mask].mean().cpu()) if bool(mask.any()) else None

        ratio_quantiles = torch.quantile(ratio, ratio.new_tensor([0.05, 0.5, 0.95])) if ratio.numel() else ratio.new_zeros(3)
        reverse_difference: float | None = None
        if src.numel():
            key = src * n + dst
            order = torch.argsort(key)
            sorted_key = key[order]
            reverse_key = dst * n + src
            positions = torch.searchsorted(sorted_key, reverse_key).clamp_max(sorted_key.numel() - 1)
            valid = sorted_key[positions] == reverse_key
            if bool(valid.any()):
                reverse_difference = float((chosen[valid] - chosen[order[positions[valid]]]).abs().mean().cpu())
        return {
            "edge_count": int(p.numel()),
            "receiver_row_mass_error_max": float(row_error.max().cpu()) if row_error.numel() else 0.0,
            "receiver_row_mass_error_mean": float(row_error.mean().cpu()) if row_error.numel() else 0.0,
            "degree_ge2_node_fraction": float(eligible.float().mean().cpu()) if n else 0.0,
            "degree_one_node_fraction": float(single.float().mean().cpu()) if n else 0.0,
            "relative_weight_mean": float(ratio.mean().cpu()) if ratio.numel() else 1.0,
            "relative_weight_mean_abs_deviation": float((ratio - 1.0).abs().mean().cpu()) if ratio.numel() else 0.0,
            "relative_weight_std": float(ratio.std(unbiased=False).cpu()) if ratio.numel() else 0.0,
            "relative_weight_p05": float(ratio_quantiles[0].cpu()),
            "relative_weight_p50": float(ratio_quantiles[1].cpu()),
            "relative_weight_p95": float(ratio_quantiles[2].cpu()),
            "within_receiver_relative_weight_std_mean_degree_ge2": mean_on(within_std, eligible),
            "neighbor_entropy_physical_mean_degree_ge2": mean_on(entropy_p, eligible),
            "neighbor_entropy_selected_mean_degree_ge2": mean_on(entropy_q, eligible),
            "neighbor_entropy_selected_mean_degree1": mean_on(entropy_q, single),
            "effective_neighbors_mean_degree_ge2": mean_on(ess_inv.clamp_min(1.0e-30).reciprocal(), eligible),
            "effective_neighbors_mean_degree1": mean_on(ess_inv.clamp_min(1.0e-30).reciprocal(), single),
            "kl_selected_to_physical_mean_degree_ge2": mean_on(kl_rows, eligible),
            "js_selected_physical_mean_degree_ge2": mean_on(js_rows, eligible),
            "score_abs_ge_0_99_fraction": float((score.abs() >= 0.99).float().mean().cpu()) if score.numel() else 0.0,
            "score_abs_mean": float(score.abs().mean().cpu()) if score.numel() else 0.0,
            "reverse_direction_selected_weight_abs_difference": reverse_difference,
        }

    @staticmethod
    def _compare_path_distributions(
        left: torch.Tensor, right: torch.Tensor, dst: torch.Tensor, degree: torch.Tensor
    ) -> dict[str, float | None]:
        left, right = left.detach().float(), right.detach().float()
        n = int(degree.numel())
        left_mass = torch.zeros(n, device=left.device)
        right_mass = torch.zeros_like(left_mass)
        left_mass.index_add_(0, dst, left)
        right_mass.index_add_(0, dst, right)
        p = left / left_mass[dst].clamp_min(1.0e-30)
        q = right / right_mass[dst].clamp_min(1.0e-30)
        midpoint = 0.5 * (p + q)
        js_edge = 0.5 * (
            p * (p.clamp_min(1.0e-30).log() - midpoint.clamp_min(1.0e-30).log())
            + q * (q.clamp_min(1.0e-30).log() - midpoint.clamp_min(1.0e-30).log())
        )
        js = torch.zeros(n, device=left.device)
        js.index_add_(0, dst, js_edge)
        eligible = degree >= 2
        denominator = p.std(unbiased=False) * q.std(unbiased=False)
        corr = float((((p - p.mean()) * (q - q.mean())).mean() / denominator).cpu()) if float(denominator.cpu()) > 1.0e-12 else None
        return {
            "edge_probability_mean_abs_difference": float((p - q).abs().mean().cpu()) if p.numel() else 0.0,
            "edge_probability_correlation": corr,
            "receiver_js_mean_degree_ge2": float(js[eligible].mean().cpu()) if bool(eligible.any()) else None,
        }

    def _forward_v21_added(self, x: torch.Tensor, edge_index: torch.Tensor, *, return_details: bool):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        degree_float = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        degree_float.index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.float32, device=x.device))
        degree = degree_float.long()
        edge_index_directed = torch.stack((src, dst), dim=0)
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        outputs: dict[str, torch.Tensor] = {}
        balances: list[torch.Tensor] = []
        profile_diagnostics: dict[str, dict[str, Any]] = {}
        selection_weights: dict[str, torch.Tensor] = {}
        selection_scores: dict[str, torch.Tensor] = {}
        path_diagnostics: dict[str, dict[str, Any]] = {}

        for modality_index, modality in enumerate(_MODALITIES):
            other = "visual" if modality == "text" else "text"
            prior = priors[modality]
            state = prior
            raw_states: list[torch.Tensor] = []
            for _ in range(self.physical_hops):
                state = self._propagate(state, src, dst, physical)
                state = state * active.to(state.dtype).unsqueeze(-1)
                raw_states.append(state)
            raw_u1 = raw_states[0]
            normalized_states = torch.stack([self._active_rms_norm(value, active) for value in raw_states], dim=0)
            if bool(active.any()):
                scale = torch.sqrt(
                    raw_u1[active].float().square().mean(dim=0) + self.eps
                ).to(prior.dtype)
            else:
                scale = prior.new_zeros((self.hidden_dim,))
            scale = scale.detach().clamp_min(self.scale_floor)

            x_cross = self.cross_projections[
                "visual_to_text" if modality == "text" else "text_to_visual"
            ](priors[other])
            cross_base = chunked_weighted_spmm(x_cross, edge_index_directed, physical, self.propagation_chunk_size)
            if self._v21_intervention["cross_off"]:
                cross_base = torch.zeros_like(prior)
            semantic_edges = getattr(self, f"semantic_{modality}_edge_index_v2")
            semantic_weight = getattr(self, f"semantic_{modality}_weight_v2").to(device=prior.device, dtype=prior.dtype)
            if semantic_edges.device != prior.device:
                semantic_edges = semantic_edges.to(prior.device)
            semantic = (
                torch.zeros_like(prior)
                if self._v21_intervention["knn_off"]
                else chunked_weighted_spmm(prior, semantic_edges, semantic_weight, self.propagation_chunk_size)
            )

            zero_scores = physical.new_zeros(src.numel())
            if self._v21_intervention["intra_selection_off"]:
                intra_scores = zero_scores
                intra_selected = physical
            else:
                intra_scores = self._score_message_edges(prior, prior, src, dst, degree_float, modality_index, 0)
                intra_selected = conserve_receiver_mass(physical, intra_scores, dst, x.size(0))
            intra_delta = chunked_weighted_spmm(
                prior, edge_index_directed, intra_selected - physical, self.propagation_chunk_size
            )

            if self.condition == "C2" and not self._v21_intervention["cross_selection_off"]:
                cross_scores = self._score_message_edges(prior, x_cross, src, dst, degree_float, modality_index, 1)
                cross_selected = conserve_receiver_mass(physical, cross_scores, dst, x.size(0))
                cross_delta = chunked_weighted_spmm(
                    x_cross, edge_index_directed, cross_selected - physical, self.propagation_chunk_size
                )
            else:
                cross_scores = zero_scores
                cross_selected = physical
                cross_delta = torch.zeros_like(prior)
            if self._v21_intervention["cross_off"]:
                cross_base = torch.zeros_like(prior)

            intra_hat = intra_delta / scale
            cross_base_hat = cross_base / scale
            cross_delta_hat = cross_delta / scale
            semantic_hat = semantic / scale
            physical_profiles = torch.einsum(
                "kl,lnd->knd", alpha, normalized_states
            )
            q_profiles = (
                physical_profiles
                + self.delta_coefficients[:, None, None] * intra_hat[None, :, :]
                + self.cross_coefficients[:, None, None] * cross_base_hat[None, :, :]
                + self.knn_coefficients[:, None, None] * semantic_hat[None, :, :]
                + self.cross_delta_coefficients[:, None, None] * cross_delta_hat[None, :, :]
            )
            expert_outputs = []
            for expert_id, expert in enumerate(self.experts):
                profile = q_profiles[expert_id]
                transformed = checkpoint(expert, profile, use_reentrant=False) if self.training and torch.is_grad_enabled() else expert(profile)
                expert_outputs.append(profile + self.expert_feature_scale * transformed)
            expert_values = torch.stack(expert_outputs, dim=0)

            base_route = self._physical_route(modality_index, prior, active)
            if self._v21_intervention["node_router_off"]:
                route = base_route
                correction_logits = prior.new_zeros((prior.size(0), self.num_experts))
            else:
                route_cross = cross_base_hat + cross_delta_hat if self.condition == "C2" else cross_base_hat
                correction_logits = self._node_correction(
                    prior, raw_u1, intra_hat, route_cross, modality_index
                )
                base_logits = self.selection_head(base_route["encoded"]).expand(prior.size(0), -1)
                route = self._top2(base_logits + correction_logits, self.top_k)
                route["strength"] = base_route["strength"]
                route["router_state"] = base_route["router_state"]
                route["static_input"] = base_route["static_input"]
                route["encoded"] = base_route["encoded"]
                importance, selection_share, balance = self._route_balance(route, active)
                route.update({"active": active, "importance": importance, "selection_share": selection_share, "balance": balance})
            if self._v21_intervention["node_router_off"]:
                route["active"] = active
            balances.append(route["balance"])
            mixture = torch.einsum("nk,knd->nd", route["route_weights"], expert_values)
            correction = route["strength"].unsqueeze(-1) * mixture
            outputs[modality] = prior + correction
            selection_weights[f"{modality}_intra"] = intra_selected
            selection_scores[f"{modality}_intra"] = intra_scores
            if self.condition == "C2":
                selection_weights[f"{modality}_cross"] = cross_selected
                selection_scores[f"{modality}_cross"] = cross_scores
            if return_details:
                profile_diagnostics[modality] = {
                    "u1_rms": self._scalar_rms(raw_u1, active),
                    "delta_rms": self._scalar_rms(intra_delta, active),
                    "delta_hat_rms": self._scalar_rms(intra_hat, active),
                    "delta_weighted_rms_by_expert": [self._scalar_rms(self.delta_coefficients[k] * intra_hat, active) for k in range(self.num_experts)],
                    "cross_rms": self._scalar_rms(cross_base, active),
                    "cross_hat_rms": self._scalar_rms(cross_base_hat, active),
                    "cross_weighted_rms_by_expert": [self._scalar_rms(self.cross_coefficients[k] * cross_base_hat, active) for k in range(self.num_experts)],
                    "cross_delta_rms": self._scalar_rms(cross_delta, active),
                    "cross_delta_hat_rms": self._scalar_rms(cross_delta_hat, active),
                    "cross_delta_weighted_rms_by_expert": [self._scalar_rms(self.cross_delta_coefficients[k] * cross_delta_hat, active) for k in range(self.num_experts)],
                    "knn_rms": self._scalar_rms(semantic, active),
                    "knn_hat_rms": self._scalar_rms(semantic_hat, active),
                    "knn_weighted_rms_by_expert": [self._scalar_rms(self.knn_coefficients[k] * semantic_hat, active) for k in range(self.num_experts)],
                    "floor_hit_dimensions": int((scale <= self.scale_floor).sum().item()),
                    "physical_profile_rms_by_expert": [self._scalar_rms(physical_profiles[k], active) for k in range(self.num_experts)],
                    "combined_profile_rms_by_expert": [self._scalar_rms(q_profiles[k], active) for k in range(self.num_experts)],
                    "expert_output_rms": self._scalar_rms(expert_values, active),
                    "expert_output_rms_by_expert": [self._scalar_rms(expert_values[k], active) for k in range(self.num_experts)],
                    "expert_output_pairwise_cosine": self._expert_pairwise_cosines(expert_values, active),
                    "weighted_graph_correction_rms": self._scalar_rms(correction, active),
                    "intrinsic_rms": self._scalar_rms(prior, active),
                    "router_correction_rms": self._scalar_rms(correction_logits, active),
                    "router_entropy": self._route_entropy(route["dense_probs"], active),
                    "dense_probability_mean": route["dense_probs"][active].mean(dim=0).detach().cpu().tolist() if bool(active.any()) else [0.0] * self.num_experts,
                    "expert_selection_share": route["selection_share"].detach().cpu().tolist(),
                    "base_expert_selection_share": base_route["selection_share"].detach().cpu().tolist(),
                    "router_top2_flip_fraction": float((route["top_indices"][active] != base_route["top_indices"][active]).any(dim=-1).float().mean().cpu()) if bool(active.any()) else 0.0,
                    "expert_coefficients": {
                        "intra": self.delta_coefficients.detach().cpu().tolist(),
                        "cross_base": self.cross_coefficients.detach().cpu().tolist(),
                        "semantic": self.knn_coefficients.detach().cpu().tolist(),
                        "cross_delta": self.cross_delta_coefficients.detach().cpu().tolist(),
                    },
                }
            del raw_states, normalized_states

        balance = 0.5 * (balances[0] + balances[1])
        aux_loss = self.balance_weight * balance
        fused_input = torch.cat((outputs["text"], outputs["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused_input)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused_input))))
        )
        info: dict[str, Any] = {
            "variant": self.condition,
            "active_nodes": active.detach(),
            "input_self_loops_removed": loops,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "interventions": dict(self._v21_intervention),
        }
        if return_details:
            for modality_index, modality in enumerate(_MODALITIES):
                path_names = ((modality, "intra", 0),) + (((modality, "cross", 1),) if self.condition == "C2" else ())
                for name, purpose, purpose_index in path_names:
                    key = f"{name}_{purpose}"
                    path_diagnostics[key] = self._selection_path_diagnostics(
                        physical, selection_weights[key], selection_scores[key], src, dst, degree
                    )
            pair_diagnostics: dict[str, Any] = {}
            path_keys = list(selection_weights)
            for left_index, left in enumerate(path_keys):
                for right in path_keys[left_index + 1 :]:
                    pair_diagnostics[f"{left}__{right}"] = self._compare_path_distributions(
                        selection_weights[left], selection_weights[right], dst, degree
                    )
            info["details"] = {
                "modalities": profile_diagnostics,
                "edge_selection": path_diagnostics,
                "role_distribution_comparisons": pair_diagnostics,
                "fusion_input_rms": self._scalar_rms(fused_input),
                "fusion_input_mean": float(fused_input.detach().float().mean().cpu()),
                "fusion_input_std": float(fused_input.detach().float().std(unbiased=False).cpu()),
                "fused_output_rms": self._scalar_rms(z),
                "fused_output_mean": float(z.detach().float().mean().cpu()),
                "fused_output_std": float(z.detach().float().std(unbiased=False).cpu()),
                "gradient_diagnostics": self._last_gradient_diagnostics,
            }
        return z, None, None, aux_loss, info

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if self.condition == "C0":
            result = super().forward(x, edge_index, return_details=return_details)
            result[4]["variant"] = "C0"
            result[4]["condition"] = "C0"
            return result
        if edge_index is None:
            raise ValueError("RM-CGE V2.1 requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("RM-CGE V2.1 is full-graph NC only")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("RM-CGE V2.1 requires a validated V7B.1 semantic cache")
        return self._forward_v21_added(x, edge_index, return_details=return_details)

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        if self.condition == "C0":
            return super().capture_gradient_diagnostics()
        parent = super().capture_gradient_diagnostics()
        params = dict(self.named_parameters())
        for name in (
            "query_projection.weight",
            "key_projection.weight",
            "purpose_edge_scorer.0.weight",
            "purpose_edge_scorer.2.weight",
            "purpose_edge_scorer.2.bias",
            "delta_coefficients",
            "cross_coefficients",
            "knn_coefficients",
            "cross_delta_coefficients",
        ):
            grad = params[name].grad
            norm = float(grad.detach().float().norm().cpu()) if grad is not None else 0.0
            parent[name] = {
                "gradient_present": grad is not None,
                "gradient_finite": bool(grad is None or torch.isfinite(grad).all()),
                "gradient_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        self._last_gradient_diagnostics = parent
        return parent


__all__ = ["Model", "CONDITIONS", "PURPOSES", "conserve_receiver_mass"]
