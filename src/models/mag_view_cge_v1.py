"""MAG View-Collaborative Experts V1.

The legacy A0 arm delegates directly to RM-CGE V2.2 M0. A1/A2 keep the
physical expert/router path and add same-source and cross-source semantic
views that call the exact same expert modules.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b
from src.models.rmcge_v2 import build_modality_knn_operators
from src.models.rmcge_v22 import Model as RMCGEV22Model
from src.models.psce_mag_v7a import chunked_weighted_spmm


CONDITIONS = ("A0_legacy", "A1_separate_intra", "A2_cross_source")
_MODALITIES = ("text", "visual")


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def build_validated_view_operators(
    unique_edge_index: torch.Tensor,
    source_flags: torch.Tensor,
    num_nodes: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Validate V7B.1's canonical cache contract then build V2 row supports."""
    if unique_edge_index.ndim != 2 or unique_edge_index.shape[0] != 2:
        raise ValueError("unique_edge_index must have shape [2,E]")
    if source_flags.shape != (unique_edge_index.size(1), 2):
        raise ValueError("source_flags must have shape [E,2]")
    if unique_edge_index.numel():
        src, dst = unique_edge_index
        if int(unique_edge_index.min()) < 0 or int(unique_edge_index.max()) >= num_nodes:
            raise ValueError("semantic cache edge contains an out-of-range global node ID")
        if bool((src >= dst).any()):
            raise ValueError("semantic cache edges must be unique canonical non-self pairs (src < dst)")
        keys = src * num_nodes + dst
        if torch.unique(keys).numel() != keys.numel():
            raise ValueError("semantic cache contains duplicate canonical pairs")
    if source_flags.is_floating_point() or source_flags.is_complex():
        raise ValueError("V7B.1 source provenance counts must use an integer dtype")
    if source_flags.numel() and bool((source_flags < 0).any()):
        raise ValueError("V7B.1 source provenance counts cannot be negative")
    if source_flags.size(0) and bool((source_flags.sum(dim=1) == 0).any()):
        raise ValueError("every V7B.1 candidate pair must have at least one source modality")
    return build_modality_knn_operators(unique_edge_index, source_flags, num_nodes)


class Model(RMCGEV22Model):
    """Three conditions over one V2.2 legacy backbone and one shared Expert Bank."""

    supports_link_prediction = False
    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    record_run_telemetry = True
    record_epoch_history = True
    collect_v7_diagnostics = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        requested = str(cfg.model.get("condition", "A0_legacy"))
        if requested not in CONDITIONS:
            raise ValueError(f"mag_view_cge_v1 condition must be one of {CONDITIONS}")
        base_cfg = copy.deepcopy(cfg)
        # This exact constructor path creates the original V2.2 M0 tensor tree.
        base_cfg.model.condition = "M0"
        super().__init__(base_cfg, data_info)
        self.study_condition = requested
        self.condition = requested
        self.variant = requested
        self.view_router_hidden = int(cfg.model.get("view_router_hidden", 64))
        self.view_edge_chunk_size = int(cfg.model.get("view_edge_chunk_size", self.propagation_chunk_size))
        self.view_scale_floor = float(cfg.model.get("view_scale_floor", self.scale_floor))
        if (self.hidden_dim, self.num_experts, self.top_k, self.view_router_hidden) != (256, 4, 2, 64):
            raise ValueError("View-CGE V1 freezes hidden=256, experts=4, Top-2, view_router_hidden=64")
        if self.view_edge_chunk_size < 1 or self.view_scale_floor <= 0:
            raise ValueError("view edge chunk and scale floor must be positive")

        # fork_rng ensures every common V2.2 parameter and the later NC head keep
        # their exact initialization stream across A0/A1/A2.
        with torch.random.fork_rng(devices=[]):
            self.view_type_embedding = nn.Embedding(2, 8)
            self.view_router = nn.Sequential(
                nn.Linear(2 * self.hidden_dim + self.modality_embed_dim + 8, self.view_router_hidden),
                nn.GELU(),
                nn.Linear(self.view_router_hidden, self.num_experts),
            )
            nn.init.zeros_(self.view_router[-1].weight)
            nn.init.zeros_(self.view_router[-1].bias)
            self.gamma_s_raw = nn.Parameter(torch.full((2,), _inverse_softplus(0.08)))
            self.gamma_c_raw = nn.Parameter(torch.full((2,), _inverse_softplus(0.05)))

        self._view_intervention = {"same_off": False, "cross_source_off": False}
        self._last_view_diagnostics: dict[str, Any] = {}
        self._initial_view_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.named_parameters()
            if name.startswith(("view_", "gamma_"))
        }

    @property
    def gamma_s(self) -> torch.Tensor:
        return F.softplus(self.gamma_s_raw)

    @property
    def gamma_c(self) -> torch.Tensor:
        return F.softplus(self.gamma_c_raw)

    def set_view_interventions(self, *, same_off: bool = False, cross_source_off: bool = False) -> None:
        self._view_intervention = {"same_off": bool(same_off), "cross_source_off": bool(cross_source_off)}

    def load_semantic_cache(self, path: str, **kwargs) -> dict[str, Any]:
        metadata = super().load_semantic_cache(path, **kwargs)
        cache = load_semantic_candidate_cache_v7b(
            path,
            expected_dataset=kwargs.get("expected_dataset"),
            expected_feature_fingerprint=kwargs.get("expected_feature_fingerprint"),
            expected_candidate_fingerprint=kwargs.get("expected_candidate_fingerprint"),
            expected_v7b_fingerprint=kwargs.get("expected_v7b_fingerprint"),
        )
        operators = build_validated_view_operators(
            cache["unique_edge_index"], cache["source_flags"], self.num_nodes
        )
        # Match the established V2 operator semantics byte-for-byte after
        # validating canonical IDs, provenance flags, and duplicate handling.
        for modality in _MODALITIES:
            setattr(self, f"semantic_{modality}_edge_index_v2", operators[modality]["edge_index"])
            setattr(self, f"semantic_{modality}_weight_v2", operators[modality]["weight"])
        return metadata

    def _view_mask(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if edge_index.numel():
            mask[edge_index[1]] = True
        return mask

    def _cross_source_context(
        self, target_modality: str, transformed_sender: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return S_other Phi_(other->target) H_other (never S_target)."""
        source_modality = "visual" if target_modality == "text" else "text"
        edge_index = getattr(self, f"semantic_{source_modality}_edge_index_v2")
        weight = getattr(self, f"semantic_{source_modality}_weight_v2")
        edge_index = edge_index.to(device=transformed_sender.device)
        weight = weight.to(device=transformed_sender.device, dtype=transformed_sender.dtype)
        context = chunked_weighted_spmm(
            transformed_sender, edge_index, weight, self.view_edge_chunk_size
        )
        return context, edge_index, source_modality

    def _shared_expert_response(
        self,
        profile: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        physical_residual: bool,
    ) -> torch.Tensor:
        """Checkpoint one mixture function; never materialize views×experts×N×D."""
        residual_scale = self.expert_feature_scale if physical_residual else 1.0

        def apply_bank(value: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
            mixture = torch.zeros_like(value[0]) if physical_residual else torch.zeros_like(value)
            for expert_id, expert in enumerate(self.experts):
                transformed = expert(value[expert_id]) if physical_residual else expert(value)
                if physical_residual:
                    candidate = value[expert_id] + residual_scale * transformed
                    mixture = mixture + routes[:, expert_id : expert_id + 1] * candidate
                else:
                    mixture = mixture + routes[:, expert_id : expert_id + 1] * transformed
            return mixture

        if self.training and torch.is_grad_enabled():
            return checkpoint(apply_bank, profile, route_weights, use_reentrant=False)
        return apply_bank(profile, route_weights)

    def _semantic_view(
        self,
        *,
        target_modality: str,
        modality_index: int,
        view_index: int,
        prior: torch.Tensor,
        raw_context: torch.Tensor,
        scale: torch.Tensor,
        edge_index: torch.Tensor,
        enabled: bool,
        collect_details: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not enabled:
            zero = torch.zeros_like(prior)
            return zero, {"enabled": False, "active_nodes": 0, "input_rms": 0.0, "expert_rms": 0.0, "gamma": 0.0, "weighted_rms": 0.0, "entropy": 0.0, "selection_share": [0.0] * self.num_experts}
        mask = self._view_mask(edge_index, prior.size(0), prior.device)
        calibrated = raw_context / scale
        # Mask at both ends: invalid support rows stay exactly zero even if a
        # router or expert develops a bias in a future implementation.
        calibrated = calibrated * mask.to(calibrated.dtype).unsqueeze(-1)
        modality_embed = self.modality_embedding.weight[modality_index].expand(prior.size(0), -1)
        role = self.view_type_embedding.weight[view_index].expand(prior.size(0), -1)
        router_input = torch.cat((prior, calibrated, modality_embed, role), dim=-1)
        logits = self.view_router(router_input)
        route = self._top2(logits, self.top_k)
        response = calibrated + self._shared_expert_response(
            calibrated, route["route_weights"], physical_residual=False
        )
        response = response * mask.to(response.dtype).unsqueeze(-1)
        gamma = (self.gamma_s if view_index == 0 else self.gamma_c)[modality_index]
        contribution = gamma * response
        contribution = contribution * mask.to(contribution.dtype).unsqueeze(-1)
        if not collect_details:
            return contribution, {"enabled": True, "weighted_rms": 0.0}
        if bool(mask.any()):
            probs = route["dense_probs"][mask]
            entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()).detach().cpu())
            selection_share = route["selected_mask"][mask].float().mean(0).detach().cpu().tolist()
        else:
            entropy, selection_share = 0.0, [0.0] * self.num_experts
        expert_input_rms: list[float] = []
        expert_output_rms: list[float] = []
        expert_correction_rms: list[float] = []
        if not self.training:
            with torch.no_grad():
                for expert in self.experts:
                    transformed = expert(calibrated.detach())
                    expert_input_rms.append(self._scalar_rms(calibrated, mask))
                    expert_output_rms.append(self._scalar_rms(transformed, mask))
                    expert_correction_rms.append(self._scalar_rms(transformed, mask))
        return contribution, {
            "enabled": True,
            "active_nodes": int(mask.sum().item()),
            "input_rms": self._scalar_rms(raw_context, mask),
            "calibrated_rms": self._scalar_rms(calibrated, mask),
            "expert_rms": self._scalar_rms(response - calibrated, mask),
            "response_rms": self._scalar_rms(response, mask),
            "gamma": float(gamma.detach().cpu()),
            "weighted_rms": self._scalar_rms(contribution, mask),
            "entropy": entropy,
            "selection_share": selection_share,
            "expert_input_rms_by_expert": expert_input_rms,
            "expert_output_rms_by_expert": expert_output_rms,
            "expert_nonlinear_correction_rms_by_expert": expert_correction_rms,
            "top2_indices": route["top_indices"].detach(),
            "mask": mask.detach(),
        }

    def _forward_view_cge(self, x: torch.Tensor, edge_index: torch.Tensor, *, return_details: bool):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        edge_index_directed = torch.stack((src, dst), dim=0)
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        outputs: dict[str, torch.Tensor] = {}
        balances: list[torch.Tensor] = []
        all_diagnostics: dict[str, Any] = {}

        for modality_index, modality in enumerate(_MODALITIES):
            other = "visual" if modality == "text" else "text"
            prior = priors[modality]
            state = prior
            physical_states = []
            for _ in range(self.physical_hops):
                state = self._propagate(state, src, dst, physical)
                state = state * active.to(state.dtype).unsqueeze(-1)
                physical_states.append(state)
            raw_u1 = physical_states[0]
            normalized_states = torch.stack([self._active_rms_norm(item, active) for item in physical_states])
            if bool(active.any()):
                scale = torch.sqrt(raw_u1[active].float().square().mean(0) + self.eps).to(prior.dtype)
            else:
                scale = prior.new_zeros((self.hidden_dim,))
            scale = scale.detach().clamp_min(self.view_scale_floor)

            phi = self.cross_projections[
                "visual_to_text" if modality == "text" else "text_to_visual"
            ](priors[other])
            physical_cross = chunked_weighted_spmm(phi, edge_index_directed, physical, self.propagation_chunk_size)
            physical_cross_hat = physical_cross / scale
            physical_profiles = torch.einsum("kl,lnd->knd", alpha, normalized_states)
            physical_profiles = physical_profiles + self.cross_coefficients[:, None, None] * physical_cross_hat[None]
            base_route = self._physical_route(modality_index, prior, active)
            zero_delta = torch.zeros_like(raw_u1)
            correction_logits = self._node_correction(prior, raw_u1, zero_delta, physical_cross_hat, modality_index)
            base_logits = self.selection_head(base_route["encoded"]).expand(prior.size(0), -1)
            physical_route = self._top2(base_logits + correction_logits, self.top_k)
            physical_route["strength"] = base_route["strength"]
            importance, physical_share, physical_balance = self._route_balance(physical_route, active)
            physical_route["balance"] = physical_balance
            balances.append(physical_balance)
            physical_mixture = self._shared_expert_response(
                physical_profiles, physical_route["route_weights"], physical_residual=True
            )
            physical_contribution = physical_route["strength"].unsqueeze(-1) * physical_mixture
            physical_contribution = physical_contribution * active.to(prior.dtype).unsqueeze(-1)

            same_edges = getattr(self, f"semantic_{modality}_edge_index_v2").to(prior.device)
            same_weight = getattr(self, f"semantic_{modality}_weight_v2").to(device=prior.device, dtype=prior.dtype)
            same_raw = chunked_weighted_spmm(prior, same_edges, same_weight, self.view_edge_chunk_size)
            same_enabled = self.study_condition in {"A1_separate_intra", "A2_cross_source"} and not self._view_intervention["same_off"]
            cross_enabled = self.study_condition == "A2_cross_source" and not self._view_intervention["cross_source_off"]
            cross_source = "visual" if modality == "text" else "text"
            cross_edges = getattr(self, f"semantic_{cross_source}_edge_index_v2").to(prior.device)
            if cross_enabled:
                cross_raw, cross_edges, cross_source = self._cross_source_context(modality, phi)
            else:
                cross_raw = torch.zeros_like(prior)
            same_contribution, same_diag = self._semantic_view(
                target_modality=modality, modality_index=modality_index, view_index=0,
                prior=prior, raw_context=same_raw, scale=scale, edge_index=same_edges, enabled=same_enabled,
                collect_details=return_details,
            )
            cross_contribution, cross_diag = self._semantic_view(
                target_modality=modality, modality_index=modality_index, view_index=1,
                prior=prior, raw_context=cross_raw, scale=scale, edge_index=cross_edges, enabled=cross_enabled,
                collect_details=return_details,
            )
            outputs[modality] = prior + physical_contribution + same_contribution + cross_contribution
            if return_details:
                physical_expert_input_rms: list[float] = []
                physical_expert_output_rms: list[float] = []
                physical_expert_correction_rms: list[float] = []
                if not self.training:
                    with torch.no_grad():
                        for expert_id, expert in enumerate(self.experts):
                            profile = physical_profiles[expert_id].detach()
                            transformed = expert(profile)
                            physical_expert_input_rms.append(self._scalar_rms(profile, active))
                            physical_expert_output_rms.append(self._scalar_rms(profile + self.expert_feature_scale * transformed, active))
                            physical_expert_correction_rms.append(self._scalar_rms(self.expert_feature_scale * transformed, active))
                all_diagnostics[modality] = {
                    "physical_active_nodes": int(active.sum().item()),
                    "physical_first_hop_rms": self._scalar_rms(raw_u1, active),
                    "physical_cross_rms": self._scalar_rms(physical_cross, active),
                    "physical_cross_calibrated_rms": self._scalar_rms(physical_cross_hat, active),
                    "physical_profile_rms": self._scalar_rms(physical_profiles),
                    "physical_expert_correction_rms": self._scalar_rms(physical_mixture - torch.einsum("nk,knd->nd", physical_route["route_weights"], physical_profiles)),
                    "physical_strength": float(physical_route["strength"].detach().mean().cpu()),
                    "physical_contribution_rms": self._scalar_rms(physical_contribution, active),
                    "physical_route_entropy": float((-(physical_route["dense_probs"][active] * physical_route["dense_probs"][active].clamp_min(1e-12).log()).sum(-1).mean()).detach().cpu()) if bool(active.any()) else 0.0,
                    "physical_selection_share": physical_share.detach().cpu().tolist(),
                    "physical_expert_input_rms_by_expert": physical_expert_input_rms,
                    "physical_expert_output_rms_by_expert": physical_expert_output_rms,
                    "physical_expert_nonlinear_correction_rms_by_expert": physical_expert_correction_rms,
                    "same_view": same_diag,
                    "cross_view": {**cross_diag, "source_modality": cross_source},
                    "semantic_to_physical_ratio": (same_diag["weighted_rms"] + cross_diag["weighted_rms"]) / max(self._scalar_rms(physical_contribution, active), 1e-12),
                }
                same_top2 = same_diag.get("top2_indices")
                cross_top2 = cross_diag.get("top2_indices")
                same_mask = same_diag.get("mask")
                cross_mask = cross_diag.get("mask")
                if same_top2 is not None and cross_top2 is not None and same_mask is not None and cross_mask is not None:
                    shared_nodes = same_mask & cross_mask
                    if bool(shared_nodes.any()):
                        same_sets = torch.sort(same_top2[shared_nodes], dim=-1).values
                        cross_sets = torch.sort(cross_top2[shared_nodes], dim=-1).values
                        all_diagnostics[modality]["same_cross_top2_exact_match_fraction"] = float((same_sets == cross_sets).all(-1).float().mean().cpu())
                    else:
                        all_diagnostics[modality]["same_cross_top2_exact_match_fraction"] = None
            del physical_states, normalized_states, same_raw, cross_raw, physical_profiles

        balance = 0.5 * (balances[0] + balances[1])
        aux_loss = self.balance_weight * balance
        fused_input = torch.cat((outputs["text"], outputs["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused_input)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused_input))))
        )
        self._last_view_diagnostics = all_diagnostics if return_details else {}
        info: dict[str, Any] = {
            "variant": self.study_condition,
            "condition": self.study_condition,
            "active_nodes": active.detach(),
            "input_self_loops_removed": loops,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "interventions": dict(self._view_intervention),
        }
        if return_details:
            # Drop large route tensors from persistent info; keep compact metrics.
            details = copy.deepcopy(all_diagnostics)
            for block in details.values():
                for view_name in ("same_view", "cross_view"):
                    for key in ("top2_indices", "mask"):
                        block[view_name].pop(key, None)
            info["details"] = {
                "modalities": details,
                "fusion_input_rms": self._scalar_rms(fused_input),
                "fused_output_rms": self._scalar_rms(z),
            }
        return z, None, None, aux_loss, info

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if edge_index is None:
            raise ValueError("View-CGE V1 supports full-graph NC only and requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("View-CGE V1 semantic caches use global node IDs; sampled subgraphs are unsupported")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("View-CGE V1 requires a validated V7B.1 semantic cache")
        if self.study_condition == "A0_legacy":
            # Temporarily expose the exact parent M0 discriminator. This branch
            # performs no new forward arithmetic and restores the public label.
            condition = self.study_condition
            self.study_condition = "M0"
            try:
                result = super().forward(x, edge_index, return_details=return_details)
            finally:
                self.study_condition = condition
            result[4]["variant"] = condition
            result[4]["condition"] = condition
            return result
        return self._forward_view_cge(x, edge_index, return_details=return_details)

    @staticmethod
    def _scalar_rms(value: torch.Tensor, active: torch.Tensor | None = None) -> float:
        if active is not None:
            value = value[active]
        if value.numel() == 0:
            return 0.0
        return float(torch.sqrt(value.detach().float().square().mean()).cpu())

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        diagnostics = super().capture_gradient_diagnostics()
        parameters = dict(self.named_parameters())
        for name in ("gamma_s_raw", "gamma_c_raw", "view_router.0.weight", "view_router.2.weight", "view_router.2.bias"):
            parameter = parameters[name]
            grad = parameter.grad
            norm = float(grad.detach().float().norm().cpu()) if grad is not None else 0.0
            diagnostics[name] = {
                "gradient_present": grad is not None,
                "gradient_finite": bool(grad is None or torch.isfinite(grad).all()),
                "gradient_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        return diagnostics


__all__ = ["Model", "CONDITIONS", "build_validated_view_operators"]
