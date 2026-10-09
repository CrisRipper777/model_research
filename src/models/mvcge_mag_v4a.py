from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from src.models.mvcge_mag_v3c import Model as V3CModel


_MODALITIES = ("text", "visual")
_VARIANTS = (
    "R0_raw",
    "R1_global_residual",
    "R2_expert_residual",
    "R3_adaptive_residual",
)


class Model(V3CModel):
    """Raw-anchored bounded role-contrast residual screen.

    The complete V3C F0/V3A C0 raw expert path remains the backbone. Fixed
    Supportive and Discrepant messages are decomposed at each Raw hop and
    normalized by that hop's Raw denominator. Their difference supplies a
    bounded residual to the shared experts; it never replaces the Raw path or
    enters an expert MLP.
    """

    no_weight_decay_parameter_names = frozenset(
        {
            "context_mix_raw",
            "role_mix_raw",
            "residual_global_raw",
            "residual_expert_raw",
        }
    )

    def __init__(self, cfg, data_info: dict[str, Any]):
        variant = str(cfg.model.get("variant", "R3_adaptive_residual"))
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        beta_max = float(cfg.model.get("residual_beta_max", 0.25))
        if not math.isclose(beta_max, 0.25, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("V4A fixes residual_beta_max=0.25")

        # Always construct the V3C F0/V3A C0 module tree first. The inherited
        # fixed role parameters and these V4A tensors are constant initialized,
        # so all historical modules and the downstream classifier keep their
        # exact initialization RNG stream.
        base_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        base_cfg.variant = "F0_raw"
        super().__init__(SimpleNamespace(model=base_cfg), data_info)
        self.variant = variant
        self.residual_beta_max = beta_max

        hadamard = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.residual_global_raw = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        self.residual_expert_raw = nn.Parameter(torch.zeros((self.num_experts,), dtype=torch.float32))
        self.residual_gamma_raw = nn.Parameter(0.5 * hadamard)

        # V3C's legacy context/effective-profile/role-mix parameters stay
        # frozen. Only the named V4A residual controls vary by screen arm.
        self.context_mix_raw.requires_grad_(False)
        self.alpha_effective_raw.requires_grad_(False)
        self.role_mix_raw.requires_grad_(False)
        self.residual_global_raw.requires_grad_(variant == "R1_global_residual")
        self.residual_expert_raw.requires_grad_(
            variant in {"R2_expert_residual", "R3_adaptive_residual"}
        )
        self.residual_gamma_raw.requires_grad_(variant == "R3_adaptive_residual")

    @staticmethod
    def _normalized_gamma(raw: torch.Tensor, eps: float) -> torch.Tensor:
        return raw / (raw.norm(p=2, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _shared_residual_profile(
        alpha: torch.Tensor, role_contrast: torch.Tensor
    ) -> torch.Tensor:
        """Reuse Raw hop weights without allowing residual gradients into alpha."""
        return torch.einsum("mk,knd->mnd", alpha.detach(), role_contrast)

    @staticmethod
    def _adaptive_residual_profile(
        gamma: torch.Tensor, role_contrast: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("mk,knd->mnd", gamma, role_contrast)

    def _residual_betas(self) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self.residual_expert_raw.new_zeros(())
        if self.variant == "R1_global_residual":
            global_beta = self.residual_beta_max * torch.tanh(self.residual_global_raw[0])
            beta = global_beta.expand(self.num_experts)
        elif self.variant in {"R2_expert_residual", "R3_adaptive_residual"}:
            global_beta = zero
            beta = self.residual_beta_max * torch.tanh(self.residual_expert_raw)
        else:
            global_beta = zero
            beta = self.residual_expert_raw.new_zeros((self.num_experts,))
        return global_beta, beta

    def _decompose_raw_hops(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        raw_norm_weight: torch.Tensor,
        physical_active: torch.Tensor,
        role: dict[str, torch.Tensor],
        *,
        diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Split each message from the previous Raw state, with one Raw norm."""
        raw_states: list[torch.Tensor] = []
        support_messages: list[torch.Tensor] = []
        discrepant_messages: list[torch.Tensor] = []
        raw_normalized: list[torch.Tensor] = []
        support_normalized: list[torch.Tensor] = []
        discrepant_normalized: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        contrasts: list[torch.Tensor] = []
        state_corrections: list[torch.Tensor] = []
        normalized_corrections: list[torch.Tensor] = []
        previous_raw = prior
        active_f = physical_active.to(prior.dtype).unsqueeze(-1)

        def propagate_role(state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            if not diagnostics and self.training and torch.is_grad_enabled():
                # Recompute the sparse gather/reduction on backward instead of
                # retaining an edge-by-hidden activation for every role hop.
                return checkpoint(
                    lambda value: self._propagate(value, src, dst, weight),
                    state,
                    use_reentrant=False,
                )
            return self._propagate(state, src, dst, weight)

        for _ in range(self.trajectory_order):
            # Keep the baseline Raw propagation operation and order intact.
            raw_state = self._propagate(previous_raw, src, dst, raw_norm_weight)
            raw_state = raw_state * active_f
            support_message = propagate_role(previous_raw, role["support_norm_weight"]) * active_f
            discrepant_propagate = self._propagate if diagnostics else propagate_role
            if diagnostics:
                discrepant_message = discrepant_propagate(
                    previous_raw, src, dst, role["discrepant_norm_weight"]
                ) * active_f
            else:
                discrepant_message = discrepant_propagate(
                    previous_raw, role["discrepant_norm_weight"]
                ) * active_f
            # Independent float32 scatter reductions can differ by several
            # ulps even though the edge weights partition Raw exactly. Apply
            # only that roundoff residual to conserve the untouched Raw op.
            state_correction = raw_state - (support_message + discrepant_message)
            discrepant_message = discrepant_message + state_correction

            if bool(physical_active.any()):
                raw_selected = raw_state[physical_active]
                denominator = torch.sqrt(raw_selected.square().mean(dim=0) + self.eps)
            else:
                denominator = raw_state.new_full((raw_state.size(-1),), math.sqrt(self.eps))
            raw_normed = self._active_rms_norm(raw_state, physical_active)
            support_normed = torch.zeros_like(support_message)
            discrepant_normed = torch.zeros_like(support_message)
            if bool(physical_active.any()):
                support_normed[physical_active] = support_message[physical_active] / denominator
                if diagnostics:
                    discrepant_normed[physical_active] = discrepant_message[physical_active] / denominator
            normalized_correction = raw_normed - (support_normed + discrepant_normed)
            discrepant_normed = discrepant_normed + normalized_correction
            contrast = support_normed - discrepant_normed

            if diagnostics:
                raw_states.append(raw_state)
                support_messages.append(support_message)
                discrepant_messages.append(discrepant_message)
            raw_normalized.append(raw_normed)
            if diagnostics:
                support_normalized.append(support_normed)
                discrepant_normalized.append(discrepant_normed)
                denominators.append(denominator)
            contrasts.append(contrast)
            if diagnostics:
                state_corrections.append(state_correction)
                normalized_corrections.append(normalized_correction)
            previous_raw = raw_state

        raw_normalized_t = torch.stack(raw_normalized, dim=0)
        contrast_t = torch.stack(contrasts, dim=0)
        if not diagnostics:
            return {"raw_trajectory": raw_normalized_t, "role_contrast": contrast_t}
        raw_states_t = torch.stack(raw_states, dim=0)
        support_messages_t = torch.stack(support_messages, dim=0)
        discrepant_messages_t = torch.stack(discrepant_messages, dim=0)
        support_normalized_t = torch.stack(support_normalized, dim=0)
        discrepant_normalized_t = torch.stack(discrepant_normalized, dim=0)
        state_error = support_messages_t + discrepant_messages_t - raw_states_t
        normalized_error = support_normalized_t + discrepant_normalized_t - raw_normalized_t
        return {
            "raw_states": raw_states_t,
            "support_messages": support_messages_t,
            "discrepant_messages": discrepant_messages_t,
            "raw_trajectory": raw_normalized_t,
            "support_trajectory": support_normalized_t,
            "discrepant_trajectory": discrepant_normalized_t,
            "role_contrast": contrast_t,
            "raw_denominators": torch.stack(denominators, dim=0),
            "state_roundoff_correction_max_abs": torch.stack(state_corrections).abs().amax(),
            "normalized_roundoff_correction_max_abs": torch.stack(normalized_corrections).abs().amax(),
            "state_partition_max_abs_error": state_error.abs().amax(),
            "normalized_partition_max_abs_error": normalized_error.abs().amax(),
            "state_partition_relative_rms_error": (
                torch.sqrt(state_error.square().mean())
                / (torch.sqrt(raw_states_t.square().mean()) + self.eps)
            ),
            "normalized_partition_relative_rms_error": (
                torch.sqrt(normalized_error.square().mean())
                / (torch.sqrt(raw_normalized_t.square().mean()) + self.eps)
            ),
        }

    def _safety_calibrate(
        self,
        raw_profiles: torch.Tensor,
        residual_profiles: torch.Tensor,
        physical_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply detached, scalar, one-sided per-expert/modality downscaling."""
        scales: list[torch.Tensor] = []
        raw_rms_values: list[torch.Tensor] = []
        residual_rms_values: list[torch.Tensor] = []
        safe_profiles: list[torch.Tensor] = []
        for expert_id in range(self.num_experts):
            raw_selected = raw_profiles[expert_id, physical_active]
            residual_selected = residual_profiles[expert_id, physical_active]
            if raw_selected.numel():
                raw_rms = torch.sqrt(raw_selected.square().mean() + self.eps)
                residual_rms = torch.sqrt(residual_selected.square().mean() + self.eps)
            else:
                raw_rms = raw_profiles.new_tensor(math.sqrt(self.eps))
                residual_rms = residual_profiles.new_tensor(math.sqrt(self.eps))
            scale = torch.minimum(
                torch.ones_like(raw_rms),
                raw_rms.detach() / (residual_rms.detach() + self.eps),
            )
            scales.append(scale)
            raw_rms_values.append(raw_rms)
            residual_rms_values.append(residual_rms)
            safe_profiles.append(residual_profiles[expert_id] * scale)
        return (
            torch.stack(safe_profiles, dim=0),
            torch.stack(scales),
            torch.stack(raw_rms_values),
            torch.stack(residual_rms_values),
        )

    def _attach_r0_diagnostics(
        self,
        result,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
    ):
        z, aux1, aux2, aux_loss, info = result
        src, dst, raw_norm, active, _ = self._normalized_operator(
            edge_index, x.size(0), x.dtype
        )
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        gamma = self._normalized_gamma(self.residual_gamma_raw, self.eps)
        global_beta, beta = self._residual_betas()
        for modality in _MODALITIES:
            item = info["details"][modality]
            role = info["role_channels"][modality]
            decomposition = self._decompose_raw_hops(
                item["prior"], src, dst, raw_norm, active, role
            )
            shared_q = self._shared_residual_profile(alpha, decomposition["role_contrast"])
            adaptive_q = self._adaptive_residual_profile(gamma, decomposition["role_contrast"])
            q_safe, scales, raw_rms, q_rms = self._safety_calibrate(
                item["raw_profiles"], shared_q, active
            )
            item.update(decomposition)
            item.update(
                {
                    "role_contrast_basis": decomposition["role_contrast"],
                    "raw_expert_outputs": item["expert_outputs"],
                    "raw_mixture": item["mixture"],
                    "residual_mixture": torch.zeros_like(item["mixture"]),
                    "scaled_raw_correction": item["scaled_mixture"],
                    "scaled_residual_correction": torch.zeros_like(item["scaled_mixture"]),
                    "scaled_total_correction": item["scaled_mixture"],
                    "residual_shared_raw": shared_q,
                    "residual_adaptive_raw": adaptive_q,
                    "residual_raw": shared_q,
                    "residual_safe": q_safe,
                    "safety_scale": scales,
                    "raw_profile_rms": raw_rms,
                    "residual_profile_rms_before_safety": q_rms,
                    "global_beta": global_beta,
                    "expert_beta": beta,
                    "final_expert_outputs": item["expert_outputs"],
                }
            )
        info["variant"] = self.variant
        return z, aux1, aux2, aux_loss, info

    def _forward_residual(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool,
    ):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, raw_norm, physical_active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )

        roles: dict[str, dict[str, torch.Tensor]] = {}
        decompositions: dict[str, dict[str, torch.Tensor]] = {}
        router_states: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            roles[modality] = self._role_partition(inputs[modality], src, dst, raw_norm)
            decompositions[modality] = self._decompose_raw_hops(
                priors[modality], src, dst, raw_norm, physical_active, roles[modality],
                diagnostics=return_details,
            )
            roles[modality].pop("semantic_features", None)
            router_states[modality] = self.router_norms[modality](
                self.router_projectors[modality](priors[modality])
            )

        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        gamma = self._normalized_gamma(self.residual_gamma_raw, self.eps)
        global_beta, beta = self._residual_betas()
        outputs: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        router_info: dict[str, dict[str, torch.Tensor]] = {}
        modality_summaries: dict[str, dict[str, torch.Tensor]] = {}
        modality_balance: dict[str, torch.Tensor] = {}

        for modality_index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            decomposition = decompositions[modality]
            raw_basis = decomposition["raw_trajectory"]
            raw_profiles = torch.einsum("mk,knd->mnd", alpha, raw_basis)
            raw_transformed = torch.stack(
                [expert(raw_profiles[r]) for r, expert in enumerate(self.experts)], dim=0
            )
            raw_expert_outputs = raw_profiles + self.expert_feature_scale * raw_transformed

            shared_q = self._shared_residual_profile(alpha, decomposition["role_contrast"])
            adaptive_q = self._adaptive_residual_profile(gamma, decomposition["role_contrast"])
            if self.variant == "R3_adaptive_residual":
                residual_raw = adaptive_q
            else:
                residual_raw = shared_q
            residual_safe, safety_scale, raw_profile_rms, residual_rms = self._safety_calibrate(
                raw_profiles, residual_raw, physical_active
            )
            scaled_residual = beta.view(self.num_experts, 1, 1) * residual_safe
            expert_outputs = raw_expert_outputs + scaled_residual

            route = self._static_route(
                modality_index, router_states[modality], physical_active
            )
            raw_mixture = torch.einsum(
                "nm,mnd->nd", route["route_weights"], raw_expert_outputs
            )
            residual_mixture = torch.einsum(
                "nm,mnd->nd", route["route_weights"], scaled_residual
            )
            mixture = torch.einsum("nm,mnd->nd", route["route_weights"], expert_outputs)
            scaled_mixture = route["strength"].unsqueeze(-1) * mixture
            scaled_raw_correction = route["strength"].unsqueeze(-1) * raw_mixture
            scaled_residual_correction = route["strength"].unsqueeze(-1) * residual_mixture
            output = prior + scaled_mixture

            outputs[modality] = output
            router_info[modality] = route
            modality_balance[modality] = route["balance"]
            modality_summaries[modality] = {
                "strength_mean": route["strength"].detach().mean(),
                "selected_pair": route["top_indices"][0].detach(),
                "top2_weights": route["route_weights"][0].detach(),
            }
            if return_details:
                details[modality] = {
                    "prior": prior,
                    "raw_operator_weight": raw_norm,
                    "alpha": alpha,
                    "raw_trajectory": raw_basis,
                    "role_contrast_basis": decomposition["role_contrast"],
                    "raw_profiles": raw_profiles,
                    "raw_expert_outputs": raw_expert_outputs,
                    "expert_inputs": raw_profiles,
                    "expert_outputs": expert_outputs,
                    "final_expert_outputs": expert_outputs,
                    "residual_shared_raw": shared_q,
                    "residual_adaptive_raw": adaptive_q,
                    "residual_raw": residual_raw,
                    "residual_safe": residual_safe,
                    "scaled_residual": scaled_residual,
                    "safety_scale": safety_scale,
                    "raw_profile_rms": raw_profile_rms,
                    "residual_profile_rms_before_safety": residual_rms,
                    "global_beta": global_beta,
                    "expert_beta": beta,
                    "raw_mixture": raw_mixture,
                    "residual_mixture": residual_mixture,
                    "mixture": mixture,
                    "scaled_raw_correction": scaled_raw_correction,
                    "scaled_residual_correction": scaled_residual_correction,
                    "scaled_mixture": scaled_mixture,
                    "output": output,
                    "gamma_raw": self.residual_gamma_raw,
                    "gamma": gamma,
                    **decomposition,
                    **roles[modality],
                    **route,
                }

        balance_loss = 0.5 * (modality_balance["text"] + modality_balance["visual"])
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
            "modalities": modality_summaries,
        }
        if return_details:
            info["details"] = details
            info["router"] = router_info
            info["role_channels"] = roles
        return z, None, None, aux_loss, info

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        if self.variant == "R0_raw":
            # Preserve the historical V3C F0/V3A C0 forward exactly.
            saved_variant = self.variant
            self.variant = "F0_raw"
            try:
                result = super().forward(x, edge_index, return_details=return_details)
            finally:
                self.variant = saved_variant
            if return_details:
                return self._attach_r0_diagnostics(result, x, edge_index)
            result[4]["variant"] = self.variant
            return result
        return self._forward_residual(x, edge_index, return_details=return_details)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(
            x.to(device), None if edge_index is None else edge_index.to(device)
        )
        return z.detach().cpu()
