"""MR-CGE V1.3-A with a shared expert-level Early Cross coefficient vector.

The frozen V1.3-A Collaborative Physical expert input is corrected immediately
before the shared Bottleneck and LayerNorm. Both the ordinary stacked path and
the low-memory checkpointed path use the same attached-autograd correction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mrcge_v13 import Model as FrozenV13Model
from src.models.mrcge_v13 import _run_bottleneck


EARLY_CROSS_DECOUPLING_VARIANT = "expert_decoupled_early_cross"


class Model(FrozenV13Model):
    """Frozen V1.3-A plus four expert-level, modality-shared Cross coefficients."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        mc = cfg.model
        variant = str(mc.get("early_cross_decoupling_variant", ""))
        enabled = bool(mc.get("early_cross_decoupling_enabled", False))
        if variant != EARLY_CROSS_DECOUPLING_VARIANT or not enabled:
            raise ValueError("Expert-Decoupled Early Cross requires its explicit variant and enabled flag")
        super().__init__(cfg, data_info)
        if self.condition != "M_repaired" or self.v13_variant != "joint_view":
            raise ValueError("Expert-Decoupled Early Cross is defined only on M_repaired / joint_view")
        if not self.condition.startswith("M_") or not hasattr(self, "filter_bank"):
            raise ValueError("Expert-Decoupled Early Cross requires the shared four-expert bank")

        # Copy this instance's initialized Physical alpha[:,1]. No random draw,
        # data lookup, checkpoint lookup, or manual/dataset-specific value.
        with torch.no_grad():
            initial_cross = self.filter_bank.coefficients(
                self.physical_hops, self.filter_bank.theta.dtype
            )[:, 1].detach().clone()
        self.cross_coefficients = nn.Parameter(initial_cross)
        self.early_cross_decoupling_variant = variant

    def _corrected_filtered_inputs(
        self,
        filtered_old: torch.Tensor,
        coefficients: torch.Tensor,
        cross_context: torch.Tensor,
        modality_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return corrected expert inputs and attached delta coefficients."""
        if int(modality_id) not in (0, 1):
            raise ValueError("modality_id must be Text (0) or Visual (1)")
        if coefficients.size(1) < 2:
            raise ValueError("Physical Cross correction requires the first-order coefficient")
        # Do not detach alpha_current here. Its gradient cancels the original
        # alpha[:,1]*beta*C path still present in filtered_old.
        correction_coefficients = self.beta[int(modality_id)] * (
            self.cross_coefficients.to(coefficients.dtype) - coefficients[:, 1]
        )
        corrected = filtered_old + correction_coefficients[:, None, None] * cross_context[None, :, :]
        return corrected, correction_coefficients

    @staticmethod
    def _per_expert_rms(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        selected = values[:, support, :]
        if selected.numel() == 0:
            return values.new_zeros((values.size(0),), dtype=torch.float32)
        return selected.detach().float().square().mean(dim=(1, 2)).sqrt()

    def _cross_input_diagnostics(
        self,
        filtered_old: torch.Tensor,
        filtered_new: torch.Tensor,
        coefficients: torch.Tensor,
        correction_coefficients: torch.Tensor,
        cross_context: torch.Tensor,
        support: torch.Tensor,
        modality_id: int,
    ) -> dict[str, Any]:
        """Summarize Cross input magnitudes without stacking extra N x H tensors."""
        alpha1 = coefficients[:, 1].detach()
        beta = self.beta[modality_id].detach()
        b = self.cross_coefficients.detach().to(coefficients.dtype)
        old_add_rms, new_add_rms, private_rms, correction_rms, actual_delta_rms = [], [], [], [], []
        for expert_id in range(self.num_experts):
            old_cross = beta * alpha1[expert_id] * cross_context
            new_cross = beta * b[expert_id] * cross_context
            correction = correction_coefficients[expert_id].detach() * cross_context
            private_filtered = filtered_old[expert_id].detach() - old_cross
            actual_delta = filtered_new[expert_id].detach() - private_filtered
            old_add_rms.append(self._rms(old_cross, support))
            new_add_rms.append(self._rms(new_cross, support))
            private_rms.append(self._rms(private_filtered, support))
            correction_rms.append(self._rms(correction, support))
            actual_delta_rms.append(self._rms(actual_delta, support))
        old_values = torch.stack(old_add_rms)
        new_values = torch.stack(new_add_rms)
        private_values = torch.stack(private_rms)
        correction_values = torch.stack(correction_rms)
        actual_values = torch.stack(actual_delta_rms)
        return {
            "alpha_physical_order1": alpha1.detach().cpu().tolist(),
            "cross_coefficients": b.detach().cpu().tolist(),
            "cross_coefficient_minus_alpha1": (b - alpha1).detach().cpu().tolist(),
            "abs_cross_coefficient_minus_alpha1": (b - alpha1).abs().detach().cpu().tolist(),
            "beta": float(beta.cpu()),
            "old_beta_alpha_cross_coefficient": (beta * alpha1).detach().cpu().tolist(),
            "new_beta_b_cross_coefficient": (beta * b).detach().cpu().tolist(),
            "cross_calibrated_rms": float(self._rms(cross_context, support).cpu()),
            "private_filtered_input_rms_by_expert": private_values.cpu().tolist(),
            "old_cross_addition_rms_by_expert": old_values.cpu().tolist(),
            "cross_addition_rms_by_expert": new_values.cpu().tolist(),
            "cross_addition_over_private_rms_by_expert": (
                new_values / private_values.clamp_min(self.eps)
            ).cpu().tolist(),
            "collaborative_minus_private_input_rms_by_expert": actual_values.cpu().tolist(),
            "difference_correction_rms_by_expert": correction_values.cpu().tolist(),
        }

    def _filter(
        self,
        basis: list[torch.Tensor],
        support: torch.Tensor,
        current: torch.Tensor,
        relation_id: int,
        modality_id: int,
        cross_context: torch.Tensor,
        *,
        collect: bool,
    ):
        # Semantic and non-expert routes remain exactly on the frozen path.
        if relation_id != 0 or not self.condition.startswith("M_"):
            return super()._filter(basis, support, current, relation_id, modality_id,
                                   cross_context, collect=collect)

        # Preserve the V1.3-A low-memory checkpoint/RNG boundary. It dynamically
        # dispatches to this class's _m_filter_impl for the expert-input delta.
        if not (collect or not (self.training and torch.is_grad_enabled())):
            return super()._filter(basis, support, current, relation_id, modality_id,
                                   cross_context, collect=collect)

        max_order = len(basis) - 1
        if max_order != self.physical_hops:
            raise RuntimeError(f"Physical filter order drift: expected {self.physical_hops}, got {max_order}")
        coefficients = self.filter_bank.coefficients(max_order, basis[0].dtype)
        stacked = torch.stack(basis[:max_order + 1], dim=0)
        filtered_old = torch.einsum("kl,lnd->knd", coefficients[:, :max_order + 1], stacked)
        filtered, correction_coefficients = self._corrected_filtered_inputs(
            filtered_old, coefficients, cross_context, modality_id
        )
        mask = support.to(filtered.dtype).unsqueeze(-1)
        outputs, residuals = [], []
        for expert_id, expert in enumerate(self.filter_bank.experts):
            residual = self.eta * _run_bottleneck(expert, filtered[expert_id])
            value = self.filter_bank.norms[expert_id](filtered[expert_id] + residual)
            outputs.append(value * mask)
            residuals.append(residual * mask)
        outputs = torch.stack(outputs)
        residuals = torch.stack(residuals)

        first = basis[1] if len(basis) > 1 else basis[0]
        logits = self.router(current, first, cross_context, modality_id, relation_id)
        dense = torch.softmax(logits.float(), dim=-1)
        top_logits, top_ids = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits.float(), dim=-1).to(outputs.dtype)
        route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights.float()).to(outputs.dtype)
        selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
        if bool(support.any()):
            importance = dense[support].mean(dim=0)
            selected_share = selected[support].float().mean(dim=0) / float(self.top_k)
            balance = self.num_experts * (importance * selected_share).sum()
        else:
            importance = dense.sum(dim=0) * 0.0
            selected_share = importance
            balance = dense.sum() * 0.0
        response = (outputs * route_weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
        route = {"logits": logits, "dense": dense, "weights": route_weights,
                 "selected": selected, "importance": importance, "share": selected_share,
                 "balance": balance}
        detail = self._filter_summary({
            "G": filtered, "E": outputs, "residual": residuals,
            "coefficients": coefficients.detach().cpu().tolist(),
        }, support) if collect else {}
        if collect:
            detail["expert_decoupled_cross"] = self._cross_input_diagnostics(
                filtered_old, filtered, coefficients, correction_coefficients,
                cross_context, support, modality_id,
            )
        return response, route, balance, detail

    def _m_filter_impl(
        self, basis: list[torch.Tensor], support: torch.Tensor, current: torch.Tensor,
        cross_context: torch.Tensor, modality_id: int, relation_id: int, *, collect: bool,
    ):
        # The shared Semantic route is untouched. Physical inputs use the same
        # signed-L1 alpha and attached-autograd delta as the stacked route.
        if relation_id != 0 or not self.condition.startswith("M_"):
            return super()._m_filter_impl(basis, support, current, cross_context,
                                          modality_id, relation_id, collect=collect)

        max_order = len(basis) - 1
        if max_order != self.physical_hops:
            raise RuntimeError(f"Physical filter order drift: expected {self.physical_hops}, got {max_order}")
        coefficients = self.filter_bank.coefficients(max_order, basis[0].dtype)
        first = basis[1]
        n = current.size(0)
        mod = self.router.modality_embedding.weight[modality_id].expand(n, -1)
        rel = self.router.relation_embedding.weight[relation_id].expand(n, -1)
        router_parts = (current, first, first - current, mod, rel, cross_context)
        first_layer = self.router.net[0]
        offset, pre_activation = 0, None
        for part in router_parts:
            width = part.size(-1)
            contribution = F.linear(part, first_layer.weight[:, offset:offset + width], bias=None)
            pre_activation = contribution if pre_activation is None else pre_activation + contribution
            offset += width
        if offset != first_layer.in_features:
            raise RuntimeError("router input component widths do not match its frozen linear layer")
        if first_layer.bias is not None:
            pre_activation = pre_activation + first_layer.bias
        router_hidden = F.gelu(pre_activation)

        mask = support.to(basis[0].dtype).unsqueeze(-1)
        correction_coefficients = self.beta[modality_id] * (
            self.cross_coefficients.to(coefficients.dtype) - coefficients[:, 1]
        )
        outputs, filtered_values, residuals = [], [], []
        for expert_id, expert in enumerate(self.filter_bank.experts):
            filtered = None
            for order in range(max_order + 1):
                term = coefficients[expert_id, order] * basis[order]
                filtered = term if filtered is None else filtered + term
            # Keep alpha attached here: this cancels its gradient on the Cross
            # coefficient already present in basis[1]. Do not detach this term.
            filtered = filtered + correction_coefficients[expert_id] * cross_context
            residual = self.eta * expert(filtered)
            value = self.filter_bank.norms[expert_id](filtered + residual) * mask
            outputs.append(value)
            if collect:
                filtered_values.append(filtered)
                residuals.append(residual * mask)

        logits = self.router.net[3](self.router.net[2](router_hidden))
        dense = torch.softmax(logits.float(), dim=-1)
        top_logits, top_ids = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits.float(), dim=-1)
        route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights)
        selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
        response = None
        for expert_id, value in enumerate(outputs):
            weighted = value * route_weights[:, expert_id:expert_id + 1].to(value.dtype)
            response = weighted if response is None else response + weighted

        if bool(support.any()):
            importance = dense[support].mean(dim=0)
            selected_share = selected[support].float().mean(dim=0) / float(self.top_k)
            balance = self.num_experts * (importance * selected_share).sum()
            dense_active = dense[support].clamp_min(1.0e-12)
            selected_active = route_weights[support].clamp_min(1.0e-12)
            entropy = -(dense_active * dense_active.log()).sum(dim=-1).mean()
            selected_entropy = -(selected_active * selected_active.log()).sum(dim=-1).mean()
        else:
            importance = dense.sum(dim=0) * 0.0
            selected_share = importance.detach()
            balance = dense.sum() * 0.0
            entropy = dense.sum() * 0.0
            selected_entropy = dense.sum() * 0.0
        route_summary = {"importance": importance.detach(), "share": selected_share.detach(),
                         "dense_entropy": entropy.detach(), "selected_entropy": selected_entropy.detach(),
                         "active_nodes": support.sum().detach()}
        if collect:
            route = {"logits": logits, "dense": dense, "weights": route_weights,
                     "selected": selected, **route_summary, "balance": balance}
            detail = self._filter_summary({
                "G": torch.stack(filtered_values), "E": torch.stack(outputs),
                "residual": torch.stack(residuals),
                "coefficients": coefficients.detach().cpu().tolist(),
            }, support)
            filtered_old = torch.stack([
                sum((coefficients[k, order] * basis[order] for order in range(max_order + 1)),
                    torch.zeros_like(basis[0])) for k in range(self.num_experts)
            ])
            filtered_new = torch.stack(filtered_values)
            detail["expert_decoupled_cross"] = self._cross_input_diagnostics(
                filtered_old, filtered_new, coefficients, correction_coefficients,
                cross_context, support, modality_id,
            )
        else:
            route = route_summary
            detail = {}
        return response, balance, route_summary["importance"], route_summary["share"], \
            route_summary["dense_entropy"], route_summary["selected_entropy"], \
            {"route": route, "detail": detail}

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        result = super().capture_gradient_diagnostics()
        gradient = self.cross_coefficients.grad
        norm = float(gradient.detach().float().norm().cpu()) if gradient is not None else 0.0
        result["cross_coefficients"] = {
            "gradient_present": gradient is not None,
            "gradient_finite": bool(gradient is None or torch.isfinite(gradient).all()),
            "gradient_norm": norm,
            "gradient_nonzero": norm > 0.0,
            "per_expert_gradient": gradient.detach().float().cpu().tolist() if gradient is not None else [],
        }
        return result


__all__ = ["Model", "EARLY_CROSS_DECOUPLING_VARIANT"]
