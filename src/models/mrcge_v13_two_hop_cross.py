"""Controlled V1.3-A two-hop Early Cross variants.

The frozen parent's private, semantic, late, fusion, and router paths are kept
intact. This subclass adds a calibrated second-hop candidate only to the
collaborative physical expert inputs, before their Bottleneck/LayerNorm.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.mrcge_v13 import (
    Model as FrozenV13Model,
    _run_bottleneck,
    capture_rng_state,
    replay_rng_state,
)
from src.models.psce_mag_v7a import _MODALITIES, chunked_weighted_spmm


VARIANTS = ("raw", "complementary")
PROJECTION_EPS = 1.0e-8
CALIBRATION_EPS = 1.0e-8
NEAR_ZERO_RELATIVE = 1.0e-3
MAX_CALIBRATION_SCALE = 4.0


class Model(FrozenV13Model):
    """V1.3-A plus one zero-initialized scalar per receiver modality."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        variant = str(cfg.model.get("two_hop_variant", ""))
        if variant not in VARIANTS:
            raise ValueError(f"two_hop_variant must be one of {VARIANTS}")
        if (str(cfg.model.get("condition", "M_repaired")) != "M_repaired"
                or str(cfg.model.get("v13_variant", "joint_view")) != "joint_view"
                or float(cfg.model.get("sender_private_ratio", 0.0)) != 0.0):
            raise ValueError("Two-Hop Cross is frozen to M_repaired / joint_view / sender_private_ratio=0")
        super().__init__(cfg, data_info)
        if not hasattr(self, "filter_bank"):
            raise RuntimeError("Two-Hop Cross requires the frozen shared M_repaired expert bank")
        self.two_hop_variant = variant
        # Deterministic initialization consumes no random numbers, preserving
        # common V1.3-A and external NC classifier initialization exactly.
        self.two_hop_raw = nn.Parameter(torch.zeros(2, dtype=self.cross_lambda_raw.dtype))
        self._last_two_hop_train: dict[str, Any] = {}
        self._last_two_hop_val: dict[str, Any] = {}
        self._last_two_hop_state: dict[str, dict[str, torch.Tensor]] = {}

    @property
    def two_hop_lambda(self) -> torch.Tensor:
        return torch.tanh(self.two_hop_raw)

    @staticmethod
    def _masked_inner(left: torch.Tensor, right: torch.Tensor, support: torch.Tensor,
                      block_size: int = 8192) -> torch.Tensor:
        """FP64 blocked Frobenius inner product over supported nodes."""
        total = torch.zeros((), dtype=torch.float64, device=left.device)
        for start in range(0, left.size(0), block_size):
            end = min(start + block_size, left.size(0))
            mask = support[start:end]
            if not bool(mask.any()):
                continue
            a = left[start:end][mask].to(torch.float64)
            b = right[start:end][mask].to(torch.float64)
            total = total + (a * b).sum()
        return total

    @classmethod
    def _masked_cosine(cls, left: torch.Tensor, right: torch.Tensor,
                       support: torch.Tensor) -> tuple[torch.Tensor | None, str]:
        if not bool(support.any()):
            return None, "empty_support"
        with torch.no_grad():
            ll = cls._masked_inner(left.detach(), left.detach(), support)
            rr = cls._masked_inner(right.detach(), right.detach(), support)
            if float(ll) <= 0.0 or float(rr) <= 0.0:
                return None, "zero_norm_operand"
            lr = cls._masked_inner(left.detach(), right.detach(), support)
            value = (lr / (ll * rr).sqrt()).clamp(-1.0, 1.0)
        return value.to(dtype=torch.float32), "ok"

    def _calibrate_candidate(self, candidate: torch.Tensor, reference: torch.Tensor,
                             support: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Shared, frozen RMS calibration for Raw and Complementary candidates."""
        ref_rms = self._rms(reference, support)
        cand_rms = self._rms(candidate, support)
        ref_value, cand_value = float(ref_rms.cpu()), float(cand_rms.cpu())
        empty = not bool(support.any())
        degenerate = empty or ref_value <= 1.0e-8 or cand_value <= NEAR_ZERO_RELATIVE * ref_value
        if degenerate:
            scale = cand_rms.new_zeros(()).detach()
            calibrated = torch.zeros_like(candidate)
            capped = False
        else:
            proposed = (ref_rms / (cand_rms + CALIBRATION_EPS)).detach()
            capped = float(proposed.cpu()) > MAX_CALIBRATION_SCALE
            scale = proposed.clamp(max=MAX_CALIBRATION_SCALE).detach()
            calibrated = candidate * scale.to(candidate.dtype)
            calibrated = calibrated * support.to(candidate.dtype).unsqueeze(-1)
        return calibrated, {
            "reference_rms": ref_rms.detach(), "candidate_rms": cand_rms.detach(),
            "scale": scale, "near_zero_degenerate": int(degenerate),
            "scale_capped": int(capped),
            "calibrated_rms": self._rms(calibrated, support).detach(),
            "support_nodes": int(support.sum().detach().cpu()),
        }

    def _two_hop_candidate(
        self, modality: str, cross_context: torch.Tensor,
        sender_private: torch.Tensor, projection: nn.Module,
        edge_index: torch.Tensor, edge_weight: torch.Tensor, support: torch.Tensor,
        *, collect: bool,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        """Build P X1 and optionally remove its stop-gradient SameNode direction."""
        x2 = chunked_weighted_spmm(cross_context, edge_index, edge_weight, self.physical_chunk_size)
        x2 = x2 * support.to(x2.dtype).unsqueeze(-1)
        x2_rms = self._rms(x2, support).detach()
        need_reference = self.two_hop_variant == "complementary" or collect
        reference = None
        raw_cos = None
        raw_cos_status = "not_computed"
        projection_coefficient = None
        residual_rms = None
        residual_cos = None
        residual_cos_status = "not_applicable"
        candidate = x2
        if need_reference:
            # This is exactly the Late SameNode candidate Phi(U_private_other).
            # Stop-grad applies only to this analytic reference branch.
            with torch.no_grad():
                reference = projection(sender_private.detach()).detach()
            raw_cos, raw_cos_status = self._masked_cosine(x2, reference, support)
            if self.two_hop_variant == "complementary":
                with torch.no_grad():
                    numerator = self._masked_inner(x2.detach(), reference, support)
                    denominator = self._masked_inner(reference, reference, support) + PROJECTION_EPS
                    coefficient = (numerator / denominator).to(dtype=x2.dtype).detach()
                candidate = (x2 - coefficient * reference)
                candidate = candidate * support.to(candidate.dtype).unsqueeze(-1)
                projection_coefficient = coefficient
                residual_rms = self._rms(candidate, support).detach()
                residual_cos, residual_cos_status = self._masked_cosine(candidate, reference, support)

        calibrated, calibration = self._calibrate_candidate(candidate, cross_context, support)
        metrics: dict[str, Any] = {
            "variant": self.two_hop_variant,
            "x2_rms": x2_rms,
            "same_node_candidate_rms": self._rms(reference, support).detach() if reference is not None else None,
            "projection_coefficient": projection_coefficient.detach() if projection_coefficient is not None else None,
            "raw_x2_same_node_cosine": raw_cos.detach() if raw_cos is not None else None,
            "raw_x2_same_node_cosine_status": raw_cos_status,
            "complementary_residual_rms": residual_rms,
            "complementary_residual_same_node_cosine": residual_cos.detach() if residual_cos is not None else None,
            "complementary_residual_same_node_cosine_status": residual_cos_status,
            **calibration,
        }
        tensors = {"x2": x2, "candidate": candidate, "calibrated": calibrated}
        if reference is not None:
            tensors["same_node_reference"] = reference
        if projection_coefficient is not None:
            tensors["projection_coefficient"] = projection_coefficient
        return calibrated, metrics, tensors

    def _filter_with_two_hop(
        self, basis: list[torch.Tensor], support: torch.Tensor, current: torch.Tensor,
        modality_id: int, cross_context: torch.Tensor, candidate: torch.Tensor,
        addition_coefficients: torch.Tensor, *, collect: bool,
    ):
        """V1.3 physical expert/router call with an explicit pre-bottleneck addend.

        `candidate` and `addition_coefficients` are passed as explicit inputs to
        the non-reentrant checkpoint path. Router inputs stay on the original
        `basis[1]` / Early-Cross tensors and never see this addition.
        """
        max_order = len(basis) - 1
        if max_order != self.physical_hops:
            raise RuntimeError(f"Physical filter order drift: expected {self.physical_hops}, got {max_order}")

        def run(values, *, collect_detail: bool):
            local_basis = list(values[:max_order + 1])
            local_support, local_current, local_cross, local_candidate, local_add = values[max_order + 1:]
            return self._m_filter_two_hop_impl(
                local_basis, local_support, local_current, local_cross,
                local_candidate, local_add, modality_id, collect=collect_detail,
            )

        args = (*basis, support, current, cross_context, candidate, addition_coefficients)
        if self.training and torch.is_grad_enabled() and not collect:
            packed = checkpoint(lambda *values: run(values, collect_detail=False)[:6],
                                *args, use_reentrant=False, preserve_rng_state=True)
            response, balance, importance, selected_share, entropy, selected_entropy = packed
            route = {"importance": importance, "share": selected_share,
                     "dense_entropy": entropy, "selected_entropy": selected_entropy,
                     "active_nodes": support.sum().detach()}
            return response, route, balance, {}
        response, balance, importance, selected_share, entropy, selected_entropy, extra = run(
            args, collect_detail=collect
        )
        return response, extra["route"], balance, extra["detail"]

    def _m_filter_two_hop_impl(
        self, basis: list[torch.Tensor], support: torch.Tensor, current: torch.Tensor,
        cross_context: torch.Tensor, candidate: torch.Tensor, addition_coefficients: torch.Tensor,
        modality_id: int, *, collect: bool,
    ):
        """Low-memory V1.3 shared-expert execution with explicit Cross addend."""
        max_order = len(basis) - 1
        coefficients = self.filter_bank.coefficients(max_order, basis[0].dtype)
        first = basis[1] if len(basis) > 1 else basis[0]
        n = current.size(0)
        mod = self.router.modality_embedding.weight[modality_id].expand(n, -1)
        rel = self.router.relation_embedding.weight[0].expand(n, -1)
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
        outputs, filtered_values, old_filtered_values, residuals = [], [], [], []
        addition_rms, addition_ratio = [], []
        for expert_id, expert in enumerate(self.filter_bank.experts):
            filtered_old = None
            for order in range(max_order + 1):
                term = coefficients[expert_id, order] * basis[order]
                filtered_old = term if filtered_old is None else filtered_old + term
            addition = addition_coefficients[expert_id] * candidate
            filtered = filtered_old + addition
            residual = self.eta * expert(filtered)
            value = self.filter_bank.norms[expert_id](filtered + residual) * mask
            outputs.append(value)
            if collect:
                filtered_values.append(filtered)
                old_filtered_values.append(filtered_old)
                residuals.append(residual * mask)
                add_rms = self._rms(addition, support)
                old_rms = self._rms(filtered_old, support)
                addition_rms.append(add_rms)
                addition_ratio.append(add_rms / old_rms.clamp_min(self.eps))

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
            detail["two_hop_injection"] = {
                "candidate_rms": float(self._rms(candidate, support).cpu()),
                "addition_coefficients": addition_coefficients.detach().float().cpu().tolist(),
                "expert_addition_rms": torch.stack(addition_rms).detach().cpu().tolist(),
                "expert_addition_over_old_input_rms": torch.stack(addition_ratio).detach().cpu().tolist(),
                "old_expert_input_rms": [float(self._rms(v, support).cpu()) for v in old_filtered_values],
            }
        else:
            route = route_summary
            detail = {}
        return response, balance, route_summary["importance"], route_summary["share"], \
            route_summary["dense_entropy"], route_summary["selected_entropy"], \
            {"route": route, "detail": detail}

    def collect_epoch_diagnostics(self) -> dict[str, float | int]:
        result = super().collect_epoch_diagnostics()
        for scope, values in (("train", self._last_two_hop_train), ("val", self._last_two_hop_val)):
            for modality, metrics in values.items():
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = float(value.detach().cpu())
                    if isinstance(value, (bool, int, float)):
                        result[f"{scope}_two_hop_{key}_{modality}"] = value
        result["two_hop_variant_raw"] = int(self.two_hop_variant == "raw")
        return result

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        result = super().capture_gradient_diagnostics()
        grad = self.two_hop_raw.grad
        norm = float(grad.detach().float().norm().cpu()) if grad is not None else 0.0
        result["two_hop_raw"] = {
            "gradient_present": grad is not None,
            "gradient_finite": bool(grad is None or torch.isfinite(grad).all()),
            "gradient_norm": norm, "gradient_nonzero": norm > 0.0,
            "per_modality_gradient": grad.detach().float().cpu().tolist() if grad is not None else [],
        }
        return result


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if edge_index is None:
            raise ValueError("MR-CGE V1.2 requires a global physical edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [nodes,{self.input_dim}]")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("MR-CGE V1.2 requires the cached global node order")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("MR-CGE V1.2 requires a validated V7B.1 semantic cache")
        raw = {"text": x[:, :self.text_dim], "visual": x[:, self.text_dim:self.input_dim]}
        h = {m: self.projectors[m](raw[m]) for m in _MODALITIES}
        physical_edges, physical_weight, physical_support, loop_count = self._physical_operator(
            edge_index, x.size(0), h["text"].dtype
        )
        bases: dict[str, dict[str, list[torch.Tensor]]] = {m: {} for m in _MODALITIES}
        supports: dict[str, dict[str, torch.Tensor]] = {m: {} for m in _MODALITIES}
        for m in _MODALITIES:
            bases[m]["physical"] = self._basis(
                h[m], physical_edges, physical_weight, physical_support,
                self.physical_hops, self.physical_chunk_size,
            )
            sem_edges = getattr(self, f"semantic_{m}_edge_index").to(x.device)
            sem_weight = getattr(self, f"semantic_{m}_weight").to(device=x.device, dtype=h[m].dtype)
            sem_support = self._support(sem_edges, x.size(0), x.device)
            bases[m]["semantic"] = self._basis(
                h[m], sem_edges, sem_weight, sem_support,
                self.semantic_hops, self.semantic_chunk_size,
            )
            supports[m]["physical"] = physical_support
            supports[m]["semantic"] = sem_support

        private: dict[str, torch.Tensor] = {}
        private_strengths: dict[str, torch.Tensor] = {}
        private_contributions: dict[str, dict[str, torch.Tensor]] = {}
        responses: dict[str, dict[str, torch.Tensor]] = {m: {} for m in _MODALITIES}
        route_details: dict[str, Any] = {}
        filter_details: dict[str, Any] = {}
        balances: list[torch.Tensor] = []
        paired_rng_states: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        zeros = {m: torch.zeros_like(h[m]) for m in _MODALITIES}
        for modality_id, m in enumerate(_MODALITIES):
            for relation_id, relation in enumerate(("physical", "semantic")):
                if relation == "physical":
                    paired_rng_states[m] = capture_rng_state(x.device)
                response, route, balance, detail = self._filter(
                    bases[m][relation], supports[m][relation], h[m], relation_id, modality_id,
                    zeros[m], collect=return_details
                )
                responses[m][relation] = response
                balances.append(balance)
                view_name = f"private_{m}_{relation}"
                route_details[view_name] = route
                filter_details[view_name] = detail
            d_physical = self.rho[modality_id] * responses[m]["physical"]
            d_semantic = self.gamma[modality_id] * responses[m]["semantic"]
            if self.v13_variant == "joint_view":
                strengths = self.joint_view(h[m], d_physical, d_semantic)
                private_strengths[m] = strengths
                physical_contribution = strengths[:, 0:1] * d_physical
                semantic_contribution = strengths[:, 1:2] * d_semantic
            else:
                physical_contribution = d_physical
                semantic_contribution = d_semantic
            private_contributions[m] = {
                "physical": physical_contribution, "semantic": semantic_contribution,
            }
            private[m] = h[m] + physical_contribution + semantic_contribution

        collaborative: dict[str, torch.Tensor] = {}
        collaborative_strengths: dict[str, torch.Tensor] = {}
        collaborative_contributions: dict[str, dict[str, torch.Tensor]] = {}
        early_details: dict[str, Any] = {}
        cross_contexts: dict[str, torch.Tensor] = {}
        two_hop_details: dict[str, Any] = {}
        two_hop_epoch: dict[str, dict[str, Any]] = {}
        two_hop_state: dict[str, dict[str, torch.Tensor]] = {}
        mechanism_forward: dict[str, float | int | torch.Tensor] = {
            "rho_text": self._diagnostic_scalar(self.rho[0], return_details),
            "rho_visual": self._diagnostic_scalar(self.rho[1], return_details),
            "gamma_text": self._diagnostic_scalar(self.gamma[0], return_details),
            "gamma_visual": self._diagnostic_scalar(self.gamma[1], return_details),
            "sender_private_ratio": self.sender_private_ratio,
        }
        collab_routes: dict[str, Any] = {}
        for modality_id, m in enumerate(_MODALITIES):
            other = "visual" if m == "text" else "text"
            projection_name = "visual_to_text" if m == "text" else "text_to_visual"
            sender_state = h[other]
            if self.v13_variant == "sender_private" and self.sender_private_ratio > 0.0:
                sender_state = ((1.0 - self.sender_private_ratio) * h[other]
                                + self.sender_private_ratio * private[other])
            raw_cross, cross_hat, calibration_scale = self._early_cross(
                m, modality_id, sender_state, self.cross_projections[projection_name],
                physical_edges, physical_weight, physical_support, bases[m]["physical"][1]
            )
            cross_contexts[m] = cross_hat
            z_two_hop, twohop_metrics, twohop_tensors = self._two_hop_candidate(
                m, cross_hat, private[other], self.cross_projections[projection_name],
                physical_edges, physical_weight, physical_support, collect=return_details,
            )
            live_coefficients = self.filter_bank.coefficients(self.physical_hops, h[m].dtype)
            lambda_value = self.two_hop_lambda[modality_id]
            addition_coefficients = self.beta[modality_id] * live_coefficients[:, 1] * lambda_value
            twohop_metrics["lambda"] = lambda_value.detach()
            twohop_metrics["beta"] = self.beta[modality_id].detach()
            twohop_metrics["alpha_order1"] = live_coefficients[:, 1].detach()
            two_hop_epoch[m] = {
                "lambda": lambda_value.detach(),
                "x2_rms": twohop_metrics["x2_rms"],
                "candidate_rms": twohop_metrics["candidate_rms"],
                "scale": twohop_metrics["scale"],
                "near_zero_degenerate": twohop_metrics["near_zero_degenerate"],
                "scale_capped": twohop_metrics["scale_capped"],
            }
            two_hop_state[m] = twohop_tensors
            collab_basis = list(bases[m]["physical"])
            collab_basis[1] = (collab_basis[1] + self.beta[modality_id] * cross_hat) * physical_support.to(h[m].dtype).unsqueeze(-1)
            with replay_rng_state(paired_rng_states[m], x.device):
                r_collab, route, balance, detail = self._filter_with_two_hop(
                    collab_basis, physical_support, h[m], modality_id, cross_hat,
                    z_two_hop, addition_coefficients, collect=return_details,
                )
            nonzero_context = cross_hat.abs().sum(dim=-1) > 0
            r_collab = self._zero_context_fallback(responses[m]["physical"], r_collab, cross_hat)
            if self.v13_variant == "joint_view":
                d_physical_collab = self.rho[modality_id] * r_collab
                d_semantic_private = self.gamma[modality_id] * responses[m]["semantic"]
                strengths_collab = self.joint_view(h[m], d_physical_collab, d_semantic_private)
                collaborative_strengths[m] = strengths_collab
                collaborative_physical_contribution = strengths_collab[:, 0:1] * d_physical_collab
                collaborative[m] = private[m] + self.rho[modality_id] * (
                    strengths_collab[:, 0:1] * r_collab
                    - private_strengths[m][:, 0:1] * responses[m]["physical"]
                )
            else:
                collaborative_physical_contribution = self.rho[modality_id] * r_collab
                collaborative[m] = private[m] + self.rho[modality_id] * (r_collab - responses[m]["physical"])
            collaborative_contributions[m] = {"physical": collaborative_physical_contribution}
            balances.append(balance)
            collab_routes[m] = route
            filter_details[f"collaborative_{m}_physical"] = detail
            neighbor_rms = self._rms(bases[m]["physical"][1], physical_support).clamp_min(self.eps)
            mechanism_forward[f"two_hop_lambda_{m}"] = self._diagnostic_scalar(lambda_value, return_details)
            for key in ("x2_rms", "same_node_candidate_rms", "projection_coefficient",
                        "raw_x2_same_node_cosine", "complementary_residual_rms",
                        "complementary_residual_same_node_cosine", "reference_rms", "candidate_rms",
                        "scale", "near_zero_degenerate", "scale_capped", "calibrated_rms"):
                value = twohop_metrics.get(key)
                if value is not None:
                    mechanism_forward[f"two_hop_{key}_{m}"] = (
                        self._diagnostic_scalar(value, return_details)
                        if isinstance(value, torch.Tensor) else value
                    )
            mechanism_forward[f"early_cross_raw_rms_{m}"] = self._diagnostic_scalar(
                self._rms(raw_cross, physical_support), return_details)
            mechanism_forward[f"early_cross_calibrated_rms_{m}"] = self._diagnostic_scalar(
                self._rms(cross_hat, physical_support), return_details)
            mechanism_forward[f"early_cross_to_neighbor_rms_{m}"] = self._diagnostic_scalar(
                self._rms(cross_hat, physical_support) / neighbor_rms, return_details)
            mechanism_forward[f"physical_private_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(self.rho[modality_id] * responses[m]["physical"], physical_support), return_details)
            mechanism_forward[f"semantic_private_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(self.gamma[modality_id] * responses[m]["semantic"], supports[m]["semantic"]), return_details)
            mechanism_forward[f"physical_collaborative_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(self.rho[modality_id] * r_collab, physical_support), return_details)
            mechanism_forward[f"weighted_physical_private_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(private_contributions[m]["physical"], physical_support), return_details)
            mechanism_forward[f"weighted_semantic_private_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(private_contributions[m]["semantic"], supports[m]["semantic"]), return_details)
            mechanism_forward[f"weighted_physical_collaborative_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(collaborative_physical_contribution, physical_support), return_details)
            if self.v13_variant == "joint_view":
                support_p = physical_support
                support_s = supports[m]["semantic"]
                for scope, strength_values in (("private", private_strengths[m]),
                                               ("collaborative", collaborative_strengths[m])):
                    for index, relation, support in ((0, "sP", support_p), (1, "sS", support_s)):
                        stats = self._strength_stats(strength_values[:, index], support)
                        for field, value in stats.items():
                            mechanism_forward[f"{scope}_{relation}_{field}_{m}"] = self._diagnostic_scalar(
                                value, return_details)
                if bool(physical_support.any()):
                    change = (collaborative_strengths[m][:, 0] - private_strengths[m][:, 0])[physical_support]
                    mechanism_forward[f"collaborative_vs_private_sP_abs_mean_{m}"] = self._diagnostic_scalar(
                        change.abs().mean(), return_details)
                    mechanism_forward[f"collaborative_vs_private_sP_signed_mean_{m}"] = self._diagnostic_scalar(
                        change.mean(), return_details)
                else:
                    zero = h[m].new_zeros(())
                    mechanism_forward[f"collaborative_vs_private_sP_abs_mean_{m}"] = self._diagnostic_scalar(
                        zero, return_details)
                    mechanism_forward[f"collaborative_vs_private_sP_signed_mean_{m}"] = self._diagnostic_scalar(
                        zero, return_details)
            if return_details:
                def plain_scalar(value):
                    if isinstance(value, torch.Tensor):
                        cpu_value = value.detach().cpu()
                        return float(cpu_value) if cpu_value.numel() == 1 else cpu_value.tolist()
                    return value
                two_hop_details[m] = {key: plain_scalar(value) for key, value in twohop_metrics.items()}
                two_hop_details[m]["expert_injection"] = detail.get("two_hop_injection", {})
                early_details[m] = {
                    "raw_rms": float(self._rms(raw_cross, physical_support).cpu()),
                    "calibrated_rms": float(self._rms(cross_hat, physical_support).cpu()),
                    "scale": float(calibration_scale.detach().cpu()),
                    "beta": float(self.beta[modality_id].detach().cpu()),
                    "nonzero_nodes": int(nonzero_context.sum().detach().cpu()),
                    "private_response_rms": float(self._rms(responses[m]["physical"], physical_support).cpu()),
                    "collaborative_response_rms": float(self._rms(r_collab, physical_support).cpu()),
                    "sender_state_rms": float(self._rms(sender_state).cpu()),
                    "sender_private_ratio": self.sender_private_ratio if self.v13_variant == "sender_private" else 0.0,
                    "early_cross_to_neighbor_rms": mechanism_forward[f"early_cross_to_neighbor_rms_{m}"],
                }

        final: dict[str, torch.Tensor] = {}
        late_details: dict[str, Any] = {}
        for target_id, target in enumerate(_MODALITIES):
            source = "visual" if target == "text" else "text"
            final[target], late_details[target] = self._late(
                target, target_id, source, h[source], private[source], collaborative[target], collect=return_details
            )
            late_ratio = late_details[target]["late_delta_ratio"]
            mechanism_forward[f"late_residual_ratio_{target}"] = (
                self._diagnostic_scalar(late_ratio, return_details)
                if isinstance(late_ratio, torch.Tensor) else late_ratio
            )

        if self.training:
            self._last_train_v13_diagnostics = dict(mechanism_forward)
            self._last_two_hop_train = two_hop_epoch
        else:
            self._last_val_v13_diagnostics = dict(mechanism_forward)
            self._last_two_hop_val = two_hop_epoch

        if self._capture_state_audit:
            self._last_state_audit = {
                "private": private,
                "collaborative": collaborative,
                "responses": responses,
                "private_strengths": private_strengths,
                "collaborative_strengths": collaborative_strengths,
                "private_contributions": private_contributions,
                "collaborative_contributions": collaborative_contributions,
                "sender_states": {},
                "early_cross": cross_contexts,
                "two_hop": two_hop_state,
            }
            self._last_two_hop_state = two_hop_state
            # Sender tensors and calibrated Cross retain autograd for gradient-path tests.
            self._last_state_audit["sender_states"] = {
                "text": ((1.0 - self.sender_private_ratio) * h["text"] + self.sender_private_ratio * private["text"])
                if self.v13_variant == "sender_private" and self.sender_private_ratio > 0.0 else h["text"],
                "visual": ((1.0 - self.sender_private_ratio) * h["visual"] + self.sender_private_ratio * private["visual"])
                if self.v13_variant == "sender_private" and self.sender_private_ratio > 0.0 else h["visual"],
            }

        balance = torch.stack(balances).mean() if self.condition.startswith("M_") else x.new_zeros(())
        aux_loss = self.balance_weight * balance
        if self.training:
            self._last_train_balance_raw_tensor = balance.detach()
        else:
            self._last_balance_raw_tensor = balance.detach()
        route_snapshots: dict[str, Any] = {}
        for view_name, route in route_details.items():
            modality, relation = view_name.rsplit("_", 2)[1:]
            if route is not None:
                route_snapshots[view_name] = {
                    "route": self._compact_route_state(route, supports[modality][relation]),
                    "support": supports[modality][relation].detach(),
                }
        for modality, route in collab_routes.items():
            if route is not None:
                route_snapshots[f"collaborative_{modality}_physical"] = {
                    "route": self._compact_route_state(route, physical_support),
                    "support": physical_support.detach(),
                }
        self._last_route_snapshot = route_snapshots
        fused = torch.cat((final["text"], final["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused) + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
        )
        info: dict[str, Any] = {
            "condition": self.condition,
            "variant": self.v13_variant,
            "v13_variant": self.v13_variant,
            "active_nodes": physical_support.detach(),
            "input_self_loops_removed": loop_count,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "unimodal_logits": {m: self.unimodal_heads[m](private[m]) for m in _MODALITIES},
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "test_evaluation": False,
        }
        if return_details:
            filter_summaries = filter_details
            route_summaries = {}
            for key, value in route_details.items():
                modality, relation = key.rsplit("_", 2)[1:]
                route_summaries[key] = self._route_summary(value, supports[modality][relation])
            collaborative_route_summaries = {
                m: self._route_summary(collab_routes[m], physical_support) for m in _MODALITIES
            }
            route_changes = {}
            if self.condition.startswith("M_"):
                for m in _MODALITIES:
                    before, after = route_details[f"private_{m}_physical"], collab_routes[m]
                    route_changes[m] = {
                        "dense_importance_delta": (after["importance"] - before["importance"]).detach().cpu().tolist(),
                        "top2_selection_share_delta": (after["share"] - before["share"]).detach().cpu().tolist(),
                        "node_top2_changed_fraction": float((after["selected"][physical_support] != before["selected"][physical_support]).any(dim=-1).float().mean().cpu()) if bool(physical_support.any()) else 0.0,
                    }
            info["details"] = {
                "physical_support_nodes": int(physical_support.sum().detach().cpu()),
                "semantic_support_nodes": {m: int(supports[m]["semantic"].sum().detach().cpu()) for m in _MODALITIES},
                "private_rms": {m: float(self._rms(private[m]).cpu()) for m in _MODALITIES},
                "physical_response_rms": {m: float(self._rms(responses[m]["physical"], physical_support).cpu()) for m in _MODALITIES},
                "semantic_response_rms": {m: float(self._rms(responses[m]["semantic"], supports[m]["semantic"]).cpu()) for m in _MODALITIES},
                "early_physical_cross": early_details,
                "two_hop_cross": two_hop_details,
                "late": late_details,
                "routes": route_summaries,
                "collaborative_routes": collaborative_route_summaries,
                "private_collaborative_route_changes": route_changes,
                "filters": filter_summaries,
                "theta_actual": self.filter_bank.theta.detach().cpu().tolist() if self.condition.startswith("M_") else self.single_filter.theta.detach().cpu().tolist(),
                "rho": self.rho.detach().cpu().tolist(),
                "gamma": self.gamma.detach().cpu().tolist(),
                "beta": self.beta.detach().cpu().tolist(),
                "balance_loss": float(balance.detach().cpu()),
                "aux_loss": float(aux_loss.detach().cpu()),
                "shared_expert_objects": len({id(e) for e in self.filter_bank.experts}) if self.condition.startswith("M_") else 0,
                "actual_experts_executed_per_filter_call": 4 if self.condition.startswith("M_") else 1,
                "fusion_input_rms": float(self._rms(fused).cpu()),
                "fused_output_rms": float(self._rms(z).cpu()),
                "v13_mechanism": dict(mechanism_forward),
            }
        self._last_diagnostics = info.get("details", {})
        return z, None, None, aux_loss, info


__all__ = ["Model", "VARIANTS"]
