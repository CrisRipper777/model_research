"""MR-CGE V1.3-A with four modality/relation-level Static-Strength logits.

The V1.3-A forward is locally mirrored so the modality id is explicit at each
Private and Collaborative strength call. All other model modules and graph
paths are inherited from the frozen implementation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mrcge_v13 import Model as FrozenV13Model
from src.models.mrcge_v13 import _MODALITIES, capture_rng_state, replay_rng_state


STATIC_STRENGTH_VARIANT = "static_strength"


class Model(FrozenV13Model):
    """V1.3-A mechanism with global per-modality Physical/Semantic scales."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        mc = cfg.model
        variant = str(mc.get("static_strength_variant", ""))
        enabled = bool(mc.get("static_strength_enabled", False))
        init = float(mc.get("static_strength_init", 0.0))
        if variant != STATIC_STRENGTH_VARIANT or not enabled or init != 0.0:
            raise ValueError("Static-Strength requires its explicit variant and enabled flag")
        super().__init__(cfg, data_info)
        if self.condition != "M_repaired" or self.v13_variant != "joint_view":
            raise ValueError("Static-Strength is defined only on M_repaired / joint_view")
        # Construct the inherited Joint View module inside its isolated RNG context
        # so all later V1.3 parameters and the external NC head keep their exact
        # seeded initialization. It is not part of the Static model parameter set.
        if not hasattr(self, "joint_view"):
            raise RuntimeError("frozen V1.3-A Joint View initialization path was not created")
        del self.joint_view
        self.static_strength_variant = variant
        self.static_logits = nn.Parameter(torch.zeros((2, 2), dtype=self.rho_raw.dtype))

    def static_strengths(self, modality_id: int) -> torch.Tensor:
        if int(modality_id) not in (0, 1):
            raise ValueError("modality_id must be 0 (Text) or 1 (Visual)")
        return 2.0 * torch.sigmoid(self.static_logits[int(modality_id)])

    def static_strengths_all(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.static_logits)

    def view_strength(self, modality_id: int, h: torch.Tensor,
                      d_physical: torch.Tensor, d_semantic: torch.Tensor) -> torch.Tensor:
        if d_physical.shape != h.shape or d_semantic.shape != h.shape:
            raise ValueError("Static-Strength inputs must match the modality hidden-state shape")
        value = self.static_strengths(modality_id).to(dtype=h.dtype)
        return value.unsqueeze(0).expand(h.size(0), -1)

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
                strengths = self.view_strength(modality_id, h[m], d_physical, d_semantic)
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
        mechanism_forward: dict[str, float | int | torch.Tensor] = {
            "rho_text": self._diagnostic_scalar(self.rho[0], return_details),
            "rho_visual": self._diagnostic_scalar(self.rho[1], return_details),
            "gamma_text": self._diagnostic_scalar(self.gamma[0], return_details),
            "gamma_visual": self._diagnostic_scalar(self.gamma[1], return_details),
            "sender_private_ratio": self.sender_private_ratio,
        }
        for modality_id, modality in enumerate(_MODALITIES):
            strengths = private_strengths[modality]
            mechanism_forward[f"static_sP_{modality}"] = self._diagnostic_scalar(
                strengths[0, 0], return_details)
            mechanism_forward[f"static_sS_{modality}"] = self._diagnostic_scalar(
                strengths[0, 1], return_details)
            mechanism_forward[f"effective_rho_sP_{modality}"] = self._diagnostic_scalar(
                self.rho[modality_id] * strengths[0, 0], return_details)
            mechanism_forward[f"effective_gamma_sS_{modality}"] = self._diagnostic_scalar(
                self.gamma[modality_id] * strengths[0, 1], return_details)
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
            collab_basis = list(bases[m]["physical"])
            collab_basis[1] = (collab_basis[1] + self.beta[modality_id] * cross_hat) * physical_support.to(h[m].dtype).unsqueeze(-1)
            with replay_rng_state(paired_rng_states[m], x.device):
                r_collab, route, balance, detail = self._filter(
                    collab_basis, physical_support, h[m], 0, modality_id, cross_hat, collect=return_details
                )
            nonzero_context = cross_hat.abs().sum(dim=-1) > 0
            r_collab = self._zero_context_fallback(responses[m]["physical"], r_collab, cross_hat)
            if self.v13_variant == "joint_view":
                d_physical_collab = self.rho[modality_id] * r_collab
                d_semantic_private = self.gamma[modality_id] * responses[m]["semantic"]
                strengths_collab = self.view_strength(modality_id, h[m], d_physical_collab, d_semantic_private)
                collaborative_strengths[m] = strengths_collab
                collaborative_physical_contribution = strengths_collab[:, 0:1] * d_physical_collab
                physical_scale = self.rho[modality_id] * strengths_collab[:, 0:1]
                collaborative[m] = private[m] + physical_scale * (
                    r_collab - responses[m]["physical"]
                )
            else:
                collaborative_physical_contribution = self.rho[modality_id] * r_collab
                collaborative[m] = private[m] + self.rho[modality_id] * (r_collab - responses[m]["physical"])
            collaborative_contributions[m] = {"physical": collaborative_physical_contribution}
            balances.append(balance)
            collab_routes[m] = route
            filter_details[f"collaborative_{m}_physical"] = detail
            neighbor_rms = self._rms(bases[m]["physical"][1], physical_support).clamp_min(self.eps)
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
            mechanism_forward[f"early_cross_response_delta_rms_{m}"] = self._diagnostic_scalar(
                self._rms(r_collab - responses[m]["physical"], physical_support), return_details)
            mechanism_forward[f"effective_early_cross_delta_rms_{m}"] = self._diagnostic_scalar(
                self._rms(physical_scale * (r_collab - responses[m]["physical"]), physical_support), return_details)
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
        else:
            self._last_val_v13_diagnostics = dict(mechanism_forward)

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
            }
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
                "static_strength_variant": self.static_strength_variant,
                "static_logits": self.static_logits.detach().cpu().tolist(),
                "static_strengths": {m: self.static_strengths(i).detach().cpu().tolist()
                                     for i, m in enumerate(_MODALITIES)},
                "effective_physical_strength": (self.rho[:, None] * self.static_strengths_all()[:, 0:1]).detach().cpu().tolist(),
                "effective_semantic_strength": (self.gamma[:, None] * self.static_strengths_all()[:, 1:2]).detach().cpu().tolist(),
                "v13_mechanism": dict(mechanism_forward),
            }
        self._last_diagnostics = info.get("details", {})
        return z, None, None, aux_loss, info

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        names = [
            "projectors.text.linear1.weight", "projectors.visual.linear1.weight",
            "cross_projections.visual_to_text.0.weight", "cross_projections.text_to_visual.0.weight",
            "query_projections.text.weight", "query_projections.visual.weight",
            "key_projections.text.weight", "key_projections.visual.weight",
            "source_selector.2.weight", "channel_gate.2.weight", "fusion_linear1.weight",
            "unimodal_heads.text.weight", "unimodal_heads.visual.weight",
            "rho_raw", "gamma_raw", "beta_raw", "cross_lambda_raw", "tau_raw",
            "static_logits", "filter_bank.theta", "router.net.3.weight",
        ]
        for expert_id in range(self.num_experts):
            names.extend([
                f"filter_bank.experts.{expert_id}.net.0.weight",
                f"filter_bank.experts.{expert_id}.net.3.weight",
            ])
        params = dict(self.named_parameters())
        result = {}
        for name in names:
            param = params[name]
            grad = param.grad
            norm = float(grad.detach().float().norm().cpu()) if grad is not None else 0.0
            result[name] = {
                "gradient_present": grad is not None,
                "gradient_finite": bool(grad is None or torch.isfinite(grad).all()),
                "gradient_norm": norm,
                "gradient_nonzero": norm > 0,
            }
        self._last_gradient_diagnostics = result
        return result


__all__ = ["Model", "STATIC_STRENGTH_VARIANT"]
