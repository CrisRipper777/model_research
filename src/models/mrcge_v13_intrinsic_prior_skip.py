"""V1.3-A Joint View with a single intrinsic-prior final-skip coefficient."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mrcge_v13 import (
    Model as FrozenV13Model,
    capture_rng_state,
    replay_rng_state,
)
from src.models.psce_mag_v7a import _MODALITIES


class Model(FrozenV13Model):
    """Frozen V1.3-A with one zero-initialized Intrinsic-Prior skip scalar."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        model_cfg = cfg.model
        if (str(model_cfg.get("condition", "M_repaired")) != "M_repaired"
                or str(model_cfg.get("v13_variant", "joint_view")) != "joint_view"
                or float(model_cfg.get("sender_private_ratio", 0.0)) != 0.0):
            raise ValueError(
                "Intrinsic-Prior Skip is frozen to M_repaired / joint_view / sender_private_ratio=0"
            )
        super().__init__(cfg, data_info)
        self.prior_skip_raw = nn.Parameter(torch.zeros((), dtype=self.fusion_skip.weight.dtype))
        self._last_intrinsic_prior_skip: dict[str, torch.Tensor] = {}

    @property
    def lambda_prior(self) -> torch.Tensor:
        return torch.tanh(self.prior_skip_raw)

    def collect_epoch_diagnostics(self) -> dict[str, float | int]:
        result = super().collect_epoch_diagnostics()
        result["intrinsic_prior_raw"] = float(self.prior_skip_raw.detach().cpu())
        result["intrinsic_prior_lambda"] = float(self.lambda_prior.detach().cpu())
        scalar_gradient = getattr(self, "_last_gradient_diagnostics", {}).get("prior_skip_raw", {})
        result["intrinsic_prior_gradient_norm"] = float(scalar_gradient.get("gradient_norm", 0.0))
        result["intrinsic_prior_gradient_nonzero"] = int(bool(scalar_gradient.get("gradient_nonzero", False)))
        return result

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        result = dict(super().capture_gradient_diagnostics())
        grad = self.prior_skip_raw.grad
        norm = float(grad.detach().float().norm().cpu()) if grad is not None else 0.0
        result["prior_skip_raw"] = {
            "gradient_present": grad is not None,
            "gradient_finite": bool(grad is None or torch.isfinite(grad).all()),
            "gradient_norm": norm,
            "gradient_nonzero": norm > 0.0,
        }
        self._last_gradient_diagnostics = result
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
        h0_fused = torch.cat((h["text"], h["visual"]), dim=-1)
        lambda_prior = self.lambda_prior.to(dtype=fused.dtype)
        skip_input = fused + lambda_prior * h0_fused
        mlp_output = self.fusion_linear2(
            self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
        )
        z = self.fusion_norm(self.fusion_skip(skip_input) + mlp_output)
        self._last_intrinsic_prior_skip = {
            "prior_skip_raw": self.prior_skip_raw.detach(),
            "lambda_prior": self.lambda_prior.detach(),
        }
        prior_diag = None
        if return_details:
            with torch.no_grad():
                weight = self.fusion_skip.weight.detach()
                old_skip_linear = F.linear(fused.detach(), weight, bias=None)
                prior_linear = F.linear(
                    lambda_prior.detach() * h0_fused.detach(), weight, bias=None
                )
                old_skip_rms = self._rms(old_skip_linear)
                prior_rms = self._rms(prior_linear)
                cosine_denominator = (
                    old_skip_linear.float().norm() * prior_linear.float().norm()
                )
                cosine = (
                    (old_skip_linear.float() * prior_linear.float()).sum()
                    / cosine_denominator.clamp_min(self.eps)
                ) if float(cosine_denominator) > 0.0 else old_skip_rms.new_zeros(())
                prior_diag = {
                    "prior_skip_raw": float(self.prior_skip_raw.detach().cpu()),
                    "lambda_prior": float(self.lambda_prior.detach().cpu()),
                    "h0_fused_rms": float(self._rms(h0_fused).cpu()),
                    "fused_rms": float(self._rms(fused).cpu()),
                    "fused_minus_h0_rms": float(self._rms(fused - h0_fused).cpu()),
                    "fusion_skip_weight_fused_rms": float(old_skip_rms.cpu()),
                    "prior_skip_linear_rms": float(prior_rms.cpu()),
                    "fusion_mlp_rms": float(self._rms(mlp_output).cpu()),
                    "prior_to_old_skip_rms_ratio": float(
                        (prior_rms / old_skip_rms.clamp_min(self.eps)).cpu()
                    ),
                    "prior_to_old_skip_cosine": float(cosine.cpu()),
                }
            if self._capture_state_audit:
                self._last_state_audit["intrinsic_prior_skip"] = {
                    "h0_fused": h0_fused,
                    "fused": fused,
                    "skip_input": skip_input,
                    "prior_skip_linear": prior_linear,
                    "old_skip_linear": old_skip_linear,
                    "mlp_output": mlp_output,
                    "z": z,
                }
        mechanism_forward["intrinsic_prior_raw"] = self._diagnostic_scalar(
            self.prior_skip_raw, return_details
        )
        mechanism_forward["intrinsic_prior_lambda"] = self._diagnostic_scalar(
            self.lambda_prior, return_details
        )
        info: dict[str, Any] = {
            "condition": self.condition,
            "variant": self.v13_variant,
            "v13_variant": self.v13_variant,
            "active_nodes": physical_support.detach(),
            "input_self_loops_removed": loop_count,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "prior_skip_raw": self.prior_skip_raw.detach(),
            "lambda_prior": self.lambda_prior.detach(),
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
                "v13_mechanism": dict(mechanism_forward),
            }
            info["details"]["intrinsic_prior_skip"] = prior_diag
        self._last_diagnostics = info.get("details", {})
        return z, None, None, aux_loss, info
