"""Controlled paper-evidence variants of the frozen Hybrid-CGE H2 model.

The historical ``hybrid_cge_v11`` implementation remains untouched. This
module keeps its H2 forward for the full reference arm, and implements only
the pre-registered mechanism switches and the physical three-hop condition.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cg_cge_v1 import receiver_sparse_attention
from src.models.hybrid_cge_v11 import Model as FrozenHybridModel
from src.models.psce_mag_v7a import _MODALITIES, chunked_weighted_spmm


PAPER_CONDITIONS = (
    "H2_full",
    "A1_no_early_physical_cross",
    "A2_no_late_completion",
    "A3_no_moe",
    "A4_no_same_node",
    "H2_physical_3hop",
    "F1_fashion_dropout_aux",
    "F2_fashion_class_balanced_4hop",
    "F4_fashion_class_balanced_3hop",
)
ABLATIONS = PAPER_CONDITIONS[1:5]


class Model(FrozenHybridModel):
    """New controlled model wrapper; the frozen H2 class is not edited."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        requested = str(cfg.model.get("paper_condition", "H2_full"))
        if requested not in PAPER_CONDITIONS:
            raise ValueError(f"paper_condition must be one of {PAPER_CONDITIONS}")

        # The frozen parent intentionally accepts four physical hops only.
        # Construct its unchanged H2 trunk first, then install the separately
        # specified three-hop alpha basis without consuming random numbers.
        base_cfg = copy.deepcopy(cfg)
        base_cfg.model.condition = "H2_hybrid_plus"
        base_cfg.model.physical_hops = 4
        base_cfg.model.late_cross_dropout = 0.15
        super().__init__(base_cfg, data_info)
        self.paper_condition = requested
        self.study_condition = requested
        self.condition = requested
        self.variant = requested

        if requested in {"H2_physical_3hop", "F4_fashion_class_balanced_3hop"}:
            # Same four-row Hadamard initialization, retaining its first three
            # physical-order columns and the parent's L2 row normalization.
            with torch.no_grad():
                first_three = self.alpha_raw.detach()[:, :3].clone()
            self.alpha_raw = nn.Parameter(first_three)
            self.physical_hops = 3
        else:
            self.physical_hops = 4

        if requested == "A3_no_moe":
            # These are every shared expert and Top-2/router parameter tree.
            # Removing modules keeps them out of both forward and optimizer.
            for name in (
                "experts", "context_encoder", "router_projectors", "router_norms",
                "modality_embedding", "selection_head", "strength_head", "node_router",
                "node_modality_embedding", "view_router", "view_type_embedding",
                "comp_router", "comp_modality_embedding", "physical_cross_coefficients",
                "interaction",
            ):
                if hasattr(self, name):
                    delattr(self, name)

    def _late_completion_controlled(
        self,
        target: str,
        target_index: int,
        source: str,
        h_source: torch.Tensor,
        u_private_source: torch.Tensor,
        u_collaborative_target: torch.Tensor,
        *,
        allow_same_node: bool,
        use_moe: bool,
        collect_details: bool,
    ):
        projection_name = "visual_to_text" if source == "visual" else "text_to_visual"
        projection = self.cross_projections[projection_name]
        sender_value = projection(0.7 * h_source + 0.3 * u_private_source)
        semantic_edges = getattr(self, f"semantic_{source}_edge_index_v2").to(
            u_collaborative_target.device
        )
        semantic_mask = self._support_mask(
            semantic_edges, u_collaborative_target.size(0), u_collaborative_target.device
        )
        query = self.query_projections[target](F.layer_norm(u_collaborative_target, (self.hidden_dim,)))
        key = self.key_projections[source](F.layer_norm(u_private_source, (self.hidden_dim,)))
        semantic_candidate, attention = receiver_sparse_attention(
            query, key, sender_value, semantic_edges, self.cross_edge_chunk_size
        )
        semantic_candidate = semantic_candidate * semantic_mask.to(semantic_candidate.dtype).unsqueeze(-1)

        if allow_same_node:
            same_candidate = projection(u_private_source)
            # MAGData has aligned rows for both views and no audited presence mask.
            same_mask = torch.ones_like(semantic_mask)
        else:
            # No same-node projection call in A4.
            same_candidate = torch.zeros_like(semantic_candidate)
            same_mask = torch.zeros_like(semantic_mask)

        cosine = F.cosine_similarity(
            semantic_candidate.detach(), same_candidate.detach(), dim=-1, eps=1.0e-8
        )
        all_nodes = torch.ones_like(semantic_mask)
        target_rms = self._masked_rms(u_collaborative_target, all_nodes).clamp_min(1.0e-6)
        semantic_relative = torch.tanh(
            self._masked_rms(semantic_candidate, semantic_mask) / target_rms
        ).expand(u_collaborative_target.size(0), 1)
        same_relative = torch.tanh(
            self._masked_rms(same_candidate, same_mask) / target_rms
        ).expand(u_collaborative_target.size(0), 1)
        selector_input = torch.cat(
            (
                u_collaborative_target, semantic_candidate, same_candidate, cosine[:, None],
                semantic_relative, same_relative,
                semantic_mask.to(u_collaborative_target.dtype)[:, None],
                same_mask.to(u_collaborative_target.dtype)[:, None],
            ), dim=-1,
        )
        source_logits = self.source_selector(selector_input).clone()
        source_logits[~semantic_mask, 0] = -torch.inf
        source_logits[~same_mask, 1] = -torch.inf
        source_weights = torch.softmax(source_logits, dim=-1)
        any_support = semantic_mask | same_mask
        none_only = torch.zeros_like(source_weights)
        none_only[:, 2] = 1.0
        source_weights = torch.where(any_support[:, None], source_weights, none_only)
        p_accept = source_weights[:, 0] + source_weights[:, 1]
        candidate = (
            source_weights[:, 0:1] * semantic_candidate
            + source_weights[:, 1:2] * same_candidate
        ) / p_accept.clamp_min(1.0e-6)[:, None]
        candidate = candidate * any_support.to(candidate.dtype).unsqueeze(-1)

        interaction_features = torch.cat(
            (
                u_collaborative_target, candidate,
                u_collaborative_target * candidate,
                (u_collaborative_target - candidate).abs(),
            ), dim=-1,
        )
        gate = torch.sigmoid(self.channel_gate(interaction_features))
        if use_moe:
            interaction = self.interaction(interaction_features)
            modality_embedding = self.comp_modality_embedding.weight[target_index].expand(
                u_collaborative_target.size(0), -1
            )
            comp_route = self._top2(
                self.comp_router(torch.cat((interaction, modality_embedding), dim=-1)), self.top_k
            )
            _, comp_share, comp_balance = self._route_balance(comp_route, any_support)
            comp_expert = self._expert_mixture(
                interaction, comp_route["route_weights"], role_index=None,
                modality_index=target_index,
            )
            late_value = candidate + self.complementary_eta * comp_expert
        else:
            # A3's pre-registered no-MoE equation: no interaction MLP, router,
            # experts, or route-balance term.
            comp_share = None
            comp_balance = u_collaborative_target.new_zeros(())
            late_value = candidate

        unbounded = self.cross_lambda[target_index] * p_accept[:, None] * gate * late_value
        unbounded = unbounded * any_support.to(unbounded.dtype).unsqueeze(-1)
        controlled, control_scale = self._relative_control(
            unbounded, u_collaborative_target, self.tau[target_index], self.eps
        )
        residual, keep = self._apply_late_cross_dropout(controlled)
        diagnostics = {}
        if collect_details:
            all_nodes = torch.ones_like(any_support)
            collaborative_rms = self._masked_rms(u_collaborative_target, all_nodes)
            diagnostics = {
                "semantic_support_nodes": int(semantic_mask.sum().detach().cpu()),
                "same_node_support_nodes": int(same_mask.sum().detach().cpu()),
                "semantic_source_weight": float(source_weights[:, 0].detach().mean().cpu()),
                "same_node_source_weight": float(source_weights[:, 1].detach().mean().cpu()),
                "none_source_weight": float(source_weights[:, 2].detach().mean().cpu()),
                "p_accept_mean": float(p_accept.detach().mean().cpu()),
                "semantic_candidate_rms": float(self._masked_rms(semantic_candidate, semantic_mask).cpu()),
                "same_node_candidate_rms": float(self._masked_rms(same_candidate, same_mask).cpu()),
                "candidate_rms": float(self._masked_rms(candidate, any_support).cpu()),
                "channel_gate_mean": float(gate.detach().mean().cpu()),
                "late_moe_enabled": use_moe,
                "complementary_expert_rms": float(self._masked_rms(comp_expert, any_support).cpu()) if use_moe else 0.0,
                "complementary_expert_selection_share": (
                    comp_share.detach().cpu().tolist() if comp_share is not None else None
                ),
                "late_cross_dropout_probability": self.cross_dropout_p,
                "late_cross_dropout_realized_fraction": float((~keep).float().mean().detach().cpu()),
                "uncontrolled_residual_rms": float(self._masked_rms(unbounded, all_nodes).cpu()),
                "controlled_residual_rms": float(self._masked_rms(controlled, all_nodes).cpu()),
                "final_residual_rms": float(self._masked_rms(residual, all_nodes).cpu()),
                "residual_to_collaborative_rms": float(
                    (self._masked_rms(controlled, all_nodes) / collaborative_rms.clamp_min(1.0e-8)).cpu()
                ),
                "relative_control_trigger_fraction": float((control_scale.detach() < 1.0).float().mean().cpu()),
                "relative_control_trigger_fraction_supported": float(
                    (control_scale.detach()[any_support] < 1.0).float().mean().cpu()
                ) if bool(any_support.any()) else 0.0,
                "tau": float(self.tau[target_index].detach().cpu()),
            }
            if attention.numel():
                diagnostics["semantic_attention_edges"] = int(attention.numel())
        return u_collaborative_target + residual, residual, source_weights, diagnostics, comp_balance

    def _late_completion(self, *args, **kwargs):
        # The frozen parent enables its same-node path only when its internal
        # condition is exactly "H2_hybrid_plus". This paper wrapper replaces
        # that name with controlled condition IDs (H2_full, H2_physical_3hop,
        # F1, ...), so delegating to the parent silently masks SameNode for
        # every arm. Route all paper conditions through the explicit helper;
        # only A4 is permitted to skip the projection and mask that logit.
        if len(args) == 7:
            positional, collect_details = args[:-1], bool(args[-1])
        else:
            positional = args
            collect_details = bool(kwargs.pop("collect_details", False))
        if kwargs:
            raise TypeError(f"unexpected _late_completion keyword arguments: {sorted(kwargs)}")
        return self._late_completion_controlled(
            *positional,
            allow_same_node=self.paper_condition != "A4_no_same_node",
            use_moe=True,
            collect_details=collect_details,
        )

    def _private_without_early_cross(
        self, modality: str, modality_index: int, prior: torch.Tensor,
        physical_states: list[torch.Tensor], physical_scale: torch.Tensor,
        active: torch.Tensor,
    ):
        normalized = torch.stack([self._active_rms_norm(item, active) for item in physical_states])
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        private_profiles = torch.einsum("kl,lnd->knd", alpha, normalized)
        base_route = self._physical_route(modality_index, prior, active)
        zero = torch.zeros_like(physical_states[0])
        correction = self._node_correction(prior, physical_states[0], zero, zero, modality_index)
        logits = self.selection_head(base_route["encoded"]).expand(prior.size(0), -1)
        route = self._top2(logits + correction, self.top_k)
        route["strength"] = base_route["strength"]
        _, _, physical_balance = self._route_balance(route, active)
        message = self._expert_mixture(
            private_profiles, route["route_weights"], role_index=None,
            modality_index=modality_index, physical_residual=True,
        )
        message = message * active.to(prior.dtype).unsqueeze(-1)
        response = route["strength"].unsqueeze(-1) * message

        semantic_edges = getattr(self, f"semantic_{modality}_edge_index_v2").to(prior.device)
        semantic_weight = getattr(self, f"semantic_{modality}_weight_v2").to(
            device=prior.device, dtype=prior.dtype
        )
        semantic_raw = chunked_weighted_spmm(
            prior, semantic_edges, semantic_weight, self.view_edge_chunk_size
        )
        semantic_mask = self._support_mask(semantic_edges, prior.size(0), prior.device)
        semantic_context = (semantic_raw / physical_scale) * semantic_mask.to(prior.dtype).unsqueeze(-1)
        role = self.view_type_embedding.weight[0].expand(prior.size(0), -1)
        modality_embedding = self.modality_embedding.weight[modality_index].expand(prior.size(0), -1)
        semantic_logits = self.view_router(
            torch.cat((prior, semantic_context, modality_embedding, role), dim=-1)
        )
        semantic_route = self._top2(semantic_logits, self.top_k)
        _, _, semantic_balance = self._route_balance(semantic_route, semantic_mask)
        semantic_expert = self._expert_mixture(
            semantic_context, semantic_route["route_weights"], role_index=0,
            modality_index=modality_index,
        )
        semantic_response = (semantic_context + semantic_expert) * semantic_mask.to(prior.dtype).unsqueeze(-1)
        private = prior + response + self.gamma_s[modality_index] * semantic_response
        return private, [physical_balance, semantic_balance]

    def _forward_without_late(self, x, edge_index, return_details=False):
        """A2 computes the unchanged early collaborative state and stops there."""
        if edge_index is None:
            raise ValueError("Hybrid-CGE paper study requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("Hybrid-CGE semantic cache requires global node IDs")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("Hybrid-CGE paper study requires a validated V7B.1 cache")
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        states, scales = {}, {}
        for m in _MODALITIES:
            states[m], scales[m] = self._physical_states(priors[m], src, dst, physical, active)
        cross_raw, cross_hat = {}, {}
        for i, m in enumerate(_MODALITIES):
            other = "visual" if m == "text" else "text"
            projection = self.cross_projections["visual_to_text" if m == "text" else "text_to_visual"]
            raw = self._physical_cross_message(
                priors[other], projection, src, dst, physical, active, self.propagation_chunk_size
            )
            cross_raw[m] = raw
            cross_hat[m] = raw / scales[m] * active.to(raw.dtype).unsqueeze(-1)
        private, collab, balances = {}, {}, []
        for i, m in enumerate(_MODALITIES):
            private[m], collab[m], local, _ = self._private_and_collaborative_state(
                m, i, priors[m], states[m], scales[m], cross_raw[m], cross_hat[m], active, return_details
            )
            balances.extend(local)
        balance = torch.stack(balances).mean() if balances else x.new_zeros(())
        z = self._fuse(collab) if hasattr(self, "_fuse") else None
        if z is None:
            fused = torch.cat((collab["text"], collab["visual"]), dim=-1)
            z = self.fusion_norm(
                self.fusion_skip(fused)
                + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
            )
        aux_loss = self.balance_weight * balance
        info = {
            "variant": self.paper_condition, "condition": self.paper_condition,
            "active_nodes": active.detach(), "input_self_loops_removed": loops,
            "balance_loss": balance.detach(), "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "late_completion_executed": False,
        }
        if self.training:
            info["unimodal_logits"] = {
                m: self.unimodal_heads[m](private[m]) for m in _MODALITIES
            }
        return z, None, None, aux_loss, info

    def _forward_without_early_cross(self, x, edge_index, return_details=False):
        """A1 omits physical cross messages and the second physical MoE pass."""
        if edge_index is None:
            raise ValueError("Hybrid-CGE paper study requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("Hybrid-CGE semantic cache requires global node IDs")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("Hybrid-CGE paper study requires a validated V7B.1 cache")
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        states, scales = {}, {}
        for m in _MODALITIES:
            states[m], scales[m] = self._physical_states(priors[m], src, dst, physical, active)
        private, balances = {}, []
        for i, m in enumerate(_MODALITIES):
            private[m], local = self._private_without_early_cross(
                m, i, priors[m], states[m], scales[m], active
            )
            balances.extend(local)
        # U_collab = U_private by design; no physical cross operator, cross
        # profile injection, corrective routing, or duplicated expert pass ran.
        completed = {}
        for i, target in enumerate(_MODALITIES):
            source = "visual" if target == "text" else "text"
            completed[target] = self._late_completion(
                target, i, source, priors[source], private[source], private[target], return_details
            )[0]
        balance = torch.stack(balances).mean() if balances else x.new_zeros(())
        fused = torch.cat((completed["text"], completed["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
        )
        aux_loss = self.balance_weight * balance
        info = {
            "variant": self.paper_condition, "condition": self.paper_condition,
            "active_nodes": active.detach(), "input_self_loops_removed": loops,
            "balance_loss": balance.detach(), "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "early_physical_cross_executed": False,
        }
        if self.training:
            info["unimodal_logits"] = {
                m: self.unimodal_heads[m](private[m]) for m in _MODALITIES
            }
        return z, None, None, aux_loss, info

    def _forward_without_moe(self, x, edge_index, return_details=False):
        """A3 replaces all expert/router paths by the registered weak equations."""
        if edge_index is None:
            raise ValueError("Hybrid-CGE paper study requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("Hybrid-CGE semantic cache requires global node IDs")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("Hybrid-CGE paper study requires a validated V7B.1 cache")
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        states, scales = {}, {}
        for m in _MODALITIES:
            states[m], scales[m] = self._physical_states(priors[m], src, dst, physical, active)
        private, collab = {}, {}
        for i, m in enumerate(_MODALITIES):
            physical_profile = torch.stack(
                [self._active_rms_norm(s, active) for s in states[m]], dim=0
            ).mean(dim=0)
            semantic_edges = getattr(self, f"semantic_{m}_edge_index_v2").to(priors[m].device)
            semantic_weight = getattr(self, f"semantic_{m}_weight_v2").to(
                device=priors[m].device, dtype=priors[m].dtype
            )
            semantic_raw = chunked_weighted_spmm(
                priors[m], semantic_edges, semantic_weight, self.view_edge_chunk_size
            )
            semantic_mask = self._support_mask(semantic_edges, priors[m].size(0), priors[m].device)
            same_source = self._active_rms_norm(semantic_raw, semantic_mask)
            private[m] = priors[m] + physical_profile + self.gamma_s[i] * same_source

            other = "visual" if m == "text" else "text"
            projection = self.cross_projections["visual_to_text" if m == "text" else "text_to_visual"]
            raw = self._physical_cross_message(
                priors[other], projection, src, dst, physical, active, self.propagation_chunk_size
            )
            cross_hat = raw / scales[m] * active.to(raw.dtype).unsqueeze(-1)
            # R_P_collab = R_P + 0.25*C_P_hat.
            collab[m] = private[m] + 0.25 * cross_hat

        completed = {}
        for i, target in enumerate(_MODALITIES):
            source = "visual" if target == "text" else "text"
            completed[target] = self._late_completion_controlled(
                target, i, source, priors[source], private[source], collab[target],
                allow_same_node=True, use_moe=False, collect_details=return_details,
            )[0]
        fused = torch.cat((completed["text"], completed["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
        )
        aux_loss = x.new_zeros(())
        info = {
            "variant": self.paper_condition, "condition": self.paper_condition,
            "active_nodes": active.detach(), "input_self_loops_removed": loops,
            "balance_loss": aux_loss.detach(), "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "moe_executed": False, "moe_router_parameters_present": False,
        }
        if self.training:
            info["unimodal_logits"] = {
                m: self.unimodal_heads[m](private[m]) for m in _MODALITIES
            }
        return z, None, None, aux_loss, info

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if self.paper_condition == "A1_no_early_physical_cross":
            return self._forward_without_early_cross(x, edge_index, return_details)
        if self.paper_condition == "A2_no_late_completion":
            return self._forward_without_late(x, edge_index, return_details)
        if self.paper_condition == "A3_no_moe":
            return self._forward_without_moe(x, edge_index, return_details)
        return super().forward(x, edge_index, return_details=return_details)

    def capture_gradient_diagnostics(self):
        if self.paper_condition != "A3_no_moe":
            return super().capture_gradient_diagnostics()
        self._gradient_diagnostics = {
            "moe_router_and_expert_parameters": {
                "present": False, "optimizer_included": False, "gradient_present": False
            },
            "cross_projections.visual_to_text.0.weight": {
                "gradient_present": self.cross_projections["visual_to_text"][0].weight.grad is not None,
                "gradient_finite": self.cross_projections["visual_to_text"][0].weight.grad is None or bool(
                    torch.isfinite(self.cross_projections["visual_to_text"][0].weight.grad).all()
                ),
            },
        }
        return self._gradient_diagnostics
