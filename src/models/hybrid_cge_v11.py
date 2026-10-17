"""Hybrid-CGE V1.1: early physical collaboration and late sparse completion.

This focused extension reuses CG-CGE V1's validated V7B.1 operators,
directional projections, four shared experts, routers, fusion and custom
receiver-sparse attention. Private states are computed without cross-modal
inputs; physical cross messages enter only the collaborative physical experts
and their existing node-router correction. Late completion uses semantic
attention and, for H2 only, same-node candidates.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cg_cge_v1 import Model as CGCGEV1Model, receiver_sparse_attention
from src.models.psce_mag_v7a import _MODALITIES, chunked_weighted_spmm


CONDITIONS = ("H1_hybrid", "H2_hybrid_plus")


class Model(CGCGEV1Model):
    """H1/H2 share initialization and all learned paths except H2 dropout use."""

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
            raise ValueError(f"hybrid_cge_v11 condition must be one of {CONDITIONS}")

        # Construct the common CG-CGE trunk through its no-FiLM path. The same
        # constructor is used for both arms so added parameters and the external
        # classifier retain matched RNG state.
        base_cfg = copy.deepcopy(cfg)
        base_cfg.model.condition = "B1_complementary"
        super().__init__(base_cfg, data_info)
        self.study_condition = condition
        self.condition = condition
        self.variant = condition
        self.use_role_film = False
        del self.role_film_scale
        del self.role_film_shift

        self.cross_dropout_p = float(
            cfg.model.get("late_cross_dropout", 0.0 if condition == "H1_hybrid" else 0.15)
        )
        expected_dropout = 0.0 if condition == "H1_hybrid" else 0.15
        if not math.isclose(self.cross_dropout_p, expected_dropout, abs_tol=1e-12):
            raise ValueError(f"{condition} requires late_cross_dropout={expected_dropout}")

        # The selector's three slots are [Semantic, SameNode, None] in both
        # conditions. H1 masks SameNode at use time, retaining identical keys.
        with torch.no_grad():
            nn.init.zeros_(self.source_selector[-1].weight)
            self.source_selector[-1].bias.copy_(
                torch.log(torch.tensor([0.4, 0.4, 0.2], dtype=self.source_selector[-1].bias.dtype))
            )

        # These scalar/vector initializations consume no random draws. b_k is
        # shared over modalities; tau is separately bounded per receiver mode.
        self.physical_cross_coefficients = nn.Parameter(torch.full((self.num_experts,), 0.25))
        tau_probability = (0.25 - 0.1) / 0.7
        tau_raw_init = math.log(tau_probability / (1.0 - tau_probability))
        self.tau_raw = nn.Parameter(torch.full((len(_MODALITIES),), tau_raw_init))
        self._last_hybrid_diagnostics: dict[str, Any] = {}

    @property
    def tau(self) -> torch.Tensor:
        return 0.1 + 0.7 * torch.sigmoid(self.tau_raw)

    def _physical_states(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        physical: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        states: list[torch.Tensor] = []
        state = prior
        active_float = active.to(prior.dtype).unsqueeze(-1)
        for _ in range(self.physical_hops):
            state = self._propagate(state, src, dst, physical) * active_float
            states.append(state)
        scale = self._active_channel_scale(states[0], active)
        return states, scale

    @staticmethod
    def _physical_cross_message(
        source_prior: torch.Tensor,
        projection: nn.Module,
        src: torch.Tensor,
        dst: torch.Tensor,
        physical: torch.Tensor,
        active: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        projected = projection(source_prior)
        message = chunked_weighted_spmm(
            projected, torch.stack((src, dst), dim=0), physical, chunk_size
        )
        return message * active.to(message.dtype).unsqueeze(-1)

    def _private_and_collaborative_state(
        self,
        modality: str,
        modality_index: int,
        prior: torch.Tensor,
        physical_states: list[torch.Tensor],
        physical_scale: torch.Tensor,
        physical_cross_raw: torch.Tensor,
        physical_cross_hat: torch.Tensor,
        active: torch.Tensor,
        collect_details: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, Any]]:
        """Compute Private and Early-Collaborative states from one hop cache.

        The Private expression is deliberately contained in this method and
        uses only ``prior`` plus this modality's graph operators. It never
        accepts the opposite modality tensor or a cross-modal message.
        """
        normalized = torch.stack(
            [self._active_rms_norm(item, active) for item in physical_states], dim=0
        )
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        private_profiles = torch.einsum("kl,lnd->knd", alpha, normalized)

        base_route = self._physical_route(modality_index, prior, active)
        zero = torch.zeros_like(physical_states[0])
        private_correction = self._node_correction(
            prior, physical_states[0], zero, zero, modality_index
        )
        base_logits = self.selection_head(base_route["encoded"]).expand(prior.size(0), -1)
        private_route = self._top2(base_logits + private_correction, self.top_k)
        private_route["strength"] = base_route["strength"]
        _, private_share, private_balance = self._route_balance(private_route, active)
        private_message = self._expert_mixture(
            private_profiles,
            private_route["route_weights"],
            role_index=None,
            modality_index=modality_index,
            physical_residual=True,
        )
        private_message = private_message * active.to(prior.dtype).unsqueeze(-1)
        private_response = private_route["strength"].unsqueeze(-1) * private_message

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
        _, semantic_share, semantic_balance = self._route_balance(semantic_route, semantic_mask)
        semantic_expert = self._expert_mixture(
            semantic_context,
            semantic_route["route_weights"],
            role_index=0,
            modality_index=modality_index,
        )
        semantic_response = (semantic_context + semantic_expert) * semantic_mask.to(prior.dtype).unsqueeze(-1)
        semantic_contribution = self.gamma_s[modality_index] * semantic_response
        private_state = prior + private_response + semantic_contribution

        # One shared router parameterization is called again with real C_hat_P.
        cross_correction = self._node_correction(
            prior, physical_states[0], zero, physical_cross_hat, modality_index
        )
        collab_route = self._top2(base_logits + cross_correction, self.top_k)
        collab_route["strength"] = base_route["strength"]
        _, collab_share, collab_balance = self._route_balance(collab_route, active)
        collab_profiles = private_profiles + (
            self.physical_cross_coefficients[:, None, None] * physical_cross_hat[None, :, :]
        )
        collab_message = self._expert_mixture(
            collab_profiles,
            collab_route["route_weights"],
            role_index=None,
            modality_index=modality_index,
            physical_residual=True,
        )
        collab_message = collab_message * active.to(prior.dtype).unsqueeze(-1)
        collab_strength = collab_route["strength"].unsqueeze(-1)
        private_strength = private_route["strength"].unsqueeze(-1)
        # This delta form includes Same-source response exactly once.
        collaborative_state = private_state + collab_strength * collab_message - private_strength * private_message

        balances = [private_balance, collab_balance, semantic_balance]
        diagnostics: dict[str, Any] = {}
        if collect_details:
            physical_mask = active
            private_top = private_route["top_indices"][active]
            collab_top = collab_route["top_indices"][active]
            top_flip = (
                float((private_top != collab_top).any(dim=-1).float().mean().detach().cpu())
                if bool(active.any()) else 0.0
            )
            diagnostics = {
                "physical_support_nodes": int(active.sum().detach().cpu()),
                "same_semantic_support_nodes": int(semantic_mask.sum().detach().cpu()),
                "physical_first_hop_rms": float(self._masked_rms(physical_states[0], active).cpu()),
                "physical_first_hop_scale_mean": float(physical_scale.detach().mean().cpu()),
                "physical_cross_raw_rms": float(self._masked_rms(physical_cross_raw, physical_mask).cpu()),
                "physical_cross_calibrated_rms": float(self._masked_rms(physical_cross_hat, physical_mask).cpu()),
                "physical_cross_coefficients": self.physical_cross_coefficients.detach().cpu().tolist(),
                "physical_private_profile_rms": float(self._masked_rms(private_profiles, active).cpu()),
                "physical_collab_profile_rms": float(self._masked_rms(collab_profiles, active).cpu()),
                "physical_private_response_rms": float(self._masked_rms(private_response, active).cpu()),
                "physical_collab_response_rms": float(self._masked_rms(collab_strength * collab_message, active).cpu()),
                "physical_cross_input_router_correction_rms": float(self._masked_rms(cross_correction, active).cpu()),
                "physical_private_route_share": private_share.detach().cpu().tolist(),
                "physical_collab_route_share": collab_share.detach().cpu().tolist(),
                "physical_route_top2_flip_fraction": top_flip,
                "same_semantic_raw_rms": float(self._masked_rms(semantic_raw, semantic_mask).cpu()),
                "same_semantic_response_rms": float(self._masked_rms(semantic_response, semantic_mask).cpu()),
                "same_semantic_weighted_rms": float(self._masked_rms(semantic_contribution, semantic_mask).cpu()),
                "same_semantic_route_share": semantic_share.detach().cpu().tolist(),
                "gamma_s": float(self.gamma_s[modality_index].detach().cpu()),
                "private_state_rms": float(self._masked_rms(private_state, torch.ones_like(active)).cpu()),
                "collaborative_state_rms": float(self._masked_rms(collaborative_state, torch.ones_like(active)).cpu()),
            }
        return private_state, collaborative_state, balances, diagnostics

    @staticmethod
    def _relative_control(
        residual: torch.Tensor,
        reference: torch.Tensor,
        tau: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_norm = reference.float().norm(dim=-1, keepdim=True)
        residual_norm = residual.float().norm(dim=-1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(residual_norm),
            tau.float().reshape(1, 1) * reference_norm / (residual_norm + eps),
        )
        return residual * scale.to(residual.dtype), scale

    def _apply_late_cross_dropout(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply non-inverted, node-wise dropout to the final Late residual."""
        if self.training and self.cross_dropout_p > 0:
            keep = torch.rand(
                (residual.size(0), 1), device=residual.device, dtype=torch.float32
            ) >= self.cross_dropout_p
            return residual * keep.to(residual.dtype), keep
        keep = torch.ones((residual.size(0), 1), device=residual.device, dtype=torch.bool)
        return residual, keep

    def _late_completion(
        self,
        target: str,
        target_index: int,
        source: str,
        h_source: torch.Tensor,
        u_private_source: torch.Tensor,
        u_collaborative_target: torch.Tensor,
        collect_details: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
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

        if self.study_condition == "H2_hybrid_plus":
            same_candidate = projection(u_private_source)
            # Current MAGData contains a row for each modality at every global
            # node and exposes no reliable presence mask. Do not infer missingness
            # from the numeric value of either feature vector.
            same_mask = torch.ones_like(semantic_mask)
        else:
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
                u_collaborative_target,
                semantic_candidate,
                same_candidate,
                cosine[:, None],
                semantic_relative,
                same_relative,
                semantic_mask.to(u_collaborative_target.dtype)[:, None],
                same_mask.to(u_collaborative_target.dtype)[:, None],
            ),
            dim=-1,
        )
        source_logits = self.source_selector(selector_input).clone()
        source_logits[~semantic_mask, 0] = -torch.inf
        source_logits[~same_mask, 1] = -torch.inf
        source_weights = torch.softmax(source_logits, dim=-1)
        any_support = semantic_mask | same_mask
        # Make the all-empty behavior explicit and exactly deterministic.
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
                u_collaborative_target,
                candidate,
                u_collaborative_target * candidate,
                (u_collaborative_target - candidate).abs(),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.channel_gate(interaction_features))
        interaction = self.interaction(interaction_features)
        modality_embedding = self.comp_modality_embedding.weight[target_index].expand(
            u_collaborative_target.size(0), -1
        )
        comp_route = self._top2(
            self.comp_router(torch.cat((interaction, modality_embedding), dim=-1)), self.top_k
        )
        _, comp_share, comp_balance = self._route_balance(comp_route, any_support)
        comp_expert = self._expert_mixture(
            interaction,
            comp_route["route_weights"],
            role_index=None,
            modality_index=target_index,
        )
        unbounded = self.cross_lambda[target_index] * p_accept[:, None] * gate * (
            candidate + self.complementary_eta * comp_expert
        )
        unbounded = unbounded * any_support.to(unbounded.dtype).unsqueeze(-1)
        controlled, control_scale = self._relative_control(
            unbounded, u_collaborative_target, self.tau[target_index], self.eps
        )
        residual, keep = self._apply_late_cross_dropout(controlled)

        if collect_details and attention.numel():
            entropy_edge = -(attention.detach().clamp_min(1.0e-30) * attention.detach().clamp_min(1.0e-30).log())
            entropy_by_receiver = entropy_edge.new_zeros(u_collaborative_target.size(0))
            entropy_by_receiver.index_add_(0, semantic_edges[1], entropy_edge)
            counts = torch.bincount(semantic_edges[1], minlength=u_collaborative_target.size(0))
            multi = counts > 1
            entropy = float(entropy_by_receiver[semantic_mask].mean().cpu()) if bool(semantic_mask.any()) else 0.0
            entropy_multi = float(entropy_by_receiver[multi].mean().cpu()) if bool(multi.any()) else 0.0
        elif collect_details:
            entropy, entropy_multi = 0.0, 0.0
        diagnostics: dict[str, Any] = {}
        if collect_details:
            raw_rms = self._masked_rms(unbounded, all_nodes)
            controlled_rms = self._masked_rms(controlled, all_nodes)
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
                "channel_gate_saturation_fraction": float(
                    ((gate.detach() < 0.05) | (gate.detach() > 0.95)).float().mean().cpu()
                ),
                "complementary_expert_rms": float(self._masked_rms(comp_expert, any_support).cpu()),
                "complementary_expert_selection_share": comp_share.detach().cpu().tolist(),
                "uncontrolled_residual_rms": float(raw_rms.cpu()),
                "controlled_residual_rms": float(controlled_rms.cpu()),
                "final_residual_rms": float(self._masked_rms(residual, all_nodes).cpu()),
                "residual_to_collaborative_rms": float(
                    (controlled_rms / self._masked_rms(u_collaborative_target, all_nodes).clamp_min(1.0e-8)).cpu()
                ),
                "relative_control_trigger_fraction": float((control_scale.detach() < 1.0).float().mean().cpu()),
                "relative_control_trigger_fraction_supported": float(
                    (control_scale.detach()[any_support] < 1.0).float().mean().cpu()
                ) if bool(any_support.any()) else 0.0,
                "relative_control_scale_mean": float(control_scale.detach().mean().cpu()),
                "tau": float(self.tau[target_index].detach().cpu()),
                "late_cross_dropout_probability": self.cross_dropout_p,
                "late_cross_dropout_realized_fraction": float((~keep).float().mean().detach().cpu()),
                "semantic_attention_entropy": entropy,
                "semantic_attention_entropy_multi_neighbor": entropy_multi,
                "query_rms": float(query.detach().float().square().mean().sqrt().cpu()),
                "key_rms": float(key.detach().float().square().mean().sqrt().cpu()),
            }
        return u_collaborative_target + residual, residual, source_weights, diagnostics, comp_balance

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        if edge_index is None:
            raise ValueError("Hybrid-CGE V1.1 requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("Hybrid-CGE semantic cache uses global IDs; sampled/local IDs are unsupported")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("Hybrid-CGE V1.1 requires a validated V7B.1 semantic cache")

        inputs = self._split_modalities(x)
        priors = {modality: self.projectors[modality](inputs[modality]) for modality in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )

        physical_states: dict[str, list[torch.Tensor]] = {}
        physical_scales: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            physical_states[modality], physical_scales[modality] = self._physical_states(
                priors[modality], src, dst, physical, active
            )

        physical_cross_raw: dict[str, torch.Tensor] = {}
        physical_cross_hat: dict[str, torch.Tensor] = {}
        for modality_index, modality in enumerate(_MODALITIES):
            other = "visual" if modality == "text" else "text"
            projection_name = "visual_to_text" if modality == "text" else "text_to_visual"
            raw = self._physical_cross_message(
                priors[other],
                self.cross_projections[projection_name],
                src,
                dst,
                physical,
                active,
                self.propagation_chunk_size,
            )
            physical_cross_raw[modality] = raw
            physical_cross_hat[modality] = (
                raw / physical_scales[modality]
            ) * active.to(raw.dtype).unsqueeze(-1)

        private: dict[str, torch.Tensor] = {}
        collaborative: dict[str, torch.Tensor] = {}
        modality_diagnostics: dict[str, Any] = {}
        balance_terms: list[torch.Tensor] = []
        for modality_index, modality in enumerate(_MODALITIES):
            (
                private[modality],
                collaborative[modality],
                local_balances,
                modality_diagnostics[modality],
            ) = self._private_and_collaborative_state(
                modality,
                modality_index,
                priors[modality],
                physical_states[modality],
                physical_scales[modality],
                physical_cross_raw[modality],
                physical_cross_hat[modality],
                active,
                return_details,
            )
            balance_terms.extend(local_balances)

        final_states: dict[str, torch.Tensor] = {}
        cross_diagnostics: dict[str, Any] = {}
        for target_index, target in enumerate(_MODALITIES):
            source = "visual" if target == "text" else "text"
            (
                final_states[target],
                residual,
                source_weights,
                cross_diagnostics[target],
                comp_balance,
            ) = self._late_completion(
                target,
                target_index,
                source,
                priors[source],
                private[source],
                collaborative[target],
                return_details,
            )
            modality_diagnostics[target]["late_completion"] = cross_diagnostics[target]
            balance_terms.append(comp_balance)

        # Equal weight per actual route invocation: 2 Private physical, 2
        # Collaborative physical, 2 Same-source and 2 Complementary calls.
        balance = torch.stack(balance_terms).mean() if balance_terms else x.new_zeros(())
        aux_loss = self.balance_weight * balance
        fused_input = torch.cat((final_states["text"], final_states["visual"]), dim=-1)
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
            "late_cross_dropout_probability": self.cross_dropout_p,
            "role_film_enabled": False,
        }
        if self.training:
            info["unimodal_logits"] = {
                modality: self.unimodal_heads[modality](private[modality])
                for modality in _MODALITIES
            }
        if return_details:
            info["details"] = {
                "modalities": modality_diagnostics,
                "early_physical_cross": {
                    modality: {
                        "raw_rms": float(self._masked_rms(physical_cross_raw[modality], active).cpu()),
                        "calibrated_rms": float(self._masked_rms(physical_cross_hat[modality], active).cpu()),
                        "coefficient_mean": float(self.physical_cross_coefficients.detach().mean().cpu()),
                    }
                    for modality in _MODALITIES
                },
                "fusion_input_rms": float(fused_input.detach().float().square().mean().sqrt().cpu()),
                "fused_output_rms": float(z.detach().float().square().mean().sqrt().cpu()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "shared_expert_objects": len({id(expert) for expert in self.experts}),
            }
        self._last_hybrid_diagnostics = info.get("details", {})
        self._last_gradient_diagnostics = getattr(self, "_gradient_diagnostics", None)
        return z, None, None, aux_loss, info

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        parameters = dict(self.named_parameters())
        names = (
            "physical_cross_coefficients",
            "cross_projections.visual_to_text.0.weight",
            "cross_projections.text_to_visual.0.weight",
            "node_router.3.weight",
            "view_router.2.weight",
            "gamma_s_raw",
            "query_projections.text.weight",
            "query_projections.visual.weight",
            "key_projections.text.weight",
            "key_projections.visual.weight",
            "source_selector.2.weight",
            "channel_gate.2.weight",
            "comp_router.3.weight",
            "tau_raw",
            "unimodal_heads.text.weight",
            "unimodal_heads.visual.weight",
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
        self._last_gradient_diagnostics = result
        self._gradient_diagnostics = result
        return result


__all__ = ["Model", "CONDITIONS", "receiver_sparse_attention"]
