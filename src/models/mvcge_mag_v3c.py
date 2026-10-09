from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.mvcge_mag_v3a import Model as V3AModel


_MODALITIES = ("text", "visual")
_VARIANTS = (
    "F0_raw",
    "F1_support",
    "F2_role_dual_smooth",
    "F3_role_functional",
)


class Model(V3AModel):
    """V3C functional-role action-space screen on the V3A C0 backbone.

    Physical edges are partitioned into fixed modality-local Supportive and
    Discrepant channels from detached raw input features. The channel edge
    weights are masks of the original V2.2/V3A normalized physical operator;
    they are never renormalized. Only F3 changes the Discrepant action from
    ordinary smoothing to a signed representation-difference trajectory.
    """

    no_weight_decay_parameter_names = frozenset(
        {"context_mix_raw", "role_mix_raw"}
    )

    def __init__(self, cfg, data_info: dict[str, Any]):
        variant = str(cfg.model.get("variant", "F3_role_functional"))
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        role_mix_init = float(cfg.model.get("role_mix_init", 0.10))
        if not math.isclose(role_mix_init, 0.10, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("V3C fixes role_mix_init=0.10")
        semantic_layernorm_eps = float(
            cfg.model.get("semantic_layernorm_eps", 1.0e-5)
        )
        if not math.isclose(
            semantic_layernorm_eps, 1.0e-5, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("V3C fixes semantic_layernorm_eps=1e-5")

        # V3A constructs the complete V2.2 R0 module tree in its historical
        # order. Its additional V3A tensors are constant initialized and do
        # not consume RNG, so the downstream NC classifier stream is retained.
        base_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        base_cfg.variant = "C0_raw"
        super().__init__(SimpleNamespace(model=base_cfg), data_info)
        self.variant = variant
        self.role_mix_init = role_mix_init
        self.semantic_layernorm_eps = semantic_layernorm_eps

        # Deterministic construction: preserve F0-F3 and classifier RNG parity.
        role_logit = math.log(role_mix_init / (1.0 - role_mix_init))
        self.role_mix_raw = nn.Parameter(
            torch.full((self.num_experts,), role_logit, dtype=torch.float32)
        )
        self.context_mix_raw.requires_grad_(False)
        self.alpha_effective_raw.requires_grad_(False)
        self.role_mix_raw.requires_grad_(
            variant in {"F2_role_dual_smooth", "F3_role_functional"}
        )

    @staticmethod
    def _semantic_features(x_raw: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
        """Apply parameter-free per-node LayerNorm and finite L2 normalization."""
        detached = x_raw.detach()
        normalized = F.layer_norm(
            detached,
            normalized_shape=(detached.size(-1),),
            weight=None,
            bias=None,
            eps=eps,
        )
        norm = torch.linalg.vector_norm(normalized, ord=2, dim=-1, keepdim=True)
        safe = normalized / norm.clamp_min(eps)
        return torch.where(norm > eps, safe, torch.zeros_like(safe))

    @staticmethod
    def _edge_cosine(
        semantic_features: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        *,
        chunk_edges: int = 16384,
    ) -> torch.Tensor:
        if src.numel() == 0:
            return semantic_features.new_empty((0,))
        chunks = []
        for start in range(0, src.numel(), chunk_edges):
            end = min(start + chunk_edges, src.numel())
            left = semantic_features.index_select(0, src[start:end])
            right = semantic_features.index_select(0, dst[start:end])
            chunks.append((left * right).sum(dim=-1).clamp(-1.0, 1.0))
        return torch.cat(chunks, dim=0)

    def _role_partition(
        self,
        x_raw: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        raw_norm_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build one label-free local-relative role partition for one modality."""
        semantic = self._semantic_features(x_raw, self.semantic_layernorm_eps)
        cosine = self._edge_cosine(semantic, src, dst)
        num_nodes = x_raw.size(0)
        incoming_count = x_raw.new_zeros((num_nodes,))
        similarity_sum = x_raw.new_zeros((num_nodes,))
        if dst.numel():
            incoming_count.index_add_(0, dst, torch.ones_like(cosine))
            similarity_sum.index_add_(0, dst, cosine)
        local_mean = similarity_sum / incoming_count.clamp_min(1.0)
        role_score = cosine - 0.5 * (local_mean[src] + local_mean[dst])
        supportive = role_score >= 0.0
        discrepant = ~supportive
        support_weight = raw_norm_weight * supportive.to(raw_norm_weight.dtype)
        discrepant_weight = raw_norm_weight * discrepant.to(raw_norm_weight.dtype)
        support_incoming_count = x_raw.new_zeros((num_nodes,))
        discrepant_incoming_count = x_raw.new_zeros((num_nodes,))
        if dst.numel():
            support_incoming_count.index_add_(
                0, dst, supportive.to(x_raw.dtype)
            )
            discrepant_incoming_count.index_add_(
                0, dst, discrepant.to(x_raw.dtype)
            )
        return {
            "semantic_features": semantic,
            "semantic_cosine": cosine,
            "incoming_count": incoming_count,
            "similarity_sum": similarity_sum,
            "local_mean": local_mean,
            "role_score": role_score,
            "supportive_mask": supportive,
            "discrepant_mask": discrepant,
            "support_norm_weight": support_weight,
            "discrepant_norm_weight": discrepant_weight,
            "support_incoming_count": support_incoming_count,
            "discrepant_incoming_count": discrepant_incoming_count,
            "support_active": support_incoming_count > 0,
            "discrepant_active": discrepant_incoming_count > 0,
            "partition_max_abs_error": (
                (support_weight + discrepant_weight - raw_norm_weight).abs().max()
                if raw_norm_weight.numel()
                else raw_norm_weight.new_zeros(())
            ),
        }

    def _role_trajectories(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        role: dict[str, torch.Tensor],
        *,
        actions: frozenset[str] = frozenset({"support", "smooth", "signed"}),
    ) -> dict[str, torch.Tensor]:
        support_active = role["support_active"]
        discrepant_active = role["discrepant_active"]
        result: dict[str, torch.Tensor] = {}

        if "support" in actions:
            support_states = []
            state = prior
            for _ in range(self.trajectory_order):
                state = self._propagate(
                    state, src, dst, role["support_norm_weight"]
                )
                state = state * support_active.to(state.dtype).unsqueeze(-1)
                support_states.append(state)
            result["support_states"] = torch.stack(support_states, dim=0)
            result["support_trajectory"] = torch.stack(
                [
                    self._active_rms_norm(value, support_active)
                    for value in support_states
                ],
                dim=0,
            )

        if "smooth" in actions or "signed" in actions:
            discrepant_states = []
            discrepant_signed_states = []
            previous = prior
            for _ in range(self.trajectory_order):
                current = self._propagate(
                    previous, src, dst, role["discrepant_norm_weight"]
                )
                current = current * discrepant_active.to(current.dtype).unsqueeze(-1)
                if "smooth" in actions:
                    discrepant_states.append(current)
                if "signed" in actions:
                    difference = (previous - current) * discrepant_active.to(
                        current.dtype
                    ).unsqueeze(-1)
                    discrepant_signed_states.append(difference)
                previous = current
            if "smooth" in actions:
                result["discrepant_states"] = torch.stack(discrepant_states, dim=0)
                result["discrepant_smooth_trajectory"] = torch.stack(
                    [
                        self._active_rms_norm(value, discrepant_active)
                        for value in discrepant_states
                    ],
                    dim=0,
                )
            if "signed" in actions:
                result["discrepant_signed_states"] = torch.stack(
                    discrepant_signed_states, dim=0
                )
                result["discrepant_signed_trajectory"] = torch.stack(
                    [
                        self._active_rms_norm(value, discrepant_active)
                        for value in discrepant_signed_states
                    ],
                    dim=0,
                )
        return result

    def _attach_f0_role_details(
        self,
        result,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
    ):
        z, aux1, aux2, aux_loss, info = result
        inputs = self._split_modalities(x)
        src, dst, raw_norm, _, _ = self._normalized_operator(
            edge_index, x.size(0), x.dtype
        )
        role_info = {}
        for modality in _MODALITIES:
            role = self._role_partition(inputs[modality], src, dst, raw_norm)
            item = info["details"][modality]
            prior = item["prior"]
            role.update(self._role_trajectories(prior, src, dst, role))
            alpha = item["alpha_raw"]
            role["support_profiles"] = torch.einsum(
                "mk,knd->mnd", alpha, role["support_trajectory"]
            )
            role["discrepant_smooth_profiles"] = torch.einsum(
                "mk,knd->mnd", alpha, role["discrepant_smooth_trajectory"]
            )
            role["discrepant_signed_profiles"] = torch.einsum(
                "mk,knd->mnd", alpha, role["discrepant_signed_trajectory"]
            )
            role["role_mix"] = torch.sigmoid(self.role_mix_raw)
            role.pop("semantic_features", None)
            role_info[modality] = role
            item.update(role)
        info["role_channels"] = role_info
        info["variant"] = self.variant
        return z, aux1, aux2, aux_loss, info

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        if self.variant == "F0_raw":
            # This exact V3A call preserves its C0 forward operation order.
            saved_variant = self.variant
            self.variant = "C0_raw"
            try:
                result = super().forward(x, edge_index, return_details=return_details)
            finally:
                self.variant = saved_variant
            if return_details:
                return self._attach_f0_role_details(result, x, edge_index)
            result[4]["variant"] = self.variant
            return result

        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, raw_norm, physical_active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )
        roles: dict[str, dict[str, torch.Tensor]] = {}
        raw_bases: dict[str, torch.Tensor] = {}
        role_bases: dict[str, dict[str, torch.Tensor]] = {}
        router_states: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            prior = priors[modality]
            raw_states = []
            state = prior
            for _ in range(self.trajectory_order):
                state = self._propagate(state, src, dst, raw_norm)
                state = state * physical_active.to(state.dtype).unsqueeze(-1)
                raw_states.append(state)
            raw_bases[modality] = torch.stack(
                [
                    self._active_rms_norm(value, physical_active)
                    for value in raw_states
                ],
                dim=0,
            )
            roles[modality] = self._role_partition(
                inputs[modality], src, dst, raw_norm
            )
            if return_details:
                actions = frozenset({"support", "smooth", "signed"})
            elif self.variant == "F2_role_dual_smooth":
                actions = frozenset({"support", "smooth"})
            elif self.variant == "F3_role_functional":
                actions = frozenset({"support", "signed"})
            else:
                actions = frozenset({"support"})
            role_bases[modality] = self._role_trajectories(
                prior, src, dst, roles[modality], actions=actions
            )
            # The edge-indexed feature matrix is only an intermediate used to
            # compute cosine scores; retaining it doubles raw feature memory.
            roles[modality].pop("semantic_features", None)
            router_states[modality] = self.router_norms[modality](
                self.router_projectors[modality](prior)
            )

        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        role_mix = torch.sigmoid(self.role_mix_raw)
        outputs: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        modality_balance: dict[str, torch.Tensor] = {}
        modality_summaries: dict[str, dict[str, torch.Tensor]] = {}
        router_info: dict[str, dict[str, torch.Tensor]] = {}
        for modality_index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            role_base = role_bases[modality]
            raw_basis = raw_bases[modality]
            support_basis = role_base["support_trajectory"]
            disc_smooth_basis = role_base.get("discrepant_smooth_trajectory")
            disc_signed_basis = role_base.get("discrepant_signed_trajectory")
            support_profiles = torch.einsum("mk,knd->mnd", alpha, support_basis)
            if return_details:
                assert disc_smooth_basis is not None
                assert disc_signed_basis is not None
                raw_profiles = torch.einsum("mk,knd->mnd", alpha, raw_basis)
                disc_smooth_profiles = torch.einsum("mk,knd->mnd", alpha, disc_smooth_basis)
                disc_signed_profiles = torch.einsum("mk,knd->mnd", alpha, disc_signed_basis)
            else:
                raw_profiles = None
                disc_smooth_profiles = (
                    torch.einsum("mk,knd->mnd", alpha, disc_smooth_basis)
                    if disc_smooth_basis is not None
                    else None
                )
                disc_signed_profiles = (
                    torch.einsum("mk,knd->mnd", alpha, disc_signed_basis)
                    if disc_signed_basis is not None
                    else None
                )

            if self.variant == "F1_support":
                profiles = support_profiles
            elif self.variant == "F2_role_dual_smooth":
                lam = role_mix.view(self.num_experts, 1, 1)
                assert disc_smooth_profiles is not None
                profiles = (1.0 - lam) * support_profiles + lam * disc_smooth_profiles
            else:
                lam = role_mix.view(self.num_experts, 1, 1)
                assert disc_signed_profiles is not None
                profiles = (1.0 - lam) * support_profiles + lam * disc_signed_profiles

            transformed = torch.stack(
                [expert(profiles[j]) for j, expert in enumerate(self.experts)], dim=0
            )
            expert_values = profiles + self.expert_feature_scale * transformed
            route = self._static_route(
                modality_index, router_states[modality], physical_active
            )
            mixture = torch.einsum("nm,mnd->nd", route["route_weights"], expert_values)
            scaled_mixture = route["strength"].unsqueeze(-1) * mixture
            output = prior + scaled_mixture
            modality_balance[modality] = route["balance"]
            outputs[modality] = output
            router_info[modality] = route
            modality_summaries[modality] = {
                "strength_mean": route["strength"].detach().mean(),
                "selected_pair": route["top_indices"][0].detach(),
                "top2_weights": route["route_weights"][0].detach(),
            }
            if return_details:
                assert raw_profiles is not None
                assert disc_smooth_profiles is not None
                assert disc_signed_profiles is not None
                details[modality] = {
                    "prior": prior,
                    "raw_trajectory": raw_basis,
                    "support_trajectory": support_basis,
                    "discrepant_smooth_trajectory": disc_smooth_basis,
                    "discrepant_signed_trajectory": disc_signed_basis,
                    "support_states": role_base["support_states"],
                    "discrepant_states": role_base["discrepant_states"],
                    "discrepant_signed_states": role_base["discrepant_signed_states"],
                    "alpha": alpha,
                    "raw_profiles": raw_profiles,
                    "support_profiles": support_profiles,
                    "discrepant_smooth_profiles": disc_smooth_profiles,
                    "discrepant_signed_profiles": disc_signed_profiles,
                    "expert_inputs": profiles,
                    "mixed_profiles": profiles,
                    "expert_outputs": expert_values,
                    "mixture": mixture,
                    "scaled_mixture": scaled_mixture,
                    "output": output,
                    "role_mix": role_mix,
                    "raw_operator_weight": raw_norm,
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
