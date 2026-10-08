from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.mvcge_mag_v22 import Model as V22Model


_MODALITIES = ("text", "visual")
_VARIANTS = (
    "C0_raw",
    "C1_effective",
    "C2_dual_shared_profile",
    "C3_dual_context_profile",
)


class Model(V22Model):
    """V3A screen for raw and modality-effective structural expert actions.

    The V2.2 R0 modules are initialized in their original order, preserving the
    historical static controller and classifier RNG stream. Additional
    parameters are constant-initialized after that stream. All variants use
    the same modality-static route and strength; only their structural context
    profiles differ.
    """

    no_weight_decay_parameter_names = frozenset({"context_mix_raw"})

    def __init__(self, cfg, data_info: dict[str, Any]):
        variant = str(cfg.model.get("variant", "C3_dual_context_profile"))
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        context_mix_init = float(cfg.model.get("context_mix_init", 0.10))
        if not math.isclose(context_mix_init, 0.10, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("V3A fixes context_mix_init=0.10")

        # Construct the complete V2.2 R0 module tree first. This preserves its
        # parameter initialization order and leaves the following NC classifier
        # on the exact same RNG stream as a V2.2 R0 model.
        base_model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        base_model_cfg.variant = "R0_modality_static"
        super().__init__(SimpleNamespace(model=base_model_cfg), data_info)
        self.variant = variant
        self.context_mix_init = context_mix_init

        hadamard = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        # Deterministic constants: these constructions consume no RNG draws.
        self.alpha_effective_raw = nn.Parameter(0.5 * hadamard)
        mix_logit = math.log(context_mix_init / (1.0 - context_mix_init))
        self.context_mix_raw = nn.Parameter(
            torch.full((self.num_experts,), mix_logit, dtype=torch.float32)
        )
        self.alpha_effective_raw.requires_grad_(variant == "C3_dual_context_profile")
        self.context_mix_raw.requires_grad_(
            variant in {"C2_dual_shared_profile", "C3_dual_context_profile"}
        )
        # The inherited legacy evidence, node-routing and compatibility
        # modules were initialized for checkpoint/RNG compatibility, but V3A
        # never uses or trains them.
        self._set_new_parameter_activity()

    @staticmethod
    def _safe_normalize_raw_features(x_raw: torch.Tensor, eps: float) -> torch.Tensor:
        detached = x_raw.detach()
        norm = torch.linalg.vector_norm(detached, ord=2, dim=-1, keepdim=True)
        safe = detached / (norm + eps)
        return torch.where(norm > 0, safe, torch.zeros_like(safe))

    @staticmethod
    def _semantic_edge_similarity(
        normalized_features: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        *,
        chunk_edges: int = 16384,
    ) -> torch.Tensor:
        """Compute fixed raw-feature cosines with bounded edge-feature memory."""
        if src.numel() == 0:
            return normalized_features.new_empty((0,))
        pieces = []
        for start in range(0, src.numel(), chunk_edges):
            end = min(start + chunk_edges, src.numel())
            source = normalized_features.index_select(0, src[start:end])
            target = normalized_features.index_select(0, dst[start:end])
            pieces.append((source * target).sum(dim=-1).clamp(-1.0, 1.0))
        return torch.cat(pieces, dim=0)

    def _effective_operator(
        self,
        x_raw: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized = self._safe_normalize_raw_features(x_raw, self.eps)
        cosine = self._semantic_edge_similarity(normalized, src, dst)
        semantic_weight = (0.5 * (1.0 + cosine)).clamp(0.0, 1.0)
        effective_degree = x_raw.new_zeros((x_raw.size(0),))
        if dst.numel():
            effective_degree.index_add_(0, dst, semantic_weight)
        positive = effective_degree > self.eps
        inv_sqrt = x_raw.new_zeros(effective_degree.shape)
        inv_sqrt[positive] = effective_degree[positive].rsqrt()
        effective_norm = semantic_weight * inv_sqrt[src] * inv_sqrt[dst]
        return {
            "semantic_similarity": cosine,
            "semantic_weight": semantic_weight,
            "effective_degree": effective_degree,
            "inv_sqrt_degree": inv_sqrt,
            "norm_weight": effective_norm,
        }

    def _static_route(
        self,
        modality_index: int,
        router_state: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n = router_state.size(0)
        if bool(active.any()):
            global_single = router_state[active].mean(dim=0, keepdim=True)
        else:
            global_single = router_state.new_zeros((1, self.router_dim))
        embedding_single = self.modality_embedding.weight[modality_index].unsqueeze(0)
        zeros = torch.zeros_like(global_single)
        static_input = torch.cat(
            [zeros, zeros, global_single, zeros, embedding_single], dim=-1
        )
        # Preserve the V2.2 R0 one-row encode and selection/strength heads.
        encoded = self.context_encoder(static_input)
        logits_single = self.selection_head(encoded)
        logits = logits_single.expand(n, -1)
        top = self._top2(logits, self.top_k)
        strength_logit_single = self.strength_head(encoded).squeeze(-1)
        strength_logit = strength_logit_single.expand(n)
        strength = torch.sigmoid(strength_logit)
        if bool(active.any()):
            dense_importance = top["dense_probs"][active].mean(dim=0)
            selection_share = (
                top["selected_mask"][active].float().mean(dim=0) / self.top_k
            )
            balance = self.num_experts * (dense_importance * selection_share).sum()
        else:
            dense_importance = logits.new_zeros((self.num_experts,))
            selection_share = dense_importance
            balance = logits.new_zeros(())
        return {
            "router_input": static_input,
            "router_blocks": {
                "global_single": global_single,
                "modality_embedding": embedding_single,
                "static_input": static_input,
            },
            "encoded_selection": encoded,
            "encoded_strength": encoded,
            "selection_logits": logits,
            "static_logits": logits,
            **top,
            "strength_logit": strength_logit,
            "strength": strength,
            "importance": dense_importance,
            "selection_share": selection_share,
            "balance": balance,
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, raw_norm, active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )

        operators: dict[str, dict[str, torch.Tensor]] = {}
        raw_bases: dict[str, torch.Tensor] = {}
        effective_bases: dict[str, torch.Tensor] = {}
        router_states: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            operators[modality] = self._effective_operator(
                inputs[modality], src, dst
            )
            raw_states = []
            state = priors[modality]
            for _ in range(self.trajectory_order):
                state = self._propagate(state, src, dst, raw_norm)
                state = state * active.to(state.dtype).unsqueeze(-1)
                raw_states.append(state)
            raw_bases[modality] = torch.stack(
                [self._active_rms_norm(value, active) for value in raw_states], dim=0
            )

            effective_states = []
            state = priors[modality]
            effective_norm = operators[modality]["norm_weight"]
            for _ in range(self.trajectory_order):
                state = self._propagate(state, src, dst, effective_norm)
                state = state * active.to(state.dtype).unsqueeze(-1)
                effective_states.append(state)
            effective_bases[modality] = torch.stack(
                [self._active_rms_norm(value, active) for value in effective_states], dim=0
            )
            router_states[modality] = self.router_norms[modality](
                self.router_projectors[modality](priors[modality])
            )

        alpha_raw = self._effective_alpha(self.alpha_raw, self.eps)
        alpha_effective = self._effective_alpha(self.alpha_effective_raw, self.eps)
        context_mix = torch.sigmoid(self.context_mix_raw)

        outputs: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        modality_balance: dict[str, torch.Tensor] = {}
        modality_summaries: dict[str, dict[str, torch.Tensor]] = {}
        router_info: dict[str, dict[str, torch.Tensor]] = {}
        for modality_index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            raw_basis = raw_bases[modality]
            effective_basis = effective_bases[modality]
            raw_profiles = torch.einsum("mk,knd->mnd", alpha_raw, raw_basis)
            effective_shared_profiles = torch.einsum(
                "mk,knd->mnd", alpha_raw, effective_basis
            )
            effective_context_profiles = torch.einsum(
                "mk,knd->mnd", alpha_effective, effective_basis
            )
            if self.variant == "C0_raw":
                profiles = raw_profiles
            elif self.variant == "C1_effective":
                profiles = effective_shared_profiles
            elif self.variant == "C2_dual_shared_profile":
                lam = context_mix.view(self.num_experts, 1, 1)
                profiles = (1.0 - lam) * raw_profiles + lam * effective_shared_profiles
            else:
                lam = context_mix.view(self.num_experts, 1, 1)
                profiles = (1.0 - lam) * raw_profiles + lam * effective_context_profiles

            transformed = torch.stack(
                [expert(profiles[j]) for j, expert in enumerate(self.experts)], dim=0
            )
            expert_values = profiles + self.expert_feature_scale * transformed
            route = self._static_route(
                modality_index, router_states[modality], active
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
                details[modality] = {
                    "prior": prior,
                    "basis": raw_basis,
                    "raw_trajectory": raw_basis,
                    "effective_trajectory": effective_basis,
                    "alpha": alpha_raw,
                    "alpha_raw": alpha_raw,
                    "alpha_effective": alpha_effective,
                    "raw_profiles": raw_profiles,
                    "effective_shared_profiles": effective_shared_profiles,
                    "effective_context_profiles": effective_context_profiles,
                    "expert_inputs": profiles,
                    "mixed_profiles": profiles,
                    "expert_outputs": expert_values,
                    "mixture": mixture,
                    "scaled_mixture": scaled_mixture,
                    "output": output,
                    "semantic_similarity": operators[modality]["semantic_similarity"],
                    "semantic_weight": operators[modality]["semantic_weight"],
                    "effective_degree": operators[modality]["effective_degree"],
                    "raw_operator_weight": raw_norm,
                    "effective_operator_weight": operators[modality]["norm_weight"],
                    "context_mix": context_mix,
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
            "active_nodes": active.detach(),
            "input_self_loops_removed": input_self_loops,
            "balance_loss": balance_loss.detach(),
            "modalities": modality_summaries,
        }
        if return_details:
            info["details"] = details
            info["router"] = router_info
            info["operators"] = operators
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
