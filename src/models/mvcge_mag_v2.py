from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops


_MODALITIES = ("text", "visual")
_VARIANTS = (
    "M0_anchor",
    "M1_direct_moe",
    "M2_anchored_moe",
    "M3_collaborative_moe",
)


class _IntrinsicProjector(nn.Module):
    """Matched modality projector used by the preceding CSE/SOSB screens."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.skip(x) + self.linear2(self.dropout(F.gelu(self.linear1(x))))
        )


class _ExpertTransform(nn.Module):
    """Bias-free shared expert transform; zero structural input stays zero."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class Model(nn.Module):
    """MvCGE-MAG V2: one anchored collaborative structural expert block.

    The model uses a shared four-profile response dictionary and Top-2 sparse
    routing. M3 adds a public cross-view context to the shared router. This is
    a single-block, MvCGE-inspired design rather than a full MvCGE reproduction.
    """

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("mvcge_mag_v2 requires positive text and visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError("mvcge_mag_v2 requires exact [text, visual] feature layout")

        self.variant = str(model_cfg.get("variant", "M3_collaborative_moe"))
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.num_layers = int(model_cfg.get("num_layers", 4))
        self.num_experts = int(model_cfg.get("num_experts", 4))
        self.top_k = int(model_cfg.get("top_k", 2))
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.router_dim = int(model_cfg.get("router_dim", 64))
        self.router_hidden_dim = int(model_cfg.get("router_hidden_dim", 128))
        self.modality_embed_dim = int(model_cfg.get("modality_embed_dim", 16))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.expert_feature_scale = float(model_cfg.get("expert_feature_scale", 0.1))
        self.residual_max = float(model_cfg.get("residual_max", 0.25))
        self.balance_weight = float(model_cfg.get("balance_weight", 0.01))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.base_gate_init = float(model_cfg.get("base_gate_init", -2.0))
        self.direct_strength_init = float(model_cfg.get("direct_strength_init", -1.15))
        self.residual_strength_init = float(model_cfg.get("residual_strength_init", -2.0))
        if (self.hidden_dim, self.num_layers, self.num_experts, self.top_k) != (256, 4, 4, 2):
            raise ValueError("V2 fixes hidden_dim=256, trajectory_order=4, M=4, TopK=2")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        self.out_dim = self.hidden_dim

        # Every variant constructs modules in this same order. Thus matched
        # same-seed projectors, fusion, routers, experts, and downstream heads
        # begin identically. Variant strength biases are overwritten only below.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
                "visual": _IntrinsicProjector(self.visual_dim, self.hidden_dim, self.dropout_p),
            }
        )
        self.beta_base_raw = nn.Parameter(torch.full((4,), 0.5))
        self.gamma_base = nn.Parameter(torch.tensor(self.base_gate_init, dtype=torch.float32))
        hadamard = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.alpha_raw = nn.Parameter(0.5 * hadamard)
        self.experts = nn.ModuleList(
            [
                _ExpertTransform(self.hidden_dim, self.expert_bottleneck, self.dropout_p)
                for _ in range(self.num_experts)
            ]
        )
        self.router_projectors = nn.ModuleDict(
            {m: nn.Linear(self.hidden_dim, self.router_dim) for m in _MODALITIES}
        )
        self.router_norms = nn.ModuleDict(
            {m: nn.LayerNorm(self.router_dim) for m in _MODALITIES}
        )
        self.modality_embedding = nn.Embedding(2, self.modality_embed_dim)
        router_input_dim = 4 * self.router_dim + self.modality_embed_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(router_input_dim, self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.selection_head = nn.Linear(self.router_hidden_dim, self.num_experts)
        self.strength_head = nn.Linear(self.router_hidden_dim, 1)
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # A zero-weight scalar head makes configured initial strengths exact;
        # it still learns node-conditioned strength through ordinary gradients.
        nn.init.zeros_(self.strength_head.weight)
        strength_bias = (
            self.direct_strength_init
            if self.variant == "M1_direct_moe"
            else self.residual_strength_init
        )
        nn.init.constant_(self.strength_head.bias, strength_bias)
        self._freeze_inactive_variant_parameters()

    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _freeze_inactive_variant_parameters(self) -> None:
        self._set_trainable(self.projectors, True)
        for module in (
            self.fusion_linear1,
            self.fusion_linear2,
            self.fusion_skip,
            self.fusion_norm,
        ):
            self._set_trainable(module, True)
        uses_anchor = self.variant in {"M0_anchor", "M2_anchored_moe", "M3_collaborative_moe"}
        uses_moe = self.variant != "M0_anchor"
        self.beta_base_raw.requires_grad_(uses_anchor)
        self.gamma_base.requires_grad_(uses_anchor)
        self._set_trainable(self.experts, uses_moe)
        self._set_trainable(self.router_projectors, uses_moe)
        self._set_trainable(self.router_norms, uses_moe)
        self._set_trainable(self.modality_embedding, uses_moe)
        self._set_trainable(self.context_encoder, uses_moe)
        self._set_trainable(self.selection_head, uses_moe)
        self._set_trainable(self.strength_head, uses_moe)
        self.alpha_raw.requires_grad_(uses_moe)

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [nodes, {self.input_dim}], got {tuple(x.shape)}")
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    @staticmethod
    def _normalized_operator(
        edge_index: torch.Tensor | None, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if edge_index is None:
            raise ValueError("mvcge_mag_v2 requires edge_index")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        edge_index = edge_index.to(dtype=torch.long)
        input_loops = int((edge_index[0] == edge_index[1]).sum().item())
        edge_index, _ = remove_self_loops(edge_index)
        src, dst = edge_index
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=dtype, device=edge_index.device))
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        norm = inv_sqrt[src] * inv_sqrt[dst]
        return src, dst, norm, degree > 0, input_loops

    @staticmethod
    def _propagate(
        state: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(state).index_add(0, dst, state[src] * norm.unsqueeze(-1))

    def _active_rms_norm(self, value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(value)
        if bool(active.any()):
            selected = value[active]
            rms = torch.sqrt(selected.square().mean(dim=0) + self.eps)
            output[active] = selected / rms
        return output

    @staticmethod
    def _effective_alpha(raw: torch.Tensor, eps: float) -> torch.Tensor:
        return raw / (raw.norm(p=2, dim=-1, keepdim=True) + eps)

    def _route(
        self,
        modality_index: int,
        modality: str,
        u: torch.Tensor,
        local: torch.Tensor,
        global_context: torch.Tensor,
        cross_modal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedding = self.modality_embedding.weight[modality_index].expand(u.size(0), -1)
        router_input = torch.cat([u, local, global_context, cross_modal, embedding], dim=-1)
        encoded = self.context_encoder(router_input)
        logits = self.selection_head(encoded)
        dense_probs = torch.softmax(logits, dim=-1)
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        route_weights = torch.zeros_like(dense_probs).scatter(1, top_indices, top_weights)
        selected_mask = torch.zeros_like(dense_probs, dtype=torch.bool).scatter(
            1, top_indices, True
        )
        strength_logit = self.strength_head(encoded).squeeze(-1)
        strength = torch.sigmoid(strength_logit)
        del modality
        return {
            "router_input": router_input,
            "selection_logits": logits,
            "dense_probs": dense_probs,
            "top_indices": top_indices,
            "route_weights": route_weights,
            "selected_mask": selected_mask,
            "strength_logit": strength_logit,
            "strength": strength,
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
        src, dst, norm, active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )
        bases: dict[str, torch.Tensor] = {}
        router_states: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            states = []
            state = priors[modality]
            for _ in range(4):
                state = self._propagate(state, src, dst, norm)
                state = state * active.to(state.dtype).unsqueeze(-1)
                states.append(state)
            bases[modality] = torch.stack(
                [self._active_rms_norm(state, active) for state in states], dim=0
            )
            router_states[modality] = self.router_norms[modality](
                self.router_projectors[modality](priors[modality])
            )

        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        beta = self.beta_base_raw / (self.beta_base_raw.norm(p=2) + self.eps)
        base_gate = torch.sigmoid(self.gamma_base)
        active_count = int(active.sum().item())
        global_contexts = {}
        for modality in _MODALITIES:
            if active_count:
                mean = router_states[modality][active].mean(dim=0, keepdim=True)
                global_contexts[modality] = mean.expand(x.size(0), -1)
            else:
                global_contexts[modality] = torch.zeros_like(router_states[modality])
        cross_modal = (
            0.5 * (router_states["text"] + router_states["visual"])
            if self.variant == "M3_collaborative_moe"
            else torch.zeros_like(router_states["text"])
        )

        outputs: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        modality_balance: dict[str, torch.Tensor] = {}
        modality_summaries: dict[str, dict[str, torch.Tensor]] = {}
        expert_inputs: dict[str, torch.Tensor] = {}
        expert_outputs: dict[str, torch.Tensor] = {}
        router_info: dict[str, dict[str, torch.Tensor]] = {}

        for index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            basis = bases[modality]
            base = torch.einsum("k,knd->nd", beta, basis)
            base_scaled = base_gate * base
            profiles = torch.einsum("mk,knd->mnd", alpha, basis)
            transformed = torch.stack(
                [expert(profiles[j]) for j, expert in enumerate(self.experts)], dim=0
            )
            expert_values = profiles + self.expert_feature_scale * transformed
            route = self._route(
                index,
                modality,
                router_states[modality],
                self._propagate(router_states[modality], src, dst, norm),
                global_contexts[modality],
                cross_modal,
            )
            mixture = torch.einsum("nm,mnd->nd", route["route_weights"], expert_values)
            strength = route["strength"]
            if self.variant == "M0_anchor":
                scaled_mixture = torch.zeros_like(mixture)
                output = prior + base_scaled
                balance = prior.new_zeros(())
            elif self.variant == "M1_direct_moe":
                scaled_mixture = strength.unsqueeze(-1) * mixture
                output = prior + scaled_mixture
                if active_count:
                    importance = route["dense_probs"][active].mean(dim=0)
                    selection_share = route["selected_mask"][active].float().mean(dim=0) / self.top_k
                    balance = self.num_experts * (importance * selection_share).sum()
                else:
                    importance = prior.new_zeros((self.num_experts,))
                    selection_share = importance
                    balance = prior.new_zeros(())
                route["importance"] = importance
                route["selection_share"] = selection_share
                modality_balance[modality] = balance
            else:
                rho = self.residual_max * strength
                scaled_mixture = rho.unsqueeze(-1) * mixture
                output = prior + base_scaled + scaled_mixture
                if active_count:
                    importance = route["dense_probs"][active].mean(dim=0)
                    selection_share = route["selected_mask"][active].float().mean(dim=0) / self.top_k
                    balance = self.num_experts * (importance * selection_share).sum()
                else:
                    importance = prior.new_zeros((self.num_experts,))
                    selection_share = importance
                    balance = prior.new_zeros(())
                route["importance"] = importance
                route["selection_share"] = selection_share
                route["rho"] = rho
                modality_balance[modality] = balance

            outputs[modality] = output
            router_info[modality] = route
            expert_inputs[modality] = profiles
            expert_outputs[modality] = expert_values
            modality_summaries[modality] = {
                "strength_mean": strength.detach().mean(),
                "route_entropy_mean": (
                    -(route["dense_probs"].clamp_min(1.0e-12)
                    * route["dense_probs"].clamp_min(1.0e-12).log()).sum(dim=-1).mean().detach()
                ),
                "base_gate": base_gate.detach(),
            }
            details[modality] = {
                "prior": prior,
                "basis": basis,
                "beta": beta,
                "base": base,
                "base_scaled": base_scaled,
                "base_gate": base_gate.reshape(()),
                "alpha": alpha,
                "expert_inputs": profiles,
                "expert_outputs": expert_values,
                "mixture": mixture,
                "scaled_mixture": scaled_mixture,
                "output": output,
                **route,
            }

        if self.variant == "M0_anchor":
            balance_loss = prior.new_zeros(())
        else:
            balance_loss = 0.5 * (modality_balance["text"] + modality_balance["visual"])
        aux_loss = self.balance_weight * balance_loss
        fused = torch.cat([outputs["text"], outputs["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
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
