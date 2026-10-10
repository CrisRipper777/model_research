from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops

from src.models.mvcge_mag_v21 import _ExpertTransform, _IntrinsicProjector


_MODALITIES = ("text", "visual")
_VARIANTS = ("a0_static", "a1_node", "a2_cross", "a3_intra")


class _ExpertAttentionDirection(nn.Module):
    """Attention from a selected source expert set to selected target experts."""

    def __init__(self, hidden_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.value = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, hidden_dim, bias=False)
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)


class ExpertCollaboration(nn.Module):
    """Node-local, selected-expert attention with an explicit Null source.

    Inputs use [N, experts, hidden] expert tensors and [N, TopK] route indices
    and normalized weights. Only the selected source and target expert tokens
    participate; no node-to-node attention is formed.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.eps = 1.0e-8
        self.directions = nn.ModuleDict(
            {
                modality: _ExpertAttentionDirection(
                    self.hidden_dim, self.attention_dim, dropout
                )
                for modality in _MODALITIES
            }
        )
        self.expert_pair_bias = nn.Parameter(torch.zeros(num_experts, num_experts))
        with torch.no_grad():
            self.expert_pair_bias.fill_diagonal_(0.1)

    @staticmethod
    def _gather_experts(expert_values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        hidden_dim = expert_values.size(-1)
        return expert_values.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )

    def _direction(
        self,
        target_modality: str,
        target_experts: torch.Tensor,
        target_indices: torch.Tensor,
        target_weights: torch.Tensor,
        source_experts: torch.Tensor,
        source_indices: torch.Tensor,
        source_weights: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        module = self.directions[target_modality]
        target_tokens = self._gather_experts(target_experts, target_indices)
        source_tokens = self._gather_experts(source_experts, source_indices)

        query = module.query(target_tokens)
        key = module.key(source_tokens)
        value = module.value(source_tokens)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.attention_dim
        )
        pair_bias = self.expert_pair_bias[
            target_indices.unsqueeze(-1), source_indices.unsqueeze(1)
        ]
        logits = logits + pair_bias
        source_log_weights = (source_weights + self.eps).log()
        logits = logits + source_log_weights.unsqueeze(1)
        null_logits = module.null_logit.expand(
            target_indices.size(0), self.top_k, 1
        )
        logits = torch.cat([logits, null_logits], dim=-1)

        # The appended Null Value is exactly zero. Bias-free Value and Output
        # projections prevent a rejected/empty source from creating a signal.
        null_value = value.new_zeros((value.size(0), 1, self.attention_dim))
        values = torch.cat([value, null_value], dim=1)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, values)
        per_target_correction = module.dropout(module.output(attended))
        per_target_correction = per_target_correction * active.to(
            per_target_correction.dtype
        ).view(-1, 1, 1)
        correction = (
            per_target_correction * target_weights.unsqueeze(-1)
        ).sum(dim=1)
        return {
            "target_tokens": target_tokens,
            "source_tokens": source_tokens,
            "attention": attention,
            "per_target_correction": per_target_correction,
            "correction": correction,
            "target_indices": target_indices,
            "source_indices": source_indices,
            "target_weights": target_weights,
            "source_weights": source_weights,
        }

    def forward(
        self,
        expert_values: dict[str, torch.Tensor],
        routes: dict[str, dict[str, torch.Tensor]],
        active: torch.Tensor,
        *,
        variant: str,
    ) -> dict[str, dict[str, torch.Tensor]]:
        results: dict[str, dict[str, torch.Tensor]] = {}
        for target in _MODALITIES:
            source = (
                target
                if variant == "a3_intra"
                else ("visual" if target == "text" else "text")
            )
            target_route = routes[target]
            source_route = routes[source]
            results[target] = self._direction(
                target,
                expert_values[target],
                target_route["top_indices"],
                target_route["top_weights"],
                expert_values[source],
                source_route["top_indices"],
                source_route["top_weights"],
                active,
            )
            results[target]["source_modality"] = source
        return results


class Model(nn.Module):
    """MACE-MAG V6A: shared Raw structural experts with selected-token collaboration."""

    requires_full_lp_sampler_depth = True
    record_run_telemetry = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("mace_mag_v6a requires positive Text and Visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError("mace_mag_v6a requires exact [text, visual] feature layout")

        self.variant = str(model_cfg.get("variant", "a2_cross"))
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.trajectory_order = int(model_cfg.get("trajectory_order", 4))
        self.num_experts = int(model_cfg.get("num_experts", 4))
        self.top_k = int(model_cfg.get("top_k", 2))
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.expert_feature_scale = float(model_cfg.get("expert_feature_scale", 0.1))
        self.router_dim = int(model_cfg.get("router_dim", 64))
        self.router_hidden_dim = int(model_cfg.get("router_hidden_dim", 128))
        self.modality_embed_dim = int(model_cfg.get("modality_embed_dim", 16))
        self.attention_dim = int(model_cfg.get("attention_dim", 64))
        self.attention_heads = int(model_cfg.get("attention_heads", 1))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.balance_weight = float(model_cfg.get("balance_weight", 0.01))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.structural_strength_init = float(
            model_cfg.get("structural_strength_init", 0.25)
        )
        self.collaboration_strength_init = float(
            model_cfg.get("collaboration_strength_init", 0.15)
        )
        if (
            self.hidden_dim,
            self.trajectory_order,
            self.num_experts,
            self.top_k,
        ) != (256, 4, 4, 2):
            raise ValueError(
                "V6A freezes hidden_dim=256, trajectory_order=4, experts=4, TopK=2"
            )
        if self.attention_dim != 64 or self.attention_heads != 1:
            raise ValueError("V6A freezes attention_dim=64 and attention_heads=1")
        if not (0.0 < self.structural_strength_init < 1.0):
            raise ValueError("structural_strength_init must be in (0, 1)")
        if not (0.0 < self.collaboration_strength_init < 1.0):
            raise ValueError("collaboration_strength_init must be in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        self.out_dim = self.hidden_dim
        self._gradient_diagnostics: dict[str, dict[str, float | bool]] | None = None

        # Shared trunk construction order is identical for all four variants.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(
                    self.text_dim, self.hidden_dim, self.dropout_p
                ),
                "visual": _IntrinsicProjector(
                    self.visual_dim, self.hidden_dim, self.dropout_p
                ),
            }
        )
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
                _ExpertTransform(
                    self.hidden_dim, self.expert_bottleneck, self.dropout_p
                )
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
        self.structural_strength_raw = nn.Parameter(
            torch.full(
                (2,),
                math.log(self.structural_strength_init / (1.0 - self.structural_strength_init)),
                dtype=torch.float32,
            )
        )
        self.collaboration_strength_raw = nn.Parameter(
            torch.full(
                (2,),
                math.log(self.collaboration_strength_init / (1.0 - self.collaboration_strength_init)),
                dtype=torch.float32,
            )
        )
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # Attention is constructed from an isolated RNG stream. It neither
        # changes the common trunk initialization nor the NC classifier seed.
        with torch.random.fork_rng(devices=[]):
            self.collaboration = ExpertCollaboration(
                hidden_dim=self.hidden_dim,
                attention_dim=self.attention_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                dropout=self.dropout_p,
            )
        uses_collaboration = self.variant in {"a2_cross", "a3_intra"}
        for parameter in self.collaboration.parameters():
            parameter.requires_grad_(uses_collaboration)
        self.collaboration_strength_raw.requires_grad_(uses_collaboration)

    def capture_gradient_diagnostics(self) -> dict[str, dict[str, float | bool]]:
        """Record first-backward gradient activity for the frozen campaign."""
        names = [
            "alpha_raw",
            "selection_head.weight",
            "experts.0.linear1.weight",
            "collaboration.expert_pair_bias",
            "collaboration_strength_raw",
            "collaboration.directions.text.query.weight",
            "collaboration.directions.text.key.weight",
            "collaboration.directions.text.value.weight",
            "collaboration.directions.text.output.weight",
            "collaboration.directions.visual.query.weight",
            "collaboration.directions.visual.key.weight",
            "collaboration.directions.visual.value.weight",
            "collaboration.directions.visual.output.weight",
        ]
        named = dict(self.named_parameters())
        result: dict[str, dict[str, float | bool]] = {}
        for name in names:
            parameter = named[name]
            grad = parameter.grad
            norm = float(grad.detach().norm().item()) if grad is not None else 0.0
            result[name] = {
                "gradient_present": grad is not None,
                "gradient_finite": bool(grad is None or torch.isfinite(grad).all()),
                "gradient_norm": norm,
                "gradient_nonzero": bool(norm > 0.0),
            }
        self._gradient_diagnostics = result
        return result

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"expected x shape [nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    @staticmethod
    def _normalized_operator(
        edge_index: torch.Tensor | None, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if edge_index is None:
            raise ValueError("mace_mag_v6a requires edge_index")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        edge_index = edge_index.to(dtype=torch.long)
        input_loops = int((edge_index[0] == edge_index[1]).sum().item())
        edge_index, _ = remove_self_loops(edge_index)
        src, dst = edge_index
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(
            0,
            dst,
            torch.ones(dst.numel(), dtype=dtype, device=edge_index.device),
        )
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        norm = inv_sqrt[src] * inv_sqrt[dst]
        return src, dst, norm, degree > 0, input_loops

    @staticmethod
    def _propagate(
        state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(state).index_add(
            0, dst, state[src] * norm.unsqueeze(-1)
        )

    def _propagate_states(
        self,
        state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        states = []
        for _ in range(self.trajectory_order):
            state = self._propagate(state, src, dst, norm)
            state = state * active.to(state.dtype).unsqueeze(-1)
            states.append(state)
        return torch.stack(states, dim=0)

    def _active_rms_norm(
        self, value: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(value)
        if bool(active.any()):
            selected = value[active]
            rms = torch.sqrt(selected.square().mean(dim=0) + self.eps)
            output[active] = selected / rms
        return output

    @staticmethod
    def _effective_alpha(raw: torch.Tensor, eps: float) -> torch.Tensor:
        return raw / (raw.norm(p=2, dim=-1, keepdim=True) + eps)

    def _router_inputs(
        self,
        modality_index: int,
        router_state: torch.Tensor,
        local: torch.Tensor,
        global_single: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = router_state.size(0)
        modality = self.modality_embedding.weight[modality_index].unsqueeze(0)
        global_nodes = global_single.expand(n, -1)
        modality_nodes = modality.expand(n, -1)
        zero_single = torch.zeros_like(global_single)
        static_input = torch.cat(
            [zero_single, zero_single, global_single, zero_single, modality], dim=-1
        )
        node_input = torch.cat(
            [router_state, local, global_nodes, torch.zeros_like(global_nodes), modality_nodes],
            dim=-1,
        )
        return static_input, node_input

    def _route(
        self,
        modality_index: int,
        static_input: torch.Tensor,
        node_input: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Match V2.2b D0/D1 call order to make the regression attributable.
        if self.variant == "a0_static":
            h_static = self.context_encoder(static_input)
            h_selection = h_static.expand(node_input.size(0), -1)
            logits_single = self.selection_head(h_static)
            logits = logits_single.expand(node_input.size(0), -1)
        else:
            h_selection = self.context_encoder(node_input)
            h_static = self.context_encoder(static_input)
            logits = self.selection_head(h_selection)
            logits_single = self.selection_head(h_static)
        dense_probs = torch.softmax(logits, dim=-1)
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        route_weights = torch.zeros_like(dense_probs).scatter(
            1, top_indices, top_weights
        )
        selected_mask = torch.zeros_like(dense_probs, dtype=torch.bool).scatter(
            1, top_indices, True
        )
        if bool(active.any()):
            importance = dense_probs[active].mean(dim=0)
            selection_share = selected_mask[active].float().mean(dim=0) / self.top_k
            balance = self.num_experts * (importance * selection_share).sum()
        else:
            importance = logits.new_zeros((self.num_experts,))
            selection_share = importance
            # Keep a finite, differentiable zero auxiliary scalar for an
            # all-isolated graph while contributing no router gradient.
            balance = logits.sum() * 0.0
        strength_raw = self.structural_strength_raw[modality_index]
        strength = torch.sigmoid(strength_raw).expand(node_input.size(0))
        return {
            "router_input": static_input if self.variant == "a0_static" else node_input,
            "static_input": static_input,
            "node_input": node_input,
            "encoded_selection": h_selection,
            "selection_logits": logits,
            "static_logits": logits_single.expand(node_input.size(0), -1),
            "dense_probs": dense_probs,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "route_weights": route_weights,
            "selected_mask": selected_mask,
            "strength_raw": strength_raw,
            "strength": strength,
            "importance": importance,
            "selection_share": selection_share,
            "balance": balance,
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
        collaboration_scale: float | torch.Tensor | None = None,
    ):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, norm, active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )

        bases: dict[str, torch.Tensor] = {}
        router_states: dict[str, torch.Tensor] = {}
        locals_: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            raw = self._propagate_states(priors[modality], src, dst, norm, active)
            bases[modality] = torch.stack(
                [self._active_rms_norm(state, active) for state in raw], dim=0
            )
            router_states[modality] = self.router_norms[modality](
                self.router_projectors[modality](priors[modality])
            )
            locals_[modality] = self._propagate(
                router_states[modality], src, dst, norm
            )

        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        active_count = int(active.sum().item())
        global_contexts = {}
        for modality in _MODALITIES:
            if active_count:
                global_contexts[modality] = router_states[modality][active].mean(
                    dim=0, keepdim=True
                )
            else:
                global_contexts[modality] = router_states[modality].new_zeros(
                    (1, self.router_dim)
                )

        expert_values: dict[str, torch.Tensor] = {}
        routes: dict[str, dict[str, torch.Tensor]] = {}
        modality_balance: dict[str, torch.Tensor] = {}
        modality_summaries: dict[str, dict[str, torch.Tensor]] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        for modality_index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            basis = bases[modality]
            profiles = torch.einsum("mk,knd->mnd", alpha, basis)
            transformed = torch.stack(
                [
                    expert(profiles[expert_id])
                    for expert_id, expert in enumerate(self.experts)
                ],
                dim=0,
            )
            values = profiles + self.expert_feature_scale * transformed
            expert_values[modality] = values.permute(1, 0, 2)

            static_input, node_input = self._router_inputs(
                modality_index,
                router_states[modality],
                locals_[modality],
                global_contexts[modality],
            )
            route = self._route(modality_index, static_input, node_input, active)
            routes[modality] = route
            modality_balance[modality] = route["balance"]
            modality_summaries[modality] = {
                "strength_mean": route["strength"].detach().mean(),
                "route_entropy_mean": (
                    -(
                        route["dense_probs"].clamp_min(1.0e-12)
                        * route["dense_probs"].clamp_min(1.0e-12).log()
                    )
                    .sum(dim=-1)
                    .mean()
                    .detach()
                ),
            }
            if return_details:
                details[modality] = {
                    "prior": prior,
                    "basis": basis,
                    "alpha": alpha,
                    "expert_inputs": profiles,
                    "expert_outputs": values,
                    "router_state": router_states[modality],
                    "local_router_state": locals_[modality],
                    **route,
                }

        balance_loss = 0.5 * (modality_balance["text"] + modality_balance["visual"])
        aux_loss = self.balance_weight * balance_loss
        if self.variant in {"a2_cross", "a3_intra"}:
            collaboration_details = self.collaboration(
                expert_values, routes, active, variant=self.variant
            )
            corrections = {
                m: collaboration_details[m]["correction"] for m in _MODALITIES
            }
            if collaboration_scale is None:
                scale = 1.0
            else:
                scale = collaboration_scale
        else:
            collaboration_details = {}
            corrections = {
                m: priors[m].new_zeros(priors[m].shape) for m in _MODALITIES
            }
            scale = 0.0

        outputs: dict[str, torch.Tensor] = {}
        for modality_index, modality in enumerate(_MODALITIES):
            route = routes[modality]
            mixture = torch.einsum(
                "nm,nmd->nd", route["route_weights"], expert_values[modality]
            )
            structural = route["strength"].unsqueeze(-1) * mixture
            if scale == 0.0:
                collaborative = torch.zeros_like(corrections[modality])
            else:
                collaborative = (
                    torch.sigmoid(self.collaboration_strength_raw[modality_index])
                    * scale
                    * corrections[modality]
                )
            outputs[modality] = priors[modality] + structural + collaborative
            modality_summaries[modality]["collaboration_strength"] = (
                torch.sigmoid(self.collaboration_strength_raw[modality_index])
                .detach()
                if self.variant in {"a2_cross", "a3_intra"}
                else priors[modality].new_zeros(())
            )

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
            info["details"] = {
                "modalities": details,
                "collaboration": collaboration_details,
                "corrections": corrections,
                "expert_values": expert_values,
                "routes": routes,
                "outputs": outputs,
                "active": active,
            }
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
