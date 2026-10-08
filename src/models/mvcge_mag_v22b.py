from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops

from src.models.mvcge_mag_v21 import _ExpertTransform, _IntrinsicProjector


_MODALITIES = ("text", "visual")
_VARIANTS = ("D0_static", "D1_free_node", "D2_struct_free", "D3_struct_residual")


class Model(nn.Module):
    """MvCGE-MAG V2.2b strength-decoupled shared-expert router.

    The intrinsic paths, RawPoly basis, expert bank, Top-2 mechanism, and late
    fusion are inherited verbatim in construction order from V2.2. Strength
    is a direct modality scalar independent of router/evidence computations.
    D2 uses structure-grounded free-node logits; D3 adds only static-centered
    residualization. Edge reliability remains router evidence and never
    changes physical propagation.
    """

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("mvcge_mag_v22b requires positive text and visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError("mvcge_mag_v22b requires exact [text, visual] feature layout")

        self.variant = str(model_cfg.get("variant", "D2_struct_free"))
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.trajectory_order = int(model_cfg.get("trajectory_order", 4))
        self.num_experts = int(model_cfg.get("num_experts", 4))
        self.top_k = int(model_cfg.get("top_k", 2))
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.router_dim = int(model_cfg.get("router_dim", 64))
        self.router_hidden_dim = int(model_cfg.get("router_hidden_dim", 128))
        self.modality_embed_dim = int(model_cfg.get("modality_embed_dim", 16))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.expert_feature_scale = float(model_cfg.get("expert_feature_scale", 0.1))
        self.balance_weight = float(model_cfg.get("balance_weight", 0.01))
        self.legacy_strength_init = -1.15
        self.direct_strength_init = float(model_cfg.get("direct_strength_init", -1.15))
        self.evidence_scalar_dim = int(model_cfg.get("evidence_scalar_dim", 11))
        self.expert_key_embed_dim = int(model_cfg.get("expert_key_embed_dim", 16))
        self.compatibility_dim = int(model_cfg.get("compatibility_dim", 64))
        self.eta_init = float(model_cfg.get("node_residual_init", 0.10))
        self.kappa_init = float(model_cfg.get("compatibility_residual_init", 0.10))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        if (self.hidden_dim, self.trajectory_order, self.num_experts, self.top_k) != (
            256,
            4,
            4,
            2,
        ):
            raise ValueError(
                "V2.2b fixes hidden_dim=256, trajectory_order=4, M=4, TopK=2"
            )
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if (self.evidence_scalar_dim, self.expert_key_embed_dim, self.compatibility_dim) != (
            11,
            16,
            64,
        ):
            raise ValueError("V2.2b fixes evidence/key/compatibility dimensions to 11/16/64")
        if self.legacy_strength_init != -1.15 or self.direct_strength_init != -1.15:
            raise ValueError("V2.2b fixes legacy and direct strength initialization to -1.15")
        if not 0.0 < self.eta_init < 1.0 or not 0.0 < self.kappa_init < 1.0:
            raise ValueError("eta_init and kappa_init must be between zero and one")
        self.out_dim = self.hidden_dim

        # Preserve V2.1's module construction order and initialization exactly.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
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

        nn.init.zeros_(self.strength_head.weight)
        nn.init.constant_(self.strength_head.bias, self.legacy_strength_init)

        # New module initialization must not advance the global RNG: the NC
        # classifier is initialized immediately after Model by the task runner.
        # Every variant constructs these modules in this same forked stream.
        with torch.random.fork_rng(devices=[]):
            self.evidence_scalar_projector = nn.Linear(
                self.evidence_scalar_dim, self.router_dim
            )
            self.evidence_norm = nn.LayerNorm(self.router_dim)
            self.expert_key_embedding = nn.Parameter(
                torch.empty(4, self.expert_key_embed_dim)
            )
            nn.init.xavier_uniform_(self.expert_key_embedding)
            self.expert_query_proj = nn.Linear(
                self.router_hidden_dim, self.compatibility_dim
            )
            self.expert_key_proj = nn.Linear(
                4 + self.expert_key_embed_dim, self.compatibility_dim
            )

        eta_raw_init = math.log(self.eta_init / (1.0 - self.eta_init))
        kappa_raw_init = math.log(self.kappa_init / (1.0 - self.kappa_init))
        self.node_residual_raw = nn.Parameter(
            torch.full((2,), eta_raw_init, dtype=torch.float32)
        )
        self.compatibility_residual_raw = nn.Parameter(
            torch.full((2,), kappa_raw_init, dtype=torch.float32)
        )
        # Constant construction is deterministic and does not consume RNG.
        # Index 0 is text and index 1 is visual.
        self.direct_strength_raw = nn.Parameter(
            torch.full((2,), self.direct_strength_init, dtype=torch.float32)
        )
        self._set_new_parameter_activity()

    def _set_new_parameter_activity(self) -> None:
        evidence_active = self.variant in {
            "D2_struct_free",
            "D3_struct_residual",
        }
        residual_active = self.variant == "D3_struct_residual"
        for module in (self.evidence_scalar_projector, self.evidence_norm):
            for parameter in module.parameters():
                parameter.requires_grad_(evidence_active)
        self.node_residual_raw.requires_grad_(residual_active)
        self.strength_head.requires_grad_(False)
        self.direct_strength_raw.requires_grad_(True)
        for parameter in (
            self.expert_key_embedding,
            *self.expert_query_proj.parameters(),
            *self.expert_key_proj.parameters(),
        ):
            parameter.requires_grad_(False)
        self.compatibility_residual_raw.requires_grad_(False)

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
            raise ValueError("mvcge_mag_v22 requires edge_index")
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

    @staticmethod
    def _edge_reliability(
        router_state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        norm_sq = router_state.square().sum(dim=-1, keepdim=True)
        safe_norm_sq = torch.where(norm_sq > 0, norm_sq, torch.ones_like(norm_sq))
        norm = torch.where(norm_sq > 0, safe_norm_sq.sqrt(), torch.zeros_like(norm_sq))
        normalized = router_state / (norm + eps)
        return (0.5 * (1.0 + (normalized[src] * normalized[dst]).sum(dim=-1))).clamp(
            0.0, 1.0
        )

    def _structural_evidence(
        self,
        prior: torch.Tensor,
        router_state: torch.Tensor,
        basis: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n = prior.size(0)
        reliability = self._edge_reliability(router_state, src, dst, self.eps)
        count = prior.new_zeros((n,))
        rel_sum = prior.new_zeros((n,)).index_add(0, dst, reliability)
        rel_sq_sum = prior.new_zeros((n,)).index_add(0, dst, reliability.square())
        count.index_add_(0, dst, torch.ones_like(reliability))
        mu = rel_sum / count.clamp_min(1.0)
        variance = (rel_sq_sum / count.clamp_min(1.0) - mu.square()).clamp_min(0.0)
        positive_variance = variance > 0
        safe_variance = torch.where(positive_variance, variance, torch.ones_like(variance))
        sigma = torch.where(
            positive_variance, safe_variance.sqrt(), torch.zeros_like(variance)
        )
        l_rel_sum = prior.new_zeros(router_state.shape).index_add(
            0, dst, router_state[src] * reliability.unsqueeze(-1)
        )
        has_reliable_neighbor = (count > 0) & (rel_sum > self.eps)
        l_rel = l_rel_sum / rel_sum.clamp_min(self.eps).unsqueeze(-1)
        l_rel = l_rel * has_reliable_neighbor.to(l_rel.dtype).unsqueeze(-1)

        log_degree = torch.log1p(count)
        degree_norm = prior.new_zeros((n,))
        if bool(active.any()):
            degree_norm[active] = log_degree[active] / (log_degree[active].max() + self.eps)

        hop_cosines = []
        hop_displacements = []
        prior_norm_sq = prior.square().sum(dim=-1)
        prior_norm = torch.where(
            prior_norm_sq > 0,
            torch.where(prior_norm_sq > 0, prior_norm_sq, torch.ones_like(prior_norm_sq)).sqrt(),
            torch.zeros_like(prior_norm_sq),
        ) + self.eps
        for order in range(self.trajectory_order):
            state = basis[order]
            hop_cosines.append(F.cosine_similarity(prior, state, dim=-1, eps=self.eps))
            delta_sq = (state - prior).square().sum(dim=-1)
            safe_delta_sq = torch.where(delta_sq > 0, delta_sq, torch.ones_like(delta_sq))
            delta_norm = torch.where(
                delta_sq > 0, safe_delta_sq.sqrt(), torch.zeros_like(delta_sq)
            )
            displacement = delta_norm / prior_norm
            hop_displacements.append(torch.log1p(displacement))
        hop_cos = torch.stack(hop_cosines, dim=-1)
        hop_disp = torch.stack(hop_displacements, dim=-1)
        active_float = active.to(prior.dtype).unsqueeze(-1)
        hop_cos = hop_cos * active_float
        hop_disp = hop_disp * active_float
        scalar = torch.cat(
            [mu.unsqueeze(-1), sigma.unsqueeze(-1), degree_norm.unsqueeze(-1), hop_cos, hop_disp],
            dim=-1,
        )
        evidence = self.evidence_norm(l_rel + self.evidence_scalar_projector(scalar))
        # LayerNorm has affine parameters; retain an exact zero evidence vector
        # for isolated nodes even after the evidence projector has trained.
        evidence = evidence * active.to(evidence.dtype).unsqueeze(-1)
        return {
            "reliability": reliability,
            "neighbor_count": count,
            "mu": mu,
            "sigma": sigma,
            "degree_norm": degree_norm,
            "hop_cos": hop_cos,
            "hop_disp": hop_disp,
            "scalar_evidence": scalar,
            "L_rel": l_rel,
            "EVID": evidence,
        }

    def _static_and_node_inputs(
        self,
        modality_index: int,
        u: torch.Tensor,
        local: torch.Tensor,
        global_single: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        n = u.size(0)
        embedding_single = self.modality_embedding.weight[modality_index].unsqueeze(0)
        global_nodes = global_single.expand(n, -1)
        embedding_nodes = embedding_single.expand(n, -1)
        zeros = torch.zeros_like(global_single)
        static_input = torch.cat(
            [zeros, zeros, global_single, zeros, embedding_single], dim=-1
        )
        fourth = evidence if self.variant in {"D2_struct_free", "D3_struct_residual"} else torch.zeros_like(u)
        node_input = torch.cat(
            [u, local, global_nodes, fourth, embedding_nodes], dim=-1
        )
        blocks = {
            "ego": u,
            "local": local,
            "global_single": global_single,
            "global_nodes": global_nodes,
            # V2.2 has no cross-modal router input. This legacy alias makes
            # that invariant visible in details while `fourth` is structural.
            "cross": torch.zeros_like(u),
            "evidence": evidence,
            "fourth": fourth,
            "modality_embedding": embedding_nodes,
            "static_input": static_input,
            "node_input": node_input,
        }
        return static_input, node_input, blocks

    @staticmethod
    def _top2(logits: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
        dense_probs = torch.softmax(logits, dim=-1)
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        route_weights = torch.zeros_like(dense_probs).scatter(1, top_indices, top_weights)
        selected_mask = torch.zeros_like(dense_probs, dtype=torch.bool).scatter(
            1, top_indices, True
        )
        return {
            "dense_probs": dense_probs,
            "top_indices": top_indices,
            "route_weights": route_weights,
            "selected_mask": selected_mask,
        }

    def _route(
        self,
        modality_index: int,
        static_input: torch.Tensor,
        node_input: torch.Tensor,
        blocks: dict[str, torch.Tensor],
        alpha: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Preserve V2.2's call order exactly for D0/D1 initial regressions.
        if self.variant == "D0_static":
            h_static = self.context_encoder(static_input)
            h_struct = h_static.expand(node_input.size(0), -1)
            static_logits = self.selection_head(h_static)
            structure_free_logits = static_logits.expand(node_input.size(0), -1)
        elif self.variant == "D1_free_node":
            h_struct = self.context_encoder(node_input)
            h_static = self.context_encoder(static_input)
            static_logits = self.selection_head(h_static)
            structure_free_logits = self.selection_head(h_struct)
        else:
            h_struct = self.context_encoder(node_input)
            h_static = self.context_encoder(static_input)
            static_logits = self.selection_head(h_static)
            structure_free_logits = self.selection_head(h_struct)

        static_logits_nodes = static_logits.expand(node_input.size(0), -1)
        eta_value = (
            torch.sigmoid(self.node_residual_raw[modality_index])
            if self.variant == "D3_struct_residual"
            else self.direct_strength_raw.new_zeros(())
        )
        if self.variant == "D0_static":
            logits = static_logits_nodes
        elif self.variant in {"D1_free_node", "D2_struct_free"}:
            logits = structure_free_logits
        else:
            logits = static_logits_nodes + eta_value * (
                structure_free_logits - static_logits_nodes
            )

        static_top = self._top2(static_logits_nodes, self.top_k)
        free_top = self._top2(structure_free_logits, self.top_k)
        top = self._top2(logits, self.top_k)
        # This direct modality scalar is the only source of strength. The
        # frozen legacy strength_head is intentionally unused by forward.
        strength_logit_single = self.direct_strength_raw[modality_index]
        strength_single = torch.sigmoid(strength_logit_single)
        strength_logit = strength_logit_single.expand(node_input.size(0))
        strength = strength_single.expand(node_input.size(0))
        if bool(active.any()):
            dense_importance = top["dense_probs"][active].mean(dim=0)
            selection_share = top["selected_mask"][active].float().mean(dim=0) / self.top_k
            balance = self.num_experts * (dense_importance * selection_share).sum()
        else:
            dense_importance = logits.new_zeros((self.num_experts,))
            selection_share = dense_importance
            balance = logits.new_zeros(())
        return {
            "router_input": static_input if self.variant == "D0_static" else node_input,
            "router_blocks": blocks,
            "encoded_selection": h_struct,
            "encoded_strength": h_static,
            "selection_logits": logits,
            "static_logits": static_logits_nodes,
            "structure_free_logits": structure_free_logits,
            "eta": eta_value,
            **top,
            "static_route_weights": static_top["route_weights"],
            "static_dense_probs": static_top["dense_probs"],
            "structure_free_route_weights": free_top["route_weights"],
            "structure_free_dense_probs": free_top["dense_probs"],
            "strength_logit": strength_logit,
            "strength_raw": strength_logit_single,
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
        src, dst, norm, active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )
        bases: dict[str, torch.Tensor] = {}
        router_states: dict[str, torch.Tensor] = {}
        locals_: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            states = []
            state = priors[modality]
            for _ in range(self.trajectory_order):
                state = self._propagate(state, src, dst, norm)
                state = state * active.to(state.dtype).unsqueeze(-1)
                states.append(state)
            bases[modality] = torch.stack(
                [self._active_rms_norm(value, active) for value in states], dim=0
            )
            router_states[modality] = self.router_norms[modality](
                self.router_projectors[modality](priors[modality])
            )
            locals_[modality] = self._propagate(router_states[modality], src, dst, norm)

        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        active_count = int(active.sum().item())
        global_contexts: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            if active_count:
                global_contexts[modality] = router_states[modality][active].mean(
                    dim=0, keepdim=True
                )
            else:
                global_contexts[modality] = router_states[modality].new_zeros(
                    (1, self.router_dim)
                )

        outputs: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        modality_balance: dict[str, torch.Tensor] = {}
        modality_summaries: dict[str, dict[str, torch.Tensor]] = {}
        router_info: dict[str, dict[str, torch.Tensor]] = {}
        for index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            basis = bases[modality]
            profiles = torch.einsum("mk,knd->mnd", alpha, basis)
            transformed = torch.stack(
                [expert(profiles[j]) for j, expert in enumerate(self.experts)], dim=0
            )
            expert_values = profiles + self.expert_feature_scale * transformed
            if self.variant in {"D2_struct_free", "D3_struct_residual"}:
                evidence_info = self._structural_evidence(
                    prior, router_states[modality], basis, src, dst, active
                )
                evidence = evidence_info["EVID"]
            else:
                evidence_info = {
                    name: prior.new_zeros((prior.size(0),))
                    for name in ("mu", "sigma", "degree_norm")
                }
                evidence_info.update(
                    {
                        "reliability": prior.new_zeros((src.numel(),)),
                        "neighbor_count": prior.new_zeros((prior.size(0),)),
                        "hop_cos": prior.new_zeros((prior.size(0), self.trajectory_order)),
                        "hop_disp": prior.new_zeros((prior.size(0), self.trajectory_order)),
                        "scalar_evidence": prior.new_zeros(
                            (prior.size(0), self.evidence_scalar_dim)
                        ),
                        "L_rel": prior.new_zeros((prior.size(0), self.router_dim)),
                        "EVID": prior.new_zeros((prior.size(0), self.router_dim)),
                    }
                )
                evidence = evidence_info["EVID"]
            static_input, node_input, blocks = self._static_and_node_inputs(
                index,
                router_states[modality],
                locals_[modality],
                global_contexts[modality],
                evidence,
            )
            route = self._route(index, static_input, node_input, blocks, alpha, active)
            mixture = torch.einsum("nm,mnd->nd", route["route_weights"], expert_values)
            scaled_mixture = route["strength"].unsqueeze(-1) * mixture
            output = prior + scaled_mixture
            modality_balance[modality] = route["balance"]
            outputs[modality] = output
            router_info[modality] = route
            modality_summaries[modality] = {
                "strength_mean": route["strength"].detach().mean(),
                "route_entropy_mean": (
                    -(route["dense_probs"].clamp_min(1.0e-12)
                    * route["dense_probs"].clamp_min(1.0e-12).log())
                    .sum(dim=-1)
                    .mean()
                    .detach()
                ),
            }
            details[modality] = {
                "prior": prior,
                "basis": basis,
                "alpha": alpha,
                "expert_inputs": profiles,
                "expert_outputs": expert_values,
                "mixture": mixture,
                "scaled_mixture": scaled_mixture,
                "output": output,
                **evidence_info,
                **route,
            }

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
