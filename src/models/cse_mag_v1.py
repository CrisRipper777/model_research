from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops


_MODALITIES = ("text", "visual")
_VARIANTS = {"independent", "shared_static", "mvcge_style", "v1"}


class _IntrinsicProjector(nn.Module):
    """Modality-specific intrinsic encoder shared by all screening variants."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.skip(x)
            + self.linear2(self.dropout(F.gelu(self.linear1(x))))
        )


class _ResponseExpert(nn.Module):
    """Small transform applied after a fixed structural response operator."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        # Bias-free transforms plus non-affine normalization guarantee that a
        # zero structural displacement remains exactly zero. This matters for
        # isolated nodes and for the protected-prior interpretation.
        self.linear1 = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.linear2 = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.norm(y)


class _Router(nn.Module):
    """Lightweight conditional composer with explicit output initialization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        *,
        gate_bias: float | None = None,
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        if gate_bias is not None:
            if output_dim < 1:
                raise ValueError("router output_dim must be positive")
            with torch.no_grad():
                self.linear2.bias[-1] = float(gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class Model(nn.Module):
    """CSE-MAG V1 screening model.

    The four variants share the same modality projectors and late-fusion head.

    independent:
        Modality-specific Local/Global response experts and routers with a
        protected intrinsic residual.
    shared_static:
        Shared Local/Global experts, but one graph-wide mixture and one gate
        shared by both modalities.
    mvcge_style:
        Direct MvCGE-style transfer: homogeneous shared graph experts and a
        shared node/modality Top-K router. This is a screening baseline, not an
        exact reproduction of the original MvCGE implementation.
    v1:
        Shared functional Local/Global experts, shared node/modality router,
        and protected modality-intrinsic residual integration.
    """

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("cse_mag_v1 requires positive text and visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "cse_mag_v1 requires exact [text, visual] feature layout: "
                f"input_dim={self.input_dim}, text_dim={self.text_dim}, "
                f"visual_dim={self.visual_dim}"
            )

        self.variant = str(model_cfg.get("variant", "v1")).strip().lower()
        if self.variant not in _VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(_VARIANTS)}, got {self.variant!r}"
            )

        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.router_hidden_dim = int(model_cfg.get("router_hidden_dim", 128))
        self.modality_embed_dim = int(model_cfg.get("modality_embed_dim", 16))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.global_order = int(model_cfg.get("global_order", 3))
        self.global_t = float(model_cfg.get("global_t", 1.0))
        self.gate_init_bias = float(model_cfg.get("gate_init_bias", -2.0))
        self.naive_num_experts = int(model_cfg.get("naive_num_experts", 4))
        self.naive_top_k = int(model_cfg.get("naive_top_k", 2))
        self.mvcge_balance_weight = float(
            model_cfg.get("mvcge_balance_weight", 0.01)
        )
        self.eps = float(model_cfg.get("eps", 1.0e-8))

        if self.hidden_dim <= 0 or self.expert_bottleneck <= 0:
            raise ValueError("hidden_dim and expert_bottleneck must be positive")
        if self.global_order < 1:
            raise ValueError("global_order must be >= 1")
        if self.global_t <= 0:
            raise ValueError("global_t must be positive")
        if self.naive_num_experts < 2:
            raise ValueError("naive_num_experts must be >= 2")
        if not 1 <= self.naive_top_k <= self.naive_num_experts:
            raise ValueError("naive_top_k must lie in [1, naive_num_experts]")
        if self.mvcge_balance_weight < 0:
            raise ValueError("mvcge_balance_weight must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

        # Keep num_layers visible to the common sampler contract.
        self.num_layers = self.global_order
        self.out_dim = self.hidden_dim

        # IMPORTANT: every screening variant instantiates modules in exactly the
        # same order. Inactive variant-specific modules are frozen afterwards.
        # This preserves common-module and downstream classifier initialization
        # under the same random seed.
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

        # Shared functional response experts used by shared_static and V1.
        self.shared_functional_experts = nn.ModuleDict(
            {
                "local": _ResponseExpert(
                    self.hidden_dim, self.expert_bottleneck, self.dropout_p
                ),
                "global": _ResponseExpert(
                    self.hidden_dim, self.expert_bottleneck, self.dropout_p
                ),
            }
        )

        # Independent functional response experts used by the independent control.
        self.independent_functional_experts = nn.ModuleDict(
            {
                modality: nn.ModuleDict(
                    {
                        "local": _ResponseExpert(
                            self.hidden_dim,
                            self.expert_bottleneck,
                            self.dropout_p,
                        ),
                        "global": _ResponseExpert(
                            self.hidden_dim,
                            self.expert_bottleneck,
                            self.dropout_p,
                        ),
                    }
                )
                for modality in _MODALITIES
            }
        )

        v1_router_dim = 3 * self.hidden_dim + self.modality_embed_dim
        independent_router_dim = 3 * self.hidden_dim
        self.modality_embedding = nn.Embedding(2, self.modality_embed_dim)
        self.shared_functional_router = _Router(
            v1_router_dim,
            self.router_hidden_dim,
            3,
            self.dropout_p,
            gate_bias=self.gate_init_bias,
        )
        self.independent_functional_routers = nn.ModuleDict(
            {
                modality: _Router(
                    independent_router_dim,
                    self.router_hidden_dim,
                    3,
                    self.dropout_p,
                    gate_bias=self.gate_init_bias,
                )
                for modality in _MODALITIES
            }
        )

        self.static_response_logits = nn.Parameter(torch.zeros(2))
        self.static_gate_logit = nn.Parameter(
            torch.tensor(self.gate_init_bias, dtype=torch.float32)
        )

        # Direct MvCGE-style transfer baseline: homogeneous experts over the
        # same local propagated state plus a graph-context-aware Top-K router.
        self.naive_experts = nn.ModuleList(
            [
                _ResponseExpert(
                    self.hidden_dim, self.expert_bottleneck, self.dropout_p
                )
                for _ in range(self.naive_num_experts)
            ]
        )
        naive_router_dim = 3 * self.hidden_dim + self.modality_embed_dim
        self.naive_router = _Router(
            naive_router_dim,
            self.router_hidden_dim,
            self.naive_num_experts,
            self.dropout_p,
        )
        self.naive_output_norm = nn.LayerNorm(
            self.hidden_dim, elementwise_affine=False
        )

        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        self._freeze_inactive_variant_parameters()

    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _freeze_inactive_variant_parameters(self) -> None:
        # Common modules are always trainable.
        self._set_trainable(self.projectors, True)
        self._set_trainable(self.fusion_linear1, True)
        self._set_trainable(self.fusion_linear2, True)
        self._set_trainable(self.fusion_skip, True)
        self._set_trainable(self.fusion_norm, True)

        shared_functional = self.variant in {"shared_static", "v1"}
        independent = self.variant == "independent"
        mvcge = self.variant == "mvcge_style"
        uses_modality_embedding = self.variant in {"v1", "mvcge_style"}

        self._set_trainable(self.shared_functional_experts, shared_functional)
        self._set_trainable(self.independent_functional_experts, independent)
        self._set_trainable(self.modality_embedding, uses_modality_embedding)
        self._set_trainable(self.shared_functional_router, self.variant == "v1")
        self._set_trainable(self.independent_functional_routers, independent)
        self.static_response_logits.requires_grad_(self.variant == "shared_static")
        self.static_gate_logit.requires_grad_(self.variant == "shared_static")
        self._set_trainable(self.naive_experts, mvcge)
        self._set_trainable(self.naive_router, mvcge)
        self._set_trainable(self.naive_output_norm, mvcge)

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"cse_mag_v1 expected x shape [nodes, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )
        if x.dtype == torch.long:
            x = x.float()
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    def _normalized_operator(
        self,
        edge_index: torch.Tensor | None,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("cse_mag_v1 requires edge_index")
        edge_index = edge_index.to(dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
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
        active = degree > 0
        return src, dst, norm, active

    @staticmethod
    def _propagate(
        state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        messages = state[src] * norm.unsqueeze(-1)
        return torch.zeros_like(state).index_add(0, dst, messages)

    def _structural_responses(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Local response: movement from intrinsic semantics toward one-hop
        # neighbor context. Isolated nodes receive no structural displacement.
        local_state = self._propagate(prior, src, dst, norm)
        local_delta = local_state - prior
        local_delta = local_delta * active.to(prior.dtype).unsqueeze(-1)

        # Global contrastive response: truncated e^{-tP} - I.
        # The alternating odd/even signs provide a PC-Conv-inspired
        # heterophily-aware multi-hop structural direction.
        global_delta = torch.zeros_like(prior)
        state = prior
        for order in range(1, self.global_order + 1):
            state = self._propagate(state, src, dst, norm)
            coefficient = ((-self.global_t) ** order) / float(math.factorial(order))
            global_delta = global_delta + coefficient * state
        return local_state, local_delta, global_delta

    def _modality_embedding(
        self,
        modality: str,
        num_nodes: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        index = 0 if modality == "text" else 1
        ids = torch.full((num_nodes,), index, dtype=torch.long, device=device)
        return self.modality_embedding(ids)

    def _functional_compose(
        self,
        modality: str,
        prior: torch.Tensor,
        local_delta: torch.Tensor,
        global_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.variant == "independent":
            expert_bank = self.independent_functional_experts[modality]
            router = self.independent_functional_routers[modality]
            query = torch.cat([prior, local_delta, global_delta], dim=-1)
            router_out = router(query)
        elif self.variant == "v1":
            expert_bank = self.shared_functional_experts
            modality_embed = self._modality_embedding(
                modality, prior.size(0), device=prior.device
            )
            query = torch.cat(
                [prior, local_delta, global_delta, modality_embed], dim=-1
            )
            router_out = self.shared_functional_router(query)
        else:
            raise AssertionError("_functional_compose called for unsupported variant")

        routing = F.softmax(router_out[:, :2], dim=-1)
        structural_gate = torch.sigmoid(router_out[:, 2:3])
        local_response = expert_bank["local"](local_delta)
        global_response = expert_bank["global"](global_delta)
        correction = (
            routing[:, 0:1] * local_response
            + routing[:, 1:2] * global_response
        )
        output = prior + structural_gate * correction
        return output, {
            "routing": routing,
            "gate": structural_gate,
            "local_response": local_response,
            "global_response": global_response,
        }

    def _shared_static_compose(
        self,
        prior: torch.Tensor,
        local_delta: torch.Tensor,
        global_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        routing = F.softmax(self.static_response_logits, dim=0)
        structural_gate = torch.sigmoid(self.static_gate_logit)
        local_response = self.shared_functional_experts["local"](local_delta)
        global_response = self.shared_functional_experts["global"](global_delta)
        correction = routing[0] * local_response + routing[1] * global_response
        output = prior + structural_gate * correction
        return output, {
            "routing": routing.unsqueeze(0).expand(prior.size(0), -1),
            "gate": structural_gate.reshape(1, 1).expand(prior.size(0), 1),
            "local_response": local_response,
            "global_response": global_response,
        }

    def _mvcge_style_compose(
        self,
        modality: str,
        prior: torch.Tensor,
        local_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        global_state = prior.mean(dim=0, keepdim=True).expand_as(prior)
        modality_embed = self._modality_embedding(
            modality, prior.size(0), device=prior.device
        )
        query = torch.cat(
            [prior, local_state, global_state, modality_embed], dim=-1
        )
        logits = self.naive_router(query)
        dense_probs = F.softmax(logits, dim=-1)

        top_values, top_indices = torch.topk(
            logits, k=self.naive_top_k, dim=-1, largest=True, sorted=False
        )
        top_weights = F.softmax(top_values, dim=-1)
        sparse_weights = torch.zeros_like(logits)
        sparse_weights.scatter_(1, top_indices, top_weights)

        expert_outputs = torch.stack(
            [expert(local_state) for expert in self.naive_experts], dim=1
        )
        mixed = (sparse_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        output = self.naive_output_norm(mixed)

        selected = torch.zeros_like(sparse_weights)
        selected.scatter_(1, top_indices, 1.0)
        importance = dense_probs.mean(dim=0)
        load = selected.mean(dim=0) / float(self.naive_top_k)
        balance_raw = self.naive_num_experts * torch.sum(importance * load)
        aux_loss = self.mvcge_balance_weight * balance_raw
        return output, aux_loss, {
            "dense_probs": dense_probs,
            "sparse_weights": sparse_weights,
            "top_indices": top_indices,
            "balance_raw": balance_raw,
        }

    @staticmethod
    def _mean_cosine(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
        a_norm = a.norm(dim=-1).clamp_min(eps)
        b_norm = b.norm(dim=-1).clamp_min(eps)
        return ((a * b).sum(dim=-1) / (a_norm * b_norm)).mean()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        inputs = self._split_modalities(x)
        priors = {
            modality: self.projectors[modality](inputs[modality])
            for modality in _MODALITIES
        }
        src, dst, norm, active = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )

        modality_outputs: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        aux_loss = priors["text"].new_zeros(())
        info: dict[str, Any] = {
            "variant": self.variant,
            "modalities": {},
        }

        for modality in _MODALITIES:
            prior = priors[modality]
            local_state, local_delta, global_delta = self._structural_responses(
                prior, src, dst, norm, active
            )

            if self.variant in {"independent", "v1"}:
                output, route_info = self._functional_compose(
                    modality, prior, local_delta, global_delta
                )
            elif self.variant == "shared_static":
                output, route_info = self._shared_static_compose(
                    prior, local_delta, global_delta
                )
            elif self.variant == "mvcge_style":
                output, modality_aux, route_info = self._mvcge_style_compose(
                    modality, prior, local_state
                )
                aux_loss = aux_loss + modality_aux
            else:
                raise AssertionError(f"unhandled variant {self.variant}")

            modality_outputs[modality] = output
            if self.variant == "mvcge_style":
                info["modalities"][modality] = {
                    "expert_prob_mean": route_info["dense_probs"].mean(dim=0).detach(),
                    "sparse_weight_mean": route_info["sparse_weights"].mean(dim=0).detach(),
                    "balance_raw": route_info["balance_raw"].detach(),
                }
            else:
                info["modalities"][modality] = {
                    "local_weight_mean": route_info["routing"][:, 0].mean().detach(),
                    "global_weight_mean": route_info["routing"][:, 1].mean().detach(),
                    "structural_gate_mean": route_info["gate"].mean().detach(),
                    "response_cosine": self._mean_cosine(
                        route_info["local_response"],
                        route_info["global_response"],
                        self.eps,
                    ).detach(),
                }

            if return_details:
                details[modality] = {
                    "prior": prior,
                    "local_state": local_state,
                    "local_delta": local_delta,
                    "global_delta": global_delta,
                    "output": output,
                    **route_info,
                }

        fused = torch.cat(
            [modality_outputs["text"], modality_outputs["visual"]], dim=-1
        )
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )
        if return_details:
            info["details"] = details
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
            x.to(device),
            None if edge_index is None else edge_index.to(device),
        )
        return z.detach().cpu()
