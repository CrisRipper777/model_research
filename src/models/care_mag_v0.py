from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops


_VARIANTS = {"prior_only", "structural_base", "static_adapter", "context_adapter"}
_MODALITIES = ("text", "visual")


class _IntrinsicProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear2(self.dropout(F.gelu(self.linear1(x)))) + self.skip(x)
        return self.norm(h)


class _TrajectoryReadout(nn.Module):
    """Node-specific readout over the three nonzero structural responses."""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.prior = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.response = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hop_embedding = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.score = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.hop_embedding, std=0.02)
        nn.init.normal_(self.score, std=hidden_dim**-0.5)

    def forward(
        self, prior: torch.Tensor, responses: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response_stack = torch.stack(responses, dim=1)  # [nodes, hops, hidden]
        joint = (
            self.prior(prior).unsqueeze(1)
            + self.response(response_stack)
            + self.hop_embedding.unsqueeze(0)
        )
        scores = torch.einsum("nkh,h->nk", torch.tanh(joint), self.score)
        alpha = torch.softmax(scores, dim=1)
        response = (alpha.unsqueeze(-1) * response_stack).sum(dim=1)
        return response, alpha


class _StateEncoder(nn.Module):
    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, state_dim)
        self.linear2 = nn.Linear(state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear2(F.gelu(self.linear1(x))))


class Model(nn.Module):
    """CARE-MAG first-stage matched architecture screen.

    The variant changes only information flow in ``forward``. Every variant
    constructs the same modules in the same order so seed-matched controls
    have identical parameter names, counts, and initial values.
    """

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("care_mag_v0 requires positive text_dim and visual_dim")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "care_mag_v0 requires exact [text, visual] feature widths: "
                f"input_dim={self.input_dim}, text_dim={self.text_dim}, "
                f"visual_dim={self.visual_dim}"
            )

        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.num_layers = int(model_cfg.get("num_layers", 3))
        self.state_dim = int(model_cfg.get("state_dim", 64))
        self.adapter_rank = int(model_cfg.get("adapter_rank", 32))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.coeff_radius = float(model_cfg.get("adapter_coeff_radius", 0.5))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.diffusion_add_self_loops = bool(
            model_cfg.get("diffusion_add_self_loops", True)
        )
        self.variant = str(model_cfg.get("variant", "context_adapter")).lower()
        if self.variant not in _VARIANTS:
            raise ValueError(
                f"care_mag_v0 variant must be one of {sorted(_VARIANTS)}, got {self.variant!r}"
            )
        if self.num_layers != 3:
            raise ValueError("care_mag_v0 v0 fixes num_layers=3")

        self.out_dim = self.hidden_dim

        # Keep construction order fixed across variants.
        modality_dims = {"text": self.text_dim, "visual": self.visual_dim}
        self.projectors = nn.ModuleDict(
            {
                modality: _IntrinsicProjector(
                    modality_dims[modality], self.hidden_dim, self.dropout_p
                )
                for modality in _MODALITIES
            }
        )
        self.trajectory_readouts = nn.ModuleDict(
            {
                modality: _TrajectoryReadout(self.hidden_dim, self.num_layers)
                for modality in _MODALITIES
            }
        )
        state_input_dim = 4 * self.hidden_dim
        self.static_tokens = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.zeros(state_input_dim))
                for modality in _MODALITIES
            }
        )
        self.state_encoders = nn.ModuleDict(
            {
                modality: _StateEncoder(state_input_dim, self.state_dim)
                for modality in _MODALITIES
            }
        )
        self.routers = nn.ModuleDict(
            {
                modality: nn.Linear(self.state_dim, self.adapter_rank)
                for modality in _MODALITIES
            }
        )
        self.adapter_v = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.empty(self.adapter_rank, self.hidden_dim))
                for modality in _MODALITIES
            }
        )
        self.adapter_u = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.empty(self.hidden_dim, self.adapter_rank))
                for modality in _MODALITIES
            }
        )
        trust_init = float(model_cfg.get("trust_init", 0.5))
        scale_init = float(model_cfg.get("adapter_scale_init", 0.1))
        self.theta_trust = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.tensor(self._logit(trust_init)))
                for modality in _MODALITIES
            }
        )
        self.theta_gamma = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.tensor(self._logit(scale_init)))
                for modality in _MODALITIES
            }
        )
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        for modality in _MODALITIES:
            nn.init.xavier_uniform_(self.adapter_v[modality])
            nn.init.normal_(self.adapter_u[modality], mean=0.0, std=1.0e-3)

    @staticmethod
    def _logit(probability: float) -> float:
        if not 0.0 < probability < 1.0:
            raise ValueError(f"initial sigmoid value must lie in (0,1), got {probability}")
        return math.log(probability / (1.0 - probability))

    def _normalized_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("care_mag_v0 requires the current edge_index")
        edge_index = edge_index.to(dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.diffusion_add_self_loops:
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        src, dst = edge_index
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=dtype, device=dst.device))
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        norm = inv_sqrt[src] * inv_sqrt[dst]
        return src, dst, norm

    @staticmethod
    def _propagate(
        state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        messages = state[src] * norm.unsqueeze(-1)
        return torch.zeros_like(state).index_add(0, dst, messages)

    def _coefficients(
        self,
        modality: str,
        prior: torch.Tensor,
        response: torch.Tensor,
        intervention: str,
        node_permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "static_adapter":
            query = self.static_tokens[modality].unsqueeze(0).expand(prior.size(0), -1)
        else:
            query = torch.cat(
                [
                    prior,
                    response,
                    (prior - response).abs(),
                    prior * response,
                ],
                dim=-1,
            )
        state = self.state_encoders[modality](query)
        coefficient = 1.0 + self.coeff_radius * torch.tanh(
            self.routers[modality](state)
        )
        if intervention == "coeff_global_mean":
            coefficient = coefficient.mean(dim=0, keepdim=True).expand_as(coefficient)
        elif intervention == "coeff_node_shuffle":
            if node_permutation is None:
                raise RuntimeError("node permutation is required for coeff_node_shuffle")
            coefficient = coefficient[node_permutation]
        return state, coefficient

    def _forward_impl(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        intervention: str = "normal",
        shuffle_seed: int = 0,
        return_details: bool = False,
    ):
        intervention = str(intervention).lower()
        if intervention not in {
            "normal",
            "adapter_off",
            "coeff_global_mean",
            "coeff_node_shuffle",
        }:
            raise ValueError(f"unsupported CARE intervention: {intervention}")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"care_mag_v0 expected x shape [nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        if edge_index is None:
            raise ValueError("care_mag_v0 requires edge_index")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        src, dst, norm = self._normalized_operator(edge_index, x.size(0), x.dtype)

        inputs = {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }
        priors: dict[str, torch.Tensor] = {}
        responses: dict[str, torch.Tensor] = {}
        alphas: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            prior = self.projectors[modality](inputs[modality])
            trajectory = prior
            modality_responses = []
            for _ in range(self.num_layers):
                trajectory = self._propagate(trajectory, src, dst, norm)
                modality_responses.append(trajectory - prior)
            response, alpha = self.trajectory_readouts[modality](
                prior, modality_responses
            )
            priors[modality] = prior
            responses[modality] = response
            alphas[modality] = alpha

        generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
        node_permutation = None
        if intervention == "coeff_node_shuffle":
            node_permutation = torch.randperm(x.size(0), generator=generator).to(x.device)

        adapted_responses: dict[str, torch.Tensor] = {}
        coefficients: dict[str, torch.Tensor] = {}
        deltas: dict[str, torch.Tensor] = {}
        states: dict[str, torch.Tensor] = {}
        embeddings: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            prior = priors[modality]
            response = responses[modality]
            state, coefficient = self._coefficients(
                modality, prior, response, intervention, node_permutation
            )
            normalized_response = F.layer_norm(
                response, (self.hidden_dim,), eps=self.eps
            )
            low_rank = F.linear(normalized_response, self.adapter_v[modality])
            hidden = F.gelu(low_rank)
            delta = F.linear(hidden * coefficient, self.adapter_u[modality])
            if self.variant == "structural_base" or intervention == "adapter_off":
                delta = torch.zeros_like(delta)
            gamma = torch.sigmoid(self.theta_gamma[modality])
            trust = torch.sigmoid(self.theta_trust[modality])
            adapted = response + gamma * delta
            if self.variant == "prior_only":
                embedding = F.layer_norm(prior, (self.hidden_dim,), eps=self.eps)
            else:
                embedding = F.layer_norm(
                    prior + trust * adapted, (self.hidden_dim,), eps=self.eps
                )
            adapted_responses[modality] = adapted
            coefficients[modality] = coefficient
            deltas[modality] = delta
            states[modality] = state
            embeddings[modality] = embedding

        fused = torch.cat([embeddings["text"], embeddings["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )

        info: dict[str, Any] = {}
        for modality in _MODALITIES:
            alpha = alphas[modality]
            response = responses[modality]
            delta = deltas[modality]
            info[modality] = {
                "lambda": torch.sigmoid(self.theta_trust[modality]).detach(),
                "gamma": torch.sigmoid(self.theta_gamma[modality]).detach(),
                "alpha_mean": alpha.mean(dim=0).detach(),
                "alpha_std": alpha.std(dim=0, unbiased=False).detach(),
                "effective_hop_mean": (
                    alpha * alpha.new_tensor([1.0, 2.0, 3.0])
                ).sum(dim=1).mean().detach(),
                "effective_hop_std": (
                    alpha * alpha.new_tensor([1.0, 2.0, 3.0])
                ).sum(dim=1).std(unbiased=False).detach(),
                "response_rms": response.square().mean().sqrt().detach(),
                "delta_rms": delta.square().mean().sqrt().detach(),
                "coefficient_mean": coefficients[modality].mean(dim=0).detach(),
                "coefficient_std": coefficients[modality].std(
                    dim=0, unbiased=False
                ).detach(),
                "coefficient_node_std": coefficients[modality].std(
                    dim=0, unbiased=False
                ).mean().detach(),
            }
        if return_details:
            info["details"] = {
                "priors": priors,
                "responses": responses,
                "alphas": alphas,
                "coefficients": coefficients,
                "deltas": deltas,
                "states": states,
                "adapted_responses": adapted_responses,
                "embeddings": embeddings,
                "z": z,
            }
        return z, info

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        intervention: str = "normal",
        shuffle_seed: int = 0,
        return_details: bool = False,
    ):
        z, aux_info = self._forward_impl(
            x,
            edge_index,
            intervention=intervention,
            shuffle_seed=shuffle_seed,
            return_details=return_details,
        )
        return z, None, None, z.new_zeros(()), aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()
