from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import open_dict

from src.models.care_mag_v0 import Model as CareMAGv0
from src.models.care_mag_v0 import _MODALITIES


_TRAJECTORY_MODES = {"global", "node"}
_ADAPTER_MODES = {"off", "static", "context"}
_INTERVENTIONS = {
    "normal",
    "traj_global_mean",
    "traj_node_shuffle",
    "adapter_off",
    "coeff_global_mean",
    "coeff_node_shuffle",
}


class Model(CareMAGv0):
    """Matched 2x3 CARE-MAG placement audit.

    The v0 module constructor is reused unchanged, so every factorial cell has
    the same parameter names, order, shapes, and seeded initialization. The
    two modes below select only information flow in ``_forward_impl``.
    """

    def __init__(self, cfg, data_info: dict[str, Any]):
        # v0's variant controls only its forward path. Give its unchanged
        # constructor one fixed valid value; v1 uses the independent factors.
        base_cfg = copy.deepcopy(cfg)
        with open_dict(base_cfg.model):
            base_cfg.model.variant = "context_adapter"
        super().__init__(base_cfg, data_info)

        self.trajectory_mode = str(cfg.model.get("trajectory_mode", "node")).lower()
        self.adapter_mode = str(cfg.model.get("adapter_mode", "context")).lower()
        if self.trajectory_mode not in _TRAJECTORY_MODES:
            raise ValueError(
                f"trajectory_mode must be one of {sorted(_TRAJECTORY_MODES)}, "
                f"got {self.trajectory_mode!r}"
            )
        if self.adapter_mode not in _ADAPTER_MODES:
            raise ValueError(
                f"adapter_mode must be one of {sorted(_ADAPTER_MODES)}, "
                f"got {self.adapter_mode!r}"
            )
        # v1 stores and consumes two independent factors; the legacy v0
        # ``variant`` field is not part of this model's forward control.
        del self.variant

    def _coefficients(
        self,
        modality: str,
        prior: torch.Tensor,
        response: torch.Tensor,
        intervention: str,
        node_permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.adapter_mode == "static":
            query = self.static_tokens[modality].unsqueeze(0).expand(prior.size(0), -1)
        else:
            query = torch.cat(
                [prior, response, (prior - response).abs(), prior * response], dim=-1
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

    def _global_trajectory(
        self,
        modality: str,
        prior: torch.Tensor,
        modality_responses: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        readout = self.trajectory_readouts[modality]
        response_stack = torch.stack(modality_responses, dim=1)
        global_prior = prior.mean(dim=0, keepdim=True)
        global_responses = response_stack.mean(dim=0)
        joint = (
            readout.prior(global_prior)
            + readout.response(global_responses)
            + readout.hop_embedding
        )
        scores = torch.einsum("kh,h->k", torch.tanh(joint), readout.score)
        alpha_row = torch.softmax(scores, dim=0)
        alpha = alpha_row.unsqueeze(0).expand(prior.size(0), -1)
        response = (alpha.unsqueeze(-1) * response_stack).sum(dim=1)
        return response, alpha, response_stack

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
        if intervention not in _INTERVENTIONS:
            raise ValueError(f"unsupported CARE intervention: {intervention}")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"care_mag_v1 expected x shape [nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        if edge_index is None:
            raise ValueError("care_mag_v1 requires edge_index")

        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        src, dst, norm = self._normalized_operator(edge_index, x.size(0), x.dtype)
        inputs = {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }
        priors: dict[str, torch.Tensor] = {}
        responses: dict[str, torch.Tensor] = {}
        alphas: dict[str, torch.Tensor] = {}
        response_stacks: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            prior = self.projectors[modality](inputs[modality])
            trajectory = prior
            modality_responses = []
            for _ in range(self.num_layers):
                trajectory = self._propagate(trajectory, src, dst, norm)
                modality_responses.append(trajectory - prior)

            if self.trajectory_mode == "global":
                response, alpha, response_stack = self._global_trajectory(
                    modality, prior, modality_responses
                )
                if return_details:
                    response_stacks[modality] = response_stack
            else:
                response, alpha = self.trajectory_readouts[modality](
                    prior, modality_responses
                )
                if intervention == "traj_global_mean":
                    alpha = alpha.mean(dim=0, keepdim=True).expand_as(alpha)
                elif intervention == "traj_node_shuffle":
                    # The same local CPU generator makes shuffles reproducible
                    # on CPU and CUDA, without consuming the training RNG.
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(shuffle_seed)
                    )
                    permutation = torch.randperm(x.size(0), generator=generator).to(
                        x.device
                    )
                    alpha = alpha[permutation]
                if intervention in {"traj_global_mean", "traj_node_shuffle"}:
                    response_stack = torch.stack(modality_responses, dim=1)
                    response = (alpha.unsqueeze(-1) * response_stack).sum(dim=1)
                    if return_details:
                        response_stacks[modality] = response_stack
                elif return_details:
                    response_stack = torch.stack(modality_responses, dim=1)
                    response_stacks[modality] = response_stack

            priors[modality] = prior
            responses[modality] = response
            alphas[modality] = alpha

        node_permutation = None
        if intervention == "coeff_node_shuffle":
            generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
            node_permutation = torch.randperm(x.size(0), generator=generator).to(
                x.device
            )

        adapted_responses: dict[str, torch.Tensor] = {}
        coefficients: dict[str, torch.Tensor] = {}
        deltas: dict[str, torch.Tensor] = {}
        states: dict[str, torch.Tensor] = {}
        embeddings: dict[str, torch.Tensor] = {}
        lambdas: dict[str, torch.Tensor] = {}
        gammas: dict[str, torch.Tensor] = {}
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
            if self.adapter_mode == "off" or intervention == "adapter_off":
                delta = torch.zeros_like(delta)
            gamma = torch.sigmoid(self.theta_gamma[modality])
            trust = torch.sigmoid(self.theta_trust[modality])
            adapted = response + gamma * delta
            embedding = F.layer_norm(
                prior + trust * adapted, (self.hidden_dim,), eps=self.eps
            )

            adapted_responses[modality] = adapted
            coefficients[modality] = coefficient
            deltas[modality] = delta
            states[modality] = state
            embeddings[modality] = embedding
            lambdas[modality] = trust
            gammas[modality] = gamma

        fused = torch.cat([embeddings["text"], embeddings["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )

        info: dict[str, Any] = {}
        hop_values = x.new_tensor([1.0, 2.0, 3.0])
        for modality in _MODALITIES:
            alpha = alphas[modality]
            response = responses[modality]
            delta = deltas[modality]
            coefficient = coefficients[modality]
            gamma_delta = gammas[modality] * delta
            response_rms = response.square().mean().sqrt()
            delta_rms = delta.square().mean().sqrt()
            gamma_delta_rms = gamma_delta.square().mean().sqrt()
            effective_hop = (alpha * hop_values).sum(dim=1)
            entropy = -(alpha.clamp_min(self.eps).log() * alpha).sum(dim=1)
            dominant = F.one_hot(alpha.argmax(dim=1), num_classes=self.num_layers).float()
            cosine = F.cosine_similarity(delta, response, dim=-1)
            info[modality] = {
                "lambda": lambdas[modality].detach(),
                "gamma": gammas[modality].detach(),
                "alpha_mean": alpha.mean(dim=0).detach(),
                "alpha_std": alpha.std(dim=0, unbiased=False).detach(),
                "effective_hop_mean": effective_hop.mean().detach(),
                "effective_hop_std": effective_hop.std(unbiased=False).detach(),
                "alpha_entropy_mean": entropy.mean().detach(),
                "alpha_entropy_std": entropy.std(unbiased=False).detach(),
                "dominant_hop_frequency": dominant.mean(dim=0).detach(),
                "response_rms": response_rms.detach(),
                "delta_rms": delta_rms.detach(),
                "delta_response_rms_ratio": (delta_rms / response_rms.clamp_min(self.eps)).detach(),
                "gamma_delta_rms": gamma_delta_rms.detach(),
                "gamma_delta_response_rms_ratio": (
                    gamma_delta_rms / response_rms.clamp_min(self.eps)
                ).detach(),
                "coefficient_mean": coefficient.mean(dim=0).detach(),
                "coefficient_std": coefficient.std(dim=0, unbiased=False).detach(),
                "coefficient_node_std": coefficient.std(
                    dim=0, unbiased=False
                ).mean().detach(),
                "delta_response_cosine_mean": cosine.mean().detach(),
                "delta_response_cosine_std": cosine.std(unbiased=False).detach(),
            }

        if return_details:
            info["details"] = {
                "priors": priors,
                "responses": responses,
                "response_stacks": response_stacks,
                "alphas": alphas,
                "coefficients": coefficients,
                "deltas": deltas,
                "states": states,
                "adapted_responses": adapted_responses,
                "embeddings": embeddings,
                "z": z,
                "trajectory_mode": self.trajectory_mode,
                "adapter_mode": self.adapter_mode,
            }
        return z, info


__all__ = ["Model"]
