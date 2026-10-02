from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops


_MODALITIES = ("text", "visual")
_FILTER_MODES = {
    "uniform",
    "positive_shared",
    "signed_shared",
    "signed_independent",
    "signed_shared_private",
}
_INTERVENTIONS = {"normal", "modality_filter_mean", "modality_filter_swap"}


class _IntrinsicProjector(nn.Module):
    """Independent modality prior projection."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.linear2(self.dropout(F.gelu(self.linear1(x)))) + self.skip(x)
        )


class Model(nn.Module):
    """Prior-retaining GPR-style polynomial filter for multimodal MAG graphs.

    The five filter modes select information flow only. Every mode constructs
    all filter parameters in the same order for matched initialization and
    state-dict layout.
    """

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("spgpr_mag_v0 requires positive text and visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "spgpr_mag_v0 requires exact [text, visual] feature widths: "
                f"input_dim={self.input_dim}, text_dim={self.text_dim}, "
                f"visual_dim={self.visual_dim}"
            )

        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.num_layers = int(model_cfg.get("num_layers", 3))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.diffusion_add_self_loops = bool(
            model_cfg.get("diffusion_add_self_loops", True)
        )
        self.filter_mode = str(model_cfg.get("filter_mode", "signed_shared")).lower()
        if self.filter_mode not in _FILTER_MODES:
            raise ValueError(
                f"filter_mode must be one of {sorted(_FILTER_MODES)}, got {self.filter_mode!r}"
            )
        if self.num_layers != 3:
            raise ValueError("spgpr_mag_v0 fixes num_layers=3 for protocol compatibility")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        self.out_dim = self.hidden_dim

        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
                "visual": _IntrinsicProjector(
                    self.visual_dim, self.hidden_dim, self.dropout_p
                ),
            }
        )

        # Construct every filter parameter in fixed order for every mode.
        self.theta_positive = nn.Parameter(torch.zeros(3))
        self.theta_signed_shared = nn.Parameter(torch.ones(3))
        self.theta_ind_text = nn.Parameter(torch.ones(3))
        self.theta_ind_visual = nn.Parameter(torch.ones(3))
        self.theta_sp_shared = nn.Parameter(torch.ones(3))
        self.theta_sp_diff = nn.Parameter(torch.zeros(3))

        scale_init = float(model_cfg.get("structural_scale_init", 0.5))
        if not 0.0 < scale_init < 1.0:
            raise ValueError("structural_scale_init must lie in (0, 1)")
        scale_logit = math.log(scale_init / (1.0 - scale_init))
        self.theta_lambda_text = nn.Parameter(torch.tensor(scale_logit))
        self.theta_lambda_visual = nn.Parameter(torch.tensor(scale_logit))

        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"spgpr_mag_v0 expected x shape [nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    @staticmethod
    def _signed_norm(raw: torch.Tensor, eps: float) -> torch.Tensor:
        return raw / (raw.abs().sum() + eps)

    def effective_gammas(self) -> dict[str, torch.Tensor]:
        """Return the effective normalized hop filters for the active mode."""
        if self.filter_mode == "uniform":
            gamma = self.theta_positive.new_full((3,), 1.0 / 3.0)
            return {modality: gamma for modality in _MODALITIES}
        if self.filter_mode == "positive_shared":
            gamma = torch.softmax(self.theta_positive, dim=0)
            return {modality: gamma for modality in _MODALITIES}
        if self.filter_mode == "signed_shared":
            gamma = self._signed_norm(self.theta_signed_shared, self.eps)
            return {modality: gamma for modality in _MODALITIES}
        if self.filter_mode == "signed_independent":
            return {
                "text": self._signed_norm(self.theta_ind_text, self.eps),
                "visual": self._signed_norm(self.theta_ind_visual, self.eps),
            }
        return {
            "text": self._signed_norm(
                self.theta_sp_shared + self.theta_sp_diff, self.eps
            ),
            "visual": self._signed_norm(
                self.theta_sp_shared - self.theta_sp_diff, self.eps
            ),
        }

    def effective_sp_decomposition(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gammas = self.effective_gammas()
        shared = (gammas["text"] + gammas["visual"]) / 2.0
        private = (gammas["text"] - gammas["visual"]) / 2.0
        return shared, private

    def _intervention_gammas(self, intervention: str) -> dict[str, torch.Tensor]:
        intervention = str(intervention).lower()
        if intervention not in _INTERVENTIONS:
            raise ValueError(f"unsupported SPGPR intervention: {intervention}")
        gammas = self.effective_gammas()
        if intervention == "modality_filter_mean":
            mean = (gammas["text"] + gammas["visual"]) / 2.0
            return {modality: mean for modality in _MODALITIES}
        if intervention == "modality_filter_swap":
            return {"text": gammas["visual"], "visual": gammas["text"]}
        return gammas

    def _normalized_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("spgpr_mag_v0 requires the current edge_index")
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        intervention: str = "normal",
        return_details: bool = False,
    ):
        if edge_index is None:
            raise ValueError("spgpr_mag_v0 requires edge_index")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, norm = self._normalized_operator(edge_index, x.size(0), x.dtype)
        trajectories: dict[str, list[torch.Tensor]] = {}
        for modality in _MODALITIES:
            state = priors[modality]
            responses = []
            for _ in range(self.num_layers):
                state = self._propagate(state, src, dst, norm)
                responses.append(state - priors[modality])
            trajectories[modality] = responses

        gammas = self._intervention_gammas(intervention)
        lambdas = {
            "text": torch.sigmoid(self.theta_lambda_text),
            "visual": torch.sigmoid(self.theta_lambda_visual),
        }
        embeddings: dict[str, torch.Tensor] = {}
        info: dict[str, Any] = {
            "filter_mode": self.filter_mode,
            "intervention": str(intervention).lower(),
            "modalities": {},
        }
        for modality in _MODALITIES:
            response = sum(
                gammas[modality][hop] * trajectories[modality][hop]
                for hop in range(self.num_layers)
            )
            embeddings[modality] = F.layer_norm(
                priors[modality] + lambdas[modality] * response,
                (self.hidden_dim,),
                eps=self.eps,
            )
            prior_rms = priors[modality].square().mean().sqrt()
            response_rms = response.square().mean().sqrt()
            info["modalities"][modality] = {
                "gamma": gammas[modality].detach(),
                "lambda": lambdas[modality].detach(),
                "prior_rms": prior_rms.detach(),
                "structural_response_rms": response_rms.detach(),
                "scaled_response_to_prior_rms": (
                    lambdas[modality] * response_rms / prior_rms.clamp_min(self.eps)
                ).detach(),
            }
        fused = torch.cat([embeddings["text"], embeddings["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )
        if return_details:
            info["details"] = {
                "priors": priors,
                "responses": {
                    m: torch.stack(trajectories[m], dim=1) for m in _MODALITIES
                },
                "embeddings": embeddings,
            }
        return z, None, None, z.new_zeros(()), info

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

