from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops


_MODALITIES = ("text", "visual")
_VARIANTS = {
    "prior_only",
    "raw_uniform_protected",
    "anchored_uniform_protected",
    "anchored_prior_protected",
    "anchored_gpr_protected",
    "anchored_gpr_direct",
}
_INTERVENTIONS = {"normal", "gamma_reset_prior", "graph_injection_off", "lambda_one"}


class _IntrinsicProjector(nn.Module):
    """Modality-specific intrinsic semantic projection."""

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
    """Prior-anchored GPR backbone with matched six-variant parameter layouts."""

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("pigpr_mag_v0 requires positive text and visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "pigpr_mag_v0 requires exact [text, visual] widths: "
                f"input_dim={self.input_dim}, text_dim={self.text_dim}, "
                f"visual_dim={self.visual_dim}"
            )

        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.num_layers = int(model_cfg.get("num_layers", 3))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.anchor_alpha = float(model_cfg.get("anchor_alpha", 0.1))
        self.global_prior_restart = float(model_cfg.get("global_prior_restart", 0.15))
        self.global_prior_order = int(model_cfg.get("global_prior_order", 2))
        self.diffusion_add_self_loops = bool(
            model_cfg.get("diffusion_add_self_loops", True)
        )
        self.variant = str(model_cfg.get("variant", "anchored_gpr_protected")).lower()
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}, got {self.variant!r}")
        if self.num_layers != 3:
            raise ValueError("pigpr_mag_v0 fixes num_layers=3 for protocol compatibility")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.0 <= self.anchor_alpha <= 1.0:
            raise ValueError("anchor_alpha must lie in [0, 1]")
        if not 0.0 <= self.global_prior_restart <= 1.0:
            raise ValueError("global_prior_restart must lie in [0, 1]")
        if not 0 <= self.global_prior_order <= self.num_layers:
            raise ValueError("global_prior_order must lie in [0, num_layers]")
        self.out_dim = self.hidden_dim

        # Keep these modules and their construction order aligned with SPGPR-v0.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
                "visual": _IntrinsicProjector(
                    self.visual_dim, self.hidden_dim, self.dropout_p
                ),
            }
        )

        # Shared unconstrained anchored GPR deviation is present in every variant.
        self.delta_gamma_global = nn.Parameter(torch.zeros(self.num_layers + 1))
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

        prior = self._make_global_prior()
        self.register_buffer("monomial_prior", prior.clone(), persistent=True)
        self.register_buffer(
            "gamma_prior",
            self._monomial_to_anchored_cumulative(prior, self.anchor_alpha),
            persistent=True,
        )

    def _make_global_prior(self) -> torch.Tensor:
        """Build CoSI's restart prior in the monomial basis."""
        prior = torch.zeros(self.num_layers + 1, dtype=torch.float32)
        restart = self.global_prior_restart
        target = self.global_prior_order
        if self.num_layers >= target:
            for order in range(target):
                prior[order] = restart * (1.0 - restart) ** order
            prior[target] = (1.0 - restart) ** target
        else:
            for order in range(self.num_layers):
                prior[order] = restart * (1.0 - restart) ** order
            prior[self.num_layers] = (1.0 - restart) ** self.num_layers
        return prior

    @staticmethod
    def _anchored_basis_matrix(
        count: int, alpha: float, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        matrix = torch.zeros((count, count), dtype=dtype, device=device)
        retention = 1.0 - float(alpha)
        matrix[0, 0] = 1.0
        for order in range(1, count):
            matrix[order, order] = retention**order
            powers = torch.arange(order, dtype=dtype, device=device)
            matrix[order, :order] = float(alpha) * retention**powers
        return matrix

    @staticmethod
    def _monomial_to_anchored_cumulative(
        coefficients: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        count = int(coefficients.numel())
        matrix = Model._anchored_basis_matrix(
            count, alpha, dtype=coefficients.dtype, device=coefficients.device
        )
        return torch.linalg.solve_triangular(
            matrix.transpose(0, 1), coefficients.unsqueeze(-1), upper=True
        ).squeeze(-1)

    def equivalent_monomial_coefficients(self, gamma: torch.Tensor) -> torch.Tensor:
        matrix = self._anchored_basis_matrix(
            int(gamma.numel()), self.anchor_alpha, dtype=gamma.dtype, device=gamma.device
        )
        return matrix.transpose(0, 1) @ gamma

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"pigpr_mag_v0 expected x shape [nodes, {self.input_dim}], got {tuple(x.shape)}"
            )
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    def _normalized_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("pigpr_mag_v0 graph variants require edge_index")
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

    def _raw_and_anchored_states(
        self, prior: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, norm: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raw = [prior]
        anchored = [prior]
        raw_state = prior
        anchored_state = prior
        for _ in range(self.num_layers):
            raw_state = self._propagate(raw_state, src, dst, norm)
            anchored_state = (
                (1.0 - self.anchor_alpha)
                * self._propagate(anchored_state, src, dst, norm)
                + self.anchor_alpha * prior
            )
            raw.append(raw_state)
            anchored.append(anchored_state)
        return raw, anchored

    def _state_sequence(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
        *,
        anchored: bool,
    ) -> list[torch.Tensor]:
        states = [prior]
        state = prior
        for _ in range(self.num_layers):
            propagated = self._propagate(state, src, dst, norm)
            state = (
                (1.0 - self.anchor_alpha) * propagated + self.anchor_alpha * prior
                if anchored
                else propagated
            )
            states.append(state)
        return states

    def effective_gamma(self) -> torch.Tensor:
        if self.variant == "anchored_prior_protected":
            return self.gamma_prior
        if self.variant in {"anchored_gpr_protected", "anchored_gpr_direct"}:
            return self.gamma_prior + self.delta_gamma_global
        if self.variant == "raw_uniform_protected":
            return self.delta_gamma_global.new_full((self.num_layers,), 1.0 / self.num_layers)
        if self.variant == "anchored_uniform_protected":
            return self.delta_gamma_global.new_full((self.num_layers,), 1.0 / self.num_layers)
        return self.delta_gamma_global.new_zeros((self.num_layers + 1,))

    def _intervention(self, intervention: str) -> str:
        value = str(intervention).lower()
        if value not in _INTERVENTIONS:
            raise ValueError(f"unsupported PIGPR intervention: {value}")
        if value != "normal" and self.variant not in {
            "anchored_gpr_protected",
            "anchored_gpr_direct",
        }:
            raise ValueError("checkpoint interventions are defined only for AGP/AGD")
        return value

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        intervention: str = "normal",
        return_details: bool = False,
    ):
        intervention = self._intervention(intervention)
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}

        if self.variant == "prior_only":
            src = dst = norm = None
            raw_states = {m: [priors[m]] for m in _MODALITIES}
            anchored_states = {m: [priors[m]] for m in _MODALITIES}
        else:
            src, dst, norm = self._normalized_operator(edge_index, x.size(0), x.dtype)
            if self.variant == "raw_uniform_protected":
                raw_states = {
                    m: self._state_sequence(priors[m], src, dst, norm, anchored=False)
                    for m in _MODALITIES
                }
                anchored_states = {m: [priors[m]] for m in _MODALITIES}
            else:
                anchored_states = {
                    m: self._state_sequence(priors[m], src, dst, norm, anchored=True)
                    for m in _MODALITIES
                }
                raw_states = {m: [priors[m]] for m in _MODALITIES}
            if return_details:
                state_pairs = {
                    m: self._raw_and_anchored_states(priors[m], src, dst, norm)
                    for m in _MODALITIES
                }
                raw_states = {m: state_pairs[m][0] for m in _MODALITIES}
                anchored_states = {m: state_pairs[m][1] for m in _MODALITIES}

        gamma = self.effective_gamma()
        if intervention == "gamma_reset_prior":
            gamma = self.gamma_prior
        lambdas = {
            "text": torch.sigmoid(self.theta_lambda_text),
            "visual": torch.sigmoid(self.theta_lambda_visual),
        }
        if intervention == "graph_injection_off":
            lambdas = {m: lambdas[m].new_zeros(()) for m in _MODALITIES}
        elif intervention == "lambda_one":
            lambdas = {m: lambdas[m].new_ones(()) for m in _MODALITIES}

        embeddings: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        info: dict[str, Any] = {
            "variant": self.variant,
            "intervention": intervention,
            "modalities": {},
        }
        for modality in _MODALITIES:
            prior = priors[modality]
            if self.variant == "prior_only":
                selected_states = [prior]
                proposal = prior
                pre_norm = prior
                embedding = F.layer_norm(prior, (self.hidden_dim,), eps=self.eps)
            elif self.variant == "raw_uniform_protected":
                selected_states = raw_states[modality][1:]
                proposal = (selected_states[0] + selected_states[1] + selected_states[2]) / 3.0
                pre_norm = prior + lambdas[modality] * (proposal - prior)
                embedding = F.layer_norm(pre_norm, (self.hidden_dim,), eps=self.eps)
            elif self.variant == "anchored_uniform_protected":
                selected_states = anchored_states[modality][1:]
                proposal = (selected_states[0] + selected_states[1] + selected_states[2]) / 3.0
                pre_norm = prior + lambdas[modality] * (proposal - prior)
                embedding = F.layer_norm(pre_norm, (self.hidden_dim,), eps=self.eps)
            else:
                selected_states = anchored_states[modality]
                proposal = sum(gamma[order] * selected_states[order] for order in range(4))
                if self.variant == "anchored_gpr_direct":
                    pre_norm = proposal
                    embedding = F.layer_norm(proposal, (self.hidden_dim,), eps=self.eps)
                else:
                    pre_norm = prior + lambdas[modality] * (proposal - prior)
                    embedding = F.layer_norm(pre_norm, (self.hidden_dim,), eps=self.eps)
            embeddings[modality] = embedding

            delta = proposal - prior
            prior_rms = prior.square().mean().sqrt()
            proposal_rms = proposal.square().mean().sqrt()
            delta_rms = delta.square().mean().sqrt()
            cosine = F.cosine_similarity(prior, proposal, dim=-1, eps=self.eps).mean()
            pre_cosine = F.cosine_similarity(prior, pre_norm, dim=-1, eps=self.eps).mean()
            info["modalities"][modality] = {
                "gamma": gamma.detach(),
                "lambda": lambdas[modality].detach(),
                "prior_rms": prior_rms.detach(),
                "proposal_rms": proposal_rms.detach(),
                "structural_delta_rms": delta_rms.detach(),
                "structural_delta_to_prior_rms": (
                    delta_rms / prior_rms.clamp_min(self.eps)
                ).detach(),
                "cosine_prior_proposal": cosine.detach(),
                "cosine_prior_pre_norm": pre_cosine.detach(),
                "pre_norm_delta_to_prior_rms": (
                    (pre_norm - prior).square().mean().sqrt()
                    / prior.square().mean().sqrt().clamp_min(self.eps)
                ).detach(),
            }
            details[modality] = {
                "prior": prior,
                "proposal": proposal,
                "pre_norm": pre_norm,
                "embedding": embedding,
                "selected_states": torch.stack(selected_states, dim=1),
                "raw_states": torch.stack(raw_states[modality], dim=1),
                "anchored_states": torch.stack(anchored_states[modality], dim=1),
                "delta": delta,
            }

        fused = torch.cat([embeddings["text"], embeddings["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )
        if return_details:
            info["details"] = {"modalities": details, "gamma": gamma}
        return z, None, None, z.new_zeros(()), info

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
        z, _, _, _, _ = self.forward(x.to(device), None if edge_index is None else edge_index.to(device))
        return z.detach().cpu()
