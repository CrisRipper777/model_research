from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class R3MAGHostConfig:
    """Frozen design settings for the H1 global response host."""

    hidden_dim: int = 128
    max_order: int = 3
    dropout: float = 0.2
    ppr_restart: float = 0.2

    def gamma_init(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        # Truncated PPR coefficients: gamma_k = alpha * (1-alpha)^k.
        alpha = float(self.ppr_restart)
        return torch.tensor(
            [alpha * (1.0 - alpha) ** k for k in range(self.max_order + 1)],
            dtype=torch.float32,
            device=device,
        )


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class GlobalDualBranchResponseHost(nn.Module):
    """Small, frozen-audit host with one global signed response per modality."""

    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        num_classes: int,
        config: R3MAGHostConfig = R3MAGHostConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        h = int(config.hidden_dim)
        self.text_projector = ModalityProjector(text_dim, h, config.dropout)
        self.visual_projector = ModalityProjector(visual_dim, h, config.dropout)
        gamma = config.gamma_init()
        self.gamma_text = nn.Parameter(gamma.clone())
        self.gamma_visual = nn.Parameter(gamma.clone())
        self.text_norm = nn.LayerNorm(h)
        self.visual_norm = nn.LayerNorm(h)
        self.fusion = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(h, num_classes)

    @staticmethod
    def propagate_basis(
        h0: torch.Tensor, operator: torch.Tensor, max_order: int
    ) -> list[torch.Tensor]:
        states = [h0]
        for _ in range(max_order):
            states.append(torch.sparse.mm(operator, states[-1]))
        return states

    @staticmethod
    def combine_global(states: list[torch.Tensor], gamma: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(states, dim=0)
        return (stacked * gamma.view(-1, 1, 1)).sum(dim=0)

    def response_bases(
        self,
        x_text: torch.Tensor,
        x_visual: torch.Tensor,
        operator: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h_text = self.text_projector(x_text)
        h_visual = self.visual_projector(x_visual)
        text_states = self.propagate_basis(h_text, operator, self.config.max_order)
        visual_states = self.propagate_basis(h_visual, operator, self.config.max_order)
        return text_states, visual_states

    def classify_responses(
        self, response_text: torch.Tensor, response_visual: torch.Tensor
    ) -> torch.Tensor:
        z_text = self.text_norm(response_text)
        z_visual = self.visual_norm(response_visual)
        return self.classifier(self.fusion(torch.cat([z_text, z_visual], dim=-1)))

    def forward(
        self,
        x_text: torch.Tensor,
        x_visual: torch.Tensor,
        operator: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        text_states, visual_states = self.response_bases(x_text, x_visual, operator)
        global_text = self.combine_global(text_states, self.gamma_text)
        global_visual = self.combine_global(visual_states, self.gamma_visual)
        logits = self.classify_responses(global_text, global_visual)
        return logits, text_states, visual_states, global_text, global_visual


def normalized_response_directions(
    states: list[torch.Tensor],
    global_response: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (basis-global) directions rescaled to each node's global norm.

    A direction whose norm is at most ``eps`` is set to zero and is counted by
    the caller. Output has shape [nodes, orders, hidden].
    """
    basis = torch.stack(states, dim=1)
    raw = basis - global_response.unsqueeze(1)
    raw_norm = raw.norm(dim=-1, keepdim=True)
    global_norm = global_response.norm(dim=-1, keepdim=True).unsqueeze(1)
    scaled = raw / raw_norm.clamp_min(eps) * global_norm
    scaled = torch.where(raw_norm > eps, scaled, torch.zeros_like(scaled))
    return scaled, raw_norm.squeeze(-1)


def norm_match(candidate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Match candidate L2 norm to reference along the final dimension."""
    cand_norm = candidate.norm(dim=-1, keepdim=True)
    ref_norm = reference.norm(dim=-1, keepdim=True)
    while ref_norm.ndim < cand_norm.ndim:
        ref_norm = ref_norm.unsqueeze(-2)
    ratio = torch.where(cand_norm > eps, ref_norm / cand_norm.clamp_min(eps), torch.zeros_like(cand_norm))
    return candidate * ratio


def make_action_candidates(
    global_response: torch.Tensor,
    directions: torch.Tensor,
    epsilon: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build [NOOP, +order0..K, -order0..K] NormMatch candidates."""
    if directions.ndim != 3 or directions.shape[0] != global_response.shape[0]:
        raise ValueError("directions must have shape [nodes, orders, hidden]")
    if float(epsilon) == 0.0:
        return global_response[:, None, :].expand(-1, 1 + 2 * directions.size(1), -1).clone()
    plus = norm_match(global_response[:, None, :] + epsilon * directions, global_response, eps)
    minus = norm_match(global_response[:, None, :] - epsilon * directions, global_response, eps)
    return torch.cat([global_response[:, None, :], plus, minus], dim=1)


def action_names(max_order: int) -> list[str]:
    return ["NOOP"] + [f"+k{k}" for k in range(max_order + 1)] + [
        f"-k{k}" for k in range(max_order + 1)
    ]
