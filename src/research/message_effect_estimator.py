from __future__ import annotations

from typing import Literal

import torch
from torch import nn

EstimatorVariant = Literal[
    "E0_heuristic",
    "E1_unimodal_pair",
    "E2_multimodal_pair",
    "E3_host_context",
]


class _PairEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.size(-1) != 1024:
            raise ValueError("pair features must have shape [batch, 1024]")
        return self.net(value)


class _HostEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MessageEffectEstimator(nn.Module):
    """Fixed small estimator for frozen-host exposed-message effects."""

    def __init__(
        self,
        variant: EstimatorVariant,
        scalar_dim: int,
        host_dim: int | None = None,
    ) -> None:
        super().__init__()
        if variant not in {
            "E0_heuristic",
            "E1_unimodal_pair",
            "E2_multimodal_pair",
            "E3_host_context",
        }:
            raise ValueError(f"unknown estimator variant: {variant}")
        if scalar_dim <= 0:
            raise ValueError("scalar_dim must be positive")
        if variant == "E3_host_context" and (host_dim is None or host_dim <= 0):
            raise ValueError("E3 requires a positive host_dim")
        self.variant = variant
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.pair_active_encoder = _PairEncoder() if variant == "E1_unimodal_pair" else None
        self.pair_text_encoder = _PairEncoder() if variant in {
            "E2_multimodal_pair",
            "E3_host_context",
        } else None
        self.pair_visual_encoder = _PairEncoder() if variant in {
            "E2_multimodal_pair",
            "E3_host_context",
        } else None
        self.host_encoder = (
            _HostEncoder(int(host_dim)) if variant == "E3_host_context" else None
        )
        block_count = {
            "E0_heuristic": 1,
            "E1_unimodal_pair": 2,
            "E2_multimodal_pair": 3,
            "E3_host_context": 4,
        }[variant]
        self.predictor = nn.Sequential(
            nn.Linear(64 * block_count, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        scalar_features: torch.Tensor,
        *,
        pair_active: torch.Tensor | None = None,
        pair_text: torch.Tensor | None = None,
        pair_visual: torch.Tensor | None = None,
        host_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scalar_features.ndim != 2:
            raise ValueError("scalar_features must have shape [batch, scalar_dim]")
        blocks = [self.scalar_encoder(scalar_features)]
        if self.variant == "E0_heuristic":
            if any(x is not None for x in (pair_active, pair_text, pair_visual, host_context)):
                raise ValueError("E0 accepts scalar cues only")
        elif self.variant == "E1_unimodal_pair":
            if pair_active is None or any(x is not None for x in (pair_text, pair_visual, host_context)):
                raise ValueError("E1 requires only the active-modality pair block")
            assert self.pair_active_encoder is not None
            blocks.append(self.pair_active_encoder(pair_active))
        elif self.variant == "E2_multimodal_pair":
            if pair_text is None or pair_visual is None or any(
                x is not None for x in (pair_active, host_context)
            ):
                raise ValueError("E2 requires Text and Visual pair blocks only")
            assert self.pair_text_encoder is not None
            assert self.pair_visual_encoder is not None
            blocks.extend((self.pair_text_encoder(pair_text), self.pair_visual_encoder(pair_visual)))
        else:
            if pair_text is None or pair_visual is None or host_context is None or pair_active is not None:
                raise ValueError("E3 requires Text/Visual pair blocks and frozen host context")
            assert self.pair_text_encoder is not None
            assert self.pair_visual_encoder is not None
            assert self.host_encoder is not None
            blocks.extend(
                (
                    self.pair_text_encoder(pair_text),
                    self.pair_visual_encoder(pair_visual),
                    self.host_encoder(host_context),
                )
            )
        return self.predictor(torch.cat(blocks, dim=-1))


def parameter_free_pair(
    receiver_state: torch.Tensor, sender_state: torch.Tensor
) -> torch.Tensor:
    """Build [LN(u_i), LN(u_j), product, abs-difference] pair features."""
    if receiver_state.shape != sender_state.shape or receiver_state.ndim != 2:
        raise ValueError("receiver and sender states must have matching [batch, hidden] shapes")
    dim = receiver_state.size(-1)
    receiver = torch.nn.functional.layer_norm(receiver_state, (dim,))
    sender = torch.nn.functional.layer_norm(sender_state, (dim,))
    return torch.cat(
        (receiver, sender, receiver * sender, (receiver - sender).abs()), dim=-1
    )
