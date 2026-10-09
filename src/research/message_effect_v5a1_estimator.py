from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .message_effect_estimator import MessageEffectEstimator


class SingleTargetMessageEffectEstimator(MessageEffectEstimator):
    """V5A feature ladder with an independent scalar output for one target."""

    def __init__(self, variant: str, scalar_dim: int, host_dim: int | None = None) -> None:
        super().__init__(variant, scalar_dim, host_dim)
        final = self.predictor[-1]
        if not isinstance(final, torch.nn.Linear) or final.out_features != 2:
            raise AssertionError("unexpected frozen V5A predictor head")
        self.predictor[-1] = torch.nn.Linear(final.in_features, 1)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super().forward(*args, **kwargs)


def receiver_equal_row_weights(receiver_ids: torch.Tensor) -> torch.Tensor:
    """Return mean-one row weights so every represented receiver has equal mass."""
    receivers = torch.as_tensor(receiver_ids, dtype=torch.long).reshape(-1).cpu()
    if receivers.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    _, inverse, counts = torch.unique(receivers, sorted=True, return_inverse=True, return_counts=True)
    n_receivers = counts.numel()
    n_rows = receivers.numel()
    return (n_rows / n_receivers / counts[inverse].float()).to(torch.float32)


def receiver_equal_mean(values: torch.Tensor, receiver_ids: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values).reshape(-1).cpu().float()
    receivers = torch.as_tensor(receiver_ids, dtype=torch.long).reshape(-1).cpu()
    if values.numel() != receivers.numel() or values.numel() == 0:
        raise ValueError("values and receiver_ids must have equal, non-zero length")
    unique, inverse = torch.unique(receivers, sorted=True, return_inverse=True)
    sums = values.new_zeros(unique.numel()).index_add_(0, inverse, values)
    counts = values.new_zeros(unique.numel()).index_add_(0, inverse, torch.ones_like(values))
    return (sums / counts).mean()


def make_edge_rank_pairs(
    rows: Sequence[dict[str, Any]],
    target: torch.Tensor,
    receivers: set[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Build unordered, exact-nontied pairs within receiver/modality/hop groups."""
    target = torch.as_tensor(target, dtype=torch.float32).reshape(-1).cpu()
    if len(rows) != target.numel():
        raise ValueError("row metadata and target must align")
    groups: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        receiver = int(row["receiver_id"])
        value = float(target[i])
        if receivers is not None and receiver not in receivers:
            continue
        if not np.isfinite(value):
            continue
        groups[(receiver, str(row["modality"]), int(row["hop"]))].append(i)
    left: list[int] = []
    right: list[int] = []
    order: list[float] = []
    weights: list[float] = []
    group_ids: list[int] = []
    group_meta: list[tuple[int, str, int]] = []
    for group_id, key in enumerate(sorted(groups)):
        members = groups[key]
        possible: list[tuple[int, int]] = []
        for a, b in combinations(members, 2):
            if float(target[a]) != float(target[b]):
                possible.append((a, b))
        if not possible:
            continue
        pair_weight = 1.0 / len(possible)
        for a, b in possible:
            left.append(a)
            right.append(b)
            order.append(float(np.sign(float(target[b]) - float(target[a]))))
            weights.append(pair_weight)
            group_ids.append(group_id)
            group_meta.append(key)
    return {
        "left": torch.tensor(left, dtype=torch.long),
        "right": torch.tensor(right, dtype=torch.long),
        "order": torch.tensor(order, dtype=torch.float32),
        "weight": torch.tensor(weights, dtype=torch.float32),
        "group_id": torch.tensor(group_ids, dtype=torch.long),
        # Keep readable group keys; modality is categorical rather than numeric.
        "groups": group_meta,
    }


def ranknet_pair_loss(score_a: torch.Tensor, score_b: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """RankNet loss: order=sign(delta_b-delta_a), so lower effects get lower scores."""
    if score_a.shape != score_b.shape or score_a.shape != order.shape:
        raise ValueError("scores and pair order must have matching shapes")
    return F.softplus(-order * (score_b - score_a))


def pair_weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if loss.shape != weights.shape:
        raise ValueError("pair losses and weights must have matching shapes")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0e-12)


def train_quantiles(values: torch.Tensor, probabilities=(0.25, 0.5, 0.75)) -> list[float]:
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    if not values.numel():
        return [float("nan") for _ in probabilities]
    return [float(x) for x in torch.quantile(values.abs(), torch.tensor(probabilities)).tolist()]
