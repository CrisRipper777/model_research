"""Small, label-free utilities for the R3-MAG H1.1/H2 preflight."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def coherent_control_donors(
    target_idx: torch.Tensor,
    degree_bucket: torch.Tensor,
    predicted_class: torch.Tensor,
    num_nodes: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Choose one matched donor per target for both modalities and all orders."""
    from src.analysis.r3mag_h1_response_audit import _assign_donors, _bucket_key

    targets = torch.as_tensor(target_idx, dtype=torch.long).cpu().tolist()
    all_nodes = np.arange(int(num_nodes), dtype=np.int64)
    joint_lists: dict[tuple[int, int], list[int]] = defaultdict(list)
    degree_lists: dict[int, list[int]] = defaultdict(list)
    for node in all_nodes.tolist():
        joint_lists[_bucket_key(degree_bucket, predicted_class, node, "joint")].append(node)
        degree_lists[_bucket_key(degree_bucket, predicted_class, node, "degree")].append(node)
    joint = {key: np.asarray(value, dtype=np.int64) for key, value in joint_lists.items()}
    by_degree = {key: np.asarray(value, dtype=np.int64) for key, value in degree_lists.items()}
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for node in targets:
        joint_key = _bucket_key(degree_bucket, predicted_class, node, "joint")
        if len(joint.get(joint_key, ())) >= 2:
            tier, key = "joint", joint_key
        else:
            degree_key = _bucket_key(degree_bucket, predicted_class, node, "degree")
            if len(by_degree.get(degree_key, ())) >= 2:
                tier, key = "degree", degree_key
            else:
                tier, key = "global", (0,)
        strata[(tier, key)].append(int(node))

    result: dict[int, int] = {}
    for group_id, ((tier, key), group_targets) in enumerate(sorted(strata.items(), key=lambda x: repr(x[0]))):
        pool = joint[key] if tier == "joint" else by_degree[key] if tier == "degree" else all_nodes
        rng = np.random.default_rng(int(seed) + 104729 * group_id)
        assigned, reuse_count = _assign_donors(group_targets, pool, rng)
        result.update(assigned)
        counts[tier] += len(group_targets)
        counts["donor_reuse_or_singleton"] += int(reuse_count)
    donors = torch.tensor([result[int(node)] for node in targets], dtype=torch.long)
    counts["self_donor_count"] = int((donors == torch.as_tensor(targets)).sum().item())
    if donors.shape != (len(targets),):
        raise AssertionError("coherent donor ids must contain exactly one donor per target")
    return donors, dict(counts)


def make_signed_permutation(hidden_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic hidden-coordinate permutation and independent signs."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(int(hidden_dim), generator=generator)
    signs = torch.where(
        torch.randint(0, 2, (int(hidden_dim),), generator=generator) == 0,
        -torch.ones(int(hidden_dim)),
        torch.ones(int(hidden_dim)),
    )
    return permutation, signs


def apply_signed_permutation(
    directions: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    if directions.size(-1) != permutation.numel() or signs.shape != permutation.shape:
        raise ValueError("signed permutation shape does not match direction hidden dimension")
    return directions.index_select(-1, permutation.to(directions.device)) * signs.to(
        device=directions.device, dtype=directions.dtype
    )


def direction_geometry(directions: torch.Tensor, eps: float = 1e-12) -> dict[str, Any]:
    """Summarize unique basis directions [nodes, orders, hidden] by their Gram geometry."""
    if directions.ndim != 3 or directions.size(1) < 2:
        raise ValueError("directions must have shape [nodes, >=2 orders, hidden]")
    x = directions.detach().float()
    normalized = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    gram = normalized @ normalized.transpose(1, 2)
    order = x.size(1)
    offdiag = ~torch.eye(order, dtype=torch.bool, device=x.device)
    pairwise = gram[:, offdiag]
    raw_gram = x @ x.transpose(1, 2)
    eigenvalues = torch.linalg.eigvalsh(raw_gram).clamp_min(0).flip(-1)
    probabilities = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    effective_rank = entropy.exp()
    return {
        "mean_pairwise_cosine": float(pairwise.mean().item()),
        "mean_absolute_pairwise_cosine": float(pairwise.abs().mean().item()),
        "mean_gram_eigenvalues_descending": eigenvalues.mean(dim=0).cpu().tolist(),
        "mean_effective_rank": float(effective_rank.mean().item()),
        "median_effective_rank": float(effective_rank.median().item()),
        "node_count": int(x.size(0)),
    }


def gram_preservation_errors(original: torch.Tensor, transformed: torch.Tensor,
                             eps: float = 1e-12) -> dict[str, float]:
    """Compare batched direction Gram matrices with scale-aware relative error.

    Elementwise relative error is also returned for transparency, but values near
    zero are ill-conditioned and must not determine the QA pass threshold.
    """
    gram = original.float() @ original.float().transpose(1, 2)
    transformed_gram = transformed.float() @ transformed.float().transpose(1, 2)
    absolute = (gram - transformed_gram).abs()
    per_node_scale = gram.abs().flatten(1).amax(dim=1).clamp_min(float(eps))
    per_node_absolute = absolute.flatten(1).amax(dim=1)
    scale_relative = per_node_absolute / per_node_scale
    elementwise_relative = absolute / gram.abs().clamp_min(1e-8)
    return {
        "absolute_max": float(absolute.max().item()),
        "scale_normalized_relative_max": float(scale_relative.max().item()),
        "elementwise_relative_max_near_zero_sensitive": float(elementwise_relative.max().item()),
    }


def build_relation_context(
    edge_index: torch.Tensor,
    h0_text: torch.Tensor,
    h0_visual: torch.Tensor,
    global_text: torch.Tensor,
    global_visual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build 25 label-free edge-cue summaries using physical undirected neighbors."""
    tensors = (h0_text, h0_visual, global_text, global_visual)
    if any(x.ndim != 2 for x in tensors) or any(x.size(0) != h0_text.size(0) for x in tensors):
        raise ValueError("all host representations must have shape [nodes, hidden]")
    device = h0_text.device
    num_nodes = h0_text.size(0)
    edge = edge_index.detach().to(device=device, dtype=torch.long)
    edge, _ = remove_self_loops(edge)
    edge = to_undirected(edge, num_nodes=num_nodes)
    edge = coalesce(edge, num_nodes=num_nodes)
    receiver, neighbor = edge
    if receiver.numel():
        keep = receiver != neighbor
        receiver, neighbor = receiver[keep], neighbor[keep]

    normalized = [F.normalize(x.detach().float(), dim=-1, eps=1e-12) for x in tensors]
    cues = []
    if receiver.numel():
        h0_t = (normalized[0][receiver] * normalized[0][neighbor]).sum(-1)
        h0_v = (normalized[1][receiver] * normalized[1][neighbor]).sum(-1)
        cg_t = (normalized[2][receiver] * normalized[2][neighbor]).sum(-1)
        cg_v = (normalized[3][receiver] * normalized[3][neighbor]).sum(-1)
        cues = [h0_t, h0_v, cg_t, cg_v, (h0_t - h0_v).abs(), (cg_t - cg_v).abs()]
        edge_cues = torch.stack(cues, dim=1)
    else:
        edge_cues = torch.empty((0, 6), device=device, dtype=torch.float32)

    degree = torch.bincount(receiver, minlength=num_nodes).to(torch.float32)
    count = degree.clamp_min(1).unsqueeze(-1)
    sums = torch.zeros((num_nodes, 6), device=device, dtype=torch.float32)
    sums_sq = torch.zeros_like(sums)
    minimum = torch.full_like(sums, float("inf"))
    maximum = torch.full_like(sums, float("-inf"))
    if receiver.numel():
        sums.index_add_(0, receiver, edge_cues)
        sums_sq.index_add_(0, receiver, edge_cues.square())
        idx = receiver[:, None].expand(-1, 6)
        minimum.scatter_reduce_(0, idx, edge_cues, reduce="amin", include_self=True)
        maximum.scatter_reduce_(0, idx, edge_cues, reduce="amax", include_self=True)
    means = sums / count
    std = (sums_sq / count - means.square()).clamp_min(0).sqrt()
    isolated = degree == 0
    minimum[isolated] = 0
    maximum[isolated] = 0
    summary = torch.cat([means, std, minimum, maximum], dim=1)
    context = torch.cat([summary, degree.log1p().unsqueeze(-1)], dim=1)
    disagreement = means[:, 4]
    if not torch.isfinite(context).all() or not torch.isfinite(disagreement).all():
        raise FloatingPointError("relation context contains nonfinite values")
    return context, disagreement, {
        "isolated_node_count": int(isolated.sum().item()),
        "physical_undirected_edge_count": int(receiver.numel() // 2),
        "context_dim": int(context.size(1)),
    }


def _derangement(values: list[int], rng: np.random.Generator) -> dict[int, int]:
    if len(values) < 2:
        return {int(values[0]): int(values[0])} if values else {}
    shuffled = [int(x) for x in rng.permutation(values).tolist()]
    shift = int(rng.integers(1, len(shuffled)))
    return dict(zip(shuffled, shuffled[shift:] + shuffled[:shift], strict=True))


def matched_context_shuffle(
    split_idx: torch.Tensor,
    degree_bucket: torch.Tensor,
    predicted_class: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Permute contexts only within one supplied split, with joint/degree/global fallback."""
    idx = torch.as_tensor(split_idx, dtype=torch.long).cpu().tolist()
    if len(set(idx)) != len(idx):
        raise ValueError("split_idx must not contain duplicate nodes")
    from src.analysis.r3mag_h1_response_audit import _bucket_key

    counts: Counter[str] = Counter()
    donor_by_target: dict[int, int] = {}
    remaining: set[int] = set(int(x) for x in idx)
    rng = np.random.default_rng(int(seed))

    # First use joint buckets, then collect unmatched singletons by degree, then global.
    joint_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node in idx:
        joint_groups[_bucket_key(degree_bucket, predicted_class, int(node), "joint")].append(int(node))
    for key, values in sorted(joint_groups.items()):
        if len(values) >= 2:
            group_seed = np.random.default_rng(rng.integers(0, 2**32 - 1))
            donor_by_target.update(_derangement(values, group_seed))
            remaining.difference_update(values)
            counts["joint"] += len(values)

    degree_groups: dict[int, list[int]] = defaultdict(list)
    for node in sorted(remaining):
        degree_groups[_bucket_key(degree_bucket, predicted_class, node, "degree")].append(node)
    for key, values in sorted(degree_groups.items()):
        if len(values) >= 2:
            group_seed = np.random.default_rng(rng.integers(0, 2**32 - 1))
            donor_by_target.update(_derangement(values, group_seed))
            remaining.difference_update(values)
            counts["degree"] += len(values)

    if len(remaining) >= 2:
        values = sorted(remaining)
        group_seed = np.random.default_rng(rng.integers(0, 2**32 - 1))
        donor_by_target.update(_derangement(values, group_seed))
        counts["global"] += len(values)
        remaining.clear()
    for node in remaining:
        donor_by_target[node] = node
        counts["singleton_self_map"] += 1

    donors = torch.tensor([donor_by_target[int(node)] for node in idx], dtype=torch.long)
    counts["self_map_count"] = int((donors == torch.as_tensor(idx)).sum().item())
    counts["split_size"] = len(idx)
    if not set(donors.tolist()).issubset(set(idx)):
        raise AssertionError("context shuffle donor crossed the supplied split boundary")
    if counts["self_map_count"] != counts.get("singleton_self_map", 0):
        raise AssertionError("only unavoidable singleton groups may self-map during context shuffle")
    return donors, dict(counts)


class ResponsePredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualAdapter(nn.Module):
    """Two-modality residual and scalar gate head; host modules remain external/frozen."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, response_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.response_dim = int(response_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2 * (self.response_dim + 1)),
        )
        final = self.network[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[self.response_dim] = -2.0
            final.bias[2 * (self.response_dim + 1) - 1] = -2.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.network(x).view(x.size(0), 2, self.response_dim + 1)
        residual = values[:, :, : self.response_dim]
        gate_logits = values[:, :, self.response_dim]
        return residual, gate_logits


def apply_bounded_residual(
    global_text: torch.Tensor,
    global_visual: torch.Tensor,
    raw_residual: torch.Tensor,
    gate_logits: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_residual.shape != (global_text.size(0), 2, global_text.size(1)):
        raise ValueError("residual must have shape [nodes, 2, hidden]")
    if gate_logits.shape != (global_text.size(0), 2):
        raise ValueError("gate_logits must have shape [nodes, 2]")
    global_pair = torch.stack([global_text, global_visual], dim=1)
    residual_norm = raw_residual.norm(dim=-1, keepdim=True)
    direction = raw_residual / (residual_norm + float(eps))
    direction = torch.where(residual_norm > float(eps), direction, torch.zeros_like(direction))
    gates = gate_logits.sigmoid()
    prior_norm = global_pair.norm(dim=-1, keepdim=True)
    corrected = global_pair + 0.2 * gates.unsqueeze(-1) * prior_norm * direction
    ratio = (corrected - global_pair).norm(dim=-1) / prior_norm.squeeze(-1).clamp_min(float(eps))
    return corrected[:, 0], corrected[:, 1], gates, ratio
