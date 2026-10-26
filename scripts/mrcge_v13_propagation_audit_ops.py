"""Independent, read-only reconstructions of frozen physical message formulas."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def physical_operator(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    loops = edge_index[0] == edge_index[1]
    kept = edge_index[:, ~loops].long()
    src, dst = kept
    degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
    if dst.numel():
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=dtype, device=edge_index.device))
    inv = degree.clamp_min(1).rsqrt()
    weight = inv[src] * inv[dst]
    return kept, weight, degree > 0, int(loops.sum().item())


def sparse_propagate(value: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(value)
    if edge_index.size(1):
        out.index_add_(0, edge_index[1], value[edge_index[0]] * edge_weight.to(value.dtype).unsqueeze(-1))
    return out


def v13_basis(initial: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
              support: torch.Tensor, max_order: int = 3) -> list[torch.Tensor]:
    result = [F.layer_norm(initial, (initial.size(-1),))]
    for _ in range(max_order):
        result.append(sparse_propagate(result[-1], edge_index, edge_weight) * support.to(initial.dtype).unsqueeze(-1))
    return result


def early_cross(source: torch.Tensor, projection, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                support: torch.Tensor, reference: torch.Tensor, eps: float = 1.0e-8):
    projected = projection(source)
    raw = sparse_propagate(projected, edge_index, edge_weight) * support.to(projected.dtype).unsqueeze(-1)
    if not bool(support.any()):
        return raw, torch.zeros_like(raw), 0.0
    raw_rms = raw[support].detach().float().square().mean().sqrt()
    target_rms = reference[support].detach().float().square().mean().sqrt()
    if float(raw_rms) <= eps:
        return raw, torch.zeros_like(raw), 0.0
    scale = float((target_rms / (raw_rms + eps)).detach())
    return raw, raw * scale, scale
