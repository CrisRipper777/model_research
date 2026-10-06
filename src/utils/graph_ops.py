from __future__ import annotations

import torch
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected


def normalized_adjacency_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build sparse symmetric-normalized ``A + I`` for an undirected graph."""
    edge = edge_index.detach().to(device=device, dtype=torch.long)
    edge, _ = remove_self_loops(edge)
    edge = to_undirected(edge, num_nodes=num_nodes)
    edge, _ = add_self_loops(edge, num_nodes=num_nodes)
    edge = coalesce(edge, num_nodes=num_nodes)
    row, col = edge
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.index_add_(0, row, torch.ones(row.numel(), dtype=dtype, device=device))
    inv_sqrt_degree = degree.clamp_min(1).pow(-0.5)
    weight = inv_sqrt_degree[row] * inv_sqrt_degree[col]
    return torch.sparse_coo_tensor(
        edge, weight, (num_nodes, num_nodes), device=device, dtype=dtype
    ).coalesce()
