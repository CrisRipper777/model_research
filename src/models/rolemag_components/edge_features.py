import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleMAGEdgeFeatureBuilder(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self._cached_structure_index: torch.Tensor | None = None
        self._cached_structure_source: torch.Tensor | None = None
        self._cached_structure_source_version: int | None = None
        self._cached_structure_features: torch.Tensor | None = None
        self._cached_structure_num_nodes: int | None = None

    def _normalize_for_similarity(self, x):
        return F.layer_norm(x, x.shape[-1:])

    def _build_undirected_sparse_adjacency(self, src, dst, num_nodes):
        row = torch.cat([src, dst], dim=0).to(torch.long)
        col = torch.cat([dst, src], dim=0).to(torch.long)
        keep = row != col
        row = row[keep]
        col = col[keep]

        if row.numel() == 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=src.device)
            empty_val = torch.empty((0,), dtype=torch.float32, device=src.device)
            return torch.sparse_coo_tensor(empty_idx, empty_val, (num_nodes, num_nodes), device=src.device).coalesce()

        indices = torch.stack([row, col], dim=0)
        values = torch.ones(row.numel(), dtype=torch.float32, device=src.device)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=src.device).coalesce()
        # Keep binary adjacency semantics after coalescing duplicate edges.
        adj = torch.sparse_coo_tensor(
            adj.indices(),
            torch.ones_like(adj.values()),
            (num_nodes, num_nodes),
            device=src.device,
        ).coalesce()
        return adj

    def _sparse_lookup(self, sparse_mat, row, col, num_nodes):
        if sparse_mat._nnz() == 0:
            return torch.zeros(row.numel(), dtype=torch.float32, device=row.device)

        indices = sparse_mat.indices()
        values = sparse_mat.values()
        keys = indices[0].to(torch.long) * num_nodes + indices[1].to(torch.long)

        order = torch.argsort(keys)
        keys_sorted = keys[order]
        values_sorted = values[order]

        query_keys = row.to(torch.long) * num_nodes + col.to(torch.long)
        pos = torch.searchsorted(keys_sorted, query_keys)
        in_range = pos < keys_sorted.numel()
        clipped = pos.clamp(max=keys_sorted.numel() - 1)
        found = in_range & (keys_sorted[clipped] == query_keys)

        out = torch.zeros(row.numel(), dtype=values_sorted.dtype, device=row.device)
        out[found] = values_sorted[clipped[found]]
        return out

    def _column_weighted_sparse_adjacency(self, adj, column_weight):
        indices = adj.indices()
        values = adj.values() * column_weight[indices[1]]
        return torch.sparse_coo_tensor(indices, values, adj.size(), device=indices.device).coalesce()

    def forward(self, h_v, h_t, edge_index, v_mask, t_mask, structure_edge_index=None):
        src = edge_index[0]
        dst = edge_index[1]
        num_nodes = h_v.size(0)
        structure_edge_index = edge_index if structure_edge_index is None else structure_edge_index
        ref_src = structure_edge_index[0]
        ref_dst = structure_edge_index[1]

        norm_h_v = self._normalize_for_similarity(h_v)
        norm_h_t = self._normalize_for_similarity(h_t)

        hv_src = norm_h_v[src]
        hv_dst = norm_h_v[dst]
        ht_src = norm_h_t[src]
        ht_dst = norm_h_t[dst]

        valid_v = (v_mask[src] & v_mask[dst]).float()
        valid_t = (t_mask[src] & t_mask[dst]).float()

        s_i = F.cosine_similarity(hv_src, hv_dst, dim=-1) * valid_v
        s_t = F.cosine_similarity(ht_src, ht_dst, dim=-1) * valid_t
        delta_ti = (s_t - s_i).abs()

        v_sem = torch.stack(
            [
                s_t,
                s_i,
                delta_ti,
                t_mask[src].float(),
                v_mask[src].float(),
                t_mask[dst].float(),
                v_mask[dst].float(),
            ],
            dim=-1,
        )

        same_structure = (
            self._cached_structure_index is not None
            and self._cached_structure_num_nodes == num_nodes
            and self._cached_structure_index.device == structure_edge_index.device
            and self._cached_structure_index.shape == structure_edge_index.shape
            and (
                (
                    self._cached_structure_source is structure_edge_index
                    and self._cached_structure_source_version
                    == int(getattr(structure_edge_index, "_version", 0))
                )
                or torch.equal(self._cached_structure_index, structure_edge_index)
            )
        )
        same_query_edges = edge_index is structure_edge_index or torch.equal(edge_index, structure_edge_index)
        if same_structure and same_query_edges and self._cached_structure_features is not None:
            v_str = self._cached_structure_features.to(dtype=v_sem.dtype)
        else:
            adj = self._build_undirected_sparse_adjacency(ref_src, ref_dst, num_nodes)
            degree = torch.sparse.sum(adj, dim=1).to_dense().float()

            src_deg = degree[src]
            dst_deg = degree[dst]
            common_mat = torch.sparse.mm(adj, adj).coalesce()
            common_count = self._sparse_lookup(common_mat, src, dst, num_nodes).float()
            union_count = src_deg + dst_deg - common_count
            jaccard = common_count / union_count.clamp_min(1.0)

            inv_log_degree = 1.0 / torch.log(degree.clamp_min(2.0) + self.eps)
            aa_mat = torch.sparse.mm(
                self._column_weighted_sparse_adjacency(adj, inv_log_degree), adj
            ).coalesce()
            aa = self._sparse_lookup(aa_mat, src, dst, num_nodes).float()

            observed = self._sparse_lookup(adj, src, dst, num_nodes).float()
            preferential_attachment = src_deg * dst_deg

            v_str = torch.stack(
                [
                    observed,
                    common_count,
                    jaccard,
                    aa,
                    preferential_attachment,
                    torch.log1p(src_deg),
                    torch.log1p(dst_deg),
                ],
                dim=-1,
            )
            if same_query_edges:
                self._cached_structure_index = structure_edge_index.detach().clone()
                self._cached_structure_source = structure_edge_index
                self._cached_structure_source_version = int(
                    getattr(structure_edge_index, "_version", 0)
                )
                self._cached_structure_num_nodes = num_nodes
                self._cached_structure_features = v_str.detach()

        return v_sem, v_str.to(v_sem.dtype)
