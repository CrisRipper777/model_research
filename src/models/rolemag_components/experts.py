import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .utils import segment_softmax, topk_incoming_mask, weighted_message_passing


class RoleMAGSharedExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_layers=3, edge_floor=1.0, use_residual=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.edge_floor = float(edge_floor)
        self.use_residual = bool(use_residual)
        self.dropout = dropout
        convs = [GCNConv(self.input_dim, hidden_dim, add_self_loops=True, normalize=True)]
        for _ in range(self.num_layers - 1):
            convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True))
        self.convs = nn.ModuleList(convs)
        self.mix = nn.Linear(hidden_dim * self.num_layers, hidden_dim)
        self.base_skip = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_base, raw_feature, edge_index, edge_weight):
        shared_edge_weight = edge_weight.to(device=h_base.device, dtype=h_base.dtype)
        shared_edge_weight = shared_edge_weight + self.edge_floor

        h = raw_feature
        layer_outputs = []
        for layer_idx, conv in enumerate(self.convs):
            h_prev = h
            h = conv(h, edge_index.to(h.device), edge_weight=shared_edge_weight)
            if self.use_residual and h_prev.size(-1) == h.size(-1):
                h = h + h_prev
            if layer_idx != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        h = self.mix(torch.cat(layer_outputs, dim=-1)) + self.base_skip(h_base)
        h = F.relu(self.proj(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class RoleMAGPolyH2Expert(nn.Module):
    def __init__(self, hidden_dim, dropout, init_theta=(1.0, -0.5, 0.5)):
        super().__init__()
        self.theta0_raw = nn.Parameter(torch.atanh(torch.tensor(float(init_theta[0]) / 2.0)))
        self.theta1_raw = nn.Parameter(torch.atanh(torch.tensor(float(init_theta[1]) / 2.0)))
        self.theta2_raw = nn.Parameter(torch.atanh(torch.tensor(float(init_theta[2]) / 2.0)))
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    @property
    def theta0(self):
        return 2.0 * torch.tanh(self.theta0_raw)

    @property
    def theta1(self):
        return 2.0 * torch.tanh(self.theta1_raw)

    @property
    def theta2(self):
        return 2.0 * torch.tanh(self.theta2_raw)

    def forward(self, h_base, edge_index, edge_weight):
        u1 = weighted_message_passing(h_base, edge_index, edge_weight, h_base.size(0))
        u2 = weighted_message_passing(u1, edge_index, edge_weight, h_base.size(0))

        out = self.theta0 * h_base + self.theta1 * u1 + self.theta2 * u2
        out = F.relu(self.proj(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class RoleMAGComplementaryExpert(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_queries,
        topk,
        lambda_bias,
        dropout,
        eps=1e-8,
        disable_topk=False,
        disable_attention_bias=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.topk = topk
        self.lambda_bias = lambda_bias
        self.eps = eps
        self.dropout = dropout
        self.disable_topk = bool(disable_topk)
        self.disable_attention_bias = bool(disable_attention_bias)

        self.q_text = nn.Linear(hidden_dim, hidden_dim)
        self.q_image = nn.Linear(hidden_dim, hidden_dim)
        self.k_text = nn.Linear(hidden_dim, hidden_dim)
        self.k_image = nn.Linear(hidden_dim, hidden_dim)
        self.v_text = nn.Linear(hidden_dim, hidden_dim)
        self.v_image = nn.Linear(hidden_dim, hidden_dim)

        self.query_tokens_t2i = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.query_tokens_i2t = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _topk_with_base_mask(self, dst, score, num_nodes, base_mask):
        if not base_mask.any():
            return torch.zeros_like(score, dtype=torch.bool)
        selected = torch.nonzero(base_mask, as_tuple=False).flatten()
        kept_selected = topk_incoming_mask(dst[selected], score[selected], num_nodes, self.topk)
        out = torch.zeros_like(score, dtype=torch.bool)
        out[selected[kept_selected]] = True
        return out

    def _build_direction_masks(self, edge_index, w_comp, w_t2i, w_i2t, num_nodes=None):
        dst = edge_index[1]
        if num_nodes is None:
            num_nodes = int(dst.max().item()) + 1 if dst.numel() > 0 else 0
        if self.disable_topk:
            keep = w_comp > 0
            return keep, keep
        comp_mask = topk_incoming_mask(dst, w_comp, num_nodes, self.topk)
        mask_t2i = self._topk_with_base_mask(dst, w_t2i, num_nodes, comp_mask)
        mask_i2t = self._topk_with_base_mask(dst, w_i2t, num_nodes, comp_mask)
        return mask_t2i, mask_i2t

    def _run_direction(self, q_base, k_base, v_base, edge_index, edge_weight, edge_mask, tokens):
        src = edge_index[0]
        dst = edge_index[1]
        num_nodes = q_base.size(0)
        edge_weight = edge_weight.to(device=q_base.device, dtype=q_base.dtype)

        k_src = k_base[src]
        v_src = v_base[src]
        q_dst_base = q_base[dst]
        bias = torch.zeros_like(edge_weight) if self.disable_attention_bias else self.lambda_bias * torch.log(edge_weight + self.eps)

        outputs = []
        scale = math.sqrt(float(self.hidden_dim))
        for idx in range(self.num_queries):
            q_dst = q_dst_base + tokens[idx]
            score = (q_dst * k_src).sum(dim=-1) / scale + bias
            alpha = segment_softmax(score, dst, num_nodes, mask=edge_mask)
            alpha = alpha.to(dtype=q_base.dtype)

            msg = v_src * alpha.unsqueeze(-1)
            out = torch.zeros(num_nodes, self.hidden_dim, device=q_base.device, dtype=q_base.dtype)
            out.index_add_(0, dst, msg)
            outputs.append(out)

        return torch.stack(outputs, dim=0).mean(dim=0)

    def forward(self, h_t, h_i, edge_index, w_t2i, w_i2t, w_comp=None, return_aux=False):
        q_t = self.q_text(h_t)
        q_i = self.q_image(h_i)

        k_t = self.k_text(h_t)
        k_i = self.k_image(h_i)

        v_t = self.v_text(h_t)
        v_i = self.v_image(h_i)

        if w_comp is None:
            w_comp = w_t2i + w_i2t

        mask_t2i, mask_i2t = self._build_direction_masks(
            edge_index=edge_index,
            w_comp=w_comp,
            w_t2i=w_t2i,
            w_i2t=w_i2t,
            num_nodes=h_t.size(0),
        )

        out_t2i = self._run_direction(q_t, k_i, v_i, edge_index, w_t2i, mask_t2i, self.query_tokens_t2i)
        out_i2t = self._run_direction(q_i, k_t, v_t, edge_index, w_i2t, mask_i2t, self.query_tokens_i2t)

        out = self.merge(torch.cat([out_t2i, out_i2t], dim=-1))
        out = F.dropout(out, p=self.dropout, training=self.training)
        if return_aux:
            return out, {"z_t2i": out_t2i, "z_i2t": out_i2t}
        return out
