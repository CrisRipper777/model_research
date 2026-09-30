import torch
from torch_geometric.utils import softmax as pyg_segment_softmax


def segment_softmax(scores, dst, num_nodes, mask=None):
    if mask is None:
        return pyg_segment_softmax(scores, dst, num_nodes=num_nodes)

    out = torch.zeros_like(scores)
    if mask.any():
        out[mask] = pyg_segment_softmax(scores[mask], dst[mask], num_nodes=num_nodes)
    return out


def weighted_message_passing(x, edge_index, edge_weight, num_nodes, eps=1e-8):
    src = edge_index[0]
    dst = edge_index[1]
    acc_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    edge_weight = edge_weight.to(device=x.device, dtype=acc_dtype)
    x_acc = x.to(dtype=acc_dtype)

    msg = x_acc[src] * edge_weight.unsqueeze(-1)
    out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=acc_dtype)
    out.index_add_(0, dst, msg)

    normalizer = torch.zeros(num_nodes, device=x.device, dtype=acc_dtype)
    normalizer.index_add_(0, dst, edge_weight)
    normalizer = normalizer.clamp_min(max(float(eps), torch.finfo(acc_dtype).tiny))

    out = out / normalizer.unsqueeze(-1)
    return out.to(dtype=x.dtype)


def topk_incoming_mask(dst, score, num_nodes, k):
    if k <= 0:
        return torch.zeros_like(score, dtype=torch.bool)

    num_edges = score.numel()
    if num_edges == 0:
        return torch.zeros_like(score, dtype=torch.bool)

    mask = torch.zeros_like(score, dtype=torch.bool)

    # First sort by score (descending), then stable-sort by destination so edges
    # for each dst are contiguous while preserving descending score order.
    score_order = torch.argsort(score, descending=True, stable=True)
    grouped_order = score_order[torch.argsort(dst[score_order], stable=True)]
    dst_grouped = dst[grouped_order]

    new_group = torch.ones(num_edges, dtype=torch.bool, device=dst.device)
    new_group[1:] = dst_grouped[1:] != dst_grouped[:-1]
    group_start = torch.nonzero(new_group, as_tuple=False).flatten()
    group_counts = torch.diff(torch.cat([group_start, group_start.new_tensor([num_edges])]))

    elem_pos = torch.arange(num_edges, device=dst.device)
    elem_group_start = torch.repeat_interleave(group_start, group_counts)
    rank_in_group = elem_pos - elem_group_start

    keep_grouped = rank_in_group < k
    mask[grouped_order[keep_grouped]] = True
    return mask
