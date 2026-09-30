from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = {"semantic", "uniform", "extent", "single_basis", "multi_basis", "extent_wide", "single_basis_static", "single_basis_target"}


def remove_self_messages(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep every directed physical message except self-loops; return edge/src/dst."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    src, dst = edge_index.long()
    keep = src != dst
    return edge_index[:, keep], src[keep], dst[keep]


def incoming_degree(dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.long, device=dst.device)
    if dst.numel():
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.long))
    return degree


def graph_standardized_log_degree(degree: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    value = torch.log1p(degree.to(torch.float32))
    std = value.std(unbiased=False)
    return (value - value.mean()) / std.clamp_min(eps)


def fixed_degree_mean(aggregate: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    """Normalize by original physical in-degree, regardless of control values."""
    return aggregate / degree.clamp_min(1).to(aggregate.dtype).unsqueeze(-1)


def compose_message(
    base: torch.Tensor,
    gate: torch.Tensor,
    correction: torch.Tensor | None = None,
    correction_strength: torch.Tensor | None = None,
) -> torch.Tensor:
    if correction is None:
        return gate * base
    if correction_strength is None:
        raise ValueError("correction_strength is required when correction is provided")
    return gate * (base + correction_strength * correction)


def neighborhood_shuffle_indices(dst: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a per-target permutation, preserving each target's control multiset."""
    dst_cpu = dst.detach().cpu().long()
    order = torch.argsort(dst_cpu, stable=True)
    sorted_dst = dst_cpu[order]
    result = torch.arange(dst_cpu.numel(), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    cursor = 0
    while cursor < order.numel():
        stop = cursor + 1
        while stop < order.numel() and sorted_dst[stop] == sorted_dst[cursor]:
            stop += 1
        group = order[cursor:stop]
        if group.numel() > 1:
            result[group] = group[torch.randperm(group.numel(), generator=generator)]
        cursor = stop
    return result


def target_mean_control(values: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Replace each edge value with the mean over its destination neighborhood."""
    if values.size(0) != dst.numel():
        raise ValueError("control rows and destination indices must have equal length")
    if dst.numel() == 0:
        return values.clone()
    sums = values.new_zeros((num_nodes, *values.shape[1:]))
    sums.index_add_(0, dst, values)
    degree = incoming_degree(dst, num_nodes).clamp_min(1)
    means = sums / degree.reshape((num_nodes,) + (1,) * (values.ndim - 1)).to(values.dtype)
    return means[dst]


def global_mean_control(values: torch.Tensor) -> torch.Tensor:
    """Replace every edge value with its global mean; preserve an empty input."""
    if values.size(0) == 0:
        return values.clone()
    return values.mean(dim=0, keepdim=True).expand_as(values).clone()


def tie_modality_controls(controls: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    tied = {}
    for name, (text, visual) in controls.items():
        average = (text + visual) * 0.5
        if name == "pi":
            average = average / average.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        tied[name] = (average, average.clone())
    return tied


def leave_one_out_context(
    projected: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Cosine(P_j, mean_{u in N(i)\{j}} P_u), detached routing evidence."""
    sums = projected.new_zeros(projected.shape)
    if src.numel():
        sums.index_add_(0, dst, projected[src])
    remaining = degree[dst] - 1
    denominator = remaining.clamp_min(1).to(projected.dtype).unsqueeze(-1)
    loo_mean = (sums[dst] - projected[src]) / denominator
    cosine = F.cosine_similarity(projected[src], loo_mean, dim=-1, eps=eps)
    return torch.where(remaining > 0, cosine, torch.zeros_like(cosine)).detach().unsqueeze(-1)


class _Projector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PairEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RelationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ModalityInterpreter(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    """One-step, modality-resolved adaptive propagation mechanism screen."""

    def __init__(self, cfg: Any, data_info: dict[str, int]):
        super().__init__()
        model_cfg = cfg.model
        self.variant = str(model_cfg.get("variant", "semantic")).strip().lower()
        if self.variant not in VARIANTS:
            raise ValueError(f"model.variant must be one of {sorted(VARIANTS)}, got {self.variant!r}")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.relation_dim = int(model_cfg.get("relation_dim", 32))
        self.relation_state_dim = int(model_cfg.get("relation_state_dim", 64))
        self.modality_embed_dim = int(model_cfg.get("modality_embed_dim", 8))
        self.edge_chunk_size = int(model_cfg.get("edge_chunk_size", 100000))
        self.eps = float(model_cfg.get("eps", 1e-8))
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")

        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim < 1 or self.visual_dim < 1:
            raise ValueError("adaptive_prop_m0 requires non-empty Text and Visual feature slices")
        if self.text_dim + self.visual_dim != int(data_info["input_dim"]):
            raise ValueError(
                "Text/Visual feature dimensions do not cover data.x: "
                f"{self.text_dim}+{self.visual_dim}!={data_info['input_dim']}"
            )

        dropout = float(model_cfg.get("dropout", 0.2))
        # Keep the A/B/C common modules in one fixed construction order. Variant
        # heads and correction factors are created only after all common modules.
        self.proj_t = _Projector(self.text_dim, self.hidden_dim, dropout)
        self.proj_v = _Projector(self.visual_dim, self.hidden_dim, dropout)

        self.relation_enabled = self.variant in {"extent", "single_basis", "multi_basis", "extent_wide", "single_basis_static", "single_basis_target"}
        if self.relation_enabled:
            self.rel_proj_t = nn.Sequential(
                nn.Linear(self.hidden_dim, self.relation_dim, bias=False),
                nn.LayerNorm(self.relation_dim),
            )
            self.rel_proj_v = nn.Sequential(
                nn.Linear(self.hidden_dim, self.relation_dim, bias=False),
                nn.LayerNorm(self.relation_dim),
            )
            pair_input_dim = 4 * self.relation_dim + 4
            self.phi_pair = _PairEncoder(pair_input_dim)
            self.phi_rel = _RelationEncoder()
            self.modality_embeddings = nn.Parameter(torch.empty(2, self.modality_embed_dim))
            nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
            self.phi_mod = _ModalityInterpreter(64 + 32 + self.modality_embed_dim)

        self.w0 = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2)]
        ) if self.variant != "semantic" else None
        if self.w0 is not None:
            for layer in self.w0:
                nn.init.xavier_uniform_(layer.weight)

        self.fusion = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.residual_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(2)])
        self.out_dim = self.hidden_dim

        self.gate_heads: nn.ModuleList | None = None
        self.correction_heads: nn.ModuleList | None = None
        self.basis_heads: nn.ModuleList | None = None
        self.static_c_logits: nn.Parameter | None = None
        self.basis_u: nn.Parameter | None = None
        self.basis_v: nn.Parameter | None = None
        if self.relation_enabled:
            if self.variant == "extent_wide":
                gate_hidden = int(model_cfg.get("wide_gate_hidden_dim", 126))
                if gate_hidden != 126:
                    raise ValueError("M0.1 A-wide fixes wide_gate_hidden_dim at 126")
                self.gate_heads = nn.ModuleList([
                    nn.Sequential(nn.Linear(64, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))
                    for _ in range(2)
                ])
                for head in self.gate_heads:
                    nn.init.normal_(head[0].weight, mean=0.0, std=1e-3)
                    nn.init.zeros_(head[0].bias)
                    nn.init.normal_(head[2].weight, mean=0.0, std=1e-3)
                    nn.init.constant_(head[2].bias, 2.0)
            else:
                self.gate_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(2)])
                for head in self.gate_heads:
                    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                    nn.init.constant_(head.bias, 2.0)
            if self.variant in {"single_basis", "multi_basis", "single_basis_target"}:
                self.correction_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(2)])
                for head in self.correction_heads:
                    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                    nn.init.constant_(head.bias, -2.0)
            elif self.variant == "single_basis_static":
                # Consume the same seeded draws as B's two correction heads so
                # the subsequently initialized rank-32 bases match B exactly.
                transient_heads = [nn.Linear(64, 1) for _ in range(2)]
                for head in transient_heads:
                    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                    nn.init.constant_(head.bias, -2.0)
                del transient_heads
                self.static_c_logits = nn.Parameter(torch.full((2,), -2.0))

            if self.variant in {"single_basis", "multi_basis", "single_basis_static", "single_basis_target"}:
                if self.variant == "multi_basis":
                    self.num_bases = 4
                    self.basis_rank = 8
                    self.basis_heads = nn.ModuleList([nn.Linear(64, 4) for _ in range(2)])
                    for head in self.basis_heads:
                        nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                        nn.init.zeros_(head.bias)
                else:
                    self.num_bases = 1
                    self.basis_rank = 32
                self.basis_v = nn.Parameter(torch.empty(2, self.num_bases, self.basis_rank, self.hidden_dim))
                self.basis_u = nn.Parameter(torch.empty(2, self.num_bases, self.hidden_dim, self.basis_rank))
                for m in range(2):
                    for k in range(self.num_bases):
                        nn.init.xavier_uniform_(self.basis_v[m, k])
                        nn.init.xavier_uniform_(self.basis_u[m, k])
                        with torch.no_grad():
                            self.basis_u[m, k].mul_(0.05)

    def split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError(
                f"expected data.x with [Text({self.text_dim}), Visual({self.visual_dim})], "
                f"got {tuple(x.shape)}"
            )
        return x[:, : self.text_dim], x[:, self.text_dim :]

    def _prepare(self, x: torch.Tensor, edge_index: torch.Tensor):
        x_t, x_v = self.split_modalities(x)
        h0 = [self.proj_t(x_t), self.proj_v(x_v)]
        edge_index, src, dst = remove_self_messages(edge_index)
        degree = incoming_degree(dst, int(x.size(0)))
        deg_z = graph_standardized_log_degree(degree, self.eps).to(dtype=h0[0].dtype)
        if self.relation_enabled:
            p = [self.rel_proj_t(h0[0]), self.rel_proj_v(h0[1])]
            contexts = [leave_one_out_context(p[m], src, dst, degree, self.eps) for m in range(2)]
        else:
            p, contexts = None, None
        return h0, edge_index, src, dst, degree, deg_z, p, contexts

    def _pair_evidence(self, p, context, deg_z, src, dst):
        target, source = p[dst], p[src]
        sim = F.cosine_similarity(target, source, dim=-1, eps=self.eps).unsqueeze(-1)
        return torch.cat(
            [
                target,
                source,
                (target - source).abs(),
                target * source,
                sim,
                context,
                deg_z[dst].unsqueeze(-1),
                deg_z[src].unsqueeze(-1),
            ],
            dim=-1,
        )

    def _functional_states(self, p, contexts, deg_z, src, dst):
        evidence_t = self._pair_evidence(p[0], contexts[0], deg_z, src, dst)
        evidence_v = self._pair_evidence(p[1], contexts[1], deg_z, src, dst)
        q_t, q_v = self.phi_pair(evidence_t), self.phi_pair(evidence_v)
        relation = self.phi_rel(torch.cat([q_t, q_v, (q_t - q_v).abs(), q_t * q_v], dim=-1))
        u = []
        for m, q in enumerate((q_t, q_v)):
            em = self.modality_embeddings[m].expand(q.size(0), -1)
            u.append(self.phi_mod(torch.cat([relation, q, em], dim=-1)))
        return u

    def _controls(self, u: torch.Tensor, modality: int):
        assert self.gate_heads is not None
        g = torch.sigmoid(self.gate_heads[modality](u))
        c, pi = None, None
        if self.correction_heads is not None and self.variant != "single_basis_target":
            c = torch.sigmoid(self.correction_heads[modality](u))
        elif self.static_c_logits is not None:
            c = torch.sigmoid(self.static_c_logits[modality]).expand(u.size(0), 1)
        if self.basis_heads is not None:
            pi = torch.softmax(self.basis_heads[modality](u), dim=-1)
        return g, c, pi

    def _basis_correction(self, h_source: torch.Tensor, modality: int, pi: torch.Tensor | None):
        assert self.basis_u is not None and self.basis_v is not None
        basis_outputs = []
        for k in range(self.num_bases):
            low = F.linear(h_source, self.basis_v[modality, k])
            basis_outputs.append(F.linear(low, self.basis_u[modality, k]))
        stacked = torch.stack(basis_outputs, dim=1)
        if self.variant in {"single_basis", "single_basis_static", "single_basis_target"}:
            return stacked[:, 0, :]
        assert pi is not None
        return (stacked * pi.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def _overridden(value, overrides, name: str, modality: int, begin: int, end: int, device, dtype):
        if overrides is None or name not in overrides:
            return value
        replacement = overrides[name][modality][begin:end]
        return replacement.to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        control_overrides: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_diagnostics: bool = False,
    ):
        if edge_index is None:
            raise ValueError("adaptive_prop_m0 requires the physical edge_index")
        h0, edge_index, src, dst, degree, deg_z, p, contexts = self._prepare(x, edge_index)
        num_nodes, num_edges = int(x.size(0)), int(src.numel())
        delta = [h0[0].new_zeros((num_nodes, self.hidden_dim)), h0[1].new_zeros((num_nodes, self.hidden_dim))]
        diagnostic_lists: dict[str, list[list[torch.Tensor]]] = {
            name: [[], []] for name in ("g", "c", "pi", "correction_ratio")
        }
        target_correction = None
        if self.variant == "single_basis_target":
            # Two-pass chunking: first form the differentiable target mean of
            # relation-functional states, then recompute edge states for g.
            assert p is not None and contexts is not None and self.correction_heads is not None
            u_sums = [h0[0].new_zeros((num_nodes, 64)), h0[1].new_zeros((num_nodes, 64))]
            for begin in range(0, num_edges, self.edge_chunk_size):
                end = min(begin + self.edge_chunk_size, num_edges)
                u_pass = self._functional_states(
                    p,
                    [contexts[0][begin:end], contexts[1][begin:end]],
                    deg_z,
                    src[begin:end],
                    dst[begin:end],
                )
                for m in range(2):
                    u_sums[m] = u_sums[m].index_add(0, dst[begin:end], u_pass[m])
            degree_den = degree.clamp_min(1).to(h0[0].dtype).unsqueeze(-1)
            u_bar = [u_sums[m] / degree_den for m in range(2)]
            target_correction = [torch.sigmoid(self.correction_heads[m](u_bar[m])) for m in range(2)]

        if self.variant == "semantic":
            pass
        elif self.variant == "uniform":
            assert self.w0 is not None
            for m in range(2):
                aggregate = h0[m].new_zeros((num_nodes, self.hidden_dim))
                for begin in range(0, num_edges, self.edge_chunk_size):
                    end = min(begin + self.edge_chunk_size, num_edges)
                    message = self.w0[m](h0[m][src[begin:end]])
                    aggregate = aggregate.index_add(0, dst[begin:end], message)
                delta[m] = fixed_degree_mean(aggregate, degree)
        else:
            assert self.w0 is not None and p is not None and contexts is not None
            for begin in range(0, num_edges, self.edge_chunk_size):
                end = min(begin + self.edge_chunk_size, num_edges)
                cs, cd = src[begin:end], dst[begin:end]
                u = self._functional_states(
                    p,
                    [contexts[0][begin:end], contexts[1][begin:end]],
                    deg_z,
                    cs,
                    cd,
                )
                g_pair, c_pair, pi_pair = [], [], []
                for m in range(2):
                    g, c, pi = self._controls(u[m], m)
                    if target_correction is not None:
                        c = target_correction[m][cd]
                    g = self._overridden(g, control_overrides, "g", m, begin, end, x.device, x.dtype)
                    if c is not None:
                        c = self._overridden(c, control_overrides, "c", m, begin, end, x.device, x.dtype)
                    if pi is not None:
                        pi = self._overridden(pi, control_overrides, "pi", m, begin, end, x.device, x.dtype)
                    g_pair.append(g)
                    c_pair.append(c)
                    pi_pair.append(pi)
                    if return_diagnostics:
                        diagnostic_lists["g"][m].append(g.detach().cpu())
                        if c is not None:
                            diagnostic_lists["c"][m].append(c.detach().cpu())
                        if pi is not None:
                            diagnostic_lists["pi"][m].append(pi.detach().cpu())

                for m in range(2):
                    hs = h0[m][cs]
                    base = self.w0[m](hs)
                    if self.correction_heads is None and self.static_c_logits is None:
                        message = compose_message(base, g_pair[m])
                        ratio = None
                    else:
                        corr = self._basis_correction(hs, m, pi_pair[m])
                        correction = c_pair[m] * corr
                        message = compose_message(base, g_pair[m], corr, c_pair[m])
                        ratio = correction.norm(dim=-1, keepdim=True) / (base.norm(dim=-1, keepdim=True) + self.eps)
                    delta[m] = delta[m].index_add(0, cd, message)
                    if return_diagnostics and ratio is not None:
                        diagnostic_lists["correction_ratio"][m].append(ratio.detach().cpu())

            delta = [fixed_degree_mean(delta[0], degree), fixed_degree_mean(delta[1], degree)]

        h_tilde = [self.residual_norms[m](h0[m] + delta[m]) for m in range(2)]
        z = self.fusion(torch.cat([h0[0], h0[1], h_tilde[0], h_tilde[1]], dim=-1))
        aux_loss = z.new_zeros(())
        aux_info: dict[str, Any] = {
            "num_nonself_messages": num_edges,
            "degree": degree.detach(),
        }
        if return_diagnostics:
            aux_info["controls"] = {
                name: tuple(
                    torch.cat(diagnostic_lists[name][m], dim=0)
                    if diagnostic_lists[name][m]
                    else torch.empty((0, 4 if name == "pi" else 1))
                    for m in range(2)
                )
                for name in diagnostic_lists
            }
            aux_info["edge_index_nonself"] = edge_index.detach().cpu()
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device | str | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size  # One-step edge chunking is controlled by edge_chunk_size.
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        z, _, _, _, _ = self(x.to(device), edge_index.to(device))
        return z.detach().cpu()


def common_parameter_names(model: Model) -> tuple[str, ...]:
    """Named parameter prefixes shared by all adaptive variants A/B/C."""
    prefixes = ("proj_t.", "proj_v.", "rel_proj_t.", "rel_proj_v.", "phi_pair.", "phi_rel.",
                "modality_embeddings", "phi_mod.", "w0.", "fusion.", "residual_norms.")
    return tuple(name for name, _ in model.named_parameters() if name.startswith(prefixes))


def parameter_counts(model: Model) -> dict[str, int]:
    prefixes = {
        "semantic_backbone": ("proj_t.", "proj_v.", "fusion.", "residual_norms."),
        "relation_encoder": ("rel_proj_t.", "rel_proj_v.", "phi_pair.", "phi_rel.", "modality_embeddings", "phi_mod."),
        "default_transform": ("w0.",),
        "gate_params": ("gate_heads.",),
        "correction_control_params": ("correction_heads.", "static_c_logits", "basis_heads."),
        "correction_basis_params": ("basis_u", "basis_v"),
    }
    counts = {key: 0 for key in prefixes}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        for key, starts in prefixes.items():
            if name.startswith(starts):
                counts[key] += parameter.numel()
                break
    counts["gate_and_correction_heads"] = counts["gate_params"] + counts["correction_control_params"]
    counts["correction_basis"] = counts["correction_basis_params"]
    counts["common_params_abc"] = counts["semantic_backbone"] + counts["relation_encoder"] + counts["default_transform"]
    counts["variant_specific_params"] = counts["gate_and_correction_heads"] + counts["correction_basis"]
    counts["model_total"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return counts
