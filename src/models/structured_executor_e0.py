from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adaptive_prop_m0 import (
    _ModalityInterpreter,
    _PairEncoder,
    _Projector,
    _RelationEncoder,
    fixed_degree_mean,
    graph_standardized_log_degree,
    incoming_degree,
    leave_one_out_context,
    remove_self_messages,
)


VARIANTS = {"static_mix", "target_mix", "edge_mix"}
FUNCTION_NAMES = ("null", "smooth", "relational", "cross_modal")


def _rms(value: torch.Tensor, eps: float) -> torch.Tensor:
    return (value.square().mean(dim=-1, keepdim=True) + eps).sqrt()


def pi_target_mean(pi: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if pi.size(0) != dst.numel():
        raise ValueError("pi rows and destination indices must have equal length")
    if dst.numel() == 0:
        return pi.clone()
    sums = pi.new_zeros((num_nodes, pi.size(-1))).index_add(0, dst, pi)
    degree = incoming_degree(dst, num_nodes).clamp_min(1).to(pi.dtype).unsqueeze(-1)
    return sums.div(degree)[dst]


def pi_global_mean(pi: torch.Tensor) -> torch.Tensor:
    if pi.size(0) == 0:
        return pi.clone()
    return pi.mean(dim=0, keepdim=True).expand_as(pi).clone()


def pi_modality_tied(pi_text: torch.Tensor, pi_visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pi_text.shape != pi_visual.shape:
        raise ValueError("Text/Visual pi tensors must have identical shapes")
    tied = (pi_text + pi_visual) * 0.5
    tied = tied / tied.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return tied, tied.clone()


def renormalize_without(pi: torch.Tensor, function_index: int, eps: float = 1e-12) -> torch.Tensor:
    if pi.ndim != 2 or pi.size(-1) != len(FUNCTION_NAMES):
        raise ValueError("pi must have shape [E,4] in null/smooth/relational/cross_modal order")
    if function_index < 0 or function_index >= len(FUNCTION_NAMES):
        raise ValueError("function_index out of range")
    kept = pi.clone()
    kept[:, function_index] = 0
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(eps)


class Model(nn.Module):
    """One-step MAG propagation with a structured, modality-resolved function bank."""

    def __init__(self, cfg: Any, data_info: dict[str, int]):
        super().__init__()
        model_cfg = cfg.model
        self.variant = str(model_cfg.get("variant", "edge_mix")).strip().lower()
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
        if self.hidden_dim != 128 or self.relation_dim != 32 or self.relation_state_dim != 64 or self.modality_embed_dim != 8:
            raise ValueError("E0 freezes hidden/relation/relation-state/modality dimensions at 128/32/64/8")

        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim < 1 or self.visual_dim < 1:
            raise ValueError("E0 requires non-empty Text and Visual feature slices")
        if self.text_dim + self.visual_dim != int(data_info["input_dim"]):
            raise ValueError("Text/Visual dimensions do not cover data.x")

        dropout = float(model_cfg.get("dropout", 0.2))

        # Keep the relation-state modules and their operation order identical to M0.
        self.proj_t = _Projector(self.text_dim, self.hidden_dim, dropout)
        self.proj_v = _Projector(self.visual_dim, self.hidden_dim, dropout)
        self.rel_proj_t = nn.Sequential(
            nn.Linear(self.hidden_dim, self.relation_dim, bias=False),
            nn.LayerNorm(self.relation_dim),
        )
        self.rel_proj_v = nn.Sequential(
            nn.Linear(self.hidden_dim, self.relation_dim, bias=False),
            nn.LayerNorm(self.relation_dim),
        )
        self.phi_pair = _PairEncoder(4 * self.relation_dim + 4)
        self.phi_rel = _RelationEncoder()
        self.modality_embeddings = nn.Parameter(torch.empty(2, self.modality_embed_dim))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        self.phi_mod = _ModalityInterpreter(64 + 32 + self.modality_embed_dim)

        # The four paths are computationally distinct. Smooth is the calibrated
        # magnitude reference; R and X retain target/source-dependent directions.
        self.smooth_transforms = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2)]
        )
        self.relational_source = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2)]
        )
        self.relational_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(4 * self.hidden_dim, self.hidden_dim), nn.GELU(),
                            nn.Linear(self.hidden_dim, self.hidden_dim)) for _ in range(2)]
        )
        self.cross_modal_source = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2)]
        )
        self.cross_modal_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(4 * self.hidden_dim, self.hidden_dim), nn.GELU(),
                            nn.Linear(self.hidden_dim, self.hidden_dim)) for _ in range(2)]
        )
        for layers in (self.smooth_transforms, self.relational_source, self.cross_modal_source):
            for layer in layers:
                nn.init.xavier_uniform_(layer.weight)

        self.router = nn.Sequential(
            nn.Linear(self.relation_state_dim, self.relation_state_dim),
            nn.GELU(),
            nn.Linear(self.relation_state_dim, len(FUNCTION_NAMES)),
        )
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.zeros_(self.router[0].bias)
        nn.init.normal_(self.router[2].weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            self.router[2].bias.copy_(torch.tensor([0.0, 2.0, 0.0, 0.0]))

        self.residual_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(2)])
        self.fusion = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.out_dim = self.hidden_dim

    def split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError(f"expected [Text({self.text_dim}), Visual({self.visual_dim})], got {tuple(x.shape)}")
        return x[:, : self.text_dim], x[:, self.text_dim :]

    def _pair_evidence(self, p: torch.Tensor, context: torch.Tensor, deg_z: torch.Tensor,
                       src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        target, source = p[dst], p[src]
        similarity = F.cosine_similarity(target, source, dim=-1, eps=self.eps).unsqueeze(-1)
        return torch.cat([
            target, source, (target - source).abs(), target * source, similarity,
            context, deg_z[dst].unsqueeze(-1), deg_z[src].unsqueeze(-1),
        ], dim=-1)

    def _functional_states(self, p, contexts, deg_z, src, dst):
        # This is intentionally operation-for-operation the M0 definition.
        evidence_t = self._pair_evidence(p[0], contexts[0], deg_z, src, dst)
        evidence_v = self._pair_evidence(p[1], contexts[1], deg_z, src, dst)
        q_t, q_v = self.phi_pair(evidence_t), self.phi_pair(evidence_v)
        relation = self.phi_rel(torch.cat([q_t, q_v, (q_t - q_v).abs(), q_t * q_v], dim=-1))
        states = []
        for modality, q in enumerate((q_t, q_v)):
            embedding = self.modality_embeddings[modality].expand(q.size(0), -1)
            states.append(self.phi_mod(torch.cat([relation, q, embedding], dim=-1)))
        return q_t, q_v, relation, states

    def relation_state(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Expose M0-compatible P/q/r/u tensors for regression tests and audits."""
        x_t, x_v = self.split_modalities(x)
        h0 = [self.proj_t(x_t), self.proj_v(x_v)]
        _, src, dst = remove_self_messages(edge_index)
        degree = incoming_degree(dst, int(x.size(0)))
        deg_z = graph_standardized_log_degree(degree, self.eps).to(dtype=h0[0].dtype)
        p = [self.rel_proj_t(h0[0]), self.rel_proj_v(h0[1])]
        contexts = [leave_one_out_context(p[m], src, dst, degree, self.eps) for m in range(2)]
        q_t, q_v, relation, states = self._functional_states(p, contexts, deg_z, src, dst)
        return {"h0": h0, "p": p, "q": [q_t, q_v], "r": relation, "u": states,
                "contexts": contexts, "src": src, "dst": dst, "degree": degree, "deg_z": deg_z}

    def _prepare(self, x: torch.Tensor, edge_index: torch.Tensor):
        x_t, x_v = self.split_modalities(x)
        h0 = [self.proj_t(x_t), self.proj_v(x_v)]
        nonself, src, dst = remove_self_messages(edge_index)
        degree = incoming_degree(dst, int(x.size(0)))
        deg_z = graph_standardized_log_degree(degree, self.eps).to(dtype=h0[0].dtype)
        p = [self.rel_proj_t(h0[0]), self.rel_proj_v(h0[1])]
        contexts = [leave_one_out_context(p[m], src, dst, degree, self.eps) for m in range(2)]
        return h0, nonself, src, dst, degree, deg_z, p, contexts

    def _route_pool(self, p, contexts, deg_z, src, dst, degree, num_nodes):
        sums = [None, None]
        for begin in range(0, int(src.numel()), self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, int(src.numel()))
            _, _, _, states = self._functional_states(
                p, [contexts[0][begin:end], contexts[1][begin:end]], deg_z,
                src[begin:end], dst[begin:end],
            )
            for modality in range(2):
                if self.variant == "static_mix":
                    part = states[modality].sum(dim=0, keepdim=True)
                    sums[modality] = part if sums[modality] is None else sums[modality] + part
                else:
                    part = states[modality].new_zeros((num_nodes, self.relation_state_dim))
                    part = part.index_add(0, dst[begin:end], states[modality])
                    sums[modality] = part if sums[modality] is None else sums[modality] + part
        if self.variant == "static_mix":
            count = max(int(src.numel()), 1)
            return [value / count if value is not None else p[0].new_zeros((1, self.relation_state_dim))
                    for value in sums]
        denominator = degree.clamp_min(1).to(dtype=p[0].dtype).unsqueeze(-1)
        return [(value / denominator) if value is not None else p[0].new_zeros((num_nodes, self.relation_state_dim))
                for value in sums]

    def _expert_outputs(self, h0, src: torch.Tensor, dst: torch.Tensor):
        smooth, relational, cross_modal = [], [], []
        raw_relational, raw_cross_modal = [], []
        ref_rms, raw_rms_rel, raw_rms_cross, scale_rel, scale_cross = [], [], [], [], []
        for modality in range(2):
            target = h0[modality][dst]
            source_same = h0[modality][src]
            smooth_m = self.smooth_transforms[modality](source_same)

            projected_same = self.relational_source[modality](source_same)
            input_rel = torch.cat([target, projected_same, (target - projected_same).abs(),
                                   target * projected_same], dim=-1)
            raw_rel = self.relational_mlps[modality](input_rel)

            source_other = h0[1 - modality][src]
            projected_other = self.cross_modal_source[modality](source_other)
            input_cross = torch.cat([target, projected_other, (target - projected_other).abs(),
                                     target * projected_other], dim=-1)
            raw_cross = self.cross_modal_mlps[modality](input_cross)

            reference = _rms(smooth_m, self.eps).detach()
            rel_rms = _rms(raw_rel, self.eps)
            cross_rms = _rms(raw_cross, self.eps)
            scale_r = reference / rel_rms
            scale_x = reference / cross_rms
            calibrated_rel = raw_rel / rel_rms * reference
            calibrated_cross = raw_cross / cross_rms * reference

            smooth.append(smooth_m)
            relational.append(calibrated_rel)
            cross_modal.append(calibrated_cross)
            raw_relational.append(raw_rel)
            raw_cross_modal.append(raw_cross)
            ref_rms.append(reference)
            raw_rms_rel.append(rel_rms)
            raw_rms_cross.append(cross_rms)
            scale_rel.append(scale_r)
            scale_cross.append(scale_x)
        return (smooth, relational, cross_modal,
                {"raw_relational": raw_relational, "raw_cross_modal": raw_cross_modal,
                 "reference_rms": ref_rms, "raw_rms_relational": raw_rms_rel,
                 "raw_rms_cross_modal": raw_rms_cross, "scale_relational": scale_rel,
                 "scale_cross_modal": scale_cross})

    @staticmethod
    def _overridden(value: torch.Tensor, overrides, name: str, modality: int,
                    begin: int, end: int) -> torch.Tensor:
        if overrides is None or name not in overrides:
            return value
        return overrides[name][modality][begin:end].to(device=value.device, dtype=value.dtype)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                control_overrides: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
                return_diagnostics: bool = False):
        if edge_index is None:
            raise ValueError("structured_executor_e0 requires physical edge_index")
        h0, nonself, src, dst, degree, deg_z, p, contexts = self._prepare(x, edge_index)
        num_nodes, num_edges = int(x.size(0)), int(src.numel())
        pooled = None if self.variant == "edge_mix" else self._route_pool(
            p, contexts, deg_z, src, dst, degree, num_nodes)
        delta = [h0[0].new_zeros((num_nodes, self.hidden_dim)),
                 h0[1].new_zeros((num_nodes, self.hidden_dim))]
        diagnostic: dict[str, list[list[torch.Tensor]]] = {
            key: [[], []] for key in (
                "pi", "cos_smooth_relational", "cos_smooth_cross_modal", "cos_relational_cross_modal",
                "smooth_rms", "raw_relational_rms", "raw_cross_modal_rms",
                "calibrated_relational_rms_ratio", "calibrated_cross_modal_rms_ratio",
                "relational_calibration_scale", "cross_modal_calibration_scale",
            )
        }
        for begin in range(0, num_edges, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, num_edges)
            cs, cd = src[begin:end], dst[begin:end]
            _, _, _, states = self._functional_states(
                p, [contexts[0][begin:end], contexts[1][begin:end]], deg_z, cs, cd)
            route_inputs = []
            for modality in range(2):
                if self.variant == "static_mix":
                    route_inputs.append(pooled[modality].expand(end - begin, -1))
                elif self.variant == "target_mix":
                    route_inputs.append(pooled[modality][cd])
                else:
                    route_inputs.append(states[modality])
            route_pi = [torch.softmax(self.router(value), dim=-1) for value in route_inputs]
            pi = [self._overridden(route_pi[m], control_overrides, "pi", m, begin, end)
                  for m in range(2)]

            smooth, relational, cross_modal, scales = self._expert_outputs(h0, cs, cd)
            for modality in range(2):
                message = (pi[modality][:, 1:2] * smooth[modality]
                           + pi[modality][:, 2:3] * relational[modality]
                           + pi[modality][:, 3:4] * cross_modal[modality])
                delta[modality] = delta[modality].index_add(0, cd, message)
                if return_diagnostics:
                    sr = _rms(smooth[modality], self.eps)
                    rr = _rms(relational[modality], self.eps)
                    xr = _rms(cross_modal[modality], self.eps)
                    cos_sr = F.cosine_similarity(smooth[modality], relational[modality], dim=-1, eps=self.eps)
                    cos_sx = F.cosine_similarity(smooth[modality], cross_modal[modality], dim=-1, eps=self.eps)
                    cos_rx = F.cosine_similarity(relational[modality], cross_modal[modality], dim=-1, eps=self.eps)
                    values = {
                        "pi": pi[modality],
                        "cos_smooth_relational": cos_sr[:, None],
                        "cos_smooth_cross_modal": cos_sx[:, None],
                        "cos_relational_cross_modal": cos_rx[:, None],
                        "smooth_rms": scales["reference_rms"][modality],
                        "raw_relational_rms": scales["raw_rms_relational"][modality],
                        "raw_cross_modal_rms": scales["raw_rms_cross_modal"][modality],
                        "calibrated_relational_rms_ratio": rr / sr,
                        "calibrated_cross_modal_rms_ratio": xr / sr,
                        "relational_calibration_scale": scales["scale_relational"][modality],
                        "cross_modal_calibration_scale": scales["scale_cross_modal"][modality],
                    }
                    for name, value in values.items():
                        diagnostic[name][modality].append(value.detach().float().cpu())

        delta = [fixed_degree_mean(value, degree) for value in delta]
        h_tilde = [self.residual_norms[m](h0[m] + delta[m]) for m in range(2)]
        z = self.fusion(torch.cat([h0[0], h0[1], h_tilde[0], h_tilde[1]], dim=-1))
        aux_info: dict[str, Any] = {"num_nonself_messages": num_edges, "degree": degree.detach()}
        if return_diagnostics:
            controls = {}
            for key in diagnostic:
                controls[key] = tuple(
                    torch.cat(diagnostic[key][m], dim=0) if diagnostic[key][m]
                    else torch.empty((0, 4 if key == "pi" else 1))
                    for m in range(2)
                )
            aux_info["controls"] = controls
            aux_info["edge_index_nonself"] = nonself.detach().cpu()
            aux_info["function_names"] = FUNCTION_NAMES
        return z, None, None, z.new_zeros(()), aux_info

    @torch.no_grad()
    def inference(self, x: torch.Tensor, edge_index: torch.Tensor,
                  device: torch.device | str | None = None, batch_size: int = 65536) -> torch.Tensor:
        del batch_size
        self.eval()
        device = torch.device(device) if device is not None else next(self.parameters()).device
        z, _, _, _, _ = self(x.to(device), edge_index.to(device))
        return z.detach().cpu()
