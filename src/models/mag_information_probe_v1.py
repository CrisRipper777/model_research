"""A small, fixed-capacity probe for MAG information-utility studies.

The model intentionally has no learned graph construction, attention, experts,
router, or auxiliary objective. Every treatment instantiates the same modules;
the treatment only masks the six fixed context slots in each modality branch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


TREATMENTS = ("I0", "I1", "I2", "I3", "I4")
SLOT_NAMES = (
    "physical_1_same",
    "physical_2_same",
    "physical_1_cross",
    "physical_2_cross",
    "semantic_same",
    "semantic_cross",
)
_TREATMENT_MASKS = {
    "I0": (0, 0, 0, 0, 0, 0),
    "I1": (1, 1, 0, 0, 0, 0),
    "I2": (1, 1, 1, 1, 0, 0),
    "I3": (1, 1, 1, 1, 1, 1),
    "I4": (1, 1, 1, 1, 1, 1),
}


def canonical_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return sorted, unique, non-loop undirected pairs in canonical u<v form."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edges = edge_index.detach().to(dtype=torch.long, device="cpu")
    if not edges.numel():
        return torch.empty((2, 0), dtype=torch.long)
    if int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes):
        raise ValueError("edge_index contains a node outside [0,num_nodes)")
    left = torch.minimum(edges[0], edges[1])
    right = torch.maximum(edges[0], edges[1])
    keep = left != right
    code = torch.unique(left[keep] * int(num_nodes) + right[keep], sorted=True)
    return torch.stack((code // int(num_nodes), code % int(num_nodes)), dim=0)


def build_row_mean_operator(
    unique_edge_index: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a loop-free symmetric COO operator with independently normalized rows.

    Rows of isolated nodes remain all zero.  The input is expected to contain
    sorted, unique, canonical undirected pairs.
    """
    edges = unique_edge_index.detach().to(dtype=torch.long, device="cpu")
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("unique_edge_index must have shape [2,E]")
    if edges.numel():
        if int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes):
            raise ValueError("unique_edge_index contains an invalid node ID")
        if bool((edges[0] >= edges[1]).any()):
            raise ValueError("unique_edge_index must contain canonical non-loop pairs")
        codes = edges[0] * int(num_nodes) + edges[1]
        if codes.numel() > 1 and not bool((codes[1:] > codes[:-1]).all()):
            raise ValueError("unique_edge_index must be sorted and duplicate-free")
        directed = torch.cat((edges, edges.flip(0)), dim=1)
        degree = torch.bincount(directed[0], minlength=int(num_nodes))
        values = degree[directed[0]].to(dtype=dtype).reciprocal()
        operator = torch.sparse_coo_tensor(
            directed, values, (int(num_nodes), int(num_nodes)), dtype=dtype
        ).coalesce()
    else:
        degree = torch.zeros(int(num_nodes), dtype=torch.long)
        operator = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=dtype),
            (int(num_nodes), int(num_nodes)),
            dtype=dtype,
        ).coalesce()
    return operator, degree


def build_physical_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make the symmetric, deduplicated physical row-mean propagation operator."""
    unique = canonical_undirected_edges(edge_index, num_nodes)
    return build_row_mean_operator(unique, num_nodes, dtype=dtype)


def semantic_edges_from_flags(
    unique_edge_index: torch.Tensor, source_flags: torch.Tensor, source_column: int
) -> torch.Tensor:
    """Select one modality's V7B candidate support without merging the views."""
    if source_column not in (0, 1):
        raise ValueError("source_column must be 0 (Text) or 1 (Visual)")
    if source_flags.dtype != torch.uint8 or source_flags.shape != (
        unique_edge_index.size(1),
        2,
    ):
        raise ValueError("source_flags must be uint8 [unique_edges,2]")
    return unique_edge_index[:, source_flags[:, source_column] > 0].contiguous()


def _spmm(operator: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if not operator.is_sparse:
        raise ValueError("graph operators must remain sparse COO tensors")
    return torch.sparse.mm(operator, state)


class _IntrinsicProjector(nn.Module):
    """Private residual projector shared in definition across Text and Visual."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.linear2(self.dropout(F.gelu(self.linear1(value))))
        return self.norm(self.skip(value) + update)


class _ContextProcessor(nn.Module):
    """Bias-free processing: an all-zero context cannot create a message."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(6 * hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, intrinsic: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        update = self.linear1(slots)
        update = self.linear2(self.dropout(F.gelu(update)))
        return self.norm(intrinsic + update), update


class Model(nn.Module):
    """Fixed-shape Text/Visual probe for information-utility comparisons.

    ``treatment`` masks slots but never changes module construction, parameters,
    tensor shapes, or initialization order. Each branch uses six aligned slots:
    two same-modality physical orders, two cross-modality physical orders, and
    one same/cross modality semantic pair.
    """

    def __init__(
        self,
        cfg: Mapping | object,
        data_info: dict[str, int],
        *,
        treatment: str,
    ):
        super().__init__()
        if treatment not in TREATMENTS:
            raise ValueError(f"treatment must be one of {TREATMENTS}")
        model_cfg = cfg.model if hasattr(cfg, "model") else cfg
        self.treatment = str(treatment)
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("Text, Visual, and hidden dimensions must be positive")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout must be in [0,1)")

        # Construction order is treatment-independent, enabling byte-identical
        # Dataset x Seed initialization across I0-I4.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
                "visual": _IntrinsicProjector(self.visual_dim, self.hidden_dim, self.dropout_p),
            }
        )
        self.context_processors = nn.ModuleDict(
            {
                "text": _ContextProcessor(self.hidden_dim, self.dropout_p),
                "visual": _ContextProcessor(self.hidden_dim, self.dropout_p),
            }
        )
        self.fusion1 = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.out_dim = self.hidden_dim

    @property
    def active_slots(self) -> tuple[str, ...]:
        return tuple(
            name for name, enabled in zip(SLOT_NAMES, _TREATMENT_MASKS[self.treatment]) if enabled
        )

    def _context_slots(
        self,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        physical_operator: torch.Tensor,
        text_operator: torch.Tensor,
        visual_operator: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute fixed physical/semantic contexts without dense adjacency products."""
        p1_text = _spmm(physical_operator, h_text)
        p2_text = _spmm(physical_operator, p1_text)
        p1_visual = _spmm(physical_operator, h_visual)
        p2_visual = _spmm(physical_operator, p1_visual)
        return {
            "text": torch.stack(
                (
                    p1_text,
                    p2_text,
                    p1_visual,
                    p2_visual,
                    _spmm(text_operator, h_text),
                    _spmm(text_operator, h_visual),
                ),
                dim=1,
            ),
            "visual": torch.stack(
                (
                    p1_visual,
                    p2_visual,
                    p1_text,
                    p2_text,
                    _spmm(visual_operator, h_visual),
                    _spmm(visual_operator, h_text),
                ),
                dim=1,
            ),
        }

    def forward(
        self,
        x_text: torch.Tensor,
        x_visual: torch.Tensor,
        physical_operator: torch.Tensor,
        text_operator: torch.Tensor,
        visual_operator: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        if x_text.ndim != 2 or x_visual.ndim != 2 or x_text.size(0) != x_visual.size(0):
            raise ValueError("Text and Visual inputs must be [N,D] with equal node counts")
        if x_text.size(1) != self.text_dim or x_visual.size(1) != self.visual_dim:
            raise ValueError("input modality dimensions do not match model construction")
        if any(not op.is_sparse for op in (physical_operator, text_operator, visual_operator)):
            raise ValueError("physical and semantic operators must be sparse COO tensors")

        h_text = self.projectors["text"](x_text)
        h_visual = self.projectors["visual"](x_visual)
        raw_slots = self._context_slots(
            h_text, h_visual, physical_operator, text_operator, visual_operator
        )
        treatment_mask = torch.tensor(
            _TREATMENT_MASKS[self.treatment], dtype=h_text.dtype, device=h_text.device
        ).view(1, len(SLOT_NAMES), 1)
        slots_text = raw_slots["text"] * treatment_mask
        slots_visual = raw_slots["visual"] * treatment_mask
        z_text, delta_text = self.context_processors["text"](
            h_text, slots_text.flatten(start_dim=1)
        )
        z_visual, delta_visual = self.context_processors["visual"](
            h_visual, slots_visual.flatten(start_dim=1)
        )

        fused_input = torch.cat((h_text, h_visual, z_text, z_visual), dim=-1)
        fused = F.gelu(self.fusion1(fused_input))
        fused = self.fusion2(self.fusion_dropout(fused))
        fused = self.fusion_norm(0.5 * (h_text + h_visual) + fused)
        if not return_details:
            return fused
        details = {
            "h_text": h_text,
            "h_visual": h_visual,
            "slots_text": slots_text,
            "slots_visual": slots_visual,
            "context_delta_text": delta_text,
            "context_delta_visual": delta_visual,
            "active_slots": list(self.active_slots),
        }
        return fused, details


def build_semantic_operators_from_flags(
    unique_edge_index: torch.Tensor,
    source_flags: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build independent Text/Visual row-normalized operators from V7B provenance."""
    text_edges = semantic_edges_from_flags(unique_edge_index, source_flags, 0)
    visual_edges = semantic_edges_from_flags(unique_edge_index, source_flags, 1)
    p_text, degree_text = build_row_mean_operator(text_edges, num_nodes, dtype=dtype)
    p_visual, degree_visual = build_row_mean_operator(visual_edges, num_nodes, dtype=dtype)
    return p_text, p_visual, degree_text, degree_visual
