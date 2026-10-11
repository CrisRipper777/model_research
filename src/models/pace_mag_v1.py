"""PACE-MAG V1: provenance-aware collaborative experts for full-graph NC.

This implementation deliberately keeps relation topology fixed.  The only
learned relation conditioning is in node messages and the shared expert router.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.data.semantic_candidates import _array_fingerprint
from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b


VARIANTS = ("e0_source_agnostic", "e1_no_cross_modal", "e2_full", "e3_single_expert_control", "e4_static_routing")
_RELATIONS = ("physical", "text", "visual")
_MODALITIES = ("text", "visual")


def _empty_sparse(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0,), dtype=dtype, device=device),
        (n, n),
        device=device,
    ).coalesce()


def canonical_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Canonicalize a graph to sorted, unique, non-loop undirected pairs."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edges = edge_index.to(dtype=torch.long)
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edges.device)
    if int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes):
        raise ValueError("edge_index contains node IDs outside [0,num_nodes)")
    left = torch.minimum(edges[0], edges[1])
    right = torch.maximum(edges[0], edges[1])
    keep = left != right
    codes = left[keep] * int(num_nodes) + right[keep]
    codes = torch.unique(codes, sorted=True)
    return torch.stack((codes // int(num_nodes), codes % int(num_nodes)), dim=0)


def _symmetric_sparse_from_unique(
    unique_edges: torch.Tensor,
    undirected_weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if unique_edges.ndim != 2 or unique_edges.size(0) != 2:
        raise ValueError("unique_edges must have shape [2,E]")
    if unique_edges.size(1) != undirected_weights.numel():
        raise ValueError("edge and weight lengths differ")
    indices = torch.cat((unique_edges, unique_edges.flip(0)), dim=1)
    values = torch.cat((undirected_weights, undirected_weights), dim=0)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=values.device).coalesce()


def _row_quality(operator: torch.Tensor, num_nodes: int) -> torch.Tensor:
    result = torch.zeros(num_nodes, dtype=operator.dtype, device=operator.device)
    if operator._nnz():
        result.index_add_(0, operator.indices()[0], operator.values())
    return result


def build_physical_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the no-self-loop symmetric D^-1/2 A D^-1/2 physical operator."""
    unique = canonical_undirected_edges(edge_index, num_nodes)
    if unique.size(1) == 0:
        operator = _empty_sparse(num_nodes, device=edge_index.device, dtype=dtype)
        return operator, _row_quality(operator, num_nodes)
    degrees = torch.bincount(unique.reshape(-1), minlength=num_nodes).to(dtype=dtype)
    weights = degrees[unique[0]].clamp_min(1).rsqrt() * degrees[unique[1]].clamp_min(1).rsqrt()
    operator = _symmetric_sparse_from_unique(unique, weights, num_nodes)
    return operator, _row_quality(operator, num_nodes)


def validate_v7b_cache_graph(
    edge_index: torch.Tensor,
    unique_edge_index: torch.Tensor,
    source_flags: torch.Tensor,
    num_nodes: int,
) -> None:
    """Check the exact V7B unique-edge/flag alignment and symmetric layout."""
    if unique_edge_index.ndim != 2 or unique_edge_index.size(0) != 2:
        raise ValueError("V7B unique_edge_index must have shape [2,E]")
    unique = unique_edge_index.to(dtype=torch.long)
    if unique.numel():
        if int(unique.min()) < 0 or int(unique.max()) >= int(num_nodes):
            raise ValueError("V7B cache edge node ID is out of range")
        if bool((unique[0] >= unique[1]).any()):
            raise ValueError("V7B unique edges must be canonical u<v non-loop pairs")
        codes = unique[0] * int(num_nodes) + unique[1]
        if codes.numel() > 1 and not bool((codes[1:] > codes[:-1]).all()):
            raise ValueError("V7B unique edges must be sorted and duplicate-free")
    if source_flags.dtype != torch.uint8 or tuple(source_flags.shape) != (unique.size(1), 2):
        raise ValueError("V7B source_flags must be uint8 [unique_edges,2]")
    if source_flags.numel() and bool((source_flags > 2).any()):
        raise ValueError("V7B source flags must be raw proposal counts in {0,1,2}")
    if source_flags.numel() and bool((source_flags.sum(dim=1) == 0).any()):
        raise ValueError("V7B candidate edge has no recorded semantic source")
    expected_directed = torch.cat((unique, unique.flip(0)), dim=1)
    if edge_index.dtype != torch.long or not torch.equal(edge_index, expected_directed):
        raise ValueError("V7B directed edges do not match the unique-edge/source-flag order")


def build_semantic_operators(
    unique_edge_index: torch.Tensor,
    source_flags: torch.Tensor,
    num_nodes: int,
    *,
    fixed_edge_weight: float = 0.8,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Create PU and its source allocation PT/PV without renormalizing PT/PV.

    The fixed edge weight matches V7B.1's fixed-KNN reference: 0.8 times the
    symmetric normalization by the candidate graph's undirected degree.
    """
    if not 0.0 < float(fixed_edge_weight) < 1.0:
        raise ValueError("fixed_edge_weight must be in (0,1)")
    unique = unique_edge_index.to(dtype=torch.long)
    flags = source_flags.to(device=unique.device)
    if unique.ndim != 2 or unique.size(0) != 2:
        raise ValueError("unique_edge_index must have shape [2,E]")
    if flags.dtype != torch.uint8 or tuple(flags.shape) != (unique.size(1), 2):
        raise ValueError("source_flags must have dtype uint8 and shape [E,2]")
    if flags.numel() and (bool((flags > 2).any()) or bool((flags.sum(dim=1) == 0).any())):
        raise ValueError("invalid V7B source flags: counts must be 0,1,2 and each edge needs a source")
    if unique.numel() and (int(unique.min()) < 0 or int(unique.max()) >= int(num_nodes)):
        raise ValueError("semantic edge node ID is out of range")
    if unique.numel() and bool((unique[0] >= unique[1]).any()):
        raise ValueError("semantic unique edges must be canonical u<v pairs")
    if unique.size(1):
        codes = unique[0] * int(num_nodes) + unique[1]
        if codes.numel() > 1 and not bool((codes[1:] > codes[:-1]).all()):
            raise ValueError("semantic unique edges must be sorted and duplicate-free")

    device = unique.device
    degree = torch.bincount(unique.reshape(-1), minlength=num_nodes).to(device=device, dtype=dtype)
    base = float(fixed_edge_weight) * (
        degree[unique[0]].clamp_min(1) * degree[unique[1]].clamp_min(1)
    ).rsqrt()
    text_present = flags[:, 0] > 0
    visual_present = flags[:, 1] > 0
    both = text_present & visual_present
    text_share = torch.where(text_present, torch.where(both, 0.5, 1.0), 0.0).to(dtype=dtype)
    visual_share = torch.where(visual_present, torch.where(both, 0.5, 1.0), 0.0).to(dtype=dtype)
    pu_weights = base
    pt_weights = base * text_share
    pv_weights = base * visual_share
    if not torch.allclose(pt_weights + pv_weights, pu_weights, atol=1e-7, rtol=1e-6):
        raise AssertionError("PT + PV does not reconstruct PU before sparse assembly")
    pu = _symmetric_sparse_from_unique(unique, pu_weights, num_nodes)
    pt = _symmetric_sparse_from_unique(unique, pt_weights, num_nodes)
    pv = _symmetric_sparse_from_unique(unique, pv_weights, num_nodes)
    q_union = _row_quality(pu, num_nodes)
    q_text = _row_quality(pt, num_nodes)
    q_visual = _row_quality(pv, num_nodes)
    if not torch.allclose(q_text + q_visual, q_union, atol=1e-7, rtol=1e-6):
        raise AssertionError("Text and Visual row qualities do not reconstruct PU row quality")
    return {
        "pu": pu,
        "pt": pt,
        "pv": pv,
        "q_union": q_union,
        "q_text": q_text,
        "q_visual": q_visual,
        "text_share": text_share,
        "visual_share": visual_share,
        "base_undirected_weight": base,
    }


def row_quality_aggregate(operator: torch.Tensor, message: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply P@message / row-quality and force empty rows to exactly zero."""
    numerator = torch.sparse.mm(operator, message)
    valid = q > 0
    result = numerator / q.clamp_min(float(eps)).unsqueeze(-1)
    return result * valid.to(result.dtype).unsqueeze(-1)


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.shortcut = nn.Linear(input_dim, hidden_dim)
        self.first = nn.Linear(input_dim, hidden_dim)
        self.second = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        value = self.dropout(F.gelu(self.first(x)))
        value = self.dropout(self.second(value))
        return self.norm(residual + value)


class CrossModalBottleneck(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(hidden_dim, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.second = nn.Linear(bottleneck, hidden_dim)
        nn.init.normal_(self.second.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.second.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.dropout(self.activation(self.first(x))))


class GraphExpert(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(4 * hidden_dim, bottleneck)
        self.second = nn.Linear(bottleneck, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.dropout(F.gelu(self.first(value))))


def _expert_parameter_count(hidden_dim: int, bottleneck: int) -> int:
    return (4 * hidden_dim + 1) * bottleneck + (bottleneck + 1) * hidden_dim


def _matched_single_bottleneck(
    hidden_dim: int,
    expert_bottleneck: int,
    num_experts: int,
    router_hidden_dim: int,
    relation_embedding_dim: int,
    modality_embedding_dim: int,
) -> tuple[int, int, int]:
    router_in = 4 * hidden_dim + relation_embedding_dim + modality_embedding_dim + 1
    router_params = (router_in + 1) * router_hidden_dim + (router_hidden_dim + 1) * num_experts
    embedding_params = 3 * relation_embedding_dim + 2 * modality_embedding_dim
    target = num_experts * _expert_parameter_count(hidden_dim, expert_bottleneck) + router_params + embedding_params
    slope = 5 * hidden_dim + 1
    intercept = hidden_dim
    width = max(1, int(round((target - intercept) / slope)))
    single = _expert_parameter_count(hidden_dim, width)
    return width, target, single


class Model(nn.Module):
    """PACE-MAG V1. Supports validated, global-ID full-graph NC only."""

    requires_global_semantic_candidates = True
    requires_full_graph_training = True
    supports_link_prediction = False
    full_graph_training = True
    record_run_telemetry = True
    collect_v7_diagnostics = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.variant = str(model_cfg.get("variant", "e2_full"))
        if self.variant not in VARIANTS:
            raise ValueError(f"PACE-MAG variant must be one of {VARIANTS}, got {self.variant!r}")
        self.num_nodes = int(data_info["num_nodes"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim < 1 or self.visual_dim < 1:
            raise ValueError("PACE-MAG requires positive Text and Visual feature dimensions")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.dropout = float(model_cfg.get("dropout", 0.2))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self._config_semantic_edge_weight = float(model_cfg.get("semantic_edge_weight", 0.8))
        self.num_experts = int(model_cfg.get("num_experts", 3))
        if self.num_experts != 3:
            raise ValueError("PACE-MAG V1 freezes num_experts=3")
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.expert_chunk_size = int(model_cfg.get("expert_chunk_size", 32768))
        self.activation_checkpointing = bool(model_cfg.get("activation_checkpointing", True))
        self.router_hidden_dim = int(model_cfg.get("router_hidden_dim", 64))
        self.relation_embedding_dim = int(model_cfg.get("relation_embedding_dim", 8))
        self.modality_embedding_dim = int(model_cfg.get("modality_embedding_dim", 8))
        self.rho_max = float(model_cfg.get("rho_max", 0.25))
        self.beta_max = float(model_cfg.get("beta_max", 1.0))
        if self.hidden_dim < 1 or self.expert_bottleneck < 1 or self.expert_chunk_size < 1 or self.rho_max <= 0 or self.beta_max <= 0:
            raise ValueError("hidden/bottleneck dimensions and bounded scalar maxima must be positive")

        self.projectors = nn.ModuleDict(
            {
                "text": ModalityProjector(self.text_dim, self.hidden_dim, self.dropout),
                "visual": ModalityProjector(self.visual_dim, self.hidden_dim, self.dropout),
            }
        )
        self.cross_modal = nn.ModuleDict(
            {
                "visual_to_text": CrossModalBottleneck(
                    self.hidden_dim, int(model_cfg.get("cross_modal_bottleneck", 64)), self.dropout
                ),
                "text_to_visual": CrossModalBottleneck(
                    self.hidden_dim, int(model_cfg.get("cross_modal_bottleneck", 64)), self.dropout
                ),
            }
        )
        rho_init = float(model_cfg.get("rho_init", 0.1))
        if not 0.0 < rho_init < self.rho_max:
            raise ValueError("rho_init must be strictly between 0 and rho_max")
        rho_probability = rho_init / self.rho_max
        self.rho_raw = nn.Parameter(torch.full((2, 2), math.log(rho_probability / (1 - rho_probability))))

        self.experts = nn.ModuleList(
            GraphExpert(self.hidden_dim, self.expert_bottleneck, self.dropout)
            for _ in range(self.num_experts)
        ) if self.variant != "e3_single_expert_control" else nn.ModuleList()
        self.single_expert_bottleneck, self.matched_dynamic_parameter_target, self.single_expert_actual_parameters = (
            _matched_single_bottleneck(
                self.hidden_dim,
                self.expert_bottleneck,
                self.num_experts,
                self.router_hidden_dim,
                self.relation_embedding_dim,
                self.modality_embedding_dim,
            )
        )
        self.single_expert = GraphExpert(
            self.hidden_dim, self.single_expert_bottleneck, self.dropout
        ) if self.variant == "e3_single_expert_control" else None

        if self.variant != "e3_single_expert_control":
            router_in_dim = 4 * self.hidden_dim + self.relation_embedding_dim + self.modality_embedding_dim + 1
            self.router = nn.Sequential(
                nn.Linear(router_in_dim, self.router_hidden_dim),
                nn.GELU(),
                nn.Linear(self.router_hidden_dim, self.num_experts),
            )
            self.relation_embedding = nn.Embedding(3, self.relation_embedding_dim)
            self.modality_embedding = nn.Embedding(2, self.modality_embedding_dim)
        else:
            self.router = None
            self.relation_embedding = None
            self.modality_embedding = None

        beta_phy_init = float(model_cfg.get("beta_phy_init", 0.5))
        beta_sem_init = float(model_cfg.get("beta_sem_init", 0.3))
        self.beta_phy_raw = nn.Parameter(torch.tensor(self._inverse_bounded(beta_phy_init, self.beta_max)))
        self.beta_sem_raw = nn.Parameter(torch.tensor(self._inverse_bounded(beta_sem_init, self.beta_max)))
        self.gamma = float(model_cfg.get("gamma", 0.2))
        self.message_norm = nn.LayerNorm(self.hidden_dim)
        self.aggregate_norm = nn.LayerNorm(self.hidden_dim)
        self.mix_norm = nn.ModuleDict({name: nn.LayerNorm(self.hidden_dim) for name in _MODALITIES})
        self.physical_norm = nn.ModuleDict({name: nn.LayerNorm(self.hidden_dim) for name in _MODALITIES})
        self.fusion = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.out_dim = self.hidden_dim

        self.semantic_cache_path: str | None = None
        self.semantic_dataset: str | None = None
        self.semantic_candidate_fingerprint: str | None = None
        self.semantic_feature_fingerprint: str | None = None
        self.semantic_v7b_fingerprint: str | None = None
        self.semantic_metadata: dict[str, Any] | None = None
        self._validated_feature_hashes: tuple[str, str] | None = None
        self.register_buffer("semantic_pu", None, persistent=False)
        self.register_buffer("semantic_pt", None, persistent=False)
        self.register_buffer("semantic_pv", None, persistent=False)
        self.register_buffer("semantic_q_union", None, persistent=False)
        self.register_buffer("semantic_q_text", None, persistent=False)
        self.register_buffer("semantic_q_visual", None, persistent=False)
        self.register_buffer("_physical_operator", None, persistent=False)
        self.register_buffer("_physical_quality", None, persistent=False)
        self.register_buffer("_physical_edge_reference", None, persistent=False)
        self._gradient_diagnostics: dict[str, Any] | None = None

    @staticmethod
    def _inverse_bounded(value: float, maximum: float) -> float:
        if not 0.0 < value < maximum:
            raise ValueError("bounded scalar initial values must lie strictly inside (0, maximum)")
        probability = value / maximum
        return math.log(probability / (1.0 - probability))

    @property
    def rho(self) -> torch.Tensor:
        return self.rho_max * torch.sigmoid(self.rho_raw)

    @property
    def beta_phy(self) -> torch.Tensor:
        return self.beta_max * torch.sigmoid(self.beta_phy_raw)

    @property
    def beta_sem(self) -> torch.Tensor:
        return self.beta_max * torch.sigmoid(self.beta_sem_raw)

    @staticmethod
    def compose_cross_modal_message(
        target_modality: torch.Tensor,
        projected_other_modality: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        return target_modality + rho * projected_other_modality

    def load_semantic_cache(
        self,
        path: str,
        *,
        expected_dataset: str | None = None,
        expected_feature_fingerprint: str | None = None,
        expected_candidate_fingerprint: str | None = None,
        expected_v7b_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        cache = load_semantic_candidate_cache_v7b(
            path,
            expected_dataset=expected_dataset,
            expected_feature_fingerprint=expected_feature_fingerprint,
            expected_candidate_fingerprint=expected_candidate_fingerprint,
            expected_v7b_fingerprint=expected_v7b_fingerprint,
        )
        metadata = cache["metadata"]
        if int(metadata.get("num_nodes", -1)) != self.num_nodes:
            raise ValueError(
                f"semantic cache num_nodes={metadata.get('num_nodes')} does not match source graph num_nodes={self.num_nodes}"
            )
        validate_v7b_cache_graph(
            cache["edge_index"], cache["unique_edge_index"], cache["source_flags"], self.num_nodes
        )
        views = build_semantic_operators(
            cache["unique_edge_index"], cache["source_flags"], self.num_nodes,
            fixed_edge_weight=float(self._model_config_value("semantic_edge_weight", 0.8)),
        )
        if self.variant == "e0_source_agnostic":
            indices = views["pu"].indices()
            half_values = views["pu"].values() * 0.5
            views["pt"] = torch.sparse_coo_tensor(indices, half_values, views["pu"].shape).coalesce()
            views["pv"] = torch.sparse_coo_tensor(indices, half_values, views["pu"].shape).coalesce()
            views["q_text"] = views["q_union"] * 0.5
            views["q_visual"] = views["q_union"] * 0.5
        self.semantic_pu = views["pu"]
        self.semantic_pt = views["pt"]
        self.semantic_pv = views["pv"]
        self.semantic_q_union = views["q_union"]
        self.semantic_q_text = views["q_text"]
        self.semantic_q_visual = views["q_visual"]
        self.semantic_cache_path = str(Path(path).resolve())
        self.semantic_dataset = str(metadata.get("dataset", ""))
        self.semantic_candidate_fingerprint = str(metadata.get("candidate_fingerprint", ""))
        self.semantic_feature_fingerprint = str(metadata.get("feature_fingerprint", ""))
        self.semantic_v7b_fingerprint = str(metadata.get("v7b_fingerprint", ""))
        self.semantic_metadata = metadata
        return metadata

    def _model_config_value(self, name: str, default: Any) -> Any:
        # The validated config value is recorded during initialization below.
        return getattr(self, f"_config_{name}", default)

    def _validate_input_features_once(self, x: torch.Tensor) -> None:
        if self.semantic_metadata is None or self._validated_feature_hashes is not None:
            return
        identity = self.semantic_metadata.get("feature_identity", {})
        expected_text = identity.get("text", {}).get("array_sha256")
        expected_visual = identity.get("visual", {}).get("array_sha256")
        if expected_text is None or expected_visual is None:
            raise ValueError("semantic cache is missing modality feature array fingerprints")
        text = np.ascontiguousarray(x[:, : self.text_dim].detach().cpu().numpy(), dtype=np.float32)
        visual = np.ascontiguousarray(x[:, self.text_dim : self.text_dim + self.visual_dim].detach().cpu().numpy(), dtype=np.float32)
        observed = (_array_fingerprint(text), _array_fingerprint(visual))
        if observed != (expected_text, expected_visual):
            raise ValueError("model input Text/Visual features do not match the semantic cache fingerprint")
        self._validated_feature_hashes = observed

    def _get_physical_operator(self, edge_index: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("PACE-MAG full-graph NC requires physical edge_index")
        if self._physical_operator is None:
            operator, q = build_physical_operator(edge_index, self.num_nodes, dtype=dtype)
            self._physical_operator = operator
            self._physical_quality = q
            self._physical_edge_reference = edge_index.detach().clone()
        elif self._physical_edge_reference.device != edge_index.device or not torch.equal(
            self._physical_edge_reference, edge_index
        ):
            operator, q = build_physical_operator(edge_index, self.num_nodes, dtype=dtype)
            self._physical_operator = operator
            self._physical_quality = q
            self._physical_edge_reference = edge_index.detach().clone()
        return self._physical_operator, self._physical_quality

    def _router_weights(
        self,
        h_norm: torch.Tensor,
        u_norm: torch.Tensor,
        q: torch.Tensor,
        valid: torch.Tensor,
        relation_id: int,
        modality_id: int,
    ) -> torch.Tensor:
        assert self.router is not None
        assert self.relation_embedding is not None and self.modality_embedding is not None
        n = h_norm.size(0)
        if bool(valid.any()):
            global_context = u_norm[valid].mean(dim=0, keepdim=True)
            h_global = h_norm[valid].mean(dim=0, keepdim=True)
            u_global = global_context
            q_global = torch.log1p(q[valid].mean()).reshape(1, 1)
        else:
            global_context = h_norm.new_zeros((1, self.hidden_dim))
            h_global = global_context
            u_global = global_context
            q_global = h_norm.new_zeros((1, 1))
        relation = self.relation_embedding.weight[relation_id].reshape(1, -1)
        modality = self.modality_embedding.weight[modality_id].reshape(1, -1)
        if self.variant == "e4_static_routing":
            features = torch.cat(
                (h_global, u_global, h_global - u_global, relation, modality, global_context, q_global), dim=-1
            )
            probabilities = torch.softmax(self.router(features), dim=-1).expand(n, -1)
        else:
            features = torch.cat(
                (
                    h_norm,
                    u_norm,
                    h_norm - u_norm,
                    relation.expand(n, -1),
                    modality.expand(n, -1),
                    global_context.expand(n, -1),
                    torch.log1p(q).unsqueeze(-1),
                ),
                dim=-1,
            )
            probabilities = torch.softmax(self.router(features), dim=-1)
        return probabilities * valid.to(probabilities.dtype).unsqueeze(-1)

    def _run_expert(self, expert: nn.Module, value: torch.Tensor) -> torch.Tensor:
        if value.size(0) <= self.expert_chunk_size:
            return expert(value)
        outputs = []
        for start in range(0, value.size(0), self.expert_chunk_size):
            chunk = value[start : start + self.expert_chunk_size]
            if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
                outputs.append(checkpoint(expert, chunk, use_reentrant=False))
            else:
                outputs.append(expert(chunk))
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        selected = value if mask is None or not bool(mask.any()) else value[mask]
        if selected.numel() == 0:
            return 0.0
        return float(selected.detach().float().square().mean().sqrt().cpu())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        if x.ndim != 2 or x.size(0) != self.num_nodes:
            raise RuntimeError("PACE-MAG V1 requires the complete global-ID node matrix")
        if x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError("PACE-MAG input feature width does not match configured Text/Visual dimensions")
        if self.semantic_pu is None or self.semantic_pt is None or self.semantic_pv is None:
            raise RuntimeError("PACE-MAG requires a validated V7B semantic cache before forward")
        self._validate_input_features_once(x)
        if edge_index is None:
            raise ValueError("PACE-MAG requires the physical graph edge_index")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        if self.semantic_pu.device != x.device:
            raise RuntimeError("semantic operators and model inputs are on different devices")
        p0, q0 = self._get_physical_operator(edge_index, x.dtype)
        pt, pv = self.semantic_pt, self.semantic_pv
        q_text, q_visual = self.semantic_q_text, self.semantic_q_visual
        if p0.device != x.device:
            raise RuntimeError("physical operator and model inputs are on different devices")

        raw = {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }
        h = {name: self.projectors[name](raw[name]) for name in _MODALITIES}
        rho = self.rho
        cross_enabled = self.variant != "e1_no_cross_modal"
        if cross_enabled:
            v_to_t = self.cross_modal["visual_to_text"](h["visual"])
            t_to_v = self.cross_modal["text_to_visual"](h["text"])
        else:
            v_to_t = t_to_v = None

        operators = {"physical": p0, "text": pt, "visual": pv}
        qualities = {"physical": q0, "text": q_text, "visual": q_visual}
        messages: dict[str, dict[str, torch.Tensor]] = {name: {} for name in _RELATIONS}
        aggregates: dict[str, dict[str, torch.Tensor]] = {name: {} for name in _RELATIONS}
        responses: dict[str, dict[str, torch.Tensor]] = {name: {} for name in _RELATIONS}
        route_probs: dict[str, dict[str, torch.Tensor]] = {name: {} for name in _RELATIONS}
        expert_values_for_details: dict[str, dict[str, torch.Tensor]] = {name: {} for name in _RELATIONS}

        for modality_id, modality in enumerate(_MODALITIES):
            messages["physical"][modality] = h[modality]
            for relation, relation_index in (("text", 0), ("visual", 1)):
                if not cross_enabled:
                    messages[relation][modality] = h[modality]
                elif modality == "text":
                    messages[relation][modality] = self.compose_cross_modal_message(
                        h[modality], v_to_t, rho[relation_index, modality_id]
                    )
                else:
                    messages[relation][modality] = self.compose_cross_modal_message(
                        h[modality], t_to_v, rho[relation_index, modality_id]
                    )

        for relation_id, relation in enumerate(_RELATIONS):
            operator = operators[relation]
            q = qualities[relation]
            valid = q > 0
            for modality_id, modality in enumerate(_MODALITIES):
                message = messages[relation][modality]
                aggregate = row_quality_aggregate(operator, message, q, self.eps)
                h_norm = self.message_norm(h[modality])
                u_norm = self.aggregate_norm(aggregate)
                difference = h_norm - u_norm
                expert_input = torch.cat((h_norm, u_norm, difference.abs(), h_norm * u_norm), dim=-1)
                if self.variant == "e3_single_expert_control":
                    assert self.single_expert is not None
                    expert_outputs = [self._run_expert(self.single_expert, expert_input)]
                    response = expert_outputs[0]
                    route = torch.ones((x.size(0), 1), dtype=x.dtype, device=x.device)
                else:
                    expert_outputs = [self._run_expert(expert, expert_input) for expert in self.experts]
                    route = self._router_weights(h_norm, u_norm, q, valid, relation_id, modality_id)
                    stacked = torch.stack(expert_outputs, dim=1)
                    response = (stacked * route.unsqueeze(-1)).sum(dim=1)
                response = response * valid.to(response.dtype).unsqueeze(-1)
                aggregates[relation][modality] = aggregate
                responses[relation][modality] = response
                route_probs[relation][modality] = route
                if return_details:
                    expert_values_for_details[relation][modality] = torch.stack(expert_outputs, dim=1)

        mixed: dict[str, torch.Tensor] = {}
        physical_response: dict[str, torch.Tensor] = {}
        semantic_response: dict[str, torch.Tensor] = {}
        q_semantic = q_text + q_visual
        for modality in _MODALITIES:
            physical_response[modality] = responses["physical"][modality]
            numerator = q_text.unsqueeze(-1) * responses["text"][modality] + q_visual.unsqueeze(-1) * responses["visual"][modality]
            semantic_valid = q_semantic > 0
            semantic_response[modality] = numerator / q_semantic.clamp_min(self.eps).unsqueeze(-1)
            semantic_response[modality] = semantic_response[modality] * semantic_valid.to(x.dtype).unsqueeze(-1)
            mixed[modality] = self.mix_norm[modality](
                h[modality] + self.beta_phy * physical_response[modality] + self.beta_sem * semantic_response[modality]
            )
        q_state = {
            modality: self.physical_norm[modality](mixed[modality] + self.gamma * torch.sparse.mm(p0, mixed[modality]))
            for modality in _MODALITIES
        }
        joined = torch.cat(
            (
                q_state["text"],
                q_state["visual"],
                (q_state["text"] - q_state["visual"]).abs(),
                q_state["text"] * q_state["visual"],
            ),
            dim=-1,
        )
        skip = torch.cat((h["text"], h["visual"]), dim=-1)
        z = self.fusion_norm(self.fusion(joined) + self.skip(skip))
        aux_loss = z.new_zeros(())

        info: dict[str, Any] = {"variant": self.variant}
        if return_details:
            details: dict[str, Any] = {
                "effective_node_fraction": {
                    relation: float((qualities[relation] > 0).float().mean().detach().cpu())
                    for relation in _RELATIONS
                },
                "message_rms": {},
                "aggregate_rms": {},
                "expert_output_rms": {},
                "expert_pairwise_cosine": {},
                "router": {},
                "response_rms": {"physical": {}, "semantic": {}},
                "beta_phy": float(self.beta_phy.detach().cpu()),
                "beta_sem": float(self.beta_sem.detach().cpu()),
                "rho": {
                    "text_target_from_text_source": float(rho[0, 0].detach().cpu()),
                    "visual_target_from_text_source": float(rho[0, 1].detach().cpu()),
                    "text_target_from_visual_source": float(rho[1, 0].detach().cpu()),
                    "visual_target_from_visual_source": float(rho[1, 1].detach().cpu()),
                },
                "gamma_fixed": self.gamma,
            }
            for relation in _RELATIONS:
                details["message_rms"][relation] = {}
                details["aggregate_rms"][relation] = {}
                details["expert_output_rms"][relation] = {}
                details["expert_pairwise_cosine"][relation] = {}
                details["router"][relation] = {}
                valid = qualities[relation] > 0
                for modality in _MODALITIES:
                    details["message_rms"][relation][modality] = self._rms(messages[relation][modality])
                    details["aggregate_rms"][relation][modality] = self._rms(aggregates[relation][modality], valid)
                    experts = expert_values_for_details[relation][modality]
                    details["expert_output_rms"][relation][modality] = [
                        self._rms(experts[:, k], valid) for k in range(experts.size(1))
                    ]
                    if experts.size(1) > 1 and bool(valid.any()):
                        selected = experts[valid].float()
                        normalized = F.normalize(selected, dim=-1, eps=self.eps)
                        similarities = []
                        for left in range(normalized.size(1)):
                            for right in range(left + 1, normalized.size(1)):
                                similarities.append((normalized[:, left] * normalized[:, right]).sum(dim=-1).mean())
                        details["expert_pairwise_cosine"][relation][modality] = [
                            float(value.detach().cpu()) for value in similarities
                        ]
                    else:
                        details["expert_pairwise_cosine"][relation][modality] = []
                    route = route_probs[relation][modality]
                    if route.size(1) > 1 and bool(valid.any()):
                        mean_route = route[valid].mean(dim=0)
                        entropy = -(route[valid].clamp_min(self.eps) * route[valid].clamp_min(self.eps).log()).sum(dim=-1).mean()
                    elif bool(valid.any()):
                        mean_route = route[valid].mean(dim=0)
                        entropy = route.new_zeros(())
                    else:
                        mean_route = route.new_zeros((route.size(1),))
                        entropy = route.new_zeros(())
                    details["router"][relation][modality] = {
                        "mean_probability": [float(value.detach().cpu()) for value in mean_route],
                        "mean_entropy": float(entropy.detach().cpu()),
                        "valid_node_probability_sum_min": float(route[valid].sum(dim=-1).min().detach().cpu()) if bool(valid.any()) else None,
                        "valid_node_probability_sum_max": float(route[valid].sum(dim=-1).max().detach().cpu()) if bool(valid.any()) else None,
                    }
            for modality in _MODALITIES:
                details["response_rms"]["physical"][modality] = self._rms(physical_response[modality])
                details["response_rms"]["semantic"][modality] = self._rms(semantic_response[modality])
            details["final_embedding"] = {
                "finite": bool(torch.isfinite(z).all()),
                "rms": self._rms(z),
            }
            info["details"] = details
        return z, None, None, aux_loss, info

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        for name, parameter in self.named_parameters():
            grad = parameter.grad
            group = name.split(".", 1)[0]
            row = diagnostics.setdefault(group, {"parameter_count": 0, "gradient_present": 0, "gradient_finite": True, "gradient_nonzero": 0, "gradient_l2_norm": 0.0})
            row["parameter_count"] += int(parameter.numel())
            if grad is not None:
                row["gradient_present"] += int(parameter.numel())
                finite = bool(torch.isfinite(grad).all())
                norm = float(grad.detach().float().norm().cpu())
                row["gradient_finite"] = row["gradient_finite"] and finite
                row["gradient_l2_norm"] += norm * norm
                row["gradient_nonzero"] += int(torch.count_nonzero(grad).item())
        for row in diagnostics.values():
            row["gradient_l2_norm"] = math.sqrt(row["gradient_l2_norm"])
        self._gradient_diagnostics = diagnostics
        return diagnostics
