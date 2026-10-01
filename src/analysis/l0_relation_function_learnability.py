from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from src.analysis.p0p1_propagation_probe import DATASETS, ProbeData, load_probe_data
from src.analysis.p11_p12_operator_rescue import load_frozen_h0, selected_validation_edges
from src.models.adaptive_prop_m0 import (
    _ModalityInterpreter,
    _PairEncoder,
    _RelationEncoder,
    graph_standardized_log_degree,
    incoming_degree,
    leave_one_out_context,
    remove_self_messages,
)
from src.models.provenance_executor_e01 import Model as E01Model


SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
TARGET_NAMES = ("smooth_utility", "delta_absdiff", "delta_product")
TARGET_COLUMNS = (
    "u_raw_smooth_text", "u_raw_absdiff_text", "u_raw_product_text",
    "u_raw_smooth_visual", "u_raw_absdiff_visual", "u_raw_product_visual",
)
UTILITY_COLUMNS = (
    "u_raw_smooth_text", "u_raw_absdiff_text", "u_raw_product_text",
    "u_raw_smooth_visual", "u_raw_absdiff_visual", "u_raw_product_visual",
)
KEY_COLUMNS = ("src", "dst", "dst_degree")
MASTER_DIM = 1542
HIDDEN_DIM = 128
EVIDENCE_VARIANTS = (
    "SIM_ONLY", "TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL",
    "ENDPOINT_LOCAL_SHUFFLED_TARGET",
)
REPRESENTATIONS = ("Q_PAIR", "R_SHARED", "U_MODAL")


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    value = pearsonr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def load_utility_table(path: Path, dataset: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Read only edge keys and the six allowed target-source utility columns."""
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=[*KEY_COLUMNS, *UTILITY_COLUMNS])
    frame["src"] = frame["src"].astype("int64")
    frame["dst"] = frame["dst"].astype("int64")
    frame["dst_degree"] = frame["dst_degree"].astype("int64")
    if frame.duplicated(["src", "dst"]).any():
        raise AssertionError(f"{dataset}/{seed}: duplicate directed physical edges in P1.3 table")
    if not np.isfinite(frame[list(UTILITY_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError(f"{dataset}/{seed}: non-finite P1.3 utility target")
    raw = frame[list(UTILITY_COLUMNS)].to_numpy(dtype=np.float64)
    y = np.stack(
        [raw[:, 0], raw[:, 1] - raw[:, 0], raw[:, 2] - raw[:, 0],
         raw[:, 3], raw[:, 4] - raw[:, 3], raw[:, 5] - raw[:, 3]],
        axis=1,
    )
    if y.shape != (len(frame), 6) or not np.isfinite(y).all():
        raise AssertionError("primary targets must be a finite [edges,6] array")
    return frame, y


def audit_utility_formula(frame: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    raw = frame[list(UTILITY_COLUMNS)].to_numpy(dtype=np.float64)
    expected = np.stack(
        [raw[:, 0], raw[:, 1] - raw[:, 0], raw[:, 2] - raw[:, 0],
         raw[:, 3], raw[:, 4] - raw[:, 3], raw[:, 5] - raw[:, 3]], axis=1,
    )
    max_abs = float(np.max(np.abs(expected - y), initial=0.0))
    if max_abs > 2e-7:
        raise AssertionError(f"target formula mismatch: {max_abs}")
    return {"rows": len(frame), "target_dim": int(y.shape[1]), "max_formula_abs_error": max_abs,
            "finite_targets": bool(np.isfinite(y).all()), "input_columns": list(frame.columns)}


def target_reliability(tables: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_names = [
        ("u_raw_smooth_text", "u_raw_absdiff_text", "u_raw_product_text"),
        ("u_raw_smooth_visual", "u_raw_absdiff_visual", "u_raw_product_visual"),
    ]
    target_names = ["U_S", "Delta_D", "Delta_P", "U_S", "Delta_D", "Delta_P"]
    seeds = SEEDS
    for dataset in DATASETS:
        for left_i in range(len(seeds)):
            for right_i in range(left_i + 1, len(seeds)):
                s1, s2 = seeds[left_i], seeds[right_i]
                def targets(frame: pd.DataFrame) -> pd.DataFrame:
                    out = frame[["src", "dst"]].copy().set_index(["src", "dst"])
                    for modality_i, (smooth_col, abs_col, prod_col) in enumerate(source_names):
                        prefix = "text" if modality_i == 0 else "visual"
                        out[f"U_S_{prefix}"] = frame[smooth_col].to_numpy(dtype=np.float64)
                        out[f"Delta_D_{prefix}"] = (
                            frame[abs_col].to_numpy(dtype=np.float64) - frame[smooth_col].to_numpy(dtype=np.float64)
                        )
                        out[f"Delta_P_{prefix}"] = (
                            frame[prod_col].to_numpy(dtype=np.float64) - frame[smooth_col].to_numpy(dtype=np.float64)
                        )
                    return out
                a = targets(tables[(dataset, s1)])
                b = targets(tables[(dataset, s2)])
                common = a.index.intersection(b.index, sort=False)
                aa, bb = a.loc[common], b.loc[common]
                counts_by_dst = pd.Series(common.get_level_values("dst")).value_counts()
                for col_idx, (modality, target_name) in enumerate(
                    [(m, t) for m in MODALITIES for t in ("U_S", "Delta_D", "Delta_P")]
                ):
                    column = f"{target_name}_{modality}"
                    x = aa[column].to_numpy(dtype=np.float64)
                    z = bb[column].to_numpy(dtype=np.float64)
                    centered_rho = float("nan")
                    if target_name in ("Delta_D", "Delta_P"):
                        dsts = common.get_level_values("dst").to_numpy()
                        dx = x - pd.Series(x).groupby(dsts).transform("mean").to_numpy()
                        dz = z - pd.Series(z).groupby(dsts).transform("mean").to_numpy()
                        centered_rho = _rho(dx, dz)
                    nsmaller = min(len(a), len(b))
                    rows.append({
                        "dataset": dataset, "seed_a": s1, "seed_b": s2,
                        "modality": modality, "target": target_name,
                        "n_a": len(a), "n_b": len(b), "overlap_edges": len(common),
                        "overlap_fraction_smaller": len(common) / nsmaller if nsmaller else np.nan,
                        "common_targets": int(len(counts_by_dst)),
                        "spearman": _rho(x, z), "pearson": _pearson(x, z),
                        "sign_agreement_positive_vs_nonpositive": float(np.mean((x > 0) == (z > 0))),
                        "exact_zero_fraction_a": float(np.mean(x == 0)),
                        "exact_zero_fraction_b": float(np.mean(z == 0)),
                        "centered_spearman": centered_rho,
                    })
    return pd.DataFrame(rows)


def balanced_group_folds(dst: np.ndarray, n_splits: int = 3, seed: int = 42) -> pd.DataFrame:
    """Greedy edge-balanced target-group fold assignment, deterministic per run."""
    series = pd.Series(dst).value_counts(sort=False)
    groups = series.index.to_numpy(dtype=np.int64)
    counts = series.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    tie_rank = np.empty(len(groups), dtype=np.int64)
    tie_rank[perm] = np.arange(len(groups))
    order = sorted(range(len(groups)), key=lambda i: (-int(counts[i]), int(tie_rank[i])))
    fold_counts = np.zeros(n_splits, dtype=np.int64)
    tie_order = rng.permutation(n_splits).tolist()
    fold_ids = np.full(len(groups), -1, dtype=np.int64)
    for idx in order:
        fold = min(range(n_splits), key=lambda f: (fold_counts[f], tie_order.index(f)))
        fold_ids[idx] = fold
        fold_counts[fold] += counts[idx]
    assignment = pd.DataFrame({"dst": groups, "fold": fold_ids, "edge_count": counts})
    if assignment["dst"].duplicated().any() or (assignment["fold"] < 0).any():
        raise AssertionError("invalid target group fold assignment")
    return assignment.sort_values("dst").reset_index(drop=True)


def inner_group_split(dst: np.ndarray, outer_train_rows: np.ndarray, seed: int,
                      val_fraction: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    groups = np.unique(dst[outer_train_rows])
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    n_val = max(1, int(round(len(groups) * val_fraction)))
    n_val = min(n_val, len(groups) - 1)
    val_groups = set(groups[:n_val].tolist())
    val_mask = np.fromiter((int(dst[row]) in val_groups for row in outer_train_rows),
                           dtype=bool, count=len(outer_train_rows))
    train_rows, val_rows = outer_train_rows[~val_mask], outer_train_rows[val_mask]
    if set(dst[train_rows]).intersection(set(dst[val_rows])):
        raise AssertionError("inner target groups overlap")
    return train_rows, val_rows


def target_balanced_weights(dst_degree: np.ndarray) -> np.ndarray:
    values = 1.0 / np.maximum(np.asarray(dst_degree, dtype=np.float64), 1.0)
    values /= values.mean()
    return values.astype(np.float32)


@dataclass
class FeatureLayout:
    master: torch.Tensor
    slices: dict[str, slice]
    masks: dict[str, torch.Tensor]
    names: list[str]


def _concat_cpu(*items: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.detach().to(device="cpu", dtype=torch.float32).contiguous() for x in items], dim=-1)


def build_master_features(h_t: torch.Tensor, h_v: torch.Tensor, src: torch.Tensor,
                          dst: torch.Tensor, degree: torch.Tensor,
                          edge_index: torch.Tensor, eps: float = 1e-8) -> FeatureLayout:
    """Construct the 1542D P1.3 H0-based observable evidence tensor on CPU."""
    h_t, h_v = h_t.detach().float().cpu(), h_v.detach().float().cpu()
    src, dst = src.detach().long().cpu(), dst.detach().long().cpu()
    edge_index = edge_index.detach().long().cpu()
    degree = degree.detach().long().cpu()
    if h_t.shape != h_v.shape or h_t.size(1) != HIDDEN_DIM:
        raise AssertionError("P1.3 H0 must have matching [nodes,128] modality matrices")
    graph_src, graph_dst = edge_index
    sums_t = h_t.new_zeros(h_t.shape).index_add_(0, graph_dst, h_t[graph_src])
    sums_v = h_v.new_zeros(h_v.shape).index_add_(0, graph_dst, h_v[graph_src])
    d = degree.clamp_min(1).to(torch.float32).unsqueeze(-1)
    full_t, full_v = sums_t[dst] / d[dst], sums_v[dst] / d[dst]
    remaining = (degree[dst] - 1).clamp_min(1).to(torch.float32).unsqueeze(-1)
    loo_t = (sums_t[dst] - h_t[src]) / remaining
    loo_v = (sums_v[dst] - h_v[src]) / remaining
    loo_t = torch.where((degree[dst] > 1).unsqueeze(-1), loo_t, torch.zeros_like(loo_t))
    loo_v = torch.where((degree[dst] > 1).unsqueeze(-1), loo_v, torch.zeros_like(loo_v))
    target_t, target_v, source_t, source_v = h_t[dst], h_v[dst], h_t[src], h_v[src]
    cos_t = F.cosine_similarity(target_t, source_t, dim=-1, eps=eps).unsqueeze(-1)
    cos_v = F.cosine_similarity(target_v, source_v, dim=-1, eps=eps).unsqueeze(-1)
    loo_cos_t = F.cosine_similarity(source_t, loo_t, dim=-1, eps=eps).unsqueeze(-1)
    loo_cos_v = F.cosine_similarity(source_v, loo_v, dim=-1, eps=eps).unsqueeze(-1)
    deg_z = graph_standardized_log_degree(degree, eps).cpu()
    scalars = torch.cat([cos_t, cos_v, loo_cos_t, loo_cos_v,
                         deg_z[dst, None], deg_z[src, None]], dim=-1)
    blocks = {
        "target": _concat_cpu(target_t, target_v),
        "source": _concat_cpu(source_t, source_v),
        "absdiff": _concat_cpu((target_t - source_t).abs(), (target_v - source_v).abs()),
        "product": _concat_cpu(target_t * source_t, target_v * source_v),
        "full_neighbor": _concat_cpu(full_t, full_v),
        "loo_neighbor": _concat_cpu(loo_t, loo_v),
        "scalars": scalars.contiguous(),
    }
    pieces = []
    slices: dict[str, slice] = {}
    pos = 0
    for name, block in blocks.items():
        slices[name] = slice(pos, pos + block.size(1))
        pieces.append(block)
        pos += block.size(1)
    master = torch.cat(pieces, dim=1).contiguous()
    if master.shape[1] != MASTER_DIM:
        raise AssertionError(f"master feature dim {master.shape[1]} != {MASTER_DIM}")
    masks: dict[str, torch.Tensor] = {}
    for variant in EVIDENCE_VARIANTS:
        mask = torch.zeros(MASTER_DIM, dtype=torch.bool)
        if variant == "SIM_ONLY":
            mask[slices["scalars"].start:slices["scalars"].start + 2] = True
        elif variant == "TARGET_ONLY":
            mask[slices["target"]] = True
            mask[slices["full_neighbor"]] = True
            mask[slices["scalars"].start + 4:slices["scalars"].start + 5] = True
        elif variant == "ENDPOINT":
            for name in ("target", "source", "absdiff", "product"):
                mask[slices[name]] = True
            mask[slices["scalars"].start:slices["scalars"].start + 2] = True
            mask[slices["scalars"].start + 4:slices["scalars"].stop] = True
        else:
            mask[:] = True
        masks[variant] = mask
    return FeatureLayout(master, slices, masks, [f"feature_{i}" for i in range(MASTER_DIM)])


class EvidenceMLP(nn.Module):
    def __init__(self, output_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(MASTER_DIM, 128), nn.GELU(), nn.Linear(128, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateReadout(nn.Module):
    def __init__(self, input_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def m0_pair_evidence(target: torch.Tensor, source: torch.Tensor, loo_similarity: torch.Tensor,
                     degree_z_target: torch.Tensor, degree_z_source: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """Exact M0 edge-pair evidence: four P blocks, two cosines and two degrees."""
    cosine = F.cosine_similarity(target, source, dim=-1, eps=eps).unsqueeze(-1)
    if loo_similarity.ndim == 1:
        loo_similarity = loo_similarity.unsqueeze(-1)
    return torch.cat([target, source, (target - source).abs(), target * source,
                      cosine, loo_similarity,
                      degree_z_target.reshape(-1, 1), degree_z_source.reshape(-1, 1)], dim=-1)


class M0StateDirectProbe(nn.Module):
    """M0 q/r/u relation path supervised directly on the six P1.3 targets."""
    def __init__(self):
        super().__init__()
        self.rel_proj_t = nn.Sequential(nn.Linear(128, 32, bias=False), nn.LayerNorm(32))
        self.rel_proj_v = nn.Sequential(nn.Linear(128, 32, bias=False), nn.LayerNorm(32))
        self.phi_pair = _PairEncoder(4 * 32 + 4)
        self.phi_rel = _RelationEncoder()
        self.modality_embeddings = nn.Parameter(torch.empty(2, 8))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        self.phi_mod = _ModalityInterpreter(64 + 32 + 8)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        self.eps = 1e-8

    def forward(self, h_t: torch.Tensor, h_v: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                graph_src: torch.Tensor, graph_dst: torch.Tensor,
                degree: torch.Tensor, deg_z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward requires exact P1.3 edge positions; use forward_from_edge_positions")

    def forward_from_edge_positions(self, h_t: torch.Tensor, h_v: torch.Tensor,
                                    edge_positions: torch.Tensor, graph_src: torch.Tensor,
                                    graph_dst: torch.Tensor, degree: torch.Tensor,
                                    deg_z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p = [self.rel_proj_t(h_t), self.rel_proj_v(h_v)]
        src, dst = graph_src[edge_positions], graph_dst[edge_positions]
        loo = []
        with torch.no_grad():
            for modality in range(2):
                all_context = leave_one_out_context(p[modality].detach(), graph_src, graph_dst, degree, self.eps)
                loo.append(all_context[edge_positions])
        pair = []
        for modality in range(2):
            target, source = p[modality][dst], p[modality][src]
            pair.append(m0_pair_evidence(target, source, loo[modality],
                                         deg_z[dst], deg_z[src], self.eps))
        q_t, q_v = self.phi_pair(pair[0]), self.phi_pair(pair[1])
        relation = self.phi_rel(torch.cat([q_t, q_v, (q_t - q_v).abs(), q_t * q_v], dim=-1))
        u = []
        for modality, q in enumerate((q_t, q_v)):
            em = self.modality_embeddings[modality].expand(q.size(0), -1)
            u.append(self.phi_mod(torch.cat([relation, q, em], dim=-1)))
        out = torch.cat([self.head(u[0]), self.head(u[1])], dim=-1)
        return out, {"q_t": q_t, "q_v": q_v, "r": relation, "u_t": u[0], "u_v": u[1],
                     "src": src, "dst": dst, "loo_t": loo[0], "loo_v": loo[1]}


def feature_stats(x: torch.Tensor, rows: np.ndarray, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    values = x[torch.as_tensor(rows, dtype=torch.long)]
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std


def standardize_targets(y: np.ndarray, outer_train_rows: np.ndarray,
                        eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = y[outer_train_rows].astype(np.float64)
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), eps)
    return mean.astype(np.float32), std.astype(np.float32)


def shuffle_targets_within_destination(y: np.ndarray, dst: np.ndarray,
                                      rows: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    output = y.copy()
    rng = np.random.default_rng(seed)
    changed_groups = 0
    eligible_groups = 0
    counts_before: dict[int, np.ndarray] = {}
    counts_after: dict[int, np.ndarray] = {}
    for target in np.unique(dst[rows]):
        group = rows[dst[rows] == target]
        if len(group) <= 1:
            continue
        eligible_groups += 1
        perm = rng.permutation(len(group))
        if np.array_equal(perm, np.arange(len(group))):
            perm = np.roll(perm, 1)
        output[group] = y[group[perm]]
        changed_groups += int(not np.array_equal(output[group], y[group]))
        counts_before[int(target)] = y[group][np.lexsort(y[group].T[::-1])]
        counts_after[int(target)] = output[group][np.lexsort(output[group].T[::-1])]
    preserved = all(np.array_equal(counts_before[t], counts_after[t]) for t in counts_before)
    return output, {"eligible_groups": eligible_groups, "changed_groups": changed_groups,
                    "multiset_preserved": bool(preserved)}


def metric_bundle(y: np.ndarray, pred: np.ndarray, dst: np.ndarray,
                  dst_degree: np.ndarray, modality: str, target_type: str) -> dict[str, Any]:
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("metric inputs must be aligned finite vectors")
    binary = (y > 0).astype(np.int8)
    pred_binary = (pred > 0).astype(np.int8)
    auc = float("nan") if len(np.unique(binary)) < 2 else float(roc_auc_score(binary, pred))
    residual_y = y.copy()
    residual_pred = pred.copy()
    for target in np.unique(dst):
        mask = dst == target
        residual_y[mask] -= y[mask].mean()
        residual_pred[mask] -= pred[mask].mean()
    per_target: list[float] = []
    eligible = 0
    for target in np.unique(dst):
        mask = dst == target
        if int(np.max(dst_degree[mask])) < 5:
            continue
        eligible += 1
        if len(y[mask]) < 2 or np.ptp(y[mask]) == 0 or np.ptp(pred[mask]) == 0:
            continue
        rho = _rho(y[mask], pred[mask])
        if np.isfinite(rho):
            per_target.append(rho)
    ranks = np.asarray(per_target, dtype=np.float64)
    return {
        "n_edges": len(y), "spearman": _rho(y, pred), "pearson": _pearson(y, pred),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "sign_auroc": auc,
        "sign_balanced_accuracy": float(balanced_accuracy_score(binary, pred_binary)) if len(y) else float("nan"),
        "sign_prevalence": float(binary.mean()) if len(y) else float("nan"),
        "within_target_residual_spearman": _rho(residual_y, residual_pred),
        "per_target_eligible_count": eligible, "per_target_valid_count": len(ranks),
        "per_target_rank_mean": float(ranks.mean()) if len(ranks) else float("nan"),
        "per_target_rank_median": float(np.median(ranks)) if len(ranks) else float("nan"),
        "per_target_rank_q25": float(np.quantile(ranks, .25)) if len(ranks) else float("nan"),
        "per_target_rank_q75": float(np.quantile(ranks, .75)) if len(ranks) else float("nan"),
        "modality": modality, "target_type": target_type,
    }


def high_margin_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    threshold = float(np.quantile(np.abs(y), .75))
    keep = np.abs(y) >= threshold
    yy, pp = y[keep], pred[keep]
    binary = (yy > 0).astype(np.int8)
    auc = float("nan") if len(np.unique(binary)) < 2 else float(roc_auc_score(binary, pp))
    return {"high_margin_threshold_abs": threshold, "high_margin_n": int(keep.sum()),
            "high_margin_sign_auroc": auc,
            "high_margin_sign_balanced_accuracy": float(balanced_accuracy_score(binary, pp > 0)) if len(yy) else float("nan"),
            "high_margin_spearman": _rho(yy, pp)}


def count_parameters(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_edge = F.huber_loss(pred, target, reduction="none", delta=1.0).mean(dim=-1)
    normalized = weights / weights.mean().clamp_min(1e-12)
    return (per_edge * normalized).mean()


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_fullbatch_probe(
    model: nn.Module,
    forward_all,
    y_scaled: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    edge_weights: np.ndarray,
    device: torch.device,
    max_epochs: int = 150,
    min_epoch: int = 20,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Train with a full supervised-edge loss; groups, weights and targets are supplied by caller."""
    model.to(device)
    model.train()
    y_t = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(edge_weights, dtype=torch.float32, device=device)
    tr = torch.as_tensor(train_rows, dtype=torch.long, device=device)
    va = torch.as_tensor(val_rows, dtype=torch.long, device=device)
    te = torch.as_tensor(test_rows, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    best_state = None
    train_loss_value = float("nan")
    finite_gradients = True
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = forward_all()
        loss = weighted_huber(pred[tr], y_t[tr], weights_t[tr])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        loss.backward()
        grad_tensors = [p.grad for p in model.parameters() if p.grad is not None]
        finite_gradients = finite_gradients and all(bool(torch.isfinite(g).all()) for g in grad_tensors)
        if not finite_gradients:
            raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        train_loss_value = float(loss.detach().item())
        model.eval()
        with torch.no_grad():
            val_pred = forward_all()
            val_loss = weighted_huber(val_pred[va], y_t[va], weights_t[va])
        current = float(val_loss.item())
        if not np.isfinite(current):
            raise FloatingPointError(f"non-finite inner validation loss at epoch {epoch}")
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            best_state = _state_cpu(model)
            wait = 0
        elif epoch >= min_epoch:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("probe training did not select an inner-validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        test_prediction = forward_all()[te].detach().cpu().numpy()
    if not np.isfinite(test_prediction).all():
        raise FloatingPointError("non-finite outer-test prediction")
    return {
        "model": model, "prediction_scaled": test_prediction,
        "best_epoch": best_epoch, "epochs_run": epoch,
        "best_inner_weighted_huber": best_loss,
        "final_train_weighted_huber": train_loss_value,
        "finite_gradients": finite_gradients,
        "parameter_count": count_parameters(model),
    }


def metrics_for_predictions(prediction_scaled: np.ndarray, y: np.ndarray,
                            target_mean: np.ndarray, target_std: np.ndarray,
                            test_rows: np.ndarray, dst: np.ndarray,
                            degree_by_edge: np.ndarray, dataset: str, seed: int,
                            fold: int, model_name: str, modalities: tuple[str, ...] = MODALITIES,
                            target_names: tuple[str, ...] = TARGET_NAMES,
                            target_offsets: tuple[int, ...] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred = prediction_scaled * target_std + target_mean
    if target_offsets is None:
        target_offsets = tuple(range(6))
    rows, high_rows = [], []
    for output_i, target_idx in enumerate(target_offsets):
        modality_i = target_idx // 3
        target_type_i = target_idx % 3
        yy = y[test_rows, target_idx].astype(np.float64)
        pp = pred[:, output_i].astype(np.float64)
        dst_test = dst[test_rows]
        degree_test = degree_by_edge[test_rows]
        metric = metric_bundle(yy, pp, dst_test, degree_test,
                              modalities[modality_i], target_names[target_type_i])
        rows.append({"dataset": dataset, "seed": seed, "fold": fold, "model": model_name,
                     **metric})
        if target_type_i in (1, 2):
            high_rows.append({"dataset": dataset, "seed": seed, "fold": fold,
                              "model": model_name, "modality": modalities[modality_i],
                              "target_type": target_names[target_type_i],
                              **high_margin_metrics(yy, pp)})
    return rows, high_rows


def _feature_standardized_variant(master: torch.Tensor, train_rows: np.ndarray,
                                  mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, std = feature_stats(master, train_rows)
    x = ((master - mean) / std).contiguous()
    x[:, ~mask] = 0.0
    return x, mean, std


def _edge_group_weights(degree: np.ndarray) -> np.ndarray:
    return target_balanced_weights(degree)


def _metric_target_selection(y: np.ndarray, modality_i: int) -> np.ndarray:
    return y[:, modality_i * 3:modality_i * 3 + 3]


def _model_initialization_audit(seed: int) -> tuple[dict[str, torch.Tensor], dict[str, int], bool]:
    states: dict[str, dict[str, torch.Tensor]] = {}
    counts: dict[str, int] = {}
    for name in EVIDENCE_VARIANTS:
        set_seed(seed)
        model = EvidenceMLP()
        states[name] = _state_cpu(model)
        counts[name] = count_parameters(model)
    bitwise = all(
        states[EVIDENCE_VARIANTS[0]].keys() == states[name].keys()
        and all(torch.equal(states[EVIDENCE_VARIANTS[0]][key], states[name][key])
                for key in states[name])
        for name in EVIDENCE_VARIANTS[1:]
    )
    if len(set(counts.values())) != 1 or not bitwise:
        raise AssertionError("EvidenceMLP variants must have identical counts and bitwise initialization")
    return states[EVIDENCE_VARIANTS[0]], counts, bitwise


def run_evidence_fit(master: torch.Tensor, masks: dict[str, torch.Tensor], y: np.ndarray,
                     dst: np.ndarray, degree_by_edge: np.ndarray,
                     train_rows: np.ndarray, val_rows: np.ndarray, test_rows: np.ndarray,
                     dataset: str, seed: int, fold: int, variant: str,
                     device: torch.device, max_epochs: int = 150,
                     save_path: Path | None = None,
                     init_state: dict[str, torch.Tensor] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    # The protocol defines target moments over the complete outer-training fold.
    outer_train = np.concatenate([train_rows, val_rows])
    y_mean, y_std = standardize_targets(y, outer_train)
    y_scaled = (y - y_mean) / y_std
    mask = masks[variant]
    x, x_mean, x_std = _feature_standardized_variant(master, outer_train, mask)
    if x.shape[1] != MASTER_DIM or not torch.isfinite(x).all():
        raise AssertionError("evidence model input must be finite 1542D standardized master features")
    model = EvidenceMLP()
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    x_device = x.to(device, non_blocking=False)
    fit = fit_fullbatch_probe(
        model, lambda: model(x_device), y_scaled, train_rows, val_rows, test_rows,
        _edge_group_weights(degree_by_edge), device, max_epochs=max_epochs,
    )
    metric_rows, high_rows = metrics_for_predictions(
        fit["prediction_scaled"], y, y_mean, y_std, test_rows, dst,
        degree_by_edge, dataset, seed, fold, variant,
    )
    metadata = {k: fit[k] for k in ("best_epoch", "epochs_run", "best_inner_weighted_huber",
                                     "final_train_weighted_huber", "finite_gradients", "parameter_count")}
    metadata.update({"input_dim": MASTER_DIM, "mask_kept_dimensions": int(mask.sum()),
                     "feature_mean_fit_rows": len(outer_train), "target_mean_fit_rows": len(outer_train),
                     "feature_stats_train_only": True, "target_stats_train_only": True})
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": _state_cpu(fit["model"]), "metadata": metadata}, save_path)
    del x_device, fit["model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_rows, high_rows, metadata


def run_direct_state_fit(h_t: torch.Tensor, h_v: torch.Tensor,
                         edge_positions: np.ndarray, graph_src: np.ndarray,
                         graph_dst: np.ndarray, degree: np.ndarray, deg_z: np.ndarray,
                         y: np.ndarray, dst: np.ndarray, degree_by_edge: np.ndarray,
                         train_rows: np.ndarray, val_rows: np.ndarray, test_rows: np.ndarray,
                         dataset: str, seed: int, fold: int,
                         device: torch.device, max_epochs: int = 150,
                         save_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    outer_train = np.concatenate([train_rows, val_rows])
    y_mean, y_std = standardize_targets(y, outer_train)
    y_scaled = (y - y_mean) / y_std
    set_seed(71021 + seed * 17 + fold)
    model = M0StateDirectProbe()
    ht = h_t.to(device)
    hv = h_v.to(device)
    edge_positions_t = torch.as_tensor(edge_positions, dtype=torch.long, device=device)
    graph_src_t = torch.as_tensor(graph_src, dtype=torch.long, device=device)
    graph_dst_t = torch.as_tensor(graph_dst, dtype=torch.long, device=device)
    degree_t = torch.as_tensor(degree, dtype=torch.long, device=device)
    deg_z_t = torch.as_tensor(deg_z, dtype=torch.float32, device=device)
    forward = lambda: model.forward_from_edge_positions(
        ht, hv, edge_positions_t, graph_src_t, graph_dst_t, degree_t, deg_z_t
    )[0]
    fit = fit_fullbatch_probe(
        model, forward, y_scaled, train_rows, val_rows, test_rows,
        _edge_group_weights(degree_by_edge), device, max_epochs=max_epochs,
    )
    metric_rows, high_rows = metrics_for_predictions(
        fit["prediction_scaled"], y, y_mean, y_std, test_rows, dst,
        degree_by_edge, dataset, seed, fold, "M0StateDirect",
    )
    metadata = {k: fit[k] for k in ("best_epoch", "epochs_run", "best_inner_weighted_huber",
                                     "final_train_weighted_huber", "finite_gradients", "parameter_count")}
    metadata.update({"input_h0_frozen": True, "relation_state_dimensions": {"q": 32, "r": 64, "u": 64},
                     "target_stats_train_only": True, "direct_utility_supervision": True})
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": _state_cpu(fit["model"]), "metadata": metadata}, save_path)
    del ht, hv, edge_positions_t, graph_src_t, graph_dst_t, degree_t, deg_z_t, fit["model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_rows, high_rows, metadata


def run_state_readout_fit(representation: np.ndarray, y: np.ndarray, dst: np.ndarray,
                          degree_by_edge: np.ndarray, train_rows: np.ndarray,
                          val_rows: np.ndarray, test_rows: np.ndarray,
                          dataset: str, seed: int, fold: int, modality_i: int,
                          representation_name: str, device: torch.device,
                          max_epochs: int = 150,
                          save_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_columns = np.arange(modality_i * 3, modality_i * 3 + 3)
    outer_train = np.concatenate([train_rows, val_rows])
    y3 = y[:, output_columns]
    y_mean, y_std = standardize_targets(y3, outer_train)
    y_scaled = (y3 - y_mean) / y_std
    x_tensor = torch.as_tensor(representation, dtype=torch.float32)
    x_mean, x_std = feature_stats(x_tensor, outer_train)
    x_standard = ((x_tensor - x_mean) / x_std).contiguous()
    x_device = x_standard.to(device)
    set_seed(88261 + seed * 19 + fold * 3 + modality_i)
    model = StateReadout()
    fit = fit_fullbatch_probe(
        model, lambda: model(x_device), y_scaled, train_rows, val_rows, test_rows,
        _edge_group_weights(degree_by_edge), device, max_epochs=max_epochs,
    )
    prediction = fit["prediction_scaled"] * y_std + y_mean
    rows, high_rows = [], []
    for out_i, target_idx in enumerate(output_columns):
        yy = y[test_rows, target_idx].astype(np.float64)
        pp = prediction[:, out_i].astype(np.float64)
        metric = metric_bundle(yy, pp, dst[test_rows], degree_by_edge[test_rows],
                              MODALITIES[modality_i], TARGET_NAMES[out_i])
        rows.append({"dataset": dataset, "seed": seed, "fold": fold,
                     "modality": MODALITIES[modality_i], "representation": representation_name,
                     "model": f"Frozen_{representation_name}", **metric})
        if out_i in (1, 2):
            high_rows.append({"dataset": dataset, "seed": seed, "fold": fold,
                              "modality": MODALITIES[modality_i],
                              "representation": representation_name,
                              "model": f"Frozen_{representation_name}",
                              "target_type": TARGET_NAMES[out_i], **high_margin_metrics(yy, pp)})
    metadata = {k: fit[k] for k in ("best_epoch", "epochs_run", "best_inner_weighted_huber",
                                     "finite_gradients", "parameter_count")}
    metadata.update({"representation_dim": int(representation.shape[1]),
                     "target_stats_train_only": True, "representation_stats_train_only": True,
                     "checkpoint_frozen": True})
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": _state_cpu(fit["model"]), "metadata": metadata}, save_path)
    del x_device, fit["model"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, high_rows, metadata


def edge_positions_for_table(data: ProbeData, frame: pd.DataFrame,
                             device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph_edges = data.edge_index.to(device)
    graph_edges, graph_src, graph_dst = remove_self_messages(graph_edges)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask[data.val_idx.to(device)] = True
    positions = torch.where(val_mask[graph_dst])[0]
    degree = incoming_degree(graph_dst, data.num_nodes)
    expected_src = frame["src"].to_numpy(dtype=np.int64)
    expected_dst = frame["dst"].to_numpy(dtype=np.int64)
    expected_degree = frame["dst_degree"].to_numpy(dtype=np.int64)
    actual = np.stack([graph_src[positions].cpu().numpy(), graph_dst[positions].cpu().numpy(),
                       degree[graph_dst[positions]].cpu().numpy()], axis=1)
    expected = np.stack([expected_src, expected_dst, expected_degree], axis=1)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise AssertionError("ordered P1.3 table support does not exactly match physical validation edges")
    return (positions.cpu().numpy().astype(np.int64), graph_src.cpu().numpy().astype(np.int64),
            graph_dst.cpu().numpy().astype(np.int64), degree.cpu().numpy().astype(np.int64))


def load_h0_and_master(dataset: str, seed: int, frame: pd.DataFrame,
                       data_root: str, device: torch.device) -> tuple[ProbeData, torch.Tensor,
                                                                     torch.Tensor, FeatureLayout,
                                                                     dict[str, Any], np.ndarray,
                                                                     np.ndarray, np.ndarray, np.ndarray]:
    data = load_probe_data(dataset, seed, data_root)
    checkpoint = Path("outputs/p0p1_propagation_heterogeneity/checkpoints/runs") / dataset / f"seed_{seed}_semantic_only.pt"
    reference = Path("outputs/p0p1_propagation_heterogeneity/runs") / dataset / f"seed_{seed}" / "probe_performance.csv"
    model, h_t, h_v, h0_record = load_frozen_h0(data, checkpoint, device, reference)
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("P1.3 semantic model must remain fully frozen")
    positions, graph_src, graph_dst, degree = edge_positions_for_table(data, frame, device)
    if len(positions) != len(frame):
        raise AssertionError("P1.3 edge row count mismatch")
    layout = build_master_features(h_t, h_v,
                                   torch.from_numpy(frame.src.to_numpy(dtype=np.int64).copy()),
                                   torch.from_numpy(frame.dst.to_numpy(dtype=np.int64).copy()),
                                   torch.from_numpy(degree),
                                   data.edge_index,)
    expected_nodes = data.num_nodes
    if h_t.shape != (expected_nodes, 128) or h_v.shape != (expected_nodes, 128):
        raise AssertionError("P1.3 H0 dimensions/regression invalid")
    return data, h_t.detach().cpu(), h_v.detach().cpu(), layout, h0_record, positions, graph_src, graph_dst, degree


def extract_frozen_e01_representations(dataset: str, seed: int, data: ProbeData,
                                       frame: pd.DataFrame, device: torch.device,
                                       edge_positions: np.ndarray, graph_src: np.ndarray,
                                       graph_dst: np.ndarray, degree: np.ndarray,
                                       checkpoint_root: Path = Path("outputs/e01_function_provenance_preservation/checkpoints"),
                                       chunk_size: int = 32768) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    checkpoint_path = checkpoint_root / dataset / f"seed_{seed}_keep_edge.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "keep_edge" or checkpoint.get("evaluate_test", True):
        raise AssertionError("expected frozen validation-only E0.1 KeepEdge checkpoint")
    info = checkpoint["data_info"]
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.load("configs/model/provenance_executor_e01.yaml")
    cfg = OmegaConf.create({"model": OmegaConf.to_container(model_cfg, resolve=True)})
    cfg.model.variant = "keep_edge"
    model = E01Model(cfg, info).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    x = torch.cat([data.x_t, data.x_v], dim=-1).to(device)
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        h0, _, src, dst, degree_t, deg_z, p, contexts = model._prepare(x, edge_index)
        pos = torch.as_tensor(edge_positions, dtype=torch.long, device=device)
        src_expected, dst_expected = src[pos], dst[pos]
        degree_expected = degree_t[dst_expected]
        expected = np.stack([src_expected.cpu().numpy(), dst_expected.cpu().numpy(),
                             degree_expected.cpu().numpy()], axis=1)
        table_keys = frame[list(KEY_COLUMNS)].to_numpy(dtype=np.int64)
        if not np.array_equal(expected, table_keys):
            raise AssertionError("frozen E0.1 state rows are not exactly aligned to P1.3 order")
        collected: dict[str, list[torch.Tensor]] = {k: [] for k in ("q_t", "q_v", "r", "u_t", "u_v")}
        for begin in range(0, len(pos), chunk_size):
            ppos = pos[begin:begin + chunk_size]
            q_t, q_v, r, u = model._functional_states(
                p, [contexts[0][ppos], contexts[1][ppos]], deg_z,
                src[ppos], dst[ppos],
            )
            for name, value in zip(("q_t", "q_v", "r", "u_t", "u_v"),
                                   (q_t, q_v, r, u[0], u[1])):
                collected[name].append(value.detach().cpu())
        raw = {name: torch.cat(parts, dim=0).numpy().astype(np.float32)
               for name, parts in collected.items()}
        for name in raw:
            if raw[name].shape[0] != len(frame) or not np.isfinite(raw[name]).all():
                raise AssertionError(f"invalid frozen representation {name}")
        reps: dict[str, np.ndarray] = {}
        reps["Q_PAIR_text"] = np.concatenate([raw["q_t"], raw["q_v"]], axis=1)
        reps["Q_PAIR_visual"] = np.concatenate([raw["q_v"], raw["q_t"]], axis=1)
        reps["R_SHARED_text"] = raw["r"]
        reps["R_SHARED_visual"] = raw["r"]
        reps["U_MODAL_text"] = raw["u_t"]
        reps["U_MODAL_visual"] = raw["u_v"]
        if any(value.shape[1] != 64 for value in reps.values()):
            raise AssertionError("all frozen Q/R/U representations must be 64D")
        if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("frozen E0.1 parameters changed trainability during extraction")
    record = {"checkpoint": str(checkpoint_path), "checkpoint_frozen": True,
              "aligned_ordered_src_dst_degree": True, "representation_dimensions":
              {key: value.shape[1] for key, value in reps.items()},
              "no_test_access": True}
    del model, x, edge_index, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reps, record
