from __future__ import annotations

import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (balanced_accuracy_score, f1_score, r2_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit

from src.analysis import l0_relation_function_learnability as l0
from src.analysis.p0p1_propagation_probe import DATASETS, ProbeData, load_probe_data
from src.analysis.p11_p12_operator_rescue import load_frozen_h0
from src.analysis.p13_joint_readout_operator_probe import operator_context


SEEDS = (42, 43, 44)
OPERATORS = ("smooth", "absdiff", "product")
OP_SHORT = {"smooth": "S", "absdiff": "D", "product": "P"}
MODALITIES = ("text", "visual")
UTILITY_COLUMNS = ("G_D_text", "G_P_text", "G_D_visual", "G_P_visual")
MASTER_VARIANTS = ("SIM_ONLY", "TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL")
NULL_SEEDS = (9917, 9918, 9919)
DATA_ROOT = "/hdd1/DataInHere/YHF/data"
P13_RUN_ROOT = Path("outputs/p13_joint_readout_operator_probe/runs")
P0_CHECKPOINT_ROOT = Path("outputs/p0p1_propagation_heterogeneity/checkpoints/runs")
P0_RUN_ROOT = Path("outputs/p0p1_propagation_heterogeneity/runs")
E01_CHECKPOINT_ROOT = Path("outputs/e01_function_provenance_preservation/checkpoints")
L0_FOLD_PATH = Path("research/l0_relation_function_learnability_audit/data/group_fold_assignment.csv")
OUT_DIR = Path("research/l01_shared_slot_function_identifiability")
RAW_DIR = Path("outputs/l01_shared_slot_function_identifiability")
SHARED_DIM = 512
SLOT_TEXT = slice(256, 384)
SLOT_VISUAL = slice(384, 512)
TARGET_INDEX = {name: i for i, name in enumerate(UTILITY_COLUMNS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rho(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    z = spearmanr(x, y).statistic
    return float(z) if np.isfinite(z) else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    x0, y0 = x - x.mean(), y - y.mean()
    return float((x0 @ y0) / max(np.linalg.norm(x0) * np.linalg.norm(y0), 1e-15))


def centered_rho(x: np.ndarray, y: np.ndarray, dst: np.ndarray) -> float:
    dx, dy = np.asarray(x, dtype=np.float64).copy(), np.asarray(y, dtype=np.float64).copy()
    for value in np.unique(dst):
        m = dst == value
        dx[m] -= dx[m].mean()
        dy[m] -= dy[m].mean()
    return rho(dx, dy)


def stratified_head_split(train_idx: np.ndarray, labels: np.ndarray, seed: int,
                          select_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split only original NC train nodes; deterministic stratified split with audited fallback."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[train_idx]
    if len(train_idx) < 2:
        raise ValueError("at least two original training nodes are required")
    counts = np.bincount(y)
    can_stratify = bool(len(counts) and counts.min() >= 2 and
                        round(len(train_idx) * select_fraction) >= len(counts) and
                        len(train_idx) - round(len(train_idx) * select_fraction) >= len(counts))
    fallback_reason = None
    if can_stratify:
        try:
            split = StratifiedShuffleSplit(n_splits=1, test_size=select_fraction, random_state=seed)
            a, b = next(split.split(train_idx, y))
            head_train, head_select = train_idx[a], train_idx[b]
            mode = "stratified"
        except ValueError as exc:
            fallback_reason = str(exc)
            can_stratify = False
    if not can_stratify:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train_idx)
        n_select = min(max(1, int(round(len(train_idx) * select_fraction))), len(train_idx) - 1)
        head_select, head_train = perm[:n_select], perm[n_select:]
        mode = "deterministic_unstratified_fallback"
    if set(head_train) & set(head_select) or set(head_train) | set(head_select) != set(train_idx):
        raise AssertionError("head split is not an exact disjoint partition of original train_idx")
    tr_classes = np.unique(np.asarray(labels)[head_train]).tolist()
    sel_classes = np.unique(np.asarray(labels)[head_select]).tolist()
    audit = {"split_mode": mode, "split_seed": int(seed), "original_train_count": len(train_idx),
             "head_train_count": len(head_train), "head_select_count": len(head_select),
             "class_counts_original": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
             "classes_missing_head_train": sorted(set(np.unique(y).tolist()) - set(tr_classes)),
             "classes_missing_head_select": sorted(set(np.unique(y).tolist()) - set(sel_classes)),
             "head_train_is_subset_train_idx": bool(set(head_train) <= set(train_idx)),
             "head_select_is_subset_train_idx": bool(set(head_select) <= set(train_idx)),
             "validation_nodes_excluded": True, "fallback_reason": fallback_reason}
    return np.sort(head_train), np.sort(head_select), audit


class SharedSlotHead(nn.Module):
    """Exactly one shared Linear(512,C); the four blocks never get operator-specific parameters."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(SHARED_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@dataclass
class SharedData:
    data: ProbeData
    seed: int
    h_t: torch.Tensor
    h_v: torch.Tensor
    contexts_t: dict[str, torch.Tensor]
    contexts_v: dict[str, torch.Tensor]
    degree: torch.Tensor
    h0_audit: dict[str, Any]
    p13_frame: pd.DataFrame
    p13_raw: pd.DataFrame
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_degree: np.ndarray
    edge_positions: np.ndarray
    graph_src: np.ndarray
    graph_dst: np.ndarray
    master: l0.FeatureLayout
    p13_alignment: dict[str, Any]


def load_shared_data(dataset: str, seed: int, device: torch.device) -> SharedData:
    if bool(device.type == "cuda" and device.index is None):
        raise ValueError("CUDA device must include an explicit index")
    p13_path = P13_RUN_ROOT / dataset / f"seed_{seed}" / "joint_operator_edge_utility.csv.gz"
    if not p13_path.is_file():
        raise FileNotFoundError(p13_path)
    p13_raw = pd.read_csv(p13_path)
    keys = ["src", "dst", "dst_degree"]
    required = [*keys, *[f"u_raw_{op}_{mod}" for mod in MODALITIES for op in OPERATORS]]
    missing = sorted(set(required) - set(p13_raw.columns))
    if missing:
        raise AssertionError(f"P1.3 per-edge output lacks required columns: {missing}")
    p13_raw = p13_raw[required].copy()
    p13_frame = p13_raw[keys].astype("int64").copy()
    data = load_probe_data(dataset, seed, DATA_ROOT)
    checkpoint = P0_CHECKPOINT_ROOT / dataset / f"seed_{seed}_semantic_only.pt"
    reference = P0_RUN_ROOT / dataset / f"seed_{seed}" / "probe_performance.csv"
    model, h_t, h_v, h0_audit = load_frozen_h0(data, checkpoint, device, reference)
    if h_t.shape != (data.num_nodes, 128) or h_v.shape != (data.num_nodes, 128):
        raise AssertionError(f"frozen H0 must be [nodes,128], got {tuple(h_t.shape)} / {tuple(h_v.shape)}")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("frozen P0 H0 network acquired trainable parameters")
    graph = data.edge_index.to(device)
    src_all, dst_all = graph
    if torch.any(src_all == dst_all):
        raise AssertionError("physical graph must have self messages removed")
    degree = torch.bincount(dst_all, minlength=data.num_nodes)
    if not torch.equal(degree.cpu(), torch.bincount(data.edge_index[1], minlength=data.num_nodes)):
        raise AssertionError("degree must be original physical incoming degree")
    contexts: list[dict[str, torch.Tensor]] = []
    for h in (h_t, h_v):
        current: dict[str, torch.Tensor] = {}
        for op in OPERATORS:
            ctx, op_degree = operator_context(h, graph, data.num_nodes, op)
            if not torch.equal(op_degree, degree):
                raise AssertionError(f"{op} context recomputed a different physical degree")
            current[op] = ctx
        contexts.append(current)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask[data.val_idx.to(device)] = True
    keep = val_mask[dst_all]
    positions_t = torch.where(keep)[0]
    edge_src_t, edge_dst_t = src_all[keep], dst_all[keep]
    edge_deg_t = degree[edge_dst_t]
    edge_keys = np.stack([edge_src_t.cpu().numpy(), edge_dst_t.cpu().numpy(), edge_deg_t.cpu().numpy()], axis=1)
    expected = p13_frame[keys].to_numpy(dtype=np.int64)
    if edge_keys.shape != expected.shape or not np.array_equal(edge_keys, expected):
        raise AssertionError(f"{dataset}/{seed}: L0.1 validation edge rows do not exactly align to P1.3")
    p13_alignment = {"rows": len(expected), "edge_order_match": True,
                     "src_dst_degree_match": True, "self_loops": False}
    master = l0.build_master_features(h_t, h_v, torch.from_numpy(expected[:, 0].copy()),
                                      torch.from_numpy(expected[:, 1].copy()),
                                      torch.bincount(data.edge_index[1], minlength=data.num_nodes),
                                      data.edge_index)
    del model
    return SharedData(data, seed, h_t.detach(), h_v.detach(), contexts[0], contexts[1], degree,
                      h0_audit, p13_frame, p13_raw, expected[:, 0].copy(), expected[:, 1].copy(),
                      expected[:, 2].copy(), positions_t.cpu().numpy().astype(np.int64),
                      src_all.cpu().numpy().astype(np.int64), dst_all.cpu().numpy().astype(np.int64),
                      master, p13_alignment)


def shared_features(shared: SharedData, op_t: str, op_v: str, nodes: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
    if op_t not in OPERATORS or op_v not in OPERATORS:
        raise ValueError("operator combination outside fixed S/D/P basis")
    if nodes is None:
        nodes = torch.arange(shared.data.num_nodes, device=shared.h_t.device)
    elif not torch.is_tensor(nodes):
        nodes = torch.as_tensor(nodes, dtype=torch.long, device=shared.h_t.device)
    else:
        nodes = nodes.to(shared.h_t.device)
    x = torch.cat([shared.h_t[nodes], shared.h_v[nodes],
                   shared.contexts_t[op_t][nodes], shared.contexts_v[op_v][nodes]], dim=-1)
    if x.size(-1) != SHARED_DIM:
        raise AssertionError("shared feature dimension must be exactly 512")
    return x


def train_shared_head(shared: SharedData, dataset_index: int, seed: int, repeat: int,
                      device: torch.device, max_epochs: int = 300) -> tuple[SharedSlotHead, dict[str, Any]]:
    split_seed = 17011 + dataset_index * 1000 + seed * 17 + repeat
    labels = shared.data.labels.numpy()
    head_train, head_select, split_audit = stratified_head_split(
        shared.data.train_idx.numpy(), labels, split_seed)
    if set(head_train) & set(shared.data.val_idx.tolist()) or set(head_select) & set(shared.data.val_idx.tolist()):
        raise AssertionError("validation targets leaked into shared-head fit split")
    set_seed(29003 + dataset_index * 101 + seed * 7 + repeat)
    head = SharedSlotHead(shared.data.num_classes).to(device)
    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    if len(linears) != 1 or linears[0].in_features != 512:
        raise AssertionError("shared predictor must expose exactly one Linear(512,C)")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    y = shared.data.labels.to(device)
    tr = torch.as_tensor(head_train, dtype=torch.long, device=device)
    sel = torch.as_tensor(head_select, dtype=torch.long, device=device)
    combinations = tuple(itertools.product(OPERATORS, repeat=2))
    best = float("inf")
    best_state, best_epoch, wait = None, 0, 0
    max_epochs = min(int(max_epochs), 300)
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        for op_t, op_v in combinations:
            x = shared_features(shared, op_t, op_v, tr)
            loss = F.cross_entropy(head(x), y[tr]) / len(combinations)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        head.eval()
        with torch.no_grad():
            losses = [F.cross_entropy(head(shared_features(shared, a, b, sel)), y[sel])
                      for a, b in combinations]
            val_loss = float(torch.stack(losses).mean().item())
        if val_loss < best:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        elif epoch >= 30:
            wait += 1
            if wait >= 30:
                break
    if best_state is None:
        raise RuntimeError("shared head did not select a head_select checkpoint")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    with torch.no_grad():
        combo_metrics = []
        for op_t, op_v in combinations:
            logits = head(shared_features(shared, op_t, op_v, sel))
            combo_metrics.append({"ce": float(F.cross_entropy(logits, y[sel]).item()),
                                  "acc": float((logits.argmax(-1) == y[sel]).float().mean().item())})
        smooth_logits = head(shared_features(shared, "smooth", "smooth", sel))
        smooth_pred = smooth_logits.argmax(-1).cpu().numpy()
        true = y[sel].cpu().numpy()
        f1 = f1_score(true, smooth_pred, labels=list(range(shared.data.num_classes)),
                      average="macro", zero_division=0)
    norms = {
        "weight_norm_H0_T": float(torch.linalg.vector_norm(head.classifier.weight[:, :128]).item()),
        "weight_norm_H0_V": float(torch.linalg.vector_norm(head.classifier.weight[:, 128:256]).item()),
        "weight_norm_struct_T": float(torch.linalg.vector_norm(head.classifier.weight[:, SLOT_TEXT]).item()),
        "weight_norm_struct_V": float(torch.linalg.vector_norm(head.classifier.weight[:, SLOT_VISUAL]).item()),
    }
    diag = {"dataset": shared.data.name, "seed": seed, "repeat": repeat,
            "best_epoch": best_epoch, "epochs_run": epoch,
            "head_select_mean_ce_9": best,
            "head_select_smooth_smooth_ce": float(F.cross_entropy(smooth_logits, y[sel]).item()),
            "head_select_smooth_smooth_accuracy": float((smooth_logits.argmax(-1) == y[sel]).float().mean().item()),
            "head_select_smooth_smooth_macro_f1": float(f1),
            "head_select_mean_accuracy_9_descriptive": float(np.mean([x["acc"] for x in combo_metrics])),
            "operator_combination_count": 9, "feature_dim": 512,
            "trainable_linear_count": 1, "validation_labels_used_for_head": False,
            "head_train_count": len(head_train), "head_select_count": len(head_select), **norms}
    split_audit = {"dataset": shared.data.name, "seed": seed, "repeat": repeat, **split_audit}
    return head, {"performance": diag, "split_audit": split_audit,
                  "head_train": head_train, "head_select": head_select}


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p, q = p.double().clamp_min(eps), q.double().clamp_min(eps)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    m = (p + q) / 2
    return 0.5 * ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1))


def substitute_edge_logits(head: nn.Linear, base_feature: torch.Tensor,
                           delta_context: torch.Tensor, modality: str,
                           brute: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return altered feature and logits, changing only one structural slot."""
    block = SLOT_TEXT if modality == "text" else SLOT_VISUAL if modality == "visual" else None
    if block is None or delta_context.shape[-1] != 128:
        raise ValueError("substitution must target a 128D Text or Visual structural slot")
    changed = base_feature.clone()
    changed[..., block] = changed[..., block] + delta_context
    if modality == "text" and not torch.equal(changed[..., SLOT_VISUAL], base_feature[..., SLOT_VISUAL]):
        raise AssertionError("Text replacement altered Visual slot")
    if modality == "visual" and not torch.equal(changed[..., SLOT_TEXT], base_feature[..., SLOT_TEXT]):
        raise AssertionError("Visual replacement altered Text slot")
    logits = F.linear(changed, head.weight, head.bias)
    if not brute:
        # Algebraic logit delta is independently checked against the edited-feature pass.
        weight_slice = head.weight[:, block]
        fast = F.linear(base_feature, head.weight, head.bias) + delta_context @ weight_slice.T
        if not torch.allclose(logits, fast, rtol=1e-7, atol=1e-8):
            raise AssertionError("fast logit update disagrees with feature replacement")
    return changed, logits


@torch.no_grad()
def build_repeat_utilities(shared: SharedData, head: SharedSlotHead, repeat: int,
                           device: torch.device, check_all: bool = False) -> tuple[pd.DataFrame, dict[str, float]]:
    src = torch.as_tensor(shared.edge_src, dtype=torch.long, device=device)
    dst = torch.as_tensor(shared.edge_dst, dtype=torch.long, device=device)
    degree = torch.as_tensor(shared.edge_degree, dtype=torch.float64, device=device)
    base = shared_features(shared, "smooth", "smooth", dst)
    base_logits = F.linear(base.double(), head.classifier.weight.double(), head.classifier.bias.double())
    labels = shared.data.labels.to(device)[dst]
    base_ce = F.cross_entropy(base_logits, labels, reduction="none")
    base_prob = base_logits.softmax(-1)
    record: dict[str, Any] = {"src": shared.edge_src, "dst": shared.edge_dst,
                              "dst_degree": shared.edge_degree, "head_repeat": repeat,
                              "CE_base": base_ce.cpu().numpy()}
    errors: dict[str, float] = {}
    for modality, h, smooth_context, alternatives in (
        ("text", shared.h_t, shared.contexts_t["smooth"], ("absdiff", "product")),
        ("visual", shared.h_v, shared.contexts_v["smooth"], ("absdiff", "product")),
    ):
        for alt in alternatives:
            message_s = h[src]
            if alt == "absdiff":
                message_alt = (h[dst] - h[src]).abs()
            else:
                message_alt = h[dst] * h[src]
            delta = (message_alt - message_s).double() / degree[:, None]
            feature_alt = base.double().clone()
            slot = SLOT_TEXT if modality == "text" else SLOT_VISUAL
            feature_alt[:, slot] += delta
            brute_logits = F.linear(feature_alt, head.classifier.weight.double(),
                                    head.classifier.bias.double())
            fast_logits = base_logits + delta @ head.classifier.weight[:, slot].double().T
            max_error = float((brute_logits - fast_logits).abs().max().item())
            key = f"{alt}_{modality}_logit_max_abs"
            errors[key] = max_error
            if not torch.allclose(brute_logits, fast_logits, rtol=1e-7, atol=1e-8):
                raise AssertionError(f"{key}: fast vs brute tolerance failed ({max_error})")
            labels_sub = labels
            ce_sub = F.cross_entropy(brute_logits, labels_sub, reduction="none")
            prob_sub = brute_logits.softmax(-1)
            gain = base_ce - ce_sub
            gain_name = f"G_{'D' if alt == 'absdiff' else 'P'}_{modality}"
            shift = torch.linalg.vector_norm(brute_logits - base_logits, dim=-1)
            prob_l1 = (prob_sub - base_prob).abs().sum(-1)
            js = js_divergence(base_prob, prob_sub)
            flip = brute_logits.argmax(-1) != base_logits.argmax(-1)
            record[gain_name] = gain.cpu().numpy()
            record[f"logit_shift_{gain_name}"] = shift.cpu().numpy()
            record[f"probability_l1_{gain_name}"] = prob_l1.cpu().numpy()
            record[f"js_{gain_name}"] = js.cpu().numpy()
            record[f"prediction_flip_{gain_name}"] = flip.cpu().numpy()
            record[f"absolute_ce_change_{gain_name}"] = (ce_sub - base_ce).abs().cpu().numpy()
    table = pd.DataFrame(record)
    if tuple(k for k in table.columns if k.startswith("G_")) != UTILITY_COLUMNS:
        raise AssertionError("the fixed four-target order must be G_D^T,G_P^T,G_D^V,G_P^V")
    return table, errors


def aggregate_repeats(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["src", "dst", "dst_degree"]
    rows = raw.groupby(keys, sort=False, as_index=False).first()[keys]
    for target in UTILITY_COLUMNS:
        g = raw.groupby(keys, sort=False)[target]
        stats = g.agg(["mean", "std", lambda x: float((x > 0).mean())]).reset_index()
        stats.columns = [*keys, f"mean_{target}", f"std_{target}", f"positive_fraction_{target}"]
        rows = rows.merge(stats, on=keys, validate="one_to_one")
        signs = raw.assign(_sign=np.sign(raw[target])).groupby(keys, sort=False)["_sign"].agg(
            lambda x: bool(len(set(x.tolist())) == 1)).reset_index(name=f"sign_consistent_{target}")
        rows = rows.merge(signs, on=keys, validate="one_to_one")
    return rows


def metric_bundle(y: np.ndarray, pred: np.ndarray, dst: np.ndarray,
                  degree: np.ndarray) -> dict[str, float]:
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    binary = (y > 0).astype(np.int8)
    auc = float("nan") if len(np.unique(binary)) < 2 else float(roc_auc_score(binary, pred))
    residual_y, residual_pred = y.copy(), pred.copy()
    target_ranks = []
    eligible = 0
    for target in np.unique(dst):
        m = dst == target
        residual_y[m] -= y[m].mean()
        residual_pred[m] -= pred[m].mean()
        if len(y[m]) >= 2 and np.max(degree[m]) >= 5:
            eligible += 1
            value = rho(y[m], pred[m])
            if np.isfinite(value):
                target_ranks.append(value)
    return {"n_edges": len(y), "spearman": rho(y, pred), "pearson": pearson(y, pred),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
            "sign_auroc": auc,
            "sign_balanced_accuracy": float(balanced_accuracy_score(binary, pred > 0)),
            "within_target_residual_spearman": rho(residual_y, residual_pred),
            "degree5_target_count": eligible,
            "degree5_target_rank_mean": float(np.mean(target_ranks)) if target_ranks else float("nan"),
            "degree5_target_rank_median": float(np.median(target_ranks)) if target_ranks else float("nan")}


def folds_for_edges(dataset: str, seed: int, dst: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, bool]:
    prior = pd.read_csv(L0_FOLD_PATH)
    selected = prior[(prior.dataset == dataset) & (prior.seed.astype(int) == int(seed))]
    support = set(np.unique(dst).astype(int).tolist())
    if set(selected.dst.astype(int)) == support and not selected.dst.duplicated().any():
        lookup = dict(zip(selected.dst.astype(int), selected.fold.astype(int)))
        assignments = np.asarray([lookup[int(v)] for v in dst], dtype=np.int64)
        return assignments, selected, True
    groups = l0.balanced_group_folds(dst, 3, seed=42 + seed)
    lookup = dict(zip(groups.dst.astype(int), groups.fold.astype(int)))
    assignments = np.asarray([lookup[int(v)] for v in dst], dtype=np.int64)
    return assignments, groups, False


def inner_split(dst: np.ndarray, train_rows: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return l0.inner_group_split(dst, train_rows, seed, val_fraction=0.10)


def shuffled_within_dst(y: np.ndarray, dst: np.ndarray, rows: np.ndarray,
                        seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    shuffled, audit = l0.shuffle_targets_within_destination(y, dst, rows, seed)
    if not audit["multiset_preserved"]:
        raise AssertionError("within-destination utility tuple multiset was not preserved")
    if not np.array_equal(np.sort(shuffled[rows], axis=0), np.sort(y[rows], axis=0)):
        # Column sorting is a conservative secondary invariant; tuple checks below are authoritative.
        raise AssertionError("shuffled utility component marginals changed")
    return shuffled, audit


def edge_weights(degree: np.ndarray) -> np.ndarray:
    return l0.target_balanced_weights(degree)


def scale_targets_for_fit(y: np.ndarray, mean: np.ndarray, std: np.ndarray,
                          rows: np.ndarray) -> np.ndarray:
    """Scale only fit/checkpoint-selection rows; outer-test target cells remain untouched zeros."""
    result = np.zeros(np.asarray(y).shape, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64)
    result[rows] = ((np.asarray(y)[rows] - mean) / std).astype(np.float32)
    return result


def fit_evidence_fold(master: torch.Tensor, masks: dict[str, torch.Tensor], y: np.ndarray,
                      dst: np.ndarray, degree: np.ndarray, train: np.ndarray, val: np.ndarray,
                      test: np.ndarray, variant: str, device: torch.device,
                      seed: int, null_shuffle_seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    y_fit = np.asarray(y, dtype=np.float64).copy()
    shuffle_audit: dict[str, Any] = {"enabled": null_shuffle_seed is not None}
    if null_shuffle_seed is not None:
        y_fit, tr_audit = shuffled_within_dst(y_fit, dst, train, null_shuffle_seed)
        # Use a separate deterministic RNG stream for inner validation correspondence destruction.
        y_fit, va_audit = shuffled_within_dst(y_fit, dst, val, null_shuffle_seed + 1000003)
        shuffle_audit.update({"inner_train": tr_audit, "inner_validation": va_audit,
                              "outer_test_unshuffled": True})
    y_mean, y_std = l0.standardize_targets(y_fit, train)
    fit_rows = np.unique(np.concatenate([train, val]))
    y_scaled = scale_targets_for_fit(y_fit, y_mean, y_std, fit_rows)
    mean, std = l0.feature_stats(master, train)
    x = ((master - mean) / std).contiguous()
    mask = masks[variant]
    x[:, ~mask] = 0
    if len(mean) != 1542 or not torch.isfinite(x).all():
        raise AssertionError("L0 master feature scaler must yield a finite 1542D representation")
    set_seed(seed)
    model = l0.EvidenceMLP(output_dim=4)
    x_device = x.to(device)
    fit = l0.fit_fullbatch_probe(model, lambda: model(x_device), y_scaled, train, val, test,
                                 edge_weights(degree), device)
    pred = fit["prediction_scaled"] * y_std + y_mean
    metadata = {k: fit[k] for k in ("best_epoch", "epochs_run", "best_inner_weighted_huber",
                                     "finite_gradients", "parameter_count")}
    metadata.update({"variant": variant, "feature_scaler_rows": len(train),
                     "target_scaler_rows": len(train), "feature_scaler_fit_rows_exact_inner_train": True,
                     "target_scaler_fit_rows_exact_inner_train": True,
                     "outer_test_targets_passed_to_trainer": False,
                     "feature_mask_kept": int(mask.sum()), "shuffle_audit": shuffle_audit})
    return pred, metadata


def fit_ridge_fold(master: torch.Tensor, masks: dict[str, torch.Tensor], y: np.ndarray,
                   dst: np.ndarray, degree: np.ndarray, train: np.ndarray, val: np.ndarray,
                   test: np.ndarray, variant: str) -> tuple[np.ndarray, dict[str, Any]]:
    del val  # Ridge has no checkpoint selection; inner validation stays outside all fitting/statistics.
    mean, std = l0.feature_stats(master, train)
    x = ((master - mean) / std).numpy()
    x[:, ~masks[variant].numpy()] = 0
    y_mean, y_std = l0.standardize_targets(y, train)
    ys = scale_targets_for_fit(y, y_mean, y_std, train)
    weights = edge_weights(degree)
    model = Ridge(alpha=1.0)
    model.fit(x[train], ys[train], sample_weight=weights[train])
    pred = model.predict(x[test]) * y_std + y_mean
    return np.asarray(pred), {"alpha": 1.0, "feature_scaler_rows": len(train),
                              "target_scaler_rows": len(train), "sample_weight": "normalized 1/d_i",
                              "variant": variant}


def direct_state_model() -> l0.M0StateDirectProbe:
    model = l0.M0StateDirectProbe()
    # Keep the exact M0 P/q/r/u path and use the required shared two-output readout per modality.
    model.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
    return model


def fit_direct_fold(shared: SharedData, y: np.ndarray, dst: np.ndarray, degree_by_edge: np.ndarray,
                    train: np.ndarray, val: np.ndarray, test: np.ndarray,
                    edge_positions: np.ndarray, graph_src: np.ndarray, graph_dst: np.ndarray,
                    device: torch.device, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    # Target vector is D_T,P_T,D_V,P_V. M0's shared head is called separately on u_T/u_V.
    y_mean, y_std = l0.standardize_targets(y, train)
    y_scaled = scale_targets_for_fit(y, y_mean, std=y_std, rows=np.unique(np.concatenate([train, val])))
    h_t, h_v = shared.h_t.to(device), shared.h_v.to(device)
    positions = torch.as_tensor(edge_positions, dtype=torch.long, device=device)
    gs = torch.as_tensor(graph_src, dtype=torch.long, device=device)
    gd = torch.as_tensor(graph_dst, dtype=torch.long, device=device)
    deg = shared.degree.to(device)
    deg_z = l0.graph_standardized_log_degree(deg).to(device)
    model = direct_state_model()
    def forward() -> torch.Tensor:
        out, _ = model.forward_from_edge_positions(h_t, h_v, positions, gs, gd, deg, deg_z)
        # M0 concatenates [D_T,P_T,D_V,P_V] already.
        return out
    fit = l0.fit_fullbatch_probe(model, forward, y_scaled, train, val, test,
                                 edge_weights(degree_by_edge), device)
    pred = fit["prediction_scaled"] * y_std + y_mean
    metadata = {k: fit[k] for k in ("best_epoch", "epochs_run", "best_inner_weighted_huber",
                                     "finite_gradients", "parameter_count")}
    metadata.update({"input_h0_frozen": True, "q_dim": 32, "r_dim": 64, "u_dim": 64,
                     "readout_output_dim_per_modality": 2, "target_scaler_rows": len(train),
                     "target_scaler_fit_rows_exact_inner_train": True,
                     "outer_test_targets_passed_to_trainer": False})
    del fit["model"], h_t, h_v
    return pred, metadata


def fit_frozen_fold(representation: np.ndarray, y: np.ndarray, dst: np.ndarray, degree: np.ndarray,
                    train: np.ndarray, val: np.ndarray, test: np.ndarray,
                    modality: str, device: torch.device, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    modality_i = MODALITIES.index(modality)
    columns = slice(modality_i * 2, modality_i * 2 + 2)
    y_mod = y[:, columns]
    y_mean, y_std = l0.standardize_targets(y_mod, train)
    y_scaled = scale_targets_for_fit(y_mod, y_mean, y_std, np.unique(np.concatenate([train, val])))
    x_tensor = torch.as_tensor(representation, dtype=torch.float32)
    x_mean, x_std = l0.feature_stats(x_tensor, train)
    x = ((x_tensor - x_mean) / x_std).contiguous()
    set_seed(seed)
    model = l0.StateReadout(64, 2)
    xd = x.to(device)
    fit = l0.fit_fullbatch_probe(model, lambda: model(xd), y_scaled, train, val, test,
                                 edge_weights(degree), device)
    pred = fit["prediction_scaled"] * y_std + y_mean
    metadata = {k: fit[k] for k in ("best_epoch", "epochs_run", "best_inner_weighted_huber",
                                     "finite_gradients", "parameter_count")}
    metadata.update({"feature_scaler_rows": len(train), "target_scaler_rows": len(train),
                     "feature_scaler_fit_rows_exact_inner_train": True,
                     "target_scaler_fit_rows_exact_inner_train": True,
                     "outer_test_targets_passed_to_trainer": False,
                     "frozen_representation_dim": 64})
    del fit["model"]
    return pred, metadata


def fold_rows(assignments: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(assignments == fold)
    train = np.flatnonzero(assignments != fold)
    if set(np.unique(assignments[test])) & set(np.unique(assignments[train])):
        raise AssertionError("outer train/test target groups overlap")
    return train, test


def full_metric_rows(y: np.ndarray, prediction: np.ndarray, test: np.ndarray,
                     dst: np.ndarray, degree: np.ndarray, dataset: str, seed: int,
                     fold: int, model: str) -> list[dict[str, Any]]:
    if prediction.shape != (len(test), 4):
        raise AssertionError(f"expected four ordered gain outputs, got {prediction.shape}")
    rows = []
    for col, target_name in enumerate(UTILITY_COLUMNS):
        modality = "text" if target_name.endswith("text") else "visual"
        operator = "absdiff" if "G_D" in target_name else "product"
        metric = metric_bundle(y[test, col], prediction[:, col], dst[test], degree[test])
        rows.append({"dataset": dataset, "seed": seed, "fold": fold, "model": model,
                     "target": target_name, "modality": modality, "operator": operator, **metric})
    return rows


def secondary_eval_rows(y: np.ndarray, prediction: np.ndarray, test: np.ndarray,
                        clean_edges: pd.DataFrame, raw_repeats: pd.DataFrame,
                        dst: np.ndarray, degree: np.ndarray,
                        dataset: str, seed: int, fold: int, model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stable_rows, effect_rows = [], []
    means = aggregate_repeats(raw_repeats).set_index(["src", "dst"])
    test_keys = pd.MultiIndex.from_arrays([clean_edges.iloc[test].src, clean_edges.iloc[test].dst])
    for col, target_name in enumerate(UTILITY_COLUMNS):
        sign_col = f"sign_consistent_{target_name}"
        stable = means.loc[test_keys, sign_col].to_numpy(dtype=bool)
        y_test, p_test = y[test, col], prediction[:, col]
        stable_metric = metric_bundle(y_test[stable], p_test[stable], dst[test][stable], degree[test][stable]) if stable.any() else {}
        stable_rows.append({"dataset": dataset, "seed": seed, "fold": fold, "model": model,
                            "target": target_name, "subset": "head_sign_consistent",
                            "subset_count": int(stable.sum()), "full_count": len(test), **stable_metric})
        threshold = float(np.quantile(np.abs(y_test), 0.75))
        high = np.abs(y_test) >= threshold
        high_metric = metric_bundle(y_test[high], p_test[high], dst[test][high], degree[test][high]) if high.any() else {}
        binary = (y_test[high] > 0).astype(np.int8)
        auc = float("nan") if len(np.unique(binary)) < 2 else float(roc_auc_score(binary, p_test[high]))
        effect_rows.append({"dataset": dataset, "seed": seed, "fold": fold, "model": model,
                            "target": target_name, "abs_gain_q75_threshold": threshold,
                            "subset_count": int(high.sum()), "full_count": len(test),
                            "high_effect_sign_auroc": auc, **high_metric})
    return stable_rows, effect_rows


def run_probe_suite(shared: SharedData, clean: pd.DataFrame, device: torch.device,
                    raw_repeats: pd.DataFrame, smoke: bool = False,
                    smoke_dir: Path | None = None) -> dict[str, Any]:
    """Run target-disjoint evidence, strict null, Ridge, direct state and frozen Q/R/U probes."""
    dataset, seed = shared.data.name, shared.seed
    y = clean[[f"mean_{col}" for col in UTILITY_COLUMNS]].to_numpy(dtype=np.float64)
    if y.shape[1] != 4 or not np.isfinite(y).all():
        raise AssertionError("clean target must be finite [edges,4]")
    dst = shared.edge_dst.astype(np.int64)
    degree = shared.edge_degree.astype(np.float64)
    assignments, assignment_table, reused = folds_for_edges(dataset, seed, dst)
    if smoke:
        folds = [int(np.unique(assignments)[0])]
    else:
        folds = sorted(np.unique(assignments).astype(int).tolist())
    # The edge positions and graph arrays are the exact ordered P1.3 support.
    from src.analysis.l0_relation_function_learnability import extract_frozen_e01_representations
    representations, frozen_alignment = extract_frozen_e01_representations(
        dataset, seed, shared.data, shared.p13_frame, device, shared.edge_positions,
        shared.graph_src, shared.graph_dst, shared.degree.cpu().numpy())
    by_fold: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    stable_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    for fold in folds:
        outer_train, test = fold_rows(assignments, fold)
        inner_train, inner_val = inner_split(dst, outer_train, seed=60119 + seed * 17 + fold)
        if set(dst[inner_train]) & set(dst[inner_val]) or set(dst[outer_train]) & set(dst[test]):
            raise AssertionError("dst-disjoint outer/inner split invariant failed")
        fold_audits.append({"fold": fold, "outer_train_edges": len(outer_train), "outer_test_edges": len(test),
                            "inner_train_edges": len(inner_train), "inner_val_edges": len(inner_val),
                            "outer_dst_disjoint": True, "inner_dst_disjoint": True,
                            "fold_assignment_reused_from_L0": reused,
                            "outer_test_utility_untouched_during_fit": True})
        init_seed = 31009 + DATASETS.index(dataset) * 1009 + seed * 13 + fold * 7
        for variant in MASTER_VARIANTS:
            prediction, metadata = fit_evidence_fold(shared.master.master, shared.master.masks, y,
                dst, degree, inner_train, inner_val, test, variant, device, init_seed)
            model_name = variant
            rows = full_metric_rows(y, prediction, test, dst, degree, dataset, seed, fold, model_name)
            for row in rows:
                row.update({k: v for k, v in metadata.items() if k != "shuffle_audit"})
            by_fold.extend(rows)
            sr, hr = secondary_eval_rows(y, prediction, test, clean, raw_repeats, dst,
                                         degree, dataset, seed, fold, model_name)
            stable_rows.extend(sr)
            high_rows.extend(hr)
        for null_seed in (NULL_SEEDS[:1] if smoke else NULL_SEEDS):
            prediction, metadata = fit_evidence_fold(shared.master.master, shared.master.masks, y,
                dst, degree, inner_train, inner_val, test, "ENDPOINT_LOCAL", device, init_seed,
                null_shuffle_seed=null_seed)
            rows = full_metric_rows(y, prediction, test, dst, degree, dataset, seed, fold,
                                    f"STRICT_SHUFFLE_{null_seed}")
            for row in rows:
                row.update({k: v for k, v in metadata.items() if k != "shuffle_audit"})
                row["shuffle_seed"] = null_seed
                row["train_tuple_multiset_preserved"] = metadata["shuffle_audit"]["inner_train"]["multiset_preserved"]
                row["inner_validation_tuple_multiset_preserved"] = metadata["shuffle_audit"]["inner_validation"]["multiset_preserved"]
                row["outer_test_unshuffled"] = True
            null_rows.extend(rows)
            sr, hr = secondary_eval_rows(y, prediction, test, clean, raw_repeats, dst,
                                         degree, dataset, seed, fold, f"STRICT_SHUFFLE_{null_seed}")
            stable_rows.extend(sr)
            high_rows.extend(hr)
        ridge_variants = ("ENDPOINT_LOCAL",) if smoke else ("TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL")
        for variant in ridge_variants:
            prediction, metadata = fit_ridge_fold(shared.master.master, shared.master.masks, y,
                dst, degree, inner_train, inner_val, test, variant)
            rows = full_metric_rows(y, prediction, test, dst, degree, dataset, seed, fold, f"RIDGE_{variant}")
            for row in rows:
                row.update(metadata)
            ridge_rows.extend(rows)
            sr, hr = secondary_eval_rows(y, prediction, test, clean, raw_repeats, dst,
                                         degree, dataset, seed, fold, f"RIDGE_{variant}")
            stable_rows.extend(sr)
            high_rows.extend(hr)
        prediction, metadata = fit_direct_fold(shared, y, dst, degree, inner_train, inner_val,
            test, shared.edge_positions, shared.graph_src, shared.graph_dst, device,
            seed=71021 + seed * 17 + fold)
        rows = full_metric_rows(y, prediction, test, dst, degree, dataset, seed, fold, "DirectState")
        for row in rows:
            row.update(metadata)
        direct_rows.extend(rows)
        sr, hr = secondary_eval_rows(y, prediction, test, clean, raw_repeats, dst,
                                     degree, dataset, seed, fold, "DirectState")
        stable_rows.extend(sr)
        high_rows.extend(hr)
        frozen_names = ("U_MODAL",) if smoke else l0.REPRESENTATIONS
        frozen_modalities = ("text",) if smoke else MODALITIES
        for rep_name in frozen_names:
            for modality in frozen_modalities:
                rep = representations[f"{rep_name}_{modality}"]
                mod_i = MODALITIES.index(modality)
                # Frozen readout receives modality's two clean alternatives in D,P order.
                y2 = y[:, mod_i * 2:(mod_i + 1) * 2]
                prediction2, metadata = fit_frozen_fold(rep, y2, dst, degree, inner_train,
                    inner_val, test, "text", device,
                    seed=88261 + seed * 19 + fold * 3 + mod_i)
                prediction4 = np.zeros((len(test), 4), dtype=np.float64)
                prediction4[:, mod_i * 2:(mod_i + 1) * 2] = prediction2
                # Record the two evaluated modality outputs (the other modality columns are omitted).
                for local_i, global_i in enumerate(range(mod_i * 2, mod_i * 2 + 2)):
                    target_name = UTILITY_COLUMNS[global_i]
                    met = metric_bundle(y[test, global_i], prediction2[:, local_i], dst[test], degree[test])
                    row = {"dataset": dataset, "seed": seed, "fold": fold,
                           "model": f"Frozen_{rep_name}", "representation": rep_name,
                           "modality": modality, "target": target_name,
                           "operator": "absdiff" if local_i == 0 else "product", **met, **metadata}
                    frozen_rows.append(row)
                sr, hr = secondary_eval_rows(y, prediction4, test, clean, raw_repeats,
                                             dst, degree, dataset, seed, fold, f"Frozen_{rep_name}_{modality}")
                stable_rows.extend([r for r in sr if r["target"].endswith(modality)])
                high_rows.extend([r for r in hr if r["target"].endswith(modality)])
        if smoke:
            break
    if smoke_dir:
        smoke_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(by_fold).to_csv(smoke_dir / "evidence_smoke.csv", index=False)
        pd.DataFrame(null_rows).to_csv(smoke_dir / "strict_shuffle_smoke.csv", index=False)
        pd.DataFrame(ridge_rows).to_csv(smoke_dir / "ridge_smoke.csv", index=False)
        pd.DataFrame(direct_rows).to_csv(smoke_dir / "direct_state_smoke.csv", index=False)
        pd.DataFrame(frozen_rows).to_csv(smoke_dir / "frozen_u_smoke.csv", index=False)
        return {"evidence_fit_count": len(MASTER_VARIANTS), "strict_shuffle_fit_count": 1,
                "ridge_fit_count": 1, "direct_state_fit_count": 1, "frozen_u_readout_fit_count": 1,
                "fold_count": 1, "split_audits": fold_audits,
                "frozen_checkpoint_alignment": frozen_alignment,
                "all_required_fits_present": bool(by_fold and null_rows and ridge_rows and direct_rows and frozen_rows)}
    return {"evidence_probe_by_fold": by_fold, "strict_shuffle_rows": null_rows,
            "ridge_rows": ridge_rows, "direct_rows": direct_rows, "frozen_rows": frozen_rows,
            "head_stable_rows": stable_rows, "high_effect_rows": high_rows,
            "fold_audits": fold_audits, "fold_assignment_reused": reused,
            "frozen_checkpoint_alignment": frozen_alignment}


def _pair_metrics(a: np.ndarray, b: np.ndarray, dst: np.ndarray) -> dict[str, Any]:
    return {"spearman": rho(a, b), "pearson": pearson(a, b),
            "sign_agreement": float(np.mean((a > 0) == (b > 0))),
            "centered_spearman": centered_rho(a, b, dst), "n_edges": len(a)}


def head_repeat_reliability(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, seed), group in raw.groupby(["dataset", "seed"]):
        for ra, rb in itertools.combinations(sorted(group.head_repeat.unique()), 2):
            a, b = group[group.head_repeat == ra], group[group.head_repeat == rb]
            joined = a.merge(b, on=["src", "dst", "dst_degree"], suffixes=("_a", "_b"), validate="one_to_one")
            dst = joined.dst.to_numpy()
            for target in UTILITY_COLUMNS:
                metrics = _pair_metrics(joined[f"{target}_a"].to_numpy(), joined[f"{target}_b"].to_numpy(), dst)
                rows.append({"dataset": dataset, "seed": seed, "repeat_a": int(ra),
                             "repeat_b": int(rb), "target": target,
                             "modality": "text" if target.endswith("text") else "visual",
                             "operator": "absdiff" if "G_D" in target else "product", **metrics})
    return pd.DataFrame(rows)


def utility_cross_seed_reliability(clean_tables: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for sa, sb in itertools.combinations(SEEDS, 2):
            a, b = clean_tables[(dataset, sa)], clean_tables[(dataset, sb)]
            joined = a.merge(b, on=["src", "dst"], suffixes=("_a", "_b"), validate="one_to_one")
            dst = joined.dst.to_numpy()
            for target in UTILITY_COLUMNS:
                xa, xb = joined[f"mean_{target}_a"].to_numpy(), joined[f"mean_{target}_b"].to_numpy()
                ma, mb = np.abs(xa) >= np.quantile(np.abs(xa), .75), np.abs(xb) >= np.quantile(np.abs(xb), .75)
                inter = int(np.sum(ma & mb))
                denom = max(1, min(int(ma.sum()), int(mb.sum())))
                rows.append({"dataset": dataset, "seed_a": sa, "seed_b": sb, "target": target,
                             "modality": "text" if target.endswith("text") else "visual",
                             "operator": "absdiff" if "G_D" in target else "product",
                             "n_a": len(a), "n_b": len(b), "overlap_edges": len(joined),
                             "overlap_fraction_smaller": len(joined) / max(1, min(len(a), len(b))),
                             **_pair_metrics(xa, xb, dst), "top_abs_q75_overlap_count": inter,
                             "top_abs_q75_overlap_fraction_smaller": inter / denom,
                             "top_abs_q75_jaccard": inter / max(1, int((ma | mb).sum()))})
    return pd.DataFrame(rows)


def clean_heterogeneity(clean_tables: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (dataset, seed), frame in clean_tables.items():
        for target in UTILITY_COLUMNS:
            name = f"mean_{target}"
            vals = frame[name].to_numpy(dtype=np.float64)
            sub = frame[frame.dst_degree >= 5]
            target_stats = []
            coexist = []
            for _, tg in sub.groupby("dst", sort=False):
                values = tg[name].to_numpy(dtype=np.float64)
                if len(values) >= 2:
                    target_stats.append(float(np.std(values, ddof=0)))
                    coexist.append(bool(np.any(values > 0) and np.any(values < 0)))
            rows.append({"dataset": dataset, "seed": seed, "target": target,
                         "modality": "text" if target.endswith("text") else "visual",
                         "operator": "absdiff" if "G_D" in target else "product",
                         "edge_count": len(vals), "positive_fraction": float(np.mean(vals > 0)),
                         "nonpositive_fraction": float(np.mean(vals <= 0)),
                         "negative_fraction": float(np.mean(vals < 0)),
                         "mean": float(vals.mean()), "median": float(np.median(vals)),
                         **{f"q{int(q*100)}": float(np.quantile(vals, q)) for q in (.10,.25,.75,.90)},
                         "degree5_target_edge_count": int(len(sub)),
                         "degree5_within_target_gain_std_mean": float(np.mean(target_stats)) if target_stats else float("nan"),
                         "degree5_within_target_gain_std_median": float(np.median(target_stats)) if target_stats else float("nan"),
                         "degree5_positive_negative_coexistence_ratio": float(np.mean(coexist)) if coexist else float("nan"),
                         "degree5_multi_edge_target_count": len(coexist)})
    return pd.DataFrame(rows)


def modality_disagreement(clean_tables: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (dataset, seed), frame in clean_tables.items():
        for op, tcol, vcol in (("absdiff", "mean_G_D_text", "mean_G_D_visual"),
                               ("product", "mean_G_P_text", "mean_G_P_visual")):
            x, y = frame[tcol].to_numpy(), frame[vcol].to_numpy()
            rows.append({"dataset": dataset, "seed": seed, "operator": op,
                         **_pair_metrics(x, y, frame.dst.to_numpy()),
                         "sign_disagreement_fraction": float(np.mean((x > 0) != (y > 0))),
                         "centered_spearman": centered_rho(x, y, frame.dst.to_numpy())})
    return pd.DataFrame(rows)


def function_preference_summary(clean_tables: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    """Descriptive G_D-G_P margin; this is never discretized into an edge-role label."""
    rows = []
    for (dataset, seed), frame in clean_tables.items():
        for modality in MODALITIES:
            diff = (frame[f"mean_G_D_{modality}"] - frame[f"mean_G_P_{modality}"]).to_numpy(dtype=np.float64)
            rows.append({"dataset": dataset, "seed": seed, "modality": modality,
                         "edge_count": len(diff), "preference_margin_mean": float(diff.mean()),
                         "preference_margin_median": float(np.median(diff)),
                         "preference_margin_positive_fraction": float(np.mean(diff > 0)),
                         "preference_margin_nonpositive_fraction": float(np.mean(diff <= 0)),
                         "preference_margin_q10": float(np.quantile(diff, .10)),
                         "preference_margin_q25": float(np.quantile(diff, .25)),
                         "preference_margin_q75": float(np.quantile(diff, .75)),
                         "preference_margin_q90": float(np.quantile(diff, .90)),
                         "hard_function_label_created": False})
    return pd.DataFrame(rows)


def output_separability(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in UTILITY_COLUMNS:
        modality = "text" if target.endswith("text") else "visual"
        operator = "absdiff" if "G_D" in target else "product"
        for (dataset, seed, repeat), frame in raw.groupby(["dataset", "seed", "head_repeat"]):
            gain = frame[target].to_numpy(dtype=np.float64)
            js = frame[f"js_{target}"].to_numpy(dtype=np.float64)
            logit = frame[f"logit_shift_{target}"].to_numpy(dtype=np.float64)
            prob_l1 = frame[f"probability_l1_{target}"].to_numpy(dtype=np.float64)
            flip = frame[f"prediction_flip_{target}"].to_numpy(dtype=np.float64)
            ce_change = frame[f"absolute_ce_change_{target}"].to_numpy(dtype=np.float64)
            def qstats(v: np.ndarray, prefix: str) -> dict[str, float]:
                return {f"{prefix}_{stat}": float(fn(v)) for stat, fn in (
                    ("mean", np.mean), ("median", np.median), ("q25", lambda a: np.quantile(a,.25)),
                    ("q75", lambda a: np.quantile(a,.75)), ("q90", lambda a: np.quantile(a,.90)))}
            rows.append({"dataset": dataset, "seed": seed, "head_repeat": repeat,
                         "modality": modality, "operator": operator, "target": target,
                         "edge_count": len(frame), "prediction_flip_fraction": float(flip.mean()),
                         "spearman_abs_gain_js": rho(np.abs(gain), js),
                         "spearman_abs_gain_logit_shift": rho(np.abs(gain), logit),
                         **qstats(logit, "logit_shift"), **qstats(prob_l1, "probability_l1"),
                         **qstats(js, "js"), **qstats(ce_change, "absolute_ce_change")})
    return pd.DataFrame(rows)


def compare_p13(clean_tables: dict[tuple[str, int], pd.DataFrame],
                p13_frames: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, clean in clean_tables.items():
        dataset, seed = key
        old = p13_frames[key]
        joined = clean.merge(old, on=["src", "dst", "dst_degree"], validate="one_to_one")
        dst = joined.dst.to_numpy()
        for modality in MODALITIES:
            for short, op in (("D", "absdiff"), ("P", "product")):
                old_col = f"u_raw_{op}_{modality}"; smooth_col = f"u_raw_smooth_{modality}"
                x = joined[old_col].to_numpy(dtype=np.float64) - joined[smooth_col].to_numpy(dtype=np.float64)
                target = f"G_{short}_{modality}"
                y = joined[f"mean_{target}"].to_numpy(dtype=np.float64)
                old_top = np.abs(x) >= np.quantile(np.abs(x), .75)
                new_top = np.abs(y) >= np.quantile(np.abs(y), .75)
                intersection = int(np.sum(old_top & new_top))
                rows.append({"dataset": dataset, "seed": seed, "modality": modality,
                             "operator": op, "historical_target": f"Delta_{short}",
                             "new_target": target, "n_edges": len(joined), "spearman": rho(x, y),
                             "pearson": pearson(x, y), "sign_agreement": float(np.mean((x > 0) == (y > 0))),
                             "within_target_centered_spearman": centered_rho(x, y, dst),
                             "top_abs_q75_overlap_count": intersection,
                             "top_abs_q75_overlap_fraction_smaller": intersection / max(1, min(old_top.sum(), new_top.sum())),
                             "top_abs_q75_jaccard": intersection / max(1, int((old_top | new_top).sum()))})
    return pd.DataFrame(rows)


def correctness_smoke(device: torch.device) -> dict[str, Any]:
    """Small deterministic checks for graph arithmetic, slot isolation, gain signs and JS."""
    h = torch.tensor([[1., -2.], [3., 4.], [-1., 5.], [2., 1.]], dtype=torch.float64)
    edge = torch.tensor([[0, 2, 3, 0, 3], [1, 1, 1, 2, 2]], dtype=torch.long)
    if torch.any(edge[0] == edge[1]):
        raise AssertionError("synthetic self loop found")
    degree = torch.bincount(edge[1], minlength=len(h))
    context_errors = {}
    for op in OPERATORS:
        ctx, d = operator_context(h, edge, len(h), op)
        src, dst = edge
        msg = h[src] if op == "smooth" else (h[src] - h[dst]).abs() if op == "absdiff" else h[src] * h[dst]
        brute = torch.zeros_like(h).index_add_(0, dst, msg) / degree.clamp_min(1).double()[:, None]
        context_errors[op] = float((ctx - brute).abs().max())
        if not torch.equal(d, degree) or not torch.allclose(ctx, brute, rtol=0, atol=0):
            raise AssertionError(f"synthetic {op} whole-context regression failed")
    torch.manual_seed(19)
    linear = nn.Linear(512, 4, dtype=torch.float64)
    feature = torch.randn(9, 512, dtype=torch.float64)
    delta = torch.randn(9, 128, dtype=torch.float64) / 3
    details = {}
    for modality in MODALITIES:
        changed, brute_logits = substitute_edge_logits(linear, feature, delta, modality, brute=True)
        block = SLOT_TEXT if modality == "text" else SLOT_VISUAL
        fast_logits = F.linear(feature, linear.weight, linear.bias) + delta @ linear.weight[:, block].T
        details[modality] = float((brute_logits - fast_logits).abs().max())
        other = SLOT_VISUAL if modality == "text" else SLOT_TEXT
        if not torch.equal(changed[:, other], feature[:, other]):
            raise AssertionError(f"{modality} substitution changed the other modality slot")
        if not torch.allclose(brute_logits, fast_logits, rtol=1e-7, atol=1e-8):
            raise AssertionError(f"synthetic {modality} exact linear update failed")
    p = torch.tensor([[1., 0.], [0.3, 0.7]], dtype=torch.float64)
    js_identical = float(js_divergence(p, p).max())
    if abs(js_identical) > 1e-14:
        raise AssertionError("JS(p,p) must be numerically zero")
    split_train, split_select, split_audit = stratified_head_split(
        np.arange(40), np.repeat(np.arange(4), 10), seed=17)
    if set(split_train) & set(split_select) or len(split_train) + len(split_select) != 40:
        raise AssertionError("synthetic head split failed")
    y = np.asarray([[1., 2., 3., 4.], [3., 4., 5., 6.], [8., 7., 6., 5.]])
    dst = np.asarray([0, 0, 1])
    shuffled, shuffle_audit = shuffled_within_dst(y, dst, np.asarray([0, 1]), seed=9917)
    if not shuffle_audit["multiset_preserved"] or not np.array_equal(shuffled[2], y[2]):
        raise AssertionError("synthetic tuple shuffle invariant failed")
    return {"synthetic_context_max_abs_error": context_errors,
            "synthetic_fast_vs_brute_max_abs_error": details,
            "js_identical_max_abs": js_identical,
            "head_split_audit": split_audit,
            "tuple_shuffle_audit": shuffle_audit,
            "self_loop_removed": True, "original_degree_fixed": True}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def fit_one_head_repeat(shared: SharedData, dataset_index: int, repeat: int,
                        device: torch.device, save: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    head, audit = train_shared_head(shared, dataset_index, shared.seed, repeat, device)
    raw, errors = build_repeat_utilities(shared, head, repeat, device, check_all=True)
    raw.insert(0, "dataset", shared.data.name)
    raw.insert(1, "seed", shared.seed)
    audit["performance"].update({"fast_vs_brute_max_abs_errors": errors,
                                  "fast_brute_rtol": 1e-7, "fast_brute_atol": 1e-8})
    if save:
        checkpoint = RAW_DIR / "checkpoints" / "shared_heads" / shared.data.name / f"seed_{shared.seed}_repeat_{repeat}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "dataset": shared.data.name, "seed": shared.seed, "repeat": repeat,
                    "input_dim": 512, "operator_combinations": list(itertools.product(OPERATORS, repeat=2)),
                    "head_train_nodes": audit["head_train"], "head_select_nodes": audit["head_select"],
                    "validation_labels_used_for_training_or_selection": False}, checkpoint)
        audit["checkpoint"] = str(checkpoint)
    return raw, audit


def run_smoke(device: torch.device) -> dict[str, Any]:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    synthetic = correctness_smoke(device)
    shared = load_shared_data("Movies", 42, device)
    if max(abs(v) for v in shared.h0_audit["checkpoint_metric_differences"].values()) > 2e-5:
        raise AssertionError("P0 frozen H0 metrics did not reproduce")
    raw, head_audit = fit_one_head_repeat(shared, 0, 0, device)
    smoke_raw_path = RAW_DIR / "smoke" / "Movies" / "seed_42_shared_head_repeat.csv.gz"
    smoke_raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(smoke_raw_path, index=False, compression="gzip")
    clean = aggregate_repeats(raw)
    clean.insert(0, "dataset", "Movies")
    clean.insert(1, "seed", 42)
    probe_audit = run_probe_suite(shared, clean, device, raw, smoke=True,
                                  smoke_dir=RAW_DIR / "smoke" / "Movies" / "seed_42_probe")
    if not probe_audit["all_required_fits_present"]:
        raise AssertionError("Movies/42 smoke probe suite incomplete")
    audit = {
        "status": "passed", "dataset": "Movies", "seed": 42,
        "source_head": "exp/l0_relation_function_learnability_audit",
        "source_sha": "9ce5f723fda0bf17264671a1e20e78f043727498",
        "device": str(device), "synthetic_checks": synthetic,
        "h0_checkpoint_regression": shared.h0_audit,
        "p13_edge_alignment": shared.p13_alignment,
        "shared_head": head_audit["performance"], "head_split_audit": head_audit["split_audit"],
        "validation_labels_not_used_for_head_fit": True,
        "utility_rows": len(raw), "utility_targets": list(UTILITY_COLUMNS),
        "probe_smoke": probe_audit,
        "no_test_labels_or_metrics_accessed": True,
        "elapsed_seconds": time.time() - start,
    }
    save_json(RAW_DIR / "smoke" / "smoke_audit.json", audit)
    return audit


def run_formal_campaign(device: torch.device) -> tuple[dict[tuple[str, int], pd.DataFrame],
                                                          dict[tuple[str, int], pd.DataFrame],
                                                          dict[str, list[dict[str, Any]]], dict[str, Any]]:
    start = time.time()
    all_raw: dict[tuple[str, int], pd.DataFrame] = {}
    clean_tables: dict[tuple[str, int], pd.DataFrame] = {}
    p13_frames: dict[tuple[str, int], pd.DataFrame] = {}
    outputs: dict[str, list[dict[str, Any]]] = {
        "head_performance": [], "head_split": [], "evidence": [], "shuffle": [],
        "ridge": [], "direct": [], "frozen": [], "head_stable": [], "high_effect": [],
        "fold_audit": [],
    }
    run_audits = []
    for dataset_index, dataset in enumerate(DATASETS):
        for seed in SEEDS:
            started = time.time()
            print(f"[l01] heads + utilities {dataset}/{seed}", flush=True)
            shared = load_shared_data(dataset, seed, device)
            repeats, audits = [], []
            for repeat in range(3):
                raw, result = fit_one_head_repeat(shared, dataset_index, repeat, device)
                repeats.append(raw); audits.append(result)
                outputs["head_performance"].append(result["performance"])
                outputs["head_split"].append(result["split_audit"])
            raw_all = pd.concat(repeats, ignore_index=True)
            raw_path = RAW_DIR / "raw_utilities" / dataset / f"seed_{seed}_substitution_utility.csv.gz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_all.to_csv(raw_path, index=False, compression="gzip")
            clean = aggregate_repeats(raw_all)
            clean.insert(0, "dataset", dataset); clean.insert(1, "seed", seed)
            clean_tables[(dataset, seed)] = clean
            p13_frames[(dataset, seed)] = shared.p13_raw.copy()
            all_raw[(dataset, seed)] = raw_all
            clean_path = RAW_DIR / "clean_utility" / dataset / f"seed_{seed}_mean_gain.csv.gz"
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            clean.to_csv(clean_path, index=False, compression="gzip")
            print(f"[l01] relation probes {dataset}/{seed}", flush=True)
            result = run_probe_suite(shared, clean, device, raw_all)
            for out_key, result_key in (("evidence", "evidence_probe_by_fold"),
                                        ("shuffle", "strict_shuffle_rows"),
                                        ("ridge", "ridge_rows"), ("direct", "direct_rows"),
                                        ("frozen", "frozen_rows"), ("head_stable", "head_stable_rows"),
                                        ("high_effect", "high_effect_rows"), ("fold_audit", "fold_audits")):
                outputs[out_key].extend(result[result_key])
            run_audits.append({"dataset": dataset, "seed": seed, "p13_edge_alignment": shared.p13_alignment,
                               "h0_regression": shared.h0_audit, "head_repeats": [x["performance"] for x in audits],
                               "head_split_audits": [x["split_audit"] for x in audits],
                               "probe_folds": result["fold_audits"],
                               "frozen_checkpoint_alignment": result["frozen_checkpoint_alignment"],
                               "elapsed_seconds": time.time() - started,
                               "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0})
            group_dir = RAW_DIR / "probe_fits" / dataset / f"seed_{seed}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for name, result_key in (("evidence", "evidence_probe_by_fold"),
                                     ("strict_shuffle", "strict_shuffle_rows"),
                                     ("ridge", "ridge_rows"), ("direct", "direct_rows"),
                                     ("frozen", "frozen_rows"), ("head_stable", "head_stable_rows"),
                                     ("high_effect", "high_effect_rows"), ("fold_audit", "fold_audits")):
                pd.DataFrame(result[result_key]).to_csv(group_dir / f"{name}.csv", index=False)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # Required primary tables are written incrementally after all 9 dataset/seed groups finish.
    data_dir = OUT_DIR / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs["head_performance"]).to_csv(data_dir / "shared_head_performance.csv", index=False)
    pd.DataFrame(outputs["head_split"]).to_csv(data_dir / "shared_head_split_audit.csv", index=False)
    for key, frame in all_raw.items():
        raw_path = RAW_DIR / "raw_utilities" / key[0] / f"seed_{key[1]}_substitution_utility.csv.gz"
        # Keep full per-repeat rows only in the ignored output tree.
        frame.to_csv(raw_path, index=False, compression="gzip")
    pd.concat(list(all_raw.values()), ignore_index=True).to_csv(
        RAW_DIR / "all_shared_slot_raw_utilities.csv.gz", index=False, compression="gzip")
    pd.concat(list(clean_tables.values()), ignore_index=True).to_csv(
        data_dir / "substitution_utility_by_repeat_summary.csv", index=False)
    pd.DataFrame(outputs["evidence"]).to_csv(data_dir / "evidence_probe_by_fold.csv", index=False)
    pd.DataFrame(outputs["shuffle"]).to_csv(data_dir / "shuffled_null_summary.csv", index=False)
    pd.DataFrame(outputs["ridge"]).to_csv(data_dir / "ridge_probe_summary.csv", index=False)
    pd.DataFrame(outputs["direct"]).to_csv(data_dir / "direct_state_probe_summary.csv", index=False)
    pd.DataFrame(outputs["frozen"]).to_csv(data_dir / "frozen_state_readout_summary.csv", index=False)
    pd.DataFrame(outputs["head_stable"]).to_csv(data_dir / "head_stable_subset.csv", index=False)
    pd.DataFrame(outputs["high_effect"]).to_csv(data_dir / "high_effect_summary.csv", index=False)
    # These audit records are retained in the detailed run manifest.
    extra = {"run_audits": run_audits, "fold_audits": outputs["fold_audit"],
             "elapsed_seconds": time.time() - start,
             "fit_counts": {"shared_heads": len(outputs["head_performance"]),
                            "evidence_mlp_real": 3*3*3*4, "strict_shuffle": 3*3*3*3,
                            "evidence_mlp_total": 3*3*3*(4+3),
                            "ridge": 3*3*3*3, "direct_state": 3*3*3,
                            "frozen_readouts": 3*3*3*2*3}}
    save_json(RAW_DIR / "formal_campaign_audit.json", extra)
    return all_raw, clean_tables, p13_frames, {**outputs, **extra}


def grouped_metric_summary(frame: pd.DataFrame, groups: list[str], output_path: Path) -> pd.DataFrame:
    if frame.empty:
        result = pd.DataFrame()
    else:
        metrics = [name for name in ("spearman", "pearson", "r2", "sign_auroc",
                    "sign_balanced_accuracy", "within_target_residual_spearman",
                    "degree5_target_rank_mean", "mean_absolute_error") if name in frame.columns]
        agg = {column: ["mean", "std", "count"] for column in metrics}
        result = frame.groupby(groups, dropna=False).agg(agg)
        result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
        result = result.reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def finalize_campaign(all_raw: dict[tuple[str, int], pd.DataFrame],
                      clean_tables: dict[tuple[str, int], pd.DataFrame],
                      p13_frames: dict[tuple[str, int], pd.DataFrame],
                      outputs: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_dir = OUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(list(all_raw.values()), ignore_index=True)
    head_rel = head_repeat_reliability(raw)
    cross_seed = utility_cross_seed_reliability(clean_tables)
    hetero = clean_heterogeneity(clean_tables)
    separability = output_separability(raw)
    modality = modality_disagreement(clean_tables)
    preference = function_preference_summary(clean_tables)
    p13_compare = compare_p13(clean_tables, p13_frames)
    head_rel.to_csv(data_dir / "utility_head_repeat_reliability.csv", index=False)
    cross_seed.to_csv(data_dir / "utility_cross_seed_reliability.csv", index=False)
    hetero.to_csv(data_dir / "substitution_heterogeneity.csv", index=False)
    separability.to_csv(data_dir / "output_separability.csv", index=False)
    modality.to_csv(data_dir / "modality_disagreement.csv", index=False)
    preference.to_csv(data_dir / "function_preference_summary.csv", index=False)
    p13_compare.to_csv(data_dir / "p13_vs_shared_slot.csv", index=False)
    evidence = pd.DataFrame(outputs["evidence"])
    shuffle = pd.DataFrame(outputs["shuffle"])
    ridge = pd.DataFrame(outputs["ridge"])
    direct = pd.DataFrame(outputs["direct"])
    frozen = pd.DataFrame(outputs["frozen"])
    evidence.to_csv(RAW_DIR / "evidence_probe_by_fold_and_target.csv.gz", index=False, compression="gzip")
    shuffle.to_csv(RAW_DIR / "strict_shuffle_by_fold_and_target.csv.gz", index=False, compression="gzip")
    ridge.to_csv(RAW_DIR / "ridge_by_fold_and_target.csv.gz", index=False, compression="gzip")
    direct.to_csv(RAW_DIR / "direct_state_by_fold_and_target.csv.gz", index=False, compression="gzip")
    frozen.to_csv(RAW_DIR / "frozen_readout_by_fold_and_target.csv.gz", index=False, compression="gzip")
    evidence_summary = grouped_metric_summary(evidence,
        ["dataset", "model", "target", "modality", "operator"], data_dir / "evidence_probe_summary.csv")
    shuffle_summary = grouped_metric_summary(shuffle,
        ["dataset", "model", "target", "modality", "operator"], data_dir / "shuffled_null_summary.csv")
    ridge_summary = grouped_metric_summary(ridge,
        ["dataset", "model", "target", "modality", "operator"], data_dir / "ridge_probe_summary.csv")
    direct_summary = grouped_metric_summary(direct,
        ["dataset", "model", "target", "modality", "operator"], data_dir / "direct_state_probe_summary.csv")
    frozen_summary = grouped_metric_summary(frozen,
        ["dataset", "model", "representation", "modality", "target", "operator"], data_dir / "frozen_state_readout_summary.csv")
    pd.DataFrame(outputs["head_stable"]).to_csv(data_dir / "head_stable_subset.csv", index=False)
    pd.DataFrame(outputs["high_effect"]).to_csv(data_dir / "high_effect_summary.csv", index=False)
    save_json(RAW_DIR / "fit_summary_counts.json", {
        "heads": len(outputs["head_performance"]), "evidence_real": 3*3*3*4,
        "strict_shuffle": 3*3*3*3, "ridge": 3*3*3*3,
        "direct_state": 3*3*3, "frozen_state": 3*3*3*2*3,
        "total_evidence_mlp_including_null": 3*3*3*7,
    })
    return {"head_repeat_reliability": head_rel, "cross_seed_reliability": cross_seed,
            "heterogeneity": hetero, "separability": separability, "modality": modality,
            "function_preference": preference,
            "p13_compare": p13_compare, "evidence_summary": evidence_summary,
            "shuffle_summary": shuffle_summary, "ridge_summary": ridge_summary,
            "direct_summary": direct_summary, "frozen_summary": frozen_summary}


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="L0.1 shared-slot function identifiability audit")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--formal-only", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda" and device.index is None:
        raise ValueError("explicit CUDA device index is required")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    smoke_audit = None
    if not args.skip_smoke and not args.formal_only:
        smoke_audit = run_smoke(device)
        print(f"[l01] smoke passed in {smoke_audit['elapsed_seconds']:.1f}s", flush=True)
    if args.smoke_only:
        return
    all_raw, clean, p13, outputs = run_formal_campaign(device)
    summaries = finalize_campaign(all_raw, clean, p13, outputs)
    manifest = {
        "branch": "exp/l01_shared_slot_function_identifiability",
        "source_branch": "exp/l0_relation_function_learnability_audit",
        "source_sha": "9ce5f723fda0bf17264671a1e20e78f043727498",
        "device": str(device), "datasets": list(DATASETS), "seeds": list(SEEDS),
        "smoke": smoke_audit if smoke_audit is not None else json.loads((RAW_DIR / "smoke/smoke_audit.json").read_text()) if (RAW_DIR / "smoke/smoke_audit.json").is_file() else None,
        "campaign": {"run_audits": outputs.get("run_audits", []),
                     "fold_audits": outputs.get("fold_audits", []),
                     "fit_counts": outputs.get("fit_counts", {}),
                     "elapsed_seconds": outputs.get("elapsed_seconds")},
        "data_tables": {path.name: {"rows": len(pd.read_csv(path))}
                        for path in sorted((OUT_DIR / "data").glob("*.csv"))},
        "no_test_evaluation_or_labels": True,
        "shared_head_validation_labels_used": False,
        "historical_artifacts_modified": False,
        "outcomes": {"analysis_tables_created": sorted(summaries.keys()),
                     "fit_counts": outputs["fit_counts"]},
        "elapsed_seconds": outputs["elapsed_seconds"],
        "protocol_deviations": [], "retries": [],
    }
    save_json(OUT_DIR / "run_manifest.json", manifest)
    print("[l01] formal campaign completed", flush=True)


if __name__ == "__main__":
    main()
