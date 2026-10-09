from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .message_effect_atlas import _safe_pearson, _safe_spearman
from .message_effect_replay import FrozenHostCache, replay_local_basis_changes

TAUS = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
ORACLE_THRESHOLDS = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
TARGET_FIELDS = {"delete": "delta_delete_matched", "comp": "delta_comp_matched"}


@torch.no_grad()
def matched_local_baseline(
    cache: FrozenHostCache, receivers: torch.Tensor, labels: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Replay the unchanged local basis and use that exact path as the baseline."""
    receivers = torch.as_tensor(receivers, dtype=torch.long, device=cache.device).reshape(-1)
    labels = torch.as_tensor(labels, dtype=torch.long, device=cache.device).reshape(-1)
    if receivers.numel() != labels.numel() or not receivers.numel():
        raise ValueError("receivers and train labels must have matching nonzero length")
    replay = replay_local_basis_changes(cache, receivers, changes=[])
    loss = F.cross_entropy(replay["logits"], labels, reduction="none")
    return {"logits": replay["logits"], "z": replay["z"], "loss": loss}


def matched_effect_from_losses(
    modified_logits: torch.Tensor, labels: torch.Tensor, local_baseline_loss: torch.Tensor
) -> torch.Tensor:
    labels = torch.as_tensor(labels, dtype=torch.long, device=modified_logits.device).reshape(-1)
    baseline = torch.as_tensor(local_baseline_loss, dtype=modified_logits.dtype, device=modified_logits.device).reshape(-1)
    if modified_logits.shape[0] != labels.numel() or baseline.numel() != labels.numel():
        raise ValueError("modified logits, labels, and local baseline must align")
    return F.cross_entropy(modified_logits, labels, reduction="none") - baseline


def old_effect_from_losses(
    modified_logits: torch.Tensor, labels: torch.Tensor, full_host_baseline_loss: torch.Tensor
) -> torch.Tensor:
    labels = torch.as_tensor(labels, dtype=torch.long, device=modified_logits.device).reshape(-1)
    baseline = torch.as_tensor(full_host_baseline_loss, dtype=modified_logits.dtype, device=modified_logits.device).reshape(-1)
    if modified_logits.shape[0] != labels.numel() or baseline.numel() != labels.numel():
        raise ValueError("modified logits, labels, and full-host baseline must align")
    return F.cross_entropy(modified_logits, labels, reduction="none") - baseline


def baseline_path_diagnostics(
    local: dict[str, torch.Tensor], full_logits: torch.Tensor, full_z: torch.Tensor,
    full_loss: torch.Tensor,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, left, right in (("logits", local["logits"], full_logits), ("z", local["z"], full_z), ("loss", local["loss"], full_loss)):
        delta = (left.float() - right.float()).reshape(-1)
        result[f"baseline_{name}_max_abs"] = float(delta.abs().max().item()) if delta.numel() else 0.0
        result[f"baseline_{name}_rms"] = float(delta.square().mean().sqrt().item()) if delta.numel() else 0.0
    return result


def matched_old_effect_diagnostics(matched: Sequence[float], old: Sequence[float]) -> dict[str, float]:
    m = np.asarray(matched, dtype=np.float64)
    o = np.asarray(old, dtype=np.float64)
    keep = np.isfinite(m) & np.isfinite(o)
    m, o = m[keep], o[keep]
    diff = m - o
    sign_keep = (m != 0) & (o != 0)
    return {
        "n": int(m.size),
        "matched_mean": float(np.mean(m)) if m.size else float("nan"),
        "old_mean": float(np.mean(o)) if o.size else float("nan"),
        "matched_minus_old_mean": float(np.mean(diff)) if diff.size else float("nan"),
        "matched_minus_old_std": float(np.std(diff)) if diff.size else float("nan"),
        "matched_minus_old_median_abs": float(np.median(np.abs(diff))) if diff.size else float("nan"),
        "matched_minus_old_p95_abs": float(np.quantile(np.abs(diff), 0.95)) if diff.size else float("nan"),
        "matched_minus_old_max_abs": float(np.max(np.abs(diff))) if diff.size else float("nan"),
        "old_matched_pearson": _safe_pearson(m.tolist(), o.tolist()),
        "old_matched_spearman": _safe_spearman(m.tolist(), o.tolist()),
        "old_matched_sign_disagreement_fraction": float(np.mean(np.sign(m[sign_keep]) != np.sign(o[sign_keep]))) if sign_keep.any() else float("nan"),
        "sign_comparison_n": int(sign_keep.sum()),
    }


def _effect(row: dict[str, Any], target: str) -> float:
    value = row.get(TARGET_FIELDS[target])
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def pairwise_agreement(true: Sequence[float], pred: Sequence[float], tau: float = 0.0) -> tuple[int, float, float]:
    eligible = correct = 0
    for (a, ta), (b, tb) in combinations(zip(true, pred), 2):
        gap = float(ta) - float(tb)
        if not math.isfinite(gap) or abs(gap) < tau or gap == 0:
            continue
        pdiff = float(a) - float(b)
        eligible += 1
        correct += 0.5 if pdiff == 0 else float((gap < 0) == (pdiff < 0))
    return eligible, correct, correct / eligible if eligible else float("nan")


def sensitivity_diagnostics(rows: Sequence[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    field = TARGET_FIELDS[target]
    modalities: dict[tuple[int, int, int], dict[str, float]] = defaultdict(dict)
    hops: dict[tuple[int, int, str], dict[int, float]] = defaultdict(dict)
    edge_groups: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = _effect(row, target)
        receiver, edge, hop, modality = int(row["receiver_id"]), int(row["directed_edge_index"]), int(row["hop"]), str(row["modality"])
        if not math.isfinite(value):
            continue
        modalities[(receiver, edge, hop)][modality] = value
        hops[(receiver, edge, modality)][hop] = value
        edge_groups[(receiver, modality, hop)].append(value)
    output: list[dict[str, Any]] = []
    paired = [(x["text"], x["visual"]) for x in modalities.values() if "text" in x and "visual" in x]
    for tau in TAUS:
        selected = [(a, b) for a, b in paired if abs(a) >= tau and abs(b) >= tau]
        output.append({
            "target": target, "diagnostic": "text_visual_sign", "tau": tau,
            "eligible_pairs": len(selected), "eligible_fraction": len(selected) / len(paired) if paired else float("nan"),
            "sign_disagreement_fraction": float(np.mean([np.sign(a) != np.sign(b) for a, b in selected])) if selected else float("nan"),
            "effect_spearman": _safe_spearman([a for a, _ in selected], [b for _, b in selected]),
            "effect_pearson": _safe_pearson([a for a, _ in selected], [b for _, b in selected]),
            "groups_n": len(paired),
        })
        complete_hops = [v for v in hops.values() if all(k in v for k in (1, 2, 3, 4))]
        mixed = [v for v in complete_hops if max(v.values()) > tau and min(v.values()) < -tau]
        hop_diffs = [abs(v[a] - v[b]) for v in complete_hops for a, b in combinations((1, 2, 3, 4), 2)]
        hop_rhos = [_safe_spearman([v[k] for k in (1, 2, 3, 4)], [1, 2, 3, 4]) for v in complete_hops]
        output.append({
            "target": target, "diagnostic": "hop_robust_sign", "tau": tau,
            "groups_n": len(complete_hops), "eligible_pairs": len(complete_hops),
            "eligible_fraction": len(complete_hops) / len(hops) if hops else float("nan"),
            "strong_mixed_sign_groups": len(mixed),
            "strong_mixed_sign_fraction": len(mixed) / len(complete_hops) if complete_hops else float("nan"),
            "mean_effect_std": float(np.mean([np.std([v[k] for k in (1, 2, 3, 4)]) for v in complete_hops])) if complete_hops else float("nan"),
            "mean_abs_pairwise_effect_difference": float(np.mean(hop_diffs)) if hop_diffs else float("nan"),
            "mean_hop_order_spearman": float(np.nanmean(hop_rhos)) if any(math.isfinite(x) for x in hop_rhos) else float("nan"),
        })
        total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in edge_groups.values())
        eligible_pairs = sum(1 for values in edge_groups.values() for a, b in combinations(values, 2) if abs(a - b) >= tau)
        output.append({
            "target": target, "diagnostic": "edge_pair_gap", "tau": tau,
            "groups_n": sum(len(v) >= 2 for v in edge_groups.values()),
            "candidate_pairs": total_pairs, "eligible_pairs": eligible_pairs,
            "eligible_pair_fraction": eligible_pairs / total_pairs if total_pairs else float("nan"),
        })
    return output


def _distribution(values: Sequence[float], prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "median", "p25", "p50", "p75", "p90", "p95")}
    return {f"{prefix}_{name}": float(value) for name, value in zip(("mean", "median", "p25", "p50", "p75", "p90", "p95"), (np.mean(x), np.median(x), *np.quantile(x, [0.25, 0.5, 0.75, 0.9, 0.95])))}


def oracle_group_rows(rows: Sequence[dict[str, Any]], target: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = (
        ("edge", lambda r: (int(r["receiver_id"]), str(r["modality"]), int(r["hop"]))),
        ("modality", lambda r: (int(r["receiver_id"]), int(r["directed_edge_index"]), int(r["hop"]))),
        ("hop", lambda r: (int(r["receiver_id"]), int(r["directed_edge_index"]), str(r["modality"]))),
    )
    group_output: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    required_size = {"edge": None, "modality": 2, "hop": 4}
    for granularity, key_fn in definitions:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = _effect(row, target)
            if math.isfinite(value):
                grouped[key_fn(row)].append(row)
        gains, randoms, oracles = [], [], []
        for key, group in grouped.items():
            values = [_effect(r, target) for r in group]
            # A coarse group is complete by construction: 2 modalities or all 4 hops.
            n = len(values)
            if n < 2 or (required_size[granularity] is not None and n != required_size[granularity]):
                continue
            random_expected, oracle = float(np.mean(values)), float(np.min(values))
            gain = random_expected - oracle
            gains.append(gain); randoms.append(random_expected); oracles.append(oracle)
            group_output.append({
                "target": target, "granularity": granularity, "group_key": ":".join(map(str, key)),
                "group_size": n, "random_expected": random_expected, "oracle_effect": oracle,
                "oracle_gain_vs_random": gain,
            })
        record: dict[str, Any] = {"target": target, "granularity": granularity, "groups_n": len(gains)}
        record.update(_distribution(randoms, "random_expected"))
        record.update(_distribution(oracles, "oracle_effect"))
        record.update(_distribution(gains, "oracle_gain_vs_random"))
        for threshold in ORACLE_THRESHOLDS:
            record[f"fraction_gain_gt_{threshold:g}"] = float(np.mean(np.asarray(gains) > threshold)) if gains else float("nan")
        summary.append(record)
    return group_output, summary


def selection_group_rows(
    rows: Sequence[dict[str, Any]], predictions: Sequence[float], target: str,
    *, model_name: str, dataset: str, estimator: str, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must align")
    grouped: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for i, (row, pred) in enumerate(zip(rows, predictions)):
        if math.isfinite(_effect(row, target)) and math.isfinite(float(pred)):
            grouped[(int(row["receiver_id"]), str(row["modality"]), int(row["hop"]))].append(i)
    outputs: list[dict[str, Any]] = []
    for key, indices in grouped.items():
        if len(indices) < 2:
            continue
        true = np.asarray([_effect(rows[i], target) for i in indices])
        pred = np.asarray([float(predictions[i]) for i in indices])
        selected = int(np.argmin(pred))
        oracle_min = float(np.min(true))
        selected_true = float(true[selected])
        random_expected = float(np.mean(true))
        top1 = float(selected_true == oracle_min)
        pair_correct = []
        for a, b in combinations(range(len(indices)), 2):
            true_diff = true[a] - true[b]
            pred_diff = pred[a] - pred[b]
            if true_diff == 0:
                continue
            pair_correct.append(0.5 if pred_diff == 0 else float((true_diff < 0) == (pred_diff < 0)))
        outputs.append({
            "dataset": dataset, "target": target, "model": model_name, "estimator": estimator,
            "seed": seed, "receiver_id": key[0], "modality": key[1], "hop": key[2],
            "group_size": len(indices), "model_selected_effect": selected_true,
            "random_expected": random_expected, "oracle_effect": oracle_min,
            "model_gain_vs_random": random_expected - selected_true,
            "oracle_gain_vs_random": random_expected - oracle_min,
            "oracle_regret": selected_true - oracle_min,
            "top1_agreement": top1, "top1_chance": 1.0 / len(indices),
            "top1_excess": top1 - 1.0 / len(indices),
            "edge_pairwise_accuracy": float(np.mean(pair_correct)) if pair_correct else float("nan"),
        })
    fields = ("model_selected_effect", "random_expected", "oracle_effect", "model_gain_vs_random", "oracle_gain_vs_random", "oracle_regret", "top1_agreement", "top1_excess", "edge_pairwise_accuracy")
    summary: dict[str, float] = {"groups_n": len(outputs)}
    for field in fields:
        values = np.asarray([float(row[field]) for row in outputs])
        summary[f"{field}_mean"] = float(values.mean()) if values.size else float("nan")
        summary[f"{field}_median"] = float(np.median(values)) if values.size else float("nan")
        summary[f"{field}_p25"] = float(np.quantile(values, 0.25)) if values.size else float("nan")
        summary[f"{field}_p75"] = float(np.quantile(values, 0.75)) if values.size else float("nan")
    summary["fraction_model_gain_vs_random_gt_0"] = float(np.mean([r["model_gain_vs_random"] > 0 for r in outputs])) if outputs else float("nan")
    return outputs, summary
