from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from .message_effect_atlas import _safe_pearson, _safe_spearman
from .message_effect_estimator import MessageEffectEstimator
from .message_effect_replay import estimator_receiver_split
from .message_effect_v5a1 import pairwise_agreement, selection_group_rows
from .message_effect_v5a1_estimator import (
    SingleTargetMessageEffectEstimator,
    make_edge_rank_pairs,
    receiver_equal_mean,
    receiver_equal_row_weights,
    ranknet_pair_loss,
)

ESTIMATORS = ("E0_heuristic", "E1_unimodal_pair", "E2_multimodal_pair", "E3_host_context")
TARGETS = ("delete", "comp")


def _save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def _feature_inputs(model: torch.nn.Module, features: dict[str, torch.Tensor], indices: torch.Tensor,
                    scalar_mean: torch.Tensor, scalar_std: torch.Tensor,
                    host_mean: torch.Tensor, host_std: torch.Tensor) -> dict[str, torch.Tensor]:
    scalar = features["scalar"].index_select(0, indices)
    result: dict[str, torch.Tensor] = {"scalar_features": (scalar - scalar_mean) / scalar_std}
    variant = model.variant
    if variant == "E1_unimodal_pair":
        modality = features["modality_code"].index_select(0, indices)
        pair = torch.where(modality[:, None] == 0,
                           features["pair_text"].index_select(0, indices),
                           features["pair_visual"].index_select(0, indices))
        result["pair_active"] = pair
    elif variant in {"E2_multimodal_pair", "E3_host_context"}:
        result["pair_text"] = features["pair_text"].index_select(0, indices)
        result["pair_visual"] = features["pair_visual"].index_select(0, indices)
        if variant == "E3_host_context":
            result["host_context"] = (features["host"].index_select(0, indices) - host_mean) / host_std
    return result


def row_splits(features: dict[str, torch.Tensor], dataset: str) -> tuple[dict[str, torch.Tensor], dict[str, set[int]]]:
    receiver = features["receiver_id"].long().cpu()
    split = estimator_receiver_split(dataset, receiver, seed=2027)
    indices = {name: torch.tensor([i for i, r in enumerate(receiver.tolist()) if int(r) in members], dtype=torch.long)
               for name, members in split.items()}
    if any(not members for members in split.values()):
        raise ValueError("receiver split contains an empty partition")
    if any(a & b for i, a in enumerate(split.values()) for b in list(split.values())[i + 1:]):
        raise AssertionError("receiver partitions overlap")
    return indices, split


def _normalizers(features: dict[str, torch.Tensor], train_idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
    scalar = features["scalar"].index_select(0, train_idx)
    host = features["host"].index_select(0, train_idx)
    return (scalar.mean(0), scalar.std(0, unbiased=False).clamp_min(1.0e-6),
            host.mean(0), host.std(0, unbiased=False).clamp_min(1.0e-6))


def _target_stats(target: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = target.index_select(0, indices)
    values = values[torch.isfinite(values)]
    if not values.numel():
        return target.new_zeros(()), target.new_ones(())
    return values.mean(), values.std(unbiased=False).clamp_min(1.0e-8)


def _predict(model: torch.nn.Module, features: dict[str, torch.Tensor], indices: torch.Tensor,
             norms: tuple[torch.Tensor, ...], device: torch.device, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            idx = indices[start:start + batch_size]
            args = _feature_inputs(model, features, idx, *norms)
            outputs.append(model(**{k: v.to(device) for k, v in args.items()}).detach().cpu())
    return torch.cat(outputs) if outputs else torch.empty((0, 1), dtype=torch.float32)


def _val_pointwise_loss(model: torch.nn.Module, features: dict[str, torch.Tensor], indices: torch.Tensor,
                        targets: list[torch.Tensor], means: list[torch.Tensor], stds: list[torch.Tensor],
                        norms: tuple[torch.Tensor, ...], device: torch.device,
                        multi: bool) -> float:
    if not indices.numel():
        return float("inf")
    model.eval()
    receivers = features["receiver_id"].index_select(0, indices)
    per_target = []
    with torch.no_grad():
        for start in range(0, indices.numel(), 2048):
            idx = indices[start:start + 2048]
            args = _feature_inputs(model, features, idx, *norms)
            pred = model(**{k: v.to(device) for k, v in args.items()}).detach().cpu()
            pred = pred if multi else pred.reshape(-1, 1)
            for j, target in enumerate(targets):
                truth = target.index_select(0, idx)
                finite = torch.isfinite(truth)
                if not finite.any():
                    continue
                scaled = (truth[finite] - means[j]) / stds[j]
                loss = F.smooth_l1_loss(pred[finite, j], scaled, reduction="none")
                per_target.append((j, idx[finite], loss))
    scores = []
    # Receiver-equal validation objective, computed separately for each output target.
    for j in range(len(targets)):
        chunks = [(idx, val) for target_j, idx, val in per_target if target_j == j]
        if not chunks:
            continue
        ix = torch.cat([x[0] for x in chunks])
        vals = torch.cat([x[1] for x in chunks])
        recv = features["receiver_id"].index_select(0, ix)
        scores.append(float(receiver_equal_mean(vals, recv)))
    return float(np.mean(scores)) if scores else float("inf")


def _row_training_weights(receiver_ids: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    if valid is None:
        return receiver_equal_row_weights(receiver_ids)
    valid = valid.bool().cpu()
    weights = torch.zeros(receiver_ids.numel(), dtype=torch.float32)
    if valid.any():
        weights[valid] = receiver_equal_row_weights(receiver_ids.cpu()[valid])
    return weights


def train_pointwise(
    dataset: str, estimator: str, seed: int, mode: str,
    features: dict[str, torch.Tensor], rows: list[dict[str, Any]],
    *, target_name: str | None = None, device: str = "cuda:1", max_epochs: int = 100,
    patience: int = 10, batch_size: int = 2048, checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Train receiver-equal matched MultiTarget (M) or independent SingleTarget (S)."""
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested training device unavailable: {device}")
    features = {k: v.detach().cpu() for k, v in features.items()}
    idx, split = row_splits(features, dataset)
    train_idx, val_idx, hold_idx = idx["EstimatorTrain"], idx["EstimatorVal"], idx["EstimatorHoldout"]
    if mode not in {"M", "S"}:
        raise ValueError("pointwise mode must be M or S")
    target_names = list(TARGETS) if mode == "M" else [str(target_name)]
    target_map = {name: features[f"target_{name}"].float() for name in TARGETS}
    target_vectors = [target_map[name] for name in target_names]
    means_stds = [_target_stats(target, train_idx[torch.isfinite(target.index_select(0, train_idx))]) for target in target_vectors]
    means, stds = [x[0] for x in means_stds], [x[1] for x in means_stds]
    norms = _normalizers(features, train_idx)
    torch.manual_seed(53000 + int(seed))
    np.random.seed(53000 + int(seed))
    scalar_width, host_width = features["scalar"].size(-1), features["host"].size(-1)
    if mode == "M":
        model: torch.nn.Module = MessageEffectEstimator(estimator, scalar_width,
                  host_dim=host_width if estimator == "E3_host_context" else None).to(dev)
    else:
        model = SingleTargetMessageEffectEstimator(estimator, scalar_width,
                  host_dim=host_width if estimator == "E3_host_context" else None).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    recv = features["receiver_id"]
    valid_masks, train_weights = [], []
    for target in target_vectors:
        valid = torch.isfinite(target)
        valid_masks.append(valid)
        train_weights.append(_row_training_weights(recv.index_select(0, train_idx), valid.index_select(0, train_idx)))
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(train_idx.numel())
        for start in range(0, train_idx.numel(), batch_size):
            local = order[start:start + batch_size]
            bidx = train_idx.index_select(0, local)
            args = _feature_inputs(model, features, bidx, *norms)
            pred = model(**{k: v.to(dev) for k, v in args.items()})
            pred = pred if mode == "M" else pred.reshape(-1, 1)
            losses = []
            for j, target in enumerate(target_vectors):
                truth = target.index_select(0, bidx)
                valid = torch.isfinite(truth)
                if not valid.any():
                    continue
                scaled = ((truth[valid] - means[j]) / stds[j]).to(dev)
                point = F.smooth_l1_loss(pred[valid, j], scaled, reduction="none")
                weights = train_weights[j].index_select(0, local).to(dev)[valid]
                losses.append((point * weights).sum() / weights.sum().clamp_min(1.0e-8))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite pointwise loss at {dataset}/{estimator}/{mode}/{seed}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("nonfinite pointwise gradient")
            optimizer.step()
        val_loss = _val_pointwise_loss(model, features, val_idx, target_vectors, means, stds, norms, dev, mode == "M")
        if not math.isfinite(val_loss):
            raise FloatingPointError("nonfinite receiver-equal validation loss")
        if val_loss < best_loss - 1.0e-8:
            best_loss, best_epoch, best_state, wait = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, {"state_dict": model.state_dict(), "dataset": dataset, "estimator": estimator,
                    "seed": seed, "mode": mode, "target_names": target_names, "best_epoch": best_epoch,
                    "best_val_receiver_equal_huber": best_loss, "scalar_mean": norms[0], "scalar_std": norms[1],
                    "host_mean": norms[2], "host_std": norms[3], "target_means": means, "target_stds": stds})
    pred_std = _predict(model, features, hold_idx, norms, dev)
    if mode == "S":
        pred_std = pred_std.reshape(-1, 1)
    predictions: dict[str, list[float]] = {}
    for j, name in enumerate(target_names):
        pred = (pred_std[:, j] * stds[j] + means[j]).tolist()
        predictions[name] = pred
    return {"model": model, "predictions": predictions, "holdout_indices": hold_idx,
            "train_indices": train_idx, "val_indices": val_idx, "split": split, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "target_names": target_names, "targets": target_map,
            "norms": norms, "means": means, "stds": stds}


def _predict_pair_scores(model: torch.nn.Module, features: dict[str, torch.Tensor], indices: torch.Tensor,
                         norms: tuple[torch.Tensor, ...], device: torch.device) -> torch.Tensor:
    if not indices.numel():
        return torch.empty(0)
    return _predict(model, features, indices, norms, device).reshape(-1)


def _rank_validation_loss(model: torch.nn.Module, features: dict[str, torch.Tensor], rows: list[dict[str, Any]],
                          target: torch.Tensor, receiver_set: set[int], norms: tuple[torch.Tensor, ...],
                          device: torch.device) -> float:
    pairs = make_edge_rank_pairs(rows, target, receiver_set)
    if not pairs["left"].numel():
        return float("inf")
    model.eval()
    endpoints = torch.cat((pairs["left"], pairs["right"]))
    with torch.no_grad():
        scores = _predict_pair_scores(model, features, endpoints, norms, device)
    n = pairs["left"].numel()
    loss = F.softplus(-pairs["order"] * (scores[n:] - scores[:n]))
    return float((loss * pairs["weight"]).sum() / pairs["weight"].sum().clamp_min(1.0e-12))


def train_ranker(dataset: str, estimator: str, target_name: str, seed: int,
                 features: dict[str, torch.Tensor], rows: list[dict[str, Any]], *,
                 device: str = "cuda:1", max_epochs: int = 100, patience: int = 10,
                 pair_batch_size: int = 4096, checkpoint_path: str | None = None) -> dict[str, Any]:
    """Fit a scalar scorer only from within-receiver/modality/hop edge comparisons."""
    dev = torch.device(device)
    features = {k: v.detach().cpu() for k, v in features.items()}
    idx, split = row_splits(features, dataset)
    train_idx, val_idx, hold_idx = idx["EstimatorTrain"], idx["EstimatorVal"], idx["EstimatorHoldout"]
    receiver = features["receiver_id"]
    train_receivers = set(int(x) for x in receiver.index_select(0, train_idx).tolist())
    val_receivers = set(int(x) for x in receiver.index_select(0, val_idx).tolist())
    if train_receivers & val_receivers:
        raise AssertionError("ranker validation receivers overlap training receivers")
    target = features[f"target_{target_name}"].float()
    train_pairs = make_edge_rank_pairs(rows, target, train_receivers)
    val_pairs = make_edge_rank_pairs(rows, target, val_receivers)
    if not train_pairs["left"].numel() or not val_pairs["left"].numel():
        raise ValueError("ranker train and validation splits must contain untied edge pairs")
    norms = _normalizers(features, train_idx)
    torch.manual_seed(53000 + int(seed))
    np.random.seed(53000 + int(seed))
    model = SingleTargetMessageEffectEstimator(estimator, features["scalar"].size(-1),
            host_dim=features["host"].size(-1) if estimator == "E3_host_context" else None).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pair_count = train_pairs["left"].numel()
    group_count = int(train_pairs["group_id"].unique().numel())
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(pair_count)
        for start in range(0, pair_count, pair_batch_size):
            pick = order[start:start + pair_batch_size]
            left = train_pairs["left"].index_select(0, pick)
            right = train_pairs["right"].index_select(0, pick)
            endpoints = torch.cat((left, right))
            args = _feature_inputs(model, features, endpoints, *norms)
            score = model(**{k: v.to(dev) for k, v in args.items()}).reshape(-1)
            n = left.numel()
            loss_each = ranknet_pair_loss(score[:n], score[n:], train_pairs["order"].index_select(0, pick).to(dev))
            weights = train_pairs["weight"].index_select(0, pick).to(dev)
            # Uniform pair batches give an unbiased estimate of the equal-weighted group objective.
            loss = (loss_each * weights).sum() * (pair_count / n) / max(group_count, 1)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite RankNet loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("nonfinite RankNet gradient")
            optimizer.step()
        val_loss = _rank_validation_loss(model, features, rows, target, val_receivers, norms, dev)
        if not math.isfinite(val_loss):
            raise FloatingPointError("nonfinite receiver-group-equal RankNet validation loss")
        if val_loss < best_loss - 1.0e-8:
            best_loss, best_epoch, best_state, wait = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("no finite ranker validation checkpoint was selected")
    model.load_state_dict(best_state)
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, {"state_dict": model.state_dict(), "dataset": dataset, "estimator": estimator,
                    "target": target_name, "seed": seed, "best_epoch": best_epoch,
                    "best_val_receiver_group_ranknet": best_loss, "scalar_mean": norms[0],
                    "scalar_std": norms[1], "host_mean": norms[2], "host_std": norms[3]})
    holdout_receivers = set(int(x) for x in receiver.index_select(0, hold_idx).tolist())
    hold_pairs = make_edge_rank_pairs(rows, target, holdout_receivers)
    all_hold_indices = torch.tensor(sorted(holdout_receivers), dtype=torch.long)
    row_index = {int(r): [] for r in holdout_receivers}
    for i, r in enumerate(receiver.tolist()):
        if int(r) in holdout_receivers:
            row_index[int(r)].append(i)
    hold_row_idx = torch.tensor([i for r in sorted(row_index) for i in row_index[r]], dtype=torch.long)
    scores = _predict_pair_scores(model, features, hold_row_idx, norms, dev)
    row_score_map = {int(idx_value): float(score) for idx_value, score in zip(hold_row_idx.tolist(), scores.tolist())}
    predictions = [float("nan") for _ in rows]
    for i, val in row_score_map.items():
        predictions[i] = val
    return {"model": model, "predictions": predictions, "holdout_indices": hold_idx,
            "train_indices": train_idx, "val_indices": val_idx, "split": split,
            "best_epoch": best_epoch, "best_validation_loss": best_loss, "target": target,
            "train_pair_count": pair_count, "train_pair_group_count": group_count,
            "holdout_pairs": hold_pairs}


def _global_metrics(truth: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if not y.size:
        return {k: float("nan") for k in ("mae", "rmse", "r2", "spearman", "pearson", "harmful_auroc", "harmful_auprc", "harmful_prevalence")}
    harmful = y < 0
    valid = len(np.unique(harmful)) == 2
    return {
        "mae": float(mean_absolute_error(y, p)), "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)) if y.size > 1 else float("nan"),
        "spearman": _safe_spearman(y.tolist(), p.tolist()), "pearson": _safe_pearson(y.tolist(), p.tolist()),
        "harmful_auroc": float(roc_auc_score(harmful, -p)) if valid else float("nan"),
        "harmful_auprc": float(average_precision_score(harmful, -p)) if valid else float("nan"),
        "harmful_prevalence": float(np.mean(harmful)),
    }


def pointwise_group_metrics(rows: Sequence[dict[str, Any]], truth: Sequence[float], pred: Sequence[float],
                            *, dataset: str, target: str, model: str, estimator: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    definitions = (
        ("edge", lambda r: (int(r["receiver_id"]), str(r["modality"]), int(r["hop"]))),
        ("modality", lambda r: (int(r["receiver_id"]), int(r["directed_edge_index"]), int(r["hop"]))),
        ("hop", lambda r: (int(r["receiver_id"]), int(r["directed_edge_index"]), str(r["modality"]))),
    )
    outputs: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for kind, keyfn in definitions:
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            if math.isfinite(float(truth[i])) and math.isfinite(float(pred[i])):
                groups[keyfn(row)].append(i)
        pair_values: list[float] = []
        rhos: list[float] = []
        ordering: list[float] = []
        sign_prediction: list[float] = []
        valid_group_count = 0
        for key, ix in groups.items():
            if len(ix) < 2:
                continue
            if kind == "modality" and len(ix) != 2:
                continue
            if kind == "hop" and len(ix) != 4:
                continue
            y = np.asarray([truth[i] for i in ix], dtype=float)
            p = np.asarray([pred[i] for i in ix], dtype=float)
            valid_group_count += 1
            rho = _safe_spearman(y.tolist(), p.tolist())
            eligible, _, pair_acc = pairwise_agreement(y, p, 0.0)
            if math.isfinite(rho):
                rhos.append(rho)
            if eligible:
                pair_values.append(pair_acc)
            group = {"dataset": dataset, "target": target, "model": model, "estimator": estimator,
                     "seed": seed, "group_type": kind, "group_key": ":".join(map(str, key)), "group_size": len(ix),
                     "spearman": rho, "pairwise_accuracy": pair_acc}
            if kind == "edge":
                group["top1_agreement"] = float(y[int(np.argmin(p))] == np.min(y))
                group["top1_chance"] = 1.0 / len(ix)
                group["top1_excess"] = group["top1_agreement"] - group["top1_chance"]
            if kind == "modality" and len(ix) == 2:
                true_order = np.sign(y[1] - y[0]); pred_order = np.sign(p[1] - p[0])
                order_acc = 1.0 if true_order == pred_order else (0.5 if true_order == 0 or pred_order == 0 else 0.0)
                true_sign_diff = (y[0] < 0) != (y[1] < 0)
                pred_sign_diff = (p[0] < 0) != (p[1] < 0)
                group["modality_ordering_accuracy"] = order_acc
                group["true_sign_disagreement"] = float(true_sign_diff)
                group["predicted_sign_disagreement_accuracy"] = float(true_sign_diff == pred_sign_diff)
                ordering.append(order_acc); sign_prediction.append(float(true_sign_diff == pred_sign_diff))
            outputs.append(group)
        aggregate[f"{kind}_groups_n"] = valid_group_count
        aggregate[f"{kind}_mean_group_spearman"] = float(np.mean(rhos)) if rhos else float("nan")
        aggregate[f"{kind}_median_group_spearman"] = float(np.median(rhos)) if rhos else float("nan")
        aggregate[f"{kind}_pairwise_accuracy"] = float(np.mean(pair_values)) if pair_values else float("nan")
        if kind == "edge":
            edge_rows = [x for x in outputs if x["group_type"] == "edge"]
            aggregate["edge_top1_agreement"] = float(np.mean([x["top1_agreement"] for x in edge_rows])) if edge_rows else float("nan")
            aggregate["edge_top1_excess"] = float(np.mean([x["top1_excess"] for x in edge_rows])) if edge_rows else float("nan")
        if kind == "modality":
            aggregate["modality_ordering_accuracy"] = float(np.mean(ordering)) if ordering else float("nan")
            aggregate["predicted_modality_sign_disagreement_accuracy"] = float(np.mean(sign_prediction)) if sign_prediction else float("nan")
    return outputs, aggregate


def pointwise_metrics(rows: Sequence[dict[str, Any]], truth: Sequence[float], pred: Sequence[float], **meta: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metric = {**meta, "n_holdout_rows": int(sum(math.isfinite(float(x)) for x in truth)), **_global_metrics(truth, pred)}
    groups, group_summary = pointwise_group_metrics(rows, truth, pred, **meta)
    metric.update(group_summary)
    selection, selection_summary = selection_group_rows(rows, pred, str(meta["target"]), model_name=str(meta["model"]),
                                                        dataset=str(meta["dataset"]), estimator=str(meta["estimator"]), seed=int(meta["seed"]))
    metric.update({f"selection_{k}": v for k, v in selection_summary.items()})
    return metric, groups, selection


def ranker_metrics(rows: Sequence[dict[str, Any]], truth: Sequence[float], pred: Sequence[float], **meta: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    edge_spearman, pairwise, top1, top_excess = [], [], [], []
    group_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if math.isfinite(float(truth[i])) and math.isfinite(float(pred[i])):
            grouped[(int(row["receiver_id"]), str(row["modality"]), int(row["hop"]))].append(i)
    for key, ix in grouped.items():
        if len(ix) < 2:
            continue
        y = np.asarray([truth[i] for i in ix]); p = np.asarray([pred[i] for i in ix])
        rho = _safe_spearman(y.tolist(), p.tolist())
        n, _, acc = pairwise_agreement(y.tolist(), p.tolist(), 0.0)
        agree = float(y[int(np.argmin(p))] == np.min(y))
        chance = 1.0 / len(ix)
        if math.isfinite(rho): edge_spearman.append(rho)
        if n: pairwise.append(acc)
        top1.append(agree); top_excess.append(agree - chance)
        group_rows.append({**meta, "receiver_id": key[0], "modality": key[1], "hop": key[2], "group_size": len(ix),
                           "edge_group_spearman": rho, "edge_pairwise_accuracy": acc,
                           "top1_agreement": agree, "top1_chance": chance, "top1_excess": agree - chance})
    metric = {**meta, "n_edge_groups": len(top1), "edge_group_spearman_mean": float(np.mean(edge_spearman)) if edge_spearman else float("nan"),
              "edge_group_spearman_median": float(np.median(edge_spearman)) if edge_spearman else float("nan"),
              "edge_pairwise_accuracy": float(np.mean(pairwise)) if pairwise else float("nan"),
              "edge_pairwise_excess_over_0.5": float(np.mean(pairwise) - 0.5) if pairwise else float("nan"),
              "edge_top1_agreement": float(np.mean(top1)) if top1 else float("nan"),
              "edge_top1_excess_over_chance": float(np.mean(top_excess)) if top_excess else float("nan")}
    # A rank score has no cross-group calibration; deliberately no global/modality/hop metrics here.
    return metric, group_rows


def ranker_sensitivity(rows: Sequence[dict[str, Any]], truth: Sequence[float], pred: Sequence[float], taus: Sequence[float], **meta: Any) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if math.isfinite(float(truth[i])) and math.isfinite(float(pred[i])):
            grouped[(int(row["receiver_id"]), str(row["modality"]), int(row["hop"]))].append(i)
    all_pairs = [(a, b) for members in grouped.values() for a, b in __import__("itertools").combinations(members, 2)]
    output = []
    for tau in taus:
        pairs = [(a, b) for a, b in all_pairs if abs(float(truth[a]) - float(truth[b])) >= tau]
        pair_scores = []
        for a, b in pairs:
            true_diff = float(truth[a]) - float(truth[b])
            pred_diff = float(pred[a]) - float(pred[b])
            pair_scores.append(1.0 if true_diff == 0 and pred_diff == 0 else
                               (0.5 if true_diff == 0 or pred_diff == 0 else float((true_diff < 0) == (pred_diff < 0))))
        acc = float(np.mean(pair_scores)) if pair_scores else float("nan")
        output.append({**meta, "tau": tau, "candidate_pairs": len(all_pairs), "eligible_pairs": len(pairs),
                       "eligible_pair_fraction": len(pairs) / len(all_pairs) if all_pairs else float("nan"),
                       "pairwise_accuracy": acc, "pairwise_excess_over_0.5": acc - 0.5 if math.isfinite(acc) else float("nan")})
    return output
