from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .message_effect_estimator import MessageEffectEstimator
from .message_effect_replay import estimator_receiver_split
from .message_effect_atlas import _safe_pearson, _safe_spearman

ESTIMATORS = (
    "E0_heuristic",
    "E1_unimodal_pair",
    "E2_multimodal_pair",
    "E3_host_context",
)


def _feature_inputs(
    model: MessageEffectEstimator,
    features: dict[str, torch.Tensor],
    indices: torch.Tensor,
    *,
    scalar_mean: torch.Tensor,
    scalar_std: torch.Tensor,
    host_mean: torch.Tensor,
    host_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scalar = features["scalar"].index_select(0, indices)
    result: dict[str, torch.Tensor] = {
        "scalar_features": (scalar - scalar_mean) / scalar_std
    }
    if model.variant == "E1_unimodal_pair":
        modality = features["modality_code"].index_select(0, indices)
        pair_text = features["pair_text"].index_select(0, indices)
        pair_visual = features["pair_visual"].index_select(0, indices)
        pair = torch.where(modality[:, None] == 0, pair_text, pair_visual)
        result["pair_active"] = pair
    elif model.variant in {"E2_multimodal_pair", "E3_host_context"}:
        result["pair_text"] = features["pair_text"].index_select(0, indices)
        result["pair_visual"] = features["pair_visual"].index_select(0, indices)
        if model.variant == "E3_host_context":
            host = features["host"].index_select(0, indices)
            result["host_context"] = (host - host_mean) / host_std
    return result


def _make_row_indices(
    features: dict[str, torch.Tensor], dataset: str, seed: int = 2027
) -> tuple[dict[str, torch.Tensor], dict[str, set[int]]]:
    split = estimator_receiver_split(dataset, features["receiver_id"], seed=seed)
    result: dict[str, torch.Tensor] = {}
    receiver_values = features["receiver_id"].tolist()
    for name, receivers in split.items():
        result[name] = torch.tensor(
            [i for i, receiver in enumerate(receiver_values) if int(receiver) in receivers],
            dtype=torch.long,
        )
    return result, split


def _target_stats(target: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = target.index_select(0, indices)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return target.new_zeros(()), target.new_ones(())
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(1.0e-8)
    return mean, std


def _evaluate_loss(
    model: MessageEffectEstimator,
    features: dict[str, torch.Tensor],
    indices: torch.Tensor,
    target_delete: torch.Tensor,
    target_comp: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    scalar_mean: torch.Tensor,
    scalar_std: torch.Tensor,
    host_mean: torch.Tensor,
    host_std: torch.Tensor,
    device: torch.device,
) -> float:
    if indices.numel() == 0:
        return float("inf")
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), 2048):
            batch = indices[start : start + 2048]
            x = _feature_inputs(
                model,
                features,
                batch,
                scalar_mean=scalar_mean,
                scalar_std=scalar_std,
                host_mean=host_mean,
                host_std=host_std,
            )
            prediction = model(**{k: v.to(device) for k, v in x.items()})
            y_delete = ((target_delete.index_select(0, batch) - target_mean[0]) / target_std[0]).to(device)
            y_comp = ((target_comp.index_select(0, batch) - target_mean[1]) / target_std[1]).to(device)
            target = torch.stack((y_delete, y_comp), dim=-1)
            valid = torch.isfinite(target)
            point_loss = F.smooth_l1_loss(
                prediction, target.nan_to_num(0.0), reduction="none"
            )
            masked = point_loss * valid
            count = valid.sum().clamp_min(1)
            losses.append(float((masked.sum() / count).item()))
    return float(np.mean(losses))


def _predict(
    model: MessageEffectEstimator,
    features: dict[str, torch.Tensor],
    indices: torch.Tensor,
    scalar_mean: torch.Tensor,
    scalar_std: torch.Tensor,
    host_mean: torch.Tensor,
    host_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), 2048):
            batch = indices[start : start + 2048]
            x = _feature_inputs(
                model,
                features,
                batch,
                scalar_mean=scalar_mean,
                scalar_std=scalar_std,
                host_mean=host_mean,
                host_std=host_std,
            )
            pred = model(**{k: v.to(device) for k, v in x.items()})
            pieces.append(pred.detach().cpu())
    if not pieces:
        return torch.empty((0, 2), dtype=torch.float32)
    standardized = torch.cat(pieces)
    return standardized * target_std.cpu() + target_mean.cpu()


def _ranking_metrics(
    rows: list[dict[str, Any]],
    true_values: list[float],
    predicted_values: list[float],
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    usable = [
        (i, row)
        for i, row in enumerate(rows)
        if math.isfinite(true_values[i]) and math.isfinite(predicted_values[i])
    ]
    definitions = (
        ("edge", lambda r: (r["receiver_id"], r["modality"], r["hop"])),
        ("modality", lambda r: (r["receiver_id"], r["directed_edge_index"], r["hop"])),
        ("hop", lambda r: (r["receiver_id"], r["directed_edge_index"], r["modality"])),
    )
    for group_type, key_fn in definitions:
        groups: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, row in usable:
            groups[key_fn(row)].append((index, row))
        spearmans: list[float] = []
        pairwise: list[float] = []
        top1: list[float] = []
        ordering: list[float] = []
        sign_disagreement: list[bool] = []
        predicted_sign_disagreement: list[bool] = []
        sign_disagreement_match: list[float] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            truth = [true_values[i] for i, _ in group]
            pred = [predicted_values[i] for i, _ in group]
            rho = _safe_spearman(truth, pred)
            if math.isfinite(rho):
                spearmans.append(rho)
            pair_agree = []
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    truth_diff = truth[a] - truth[b]
                    pred_diff = pred[a] - pred[b]
                    pair_agree.append(
                        1.0 if truth_diff == pred_diff == 0 else
                        (0.5 if truth_diff == 0 or pred_diff == 0 else float((truth_diff < 0) == (pred_diff < 0)))
                    )
            pairwise.extend(pair_agree)
            if group_type == "edge":
                true_top = min(range(len(truth)), key=lambda i: truth[i])
                pred_top = min(range(len(pred)), key=lambda i: pred[i])
                top1.append(float(true_top == pred_top))
            elif group_type == "modality" and len(group) == 2:
                true_order = (truth[0] < truth[1]) - (truth[0] > truth[1])
                pred_order = (pred[0] < pred[1]) - (pred[0] > pred[1])
                ordering.append(
                    1.0 if true_order == pred_order else
                    (0.5 if true_order == 0 or pred_order == 0 else 0.0)
                )
                true_diff = (truth[0] < 0) != (truth[1] < 0)
                pred_diff = (pred[0] < 0) != (pred[1] < 0)
                sign_disagreement.append(true_diff)
                predicted_sign_disagreement.append(pred_diff)
                sign_disagreement_match.append(float(true_diff == pred_diff))
        metric_rows.append(
            {
                "group_type": group_type,
                "groups_n": sum(len(g) >= 2 for g in groups.values()),
                "mean_group_spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
                "median_group_spearman": float(np.median(spearmans)) if spearmans else float("nan"),
                "pairwise_ranking_accuracy": float(np.mean(pairwise)) if pairwise else float("nan"),
                "top1_rank_agreement": float(np.mean(top1)) if top1 else float("nan"),
                "modality_ordering_accuracy": float(np.mean(ordering)) if ordering else float("nan"),
                "true_modality_sign_disagreement_fraction": (
                    float(np.mean(sign_disagreement)) if sign_disagreement else float("nan")
                ),
                "predicted_modality_sign_disagreement_accuracy": (
                    float(np.mean(sign_disagreement_match)) if sign_disagreement_match else float("nan")
                ),
            }
        )
    return metric_rows


def _global_metrics(truth: list[float], prediction: list[float]) -> dict[str, float]:
    if not truth:
        return {key: float("nan") for key in (
            "mae", "rmse", "r2", "spearman", "pearson", "harmful_auroc",
            "harmful_auprc", "harmful_prevalence",
        )}
    harmful = [float(value < 0) for value in truth]
    valid_classes = len(set(harmful)) == 2
    scores: dict[str, float] = {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else float("nan"),
        "spearman": _safe_spearman(truth, prediction),
        "pearson": _safe_pearson(truth, prediction),
        "harmful_auroc": (
            float(roc_auc_score(harmful, [-x for x in prediction])) if valid_classes else float("nan")
        ),
        "harmful_auprc": (
            float(average_precision_score(harmful, [-x for x in prediction]))
            if valid_classes else float("nan")
        ),
        "harmful_prevalence": float(np.mean(harmful)),
    }
    return scores


def train_evaluate_estimators(
    dataset: str,
    features: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    *,
    device: str = "cuda:1",
    seeds: tuple[int, ...] = (0, 1, 2),
    max_epochs: int = 100,
    patience: int = 10,
    checkpoint_dir: Path | None = None,
    return_prediction_rows: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Train fixed E0-E3 models on receiver-disjoint train-only effect targets."""
    device_t = torch.device(device)
    if device_t.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested estimator device is unavailable: {device}")
    features = {k: v.detach().cpu() for k, v in features.items()}
    if len(rows) != features["receiver_id"].numel():
        raise ValueError("row table and feature cache are misaligned")
    target_delete = features["target_delete"].float()
    target_comp = features["target_comp"].float()
    estimator_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_indices_by_seed: dict[int, dict[str, torch.Tensor]] = {}
    split_sets_by_seed: dict[int, dict[str, set[int]]] = {}
    for seed in seeds:
        split_indices_by_seed[seed], split_sets_by_seed[seed] = _make_row_indices(
            features, dataset, 2027
        )
        sets = list(split_sets_by_seed[seed].values())
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
            raise AssertionError("estimator receiver partitions overlap")
    if any(not split for split in split_sets_by_seed.values()):
        raise ValueError("stable estimator split produced an empty receiver partition")

    scalar_width = features["scalar"].size(-1)
    host_width = features["host"].size(-1)
    for estimator in ESTIMATORS:
        for seed in seeds:
            torch.manual_seed(53000 + seed)
            np.random.seed(53000 + seed)
            split_indices = split_indices_by_seed[seed]
            train_idx = split_indices["EstimatorTrain"]
            val_idx = split_indices["EstimatorVal"]
            holdout_idx = split_indices["EstimatorHoldout"]
            scalar_train = features["scalar"].index_select(0, train_idx)
            scalar_mean = scalar_train.mean(dim=0)
            scalar_std = scalar_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
            host_train = features["host"].index_select(0, train_idx)
            host_mean = host_train.mean(dim=0)
            host_std = host_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
            delete_mean, delete_std = _target_stats(target_delete, train_idx)
            comp_mean, comp_std = _target_stats(target_comp, train_idx)
            target_mean = torch.stack((delete_mean, comp_mean))
            target_std = torch.stack((delete_std, comp_std))
            model = MessageEffectEstimator(
                estimator,
                scalar_width,
                host_dim=host_width if estimator == "E3_host_context" else None,
            ).to(device_t)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
            train_receivers = features["receiver_id"].index_select(0, train_idx).tolist()
            counts: dict[int, int] = defaultdict(int)
            for receiver in train_receivers:
                counts[int(receiver)] += 1
            row_weights = torch.tensor(
                [1.0 / counts[int(receiver)] for receiver in train_receivers],
                dtype=torch.float32,
            )
            best_loss = float("inf")
            best_epoch = 0
            best_state = None
            wait = 0
            completed_epochs = 0
            for epoch in range(1, max_epochs + 1):
                model.train()
                order = torch.randperm(train_idx.numel())
                for start in range(0, train_idx.numel(), 2048):
                    local = order[start : start + 2048]
                    batch_idx = train_idx.index_select(0, local)
                    x = _feature_inputs(
                        model,
                        features,
                        batch_idx,
                        scalar_mean=scalar_mean,
                        scalar_std=scalar_std,
                        host_mean=host_mean,
                        host_std=host_std,
                    )
                    prediction = model(**{k: v.to(device_t) for k, v in x.items()})
                    y_delete = (
                        target_delete.index_select(0, batch_idx) - delete_mean
                    ) / delete_std
                    y_comp = (
                        target_comp.index_select(0, batch_idx) - comp_mean
                    ) / comp_std
                    targets = torch.stack((y_delete, y_comp), dim=-1).to(device_t)
                    valid = torch.isfinite(targets)
                    weights = row_weights.index_select(0, local).to(device_t)[:, None]
                    point_loss = F.smooth_l1_loss(
                        prediction, targets.nan_to_num(0.0), reduction="none"
                    )
                    weighted = point_loss * valid * weights
                    denominator = (valid * weights).sum().clamp_min(1.0)
                    loss = weighted.sum() / denominator
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                completed_epochs = epoch
                val_loss = _evaluate_loss(
                    model,
                    features,
                    val_idx,
                    target_delete,
                    target_comp,
                    target_mean,
                    target_std,
                    scalar_mean,
                    scalar_std,
                    host_mean,
                    host_std,
                    device_t,
                )
                if val_loss < best_loss - 1.0e-7:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
            if best_state is None:
                raise RuntimeError(f"{estimator}/{seed}: no finite validation checkpoint")
            model.load_state_dict(best_state)
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "dataset": dataset,
                        "estimator": estimator,
                        "seed": seed,
                        "best_epoch": best_epoch,
                    },
                    checkpoint_dir / f"{dataset}_{estimator}_seed{seed}.pt",
                )
            prediction = _predict(
                model,
                features,
                holdout_idx,
                scalar_mean,
                scalar_std,
                host_mean,
                host_std,
                target_mean,
                target_std,
                device_t,
            )
            holdout_receivers = features["receiver_id"].index_select(0, holdout_idx).tolist()
            receiver_split = split_sets_by_seed[seed]
            if set(int(x) for x in holdout_receivers) != receiver_split["EstimatorHoldout"]:
                raise AssertionError("holdout rows do not match the declared receiver split")
            indices = holdout_idx.tolist()
            target_specs = (
                ("delete", target_delete, 0),
                ("comp", target_comp, 1),
            )
            for target_name, target, output_index in target_specs:
                target_values = target.index_select(0, holdout_idx).tolist()
                predictions = prediction[:, output_index].tolist()
                pairs = [
                    (float(a), float(b), rows[row_index])
                    for a, b, row_index in zip(target_values, predictions, indices)
                    if math.isfinite(float(a)) and math.isfinite(float(b))
                ]
                truth = [x[0] for x in pairs]
                pred = [x[1] for x in pairs]
                metric = _global_metrics(truth, pred)
                rank = _ranking_metrics(
                    [x[2] for x in pairs], truth, pred
                )
                rank_map = {r["group_type"]: r for r in rank}
                edge_rank = rank_map["edge"]
                modality_rank = rank_map["modality"]
                hop_rank = rank_map["hop"]
                estimator_rows.append(
                    {
                        "dataset": dataset,
                        "estimator": estimator,
                        "seed": seed,
                        "target": target_name,
                        "n_holdout_rows": len(truth),
                        "n_holdout_receivers": len(receiver_split["EstimatorHoldout"]),
                        "best_epoch": best_epoch,
                        **metric,
                        "edge_ranking_spearman_mean": edge_rank["mean_group_spearman"],
                        "edge_ranking_spearman_median": edge_rank["median_group_spearman"],
                        "edge_pairwise_accuracy": edge_rank["pairwise_ranking_accuracy"],
                        "edge_top1_agreement": edge_rank["top1_rank_agreement"],
                        "modality_ordering_accuracy": modality_rank["modality_ordering_accuracy"],
                        "true_modality_sign_disagreement_fraction": modality_rank["true_modality_sign_disagreement_fraction"],
                        "predicted_modality_sign_disagreement_accuracy": modality_rank["predicted_modality_sign_disagreement_accuracy"],
                        "hop_ranking_spearman_mean": hop_rank["mean_group_spearman"],
                        "hop_pairwise_accuracy": hop_rank["pairwise_ranking_accuracy"],
                    }
                )
                for group_metric in rank:
                    group_rows.append(
                        {
                            "dataset": dataset,
                            "estimator": estimator,
                            "seed": seed,
                            "target": target_name,
                            **group_metric,
                        }
                    )
                if return_prediction_rows:
                    for (true_value, predicted_value, row) in pairs:
                        prediction_rows.append(
                            {
                                "dataset": dataset,
                                "estimator": estimator,
                                "seed": seed,
                                "target": target_name,
                                "receiver_id": row["receiver_id"],
                                "sender_id": row["sender_id"],
                                "directed_edge_index": row["directed_edge_index"],
                                "modality": row["modality"],
                                "hop": row["hop"],
                                "truth": true_value,
                                "prediction": predicted_value,
                            }
                        )
            del model, optimizer, prediction
            if device_t.type == "cuda":
                torch.cuda.empty_cache()
    return estimator_rows, group_rows, prediction_rows


def estimator_comparisons(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    pairs = (
        ("E1_unimodal_pair", "E0_heuristic"),
        ("E2_multimodal_pair", "E1_unimodal_pair"),
        ("E3_host_context", "E2_multimodal_pair"),
    )
    fields = (
        "spearman",
        "mae",
        "rmse",
        "edge_ranking_spearman_mean",
        "edge_pairwise_accuracy",
        "edge_top1_agreement",
        "modality_ordering_accuracy",
        "hop_ranking_spearman_mean",
        "hop_pairwise_accuracy",
    )
    key_to_row = {
        (r["dataset"], r["target"], r["estimator"], int(r["seed"])): r for r in metrics
    }
    datasets = sorted({r["dataset"] for r in metrics})
    targets = sorted({r["target"] for r in metrics})
    for dataset in datasets:
        for target in targets:
            for comparison, reference in pairs:
                for field in fields:
                    differences = []
                    for seed in (0, 1, 2):
                        left = key_to_row.get((dataset, target, comparison, seed))
                        right = key_to_row.get((dataset, target, reference, seed))
                        if left is None or right is None:
                            continue
                        a, b = left[field], right[field]
                        if math.isfinite(float(a)) and math.isfinite(float(b)):
                            differences.append(float(a) - float(b))
                    comparisons.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "comparison": f"{comparison}-{reference}",
                            "metric": field,
                            "n_seeds": len(differences),
                            "mean_difference": float(np.mean(differences)) if differences else float("nan"),
                            "std_difference": float(np.std(differences)) if differences else float("nan"),
                        }
                    )
    return comparisons
