#!/usr/bin/env python3
"""Validation-only analysis of RM-CGE V2.1 Stable Adaptation checkpoints."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_rmcge_v21_classes import metrics_for
from scripts.run_rmcge_v21 import DATASETS, SEEDS, build_config, data_info, sha256_file, verify_cache_entries
from scripts.run_rmcge_v21_stable_adaptation import OUTPUT
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _resolve_nc_eval_labels

STUDY = ROOT / "research/rmcge_v21_stable_adaptation"
DATA = STUDY / "data"
WARM_CONDITIONS = ("W0", "W1", "W2")
CONDITION_MAP = {"W0": "C0", "W1": "C1", "W2": "C2"}
CONTRASTS = (("W1-W0", "W1", "W0"), ("W2-W1", "W2", "W1"), ("W2-W0", "W2", "W0"))
CHECKPOINTS = ("best_accuracy", "best_f1")
DEGREE_GROUPS = (
    ("degree_0", 0, 0), ("degree_1", 1, 1), ("degree_2_3", 2, 3),
    ("degree_4_7", 4, 7), ("degree_ge8", 8, None), ("degree_lt2", 0, 1), ("degree_ge2", 2, None),
)


def _write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                normalized[field] = value
            writer.writerow(normalized)


def _group_mask(degree: torch.Tensor, lower: int, upper: int | None) -> torch.Tensor:
    result = degree >= lower
    if upper is not None:
        result &= degree <= upper
    return result


def _quantile(values: torch.Tensor, q: float) -> float | None:
    return float(torch.quantile(values.float(), q)) if values.numel() else None


def _verify_run_index() -> dict[tuple[str, int, str], dict[str, str]]:
    path = DATA / "runs.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_key = {(row["dataset"], int(row["seed"]), row["condition"]): row for row in rows}
    expected = {(dataset, seed, condition) for dataset in DATASETS for seed in SEEDS for condition in WARM_CONDITIONS}
    if set(by_key) != expected:
        raise RuntimeError(f"warm-start run matrix incomplete: missing={sorted(expected-set(by_key))}")
    if any(row["status"] != "completed_and_verified" or row["test_evaluation"].lower() != "false" for row in rows):
        raise RuntimeError("run registry includes a failure or Test-enabled run")
    return by_key


@torch.no_grad()
def analyze(device: str = "cuda:0") -> dict[str, Any]:
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this validation checkpoint audit")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    caches = verify_cache_entries()
    run_index = _verify_run_index()
    global_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    cold_warm_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    split_fingerprints: dict[str, Any] = {}

    for dataset in DATASETS:
        for seed in SEEDS:
            cfg0 = build_config(dataset, "C0", seed, device, caches[dataset])
            data = load_mag_data(cfg0, "nc", seed)
            eval_labels = _resolve_nc_eval_labels(data, include_test=False)
            val_idx_cpu = data.val_idx.detach().cpu().long()
            val_idx = val_idx_cpu.to(device)
            y_val = data.y[val_idx_cpu].detach().cpu().long()
            x, edge = data.x.to(device), data.edge_index.to(device)
            physical_degree = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            physical_degree.index_add_(0, edge[1], torch.ones(edge.size(1), dtype=torch.long, device=device))
            split_key = f"{dataset}/seed_{seed}"
            split_fingerprints[split_key] = {
                "feature_sha256": __import__("scripts.run_rmcge_v21", fromlist=["tensor_fingerprint"]).tensor_fingerprint(data.x),
                "physical_edge_index_sha256": __import__("scripts.run_rmcge_v21", fromlist=["tensor_fingerprint"]).tensor_fingerprint(data.edge_index),
                "train_index_sha256": __import__("scripts.run_rmcge_v21", fromlist=["tensor_fingerprint"]).tensor_fingerprint(data.train_idx),
                "validation_index_sha256": __import__("scripts.run_rmcge_v21", fromlist=["tensor_fingerprint"]).tensor_fingerprint(data.val_idx),
                "semantic_cache_sha256": sha256_file(Path(caches[dataset]["cache_path_abs"])),
                "test_label_values_indexed": False,
            }
            predictions: dict[tuple[str, str], dict[str, Any]] = {}
            logits_by_key: dict[tuple[str, str], torch.Tensor] = {}
            for condition in WARM_CONDITIONS:
                run = run_index[(dataset, seed, condition)]
                run_path = ROOT / run["result_path"]
                result = json.loads(run_path.read_text(encoding="utf-8"))
                if result.get("status") != "completed_and_verified" or result.get("test_evaluation"):
                    raise RuntimeError(f"unverified warm-start run {dataset}/{condition}/{seed}")
                for checkpoint in CHECKPOINTS:
                    checkpoint_path = ROOT / run["best_accuracy_checkpoint" if checkpoint == "best_accuracy" else "best_f1_checkpoint"]
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(checkpoint_path)
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if any(key.startswith("test_") for key in payload.get("metrics", {})):
                        raise RuntimeError(f"Test metric exists in {checkpoint_path}")
                    if payload.get("run_metadata", {}).get("source_checkpoint_sha256") != run["source_checkpoint_sha256"]:
                        raise RuntimeError(f"source checkpoint lineage differs for {dataset}/{condition}/{seed}")
                    if payload["run_metadata"].get("test_evaluation") is not False:
                        raise RuntimeError(f"Test evaluation was enabled in checkpoint metadata {checkpoint_path}")
                    cfg = build_config(dataset, CONDITION_MAP[condition], seed, device, caches[dataset])
                    model = build_model(cfg, data_info(data))
                    model.load_semantic_cache(
                        caches[dataset]["cache_path_abs"], expected_dataset=dataset,
                        expected_feature_fingerprint=caches[dataset]["feature_fingerprint"],
                        expected_candidate_fingerprint=caches[dataset]["candidate_fingerprint"],
                        expected_v7b_fingerprint=caches[dataset]["v7b_fingerprint"],
                    )
                    model.load_state_dict(payload["model_state"], strict=True)
                    model.to(device).eval()
                    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                    classifier.load_state_dict(payload["head_state"], strict=True)
                    embeddings = model(x, edge)[0]
                    val_logits = classifier(embeddings[val_idx]).float()
                    values = metrics_for(val_logits.cpu(), y_val, eval_labels)
                    saved = payload["metrics"]
                    for output_name, saved_name in (("acc", "val_acc"), ("macro_f1", "val_macro_f1"), ("ce", "val_ce")):
                        if abs(values[output_name] - float(saved[saved_name])) > 2e-5:
                            raise RuntimeError(f"reloaded warm checkpoint metric mismatch {dataset}/{condition}/{seed}/{checkpoint}/{output_name}")
                    key = (condition, checkpoint)
                    predictions[key] = values
                    logits_by_key[key] = val_logits.detach().cpu()
                    global_rows.append({
                        "dataset": dataset, "seed": seed, "condition": condition, "checkpoint": checkpoint,
                        "selected_stage": payload["selected_stage"], "epoch": payload["epoch"],
                        "validation_count": int(y_val.numel()), "accuracy": values["acc"],
                        "macro_f1": values["macro_f1"], "macro_recall": values["macro_recall"],
                        "cross_entropy": values["ce"], "num_macro_classes": len(eval_labels),
                    })
                    for class_id, metric in values["class_metrics"].items():
                        class_rows.append({
                            "dataset": dataset, "seed": seed, "condition": condition, "checkpoint": checkpoint,
                            "row_type": "model", "contrast": "", "class_id": class_id, **metric,
                        })
                    for row_index, class_id in enumerate(eval_labels):
                        confusion_rows.append({
                            "dataset": dataset, "seed": seed, "condition": condition, "checkpoint": checkpoint,
                            "true_class": class_id,
                            **{f"pred_{target_class}": int(values["confusion"][row_index, col]) for col, target_class in enumerate(eval_labels)},
                        })
                    del model, classifier, embeddings, val_logits, payload
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

            for checkpoint in CHECKPOINTS:
                for contrast, right_condition, left_condition in CONTRASTS:
                    right = predictions[(right_condition, checkpoint)]
                    left = predictions[(left_condition, checkpoint)]
                    rpred, lpred, target = right["prediction"], left["prediction"], right["target"]
                    delta_ce = right["ce_each"] - left["ce_each"]
                    improved, worsened = delta_ce < 0, delta_ce > 0
                    transition_rows.append({
                        "dataset": dataset, "seed": seed, "checkpoint": checkpoint, "contrast": contrast,
                        "n_validation": int(target.numel()),
                        "left_correct_right_wrong": int(((lpred == target) & (rpred != target)).sum()),
                        "left_wrong_right_correct": int(((lpred != target) & (rpred == target)).sum()),
                        "prediction_flip_count": int((lpred != rpred).sum()),
                        "prediction_flip_fraction": float((lpred != rpred).float().mean()),
                        "ce_improved_count": int(improved.sum()), "ce_improved_fraction": float(improved.float().mean()),
                        "ce_improvement_mean": float((-delta_ce[improved]).mean()) if bool(improved.any()) else 0.0,
                        "ce_improvement_p90": _quantile(-delta_ce[improved], 0.9),
                        "ce_worsened_count": int(worsened.sum()), "ce_worsened_fraction": float(worsened.float().mean()),
                        "ce_worsening_mean": float(delta_ce[worsened].mean()) if bool(worsened.any()) else 0.0,
                        "ce_worsening_p90": _quantile(delta_ce[worsened], 0.9),
                        "ce_delta_mean_right_minus_left": float(delta_ce.mean()),
                        "ce_delta_p05": _quantile(delta_ce, 0.05), "ce_delta_p95": _quantile(delta_ce, 0.95),
                    })
                    for class_id in eval_labels:
                        rclass, lclass = right["class_metrics"][class_id], left["class_metrics"][class_id]
                        class_rows.append({
                            "dataset": dataset, "seed": seed, "condition": "", "checkpoint": checkpoint,
                            "row_type": "paired_delta", "contrast": contrast, "class_id": class_id,
                            "support": rclass["support"] - lclass["support"],
                            "predicted_count": rclass["predicted_count"] - lclass["predicted_count"],
                            "precision": rclass["precision"] - lclass["precision"],
                            "recall": rclass["recall"] - lclass["recall"],
                            "f1": rclass["f1"] - lclass["f1"],
                            "low_support_lt10": rclass["low_support_lt10"] or lclass["low_support_lt10"],
                        })
                    for metric, field in (("val_accuracy", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce")):
                        paired_rows.append({
                            "dataset": dataset, "seed": seed, "checkpoint": checkpoint,
                            "contrast": contrast, "metric": metric,
                            "left_value": left[field], "right_value": right[field],
                            "difference": right[field] - left[field],
                        })
                    for group, lower, upper in DEGREE_GROUPS:
                        val_degree = physical_degree[val_idx]
                        mask = _group_mask(val_degree, lower, upper).detach().cpu()
                        if not bool(mask.any()):
                            continue
                        for condition in WARM_CONDITIONS:
                            values = predictions[(condition, checkpoint)]
                            pred, sub_target = values["prediction"][mask], target[mask]
                            f1 = precision_recall_fscore_support(sub_target.numpy(), pred.numpy(), labels=eval_labels, average="macro", zero_division=0)[2]
                            degree_rows.append({
                                "dataset": dataset, "seed": seed, "checkpoint": checkpoint, "row_type": "model",
                                "group": group, "condition": condition, "contrast": "", "n_validation": int(mask.sum()),
                                "accuracy": float((pred == sub_target).float().mean()), "macro_f1": float(f1),
                                "cross_entropy": float(values["ce_each"][mask].mean()), "prediction_flip_fraction": 0.0,
                            })
                        lp, rp = left["prediction"][mask], right["prediction"][mask]
                        left_f1 = precision_recall_fscore_support(target[mask].numpy(), lp.numpy(), labels=eval_labels, average="macro", zero_division=0)[2]
                        right_f1 = precision_recall_fscore_support(target[mask].numpy(), rp.numpy(), labels=eval_labels, average="macro", zero_division=0)[2]
                        degree_rows.append({
                            "dataset": dataset, "seed": seed, "checkpoint": checkpoint, "row_type": "paired_delta",
                            "group": group, "condition": "", "contrast": contrast, "n_validation": int(mask.sum()),
                            "accuracy": float((rp == target[mask]).float().mean() - (lp == target[mask]).float().mean()),
                            "macro_f1": float(right_f1 - left_f1),
                            "cross_entropy": float(right["ce_each"][mask].mean() - left["ce_each"][mask].mean()),
                            "prediction_flip_fraction": float((rp != lp).float().mean()),
                        })

                # Common t0 improvements are the paired within-run warm adaptation deltas.
                for condition in WARM_CONDITIONS:
                    result = json.loads((ROOT / run_index[(dataset, seed, condition)]["result_path"]).read_text(encoding="utf-8"))
                    t0 = result["t0_metrics"]["validation"]
                    selected = result["best_accuracy_metrics"]["validation"] if checkpoint == "best_accuracy" else result["best_f1_metrics"]["validation"]
                    cold_warm_rows.append({
                        "dataset": dataset, "seed": seed, "condition": condition, "checkpoint": checkpoint,
                        "comparison": "selected_minus_common_t0", "metric": "val_accuracy",
                        "cold_or_t0_value": t0["accuracy"], "warm_value": selected["accuracy"], "difference": selected["accuracy"] - t0["accuracy"],
                        "causal_isolation": False,
                    })
                    for key, metric in (("val_macro_f1", "macro_f1"), ("val_ce", "cross_entropy")):
                        cold_warm_rows.append({
                            "dataset": dataset, "seed": seed, "condition": condition, "checkpoint": checkpoint,
                            "comparison": "selected_minus_common_t0", "metric": key,
                            "cold_or_t0_value": t0[metric], "warm_value": selected[metric],
                            "difference": selected[metric] - t0[metric], "causal_isolation": False,
                        })
            _append_warm_history(dataset, seed, epoch_rows)
            del data, x, edge, physical_degree, predictions, logits_by_key
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    _write_csv("warmstart_checkpoint_metrics.csv", global_rows, ["dataset", "seed", "condition", "checkpoint", "selected_stage", "epoch", "validation_count", "accuracy", "macro_f1", "macro_recall", "cross_entropy", "num_macro_classes"])
    _write_csv("warmstart_classwise_metrics.csv", class_rows, ["dataset", "seed", "condition", "checkpoint", "row_type", "contrast", "class_id", "support", "predicted_count", "precision", "recall", "f1", "low_support_lt10"])
    confusion_fields = ["dataset", "seed", "condition", "checkpoint", "true_class"] + sorted({key for row in confusion_rows for key in row if key.startswith("pred_")}, key=lambda key: int(key[5:]))
    _write_csv("warmstart_confusion_matrices.csv", confusion_rows, confusion_fields)
    _write_csv("warmstart_prediction_transitions.csv", transition_rows, ["dataset", "seed", "checkpoint", "contrast", "n_validation", "left_correct_right_wrong", "left_wrong_right_correct", "prediction_flip_count", "prediction_flip_fraction", "ce_improved_count", "ce_improved_fraction", "ce_improvement_mean", "ce_improvement_p90", "ce_worsened_count", "ce_worsened_fraction", "ce_worsening_mean", "ce_worsening_p90", "ce_delta_mean_right_minus_left", "ce_delta_p05", "ce_delta_p95"])
    _write_csv("warmstart_degree_group_metrics.csv", degree_rows, ["dataset", "seed", "checkpoint", "row_type", "group", "condition", "contrast", "n_validation", "accuracy", "macro_f1", "cross_entropy", "prediction_flip_fraction"])
    _write_csv("paired_comparisons.csv", paired_rows, ["dataset", "seed", "checkpoint", "contrast", "metric", "left_value", "right_value", "difference"])
    _write_paired_summary(paired_rows)
    _write_csv("warmstart_t0_deltas.csv", cold_warm_rows, ["dataset", "seed", "condition", "checkpoint", "comparison", "metric", "cold_or_t0_value", "warm_value", "difference", "causal_isolation"])
    _write_coldstart_warmstart_comparison(global_rows)
    _write_csv("warmstart_epoch_comparison.csv", epoch_rows, ["dataset", "condition", "seed", "stage", "epoch", "train_ce", "train_accuracy", "train_macro_f1", "val_ce", "val_accuracy", "val_macro_f1", "is_best_accuracy_epoch", "is_best_f1_epoch", "is_early_stop_epoch"])
    result = {
        "runs": len(run_index), "checkpoint_reloads": len(global_rows),
        "classwise_rows": len(class_rows), "prediction_transition_rows": len(transition_rows),
        "degree_group_rows": len(degree_rows), "paired_rows": len(paired_rows),
        "test_evaluation": False, "test_label_values_indexed": False,
        "split_fingerprints": split_fingerprints,
    }
    (DATA / "analysis_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _write_paired_summary(rows: list[dict[str, Any]]) -> None:
    summary = []
    for dataset in DATASETS:
        for checkpoint in CHECKPOINTS:
            for contrast, _, _ in CONTRASTS:
                for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                    matching = [row for row in rows if row["dataset"] == dataset and row["checkpoint"] == checkpoint and row["contrast"] == contrast and row["metric"] == metric]
                    deltas = [float(row["difference"]) for row in sorted(matching, key=lambda row: int(row["seed"]))]
                    summary.append({
                        "dataset": dataset, "checkpoint": checkpoint, "contrast": contrast, "metric": metric,
                        "mean_difference": statistics.mean(deltas),
                        "sample_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                        **{f"seed_{seed}": value for seed, value in zip(SEEDS, deltas)},
                        "positive_seed_count": sum(value > 0 for value in deltas),
                    })
    _write_csv("paired_comparison_summary.csv", summary, ["dataset", "checkpoint", "contrast", "metric", "mean_difference", "sample_sd", "seed_42", "seed_43", "seed_44", "positive_seed_count"])


def _write_coldstart_warmstart_comparison(warm_rows: list[dict[str, Any]]) -> None:
    with (DATA / "checkpoint_metrics.csv").open(encoding="utf-8", newline="") as stream:
        cold_rows = list(csv.DictReader(stream))
    cold_by_key = {(row["dataset"], int(row["seed"]), row["condition"], row["checkpoint"]): row for row in cold_rows}
    output = []
    for warm in warm_rows:
        cold_condition = CONDITION_MAP[warm["condition"]]
        cold = cold_by_key[(warm["dataset"], int(warm["seed"]), cold_condition, warm["checkpoint"])]
        for metric, cold_field, warm_field in (
            ("val_accuracy", "accuracy", "accuracy"),
            ("val_macro_f1", "macro_f1", "macro_f1"),
            ("val_ce", "cross_entropy", "cross_entropy"),
        ):
            cold_value, warm_value = float(cold[cold_field]), float(warm[warm_field])
            output.append({
                "dataset": warm["dataset"], "seed": warm["seed"], "cold_condition": cold_condition,
                "warm_condition": warm["condition"], "checkpoint": warm["checkpoint"], "metric": metric,
                "cold_value": cold_value, "warm_value": warm_value, "difference_warm_minus_cold": warm_value - cold_value,
                "interpretation": "descriptive_only_init_budget_lr_optimizer_state_and_selection_all_differ",
                "causal_isolation": False,
            })
    _write_csv("coldstart_warmstart_comparison.csv", output, ["dataset", "seed", "cold_condition", "warm_condition", "checkpoint", "metric", "cold_value", "warm_value", "difference_warm_minus_cold", "interpretation", "causal_isolation"])


def _append_warm_history(dataset: str, seed: int, output: list[dict[str, Any]]) -> None:
    for condition in WARM_CONDITIONS:
        path = OUTPUT / "runs" / dataset / condition / f"seed_{seed}" / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        acc_epoch, f1_epoch = int(result["best_accuracy_epoch"]), int(result["best_f1_epoch"])
        joint_history = [item for item in result["history"] if item["stage"] == "joint"]
        stop_epoch = max((int(item["epoch"]) for item in joint_history), default=0)
        for item in result["history"]:
            output.append({
                "dataset": dataset, "condition": condition, "seed": seed,
                "stage": item["stage"], "epoch": item["epoch"], "train_ce": item.get("train_ce"),
                "train_accuracy": item.get("train_accuracy"), "train_macro_f1": item.get("train_macro_f1"),
                "val_ce": item.get("val_ce"), "val_accuracy": item.get("val_accuracy"),
                "val_macro_f1": item.get("val_macro_f1"),
                "is_best_accuracy_epoch": bool(item["stage"] == "joint" and int(item["epoch"]) == acc_epoch),
                "is_best_f1_epoch": bool(item["stage"] == "joint" and int(item["epoch"]) == f1_epoch),
                "is_early_stop_epoch": bool(item["stage"] == "joint" and int(item["epoch"]) == stop_epoch),
            })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", help="set CUDA_VISIBLE_DEVICES=1 to use the free RTX 3090")
    args = parser.parse_args()
    print(json.dumps(analyze(args.device), indent=2, sort_keys=True))
