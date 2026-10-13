#!/usr/bin/env python3
"""Read-only classwise, node-transition, degree, and trajectory audit for V2.1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v21 import (  # noqa: E402
    DATASETS,
    SEEDS,
    build_config,
    verify_cache_entries,
    data_info,
    module_hash,
    sha256_file,
    tensor_fingerprint,
    write_json,
)

OUT = ROOT / "research/rmcge_v21_stable_adaptation"
DATA = OUT / "data"
OLD_DATA = ROOT / "research/rmcge_v21/data"
OLD_RUNS = ROOT / "outputs/rmcge_v21/runs"
CONDITIONS = ("C0", "C1", "C2")
CHECKPOINTS = ("best_accuracy", "best_f1")
CONTRASTS = (("C1-C0", "C1", "C0"), ("C2-C1", "C2", "C1"), ("C2-C0", "C2", "C0"))
DEGREE_GROUPS = (("degree_0", 0, 0), ("degree_1", 1, 1), ("degree_2_3", 2, 3), ("degree_4_7", 4, 7), ("degree_ge8", 8, None), ("degree_lt2", 0, 1), ("degree_ge2", 2, None))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def metrics_for(logits: torch.Tensor, target: torch.Tensor, labels: list[int]) -> dict[str, Any]:
    logits = logits.detach().float().cpu()
    target = target.detach().long().cpu()
    pred = logits.argmax(-1)
    matrix = confusion_matrix(target.numpy(), pred.numpy(), labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        target.numpy(), pred.numpy(), labels=labels, zero_division=0
    )
    ce_each = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    return {
        "prediction": pred,
        "target": target,
        "ce_each": ce_each,
        "acc": float((pred == target).float().mean()),
        "macro_f1": float(np.mean(f1)),
        "macro_recall": float(np.mean(recall)),
        "ce": float(ce_each.mean()),
        "class_metrics": {
            int(class_id): {
                "support": int(support[index]),
                "predicted_count": int(matrix[:, index].sum()),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "low_support_lt10": bool(support[index] < 10),
            }
            for index, class_id in enumerate(labels)
        },
        "confusion": matrix,
    }


def _group_mask(degree: torch.Tensor, lower: int, upper: int | None) -> torch.Tensor:
    mask = degree >= lower
    if upper is not None:
        mask &= degree <= upper
    return mask


def _percentile(values: torch.Tensor, q: float) -> float | None:
    return float(torch.quantile(values.float(), q)) if values.numel() else None


def main(device: str = "cuda:0") -> None:
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels

    DATA.mkdir(parents=True, exist_ok=True)
    caches = verify_cache_entries()
    all_class_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    all_global_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    dataset_split_fingerprints: dict[str, Any] = {}

    torch.cuda.set_device(torch.device(device))
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg0 = build_config(dataset, "C0", seed, device, caches[dataset])
            data = load_mag_data(cfg0, "nc", seed)
            labels = _resolve_nc_eval_labels(data, include_test=False)
            val_idx_cpu = data.val_idx.detach().cpu().long()
            val_idx = val_idx_cpu.to(device)
            x, edge = data.x.to(device), data.edge_index.to(device)
            # MAG's physical graph is stored as directed edges. Receiver degree is the
            # same physical degree convention used by V2.1 selection diagnostics.
            physical_degree = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            physical_degree.index_add_(0, edge[1], torch.ones(edge.size(1), dtype=torch.long, device=device))
            y_val = data.y[val_idx_cpu].detach().cpu().long()
            ds_key = f"{dataset}/seed_{seed}"
            dataset_split_fingerprints[ds_key] = {
                "node_count": int(data.num_nodes),
                "feature_sha256": tensor_fingerprint(data.x),
                "physical_edge_index_sha256": tensor_fingerprint(data.edge_index),
                "train_index_sha256": tensor_fingerprint(data.train_idx),
                "validation_index_sha256": tensor_fingerprint(data.val_idx),
                "train_labels_sha256": tensor_fingerprint(data.y[data.train_idx]),
                "validation_labels_sha256": tensor_fingerprint(data.y[data.val_idx]),
                "label_ids_used": labels,
                "test_label_values_indexed": False,
            }
            predictions: dict[tuple[str, str], dict[str, Any]] = {}
            for condition in CONDITIONS:
                old_run_row = _read_old_run_row(dataset, condition, seed)
                for checkpoint_kind in CHECKPOINTS:
                    relative = old_run_row[
                        "checkpoint_path" if checkpoint_kind == "best_accuracy" else "best_f1_checkpoint_path"
                    ]
                    checkpoint_path = ROOT / relative
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(checkpoint_path)
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    _validate_payload(payload, dataset, condition, seed, checkpoint_kind, caches[dataset])
                    cfg = build_config(dataset, condition, seed, device, caches[dataset])
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
                    classifier.eval()
                    with torch.no_grad():
                        embeddings = model(x, edge)[0]
                        logits = classifier(embeddings[val_idx]).float().cpu()
                    values = metrics_for(logits, y_val, labels)
                    stored = payload.get("metrics", {})
                    for metric_key, actual in (("val_acc", values["acc"]), ("val_macro_f1", values["macro_f1"]), ("val_ce", values["ce"])):
                        if metric_key in stored and abs(float(stored[metric_key]) - actual) > 2.0e-5:
                            raise RuntimeError(
                                f"checkpoint metric mismatch for {dataset}/{condition}/{seed}/{checkpoint_kind} "
                                f"{metric_key}: stored={stored[metric_key]} reloaded={actual}"
                            )
                    predictions[(condition, checkpoint_kind)] = values
                    all_global_rows.append({
                        "dataset": dataset, "seed": seed, "checkpoint": checkpoint_kind,
                        "condition": condition, "epoch": int(payload["epoch"]),
                        "validation_count": int(y_val.numel()), "accuracy": values["acc"],
                        "macro_f1": values["macro_f1"], "macro_recall": values["macro_recall"],
                        "cross_entropy": values["ce"], "num_macro_classes": len(labels),
                        "low_support_class_count": sum(item["low_support_lt10"] for item in values["class_metrics"].values()),
                    })
                    for class_id, class_record in values["class_metrics"].items():
                        all_class_rows.append({
                            "dataset": dataset, "seed": seed, "checkpoint": checkpoint_kind,
                            "row_type": "model", "condition": condition, "contrast": "",
                            "class_id": class_id, **class_record,
                        })
                    for row_index, class_id in enumerate(labels):
                        all_confusion_rows.append({
                            "dataset": dataset, "seed": seed, "checkpoint": checkpoint_kind,
                            "condition": condition, "true_class": class_id,
                            **{f"pred_{target_class}": int(values["confusion"][row_index, col_index])
                               for col_index, target_class in enumerate(labels)},
                        })
                    source_records.append({
                        "dataset": dataset, "condition": condition, "seed": seed,
                        "checkpoint": checkpoint_kind, "path": relative,
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "checkpoint_bytes": checkpoint_path.stat().st_size,
                        "run_config_sha256": old_run_row["config_sha256"],
                        "resolved_config_path": str((checkpoint_path.parent / "resolved_config.yaml").relative_to(ROOT)),
                        "resolved_config_sha256": sha256_file(checkpoint_path.parent / "resolved_config.yaml"),
                        "run_metrics_sha256": sha256_file(ROOT / old_run_row["run_metrics_path"]),
                        "selection": payload["selection"], "epoch": int(payload["epoch"]),
                        "model_state_sha256": state_fingerprint(payload["model_state"]),
                        "classifier_state_sha256": state_fingerprint(payload["head_state"]),
                        "study_provenance": payload.get("run_metadata", {}).get("study_provenance", {}),
                        "semantic_feature_fingerprint": payload.get("run_metadata", {}).get("semantic_feature_fingerprint"),
                        "semantic_candidate_fingerprint": payload.get("run_metadata", {}).get("semantic_candidate_fingerprint"),
                        "semantic_v7b_fingerprint": payload.get("run_metadata", {}).get("semantic_v7b_fingerprint"),
                        "data_info": payload.get("data_info", {}),
                    })
                    del embeddings, logits, model, classifier, payload
                    torch.cuda.empty_cache()

            for selection in CHECKPOINTS:
                for contrast, right_name, left_name in CONTRASTS:
                    right = predictions[(right_name, selection)]
                    left = predictions[(left_name, selection)]
                    for class_id in labels:
                        rclass = right["class_metrics"][class_id]
                        lclass = left["class_metrics"][class_id]
                        all_class_rows.append({
                            "dataset": dataset, "seed": seed, "checkpoint": selection,
                            "row_type": "paired_delta", "condition": "", "contrast": contrast,
                            "class_id": class_id,
                            "support": rclass["support"] - lclass["support"],
                            "predicted_count": rclass["predicted_count"] - lclass["predicted_count"],
                            "precision": rclass["precision"] - lclass["precision"],
                            "recall": rclass["recall"] - lclass["recall"],
                            "f1": rclass["f1"] - lclass["f1"],
                            "low_support_lt10": rclass["low_support_lt10"] or lclass["low_support_lt10"],
                        })
                    right_pred, left_pred = right["prediction"], left["prediction"]
                    target = right["target"]
                    right_ce, left_ce = right["ce_each"], left["ce_each"]
                    delta_ce = right_ce - left_ce
                    improved = delta_ce < 0
                    worsened = delta_ce > 0
                    transition_rows.append({
                        "dataset": dataset, "seed": seed, "checkpoint": selection, "contrast": contrast,
                        "n_validation": int(target.numel()),
                        "left_correct_right_wrong": int(((left_pred == target) & (right_pred != target)).sum()),
                        "left_wrong_right_correct": int(((left_pred != target) & (right_pred == target)).sum()),
                        "prediction_flip_count": int((left_pred != right_pred).sum()),
                        "prediction_flip_fraction": float((left_pred != right_pred).float().mean()),
                        "ce_improved_count": int(improved.sum()), "ce_improved_fraction": float(improved.float().mean()),
                        "ce_improvement_mean": float((-delta_ce[improved]).mean()) if bool(improved.any()) else 0.0,
                        "ce_improvement_p90": _percentile(-delta_ce[improved], 0.90),
                        "ce_worsened_count": int(worsened.sum()), "ce_worsened_fraction": float(worsened.float().mean()),
                        "ce_worsening_mean": float(delta_ce[worsened].mean()) if bool(worsened.any()) else 0.0,
                        "ce_worsening_p90": _percentile(delta_ce[worsened], 0.90),
                        "ce_delta_mean_right_minus_left": float(delta_ce.mean()),
                        "ce_delta_p05": _percentile(delta_ce, 0.05), "ce_delta_p95": _percentile(delta_ce, 0.95),
                    })
                    for group_name, lower, upper in DEGREE_GROUPS:
                        group_mask_all = _group_mask(physical_degree, lower, upper)
                        mask = group_mask_all[val_idx].detach().cpu()
                        if not bool(mask.any()):
                            continue
                        for condition in CONDITIONS:
                            values = predictions[(condition, selection)]
                            subset_pred = values["prediction"][mask]
                            subset_target = target[mask]
                            subset_ce = values["ce_each"][mask]
                            macro = precision_recall_fscore_support(
                                subset_target.numpy(), subset_pred.numpy(), labels=labels,
                                average="macro", zero_division=0,
                            )[2]
                            degree_rows.append({
                                "dataset": dataset, "seed": seed, "checkpoint": selection,
                                "row_type": "model", "group": group_name, "condition": condition, "contrast": "",
                                "n_validation": int(mask.sum()), "accuracy": float((subset_target == subset_pred).float().mean()),
                                "macro_f1": float(macro), "cross_entropy": float(subset_ce.mean()),
                                "prediction_flip_fraction": 0.0,
                            })
                        left_values = predictions[(left_name, selection)]
                        right_values = predictions[(right_name, selection)]
                        deg_flip = (left_values["prediction"][mask] != right_values["prediction"][mask])
                        left_acc = float((left_values["prediction"][mask] == target[mask]).float().mean())
                        right_acc = float((right_values["prediction"][mask] == target[mask]).float().mean())
                        left_f1 = precision_recall_fscore_support(
                            target[mask].numpy(), left_values["prediction"][mask].numpy(), labels=labels,
                            average="macro", zero_division=0,
                        )[2]
                        right_f1 = precision_recall_fscore_support(
                            target[mask].numpy(), right_values["prediction"][mask].numpy(), labels=labels,
                            average="macro", zero_division=0,
                        )[2]
                        degree_rows.append({
                            "dataset": dataset, "seed": seed, "checkpoint": selection,
                            "row_type": "paired_delta", "group": group_name, "condition": "", "contrast": contrast,
                            "n_validation": int(mask.sum()), "accuracy": right_acc - left_acc,
                            "macro_f1": float(right_f1 - left_f1),
                            "cross_entropy": float(right_values["ce_each"][mask].mean() - left_values["ce_each"][mask].mean()),
                            "prediction_flip_fraction": float(deg_flip.float().mean()),
                        })
            _append_history(dataset, seed, history_rows)
            del data, x, edge, physical_degree, predictions
            torch.cuda.empty_cache()

    class_fields = ["dataset", "seed", "checkpoint", "row_type", "condition", "contrast", "class_id", "support", "predicted_count", "precision", "recall", "f1", "low_support_lt10"]
    write_csv(DATA / "classwise_metrics.csv", all_class_rows, class_fields)
    write_csv(DATA / "checkpoint_metrics.csv", all_global_rows, ["dataset", "seed", "checkpoint", "condition", "epoch", "validation_count", "accuracy", "macro_f1", "macro_recall", "cross_entropy", "num_macro_classes", "low_support_class_count"])
    transition_fields = ["dataset", "seed", "checkpoint", "contrast", "n_validation", "left_correct_right_wrong", "left_wrong_right_correct", "prediction_flip_count", "prediction_flip_fraction", "ce_improved_count", "ce_improved_fraction", "ce_improvement_mean", "ce_improvement_p90", "ce_worsened_count", "ce_worsened_fraction", "ce_worsening_mean", "ce_worsening_p90", "ce_delta_mean_right_minus_left", "ce_delta_p05", "ce_delta_p95"]
    write_csv(DATA / "prediction_transitions.csv", transition_rows, transition_fields)
    degree_fields = ["dataset", "seed", "checkpoint", "row_type", "group", "condition", "contrast", "n_validation", "accuracy", "macro_f1", "cross_entropy", "prediction_flip_fraction"]
    write_csv(DATA / "degree_group_metrics.csv", degree_rows, degree_fields)
    history_fields = ["dataset", "condition", "seed", "epoch", "train_ce", "train_accuracy", "train_macro_f1", "val_ce", "val_accuracy", "val_macro_f1", "best_accuracy_epoch", "best_f1_epoch", "early_stop_epoch"]
    write_csv(DATA / "coldstart_epoch_comparison.csv", history_rows, history_fields)
    confusion_fields = ["dataset", "seed", "checkpoint", "condition", "true_class"] + sorted({key for row in all_confusion_rows for key in row if key.startswith("pred_")}, key=lambda key: int(key[5:]))
    write_csv(DATA / "confusion_matrices.csv", all_confusion_rows, confusion_fields)
    write_json(DATA / "source_checkpoints.json", {
        "source_branch": "exp/rmcge_v21_message_purpose_selection",
        "source_commit": "218d6d230e5afc16e259c09777f6363de5e0c6cf",
        "records": source_records,
        "test_evaluation": False,
    })
    write_json(DATA / "phase_a_manifest.json", {
        "script": "scripts/diagnose_rmcge_v21_classes.py",
        "device": torch.cuda.get_device_name(torch.device(device)),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "split_fingerprints": dataset_split_fingerprints,
        "semantic_cache_sha256": {name: sha256_file(Path(caches[name]["cache_path_abs"])) for name in DATASETS},
        "records": len(source_records),
        "test_labels_indexed": False,
    })
    print(json.dumps({
        "source_records": len(source_records), "classwise_rows": len(all_class_rows),
        "transition_rows": len(transition_rows), "degree_rows": len(degree_rows),
        "history_rows": len(history_rows), "test_labels_indexed": False,
    }, indent=2))


def _read_old_run_row(dataset: str, condition: str, seed: int) -> dict[str, str]:
    path = OLD_DATA / "runs.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    found = [row for row in rows if row["dataset"] == dataset and row["condition"] == condition and int(row["seed"]) == seed]
    if len(found) != 1 or found[0]["status"] != "completed_and_verified" or found[0]["test_metrics_present"] != "false":
        raise RuntimeError(f"historical V2.1 row unavailable or invalid: {dataset}/{condition}/{seed}")
    return found[0]


def _validate_payload(payload: dict[str, Any], dataset: str, condition: str, seed: int, kind: str, cache: dict[str, Any]) -> None:
    expected = "best_val_accuracy" if kind == "best_accuracy" else "best_val_macro_f1_within_accuracy_stopping_horizon"
    if payload.get("task") != "nc" or payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise RuntimeError(f"checkpoint protocol mismatch for {dataset}/{condition}/{seed}/{kind}")
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != expected:
        raise RuntimeError(f"checkpoint identity/selection mismatch for {dataset}/{condition}/{seed}/{kind}")
    provenance = payload.get("run_metadata", {}).get("study_provenance", {})
    expected_provenance = {
        "study": "RM-CGE V2.1", "dataset": dataset, "condition": condition, "seed": seed,
        "official_train_val_split": True, "test_evaluation": False,
    }
    if provenance != expected_provenance:
        raise RuntimeError(f"checkpoint study metadata mismatch: {provenance} != {expected_provenance}")
    for metadata_key, expected_value in (
        ("semantic_feature_fingerprint", cache["feature_fingerprint"]),
        ("semantic_candidate_fingerprint", cache["candidate_fingerprint"]),
        ("semantic_v7b_fingerprint", cache["v7b_fingerprint"]),
    ):
        if payload.get("run_metadata", {}).get(metadata_key) != expected_value:
            raise RuntimeError(f"checkpoint {metadata_key} mismatch for {dataset}/{condition}/{seed}")
    if any(str(key).startswith("test_") for key in payload.get("metrics", {})):
        raise RuntimeError("Test metric found in historical checkpoint")
    if any(not torch.isfinite(value).all() for state_name in ("model_state", "head_state")
           for value in payload[state_name].values() if torch.is_floating_point(value)):
        raise RuntimeError(f"nonfinite tensor in {dataset}/{condition}/{seed}/{kind}")


def _append_history(dataset: str, seed: int, output: list[dict[str, Any]]) -> None:
    row = _read_old_run_row(dataset, "C0", seed)
    for condition in CONDITIONS:
        run = _read_old_run_row(dataset, condition, seed)
        metrics_path = ROOT / run["run_metrics_path"]
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))["runs"][0]
        metadata = payload["metadata"]
        history = metadata.get("epoch_history", [])
        best_acc, best_f1 = int(run["best_accuracy_epoch"]), int(run["best_f1_epoch"])
        stop_epoch = int(metadata["epochs_completed"])
        if not history or len(history) != stop_epoch:
            raise RuntimeError(f"invalid historical epoch trace {dataset}/{condition}/{seed}")
        for item in history:
            values = {key: float(item[key]) for key in ("train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce", "val_acc", "val_macro_f1")}
            if any(not math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"nonfinite historical epoch metric {dataset}/{condition}/{seed}/{item['epoch']}")
            output.append({
                "dataset": dataset, "condition": condition, "seed": seed, "epoch": int(item["epoch"]),
                "train_ce": values["train_ce_eval"], "train_accuracy": values["train_accuracy_eval"],
                "train_macro_f1": values["train_macro_f1_eval"], "val_ce": values["val_ce"],
                "val_accuracy": values["val_acc"], "val_macro_f1": values["val_macro_f1"],
                "best_accuracy_epoch": int(item["epoch"]) == best_acc,
                "best_f1_epoch": int(item["epoch"]) == best_f1,
                "early_stop_epoch": int(item["epoch"]) == stop_epoch,
            })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args.device)
