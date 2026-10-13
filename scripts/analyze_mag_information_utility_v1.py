#!/usr/bin/env python3
"""Validate and summarize the frozen MAG Information Utility V1 matrix.

This script is intentionally validation-only with respect to model artifacts:
it reads the 45 saved validation runs, their epoch histories, and validation
predictions, then writes small tracked aggregates and reports. It never loads
Test labels or checkpoints and refuses to publish a partial matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/mag_information_utility_v1"
DATA = RESEARCH / "data"
OUTPUT = ROOT / "outputs/mag_information_utility_v1"
RUNS = OUTPUT / "runs"
PREDICTIONS = OUTPUT / "validation_predictions"
NODE_GROUPS = OUTPUT / "node_groups"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
CONDITIONS = ("I0", "I1", "I2", "I3", "I4")
BASELINE_COMMIT = "4cd655a411a6c0ab12e067285cd2202208b0ab18"
IMPLEMENTATION_FREEZE_COMMIT = "76cf4ea73192029501425321fe1683de203140ed"
FROZEN_CORE_PATHS = (
    "configs/model/mag_information_probe_v1.yaml",
    "configs/study/mag_information_utility_v1.yaml",
    "research/mag_information_utility_v1/EXPERIMENT_DESIGN.md",
    "research/mag_information_utility_v1/GRAPH_SIGNAL_REPORT.md",
    "research/mag_information_utility_v1/data/graph_structure_stats.csv",
    "research/mag_information_utility_v1/data/node_group_definitions.json",
    "research/mag_information_utility_v1/data/pre_run_data_manifest.json",
    "research/mag_information_utility_v1/data/random_graph_audit.json",
    "research/mag_information_utility_v1/data/train_only_label_relation_stats.csv",
    "src/models/mag_information_probe_v1.py",
)
ALLOWED_POST_FREEZE_PATHS = frozenset(
    {
        "research/mag_information_utility_v1/IMPLEMENTATION_AUDIT.md",
        "research/mag_information_utility_v1/data/failures.json",
        "scripts/analyze_mag_information_utility_v1.py",
        "scripts/run_mag_information_utility_v1.py",
        "tests/test_mag_information_utility_analysis_v1.py",
        "tests/test_mag_information_utility_v1.py",
    }
)
CONTRASTS = (("I1", "I0"), ("I2", "I1"), ("I3", "I2"), ("I3", "I4"))
PRIMARY_METRICS = (
    "validation_accuracy",
    "validation_macro_f1",
    "validation_cross_entropy",
    "train_validation_accuracy_gap",
    "training_seconds",
    "cuda_peak_allocated_bytes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE
    ).stdout


def _validate_source_commits(source_commits: set[str]) -> str:
    """Require run sources to descend from Freeze with unchanged model/protocol.

    Runtime commits may adjust provenance or checkpoint diagnostics after
    Freeze. The allowlist bounds those changes; hashes of model, configuration,
    and pre-run protocol/data audit files must remain identical to Freeze.
    """
    if not source_commits:
        raise ValueError("no source commits found in completed runs")
    reference_hashes: dict[str, str] | None = None
    for commit in sorted(source_commits):
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", IMPLEMENTATION_FREEZE_COMMIT, commit],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"run source {commit} is not descended from implementation freeze "
                f"{IMPLEMENTATION_FREEZE_COMMIT}"
            ) from exc
        changed = set(
            _git_bytes("diff", "--name-only", f"{IMPLEMENTATION_FREEZE_COMMIT}..{commit}")
            .decode("utf-8")
            .splitlines()
        )
        unexpected = changed - ALLOWED_POST_FREEZE_PATHS
        if unexpected:
            raise ValueError(
                f"run source {commit} changes files outside the runtime-fix allowlist: "
                f"{sorted(unexpected)}"
            )
        hashes = {
            path: hashlib.sha256(_git_bytes("show", f"{commit}:{path}")).hexdigest()
            for path in FROZEN_CORE_PATHS
        }
        freeze_hashes = {
            path: hashlib.sha256(
                _git_bytes("show", f"{IMPLEMENTATION_FREEZE_COMMIT}:{path}")
            ).hexdigest()
            for path in FROZEN_CORE_PATHS
        }
        if hashes != freeze_hashes:
            differing = [path for path in FROZEN_CORE_PATHS if hashes[path] != freeze_hashes[path]]
            raise ValueError(f"frozen model/protocol files differ at {commit}: {differing}")
        if reference_hashes is not None and hashes != reference_hashes:
            raise ValueError(f"frozen model/protocol hashes differ across run sources at {commit}")
        reference_hashes = hashes
    digest = hashlib.sha256()
    for path, value in sorted((reference_hashes or {}).items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _finite_tree(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{context}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite number at {context}: {value}")


def _load_history(path: Path, expected_dataset: str, expected_condition: str, expected_seed: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty epoch history: {path}")
    for index, row in enumerate(rows, 1):
        if (row.get("dataset"), row.get("condition"), int(row.get("seed", -1))) != (
            expected_dataset,
            expected_condition,
            expected_seed,
        ):
            raise ValueError(f"history identity mismatch at {path}:{index + 1}")
        if int(row.get("epoch", -1)) != index:
            raise ValueError(f"non-contiguous epochs in {path}; expected epoch {index}")
        for name, value in row.items():
            if name in {"dataset", "condition", "seed", "epoch", "all_metrics_finite"}:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"non-finite {name} in {path}, epoch {index}")
        if row.get("all_metrics_finite", "").lower() != "true":
            raise ValueError(f"history finite-metric marker failed in {path}, epoch {index}")
    return rows


def _metric_metrics(run: dict[str, Any]) -> dict[str, float]:
    train = run["best_accuracy_checkpoint_metrics"]["train"]
    val = run["best_accuracy_checkpoint_metrics"]["validation"]
    return {
        "train_accuracy": float(train["accuracy"]),
        "train_macro_f1": float(train["macro_f1"]),
        "train_cross_entropy": float(train["cross_entropy"]),
        "validation_accuracy": float(val["accuracy"]),
        "validation_macro_f1": float(val["macro_f1"]),
        "validation_cross_entropy": float(val["cross_entropy"]),
        "train_validation_accuracy_gap": float(train["accuracy"] - val["accuracy"]),
    }


def _validate_run(dataset: str, condition: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stem = f"{dataset}_{condition}_seed_{seed}"
    metrics_path = RUNS / dataset / f"{stem}.json"
    run = _load_json(metrics_path)
    identity = (run.get("dataset"), run.get("condition"), int(run.get("seed", -1)))
    if identity != (dataset, condition, seed):
        raise ValueError(f"run identity mismatch in {metrics_path}: {identity}")
    if run.get("status") != "complete":
        raise ValueError(f"run is not complete: {metrics_path}")
    if run.get("evaluate_test") is not False or run.get("test_labels_indexed") is not False:
        raise ValueError(f"Test boundary check failed: {metrics_path}")
    if run.get("protocol") != "unified_full_graph_nc_v1":
        raise ValueError(f"protocol mismatch: {metrics_path}")
    if not run.get("best_accuracy_reload_reproduced") or not run.get("best_f1_reload_reproduced"):
        raise ValueError(f"checkpoint reload marker failed: {metrics_path}")
    for metric_key in ("best_accuracy_checkpoint_metrics", "best_f1_checkpoint_metrics"):
        _finite_tree(run[metric_key], f"{metrics_path}:{metric_key}")
    provenance = run.get("provenance", {})
    if provenance.get("evaluate_test") is not False or provenance.get("test_labels_indexed") is not False:
        raise ValueError(f"provenance Test boundary check failed: {metrics_path}")
    if provenance.get("dataset") != dataset or int(provenance.get("seed", -1)) != seed:
        raise ValueError(f"provenance identity mismatch: {metrics_path}")

    for ckpt_key, sha_key in (
        ("best_accuracy_checkpoint_path", "best_accuracy_checkpoint_sha256"),
        ("best_f1_checkpoint_path", "best_f1_checkpoint_sha256"),
    ):
        ckpt = ROOT / run[ckpt_key]
        if not ckpt.is_file() or sha256_file(ckpt) != run[sha_key]:
            raise ValueError(f"checkpoint missing or hash mismatch: {ckpt}")

    node_path = ROOT / run["validation_predictions_path_local"]
    if not node_path.is_file() or sha256_file(node_path) != run["validation_predictions_sha256"]:
        raise ValueError(f"validation predictions missing or hash mismatch: {node_path}")
    with np.load(node_path, allow_pickle=False) as data:
        required = {"validation_node_ids", "validation_labels", "predictions", "per_node_cross_entropy"}
        if set(data.files) != required:
            raise ValueError(f"unexpected validation prediction arrays in {node_path}: {data.files}")
        arrays = {key: np.asarray(data[key]) for key in required}
    lengths = {value.shape[0] for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"validation prediction arrays have inconsistent/empty lengths: {node_path}")
    ids = arrays["validation_node_ids"].astype(np.int64, copy=False)
    labels = arrays["validation_labels"].astype(np.int64, copy=False)
    predictions = arrays["predictions"].astype(np.int64, copy=False)
    ce = arrays["per_node_cross_entropy"].astype(np.float64, copy=False)
    if np.unique(ids).size != ids.size or ids.min() < 0:
        raise ValueError(f"validation node IDs are duplicate or invalid: {node_path}")
    if not np.isfinite(ce).all() or (ce < 0).any():
        raise ValueError(f"validation per-node CE is invalid: {node_path}")
    class_ids = run["training_settings"]["macro_f1_classes"]
    if not class_ids or labels.min() < min(class_ids) or labels.max() > max(class_ids):
        raise ValueError(f"validation labels are outside configured classes: {node_path}")
    if predictions.min() < min(class_ids) or predictions.max() > max(class_ids):
        raise ValueError(f"validation predictions are outside configured classes: {node_path}")
    recalc_acc = float((predictions == labels).mean())
    recalc_f1 = float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0))
    recalc_ce = float(ce.mean())
    acc_metrics = run["best_accuracy_checkpoint_metrics"]["validation"]
    for key, actual in (("accuracy", recalc_acc), ("macro_f1", recalc_f1), ("cross_entropy", recalc_ce)):
        if not math.isclose(float(acc_metrics[key]), actual, rel_tol=0.0, abs_tol=2e-6):
            raise ValueError(f"saved validation predictions do not reproduce {key}: {metrics_path}")

    history = _load_history(
        RUNS / dataset / f"{stem}_epoch_history.csv", dataset, condition, seed
    )
    if len(history) != int(run["epochs_completed"]):
        raise ValueError(f"epoch count mismatch: {metrics_path}")
    if int(run["best_accuracy_epoch"]) > len(history) or int(run["best_f1_epoch"]) > len(history):
        raise ValueError(f"selected epoch is outside history: {metrics_path}")
    run["_metrics_path"] = str(metrics_path.relative_to(ROOT))
    run["_node_arrays"] = {"ids": ids, "labels": labels, "predictions": predictions, "ce": ce}
    return run, history


def _run_rows(runs: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            block = [runs[(dataset, condition, seed)] for condition in CONDITIONS]
            init_pairs = {
                (
                    row["parameter_audit"]["model_initial_sha256"],
                    row["parameter_audit"]["classifier_initial_sha256"],
                )
                for row in block
            }
            if len(init_pairs) != 1:
                raise ValueError(f"treatments do not share initial model and head: {dataset} seed {seed}")
            if len({row["total_parameter_count"] for row in block}) != 1:
                raise ValueError(f"parameter count differs among treatments: {dataset} seed {seed}")
            if len({row["provenance"]["split_sha256"] for row in block}) != 1:
                raise ValueError(f"official split differs among treatments: {dataset} seed {seed}")
            if len({row["provenance"]["feature_fingerprint"] for row in block}) != 1:
                raise ValueError(f"feature fingerprint differs among treatments: {dataset} seed {seed}")
            if len({row["provenance"]["candidate_fingerprint"] for row in block}) != 1:
                raise ValueError(f"candidate cache differs among treatments: {dataset} seed {seed}")
            if len({row["provenance"]["physical_edge_fingerprint"] for row in block}) != 1:
                raise ValueError(f"physical graph fingerprint differs among treatments: {dataset} seed {seed}")
            for row in block:
                metrics = _metric_metrics(row)
                env = row["environment"]
                audit = row["parameter_audit"]
                output = {
                    "run_id": f"{dataset}__{row['condition']}__seed{seed}",
                    "dataset": dataset,
                    "seed": seed,
                    "condition": row["condition"],
                    "source_commit": row["source_commit"],
                    "protocol": row["protocol"],
                    "best_accuracy_epoch": row["best_accuracy_epoch"],
                    "best_f1_epoch": row["best_f1_epoch"],
                    "epochs_completed": row["epochs_completed"],
                    **metrics,
                    "best_f1_checkpoint_validation_accuracy": row["best_f1_checkpoint_metrics"]["validation"]["accuracy"],
                    "best_f1_checkpoint_validation_macro_f1": row["best_f1_checkpoint_metrics"]["validation"]["macro_f1"],
                    "best_f1_checkpoint_validation_cross_entropy": row["best_f1_checkpoint_metrics"]["validation"]["cross_entropy"],
                    "training_seconds": row["training_seconds"],
                    "mean_epoch_seconds": row["mean_epoch_seconds"],
                    "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"],
                    "total_parameter_count": row["total_parameter_count"],
                    "model_initial_sha256": audit["model_initial_sha256"],
                    "classifier_initial_sha256": audit["classifier_initial_sha256"],
                    "joint_initial_sha256": audit["joint_initial_sha256"],
                    "active_context_slots": json.dumps(audit["active_context_slots"]),
                    "context_slot_gradient_norms": json.dumps(audit["context_slot_gradient_norms"], sort_keys=True),
                    "checkpoint_accuracy_sha256": row["best_accuracy_checkpoint_sha256"],
                    "checkpoint_f1_sha256": row["best_f1_checkpoint_sha256"],
                    "validation_predictions_sha256": row["validation_predictions_sha256"],
                    "best_accuracy_reload_max_node_ce_abs_diff": row["best_accuracy_reload_max_node_ce_abs_diff"],
                    "split_sha256": row["provenance"]["split_sha256"],
                    "feature_fingerprint": row["provenance"]["feature_fingerprint"],
                    "candidate_fingerprint": row["provenance"]["candidate_fingerprint"],
                    "random_text_cache_sha256": row["provenance"]["random_text_cache_sha256"],
                    "random_visual_cache_sha256": row["provenance"]["random_visual_cache_sha256"],
                    "gpu_name": env["selected_gpu"]["name"],
                    "gpu_index": env["selected_gpu"]["index"],
                    "driver_version": env["selected_gpu"]["driver_version"],
                    "python_version": env["python"],
                    "numpy_version": env["numpy"],
                    "torch_version": env["torch"],
                    "cuda_build": env["torch_cuda_build"],
                    "cudnn_version": env["cudnn"],
                    "tf32_settings": json.dumps(env["tf32"], sort_keys=True),
                    "determinism_settings": json.dumps(env["determinism"], sort_keys=True),
                    "training_settings": json.dumps(row["training_settings"], sort_keys=True),
                    "metrics_path": row["_metrics_path"],
                }
                rows.append(output)
    return rows


def _epoch_rows(history_by_run: dict[tuple[str, str, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for key, history in sorted(history_by_run.items()):
        dataset, condition, seed = key
        for row in history:
            result.append({"run_id": f"{dataset}__{condition}__seed{seed}", **row})
    return result


def _paired_rows(runs: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for candidate, reference in CONTRASTS:
            for metric in PRIMARY_METRICS:
                deltas = []
                for seed in SEEDS:
                    left = runs[(dataset, reference, seed)]
                    right = runs[(dataset, candidate, seed)]
                    if metric.startswith("validation_") or metric == "train_validation_accuracy_gap":
                        left_value = _metric_metrics(left)[metric]
                        right_value = _metric_metrics(right)[metric]
                    else:
                        left_value = float(left[metric])
                        right_value = float(right[metric])
                    delta = right_value - left_value
                    deltas.append(delta)
                    rows.append({
                        "row_type": "paired_seed",
                        "dataset": dataset,
                        "contrast": f"{candidate}-{reference}",
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "seed": seed,
                        "reference_value": left_value,
                        "candidate_value": right_value,
                        "paired_delta": delta,
                        "mean_paired_delta": "",
                        "sample_sd_paired_delta": "",
                        "positive_seed_count": "",
                        "seeds": len(SEEDS),
                    })
                rows.append({
                    "row_type": "dataset_summary",
                    "dataset": dataset,
                    "contrast": f"{candidate}-{reference}",
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "seed": "",
                    "reference_value": "",
                    "candidate_value": "",
                    "paired_delta": "",
                    "mean_paired_delta": float(statistics.mean(deltas)),
                    "sample_sd_paired_delta": float(statistics.stdev(deltas)),
                    "positive_seed_count": int(sum(value > 0 for value in deltas)),
                    "seeds": len(SEEDS),
                })
    return rows


def _checkpoint_rows(runs: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run = runs[(dataset, condition, seed)]
                acc = run["best_accuracy_checkpoint_metrics"]
                f1 = run["best_f1_checkpoint_metrics"]
                result.append({
                    "dataset": dataset,
                    "condition": condition,
                    "seed": seed,
                    "best_accuracy_epoch": run["best_accuracy_epoch"],
                    "best_f1_epoch": run["best_f1_epoch"],
                    "epoch_difference_f1_minus_accuracy": int(run["best_f1_epoch"]) - int(run["best_accuracy_epoch"]),
                    "accuracy_checkpoint_val_accuracy": acc["validation"]["accuracy"],
                    "accuracy_checkpoint_val_macro_f1": acc["validation"]["macro_f1"],
                    "accuracy_checkpoint_val_ce": acc["validation"]["cross_entropy"],
                    "f1_checkpoint_val_accuracy": f1["validation"]["accuracy"],
                    "f1_checkpoint_val_macro_f1": f1["validation"]["macro_f1"],
                    "f1_checkpoint_val_ce": f1["validation"]["cross_entropy"],
                    "f1_selection_minus_accuracy_selection_accuracy": f1["validation"]["accuracy"] - acc["validation"]["accuracy"],
                    "f1_selection_minus_accuracy_selection_macro_f1": f1["validation"]["macro_f1"] - acc["validation"]["macro_f1"],
                    "f1_selection_minus_accuracy_selection_ce": f1["validation"]["cross_entropy"] - acc["validation"]["cross_entropy"],
                    "accuracy_checkpoint_reloaded": run["best_accuracy_reload_reproduced"],
                    "f1_checkpoint_reloaded": run["best_f1_reload_reproduced"],
                })
    return result


def _load_group_definitions() -> dict[str, Any]:
    value = _load_json(DATA / "node_group_definitions.json")
    if value.get("labels_used") is True:
        raise ValueError("node-group definitions claim label use")
    return value["datasets"]


def _node_group_rows(
    runs: dict[tuple[str, str, int], dict[str, Any]], group_defs: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        group_path = NODE_GROUPS / f"{dataset}.npz"
        if not group_path.is_file():
            raise FileNotFoundError(group_path)
        with np.load(group_path, allow_pickle=False) as archive:
            code_arrays = {key: np.asarray(archive[key]).astype(np.int64, copy=False) for key in archive.files}
        definitions = group_defs[dataset]["groups"]
        if sha256_file(group_path) != group_defs[dataset]["sha256"]:
            raise ValueError(f"pre-label node group cache hash mismatch for {dataset}")
        if set(code_arrays) != set(definitions):
            raise ValueError(f"node group keys differ from frozen definitions for {dataset}")
        n_classes = len(runs[(dataset, CONDITIONS[0], SEEDS[0])]["training_settings"]["macro_f1_classes"])
        class_ids = list(range(n_classes))
        for seed in SEEDS:
            seed_runs = {condition: runs[(dataset, condition, seed)] for condition in CONDITIONS}
            primary = seed_runs[CONDITIONS[0]]["_node_arrays"]
            ids = primary["ids"]
            labels = primary["labels"]
            for condition in CONDITIONS[1:]:
                arrays = seed_runs[condition]["_node_arrays"]
                if not np.array_equal(ids, arrays["ids"]) or not np.array_equal(labels, arrays["labels"]):
                    raise ValueError(f"validation node/label order differs across paired runs: {dataset} seed {seed}")
            if ids.max(initial=-1) >= next(iter(code_arrays.values())).size:
                raise ValueError(f"validation node ID exceeds frozen group arrays for {dataset}")

            for group_name, codes_all in code_arrays.items():
                if codes_all.shape != next(iter(code_arrays.values())).shape:
                    raise ValueError(f"node group arrays have inconsistent shape for {dataset}")
                group_code_values = sorted(set(codes_all[codes_all >= 0].tolist()))
                definition = definitions[group_name]
                names = list(definition.get("groups", []))
                for code in group_code_values:
                    mask = codes_all[ids] == code
                    if not mask.any():
                        continue
                    group_labels = labels[mask]
                    for condition in CONDITIONS:
                        arrays = seed_runs[condition]["_node_arrays"]
                        preds = arrays["predictions"][mask]
                        losses = arrays["ce"][mask]
                        metric_rows.append({
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "group_variable": group_name,
                            "group": names[code] if code < len(names) else f"missing_{code}",
                            "group_code": code,
                            "validation_node_count": int(mask.sum()),
                            "accuracy": float((preds == group_labels).mean()),
                            "macro_f1_all_configured_classes": float(f1_score(group_labels, preds, labels=class_ids, average="macro", zero_division=0)),
                            "cross_entropy": float(losses.mean()),
                            "small_group_warning": bool(mask.sum() < 30),
                            "group_definition_sha256": group_defs[dataset]["sha256"],
                        })

            # Conditional error changes are paired on the exact validation nodes.
            for candidate, reference in (("I3", "I4"), ("I2", "I1")):
                cand = seed_runs[candidate]["_node_arrays"]
                ref = seed_runs[reference]["_node_arrays"]
                scopes: list[tuple[str, str, np.ndarray]] = [("all_validation", "all", np.ones(ids.size, dtype=bool))]
                for group_name, codes_all in code_arrays.items():
                    definitions_for_group = definitions[group_name]
                    for code in sorted(set(codes_all[codes_all >= 0].tolist())):
                        label = definitions_for_group.get("groups", [])[code] if code < len(definitions_for_group.get("groups", [])) else f"missing_{code}"
                        scopes.append((group_name, label, codes_all[ids] == code))
                for group_name, label, mask in scopes:
                    if not mask.any():
                        continue
                    improvement = ref["ce"][mask] - cand["ce"][mask]
                    tolerance = 1.0e-12
                    changed = cand["predictions"][mask] != ref["predictions"][mask]
                    conditional_rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "contrast": f"{candidate}-{reference}",
                        "candidate": candidate,
                        "reference": reference,
                        "group_variable": group_name,
                        "group": label,
                        "validation_node_count": int(mask.sum()),
                        "reference_minus_candidate_mean_ce": float(improvement.mean()),
                        "improved_node_fraction": float((improvement > tolerance).mean()),
                        "worsened_node_fraction": float((improvement < -tolerance).mean()),
                        "tied_node_fraction": float((np.abs(improvement) <= tolerance).mean()),
                        "prediction_flip_fraction": float(changed.mean()),
                        "small_group_warning": bool(mask.sum() < 30),
                    })
    return metric_rows, conditional_rows


def _r0_reference_rows(runs: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    path = ROOT / "research/psce_mag_v7b1_corrective/data/run_rows.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    reference = {
        (row["dataset"], int(row["seed"])): row
        for row in source
        if row.get("mode") == "formal" and row.get("variant") == "r0_raw" and row.get("status") == "completed"
    }
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            row = reference[(dataset, seed)]
            params = int(row["metadata"]["model_parameters"]) + int(row["metadata"]["classifier_parameters"])
            for condition in CONDITIONS:
                run = runs[(dataset, condition, seed)]
                metrics = _metric_metrics(run)
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "probe_val_accuracy": metrics["validation_accuracy"],
                    "r0_raw_val_accuracy": float(row["metrics"]["val_acc"]),
                    "accuracy_difference_probe_minus_r0": metrics["validation_accuracy"] - float(row["metrics"]["val_acc"]),
                    "probe_val_macro_f1": metrics["validation_macro_f1"],
                    "r0_raw_val_macro_f1": float(row["metrics"]["val_macro_f1"]),
                    "macro_f1_difference_probe_minus_r0": metrics["validation_macro_f1"] - float(row["metrics"]["val_macro_f1"]),
                    "probe_parameters": int(run["total_parameter_count"]),
                    "r0_raw_parameters": params,
                    "capacity_matched": False,
                    "r0_reference_run_id": row["run_id"],
                    "r0_reference_freeze_commit": row.get("freeze_commit_sha"),
                })
    return rows


def _summary_by_condition(runs: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    summary: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for condition in CONDITIONS:
            summary[dataset][condition] = {}
            for metric in ("validation_accuracy", "validation_macro_f1", "validation_cross_entropy"):
                vals = [_metric_metrics(runs[(dataset, condition, seed)])[metric] for seed in SEEDS]
                summary[dataset][condition][metric] = (statistics.mean(vals), statistics.stdev(vals))
    return summary


def _fmt(mean: float, sd: float, *, percent: bool = False) -> str:
    factor = 100.0 if percent else 1.0
    digits = 2 if percent else 4
    return f"{mean*factor:.{digits}f} ± {sd*factor:.{digits}f}"


def _render_report(
    runs: dict[tuple[str, str, int], dict[str, Any]],
    paired: list[dict[str, Any]],
    r0_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    summary = _summary_by_condition(runs)
    info_lines = [
        "# MAG Information Utility V1 — Results",
        "",
        "All reported endpoint results use the Validation Accuracy selected checkpoint. Values are mean ± sample SD across the three paired training seeds (42/43/44). The seed is the training replication unit; Validation nodes are not treated as independent experimental replicates. No Test evaluation was performed.",
        "",
        "## Validation performance",
        "",
        "| Dataset | Treatment | Accuracy (%) | Macro-F1 (%) | Validation CE | Parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            acc, f1, ce = (summary[dataset][condition][key] for key in ("validation_accuracy", "validation_macro_f1", "validation_cross_entropy"))
            params = runs[(dataset, condition, SEEDS[0])]["total_parameter_count"]
            info_lines.append(f"| {dataset} | {condition} | {_fmt(*acc, percent=True)} | {_fmt(*f1, percent=True)} | {_fmt(*ce)} | {params} |")
    info_lines += [
        "",
        "## Paired information contrasts",
        "",
        "Positive Accuracy/Macro-F1 deltas favor the candidate; negative CE deltas favor the candidate. `positive_seed_count` counts positive numerical differences regardless of whether that direction improves the metric. Each dataset has three paired seed differences; summaries are descriptive and do not establish statistical significance.",
        "",
        "| Dataset | Contrast | Metric | Mean paired delta | Sample SD | Positive seeds |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in paired:
        if row["row_type"] != "dataset_summary":
            continue
        percent = row["metric"] in {"validation_accuracy", "validation_macro_f1"}
        info_lines.append(
            f"| {row['dataset']} | {row['contrast']} | {row['metric']} | {_fmt(float(row['mean_paired_delta']), float(row['sample_sd_paired_delta']), percent=percent)} | {int(row['positive_seed_count'])}/3 |"
        )
    info_lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- I1−I0 estimates the value of adding physical neighborhood channels within this fixed probe.",
        "- I2−I1 changes where and how the same physical information is processed across modalities. It is an inductive-bias comparison, not additional raw information.",
        "- I3−I2 opens real Text and Visual semantic channels, so it changes the active input set and is not a pure topology-only contrast.",
        "- I3−I4 is the topology-content comparison: the model, slots, initialization, optimizer, and degrees are paired while the semantic graph topology changes. It uses one frozen random topology per view and dataset, so the null is one topology sample.",
        "- Equal total parameter counts do not imply equal effective capacity: closed input slots can receive no input-related gradient.",
        "- Historical V7B.1 R0 Raw is shown only as a capability reference. It is not capacity-matched to this probe.",
        "",
        "## Train/Validation fit and checkpoint selection",
        "",
        "The primary endpoint uses the Best-Accuracy checkpoint. Train metrics below are evaluated at that same checkpoint; the separate Best-Macro-F1 checkpoint is not mixed into the primary tables.",
        "",
        "| Dataset | Treatment | Train Accuracy (%) | Validation Accuracy (%) | Train−Validation Accuracy gap (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            vals = [runs[(dataset, condition, seed)] for seed in SEEDS]
            train_acc = [float(row["best_accuracy_checkpoint_metrics"]["train"]["accuracy"]) for row in vals]
            gap = [float(row["best_accuracy_checkpoint_metrics"]["train"]["accuracy"] - row["best_accuracy_checkpoint_metrics"]["validation"]["accuracy"]) for row in vals]
            val_acc = [float(row["best_accuracy_checkpoint_metrics"]["validation"]["accuracy"]) for row in vals]
            info_lines.append(
                f"| {dataset} | {condition} | {_fmt(statistics.mean(train_acc), statistics.stdev(train_acc), percent=True)} | "
                f"{_fmt(statistics.mean(val_acc), statistics.stdev(val_acc), percent=True)} | "
                f"{_fmt(statistics.mean(gap), statistics.stdev(gap), percent=True)} |"
            )
    info_lines += [
        "",
        "| Dataset | Treatment | Best-F1 minus Best-Accuracy: Accuracy (pp) | Macro-F1 (pp) | CE |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            values = []
            for seed in SEEDS:
                row = runs[(dataset, condition, seed)]
                acc = row["best_accuracy_checkpoint_metrics"]["validation"]
                f1 = row["best_f1_checkpoint_metrics"]["validation"]
                values.append((
                    float(f1["accuracy"] - acc["accuracy"]),
                    float(f1["macro_f1"] - acc["macro_f1"]),
                    float(f1["cross_entropy"] - acc["cross_entropy"]),
                ))
            columns = list(zip(*values))
            info_lines.append(
                f"| {dataset} | {condition} | {_fmt(statistics.mean(columns[0]), statistics.stdev(columns[0]), percent=True)} | "
                f"{_fmt(statistics.mean(columns[1]), statistics.stdev(columns[1]), percent=True)} | "
                f"{_fmt(statistics.mean(columns[2]), statistics.stdev(columns[2]))} |"
            )
    info_lines += [
        "",
        "## Physical and semantic neighborhood diagnostics",
        "",
        "See `GRAPH_SIGNAL_REPORT.md` and `data/graph_structure_stats.csv` for feature geometry, degree, connectivity, and physical-overlap diagnostics. Graph homophily is reported only for Train–Train edges and is not a predictive utility estimate. Degree matching does not match component structure or physical overlap; see `data/random_graph_audit.json`.",
        "",
        "## Train/Validation fit and checkpoint criteria",
        "",
        "`data/runs.csv` and `data/epoch_history.csv` retain Train/Validation CE, Accuracy, Macro-F1, and the Train–Validation Accuracy gap. `data/checkpoint_tradeoffs.csv` compares independently selected Best-Accuracy and Best-Macro-F1 checkpoints; endpoint claims above use only Best-Accuracy checkpoints.",
        "",
        "## Historical R0 Raw reference",
        "",
        "| Dataset | Treatment | Probe−R0 Accuracy (pp) | Probe−R0 Macro-F1 (pp) | Probe parameters | R0 parameters |",
        "|---|---|---:|---:|---:|---:|",
    ]
    r0_by_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in r0_rows:
        r0_by_block[(row["dataset"], row["condition"])].append(row)
    for dataset in DATASETS:
        for condition in CONDITIONS:
            values = r0_by_block[(dataset, condition)]
            acc = [float(row["accuracy_difference_probe_minus_r0"]) for row in values]
            f1 = [float(row["macro_f1_difference_probe_minus_r0"]) for row in values]
            first = values[0]
            info_lines.append(
                f"| {dataset} | {condition} | {_fmt(statistics.mean(acc), statistics.stdev(acc), percent=True)} | "
                f"{_fmt(statistics.mean(f1), statistics.stdev(f1), percent=True)} | {first['probe_parameters']} | {first['r0_raw_parameters']} |"
            )
    info_lines += [
        "",
        "R0 Raw uses a different architecture and capacity; these gaps calibrate probe strength and do not isolate graph utility. Per-seed reference values are retained in `data/r0_reference_comparison.csv`.",
        "",
    ]

    # Compact, descriptive cross-dataset consistency readout for the decision memo.
    delta_summary = {
        (row["dataset"], row["contrast"], row["metric"]): row
        for row in paired if row["row_type"] == "dataset_summary"
    }
    decision = [
        "# MAG Information Utility V1 — Decision Memo",
        "",
        "This memo reports descriptive evidence from the frozen 3-dataset × 5-treatment × 3-seed experiment. It does not claim significance or generalize beyond this probe and these datasets.",
        "",
        "## Evidence by question",
        "",
    ]
    contrast_questions = [
        ("I1-I0", "Physical propagation"),
        ("I2-I1", "Cross-modal physical processing"),
        ("I3-I2", "Opening real semantic channels"),
        ("I3-I4", "Real versus degree-matched random topology"),
    ]
    for contrast, label in contrast_questions:
        direction_rows = [delta_summary[(dataset, contrast, "validation_accuracy")] for dataset in DATASETS]
        f1_rows = [delta_summary[(dataset, contrast, "validation_macro_f1")] for dataset in DATASETS]
        ce_rows = [delta_summary[(dataset, contrast, "validation_cross_entropy")] for dataset in DATASETS]
        ups = sum(int(row["positive_seed_count"]) for row in direction_rows)
        signs = ", ".join(f"{dataset}: {float(direction_rows[i]['mean_paired_delta'])*100:+.2f} pp" for i, dataset in enumerate(DATASETS))
        decision.append(
            f"- **{label} ({contrast})**: Accuracy paired mean deltas are {signs}; positive seed differences: {ups}/9. Macro-F1 and CE paired summaries are available in `data/paired_information_gains.csv`. Treat a mixed sign pattern as dataset- or seed-dependent evidence, not a universal effect."
        )
    decision += [
        "",
        "## Simple-probe capacity calibration",
        "",
    ]
    for dataset in DATASETS:
        row_i1 = r0_by_block[(dataset, "I1")]
        row_i2 = r0_by_block[(dataset, "I2")]
        i1 = [float(row["accuracy_difference_probe_minus_r0"]) for row in row_i1]
        i2 = [float(row["accuracy_difference_probe_minus_r0"]) for row in row_i2]
        decision.append(
            f"- **{dataset}**: I1−R0 Accuracy = {_fmt(statistics.mean(i1), statistics.stdev(i1), percent=True)} pp; I2−R0 = {_fmt(statistics.mean(i2), statistics.stdev(i2), percent=True)} pp. R0 has a different, generally larger model, so a gap limits how far this probe's relation findings should be generalized."
        )
    decision += [
        "",
        "## What this supports for future MAG graph experts",
        "",
        "Use the observed contrasts to decide which information paths deserve a capacity-matched follow-up on a stronger physical-graph backbone. A stable physical gain would support experts that process physical neighborhoods by modality; a repeatable I2 gain would motivate testing explicit cross-modal neighborhood processing; a repeatable I3−I4 gain would justify retaining semantic topology as a candidate relation. None of these probe results alone establishes a general architectural rule.",
        "",
        "## Capability calibration and caveats",
        "",
        "Compare I0–I4 with historical V7B.1 R0 Raw using `data/r0_reference_comparison.csv`. R0 has different parameter count and architecture, so a gap bounds how strongly this simple probe can support claims about semantic utility. Random graphs preserve the exact degree sequence and edge count, but their connected components and physical overlap differ from the real KNN views. Only one random topology per dataset and source was evaluated.",
        "",
        "No Test labels were used. Validation subgroup analyses are descriptive and do not create filtering rules. Small groups are marked in `data/node_group_metrics.csv` and `data/conditional_error_analysis.csv`.",
        "",
    ]
    return "\n".join(info_lines), "", "\n".join(decision)


def _render_node_report(metric_rows: list[dict[str, Any]], conditional_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MAG Information Utility V1 — Validation Node Groups",
        "",
        "Groups were defined before training from unlabeled graph degree and feature-to-neighborhood similarity, using whole-graph quantiles. Validation labels were used only to calculate the metrics in the tables. Validation nodes are not independent training replications; all results are descriptive across three training seeds.",
        "",
        "## Conditional error changes",
        "",
        "For I3−I4 and I2−I1, positive `reference_minus_candidate_mean_ce` means lower candidate CE. Fractions are paired over the same Validation nodes. Groups with fewer than 30 nodes are marked for caution in the CSV.",
        "",
        "| Dataset | Contrast | Group variable | Group | Mean reference−candidate CE | Improved | Worsened | Ties | Flipped predictions |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in conditional_rows:
        grouped[(row["dataset"], row["contrast"], row["group_variable"], row["group"])].append(row)
    for key, values in sorted(grouped.items()):
        mean = statistics.mean(float(row["reference_minus_candidate_mean_ce"]) for row in values)
        improved = statistics.mean(float(row["improved_node_fraction"]) for row in values)
        worsened = statistics.mean(float(row["worsened_node_fraction"]) for row in values)
        tied = statistics.mean(float(row["tied_node_fraction"]) for row in values)
        flips = statistics.mean(float(row["prediction_flip_fraction"]) for row in values)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {mean:.5f} | {improved:.1%} | {worsened:.1%} | {tied:.1%} | {flips:.1%} |")
    lines += [
        "",
        "## Treatment metrics by predeclared group",
        "",
        "Full per-seed Accuracy, all-configured-class Macro-F1, CE, sample count, and small-group flags are in `data/node_group_metrics.csv`. Per-node conditional changes are in `data/conditional_error_analysis.csv`.",
        "",
        "These subgroup summaries can identify where errors change. They do not justify label-informed graph edits or node filters without a separate training-only design and held-out confirmation.",
        "",
    ]
    return "\n".join(lines)


def _write_reports(
    runs: dict[tuple[str, str, int], dict[str, Any]],
    paired: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    conditional_rows: list[dict[str, Any]],
    r0_rows: list[dict[str, Any]],
) -> None:
    info, _, decision = _render_report(runs, paired, r0_rows)
    (RESEARCH / "INFORMATION_UTILITY_REPORT.md").write_text(info, encoding="utf-8")
    (RESEARCH / "NODE_GROUP_ANALYSIS.md").write_text(_render_node_report(node_rows, conditional_rows), encoding="utf-8")
    (RESEARCH / "DECISION_MEMO.md").write_text(decision, encoding="utf-8")


def _merge_failures() -> list[Any]:
    tracked = DATA / "failures.json"
    runtime = OUTPUT / "failures.json"
    values: list[Any] = []
    for path in (tracked, runtime):
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                values.extend(value)
    seen = set()
    unique = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def analyze() -> dict[str, Any]:
    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    histories: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                key = (dataset, condition, seed)
                runs[key], histories[key] = _validate_run(*key)

    source_commits = {row["source_commit"] for row in runs.values()}
    frozen_core_sha256 = _validate_source_commits(source_commits)
    for dataset in DATASETS:
        text_hashes = {runs[(dataset, c, s)]["provenance"]["random_text_cache_sha256"] for c in CONDITIONS for s in SEEDS}
        visual_hashes = {runs[(dataset, c, s)]["provenance"]["random_visual_cache_sha256"] for c in CONDITIONS for s in SEEDS}
        if len(text_hashes) != 1 or len(visual_hashes) != 1:
            raise ValueError(f"random semantic topology changed across conditions/seeds for {dataset}")

    run_rows = _run_rows(runs)
    epoch_rows = _epoch_rows(histories)
    paired = _paired_rows(runs)
    checkpoint_rows = _checkpoint_rows(runs)
    group_defs = _load_group_definitions()
    node_rows, conditional_rows = _node_group_rows(runs, group_defs)
    r0_rows = _r0_reference_rows(runs)

    write_csv(DATA / "runs.csv", run_rows, list(run_rows[0]))
    write_csv(DATA / "epoch_history.csv", epoch_rows, list(epoch_rows[0]))
    write_csv(DATA / "paired_information_gains.csv", paired, list(paired[0]))
    write_csv(DATA / "checkpoint_tradeoffs.csv", checkpoint_rows, list(checkpoint_rows[0]))
    write_csv(DATA / "node_group_metrics.csv", node_rows, list(node_rows[0]))
    write_csv(DATA / "conditional_error_analysis.csv", conditional_rows, list(conditional_rows[0]))
    write_csv(DATA / "r0_reference_comparison.csv", r0_rows, list(r0_rows[0]))
    write_json(DATA / "failures.json", _merge_failures())

    random_audit = _load_json(DATA / "random_graph_audit.json")
    graph_manifest = _load_json(DATA / "pre_run_data_manifest.json")
    checks = {
        "expected_runs": 45,
        "observed_runs": len(run_rows),
        "all_runs_complete": len(run_rows) == 45,
        "implementation_freeze_commit": IMPLEMENTATION_FREEZE_COMMIT,
        "frozen_model_protocol_sha256": frozen_core_sha256,
        "test_evaluation_disabled_for_all_runs": all(row["evaluate_test"] is False and row["test_labels_indexed"] is False for row in runs.values()),
        "all_checkpoints_saved_and_reloaded": all(row["best_accuracy_reload_reproduced"] and row["best_f1_reload_reproduced"] for row in runs.values()),
        "paired_initialization_validated": True,
        "finite_epoch_histories_and_metrics": True,
        "random_topology_frozen_across_training_seeds": True,
        "random_graph_audit": random_audit,
        "data_and_cache_manifest": graph_manifest,
        "source_commits": sorted(source_commits),
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(DATA / "implementation_checks.json", checks)
    files = [
        "EXPERIMENT_DESIGN.md",
        "IMPLEMENTATION_AUDIT.md",
        "GRAPH_SIGNAL_REPORT.md",
        "data/graph_structure_stats.csv",
        "data/train_only_label_relation_stats.csv",
        "data/random_graph_audit.json",
        "data/pre_run_data_manifest.json",
        "data/node_group_definitions.json",
        "data/runs.csv",
        "data/epoch_history.csv",
        "data/paired_information_gains.csv",
        "data/checkpoint_tradeoffs.csv",
        "data/node_group_metrics.csv",
        "data/conditional_error_analysis.csv",
        "data/r0_reference_comparison.csv",
        "data/implementation_checks.json",
        "data/failures.json",
    ]
    manifest = {
        "study": "MAG Information Utility V1",
        "branch": "exp/mag_information_utility_v1",
        "baseline_commit": BASELINE_COMMIT,
        "implementation_freeze_commit": IMPLEMENTATION_FREEZE_COMMIT,
        "source_commits": sorted(source_commits),
        "frozen_model_protocol_sha256": frozen_core_sha256,
        "datasets": list(DATASETS),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "expected_runs": 45,
        "completed_runs": len(run_rows),
        "protocol": "unified_full_graph_nc_v1",
        "test_evaluation": False,
        "random_semantic_graphs_fixed_across_training_seeds": True,
        "files": {relative: sha256_file(RESEARCH / relative) for relative in files if (RESEARCH / relative).is_file()},
    }
    _write_reports(runs, paired, node_rows, conditional_rows, r0_rows)
    for relative in ("INFORMATION_UTILITY_REPORT.md", "NODE_GROUP_ANALYSIS.md", "DECISION_MEMO.md"):
        manifest["files"][relative] = sha256_file(RESEARCH / relative)
    write_json(DATA / "manifest.json", manifest)
    return {
        "completed_runs": len(run_rows),
        "freeze_commit": IMPLEMENTATION_FREEZE_COMMIT,
        "source_commits": sorted(source_commits),
        "outputs": ["runs.csv", "epoch_history.csv", "paired_information_gains.csv", "checkpoint_tradeoffs.csv", "node_group_metrics.csv", "conditional_error_analysis.csv", "r0_reference_comparison.csv"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-and-analyze", action="store_true", help="validate all 45 runs and write tracked analysis artifacts")
    args = parser.parse_args()
    if not args.validate_and_analyze:
        parser.error("pass --validate-and-analyze after the full frozen matrix is complete")
    print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
