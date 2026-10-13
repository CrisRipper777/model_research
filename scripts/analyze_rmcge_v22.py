#!/usr/bin/env python3
"""Validate and summarize the frozen RM-CGE V2.2 evidence bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v22 import (  # noqa: E402
    CLASS_FIELDS,
    CONDITIONS,
    DATASETS,
    DATA_DIR,
    DEGREE_FIELDS,
    GATE_FIELDS,
    HISTORY_FIELDS,
    INTERVENTION_FIELDS,
    MESSAGE_FIELDS,
    RESOURCE_FIELDS,
    ROUTER_FIELDS,
    RUN_FIELDS,
    SEEDS,
    STUDY,
    TRANSITION_FIELDS,
    CONFUSION_FIELDS,
    read_rows,
    write_rows,
)


CONTRASTS = (
    ("M1", "M0"), ("M2", "M0"), ("M3", "M0"), ("M3", "M1"),
    ("M3", "M2"), ("M4", "M3"), ("M4", "M0"),
)
METRICS = (
    ("best_accuracy.val_acc", "val_acc", "fraction"),
    ("best_accuracy.val_macro_f1", "val_macro_f1", "fraction"),
    ("best_accuracy.val_ce", "val_ce", "raw"),
    ("best_f1.val_acc", "best_f1_val_acc", "fraction"),
    ("best_f1.val_macro_f1", "best_f1_val_macro_f1", "fraction"),
    ("best_f1.val_ce", "best_f1_val_ce", "raw"),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: str, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {label}: {value}")
    return parsed


def _assert_finite_nested(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value in {label}")
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_nested(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_nested(item, label)


def _key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["dataset"], row["condition"], int(row["seed"])


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _validate_table(name: str, rows: list[dict[str, str]], fields: list[str], keys: tuple[str, ...], expected_count: int | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"{name} is empty")
    actual_fields = list(rows[0])
    if actual_fields != fields:
        raise ValueError(f"{name} header mismatch: {actual_fields} != {fields}")
    seen = set()
    for index, row in enumerate(rows, 2):
        key = tuple(row[field] for field in keys)
        if key in seen:
            raise ValueError(f"duplicate key in {name} at row {index}: {key}")
        seen.add(key)
        for field, value in row.items():
            if field in keys or value == "":
                continue
            # JSON and text path fields are not scalar metrics.
            if field in {"status", "run_dir", "best_accuracy_checkpoint", "best_f1_checkpoint", "run_metrics_path", "semantic_cache_sha256", "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint", "split_fingerprint", "checkpoint", "intervention", "degree_group", "modality", "mode", "reference_condition", "error", "cuda_device_name"}:
                continue
            try:
                parsed = json.loads(value) if value.startswith(("[", "{")) else float(value)
            except (ValueError, json.JSONDecodeError):
                continue
            _assert_finite_nested(parsed, f"{name}.{field} at row {index}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"{name} row count {len(rows)} != expected {expected_count}")
    return {"rows": len(rows), "fields": actual_fields, "unique_keys": len(seen), "key_fields": list(keys)}


def validate_runs(runs: list[dict[str, str]], manifest: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, str]]:
    expected = {(dataset, condition, seed) for dataset in DATASETS for condition in CONDITIONS for seed in SEEDS}
    actual = {_key(row) for row in runs}
    if actual != expected:
        raise ValueError(f"run key set incomplete: missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}")
    lookup = {_key(row): row for row in runs}
    for key, row in lookup.items():
        if row["status"] != "complete" or row["test_evaluation"].lower() != "false":
            raise ValueError(f"run is incomplete or Test enabled: {key}")
        if any(row.get(field, "") == "" for field in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce")):
            raise ValueError(f"missing selected metrics: {key}")
        for field in (
            "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce",
            "train_acc", "train_macro_f1", "train_ce", "train_ce_eval", "train_val_acc_gap",
            "train_val_macro_f1_gap", "train_val_ce_gap", "wall_seconds",
        ):
            _finite(row[field], f"runs.csv {key} {field}")
        if not row["semantic_cache_sha256"] or not row["split_fingerprint"]:
            raise ValueError(f"missing cache/split provenance: {key}")
        dataset, _condition, seed = key
        provenance = manifest["datasets"][dataset]
        expected_split = provenance["seed_split_fingerprints"][str(seed)]
        expected = {
            "semantic_cache_sha256": provenance["cache_sha256"],
            "feature_fingerprint": provenance["feature_fingerprint"],
            "candidate_fingerprint": provenance["candidate_fingerprint"],
            "v7b_fingerprint": provenance["v7b_fingerprint"],
            "split_fingerprint": expected_split,
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(f"manifest provenance mismatch for {key} field={field}: {row[field]} != {value}")
    for dataset in DATASETS:
        for seed in SEEDS:
            paired = [lookup[(dataset, condition, seed)] for condition in CONDITIONS]
            if len({row["initial_model_sha256"] for row in paired}) != 1:
                raise ValueError(f"initial model state mismatch across arms: {dataset}/seed_{seed}")
            if len({row["initial_classifier_sha256"] for row in paired}) != 1:
                raise ValueError(f"initial classifier mismatch across arms: {dataset}/seed_{seed}")
            if len({row["split_fingerprint"] for row in paired}) != 1:
                raise ValueError(f"official split mismatch across arms: {dataset}/seed_{seed}")
            if len({row["semantic_cache_sha256"] for row in paired}) != 1:
                raise ValueError(f"semantic cache mismatch across arms: {dataset}/seed_{seed}")
    if manifest.get("test_evaluation") is not False or manifest.get("test_label_values_indexed") is not False:
        raise ValueError("manifest does not attest Test disabled")
    return lookup


def paired_results(lookup: dict[tuple[str, str, int], dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for left, right in CONTRASTS:
            for seed in SEEDS:
                left_row, right_row = lookup[(dataset, left, seed)], lookup[(dataset, right, seed)]
                for metric, field, _kind in METRICS:
                    left_value, right_value = float(left_row[field]), float(right_row[field])
                    rows.append({
                        "dataset": dataset, "seed": seed, "contrast": f"{left}-{right}", "metric": metric,
                        "difference": left_value - right_value, "left_value": left_value, "right_value": right_value,
                    })
    return rows


def summary_rows(lookup: dict[tuple[str, str, int], dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    metrics = [*METRICS, ("best_accuracy.epoch", "best_accuracy_epoch", "raw"), ("best_f1.epoch", "best_f1_epoch", "raw"),
               ("train_val.acc_gap", "train_val_acc_gap", "fraction"), ("train_val.macro_f1_gap", "train_val_macro_f1_gap", "fraction"),
               ("train_val.eval_ce_gap", "train_val_ce_gap", "raw"),
               ("runtime.wall_seconds", "wall_seconds", "raw"), ("runtime.peak_allocated_bytes", "peak_allocated_bytes", "raw"),
               ("parameters.model", "model_parameters", "raw"), ("parameters.classifier", "classifier_parameters", "raw")]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            paired = [lookup[(dataset, condition, seed)] for seed in SEEDS]
            for name, field, kind in metrics:
                values = [float(row[field]) for row in paired]
                mean, sd = _mean_sd(values)
                rows.append({
                    "dataset": dataset, "condition": condition, "metric": name, "mean": mean,
                    "sample_sd": sd, "seed_42": values[0], "seed_43": values[1], "seed_44": values[2],
                    "unit": "fraction" if kind == "fraction" else "raw",
                })
    return rows


def _fmt(values: list[float], *, pct: bool = False, digits: int = 2) -> str:
    mean, sd = _mean_sd(values)
    scale = 100 if pct else 1
    return f"{mean * scale:.{digits}f} ± {sd * scale:.{digits}f}"


def _metric_table(lookup, metric_field: str, heading: str, percent: bool = False) -> list[str]:
    lines = [heading, "| Dataset | " + " | ".join(CONDITIONS) + " |", "|---|" + "---:|" * len(CONDITIONS)]
    for dataset in DATASETS:
        cells = []
        for condition in CONDITIONS:
            values = [float(lookup[(dataset, condition, seed)][metric_field]) for seed in SEEDS]
            cells.append(_fmt(values, pct=percent))
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    return lines


def _paired_summary_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["dataset"], row["contrast"], row["metric"]) for row in paired})
    for dataset, contrast, metric in keys:
        rows = [row for row in paired if (row["dataset"], row["contrast"], row["metric"]) == (dataset, contrast, metric)]
        values = [float(row["difference"]) for row in rows]
        mean, sd = _mean_sd(values)
        output.append({
            "dataset": dataset, "contrast": contrast, "metric": metric, "mean_paired_delta": mean,
            "sample_sd": sd, "seed_42": values[0], "seed_43": values[1], "seed_44": values[2], "n": len(values),
        })
    return output


def _paired_table(summary: list[dict[str, Any]], metric: str, *, percentage: bool = False) -> list[str]:
    lines = [f"| Dataset | Contrast | Mean paired Δ | SD | Seed 42 | Seed 43 | Seed 44 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["metric"] != metric:
            continue
        scale = 100 if percentage else 1
        lines.append(
            f"| {row['dataset']} | {row['contrast']} | {row['mean_paired_delta']*scale:+.4f} | {row['sample_sd']*scale:.4f} | "
            f"{row['seed_42']*scale:+.4f} | {row['seed_43']*scale:+.4f} | {row['seed_44']*scale:+.4f} |"
        )
    return lines


def _content_summary(gates: list[dict[str, str]], messages: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in DATASETS:
        result[dataset] = {}
        for condition in ("M1", "M2", "M3", "M4"):
            gate_rows = [row for row in gates if row["dataset"] == dataset and row["condition"] == condition]
            message_rows = [row for row in messages if row["dataset"] == dataset and row["condition"] == condition]
            result[dataset][condition] = {
                "gate_channel_std_mean": statistics.mean([statistics.mean(json.loads(row["channel_std"])) for row in gate_rows]) if gate_rows else None,
                "gate_saturation_fraction_mean": statistics.mean([float(row["abs_ge_0_99_fraction"]) for row in gate_rows]) if gate_rows else None,
                "query_roll_gate_rms_mean": statistics.mean([float(row["receiver_query_roll_gate_rms"]) for row in gate_rows]) if gate_rows else None,
                "content_normalized_rms_mean": statistics.mean([float(row["content_normalized_rms"]) for row in message_rows]) if message_rows else None,
                "content_cross_cosine_mean": statistics.mean([float(row["content_cross_cosine"]) for row in message_rows]) if message_rows else None,
                "eta_contribution_rms_mean": statistics.mean([
                    statistics.mean(json.loads(row["eta_weighted_rms_by_expert"])) for row in message_rows
                ]) if message_rows else None,
            }
    return result


def audit_and_analyze() -> dict[str, Any]:
    manifest = _read_json(DATA_DIR / "manifest.json")
    runs = read_rows(DATA_DIR / "runs.csv")
    lookup = validate_runs(runs, manifest)
    history = read_rows(DATA_DIR / "epoch_history.csv")
    classes = read_rows(DATA_DIR / "classwise_metrics.csv")
    confusion = read_rows(DATA_DIR / "confusion_matrices.csv")
    degrees = read_rows(DATA_DIR / "degree_group_metrics.csv")
    transitions = read_rows(DATA_DIR / "prediction_transitions.csv")
    gates = read_rows(DATA_DIR / "content_gate_diagnostics.csv")
    messages = read_rows(DATA_DIR / "message_scale_diagnostics.csv")
    routers = read_rows(DATA_DIR / "expert_router_diagnostics.csv")
    interventions = read_rows(DATA_DIR / "frozen_interventions.csv")
    resources = read_rows(DATA_DIR / "resource_profile.csv")
    artifacts = _read_json(DATA_DIR / "artifact_integrity.json")
    implementation_checks = _read_json(DATA_DIR / "implementation_checks.json")
    gradient_audit = _read_json(DATA_DIR / "gradient_audit.json")
    failures = _read_json(DATA_DIR / "failures.json") if (DATA_DIR / "failures.json").exists() else {"failures": []}

    schemas = {
        "runs.csv": _validate_table("runs.csv", runs, RUN_FIELDS, ("dataset", "condition", "seed"), 45),
        "epoch_history.csv": _validate_table("epoch_history.csv", history, HISTORY_FIELDS, ("dataset", "condition", "seed", "epoch")),
        "classwise_metrics.csv": _validate_table("classwise_metrics.csv", classes, CLASS_FIELDS, ("dataset", "condition", "seed", "checkpoint", "class_id")),
        "confusion_matrices.csv": _validate_table("confusion_matrices.csv", confusion, CONFUSION_FIELDS, ("dataset", "condition", "seed", "checkpoint", "true_class", "predicted_class")),
        "degree_group_metrics.csv": _validate_table("degree_group_metrics.csv", degrees, DEGREE_FIELDS, ("dataset", "condition", "seed", "checkpoint", "degree_group")),
        "prediction_transitions.csv": _validate_table("prediction_transitions.csv", transitions, TRANSITION_FIELDS, ("dataset", "condition", "seed", "class_id")),
        "content_gate_diagnostics.csv": _validate_table("content_gate_diagnostics.csv", gates, GATE_FIELDS, ("dataset", "condition", "seed", "modality")),
        "message_scale_diagnostics.csv": _validate_table("message_scale_diagnostics.csv", messages, MESSAGE_FIELDS, ("dataset", "condition", "seed", "modality")),
        "expert_router_diagnostics.csv": _validate_table("expert_router_diagnostics.csv", routers, ROUTER_FIELDS, ("dataset", "condition", "seed", "modality")),
        "frozen_interventions.csv": _validate_table("frozen_interventions.csv", interventions, INTERVENTION_FIELDS, ("dataset", "condition", "seed", "intervention")),
        "resource_profile.csv": _validate_table("resource_profile.csv", resources, RESOURCE_FIELDS, ("dataset", "condition", "seed"), 45),
    }
    expected_diagnostic_rows = {
        "degree_group_metrics.csv": 45 * 2 * 5,
        "content_gate_diagnostics.csv": 36 * 2,
        "message_scale_diagnostics.csv": 45 * 2,
        "expert_router_diagnostics.csv": 45 * 2,
        "frozen_interventions.csv": 9 * (3 + 3 + 3 + 4 + 5),
    }
    diagnostic_tables = {
        "degree_group_metrics.csv": degrees,
        "content_gate_diagnostics.csv": gates,
        "message_scale_diagnostics.csv": messages,
        "expert_router_diagnostics.csv": routers,
        "frozen_interventions.csv": interventions,
    }
    for name, expected_count in expected_diagnostic_rows.items():
        if len(diagnostic_tables[name]) != expected_count:
            raise ValueError(f"{name} row count {len(diagnostic_tables[name])} != {expected_count}")
    run_keys = set(lookup)
    history_keys = {(row["dataset"], row["condition"], int(row["seed"])) for row in history}
    if history_keys != run_keys:
        raise ValueError("epoch history does not cover exactly the 45 formal runs")
    for key in run_keys:
        epochs = sorted(int(row["epoch"]) for row in history if (row["dataset"], row["condition"], int(row["seed"])) == key)
        if epochs != list(range(1, epochs[-1] + 1)) or epochs[-1] > 300:
            raise ValueError(f"epoch history has a gap or invalid horizon: {key}")
    for key in run_keys:
        observed = {
            checkpoint: {
                int(row["class_id"])
                for row in classes
                if (row["dataset"], row["condition"], int(row["seed"])) == key and row["checkpoint"] == checkpoint
            }
            for checkpoint in ("best_accuracy", "best_f1")
        }
        if set(observed) != {"best_accuracy", "best_f1"} or observed["best_accuracy"] != observed["best_f1"] or not observed["best_accuracy"]:
            raise ValueError(f"classwise checkpoints/classes incomplete or inconsistent: {key}")
    checkpoint_records = artifacts.get("checkpoints", [])
    if len(checkpoint_records) != 90:
        raise ValueError("expected exactly 90 checkpoint integrity records")
    checkpoint_keys = set()
    for record in checkpoint_records:
        key = (record["dataset"], record["condition"], int(record["seed"]), record["checkpoint"])
        if key in checkpoint_keys:
            raise ValueError(f"duplicate checkpoint integrity record: {key}")
        checkpoint_keys.add(key)
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint missing from integrity record {key}: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"] or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"checkpoint integrity mismatch: {key}")
    expected_checkpoint_keys = {
        (dataset, condition, seed, checkpoint)
        for dataset in DATASETS for condition in CONDITIONS for seed in SEEDS
        for checkpoint in ("best_accuracy", "best_f1")
    }
    if checkpoint_keys != expected_checkpoint_keys:
        raise ValueError(f"checkpoint integrity key mismatch: missing={expected_checkpoint_keys-checkpoint_keys} unexpected={checkpoint_keys-expected_checkpoint_keys}")
    check_rows = implementation_checks.get("formal_runs", [])
    check_lookup = {(row["dataset"], row["condition"], int(row["seed"])): row for row in check_rows}
    if len(check_rows) != 45 or set(check_lookup) != set(lookup):
        raise ValueError("implementation_checks formal run coverage is not exactly 45")
    for key, row in check_lookup.items():
        if not row.get("checkpoint_reload_verified") or not row.get("checkpoint_val_metrics_match") or not row.get("validation_rng_unchanged"):
            raise ValueError(f"checkpoint or Validation RNG audit failed: {key}")
        if row["initial_model_sha256"] != lookup[key]["initial_model_sha256"] or row["initial_classifier_sha256"] != lookup[key]["initial_classifier_sha256"]:
            raise ValueError(f"implementation-check initialization fingerprint mismatch: {key}")
        if row.get("test_evaluation") is not False or row.get("test_label_values_indexed") is not False:
            raise ValueError(f"Test-label boundary assertion failed: {key}")
        if key[1] != "M0" and not row.get("content_final_layer_first_step_gradient_nonzero"):
            raise ValueError(f"content Gate first-step gradient missing: {key}")
        if key[1] != "M0":
            update = float(row["best_checkpoint_content_gate_output_weight_update_rms"])
            if not math.isfinite(update) or update <= 0:
                raise ValueError(f"trained content Gate output layer has no finite nonzero checkpoint update: {key}")
    gradient_rows = gradient_audit.get("runs", [])
    gradient_keys = {
        (row["dataset"], row["condition"], int(row["seed"]))
        for row in gradient_rows
    }
    if len(gradient_rows) != 45 or gradient_keys != set(lookup):
        raise ValueError("gradient audit does not cover all 45 runs")
    for row in classes:
        for field in ("precision", "recall", "f1"):
            _finite(row[field], f"classwise {field}")
    for row in interventions:
        if row["status"] != "ok":
            raise ValueError("failed frozen intervention present")

    paired = paired_results(lookup)
    paired_summary = _paired_summary_rows(paired)
    summary = summary_rows(lookup)
    write_rows(DATA_DIR / "paired_comparisons.csv", paired, ["dataset", "seed", "contrast", "metric", "difference", "left_value", "right_value"])
    write_rows(DATA_DIR / "paired_comparison_summary.csv", paired_summary, ["dataset", "contrast", "metric", "mean_paired_delta", "sample_sd", "seed_42", "seed_43", "seed_44", "n"])
    write_rows(DATA_DIR / "summary.csv", summary, ["dataset", "condition", "metric", "mean", "sample_sd", "seed_42", "seed_43", "seed_44", "unit"])

    schema_audit = {
        "generated_utc": manifest.get("last_batch_completed_utc"),
        "test_evaluation": False,
        "test_label_values_indexed": False,
        "tables": schemas,
        "paired_comparison_rows": len(paired),
        "paired_summary_rows": len(paired_summary),
        "checkpoint_hash_records": len(artifacts["checkpoints"]),
        "checkpoint_sha_and_size_verified": len(checkpoint_records),
        "implementation_check_run_count": len(check_rows),
        "gradient_audit_run_count": len(gradient_keys),
        "failure_record_count": len(failures.get("failures", [])),
        "no_test_metrics": all(
            not any(key.startswith("test_") and key not in {"test_evaluation", "test_label_values_indexed"} for key in row)
            for row in runs
        ),
    }
    if not schema_audit["no_test_metrics"]:
        raise ValueError("Test metric field found in registered runs")
    atomic = DATA_DIR / "analysis_audit.json"
    atomic.write_text(json.dumps(schema_audit, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    integrity = {**artifacts, "analysis_schema_audit": schema_audit, "complete_run_count": len(runs)}
    (DATA_DIR / "artifact_integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    content = _content_summary(gates, messages)
    (DATA_DIR / "content_summary.json").write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    report = _pilot_report(lookup, paired_summary, manifest, schema_audit, failures)
    (STUDY / "PILOT_REPORT.md").write_text(report, encoding="utf-8")
    (STUDY / "CONTENT_MECHANISM_REPORT.md").write_text(_mechanism_report(content, interventions, routers, messages), encoding="utf-8")
    (STUDY / "CLASSWISE_REPORT.md").write_text(_classwise_report(classes, degrees, transitions), encoding="utf-8")
    (STUDY / "DECISION_MEMO.md").write_text(_decision_memo(lookup, classes, degrees, messages, gates, routers, interventions, resources, failures), encoding="utf-8")
    return schema_audit


def atomic_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _pilot_report(lookup, paired_summary, manifest, audit, failures) -> str:
    lines = [
        "# RM-CGE V2.2 Pilot and Paired Results", "",
        "All reported scores are Train/Validation only. No Test evaluation was run and Test labels were masked before the NC trainer received data. Means and sample standard deviations use seeds 42, 43, and 44.", "",
        "## Best Accuracy checkpoint", "",
    ]
    lines += _metric_table(lookup, "val_acc", "### Validation Accuracy (%)", True) + [""]
    lines += _metric_table(lookup, "val_macro_f1", "### Validation Macro-F1 (%)", True) + [""]
    lines += _metric_table(lookup, "val_ce", "### Validation CE") + [""]
    lines += ["## Best Macro-F1 checkpoint", ""]
    lines += _metric_table(lookup, "best_f1_val_acc", "### Validation Accuracy (%)", True) + [""]
    lines += _metric_table(lookup, "best_f1_val_macro_f1", "### Validation Macro-F1 (%)", True) + [""]
    lines += _metric_table(lookup, "best_f1_val_ce", "### Validation CE") + [""]
    lines += [
        "## Best Accuracy checkpoint: Train and Validation CE", "",
        "| Dataset | Arm | Train-mode CE (pre-update) | Train eval CE | Validation CE | Eval Train−Validation CE gap |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            rows = [lookup[(dataset, condition, seed)] for seed in SEEDS]
            lines.append(
                f"| {dataset} | {condition} | {_fmt([float(r['train_ce']) for r in rows], digits=4)} | "
                f"{_fmt([float(r['train_ce_eval']) for r in rows], digits=4)} | "
                f"{_fmt([float(r['val_ce']) for r in rows], digits=4)} | "
                f"{_fmt([float(r['train_val_ce_gap']) for r in rows], digits=4)} |"
            )
    lines += [
        "", "Train-mode CE is measured on the stochastic pre-optimizer-step training forward. Train eval CE and Validation CE are measured in eval/no-grad mode from the Best Accuracy checkpoint after the update; their gap is the comparable generalization diagnostic. Best F1 checkpoint Validation metrics are reported separately above.", "",
    ]
    lines += ["## Paired Validation differences", "", "Differences are left arm minus right arm; all per-seed values are retained in `data/paired_comparisons.csv`.", ""]
    for metric, heading, pct in (("best_accuracy.val_acc", "### Best Accuracy: Accuracy (percentage points)", True), ("best_accuracy.val_macro_f1", "### Best Accuracy: Macro-F1 (percentage points)", True), ("best_accuracy.val_ce", "### Best Accuracy: CE", False), ("best_f1.val_acc", "### Best F1 checkpoint: Accuracy (percentage points)", True), ("best_f1.val_macro_f1", "### Best F1 checkpoint: Macro-F1 (percentage points)", True), ("best_f1.val_ce", "### Best F1 checkpoint: CE", False)):
        lines += [heading, ""] + _paired_table(paired_summary, metric, percentage=pct) + [""]
    lines += [
        "## Training, resources, and evidence checks", "",
        "| Dataset | Arm | Best Acc epoch (mean ± SD) | Best F1 epoch (mean ± SD) | Train−Val Acc gap (pp) | Train−Val F1 gap (pp) | Eval Train−Val CE gap | Mean wall seconds | Peak allocated GPU GiB | Model / head parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            rows = [lookup[(dataset, condition, seed)] for seed in SEEDS]
            lines.append(
                f"| {dataset} | {condition} | {_fmt([float(r['best_accuracy_epoch']) for r in rows], digits=1)} | {_fmt([float(r['best_f1_epoch']) for r in rows], digits=1)} | "
                f"{_fmt([float(r['train_val_acc_gap']) for r in rows], pct=True)} | {_fmt([float(r['train_val_macro_f1_gap']) for r in rows], pct=True)} | "
                f"{_fmt([float(r['train_val_ce_gap']) for r in rows], digits=4)} | "
                f"{_fmt([float(r['wall_seconds']) for r in rows], digits=1)} | {max(float(r['peak_allocated_bytes'] or 0) for r in rows)/2**30:.2f} | "
                f"{rows[0]['model_parameters']} / {rows[0]['classifier_parameters']} |"
            )
    lines += ["", f"Preflight runtime: Python {manifest['environment']['python']}, PyTorch {manifest['environment']['pytorch']}, CUDA build {manifest['environment']['cuda_build']}, cuDNN {manifest['environment']['cudnn']}; device {', '.join(manifest['environment']['devices'])}. TF32 matmul={manifest['environment']['matmul_tf32']}, cuDNN TF32={manifest['environment']['cudnn_tf32']}, deterministic algorithms={manifest['environment']['deterministic_algorithms']}.", "", f"Schema audit passed for {audit['tables']['runs.csv']['rows']} runs and {audit['checkpoint_hash_records']} checkpoint hashes; {audit['failure_record_count']} retained execution failures are documented in `data/failures.json`. Test metrics absent.", ""]
    return "\n".join(lines)


def _mechanism_report(content, interventions, routers, messages) -> str:
    lines = [
        "# Content mechanism report", "",
        "These measurements are evaluated on each Best Accuracy checkpoint. Gate activity establishes that a pathway learned; frozen interventions show checkpoint sensitivity. Neither alone proves standalone causal utility. Pair input channels and active capacity differ across M1/M2/M3.", "",
        "## Gate and residual summary", "",
        "| Dataset | Arm | Mean channel SD | Saturated fraction | Query roll gate RMS | Normalized content RMS | Content/Cross cosine | Eta weighted RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in ("M1", "M2", "M3", "M4"):
            entry = content[dataset][condition]
            vals = [entry[k] for k in ("gate_channel_std_mean", "gate_saturation_fraction_mean", "query_roll_gate_rms_mean", "content_normalized_rms_mean", "content_cross_cosine_mean", "eta_contribution_rms_mean")]
            if any(value is None for value in vals):
                vals = [float("nan")] * 6
            lines.append(f"| {dataset} | {condition} | {vals[0]:.5g} | {vals[1]:.5g} | {vals[2]:.5g} | {vals[3]:.5g} | {vals[4]:.5g} | {vals[5]:.5g} |")
    lines += ["", "The CSV retains 32-channel means, standard deviations and correlation matrices per Dataset × Seed × Modality. M1 and M2 have no pair-query sensitivity by construction. M3/M4 use a fixed label-free receiver-query roll for both gate and frozen-logit sensitivity.", "", "## Physical, cross-modal, and semantic scale", "", "All entries are means over three seeds. Physical Profile, eta-weighted Content, and expert-output columns retain the four expert-wise RMS values, in expert order. `u1` is the physical one-hop reference; `Cross` is projected cross-modal context; `KNN` is the semantic-neighbor context. `*_hat` values are the normalized inputs actually used by the expert path.", "", "| Dataset | Arm | Modality | Physical u1 RMS | Cross raw / normalized RMS | KNN raw / normalized RMS | Four-order Physical Profile RMS by expert | Content normalized RMS | Eta-weighted Content RMS by expert | Expert output RMS by expert |", "|---|---|---|---:|---:|---:|---|---:|---|---|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for modality in ("text", "visual"):
                rows = [r for r in messages if r["dataset"] == dataset and r["condition"] == condition and r["modality"] == modality]
                if not rows:
                    continue
                def scalar(field: str) -> float:
                    return statistics.mean(float(row[field]) for row in rows)
                def vector(field: str) -> list[float]:
                    values = [json.loads(row[field]) for row in rows]
                    return [statistics.mean(float(item[index]) for item in values) for index in range(len(values[0]))]
                phys = ", ".join(f"{value:.4g}" for value in vector("physical_profile_rms_by_expert"))
                eta = ", ".join(f"{value:.4g}" for value in vector("eta_weighted_rms_by_expert"))
                output = ", ".join(f"{value:.4g}" for value in vector("expert_output_rms_by_expert"))
                lines.append(f"| {dataset} | {condition} | {modality} | {scalar('u1_rms'):.4g} | {scalar('cross_rms'):.4g} / {scalar('cross_hat_rms'):.4g} | {scalar('knn_rms'):.4g} / {scalar('knn_hat_rms'):.4g} | [{phys}] | {scalar('content_normalized_rms'):.4g} | [{eta}] | [{output}] |")
    lines += ["", "These scale measurements show whether semantic KNN and cross-modal context reach a similar or different magnitude from physical propagation and whether the learned content correction survives normalization and expert-specific `eta`. Scale alone does not establish that a component supplies useful label information. Modality-specific per-seed values remain in `message_scale_diagnostics.csv`.", "", "## Target-modality differences", "", "Positive Text−Visual values mean the Text-target pathway has the larger scale. These are descriptive averages over the three paired seeds.", "", "| Dataset | Arm | Normalized Content RMS Δ (Text−Visual) | Eta-weighted Content RMS Δ | Residual/Cross cosine Δ |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in ("M1", "M2", "M3", "M4"):
            values = {}
            for modality in ("text", "visual"):
                rows = [r for r in messages if r["dataset"] == dataset and r["condition"] == condition and r["modality"] == modality]
                if not rows:
                    values[modality] = (float("nan"), float("nan"), float("nan"))
                    continue
                content_scale = statistics.mean(float(r["content_normalized_rms"]) for r in rows)
                eta_scale = statistics.mean(statistics.mean(json.loads(r["eta_weighted_rms_by_expert"])) for r in rows)
                cosine = statistics.mean(float(r["content_cross_cosine"]) for r in rows)
                values[modality] = (content_scale, eta_scale, cosine)
            delta = [values["text"][idx] - values["visual"][idx] for idx in range(3)]
            lines.append(f"| {dataset} | {condition} | {delta[0]:+.5g} | {delta[1]:+.5g} | {delta[2]:+.5g} |")
    lines += ["", "Differences indicate that the two target directions operate at different trained scales; they do not establish that one modality is more useful. Per-seed modality-specific gate distributions, full 32-channel correlations, and content measurements remain available in the data tables.", "", "## Frozen interventions", "", "| Dataset | Arm | Intervention | Mean Δ Accuracy (pp) | Mean Δ Macro-F1 (pp) | Mean logit Δ RMS | Mean prediction flip fraction |", "|---|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        conds = sorted({r["condition"] for r in interventions if r["dataset"] == dataset})
        for condition in conds:
            names = sorted({r["intervention"] for r in interventions if r["dataset"] == dataset and r["condition"] == condition})
            for name in names:
                rows = [r for r in interventions if r["dataset"] == dataset and r["condition"] == condition and r["intervention"] == name]
                lines.append(f"| {dataset} | {condition} | {name} | {_fmt([float(r['val_acc_delta']) for r in rows], pct=True)} | {_fmt([float(r['val_macro_f1_delta']) for r in rows], pct=True)} | {_fmt([float(r['logit_delta_rms']) for r in rows], digits=5)} | {_fmt([float(r['prediction_flip_fraction']) for r in rows], pct=True)} |")
    lines += ["", "## Expert scale and router", "", "`message_scale_diagnostics.csv` and `expert_router_diagnostics.csv` preserve each modality and seed's one-hop physical RMS, Cross Base, KNN, IntraDelta, low-rank/residual/normalized content, eta contribution by expert, profile/output RMS, Cross cosine, routing entropy and expert share. A large residual is not sufficient evidence of useful class information; compare with paired metrics and interventions.", ""]
    return "\n".join(lines)


def _classwise_report(classes, degrees, transitions) -> str:
    lines = ["# Classwise and degree diagnostics", "", "Metrics are Validation-only and use the named checkpoint. No subgroup result was used to tune this frozen study.", "", "## ele-fashion: Classes 11 and 1", "", "| Checkpoint | Arm | Class | Recall mean ± sample SD (%) | Predicted count mean ± SD | Support mean |", "|---|---|---:|---:|---:|---:|"]
    for checkpoint in ("best_accuracy", "best_f1"):
        for condition in CONDITIONS:
            for cls in (11, 1):
                rows = [row for row in classes if row["dataset"] == "ele-fashion" and row["condition"] == condition and row["checkpoint"] == checkpoint and int(row["class_id"]) == cls]
                if not rows:
                    continue
                lines.append(f"| {checkpoint} | {condition} | {cls} | {_fmt([float(r['recall']) for r in rows], pct=True)} | {_fmt([float(r['predicted_count']) for r in rows], digits=1)} | {statistics.mean([float(r['support']) for r in rows]):.1f} |")
    lines += ["", "## Degree groups", "", "`degree_group_metrics.csv` reports Best Accuracy and Best F1 Validation Accuracy/Macro-F1/CE for degree 0, 1, 2–3, 4–7, and ≥8, along with full-graph normalized content RMS for each modality. Degree=1 can receive a nonzero content residual; any class prediction change there is still a trained-checkpoint comparison and may reflect shared-parameter changes.", "", "## Correctness transitions", "", "`prediction_transitions.csv` reports per-seed correct→wrong, wrong→correct, unchanged-correct and unchanged-wrong counts relative to M0 for each class and overall at Best Accuracy. These are paired Validation predictions, not Test performance.", ""]
    return "\n".join(lines)


def _paired_observation(lookup, dataset: str, left: str, right: str, field: str, *, percentage: bool = False) -> str:
    scale = 100.0 if percentage else 1.0
    values = [
        (float(lookup[(dataset, left, seed)][field]) - float(lookup[(dataset, right, seed)][field])) * scale
        for seed in SEEDS
    ]
    mean, sd = _mean_sd(values)
    unit = " pp" if percentage else ""
    seed_values = ", ".join(f"{value:+.3f}" for value in values)
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    return f"{mean:+.3f} ± {sd:.3f}{unit} (seeds 42/43/44: {seed_values}; Δ>0 {positive}/3, Δ<0 {negative}/3)"


def _mean_rows(rows: list[dict[str, str]], field: str, *, percentage: bool = False) -> tuple[float, float]:
    values = [float(row[field]) for row in rows if row.get(field, "") not in ("", "nan", "NaN")]
    if not values:
        return float("nan"), float("nan")
    mean, sd = _mean_sd(values)
    scale = 100.0 if percentage else 1.0
    return mean * scale, sd * scale


def _historical_v21_c0_means() -> dict[str, dict[str, float]]:
    """Read historical V2.1 C0 means as descriptive context, never as a gate."""
    path = ROOT / "research/rmcge_v21/PILOT_REPORT.md"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    best_acc = text.split("## Best Accuracy checkpoint", 1)[1].split("## Best Macro-F1 checkpoint", 1)[0]
    output: dict[str, dict[str, float]] = {}
    for heading, metric in (
        ("### Validation Accuracy (%)", "val_acc"),
        ("### Validation Macro-F1 (%)", "val_macro_f1"),
        ("### Validation CE", "val_ce"),
    ):
        start = best_acc.find(heading)
        if start < 0:
            continue
        content_start = start + len(heading)
        next_headings = [position for marker in ("\n### ", "\n## ") if (position := best_acc.find(marker, content_start)) >= 0]
        section = best_acc[content_start:min(next_headings)] if next_headings else best_acc[content_start:]
        for line in section.splitlines():
            match = re.match(r"\|\s*(Movies|Grocery|ele-fashion)\s*\|\s*([0-9.]+)\s*±", line)
            if match:
                output.setdefault(match.group(1), {})[metric] = float(match.group(2))
    return output


def _decision_memo(lookup, classes, degrees, messages, gates, routers, interventions, resources, failures) -> str:
    lines = ["# Decision memo", "", "This memo distinguishes (a) measured paired Validation facts, (b) frozen-checkpoint sensitivity, and (c) hypotheses this experiment cannot identify. Three seeds are descriptive; no significance claim is made. All deltas below are left arm minus right arm, with seed order 42/43/44.", ""]
    historical = _historical_v21_c0_means()
    lines += ["## 1. Does M0 reproduce the strong collaborative baseline?", ""]
    lines.append("The frozen implementation audit found M0/C0 shared tensors and classifier exact and the real-data initialization forward within the preregistered `5e-5` tolerance (largest observed difference `2.0266e-6`). This supports forward compatibility. The new M0 is the internal baseline for all study claims; historical V2.1 C0 is context only:")
    lines += ["", "| Dataset | New M0 Acc % | Historical V2.1 C0 Acc % | Δ pp | New M0 Macro-F1 % | Historical C0 Macro-F1 % | Δ pp | New M0 CE | Historical C0 CE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        new_acc, _ = _mean_rows([lookup[(dataset, "M0", seed)] for seed in SEEDS], "val_acc", percentage=True)
        new_f1, _ = _mean_rows([lookup[(dataset, "M0", seed)] for seed in SEEDS], "val_macro_f1", percentage=True)
        new_ce, _ = _mean_rows([lookup[(dataset, "M0", seed)] for seed in SEEDS], "val_ce")
        old = historical.get(dataset, {})
        old_acc, old_f1, old_ce = old.get("val_acc", float("nan")), old.get("val_macro_f1", float("nan")), old.get("val_ce", float("nan"))
        lines.append(f"| {dataset} | {new_acc:.3f} | {old_acc:.2f} | {new_acc-old_acc:+.3f} | {new_f1:.3f} | {old_f1:.2f} | {new_f1-old_f1:+.3f} | {new_ce:.4f} | {old_ce:.4f} |")
    lines += ["", "These historical comparisons are descriptive, not exact replay criteria. Differences can include stochastic training trajectory; they do not invalidate the audited V2.1 C0 mathematics or the internally paired comparisons.", ""]
    lines += ["## Best Accuracy versus Best F1 checkpoint", "", "The Best F1 checkpoint is selected independently and can trade Accuracy for Macro-F1. The following reports that selection gap for the two pair-content arms:", "", "| Dataset | Arm | Best F1 ckpt minus Best Accuracy: Accuracy pp | Macro-F1 pp | CE |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in ("M3", "M4"):
            rows = [lookup[(dataset, condition, seed)] for seed in SEEDS]
            acc = [float(r["best_f1_val_acc"]) - float(r["val_acc"]) for r in rows]
            f1 = [float(r["best_f1_val_macro_f1"]) - float(r["val_macro_f1"]) for r in rows]
            ce = [float(r["best_f1_val_ce"]) - float(r["val_ce"]) for r in rows]
            lines.append(f"| {dataset} | {condition} | {_fmt(acc, pct=True)} | {_fmt(f1, pct=True)} | {_fmt(ce, digits=4)} |")
    lines += ["", "These are checkpoint-selection differences within the same trained run, not separate condition effects. The full M0–M4 Best F1 paired contrasts are in `PILOT_REPORT.md` and `data/paired_comparisons.csv`.", ""]

    lines += ["## 2–3. Are sender-only or receiver-only transforms sufficient?", "", "Best-Accuracy checkpoint paired contrasts (percentage-point Acc / Macro-F1, then raw CE):", "", "| Dataset | M1−M0 Acc / F1 / CE | M2−M0 Acc / F1 / CE | M3−M2 Acc / F1 / CE |", "|---|---|---|---|"]
    for dataset in DATASETS:
        cells = []
        for left, right in (("M1", "M0"), ("M2", "M0"), ("M3", "M2")):
            acc = _paired_observation(lookup, dataset, left, right, "val_acc", percentage=True)
            f1 = _paired_observation(lookup, dataset, left, right, "val_macro_f1", percentage=True)
            ce = _paired_observation(lookup, dataset, left, right, "val_ce")
            # Keep each metric's per-seed vector explicit in the companion CSV and summarize here.
            cells.append(f"Acc {acc}; F1 {f1}; CE {ce}")
        lines.append(f"| {dataset} | " + "<br>".join(cells) + " |")
    lines += ["", "Observed pattern: M1's paired changes are small on Movies/Grocery and negative on both Accuracy and Macro-F1 for ele-fashion. M2 has modest mixed changes; receiver-only gating alone is not sufficient for a consistent cross-dataset gain. M3−M2 adds Macro-F1 on Movies and ele-fashion on average, but not Grocery. These are measured paired associations, not capacity-matched causal effects.", ""]

    lines += ["## 4–6. Pair behavior and M4 interaction", "", "| Dataset | Contrast | Acc Δ pp | Macro-F1 Δ pp | CE Δ |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for left, right in (("M3", "M0"), ("M3", "M1"), ("M3", "M2"), ("M4", "M3"), ("M4", "M0")):
            lines.append(f"| {dataset} | {left}−{right} | {_paired_observation(lookup,dataset,left,right,'val_acc',percentage=True)} | {_paired_observation(lookup,dataset,left,right,'val_macro_f1',percentage=True)} | {_paired_observation(lookup,dataset,left,right,'val_ce')} |")
    lines += ["", "Pair-path evidence must be read at three levels. Gate standard deviations/correlations and fixed receiver-query roll sensitivity indicate whether the pair-dependent pathway responds to its inputs; trained checkpoint metrics indicate utility; frozen interventions indicate sensitivity with all other trained parameters held fixed. The full channel statistics are in `content_gate_diagnostics.csv`.", "", "| Dataset | Arm | Gate channel SD | Saturated fraction | Receiver-query-roll Gate RMS | Query-roll Δ Acc / F1 pp; flip % |", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in ("M3", "M4"):
            gr = [r for r in gates if r["dataset"] == dataset and r["condition"] == condition]
            channel_sd = statistics.mean(statistics.mean(json.loads(r["channel_std"])) for r in gr) if gr else float("nan")
            saturation = statistics.mean(float(r["abs_ge_0_99_fraction"]) for r in gr) if gr else float("nan")
            roll = statistics.mean(float(r["receiver_query_roll_gate_rms"]) for r in gr) if gr else float("nan")
            ir = [r for r in interventions if r["dataset"] == dataset and r["condition"] == condition and r["intervention"] in {"receiver_query_roll", "receiver_query_permutation"}]
            if ir:
                da, _ = _mean_rows(ir, "val_acc_delta", percentage=True)
                df, _ = _mean_rows(ir, "val_macro_f1_delta", percentage=True)
                flips, _ = _mean_rows(ir, "prediction_flip_fraction", percentage=True)
                sensitivity = f"{da:+.3f} / {df:+.3f}; {flips:.2f}"
            else:
                sensitivity = "not recorded"
            lines.append(f"| {dataset} | {condition} | {channel_sd:.5g} | {saturation:.5g} | {roll:.5g} | {sensitivity} |")
    lines += ["", "Observed conclusion: M3's edge-conditioned Gate responds to receiver-query changes in every dataset, and frozen query rolls change Validation logits/predictions. M3 has favorable paired Macro-F1 deltas over M2 in Movies and ele-fashion but not Grocery; it is not a consistent cross-dataset improvement over M1/M2 or M0. M4−M3 is mixed on Movies/Grocery and negative on ele-fashion Accuracy and Macro-F1; this does not establish robust cooperation with Intra selection. A nonzero receiver-query roll effect shows frozen output sensitivity, not a retrained ablation or proof of class benefit. M3−M1 and M3−M2 are not strict capacity-matched effects because the gate input channels and active capacity differ. M4−M3 is a trained condition-bundle comparison; the M4 `intra_selection_off` intervention is complementary frozen sensitivity, not a factorial interaction estimate.", ""]

    intra = [r for r in interventions if r["condition"] == "M4" and r["intervention"] == "intra_selection_off"]
    lines += ["## 7. Do degree-1 nodes benefit?", ""]
    lines += ["| Dataset | Contrast | Degree-1 Acc Δ pp | Degree-1 Macro-F1 Δ pp | Mean Text/Visual Content RMS |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in ("M1", "M2", "M3", "M4"):
            base = {(r["seed"], r["checkpoint"]): r for r in degrees if r["dataset"] == dataset and r["condition"] == "M0" and r["degree_group"] == "1"}
            cand = [r for r in degrees if r["dataset"] == dataset and r["condition"] == condition and r["degree_group"] == "1" and r["checkpoint"] == "best_accuracy"]
            acc_delta, f1_delta = [], []
            for row in cand:
                base_row = base.get((row["seed"], row["checkpoint"]))
                if base_row and row["accuracy"] and row["macro_f1"] and base_row["accuracy"] and base_row["macro_f1"]:
                    acc_delta.append(float(row["accuracy"]) - float(base_row["accuracy"]))
                    f1_delta.append(float(row["macro_f1"]) - float(base_row["macro_f1"]))
            text_rms = [float(r["content_text_rms"]) for r in cand if r["content_text_rms"]]
            visual_rms = [float(r["content_visual_rms"]) for r in cand if r["content_visual_rms"]]
            acc_text = _fmt(acc_delta, pct=True) if acc_delta else "n/a"
            f1_text = _fmt(f1_delta, pct=True) if f1_delta else "n/a"
            content_text = f"{statistics.mean(text_rms):.4g}/{statistics.mean(visual_rms):.4g}" if text_rms and visual_rms else "n/a"
            lines.append(f"| {dataset} | {condition}−M0 | {acc_text} | {f1_text} | {content_text} |")
    lines += ["", f"Degree-group prediction changes remain effects of the trained model as a whole: shared parameters, multi-hop propagation, and direct one-hop Content can all contribute. The subgroup table does not isolate a direct causal effect. M4 `intra_selection_off` frozen Validation effect across datasets: Acc Δ {_fmt([float(r['val_acc_delta']) for r in intra], pct=True)} pp; Macro-F1 Δ {_fmt([float(r['val_macro_f1_delta']) for r in intra], pct=True)} pp.", ""]
    lines += ["Degree ≥2 subgroup deltas versus M0 are shown separately below; the bins are retained rather than combined because subgroup Macro-F1 is not additive:", "", "| Dataset | Arm | Degree 2–3 Acc/F1 Δ pp | Degree 4–7 Acc/F1 Δ pp | Degree ≥8 Acc/F1 Δ pp |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in ("M1", "M2", "M3", "M4"):
            values = []
            for group in ("2-3", "4-7", ">=8"):
                base_rows = {(r["seed"], r["checkpoint"]): r for r in degrees if r["dataset"] == dataset and r["condition"] == "M0" and r["degree_group"] == group and r["checkpoint"] == "best_accuracy"}
                arm_rows = [r for r in degrees if r["dataset"] == dataset and r["condition"] == condition and r["degree_group"] == group and r["checkpoint"] == "best_accuracy"]
                acc_delta, f1_delta = [], []
                for row in arm_rows:
                    base_row = base_rows.get((row["seed"], row["checkpoint"]))
                    if base_row and row["accuracy"] and row["macro_f1"] and base_row["accuracy"] and base_row["macro_f1"]:
                        acc_delta.append(float(row["accuracy"]) - float(base_row["accuracy"]))
                        f1_delta.append(float(row["macro_f1"]) - float(base_row["macro_f1"]))
                values.append(f"Acc {_fmt(acc_delta, pct=True)} / F1 {_fmt(f1_delta, pct=True)}" if acc_delta else "n/a")
            lines.append(f"| {dataset} | {condition}−M0 | " + " | ".join(values) + " |")
    lines.append("")

    lines += ["## 8. Fashion class 11 and class 1", "", "Best Accuracy and Best F1 checkpoint recall deltas versus M0 (percentage points; three-seed mean ± sample SD):", "", "| Checkpoint | Arm | Class 11 Recall Δ pp | Class 1 Recall Δ pp |", "|---|---|---:|---:|"]
    for checkpoint in ("best_accuracy", "best_f1"):
        for condition in ("M1", "M2", "M3", "M4"):
            cells = []
            for cls in (11, 1):
                cand = [r for r in classes if r["dataset"] == "ele-fashion" and r["condition"] == condition and r["checkpoint"] == checkpoint and int(r["class_id"]) == cls]
                base = {(r["seed"]): r for r in classes if r["dataset"] == "ele-fashion" and r["condition"] == "M0" and r["checkpoint"] == checkpoint and int(r["class_id"]) == cls}
                deltas = [float(r["recall"]) - float(base[r["seed"]]["recall"]) for r in cand if r["seed"] in base]
                cells.append(_fmt(deltas, pct=True) if deltas else "n/a")
            lines.append(f"| {checkpoint} | {condition} | {cells[0]} | {cells[1]} |")
    lines += ["", "Observed classwise pattern: at Best Accuracy, M2 retains a mean Class 11 recall loss while M3/M4 are closer to the internal M0 reference; Class 1 improves modestly for M2/M3/M4. At Best F1, all four arms have lower mean Class 11 and Class 1 recall than M0, with large seed spread for Class 11. Thus the prior Fashion minority-class concern is reduced for some Accuracy-selected arms, not resolved across checkpoint criteria. Full precision/recall/F1/support/predicted counts and confusion matrices are retained for each seed and both checkpoints; no Validation subgroup was used to retune the frozen model.", ""]

    lines += ["## 9–11. Expert contribution, Cross redundancy, and Router", "", "| Dataset | Arm | Content normalized RMS / Cross-hat RMS | Residual–Cross cosine | Eta-weighted Content RMS / Physical Profile RMS | Router entropy (max normalized expert share) | Content-off Δ Acc / F1 pp; flip % |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in ("M0", "M1", "M2", "M3", "M4"):
            mr = [r for r in messages if r["dataset"] == dataset and r["condition"] == condition]
            normalized = statistics.mean(float(r["content_normalized_rms"]) for r in mr) if mr else float("nan")
            cross = statistics.mean(float(r["cross_hat_rms"]) for r in mr) if mr else float("nan")
            cosine = statistics.mean(float(r["content_cross_cosine"]) for r in mr) if mr else float("nan")
            eta_vals = [v for r in mr for v in json.loads(r["eta_weighted_rms_by_expert"])]
            phys_vals = [v for r in mr for v in json.loads(r["physical_profile_rms_by_expert"])]
            eta = statistics.mean(eta_vals) if eta_vals else 0.0
            physical = statistics.mean(phys_vals) if phys_vals else float("nan")
            rr = [r for r in resources if r["dataset"] == dataset and r["condition"] == condition]
            router_rows = [r for r in routers if r["dataset"] == dataset and r["condition"] == condition]
            entropy = statistics.mean(float(r["router_entropy"]) for r in router_rows) if router_rows else float("nan")
            share_vectors = [json.loads(r["expert_selection_share"]) for r in router_rows]
            max_share = statistics.mean(max(float(v) for v in vec) for vec in share_vectors) if share_vectors else float("nan")
            off = [r for r in interventions if r["dataset"] == dataset and r["condition"] == condition and r["intervention"] == "content_off"]
            da, _ = _mean_rows(off, "val_acc_delta", percentage=True) if off else (0.0, 0.0)
            df, _ = _mean_rows(off, "val_macro_f1_delta", percentage=True) if off else (0.0, 0.0)
            flips, _ = _mean_rows(off, "prediction_flip_fraction", percentage=True) if off else (0.0, 0.0)
            lines.append(f"| {dataset} | {condition} | {normalized:.4g}/{cross:.4g} | {cosine:.4g} | {eta:.4g}/{physical:.4g} | {entropy:.4g}/{max_share:.3f} | {da:+.3f}/{df:+.3f}; {flips:.2f} |")
    lines += ["", "Observed interpretation: every active arm has a nonzero first-step final ContentGate gradient and a nonzero learned output-layer update; content-off interventions change logits/predictions after the shared expert/router stack, so the pathway is not completely erased downstream. Residual–Cross cosine is comparatively low in these summaries, which argues against simple vector duplication. Cross Base off generally produces larger logit RMS; metric sensitivity is material but varies by dataset and arm. This does not rule out functional overlap or establish the causal utility of either component. Router entropy/share describe allocation (notably low average entropy on Grocery); there is no direct evidence that the Router wholly overrides Content because frozen content-off comparisons remain sensitive, but those interventions do not retrain or isolate Router-mediated effects.", ""]

    lines += ["## 12. Compute, memory, and effective capacity", "", "| Dataset | Arm | Parameters | Mean trainable parameters | Mean wall seconds | Peak allocated GPU GiB |", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            rr = [r for r in resources if r["dataset"] == dataset and r["condition"] == condition]
            if not rr:
                continue
            params = int(float(rr[0]["parameter_count"]))
            trainable = statistics.mean(float(r["trainable_parameter_count"]) for r in rr)
            wall = statistics.mean(float(r["wall_seconds"]) for r in rr)
            peak = max(float(r["peak_allocated_bytes"]) for r in rr) / 2**30
            lines.append(f"| {dataset} | {condition} | {params:,} | {trainable:,.0f} | {wall:.1f} | {peak:.2f} |")
    lines += ["", "The Pair path uses rank-32 chunked edge computation, so no dense N×N adjacency or full E×256 edge tensor is formed. Runtime and activation memory still vary with graph size and sparse scatter. M0 carries the same declared but inactive module layout for matched initialization; equal parameter count is not equal active capacity. M1/M2/M3 gate inputs differ, so their paired comparison is not a strict capacity-matched causal ablation.", ""]

    lines += ["## 13–14. Research judgment and next discriminating experiment", "", "Low-rank message content modeling is supported as an implemented, trainable mechanism. A claim of general classification effectiveness requires consistent paired Validation gains across datasets/checkpoints and tolerable resource cost; the contrast tables above are the measured basis. Accuracy and Macro-F1 can trade off: inspect both checkpoint criteria, and do not describe an F1-only gain as an unqualified improvement. Dropout `.2` and Weight Decay `1e-4` were fixed, so neither main effect nor their interaction with Gate saturation, residual scale, or early stopping is identified.", "", "If the paired gains are small or seed-sensitive, the most discriminating next study would compare true receiver Query matching with a label-free permuted/mismatched Query under matched active parameterization, plus an explicit pair-content versus scalar-edge-weight control. That addresses whether alignment—not just added capacity or extra optimization pathways—accounts for any M3 advantage. This is a proposal only; no new condition is run here.", "", "Run-to-run trajectory variation is summarized by the three-seed SDs and retained per-seed deltas. The Fashion/M0 seed-42 analysis failure was corrected by aligning the analyzer Macro-F1 label set to Train+Validation; the untouched failed record/checkpoints and successful rerun are preserved. Small discrepancies among retries/three seeds remain stochastic trajectory evidence, not proof of a specific CUDA cause. Deterministic algorithms were not enabled for the formal environment because the inherited sparse propagation path may not support every deterministic kernel.", ""]

    lines += ["## Fixed optimization factors and failure provenance", "", "AdamW, learning rate `1e-3`, dropout `.2`, weight decay `1e-4`, epochs/early stopping, and remaining training settings are fixed across arms. No independent Dropout or Weight Decay effect or factorial interaction is estimated. Historical pilot scores are reference only; new conclusions use paired internal M0–M4 training.", "", "| Dataset / arm / seed | Failure stage | Recorded symptom | Resolution and evidence boundary |", "|---|---|---|---|"]
    for failure in failures.get("failures", []):
        dataset = failure.get("dataset", "-")
        condition = failure.get("condition", "-")
        seed = failure.get("seed", "-")
        stage = failure.get("stage", failure.get("classification", "-"))
        symptom = str(failure.get("error", "" )).replace("|", "/").splitlines()[0]
        resolution = str(failure.get("resolution", "Preserved raw failure record; no metric was rewritten." )).replace("|", "/").replace("\n", " ")
        lines.append(f"| {dataset}/{condition}/seed {seed} | {stage} | {symptom[:180]} | {resolution[:260]} |")
    lines += ["", "All raw tracebacks, prior artifacts, and successful rerun provenance remain in `data/failures.json`, the run manifest, and the ignored local output tree. The analyzer correction affects only post-training Validation metric reproduction; it does not alter model weights, optimizer updates, formal training, or Test-label access.", ""]
    return "\n".join(lines)


def main() -> None:
    audit = audit_and_analyze()
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
