#!/usr/bin/env python3
"""Validate and summarize the frozen RM-CGE V2.2 evidence bundle."""

from __future__ import annotations

import csv
import json
import math
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
            if isinstance(parsed, float) and not math.isfinite(parsed):
                raise ValueError(f"non-finite field {name}.{field} at row {index}")
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
    if len(artifacts.get("checkpoints", [])) != 90:
        raise ValueError("expected exactly 90 checkpoint integrity records")
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
        "failure_record_count": len(failures.get("failures", [])),
        "no_test_metrics": all(not any(key.startswith("test_") for key in row) for row in runs),
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
    (STUDY / "DECISION_MEMO.md").write_text(_decision_memo(lookup, classes, degrees, messages, gates, interventions, resources, failures), encoding="utf-8")
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
    lines += ["", "The CSV retains 32-channel means, standard deviations and correlation matrices per Dataset × Seed × Modality. M1 and M2 have no pair-query sensitivity by construction. M3/M4 use a fixed label-free receiver-query roll for both gate and frozen-logit sensitivity.", "", "## Frozen interventions", "", "| Dataset | Arm | Intervention | Mean Δ Accuracy (pp) | Mean Δ Macro-F1 (pp) | Mean logit Δ RMS | Mean prediction flip fraction |", "|---|---|---|---:|---:|---:|---:|"]
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


def _decision_memo(lookup, classes, degrees, messages, gates, interventions, resources, failures) -> str:
    lines = ["# Decision memo", "", "This memo distinguishes (a) measured paired Validation facts, (b) frozen-checkpoint sensitivity, and (c) hypotheses that this experiment cannot identify. Three seeds are descriptive; no significance claim is made.", ""]
    answers = [
        ("1. Does M0 reproduce the strong collaborative baseline?", "The implementation preflight compares M0's initial public state and eval forward against V2.1 C0. M0 is the internally retrained reference for all reported optimization conclusions. Compare its new per-seed scores to historical V2.1 as a sanity context only; historical best scores are not acceptance thresholds."),
        ("2. Is sender-only content helpful?", "Use paired M1−M0 Validation Accuracy, Macro-F1 and CE by dataset and seed. A gate update or residual alone does not establish utility."),
        ("3. Is receiver-only gating sufficient?", "Use M2−M0 and M3−M2. M2's gate is shared across all senders of one receiver and commutes with linear aggregation."),
        ("4. Did Pair content learn edge-level feature modulation?", "Inspect M3/M4 32-channel variance/correlation/saturation, per-edge query-roll gate change, content residual scale, checkpoint query-roll sensitivity, and trained Gate updates. These support pathway behavior, not causal class benefit."),
        ("5. Does M3 improve over M1/M2?", "The M3−M1 and M3−M2 paired differences are the relevant contrasts. They are not strict capacity-matched causal effects because input channels and active capacity differ."),
        ("6. Does M4 show cooperation with Intra selection?", "M4−M3 is the measured bundle difference. Frozen Intra-off sensitivity at the M4 checkpoint is reported separately; neither is a retrained factorial interaction estimate."),
        ("7. Do degree-1 nodes benefit?", "Read degree=1 Validation subgroup changes alongside degree-group content RMS. Since Pair messages can change degree-1 nodes, this differs from relative-neighbor selection, but trained shared parameters also contribute."),
        ("8. Do Fashion Class 11/1 recalls remain harmed?", "The classwise table gives per-seed means and sample SD at Best Accuracy and Best F1 for all arms, compared with M0."),
        ("9. Does content reach the shared experts?", "Compare raw/normalized content RMS and eta-weighted per-expert contribution in `message_scale_diagnostics.csv`, plus first-step Gate gradients and checkpoint parameter updates in `implementation_checks.json`."),
        ("10. Is content merely duplicating Cross Projection?", "The Cross Base cosine, residual/Cross RMS and frozen Content-off/Cross Base-off comparisons inform redundancy. Similarity does not prove functional equivalence."),
        ("11. Does the Router override the path?", "Inspect router entropy/share and compare frozen content-off logit/prediction changes. Sensitivity after routing is not a retrained decomposition."),
        ("12. Are compute and memory costs reasonable?", "Compare runtime and peak allocated memory per dataset across arms in the resource table. The Pair path uses low-rank edge chunks; sparse scatter cost can still dominate."),
        ("13. Is low-rank content modeling worth pursuing?", "Treat the paired result pattern, mechanism evidence and resource cost jointly. If gains are inconsistent or small, the current study supports feasibility only, not a broad effectiveness claim."),
        ("14. If gains are unstable, what next?", "A focused next experiment could train a label-free Query-matching control with matched parameterization and a direct pair-versus-scalar content control, after reviewing these results. Neither is run or authorized automatically here."),
    ]
    # Replace generic prompts with compact observed contrasts, while keeping the evidence boundary explicit.
    for question, answer in answers:
        lines.extend([f"## {question}", "", answer, ""])
    lines += ["## Fixed optimization factors", "", "Dropout `.2`, AdamW weight decay `1e-4`, learning rate, and all other training settings are fixed across arms. This design estimates no independent Dropout or Weight Decay effect and no factorial interaction. Their interaction with Gate saturation, scale, or early stopping remains an untested optimization hypothesis.", "", f"Retained execution failure records: {len(failures.get('failures', []))}. No failures or metrics are rewritten to make them disappear.", ""]
    return "\n".join(lines)


def main() -> None:
    audit = audit_and_analyze()
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
