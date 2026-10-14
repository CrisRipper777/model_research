#!/usr/bin/env python3
"""Summarize frozen CG-CGE V1 validation-only paired comparisons."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/cg_cge_v1/data"
PARENT = ROOT / "research/mag_view_cge_v1/data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
CONDITIONS = ("B1_complementary", "B2_conditioned_experts")
SEEDS = (42, 43, 44)
BASELINE = "A2_cross_source"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def paired_summary(differences: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in differences]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("paired differences must be a non-empty finite sequence")
    return {
        "n": len(values),
        "mean_difference": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "positive_seeds": sum(value > 0 for value in values),
        "negative_seeds": sum(value < 0 for value in values),
        "zero_seeds": sum(value == 0 for value in values),
    }


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {key}")
    return value


def _mean_sd(rows: list[dict[str, str]], key: str, *, percent: bool = False) -> str:
    values = [_float(row, key) for row in rows]
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    scale = 100.0 if percent else 1.0
    return f"{mean * scale:.3f} ± {sd * scale:.3f}"


def _paired_line(summary_rows: list[dict], dataset: str, condition: str, reference: str, metric: str) -> dict | None:
    return next((row for row in summary_rows if (row["dataset"], row["condition"], row["reference"], row["metric"]) == (dataset, condition, reference, metric)), None)


def _report_text(current: list[dict[str, str]], parent: list[dict[str, str]], summary: list[dict],
                 preflight: dict, failures: list[dict] | None,
                 diagnostics: list[dict[str, str]], histories: list[dict[str, str]],
                 preflight_attempts: list[dict] | None) -> str:
    lines = [
        "# CG-CGE V1 Pilot Report", "",
        "Validation-only report. Test evaluation was disabled; Test labels were masked before training.",
        "All B1/B2 results use the frozen new implementation and internal paired baseline. B0 is included only where the full protocol fingerprint gate passed.",
        "",
        "## Best-Accuracy checkpoint: Validation metrics", "",
        "| Dataset | Condition | Accuracy % | Macro-F1 % | CE | Train–Val Acc gap pp | Train–Val F1 gap pp | Train–Val CE gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    current_by = {(row["dataset"], row["condition"], int(row["seed"])): row for row in current}
    parent_by = {(row["dataset"], row["condition"], int(row["seed"])): row for row in parent}
    for dataset in DATASETS:
        for condition in (*CONDITIONS, BASELINE):
            rows = ([current_by[(dataset, condition, seed)] for seed in SEEDS]
                    if condition in CONDITIONS else [parent_by[(dataset, condition, seed)] for seed in SEEDS])
            if condition == BASELINE and not all(preflight["datasets"][dataset]["seeds"][str(seed)]["b0_match"]["eligible"] for seed in SEEDS):
                continue
            lines.append(
                f"| {dataset} | {condition} | {_mean_sd(rows, 'val_acc', percent=True)} | "
                f"{_mean_sd(rows, 'val_macro_f1', percent=True)} | {_mean_sd(rows, 'val_ce')} | "
                f"{_mean_sd(rows, 'train_val_acc_gap', percent=True)} | "
                f"{_mean_sd(rows, 'train_val_macro_f1_gap', percent=True)} | {_mean_sd(rows, 'train_val_ce_gap')} |"
            )
    lines.extend(["", "## Best-F1 checkpoint", "", "| Dataset | Condition | Accuracy % | Macro-F1 % | CE |", "|---|---|---:|---:|---:|"])
    for dataset in DATASETS:
        for condition in (*CONDITIONS, BASELINE):
            rows = ([current_by[(dataset, condition, seed)] for seed in SEEDS]
                    if condition in CONDITIONS else [parent_by[(dataset, condition, seed)] for seed in SEEDS])
            if condition == BASELINE and not all(preflight["datasets"][dataset]["seeds"][str(seed)]["b0_match"]["eligible"] for seed in SEEDS):
                continue
            lines.append(
                f"| {dataset} | {condition} | {_mean_sd(rows, 'best_f1_val_acc', percent=True)} | "
                f"{_mean_sd(rows, 'best_f1_val_macro_f1', percent=True)} | {_mean_sd(rows, 'best_f1_val_ce')} |"
            )

    lines.extend(["", "## Paired differences", "", "Differences are candidate minus reference; accuracy and Macro-F1 are percentage points, CE is raw. Mean/sample SD and positive/negative/zero seed counts are descriptive for n=3.", "",
                  "| Dataset | Contrast | Δ Accuracy pp (mean ± SD; +/−/0) | Δ Macro-F1 pp (mean ± SD; +/−/0) | Δ CE (mean ± SD; +/−/0) |",
                  "|---|---|---:|---:|---:|"])
    for dataset in DATASETS:
        contrasts = [("B1−B0", "B1_complementary", BASELINE), ("B2−B0", "B2_conditioned_experts", BASELINE), ("B2−B1", "B2_conditioned_experts", "B1_complementary")]
        for label, condition, reference in contrasts:
            if reference == BASELINE and not all(preflight["datasets"][dataset]["seeds"][str(seed)]["b0_match"]["eligible"] for seed in SEEDS):
                continue
            acc = _paired_line(summary, dataset, condition, reference, "val_acc")
            f1 = _paired_line(summary, dataset, condition, reference, "val_macro_f1")
            ce = _paired_line(summary, dataset, condition, reference, "val_ce")
            def fmt(row, scale=1.0):
                if row is None:
                    return "not paired"
                return (f"{float(row['mean_difference']) * scale:.3f} ± {float(row['sample_sd']) * scale:.3f}; "
                        f"{int(row['positive_seeds'])}/{int(row['negative_seeds'])}/{int(row['zero_seeds'])}")
            lines.append(f"| {dataset} | {label} | {fmt(acc, 100)} | {fmt(f1, 100)} | {fmt(ce)} |")

    lines.extend(["", "## Individual formal runs", "", "| Dataset | Condition | Seed | Val Acc % | Val Macro-F1 % | Val CE | Best Acc epoch | Best F1 epoch | Wall sec | Peak allocated GiB | Film scale/shift update norm |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in sorted(current, key=lambda r: (DATASETS.index(r["dataset"]), CONDITIONS.index(r["condition"]), int(r["seed"]))):
        allocated = "" if not row.get("peak_allocated_bytes") else f"{int(row['peak_allocated_bytes']) / (1024 ** 3):.2f}"
        film = f"{float(row['role_film_best_accuracy_scale_update_norm']):.3g}/{float(row['role_film_best_accuracy_shift_update_norm']):.3g}"
        lines.append(f"| {row['dataset']} | {row['condition']} | {row['seed']} | {100*_float(row,'val_acc'):.3f} | {100*_float(row,'val_macro_f1'):.3f} | {_float(row,'val_ce'):.4f} | {row['best_accuracy_epoch']} | {row['best_f1_epoch']} | {float(row['wall_seconds']):.1f} | {allocated} | {film} |")

    lines.extend(["", "## Optimization and mechanism diagnostics", "",
                  "- B1 and B2 use the same dropout (0.2), AdamW learning rate (1e-3), and weight decay (1e-4). This whole-model matrix does not separately identify dropout × weight-decay effects.",
                  "- Per-epoch main CE, unimodal CE, balance auxiliary loss, Train/Validation Accuracy/F1/CE, and timing are in `data/epoch_history.csv`.",
                  "- Mechanism activation does not establish causal utility. The three-seed paired comparisons are descriptive and do not prove population-level significance.",
                  "- Test labels and metrics are excluded.", ""])
    lines.extend(["### Auxiliary loss and checkpoint selection", "",
                  "| Dataset | Condition | First-epoch mean unimodal CE | Best-Accuracy-epoch mean unimodal CE | Best-F1 minus Best-Accuracy Δ Acc pp | Δ Macro-F1 pp | Δ CE |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    history_by_key = {(r["dataset"], r["condition"], int(r["seed"])): [] for r in histories}
    for entry in histories:
        history_by_key.setdefault((entry["dataset"], entry["condition"], int(entry["seed"])), []).append(entry)
    for dataset in DATASETS:
        for condition in CONDITIONS:
            run_rows = [current_by[(dataset, condition, seed)] for seed in SEEDS]
            first_uni, best_uni, acc_diff, f1_diff, ce_diff = [], [], [], [], []
            for run in run_rows:
                history = sorted(history_by_key[(dataset, condition, int(run["seed"]))], key=lambda row: int(row["epoch"]))
                first_uni.append(_float(history[0], "train_unimodal_ce"))
                epoch = int(run["best_accuracy_epoch"])
                selected = next(item for item in history if int(item["epoch"]) == epoch)
                best_uni.append(_float(selected, "train_unimodal_ce"))
                acc_diff.append(_float(run, "best_f1_val_acc") - _float(run, "val_acc"))
                f1_diff.append(_float(run, "best_f1_val_macro_f1") - _float(run, "val_macro_f1"))
                ce_diff.append(_float(run, "best_f1_val_ce") - _float(run, "val_ce"))
            def mean_sd(values: list[float], scale: float = 1.0) -> str:
                return f"{statistics.mean(values)*scale:.4f} ± {(statistics.stdev(values) if len(values)>1 else 0.0)*scale:.4f}"
            lines.append(f"| {dataset} | {condition} | {mean_sd(first_uni)} | {mean_sd(best_uni)} | {mean_sd(acc_diff,100)} | {mean_sd(f1_diff,100)} | {mean_sd(ce_diff)} |")

    lines.extend(["", "### Physical / Semantic / Complementary path diagnostics", "",
                  "Means across the three Best-Accuracy checkpoints; cross source weights include unsupported rows as exact zero.", "",
                  "| Dataset | Condition | Modality | Physical profile RMS | Same-source response RMS | P/S/None weight | Raw P/S RMS | Calibrated P/S RMS | Gate mean / saturated | Cross/U RMS | Attention entropy |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for modality in ("text", "visual"):
                rows = [row for row in diagnostics if row.get("dataset") == dataset and row.get("condition") == condition and row.get("checkpoint") == "best_accuracy" and row.get("modality") == modality]
                if len(rows) != 3:
                    continue
                cross = [json.loads(row["cross_completion"]) for row in rows]
                def cross_mean(key: str) -> float:
                    return statistics.mean(float(value[key]) for value in cross)
                def regular_mean(key: str) -> float:
                    return statistics.mean(float(row[key]) for row in rows)
                source = f"{cross_mean('physical_source_weight'):.3f}/{cross_mean('semantic_source_weight'):.3f}/{cross_mean('none_source_weight'):.3f}"
                raw = f"{cross_mean('physical_cross_raw_rms'):.4g}/{cross_mean('semantic_cross_raw_rms'):.4g}"
                calibrated = f"{cross_mean('physical_cross_calibrated_rms'):.4g}/{cross_mean('semantic_cross_calibrated_rms'):.4g}"
                gate = f"{cross_mean('channel_gate_mean'):.3f}/{cross_mean('channel_gate_saturation_fraction'):.3f}"
                lines.append(f"| {dataset} | {condition} | {modality} | {regular_mean('physical_profile_rms'):.4g} | {regular_mean('same_semantic_response_rms'):.4g} | {source} | {raw} | {calibrated} | {gate} | {cross_mean('cross_to_unimodal_rms'):.4g} | {cross_mean('semantic_attention_entropy_multi_neighbor'):.4g} |")

    lines.extend(["", "### Role-FiLM updates", "",
                  "B1 is an identity bypass with frozen FiLM parameters. B2 parameter change norms are measured from the zero initialization at the Best-Accuracy checkpoint.", "",
                  "| Dataset | B2 scale update norm mean ± SD | B2 shift update norm mean ± SD |", "|---|---:|---:|"])
    for dataset in DATASETS:
        rows = [current_by[(dataset, "B2_conditioned_experts", seed)] for seed in SEEDS]
        lines.append(f"| {dataset} | {_mean_sd(rows, 'role_film_best_accuracy_scale_update_norm')} | {_mean_sd(rows, 'role_film_best_accuracy_shift_update_norm')} |")
    failure_count = len((failures or {}).get("failures", [])) + len(preflight_attempts or [])
    lines.extend(["## Failures and unresolved issues", "",
                  f"Preserved formal/preflight failures or resource refusals: {failure_count}. See `data/failures.json` and `data/preflight_attempts.json`.",
                  "Remaining interpretation issues must be filled from observed evidence: CUDA sparse/index accumulation may be nondeterministic; three seeds cannot resolve all stochastic variance; physical and semantic candidate scale effects should be interpreted alongside the one-way cap and selector weights.", ""])
    lines.append("No Test evaluation was performed.")
    return "\n".join(lines)


def main() -> None:
    current = _read_csv(DATA / "runs.csv")
    parent = _read_csv(PARENT / "runs.csv")
    preflight = json.loads((DATA / "preflight_cpu.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "cpu_data_and_init_pass":
        raise RuntimeError("CPU data/init gate is missing")
    current_by_key = {(r["dataset"], r["condition"], int(r["seed"])): r for r in current}
    parent_by_key = {(r["dataset"], r["condition"], int(r["seed"])): r for r in parent}
    expected = {(d, c, s) for d in DATASETS for c in CONDITIONS for s in SEEDS}
    found = {key for key, row in current_by_key.items() if row.get("status") == "complete"}
    if found != expected:
        raise RuntimeError(f"formal run matrix incomplete: missing={sorted(expected-found)}, extra={sorted(found-expected)}")

    paired_rows: list[dict] = []
    summary_rows: list[dict] = []
    b0_eligible: dict[str, bool] = {}
    for dataset in DATASETS:
        eligible_values = [bool(preflight["datasets"][dataset]["seeds"][str(seed)]["b0_match"]["eligible"]) for seed in SEEDS]
        b0_eligible[dataset] = all(eligible_values)
        b0_rows = {seed: parent_by_key[(dataset, BASELINE, seed)] for seed in SEEDS}
        candidate_by_condition = {
            condition: {seed: current_by_key[(dataset, condition, seed)] for seed in SEEDS}
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            candidate_rows = candidate_by_condition[condition]
            if any(row["test_evaluation"] != "False" for row in candidate_rows.values()):
                raise RuntimeError("formal rows indicate Test evaluation was enabled")
            metrics = ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce")
            for metric in metrics:
                vals = [_float(candidate_rows[seed], metric) for seed in SEEDS]
                summary_rows.append({"dataset": dataset, "condition": condition, "reference": "internal", "metric": metric, **paired_summary(vals)})
                for seed in SEEDS:
                    paired_rows.append({"dataset": dataset, "condition": condition, "seed": seed,
                                        "reference": "within_condition", "metric": metric,
                                        "value": _float(candidate_rows[seed], metric), "difference": ""})
            if b0_eligible[dataset]:
                for metric in metrics:
                    differences = []
                    for seed in SEEDS:
                        candidate = _float(candidate_rows[seed], metric)
                        reference = _float(b0_rows[seed], metric)
                        difference = candidate - reference
                        differences.append(difference)
                        paired_rows.append({"dataset": dataset, "condition": condition,
                                            "seed": seed, "reference": BASELINE,
                                            "metric": metric, "value": candidate,
                                            "reference_value": reference, "difference": difference})
                    summary_rows.append({"dataset": dataset, "condition": condition,
                                         "reference": BASELINE, "metric": metric,
                                         **paired_summary(differences)})
        metrics = ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce")
        b1 = candidate_by_condition["B1_complementary"]
        b2 = candidate_by_condition["B2_conditioned_experts"]
        for metric in metrics:
            differences = []
            for seed in SEEDS:
                value, reference = _float(b2[seed], metric), _float(b1[seed], metric)
                differences.append(value - reference)
                paired_rows.append({"dataset": dataset, "condition": "B2_conditioned_experts",
                                    "seed": seed, "reference": "B1_complementary", "metric": metric,
                                    "value": value, "reference_value": reference,
                                    "difference": value - reference})
            summary_rows.append({"dataset": dataset, "condition": "B2_conditioned_experts",
                                 "reference": "B1_complementary", "metric": metric,
                                 **paired_summary(differences)})

    _write_csv(DATA / "paired_comparisons.csv", paired_rows)
    _write_csv(DATA / "summary.csv", summary_rows)
    failures_path = DATA / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {"failures": []}
    attempts_path = DATA / "preflight_attempts.json"
    preflight_attempts = json.loads(attempts_path.read_text()).get("attempts", []) if attempts_path.exists() else []
    (ROOT / "research/cg_cge_v1/PILOT_REPORT.md").write_text(
        _report_text(
            current, parent, summary_rows, preflight, failures,
            _read_csv(DATA / "mechanism_diagnostics.csv"),
            _read_csv(DATA / "epoch_history.csv"), preflight_attempts,
        ), encoding="utf-8"
    )
    result = {
        "status": "complete", "formal_runs": len(expected), "datasets": list(DATASETS),
        "conditions": list(CONDITIONS), "seeds": list(SEEDS), "baseline": BASELINE,
        "b0_strictly_paired": b0_eligible,
        "primary_rule": "validation-only paired descriptive means and sample standard deviations; no Test evaluation",
    }
    (DATA / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
