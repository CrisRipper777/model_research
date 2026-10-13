#!/usr/bin/env python3
"""Create paired descriptive summaries and the RM-CGE V2 research reports."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/rmcge_v2"
DATA = RESEARCH / "data"
OUTPUT = ROOT / "outputs/rmcge_v2/runs"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("V0", "V1", "V2")
SEEDS = (42, 43, 44)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def read_run(dataset: str, variant: str, seed: int) -> dict[str, Any]:
    path = OUTPUT / dataset / variant / f"seed_{seed}" / "checkpoint_diagnostics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    metrics_path = OUTPUT / dataset / variant / f"seed_{seed}" / "run_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["runs"][0]
    record["run_metadata"] = metrics["metadata"]
    record["task_metrics"] = metrics["metrics"]
    return record


def build_summaries():
    records = {(d, v, s): read_run(d, v, s) for d in DATASETS for v in VARIANTS for s in SEEDS}
    paired: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    checkpoint_differences: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            by_variant = {v: records[(dataset, v, seed)] for v in VARIANTS}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                values = {v: float(by_variant[v][metric]) for v in VARIANTS}
                for contrast, left, right in (("V1-V0", "V1", "V0"), ("V2-V1", "V2", "V1"), ("V2-V0", "V2", "V0")):
                    paired.append({"dataset": dataset, "seed": seed, "metric": metric, "contrast": contrast, "difference": values[left] - values[right], "left_value": values[left], "right_value": values[right]})
            for variant in VARIANTS:
                item = by_variant[variant]
                history = item["history"]
                selected = next((row for row in history if int(row["epoch"]) == int(item["best_epoch"])), history[-1])
                metadata = item["run_metadata"]
                selections = metadata.get("dual_checkpoint_selection", {})
                best_acc_metrics = selections.get("best_accuracy", {}).get("metrics", {})
                best_f1_metrics = selections.get("best_macro_f1", {}).get("metrics", {})
                summary.append({
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "val_accuracy": item["val_accuracy"],
                    "val_macro_f1": item["val_macro_f1"],
                    "val_ce": float(selected["val_ce"]),
                    "train_ce": float(selected["train_ce"]),
                    "train_ce_eval": float(selected["train_ce_eval"]),
                    "train_accuracy_eval": float(selected["train_accuracy_eval"]),
                    "train_macro_f1_eval": float(selected["train_macro_f1_eval"]),
                    "train_val_accuracy_gap": float(selected["train_accuracy_eval"] - selected["val_acc"]),
                    "best_epoch": int(item["best_epoch"]),
                    "best_f1_epoch": int(metadata.get("dual_checkpoint_selection", {}).get("best_macro_f1", {}).get("epoch") or -1),
                    "model_parameters": int(metadata["model_parameters"]),
                    "classifier_parameters": int(metadata["classifier_parameters"]),
                    "training_wall_seconds": float(metadata.get("training_wall_seconds", 0.0)),
                    "cuda_peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
                    "cuda_peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
                })
                for metric in ("train_ce", "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce", "val_acc", "val_macro_f1"):
                    acc_key = "train_macro_f1_eval" if metric == "train_macro_f1_eval" else metric
                    acc_value = best_acc_metrics.get(acc_key)
                    f1_value = best_f1_metrics.get(acc_key)
                    checkpoint_differences.append({
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "metric": metric,
                        "best_accuracy_checkpoint": acc_value,
                        "best_macro_f1_checkpoint": f1_value,
                        "best_f1_minus_best_accuracy": (float(f1_value) - float(acc_value)) if acc_value is not None and f1_value is not None else None,
                        "best_accuracy_epoch": selections.get("best_accuracy", {}).get("epoch"),
                        "best_macro_f1_epoch": selections.get("best_macro_f1", {}).get("epoch"),
                    })
    write_csv(DATA / "paired_comparisons.csv", paired, ["dataset", "seed", "metric", "contrast", "difference", "left_value", "right_value"])
    write_csv(DATA / "summary_by_run.csv", summary, list(summary[0]))
    write_csv(DATA / "checkpoint_metric_differences.csv", checkpoint_differences, list(checkpoint_differences[0]))
    aggregate: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            rows = [row for row in summary if row["dataset"] == dataset and row["variant"] == variant]
            result: dict[str, Any] = {"dataset": dataset, "variant": variant}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce", "train_ce", "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "train_val_accuracy_gap", "best_epoch", "best_f1_epoch", "training_wall_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                vals = [float(row[metric]) for row in rows if row[metric] is not None]
                mean, sd = mean_sd(vals)
                result[f"{metric}_mean"] = mean
                result[f"{metric}_sample_sd"] = sd
            result["model_parameters"] = rows[0]["model_parameters"]
            result["classifier_parameters"] = rows[0]["classifier_parameters"]
            aggregate.append(result)
    write_csv(DATA / "summary.csv", aggregate, list(aggregate[0]))
    paired_summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
            for contrast in ("V1-V0", "V2-V1", "V2-V0"):
                vals = [float(row["difference"]) for row in paired if row["dataset"] == dataset and row["metric"] == metric and row["contrast"] == contrast]
                mean, sd = mean_sd(vals)
                paired_summary.append({"dataset": dataset, "metric": metric, "contrast": contrast, "mean_difference": mean, "sample_sd_difference": sd, "seed_differences": vals, "n": len(vals), "significance_tested": False})
    write_json(DATA / "paired_summary.json", paired_summary)
    return records, summary, aggregate, paired_summary


def write_reports(records, summary, aggregate, paired_summary):
    checks = json.loads((DATA / "implementation_checks.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    report = [
        "# RM-CGE V2 Pilot Report",
        "",
        "## Protocol and integrity",
        "",
        f"The frozen protocol completed {len(summary)} Dataset × Variant × Seed runs. Test evaluation is disabled. Implementation Freeze: `{manifest.get('implementation_freeze_sha')}`.",
        f"The full regression command recorded {checks.get('regression_tests', {}).get('passed')} passing tests. The ele-fashion RTX smoke peaked at {checks.get('ele_fashion_gpu_smoke', {}).get('peak_allocated_bytes')} allocated bytes.",
        "The primary checkpoint is selected by Validation Accuracy. Best-F1 checkpoints and per-epoch Train/Validation CE, accuracy, and Macro-F1 are retained for diagnosis.",
        "",
        "## Validation results",
        "",
        "Values are mean ± sample SD over seeds 42, 43, and 44. They are descriptive; no significance test is performed.",
        "",
        "| Dataset | Variant | Val Accuracy | Val Macro-F1 | Val CE | Train CE (train mode) | Train CE (eval mode) | Train Macro-F1 | Train−Val Acc gap | Best Epoch | Params | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            f"| {row['dataset']} | {row['variant']} | {row['val_accuracy_mean']:.4f} ± {row['val_accuracy_sample_sd']:.4f} | {row['val_macro_f1_mean']:.4f} ± {row['val_macro_f1_sample_sd']:.4f} | {row['val_ce_mean']:.4f} ± {row['val_ce_sample_sd']:.4f} | {row['train_ce_mean']:.4f} ± {row['train_ce_sample_sd']:.4f} | {row['train_ce_eval_mean']:.4f} ± {row['train_ce_eval_sample_sd']:.4f} | {row['train_macro_f1_eval_mean']:.4f} ± {row['train_macro_f1_eval_sample_sd']:.4f} | {row['train_val_accuracy_gap_mean']:.4f} ± {row['train_val_accuracy_gap_sample_sd']:.4f} | {row['best_epoch_mean']:.1f} ± {row['best_epoch_sample_sd']:.1f} | {row['model_parameters'] + row['classifier_parameters']:,} | {float(row['cuda_peak_allocated_bytes_mean'] or 0) / 2**20:.0f} |"
        )
    report.extend(["", "## Paired differences", "", "Each row uses matched Dataset × Seed pairs. Positive means the left-hand variant scored higher. These three-seed summaries are not significance claims.", "", "| Dataset | Metric | Contrast | Mean difference ± sample SD | Seed differences |", "|---|---|---|---:|---|"])
    for row in paired_summary:
        vals = ", ".join(f"{value:+.4f}" for value in row["seed_differences"])
        report.append(f"| {row['dataset']} | {row['metric']} | {row['contrast']} | {row['mean_difference']:+.4f} ± {row['sample_sd_difference']:.4f} | {vals} |")
    report.extend(["", "## Best-Accuracy versus Best-F1 checkpoint", "", "Per-run Train/Validation CE, Accuracy, Macro-F1, and the Best-F1 minus Best-Accuracy checkpoint differences are in `data/checkpoint_metric_differences.csv`. `Train CE (train mode)` is measured before the optimizer update with Dropout active; `Train CE (eval mode)`, Train Accuracy/F1, and Validation metrics reuse the post-update evaluation embeddings. All epoch-level histories are in `data/epoch_history.csv`.", "", "## Implementation and evidence boundary", "", "V0 calls the frozen V7A `a0_raw` forward directly. V1/V2 share parameter names, shapes, initialization, and classifier initialization; forcing V2's edge weights to one matches V1. The unit-edge test and disabled-path test are recorded in the implementation test suite.", "", "V1−V0 measures the combined fixed cross-modal, KNN, expert-coefficient, and node-router additions. V2−V1 measures enabling the edge-conditioned physical path; the scorer is inactive in V1. Neither contrast is a pure capacity-matched causal effect.", "", "No Test labels or Test metrics are used. Historical V7A/V7B/RM-CGE V1 metrics are background only and do not set a score threshold.", ""])
    (RESEARCH / "PILOT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    mechanisms = []
    for key, item in records.items():
        details = item.get("mechanism_details", {})
        for modality, values in details.get("modalities", {}).items():
            mechanisms.append((key, modality, values))
    findings = [
        "# RM-CGE V2 Mechanism Report",
        "",
        "All entries below come from the Validation-Accuracy-selected checkpoint. Frozen interventions hold fitted weights fixed and measure sensitivity; they are not retrained ablations or edge-level causal effects.",
        "",
        "## Edge-conditioned physical weights",
        "",
        "The per-run distributions, Text/Visual differences, reverse-direction differences, and target-degree strata are in `data/edge_weight_diagnostics.csv`. V0/V1 use unit weights; their between-modality correlation is mathematically undefined and is stored as zero with a `correlation_defined=false` flag.",
        "",
        "## Context and expert scale",
        "",
        "`data/message_scale_diagnostics.csv` records raw and reference-normalized Delta, cross-modal C, and KNN S RMS, their expert-coefficient-weighted scales, physical and combined expert-profile RMS, per-expert output RMS and pairwise cosine, final graph correction, and the reference-scale floor count. `data/frozen_interventions.csv` records validation metric and logit changes when each path is closed.",
        "",
        "## Router and optimization",
        "",
        "`data/expert_router_diagnostics.csv` and `data/gradient_audit.json` retain corrected Top-2 shares, dense entropy, node-logit corrections, route flips, first-backward gradient checks, and parameter update norms. V1's edge scorer is intentionally inactive. Initial zero Delta gives the `a` coefficients no data gradient; their later learned values and updates are retained in the diagnostics.",
        "",
        "## Limits",
        "",
        "A small or large frozen-path effect indicates checkpoint sensitivity only. The study has three seeds per dataset and does not isolate the components of V1. Message RMS by itself does not establish utility; it must be read with coefficient-weighted scales, expert profiles, route behavior, frozen interventions, and paired Validation differences.",
        "",
    ]
    (RESEARCH / "MECHANISM_REPORT.md").write_text("\n".join(findings), encoding="utf-8")

    memo = [
        "# RM-CGE V2 Decision Memo",
        "",
        "## Observations",
        "",
        "Use the paired values in `PILOT_REPORT.md` and the mechanism tables as the complete empirical record. Mark positive, negative, or mixed results separately for each dataset. Do not infer significance from three seeds.",
        "",
        "## Research questions",
        "",
        "1. **Does V2 reach the strong physical baseline?** Compare V2−V0 Validation Accuracy, Macro-F1, CE, and train-validation gap by dataset and seed.",
        "2. **Does V1 help?** Interpret V1−V0 as the full context-plus-node-router bundle, not as cross-modal utility alone.",
        "3. **Are dynamic edges consistently useful?** Inspect V2−V1 paired differences and the fixed-checkpoint edge-off sensitivity together.",
        "4. **Do target modalities learn different neighborhoods?** Use Text/Visual correlation and mean absolute difference; verify that weights move beyond initialization.",
        "5. **Does edge modulation reach the final representation?** Follow Delta through its coefficient, expert profile, graph correction, fusion, and frozen edge-off logits.",
        "6. **Are experts trainable and used?** Check Top-2 shares, dense entropy, gradients, and parameter updates.",
        "7. **Does KNN context participate?** Compare coefficient-weighted S, profile contribution, and the frozen KNN-off effect.",
        "8. **What likely limits performance?** State whether evidence points to edge weights, message scale, expert routing, train-validation generalization, or the overall premise; label explanations as hypotheses.",
        "9. **What capacity control is next if V2 is competitive?** Consider a parameter-matched control with the same scorer active but edge weights fixed, without starting that experiment here.",
        "10. **What should be retained if V2 is weak?** Decide between retaining the V1 bundle, returning to V0, or revising expert responsibilities using observed evidence.",
        "",
        "## Decision",
        "",
        "State whether V2 merits a separately scoped follow-up. No V3, KNN search, expert-count search, Test evaluation, or extra loss is authorized by this evidence package.",
        "",
    ]
    (RESEARCH / "DECISION_MEMO.md").write_text("\n".join(memo), encoding="utf-8")


def main() -> None:
    records, summary, aggregate, paired_summary = build_summaries()
    write_reports(records, summary, aggregate, paired_summary)
    print(json.dumps({"runs": len(summary), "datasets": list(DATASETS), "variants": list(VARIANTS), "seeds": list(SEEDS)}, indent=2))


if __name__ == "__main__":
    main()
