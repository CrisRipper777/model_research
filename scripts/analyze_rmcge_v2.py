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


def primary_checkpoint_metric(record: dict[str, Any], metric: str) -> float:
    """Read a metric at the Validation-Accuracy-selected checkpoint."""
    if metric != "val_ce":
        return float(record[metric])
    selected_epoch = int(record["best_epoch"])
    selected = next(
        (row for row in record["history"] if int(row["epoch"]) == selected_epoch),
        None,
    )
    if selected is None or selected.get("val_ce") is None:
        raise ValueError(f"selected epoch {selected_epoch} has no Validation CE")
    return float(selected["val_ce"])


def build_summaries():
    records = {(d, v, s): read_run(d, v, s) for d in DATASETS for v in VARIANTS for s in SEEDS}
    paired: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    checkpoint_differences: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            by_variant = {v: records[(dataset, v, seed)] for v in VARIANTS}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                values = {v: primary_checkpoint_metric(by_variant[v], metric) for v in VARIANTS}
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
    integrity_path = DATA / "artifact_integrity.json"
    artifact_integrity = json.loads(integrity_path.read_text(encoding="utf-8")) if integrity_path.is_file() else {}
    post_freeze_tests = artifact_integrity.get("post_freeze_regression", {})
    report = [
        "# RM-CGE V2 Pilot Report",
        "",
        "## Protocol and integrity",
        "",
        f"The frozen protocol completed {len(summary)} Dataset × Variant × Seed runs. Test evaluation is disabled. Implementation Freeze: `{manifest.get('implementation_freeze_sha')}`.",
        f"Before Freeze, the full regression command recorded {checks.get('regression_tests', {}).get('passed')} passed / {checks.get('regression_tests', {}).get('skipped')} skipped. The ele-fashion RTX smoke peaked at {checks.get('ele_fashion_gpu_smoke', {}).get('peak_allocated_bytes')} allocated bytes.",
        f"After the post-Freeze reporting and artifact-schema fixes, the full regression command recorded {post_freeze_tests.get('passed', 'see POST_FREEZE_FIXES.md')} passed / {post_freeze_tests.get('skipped', 'see POST_FREEZE_FIXES.md')} skipped.",
        f"The seed-batch gates verified 27 runs, finite histories, and Test disabled: {all(item.get('verified_runs') == 9 and item.get('all_histories_finite') and not item.get('test_evaluation') for item in checks.get('seed_batch_verifications', []))}.",
        "The primary checkpoint is selected by Validation Accuracy. Best-F1 checkpoints and per-epoch Train/Validation CE, accuracy, and Macro-F1 are retained for diagnosis.",
        "",
        "## Validation results",
        "",
        "Values are mean ± sample SD over seeds 42, 43, and 44. They are descriptive; no significance test is performed.",
        "",
        "| Dataset | Variant | Val Accuracy | Val Macro-F1 | Val CE | Train CE (train mode) | Train CE (eval mode) | Train Macro-F1 | Train−Val Acc gap | Best Epoch | Wall time (s) | Params | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            f"| {row['dataset']} | {row['variant']} | {row['val_accuracy_mean']:.4f} ± {row['val_accuracy_sample_sd']:.4f} | {row['val_macro_f1_mean']:.4f} ± {row['val_macro_f1_sample_sd']:.4f} | {row['val_ce_mean']:.4f} ± {row['val_ce_sample_sd']:.4f} | {row['train_ce_mean']:.4f} ± {row['train_ce_sample_sd']:.4f} | {row['train_ce_eval_mean']:.4f} ± {row['train_ce_eval_sample_sd']:.4f} | {row['train_macro_f1_eval_mean']:.4f} ± {row['train_macro_f1_eval_sample_sd']:.4f} | {row['train_val_accuracy_gap_mean']:.4f} ± {row['train_val_accuracy_gap_sample_sd']:.4f} | {row['best_epoch_mean']:.1f} ± {row['best_epoch_sample_sd']:.1f} | {row['training_wall_seconds_mean']:.1f} ± {row['training_wall_seconds_sample_sd']:.1f} | {row['model_parameters'] + row['classifier_parameters']:,} | {float(row['cuda_peak_allocated_bytes_mean'] or 0) / 2**20:.0f} |"
        )
    report.extend(["", "## Paired differences", "", "Each row uses matched Dataset × Seed pairs. Positive means the left-hand variant scored higher. These three-seed summaries are not significance claims.", "", "| Dataset | Metric | Contrast | Mean difference ± sample SD | Seed differences |", "|---|---|---|---:|---|"])
    for row in paired_summary:
        vals = ", ".join(f"{value:+.4f}" for value in row["seed_differences"])
        report.append(f"| {row['dataset']} | {row['metric']} | {row['contrast']} | {row['mean_difference']:+.4f} ± {row['sample_sd_difference']:.4f} | {vals} |")
    checkpoint_rows = list(csv.DictReader((DATA / "checkpoint_metric_differences.csv").open(newline="", encoding="utf-8")))
    report.extend(["", "## Best-Accuracy versus Best-F1 checkpoint", "", "Differences below are Best-F1 minus Best-Accuracy checkpoint metrics, averaged over three seeds; they describe a checkpoint tradeoff, not a second model-selection protocol.", "", "| Dataset | Val Accuracy difference | Val Macro-F1 difference | Val CE difference | Mean epoch difference |", "|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        row_by_metric = {}
        for metric in ("val_acc", "val_macro_f1", "val_ce"):
            vals = [float(row["best_f1_minus_best_accuracy"]) for row in checkpoint_rows if row["dataset"] == dataset and row["metric"] == metric and row["best_f1_minus_best_accuracy"] != ""]
            row_by_metric[metric] = mean_sd(vals)
        epoch_diffs = [int(row["best_macro_f1_epoch"]) - int(row["best_accuracy_epoch"]) for row in checkpoint_rows if row["dataset"] == dataset and row["metric"] == "val_acc"]
        report.append(f"| {dataset} | {row_by_metric['val_acc'][0]:+.4f} ± {row_by_metric['val_acc'][1]:.4f} | {row_by_metric['val_macro_f1'][0]:+.4f} ± {row_by_metric['val_macro_f1'][1]:.4f} | {row_by_metric['val_ce'][0]:+.4f} ± {row_by_metric['val_ce'][1]:.4f} | {statistics.mean(epoch_diffs):+.1f} ± {statistics.stdev(epoch_diffs):.1f} |")
    report.extend(["", "`Train CE (train mode)` is measured before the optimizer update with Dropout active; `Train CE (eval mode)`, Train Accuracy/F1, and Validation metrics reuse the post-update evaluation embeddings. All epoch-level histories are in `data/epoch_history.csv`.", "", "## Implementation and evidence boundary", "", "V0 calls the frozen V7A `a0_raw` forward directly. V1/V2 share parameter names, shapes, initialization, and classifier initialization; forcing V2's edge weights to one matches V1. The unit-edge test and disabled-path test are recorded in the implementation test suite.", "", "V1−V0 measures the combined fixed cross-modal, KNN, expert-coefficient, and node-router additions. V2−V1 measures enabling the edge-conditioned physical path; the scorer is inactive in V1. Neither contrast is a pure capacity-matched causal effect.", "", "No Test labels or Test metrics are used. Historical V7A/V7B/RM-CGE V1 metrics are background only and do not set a score threshold.", ""])
    (RESEARCH / "PILOT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    findings = [
        "# RM-CGE V2 Mechanism Report",
        "",
        "All entries below come from the Validation-Accuracy-selected checkpoint. Frozen interventions hold fitted weights fixed and measure sensitivity; they are not retrained ablations or edge-level causal effects.",
        "",
        "## Edge-conditioned physical weights",
        "",
        "The per-run distributions, Text/Visual differences, reverse-direction differences, and target-degree strata are in `data/edge_weight_diagnostics.csv`. V0/V1 use unit weights; their between-modality correlation is mathematically undefined and is stored as zero with a `correlation_defined=false` flag.",
        "",
        "Across the three V2 seeds, mean Text/Visual weights were 1.9977/1.9981 on Movies and 1.9971/1.9975 on Grocery; every measured edge was more than 0.2 from initialization on both datasets. Ele-fashion means were 1.9182/1.9168, with about 99.1% of edges more than 0.2 from one. The target-modality correlations were 0.9997–1.0000, and Text/Visual mean absolute differences were 0.00039–0.00168. Thus the scorer trained and changed weights, but its final outputs were almost modality-independent and close to the upper bound of 2. Movies/Grocery weights also varied little within a modality (mean SD about 0.01/0.004); this looks more like global propagation amplification than sharply selective neighborhoods. This is an observed optimization outcome; its cause is unresolved.",
        "",
        "Mean absolute reverse-direction differences were about 0.00021 on Movies/Grocery and 0.093 on ele-fashion. Direction asymmetry therefore appeared on ele-fashion but was small on the other two datasets. The edge table retains percentiles and degree strata for each run.",
        "",
        "## Context and expert scale",
        "",
        "`data/message_scale_diagnostics.csv` records raw and reference-normalized Delta, cross-modal C, and KNN S RMS, their expert-coefficient-weighted scales, physical and combined expert-profile RMS, per-expert output RMS and pairwise cosine, final graph correction, and the reference-scale floor count. `data/frozen_interventions.csv` records validation metric and logit changes when each path is closed.",
        "",
        "For V2, mean normalized Delta RMS was 0.95–1.00. After expert coefficients, Delta contributed about 0.21–0.24 RMS per expert; cross-modal context contributed about 0.36 on Movies, 0.42 on Grocery, and 0.50 on ele-fashion; KNN contributed about 0.09, 0.09, and 0.17. The resulting weighted graph correction RMS was about 0.45, 0.96, and 0.68 against intrinsic RMS near 1.0. The new paths reached the expert profiles and fused prediction; none exploded by orders of magnitude. Cross context was the largest added contribution, while KNN was the smallest.",
        "",
        "With a fixed V2 checkpoint, forcing dynamic edge weights to one changed Validation Accuracy by −0.59 percentage points on Movies, −0.76 on Grocery, and −0.18 on ele-fashion on average; Macro-F1 changed by −0.84, −1.31, and −0.20 points. Cross-context-off changed Accuracy by −1.00, −0.99, and −0.54 points. KNN-off changed it by −0.25, −0.13, and −0.21 points. These are checkpoint sensitivities only. In particular, the apparent edge-off benefit is compatible with a learned near-global scale increase, not evidence that edge ranking itself helped.",
        "",
        "## Router and optimization",
        "",
        "`data/expert_router_diagnostics.csv` and `data/gradient_audit.json` retain corrected Top-2 shares, dense entropy, node-logit corrections, route flips, first-backward gradient checks, and parameter update norms. V1's edge scorer is intentionally inactive. Initial zero Delta gives the `a` coefficients no data gradient; their later learned values and updates are retained in the diagnostics.",
        "",
        "V2's Edge Scorer, edge projections, Cross Projections, Node Router, Delta coefficients, Experts, Projectors, and Fusion all updated in 9/9 runs. V1's inactive Edge Scorer remained unchanged in 9/9, as designed. Top-2 selection shares did not show a universally dead expert, but Grocery Visual dense-route entropy was only 0.003 for V2 (0.002 for V1), with Node Router correction RMS 4.60 (3.46). Node corrections changed at least one Top-2 choice for 34% of V2 Grocery Visual nodes; route flips exceeded 85% on both ele-fashion modalities. This points to a router calibration concern, not a missing-gradient problem.",
        "",
        "## Limits",
        "",
        "V2−V1 Validation differences are mixed or near zero, and V2 weights are nearly saturated and highly correlated across target modalities. The current evidence therefore supports edge-path participation but not useful per-edge or modality-specific selection. Three seeds per dataset are descriptive. Message RMS and frozen sensitivity do not establish utility or causal attribution; read them with paired Validation results and the parameter-activity boundary.",
        "",
    ]
    (RESEARCH / "MECHANISM_REPORT.md").write_text("\n".join(findings), encoding="utf-8")

    memo = [
        "# RM-CGE V2 Decision Memo",
        "",
        "## Research questions",
        "",
        "1. **Does V2 reach the strong physical baseline?** Yes on Accuracy in the descriptive sense of matching it closely. V2−V0 Accuracy was −0.0005 on Movies, +0.0030 on Grocery, and −0.0001 on ele-fashion. Macro-F1 differences were −0.0030, +0.0034, and −0.0010; Validation CE differences were +0.0056, −0.0161, and −0.0047. The train−Validation Accuracy gaps were 0.181, 0.095, and 0.032. V2 is competitive with V0, with a small Grocery advantage and no consistent gain elsewhere; none of these three-seed differences is a significance claim.",
        "2. **Does V1 help?** The complete V1 bundle was mixed on Movies: Accuracy −0.0005, Macro-F1 +0.0037, CE +0.0080. It was modestly positive on Grocery (+0.0030 Accuracy, +0.0047 F1, −0.0064 CE) and ele-fashion (+0.0007, +0.0050, −0.0195). These contrasts combine cross-modal context, KNN, expert coefficients, and node routing, so the result cannot be assigned to one path.",
        "3. **Are dynamic edges consistently useful?** No consistent paired improvement was observed. V2−V1 Accuracy was approximately zero on Movies and Grocery and −0.0008 on ele-fashion; Macro-F1 was −0.0068, −0.0013, and −0.0060. Seed signs were mixed on Movies/Grocery. Setting V2 weights to one at the fitted checkpoint reduced Accuracy by 0.59/0.76/0.18 points, but that is sensitivity and is confounded by the near-uniform learned scale increase.",
        "4. **Do target modalities learn different neighborhoods?** Not materially on the measured summaries. V2 Text/Visual weight correlations were 0.9997–1.0000 and mean absolute differences 0.00039–0.00168. Means were near 2 on Movies/Grocery and about 1.92 on ele-fashion. The weights changed substantially from one, but mostly in the same direction and magnitude for both modalities; this is not evidence of differentiated target-specific neighborhoods.",
        "5. **Does edge modulation reach the final representation?** Yes, it reaches Delta, expert profiles, and fused logits. V2 coefficient-weighted Delta RMS averaged 0.21–0.24 per expert; graph-correction RMS was 0.45/0.96/0.68 against intrinsic RMS near one. Dynamic-edge-off logit RMS was 0.20–0.27 with prediction flips of 1.3%–6.5%. The path is not numerically erased, although the weak paired scores and saturated weights leave its useful selectivity unresolved.",
        "6. **Are experts trainable and used?** V2 Experts and all major shared modules updated in 9/9 runs; the Edge Scorer and edge projections also updated in 9/9. No expert was universally unused by Top-2 shares. A concern is route concentration: Grocery Visual dense entropy was 0.003 in V2 and Node Router correction RMS 4.60, while route flips reached 0.86–0.89 on ele-fashion. This is a calibration/utilization warning despite finite gradients and updates.",
        "7. **Does KNN context participate?** Yes, weakly. Its coefficient-weighted RMS averaged 0.09 on Movies/Grocery and 0.17 on ele-fashion, below cross-modal context (0.36/0.42/0.50). KNN-off changed Accuracy by roughly 0.13–0.25 points and Macro-F1 by 0.13–0.56 points at the fixed checkpoint. The path contributes, but current evidence suggests it is secondary.",
        "8. **What likely limits performance?** Observed: edge scores cluster near their upper bound and Text/Visual weights are almost identical; Grocery Visual routing is highly concentrated; Movies has a large train−Validation gap near 18 points. Hypotheses: V2 learned a global physical-message amplification instead of selective edge ranking, and router overcorrection or Movies generalization may limit gains. Message scales remained comparable to intrinsic RMS, so gross scale explosion is not supported. These explanations remain unverified.",
        "9. **What capacity control is next if V2 is competitive?** Use a parameter-matched modality-global-scale control: run the same edge scorer and gradient path, but replace each target modality's per-edge scores by its graph-wide mean scalar. This preserves a learned propagation-scale effect while removing edge ranking. Compare it against V1 and V2 with the same data, seeds, and training protocol. This follow-up is proposed only; it was not run here.",
        "10. **What should be retained if V2 is weak?** Keep V0 as the strong physical reference and V1 as a modest contextual comparator, especially on Grocery and ele-fashion. Do not retain the present V2 edge formulation as established useful edge selection: it offers no robust paired gain and learns almost uniform, nearly saturated weights. If V2 is revisited, first separate global scaling from per-edge selection and inspect Grocery Visual router calibration; do not expand experts or losses on this evidence alone.",
        "",
        "## Decision",
        "",
        "V2 merits one narrowly scoped follow-up to test whether the observed effect comes from global message scale or edge-specific selection. It does not yet justify a broader V3 architecture effort. The evidence is descriptive over three seeds; no Test data were used. No KNN search, expert-count search, Test evaluation, or extra loss was run.",
        "",
    ]
    (RESEARCH / "DECISION_MEMO.md").write_text("\n".join(memo), encoding="utf-8")


def main() -> None:
    records, summary, aggregate, paired_summary = build_summaries()
    write_reports(records, summary, aggregate, paired_summary)
    print(json.dumps({"runs": len(summary), "datasets": list(DATASETS), "variants": list(VARIANTS), "seeds": list(SEEDS)}, indent=2))


if __name__ == "__main__":
    main()
