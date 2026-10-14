#!/usr/bin/env python3
"""Validate, summarize, and report the frozen View-CGE V1 campaign."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/mag_view_cge_v1"
DATA = STUDY / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
CONDITIONS = ("A0_legacy", "A1_separate_intra", "A2_cross_source")
SEEDS = (42, 43, 44)
RUN_KEYS = {(d, c, s) for d in DATASETS for c in CONDITIONS for s in SEEDS}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def validate_runs(rows: list[dict[str, str]], *, require_complete: bool = True, manifest: dict[str, Any] | None = None) -> dict[tuple[str, str, int], dict[str, str]]:
    lookup: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["condition"], int(row["seed"]))
        if key not in RUN_KEYS:
            raise ValueError(f"unexpected run key {key}")
        if key in lookup:
            raise ValueError(f"duplicate run key {key}")
        if row.get("status") != "complete":
            raise ValueError(f"run is not complete: {key} ({row.get('status')})")
        if row.get("test_evaluation", "false").lower() != "false":
            raise ValueError(f"Test evaluation was enabled in {key}")
        if any(name.startswith("test_") and name != "test_evaluation" for name in row):
            raise ValueError(f"Test metric field leaked into run row {key}")
        for field in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_macro_f1", "train_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "best_f1_train_acc", "best_f1_train_macro_f1", "best_f1_train_ce"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} in {key}")
        lookup[key] = row
        if manifest is not None:
            dataset, _condition, seed = key
            provenance = manifest["datasets"][dataset]
            expected = {
                "semantic_cache_sha256": provenance["cache_sha256"],
                "feature_fingerprint": provenance["feature_fingerprint"],
                "candidate_fingerprint": provenance["candidate_fingerprint"],
                "v7b_fingerprint": provenance["v7b_fingerprint"],
                "split_fingerprint": provenance["seed_split_fingerprints"][str(seed)],
            }
            for field, value in expected.items():
                if row.get(field) != value:
                    raise ValueError(f"manifest provenance mismatch for {key} field={field}")
    if require_complete and set(lookup) != RUN_KEYS:
        raise ValueError(f"campaign incomplete; missing={sorted(RUN_KEYS - set(lookup))}")
    for dataset in DATASETS:
        for seed in SEEDS:
            paired = [lookup[(dataset, condition, seed)] for condition in CONDITIONS if (dataset, condition, seed) in lookup]
            if paired and (
                len({row["initial_model_sha256"] for row in paired}) != 1
                or len({row["initial_classifier_sha256"] for row in paired}) != 1
                or len({row["split_fingerprint"] for row in paired}) != 1
                or len({row["semantic_cache_sha256"] for row in paired}) != 1
            ):
                raise ValueError(f"Dataset × Seed initialization/split/cache mismatch: {dataset}/seed_{seed}")
    return lookup


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def compute_paired_comparisons(lookup: dict[tuple[str, str, int], dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    contrasts = (("A1_separate_intra", "A0_legacy"), ("A2_cross_source", "A1_separate_intra"), ("A2_cross_source", "A0_legacy"))
    for dataset in DATASETS:
        for left, right in contrasts:
            for metric in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce"):
                differences = []
                for seed in SEEDS:
                    lv = float(lookup[(dataset, left, seed)][metric])
                    rv = float(lookup[(dataset, right, seed)][metric])
                    delta = lv - rv
                    differences.append(delta)
                    rows.append({"dataset": dataset, "seed": seed, "contrast": f"{left}-{right}", "metric": metric, "left_value": lv, "right_value": rv, "difference": delta})
                mean, sd = _mean_sd(differences)
                rows.append({"dataset": dataset, "seed": "mean_sample_sd", "contrast": f"{left}-{right}", "metric": metric, "left_value": "", "right_value": "", "difference": mean, "sample_sd": sd, "positive_seed_count": sum(value > 0 for value in differences), "negative_seed_count": sum(value < 0 for value in differences)})
    return rows


def compute_summary(lookup: dict[tuple[str, str, int], dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    metrics = ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce", "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap", "best_f1_train_acc", "best_f1_train_macro_f1", "best_f1_train_ce", "best_f1_train_val_acc_gap", "best_f1_train_val_macro_f1_gap", "best_f1_train_val_ce_gap")
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for metric in metrics:
                values = [float(lookup[(dataset, condition, seed)][metric]) for seed in SEEDS]
                mean, sd = _mean_sd(values)
                rows.append({"dataset": dataset, "condition": condition, "metric": metric, "mean": mean, "sample_sd": sd, **{f"seed_{seed}": values[i] for i, seed in enumerate(SEEDS)}})
    return rows


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return out


def _pct(mean: float, sd: float) -> str:
    return f"{100*mean:.3f} ± {100*sd:.3f}%"


def main() -> None:
    run_rows = read_csv(DATA / "runs.csv")
    history = read_csv(DATA / "epoch_history.csv")
    failures = json.loads((DATA / "failures.json").read_text()) if (DATA / "failures.json").exists() else {"failures": []}
    manifest = json.loads((DATA / "manifest.json").read_text())
    lookup = validate_runs(run_rows, manifest=manifest)
    preflight = json.loads((DATA / "preflight.json").read_text())
    views = read_csv(DATA / "view_diagnostics.csv")
    routers = read_csv(DATA / "router_diagnostics.csv")
    paired = compute_paired_comparisons(lookup)
    summary = compute_summary(lookup)
    write_csv(DATA / "paired_comparisons.csv", paired)
    write_csv(DATA / "summary.csv", summary)
    if len(history) != sum(int(row["epochs_completed"]) for row in run_rows):
        raise ValueError("epoch history row count does not equal summed completed epochs")
    if len(run_rows) != 27 or any(row.get("test_evaluation", "false").lower() != "false" for row in run_rows):
        raise ValueError("expected exactly 27 Test-disabled complete runs")

    lines = [
        "# MAG View-Collaborative Experts V1: Pilot Report", "",
        "## Protocol and provenance", "",
        f"- Branch: `{manifest.get('branch', 'unknown')}`; source/freeze commit: `{manifest.get('implementation_freeze_commit', 'not recorded')}`.",
        f"- Runs: {len(run_rows)}/27; epoch-history rows: {len(history)}; retained failures: {len(failures.get('failures', []))}.",
        f"- Protocol: `{manifest.get('protocol_version')}`; Test evaluation disabled and Test labels masked before NC training/evaluation.",
        f"- Device: {preflight.get('device_name')}; physical GPU {preflight.get('physical_gpu')}; PyTorch `{preflight.get('environment', {}).get('pytorch')}`, CUDA build `{preflight.get('environment', {}).get('cuda_build')}`.",
        "- Results are Validation-only and paired by Dataset × Seed. The historical reports are context, not the acceptance target.", "",
        "## Best Accuracy checkpoint", "",
    ]
    for metric, label in (("val_acc", "Validation Accuracy"), ("val_macro_f1", "Validation Macro-F1"), ("val_ce", "Validation CE")):
        lines.extend([f"### {label}", ""])
        table_rows = []
        for dataset in DATASETS:
            for condition in CONDITIONS:
                values = [float(lookup[(dataset, condition, seed)][metric]) for seed in SEEDS]
                mean, sd = _mean_sd(values)
                epochs = [int(lookup[(dataset, condition, seed)]["best_accuracy_epoch"]) for seed in SEEDS]
                table_rows.append([dataset, condition, _pct(mean, sd) if metric != "val_ce" else f"{mean:.5f} ± {sd:.5f}", ", ".join(map(str, epochs))])
        lines.extend(_markdown_table(table_rows, ["Dataset", "Condition", label, "Best epochs"])); lines.append("")

    lines.extend(["## Best Macro-F1 checkpoint", ""])
    f1_rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            vals = [float(lookup[(dataset, condition, seed)]["best_f1_val_macro_f1"]) for seed in SEEDS]
            mean, sd = _mean_sd(vals)
            acc = [float(lookup[(dataset, condition, seed)]["best_f1_val_acc"]) for seed in SEEDS]
            f1_rows.append([dataset, condition, _pct(mean, sd), _pct(*_mean_sd(acc))])
    lines.extend(_markdown_table(f1_rows, ["Dataset", "Condition", "Macro-F1", "Accuracy"])); lines.append("")

    lines.extend(["## Paired contrasts", "", "Values are left minus right; per-seed rows are in `data/paired_comparisons.csv`. These are descriptive three-seed comparisons, not significance tests.", ""])
    contrast_rows = []
    for dataset in DATASETS:
        for contrast in ("A1_separate_intra-A0_legacy", "A2_cross_source-A1_separate_intra", "A2_cross_source-A0_legacy"):
            entries = [r for r in paired if r["dataset"] == dataset and r["contrast"] == contrast and r["seed"] == "mean_sample_sd"]
            acc = next(r for r in entries if r["metric"] == "val_acc")
            f1 = next(r for r in entries if r["metric"] == "val_macro_f1")
            contrast_rows.append([dataset, contrast, f"{100*acc['difference']:.3f} ± {100*acc['sample_sd']:.3f} pp", f"{100*f1['difference']:.3f} ± {100*f1['sample_sd']:.3f} pp", f"{acc['positive_seed_count']}/3", f"{f1['positive_seed_count']}/3"])
    lines.extend(_markdown_table(contrast_rows, ["Dataset", "Contrast", "Accuracy Δ", "Macro-F1 Δ", "Acc positive", "F1 positive"])); lines.append("")

    lines.extend(["## Train/Validation, capacity, and resource profile", ""])
    resource_rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            rows = [lookup[(dataset, condition, seed)] for seed in SEEDS]
            resource_rows.append([
                dataset, condition,
                " / ".join(f"{statistics.mean(float(r[k]) for r in rows):.4f}" for k in ("train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap")),
                " / ".join(f"{statistics.mean(float(r[k]) for r in rows):.4f}" for k in ("best_f1_train_val_acc_gap", "best_f1_train_val_macro_f1_gap", "best_f1_train_val_ce_gap")),
                f"{statistics.mean(float(r['model_parameters']) + float(r['classifier_parameters']) for r in rows):.0f}",
                f"{statistics.mean(float(r['model_parameters']) + float(r['classifier_parameters']) for r in rows):.0f}",
                f"{statistics.mean(float(r['wall_seconds']) for r in rows):.1f}",
                f"{statistics.mean(float(r['peak_allocated_gib']) for r in rows):.2f}",
            ])
    lines.extend(_markdown_table(resource_rows, ["Dataset", "Condition", "Best-Acc Train−Val Acc/F1/CE gap means", "Best-F1 Train−Val Acc/F1/CE gap means", "Params", "Trainable Params", "Wall s", "Peak allocated GiB"])); lines.append("")

    lines.extend(["## View and Router diagnostics", "", "Scale and usage values describe what the frozen checkpoint computes, not causal task utility. Details are retained in `data/view_diagnostics.csv` and `data/router_diagnostics.csv`.", ""])
    view_rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for view in ("physical", "same", "cross"):
                selected = [r for r in views if r["dataset"] == dataset and r["condition"] == condition and r["checkpoint"] == "best_accuracy" and r["view"] == view]
                rms = [float(r["weighted_contribution_rms"]) for r in selected if r.get("weighted_contribution_rms") not in ("", None)]
                gamma = [float(r["gamma"]) for r in selected if r.get("gamma") not in ("", None)]
                raw_rms = [float(r["input_rms"]) for r in selected if r.get("input_rms") not in ("", None)]
                calibrated_rms = [float(r["calibrated_rms"]) for r in selected if r.get("calibrated_rms") not in ("", None)]
                correction_rms = [float(r["expert_correction_rms"]) for r in selected if r.get("expert_correction_rms") not in ("", None)]
                view_rows.append([dataset, condition, view, f"{statistics.mean(raw_rms):.5f}" if raw_rms else "—", f"{statistics.mean(calibrated_rms):.5f}" if calibrated_rms else "—", f"{statistics.mean(correction_rms):.5f}" if correction_rms else "—", f"{statistics.mean(rms):.5f}" if rms else "—", f"{statistics.mean(gamma):.5f}" if gamma else "—"])
    lines.extend(_markdown_table(view_rows, ["Dataset", "Condition", "View", "Input RMS", "Calibrated RMS", "Expert correction RMS", "Weighted contribution RMS", "Gamma/strength"])); lines.append("")
    route_use = defaultdict(list)
    for row in routers:
        if row.get("checkpoint") == "best_accuracy":
            route_use[(row["dataset"], row["condition"], row["view"])].append(row)
    route_rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for view in ("physical", "same", "cross"):
                selected = route_use.get((dataset, condition, view), [])
                entropies = [float(r["entropy"]) for r in selected if r.get("entropy") not in ("", None)]
                shares = []
                overlaps = [float(r["exact_same_cross_top2_match"]) for r in selected if r.get("exact_same_cross_top2_match") not in ("", None, "null")]
                for row in selected:
                    try:
                        share = json.loads(row["top2_selection_share"]) if isinstance(row["top2_selection_share"], str) else row["top2_selection_share"]
                        if share:
                            shares.append([float(value) for value in share])
                    except (ValueError, TypeError):
                        continue
                mean_share = [statistics.mean(values) for values in zip(*shares)] if shares else []
                route_rows.append([dataset, condition, view, f"{statistics.mean(entropies):.4f}" if entropies else "—", json.dumps(mean_share), f"{statistics.mean(overlaps):.4f}" if overlaps else "—"])
    lines.extend(_markdown_table(route_rows, ["Dataset", "Condition", "View", "Router entropy", "Mean Top-2 selection share", "Same/Cross exact-route match"])); lines.append("")

    intervention_rows = read_csv(DATA / "frozen_interventions.csv")
    sensitivity_table = []
    for dataset in DATASETS:
        for condition in CONDITIONS[1:]:
            for name in sorted({r["intervention"] for r in intervention_rows if r["dataset"] == dataset and r["condition"] == condition}):
                selected = [r for r in intervention_rows if r["dataset"] == dataset and r["condition"] == condition and r["intervention"] == name]
                sensitivity_table.append([dataset, condition, name, f"{100*statistics.mean(float(r['val_acc_delta']) for r in selected):.3f} pp", f"{100*statistics.mean(float(r['val_macro_f1_delta']) for r in selected):.3f} pp", f"{statistics.mean(float(r['prediction_flip_fraction']) for r in selected):.4f}"])
    lines.extend(["## Frozen view shutoff sensitivity", "", "These are checkpoint interventions with no retraining.", ""])
    lines.extend(_markdown_table(sensitivity_table, ["Dataset", "Condition", "View shutoff", "Accuracy Δ", "Macro-F1 Δ", "Prediction flip fraction"])); lines.append("")

    class_rows = read_csv(DATA / "classwise_metrics.csv")
    fashion_rows = [r for r in class_rows if r["dataset"] == "ele-fashion" and r["checkpoint"] in {"best_accuracy", "best_f1"} and int(r["class_id"]) in {1, 11}]
    lines.extend(["## ele-fashion minority classes", "", "Precision, recall, F1, support and predicted count are per run in `data/classwise_metrics.csv`; the selected rows below show per-seed class results.", ""])
    class_table = [[r["checkpoint"], r["condition"], r["seed"], r["class_id"], f"{float(r['precision']):.4f}", f"{float(r['recall']):.4f}", f"{float(r['f1']):.4f}", r["support"], r["predicted_count"]] for r in fashion_rows]
    lines.extend(_markdown_table(class_table, ["Checkpoint", "Condition", "Seed", "Class", "Precision", "Recall", "F1", "Support", "Predicted"])); lines.append("")

    lines.extend([
        "## Interpretation and limits", "",
        "A1−A0 measures the complete separated same-source semantic-view condition; A2−A1 measures the additional cross-source view bundle. Neither contrast isolates a single learned scalar or a retrained component ablation. Frozen view shutoff is checkpoint sensitivity only.",
        "The primary readout is the paired Validation Accuracy and Macro-F1 pattern over these three fixed seeds. Cross-dataset consistency, minority-class behavior, CE, Train/Validation gaps, router use, contribution scales, and compute cost must be considered together. This study does not use Test metrics and makes no Test-set claim.",
        "", "### Outstanding uncertainties", "",
        "Three seeds give limited uncertainty estimates. View routes share the existing experts and are co-trained with the classifier, so expert-use diagnostics do not establish independent mechanism utility. Any remaining seed variability, optimizer sensitivity, and feature/graph interaction are not identified by this experiment matrix.", "",
    ])
    (STUDY / "PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    test_evidence_path = DATA / "test_results.json"
    test_evidence = json.loads(test_evidence_path.read_text()) if test_evidence_path.exists() else {"status": "not recorded"}
    integrity = json.loads((DATA / "artifact_integrity.json").read_text()) if (DATA / "artifact_integrity.json").exists() else {"checkpoints": []}
    expected_checkpoints = 2 * len(RUN_KEYS)
    verified_checkpoints = 0
    for item in integrity.get("checkpoints", []):
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"checkpoint missing or size changed: {path}")
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"checkpoint SHA256 mismatch: {path}")
        verified_checkpoints += 1
    if len(run_rows) == 27 and verified_checkpoints != expected_checkpoints:
        raise ValueError(f"expected {expected_checkpoints} checkpoints, verified {verified_checkpoints}")
    preflight_rows = preflight.get("datasets", {})
    errors = [float(row["a0_forward_max_abs_error"]) for row in preflight_rows.values()]
    max_preflight_error = max(errors, default=float("nan"))
    peak_gib = max((float(r["peak_allocated_gib"]) for r in run_rows), default=0.0)
    failure_count = len(failures.get("failures", []))
    audit = [
        "# Implementation Audit", "",
        f"- Branch: `{manifest.get('branch', 'unknown')}`.",
        f"- Implementation Freeze Commit: `{manifest.get('implementation_freeze_commit', 'unknown')}`.",
        f"- Evidence/run count: {len(run_rows)}/27; epoch-history rows {len(history)}; failed/retried attempts recorded: {failure_count}.",
        f"- Test results: `{test_evidence.get('summary', test_evidence.get('status'))}`.",
        f"- Checkpoints independently checked for file size and SHA256: {verified_checkpoints}/{expected_checkpoints}.",
        f"- A0 real-data forward compatibility maximum absolute error: {max_preflight_error:.9g}; per-dataset values are in `data/preflight.json`.",
        f"- Seed42 initialization and split gate: `{(DATA / 'seed42_gate.json').is_file() and json.loads((DATA / 'seed42_gate.json').read_text()).get('status') == 'pass'}`.",
        f"- Physical GPU: {manifest.get('physical_gpu', preflight.get('physical_gpu'))}; device `{preflight.get('device_name')}`; maximum formal allocated memory {peak_gib:.2f} GiB.",
        "- Test evaluation: disabled. The runner replaced Test labels with `-100` before calling NC; no test metric columns are accepted by the analyzer.",
        "- Split and cache fingerprints: `data/manifest.json`; initial model/head hashes and split match are checked per Dataset × Seed.",
        "- Run histories, classwise scores, confusion matrices, degree groups, transitions, view/router diagnostics and frozen shutoff sensitivity are retained as CSV/JSON under `data/`.",
        "", "## Engineering findings", "",
        "The preflight first found that V7B.1 source provenance values are proposal counts, not Boolean flags; validation was corrected to preserve the existing `>0` operator semantics. The first full-graph backward then found an extra expert axis in the physical mixture, causing a 159 GiB illegal indexing allocation. The mixture output shape was fixed, asserted in tests, and the A1/A2 full-graph one-step preflight passed on all three datasets. No formal training began before these fixes.",
        "", "## Provenance and safety checks", "",
        "- A0 delegates to the frozen V2.2 M0 forward. The new common parameter tree and classifier initialization were compared against it on real data.",
        "- A1/A2 use V7B.1 Text/Visual source supports and global node IDs. The cross-source direction tests confirm Text receives Visual-KNN and Visual receives Text-KNN messages.",
        "- Semantic view masks are applied after expert response; physically isolated nodes retain valid semantic messages.",
        "- Full graph only; link prediction is explicitly unsupported. No labels were used for graph/cache creation.",
        "- Current evidence does not establish causal utility for individual routed experts; frozen closure is checkpoint sensitivity only.", "",
    ]
    (STUDY / "IMPLEMENTATION_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")

    aggregate = {}
    for contrast in (f"{CONDITIONS[1]}-{CONDITIONS[0]}", f"{CONDITIONS[2]}-{CONDITIONS[1]}", f"{CONDITIONS[2]}-{CONDITIONS[0]}"):
        aggregate[contrast] = {}
        for metric in ("val_acc", "val_macro_f1"):
            values = [r["difference"] for r in paired if r["contrast"] == contrast and r["metric"] == metric and r["seed"] != "mean_sample_sd"]
            aggregate[contrast][metric] = {"mean": statistics.mean(values), "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values)}
    memo = [
        "# Decision Memo", "",
        "## Decision basis", "",
        "This memo uses only the internal A0/A1/A2 Validation comparisons paired by dataset and seed. Historical V7A/RM-CGE results are design context, not an exact score target. Three seeds per dataset support descriptive replication only; no significance claim is made.", "",
        "## Measured paired results", "",
    ]
    for contrast, metrics in aggregate.items():
        acc, f1 = metrics["val_acc"], metrics["val_macro_f1"]
        memo.append(f"- `{contrast}`: Accuracy mean Δ {100*acc['mean']:.3f} pp ({acc['positive']}/9 positive pairs); Macro-F1 mean Δ {100*f1['mean']:.3f} pp ({f1['positive']}/9 positive pairs).")
    memo.extend([
        "", "## Interpretation", "",
        "A1−A0 is the complete condition comparison for separating source-specific KNN context into its own semantic view. A2−A1 adds the cross-source support view. These contrasts do not isolate gamma, a particular expert, or an independent retraining intervention. Interpret any aggregate improvement alongside dataset-specific pairs, best-F1 checkpoints, Train/Validation gaps, classwise and degree-stratified results, view contribution scales, and GPU cost.",
        "The view/router diagnostics establish that the implementation executed with measured nonzero propagation and expert responses where supports exist; they do not by themselves prove classification value. Frozen view closure is checkpoint sensitivity. It is not a retrained causal ablation.",
        "", "## Scope limits", "",
        "No Test metrics were produced. The study does not estimate a universal effect beyond these datasets/splits/seeds, and does not separately identify optimization, stochastic, or class-prior causes of run variation. No new model version is prescribed by this memo.", "",
    ])
    (STUDY / "DECISION_MEMO.md").write_text("\n".join(memo), encoding="utf-8")
    print(f"validated and summarized {len(run_rows)} runs; wrote reports under {STUDY}")


if __name__ == "__main__":
    main()
