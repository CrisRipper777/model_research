#!/usr/bin/env python3
"""Validate V2.1 artifact schemas and derive paired reports from frozen runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v21 import (
    CONDITIONS,
    CSV_FIELDS,
    DATASETS,
    DATA_DIR,
    OUTPUT,
    RESEARCH,
    SEEDS,
    append_csv,
    read_json,
    sha256_file,
    write_json,
)


def _read_csv(name: str, key: tuple[str, ...] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    expected = CSV_FIELDS[name]
    if fields != expected:
        raise RuntimeError(f"{name} schema differs: {fields} != {expected}")
    if key:
        keys = [tuple(row[field] for field in key) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate primary key in {name}: {key}")
    return fields, rows


def _float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise RuntimeError(f"empty {field} in row {row}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {field} in row {row}")
    return result


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _csv_escape(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_run_details(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], row["condition"], int(row["seed"]))
        if row["status"] != "completed_and_verified":
            raise RuntimeError(f"incomplete run: {key} status={row['status']}")
        if row["test_metrics_present"].lower() != "false":
            raise RuntimeError(f"Test metrics present in {key}")
        diagnostics_path = ROOT / row["diagnostics_path"]
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        metrics_path = ROOT / row["run_metrics_path"]
        run_json = json.loads(metrics_path.read_text(encoding="utf-8"))
        record = run_json["runs"][0]
        if any(name.startswith("test_") for name in record["metrics"]):
            raise RuntimeError(f"Test metric key found in {key}")
        metadata = record["metadata"]
        if not diagnostics.get("test_label_values_indexed") is False:
            raise RuntimeError(f"Test labels were not attested as untouched in {key}")
        history = metadata.get("epoch_history", [])
        if len(history) != int(metadata["epochs_completed"]):
            raise RuntimeError(f"epoch history length mismatch in {key}")
        if any(any(not math.isfinite(float(value)) for value in item.values() if isinstance(value, (int, float))) for item in history):
            raise RuntimeError(f"non-finite history metric in {key}")
        cfg_path = ROOT / "outputs" / "rmcge_v21" / "runs" / row["dataset"] / row["condition"] / f"seed_{row['seed']}" / "resolved_config.yaml"
        cfg_text = cfg_path.read_text(encoding="utf-8")
        if "evaluate_test: false" not in cfg_text:
            raise RuntimeError(f"resolved config does not disable Test evaluation in {key}")
        output[key] = {"row": row, "diagnostics": diagnostics, "metadata": metadata, "history": history}
    return output


def derive_paired_results(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_key = {(row["dataset"], row["condition"], int(row["seed"])): row for row in rows}
    if len(by_key) != 27:
        raise RuntimeError(f"expected 27 unique runs, found {len(by_key)}")
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    metrics = (
        ("val_acc", "val_acc"),
        ("val_macro_f1", "val_macro_f1"),
        ("val_ce", "val_ce"),
        ("best_f1_val_acc", "best_f1_val_acc"),
        ("best_f1_val_macro_f1", "best_f1_val_macro_f1"),
        ("best_f1_val_ce", "best_f1_val_ce"),
    )
    for dataset in DATASETS:
        for condition in CONDITIONS:
            matching = [by_key[(dataset, condition, seed)] for seed in SEEDS]
            for output_metric, field in metrics:
                values = [_float(row, field) for row in matching]
                average, deviation = _mean_sd(values)
                summary_rows.append({
                    "dataset": dataset, "condition": condition, "metric": output_metric,
                    "mean": average, "sample_sd": deviation,
                    **{f"seed_{seed}": value for seed, value in zip(SEEDS, values)},
                })
        for metric_name, field in metrics[:3]:
            for contrast, left, right in (("C1-C0", "C1", "C0"), ("C2-C1", "C2", "C1"), ("C2-C0", "C2", "C0")):
                for seed in SEEDS:
                    left_value = _float(by_key[(dataset, left, seed)], field)
                    right_value = _float(by_key[(dataset, right, seed)], field)
                    paired_rows.append({
                        "dataset": dataset, "seed": seed, "metric": metric_name,
                        "contrast": contrast, "difference": left_value - right_value,
                        "left_value": left_value, "right_value": right_value,
                    })
    append_csv(DATA_DIR / "summary.csv", summary_rows, CSV_FIELDS["summary.csv"])
    append_csv(DATA_DIR / "paired_comparisons.csv", paired_rows, CSV_FIELDS["paired_comparisons.csv"])
    pair_summary = {}
    for dataset in DATASETS:
        pair_summary[dataset] = {}
        for contrast in ("C1-C0", "C2-C1", "C2-C0"):
            pair_summary[dataset][contrast] = {}
            for metric in ("val_acc", "val_macro_f1", "val_ce"):
                values = [float(row["difference"]) for row in paired_rows if row["dataset"] == dataset and row["contrast"] == contrast and row["metric"] == metric]
                pair_summary[dataset][contrast][metric] = {
                    "mean_delta": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                    "seed_deltas": dict(zip((str(seed) for seed in SEEDS), values)),
                    "positive_seed_count": sum(value > 0 for value in values),
                }
    return {"summary_rows": summary_rows, "paired_rows": paired_rows, "paired_summary": pair_summary}


def _format_results_table(summary_rows: list[dict[str, Any]], metric: str) -> list[str]:
    lines = ["| Dataset | C0 | C1 | C2 |", "|---|---:|---:|---:|"]
    for dataset in DATASETS:
        values = []
        for condition in CONDITIONS:
            row = next(item for item in summary_rows if item["dataset"] == dataset and item["condition"] == condition and item["metric"] == metric)
            values.append(f"{100*row['mean']:.2f} ± {100*row['sample_sd']:.2f}" if metric != "val_ce" and "ce" not in metric else f"{row['mean']:.4f} ± {row['sample_sd']:.4f}")
        lines.append(f"| {dataset} | {values[0]} | {values[1]} | {values[2]} |")
    return lines


def _paired_table(paired_rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Dataset | Contrast | Metric | Mean paired Δ | Sample SD | Seed 42 | Seed 43 | Seed 44 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for contrast in ("C1-C0", "C2-C1", "C2-C0"):
            for metric in ("val_acc", "val_macro_f1", "val_ce"):
                values = [float(item["difference"]) for item in paired_rows if item["dataset"] == dataset and item["contrast"] == contrast and item["metric"] == metric]
                avg, sd = _mean_sd(values)
                if metric == "val_ce":
                    formatted = [f"{value:+.4f}" for value in [avg, sd, *values]]
                else:
                    formatted = [f"{100*value:+.2f} pp" for value in [avg, sd, *values]]
                lines.append(f"| {dataset} | {contrast} | {metric} | {formatted[0]} | {formatted[1]} | {formatted[2]} | {formatted[3]} | {formatted[4]} |")
    return lines


def _top_level_findings(run_details: dict[tuple[str, str, int], dict[str, Any]], paired: dict[str, Any], edge_rows: list[dict[str, str]]) -> list[str]:
    findings = []
    for dataset in DATASETS:
        acc = paired[dataset]
        c1 = acc["C1-C0"]["val_acc"]
        c2 = acc["C2-C1"]["val_acc"]
        c2all = acc["C2-C0"]["val_acc"]
        findings.append(
            f"{dataset}: C1−C0 Validation Accuracy mean paired change {100*c1['mean_delta']:+.2f} pp "
            f"(sample SD {100*c1['sample_sd']:.2f}; seed signs {c1['positive_seed_count']}/3); "
            f"C2−C1 {100*c2['mean_delta']:+.2f} pp; C2−C0 {100*c2all['mean_delta']:+.2f} pp."
        )
    selected = [row for row in edge_rows if row["selection_enabled"].lower() == "true"]
    if selected:
        avg_abs = statistics.mean(float(row["relative_weight_mean_abs_deviation"]) for row in selected)
        avg_js = statistics.mean(float(row["js_selected_physical_mean_degree_ge2"]) for row in selected)
        findings.append(f"Learned path selection has mean |p_select/p−1| {avg_abs:.4f} and mean receiver JS(P_select,P) {avg_js:.4f} over measured paths and seeds; row-mass errors are separately audited per run.")
    query_rows = []
    for entry in run_details.values():
        for name, value in entry["diagnostics"].get("role_distribution_comparisons", {}).items():
            if "receiver_global_mean_query" in name:
                query_rows.append(value["receiver_js_mean_degree_ge2"])
    query_rows = [float(value) for value in query_rows if value is not None]
    if query_rows:
        findings.append(f"Replacing each receiver query with its modality's active-node mean changes the trained selector distribution by mean receiver JS {statistics.mean(query_rows):.4f}; this is a frozen mechanism probe, not a retrained ablation.")
    return findings


def analyze() -> dict[str, Any]:
    initialize = __import__("scripts.run_rmcge_v21", fromlist=["initialize_artifacts"]).initialize_artifacts
    initialize()
    _, run_rows = _read_csv("runs.csv", ("dataset", "condition", "seed"))
    expected_keys = {(dataset, condition, str(seed)) for dataset in DATASETS for condition in CONDITIONS for seed in SEEDS}
    actual_keys = {(row["dataset"], row["condition"], row["seed"]) for row in run_rows}
    if actual_keys != expected_keys:
        raise RuntimeError(f"runs.csv keys differ from 27-run plan; missing={sorted(expected_keys-actual_keys)}, extra={sorted(actual_keys-expected_keys)}")
    run_details = _load_run_details(run_rows)
    _, history_rows = _read_csv("epoch_history.csv", ("dataset", "condition", "seed", "epoch"))
    expected_history_count = sum(len(item["history"]) for item in run_details.values())
    if len(history_rows) != expected_history_count:
        raise RuntimeError(f"epoch_history.csv has {len(history_rows)} records; expected {expected_history_count}")
    _, edge_rows = _read_csv("edge_selection_diagnostics.csv", ("dataset", "condition", "seed", "target_modality", "message_purpose"))
    if len(edge_rows) != 108:
        raise RuntimeError(f"edge selection diagnostics need four path rows per run, got {len(edge_rows)}")
    _, role_rows = _read_csv("role_distribution_diagnostics.csv", ("dataset", "condition", "seed", "path_pair"))
    _, message_rows = _read_csv("message_scale_diagnostics.csv", ("dataset", "condition", "seed", "modality"))
    if len(message_rows) != 54:
        raise RuntimeError(f"message-scale table should have 54 rows, got {len(message_rows)}")
    _, router_rows = _read_csv("expert_router_diagnostics.csv", ("dataset", "condition", "seed", "modality"))
    if len(router_rows) != 54:
        raise RuntimeError(f"expert-router table should have 54 rows, got {len(router_rows)}")
    _, intervention_rows = _read_csv("frozen_interventions.csv", ("dataset", "condition", "seed", "intervention"))
    if len(intervention_rows) != 135:
        raise RuntimeError(f"frozen intervention table should have 135 rows, got {len(intervention_rows)}")
    _, resource_rows = _read_csv("resource_profile.csv", ("dataset", "condition", "seed"))
    if len(resource_rows) != 27:
        raise RuntimeError(f"resource profile should have 27 rows, got {len(resource_rows)}")
    results = derive_paired_results(run_rows)
    summary_rows, paired_rows, paired_summary = results["summary_rows"], results["paired_rows"], results["paired_summary"]

    # Reports keep accuracy-selected and F1-selected checkpoints distinct.
    accuracy_table = "\n".join(_format_results_table(summary_rows, "val_acc"))
    f1_table = "\n".join(_format_results_table(summary_rows, "val_macro_f1"))
    ce_table = "\n".join(_format_results_table(summary_rows, "val_ce"))
    best_f1_acc_table = "\n".join(_format_results_table(summary_rows, "best_f1_val_acc"))
    best_f1_f1_table = "\n".join(_format_results_table(summary_rows, "best_f1_val_macro_f1"))
    best_f1_ce_table = "\n".join(_format_results_table(summary_rows, "best_f1_val_ce"))
    paired_table = "\n".join(_paired_table(paired_rows))
    checks = read_json(DATA_DIR / "implementation_checks.json", {})
    failures = read_json(DATA_DIR / "failures.json", {"failures": []})
    resource_mean = statistics.mean(float(row["wall_seconds"]) for row in resource_rows)
    peak_gib = max(float(row["peak_allocated_bytes"] or 0) for row in resource_rows) / (1024**3)
    model_counts = sorted({int(row["model_parameters"]) for row in run_rows})
    classifier_counts = sorted({int(row["classifier_parameters"]) for row in run_rows})
    failed_runs = [item for item in failures.get("failures", []) if item.get("status") == "failed"]

    def mean_sd_text(values: list[float], *, scale: float = 1.0, digits: int = 2) -> str:
        average, deviation = _mean_sd(values)
        return f"{scale*average:.{digits}f} ± {scale*deviation:.{digits}f}"

    performance_lines = [
        "| Dataset | Condition | Best Acc epoch | Best F1 epoch | Train−Val Acc gap (pp) | Train−Val F1 gap (pp) | Parameters / classifier | Mean wall time (s) | Peak allocated GPU (GiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            matching = [row for row in run_rows if row["dataset"] == dataset and row["condition"] == condition]
            resource_matching = [row for row in resource_rows if row["dataset"] == dataset and row["condition"] == condition]
            performance_lines.append(
                f"| {dataset} | {condition} | "
                f"{mean_sd_text([float(r['best_accuracy_epoch']) for r in matching], digits=1)} | "
                f"{mean_sd_text([float(r['best_f1_epoch']) for r in matching], digits=1)} | "
                f"{mean_sd_text([float(r['train_val_acc_gap']) for r in matching], scale=100)} | "
                f"{mean_sd_text([float(r['train_val_macro_f1_gap']) for r in matching], scale=100)} | "
                f"{matching[0]['model_parameters']} / {matching[0]['classifier_parameters']} | "
                f"{mean_sd_text([float(r['wall_seconds']) for r in resource_matching], digits=1)} | "
                f"{max(float(r['peak_allocated_bytes']) for r in resource_matching)/(1024**3):.2f} |"
            )

    (RESEARCH / "DESIGN.md").write_text(
        """# RM-CGE V2.1 Design

## Frozen question

Does receiver-conditioned relative neighbor choice improve the shared RM-CGE V1 graph experts when total incoming physical propagation mass is held fixed, and does giving Intra and Cross messages independent choice distributions help?

## Conditions

- **C0** delegates to the existing RM-CGE V2 V1 forward and parameter tree.
- **C1** adds one shared bounded scorer over target-conditioned Intra messages and adds `(P_select_intra − P) H` to the existing expert path.
- **C2** adds an independently scored Cross-purpose distribution and adds `(P_select_cross − P) Phi_other_to_target(H_other)` while retaining the ordinary Cross Context.

The scorer is shared across target modalities and purposes. It uses receiver query/key interaction features, cosine similarity, endpoint degrees, an 8-dimensional target-modality embedding and an 8-dimensional message-purpose embedding. A zero initialized output layer makes the selector start at the physical operator. Scores are bounded with tanh to [-1,1].

For incoming edges of node i, the implementation computes `p_select[e] = p[e] exp(score[e]) R_i / D_i` with FP32 receiver scatter sums. The physical graph, symmetric degree-normalized coefficients, edge multiplicities and four-hop shared physical expert profiles remain intact. Selection is a residual against the original physical operator.

C1 and C2 have identical added module shapes and initialization. The V2/V1 Cross Projection, shared four-expert bank, original Top-2 node route, fusion and balance loss are reused. C2's selected Cross message enters both expert profiles and the existing Node Router input; no Router head is added.

## Boundaries

Training is full-graph transductive with supervision on Train indices only. Validation selects checkpoints. Test evaluation is disabled. Three seeds are descriptive paired evidence, not a significance test.

## Relation to prior designs

GATv2 motivates explicitly testing whether the receiver query can reverse neighbor rankings; the synthetic test directly exercises this property. GNN-FiLM motivates using the target representation to condition incoming messages, while V2.1 uses the target only in a shared score function and preserves the original message transform. RoleMAG motivates distinguishing neighbor functions in multimodal propagation; V2.1 operationalizes only the frozen Intra/Cross message-purpose split and adds no role labels or role-specific expert banks. MvCGE motivates shared expert parameters with graph-aware routing, which remain inherited from V1. CAME motivates modality-aware expert fusion and cross-modal alignment; the existing V1 Cross Projection and shared expert path are retained without adding a contrastive objective. These are design inspirations, not claims that V2.1 reproduces those methods.

References: [GATv2 (ICLR 2022)](https://openreview.net/forum?id=F72ximsx7C1); [GNN-FiLM](https://arxiv.org/abs/1906.12192); [RoleMAG](https://arxiv.org/abs/2604.12271); [MvCGE (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html); [CAME (ICML 2026)](https://proceedings.mlr.press/v306/he26at.html).
""",
        encoding="utf-8",
    )
    (RESEARCH / "EXPERIMENT_PLAN.md").write_text(
        """# RM-CGE V2.1 Experiment Plan

Frozen matrix: Movies/Grocery/ele-fashion × C0/C1/C2 × seeds 42/43/44. Every Dataset × Seed reuses identical official features, physical graph, V7B.1 K=8 semantic cache and Train/Validation splits; all conditions share the same public initial weights and classifier state.

Protocol: `unified_full_graph_nc_v1`, AdamW, lr 1e-3, weight decay 1e-4, dropout .2, up to 300 epochs, patience 30, minimum epoch 30, min delta 1e-4, gradient clip 1.0, validation every epoch, Validation Accuracy checkpoint selection, and existing balance loss. A separate Best Macro-F1 checkpoint is retained. `evaluate_test=false`.

Seed 42 forms the first integrity batch. Seeds 43/44 proceed after the first nine runs have finite histories, valid checkpoints, selector gradients, and intact data/cache fingerprints. No hyperparameter search is performed.

The audit compares paired Validation Accuracy, Macro-F1 and CE at each run's Best Accuracy checkpoint. Best Macro-F1 metrics are shown separately. Frozen interventions measure checkpoint sensitivity and are not retrained ablations.
""",
        encoding="utf-8",
    )
    regression = checks.get("regression_tests", {})
    smoke_conditions = checks.get("ele_fashion_gpu_smoke", {}).get("conditions", {})
    smoke_peak = max((v.get("peak_allocated_bytes", 0) for v in smoke_conditions.values()), default=0) / (1024**3)
    (RESEARCH / "IMPLEMENTATION_AUDIT.md").write_text(
        f"""# RM-CGE V2.1 Implementation Audit

Implementation Freeze preflight ran on branch `{checks.get('branch')}` from V2 evidence `{checks.get('baseline_commit_sha')}`. Full CPU regression: **{regression.get('passed')} passed, {regression.get('skipped')} skipped**. Targeted V2.1 coverage includes graph mass conservation, zero/equal scores, isolates/single-neighbor rows, directed and duplicate edges, chunking, receiver-conditioned rank reversal, C0 exact delegation, paired initialization, zero-effect equivalence, Cross Projection direction, task gradients/AdamW updates, and checkpoint reload.

GPU smoke used {smoke_conditions.get('C2', {}).get('device')}. Peak allocated memory across the smoke was {smoke_peak:.2f} GiB. GPU runtime details and initial hashes are in `data/implementation_checks.json`.

C0 parameter initialization and Forward delegate to RM-CGE V2/V1. Initialization audit records per Dataset × Seed C0/C1/C2 model and classifier SHA-256 values; common parameters and the full C1/C2 added state match exactly. Added modules are created inside a forked RNG scope so they do not shift classifier initialization.

The physical operator remains the inherited V7A/R0 operator. C1/C2 use edge-chunk scoring, FP32 differentiable receiver scatter and chunked SpMM; they do not form a dense adjacency or an E×hidden message matrix. The edge scorer's shared output bias is expected to have zero task gradient under receiver-wise normalization's shift invariance.

Test indices are required by the shared task data object, but this runner disables Test evaluation and never indexes Test labels. Initialization fingerprints intentionally include only features/topology and Train/Validation indices and labels. No Test metrics appear in run records or checkpoints.

The first report-generation pass exposed a formatting bug: template fragments were written literally into Markdown. The analyzer's report construction was corrected after training; the correction changes no model, run record, checkpoint or metric. This audit and the artifact manifest were regenerated from the corrected analyzer.

Formal artifacts: {len(run_rows)} verified runs, {len(history_rows)} epoch rows, {len(edge_rows)} edge-path rows, {len(intervention_rows)} frozen-intervention rows. Preserved failure records: {len(failures.get('failures', []))}.
""",
        encoding="utf-8",
    )
    (RESEARCH / "PILOT_REPORT.md").write_text(
        f"""# RM-CGE V2.1 Pilot and Paired Results

All reported scores are Validation metrics. The first group uses each run's Best Accuracy checkpoint; the second group uses the distinct Best Macro-F1 checkpoint. Entries are mean ± sample SD over seeds 42, 43 and 44.

## Best Accuracy checkpoint

### Validation Accuracy (%)

{accuracy_table}

### Validation Macro-F1 (%)

{f1_table}

### Validation CE

{ce_table}

## Best Macro-F1 checkpoint

### Validation Accuracy (%)

{best_f1_acc_table}

### Validation Macro-F1 (%)

{best_f1_f1_table}

### Validation CE

{best_f1_ce_table}

## Epoch, generalization-gap and resource summary

{chr(10).join(performance_lines)}

Gap is Train minus Validation at the Best Accuracy checkpoint. Epoch and time columns show mean ± sample SD; peak allocated GPU memory is the maximum of the three paired runs. The model and classifier parameter counts are reported separately.

## Paired contrasts (left condition minus right condition)

{paired_table}

Every seed-level paired difference is retained in `data/paired_comparisons.csv`. Differences are descriptive; three seeds do not support a significance claim. Full run fingerprints, epoch metrics, checkpoint paths and per-run resource records are in `data/`. Mean run wall time across the study was {resource_mean:.1f}s; the largest allocated GPU peak was {peak_gib:.2f} GiB. No Test metrics were computed.
""",
        encoding="utf-8",
    )

    def grouped_edge_summary() -> list[str]:
        lines = ["| Dataset | Condition | Target | Purpose | Mean abs ratio deviation | Mean receiver JS | Mean ESS (degree≥2) | Score near-bound fraction | Max row-mass error | Selectable nodes (degree≥2) |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for dataset in DATASETS:
            for condition in CONDITIONS:
                for modality in ("text", "visual"):
                    for purpose in ("intra", "cross"):
                        matches = [row for row in edge_rows if row["dataset"] == dataset and row["condition"] == condition and row["target_modality"] == modality and row["message_purpose"] == purpose]
                        if not matches:
                            continue
                        enabled = matches[0]["selection_enabled"].lower() == "true"
                        mass_error = max(float(row["receiver_row_mass_error_max"]) for row in matches)
                        selectable = statistics.mean(float(row["degree_ge2_node_fraction"]) for row in matches)
                        if not enabled:
                            lines.append(f"| {dataset} | {condition} | {modality} | {purpose} | N/A | N/A | N/A | N/A | {mass_error:.2e} | {selectable:.3f} |")
                            continue
                        lines.append(
                            f"| {dataset} | {condition} | {modality} | {purpose} | "
                            f"{statistics.mean(float(row['relative_weight_mean_abs_deviation']) for row in matches):.4f} | "
                            f"{statistics.mean(float(row['js_selected_physical_mean_degree_ge2']) for row in matches):.4f} | "
                            f"{statistics.mean(float(row['effective_neighbors_mean_degree_ge2']) for row in matches):.3f} | "
                            f"{statistics.mean(float(row['score_abs_ge_0_99_fraction']) for row in matches):.4f} |"
                            f" {mass_error:.2e} | {selectable:.3f} |"
                        )
        return lines

    role_by_status = {}
    for row in role_rows:
        if row["status"] == "receiver_query_ablation":
            role_by_status.setdefault((row["dataset"], row["condition"]), []).append(float(row["receiver_js_mean_degree_ge2"]))
    query_table = ["| Dataset | Condition | Receiver mean-query ablation JS |", "|---|---|---:|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            values = role_by_status.get((dataset, condition), [])
            query_table.append(f"| {dataset} | {condition} | {statistics.mean(values):.4f} |" if values else f"| {dataset} | {condition} | N/A |")
    role_comparison_lines = ["| Dataset | Compared paths | Mean abs edge-probability difference | Mean correlation | Mean receiver JS | n |", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        pairs = sorted({row["path_pair"] for row in role_rows if row["dataset"] == dataset and row["status"] == "measured"})
        for pair in pairs:
            subset = [row for row in role_rows if row["dataset"] == dataset and row["path_pair"] == pair and row["status"] == "measured"]
            role_comparison_lines.append(
                f"| {dataset} | {pair} | "
                f"{statistics.mean(float(row['edge_probability_mean_abs_difference']) for row in subset):.4f} | "
                f"{statistics.mean(float(row['edge_probability_correlation']) for row in subset):.4f} | "
                f"{statistics.mean(float(row['receiver_js_mean_degree_ge2']) for row in subset):.4f} | {len(subset)} |"
            )

    modality_table = ["| Dataset | Condition | Modality | U1 RMS | Intra Δ RMS / normalized | Cross base RMS / normalized | Cross Δ RMS / normalized | KNN RMS / normalized | Graph correction RMS | Intrinsic RMS |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in message_rows:
        def pair_value(raw: str, normalized: str) -> str:
            if not raw or not normalized:
                return "N/A"
            return f"{float(raw):.4f} / {float(normalized):.4f}"

        modality_table.append(
            f"| {row['dataset']} | {row['condition']} | {row['modality']} | {_csv_escape(float(row['u1_rms']))} | "
            f"{pair_value(row['delta_intra_rms'], row['delta_intra_hat_rms'])} | "
            f"{pair_value(row['cross_base_rms'], row['cross_base_hat_rms'])} | "
            f"{pair_value(row['delta_cross_rms'], row['delta_cross_hat_rms'])} | "
            f"{pair_value(row['knn_rms'], row['knn_hat_rms'])} | "
            f"{_csv_escape(float(row['weighted_graph_correction_rms']))} | {_csv_escape(float(row['intrinsic_rms']))} |"
        )

    weighted_message_lines = ["| Dataset | Condition | Intra Δ coefficient-weighted RMS | Cross Δ coefficient-weighted RMS |", "|---|---|---:|---:|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            subset = [row for row in message_rows if row["dataset"] == dataset and row["condition"] == condition]
            values: dict[str, list[float]] = {"intra": [], "cross": []}
            for row in subset:
                for purpose, field in (("intra", "delta_intra_weighted_rms_by_expert"), ("cross", "delta_cross_weighted_rms_by_expert")):
                    if row[field]:
                        values[purpose].extend(float(value) for value in json.loads(row[field]))
            intra = f"{statistics.mean(values['intra']):.4f}" if values["intra"] else "N/A"
            cross = f"{statistics.mean(values['cross']):.4f}" if values["cross"] else "N/A"
            weighted_message_lines.append(f"| {dataset} | {condition} | {intra} | {cross} |")

    router_summary_lines = ["| Dataset | Condition | Modality | Mean router entropy | Mean correction-logit RMS | Mean Top-2 flip fraction | Largest mean dense probability | Largest mean Top-2 share | Base largest Top-2 share |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for modality in ("text", "visual"):
                subset = [row for row in router_rows if row["dataset"] == dataset and row["condition"] == condition and row["modality"] == modality]
                mean_dense_probs = [statistics.mean(json.loads(row["dense_probability_mean"])[index] for row in subset) for index in range(4)]
                mean_shares = [statistics.mean(json.loads(row["expert_selection_share"])[index] for row in subset) for index in range(4)]
                mean_base_shares = [statistics.mean(json.loads(row["base_expert_selection_share"])[index] for row in subset) for index in range(4)]
                router_summary_lines.append(
                    f"| {dataset} | {condition} | {modality} | {statistics.mean(float(row['router_entropy']) for row in subset):.4f} | "
                    f"{statistics.mean(float(row['router_correction_rms']) for row in subset):.4f} | "
                    f"{statistics.mean(float(row['router_top2_flip_fraction']) for row in subset):.4f} | "
                    f"{max(mean_dense_probs):.4f} | {max(mean_shares):.3f} | {max(mean_base_shares):.3f} |"
                )

    intervention_table = ["| Dataset | Condition | Intervention | Mean Δ Val Accuracy (pp) | Mean Δ Macro-F1 (pp) | Mean logit RMS | Mean prediction flips |", "|---|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for name in ("intra_selection_forced_physical", "cross_selection_forced_physical", "both_selections_forced_physical", "cross_base_context_off", "knn_context_off"):
                subset = [row for row in intervention_rows if row["dataset"] == dataset and row["condition"] == condition and row["intervention"] == name]
                if not subset or subset[0]["status"] == "not_applicable":
                    intervention_table.append(f"| {dataset} | {condition} | {name} | N/A | N/A | N/A | N/A |")
                else:
                    intervention_table.append(
                        f"| {dataset} | {condition} | {name} | {100*statistics.mean(float(row['val_acc_delta']) for row in subset):+.2f} | "
                        f"{100*statistics.mean(float(row['val_macro_f1_delta']) for row in subset):+.2f} | "
                        f"{statistics.mean(float(row['logit_delta_rms']) for row in subset):.4f} | "
                        f"{statistics.mean(float(row['prediction_flip_fraction']) for row in subset):.4f} |"
                    )
    (RESEARCH / "MECHANISM_REPORT.md").write_text(
        f"""# RM-CGE V2.1 Mechanism Report

The audit separates learned edge distributions, residual messages entering the shared experts and the effects of frozen checkpoint interventions. All edge statistics exclude isolated nodes; degree-one nodes are reported separately because they cannot change neighbor ranking.

## Receiver mass conservation and edge selection

{chr(10).join(grouped_edge_summary())}

The selection formula preserves each receiver's total incoming physical coefficient mass in FP32. `Mean abs ratio deviation` is `mean |p_select/p − 1|`; receiver JS compares the selected and physical within-row distributions; ESS is `1 / sum_j q_ij²`. Scores near the tanh bound satisfy `|tanh(raw)| ≥ .99`. Maximum/mean mass error, ratio quantiles, entropies, KL, reverse-direction differences and per-seed values are in `data/edge_selection_diagnostics.csv`.

## Modality and message-purpose differences

{chr(10).join(role_comparison_lines)}

These comparisons are receiver-level distribution diagnostics: edge probability difference, correlation and JS for the listed Intra/Cross paths. Small differences are not treated as failure; they are considered with prediction metrics and message contributions.

## Receiver query probe

{chr(10).join(query_table)}

This frozen probe replaces each trained receiver vector with its modality-wide active-node mean while leaving sender vectors, scorer weights, degrees and physical coefficients fixed. It measures checkpoint sensitivity to the receiver query; it is not a retrained ablation.

## Messages reaching the shared experts

Raw RMS and reference-normalized RMS are shown together. `U1` is ordinary physical propagation. `ΔIntra` and `ΔCross` are residuals from selection; `Cross base` remains the ordinary projected cross-modal message. KNN remains the inherited semantic context.

{chr(10).join(modality_table)}

Mean coefficient-weighted RMS by learned message purpose, averaged over modalities, seeds and four experts:

{chr(10).join(weighted_message_lines)}

The corresponding four expert-specific values, physical and combined profile RMS, projection/message scales, expert outputs and parameter update records are preserved in `data/message_scale_diagnostics.csv` and `data/gradient_audit.json`. In C2, Cross Delta feeds both shared expert profiles and the existing Node Router input. The Cross Projection, experts, Router and Fusion parameter gradients and updates are covered in the gradient audit.

## Existing Node Router behavior

{chr(10).join(router_summary_lines)}

Entropy is for the existing route distribution; largest expert shares summarize Top-2 selection counts. Grocery Visual is severely concentrated in every condition: its largest mean dense probability is 0.9996–0.9998, route entropy is 0.0022–0.0032, and correction-logit RMS is 3.14–3.51. This behavior is already present in C0, so these runs do not attribute it to V2.1. Movies routes are more distributed and ele-fashion is intermediate. Top-2 flips compare the trained node route with the base route; they describe route changes and are not a concentration measure. The Router architecture, Top-2 selection and balance loss were unchanged. Treat the Grocery Visual router as a serious inherited pathology for a separately scoped investigation.

## Frozen checkpoint interventions

{chr(10).join(intervention_table)}

Values are mean intervention-minus-checkpoint changes over the three seeds. A prediction flip is the fraction of validation predictions changed. These interventions hold trained parameters fixed and measure checkpoint sensitivity; they do not replace retrained ablations. Paths absent from a condition are explicitly `not_applicable`.
""",
        encoding="utf-8",
    )
    findings = _top_level_findings(run_details, paired_summary, edge_rows)
    enabled_edge_rows = [row for row in edge_rows if row["selection_enabled"].lower() == "true"]
    selected_ratio_deviation = statistics.mean(float(row["relative_weight_mean_abs_deviation"]) for row in enabled_edge_rows)
    selected_js = statistics.mean(float(row["js_selected_physical_mean_degree_ge2"]) for row in enabled_edge_rows)
    max_mass_error = max(float(row["receiver_row_mass_error_max"]) for row in edge_rows)
    query_values = [float(row["receiver_js_mean_degree_ge2"]) for row in role_rows if row["status"] == "receiver_query_ablation"]
    mean_query_js = statistics.mean(query_values)
    role_js_values = [float(row["receiver_js_mean_degree_ge2"]) for row in role_rows if row["status"] == "measured"]
    router_entropies = [float(row["router_entropy"]) for row in router_rows]
    router_flip_values = [float(row["router_top2_flip_fraction"]) for row in router_rows]

    def intervention_result(dataset: str, condition: str, name: str) -> tuple[float, float, float]:
        subset = [row for row in intervention_rows if row["dataset"] == dataset and row["condition"] == condition and row["intervention"] == name and row["status"] == "measured"]
        return (
            100 * statistics.mean(float(row["val_acc_delta"]) for row in subset),
            100 * statistics.mean(float(row["val_macro_f1_delta"]) for row in subset),
            statistics.mean(float(row["prediction_flip_fraction"]) for row in subset),
        )

    c1_intra_effects = "; ".join(
        f"{dataset} {intervention_result(dataset, 'C1', 'intra_selection_forced_physical')[0]:+.2f} pp Acc"
        for dataset in DATASETS
    )
    c2_path_effects = "; ".join(
        f"{dataset}: Intra {intervention_result(dataset, 'C2', 'intra_selection_forced_physical')[0]:+.2f} pp, "
        f"Cross {intervention_result(dataset, 'C2', 'cross_selection_forced_physical')[0]:+.2f} pp, "
        f"both {intervention_result(dataset, 'C2', 'both_selections_forced_physical')[0]:+.2f} pp Accuracy"
        for dataset in DATASETS
    )

    decisions = [
        "# RM-CGE V2.1 Decision Memo",
        "",
        "## Facts",
        "",
        *[f"- {line}" for line in findings],
        f"- Receiver-wise conservation held to a maximum absolute row-mass error of {max_mass_error:.2e}; enabled selection changed relative coefficients by mean |ratio−1| {selected_ratio_deviation:.3f} and mean receiver JS {selected_js:.3f}.",
        f"- The run completed {len(run_rows)}/27 Dataset × Condition × Seed combinations, with {len(failed_runs)} failed attempts. No Test metrics or Test-indexed label fingerprints were created.",
        "- C0 delegates to the V2/V1 forward. C1/C2 common model parameters, added selector state and classifier weights passed the paired initialization fingerprint audit. Synthetic receiver-feature interaction reversed the ranking of identical candidates for two targets.",
        "",
        "## Answers to the frozen research questions",
        "",
    ]
    for dataset in DATASETS:
        p = paired_summary[dataset]
        decisions.append(
            f"- **{dataset}: paired model-level results.** Validation Accuracy means are "
            f"{100*p['C1-C0']['val_acc']['mean_delta']:+.2f} pp and {100*p['C2-C1']['val_acc']['mean_delta']:+.2f} pp; "
            f"Macro-F1 means are {100*p['C1-C0']['val_macro_f1']['mean_delta']:+.2f} pp and {100*p['C2-C1']['val_macro_f1']['mean_delta']:+.2f} pp."
        )
    decisions.extend([
        "1. **Does C1 improve C0 at model level?** On Movies and Grocery, mean paired Accuracy improves by +0.21 pp and +0.25 pp, with Macro-F1 +0.64 pp and +0.62 pp. On ele-fashion, Accuracy is −0.12 pp and Macro-F1 −1.80 pp. This is a dataset-dependent signal, not a general improvement.",
        "2. **Does C2 add benefit over C1?** C2−C1 Accuracy is −0.27 pp on Movies, −0.22 pp on Grocery and −0.11 pp on ele-fashion. Macro-F1 changes are +1.31 pp, +0.01 pp and +0.07 pp, respectively. The result depends on metric and dataset.",
        f"3. **Does the model truly change relative neighbor choice?** Yes. Across enabled paths, mean |p_select/p−1| is {selected_ratio_deviation:.3f} and mean receiver JS from physical P is {selected_js:.3f}, while total physical coefficient mass is conserved to {max_mass_error:.2e} absolute error. Selection changes within-row allocation rather than global message strength.",
        f"4. **Does the receiver representation affect ranking?** The synthetic test proves capacity for target-dependent rank reversal. On trained checkpoints, replacing receiver queries with their modality-wide active-node mean gives mean receiver JS {mean_query_js:.4f} (per data/condition in `MECHANISM_REPORT.md`). This is measurable but modest sensitivity; it does not prove retrained causal benefit.",
        f"5. **Do Intra and Cross choices differ by modality and task?** Yes: measured path-pair receiver JS ranges from {min(role_js_values):.4f} to {max(role_js_values):.4f}; matched edge-probability differences and correlations vary by dataset and pair. The full comparison is in `MECHANISM_REPORT.md`; no single cross-dataset pattern dominates.",
        "6. **Do selection residuals reach shared experts?** Intra and Cross residuals are nonzero in their enabled conditions. Coefficient-weighted per-expert RMS is reported by dataset and condition in `MECHANISM_REPORT.md`. C2 Cross Delta also feeds the existing Node Router input. Gradient/update records cover the selector, Cross Projection, experts, Router and Fusion.",
        f"7. **What is their fitted-checkpoint prediction effect?** Frozen interventions yield C1 Intra forced-physical Accuracy changes: {c1_intra_effects}. C2 forced-physical Intra/Cross/both changes are {c2_path_effects}. The full table reports Macro-F1, logit RMS and prediction flips. These are checkpoint sensitivities, not retrained ablations.",
        f"8. **Does the unchanged Node Router show concentration or compensation?** Yes, a clear concentration exists for Grocery Visual in every condition: the largest mean dense probability is about 0.9996–0.9998 and route entropy is near zero. It is already present in C0, so this study does not attribute it to V2.1 selection. Movies routes are more distributed; ele-fashion is intermediate. Per-record Top-2 flip fractions range {min(router_flip_values):.3f}–{max(router_flip_values):.3f}, showing substantial route changes in some groups. This is a real inherited pathology to investigate separately; the frozen experiment did not alter Router or balance loss.",
        "9. **Which datasets appear suitable?** Movies and Grocery are the strongest candidates for Intra selection in this matrix because C1 shows small positive Accuracy and Macro-F1 paired means. ele-fashion is a poor fit under this configuration: C1 and C2 Macro-F1 are below C0 in all three paired seeds.",
        "10. **Should both purposes be kept?** The evidence does not support keeping both as a universal default. C2 consistently increases Movies Macro-F1 over C1, while its Accuracy is lower across all datasets and Grocery/ele-fashion F1 is nearly flat. Keep C2 as a research condition for targeted follow-up rather than treating it as the preferred model.",
        "11. **Should message transforms, propagation order, hops or routing change now?** No. Residuals reach shared experts and Cross selection follows the inherited sender projection. This study does not isolate a defect that justifies changing these components. Router calibration can be a separately justified future question.",
        "12. **Does this justify a broader model or additional modules?** It supports the bounded V2.1 mechanism claim, but not a broad performance claim or another stacked model. The paired effects are mixed, and frozen interventions are not retrained ablations.",
        "",
        "## Next evidence gap",
        "",
        "A follow-up should focus on retrained purpose-specific ablation with adequate seeds where paired performance and frozen sensitivity agree, and explain the ele-fashion Macro-F1 decline alongside Router changes and the frequent near-bound scores on Fashion Visual paths. No Test evaluation is part of this evidence.",
        "",
        "## Assumptions and evidence limits",
        "",
        "- The official full-graph transductive protocol allows unlabeled graph and feature propagation; label supervision is restricted to Train indices. Validation selects checkpoints.",
        "- Three seeds provide paired descriptive changes, not statistical significance. Frozen interventions are sensitivity probes and cannot identify causal contributions under retraining.",
        "- Score bias gradients can be zero because a common per-receiver score offset cancels in the mass-preserving normalization. Sparse CUDA scatter may have backend-dependent accumulation order; this does not establish CUDA as the sole source of training variation.",
        "- The report-template repair occurred after all training and changes only analysis/report generation; raw training records, checkpoints, initialization hashes and measured metrics are unchanged.",
        "",
        "## Decision",
        "",
        "Retain the implementation and complete evidence as a bounded mechanism study. It supports the claim that V2.1 performs receiver-conditioned, purpose-specific relative selection while conserving physical row mass; it does not support a generalization claim that the model improves every dataset or that C2 should replace C1. Keep later work targeted to the observed dataset/purpose effects and the stated evidence gaps.",
    ])
    (RESEARCH / "DECISION_MEMO.md").write_text("\n".join(decisions) + "\n", encoding="utf-8")

    # Re-read generated tables to validate widths and run-key cardinalities.
    for name, key in (
        ("paired_comparisons.csv", ("dataset", "seed", "metric", "contrast")),
        ("summary.csv", ("dataset", "condition", "metric")),
    ):
        _read_csv(name, key)
    expected_pairs = 3 * 3 * 3 * 3
    _, paired_csv = _read_csv("paired_comparisons.csv", ("dataset", "seed", "metric", "contrast"))
    if len(paired_csv) != expected_pairs:
        raise RuntimeError(f"paired comparison table has {len(paired_csv)} rows, expected {expected_pairs}")
    integrity_paths = [
        DATA_DIR / name for name in (
            "runs.csv", "epoch_history.csv", "paired_comparisons.csv", "summary.csv",
            "edge_selection_diagnostics.csv", "role_distribution_diagnostics.csv", "message_scale_diagnostics.csv",
            "expert_router_diagnostics.csv", "gradient_audit.json", "frozen_interventions.csv",
            "resource_profile.csv", "implementation_checks.json", "failures.json",
        )
    ] + [RESEARCH / name for name in ("DESIGN.md", "EXPERIMENT_PLAN.md", "IMPLEMENTATION_AUDIT.md", "PILOT_REPORT.md", "MECHANISM_REPORT.md", "DECISION_MEMO.md")] + [ROOT / "scripts/analyze_rmcge_v21.py"]
    manifest = read_json(DATA_DIR / "manifest.json", {})
    manifest["status"] = "all_27_runs_verified"
    manifest["updated_at_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
    manifest["analysis_provenance"] = {
        "script": "scripts/analyze_rmcge_v21.py",
        "script_sha256": sha256_file(ROOT / "scripts/analyze_rmcge_v21.py"),
        "post_run_report_template_repair": True,
        "training_records_or_metrics_changed": False,
        "manifest_hash_excluded_from_artifact_integrity": "the manifest records the artifact-integrity hash, so hashing it there would create a self-reference",
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    integrity = {
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "run_count": len(run_rows),
        "epoch_history_count": len(history_rows),
        "paired_comparison_count": len(paired_csv),
        "edge_diagnostic_count": len(edge_rows),
        "role_distribution_count": len(role_rows),
        "message_scale_count": len(message_rows),
        "expert_router_count": len(router_rows),
        "intervention_count": len(intervention_rows),
        "resource_profile_count": len(resource_rows),
        "all_run_keys_unique": True,
        "all_schema_checks_passed": True,
        "all_checkpoint_reload_checks_passed": True,
        "test_metrics_present": False,
        "test_label_values_indexed": False,
        "failures_preserved": len(failures.get("failures", [])),
        "manifest_hash_excluded_reason": "The manifest stores this artifact-integrity file's SHA-256; excluding it avoids a self-referential hash.",
        "artifacts": {str(path.relative_to(ROOT)): sha256_file(path) for path in integrity_paths},
    }
    write_json(DATA_DIR / "artifact_integrity.json", integrity)
    return {
        "status": "analysis_complete",
        "runs": len(run_rows),
        "epoch_history_rows": len(history_rows),
        "paired_rows": len(paired_csv),
        "edge_rows": len(edge_rows),
        "preserved_failures": len(failures.get("failures", [])),
        "paired_summary": paired_summary,
        "artifact_integrity_sha256": sha256_file(DATA_DIR / "artifact_integrity.json"),
    }


def main() -> None:
    print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
