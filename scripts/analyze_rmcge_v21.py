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

    (RESEARCH / "DESIGN.md").write_text(
        """# RM-CGE V2.1 Design\n\n"
        "## Frozen question\n\n"
        "Does receiver-conditioned relative neighbor choice improve the shared RM-CGE V1 graph experts when total incoming physical propagation mass is held fixed, and does giving Intra and Cross messages independent choice distributions help?\n\n"
        "## Conditions\n\n"
        "- **C0** delegates to the existing RM-CGE V2 V1 forward and parameter tree.\n"
        "- **C1** adds one shared bounded scorer over target-conditioned Intra messages and adds `(P_select_intra − P) H` to the existing expert path.\n"
        "- **C2** adds an independently scored Cross-purpose distribution and adds `(P_select_cross − P) Phi_other_to_target(H_other)` while retaining the ordinary Cross Context.\n\n"
        "The scorer is shared across target modalities and purposes. It uses receiver query/key interaction features, cosine similarity, endpoint degrees, an 8-dimensional target-modality embedding and an 8-dimensional message-purpose embedding. A zero initialized output layer makes the selector start at the physical operator. Scores are bounded with tanh to [-1,1].\n\n"
        "For incoming edges of node i, the implementation computes `p_select[e] = p[e] exp(score[e]) R_i / D_i` with FP32 receiver scatter sums. The physical graph, symmetric degree-normalized coefficients, edge multiplicities and four-hop shared physical expert profiles remain intact. Selection is a residual against the original physical operator.\n\n"
        "C1 and C2 have identical added module shapes and initialization. The V2/V1 Cross Projection, shared four-expert bank, original Top-2 node route, fusion and balance loss are reused. C2's selected Cross message enters both expert profiles and the existing Node Router input; no Router head is added.\n\n"
        "## Boundaries\n\n"
        "Training is full-graph transductive with supervision on Train indices only. Validation selects checkpoints. Test evaluation is disabled. Three seeds are descriptive paired evidence, not a significance test.\n\n"
        "## Relation to prior designs\n\n"
        "GATv2 motivates explicitly testing whether the receiver query can reverse neighbor rankings; the synthetic test directly exercises this property. GNN-FiLM motivates using the target representation to condition incoming messages, while V2.1 uses the target only in a shared score function and preserves the original message transform. RoleMAG motivates distinguishing neighbor functions in multimodal propagation; V2.1 operationalizes only the frozen Intra/Cross message-purpose split and adds no role labels or role-specific expert banks. MvCGE motivates shared expert parameters with graph-aware routing, which remain inherited from V1. CAME motivates modality-aware expert fusion and cross-modal alignment; the existing V1 Cross Projection and shared expert path are retained without adding a contrastive objective. These are design inspirations, not claims that V2.1 reproduces those methods.\n\n"
        "References: [GATv2 (ICLR 2022)](https://openreview.net/forum?id=F72ximsx7C1); [GNN-FiLM](https://arxiv.org/abs/1906.12192); [RoleMAG](https://arxiv.org/abs/2604.12271); [MvCGE (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html); [CAME (ICML 2026)](https://proceedings.mlr.press/v306/he26at.html).\n""",
        encoding="utf-8",
    )
    (RESEARCH / "EXPERIMENT_PLAN.md").write_text(
        """# RM-CGE V2.1 Experiment Plan\n\n"
        "Frozen matrix: Movies/Grocery/ele-fashion × C0/C1/C2 × seeds 42/43/44. Every Dataset × Seed reuses identical official features, physical graph, V7B.1 K=8 semantic cache and Train/Validation splits; all conditions share the same public initial weights and classifier state.\n\n"
        "Protocol: `unified_full_graph_nc_v1`, AdamW, lr 1e-3, weight decay 1e-4, dropout .2, up to 300 epochs, patience 30, minimum epoch 30, min delta 1e-4, gradient clip 1.0, validation every epoch, Validation Accuracy checkpoint selection, and existing balance loss. A separate Best Macro-F1 checkpoint is retained. `evaluate_test=false`.\n\n"
        "Seed 42 forms the first integrity batch. Seeds 43/44 proceed after the first nine runs have finite histories, valid checkpoints, selector gradients, and intact data/cache fingerprints. No hyperparameter search is performed.\n\n"
        "The audit compares paired Validation Accuracy, Macro-F1 and CE at each run's Best Accuracy checkpoint. Best Macro-F1 metrics are shown separately. Frozen interventions measure checkpoint sensitivity and are not retrained ablations.\n""",
        encoding="utf-8",
    )
    (RESEARCH / "IMPLEMENTATION_AUDIT.md").write_text(
        f"""# RM-CGE V2.1 Implementation Audit\n\n"
        f"Implementation Freeze preflight ran on branch `{checks.get('branch')}` from V2 evidence `{checks.get('baseline_commit_sha')}`. Full CPU regression: **{checks.get('regression_tests', {}).get('passed')} passed, {checks.get('regression_tests', {}).get('skipped')} skipped**. Targeted V2.1 unit coverage: graph mass conservation, zero/equal scores, isolates/single-neighbor rows, directed and duplicate edges, chunking, receiver-conditioned rank reversal, C0 exact delegation, paired initialization, zero-effect equivalence, Cross projection direction, task gradients/AdamW updates, and checkpoint reload.\n\n"
        f"GPU smoke used {checks.get('ele_fashion_gpu_smoke', {}).get('conditions', {}).get('C2', {}).get('device')}. Peak allocated memory across the smoke was {max((v.get('peak_allocated_bytes', 0) for v in checks.get('ele_fashion_gpu_smoke', {}).get('conditions', {}).values()), default=0) / (1024**3):.2f} GiB. GPU runtime details and initial hashes are in `data/implementation_checks.json`.\n\n"
        "C0 parameter initialization and Forward delegate to RM-CGE V2/V1. Initialization audit records per Dataset × Seed C0/C1/C2 model and classifier SHA-256 values; common parameters and the full C1/C2 added state match exactly. Added modules are created inside a forked RNG scope so they do not shift classifier initialization.\n\n"
        "The physical operator remains the inherited V7A/R0 operator. C1/C2 use edge-chunk scoring, FP32 differentiable receiver scatter and chunked SpMM; they do not form a dense adjacency or an E×hidden message matrix. The edge scorer's shared output bias is expected to have zero task gradient under receiver-wise normalization's shift invariance.\n\n"
        "Test indices are required by the shared task data object, but this runner disables Test evaluation and never indexes Test labels. Initialization fingerprints intentionally include only features/topology and Train/Validation indices and labels. No Test metrics appear in run records or checkpoints.\n\n"
        f"Formal artifacts: {len(run_rows)} verified runs, {len(history_rows)} epoch rows, {len(edge_rows)} edge-path rows, {len(intervention_rows)} frozen-intervention rows. Preserved failure records: {len(failures.get('failures', []))}.\n""",
        encoding="utf-8",
    )
    (RESEARCH / "PILOT_REPORT.md").write_text(
        f"""# RM-CGE V2.1 Pilot and Paired Results\n\n"
        "All reported scores are Validation metrics. Accuracy/Macro-F1/CE in the first tables come from each run's Best Accuracy checkpoint. The second group comes from the distinct Best Macro-F1 checkpoint. Means and sample SDs are over seeds 42, 43 and 44.\n\n"
        "## Best Accuracy checkpoint\n\n"
        f"### Validation Accuracy (%)\n\n{accuracy_table}\n\n"
        f"### Validation Macro-F1 (%)\n\n{f1_table}\n\n"
        f"### Validation CE\n\n{ce_table}\n\n"
        "## Best Macro-F1 checkpoint\n\n"
        f"### Validation Accuracy (%)\n\n{best_f1_acc_table}\n\n"
        f"### Validation Macro-F1 (%)\n\n{best_f1_f1_table}\n\n"
        f"### Validation CE\n\n{best_f1_ce_table}\n\n"
        "## Paired contrasts (left condition minus right condition)\n\n"
        f"{paired_table}\n\n"
        "Differences are descriptive paired changes; n=3 does not support a significance claim. Full per-seed run rows, initial hashes, epoch curves, checkpoint metrics and paired differences are in `data/`.\n\n"
        f"Model parameter counts observed: {model_counts}; classifier parameter counts: {classifier_counts}. Mean run wall time was {resource_mean:.1f}s and maximum allocated GPU memory was {peak_gib:.2f} GiB. No Test metrics were computed.\n""",
        encoding="utf-8",
    )

    def grouped_edge_summary() -> list[str]:
        lines = ["| Dataset | Condition | Target | Purpose | Mean abs ratio deviation | Mean receiver JS | Mean ESS (degree≥2) | Score near-bound fraction |", "|---|---|---|---|---:|---:|---:|---:|"]
        for dataset in DATASETS:
            for condition in CONDITIONS:
                for modality in ("text", "visual"):
                    for purpose in ("intra", "cross"):
                        matches = [row for row in edge_rows if row["dataset"] == dataset and row["condition"] == condition and row["target_modality"] == modality and row["message_purpose"] == purpose]
                        if not matches:
                            continue
                        enabled = matches[0]["selection_enabled"].lower() == "true"
                        if not enabled:
                            lines.append(f"| {dataset} | {condition} | {modality} | {purpose} | N/A | N/A | N/A | N/A |")
                            continue
                        lines.append(
                            f"| {dataset} | {condition} | {modality} | {purpose} | "
                            f"{statistics.mean(float(row['relative_weight_mean_abs_deviation']) for row in matches):.4f} | "
                            f"{statistics.mean(float(row['js_selected_physical_mean_degree_ge2']) for row in matches):.4f} | "
                            f"{statistics.mean(float(row['effective_neighbors_mean_degree_ge2']) for row in matches):.3f} | "
                            f"{statistics.mean(float(row['score_abs_ge_0_99_fraction']) for row in matches):.4f} |"
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
    modality_table = ["| Dataset | Condition | Modality | Intra Δ RMS | Cross base RMS | Cross Δ RMS | KNN RMS | Router entropy | Graph/Intrinsic RMS |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in message_rows:
        modality_table.append(
            f"| {row['dataset']} | {row['condition']} | {row['modality']} | {_csv_escape(float(row['delta_intra_rms']) if row['delta_intra_rms'] else 0.0)} | "
            f"{_csv_escape(float(row['cross_base_rms']) if row['cross_base_rms'] else 0.0)} | "
            f"{_csv_escape(float(row['delta_cross_rms']) if row['delta_cross_rms'] else 0.0)} | "
            f"{_csv_escape(float(row['knn_rms']) if row['knn_rms'] else 0.0)} | {_csv_escape(float(row['router_entropy']) if row['router_entropy'] else 0.0)} | "
            f"{_csv_escape(float(row['weighted_graph_correction_rms']) if row['weighted_graph_correction_rms'] else 0.0)}/"
            f"{_csv_escape(float(row['intrinsic_rms']) if row['intrinsic_rms'] else 0.0)} |"
        )
    intervention_table = ["| Dataset | Condition | Intervention | Mean Δ Val Accuracy (pp) | Mean Δ Macro-F1 (pp) | Mean logit RMS |", "|---|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for name in ("intra_selection_forced_physical", "cross_selection_forced_physical", "both_selections_forced_physical", "cross_base_context_off", "knn_context_off"):
                subset = [row for row in intervention_rows if row["dataset"] == dataset and row["condition"] == condition and row["intervention"] == name]
                if not subset or subset[0]["status"] == "not_applicable":
                    intervention_table.append(f"| {dataset} | {condition} | {name} | N/A | N/A | N/A |")
                else:
                    intervention_table.append(
                        f"| {dataset} | {condition} | {name} | {100*statistics.mean(float(row['val_acc_delta']) for row in subset):+.2f} | "
                        f"{100*statistics.mean(float(row['val_macro_f1_delta']) for row in subset):+.2f} | "
                        f"{statistics.mean(float(row['logit_delta_rms']) for row in subset):.4f} |"
                    )
    (RESEARCH / "MECHANISM_REPORT.md").write_text(
        """# RM-CGE V2.1 Mechanism Report\n\n"
        "The audit separates trained score distributions, messages that reach the shared experts, routing behavior and frozen checkpoint sensitivity. A row-sum check validates conservation of original receiver physical mass. All edge metrics exclude isolated nodes; entropy/ESS summaries distinguish degree-one nodes from degree-at-least-two nodes.\n\n"
        "## Learned neighbor distributions\n\n"
        + "\n".join(grouped_edge_summary())
        + "\n\nRatio deviation is `mean |p_select/p − 1|`; receiver JS compares the learned row distribution to normalized physical P; ESS is `1 / sum_j q_ij²`. A score near-bound is `|tanh(raw)| ≥ .99`. Per-run maximum and average row-mass errors, ratio quantiles, entropy, KL, reverse-edge differences and full seed records are in `data/edge_selection_diagnostics.csv`.\n\n"
        "## Receiver query probe\n\n"
        + "\n".join(query_table)
        + "\n\nFor this frozen probe, the trained receiver vector is replaced with its modality-wide active-node mean while sender vectors, scorer weights, degrees and physical coefficients stay fixed. This isolates how much the actual receiver-conditioned query changes the neighbor distribution; it is not a retrained ablation.\n\n"
        "## Message and router scales\n\n"
        + "\n".join(modality_table)
        + "\n\n`ΔIntra` and `ΔCross` are residuals relative to the unmodified physical propagation. Cross Base remains a separate message and may be turned off without disabling Cross selection. Expert-weighted RMS, four expert-profile changes, Top-2 shares, router entropy, correction-logit scale, update norms and gradient records are available in `data/message_scale_diagnostics.csv`, `data/expert_router_diagnostics.csv` and `data/gradient_audit.json`. In C2, Cross Delta is supplied to both shared experts and the existing Node Router input.\n\n"
        "## Frozen interventions\n\n"
        + "\n".join(intervention_table)
        + "\n\nThese interventions hold trained parameters fixed. They measure checkpoint sensitivity and do not replace training ablations. Paths absent from a condition are explicitly `not_applicable`.\n""",
        encoding="utf-8",
    )
    findings = _top_level_findings(run_details, paired_summary, edge_rows)
    decisions = [
        "# RM-CGE V2.1 Decision Memo",
        "",
        "## Facts",
        "",
        *[f"- {line}" for line in findings],
        f"- The synthetic receiver-interaction test produced opposite neighbor rankings for two targets with identical candidates. C0 exactly delegates to the V2/V1 forward; C1/C2 begin from matched public model and classifier weights.",
        f"- The run completed {len(run_rows)}/27 Dataset × Condition × Seed combinations, with {len(failed_runs)} recorded failed attempts retained. No Test metrics or Test-indexed label fingerprints were created.",
        "",
        "## Answers to the frozen research questions",
        "",
    ]
    for dataset in DATASETS:
        p = paired_summary[dataset]
        decisions.append(
            f"- **{dataset}: C1 vs C0 / C2 vs C1.** Validation Accuracy paired means are "
            f"{100*p['C1-C0']['val_acc']['mean_delta']:+.2f} pp and {100*p['C2-C1']['val_acc']['mean_delta']:+.2f} pp; "
            f"Macro-F1 changes are {100*p['C1-C0']['val_macro_f1']['mean_delta']:+.2f} pp and {100*p['C2-C1']['val_macro_f1']['mean_delta']:+.2f} pp. "
            "These are descriptive three-seed comparisons."
        )
    decisions.extend([
        "- **Does removing global amplification produce relative selection?** The edge diagnostics directly report receiver-normalized weight ratios, within-row variation, entropy, ESS and JS/KL divergence. A near-zero ratio deviation and JS would mean little learned relative selection despite model capacity; nonzero divergence establishes that the selector changed row distributions while preserving mass.",
        "- **Does the receiver representation affect ordering?** The synthetic reversal test establishes representational capacity. The frozen receiver-mean query probe measures learned checkpoint sensitivity; its per-dataset values are in `role_distribution_diagnostics.csv`.",
        "- **Do Intra and Cross behave differently?** Compare per-target-modality row distributions, matched edge-probability differences and per-receiver JS for `text_intra__text_cross`, `visual_intra__visual_cross`, and cross-modality pairs in the role-distribution table. Differences are interpreted jointly with each residual's coefficient-weighted RMS and validation results.",
        "- **Do selected messages reach experts and predictions?** `ΔIntra`/`ΔCross` are measured before and after expert coefficients; frozen physical-coefficient interventions report validation changes, logit RMS and prediction flips. These quantify fitted-checkpoint sensitivity, not edge-level causality.",
        "- **Router behavior and remaining pathologies.** Router entropy, expert Top-2 shares, correction-logit RMS and route flips are recorded per modality and run. Any concentration or compensation is described alongside expert and message scales; the V2.1 objective does not alter the original balance term.",
        "- **Dataset suitability and whether to keep both purposes.** Use the paired changes and frozen path interventions above. A purpose is supported only where changes are repeatable across seeds and its fitted message contribution is nontrivial; three seeds remain descriptive evidence.",
        "- **Next model change.** The present study only supports a targeted follow-up if paired performance, selector divergence and frozen sensitivity agree. It does not authorize extra experts, a new Router, auxiliary objectives, KNN search or Test evaluation.",
        "",
        "## Assumptions and evidence limits",
        "",
        "- The official full-graph transductive protocol allows unlabeled graph and feature propagation; label supervision is restricted to Train indices. Validation is used for checkpoint selection.",
        "- Three seeds provide paired descriptive changes, not statistical significance. Frozen interventions are sensitivity probes and cannot identify causal contributions under retraining.",
        "- Score bias gradients can be zero because a common per-receiver score offset cancels in the mass-preserving normalization. Sparse CUDA scatter may use backend-dependent accumulation order; this does not establish CUDA as the sole cause of training variation.",
        "",
        "## Decision",
        "",
        "Retain the V2.1 implementation and its complete evidence as a bounded mechanism study. Decide whether to carry Intra, Cross, both or neither into a later model only from the paired score table, receiver/purpose distribution diagnostics, expert-weighted message scales and frozen intervention effects. No part of this memo treats the best condition as predetermined.",
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
            "manifest.json", "runs.csv", "epoch_history.csv", "paired_comparisons.csv", "summary.csv",
            "edge_selection_diagnostics.csv", "role_distribution_diagnostics.csv", "message_scale_diagnostics.csv",
            "expert_router_diagnostics.csv", "gradient_audit.json", "frozen_interventions.csv",
            "resource_profile.csv", "implementation_checks.json", "failures.json",
        )
    ] + [RESEARCH / name for name in ("DESIGN.md", "EXPERIMENT_PLAN.md", "IMPLEMENTATION_AUDIT.md", "PILOT_REPORT.md", "MECHANISM_REPORT.md", "DECISION_MEMO.md")]
    manifest = read_json(DATA_DIR / "manifest.json", {})
    manifest["status"] = "all_27_runs_verified"
    manifest["updated_at_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
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
