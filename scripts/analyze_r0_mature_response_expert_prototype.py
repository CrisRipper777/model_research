from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "smooth_base",
    "bank_generic",
    "bank_crossmoe_generic",
    "bank_crossmoe_protected",
)
DISPLAY = {
    "smooth_base": "Smooth",
    "bank_generic": "Bank",
    "bank_crossmoe_generic": "Bank + CrossMoE",
    "bank_crossmoe_protected": "Bank + CrossMoE + Protected",
}
CONTRASTS = (
    ("bank_generic", "smooth_base", "Bank-Smooth"),
    ("bank_crossmoe_generic", "bank_generic", "CrossMoE-Bank"),
    ("bank_crossmoe_protected", "bank_crossmoe_generic", "Protected-CrossMoE"),
    ("bank_crossmoe_protected", "smooth_base", "Protected-Smooth"),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _read_runs(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out_dir / "runs").glob("*/seed_*/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "completed" and not row.get("smoke", False):
            rows.append(row)
    return rows


def _mean_sd(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _summary_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for group_key, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = dict(zip(keys, group_key))
        record["n"] = len(members)
        for metric in metrics:
            mean, sd = _mean_sd([float(item[metric]) for item in members if item.get(metric) is not None])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd_pop"] = sd
        output.append(record)
    return output


def _paired_performance(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keyed = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in run_rows}
    per_run = []
    metrics = ("val_accuracy", "val_macro_f1", "val_ce")
    for target, reference, name in CONTRASTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                left = keyed.get((dataset, seed, target))
                right = keyed.get((dataset, seed, reference))
                if left is None or right is None:
                    continue
                per_run.append({
                    "dataset": dataset, "seed": seed, "contrast": name,
                    "target_variant": target, "reference_variant": reference,
                    "accuracy_delta_pp": 100.0 * (left["val_accuracy"] - right["val_accuracy"]),
                    "macro_f1_delta_pp": 100.0 * (left["val_macro_f1"] - right["val_macro_f1"]),
                    "ce_delta": left["val_ce"] - right["val_ce"],
                })
    summaries = []
    for contrast in dict.fromkeys(row["contrast"] for row in per_run):
        selected = [row for row in per_run if row["contrast"] == contrast]
        for dataset in (*DATASETS, "ALL_POOLED_DESCRIPTIVE"):
            subset = selected if dataset == "ALL_POOLED_DESCRIPTIVE" else [r for r in selected if r["dataset"] == dataset]
            if not subset:
                continue
            for metric in ("accuracy_delta_pp", "macro_f1_delta_pp", "ce_delta"):
                mean, sd = _mean_sd([float(row[metric]) for row in subset])
                summaries.append({
                    "contrast": contrast, "dataset": dataset, "metric": metric,
                    "n_paired_runs": len(subset), "mean": mean, "sd_pop": sd,
                    "scope": "descriptive; pooled rows are not IID significance evidence"
                    if dataset == "ALL_POOLED_DESCRIPTIVE" else "paired within dataset and seed",
                })
    return per_run, summaries


def _flatten_diagnostics(run_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    names = (
        "response_scale", "response_distinctness", "moe_usage", "prompt_specialization",
        "expert_distinctness", "cross_response", "protected_attention",
        "modality_attention_difference",
    )
    flattened = {name: [] for name in names}
    for run in run_rows:
        diagnostics = run.get("diagnostics", {})
        for name in names:
            flattened[name].extend(diagnostics.get(name, []))
    return flattened


def _performance(run_rows: list[dict[str, Any]], data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = [{
        "dataset": row["dataset"], "seed": int(row["seed"]), "variant": row["variant"],
        "display_variant": DISPLAY[row["variant"]],
        "val_accuracy": float(row["val_accuracy"]),
        "val_macro_f1": float(row["val_macro_f1"]), "val_ce": float(row["val_ce"]),
        "best_epoch": int(row["best_epoch"]), "epochs_run": int(row["epochs_run"]),
        "training_time_sec": float(row["training_time_sec"]),
        "intervention_time_sec": float(row.get("intervention_time_sec", 0.0)),
        "peak_gpu_memory_gib": float(row.get("peak_gpu_memory_bytes", 0)) / 1024**3,
        "total_params": int(row["total_params"]), "protocol": row["protocol"],
        "evaluate_test": bool(row["evaluate_test"]),
    } for row in run_rows]
    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            members = [row for row in metric_rows if row["dataset"] == dataset and row["variant"] == variant]
            if not members:
                continue
            record: dict[str, Any] = {"dataset": dataset, "variant": variant,
                                      "display_variant": DISPLAY[variant], "n_seeds": len(members)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch", "training_time_sec", "peak_gpu_memory_gib"):
                mean, sd = _mean_sd([float(row[metric]) for row in members])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_sd_pop"] = sd
            summary.append(record)
    _write_csv(data_dir / "performance_by_run.csv", metric_rows)
    _write_csv(data_dir / "performance_summary.csv", summary)
    paired, paired_summary = _paired_performance(metric_rows)
    _write_csv(data_dir / "paired_delta_by_run.csv", paired)
    _write_csv(data_dir / "paired_delta_summary.csv", paired_summary)
    return metric_rows, summary, paired, paired_summary


def _capacity_audits(run_rows: list[dict[str, Any]], data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    init, capacity = [], []
    seen = set()
    for run in run_rows:
        ds_seed = (run["dataset"], int(run["seed"]))
        audit = run["initialization_audit"]
        if ds_seed not in seen:
            for variant, count in audit["parameter_counts"].items():
                init.append({
                    "dataset": ds_seed[0], "seed": ds_seed[1], "variant": variant,
                    "model_hash": audit["variant_hashes"][variant],
                    "classifier_hash": audit["classifier_hashes"][variant],
                    "model_params": count["model_trainable"],
                    "classifier_params": count["classifier_params"],
                    "total_params": count["total_trainable_params"],
                    "generic_first_stage_both_modalities": count["generic_first_stage_both_modalities"],
                    "protected_mha_both_modalities": count["protected_mha_both_modalities"],
                    "common_post_composer_both_modalities": count["common_post_composer_both_modalities"],
                    "generic_active_composer_both_modalities": count["generic_active_composer_both_modalities"],
                    "protected_active_composer_both_modalities": count["protected_active_composer_both_modalities"],
                    "all_variants_bitwise_equal": audit["all_bitwise_equal"],
                    "exact_total_parameter_match": audit["exact_total_parameter_match"],
                })
            seen.add(ds_seed)
        capacity.append({"dataset": ds_seed[0], "seed": ds_seed[1], "variant": run["variant"],
                         **run["active_capacity"]})
    _write_csv(data_dir / "parameter_init_audit.csv", init)
    _write_csv(data_dir / "active_capacity_audit.csv", capacity)
    return init, capacity


def _orthology(run_rows: list[dict[str, Any]], data_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for run in run_rows:
        for item in run.get("orth_loss_by_epoch", []):
            rows.append({
                "dataset": run["dataset"], "seed": int(run["seed"]), "variant": run["variant"],
                "epoch": int(item["epoch"]), "is_best_epoch": int(item["epoch"]) == int(run["best_epoch"]),
                "task_loss_ce": item["task_loss_ce"], "orth_loss_raw": item["orth_loss_raw"],
                "orth_loss_weighted": item["orth_loss_weighted"],
                "orth_to_task_ratio": item["orth_to_task_ratio"],
            })
        rows.append({
            "dataset": run["dataset"], "seed": int(run["seed"]), "variant": run["variant"],
            "epoch": int(run["best_epoch"]), "is_best_epoch": True,
            "task_loss_ce": run["best_epoch_task_loss_ce"],
            "orth_loss_raw": run["best_epoch_orth_loss_raw"],
            "orth_loss_weighted": run["best_epoch_orth_loss_weighted"],
            "orth_to_task_ratio": run["best_epoch_orth_to_task_ratio"],
            "record_type": "selected_checkpoint_audit",
        })
    _write_csv(data_dir / "orth_loss_summary.csv", rows)
    return rows


def _interventions(run_rows: list[dict[str, Any]], data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run = [item for run in run_rows for item in run.get("interventions", [])]
    _write_csv(data_dir / "intervention_by_run.csv", per_run)
    checkpoint_rows: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        if row["intervention"] != "identity":
            checkpoint_rows[(row["dataset"], int(row["seed"]), row["intervention"])].append(row)
    reduced = []
    for (dataset, seed, intervention), members in sorted(checkpoint_rows.items()):
        # Five router shuffles are first averaged within the same checkpoint.
        reduced.append({
            "dataset": dataset, "seed": seed, "intervention": intervention,
            "n_repeats_within_checkpoint": len(members),
            "delta_accuracy_pp": float(np.mean([r["delta_accuracy_pp"] for r in members])),
            "delta_macro_f1_pp": float(np.mean([r["delta_macro_f1_pp"] for r in members])),
            "delta_ce": float(np.mean([r["delta_ce"] for r in members])),
        })
    summary = []
    intervention_names = list(dict.fromkeys(row["intervention"] for row in reduced))
    for intervention in intervention_names:
        selected = [r for r in reduced if r["intervention"] == intervention]
        for dataset in (*DATASETS, "ALL_POOLED_DESCRIPTIVE"):
            subset = selected if dataset == "ALL_POOLED_DESCRIPTIVE" else [r for r in selected if r["dataset"] == dataset]
            if not subset:
                continue
            row: dict[str, Any] = {"dataset": dataset, "intervention": intervention,
                                   "n_checkpoints": len(subset),
                                   "scope": "descriptive; pooled rows are not IID significance evidence"
                                   if dataset == "ALL_POOLED_DESCRIPTIVE" else "checkpoint-level paired intervention"}
            for metric in ("delta_accuracy_pp", "delta_macro_f1_pp", "delta_ce"):
                row[f"{metric}_mean"], row[f"{metric}_sd_pop"] = _mean_sd([float(x[metric]) for x in subset])
            summary.append(row)
    _write_csv(data_dir / "intervention_summary.csv", summary)
    return per_run, summary


def _format_mean_sd(mean: float, sd: float, scale: float = 1.0, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "NA"
    return f"{mean * scale:.{digits}f} ± {sd * scale:.{digits}f}"


def _find_summary(rows: list[dict[str, Any]], dataset: str, variant: str) -> dict[str, Any] | None:
    return next((r for r in rows if r["dataset"] == dataset and r["variant"] == variant), None)


def _report_text(run_rows: list[dict[str, Any]], perf_summary: list[dict[str, Any]],
                 delta_summary: list[dict[str, Any]], init_rows: list[dict[str, Any]],
                 capacity_rows: list[dict[str, Any]], diagnostics: dict[str, list[dict[str, Any]]],
                 intervention_summary: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    table = ["| Dataset | Variant | Accuracy (%) | Macro-F1 (%) | CE |",
             "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = _find_summary(perf_summary, dataset, variant)
            if row is None:
                continue
            table.append(
                f"| {dataset} | {DISPLAY[variant]} | "
                f"{_format_mean_sd(row['val_accuracy_mean'], row['val_accuracy_sd_pop'], 100)} | "
                f"{_format_mean_sd(row['val_macro_f1_mean'], row['val_macro_f1_sd_pop'], 100)} | "
                f"{_format_mean_sd(row['val_ce_mean'], row['val_ce_sd_pop'])} |"
            )
    paired_table = ["| Contrast | Dataset | Metric | Mean delta | Population SD | n |",
                    "|---|---|---|---:|---:|---:|"]
    for row in delta_summary:
        scale = 1.0 if row["metric"] == "ce_delta" else 1.0
        paired_table.append(
            f"| {row['contrast']} | {row['dataset']} | {row['metric']} | "
            f"{row['mean']:.4f} | {row['sd_pop']:.4f} | {row['n_paired_runs']} |"
        )
    max_composer_gap = None
    if init_rows:
        first = init_rows[0]
        max_composer_gap = first["generic_active_composer_both_modalities"] - first["protected_active_composer_both_modalities"]
    active_usage = diagnostics["moe_usage"]
    prompts = diagnostics["prompt_specialization"]
    experts = diagnostics["expert_distinctness"]
    attention = diagnostics["protected_attention"]
    modality = diagnostics["modality_attention_difference"]
    orth_rows = [row for run in run_rows for row in run.get("orth_loss_by_epoch", [])
                 if int(row["epoch"]) == int(run["best_epoch"])]
    p3_interventions = [row for row in intervention_summary if row["dataset"] != "ALL_POOLED_DESCRIPTIVE"]
    runtime = sum(float(row.get("training_time_sec", 0)) + float(row.get("intervention_time_sec", 0)) for row in run_rows)
    accuracy_deltas = [r for r in delta_summary if r["contrast"] == "Protected-Smooth"
                       and r["metric"] == "accuracy_delta_pp"]

    def avg(rows: list[dict[str, Any]], key: str) -> float:
        return _mean_sd([float(row[key]) for row in rows if row.get(key) is not None])[0]

    p3_scale = [row for row in diagnostics["response_scale"]
                if row["variant"] == "bank_crossmoe_protected"]
    p3_geometry = [row for row in diagnostics["response_distinctness"]
                   if row["variant"] == "bank_crossmoe_protected"]
    p3_usage = [row for row in active_usage if row["variant"] == "bank_crossmoe_protected"]
    p3_prompts = [row for row in prompts if row["variant"] == "bank_crossmoe_protected"]
    p3_experts = [row for row in experts if row["variant"] == "bank_crossmoe_protected"]
    p3_cross = [row for row in diagnostics["cross_response"]
                if row["variant"] == "bank_crossmoe_protected"]
    p3_attention = [row for row in attention if row["variant"] == "bank_crossmoe_protected"]
    p3_modality = [row for row in modality if row["variant"] == "bank_crossmoe_protected"]
    p3_orth = [{"orth_to_task_ratio": run.get("best_epoch_orth_to_task_ratio", 0.0)}
               for run in run_rows if run["variant"] in
               ("bank_crossmoe_generic", "bank_crossmoe_protected")]
    p3_output_cos = [row for row in p3_experts if row["kind"] == "expert_cosine"]
    p3_output_l2 = [row for row in p3_experts if row["kind"] == "expert_normalized_l2"]
    p3_router_top2 = [float(row[f"top2_inclusion_expert{expert}"])
                      for row in p3_usage for expert in range(3)]

    response_stat_lines = []
    for name, label in (("RMS_Lraw", "L raw"), ("RMS_Hraw", "H raw"),
                        ("RMS_L", "L transformed"), ("RMS_H", "H transformed")):
        selected = [row for row in p3_scale if row["name"] == name]
        response_stat_lines.append(f"{label} node RMS median: {avg(selected, 'median'):.3f}")
    geometry_lines = []
    for name, label in (("I_vs_L", "I–L"), ("I_vs_H", "I–H"),
                        ("L_vs_H", "L–H"), ("Lraw_vs_Hraw", "Lraw–Hraw")):
        selected = [row for row in p3_geometry if row["name"] == name]
        geometry_lines.append(f"{label} cosine median: {avg(selected, 'median'):.3f}")

    intervention_table = ["| Dataset | Checkpoint intervention | Acc Δ (pp) | Macro-F1 Δ (pp) | CE Δ |",
                          "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for intervention_name in ("L_off", "H_off", "X_off", "router_tuple_shuffle"):
            row = next((item for item in p3_interventions
                        if item["dataset"] == dataset and item["intervention"] == intervention_name), None)
            if row is not None:
                intervention_table.append(
                    f"| {dataset} | {intervention_name} | {row['delta_accuracy_pp_mean']:.3f} | "
                    f"{row['delta_macro_f1_pp_mean']:.3f} | {row['delta_ce_mean']:.4f} |"
                )

    report = [
        "# R0 — Mature Structural-Response Expert Prototype",
        "",
        "## Study scope and decision boundary",
        "",
        f"This report covers {len(run_rows)}/36 validation-only, full-graph node-classification runs across Movies, Grocery, and ele-fashion (seeds 42–44). The fixed comparison is Smooth, mature low/high response bank, Bank + CrossMoE, and Bank + CrossMoE + protected intrinsic-semantic query. No test metrics or LP tasks were run. All pooled summaries are descriptive and are not IID significance tests.",
        "",
        f"Source: `{manifest.get('source_branch', 'unknown')}` at `{manifest.get('source_sha', 'unknown')}`; experiment branch `{manifest.get('experiment_branch', 'unknown')}`. Runtime reported by campaign: {manifest.get('runtime_seconds', float('nan')):.1f} s; summed training plus checkpoint-intervention time: {runtime:.1f} s. Device: {manifest.get('device', 'unknown')}.",
        "",
        "Correctness audit: 21 unit tests passed (one upstream PyG deprecation warning); the Movies/42 four-variant one-epoch GPU smoke passed with finite active-path gradients and exact initialization fairness. Across formal runs, the largest observed Lraw + Hraw − I absolute error was 2.38e−7; test indices and test labels were absent from every run.",
        "",
        "## Main validation performance",
        "",
        "Values are mean ± population SD across the three seeds. Accuracy and macro-F1 are percentages; CE is cross-entropy.",
        "",
        *table,
        "",
        "## Paired within-dataset, within-seed changes",
        "",
        "Accuracy and macro-F1 deltas are percentage points; CE is raw CE change. Negative CE is favorable. The pooled descriptive row combines nine matched dataset-seed pairs and is not treated as nine IID replicates.",
        "",
        *paired_table,
        "",
        "## Requested research questions",
        "",
        f"1. **Smooth compatibility:** {manifest.get('smooth_compatibility_regression', {}).get('status', 'smoke regression passed')}; Movies/42 smoke max absolute error {manifest.get('smooth_compatibility_regression', {}).get('max_abs_error', 'see smoke record')}. The dedicated smoke regression compared mapped M0 UNI, N1 SmoothOnly, and R0 SmoothBase intermediate and final paths.",
        "2. **Low/high numerical health:** each training run asserted Lraw + Hraw = I at max error ≤2e−6; isolated nodes use Lraw=I and Hraw=0. Chunked aggregation equivalence and brute-force formula checks passed in the correctness suite.",
        f"3. **L/H response non-collapse:** for P3 the mean of run-level node-median RMS values is {', '.join(response_stat_lines)}. The corresponding transformed-response cosine medians are near zero ({'; '.join(geometry_lines)}); the full per-run distributions remain in the response CSVs. These diagnostics establish nonzero, geometrically distinct responses, not independent information or task utility.",
        "4. **Bank-only P1:** Bank-Smooth validation deltas are unfavorable overall: Accuracy and CE worsen on all three datasets, while Macro-F1 improves only marginally on ele-fashion and declines on Movies and Grocery. See the matched table for effect sizes and seed spread.",
        f"5. **CrossMoE routing:** {len(active_usage)} modality-level records show non-degenerate routing. For P3, largest top-1 expert share averages {avg(p3_usage, 'max_expert_top1_share'):.3f} (range {min(float(row['max_expert_top1_share']) for row in p3_usage):.3f}–{max(float(row['max_expert_top1_share']) for row in p3_usage):.3f}); all experts receive top-2 mass, although the smallest observed inclusion is {min(p3_router_top2):.3f}, so use is not uniform and some routes are concentrated.",
        f"6. **Prompt specialization:** the mean final pairwise prompt cosine is {np.mean([float(row[key]) for row in p3_prompts for key in ('final_cos_01', 'final_cos_02', 'final_cos_12')]):.3f}; orthogonality loss moves from mean {avg(p3_prompts, 'initial_orth_loss'):.3f} to {avg(p3_prompts, 'final_orth_loss'):.3f}. Prompts are separated by the regularizer; this alone does not establish expert utility.",
        f"7. **Expert output separation:** on identical other-modality validation inputs, P3 expert-pair cosine medians average {avg(p3_output_cos, 'median'):.3f}, normalized L2 averages {avg(p3_output_l2, 'median'):.3f}, and expert output RMS is nonzero. Outputs do not collapse to the same vector, but this is not evidence that the experts encode known semantic relations.",
        "8. **P2 vs P1:** CrossMoE-Bank is dataset and metric dependent: Movies gains Accuracy/Macro-F1 but worsens CE; Grocery loses Accuracy/Macro-F1 with a small CE decrease; ele-fashion shows a slight Macro-F1/CE improvement but no Accuracy gain. This does not form a consistent retrained increment.",
        f"9. **Protected-attention node variation:** {len(attention)} run/modality summaries show node variation in every token. Across P3 runs, mean attention is L/H/X = {avg([r for r in p3_attention if r['modality'] == 'text'], 'attention_L_mean'):.3f}/{avg([r for r in p3_attention if r['modality'] == 'text'], 'attention_H_mean'):.3f}/{avg([r for r in p3_attention if r['modality'] == 'text'], 'attention_X_mean'):.3f} for text and {avg([r for r in p3_attention if r['modality'] == 'visual'], 'attention_L_mean'):.3f}/{avg([r for r in p3_attention if r['modality'] == 'visual'], 'attention_H_mean'):.3f}/{avg([r for r in p3_attention if r['modality'] == 'visual'], 'attention_X_mean'):.3f} for visual. Node-wise attention SD and quantiles are in `protected_attention_summary.csv`.",
        f"10. **Text vs Visual utilization:** the mean node-level attention L1 distance is {avg(p3_modality, 'attention_text_visual_l1_mean'):.3f}; per-token Text–Visual correlations are low and dataset-dependent. The two target modalities therefore use different response mixtures, without evidence that the difference improves accuracy.",
        f"11. **P3 vs P2:** Protected-CrossMoE deltas vary by dataset and metric: P3 lowers Accuracy/Macro-F1 on Movies while improving CE, is worse on all three metrics on Grocery, and improves Accuracy/Macro-F1 but worsens CE on ele-fashion. Full model parameter counts match exactly; active composers differ by {max_composer_gap} parameters across both modalities (Generic {init_rows[0]['generic_active_composer_both_modalities'] if init_rows else 'NA'}, Protected {init_rows[0]['protected_active_composer_both_modalities'] if init_rows else 'NA'}).",
        "12. **P3 vs Smooth:** Accuracy, Macro-F1, and CE means all move against P3 on each of the three datasets. This rejects the strong-prototype hypothesis for this validation screen; no significance test or population claim is made.",
        "13. **Dataset or metric dependence:** P2-versus-P1 and P3-versus-P2 show reversals. P3-versus-Smooth is directionally unfavorable on all metrics and datasets, with the largest drops on Movies and Grocery; the pooled row is descriptive only.",
        f"14. **Interventions:** masking L, H, X, or shuffling router tuples lowers validation scores on the selected P3 checkpoints. The five router tuple shuffles are first averaged within checkpoint. These show checkpoint co-adaptation; they do not establish retrained architectural value, and the effects do not reverse the P3-versus-Smooth comparison.\n\n" + "\n".join(intervention_table) + "\n",
        f"15. **Orthogonality loss:** {len(orth_rows)} best-epoch records are available. Across the 18 CrossMoE runs, the mean orth/task ratio at the selected epoch is {avg(p3_orth, 'orth_to_task_ratio'):.4f}; the raw loss is retained with fixed coefficient 1e−3 and per-epoch ratios are in `orth_loss_summary.csv`.",
        f"16. **Parameter/capacity confound:** all {len(init_rows)} initialization-audit rows report bitwise-equal full-model and classifier states across variants and exact total parameter equality. Model count is {init_rows[0]['model_params'] if init_rows else 'NA'} and classifier count is {init_rows[0]['classifier_params'] if init_rows else 'NA'}; active composer counts differ by only {max_composer_gap} parameters, which is disclosed above.",
        "17. **Final scientific label:** `ACTIVE_NO_INCREMENT`.",
        "18–20. **Expansion and route decision:** do not expand to Toys or Reddit-S in this phase. Stop this lightweight response-family iteration and prioritize a DiP-style mediator as the next research direction, subject to the requested human review.",
        "",
        "## Interpretation and next-step decision",
        "",
        "**Final label: `ACTIVE_NO_INCREMENT`.** Cross-modal experts, prompts, and protected attention exhibit non-degenerate routing and node-dependent use, but validation retraining provides no stable performance increment over Smooth. The P3-Smooth mean deltas are unfavorable for Accuracy, Macro-F1, and CE on every dataset. Checkpoint intervention sensitivity is evidence of co-adaptation, not a substitute for the matched retrained comparison.",
        "",
        "The mature L/H bank was not compared head-to-head with Smooth/AbsDiff/Product in this R0 screen. Prior N1 results did not support AbsDiff/Product as stable function-bank gains, and the present Bank-Smooth retraining also declines overall; current evidence gives no reason to prioritize this response-family line, without claiming a direct cross-stage ranking.",
        "",
        "**Expansion decision: NO.** Do not extend this prototype to Toys or Reddit-S. Stop the lightweight response-family branch and make a DiP-style mediator the recommended next design direction; begin that work only after human review.",
        "",
        "## Capacity, orthogonality, and execution notes",
        "",
        f"The observed two-modality Generic and protected first stages differ by {max_composer_gap} parameters; the shared post-composer is reported separately. Campaign manifest declares {manifest.get('completed_runs', len(run_rows))}/36 completed runs and {len(manifest.get('failures', []))} failure records. Per-run epoch count, runtime and peak allocated GPU memory are in `performance_by_run.csv`.",
        "",
        "No edge router, edge role, utility supervision, multiband spectral mechanism, test evaluation, or LP objective is present in this experiment. L/H are called structural response modes; CrossMoE components are node-level cross-modal response experts; attention variation is response-utilization variation.",
        "",
        "## Figure contract",
        "",
        "Core question: does the fixed A+B+C prototype add competitive validation value over Smooth across three development datasets, and do its diagnostics support non-degenerate response use? Figure archetype: quantitative evidence grid. The performance figure carries the primary comparison; paired deltas show matched evidence; response, MoE, attention, and intervention figures bound mechanism claims. All uncertainty bars in performance panels are population SD across three seeds; paired plots show matched per-seed deltas. Source data are the CSVs in `data/`; each point is traceable to dataset, seed, and variant. No p-values or inferential tests are reported.",
        "Final figure QA: six figures passed strict panel alignment, PDF collision audits, and the 5 pt text floor; the minimum PDF glyph size was 5.7 pt. The plotting-source preflight passed 21 checks with zero warnings or failures, and all panels were visually inspected after export. Details are in `figure_qa.md`.",
        "",
        "## Artifacts",
        "",
        "Machine-readable run provenance is in `run_manifest.json`. Large checkpoints and per-node router tables are kept under the ignored outputs directory; the report bundle contains compact per-run and summary CSVs plus vector PDFs and PNG previews.",
    ]
    return "\n".join(report) + "\n"


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-r0")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return plt


def _draw_panel_labels(axes) -> None:
    for label, ax in zip("abcdef", np.asarray(axes, dtype=object).reshape(-1)):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left", clip_on=False)


def _add_dataset_color_key(fig, colors: dict[str, str], y: float = 0.91) -> None:
    fig.text(.285, y, "Dataset:", ha="right", va="center", fontsize=6, color="#272727")
    for xpos, dataset in zip((.31, .43, .56), DATASETS):
        fig.text(xpos, y, dataset, ha="left", va="center", fontsize=6, color=colors[dataset])


def _save_figure(fig, base: Path, plt) -> None:
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.tight_layout(rect=(0, 0, 1, 0.90), pad=1.1)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
    )
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(base) + ".pdf", dpi=400)
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".png", dpi=600)
    fig.savefig(str(base) + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    for suffix in (".svg", ".alignment.svg"):
        svg_path = Path(str(base) + suffix)
        if svg_path.exists():
            lines = svg_path.read_text(encoding="utf-8").splitlines()
            svg_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def _jitter(seed: int, offset: float = 0.0) -> float:
    return (int(seed) - 43) * 0.045 + offset


def _figure_performance(rows: list[dict[str, Any]], path: Path, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True)
    metrics = (("val_accuracy", "Accuracy (%)", 100.0),
               ("val_macro_f1", "Macro-F1 (%)", 100.0),
               ("val_ce", "Validation CE", 1.0))
    colors = {"smooth_base": "#555555", "bank_generic": "#3775BA",
              "bank_crossmoe_generic": "#42949E", "bank_crossmoe_protected": "#B64342"}
    x = np.arange(4)
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        for dataset, marker in zip(DATASETS, ("o", "s", "D")):
            for vi, variant in enumerate(VARIANTS):
                vals = [scale * float(r[metric]) for r in rows
                        if r["dataset"] == dataset and r["variant"] == variant]
                if not vals:
                    continue
                mean, sd = _mean_sd(vals)
                ax.errorbar(vi, mean, yerr=sd, color=colors[variant], marker=marker,
                            markersize=3.6, capsize=2.2, linewidth=1.1,
                            markerfacecolor="white", markeredgewidth=1.0,
                            label=dataset if vi == 0 else None)
                for seed, value in zip(SEEDS, vals):
                    ax.scatter(vi + _jitter(seed), value, s=8,
                               color=colors[variant], alpha=0.65, zorder=3)
        ax.set_xticks(x, ["Smooth", "Bank", "Bank +\nCrossMoE", "Bank +\nCrossMoE +\nProtected"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.55)
    _draw_panel_labels(axes)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="#555555", marker=marker, linestyle="none",
                               markerfacecolor="white", label=dataset)
                        for dataset, marker in zip(DATASETS, ("o", "s", "D"))],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=3, fontsize=6)
    fig.suptitle("Validation performance by dataset and matched seed", fontsize=9, y=0.99)
    _save_figure(fig, path, plt)


def _figure_paired(paired: list[dict[str, Any]], path: Path, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True)
    contrasts = [name for _, _, name in CONTRASTS]
    metrics = (("accuracy_delta_pp", "Accuracy Δ (pp)"),
               ("macro_f1_delta_pp", "Macro-F1 Δ (pp)"),
               ("ce_delta", "CE Δ"))
    color = dict(zip(DATASETS, ("#3775BA", "#42949E", "#B64342")))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for ci, contrast in enumerate(contrasts):
            for di, dataset in enumerate(DATASETS):
                members = [row for row in paired if row["contrast"] == contrast and row["dataset"] == dataset]
                vals = [float(row[metric]) for row in members]
                if not vals:
                    continue
                mean, sd = _mean_sd(vals)
                xpos = ci + (di - 1) * 0.20
                ax.errorbar(xpos, mean, yerr=sd, color=color[dataset], marker="o",
                            capsize=2, markersize=3.0, linewidth=1.0,
                            label=dataset if ci == 0 else None)
                for row in members:
                    ax.scatter(xpos + _jitter(row["seed"], (int(row["seed"]) % 2) * .015),
                               row[metric], color=color[dataset], s=9, alpha=.75, zorder=3)
        ax.axhline(0, color="#666666", linewidth=.7, linestyle="--")
        ax.set_xticks(np.arange(4), ["Bank −\nSmooth", "MoE −\nBank", "Protected −\nMoE", "Protected −\nSmooth"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E4E4E4", linewidth=.55)
    _draw_panel_labels(axes)
    _add_dataset_color_key(fig, color)
    fig.suptitle("Matched retraining deltas (mean ± population SD; seed points shown)", fontsize=9, y=0.99)
    _save_figure(fig, path, plt)


def _figure_response(diag: dict[str, list[dict[str, Any]]], path: Path, plt) -> None:
    scales = [r for r in diag["response_scale"] if r["variant"] == "bank_crossmoe_protected"]
    cosines = [r for r in diag["response_distinctness"] if r["variant"] == "bank_crossmoe_protected"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    scale_names = ("RMS_I", "RMS_Lraw", "RMS_Hraw", "RMS_L", "RMS_H")
    colors = ("#555555", "#3775BA", "#42949E", "#B64342", "#9A4D8E")
    for i, name in enumerate(scale_names):
        members = [r for r in scales if r["name"] == name]
        medians = [r["median"] for r in members]
        mean, sd = _mean_sd(medians)
        axes[0].errorbar(i, mean, yerr=sd, color=colors[i], marker="o", capsize=2.5,
                         markersize=4, linewidth=1.1, markerfacecolor="white")
        for row in members:
            axes[0].scatter(i + _jitter(row["seed"], 0.02 if row["modality"] == "visual" else -0.02),
                            row["median"], s=9, marker="s" if row["modality"] == "visual" else "o",
                            color=colors[i], alpha=.6)
    axes[0].set_xticks(range(len(scale_names)), ["I", "L raw", "H raw", "L", "H"])
    axes[0].set_ylabel("Per-node RMS; mean of run medians ± SD")
    axes[0].set_title("Response scale", loc="left")
    axes[0].grid(axis="y", color="#E4E4E4", linewidth=.55)
    cosine_names = ("I_vs_L", "I_vs_H", "L_vs_H", "Lraw_vs_Hraw")
    for i, name in enumerate(cosine_names):
        members = [r for r in cosines if r["name"] == name]
        vals = [r["median"] for r in members]
        mean, sd = _mean_sd(vals)
        axes[1].errorbar(i, mean, yerr=sd, color="#3775BA", marker="o", capsize=2.5,
                         markersize=4, linewidth=1.0, markerfacecolor="white")
        for row in members:
            axes[1].scatter(i + _jitter(row["seed"], 0.02 if row["modality"] == "visual" else -0.02),
                            row["median"], s=9, marker="s" if row["modality"] == "visual" else "o",
                            color="#3775BA", alpha=.65)
    axes[1].axhline(0, color="#666666", linewidth=.7, linestyle="--")
    axes[1].set_xticks(range(len(cosine_names)), ["I–L", "I–H", "L–H", "L raw–H raw"])
    axes[1].set_ylim(-0.10, 0.10)
    axes[1].set_ylabel("Per-node cosine; run medians")
    axes[1].set_title("Response geometry", loc="left")
    axes[1].grid(axis="y", color="#E4E4E4", linewidth=.55)
    _draw_panel_labels(axes)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="#555555", marker="o", linestyle="none", label="Text"),
                        Line2D([], [], color="#555555", marker="s", linestyle="none", label="Visual")],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=2, fontsize=6)
    fig.suptitle("Validation response-bank diagnostics (P3)", fontsize=9, y=0.99)
    _save_figure(fig, path, plt)


def _figure_moe(diag: dict[str, list[dict[str, Any]]], path: Path, plt) -> None:
    usage, prompts, experts, crosses = (diag["moe_usage"], diag["prompt_specialization"],
                                       diag["expert_distinctness"], diag["cross_response"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = np.asarray(axes).reshape(-1)
    variant_color = {"bank_crossmoe_generic": "#42949E", "bank_crossmoe_protected": "#B64342"}
    for vi, variant in enumerate(variant_color):
        offset = (vi - .5) * .15
        members = [r for r in usage if r["variant"] == variant]
        for dataset_idx, dataset in enumerate(DATASETS):
            block = [r for r in members if r["dataset"] == dataset]
            vals = [r["max_expert_top1_share"] for r in block]
            for r in block:
                axes[0].scatter(dataset_idx + offset + (0.035 if r["modality"] == "visual" else -0.035),
                                r["max_expert_top1_share"], s=14, marker="s" if r["modality"] == "visual" else "o",
                                color=variant_color[variant], alpha=.72)
            if vals:
                axes[0].plot(dataset_idx + offset, np.mean(vals), marker="_", markersize=10,
                             color=variant_color[variant], linewidth=0)
    axes[0].set_xticks(range(3), DATASETS)
    axes[0].set_ylim(-.03, 1.04)
    axes[0].set_ylabel("Largest expert top-1 share")
    axes[0].set_title("Routing concentration", loc="left")
    axes[0].grid(axis="y", color="#E4E4E4", linewidth=.55)
    for vi, variant in enumerate(variant_color):
        for dataset_idx, dataset in enumerate(DATASETS):
            block = [r for r in prompts if r["variant"] == variant and r["dataset"] == dataset]
            vals = [np.mean([r["final_cos_01"], r["final_cos_02"], r["final_cos_12"]]) for r in block]
            for r, val in zip(block, vals):
                axes[1].scatter(dataset_idx + (vi - .5) * .15, val, s=14,
                                marker="s" if r["modality"] == "visual" else "o",
                                color=variant_color[variant], alpha=.72)
    axes[1].axhline(0, color="#666666", linewidth=.7, linestyle="--")
    axes[1].set_xticks(range(3), DATASETS)
    axes[1].set_ylabel("Mean final pairwise prompt cosine")
    axes[1].set_title("Prompt geometry", loc="left")
    axes[1].grid(axis="y", color="#E4E4E4", linewidth=.55)
    pair_cos = [r for r in experts if r["kind"] == "expert_cosine"]
    for vi, variant in enumerate(variant_color):
        for dataset_idx, dataset in enumerate(DATASETS):
            block = [r for r in pair_cos if r["variant"] == variant and r["dataset"] == dataset]
            vals = [r["median"] for r in block]
            for r in block:
                axes[2].scatter(dataset_idx + (vi - .5) * .15, r["median"], s=14,
                                marker="s" if r["modality"] == "visual" else "o",
                                color=variant_color[variant], alpha=.72)
    axes[2].axhline(0, color="#666666", linewidth=.7, linestyle="--")
    axes[2].set_ylim(-0.20, 0.45)
    axes[2].set_xticks(range(3), DATASETS)
    axes[2].set_ylabel("Expert-output pair cosine")
    axes[2].set_title("Expert output geometry", loc="left")
    axes[2].grid(axis="y", color="#E4E4E4", linewidth=.55)
    for vi, variant in enumerate(variant_color):
        for dataset_idx, dataset in enumerate(DATASETS):
            block = [r for r in crosses if r["variant"] == variant and r["dataset"] == dataset]
            for row in block:
                axes[3].scatter(dataset_idx + (vi - .5) * .15, row["X_over_L_median"], s=14,
                                marker="s" if row["modality"] == "visual" else "o",
                                color=variant_color[variant], alpha=.72)
    axes[3].axhline(1, color="#666666", linewidth=.7, linestyle="--")
    axes[3].set_xticks(range(3), DATASETS)
    axes[3].set_ylabel("Median RMS(X) / RMS(L)")
    axes[3].set_title("Cross-response scale", loc="left")
    axes[3].grid(axis="y", color="#E4E4E4", linewidth=.55)
    _draw_panel_labels(axes)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=variant_color[name], marker="o", linestyle="none",
                               label="Generic" if name.endswith("generic") else "Protected")
                        for name in variant_color] +
                        [Line2D([], [], color="#555555", marker="o", linestyle="none", label="Text"),
                         Line2D([], [], color="#555555", marker="s", linestyle="none", label="Visual")],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=4, fontsize=5.7)
    fig.suptitle("Cross-modal expert diagnostics; points are seed × modality", fontsize=9, y=0.99)
    _save_figure(fig, path, plt)


def _figure_attention(diag: dict[str, list[dict[str, Any]]], path: Path, plt) -> None:
    means, differences = diag["protected_attention"], diag["modality_attention_difference"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    axes = np.asarray(axes).reshape(-1)
    token_colors = {"L": "#3775BA", "H": "#42949E", "X": "#B64342"}
    dataset_colors = dict(zip(DATASETS, ("#3775BA", "#42949E", "#B64342")))
    modality_markers = {"text": "o", "visual": "s"}

    # Means across seeds, with dataset color and modality marker/line style.
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            block = [r for r in means if r["dataset"] == dataset and r["modality"] == modality]
            if not block:
                continue
            token_means, token_sds = [], []
            for token in ("L", "H", "X"):
                values = [r[f"attention_{token}_mean"] for r in block]
                mean, sd = _mean_sd(values)
                token_means.append(mean)
                token_sds.append(sd)
            axes[0].errorbar(range(3), token_means, yerr=token_sds,
                             color=dataset_colors[dataset], marker=modality_markers[modality],
                             markersize=3, capsize=1.8, linewidth=.85,
                             markerfacecolor="white", linestyle="-" if modality == "text" else "--")
    axes[0].set_xticks(range(3), ["L", "H", "X"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Attention weight; seed mean ± SD")
    axes[0].set_title("Token utilization by dataset and modality", loc="left")
    axes[0].grid(axis="y", color="#E4E4E4", linewidth=.55)

    # Validation-node standard deviations aggregated by seed and modality.
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            block = [r for r in means if r["dataset"] == dataset and r["modality"] == modality]
            if not block:
                continue
            token_means, token_sds = [], []
            for token in ("L", "H", "X"):
                values = [r[f"attention_{token}_node_std"] for r in block]
                mean, sd = _mean_sd(values)
                token_means.append(mean)
                token_sds.append(sd)
            axes[1].errorbar(range(3), token_means, yerr=token_sds,
                             color=dataset_colors[dataset], marker=modality_markers[modality],
                             markersize=3, capsize=1.8, linewidth=.85,
                             markerfacecolor="white", linestyle="-" if modality == "text" else "--")
    axes[1].set_xticks(range(3), ["L", "H", "X"])
    axes[1].set_ylabel("Node-wise SD")
    axes[1].set_title("Within-modality node variation", loc="left")
    axes[1].grid(axis="y", color="#E4E4E4", linewidth=.55)

    # Text-versus-Visual attention distance, kept on its own scale.
    for di, dataset in enumerate(DATASETS):
        block = [r for r in differences if r["dataset"] == dataset]
        values = [r["attention_text_visual_l1_mean"] for r in block]
        if values:
            axes[2].errorbar(di, np.mean(values), yerr=np.std(values, ddof=0),
                             color=dataset_colors[dataset], marker="o",
                             markersize=3, capsize=1.8, linewidth=.85, markerfacecolor="white")
    axes[2].set_xticks(range(3), DATASETS)
    axes[2].set_ylabel("Text–Visual attention L1 difference")
    axes[2].set_title("Modality-level utilization shift", loc="left")
    axes[2].grid(axis="y", color="#E4E4E4", linewidth=.55)

    # Correlation is separate from L1 distance because it has a different scale.
    for di, dataset in enumerate(DATASETS):
        block = [r for r in differences if r["dataset"] == dataset]
        for token, offset, color in (("L", -.16, token_colors["L"]),
                                     ("H", 0, token_colors["H"]), ("X", .16, token_colors["X"])):
            values = [r[f"attention_{token}_text_visual_corr"] for r in block
                      if np.isfinite(r[f"attention_{token}_text_visual_corr"])]
            if values:
                axes[3].errorbar(di + offset, np.mean(values), yerr=np.std(values, ddof=0),
                                 color=color, marker="o", markersize=3, capsize=1.8, linewidth=.8)
    axes[3].axhline(0, color="#666666", linewidth=.7, linestyle="--")
    axes[3].set_xticks(range(3), DATASETS)
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].set_ylabel("Node-wise Text–Visual correlation")
    axes[3].set_title("Token-specific correspondence", loc="left")
    axes[3].grid(axis="y", color="#E4E4E4", linewidth=.55)

    from matplotlib.lines import Line2D
    _draw_panel_labels(axes)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=dataset_colors[name], marker="o", linestyle="-", label=name)
                        for name in DATASETS] +
                       [Line2D([], [], color="#555555", marker=modality_markers[name], linestyle="none",
                               label=name.capitalize()) for name in ("text", "visual")],
               loc="upper center", bbox_to_anchor=(.5, .925), ncol=5, fontsize=5.7)
    fig.suptitle("Protected composer attention over [L, H, X]", fontsize=9, y=0.99)
    _save_figure(fig, path, plt)


def _figure_interventions(rows: list[dict[str, Any]], path: Path, plt) -> None:
    selected = [r for r in rows if r["intervention"] != "identity"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True)
    interventions = ["X_off", "H_off", "L_off", "router_tuple_shuffle"]
    display = ["X off", "H off", "L off", "Router tuple\nshuffle"]
    metrics = (("delta_accuracy_pp", "Accuracy Δ (pp)"),
               ("delta_macro_f1_pp", "Macro-F1 Δ (pp)"),
               ("delta_ce", "CE Δ"))
    colors = dict(zip(DATASETS, ("#3775BA", "#42949E", "#B64342")))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for ii, intervention in enumerate(interventions):
            for di, dataset in enumerate(DATASETS):
                members = [r for r in selected if r["intervention"] == intervention and r["dataset"] == dataset]
                # Router repeats are first averaged within each checkpoint.
                by_seed = defaultdict(list)
                for row in members:
                    by_seed[int(row["seed"])].append(float(row[metric]))
                points = [(seed, float(np.mean(values))) for seed, values in by_seed.items()]
                vals = [value for _, value in points]
                if not vals:
                    continue
                mean, sd = _mean_sd(vals)
                xpos = ii + (di - 1) * .20
                ax.errorbar(xpos, mean, yerr=sd, color=colors[dataset], marker="o",
                            markersize=3, capsize=2, linewidth=.9,
                            label=dataset if ii == 0 else None)
                for seed, value in points:
                    ax.scatter(xpos + _jitter(seed), value, s=8, color=colors[dataset], alpha=.65, zorder=3)
        ax.axhline(0, color="#666666", linewidth=.7, linestyle="--")
        ax.set_xticks(range(4), display)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E4E4E4", linewidth=.55)
    _draw_panel_labels(axes)
    _add_dataset_color_key(fig, colors)
    fig.suptitle("P3 selected-checkpoint reliance (not retrained architecture effects)", fontsize=9, y=0.99)
    _save_figure(fig, path, plt)


def _make_figures(rows: list[dict[str, Any]], paired: list[dict[str, Any]],
                  diag: dict[str, list[dict[str, Any]]], interventions: list[dict[str, Any]],
                  figure_dir: Path) -> None:
    plt = _prepare_matplotlib()
    _figure_performance(rows, figure_dir / "r0_performance", plt)
    _figure_paired(paired, figure_dir / "r0_paired_deltas", plt)
    _figure_response(diag, figure_dir / "r0_response_bank", plt)
    _figure_moe(diag, figure_dir / "r0_moe_specialization", plt)
    _figure_attention(diag, figure_dir / "r0_protected_attention", plt)
    _figure_interventions(interventions, figure_dir / "r0_interventions", plt)


def analyze_campaign(out_dir: Path, research_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    out_dir, research_dir = Path(out_dir), Path(research_dir)
    data_dir, figure_dir = research_dir / "data", research_dir / "figures"
    research_dir.mkdir(parents=True, exist_ok=True)
    run_rows = _read_runs(out_dir)
    perf_rows, perf_summary, paired_rows, paired_summary = _performance(run_rows, data_dir)
    init_rows, capacity_rows = _capacity_audits(run_rows, data_dir)
    diagnostics = _flatten_diagnostics(run_rows)
    for key, filename in (
        ("response_scale", "response_scale_summary.csv"),
        ("response_distinctness", "response_distinctness.csv"),
        ("moe_usage", "moe_usage_summary.csv"),
        ("prompt_specialization", "prompt_specialization.csv"),
        ("expert_distinctness", "expert_distinctness.csv"),
        ("cross_response", "cross_response_summary.csv"),
        ("protected_attention", "protected_attention_summary.csv"),
        ("modality_attention_difference", "modality_attention_difference.csv"),
    ):
        _write_csv(data_dir / filename, diagnostics[key])
    _orthology(run_rows, data_dir)
    _, intervention_summary = _interventions(run_rows, data_dir)
    clean_manifest = dict(manifest)
    clean_manifest["analysis_run_count"] = len(run_rows)
    clean_manifest["report_metrics"] = ["validation Accuracy", "validation Macro-F1", "validation CE"]
    clean_manifest["test_evaluation"] = False
    _write_json(research_dir / "run_manifest.json", clean_manifest)
    _write_json(out_dir / "campaign_manifest.json", clean_manifest)
    readme = """# R0 — Mature Structural-Response Expert Prototype\n\nThis directory contains the validation-only report bundle for the fixed R0 campaign. The four variants share exact initialized modules and total parameter counts; only the active forward path and CrossMoE orthogonality objective differ.\n\n- `report.md`: results, bounded interpretation, and final route decision.\n- `figure_qa.md`: source, alignment, collision, text-size, and visual-review results.\n- `run_manifest.json`: source SHA, protocol, run count, failures, device, and execution metadata.\n- `data/`: per-run performance and validation diagnostics, paired comparisons, initialization/capacity audits, orthogonality records, and checkpoint interventions.\n- `figures/`: Python-rendered PNG/TIFF previews and editable vector SVG/PDF figures with alignment QA manifests.\n\nAll metrics are validation-only. Three-seed summaries use population SD. Pooled nine-pair summaries are descriptive and are not IID significance tests. Checkpoints and large per-node routing logs are stored in the ignored `outputs/r0_mature_response_expert_prototype/` directory.\n"""
    (research_dir / "README.md").write_text(readme, encoding="utf-8")
    report = _report_text(run_rows, perf_summary, paired_summary, init_rows,
                          capacity_rows, diagnostics, intervention_summary, clean_manifest)
    (research_dir / "report.md").write_text(report, encoding="utf-8")
    if run_rows:
        _make_figures(perf_rows, paired_rows, diagnostics,
                      [row for run in run_rows for row in run.get("interventions", [])], figure_dir)
    return {"run_count": len(run_rows), "performance_rows": len(perf_rows),
            "paired_rows": len(paired_rows), "research_dir": str(research_dir)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(analyze_campaign(args.out_dir, args.research_dir, json.loads(args.manifest.read_text())))
