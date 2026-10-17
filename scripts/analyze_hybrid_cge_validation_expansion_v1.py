#!/usr/bin/env python3
"""Validation-only analysis and report generation for Hybrid-CGE expansion."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/hybrid_cge_validation_expansion_v1"
DATA = STUDY / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion", "Toys", "Reddit-S")
OPT_DATASETS = DATASETS[:3]
SEEDS = (42, 43, 44)
BASELINES = ("mlp", "sage", "mmgcn", "rolemag")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def avg_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def valid_complete_rows() -> list[dict[str, Any]]:
    raw = read_csv(DATA / "runs.csv")
    rows = [r for r in raw if r.get("status") == "complete"]
    for row in rows:
        if row.get("test_evaluation") != "False":
            raise RuntimeError(f"Test evaluation is not explicitly false for {row.get('dataset')}/{row.get('condition')}/{row.get('seed')}")
        for key in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_macro_f1", "train_ce_eval"):
            if row.get(key, "") and not math.isfinite(float(row[key])):
                raise RuntimeError(f"non-finite metric {key} in {row}")
        for key in ("best_accuracy_checkpoint", "best_accuracy_checkpoint_sha256"):
            if not row.get(key):
                raise RuntimeError(f"missing Best-Accuracy checkpoint evidence: {row}")
    return rows


def summarize_condition_rows(rows: list[dict[str, Any]], condition: str, datasets: tuple[str, ...]) -> dict[str, Any]:
    selected = [r for r in rows if r.get("condition") == condition and r.get("dataset") in datasets]
    expected = len(datasets) * len(SEEDS)
    return {
        "condition": condition,
        "n": len(selected),
        "complete_expected": len(selected) == expected,
        **{metric: (mean([float(r[metric]) for r in selected]) if selected else None)
           for metric in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce",
                          "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap")
           if all(metric in r and r[metric] != "" for r in selected)},
        "per_dataset": {
            dataset: {metric: (mean([float(r[metric]) for r in selected if r["dataset"] == dataset])
                               if any(r["dataset"] == dataset for r in selected) else None)
                      for metric in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_macro_f1")}
            for dataset in datasets
        },
    }


def optimization_analysis(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    historical = read_csv(ROOT / "research/hybrid_cge_v11/data/runs.csv")
    old = []
    for row in historical:
        if row.get("condition") == "H2_hybrid_plus" and row.get("status") == "complete":
            old.append({
                **row, "condition": "H2-Frozen", "model": "hybrid_cge_v11",
                **{key: float(row[key]) for key in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce",
                                                     "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap")},
            })
    new = [r for r in rows if r.get("model") == "hybrid_cge_v11" and r.get("condition") in ("H2-R1", "H2-R2") and r.get("dataset") in OPT_DATASETS]
    paired_rows = []
    for condition in ("H2-R1", "H2-R2"):
        for dataset in OPT_DATASETS:
            for seed in SEEDS:
                a = next((r for r in new if r["condition"] == condition and r["dataset"] == dataset and int(r["seed"]) == seed), None)
                b = next((r for r in old if r["dataset"] == dataset and int(r["seed"]) == seed), None)
                if a is None or b is None:
                    continue
                paired_rows.append({"dataset": dataset, "seed": seed, "condition": condition,
                                    **{f"delta_{metric}": float(a[metric]) - float(b[metric]) for metric in
                                       ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_macro_f1",
                                        "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap")},
                                    "candidate_checkpoint_sha256": a["best_accuracy_checkpoint_sha256"],
                                    "historical_checkpoint_sha256": b["best_accuracy_checkpoint_sha256"]})
    write_csv(DATA / "optimization_paired_differences.csv", paired_rows)
    summaries = {name: summarize_condition_rows(old + new, name, OPT_DATASETS) for name in ("H2-Frozen", "H2-R1", "H2-R2")}
    # A newly trained condition is eligible only if its mean Validation Accuracy exceeds the historical H2 mean.
    baseline_acc = float(summaries["H2-Frozen"]["val_acc"] or 0.0)
    candidates = sorted(("H2-R1", "H2-R2"), key=lambda c: (
        float(summaries[c]["val_acc"] or -1), float(summaries[c]["val_macro_f1"] or -1),
        -float(summaries[c]["val_ce"] or math.inf)), reverse=True)
    selected = candidates[0] if float(summaries[candidates[0]]["val_acc"] or -1) > baseline_acc else "H2-Frozen"
    choice = read_json(DATA / "global_configuration_selection.json", {})
    if choice and choice.get("selected_global_config") != selected:
        raise RuntimeError("runner and independent analyzer disagree on global configuration choice")
    summary = {"summaries": summaries, "paired_cells": len(paired_rows), "selected_global_config": selected,
               "selection_metric": "mean validation accuracy across 3 datasets x 3 seeds",
               "secondary_metrics": ["Macro-F1", "CE", "train-validation gaps"], "test_used": False}
    (DATA / "optimization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Hybrid-CGE configuration optimization", "",
             "H2-Frozen is the historical H2 result and was not rerun. New R1/R2 runs use this branch's frozen runner, fixed data/cache/splits and paired seeds. The table is Validation-only.", "",
             "| Condition | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 | Train−Val Accuracy gap | Train−Val F1 gap |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("H2-Frozen", "H2-R1", "H2-R2"):
        s = summaries[name]
        lines.append(f"| {name} | {s['n']}/9 | {pct(s.get('val_acc'))} | {pct(s.get('val_macro_f1'))} | {num(s.get('val_ce'))} | {pct(s.get('best_f1_val_macro_f1'))} | {pct(s.get('train_val_acc_gap'))} | {pct(s.get('train_val_macro_f1_gap'))} |")
    lines += ["", f"**Selected global configuration:** `{selected}`. Mean accuracy determines eligibility; if no candidate exceeds H2-Frozen, H2-Frozen is retained. The paired seed-by-dataset deltas are in `data/optimization_paired_differences.csv`.", "",
              "R1 changes only model dropout (.20 → .35) while keeping AdamW weight decay and private auxiliary CE at the H2 values. R2 changes dropout, weight decay and private auxiliary CE together; its individual contributions and their interaction cannot be identified from this bundled comparison. Three seeds support descriptive paired comparisons, not strong inferential claims.", "",
              "Historical Pilot scores are reference evidence, not exact acceptance thresholds. This study compares internal validation outcomes and does not inspect Test labels or metrics.", ""]
    return summary, "\n".join(lines)


def checkpoint_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if not row.get("model"):
            continue
        result.append({"dataset": row["dataset"], "model": row["model"], "condition": row["condition"], "seed": row["seed"],
                       "best_accuracy_epoch": row.get("best_accuracy_epoch"), "best_accuracy_val_acc": row.get("val_acc"),
                       "best_accuracy_val_macro_f1": row.get("val_macro_f1"), "best_accuracy_val_ce": row.get("val_ce"),
                       "best_f1_epoch": row.get("best_f1_epoch"), "best_f1_val_acc": row.get("best_f1_val_acc"),
                       "best_f1_val_macro_f1": row.get("best_f1_val_macro_f1"), "best_f1_val_ce": row.get("best_f1_val_ce"),
                       "best_accuracy_checkpoint_sha256": row.get("best_accuracy_checkpoint_sha256"),
                       "best_f1_checkpoint_sha256": row.get("best_f1_checkpoint_sha256"),
                       "best_f1_checkpoint_available": bool(row.get("best_f1_checkpoint_sha256"))})
    return result


def historical_h2_rows() -> list[dict[str, Any]]:
    result = []
    for row in read_csv(ROOT / "research/hybrid_cge_v11/data/runs.csv"):
        if row.get("condition") != "H2_hybrid_plus" or row.get("status") != "complete":
            continue
        result.append({
            **row, "model": "hybrid_cge_v11", "condition": "H2-Frozen",
            "seed": int(row["seed"]), "historical_reference": True,
        })
    return result


def generalization_analysis(rows: list[dict[str, Any]], selected: str) -> str:
    model_rows = [r for r in rows if r.get("model") == "hybrid_cge_v11" and r.get("dataset") in ("Toys", "Reddit-S") and r.get("condition") == selected]
    audit = read_json(DATA / "data_audit.json", {"datasets": {}})
    lines = ["# Hybrid-CGE generalization to Toys and Reddit-S", "",
             f"The single globally selected condition is `{selected}` from the Movies/Grocery/ele-fashion validation study. Toys and Reddit-S did not participate in configuration selection. All available results use Validation only.", "",
             "| Dataset | n | Best-Accuracy Val Acc | Same-checkpoint Macro-F1 | Val CE | Best-F1 Macro-F1 | Train−Val Acc gap |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("Toys", "Reddit-S"):
        d = [r for r in model_rows if r["dataset"] == dataset]
        def avg(field: str):
            return mean([float(r[field]) for r in d if r.get(field, "") != ""]) if d and all(r.get(field, "") != "" for r in d) else None
        lines.append(f"| {dataset} | {len(d)}/3 | {pct(avg('val_acc'))} | {pct(avg('val_macro_f1'))} | {num(avg('val_ce'))} | {pct(avg('best_f1_val_macro_f1'))} | {pct(avg('train_val_acc_gap'))} |")
        entry = audit.get("datasets", {}).get(dataset, {})
        if entry:
            cache = entry.get("semantic_cache") or {}
            lines += ["", f"**{dataset} evidence:** {entry.get('num_nodes')} nodes, {entry.get('num_edges_directed_after_preprocessing')} directed preprocessed physical edges; configured text/visual dimensions {entry.get('feature_dims')}; feature fingerprint `{entry.get('feature_fingerprint')}`; physical edge fingerprint `{entry.get('physical_edge_fingerprint')}`; V7B cache SHA256 `{cache.get('sha256')}`; cache feature/candidate/V7B fingerprints `{cache.get('feature_fingerprint')}` / `{cache.get('candidate_fingerprint')}` / `{cache.get('v7b_fingerprint')}`.",
                      f"Feature paths and SHA256 values, per-seed project split hashes, and source attribution are in `data/data_audit.json`. Test evaluation stayed disabled."]
    if len(model_rows) != 6:
        lines += ["", f"**Status:** {len(model_rows)}/6 runs are complete. Missing or failed cells remain unresolved; no score is imputed."]
    lines += ["", "A low result on either dataset is reported as observed under the frozen global configuration; no dataset-specific retuning or architecture change is applied in this campaign.", ""]
    return "\n".join(lines)


def baseline_analysis(rows: list[dict[str, Any]], selected: str) -> tuple[list[dict[str, Any]], str]:
    models = ("hybrid_cge_v11", *BASELINES)
    summary_rows = []
    lines = ["# Same-protocol baseline comparison", "",
             "Only project-run results appear in the main table; published benchmark numbers are not mixed into it. Each model sees identical dataset features, physical graph, Train/Validation/Test indices and labels (Test labels masked before Trainer), subject to its method-specific graph/feature pathway. All models use AdamW/LR 1e-3, up to 300 epochs, patience 30, min epoch 30 and gradient clip 1.0. MLP is feature-only by design.", "",
             "| Dataset | Model | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in models:
            condition = selected if model == "hybrid_cge_v11" else model
            selected_rows = [r for r in rows if r["dataset"] == dataset and r["model"] == model and r["condition"] == condition]
            avg_acc, sd_acc = avg_sd([float(r["val_acc"]) for r in selected_rows])
            avg_f1, sd_f1 = avg_sd([float(r["val_macro_f1"]) for r in selected_rows])
            avg_ce, sd_ce = avg_sd([float(r["val_ce"]) for r in selected_rows])
            avg_best_f1, _ = avg_sd([float(r["best_f1_val_macro_f1"]) for r in selected_rows if r.get("best_f1_val_macro_f1", "") != ""])
            summary_rows.append({"dataset": dataset, "model": model, "condition": condition, "n": len(selected_rows),
                                 "val_acc_mean": avg_acc, "val_acc_sd": sd_acc, "val_macro_f1_mean": avg_f1,
                                 "val_macro_f1_sd": sd_f1, "val_ce_mean": avg_ce, "val_ce_sd": sd_ce,
                                 "best_f1_macro_f1_mean": avg_best_f1,
                                 "best_f1_checkpoint_count": sum(bool(r.get("best_f1_checkpoint_sha256")) for r in selected_rows)})
            lines.append(f"| {dataset} | {model if model != 'hybrid_cge_v11' else 'Hybrid-CGE ('+condition+')'} | {len(selected_rows)}/3 | {pct(avg_acc)} ± {pct(sd_acc)} | {pct(avg_f1)} ± {pct(sd_f1)} | {num(avg_ce)} ± {num(sd_ce)} | {pct(avg_best_f1)} |")
    lines += ["", "## Implementation fairness and limits", "",
              "- `MLP` uses the same node features, labels and splits but no message passing, as required by the method. Its current feature-only trainer emits a Best-Accuracy checkpoint; a Best-F1 checkpoint is unavailable, so its Best-F1 column is blank.",
              "- `GraphSAGE` uses the repository's supervised full-graph SAGEConv adaptation on the same transductive graph. This differs from sampled inductive GraphSAGE training in the original method family.",
              "- `MMGCN` splits concatenated inputs in `[text, visual]` order and uses parallel modality branches. Its source-style `id_embedding` is a plain tensor rather than a registered parameter; it is therefore fixed at seeded initialization and not optimized or stored in `state_dict`. The run record stores its hash and evaluation recreates it from seed/config. Treat scores as a project adaptation, not exact official MAGB/MMGCN reproduction.",
              "- `RoleMAG` is the project's supervised NC wrapper: it explicitly converts `[text, visual]` input to the core's modality order, forms nonzero/finite feature masks, uses default-zero optional routing regularizers, and applies the configured complementary/heterophily warmup/ramp. It is not described as an official repository reproduction.",
              "- Best-F1 checkpoints for supported graph-model paths are the highest Validation Macro-F1 observed before Validation-Accuracy early stopping. Metrics for Best-Accuracy and Best-F1 are not interchangeable. MLP reports its Best-Accuracy checkpoint only.",
              "", "The table uses mean and sample SD across three seeds. Public MAGB/OpenMAG numbers use different models, feature choices and/or split/training protocols, so they are contextual literature only and are not numerically compared here. See [MAGB](https://github.com/sktsherlock/MAGB), [OpenMAG](https://github.com/YUKI-N810/OpenMAG), the [GraphSAGE paper](https://arxiv.org/abs/1706.02216), and the [official MMGCN repository](https://github.com/iLearn-Lab/MM19-MMGCN).", ""]
    return summary_rows, "\n".join(lines)


def decision_memo(rows: list[dict[str, Any]], selected: str, baseline_summary: list[dict[str, Any]]) -> str:
    completed = [r for r in rows if r.get("status") == "complete"]
    expected_opt = 18
    expected_gen = 6
    expected_baseline = len(DATASETS) * len(BASELINES) * len(SEEDS)
    opt = [r for r in completed if r.get("model") == "hybrid_cge_v11" and r.get("dataset") in OPT_DATASETS and r.get("condition") in ("H2-R1", "H2-R2")]
    gen = [r for r in completed if r.get("model") == "hybrid_cge_v11" and r.get("dataset") in ("Toys", "Reddit-S")]
    base = [r for r in completed if r.get("model") in BASELINES]
    all_hyb = [r for r in completed if r.get("model") == "hybrid_cge_v11"]
    by_dataset = {}
    for d in DATASETS:
        eligible = [x for x in baseline_summary if x["dataset"] == d and x["n"] > 0]
        by_dataset[d] = max(eligible, key=lambda x: (x["val_acc_mean"] or -1, x["val_macro_f1_mean"] or -1))["model"] if eligible else "insufficient runs"
    failures = read_json(DATA / "failures.json", {"failures": []}).get("failures", [])
    checkpoint_count = len(checkpoint_summary([*completed, *historical_h2_rows()]))
    lines = ["# Decision memo", "",
             f"**Selected global Hybrid-CGE configuration:** `{selected}`. Optimization status: {len(opt)}/{expected_opt}; generalization: {len(gen)}/{expected_gen}; baselines: {len(base)}/{expected_baseline}.", "",
             "## Research decisions", "",
             f"- H2-Frozen remains the reference and was not rerun. The selected global setting is `{selected}` under the registered Validation-Accuracy rule; full paired results and secondary Macro-F1/CE/gap metrics are in `OPTIMIZATION_REPORT.md`.",
             f"- H2 cross-dataset transfer status: {len(gen)}/{expected_gen} Toys/Reddit-S runs completed. Use `GENERALIZATION_REPORT.md` and data hashes to judge effectiveness; no Test evaluation is performed.",
             "- Same-protocol baseline competitiveness is descriptive across five datasets and three seeds. Current per-dataset Validation-Accuracy leaders among completed cells: " + ", ".join(f"{d}: {m}" for d, m in by_dataset.items()) + ".",
             "- Accuracy/Macro-F1 trade-offs should be judged from paired seed rows and the separate Best-F1 checkpoint results. Three seeds are not enough to make strong inferential claims.",
             "- Freeze the Hybrid-CGE body for paper-oriented evaluation if the completed baseline and generalization evidence is competitive and free of unresolved data/training defects. Do not claim SOTA from this local protocol comparison.",
             "- If a remaining gap is concentrated in Train–Validation gaps, training regularization/selection is the more direct next investigation; if Train and Validation both underperform, inspect feature quality and cross-dataset representation alignment before adding architectural components. This is a decision heuristic, not a causal conclusion.",
             "", "## Completion and failure status", "",
             f"Completed run rows: {len(completed)}. Recorded preparation/preflight/training failures: {len(failures)}. Failure details and resolutions remain in `data/failures.json`.",
             f"Checkpoint-level rows including historical H2-Frozen references: {checkpoint_count}. Artifact hashes and source/data/cache/config fingerprints are in `data/runs.csv` and `data/checkpoint_selection_summary.csv`.",
             "", "No Test labels or metrics were used for configuration selection or reports.", ""]
    return "\n".join(lines)


def main() -> None:
    rows = valid_complete_rows()
    selection, optimization_text = optimization_analysis(rows)
    selected = selection["selected_global_config"]
    old_h2 = historical_h2_rows()
    comparison_rows = [*rows, *old_h2]
    checkpoint_rows = checkpoint_summary(comparison_rows)
    write_csv(DATA / "checkpoint_selection_summary.csv", checkpoint_rows)
    generalization_text = generalization_analysis(rows, selected)
    baseline_rows, baseline_text = baseline_analysis(comparison_rows, selected)
    write_csv(DATA / "baseline_summary.csv", baseline_rows)
    decision_text = decision_memo(rows, selected, baseline_rows)
    (STUDY / "OPTIMIZATION_REPORT.md").write_text(optimization_text, encoding="utf-8")
    (STUDY / "GENERALIZATION_REPORT.md").write_text(generalization_text, encoding="utf-8")
    (STUDY / "BASELINE_REPORT.md").write_text(baseline_text, encoding="utf-8")
    (STUDY / "DECISION_MEMO.md").write_text(decision_text, encoding="utf-8")
    manifest = read_json(DATA / "manifest.json", {})
    manifest.update({"created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
                     "completed_run_rows": len(rows), "test_evaluation": False,
                     "selected_global_config": selected,
                     "run_status_counts": {"optimization": len([r for r in rows if r["dataset"] in OPT_DATASETS and r["condition"] in ("H2-R1", "H2-R2")]),
                                           "generalization": len([r for r in rows if r["dataset"] in ("Toys", "Reddit-S") and r["model"] == "hybrid_cge_v11"]),
                                           "baselines": len([r for r in rows if r["model"] in BASELINES])}})
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ANALYSIS COMPLETE rows={len(rows)} selected={selected}")


if __name__ == "__main__":
    main()
