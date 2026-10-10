#!/usr/bin/env python3
"""Analyze the frozen validation-only PSCE-MAG V7A campaign."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/psce_mag_v7a"
DATA_ROOT = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("a0_raw", "a1_static_dual", "a2_conditional_dual")
COMPARISONS = (
    ("a1_minus_a0", "a1_static_dual", "a0_raw"),
    ("a2_minus_a1", "a2_conditional_dual", "a1_static_dual"),
    ("a2_minus_a0", "a2_conditional_dual", "a0_raw"),
)

HISTORICAL = {
    "Movies": {
        "V4A_R0_raw": {"val_acc_mean_pct": 56.029, "val_acc_sd_pct": 0.385, "val_macro_f1_mean_pct": 47.997, "val_macro_f1_sd_pct": 1.637},
        "PIGPR_C1_RGD": {"val_acc_mean_pct": 56.389, "val_acc_sd_pct": 0.170, "val_macro_f1_mean_pct": 49.698, "val_macro_f1_sd_pct": 0.281},
    },
    "Grocery": {
        "V4A_R0_raw": {"val_acc_mean_pct": 83.094, "val_acc_sd_pct": 0.325, "val_macro_f1_mean_pct": 75.766, "val_macro_f1_sd_pct": 1.387},
        "PIGPR_C1_RGD": {"val_acc_mean_pct": 83.094, "val_acc_sd_pct": 0.193, "val_macro_f1_mean_pct": 77.342, "val_macro_f1_sd_pct": 0.228},
    },
    "ele-fashion": {
        "V4A_R0_raw": {"val_acc_mean_pct": 87.413, "val_acc_sd_pct": 0.128, "val_macro_f1_mean_pct": 74.778, "val_macro_f1_sd_pct": 1.560},
        "PIGPR_C1_RGD": {"val_acc_mean_pct": 87.324, "val_acc_sd_pct": 0.042, "val_macro_f1_mean_pct": 74.536, "val_macro_f1_sd_pct": 0.252},
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def sd(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _metric_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, float]]:
    result = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]), row["variant"])
        metrics = row["metrics"]
        if any(str(name).lower().startswith("test") for name in metrics):
            raise RuntimeError(f"Test metric key found in {key}")
        result[key] = {
            "val_acc": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
        }
    expected = {
        (dataset, seed, variant)
        for dataset in DATASETS
        for seed in SEEDS
        for variant in VARIANTS
    }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(f"expected exactly 27 formal runs; missing={missing}, extra={extra}")
    return result


def _summary_rows(metrics: dict[tuple[str, int, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            cells = [metrics[(dataset, seed, variant)] for seed in SEEDS]
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "n": len(cells),
                    "val_acc_mean": mean([cell["val_acc"] for cell in cells]),
                    "val_acc_sd": sd([cell["val_acc"] for cell in cells]),
                    "val_macro_f1_mean": mean([cell["val_macro_f1"] for cell in cells]),
                    "val_macro_f1_sd": sd([cell["val_macro_f1"] for cell in cells]),
                    "best_epoch_mean": mean(
                        [
                            float(row["metadata"].get("best_epoch", 0))
                            for row in read_json(DATA_ROOT / "run_rows.json")
                            if row["dataset"] == dataset and row["variant"] == variant
                        ]
                    ),
                }
            )
    return rows


def _paired_rows(metrics: dict[tuple[str, int, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, candidate, baseline in COMPARISONS:
        for dataset in DATASETS:
            acc_deltas = []
            f1_deltas = []
            for seed in SEEDS:
                left = metrics[(dataset, seed, candidate)]
                right = metrics[(dataset, seed, baseline)]
                acc_delta = (left["val_acc"] - right["val_acc"]) * 100.0
                f1_delta = (left["val_macro_f1"] - right["val_macro_f1"]) * 100.0
                acc_deltas.append(acc_delta)
                f1_deltas.append(f1_delta)
                rows.append(
                    {
                        "comparison": label,
                        "dataset": dataset,
                        "seed": seed,
                        "candidate": candidate,
                        "baseline": baseline,
                        "val_acc_delta_pp": acc_delta,
                        "val_macro_f1_delta_pp": f1_delta,
                    }
                )
            rows.append(
                {
                    "comparison": label,
                    "dataset": dataset,
                    "seed": "ALL",
                    "candidate": candidate,
                    "baseline": baseline,
                    "val_acc_delta_pp": mean(acc_deltas),
                    "val_acc_delta_sd_pp": sd(acc_deltas),
                    "val_acc_positive_pairs": sum(value > 0 for value in acc_deltas),
                    "val_macro_f1_delta_pp": mean(f1_deltas),
                    "val_macro_f1_delta_sd_pp": sd(f1_deltas),
                    "val_macro_f1_positive_pairs": sum(value > 0 for value in f1_deltas),
                }
            )
        acc_deltas = []
        f1_deltas = []
        for dataset in DATASETS:
            for seed in SEEDS:
                left = metrics[(dataset, seed, candidate)]
                right = metrics[(dataset, seed, baseline)]
                acc_deltas.append((left["val_acc"] - right["val_acc"]) * 100.0)
                f1_deltas.append((left["val_macro_f1"] - right["val_macro_f1"]) * 100.0)
        rows.append(
            {
                "comparison": label,
                "dataset": "ALL",
                "seed": "ALL",
                "candidate": candidate,
                "baseline": baseline,
                "val_acc_delta_pp": mean(acc_deltas),
                "val_acc_delta_sd_pp": sd(acc_deltas),
                "val_acc_positive_pairs": sum(value > 0 for value in acc_deltas),
                "val_macro_f1_delta_pp": mean(f1_deltas),
                "val_macro_f1_delta_sd_pp": sd(f1_deltas),
                "val_macro_f1_positive_pairs": sum(value > 0 for value in f1_deltas),
            }
        )
    return rows


def _diagnostic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        diagnostic = row.get("semantic_checkpoint_diagnostics")
        base = {"dataset": row["dataset"], "seed": row["seed"], "variant": row["variant"]}
        if not diagnostic:
            result.append({**base, "semantic_path_present": False})
            continue
        edge = diagnostic.get("edge_weight", {})
        closed = diagnostic.get("closed_path", {})
        for modality, values in diagnostic.get("modalities", {}).items():
            result.append(
                {
                    **base,
                    "semantic_path_present": True,
                    "modality": modality,
                    "edge_weight_mean": edge.get("mean"),
                    "edge_weight_std": edge.get("std"),
                    "edge_weight_p10": edge.get("p10"),
                    "edge_weight_p50": edge.get("p50"),
                    "edge_weight_p90": edge.get("p90"),
                    "semantic_hop1_rms": (values.get("semantic_hop_rms") or [None, None])[0],
                    "semantic_hop2_rms": (values.get("semantic_hop_rms") or [None, None])[1],
                    "raw_hop1_rms": (values.get("raw_reference_hop_rms") or [None, None])[0],
                    "raw_hop2_rms": (values.get("raw_reference_hop_rms") or [None, None])[1],
                    "semantic_raw_cosine_hop1": (values.get("semantic_raw_cosine") or [None, None])[0],
                    "semantic_raw_cosine_hop2": (values.get("semantic_raw_cosine") or [None, None])[1],
                    "semantic_strength": (values.get("semantic_strength") or {}).get("mean"),
                    "semantic_contribution_rms": values.get("semantic_contribution_rms"),
                    "semantic_to_physical_contribution_rms": values.get("semantic_to_physical_contribution_rms"),
                    "top2_selection_share_json": json.dumps(values.get("semantic_expert_selection_share")),
                    "physical_strength": values.get("physical_strength_mean"),
                    "shared_expert_gradient_nonzero": any(
                        item.get("gradient_nonzero", False)
                        for name, item in (diagnostic.get("gradient_diagnostics") or {}).items()
                        if name.startswith("experts.")
                    ),
                    "relation_scorer_gradient_norm": max(
                        [
                            item.get("gradient_norm", 0.0)
                            for name, item in (diagnostic.get("gradient_diagnostics") or {}).items()
                            if name.startswith("relation_scorer.")
                        ]
                        or [0.0]
                    ),
                    "semantic_closed_output_delta_rms": closed.get("output_delta_rms"),
                    "semantic_closed_val_acc": closed.get("val_acc"),
                    "semantic_closed_val_macro_f1": closed.get("val_macro_f1"),
                    "semantic_closed_val_acc_delta": closed.get("val_acc_delta_from_open"),
                    "semantic_closed_val_macro_f1_delta": closed.get("val_macro_f1_delta_from_open"),
                }
            )
    return result


def _resource_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        meta = row.get("metadata", {})
        output.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "variant": row["variant"],
                "status": row.get("status"),
                "best_epoch": meta.get("best_epoch"),
                "epochs_completed": meta.get("epochs_completed"),
                "training_wall_seconds": meta.get("training_wall_seconds"),
                "mean_epoch_wall_seconds": meta.get("mean_epoch_wall_seconds"),
                "mean_train_step_seconds": meta.get("mean_train_step_seconds"),
                "model_parameters": meta.get("model_parameters"),
                "classifier_parameters": meta.get("classifier_parameters"),
                "cuda_device_name": meta.get("cuda_device_name"),
                "cuda_peak_allocated_bytes": meta.get("cuda_peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": meta.get("cuda_peak_reserved_bytes"),
                "semantic_candidate_fingerprint": meta.get("semantic_candidate_fingerprint"),
            }
        )
    return output


def _fmt_pct(value: float, spread: float | None = None) -> str:
    if spread is None:
        return f"{value * 100:.3f}%"
    return f"{value * 100:.3f} ± {spread * 100:.3f}%"


def make_report(
    manifest: dict[str, Any],
    candidate_summary: dict[str, Any],
    preflight: dict[str, Any],
    test_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> str:
    candidate_by_ds = {row["dataset"]: row for row in candidate_summary["datasets"]}
    preflight_by_ds = {row["dataset"]: row for row in preflight.get("datasets", [])}
    summary_lookup = {(row["dataset"], row["variant"]): row for row in summary_rows}
    all_pairs = {
        row["comparison"]: row for row in paired_rows if row["dataset"] == "ALL"
    }
    report: list[str] = [
        "# PSCE-MAG V7A: Physical–Semantic Collaborative Experts",
        "",
        "## Protocol and provenance",
        "",
        f"- Baseline: `{manifest['provenance']['baseline_commit_sha']}`.",
        f"- Frozen implementation: `{manifest['provenance']['freeze_commit_sha']}`.",
        f"- Branch: `{manifest['provenance']['branch']}`.",
        f"- Protocol: `{manifest['protocol']}`; Validation Accuracy selected checkpoints; Macro-F1 diagnostic.",
        f"- Formal runs: {manifest['completed_runs']}/27; Test evaluation disabled and no Test metric keys were present.",
        f"- Full repository pytest: passed={test_summary['passed']}, exit={test_summary['exit_code']}.",
        "- Runs use full-graph NC with fixed K=8 per modality, hidden size 256, four shared experts, and seeds 42–44.",
        "",
        "## Validation results",
        "",
        "Accuracy and Macro-F1 values are shown as percent mean ± population SD over three seeds.",
        "",
        "| Dataset | Variant | Validation Accuracy | Validation Macro-F1 | Best epoch mean |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = summary_lookup[(dataset, variant)]
            report.append(
                f"| {dataset} | `{variant}` | {_fmt_pct(row['val_acc_mean'], row['val_acc_sd'])} | "
                f"{_fmt_pct(row['val_macro_f1_mean'], row['val_macro_f1_sd'])} | {row['best_epoch_mean']:.1f} |"
            )
    report += ["", "### Paired comparisons", "", "Deltas are paired by dataset and seed; positive counts are descriptive, not significance tests.", "", "| Comparison | Accuracy Δ (pp) | Macro-F1 Δ (pp) | Positive Accuracy pairs | Positive F1 pairs |", "|---|---:|---:|---:|---:|"]
    for label, _, _ in COMPARISONS:
        row = all_pairs[label]
        report.append(
            f"| {label.replace('_minus_', ' − ')} | {row['val_acc_delta_pp']:+.3f} ± {row['val_acc_delta_sd_pp']:.3f} | "
            f"{row['val_macro_f1_delta_pp']:+.3f} ± {row['val_macro_f1_delta_sd_pp']:.3f} | "
            f"{row['val_acc_positive_pairs']}/9 | {row['val_macro_f1_positive_pairs']}/9 |"
        )
    report += ["", "Per-dataset and per-seed deltas are in `data/paired_comparisons.csv`.", "", "## Candidate graph and retrieval", "", "| Dataset | Nodes | Undirected semantic edges | Nonphysical fraction | Text/Visual Jaccard | Covered nodes | IVF recall@8 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        row = candidate_by_ds[dataset]
        recall = row["retrieval_recall_sample"]
        rtext = recall["text"].get("recall_at_k_mean")
        rvisual = recall["visual"].get("recall_at_k_mean")
        recall_label = "exact" if rtext == 1.0 and rvisual == 1.0 else f"{rtext:.3f}/{rvisual:.3f}"
        report.append(
            f"| {dataset} | {row['num_nodes']} | {row['undirected_candidate_edge_count']} | "
            f"{row['candidate_nonphysical_ratio']:.3f} | {row['text_visual_overlap_jaccard']:.3f} | "
            f"{row['covered_node_count']} | {recall_label} |"
        )
    report += [
        "",
        "FAISS provenance, degree/similarity quantiles, cache paths, and fingerprints are in `data/candidate_graph_summary.json`. The nonphysical-edge fraction describes added topology only; it is not evidence of task value.",
        "",
        "## GPU preflight and resources",
        "",
        f"GPU preflight passed: **{preflight.get('passed')}** on `{preflight.get('environment', {}).get('selected_device', {}).get('device_name')}`.",
        "",
        "| Dataset | Training step | Validation inference | Peak allocated | Peak reserved | Train step (s) | Relation score (s) | Semantic propagation (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        row = preflight_by_ds[dataset]
        report.append(
            f"| {dataset} | {row.get('status')} | {row.get('validation_metrics', {}).get('val_acc', float('nan')):.4f} acc | "
            f"{row.get('cuda_peak_allocated_bytes', 0) / 1024**3:.2f} GiB | "
            f"{row.get('cuda_peak_reserved_bytes', 0) / 1024**3:.2f} GiB | "
            f"{row.get('training_step_seconds', float('nan')):.2f} | "
            f"{row.get('relation_scoring_seconds', float('nan')):.2f} | "
            f"{row.get('semantic_propagation_seconds', float('nan')):.2f} |"
        )
    report += [
        "",
        "Formal per-run epoch time and peak GPU memory are in `data/resource_profile.json`. No GPU process was stopped.",
        "",
        "## Checkpoint semantic-branch diagnostics",
        "",
        "Learned edge weights, route strengths, and branch closure report model usage; they are not causal edge-utility estimates. `semantic_diagnostics.csv` records per-run edge-weight quantiles, semantic/Raw state RMS and cosine, shared-expert gradients, semantic contribution ratios, and Validation metrics after closing the semantic path at the selected checkpoint.",
        "",
        "## Historical context",
        "",
        "| Dataset | Historical V4A R0 Accuracy / Macro-F1 | Historical PIGPR-C1 RGD Accuracy / Macro-F1 |",
        "|---|---:|---:|",
    ]
    for dataset in DATASETS:
        v4 = HISTORICAL[dataset]["V4A_R0_raw"]
        pig = HISTORICAL[dataset]["PIGPR_C1_RGD"]
        report.append(
            f"| {dataset} | {v4['val_acc_mean_pct']:.3f} ± {v4['val_acc_sd_pct']:.3f}% / {v4['val_macro_f1_mean_pct']:.3f} ± {v4['val_macro_f1_sd_pct']:.3f}% | "
            f"{pig['val_acc_mean_pct']:.3f} ± {pig['val_acc_sd_pct']:.3f}% / {pig['val_macro_f1_mean_pct']:.3f} ± {pig['val_macro_f1_sd_pct']:.3f}% |"
        )
    report += [
        "",
        "These earlier reports/configurations are descriptive context only; they are not strict paired baselines for the V7A campaign.",
        "",
        "## Objective readout",
        "",
    ]
    a1a0 = all_pairs["a1_minus_a0"]
    a2a1 = all_pairs["a2_minus_a1"]
    a2a0 = all_pairs["a2_minus_a0"]
    report.append(
        f"- Semantic context (A1−A0): {a1a0['val_acc_delta_pp']:+.3f} pp Accuracy and {a1a0['val_macro_f1_delta_pp']:+.3f} pp Macro-F1 on average; positive on {a1a0['val_acc_positive_pairs']}/9 Accuracy and {a1a0['val_macro_f1_positive_pairs']}/9 F1 pairs."
    )
    report.append(
        f"- Conditional use (A2−A1): {a2a1['val_acc_delta_pp']:+.3f} pp Accuracy and {a2a1['val_macro_f1_delta_pp']:+.3f} pp Macro-F1; positive on {a2a1['val_acc_positive_pairs']}/9 Accuracy and {a2a1['val_macro_f1_positive_pairs']}/9 F1 pairs."
    )
    report.append(
        f"- Full model (A2−A0): {a2a0['val_acc_delta_pp']:+.3f} pp Accuracy and {a2a0['val_macro_f1_delta_pp']:+.3f} pp Macro-F1."
    )
    report += [
        "",
        "These are descriptive results over three fixed seeds and three datasets. Read the per-dataset paired rows before making a deployment or scientific claim. No mechanism is selected or expanded from a gate magnitude alone.",
        "",
        "## Limitations and artifacts",
        "",
        "- Full-graph validation-only NC was evaluated. Link prediction is explicitly unsupported because its sampler uses local IDs; no LP validation is claimed.",
        "- The approximate FAISS retrieval recall sample is feature-only and does not use labels or validation outcomes.",
        "- Historical comparisons differ in run/config provenance and are not paired.",
        "- Failed/retried cells are recorded in `data/failures.json` and the campaign manifest.",
        "- Candidate caches, checkpoints, and training logs are stored under ignored `outputs/psce_mag_v7a/`.",
        "",
        f"Report generated: {now()}.",
    ]
    return "\n".join(report) + "\n"


def main() -> None:
    manifest = read_json(DATA_ROOT / "campaign_manifest.json")
    candidates = read_json(DATA_ROOT / "candidate_graph_summary.json")
    preflight = read_json(DATA_ROOT / "preflight_summary.json")
    tests = read_json(DATA_ROOT / "test_summary.json")
    rows = read_json(DATA_ROOT / "run_rows.json")
    metrics = _metric_map(rows)
    summaries = _summary_rows(metrics)
    paired = _paired_rows(metrics)
    diagnostics = _diagnostic_rows(rows)
    resources = _resource_rows(rows)
    write_csv(DATA_ROOT / "summary.csv", summaries)
    write_csv(DATA_ROOT / "paired_comparisons.csv", paired)
    write_csv(DATA_ROOT / "semantic_diagnostics.csv", diagnostics)
    write_csv(DATA_ROOT / "resource_profile.csv", resources)
    report = make_report(
        manifest,
        candidates,
        preflight,
        tests,
        rows,
        summaries,
        paired,
        diagnostics,
        resources,
    )
    (RESEARCH / "REPORT.md").write_text(report, encoding="utf-8")
    (RESEARCH / "README.md").write_text(
        "# PSCE-MAG V7A artifacts\n\nSee [REPORT.md](REPORT.md) for the frozen full-graph validation-only NC campaign and [data/](data/) for structured outputs. Checkpoints, candidate caches, and logs are kept under ignored `outputs/psce_mag_v7a/`.\n",
        encoding="utf-8",
    )
    resource_payload = {
        "generated_at_utc": now(),
        "formal_runs": resources,
        "preflight": preflight.get("datasets", []),
    }
    (DATA_ROOT / "resource_profile.json").write_text(
        json.dumps(resource_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Wrote {RESEARCH / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
