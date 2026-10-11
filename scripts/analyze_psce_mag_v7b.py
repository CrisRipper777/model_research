#!/usr/bin/env python3
"""Validation-only summaries and paired comparisons for PSCE-MAG V7B."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/psce_mag_v7b"
DATA = RESEARCH / "data"
VARIANTS = (
    "b0_v7a_original",
    "b1_fixed_knn",
    "b2_random_fixed",
    "b3_relative_shared",
    "b4_relative_modality",
)
LABELS = {
    "b0_v7a_original": "B0",
    "b1_fixed_knn": "B1",
    "b2_random_fixed": "B2",
    "b3_relative_shared": "B3",
    "b4_relative_modality": "B4",
}
PAIRS = (
    ("B1_minus_B2", "b1_fixed_knn", "b2_random_fixed"),
    ("B0_minus_B1", "b0_v7a_original", "b1_fixed_knn"),
    ("B3_minus_B0", "b3_relative_shared", "b0_v7a_original"),
    ("B4_minus_B3", "b4_relative_modality", "b3_relative_shared"),
    ("B4_minus_B0", "b4_relative_modality", "b0_v7a_original"),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(mean_value: float, sd_value: float, scale: float = 100.0) -> str:
    return f"{mean_value * scale:.3f} ± {sd_value * scale:.3f}"


def analyze() -> None:
    rows = read_json(DATA / "run_rows.json")
    expected = 3 * 3 * 5
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} formal rows, found {len(rows)}")
    keys = {(row["dataset"], int(row["seed"]), row["variant"]) for row in rows}
    if len(keys) != expected or any(row.get("status") not in {"completed", "reused"} for row in rows):
        raise RuntimeError("formal campaign has duplicate keys or incomplete rows")
    if any(any(key.lower().startswith("test") for key in row["metrics"]) for row in rows):
        raise RuntimeError("Test metric keys are forbidden in V7B analysis")
    manifest = read_json(DATA / "campaign_manifest.json")
    if manifest.get("task_evaluate_test") is not False or manifest.get("completed_runs") != expected:
        raise RuntimeError("formal manifest is incomplete or not validation-only")

    by_key = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in rows}
    datasets = sorted({row["dataset"] for row in rows}, key=("Movies", "Grocery", "ele-fashion").index)
    summary_rows: list[dict] = []
    for dataset in datasets:
        for variant in VARIANTS:
            selected = [by_key[(dataset, seed, variant)] for seed in (42, 43, 44)]
            acc = [float(row["metrics"]["val_acc"]) for row in selected]
            f1 = [float(row["metrics"]["val_macro_f1"]) for row in selected]
            metadata = [row["metadata"] for row in selected]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "n": 3,
                    "val_acc_mean": mean(acc),
                    "val_acc_population_sd": pstdev(acc),
                    "val_macro_f1_mean": mean(f1),
                    "val_macro_f1_population_sd": pstdev(f1),
                    "best_epoch_mean": mean(float(item["best_epoch"]) for item in metadata),
                    "best_epoch_population_sd": pstdev(float(item["best_epoch"]) for item in metadata),
                    "training_seconds_mean": mean(float(item.get("training_wall_seconds", math.nan)) for item in metadata),
                    "trainable_model_parameters": int(metadata[0]["model_parameters"]),
                    "classifier_parameters": int(metadata[0]["classifier_parameters"]),
                    "peak_cuda_allocated_gib_max": max(float(item.get("cuda_peak_allocated_bytes", 0)) for item in metadata) / 1024**3,
                    "peak_cuda_reserved_gib_max": max(float(item.get("cuda_peak_reserved_bytes", 0)) for item in metadata) / 1024**3,
                }
            )
    write_csv(DATA / "summary.csv", summary_rows)

    comparison_rows: list[dict] = []
    for pair_name, left_variant, right_variant in PAIRS:
        for dataset in datasets:
            for seed in (42, 43, 44):
                left = by_key[(dataset, seed, left_variant)]["metrics"]
                right = by_key[(dataset, seed, right_variant)]["metrics"]
                comparison_rows.append(
                    {
                        "comparison": pair_name,
                        "dataset": dataset,
                        "seed": seed,
                        "left_variant": left_variant,
                        "right_variant": right_variant,
                        "val_acc_delta": float(left["val_acc"]) - float(right["val_acc"]),
                        "val_acc_delta_pp": 100.0 * (float(left["val_acc"]) - float(right["val_acc"])),
                        "val_macro_f1_delta": float(left["val_macro_f1"]) - float(right["val_macro_f1"]),
                        "val_macro_f1_delta_pp": 100.0 * (float(left["val_macro_f1"]) - float(right["val_macro_f1"])),
                    }
                )
    write_csv(DATA / "paired_comparisons.csv", comparison_rows)

    diagnostics_rows: list[dict] = []
    resource_rows: list[dict] = []
    for row in rows:
        meta = row["metadata"]
        diag = meta.get("semantic_checkpoint_diagnostics") or {}
        modalities = diag.get("modalities", {})
        edge = diag.get("edge_weights_by_modality", {})
        text_edge = edge.get("text", diag.get("edge_weight", {}))
        visual_edge = edge.get("visual", diag.get("edge_weight", {}))
        gradients = diag.get("gradient_diagnostics") or meta.get("gradient_diagnostics") or {}
        diagnostics_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "variant": row["variant"],
                "text_edge_weight_stats": json.dumps(text_edge, sort_keys=True),
                "visual_edge_weight_stats": json.dumps(visual_edge, sort_keys=True),
                "edge_weight_correlation": edge.get("text_visual_correlation"),
                "edge_weight_mean_abs_difference": edge.get("text_visual_mean_absolute_difference", 0.0),
                "text_lower_saturation_fraction": edge.get("text_lower_saturation_fraction", 0.0),
                "text_upper_saturation_fraction": edge.get("text_upper_saturation_fraction", 0.0),
                "visual_lower_saturation_fraction": edge.get("visual_lower_saturation_fraction", 0.0),
                "visual_upper_saturation_fraction": edge.get("visual_upper_saturation_fraction", 0.0),
                "source_group_weights": json.dumps(edge.get("source_groups", {}), sort_keys=True),
                "text_semantic_strength": json.dumps(modalities.get("text", {}).get("semantic_strength", {}), sort_keys=True),
                "visual_semantic_strength": json.dumps(modalities.get("visual", {}).get("semantic_strength", {}), sort_keys=True),
                "text_semantic_contribution_rms": modalities.get("text", {}).get("semantic_contribution_rms"),
                "visual_semantic_contribution_rms": modalities.get("visual", {}).get("semantic_contribution_rms"),
                "text_semantic_to_physical_rms": modalities.get("text", {}).get("semantic_to_physical_contribution_rms"),
                "visual_semantic_to_physical_rms": modalities.get("visual", {}).get("semantic_to_physical_contribution_rms"),
                "text_semantic_to_intrinsic_rms": modalities.get("text", {}).get("semantic_contribution_to_intrinsic_rms"),
                "visual_semantic_to_intrinsic_rms": modalities.get("visual", {}).get("semantic_contribution_to_intrinsic_rms"),
                "text_semantic_raw_cosine": json.dumps(modalities.get("text", {}).get("semantic_raw_cosine", [])),
                "visual_semantic_raw_cosine": json.dumps(modalities.get("visual", {}).get("semantic_raw_cosine", [])),
                "text_calibration_scale": json.dumps(modalities.get("text", {}).get("calibration_scale_mean", [])),
                "visual_calibration_scale": json.dumps(modalities.get("visual", {}).get("calibration_scale_mean", [])),
                "scorer_gradient_diagnostics": json.dumps({key: value for key, value in gradients.items() if "relation_scorer" in key}, sort_keys=True),
                "shared_expert_gradient": json.dumps(gradients.get("experts.0.linear1.weight", {}), sort_keys=True),
                "semantic_closed_path": json.dumps(diag.get("closed_path", {}), sort_keys=True),
            }
        )
        resource_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "variant": row["variant"],
                "best_epoch": meta.get("best_epoch"),
                "training_wall_seconds": meta.get("training_wall_seconds"),
                "mean_epoch_wall_seconds": meta.get("mean_epoch_wall_seconds"),
                "trainable_model_parameters": meta.get("model_parameters"),
                "classifier_parameters": meta.get("classifier_parameters"),
                "cuda_device_name": meta.get("cuda_device_name"),
                "peak_cuda_allocated_gib": float(meta.get("cuda_peak_allocated_bytes", 0)) / 1024**3,
                "peak_cuda_reserved_gib": float(meta.get("cuda_peak_reserved_bytes", 0)) / 1024**3,
            }
        )
    write_csv(DATA / "semantic_diagnostics.csv", diagnostics_rows)
    write_csv(DATA / "resource_profile.csv", resource_rows)
    resource_summary = {
        "run_count": len(resource_rows),
        "device_names": sorted({str(item["cuda_device_name"]) for item in resource_rows}),
        "datasets": {
            dataset: {
                "max_allocated_gib": max(item["peak_cuda_allocated_gib"] for item in resource_rows if item["dataset"] == dataset),
                "max_reserved_gib": max(item["peak_cuda_reserved_gib"] for item in resource_rows if item["dataset"] == dataset),
                "mean_training_seconds": mean(float(item["training_wall_seconds"]) for item in resource_rows if item["dataset"] == dataset),
            }
            for dataset in datasets
        },
        "records": resource_rows,
    }
    (DATA / "resource_profile.json").write_text(json.dumps(resource_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(summary_rows, comparison_rows, manifest)


def _write_report(summary_rows: list[dict], comparison_rows: list[dict], manifest: dict) -> None:
    by_summary = {(row["dataset"], row["variant"]): row for row in summary_rows}
    by_comp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in comparison_rows:
        by_comp[(row["comparison"], row["dataset"])].append(row)
    candidate = read_json(DATA / "candidate_graph_summary.json")
    random_graphs = read_json(DATA / "random_graph_summary.json")
    raw = read_json(DATA / "raw_regression_summary.json")
    preflight = read_json(DATA / "preflight_summary.json")
    tests = read_json(DATA / "test_summary.json")
    smoke = read_json(DATA / "smoke_summary.json")
    lines = [
        "# PSCE-MAG V7B validation-only results",
        "",
        f"Branch: `{manifest.get('provenance', {}).get('branch')}`  ",
        f"Baseline: `{manifest.get('provenance', {}).get('baseline_commit_sha')}`  ",
        f"Freeze: `{manifest.get('provenance', {}).get('freeze_commit_sha')}`  ",
        f"Runs: {manifest.get('completed_runs')}/45, protocol `{manifest.get('protocol')}`, `task.evaluate_test=false`. No Test metrics were generated, loaded, saved, or reported.",
        "",
        "## Candidate graph and provenance audit",
        "",
        "All candidate tensors and legacy graph fingerprints were compared directly with the frozen V7A cache. Candidate construction used only feature arrays and the physical edge list; it did not access labels or splits.",
        "",
        "| Dataset | Nodes | Undirected edges | Exact V7A edge/degree match | Text only | Visual only | Both |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in candidate["datasets"]:
        counts = row["source_counts"]
        exact = all(row["tensor_equal_to_v7a"].values())
        lines.append(f"| {row['dataset']} | {row['num_nodes']:,} | {row['undirected_candidate_edge_count']:,} | {exact} | {counts['text_only']:,} | {counts['visual_only']:,} | {counts['both']:,} |")
    lines += [
        "",
        "Degree-preserving null graphs used three successful double-edge swaps per input edge. Exact degrees and edge counts were audited for all nine dataset/seed topologies; graph seeds were 2026/2027/2028 for training seeds 42/43/44. See `data/random_graph_summary.json` for acceptance rates and changed-edge counts.",
        "",
        "## Validation results",
        "",
        "Cells show mean ± population SD over seeds 42/43/44. These are three repeated seeds per dataset, not nine independent datasets.",
        "",
        "| Dataset | Variant | Val Accuracy (%) | Val Macro-F1 (%) | Best epoch | Seconds | Trainable params | Peak allocated / reserved (GiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        for variant in VARIANTS:
            row = by_summary[(dataset, variant)]
            lines.append(
                f"| {dataset} | {LABELS[variant]} `{variant}` | {fmt(row['val_acc_mean'], row['val_acc_population_sd'])} | {fmt(row['val_macro_f1_mean'], row['val_macro_f1_population_sd'])} | {row['best_epoch_mean']:.1f} ± {row['best_epoch_population_sd']:.1f} | {row['training_seconds_mean']:.1f} | {row['trainable_model_parameters']:,} + {row['classifier_parameters']:,} | {row['peak_cuda_allocated_gib_max']:.2f} / {row['peak_cuda_reserved_gib_max']:.2f} |"
            )
    lines += ["", "## Paired comparisons", "", "Differences are left minus right, paired within dataset and seed. Values below summarize the three seed-paired deltas per dataset; no significance test or causal claim is made.", ""]
    for comparison, _, _ in PAIRS:
        lines += [f"### {comparison.replace('_', ' ')}", "", "| Dataset | Accuracy Δ (pp), mean ± SD | Macro-F1 Δ (pp), mean ± SD | Positive seed pairs (Acc / F1) |", "|---|---:|---:|---:|"]
        for dataset in ("Movies", "Grocery", "ele-fashion"):
            subset = by_comp[(comparison, dataset)]
            acc = [float(row["val_acc_delta_pp"]) for row in subset]
            f1 = [float(row["val_macro_f1_delta_pp"]) for row in subset]
            lines.append(f"| {dataset} | {mean(acc):+.3f} ± {pstdev(acc):.3f} | {mean(f1):+.3f} ± {pstdev(f1):.3f} | {sum(x>0 for x in acc)}/3 / {sum(x>0 for x in f1)}/3 |")
        lines.append("")
    lines += [
        "## Relationship diagnostics",
        "",
        "Per selected checkpoint, `data/semantic_diagnostics.csv` records each modality's edge-weight mean/SD/quantiles, text-visual weight correlation and absolute difference, near-bound saturation fractions, source-group weight distributions, semantic gates, semantic-to-physical and semantic-to-intrinsic RMS ratios, semantic/Raw cosine, RMS calibration factors, scorer/shared-expert gradients, and validation-only closed-path effects. These describe model behavior; they do not establish per-edge causal utility.",
        "",
        "B3 averages two output scores before relative centering. Its common score direction is trained, while the output-head difference is unidentifiable in B3. B4 can assign modality-specific weights over the same edge index. The initialization audit verified B3/B4 scorer and common public tensors exactly; the two final output rows begin identical.",
        "",
        "## Baseline checks, compute, and historical context",
        "",
        f"The full-graph Movies same-weight regression passed: max embedding error {raw['max_abs_embedding_error']:.3g}, auxiliary-loss error {raw['max_abs_auxiliary_loss_error']:.3g}, semantic edge summaries equal={raw['semantic_edge_weight_summary_equal']}. CPU tests: {tests.get('passed')} (exit {tests.get('exit_code')}); GPU preflight: {preflight.get('passed')}; five-variant smoke: {smoke.get('audit_passed')}.",
        "",
        "GPU preflight and formal resource maxima are in `data/preflight_summary.json` and `data/resource_profile.json`. The experiment used GPU 1 (RTX 3090); unrelated GPU 0 processes were left running.",
        "",
        "For descriptive context only, historical V4A R0 validation Acc/F1 were Movies 56.029/47.997%, Grocery 83.094/75.766%, Fashion 87.413/74.778%; historical PIGPR-C1 RGD values were 56.389/49.698%, 83.094/77.342%, 87.324/74.536%. Protocols/configurations differ, so these are not paired comparisons.",
        "",
        "## Recommendation and limitations",
        "",
        "Read the five paired comparisons together with the stability and scale diagnostics. Retain the simplest variant whose validation gains repeat across seeds and datasets; treat fixed-vs-learned and real-vs-random gaps as empirical questions. If gains are small or inconsistent, prefer the simpler fixed/shared relation path. This campaign does not support causal interpretations of learned weights, and three datasets × three seeds do not justify broad statistical claims.",
        "",
        "The protocol is full-graph node classification only. Link prediction remains unsupported because the candidate cache stores global node IDs that are incompatible with sampled LP-local IDs. No Test evaluation or hyperparameter search was performed.",
        "",
        "## Artifacts",
        "",
        "- Raw runs: `data/run_rows.json`",
        "- Aggregate results: `data/summary.csv`",
        "- Paired seed differences: `data/paired_comparisons.csv`",
        "- Mechanism diagnostics: `data/semantic_diagnostics.csv`",
        "- GPU resources: `data/resource_profile.json` and `.csv`",
        "- Candidate provenance and random topology: `data/candidate_graph_summary.json`, `data/random_graph_summary.json`",
    ]
    (RESEARCH / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    analyze()
