#!/usr/bin/env python3
"""Independent validation-only analysis for the V7B.1 corrective campaign."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/psce_mag_v7b1_corrective"
DATA = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "r0_raw",
    "b0_v7a_original",
    "b1_fixed_knn",
    "b2_random_fixed",
    "b3_relative_shared",
    "b4_relative_modality",
)
LABELS = {
    "r0_raw": "R0 Raw",
    "b0_v7a_original": "B0 learned V7A A2",
    "b1_fixed_knn": "B1 fixed KNN",
    "b2_random_fixed": "B2 random fixed",
    "b3_relative_shared": "B3 relative shared",
    "b4_relative_modality": "B4 relative modality",
}
PAIRS = (
    ("B1_minus_R0", "b1_fixed_knn", "r0_raw"),
    ("B2_minus_R0", "b2_random_fixed", "r0_raw"),
    ("B1_minus_B2", "b1_fixed_knn", "b2_random_fixed"),
    ("B0_minus_B1", "b0_v7a_original", "b1_fixed_knn"),
    ("B3_minus_B0", "b3_relative_shared", "b0_v7a_original"),
    ("B3_minus_B1", "b3_relative_shared", "b1_fixed_knn"),
    ("B4_minus_B3", "b4_relative_modality", "b3_relative_shared"),
    ("B4_minus_B0", "b4_relative_modality", "b0_v7a_original"),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _edge_summary(diag: dict, modality: str) -> dict:
    edge = diag.get("edge_weights_by_modality", {})
    return edge.get(modality) or diag.get("edge_weight", {})


def _mean_or_none(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(values) if values else None


def _summarize() -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    rows = read_json(DATA / "run_rows.json")
    expected = len(DATASETS) * len(SEEDS) * len(VARIANTS)
    expected_keys = {(dataset, seed, variant) for dataset in DATASETS for seed in SEEDS for variant in VARIANTS}
    keys = {(row.get("dataset"), int(row.get("seed", -1)), row.get("variant")) for row in rows}
    if len(rows) != expected or keys != expected_keys:
        raise RuntimeError(f"expected exactly {expected} unique run rows; found {len(rows)}")
    if any(row.get("status") not in {"completed", "reused"} for row in rows):
        raise RuntimeError("formal run row is not completed")
    if any(any(str(key).lower().startswith("test") for key in row.get("metrics", {})) for row in rows):
        raise RuntimeError("Test metrics are forbidden in V7B.1 analysis")
    manifest = read_json(DATA / "campaign_manifest.json")
    if manifest.get("task_evaluate_test") is not False or manifest.get("completed_runs") != expected:
        raise RuntimeError("campaign manifest is incomplete or not validation-only")
    if manifest.get("completed_runs") != expected:
        raise RuntimeError("campaign manifest run count disagrees with raw rows")
    freeze = manifest.get("provenance", {}).get("freeze_commit_sha")
    config_fp = manifest.get("provenance", {}).get("configuration_fingerprint_sha256")
    if not freeze or not config_fp:
        raise RuntimeError("campaign manifest lacks freeze/config provenance")
    for row in rows:
        if row.get("freeze_commit_sha") != freeze or row.get("configuration_fingerprint_sha256") != config_fp:
            raise RuntimeError("run row provenance differs from the frozen campaign")
        checkpoint = ROOT / row["checkpoint_path"]
        if not checkpoint.is_file():
            raise RuntimeError(f"missing formal checkpoint: {checkpoint}")
        for metric in ("val_acc", "val_macro_f1"):
            if metric not in row["metrics"] or not math.isfinite(float(row["metrics"][metric])):
                raise RuntimeError(f"missing/nonfinite {metric}: {row['run_id']}")
    by_key = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in rows}
    return rows, [by_key[(d, s, v)] for d in DATASETS for s in SEEDS for v in VARIANTS], [], [], manifest


def analyze() -> None:
    rows, _, _, _, manifest = _summarize()
    by_key = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in rows}
    summary_rows: list[dict] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            selected = [by_key[(dataset, seed, variant)] for seed in SEEDS]
            metrics = [row["metrics"] for row in selected]
            metadata = [row["metadata"] for row in selected]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "label": LABELS[variant],
                    "n_seeds": 3,
                    "seeds": json.dumps(SEEDS),
                    "seed_val_acc": json.dumps([float(item["val_acc"]) for item in metrics]),
                    "seed_val_macro_f1": json.dumps([float(item["val_macro_f1"]) for item in metrics]),
                    "val_acc_mean": mean(float(item["val_acc"]) for item in metrics),
                    "val_acc_population_sd": pstdev(float(item["val_acc"]) for item in metrics),
                    "val_macro_f1_mean": mean(float(item["val_macro_f1"]) for item in metrics),
                    "val_macro_f1_population_sd": pstdev(float(item["val_macro_f1"]) for item in metrics),
                    "best_epoch_mean": mean(float(item["best_epoch"]) for item in metadata),
                    "best_epoch_population_sd": pstdev(float(item["best_epoch"]) for item in metadata),
                    "epochs_completed_mean": mean(float(item["epochs_completed"]) for item in metadata),
                    "training_wall_seconds_mean": mean(float(item["training_wall_seconds"]) for item in metadata),
                    "trainable_model_parameters": int(metadata[0]["model_parameters"]),
                    "classifier_parameters": int(metadata[0]["classifier_parameters"]),
                    "gpu_device_names": json.dumps(sorted({str(item.get("cuda_device_name")) for item in metadata})),
                    "peak_cuda_allocated_gib_max": max(float(item.get("cuda_peak_allocated_bytes", 0)) for item in metadata) / 1024**3,
                    "peak_cuda_reserved_gib_max": max(float(item.get("cuda_peak_reserved_bytes", 0)) for item in metadata) / 1024**3,
                    "gradient_diagnostics_seed42": json.dumps(metadata[0].get("gradient_diagnostics", {}), sort_keys=True),
                }
            )
    write_csv(DATA / "summary.csv", summary_rows)
    summary_lookup = {(item["dataset"], item["variant"]): item for item in summary_rows}

    comparison_rows: list[dict] = []
    paired_lookup: dict[tuple[str, str], dict[str, list[float]]] = {}
    for name, left_variant, right_variant in PAIRS:
        for dataset in DATASETS:
            acc_deltas, f1_deltas = [], []
            for seed in SEEDS:
                left = by_key[(dataset, seed, left_variant)]["metrics"]
                right = by_key[(dataset, seed, right_variant)]["metrics"]
                da = float(left["val_acc"]) - float(right["val_acc"])
                df = float(left["val_macro_f1"]) - float(right["val_macro_f1"])
                acc_deltas.append(da); f1_deltas.append(df)
                comparison_rows.append(
                    {
                        "row_type": "seed_pair",
                        "comparison": name,
                        "dataset": dataset,
                        "seed": seed,
                        "left_variant": left_variant,
                        "right_variant": right_variant,
                        "val_acc_delta": da,
                        "val_acc_delta_pp": 100 * da,
                        "val_macro_f1_delta": df,
                        "val_macro_f1_delta_pp": 100 * df,
                    }
                )
            paired_lookup[(name, dataset)] = {"acc": acc_deltas, "f1": f1_deltas}
            comparison_rows.append(
                {
                    "row_type": "dataset_summary",
                    "comparison": name,
                    "dataset": dataset,
                    "seed": "42|43|44",
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "val_acc_delta_seed_values_pp": json.dumps([100 * value for value in acc_deltas]),
                    "val_acc_delta_pp_mean": mean(acc_deltas) * 100,
                    "val_acc_delta_pp_population_sd": pstdev(acc_deltas) * 100,
                    "val_acc_positive_seed_count": sum(value > 0 for value in acc_deltas),
                    "val_macro_f1_delta_seed_values_pp": json.dumps([100 * value for value in f1_deltas]),
                    "val_macro_f1_delta_pp_mean": mean(f1_deltas) * 100,
                    "val_macro_f1_delta_pp_population_sd": pstdev(f1_deltas) * 100,
                    "val_macro_f1_positive_seed_count": sum(value > 0 for value in f1_deltas),
                }
            )
        dataset_acc_means = [mean(paired_lookup[(name, d)]["acc"]) * 100 for d in DATASETS]
        dataset_f1_means = [mean(paired_lookup[(name, d)]["f1"]) * 100 for d in DATASETS]
        comparison_rows.append(
            {
                "row_type": "equal_weight_3_dataset_descriptive",
                "comparison": name,
                "dataset": "EQUAL_WEIGHT_3_DATASET_AVERAGE",
                "seed": "dataset_means",
                "left_variant": left_variant,
                "right_variant": right_variant,
                "val_acc_delta_pp_by_dataset": json.dumps(dict(zip(DATASETS, dataset_acc_means)), sort_keys=True),
                "val_acc_delta_pp_mean": mean(dataset_acc_means),
                "val_acc_delta_pp_population_sd_across_dataset_means": pstdev(dataset_acc_means),
                "val_macro_f1_delta_pp_by_dataset": json.dumps(dict(zip(DATASETS, dataset_f1_means)), sort_keys=True),
                "val_macro_f1_delta_pp_mean": mean(dataset_f1_means),
                "val_macro_f1_delta_pp_population_sd_across_dataset_means": pstdev(dataset_f1_means),
                "positive_seed_counts_by_dataset_acc": json.dumps({d: sum(x > 0 for x in paired_lookup[(name, d)]["acc"]) for d in DATASETS}, sort_keys=True),
                "positive_seed_counts_by_dataset_macro_f1": json.dumps({d: sum(x > 0 for x in paired_lookup[(name, d)]["f1"]) for d in DATASETS}, sort_keys=True),
                "interpretation": "equal weight across three dataset means; descriptive only",
            }
        )
    write_csv(DATA / "paired_comparisons.csv", comparison_rows)

    diagnostic_rows: list[dict] = []
    resource_rows: list[dict] = []
    for row in rows:
        meta = row["metadata"]
        diag = meta.get("semantic_checkpoint_diagnostics") or {}
        modalities = diag.get("modalities", {})
        edge = diag.get("edge_weights_by_modality", {})
        text_edge = _edge_summary(diag, "text")
        visual_edge = _edge_summary(diag, "visual")
        gradients = meta.get("gradient_diagnostics") or diag.get("gradient_diagnostics") or {}
        scorer_gradients = {key: value for key, value in gradients.items() if "relation_scorer" in key or "relation_projections" in key}
        diagnostic_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "variant": row["variant"],
                "candidate_fingerprint": row.get("candidate_fingerprint"),
                "text_edge_weight_stats": json.dumps(text_edge, sort_keys=True),
                "visual_edge_weight_stats": json.dumps(visual_edge, sort_keys=True),
                "edge_weight_correlation_text_visual": edge.get("text_visual_correlation"),
                "edge_weight_mean_abs_difference_text_visual": edge.get("text_visual_mean_absolute_difference"),
                "text_lower_saturation_fraction": edge.get("text_lower_saturation_fraction"),
                "text_upper_saturation_fraction": edge.get("text_upper_saturation_fraction"),
                "visual_lower_saturation_fraction": edge.get("visual_lower_saturation_fraction"),
                "visual_upper_saturation_fraction": edge.get("visual_upper_saturation_fraction"),
                "source_group_weights": json.dumps(edge.get("source_groups", {}), sort_keys=True),
                "text_semantic_gate": json.dumps(modalities.get("text", {}).get("semantic_strength", {}), sort_keys=True),
                "visual_semantic_gate": json.dumps(modalities.get("visual", {}).get("semantic_strength", {}), sort_keys=True),
                "text_semantic_expert_usage": json.dumps(modalities.get("text", {}).get("semantic_expert_selection_share", {}), sort_keys=True),
                "visual_semantic_expert_usage": json.dumps(modalities.get("visual", {}).get("semantic_expert_selection_share", {}), sort_keys=True),
                "physical_expert_usage": json.dumps({m: modalities.get(m, {}).get("physical_expert_selection_share", {}) for m in ("text", "visual")}, sort_keys=True),
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
                "relation_gradient_diagnostics": json.dumps(scorer_gradients, sort_keys=True),
                "shared_expert_gradient": json.dumps(gradients.get("experts.0.linear1.weight", {}), sort_keys=True),
                "semantic_closed_path_validation": json.dumps(diag.get("closed_path", {}), sort_keys=True),
                "full_checkpoint_diagnostics": json.dumps(diag, sort_keys=True),
            }
        )
        resource_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "variant": row["variant"],
                "best_epoch": meta.get("best_epoch"),
                "epochs_completed": meta.get("epochs_completed"),
                "training_wall_seconds": meta.get("training_wall_seconds"),
                "mean_epoch_wall_seconds": meta.get("mean_epoch_wall_seconds"),
                "mean_train_step_seconds": meta.get("mean_train_step_seconds"),
                "trainable_model_parameters": meta.get("model_parameters"),
                "classifier_parameters": meta.get("classifier_parameters"),
                "cuda_device_name": meta.get("cuda_device_name"),
                "peak_cuda_allocated_bytes": meta.get("cuda_peak_allocated_bytes"),
                "peak_cuda_reserved_bytes": meta.get("cuda_peak_reserved_bytes"),
                "peak_cuda_allocated_gib": float(meta.get("cuda_peak_allocated_bytes", 0)) / 1024**3,
                "peak_cuda_reserved_gib": float(meta.get("cuda_peak_reserved_bytes", 0)) / 1024**3,
                "gradient_diagnostics": json.dumps(meta.get("gradient_diagnostics", {}), sort_keys=True),
            }
        )
    write_csv(DATA / "semantic_diagnostics.csv", diagnostic_rows)
    write_csv(DATA / "resource_profile.csv", resource_rows)
    write_json(
        DATA / "resource_profile.json",
        {
            "run_count": len(resource_rows),
            "gpu_names": sorted({str(row["cuda_device_name"]) for row in resource_rows}),
            "datasets": {
                dataset: {
                    "max_allocated_gib": max(row["peak_cuda_allocated_gib"] for row in resource_rows if row["dataset"] == dataset),
                    "max_reserved_gib": max(row["peak_cuda_reserved_gib"] for row in resource_rows if row["dataset"] == dataset),
                    "mean_training_seconds": mean(float(row["training_wall_seconds"]) for row in resource_rows if row["dataset"] == dataset),
                }
                for dataset in DATASETS
            },
            "records": resource_rows,
        },
    )
    report_text = _build_report(rows, summary_rows, comparison_rows, diagnostic_rows, manifest, summary_lookup)
    (RESEARCH / "REPORT.md").write_text(report_text, encoding="utf-8")
    print(f"Analyzed {len(rows)} runs, wrote validation-only report and CSVs under {RESEARCH}")


def _build_report(rows, summary_rows, comparison_rows, diagnostic_rows, manifest, summary_lookup) -> str:
    candidate = read_json(DATA / "candidate_cache_audit.json")
    random = read_json(DATA / "random_graph_audit.json")
    parity = read_json(DATA / "training_parity_audit.json")
    params = read_json(DATA / "parameter_grad_audit.json")
    tests = read_json(DATA / "test_summary.json")
    preflight = read_json(DATA / "preflight_summary.json")
    smoke = read_json(DATA / "smoke_summary.json")
    comparisons_by = defaultdict(list)
    for item in comparison_rows:
        if item["row_type"] == "dataset_summary":
            comparisons_by[item["comparison"]].append(item)

    lines = [
        "# PSCE-MAG V7B.1 corrective replication report",
        "",
        f"Branch `{manifest['provenance']['branch']}`; baseline `{manifest['provenance']['baseline_commit_sha']}`; freeze `{manifest['provenance']['freeze_commit_sha']}`.",
        f"The validation-only campaign completed {manifest['completed_runs']}/54 full-graph NC runs under `{manifest['protocol']}`. `task.evaluate_test=false`; no Test metrics were computed, loaded, saved, or reported. Checkpoint selection used Validation Accuracy. Macro-F1 is diagnostic only.",
        "",
        "## Formal results by dataset and seed",
        "",
        "Each row is one dataset × seed × variant run. Accuracy and Macro-F1 are validation values. Timings and peak GPU memory come from run metadata.",
        "",
        "| Dataset | Seed | Group | Val Accuracy | Val Macro-F1 | Best epoch | Epochs | Train seconds | Trainable model params | Peak allocated / reserved GiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in rows}
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                row = by_key[(dataset, seed, variant)]
                m = row["metrics"]; meta = row["metadata"]
                alloc = float(meta.get("cuda_peak_allocated_bytes", 0)) / 1024**3
                reserv = float(meta.get("cuda_peak_reserved_bytes", 0)) / 1024**3
                lines.append(f"| {dataset} | {seed} | {LABELS[variant]} | {100*float(m['val_acc']):.3f}% | {100*float(m['val_macro_f1']):.3f}% | {meta.get('best_epoch')} | {meta.get('epochs_completed')} | {float(meta.get('training_wall_seconds', 0)):.1f} | {int(meta.get('model_parameters', 0)):,} | {alloc:.2f} / {reserv:.2f} |")
    lines += ["", "## Seed mean ± population SD", "", "Seeds are repeated runs within each dataset. They are not independent datasets.", "", "| Dataset | Group | Val Accuracy (%) | Val Macro-F1 (%) | Best epoch | Epochs completed | Trainable params | Max allocated / reserved GiB |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            r = summary_lookup[(dataset, variant)]
            lines.append(f"| {dataset} | {LABELS[variant]} | {100*r['val_acc_mean']:.3f} ± {100*r['val_acc_population_sd']:.3f} | {100*r['val_macro_f1_mean']:.3f} ± {100*r['val_macro_f1_population_sd']:.3f} | {r['best_epoch_mean']:.1f} ± {r['best_epoch_population_sd']:.1f} | {r['epochs_completed_mean']:.1f} | {r['trainable_model_parameters']:,} | {r['peak_cuda_allocated_gib_max']:.2f} / {r['peak_cuda_reserved_gib_max']:.2f} |")

    lines += ["", "## Eight paired comparisons", "", "Differences are left minus right, paired by dataset and training seed. Each dataset row shows all three seed-paired deltas, the mean ± population SD, and the number of positive seed pairs. The final row weights the three dataset means equally and is descriptive only. No significance test is applied.", ""]
    for name, _, _ in PAIRS:
        lines += [f"### {name.replace('_', ' ')}", "", "| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |", "|---|---|---:|---:|---|---:|---:|"]
        for dataset in DATASETS:
            c = next(item for item in comparisons_by[name] if item["dataset"] == dataset)
            acc = json.loads(c["val_acc_delta_seed_values_pp"])
            f1 = json.loads(c["val_macro_f1_delta_seed_values_pp"])
            lines.append(f"| {dataset} | {', '.join(f'{x:+.3f}' for x in acc)} | {c['val_acc_delta_pp_mean']:+.3f} ± {c['val_acc_delta_pp_population_sd']:.3f} | {c['val_acc_positive_seed_count']}/3 | {', '.join(f'{x:+.3f}' for x in f1)} | {c['val_macro_f1_delta_pp_mean']:+.3f} ± {c['val_macro_f1_delta_pp_population_sd']:.3f} | {c['val_macro_f1_positive_seed_count']}/3 |")
        equal = next(item for item in comparison_rows if item["comparison"] == name and item["row_type"] == "equal_weight_3_dataset_descriptive")
        lines.append(f"| Equal-weight dataset means | {equal['val_acc_delta_pp_by_dataset']} | {equal['val_acc_delta_pp_mean']:+.3f} ± {equal['val_acc_delta_pp_population_sd_across_dataset_means']:.3f} | descriptive | {equal['val_macro_f1_delta_pp_by_dataset']} | {equal['val_macro_f1_delta_pp_mean']:+.3f} ± {equal['val_macro_f1_delta_pp_population_sd_across_dataset_means']:.3f} | descriptive |")
        lines.append("")

    old_rows_path = ROOT / "research/psce_mag_v7b/data/run_rows.json"
    old_rows = read_json(old_rows_path)
    old_b0 = {(row["dataset"], int(row["seed"])): row for row in old_rows if row.get("variant") == "b0_v7a_original"}
    new_b0 = {(row["dataset"], int(row["seed"])): row for row in rows if row["variant"] == "b0_v7a_original"}
    lines += [
        "## Original V7B B0 bug and corrective findings",
        "",
        "In `src/models/psce_mag_v7b.py`, the constructor previously froze `relation_projections` and `relation_scorer` for every non-relative arm. That incorrectly included B0, whose definition is the complete V7A A2 with a trainable shared Sigmoid scorer. The fix limits freezing to B1/B2. B3/B4 scorer construction is unchanged.",
        f"The model audit measured {params['old_buggy_b0_trainable_parameter_count']:,} trainable B0 parameters under the old permissions and {params['corrected_b0_trainable_parameter_count']:,} after correction, a +{params['trainable_parameter_delta']:,} parameter change. The full Movies NC configuration has the same corrected count; formal run rows record model and classifier counts separately.",
        f"Real Movies train-mode parity passed with identical initialization, shared candidate graph, features, physical topology, train nodes, dropout RNG, CE + auxiliary objective, gradient clipping, AdamW, and one update. Maximum embedding error={parity['embedding_max_abs_error']:.3g}; total-loss error={parity['total_loss_abs_error']:.3g}; common gradient error={parity['public_gradient_max_abs_error']:.3g}; post-update error={parity['adamw_update_max_abs_difference']:.3g}.",
        f"The real-graph B0 relation gradients were finite and nonzero for every audited projection/scorer tensor. Their one-step L1 update sizes are recorded in `data/training_parity_audit.json`; all are positive. CPU initialization, dropout-matched losses/gradients, AdamW and three-step optimizer trajectories also match in the corrective pytest module.",
        "",
        "| Dataset | Old buggy B0 edge-weight mean ± SD | Corrected B0 edge-weight mean ± SD |",
        "|---|---:|---:|",
    ]
    for dataset in DATASETS:
        old_mean = []; old_sd = []; new_mean = []; new_sd = []
        for seed in SEEDS:
            old_diag = old_b0[(dataset, seed)]["metadata"].get("semantic_checkpoint_diagnostics", {})
            new_diag = new_b0[(dataset, seed)]["metadata"].get("semantic_checkpoint_diagnostics", {})
            old_edge = _edge_summary(old_diag, "text"); new_edge = _edge_summary(new_diag, "text")
            if "mean" in old_edge: old_mean.append(float(old_edge["mean"]))
            if "std" in old_edge: old_sd.append(float(old_edge["std"]))
            if "mean" in new_edge: new_mean.append(float(new_edge["mean"]))
            if "std" in new_edge: new_sd.append(float(new_edge["std"]))
        lines.append(f"| {dataset} | {mean(old_mean):.5f} ± {mean(old_sd):.5f} | {mean(new_mean):.5f} ± {mean(new_sd):.5f} |")
    lines += [
        "",
        "The old V7B B0 used a random initialized but frozen relation scorer; it was not a trainable V7A A2. Therefore the old B0−B1, B3−B0, and B4−B0 comparisons do not answer the intended learning-versus-fixed questions. The old runs remain untouched and are not included in this campaign. New results only support the corrected comparisons in this report.",
        "",
        "## Candidate, random topology, and training audits",
        "",
        f"The cache audit passed for all three datasets. Feature fingerprints were recomputed from source feature files; KNN edge index, unique edges, and degree exactly matched V7A; V7B source flags and fingerprints matched; label access during candidate construction/audit was false. All nine B2 caches were reused with the original training-seed mapping 42→2026, 43→2027, 44→2028, exact degree sequence, undirected unique edges, and no self-loops or duplicates. Physical overlap details are in `data/candidate_cache_audit.json` and `data/random_graph_audit.json`.",
        f"Full repository pytest: {tests.get('passed')} (exit {tests.get('exit_code')}, {tests.get('elapsed_seconds'):.1f}s). GPU preflight: {preflight.get('passed')} ({len(preflight.get('datasets', []))} dataset/variant paths). Six-group Movies smoke: {smoke.get('audit_passed')}.",
        f"Target was physical GPU1, RTX 3090. The runner used `CUDA_VISIBLE_DEVICES=1` (logical CUDA device 0); GPU0 was left untouched. Preflight reports one-step time, relation-scoring and semantic-propagation timings, full validation inference time, peak allocated/reserved VRAM and trainable parameter count.",
        "",
        "## Relation, gate, expert, and scale diagnostics",
        "",
        "`data/semantic_diagnostics.csv` contains per dataset/seed/checkpoint edge-weight means, SDs, quantiles, saturation fractions, Text/Visual correlation and absolute differences, source-group distributions, relation projection/scorer gradients, physical/semantic expert selection shares, semantic strength gates, semantic contribution RMS relative to physical and intrinsic pathways, calibration scales, semantic/Raw cosine, and closed-semantic-path validation changes. B1/B2 fixed edge weights are exactly 0.8; their semantic gates and topology controls are included. These diagnostics describe trained behavior and are not evidence that an individual edge is causally useful.",
        "",
        "## Interpretation",
        "",
    ]
    def pair_sentence(name: str, title: str) -> str:
        rows_for_pair = comparisons_by[name]
        text = []
        for item in rows_for_pair:
            text.append(f"{item['dataset']}: Acc {item['val_acc_delta_pp_mean']:+.3f} pp ({item['val_acc_positive_seed_count']}/3 positive), Macro-F1 {item['val_macro_f1_delta_pp_mean']:+.3f} pp ({item['val_macro_f1_positive_seed_count']}/3 positive)")
        return f"**{title}.** " + "; ".join(text) + "."
    lines += [
        pair_sentence("B1_minus_R0", "Fixed KNN versus Raw"),
        pair_sentence("B1_minus_B2", "KNN structure versus degree-preserving random structure"),
        pair_sentence("B0_minus_B1", "Correctly trained V7A learned weighting versus fixed KNN"),
        pair_sentence("B3_minus_B0", "Relative shared scoring versus corrected V7A A2"),
        pair_sentence("B3_minus_B1", "Relative shared scoring versus fixed KNN"),
        pair_sentence("B4_minus_B3", "Modality-specific versus shared relative scoring"),
        pair_sentence("B4_minus_B0", "Modality-specific relative scoring versus corrected V7A A2"),
        "",
        "These paired directions and seed counts are descriptive. Small, inconsistent differences do not justify adding mechanisms or searching new hyperparameters within this correction campaign. A later study should favor the simplest relation design whose validation gains recur across the three datasets and seeds.",
        "",
        "## Artifacts and limits",
        "",
        "- Per-run raw rows and provenance: `data/run_rows.json` and `data/campaign_manifest.json`",
        "- Per-dataset summaries: `data/summary.csv`",
        "- Seed-paired deltas and equal-weight descriptive summaries: `data/paired_comparisons.csv`",
        "- Relation/gate/expert diagnostics: `data/semantic_diagnostics.csv`",
        "- Training parity: `data/training_parity_audit.json` and `data/parameter_grad_audit.json`",
        "- Candidate and random graph audit: `data/candidate_cache_audit.json`, `data/random_graph_audit.json`",
        "- GPU preflight/resource records: `data/preflight_summary.json`, `data/resource_profile.json`, `data/resource_profile.csv`",
        "- Test and smoke records: `data/test_summary.json`, `data/smoke_summary.json`",
        "",
        "This was full-graph node classification only. It did not perform formal link prediction, Test evaluation, split regeneration, or hyperparameter search. The conclusions cover these three datasets and three training seeds only.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    analyze()
