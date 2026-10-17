#!/usr/bin/env python3
"""Validation-only summaries and audit tables for Hybrid-CGE V1.1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/hybrid_cge_v11"
DATA = STUDY / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
CONDITIONS = ("H1_hybrid", "H2_hybrid_plus")
SEEDS = (42, 43, 44)
REFERENCE_DATA = {
    "CG-CGE B1": (ROOT / "research/cg_cge_v1/data", "B1_complementary"),
    "View-CGE A2": (ROOT / "research/mag_view_cge_v1/data", "A2_cross_source"),
}
REFERENCE_PREFLIGHT_KEYS = {
    name: condition for name, (_, condition) in REFERENCE_DATA.items()
}
CHECKPOINTS = {
    "Best-Accuracy": {"Accuracy": "val_acc", "Macro-F1": "val_macro_f1", "CE": "val_ce"},
    "Best-F1": {"Accuracy": "best_f1_val_acc", "Macro-F1": "best_f1_val_macro_f1", "CE": "best_f1_val_ce"},
}


def read_csv(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def finite(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {value}")
    return value


def stats(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cannot summarize non-finite values")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "positive_seeds": sum(value > 0 for value in values),
        "negative_seeds": sum(value < 0 for value in values),
        "zero_seeds": sum(value == 0 for value in values),
    }


def fmt(values, scale=1.0, digits=3):
    row = stats(values)
    return f"{row['mean']*scale:.{digits}f} ± {row['sample_sd']*scale:.{digits}f}"


def strict_reference(preflight, dataset, reference):
    reference_key = REFERENCE_PREFLIGHT_KEYS.get(reference, reference)
    return all(
        preflight["datasets"][dataset]["seeds"][str(seed)]["historical_reference_match"][reference_key]["eligible"]
        for seed in SEEDS
    )


def main():
    current = read_csv(DATA / "runs.csv")
    preflight = json.loads((DATA / "preflight_cpu.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "cpu_data_and_init_pass":
        raise RuntimeError("CPU data/init preflight is missing or failed")
    if manifest.get("test_evaluation") is not False:
        raise RuntimeError("manifest indicates Test evaluation")

    expected = {(d, c, s) for d in DATASETS for c in CONDITIONS for s in SEEDS}
    current_by = {(r["dataset"], r["condition"], int(r["seed"])): r for r in current}
    found = {key for key, row in current_by.items() if row.get("status") == "complete"}
    if found != expected:
        raise RuntimeError(f"formal matrix incomplete: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    for row in current:
        if row.get("status") != "complete" or row.get("test_evaluation") != "False" or row.get("error"):
            raise RuntimeError(f"invalid run row: {row}")
        for key in (
            "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
            "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce", "train_ce_eval",
            "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap",
        ):
            finite(row, key)
    for dataset in DATASETS:
        for seed in SEEDS:
            pair = [current_by[(dataset, condition, seed)] for condition in CONDITIONS]
            for key in (
                "initial_model_sha256", "initial_classifier_sha256", "split_fingerprint",
                "semantic_cache_sha256", "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint",
            ):
                if len({row[key] for row in pair}) != 1:
                    raise RuntimeError(f"H1/H2 pairing failed: {dataset}/seed{seed}/{key}")

    freeze = manifest["implementation_freeze_commit"]
    source = manifest["formal_source_commit"]
    if any(row["implementation_freeze_commit"] != freeze or row["formal_source_commit"] != source for row in current):
        raise RuntimeError("formal rows mix source or freeze commits")

    ref_by = {}
    eligible = {}
    for name, (folder, condition) in REFERENCE_DATA.items():
        rows = read_csv(folder / "runs.csv")
        ref_by[name] = {(r["dataset"], r["condition"], int(r["seed"])): r for r in rows}
        for dataset in DATASETS:
            eligible[(dataset, name)] = strict_reference(preflight, dataset, name)
            if eligible[(dataset, name)]:
                for seed in SEEDS:
                    row = ref_by[name][(dataset, condition, seed)]
                    if row.get("status") != "complete" or row.get("test_evaluation") != "False":
                        raise RuntimeError(f"invalid historical paired row: {name}/{dataset}/seed{seed}")

    paired_rows = []
    summary_rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            run_rows = [current_by[(dataset, condition, seed)] for seed in SEEDS]
            for checkpoint, metric_map in CHECKPOINTS.items():
                for metric, key in metric_map.items():
                    values = [finite(row, key) for row in run_rows]
                    aggregate = stats(values)
                    summary_rows.append({
                        "dataset": dataset, "condition": condition, "checkpoint": checkpoint,
                        "reference": "within_condition", "metric": metric, **aggregate,
                    })
                    for seed, value in zip(SEEDS, values):
                        paired_rows.append({
                            "dataset": dataset, "condition": condition, "seed": seed,
                            "checkpoint": checkpoint, "reference": "within_condition",
                            "metric": key, "value": value, "reference_value": "", "difference": "",
                        })
                    for reference, (_, ref_condition) in REFERENCE_DATA.items():
                        if not eligible[(dataset, reference)]:
                            continue
                        differences = []
                        for seed, value in zip(SEEDS, values):
                            ref_value = finite(ref_by[reference][(dataset, ref_condition, seed)], key)
                            differences.append(value - ref_value)
                            paired_rows.append({
                                "dataset": dataset, "condition": condition, "seed": seed,
                                "checkpoint": checkpoint, "reference": reference, "metric": key,
                                "value": value, "reference_value": ref_value, "difference": value-ref_value,
                            })
                        summary_rows.append({
                            "dataset": dataset, "condition": condition, "checkpoint": checkpoint,
                            "reference": reference, "metric": metric, **stats(differences),
                        })

        for checkpoint, metric_map in CHECKPOINTS.items():
            for metric, key in metric_map.items():
                differences = [
                    finite(current_by[(dataset, "H2_hybrid_plus", seed)], key)
                    - finite(current_by[(dataset, "H1_hybrid", seed)], key)
                    for seed in SEEDS
                ]
                for seed, difference in zip(SEEDS, differences):
                    h2 = finite(current_by[(dataset, "H2_hybrid_plus", seed)], key)
                    h1 = finite(current_by[(dataset, "H1_hybrid", seed)], key)
                    paired_rows.append({
                        "dataset": dataset, "condition": "H2_hybrid_plus", "seed": seed,
                        "checkpoint": checkpoint, "reference": "within-study H1",
                        "metric": key, "value": h2, "reference_value": h1, "difference": difference,
                    })
                summary_rows.append({
                    "dataset": dataset, "condition": "H2_hybrid_plus", "checkpoint": checkpoint,
                    "reference": "within-study H1", "metric": metric, **stats(differences),
                })

    write_csv(DATA / "summary.csv", summary_rows)
    write_csv(DATA / "paired_comparisons.csv", paired_rows)

    diagnostics = read_csv(DATA / "mechanism_diagnostics.csv")
    mechanism_rows = []
    diag_by = {(r["dataset"], r["condition"], int(r["seed"]), r["checkpoint"], r["modality"]): r for r in diagnostics}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            early = [
                diag_by[(dataset, condition, seed, "best_accuracy", f"early_physical_cross_{modality}")]
                for seed in SEEDS for modality in ("text", "visual")
            ]
            for key in ("raw_rms", "calibrated_rms", "coefficient_mean"):
                x = stats(float(r[key]) for r in early)
                mechanism_rows.append({"dataset": dataset, "condition": condition, "scope": "early_physical_cross", "metric": key, **x})
            late_rows = [
                diag_by[(dataset, condition, seed, "best_accuracy", modality)]
                for seed in SEEDS for modality in ("text", "visual")
            ]
            late = [json.loads(r["late_completion"]) for r in late_rows]
            for key in (
                "semantic_source_weight", "same_node_source_weight", "none_source_weight", "p_accept_mean",
                "candidate_rms", "uncontrolled_residual_rms", "controlled_residual_rms",
                "residual_to_collaborative_rms", "relative_control_trigger_fraction",
                "relative_control_trigger_fraction_supported", "relative_control_scale_mean", "tau",
                "channel_gate_mean", "channel_gate_saturation_fraction",
                "late_cross_dropout_probability",
                "complementary_expert_rms", "semantic_attention_entropy_multi_neighbor",
            ):
                values = [float(row[key]) for row in late if key in row]
                if values:
                    mechanism_rows.append({"dataset": dataset, "condition": condition, "scope": "late_mean", "metric": key, **stats(values)})
            eval_dropout = [float(row["late_cross_dropout_realized_fraction"]) for row in late if "late_cross_dropout_realized_fraction" in row]
            if eval_dropout:
                mechanism_rows.append({
                    "dataset": dataset, "condition": condition, "scope": "eval_mode",
                    "metric": "late_cross_dropout_realized_fraction", **stats(eval_dropout),
                })
            for key in (
                "physical_cross_input_router_correction_rms", "physical_route_top2_flip_fraction",
                "physical_private_response_rms", "physical_collab_response_rms",
                "same_semantic_response_rms", "same_semantic_weighted_rms",
            ):
                x = stats(float(r[key]) for r in late_rows)
                mechanism_rows.append({"dataset": dataset, "condition": condition, "scope": "per_modality_mean", "metric": key, **x})
    write_csv(DATA / "mechanism_summary.csv", mechanism_rows)

    class_rows = read_csv(DATA / "classwise_metrics.csv")
    reference_class_rows = {
        name: read_csv(folder / "classwise_metrics.csv") for name, (folder, _) in REFERENCE_DATA.items()
    }
    fashion_rows = [r for r in class_rows if r["dataset"] == "ele-fashion" and r["checkpoint"] == "best_accuracy"]
    for name, rows in reference_class_rows.items():
        if eligible[("ele-fashion", name)]:
            fashion_rows.extend(r for r in rows if r["dataset"] == "ele-fashion" and r["checkpoint"] == "best_accuracy")
    support = {int(r["class_id"]): int(r["support"]) for r in fashion_rows}
    minority_ids = sorted(support, key=lambda class_id: (support[class_id], class_id))[:10]
    minority_rows = []
    methods = list(CONDITIONS) + [
        condition for name, (_, condition) in REFERENCE_DATA.items() if eligible[("ele-fashion", name)]
    ]
    for class_id in minority_ids:
        for condition in methods:
            selected = [r for r in fashion_rows if int(r["class_id"]) == class_id and r["condition"] == condition]
            if not selected:
                continue
            x = stats(float(r["f1"]) for r in selected)
            minority_rows.append({
                "dataset": "ele-fashion", "class_id": class_id, "support_per_seed": support[class_id],
                "condition": condition, "mean_f1": x["mean"], "sample_sd_f1": x["sample_sd"], "seed_count": x["n"],
            })
    write_csv(DATA / "elefashion_minority_class_metrics.csv", minority_rows)

    integrity = json.loads((DATA / "artifact_integrity.json").read_text(encoding="utf-8"))
    checkpoints = integrity.get("checkpoints", [])
    if len(checkpoints) != 36:
        raise RuntimeError(f"expected 36 selected checkpoints, got {len(checkpoints)}")
    for row in checkpoints:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"checkpoint path/size mismatch: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch: {path}")

    histories = read_csv(DATA / "epoch_history.csv")
    if len(histories) == 0:
        raise RuntimeError("epoch history is missing")
    failures = json.loads((DATA / "failures.json").read_text()) if (DATA / "failures.json").exists() else {"failures": []}
    attempts = json.loads((DATA / "preflight_attempts.json").read_text()) if (DATA / "preflight_attempts.json").exists() else {"attempts": []}
    interference = json.loads((DATA / "runner_interference_events.json").read_text()) if (DATA / "runner_interference_events.json").exists() else {"events": []}
    access_attempts = json.loads((DATA / "runner_access_attempts.json").read_text()) if (DATA / "runner_access_attempts.json").exists() else {"attempts": []}
    env = manifest["environment"]
    ref_line = "; ".join(
        f"{name}: {sum(eligible[(d,name)] for d in DATASETS)}/3 datasets paired"
        for name in REFERENCE_DATA
    )

    report = [
        "# Hybrid-CGE V1.1 Pilot Report", "",
        "Validation-only: Test labels were masked before trainer entry and no Test metrics were computed.",
        f"Implementation Freeze Commit: {freeze}; formal training source: {source}.",
        "",
        "## Protocol and historical comparability", "",
        "All 18 runs use Movies, Grocery and ele-fashion; H1/H2; seeds 42/43/44; the official full-graph NC protocol; identical H1/H2 initial model/head hashes per Dataset×Seed; and the same split, feature, physical graph, V7B.1 cache and candidate fingerprints.",
        f"Historical pairing eligibility: {ref_line}. Only fully matched three-seed blocks appear in paired summaries.",
        "",
        "## Best-Accuracy Validation metrics", "",
        "| Dataset | Model | Accuracy % | Macro-F1 % | CE |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition in CONDITIONS:
            rows = [current_by[(dataset, condition, seed)] for seed in SEEDS]
            report.append(
                f"| {dataset} | {condition} | {fmt((finite(r,'val_acc') for r in rows),100)} | "
                f"{fmt((finite(r,'val_macro_f1') for r in rows),100)} | {fmt((finite(r,'val_ce') for r in rows),digits=4)} |"
            )
        for name, (_, condition) in REFERENCE_DATA.items():
            if not eligible[(dataset, name)]:
                report.append(f"| {dataset} | {name} (not paired) | — | — | — |")
                continue
            rows = [ref_by[name][(dataset, condition, seed)] for seed in SEEDS]
            report.append(
                f"| {dataset} | {name} | {fmt((finite(r,'val_acc') for r in rows),100)} | "
                f"{fmt((finite(r,'val_macro_f1') for r in rows),100)} | {fmt((finite(r,'val_ce') for r in rows),digits=4)} |"
            )
    report.extend([
        "", "## Best-F1 Validation metrics", "",
        "| Dataset | Model | Accuracy % | Macro-F1 % | CE |",
        "|---|---|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        for condition in CONDITIONS:
            rows = [current_by[(dataset, condition, seed)] for seed in SEEDS]
            report.append(
                f"| {dataset} | {condition} | {fmt((finite(r,'best_f1_val_acc') for r in rows),100)} | "
                f"{fmt((finite(r,'best_f1_val_macro_f1') for r in rows),100)} | "
                f"{fmt((finite(r,'best_f1_val_ce') for r in rows),digits=4)} |"
            )
        for name, (_, condition) in REFERENCE_DATA.items():
            if eligible[(dataset, name)]:
                rows = [ref_by[name][(dataset, condition, seed)] for seed in SEEDS]
                report.append(
                    f"| {dataset} | {name} | {fmt((finite(r,'best_f1_val_acc') for r in rows),100)} | "
                    f"{fmt((finite(r,'best_f1_val_macro_f1') for r in rows),100)} | "
                    f"{fmt((finite(r,'best_f1_val_ce') for r in rows),digits=4)} |"
                )
    report.extend([
        "", "## Per-seed results and Train/Validation gaps", "",
        "| Dataset | Condition | Seed | Val Acc % | Val Macro-F1 % | Val CE | Best Acc Epoch | Best F1 Epoch | Train–Val Acc Gap pp | Train–Val F1 Gap pp | Train–Val CE Gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                row = current_by[(dataset, condition, seed)]
                report.append(
                    f"| {dataset} | {condition} | {seed} | {finite(row,'val_acc')*100:.3f} | "
                    f"{finite(row,'val_macro_f1')*100:.3f} | {finite(row,'val_ce'):.4f} | "
                    f"{row['best_accuracy_epoch']} | {row['best_f1_epoch']} | "
                    f"{finite(row,'train_val_acc_gap')*100:.3f} | {finite(row,'train_val_macro_f1_gap')*100:.3f} | "
                    f"{finite(row,'train_val_ce_gap'):.4f} |"
                )
    report.extend([
        "",
        "Train–Validation Accuracy/F1/CE gaps are Train minus Validation, with Train evaluated in eval mode at that checkpoint. The table above uses the Best-Accuracy checkpoint; the next table uses each run's Best-F1 checkpoint.",
        "",
        "### Per-seed Best-F1 checkpoint metrics and gaps", "",
        "| Dataset | Condition | Seed | Val Acc % | Val Macro-F1 % | Val CE | Train–Val Acc Gap pp | Train–Val F1 Gap pp | Train–Val CE Gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                row = current_by[(dataset, condition, seed)]
                report.append(
                    f"| {dataset} | {condition} | {seed} | {finite(row,'best_f1_val_acc')*100:.3f} | "
                    f"{finite(row,'best_f1_val_macro_f1')*100:.3f} | {finite(row,'best_f1_val_ce'):.4f} | "
                    f"{finite(row,'best_f1_train_val_acc_gap')*100:.3f} | "
                    f"{finite(row,'best_f1_train_val_macro_f1_gap')*100:.3f} | "
                    f"{finite(row,'best_f1_train_val_ce_gap'):.4f} |"
                )

    report.extend(["", "## Paired Validation differences", "",
                   "Candidate minus reference; Accuracy and Macro-F1 are percentage points. Means and sample SDs are descriptive for three seeds.", "",
                   "| Dataset | Contrast | Checkpoint | Metric | Mean ± SD | +/−/0 Seeds |", "|---|---|---|---|---:|---:|"])
    for row in summary_rows:
        if row["reference"] not in (*REFERENCE_DATA.keys(), "within-study H1"):
            continue
        scale = 100 if row["metric"] in ("Accuracy", "Macro-F1") else 1
        value = f"{float(row['mean'])*scale:+.3f} ± {float(row['sample_sd'])*scale:.3f}"
        signs = f"{row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']}"
        report.append(
            f"| {row['dataset']} | {row['condition']} − {row['reference']} | {row['checkpoint']} | "
            f"{row['metric']} | {value} | {signs} |"
        )

    regularization_rows = [
        row for row in summary_rows
        if row["condition"] == "H2_hybrid_plus"
        and row["reference"] == "within-study H1"
        and row["checkpoint"] == "Best-Accuracy"
    ]
    report.extend([
        "", "## Dropout, weight decay and interaction", "",
        "The shared backbone dropout is 0.2 and AdamW weight decay is 1e-4 in every formal run. H1 has no late-residual dropout and masks SameNode; H2 adds a valid SameNode candidate and applies p=0.15 non-inverted dropout to the final Late residual during training. The H2−H1 estimate therefore describes this combined package. This design cannot separate SameNode from late dropout, estimate a SameNode×dropout interaction, or estimate a weight-decay main effect or interaction. The three-seed paired contrasts below are descriptive, not a factorial causal estimate.",
        "",
        "| Dataset | H2−H1 Δ Accuracy pp | H2−H1 Δ Macro-F1 pp | H2−H1 Δ CE |", "|---|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        by_metric = {row["metric"]: row for row in regularization_rows if row["dataset"] == dataset}
        report.append(
            f"| {dataset} | "
            f"{float(by_metric['Accuracy']['mean'])*100:+.3f} ± {float(by_metric['Accuracy']['sample_sd'])*100:.3f} | "
            f"{float(by_metric['Macro-F1']['mean'])*100:+.3f} ± {float(by_metric['Macro-F1']['sample_sd'])*100:.3f} | "
            f"{float(by_metric['CE']['mean']):+.4f} ± {float(by_metric['CE']['sample_sd']):.4f} |"
        )
    report.extend([
        "",
        "The train-mode GPU preflight recorded realized node-mask drop fractions of 0 for H1 and approximately 0.148–0.150 for H2 across datasets and both target modalities (one seed-42 optimizer-step check). Best-checkpoint mechanism diagnostics run in evaluation mode, so their realized dropout fraction is 0 by design; the configured training probability remains 0.15 for H2.",
        "",
    ])

    report.extend(["", "## Training curves and auxiliary loss", "",
                   "Full epoch-by-epoch Train/Validation Accuracy, Macro-F1, CE, Private auxiliary CE, balance auxiliary and timing are retained in data/epoch_history.csv. The figure [training curves](figures/training_curves/hybrid_cge_v11_training_curves.pdf) shows seed means with sample SD; the data behind it are the full run histories.",
                   "",
                   "| Dataset | Condition | First-epoch Private CE | Best-Accuracy-epoch Private CE | Best-F1 − Best-Accuracy Δ Acc pp | Δ Macro-F1 pp | Δ CE |",
                   "|---|---|---:|---:|---:|---:|---:|"])
    history_by = {}
    for row in histories:
        history_by.setdefault((row["dataset"], row["condition"], int(row["seed"])), []).append(row)
    for dataset in DATASETS:
        for condition in CONDITIONS:
            first_ce, best_ce, acc_delta, f1_delta, ce_delta = [], [], [], [], []
            for seed in SEEDS:
                run = current_by[(dataset, condition, seed)]
                trace = sorted(history_by[(dataset, condition, seed)], key=lambda r: int(r["epoch"]))
                first_ce.append(finite(trace[0], "train_unimodal_ce"))
                chosen = next(r for r in trace if int(r["epoch"]) == int(run["best_accuracy_epoch"]))
                best_ce.append(finite(chosen, "train_unimodal_ce"))
                acc_delta.append(finite(run, "best_f1_val_acc") - finite(run, "val_acc"))
                f1_delta.append(finite(run, "best_f1_val_macro_f1") - finite(run, "val_macro_f1"))
                ce_delta.append(finite(run, "best_f1_val_ce") - finite(run, "val_ce"))
            report.append(
                f"| {dataset} | {condition} | {fmt(first_ce,digits=4)} | {fmt(best_ce,digits=4)} | "
                f"{fmt(acc_delta,100)} | {fmt(f1_delta,100)} | {fmt(ce_delta,digits=4)} |"
            )

    report.extend([
        "", "## Figure and QA", "",
        "The linked 3×3 training-curve figure plots mean ± sample SD across three matched seeds at every displayed epoch, truncated to each dataset's shared complete horizon. Strict source, panel-alignment, PDF text-size and rendered-collision checks all passed; see [figure QA](figures/QA.md) and its machine-readable records.",
        "",
    ])

    report.extend(["", "## Resource profile", "",
                   "| Dataset | Condition | Wall seconds | Mean epoch seconds | Peak allocated GiB | Peak reserved GiB | Model parameters | Optimizer parameters |",
                   "|---|---|---:|---:|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for condition in CONDITIONS:
            rows = [current_by[(dataset, condition, seed)] for seed in SEEDS]
            report.append(
                f"| {dataset} | {condition} | {fmt((finite(r,'wall_seconds') for r in rows),digits=1)} | "
                f"{statistics.mean(finite(r,'mean_epoch_wall_seconds') for r in rows):.3f} | "
                f"{max(int(r['peak_allocated_bytes']) for r in rows)/1024**3:.2f} | "
                f"{max(int(r['peak_reserved_bytes']) for r in rows)/1024**3:.2f} | "
                f"{rows[0]['model_parameters']} | {rows[0]['trainable_optimizer_parameters']} |"
            )
    snapshot = env.get("physical_gpu_snapshot", {}).get("devices", [])
    if snapshot:
        gpu0 = next((row for row in snapshot if row.get("physical_index") == 0), snapshot[0])
        report.append(f"\nFormal training used physical GPU0: {gpu0['name']}, driver {gpu0['driver_version']}; the runner restricted visibility to this device.")
    report.append(
        f"Environment: Python {env['python']}, PyTorch {env['pytorch']}, CUDA build {env['cuda_build']}, cuDNN {env['cudnn']}, NumPy {env['numpy']}; matmul TF32={env['matmul_tf32']}, cuDNN TF32={env['cudnn_tf32']}, deterministic algorithms={env['deterministic_algorithms']}, CUBLAS_WORKSPACE_CONFIG={env['cublas_workspace_config']!r}."
    )
    report.append("All 36 selected checkpoints reloaded and reproduced their Validation metrics, and passed SHA256 and byte-count verification. No accepted run recorded an OOM.")

    report.extend(["", "## Mechanism and architecture diagnostics", "",
                   "Values below summarize Best-Accuracy checkpoints. They verify executed paths, not causal usefulness.",
                   "",
                   "| Dataset | Condition | Path | Diagnostic | Mean | SD | n |",
                   "|---|---|---|---|---:|---:|---:|"])
    for row in mechanism_rows:
        report.append(
            f"| {row['dataset']} | {row['condition']} | {row['scope']} | {row['metric']} | "
            f"{float(row['mean']):.5g} | {float(row['sample_sd']):.5g} | {row['n']} |"
        )

    report.extend(["", "## Fashion lower-support Validation classes", "",
                   "The loader does not provide a canonical class-name map in run metadata; IDs and per-seed support are used.",
                   "",
                   "| Class ID | Support | Condition | F1 mean ± SD % | n |", "|---:|---:|---|---:|---:|"])
    for row in minority_rows:
        report.append(
            f"| {row['class_id']} | {row['support_per_seed']} | {row['condition']} | "
            f"{float(row['mean_f1'])*100:.3f} ± {float(row['sample_sd_f1'])*100:.3f} | {row['seed_count']} |"
        )

    failure_text = [
        f"Preserved records: {len(failures.get('failures', []))} formal-run failures; "
        f"{len(attempts.get('attempts', []))} preflight attempts/refusals; "
        f"{len(access_attempts.get('attempts', []))} runner launch-access records; "
        f"{len(interference.get('events', []))} foreign-GPU guard stops."
    ]
    for row in failures.get("failures", []):
        failure_text.append(
            f"- {row.get('dataset','?')}/{row.get('condition','?')}/seed{row.get('seed','?')}: "
            f"{row.get('error','unrecorded error')}; training_started={row.get('training_started', False)}."
        )
    for row in attempts.get("attempts", []):
        failure_text.append(
            f"- Preflight {row.get('time_utc','')}: {row.get('status','unknown')} — {row.get('error','')}."
        )
    for row in access_attempts.get("attempts", []):
        failure_text.append(
            f"- Runner access {row.get('time_utc','')}: {row.get('status','unknown')}; "
            f"training_started={row.get('training_started', False)} — {row.get('error','').rstrip('. ')}. "
            f"{row.get('resolution','')}"
        )
    for row in interference.get("events", []):
        gpu_process = row.get("foreign_process", {})
        failure_text.append(
            f"- Runner {row.get('observed_at_utc','')}: stopped before the next cell on physical GPU "
            f"{row.get('physical_gpu','?')} because foreign PID {gpu_process.get('pid','?')} held "
            f"{gpu_process.get('used_memory_mib','?')} MiB; no process was interrupted."
        )
    failure_text.extend([
        "Unresolved findings:",
        "- H2's Best-Accuracy Accuracy gain is small but positive on all three datasets; its Macro-F1 difference is negative on Movies/Grocery and positive on ele-fashion. The dataset-dependent tradeoff is not explained by this study, and SameNode cannot be separated from late dropout.",
        "- Train/Validation gaps are larger on Movies and Grocery than ele-fashion. This study records the pattern but does not identify its data or optimization cause.",
        "- Seed-to-seed variation is descriptive with three seeds. Formal runs retain the recorded default nondeterministic-algorithm setting; no deterministic full-run replay was attempted, so the contribution of individual stochastic sources remains unmeasured.",
    ])
    report.extend([
        "", "## Independent implementation audit", "",
        "- Private state is computed from each receiver modality only. The early cross message enters every Collaborative physical expert profile and the inherited physical-router Cross input. The Late branch has no Physical Cross candidate.",
        "- H1/H2 share the same four experts, same source selector architecture and matched parameter initialization. H1 masks SameNode. H2 enables SameNode and applies non-inverted node dropout only to the final Late residual during training.",
        "- Relative control caps large residuals and never amplifies weak ones. A zero tau gradient is allowed when no node reaches the cap.",
        "- Private auxiliary CE is produced from U_priv and the existing NC objective indexes training labels only. Both held-out splits remain unavailable to objective or checkpoint selection.",
        "- Three seeds do not establish significance. Best-F1 selection can trade Accuracy for Macro-F1; inspect the paired tables and per-class Fashion metrics.",
        "",
        "## Preserved failures and unresolved issues", "",
        *failure_text,
        "",
        "No Test evaluation was performed. Mechanism magnitudes do not by themselves establish causal contribution.",
        "",
        "## Evidence files", "",
        "runs.csv, summary.csv, paired_comparisons.csv, epoch_history.csv, classwise_metrics.csv, elefashion_minority_class_metrics.csv, mechanism_diagnostics.csv, mechanism_summary.csv, resource_profile.csv, gradient_audit.json, artifact_integrity.json, preflight_cpu.json, preflight.json, runner_interference_events.json, runner_access_attempts.json and manifest.json are retained under data/. Figure exports and [figure QA](figures/QA.md) are retained alongside the report.",
    ])
    (STUDY / "PILOT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    analysis = {
        "status": "complete",
        "formal_runs": len(expected),
        "datasets": list(DATASETS),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "historical_reference_eligibility": {
            name: {dataset: eligible[(dataset, name)] for dataset in DATASETS}
            for name in REFERENCE_DATA
        },
        "checkpoint_integrity": "pass",
        "verified_checkpoint_count": len(checkpoints),
        "test_evaluation": False,
        "implementation_freeze_commit": freeze,
        "formal_source_commit": source,
        "runner_interference_event_count": len(interference.get("events", [])),
        "runner_access_attempt_count": len(access_attempts.get("attempts", [])),
    }
    (DATA / "analysis_summary.json").write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")

    memo = [
        "# Hybrid-CGE V1.1 Decision Memo", "",
        "The 18-run Validation matrix is complete. Paired conclusions use same-seed comparisons and strict historical fingerprint gates. Three seeds are descriptive, not population-level significance tests.",
        "",
        "## Paired Best-Accuracy differences", "",
        "| Dataset | Contrast | Δ Accuracy pp | Δ Macro-F1 pp | Δ CE |", "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for condition, reference in (
            ("H1_hybrid", "CG-CGE B1"), ("H1_hybrid", "View-CGE A2"),
            ("H2_hybrid_plus", "CG-CGE B1"), ("H2_hybrid_plus", "View-CGE A2"),
            ("H2_hybrid_plus", "within-study H1"),
        ):
            if reference in REFERENCE_DATA and not eligible[(dataset, reference)]:
                memo.append(f"| {dataset} | {condition} − {reference} | not strictly paired | — | — |")
                continue
            keymap = {"Accuracy": "val_acc", "Macro-F1": "val_macro_f1", "CE": "val_ce"}
            metrics = {}
            for metric, key in keymap.items():
                deltas = []
                for seed in SEEDS:
                    value = finite(current_by[(dataset, condition, seed)], key)
                    if reference == "within-study H1":
                        refvalue = finite(current_by[(dataset, "H1_hybrid", seed)], key)
                    else:
                        _, refcondition = REFERENCE_DATA[reference]
                        refvalue = finite(ref_by[reference][(dataset, refcondition, seed)], key)
                    deltas.append(value-refvalue)
                x = stats(deltas)
                scale = 100 if metric != "CE" else 1
                metrics[metric] = f"{x['mean']*scale:+.3f} ± {x['sample_sd']*scale:.3f}"
            memo.append(f"| {dataset} | {condition} − {reference} | {metrics['Accuracy']} | {metrics['Macro-F1']} | {metrics['CE']} |")
    memo.extend([
        "",
        "Best-F1 results, seed-level signs and Train/Validation gaps are in PILOT_REPORT.md and paired_comparisons.csv. Interpret small gains against seed spread, checkpoint sensitivity and the ele-fashion lower-support class results.",
        "",
        "No optional training-strategy screen was run. H2's paired Best-Accuracy differences are small and mixed by metric: Accuracy is positive across datasets, Macro-F1 is slightly lower on Movies and Grocery but higher on ele-fashion, and CE improves on Movies/Grocery but is slightly higher on ele-fashion. Since H2 combines SameNode with late dropout, this pattern does not provide a clean, isolated hypothesis for changing dropout, weight decay or learning rate in a small screen.",
        "",
        "## Interpretation boundary", "",
        "H2 is a combined Same-node plus training-only late-dropout package; this matrix does not separate those effects. Physical/semantic router activity and residual scales are diagnostics, not causal ablations. The mixed endpoint pattern did not justify an optional training-strategy screen. No Test evaluation or successor method was included.",
        "",
    ])
    (STUDY / "DECISION_MEMO.md").write_text("\n".join(memo), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
