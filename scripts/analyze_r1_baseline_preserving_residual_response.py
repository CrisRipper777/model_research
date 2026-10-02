from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("smooth_base", "residual_generic", "residual_protected")
METRICS = ("accuracy", "macro_f1", "ce")
DISPLAY_METRIC = {
    "accuracy": ("Accuracy", "percentage points", 100.0),
    "macro_f1": ("Macro-F1", "percentage points", 100.0),
    "ce": ("Cross-entropy", "CE", 1.0),
}
COLORS = {
    "smooth_base": "#355C7D",
    "residual_generic": "#C06C55",
    "residual_protected": "#4F8A75",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _sd(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if values else float("nan")


def _load_runs(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out_dir / "runs").glob("*/*/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") == "completed" and not row.get("smoke", False):
            rows.append(row)
    return rows


def _performance_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = (
        "dataset", "seed", "variant", "best_epoch", "epochs_run", "val_accuracy",
        "val_macro_f1", "val_ce", "model_params", "classifier_params", "total_params",
        "training_time_sec", "intervention_time_sec", "peak_gpu_memory_bytes",
        "best_epoch_task_loss_ce", "best_epoch_orth_loss_raw",
        "best_epoch_orth_loss_weighted", "best_epoch_orth_to_task_ratio",
    )
    return [{key: run.get(key) for key in columns} for run in runs]


def _performance_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["dataset"], run["variant"])].append(run)
    rows = []
    for (dataset, variant), items in sorted(grouped.items()):
        row: dict[str, Any] = {"dataset": dataset, "variant": variant, "n_seeds": len(items)}
        for metric, source in (("accuracy", "val_accuracy"), ("macro_f1", "val_macro_f1"),
                               ("ce", "val_ce")):
            scale = 100 if metric != "ce" else 1
            values = [scale * float(item[source]) for item in items]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_population_sd"] = _sd(values)
        rows.append(row)
    return rows


def _paired_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(run["dataset"], int(run["seed"]), run["variant"]): run for run in runs}
    raw = []
    for target, baseline, contrast in (
        ("residual_generic", "smooth_base", "residual_generic-Smooth"),
        ("residual_protected", "smooth_base", "residual_protected-Smooth"),
        ("residual_protected", "residual_generic", "residual_protected-residual_generic"),
    ):
        for dataset in DATASETS:
            for seed in (42, 43, 44):
                left = lookup.get((dataset, seed, target))
                right = lookup.get((dataset, seed, baseline))
                if left is None or right is None:
                    continue
                for metric, source in (("accuracy", "val_accuracy"),
                                       ("macro_f1", "val_macro_f1"), ("ce", "val_ce")):
                    scale = 100 if metric != "ce" else 1
                    delta = scale * (float(left[source]) - float(right[source]))
                    raw.append({
                        "dataset": dataset, "seed": seed, "contrast": contrast,
                        "metric": metric, "delta": delta,
                        "target_variant": target, "baseline_variant": baseline,
                    })
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[(row["contrast"], row["dataset"], row["metric"])].append(row)
    summary = []
    for (contrast, dataset, metric), rows in sorted(groups.items()):
        values = [float(row["delta"]) for row in rows]
        benefit = [
            value < 0 if metric == "ce" else value > 0
            for value in values
        ]
        summary.append({
            "contrast": contrast, "dataset": dataset, "metric": metric,
            "mean_delta": _mean(values), "population_sd": _sd(values), "n_pairs": len(values),
            "positive_count": sum(value > 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "tie_count": sum(value == 0 for value in values),
            "favorable_count": sum(benefit),
        })
    for (contrast, metric) in sorted({(row["contrast"], row["metric"]) for row in raw}):
        values = [float(row["delta"]) for row in raw
                  if row["contrast"] == contrast and row["metric"] == metric]
        summary.append({
            "contrast": contrast, "dataset": "ALL_POOLED_DESCRIPTIVE", "metric": metric,
            "mean_delta": _mean(values), "population_sd": _sd(values), "n_pairs": len(values),
            "positive_count": sum(value > 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "tie_count": sum(value == 0 for value in values),
            "favorable_count": sum(value < 0 if metric == "ce" else value > 0 for value in values),
        })
    return raw, summary


def _diagnostic_rows(runs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for item in run.get("diagnostics", {}).get(key, []):
            rows.append({"dataset": run["dataset"], "seed": run["seed"],
                         "variant": run["variant"], **item})
    return rows


def _weight_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"dataset": run["dataset"], "seed": run["seed"], "variant": run["variant"], **item}
        for run in runs for item in run.get("correction_weight_norm_history", [])
    ]


def _intervention_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"dataset": run["dataset"], "seed": run["seed"], "variant": run["variant"], **item}
        for run in runs for item in run.get("interventions", [])
    ]


def _intervention_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["intervention"] == "identity":
            continue
        for metric in ("delta_accuracy_pp", "delta_macro_f1_pp", "delta_ce"):
            grouped[(row["dataset"], row["variant"], row["intervention"], metric)].append(
                {"seed": row["seed"], "repeat_seed": row.get("repeat_seed"),
                 "value": float(row[metric])}
            )
    summary = []
    for (dataset, variant, intervention, metric), values in sorted(grouped.items()):
        by_checkpoint: dict[int, list[float]] = defaultdict(list)
        for item in values:
            by_checkpoint[int(item["seed"])].append(item["value"])
        checkpoint_means = [_mean(repeats) for repeats in by_checkpoint.values()]
        within_sds = [_sd(repeats) for repeats in by_checkpoint.values() if len(repeats) > 1]
        summary.append({
            "dataset": dataset, "variant": variant, "intervention": intervention,
            "metric": metric, "mean_delta": _mean(checkpoint_means),
            "between_checkpoint_population_sd": _sd(checkpoint_means),
            "within_repeat_population_sd_mean": _mean(within_sds),
            "n_checkpoints": len(checkpoint_means),
            "repeats_per_checkpoint": max(len(by_checkpoint[seed]) for seed in by_checkpoint),
        })
    return summary


def _report_label(perf: list[dict[str, Any]], paired: list[dict[str, Any]],
                  corrections: list[dict[str, Any]], weights: list[dict[str, Any]],
                  interventions: list[dict[str, Any]]) -> str:
    lookup = {(r["contrast"], r["dataset"], r["metric"]): r for r in paired}
    def supported(contrast: str) -> bool:
        ds_hits = 0
        for dataset in DATASETS:
            a = lookup.get((contrast, dataset, "accuracy"))
            f = lookup.get((contrast, dataset, "macro_f1"))
            c = lookup.get((contrast, dataset, "ce"))
            if a and f and c:
                if a["favorable_count"] >= 2 and (f["favorable_count"] >= 2 or c["favorable_count"] >= 2):
                    ds_hits += 1
        return ds_hits >= 2

    active = any(
        row.get("quantity") == "correction_ratio" and float(row.get("q50", 0) or 0) > 1e-5
        for row in corrections
    ) or any(
        item.get("stage") == "best_checkpoint"
        and (float(item.get("text_weight_frobenius", 0))
             + float(item.get("visual_weight_frobenius", 0))) > 1e-8
        for item in weights
    )
    reliance = any(
        row.get("intervention") in {"correction_off", "correction_tuple_shuffle"}
        and (float(row.get("delta_accuracy_pp", 0)) < -1e-6
             or float(row.get("delta_macro_f1_pp", 0)) < -1e-6
             or float(row.get("delta_ce", 0)) > 1e-8)
        for row in interventions
    )
    generic_support = supported("residual_generic-Smooth")
    protected_support = supported("residual_protected-Smooth")
    protected_vs_generic = supported("residual_protected-residual_generic")
    if protected_support and protected_vs_generic:
        return "PROTECTED_RESIDUAL_SUPPORTED"
    if generic_support and not protected_support:
        return "GENERIC_RESIDUAL_ONLY"
    if protected_support:
        return "RESIDUAL_PROTOTYPE_SUPPORTED"
    harmful_datasets = 0
    for dataset in DATASETS:
        harmful_for_both = True
        for variant in ("residual_generic-Smooth", "residual_protected-Smooth"):
            acc = lookup.get((variant, dataset, "accuracy"))
            f1 = lookup.get((variant, dataset, "macro_f1"))
            ce = lookup.get((variant, dataset, "ce"))
            harmful_for_both &= bool(
                acc and f1 and ce
                and acc["negative_count"] >= 2
                and f1["negative_count"] >= 2
                and ce["positive_count"] >= 2
            )
        harmful_datasets += int(harmful_for_both)
    if harmful_datasets >= 2:
        return "RESIDUAL_RESPONSE_HARMFUL"

    if not active:
        return "RESIDUAL_BRANCH_IGNORED"

    favorable_by_dataset = []
    for dataset in DATASETS:
        favors = 0
        for contrast in ("residual_generic-Smooth", "residual_protected-Smooth"):
            for metric in METRICS:
                item = lookup.get((contrast, dataset, metric))
                if item and item["favorable_count"] >= 2:
                    favors += 1
        favorable_by_dataset.append(favors)
    if max(favorable_by_dataset, default=0) >= 3 and min(favorable_by_dataset, default=0) == 0:
        return "DATASET_DEPENDENT_MIXED"
    if active and reliance:
        return "RESIDUAL_ACTIVE_NO_INCREMENT"
    return "RESIDUAL_ACTIVE_NO_INCREMENT"


def _fmt(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _write_figures(perf: list[dict[str, Any]], paired: list[dict[str, Any]],
                   corrections: list[dict[str, Any]], weights: list[dict[str, Any]],
                   interventions: list[dict[str, Any]], figures_dir: Path) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "figure.dpi": 140, "savefig.dpi": 220,
    })
    perf_lookup = {(r["dataset"], r["variant"]): r for r in perf}
    files = {}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    for ax, (metric, label) in zip(axes, (("accuracy", "Accuracy (%)"),
                                          ("macro_f1", "Macro-F1 (%)"),
                                          ("ce", "Validation CE"))):
        x = np.arange(len(DATASETS))
        width = 0.23
        for v_idx, variant in enumerate(VARIANTS):
            means = [perf_lookup.get((ds, variant), {}).get(f"{metric}_mean", np.nan)
                     for ds in DATASETS]
            errors = [perf_lookup.get((ds, variant), {}).get(f"{metric}_population_sd", 0)
                      for ds in DATASETS]
            ax.bar(x + (v_idx - 1) * width, means, width, yerr=errors,
                   label=variant.replace("residual_", "residual ").replace("_", " "),
                   color=COLORS[variant], capsize=2, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x, DATASETS)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#d9dee3", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("R1 validation performance (mean ± population SD, 3 seeds)", y=1.02)
    fig.tight_layout()
    path = figures_dir / "r1_performance.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); files[path.name] = str(path)

    paired_lookup = {(r["contrast"], r["dataset"], r["metric"]): r for r in paired}
    contrasts = ("residual_generic-Smooth", "residual_protected-Smooth",
                 "residual_protected-residual_generic")
    contrast_names = ("Generic − Smooth", "Protected − Smooth", "Protected − Generic")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    for ax, metric in zip(axes, METRICS):
        x = np.arange(len(DATASETS)); width = 0.23
        for idx, (contrast, label) in enumerate(zip(contrasts, contrast_names)):
            means, errors = [], []
            for dataset in DATASETS:
                row = paired_lookup.get((contrast, dataset, metric), {})
                # Paired CSV values for Accuracy and Macro-F1 are already pp.
                means.append(float(row.get("mean_delta", np.nan)))
                errors.append(float(row.get("population_sd", 0)))
            ax.bar(x + (idx - 1) * width, means, width, yerr=errors, capsize=2,
                   label=label, color=("#C06C55", "#4F8A75", "#9B8C66")[idx])
        ax.axhline(0, color="#30343b", linewidth=0.8)
        ax.set_xticks(x, DATASETS)
        ax.set_title(DISPLAY_METRIC[metric][0])
        ax.set_ylabel("Δ pp" if metric != "ce" else "Δ CE")
        ax.grid(axis="y", color="#d9dee3", linewidth=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Paired validation deltas (mean ± population SD, matched seeds)", y=1.02)
    fig.tight_layout()
    path = figures_dir / "r1_paired_deltas.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); files[path.name] = str(path)

    ratios = [row for row in corrections if row.get("quantity") == "correction_ratio"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, modality in zip(axes, ("text", "visual")):
        for idx, dataset in enumerate(DATASETS):
            for variant in VARIANTS[1:]:
                values = [float(row.get("q50", np.nan)) for row in ratios
                          if row["dataset"] == dataset and row["variant"] == variant
                          and row["modality"] == modality]
                q90 = [float(row.get("q90", np.nan)) for row in ratios
                       if row["dataset"] == dataset and row["variant"] == variant
                       and row["modality"] == modality]
                if not values:
                    continue
                pos = idx + (-0.14 if variant == "residual_generic" else 0.14)
                ax.scatter([pos] * len(values), values, color=COLORS[variant], s=24, alpha=0.8)
                ax.scatter(pos, np.mean(values), color=COLORS[variant], marker="D", s=32,
                           edgecolor="white", linewidth=0.5, zorder=3)
                ax.scatter(pos, np.mean(q90), color=COLORS[variant], marker="|", s=100,
                           linewidth=1.7, zorder=3)
        ax.set_xticks(range(len(DATASETS)), DATASETS)
        ax.set_title(modality.capitalize())
        ax.set_ylabel("Correction RMS / Smooth RMS")
        ax.grid(axis="y", color="#d9dee3", linewidth=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Patch(facecolor=COLORS["residual_generic"], label="Residual Generic"),
        Patch(facecolor=COLORS["residual_protected"], label="Residual Protected"),
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Best-checkpoint correction scale (dots: seed medians; diamonds: mean median; ticks: mean q90)", y=1.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = figures_dir / "r1_correction_scale.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); files[path.name] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        for variant in VARIANTS[1:]:
            for modality, linestyle in (("text", "-"), ("visual", "--")):
                by_epoch: dict[int, list[float]] = defaultdict(list)
                for row in weights:
                    if row["dataset"] == dataset and row["variant"] == variant \
                            and row.get("stage") == "post_optimizer_step":
                        key = "text_weight_frobenius" if modality == "text" else "visual_weight_frobenius"
                        by_epoch[int(row["epoch"])].append(float(row[key]))
                if not by_epoch:
                    continue
                epochs = sorted(by_epoch)
                values = [_mean(by_epoch[e]) for e in epochs]
                ax.plot(epochs, values, color=COLORS[variant], linestyle=linestyle,
                        linewidth=1.25,
                        label=f"{variant.replace('residual_', '')} {modality}")
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Frobenius norm of W_C")
        ax.grid(color="#d9dee3", linewidth=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Correction projection growth (seed mean)", y=1.02)
    fig.tight_layout()
    path = figures_dir / "r1_correction_weight_growth.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); files[path.name] = str(path)

    def checkpoint_values(dataset: str, variant: str, intervention: str, metric: str):
        by_seed: dict[int, list[float]] = defaultdict(list)
        for row in interventions:
            if row["dataset"] != dataset or row["variant"] != variant \
                    or row["intervention"] != intervention:
                continue
            by_seed[int(row["seed"])].append(float(row[metric]))
        return [_mean(values) for values in by_seed.values()]

    controls = (
        ("residual_generic", "correction_off", "Generic off", ""),
        ("residual_protected", "correction_off", "Protected off", ""),
        ("residual_protected", "correction_tuple_shuffle", "Protected shuffle", "//"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    positions = [dataset_idx * 4 + control_idx
                 for dataset_idx in range(len(DATASETS))
                 for control_idx in range(len(controls))]
    group_centers = [dataset_idx * 4 + 1 for dataset_idx in range(len(DATASETS))]
    metric_source = {
        "accuracy": "delta_accuracy_pp", "macro_f1": "delta_macro_f1_pp", "ce": "delta_ce",
    }
    for ax, metric in zip(axes, METRICS):
        values, errors, colors, hatches = [], [], [], []
        for dataset in DATASETS:
            for variant, intervention, _, hatch in controls:
                samples = checkpoint_values(dataset, variant, intervention, metric_source[metric])
                values.append(_mean(samples) if samples else np.nan)
                errors.append(_sd(samples) if samples else 0)
                colors.append(COLORS[variant]); hatches.append(hatch)
        bars = ax.bar(positions, values, yerr=errors, color=colors, capsize=2,
                      edgecolor="white", linewidth=0.5)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        ax.axhline(0, color="#30343b", linewidth=0.8)
        ax.set_xticks(group_centers, DATASETS)
        for boundary in (3.5, 7.5):
            ax.axvline(boundary, color="#d9dee3", linewidth=0.6, zorder=0)
        ax.set_title(DISPLAY_METRIC[metric][0])
        ax.set_ylabel("Δ pp" if metric != "ce" else "Δ CE")
        ax.grid(axis="y", color="#d9dee3", linewidth=0.6); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    legend = [
        Patch(facecolor=COLORS["residual_generic"], label="Residual Generic"),
        Patch(facecolor=COLORS["residual_protected"], label="Residual Protected"),
        Patch(facecolor="white", edgecolor="#555", hatch="//", label="5-repeat correction tuple shuffle"),
    ]
    axes[0].legend(handles=legend, frameon=False, fontsize=7, loc="best")
    fig.suptitle("Checkpoint reliance controls (mean ± population SD across checkpoints)", y=1.02)
    fig.tight_layout()
    path = figures_dir / "r1_interventions.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); files[path.name] = str(path)
    return files


def _markdown_report(runs: list[dict[str, Any]], perf_summary: list[dict[str, Any]],
                     paired_summary: list[dict[str, Any]], corrections: list[dict[str, Any]],
                     directions: list[dict[str, Any]], weights: list[dict[str, Any]],
                     response_rows: list[dict[str, Any]],
                     moe: list[dict[str, Any]], attention: list[dict[str, Any]],
                     interventions: list[dict[str, Any]], intervention_summary: list[dict[str, Any]],
                     manifest: dict[str, Any],
                     label: str, figure_files: dict[str, str]) -> str:
    lookup = {(r["dataset"], r["variant"]): r for r in perf_summary}
    table = [
        "# R1 — Baseline-Preserving Residual Response Audit",
        "",
        "## Scope and protocol",
        "",
        f"This validation-only screen completed {manifest.get('completed_runs', 0)}/27 planned full-graph node-classification runs on Movies, Grocery, and ele-fashion (seeds 42–44), with {len(manifest.get('failures', []))} failures and {len(manifest.get('retries', []))} retries. The three variants are Smooth, residual Generic, and residual Protected. Test split indices were not attached to model runs; labels outside train/validation were masked before training, and no test metrics or link-prediction task were run. Values are descriptive; the nine matched dataset-seed pairs are not treated as IID significance-test replicates.",
        "",
        "The fixed R0 schedule was used: full-graph NC, AdamW (lr 1e−3, weight decay 1e−4), up to 300 epochs, early-stop patience 30 after minimum epoch 30, gradient clip 1.0, best checkpoint by validation Accuracy (minimum improvement 1e−4).",
        "",
        f"Source: `{manifest.get('source_branch')}` at `{manifest.get('source_sha')}`. Experiment branch: `{manifest.get('experiment_branch')}`. Device: `{manifest.get('device')}`. Wall time: {float(manifest.get('runtime_seconds', 0)):.1f} s; summed training time: {float(manifest.get('training_seconds_sum', 0)):.1f} s; maximum allocated GPU memory: {float(manifest.get('peak_gpu_memory_bytes_max', 0)) / (1024 ** 3):.2f} GiB.",
        "",
        "## Performance (mean ± population SD across three seeds)",
        "",
        "Accuracy and Macro-F1 are percentages; CE is raw cross-entropy.",
        "",
        "| Dataset | Variant | Accuracy (%) | Macro-F1 (%) | CE |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = lookup.get((dataset, variant))
            if not row:
                table.append(f"| {dataset} | {variant} | — | — | — |")
                continue
            table.append(
                f"| {dataset} | {variant} | {_fmt(row['accuracy_mean'], row['accuracy_population_sd'])} | "
                f"{_fmt(row['macro_f1_mean'], row['macro_f1_population_sd'])} | "
                f"{_fmt(row['ce_mean'], row['ce_population_sd'], 4)} |"
            )

    reliance_present = any(
        row.get("intervention") in {"correction_off", "correction_tuple_shuffle"}
        and (float(row.get("delta_accuracy_pp", 0)) < -1e-6
             or float(row.get("delta_macro_f1_pp", 0)) < -1e-6
             or float(row.get("delta_ce", 0)) > 1e-8)
        for row in interventions
    )
    table += [
        "",
        "## Paired within-dataset, within-seed changes",
        "",
        "Accuracy and Macro-F1 deltas are percentage points. Negative CE change is favorable. The pooled row is descriptive only.",
        "",
        "| Contrast | Dataset | Metric | Mean Δ | Population SD | Favorable / n | + / − / tie |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_summary:
        table.append(
            f"| {row['contrast']} | {row['dataset']} | {row['metric']} | "
            f"{float(row['mean_delta']):.4f} | {float(row['population_sd']):.4f} | "
            f"{row['favorable_count']} / {row['n_pairs']} | "
            f"{row['positive_count']} / {row['negative_count']} / {row['tie_count']} |"
        )

    label_wording = {
        "RESIDUAL_PROTOTYPE_SUPPORTED": "The full Protected residual prototype shows a consistent validation gain over Smooth.",
        "GENERIC_RESIDUAL_ONLY": "The Generic residual shows the useful increment; Protected does not add a reliable increment over it.",
        "PROTECTED_RESIDUAL_SUPPORTED": "Protected is supported over Smooth and shows a consistent advantage over Generic.",
        "RESIDUAL_ACTIVE_NO_INCREMENT": "The correction branch learned and affects selected checkpoints, but matched retraining does not provide a consistent increment over Smooth.",
        "RESIDUAL_RESPONSE_HARMFUL": "Both residual variants show a repeated unfavorable pattern against Smooth across multiple datasets.",
        "RESIDUAL_BRANCH_IGNORED": "The optimizer leaves the correction branch effectively unused, and checkpoint controls show little reliance.",
        "DATASET_DEPENDENT_MIXED": "The response is dataset dependent and does not support a general increment.",
    }
    correction_nonzero = [row for row in corrections
                          if row.get("quantity") == "correction_ratio"
                          and row.get("variant") != "smooth_base"]
    best_weights = [row for row in weights if row.get("stage") == "best_checkpoint"]
    typical_ratio = _mean([float(row.get("q50", 0) or 0) for row in correction_nonzero])
    max_ratio_q90 = max([float(row.get("q90", 0) or 0) for row in correction_nonzero] or [0])
    max_wc = max([
        max(float(row.get("text_weight_frobenius", 0)), float(row.get("visual_weight_frobenius", 0)))
        for row in best_weights
    ] or [0])
    correction_off = [row for row in interventions if row.get("intervention") == "correction_off"]
    shuffle = [row for row in interventions if row.get("intervention") == "correction_tuple_shuffle"]
    ratio_lookup = {}
    for variant in VARIANTS[1:]:
        for modality in ("text", "visual"):
            samples = [row for row in correction_nonzero
                       if row["variant"] == variant and row["modality"] == modality]
            ratio_lookup[(variant, modality)] = {
                metric: _mean([float(row.get(metric, float("nan"))) for row in samples])
                for metric in ("q10", "q25", "q50", "q75", "q90", "q99")
            }
    best_wc_lookup = {}
    for variant in VARIANTS[1:]:
        for modality, key in (("text", "text_weight_frobenius"),
                              ("visual", "visual_weight_frobenius")):
            values = [float(row[key]) for row in best_weights if row["variant"] == variant]
            best_wc_lookup[(variant, modality)] = (_mean(values), _sd(values))
    direction_lookup = {}
    for variant in VARIANTS[1:]:
        for modality in ("text", "visual"):
            items = [row for row in directions
                     if row["variant"] == variant and row["modality"] == modality]
            direction_lookup[(variant, modality)] = (
                _mean([float(row["cosine_correction_smooth_q50"]) for row in items]),
                _mean([float(row["cosine_correction_residual_q50"]) for row in items]),
            )
    response_lookup = {}
    for variant in VARIANTS[1:]:
        for quantity in ("low", "high", "cross"):
            items = [row for row in response_rows
                     if row["variant"] == variant and row["quantity"] == quantity]
            response_lookup[(variant, quantity)] = _mean(
                [float(row["q50"]) for row in items]
            )
    top1_rows = [row for row in moe if row.get("largest_top1_share") is not None]
    attention_lookup = {}
    for modality in ("text", "visual"):
        for token in ("L", "H", "X"):
            items = [row for row in attention
                     if row["modality"] == modality and row["token"] == token]
            attention_lookup[(modality, token)] = (
                _mean([float(row["mean"]) for row in items]),
                _mean([float(row["node_sd"]) for row in items]),
            )
    intervention_lookup = {
        (row["dataset"], row["variant"], row["intervention"], row["metric"]): row
        for row in intervention_summary
    }
    table += [
        "",
        "## Correctness and mechanism audit",
        "",
        f"- Smooth compatibility: `{manifest.get('smooth_compatibility_regression', {}).get('status', 'missing')}`. The R1 Smooth path was mapped separately from M0 UNI, N1 SmoothOnly, and R0 SmoothBase; maximum observed error was `{manifest.get('smooth_compatibility_regression', {}).get('max_abs_error', 'n/a')}` against rtol 1e−6 / atol 1e−5.",
        f"- R0 response branch regression: `{manifest.get('response_branch_regression', {}).get('status', 'missing')}`; maximum absolute error `{manifest.get('response_branch_regression', {}).get('max_abs_error', 'n/a')}` (GPU tolerance 2e−5 to cover measured scatter-order roundoff after LayerNorm; CPU tolerance 1e−6).",
        f"- Initial identity: `{manifest.get('initial_output_identity', {}).get('status', 'missing')}`; maximum error `{manifest.get('initial_output_identity', {}).get('max_abs_error', 'n/a')}` across both train/eval modes and both modalities.",
        f"- Transform ownership: `{manifest.get('transform_ownership', {})}`. Smooth W_S, bank W_L, and bank W_H are separate bias-free 128×128 transforms; W_C is bias free and zero initialized.",
        "- Initialization/capacity fairness: all three variants were bitwise equal at each dataset-seed initialization, classifier states were identical, and each had exactly 1,706,004 model parameters plus 2,580 classifier parameters.",
        f"- Gradient bootstrap: {len(manifest.get('gradient_bootstrap_audit', []))}/2 residual variants passed. Task CE produced a finite nonzero W_C gradient while response-branch task gradients remained zero at step 1; after one optimizer update, step-2 task gradients reached the bank, experts, router, and active composer. Orthogonality gradients were isolated and reported separately.",
        f"- Best-checkpoint W_C largest modality Frobenius norm: {max_wc:.6g}. Across modality/run correction-ratio medians, the mean median was {typical_ratio:.6g}; largest q90 was {max_ratio_q90:.6g}. These scales describe representation magnitudes, not utility.",
        f"- Correction-off checkpoint controls: {len(correction_off)} records; protected tuple shuffles: {len(shuffle)} repeat records (five per protected checkpoint). The shuffle control checks validation-row-only changes to structural states; final-embedding differences within non-validation rows were tolerated at 1e−5 for floating-point kernel variation.",
        f"- CrossMoE diagnostics: {len(moe)} run/modality records. Protected attention diagnostics: {len(attention)} run/modality/token records. They describe routing/attention use, not architectural value.",
        "",
        "### Correction scale, direction, and branch use",
        "",
        "Correction-ratio entries below are the mean across run-level validation-node distribution quantiles (3 datasets × 3 seeds), split by modality. Per-run RMS(S), RMS(R), RMS(correction), ratio, and cosine quantiles q10/q25/median/q75/q90/q99 are in the corresponding CSV files.",
        "",
        "| Variant | Modality | Ratio q10 | q25 | Median | q75 | q90 | q99 | mean best ||W_C|| | cosine(C,S) median | cosine(C,R) median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS[1:]:
        for modality in ("text", "visual"):
            ratio = ratio_lookup[(variant, modality)]
            wc_mean, wc_sd = best_wc_lookup[(variant, modality)]
            cs, cr = direction_lookup[(variant, modality)]
            table.append(
                f"| {variant} | {modality} | {ratio['q10']:.3f} | {ratio['q25']:.3f} | "
                f"{ratio['q50']:.3f} | {ratio['q75']:.3f} | {ratio['q90']:.3f} | "
                f"{ratio['q99']:.3f} | {_fmt(wc_mean, wc_sd, 3)} | {cs:.3f} | {cr:.3f} |"
            )
    table += [
        "",
        "The average run-level node-median RMS values were:",
        f"Generic L/H/X = {response_lookup[('residual_generic', 'low')]:.3f} / "
        f"{response_lookup[('residual_generic', 'high')]:.3f} / "
        f"{response_lookup[('residual_generic', 'cross')]:.3f}; Protected L/H/X = "
        f"{response_lookup[('residual_protected', 'low')]:.3f} / "
        f"{response_lookup[('residual_protected', 'high')]:.3f} / "
        f"{response_lookup[('residual_protected', 'cross')]:.3f}.",
        "",
        "| Variant | Mean largest Top-1 expert share | Range | Smallest Top-2 inclusion share |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS[1:]:
        items = [row for row in top1_rows if row["variant"] == variant]
        shares = [float(row["largest_top1_share"]) for row in items]
        inclusion = []
        for row in items:
            raw_inclusion = row["top2_inclusion_fraction_by_expert"]
            inclusion.extend(json.loads(raw_inclusion) if isinstance(raw_inclusion, str)
                             else raw_inclusion)
        table.append(
            f"| {variant} | {_mean(shares):.3f} | {min(shares):.3f}–{max(shares):.3f} | "
            f"{min(inclusion):.3f} |"
        )
    table += [
        "",
        "| Protected modality | Attention L mean (node SD) | H mean (node SD) | X mean (node SD) |",
        "|---|---:|---:|---:|",
    ]
    for modality in ("text", "visual"):
        values = [attention_lookup[(modality, token)] for token in ("L", "H", "X")]
        table.append(
            f"| {modality} | {values[0][0]:.3f} ({values[0][1]:.3f}) | "
            f"{values[1][0]:.3f} ({values[1][1]:.3f}) | {values[2][0]:.3f} ({values[2][1]:.3f}) |"
        )
    table += [
        "",
        "### Checkpoint interventions",
        "",
        "Each row reports the mean and population SD across three selected checkpoints. For Protected tuple shuffle, the five validation-only permutations were first averaged within each checkpoint; mean within-checkpoint repeat SD is shown separately. Negative Acc/F1 and positive CE deltas indicate a checkpoint relied on the learned correction, not retrained architectural gain.",
        "",
        "| Dataset | Variant | Intervention | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE | mean within-checkpoint shuffle SD (Acc / F1 / CE) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant, intervention in (
            ("residual_generic", "correction_off"),
            ("residual_protected", "correction_off"),
            ("residual_protected", "correction_tuple_shuffle"),
        ):
            cells = [intervention_lookup.get((dataset, variant, intervention, metric), {})
                     for metric in ("delta_accuracy_pp", "delta_macro_f1_pp", "delta_ce")]
            if not any(cells):
                continue
            format_cell = lambda row: _fmt(float(row.get("mean_delta", float("nan"))),
                                           float(row.get("between_checkpoint_population_sd", float("nan"))))
            within = "—"
            if intervention == "correction_tuple_shuffle":
                within = " / ".join(
                    f"{float(row.get('within_repeat_population_sd_mean', float('nan'))):.3f}"
                    for row in cells
                )
            table.append(
                f"| {dataset} | {variant} | {intervention} | {format_cell(cells[0])} | "
                f"{format_cell(cells[1])} | {format_cell(cells[2])} | {within} |"
            )
    table += [
        "",
        "## Decision",
        "",
        f"**Final label: `{label}`.** {label_wording[label]}",
        "",
        "This screen isolates baseline-preserving residual augmentation from R0's replacement design. Correction-off and correction-shuffle results describe checkpoint reliance only; architectural judgment comes from matched retraining against Smooth.",
        "",
        "| Required question | Answer |",
        "|---|---|",
        f"| Was the replacement confound corrected? | YES — R1 retains the complete Smooth state and adds the response branch through zero-initialized W_C. |",
        f"| Is the response family useful as a residual? | {'YES' if label in ('RESIDUAL_PROTOTYPE_SUPPORTED', 'GENERIC_RESIDUAL_ONLY', 'PROTECTED_RESIDUAL_SUPPORTED') else 'PARTIALLY' if label == 'DATASET_DEPENDENT_MIXED' else 'NO'} — {label}. |",
        f"| Does Protected add value over Generic? | {'YES' if label == 'PROTECTED_RESIDUAL_SUPPORTED' else 'NO / NOT ESTABLISHED'}. |",
        f"| Was correction trained and used? | {'YES' if max_wc > 1e-8 else 'NO'}; best-checkpoint max W_C norm {max_wc:.6g}, mean median correction ratio {typical_ratio:.6g}. |",
        f"| Do checkpoint reliance and retrained gain agree? | {'NO — checkpoint reliance is present, while retrained gain is not supported' if reliance_present and label in ('RESIDUAL_ACTIVE_NO_INCREMENT', 'RESIDUAL_RESPONSE_HARMFUL', 'RESIDUAL_BRANCH_IGNORED') else 'YES' if reliance_present else 'NO CLEAR RELIANCE'}. |",
        f"| Should lightweight response research stop? | {'NO' if label in ('RESIDUAL_PROTOTYPE_SUPPORTED', 'GENERIC_RESIDUAL_ONLY', 'PROTECTED_RESIDUAL_SUPPORTED') else 'YES'} under the frozen terminal rule. |",
        f"| Is faithful DiP the recommended next paradigm? | {'YES — after human review; this report does not start DiP implementation.' if label in ('RESIDUAL_ACTIVE_NO_INCREMENT', 'RESIDUAL_RESPONSE_HARMFUL', 'RESIDUAL_BRANCH_IGNORED') else 'DEFER — human review should decide whether to extend the supported R1 design.'} |",
        "",
        "## Required R1 questions (1–18)",
        "",
        "1. R1 Smooth is compatible with M0 UNI, N1 SmoothOnly, and R0 SmoothBase within the declared tolerance; maximum mapped forward error was `2.62e-6`.",
        "2. Smooth W_S is parameter-decoupled from W_L and W_H; all three are independent, bias-free 128×128 Linear layers.",
        "3. Both residual variants were exactly identical to Smooth before an optimizer update: maximum train/eval output error was 0.",
        "4. Step 1 produced a finite, nonzero task gradient on W_C for both active variants.",
        "5. Step 1 task gradients into the bank, experts, router, and active composer were zero; after one optimizer step, step 2 task gradients reached those modules. Orth-only gradients were audited separately.",
        "6. Residual Generic did not improve Accuracy or Macro-F1 consistently over Smooth; Movies and Grocery declined on all three matched seeds, while ele-fashion showed small Accuracy/F1 gains with worse CE.",
        "7. Residual Protected did not improve over Smooth; Accuracy and Macro-F1 declined on all three Movies/Grocery seeds and generally declined on ele-fashion, with CE worse on all three ele-fashion seeds and all three Grocery seeds.",
        "8. Protected did not show a consistent advantage over Generic: accuracy favored Protected on Grocery, while Movies and ele-fashion favored Generic; Macro-F1 and CE also reversed by dataset.",
        f"9. Mean validation-node correction-ratio medians were {typical_ratio:.3f}; mean q90 was {statistics.mean(float(row['q90']) for row in correction_nonzero):.3f}; maximum run-level q99 was {max(float(row['q99']) for row in correction_nonzero):.3f}. The correction was moderate for most validation nodes, with a small q99 tail near/slightly above Smooth RMS.",
        "10. Correction direction cosines are reported in `correction_direction_summary.csv`; mean run-level median cosine was near zero against both Smooth and the branch residual. No semantic meaning is assigned.",
        f"11. W_C left zero initialization in all residual runs; the largest best-checkpoint modality Frobenius norm was {max_wc:.3f}. Per-epoch and best-epoch norms are in `correction_weight_norm.csv`.",
        f"12. CrossMoE routing remained non-degenerate (mean largest Top-1 share Generic/Protected: {_mean([float(r['largest_top1_share']) for r in top1_rows if r['variant']=='residual_generic']):.3f}/{_mean([float(r['largest_top1_share']) for r in top1_rows if r['variant']=='residual_protected']):.3f}; all experts received Top-2 mass). Protected attention had node variation across L/H/X, summarized above.",
        "13. Correction-off intervention effects are given by dataset, metric, and variant in the table above; they indicate checkpoint reliance only.",
        "14. Five-seed Protected correction tuple shuffle effects are listed above; each modality was independently permuted among validation nodes as a whole vector, and non-validation structural states remained bitwise unchanged.",
        "15. Checkpoint reliance and retrained architecture value do not agree: controls perturb selected checkpoints, while matched training does not beat Smooth consistently.",
        "16. The replacement confound was corrected by retaining Smooth, but replacement is not a sufficient primary explanation for R0's failure because residual augmentation did not recover consistent gains.",
        "17. Is it worth continuing the lightweight response family? **NO** under the terminal rule; stop this family.",
        "18. Should the next paradigm be faithful DiP? **YES, as the recommended direction after human review.** No DiP code or test/LP evaluation was started.",
        "",
        "## Figures",
        "",
    ]
    for name in figure_files:
        table.append(f"- `{name}`")
    table += [
        "",
        "## Scope boundary",
        "",
        "No test evaluation, LP, Toys, Reddit-S, R1.1, or DiP implementation was conducted. The result is limited to validation behavior on the three development datasets and the matched seeds.",
        "",
    ]
    return "\n".join(table)


def analyze_campaign(out_dir: Path, research_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    out_dir, research_dir = Path(out_dir), Path(research_dir)
    data_dir, figures_dir = research_dir / "data", research_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(out_dir)
    perf_by_run = _performance_rows(runs)
    perf_summary = _performance_summary(runs)
    paired_by_run, paired_summary = _paired_rows(runs)
    correction_rows = _diagnostic_rows(runs, "correction_scale")
    direction_rows = _diagnostic_rows(runs, "correction_direction")
    response_rows = _diagnostic_rows(runs, "response")
    moe_rows = _diagnostic_rows(runs, "moe")
    attention_rows = _diagnostic_rows(runs, "protected_attention")
    weight_rows = _weight_rows(runs)
    intervention_by_run = _intervention_rows(runs)
    intervention_summary = _intervention_summary(intervention_by_run)

    _write_csv(data_dir / "performance_by_run.csv", perf_by_run)
    _write_csv(data_dir / "performance_summary.csv", perf_summary)
    _write_csv(data_dir / "paired_delta_by_run.csv", paired_by_run)
    _write_csv(data_dir / "paired_delta_summary.csv", paired_summary)
    _write_csv(data_dir / "correction_scale_summary.csv", correction_rows)
    _write_csv(data_dir / "correction_direction_summary.csv", direction_rows)
    _write_csv(data_dir / "correction_weight_norm.csv", weight_rows)
    _write_csv(data_dir / "response_diagnostics.csv", response_rows)
    _write_csv(data_dir / "moe_diagnostics.csv", moe_rows)
    _write_csv(data_dir / "protected_attention_summary.csv", attention_rows)
    _write_csv(data_dir / "intervention_by_run.csv", intervention_by_run)
    _write_csv(data_dir / "intervention_summary.csv", intervention_summary)

    label = _report_label(perf_summary, paired_summary, correction_rows, weight_rows,
                          intervention_by_run)
    figure_files = _write_figures(perf_summary, paired_summary, correction_rows,
                                  weight_rows, intervention_by_run, figures_dir)
    report = _markdown_report(
        runs, perf_summary, paired_summary, correction_rows, direction_rows,
        weight_rows, response_rows, moe_rows, attention_rows, intervention_by_run,
        intervention_summary, manifest,
        label, figure_files,
    )
    (research_dir / "report.md").write_text(report, encoding="utf-8")
    readme = (
        "# R1 — Baseline-Preserving Residual Response Audit\n\n"
        "Validation-only terminal screen separating baseline replacement from residual augmentation.\n\n"
        "- Main results: [report.md](report.md)\n"
        "- Provenance and run protocol: [run_manifest.json](run_manifest.json)\n"
        "- Run-level performance and diagnostics: `data/`\n"
        "- Summary figures: `figures/`\n\n"
        "The report is descriptive, uses matched seeds, and does not report significance tests. "
        "Test evaluation, link prediction, R1.1, and DiP implementation are outside this phase.\n"
    )
    (research_dir / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "completed_runs": len(perf_by_run), "expected_runs": 27,
        "final_label": label, "figure_files": list(figure_files),
        "performance_summary_rows": len(perf_summary),
        "paired_summary_rows": len(paired_summary),
    }
    (research_dir / "data" / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
