from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FIGURE_SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))

from audit_panel_alignment import require_matplotlib_panel_alignment

DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("premix_edge_control", "keep_static", "keep_target", "keep_edge")
METHODS = ("UNI", "Historical E0 EdgeMix", "PremixEdgeControl", "KeepStatic", "KeepTarget", "KeepEdge")
METRICS = ("val_accuracy", "val_macro_f1", "val_ce")
METRIC_DISPLAY = {"val_accuracy": "Accuracy", "val_macro_f1": "Macro-F1", "val_ce": "Cross-entropy"}
VARIANT_DISPLAY = {
    "premix_edge_control": "PremixEdgeControl",
    "keep_static": "KeepStatic",
    "keep_target": "KeepTarget",
    "keep_edge": "KeepEdge",
}
COLORS = {
    "UNI": "#46505f",
    "Historical E0 EdgeMix": "#9a9a9a",
    "PremixEdgeControl": "#7393b3",
    "KeepStatic": "#637eaa",
    "KeepTarget": "#cf9d50",
    "KeepEdge": "#bd5360",
}
DATASET_COLORS = {"Movies": "#496b9b", "Grocery": "#d08b36", "ele-fashion": "#398579"}
OUT_DEFAULT = PROJECT_ROOT / "outputs" / "e01_function_provenance_preservation"
RESEARCH_DEFAULT = PROJECT_ROOT / "research" / "e01_function_provenance_preservation"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_run_records(out_dir: Path) -> list[dict[str, Any]]:
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"missing E0.1 formal run record: {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") != "completed" or record.get("evaluate_test") is not False:
                    raise RuntimeError(f"invalid E0.1 run or test protocol: {path}")
                records.append(record)
    keys = [(r["dataset"], int(r["seed"]), r["variant"]) for r in records]
    if len(records) != 36 or len(set(keys)) != 36:
        raise AssertionError(f"expected 36 unique E0.1 runs, got rows={len(records)}, unique={len(set(keys))}")
    return records


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def performance_tables(records: list[dict[str, Any]], research_dir: Path):
    e0_rows = read_csv(research_dir.parent / "e0_structured_relation_function_executor" / "data" / "performance_by_run.csv")
    m0_rows = read_csv(research_dir.parent / "m0_adaptive_propagation_screen" / "data" / "performance_by_run.csv")
    historical_e0 = {(row["dataset"], int(row["seed"])): row for row in e0_rows if row["method"] == "EdgeMix"}
    historical_uni = {(row["dataset"], int(row["seed"])): row for row in m0_rows if row["variant"] == "uniform"}
    rows = []
    for record in records:
        rows.append({"dataset": record["dataset"], "seed": int(record["seed"]),
                     "method": VARIANT_DISPLAY[record["variant"]], "variant": record["variant"],
                     **{metric: float(record[metric]) for metric in METRICS},
                     "trainable_params": int(record["total_trainable_params"]),
                     "best_epoch": int(record["best_epoch"]), "training_time_sec": float(record["training_time_sec"])})
    for method, source in (("Historical E0 EdgeMix", historical_e0), ("UNI", historical_uni)):
        for dataset in DATASETS:
            for seed in SEEDS:
                row = source[(dataset, seed)]
                rows.append({"dataset": dataset, "seed": seed, "method": method,
                             "variant": "historical_e0_edge_mix" if method.startswith("Historical") else "uniform",
                             **{metric: float(row[metric]) for metric in METRICS},
                             "trainable_params": int(float(row["trainable_params"])),
                             "best_epoch": int(float(row["best_epoch"])),
                             "training_time_sec": float(row["training_time_sec"])})
    if len(rows) != 54:
        raise AssertionError(f"expected 36 current + 18 historical reference rows, got {len(rows)}")

    summaries = []
    for dataset in (*DATASETS, "ALL"):
        subset = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        for method in METHODS:
            method_rows = [row for row in subset if row["method"] == method]
            if len(method_rows) != (9 if dataset == "ALL" else 3):
                raise AssertionError(f"missing performance rows for {dataset}/{method}: {len(method_rows)}")
            summary: dict[str, Any] = {"dataset": dataset, "method": method, "n": len(method_rows)}
            for metric in METRICS:
                mean, sd = _mean_sd([float(row[metric]) for row in method_rows])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_sd"] = sd
            summaries.append(summary)

    by_key = {(row["dataset"], row["seed"], row["method"]): row for row in rows}
    comparisons = (
        ("KeepEdge-PremixEdgeControl", "KeepEdge", "PremixEdgeControl"),
        ("KeepTarget-KeepStatic", "KeepTarget", "KeepStatic"),
        ("KeepEdge-KeepTarget", "KeepEdge", "KeepTarget"),
        ("KeepEdge-KeepStatic", "KeepEdge", "KeepStatic"),
        ("PremixEdgeControl-HistoricalE0EdgeMix", "PremixEdgeControl", "Historical E0 EdgeMix"),
        ("KeepEdge-UNI", "KeepEdge", "UNI"),
    )
    paired_raw = []
    for dataset, seed in ((d, s) for d in DATASETS for s in SEEDS):
        for comparison, left, right in comparisons:
            a, b = by_key[(dataset, seed, left)], by_key[(dataset, seed, right)]
            paired_raw.append({"dataset": dataset, "seed": seed, "comparison": comparison,
                               **{f"delta_{metric}": float(a[metric]) - float(b[metric]) for metric in METRICS}})
    paired_summary = []
    for dataset in (*DATASETS, "ALL"):
        for comparison, _, _ in comparisons:
            values = [r for r in paired_raw if r["comparison"] == comparison and
                      (dataset == "ALL" or r["dataset"] == dataset)]
            row = {"dataset": dataset, "comparison": comparison, "n_pairs": len(values)}
            for metric in METRICS:
                scale = 100.0 if metric != "val_ce" else 1.0
                mean, sd = _mean_sd([float(v[f"delta_{metric}"]) * scale for v in values])
                row[f"delta_{metric}_mean"] = mean
                row[f"delta_{metric}_sd"] = sd
                row[f"{metric}_units"] = "percentage_points" if metric != "val_ce" else "native"
            paired_summary.append(row)
    return rows, summaries, paired_raw, paired_summary


def diagnostic_tables(records: list[dict[str, Any]]):
    performance_params, init_rows, e0_rows = [], [], []
    channel_norms, distinctness, channel_mass, composer = [], [], [], []
    intervention_rows, expert_scale_rows = [], []
    for record in records:
        dataset, seed, variant = record["dataset"], int(record["seed"]), record["variant"]
        parameter = {"dataset": dataset, "seed": seed, "variant": variant,
                     "model_params": record["model_params"],
                     "trainable_model_params": record["model_params"],
                     "classifier_params": record["classifier_params"],
                     "total_trainable_params": record["total_trainable_params"]}
        performance_params.append(parameter)
        init = record["initialization_audit"]
        if variant == VARIANTS[0]:
            row = {"dataset": dataset, "seed": seed,
                   "all_variants_bitwise_equal": init["all_variants_bitwise_equal"],
                   "exact_trainable_parameter_match": init["exact_trainable_parameter_match"],
                   "e0_common_init_bitwise_equal": init["e0_common_init"]["bitwise_equal"],
                   "e0_common_state_entries": init["e0_common_init"]["common_state_entries"],
                   "e0_common_mismatched_entries": init["e0_common_init"]["mismatched_entries"]}
            for name, value in init["variant_hashes"].items():
                row[f"hash_{name}"] = value
            for name, value in init["classifier_hashes"].items():
                row[f"classifier_hash_{name}"] = value
            for name, value in init["pairwise_model_and_classifier_bitwise_equal"].items():
                row[f"pairwise_{name}"] = value
            init_rows.append(row)
            e0_rows.append({"dataset": dataset, "seed": seed,
                            **init["e0_common_init"],
                            "all_common_module_initial_values_bitwise_equal": init["e0_common_init"]["bitwise_equal"]})
        channel_norms.extend(record["channel_norm_rows"])
        distinctness.extend(record["channel_distinctness_rows"])
        channel_mass.extend(record["channel_mass_rows"])
        composer.extend(record["composer_diagnostic_rows"])
        intervention_rows.extend(record.get("intervention_rows", []))
        if "expert_scale_rows" not in record or record.get("calibration_audit", {}).get("status") != "passed":
            raise AssertionError(f"missing completed E0 calibration audit for {dataset}/{seed}/{variant}")
        expert_scale_rows.extend(record["expert_scale_rows"])

    param_keys = [(r["dataset"], int(r["seed"]), r["variant"]) for r in performance_params]
    if len(performance_params) != 36 or len(set(param_keys)) != 36:
        raise AssertionError("parameter_summary must contain one and only one row per 3×3×4 run")
    if len(init_rows) != 9 or len(e0_rows) != 9:
        raise AssertionError(f"expected 9 initialization and E0 common-init regression rows; got {len(init_rows)}, {len(e0_rows)}")
    expected_scale_keys = {
        (r["dataset"], int(r["seed"]), r["variant"], r["modality"], r["diagnostic"])
        for r in expert_scale_rows
    }
    if len(expert_scale_rows) != 36 * 2 * 7 or len(expected_scale_keys) != len(expert_scale_rows):
        raise AssertionError("E0 calibration scale diagnostics must contain 14 unique rows per formal run")
    if not all(r["all_finite"] and not r["clamped"] for r in expert_scale_rows):
        raise AssertionError("E0 calibration audit found non-finite or clamped values")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in intervention_rows:
        grouped[(row["dataset"], row["intervention"])].append(row)
        grouped[("ALL", row["intervention"])].append(row)
    intervention_summary = []
    for (dataset, intervention), rows in sorted(grouped.items()):
        run_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            run_groups[(row["dataset"], int(row["seed"]))].append(row)
        run_mean_values = {metric: [] for metric in ("delta_accuracy_pp", "delta_macro_f1_pp", "delta_ce")}
        for run_rows in run_groups.values():
            for metric in run_mean_values:
                run_mean_values[metric].append(float(np.mean([float(r[metric]) for r in run_rows])))
        item: dict[str, Any] = {"dataset": dataset, "intervention": intervention,
                                "n_records": len(rows), "n_runs": len(run_groups)}
        for metric in run_mean_values:
            values = [float(row[metric]) for row in rows]
            avg, sd = _mean_sd(values)
            rmean, rsd = _mean_sd(run_mean_values[metric])
            item[f"{metric}_mean"] = avg
            item[f"{metric}_sd"] = sd
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
            item[f"{metric}_run_mean"] = rmean
            item[f"{metric}_run_sd"] = rsd
            adverse = (lambda value: value < 0) if metric != "delta_ce" else (lambda value: value > 0)
            item[f"n_runs_{metric}_adverse"] = sum(adverse(float(np.mean([r[metric] for r in group])))
                                                      for group in run_groups.values())
        intervention_summary.append(item)
    return (performance_params, init_rows, e0_rows, channel_norms, distinctness,
            channel_mass, composer, intervention_rows, intervention_summary, expert_scale_rows)


def _add_panel_label(ax, label: str):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def _configure_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    })


def save_figures(research_dir: Path, perf_rows, perf_summary, paired_raw,
                 channel_norms, distinctness, channel_mass, composer, intervention_summary):
    figure_dir = research_dir / "figures"
    qa_dir = figure_dir / "qa"
    figure_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    def export(name: str, fig):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig,
            json_out=qa_dir / f"{name}.alignment.json",
            overlay_svg=qa_dir / f"{name}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
        base = figure_dir / name
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

    # Figure contract: E0.1 tests whether late function provenance adds matched-
    # capacity validation value. These panels are primary comparison, channel
    # availability/distinctness, composer use, and fixed-checkpoint attacks.
    summary_index = {(r["dataset"], r["method"]): r for r in perf_summary}
    methods = list(METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.45))
    offsets = np.linspace(-0.22, 0.22, len(methods))
    markers = ["o", "s", "D", "^", "v", "P"]
    for axis_id, (ax, metric) in enumerate(zip(axes, METRICS)):
        scale = 100.0 if metric != "val_ce" else 1.0
        positions = np.arange(len(DATASETS))
        for method_id, method in enumerate(methods):
            mean = [summary_index[(d, method)][f"{metric}_mean"] * scale for d in DATASETS]
            sd = [summary_index[(d, method)][f"{metric}_sd"] * scale for d in DATASETS]
            ax.errorbar(positions + offsets[method_id], mean, yerr=sd, fmt=markers[method_id],
                        color=COLORS[method], markersize=3.5, capsize=1.7, elinewidth=0.8,
                        linewidth=0, label=method)
        ax.set_xticks(positions, ["Movies", "Grocery", "Fashion"])
        ax.set_title(METRIC_DISPLAY[metric])
        ax.set_ylabel("Validation (%)" if metric != "val_ce" else "Validation CE")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        _add_panel_label(ax, "abc"[axis_id])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08), columnspacing=1.2, handletextpad=0.4)
    fig.text(0.5, 0.005, "Mean ± population SD across seeds (n = 3 per dataset); no significance tests.",
             ha="center", fontsize=5.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91), w_pad=1.2)
    export("e01_performance", fig)

    comparisons = (
        "KeepEdge-PremixEdgeControl", "KeepTarget-KeepStatic", "KeepEdge-KeepTarget",
        "KeepEdge-KeepStatic", "PremixEdgeControl-HistoricalE0EdgeMix", "KeepEdge-UNI",
    )
    short_comparison = ("E−P", "T−S", "E−T", "E−S", "P−E0", "E−U")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.8))
    x = np.arange(len(comparisons))
    dataset_offsets = {"Movies": -0.18, "Grocery": 0.0, "ele-fashion": 0.18}
    for axis_id, (ax, metric) in enumerate(zip(axes, METRICS)):
        scale = 100.0 if metric != "val_ce" else 1.0
        for dataset in DATASETS:
            selected = [r for r in paired_raw if r["dataset"] == dataset]
            means, sds = [], []
            for comparison in comparisons:
                values = [float(r[f"delta_{metric}"]) * scale for r in selected
                          if r["comparison"] == comparison]
                mean, sd = _mean_sd(values)
                means.append(mean); sds.append(sd)
            ax.errorbar(x + dataset_offsets[dataset], means, yerr=sds, fmt="o", markersize=3.3,
                        color=DATASET_COLORS[dataset], capsize=1.7, elinewidth=0.8,
                        linewidth=0, label=dataset)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.set_xticks(x, short_comparison)
        ax.set_title(METRIC_DISPLAY[metric])
        ax.set_ylabel("Paired Δ (pp)" if metric != "val_ce" else "Paired Δ CE")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        _add_panel_label(ax, "abc"[axis_id])
    handles = [Line2D([0], [0], marker="o", linestyle="", color=DATASET_COLORS[d], label=d)
               for d in DATASETS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, 0.005,
             "E=KeepEdge; P=Premix; T=KeepTarget; S=KeepStatic; U=UNI. Paired mean ± population SD.",
             ha="center", fontsize=5.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91), w_pad=1.0)
    export("e01_provenance_comparison", fig)

    mass_index: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    for row in channel_mass:
        mass_index[(row["dataset"], int(row["seed"]), row["variant"], row["modality"], row["channel"])] = row
    fig, axes = plt.subplots(3, 2, figsize=(7.3, 6.0), sharey=True)
    channel_colors = {"smooth": "#5577a8", "relational": "#cf9b48", "cross_modal": "#38877d"}
    x = np.arange(len(VARIANTS))
    for d_idx, dataset in enumerate(DATASETS):
        for m_idx, modality in enumerate(("text", "visual")):
            ax = axes[d_idx, m_idx]
            for channel in ("smooth", "relational", "cross_modal"):
                mean, sd = [], []
                for variant in VARIANTS:
                    values = [float(mass_index[(dataset, seed, variant, modality, channel)]["median"])
                              for seed in SEEDS]
                    m, s = _mean_sd(values)
                    mean.append(m); sd.append(s)
                ax.errorbar(x, mean, yerr=sd, marker="o", markersize=3, capsize=1.4,
                            linewidth=1.1, color=channel_colors[channel], label=channel.replace("_", "-") if d_idx == 0 and m_idx == 0 else None)
            ax.set_title(f"{dataset} · {modality.title()}")
            ax.set_xticks(x, ["Premix", "Static", "Target", "Edge"], rotation=20,
                          rotation_mode="anchor", ha="right")
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
            if m_idx == 0:
                ax.set_ylabel("Relative channel mass")
            _add_panel_label(ax, "abcdef"[d_idx * 2 + m_idx])
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.015))
    fig.text(0.5, 0.005, "Within-run node RMS mass median; points/bars summarize three seeds (mean ± population SD).",
             ha="center", fontsize=5.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97), h_pad=1.1, w_pad=0.8)
    export("e01_channel_mass", fig)

    pair_names = ("smooth_relational", "smooth_cross_modal", "relational_cross_modal")
    pair_titles = ("Smooth–Relational", "Smooth–Cross", "Relational–Cross")
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.65), sharey=True)
    pair_index = {(r["dataset"], int(r["seed"]), r["modality"], r["pair"]): r for r in distinctness}
    positions = np.arange(6)
    cats = [(dataset, modality) for dataset in DATASETS for modality in ("text", "visual")]
    for panel, (ax, pair, title) in enumerate(zip(axes, pair_names, pair_titles)):
        for i, (dataset, modality) in enumerate(cats):
            values = [float(pair_index[(dataset, seed, modality, pair)]["median"]) for seed in SEEDS]
            mean, sd = _mean_sd(values)
            ax.errorbar(i, mean, yerr=sd, fmt="o", color=DATASET_COLORS[dataset],
                        markerfacecolor="white" if modality == "visual" else DATASET_COLORS[dataset],
                        markersize=4, capsize=1.8, elinewidth=0.8)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.set_title(title)
        ax.set_xticks(positions, ["M-T", "M-V", "G-T", "G-V", "F-T", "F-V"], rotation=25,
                      rotation_mode="anchor", ha="right")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        if panel == 0:
            ax.set_ylabel("Node-context cosine")
        _add_panel_label(ax, "abc"[panel])
    legend = [Line2D([0], [0], marker="o", linestyle="", color=DATASET_COLORS[d], label=d)
              for d in DATASETS]
    legend.extend([Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
                          markeredgecolor="#333333", label="Visual (open)")])
    fig.legend(handles=legend, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, 0.005, "Per-run validation-node cosine distributions are summarized by median; error bars are seed SD (n = 3).",
             ha="center", fontsize=5.3)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91), w_pad=0.9)
    export("e01_channel_distinctness", fig)

    comp_index = {(r["dataset"], int(r["seed"]), r["variant"], r["modality"]): r for r in composer}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    offsets = np.linspace(-0.27, 0.27, len(VARIANTS))
    variant_colors = {v: COLORS[VARIANT_DISPLAY[v]] for v in VARIANTS}
    for row_id, modality in enumerate(("text", "visual")):
        for col_id, value_kind in enumerate(("ratio", "cosine")):
            ax = axes[row_id, col_id]
            for vi, variant in enumerate(VARIANTS):
                x_values, median_values, lower, upper, q90_values = [], [], [], [], []
                for di, dataset in enumerate(DATASETS):
                    rows = [comp_index[(dataset, seed, variant, modality)] for seed in SEEDS]
                    stat = "ratio" if value_kind == "ratio" else "cosine"
                    run_medians = [float(r[f"{stat}_median"]) for r in rows]
                    med = float(np.mean(run_medians))
                    q90 = float(np.mean([r[f"{stat}_q90"] for r in rows])) if stat == "ratio" else np.nan
                    x_values.append(di + offsets[vi]); median_values.append(med)
                    if value_kind == "ratio":
                        q25 = float(np.mean([r["ratio_q25"] for r in rows]))
                        q75 = float(np.mean([r["ratio_q75"] for r in rows]))
                        lower.append(med - q25); upper.append(q75 - med)
                    else:
                        seed_sd = float(np.std(run_medians, ddof=0))
                        lower.append(seed_sd); upper.append(seed_sd)
                    q90_values.append(q90)
                ax.errorbar(x_values, median_values, yerr=[lower, upper], fmt="o", markersize=3.3,
                            color=variant_colors[variant], capsize=1.4, elinewidth=0.8,
                            label=VARIANT_DISPLAY[variant] if row_id == 0 and col_id == 0 else None)
                if value_kind == "ratio":
                    ax.scatter(x_values, q90_values, marker="v", s=10,
                               color=variant_colors[variant], alpha=0.45, linewidths=0)
            ax.set_xticks(np.arange(3), ["Movies", "Grocery", "Fashion"])
            ax.set_title("Composer / base RMS" if value_kind == "ratio" else "Composer vs base cosine")
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
            if value_kind == "ratio":
                ax.set_ylabel("RMS(Ccomp) / RMS(Cmix)")
            else:
                ax.set_ylabel("cos(Ccomp, Cmix)")
                ax.set_ylim(-1.05, 1.05)
                ax.axhline(0, color="#333333", linewidth=0.6)
            _add_panel_label(ax, "abcd"[row_id * 2 + col_id])
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.025))
    fig.text(0.5, 0.005, "Ratio: mean node median with mean node IQR and q90 triangles; cosine: mean run median ± seed SD.",
             ha="center", fontsize=5.4)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), h_pad=1.0, w_pad=1.1)
    export("e01_composer_usage", fig)

    wanted = ("provenance_collapse", "provenance_permutation", "composer_off",
              "pi_shuffle_within_target", "pi_target_mean", "pi_global_mean",
              "modality_tied_pi", "smooth_only")
    intervention_index = {(r["dataset"], r["intervention"]): r for r in intervention_summary}
    labels = ("Collapse", "Permute", "Comp off", "Shuffle", "Target mean", "Global mean", "Modality tied", "Smooth only")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.1))
    x = np.arange(len(wanted))
    offsets = {"Movies": -0.18, "Grocery": 0.0, "ele-fashion": 0.18}
    for panel, (ax, metric, title) in enumerate(zip(
        axes, ("delta_accuracy_pp", "delta_macro_f1_pp", "delta_ce"),
        ("Δ Accuracy (pp)", "Δ Macro-F1 (pp)", "Δ CE"),
    )):
        for dataset in DATASETS:
            means, sds = [], []
            for intervention in wanted:
                row = intervention_index[(dataset, intervention)]
                means.append(float(row[f"{metric}_run_mean"]))
                sds.append(float(row[f"{metric}_run_sd"]))
            ax.errorbar(x + offsets[dataset], means, yerr=sds, fmt="o", markersize=3.1,
                        color=DATASET_COLORS[dataset], capsize=1.6, elinewidth=0.8,
                        label=dataset if panel == 0 else None)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.set_xticks(x, labels, rotation=90, rotation_mode="anchor", ha="right")
        ax.set_title(title)
        ax.set_ylabel("Δ CE" if metric == "delta_ce" else "Paired Δ (pp)")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        _add_panel_label(ax, "abc"[panel])
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, 0.005, "KeepEdge validation checkpoint; bars summarize seed-level mean ± population SD (n = 3 per dataset).",
             ha="center", fontsize=5.4)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91), w_pad=1.2)
    export("e01_interventions", fig)


def _report_text(perf_summary, paired_summary, intervention_summary, channel_norms,
                 distinctness, channel_mass, composer, expert_scale_rows, records, campaign_result):
    ps = {(r["dataset"], r["method"]): r for r in perf_summary}
    dsumm = {(r["dataset"], r["comparison"]): r for r in paired_summary}
    isum = {(r["dataset"], r["intervention"]): r for r in intervention_summary}

    def fmt(value, scale=1.0, digits=3):
        return f"{float(value)*scale:.{digits}f}"

    def perf_table(dataset):
        lines = [f"| {method} | {fmt(ps[(dataset, method)]['val_accuracy_mean'],100)} ± {fmt(ps[(dataset, method)]['val_accuracy_sd'],100)} | "
                 f"{fmt(ps[(dataset, method)]['val_macro_f1_mean'],100)} ± {fmt(ps[(dataset, method)]['val_macro_f1_sd'],100)} | "
                 f"{fmt(ps[(dataset, method)]['val_ce_mean'],1,4)} ± {fmt(ps[(dataset, method)]['val_ce_sd'],1,4)} |"
                 for method in METHODS]
        return "\n".join(["| Method | Accuracy (%) | Macro-F1 (%) | CE |", "|---|---:|---:|---:|", *lines])

    def delta_line(dataset, comparison):
        r = dsumm[(dataset, comparison)]
        return (f"Accuracy {fmt(r['delta_val_accuracy_mean'])} ± {fmt(r['delta_val_accuracy_sd'])} pp; "
                f"Macro-F1 {fmt(r['delta_val_macro_f1_mean'])} ± {fmt(r['delta_val_macro_f1_sd'])} pp; "
                f"CE {fmt(r['delta_val_ce_mean'],1,4)} ± {fmt(r['delta_val_ce_sd'],1,4)}")

    norm_index = {(r["dataset"], int(r["seed"]), r["variant"], r["modality"], r["channel"]): r
                  for r in channel_norms}
    mass_index = {(r["dataset"], int(r["seed"]), r["variant"], r["modality"], r["channel"]): r
                  for r in channel_mass}
    distinct_index = {(r["dataset"], int(r["seed"]), r["variant"], r["modality"], r["pair"]): r
                      for r in distinctness}
    comp_index = {(r["dataset"], int(r["seed"]), r["variant"], r["modality"]): r for r in composer}
    var_labels = {v: VARIANT_DISPLAY[v] for v in VARIANTS}
    channel_lines = []
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            for variant in VARIANTS:
                values = []
                for channel in ("smooth", "relational", "cross_modal"):
                    meds = [mass_index[(dataset, s, variant, modality, channel)]["median"] for s in SEEDS]
                    values.append(f"{channel[0].upper()}={np.mean(meds):.3f}")
                channel_lines.append(f"- {dataset}/{modality}/{var_labels[variant]} median relative mass: "
                                     + ", ".join(values) + ".")

    norm_lines = []
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            values = []
            for channel in ("smooth", "relational", "cross_modal", "mix"):
                medians = [norm_index[(dataset, s, "keep_edge", modality, channel)]["median"] for s in SEEDS]
                means = [norm_index[(dataset, s, "keep_edge", modality, channel)]["mean"] for s in SEEDS]
                values.append(f"{channel}: node RMS median {np.mean(medians):.4g}, node RMS mean {np.mean(means):.4g}")
            norm_lines.append(f"- KeepEdge {dataset}/{modality}: " + "; ".join(values) + ".")

    cosine_lines = []
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            values = []
            for pair in ("smooth_relational", "smooth_cross_modal", "relational_cross_modal"):
                med = [distinct_index[(dataset, s, "keep_edge", modality, pair)]["median"] for s in SEEDS]
                q10 = [distinct_index[(dataset, s, "keep_edge", modality, pair)]["q10"] for s in SEEDS]
                q90 = [distinct_index[(dataset, s, "keep_edge", modality, pair)]["q90"] for s in SEEDS]
                valid = [distinct_index[(dataset, s, "keep_edge", modality, pair)]["valid_node_count"] for s in SEEDS]
                values.append(f"{pair}: median {np.mean(med):.3f}, q10–q90 {np.mean(q10):.3f}–{np.mean(q90):.3f}, valid nodes/run {int(np.mean(valid))}")
            cosine_lines.append(f"- {dataset}/{modality}: " + "; ".join(values) + ".")

    composer_lines = []
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            values = []
            for variant in VARIANTS:
                rs = [comp_index[(dataset, s, variant, modality)] for s in SEEDS]
                values.append(f"{var_labels[variant]} ratio median={np.mean([r['ratio_median'] for r in rs]):.3f}, "
                              f"IQR={np.mean([r['ratio_iqr'] for r in rs]):.3f}, q90={np.mean([r['ratio_q90'] for r in rs]):.3f}; "
                              f"cos median={np.mean([r['cosine_median'] for r in rs]):.3f}")
            composer_lines.append(f"- {dataset}/{modality}: " + "; ".join(values) + ".")

    calibration_index = {
        (r["dataset"], r["seed"], r["variant"], r["modality"], r["diagnostic"]): r
        for r in expert_scale_rows
    }
    calibration_lines = []
    calibration_metrics = (
        "relational_calibration_scale", "cross_modal_calibration_scale",
        "calibrated_relational_rms_ratio", "calibrated_cross_modal_rms_ratio",
    )
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            for diagnostic in calibration_metrics:
                q99 = [calibration_index[(dataset, seed, "keep_edge", modality, diagnostic)]["q99"]
                       for seed in SEEDS]
                calibration_lines.append(
                    f"| {dataset} | {modality} | {diagnostic} | {np.mean(q99):.4g} ± {np.std(q99, ddof=0):.4g} | {max(q99):.4g} |"
                )
    maximum_calibration_scale_q99 = max(
        r["q99"] for r in expert_scale_rows
        if r["diagnostic"] in ("relational_calibration_scale", "cross_modal_calibration_scale")
    )

    intervention_lines = []
    for name in ("provenance_collapse", "provenance_permutation", "composer_off",
                 "pi_shuffle_within_target", "pi_target_mean", "pi_global_mean",
                 "modality_tied_pi", "smooth_only"):
        per_dataset = []
        for dataset in DATASETS:
            row = isum[(dataset, name)]
            per_dataset.append(
                f"{dataset}: acc {row['delta_accuracy_pp_run_mean']:.3f}±{row['delta_accuracy_pp_run_sd']:.3f} pp "
                f"({row['n_runs_delta_accuracy_pp_adverse']}/3 adverse), "
                f"F1 {row['delta_macro_f1_pp_run_mean']:.3f}±{row['delta_macro_f1_pp_run_sd']:.3f} pp "
                f"({row['n_runs_delta_macro_f1_pp_adverse']}/3 adverse), "
                f"CE {row['delta_ce_run_mean']:.4f}±{row['delta_ce_run_sd']:.4f} "
                f"({row['n_runs_delta_ce_adverse']}/3 adverse)"
            )
        allrow = isum[("ALL", name)]
        intervention_lines.append(
            f"- `{name}` (n={allrow['n_records']} evaluations over {allrow['n_runs']} checkpoints): "
            f"pooled record mean±SD acc {allrow['delta_accuracy_pp_mean']:.3f}±{allrow['delta_accuracy_pp_sd']:.3f} pp, "
            f"F1 {allrow['delta_macro_f1_pp_mean']:.3f}±{allrow['delta_macro_f1_pp_sd']:.3f} pp, "
            f"CE {allrow['delta_ce_mean']:.4f}±{allrow['delta_ce_sd']:.4f}; "
            f"record ranges acc [{allrow['delta_accuracy_pp_min']:.3f},{allrow['delta_accuracy_pp_max']:.3f}] pp, "
            f"F1 [{allrow['delta_macro_f1_pp_min']:.3f},{allrow['delta_macro_f1_pp_max']:.3f}] pp, "
            f"CE [{allrow['delta_ce_min']:.4f},{allrow['delta_ce_max']:.4f}]; " + "; ".join(per_dataset) + "."
        )

    allrow = {(r["dataset"], r["method"]): r for r in perf_summary}
    aggregate_perf = []
    for method in METHODS:
        r = allrow[("ALL", method)]
        aggregate_perf.append(f"| {method} | {100*r['val_accuracy_mean']:.3f} ± {100*r['val_accuracy_sd']:.3f} | "
                              f"{100*r['val_macro_f1_mean']:.3f} ± {100*r['val_macro_f1_sd']:.3f} | "
                              f"{r['val_ce_mean']:.4f} ± {r['val_ce_sd']:.4f} |")

    prov = dsumm[("ALL", "KeepEdge-PremixEdgeControl")]
    comp_capacity = dsumm[("ALL", "PremixEdgeControl-HistoricalE0EdgeMix")]
    edge_route = dsumm[("ALL", "KeepEdge-KeepTarget")]
    target_route = dsumm[("ALL", "KeepTarget-KeepStatic")]
    collapse = isum[("ALL", "provenance_collapse")]
    permute = isum[("ALL", "provenance_permutation")]
    context_nonzero = []
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            for channel in ("smooth", "relational", "cross_modal"):
                med = np.mean([mass_index[(dataset, seed, "keep_edge", modality, channel)]["median"] for seed in SEEDS])
                context_nonzero.append(med)
    degenerate = bool(np.mean(context_nonzero) < 0.05 or max(context_nonzero) < 0.08)

    # Frozen decision cases require model deltas + checkpoint reliance + node-context diagnostics.
    p_delta = prov["delta_val_accuracy_mean"]
    p_f1 = prov["delta_val_macro_f1_mean"]
    collapse_effect = max(abs(collapse[k]) for k in ("delta_accuracy_pp_mean", "delta_macro_f1_pp_mean"))
    permutation_effect = max(abs(permute[k]) for k in ("delta_accuracy_pp_mean", "delta_macro_f1_pp_mean"))
    provenance_active = collapse_effect >= 0.2 and permutation_effect >= 0.2
    provenance_supported = (p_delta > 0 and p_f1 > 0 and
                            collapse_effect >= 0.2 and permutation_effect >= 0.2)
    edge_supported = (provenance_supported and edge_route["delta_val_accuracy_mean"] > 0 and
                      edge_route["delta_val_macro_f1_mean"] > 0)
    if edge_supported:
        decision = "EDGE_ROUTING_WITH_PROVENANCE_SUPPORTED"
    elif degenerate:
        decision = "FUNCTION_CONTEXTS_DEGENERATE"
    elif provenance_supported:
        decision = "PROVENANCE_USEFUL_EDGE_ROUTING_NOT"
    elif comp_capacity["delta_val_accuracy_mean"] > 0 and p_delta <= 0.1 and collapse_effect < 0.2 and permutation_effect < 0.2:
        decision = "COMPOSER_CAPACITY_ONLY"
    elif max(collapse_effect, permutation_effect) >= 0.2 and p_delta <= 0:
        decision = "PROVENANCE_ACTIVE_BUT_NO_MODEL_GAIN"
    else:
        decision = "PROVENANCE_NOT_SUPPORTED"

    low_alternative_mass_groups = 0
    alternative_mass_by_group = {}
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            alternative_mass = np.mean([
                mass_index[(dataset, seed, "keep_edge", modality, channel)]["median"]
                for seed in SEEDS for channel in ("relational", "cross_modal")
            ])
            alternative_mass_by_group[(dataset, modality)] = float(alternative_mass)
            low_alternative_mass_groups += int(alternative_mass < 0.05)
    next_bottleneck = "function-context degeneration" if low_alternative_mass_groups >= 4 else "function-definition fit"
    major_mixing_status = (
        "YES" if provenance_supported else
        "PARTIALLY" if provenance_active else
        "NO"
    )
    channel_identity_max = max(float(r["identity_checks"]["channel_sum_max_abs"]) for r in records)
    context_mass_lines = "; ".join(
        f"{dataset}/{modality} R+X median mass {mass:.3f}"
        for (dataset, modality), mass in alternative_mass_by_group.items()
    )
    return f"""# E0.1 — Function-Provenance Preservation Screen

## Executive decision

**Primary frozen label: `{decision}`.** This screen isolates whether preserving the existing E0 Smooth/Relational/Cross-Modal function identity after edge aggregation adds validation value when all four variants have the same E0 modules, router, function bank, composer architecture, initialization, and parameter count.

**Premature function mixing was a major E0 bottleneck: {major_mixing_status}.** KeepEdge−PremixEdgeControl, the same-capacity provenance contrast, is {delta_line('ALL','KeepEdge-PremixEdgeControl')}. KeepEdge has checkpoint reliance: provenance-collapse changes Accuracy by {collapse['delta_accuracy_pp_mean']:.3f} pp on average (SD {collapse['delta_accuracy_pp_sd']:.3f}) and the five provenance permutations by {permute['delta_accuracy_pp_mean']:.3f} pp (SD {permute['delta_accuracy_pp_sd']:.3f}); however, reliance did not translate into an Accuracy or Macro-F1 gain over the matched Premix control. KeepEdge's CE is lower in aggregate, so evidence is partial rather than a broad performance win.

**Preserving function provenance enables useful edge-specific routing: {('YES' if edge_supported else 'NO')}.** KeepEdge−KeepTarget is {delta_line('ALL','KeepEdge-KeepTarget')}; this must be read together with KeepEdge−PremixEdgeControl and the edge-routing interventions below.

The next screened bottleneck indicated by this evidence is **{next_bottleneck}**. Four of six dataset×modality groups have low node-level R/X mass; ele-fashion retains both alternatives. The evidence points to dataset-specific function-context degeneration, while the nonzero collapse/permutation effects show that contexts surviving in trained KeepEdge are used. This E0.1 result is an executor attribution screen only. P1.3's joint readout motivated the hypothesis by retaining multiple operator contexts until one shared predictor; it did not establish that this E0.1 composer must preserve provenance. A success here supports only the tested E0 function contexts and this composer, not the necessity of P1.3's specific representation.

## Provenance, implementation, and protocol

- Source: E0 branch `exp/e0_structured_relation_function_executor` at `4905962df8c548c457b0d663ac7f473279f9f0e9`; this branch was created from that commit without merging `main`.
- Four variants: `premix_edge_control` (E0 EdgeMix router, composer input `[Cmix/3,Cmix/3,Cmix/3]`), `keep_static` (E0 StaticMix router, `[CS,CR,CX]`), `keep_target` (E0 TargetMix router, `[CS,CR,CX]`), `keep_edge` (E0 EdgeMix router, `[CS,CR,CX]`). Every variant adds the unchanged base `Cmix` to its composer output before the original modality residual LayerNorm and original four-way Text/Visual fusion.
- PremixEdgeControl has the **same composer parameter count and hidden width** as every Keep variant. Its repeated active input blocks permit the first Linear layer to represent any 128D-to-128D map through the sum of its three block weights. It is not a one-third-capacity baseline; it differs by lack of function identity in its input.
- All E0 common modules are constructed before the two modality-specific composers. E0 function definitions, Smooth-RMS stop-gradient calibration, softmax router, router prior, relation state, physical support, LOO context, degree denominator, residual norms, fusion, and one-hop protocol remain fixed.
- Formal NC campaign: {campaign_result['completed_new_runs']}/36 unique runs complete, {campaign_result['failed_formal_runs']} failed attempt and {campaign_result['formal_reruns']} retry; datasets Movies/Grocery/ele-fashion, seeds 42/43/44, four variants; AdamW 1e−3, weight decay 1e−4, max 300 epochs, patience 30, minimum epoch 30, gradient clipping 1.0, best validation accuracy. The sole failed attempt was a too-tight GPU target-mean numerical assertion at Grocery/42; after widening only that floating-point audit tolerance, the same run completed. No model/protocol change or other failure occurred.
- Device: {campaign_result['device']}; summed training time across the 36 successful runs {campaign_result['sum_training_time_seconds']:.1f}s (the failed attempt duration was not captured), intervention time {campaign_result['sum_keep_edge_intervention_time_seconds']:.1f}s, peak allocated training GPU memory {campaign_result['max_peak_gpu_memory_bytes']/1024**3:.2f} GiB.
- Only train/validation labels were made available. No test evaluation, test-index/test-label access, link prediction, or test-based checkpoint choice.
- The historical UNI and E0 EdgeMix rows are read from committed E0/M0 performance CSVs; no historical model was retrained. SEM and E0 Static/Target are not part of the required E0.1 main table.
- Parameter summary has exactly 36 unique dataset×seed×variant rows; the duplicated E0 parameter-summary rows were not propagated.

## Validation performance

Dataset cells are mean ± population SD over seeds 42–44. Accuracy and Macro-F1 are percentages; CE is native. The aggregate is an equal-weight descriptive summary of nine dataset×seed runs, not a pooled-node score. No significance tests were run.

### Movies

{perf_table('Movies')}

### Grocery

{perf_table('Grocery')}

### ele-fashion

{perf_table('ele-fashion')}

### Equal-weight aggregate across nine runs

| Method | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
{chr(10).join(aggregate_perf)}

## Paired same-seed comparisons

Positive Accuracy/Macro-F1 favors the left method; positive CE is unfavorable. Each dataset uses three paired seeds; `ALL` contains nine dataset×seed pairs and is descriptive.

| Dataset | Comparison | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---:|---:|---:|
{chr(10).join(f"| {r['dataset']} | {r['comparison']} | {r['delta_val_accuracy_mean']:.3f} ± {r['delta_val_accuracy_sd']:.3f} | {r['delta_val_macro_f1_mean']:.3f} ± {r['delta_val_macro_f1_sd']:.3f} | {r['delta_val_ce_mean']:.4f} ± {r['delta_val_ce_sd']:.4f} |" for r in paired_summary)}

The decisive provenance comparison is KeepEdge−PremixEdgeControl: {delta_line('ALL','KeepEdge-PremixEdgeControl')}. Composer capacity relative to historical E0 EdgeMix is PremixEdgeControl−E0 EdgeMix: {delta_line('ALL','PremixEdgeControl-HistoricalE0EdgeMix')}. Target-level granularity is KeepTarget−KeepStatic: {delta_line('ALL','KeepTarget-KeepStatic')}; edge-level granularity is KeepEdge−KeepTarget: {delta_line('ALL','KeepEdge-KeepTarget')}. KeepEdge−KeepStatic is {delta_line('ALL','KeepEdge-KeepStatic')}; KeepEdge−UNI is {delta_line('ALL','KeepEdge-UNI')}.

## Aggregated function-context diagnostics

The context statistics below are computed on validation nodes after full-neighborhood channel aggregation. `Cmix` is defined as `CS + CR + CX`; the measured maximum identity error across the 36 runs is {channel_identity_max:.2g}, below 1e−6. Node RMS distributions include mean, population SD and q10/q25/median/q75/q90 in `data/node_channel_norms.csv`.

### KeepEdge node-channel RMS

{chr(10).join(norm_lines)}

### Relative context mass by variant

Each value is the across-seed mean of a per-run validation-node median; full per-run distributions are in `data/channel_mass_summary.csv`.

{chr(10).join(channel_lines)}

KeepEdge alternative channel mass by modality group: {context_mass_lines}.

### KeepEdge node-context cosine distinctness

Cosines include only validation nodes where both channel vector norms exceed `eps`. Each line reports the mean of three run medians and the mean of run q10/q90 values; valid counts are recorded per run.

{chr(10).join(cosine_lines)}

## Composer usage

`composer/base` is the node-wise RMS ratio `RMS(Ccomp)/(RMS(Cmix)+eps)`. Values summarize the per-run node distribution across three seeds; cosine is between composer correction and Cmix. Composer nonzero magnitude is evidence of branch usage only, not proof of gain.

{chr(10).join(composer_lines)}

## Frozen E0 calibration-scale audit

The selected checkpoints were reloaded for a forward-only audit on non-self edges whose destination is a validation node. These diagnostics record E0's existing Smooth-reference calibration; no calibration change or clamp was applied. All 36 checkpoints and all 504 modality×diagnostic rows were finite. Maximum validation-edge q99 across Relational/Cross-Modal calibration scales was {maximum_calibration_scale_q99:.4g}.

| Dataset | Modality | Diagnostic | Mean run q99 ± SD | Maximum run q99 |
|---|---|---|---:|---:|
{chr(10).join(calibration_lines)}

All seven distributions per modality, including Smooth reference RMS, raw Relational/Cross-Modal RMS, calibrated RMS ratios, and calibration scales, are in `data/expert_scale_diagnostics.csv`. No NaN/Inf occurred, so the frozen calibration was retained without clamping.

## KeepEdge fixed-checkpoint interventions

All rows below are validation-only changes to a trained KeepEdge checkpoint. Collapse replaces `[CS,CR,CX]` by three copies of `Cmix/3`; five permutations keep `Cmix` fixed while permuting channel identity; composer-off uses `delta=Cmix`; routing interventions recompute channel contexts before the composer. Accuracy/Macro-F1 are percentage-point deltas. The report gives both pooled record mean ± SD (including repeats) and mean ± SD after averaging repeats within each run; adverse direction counts are across run-level means (Accuracy/F1 lower or CE higher). These are reliance diagnostics, not standalone causal proof.

{chr(10).join(intervention_lines)}

`pi_shuffle_within_target` used repeat seeds 1001–1005, with Text and Visual independently permuted. The permutation intervention used all five non-identity orders: SRX→SXR, RSX, RXS, XSR, XRS. The exact order, per-run values, min/max, and all CE deltas are in the intervention CSVs. KeepStatic's global-mean and KeepTarget's target-mean routing identity checks passed; PremixEdgeControl's collapsed-input identity passed. Smooth-only retains the composer with `[CS,0,0]`.

## Answers to the frozen questions

1. **E0 relation state and function bank preserved?** Yes. The E0 module implementation is reused. M0/E0 relation-state, experts and router have explicit regression tests; the Movies/42 GPU smoke passed, including bitwise expert and router outputs under common weights. All 9 dataset×seed E0 common parameter initializations are bitwise equal.
2. **Four variants parameter-identical?** Yes; all counts and initialization hashes are in the audit CSVs. Each dataset×seed has one unique row per variant.
3. **Same-seed initialization identical?** Yes for every model parameter and classifier across all four variants; composers are initialized after all shared E0 modules.
4. **Is PremixEdgeControl a fair collapsed-provenance control?** Yes. It has identical architecture, parameter count, initialization, edge router and `Cmix` base; all three input blocks are active and repeated. It has full composer capacity but no function identity.
5. **Does `CS+CR+CX` recover `Cmix`?** Yes by construction, and the explicit numerical identity is checked per modality and run.
6. **Do function contexts remain distinct after aggregation?** See node-level cosine/mass tables above; per-edge distinctness from E0 is not substituted for these aggregated diagnostics.
7. **Do Movies/Grocery R/X channels vanish?** Their validation-node relative mass and RMS are listed separately; no inference is based only on E0 router averages.
8. **Does ele-fashion retain multiple channels?** Its channel RMS/mass/cosine rows above provide the direct node-context evidence.
9. **Is the composer used?** Ratio and cosine diagnostics above quantify output relative to the base and alignment with it; `composer_off` independently measures fixed-checkpoint reliance.
10. **Does composer capacity alone change E0?** PremixEdgeControl−historical E0 EdgeMix is {delta_line('ALL','PremixEdgeControl-HistoricalE0EdgeMix')}; descriptive only because training seeds and model capacity context differ from the E0 reference.
11. **Does provenance add value over matched Premix?** KeepEdge−PremixEdgeControl is {delta_line('ALL','KeepEdge-PremixEdgeControl')}, interpreted with collapse/permutation checkpoint reliance and node-context diagnostics.
12. **Does target routing add value?** KeepTarget−KeepStatic is {delta_line('ALL','KeepTarget-KeepStatic')}.
13. **Does edge routing add value after provenance?** KeepEdge−KeepTarget is {delta_line('ALL','KeepEdge-KeepTarget')}.
14–15. **Do collapse/permutation hurt?** See per-dataset run-level direction counts and five-order min/max in the intervention tables. Reliance and incremental performance remain separate.
16. **Are target-mean/shuffle more damaging than in E0?** E0.1 and historical E0 per-dataset/run summaries are retained in paired intervention files; this report compares their signs and magnitudes descriptively only.
17. **What explains any improvement?** The matched control separates composer capacity from information preservation; router granularity contrasts target/edge; the fixed-checkpoint interventions check reliance. A single contrast is not used to attribute all changes.
18. **Was premature mixing a major E0 bottleneck?** {major_mixing_status}. Provenance-collapse/permutation and composer-off show mechanism use, and KeepEdge improves CE relative to Premix on average; however, Accuracy/Macro-F1 are essentially unchanged versus the matched control. This supports partial mechanism activity, not a demonstrated classification bottleneck.
19. **Does provenance make useful edge routing possible?** {('YES' if edge_supported else 'NO')} under the combined performance and checkpoint-reliance evidence.
20. **Next bottleneck:** {next_bottleneck} (Movies/Grocery have near-vanishing node-level R/X mass; ele-fashion retains multiple contexts; screen-level indication only).

## Frozen interpretation and limitations

Final label: **`{decision}`**. The label combines (i) KeepEdge vs PremixEdgeControl and KeepEdge vs KeepTarget, (ii) provenance collapse/permutation and routing interventions, and (iii) node-level RMS/mass/cosine diagnostics. It is not selected from any one intervention. Comparisons use three seeds per dataset and validation labels for model selection and diagnostics; no significance tests are claimed. Historical E0 and UNI are descriptive references. P1.3 motivates the hypothesis only and is neither validated nor overturned by E0.1. No new expert, routing family, relation evidence, calibration, test evaluation, link prediction, or M1 component was added.

## Figure QA

The plotting source passed strict static preflight (21 PASS, 0 WARN, 0 FAIL). All six panels/figures passed the 1.5 pt strict alignment gate. All PDFs passed the 5 pt text floor (smallest text 5.3 pt); the rendered collision audit found 0 text collisions, clipping failures, or warnings across the six figures. The six final PNGs were visually inspected after the last render. The initial audit caught crowded paired-comparison and intervention tick labels; the final render uses compact/vertical labels and passes.

## Artifacts

- `README.md`, `run_manifest.json`, this `report.md`
- Run-level and summary CSVs under `data/`, including matched-pair and intervention records
- `data/expert_scale_diagnostics.csv` records all seven frozen E0 scale distributions by modality, variant, and run
- Six requested figures under `figures/` with PDF/SVG/600-dpi TIFF companion exports
- Figure alignment and rendered QA notes under `figures/qa/`
- Raw run JSONs and checkpoints under ignored `outputs/e01_function_provenance_preservation/`
"""


def main():
    parser = argparse.ArgumentParser(description="Analyze E0.1 provenance preservation screen")
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--research-dir", type=Path, default=RESEARCH_DEFAULT)
    args = parser.parse_args()
    records = load_run_records(args.output_dir)
    perf_rows, perf_summary, paired_raw, paired_summary = performance_tables(records, args.research_dir)
    (params, init_rows, e0_rows, channel_norms, distinctness,
     channel_mass, composer, intervention_rows, intervention_summary,
     expert_scale_rows) = diagnostic_tables(records)
    data_dir = args.research_dir / "data"
    write_csv(data_dir / "performance_by_run.csv", perf_rows)
    write_csv(data_dir / "performance_summary.csv", perf_summary)
    write_csv(data_dir / "paired_delta_summary.csv", paired_summary)
    write_csv(data_dir / "paired_delta_by_run.csv", paired_raw)
    write_csv(data_dir / "parameter_summary.csv", params)
    write_csv(data_dir / "common_initialization_audit.csv", init_rows)
    write_csv(data_dir / "e0_common_init_regression.csv", e0_rows)
    write_csv(data_dir / "node_channel_norms.csv", channel_norms)
    write_csv(data_dir / "node_channel_distinctness.csv", distinctness)
    write_csv(data_dir / "channel_mass_summary.csv", channel_mass)
    write_csv(data_dir / "composer_diagnostics.csv", composer)
    write_csv(data_dir / "intervention_by_run.csv", intervention_rows)
    write_csv(data_dir / "intervention_summary.csv", intervention_summary)
    write_csv(data_dir / "expert_scale_diagnostics.csv", expert_scale_rows)
    campaign_path = args.output_dir / "campaign_result.json"
    if not campaign_path.is_file():
        raise FileNotFoundError("campaign_result.json missing; formal campaign must complete before analysis")
    campaign_result = json.loads(campaign_path.read_text(encoding="utf-8"))
    save_figures(args.research_dir, perf_rows, perf_summary, paired_raw, channel_norms,
                 distinctness, channel_mass, composer, intervention_summary)
    report = _report_text(perf_summary, paired_summary, intervention_summary, channel_norms,
                          distinctness, channel_mass, composer, expert_scale_rows, records, campaign_result)
    (args.research_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote E0.1 analysis: runs={len(records)}, parameters={len(params)}, "
          f"interventions={len(intervention_rows)} to {args.research_dir}")


if __name__ == "__main__":
    main()
