from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("smooth_only", "static_strength", "same_state_strength", "cross_state_strength")
COMPARISONS = (
    ("static_strength", "smooth_only", "static-smooth"),
    ("same_state_strength", "static_strength", "same-static"),
    ("cross_state_strength", "same_state_strength", "cross-same"),
    ("cross_state_strength", "smooth_only", "cross-smooth"),
)
METRICS = ("val_accuracy", "val_macro_f1", "val_ce")


def _save_figure(fig, figures_dir: Path, stem: str) -> None:
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as exc:
        raise RuntimeError("Panel-alignment helper is required for N1 multi-panel figures") from exc
    require_matplotlib_panel_alignment(
        fig,
        json_out=figures_dir / f"{stem}.alignment.json",
        overlay_svg=figures_dir / f"{stem}.alignment-overlay.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    svg_path = figures_dir / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})


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


def _mean_sd(values: list[float]) -> tuple[float, float]:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not clean.size:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


def _metric_summary(rows: list[dict[str, Any]], group_fields: tuple[str, ...],
                    value_field: str, metric_label: str) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_fields)].append(float(row[value_field]))
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        mean, std = _mean_sd(values)
        output.append({**dict(zip(group_fields, key)), "metric": metric_label,
                       "mean": mean, "population_sd": std, "n": len(values)})
    return output


def _summary_for_performance(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for metric in METRICS:
        output += _metric_summary(runs, ("dataset", "variant"), metric, metric)
        pooled = [{**row, "dataset": "ALL_9_RUNS"} for row in runs]
        output += _metric_summary(pooled, ("dataset", "variant"), metric, metric)
    return output


def _paired_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in runs}
    raw = []
    for left, right, label in COMPARISONS:
        for dataset in DATASETS:
            for seed in (42, 43, 44):
                a, b = lookup.get((dataset, seed, left)), lookup.get((dataset, seed, right))
                if a is None or b is None:
                    continue
                row = {"dataset": dataset, "seed": seed, "comparison": label,
                       "left_variant": left, "right_variant": right}
                for metric in METRICS:
                    row[f"delta_{metric}"] = float(a[metric]) - float(b[metric])
                raw.append(row)
    summary = []
    for comparison in (label for _, _, label in COMPARISONS):
        subsets = [(dataset, [r for r in raw if r["comparison"] == comparison and r["dataset"] == dataset])
                   for dataset in DATASETS]
        subsets.append(("ALL_9_PAIRS", [r for r in raw if r["comparison"] == comparison]))
        for dataset, rows in subsets:
            for metric in METRICS:
                values = [float(r[f"delta_{metric}"]) for r in rows]
                mean, std = _mean_sd(values)
                summary.append({
                    "dataset": dataset, "comparison": comparison, "metric": metric,
                    "mean_delta": mean, "population_sd": std, "n_pairs": len(values),
                    "positive_runs": sum(v > 0 for v in values),
                    "negative_runs": sum(v < 0 for v in values),
                    "tie_runs": sum(v == 0 for v in values),
                })
    return raw, summary


def _flatten_diagnostics(runs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for item in run.get("diagnostics", {}).get(key, []):
            rows.append(item)
    return rows


def _intervention_tables(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = []
    for run in runs:
        raw.extend(run.get("interventions", []))
    summary = []
    group_names = sorted({row["intervention"] for row in raw})
    for intervention in group_names:
        groups = [(dataset, [r for r in raw if r["intervention"] == intervention and r["dataset"] == dataset])
                  for dataset in DATASETS]
        groups.append(("ALL_9_RUNS", [r for r in raw if r["intervention"] == intervention]))
        for dataset, rows in groups:
            for metric in ("delta_accuracy", "delta_macro_f1", "delta_ce"):
                values = [float(r[metric]) for r in rows]
                mean, std = _mean_sd(values)
                per_run_sd = []
                if intervention == "node_beta_shuffle":
                    for seed in sorted({int(r["seed"]) for r in rows}):
                        repeat_values = [float(r[metric]) for r in rows if int(r["seed"]) == seed]
                        if len(repeat_values) > 1:
                            per_run_sd.append(float(np.std(repeat_values, ddof=0)))
                summary.append({
                    "dataset": dataset, "intervention": intervention, "metric": metric,
                    "mean_delta": mean, "population_sd": std, "n_repeat_records": len(values),
                    "n_runs": len({(r["dataset"], int(r["seed"])) for r in rows}),
                    "positive_records": sum(v > 0 for v in values),
                    "negative_records": sum(v < 0 for v in values),
                    "tie_records": sum(v == 0 for v in values),
                    "mean_within_run_repeat_sd": float(np.mean(per_run_sd)) if per_run_sd else float("nan"),
                })
    return raw, summary


def _performance_figure(summary: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    labels = ["Smooth", "Static", "Same", "Cross"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for ax, metric, title in zip(axes, METRICS, ("Validation accuracy", "Macro-F1", "Cross-entropy")):
        for dataset_i, dataset in enumerate(DATASETS):
            rows = [r for r in summary if r["dataset"] == dataset and r["metric"] == metric]
            rows = sorted(rows, key=lambda row: VARIANTS.index(row["variant"]))
            xs = np.arange(4) + (dataset_i - 1) * 0.22
            ax.errorbar(xs, [r["mean"] for r in rows], yerr=[r["population_sd"] for r in rows],
                        marker="o", capsize=2, linestyle="none", label=dataset)
        ax.set_xticks(np.arange(4), labels)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
            tick.set_rotation_mode("anchor")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Accuracy / Macro-F1 (fraction)")
    axes[2].set_ylabel("CE")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(DATASETS),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, path.parent, path.stem)
    plt.close(fig)


def _paired_figure(summary: list[dict[str, Any]], path: Path) -> None:
    comparisons = ["static-smooth", "same-static", "cross-same", "cross-smooth"]
    metrics = METRICS
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex="col")
    for i, dataset in enumerate(DATASETS):
        for j, comparison in enumerate(comparisons):
            ax = axes[i, j]
            rows = [r for r in summary if r["dataset"] == dataset and r["comparison"] == comparison]
            rows = {r["metric"]: r for r in rows}
            vals = [rows[m]["mean_delta"] * (100 if m != "val_ce" else 1) for m in metrics]
            sd = [rows[m]["population_sd"] * (100 if m != "val_ce" else 1) for m in metrics]
            ax.errorbar(np.arange(3), vals, yerr=sd, marker="o", capsize=2,
                        linestyle="none", color="#4c78a8")
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_xticks(np.arange(3), ["Acc", "F1", "CE"])
            ax.grid(axis="y", alpha=.25)
            if i == 0:
                ax.set_title(comparison)
            if j == 0:
                ax.set_ylabel(f"{dataset}\nΔ pp; CE native")
    fig.tight_layout()
    _save_figure(fig, path.parent, path.stem)
    plt.close(fig)


def _strength_figure(strength_rows, variation_rows, contribution_rows, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    dyn = ("same_state_strength", "cross_state_strength")
    labels = []
    for dataset in DATASETS:
        for variant in dyn:
            labels.append(f"{dataset}\n{'Same' if variant.startswith('same') else 'Cross'}")
    colors = {"D": "#4c78a8", "P": "#f58518"}
    for ax, rows, metric, title in (
        (axes[0], strength_rows, "mean", "Mean strength β"),
        (axes[1], variation_rows, "validation_std_i", "Node-wise β standard deviation"),
        (axes[2], contribution_rows, "median", "Effective alternative / Smooth contribution"),
    ):
        x_positions = np.arange(len(labels))
        for fn, offset in (("D", -0.08), ("P", 0.08)):
            vals, errors = [], []
            for dataset in DATASETS:
                for variant in dyn:
                    if rows is variation_rows:
                        match = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant
                                 and r.get("modality") == "text" and r.get("function") == fn]
                    else:
                        group = f"beta_{fn}" if rows is strength_rows else f"effective_A_{fn}"
                        match = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant
                                 and r.get("modality") == "text" and r.get("group") == group
                                 and r.get("scope") == "validation_targets"]
                    mean, sd = _mean_sd([float(row[metric]) for row in match])
                    vals.append(mean); errors.append(sd)
            ax.errorbar(x_positions + offset, vals, yerr=errors, marker="o", linestyle="none",
                        capsize=2, label=fn, color=colors[fn])
        ax.set_title(title)
        ax.set_xticks(x_positions, labels, fontsize=7)
        ax.grid(axis="y", alpha=.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
    fig.suptitle("Text modality; mean ± population SD across seeds", y=1.10, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save_figure(fig, path.parent, path.stem)
    plt.close(fig)


def _channel_scale_figure(rms_rows, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dataset in zip(axes, DATASETS):
        x = np.arange(4)
        for fn, color, offset in (("D_over_S", "#4c78a8", -0.08), ("P_over_S", "#f58518", 0.08)):
            vals, errors = [], []
            group_name = "RMS_ratio_D_over_S" if fn == "D_over_S" else "RMS_ratio_P_over_S"
            for variant in VARIANTS:
                rows = [r["median"] for r in rms_rows if r["dataset"] == dataset
                        and r["variant"] == variant and r.get("group") == group_name]
                mean, sd = _mean_sd(rows)
                vals.append(mean); errors.append(sd)
            ax.errorbar(x + offset, vals, yerr=errors, marker="o", capsize=2, color=color,
                        linestyle="none", label=fn.replace("_over_S", "/ Smooth"))
        ax.set_title(dataset); ax.set_xticks(x, ["Smooth", "Static", "Same", "Cross"])
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_rotation_mode("anchor")
        ax.tick_params(axis="x", length=2, pad=8)
        ax.grid(axis="y", alpha=.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(DATASETS),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, path.parent, path.stem)
    plt.close(fig)


def _intervention_figure(summary, path: Path) -> None:
    interventions = ["D_off", "P_off", "DP_off", "validation_global_mean", "node_beta_shuffle", "modality_tied"]
    metrics = ("delta_accuracy", "delta_macro_f1", "delta_ce")
    titles = ("Δ validation accuracy", "Δ Macro-F1", "Δ CE")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(interventions))
        for ds_i, dataset in enumerate(DATASETS):
            means, sds = [], []
            for intervention in interventions:
                rows = [r for r in summary if r["dataset"] == dataset and r["intervention"] == intervention
                        and r["metric"] == metric]
                factor = 100 if metric in {"delta_accuracy", "delta_macro_f1"} else 1
                means.append(float(rows[0]["mean_delta"]) * factor if rows else np.nan)
                sds.append(float(rows[0]["population_sd"]) * factor if rows else np.nan)
            ax.errorbar(x + (ds_i - 1) * 0.18, means, yerr=sds, marker="o", capsize=2,
                        linestyle="none", label=dataset)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(title)
        ax.set_ylabel("Percentage points" if metric != "delta_ce" else "Native CE")
        ax.set_xticks(x, [s.replace("_", "\n") for s in interventions], fontsize=7)
        ax.tick_params(axis="x", length=2, pad=8)
        ax.grid(axis="y", alpha=.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(DATASETS),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(fig, path.parent, path.stem)
    plt.close(fig)


def analyze_campaign(out_dir: Path, research_dir: Path, campaign_manifest: dict[str, Any]) -> dict[str, Any]:
    run_paths = sorted((out_dir / "runs").glob("*/*/*.json"))
    runs = []
    for path in run_paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") == "completed" and not row.get("smoke", False):
            runs.append(row)
    runs.sort(key=lambda row: (DATASETS.index(row["dataset"]), int(row["seed"]), VARIANTS.index(row["variant"])))
    data_dir, figures_dir = research_dir / "data", research_dir / "figures"
    performance_rows = [{key: row.get(key) for key in (
        "dataset", "seed", "variant", *METRICS, "best_epoch", "epochs_run", "model_params",
        "classifier_params", "total_params", "training_time_sec", "intervention_time_sec",
        "peak_gpu_memory_bytes", "protocol", "evaluate_test", "test_idx_attached", "test_labels_exposed",
    )} for row in runs]
    performance_summary = _summary_for_performance(runs)
    paired_raw, paired_summary = _paired_rows(runs)
    strength_rows = _flatten_diagnostics(runs, "strength")
    variation_rows = _flatten_diagnostics(runs, "variation")
    contribution_rows = _flatten_diagnostics(runs, "contribution")
    channel_rms_rows = _flatten_diagnostics(runs, "channel_rms")
    distinct_rows = _flatten_diagnostics(runs, "distinctness")
    disagreement_rows = _flatten_diagnostics(runs, "disagreement")
    intervention_rows, intervention_summary = _intervention_tables(runs)
    _write_csv(data_dir / "performance_by_run.csv", performance_rows)
    _write_csv(data_dir / "performance_summary.csv", performance_summary)
    _write_csv(data_dir / "paired_delta_by_run.csv", paired_raw)
    _write_csv(data_dir / "paired_delta_summary.csv", paired_summary)
    _write_csv(data_dir / "strength_summary.csv", strength_rows)
    _write_csv(data_dir / "strength_node_variation.csv", variation_rows)
    _write_csv(data_dir / "effective_contribution_summary.csv", contribution_rows)
    _write_csv(data_dir / "channel_rms_summary.csv", channel_rms_rows)
    _write_csv(data_dir / "channel_distinctness.csv", distinct_rows)
    _write_csv(data_dir / "modality_strength_disagreement.csv", disagreement_rows)
    _write_csv(data_dir / "intervention_by_run.csv", intervention_rows)
    _write_csv(data_dir / "intervention_summary.csv", intervention_summary)

    _performance_figure(performance_summary, figures_dir / "n1_performance.png")
    _paired_figure(paired_summary, figures_dir / "n1_paired_deltas.png")
    _strength_figure(strength_rows, variation_rows, contribution_rows,
                     figures_dir / "n1_strength_usage.png")
    _channel_scale_figure(channel_rms_rows, figures_dir / "n1_channel_scale.png")
    _intervention_figure(intervention_summary, figures_dir / "n1_interventions.png")

    run_manifest = {
        **campaign_manifest,
        "analyzed_run_count": len(runs),
        "run_records": [{key: row.get(key) for key in (
            "dataset", "seed", "variant", "status", "best_epoch", "val_accuracy", "val_macro_f1",
            "val_ce", "training_time_sec", "peak_gpu_memory_bytes", "checkpoint",
        )} for row in runs],
        "test_and_lp_boundary": {"evaluate_test": False, "test_labels_exposed": False, "link_prediction": False},
        "analysis_notes": [
            "All paired comparisons match dataset and seed.",
            "The pooled nine-run summaries are descriptive and are not IID significance tests.",
            "Checkpoint interventions measure reliance in trained cross-state checkpoints, not retrained architecture value.",
        ],
    }
    research_dir.mkdir(parents=True, exist_ok=True)
    (research_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, allow_nan=True), encoding="utf-8")
    (research_dir / "README.md").write_text(
        "# N1 — Recipient-Conditioned Function-Strength Mixing Screen\n\n"
        "Validation-only MAG node-classification mechanism screen on Movies, Grocery and ele-fashion; seeds 42–44.\n\n"
        "Run from the repository root with `conda activate yhf_env`, then `python scripts/run_n1_recipient_function_strength_mixer.py --smoke --device cuda:1` and `python scripts/run_n1_recipient_function_strength_mixer.py --campaign --regression --device cuda:1`.\n\n"
        "The campaign uses full-graph training, train labels only, validation-accuracy checkpoint selection and the frozen unified NC optimizer/scheduler/early-stop settings. `task.evaluate_test=false`; no test indices or link-prediction task are used.\n\n"
        "Machine-readable tables are in `data/`; figures are in `figures/`; the N1 implementation and checks live in `src/models/adaptive_prop_n1.py`, `scripts/` and `tests/`. Large run records and checkpoints are gitignored under `outputs/n1_recipient_function_strength_mixer/`.\n",
        encoding="utf-8",
    )
    _write_report(research_dir / "report.md", runs, performance_summary, paired_summary,
                  intervention_summary, campaign_manifest)
    return run_manifest


def _write_report(path: Path, runs, performance_summary, paired_summary, intervention_summary, manifest):
    lines = [
        "# N1 — Recipient-Conditioned Function-Strength Mixing Screen",
        "",
        "## Status and protocol",
        "",
        f"Campaign completed {manifest.get('completed_runs', 0)}/36 runs; recorded failures: {len(manifest.get('failures', []))}.",
        f"Source SHA: `{manifest.get('source_sha', 'not recorded')}`; branch: `{manifest.get('experiment_branch', 'not recorded')}`.",
        f"Device: `{manifest.get('device', 'not recorded')}`; test evaluation disabled: `{manifest.get('evaluate_test', False)}`.",
        "All performance comparisons pair the same dataset and seed. Nine-run pooled summaries are descriptive, not IID tests.",
        "",
        "## Validation performance",
        "",
        "Accuracy and Macro-F1 are fractions; CE is native scale. Values are mean ± population SD over seeds.",
        "",
    ]
    lookup = {(r["dataset"], r["variant"], r["metric"]): r for r in performance_summary}
    for dataset in (*DATASETS, "ALL_9_RUNS"):
        lines += [f"### {dataset}", "", "| Variant | Accuracy | Macro-F1 | CE |", "|---|---:|---:|---:|"]
        for variant in VARIANTS:
            cells = []
            for metric in METRICS:
                row = lookup.get((dataset, variant, metric))
                cells.append(f"{row['mean']:.4f} ± {row['population_sd']:.4f}" if row else "—")
            lines.append(f"| {variant} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
    lines += [
        "## Paired deltas",
        "",
        "Positive values are left minus right; accuracy and Macro-F1 use fraction units. Counts are paired run directions.",
        "",
        "| Dataset | Comparison | Metric | Mean ± population SD | + / − / tie |",
        "|---|---|---|---:|---:|",
    ]
    for row in paired_summary:
        if row["dataset"] == "ALL_9_PAIRS":
            continue
        lines.append(f"| {row['dataset']} | {row['comparison']} | {row['metric']} | "
                     f"{row['mean_delta']:.5f} ± {row['population_sd']:.5f} | "
                     f"{row['positive_runs']} / {row['negative_runs']} / {row['tie_runs']} |")
    lines += [
        "",
        "## Checkpoint interventions",
        "",
        "These interventions are inference-only reliance checks on `cross_state_strength`; they do not replace retrained controls.",
        "See `data/intervention_by_run.csv` and `data/intervention_summary.csv` for per-repeat values and run-level shuffle spread.",
        "",
        "## Strength and channel diagnostics",
        "",
        "Beta summaries, node-wise variation, effective contributions, channel RMS, pairwise channel cosine and Text/Visual strength differences are in the corresponding CSVs under `data/`.",
        "",
        "## Scientific judgment",
        "",
        "Interpret the matched retrained comparisons (`static-smooth`, `same-static`, `cross-same`) separately from checkpoint interventions. Beta variation alone is not evidence of task value; beta must be read together with channel RMS and effective contribution.",
        "",
        "Final mechanism label and the recommendation about expanding to Toys / Reddit-S should be completed after reviewing the per-dataset deltas, node variation and checkpoint interventions.",
        "",
        "## Artifacts",
        "",
        "See `README.md`, `run_manifest.json`, `data/` and `figures/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
