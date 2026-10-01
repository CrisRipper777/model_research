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
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "e0_structured_relation_function_executor"
DEFAULT_RESEARCH = ROOT / "research" / "e0_structured_relation_function_executor"
QA_DIR = DEFAULT_RESEARCH / "figures" / "qa"
_FIGURE_SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(_FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_FIGURE_SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("static_mix", "target_mix", "edge_mix")
METHODS = ("SEM", "UNI", "StaticMix", "TargetMix", "EdgeMix")
METRICS = ("val_accuracy", "val_macro_f1", "val_ce")
DISPLAY = {"static_mix": "StaticMix", "target_mix": "TargetMix", "edge_mix": "EdgeMix"}
METHOD_VARIANT = {value: key for key, value in DISPLAY.items()}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalize_svg_text(path: Path) -> None:
    """Remove renderer-added line-end spaces; SVG path newlines remain separators."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip(" \t\r") for line in lines) + "\n", encoding="utf-8")


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def load_runs(out_dir: Path) -> list[dict[str, Any]]:
    records = []
    for dataset in DATASETS:
        for seed in (42, 43, 44):
            for variant in VARIANTS:
                path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"missing formal E0 run: {path}")
                record = load_json(path)
                if record.get("status") != "completed":
                    raise RuntimeError(f"incomplete E0 run: {path}")
                records.append(record)
    if len(records) != 27:
        raise AssertionError(f"expected 27 completed E0 records, got {len(records)}")
    return records


def performance_tables(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_path = ROOT / "research" / "m0_adaptive_propagation_screen" / "data" / "performance_by_run.csv"
    with baseline_path.open(newline="", encoding="utf-8") as stream:
        baseline = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append({"dataset": record["dataset"], "seed": record["seed"],
                     "method": DISPLAY[record["variant"]], "variant": record["variant"],
                     **{metric: record[metric] for metric in METRICS},
                     "trainable_params": record["total_trainable_params"],
                     "best_epoch": record["best_epoch"], "epochs_run": record["epochs_run"],
                     "training_time_sec": record["training_time_sec"],
                     "intervention_time_sec": record["intervention_time_sec"],
                     "peak_gpu_memory_gb": record["peak_gpu_memory_bytes"] / 1024**3})
    for record in baseline:
        if record["variant"] not in ("semantic", "uniform"):
            continue
        method = "SEM" if record["variant"] == "semantic" else "UNI"
        rows.append({"dataset": record["dataset"], "seed": int(record["seed"]), "method": method,
                     "variant": record["variant"], **{metric: float(record[metric]) for metric in METRICS},
                     "trainable_params": int(record["trainable_params"]),
                     "best_epoch": int(record["best_epoch"]), "epochs_run": int(record["epochs_run"]),
                     "training_time_sec": float(record["training_time_sec"]),
                     "intervention_time_sec": float(record["intervention_time_sec"]),
                     "peak_gpu_memory_gb": float(record["peak_gpu_memory_gb"])})
    if len(rows) != 45:
        raise AssertionError(f"expected 27 E0 + 18 historical UNI/SEM reference rows, got {len(rows)}")

    summaries = []
    paired = []
    for dataset in (*DATASETS, "ALL"):
        group = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        for method in METHODS:
            values = [row for row in group if row["method"] == method]
            summary: dict[str, Any] = {"dataset": dataset, "method": method, "n": len(values)}
            for metric in METRICS:
                avg, sd = mean_sd([float(row[metric]) for row in values])
                summary[f"{metric}_mean"] = avg
                summary[f"{metric}_sd"] = sd
            summaries.append(summary)
        by_key = {(row["dataset"], row["seed"], row["method"]): row for row in rows}
        comparisons = (("StaticMix-UNI", "StaticMix", "UNI"),
                       ("TargetMix-StaticMix", "TargetMix", "StaticMix"),
                       ("EdgeMix-TargetMix", "EdgeMix", "TargetMix"),
                       ("EdgeMix-StaticMix", "EdgeMix", "StaticMix"),
                       ("EdgeMix-UNI", "EdgeMix", "UNI"))
        if dataset == "ALL":
            pairs = [(seed_dataset, seed) for seed_dataset in DATASETS for seed in (42, 43, 44)]
        else:
            pairs = [(dataset, seed) for seed in (42, 43, 44)]
        for label, left, right in comparisons:
            selected = []
            for seed_dataset, seed in pairs:
                selected.append((by_key[(seed_dataset, seed, left)], by_key[(seed_dataset, seed, right)]))
            row: dict[str, Any] = {"dataset": dataset, "comparison": label, "n_pairs": len(selected)}
            for metric in METRICS:
                deltas = [float(a[metric]) - float(b[metric]) for a, b in selected]
                avg, sd = mean_sd(deltas)
                scale = 100.0 if metric != "val_ce" else 1.0
                row[f"delta_{metric}_mean"] = avg * scale
                row[f"delta_{metric}_sd"] = sd * scale
                row[f"{metric}_units"] = "percentage_points" if metric != "val_ce" else "native"
            paired.append(row)
    return rows, summaries, paired


def diagnostic_tables(records: list[dict[str, Any]]):
    router, within, disagreement, distinct, scales = [], [], [], [], []
    intervention_rows, parameter_rows, init_rows = [], [], []
    for record in records:
        dataset, seed, variant = record["dataset"], record["seed"], record["variant"]
        diag = record["diagnostics"]
        for item in diag["router"]:
            router.append({"dataset": dataset, "seed": seed, **item})
        for item in diag["within_node"]:
            within.append({"dataset": dataset, "seed": seed, **item})
        disagreement.append({"dataset": dataset, "seed": seed, **diag["modality_disagreement"]})
        for item in diag["function_distinctness"]:
            distinct.append({"dataset": dataset, "seed": seed, **item})
        for item in diag["expert_scales"]:
            scales.append({"dataset": dataset, "seed": seed, **item})
        audit = record["initialization_audit"]
        counts = audit["parameter_counts"]
        for name, count in counts.items():
            parameter_rows.append({"dataset": dataset, "seed": seed, "variant": name,
                                   **count, "classifier_params": record["classifier_params"],
                                   "total_trainable_params": count["trainable_model_params"] + record["classifier_params"]})
        if variant == "static_mix":
            init_rows.append({
                "dataset": dataset, "seed": seed,
                "hash_static_mix": audit["variant_hashes"]["static_mix"],
                "hash_target_mix": audit["variant_hashes"]["target_mix"],
                "hash_edge_mix": audit["variant_hashes"]["edge_mix"],
                "static_target_bitwise_equal": audit["pairwise_bitwise_equal"]["static_mix__target_mix"],
                "static_edge_bitwise_equal": audit["pairwise_bitwise_equal"]["static_mix__edge_mix"],
                "target_edge_bitwise_equal": audit["pairwise_bitwise_equal"]["target_mix__edge_mix"],
                "common_all_bitwise_equal": audit["common_all_bitwise_equal"],
                "exact_trainable_parameter_match": audit["exact_trainable_parameter_match"],
                "model_params_static": audit["parameter_counts"]["static_mix"]["model_params"],
                "model_params_target": audit["parameter_counts"]["target_mix"]["model_params"],
                "model_params_edge": audit["parameter_counts"]["edge_mix"]["model_params"],
                "trainable_model_params_static": audit["parameter_counts"]["static_mix"]["trainable_model_params"],
                "trainable_model_params_target": audit["parameter_counts"]["target_mix"]["trainable_model_params"],
                "trainable_model_params_edge": audit["parameter_counts"]["edge_mix"]["trainable_model_params"],
            })
        interventions = record.get("interventions")
        if interventions:
            for item in interventions["rows"]:
                intervention_rows.append({"dataset": dataset, "seed": seed,
                    "base_val_accuracy": interventions["normal"]["val_accuracy"],
                    "base_val_macro_f1": interventions["normal"]["val_macro_f1"],
                    "base_val_ce": interventions["normal"]["val_ce"], **item})

    intervention_summary = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in intervention_rows:
        groups[(row["dataset"], row["intervention"])].append(row)
        groups[("ALL", row["intervention"])].append(row)
    for (dataset, intervention), rows in sorted(groups.items()):
        item: dict[str, Any] = {"dataset": dataset, "intervention": intervention, "n_records": len(rows),
                                "n_runs": len({(row["dataset"], row["seed"]) for row in rows})}
        for metric in ("delta_accuracy", "delta_macro_f1", "delta_ce"):
            values = [float(row[metric]) for row in rows]
            avg, sd = mean_sd(values)
            scale = 100.0 if metric != "delta_ce" else 1.0
            item[f"{metric}_mean"] = avg * scale
            item[f"{metric}_sd"] = sd * scale
        intervention_summary.append(item)
    return router, within, disagreement, distinct, scales, intervention_rows, intervention_summary, parameter_rows, init_rows


def save_figures(research_dir: Path, rows, router, within, disagreement, distinct, intervention_summary):
    figdir = research_dir / "figures"
    qadir = figdir / "qa"
    figdir.mkdir(parents=True, exist_ok=True)
    qadir.mkdir(parents=True, exist_ok=True)
    colors = {"SEM": "#9b9b9b", "UNI": "#34495e", "StaticMix": "#4c78a8",
              "TargetMix": "#f2a541", "EdgeMix": "#d45050"}
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "figure.dpi": 150, "savefig.dpi": 600})

    # E0 figure contract: the central claim is that edge-specific routing adds
    # little over target-level routing; panels show task deltas, route variation,
    # modality disagreement, and calibrated function separation as distinct evidence.
    def export(name: str, fig):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig,
            json_out=qadir / f"{name}.alignment.json",
            overlay_svg=qadir / f"{name}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
        alignment_svg = qadir / f"{name}.alignment.svg"
        if alignment_svg.is_file():
            normalize_svg_text(alignment_svg)
        fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
        figure_svg = figdir / f"{name}.svg"
        fig.savefig(figure_svg, bbox_inches="tight")
        normalize_svg_text(figure_svg)
        fig.savefig(figdir / f"{name}.png", dpi=600, bbox_inches="tight")
        fig.savefig(figdir / f"{name}.tiff", dpi=600, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4))
    for ax, metric, title, scale in zip(axes, METRICS, ("Accuracy", "Macro-F1", "Cross-entropy"), (100, 100, 1)):
        positions = np.arange(len(DATASETS))
        width = 0.14
        for j, method in enumerate(METHODS):
            means, sds = [], []
            for dataset in DATASETS:
                values = [float(row[metric]) * scale for row in rows if row["dataset"] == dataset and row["method"] == method]
                mean, sd = mean_sd(values); means.append(mean); sds.append(sd)
            ax.errorbar(positions + (j - 2) * width, means, yerr=sds, fmt="o", capsize=2,
                        color=colors[method], label=method, markersize=4)
        ax.set_xticks(positions, ("Movies", "Grocery", "Fashion"))
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
        if metric != "val_ce": ax.set_ylabel("Validation (%)")
        else: ax.set_ylabel("Validation CE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .92)); export("e0_performance_screen", fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), sharey=True)
    functions = ("null", "smooth", "relational", "cross_modal")
    function_colors = {"null":"#aaaaaa", "smooth":"#4c78a8", "relational":"#f2a541", "cross_modal":"#d45050"}
    for ax, dataset in zip(axes, DATASETS):
        for variant_index, variant in enumerate(VARIANTS):
            subset = [row for row in router if row["dataset"] == dataset and row["variant"] == variant]
            for modality_index, modality in enumerate(("text", "visual")):
                found = next(row for row in subset if row["modality"] == modality)
                x = variant_index * 3 + modality_index
                bottom = 0
                for function in functions:
                    value = float(found[f"pi_{function}_mean"])
                    ax.bar(x, value, bottom=bottom, width=.72, color=function_colors[function],
                           alpha=1.0 if modality == "text" else .6, linewidth=0)
                    bottom += value
        ax.set_title(dataset)
        ax.set_ylim(0, 1)
        ax.set_xticks([.5, 3.5, 6.5], ["StaticMix", "TargetMix", "EdgeMix"], fontsize=7)
        ax.tick_params(axis="both", which="both", length=0, pad=8)
    axes[0].set_ylabel("Mean routing probability")
    legend_items = [Patch(facecolor=function_colors[name], label=name.replace("_", "-")) for name in functions]
    legend_items.extend([Patch(facecolor="#555555", alpha=1.0, label="Text"),
                         Patch(facecolor="#555555", alpha=.6, label="Visual")])
    fig.legend(handles=legend_items, frameon=False, ncol=6, handletextpad=1.2, columnspacing=1.5,
               loc="upper center", bbox_to_anchor=(.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, .9)); export("e0_routing_usage", fig)

    edge_rows = [row for row in within if row["variant"] == "edge_mix" and row.get("eligible_target_count", 0) > 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, modality in zip(axes, ("text", "visual")):
        vals = [[float(row.get("within_node_l1_median", np.nan)) for row in edge_rows
                 if row["dataset"] == dataset and row["modality"] == modality] for dataset in DATASETS]
        ax.boxplot(vals, tick_labels=DATASETS, showfliers=False)
        ax.set_title(f"{modality.title()} EdgeMix"); ax.set_ylabel("Within-target L1 routing variation")
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); export("e0_within_node_routing", fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels, means, sds, cols = [], [], [], []
    for dataset in DATASETS:
        for variant in VARIANTS:
            subset = [r for r in disagreement if r["dataset"] == dataset and r["variant"] == variant]
            if subset:
                labels.append(f"{dataset}\n{variant.replace('_mix','')}")
                avg, sd = mean_sd([float(r["mean_l1_distance"]) for r in subset]); means.append(avg); sds.append(sd)
                cols.append(colors[DISPLAY[variant]])
    ax.bar(np.arange(len(labels)), means, yerr=sds, capsize=3, color=cols)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=8); ax.set_ylabel("Mean L1(pi Text, pi Visual)")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    export("e0_modality_disagreement", fig)

    pairs = (("cos_smooth_relational", "Smooth–Relational"),
             ("cos_smooth_cross_modal", "Smooth–Cross-modal"),
             ("cos_relational_cross_modal", "Relational–Cross-modal"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    for ax, (pair, title) in zip(axes, pairs):
        vals = []
        labels = []
        for dataset in DATASETS:
            for modality in ("text", "visual"):
                data = [float(row[f"{pair}_median"]) for row in distinct
                        if row["dataset"] == dataset and row["modality"] == modality and row["variant"] == "edge_mix"]
                if data:
                    vals.append(data)
                    short = {"Movies": "Mov", "Grocery": "Gro", "ele-fashion": "Ele"}[dataset]
                    labels.append(f"{short}{modality[0].upper()}")
        ax.boxplot(vals, tick_labels=labels, showfliers=False); ax.axhline(0, color="black", lw=.6)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=5.5, length=0, pad=8)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Median per-edge cosine similarity")
    fig.tight_layout(); export("e0_function_distinctness", fig)

    wanted = ("pi_shuffle_within_target", "pi_target_mean", "pi_global_mean", "modality_tied_pi",
              "smooth_only", "null_off", "relational_off", "cross_off")
    subset = [row for row in intervention_summary if row["dataset"] == "ALL" and row["intervention"] in wanted]
    subset.sort(key=lambda row: wanted.index(row["intervention"]))
    display = {"pi_shuffle_within_target": "shuffle (within target)", "pi_target_mean": "target mean",
               "pi_global_mean": "global mean", "modality_tied_pi": "modality tied",
               "smooth_only": "smooth only", "null_off": "null off",
               "relational_off": "relational off", "cross_off": "cross off"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.2), sharey=False)
    for ax, metric, title in zip(axes, ("delta_accuracy", "delta_macro_f1", "delta_ce"),
                                 ("Δ Accuracy (pp)", "Δ Macro-F1 (pp)", "Δ CE")):
        ax.barh([display[row["intervention"]] for row in subset], [row[f"{metric}_mean"] for row in subset],
                xerr=[row[f"{metric}_sd"] for row in subset], color="#4c78a8", alpha=.88, capsize=2)
        ax.axvline(0, color="black", lw=.7); ax.set_title(title); ax.grid(axis="x", alpha=.18)
        ax.tick_params(axis="y", labelsize=5.5)
    fig.tight_layout(); export("e0_interventions", fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize E0 structured relation-function executor campaign")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    args = parser.parse_args()
    records = load_runs(args.output_dir)
    rows, summaries, paired = performance_tables(records)
    (router, within, disagreement, distinct, scales, intervention_rows,
     intervention_summary, parameter_rows, init_rows) = diagnostic_tables(records)
    data_dir = args.research_dir / "data"
    write_csv(data_dir / "performance_by_run.csv", rows)
    write_csv(data_dir / "performance_summary.csv", summaries)
    write_csv(data_dir / "paired_delta_summary.csv", paired)
    write_csv(data_dir / "router_diagnostics.csv", router)
    write_csv(data_dir / "within_node_routing_variation.csv", within)
    write_csv(data_dir / "modality_routing_disagreement.csv", disagreement)
    write_csv(data_dir / "function_output_distinctness.csv", distinct)
    write_csv(data_dir / "expert_scale_diagnostics.csv", scales)
    write_csv(data_dir / "intervention_by_run.csv", intervention_rows)
    write_csv(data_dir / "intervention_summary.csv", intervention_summary)
    write_csv(data_dir / "parameter_summary.csv", parameter_rows)
    write_csv(data_dir / "common_initialization_audit.csv", init_rows)
    save_figures(args.research_dir, rows, router, within, disagreement, distinct, intervention_summary)
    print(f"Wrote E0 tables and figures to {args.research_dir}")
    print(f"Rows: performance={len(rows)}, router={len(router)}, within-node={len(within)}, "
          f"interventions={len(intervention_rows)}")


if __name__ == "__main__":
    main()
