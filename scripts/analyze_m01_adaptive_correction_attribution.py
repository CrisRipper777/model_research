from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "m01_adaptive_correction_attribution"
RESEARCH_OUT = PROJECT_ROOT / "research" / "m01_adaptive_correction_attribution"
M0_DATA = PROJECT_ROOT / "research" / "m0_adaptive_propagation_screen" / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
NEW_VARIANTS = ("extent_wide", "single_basis_static", "single_basis_target")
VARIANT_LABELS = {
    "semantic": "SEM", "uniform": "UNI", "extent": "A", "extent_wide": "A-wide",
    "single_basis_static": "B-static", "single_basis_target": "B-target", "single_basis": "B",
}
METRICS = {"val_accuracy": "accuracy", "val_macro_f1": "macro_f1", "val_ce": "ce"}
COMPARISONS = [
    ("extent_wide", "extent", "A-wide - A"),
    ("extent_wide", "uniform", "A-wide - UNI"),
    ("single_basis", "extent_wide", "B - A-wide"),
    ("single_basis_static", "extent_wide", "B-static - A-wide"),
    ("single_basis_target", "extent_wide", "B-target - A-wide"),
    ("single_basis", "single_basis_static", "B - B-static"),
    ("single_basis", "single_basis_target", "B - B-target"),
    ("single_basis_target", "single_basis_static", "B-target - B-static"),
    ("single_basis", "uniform", "B - UNI"),
]
INTERVENTIONS = ["function_off", "g_shuffle_only", "c_shuffle_only", "c_target_mean", "c_global_mean", "edge_control_shuffle"]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _all_run_records(out_dir: Path):
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in NEW_VARIANTS:
                p = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if not p.is_file():
                    raise FileNotFoundError(p)
                rows.append(_read_json(p))
    return rows


def build_performance(out_dir: Path, data_dir: Path):
    m0 = pd.read_csv(M0_DATA / "performance_by_run.csv")
    m0 = m0[m0.variant.isin(["semantic", "uniform", "extent", "single_basis"])].copy()
    m0 = m0.rename(columns={"name": "historical_name"})
    old_rows = []
    for row in m0.to_dict("records"):
        old_rows.append({"dataset": row["dataset"], "seed": int(row["seed"]), "variant": row["variant"],
                         "label": VARIANT_LABELS[row["variant"]], "source": "M0 committed",
                         **{k: float(row[k]) for k in METRICS}})
    records = _all_run_records(out_dir)
    new_rows = []
    for r in records:
        row = {"dataset": r["dataset"], "seed": int(r["seed"]), "variant": r["variant"],
               "label": VARIANT_LABELS[r["variant"]], "source": "M0.1 trained", **{k: float(r[k]) for k in METRICS},
               "best_epoch": int(r["best_epoch"]), "epochs_run": int(r["epochs_run"]),
               "trainable_params": int(r["trainable_params"]), "peak_gpu_memory_gb": r["peak_gpu_memory_bytes"] / 1024**3,
               "training_time_sec": float(r["training_time_sec"])}
        new_rows.append(row)
    old_df = pd.DataFrame(old_rows)
    new_df = pd.DataFrame(new_rows)
    full = pd.concat([old_df, new_df], ignore_index=True)
    full["label"] = pd.Categorical(full["label"], ["SEM", "UNI", "A", "A-wide", "B-static", "B-target", "B"])
    full = full.sort_values(["dataset", "seed", "label"]).reset_index(drop=True)
    new_df.to_csv(data_dir / "performance_new_variants.csv", index=False)
    full.to_csv(data_dir / "performance_full_comparison.csv", index=False)
    summary_rows = []
    for (dataset, variant, label), group in full.groupby(["dataset", "variant", "label"], observed=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            summary_rows.append({"scope": "dataset", "dataset": dataset, "variant": variant, "label": str(label),
                                 "metric": metric.removeprefix("val_"), "n": len(values),
                                 "mean": float(np.mean(values)), "population_sd": float(np.std(values, ddof=0))})
    for (variant, label), group in full.groupby(["variant", "label"], observed=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            summary_rows.append({"scope": "all_dataset_seed_pairs", "dataset": "ALL", "variant": variant,
                                 "label": str(label), "metric": metric.removeprefix("val_"), "n": len(values),
                                 "mean": float(np.mean(values)), "population_sd": float(np.std(values, ddof=0))})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(data_dir / "performance_summary.csv", index=False)
    return full, new_df, summary


def build_paired(full: pd.DataFrame, data_dir: Path):
    index = full.set_index(["dataset", "seed", "variant"])
    by_run = []
    for left, right, label in COMPARISONS:
        for dataset in DATASETS:
            for seed in SEEDS:
                a = index.loc[(dataset, seed, left)]
                b = index.loc[(dataset, seed, right)]
                row = {"comparison": label, "left_variant": left, "right_variant": right,
                       "dataset": dataset, "seed": seed}
                for col, short in METRICS.items():
                    row[f"delta_{short}"] = float(a[col] - b[col])
                by_run.append(row)
    run_df = pd.DataFrame(by_run)
    run_df.to_csv(data_dir / "paired_attribution_by_run.csv", index=False)
    summary_rows = []
    for comparison, group in run_df.groupby("comparison", sort=False):
        for scope, scoped in [("all_dataset_seed_pairs", group)] + [(d, group[group.dataset == d]) for d in DATASETS]:
            for metric in ("accuracy", "macro_f1", "ce"):
                values = scoped[f"delta_{metric}"].to_numpy(dtype=float)
                summary_rows.append({"comparison": comparison, "scope": "all" if scope == "all_dataset_seed_pairs" else "dataset",
                                     "dataset": "ALL" if scope == "all_dataset_seed_pairs" else scope,
                                     "metric": metric, "n": int(values.size), "mean_delta": float(values.mean()),
                                     "population_sd": float(values.std(ddof=0)), "median_delta": float(np.median(values)),
                                     "n_positive": int((values > 0).sum()), "n_negative": int((values < 0).sum())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(data_dir / "paired_attribution_summary.csv", index=False)
    return run_df, summary


def build_parameters(out_dir: Path, data_dir: Path):
    old = pd.read_csv(M0_DATA / "parameter_summary.csv")
    old = old[old.variant.isin(["extent", "single_basis"])].copy()
    run_records = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in _all_run_records(out_dir)}
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in ("extent", "extent_wide", "single_basis_static", "single_basis_target", "single_basis"):
                if variant in {"extent", "single_basis"}:
                    src = old[(old.dataset == dataset) & (old.seed == seed) & (old.variant == variant)].iloc[0]
                    gate = int(src.gate_and_correction_heads)
                    correction_ctrl = 0 if variant == "extent" else 130
                    basis = int(src.correction_basis)
                    params = {"semantic_backbone": int(src.semantic_backbone), "relation_encoder": int(src.relation_encoder),
                              "default_transform": int(src.default_transform), "gate_params": gate,
                              "correction_control_params": correction_ctrl, "correction_basis_params": basis,
                              "model_total": int(src.model_total), "classifier_params": int(src.classifier_params),
                              "trainable_params": int(src.trainable_params)}
                else:
                    rec = run_records[(dataset, seed, variant)]
                    c = rec["parameter_counts"]
                    params = {"semantic_backbone": c["semantic_backbone"], "relation_encoder": c["relation_encoder"],
                              "default_transform": c["default_transform"], "gate_params": c["gate_params"],
                              "correction_control_params": c["correction_control_params"],
                              "correction_basis_params": c["correction_basis_params"], "model_total": rec["model_params"],
                              "classifier_params": rec["classifier_params"], "trainable_params": rec["trainable_params"]}
                rows.append({"dataset": dataset, "seed": seed, "variant": variant, "label": VARIANT_LABELS[variant], **params})
    out = pd.DataFrame(rows)
    out.to_csv(data_dir / "parameter_attribution.csv", index=False)
    return out


def build_audits(out_dir: Path, data_dir: Path):
    rows = []
    records = _all_run_records(out_dir)
    seen = set()
    for r in records:
        audit = r["initialization_audit"]
        key = (r["dataset"], int(r["seed"]))
        if key in seen:
            continue
        seen.add(key)
        row = {"dataset": key[0], "seed": key[1], "common_all_equal": audit["common_all_equal"],
               "b_target_correction_heads_equal": audit["b_target_correction_heads_equal"],
               "b_static_target_basis_equal_to_b": audit["b_static_target_basis_equal_to_b"],
               "a_wide_b_model_param_delta": audit["a_wide_b_model_param_delta"],
               "a_wide_b_within_32": audit["a_wide_b_within_32"],
               "b_target_b_param_delta": audit["b_target_b_param_delta"], "b_target_b_params_equal": audit["b_target_b_params_equal"],
               "b_b_static_param_delta": audit["b_b_static_param_delta"], "b_static_difference_is_128": audit["b_static_difference_is_128"]}
        for name, digest in audit["common_hashes"].items():
            row[f"common_hash_{name}"] = digest
        for variant, counts in audit["parameter_counts"].items():
            row[f"params_{variant}"] = counts["model_total"]
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["dataset", "seed"])
    out.to_csv(data_dir / "common_initialization_audit.csv", index=False)
    return out


def build_interventions(out_dir: Path, data_dir: Path):
    records = []
    diag_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            r = _read_json(out_dir / "b_attributions" / dataset / f"seed_{seed}.json")
            for row in r["rows"]:
                if row["intervention"] not in INTERVENTIONS:
                    continue
                records.append({"dataset": dataset, "seed": seed, **row})
            for row in r["within_node_c_variation"]:
                diag_rows.append({"dataset": dataset, "seed": seed, **row})
    df = pd.DataFrame(records)
    df.to_csv(data_dir / "b_intervention_by_run.csv", index=False)
    # First collapse the five intervention repeats within each checkpoint, then summarize their per-run means.
    summary_rows = []
    for intervention in INTERVENTIONS:
        sub = df[df.intervention == intervention]
        per_run_rows = []
        for (dataset, seed), group in sub.groupby(["dataset", "seed"]):
            row = {"dataset": dataset, "seed": seed, "intervention": intervention}
            for field in ("delta_accuracy", "delta_macro_f1", "delta_ce"):
                vals = group[field].to_numpy(dtype=float)
                row[f"{field}_repeat_mean"] = float(vals.mean())
                row[f"{field}_repeat_population_sd"] = float(vals.std(ddof=0))
            per_run_rows.append(row)
            summary_rows.append({"scope": "checkpoint", **row, "n_repeats": len(group)})
        if per_run_rows:
            per = pd.DataFrame(per_run_rows)
            row = {"scope": "all_9_checkpoints", "dataset": "ALL", "seed": np.nan, "intervention": intervention,
                   "n_checkpoints": int(len(per))}
            for field in ("delta_accuracy_repeat_mean", "delta_macro_f1_repeat_mean", "delta_ce_repeat_mean"):
                values = per[field].to_numpy(dtype=float)
                base = field.replace("_repeat_mean", "")
                row[f"{base}_mean"] = float(values.mean())
                row[f"{base}_population_sd"] = float(values.std(ddof=0))
                row[f"{base}_median"] = float(np.median(values))
                repeat_sd = per[f"{base}_repeat_population_sd"].to_numpy(dtype=float)
                row[f"{base}_within_checkpoint_repeat_sd_mean"] = float(repeat_sd.mean())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(data_dir / "b_intervention_summary.csv", index=False)
    cvar = pd.DataFrame(diag_rows)
    cvar.to_csv(data_dir / "within_node_c_variation.csv", index=False)
    decomp_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            r = _read_json(out_dir / "b_attributions" / dataset / f"seed_{seed}.json")
            for row in r["c_variance_decomposition"]:
                decomp_rows.append({"dataset": dataset, "seed": seed, **row})
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(data_dir / "c_variance_decomposition.csv", index=False)
    return df, summary, cvar, decomp


def verify_legacy_interventions(out_dir: Path, data_dir: Path):
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            old_path = PROJECT_ROOT / "outputs" / "m0_adaptive_propagation_screen" / "runs" / dataset / f"seed_{seed}" / "single_basis.json"
            new_path = out_dir / "b_attributions" / dataset / f"seed_{seed}.json"
            old = _read_json(old_path)
            new = _read_json(new_path)
            comparisons = [("normal", None, old["interventions"]["normal"], new["normal"])]
            for row in old["interventions"]["rows"]:
                if row["intervention"] == "function_off":
                    comparisons.append(("function_off", None, row, next(r for r in new["rows"] if r["intervention"] == "function_off")))
                elif row["intervention"] == "edge_control_shuffle":
                    comparisons.append(("edge_control_shuffle", int(row["repeat_seed"]), row,
                                        next(r for r in new["rows"] if r["intervention"] == "edge_control_shuffle" and int(r["repeat_seed"]) == int(row["repeat_seed"]))))
            for name, repeat, expected, observed in comparisons:
                for key in METRICS:
                    diff = abs(float(expected[key]) - float(observed[key]))
                    rows.append({"dataset": dataset, "seed": seed, "intervention": name, "repeat_seed": repeat,
                                 "metric": key.removeprefix("val_"), "m0_value": float(expected[key]),
                                 "m01_recomputed_value": float(observed[key]), "absolute_difference": diff,
                                 "match_within_1e_6": diff <= 1e-6})
    result = pd.DataFrame(rows)
    result.to_csv(data_dir / "m0_intervention_reproduction.csv", index=False)
    if not result.match_within_1e_6.all():
        raise AssertionError(f"M0 B legacy intervention reproduction mismatch: max abs diff={result.absolute_difference.max()}")
    return result


def build_granularity(performance: pd.DataFrame, out_dir: Path, data_dir: Path):
    selected = performance[performance.variant.isin(NEW_VARIANTS)].copy()
    diagnostic_rows = []
    intervention_names = ("extent_off", "function_off", "g_shuffle_only", "c_shuffle_only",
                          "c_target_mean", "c_global_mean", "modality_tied")
    for variant in NEW_VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                record = _read_json(out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json")
                cd = record["control_diagnostics"]
                intervention_groups = {}
                for item in record["interventions"]["rows"]:
                    intervention_groups.setdefault(item["intervention"], []).append(item)
                for modality in range(2):
                    g = cd["g"][modality]
                    ratio = cd["correction_ratio"][modality]
                    within_g = cd["within_node_g_std"][modality]
                    row = {"dataset": dataset, "seed": seed, "variant": variant,
                           "modality": "text" if modality == 0 else "visual",
                           **{f"g_{key}": g.get(key, float("nan")) for key in ("mean", "std", "q10", "q25", "median", "q75", "q90")},
                           "g_fraction_lt_0_1": cd.get("g_fraction_lt_0_1", [float("nan")]*2)[modality],
                           "g_fraction_gt_0_9": cd.get("g_fraction_gt_0_9", [float("nan")]*2)[modality],
                           "mean_abs_g_tv": cd.get("mean_abs_g_tv", float("nan")),
                           "within_g_target_count": within_g.get("eligible_targets", 0),
                           "within_g_std_mean": within_g.get("mean", float("nan")),
                           "within_g_std_q10": within_g.get("q10", float("nan")),
                           "within_g_std_q25": within_g.get("q25", float("nan")),
                           "within_g_std_median": within_g.get("median", float("nan")),
                           "within_g_std_q75": within_g.get("q75", float("nan")),
                           "within_g_std_q90": within_g.get("q90", float("nan")),
                           "correction_default_ratio_mean": ratio.get("mean", float("nan")),
                           "correction_default_ratio_median": ratio.get("median", float("nan"))}
                    c = cd["c"][modality]
                    for key in ("mean", "std", "q10", "q25", "median", "q75", "q90"):
                        row[f"edge_expanded_c_{key}"] = c.get(key, float("nan"))
                    within_c = cd["within_node_c_std"][modality]
                    row.update({"within_c_target_count": within_c.get("eligible_targets", 0),
                                "within_c_std_mean": within_c.get("mean", float("nan")),
                                "within_c_std_median": within_c.get("median", float("nan")),
                                "within_c_std_q25": within_c.get("q25", float("nan")),
                                "within_c_std_q75": within_c.get("q75", float("nan"))})
                    if variant == "single_basis_static":
                        row["learned_c_static"] = cd["learned_c_static"][modality]
                    if variant == "single_basis_target":
                        target_c = cd["target_level_c"][modality]
                        for key in ("mean", "std", "q10", "q25", "median", "q75", "q90"):
                            row[f"target_level_c_{key}"] = target_c[key]
                        row["target_level_c_target_count"] = cd["target_level_c_target_count"]
                        row["b_target_within_target_max_abs_deviation"] = cd["b_target_within_target_max_abs_deviation"]
                    for intervention in intervention_names:
                        values = intervention_groups.get(intervention, [])
                        for metric in ("accuracy", "macro_f1", "ce"):
                            vals = [float(item[f"delta_{metric}"]) for item in values]
                            row[f"{intervention}_delta_{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
                            row[f"{intervention}_delta_{metric}_repeat_sd"] = float(np.std(vals, ddof=0)) if vals else float("nan")
                    diagnostic_rows.append(row)
    diagnostics = pd.DataFrame(diagnostic_rows)
    performance_cols = ["dataset", "seed", "variant", "label", "val_accuracy", "val_macro_f1", "val_ce"]
    result = selected[performance_cols].merge(diagnostics, on=["dataset", "seed", "variant"], how="left")
    result.to_csv(data_dir / "control_granularity_summary.csv", index=False)
    return result


def make_figures(full, paired, intervention_summary, fig_dir: Path):
    colors = {"SEM": "#87919b", "UNI": "#507f9b", "A": "#6d947c", "A-wide": "#9baa6d",
              "B-static": "#ca9860", "B-target": "#be7868", "B": "#725b89"}
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2, "legend.fontsize": 6.5, "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })

    def export(fig, filename: str, axes, panel_ids=None, layout_rect=(0, 0, 1, 1)):
        fig.tight_layout(pad=1.0, rect=layout_rect)
        stem = fig_dir / filename
        qa_dir = fig_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=panel_ids,
            json_out=str(qa_dir / f"{filename}.alignment.json"),
            overlay_svg=str(qa_dir / f"{filename}.alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=panel_ids is not None, strict=True,
        )
        # Export editable vector PDF/SVG and 600-dpi PNG/TIFF raster files.
        for suffix in (".png", ".pdf", ".svg", ".tif"):
            path = stem.with_suffix(suffix)
            if suffix == ".tif":
                fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=.03,
                            pil_kwargs={"compression": "tiff_lzw"})
            else:
                fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=.03)
        svg_path = stem.with_suffix(".svg")
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)

    # Figure-level claim: the edge-conditioned B mechanism has no consistent
    # all-metric advantage over a capacity-matched extent controller.
    labels = ["SEM", "UNI", "A", "A-wide", "B-static", "B-target", "B"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, metric, title, ylabel in zip(
        axes, ("val_accuracy", "val_macro_f1", "val_ce"),
        ("Accuracy", "Macro-F1", "Cross-entropy"),
        ("mean validation score", "mean validation score", "mean validation loss"),
    ):
        means = [float(full.loc[full.label == label, metric].mean()) for label in labels]
        sds = [float(full.loc[full.label == label, metric].std(ddof=0)) for label in labels]
        ax.bar(range(len(labels)), means, yerr=sds, color=[colors[v] for v in labels], capsize=2.2, linewidth=0)
        ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right", rotation_mode="anchor")
        ax.set_title(title, pad=6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18, linewidth=.45)
        ax.set_axisbelow(True)
    for ax, letter in zip(axes, "abc"):
        ax.text(-.10, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    export(fig, "m01_performance_attribution", axes, ["a", "b", "c"])

    # Capacity control: paired seed trajectories over the three datasets.
    cap = full[full.label.isin(["A", "A-wide", "B"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        block = cap[cap.dataset == dataset]
        for label in ("A", "A-wide", "B"):
            vals = block[block.label == label].sort_values("seed").val_accuracy.to_numpy()
            ax.plot([42, 43, 44], vals, marker="o", markersize=3.2, linewidth=1.05, label=label, color=colors[label])
        ax.set_title(dataset, pad=5)
        ax.set_xlabel("seed")
        ax.grid(alpha=.18, linewidth=.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("validation accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(.5, .015), handlelength=1.3, columnspacing=1.2)
    for ax, letter in zip(axes, "abc"):
        ax.text(-.10, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    export(fig, "m01_capacity_control", axes, ["a", "b", "c"], layout_rect=(0, .12, 1, 1))

    # Granularity control: trained global, target, and edge correction.
    gran = full[full.label.isin(["B-static", "B-target", "B"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        block = gran[gran.dataset == dataset]
        for label in ("B-static", "B-target", "B"):
            vals = block[block.label == label].sort_values("seed").val_accuracy.to_numpy()
            ax.plot([42, 43, 44], vals, marker="o", markersize=3.2, linewidth=1.05, label=label, color=colors[label])
        ax.set_title(dataset, pad=5)
        ax.set_xlabel("seed")
        ax.grid(alpha=.18, linewidth=.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("validation accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(.5, .015), handlelength=1.3, columnspacing=1.2)
    for ax, letter in zip(axes, "abc"):
        ax.text(-.10, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    export(fig, "m01_c_granularity", axes, ["a", "b", "c"], layout_rect=(0, .12, 1, 1))

    # Intervention attribution: effect on an already trained M0-B checkpoint.
    sh = intervention_summary[intervention_summary.scope == "all_9_checkpoints"].set_index("intervention")
    order = [name for name in INTERVENTIONS if name in sh.index]
    vals = [float(sh.loc[name, "delta_accuracy_mean"]) * 100 for name in order]
    errs = [float(sh.loc[name, "delta_accuracy_population_sd"]) * 100 for name in order]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(range(len(order)), vals, yerr=errs, color="#64829a", capsize=3, linewidth=0)
    ax.axhline(0, color="#333333", lw=.7)
    ax.set_xticks(range(len(order)), [s.replace("_", " ") for s in order], rotation=24, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Δ validation accuracy (percentage points)")
    ax.grid(axis="y", alpha=.18, linewidth=.45)
    ax.set_axisbelow(True)
    ax.text(-.08, 1.04, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    export(fig, "m01_b_interventions", [ax], None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--research-dir", type=Path, default=RESEARCH_OUT)
    args = parser.parse_args()
    data_dir, fig_dir = args.research_dir / "data", args.research_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    full, new, perf_summary = build_performance(args.output_dir, data_dir)
    paired_run, paired_summary = build_paired(full, data_dir)
    parameters = build_parameters(args.output_dir, data_dir)
    audits = build_audits(args.output_dir, data_dir)
    intervention_run, intervention_summary, within_c, decomp = build_interventions(args.output_dir, data_dir)
    legacy_check = verify_legacy_interventions(args.output_dir, data_dir)
    granularity = build_granularity(full, args.output_dir, data_dir)
    make_figures(full, paired_run, intervention_summary, fig_dir)
    summary = {
        "new_runs": len(new), "old_comparison_runs": len(full) - len(new),
        "b_attribution_checkpoints": int(intervention_run[["dataset", "seed"]].drop_duplicates().shape[0]),
        "legacy_b_intervention_rows_verified": int(len(legacy_check)),
        "legacy_b_intervention_max_abs_diff": float(legacy_check.absolute_difference.max()),
        "all_parameter_audits_pass": bool(audits[["common_all_equal", "b_target_correction_heads_equal", "b_static_target_basis_equal_to_b", "a_wide_b_within_32", "b_target_b_params_equal", "b_static_difference_is_128"]].all().all()),
        "parameter_summary": parameters.groupby("label")["trainable_params"].agg(["mean", "std"]).to_dict(orient="index"),
        "output_dir": str(args.research_dir.relative_to(PROJECT_ROOT)),
    }
    (args.research_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
