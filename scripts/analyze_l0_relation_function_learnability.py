from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path("/home/m3/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

RESEARCH = PROJECT_ROOT / "research/l0_relation_function_learnability_audit"
DATA = RESEARCH / "data"
FIGURES = RESEARCH / "figures"
PDF_QA_PYTHON = Path("/home/m3/miniconda3/bin/python")
COLORS = {"text": "#3B6FB6", "visual": "#D08A3F", "green": "#4D9872",
          "purple": "#8065A6", "gray": "#7A828A", "red": "#B45A5A"}
MODEL_ORDER = ["SIM_ONLY", "TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL",
               "ENDPOINT_LOCAL_SHUFFLED_TARGET"]
MODEL_LABEL = {"SIM_ONLY": "SIM", "TARGET_ONLY": "TGT", "ENDPOINT": "END",
               "ENDPOINT_LOCAL": "LOC", "ENDPOINT_LOCAL_SHUFFLED_TARGET": "SHUF"}
DATASETS = ["Movies", "Grocery", "ele-fashion"]
MODALITIES = ["text", "visual"]
TARGETS = ["smooth_utility", "delta_absdiff", "delta_product"]
METRICS = ["spearman", "pearson", "r2", "sign_auroc", "sign_balanced_accuracy",
           "sign_prevalence", "within_target_residual_spearman", "per_target_eligible_count",
           "per_target_valid_count", "per_target_rank_mean", "per_target_rank_median",
           "per_target_rank_q25", "per_target_rank_q75"]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def aggregate_metrics(frame: pd.DataFrame, key_columns: list[str],
                      method_column: str = "model") -> pd.DataFrame:
    metric_columns = [column for column in METRICS if column in frame.columns]
    seed_keys = ["dataset", "seed", *[x for x in key_columns if x not in ("dataset", "seed")]]
    by_seed = frame.groupby(seed_keys, dropna=False, as_index=False)[metric_columns].mean(numeric_only=True)
    grouping = [x for x in key_columns if x != "seed"]
    rows: list[dict[str, Any]] = []
    for key, group in by_seed.groupby(grouping, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(grouping, key))
        row["seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_seed_sd"] = float(values.std(ddof=0)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def seed_means(frame: pd.DataFrame, keys: list[str], value_columns: list[str]) -> pd.DataFrame:
    group_seed = ["dataset", "seed", *[k for k in keys if k not in ("dataset", "seed")]]
    return frame.groupby(group_seed, as_index=False, dropna=False)[value_columns].mean(numeric_only=True)


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render a Markdown table without relying on the optional tabulate package."""
    def cell(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt) if np.isfinite(value) else "NA"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def save_summaries() -> dict[str, pd.DataFrame]:
    evidence = pd.read_csv(DATA / "evidence_probe_by_fold.csv")
    direct = pd.read_csv(DATA / "direct_state_probe_by_fold.csv")
    frozen = pd.read_csv(DATA / "frozen_state_readout_by_fold.csv")
    evidence_summary = aggregate_metrics(evidence, ["dataset", "model", "modality", "target_type"])
    direct_summary = aggregate_metrics(direct, ["dataset", "model", "modality", "target_type"])
    frozen_summary = aggregate_metrics(frozen, ["dataset", "model", "modality", "target_type", "representation"])
    atomic_csv(evidence_summary, DATA / "evidence_probe_summary.csv")
    atomic_csv(direct_summary, DATA / "direct_state_probe_summary.csv")
    atomic_csv(frozen_summary, DATA / "frozen_state_readout_summary.csv")

    within = []
    columns = ["dataset", "seed", "fold", "model", "modality", "target_type",
               "within_target_residual_spearman", "per_target_eligible_count",
               "per_target_valid_count", "per_target_rank_mean", "per_target_rank_median",
               "per_target_rank_q25", "per_target_rank_q75"]
    for family, frame in (("EvidenceMLP", evidence), ("M0StateDirect", direct), ("FrozenStateReadout", frozen)):
        current = frame.copy()
        if "representation" in current:
            current["model"] = current["model"].astype(str) + ":" + current["representation"].astype(str)
        current["family"] = family
        within.extend(current[[*columns, "family"]].to_dict("records"))
    atomic_csv(pd.DataFrame(within), DATA / "within_target_predictability.csv")

    high_raw_path = DATA / "high_margin_by_fold.csv"
    if not high_raw_path.exists():
        high_raw_path = DATA / "high_margin_summary.csv"
        high_raw = pd.read_csv(high_raw_path)
        atomic_csv(high_raw, DATA / "high_margin_by_fold.csv")
    else:
        high_raw = pd.read_csv(high_raw_path)
    high_metrics = ["high_margin_threshold_abs", "high_margin_n", "high_margin_sign_auroc",
                    "high_margin_sign_balanced_accuracy", "high_margin_spearman"]
    evidence_high = high_raw[high_raw.model.isin(MODEL_ORDER)].copy()
    direct_high = high_raw[high_raw.model == "M0StateDirect"].copy()
    frozen_high = high_raw[high_raw.model.astype(str).str.startswith("Frozen_")].copy()
    high = pd.concat([evidence_high, direct_high, frozen_high], ignore_index=True)
    high_seed = high.groupby(["dataset", "seed", "fold", "model", "modality", "target_type"],
                             as_index=False, dropna=False)[high_metrics].mean(numeric_only=True)
    high_groups = ["dataset", "model", "modality", "target_type"]
    high_rows = []
    for key, group in high_seed.groupby(high_groups, dropna=False, sort=True):
        if not isinstance(key, tuple): key = (key,)
        row = dict(zip(high_groups, key))
        row["seed_count"] = int(group.seed.nunique())
        for metric in high_metrics:
            arr = group[metric].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f"{metric}_mean"] = float(arr.mean()) if len(arr) else np.nan
            row[f"{metric}_seed_sd"] = float(arr.std(ddof=0)) if len(arr) else np.nan
        high_rows.append(row)
    high_summary = pd.DataFrame(high_rows)
    atomic_csv(high_summary, DATA / "high_margin_summary.csv")

    parameters = pd.read_csv(DATA / "parameter_summary.csv")
    parameter_keys = [c for c in ["dataset", "seed", "fold", "model", "variant", "modality", "representation"]
                      if c in parameters and (c in {"dataset", "seed", "fold", "model"} or parameters[c].notna().any())]
    parameters = parameters.drop_duplicates(parameter_keys)
    atomic_csv(parameters, DATA / "parameter_summary.csv")
    return {"evidence": evidence, "direct": direct, "frozen": frozen,
            "evidence_summary": evidence_summary, "direct_summary": direct_summary,
            "frozen_summary": frozen_summary, "within": pd.DataFrame(within),
            "high": high_summary, "high_raw": high_raw, "parameters": parameters}


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.8,
        "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
    })


def add_labels(axes, labels: list[str]) -> None:
    for ax, label in zip(np.asarray(axes).reshape(-1), labels):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left", clip_on=False)


def export_figure(fig, stem: str, labels: list[str]) -> dict[str, Any]:
    path = FIGURES / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    add_labels(fig.axes, labels)
    fig.tight_layout(pad=1.2)
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=f"{path}.alignment.json", overlay_svg=f"{path}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(f"{path}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.svg", bbox_inches="tight")
    fig.savefig(f"{path}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pdf = Path(f"{path}.pdf")
    text_audit = subprocess.run([str(PDF_QA_PYTHON), str(SKILL_SCRIPTS / "audit_pdf_text.py"),
                                 str(pdf), "--min-pt", "5", "--json"],
                                capture_output=True, text=True)
    collision = subprocess.run([str(PDF_QA_PYTHON), str(SKILL_SCRIPTS / "audit_figure_collisions.py"),
                                str(pdf), "--json-out", f"{path}.collision.json",
                                "--overlay-pdf", f"{path}.collision-overlay.pdf"],
                               capture_output=True, text=True)
    if text_audit.returncode != 0 or collision.returncode not in (0, 1):
        raise RuntimeError(f"figure QA failed for {stem}: text={text_audit.stdout}{text_audit.stderr}; collision={collision.stdout}{collision.stderr}")
    if collision.returncode == 1:
        raise RuntimeError(f"collision audit found blocking findings for {stem}: {collision.stdout}{collision.stderr}")
    text_json = None
    try:
        text_json = json.loads(text_audit.stdout)
    except json.JSONDecodeError:
        pass
    coll_json = json.loads(Path(f"{path}.collision.json").read_text())
    if alignment is None:
        alignment = json.loads(Path(f"{path}.alignment.json").read_text())
    return {"stem": stem, "alignment": alignment, "text_audit": text_json,
            "collision_audit": coll_json, "return_codes": [text_audit.returncode, collision.returncode]}


def _mean_sd_line(ax, frame: pd.DataFrame, model_col: str, metric: str,
                  model_order: list[str], label: str, color: str, linestyle: str = "-",
                  marker: str = "o", alpha: float = 1.0) -> None:
    x, mean, sd = [], [], []
    for i, model in enumerate(model_order):
        sub = frame[frame[model_col] == model]
        if sub.empty:
            continue
        row = sub.iloc[0]
        value = row.get(f"{metric}_mean", row.get(metric, np.nan))
        spread = row.get(f"{metric}_seed_sd", 0.0)
        if not np.isfinite(value):
            continue
        x.append(i); mean.append(value); sd.append(spread if np.isfinite(spread) else 0.0)
    if x:
        ax.errorbar(x, mean, yerr=sd, label=label, color=color, linestyle=linestyle,
                    marker=marker, markersize=3.2, linewidth=1.1, capsize=1.5, alpha=alpha)


def figure_target_reliability() -> dict[str, Any]:
    frame = pd.read_csv(DATA / "target_reliability_by_seed_pair.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharey=True)
    x0 = np.arange(3)
    offsets = {"text": -0.08, "visual": 0.08}
    target_order = ["U_S", "Delta_D", "Delta_P"]
    for ax, dataset in zip(axes, DATASETS):
        for modality, color in (("text", COLORS["text"]), ("visual", COLORS["visual"])):
            sub = frame[(frame.dataset == dataset) & (frame.modality == modality)]
            means, sds = [], []
            for target in target_order:
                values = sub[sub.target == target].spearman.to_numpy(dtype=float)
                means.append(np.nanmean(values)); sds.append(np.nanstd(values, ddof=0))
            ax.errorbar(x0 + offsets[modality], means, yerr=sds, label=modality.title(),
                        color=color, marker="o", linewidth=1.2, capsize=2, markersize=3.3)
        ax.axhline(0, color="#777777", linewidth=.6, linestyle="--")
        ax.set_title(dataset)
        ax.set_xticks(x0, ["U_S", "ΔD", "ΔP"])
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", color="#E5E7EA", linewidth=.5)
    axes[0].set_ylabel("Cross-seed Spearman ρ")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
    fig.suptitle("P1.3 utility target reliability", y=1.04, fontsize=8)
    return export_figure(fig, "l0_target_reliability", ["a", "b", "c"])


def figure_evidence(summary: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.0), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = summary[summary.dataset == dataset]
        for modality, color in (("text", COLORS["text"]), ("visual", COLORS["visual"])):
            for target, linestyle, label_t in (("delta_absdiff", "-", "ΔD"),
                                                ("delta_product", "--", "ΔP")):
                sf = sub[(sub.modality == modality) & (sub.target_type == target)].set_index("model").reindex(MODEL_ORDER).reset_index()
                _mean_sd_line(ax, sf, "model", "spearman", MODEL_ORDER,
                              f"{modality[0].upper()} {label_t}", color, linestyle,
                              marker="o" if target == "delta_absdiff" else "s")
        ax.set_title(dataset)
        ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABEL[x] for x in MODEL_ORDER], rotation=30, ha="right")
        ax.axhline(0, color="#777777", linewidth=.6, linestyle=":")
        ax.grid(axis="y", color="#E5E7EA", linewidth=.5)
    axes[0].set_ylabel("Outer-fold Spearman ρ")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.4, ncol=1)
    fig.suptitle("Function-delta predictability from observable evidence", y=1.04, fontsize=8)
    return export_figure(fig, "l0_evidence_predictability", ["a", "b", "c"])


def figure_within(summary: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.8), sharex="col", sharey="row")
    for col, dataset in enumerate(DATASETS):
        sub = summary[(summary.dataset == dataset) & summary.target_type.isin(["delta_absdiff", "delta_product"])]
        for row, metric in enumerate(("within_target_residual_spearman", "per_target_rank_mean")):
            ax = axes[row, col]
            for modality, color in (("text", COLORS["text"]), ("visual", COLORS["visual"])):
                for target, linestyle, mark in (("delta_absdiff", "-", "o"), ("delta_product", "--", "s")):
                    sf = sub[(sub.modality == modality) & (sub.target_type == target)].set_index("model").reindex(MODEL_ORDER).reset_index()
                    _mean_sd_line(ax, sf, "model", metric, MODEL_ORDER,
                                  f"{modality[0].upper()} {'ΔD' if target == 'delta_absdiff' else 'ΔP'}",
                                  color, linestyle, mark)
            ax.set_title(dataset if row == 0 else "")
            ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABEL[x] for x in MODEL_ORDER], rotation=30, ha="right")
            ax.axhline(0, color="#777777", linewidth=.6, linestyle=":")
            ax.grid(axis="y", color="#E5E7EA", linewidth=.5)
    axes[0, 0].set_ylabel("Within-target residual ρ")
    axes[1, 0].set_ylabel("Mean per-target rank ρ\n(degree ≥ 5)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.1, ncol=1)
    fig.suptitle("Edge-specific predictability within recipient targets", y=1.015, fontsize=8)
    return export_figure(fig, "l0_within_target_predictability", list("abcdef"))


def figure_state(summary_e: pd.DataFrame, summary_d: pd.DataFrame,
                 summary_f: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.7), sharex="col", sharey="row")
    state_order = ["ENDPOINT_LOCAL", "M0StateDirect", "Frozen_Q_PAIR", "Frozen_R_SHARED", "Frozen_U_MODAL"]
    state_labels = ["LOCAL", "Direct", "Q", "R", "U"]
    state_colors = [COLORS["purple"], COLORS["gray"], COLORS["text"], COLORS["green"], COLORS["visual"]]
    for col, dataset in enumerate(DATASETS):
        for row, metric in enumerate(("spearman", "within_target_residual_spearman")):
            ax = axes[row, col]
            for modality in MODALITIES:
                for target, ls in (("delta_absdiff", "-"), ("delta_product", "--")):
                    means, sds = [], []
                    for name in state_order:
                        source = summary_e if name == "ENDPOINT_LOCAL" else summary_d if name == "M0StateDirect" else summary_f
                        if name == "ENDPOINT_LOCAL":
                            model, rep = name, None
                        elif name == "M0StateDirect":
                            model, rep = name, None
                        else:
                            model, rep = name, name.removeprefix("Frozen_")
                        sel = source[(source.dataset == dataset) & (source.modality == modality) &
                                     (source.target_type == target) & (source.model == model)]
                        if rep is not None and "representation" in sel:
                            sel = sel[sel.representation == rep]
                        if sel.empty:
                            means.append(np.nan); sds.append(0.0)
                        else:
                            means.append(float(sel.iloc[0][f"{metric}_mean"]))
                            sds.append(float(sel.iloc[0][f"{metric}_seed_sd"]))
                    x = np.arange(len(state_order))
                    ax.errorbar(x, means, yerr=sds, color=COLORS[modality], linestyle=ls,
                                marker="o" if target == "delta_absdiff" else "s",
                                markersize=2.8, linewidth=.9, capsize=1.3,
                                label=f"{modality[0].upper()} {'ΔD' if target == 'delta_absdiff' else 'ΔP'}")
            ax.set_title(dataset if row == 0 else "")
            ax.set_xticks(range(len(state_order)), state_labels)
            ax.axhline(0, color="#777777", linewidth=.6, linestyle=":")
            ax.grid(axis="y", color="#E5E7EA", linewidth=.5)
    axes[0, 0].set_ylabel("Total Spearman ρ")
    axes[1, 0].set_ylabel("Within-target residual ρ")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.1, ncol=1)
    fig.suptitle("Direct capacity versus task-trained frozen state decodability", y=1.015, fontsize=8)
    return export_figure(fig, "l0_state_capacity", list("abcdef"))


def figure_high_margin(high: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.0), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = high[(high.dataset == dataset) & high.model.isin(MODEL_ORDER)]
        for modality, color in (("text", COLORS["text"]), ("visual", COLORS["visual"])):
            for target, linestyle, marker in (("delta_absdiff", "-", "o"), ("delta_product", "--", "s")):
                sf = sub[(sub.modality == modality) & (sub.target_type == target)].set_index("model").reindex(MODEL_ORDER).reset_index()
                _mean_sd_line(ax, sf, "model", "high_margin_sign_auroc", MODEL_ORDER,
                              f"{modality[0].upper()} {'ΔD' if target == 'delta_absdiff' else 'ΔP'}",
                              color, linestyle, marker)
        ax.set_title(dataset)
        ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABEL[x] for x in MODEL_ORDER], rotation=30, ha="right")
        ax.axhline(.5, color="#777777", linewidth=.6, linestyle=":")
        ax.grid(axis="y", color="#E5E7EA", linewidth=.5)
    axes[0].set_ylabel("High-|Δ| sign AUROC")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.4, ncol=1)
    fig.suptitle("Predictability among the test-fold top quartile of |function delta|", y=1.04, fontsize=8)
    return export_figure(fig, "l0_high_margin", ["a", "b", "c"])


def make_figures(summaries: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    qa = [figure_target_reliability(),
          figure_evidence(summaries["evidence_summary"]),
          figure_within(summaries["evidence_summary"]),
          figure_state(summaries["evidence_summary"], summaries["direct_summary"], summaries["frozen_summary"]),
          figure_high_margin(summaries["high"])]
    (FIGURES / "figure_qa_summary.json").write_text(json.dumps(qa, indent=2, default=str), encoding="utf-8")
    return qa


def make_report(summaries: dict[str, pd.DataFrame], diagnostic_label: str | None = None,
                findings: str = "") -> None:
    # The evidence tables in the CSVs are authoritative; narrative interpretation is appended after review.
    rel = pd.read_csv(DATA / "target_reliability_summary.csv")
    evidence = summaries["evidence_summary"]
    direct = summaries["direct_summary"]
    frozen = summaries["frozen_summary"]
    high = summaries["high"]
    def mean_sd(mean_values: pd.Series, sd_values: pd.Series) -> list[str]:
        result = []
        for mean, sd in zip(mean_values, sd_values):
            if pd.isna(mean):
                result.append("NA")
            elif pd.isna(sd):
                result.append(f"{float(mean):.3f} ± NA")
            else:
                result.append(f"{float(mean):.3f} ± {float(sd):.3f}")
        return result

    reliability_display = pd.DataFrame({
        "Dataset": rel.dataset,
        "Modality": rel.modality,
        "Target": rel.target,
        "Common edges (pair mean)": rel.overlap_edges_mean,
        "Overlap / smaller table": rel.overlap_fraction_smaller_mean,
        "Cross-seed Spearman (mean ± pair SD)": mean_sd(rel.spearman_mean, rel.spearman_seed_pair_sd),
        "Centered Delta Spearman": rel.centered_spearman_mean,
        "Sign agreement": rel.sign_agreement_mean,
    })
    reliability_md = markdown_table(reliability_display)
    focus = evidence[evidence.target_type.isin(["delta_absdiff", "delta_product"])]
    evidence_display = pd.DataFrame({
        "Dataset": focus.dataset,
        "Modality": focus.modality,
        "Target": focus.target_type,
        "Probe": focus.model,
        "Spearman mean ± seed SD": mean_sd(focus.spearman_mean, focus.spearman_seed_sd),
        "Sign AUROC": focus.sign_auroc_mean,
        "Within-target residual ρ": focus.within_target_residual_spearman_mean,
        "Mean per-target rank ρ ± seed SD": mean_sd(
            focus.per_target_rank_mean_mean, focus.per_target_rank_mean_seed_sd),
    })
    evidence_md = markdown_table(evidence_display)
    direct_display = pd.DataFrame({
        "Dataset": direct.dataset,
        "Modality": direct.modality,
        "Target": direct.target_type,
        "Spearman mean ± seed SD": mean_sd(direct.spearman_mean, direct.spearman_seed_sd),
        "Sign AUROC": direct.sign_auroc_mean,
        "Within-target residual ρ": direct.within_target_residual_spearman_mean,
        "Per-target rank ρ": direct.per_target_rank_mean_mean,
    })
    direct_md = markdown_table(direct_display)
    frozen_display = pd.DataFrame({
        "Dataset": frozen.dataset,
        "Modality": frozen.modality,
        "Target": frozen.target_type,
        "Representation": frozen.representation,
        "Spearman mean ± seed SD": mean_sd(frozen.spearman_mean, frozen.spearman_seed_sd),
        "Sign AUROC": frozen.sign_auroc_mean,
        "Within-target residual ρ": frozen.within_target_residual_spearman_mean,
        "Per-target rank ρ": frozen.per_target_rank_mean_mean,
    })
    frozen_md = markdown_table(frozen_display)
    high_focus = high[high.model.isin(["ENDPOINT_LOCAL", "ENDPOINT_LOCAL_SHUFFLED_TARGET",
                                      "M0StateDirect", "Frozen_U_MODAL"])]
    high_display = pd.DataFrame({
        "Dataset": high_focus.dataset,
        "Modality": high_focus.modality,
        "Target": high_focus.target_type,
        "Probe": high_focus.model,
        "High-margin n": high_focus.high_margin_n_mean,
        "Sign AUROC": high_focus.high_margin_sign_auroc_mean,
        "Balanced accuracy": high_focus.high_margin_sign_balanced_accuracy_mean,
        "Spearman": high_focus.high_margin_spearman_mean,
    })
    high_md = markdown_table(high_display)
    label = diagnostic_label or "PENDING_REVIEW"
    text = f"""# L0 — Relation-to-Function Learnability Audit

## Question and scope

This audit asks whether observable semantic and local relation evidence predicts the P1.3 validation-derived utility vector `[U_S^T, Δ_D^T, Δ_P^T, U_S^V, Δ_D^V, Δ_P^V]` on target-disjoint groups. `Δ_D = U_AbsDiff − U_Smooth` and `Δ_P = U_Product − U_Smooth`. Positive values indicate larger marginal utility under the P1.3 shared joint readout for that tested message; they are not operator labels or ground-truth edge roles.

The P1.3 utility uses validation labels: validation Accuracy selected the joint head and validation CE defines each removal utility. Although the outer CV keeps all edges for a destination in one fold, P1.3 selected its joint head using the full validation set. L0 is therefore an in-universe learnability diagnostic, not an independent generalization estimate. No original NC test labels or test metrics were accessed. The inputs reuse the same transductive graph and frozen P1.3 H0 representations as the source utility audit.

P1.3 Smooth/AbsDiff/Product targets are not mapped to E0 Smooth/Relational/Cross-Modal functions. The direct-utility probe receives stronger supervision than NC task training and does not imply an end-to-end router can learn the same signal.

## Protocol

- Reused all nine full P1.3 edge tables and all nine joint-head checkpoints; no P1.3 regeneration was needed. Utility rows are checked in exact `(src,dst,dst_degree)` order against each frozen P1.3 H0 graph support.
- Cross-seed reliability aligns by `(src,dst)` and reports overlap, Pearson/Spearman, positive-vs-nonpositive sign agreement, exact-zero fraction and target-centered Delta Spearman.
- Three deterministic, edge-count-balanced outer folds split by `dst`; all probes share each assignment. Inner early-stopping groups are also `dst`-disjoint (90/10 by unique outer-training target count).
- Target moments and full 1542D feature moments use outer-training edges only. Feature masking is applied after standardization. The six-target loss is target-balanced with `1/d_dst`, normalized to mean one, and uses standardized Huber loss, AdamW (`1e-3`, `1e-4`), gradient clipping 1, 150 epoch maximum, min epoch 20 and patience 15.
- All five EvidenceMLP inputs use `Linear(1542,128) → GELU → Linear(128,6)` and identical parameter counts/initialization. The shuffled control moves six-output tuples only among edges of the same destination in actual training groups.
- DirectState uses frozen P1.3 H0 and the M0 128→32 relation projection, exact M0 pair evidence, q/r/u encoders and a shared utility head. Frozen E0.1 Q/R/U readouts do not update KeepEdge checkpoints.
- Metrics are first averaged over the three outer folds within dataset×seed, then summarized as three-seed mean ± population SD. Edges are never pooled across seeds.

## 1–2. Target reliability by dataset and modality

{reliability_md}

Reliability is target-specific and should bound all probe interpretations. `U_S` is utility, while the two deltas are the primary function-sensitive targets. Exact seed-pair values, overlap counts and zero fractions are in `target_reliability_by_seed_pair.csv`.

## 3–10. EvidenceMLP predictability and edge specificity

{evidence_md}

The shuffled control preserves each target's six-vector multiset within each training destination, so LOCAL-vs-SHUFFLED contrasts test edge-evidence correspondence. `TARGET_ONLY` measures target/context-level signal. The distinction between total and within-target metrics is essential: a positive total correlation alone does not show that the probe can rank different incoming edges for one recipient.

## 11. High-margin diagnostic

Within each outer test fold and Delta target, the high-margin subset is defined as `|true Delta| >= test-fold q75`; it is used for evaluation only. No rows are excluded from primary fits or metrics.

{high_md}

## 12–13. M0StateDirect capacity under utility supervision

{direct_md}

This is a utility-supervised capacity probe, not an NC architecture or router. Compare it with `ENDPOINT_LOCAL` descriptively; direct utility supervision is stronger than task-derived learning.

## 14–17. Frozen task-trained E0.1 state decodability

{frozen_md}

`U_MODAL` is the E0.1 router input. `Q_PAIR`, `R_SHARED` and `U_MODAL` are 64D and were extracted from fully frozen KeepEdge checkpoints after exact ordered edge alignment. Differences from DirectState locate where task-trained state may fail to retain a directly supervised signal; they are not matched-supervision performance comparisons.

## 18. Diagnostic judgment

**Final label: `{label}`.**

The dataset-specific rationale and all individual question answers are recorded below. The label is based on the full reliability, correspondence-control, within-target, high-margin and representation evidence rather than a single metric.

This classification combines cross-seed target reliability, target-only vs endpoint/local evidence, the correspondence-shuffled control, within-target residual/rank metrics, high-margin results, DirectState, and frozen Q/R/U. It does not use a universal Spearman threshold.

## 19–20. Limits and next step

The target is a P1.3 diagnostic derived from validation labels and a joint head selected on that same validation split. Outer target-group CV prevents same-destination probe train/test leakage, but it cannot undo the upstream P1.3 validation reuse. The three seeds are limited and their shared edge support is not an independent sample. Near-tie utility remains in all primary analyses; high-margin results are secondary only.

The next step should be selected after human review of this attribution. This report does not design or implement M1, change E0/E0.1, or train an end-to-end utility router.

## Correctness, execution and self-audit

- Utility formula, finite-target, artifact-presence, edge-order, master dimension, neighborhood/LOO construction, degree standardization, masks, group splits, training-only normalization, balancing weights, tuple shuffle, exact pair evidence, DirectState output shapes and degenerate AUROC behavior are covered by the L0 correctness tests. Movies/42 smoke additionally verifies P1.3 H0 regression, frozen E0.1 extraction/alignment, finite losses/gradients, all readout shapes, no visible test labels, and GPU memory.
- All EvidenceMLP variants share the same 1542D input shape and equal parameter count; their initialization is bitwise equal within dataset×seed×fold. The direct probe has direct utility labels; this does not imply NC training should learn the same mapping.
- No label, CE, utility, logits, preferred-channel, source ID or destination ID enters any feature tensor. Node IDs are used only for indexing, grouping and alignment.
- Utility targets are never called ground truth. No P1.3 AbsDiff/Product to E0 Relational/Cross mapping is made. High-margin edges do not replace or filter the full-edge analysis.

### Explicit answers to questions 1–20

{findings}

### A–J self-audit

- **A. Utility interpretation:** validation-derived P1.3 utilities are treated as diagnostic targets, not ground truth.
- **B. Same-target leakage:** every incoming edge for a `dst` stays in one outer fold and one inner split.
- **C. Target-level versus edge-level:** total correlations are interpreted alongside destination-centered residual correlations and per-target ranks.
- **D. Input leakage:** no labels, CE, utility, logits, preferred channel or numeric IDs enter the feature tensor.
- **E. Probe capacity fairness:** every EvidenceMLP uses 1542 inputs and the same parameter count; initialization is bitwise equal within each fold.
- **F. Direct supervision:** DirectState is described only as utility-supervised representational capacity, not evidence that NC training should learn it.
- **G. Function mapping:** P1.3 AbsDiff/Product are not mapped onto E0 Relational/Cross-Modal.
- **H. Target noise:** cross-seed reliability, overlap and centered Delta stability qualify all conclusions.
- **I. Within-target analysis:** residual Spearman and per-target ranking are reported; total Spearman alone is not used to claim edge-specific predictability.
- **J. High-margin selection:** the top `|Delta|` quartile is diagnostic only; near-tie edges remain in every primary analysis.

## Files

Run-level fold metrics, seed-aggregated summaries, reliability records, assignment CSV, high-margin rows, parameter audit, smoke audit and figures are under `data/` and `figures/`. Large predictions/checkpoints and extracted E0.1 state caches are in ignored `outputs/l0_relation_function_learnability_audit/`.
"""
    (RESEARCH / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--diagnostic-label", default=None)
    parser.add_argument("--findings", default="")
    parser.add_argument("--findings-file", default=None)
    args = parser.parse_args()
    summaries = save_summaries()
    if args.figures:
        qa = make_figures(summaries)
        print(json.dumps({"figures": qa}, indent=2, default=str))
    if args.report:
        findings = Path(args.findings_file).read_text(encoding="utf-8") if args.findings_file else args.findings
        make_report(summaries, args.diagnostic_label, findings)


if __name__ == "__main__":
    main()
