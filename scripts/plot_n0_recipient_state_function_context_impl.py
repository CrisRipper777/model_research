from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIGURE_SKILL_SCRIPTS = Path("/home/m3/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

OUT = ROOT / "research/n0_recipient_state_function_context"
DATA = OUT / "data"
FIG = OUT / "figures"
QA = ROOT / "outputs/n0_recipient_state_function_context/figure_qa"
DATASETS = ("Movies", "Grocery", "ele-fashion")
TARGETS = ("G_D_text", "G_P_text", "G_D_visual", "G_P_visual")
TARGET_LABELS = {"G_D_text": "Text · AbsDiff", "G_P_text": "Text · Product",
                 "G_D_visual": "Visual · AbsDiff", "G_P_visual": "Visual · Product"}
COLORS = {"G_D_text": "#007C91", "G_P_text": "#4A6FA5",
          "G_D_visual": "#CD7F32", "G_P_visual": "#8E6C9E"}
BACKGROUND_ORDER = ("SS", "SD", "SP", "DS", "DD", "DP", "PS", "PD", "PP")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def panel_figure(title: str, ylabel: str, height: float = 2.8):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, height), sharey=True)
    fig.subplots_adjust(left=.085, right=.995, bottom=.27, top=.80, wspace=.30)
    fig.suptitle(title, fontsize=9, y=.97)
    fig.text(.995, .012, "Error bars show seed mean ± population SD (3 seeds)",
             ha="right", va="bottom", fontsize=6, color="#444444")
    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        ax.set_title(dataset, pad=6)
        ax.text(-.16, 1.04, "abc"[i], transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left", clip_on=False)
        ax.set_ylabel(ylabel if i == 0 else "")
        ax.grid(axis="y", color="#E7E9EC", linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
    return fig, axes


def seed_mean_sd(frame: pd.DataFrame, group: list[str], col: str) -> pd.DataFrame:
    return frame.groupby(group, sort=False)[col].agg(
        mean="mean", sd=lambda x: float(np.nanstd(x, ddof=0))).reset_index()


def seed_mean_sd_metrics(frame: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    named = {}
    for metric in metrics:
        named[f"{metric}_mean"] = (metric, "mean")
        named[f"{metric}_sd"] = (metric, lambda values: float(np.nanstd(values, ddof=0)))
    result = frame.groupby(group, sort=False).agg(**named).reset_index()
    return result


def save(fig: plt.Figure, stem: str) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(
        fig, json_out=str(QA / f"{stem}.alignment.json"),
        overlay_svg=str(QA / f"{stem}.alignment.svg"), tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    svg_path = FIG / f"{stem}.svg"
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {"figure": stem, "panel_alignment": "passed", "formats": ["png", "svg", "pdf", "tiff"],
            "dpi_png": 600}


def plot_gain_range() -> dict:
    data = read("contextuality_by_run.csv")
    agg = seed_mean_sd(data, ["dataset", "target"], "mean_context_range")
    fig, axes = panel_figure("Conditional gain range across recipient backgrounds",
                             "Mean per-edge gain range", 2.9)
    x = np.arange(len(TARGETS)); width = .68
    for ax, dataset in zip(axes, DATASETS):
        block = agg[agg.dataset == dataset].set_index("target").reindex(TARGETS)
        ax.bar(x, block["mean"], width, color=[COLORS[t] for t in TARGETS],
               yerr=block["sd"], capsize=2, linewidth=0, zorder=3)
        ax.set_xticks(x, ["D·T", "P·T", "D·V", "P·V"])
    fig.supxlabel("Candidate modality and replacement operator", y=.06)
    return save(fig, "n0_context_gain_range")


def plot_sign_switch() -> dict:
    data = read("sign_switch_summary.csv")
    data = data[data.scope == "full_grid"]
    by_seed = data.groupby(["dataset", "seed", "target"], sort=False)[
        ["ordinary_sign_switch_fraction", "robust_sign_switch_fraction"]].mean().reset_index()
    data = seed_mean_sd_metrics(by_seed, ["dataset", "target"],
        ["ordinary_sign_switch_fraction", "robust_sign_switch_fraction"])
    fig, axes = panel_figure("Conditional gain sign switching across the full grid",
                             "Fraction of validation edges", 2.9)
    x = np.arange(len(TARGETS)); width = .34
    for ax, dataset in zip(axes, DATASETS):
        block = data[data.dataset == dataset].set_index("target").reindex(TARGETS)
        ax.bar(x - width/2, block.ordinary_sign_switch_fraction_mean, width,
               yerr=block.ordinary_sign_switch_fraction_sd, capsize=2,
               color="#7693A8", label="Ordinary", zorder=3)
        ax.bar(x + width/2, block.robust_sign_switch_fraction_mean, width,
               yerr=block.robust_sign_switch_fraction_sd, capsize=2,
               color="#D88762", label="All-head robust", zorder=3)
        ax.set_xticks(x, ["D·T", "P·T", "D·V", "P·V"])
        ax.set_ylim(0, .12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .85))
    fig.subplots_adjust(top=.74)
    fig.supxlabel("Candidate modality and replacement operator", y=.06)
    return save(fig, "n0_sign_switch")


def plot_rank_stability() -> dict:
    data = read("within_target_rank_stability.csv")
    data = data[data.background != "SS"]
    ceiling_raw = read("head_repeat_rank_stability.csv")
    ceiling_seed = ceiling_raw.groupby(["dataset", "seed"], sort=False)["mean"].mean().reset_index()
    ceiling = {row.dataset: (row.mean, row.sd) for row in seed_mean_sd(
        ceiling_seed, ["dataset"], "mean").itertuples(index=False)}
    fig, axes = panel_figure("Within-target edge ranking across recipient backgrounds",
                             "Mean target-level Spearman ρ", 3.0)
    x = np.arange(len(BACKGROUND_ORDER) - 1)
    backgrounds = [b for b in BACKGROUND_ORDER if b != "SS"]
    for ax, dataset in zip(axes, DATASETS):
        block = data[data.dataset == dataset]
        for target in TARGETS:
            by_seed = block[block.target == target].groupby(["seed", "background"], sort=False)["mean"].mean().reset_index()
            vals = seed_mean_sd(by_seed, ["background"], "mean").set_index("background")
            vals = vals.reindex(backgrounds)
            ax.errorbar(x, vals["mean"], yerr=vals["sd"], marker="o", ms=2.5, lw=1,
                        capsize=1.5, color=COLORS[target], label=TARGET_LABELS[target])
        center, spread = ceiling[dataset]
        ax.axhspan(center-spread, center+spread, color="#444444", alpha=.10, zorder=1)
        ax.axhline(center, color="#444444", ls="--", lw=.9,
                   label="SS head-repeat mean")
        ax.set_xticks(x, backgrounds)
        ax.set_ylim(-1.05, 1.05)
    axes[-1].legend(frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0))
    fig.supxlabel("Alternative recipient background (SS reference omitted)", y=.06)
    fig.subplots_adjust(right=.87)
    return save(fig, "n0_rank_stability")


def plot_same_cross() -> dict:
    same = read("same_modal_context.csv")
    cross = read("cross_modal_context.csv")
    data = pd.concat([same, cross], ignore_index=True)
    by_seed = data.groupby(["dataset", "seed", "context_scope", "target"], sort=False)["mean_abs_gain_change"].mean().reset_index()
    agg = seed_mean_sd(by_seed, ["dataset", "context_scope", "target"], "mean_abs_gain_change")
    fig, axes = panel_figure("Same-modal and cross-modal recipient-state effects",
                             "Mean |gain change from SS|", 2.9)
    x = np.arange(len(TARGETS)); width = .34
    for ax, dataset in zip(axes, DATASETS):
        subset = agg[agg.dataset == dataset]
        for offset, scope, color, label in ((-width/2, "same_modal", "#4A6FA5", "Same modality"),
                                            (width/2, "cross_modal", "#CD7F32", "Other modality")):
            block = subset[subset.context_scope == scope].set_index("target").reindex(TARGETS)
            ax.bar(x + offset, block["mean"], width, yerr=block["sd"], capsize=2,
                   color=color, label=label, linewidth=0, zorder=3)
        ax.set_xticks(x, ["D·T", "P·T", "D·V", "P·V"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .85))
    fig.subplots_adjust(top=.74)
    fig.supxlabel("Candidate modality and replacement operator", y=.06)
    return save(fig, "n0_same_vs_cross_modal")


def plot_context_head() -> dict:
    data = read("head_uncertainty_vs_context.csv")
    agg = seed_mean_sd(data, ["dataset", "target"], "mean_context_to_head_ratio")
    fig, axes = panel_figure("Recipient-state variation relative to head-repeat variation",
                             "Context-to-head SD ratio", 2.9)
    x = np.arange(len(TARGETS))
    for ax, dataset in zip(axes, DATASETS):
        block = agg[agg.dataset == dataset].set_index("target").reindex(TARGETS)
        ax.bar(x, block["mean"], .68, color=[COLORS[t] for t in TARGETS],
               yerr=block["sd"], capsize=2, linewidth=0, zorder=3)
        ax.axhline(1.0, color="#555555", lw=.8, ls="--", zorder=2)
        ax.set_xticks(x, ["D·T", "P·T", "D·V", "P·V"])
    fig.supxlabel("Candidate modality and replacement operator", y=.06)
    return save(fig, "n0_context_vs_head_uncertainty")


def plot_gradient_decomposition() -> dict:
    data = read("gradient_decomposition.csv")
    by_seed = data.groupby(["dataset", "seed", "target", "approximation"], sort=False)[
        "relative_error_median"].mean().reset_index()
    data = seed_mean_sd(by_seed, ["dataset", "target", "approximation"], "relative_error_median")
    fig, axes = panel_figure("Second-order CE geometry improves gain approximation",
                             "Mean relative absolute gain error", 2.9)
    x = np.arange(len(TARGETS)); width = .34
    styles = (("first_order", -width/2, "#7693A8", "First order"),
              ("second_order", width/2, "#D88762", "Second order"))
    for ax, dataset in zip(axes, DATASETS):
        subset = data[data.dataset == dataset]
        for approximation, offset, color, label in styles:
            block = subset[subset.approximation == approximation].set_index("target").reindex(TARGETS)
            if (not np.isfinite(block["mean"]).all()) or (block["mean"] <= 0).any():
                raise ValueError("Log-scale relative errors must be strictly positive")
            ax.bar(x + offset, block["mean"], width, yerr=block["sd"], capsize=2,
                   color=color, label=label, linewidth=0, zorder=3)
        ax.set_xticks(x, ["D·T", "P·T", "D·V", "P·V"])
        ax.set_yscale("log")
        ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ax.set_yticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"])
        ax.minorticks_off()
        ax.set_ylim(1e-5, .5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .85))
    fig.subplots_adjust(top=.74)
    fig.supxlabel("Candidate modality and replacement operator", y=.06)
    return save(fig, "n0_gradient_decomposition")


def generate_all() -> dict:
    results = [plot_gain_range(), plot_sign_switch(), plot_rank_stability(),
               plot_same_cross(), plot_context_head(), plot_gradient_decomposition()]
    manifest = {"status": "passed", "figures": results,
                "required_pngs": [f"{name}.png" for name in (
                    "n0_context_gain_range", "n0_sign_switch", "n0_rank_stability",
                    "n0_same_vs_cross_modal", "n0_context_vs_head_uncertainty",
                    "n0_gradient_decomposition")],
                "notes": "Panel alignment audited before export; collision and PDF text audits are stored alongside outputs; source tables aggregate over three seeds where figures summarize runs."}
    QA.mkdir(parents=True, exist_ok=True)
    (QA / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
