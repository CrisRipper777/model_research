from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
FIGURE_SKILL_SCRIPTS = Path("/home/m3/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


# Editable text and consistent sans-serif typography are applied before any figure.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

OUT = PROJECT / "research/l01_shared_slot_function_identifiability"
DATA = OUT / "data"
FIG = OUT / "figures"
QA = PROJECT / "outputs/l01_shared_slot_function_identifiability/figure_qa"
EXCLUSION_AUDIT: list[dict[str, object]] = []
DATASETS = ("Movies", "Grocery", "ele-fashion")
COLORS = {"G_D_text": "#007C91", "G_P_text": "#4A6FA5",
          "G_D_visual": "#CD7F32", "G_P_visual": "#8E6C9E"}
TARGET_LABELS = {"G_D_text": "Text · AbsDiff", "G_P_text": "Text · Product",
                 "G_D_visual": "Visual · AbsDiff", "G_P_visual": "Visual · Product"}
TARGET_SHORT = {"G_D_text": "D · T", "G_P_text": "P · T",
                "G_D_visual": "D · V", "G_P_visual": "P · V"}


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    out = FIG / stem
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(QA / f"{stem}.alignment.json"),
        overlay_svg=str(QA / f"{stem}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def axes3(title: str, ylabel: str, height: float = 2.7, sharey: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, height), sharey=sharey, constrained_layout=False)
    fig.subplots_adjust(left=.09, right=.995, bottom=.26, top=.82, wspace=.34)
    fig.suptitle(title, fontsize=9, y=.98)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        ax.set_title(dataset, pad=7)
        ax.text(-.16, 1.04, "abc"[index], transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left", clip_on=False)
        ax.set_ylabel(ylabel if index == 0 else "")
        ax.grid(axis="y", color="#E7E9EC", linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
    return fig, axes


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_reliability() -> None:
    head = _read("utility_head_repeat_reliability.csv")
    cross = _read("utility_cross_seed_reliability.csv")
    fig, axes = axes3("Repeat and seed reliability of clean shared-slot gains",
                      "Edge-aligned Spearman correlation", 2.85)
    # Reserve a clear band for the shared legend between the title and panels.
    fig.subplots_adjust(top=.76)
    xloc = np.arange(4)
    targets = list(COLORS)
    for ax, dataset in zip(axes, DATASETS):
        h = head[head.dataset == dataset]
        c = cross[cross.dataset == dataset]
        for x, target in zip(xloc, targets):
            h_values = h.loc[h.target == target, "spearman"]
            c_values = c.loc[c.target == target, "spearman"]
            hv = h_values.dropna().to_numpy()
            cv = c_values.dropna().to_numpy()
            EXCLUSION_AUDIT.append({"figure": "l01_utility_reliability", "dataset": dataset,
                "target": target, "source": "head_repeat_spearman", "before": len(h_values),
                "after": len(hv), "excluded": int(len(h_values)-len(hv)),
                "reason": "undefined Spearman for constant utility vectors"})
            EXCLUSION_AUDIT.append({"figure": "l01_utility_reliability", "dataset": dataset,
                "target": target, "source": "cross_seed_spearman", "before": len(c_values),
                "after": len(cv), "excluded": int(len(c_values)-len(cv)),
                "reason": "undefined Spearman for constant utility vectors"})
            color = COLORS[target]
            if len(hv):
                ax.scatter(np.full(len(hv), x - .10), hv, s=10, color=color, alpha=.42,
                           edgecolors="none", zorder=2)
                ax.errorbar(x - .10, np.median(hv),
                            yerr=[[np.median(hv)-np.quantile(hv,.25)], [np.quantile(hv,.75)-np.median(hv)]],
                            fmt="o", ms=4, color=color, capsize=2, lw=1.1, zorder=3)
            if len(cv):
                ax.scatter(np.full(len(cv), x + .10), cv, s=12, marker="s", color="#353B42",
                           alpha=.58, edgecolors="none", zorder=2)
                ax.errorbar(x + .10, np.median(cv),
                            yerr=[[np.median(cv)-np.quantile(cv,.25)], [np.quantile(cv,.75)-np.median(cv)]],
                            fmt="s", ms=4, color="#353B42", capsize=2, lw=1.1, zorder=3)
        ax.set_xticks(xloc, [TARGET_SHORT[t] for t in targets])
        ax.set_ylim(-.45, .85)
        ax.axhline(0, color="#6E747B", linewidth=.7, linestyle="--")
    handles = [plt.Line2D([], [], marker="o", color="#444", linestyle="none", label="Head repeats"),
               plt.Line2D([], [], marker="s", color="#353B42", linestyle="none", label="Across seeds")]
    # Keep the shared key in the title band, clear of the dataset headings.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .94), ncol=2, frameon=False)
    save_figure(fig, "l01_utility_reliability")


def plot_p13_transfer() -> None:
    frame = _read("p13_vs_shared_slot.csv")
    fig, axes = axes3("Historical P1.3 deltas versus clean shared-slot gains",
                      "Historical Δ vs clean gain Spearman", 2.8)
    ordered = [("absdiff", "text", "D · T"), ("product", "text", "P · T"),
               ("absdiff", "visual", "D · V"), ("product", "visual", "P · V")]
    for ax, dataset in zip(axes, DATASETS):
        sub = frame[frame.dataset == dataset]
        for x, (operator, modality, label) in enumerate(ordered):
            values = sub[(sub.operator == operator) & (sub.modality == modality)]
            color = COLORS["G_D_text" if operator == "absdiff" and modality == "text" else
                           "G_P_text" if operator == "product" and modality == "text" else
                           "G_D_visual" if operator == "absdiff" else "G_P_visual"]
            ax.scatter(np.full(len(values), x), values.spearman, s=18, color=color,
                       edgecolor="white", linewidth=.35, zorder=3)
            if len(values):
                mean = values.spearman.mean()
                sd = values.spearman.std(ddof=1) if len(values) > 1 else 0
                ax.errorbar(x, mean, yerr=sd, fmt="_", color="#282D32", capsize=2, lw=1.1, zorder=4)
        ax.axhline(0, color="#6E747B", lw=.7, ls="--")
        ax.set_xticks(range(4), [v[2] for v in ordered])
        ax.set_ylim(-.45, .85)
    fig.text(.5, .035, "Each point is one seed; black tick and whisker show mean ± seed SD", ha="center", fontsize=6)
    save_figure(fig, "l01_p13_vs_shared_slot")


def plot_output_separation() -> None:
    frame = _read("output_separability.csv")
    fig, axes = axes3("Empirical output separation under single-edge substitutions",
                      "Per-edge Jensen–Shannon divergence", 2.9)
    ordered = list(COLORS)
    for ax, dataset in zip(axes, DATASETS):
        sub = frame[frame.dataset == dataset]
        for x, target in enumerate(ordered):
            values = sub[sub.target == target]
            if values.empty:
                continue
            color = COLORS[target]
            med = float(values.js_median.median())
            low = float(values.js_q25.median())
            high = float(values.js_q75.median())
            ax.errorbar(x, med, yerr=[[max(0, med-low)], [max(0, high-med)]], fmt="o",
                        color=color, markersize=4, capsize=2, lw=1.2, zorder=3)
            ax.scatter(np.full(len(values), x), values.js_median, s=8, color=color, alpha=.4, zorder=2)
        ax.set_xticks(range(4), [TARGET_SHORT[t] for t in ordered])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Jensen–Shannon divergence" if ax is axes[0] else "")
    fig.text(.5, .035, "Points show dataset/seed/head-repeat edge-distribution medians; bars span median q25–q75", ha="center", fontsize=6)
    save_figure(fig, "l01_output_separability")


def load_predictability() -> pd.DataFrame:
    files = ["evidence_probe_summary.csv", "shuffled_null_summary.csv",
             "ridge_probe_summary.csv", "direct_state_probe_summary.csv",
             "frozen_state_readout_summary.csv"]
    frames = [pd.read_csv(DATA / name) for name in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def _model_order() -> list[str]:
    return ["TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL", "STRICT_SHUFFLE",
            "RIDGE_TARGET_ONLY", "RIDGE_ENDPOINT", "RIDGE_ENDPOINT_LOCAL",
            "DirectState", "Frozen_Q_PAIR", "Frozen_R_SHARED", "Frozen_U_MODAL"]


def _label_model(name: str) -> str:
    return {"TARGET_ONLY": "Target MLP", "ENDPOINT": "Endpoint MLP",
            "ENDPOINT_LOCAL": "Local MLP", "STRICT_SHUFFLE": "Strict shuffle",
            "RIDGE_TARGET_ONLY": "Target Ridge", "RIDGE_ENDPOINT": "Endpoint Ridge",
            "RIDGE_ENDPOINT_LOCAL": "Local Ridge", "DirectState": "Direct state",
            "Frozen_Q_PAIR": "Frozen Q", "Frozen_R_SHARED": "Frozen R",
            "Frozen_U_MODAL": "Frozen U"}.get(name, name)


def _short_model_code(name: str) -> str:
    return {"TARGET_ONLY": "T", "ENDPOINT": "E", "ENDPOINT_LOCAL": "L",
            "STRICT_SHUFFLE": "Sh", "RIDGE_TARGET_ONLY": "RT",
            "RIDGE_ENDPOINT": "RE", "RIDGE_ENDPOINT_LOCAL": "RL",
            "DirectState": "DS", "Frozen_Q_PAIR": "FQ",
            "Frozen_R_SHARED": "FR", "Frozen_U_MODAL": "FU"}.get(name, name)


def canonical_model(name: str) -> str:
    if name.startswith("STRICT_SHUFFLE_"):
        return "STRICT_SHUFFLE"
    return name


def plot_predictability(metric: str, ylabel: str, stem: str, title: str,
                        models: list[str], ylimits: tuple[float, float] | None = None) -> None:
    frame = load_predictability()
    frame["model_key"] = frame.model.map(canonical_model)
    metric_col = f"{metric}_mean"
    sd_col = f"{metric}_std"
    if metric_col not in frame:
        raise KeyError(metric_col)
    fig, axes = axes3(title, ylabel, 3.15)
    targets = list(COLORS)
    offsets = np.linspace(-.24, .24, len(targets))
    for ax, dataset in zip(axes, DATASETS):
        sub = frame[(frame.dataset == dataset) & frame.model_key.isin(models)]
        for x, model in enumerate(models):
            for offset, target in zip(offsets, targets):
                values = sub[sub.target == target]
                values = values[values.model_key == model]
                if values.empty:
                    continue
                mu = float(values[metric_col].mean())
                sd = float(values[sd_col].fillna(0).mean()) if sd_col in values else 0
                ax.errorbar(x + offset, mu, yerr=sd, fmt="o", ms=3.4,
                            color=COLORS[target], ecolor=COLORS[target], alpha=.9,
                            capsize=1.8, elinewidth=.7, zorder=3)
        ax.axhline(0, color="#747A81", lw=.7, ls="--")
        ax.set_xticks(range(len(models)), [_short_model_code(m) for m in models])
        ax.set_xlim(-.55, len(models)-.45)
        if ylimits:
            ax.set_ylim(*ylimits)
    handles = [plt.Line2D([], [], marker="o", color=COLORS[target], linestyle="none", label=TARGET_SHORT[target])
               for target in targets]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .89), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.30, top=.77)
    fig.text(.5, .025,
             "T/E/L/Sh = target/endpoint/endpoint+local/shuffled MLP; RT/RE/RL = Ridge; DS = DirectState; FQ/FR/FU = frozen Q/R/U",
             ha="center", va="bottom", fontsize=5.5)
    save_figure(fig, stem)


def plot_state_capacity() -> None:
    models = ["ENDPOINT_LOCAL", "DirectState", "Frozen_Q_PAIR", "Frozen_R_SHARED", "Frozen_U_MODAL"]
    plot_predictability("within_target_residual_spearman", "Within-target residual Spearman",
        "l01_state_capacity", "Capacity/readout comparison on clean shared-slot gains",
        models, (-.35, .55))


def generate_all() -> dict[str, str]:
    required = ["utility_head_repeat_reliability.csv", "utility_cross_seed_reliability.csv",
                "p13_vs_shared_slot.csv", "output_separability.csv", "evidence_probe_summary.csv",
                "shuffled_null_summary.csv", "ridge_probe_summary.csv",
                "direct_state_probe_summary.csv", "frozen_state_readout_summary.csv"]
    missing = [name for name in required if not (DATA / name).is_file()]
    if missing:
        raise FileNotFoundError(f"formal campaign summary files missing: {missing}")
    plot_reliability()
    plot_p13_transfer()
    plot_output_separation()
    plot_predictability("spearman", "Total edge Spearman", "l01_edge_predictability",
        "Total edge predictability is dataset- and target-dependent",
        _model_order(), (-.25, .65))
    plot_predictability("within_target_residual_spearman", "Within-target residual Spearman",
        "l01_within_target_predictability", "Most local probes add little within-recipient edge ranking",
        ["TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL", "STRICT_SHUFFLE",
         "RIDGE_ENDPOINT", "RIDGE_ENDPOINT_LOCAL", "DirectState", "Frozen_U_MODAL"], (-.35, .55))
    plot_state_capacity()
    output = {"figures": sorted(p.name for p in FIG.glob("*.png")),
              "qa_directory": str(QA), "backend": "python/matplotlib",
              "exports_per_figure": ["png", "svg", "pdf", "tiff"],
              "panel_alignment_tolerance_pt": 1.5,
              "data_exclusion_audit": EXCLUSION_AUDIT}
    (QA / "figure_qa_summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(json.dumps(generate_all(), indent=2))
