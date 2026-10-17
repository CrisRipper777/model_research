#!/usr/bin/env python3
"""Plot balanced-epoch, three-seed validation training curves for Hybrid-CGE V1.1.

Figure contract
---------------
Core conclusion: compare the validation learning trajectories of H1 and H2
within each dataset while keeping the seed count balanced at every displayed
epoch. This is descriptive evidence and does not isolate SameNode from H2's
training-only late-residual dropout.
Archetype: quantitative grid; 3 dataset rows × 3 metric columns.
Backend: Python/Matplotlib for rendering and all figure exports.
Source data: research/hybrid_cge_v11/data/runs.csv and epoch_history.csv.
Statistics: mean ± sample SD across the three matched seeds; n=3 throughout.
Epoch support: truncated to the minimum completed epoch across both conditions
and all seeds for each dataset, with no interpolation or imputation.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/hybrid_cge_v11/data"
OUT = ROOT / "research/hybrid_cge_v11/figures/training_curves"
DATASETS = ("Movies", "Grocery", "ele-fashion")
CONDITIONS = ("H1_hybrid", "H2_hybrid_plus")
SEEDS = (42, 43, 44)
COLORS = {"H1_hybrid": "#3569A8", "H2_hybrid_plus": "#D9782D"}
DISPLAY = {"H1_hybrid": "H1", "H2_hybrid_plus": "H2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def aggregate(rows: list[dict[str, str]], value_key: str, scale: float = 1.0):
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_epoch[int(row["epoch"])].append(float(row[value_key]) * scale)
    epochs = sorted(by_epoch)
    if not epochs or any(len(by_epoch[epoch]) != len(SEEDS) for epoch in epochs):
        raise RuntimeError(f"unbalanced seed support in {value_key}")
    means = np.asarray([statistics.mean(by_epoch[e]) for e in epochs])
    spread = np.asarray([statistics.stdev(by_epoch[e]) for e in epochs])
    return np.asarray(epochs), means, spread


def main() -> None:
    runs = read_csv(DATA / "runs.csv")
    history = read_csv(DATA / "epoch_history.csv")
    expected = {(dataset, condition, seed) for dataset in DATASETS for condition in CONDITIONS for seed in SEEDS}
    run_by = {(row["dataset"], row["condition"], int(row["seed"])): row for row in runs}
    if set(run_by) != expected or any(row["status"] != "complete" for row in runs):
        raise RuntimeError("training curves require the complete 18-run matrix")
    history_by: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in history:
        history_by[(row["dataset"], row["condition"], int(row["seed"]))].append(row)

    for dataset in DATASETS:
        expected_max = min(
            int(run_by[(dataset, condition, seed)]["epochs_completed"])
            for condition in CONDITIONS for seed in SEEDS
        )
        if expected_max < 2:
            raise RuntimeError(f"insufficient common epoch support: {dataset}")
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_rows = history_by[(dataset, condition, seed)]
                present_epochs = {int(row["epoch"]) for row in run_rows}
                if not set(range(1, expected_max + 1)).issubset(present_epochs):
                    raise RuntimeError(f"incomplete epoch history: {dataset}/{condition}/seed{seed}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.15,
    })
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.15), sharex="col", constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.875, wspace=0.30, hspace=0.34)
    column_titles = ("Validation Accuracy", "Validation Macro-F1", "Train / Validation CE")
    panel_letters = iter("abcdefghi")

    for row_index, dataset in enumerate(DATASETS):
        horizon = min(
            int(run_by[(dataset, condition, seed)]["epochs_completed"])
            for condition in CONDITIONS for seed in SEEDS
        )
        for column_index, ax in enumerate(axes[row_index]):
            ax.text(-0.12, 1.09, next(panel_letters), transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, fontweight="bold", clip_on=False)
            if row_index == 0:
                ax.set_title(column_titles[column_index], pad=6)
            ax.grid(axis="y", color="#D9DEE5", linewidth=0.45, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(1, horizon)
            ax.margins(x=0.015)

        axes[row_index, 0].set_ylabel(f"{dataset}\nAccuracy (%)")
        axes[row_index, 1].set_ylabel("Macro-F1 (%)")
        axes[row_index, 2].set_ylabel("Cross-entropy")

        for condition in CONDITIONS:
            rows = [
                row for seed in SEEDS
                for row in history_by[(dataset, condition, seed)]
                if int(row["epoch"]) <= horizon
            ]
            for column, metric, scale in (
                (0, "val_acc", 100.0),
                (1, "val_macro_f1", 100.0),
            ):
                x, mean, sd = aggregate(rows, metric, scale)
                color = COLORS[condition]
                axes[row_index, column].plot(x, mean, color=color, label=DISPLAY[condition])
                axes[row_index, column].fill_between(
                    x, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0
                )

            ax_ce = axes[row_index, 2]
            color = COLORS[condition]
            for key, style, label in (
                ("train_ce_eval", "--", f"{DISPLAY[condition]} train"),
                ("val_ce", "-", f"{DISPLAY[condition]} validation"),
            ):
                x, mean, sd = aggregate(rows, key)
                ax_ce.plot(x, mean, color=color, linestyle=style, label=label)
                ax_ce.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.09, linewidth=0)

        for column in (0, 1):
            axes[row_index, column].set_ylim(0, 100)
        for ax in axes[row_index]:
            ax.set_xlabel("Epoch")

    condition_handles = [
        Line2D([0], [0], color=COLORS[condition], lw=1.4, label=DISPLAY[condition])
        for condition in CONDITIONS
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", lw=1.2, linestyle="--", label="Train CE"),
        Line2D([0], [0], color="#555555", lw=1.2, linestyle="-", label="Validation CE"),
    ]
    fig.legend(handles=condition_handles + style_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.967), ncol=4, frameon=False,
               handlelength=1.8, columnspacing=1.4)
    fig.suptitle(
        "Hybrid-CGE V1.1 validation learning curves",
        fontsize=9, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.025,
        "Lines show mean ± sample SD across three seeds; each dataset is shown only through its shared complete epoch horizon (n=3 at every epoch).",
        ha="center", va="center", fontsize=5.8,
    )

    skills_scripts = Path.home() / ".codex/skills/nature-figure/scripts"
    sys.path.insert(0, str(skills_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "hybrid_cge_v11_training_curves"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(str(base) + ".pdf", dpi=600)
    fig.savefig(str(base) + ".png", dpi=600)
    fig.savefig(str(base) + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(str(base) + ".svg")
    plt.close(fig)
    for svg_path in (Path(str(base) + ".svg"), Path(str(base) + ".alignment.svg")):
        if svg_path.is_file():
            content = svg_path.read_text(encoding="utf-8")
            svg_path.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n", encoding="utf-8")
    print(base)


if __name__ == "__main__":
    main()
