#!/usr/bin/env python3
"""Python plotting and lightweight export QA for the RM-CGE study figures."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "sans-serif"],
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/rmcge_v1_generalization/data"
OUT = ROOT / "research/rmcge_v1_generalization/figures"
DATASETS = ("Movies", "Grocery", "ele-fashion")
TREATMENTS = ("O0", "O1", "O2", "O3")
COLORS = {"O0": "#4C78A8", "O1": "#F58518", "O2": "#54A24B", "O3": "#E45756"}


def _csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig, stem: str, title_tokens: list[str]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    clipped = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        bbox = text.get_window_extent(renderer=renderer)
        if not text.get_clip_on() and (bbox.x0 < page.x0 - 1.5 or bbox.y0 < page.y0 - 1.5 or bbox.x1 > page.x1 + 1.5 or bbox.y1 > page.y1 + 1.5):
            clipped.append(text.get_text())
    axes = [axis.get_window_extent(renderer=renderer) for axis in fig.axes]
    overlaps = []
    for left_index, left in enumerate(axes):
        for right_index in range(left_index + 1, len(axes)):
            right = axes[right_index]
            intersection = left.overlaps(right)
            if intersection:
                dx = min(left.x1, right.x1) - max(left.x0, right.x0)
                dy = min(left.y1, right.y1) - max(left.y0, right.y0)
                if dx > 1.0 and dy > 1.0:
                    overlaps.append([left_index, right_index, float(dx), float(dy)])
    from audit_panel_alignment import require_matplotlib_panel_alignment

    alignment_path = OUT / f"{stem}.alignment.json"
    alignment = require_matplotlib_panel_alignment(
        fig,
        json_out=alignment_path,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    svg_path = OUT / f"{stem}.svg"
    pdf_path = OUT / f"{stem}.pdf"
    png_path = OUT / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    paths = {
        "svg": str(svg_path.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "png": str(png_path.relative_to(ROOT)),
    }
    plt.close(fig)
    pdf_text = ""
    if shutil.which("pdftotext"):
        result = subprocess.run(
            ["pdftotext", str(OUT / f"{stem}.pdf"), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        pdf_text = result.stdout
    missing_tokens = [token for token in title_tokens if token.casefold() not in pdf_text.casefold()] if pdf_text else []
    return {
        "stem": stem,
        "files": paths,
        "alignment_json": str(alignment_path.relative_to(ROOT)),
        "alignment_status": alignment.get("status", alignment.get("verdict", "passed")),
        "canvas_text_clipping": clipped,
        "axes_collisions": overlaps,
        "pdf_text_characters": len(pdf_text),
        "pdf_missing_expected_text": missing_tokens,
        "passed": not clipped and not overlaps and (not pdf_text or not missing_tokens),
    }


def _legend_treatments(fig, handles=None, **kwargs):
    if handles is None:
        handles = [Line2D([0], [0], color=COLORS[t], lw=2, label=t) for t in TREATMENTS]
    columns = kwargs.pop("ncol", len(handles))
    fig.legend(handles=handles, loc="upper center", ncol=columns, frameon=False, **kwargs)


def _training_curves() -> dict[str, Any]:
    rows = _csv("epoch_history.csv")
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 10.4), sharex="col", layout="constrained")
    fig.suptitle("RM-CGE V1 validation trajectories", fontsize=16, fontweight="bold", y=1.13)
    study_max_epoch = max(int(row["epoch"]) for row in rows)
    metrics = (("train_ce", "val_ce", "Cross-Entropy"), ("train_accuracy_eval", "val_acc", "Accuracy"))
    for row_index, dataset in enumerate(DATASETS):
        for column, (train_metric, val_metric, title) in enumerate(metrics):
            axis = axes[row_index, column]
            for treatment in TREATMENTS:
                group = [r for r in rows if r["dataset"] == dataset and r["treatment"] == treatment]
                for record in group:
                    # Plot each seed trace once; curves stop at their own early-stopping epoch.
                    if record["epoch"] != next(r["epoch"] for r in group if int(r["seed"]) == int(record["seed"])):
                        continue
                    seed_rows = sorted((r for r in group if int(r["seed"]) == int(record["seed"])), key=lambda item: int(item["epoch"]))
                    x = np.asarray([int(r["epoch"]) for r in seed_rows])
                    axis.plot(x, [float(r[train_metric]) for r in seed_rows], color=COLORS[treatment], alpha=0.20, lw=0.8)
                    axis.plot(x, [float(r[val_metric]) for r in seed_rows], color=COLORS[treatment], alpha=0.23, lw=0.8, ls="--")
                # Epoch-specific means use only runs that reached each epoch.
                max_epoch = max(int(row["epoch"]) for row in group)
                x_values, train_mean, val_mean = [], [], []
                for epoch in range(1, max_epoch + 1):
                    available = [r for r in group if int(r["epoch"]) == epoch]
                    if available:
                        x_values.append(epoch)
                        train_mean.append(np.mean([float(r[train_metric]) for r in available]))
                        val_mean.append(np.mean([float(r[val_metric]) for r in available]))
                axis.plot(x_values, train_mean, color=COLORS[treatment], lw=2.2, label=f"{treatment} Train")
                axis.plot(x_values, val_mean, color=COLORS[treatment], lw=2.2, ls="--", label=f"{treatment} Val")
            axis.set_title(f"{dataset} · {title}", fontsize=11)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss" if column == 0 else "Fraction")
            axis.grid(alpha=0.22, linewidth=0.6)
            axis.set_xlim(0, study_max_epoch)
            if column == 1:
                axis.set_ylim(bottom=0)
    handles = [
        Line2D([0], [0], color="#333333", lw=2, ls="-", label="Train"),
        Line2D([0], [0], color="#333333", lw=2, ls="--", label="Validation"),
        *[Line2D([0], [0], color=COLORS[t], lw=2, label=t) for t in TREATMENTS],
    ]
    _legend_treatments(fig, handles, bbox_to_anchor=(0.5, 1.045))
    return _save(fig, "training_curves", ["RM-CGE V1 validation trajectories", "Cross-Entropy", "Accuracy"])


def _paired_effect_figure() -> dict[str, Any]:
    rows = _csv("paired_comparisons.csv")
    contrasts = ("O1-O0", "O2-O0", "O3-O1", "O3-O2", "Interaction=(O3-O2)-(O1-O0)")
    metrics = ("val_accuracy", "val_macro_f1", "validation_ce")
    pretty = {"val_accuracy": "Validation Accuracy (pp)", "val_macro_f1": "Validation Macro-F1 (pp)", "validation_ce": "Validation CE"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.8))
    fig.suptitle("Seed-paired regularization contrasts · Best-Accuracy checkpoints", fontsize=15, fontweight="bold", y=0.985)
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(metrics):
            axis = axes[i, j]
            for x_index, contrast in enumerate(contrasts):
                group = [r for r in rows if r["dataset"] == dataset and r["metric"] == metric and r["contrast"] == contrast]
                values = np.asarray([float(r["paired_difference"]) for r in group])
                scale = 100.0 if metric != "validation_ce" else 1.0
                values *= scale
                jitter = np.linspace(-0.07, 0.07, len(values))
                axis.scatter(x_index + jitter, values, s=28, color="#4C78A8", alpha=0.85, zorder=3)
                if len(values):
                    mean = values.mean()
                    sd = values.std(ddof=1) if len(values) > 1 else 0.0
                    axis.errorbar(x_index, mean, yerr=sd, fmt="D", color="#E45756", capsize=3, markersize=5, zorder=4)
            axis.axhline(0, color="#444444", lw=0.9, alpha=0.7)
            axis.set_xticks(range(len(contrasts)))
            axis.set_xticklabels(["O1−O0", "O2−O0", "O3−O1", "O3−O2", "Int."], rotation=35, rotation_mode="anchor", ha="right")
            axis.set_title(f"{dataset} · {pretty[metric]}", fontsize=10.5)
            axis.set_ylabel("Seed-paired difference")
            axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19, top=0.88, wspace=0.34, hspace=0.38)
    fig.text(0.5, 0.025, "Acc/F1 differences are percentage points; CE differences use CE units. Int. is the dropout × weight-decay difference-in-differences. Points are paired seeds; diamonds/bars show mean ± sample SD (n=3).", ha="center", fontsize=9)
    return _save(fig, "paired_factorial_effects", ["Seed-paired regularization contrasts", "Best-Accuracy"])


def _checkpoint_tradeoff_figure() -> dict[str, Any]:
    rows = _csv("checkpoint_validation_replay.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    fig.suptitle("Validation Accuracy and Macro-F1 by checkpoint rule", fontsize=15, fontweight="bold", y=0.99)
    markers = {42: "o", 43: "s", 44: "^"}
    for axis, dataset in zip(axes, DATASETS):
        subset = [row for row in rows if row["dataset"] == dataset]
        for treatment in TREATMENTS:
            for seed in (42, 43, 44):
                acc = next(row for row in subset if row["treatment"] == treatment and int(row["seed"]) == seed and row["selection"] == "best_acc")
                f1 = next(row for row in subset if row["treatment"] == treatment and int(row["seed"]) == seed and row["selection"] == "best_f1")
                axis.plot([float(acc["val_acc"]), float(f1["val_acc"])], [float(acc["val_macro_f1"]), float(f1["val_macro_f1"])], color=COLORS[treatment], alpha=0.28, lw=0.8)
                axis.scatter(float(acc["val_acc"]), float(acc["val_macro_f1"]), marker=markers[seed], facecolor=COLORS[treatment], edgecolor="white", linewidth=0.4, s=42, zorder=3)
                axis.scatter(float(f1["val_acc"]), float(f1["val_macro_f1"]), marker=markers[seed], facecolor="none", edgecolor=COLORS[treatment], linewidth=1.2, s=48, zorder=3)
        axis.set_title(dataset)
        axis.set_xlabel("Validation Accuracy")
        axis.set_ylabel("Validation Macro-F1")
        axis.grid(alpha=0.22, linewidth=0.6)
    treatment_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[t], markeredgecolor="white", label=t, markersize=7) for t in TREATMENTS]
    rule_handles = [Line2D([0], [0], marker="o", color="#444444", markerfacecolor="#444444", label="Best Accuracy", markersize=6), Line2D([0], [0], marker="o", color="#444444", markerfacecolor="white", label="Best Macro-F1", markersize=6)]
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.82, wspace=0.32)
    _legend_treatments(fig, treatment_handles + rule_handles, bbox_to_anchor=(0.5, 0.945), ncol=6)
    return _save(fig, "validation_acc_macro_f1", ["Validation Accuracy and Macro-F1", "checkpoint rule"])


def _best_epoch_figure() -> dict[str, Any]:
    rows = _csv("checkpoint_validation_replay.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.suptitle("Checkpoint selection epochs", fontsize=15, fontweight="bold", y=0.99)
    offsets = {"best_acc": -0.12, "best_f1": 0.12}
    marker = {"best_acc": "o", "best_f1": "s"}
    for axis, dataset in zip(axes, DATASETS):
        for i, treatment in enumerate(TREATMENTS):
            for selection in ("best_acc", "best_f1"):
                subset = [r for r in rows if r["dataset"] == dataset and r["treatment"] == treatment and r["selection"] == selection]
                for row in subset:
                    axis.scatter(i + offsets[selection], int(row["selection_epoch"]), marker=marker[selection], color=COLORS[treatment], edgecolor="white", linewidth=0.45, s=42)
        axis.set_xticks(range(len(TREATMENTS)), TREATMENTS)
        axis.set_title(dataset)
        axis.set_ylabel("Selected epoch")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    handles = [*[Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[t], label=t, markersize=7) for t in TREATMENTS], Line2D([0], [0], marker="o", color="#444444", markerfacecolor="#444444", label="Best Accuracy", markersize=6), Line2D([0], [0], marker="s", color="#444444", markerfacecolor="#444444", label="Best Macro-F1", markersize=6)]
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.82, wspace=0.30)
    _legend_treatments(fig, handles, bbox_to_anchor=(0.5, 0.945), ncol=6)
    return _save(fig, "best_epoch_distribution", ["Checkpoint selection epochs", "Selected epoch"])


def _class_f1_figure() -> dict[str, Any]:
    rows = _csv("validation_class_metrics.csv")
    treatment = "O0"
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    fig.suptitle("Per-class Validation F1: checkpoint-rule comparison · O0", fontsize=14.5, fontweight="bold", y=0.99)
    for axis, dataset in zip(axes, DATASETS):
        selected = [r for r in rows if r["dataset"] == dataset and r["treatment"] == treatment]
        supports = {}
        means = {selection: {} for selection in ("best_acc", "best_f1")}
        for class_id in sorted({int(r["class_id"]) for r in selected}):
            supports[class_id] = max(int(r["validation_count"]) for r in selected if int(r["class_id"]) == class_id)
            for selection in means:
                values = [float(r["f1"]) for r in selected if int(r["class_id"]) == class_id and r["selection"] == selection]
                means[selection][class_id] = float(np.mean(values))
        classes = sorted(supports, key=lambda class_id: (supports[class_id], class_id))
        x = np.arange(len(classes))
        for class_id, x_pos in zip(classes, x):
            axis.plot([x_pos, x_pos], [means["best_acc"][class_id], means["best_f1"][class_id]], color="#777777", alpha=0.23, lw=0.7)
        axis.scatter(x, [means["best_acc"][c] for c in classes], s=[20 + 2 * min(supports[c], 80) for c in classes], marker="o", color="#4C78A8", alpha=0.75, label="Best Accuracy")
        axis.scatter(x, [means["best_f1"][c] for c in classes], s=[20 + 2 * min(supports[c], 80) for c in classes], marker="s", facecolors="none", edgecolors="#F58518", linewidth=1.0, label="Best Macro-F1")
        axis.set_title(dataset)
        axis.set_xlabel("Classes ordered by Validation support")
        axis.set_ylabel("Per-class F1 (mean over three seeds)")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C78A8", label="Best Accuracy", markersize=7), Line2D([0], [0], marker="s", color="#F58518", markerfacecolor="white", label="Best Macro-F1", markersize=7)]
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.82, wspace=0.31)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)
    return _save(fig, "class_f1_checkpoint_tradeoff", ["Per-class Validation F1", "O0"])


def create_figures() -> dict[str, Any]:
    results = [
        _training_curves(),
        _paired_effect_figure(),
        _checkpoint_tradeoff_figure(),
        _best_epoch_figure(),
        _class_f1_figure(),
    ]
    report = {"backend": "Python Matplotlib Agg", "figures": results, "passed": all(row["passed"] for row in results)}
    if not report["passed"]:
        raise RuntimeError(f"figure QA failed: {report}")
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(create_figures(), indent=2))
