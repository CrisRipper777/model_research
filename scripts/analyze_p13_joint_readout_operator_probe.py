from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.p0p1_propagation_probe import DATASETS  # noqa: E402
from src.analysis.p11_p12_operator_rescue import MODALITIES, OPERATORS  # noqa: E402
from src.analysis.p13_joint_readout_operator_probe import (  # noqa: E402
    BLOCKS,
    modality_sign_disagreement,
    preferred_channel_from_utilities,
    summarize_within_node_preferences,
    write_json,
)

_SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if _SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(_SKILL_SCRIPTS))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError as exc:  # pragma: no cover - environment-specific figure QA helper
    raise RuntimeError("The Python figure QA helper audit_panel_alignment.py is required to render these figures") from exc

SEEDS = (42, 43, 44)
OP_COLORS = {"smooth": "#4776a8", "absdiff": "#d38b30", "product": "#4a9671", "none": "#bfc3c7"}
QUADRANT_ORDER = ("sp_ap", "sp_an", "sn_ap", "sn_an")
QUADRANT_LABELS = {"sp_ap": "S+ / alt+", "sp_an": "S+ / alt≤0", "sn_ap": "S≤0 / alt+", "sn_an": "S≤0 / alt≤0"}
QUADRANT_COLORS = {"sp_ap": "#4a9671", "sp_an": "#8da9c6", "sn_ap": "#d38b30", "sn_an": "#c66a60"}
PREF_ORDER = (*OPERATORS, "none")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _summary_rows(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Append equal-weight three-seed means and population SDs to seed rows."""
    if df.empty:
        return df
    df = df.copy()
    if "summary_level" not in df.columns:
        df["summary_level"] = "seed"
    summaries = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for level, fn in (("dataset_3seed_mean", np.mean), ("dataset_3seed_seed_sd", lambda x: np.std(x, ddof=0))):
            row = {**base, "seed": np.nan, "summary_level": level}
            for col in metric_cols:
                vals = group[col].dropna().to_numpy(dtype=float)
                row[col] = float(fn(vals)) if len(vals) else np.nan
            summaries.append(row)
    return pd.concat([df, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def build_utility_and_transition(edge_runs: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    utility_rows, transition_rows = [], []
    for (dataset, seed), frame in edge_runs.items():
        n = len(frame)
        for modality in MODALITIES:
            for operator in OPERATORS:
                values = frame[f"u_raw_{operator}_{modality}"].to_numpy(dtype=float)
                q25, median, q75, q90 = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
                utility_rows.append({
                    "dataset": dataset, "seed": seed, "modality": modality, "operator": operator,
                    "edge_count": n, "mean_raw_utility": float(np.mean(values)), "median_raw_utility": float(median),
                    "q25_raw_utility": float(q25), "q75_raw_utility": float(q75), "iqr_raw_utility": float(q75-q25),
                    "q90_raw_utility": float(q90), "positive_fraction": float(np.mean(values > 0)),
                    "negative_fraction": float(np.mean(values < 0)), "exact_zero_fraction": float(np.mean(values == 0)),
                })
            smooth = frame[f"u_raw_smooth_{modality}"].to_numpy(dtype=float)
            for alternative in ("absdiff", "product"):
                other = frame[f"u_raw_{alternative}_{modality}"].to_numpy(dtype=float)
                sp, sn = smooth > 0, smooth <= 0
                ap, an = other > 0, other <= 0
                counts = {
                    "sp_ap": int(np.sum(sp & ap)), "sp_an": int(np.sum(sp & an)),
                    "sn_ap": int(np.sum(sn & ap)), "sn_an": int(np.sum(sn & an)),
                }
                negative_strict = smooth < 0
                joint = negative_strict & (other > 0)
                transition_rows.append({
                    "dataset": dataset, "seed": seed, "modality": modality,
                    "comparison": f"smooth_vs_{alternative}", "alternative": alternative, "edge_count": n,
                    **{f"count_{k}": v for k, v in counts.items()},
                    **{f"fraction_{k}": v / n for k, v in counts.items()},
                    "smooth_exact_zero_count": int(np.sum(smooth == 0)),
                    "alternative_exact_zero_count": int(np.sum(other == 0)),
                    "smooth_negative_strict_count": int(np.sum(negative_strict)),
                    "smooth_negative_alternative_positive_joint_count": int(np.sum(joint)),
                    "p_smooth_negative_alt_positive": float(np.mean(joint)),
                    "p_alt_positive_given_smooth_negative": float(np.sum(joint) / np.sum(negative_strict)) if np.sum(negative_strict) else np.nan,
                    "smooth_positive_alt_positive_count": counts["sp_ap"],
                    "smooth_nonpositive_alt_positive_count": counts["sn_ap"],
                })
    utilities = pd.DataFrame(utility_rows)
    transitions = pd.DataFrame(transition_rows)
    utility_summary = _summary_rows(utilities, ["dataset", "modality", "operator"], [
        "mean_raw_utility", "median_raw_utility", "q25_raw_utility", "q75_raw_utility", "iqr_raw_utility", "q90_raw_utility",
        "positive_fraction", "negative_fraction", "exact_zero_fraction",
    ])
    transition_summary = _summary_rows(transitions, ["dataset", "modality", "comparison", "alternative"], [
        *[f"fraction_{k}" for k in QUADRANT_ORDER], "p_smooth_negative_alt_positive", "p_alt_positive_given_smooth_negative",
    ])
    return utility_summary, transition_summary


def build_preferences(edge_runs: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preference_rows, diversity_rows, target_diversity_rows, modality_rows, margin_rows = [], [], [], [], []
    for (dataset, seed), frame in edge_runs.items():
        preferences_by_modality: dict[str, list[str]] = {}
        for modality in MODALITIES:
            utility_matrix = np.column_stack([frame[f"u_raw_{op}_{modality}"].to_numpy(float) for op in OPERATORS])
            best_index = np.argmax(utility_matrix, axis=1)
            max_values = utility_matrix[np.arange(len(frame)), best_index]
            choices = np.array(OPERATORS, dtype=object)[best_index]
            choices[max_values <= 0] = "none"
            preferences = choices.tolist()
            preferences_by_modality[modality] = preferences
            counts = {op: int(np.sum(choices == op)) for op in PREF_ORDER}
            preference_rows.append({
                "dataset": dataset, "seed": seed, "modality": modality, "edge_count": len(frame),
                **{f"count_{op}": counts[op] for op in PREF_ORDER},
                **{f"fraction_{op}": counts[op] / len(frame) for op in PREF_ORDER},
                "none_definition": "all three tested raw marginal utilities are nonpositive",
            })
            eligible = frame["dst_degree"].to_numpy(int) >= 5
            targets = frame.loc[eligible, "dst"].to_numpy(int)
            degrees = frame.loc[eligible, "dst_degree"].to_numpy(int)
            pref = np.asarray(preferences, dtype=object)[eligible].tolist()
            per_target, summary = summarize_within_node_preferences(targets, degrees, pref, degree_min=5)
            diversity_rows.append({"dataset": dataset, "seed": seed, "modality": modality, **summary})
            target_diversity_rows.extend({"dataset": dataset, "seed": seed, "modality": modality,
                                          "summary_level": "target", **row} for row in per_target)
            margins = np.sort(utility_matrix, axis=1)[:, ::-1]
            q25, median, q75, q90 = np.quantile(margins[:, 0] - margins[:, 1], [0.25, 0.5, 0.75, 0.9])
            margin_rows.append({
                "dataset": dataset, "seed": seed, "modality": modality, "edge_count": len(frame),
                "preference_margin_median": float(median), "preference_margin_q25": float(q25),
                "preference_margin_q75": float(q75), "preference_margin_iqr": float(q75-q25),
                "preference_margin_q90": float(q90), "mean_preference_margin": float(np.mean(margins[:, 0]-margins[:, 1])),
            })

        text_pref = np.asarray(preferences_by_modality["text"], dtype=object)
        visual_pref = np.asarray(preferences_by_modality["visual"], dtype=object)
        overall_disagreement = float(np.mean(text_pref != visual_pref))
        active = (text_pref != "none") & (visual_pref != "none")
        active_disagreement = float(np.mean(text_pref[active] != visual_pref[active])) if np.any(active) else np.nan
        for text_choice in PREF_ORDER:
            for visual_choice in PREF_ORDER:
                count = int(np.sum((text_pref == text_choice) & (visual_pref == visual_choice)))
                modality_rows.append({
                    "dataset": dataset, "seed": seed, "text_preference": text_choice,
                    "visual_preference": visual_choice, "edge_count": len(frame), "cell_count": count,
                    "cell_fraction": count / len(frame), "overall_preference_disagreement": overall_disagreement,
                    "both_active_edge_count": int(np.sum(active)), "active_preference_disagreement": active_disagreement,
                })

    preferences = pd.DataFrame(preference_rows)
    preference_summary = _summary_rows(preferences, ["dataset", "modality"], [f"fraction_{op}" for op in PREF_ORDER])
    diversity_seed = pd.DataFrame(diversity_rows)
    diversity_summary = _summary_rows(diversity_seed, ["dataset", "modality"], [
        "multi_function_neighborhood_ratio", "median_distinct_positive_functions_per_target",
        "no_positive_preference_target_fraction", "median_positive_preferred_edge_count",
    ])
    diversity_targets = pd.DataFrame(target_diversity_rows)
    diversity = pd.concat([diversity_targets, diversity_summary], ignore_index=True, sort=False)
    modality = pd.DataFrame(modality_rows)
    modality_summary = _summary_rows(modality, ["dataset", "text_preference", "visual_preference"], [
        "cell_fraction", "overall_preference_disagreement", "active_preference_disagreement",
    ])
    margin = pd.DataFrame(margin_rows)
    margin_summary = _summary_rows(margin, ["dataset", "modality"], [
        "preference_margin_median", "preference_margin_q25", "preference_margin_q75", "preference_margin_iqr",
        "preference_margin_q90", "mean_preference_margin",
    ])
    return preference_summary, diversity, modality_summary, margin_summary


def build_channel_audits(run_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    usage_rows, ablation_rows = [], []
    for record in run_records:
        for row in record["channel_usage"]:
            usage_rows.append({"dataset": record["dataset"], "seed": record["seed"], **row})
        for row in record["whole_channel_ablation"]:
            ablation_rows.append({"dataset": record["dataset"], "seed": record["seed"], **row})
    usage = pd.DataFrame(usage_rows)
    usage = _summary_rows(usage, ["dataset", "block"], [
        "weight_frobenius_norm", "validation_feature_rms", "mean_logit_contribution_l2", "median_logit_contribution_l2",
    ])
    ablation = pd.DataFrame(ablation_rows)
    ablation = _summary_rows(ablation, ["dataset", "block"], [
        "delta_val_ce", "delta_val_acc", "delta_val_macro_f1",
    ])
    return usage, ablation


def build_modality_sign_disagreement() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = Path("outputs/p13_joint_readout_operator_probe/runs") / dataset / f"seed_{seed}" / "joint_operator_edge_utility.csv.gz"
            frame = pd.read_csv(path, usecols=["src", "dst", *[f"u_raw_{op}_{mod}" for op in OPERATORS for mod in MODALITIES]])
            for operator in OPERATORS:
                text = frame[f"u_raw_{operator}_text"].to_numpy(float)
                visual = frame[f"u_raw_{operator}_visual"].to_numpy(float)
                sign_metrics = modality_sign_disagreement(text, visual)
                rows.append({
                    "dataset": dataset, "seed": seed, "operator": operator, "edge_count": len(frame),
                    "text_visual_sign_disagreement": sign_metrics["three_way_sign_disagreement"],
                    "positive_vs_nonpositive_sign_disagreement": sign_metrics["positive_vs_nonpositive_sign_disagreement"],
                    "text_positive_fraction": float(np.mean(text > 0)),
                    "visual_positive_fraction": float(np.mean(visual > 0)),
                    "text_exact_zero_fraction": sign_metrics["text_exact_zero_fraction"],
                    "visual_exact_zero_fraction": sign_metrics["visual_exact_zero_fraction"],
                })
    out = pd.DataFrame(rows)
    p12 = pd.read_csv("research/p11_p12_operator_rescue/p12/operator_modality_disagreement.csv")
    p12 = p12[p12.summary_level == "seed"][["dataset", "seed", "operator", "text_visual_sign_disagreement"]]
    p12 = p12.rename(columns={"text_visual_sign_disagreement": "p12_positive_vs_nonpositive_disagreement"})
    out = out.merge(p12, on=["dataset", "seed", "operator"], how="left", validate="one_to_one")
    out["p13_positive_vs_nonpositive_disagreement_minus_p12"] = (
        out.positive_vs_nonpositive_sign_disagreement - out.p12_positive_vs_nonpositive_disagreement
    )
    return _summary_rows(out, ["dataset", "operator"], [
        "text_visual_sign_disagreement", "positive_vs_nonpositive_sign_disagreement",
        "p12_positive_vs_nonpositive_disagreement", "p13_positive_vs_nonpositive_disagreement_minus_p12",
        "text_positive_fraction", "visual_positive_fraction", "text_exact_zero_fraction", "visual_exact_zero_fraction",
    ])


def build_p12_comparison(transitions: pd.DataFrame, modality_disagreement: pd.DataFrame) -> pd.DataFrame:
    p12_rescue = pd.read_csv("research/p11_p12_operator_rescue/p12/operator_rescue_summary.csv")
    p12_rescue = p12_rescue[p12_rescue.summary_level == "seed"]
    p13 = transitions[transitions.summary_level == "seed"].copy()
    rows = []
    for _, row in p13.iterrows():
        alt = row["alternative"]
        old = p12_rescue[
            (p12_rescue.dataset == row.dataset)
            & (p12_rescue.seed.astype(int) == int(row.seed))
            & (p12_rescue.modality == row.modality)
        ]
        if len(old) != 1:
            raise AssertionError(f"P1.2 transition row missing for {row.dataset}/{row.seed}/{row.modality}")
        old = old.iloc[0]
        p12_joint = float(old[f"{alt}_joint_prevalence"])
        p12_cond = float(old[f"{alt}_rescue_conditional"])
        for metric, p12_value, p13_value in (
            ("S_negative_alt_positive_joint_prevalence", p12_joint, float(row.p_smooth_negative_alt_positive)),
            ("alt_positive_given_S_negative", p12_cond, float(row.p_alt_positive_given_smooth_negative)),
        ):
            rows.append({"dataset": row.dataset, "seed": int(row.seed), "modality": row.modality,
                         "comparison_type": "transition", "operator": alt, "metric": metric,
                         "p12_value": p12_value, "p13_value": p13_value, "p13_minus_p12": p13_value-p12_value})
    sign = modality_disagreement[modality_disagreement.summary_level == "seed"]
    for _, row in sign.iterrows():
        rows.append({"dataset": row.dataset, "seed": int(row.seed), "modality": "text_visual",
                     "comparison_type": "operator_modality_sign_disagreement", "operator": row.operator,
                     "metric": "T_V_positive_vs_nonpositive_disagreement", "p12_value": float(row.p12_positive_vs_nonpositive_disagreement),
                     "p13_value": float(row.positive_vs_nonpositive_sign_disagreement),
                     "p13_minus_p12": float(row.p13_positive_vs_nonpositive_disagreement_minus_p12)})
    out = pd.DataFrame(rows)
    summaries = []
    for keys, group in out.groupby(["dataset", "modality", "comparison_type", "operator", "metric"], sort=True):
        ds, modality, kind, op, metric = keys
        for level, fn in (("dataset_3seed_mean", np.mean), ("dataset_3seed_seed_sd", lambda x: np.std(x, ddof=0))):
            summaries.append({
                "dataset": ds, "seed": np.nan, "modality": modality, "comparison_type": kind,
                "operator": op, "metric": metric, "summary_level": level,
                "p12_value": float(fn(group["p12_value"].to_numpy(float))),
                "p13_value": float(fn(group["p13_value"].to_numpy(float))),
                "p13_minus_p12": float(fn(group["p13_minus_p12"].to_numpy(float))),
            })
    out["summary_level"] = "seed"
    return pd.concat([out, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def build_edge_sample(edge_runs: dict[tuple[str, int], pd.DataFrame], sample_per_run: int = 100) -> pd.DataFrame:
    parts = []
    dataset_offset = {name: i for i, name in enumerate(DATASETS)}
    for (dataset, seed), frame in edge_runs.items():
        rng = np.random.default_rng(20260930 + 100 * dataset_offset[dataset] + seed)
        take = np.sort(rng.choice(len(frame), size=min(sample_per_run, len(frame)), replace=False))
        sample = frame.iloc[take].copy().reset_index(drop=True)
        sample["dataset"] = dataset
        sample["seed"] = seed
        for modality in MODALITIES:
            utility_cols = [f"u_raw_{op}_{modality}" for op in OPERATORS]
            values = sample[utility_cols].to_numpy(float)
            ix = values.argmax(axis=1)
            best = np.asarray(OPERATORS, dtype=object)[ix]
            best[values.max(axis=1) <= 0] = "none"
            sample[f"preferred_channel_{modality}"] = best
        parts.append(sample)
    return pd.concat(parts, ignore_index=True)


def _add_labels(axes: np.ndarray) -> None:
    for i, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.05, chr(ord("a")+i), transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=10)


def _save_figure(fig: plt.Figure, path: Path, qa_dir: Path, alignment_json: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(qa_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(qa_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(qa_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    if alignment_json is not None and not alignment_json.exists():
        raise RuntimeError(f"panel alignment report was not produced: {alignment_json}")
    plt.close(fig)


def render_figures(
    transitions: pd.DataFrame,
    preferences: pd.DataFrame,
    modality_preference: pd.DataFrame,
    comparison: pd.DataFrame,
    figure_dir: Path,
    qa_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: full four-cell sign transitions, all seed rows plus the 3-seed mean.
    seed_transition = transitions[transitions.summary_level == "seed"]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 9.2), sharex=True, sharey=True)
    x = np.arange(4)
    for i, dataset in enumerate(DATASETS):
        for j, modality in enumerate(MODALITIES):
            ax = axes[i, j]
            subset = seed_transition[(seed_transition.dataset == dataset) & (seed_transition.modality == modality)]
            cursor = 0
            xpos, labels = [], []
            for alternative in ("absdiff", "product"):
                for seed in SEEDS:
                    row = subset[(subset.alternative == alternative) & (subset.seed.astype(int) == seed)].iloc[0]
                    xpos.append(cursor)
                    labels.append(f"{alternative[:1].upper()}\n{seed}")
                    bottom = 0.0
                    for quadrant in QUADRANT_ORDER:
                        value = row[f"fraction_{quadrant}"]
                        ax.bar(cursor, value, bottom=bottom, width=0.72, color=QUADRANT_COLORS[quadrant], edgecolor="white", linewidth=0.3,
                               label=QUADRANT_LABELS[quadrant] if cursor == 0 else None)
                        bottom += value
                    cursor += 1
                means = subset[(subset.alternative == alternative) & (subset.summary_level == "dataset_3seed_mean")]
                # The mean row is stored separately in transition_summary; find it by summary level.
                means = transitions[(transitions.dataset == dataset) & (transitions.modality == modality)
                                    & (transitions.alternative == alternative) & (transitions.summary_level == "dataset_3seed_mean")]
                if len(means):
                    row = means.iloc[0]
                    xpos.append(cursor)
                    labels.append(f"{alternative[:1].upper()}\nmean")
                    bottom = 0.0
                    for quadrant in QUADRANT_ORDER:
                        value = row[f"fraction_{quadrant}"]
                        ax.bar(cursor, value, bottom=bottom, width=0.72, color=QUADRANT_COLORS[quadrant], edgecolor="black", linewidth=0.35,
                               label=QUADRANT_LABELS[quadrant] if cursor == 0 else None)
                        bottom += value
                    cursor += 1
                cursor += 0.6
            ax.set_title(f"{dataset} · {modality}", loc="left", fontsize=9)
            ax.set_xticks(xpos, labels, fontsize=7)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color="#dddddd", linewidth=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Fraction of aligned edges")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.18, hspace=0.36, wspace=0.08)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.035), fontsize=8)
    fig.suptitle("Same-model sign transitions across fixed propagation functions", y=0.98, fontsize=12)
    align = qa_dir / "p13_sign_transition.alignment.json"
    require_matplotlib_panel_alignment(fig, json_out=str(align), tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    _save_figure(fig, figure_dir / "p13_sign_transition.png", qa_dir, align)

    # Figure 2: positive-best distribution for every seed and equal-weight mean.
    pref_seed = preferences[preferences.summary_level == "seed"]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.6), sharey=True)
    for i, dataset in enumerate(DATASETS):
        for j, modality in enumerate(MODALITIES):
            ax = axes[i, j]
            rows = pref_seed[(pref_seed.dataset == dataset) & (pref_seed.modality == modality)]
            xpos, labels = [], []
            cursor = 0
            for seed in SEEDS:
                row = rows[rows.seed.astype(int) == seed].iloc[0]
                bottom = 0.0
                for op in PREF_ORDER:
                    value = row[f"fraction_{op}"]
                    ax.bar(cursor, value, bottom=bottom, width=0.7, color=OP_COLORS[op], edgecolor="white", linewidth=0.3,
                           label=op if cursor == 0 else None)
                    bottom += value
                xpos.append(cursor); labels.append(str(seed)); cursor += 1
            meanrow = preferences[(preferences.dataset == dataset) & (preferences.modality == modality)
                                  & (preferences.summary_level == "dataset_3seed_mean")].iloc[0]
            bottom = 0.0
            for op in PREF_ORDER:
                value = meanrow[f"fraction_{op}"]
                ax.bar(cursor, value, bottom=bottom, width=0.7, color=OP_COLORS[op], edgecolor="black", linewidth=0.35,
                       label=op if cursor == 0 else None)
                bottom += value
            xpos.append(cursor); labels.append("mean")
            ax.set_title(f"{dataset} · {modality}", loc="left", fontsize=9)
            ax.set_xticks(xpos, labels, fontsize=7)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color="#dddddd", linewidth=0.5); ax.set_axisbelow(True)
            if j == 0: ax.set_ylabel("Fraction of edges")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.17, hspace=0.34, wspace=0.08)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.035), fontsize=8)
    fig.suptitle("Positive-best channel by physical edge", y=0.98, fontsize=12)
    align = qa_dir / "p13_preferred_channel.alignment.json"
    require_matplotlib_panel_alignment(fig, json_out=str(align), tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    _save_figure(fig, figure_dir / "p13_preferred_channel.png", qa_dir, align)

    # Figure 3: 4×4 matched-edge text/visual preferred-channel matrices by dataset.
    seed_mat = modality_preference[modality_preference.summary_level == "seed"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        subset = seed_mat[seed_mat.dataset == dataset]
        matrix = np.zeros((4, 4), dtype=float)
        for r, text_op in enumerate(PREF_ORDER):
            for c, vis_op in enumerate(PREF_ORDER):
                vals = subset[(subset.text_preference == text_op) & (subset.visual_preference == vis_op)].cell_fraction.to_numpy(float)
                matrix[r, c] = np.mean(vals)
        image = ax.imshow(matrix, vmin=0, vmax=max(0.5, float(np.max(matrix))), cmap="Blues", aspect="equal")
        for r in range(4):
            for c in range(4):
                color = "white" if matrix[r, c] > 0.3 else "#222222"
                ax.text(c, r, f"{matrix[r,c]:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(4), PREF_ORDER, rotation=35, rotation_mode="anchor", ha="right")
        ax.set_yticks(range(4), PREF_ORDER)
        ax.set_title(dataset, loc="left", fontsize=9)
        ax.set_xlabel("Visual preferred channel")
        if i == 0: ax.set_ylabel("Text preferred channel")
    fig.colorbar(image, ax=axes, shrink=0.82, label="Mean matched-edge fraction")
    fig.suptitle("Text–visual preferred-function coupling", y=1.03, fontsize=12)
    fig.text(0.5, -0.03, "Each cell averages within-seed edge fractions over seeds 42–44; rows and columns include none.", ha="center", fontsize=8)
    align = qa_dir / "p13_modality_preference_matrix.alignment.json"
    require_matplotlib_panel_alignment(fig, json_out=str(align), tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    _save_figure(fig, figure_dir / "p13_modality_preference_matrix.png", qa_dir, align)

    # Figure 4: paired P1.2/P1.3 transitions, displayed by dataset and metric family.
    seed_comp = comparison[comparison.summary_level == "seed"]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.5), sharey="row")
    for col, dataset in enumerate(DATASETS):
        for row_index, metric in enumerate(("S_negative_alt_positive_joint_prevalence", "alt_positive_given_S_negative")):
            ax = axes[row_index, col]
            subset = seed_comp[(seed_comp.dataset == dataset) & (seed_comp.comparison_type == "transition") & (seed_comp.metric == metric)]
            xloc = {"p12": 0, "p13": 1}
            group_id = 0
            for modality in MODALITIES:
                for operator in ("absdiff", "product"):
                    run_rows = subset[(subset.modality == modality) & (subset.operator == operator)]
                    p12vals, p13vals = [], []
                    offset = (group_id - 1.5) * 0.045
                    for seed in SEEDS:
                        one = run_rows[run_rows.seed.astype(int) == seed].iloc[0]
                        p12vals.append(float(one.p12_value)); p13vals.append(float(one.p13_value))
                        ax.plot([0 + offset, 1 + offset], [one.p12_value, one.p13_value],
                                color=OP_COLORS[operator], alpha=0.18, linewidth=0.8)
                    means = [float(np.mean(p12vals)), float(np.mean(p13vals))]
                    stds = [float(np.std(p12vals, ddof=0)), float(np.std(p13vals, ddof=0))]
                    ax.errorbar([0+offset, 1+offset], means, yerr=stds, color=OP_COLORS[operator], marker="o" if modality == "text" else "s",
                                linestyle="-" if modality == "text" else "--", linewidth=1.2, markersize=3.5, capsize=2,
                                label=f"{modality} · {operator}" if col == 0 and row_index == 0 else None)
                    group_id += 1
            ax.set_xticks([0, 1], ["P1.2", "P1.3"])
            ax.set_xlim(-0.22, 1.22)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color="#dddddd", linewidth=0.5); ax.set_axisbelow(True)
            if row_index == 0: ax.set_title(dataset, loc="left", fontsize=9)
            if col == 0: ax.set_ylabel("Joint prevalence" if row_index == 0 else "Conditional transition")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.20, hspace=0.30, wspace=0.14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.065), fontsize=7.5)
    fig.suptitle("Independent-head to joint-head transition comparison", y=0.98, fontsize=12)
    fig.text(0.5, 0.015, "Lines connect matched seeds; markers show 3-seed means with population SD. This compares diagnostics, not model performance.", ha="center", fontsize=8)
    align = qa_dir / "p13_p12_vs_p13.alignment.json"
    require_matplotlib_panel_alignment(fig, json_out=str(align), tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    _save_figure(fig, figure_dir / "p13_p12_vs_p13.png", qa_dir, align)


def _load_runs(output_dir: Path, datasets: list[str], seeds: list[int]) -> tuple[dict[tuple[str, int], pd.DataFrame], list[dict[str, Any]]]:
    edge_runs, records = {}, []
    for dataset in datasets:
        for seed in seeds:
            run_dir = output_dir / "runs" / dataset / f"seed_{seed}"
            result_path = run_dir / "run_result.json"
            edge_path = run_dir / "joint_operator_edge_utility.csv.gz"
            if not result_path.is_file() or not edge_path.is_file():
                raise FileNotFoundError(f"incomplete P1.3 run: {run_dir}")
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if record.get("status") != "completed" or record.get("test_evaluation") is not False:
                raise AssertionError(f"run status/test guard failed: {result_path}")
            frame = pd.read_csv(edge_path)
            if set(frame.dataset.astype(str)) != {dataset} or set(frame.seed.astype(int)) != {seed}:
                raise AssertionError(f"edge table dataset/seed mismatch: {edge_path}")
            if any((frame[f"ce_removed_{op}_{mod}"] - frame.ce_full - frame[f"u_raw_{op}_{mod}"]).abs().max() > 2e-10
                   for op in OPERATORS for mod in MODALITIES):
                raise AssertionError(f"raw utility does not equal removed CE minus shared full CE: {edge_path}")
            edge_runs[(dataset, seed)] = frame
            records.append(record)
    return edge_runs, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize P1.3 joint-readout operator counterfactuals")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p13_joint_readout_operator_probe"))
    parser.add_argument("--research-dir", type=Path, default=Path("research/p13_joint_readout_operator_probe"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--sample-per-run", type=int, default=100)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    data_dir = args.research_dir / "data"
    figure_dir = args.research_dir / "figures"
    qa_dir = args.output_dir / "figure_qa"
    data_dir.mkdir(parents=True, exist_ok=True)
    edge_runs, records = _load_runs(args.output_dir, args.datasets, args.seeds)
    utility, transitions = build_utility_and_transition(edge_runs)
    preferences, diversity, modality, margins = build_preferences(edge_runs)
    usage, ablation = build_channel_audits(records)
    sign_disagreement = build_modality_sign_disagreement()
    comparison = build_p12_comparison(transitions, sign_disagreement)
    sample = build_edge_sample(edge_runs, args.sample_per_run)
    tables = {
        "joint_probe_performance.csv": pd.DataFrame([r["head_performance"] for r in records]),
        "joint_operator_utility_summary.csv": utility,
        "joint_sign_transition.csv": transitions,
        "preferred_channel_summary.csv": preferences,
        "within_node_function_diversity.csv": diversity,
        "modality_preference_disagreement.csv": modality,
        "operator_modality_sign_disagreement.csv": sign_disagreement,
        "preference_margin_summary.csv": margins,
        "channel_usage_summary.csv": usage,
        "whole_channel_ablation.csv": ablation,
        "p12_vs_p13_transition_comparison.csv": comparison,
        "edge_sample.csv": sample,
    }
    for name, frame in tables.items():
        frame.to_csv(data_dir / name, index=False)
    if not args.skip_figures:
        render_figures(transitions, preferences, modality, comparison, figure_dir, qa_dir)
    manifest_path = args.research_dir / "run_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {
            "stage": "P1.3 — Joint-Readout Operator Counterfactual",
            "source_branch": "exp/p11_p12_operator_rescue",
            "source_sha": "17ca9420b614998f0192dac558c1a31d4c28a32e",
            "branch": "exp/p13_joint_readout_operator_probe",
            "datasets": args.datasets,
            "seeds": args.seeds,
            "completed_dataset_seed_runs": len(records),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available",
            "python": sys.version,
            "torch": __import__("torch").__version__,
            "pyg": __import__("torch_geometric").__version__,
            "validation_only": True,
            "test_evaluation": False,
            "link_prediction": False,
            "run_records": [f"{r['dataset']}/seed_{r['seed']}" for r in records],
            "files": list(tables),
        })
    print(json.dumps({"runs": len(records), "tables": {k: len(v) for k, v in tables.items()}, "figures": not args.skip_figures}, indent=2))


if __name__ == "__main__":
    main()
