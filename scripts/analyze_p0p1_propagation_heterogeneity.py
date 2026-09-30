from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.p0p1_propagation_probe import DATASETS, deterministic_edge_sample  # noqa: E402


MODALITIES = {
    "text": ("utility_text_ce", "raw_text_cosine"),
    "visual": ("utility_visual_ce", "raw_visual_cosine"),
}
COLORS = {"Movies": "#4277b8", "Grocery": "#e18932", "ele-fashion": "#4b9b72"}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _describe(values: pd.Series | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {key: float("nan") for key in ("mean", "median", "std", "q25", "q75", "iqr", "positive_fraction", "negative_fraction")}
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "positive_fraction": float(np.mean(x > 0)),
        "negative_fraction": float(np.mean(x < 0)),
    }


def _spearman(x: pd.Series, y: pd.Series) -> float:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if valid.sum() < 3 or np.unique(x.to_numpy()[valid]).size < 2 or np.unique(y.to_numpy()[valid]).size < 2:
        return float("nan")
    return float(spearmanr(x.to_numpy()[valid], y.to_numpy()[valid]).statistic)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if valid.sum() < 3 or np.unique(x.to_numpy()[valid]).size < 2 or np.unique(y.to_numpy()[valid]).size < 2:
        return float("nan")
    return float(pearsonr(x.to_numpy()[valid], y.to_numpy()[valid]).statistic)


def _tail_sets(group: pd.DataFrame, left: str, right: str) -> tuple[float, float]:
    n = len(group)
    if not n:
        return float("nan"), float("nan")
    k = max(1, int(np.ceil(0.2 * n)))
    left_values = group[left].to_numpy(dtype=float)
    right_values = group[right].to_numpy(dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    ids = np.flatnonzero(valid)
    if not ids.size:
        return float("nan"), float("nan")
    k = min(max(1, int(np.ceil(0.2 * ids.size))), ids.size)
    lorder = ids[np.argsort(left_values[ids], kind="stable")]
    rorder = ids[np.argsort(right_values[ids], kind="stable")]

    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        aset, bset = set(a.tolist()), set(b.tolist())
        return len(aset & bset) / max(len(aset | bset), 1)

    return jaccard(lorder[-k:], rorder[-k:]), jaccard(lorder[:k], rorder[:k])


def _utility_summaries(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows: list[dict] = []
    node_rows: list[dict] = []
    node_metric_rows: list[dict] = []
    for (dataset, seed), run in edges.groupby(["dataset", "seed"], sort=True):
        for modality, (utility_col, _) in MODALITIES.items():
            utility = run[utility_col].astype(float)
            global_rows.append(
                {"dataset": dataset, "seed": int(seed), "modality": modality, "edge_count": int(utility.notna().sum()), **_describe(utility)}
            )
            per_node: list[dict] = []
            for dst, group in run.groupby("dst", sort=True):
                values = group[utility_col].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
                per_node.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "dst": int(dst),
                        "modality": modality,
                        "dst_degree": int(group["dst_degree"].iloc[0]),
                        "edge_count": int(values.size),
                        "utility_iqr": float(q75 - q25),
                        "utility_q90_q10": float(q90 - q10),
                        "mixed_sign": bool((values > 0).any() and (values < 0).any()),
                    }
                )
            node_rows.extend(per_node)
            node_df = pd.DataFrame(per_node)
            node_metric_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "modality": modality,
                    "target_node_count": int(len(node_df)),
                    "median_node_iqr": float(node_df["utility_iqr"].median()),
                    "median_node_q90_q10": float(node_df["utility_q90_q10"].median()),
                    "mixed_sign_neighborhood_ratio": float(node_df["mixed_sign"].mean()),
                }
            )
    return pd.DataFrame(global_rows), pd.DataFrame(node_metric_rows), pd.DataFrame(node_rows)


def _modality_summaries(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h2_rows: list[dict] = []
    h3_rows: list[dict] = []
    bin_rows: list[dict] = []
    for (dataset, seed), run in edges.groupby(["dataset", "seed"], sort=True):
        nonzero = (run.utility_text_ce != 0) & (run.utility_visual_ce != 0)
        signs = run.loc[nonzero, "utility_text_ce"] * run.loc[nonzero, "utility_visual_ce"] < 0
        diffs = (run.utility_text_ce - run.utility_visual_ce).abs()
        eligible = run[run.dst_degree >= 5]
        top_j, bottom_j = [], []
        for _, group in eligible.groupby("dst", sort=True):
            high, low = _tail_sets(group, "utility_text_ce", "utility_visual_ce")
            top_j.append(high)
            bottom_j.append(low)
        h2_rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "directed_edge_count": int(len(run)),
                "spearman_text_visual_utility": _spearman(run.utility_text_ce, run.utility_visual_ce),
                "pearson_text_visual_utility": _pearson(run.utility_text_ce, run.utility_visual_ce),
                "sign_disagreement_ratio_nonzero_pairs": float(signs.mean()) if len(signs) else float("nan"),
                "zero_utility_pair_fraction": float((~nonzero).mean()),
                "abs_utility_diff_median": float(diffs.median()),
                "abs_utility_diff_iqr": float(diffs.quantile(0.75) - diffs.quantile(0.25)),
                "degree_ge5_target_nodes": int(eligible.dst.nunique()),
                "top20_jaccard_mean": float(np.mean(top_j)) if top_j else float("nan"),
                "top20_jaccard_median": float(np.median(top_j)) if top_j else float("nan"),
                "bottom20_jaccard_mean": float(np.mean(bottom_j)) if bottom_j else float("nan"),
                "bottom20_jaccard_median": float(np.median(bottom_j)) if bottom_j else float("nan"),
            }
        )

        for modality, (utility_col, similarity_col) in MODALITIES.items():
            sim = run[similarity_col].astype(float)
            utility = run[utility_col].astype(float)
            eligible_top, eligible_bottom = [], []
            for _, group in eligible.groupby("dst", sort=True):
                high, low = _tail_sets(group, similarity_col, utility_col)
                eligible_top.append(high)
                eligible_bottom.append(low)
            labels = utility.to_numpy() > 0
            valid_auc = np.isfinite(sim.to_numpy())
            auc = float("nan")
            if valid_auc.sum() > 0 and np.unique(labels[valid_auc]).size == 2:
                auc = float(roc_auc_score(labels[valid_auc], sim.to_numpy()[valid_auc]))
            h3_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "modality": modality,
                    "edge_count": int(len(run)),
                    "spearman_raw_similarity_utility": _spearman(sim, utility),
                    "pearson_raw_similarity_utility": _pearson(sim, utility),
                    "auroc_raw_similarity_to_positive_utility": auc,
                    "degree_ge5_target_nodes": int(eligible.dst.nunique()),
                    "high_similarity_vs_high_utility_top20_jaccard_mean": float(np.mean(eligible_top)) if eligible_top else float("nan"),
                    "high_similarity_vs_high_utility_top20_jaccard_median": float(np.median(eligible_top)) if eligible_top else float("nan"),
                    "low_similarity_vs_low_utility_bottom20_jaccard_mean": float(np.mean(eligible_bottom)) if eligible_bottom else float("nan"),
                    "low_similarity_vs_low_utility_bottom20_jaccard_median": float(np.median(eligible_bottom)) if eligible_bottom else float("nan"),
                }
            )

            valid = run[[similarity_col, utility_col]].dropna()
            if len(valid) >= 10 and valid[similarity_col].nunique() >= 2:
                try:
                    bin_ids = pd.qcut(valid[similarity_col], q=10, labels=False, duplicates="drop")
                except ValueError:
                    bin_ids = None
                if bin_ids is not None:
                    valid = valid.assign(_bin=np.asarray(bin_ids))
                    for bin_id, group in valid.groupby("_bin", sort=True):
                        desc = _describe(group[utility_col])
                        bin_rows.append(
                            {
                                "dataset": dataset,
                                "seed": int(seed),
                                "modality": modality,
                                "similarity_bin": int(bin_id),
                                "edge_count": int(len(group)),
                                "similarity_min": float(group[similarity_col].min()),
                                "similarity_max": float(group[similarity_col].max()),
                                "median_utility": desc["median"],
                                "utility_iqr": desc["iqr"],
                                "positive_utility_ratio": desc["positive_fraction"],
                                "negative_utility_ratio": desc["negative_fraction"],
                            }
                        )
    return pd.DataFrame(h2_rows), pd.DataFrame(h3_rows), pd.DataFrame(bin_rows)


def _margin_sanity(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, seed), run in edges.groupby(["dataset", "seed"], sort=True):
        for modality in ("text", "visual"):
            ce = run[f"utility_{modality}_ce"].astype(float)
            margin = run[f"utility_{modality}_margin"].astype(float)
            nonzero = (ce != 0) & (margin != 0)
            same_sign = (ce[nonzero] > 0) == (margin[nonzero] > 0)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "modality": modality,
                    "edge_count": int(len(run)),
                    "nonzero_pair_count": int(nonzero.sum()),
                    "ce_margin_sign_agreement_nonzero": float(same_sign.mean()) if len(same_sign) else float("nan"),
                    "ce_margin_spearman": _spearman(ce, margin),
                    "zero_ce_fraction": float((ce == 0).mean()),
                    "zero_margin_fraction": float((margin == 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _load_runs(output_root: Path, expected_datasets: list[str], expected_seeds: list[int]):
    result_files = sorted((output_root / "runs").glob("*/seed_*/run_result.json"))
    expected = {(d, s) for d in expected_datasets for s in expected_seeds}
    loaded: dict[tuple[str, int], dict] = {}
    edge_frames, group_frames, performance = [], [], []
    for path in result_files:
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (item["dataset"], int(item["seed"]))
        if key not in expected:
            continue
        loaded[key] = item
        edge_path = path.parent / "validation_message_evidence.csv.gz"
        group_path = path.parent / "group_interventions.csv"
        if not edge_path.is_file() or not group_path.is_file():
            raise FileNotFoundError(f"Completed run is missing an output file: {path.parent}")
        edge_frames.append(pd.read_csv(edge_path))
        group_frames.append(pd.read_csv(group_path))
        performance.extend(item["probe_performance"])
    missing = expected - loaded.keys()
    if missing:
        raise RuntimeError(f"Campaign incomplete; missing runs: {sorted(missing)}")
    return pd.concat(edge_frames, ignore_index=True), pd.concat(group_frames, ignore_index=True), pd.DataFrame(performance), loaded


def _group_summary(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "seed", "modality", "policy"]
    return (
        raw.groupby(keys, as_index=False)
        .agg(
            intervention_rows=("delta_val_ce", "count"),
            random_repeats=("repeat", "count"),
            masked_messages=("masked_messages", "mean"),
            delta_val_ce_mean=("delta_val_ce", "mean"),
            delta_val_ce_std=("delta_val_ce", "std"),
            delta_accuracy_mean=("delta_accuracy", "mean"),
            delta_accuracy_std=("delta_accuracy", "std"),
            delta_macro_f1_mean=("delta_macro_f1", "mean"),
            delta_macro_f1_std=("delta_macro_f1", "std"),
            baseline_ce=("baseline_ce", "first"),
            baseline_accuracy=("baseline_accuracy", "first"),
            baseline_macro_f1=("baseline_macro_f1", "first"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _make_figures(
    edge_sample: pd.DataFrame,
    node_rows: pd.DataFrame,
    group_summary: pd.DataFrame,
    figures_dir: Path,
    seed: int,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, modality in zip(axes, ("text", "visual")):
        utility_col, sim_col = MODALITIES[modality]
        plot = deterministic_edge_sample(edge_sample, 5000, seed + (17 if modality == "text" else 29))
        for dataset in DATASETS:
            points = plot[plot.dataset == dataset]
            if points.empty:
                continue
            axis.scatter(points[sim_col], points[utility_col], s=8, alpha=0.2, color=COLORS[dataset], label=dataset, rasterized=True)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xlabel(f"Raw {modality.title()} cosine")
        axis.set_ylabel(f"Relative CE utility ({modality.title()})")
        axis.set_title(f"{modality.title()} message")
        axis.legend(frameon=False, markerscale=1.8)
    fig.suptitle("Raw semantic similarity and counterfactual message utility")
    fig.savefig(figures_dir / "similarity_vs_utility.png", dpi=180)
    plt.close(fig)

    modalities = ["text", "visual"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels, series, colors = [], [], []
    for dataset in DATASETS:
        for modality in modalities:
            values = node_rows.loc[
                (node_rows.dataset == dataset) & (node_rows.modality == modality), "utility_iqr"
            ].to_numpy()
            series.append(values)
            labels.append(f"{dataset}\n{modality.title()}")
            colors.append("#477ab3" if modality == "text" else "#d48a3b")
    boxes = axes[0].boxplot(series, patch_artist=True, showfliers=False, medianprops={"color": "black"})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axes[0].set_xticks(range(1, len(labels) + 1), labels, rotation=0)
    axes[0].set_ylabel("Within-node utility IQR")
    axes[0].set_title("Per-target spread")
    mixed = node_rows.groupby(["dataset", "seed", "modality"], as_index=False)["mixed_sign"].mean()
    for modality, color, marker in (("text", "#477ab3", "o"), ("visual", "#d48a3b", "s")):
        subset = mixed[mixed.modality == modality]
        for dataset in DATASETS:
            points = subset[subset.dataset == dataset]
            axes[1].scatter(points.seed, points.mixed_sign, color=color, marker=marker, s=45, alpha=0.8, label=f"{dataset} {modality}")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Mixed-sign neighborhood ratio")
    axes[1].set_title("Same target has helpful and harmful messages")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Neighborhood-level edge utility heterogeneity")
    fig.savefig(figures_dir / "neighborhood_utility_heterogeneity.png", dpi=180)
    plt.close(fig)

    plot = deterministic_edge_sample(edge_sample, 10000, seed + 47)
    fig, axis = plt.subplots(figsize=(6.3, 6.0), constrained_layout=True)
    signs = np.where(
        (plot.utility_text_ce > 0) & (plot.utility_visual_ce > 0), 0,
        np.where((plot.utility_text_ce > 0) & (plot.utility_visual_ce < 0), 1,
        np.where((plot.utility_text_ce < 0) & (plot.utility_visual_ce > 0), 2, 3)),
    )
    quadrant_colors = np.array(["#4d9b70", "#ce6658", "#557fb4", "#888888"])
    axis.scatter(plot.utility_text_ce, plot.utility_visual_ce, c=quadrant_colors[signs], s=9, alpha=0.28, rasterized=True)
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.axvline(0, color="#333333", linewidth=0.9)
    axis.set_xlabel("Text utility (relative CE change)")
    axis.set_ylabel("Visual utility (relative CE change)")
    axis.set_title(f"Same directed edge, deterministic sample n={len(plot):,}")
    for x, y, label in ((0.04, 0.96, "(-,+)"), (0.68, 0.96, "(+,+)"), (0.04, 0.04, "(-,-)"), (0.68, 0.04, "(+,-)")):
        axis.text(x, y, label, transform=axis.transAxes, fontsize=10, color="#333333")
    fig.savefig(figures_dir / "text_visual_same_edge_utility.png", dpi=180)
    plt.close(fig)

    display = group_summary.copy()
    policy_order = ["bottom_utility_20pct", "random_control", "top_utility_20pct"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    for axis, metric, title, ylabel in (
        (axes[0], "delta_val_ce_mean", "Validation CE", "Δ CE (lower is better)"),
        (axes[1], "delta_accuracy_mean", "Validation accuracy", "Δ accuracy"),
    ):
        x = np.arange(len(policy_order))
        width = 0.34
        for offset, modality, color in ((-width / 2, "text", "#477ab3"), (width / 2, "visual", "#d48a3b")):
            means, errors = [], []
            for policy in policy_order:
                values = display.loc[(display.modality == modality) & (display.policy == policy), metric]
                means.append(float(values.mean()) if len(values) else np.nan)
                errors.append(float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0)
            axis.bar(x + offset, means, width, yerr=errors, capsize=3, color=color, alpha=0.8, label=modality.title())
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, ["Remove bottom\nutility 20%", "Random\nmatched count", "Remove top\nutility 20%"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(frameon=False)
    fig.suptitle("Group message-removal intervention (mean across dataset × seed runs)")
    fig.savefig(figures_dir / "group_intervention.png", dpi=180)
    plt.close(fig)


def analyze(output_root: Path, research_root: Path, datasets: list[str], seeds: list[int]) -> dict:
    edges, group_raw, performance, run_records = _load_runs(output_root, datasets, seeds)
    research_root.mkdir(parents=True, exist_ok=True)
    data_dir = research_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = research_root / "figures"

    utility, node_summary, node_rows = _utility_summaries(edges)
    h2, h3, bins = _modality_summaries(edges)
    margin_sanity = _margin_sanity(edges)
    intervention = _group_summary(group_raw)
    sample_parts = []
    for (dataset, seed), run in edges.groupby(["dataset", "seed"], sort=True):
        stable_dataset_code = sum((index + 1) * ord(char) for index, char in enumerate(dataset))
        sampled = deterministic_edge_sample(run, 8000, int(seed) * 1009 + stable_dataset_code)
        sample_parts.append(sampled)
    edge_sample = pd.concat(sample_parts, ignore_index=True)

    performance.sort_values(["dataset", "seed", "probe"]).to_csv(research_root / "probe_performance.csv", index=False)
    utility.to_csv(research_root / "utility_summary_by_seed.csv", index=False)
    node_summary.to_csv(research_root / "node_heterogeneity_summary.csv", index=False)
    h2.to_csv(research_root / "modality_disagreement_summary.csv", index=False)
    h3.to_csv(research_root / "similarity_utility_summary.csv", index=False)
    bins.to_csv(research_root / "similarity_utility_bins.csv", index=False)
    margin_sanity.to_csv(research_root / "margin_utility_sanity_summary.csv", index=False)
    intervention.to_csv(research_root / "group_intervention_summary.csv", index=False)
    edge_sample.to_csv(research_root / "edge_sample.csv", index=False)
    node_rows.to_csv(data_dir / "node_heterogeneity_per_target.csv", index=False)
    group_raw.to_csv(data_dir / "group_intervention_repeats.csv", index=False)
    _make_figures(edge_sample, node_rows, intervention, figures_dir, seed=20260930)

    summary = {
        "datasets": datasets,
        "seeds": seeds,
        "completed_runs": [
            {"dataset": d, "seed": int(s), "messages": int(run_records[(d, int(s))]["validation_target_messages"])}
            for d in datasets for s in seeds
        ],
        "probe_performance": performance.to_dict(orient="records"),
        "utility_summary_by_seed": utility.to_dict(orient="records"),
        "node_heterogeneity_summary": node_summary.to_dict(orient="records"),
        "modality_disagreement_summary": h2.to_dict(orient="records"),
        "similarity_utility_summary": h3.to_dict(orient="records"),
        "margin_utility_sanity_summary": margin_sanity.to_dict(orient="records"),
        "group_intervention_summary": intervention.to_dict(orient="records"),
        "test_evaluation": False,
    }
    (data_dir / "analysis_facts.json").write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate P0/P1 propagation utility evidence")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p0p1_propagation_heterogeneity"))
    parser.add_argument("--research-dir", type=Path, default=Path("research/p0p1_propagation_heterogeneity"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    summary = analyze(args.output_dir, args.research_dir, args.datasets, args.seeds)
    print(f"completed_runs={len(summary['completed_runs'])}")
    print(f"edge_sample_rows={len(pd.read_csv(args.research_dir / 'edge_sample.csv'))}")
    print(f"research_dir={args.research_dir.resolve()}")


if __name__ == "__main__":
    main()
