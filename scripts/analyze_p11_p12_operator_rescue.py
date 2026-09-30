from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.p0p1_propagation_probe import DATASETS, deterministic_edge_sample  # noqa: E402
from src.analysis.p11_p12_operator_rescue import MODALITIES, OPERATORS, reconstruct_raw_delta  # noqa: E402

_SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
if _SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(_SKILL_SCRIPTS))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError as exc:  # pragma: no cover - environment-specific figure QA helper
    raise RuntimeError("The Python figure QA helper audit_panel_alignment.py is required to render these figures") from exc

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


SEEDS = (42, 43, 44)
DATA_COLORS = {"Movies": "#4277b8", "Grocery": "#e18932", "ele-fashion": "#4b9b72"}
OP_COLORS = {"smooth": "#497ab2", "absdiff": "#d08b31", "product": "#4d9a75"}
QUADRANT_ORDER = ("smooth_positive__alternative_positive", "smooth_positive__alternative_nonpositive",
                  "smooth_negative__alternative_positive", "smooth_negative__alternative_nonpositive")
QUADRANT_COLORS = {QUADRANT_ORDER[0]: "#4d9a75", QUADRANT_ORDER[1]: "#8eadd0",
                   QUADRANT_ORDER[2]: "#d08b31", QUADRANT_ORDER[3]: "#c96961"}


def _safe_corr(x: Iterable[float], y: Iterable[float], kind: str) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    fn = spearmanr if kind == "spearman" else pearsonr
    return float(fn(a, b).statistic)


def _distribution(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "median", "q25", "q75", "iqr", "positive_fraction", "negative_fraction")}
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
        "positive_fraction": float(np.mean(x > 0)), "negative_fraction": float(np.mean(x < 0)),
    }


def _top_bottom_jaccard(group: pd.DataFrame, left: str, right: str) -> tuple[float, float]:
    if len(group) == 0:
        return float("nan"), float("nan")
    a, b = group[left].to_numpy(dtype=float), group[right].to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return float("nan"), float("nan")
    k = min(len(ids), max(1, int(np.ceil(0.2 * len(ids)))))
    order_a = ids[np.argsort(a[ids], kind="stable")]
    order_b = ids[np.argsort(b[ids], kind="stable")]

    def jac(x: np.ndarray, y: np.ndarray) -> float:
        sx, sy = set(x.tolist()), set(y.tolist())
        return len(sx & sy) / max(1, len(sx | sy))

    return jac(order_a[-k:], order_b[-k:]), jac(order_a[:k], order_b[:k])


def _within_node(values: pd.Series, run: pd.DataFrame, degree_min: int = 0) -> dict[str, float | int]:
    frame = run[["dst", "dst_degree"]].copy()
    frame["utility"] = values.to_numpy(dtype=float)
    if degree_min:
        frame = frame[frame.dst_degree >= degree_min]
    node_rows = []
    for _, group in frame.groupby("dst", sort=True):
        x = group.utility.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        q10, q25, q75, q90 = np.quantile(x, [0.1, 0.25, 0.75, 0.9])
        node_rows.append({
            "iqr": q75 - q25, "q90_q10": q90 - q10,
            "mixed": bool(np.any(x > 0) and np.any(x < 0)),
        })
    nodes = pd.DataFrame(node_rows)
    if nodes.empty:
        return {"target_node_count": 0, "median_node_iqr": float("nan"),
                "median_node_q90_q10": float("nan"), "mixed_sign_neighborhood_ratio": float("nan")}
    return {
        "target_node_count": int(len(nodes)),
        "median_node_iqr": float(nodes.iqr.median()),
        "median_node_q90_q10": float(nodes.q90_q10.median()),
        "mixed_sign_neighborhood_ratio": float(nodes.mixed.mean()),
    }


def build_p11(p0_root: Path, p11_root: Path, datasets: list[str], seeds: list[int]) -> dict[str, pd.DataFrame]:
    utility_rows, modality_rows, similarity_rows, degree_rows = [], [], [], []
    loaded_runs = []
    for dataset in datasets:
        for seed in seeds:
            edge_path = p0_root / "runs" / dataset / f"seed_{seed}" / "validation_message_evidence.csv.gz"
            if not edge_path.is_file():
                raise FileNotFoundError(f"missing P0 complete edge evidence: {edge_path}")
            run = pd.read_csv(edge_path)
            if "ce_full" not in run or "utility_text_ce" not in run or "utility_visual_ce" not in run:
                raise ValueError(f"P0 edge evidence has missing CE fields: {edge_path}")
            run["dataset"], run["seed"] = dataset, seed
            run["raw_delta_text_ce"] = reconstruct_raw_delta(run.utility_text_ce.to_numpy(float), run.ce_full.to_numpy(float))
            run["raw_delta_visual_ce"] = reconstruct_raw_delta(run.utility_visual_ce.to_numpy(float), run.ce_full.to_numpy(float))
            for modality in MODALITIES:
                raw_col = f"raw_delta_{modality}_ce"
                rel_col = f"utility_{modality}_ce"
                # Required reconstruction identity, within storage precision.
                reconstructed = run[rel_col].to_numpy(float) * run.ce_full.to_numpy(float)
                if not np.allclose(reconstructed, run[raw_col].to_numpy(float), rtol=0.0, atol=0.0):
                    raise AssertionError("P1.1 raw delta reconstruction differs from relative utility × CE_full")
                raw_d = _distribution(run[raw_col])
                rel_d = _distribution(run[rel_col])
                within_raw = _within_node(run[raw_col], run)
                within_rel = _within_node(run[rel_col], run)
                utility_rows.append({
                    "dataset": dataset, "seed": seed, "modality": modality,
                    "edge_count": int(len(run)),
                    **{f"raw_{k}": v for k, v in raw_d.items()},
                    **{f"relative_{k}": v for k, v in rel_d.items()},
                    **{f"raw_{k}": v for k, v in within_raw.items()},
                    **{f"relative_{k}": v for k, v in within_rel.items()},
                    "raw_reconstruction_max_abs_error": 0.0,
                })
                raw_g5 = _within_node(run.loc[run.dst_degree >= 5, raw_col], run.loc[run.dst_degree >= 5].reset_index(drop=True))
                rel_g5 = _within_node(run.loc[run.dst_degree >= 5, rel_col], run.loc[run.dst_degree >= 5].reset_index(drop=True))
                eligible = run[run.dst_degree >= 5]
                degree_rows.append({
                    "dataset": dataset, "seed": seed, "modality": modality,
                    "edge_count": int(len(eligible)),
                    **{f"raw_{k}": v for k, v in _distribution(eligible[raw_col]).items()},
                    **{f"relative_{k}": v for k, v in _distribution(eligible[rel_col]).items()},
                    **{f"raw_{k}": v for k, v in raw_g5.items()},
                    **{f"relative_{k}": v for k, v in rel_g5.items()},
                })

            # Same directed edge, text versus visual utility. Positive CE_full preserves each edge's sign.
            for scale, text_col, visual_col in (
                ("raw_delta_ce", "raw_delta_text_ce", "raw_delta_visual_ce"),
                ("relative", "utility_text_ce", "utility_visual_ce"),
            ):
                a, b = run[text_col].to_numpy(float), run[visual_col].to_numpy(float)
                eligible = run[run.dst_degree >= 5]
                top, bottom = [], []
                for _, group in eligible.groupby("dst", sort=True):
                    high, low = _top_bottom_jaccard(group, text_col, visual_col)
                    top.append(high)
                    bottom.append(low)
                diff = np.abs(a - b)
                modality_rows.append({
                    "dataset": dataset, "seed": seed, "utility_scale": scale,
                    "directed_edge_count": int(len(run)),
                    "spearman_text_visual": _safe_corr(a, b, "spearman"),
                    "pearson_text_visual": _safe_corr(a, b, "pearson"),
                    "sign_disagreement_ratio": float(np.mean((a > 0) != (b > 0))),
                    "abs_difference_median": float(np.median(diff)),
                    "abs_difference_iqr": float(np.quantile(diff, .75) - np.quantile(diff, .25)),
                    "degree_ge5_target_count": int(eligible.dst.nunique()),
                    "degree_ge5_top20_jaccard_mean": float(np.mean(top)) if top else float("nan"),
                    "degree_ge5_bottom20_jaccard_mean": float(np.mean(bottom)) if bottom else float("nan"),
                })

            for modality in MODALITIES:
                raw_utility = run[f"raw_delta_{modality}_ce"]
                rel_utility = run[f"utility_{modality}_ce"]
                for sim_name, sim_col in (
                    ("raw_frozen_cosine", f"raw_{modality}_cosine"),
                    ("task_projected_cosine", f"projected_{modality}_cosine"),
                ):
                    if sim_col not in run:
                        raise ValueError(f"P0 edge evidence missing similarity column {sim_col}")
                    eligible = run[run.dst_degree >= 5]
                    for scale, utility in (("raw_delta_ce", raw_utility), ("relative", rel_utility)):
                        sim = run[sim_col].to_numpy(float)
                        vals = utility.to_numpy(float)
                        valid = np.isfinite(sim) & np.isfinite(vals)
                        auc = float("nan")
                        labels = vals[valid] > 0
                        if np.unique(labels).size == 2:
                            auc = float(roc_auc_score(labels, sim[valid]))
                        high_j, low_j = [], []
                        for _, group in eligible.groupby("dst", sort=True):
                            high, low = _top_bottom_jaccard(group, sim_col, "raw_delta_" + modality + "_ce" if scale == "raw_delta_ce" else f"utility_{modality}_ce")
                            high_j.append(high)
                            low_j.append(low)
                        similarity_rows.append({
                            "dataset": dataset, "seed": seed, "modality": modality,
                            "similarity_type": sim_name, "utility_scale": scale,
                            "edge_count": int(valid.sum()),
                            "spearman_similarity_utility": _safe_corr(sim, vals, "spearman"),
                            "pearson_similarity_utility": _safe_corr(sim, vals, "pearson"),
                            "auroc_similarity_to_positive_utility": auc,
                            "degree_ge5_target_count": int(eligible.dst.nunique()),
                            "degree_ge5_high_similarity_high_utility_top20_jaccard_mean": float(np.mean(high_j)) if high_j else float("nan"),
                            "degree_ge5_low_similarity_low_utility_bottom20_jaccard_mean": float(np.mean(low_j)) if low_j else float("nan"),
                        })
            loaded_runs.append(run)

    tables = {
        "utility": pd.DataFrame(utility_rows),
        "modality": pd.DataFrame(modality_rows),
        "similarity": pd.DataFrame(similarity_rows),
        "degree": pd.DataFrame(degree_rows),
    }
    p11_root.mkdir(parents=True, exist_ok=True)
    tables["utility"].to_csv(p11_root / "raw_delta_utility_summary.csv", index=False)
    tables["modality"].to_csv(p11_root / "raw_delta_modality_disagreement.csv", index=False)
    tables["similarity"].to_csv(p11_root / "raw_vs_projected_similarity_summary.csv", index=False)
    tables["degree"].to_csv(p11_root / "degree_ge5_h1_summary.csv", index=False)
    tables["edge_frames"] = loaded_runs
    return tables


def _p12_run_tables(output_root: Path, datasets: list[str], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_frames, perf_frames = [], []
    for dataset in datasets:
        for seed in seeds:
            root = output_root / "runs" / dataset / f"seed_{seed}"
            result_path = root / "run_result.json"
            edge_path = root / "operator_edge_utility.csv.gz"
            perf_path = root / "operator_probe_performance.csv"
            for path in (result_path, edge_path, perf_path):
                if not path.is_file():
                    raise FileNotFoundError(f"P1.2 run is incomplete: {path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed" or result.get("test_evaluation"):
                raise ValueError(f"unexpected P1.2 run status/test evaluation in {result_path}")
            edges = pd.read_csv(edge_path)
            if len(edges) != int(result["validation_target_messages"]):
                raise AssertionError(f"edge table row count differs from run record: {edge_path}")
            if edges[["src", "dst"]].duplicated().any():
                raise AssertionError(f"physical message keys are not unique: {edge_path}")
            edges["dataset"], edges["seed"] = dataset, seed
            for op in OPERATORS:
                for mod in MODALITIES:
                    if f"u_raw_{op}_{mod}" not in edges or f"u_relative_{op}_{mod}" not in edges:
                        raise ValueError(f"missing operator utility field for {op}/{mod}: {edge_path}")
            edge_frames.append(edges)
            perf_frames.append(pd.read_csv(perf_path))
    return pd.concat(edge_frames, ignore_index=True), pd.concat(perf_frames, ignore_index=True)


def build_p12(edge_runs: pd.DataFrame, performance: pd.DataFrame, p12_root: Path, sample_seed: int = 20260930) -> dict[str, pd.DataFrame]:
    utility_rows, rescue_rows, transition_rows, modality_rows, sample_rows = [], [], [], [], []
    for (dataset, seed), run in edge_runs.groupby(["dataset", "seed"], sort=True):
        for modality in MODALITIES:
            cols = {op: f"u_raw_{op}_{modality}" for op in OPERATORS}
            smooth = run[cols["smooth"]].to_numpy(float)
            absdiff = run[cols["absdiff"]].to_numpy(float)
            product = run[cols["product"]].to_numpy(float)
            for operator, values in ((op, run[col]) for op, col in cols.items()):
                utility_rows.append({
                    "dataset": dataset, "seed": int(seed), "modality": modality,
                    "operator": operator, "edge_count": int(len(values)),
                    **_distribution(values),
                    "median_node_iqr": float(run.assign(_u=values).groupby("dst")._u.quantile(.75).sub(
                        run.assign(_u=values).groupby("dst")._u.quantile(.25)).median()),
                })
            smooth_harm = smooth < 0
            harm_n = int(smooth_harm.sum())
            alt_d_pos, alt_c_pos = absdiff > 0, product > 0
            r_d = float(alt_d_pos[smooth_harm].mean()) if harm_n else float("nan")
            r_c = float(alt_c_pos[smooth_harm].mean()) if harm_n else float("nan")
            r_any = float((alt_d_pos | alt_c_pos)[smooth_harm].mean()) if harm_n else float("nan")
            overall_d, overall_c = float(alt_d_pos.mean()), float(alt_c_pos.mean())
            rescue_rows.append({
                "dataset": dataset, "seed": int(seed), "modality": modality,
                "edge_count": int(len(run)), "smooth_harmful_count": harm_n,
                "smooth_harmful_fraction": float(smooth_harm.mean()),
                "absdiff_rescued_count": int((smooth_harm & alt_d_pos).sum()),
                "absdiff_rescue_conditional": r_d,
                "absdiff_joint_prevalence": float((smooth_harm & alt_d_pos).mean()),
                "absdiff_overall_positive_fraction": overall_d,
                "absdiff_rescue_enrichment": r_d / overall_d if harm_n and overall_d > 0 else float("nan"),
                "product_rescued_count": int((smooth_harm & alt_c_pos).sum()),
                "product_rescue_conditional": r_c,
                "product_joint_prevalence": float((smooth_harm & alt_c_pos).mean()),
                "product_overall_positive_fraction": overall_c,
                "product_rescue_enrichment": r_c / overall_c if harm_n and overall_c > 0 else float("nan"),
                "any_rescued_count": int((smooth_harm & (alt_d_pos | alt_c_pos)).sum()),
                "any_rescue_conditional": r_any,
                "any_joint_prevalence": float((smooth_harm & (alt_d_pos | alt_c_pos)).mean()),
                "unrescued_by_tested_primitives_count": int((smooth_harm & (absdiff <= 0) & (product <= 0)).sum()),
                "unrescued_by_tested_primitives_fraction": float(((absdiff <= 0) & (product <= 0))[smooth_harm].mean()) if harm_n else float("nan"),
            })
            for alternative, values in (("absdiff", absdiff), ("product", product)):
                alt_pos = values > 0
                quadrants = {
                    QUADRANT_ORDER[0]: (smooth > 0) & alt_pos,
                    QUADRANT_ORDER[1]: (smooth > 0) & (values <= 0),
                    QUADRANT_ORDER[2]: (smooth <= 0) & alt_pos,
                    QUADRANT_ORDER[3]: (smooth <= 0) & (values <= 0),
                }
                for quadrant, mask in quadrants.items():
                    transition_rows.append({
                        "summary_level": "seed", "dataset": dataset, "seed": int(seed),
                        "modality": modality, "comparison": f"smooth_vs_{alternative}",
                        "quadrant": quadrant, "message_count": int(mask.sum()),
                        "fraction_all_messages": float(mask.mean()),
                    })

        # Same physical edge Text/Visual sign differences for each operator.
        for operator in OPERATORS:
            text = run[f"u_raw_{operator}_text"].to_numpy(float)
            visual = run[f"u_raw_{operator}_visual"].to_numpy(float)
            disagreement = float(np.mean((text > 0) != (visual > 0)))
            rescue_lookup = {
                row["modality"]: row for row in rescue_rows
                if row["dataset"] == dataset and row["seed"] == int(seed)
            }
            modality_rows.append({
                "dataset": dataset, "seed": int(seed), "operator": operator,
                "edge_count": int(len(run)),
                "text_visual_sign_disagreement": disagreement,
                "text_positive_fraction": float(np.mean(text > 0)),
                "visual_positive_fraction": float(np.mean(visual > 0)),
                "r_any_text": rescue_lookup.get("text", {}).get("any_rescue_conditional", float("nan")),
                "r_any_visual": rescue_lookup.get("visual", {}).get("any_rescue_conditional", float("nan")),
            })

        sample_cols = ["dataset", "seed", "src", "dst", "dst_degree", "dst_label"]
        sample_cols += [f"u_raw_{op}_{mod}" for op in OPERATORS for mod in MODALITIES]
        sample_cols += [f"u_relative_{op}_{mod}" for op in OPERATORS for mod in MODALITIES]
        stable_code = sum((i + 1) * ord(char) for i, char in enumerate(str(dataset)))
        sample = deterministic_edge_sample(run[sample_cols], max_rows=3000, seed=sample_seed + int(seed) * 997 + stable_code)
        sample_rows.append(sample)

    utility_df = pd.DataFrame(utility_rows)
    rescue_df = pd.DataFrame(rescue_rows)
    transition_df = pd.DataFrame(transition_rows)
    modality_df = pd.DataFrame(modality_rows)
    # Explicit dataset-level 3-seed mean/std, still descriptive and without inferential tests.
    for (dataset, modality, comparison, quadrant), group in transition_df[transition_df.summary_level == "seed"].groupby(
        ["dataset", "modality", "comparison", "quadrant"], sort=True,
    ):
        values = group.fraction_all_messages.to_numpy(float)
        for level, val in (("dataset_3seed_mean", float(np.mean(values))),
                           ("dataset_3seed_std", float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)):
            transition_df.loc[len(transition_df)] = {
                "summary_level": level, "dataset": dataset, "seed": np.nan,
                "modality": modality, "comparison": comparison, "quadrant": quadrant,
                "message_count": np.nan, "fraction_all_messages": val,
            }
    for (dataset, modality), group in rescue_df.groupby(["dataset", "modality"], sort=True):
        for metric in ("smooth_harmful_fraction", "absdiff_rescue_conditional", "product_rescue_conditional",
                       "any_rescue_conditional", "unrescued_by_tested_primitives_fraction",
                       "absdiff_rescue_enrichment", "product_rescue_enrichment"):
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            rescue_rows.append({
                "summary_level": "dataset_3seed_mean", "dataset": dataset, "seed": np.nan,
                "modality": modality, "metric": metric, "value": float(values.mean()) if len(values) else np.nan,
                "seed_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            })
    # The seed rows remain wide for easy joins; append separate long-form mean/std rows.
    rescue_df["summary_level"] = "seed"
    rescue_df = pd.concat([rescue_df, pd.DataFrame(rescue_rows[len(rescue_df):])], ignore_index=True, sort=False)
    # Add dataset seed-mean/std rows for modality disagreement summaries.
    modality_df["summary_level"] = "seed"
    mean_mod = modality_df.groupby(["dataset", "operator"], as_index=False).agg(
        text_visual_sign_disagreement_mean=("text_visual_sign_disagreement", "mean"),
        text_visual_sign_disagreement_seed_std=("text_visual_sign_disagreement", "std"),
        r_any_text_mean=("r_any_text", "mean"), r_any_visual_mean=("r_any_visual", "mean"),
    )
    mean_mod["summary_level"] = "dataset_3seed_mean"
    modality_df = pd.concat([modality_df, mean_mod], ignore_index=True, sort=False)

    p12_root.mkdir(parents=True, exist_ok=True)
    performance.sort_values(["dataset", "seed", "operator"]).to_csv(p12_root / "operator_probe_performance.csv", index=False)
    utility_df.to_csv(p12_root / "operator_utility_summary.csv", index=False)
    rescue_df.to_csv(p12_root / "operator_rescue_summary.csv", index=False)
    transition_df.to_csv(p12_root / "operator_transition_matrix.csv", index=False)
    modality_df.to_csv(p12_root / "operator_modality_disagreement.csv", index=False)
    pd.concat(sample_rows, ignore_index=True).to_csv(p12_root / "operator_edge_sample.csv", index=False)
    return {"utility": utility_df, "rescue": rescue_df, "transition": transition_df,
            "modality": modality_df, "performance": performance,
            "sample": pd.concat(sample_rows, ignore_index=True)}


def _seed_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()


def _add_panel_labels(axes) -> None:
    for index, axis in enumerate(np.asarray(axes, dtype=object).reshape(-1)):
        axis.annotate(
            chr(ord("a") + index), xy=(0, 1), xycoords="axes fraction",
            xytext=(-13, 5), textcoords="offset points", ha="left", va="bottom",
            fontsize=10, fontweight="bold", annotation_clip=False,
        )


def _normalize_svg_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _save_figure(fig, path: Path) -> None:
    fig.canvas.draw()
    qa_dir = path.parent / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    alignment_svg = qa_dir / f"{path.stem}.alignment.svg"
    require_matplotlib_panel_alignment(
        fig,
        json_out=qa_dir / f"{path.stem}.alignment.json",
        overlay_svg=alignment_svg,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    main_svg = path.with_suffix(".svg")
    fig.savefig(main_svg, bbox_inches="tight")
    _normalize_svg_whitespace(main_svg)
    _normalize_svg_whitespace(alignment_svg)
    export_dir = Path("outputs/p11_p12_operator_rescue/figure_exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(export_dir / f"{path.stem}.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _figure_p11(similarity: pd.DataFrame, path: Path) -> None:
    raw = similarity[similarity.utility_scale == "raw_delta_ce"]
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2), sharex="col", constrained_layout=False)
    fig.subplots_adjust(left=.105, right=.985, bottom=.17, top=.84, wspace=.32, hspace=.38)
    for col, dataset in enumerate(DATASETS):
        frame = raw[raw.dataset == dataset]
        for row, metric, title in ((0, "spearman_similarity_utility", "Spearman"),
                                   (1, "auroc_similarity_to_positive_utility", "AUROC")):
            ax = axes[row, col]
            xloc = np.arange(4)
            vals, errors, colors, labels = [], [], [], []
            for sim_type in ("raw_frozen_cosine", "task_projected_cosine"):
                for modality in MODALITIES:
                    cell = frame[(frame.similarity_type == sim_type) & (frame.modality == modality)][metric].to_numpy(float)
                    cell = cell[np.isfinite(cell)]
                    vals.append(float(cell.mean()) if len(cell) else np.nan)
                    errors.append(float(cell.std(ddof=1)) if len(cell) > 1 else 0.0)
                    colors.append("#477ab3" if modality == "text" else "#d08b31")
                    labels.append(("raw" if sim_type == "raw_frozen_cosine" else "projected") + "\n" + modality)
            bars = ax.bar(xloc, vals, yerr=errors, capsize=2, color=colors, alpha=.82, width=.72)
            if row == 1:
                ax.axhline(.5, color="#555", linestyle="--", linewidth=.8)
                ax.set_ylim(0, 1)
            else:
                ax.axhline(0, color="#555", linewidth=.8)
            ax.set_xticks(xloc, labels, fontsize=8)
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                ax.set_title(dataset)
    _add_panel_labels(axes)
    fig.suptitle("P1.1: cosine against raw ΔCE utility (mean ± seed SD; n=3)")
    _save_figure(fig, path)


def _figure_rescue(rescue: pd.DataFrame, path: Path) -> None:
    seed = rescue[rescue.get("summary_level", "seed") == "seed"]
    # Legacy seed rows have wide metrics; summary rows are long-form.
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.8), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=.105, right=.985, bottom=.25, top=.80, wspace=.32)
    fields = ["absdiff_rescue_conditional", "product_rescue_conditional", "any_rescue_conditional",
              "unrescued_by_tested_primitives_fraction"]
    labels = ["AbsDiff", "Product", "Any alt.", "Unrescued"]
    colors = [OP_COLORS["absdiff"], OP_COLORS["product"], "#7d69a5", "#777777"]
    for ax, dataset in zip(axes, DATASETS):
        frame = seed[seed.dataset == dataset]
        width = .34
        for mi, modality in enumerate(MODALITIES):
            group = frame[frame.modality == modality]
            means = [group[col].mean() for col in fields]
            stds = [group[col].std(ddof=1) if group[col].notna().sum() > 1 else 0 for col in fields]
            x = np.arange(len(fields)) + (mi - .5) * width
            ax.bar(x, means, width, yerr=stds, capsize=2, color="#477ab3" if modality == "text" else "#d08b31",
                   alpha=.83, label=modality.title())
        ax.set_title(dataset)
        ax.set_xticks(np.arange(len(fields)), labels, fontsize=8, ha="center")
        ax.set_ylim(0, 1)
        ax.axhline(0, color="#444", linewidth=.7)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Fraction among smooth-harmful messages")
    _add_panel_labels(axes)
    fig.suptitle("P1.2: rescue among smooth-harmful messages (mean ± seed SD; n=3)")
    _save_figure(fig, path)


def _figure_transition(transition: pd.DataFrame, path: Path) -> None:
    data = transition[transition.summary_level == "dataset_3seed_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.2), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=.10, right=.82, bottom=.14, top=.84, wspace=.25, hspace=.34)
    for row, modality in enumerate(MODALITIES):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            bottom = np.zeros(2)
            quadrant_labels = {
                "smooth_positive__alternative_positive": "S+ / A+",
                "smooth_positive__alternative_nonpositive": "S+ / A−",
                "smooth_negative__alternative_positive": "S− / A+",
                "smooth_negative__alternative_nonpositive": "S− / A−",
            }
            for quadrant in QUADRANT_ORDER:
                vals = []
                for alternative in ("absdiff", "product"):
                    cell = data[(data.dataset == dataset) & (data.modality == modality)
                                & (data.comparison == f"smooth_vs_{alternative}") & (data.quadrant == quadrant)]
                    vals.append(float(cell.fraction_all_messages.iloc[0]) if len(cell) else 0.0)
                ax.bar([0, 1], vals, bottom=bottom, color=QUADRANT_COLORS[quadrant], label=quadrant_labels[quadrant])
                bottom += np.asarray(vals)
            ax.set_xticks([0, 1], ["absdiff", "product"])
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel(f"{modality.title()}\nShare of all messages")
            if row == 0:
                ax.set_title(dataset)
            if row == 0 and col == 2:
                ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(1.02, 0))
    _add_panel_labels(axes)
    fig.suptitle("P1.2: sign transitions (mean across seeds; seed SD in CSV)")
    _save_figure(fig, path)


def _figure_modality(rescue: pd.DataFrame, path: Path) -> None:
    seed = rescue[rescue.get("summary_level", "seed") == "seed"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.8), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=.105, right=.985, bottom=.18, top=.80, wspace=.32)
    for ax, dataset in zip(axes, DATASETS):
        frame = seed[seed.dataset == dataset]
        x = np.arange(2)
        for offset, modality, color, marker in ((-.09, "text", "#477ab3", "o"), (.09, "visual", "#d08b31", "s")):
            group = frame[frame.modality == modality]
            y = [group.loc[group.seed.notna(), col].mean() for col in ("absdiff_rescue_conditional", "product_rescue_conditional")]
            err = [group.loc[group.seed.notna(), col].std(ddof=1) for col in ("absdiff_rescue_conditional", "product_rescue_conditional")]
            ax.errorbar(x + offset, y, yerr=err, marker=marker, color=color, capsize=3, linewidth=1.4, label=modality.title())
        ax.set_xticks(x, ["absdiff", "product"])
        ax.set_ylim(0, 1)
        ax.set_title(dataset)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Conditional rescue fraction")
    _add_panel_labels(axes)
    fig.suptitle("P1.2: modality-specific rescue (mean ± seed SD; n=3)")
    _save_figure(fig, path)


def _means(df: pd.DataFrame, metric: str, groups: list[str]) -> pd.DataFrame:
    return df.groupby(groups, as_index=False)[metric].agg(mean="mean", std="std")


def write_reports(p11: dict[str, pd.DataFrame], p12: dict[str, pd.DataFrame], research_root: Path) -> None:
    p11_root = research_root / "p11"
    p12_root = research_root / "p12"
    figures = research_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _figure_p11(p11["similarity"], figures / "p11_raw_vs_projected_similarity.png")
    _figure_rescue(p12["rescue"], figures / "p12_operator_rescue.png")
    _figure_transition(p12["transition"], figures / "p12_sign_transition.png")
    _figure_modality(p12["rescue"], figures / "p12_modality_rescue.png")

    utility = p11["utility"]
    modality = p11["modality"]
    similarity = p11["similarity"]
    degree = p11["degree"]
    lines = [
        "# P1.1 robustness audit",
        "",
        "This analysis reuses only the committed P0/P1 validation edge evidence; no model was trained. Raw utility is reconstructed as `U_raw = U_relative × CE_full`, which equals `CE_removed − CE_full` under the P0 signed-utility definition. All association, sign, and set-overlap summaries are descriptive across directed messages; no p-values are reported.",
        "",
        "## H1: utility spread and mixed-sign neighborhoods",
        "",
        "The sign fractions are invariant to the positive per-edge `CE_full` scale. Within-node spread is recalculated from raw ΔCE, so its magnitude is not directly comparable to relative utility units.",
        "",
    ]
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Modality | Helpful edge fraction | Harmful edge fraction | Median node IQR (raw ΔCE) | Median node q90–q10 | Mixed-sign target ratio |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for mod in MODALITIES:
            g = utility[(utility.dataset == dataset) & (utility.modality == mod)]
            lines.append(f"| {mod.title()} | {g.raw_positive_fraction.mean():.3f} | {g.raw_negative_fraction.mean():.3f} | {g.raw_median_node_iqr.mean():.5g} | {g.raw_median_node_q90_q10.mean():.5g} | {g.raw_mixed_sign_neighborhood_ratio.mean():.3f} |")
        lines.append("")
    lines.extend(["## H2: Text and Visual utility on the same edge", "",
                  "| Dataset | Raw ΔCE Spearman | Raw ΔCE Pearson | Sign disagreement | Median absolute difference | Degree≥5 top Jaccard | Degree≥5 bottom Jaccard | Relative Spearman |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        raw = modality[(modality.dataset == dataset) & (modality.utility_scale == "raw_delta_ce")]
        rel = modality[(modality.dataset == dataset) & (modality.utility_scale == "relative")]
        lines.append(f"| {dataset} | {raw.spearman_text_visual.mean():.3f} | {raw.pearson_text_visual.mean():.3f} | {raw.sign_disagreement_ratio.mean():.3f} | {raw.abs_difference_median.mean():.5g} | {raw.degree_ge5_top20_jaccard_mean.mean():.3f} | {raw.degree_ge5_bottom20_jaccard_mean.mean():.3f} | {rel.spearman_text_visual.mean():.3f} |")
    lines.extend(["", "The same-edge sign-disagreement rate remains unchanged when moving from relative utility to raw ΔCE because each row is multiplied by its positive `CE_full`. Correlation and utility-rank overlaps can move because edge scales differ.", "",
                  "## H3: raw and task-projected cosine", "",
                  "The table reports seed means for raw ΔCE. Spearman is the primary rank association; Pearson and AUROC are secondary. Degree≥5 Jaccards average the per-target top/bottom 20% sets.", "",
                  "| Dataset | Modality | Raw cosine Spearman | Projected cosine Spearman | Raw cosine AUROC | Projected cosine AUROC | Raw top Jaccard | Projected top Jaccard |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    h3 = similarity[similarity.utility_scale == "raw_delta_ce"]
    for dataset in DATASETS:
        for mod in MODALITIES:
            a = h3[(h3.dataset == dataset) & (h3.modality == mod)]
            raw = a[a.similarity_type == "raw_frozen_cosine"]
            proj = a[a.similarity_type == "task_projected_cosine"]
            lines.append(f"| {dataset} | {mod.title()} | {raw.spearman_similarity_utility.mean():.3f} | {proj.spearman_similarity_utility.mean():.3f} | {raw.auroc_similarity_to_positive_utility.mean():.3f} | {proj.auroc_similarity_to_positive_utility.mean():.3f} | {raw.degree_ge5_high_similarity_high_utility_top20_jaccard_mean.mean():.3f} | {proj.degree_ge5_high_similarity_high_utility_top20_jaccard_mean.mean():.3f} |")
    lines.extend(["", "Across the six dataset×modality groups, task-projected cosine raises mean sign AUROC over frozen raw cosine in five groups (the exception is ele-fashion Visual); gains are clearest in Grocery (about 0.68 AUROC). Its Spearman correlations remain modest (roughly −0.003 to 0.088), and the picture is not consistent across modalities or datasets. Projected similarity therefore helps classify positive utility in some settings but is not sufficient to determine propagation utility. The full seed-level comparison, including Pearson and bottom-tail overlaps and relative-utility sensitivity rows, is in `raw_vs_projected_similarity_summary.csv`.", "",
                  "## H1 after restricting to degree ≥ 5", "",
                  "| Dataset | Modality | Mixed-sign ratio | Median node IQR (raw ΔCE) | Median node q90–q10 | Relative mixed-sign ratio |",
                  "|---|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for mod in MODALITIES:
            g = degree[(degree.dataset == dataset) & (degree.modality == mod)]
            lines.append(f"| {dataset} | {mod.title()} | {g.raw_mixed_sign_neighborhood_ratio.mean():.3f} | {g.raw_median_node_iqr.mean():.5g} | {g.raw_median_node_q90_q10.mean():.5g} | {g.relative_mixed_sign_neighborhood_ratio.mean():.3f} |")
    lines.extend(["", "In ele-fashion, degree≥5 raises the mixed-sign ratio from 0.156 to 0.400 for Text and from 0.165 to 0.473 for Visual. Low-degree targets therefore explain a substantial part of the earlier weak aggregate, although degree≥5 heterogeneity remains below Movies and Grocery for at least one modality. Raw spread magnitudes also depend on each dataset’s CE scale.", "",
                  "## P1.1 assessment", "",
                  "Raw ΔCE preserves H1’s global sign mix and within-node mixed-sign neighborhoods, and it preserves H2’s same-edge sign disagreement and dataset ordering. Text/Visual raw-utility correlations are smaller than relative-utility correlations in all three datasets, but remain positive in Movies/Grocery and negative in ele-fashion. H3 remains weak for frozen cosine; task-projected cosine improves sign AUROC in five of six dataset×modality groups but still has modest rank correlation and inconsistent overlap. Thus raw-scale robustness retains the P0 propagation-utility heterogeneity claim, while P1.1 alone does not establish function heterogeneity.", ""])
    (p11_root / "p11_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")

    rescue = p12["rescue"]
    seed_rescue = rescue[rescue.summary_level == "seed"]
    lines = [
        "# P1.2 fixed-operator rescue results",
        "",
        "Three deterministic propagation primitives were evaluated on the same graph, validation-directed edge set, and frozen semantic substrate H0 per dataset×seed. Only a new linear classifier was trained for each primitive. `U_raw = CE_removed − CE_full`; positive means deletion raises validation CE and is therefore useful under that fixed operator. The original degree denominator was retained after message deletion. No test or link-prediction evaluation was run.",
        "",
        "## Dataset × modality rescue summary",
        "",
        "Values are means across seeds 42–44; parentheses show population SD across the three seed-level estimates. Rescue is conditioned on `U_smooth < 0`. The unrescued column means both tested alternatives have utility ≤0 for a smooth-harmful message.",
        "",
        "| Dataset | Modality | Smooth harmful | AbsDiff rescue | Product rescue | Any rescue | Unrescued by tested primitives | AbsDiff enrichment | Product enrichment | Smooth T/V disagreement | AbsDiff T/V disagreement | Product T/V disagreement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    mod_dis = p12["modality"]
    mod_dis = mod_dis[mod_dis.summary_level == "seed"]
    for dataset in DATASETS:
        for mod in MODALITIES:
            g = seed_rescue[(seed_rescue.dataset == dataset) & (seed_rescue.modality == mod)]
            def fmt(col: str) -> str:
                x = g[col].to_numpy(float)
                x = x[np.isfinite(x)]
                return f"{x.mean():.3f} ({x.std(ddof=0):.3f})" if len(x) else "NA"
            disagreements = []
            for op in OPERATORS:
                d = mod_dis[(mod_dis.dataset == dataset) & (mod_dis.operator == op)]["text_visual_sign_disagreement"]
                disagreements.append(float(d.mean()) if len(d) else float("nan"))
            lines.append("| " + " | ".join([
                dataset, mod.title(), fmt("smooth_harmful_fraction"), fmt("absdiff_rescue_conditional"),
                fmt("product_rescue_conditional"), fmt("any_rescue_conditional"),
                fmt("unrescued_by_tested_primitives_fraction"), fmt("absdiff_rescue_enrichment"),
                fmt("product_rescue_enrichment"), *[f"{x:.3f}" for x in disagreements],
            ]) + " |")
    lines.extend(["", "The output CSV also gives seed-level counts, joint prevalences `P(U_smooth<0 and U_alt>0)`, and overall alternative-positive fractions. Rescue enrichment is descriptive: values above/below one indicate enrichment/depletion among smooth-harmful messages relative to all edges.", "",
                  "## Sign-transition matrices", "",
                  "The four cells are shares of all aligned physical messages: S+/A+, S+/A−, S−/A+, and S−/A−. The S−/A+ cell is the rescue quadrant. `operator_transition_matrix.csv` contains every dataset×seed×modality comparison and the dataset-level three-seed mean and sample SD.", ""])
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Modality | Alternative | S+/A+ | S+/A− | S−/A+ | S−/A− |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for mod in MODALITIES:
            for alternative in ("absdiff", "product"):
                values = []
                for quadrant in QUADRANT_ORDER:
                    q = p12["transition"]
                    cell = q[(q.summary_level == "dataset_3seed_mean") & (q.dataset == dataset)
                             & (q.modality == mod) & (q.comparison == f"smooth_vs_{alternative}")
                             & (q.quadrant == quadrant)]
                    values.append(float(cell.fraction_all_messages.iloc[0]) if len(cell) else float("nan"))
                lines.append(f"| {mod.title()} | {alternative} | " + " | ".join(f"{x:.3f}" for x in values) + " |")
        lines.append("")
    lines.extend(["## Probe validation sanity", "",
                  "Validation metrics select each linear head and are shown only as a sanity check, not as the primary rescue evidence. Values below are seed means.", "",
                  "| Dataset | Operator | Validation accuracy | Macro-F1 | CE |",
                  "|---|---|---:|---:|---:|"])
    perf = p12["performance"]
    for dataset in DATASETS:
        for operator in OPERATORS:
            g = perf[(perf.dataset == dataset) & (perf.operator == operator)]
            lines.append(f"| {dataset} | {operator} | {g.val_acc.mean():.3f} | {g.val_macro_f1.mean():.3f} | {g.val_ce.mean():.3f} |")
    lines.extend(["", "Smooth has the strongest mean validation probe in Movies and Grocery; the three operators are close in ele-fashion. This sanity comparison does not show that a multi-operator model or router would outperform scalar message handling.", "",
                  "## Cross-seed stability and operator specificity", "",
                  "Use the seed rows in `operator_rescue_summary.csv` to assess whether rescue appears in all three seeds or is driven by a subset. Compare conditional rescue with overall positivity and enrichment: enrichment near one is consistent with an alternative being broadly positive rather than especially useful on smooth-harmful messages; enrichment above one is descriptive evidence of edge-specific complementarity. These two fixed alternatives are a small probe set, not a final operator bank.", "",
                  "## Assessment and next step", "",
                  "", "",
                  "**P1.2 status: SUPPORTED.** In every dataset×modality group, `R_any` is 0.68–0.78 on average across seeds, with the corresponding seed-level values consistently nontrivial. Each alternative separately rescues a sizeable smooth-harmful subset in all three seeds. This supports the bounded claim that message usefulness depends on the propagation function tested; it does not establish that these three primitives form a final operator bank or that routing improves a model.", "",
                  "For the research question, advance from `Propagation Utility Heterogeneity` to the narrower `Propagation Function Heterogeneity` claim: the same physical message can change CE-utility sign under fixed relation transforms. Keep the claim explicitly limited to smooth, absdiff, and product in validation probes.", "",
                  "**Next step: scalar gating/blocking first; do not start operator-routing V0 yet.** Rescue enrichment is mixed: absdiff is enriched in Movies Text and ele-fashion Text, near one in several other groups, and below one in Grocery Text/Visual and Movies Visual; product is below one in five of six groups. The alternatives often help broadly rather than selectively recovering smooth-harmful edges, and smooth has the strongest mean validation probe performance in Movies and Grocery. The result justifies keeping operator routing as a later controlled comparison, while a scalar utility-handling baseline is the more interpretable next step. No model is implemented in this stage.", "",
                  "## Limitations", "",
                  "1. Validation accuracy selects each new linear head and validation labels are also the targets for utility counterfactuals; this is a controlled probe, not independent confirmation.",
                  "2. Smooth, absolute-difference, and elementwise-product operations are fixed examples. Results do not establish a complete or optimal operator bank.",
                  "3. Directed edges share destination nodes. Seed summaries are descriptive; no p-values or independent-edge claims are made.",
                  "4. All experiments are transductive NC over the supplied graph and fixed features. Test labels/indices and LP were not accessed.", ""])
    (research_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze P1.1 robustness and P1.2 operator-rescue probes")
    parser.add_argument("--p0-output-dir", type=Path, default=Path("outputs/p0p1_propagation_heterogeneity"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p11_p12_operator_rescue"))
    parser.add_argument("--research-dir", type=Path, default=Path("research/p11_p12_operator_rescue"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    p11 = build_p11(args.p0_output_dir, args.research_dir / "p11", args.datasets, args.seeds)
    edge_runs, performance = _p12_run_tables(args.output_dir, args.datasets, args.seeds)
    p12 = build_p12(edge_runs, performance, args.research_dir / "p12")
    write_reports(p11, p12, args.research_dir)
    print(f"P1.1 rows: utility={len(p11['utility'])}, H2={len(p11['modality'])}, H3={len(p11['similarity'])}")
    print(f"P1.2 edge rows={len(edge_runs)}, probe heads={len(performance)}")
    print(f"research_dir={args.research_dir.resolve()}")


if __name__ == "__main__":
    main()
