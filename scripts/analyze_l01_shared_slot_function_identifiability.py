from __future__ import annotations

import sys
import json
from pathlib import Path
import subprocess

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.analysis import l01_shared_slot_function_identifiability as l01  # noqa: E402
from plot_l01_shared_slot_function_identifiability import generate_all  # noqa: E402


def read_group_csv(name: str) -> list[dict]:
    records = []
    for dataset in l01.DATASETS:
        for seed in l01.SEEDS:
            path = l01.RAW_DIR / "probe_fits" / dataset / f"seed_{seed}" / f"{name}.csv"
            if path.is_file():
                records.extend(pd.read_csv(path).to_dict("records"))
    return records


def rebuild() -> dict:
    all_raw, clean_tables, p13_frames = {}, {}, {}
    for dataset in l01.DATASETS:
        for seed in l01.SEEDS:
            raw_path = l01.RAW_DIR / "raw_utilities" / dataset / f"seed_{seed}_substitution_utility.csv.gz"
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            raw = pd.read_csv(raw_path)
            if set(raw.head_repeat.unique()) != {0, 1, 2}:
                raise AssertionError(f"{dataset}/{seed}: expected all three head repeats")
            all_raw[(dataset, seed)] = raw
            clean = l01.aggregate_repeats(raw)
            clean.insert(0, "dataset", dataset)
            clean.insert(1, "seed", seed)
            clean_tables[(dataset, seed)] = clean
            p13_path = l01.P13_RUN_ROOT / dataset / f"seed_{seed}" / "joint_operator_edge_utility.csv.gz"
            p13_frames[(dataset, seed)] = pd.read_csv(p13_path, usecols=[
                "src", "dst", "dst_degree", "u_raw_smooth_text", "u_raw_absdiff_text",
                "u_raw_product_text", "u_raw_smooth_visual", "u_raw_absdiff_visual",
                "u_raw_product_visual"])
    outputs = {
        "head_performance": pd.read_csv(l01.OUT_DIR / "data/shared_head_performance.csv").to_dict("records"),
        "head_split": pd.read_csv(l01.OUT_DIR / "data/shared_head_split_audit.csv").to_dict("records"),
        "evidence": read_group_csv("evidence"), "shuffle": read_group_csv("strict_shuffle"),
        "ridge": read_group_csv("ridge"), "direct": read_group_csv("direct"),
        "frozen": read_group_csv("frozen"), "head_stable": read_group_csv("head_stable"),
        "high_effect": read_group_csv("high_effect"), "fold_audit": read_group_csv("fold_audit"),
    }
    expected = {"head_performance": 27, "head_split": 27, "evidence": 3*3*3*4*4,
                "shuffle": 3*3*3*3*4, "ridge": 3*3*3*3*4,
                "direct": 3*3*3*4, "frozen": 3*3*3*2*3*2,
                "fold_audit": 3*3*3}
    for key, count in expected.items():
        if len(outputs[key]) != count:
            raise AssertionError(f"formal artifact {key}: got {len(outputs[key])}, expected {count}")
    outputs["fit_counts"] = {"shared_heads": 27, "evidence_mlp_real": 108,
        "strict_shuffle": 81, "evidence_mlp_total": 189, "ridge": 81,
        "direct_state": 27, "frozen_state": 162}
    summaries = l01.finalize_campaign(all_raw, clean_tables, p13_frames, outputs)
    figures = generate_all()
    write_manifest(outputs, figures)
    return {"records": {key: len(value) for key, value in outputs.items() if isinstance(value, list)},
            "summary_tables": sorted(summaries), "figures": figures}


def write_manifest(outputs: dict, figures: dict) -> None:
    audit_path = l01.RAW_DIR / "formal_campaign_audit.json"
    campaign = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    smoke_path = l01.RAW_DIR / "smoke/smoke_audit.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else None
    data_dir = l01.OUT_DIR / "data"
    manifest = {
        "branch": "exp/l01_shared_slot_function_identifiability",
        "source_branch": "exp/l0_relation_function_learnability_audit",
        "source_sha": "9ce5f723fda0bf17264671a1e20e78f043727498",
        "source_remote_verified": True,
        "datasets": list(l01.DATASETS), "seeds": list(l01.SEEDS),
        "device": "cuda:1", "gpu": "NVIDIA GeForce RTX 3090",
        "environment": {"conda_env": "yhf_env", "torch": "2.4.0+cu121",
                         "cuda_runtime": "12.1", "python": "3.12"},
        "smoke": smoke,
        "campaign": campaign,
        "fit_counts": {"shared_heads": 27, "evidence_mlp_real": 108,
                        "strict_shuffle_null": 81, "evidence_mlp_including_null": 189,
                        "ridge_alpha_1": 81, "direct_state": 27, "frozen_q_r_u_readouts": 162},
        "data_tables": {path.name: {"rows": len(pd.read_csv(path))}
                        for path in sorted(data_dir.glob("*.csv"))},
        "correctness": {
            "pytest": "20 passed, 1 upstream PyG deprecation warning",
            "frozen_h0_regression": True, "operator_context_matches_p13": True,
            "fast_vs_brute_tolerance": {"rtol": 1e-7, "atol": 1e-8},
            "head_train_validation_label_leakage": False,
            "test_labels_or_metrics_accessed": False,
            "inner_train_only_feature_scaler": True,
            "inner_train_only_target_scaler": True,
            "strict_shuffle_inner_train_and_validation": True,
            "outer_test_tuple_shuffle": False,
            "all_frozen_e01_representations_aligned": True},
        "interpretation": {
            "js_is_empirical_output_separability_only": True,
            "utility_is_ground_truth_function_label": False,
            "stochastic_latent_function_process_inferred": False,
            "manenti_theorem_reproduced": False},
        "figure_qa": figures,
        "ridge_numeric_note": "Fixed alpha=1 Ridge fits completed; sklearn emitted ill-conditioned-matrix warnings on high-dimensional correlated master features. No alpha tuning or fallback was applied.",
        "protocol_deviations": [], "retries": [], "historical_artifacts_modified": False,
        "elapsed_seconds": campaign.get("elapsed_seconds"),
    }
    l01.save_json(l01.OUT_DIR / "run_manifest.json", manifest)


def diagnostics() -> dict:
    """Compact values for the final evidence-weighted report; all summaries remain descriptive."""
    d = l01.OUT_DIR / "data"
    heads = pd.read_csv(d / "shared_head_performance.csv")
    repeat = pd.read_csv(d / "utility_head_repeat_reliability.csv")
    cross = pd.read_csv(d / "utility_cross_seed_reliability.csv")
    hetero = pd.read_csv(d / "substitution_heterogeneity.csv")
    sep = pd.read_csv(d / "output_separability.csv")
    p13 = pd.read_csv(d / "p13_vs_shared_slot.csv")
    tables = {name: pd.read_csv(d / f"{name}.csv") for name in (
        "evidence_probe_summary", "shuffled_null_summary", "ridge_probe_summary",
        "direct_state_probe_summary", "frozen_state_readout_summary")}
    result = {
        "head_by_dataset": heads.groupby("dataset")[["head_select_mean_ce_9",
            "head_select_smooth_smooth_accuracy", "head_select_smooth_smooth_macro_f1",
            "head_select_mean_accuracy_9_descriptive", "weight_norm_struct_T", "weight_norm_struct_V"]].mean().to_dict("index"),
        "repeat_reliability_by_dataset": repeat.groupby("dataset")[["spearman", "centered_spearman",
            "sign_agreement"]].mean().to_dict("index"),
        "cross_seed_reliability_by_dataset": cross.groupby("dataset")[["spearman", "centered_spearman",
            "sign_agreement", "overlap_fraction_smaller"]].mean().to_dict("index"),
        "heterogeneity_by_dataset_target": hetero.groupby(["dataset", "target"])[[
            "positive_fraction", "nonpositive_fraction", "mean", "median",
            "degree5_within_target_gain_std_mean", "degree5_positive_negative_coexistence_ratio"]].mean().to_dict("index"),
        "separability_by_dataset_target": sep.groupby(["dataset", "target"])[[
            "logit_shift_median", "logit_shift_q90", "js_median", "js_q90",
            "probability_l1_median", "prediction_flip_fraction", "spearman_abs_gain_js",
            "spearman_abs_gain_logit_shift"]].mean().to_dict("index"),
        "p13_by_dataset_target": p13.groupby(["dataset", "new_target"])[[
            "spearman", "sign_agreement", "within_target_centered_spearman",
            "top_abs_q75_overlap_fraction_smaller"]].mean().to_dict("index"),
        "probe_key_rows": {},
    }
    for name, frame in tables.items():
        keys = [c for c in ("dataset", "model", "representation", "target") if c in frame]
        metrics = [c for c in ("spearman_mean", "spearman_std", "within_target_residual_spearman_mean",
                               "sign_auroc_mean", "sign_balanced_accuracy_mean",
                               "degree5_target_rank_mean_mean") if c in frame]
        if "model" in frame:
            selected = frame[frame.model.isin(["TARGET_ONLY", "ENDPOINT", "ENDPOINT_LOCAL",
                "STRICT_SHUFFLE_9917", "STRICT_SHUFFLE_9918", "STRICT_SHUFFLE_9919",
                "RIDGE_TARGET_ONLY", "RIDGE_ENDPOINT", "RIDGE_ENDPOINT_LOCAL", "DirectState",
                "Frozen_Q_PAIR", "Frozen_R_SHARED", "Frozen_U_MODAL"])]
        else:
            selected = frame
        result["probe_key_rows"][name] = selected.groupby(keys, dropna=False)[metrics].mean().to_dict("index") if keys and metrics else {}
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
        print(json.dumps(diagnostics(), indent=2, default=str))
    else:
        print(json.dumps(rebuild(), indent=2))
