from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("semantic", "uniform", "extent", "single_basis", "multi_basis")
VARIANT_LABELS = {
    "semantic": "SEM", "uniform": "UNI", "extent": "M0-A",
    "single_basis": "M0-B", "multi_basis": "M0-C",
}
OUT_DIR = PROJECT_ROOT / "outputs" / "m0_adaptive_propagation_screen"
RESEARCH_DIR = PROJECT_ROOT / "research" / "m0_adaptive_propagation_screen"


def read_runs(out_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted((out_dir / "runs").glob("*/*/*.json")):
        if path.name == "smoke_result.json":
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") == "completed":
            runs.append(item)
    return runs


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def csv_out(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def basis_similarity(checkpoint_path: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    result = {}
    for modality, name in enumerate(("text", "visual")):
        u, v = state["basis_u"][modality], state["basis_v"][modality]
        matrices = torch.stack([u[k] @ v[k] for k in range(u.size(0))])
        flat = F.normalize(matrices.flatten(1), dim=-1, eps=1e-12)
        cosine = flat @ flat.T
        mask = ~torch.eye(cosine.size(0), dtype=torch.bool)
        offdiag = cosine[mask].abs()
        result[name] = {
            "cosine_matrix": cosine.tolist(),
            "offdiag_mean_abs_cosine": float(offdiag.mean()),
            "offdiag_max_abs_cosine": float(offdiag.max()),
        }
    return result


def build_tables(runs: list[dict[str, Any]], out_dir: Path):
    perf = []
    params = []
    gate = []
    corr = []
    basis = []
    within = []
    interventions = []
    init_audit = []
    for run in runs:
        key = {"dataset": run["dataset"], "seed": run["seed"], "variant": run["variant"], "name": VARIANT_LABELS[run["variant"]]}
        perf.append({
            **key, "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
            "val_accuracy": run["val_accuracy"], "val_macro_f1": run["val_macro_f1"],
            "val_ce": run["val_ce"], "trainable_params": run["trainable_params"],
            "peak_gpu_memory_gb": run["peak_gpu_memory_bytes"] / (1024**3),
            "training_time_sec": run["training_time_sec"],
            "intervention_time_sec": run["intervention_time_sec"],
        })
        pc = run["parameter_counts"]
        params.append({**key, "model_params": run["model_params"], "classifier_params": run["classifier_params"],
                       "trainable_params": run["trainable_params"], **pc})
        diag = run["control_diagnostics"]
        for m, modality in enumerate(("Text", "Visual")):
            gs = diag["g"][m]
            if gs:
                gate.append({**key, "modality": modality, **gs,
                             "fraction_lt_0_1": gs["fraction_lt_0_1"],
                             "fraction_gt_0_9": gs["fraction_gt_0_9"],
                             "mean_abs_g_tv": diag["mean_abs_g_tv"],
                             **{f"within_node_{k}": v for k, v in diag["within_node_g_std"][m].items()}})
            ws = diag["within_node_g_std"][m]
            within.append({**key, "modality": modality, **ws})
            cs = diag["c"][m]
            if cs:
                corr.append({**key, "modality": modality, "control": "c", **cs,
                             "fraction_lt_0_1": cs["fraction_lt_0_1"],
                             "fraction_gt_0_9": cs["fraction_gt_0_9"],
                             "mean_abs_c_tv": diag["mean_abs_c_tv"]})
                rs = diag["correction_ratio"][m]
                corr.append({**key, "modality": modality, "control": "correction_to_default_ratio", **rs,
                             "fraction_lt_0_1": None, "fraction_gt_0_9": None,
                             "mean_abs_c_tv": diag["mean_abs_c_tv"]})
            pis = diag["pi"][m]
            if pis:
                checkpoint = PROJECT_ROOT / run["checkpoint"]
                cosine = basis_similarity(checkpoint)[modality.lower()]
                basis.append({
                    **key, "modality": modality,
                    "pi_basis_mean": json.dumps(pis["basis_mean"]),
                    "pi_basis_std": json.dumps(pis["basis_std"]),
                    "normalized_entropy_mean": pis["normalized_entropy"]["mean"],
                    "normalized_entropy_median": pis["normalized_entropy"]["median"],
                    "normalized_entropy_q10": pis["normalized_entropy"]["q10"],
                    "normalized_entropy_q90": pis["normalized_entropy"]["q90"],
                    "effective_bases_mean": pis["effective_bases"]["mean"],
                    "effective_bases_median": pis["effective_bases"]["median"],
                    "effective_bases_q10": pis["effective_bases"]["q10"],
                    "effective_bases_q90": pis["effective_bases"]["q90"],
                    "mean_l1_pi_tv": diag["mean_l1_pi_tv"],
                    "basis_cosine_matrix": json.dumps(cosine["cosine_matrix"]),
                    "basis_offdiag_mean_abs_cosine": cosine["offdiag_mean_abs_cosine"],
                    "basis_offdiag_max_abs_cosine": cosine["offdiag_max_abs_cosine"],
                })
        for intervention in run["interventions"]["rows"]:
            interventions.append({**key, **intervention})
        audit = run["initialization_audit"]
        init_audit.append({
            **key, "common_all_equal": audit["common_all_equal"],
            "common_hash_extent": audit["common_hashes"]["extent"],
            "common_hash_single_basis": audit["common_hashes"]["single_basis"],
            "common_hash_multi_basis": audit["common_hashes"]["multi_basis"],
            "basis_params_single": audit["basis_params_single"],
            "basis_params_multi": audit["basis_params_multi"],
            "basis_params_equal": audit["basis_params_equal"],
        })

    data_dir = RESEARCH_DIR / "data"
    csv_out(data_dir / "performance_by_run.csv", perf)
    csv_out(data_dir / "parameter_summary.csv", params)
    csv_out(data_dir / "gate_diagnostics.csv", gate)
    csv_out(data_dir / "correction_diagnostics.csv", corr)
    csv_out(data_dir / "basis_diagnostics.csv", basis)
    csv_out(data_dir / "within_node_control_variation.csv", within)
    csv_out(data_dir / "intervention_by_run.csv", interventions)
    csv_out(data_dir / "common_initialization_audit.csv", init_audit)

    perf_df = pd.DataFrame(perf)
    summary = []
    for group_name, group in list(perf_df.groupby(["dataset", "variant", "name"], sort=False)) + [
        (("ALL", variant, VARIANT_LABELS[variant]), perf_df[perf_df.variant == variant]) for variant in VARIANTS
    ]:
        dataset, variant, name = group_name
        row = {"dataset": dataset, "variant": variant, "name": name, "n": len(group)}
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch", "training_time_sec", "peak_gpu_memory_gb"):
            if len(group):
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(group[metric])
            else:
                row[f"{metric}_mean"], row[f"{metric}_std"] = float("nan"), float("nan")
        summary.append(row)
    csv_out(data_dir / "performance_summary.csv", summary)

    # Same dataset/seed paired differences, with positive defined as first minus second.
    metric_cols = ("val_accuracy", "val_macro_f1", "val_ce")
    lookup = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in perf}
    comparisons = (("UNI-SEM", "uniform", "semantic"), ("A-UNI", "extent", "uniform"),
                   ("B-A", "single_basis", "extent"), ("C-B", "multi_basis", "single_basis"),
                   ("C-A", "multi_basis", "extent"))
    paired = []
    for label, left, right in comparisons:
        for dataset in DATASETS + ("ALL",):
            rows = []
            for ds in DATASETS if dataset == "ALL" else (dataset,):
                for seed in (42, 43, 44):
                    if (ds, seed, left) in lookup and (ds, seed, right) in lookup:
                        rows.append({metric: lookup[(ds, seed, left)][metric] - lookup[(ds, seed, right)][metric] for metric in metric_cols})
            row = {"dataset": dataset, "comparison": label, "left": left, "right": right, "n": len(rows)}
            for metric in metric_cols:
                vals = [r[metric] for r in rows]
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(vals) if vals else (float("nan"), float("nan"))
                row[f"{metric}_by_seed"] = json.dumps(vals)
            paired.append(row)
    csv_out(data_dir / "paired_delta_summary.csv", paired)

    int_df = pd.DataFrame(interventions)
    per_run_interventions = []
    int_summary = []
    if len(int_df):
        run_groups = ["dataset", "seed", "variant", "name", "intervention"]
        for key, group in int_df.groupby(run_groups, sort=False):
            dataset, seed, variant, name, intervention = key
            row = {"dataset": dataset, "seed": int(seed), "variant": variant, "name": name,
                   "intervention": intervention, "n_repeats": len(group)}
            for metric in ("delta_ce", "delta_accuracy", "delta_macro_f1"):
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_repeat_std"] = float(group[metric].std(ddof=0))
            per_run_interventions.append(row)
        per_run_df = pd.DataFrame(per_run_interventions)
        summary_groups = ["dataset", "variant", "name", "intervention"]
        keys = list(per_run_df.groupby(summary_groups, sort=False))
        for variant in VARIANTS:
            for intervention in per_run_df[per_run_df.variant == variant].intervention.unique():
                part = per_run_df[(per_run_df.variant == variant) & (per_run_df.intervention == intervention)]
                if not part.empty:
                    keys.append((("ALL", variant, VARIANT_LABELS[variant], intervention), part))
        for key, group in keys:
            dataset, variant, name, intervention = key
            row = {"dataset": dataset, "variant": variant, "name": name,
                   "intervention": intervention, "n_runs": len(group),
                   "n_repeat_records": int(group.n_repeats.sum())}
            for metric in ("delta_ce", "delta_accuracy", "delta_macro_f1"):
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean_std(group[f"{metric}_mean"])
                row[f"{metric}_within_run_repeat_std_mean"] = float(group[f"{metric}_repeat_std"].mean())
            int_summary.append(row)
    csv_out(data_dir / "intervention_summary.csv", int_summary)
    return summary, paired, gate, corr, basis, int_summary


def _save_figures(perf_summary, paired, gate, corr, basis, intervention_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RESEARCH_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"semantic": "#777777", "uniform": "#3978a8", "extent": "#cc7a00", "single_basis": "#2b8c72", "multi_basis": "#8055a0"}
    summary_df = pd.DataFrame(perf_summary)
    datasets = list(DATASETS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("val_accuracy", "val_macro_f1", "val_ce"), ("Validation accuracy", "Validation Macro-F1", "Validation CE")):
        for variant in VARIANTS:
            rows = summary_df[(summary_df.dataset.isin(datasets)) & (summary_df.variant == variant)]
            values = [float(rows[rows.dataset == ds][f"{metric}_mean"].iloc[0]) if not rows[rows.dataset == ds].empty else np.nan for ds in datasets]
            errs = [float(rows[rows.dataset == ds][f"{metric}_std"].iloc[0]) if not rows[rows.dataset == ds].empty else 0 for ds in datasets]
            ax.errorbar(range(3), values, yerr=errs, marker="o", capsize=3, color=colors[variant], label=VARIANT_LABELS[variant])
        ax.set_title(title)
        ax.set_xticks(range(3), datasets, rotation=15)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "performance_screen.png", dpi=180)
    plt.close(fig)

    pair_df = pd.DataFrame(paired)
    pair_df = pair_df[pair_df.dataset == "ALL"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, metric, title in zip(axes, ("val_accuracy", "val_macro_f1", "val_ce"), ("Δ Accuracy", "Δ Macro-F1", "Δ CE")):
        rows = pair_df.set_index("comparison")
        means = [rows.loc[label, f"{metric}_mean"] for label in ("UNI-SEM", "A-UNI", "B-A", "C-B", "C-A")]
        stds = [rows.loc[label, f"{metric}_std"] for label in ("UNI-SEM", "A-UNI", "B-A", "C-B", "C-A")]
        ax.bar(range(5), means, yerr=stds, color="#4d8497", capsize=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(5), ("UNI−SEM", "A−UNI", "B−A", "C−B", "C−A"), rotation=25)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "paired_deltas.png", dpi=180)
    plt.close(fig)

    gate_df = pd.DataFrame(gate)
    if len(gate_df):
        gate_df = gate_df[(gate_df.dataset == "ALL")]
    # Show run-level validation of edge gate spread by dataset and modality.
    raw_gate_path = RESEARCH_DIR / "data" / "gate_diagnostics.csv"
    raw_gate = pd.read_csv(raw_gate_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, modality in zip(axes, ("Text", "Visual")):
        subset = raw_gate[(raw_gate.modality == modality) & (raw_gate.variant.isin(["extent", "single_basis", "multi_basis"]))]
        labels, values, errors = [], [], []
        for variant in ("extent", "single_basis", "multi_basis"):
            for dataset in datasets:
                part = subset[(subset.variant == variant) & (subset.dataset == dataset)]
                labels.append(f"{VARIANT_LABELS[variant]}\n{dataset}")
                values.append(float(part["std"].mean()) if not part.empty else 0.0)
                errors.append(float(part["std"].std(ddof=0)) if len(part) > 1 else 0.0)
        ax.bar(range(len(labels)), values, yerr=errors, color=[colors[v] for v in ("extent", "single_basis", "multi_basis") for _ in datasets], capsize=2)
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"{modality}: gate SD")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "gate_distributions.png", dpi=180)
    plt.close(fig)

    corr_df = pd.DataFrame(corr)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, control in zip(axes, ("c", "correction_to_default_ratio")):
        subset = corr_df[corr_df.control == control]
        for variant in ("single_basis", "multi_basis"):
            part = subset[subset.variant == variant]
            positions = np.arange(3) + (-0.16 if variant == "single_basis" else 0.16)
            medians = [float(part[part.dataset == ds]["median"].mean()) if not part[part.dataset == ds].empty else np.nan for ds in datasets]
            q25 = [float(part[part.dataset == ds]["q25"].mean()) if not part[part.dataset == ds].empty else np.nan for ds in datasets]
            q75 = [float(part[part.dataset == ds]["q75"].mean()) if not part[part.dataset == ds].empty else np.nan for ds in datasets]
            ax.errorbar(positions, medians, yerr=[np.array(medians)-q25, np.array(q75)-medians], fmt="o", color=colors[variant], label=VARIANT_LABELS[variant], capsize=3)
        ax.set_xticks(range(3), datasets, rotation=15)
        ax.set_title("Correction strength c" if control == "c" else "Correction / default norm")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(fig_dir / "correction_usage.png", dpi=180)
    plt.close(fig)

    basis_df = pd.DataFrame(basis)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        subset = basis_df[basis_df.dataset == dataset]
        vals = [float(subset[subset.modality == mod].normalized_entropy_mean.mean()) if not subset[subset.modality == mod].empty else np.nan for mod in ("Text", "Visual")]
        ax.bar(("Text", "Visual"), vals, color=("#3978a8", "#d17c37"))
        ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_title(dataset)
        ax.set_ylabel("Normalized routing entropy")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "basis_usage.png", dpi=180)
    plt.close(fig)

    int_df = pd.DataFrame(intervention_summary)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, variant in zip(axes, ("extent", "single_basis", "multi_basis")):
        subset = int_df[(int_df.dataset == "ALL") & (int_df.variant == variant)]
        labels, values, errors = [], [], []
        for intervention in ("extent_off", "function_off", "edge_control_shuffle", "modality_tied", "basis_uniform"):
            row = subset[subset.intervention == intervention]
            if not row.empty:
                labels.append(intervention.replace("_", "\n"))
                values.append(float(row.delta_ce_mean.iloc[0]))
                errors.append(float(row.delta_ce_std.iloc[0]))
        if labels:
            ax.bar(range(len(labels)), values, yerr=errors, color=colors[variant], capsize=3)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(VARIANT_LABELS[variant] + ": Δ validation CE")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "intervention_effects.png", dpi=180)
    plt.close(fig)


def update_manifest(runs: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    import subprocess
    import platform
    import torch_geometric

    def git(*args):
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True)
        return result.stdout.strip()

    completed = [{
        "dataset": r["dataset"], "seed": r["seed"], "variant": r["variant"],
        "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
        "val_accuracy": r["val_accuracy"], "val_macro_f1": r["val_macro_f1"], "val_ce": r["val_ce"],
        "training_time_sec": r["training_time_sec"], "peak_gpu_memory_bytes": r["peak_gpu_memory_bytes"],
        "checkpoint": r.get("checkpoint"), "status": r["status"],
    } for r in runs]
    failed_path = out_dir / "failed_runs.json"
    failed = json.loads(failed_path.read_text(encoding="utf-8")) if failed_path.exists() else []
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": prop.name, "total_memory_bytes": int(prop.total_memory)})
    manifest = {
        "stage": "M0 — Adaptive Propagation Mechanism Screen",
        "source_branch": "exp/p13_joint_readout_operator_probe",
        "source_sha": "83487eada5ca4909a83140a3de498514f2d6d7d8",
        "source_remote_ref": "origin/exp/p13_joint_readout_operator_probe",
        "source_remote_sha": "83487eada5ca4909a83140a3de498514f2d6d7d8",
        "final_branch": git("branch", "--show-current"),
        "git_status_at_start": "clean",
        "environment": {
            "python": platform.python_version(), "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "pyg": torch_geometric.__version__,
            "cuda_available": torch.cuda.is_available(), "devices": gpus,
            "selected_device": "cuda:0 for the first 40 unique completed runs; cuda:1 for the final 5 after user-authorized device switch",
        },
        "data_paths": {
            "data_root": "/hdd1/DataInHere/YHF/data",
            "Movies_text": "/hdd1/DataInHere/YHF/data/Movies/TextFeature/Movies_roberta_base_512_mean.npy",
            "Movies_visual": "/hdd1/DataInHere/YHF/data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy",
            "Grocery_text": "/hdd1/DataInHere/YHF/data/Grocery/TextFeature/Grocery_roberta_base_256_mean.npy",
            "Grocery_visual": "/hdd1/DataInHere/YHF/data/Grocery/ImageFeature/Grocery_openai_clip-vit-large-patch14.npy",
            "ele-fashion_joint": "/hdd1/DataInHere/YHF/data/ele-fashion/clip_feat.pt",
            "split_root": "/hdd1/DataInHere/YHF/data/MAGB_split plus official ele-fashion split.pt",
        },
        "datasets": list(DATASETS), "seeds": [42, 43, 44], "variants": list(VARIANTS),
        "protocol": {
            "name": "unified_full_graph_nc_v1", "full_graph_training": True,
            "train_labels_only": True, "selection": "best validation accuracy",
            "evaluate_test": False, "test_split_field_read": False, "test_indices_attached_to_data": False,
            "test_labels_exposed_to_runner": False, "link_prediction": False,
            "per_dataset_tuning": False, "split_modified": False,
        },
        "model_hyperparameters": {
            "hidden_dim": 128, "projection": ["Linear(input,256)", "GELU", "Dropout(0.2)", "Linear(256,128)", "LayerNorm(128)"],
            "relation_dim": 32, "pair_hidden_dim": 64, "pair_output_dim": 32,
            "relation_state_dim": 64, "modality_embed_dim": 8,
            "fusion": ["Linear(512,256)", "GELU", "Dropout(0.2)", "Linear(256,128)", "LayerNorm(128)"],
            "edge_chunk_size": 100000, "basis_K": 4, "single_basis_rank": 32,
            "multi_basis_rank_each": 8, "correction_basis_params_B": 16384,
            "correction_basis_params_C": 16384,
            "aggregation": "sum messages divided by original non-self incoming physical degree",
            "structure": "one propagation step; no self-loop messages; preserve every other directed physical message",
            "auxiliary_losses": 0.0,
        },
        "initialization": {
            "gate_bias": 2.0, "correction_bias": -2.0, "basis_logits_bias": 0.0,
            "basis_logits_weight_std": 0.001, "basis_u_xavier_scale": 0.05,
            "basis_v": "Xavier uniform", "common_abc_modules": "fixed order, exact tensor equality audited",
        },
        "training": {
            "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
            "max_epochs": 300, "patience": 30, "minimum_epoch": 30,
            "gradient_clip_norm": 1.0, "validation_frequency": 1,
        },
        "commands": [
            "conda run --no-capture-output -n yhf_env python scripts/run_m0_adaptive_propagation.py --smoke --device cuda:0",
            "conda run --no-capture-output -n yhf_env python scripts/run_m0_adaptive_propagation.py --campaign --device cuda:0",
            "conda run --no-capture-output -n yhf_env python scripts/run_m0_adaptive_propagation.py --campaign --device cuda:1",
            "conda run --no-capture-output -n yhf_env python scripts/analyze_m0_adaptive_propagation.py",
            "conda run --no-capture-output -n yhf_env python -m pytest -q tests/test_m0_adaptive_propagation.py",
        ],
        "device_usage": {
            "cuda:0": {"completed_unique_runs": 40, "note": "GPU 0 was shared with an unrelated process during part of the campaign."},
            "cuda:1": {"completed_unique_runs": 5, "note": "The campaign resumed here after the user authorized switching when GPU 0 was occupied."},
        },
        "output_paths": {
            "checkpoints": "outputs/m0_adaptive_propagation_screen/checkpoints",
            "raw_runs": "outputs/m0_adaptive_propagation_screen/runs",
            "tracked_research": "research/m0_adaptive_propagation_screen",
        },
        "completed_runs": completed, "completed_run_count": len(completed),
        "failed_runs": failed, "failed_run_count": len(failed),
        "retry_resolution": [
            {
                "dataset": item["dataset"], "seed": item["seed"], "variant": item["variant"],
                "failed_attempt_error": item["error"], "final_status": "completed",
                "resolution": "Fixed the SEM empty-control diagnostic edge case and reran the resumable campaign.",
            }
            for item in failed
            if any((r["dataset"], r["seed"], r["variant"]) == (item["dataset"], item["seed"], item["variant"]) for r in runs)
        ],
        "implementation_deviations": [
            "M0 uses an isolated NC loader that retrieves only train_idx and val_idx split fields; test_idx is never attached to the data object.",
            "CUDA forward/inference equivalence uses 1e-5 tolerance after observed CUDA index_add roundoff of 1.43e-6; CPU chunk equivalence is checked at 1e-6.",
        ],
        "smoke": json.loads((out_dir / "smoke" / "Movies_seed_42" / "smoke_result.json").read_text(encoding="utf-8"))
        if (out_dir / "smoke" / "Movies_seed_42" / "smoke_result.json").exists() else None,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Analyze M0 adaptive propagation runs")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    runs = read_runs(args.output_dir)
    summary, paired, gate, corr, basis, intervention_summary = build_tables(runs, args.output_dir)
    if runs:
        _save_figures(summary, paired, gate, corr, basis, intervention_summary)
    manifest = update_manifest(runs, args.output_dir)
    (RESEARCH_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    completed = manifest["completed_run_count"]
    print(f"[analysis] runs={completed} report_dir={RESEARCH_DIR}", flush=True)


if __name__ == "__main__":
    main()
