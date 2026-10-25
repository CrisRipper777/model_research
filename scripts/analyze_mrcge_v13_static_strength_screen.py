#!/usr/bin/env python3
"""Independently audit and summarize the Static-Strength validation screen."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mrcge_v13_static_strength_screen as runner


DATASETS = runner.DATASETS
TARGETS = runner.TARGET_CELLS
METRICS = ("val_acc", "val_macro_f1", "val_ce")
MODEL_SPECS = (
    ("mrcge_v12", "M_repaired", "V1.2 M_repaired"),
    ("mrcge_v13", "joint_view", "V1.3-A Joint View"),
    (runner.STATIC_MODEL, runner.STATIC_CONDITION, "V1.3 Static-Strength"),
)
COMPARE_FIELDS = (
    "dataset", "seed", "model", "condition", "display_name", "source", "attempt", "status",
    "best_acc_epoch", "best_f1_epoch", "epochs_completed", "val_acc", "val_macro_f1", "val_ce",
    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1",
    "train_ce", "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap", "model_parameters",
    "classifier_parameters", "total_parameters", "wall_seconds", "peak_allocated_bytes",
    "peak_reserved_bytes", "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path",
    "best_f1_checkpoint_sha256", "split_fingerprint", "source_git_sha", "test_evaluation",
)


def metric(row: dict, key: str, *, optional: bool = False) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        if optional:
            return None
        raise RuntimeError(f"required metric {key} missing from {row}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite metric {key} in {row}")
    return result


def match_one(rows: list[dict[str, str]], predicate, label: str) -> dict[str, str]:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def verify_checkpoints(row: dict[str, str], label: str) -> None:
    if row.get("test_evaluation", "false").lower() not in ("false", "0", ""):
        raise RuntimeError(f"Test evaluation is marked enabled for {label}")
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                               ("best_f1_checkpoint_path", "best_f1_checkpoint_sha256")):
        path_text, expected = row.get(path_key, ""), row.get(hash_key, "")
        path = Path(path_text)
        if not path_text or not expected or not path.is_file() or runner.sha256_file(path) != expected:
            raise RuntimeError(f"checkpoint hash audit failed for {label}: {path}")


def history_refs(dataset: str, seed: int) -> tuple[dict[str, str], dict[str, str]]:
    v13_rows = runner.read_rows(ROOT / "research/mrcge_v13/data/runs.csv")
    v13 = match_one(v13_rows, lambda row: row.get("dataset") == dataset
        and row.get("variant") == "joint_view" and row.get("seed") == str(seed)
        and row.get("status") == "complete", f"V1.3-A {dataset}/seed{seed}")
    verify_checkpoints(v13, f"V1.3-A {dataset}/seed{seed}")

    v12_rows = runner.read_rows(ROOT / "research/mrcge_v12_confirmation/data/combined_selected_runs.csv")
    v12 = match_one(v12_rows, lambda row: row.get("dataset") == dataset
        and row.get("condition") == "M_repaired" and row.get("seed") == str(seed)
        and row.get("status") == "complete"
        and row.get("selected_for_combined", "").lower() == "true"
        and row.get("eligible_complete", "").lower() == "true",
        f"selected V1.2 M_repaired {dataset}/seed{seed}")
    if v12.get("checkpoint_accuracy_verified", "").lower() != "true" or \
            v12.get("checkpoint_best_f1_verified", "").lower() != "true":
        raise RuntimeError(f"selected V1.2 row lacks dual checkpoint verification: {dataset}/seed{seed}")
    verify_checkpoints(v12, f"V1.2 M_repaired {dataset}/seed{seed}")
    expected_split = str(runner.OmegaConf.load(
        ROOT / "configs/study/mrcge_v13_static_strength_screen.yaml").split_fingerprints[dataset])
    for row, name in ((v12, "V1.2"), (v13, "V1.3-A")):
        if row.get("split_fingerprint") != expected_split:
            raise RuntimeError(f"{name} {dataset}/seed{seed} has a different split fingerprint")
    if v12.get("split_file_sha256") != v13.get("split_file_sha256"):
        raise RuntimeError(f"V1.2/V1.3-A split file SHA mismatch for {dataset}/seed{seed}")
    return v12, v13


def static_run(dataset: str, seed: int) -> dict[str, str]:
    rows = runner.read_rows(runner.DATA / "runs.csv")
    matches = [row for row in rows if row.get("model") == runner.STATIC_MODEL
        and row.get("condition") == runner.STATIC_CONDITION and row.get("dataset") == dataset
        and row.get("seed") == str(seed) and row.get("status") == "complete"]
    if not matches:
        raise RuntimeError(f"missing successful Static-Strength run for {dataset}/seed{seed}")
    row = max(matches, key=lambda item: int(item.get("attempt", 0)))
    if row.get("checkpoint_reload_verified", "").lower() != "true" or \
            row.get("validation_rng_unchanged", "").lower() != "true":
        raise RuntimeError(f"Static-Strength checkpoint/RNG audit failed for {dataset}/seed{seed}")
    verify_checkpoints(row, f"Static-Strength {dataset}/seed{seed}")
    if row.get("split_fingerprint") != str(runner.OmegaConf.load(
            ROOT / "configs/study/mrcge_v13_static_strength_screen.yaml").split_fingerprints[dataset]):
        raise RuntimeError(f"Static-Strength split fingerprint mismatch for {dataset}/seed{seed}")
    return row


def canonical(row: dict[str, str], model: str, condition: str, display: str,
              source: str) -> dict[str, object]:
    result = {field: row.get(field, "") for field in COMPARE_FIELDS}
    result.update({"model": model, "condition": condition, "display_name": display,
                   "source": source, "test_evaluation": False})
    if model == "mrcge_v13":
        result["best_acc_epoch"] = row.get("best_acc_epoch", "")
        result["epochs_completed"] = row.get("epochs_completed", "")
    return result


def get_baseline_view_diagnostics(dataset: str, seed: int, v13: dict[str, str]) -> dict[str, str]:
    path = ROOT / "research/mrcge_v13/data/view_diagnostics.csv"
    rows = runner.read_rows(path)
    row = match_one(rows, lambda item: item.get("dataset") == dataset
        and item.get("variant") == "joint_view" and item.get("seed") == str(seed)
        and item.get("scope") == "best_accuracy_validation"
        and item.get("source_git_sha") == v13.get("source_git_sha"),
        f"V1.3-A best-accuracy view diagnostics {dataset}/seed{seed}")
    for modality in ("text", "visual"):
        for relation in ("sP", "sS"):
            for suffix in ("mean", "std", "p10", "p50", "p90"):
                metric(row, f"private_{relation}_{suffix}_{modality}")
    return row


def add_strength_record(records: list[dict[str, object]], run: dict[str, str],
                        model: str, condition: str, source: str,
                        diagnostic: dict[str, object]) -> None:
    record = {"model": model, "condition": condition, "dataset": run["dataset"],
              "seed": int(run["seed"]), "source": source, "test_evaluation": False}
    record.update(diagnostic)
    records.append(record)


def make_strength_diagnostics(static_runs: dict[tuple[str, int], dict[str, str]],
                              refs: dict[tuple[str, int], tuple[dict[str, str], dict[str, str]]]
                              ) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key, run in static_runs.items():
        diag_path = Path(run["mechanism_diagnostics_path"])
        payload = runner.read_json(diag_path, {})
        info = payload.get("info", {})
        detail = info.get("details", {})
        mechanism = detail.get("v13_mechanism", {})
        model_cfg = runner.OmegaConf.load(Path(run["config_snapshot_path"]))
        checkpoint_payload = __import__("torch").load(run["checkpoint_path"], map_location="cpu", weights_only=False)
        state = checkpoint_payload.get("model_state", {})
        logits = state.get("static_logits")
        if logits is None or tuple(logits.shape) != (2, 2):
            raise RuntimeError(f"Static checkpoint lacks [2,2] logits: {diag_path}")
        import torch
        strengths = (2.0 * torch.sigmoid(logits.float())).tolist()
        rho = [float(torch.sigmoid(torch.as_tensor(state["rho_raw"][i])).item()) for i in range(2)]
        gamma = [float(torch.sigmoid(torch.as_tensor(state["gamma_raw"][i])).item()) for i in range(2)]
        if not model_cfg.model.static_strength_enabled or model_cfg.model.static_strength_variant != "static_strength":
            raise RuntimeError("Static resolved config lost its explicit mechanism flags")
        record: dict[str, object] = {
            "static_logits": json.dumps(logits.tolist(), separators=(",", ":")),
            "sP_text": strengths[0][0], "sS_text": strengths[0][1],
            "sP_visual": strengths[1][0], "sS_visual": strengths[1][1],
            "rho_text": rho[0], "rho_visual": rho[1], "gamma_text": gamma[0], "gamma_visual": gamma[1],
            "rho_sP_text": rho[0] * strengths[0][0], "rho_sP_visual": rho[1] * strengths[1][0],
            "gamma_sS_text": gamma[0] * strengths[0][1], "gamma_sS_visual": gamma[1] * strengths[1][1],
            "private_weighted_physical_rms_text": metric(mechanism, "weighted_physical_private_contribution_rms_text"),
            "private_weighted_physical_rms_visual": metric(mechanism, "weighted_physical_private_contribution_rms_visual"),
            "collaborative_weighted_physical_rms_text": metric(mechanism, "weighted_physical_collaborative_contribution_rms_text"),
            "collaborative_weighted_physical_rms_visual": metric(mechanism, "weighted_physical_collaborative_contribution_rms_visual"),
            "weighted_semantic_rms_text": metric(mechanism, "weighted_semantic_private_contribution_rms_text"),
            "weighted_semantic_rms_visual": metric(mechanism, "weighted_semantic_private_contribution_rms_visual"),
            "early_cross_residual_ratio_text": metric(mechanism, "early_cross_to_neighbor_rms_text"),
            "early_cross_residual_ratio_visual": metric(mechanism, "early_cross_to_neighbor_rms_visual"),
            "early_cross_delta_rms_text": metric(mechanism, "effective_early_cross_delta_rms_text"),
            "early_cross_delta_rms_visual": metric(mechanism, "effective_early_cross_delta_rms_visual"),
            "late_residual_ratio_text": metric(mechanism, "late_residual_ratio_text"),
            "late_residual_ratio_visual": metric(mechanism, "late_residual_ratio_visual"),
            "model_parameters": int(run["model_parameters"]),
            "total_parameters": int(run["total_parameters"]),
            "peak_allocated_bytes": int(run["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(run["peak_reserved_bytes"]),
            "mechanism_diagnostics_path": str(diag_path),
            "diagnostic_source": "best_validation_accuracy_checkpoint",
        }
        if max(strengths[0] + strengths[1]) > 2.0 or min(strengths[0] + strengths[1]) <= 0.0:
            raise RuntimeError("Static checkpoint strength fell outside the mathematical (0,2) range")
        add_strength_record(output, run, runner.STATIC_MODEL, runner.STATIC_CONDITION,
                            "trained_checkpoint", record)

    for key, (v12, v13) in refs.items():
        view = get_baseline_view_diagnostics(key[0], key[1], v13)
        record: dict[str, object] = {"rho_text": metric(view, "rho_text"),
            "rho_visual": metric(view, "rho_visual"), "gamma_text": metric(view, "gamma_text"),
            "gamma_visual": metric(view, "gamma_visual"),
            "effective_physical_text_mean": metric(view, "rho_text") * metric(view, "private_sP_mean_text"),
            "effective_physical_visual_mean": metric(view, "rho_visual") * metric(view, "private_sP_mean_visual"),
            "effective_semantic_text_mean": metric(view, "gamma_text") * metric(view, "private_sS_mean_text"),
            "effective_semantic_visual_mean": metric(view, "gamma_visual") * metric(view, "private_sS_mean_visual"),
            "physical_strength_text_mean": metric(view, "private_sP_mean_text"),
            "physical_strength_text_std": metric(view, "private_sP_std_text"),
            "physical_strength_text_p10": metric(view, "private_sP_p10_text"),
            "physical_strength_text_p50": metric(view, "private_sP_p50_text"),
            "physical_strength_text_p90": metric(view, "private_sP_p90_text"),
            "semantic_strength_text_mean": metric(view, "private_sS_mean_text"),
            "semantic_strength_text_std": metric(view, "private_sS_std_text"),
            "semantic_strength_text_p10": metric(view, "private_sS_p10_text"),
            "semantic_strength_text_p50": metric(view, "private_sS_p50_text"),
            "semantic_strength_text_p90": metric(view, "private_sS_p90_text"),
            "physical_strength_visual_mean": metric(view, "private_sP_mean_visual"),
            "physical_strength_visual_std": metric(view, "private_sP_std_visual"),
            "physical_strength_visual_p10": metric(view, "private_sP_p10_visual"),
            "physical_strength_visual_p50": metric(view, "private_sP_p50_visual"),
            "physical_strength_visual_p90": metric(view, "private_sP_p90_visual"),
            "semantic_strength_visual_mean": metric(view, "private_sS_mean_visual"),
            "semantic_strength_visual_std": metric(view, "private_sS_std_visual"),
            "semantic_strength_visual_p10": metric(view, "private_sS_p10_visual"),
            "semantic_strength_visual_p50": metric(view, "private_sS_p50_visual"),
            "semantic_strength_visual_p90": metric(view, "private_sS_p90_visual"),
            "early_cross_to_neighbor_text": metric(view, "early_cross_to_neighbor_rms_text"),
            "early_cross_to_neighbor_visual": metric(view, "early_cross_to_neighbor_rms_visual"),
            "late_residual_ratio_text": metric(view, "late_residual_ratio_text"),
            "late_residual_ratio_visual": metric(view, "late_residual_ratio_visual"),
            "model_parameters": v13.get("model_parameters", ""),
            "total_parameters": v13.get("total_parameters", ""),
            "peak_allocated_bytes": v13.get("peak_allocated_bytes", ""),
            "peak_reserved_bytes": v13.get("peak_reserved_bytes", ""),
            "nodewise_joint_strength_diagnostics_path": str(ROOT / "research/mrcge_v13/data/view_diagnostics.csv"),
            "diagnostic_source": "frozen_V1.3-A_best_accuracy_validation_checkpoint"}
        add_strength_record(output, v13, "mrcge_v13", "joint_view", "historical_reference", record)
    return output


def paired_rows(records: dict[tuple[str, int], dict[str, dict]],
                datasets_by_seed: dict[str, tuple[int, ...]]) -> list[dict[str, object]]:
    output = []
    comparisons = (("Static-Strength minus V1.2", "static", "v12"),
                   ("Joint View minus V1.2", "joint", "v12"),
                   ("Joint View minus Static-Strength", "joint", "static"))
    for dataset, seeds in datasets_by_seed.items():
        for seed in seeds:
            for contrast, left_key, right_key in comparisons:
                left, right = records[(dataset, seed)][left_key], records[(dataset, seed)][right_key]
                row: dict[str, object] = {"dataset": dataset, "seed": seed, "contrast": contrast,
                    "model_a": left["display_name"], "model_b": right["display_name"],
                    "test_evaluation": False}
                for name in METRICS:
                    delta = float(metric(left, name)) - float(metric(right, name))
                    row[f"delta_{name}"] = delta
                    row[f"delta_{name}_pp"] = 100.0 * delta if name != "val_ce" else ""
                output.append(row)
    return output


def summarize(comparison: list[dict[str, object]], paired: list[dict[str, object]],
              datasets_by_seed: dict[str, tuple[int, ...]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset, seeds in datasets_by_seed.items():
        for display in (spec[2] for spec in MODEL_SPECS):
            rows = [row for row in comparison if row["dataset"] == dataset
                    and row["display_name"] == display]
            if len(rows) != len(seeds):
                raise RuntimeError(f"incomplete {display} group for {dataset}: {len(rows)}/{len(seeds)}")
            item: dict[str, object] = {"dataset": dataset, "row_type": "model", "model": display,
                "n": len(rows), "seeds": ",".join(map(str, seeds)), "test_evaluation": False}
            for name in METRICS:
                values = [float(metric(row, name)) for row in rows]
                item[f"{name}_mean"] = statistics.mean(values)
                item[f"{name}_population_sd"] = statistics.pstdev(values)
            for name in ("train_acc", "train_ce", "wall_seconds", "total_parameters",
                         "peak_allocated_bytes", "peak_reserved_bytes"):
                values = [float(row[name]) for row in rows if row.get(name) not in (None, "")]
                item[f"{name}_mean"] = statistics.mean(values) if values else ""
                item[f"{name}_population_sd"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
            output.append(item)
        for contrast in ("Static-Strength minus V1.2", "Joint View minus V1.2",
                         "Joint View minus Static-Strength"):
            rows = [row for row in paired if row["dataset"] == dataset and row["contrast"] == contrast]
            if len(rows) != len(seeds):
                raise RuntimeError(f"incomplete {contrast} group for {dataset}")
            item = {"dataset": dataset, "row_type": "paired_difference", "model": contrast,
                    "n": len(rows), "seeds": ",".join(map(str, seeds)), "test_evaluation": False}
            for name in METRICS:
                values = [float(row[f"delta_{name}"]) for row in rows]
                item[f"delta_{name}_mean"] = statistics.mean(values)
                item[f"delta_{name}_population_sd"] = statistics.pstdev(values)
                item[f"delta_{name}_pp_mean"] = 100.0 * statistics.mean(values) if name != "val_ce" else ""
                item[f"delta_{name}_pp_population_sd"] = 100.0 * statistics.pstdev(values) if name != "val_ce" else ""
            output.append(item)
    return output


def make_hybrid_rows() -> list[dict[str, object]]:
    rows = runner.read_rows(ROOT / "research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv")
    output = []
    for seed in (42, 43, 44):
        row = match_one(rows, lambda item: item.get("model") == runner.HYBRID_MODEL
            and item.get("condition") == "H2_physical_3hop" and item.get("dataset") == "Movies"
            and item.get("seed") == str(seed), f"fixed-split Hybrid Movies/seed{seed}")
        if row.get("split_fingerprint") != runner.EXPECTED_MOVIES_SPLIT:
            raise RuntimeError(f"Hybrid Movies/seed{seed} fixed split mismatch")
        output.append({"dataset": "Movies", "seed": seed, "model": runner.HYBRID_MODEL,
            "condition": row.get("condition"), "source": row.get("source"),
            **{key: row.get(key, "") for key in (
                "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
                "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce", "best_acc_epoch",
                "best_f1_epoch", "total_parameters", "wall_seconds", "peak_allocated_bytes",
                "peak_reserved_bytes", "split_fingerprint", "checkpoint_path", "checkpoint_sha256",
                "best_f1_checkpoint_path", "best_f1_checkpoint_sha256")}, "test_evaluation": False})
    return output


def write_reports(comparison, paired, summaries, strengths, hybrid, preflight, manifest) -> None:
    study = runner.STUDY
    train_sha = sorted({row.get("source_git_sha") for row in comparison
                        if row.get("model") == runner.STATIC_MODEL})
    lines = ["# MR-CGE V1.3 Static-Strength Calibration Screen",
        "", f"Base SHA: `{runner.BASE_SHA}`.",
        f"Training source SHA: `{train_sha[0] if train_sha else 'not yet frozen'}`.",
        f"Physical GPU: `{runner.PHYSICAL_GPU}`. Test evaluation: disabled. Link prediction: disabled.",
        "", "## Validation comparison", "",
        "Accuracy/F1 are shown as percentages; CE is the original cross-entropy value. Each row is paired by the same dataset and seed.", "",
        "| Dataset | Seed | V1.2 Acc/F1/CE | Static Acc/F1/CE | Joint View Acc/F1/CE |", "|---|---:|---:|---:|---:|"]
    by_key = {}
    for row in comparison:
        by_key.setdefault((row["dataset"], int(row["seed"])), {})[row["display_name"]] = row
    for dataset in DATASETS:
        seeds = (42, 43, 44) if dataset == "Movies" else (42, 43)
        for seed in seeds:
            values = by_key[(dataset, seed)]
            formatted = []
            for name in ("V1.2 M_repaired", "V1.3 Static-Strength", "V1.3-A Joint View"):
                row = values[name]
                formatted.append(f"{100*float(row['val_acc']):.2f} / {100*float(row['val_macro_f1']):.2f} / {float(row['val_ce']):.4f}")
            lines.append(f"| {dataset} | {seed} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")
    lines.extend(["", "## Paired differences", "",
        "Accuracy and Macro-F1 differences are percentage points (pp); CE differences are raw CE units. Means and population SDs are across the listed seeds, with no significance test.", ""])
    for dataset in DATASETS:
        lines.extend([f"### {dataset}", "", "| Contrast | Seeds | Δ Accuracy (pp), mean ± SD | Δ Macro-F1 (pp), mean ± SD | Δ CE, mean ± SD |",
                      "|---|---|---:|---:|---:|"])
        for contrast in ("Static-Strength minus V1.2", "Joint View minus V1.2", "Joint View minus Static-Strength"):
            selected = [row for row in paired if row["dataset"] == dataset and row["contrast"] == contrast]
            n = len(selected)
            acc = [100*float(row["delta_val_acc"]) for row in selected]
            f1 = [100*float(row["delta_val_macro_f1"]) for row in selected]
            ce = [float(row["delta_val_ce"]) for row in selected]
            lines.append(f"| {contrast} | {','.join(str(row['seed']) for row in selected)} | {statistics.mean(acc):+.2f} ± {statistics.pstdev(acc):.2f} | {statistics.mean(f1):+.2f} ± {statistics.pstdev(f1):.2f} | {statistics.mean(ce):+.4f} ± {statistics.pstdev(ce):.4f} |")
    lines.extend(["", "## Static mechanism and cost", "",
        "Strengths are evaluated at each Best-Accuracy checkpoint. `sP` and `sS` are global within modality; effective strengths multiply the learned V1.3-A `rho`/`gamma`. Early Cross residual ratios use the frozen model's calibrated Cross-to-physical-neighbor RMS diagnostic.", "",
        "| Dataset | Seed | sP Text/Visual | sS Text/Visual | rho·sP Text/Visual | gamma·sS Text/Visual | Model params | Peak reserved (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in strengths:
        if row.get("model") != runner.STATIC_MODEL:
            continue
        lines.append(f"| {row['dataset']} | {row['seed']} | {float(row['sP_text']):.4f} / {float(row['sP_visual']):.4f} | {float(row['sS_text']):.4f} / {float(row['sS_visual']):.4f} | {float(row['rho_sP_text']):.4f} / {float(row['rho_sP_visual']):.4f} | {float(row['gamma_sS_text']):.4f} / {float(row['gamma_sS_visual']):.4f} | {row['model_parameters']} | {int(row['peak_reserved_bytes'])/1024**3:.2f} |")
    lines.extend(["", "The companion `strength_diagnostics.csv` includes weighted Private/Collaborative Physical RMS, weighted Semantic RMS, Early Cross residual/delta, Late residual ratio, and the frozen Joint View node-wise mean/SD/quantiles. The result does not identify the pure causal effect of node-wise reliability: Static adds four parameters on top of learned `rho/gamma`, so parameterization and optimization trajectories also differ.",
        "", "## Fixed-split Hybrid-CGE context", "",
        "This is performance context for Movies only and is excluded from Static-vs-Joint architectural attribution.", "",
        "| Seed | Hybrid Acc | Macro-F1 | CE | Total parameters | Peak reserved (GiB) |", "|---:|---:|---:|---:|---:|---:|"])
    for row in hybrid:
        lines.append(f"| {row['seed']} | {100*float(row['val_acc']):.2f}% | {100*float(row['val_macro_f1']):.2f}% | {float(row['val_ce']):.4f} | {row['total_parameters']} | {int(float(row['peak_reserved_bytes']))/1024**3:.2f} |")
    completed = int(manifest.get("completed_cells", 0))
    lines.extend(["", "## Status and interpretation", "",
        f"Successful authorized cells: {completed}/7; failed attempts: {manifest.get('failed_attempts', 0)}; resource-blocked attempts: {manifest.get('blocked_resource_attempts', 0)}; not yet attempted: {manifest.get('unstarted_cells', 7)}.", "",
        "Interpret the paired seed-level pattern across datasets. Do not select a winner from one seed, infer statistical significance, or attribute a Static-vs-Joint difference solely to node-wise reliability. This screen uses two seeds outside Movies and historical reference campaigns; optimization paths and small-sample variation remain limitations.", "",
        "## Audit constraints", "",
        "All data use frozen splits and V7B.1 K=8 semantic caches. Test labels were masked before Trainer entry; no Test metrics or link prediction were run. New training was isolated to physical GPU 1 in one-cell subprocesses with a fresh ownership guard and at least 8 GiB free at launch, plus a 2 GiB margin above the measured peak-reserved memory.", ""])
    (study / "SCREENING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    preflight_report = ["# Preflight report", "",
        f"CPU/data preflight: `{preflight.get('status')}` at {preflight.get('recorded_utc')}.",
        f"GPU preflight status: `{preflight.get('gpu_preflight_status', 'pending')}`.",
        f"Base: `{runner.BASE_SHA}`; training source: `{train_sha[0] if train_sha else 'not frozen'}`.", "",
        "## Data protocol", ""]
    for dataset, info in preflight.get("dataset_records", {}).items():
        preflight_report.append(f"- **{dataset}**: nodes={info.get('num_nodes')}, edges={info.get('num_edges')}, classes={info.get('num_classes')}, Text/Visual dims={info.get('text_dim')}/{info.get('visual_dim')}; split `{info.get('split_fingerprint')}`, split-file SHA `{info.get('split_file_sha256')}`, semantic-cache SHA `{info.get('semantic_cache_sha256')}`; Test masked={info.get('test_labels_masked_before_trainer')}; Test evaluation={info.get('test_evaluation')}.")
    preflight_report.extend(["", "## Per-cell GPU preflight", ""])
    for row in preflight.get("gpu_preflight_cells", []):
        preflight_report.append(f"- {row.get('dataset')} seed {row.get('seed')}: `{row.get('status')}`; peak reserved={row.get('peak_reserved_bytes', 'n/a')} bytes; GPU={row.get('gpu_name', 'n/a')}; reason={row.get('error', '')}.")
    (study / "PREFLIGHT_REPORT.md").write_text("\n".join(preflight_report) + "\n", encoding="utf-8")

    data_audit = ["# Data protocol audit", "",
        "The runner resolves each dataset's configured fixed split path and verifies its loaded split fingerprint against the frozen study value and the same-seed V1.3-A record. Feature, physical-edge, split-file, V7B.1 candidate, V7B, and semantic-cache hashes are checked before training. Automatic split creation is disabled. Test labels are replaced with -100 before the Trainer; Test evaluation and LP are disabled.", "",
        "| Dataset | Nodes | Text dim | Visual dim | Split fingerprint | Split SHA256 | Cache SHA256 |", "|---|---:|---:|---:|---|---|---|"]
    for dataset, info in preflight.get("dataset_records", {}).items():
        data_audit.append(f"| {dataset} | {info['num_nodes']} | {info['text_dim']} | {info['visual_dim']} | `{info['split_fingerprint']}` | `{info['split_file_sha256']}` | `{info['semantic_cache_sha256']}` |")
    (study / "DATA_PROTOCOL_AUDIT.md").write_text("\n".join(data_audit) + "\n", encoding="utf-8")

    implementation = ["# Implementation audit", "",
        "Static-Strength subclasses the frozen V1.3-A M_repaired backbone and mirrors its Forward locally so modality_id is explicit at both Private and Collaborative scale calls. The only new trainable tensor is `static_logits[2,2]`; it starts at zero and maps through `2*sigmoid`. The inherited Joint View MLP is constructed inside its isolated RNG scope to preserve the common backbone and external classifier initialization, then removed from the parameter tree. Private uses `sP*rho*R_P + sS*gamma*R_S`; Collaborative adds `rho*sP*(R_P_collab-R_P_priv)` and reuses the same modality-level scales. No module hooks, global mutable mode, or dataset-name dispatch is used.", "",
        "The static tests cover factory/config gating, exact shared initialization and external head initialization, initial Eval/Train equivalence, paired-dropout replay, common gradients, nonzero Static task gradients, modality/relation separation, node constancy, identical Private/Collaborative scales, both Early Cross directions, no-support behavior, low-memory checkpointing, strict checkpoint reload, and validation RNG preservation. Historical V1.3-A, NoP0, Hybrid-CGE source and result files are hash-checked and read-only.", "",
        f"Training source SHA: `{train_sha[0] if train_sha else 'not frozen'}`. Static model parameters: four beyond the V1.3-A common model, with Joint View removed.", ""]
    (study / "IMPLEMENTATION_AUDIT.md").write_text("\n".join(implementation), encoding="utf-8")

    handoff = ["# Decision handoff", "",
        "This screening is complete for the seven authorized Validation cells only. Results are descriptive, with paired same-seed comparisons and population SDs; no significance tests were performed. Static-vs-Joint is not a pure causal test of node-wise reliability because Static adds four free scales while retaining rho/gamma, and training trajectories differ.", "",
        "Use the per-seed tables and diagnostics to judge whether Joint View repeatedly improves over Static and V1.2 across datasets, while considering parameter and memory costs. Movies Hybrid-CGE is a separate fixed-split performance reference. No winner is automatically selected.", "",
        "No Test, LP, additional seeds/datasets, NoP0, Early Cross disentangling, Static+Joint combination, or V1.4 work was performed. The branch is ready for independent review.", ""]
    (study / "DECISION_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")


def analyze() -> dict[str, object]:
    preflight = runner.read_json(runner.DATA / "preflight.json", {})
    if preflight.get("status") != "pass" or preflight.get("gpu_preflight_status") != "pass":
        raise RuntimeError("analysis requires passing CPU and all seven matching-source GPU preflights")
    protected = preflight.get("protected_historical_file_sha256s", {})
    runner.verify_protected_history_file_hashes(protected)
    datasets_by_seed = {"Movies": (42, 43, 44), "Grocery": (42, 43), "ele-fashion": (42, 43)}
    records: dict[tuple[str, int], dict[str, dict]] = {}
    comparison: list[dict[str, object]] = []
    static_runs: dict[tuple[str, int], dict[str, str]] = {}
    references: dict[tuple[str, int], tuple[dict[str, str], dict[str, str]]] = {}
    for dataset, seeds in datasets_by_seed.items():
        for seed in seeds:
            v12, v13 = history_refs(dataset, seed)
            static = static_run(dataset, seed)
            if static.get("source_git_sha") != preflight.get("training_source_sha"):
                raise RuntimeError(f"Static source SHA differs from frozen training SHA for {dataset}/seed{seed}")
            key = (dataset, seed)
            references[key] = (v12, v13)
            static_runs[key] = static
            bundle = {
                "v12": canonical(v12, "mrcge_v12", "M_repaired", "V1.2 M_repaired", "frozen_selected_reference"),
                "joint": canonical(v13, "mrcge_v13", "joint_view", "V1.3-A Joint View", "frozen_reference"),
                "static": canonical(static, runner.STATIC_MODEL, runner.STATIC_CONDITION,
                                     "V1.3 Static-Strength", "new_training"),
            }
            records[key] = bundle
            comparison.extend(bundle.values())
    paired = paired_rows(records, datasets_by_seed)
    summaries = summarize(comparison, paired, datasets_by_seed)
    strengths = make_strength_diagnostics(static_runs, references)
    hybrid = make_hybrid_rows()

    runner.write_rows(runner.DATA / "model_comparison.csv", comparison,
                      list(dict.fromkeys([*COMPARE_FIELDS, "source", "display_name"])))
    runner.write_rows(runner.DATA / "paired_differences.csv", paired,
                      ("dataset", "seed", "contrast", "model_a", "model_b", "delta_val_acc", "delta_val_acc_pp",
                       "delta_val_macro_f1", "delta_val_macro_f1_pp", "delta_val_ce", "delta_val_ce_pp", "test_evaluation"))
    runner.write_rows(runner.DATA / "dataset_summary.csv", summaries)
    runner.write_rows(runner.DATA / "strength_diagnostics.csv", strengths)
    runner.write_rows(runner.DATA / "hybrid_fixed_split_reference.csv", hybrid)

    resource_path = runner.DATA / "resource_summary.csv"
    resource_rows = runner.read_rows(resource_path)
    resource_rows = [row for row in resource_rows if row.get("record_type") != "training_peak_summary"]
    for key, run in static_runs.items():
        resource_rows.append({"record_type": "training_peak_summary", "model": runner.STATIC_MODEL,
            "condition": runner.STATIC_CONDITION, "dataset": key[0], "seed": key[1],
            "attempt": run["attempt"], "phase": "formal_training_peak", "physical_gpu": runner.PHYSICAL_GPU,
            "gpu_name": run.get("gpu_name", ""), "peak_allocated_bytes": run.get("peak_allocated_bytes", ""),
            "peak_reserved_bytes": run.get("peak_reserved_bytes", ""), "wall_seconds": run.get("wall_seconds", ""),
            "test_evaluation": False})
    runner.write_rows(resource_path, resource_rows)

    all_runs = runner.read_rows(runner.DATA / "runs.csv")
    manifest = runner.read_json(runner.DATA / "manifest.json", {})
    completed = {(row.get("model"), row.get("condition"), row.get("dataset"), row.get("seed"))
                 for row in all_runs if row.get("status") == "complete"}
    manifest.update({"completed_cells": len(completed),
        "failed_attempts": sum(row.get("status") == "failed" for row in all_runs),
        "blocked_resource_attempts": sum(row.get("status") == "blocked_resource" for row in all_runs),
        "unstarted_cells": max(0, len(TARGETS) - len(completed)),
        "training_source_sha": preflight.get("training_source_sha"),
        "analysis_completed_utc": runner.now(), "test_evaluation": False, "link_prediction": False})
    runner.json_write(runner.DATA / "manifest.json", manifest)
    write_reports(comparison, paired, summaries, strengths, hybrid, preflight, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
