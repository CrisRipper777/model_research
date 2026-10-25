#!/usr/bin/env python3
"""Audit MR-CGE V1.3 records and produce same-seed Validation comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any

import torch
from torch import nn

from scripts import run_mrcge_v13_screen as runner

DATA = ROOT / "research/mrcge_v13/data"
OUTPUT = ROOT / "outputs/mrcge_v13_screen/runs"
V12_SELECTED = ROOT / "research/mrcge_v12_confirmation/data/combined_selected_runs.csv"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("joint_view", "sender_private")
SEEDS = (42, 43)
BASE_SHA = "a7930442e85ff9450fded54d5869915b434c7431"
TRAINING_SOURCE_SHA = "169cddd72aaaba00a5ce3fd25657981dd40c5e27"
FROZEN_HASHES = {
    "src/models/mrcge_v12.py": "d9a7632f725a7c742dc083fe7737bc5c7384e9fa3bc91ffa64ead01e4a5d4ca6",
    "configs/model/mrcge_v12.yaml": "c0e5782fee74bda70a6cc42c3cf50b57e169ccf00981e53f595bd68b04750eed",
    "src/tasks/nc.py": "362350f07bb2d076b0f4f791872fbf0105ba695681bc15b3f2a09353bc6aacfd",
    "scripts/run_mrcge_v12_confirmation.py": "cbf6b6eff5f772f4e3a5d024454fa76294f27774cd667dccce5e8bccab59739e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_file_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(["git", "show", f"{commit}:{relative_path}"], cwd=ROOT,
                            check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def close(actual: Any, expected: Any, tolerance: float = 3.0e-6) -> bool:
    try:
        return math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def selected_v12_rows() -> dict[tuple[str, int], dict[str, str]]:
    rows = [row for row in read_csv(V12_SELECTED)
            if row.get("condition") == "M_repaired"
            and row.get("dataset") in DATASETS and int(row.get("seed", -1)) in SEEDS
            and row.get("selected_for_combined", "").lower() == "true"]
    selected = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]))
        if key in selected:
            raise RuntimeError(f"duplicate V1.2 selected reference row {key}")
        selected[key] = row
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if set(selected) != expected:
        raise RuntimeError(f"V1.2 selected reference matrix mismatch: {set(selected) ^ expected}")
    return selected


def validate_and_build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_rows = read_csv(DATA / "runs.csv")
    target = {(dataset, variant, seed) for dataset in DATASETS for variant in VARIANTS for seed in SEEDS}
    attempts: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in run_rows:
        key = (row.get("dataset", ""), row.get("variant", ""),
               int(row.get("seed", -1) or -1), int(row.get("attempt", -1) or -1))
        if key in attempts:
            raise RuntimeError(f"duplicate immutable run attempt row {key}")
        attempts[key] = row

    complete: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in run_rows:
        if row.get("status") != "complete":
            continue
        key = (row["dataset"], row["variant"], int(row["seed"]))
        if key not in target:
            raise RuntimeError(f"unauthorized completed study cell {key}")
        if key in complete:
            raise RuntimeError(f"more than one completed attempt for cell {key}")
        complete[key] = row
    if set(complete) != target:
        raise RuntimeError(f"completed matrix incomplete: missing={sorted(target - set(complete))}")

    frozen_mismatches = {}
    for relative, expected_sha in FROZEN_HASHES.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected_sha:
            frozen_mismatches[relative] = {"expected": expected_sha, "actual": actual}
    if frozen_mismatches:
        raise RuntimeError(f"frozen V1.2 files changed: {frozen_mismatches}")
    diagnostic_model_sha = sha256_file(ROOT / "src/models/mrcge_v13.py")
    current_trainer_sha = sha256_file(ROOT / "src/tasks/nc.py")
    v12 = selected_v12_rows()
    preflight = json.loads((DATA / "preflight.json").read_text(encoding="utf-8"))
    frozen_data = {(row["dataset"], int(row["seed"])): row for row in preflight.get("datasets", [])}
    verified_caches = runner.cache_entries()
    metrics = ("val_acc", "val_macro_f1", "val_ce")
    best_f1_metrics = ("best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce")
    paired: list[dict[str, Any]] = []
    checkpoint_audits = []
    successful_history_keys = set()

    for key, row in sorted(complete.items()):
        dataset, variant, seed = key
        attempt = int(row["attempt"])
        attempt_dir = Path(row["attempt_dir"])
        config_path = attempt_dir / "config_snapshot.yaml"
        metrics_path = attempt_dir / "run_metrics.json"
        acc_path = Path(row["checkpoint_path"])
        f1_path = Path(row["best_f1_checkpoint_path"])
        for name, path, expected_sha in (
            ("accuracy_checkpoint", acc_path, row["checkpoint_sha256"]),
            ("f1_checkpoint", f1_path, row["best_f1_checkpoint_sha256"]),
        ):
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise RuntimeError(f"{key} {name} is missing or SHA256 differs: {path}")
        if not config_path.is_file() or sha256_file(config_path) != row["config_sha256"]:
            raise RuntimeError(f"{key} config snapshot SHA256 differs")
        if row["base_sha"] != BASE_SHA:
            raise RuntimeError(f"{key} has unexpected V1.2 base SHA")
        if row.get("test_evaluation", "").lower() != "false" or row.get("validation_rng_unchanged", "").lower() != "true":
            raise RuntimeError(f"{key} Test-off or Validation RNG audit failed")
        source_sha = row["source_git_sha"]
        subprocess.run(["git", "cat-file", "-e", f"{source_sha}^{{commit}}"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        source_model_sha = git_file_sha256(source_sha, "src/models/mrcge_v13.py")
        if row["model_source_sha256"] != source_model_sha or row["trainer_source_sha256"] != current_trainer_sha:
            raise RuntimeError(f"{key} model/trainer source hash differs from the recorded training commit")

        run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if run_metrics.get("run_seeds") != [seed] or run_metrics.get("protocol_version") != "unified_full_graph_nc_v1":
            raise RuntimeError(f"{key} run metrics provenance mismatch")
        metadata = run_metrics["runs"][0]["metadata"]
        dual = metadata["dual_checkpoint_selection"]
        history = metadata.get("epoch_history", [])
        if not history or len(history) != int(metadata.get("epochs_completed", -1)):
            raise RuntimeError(f"{key} checkpoint metadata has incomplete epoch history")
        history_key = (dataset, variant, seed, attempt)
        successful_history_keys.add(history_key)
        csv_history = [r for r in read_csv(DATA / "epoch_history.csv")
                       if r.get("dataset") == dataset and r.get("variant") == variant
                       and int(r.get("seed", -1)) == seed and int(r.get("attempt", -1)) == attempt]
        if len(csv_history) != len(history):
            raise RuntimeError(f"{key} CSV epoch history row count differs from run metadata")

        provenance = {
            "dataset": dataset, "variant": variant, "backbone_condition": "M_repaired",
            "seed": seed, "attempt": attempt, "source_git_sha": source_sha,
            "base_sha": BASE_SHA, "model_source_sha256": source_model_sha,
            "task_source_sha256": current_trainer_sha,
            "feature_fingerprint": row["feature_fingerprint"],
            "dataset_fingerprint": row["dataset_fingerprint"],
            "physical_edge_fingerprint": row["physical_edge_fingerprint"],
            "split_fingerprint": row["split_fingerprint"],
            "split_file_sha256": row["split_file_sha256"],
            "semantic_cache_sha256": row["semantic_cache_sha256"],
            "candidate_fingerprint": row["candidate_fingerprint"],
            "v7b_fingerprint": row["v7b_fingerprint"],
            "test_evaluation": False, "test_label_values_indexed": False,
            "link_prediction": False,
        }
        expected_provenance = {**provenance, "config_sha256": row["config_sha256"],
                               "sender_private_ratio": 0.3 if variant == "sender_private" else 0.0}
        for checkpoint_kind, path, selection, epoch, field_map in (
            ("best_accuracy", acc_path, "best_val_accuracy", int(row["best_acc_epoch"]),
             {"val_acc": "val_acc", "val_macro_f1": "val_macro_f1", "val_ce": "val_ce"}),
            ("best_f1", f1_path, "best_val_macro_f1_within_accuracy_stopping_horizon",
             int(row["best_f1_epoch"]),
             {"val_acc": "best_f1_val_acc", "val_macro_f1": "best_f1_val_macro_f1", "val_ce": "best_f1_val_ce"}),
        ):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("task") != "nc" or checkpoint.get("protocol_version") != "unified_full_graph_nc_v1":
                raise RuntimeError(f"{key} {checkpoint_kind} checkpoint task/protocol mismatch")
            if int(checkpoint.get("seed", -1)) != seed or int(checkpoint.get("epoch", -1)) != epoch:
                raise RuntimeError(f"{key} {checkpoint_kind} checkpoint seed/epoch mismatch")
            if checkpoint.get("selection") != selection:
                raise RuntimeError(f"{key} {checkpoint_kind} checkpoint selection mismatch")
            checkpoint_meta = checkpoint["run_metadata"]
            study_provenance = checkpoint_meta.get("study_provenance", {})
            for field, expected in expected_provenance.items():
                if study_provenance.get(field) != expected:
                    raise RuntimeError(f"{key} {checkpoint_kind} provenance {field} mismatch")
            selection_record = dual["best_accuracy" if checkpoint_kind == "best_accuracy" else "best_macro_f1"]
            if int(selection_record["epoch"]) != epoch:
                raise RuntimeError(f"{key} {checkpoint_kind} run-metric epoch mismatch")
            for metric, row_field in field_map.items():
                expected = float(row[row_field])
                saved_metric = float(checkpoint["metrics"][metric])
                selected_metric = float(selection_record["metrics"][metric])
                if not close(saved_metric, expected) or not close(selected_metric, expected):
                    raise RuntimeError(f"{key} {checkpoint_kind} reload/selection {metric} mismatch")
            checkpoint_audits.append({
                "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                "checkpoint_kind": checkpoint_kind, "checkpoint_path": str(path),
                "checkpoint_sha256": sha256_file(path), "selection": selection,
                "epoch": epoch, "metadata_metrics_match_run_row": True,
                "study_provenance_matches_run_row": True,
            })
            del checkpoint

        reference = v12[(dataset, seed)]
        fingerprint_pairs = (
            ("feature_fingerprint", "feature_fingerprint"),
            ("dataset_fingerprint", "dataset_fingerprint"),
            ("physical_edge_fingerprint", "physical_edge_fingerprint"),
            ("split_fingerprint", "split_fingerprint"),
            ("semantic_cache_sha256", "semantic_cache_sha256"),
            ("v7b_fingerprint", "v7b_fingerprint"),
        )
        for current_field, base_field in fingerprint_pairs:
            if reference.get(base_field) and row[current_field] != reference[base_field]:
                raise RuntimeError(f"{key} differs from same-seed V1.2 fingerprint {current_field}")
        cpu_record = frozen_data.get((dataset, seed))
        cpu_pairs = (
            ("feature_fingerprint", "feature_fingerprint"),
            ("dataset_fingerprint", "dataset_fingerprint"),
            ("physical_edge_fingerprint", "physical_edge_fingerprint"),
            ("split_fingerprint", "split_fingerprint"),
            ("split_file_sha256", "split_file_sha256"),
            ("semantic_cache_sha256", "semantic_cache_sha256"),
            ("v7b_fingerprint", "v7b_fingerprint"),
        )
        if cpu_record is None:
            raise RuntimeError(f"{key} is missing its committed CPU preflight data row")
        for current_field, preflight_field in cpu_pairs:
            if row[current_field] != cpu_record.get(preflight_field):
                raise RuntimeError(f"{key} differs from committed CPU preflight {current_field}")
        verified_cache = verified_caches[dataset]
        for field in ("feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint"):
            if row[field] != verified_cache.get(field):
                raise RuntimeError(f"{key} differs from the verified V7B.1 cache {field}")
        if reference.get("split_file_sha256") and row["split_file_sha256"] != reference["split_file_sha256"]:
            raise RuntimeError(f"{key} split-file SHA differs from V1.2 reference record")
        paired_row: dict[str, Any] = {
            "dataset": dataset, "seed": seed, "variant": variant, "attempt": attempt,
            "base_condition": "M_repaired", "base_source_campaign": reference.get("source_campaign", ""),
            "base_source_git_sha": reference.get("source_git_sha", ""),
            "v13_source_git_sha": source_sha, "v13_checkpoint_sha256": row["checkpoint_sha256"],
            "v13_best_f1_checkpoint_sha256": row["best_f1_checkpoint_sha256"],
            "test_evaluation": False,
        }
        for metric in metrics:
            base_value = float(reference[metric])
            candidate_value = float(row[metric])
            paired_row[f"base_{metric}"] = base_value
            paired_row[f"v13_{metric}"] = candidate_value
            paired_row[f"delta_{metric}"] = candidate_value - base_value
        paired_row["delta_val_acc_pp"] = 100.0 * paired_row["delta_val_acc"]
        paired_row["delta_val_macro_f1_pp"] = 100.0 * paired_row["delta_val_macro_f1"]
        for metric in best_f1_metrics:
            paired_row[f"base_{metric}"] = float(reference[metric])
            paired_row[f"v13_{metric}"] = float(row[metric])
            paired_row[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
        paired_row.update({
            "v12_best_acc_epoch": int(reference["best_acc_epoch"]),
            "v12_best_f1_epoch": int(reference["best_f1_epoch"]),
            "v13_best_acc_epoch": int(row["best_acc_epoch"]),
            "v13_best_f1_epoch": int(row["best_f1_epoch"]),
            "v12_total_parameters": int(reference["total_parameters"]),
            "v13_total_parameters": int(row["total_parameters"]),
            "v12_wall_seconds": float(reference["wall_seconds"]),
            "v13_wall_seconds": float(row["wall_seconds"]),
            "v13_peak_reserved_bytes": int(row["peak_reserved_bytes"]),
        })
        paired.append(paired_row)

    history_rows = read_csv(DATA / "epoch_history.csv")
    for row in history_rows:
        key = (row.get("dataset", ""), row.get("variant", ""), int(row.get("seed", -1)),
               int(row.get("attempt", -1)))
        if key not in successful_history_keys:
            raise RuntimeError(f"orphaned epoch-history row outside successful attempts: {key}")
    return paired, {
        "successful_cells": len(complete), "target_cells": len(target),
        "source_git_shas": sorted({r["source_git_sha"] for r in complete.values()}),
        "training_model_source_sha256": sorted({r["model_source_sha256"] for r in complete.values()}),
        "diagnostic_model_source_sha256": diagnostic_model_sha,
        "trainer_source_sha256": current_trainer_sha,
        "frozen_v12_hashes_verified": True,
        "checkpoint_audits": checkpoint_audits,
        "checkpoint_count_verified": len(checkpoint_audits),
        "epoch_history_rows": len(history_rows),
        "completed_attempts": len(complete),
        "failed_attempt_rows": sum(r.get("status") == "failed" for r in run_rows),
        "blocked_resource_attempt_rows": sum(r.get("status") == "blocked_resource" for r in run_rows),
    }


def summarize(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[(row["dataset"], row["variant"])].append(row)
    for dataset in DATASETS:
        base_rows = [r for r in read_csv(V12_SELECTED)
                     if r.get("dataset") == dataset and r.get("condition") == "M_repaired"
                     and r.get("selected_for_combined", "").lower() == "true"
                     and int(r.get("seed", -1)) in SEEDS]
        if len(base_rows) != len(SEEDS):
            raise RuntimeError(f"{dataset}: expected two V1.2 reference seeds")
        all_conditions = [("V1.2_M_repaired", base_rows)]
        all_conditions.extend((variant, groups[(dataset, variant)]) for variant in VARIANTS)
        for condition, rows in all_conditions:
            if len(rows) != 2:
                raise RuntimeError(f"{dataset}/{condition}: expected two corresponding seed rows")
            if condition == "V1.2_M_repaired":
                acc = [float(r["val_acc"]) for r in rows]
                f1 = [float(r["val_macro_f1"]) for r in rows]
                ce = [float(r["val_ce"]) for r in rows]
                f1_acc = [float(r["best_f1_val_acc"]) for r in rows]
                f1_f1 = [float(r["best_f1_val_macro_f1"]) for r in rows]
                f1_ce = [float(r["best_f1_val_ce"]) for r in rows]
                delta_acc = delta_f1 = delta_ce = []
                base_parameters = [int(r["total_parameters"]) for r in rows]
                parameters = base_parameters
                wall = [float(r["wall_seconds"]) for r in rows]
                peak = [int(r["peak_reserved_bytes"]) for r in rows]
            else:
                acc = [float(r["v13_val_acc"]) for r in rows]
                f1 = [float(r["v13_val_macro_f1"]) for r in rows]
                ce = [float(r["v13_val_ce"]) for r in rows]
                f1_acc = [float(r["v13_best_f1_val_acc"]) for r in rows]
                f1_f1 = [float(r["v13_best_f1_val_macro_f1"]) for r in rows]
                f1_ce = [float(r["v13_best_f1_val_ce"]) for r in rows]
                delta_acc = [float(r["delta_val_acc"]) for r in rows]
                delta_f1 = [float(r["delta_val_macro_f1"]) for r in rows]
                delta_ce = [float(r["delta_val_ce"]) for r in rows]
                parameters = [int(r["v13_total_parameters"]) for r in rows]
                wall = [float(r["v13_wall_seconds"]) for r in rows]
                peak = [int(r["v13_peak_reserved_bytes"]) for r in rows]
            record: dict[str, Any] = {
                "dataset": dataset, "condition": condition,
                "variant": "base" if condition == "V1.2_M_repaired" else condition,
                "n_seeds": 2, "seeds": "42;43",
                "mean_val_acc": fmean(acc), "sd_val_acc": pstdev(acc),
                "mean_val_macro_f1": fmean(f1), "sd_val_macro_f1": pstdev(f1),
                "mean_val_ce": fmean(ce), "sd_val_ce": pstdev(ce),
                "mean_best_f1_val_acc": fmean(f1_acc), "mean_best_f1_val_macro_f1": fmean(f1_f1),
                "mean_best_f1_val_ce": fmean(f1_ce),
                "mean_total_parameters": fmean(parameters), "mean_wall_seconds": fmean(wall),
                "mean_peak_reserved_mib": fmean(peak) / (1024 * 1024),
                "mean_delta_val_acc": fmean(delta_acc) if delta_acc else "",
                "sd_delta_val_acc": pstdev(delta_acc) if delta_acc else "",
                "mean_delta_val_acc_pp": 100.0 * fmean(delta_acc) if delta_acc else "",
                "mean_delta_val_macro_f1": fmean(delta_f1) if delta_f1 else "",
                "sd_delta_val_macro_f1": pstdev(delta_f1) if delta_f1 else "",
                "mean_delta_val_macro_f1_pp": 100.0 * fmean(delta_f1) if delta_f1 else "",
                "mean_delta_val_ce": fmean(delta_ce) if delta_ce else "",
                "sd_delta_val_ce": pstdev(delta_ce) if delta_ce else "",
                "interpretation": "descriptive two-seed mean; historical V1.2 is a separate campaign",
            }
            result.append(record)
    return result


def summarize_resources() -> list[dict[str, Any]]:
    snapshots = read_csv(DATA / "resource_summary.csv")
    preflight = json.loads((DATA / "preflight.json").read_text(encoding="utf-8"))
    passing = {(r["dataset"], r["variant"], int(r["seed"])): r
               for r in preflight.get("gpu_preflight_cells", [])
               if r.get("attempt") == 2 and r.get("status") == "pass"}
    runs = [r for r in read_csv(DATA / "runs.csv") if r.get("status") == "complete"]
    records = []
    for run in runs:
        key = (run["dataset"], run["variant"], int(run["seed"]))
        p = passing.get(key)
        if p is None:
            raise RuntimeError(f"{key}: missing corrected-source GPU preflight row")
        own = [r for r in snapshots if r.get("dataset") == key[0] and r.get("variant") == key[1]
               and int(r.get("seed", -1) or -1) == key[2]
               and int(r.get("attempt", -1) or -1) == int(run["attempt"])]
        records.append({
            "dataset": key[0], "variant": key[1], "seed": key[2], "attempt": int(run["attempt"]),
            "physical_gpu": 1, "gpu_name": "NVIDIA GeForce RTX 3090",
            "gpu_preflight_peak_allocated_bytes": p["peak_allocated_bytes"],
            "gpu_preflight_peak_reserved_bytes": p["peak_reserved_bytes"],
            "required_start_free_mib_with_2gib_margin": p["required_start_free_mib_with_2gib_margin"],
            "formal_peak_allocated_bytes": run["peak_allocated_bytes"],
            "formal_peak_reserved_bytes": run["peak_reserved_bytes"],
            "formal_wall_seconds": run["wall_seconds"],
            "resource_snapshots_for_attempt": len(own),
            "fresh_guard_checks": sum(r.get("guard_status") == "pass" for r in own),
            "guard_blocked": False, "test_evaluation": False,
        })
    return records


def posthoc_validation_diagnostics(paired: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly reload each Best-Accuracy checkpoint on CPU and recalculate Validation stats."""
    run_map = {(r["dataset"], r["variant"], int(r["seed"])): r
               for r in read_csv(DATA / "runs.csv") if r.get("status") == "complete"}
    caches = runner.cache_entries()
    rows: list[dict[str, Any]] = []
    for pair in paired:
        dataset, variant, seed = pair["dataset"], pair["variant"], int(pair["seed"])
        run = run_map[(dataset, variant, seed)]
        cfg, safe, _, aligned = runner.load_safe_data(dataset, variant, seed, caches[dataset], "cpu")
        runner.set_seed(seed)
        model = runner.build_model(cfg, runner.data_info(safe))
        model.load_semantic_cache(
            caches[dataset]["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=caches[dataset]["feature_fingerprint"],
            expected_candidate_fingerprint=caches[dataset]["candidate_fingerprint"],
            expected_v7b_fingerprint=caches[dataset]["v7b_fingerprint"],
        )
        checkpoint = torch.load(Path(run["checkpoint_path"]), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        classifier = nn.Linear(model.out_dim, int(safe.num_classes))
        classifier.load_state_dict(checkpoint["head_state"], strict=True)
        model.eval(); classifier.eval()
        model._capture_state_audit = True
        rng_before = torch.random.get_rng_state().clone()
        with torch.no_grad():
            embeddings, _, _, _, _ = model(safe.x, safe.edge_index, return_details=False)
            logits = classifier(embeddings)
        rng_unchanged = torch.equal(rng_before, torch.random.get_rng_state())
        if not rng_unchanged:
            raise RuntimeError(f"{dataset}/{variant}/seed{seed} posthoc Validation changed RNG")
        metrics = runner.v12_utils.metric_bundle(
            logits, safe.y, safe.val_idx, list(range(int(safe.num_classes)))
        )
        for metric, field in (("acc", "val_acc"), ("macro_f1", "val_macro_f1"), ("ce", "val_ce")):
            if not close(metrics[metric], float(run[field])):
                raise RuntimeError(f"{dataset}/{variant}/seed{seed} CPU reload {metric} differs from run record")
        epoch_diagnostics = model.collect_epoch_diagnostics()
        record: dict[str, Any] = {
            "dataset": dataset, "variant": variant, "seed": seed,
            "attempt": int(run["attempt"]), "scope": "posthoc_best_accuracy_validation",
            "training_source_git_sha": run["source_git_sha"],
            "diagnostic_source_git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "diagnostic_model_source_sha256": sha256_file(ROOT / "src/models/mrcge_v13.py"),
            "split_fingerprint": run["split_fingerprint"],
            "split_file_sha256": aligned["split_file_sha256"],
            "reload_metrics_match_run_row": True, "validation_rng_unchanged": rng_unchanged,
            "reloaded_val_acc": metrics["acc"], "reloaded_val_macro_f1": metrics["macro_f1"],
            "reloaded_val_ce": metrics["ce"],
        }
        weighted_keys = (
            "val_v13_weighted_physical_private_contribution_rms_text",
            "val_v13_weighted_semantic_private_contribution_rms_text",
            "val_v13_weighted_physical_collaborative_contribution_rms_text",
            "val_v13_weighted_physical_private_contribution_rms_visual",
            "val_v13_weighted_semantic_private_contribution_rms_visual",
            "val_v13_weighted_physical_collaborative_contribution_rms_visual",
            "val_v13_early_cross_raw_rms_text", "val_v13_early_cross_calibrated_rms_text",
            "val_v13_early_cross_to_neighbor_rms_text", "val_v13_late_residual_ratio_text",
            "val_v13_early_cross_raw_rms_visual", "val_v13_early_cross_calibrated_rms_visual",
            "val_v13_early_cross_to_neighbor_rms_visual", "val_v13_late_residual_ratio_visual",
            "val_v13_private_sP_mean_text", "val_v13_private_sS_mean_text",
            "val_v13_private_sP_mean_visual", "val_v13_private_sS_mean_visual",
            "val_v13_collaborative_vs_private_sP_abs_mean_text",
            "val_v13_collaborative_vs_private_sP_abs_mean_visual",
        )
        for key in weighted_keys:
            if key in epoch_diagnostics:
                record[key.removeprefix("val_v13_")] = epoch_diagnostics[key]
        rows.append(record)
        del model, classifier, embeddings, logits, checkpoint, safe
    metadata = {
        "status": "pass", "strict_checkpoint_reloads": len(rows),
        "validation_metrics_match_run_rows": True, "validation_rng_unchanged": True,
        "diagnostic_source_git_sha": rows[0]["diagnostic_source_git_sha"] if rows else "",
        "diagnostic_model_source_sha256": rows[0]["diagnostic_model_source_sha256"] if rows else "",
        "weighted_contribution_diagnostics_recomputed": len(rows) * 6,
        "test_evaluation": False,
    }
    return rows, metadata


def main() -> None:
    paired, audit = validate_and_build_rows()
    posthoc_rows, posthoc_audit = posthoc_validation_diagnostics(paired)
    summary = summarize(paired)
    resource_by_cell = summarize_resources()
    paired_fields = list(paired[0])
    summary_fields = list(summary[0])
    resource_fields = list(resource_by_cell[0])
    write_csv(DATA / "paired_differences.csv", paired, paired_fields)
    write_csv(DATA / "dataset_summary.csv", summary, summary_fields)
    write_csv(DATA / "resource_by_cell.csv", resource_by_cell, resource_fields)
    write_csv(DATA / "weighted_contribution_diagnostics.csv", posthoc_rows, list(posthoc_rows[0]))
    audit["posthoc_validation_reload_audit"] = posthoc_audit
    (DATA / "checkpoint_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "completed_cells": audit["successful_cells"],
        "successful_unique_cells": audit["successful_cells"],
        "failed_attempt_rows": audit["failed_attempt_rows"],
        "blocked_resource_attempt_rows": audit["blocked_resource_attempt_rows"],
        "failed_cells": 0, "blocked_resource_cells": 0, "unstarted_cells": 0,
        "checkpoint_sha_and_metadata_audit": "pass",
        "same_seed_v12_fingerprint_audit": "pass",
        "frozen_v12_files_audit": "pass",
        "posthoc_validation_reload_audit": "pass",
        "test_evaluation": False, "link_prediction": False,
        "implementation_source_git_sha": TRAINING_SOURCE_SHA,
        "formal_training_source_git_sha": audit["source_git_shas"][0] if len(audit["source_git_shas"]) == 1 else audit["source_git_shas"],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "audit_status": "pass", "completed_cells": audit["successful_cells"],
        "paired_rows": len(paired), "dataset_summary_rows": len(summary),
        "verified_checkpoints": audit["checkpoint_count_verified"],
        "posthoc_reloaded_checkpoints": posthoc_audit["strict_checkpoint_reloads"],
        "failed_attempt_rows": audit["failed_attempt_rows"],
        "blocked_resource_attempt_rows": audit["blocked_resource_attempt_rows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
