#!/usr/bin/env python3
"""Collect and audit NC5/LP2 artifacts. This command never launches training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mrcge_v13_nc5_lp2_utils import (  # noqa: E402
    BASE_SHA, DATA_DIR, HISTORICAL_NC_DATASETS, LP_DATASETS, NC_DATASETS,
    NEW_NC_DATASETS, OUTPUT, SEEDS, csv_read, csv_write, json_write,
    now_utc, sha256_file,
)
from src.models import LinkPredictor  # noqa: E402,F401  (documents the saved LP decoder type)
from src.tasks.mrcge_v13_full_graph_lp import PROTOCOL_VERSION as LP_PROTOCOL  # noqa: E402


HISTORICAL_RUNS = ROOT / "research/mrcge_v13_seed44_completion/data/runs.csv"
NC_NEW = DATA_DIR / "nc_new_runs.csv"
LP_NEW = DATA_DIR / "lp_runs.csv"


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _verify_file(path_value: Any, expected_sha: Any, label: str) -> tuple[bool, str]:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return False, f"{label} missing: {path}"
    actual = sha256_file(path)
    if actual != str(expected_sha or ""):
        return False, f"{label} SHA256 mismatch: {path}"
    return True, ""


def _verify_prepared_nc(dataset: str, row: dict[str, Any], label: str, problems: list[str]) -> None:
    """Compare a run with preparation using file hashes across historical hash schemes."""
    fingerprints_path = DATA_DIR / "fingerprints.json"
    if not fingerprints_path.is_file():
        problems.append(f"{label}: prepared fingerprints.json is missing")
        return
    prepared_all = json.loads(fingerprints_path.read_text(encoding="utf-8"))
    prepared = prepared_all.get("datasets", {}).get(dataset)
    if not prepared:
        problems.append(f"{label}: no prepared NC fingerprints for {dataset}")
        return
    expected = {
        "split_sha256": prepared.get("split", {}).get("sha256"),
        "feature_fingerprint": prepared.get("feature_fingerprint"),
        "semantic_cache_sha256": prepared.get("semantic_cache_sha256"),
        "candidate_fingerprint": prepared.get("candidate_fingerprint"),
        "v7b_fingerprint": prepared.get("v7b_fingerprint"),
    }
    # New records share the preparation's label-free split fingerprint and canonical
    # physical fingerprint. Historical V1.3 rows used label-aware/directed tensor hashes;
    # exact split/cache file SHA256s provide the cross-version identity check.
    if label.startswith("new NC"):
        expected["split_fingerprint"] = prepared.get("split", {}).get("split_fingerprint")
        expected["physical_edge_fingerprint"] = prepared.get("physical_edge_fingerprint")
    for field, target in expected.items():
        actual = row.get(field, "")
        if field == "split_sha256" and not actual:
            actual = row.get("split_file_sha256", "")
        if target and str(actual) != str(target):
            problems.append(f"{label}: {dataset} {field} differs from prepared provenance")

    okay, error = _verify_file(
        prepared.get("split", {}).get("path"), prepared.get("split", {}).get("sha256"),
        f"prepared {dataset} NC split",
    )
    if not okay:
        problems.append(error)
    okay, error = _verify_file(
        prepared.get("semantic_cache_path"), prepared.get("semantic_cache_sha256"),
        f"prepared {dataset} semantic cache",
    )
    if not okay:
        problems.append(error)
    for source_key, digest_key in (
        ("feature_source_text", "feature_file_sha256_text"),
        ("feature_source_visual", "feature_file_sha256_visual"),
    ):
        okay, error = _verify_file(prepared.get(source_key), prepared.get(digest_key), f"prepared {dataset} {source_key}")
        if not okay:
            problems.append(error)
    cache_metadata = prepared.get("cache_metadata", {})
    if cache_metadata.get("physical_edge_fingerprint") != prepared.get("physical_edge_fingerprint"):
        problems.append(f"{label}: prepared cache physical graph fingerprint is inconsistent")
    if cache_metadata.get("feature_fingerprint") != prepared.get("feature_fingerprint"):
        problems.append(f"{label}: prepared cache feature fingerprint is inconsistent")


def _verify_prepared_lp(dataset: str, row: dict[str, Any], problems: list[str]) -> None:
    fingerprints_path = DATA_DIR / "fingerprints.json"
    if not fingerprints_path.is_file():
        problems.append("LP run has no prepared fingerprints.json")
        return
    payload = json.loads(fingerprints_path.read_text(encoding="utf-8"))
    if not payload.get("prepared_phases", {}).get("lp", {}).get("complete"):
        problems.append(f"LP run exists without a complete LP preparation phase: {dataset}")
    prepared = payload.get("datasets", {}).get(dataset)
    if not prepared:
        problems.append(f"LP run has no prepared provenance for {dataset}")
        return
    expected = {
        "split_sha256": prepared.get("split", {}).get("sha256"),
        "train_graph_fingerprint": prepared.get("physical_edge_fingerprint"),
        "feature_fingerprint": prepared.get("feature_fingerprint"),
        "semantic_cache_sha256": prepared.get("semantic_cache_sha256"),
        "candidate_fingerprint": prepared.get("candidate_fingerprint"),
        "v7b_fingerprint": prepared.get("v7b_fingerprint"),
    }
    for field, target in expected.items():
        if target and str(row.get(field, "")) != str(target):
            problems.append(f"LP {dataset} {field} differs from prepared provenance")
    for path, digest, label in (
        (prepared.get("split", {}).get("path"), prepared.get("split", {}).get("sha256"), "LP official split"),
        (prepared.get("semantic_cache_path"), prepared.get("semantic_cache_sha256"), "LP semantic cache"),
    ):
        okay, error = _verify_file(path, digest, f"prepared {dataset} {label}")
        if not okay:
            problems.append(error)

    for source_key, digest_key in (
        ("feature_source_text", "feature_file_sha256_text"),
        ("feature_source_visual", "feature_file_sha256_visual"),
    ):
        okay, error = _verify_file(prepared.get(source_key), prepared.get(digest_key), f"prepared {dataset} {source_key}")
        if not okay:
            problems.append(error)
    cache_metadata = prepared.get("cache_metadata", {})
    if cache_metadata.get("physical_edge_fingerprint") != prepared.get("physical_edge_fingerprint"):
        problems.append(f"LP {dataset} cache physical graph fingerprint is inconsistent")
    if cache_metadata.get("feature_fingerprint") != prepared.get("feature_fingerprint"):
        problems.append(f"LP {dataset} cache feature fingerprint is inconsistent")


def _historical_nc_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows = csv_read(HISTORICAL_RUNS)
    problems = []
    selected = [row for row in rows if row.get("model") == "mrcge_v13"
                and row.get("condition") == "M_repaired" and row.get("variant") == "joint_view"
                and float(row.get("sender_private_ratio", 0.0) or 0.0) == 0.0
                and row.get("record_origin") in {"reused_historical", "new_training"}]
    # The checked historical campaign has these nine Full records. Only the three requested
    # historical datasets are reused; other rows, if any, do not enter this cohort.
    output = []
    for dataset in HISTORICAL_NC_DATASETS:
        for seed in SEEDS:
            matches = [row for row in selected if row.get("dataset") == dataset and int(row.get("seed", -1)) == seed]
            if len(matches) != 1:
                problems.append(f"expected exactly one historical Full row for {dataset}/seed{seed}; got {len(matches)}")
                continue
            source = matches[0]
            row_problem_count = len(problems)
            if source.get("status") != "complete" or source.get("selection_checkpoint") != "best_validation_accuracy":
                problems.append(f"historical Full status/selection invalid for {dataset}/seed{seed}")
            if _is_true(source.get("test_evaluation")):
                problems.append(f"historical row unexpectedly evaluated Test for {dataset}/seed{seed}")
            if source.get("test_labels_masked") not in ("True", "true"):
                problems.append(f"historical Test labels were not recorded as masked for {dataset}/seed{seed}")
            if source.get("checkpoint_reload_verified") not in ("True", "true"):
                problems.append(f"historical checkpoint reload was not verified for {dataset}/seed{seed}")
            okay, error = _verify_file(source.get("checkpoint_path"), source.get("checkpoint_sha256"), "historical NC checkpoint")
            if not okay:
                problems.append(error)
            okay, error = _verify_file(source.get("best_f1_checkpoint_path"), source.get("best_f1_checkpoint_sha256"), "historical Best-F1 checkpoint")
            if not okay:
                problems.append(error)
            _verify_prepared_nc(dataset, source, f"historical NC {dataset}/seed{seed}", problems)
            for metric in ("val_acc", "val_macro_f1", "val_ce"):
                if _float(source, metric) is None:
                    problems.append(f"historical {dataset}/seed{seed} has invalid {metric}")
            output.append({
                "dataset": dataset, "seed": seed,
                "status": "complete" if len(problems) == row_problem_count else "invalid_historical",
                "attempt": "",
                "record_origin": "reused_historical", "selection": "best_validation_accuracy",
                "best_epoch": source.get("best_acc_epoch", ""), "val_acc": source.get("val_acc", ""),
                "val_macro_f1": source.get("val_macro_f1", ""), "val_ce": source.get("val_ce", ""),
                "train_acc": source.get("train_acc", ""), "train_ce": source.get("train_ce", ""),
                "checkpoint_path": source.get("checkpoint_path", ""), "checkpoint_sha256": source.get("checkpoint_sha256", ""),
                "best_f1_checkpoint_path": source.get("best_f1_checkpoint_path", ""),
                "best_f1_checkpoint_sha256": source.get("best_f1_checkpoint_sha256", ""),
                "split_path": source.get("split_file_path", ""), "split_sha256": source.get("split_file_sha256", ""),
                "split_fingerprint": source.get("split_fingerprint", ""),
                "split_fingerprint_scheme": "historical_label_aware_partition_tensor_fingerprint",
                "feature_fingerprint": source.get("feature_fingerprint", ""),
                "physical_edge_fingerprint": source.get("physical_edge_fingerprint", ""),
                "semantic_cache_path": "", "semantic_cache_sha256": source.get("semantic_cache_sha256", ""),
                "candidate_fingerprint": source.get("candidate_fingerprint", ""), "v7b_fingerprint": source.get("v7b_fingerprint", ""),
                "test_evaluation": False, "test_labels_masked": True,
                "checkpoint_reload_verified": source.get("checkpoint_reload_verified", ""),
                "physical_edge_fingerprint_scheme": "historical_directed_tensor_fingerprint",
                "source_commit": source.get("source_git_sha", ""), "wall_seconds": "", "peak_memory_bytes": "",
                "gpu_name": "", "attempt_dir": "", "error": "",
            })
    return output, problems


def _collect_nc_new() -> tuple[list[dict[str, Any]], list[str]]:
    rows = csv_read(NC_NEW)
    problems = []
    complete: dict[tuple[str, int], dict[str, Any]] = {}
    output = []
    expected = {(dataset, seed) for dataset in NEW_NC_DATASETS for seed in SEEDS}
    for source in rows:
        row = dict(source)
        try:
            seed = int(row.get("seed", -1))
        except (TypeError, ValueError):
            seed = -1
        key = (row.get("dataset", ""), seed)
        if key not in expected:
            problems.append(f"unexpected NC run row: {key}")
            continue
        if row.get("status") == "complete":
            if key in complete:
                problems.append(f"duplicate successful NC run row: {key}")
                continue
            complete[key] = row
            if row.get("record_origin") != "new_training" or row.get("selection") != "best_validation_accuracy":
                problems.append(f"new NC {key} did not use the designated Best-Validation-Accuracy selection")
                row["status"] = "protocol_mismatch"
                complete.pop(key, None)
            okay, error = _verify_file(row.get("checkpoint_path"), row.get("checkpoint_sha256"), "new NC checkpoint")
            if not okay:
                problems.append(error)
                row["status"] = "invalid_checkpoint"
                complete.pop(key, None)
            okay, error = _verify_file(row.get("best_f1_checkpoint_path"), row.get("best_f1_checkpoint_sha256"), "new NC Best-F1 checkpoint")
            if not okay:
                problems.append(error)
                row["status"] = "invalid_best_f1_checkpoint"
                complete.pop(key, None)
            if row.get("test_evaluation") != "False" or row.get("test_labels_masked") != "True":
                problems.append(f"new NC Test protocol mismatch for {key}")
                row["status"] = "protocol_mismatch"
                complete.pop(key, None)
            if row.get("checkpoint_reload_verified") != "True":
                problems.append(f"new NC checkpoint reload was not verified for {key}")
                row["status"] = "checkpoint_reload_unverified"
                complete.pop(key, None)
            _verify_prepared_nc(key[0], row, f"new NC {key}", problems)
            for metric in ("val_acc", "val_macro_f1", "val_ce"):
                if _float(row, metric) is None:
                    problems.append(f"new NC {key} has invalid {metric}")
                    row["status"] = "invalid_metrics"
                    complete.pop(key, None)
        output.append(row)
    return output, problems


def _collect_lp() -> tuple[list[dict[str, Any]], list[str]]:
    rows = csv_read(LP_NEW)
    problems = []
    expected = {(dataset, seed) for dataset in LP_DATASETS for seed in SEEDS}
    seen = set()
    output = []
    for source in rows:
        row = dict(source)
        try:
            seed = int(row.get("seed", -1))
        except (ValueError, TypeError):
            seed = -1
        key = (row.get("dataset", ""), seed)
        if key not in expected:
            problems.append(f"unexpected LP run row: {key}")
            continue
        if row.get("status") == "complete":
            if key in seen:
                problems.append(f"duplicate successful LP row: {key}")
                continue
            seen.add(key)
            if row.get("protocol_version") != LP_PROTOCOL:
                problems.append(f"LP protocol version mismatch for {key}")
                row["status"] = "protocol_mismatch"
            if row.get("test_evaluation") != "False" or row.get("test_metrics") != "not_evaluated":
                problems.append(f"LP Test was not explicitly disabled for {key}")
                row["status"] = "protocol_mismatch"
            if row.get("checkpoint_reload_verified") != "True":
                problems.append(f"LP checkpoint reload was not verified for {key}")
                row["status"] = "checkpoint_reload_unverified"
            for metric in ("val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10"):
                if _float(row, metric) is None:
                    problems.append(f"LP {key} has invalid {metric}")
                    row["status"] = "invalid_metrics"
            _verify_prepared_lp(key[0], row, problems)
            okay, error = _verify_file(row.get("checkpoint_path"), row.get("checkpoint_sha256"), "LP checkpoint")
            if not okay:
                problems.append(error)
                row["status"] = "invalid_checkpoint"
            else:
                checkpoint = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=False)
                meta = checkpoint.get("metadata", {})
                expected_meta = {
                    "protocol_version": LP_PROTOCOL, "seed": seed, "fold_seed": 42, "fold_count": 3,
                    "test_evaluation": False, "test_metrics": "not_evaluated", "checkpoint_reload_verified": True,
                }
                mismatch = {key_name: (meta.get(key_name), value) for key_name, value in expected_meta.items()
                            if meta.get(key_name) != value}
                if mismatch:
                    problems.append(f"LP checkpoint metadata mismatch for {key}: {mismatch}")
                    row["status"] = "protocol_mismatch"
                for state_key in ("model_state", "projection_state", "decoder_state"):
                    if not isinstance(checkpoint.get(state_key), dict) or not checkpoint[state_key]:
                        problems.append(f"LP checkpoint missing {state_key} for {key}")
                        row["status"] = "invalid_checkpoint"
        output.append(row)
    return output, problems


def _summary(rows: list[dict[str, Any]], datasets: tuple[str, ...], metrics: tuple[str, ...], expected_seeds: tuple[int, ...], source_key: str | None = None):
    summary = []
    for dataset in datasets:
        cohort = [row for row in rows if row.get("dataset") == dataset and row.get("status") == "complete"]
        seeds = [int(row["seed"]) for row in cohort]
        row: dict[str, Any] = {
            "dataset": dataset, "n": len(cohort), "seeds": ";".join(map(str, sorted(seeds))),
            "status": "complete" if set(seeds) == set(expected_seeds) else "partial",
        }
        if source_key:
            row[source_key] = "reused_historical" if dataset in HISTORICAL_NC_DATASETS else "new_training"
        for metric in metrics:
            values = [_float(value, metric) for value in cohort]
            values = [value for value in values if value is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else ""
            row[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) >= 2 else (0.0 if len(values) == 1 else "")
            for seed in expected_seeds:
                found = next((item for item in cohort if int(item["seed"]) == seed), None)
                row[f"seed{seed}_{metric}"] = _float(found, metric) if found else ""
        summary.append(row)
    return summary


def finalize() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    historical, historical_problems = _historical_nc_rows()
    new_nc, new_nc_problems = _collect_nc_new()
    lp_rows, lp_problems = _collect_lp()
    all_nc = historical + new_nc
    nc_map = {(row.get("dataset"), int(row.get("seed", -1))): row for row in all_nc if row.get("status") == "complete"}
    expected_nc = {(dataset, seed) for dataset in NC_DATASETS for seed in SEEDS}
    expected_lp = {(dataset, seed) for dataset in LP_DATASETS for seed in SEEDS}
    nc_complete = set(nc_map) == expected_nc
    lp_complete = {(row.get("dataset"), int(row.get("seed", -1))) for row in lp_rows if row.get("status") == "complete"} == expected_lp
    for row in new_nc:
        row.setdefault("physical_edge_fingerprint_scheme", "canonical_undirected_unique_edges_sha256")
        row.setdefault("split_fingerprint_scheme", "label_free_sorted_partition_id_sha256")

    nc_fields = [
        "dataset", "seed", "status", "attempt", "record_origin", "selection", "best_epoch",
        "val_acc", "val_macro_f1", "val_ce", "train_acc", "train_ce", "checkpoint_path",
        "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
        "split_path", "split_sha256", "split_fingerprint", "feature_fingerprint",
        "physical_edge_fingerprint", "semantic_cache_path", "semantic_cache_sha256",
        "candidate_fingerprint", "v7b_fingerprint", "test_evaluation", "test_labels_masked",
        "checkpoint_reload_verified", "physical_edge_fingerprint_scheme", "split_fingerprint_scheme", "source_commit", "wall_seconds", "peak_memory_bytes",
        "gpu_name", "attempt_dir", "error",
    ]
    nc_summary = _summary(all_nc, NC_DATASETS, ("val_acc", "val_macro_f1", "val_ce"), SEEDS, "record_origin")
    lp_summary = _summary(lp_rows, LP_DATASETS,
                           ("val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10"), SEEDS)
    for row in lp_summary:
        row["protocol_version"] = LP_PROTOCOL
        row["protocol"] = "Full-Graph Three-Fold Edge-Holdout; differs from unified_sampled_lp_v1"
        row["test_metrics"] = "not_evaluated"
    csv_write(DATA_DIR / "nc_runs.csv", all_nc, nc_fields)
    nc_summary_fields = list(dict.fromkeys(key for row in nc_summary for key in row))
    csv_write(DATA_DIR / "nc_summary.csv", nc_summary, nc_summary_fields)
    lp_fields = [
        "dataset", "seed", "status", "attempt", "protocol_version", "best_epoch",
        "val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10",
        "checkpoint_path", "checkpoint_sha256", "split_path", "split_sha256",
        "train_graph_fingerprint", "feature_fingerprint", "semantic_cache_path",
        "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
        "fold_seed", "fold_count", "test_evaluation", "test_metrics", "checkpoint_reload_verified",
        "source_commit", "wall_seconds", "peak_memory_bytes", "gpu_name", "attempt_dir", "error",
    ]
    csv_write(DATA_DIR / "lp_runs.csv", lp_rows, lp_fields)
    lp_summary_fields = list(dict.fromkeys(key for row in lp_summary for key in row))
    csv_write(DATA_DIR / "lp_summary.csv", lp_summary, lp_summary_fields)

    fingerprints_path = DATA_DIR / "fingerprints.json"
    fingerprints = json.loads(fingerprints_path.read_text()) if fingerprints_path.is_file() else {"datasets": {}, "prepared_phases": {}}
    lp_preflight_path = DATA_DIR / "lp_split_preflight_audit.json"
    lp_preflight = json.loads(lp_preflight_path.read_text()) if lp_preflight_path.is_file() else {
        "status": "not_run", "datasets": {},
    }
    protocol_audit = {
        "base_sha": BASE_SHA, "protocol_version_lp": LP_PROTOCOL,
        "nc_expected_cells": len(expected_nc), "nc_complete_cells": len(set(nc_map) & expected_nc),
        "lp_expected_cells": len(expected_lp),
        "lp_complete_cells": len({(row.get("dataset"), int(row.get("seed", -1))) for row in lp_rows if row.get("status") == "complete"} & expected_lp),
        "nc_uses_reused_historical_9_rows": len([row for row in all_nc if row.get("record_origin") == "reused_historical"]),
        "new_nc_required_cells": len(NEW_NC_DATASETS) * len(SEEDS),
        "new_nc_completed_cells": len([row for row in new_nc if row.get("status") == "complete"]),
        "all_nc_full_model_joint_view_m_repaired": all(
            row.get("record_origin") != "reused_historical" or row.get("selection") == "best_validation_accuracy"
            for row in all_nc
        ),
        "fixed_nc_split_seed": 42, "lp_fold_seed": 42, "lp_fold_count": 3,
        "lp_train_graph_source": "LP Train Positive only; both directions; validation/test positives excluded",
        "lp_semantic_source": "global node attributes, V7B.1 K=8; no LP edge labels used for candidate construction",
        "lp_split_preflight_status": lp_preflight.get("status", "not_run"),
        "lp_split_preflight_blocked_datasets": sorted(lp_preflight.get("datasets", {})),
        "test_evaluation": False, "test_metrics": "not_evaluated",
        "sampled_lp_protocol_equivalence_claimed": False,
        "status": "pass" if not historical_problems and not new_nc_problems and not lp_problems else "issues",
        "issues": [*historical_problems, *new_nc_problems, *lp_problems],
    }
    json_write(DATA_DIR / "protocol_audit.json", protocol_audit)
    status = "complete" if nc_complete and lp_complete and protocol_audit["status"] == "pass" else "partial"
    failures = [row for row in [*new_nc, *lp_rows] if row.get("status") not in {"complete"}]
    missing_cells = [
        {"phase": "nc", "dataset": dataset, "seed": seed, "status": "missing"}
        for dataset, seed in sorted(expected_nc - set(nc_map))
    ] + [
        {"phase": "lp", "dataset": dataset, "seed": seed, "status": "missing"}
        for dataset, seed in sorted(expected_lp - {
            (row.get("dataset"), int(row.get("seed", -1))) for row in lp_rows if row.get("status") == "complete"
        })
    ]
    manifest = {
        "study": "MR-CGE V1.3-A five-dataset NC and two-dataset LP evaluation",
        "branch": "exp/mrcge_v13_nc5_lp2_fullgraph", "base_sha": BASE_SHA,
        "status": status, "nc_complete": nc_complete, "lp_complete": lp_complete,
        "nc_expected_cells": len(expected_nc), "nc_completed_cells": len(set(nc_map) & expected_nc),
        "lp_expected_cells": len(expected_lp),
        "lp_completed_cells": protocol_audit["lp_complete_cells"],
        "formal_training_launched_by_collector": False,
        "evaluate_test": False, "test_metrics": "not_evaluated",
        "historical_nc_rows_reused": 9, "new_nc_rows_expected": 6, "lp_rows_expected": 6,
        "lp_protocol_version": LP_PROTOCOL,
        "failures_or_incomplete_rows": [*failures, *missing_cells],
        "lp_split_preflight_status": lp_preflight.get("status", "not_run"),
        "lp_split_preflight_blocked_datasets": sorted(lp_preflight.get("datasets", {})),
        "historical_integrity_issues": historical_problems,
        "protocol_audit_issues": protocol_audit["issues"],
        "fingerprints_file": str(fingerprints_path) if fingerprints_path.is_file() else "not_prepared",
        "finalized_utc": now_utc(),
    }
    json_write(DATA_DIR / "manifest.json", manifest)
    print(f"COLLECTED status={status} nc={manifest['nc_completed_cells']}/{manifest['nc_expected_cells']} "
          f"lp={manifest['lp_completed_cells']}/{manifest['lp_expected_cells']} "
          "test=not_evaluated; no training started", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true", required=True)
    args = parser.parse_args()
    if args.finalize:
        finalize()


if __name__ == "__main__":
    main()
