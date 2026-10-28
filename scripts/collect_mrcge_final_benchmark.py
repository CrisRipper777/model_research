#!/usr/bin/env python3
"""Collect completed MR-CGE Final cells and compute three-seed sample SD tables."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mrcge_final_benchmark_utils import (  # noqa: E402
    LP_DATASETS, NC_DATASETS, OUTPUT_ROOT, SEEDS, sha256_file,
)

RESEARCH = ROOT / "research/mrcge_final_benchmark"
DATA = RESEARCH / "data"


def _completed_results(phase: str, datasets: tuple[str, ...]) -> list[dict]:
    rows = []
    for dataset in datasets:
        for seed in SEEDS:
            parent = OUTPUT_ROOT / phase / dataset / f"seed_{seed}"
            matches = []
            for attempt_dir in parent.glob("attempt_*"):
                status_path = attempt_dir / "status.json"
                result_path = attempt_dir / "result.json"
                if not status_path.is_file() or not result_path.is_file():
                    continue
                status = json.loads(status_path.read_text(encoding="utf-8"))
                result = json.loads(result_path.read_text(encoding="utf-8"))
                checkpoint = Path(result.get("checkpoint_path", ""))
                if status.get("status") != "complete" or not checkpoint.is_file():
                    continue
                if sha256_file(checkpoint) != result.get("checkpoint_sha256"):
                    raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
                if result.get("dataset") != dataset or int(result.get("seed", -1)) != seed:
                    raise RuntimeError(f"result identity mismatch in {result_path}")
                matches.append(result)
            if len(matches) > 1:
                raise RuntimeError(f"multiple complete results exist for {phase}/{dataset}/seed{seed}")
            if matches:
                rows.append(matches[0])
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict], datasets: tuple[str, ...], phase: str) -> list[dict]:
    metric_names = (
        ("test_accuracy", "test_macro_f1") if phase == "nc"
        else ("test_mrr", "test_hits_at_1", "test_hits_at_3", "test_hits_at_10")
    )
    result = []
    for dataset in datasets:
        selected = sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: int(row["seed"]))
        by_seed = {int(row["seed"]): row for row in selected}
        output = {
            "dataset": dataset,
            "n": len(selected),
            "seeds": ";".join(str(seed) for seed in sorted(by_seed)),
            "status": "complete" if set(by_seed) == set(SEEDS) else "partial",
        }
        for metric in metric_names:
            values = []
            for row in selected:
                block = row["test"]
                key = "accuracy" if metric == "test_accuracy" else "macro_f1" if metric == "test_macro_f1" else metric.removeprefix("test_").replace("hits_at_", "hits@")
                values.append(float(block[key]))
            output[f"{metric}_mean"] = statistics.mean(values) if values else ""
            # Sample standard deviation is undefined with fewer than two seeds.
            output[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) >= 2 else ""
        result.append(output)
    return result


def _nc_run_row(result: dict) -> dict:
    val, test, split = result["validation"], result["test"], result["split"]
    return {
        "dataset": result["dataset"], "seed": result["seed"], "status": "complete",
        "model_version": result["model_version"], "source_commit": result["source_commit"],
        "source_model_commit": result["source_model_commit"], "config_path": result["config_path"],
        "selection": result["selection"], "best_epoch": result["best_epoch"],
        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
        "val_cross_entropy": val["cross_entropy"], "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"], "checkpoint_path": result["checkpoint_path"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_reload_verified": result["checkpoint_reload_verified"],
        "test_labels_masked_during_training": result["test_labels_masked_during_training"],
        "split_path": split["split_path"],
        "split_sha256": split["split_sha256"], "split_fingerprint": result["split"].get("split_fingerprint", ""),
        "feature_fingerprint": split["feature_fingerprint"],
        "physical_edge_fingerprint": result["physical_edge_fingerprint"],
        "semantic_cache_path": result["semantic_cache_path"], "semantic_cache_sha256": result["semantic_cache_sha256"],
        "candidate_fingerprint": result["candidate_fingerprint"], "v7b_fingerprint": result["v7b_fingerprint"],
        "source_commit": result["source_commit"],
    }


def _lp_run_row(result: dict) -> dict:
    val, test, split = result["validation"], result["test"], result["split"]
    return {
        "dataset": result["dataset"], "seed": result["seed"], "status": "complete",
        "model_version": result["model_version"], "protocol_version": result["protocol_version"],
        "split_protocol": result["split_protocol"],
        "source_commit": result["source_commit"], "source_model_commit": result["source_model_commit"],
        "config_path": result["config_path"],
        "selection": result["selection"], "best_epoch": result["best_epoch"],
        "val_mrr": val["mrr"], "val_hits_at_1": val["hits@1"],
        "val_hits_at_3": val["hits@3"], "val_hits_at_10": val["hits@10"],
        "test_mrr": test["mrr"], "test_hits_at_1": test["hits@1"],
        "test_hits_at_3": test["hits@3"], "test_hits_at_10": test["hits@10"],
        "test_negative_candidates": result["test_negative_candidates"],
        "test_evaluation_count": result["test_evaluation_count"],
        "checkpoint_path": result["checkpoint_path"], "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_reload_verified": result["checkpoint_reload_verified"],
        "positive_edge_mask_backend": result["positive_edge_mask_backend"],
        "split_path": split["split_path"], "split_sha256": split["split_sha256"],
        "split_fingerprint": split["split_fingerprint"], "feature_fingerprint": split["feature_fingerprint"],
        "physical_edge_fingerprint": result["physical_edge_fingerprint"],
        "semantic_cache_path": result["semantic_cache_path"], "semantic_cache_sha256": result["semantic_cache_sha256"],
        "candidate_fingerprint": result["candidate_fingerprint"], "v7b_fingerprint": result["v7b_fingerprint"],
        "semantic_context_approximation": result["semantic_source_physical_context"]["approximation"],
        "source_commit": result["source_commit"],
    }


def _fmt(mean, sd) -> str:
    if mean == "" or sd == "":
        return "Pending"
    return f"{100 * float(mean):.2f} ± {100 * float(sd):.2f}"


def _write_tables(nc_summary: list[dict], lp_summary: list[dict]) -> None:
    nc = {row["dataset"]: row for row in nc_summary}
    nc_lines = [
        "# Table 1. Node classification", "",
        "Values are mean ± sample standard deviation over seeds 42, 43, and 44. All values use the Best-Validation-Accuracy checkpoint; Macro-F1 uses every configured class ID.", "",
        "| Model | Movies | Grocery | Toys | Reddit-S | ele-fashion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metric in (("Test Accuracy", "test_accuracy"), ("Test Macro-F1", "test_macro_f1")):
        cells = [label]
        for dataset in NC_DATASETS:
            row = nc[dataset]
            cells.append(_fmt(row.get(f"{metric}_mean", ""), row.get(f"{metric}_sample_sd", "")))
        nc_lines.append("| MR-CGE Final (V1.3-A Full), " + " | ".join(cells) + " |")
    (RESEARCH / "TABLE_1_NC.md").write_text("\n".join(nc_lines) + "\n", encoding="utf-8")

    lp = {row["dataset"]: row for row in lp_summary}
    lp_lines = [
        "# Table 2. Link prediction", "",
        "Values are mean ± sample standard deviation over seeds 42, 43, and 44. MR-CGE Final uses Best-Validation-MRR selection and official fixed Test negatives; official split overlap/self-loop effects are accepted as-is.", "",
        "| Model | Dataset | Test MRR | Hits@1 | Hits@3 | Hits@10 | Comparability |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for dataset in LP_DATASETS:
        row = lp[dataset]
        values = [_fmt(row.get(f"{metric}_mean", ""), row.get(f"{metric}_sample_sd", ""))
                  for metric in ("test_mrr", "test_hits_at_1", "test_hits_at_3", "test_hits_at_10")]
        lp_lines.append(f"| MR-CGE Final (V1.3-A Full) | {dataset} | " + " | ".join(values) + " | Final sampled protocol |")
    lp_lines.extend([
        "", "Historical V1.3 NC5/LP2 records are Validation-only and use a three-fold full-graph edge-holdout protocol, so they are not strict Test comparisons. No other model's Test result with this split, feature version, metric definition, and protocol was found in the repository.",
    ])
    (RESEARCH / "TABLE_2_LP.md").write_text("\n".join(lp_lines) + "\n", encoding="utf-8")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    nc_results = _completed_results("nc", NC_DATASETS)
    lp_results = _completed_results("lp", LP_DATASETS)
    nc_rows = [_nc_run_row(row) for row in nc_results]
    lp_rows = [_lp_run_row(row) for row in lp_results]
    _write_csv(DATA / "nc_runs.csv", list(nc_rows[0]) if nc_rows else [
        "dataset", "seed", "status", "model_version", "source_commit", "source_model_commit", "config_path",
        "selection", "best_epoch", "val_accuracy", "val_macro_f1", "val_cross_entropy",
        "test_accuracy", "test_macro_f1", "checkpoint_path", "checkpoint_sha256", "checkpoint_reload_verified",
        "test_labels_masked_during_training", "split_path", "split_sha256",
        "split_fingerprint", "feature_fingerprint", "physical_edge_fingerprint", "semantic_cache_path",
        "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    ], nc_rows)
    _write_csv(DATA / "lp_runs.csv", list(lp_rows[0]) if lp_rows else [
        "dataset", "seed", "status", "model_version", "protocol_version", "split_protocol", "source_commit", "source_model_commit",
        "config_path", "selection", "best_epoch", "val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10",
        "test_mrr", "test_hits_at_1", "test_hits_at_3", "test_hits_at_10", "test_negative_candidates", "test_evaluation_count",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_reload_verified", "positive_edge_mask_backend",
        "split_path", "split_sha256", "split_fingerprint", "feature_fingerprint", "physical_edge_fingerprint",
        "semantic_cache_path", "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
        "semantic_context_approximation",
    ], lp_rows)
    nc_summary = _summary_rows(nc_results, NC_DATASETS, "nc")
    lp_summary = _summary_rows(lp_results, LP_DATASETS, "lp")
    _write_csv(DATA / "nc_summary.csv", list(nc_summary[0]) if nc_summary else [
        "dataset", "n", "seeds", "status", "test_accuracy_mean", "test_accuracy_sample_sd",
        "test_macro_f1_mean", "test_macro_f1_sample_sd",
    ], nc_summary)
    _write_csv(DATA / "lp_summary.csv", list(lp_summary[0]) if lp_summary else [
        "dataset", "n", "seeds", "status", "test_mrr_mean", "test_mrr_sample_sd",
        "test_hits_at_1_mean", "test_hits_at_1_sample_sd", "test_hits_at_3_mean", "test_hits_at_3_sample_sd",
        "test_hits_at_10_mean", "test_hits_at_10_sample_sd",
    ], lp_summary)
    _write_tables(nc_summary, lp_summary)
    print(f"Collected NC={len(nc_rows)}/15, LP={len(lp_rows)}/6 completed cells.")
    print("Three-seed summary standard deviations use the sample estimator (ddof=1).")


if __name__ == "__main__":
    main()
