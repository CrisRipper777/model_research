from __future__ import annotations

import csv
import math

from scripts.analyze_mag_information_utility_v1 import (
    CONDITIONS,
    DATASETS,
    SEEDS,
    _load_history,
    _paired_rows,
    _render_report,
)


def _synthetic_runs():
    runs = {}
    for dataset_index, dataset in enumerate(DATASETS):
        for condition_index, condition in enumerate(CONDITIONS):
            for seed_index, seed in enumerate(SEEDS):
                accuracy = 0.70 + dataset_index * 0.02 + seed_index * 0.001 + condition_index * 0.01
                train = {"accuracy": accuracy + 0.05, "macro_f1": accuracy - 0.02, "cross_entropy": 0.7}
                validation = {"accuracy": accuracy, "macro_f1": accuracy - 0.10, "cross_entropy": 1.2 - condition_index * 0.02}
                f1_validation = {"accuracy": accuracy - 0.01, "macro_f1": accuracy - 0.05, "cross_entropy": 1.1}
                runs[(dataset, condition, seed)] = {
                    "best_accuracy_checkpoint_metrics": {"train": train, "validation": validation},
                    "best_f1_checkpoint_metrics": {"train": train, "validation": f1_validation},
                    "total_parameter_count": 1000,
                    "training_seconds": 10.0 + seed_index + condition_index,
                    "cuda_peak_allocated_bytes": 1024 + seed_index + condition_index,
                }
    return runs


def test_paired_rows_keep_dataset_seed_differences_and_sample_sd():
    paired = _paired_rows(_synthetic_runs())
    rows = [
        row for row in paired
        if row["dataset"] == "Movies"
        and row["contrast"] == "I1-I0"
        and row["metric"] == "validation_accuracy"
    ]
    per_seed = [row for row in rows if row["row_type"] == "paired_seed"]
    summary = next(row for row in rows if row["row_type"] == "dataset_summary")
    assert all(math.isclose(row["paired_delta"], 0.01, rel_tol=0.0, abs_tol=1e-12) for row in per_seed)
    assert math.isclose(summary["mean_paired_delta"], 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert summary["sample_sd_paired_delta"] < 1e-12
    assert summary["positive_seed_count"] == 3


def test_report_renderer_keeps_accuracy_selected_endpoint_and_r0_capacity_caveat():
    runs = _synthetic_runs()
    paired = _paired_rows(runs)
    r0_rows = [
        {
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "accuracy_difference_probe_minus_r0": -0.08,
            "macro_f1_difference_probe_minus_r0": -0.12,
            "probe_parameters": 1000,
            "r0_raw_parameters": 2000,
        }
        for dataset in DATASETS
        for condition in CONDITIONS
        for seed in SEEDS
    ]
    report, _, decision = _render_report(runs, paired, r0_rows)
    assert "Validation Accuracy selected checkpoint" in report
    assert "Best-F1 minus Best-Accuracy" in report
    assert "Probe parameters | R0 parameters" in report
    assert "not capacity-matched" in report
    assert "I1−R0 Accuracy" in decision


def test_epoch_history_loader_rejects_noncontiguous_or_nonfinite_values(tmp_path):
    path = tmp_path / "history.csv"
    fields = ["dataset", "condition", "seed", "epoch", "val_accuracy", "all_metrics_finite"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"dataset": "Movies", "condition": "I0", "seed": 42, "epoch": 1, "val_accuracy": 0.5, "all_metrics_finite": True})
    assert len(_load_history(path, "Movies", "I0", 42)) == 1

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow({"dataset": "Movies", "condition": "I0", "seed": 42, "epoch": 3, "val_accuracy": "nan", "all_metrics_finite": True})
    try:
        _load_history(path, "Movies", "I0", 42)
    except ValueError as exc:
        assert "non-contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous epoch history was accepted")
