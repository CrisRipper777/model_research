from __future__ import annotations

import pytest

from scripts.analyze_mag_view_cge_v1 import CONDITIONS, DATASETS, SEEDS, compute_paired_comparisons, validate_runs


def _rows():
    rows = []
    for dataset_idx, dataset in enumerate(DATASETS):
        for condition_idx, condition in enumerate(CONDITIONS):
            for seed in SEEDS:
                base = 0.5 + dataset_idx / 10 + seed / 10000
                acc = base + condition_idx / 100
                f1 = base - condition_idx / 200
                rows.append({
                    "dataset": dataset, "condition": condition, "seed": str(seed), "status": "complete",
                    "test_evaluation": "False", "val_acc": str(acc), "val_macro_f1": str(f1), "val_ce": "1.0",
                    "best_f1_val_acc": str(acc), "best_f1_val_macro_f1": str(f1), "best_f1_val_ce": "1.0",
                    "train_acc": str(acc), "train_macro_f1": str(f1), "train_ce": "1.0",
                    "best_f1_train_acc": str(acc), "best_f1_train_macro_f1": str(f1), "best_f1_train_ce": "1.0",
                    "initial_model_sha256": "model", "initial_classifier_sha256": "head",
                    "split_fingerprint": f"split-{dataset}-{seed}", "semantic_cache_sha256": f"cache-{dataset}",
                    "feature_fingerprint": f"feature-{dataset}", "candidate_fingerprint": f"candidate-{dataset}",
                    "v7b_fingerprint": f"v7b-{dataset}",
                })
    return rows


def test_run_validation_and_paired_contrasts():
    lookup = validate_runs(_rows())
    paired = compute_paired_comparisons(lookup)
    row = next(item for item in paired if item["dataset"] == DATASETS[0] and item["contrast"] == f"{CONDITIONS[1]}-{CONDITIONS[0]}" and item["metric"] == "val_acc" and item["seed"] == 42)
    assert abs(row["difference"] - 0.01) < 1e-9
    summary = next(item for item in paired if item["dataset"] == DATASETS[1] and item["contrast"] == f"{CONDITIONS[2]}-{CONDITIONS[0]}" and item["metric"] == "val_macro_f1" and item["seed"] == "mean_sample_sd")
    assert abs(summary["difference"] + 0.01) < 1e-9
    assert summary["positive_seed_count"] == 0


def test_run_validation_rejects_test_metrics_and_duplicate_keys():
    rows = _rows()
    rows[0]["test_acc"] = "0.9"
    with pytest.raises(ValueError, match="Test metric field"):
        validate_runs(rows)
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate run key"):
        validate_runs(rows)


def test_run_validation_checks_manifest_cache_and_seed_split_provenance():
    rows = _rows()
    manifest = {
        "datasets": {
            dataset: {
                "cache_sha256": f"cache-{dataset}",
                "feature_fingerprint": f"feature-{dataset}",
                "candidate_fingerprint": f"candidate-{dataset}",
                "v7b_fingerprint": f"v7b-{dataset}",
                "seed_split_fingerprints": {str(seed): f"split-{dataset}-{seed}" for seed in SEEDS},
            }
            for dataset in DATASETS
        }
    }
    validate_runs(rows, manifest=manifest)
    rows[0]["feature_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="manifest provenance mismatch"):
        validate_runs(rows, manifest=manifest)
