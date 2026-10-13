from __future__ import annotations

import os
import subprocess

import torch
import pytest

from src.data.types import MAGData
from scripts.analyze_rmcge_v22 import paired_results, validate_runs
from scripts.run_rmcge_v22 import _assert_physical_gpu_idle, _degree_rows, dataset_fingerprints, mask_test_labels, metric_bundle
from scripts.analyze_rmcge_v22 import CONDITIONS, DATASETS, SEEDS


def _data(labels):
    return MAGData(
        name="toy", source="unit", task="nc", x=torch.arange(24).reshape(6, 4).float(),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]), num_nodes=6,
        x_t=torch.ones(6, 2), x_i=torch.ones(6, 2), y=torch.tensor(labels),
        train_idx=torch.tensor([0, 1]), val_idx=torch.tensor([2, 3]), test_idx=torch.tensor([4, 5]),
        num_classes=3,
    )


def test_test_label_mask_is_applied_before_training_and_fingerprinting_ignores_test_values():
    data = _data([0, 1, 2, 1, 0, 2])
    safe = mask_test_labels(data)
    assert safe.y.tolist() == [0, 1, 2, 1, -100, -100]
    changed_test = _data([0, 1, 2, 1, 2, 0])
    assert dataset_fingerprints(data) == dataset_fingerprints(changed_test)


def test_metric_bundle_uses_train_validation_supported_classes_without_test_labels():
    labels = torch.tensor([1, 1, -100, -100])
    logits = torch.tensor([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0], [9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    result = metric_bundle(logits, labels, torch.tensor([0, 1]), 3, eval_labels=[1])
    assert result["macro_f1"] == 1.0


def _run_rows():
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                value = (CONDITIONS.index(condition) + seed / 1000 + DATASETS.index(dataset) / 100)
                rows.append({
                    "dataset": dataset, "condition": condition, "seed": seed,
                    "status": "complete", "test_evaluation": "false",
                    "val_acc": str(value), "val_macro_f1": str(value + 0.01), "val_ce": str(value + 0.1),
                    "best_f1_val_acc": str(value - 0.01), "best_f1_val_macro_f1": str(value + 0.02),
                    "best_f1_val_ce": str(value + 0.2), "train_acc": str(value + 0.03),
                    "train_macro_f1": str(value + 0.04), "train_ce": "0.5", "train_ce_eval": "0.4",
                    "train_val_acc_gap": "0.02", "train_val_macro_f1_gap": "0.03", "train_val_ce_gap": "-0.1",
                    "wall_seconds": "1.0", "semantic_cache_sha256": "cache", "split_fingerprint": "split",
                    "initial_model_sha256": "model", "initial_classifier_sha256": "head",
                })
    return rows


def test_run_manifest_validation_and_all_required_paired_contrasts():
    rows = _run_rows()
    lookup = validate_runs(rows, {"test_evaluation": False, "test_label_values_indexed": False})
    paired = paired_results(lookup)
    assert len(paired) == 3 * 7 * 3 * 6
    m3_m2 = [row for row in paired if row["dataset"] == "Movies" and row["contrast"] == "M3-M2" and row["metric"] == "best_accuracy.val_acc"]
    assert len(m3_m2) == 3
    assert all(abs(row["difference"] - 1.0) < 1e-12 for row in m3_m2)


def test_degree_groups_use_global_node_logits_and_include_degree_one_content():
    degree = torch.tensor([0, 1, 2, 3, 4, 8])
    val_idx = torch.tensor([1, 2, 3, 4, 5])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    all_logits = torch.tensor([
        [5.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 5.0, 0], [0, 0, 5.0], [0, 0, 5.0],
    ])
    pred_val = all_logits[val_idx].argmax(-1)
    content = {"text": torch.ones(6, 256), "visual": torch.full((6, 256), 2.0)}
    rows = _degree_rows("Movies", "M3", 42, "best_accuracy", degree, val_idx, labels, pred_val, all_logits, content, 3)
    degree1 = next(row for row in rows if row["degree_group"] == "1")
    assert degree1["val_count"] == 1
    assert degree1["accuracy"] == 1.0
    assert degree1["content_text_rms"] == 1.0
    assert degree1["content_visual_rms"] == 2.0


def test_gpu_guard_ignores_own_cuda_context_and_rejects_other_process(monkeypatch):
    own_pid = os.getpid()
    responses = iter([
        subprocess.CompletedProcess([], 0, "0, uuid0, 22858\n1, uuid1, 583\n", ""),
        subprocess.CompletedProcess([], 0, f"uuid1, {own_pid}, 568\n", ""),
    ])
    monkeypatch.setattr("scripts.run_rmcge_v22.subprocess.run", lambda *args, **kwargs: next(responses))
    _assert_physical_gpu_idle(1)

    responses = iter([
        subprocess.CompletedProcess([], 0, "0, uuid0, 22858\n1, uuid1, 583\n", ""),
        subprocess.CompletedProcess([], 0, f"uuid1, {own_pid + 1}, 10\n", ""),
    ])
    monkeypatch.setattr("scripts.run_rmcge_v22.subprocess.run", lambda *args, **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="other compute processes"):
        _assert_physical_gpu_idle(1)
