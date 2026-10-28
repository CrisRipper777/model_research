#!/usr/bin/env python3
"""Run the frozen MR-CGE Final NC benchmark, one dataset/seed cell at a time."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mrcge_final_benchmark_utils import (  # noqa: E402
    MODEL_VERSION, NC_DATASETS, SEEDS, build_config, close_logger, create_logger,
    current_commit, data_info, json_write, load_prepared_manifest, next_attempt_dir,
    resolved_config_yaml, sha256_file, verify_loaded_physical_graph,
    verify_prepared_inputs,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.inference import infer_all_embeddings  # noqa: E402
from src.tasks.nc import _evaluate_indices_from_embeddings  # noqa: E402


def _evaluate_final_checkpoint(
    cfg, data, checkpoint_path: Path, device: torch.device, logger,
    on_test_start, on_test_complete,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("selection") != "best_val_accuracy":
        raise RuntimeError("NC checkpoint was not selected by Best-Validation-Accuracy")
    if int(checkpoint.get("seed", -1)) != int(cfg.seed):
        raise RuntimeError("NC checkpoint seed does not match the requested seed")
    labels = list(range(int(data.num_classes)))
    model = build_model(cfg, data_info(data))
    model.load_semantic_cache(
        str(cfg.model.semantic_cache_path),
        expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=str(cfg.model.semantic_feature_fingerprint),
        expected_candidate_fingerprint=str(cfg.model.semantic_candidate_fingerprint),
        expected_v7b_fingerprint=str(cfg.model.semantic_v7b_fingerprint),
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"])
    model.eval()
    classifier.eval()
    embeddings = infer_all_embeddings(
        model, data, device, True, int(cfg.task.inference_batch_size), "full"
    )
    val_metrics = _evaluate_indices_from_embeddings(
        classifier, embeddings, data.y, data.val_idx, device,
        int(cfg.task.inference_batch_size), eval_labels=labels, compute_macro_f1=True,
    )
    expected_val = checkpoint.get("metrics", {})
    if not math.isclose(float(val_metrics["acc"]), float(expected_val["val_acc"]), rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError("reloaded NC checkpoint does not reproduce its selected Validation Accuracy")
    if not math.isclose(float(val_metrics["macro_f1"]), float(expected_val["val_macro_f1"]), rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError("reloaded NC checkpoint does not reproduce its Validation Macro-F1")
    logger.info("Frozen checkpoint reloaded; Best-Val-Accuracy epoch=%s", checkpoint["epoch"])

    # This is the sole NC Test evaluation for the selected checkpoint.
    on_test_start()
    test_metrics = _evaluate_indices_from_embeddings(
        classifier, embeddings, data.y, data.test_idx, device,
        int(cfg.task.inference_batch_size), eval_labels=labels, compute_macro_f1=True,
    )
    on_test_complete(val_metrics, test_metrics)
    del embeddings
    return checkpoint, val_metrics, test_metrics


def _run_cell(dataset: str, seed: int, device_name: str, manifest: dict, logger_name: str) -> bool:
    record = manifest["datasets"][dataset]
    parent, attempt, output_dir = next_attempt_dir("nc", dataset, seed)
    if output_dir is None:
        print(f"[skip] nc/{dataset}/seed{seed}: verified completed checkpoint in {parent}")
        return True

    logger = create_logger(output_dir, f"{logger_name}_{dataset}_{seed}_{attempt}")
    test_evaluation_started = False
    checkpoint_sha = None
    checkpoint_path = output_dir / "best_validation.pt"
    json_write(output_dir / "status.json", {
        "status": "running", "phase": "nc", "dataset": dataset, "seed": seed,
        "attempt": attempt, "started_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
    })
    try:
        cfg = build_config("nc", dataset, seed, device_name, record)
        provenance = verify_prepared_inputs(cfg, record)
        cfg.task.save_ckpt_path = str(checkpoint_path)
        cfg.task.run_metrics_path = str(output_dir / "trainer_metrics.json")
        (output_dir / "config_resolved.yaml").write_text(resolved_config_yaml(cfg), encoding="utf-8")
        logger.info("Model version: %s | source commit: %s", MODEL_VERSION, current_commit())
        logger.info("Protocol: %s | dataset=%s | seed=%d", cfg.task.protocol_version, dataset, seed)
        logger.info("Split/cache provenance: %s", json.dumps(provenance, ensure_ascii=False))
        logger.info("Test labels are masked during training; task.evaluate_test=%s", cfg.task.evaluate_test)

        device = torch.device(device_name)
        data_train = load_mag_data(cfg, "nc", seed)
        physical_fp = verify_loaded_physical_graph(data_train, record)
        test_idx = data_train.test_idx.detach().cpu().clone()
        data_train.y = data_train.y.clone()
        data_train.y[test_idx] = -100
        if bool((data_train.y[test_idx] != -100).any()):
            raise RuntimeError("NC Test labels were not masked before Trainer entry")
        logger.info("NC split loaded from %s; Test label count masked=%d", data_train.info["node_split_path"] if "node_split_path" in data_train.info else data_train.info["nc_split_path"], int(test_idx.numel()))

        from src.tasks.nc import run_nc
        run_nc(cfg, data_train, device, logger)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Trainer did not save Best-Validation-Accuracy checkpoint: {checkpoint_path}")
        checkpoint_sha = sha256_file(checkpoint_path)
        frozen_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if frozen_checkpoint.get("selection") != "best_val_accuracy":
            raise RuntimeError("Trainer saved a checkpoint with an unexpected selection rule")
        json_write(output_dir / "status.json", {
            "status": "checkpoint_frozen", "phase": "nc", "dataset": dataset, "seed": seed,
            "attempt": attempt, "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha, "best_epoch": frozen_checkpoint.get("epoch"),
            "test_evaluated": False,
        })
        logger.info("Best Validation checkpoint frozen at epoch %s (%s)", frozen_checkpoint["epoch"], checkpoint_sha)

        # Reload after the checkpoint is fixed. Only this fresh data object exposes Test labels.
        del data_train
        data_eval = load_mag_data(cfg, "nc", seed)
        if verify_loaded_physical_graph(data_eval, record) != physical_fp:
            raise RuntimeError("NC physical graph changed between training and final evaluation")
        if sha256_file(Path(data_eval.info.get("nc_split_path", data_eval.info.get("node_split_path")))) != provenance["split_sha256"]:
            raise RuntimeError("NC split changed between training and final evaluation")

        def mark_test_started():
            nonlocal test_evaluation_started
            test_evaluation_started = True
            json_write(output_dir / "status.json", {
                "status": "test_evaluation_started", "phase": "nc", "dataset": dataset,
                "seed": seed, "attempt": attempt, "checkpoint_sha256": checkpoint_sha,
                "test_evaluation_started": True,
            })

        def save_test_metrics(val_metrics, test_metrics):
            json_write(output_dir / "post_checkpoint_metrics.json", {
                "validation": {key: float(value) for key, value in val_metrics.items()},
                "test": {key: float(value) for key, value in test_metrics.items()},
                "test_evaluation_count": 1,
            })

        checkpoint, val_metrics, test_metrics = _evaluate_final_checkpoint(
            cfg, data_eval, checkpoint_path, device, logger, mark_test_started, save_test_metrics
        )
        metric_values = [*val_metrics.values(), *test_metrics.values()]
        if not all(math.isfinite(float(value)) for value in metric_values):
            raise RuntimeError("NC benchmark produced a non-finite metric")
        result = {
            "phase": "nc", "dataset": dataset, "seed": seed,
            "protocol_version": str(cfg.task.protocol_version),
            "model_version": MODEL_VERSION, "source_commit": current_commit(),
            "source_model_commit": "4177ec8aec4e05ad513c0357f36da7b926a5c040",
            "selection": "best_val_accuracy", "best_epoch": int(checkpoint["epoch"]),
            "validation": {
                "accuracy": float(val_metrics["acc"]),
                "macro_f1": float(val_metrics["macro_f1"]),
                "cross_entropy": float(val_metrics["cross_entropy"]),
            },
            "test": {
                "accuracy": float(test_metrics["acc"]),
                "macro_f1": float(test_metrics["macro_f1"]),
                "macro_f1_class_ids": list(range(int(data_eval.num_classes))),
                "evaluation_count": 1,
            },
            "test_labels_masked_during_training": True,
            "test_evaluated_after_checkpoint_freeze_and_reload": True,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_reload_verified": True,
            "split": provenance,
            "physical_edge_fingerprint": physical_fp,
            "semantic_cache_path": provenance["semantic_cache_path"],
            "semantic_cache_sha256": provenance["semantic_cache_sha256"],
            "candidate_fingerprint": provenance["candidate_fingerprint"],
            "v7b_fingerprint": provenance["v7b_fingerprint"],
            "config_path": str((output_dir / "config_resolved.yaml").resolve()),
        }
        json_write(output_dir / "result.json", result)
        json_write(output_dir / "status.json", {
            "status": "complete", "phase": "nc", "dataset": dataset, "seed": seed,
            "attempt": attempt, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": checkpoint_sha, "test_evaluated": True,
            "test_evaluation_started": True,
        })
        logger.info("Result: Val Acc %.6f | Test Acc %.6f | Test Macro-F1 %.6f",
                    result["validation"]["accuracy"], result["test"]["accuracy"], result["test"]["macro_f1"])
        return True
    except Exception:
        error = traceback.format_exc()
        logger.error("NC cell failed:\n%s", error)
        json_write(output_dir / "error.json", {"error": error})
        json_write(output_dir / "status.json", {
            "status": "failed", "phase": "nc", "dataset": dataset, "seed": seed,
            "attempt": attempt, "test_evaluated": test_evaluation_started,
            "test_evaluation_started": test_evaluation_started,
            "checkpoint_sha256": checkpoint_sha,
        })
        return False
    finally:
        close_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=NC_DATASETS, default=list(NC_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = load_prepared_manifest()
    ok = True
    for dataset in args.datasets:
        for seed in args.seeds:
            ok = _run_cell(dataset, seed, args.device, manifest, "mrcge_final_nc") and ok
    if not ok:
        raise SystemExit("one or more NC cells failed; inspect each attempt's train.log/error.json")


if __name__ == "__main__":
    main()
