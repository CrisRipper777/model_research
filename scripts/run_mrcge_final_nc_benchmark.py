#!/usr/bin/env python3
"""Run the frozen MR-CGE Final NC benchmark with one grouped multi-seed run per dataset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
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


def _contiguous_seed_groups(seeds: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for seed in sorted(seeds):
        if not groups or seed != groups[-1][-1] + 1:
            groups.append([seed])
        else:
            groups[-1].append(seed)
    return groups


def _group_checkpoint_path(base: Path, run_id: int, run_count: int) -> Path:
    if run_count == 1:
        return base
    return base.with_name(f"{base.stem}_run{run_id + 1}{base.suffix}")


def _run_seed_group(
    dataset: str, seeds: list[int], output_dirs: dict[int, Path], device_name: str,
    manifest: dict, logger_name: str,
) -> bool:
    record = manifest["datasets"][dataset]
    anchor_seed = seeds[0]
    anchor_dir = output_dirs[anchor_seed]
    anchor_attempt = int(anchor_dir.name.removeprefix("attempt_"))
    logger = create_logger(anchor_dir, f"{logger_name}_{dataset}_{anchor_seed}_group")
    device = torch.device(device_name)
    group_info = {
        "phase": "nc", "dataset": dataset, "seeds": seeds,
        "trainer_runs": len(seeds), "anchor_seed": anchor_seed,
        "anchor_attempt": anchor_attempt,
        "train_log": str((anchor_dir / "train.log").resolve()),
    }
    cell_configs = {}
    provenances = {}
    for seed in seeds:
        out_dir = output_dirs[seed]
        attempt = int(out_dir.name.removeprefix("attempt_"))
        json_write(out_dir / "status.json", {
            "status": "running", "phase": "nc", "dataset": dataset,
            "seed": seed, "attempt": attempt, "started_utc": datetime.now(timezone.utc).isoformat(),
            "test_evaluated": False, "trainer_group_seeds": seeds,
        })
        json_write(out_dir / "trainer_group.json", group_info)
        cfg = build_config("nc", dataset, seed, device_name, record)
        cfg.num_runs = 1
        cfg.task.save_ckpt_path = str(out_dir / "best_validation.pt")
        cfg.task.run_metrics_path = str(anchor_dir / "trainer_metrics.json")
        provenances[seed] = verify_prepared_inputs(cfg, record)
        (out_dir / "config_resolved.yaml").write_text(resolved_config_yaml(cfg), encoding="utf-8")
        cell_configs[seed] = cfg

    trainer_cfg = build_config("nc", dataset, anchor_seed, device_name, record)
    trainer_cfg.num_runs = len(seeds)
    group_checkpoint_base = anchor_dir / "trainer_group_checkpoint.pt"
    trainer_cfg.task.save_ckpt_path = str(group_checkpoint_base)
    trainer_cfg.task.run_metrics_path = str(anchor_dir / "trainer_metrics.json")
    group_config_path = anchor_dir / "trainer_group_config.yaml"
    group_config_path.write_text(resolved_config_yaml(trainer_cfg), encoding="utf-8")
    group_info["trainer_config"] = str(group_config_path.resolve())
    for seed in seeds:
        json_write(output_dirs[seed] / "trainer_group.json", group_info)

    try:
        logger.info("Model version: %s | source commit: %s", MODEL_VERSION, current_commit())
        logger.info("Grouped NC run: dataset=%s | seeds=%s | num_runs=%d", dataset, seeds, len(seeds))
        logger.info("Protocol: %s", trainer_cfg.task.protocol_version)
        logger.info("Test labels are masked during training; task.evaluate_test=%s", trainer_cfg.task.evaluate_test)
        logger.info("Trainer group config: %s", group_config_path)

        data_train = load_mag_data(trainer_cfg, "nc", anchor_seed)
        physical_fp = verify_loaded_physical_graph(data_train, record)
        test_idx = data_train.test_idx.detach().cpu().clone()
        data_train.y = data_train.y.clone()
        data_train.y[test_idx] = -100
        if bool((data_train.y[test_idx] != -100).any()):
            raise RuntimeError("NC Test labels were not masked before Trainer entry")
        logger.info("NC split loaded from %s; Test label count masked=%d",
                    data_train.info["node_split_path"] if "node_split_path" in data_train.info else data_train.info["nc_split_path"],
                    int(test_idx.numel()))

        from src.tasks.nc import run_nc
        run_nc(trainer_cfg, data_train, device, logger)

        for run_id, seed in enumerate(seeds):
            source = _group_checkpoint_path(group_checkpoint_base, run_id, len(seeds))
            if not source.is_file():
                raise FileNotFoundError(f"Trainer did not save Best-Validation-Accuracy checkpoint for seed {seed}: {source}")
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            if checkpoint.get("selection") != "best_val_accuracy" or int(checkpoint.get("seed", -1)) != seed:
                raise RuntimeError(f"Trainer saved an invalid Best-Validation-Accuracy checkpoint for seed {seed}")
            destination = output_dirs[seed] / "best_validation.pt"
            shutil.move(str(source), str(destination))
            checkpoint_sha = sha256_file(destination)
            attempt = int(output_dirs[seed].name.removeprefix("attempt_"))
            json_write(output_dirs[seed] / "status.json", {
                "status": "checkpoint_frozen", "phase": "nc", "dataset": dataset,
                "seed": seed, "attempt": attempt, "checkpoint_path": str(destination.resolve()),
                "checkpoint_sha256": checkpoint_sha, "best_epoch": checkpoint.get("epoch"),
                "test_evaluated": False, "trainer_group_seeds": seeds,
            })
            logger.info("Best Validation checkpoint frozen for seed %d at epoch %s (%s)",
                        seed, checkpoint["epoch"], checkpoint_sha)
        del data_train
    except Exception:
        error = traceback.format_exc()
        logger.error("NC grouped training failed:\n%s", error)
        for seed in seeds:
            out_dir = output_dirs[seed]
            attempt = int(out_dir.name.removeprefix("attempt_"))
            checkpoint_path = out_dir / "best_validation.pt"
            json_write(out_dir / "error.json", {"error": error})
            json_write(out_dir / "status.json", {
                "status": "failed", "phase": "nc", "dataset": dataset,
                "seed": seed, "attempt": attempt, "test_evaluated": False,
                "test_evaluation_started": False,
                "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
                "trainer_group_seeds": seeds,
            })
        close_logger(logger)
        return False

    all_ok = True
    for seed in seeds:
        out_dir = output_dirs[seed]
        attempt = int(out_dir.name.removeprefix("attempt_"))
        cfg = cell_configs[seed]
        checkpoint_path = out_dir / "best_validation.pt"
        checkpoint_sha = sha256_file(checkpoint_path)
        test_evaluation_started = False
        checkpoint = data_eval = None
        try:
            data_eval = load_mag_data(cfg, "nc", seed)
            if verify_loaded_physical_graph(data_eval, record) != physical_fp:
                raise RuntimeError("NC physical graph changed between training and final evaluation")
            split_path = Path(data_eval.info.get("nc_split_path", data_eval.info.get("node_split_path")))
            if sha256_file(split_path) != provenances[seed]["split_sha256"]:
                raise RuntimeError("NC split changed between training and final evaluation")

            def mark_test_started():
                nonlocal test_evaluation_started
                test_evaluation_started = True
                json_write(out_dir / "status.json", {
                    "status": "test_evaluation_started", "phase": "nc", "dataset": dataset,
                    "seed": seed, "attempt": attempt, "checkpoint_sha256": checkpoint_sha,
                    "test_evaluation_started": True, "trainer_group_seeds": seeds,
                })

            def save_test_metrics(val_metrics, test_metrics):
                json_write(out_dir / "post_checkpoint_metrics.json", {
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
            provenance = provenances[seed]
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
                "config_path": str((out_dir / "config_resolved.yaml").resolve()),
                "trainer_group": group_info,
            }
            json_write(out_dir / "result.json", result)
            json_write(out_dir / "status.json", {
                "status": "complete", "phase": "nc", "dataset": dataset, "seed": seed,
                "attempt": attempt, "completed_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_sha256": checkpoint_sha, "test_evaluated": True,
                "test_evaluation_started": True, "trainer_group_seeds": seeds,
            })
            logger.info("Seed %d result: Val Acc %.6f | Test Acc %.6f | Test Macro-F1 %.6f",
                        seed, result["validation"]["accuracy"], result["test"]["accuracy"], result["test"]["macro_f1"])
        except Exception:
            all_ok = False
            error = traceback.format_exc()
            logger.error("NC final evaluation failed for seed %d:\n%s", seed, error)
            json_write(out_dir / "error.json", {"error": error})
            json_write(out_dir / "status.json", {
                "status": "failed", "phase": "nc", "dataset": dataset,
                "seed": seed, "attempt": attempt, "test_evaluated": test_evaluation_started,
                "test_evaluation_started": test_evaluation_started,
                "checkpoint_sha256": checkpoint_sha,
                "trainer_group_seeds": seeds,
            })
        finally:
            del checkpoint, data_eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    group_info["status"] = "complete" if all_ok else "partial"
    for seed in seeds:
        json_write(output_dirs[seed] / "trainer_group.json", group_info)
    close_logger(logger)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=NC_DATASETS, default=list(NC_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = load_prepared_manifest()
    ok = True
    for dataset in args.datasets:
        pending_dirs = {}
        for seed in sorted(set(args.seeds)):
            parent, attempt, output_dir = next_attempt_dir("nc", dataset, seed)
            if output_dir is None:
                print(f"[skip] nc/{dataset}/seed{seed}: verified completed checkpoint in {parent}")
            else:
                pending_dirs[seed] = output_dir
        for seed_group in _contiguous_seed_groups(list(pending_dirs)):
            group_dirs = {seed: pending_dirs[seed] for seed in seed_group}
            ok = _run_seed_group(dataset, seed_group, group_dirs, args.device, manifest,
                                 "mrcge_final_nc") and ok
    if not ok:
        raise SystemExit("one or more NC cells failed; inspect each attempt's train.log/error.json")


if __name__ == "__main__":
    main()
