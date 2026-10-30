#!/usr/bin/env python3
"""Run the frozen MR-CGE Final LP benchmark with one grouped multi-seed run per dataset."""

from __future__ import annotations

import argparse
import copy
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
    LP_DATASETS, MODEL_VERSION, SEEDS, build_config, close_logger, create_logger,
    current_commit, data_info, json_write, load_prepared_manifest, next_attempt_dir,
    resolved_config_yaml, sha256_file, verify_loaded_physical_graph,
    verify_prepared_inputs,
)
from src.data import EdgeSplit, load_mag_data  # noqa: E402
from src.models import LinkPredictor, build_model  # noqa: E402
from src.tasks.inference import infer_all_embeddings  # noqa: E402
from src.tasks.lp import _evaluate_split, _prepare_eval_embeddings, run_lp  # noqa: E402


def _training_copy_without_test_candidates(data):
    """Keep official positives for the existing negative sampler, hide Test negatives."""
    training = copy.copy(data)
    split = data.edge_split
    training.edge_split = EdgeSplit(
        train={key: value.clone() for key, value in split.train.items()},
        valid={key: value.clone() for key, value in split.valid.items()},
        test={key: split.test[key].clone() for key in ("source_node", "target_node")},
        metadata=copy.deepcopy(split.metadata),
    )
    return training


def _load_model_bundle(cfg, data, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("selection") != "best_val_mrr":
        raise RuntimeError("LP checkpoint was not selected by Best-Validation-MRR")
    if int(checkpoint.get("seed", -1)) != int(cfg.seed):
        raise RuntimeError("LP checkpoint seed does not match the requested seed")

    model = build_model(cfg, data_info(data))
    model.load_semantic_cache(
        str(cfg.model.semantic_cache_path),
        expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=str(cfg.model.semantic_feature_fingerprint),
        expected_candidate_fingerprint=str(cfg.model.semantic_candidate_fingerprint),
        expected_v7b_fingerprint=str(cfg.model.semantic_v7b_fingerprint),
    )
    model = model.to(device)
    model.bind_lp_graph(data.x, data.edge_index)
    model.load_state_dict(checkpoint["model_state"])

    projection = nn.Linear(model.out_dim, int(cfg.task.decoder.proj_dim)).to(device)
    projection.load_state_dict(checkpoint["proj_state"])
    predictor = LinkPredictor(
        in_dim=int(cfg.task.decoder.proj_dim),
        hidden_dim=int(cfg.task.decoder.hidden_dim),
        num_layers=int(cfg.task.decoder.num_layers),
        dropout=float(cfg.task.decoder.dropout),
    ).to(device)
    predictor.load_state_dict(checkpoint["head_state"])
    model.eval()
    projection.eval()
    predictor.eval()
    return checkpoint, model, projection, predictor


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
        "phase": "lp", "dataset": dataset, "seeds": seeds,
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
            "status": "running", "phase": "lp", "dataset": dataset,
            "seed": seed, "attempt": attempt, "started_utc": datetime.now(timezone.utc).isoformat(),
            "test_evaluated": False, "trainer_group_seeds": seeds,
        })
        json_write(out_dir / "trainer_group.json", group_info)
        cfg = build_config("lp", dataset, seed, device_name, record)
        cfg.num_runs = 1
        cfg.task.save_ckpt_path = str(out_dir / "best_validation.pt")
        provenances[seed] = verify_prepared_inputs(cfg, record)
        (out_dir / "config_resolved.yaml").write_text(resolved_config_yaml(cfg), encoding="utf-8")
        cell_configs[seed] = cfg

    trainer_cfg = build_config("lp", dataset, anchor_seed, device_name, record)
    trainer_cfg.num_runs = len(seeds)
    group_checkpoint_base = anchor_dir / "trainer_group_checkpoint.pt"
    trainer_cfg.task.save_ckpt_path = str(group_checkpoint_base)
    group_config_path = anchor_dir / "trainer_group_config.yaml"
    group_config_path.write_text(resolved_config_yaml(trainer_cfg), encoding="utf-8")
    group_info["trainer_config"] = str(group_config_path.resolve())
    for seed in seeds:
        json_write(output_dirs[seed] / "trainer_group.json", group_info)

    try:
        logger.info("Model version: %s | source commit: %s", MODEL_VERSION, current_commit())
        logger.info("Grouped LP run: dataset=%s | seeds=%s | num_runs=%d", dataset, seeds, len(seeds))
        logger.info("Protocol: %s", trainer_cfg.task.protocol_version)
        logger.info("Official split/cache provenance: %s",
                    json.dumps(provenances[anchor_seed], ensure_ascii=False))
        logger.info("Semantic-source physical context approximation: fanout %s per hop from Train graph",
                    list(trainer_cfg.model.semantic_context_physical_fanout))
        logger.info("All cached semantic candidates/weights/flags are retained; no local KNN or candidate truncation")
        logger.info("Trainer group config: %s", group_config_path)

        data_full = load_mag_data(trainer_cfg, "lp", anchor_seed)
        physical_fp = verify_loaded_physical_graph(data_full, record)
        logger.info("Official edge split loaded from %s", data_full.info["edge_split_path"])
        data_train = _training_copy_without_test_candidates(data_full)
        del data_full
        if "target_node_neg" in data_train.edge_split.test:
            raise RuntimeError("Test negative candidates remained visible to LP Trainer")

        run_lp(trainer_cfg, data_train, device, logger)
        for run_id, seed in enumerate(seeds):
            source = _group_checkpoint_path(group_checkpoint_base, run_id, len(seeds))
            if not source.is_file():
                raise FileNotFoundError(f"Trainer did not save Best-Validation-MRR checkpoint for seed {seed}: {source}")
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            if checkpoint.get("selection") != "best_val_mrr" or int(checkpoint.get("seed", -1)) != seed:
                raise RuntimeError(f"Trainer saved an invalid Best-Validation-MRR checkpoint for seed {seed}")
            destination = output_dirs[seed] / "best_validation.pt"
            shutil.move(str(source), str(destination))
            checkpoint_sha = sha256_file(destination)
            attempt = int(output_dirs[seed].name.removeprefix("attempt_"))
            json_write(output_dirs[seed] / "status.json", {
                "status": "checkpoint_frozen", "phase": "lp", "dataset": dataset,
                "seed": seed, "attempt": attempt, "checkpoint_path": str(destination.resolve()),
                "checkpoint_sha256": checkpoint_sha, "best_epoch": checkpoint.get("epoch"),
                "test_evaluated": False, "trainer_group_seeds": seeds,
            })
            logger.info("Best Validation MRR checkpoint frozen for seed %d at epoch %s (%s)",
                        seed, checkpoint["epoch"], checkpoint_sha)
        del data_train
    except Exception:
        error = traceback.format_exc()
        logger.error("LP grouped training failed:\n%s", error)
        for seed in seeds:
            out_dir = output_dirs[seed]
            attempt = int(out_dir.name.removeprefix("attempt_"))
            checkpoint_path = out_dir / "best_validation.pt"
            json_write(out_dir / "error.json", {"error": error})
            json_write(out_dir / "status.json", {
                "status": "failed", "phase": "lp", "dataset": dataset,
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
        checkpoint = model = projection = predictor = embeddings = embeddings_eval = data_eval = None
        try:
            data_eval = load_mag_data(cfg, "lp", seed)
            if verify_loaded_physical_graph(data_eval, record) != physical_fp:
                raise RuntimeError("LP Train physical graph changed between training and final evaluation")
            if sha256_file(Path(data_eval.info["edge_split_path"])) != provenances[seed]["split_sha256"]:
                raise RuntimeError("official LP split changed between training and final evaluation")

            checkpoint, model, projection, predictor = _load_model_bundle(
                cfg, data_eval, checkpoint_path, device
            )
            with torch.no_grad():
                embeddings = infer_all_embeddings(
                    model, data_eval, device, True, int(cfg.task.inference_batch_size), "full"
                )
                embeddings = projection(embeddings.to(device)).detach().cpu()
            embeddings_eval = _prepare_eval_embeddings(
                embeddings, device, bool(cfg.task.eval_preload_node_emb), logger
            )
            val_metrics = _evaluate_split(
                embeddings_eval, predictor, data_eval.edge_split.valid, device,
                int(cfg.task.eval_edge_batch_size),
            )
            expected_val_mrr = float(checkpoint.get("metrics", {}).get("val_mrr", float("nan")))
            if not math.isfinite(expected_val_mrr) or not math.isclose(
                float(val_metrics["mrr"]), expected_val_mrr, rel_tol=0.0, abs_tol=1e-4
            ):
                raise RuntimeError("reloaded LP checkpoint does not reproduce its selected Validation MRR")
            val_mrr_reload_delta = abs(float(val_metrics["mrr"]) - expected_val_mrr)
            logger.info("Frozen checkpoint reloaded for seed %d; Validation MRR reproduced: %.8f (|delta|=%.3g)",
                        seed, val_metrics["mrr"], val_mrr_reload_delta)

            test_evaluation_started = True
            json_write(out_dir / "status.json", {
                "status": "test_evaluation_started", "phase": "lp", "dataset": dataset,
                "seed": seed, "attempt": attempt, "checkpoint_sha256": checkpoint_sha,
                "test_evaluation_started": True, "trainer_group_seeds": seeds,
            })
            test_metrics = _evaluate_split(
                embeddings_eval, predictor, data_eval.edge_split.test, device,
                int(cfg.task.eval_edge_batch_size),
            )
            json_write(out_dir / "post_checkpoint_metrics.json", {
                "validation": {key: float(value) for key, value in val_metrics.items()},
                "validation_mrr_checkpoint_reload_abs_delta": val_mrr_reload_delta,
                "test": {key: float(value) for key, value in test_metrics.items()},
                "test_evaluation_count": 1,
            })
            metric_values = [*val_metrics.values(), *test_metrics.values()]
            if not all(math.isfinite(float(value)) for value in metric_values):
                raise RuntimeError("LP benchmark produced a non-finite metric")
            provenance = provenances[seed]
            result = {
                "phase": "lp", "dataset": dataset, "seed": seed,
                "protocol_version": str(cfg.task.protocol_version),
                "split_protocol": "official lp-edge-split.pt loaded unchanged; original positive overlap/self-loops accepted as-is",
                "model_version": MODEL_VERSION, "source_commit": current_commit(),
                "source_model_commit": "4177ec8aec4e05ad513c0357f36da7b926a5c040",
                "selection": "best_val_mrr", "best_epoch": int(checkpoint["epoch"]),
                "validation": {key: float(value) for key, value in val_metrics.items()},
                "test": {key: float(value) for key, value in test_metrics.items()},
                "test_negative_candidates": "official fixed candidates from lp-edge-split.pt",
                "test_evaluation_count": 1,
                "test_evaluated_after_checkpoint_freeze_and_reload": True,
                "positive_edge_mask_backend": "global_eid",
                "link_neighbor_fanouts": list(cfg.task.num_neighbors),
                "semantic_candidates_truncated": False,
                "semantic_weights_renormalized": False,
                "semantic_source_physical_context": {
                    "fanout_per_hop": list(cfg.model.semantic_context_physical_fanout),
                    "graph": "official Train-positive physical graph only",
                    "approximation": "all semantic candidates and original weights are retained; supplementary Late-Completion source physical context uses fixed fanout [1,1,1] for three hops, so this is not full-graph-equivalent",
                },
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
                "status": "complete", "phase": "lp", "dataset": dataset, "seed": seed,
                "attempt": attempt, "checkpoint_sha256": checkpoint_sha,
                "test_evaluated": True, "test_evaluation_started": True,
                "trainer_group_seeds": seeds,
            })
            logger.info("Seed %d result: Val MRR %.6f | Test MRR %.6f | Hits@1 %.6f | Hits@3 %.6f | Hits@10 %.6f",
                        seed, result["validation"]["mrr"], result["test"]["mrr"], result["test"]["hits@1"],
                        result["test"]["hits@3"], result["test"]["hits@10"])
        except Exception:
            all_ok = False
            error = traceback.format_exc()
            logger.error("LP final evaluation failed for seed %d:\n%s", seed, error)
            json_write(out_dir / "error.json", {"error": error})
            json_write(out_dir / "status.json", {
                "status": "failed", "phase": "lp", "dataset": dataset,
                "seed": seed, "attempt": attempt, "test_evaluated": test_evaluation_started,
                "test_evaluation_started": test_evaluation_started,
                "checkpoint_sha256": checkpoint_sha,
                "trainer_group_seeds": seeds,
            })
        finally:
            del checkpoint, model, projection, predictor, embeddings, embeddings_eval, data_eval
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    group_info["status"] = "complete" if all_ok else "partial"
    for seed in seeds:
        json_write(output_dirs[seed] / "trainer_group.json", group_info)
    close_logger(logger)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=LP_DATASETS, default=list(LP_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = load_prepared_manifest()
    ok = True
    for dataset in args.datasets:
        pending_dirs = {}
        for seed in sorted(set(args.seeds)):
            parent, attempt, output_dir = next_attempt_dir("lp", dataset, seed)
            if output_dir is None:
                print(f"[skip] lp/{dataset}/seed{seed}: verified completed checkpoint in {parent}")
            else:
                pending_dirs[seed] = output_dir
        for seed_group in _contiguous_seed_groups(list(pending_dirs)):
            group_dirs = {seed: pending_dirs[seed] for seed in seed_group}
            ok = _run_seed_group(dataset, seed_group, group_dirs, args.device, manifest,
                                 "mrcge_final_lp") and ok
    if not ok:
        raise SystemExit("one or more LP cells failed; inspect each attempt's train.log/error.json")


if __name__ == "__main__":
    main()
