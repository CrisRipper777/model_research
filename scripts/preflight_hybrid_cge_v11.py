#!/usr/bin/env python3
"""Data, pairing, and optional real-GPU preflight for Hybrid-CGE V1.1."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_rmcge_v22 as legacy  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


DATASETS = legacy.DATASETS
CONDITIONS = ("H1_hybrid", "H2_hybrid_plus")
SEEDS = (42, 43, 44)
STUDY = ROOT / "research/hybrid_cge_v11"
DATA_DIR = STUDY / "data"
PARENT_MANIFEST = ROOT / "research/mag_view_cge_v1/data/manifest.json"
PARENT_RUNS = ROOT / "research/mag_view_cge_v1/data/runs.csv"
CG_PARENT_MANIFEST = ROOT / "research/cg_cge_v1/data/manifest.json"
CG_PARENT_RUNS = ROOT / "research/cg_cge_v1/data/runs.csv"


def build_config(dataset: str, condition: str, seed: int, cache: dict[str, Any], device: str):
    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create({
        "paths": root_cfg.paths,
        "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
        "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
        "model": OmegaConf.load(ROOT / "configs/model/hybrid_cge_v11.yaml"),
        "logging": root_cfg.logging,
        "seed": int(seed), "num_runs": 1, "device": str(device),
    })
    cfg.dataset.name = dataset
    cfg.model.condition = condition
    cfg.model.late_cross_dropout = 0.0 if condition == "H1_hybrid" else 0.15
    cfg.model.semantic_cache_path = cache["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    cfg.task.name = "nc"
    cfg.task.protocol_version = legacy.PROTOCOL
    cfg.task.training_mode = "full_graph"
    cfg.task.optimizer, cfg.task.lr, cfg.task.weight_decay = "adamw", 1.0e-3, 1.0e-4
    cfg.task.epochs, cfg.task.patience = 300, 30
    cfg.task.early_stop_min_epoch, cfg.task.early_stop_min_delta = 30, 1.0e-4
    cfg.task.grad_clip, cfg.task.eval_every = 1.0, 1
    cfg.task.inference_mode, cfg.task.inference_batch_size = "full", 4096
    cfg.task.scheduler = None
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 1.0
    cfg.task.loss.unimodal_weight = 0.3
    cfg.task.record_extended_validation = True
    cfg.task.record_train_ce_separately = True
    cfg.task.extended_checkpoint_metadata = {
        "study": "Hybrid-CGE V1.1", "dataset": dataset, "condition": condition,
        "seed": int(seed), "official_train_val_split": True,
        "test_evaluation": False, "test_label_values_indexed": False,
    }
    cfg.paths.output_root = str(ROOT / "outputs/hybrid_cge_v11/hydra")
    OmegaConf.resolve(cfg)
    return cfg


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mask_test_labels(data):
    safe = copy.copy(data)
    safe.y = data.y.clone()
    safe.y[data.test_idx] = -100
    return safe


def load_model(cfg, data, cache, device):
    model = build_model(cfg, legacy.data_info(data))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=cfg.dataset.name,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    return model.to(device)


def _reference_matches(
    dataset: str,
    seed: int,
    fingerprints: dict,
    cache: dict,
    parent_manifest: dict,
    view_runs: list[dict[str, str]],
    cg_manifest: dict,
    cg_runs: list[dict[str, str]],
) -> dict[str, Any]:
    parent = parent_manifest["datasets"][dataset]
    actual = {
        "split_fingerprint": fingerprints["split_fingerprint"],
        "feature_fingerprint": cache["feature_fingerprint"],
        "raw_features_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "cache_sha256": legacy.sha256_file(Path(cache["cache_path_abs"])),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
        "protocol": legacy.PROTOCOL,
    }
    view_expected = {
        "split_fingerprint": parent["seed_split_fingerprints"][str(seed)],
        "feature_fingerprint": parent["feature_fingerprint"],
        "raw_features_fingerprint": parent["features_fingerprint"],
        "physical_edge_fingerprint": parent["physical_edge_fingerprint"],
        "cache_sha256": parent["cache_sha256"],
        "candidate_fingerprint": parent["candidate_fingerprint"],
        "v7b_fingerprint": parent["v7b_fingerprint"],
        "protocol": parent_manifest["protocol_version"],
    }
    row_expected = {
        "split_fingerprint": actual["split_fingerprint"],
        "feature_fingerprint": actual["feature_fingerprint"],
        "semantic_cache_sha256": actual["cache_sha256"],
        "candidate_fingerprint": actual["candidate_fingerprint"],
        "v7b_fingerprint": actual["v7b_fingerprint"],
    }
    view_row = next((row for row in view_runs if row["dataset"] == dataset and row["condition"] == "A2_cross_source" and int(row["seed"]) == seed), None)
    view_row_match = bool(view_row) and view_row.get("status") == "complete" and view_row.get("test_evaluation") == "False" and all(
        view_row.get(key) == value for key, value in row_expected.items()
    )
    cg_row = next((row for row in cg_runs if row["dataset"] == dataset and row["condition"] == "B1_complementary" and int(row["seed"]) == seed), None)
    cg_row_match = bool(cg_row) and cg_row.get("status") == "complete" and cg_row.get("test_evaluation") == "False" and all(
        cg_row.get(key) == value for key, value in row_expected.items()
    ) and cg_manifest.get("protocol_version") == legacy.PROTOCOL
    return {
        "A2_cross_source": {
            "eligible": actual == view_expected and view_row_match,
            "manifest_match": actual == view_expected,
            "run_row_match": view_row_match,
            "expected": view_expected,
            "actual": actual,
        },
        "B1_complementary": {
            "eligible": cg_row_match,
            "run_row_match": cg_row_match,
            "protocol_match": cg_manifest.get("protocol_version") == legacy.PROTOCOL,
            "actual": actual,
            "reference_row": cg_row,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--cpu-only", action="store_true", help="run data/cache/initialization checks without CUDA")
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={args.physical_gpu}; got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    env = legacy.environment_record()
    device = torch.device("cpu" if args.cpu_only else "cuda:0")
    if not args.cpu_only:
        if not env["cuda_available"] or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA unavailable or requested physical device is not isolated")
        legacy._assert_physical_gpu_idle(args.physical_gpu)
    caches = legacy.verify_cache_entries()
    parent_manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    with PARENT_RUNS.open(encoding="utf-8", newline="") as stream:
        view_runs = list(csv.DictReader(stream))
    cg_manifest = json.loads(CG_PARENT_MANIFEST.read_text(encoding="utf-8"))
    with CG_PARENT_RUNS.open(encoding="utf-8", newline="") as stream:
        cg_runs = list(csv.DictReader(stream))
    records: dict[str, Any] = {}

    for dataset in args.datasets:
        cache = caches[dataset]
        by_seed: dict[str, Any] = {}
        for seed in SEEDS:
            cfg = build_config(dataset, CONDITIONS[0], seed, cache, str(device))
            raw = load_mag_data(cfg, "nc", seed)
            safe = mask_test_labels(raw)
            if bool((safe.y[raw.test_idx] != -100).any()):
                raise RuntimeError(f"Test labels were not masked: {dataset}/seed{seed}")
            if not torch.equal(safe.y[raw.train_idx], raw.y[raw.train_idx]) or not torch.equal(safe.y[raw.val_idx], raw.y[raw.val_idx]):
                raise RuntimeError(f"Train/Validation labels changed during masking: {dataset}/seed{seed}")
            fingerprints = legacy.dataset_fingerprints(raw)
            references = _reference_matches(
                dataset, seed, fingerprints, cache, parent_manifest,
                view_runs, cg_manifest, cg_runs,
            )

            init: dict[str, Any] = {}
            for condition in CONDITIONS:
                set_seed(seed)
                cond_cfg = build_config(dataset, condition, seed, cache, "cpu")
                model = load_model(cond_cfg, safe, cache, torch.device("cpu"))
                head = torch.nn.Linear(model.out_dim, int(safe.num_classes))
                init[condition] = {
                    "model_sha256": state_hash(model),
                    "classifier_sha256": state_hash(head),
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "shared_expert_objects": len({id(expert) for expert in model.experts}),
                    "role_film_parameter_count": sum(p.numel() for name, p in model.named_parameters() if "role_film" in name),
                }
                del model, head
            if init[CONDITIONS[0]]["model_sha256"] != init[CONDITIONS[1]]["model_sha256"]:
                raise RuntimeError(f"H1/H2 model initial state mismatch: {dataset}/seed{seed}")
            if init[CONDITIONS[0]]["classifier_sha256"] != init[CONDITIONS[1]]["classifier_sha256"]:
                raise RuntimeError(f"H1/H2 classifier initial state mismatch: {dataset}/seed{seed}")

            by_seed[str(seed)] = {
                "fingerprints": fingerprints,
                "historical_reference_match": references,
                "initialization": init,
                "test_labels_masked": True,
                "train_count": int(raw.train_idx.numel()),
                "validation_count": int(raw.val_idx.numel()),
                "test_count_masked": int(raw.test_idx.numel()),
            }
            del raw, safe

        records[dataset] = {
            "cache_path": cache["cache_path_abs"],
            "cache_sha256": legacy.sha256_file(Path(cache["cache_path_abs"])),
            "feature_fingerprint": cache["feature_fingerprint"],
            "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"],
            "seeds": by_seed,
        }

    result: dict[str, Any] = {
        "status": "cpu_data_and_init_pass" if args.cpu_only else "running",
        "study": "Hybrid-CGE V1.1", "branch": "exp/hybrid_cge_v11",
        "parent_commit": "a7040189ee1237c80ebe2203ca3d204be4c43462",
        "protocol": legacy.PROTOCOL, "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "datasets": records, "environment": env, "physical_gpu": None if args.cpu_only else args.physical_gpu,
        "gpu_training_preflight": None,
    }

    if not args.cpu_only:
        training_checks: dict[str, Any] = {}
        for dataset in args.datasets:
            cache = caches[dataset]
            cfg = build_config(dataset, CONDITIONS[1], 42, cache, str(device))
            raw = load_mag_data(cfg, "nc", 42)
            safe = mask_test_labels(raw)
            x = safe.x.to(device)
            edge = safe.edge_index.to(device)
            y = safe.y.to(device)
            train_idx = safe.train_idx.to(device)
            dataset_checks = {}
            for condition in CONDITIONS:
                torch.cuda.empty_cache()
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(device)
                set_seed(42)
                cond_cfg = build_config(dataset, condition, 42, cache, str(device))
                model = load_model(cond_cfg, safe, cache, device).train()
                head = torch.nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
                optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
                before_cpu = torch.random.get_rng_state().clone()
                before_cuda = torch.cuda.get_rng_state(device).clone()
                model.eval()
                with torch.no_grad():
                    eval_result = model(x, edge)
                eval_rng_unchanged = (
                    torch.equal(before_cpu, torch.random.get_rng_state())
                    and torch.equal(before_cuda, torch.cuda.get_rng_state(device))
                )
                del eval_result
                model.train()
                if not eval_rng_unchanged:
                    raise RuntimeError(f"evaluation changed CPU RNG: {dataset}/{condition}")

                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                z, _, _, aux, info = model(x, edge, return_details=True)
                main_ce = torch.nn.functional.cross_entropy(head(z[train_idx]), y[train_idx])
                uni = 0.5 * (
                    torch.nn.functional.cross_entropy(info["unimodal_logits"]["text"][train_idx], y[train_idx])
                    + torch.nn.functional.cross_entropy(info["unimodal_logits"]["visual"][train_idx], y[train_idx])
                )
                loss = main_ce + 0.3 * uni + aux
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss: {dataset}/{condition}")
                loss.backward()
                gradients = model.capture_gradient_diagnostics()
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"non-finite gradient {name}: {dataset}/{condition}")
                for name, param in head.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"non-finite classifier gradient {name}: {dataset}/{condition}")
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(device)
                for name, param in list(model.named_parameters()) + [
                    (f"classifier.{name}", param) for name, param in head.named_parameters()
                ]:
                    if not torch.isfinite(param).all():
                        raise RuntimeError(f"non-finite post-update parameter {name}: {dataset}/{condition}")

                # Exercise the same state-dict serialization used by formal
                # checkpoints, then prove validation predictions survive reload.
                model.eval()
                head.eval()
                eval_cpu_before = torch.random.get_rng_state().clone()
                eval_cuda_before = torch.cuda.get_rng_state(device).clone()
                with torch.no_grad():
                    z_saved, _, _, _, _ = model(x, edge)
                    logits_saved = head(z_saved).detach().cpu()
                eval_rng_unchanged_after_step = (
                    torch.equal(eval_cpu_before, torch.random.get_rng_state())
                    and torch.equal(eval_cuda_before, torch.cuda.get_rng_state(device))
                )
                if not eval_rng_unchanged_after_step:
                    raise RuntimeError(f"post-update evaluation changed RNG: {dataset}/{condition}")
                roundtrip_path = ROOT / "outputs/hybrid_cge_v11/preflight_checkpoints" / dataset / f"{condition}.pt"
                roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "head": head.state_dict()}, roundtrip_path)
                roundtrip = torch.load(roundtrip_path, map_location="cpu", weights_only=False)
                model.load_state_dict(roundtrip["model"], strict=True)
                head.load_state_dict(roundtrip["head"], strict=True)
                with torch.no_grad():
                    z_loaded, _, _, _, _ = model(x, edge)
                    logits_loaded = head(z_loaded).detach().cpu()
                roundtrip_max_abs_diff = float((logits_saved - logits_loaded).abs().max())
                if not torch.isfinite(logits_loaded).all() or not torch.allclose(
                    logits_saved, logits_loaded, atol=1.0e-6, rtol=1.0e-6
                ):
                    raise RuntimeError(f"checkpoint roundtrip mismatch {dataset}/{condition}: {roundtrip_max_abs_diff}")
                del roundtrip, z_saved, z_loaded, logits_saved, logits_loaded
                dataset_checks[condition] = {
                    "loss": float(loss.detach().cpu()), "main_ce": float(main_ce.detach().cpu()),
                    "unimodal_ce": float(uni.detach().cpu()), "balance_aux": float(aux.detach().cpu()),
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "gradient_diagnostics": gradients,
                    "evaluation_rng_unchanged": eval_rng_unchanged,
                    "post_update_evaluation_rng_unchanged": eval_rng_unchanged_after_step,
                    "checkpoint_roundtrip_max_logit_abs_diff": roundtrip_max_abs_diff,
                    "checkpoint_roundtrip_path": str(roundtrip_path),
                    "detail_diagnostics": info.get("details", {}),
                }
                required = (
                    "physical_cross_coefficients",
                    "cross_projections.visual_to_text.0.weight",
                    "cross_projections.text_to_visual.0.weight",
                    "node_router.3.weight", "view_router.2.weight", "gamma_s_raw",
                    "query_projections.text.weight", "query_projections.visual.weight",
                    "key_projections.text.weight", "key_projections.visual.weight",
                    "source_selector.2.weight", "channel_gate.2.weight",
                    "comp_router.3.weight", "unimodal_heads.text.weight",
                    "unimodal_heads.visual.weight",
                )
                for name in required:
                    if not gradients[name]["gradient_finite"] or not gradients[name]["gradient_nonzero"]:
                        raise RuntimeError(f"required active-path gradient missing: {name}: {dataset}/{condition}")
                if any("role_film" in name for name, _ in model.named_parameters()):
                    raise RuntimeError("Hybrid-CGE V1.1 must not register Role-FiLM parameters")
                # tau may legitimately have zero gradient when the cap does
                # not bind; retain the measured gradient without failing.
                del z, info, optimizer, head, model
                torch.cuda.empty_cache()
            training_checks[dataset] = dataset_checks
            del raw, safe, x, edge, y, train_idx
        result["gpu_training_preflight"] = training_checks
        result["status"] = "pass"
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / ("preflight_cpu.json" if args.cpu_only else "preflight.json")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "path": str(output_path), "datasets": args.datasets}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--cpu-only" not in sys.argv:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            attempt_path = DATA_DIR / "preflight_attempts.json"
            previous = json.loads(attempt_path.read_text()) if attempt_path.exists() else {"attempts": []}
            previous["attempts"].append({
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "physical_gpu": int(sys.argv[sys.argv.index("--physical-gpu") + 1]) if "--physical-gpu" in sys.argv else 1,
                "status": "failed_or_refused",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "test_evaluation": False,
            })
            attempt_path.write_text(json.dumps(previous, indent=2, sort_keys=True), encoding="utf-8")
        raise
