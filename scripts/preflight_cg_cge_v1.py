#!/usr/bin/env python3
"""Data, pairing, and optional real-GPU preflight for CG-CGE V1."""

from __future__ import annotations

import argparse
import copy
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
CONDITIONS = ("B1_complementary", "B2_conditioned_experts")
SEEDS = (42, 43, 44)
STUDY = ROOT / "research/cg_cge_v1"
DATA_DIR = STUDY / "data"
PARENT_MANIFEST = ROOT / "research/mag_view_cge_v1/data/manifest.json"
PARENT_RUNS = ROOT / "research/mag_view_cge_v1/data/runs.csv"


def build_config(dataset: str, condition: str, seed: int, cache: dict[str, Any], device: str):
    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create({
        "paths": root_cfg.paths,
        "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
        "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
        "model": OmegaConf.load(ROOT / "configs/model/cg_cge_v1.yaml"),
        "logging": root_cfg.logging,
        "seed": int(seed), "num_runs": 1, "device": str(device),
    })
    cfg.dataset.name = dataset
    cfg.model.condition = condition
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
        "study": "CG-CGE V1", "dataset": dataset, "condition": condition,
        "seed": int(seed), "official_train_val_split": True,
        "test_evaluation": False, "test_label_values_indexed": False,
    }
    cfg.paths.output_root = str(ROOT / "outputs/cg_cge_v1/hydra")
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


def _b0_matches(dataset: str, seed: int, fingerprints: dict, cache: dict, parent_manifest: dict) -> dict[str, Any]:
    parent = parent_manifest["datasets"][dataset]
    expected = {
        "split_fingerprint": parent["seed_split_fingerprints"][str(seed)],
        "feature_fingerprint": parent["feature_fingerprint"],
        "raw_features_fingerprint": parent["features_fingerprint"],
        "physical_edge_fingerprint": parent["physical_edge_fingerprint"],
        "cache_sha256": parent["cache_sha256"],
        "candidate_fingerprint": parent["candidate_fingerprint"],
        "v7b_fingerprint": parent["v7b_fingerprint"],
        "protocol": parent_manifest["protocol_version"],
    }
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
    return {"eligible": expected == actual, "expected": expected, "actual": actual}


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
            b0 = _b0_matches(dataset, seed, fingerprints, cache, parent_manifest)

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
                    "role_film_parameters": model.role_film_scale.numel() + model.role_film_shift.numel(),
                    "role_film_trainable_parameters": sum(p.numel() for p in (model.role_film_scale, model.role_film_shift) if p.requires_grad),
                }
                del model, head
            if init[CONDITIONS[0]]["model_sha256"] != init[CONDITIONS[1]]["model_sha256"]:
                raise RuntimeError(f"B1/B2 model initial state mismatch: {dataset}/seed{seed}")
            if init[CONDITIONS[0]]["classifier_sha256"] != init[CONDITIONS[1]]["classifier_sha256"]:
                raise RuntimeError(f"B1/B2 classifier initial state mismatch: {dataset}/seed{seed}")

            by_seed[str(seed)] = {
                "fingerprints": fingerprints,
                "b0_match": b0,
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
        "study": "CG-CGE V1", "branch": "exp/cg_cge_v1_complementary_experts",
        "parent_commit": "ffdb8c2b53ea4c67793d26c420832aa9a118cb58",
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
                z, _, _, aux, info = model(x, edge)
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
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(device)
                dataset_checks[condition] = {
                    "loss": float(loss.detach().cpu()), "main_ce": float(main_ce.detach().cpu()),
                    "unimodal_ce": float(uni.detach().cpu()), "balance_aux": float(aux.detach().cpu()),
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "gradient_diagnostics": gradients,
                    "evaluation_rng_unchanged": eval_rng_unchanged,
                }
                if not gradients["query_projections.text.weight"]["gradient_nonzero"]:
                    raise RuntimeError(f"receiver query got no gradient: {dataset}/{condition}")
                if not gradients["view_router.2.weight"]["gradient_nonzero"]:
                    raise RuntimeError(f"physical view router got no gradient: {dataset}/{condition}")
                if condition == "B2_conditioned_experts" and not gradients["role_film_scale"]["gradient_nonzero"]:
                    raise RuntimeError(f"B2 Role-FiLM got no gradient: {dataset}")
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
