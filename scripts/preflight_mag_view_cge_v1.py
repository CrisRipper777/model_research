#!/usr/bin/env python3
"""Validate View-CGE V1 on real full graphs before the implementation freeze."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
CONDITIONS = ("A0_legacy", "A1_separate_intra", "A2_cross_source")
DATA_DIR = ROOT / "research/mag_view_cge_v1/data"


def config(dataset: str, condition: str, seed: int, model_name: str, cache: dict, device: str):
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create({
        "paths": root.paths,
        "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
        "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
        "model": OmegaConf.load(ROOT / f"configs/model/{model_name}.yaml"),
        "logging": root.logging,
        "seed": seed,
        "num_runs": 1,
        "device": device,
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
    cfg.task.record_extended_validation = True
    cfg.task.record_train_ce_separately = True
    cfg.paths.output_root = str(ROOT / "outputs/mag_view_cge_v1/hydra")
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


def load_model(cfg, data, cache, device):
    model = build_model(cfg, legacy.data_info(data))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=cfg.dataset.name,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    return model.to(device)


def build_head(model, data, device):
    return torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--skip-training-step", action="store_true")
    args = parser.parse_args()
    env = legacy.environment_record()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {str(args.physical_gpu), f"{args.physical_gpu},"}:
        raise RuntimeError(f"launch must isolate physical GPU {args.physical_gpu}; CUDA_VISIBLE_DEVICES={visible!r}")
    if not env["cuda_available"] or torch.cuda.device_count() != 1:
        raise RuntimeError("expected one CUDA device after selecting the requested physical GPU")
    legacy._assert_physical_gpu_idle(args.physical_gpu)
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    scratch = torch.empty(1, device=device)
    del scratch
    torch.cuda.synchronize(device)

    caches = legacy.verify_cache_entries()
    records: dict[str, object] = {
        "status": "running", "physical_gpu": args.physical_gpu, "logical_device": str(device),
        "device_name": device_name, "environment": env,
        "device_nodes_visible": [str(p) for p in Path("/dev").glob("nvidia*")],
        "datasets": {}, "formal_runs_started": False,
    }

    for dataset in args.datasets:
        cfg_data = config(dataset, "A0_legacy", 42, "mag_view_cge_v1", caches[dataset], str(device))
        raw = load_mag_data(cfg_data, "nc", 42)
        safe = legacy.mask_test_labels(raw)
        fingerprint = legacy.dataset_fingerprints(raw)
        if bool((safe.y[raw.test_idx] != -100).any()):
            raise RuntimeError(f"Test labels were not masked for {dataset}")
        if not torch.equal(safe.y[raw.train_idx], raw.y[raw.train_idx]) or not torch.equal(safe.y[raw.val_idx], raw.y[raw.val_idx]):
            raise RuntimeError(f"masking changed Train/Validation labels for {dataset}")

        # Real-data A0 forward comparison with the actual legacy model class.
        set_seed(42)
        old_cfg = config(dataset, "M0", 42, "rmcge_v22", caches[dataset], str(device))
        old = load_model(old_cfg, safe, caches[dataset], device).eval()
        old_head = build_head(old, safe, device)
        old_model_hash = state_hash(old)
        set_seed(42)
        new_cfg = config(dataset, "A0_legacy", 42, "mag_view_cge_v1", caches[dataset], str(device))
        new = load_model(new_cfg, safe, caches[dataset], device).eval()
        new_head = build_head(new, safe, device)
        new_model_hash = state_hash(new)
        common_parameter_errors = [
            (dict(old.named_parameters())[name] - dict(new.named_parameters())[name]).abs().max()
            for name, _ in old.named_parameters()
        ]
        x = safe.x.to(device)
        edge = safe.edge_index.to(device)
        with torch.no_grad():
            old_z = old(x, edge)[0]
            new_z = new(x, edge)[0]
        max_forward_error = float((old_z - new_z).abs().max().cpu())
        max_common_parameter_error = max(float(error.cpu()) for error in common_parameter_errors)
        if max_forward_error > 5.0e-5 or max_common_parameter_error != 0.0:
            raise RuntimeError(f"A0 compatibility failed for {dataset}: output={max_forward_error}, params={max_common_parameter_error}")
        if not torch.equal(old_head.weight, new_head.weight) or not torch.equal(old_head.bias, new_head.bias):
            raise RuntimeError(f"A0 NC-head initialization differs from the V2.2 M0 head for {dataset}")
        base_record = {
            "dataset_fingerprints": fingerprint,
            "cache_sha256": legacy.sha256_file(Path(caches[dataset]["cache_path_abs"])),
            "cache_fingerprints": {k: caches[dataset][k] for k in ("feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint")},
            "num_nodes": raw.num_nodes, "num_edges": raw.num_edges,
            "train_count": int(raw.train_idx.numel()), "validation_count": int(raw.val_idx.numel()),
            "test_count_masked": int(raw.test_idx.numel()),
            "a0_forward_max_abs_error": max_forward_error,
            "a0_common_parameter_max_abs_error": max_common_parameter_error,
            "old_model_hash": old_model_hash, "new_model_hash": new_model_hash,
            "a0_classifier_weight_sha256": hashlib.sha256(new_head.weight.detach().cpu().numpy().tobytes()).hexdigest(),
            "a0_classifier_bias_sha256": hashlib.sha256(new_head.bias.detach().cpu().numpy().tobytes()).hexdigest(),
        }
        del old, old_head, old_z, new_z, new_head
        del old_cfg, new_cfg

        # Compare all three public model/head initial states under one seed.
        init_hashes = {}
        for condition in CONDITIONS:
            set_seed(42)
            cfg = config(dataset, condition, 42, "mag_view_cge_v1", caches[dataset], str(device))
            model = load_model(cfg, safe, caches[dataset], device)
            head = build_head(model, safe, device)
            init_hashes[condition] = {"model": state_hash(model), "head": state_hash(head)}
            del model, head
        if len({v["model"] for v in init_hashes.values()}) != 1 or len({v["head"] for v in init_hashes.values()}) != 1:
            raise RuntimeError(f"condition initialization mismatch for {dataset}/seed42")
        base_record["condition_initialization_hashes"] = init_hashes

        # Evaluation must leave both CPU and selected-device RNG untouched.
        set_seed(42)
        eval_cfg = config(dataset, "A2_cross_source", 42, "mag_view_cge_v1", caches[dataset], str(device))
        eval_model = load_model(eval_cfg, safe, caches[dataset], device).eval()
        eval_head = build_head(eval_model, safe, device).eval()
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(device).clone()
        with torch.no_grad():
            eval_model(x, edge)
        rng_unchanged = torch.equal(cpu_before, torch.random.get_rng_state()) and torch.equal(cuda_before, torch.cuda.get_rng_state(device))
        if not rng_unchanged:
            raise RuntimeError(f"evaluation changed CPU/CUDA RNG state for {dataset}")
        base_record["validation_rng_unchanged"] = rng_unchanged
        del eval_model, eval_head

        if not args.skip_training_step:
            steps = {}
            # Full-graph one-step safety/resource check of both active condition sets.
            x_gpu = x
            edge_gpu = edge
            y_gpu = safe.y.to(device)
            train_idx = safe.train_idx.to(device)
            for condition in ("A1_separate_intra", "A2_cross_source"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                set_seed(42)
                cfg = config(dataset, condition, 42, "mag_view_cge_v1", caches[dataset], str(device))
                model = load_model(cfg, safe, caches[dataset], device).train()
                head = build_head(model, safe, device).train()
                optimizer = torch.optim.AdamW(
                    list(model.parameters()) + list(head.parameters()), lr=1.0e-3, weight_decay=1.0e-4
                )
                initial_gamma_s = model.gamma_s_raw.detach().clone()
                initial_gamma_c = model.gamma_c_raw.detach().clone()
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                z, _, _, aux, _ = model(x_gpu, edge_gpu)
                loss = torch.nn.functional.cross_entropy(head(z[train_idx]), y_gpu[train_idx]) + aux
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite {condition} loss on {dataset}")
                loss.backward()
                gradients = model.capture_gradient_diagnostics()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"non-finite gradient {name} for {dataset}/{condition}")
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                required = ["gamma_s_raw", "view_router.2.weight"]
                if condition == "A2_cross_source":
                    required += ["gamma_c_raw"]
                for name in required:
                    if not gradients.get(name, {}).get("gradient_nonzero", False):
                        raise RuntimeError(f"required first-step gradient is zero: {dataset}/{condition}/{name}")
                if condition == "A2_cross_source":
                    cross_grad = model.cross_projections["visual_to_text"][0].weight.grad
                    if cross_grad is None or not torch.isfinite(cross_grad).all() or float(cross_grad.detach().float().norm()) <= 0.0:
                        raise RuntimeError("A2 cross projection gradient is missing, non-finite, or zero")
                if torch.equal(initial_gamma_s, model.gamma_s_raw.detach()):
                    raise RuntimeError(f"gamma_S did not update for {dataset}/{condition}")
                if condition == "A2_cross_source" and torch.equal(initial_gamma_c, model.gamma_c_raw.detach()):
                    raise RuntimeError(f"gamma_C did not update for {dataset}/{condition}")
                steps[condition] = {
                    "loss": float(loss.detach().cpu()), "seconds": elapsed,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "gradient_diagnostics": gradients,
                    "cross_projection_gradient_norm": float(model.cross_projections["visual_to_text"][0].weight.grad.detach().float().norm().cpu()) if condition == "A2_cross_source" else None,
                    "gamma_s_updated": not torch.equal(initial_gamma_s, model.gamma_s_raw.detach()),
                    "gamma_c_updated": not torch.equal(initial_gamma_c, model.gamma_c_raw.detach()),
                }
                del z, loss, optimizer, head, model
                torch.cuda.empty_cache()
            base_record["full_graph_training_step"] = steps
        records["datasets"][dataset] = base_record
        del raw, safe, x, edge
        torch.cuda.empty_cache()

    records["status"] = "pass"
    records["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "preflight.json"
    path.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": records["status"], "gpu": device_name, "path": str(path), "datasets": list(records["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
