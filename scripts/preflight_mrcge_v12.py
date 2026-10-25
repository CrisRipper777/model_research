#!/usr/bin/env python3
"""CPU dataset/init checks and guarded V1.1 checkpoint regression for MR-CGE V1.2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts import run_mrcge_v12_screen as screen
from scripts import run_mrcge_v11_pilot as v11
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _resolve_nc_eval_labels
from src.utils.seeds import set_seed

STUDY = ROOT / "research/mrcge_v12"
DATA = STUDY / "data"
CORE_DATASETS = ("Movies", "Grocery", "Toys")


def write_json(path: Path, value) -> None:
    screen.json_write(path, value)


def module_hash(module: nn.Module) -> str:
    return screen.state_hash(module)


def cpu_preflight() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    cache_map = screen.cache_entries()
    records = []
    init_by_dataset = {}
    for dataset in (*CORE_DATASETS, "ele-fashion"):
        cache = cache_map[dataset]
        cfg_base = screen.build_cfg(dataset, "S_repaired", screen.SEED, "cpu", cache)
        raw = load_mag_data(cfg_base, "nc", screen.SEED)
        safe = screen.mask_test_labels(raw)
        fingerprints = v11.legacy.dataset_fingerprints(raw)
        if cfg_base.task.evaluate_test is not False:
            raise RuntimeError("Test evaluation must be disabled in the CPU preflight config")
        if bool((safe.y[safe.test_idx] != -100).any()):
            raise RuntimeError("CPU preflight did not mask all Test labels")
        if not torch.equal(safe.y[safe.train_idx], raw.y[raw.train_idx]):
            raise RuntimeError("CPU Test masking changed Train labels")
        if not torch.equal(safe.y[safe.val_idx], raw.y[raw.val_idx]):
            raise RuntimeError("CPU Test masking changed Validation labels")
        for condition in screen.CONDITIONS:
            cfg = screen.build_cfg(dataset, condition, screen.SEED, "cpu", cache)
            set_seed(screen.SEED)
            model = build_model(cfg, screen.model_info(safe))
            cache_meta = model.load_semantic_cache(
                cache["cache_path_abs"], expected_dataset=dataset,
                expected_feature_fingerprint=cache["feature_fingerprint"],
                expected_candidate_fingerprint=cache["candidate_fingerprint"],
                expected_v7b_fingerprint=cache["v7b_fingerprint"],
            )
            head = nn.Linear(model.out_dim, int(safe.num_classes))
            init = {
                "model_sha256": module_hash(model),
                "common_sha256": screen.state_hash(model, screen.COMMON_PREFIXES),
                "classifier_sha256": module_hash(head),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "state_keys": sorted(model.state_dict()),
            }
            init_by_dataset.setdefault(dataset, {})[condition] = init
            records.append({
                "dataset": dataset, "condition": condition, "status": "pass",
                "num_nodes": int(safe.num_nodes), "num_edges": int(safe.edge_index.size(1)),
                "input_dim": int(safe.input_dim), "text_dim": int(safe.x_t.size(1)),
                "visual_dim": int(safe.x_i.size(1)), "num_classes": int(safe.num_classes),
                "feature_fingerprint": cache["feature_fingerprint"],
                "physical_edge_fingerprint": fingerprints["physical_edge_index"],
                "split_fingerprint": fingerprints["split_fingerprint"],
                "semantic_cache_path": cache["cache_path_abs"],
                "semantic_cache_sha256": screen.sha256_file(Path(cache["cache_path_abs"])),
                "candidate_fingerprint": cache["candidate_fingerprint"],
                "v7b_fingerprint": cache["v7b_fingerprint"],
                "cache_num_nodes": int(cache_meta["num_nodes"]),
                "test_labels_masked": True, "test_evaluation": False,
                **init,
            })
            del model, head
        del raw, safe

    init_checks = {}
    for dataset, conditions in init_by_dataset.items():
        s, f = conditions["S_repaired"], conditions["F_sinit"]
        mr, ma = conditions["M_repaired"], conditions["M_anchor"]
        init_checks[dataset] = {
            "S_F_common_match": s["common_sha256"] == f["common_sha256"],
            "S_F_external_classifier_match": s["classifier_sha256"] == f["classifier_sha256"],
            "S_F_parameter_delta_is_theta_only": f["parameter_count"] - s["parameter_count"] == 4,
            "M_repaired_M_anchor_common_match": mr["common_sha256"] == ma["common_sha256"],
            "M_repaired_M_anchor_state_keys_match": mr["state_keys"] == ma["state_keys"],
        }
        if not all(init_checks[dataset].values()):
            raise RuntimeError(f"condition initialization audit failed for {dataset}: {init_checks[dataset]}")

    payload = {
        "cpu_status": "pass", "recorded_utc": screen.now(),
        "source_git_sha": screen.source_sha(),
        "working_tree_dirty_at_cpu_preflight": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()),
        "physical_hops": 3, "semantic_hops": 1,
        "dataset_cache_records": records, "initialization_checks": init_checks,
        "unit_test_command": "/home/m3/miniconda3/envs/yhf_env/bin/python -m pytest -q tests/test_mrcge_v12.py",
        "test_evaluation": False,
    }
    write_json(DATA / "preflight.json", payload)
    write_json(DATA / "manifest.json", {
        "study": "MR-CGE V1.2 Shared Graph Filter Performance Optimization",
        "branch": "exp/mrcge_v12_anchor_performance_screen",
        "base_sha": "0e2496efe852c76fcd06c1477de12f41ac085e7c",
        "training_frozen_sha": None, "datasets": [*CORE_DATASETS, "ele-fashion"],
        "conditions": list(screen.CONDITIONS), "seeds": [42], "physical_gpu": 1,
        "test_evaluation": False, "test_label_values_indexed": False,
        "formal_training_started": False, "test_labels_were_not_mutated": True,
    })
    return payload


def regression_cfg(dataset: str, condition: str, cache: dict, legacy: bool):
    cfg = screen.build_cfg(dataset, condition, screen.SEED, "cuda:0", cache)
    if legacy:
        cfg.model = OmegaConf.load(ROOT / "configs/model/mrcge_v11.yaml")
        cfg.model.name = "mrcge_v11"
        cfg.model.condition = "M"
        cfg.model.semantic_cache_path = cache["cache_path_abs"]
        cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
        cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
        cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    return cfg


def regression_worker(physical_gpu: int = 1) -> dict:
    if __import__("os").environ.get("MRCGE_GUARDED_REGRESSION") != "1" or \
            __import__("os").environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("V1.1 checkpoint regression must be launched by the GPU1 ownership guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("regression worker requires isolated GPU1")
    baseline = {int(v) for v in __import__("os").environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if v}
    screen.guard_gpu(physical_gpu, baseline, {__import__("os").getpid()}, phase="before_launch")
    cache_map = screen.cache_entries()
    records = []
    device = torch.device("cuda:0")
    for dataset in CORE_DATASETS:
        old_row = next((r for r in screen.read_rows(ROOT / "research/mrcge_v11/data/runs.csv")
                        if r.get("dataset") == dataset and r.get("condition") == "M"
                        and r.get("seed") == "42" and r.get("status") == "complete"), None)
        if old_row is None:
            raise RuntimeError(f"no verified V1.1 M checkpoint row for {dataset}")
        checkpoint = Path(old_row["checkpoint_path"])
        if not checkpoint.is_file() or screen.sha256_file(checkpoint) != old_row["checkpoint_sha256"]:
            raise RuntimeError(f"V1.1 checkpoint integrity mismatch: {checkpoint}")
        cache = cache_map[dataset]
        cfg_v11 = regression_cfg(dataset, "M_repaired", cache, True)
        cfg_v12 = regression_cfg(dataset, "M_repaired", cache, False)
        raw = load_mag_data(cfg_v12, "nc", screen.SEED)
        safe = screen.mask_test_labels(raw)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        set_seed(screen.SEED)
        old_model = build_model(cfg_v11, screen.model_info(safe))
        old_model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
        old_model.load_state_dict(payload["model_state"], strict=True)
        old_model = old_model.to(device).eval()
        set_seed(screen.SEED)
        new_model = build_model(cfg_v12, screen.model_info(safe))
        new_model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
        result = new_model.load_state_dict(payload["model_state"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"V1.1/V1.2 strict state_dict mismatch for {dataset}")
        new_model = new_model.to(device).eval()
        old_head = nn.Linear(old_model.out_dim, int(safe.num_classes)).to(device)
        new_head = nn.Linear(new_model.out_dim, int(safe.num_classes)).to(device)
        old_head.load_state_dict(payload["head_state"], strict=True)
        new_head.load_state_dict(payload["head_state"], strict=True)
        x, edges = safe.x.to(device), safe.edge_index.to(device)
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        with torch.no_grad():
            old_z, _, _, _, _ = old_model(x, edges, return_details=False)
            old_logits = old_head(old_z).float().cpu()
            old_z_cpu = old_z.float().cpu()
            del old_z
            old_repeat_z, _, _, _, _ = old_model(x, edges, return_details=False)
            old_repeat_logits = old_head(old_repeat_z).float().cpu()
            old_repeat_z_cpu = old_repeat_z.float().cpu()
            del old_repeat_z
            new_z, _, _, _, _ = new_model(x, edges, return_details=False)
            new_logits = new_head(new_z).float().cpu()
            new_z_cpu = new_z.float().cpu()
            del new_z
        rng_unchanged = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(
            cuda_rng, torch.cuda.get_rng_state(device)
        )
        max_abs_z = float((old_z_cpu - new_z_cpu).abs().max())
        max_abs_logits = float((old_logits - new_logits).abs().max())
        old_repeat_embedding_error = float((old_z_cpu - old_repeat_z_cpu).abs().max())
        old_repeat_logit_error = float((old_logits - old_repeat_logits).abs().max())
        z_close = bool(torch.allclose(old_z_cpu, new_z_cpu, atol=1.0e-5, rtol=1.0e-4))
        logits_close = bool(torch.allclose(old_logits, new_logits, atol=1.0e-5, rtol=1.0e-4))
        z_bad = ~torch.isclose(old_z_cpu, new_z_cpu, atol=1.0e-5, rtol=1.0e-4)
        logits_bad = ~torch.isclose(old_logits, new_logits, atol=1.0e-5, rtol=1.0e-4)
        explained_by_cuda_scatter_variation = (
            max_abs_z <= 1.25 * old_repeat_embedding_error + 1.0e-5
            and max_abs_logits <= 1.25 * old_repeat_logit_error + 1.0e-5
        )
        old_val = screen.metric_bundle(old_logits, safe.y, safe.val_idx, _resolve_nc_eval_labels(safe, include_test=False))
        new_val = screen.metric_bundle(new_logits, safe.y, safe.val_idx, _resolve_nc_eval_labels(safe, include_test=False))
        metric_match = all(old_val[key] == new_val[key] for key in ("acc", "macro_f1")) and abs(old_val["ce"] - new_val["ce"]) <= 1e-6
        record = {
            "dataset": dataset, "condition": "M_repaired", "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": old_row["checkpoint_sha256"], "strict_state_dict": True,
            "state_keys_match": list(old_model.state_dict()) == list(new_model.state_dict()),
            "old_v11_val": old_val, "new_v12_val": new_val,
            "validation_metrics_match": metric_match, "max_abs_embedding_error": max_abs_z,
            "max_abs_validation_logit_error": max_abs_logits,
            "v11_repeat_max_abs_embedding_variation": old_repeat_embedding_error,
            "v11_repeat_max_abs_logit_variation": old_repeat_logit_error,
            "embedding_elements_outside_atol1e-5_rtol1e-4": int(z_bad.sum()),
            "embedding_element_count": int(z_bad.numel()),
            "logit_elements_outside_atol1e-5_rtol1e-4": int(logits_bad.sum()),
            "logit_element_count": int(logits_bad.numel()),
            "embeddings_allclose_atol1e-5_rtol1e-4": z_close,
            "logits_allclose_atol1e-5_rtol1e-4": logits_close,
            "difference_explained_by_repeated_v11_cuda_scatter_variation": explained_by_cuda_scatter_variation,
            "validation_rng_unchanged": rng_unchanged,
            "test_labels_masked": bool((safe.y[safe.test_idx] == -100).all()),
            "test_evaluation": False,
        }
        if not (record["strict_state_dict"] and record["state_keys_match"] and metric_match
                and rng_unchanged and (z_close and logits_close or explained_by_cuda_scatter_variation)):
            raise RuntimeError(f"V1.1 M_repaired eval regression failed for {dataset}: {record}")
        records.append(record)
        del raw, safe, old_model, new_model, old_head, new_head, payload, x, edges, old_z_cpu, new_z_cpu, old_logits, new_logits, old_repeat_z_cpu, old_repeat_logits
        torch.cuda.empty_cache()
        screen.guard_gpu(physical_gpu, baseline, {__import__("os").getpid()}, phase="during_run")
    result = {
        "recorded_utc": screen.now(), "source_git_sha": screen.source_sha(),
        "physical_gpu": physical_gpu, "status": "pass", "regression_records": records,
        "test_evaluation": False,
    }
    write_json(DATA / "v11_checkpoint_regression.json", result)
    return result


def run_guarded_regression(physical_gpu: int = 1) -> dict:
    if physical_gpu != 1:
        raise ValueError("this task authorizes regression checks on physical GPU1 only")
    baseline_pids = screen.capture_gpu_baseline(physical_gpu)
    before = screen.guard_gpu(physical_gpu, baseline_pids, set(), phase="before_launch",
                              required_free_mib=8192)
    screen.append_guard_resource(before, dataset="V1.1_regression", condition="M_repaired",
                                 seed=42, attempt=None, stage="regression_before_launch")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["MRCGE_GUARDED_REGRESSION"] = "1"
    env["MRCGE_BASELINE_GPU_PIDS"] = ",".join(str(value) for value in sorted(baseline_pids))
    log_path = ROOT / "outputs/mrcge_v12_screen/preflight_logs/v11_checkpoint_regression_gpu1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--regression-worker",
               "--physical-gpu", str(physical_gpu)]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(5)
            try:
                snapshot = screen.guard_gpu(physical_gpu, baseline_pids, {process.pid}, phase="during_run")
                screen.append_guard_resource(snapshot, dataset="V1.1_regression", condition="M_repaired",
                                             seed=42, attempt=None, stage="regression_during_run")
            except Exception as exc:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                record = {
                    "stage": "shared_gpu_runtime_guard", "dataset": "V1.1_regression",
                    "condition": "M_repaired", "physical_gpu": physical_gpu,
                    "training_started": False, "test_evaluation": False,
                    "log_path": str(log_path), "error": repr(exc),
                    "terminated_own_child_only": True,
                }
                screen.append_failure(record)
                raise RuntimeError(f"guarded V1.1 regression stopped: {exc}") from exc
    if process.returncode != 0:
        error = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        screen.append_failure({"stage": "v11_checkpoint_regression", "dataset": "all",
                               "physical_gpu": physical_gpu, "training_started": False,
                               "test_evaluation": False, "log_path": str(log_path),
                               "error": f"regression child exited {process.returncode}", "tail": error})
        raise RuntimeError(f"V1.1 checkpoint regression failed; see {log_path}")
    return json.loads((DATA / "v11_checkpoint_regression.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--regression-worker", action="store_true")
    parser.add_argument("--run-regression", action="store_true")
    parser.add_argument("--physical-gpu", type=int, default=1)
    args = parser.parse_args()
    if args.cpu:
        result = cpu_preflight()
        print(json.dumps({"cpu_status": result["cpu_status"], "datasets": len(result["dataset_cache_records"])}))
    elif args.regression_worker:
        result = regression_worker(args.physical_gpu)
        print(json.dumps({"status": result["status"], "datasets": len(result["regression_records"])}))
    elif args.run_regression:
        result = run_guarded_regression(args.physical_gpu)
        print(json.dumps({"status": result["status"], "datasets": len(result["regression_records"])}))
    else:
        parser.error("select --cpu or the guarded internal --regression-worker action")


if __name__ == "__main__":
    main()
