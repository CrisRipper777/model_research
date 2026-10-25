#!/usr/bin/env python3
"""Run the frozen MR-CGE V1.2 validation-only performance screen."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from scripts import run_rmcge_v22 as legacy
from src.data import load_mag_data
from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b
from src.models import build_model
from src.tasks.inference import infer_all_embeddings
from src.tasks.nc import _compose_full_graph_loss, _resolve_nc_eval_labels, run_nc
from src.utils.seeds import set_seed


STUDY = ROOT / "research/mrcge_v12"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v12_screen"
RUNS = OUTPUT / "runs"
DATASETS = ("Movies", "Grocery", "Toys", "ele-fashion")
CORE_DATASETS = ("Movies", "Grocery", "Toys")
BONUS_DATASET = "ele-fashion"
CONDITIONS = ("S_repaired", "F_sinit", "M_repaired", "M_anchor")
SEED = 42
PROTOCOL = "unified_full_graph_nc_v1"
MODEL_NAME = "mrcge_v12"
PHYSICAL_GPU = 1
RUN_FIELDS = (
    "dataset", "condition", "seed", "attempt", "status", "best_acc_epoch", "best_f1_epoch",
    "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
    "best_f1_val_ce", "train_acc", "train_ce", "model_parameters", "total_parameters",
    "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "source_git_sha",
    "config_sha256", "model_source_sha256", "feature_fingerprint", "physical_edge_fingerprint",
    "dataset_fingerprint", "split_fingerprint", "semantic_cache_sha256", "v7b_fingerprint",
    "initial_common_state_sha256", "initial_model_sha256", "initial_classifier_sha256",
    "attempt_dir", "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "test_evaluation", "validation_rng_unchanged", "error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default), encoding="utf-8")
    temp.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def append_csv(path: Path, row: dict[str, Any], fields: tuple[str, ...], key_fields: tuple[str, ...]) -> None:
    rows = read_rows(path)
    key = tuple(str(row.get(name, "")) for name in key_fields)
    if any(tuple(old.get(name, "") for name in key_fields) == key for old in rows):
        raise RuntimeError(f"refusing to overwrite existing CSV attempt key {key}")
    all_fields = list(dict.fromkeys([*fields, *(name for old in rows for name in old), *row.keys()]))
    write_rows(path, [*rows, row], all_fields)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(module: nn.Module, prefixes: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if prefixes is not None and not name.startswith(prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


COMMON_PREFIXES = (
    "projectors.", "cross_projections.", "query_projections.", "key_projections.",
    "source_selector.", "channel_gate.", "cross_lambda_raw", "tau_raw", "rho_raw",
    "gamma_raw", "beta_raw", "unimodal_heads.", "fusion_",
)


def model_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def mask_test_labels(data):
    safe = copy.copy(data)
    safe.y = data.y.clone()
    safe.y[data.test_idx] = -100
    if bool((safe.y[data.test_idx] != -100).any()):
        raise RuntimeError("Test labels were not set to -100 before Trainer entry")
    if not torch.equal(safe.y[data.train_idx], data.y[data.train_idx]):
        raise RuntimeError("Test masking changed Train labels")
    if not torch.equal(safe.y[data.val_idx], data.y[data.val_idx]):
        raise RuntimeError("Test masking changed Validation labels")
    return safe


def build_cfg(dataset: str, condition: str, seed: int, device: str, cache: dict[str, Any]):
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create({
        "paths": root.paths,
        "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
        "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
        "model": OmegaConf.load(ROOT / "configs/model/mrcge_v12.yaml"),
        "logging": root.logging,
        "seed": int(seed), "num_runs": 1, "device": str(device),
    })
    cfg.dataset.name = dataset
    cfg.model.name = MODEL_NAME
    cfg.model.condition = condition
    cfg.model.semantic_cache_path = cache["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    cfg.task.name = "nc"
    cfg.task.protocol_version = PROTOCOL
    cfg.task.training_mode = "full_graph"
    cfg.task.optimizer = "adamw"
    cfg.task.lr = 1.0e-3
    cfg.task.weight_decay = 1.0e-4
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 1.0e-4
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.inference_mode = "full"
    cfg.task.inference_batch_size = 4096
    cfg.task.scheduler = None
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 1.0
    cfg.task.loss.unimodal_weight = 0.3
    cfg.task.record_extended_validation = True
    cfg.task.record_train_ce_separately = True
    cfg.task.best_f1_ckpt_path = None
    cfg.paths.output_root = str(OUTPUT / "hydra")
    OmegaConf.resolve(cfg)
    return cfg


def cache_entries() -> dict[str, dict[str, Any]]:
    entries = legacy.verify_cache_entries()
    # Toys was already covered by the V7B.1 cache used in the validation
    # expansion. Re-verify that cache against the current raw features before
    # exposing it to this pilot; the legacy verifier is intentionally limited
    # to the original three-dataset campaign.
    toys_audit_path = ROOT / "research/hybrid_cge_validation_expansion_v1/data/data_audit.json"
    if toys_audit_path.is_file():
        toys_data_audit = json.loads(toys_audit_path.read_text(encoding="utf-8"))["datasets"]["Toys"]
        for source in toys_data_audit["source_files"].values():
            path = Path(source["path"])
            if not path.is_file() or sha256_file(path) != source["sha256"]:
                raise RuntimeError(f"Toys source file differs from the existing V7B.1 cache audit: {path}")
        toys_cache = toys_data_audit["semantic_cache"]
        cache_path = Path(toys_cache["path"])
        if not cache_path.is_file() or sha256_file(cache_path) != toys_cache["sha256"]:
            raise RuntimeError(f"Toys V7B.1 cache differs from its recorded audit: {cache_path}")
        cache_payload = load_semantic_candidate_cache_v7b(
            cache_path,
            expected_dataset="Toys",
            expected_feature_fingerprint=toys_cache["feature_fingerprint"],
            expected_candidate_fingerprint=toys_cache["candidate_fingerprint"],
            expected_v7b_fingerprint=toys_cache["v7b_fingerprint"],
        )
        metadata = cache_payload["metadata"]
        if metadata.get("candidate_phase_label_access") is not False or metadata.get("cache_contains_labels") is not False:
            raise RuntimeError("Toys V7B.1 cache does not certify label-free candidate construction")
        entries["Toys"] = {
            "dataset": "Toys", **toys_cache,
            "cache_path_abs": str(cache_path.resolve()),
            "cache_audit": {
                "dataset": "Toys", "cache_path": str(cache_path.resolve()),
                "num_nodes": int(metadata["num_nodes"]),
                "feature_fingerprint": toys_cache["feature_fingerprint"],
                "candidate_fingerprint": toys_cache["candidate_fingerprint"],
                "v7b_fingerprint": toys_cache["v7b_fingerprint"],
                "candidate_phase_label_access": metadata.get("candidate_phase_label_access"),
                "cache_contains_labels": metadata.get("cache_contains_labels"),
            },
        }
    missing = [dataset for dataset in DATASETS if dataset not in entries]
    if missing:
        raise RuntimeError(f"verified V7B.1 semantic caches are missing for {missing}")
    return entries


def source_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def initial_state_audit(cfg, data, cache, device: torch.device) -> dict[str, str]:
    set_seed(int(cfg.seed))
    model = build_model(cfg, model_info(data))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    return {
        "initial_model_sha256": state_hash(model),
        "initial_common_state_sha256": state_hash(model, COMMON_PREFIXES),
        "initial_classifier_sha256": state_hash(head),
        "model_parameters": str(sum(p.numel() for p in model.parameters())),
        "total_parameters": str(sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())),
    }


def smi_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    devices = []
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) != 8:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
        devices.append({
            "physical_index": int(columns[0]), "uuid": columns[1], "name": columns[2],
            "memory_used_mib": int(columns[3]), "memory_total_mib": int(columns[4]),
            "memory_free_mib": int(columns[5]), "utilization_percent": int(columns[6]),
            "temperature_c": int(columns[7]),
        })
    apps_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    apps = []
    for line in apps_result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) >= 2:
            try:
                apps.append({"gpu_uuid": columns[0], "pid": int(columns[1]), "used_memory_mib": int(columns[2]) if len(columns) > 2 else None})
            except ValueError:
                continue
    return {"captured_utc": now(), "devices": devices, "compute_processes": apps}


def guard_gpu(physical_gpu: int, baseline_pids: set[int], allowed_pids: set[int], *, phase: str,
              required_free_mib: int | None = None) -> dict[str, Any]:
    snap = smi_snapshot()
    selected = next((item for item in snap["devices"] if item["physical_index"] == physical_gpu), None)
    if selected is None:
        raise RuntimeError(f"physical GPU {physical_gpu} is absent from nvidia-smi")
    current = {
        app["pid"] for app in snap["compute_processes"]
        if app["gpu_uuid"] == selected["uuid"]
    }
    permitted = baseline_pids | allowed_pids
    unexpected = sorted(current - permitted)
    if unexpected:
        raise RuntimeError(f"shared-GPU guard saw new foreign compute PIDs {unexpected}; refusing/aborting this run")
    minimum_free = required_free_mib if required_free_mib is not None else (8192 if phase == "before_launch" else 2048)
    if selected["memory_free_mib"] < minimum_free:
        raise RuntimeError(f"GPU{physical_gpu} free memory {selected['memory_free_mib']} MiB below {minimum_free} MiB guard threshold")
    if selected["temperature_c"] >= (88 if phase == "before_launch" else 91):
        raise RuntimeError(f"GPU{physical_gpu} temperature {selected['temperature_c']} C exceeds the {phase} limit")
    if phase == "before_launch" and selected["utilization_percent"] >= 90:
        raise RuntimeError(f"GPU{physical_gpu} baseline utilization {selected['utilization_percent']}% is too high to start a shared run")
    snap["guard"] = {
        "phase": phase, "physical_gpu": physical_gpu, "baseline_compute_pids": sorted(baseline_pids),
        "allowed_process_pids": sorted(allowed_pids), "current_compute_pids": sorted(current),
        "unexpected_compute_pids": unexpected, "required_free_mib": minimum_free, "status": "pass",
    }
    return snap


def append_guard_resource(snapshot: dict[str, Any], *, dataset: str, condition: str,
                          seed: int, attempt: int | None, stage: str) -> None:
    selected = next(item for item in snapshot["devices"] if item["physical_index"] == PHYSICAL_GPU)
    pids = sorted(app["pid"] for app in snapshot["compute_processes"] if app["gpu_uuid"] == selected["uuid"])
    row = {
        "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
        "phase": f"guard_{stage}", "captured_utc": snapshot["captured_utc"],
        "physical_gpu": PHYSICAL_GPU, "gpu_uuid": selected["uuid"], "gpu_name": selected["name"],
        "memory_free_mib": selected["memory_free_mib"], "memory_used_mib": selected["memory_used_mib"],
        "utilization_percent": selected["utilization_percent"], "temperature_c": selected["temperature_c"],
        "compute_pids": json.dumps(pids), "guard_status": snapshot.get("guard", {}).get("status"),
        "required_free_mib": snapshot.get("guard", {}).get("required_free_mib"),
        "test_evaluation": False,
    }
    path = DATA / "resources.csv"
    previous = read_rows(path)
    fields = tuple(dict.fromkeys([*(previous[0].keys() if previous else ()), *row.keys()]))
    write_rows(path, [*previous, row], fields)


def append_failure(record: dict[str, Any]) -> None:
    path = DATA / "failures.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"failures": []}
    payload.setdefault("failures", []).append({"time_utc": now(), "test_evaluation": False, **record})
    json_write(path, payload)


def capture_gpu_baseline(physical_gpu: int = PHYSICAL_GPU) -> set[int]:
    try:
        snap = smi_snapshot()
        selected = next((item for item in snap["devices"] if item["physical_index"] == physical_gpu), None)
        if selected is None:
            raise RuntimeError(f"physical GPU {physical_gpu} is missing")
    except Exception as exc:
        record = {
            "stage": "gpu_baseline_snapshot", "action": "run_pilot",
            "physical_gpu": physical_gpu, "seed": SEED, "attempt": None,
            "training_started": False, "source_git_sha": source_sha(),
            "error": repr(exc),
        }
        diagnostic = getattr(exc, "stderr", None) or getattr(exc, "output", None)
        if diagnostic:
            record["diagnostic"] = diagnostic.decode(errors="replace") if isinstance(diagnostic, bytes) else str(diagnostic)
        append_failure(record)
        raise
    pids = {
        app["pid"] for app in snap["compute_processes"]
        if app["gpu_uuid"] == selected["uuid"]
    }
    payload = {
        **snap, "authorized_shared_physical_gpu": physical_gpu,
        "authorized_baseline_compute_pids": sorted(pids),
        "authorization_note": f"Physical CUDA {physical_gpu} use authorized for this MR-CGE run; no foreign PID may be added or modified.",
    }
    json_write(DATA / "gpu_baseline.json", payload)
    return pids


def checked_complete_rows() -> set[tuple[str, str, int]]:
    complete = set()
    for row in read_rows(DATA / "runs.csv"):
        if row.get("status") != "complete" or row.get("test_evaluation", "").lower() != "false":
            continue
        checkpoint = Path(row.get("checkpoint_path", ""))
        if not checkpoint.is_file() or sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            continue
        complete.add((row["dataset"], row["condition"], int(row["seed"])))
    return complete


def next_attempt(dataset: str, condition: str, seed: int) -> tuple[int, Path]:
    parent = RUNS / dataset / condition / f"seed_{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    previous = [int(path.name.split("_")[-1]) for path in parent.glob("attempt_*") if path.is_dir()]
    attempt = max(previous, default=0) + 1
    attempt_dir = parent / f"attempt_{attempt:02d}"
    if attempt_dir.exists():
        raise FileExistsError(attempt_dir)
    return attempt, attempt_dir


def metric_bundle(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, labels_for_f1: list[int]) -> dict[str, float]:
    idx = indices.detach().cpu().long()
    target = labels.detach().cpu()[idx].long()
    pred_logits = logits.detach().cpu()[idx]
    pred = pred_logits.argmax(dim=-1)
    return {
        "acc": float((pred == target).float().mean()),
        "macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=labels_for_f1, average="macro", zero_division=0)),
        "ce": float(nn.functional.cross_entropy(pred_logits, target)),
    }


def validation_logits_audit(cfg, data, cache, checkpoint_path: Path, device: torch.device, *, details: bool):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, model_info(data))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval(); classifier.eval()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    with torch.no_grad():
        output, _, _, _, info = model(data.x.to(device), data.edge_index.to(device), return_details=details)
        logits = classifier(output).detach().cpu()
    rng_unchanged = torch.equal(cpu_rng, torch.random.get_rng_state())
    if device.type == "cuda":
        rng_unchanged = rng_unchanged and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    if not rng_unchanged:
        raise RuntimeError("Validation checkpoint diagnostic changed RNG state")
    labels_for_f1 = _resolve_nc_eval_labels(data, include_test=False)
    val = metric_bundle(logits, data.y, data.val_idx, labels_for_f1)
    train = metric_bundle(logits, data.y, data.train_idx, labels_for_f1)
    if bool((data.y[data.test_idx] != -100).any()):
        raise RuntimeError("a Test label was not masked in the validation-only audit")
    return {"payload": payload, "train": train, "val": val, "info": info, "rng_unchanged": rng_unchanged}


def gpu_m_filter_equivalence_audit(cfg, data, cache, model: nn.Module, device: torch.device) -> dict[str, Any] | None:
    """Compare the old and checkpointed M filter/router on identical CUDA inputs."""
    if not str(cfg.model.condition).startswith("M_"):
        return None
    from src.models import mrcge_v11

    old_cfg = copy.deepcopy(cfg)
    old_cfg.model.name = "mrcge_v11"
    old_cfg.model.condition = "M"
    old_model = mrcge_v11.Model(old_cfg, model_info(data))
    old_model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    old_model.load_state_dict(model.state_dict(), strict=True)
    old_model = old_model.to(device).train()
    model.train()

    set_seed(911)
    n = 192
    basis = [torch.randn(n, model.hidden_dim, device=device) for _ in range(4)]
    support = torch.ones(n, dtype=torch.bool, device=device)
    current = torch.randn(n, model.hidden_dim, device=device)
    cross = torch.randn(n, model.hidden_dim, device=device) * 0.01
    input_tensors = [value.detach() for value in basis]

    def run_one(target_model: nn.Module, *, optimized: bool) -> dict[str, Any]:
        target_model.zero_grad(set_to_none=True)
        call_basis = [value.detach().requires_grad_(True) for value in input_tensors]
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        response, _, balance, _ = target_model._filter(
            call_basis, support, current, 0, 0, cross, collect=False
        )
        rng_after_forward = torch.cuda.get_rng_state(device).clone()
        loss = response.square().mean() + 0.01 * balance
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        rng_after_backward = torch.cuda.get_rng_state(device).clone()
        gradients = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in target_model.named_parameters()
            if name.startswith(("filter_bank.", "router.")) and parameter.grad is not None
        }
        return {
            "response": response.detach().float().clone(), "balance": balance.detach().float().clone(),
            "loss": loss.detach().float().clone(), "gradients": gradients,
            "rng_after_forward": rng_after_forward, "rng_after_backward": rng_after_backward,
            "wall_seconds": elapsed,
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

    set_seed(912)
    old_record = run_one(old_model, optimized=False)
    set_seed(912)
    new_record = run_one(model, optimized=True)
    response_close = torch.allclose(old_record["response"], new_record["response"], atol=1.0e-5, rtol=1.0e-4)
    loss_close = torch.allclose(old_record["loss"], new_record["loss"], atol=1.0e-6, rtol=1.0e-5)
    grad_names_match = old_record["gradients"].keys() == new_record["gradients"].keys()
    grad_errors = {
        name: float((old_record["gradients"][name] - new_record["gradients"][name]).abs().max().cpu())
        for name in old_record["gradients"].keys() & new_record["gradients"].keys()
    }
    gradients_close = grad_names_match and all(
        torch.allclose(old_record["gradients"][name], new_record["gradients"][name], atol=2.0e-5, rtol=2.0e-4)
        for name in grad_errors
    )
    rng_forward_match = torch.equal(old_record["rng_after_forward"], new_record["rng_after_forward"])
    rng_backward_restored = (
        torch.equal(old_record["rng_after_forward"], old_record["rng_after_backward"])
        and torch.equal(new_record["rng_after_forward"], new_record["rng_after_backward"])
    )
    passed = bool(response_close and loss_close and gradients_close and rng_forward_match and rng_backward_restored)
    summary = {
        "status": "pass" if passed else "failed", "num_nodes": n,
        "response_allclose_atol1e-5_rtol1e-4": bool(response_close),
        "loss_allclose_atol1e-6_rtol1e-5": bool(loss_close),
        "gradient_names_match": bool(grad_names_match), "gradients_allclose_atol2e-5_rtol2e-4": bool(gradients_close),
        "max_abs_gradient_error": max(grad_errors.values(), default=0.0),
        "rng_after_forward_match": bool(rng_forward_match),
        "checkpoint_backward_rng_restored": bool(rng_backward_restored),
        "old_v11": {key: old_record[key] for key in ("wall_seconds", "baseline_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes")},
        "new_v12": {key: new_record[key] for key in ("wall_seconds", "baseline_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes")},
        "old_peak_allocated_increment_bytes": old_record["peak_allocated_bytes"] - old_record["baseline_allocated_bytes"],
        "new_peak_allocated_increment_bytes": new_record["peak_allocated_bytes"] - new_record["baseline_allocated_bytes"],
        "test_evaluation": False,
    }
    del old_model, old_record, new_record, input_tensors, basis, support, current, cross
    torch.cuda.empty_cache()
    set_seed(int(cfg.seed))
    if not passed:
        raise RuntimeError(f"old/new CUDA M filter equivalence audit failed: {summary}")
    return summary


def append_diagnostics(dataset: str, condition: str, seed: int, attempt: int,
                       details: dict[str, Any], run_row: dict[str, Any]) -> None:
    expert_rows = []
    route_rows = []
    theta = details.get("theta_actual", [])
    filter_map = details.get("filters", {})
    for view, summary in filter_map.items():
        coeffs = summary["coefficients"]
        g_rms = summary["G_expert_rms"]
        e_rms = summary["E_expert_rms"]
        residual_rms = summary["expert_residual_rms"]
        for expert_id in range(len(g_rms)):
            expert_rows.append({
                "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
                "view": view, "expert": expert_id,
                "theta_actual": json.dumps(theta[expert_id] if theta and isinstance(theta[0], list) else theta),
                "effective_coefficients": json.dumps(coeffs[expert_id] if isinstance(coeffs[0], list) else coeffs),
                "G_rms": g_rms[expert_id], "G_pairwise_cosine": json.dumps(summary["G_pairwise_cosine"], sort_keys=True),
                "E_rms": e_rms[expert_id], "E_pairwise_cosine": json.dumps(summary["E_pairwise_cosine"], sort_keys=True),
                "expert_residual_rms": residual_rms[expert_id],
                "support_nodes": details["physical_support_nodes"] if "physical" in view else details["semantic_support_nodes"][view.split("_")[1]],
                "source_git_sha": run_row["source_git_sha"],
            })
    for view, summary in details.get("routes", {}).items():
        if summary is not None:
            route_rows.append({
                "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
                "view": view, **summary, "route_scope": "private", "source_git_sha": run_row["source_git_sha"],
            })
    for modality, summary in details.get("collaborative_routes", {}).items():
        if summary is not None:
            route_rows.append({
                "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
                "view": f"collaborative_{modality}_physical", **summary,
                "route_scope": "collaborative", "route_change": details.get("private_collaborative_route_changes", {}).get(modality),
                "source_git_sha": run_row["source_git_sha"],
            })
    expert_path = DATA / "filter_diagnostics.csv"
    route_path = DATA / "route_diagnostics.csv"
    old_expert = read_rows(expert_path)
    old_route = read_rows(route_path)
    if expert_rows:
        keys = tuple(dict.fromkeys([*(old_expert[0].keys() if old_expert else ()), *(k for row in expert_rows for k in row)]))
        write_rows(expert_path, [*old_expert, *expert_rows], keys)
    if route_rows:
        keys = tuple(dict.fromkeys([*(old_route[0].keys() if old_route else ()), *(k for row in route_rows for k in row)]))
        write_rows(route_path, [*old_route, *route_rows], keys)
    early_rows = []
    for modality, item in details.get("early_physical_cross", {}).items():
        early_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
            "view": f"early_cross_{modality}", **item,
        })
    for modality, item in details.get("late", {}).items():
        early_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
            "view": f"late_{modality}", **item,
        })
    response_path = DATA / "response_diagnostics.csv"
    old_response = read_rows(response_path)
    if early_rows:
        keys = tuple(dict.fromkeys([*(old_response[0].keys() if old_response else ()), *(k for row in early_rows for k in row)]))
        write_rows(response_path, [*old_response, *early_rows], keys)


def append_epoch_history(dataset: str, condition: str, seed: int, attempt: int,
                         source: str, history: list[dict[str, Any]]) -> None:
    path = DATA / "epoch_history.csv"
    old = read_rows(path)
    rows = [
        {"dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
         "source_git_sha": source, **item}
        for item in history
    ]
    if rows:
        fields = tuple(dict.fromkeys([*(old[0].keys() if old else ()), *(key for row in rows for key in row)]))
        write_rows(path, [*old, *rows], fields)


def worker_train(dataset: str, condition: str, seed: int, attempt: int, attempt_dir_text: str,
                 physical_gpu: int) -> None:
    if os.environ.get("MRCGE_GUARDED_CELL") != "1":
        raise RuntimeError("training process must be launched by the shared-GPU-guarded parent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must isolate the authorized physical GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("authorized GPU is unavailable or not isolated in yhf_env")
    baseline_pids = {int(value) for value in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if value}
    expected_peak_mib = int(os.environ.get("MRCGE_EXPECTED_PEAK_RESERVED_MIB", "0"))
    required_start_free = max(8192, expected_peak_mib + 2048)
    guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="before_launch",
              required_free_mib=required_start_free)
    attempt_dir = Path(attempt_dir_text)
    if not attempt_dir.is_dir() or not (attempt_dir / "attempt_started.json").is_file():
        raise RuntimeError("parent did not reserve a unique immutable attempt directory")
    cache = cache_entries()[dataset]
    cfg = build_cfg(dataset, condition, seed, "cuda:0", cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = mask_test_labels(raw)
    if not torch.equal(safe.y[raw.train_idx], raw.y[raw.train_idx]) or not torch.equal(safe.y[raw.val_idx], raw.y[raw.val_idx]):
        raise RuntimeError("Train or Validation labels changed during Test masking")
    if safe.x.size(0) != safe.num_nodes or safe.y.size(0) != safe.num_nodes:
        raise RuntimeError("global feature/label node alignment failed")
    device = torch.device("cuda:0")
    set_seed(seed)
    initial = initial_state_audit(cfg, safe, cache, device)
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    snapshot = OmegaConf.to_yaml(cfg, resolve=True)
    (attempt_dir / "config_snapshot.yaml").write_text(snapshot, encoding="utf-8")
    source = source_sha()
    model_path = ROOT / "src/models/mrcge_v12.py"
    task_path = ROOT / "src/tasks/nc.py"
    fingerprints = legacy.dataset_fingerprints(raw)
    cache_path = Path(cache["cache_path_abs"])
    cache_file_sha = sha256_file(cache_path)
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.2", "dataset": dataset, "condition": condition,
        "seed": seed, "attempt": attempt, "protocol_version": PROTOCOL,
        "source_git_sha": source,
        "config_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
        "model_source_sha256": sha256_file(model_path),
        "task_source_sha256": sha256_file(task_path),
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "semantic_cache_sha256": cache_file_sha,
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
        "initial_common_state_sha256": initial["initial_common_state_sha256"],
        "initial_model_sha256": initial["initial_model_sha256"],
        "initial_classifier_sha256": initial["initial_classifier_sha256"],
        "test_evaluation": False, "test_label_values_indexed": False,
    }
    log_path = attempt_dir / "train.log"
    logger = logging.getLogger(f"mrcge_v12.{dataset}.{condition}.{seed}.{attempt}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)

    # Capture the initial router state and per-view selection before any update.
    set_seed(seed)
    init_model = build_model(cfg, model_info(safe))
    init_model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    init_model = init_model.to(device).eval()
    with torch.no_grad():
        _, _, _, _, initial_info = init_model(safe.x.to(device), safe.edge_index.to(device), return_details=True)
    init_diagnostics_path = attempt_dir / "initial_diagnostics.json"
    json_write(init_diagnostics_path, initial_info.get("details", {}))
    del init_model, initial_info
    torch.cuda.empty_cache()

    started = time.perf_counter()
    try:
        run_nc(cfg, safe, device, logger)
        guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="during_run")
        packed = json.loads(Path(cfg.task.run_metrics_path).read_text(encoding="utf-8"))
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("Trainer metrics protocol or seed provenance mismatch")
        entry = packed["runs"][0]
        metadata = entry["metadata"]
        selection = metadata["dual_checkpoint_selection"]
        acc_selected = selection["best_accuracy"]
        f1_selected = selection["best_macro_f1"]
        training_initial = metadata["initial_parameter_audit"]
        if training_initial["model_state_sha256"] != initial["initial_model_sha256"]:
            raise RuntimeError("Trainer initial model hash differs from actual pre-run state_dict hash")
        if training_initial["classifier_state_sha256"] != initial["initial_classifier_sha256"]:
            raise RuntimeError("Trainer initial classifier hash differs from actual pre-run state_dict hash")
        accuracy_path = Path(cfg.task.save_ckpt_path)
        f1_path = Path(cfg.task.best_f1_ckpt_path)
        if not accuracy_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Best Validation Accuracy or Best Validation Macro-F1 checkpoint missing")
        if not math.isfinite(float(acc_selected["metrics"]["val_acc"])) or not math.isfinite(float(acc_selected["metrics"]["val_macro_f1"])):
            raise RuntimeError("non-finite Validation selection metric")
        device_obj = torch.device("cuda:0")
        acc_audit = validation_logits_audit(cfg, safe, cache, accuracy_path, device_obj, details=True)
        f1_audit = validation_logits_audit(cfg, safe, cache, f1_path, device_obj, details=False)
        for selection_record, audit, name in ((acc_selected, acc_audit, "Best Accuracy"), (f1_selected, f1_audit, "Best F1")):
            for field in ("val_acc", "val_macro_f1", "val_ce"):
                if field in selection_record["metrics"] and abs(float(selection_record["metrics"][field]) - audit["val"][{"val_acc":"acc", "val_macro_f1":"macro_f1", "val_ce":"ce"}[field]]) > 3e-6:
                    raise RuntimeError(f"reloaded {name} checkpoint {field} differs from Trainer metrics")
        acc_sha, f1_sha = sha256_file(accuracy_path), sha256_file(f1_path)
        metrics = acc_selected["metrics"]
        f1_metrics = f1_selected["metrics"]
        history = metadata.get("epoch_history", [])
        if not history or not all(math.isfinite(float(row["train_loss"])) and math.isfinite(float(row["val_acc"])) for row in history):
            raise RuntimeError("finite per-epoch history is missing")
        if not all(audit["rng_unchanged"] for audit in (acc_audit, f1_audit)):
            raise RuntimeError("Validation RNG audit failed")
        if acc_audit["info"].get("details") is None:
            raise RuntimeError("best-checkpoint expert diagnostics are missing")
        run_row = {
            "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
            "status": "complete", "best_acc_epoch": acc_selected["epoch"],
            "best_f1_epoch": f1_selected["epoch"], "val_acc": acc_audit["val"]["acc"],
            "val_macro_f1": acc_audit["val"]["macro_f1"], "val_ce": acc_audit["val"]["ce"],
            "best_f1_val_acc": f1_audit["val"]["acc"], "best_f1_val_macro_f1": f1_audit["val"]["macro_f1"],
            "best_f1_val_ce": f1_audit["val"]["ce"], "train_acc": acc_audit["train"]["acc"],
            "train_ce": acc_audit["train"]["ce"], "model_parameters": metadata["model_parameters"],
            "total_parameters": int(metadata["model_parameters"]) + int(metadata["classifier_parameters"]),
            "wall_seconds": float(metadata.get("training_wall_seconds", time.perf_counter() - started)),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "source_git_sha": source, "config_sha256": cfg.task.extended_checkpoint_metadata["config_sha256"],
            "model_source_sha256": cfg.task.extended_checkpoint_metadata["model_source_sha256"],
            "feature_fingerprint": cache["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "semantic_cache_sha256": cache_file_sha, "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_common_state_sha256": initial["initial_common_state_sha256"],
            "initial_model_sha256": initial["initial_model_sha256"],
            "initial_classifier_sha256": initial["initial_classifier_sha256"],
            "checkpoint_path": str(accuracy_path), "checkpoint_sha256": acc_sha,
            "attempt_dir": str(attempt_dir),
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": f1_sha,
            "test_evaluation": False,
            "validation_rng_unchanged": bool(acc_audit["rng_unchanged"] and f1_audit["rng_unchanged"]),
            "error": "",
        }
        append_csv(DATA / "runs.csv", run_row, RUN_FIELDS, ("dataset", "condition", "seed", "attempt"))
        append_epoch_history(dataset, condition, seed, attempt, source, history)
        append_diagnostics(dataset, condition, seed, attempt, acc_audit["info"]["details"], run_row)
        gradient_rows = []
        gradients = metadata.get("gradient_diagnostics", {}) or {}
        for name, values in gradients.items():
            gradient_rows.append({"dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
                                  "parameter": name, **values, "source_git_sha": source})
        grad_path = DATA / "loss_gradient_diagnostics.csv"
        old_grad = read_rows(grad_path)
        if gradient_rows:
            fields = tuple(dict.fromkeys([*(old_grad[0].keys() if old_grad else ()), *(key for row in gradient_rows for key in row)]))
            write_rows(grad_path, [*old_grad, *gradient_rows], fields)
        resource = {
            "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
            "phase": "screen_train", "wall_seconds": run_row["wall_seconds"],
            "peak_allocated_bytes": run_row["peak_allocated_bytes"],
            "peak_reserved_bytes": run_row["peak_reserved_bytes"],
            "gpu_name": torch.cuda.get_device_name(device), "test_evaluation": False,
        }
        resource_path = DATA / "resources.csv"
        old_resource = read_rows(resource_path)
        fields = tuple(dict.fromkeys([*(old_resource[0].keys() if old_resource else ()), *resource.keys()]))
        write_rows(resource_path, [*old_resource, resource], fields)
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(), **run_row})
        print(f"COMPLETE {dataset}/{condition}/seed{seed}: val_acc={run_row['val_acc']:.6f} val_f1={run_row['val_macro_f1']:.6f}", flush=True)
    except Exception as exc:
        error = {
            "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
            "attempt_dir": str(attempt_dir), "training_started": (attempt_dir / "run_metrics.json").exists(),
            "stage": "training_worker", "error": repr(exc), "traceback": traceback.format_exc(),
            "source_git_sha": source, "test_evaluation": False,
        }
        append_failure(error)
        failed_row = {name: "" for name in RUN_FIELDS}
        failed_row.update({"dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
                           "status": "failed", "source_git_sha": source, "test_evaluation": False,
                           "error": repr(exc)})
        append_csv(DATA / "runs.csv", failed_row, RUN_FIELDS, ("dataset", "condition", "seed", "attempt"))
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(), **error})
        raise
    finally:
        torch.cuda.empty_cache()


def worker_gpu_preflight(dataset: str, condition: str, seed: int, physical_gpu: int,
                         attempt: int | None = None) -> dict[str, Any]:
    if os.environ.get("MRCGE_GUARDED_PREFLIGHT") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("GPU preflight worker must be isolated and launched by the guarded parent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or the physical GPU is not isolated in yhf_env")
    started_utc = now()
    source = source_sha()
    baseline_pids = {int(value) for value in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if value}
    guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="before_launch")
    cache = cache_entries()[dataset]
    cfg = build_cfg(dataset, condition, seed, "cuda:0", cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = mask_test_labels(raw)
    device = torch.device("cuda:0")
    set_seed(seed)
    model = build_model(cfg, model_info(safe))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device)
    m_filter_equivalence = gpu_m_filter_equivalence_audit(cfg, safe, cache, model, device)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1.0e-3, weight_decay=1.0e-4)
    x, edges, labels = safe.x.to(device), safe.edge_index.to(device), safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    if bool((labels[train_idx] < 0).any()):
        raise RuntimeError("single-step preflight found invalid Train labels")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); classifier.train()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits_z, _, _, aux, info = model(x, edges, return_details=False)
    main_ce = nn.functional.cross_entropy(classifier(logits_z[train_idx]), labels[train_idx])
    total_loss, private_ce = _compose_full_graph_loss(
        model=model, info=info, main_ce=main_ce, aux_loss=aux, aux_weight=1.0,
        unimodal_weight=0.3, labels=labels, train_idx=train_idx,
    )
    total_loss.backward()
    gradients = model.capture_gradient_diagnostics()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), 1.0, error_if_nonfinite=True)
    finite = bool(torch.isfinite(logits_z).all() and torch.isfinite(total_loss)) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in list(model.parameters()) + list(classifier.parameters())
    )
    if not finite:
        raise RuntimeError("single-step preflight produced non-finite loss, embeddings, or gradients")
    optimizer.step()
    torch.cuda.synchronize(device)
    train_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    train_peak_reserved = int(torch.cuda.max_memory_reserved(device))

    model.eval(); classifier.eval()
    torch.cuda.reset_peak_memory_stats(device)
    rng_cpu = torch.random.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        _, _, _, _, eval_info = model(x, edges, return_details=True)
    torch.cuda.synchronize(device)
    validation_rng_unchanged = torch.equal(rng_cpu, torch.random.get_rng_state()) and torch.equal(
        rng_cuda, torch.cuda.get_rng_state(device)
    )
    if not validation_rng_unchanged:
        raise RuntimeError("no-grad diagnostics changed Validation RNG state")
    eval_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    eval_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    required_start_free_mib = (max(train_peak_reserved, eval_peak_reserved) + (1024 * 1024 - 1)) // (1024 * 1024) + 2048
    record = {
        "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt, "status": "pass",
        "stage": "gpu_single_step", "physical_gpu": physical_gpu,
        "started_utc": started_utc, "source_git_sha": source,
        "num_nodes": safe.num_nodes, "num_edges": int(safe.edge_index.size(1)),
        "feature_dim": int(safe.input_dim), "test_labels_masked": True, "test_evaluation": False,
        "main_ce": float(main_ce.detach().cpu()), "private_ce": float(private_ce.detach().cpu()),
        "balance_loss_raw": float(info["balance_loss"].detach().cpu()), "aux_loss_scaled": float(aux.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()), "all_gradients_finite": finite,
        "gradient_diagnostics": gradients,
        "m_filter_equivalence": m_filter_equivalence,
        "eval_diagnostics": eval_info.get("details", {}),
        "validation_rng_unchanged": validation_rng_unchanged,
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "eval_peak_allocated_bytes": eval_peak_allocated,
        "eval_peak_reserved_bytes": eval_peak_reserved,
        "peak_allocated_bytes": max(train_peak_allocated, eval_peak_allocated),
        "peak_reserved_bytes": max(train_peak_reserved, eval_peak_reserved),
        "required_start_free_mib_with_2gib_margin": required_start_free_mib,
        "checkpoint_boundary": "full shared expert plus router; use_reentrant=False; preserve_rng_state=True",
        "wall_seconds": time.perf_counter() - start,
        "gpu_name": torch.cuda.get_device_name(device), "guard_snapshot": guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="during_run"),
        "completed_utc": now(),
    }
    return record


def run_guarded_child(action: str, dataset: str, condition: str, seed: int, physical_gpu: int,
                      baseline_pids: set[int], *, attempt: int | None = None,
                      attempt_dir: Path | None = None,
                      expected_peak_reserved_mib: int | None = None) -> None:
    phase = "before_launch"
    try:
        required_start_free = max(8192, (expected_peak_reserved_mib or 0) + 2048) if action == "worker-train" else 8192
        start_snapshot = guard_gpu(physical_gpu, baseline_pids, set(), phase=phase,
                                   required_free_mib=required_start_free)
        append_guard_resource(start_snapshot, dataset=dataset, condition=condition,
                              seed=seed, attempt=attempt, stage=f"{action}_before_launch")
    except Exception as exc:
        record = {
            "stage": "pre_run_gpu_guard", "action": action, "dataset": dataset,
            "condition": condition, "seed": seed, "attempt": attempt,
            "physical_gpu": physical_gpu, "training_started": False, "error": repr(exc),
        }
        append_failure(record)
        raise
    if action == "worker-train" and attempt_dir is not None:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        json_write(attempt_dir / "attempt_started.json", {
            "dataset": dataset, "condition": condition, "seed": seed, "attempt": attempt,
            "started_utc": now(), "source_git_sha": source_sha(), "stage": "formal_screening",
            "expected_peak_reserved_mib": expected_peak_reserved_mib,
            "test_evaluation": False,
        })
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["MRCGE_BASELINE_GPU_PIDS"] = ",".join(str(pid) for pid in sorted(baseline_pids))
    if action == "worker-train":
        env["MRCGE_EXPECTED_PEAK_RESERVED_MIB"] = str(expected_peak_reserved_mib or 0)
        env["MRCGE_GUARDED_CELL"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-train",
                   "--dataset", dataset, "--condition", condition, "--seed", str(seed),
                   "--attempt", str(attempt), "--attempt-dir", str(attempt_dir),
                   "--physical-gpu", str(physical_gpu)]
        log_dir = OUTPUT / "guard_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{dataset}_{condition}_seed{seed}_attempt{attempt:02d}.log"
    else:
        env["MRCGE_GUARDED_PREFLIGHT"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-gpu-preflight",
                   "--dataset", dataset, "--condition", condition, "--seed", str(seed),
                   "--physical-gpu", str(physical_gpu)]
        if attempt is not None:
            command.extend(["--attempt", str(attempt)])
        log_dir = OUTPUT / "preflight_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file = log_dir / f"{dataset}_{condition}_seed{seed}_gpu{physical_gpu}_preflight{attempt or 0}_{stamp}.log"
    with log_file.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(5)
            try:
                runtime_snapshot = guard_gpu(physical_gpu, baseline_pids, {process.pid}, phase="during_run")
                append_guard_resource(runtime_snapshot, dataset=dataset, condition=condition,
                                      seed=seed, attempt=attempt, stage=f"{action}_during_run")
            except Exception as exc:
                # Stop only this study's own child; never signal a pre-existing process.
                try:
                    failure_snapshot = smi_snapshot()
                except Exception:
                    failure_snapshot = None
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                record = {
                    "stage": "shared_gpu_runtime_guard", "action": action, "dataset": dataset,
                    "condition": condition, "seed": seed, "attempt": attempt,
                    "child_pid": process.pid, "physical_gpu": physical_gpu,
                    "source_git_sha": source_sha(), "log_path": str(log_file),
                    "gpu_snapshot": failure_snapshot,
                    "training_started": action == "worker-train",
                    "error": repr(exc), "terminated_own_child_only": True,
                }
                append_failure(record)
                if action == "worker-train" and attempt_dir is not None:
                    existing = any(
                        row.get("dataset") == dataset and row.get("condition") == condition
                        and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)
                        for row in read_rows(DATA / "runs.csv")
                    )
                    if not existing:
                        failed = {name: "" for name in RUN_FIELDS}
                        failed.update({
                            "dataset": dataset, "condition": condition, "seed": seed,
                            "attempt": attempt, "status": "failed", "source_git_sha": source_sha(),
                            "test_evaluation": False,
                            "error": f"shared GPU runtime guard aborted this attempt; see {log_file}",
                        })
                        append_csv(DATA / "runs.csv", failed, RUN_FIELDS,
                                   ("dataset", "condition", "seed", "attempt"))
                    json_write(attempt_dir / "status.json", {
                        "status": "failed", "failed_utc": now(), **record,
                    })
                raise RuntimeError(str(exc)) from exc
        code = process.returncode
    if code != 0:
        text_log = log_file.read_text(encoding="utf-8", errors="replace") if log_file.is_file() else ""
        record = {
            "stage": action, "dataset": dataset, "condition": condition, "seed": seed,
            "attempt": attempt, "child_exit_code": code,
            "training_started": action == "worker-train", "log_path": str(log_file),
            "tail": text_log[-8000:], "error": f"child exited {code}",
        }
        # The worker normally wrote its own detailed failure row. Preserve a
        # parent-level launch record only if no row exists for this attempt.
        append_failure(record)
        if action == "worker-train" and not any(
            row.get("dataset") == dataset and row.get("condition") == condition
            and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)
            for row in read_rows(DATA / "runs.csv")
        ):
            failed = {name: "" for name in RUN_FIELDS}
            failed.update({"dataset": dataset, "condition": condition, "seed": seed,
                           "attempt": attempt, "status": "failed", "source_git_sha": source_sha(),
                           "test_evaluation": False, "error": f"child exited {code}; see {log_file}"})
            append_csv(DATA / "runs.csv", failed, RUN_FIELDS, ("dataset", "condition", "seed", "attempt"))
        if action == "worker-train" and attempt_dir is not None:
            json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(), **record})
        raise RuntimeError(f"guarded {action} failed for {dataset}/{condition}/seed{seed}; see {log_file}")


def _source_file_hashes() -> dict[str, str]:
    paths = {
        "model": ROOT / "src/models/mrcge_v12.py",
        "trainer": ROOT / "src/tasks/nc.py",
        "launcher": Path(__file__).resolve(),
        "model_config": ROOT / "configs/model/mrcge_v12.yaml",
        "study_config": ROOT / "configs/study/mrcge_v12_screen.yaml",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _preflight_pairs(datasets: tuple[str, ...] | None) -> list[tuple[str, str]]:
    if datasets is None:
        return [(dataset, condition) for dataset in CORE_DATASETS for condition in CONDITIONS] + [
            (BONUS_DATASET, condition) for condition in ("M_repaired", "M_anchor")
        ]
    unknown = set(datasets) - set(DATASETS)
    if unknown or not datasets:
        raise ValueError(f"invalid GPU preflight dataset scope: {datasets}")
    result = []
    for dataset in datasets:
        conditions = ("M_repaired", "M_anchor") if dataset == BONUS_DATASET else CONDITIONS
        result.extend((dataset, condition) for condition in conditions)
    return result


def run_gpu_preflight(physical_gpu: int = PHYSICAL_GPU, datasets: tuple[str, ...] | None = None) -> None:
    preflight_path = DATA / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("CPU/data preflight must pass before GPU preflight")
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if payload.get("cpu_status") != "pass":
        raise RuntimeError("CPU/data preflight has not passed")
    pairs = _preflight_pairs(datasets)
    previous = [int(row.get("attempt", 0)) for row in payload.get("gpu_single_step", [])]
    previous += [int(row.get("attempt", 0)) for row in payload.get("gpu_preflight_attempts", [])]
    attempt = max(previous, default=0) + 1
    frozen_files = _source_file_hashes()
    payload.setdefault("gpu_preflight_attempts", []).append({
        "attempt": attempt, "physical_gpu": physical_gpu, "started_utc": now(),
        "checks": [f"{d}/{c}" for d, c in pairs], "source_git_sha": source_sha(),
        "source_file_sha256s": frozen_files, "working_tree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()), "status": "running", "training_started": False, "test_evaluation": False,
    })
    json_write(preflight_path, payload)
    baseline_pids = capture_gpu_baseline(physical_gpu)
    records: list[dict[str, Any]] = []
    ownership_abort = False
    for dataset, condition in pairs:
        if ownership_abort:
            record = {
                "dataset": dataset, "condition": condition, "seed": SEED,
                "status": "blocked_resource", "stage": "gpu_ownership_guard",
                "physical_gpu": physical_gpu, "attempt": attempt,
                "reason": "stopped after a shared-GPU ownership guard event",
                "source_git_sha": source_sha(), "test_evaluation": False,
            }
            payload = json.loads(preflight_path.read_text(encoding="utf-8"))
            payload.setdefault("gpu_single_step", []).append(record)
            json_write(preflight_path, payload)
            records.append(record)
            continue
        try:
            run_guarded_child("worker-gpu-preflight", dataset, condition, SEED,
                              physical_gpu, baseline_pids, attempt=attempt)
        except Exception as exc:
            failures = json.loads((DATA / "failures.json").read_text(encoding="utf-8")).get("failures", [])
            failure = next((row for row in reversed(failures)
                            if row.get("dataset") == dataset and row.get("condition") == condition
                            and row.get("attempt") == attempt), {})
            error_text = " ".join((str(failure.get("error", repr(exc))), str(failure.get("tail", ""))))
            ownership_abort = failure.get("stage") in ("shared_gpu_runtime_guard", "pre_run_gpu_guard")
            blocked = {
                "dataset": dataset, "condition": condition, "seed": SEED,
                "status": "blocked_resource" if ("memory" in error_text.lower() or "guard" in error_text.lower() or "cuda out of memory" in error_text.lower()) else "failed",
                "stage": "gpu_single_step", "physical_gpu": physical_gpu, "attempt": attempt,
                "reason": error_text, "log_path": failure.get("log_path"),
                "gpu_snapshot": failure.get("gpu_snapshot"), "source_git_sha": source_sha(),
                "source_file_sha256s": frozen_files, "training_started": False,
                "test_evaluation": False, "recorded_utc": now(),
            }
            payload = json.loads(preflight_path.read_text(encoding="utf-8"))
            payload.setdefault("gpu_single_step", []).append(blocked)
            json_write(preflight_path, payload)
            records.append(blocked)
            continue
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        row = next((item for item in reversed(payload.get("gpu_single_step", []))
                    if item.get("dataset") == dataset and item.get("condition") == condition
                    and int(item.get("seed", -1)) == SEED and int(item.get("attempt", -1)) == attempt), None)
        if row is None or row.get("status") != "pass":
            raise RuntimeError(f"GPU preflight record missing or failed for {dataset}/{condition}")
        row["source_file_sha256s"] = frozen_files
        json_write(preflight_path, payload)
        records.append(row)

    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    indexed = {(r.get("dataset"), r.get("condition")): r for r in records}
    core_pass = all(indexed.get((d, c), {}).get("status") == "pass" for d in CORE_DATASETS for c in CONDITIONS)
    fashion_pass = all(indexed.get((BONUS_DATASET, c), {}).get("status") == "pass"
                       for c in ("M_repaired", "M_anchor"))
    payload["gpu_single_step_status"] = "pass" if core_pass else "blocked"
    payload["gpu_core_preflight_status"] = "pass" if core_pass else "blocked"
    payload["ele_fashion_m_preflight_status"] = "pass" if fashion_pass else "blocked_resource"
    payload["gpu_preflight_attempts"][-1].update({
        "completed_utc": now(), "status": "pass" if core_pass and fashion_pass else "partial",
        "core_status": payload["gpu_core_preflight_status"],
        "ele_fashion_m_status": payload["ele_fashion_m_preflight_status"],
        "completed_checks": len(records), "training_started": False,
    })
    payload["gpu_single_step_completed_utc"] = now()
    payload["gpu_single_step_attempt"] = attempt
    payload["gpu_single_step_physical_gpu"] = physical_gpu
    json_write(preflight_path, payload)


def _record_blocked_cell(dataset: str, condition: str, reason: str, *, status: str = "blocked_resource") -> None:
    row = {name: "" for name in RUN_FIELDS}
    row.update({"dataset": dataset, "condition": condition, "seed": SEED, "attempt": 0,
                "status": status, "source_git_sha": source_sha(), "test_evaluation": False,
                "error": reason})
    try:
        append_csv(DATA / "runs.csv", row, RUN_FIELDS, ("dataset", "condition", "seed", "attempt"))
    except RuntimeError:
        pass
    append_failure({"stage": "screening_resource_gate", "dataset": dataset,
                    "condition": condition, "seed": SEED, "attempt": 0,
                    "training_started": False, "status": status, "error": reason})


def _passing_preflight_map(preflight: dict[str, Any], physical_gpu: int) -> dict[tuple[str, str], dict[str, Any]]:
    hashes = _source_file_hashes()
    result = {}
    for row in preflight.get("gpu_single_step", []):
        if row.get("physical_gpu") != physical_gpu or row.get("status") != "pass":
            continue
        if row.get("source_file_sha256s") != hashes:
            continue
        result[(row["dataset"], row["condition"])] = row
    return result


def run_pilot(physical_gpu: int = PHYSICAL_GPU, datasets: tuple[str, ...] | None = None) -> None:
    scoped_datasets = tuple(datasets or CORE_DATASETS)
    if not scoped_datasets or set(scoped_datasets) - set(CORE_DATASETS):
        raise ValueError(f"formal screening is restricted to {CORE_DATASETS}")
    preflight_path = DATA / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("CPU and GPU preflight must pass before formal screening")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("cpu_status") != "pass":
        raise RuntimeError("CPU/data preflight did not pass")
    available = _passing_preflight_map(preflight, physical_gpu)
    if not all((d, c) in available for d in scoped_datasets for c in CONDITIONS):
        raise RuntimeError("a requested dataset/condition lacks an equivalent passing GPU preflight on this source")
    baseline_pids = capture_gpu_baseline(physical_gpu)
    complete = checked_complete_rows()
    for dataset in scoped_datasets:
        for condition in CONDITIONS:
            if (dataset, condition, SEED) in complete:
                print(f"SKIP verified {dataset}/{condition}/seed{SEED}", flush=True)
                continue
            attempt, attempt_dir = next_attempt(dataset, condition, SEED)
            expected_mib = math.ceil(int(available[(dataset, condition)]["peak_reserved_bytes"]) / (1024 * 1024))
            try:
                run_guarded_child("worker-train", dataset, condition, SEED, physical_gpu,
                                  baseline_pids, attempt=attempt, attempt_dir=attempt_dir,
                                  expected_peak_reserved_mib=expected_mib)
            except RuntimeError as exc:
                message = str(exc)
                if (not attempt_dir.exists() or any(word in message.lower() for word in
                        ("free memory", "guard", "temperature", "utilization", "out of memory"))):
                    _record_blocked_cell(dataset, condition, message)
                raise


def run_fashion_bonus(physical_gpu: int = PHYSICAL_GPU) -> None:
    preflight = json.loads((DATA / "preflight.json").read_text(encoding="utf-8"))
    available = _passing_preflight_map(preflight, physical_gpu)
    targets = [(BONUS_DATASET, c) for c in ("M_repaired", "M_anchor")]
    if not all(pair in available for pair in targets):
        for dataset, condition in targets:
            _record_blocked_cell(dataset, condition,
                                 "blocked_resource: both Fashion M preflights and the source/GPU guard must pass",
                                 status="blocked_resource")
        return
    baseline_pids = capture_gpu_baseline(physical_gpu)
    for dataset, condition in targets:
        attempt, attempt_dir = next_attempt(dataset, condition, SEED)
        expected_mib = math.ceil(int(available[(dataset, condition)]["peak_reserved_bytes"]) / (1024 * 1024))
        try:
            run_guarded_child("worker-train", dataset, condition, SEED, physical_gpu,
                              baseline_pids, attempt=attempt, attempt_dir=attempt_dir,
                              expected_peak_reserved_mib=expected_mib)
        except RuntimeError as exc:
            resource_failure = any(word in str(exc).lower() for word in
                                   ("free memory", "guard", "temperature", "utilization", "out of memory"))
            if not resource_failure:
                raise
            _record_blocked_cell(dataset, condition, f"blocked_resource: {exc}", status="blocked_resource")
            # A changed ownership snapshot or a failed resource guard stops the pair.
            if condition == "M_repaired":
                _record_blocked_cell(BONUS_DATASET, "M_anchor",
                                     "blocked_resource: M_repaired did not pass the live GPU guard",
                                     status="blocked_resource")
                return
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-train", action="store_true")
    parser.add_argument("--worker-gpu-preflight", action="store_true")
    parser.add_argument("--run-gpu-preflight", action="store_true")
    parser.add_argument("--run-pilot", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--attempt-dir")
    parser.add_argument("--physical-gpu", type=int, default=PHYSICAL_GPU)
    parser.add_argument("--only-dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.worker_train:
        worker_train(args.dataset, args.condition, args.seed, args.attempt, args.attempt_dir, args.physical_gpu)
    elif args.worker_gpu_preflight:
        record = worker_gpu_preflight(args.dataset, args.condition, args.seed,
                                      args.physical_gpu, args.attempt)
        path = DATA / "preflight.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        payload.setdefault("gpu_single_step", []).append(record)
        json_write(path, payload)
        print(f"GPU_PREFLIGHT_PASS {args.dataset}/{args.condition}/seed{args.seed}", flush=True)
    elif args.run_gpu_preflight:
        run_gpu_preflight(args.physical_gpu, (args.only_dataset,) if args.only_dataset else None)
    elif args.run_pilot:
        run_pilot(args.physical_gpu, (args.only_dataset,) if args.only_dataset else None)
    else:
        parser.error("select --run-gpu-preflight, --run-pilot or an internal worker action")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
