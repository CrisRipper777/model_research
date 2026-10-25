#!/usr/bin/env python3
"""Frozen-protocol, shared-GPU-guarded MR-CGE V1.3 Validation screen."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
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

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts import run_mrcge_v12_confirmation as v12_utils
from scripts import run_rmcge_v22 as data_utils
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed


STUDY = ROOT / "research/mrcge_v13"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_screen"
RUNS = OUTPUT / "runs"
V12_SELECTED = ROOT / "research/mrcge_v12_confirmation/data/combined_selected_runs.csv"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("joint_view", "sender_private")
SEEDS = (42, 43)
MODEL_NAME = "mrcge_v13"
BACKBONE_CONDITION = "M_repaired"
SENDER_PRIVATE_RATIO = 0.3
PHYSICAL_GPU = 1
BASE_SHA = "a7930442e85ff9450fded54d5869915b434c7431"
BASE_BRANCH = "exp/mrcge_v12_multiseed_confirmation"
BRANCH = "exp/mrcge_v13_view_cross_screen"
PROTOCOL = "unified_full_graph_nc_v1"

RUN_FIELDS = (
    "dataset", "variant", "seed", "attempt", "status", "best_acc_epoch", "best_f1_epoch",
    "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
    "best_f1_val_ce", "train_acc", "train_ce", "model_parameters", "classifier_parameters",
    "total_parameters", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
    "source_git_sha", "base_sha", "model_source_sha256", "trainer_source_sha256", "config_sha256",
    "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint",
    "split_file_sha256", "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    "initial_model_sha256", "initial_classifier_sha256", "checkpoint_path", "checkpoint_sha256",
    "best_f1_checkpoint_path", "best_f1_checkpoint_sha256", "attempt_dir", "test_evaluation",
    "validation_rng_unchanged", "error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError(f"refusing to overwrite immutable attempt row {key}")
    all_fields = list(dict.fromkeys([*fields, *(name for old in rows for name in old), *row.keys()]))
    write_rows(path, [*rows, row], all_fields)


def append_failure(record: dict[str, Any]) -> None:
    path = DATA / "failures.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"failures": []}
    payload.setdefault("failures", []).append({"time_utc": now(), "test_evaluation": False, **record})
    json_write(path, payload)


def source_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def source_file_hashes() -> dict[str, str]:
    paths = {
        "v13_model": ROOT / "src/models/mrcge_v13.py",
        "v12_model_frozen": ROOT / "src/models/mrcge_v12.py",
        "trainer": ROOT / "src/tasks/nc.py",
        "inference": ROOT / "src/tasks/inference.py",
        "launcher": Path(__file__).resolve(),
        "v13_model_config": ROOT / "configs/model/mrcge_v13.yaml",
        "v12_model_config_frozen": ROOT / "configs/model/mrcge_v12.yaml",
        "study_config": ROOT / "configs/study/mrcge_v13_screen.yaml",
        "v12_confirmation_runner": ROOT / "scripts/run_mrcge_v12_confirmation.py",
        "dataset_loader": ROOT / "src/data/loaders.py",
        "split_loader": ROOT / "src/data/splits.py",
        "graph_utils": ROOT / "src/data/graph_utils.py",
        "semantic_cache_loader": ROOT / "src/data/semantic_candidates_v7b.py",
        "data_types": ROOT / "src/data/types.py",
        "seed_utils": ROOT / "src/utils/seeds.py",
        "nc_config": ROOT / "configs/task/nc.yaml",
        "v12_report": ROOT / "research/mrcge_v12_confirmation/MULTISEED_REPORT.md",
        "v12_handoff": ROOT / "research/mrcge_v12_confirmation/DECISION_HANDOFF.md",
        "v12_combined_runs": V12_SELECTED,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hashes.update({f"dataset_config:{dataset}": sha256_file(ROOT / f"configs/dataset/{dataset}.yaml")
                   for dataset in DATASETS})
    return hashes


def target_cells(datasets: tuple[str, ...] | None = None,
                 variants: tuple[str, ...] | None = None,
                 seeds: tuple[int, ...] | None = None) -> list[tuple[str, str, int]]:
    cells = [(dataset, variant, seed) for dataset in DATASETS for variant in VARIANTS for seed in SEEDS]
    if datasets:
        unknown = set(datasets) - set(DATASETS)
        if unknown:
            raise ValueError(f"unsupported V1.3 dataset selection: {sorted(unknown)}")
        cells = [cell for cell in cells if cell[0] in datasets]
    if variants:
        unknown = set(variants) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unsupported V1.3 variant selection: {sorted(unknown)}")
        cells = [cell for cell in cells if cell[1] in variants]
    if seeds:
        if set(seeds) - set(SEEDS):
            raise ValueError(f"unsupported V1.3 seeds: {sorted(set(seeds) - set(SEEDS))}")
        cells = [cell for cell in cells if cell[2] in seeds]
    if not cells:
        raise ValueError("selection contains no authorized V1.3 screen cells")
    return cells


def cache_entries() -> dict[str, dict[str, Any]]:
    entries = data_utils.verify_cache_entries()
    missing = set(DATASETS) - set(entries)
    if missing:
        raise RuntimeError(f"verified V7B.1 cache entries are missing: {sorted(missing)}")
    return {dataset: entries[dataset] for dataset in DATASETS}


def make_cfg(dataset: str, variant: str, seed: int, device: str, cache: dict[str, Any]):
    if variant not in VARIANTS and variant != "base":
        raise ValueError(f"invalid v13_variant: {variant}")
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create({
        "paths": root.paths,
        "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
        "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
        "model": OmegaConf.load(ROOT / "configs/model/mrcge_v13.yaml"),
        "logging": root.logging,
        "seed": int(seed), "num_runs": 1, "device": str(device),
    })
    cfg.dataset.name = dataset
    cfg.dataset.auto_generate_splits = False
    frozen_cfg = OmegaConf.create({
        "paths": root.paths,
        "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
        "seed": 42,
    })
    OmegaConf.resolve(frozen_cfg)
    if "nc_split_path" in cfg.dataset:
        cfg.dataset.nc_split_path = str(Path(frozen_cfg.dataset.nc_split_path).resolve())
    elif "node_split_path" in cfg.dataset:
        cfg.dataset.node_split_path = str(Path(frozen_cfg.dataset.node_split_path).resolve())
    cfg.model.name = MODEL_NAME
    cfg.model.condition = BACKBONE_CONDITION
    cfg.model.v13_variant = variant
    cfg.model.sender_private_ratio = SENDER_PRIVATE_RATIO if variant == "sender_private" else 0.0
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
    cfg.task.save_ckpt_path = None
    cfg.task.best_f1_ckpt_path = None
    cfg.paths.output_root = str(OUTPUT / "hydra")
    OmegaConf.resolve(cfg)
    return cfg


def data_info(data) -> dict[str, int]:
    return {"input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
            "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def mask_test_labels(data):
    return v12_utils.mask_test_labels(data)


def selected_v12_reference(dataset: str, seed: int) -> dict[str, str]:
    rows = [row for row in read_rows(V12_SELECTED)
            if row.get("dataset") == dataset and row.get("condition") == BACKBONE_CONDITION
            and row.get("seed") == str(seed) and row.get("selected_for_combined", "").lower() == "true"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one selected V1.2 M_repaired row for {dataset}/seed{seed}; got {len(rows)}")
    return rows[0]


def assert_data_alignment(dataset: str, seed: int, data, cache: dict[str, Any], fingerprints: dict[str, Any], split_sha: str) -> dict[str, Any]:
    reference = selected_v12_reference(dataset, seed)
    current = {
        "feature_fingerprint": cache["feature_fingerprint"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "dataset_fingerprint": fingerprints["features"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "v7b_fingerprint": cache["v7b_fingerprint"],
    }
    mismatch = {key: {"v12_seed_reference": reference.get(key), "current": value}
                for key, value in current.items() if reference.get(key) != value}
    if mismatch:
        raise RuntimeError(f"V1.3 data/cache differs from same-seed V1.2 M_repaired: {mismatch}")
    split_path_text = data.info.get("nc_split_path") or data.info.get("node_split_path")
    if not split_path_text:
        raise RuntimeError(f"{dataset}/seed{seed} loader omitted the resolved split path")
    split_path = Path(split_path_text).resolve()
    if not split_path.is_file() or sha256_file(split_path) != split_sha:
        raise RuntimeError(f"{dataset}/seed{seed} split file is missing or changed")
    return {**current, "split_file_path": str(split_path), "split_file_sha256": split_sha,
            "v12_reference_source_campaign": reference.get("source_campaign", ""),
            "v12_reference_source_git_sha": reference.get("source_git_sha", "")}


def tensor_hashes(module: nn.Module) -> dict[str, str]:
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in sorted(module.state_dict().items())}


def module_hash(module: nn.Module) -> str:
    return v12_utils.state_hash(module)


def initial_variant_audit(dataset: str, seed: int, info: dict[str, int]) -> dict[str, Any]:
    cache = {"cache_path_abs": "unused-for-initialization", "feature_fingerprint": "0" * 64,
             "candidate_fingerprint": "0" * 64, "v7b_fingerprint": "0" * 64}
    reference_cfg = v12_utils.build_cfg(dataset, BACKBONE_CONDITION, seed, "cpu", cache)
    set_seed(seed)
    reference_model = build_model(reference_cfg, info)
    reference_head = nn.Linear(reference_model.out_dim, info["num_classes"])
    ref_state = reference_model.state_dict()
    ref_head_hash = module_hash(reference_head)
    variant_hashes = {}
    for variant in ("base", "joint_view", "sender_private"):
        cfg = make_cfg(dataset, variant, seed, "cpu", cache)
        set_seed(seed)
        model = build_model(cfg, info)
        head = nn.Linear(model.out_dim, info["num_classes"])
        state = model.state_dict()
        missing = set(ref_state) - set(state)
        if missing:
            raise RuntimeError(f"{dataset}/seed{seed}/{variant} is missing V1.2 state keys: {sorted(missing)}")
        differing = [name for name, value in ref_state.items() if not torch.equal(value, state[name])]
        if differing:
            raise RuntimeError(f"{dataset}/seed{seed}/{variant} changed frozen initial tensors: {differing[:8]}")
        head_sha = module_hash(head)
        if head_sha != ref_head_hash:
            raise RuntimeError(f"{dataset}/seed{seed}/{variant} changed the external classifier initialization")
        if variant != "joint_view" and set(ref_state) != set(state):
            raise RuntimeError(f"{dataset}/seed{seed}/{variant} base state_dict keys differ from V1.2")
        variant_hashes[variant] = {
            "model_state_sha256": module_hash(model),
            "classifier_state_sha256": head_sha,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "total_parameters": sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
            "state_tensor_sha256": tensor_hashes(model),
            "shared_v12_tensors_exact": True,
        }
        del model, head
    return {"dataset": dataset, "seed": seed, "status": "pass", "v12_model_state_sha256": module_hash(reference_model),
            "v12_classifier_state_sha256": ref_head_hash, "variants": variant_hashes,
            "joint_view_only_additional_state": [k for k in variant_hashes["joint_view"]["state_tensor_sha256"] if k.startswith("joint_view.")],
            "classifier_rng_unchanged_by_a_initialization": True}


def cpu_data_preflight() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "cat-file", "-e", f"{BASE_SHA}^{{commit}}"], cwd=ROOT, check=True)
    caches = cache_entries()
    groups = sorted({(dataset, seed) for dataset, _, seed in target_cells()})
    data_rows, init_rows, reference_rows = [], [], []
    split_fingerprints: dict[str, set[str]] = {}
    split_paths: dict[str, set[str]] = {}
    for dataset, seed in groups:
        cache = caches[dataset]
        cfg = make_cfg(dataset, "base", seed, "cpu", cache)
        raw = load_mag_data(cfg, "nc", seed)
        safe = mask_test_labels(raw)
        fingerprints = data_utils.dataset_fingerprints(raw)
        split_path_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
        if not split_path_text:
            raise RuntimeError(f"{dataset}/seed{seed} did not report a resolved frozen split path")
        split_path = Path(split_path_text).resolve()
        if not split_path.is_file():
            raise RuntimeError(f"frozen split missing: {split_path}")
        split_sha = sha256_file(split_path)
        aligned = assert_data_alignment(dataset, seed, safe, cache, fingerprints, split_sha)
        split_fingerprints.setdefault(dataset, set()).add(fingerprints["split_fingerprint"])
        split_paths.setdefault(dataset, set()).add(str(split_path))
        partition = torch.cat([raw.train_idx, raw.val_idx, raw.test_idx])
        if partition.numel() != raw.num_nodes or torch.unique(partition).numel() != raw.num_nodes:
            raise RuntimeError(f"{dataset}/seed{seed} Train/Validation/Test indices do not partition nodes")
        if bool((safe.y[safe.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
            raise RuntimeError(f"{dataset}/seed{seed} failed Test label masking/Test-off checks")
        train_val_labels = safe.y[torch.cat([safe.train_idx, safe.val_idx])]
        observed = sorted(int(value) for value in torch.unique(train_val_labels[train_val_labels >= 0]).tolist())
        if any(value < 0 or value >= int(safe.num_classes) for value in observed):
            raise RuntimeError(f"{dataset}/seed{seed} Train/Validation labels exceed output classes")
        data_rows.append({
            "dataset": dataset, "seed": seed, "status": "pass", "num_nodes": int(safe.num_nodes),
            "num_edges": int(safe.edge_index.size(1)), "num_classes": int(safe.num_classes),
            "input_dim": int(safe.input_dim), "text_dim": int(safe.x_t.size(1)), "visual_dim": int(safe.x_i.size(1)),
            **aligned, "observed_train_val_label_ids": observed,
            "test_labels_masked": True, "test_evaluation": False,
        })
        init_rows.append(initial_variant_audit(dataset, seed, data_info(safe)))
        reference = selected_v12_reference(dataset, seed)
        reference_rows.append({
            "dataset": dataset, "seed": seed, "condition": BACKBONE_CONDITION,
            "source_campaign": reference.get("source_campaign"),
            "source_git_sha": reference.get("source_git_sha"),
            "val_acc": float(reference["val_acc"]), "val_macro_f1": float(reference["val_macro_f1"]),
            "val_ce": float(reference["val_ce"]),
            "best_f1_val_acc": float(reference["best_f1_val_acc"]),
            "best_f1_val_macro_f1": float(reference["best_f1_val_macro_f1"]),
            "best_f1_val_ce": float(reference["best_f1_val_ce"]),
            "best_acc_epoch": int(reference["best_acc_epoch"]),
            "best_f1_epoch": int(reference["best_f1_epoch"]),
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "best_f1_checkpoint_sha256": reference["best_f1_checkpoint_sha256"],
            "historical_same_seed_not_same_job": True,
        })
        del raw, safe
    if any(len(values) != 1 for values in split_fingerprints.values()) or any(len(values) != 1 for values in split_paths.values()):
        raise RuntimeError(f"seed 42/43 did not reuse one frozen split per dataset: {split_fingerprints}, {split_paths}")

    history = json.loads((DATA / "preflight.json").read_text(encoding="utf-8")) if (DATA / "preflight.json").is_file() else {}
    result = {
        "study": "MR-CGE V1.3 view collaboration and cross screen",
        "cpu_status": "pass", "recorded_utc": now(), "source_git_sha": source_sha(),
        "source_file_sha256s": source_file_hashes(), "base_sha": BASE_SHA,
        "base_branch": BASE_BRANCH, "branch": BRANCH, "datasets": data_rows,
        "initialization_audits": init_rows, "v12_reference_rows": reference_rows,
        "gpu_preflight_cells": history.get("gpu_preflight_cells", []),
        "gpu_preflight_attempts": history.get("gpu_preflight_attempts", []),
        "cpu_preflight_attempts": history.get("cpu_preflight_attempts", []),
        "test_evaluation": False, "link_prediction": False,
    }
    json_write(DATA / "preflight.json", result)
    frozen = {
        "base_sha": BASE_SHA, "base_branch": BASE_BRANCH, "source_branch": BRANCH,
        "training_protocol": PROTOCOL, "backbone_condition": BACKBONE_CONDITION,
        "frozen_source_file_sha256s": result["source_file_sha256s"],
        "dataset_split_cache_records": data_rows, "same_seed_v12_references": reference_rows,
        "historical_reference_is_same_job": False, "test_evaluation": False, "link_prediction": False,
    }
    json_write(DATA / "frozen_baseline.json", frozen)
    manifest_path = DATA / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    json_write(manifest_path, {
        **old, "study": "MR-CGE V1.3 view collaboration and cross screen",
        "branch": BRANCH, "base_sha": BASE_SHA, "base_branch": BASE_BRANCH,
        "datasets": list(DATASETS), "variants": list(VARIANTS), "seeds": list(SEEDS),
        "new_cells": 12, "backbone_condition": BACKBONE_CONDITION,
        "physical_gpu": PHYSICAL_GPU, "cpu_preflight_status": "pass",
        "training_source_git_sha": source_sha(), "test_evaluation": False, "link_prediction": False,
        "completed_cells": old.get("completed_cells", 0), "failed_cells": old.get("failed_cells", 0),
        "blocked_resource_cells": old.get("blocked_resource_cells", 0),
        "unstarted_cells": old.get("unstarted_cells", 12),
    })
    return result


def smi_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True,
    )
    devices = []
    for line in gpu.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 8:
            devices.append({"physical_index": int(p[0]), "uuid": p[1], "name": p[2],
                            "memory_used_mib": int(p[3]), "memory_total_mib": int(p[4]),
                            "memory_free_mib": int(p[5]), "utilization_percent": int(p[6]),
                            "temperature_c": int(p[7])})
    apps_run = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    apps = []
    for line in apps_run.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2:
            try:
                apps.append({"gpu_uuid": p[0], "pid": int(p[1]),
                             "used_memory_mib": int(p[2]) if len(p) > 2 else None})
            except ValueError:
                pass
    return {"captured_utc": now(), "devices": devices, "compute_processes": apps}


def guard_gpu(physical_gpu: int, baseline_pids: set[int], allowed_pids: set[int], *, phase: str,
              required_free_mib: int | None = None) -> dict[str, Any]:
    snapshot = smi_snapshot()
    selected = next((row for row in snapshot["devices"] if row["physical_index"] == physical_gpu), None)
    if selected is None:
        raise RuntimeError(f"physical GPU {physical_gpu} is absent from nvidia-smi")
    current = {row["pid"] for row in snapshot["compute_processes"] if row["gpu_uuid"] == selected["uuid"]}
    unexpected = sorted(current - baseline_pids - allowed_pids)
    if unexpected:
        raise RuntimeError(f"shared-GPU guard saw new foreign compute PIDs {unexpected}; refusing/aborting this run")
    minimum_free = required_free_mib if required_free_mib is not None else (8192 if phase == "before_launch" else 2048)
    if selected["memory_free_mib"] < minimum_free:
        raise RuntimeError(f"GPU{physical_gpu} free memory {selected['memory_free_mib']} MiB below {minimum_free} MiB guard threshold")
    temperature_limit = 88 if phase == "before_launch" else 91
    if selected["temperature_c"] >= temperature_limit:
        raise RuntimeError(f"GPU{physical_gpu} temperature {selected['temperature_c']} C exceeds {phase} limit {temperature_limit} C")
    if phase == "before_launch" and selected["utilization_percent"] >= 90:
        raise RuntimeError(f"GPU{physical_gpu} baseline utilization {selected['utilization_percent']}% is too high to start")
    snapshot["guard"] = {"phase": phase, "physical_gpu": physical_gpu,
                         "baseline_compute_pids": sorted(baseline_pids), "allowed_process_pids": sorted(allowed_pids),
                         "current_compute_pids": sorted(current), "unexpected_compute_pids": unexpected,
                         "required_free_mib": minimum_free, "status": "pass"}
    return snapshot


def append_resource(snapshot: dict[str, Any], dataset: str, variant: str, seed: int,
                    attempt: int | None, phase: str, extra: dict[str, Any] | None = None) -> None:
    selected = next(row for row in snapshot["devices"] if row["physical_index"] == PHYSICAL_GPU)
    pids = [row["pid"] for row in snapshot["compute_processes"] if row["gpu_uuid"] == selected["uuid"]]
    row = {"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
           "phase": phase, "captured_utc": snapshot["captured_utc"], "physical_gpu": PHYSICAL_GPU,
           "gpu_uuid": selected["uuid"], "gpu_name": selected["name"],
           "memory_free_mib": selected["memory_free_mib"], "memory_used_mib": selected["memory_used_mib"],
           "utilization_percent": selected["utilization_percent"], "temperature_c": selected["temperature_c"],
           "compute_pids": json.dumps(sorted(pids)),
           "guard_status": snapshot.get("guard", {}).get("status"),
           "required_free_mib": snapshot.get("guard", {}).get("required_free_mib"),
           "test_evaluation": False}
    row.update(extra or {})
    path = DATA / "resource_summary.csv"
    old = read_rows(path)
    fields = tuple(dict.fromkeys([*(old[0].keys() if old else ()), *row.keys()]))
    write_rows(path, [*old, row], fields)


def capture_gpu_baseline(physical_gpu: int, dataset: str, variant: str, seed: int) -> set[int]:
    snapshot = smi_snapshot()
    selected = next((row for row in snapshot["devices"] if row["physical_index"] == physical_gpu), None)
    if selected is None:
        raise RuntimeError(f"physical GPU {physical_gpu} is missing")
    pids = {row["pid"] for row in snapshot["compute_processes"] if row["gpu_uuid"] == selected["uuid"]}
    snapshot["authorized_baseline_compute_pids"] = sorted(pids)
    snapshot["authorized_shared_physical_gpu"] = physical_gpu
    snapshot["authorization_note"] = "CUDA 1 is authorized for this task; only newly appearing foreign PIDs trigger abort."
    json_write(DATA / "gpu_baseline.json", snapshot)
    append_resource({**snapshot, "guard": {"status": "baseline", "required_free_mib": 8192}},
                    dataset, variant, seed, None, "gpu_baseline")
    return pids


def current_preflight_map(payload: dict[str, Any], physical_gpu: int) -> dict[tuple[str, str, int], dict[str, Any]]:
    hashes = source_file_hashes()
    result = {}
    for row in payload.get("gpu_preflight_cells", []):
        if row.get("physical_gpu") != physical_gpu or row.get("status") != "pass":
            continue
        if row.get("source_file_sha256s") != hashes:
            continue
        result[(row["dataset"], row["variant"], int(row["seed"]))] = row
    return result


def load_safe_data(dataset: str, variant: str, seed: int, cache: dict[str, Any], device: str):
    cfg = make_cfg(dataset, variant, seed, device, cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = mask_test_labels(raw)
    fingerprints = data_utils.dataset_fingerprints(raw)
    split_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
    if not split_text:
        raise RuntimeError(f"{dataset}/seed{seed}: resolved split path absent")
    split_path = Path(split_text).resolve()
    if not split_path.is_file():
        raise RuntimeError(f"frozen split file missing: {split_path}")
    split_sha = sha256_file(split_path)
    alignment = assert_data_alignment(dataset, seed, safe, cache, fingerprints, split_sha)
    if not torch.equal(safe.y[raw.train_idx], raw.y[raw.train_idx]) or not torch.equal(safe.y[raw.val_idx], raw.y[raw.val_idx]):
        raise RuntimeError("Test masking changed Train or Validation labels")
    if bool((safe.y[raw.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
        raise RuntimeError("Test labels must be masked and Test evaluation disabled before trainer entry")
    if safe.x.size(0) != safe.num_nodes or safe.y.size(0) != safe.num_nodes:
        raise RuntimeError("global features/labels do not match frozen node order")
    return cfg, safe, fingerprints, alignment


def worker_gpu_preflight(dataset: str, variant: str, seed: int, physical_gpu: int,
                         attempt: int, record_path_text: str) -> None:
    if os.environ.get("MRCGE_GUARDED_PREFLIGHT") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("GPU preflight worker must be launched by the isolated shared-GPU guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed in yhf_env")
    baseline_pids = {int(v) for v in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if v}
    guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    started = now()
    source = source_sha()
    hashes = source_file_hashes()
    cache = cache_entries()[dataset]
    cfg, safe, fingerprints, aligned = load_safe_data(dataset, variant, seed, cache, "cuda:0")
    device = torch.device("cuda:0")
    set_seed(seed)
    model = build_model(cfg, data_info(safe)).to(device)
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=1e-4)
    x, edges, labels = safe.x.to(device), safe.edge_index.to(device), safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    if bool((labels[train_idx] < 0).any()):
        raise RuntimeError("GPU preflight found invalid Train labels")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); classifier.train()
    tic = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux, info = model(x, edges, return_details=False)
    main_ce = nn.functional.cross_entropy(classifier(z[train_idx]), labels[train_idx])
    loss, private_ce = _compose_full_graph_loss(
        model=model, info=info, main_ce=main_ce, aux_loss=aux, aux_weight=1.0,
        unimodal_weight=0.3, labels=labels, train_idx=train_idx,
    )
    if not torch.isfinite(loss) or not torch.isfinite(z).all():
        raise RuntimeError("GPU preflight produced non-finite embeddings or loss")
    loss.backward()
    gradients = model.capture_gradient_diagnostics()
    all_params = list(model.parameters()) + list(classifier.parameters())
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in all_params):
        raise RuntimeError("GPU preflight produced non-finite gradients")
    if variant == "joint_view":
        grad = model.joint_view.output.weight.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or not bool((grad.norm(dim=1) > 0).all()):
            raise RuntimeError("joint_view preflight requires finite nonzero gradients for both output strengths")
    torch.nn.utils.clip_grad_norm_(all_params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    train_allocated = int(torch.cuda.max_memory_allocated(device))
    train_reserved = int(torch.cuda.max_memory_reserved(device))
    model.eval(); classifier.eval()
    torch.cuda.reset_peak_memory_stats(device)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        _, _, _, _, eval_info = model(x, edges, return_details=True)
    torch.cuda.synchronize(device)
    rng_unchanged = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    if not rng_unchanged:
        raise RuntimeError("Validation-mode diagnostic forward changed RNG state")
    eval_allocated = int(torch.cuda.max_memory_allocated(device))
    eval_reserved = int(torch.cuda.max_memory_reserved(device))
    peak_allocated, peak_reserved = max(train_allocated, eval_allocated), max(train_reserved, eval_reserved)
    required = math.ceil(peak_reserved / (1024 * 1024)) + 2048
    record = {
        "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt, "status": "pass",
        "stage": "gpu_single_step", "physical_gpu": physical_gpu, "started_utc": started,
        "completed_utc": now(), "source_git_sha": source, "source_file_sha256s": hashes,
        "num_nodes": int(safe.num_nodes), "num_edges": int(safe.edge_index.size(1)),
        "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": aligned["split_file_sha256"],
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]), "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"], "test_labels_masked": True, "test_evaluation": False,
        "main_ce": float(main_ce.detach().cpu()), "private_ce": float(private_ce.detach().cpu()),
        "aux_loss": float(aux.detach().cpu()), "total_loss": float(loss.detach().cpu()),
        "all_gradients_finite": True, "gradient_diagnostics": gradients,
        "mechanism_diagnostics_eval": eval_info.get("details", {}).get("v13_mechanism", {}),
        "validation_rng_unchanged": rng_unchanged,
        "train_peak_allocated_bytes": train_allocated, "train_peak_reserved_bytes": train_reserved,
        "eval_peak_allocated_bytes": eval_allocated, "eval_peak_reserved_bytes": eval_reserved,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "required_start_free_mib_with_2gib_margin": required,
        "wall_seconds": time.perf_counter() - tic, "gpu_name": torch.cuda.get_device_name(device),
        "guard_snapshot": guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="during_run"),
        "frozen_data_alignment": aligned,
    }
    json_write(Path(record_path_text), record)
    del model, classifier, optimizer, z, info, eval_info
    torch.cuda.empty_cache()


def next_attempt(dataset: str, variant: str, seed: int) -> tuple[int, Path]:
    rows = [row for row in read_rows(DATA / "runs.csv") if row.get("dataset") == dataset
            and row.get("variant") == variant and row.get("seed") == str(seed)]
    attempt = max([int(row.get("attempt", 0) or 0) for row in rows] + [0]) + 1
    path = RUNS / dataset / variant / f"seed_{seed}" / f"attempt_{attempt:02d}"
    if path.exists():
        raise RuntimeError(f"unique attempt directory already exists: {path}")
    return attempt, path


def record_noncomplete(dataset: str, variant: str, seed: int, attempt: int, status: str,
                       stage: str, reason: str, *, training_started: bool, details: dict[str, Any] | None = None) -> None:
    row = {key: "" for key in RUN_FIELDS}
    row.update({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                "status": status, "source_git_sha": source_sha(), "base_sha": BASE_SHA,
                "test_evaluation": False, "error": f"{stage}: {reason}"})
    append_csv(DATA / "runs.csv", row, RUN_FIELDS, ("dataset", "variant", "seed", "attempt"))
    append_failure({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                    "status": status, "stage": stage, "training_started": training_started,
                    "error": reason, **(details or {})})


def run_guarded_child(action: str, dataset: str, variant: str, seed: int, physical_gpu: int,
                      baseline_pids: set[int], *, attempt: int,
                      attempt_dir: Path | None = None, record_path: Path | None = None,
                      expected_peak_reserved_mib: int | None = None) -> None:
    required = max(8192, (expected_peak_reserved_mib or 0) + 2048) if action == "worker-train" else 8192
    try:
        snap = guard_gpu(physical_gpu, baseline_pids, set(), phase="before_launch", required_free_mib=required)
        append_resource(snap, dataset, variant, seed, attempt, f"{action}_before_launch")
    except Exception as exc:
        try:
            failed_snapshot = smi_snapshot()
            failed_snapshot["guard"] = {"status": "blocked", "phase": "before_launch",
                                         "required_free_mib": required, "reason": repr(exc)}
            append_resource(failed_snapshot, dataset, variant, seed, attempt, f"{action}_blocked_before_launch")
        except Exception:
            failed_snapshot = None
        append_failure({"stage": "pre_run_gpu_guard", "action": action, "dataset": dataset,
                        "variant": variant, "seed": seed, "attempt": attempt,
                        "physical_gpu": physical_gpu, "training_started": False,
                        "gpu_snapshot": failed_snapshot, "error": repr(exc)})
        raise
    if action == "worker-train":
        if attempt_dir is None:
            raise ValueError("formal training requires an attempt directory")
        attempt_dir.mkdir(parents=True, exist_ok=False)
        json_write(attempt_dir / "attempt_started.json", {
            "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
            "started_utc": now(), "source_git_sha": source_sha(), "base_sha": BASE_SHA,
            "test_evaluation": False, "expected_peak_reserved_mib": expected_peak_reserved_mib,
        })
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["MRCGE_BASELINE_GPU_PIDS"] = ",".join(map(str, sorted(baseline_pids)))
    if action == "worker-train":
        env["MRCGE_GUARDED_CELL"] = "1"
        env["MRCGE_EXPECTED_PEAK_RESERVED_MIB"] = str(expected_peak_reserved_mib or 0)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-train", "--dataset", dataset,
                   "--variant", variant, "--seed", str(seed), "--attempt", str(attempt),
                   "--attempt-dir", str(attempt_dir), "--physical-gpu", str(physical_gpu)]
        log_file = OUTPUT / "guard_logs" / f"{dataset}_{variant}_seed{seed}_attempt{attempt:02d}.log"
    else:
        env["MRCGE_GUARDED_PREFLIGHT"] = "1"
        if record_path is None:
            raise ValueError("GPU preflight requires a record path")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-gpu-preflight",
                   "--dataset", dataset, "--variant", variant, "--seed", str(seed),
                   "--attempt", str(attempt), "--record-path", str(record_path),
                   "--physical-gpu", str(physical_gpu)]
        log_file = OUTPUT / "preflight_logs" / f"{dataset}_{variant}_seed{seed}_attempt{attempt:02d}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(5)
            try:
                snap = guard_gpu(physical_gpu, baseline_pids, {process.pid}, phase="during_run")
                append_resource(snap, dataset, variant, seed, attempt, f"{action}_during_run")
            except Exception as exc:
                try:
                    failure_snapshot = smi_snapshot()
                except Exception:
                    failure_snapshot = None
                # Stop only the worker launched by this task.
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                append_failure({"stage": "shared_gpu_runtime_guard", "action": action,
                                "dataset": dataset, "variant": variant, "seed": seed,
                                "attempt": attempt, "child_pid": process.pid,
                                "physical_gpu": physical_gpu, "source_git_sha": source_sha(),
                                "log_path": str(log_file), "gpu_snapshot": failure_snapshot,
                                "training_started": action == "worker-train",
                                "terminated_own_child_only": True, "error": repr(exc)})
                if action == "worker-train" and attempt_dir is not None:
                    json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(),
                                                                "error": repr(exc), "log_path": str(log_file)})
                raise RuntimeError(str(exc)) from exc
        return_code = process.returncode
    if return_code != 0:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-10000:] if log_file.is_file() else ""
        append_failure({"stage": action, "dataset": dataset, "variant": variant,
                        "seed": seed, "attempt": attempt, "child_exit_code": return_code,
                        "training_started": action == "worker-train", "log_path": str(log_file),
                        "tail": tail, "error": f"worker exited with code {return_code}"})
        raise RuntimeError(f"guarded {action} failed for {dataset}/{variant}/seed{seed}; see {log_file}")


def worker_train(dataset: str, variant: str, seed: int, attempt: int,
                 attempt_dir_text: str, physical_gpu: int) -> None:
    if os.environ.get("MRCGE_GUARDED_CELL") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("formal worker must be started by the GPU 1 ownership guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed in yhf_env")
    baseline_pids = {int(v) for v in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if v}
    expected_peak = int(os.environ.get("MRCGE_EXPECTED_PEAK_RESERVED_MIB", "0"))
    guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="before_launch",
              required_free_mib=max(8192, expected_peak + 2048))
    attempt_dir = Path(attempt_dir_text)
    if not (attempt_dir / "attempt_started.json").is_file():
        raise RuntimeError("guarded parent did not reserve this unique attempt directory")
    cache = cache_entries()[dataset]
    cfg, safe, fingerprints, aligned = load_safe_data(dataset, variant, seed, cache, "cuda:0")
    device = torch.device("cuda:0")
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    source = source_sha()
    model_path = ROOT / "src/models/mrcge_v13.py"
    trainer_path = ROOT / "src/tasks/nc.py"
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3 view/cross screen", "dataset": dataset,
        "variant": variant, "backbone_condition": BACKBONE_CONDITION,
        "seed": seed, "attempt": attempt, "source_git_sha": source, "base_sha": BASE_SHA,
        "protocol_version": PROTOCOL, "model_source_sha256": sha256_file(model_path),
        "task_source_sha256": sha256_file(trainer_path),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": aligned["split_file_sha256"],
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
        "sender_private_ratio": cfg.model.sender_private_ratio,
        "test_evaluation": False, "test_label_values_indexed": False,
        "link_prediction": False,
    }
    initial = v12_utils.initial_state_audit(cfg, safe, cache, device)
    initial_cfg = copy.deepcopy(cfg)
    set_seed(seed)
    initial_model = build_model(initial_cfg, data_info(safe))
    initial_model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    initial_model = initial_model.to(device).eval()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        _, _, _, _, initial_info = initial_model(safe.x.to(device), safe.edge_index.to(device), return_details=True)
    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise RuntimeError("initial Validation diagnostics changed RNG state")
    initial_mechanism = initial_info.get("details", {}).get("v13_mechanism", {})
    json_write(attempt_dir / "initial_diagnostics.json", initial_info.get("details", {}))
    del initial_model, initial_info
    torch.cuda.empty_cache()

    log_path = attempt_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"mrcge_v13.{dataset}.{variant}.{seed}.{attempt}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    started = time.perf_counter()
    try:
        run_nc(cfg, safe, device, logger)
        guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="during_run")
        packed = json.loads(Path(cfg.task.run_metrics_path).read_text(encoding="utf-8"))
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("trainer protocol or run seed provenance mismatch")
        entry = packed["runs"][0]
        metadata = entry["metadata"]
        dual = metadata["dual_checkpoint_selection"]
        acc_sel, f1_sel = dual["best_accuracy"], dual["best_macro_f1"]
        if metadata["initial_parameter_audit"]["model_state_sha256"] != initial["initial_model_sha256"]:
            raise RuntimeError("trainer initial model differs from audited initialization")
        if metadata["initial_parameter_audit"]["classifier_state_sha256"] != initial["initial_classifier_sha256"]:
            raise RuntimeError("trainer external classifier differs from audited initialization")
        acc_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        if not acc_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Best-Accuracy or Best-F1 Validation checkpoint is missing")
        acc_audit = v12_utils.validation_logits_audit(cfg, safe, cache, acc_path, device, details=True)
        f1_audit = v12_utils.validation_logits_audit(cfg, safe, cache, f1_path, device, details=True)
        metric_map = {"val_acc": "acc", "val_macro_f1": "macro_f1", "val_ce": "ce"}
        for selected, audit, label in ((acc_sel, acc_audit, "Best Accuracy"), (f1_sel, f1_audit, "Best F1")):
            for field, key in metric_map.items():
                if abs(float(selected["metrics"][field]) - float(audit["val"][key])) > 3e-6:
                    raise RuntimeError(f"reloaded {label} checkpoint {field} differs from trainer metric")
        if not (acc_audit["rng_unchanged"] and f1_audit["rng_unchanged"]):
            raise RuntimeError("checkpoint Validation audit changed RNG state")
        if bool((safe.y[safe.test_idx] != -100).any()):
            raise RuntimeError("Test label was unmasked before trainer audit")
        history = metadata.get("epoch_history", [])
        if not history or not all(all(k in row and math.isfinite(float(row[k]))
                                      for k in ("train_loss", "val_acc", "val_macro_f1", "val_ce"))
                                  for row in history):
            raise RuntimeError("finite per-epoch Validation history is incomplete")
        details = acc_audit["info"].get("details", {})
        if not details.get("v13_mechanism"):
            raise RuntimeError("selected checkpoint mechanism diagnostics are missing")
        run_row = {
            "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt, "status": "complete",
            "best_acc_epoch": acc_sel["epoch"], "best_f1_epoch": f1_sel["epoch"],
            "val_acc": acc_audit["val"]["acc"], "val_macro_f1": acc_audit["val"]["macro_f1"],
            "val_ce": acc_audit["val"]["ce"], "best_f1_val_acc": f1_audit["val"]["acc"],
            "best_f1_val_macro_f1": f1_audit["val"]["macro_f1"], "best_f1_val_ce": f1_audit["val"]["ce"],
            "train_acc": acc_audit["train"]["acc"], "train_ce": acc_audit["train"]["ce"],
            "model_parameters": metadata["model_parameters"],
            "classifier_parameters": metadata["classifier_parameters"],
            "total_parameters": int(metadata["model_parameters"]) + int(metadata["classifier_parameters"]),
            "wall_seconds": float(metadata.get("training_wall_seconds", time.perf_counter() - started)),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "source_git_sha": source, "base_sha": BASE_SHA,
            "model_source_sha256": sha256_file(model_path), "trainer_source_sha256": sha256_file(trainer_path),
            "config_sha256": cfg.task.extended_checkpoint_metadata["config_sha256"],
            "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": aligned["split_file_sha256"],
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_model_sha256": initial["initial_model_sha256"],
            "initial_classifier_sha256": initial["initial_classifier_sha256"],
            "checkpoint_path": str(acc_path), "checkpoint_sha256": sha256_file(acc_path),
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": sha256_file(f1_path),
            "attempt_dir": str(attempt_dir), "test_evaluation": False,
            "validation_rng_unchanged": True, "error": "",
        }
        append_csv(DATA / "runs.csv", run_row, RUN_FIELDS, ("dataset", "variant", "seed", "attempt"))
        histories = [{"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                      "source_git_sha": source, **row} for row in history]
        hist_path = DATA / "epoch_history.csv"
        old_hist = read_rows(hist_path)
        if histories:
            fields = tuple(dict.fromkeys([*(old_hist[0].keys() if old_hist else ()),
                                          *(k for row in histories for k in row)]))
            write_rows(hist_path, [*old_hist, *histories], fields)
        diagnostics = []
        selected_mechanism = details["v13_mechanism"]
        diagnostics.append({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                            "scope": "best_accuracy_validation", "source_git_sha": source, **selected_mechanism})
        if initial_mechanism:
            diagnostics.append({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                                "scope": "initial_validation", "source_git_sha": source, **initial_mechanism})
        view_path = DATA / "view_diagnostics.csv"
        old_view = read_rows(view_path)
        if diagnostics:
            fields = tuple(dict.fromkeys([*(old_view[0].keys() if old_view else ()),
                                          *(k for row in diagnostics for k in row)]))
            write_rows(view_path, [*old_view, *diagnostics], fields)
        gradient_path = DATA / "gradient_diagnostics.csv"
        gradient_rows = []
        for name, values in (metadata.get("gradient_diagnostics", {}) or {}).items():
            gradient_rows.append({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                                  "parameter": name, "source_git_sha": source, **values})
        old_grad = read_rows(gradient_path)
        if gradient_rows:
            fields = tuple(dict.fromkeys([*(old_grad[0].keys() if old_grad else ()),
                                          *(k for row in gradient_rows for k in row)]))
            write_rows(gradient_path, [*old_grad, *gradient_rows], fields)
        append_resource({"devices": [{"physical_index": physical_gpu, "uuid": "runtime",
                                      "name": torch.cuda.get_device_name(device),
                                      "memory_free_mib": "", "memory_used_mib": "",
                                      "utilization_percent": "", "temperature_c": ""}],
                         "compute_processes": [], "captured_utc": now(), "guard": {"status": "training_complete"}},
                        dataset, variant, seed, attempt, "screen_train",
                        {"wall_seconds": run_row["wall_seconds"],
                         "peak_allocated_bytes": run_row["peak_allocated_bytes"],
                         "peak_reserved_bytes": run_row["peak_reserved_bytes"], "gpu_name": torch.cuda.get_device_name(device)})
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(), **run_row})
        print(f"COMPLETE {dataset}/{variant}/seed{seed}: val_acc={run_row['val_acc']:.6f} val_f1={run_row['val_macro_f1']:.6f}", flush=True)
    except Exception as exc:
        append_failure({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                        "stage": "training_worker", "status": "failed", "training_started": True,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                        "source_git_sha": source, "attempt_dir": str(attempt_dir)})
        failed = {name: "" for name in RUN_FIELDS}
        failed.update({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                       "status": "failed", "source_git_sha": source, "base_sha": BASE_SHA,
                       "test_evaluation": False, "error": repr(exc)})
        if not any(row.get("dataset") == dataset and row.get("variant") == variant
                   and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)
                   for row in read_rows(DATA / "runs.csv")):
            append_csv(DATA / "runs.csv", failed, RUN_FIELDS, ("dataset", "variant", "seed", "attempt"))
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(), "error": repr(exc)})
        raise
    finally:
        torch.cuda.empty_cache()


def run_gpu_preflight(physical_gpu: int, datasets=None, variants=None, seeds=None) -> None:
    if physical_gpu != PHYSICAL_GPU:
        raise RuntimeError("this task authorizes physical GPU 1 only")
    path = DATA / "preflight.json"
    if not path.is_file() or json.loads(path.read_text()).get("cpu_status") != "pass":
        raise RuntimeError("run --cpu-preflight first")
    cells = target_cells(datasets, variants, seeds)
    payload = json.loads(path.read_text(encoding="utf-8"))
    attempt = max([int(row.get("attempt", 0)) for row in payload.get("gpu_preflight_attempts", [])] + [0]) + 1
    campaign = {"attempt": attempt, "physical_gpu": physical_gpu, "started_utc": now(),
                "cells": [{"dataset": d, "variant": v, "seed": s} for d, v, s in cells],
                "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
                "status": "running", "passing_cells": 0, "completed_cells": 0,
                "training_started": False, "test_evaluation": False}
    payload.setdefault("gpu_preflight_attempts", []).append(campaign)
    json_write(path, payload)
    try:
        baseline = capture_gpu_baseline(physical_gpu, cells[0][0], cells[0][1], cells[0][2])
    except Exception as exc:
        for dataset, variant, seed in cells:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.setdefault("gpu_preflight_cells", []).append({
                "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                "status": "blocked_resource", "stage": "gpu_baseline_snapshot", "physical_gpu": physical_gpu,
                "reason": repr(exc), "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
                "training_started": False, "test_evaluation": False, "recorded_utc": now(),
            })
        payload["gpu_preflight_attempts"][-1].update({"completed_utc": now(), "completed_cells": len(cells),
                                                       "status": "blocked_resource", "passing_cells": 0})
        json_write(path, payload)
        append_failure({"stage": "gpu_baseline_snapshot", "status": "blocked_resource",
                        "training_started": False, "error": repr(exc)})
        return
    existing = current_preflight_map(payload, physical_gpu)
    stop_after_ownership = False
    passed = 0
    for dataset, variant, seed in cells:
        key = (dataset, variant, seed)
        if key in existing:
            passed += 1
            continue
        if stop_after_ownership:
            row = {"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                   "status": "blocked_resource", "stage": "gpu_ownership_guard", "physical_gpu": physical_gpu,
                   "reason": "not launched after a new external compute PID appeared",
                   "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
                   "training_started": False, "test_evaluation": False, "recorded_utc": now()}
            payload = json.loads(path.read_text(encoding="utf-8")); payload.setdefault("gpu_preflight_cells", []).append(row)
            json_write(path, payload); append_failure({**row, "error": row["reason"]})
            continue
        record_path = OUTPUT / "preflight_attempts" / dataset / variant / f"seed_{seed}" / f"attempt_{attempt:02d}" / "preflight.json"
        try:
            run_guarded_child("worker-gpu-preflight", dataset, variant, seed, physical_gpu,
                              baseline, attempt=attempt, record_path=record_path)
            row = json.loads(record_path.read_text(encoding="utf-8"))
            row["source_file_sha256s"] = source_file_hashes()
            payload = json.loads(path.read_text(encoding="utf-8")); payload.setdefault("gpu_preflight_cells", []).append(row)
            json_write(path, payload); passed += 1
        except Exception as exc:
            failures = json.loads((DATA / "failures.json").read_text(encoding="utf-8")).get("failures", [])
            failure = next((r for r in reversed(failures) if r.get("dataset") == dataset
                            and r.get("variant") == variant and int(r.get("seed", -1)) == seed
                            and int(r.get("attempt", -1) or -1) == attempt), {})
            reason = str(failure.get("error", repr(exc)))
            blocked = any(term in reason.lower() for term in ("guard", "memory", "temperature", "utilization", "foreign compute", "cuda"))
            row = {"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                   "status": "blocked_resource" if blocked else "failed", "stage": "gpu_single_step",
                   "physical_gpu": physical_gpu, "reason": reason, "source_git_sha": source_sha(),
                   "source_file_sha256s": source_file_hashes(), "training_started": False,
                   "test_evaluation": False, "recorded_utc": now(), "log_path": str(OUTPUT / "preflight_logs" / f"{dataset}_{variant}_seed{seed}_attempt{attempt:02d}.log")}
            payload = json.loads(path.read_text(encoding="utf-8")); payload.setdefault("gpu_preflight_cells", []).append(row)
            json_write(path, payload)
            if "new foreign compute pids" in reason.lower():
                stop_after_ownership = True
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gpu_preflight_attempts"][-1].update({"completed_utc": now(), "completed_cells": len(cells),
                                                   "passing_cells": passed,
                                                   "status": "pass" if passed == len(cells) else "partial",
                                                   "training_started": False})
    json_write(path, payload)
    print(json.dumps({"gpu_preflight_status": payload["gpu_preflight_attempts"][-1]["status"],
                      "passing_cells": passed, "requested_cells": len(cells)}))


def checked_complete_keys() -> set[tuple[str, str, int]]:
    complete = set()
    for row in read_rows(DATA / "runs.csv"):
        if row.get("status") != "complete" or row.get("test_evaluation", "").lower() != "false":
            continue
        acc = Path(row.get("checkpoint_path", "")); f1 = Path(row.get("best_f1_checkpoint_path", ""))
        if acc.is_file() and f1.is_file() and sha256_file(acc) == row.get("checkpoint_sha256") \
                and sha256_file(f1) == row.get("best_f1_checkpoint_sha256"):
            complete.add((row["dataset"], row["variant"], int(row["seed"])))
    return complete


def run_screen(physical_gpu: int, datasets=None, variants=None, seeds=None) -> None:
    if physical_gpu != PHYSICAL_GPU:
        raise RuntimeError("this task authorizes physical GPU 1 only")
    preflight_path = DATA / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("run CPU and GPU preflight before formal training")
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if payload.get("cpu_status") != "pass":
        raise RuntimeError("CPU/data preflight did not pass")
    cells = target_cells(datasets, variants, seeds)
    available = current_preflight_map(payload, physical_gpu)
    complete = checked_complete_keys()
    try:
        baseline = capture_gpu_baseline(physical_gpu, cells[0][0], cells[0][1], cells[0][2])
    except Exception as exc:
        for dataset, variant, seed in cells:
            if (dataset, variant, seed) in complete:
                continue
            attempt, _ = next_attempt(dataset, variant, seed)
            record_noncomplete(dataset, variant, seed, attempt, "blocked_resource",
                               "gpu_baseline_snapshot", repr(exc), training_started=False)
        update_manifest(cells)
        return
    stopped_by_ownership = False
    for dataset, variant, seed in cells:
        key = (dataset, variant, seed)
        if key in complete:
            print(f"SKIP verified {dataset}/{variant}/seed{seed}", flush=True)
            continue
        attempt, attempt_dir = next_attempt(dataset, variant, seed)
        preflight = available.get(key)
        if preflight is None:
            record_noncomplete(dataset, variant, seed, attempt, "blocked_resource", "missing_gpu_preflight",
                               "no matching passing GPU preflight for frozen source and GPU 1", training_started=False)
            continue
        if stopped_by_ownership:
            record_noncomplete(dataset, variant, seed, attempt, "blocked_resource", "gpu_ownership_guard",
                               "not launched after a new external compute PID appeared", training_started=False)
            continue
        expected_mib = math.ceil(int(preflight["peak_reserved_bytes"]) / (1024 * 1024))
        try:
            run_guarded_child("worker-train", dataset, variant, seed, physical_gpu,
                              baseline, attempt=attempt, attempt_dir=attempt_dir,
                              expected_peak_reserved_mib=expected_mib)
            complete.add(key)
        except Exception as exc:
            failures = json.loads((DATA / "failures.json").read_text(encoding="utf-8")).get("failures", [])
            failure = next((row for row in reversed(failures) if row.get("dataset") == dataset
                            and row.get("variant") == variant and int(row.get("seed", -1)) == seed
                            and int(row.get("attempt", -1) or -1) == attempt), {})
            reason = str(failure.get("error", repr(exc)))
            existing = [row for row in read_rows(DATA / "runs.csv")
                        if row.get("dataset") == dataset and row.get("variant") == variant
                        and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)]
            if not existing:
                status = "failed" if attempt_dir.exists() else "blocked_resource"
                record_noncomplete(dataset, variant, seed, attempt, status,
                                   "formal_worker" if status == "failed" else "formal_prelaunch_guard",
                                   reason, training_started=attempt_dir.exists(), details={"log_path": failure.get("log_path")})
            if "new foreign compute pids" in reason.lower():
                stopped_by_ownership = True
            print(f"NOT COMPLETE {dataset}/{variant}/seed{seed}: {exc}", flush=True)
    update_manifest(cells)


def update_manifest(cells: list[tuple[str, str, int]]) -> None:
    rows = read_rows(DATA / "runs.csv")
    latest = {}
    target = set(cells)
    for row in rows:
        key = (row.get("dataset"), row.get("variant"), int(row.get("seed", -1)))
        if key in target:
            if key not in latest or int(row.get("attempt", 0) or 0) >= int(latest[key].get("attempt", 0) or 0):
                latest[key] = row
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest.update({
        "completed_cells": sum(row.get("status") == "complete" for row in latest.values()),
        "failed_cells": sum(row.get("status") == "failed" for row in latest.values()),
        "blocked_resource_cells": sum(row.get("status") == "blocked_resource" for row in latest.values()),
        "unstarted_cells": max(0, len(cells) - len(latest)),
        "failed_attempt_rows": sum(row.get("status") == "failed" for row in rows),
        "blocked_resource_attempt_rows": sum(row.get("status") == "blocked_resource" for row in rows),
        "formal_training_started": any(row.get("status") == "complete" for row in rows),
        "last_formal_attempt_utc": now(),
    })
    json_write(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--run-gpu-preflight", action="store_true")
    parser.add_argument("--run-screen", action="store_true")
    parser.add_argument("--worker-gpu-preflight", action="store_true")
    parser.add_argument("--worker-train", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=("base", *VARIANTS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--attempt-dir")
    parser.add_argument("--record-path")
    parser.add_argument("--physical-gpu", type=int, default=PHYSICAL_GPU)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.worker_gpu_preflight:
        worker_gpu_preflight(args.dataset, args.variant, args.seed, args.physical_gpu,
                             args.attempt, args.record_path)
    elif args.worker_train:
        worker_train(args.dataset, args.variant, args.seed, args.attempt, args.attempt_dir, args.physical_gpu)
    elif args.cpu_preflight:
        try:
            result = cpu_data_preflight()
        except Exception as exc:
            path = DATA / "preflight.json"
            previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            attempts = previous.setdefault("cpu_preflight_attempts", [])
            attempt = max([int(row.get("attempt", 0)) for row in attempts] + [0]) + 1
            row = {"attempt": attempt, "stage": "cpu_data_initialization_preflight", "status": "failed",
                   "reason": repr(exc), "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
                   "recorded_utc": now(), "training_started": False, "test_evaluation": False}
            attempts.append(row); json_write(path, previous)
            append_failure({**row, "error": repr(exc)})
            raise
        attempts = result.setdefault("cpu_preflight_attempts", [])
        attempts.append({"attempt": max([int(row.get("attempt", 0)) for row in attempts] + [0]) + 1,
                         "stage": "cpu_data_initialization_preflight", "status": "pass",
                         "recorded_utc": now(), "source_git_sha": source_sha(),
                         "training_started": False, "test_evaluation": False})
        json_write(DATA / "preflight.json", result)
        print(json.dumps({"cpu_status": "pass", "data_groups": len(result["datasets"]),
                          "initialization_audits": len(result["initialization_audits"])}))
    elif args.run_gpu_preflight:
        run_gpu_preflight(args.physical_gpu,
                          (args.dataset,) if args.dataset else None,
                          (args.variant,) if args.variant and args.variant != "base" else None,
                          (args.seed,) if args.seed is not None else None)
    elif args.run_screen:
        run_screen(args.physical_gpu, (args.dataset,) if args.dataset else None,
                   (args.variant,) if args.variant and args.variant != "base" else None,
                   (args.seed,) if args.seed is not None else None)
    else:
        parser.error("select --cpu-preflight, --run-gpu-preflight, --run-screen, or a worker action")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
