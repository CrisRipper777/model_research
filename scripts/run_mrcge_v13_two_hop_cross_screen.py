#!/usr/bin/env python3
"""Guarded Validation screen for MR-CGE V1.3 Two-Hop Cross.

Only the registered Raw and Complementary arms and the fixed Stage 1/2 cells
are accepted. GPU work is always delegated to one isolated child on physical
GPU 1 and watched by the existing V1.3 shared-GPU ownership guard.
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts import run_hybrid_cge_paper_evidence_v1 as hybrid_runner
from scripts import run_mrcge_v12_confirmation as v12_utils
from scripts import run_mrcge_v13_screen as v13_runner
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed


STUDY = ROOT / "research/mrcge_v13_two_hop_cross"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_two_hop_cross"
RUNS = OUTPUT / "runs"
MODEL_NAME = "mrcge_v13_two_hop_cross"
VARIANTS = ("raw", "complementary")
DATASETS = ("Movies", "Grocery", "ele-fashion")
STAGE1 = tuple(("Movies", variant, seed) for variant in VARIANTS for seed in (42, 43, 44))
STAGE2 = tuple((dataset, "__selected__", seed)
               for dataset in ("Grocery", "ele-fashion") for seed in (42, 43))
PHYSICAL_GPU = 1
BASE_SHA = "988419283ef8f49c392de954aa739afc95c4c8cc"
BASE_BRANCH = "exp/mrcge_v13_propagation_mechanism_audit"
BRANCH = "exp/mrcge_v13_two_hop_cross_screen"
PROTOCOL = "unified_full_graph_nc_v1"

RUN_FIELDS = (
    "dataset", "variant", "seed", "attempt", "status", "best_acc_epoch", "best_f1_epoch",
    "epochs_completed", "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc",
    "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce",
    "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap", "model_parameters",
    "classifier_parameters", "total_parameters", "wall_seconds", "peak_allocated_bytes",
    "peak_reserved_bytes", "source_git_sha", "base_sha", "model_source_sha256",
    "trainer_source_sha256", "config_sha256", "feature_fingerprint", "dataset_fingerprint",
    "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
    "candidate_fingerprint", "v7b_fingerprint", "initial_model_sha256", "initial_classifier_sha256",
    "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "attempt_dir", "test_evaluation", "validation_rng_unchanged", "checkpoint_reload_verified", "error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                    default=json_default), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_csv(path: Path, row: dict[str, Any], key_fields: tuple[str, ...]) -> None:
    rows = read_rows(path)
    key = tuple(str(row.get(field, "")) for field in key_fields)
    if any(tuple(old.get(field, "") for field in key_fields) == key for old in rows):
        raise RuntimeError(f"refusing to overwrite immutable run row {key}")
    fields = list(dict.fromkeys([*RUN_FIELDS, *(key for old in rows for key in old), *row.keys()]))
    write_rows(path, [*rows, row], fields)


def source_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


FROZEN_BASE_PATHS = (
    "src/models/mrcge_v13.py", "src/models/mrcge_v12.py", "src/tasks/nc.py",
    "src/tasks/inference.py", "src/data/loaders.py", "src/data/splits.py",
    "src/data/graph_utils.py", "src/data/semantic_candidates_v7b.py", "src/data/types.py",
    "src/utils/seeds.py", "src/models/__init__.py", "configs/model/mrcge_v13.yaml",
    "configs/model/mrcge_v12.yaml", "configs/task/nc.yaml", "configs/config.yaml",
    "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml", "configs/dataset/ele-fashion.yaml",
    "scripts/run_mrcge_v13_screen.py", "scripts/run_mrcge_v12_confirmation.py",
    "scripts/run_hybrid_cge_paper_evidence_v1.py",
)


def source_file_hashes() -> dict[str, str]:
    paths = [*FROZEN_BASE_PATHS, "src/models/mrcge_v13_two_hop_cross.py",
             "configs/model/mrcge_v13_two_hop_cross.yaml",
             "scripts/run_mrcge_v13_two_hop_cross_screen.py",
             "tests/test_mrcge_v13_two_hop_cross.py"]
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def git_file_sha(commit: str, relative: str) -> str:
    raw = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                         check=True, capture_output=True).stdout
    return hashlib.sha256(raw).hexdigest()


def verify_source_freeze() -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("working branch does not descend from the registered base commit")
    for relative in FROZEN_BASE_PATHS:
        if git_file_sha(BASE_SHA, relative) != sha256_file(ROOT / relative):
            raise RuntimeError(f"frozen V1.3-A/trainer/data source differs from base: {relative}")
    for relative, digest in source_file_hashes().items():
        if git_file_sha("HEAD", relative) != digest:
            raise RuntimeError(f"source file is not committed/frozen at HEAD: {relative}")


def fixed_split_path(dataset: str) -> Path:
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    cfg = OmegaConf.create({"paths": root.paths, "dataset": ds, "seed": 42})
    OmegaConf.resolve(cfg)
    for key in ("nc_split_path", "node_split_path"):
        if key in cfg.dataset:
            path = Path(str(cfg.dataset[key])).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
    raise RuntimeError(f"{dataset}: no fixed node split path in config")


def build_cfg(dataset: str, variant: str, seed: int, device: str, cache: dict[str, Any]):
    if variant not in VARIANTS:
        raise ValueError(f"unregistered variant: {variant}")
    cfg = v13_runner.make_cfg(dataset, "joint_view", seed, device, cache)
    cfg.model = OmegaConf.load(ROOT / "configs/model/mrcge_v13_two_hop_cross.yaml")
    cfg.model.name = MODEL_NAME
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
    cfg.model.two_hop_variant = variant
    cfg.model.semantic_cache_path = cache["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    split = fixed_split_path(dataset)
    if "nc_split_path" in cfg.dataset:
        cfg.dataset.nc_split_path = str(split)
    elif "node_split_path" in cfg.dataset:
        cfg.dataset.node_split_path = str(split)
    cfg.dataset.auto_generate_splits = False
    cfg.seed = int(seed)
    cfg.device = device
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
    cfg.task.loss.class_balanced_main_ce = False
    cfg.task.record_extended_validation = True
    cfg.task.record_train_ce_separately = True
    cfg.task.save_ckpt_path = None
    cfg.task.best_f1_ckpt_path = None
    cfg.task.run_metrics_path = None
    cfg.paths.output_root = str(OUTPUT / "hydra")
    OmegaConf.resolve(cfg)
    return cfg


def data_info(data) -> dict[str, int]:
    return {"input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
            "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def load_safe_data(dataset: str, variant: str, seed: int, cache: dict[str, Any], device: str):
    cfg = build_cfg(dataset, variant, seed, device, cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = v12_utils.mask_test_labels(raw)
    split_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
    split_path = Path(split_text).resolve() if split_text else None
    if split_path != fixed_split_path(dataset) or not split_path.is_file():
        raise RuntimeError(f"{dataset}/seed{seed}: loader did not use the fixed project split")
    if cfg.dataset.auto_generate_splits is not False or cfg.task.evaluate_test is not False:
        raise RuntimeError("fixed split and Validation-only evaluation must be frozen")
    if bool((safe.y[raw.test_idx] != -100).any()):
        raise RuntimeError("Test labels were not masked before Trainer entry")
    fp = v13_runner.data_utils.dataset_fingerprints(raw)
    if not torch.isfinite(raw.x).all() or raw.x.size(0) != raw.num_nodes:
        raise RuntimeError("features are non-finite or not in global graph node order")
    return cfg, raw, safe, fp, split_path


def historical_v13(dataset: str, seed: int) -> dict[str, str]:
    rows = [row for row in read_rows(ROOT / "research/mrcge_v13/data/runs.csv")
            if row.get("dataset") == dataset and row.get("variant") == "joint_view"
            and row.get("seed") == str(seed) and row.get("status") == "complete"]
    if not rows and dataset == "Movies" and seed == 44:
        rows = [row for row in read_rows(ROOT / "research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv")
                if row.get("dataset") == dataset and row.get("model") == "mrcge_v13"
                and row.get("condition") == "joint_view" and row.get("seed") == str(seed)
                and row.get("status") == "complete"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one V1.3-A fixed-split reference for {dataset}/seed{seed}, found {len(rows)}")
    return rows[0]


def verify_reference_checkpoints(row: dict[str, str], label: str) -> dict[str, Any]:
    result = {}
    for name, path_key, sha_key in (("best_accuracy", "checkpoint_path", "checkpoint_sha256"),
                                    ("best_f1", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256")):
        path = Path(row.get(path_key, ""))
        expected = row.get(sha_key, "")
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"{label}: historical {name} checkpoint unavailable or hash mismatch: {path}")
        result[name] = {"path": str(path), "sha256": expected}
    return result


def historical_references() -> dict[str, Any]:
    references: dict[str, Any] = {"v13_a": {}, "hybrid_h2_context_only": {}}
    for dataset, variant, seed in (*STAGE1, *[(d, "raw", s) for d in ("Grocery", "ele-fashion") for s in (42, 43)]):
        del variant
        row = historical_v13(dataset, seed)
        references["v13_a"][f"{dataset}_seed{seed}"] = {
            "metrics": {key: float(row[key]) for key in ("val_acc", "val_macro_f1", "val_ce")},
            "checkpoint_hashes": verify_reference_checkpoints(row, f"V1.3-A {dataset}/seed{seed}"),
            "source_git_sha": row.get("source_git_sha", ""),
            "dataset_fingerprint": row.get("dataset_fingerprint", ""),
            "physical_edge_fingerprint": row.get("physical_edge_fingerprint", ""),
            "split_fingerprint": row.get("split_fingerprint", ""),
            "split_file_sha256": row.get("split_file_sha256", ""),
            "semantic_cache_sha256": row.get("semantic_cache_sha256", ""),
            "candidate_fingerprint": row.get("candidate_fingerprint", ""),
            "v7b_fingerprint": row.get("v7b_fingerprint", ""),
        }
    rows = [row for row in read_rows(ROOT / "research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv")
            if row.get("dataset") == "Movies" and row.get("model") == "hybrid_cge_paper_v1"
            and row.get("condition") == "H2_physical_3hop" and row.get("seed") in {"42", "43", "44"}
            and row.get("status") in {"complete", "historical_reference"}]
    if {row.get("seed") for row in rows} != {"42", "43", "44"}:
        raise RuntimeError("fixed-split Hybrid-CGE H2 Movies context rows are incomplete")
    for row in rows:
        references["hybrid_h2_context_only"][f"Movies_seed{row['seed']}"] = {
            "metrics": {key: float(row[key]) for key in ("val_acc", "val_macro_f1", "val_ce")},
            "status": row["status"], "source_git_sha": row.get("source_git_sha", ""),
            "checkpoint_hash": row.get("checkpoint_sha256", ""),
            "note": "historical Validation context only; never loaded or re-evaluated",
        }
    return references


def protected_history_hashes() -> dict[str, str]:
    manifest_path = ROOT / "research/mrcge_v13_propagation_audit/data/manifest.json"
    manifest = read_json(manifest_path, {})
    expected = manifest.get("protected_historical_file_sha256", {})
    if not expected:
        raise RuntimeError("protected historical-file hash map is missing")
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"protected historical file changed or missing: {relative}")
    return expected


def initialize_audit(dataset: str, seed: int, variant: str, safe, cache) -> dict[str, Any]:
    base_cfg = v13_runner.make_cfg(dataset, "joint_view", seed, "cpu", cache)
    base_cfg.model.name = "mrcge_v13"
    cfg = build_cfg(dataset, variant, seed, "cpu", cache)
    set_seed(seed)
    base = hybrid_runner.get_model(base_cfg, safe, cache, torch.device("cpu"))
    base_head = nn.Linear(base.out_dim, int(safe.num_classes))
    set_seed(seed)
    candidate = hybrid_runner.get_model(cfg, safe, cache, torch.device("cpu"))
    head = nn.Linear(candidate.out_dim, int(safe.num_classes))
    base_state, candidate_state = base.state_dict(), candidate.state_dict()
    if set(candidate_state) != set(base_state) | {"two_hop_raw"}:
        raise RuntimeError("candidate initialization contains unexpected state keys")
    if any(not torch.equal(candidate_state[key], value) for key, value in base_state.items()):
        raise RuntimeError("shared V1.3-A model initialization differs")
    if not torch.equal(base_head.weight, head.weight) or not torch.equal(base_head.bias, head.bias):
        raise RuntimeError("external NC classifier initialization differs")
    if not torch.equal(candidate.two_hop_raw, torch.zeros_like(candidate.two_hop_raw)):
        raise RuntimeError("new two_hop_raw parameter is not zero initialized")
    result = {"dataset": dataset, "seed": seed, "variant": variant, "status": "pass",
              "shared_initial_state_equal": True, "external_classifier_equal": True,
              "two_hop_raw_zero": True, "model_sha256": hybrid_runner.module_hash(candidate),
              "classifier_sha256": hybrid_runner.module_hash(head)}
    del base, base_head, candidate, head
    return result


def cpu_preflight() -> dict[str, Any]:
    verify_source_freeze()
    protected = protected_history_hashes()
    caches = v13_runner.cache_entries()
    references = historical_references()
    dataset_records, initialization = {}, []
    cells = [*STAGE1, *[(d, v, s) for d in ("Grocery", "ele-fashion")
                        for v in VARIANTS for s in (42, 43)]]
    for dataset in DATASETS:
        dataset_records[dataset] = None
    for dataset, variant, seed in cells:
        cfg, raw, safe, fingerprints, split_path = load_safe_data(dataset, variant, seed,
                                                                  caches[dataset], "cpu")
        ref = historical_v13(dataset, seed)
        actual = {
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "split_file_sha256": sha256_file(split_path),
            "semantic_cache_sha256": sha256_file(caches[dataset]["cache_path_abs"]),
            "candidate_fingerprint": caches[dataset]["candidate_fingerprint"],
            "v7b_fingerprint": caches[dataset]["v7b_fingerprint"],
        }
        for field, value in actual.items():
            if ref.get(field) != value:
                raise RuntimeError(f"{dataset}/seed{seed}: {field} differs from V1.3-A reference")
        if bool((safe.y[raw.test_idx] != -100).any()):
            raise RuntimeError("Test labels were not masked before preflight/Trainer")
        if dataset_records[dataset] is None:
            dataset_records[dataset] = {
                **actual, "split_path": str(split_path), "semantic_cache_path": caches[dataset]["cache_path_abs"],
                "num_nodes": int(safe.num_nodes), "num_edges": int(raw.edge_index.size(1)),
                "num_classes": int(safe.num_classes), "text_dim": int(safe.x_t.size(1)),
                "visual_dim": int(safe.x_i.size(1)), "test_labels_masked_before_trainer": True,
                "test_evaluation": False, "link_prediction": False,
            }
        initialization.append(initialize_audit(dataset, seed, variant, safe, caches[dataset]))
        del cfg, raw, safe
    # The new parameter must consume no RNG and Raw/Complementary must start identically.
    for dataset, seed in sorted({(d, s) for d, _, s in cells}):
        raw_audit = next(row for row in initialization if row["dataset"] == dataset and row["seed"] == seed and row["variant"] == "raw")
        comp_audit = next(row for row in initialization if row["dataset"] == dataset and row["seed"] == seed and row["variant"] == "complementary")
        if raw_audit["model_sha256"] != comp_audit["model_sha256"] or raw_audit["classifier_sha256"] != comp_audit["classifier_sha256"]:
            raise RuntimeError(f"Raw/Complementary initialization differs for {dataset}/seed{seed}")
    result = {
        "study": "MR-CGE V1.3 Two-Hop Cross Controlled Screening", "status": "pass",
        "stage": "cpu_preflight", "recorded_utc": now(), "base_sha": BASE_SHA,
        "base_branch": BASE_BRANCH, "branch": BRANCH, "source_git_sha": source_sha(),
        "source_file_sha256s": source_file_hashes(), "physical_gpu": PHYSICAL_GPU,
        "authorized_stage1_cells": [{"dataset": d, "variant": v, "seed": s} for d, v, s in STAGE1],
        "conditional_stage2_cells": [{"dataset": d, "variant": v, "seed": s} for d, v, s in cells if d != "Movies"],
        "dataset_records": dataset_records, "initialization_equivalence": initialization,
        "historical_references": references, "protected_historical_file_sha256s": protected,
        "test_evaluation": False, "link_prediction": False,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    for filename, fields in {
        "runs.csv": RUN_FIELDS,
        "epoch_history.csv": ("dataset", "variant", "seed", "attempt", "epoch"),
        "gpu_resources.csv": ("phase", "dataset", "variant", "seed", "attempt", "captured_utc", "physical_gpu", "gpu_uuid", "gpu_name", "memory_free_mib", "required_free_mib", "compute_pids", "guard_status"),
        "paired_differences.csv": ("variant", "seed", "delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce"),
        "dataset_summary.csv": ("dataset", "variant", "n", "val_acc_mean", "val_macro_f1_mean", "val_ce_mean"),
    }.items():
        if not (DATA / filename).exists():
            write_rows(DATA / filename, [], fields)
    json_write(DATA / "failures.json", read_json(DATA / "failures.json", {"failures": []}))
    json_write(DATA / "preflight.json", result)
    json_write(DATA / "manifest.json", {
        "study": result["study"], "branch": BRANCH, "base_sha": BASE_SHA,
        "source_git_sha": source_sha(), "cpu_preflight_status": "pass", "gpu_preflight_status": "pending",
        "training_status": "not_started", "authorized_stage1_cells": result["authorized_stage1_cells"],
        "conditional_stage2_cells": result["conditional_stage2_cells"], "completed_cells": 0,
        "test_evaluation": False, "link_prediction": False, "updated_utc": now(),
    })
    return result


def _device_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in snapshot["devices"] if row["physical_index"] == PHYSICAL_GPU)


def record_resource(snapshot: dict[str, Any], dataset: str, variant: str, seed: int,
                    attempt: int, phase: str) -> None:
    device = _device_snapshot(snapshot)
    row = {"phase": phase, "dataset": dataset, "variant": variant, "seed": seed,
           "attempt": attempt, "captured_utc": snapshot.get("captured_utc"),
           "physical_gpu": PHYSICAL_GPU, "gpu_uuid": device["uuid"], "gpu_name": device["name"],
           "memory_free_mib": device["memory_free_mib"],
           "required_free_mib": snapshot.get("guard", {}).get("required_free_mib", ""),
           "compute_pids": json.dumps(sorted(app["pid"] for app in snapshot["compute_processes"]
                                                 if app["gpu_uuid"] == device["uuid"])),
           "guard_status": snapshot.get("guard", {}).get("status", "snapshot")}
    old = read_rows(DATA / "gpu_resources.csv")
    fields = list(dict.fromkeys([*(old[0].keys() if old else ()), *row.keys()]))
    write_rows(DATA / "gpu_resources.csv", [*old, row], fields)


def load_model(cfg, safe, cache, device):
    model = build_model(cfg, data_info(safe))
    model.load_semantic_cache(cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"])
    return model.to(device)


def worker_preflight(dataset: str, variant: str, seed: int, attempt: int, record_path: Path) -> None:
    if os.environ.get("MR_CGE_GUARDED_PREFLIGHT") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("preflight child must be launched by the GPU 1 ownership guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA device isolation failed")
    baseline = {int(pid) for pid in os.environ.get("MR_CGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    v13_runner.guard_gpu(1, baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    cache = v13_runner.cache_entries()[dataset]
    cfg, raw, safe, fingerprints, split_path = load_safe_data(dataset, variant, seed, cache, "cuda:0")
    device = torch.device("cuda:0")
    set_seed(seed)
    model = load_model(cfg, safe, cache, device)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *classifier.parameters()],
                                  lr=float(cfg.task.lr), weight_decay=float(cfg.task.weight_decay))
    x, edges, labels = safe.x.to(device), safe.edge_index.to(device), safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    if bool((labels[train_idx] < 0).any()):
        raise RuntimeError("masked or invalid label found in Train indices")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); classifier.train(); optimizer.zero_grad(set_to_none=True)
    z, _, _, aux, info = model(x, edges, return_details=False)
    main_ce = F.cross_entropy(classifier(z[train_idx]), labels[train_idx])
    loss, private_ce = _compose_full_graph_loss(model=model, info=info, main_ce=main_ce,
        aux_loss=aux, aux_weight=float(cfg.task.loss.aux_weight),
        unimodal_weight=float(cfg.task.loss.unimodal_weight), labels=labels, train_idx=train_idx)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(z).all()):
        raise RuntimeError("single-step preflight produced non-finite output/loss")
    loss.backward()
    grads = model.capture_gradient_diagnostics()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in [*model.parameters(), *classifier.parameters()]):
        raise RuntimeError("single-step preflight produced a non-finite gradient")
    if not grads["two_hop_raw"]["gradient_finite"] or not grads["two_hop_raw"]["gradient_nonzero"]:
        raise RuntimeError("t must receive a finite, nonzero task gradient")
    torch.nn.utils.clip_grad_norm_([*model.parameters(), *classifier.parameters()], float(cfg.task.grad_clip), error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    model.eval(); classifier.eval()
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        eval_z, _, _, eval_aux, eval_info = model(x, edges, return_details=True)
    rng_ok = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    if not rng_ok or not bool(torch.isfinite(eval_z).all()) or not bool(torch.isfinite(eval_aux)):
        raise RuntimeError("eval diagnostics are non-finite or changed RNG")
    required = math.ceil(peak_reserved / (1024 * 1024)) + 2048
    snapshot = v13_runner.guard_gpu(1, baseline, {os.getpid()}, phase="during_run")
    record = {
        "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
        "status": "pass", "stage": "single_step_forward_backward", "physical_gpu": 1,
        "recorded_utc": now(), "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
        "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": sha256_file(split_path),
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "test_labels_masked": bool((safe.y[safe.test_idx] == -100).all()), "test_evaluation": False,
        "main_ce": float(main_ce.detach().cpu()), "private_ce": float(private_ce.detach().cpu()),
        "aux_loss": float(aux.detach().cpu()), "total_loss": float(loss.detach().cpu()),
        "gradient_diagnostics": grads, "t_gradient": grads["two_hop_raw"],
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "required_start_free_mib_with_2gib_margin": required,
        "gpu_name": torch.cuda.get_device_name(device), "guard_snapshot": snapshot,
        "validation_rng_unchanged": rng_ok, "test_evaluation_actual": False,
    }
    json_write(record_path, record)
    del model, classifier, optimizer, raw, safe, z, eval_z, info, eval_info
    torch.cuda.empty_cache()


def guarded_child(action: str, dataset: str, variant: str, seed: int, attempt: int,
                  baseline: set[int], *, record_path: Path | None = None,
                  attempt_dir: Path | None = None, required_free_mib: int = 8192) -> dict[str, Any]:
    launch = v13_runner.guard_gpu(1, baseline, set(), phase="before_launch",
                                  required_free_mib=required_free_mib)
    record_resource(launch, dataset, variant, seed, attempt, f"{action}_before_launch")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["MR_CGE_BASELINE_GPU_PIDS"] = ",".join(map(str, sorted(baseline)))
    if action == "preflight":
        env["MR_CGE_GUARDED_PREFLIGHT"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-preflight",
                   "--dataset", dataset, "--variant", variant, "--seed", str(seed),
                   "--attempt", str(attempt), "--record-path", str(record_path)]
    else:
        env["MR_CGE_GUARDED_TRAIN"] = "1"
        env["MR_CGE_EXPECTED_PEAK_RESERVED_MIB"] = str(max(0, required_free_mib - 2048))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-train",
                   "--dataset", dataset, "--variant", variant, "--seed", str(seed),
                   "--attempt", str(attempt), "--attempt-dir", str(attempt_dir)]
    logdir = OUTPUT / ("preflight_logs" if action == "preflight" else "guard_logs")
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"{action}_{dataset}_{variant}_seed{seed}_attempt{attempt:02d}.log"
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(4)
            try:
                snap = v13_runner.guard_gpu(1, baseline, {proc.pid}, phase="during_run")
                record_resource(snap, dataset, variant, seed, attempt, f"{action}_during_run")
            except Exception as exc:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=10)
                raise RuntimeError(f"runtime GPU guard stopped our {action} child: {exc}") from exc
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"{action} child exit {proc.returncode}: {log_path}\n{tail}")
    return {"launch_guard": launch, "log_path": str(log_path)}


def current_baseline() -> set[int]:
    snap = v13_runner.smi_snapshot()
    device = next((item for item in snap["devices"] if item["physical_index"] == 1), None)
    if device is None:
        raise RuntimeError("physical GPU 1 missing")
    return {int(app["pid"]) for app in snap["compute_processes"] if app["gpu_uuid"] == device["uuid"]}


def gpu_preflight() -> dict[str, Any]:
    verify_source_freeze()
    pre = read_json(DATA / "preflight.json", {})
    if pre.get("status") != "pass":
        raise RuntimeError("CPU preflight must pass before GPU preflight")
    cells = [(d, v, 42) for d in DATASETS for v in VARIANTS]
    attempts = read_json(DATA / "gpu_preflight.json", {"attempts": []})
    attempt = max([int(row.get("attempt", 0)) for row in attempts["attempts"]] + [0]) + 1
    campaign = {"attempt": attempt, "started_utc": now(), "cells": cells,
                "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
                "status": "running", "training_started": False, "physical_gpu": 1}
    attempts["attempts"].append(campaign)
    json_write(DATA / "gpu_preflight.json", attempts)
    passed, stopped = 0, False
    for dataset, variant, seed in cells:
        record_path = OUTPUT / "preflight_attempts" / dataset / variant / f"seed_{seed}" / f"attempt_{attempt:02d}.json"
        if stopped:
            row = {"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                   "status": "blocked_resource", "reason": "not launched after ownership/resource guard stop",
                   "source_git_sha": source_sha(), "training_started": False, "test_evaluation": False}
            attempts.setdefault("cells", []).append(row); json_write(DATA / "gpu_preflight.json", attempts)
            continue
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            baseline = current_baseline()
            guarded_child("preflight", dataset, variant, seed, attempt, baseline, record_path=record_path)
            row = read_json(record_path, {})
            if row.get("status") != "pass" or row.get("source_file_sha256s") != source_file_hashes():
                raise RuntimeError("GPU preflight record failed status/source audit")
            attempts.setdefault("cells", []).append(row); passed += 1
        except Exception as exc:
            text = repr(exc)
            row = {"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                   "status": "blocked_resource" if any(k in text.lower() for k in
                   ("guard", "memory", "foreign compute", "temperature", "utilization", "cuda")) else "failed",
                   "reason": text, "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
                   "training_started": False, "test_evaluation": False}
            attempts.setdefault("cells", []).append(row)
            json_write(DATA / "failures.json", {"failures": [*read_json(DATA / "failures.json", {}).get("failures", []), row]})
            if row["status"] == "blocked_resource":
                stopped = True
        json_write(DATA / "gpu_preflight.json", attempts)
    campaign.update({"completed_utc": now(), "passing_cells": passed,
                     "status": "pass" if passed == len(cells) else "incomplete",
                     "training_started": False})
    attempts["attempts"][-1] = campaign
    json_write(DATA / "gpu_preflight.json", attempts)
    manifest = read_json(DATA / "manifest.json", {})
    manifest.update({"gpu_preflight_status": campaign["status"], "gpu_preflight_passed": passed,
                     "updated_utc": now()})
    json_write(DATA / "manifest.json", manifest)
    return campaign


def next_attempt(dataset: str, variant: str, seed: int) -> tuple[int, Path]:
    rows = [row for row in read_rows(DATA / "runs.csv") if row.get("dataset") == dataset
            and row.get("variant") == variant and row.get("seed") == str(seed)]
    attempt = max([int(row.get("attempt", 0) or 0) for row in rows] + [0]) + 1
    path = RUNS / dataset / variant / f"seed_{seed}" / f"attempt_{attempt:02d}"
    if path.exists():
        raise RuntimeError(f"refusing to reuse attempt directory {path}")
    return attempt, path


def worker_train(dataset: str, variant: str, seed: int, attempt: int, attempt_dir: Path) -> None:
    if os.environ.get("MR_CGE_GUARDED_TRAIN") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("training child must be launched by the GPU 1 ownership guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA device isolation failed")
    baseline = {int(pid) for pid in os.environ.get("MR_CGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    required = max(8192, int(os.environ.get("MR_CGE_EXPECTED_PEAK_RESERVED_MIB", "0")) + 2048)
    v13_runner.guard_gpu(1, baseline, {os.getpid()}, phase="before_launch", required_free_mib=required)
    if not (attempt_dir / "attempt_started.json").is_file():
        raise RuntimeError("parent has not reserved this formal attempt")
    current = [row for row in read_json(DATA / "gpu_preflight.json", {}).get("cells", [])
               if row.get("dataset") == dataset and row.get("variant") == variant and row.get("status") == "pass"
               and row.get("source_file_sha256s") == source_file_hashes()]
    if not current:
        raise RuntimeError("no passing exact-source preflight for this variant/dataset")
    pre = current[-1]
    expected = max(8192, math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024)) + 2048)
    if required < expected:
        raise RuntimeError(f"formal launch memory requirement {required} is below preflight estimate {expected}")
    cache = v13_runner.cache_entries()[dataset]
    cfg, raw, safe, fingerprints, split_path = load_safe_data(dataset, variant, seed, cache, "cuda:0")
    device = torch.device("cuda:0")
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    source = source_sha()
    model_path, trainer_path = ROOT / "src/models/mrcge_v13_two_hop_cross.py", ROOT / "src/tasks/nc.py"
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3 Two-Hop Cross Controlled Screening", "dataset": dataset,
        "variant": variant, "condition": "M_repaired", "v13_variant": "joint_view",
        "seed": seed, "attempt": attempt, "source_git_sha": source, "base_sha": BASE_SHA,
        "protocol_version": PROTOCOL, "model_source_sha256": sha256_file(model_path),
        "trainer_source_sha256": sha256_file(trainer_path), "config_sha256": config_hash,
        "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": sha256_file(split_path),
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "test_evaluation": False, "test_label_values_indexed": False, "link_prediction": False,
    }
    initial_model_sha, initial_classifier_sha, _ = hybrid_runner.init_pair_hashes(cfg, safe, cache, seed)
    log_path = attempt_dir / "train.log"
    logger = logging.getLogger(f"mrcge_two_hop.{dataset}.{variant}.{seed}.{attempt}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        run_nc(cfg, safe, device, logger)
        v13_runner.guard_gpu(1, baseline, {os.getpid()}, phase="during_run")
        packed = read_json(Path(cfg.task.run_metrics_path), {})
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("trainer protocol/seed provenance mismatch")
        metadata = packed["runs"][0]["metadata"]
        selection = metadata["dual_checkpoint_selection"]
        init_audit = metadata["initial_parameter_audit"]
        if init_audit["model_state_sha256"] != initial_model_sha or init_audit["classifier_state_sha256"] != initial_classifier_sha:
            raise RuntimeError("trainer model or classifier initialization differs from the audited pair")
        acc_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        if not acc_path.is_file() or not f1_path.is_file():
            raise RuntimeError("best Validation Accuracy/F1 checkpoint missing")
        acc = v12_utils.validation_logits_audit(cfg, safe, cache, acc_path, device, details=True)
        f1 = v12_utils.validation_logits_audit(cfg, safe, cache, f1_path, device, details=True)
        if not (acc["rng_unchanged"] and f1["rng_unchanged"]):
            raise RuntimeError("checkpoint reload audit changed RNG state")
        for selected, evaluated, label in ((selection["best_accuracy"], acc, "accuracy"),
                                           (selection["best_macro_f1"], f1, "F1")):
            for field, metric in (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce")):
                if abs(float(selected["metrics"][field]) - float(evaluated["val"][metric])) > 3e-6:
                    raise RuntimeError(f"reloaded best-{label} checkpoint {field} differs from trainer metrics")
        if bool((safe.y[safe.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
            raise RuntimeError("Test label/evaluation boundary failed")
        history = metadata.get("epoch_history", [])
        required_history = ("train_loss", "val_acc", "val_macro_f1", "val_ce")
        if not history or not all(all(k in row and math.isfinite(float(row[k])) for k in required_history) for row in history):
            raise RuntimeError("finite epoch history is absent or incomplete")
        details = acc["info"].get("details", {})
        mechanism = details.get("two_hop_cross", {})
        if set(mechanism) != {"text", "visual"}:
            raise RuntimeError("Best-Accuracy checkpoint is missing two-hop mechanism diagnostics")
        train_metrics, val_metrics = acc["train"], acc["val"]
        f1_val = f1["val"]
        record = {
            "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt, "status": "complete",
            "best_acc_epoch": selection["best_accuracy"]["epoch"], "best_f1_epoch": selection["best_macro_f1"]["epoch"],
            "epochs_completed": len(history), "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"], "val_ce": val_metrics["ce"],
            "best_f1_val_acc": f1_val["acc"], "best_f1_val_macro_f1": f1_val["macro_f1"],
            "best_f1_val_ce": f1_val["ce"], "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"], "train_ce": train_metrics["ce"],
            "train_val_acc_gap": train_metrics["acc"] - val_metrics["acc"],
            "train_val_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"],
            "train_val_ce_gap": train_metrics["ce"] - val_metrics["ce"],
            "model_parameters": metadata["model_parameters"], "classifier_parameters": metadata["classifier_parameters"],
            "total_parameters": int(metadata["model_parameters"]) + int(metadata["classifier_parameters"]),
            "wall_seconds": metadata.get("training_wall_seconds", time.perf_counter() - started),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"), "source_git_sha": source,
            "base_sha": BASE_SHA, "model_source_sha256": sha256_file(model_path),
            "trainer_source_sha256": sha256_file(trainer_path), "config_sha256": config_hash,
            "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": sha256_file(split_path),
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_model_sha256": initial_model_sha, "initial_classifier_sha256": initial_classifier_sha,
            "checkpoint_path": str(acc_path), "checkpoint_sha256": sha256_file(acc_path),
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": sha256_file(f1_path),
            "attempt_dir": str(attempt_dir), "test_evaluation": False,
            "validation_rng_unchanged": True, "checkpoint_reload_verified": True, "error": "",
        }
        append_csv(DATA / "runs.csv", record, ("dataset", "variant", "seed", "attempt"))
        hist_rows = [{"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt, **row} for row in history]
        old = read_rows(DATA / "epoch_history.csv")
        fields = list(dict.fromkeys([*(old[0].keys() if old else ()), *(key for row in hist_rows for key in row)]))
        write_rows(DATA / "epoch_history.csv", [*old, *hist_rows], fields)
        json_write(attempt_dir / "mechanism_diagnostics.json", {
            "best_accuracy": mechanism, "epoch_trajectory": [
                {key: value for key, value in row.items() if "two_hop_" in key}
                for row in history], "test_evaluation": False,
        })
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(), **record})
    except Exception as exc:
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(),
                    "error": repr(exc), "traceback": traceback.format_exc(), "test_evaluation": False})
        raise
    finally:
        torch.cuda.empty_cache()


def stage1_decision() -> dict[str, Any]:
    rows = read_rows(DATA / "runs.csv")
    references = read_json(DATA / "preflight.json", {})["historical_references"]["v13_a"]
    all_by = {(r.get("variant"), int(r.get("seed", -1))): r for r in rows
              if r.get("dataset") == "Movies" and r.get("variant") in VARIANTS and r.get("seed") in {"42", "43", "44"}}
    paired, arm_results = [], {}
    for variant in VARIANTS:
        diffs = []
        failures = []
        for seed in (42, 43, 44):
            row = all_by.get((variant, seed))
            if row is None or row.get("status") != "complete":
                failures.append({"seed": seed, "status": row.get("status") if row else "missing"})
                continue
            failed_attempts = [item for item in rows if item.get("dataset") == "Movies"
                               and item.get("variant") == variant and item.get("seed") == str(seed)
                               and item.get("status") == "failed"]
            if failed_attempts:
                failures.append({"seed": seed, "status": "failed_attempt_recorded",
                                 "attempts": [item.get("attempt") for item in failed_attempts]})
            ref = references[f"Movies_seed{seed}"]["metrics"]
            delta = {"variant": variant, "seed": seed,
                     "delta_val_acc_pp": 100.0 * (float(row["val_acc"]) - ref["val_acc"]),
                     "delta_val_macro_f1_pp": 100.0 * (float(row["val_macro_f1"]) - ref["val_macro_f1"]),
                     "delta_val_ce": float(row["val_ce"]) - ref["val_ce"]}
            if not all(math.isfinite(delta[key]) for key in ("delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce")):
                failures.append({"seed": seed, "status": "non_finite_metrics"})
            diffs.append(delta); paired.append(delta)
        acc = [row["delta_val_acc_pp"] for row in diffs]
        f1 = [row["delta_val_macro_f1_pp"] for row in diffs]
        ce = [row["delta_val_ce"] for row in diffs]
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        sd = lambda xs: math.sqrt(sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) if xs else None
        finite = len(diffs) == 3 and not failures
        gate = bool(finite and mean(acc) >= 0.30 and sum(value > 0 for value in acc) >= 2
                    and mean(f1) >= 0.0 and mean(ce) <= 0.020)
        arm_results[variant] = {
            "paired": diffs, "n": len(diffs), "failures": failures,
            "delta_val_acc_pp_mean": mean(acc), "delta_val_acc_pp_population_sd": sd(acc),
            "delta_val_macro_f1_pp_mean": mean(f1), "delta_val_macro_f1_pp_population_sd": sd(f1),
            "delta_val_ce_mean": mean(ce), "delta_val_ce_population_sd": sd(ce),
            "positive_accuracy_seeds": sum(value > 0 for value in acc), "promotion_gate_pass": gate,
        }
    if arm_results["raw"]["promotion_gate_pass"] and arm_results["complementary"]["promotion_gate_pass"]:
        a, b = arm_results["raw"], arm_results["complementary"]
        if abs(a["delta_val_acc_pp_mean"] - b["delta_val_acc_pp_mean"]) > 0.10:
            selected = "raw" if a["delta_val_acc_pp_mean"] > b["delta_val_acc_pp_mean"] else "complementary"
            tie = "accuracy"
        elif a["delta_val_macro_f1_pp_mean"] != b["delta_val_macro_f1_pp_mean"]:
            selected = "raw" if a["delta_val_macro_f1_pp_mean"] > b["delta_val_macro_f1_pp_mean"] else "complementary"
            tie = "macro_f1"
        elif a["delta_val_ce_mean"] != b["delta_val_ce_mean"]:
            selected = "raw" if a["delta_val_ce_mean"] < b["delta_val_ce_mean"] else "complementary"
            tie = "cross_entropy"
        else:
            selected, tie = "raw", "raw_final_tie_break"
    elif arm_results["raw"]["promotion_gate_pass"]:
        selected, tie = "raw", "only_passing_arm"
    elif arm_results["complementary"]["promotion_gate_pass"]:
        selected, tie = "complementary", "only_passing_arm"
    else:
        selected, tie = None, "no_arm_passed"
    old = [row for row in read_rows(DATA / "paired_differences.csv")
           if not any(row.get("variant") == diff["variant"] and row.get("seed") == str(diff["seed"])
                      for diff in paired)]
    write_rows(DATA / "paired_differences.csv", [*old, *paired],
               list(dict.fromkeys([*(old[0].keys() if old else ()), "variant", "seed", "delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce"])))
    decision = {"recorded_utc": now(), "stage1_complete": all(len(arm_results[v]["paired"]) == 3 for v in VARIANTS),
                "promotion_gates": arm_results, "selected_variant": selected, "selection_tie_break": tie,
                "stage2_authorized": selected is not None,
                "rule": "mean Delta Acc >= +0.30 pp; >=2/3 positive; mean Delta F1 >=0 pp; mean Delta CE <=+0.020; no failures"}
    json_write(DATA / "stage1_screening_decision.json", decision)
    return decision


def write_report(decision: dict[str, Any] | None) -> None:
    rows = read_rows(DATA / "runs.csv")
    pre = read_json(DATA / "preflight.json", {})
    lines = ["# MR-CGE V1.3 Two-Hop Cross Controlled Screening", "",
             f"Recorded: {now()}", f"Source commit: `{source_sha()}`", f"Base: `{BASE_SHA}`", "",
             "## Protocol", "",
             "Validation-only full-graph NC with fixed seed-42 splits, V7B.1 cache, Test labels masked before Trainer, and Test/LP disabled. GPU work is restricted to physical GPU 1 under the existing ownership guard.", "",
             "The H2 results below are historical Validation context only and were not loaded or re-evaluated.", "",
             "| Movies seed | H2 Accuracy | H2 Macro-F1 | H2 CE |", "|---:|---:|---:|---:|"]
    for seed in (42, 43, 44):
        h2 = pre["historical_references"]["hybrid_h2_context_only"][f"Movies_seed{seed}"]["metrics"]
        lines.append(f"| {seed} | {h2['val_acc']:.6f} | {h2['val_macro_f1']:.6f} | {h2['val_ce']:.6f} |")
    lines += ["", "## Run results", "", "| Dataset | Arm | Seed | Status | Val Acc | Val Macro-F1 | Val CE | Train–Val Acc gap | Best epoch |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("status") not in {"complete", "failed", "blocked_resource"}:
            continue
        lines.append(f"| {row.get('dataset')} | {row.get('variant')} | {row.get('seed')} | {row.get('status')} | {float(row['val_acc']):.6f} | {float(row['val_macro_f1']):.6f} | {float(row['val_ce']):.6f} | {float(row['train_val_acc_gap']):.6f} | {row.get('best_acc_epoch')} |" if row.get("status") == "complete" else f"| {row.get('dataset')} | {row.get('variant')} | {row.get('seed')} | {row.get('status')} | — | — | — | — | — |")
    lines += ["", "## Best-Accuracy checkpoint mechanism diagnostics", "",
              "| Dataset | Arm | Seed | lambda (text, visual) | Calibrated candidate RMS (text, visual) | Scale (text, visual) | Expert addition RMS (text) | Expert addition RMS (visual) |", "|---|---|---:|---|---|---|---|---|"]
    mechanism_rows = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        path = Path(row["attempt_dir"]) / "mechanism_diagnostics.json"
        payload = read_json(path, {})
        diag = payload.get("best_accuracy", {})
        items = [diag.get(modality, {}) for modality in ("text", "visual")]
        lambdas = [item.get("lambda") for item in items]
        rms = [item.get("calibrated_rms") for item in items]
        scales = [item.get("scale") for item in items]
        injections = [item.get("expert_injection", {}).get("expert_addition_rms", []) for item in items]
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['seed']} | {lambdas} | {rms} | {scales} | {injections[0]} | {injections[1]} |")
        for modality, item in zip(("text", "visual"), items):
            injection = item.get("expert_injection", {})
            mechanism_rows.append({"dataset": row["dataset"], "variant": row["variant"], "seed": row["seed"],
                "modality": modality, "lambda": item.get("lambda"), "x2_rms": item.get("x2_rms"),
                "same_node_candidate_rms": item.get("same_node_candidate_rms"),
                "projection_coefficient": item.get("projection_coefficient"),
                "raw_x2_same_node_cosine": item.get("raw_x2_same_node_cosine"),
                "candidate_rms": item.get("candidate_rms"), "calibrated_rms": item.get("calibrated_rms"),
                "scale": item.get("scale"), "scale_capped": item.get("scale_capped"),
                "near_zero_degenerate": item.get("near_zero_degenerate"),
                "expert_addition_rms": json.dumps(injection.get("expert_addition_rms", [])),
                "expert_addition_over_old_input_rms": json.dumps(injection.get("expert_addition_over_old_input_rms", [])),
                "test_evaluation": False})
    if mechanism_rows:
        write_rows(DATA / "two_hop_mechanism.csv", mechanism_rows,
                   list(dict.fromkeys(key for item in mechanism_rows for key in item)))
    if decision:
        lines += ["", "## Stage 1 paired Validation screen", "",
                  "| Arm | Mean ΔAcc (pp) | SD | Positive seeds | Mean ΔF1 (pp) | Mean ΔCE | Gate |", "|---|---:|---:|---:|---:|---:|---|"]
        for variant in VARIANTS:
            item = decision["promotion_gates"][variant]
            lines.append(f"| {variant} | {item['delta_val_acc_pp_mean'] if item['delta_val_acc_pp_mean'] is not None else float('nan'):.4f} | {item['delta_val_acc_pp_population_sd'] if item['delta_val_acc_pp_population_sd'] is not None else float('nan'):.4f} | {item['positive_accuracy_seeds']}/3 | {item['delta_val_macro_f1_pp_mean'] if item['delta_val_macro_f1_pp_mean'] is not None else float('nan'):.4f} | {item['delta_val_ce_mean'] if item['delta_val_ce_mean'] is not None else float('nan'):.5f} | {'PASS' if item['promotion_gate_pass'] else 'FAIL'} |")
        lines += ["", f"Selected arm: **{decision['selected_variant'] or 'none'}** ({decision['selection_tie_break']})."]
    lines += ["", "## Interpretation", "", "This is an exploratory Validation screen, not a significance test or evidence of Test generalization. Stage 2 runs only the single Stage 1 selected arm on Grocery and ele-fashion seeds 42/43. No selection means no Stage 2 runs.", ""]
    (STUDY / "SCREENING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def record_failed(dataset: str, variant: str, seed: int, attempt: int, status: str, reason: str, attempt_dir: Path):
    row = {field: "" for field in RUN_FIELDS}
    row.update({"dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
                "status": status, "source_git_sha": source_sha(), "base_sha": BASE_SHA,
                "attempt_dir": str(attempt_dir), "error": reason, "test_evaluation": False})
    append_csv(DATA / "runs.csv", row, ("dataset", "variant", "seed", "attempt"))
    fails = read_json(DATA / "failures.json", {"failures": []})
    fails["failures"].append({"dataset": dataset, "variant": variant, "seed": seed,
        "attempt": attempt, "status": status, "stage": "formal_train", "error": reason,
        "test_evaluation": False, "recorded_utc": now()})
    json_write(DATA / "failures.json", fails)


def run_cells(cells: tuple[tuple[str, str, int], ...], *, stage: str) -> bool:
    stop = False
    for dataset, variant, seed in cells:
        if stop:
            break
        done = any(r.get("dataset") == dataset and r.get("variant") == variant
                   and r.get("seed") == str(seed) and r.get("status") == "complete" for r in read_rows(DATA / "runs.csv"))
        if done:
            continue
        pre_rows = [row for row in read_json(DATA / "gpu_preflight.json", {}).get("cells", [])
                    if row.get("dataset") == dataset and row.get("variant") == variant
                    and row.get("status") == "pass" and row.get("source_file_sha256s") == source_file_hashes()]
        if not pre_rows:
            raise RuntimeError(f"missing current-source GPU preflight for {dataset}/{variant}")
        pre = pre_rows[-1]
        required = max(8192, math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024)) + 2048)
        attempt, attempt_dir = next_attempt(dataset, variant, seed)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        json_write(attempt_dir / "attempt_started.json", {
            "dataset": dataset, "variant": variant, "seed": seed, "attempt": attempt,
            "stage": stage, "started_utc": now(), "source_git_sha": source_sha(),
            "source_file_sha256s": source_file_hashes(), "preflight_record": str(pre.get("record_path", "")),
            "required_free_mib": required, "test_evaluation": False,
        })
        try:
            baseline = current_baseline()
            guarded_child("train", dataset, variant, seed, attempt, baseline,
                          attempt_dir=attempt_dir, required_free_mib=required)
            if not any(r.get("dataset") == dataset and r.get("variant") == variant
                       and r.get("seed") == str(seed) and r.get("attempt") == str(attempt)
                       and r.get("status") == "complete" for r in read_rows(DATA / "runs.csv")):
                raise RuntimeError("training child exited without a verified complete run record")
            print(f"COMPLETE {stage} {dataset}/{variant}/seed{seed}", flush=True)
        except Exception as exc:
            reason = repr(exc)
            status = "blocked_resource" if any(token in reason.lower() for token in
                ("guard", "memory", "foreign compute", "temperature", "utilization", "cuda")) else "failed"
            if not any(r.get("dataset") == dataset and r.get("variant") == variant
                       and r.get("seed") == str(seed) and r.get("attempt") == str(attempt) for r in read_rows(DATA / "runs.csv")):
                record_failed(dataset, variant, seed, attempt, status, reason, attempt_dir)
            json_write(attempt_dir / "status.json", {"status": status, "error": reason, "recorded_utc": now()})
            print(f"{status.upper()} {stage} {dataset}/{variant}/seed{seed}: {reason}", flush=True)
            if status == "blocked_resource":
                stop = True
    return not stop


def run_screen() -> dict[str, Any]:
    verify_source_freeze()
    pre = read_json(DATA / "preflight.json", {})
    gpu = read_json(DATA / "gpu_preflight.json", {})
    expected = {(d, v) for d in DATASETS for v in VARIANTS}
    passed = {(row.get("dataset"), row.get("variant")) for row in gpu.get("cells", [])
              if row.get("status") == "pass" and row.get("source_file_sha256s") == source_file_hashes()}
    if pre.get("status") != "pass" or not expected <= passed:
        raise RuntimeError(f"all six current-source variant-by-dataset GPU preflights must pass; missing {sorted(expected-passed)}")
    stage1_ok = run_cells(STAGE1, stage="stage1_movies")
    rows = read_rows(DATA / "runs.csv")
    stage1_complete = all(any(r.get("dataset") == "Movies" and r.get("variant") == variant
                               and r.get("seed") == str(seed) and r.get("status") == "complete" for r in rows)
                          for variant in VARIANTS for seed in (42, 43, 44))
    decision = stage1_decision() if stage1_complete else None
    write_report(decision)
    protected_history_hashes()
    if stage1_ok and decision and decision["selected_variant"]:
        cells = tuple((d, decision["selected_variant"], s) for d in ("Grocery", "ele-fashion") for s in (42, 43))
        run_cells(cells, stage="stage2_cross_dataset")
    elif decision:
        manifest = read_json(DATA / "manifest.json", {})
        manifest["stage2_status"] = "not_run_no_arm_passed"
        json_write(DATA / "manifest.json", manifest)
    manifest = read_json(DATA / "manifest.json", {})
    rows = read_rows(DATA / "runs.csv")
    manifest.update({"stage1_complete": stage1_complete,
        "stage1_completed_cells": sum(r.get("dataset") == "Movies" and r.get("status") == "complete" for r in rows),
        "stage2_selected_variant": decision["selected_variant"] if decision else None,
        "stage2_completed_cells": sum(r.get("dataset") != "Movies" and r.get("status") == "complete" for r in rows),
        "training_status": "complete" if decision and (not decision["selected_variant"] or
                           sum(r.get("dataset") != "Movies" and r.get("status") == "complete" for r in rows) == 4)
                           else "partial", "test_evaluation": False, "updated_utc": now()})
    json_write(DATA / "manifest.json", manifest)
    summary_rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            selected = [row for row in rows if row.get("dataset") == dataset and row.get("variant") == variant and row.get("status") == "complete"]
            if not selected:
                continue
            summary_rows.append({"dataset": dataset, "variant": variant, "n": len(selected),
                "seeds": ";".join(sorted(row["seed"] for row in selected)),
                "val_acc_mean": sum(float(row["val_acc"]) for row in selected) / len(selected),
                "val_macro_f1_mean": sum(float(row["val_macro_f1"]) for row in selected) / len(selected),
                "val_ce_mean": sum(float(row["val_ce"]) for row in selected) / len(selected)})
    write_rows(DATA / "dataset_summary.csv", summary_rows,
               ("dataset", "variant", "n", "seeds", "val_acc_mean", "val_macro_f1_mean", "val_ce_mean"))
    write_report(decision)
    protected_history_hashes()
    return manifest


def run_all() -> None:
    cpu_preflight()
    gpu_preflight()
    run_screen()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-preflight", action="store_true")
    group.add_argument("--gpu-preflight", action="store_true")
    group.add_argument("--run-screen", action="store_true")
    group.add_argument("--run-all", action="store_true")
    group.add_argument("--worker-preflight", action="store_true")
    group.add_argument("--worker-train", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--record-path")
    parser.add_argument("--attempt-dir")
    args = parser.parse_args()
    if args.worker_preflight or args.worker_train:
        if not all((args.dataset, args.variant, args.seed is not None, args.attempt is not None)):
            raise SystemExit("worker needs dataset, variant, seed and attempt")
        if args.worker_preflight:
            if not args.record_path:
                raise SystemExit("preflight worker needs --record-path")
            worker_preflight(args.dataset, args.variant, args.seed, args.attempt, Path(args.record_path))
        else:
            if not args.attempt_dir:
                raise SystemExit("training worker needs --attempt-dir")
            worker_train(args.dataset, args.variant, args.seed, args.attempt, Path(args.attempt_dir))
        return
    if args.cpu_preflight:
        result = cpu_preflight()
    elif args.gpu_preflight:
        result = gpu_preflight()
    elif args.run_screen:
        result = run_screen()
    else:
        run_all(); result = read_json(DATA / "manifest.json", {})
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
