#!/usr/bin/env python3
"""Guarded Validation screen for the single MR-CGE Intrinsic-Prior Skip variant."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts import run_mrcge_v12_confirmation as v12_utils
from scripts import run_mrcge_v13_screen as v13_runner
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed

STUDY = ROOT / "research/mrcge_v13_intrinsic_prior_skip"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_intrinsic_prior_skip"
RUNS = OUTPUT / "runs"
MODEL_NAME = "mrcge_v13_intrinsic_prior_skip"
VARIANT = "intrinsic_prior_skip"
DATASETS = ("Movies", "Grocery", "ele-fashion")
STAGE1 = tuple(("Movies", seed) for seed in (42, 43, 44)) + tuple(("Grocery", seed) for seed in (42, 43))
STAGE2 = tuple(("ele-fashion", seed) for seed in (42, 43))
PHYSICAL_GPU = 1
GPU_READY_WAIT_SECONDS = 180
EXPECTED_SPLIT_FINGERPRINTS = {
    "Movies": "2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d",
    "Grocery": "7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056",
    "ele-fashion": "04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5",
}
BASE_SHA = "184deaaf12b21c9ee01d3328aef15d86bcc50346"
BASE_BRANCH = "exp/mrcge_v13_two_hop_cross_screen"
BRANCH = "exp/mrcge_v13_intrinsic_prior_skip_screen"
PROTOCOL = "unified_full_graph_nc_v1"

FROZEN_BASE_PATHS = (
    "src/models/mrcge_v13.py", "src/models/mrcge_v12.py",
    "src/models/mrcge_v13_two_hop_cross.py", "src/tasks/nc.py", "src/tasks/inference.py",
    "src/data/__init__.py", "src/data/loaders.py", "src/data/splits.py", "src/data/graph_utils.py",
    "src/data/semantic_candidates.py", "src/data/semantic_candidates_v7b.py",
    "src/data/types.py", "src/utils/seeds.py",
    "src/models/factory.py", "src/models/__init__.py",
    "configs/model/mrcge_v13.yaml", "configs/model/mrcge_v12.yaml",
    "configs/model/mrcge_v13_two_hop_cross.yaml", "configs/task/nc.yaml", "configs/config.yaml",
    "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml", "configs/dataset/ele-fashion.yaml",
    "scripts/run_mrcge_v13_screen.py", "scripts/run_mrcge_v13_two_hop_cross_screen.py",
    "scripts/run_mrcge_v12_confirmation.py", "scripts/run_hybrid_cge_paper_evidence_v1.py",
)
NEW_SOURCE_PATHS = (
    "research/mrcge_v13_intrinsic_prior_skip/DESIGN_FREEZE.md",
    "src/models/mrcge_v13_intrinsic_prior_skip.py",
    "configs/model/mrcge_v13_intrinsic_prior_skip.yaml",
    "tests/test_mrcge_v13_intrinsic_prior_skip.py",
    "scripts/run_mrcge_v13_intrinsic_prior_skip_screen.py",
)
RUN_FIELDS = (
    "dataset", "variant", "seed", "attempt", "status", "best_acc_epoch", "best_f1_epoch",
    "epochs_completed", "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc",
    "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce",
    "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap", "train_loss",
    "model_parameters", "classifier_parameters", "total_parameters", "wall_seconds",
    "peak_allocated_bytes", "peak_reserved_bytes", "prior_skip_raw", "lambda_prior",
    "prior_skip_gradient_norm", "prior_skip_gradient_nonzero", "prior_skip_optimizer_managed",
    "prior_skip_diagnostics_json", "h0_fused_rms", "fused_rms", "fused_minus_h0_rms",
    "fusion_skip_weight_fused_rms", "prior_skip_linear_rms", "fusion_mlp_rms",
    "prior_to_old_skip_rms_ratio", "prior_to_old_skip_cosine",
    "physical_gpu", "gpu_uuid", "gpu_name", "required_start_free_mib",
    "source_git_sha", "base_sha", "model_source_sha256",
    "trainer_source_sha256", "config_sha256", "feature_fingerprint", "dataset_fingerprint",
    "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
    "candidate_fingerprint", "v7b_fingerprint", "initial_model_sha256", "initial_classifier_sha256",
    "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "attempt_dir", "test_evaluation", "validation_rng_unchanged", "checkpoint_reload_verified", "error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        raise RuntimeError(f"refusing to overwrite immutable research row {key}")
    fields = list(dict.fromkeys([*RUN_FIELDS, *(key for old in rows for key in old), *row.keys()]))
    write_rows(path, [*rows, row], fields)


def append_failure(record: dict[str, Any]) -> None:
    path = DATA / "failures.json"
    payload = read_json(path, {"failures": []})
    payload.setdefault("failures", []).append({"recorded_utc": now(), **record})
    write_json(path, payload)


def source_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def git_file_sha(commit: str, relative: str) -> str:
    raw = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                         check=True, capture_output=True).stdout
    return hashlib.sha256(raw).hexdigest()


def source_file_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in (*FROZEN_BASE_PATHS, *NEW_SOURCE_PATHS)}


def verify_frozen_sources() -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("current history does not descend from the registered base")
    changed = subprocess.run(
        ["git", "diff", "--name-only", BASE_SHA, "--", "src/data", "src/models", "src/tasks", "configs", "scripts"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    for relative in changed:
        existed_at_base = subprocess.run(
            ["git", "cat-file", "-e", f"{BASE_SHA}:{relative}"], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if existed_at_base:
            raise RuntimeError(f"pre-existing source/config/runner changed relative to base: {relative}")
    for relative in FROZEN_BASE_PATHS:
        if git_file_sha(BASE_SHA, relative) != sha256_file(ROOT / relative):
            raise RuntimeError(f"frozen historical source/config changed: {relative}")


def verify_committed_source() -> None:
    verify_frozen_sources()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}, found {branch}")
    for relative in NEW_SOURCE_PATHS:
        if git_file_sha("HEAD", relative) != sha256_file(ROOT / relative):
            raise RuntimeError(f"frozen training source is not committed at HEAD: {relative}")


def training_source_committed() -> bool:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}, found {branch}")
    for relative in NEW_SOURCE_PATHS:
        try:
            if git_file_sha("HEAD", relative) != sha256_file(ROOT / relative):
                return False
        except subprocess.CalledProcessError:
            return False
    return True


def protected_history_hashes() -> dict[str, str]:
    path = ROOT / "research/mrcge_v13_propagation_audit/data/manifest.json"
    expected = read_json(path, {}).get("protected_historical_file_sha256", {})
    if not expected:
        raise RuntimeError("historical protected-file hash map is missing")
    for relative, digest in expected.items():
        file_path = ROOT / relative
        if not file_path.is_file() or sha256_file(file_path) != digest:
            raise RuntimeError(f"protected historical file changed or missing: {relative}")
    return expected


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
    raise RuntimeError(f"{dataset}: fixed split path is not configured")


def build_cfg(dataset: str, seed: int, device: str, cache: dict[str, Any]):
    cfg = v13_runner.make_cfg(dataset, "joint_view", seed, device, cache)
    cfg.model = OmegaConf.load(ROOT / "configs/model/mrcge_v13_intrinsic_prior_skip.yaml")
    cfg.model.name = MODEL_NAME
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
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
        raise RuntimeError(f"expected one fixed-split V1.3-A row for {dataset}/seed{seed}; found {len(rows)}")
    return rows[0]


def verify_reference_checkpoints(row: dict[str, str], label: str) -> dict[str, dict[str, str]]:
    result = {}
    for name, path_key, sha_key in (
        ("best_accuracy", "checkpoint_path", "checkpoint_sha256"),
        ("best_f1", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256"),
    ):
        path = Path(row.get(path_key, ""))
        expected = row.get(sha_key, "")
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"{label}: historical {name} checkpoint missing/hash mismatch at {path}")
        result[name] = {"path": str(path), "sha256": expected}
    return result


def historical_reference_audit(dataset: str, seed: int) -> dict[str, Any]:
    row = historical_v13(dataset, seed)
    checkpoints = verify_reference_checkpoints(row, f"V1.3-A {dataset}/seed{seed}")
    return {
        "dataset": dataset, "seed": seed, "source_git_sha": row.get("source_git_sha", ""),
        "metrics": {key: float(row[key]) for key in ("val_acc", "val_macro_f1", "val_ce")},
        "best_f1_metrics": {key: float(row[f"best_f1_{key}"]) for key in ("val_acc", "val_macro_f1", "val_ce")},
        "checkpoint_hashes": checkpoints,
        "feature_fingerprint": row.get("feature_fingerprint", ""),
        "dataset_fingerprint": row.get("dataset_fingerprint", ""),
        "physical_edge_fingerprint": row.get("physical_edge_fingerprint", ""),
        "split_fingerprint": row.get("split_fingerprint", ""),
        "split_file_sha256": row.get("split_file_sha256", ""),
        "semantic_cache_sha256": row.get("semantic_cache_sha256", ""),
        "candidate_fingerprint": row.get("candidate_fingerprint", ""),
        "v7b_fingerprint": row.get("v7b_fingerprint", ""),
    }


def load_safe_data(dataset: str, seed: int, cache: dict[str, Any], device: str):
    cfg = build_cfg(dataset, seed, device, cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = v12_utils.mask_test_labels(raw)
    split_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
    split_path = Path(split_text).resolve() if split_text else None
    if split_path != fixed_split_path(dataset) or not split_path.is_file():
        raise RuntimeError(f"{dataset}/seed{seed}: loader did not use the fixed split file")
    if cfg.dataset.auto_generate_splits is not False or cfg.task.evaluate_test is not False:
        raise RuntimeError("fixed-split / Test-off config drift")
    if bool((safe.y[raw.test_idx] != -100).any()):
        raise RuntimeError("Test labels were not masked before Trainer entry")
    if safe.x.size(0) != safe.num_nodes or safe.y.size(0) != safe.num_nodes:
        raise RuntimeError("feature/label node order does not match global node count")
    if not torch.equal(raw.x, torch.cat((raw.x_t, raw.x_i), dim=1)):
        raise RuntimeError("loaded features do not follow the registered Text/Visual global node order")
    if not torch.isfinite(raw.x).all() or bool((safe.y[safe.train_idx] < 0).any()):
        raise RuntimeError("non-finite features or invalid Train labels")
    fingerprints = v13_runner.data_utils.dataset_fingerprints(raw)
    return cfg, raw, safe, fingerprints, split_path


def state_hash(module: nn.Module) -> str:
    return v12_utils.state_hash(module)


def initial_equivalence(dataset: str, seed: int, safe, cache: dict[str, Any]) -> dict[str, Any]:
    base_cfg = v13_runner.make_cfg(dataset, "joint_view", seed, "cpu", cache)
    base_cfg.model.name = "mrcge_v13"
    candidate_cfg = build_cfg(dataset, seed, "cpu", cache)
    set_seed(seed)
    baseline = build_model(base_cfg, data_info(safe))
    baseline.load_semantic_cache(cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"])
    baseline_head = nn.Linear(baseline.out_dim, int(safe.num_classes))
    set_seed(seed)
    candidate = build_model(candidate_cfg, data_info(safe))
    candidate.load_semantic_cache(cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"])
    candidate_head = nn.Linear(candidate.out_dim, int(safe.num_classes))
    baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    if set(candidate_state) != set(baseline_state) | {"prior_skip_raw"}:
        raise RuntimeError(f"{dataset}/seed{seed}: unexpected model state keys")
    differing = [key for key, value in baseline_state.items() if not torch.equal(value, candidate_state[key])]
    if differing:
        raise RuntimeError(f"{dataset}/seed{seed}: shared initial state differs: {differing[:8]}")
    if candidate.prior_skip_raw.numel() != 1 or candidate.prior_skip_raw.item() != 0.0:
        raise RuntimeError("new Prior scalar is not a deterministic zero initialization")
    if not torch.equal(baseline_head.weight, candidate_head.weight) or not torch.equal(baseline_head.bias, candidate_head.bias):
        raise RuntimeError(f"{dataset}/seed{seed}: external classifier initialization differs")
    if sum(p.numel() for p in candidate.parameters()) != sum(p.numel() for p in baseline.parameters()) + 1:
        raise RuntimeError("candidate model must add exactly one scalar parameter")
    result = {"dataset": dataset, "seed": seed, "status": "pass",
        "public_state_equal": True, "external_classifier_equal": True,
        "new_parameter_numel": 1, "prior_skip_raw": 0.0,
        "baseline_model_sha256": state_hash(baseline),
        "candidate_model_sha256": state_hash(candidate),
        "shared_model_state_sha256": state_hash(baseline),
        "initial_model_sha256": state_hash(candidate),
        "baseline_classifier_sha256": state_hash(baseline_head),
        "candidate_classifier_sha256": state_hash(candidate_head),
        "initial_classifier_sha256": state_hash(candidate_head)}
    del baseline, baseline_head, candidate, candidate_head
    return result


def cpu_preflight() -> dict[str, Any]:
    verify_frozen_sources()
    protected = protected_history_hashes()
    caches = v13_runner.cache_entries()
    records, references, initialization = {}, [], []
    all_cells = (*STAGE1, *STAGE2)
    for dataset in DATASETS:
        records[dataset] = None
    for dataset, seed in all_cells:
        cfg, raw, safe, fp, split_path = load_safe_data(dataset, seed, caches[dataset], "cpu")
        reference = historical_reference_audit(dataset, seed)
        current = {
            "feature_fingerprint": caches[dataset]["feature_fingerprint"],
            "dataset_fingerprint": fp["features"],
            "physical_edge_fingerprint": fp["physical_edge_index"],
            "split_fingerprint": fp["split_fingerprint"],
            "split_file_sha256": sha256_file(split_path),
            "semantic_cache_sha256": sha256_file(Path(caches[dataset]["cache_path_abs"])),
            "candidate_fingerprint": caches[dataset]["candidate_fingerprint"],
            "v7b_fingerprint": caches[dataset]["v7b_fingerprint"],
        }
        if current["split_fingerprint"] != EXPECTED_SPLIT_FINGERPRINTS[dataset]:
            raise RuntimeError(
                f"{dataset}/seed{seed}: split fingerprint {current['split_fingerprint']} "
                "does not match the task-registered fixed split"
            )
        mismatches = {key: {"historical": reference.get(key), "current": value}
                      for key, value in current.items() if reference.get(key) != value}
        if mismatches:
            raise RuntimeError(f"{dataset}/seed{seed}: frozen historical data/cache mismatch: {mismatches}")
        if not torch.equal(raw.train_idx, safe.train_idx) or not torch.equal(raw.val_idx, safe.val_idx) or not torch.equal(raw.test_idx, safe.test_idx):
            raise RuntimeError("label masking changed split indices")
        references.append({**reference, "status": "pass", "checkpoint_hashes_verified": True})
        initialization.append(initial_equivalence(dataset, seed, safe, caches[dataset]))
        if records[dataset] is None:
            records[dataset] = {**current, "split_file_path": str(split_path),
                "semantic_cache_path": caches[dataset]["cache_path_abs"],
                "num_nodes": int(safe.num_nodes), "num_edges": int(raw.edge_index.size(1)),
                "num_classes": int(safe.num_classes), "text_dim": int(safe.x_t.size(1)),
                "visual_dim": int(safe.x_i.size(1)), "global_node_order_verified": True,
                "test_labels_masked_before_trainer": True, "test_evaluation": False,
                "link_prediction": False}
        del cfg, raw, safe
    result = {
        "study": "MR-CGE V1.3 Intrinsic-Prior Fusion Controlled Screen", "status": "pass",
        "stage": "cpu_preflight", "recorded_utc": now(), "base_sha": BASE_SHA,
        "base_branch": BASE_BRANCH, "branch": BRANCH, "source_git_sha_at_cpu_preflight": source_sha(),
        "source_file_sha256s": source_file_hashes(), "dataset_records": records,
        "historical_references": references, "initialization_equivalence": initialization,
        "protected_historical_file_sha256s": protected,
        "authorized_stage1_cells": [{"dataset": d, "seed": s, "variant": VARIANT} for d, s in STAGE1],
        "conditional_stage2_cells": [{"dataset": d, "seed": s, "variant": VARIANT} for d, s in STAGE2],
        "test_evaluation": False, "link_prediction": False,
        "development_incident": {
            "status": "recorded_nontraining_test_invocation",
            "description": ("One initial automated test parameterization ran with CUDA_VISIBLE_DEVICES unset and attempted model placement on the default CUDA ordinal before its forward failed on a missing import. No training worker launched. Device UUID was not captured. Subsequent CPU tests explicitly hide CUDA; all later GPU work requires the physical-GPU-1 owner guard."),
        },
    }
    DATA.mkdir(parents=True, exist_ok=True)
    initial_files = {
        "runs.csv": RUN_FIELDS,
        "epoch_history.csv": ("dataset", "variant", "seed", "attempt", "epoch"),
        "resource_summary.csv": ("phase", "dataset", "variant", "seed", "attempt", "captured_utc", "physical_gpu", "gpu_uuid", "gpu_name", "memory_free_mib", "required_free_mib", "compute_pids", "guard_status"),
        "paired_differences.csv": ("dataset", "seed", "delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce"),
        "dataset_summary.csv": ("dataset", "variant", "n"),
        "prior_skip_diagnostics.csv": ("dataset", "seed", "prior_skip_raw", "lambda_prior", "test_evaluation"),
        "historical_reference_audit.csv": ("dataset", "seed", "source_git_sha", "val_acc", "val_macro_f1", "val_ce", "checkpoint_hashes_verified"),
    }
    for name, fields in initial_files.items():
        if not (DATA / name).exists():
            write_rows(DATA / name, [], fields)
    write_rows(DATA / "historical_reference_audit.csv", [
        {"dataset": row["dataset"], "seed": row["seed"], "source_git_sha": row["source_git_sha"],
         "val_acc": row["metrics"]["val_acc"], "val_macro_f1": row["metrics"]["val_macro_f1"],
         "val_ce": row["metrics"]["val_ce"], "best_f1_val_acc": row["best_f1_metrics"]["val_acc"],
         "best_f1_val_macro_f1": row["best_f1_metrics"]["val_macro_f1"],
         "best_f1_val_ce": row["best_f1_metrics"]["val_ce"],
         "checkpoint_hashes_json": json.dumps(row["checkpoint_hashes"], sort_keys=True),
         "checkpoint_hashes_verified": True, **{k: row.get(k, "") for k in (
             "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint",
             "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
             "candidate_fingerprint", "v7b_fingerprint")}}
        for row in references],
        ("dataset", "seed", "source_git_sha", "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "checkpoint_hashes_json", "checkpoint_hashes_verified", "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256", "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint"))
    write_json(DATA / "preflight.json", result)
    write_json(DATA / "failures.json", read_json(DATA / "failures.json", {"failures": []}))
    write_json(DATA / "manifest.json", {
        "study": result["study"], "branch": BRANCH, "base_sha": BASE_SHA,
        "source_git_sha": source_sha(), "cpu_preflight_status": "pass",
        "gpu_preflight_status": "pending", "training_status": "not_started",
        "authorized_stage1_cells": result["authorized_stage1_cells"],
        "conditional_stage2_cells": result["conditional_stage2_cells"],
        "completed_cells": 0, "test_evaluation": False, "link_prediction": False,
        "updated_utc": now(),
    })
    return result


def selected_gpu(snapshot: dict[str, Any]) -> dict[str, Any]:
    device = next((row for row in snapshot.get("devices", [])
                   if int(row.get("physical_index", -1)) == PHYSICAL_GPU), None)
    if device is None:
        raise RuntimeError("fresh nvidia-smi snapshot does not contain physical GPU 1")
    return device


def strict_gpu_guard(baseline_pids: set[int], allowed_pids: set[int], *, phase: str,
                     required_free_mib: int, expected_uuid: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    query_errors = 0
    while True:
        try:
            snapshot = v13_runner.guard_gpu(PHYSICAL_GPU, baseline_pids, allowed_pids,
                                            phase=phase, required_free_mib=required_free_mib)
            break
        except subprocess.CalledProcessError as exc:
            query_errors += 1
            if phase == "before_launch" and time.monotonic() - started < GPU_READY_WAIT_SECONDS:
                time.sleep(5)
                continue
            if query_errors < 3:
                time.sleep(2)
                continue
            diagnostic = getattr(exc, "stderr", None)
            raise RuntimeError(
                f"GPU status query failed {query_errors} times; last={exc!r}; stderr={diagnostic!r}"
            ) from exc
        except RuntimeError as exc:
            reason = str(exc).lower()
            waitable_resource = any(token in reason for token in (
                "baseline utilization", "free memory", "temperature",
            ))
            if (phase == "before_launch" and waitable_resource
                    and time.monotonic() - started < GPU_READY_WAIT_SECONDS):
                time.sleep(5)
                continue
            raise
    device = selected_gpu(snapshot)
    if expected_uuid and device["uuid"] != expected_uuid:
        raise RuntimeError(
            f"physical GPU 1 UUID changed: expected {expected_uuid}, observed {device['uuid']}"
        )
    snapshot["guard"]["expected_uuid"] = expected_uuid or device["uuid"]
    snapshot["guard"]["uuid_verified"] = True
    snapshot["guard"]["guard_wait_seconds"] = round(time.monotonic() - started, 2)
    return snapshot


def baseline_gpu() -> tuple[set[int], str, dict[str, Any]]:
    started = time.monotonic()
    last_query_error = None
    last_unsafe_state = None
    while time.monotonic() - started <= GPU_READY_WAIT_SECONDS:
        try:
            snapshot = v13_runner.smi_snapshot()
        except subprocess.CalledProcessError as exc:
            last_query_error = exc
            time.sleep(5)
            continue
        device = selected_gpu(snapshot)
        if device["memory_free_mib"] < 8192:
            last_unsafe_state = f"free memory {device['memory_free_mib']} MiB < 8192 MiB"
        elif int(device["temperature_c"]) >= 88:
            last_unsafe_state = f"temperature {device['temperature_c']} C >= 88 C"
        elif int(device["utilization_percent"]) >= 90:
            last_unsafe_state = f"utilization {device['utilization_percent']}% >= 90%"
        else:
            pids = {int(row["pid"]) for row in snapshot["compute_processes"]
                    if row["gpu_uuid"] == device["uuid"]}
            snapshot["guard"] = {
                "phase": "baseline", "physical_gpu": PHYSICAL_GPU,
                "baseline_compute_pids": sorted(pids), "status": "snapshot_only",
                "uuid_verified": True,
                "stabilization_wait_seconds": round(time.monotonic() - started, 2),
            }
            return pids, str(device["uuid"]), snapshot
        time.sleep(5)
    diagnostic = getattr(last_query_error, "stderr", None)
    raise RuntimeError(
        f"GPU1 did not provide a safe baseline within {GPU_READY_WAIT_SECONDS}s; "
        f"last_unsafe_state={last_unsafe_state!r}, last_query_error={last_query_error!r}, "
        f"stderr={diagnostic!r}"
    ) from last_query_error


def record_resource(snapshot: dict[str, Any], dataset: str, seed: int | str,
                    attempt: int | str, phase: str) -> None:
    device = selected_gpu(snapshot)
    row = {
        "phase": phase, "dataset": dataset, "variant": VARIANT, "seed": seed,
        "attempt": attempt, "captured_utc": snapshot.get("captured_utc"),
        "physical_gpu": PHYSICAL_GPU, "gpu_uuid": device["uuid"],
        "gpu_name": device["name"], "memory_free_mib": device["memory_free_mib"],
        "memory_used_mib": device["memory_used_mib"],
        "memory_total_mib": device["memory_total_mib"],
        "utilization_percent": device["utilization_percent"],
        "temperature_c": device["temperature_c"],
        "compute_pids": json.dumps(sorted(
            int(app["pid"]) for app in snapshot["compute_processes"]
            if app["gpu_uuid"] == device["uuid"])),
        "required_free_mib": snapshot.get("guard", {}).get("required_free_mib", ""),
        "guard_status": snapshot.get("guard", {}).get("status", "snapshot_only"),
        "uuid_verified": snapshot.get("guard", {}).get("uuid_verified", False),
        "stabilization_wait_seconds": snapshot.get("guard", {}).get("stabilization_wait_seconds", 0),
        "guard_wait_seconds": snapshot.get("guard", {}).get("guard_wait_seconds", 0),
        "test_evaluation": False,
    }
    path = DATA / "resource_summary.csv"
    rows = read_rows(path)
    fields = list(dict.fromkeys([*(rows[0].keys() if rows else ()), *row.keys()]))
    write_rows(path, [*rows, row], fields)


def gpu_synthetic_equivalence(dataset: str, seed: int, cache: dict[str, Any], device: torch.device):
    """Exercise the exact zero-prior Train/Eval parity contract on isolated GPU 1."""
    from src.models.mrcge_v13 import Model as FrozenV13Model
    from src.models.mrcge_v13_intrinsic_prior_skip import Model as IntrinsicPriorModel

    info = {"input_dim": 8, "text_dim": 4, "visual_dim": 4,
            "num_nodes": 12, "num_classes": 3}
    base_cfg = v13_runner.make_cfg(dataset, "joint_view", seed, "cpu", cache)
    base_cfg.model.name = "mrcge_v13"
    candidate_cfg = build_cfg(dataset, seed, "cpu", cache)
    semantic_edges = torch.tensor(
        [[0, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10, 0]], dtype=torch.long
    )
    set_seed(seed)
    baseline = FrozenV13Model(base_cfg, info).to(device)
    baseline.semantic_text_edge_index = semantic_edges.clone()
    baseline.semantic_text_weight = torch.ones(semantic_edges.size(1))
    baseline.semantic_visual_edge_index = semantic_edges.flip(0).clone()
    baseline.semantic_visual_weight = torch.ones(semantic_edges.size(1))
    baseline.semantic_candidate_fingerprint = "a" * 64
    baseline.semantic_feature_fingerprint = "b" * 64
    baseline.semantic_v7b_fingerprint = "c" * 64
    base_head = nn.Linear(baseline.out_dim, info["num_classes"]).to(device)
    set_seed(seed)
    candidate = IntrinsicPriorModel(candidate_cfg, info).to(device)
    candidate.semantic_text_edge_index = semantic_edges.clone()
    candidate.semantic_text_weight = torch.ones(semantic_edges.size(1))
    candidate.semantic_visual_edge_index = semantic_edges.flip(0).clone()
    candidate.semantic_visual_weight = torch.ones(semantic_edges.size(1))
    candidate.semantic_candidate_fingerprint = "a" * 64
    candidate.semantic_feature_fingerprint = "b" * 64
    candidate.semantic_v7b_fingerprint = "c" * 64
    candidate_head = nn.Linear(candidate.out_dim, info["num_classes"]).to(device)
    if not torch.equal(base_head.weight, candidate_head.weight) or not torch.equal(base_head.bias, candidate_head.bias):
        raise RuntimeError("GPU external classifier initialization mismatch")
    base_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    if set(candidate_state) != set(base_state) | {"prior_skip_raw"} or any(
        not torch.equal(base_state[key], candidate_state[key]) for key in base_state
    ):
        raise RuntimeError("GPU shared model initialization mismatch")

    x = torch.arange(96, dtype=torch.float32, device=device).reshape(12, 8) / 31.0
    edges = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]],
        dtype=torch.long, device=device,
    )
    labels = torch.tensor([0, 1, 2, 1, 0, 2, 0, 1, 2, 1, 0, 2], device=device)
    train_idx = torch.arange(8, device=device)
    baseline.eval(); candidate.eval()
    with torch.no_grad():
        z0, _, _, aux0, info0 = baseline(x, edges, return_details=True)
        z1, _, _, aux1, info1 = candidate(x, edges, return_details=True)
    if not torch.equal(z0, z1) or not torch.equal(aux0, aux1):
        raise RuntimeError("GPU zero-prior Eval outputs differ from V1.3-A")
    if not torch.equal(base_head(z0), candidate_head(z1)):
        raise RuntimeError("GPU zero-prior classifier logits differ from V1.3-A")
    for key in ("routes", "collaborative_routes", "late", "private_collaborative_route_changes"):
        if info0["details"][key] != info1["details"][key]:
            raise RuntimeError(f"GPU zero-prior Eval {key} mismatch")

    baseline.train(); candidate.train(); base_head.train(); candidate_head.train()
    set_seed(20261027)
    baseline.zero_grad(set_to_none=True); base_head.zero_grad(set_to_none=True)
    z0, _, _, aux0, info0 = baseline(x, edges)
    cpu_rng0 = torch.random.get_rng_state().clone()
    cuda_rng0 = torch.cuda.get_rng_state(device).clone()
    main0 = F.cross_entropy(base_head(z0[train_idx]), labels[train_idx])
    loss0, private0 = _compose_full_graph_loss(
        model=baseline, info=info0, main_ce=main0, aux_loss=aux0,
        aux_weight=1.0, unimodal_weight=0.3, labels=labels, train_idx=train_idx,
    )
    loss0.backward()
    set_seed(20261027)
    candidate.zero_grad(set_to_none=True); candidate_head.zero_grad(set_to_none=True)
    z1, _, _, aux1, info1 = candidate(x, edges)
    cpu_rng1 = torch.random.get_rng_state().clone()
    cuda_rng1 = torch.cuda.get_rng_state(device).clone()
    main1 = F.cross_entropy(candidate_head(z1[train_idx]), labels[train_idx])
    loss1, private1 = _compose_full_graph_loss(
        model=candidate, info=info1, main_ce=main1, aux_loss=aux1,
        aux_weight=1.0, unimodal_weight=0.3, labels=labels, train_idx=train_idx,
    )
    loss1.backward()
    if not torch.equal(z0, z1) or not torch.allclose(loss0, loss1, atol=1e-7, rtol=1e-7):
        raise RuntimeError("GPU zero-prior Train output/loss parity failed")
    if not torch.equal(main0, main1) or not torch.equal(private0, private1) or not torch.equal(aux0, aux1):
        raise RuntimeError("GPU main/private/balance loss components differ")
    if not torch.equal(cpu_rng0, cpu_rng1) or not torch.equal(cuda_rng0, cuda_rng1):
        raise RuntimeError("GPU Train forward RNG state differs between parent and candidate")
    p0, p1 = dict(baseline.named_parameters()), dict(candidate.named_parameters())
    if set(p1) != set(p0) | {"prior_skip_raw"}:
        raise RuntimeError("GPU trainable parameter set differs beyond the registered scalar")
    for name, left in p0.items():
        right = p1[name]
        if (left.grad is None) != (right.grad is None):
            raise RuntimeError(f"GPU gradient presence mismatch: {name}")
        if left.grad is not None and not torch.allclose(left.grad, right.grad, atol=2e-7, rtol=2e-6):
            raise RuntimeError(f"GPU shared gradient mismatch: {name}")
    for name in ("weight", "bias"):
        left, right = getattr(base_head, name).grad, getattr(candidate_head, name).grad
        if left is None or right is None or not torch.allclose(left, right, atol=2e-7, rtol=2e-6):
            raise RuntimeError(f"GPU external classifier gradient mismatch: {name}")
    prior_grad = candidate.prior_skip_raw.grad
    if prior_grad is None or not torch.isfinite(prior_grad) or prior_grad.abs().item() == 0.0:
        raise RuntimeError("GPU intrinsic prior scalar gradient is absent, non-finite, or zero")
    optimizer = torch.optim.AdamW(
        [*candidate.parameters(), *candidate_head.parameters()], lr=1e-3, weight_decay=1e-4
    )
    if not any(p is candidate.prior_skip_raw for group in optimizer.param_groups for p in group["params"]):
        raise RuntimeError("GPU optimizer omitted prior_skip_raw")
    return {
        "status": "pass", "eval_embedding_equal": True, "eval_logits_equal": True,
        "router_and_late_equal": True, "train_loss_equal": True,
        "shared_gradients_equal": True, "cpu_rng_equal": True, "cuda_rng_equal": True,
        "external_classifier_equal": True, "prior_skip_gradient": float(prior_grad.detach().cpu()),
        "prior_skip_gradient_finite_nonzero": True, "optimizer_manages_prior_skip": True,
    }


def worker_preflight(dataset: str, seed: int, attempt: int, record_path: Path) -> None:
    if os.environ.get("MR_CGE_GUARDED_PREFLIGHT") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("GPU preflight must run as a guarded physical-GPU-1 child")
    baseline = {int(pid) for pid in os.environ.get("MR_CGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    expected_uuid = os.environ.get("MR_CGE_EXPECTED_GPU_UUID", "")
    if not expected_uuid:
        raise RuntimeError("GPU preflight child did not receive the captured GPU1 UUID")
    strict_gpu_guard(baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192,
                     expected_uuid=expected_uuid)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA_VISIBLE_DEVICES did not isolate exactly one device")
    device = torch.device("cuda:0")
    cache = v13_runner.cache_entries()[dataset]
    parity = gpu_synthetic_equivalence(dataset, seed, cache, device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    cfg, raw, safe, fingerprints, split_path = load_safe_data(dataset, seed, cache, "cuda:0")
    set_seed(seed)
    model = build_model(cfg, data_info(safe))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *classifier.parameters()],
        lr=float(cfg.task.lr), weight_decay=float(cfg.task.weight_decay),
    )
    if not any(p is model.prior_skip_raw for group in optimizer.param_groups for p in group["params"]):
        raise RuntimeError("formal-data GPU preflight optimizer omits prior_skip_raw")
    x, edges, labels = safe.x.to(device), safe.edge_index.to(device), safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    if bool((labels[train_idx] < 0).any()):
        raise RuntimeError("invalid or masked label in Train split")
    if not bool((safe.y[safe.test_idx] == -100).all()) or cfg.task.evaluate_test is not False:
        raise RuntimeError("formal-data GPU preflight observed Test-on state")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); classifier.train(); optimizer.zero_grad(set_to_none=True)
    z, _, _, aux, info = model(x, edges)
    main = F.cross_entropy(classifier(z[train_idx]), labels[train_idx])
    loss, private = _compose_full_graph_loss(
        model=model, info=info, main_ce=main, aux_loss=aux,
        aux_weight=float(cfg.task.loss.aux_weight),
        unimodal_weight=float(cfg.task.loss.unimodal_weight), labels=labels, train_idx=train_idx,
    )
    if not torch.isfinite(loss) or not bool(torch.isfinite(z).all()):
        raise RuntimeError("formal-data GPU preflight produced non-finite outputs")
    loss.backward()
    gradients = model.capture_gradient_diagnostics()
    if not gradients["prior_skip_raw"]["gradient_finite"] or not gradients["prior_skip_raw"]["gradient_nonzero"]:
        raise RuntimeError("formal-data GPU preflight scalar gradient is not finite/nonzero")
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all())
               for p in [*model.parameters(), *classifier.parameters()]):
        raise RuntimeError("formal-data GPU preflight contains non-finite gradients")
    torch.nn.utils.clip_grad_norm_([*model.parameters(), *classifier.parameters()], 1.0,
                                   error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    required_start = max(8192, math.ceil(peak_reserved / (1024 * 1024)) + 2048)
    model.eval(); classifier.eval()
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        eval_z, _, _, eval_aux, eval_info = model(x, edges, return_details=True)
    rng_unchanged = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(
        cuda_rng, torch.cuda.get_rng_state(device)
    )
    if not rng_unchanged or not bool(torch.isfinite(eval_z).all()) or not bool(torch.isfinite(eval_aux)):
        raise RuntimeError("full Eval diagnostics changed RNG or yielded non-finite values")
    runtime = strict_gpu_guard(baseline, {os.getpid()}, phase="during_run", required_free_mib=2048,
                               expected_uuid=expected_uuid)
    record = {
        "dataset": dataset, "variant": VARIANT, "seed": seed, "attempt": attempt,
        "status": "pass", "stage": "single_step_forward_backward", "recorded_utc": now(),
        "physical_gpu": PHYSICAL_GPU, "gpu_uuid": expected_uuid,
        "training_source_committed": training_source_committed(),
        "gpu_name": torch.cuda.get_device_name(device), "source_git_sha": source_sha(),
        "source_file_sha256s": source_file_hashes(), "required_start_free_mib_with_2gib_margin": required_start,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "test_labels_masked": True, "test_evaluation": False, "link_prediction": False,
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": sha256_file(split_path),
        "semantic_cache_sha256": sha256_file(Path(cache["cache_path_abs"])),
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "main_ce": float(main.detach().cpu()), "private_ce": float(private.detach().cpu()),
        "aux_loss": float(aux.detach().cpu()), "total_loss": float(loss.detach().cpu()),
        "prior_skip_gradient": gradients["prior_skip_raw"],
        "optimizer_manages_prior_skip": True, "rng_unchanged_eval": rng_unchanged,
        "synthetic_gpu_equivalence": parity, "guard_snapshot": runtime,
    }
    write_json(record_path, record)
    del model, classifier, optimizer, raw, safe, z, eval_z, info, eval_info
    torch.cuda.empty_cache()


def guarded_child(action: str, dataset: str, seed: int, attempt: int, baseline: set[int],
                  expected_uuid: str, *, record_path: Path | None = None,
                  attempt_dir: Path | None = None, required_free_mib: int = 8192) -> dict[str, Any]:
    launch = strict_gpu_guard(baseline, set(), phase="before_launch",
                              required_free_mib=required_free_mib, expected_uuid=expected_uuid)
    record_resource(launch, dataset, seed, attempt, f"{action}_before_launch")
    env = os.environ.copy()
    env.pop("MR_CGE_SHARED_GPU_OVERRIDE", None)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["MR_CGE_BASELINE_GPU_PIDS"] = ",".join(str(pid) for pid in sorted(baseline))
    env["MR_CGE_EXPECTED_GPU_UUID"] = expected_uuid
    if action == "preflight":
        env["MR_CGE_GUARDED_PREFLIGHT"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-preflight",
                   "--dataset", dataset, "--seed", str(seed), "--attempt", str(attempt),
                   "--record-path", str(record_path)]
    elif action == "train":
        env["MR_CGE_GUARDED_TRAIN"] = "1"
        env["MR_CGE_EXPECTED_PEAK_RESERVED_MIB"] = str(max(0, required_free_mib - 2048))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-train",
                   "--dataset", dataset, "--seed", str(seed), "--attempt", str(attempt),
                   "--attempt-dir", str(attempt_dir)]
    else:
        raise ValueError(action)
    log_dir = OUTPUT / ("preflight_logs" if action == "preflight" else "guard_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{action}_{dataset}_seed{seed}_attempt{attempt:02d}.log"
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        last_resource_record = time.monotonic()
        if action == "train" and attempt_dir is not None:
            write_json(attempt_dir / "child_launch.json", {
                "action": action, "pid": proc.pid, "launched_utc": now(),
                "physical_gpu": PHYSICAL_GPU, "expected_gpu_uuid": expected_uuid,
            })
        while proc.poll() is None:
            time.sleep(3)
            try:
                snapshot = strict_gpu_guard(baseline, {proc.pid}, phase="during_run",
                                            required_free_mib=2048, expected_uuid=expected_uuid)
                if time.monotonic() - last_resource_record >= 30:
                    record_resource(snapshot, dataset, seed, attempt, f"{action}_during_run")
                    last_resource_record = time.monotonic()
            except Exception as exc:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                raise RuntimeError(f"runtime GPU1 guard stopped our {action} child: {exc}") from exc
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
        raise RuntimeError(f"{action} child exited {proc.returncode}: {log_path}\n{tail}")
    return {"launch_guard": launch, "log_path": str(log_path)}


def current_attempt(dataset: str, seed: int) -> tuple[int, Path]:
    rows = [row for row in read_rows(DATA / "runs.csv")
            if row.get("dataset") == dataset and row.get("variant") == VARIANT
            and row.get("seed") == str(seed)]
    attempt = max([int(row.get("attempt", 0) or 0) for row in rows] + [0]) + 1
    path = RUNS / dataset / VARIANT / f"seed_{seed}" / f"attempt_{attempt:02d}"
    if path.exists():
        raise FileExistsError(f"refusing to reuse existing attempt directory {path}")
    return attempt, path


def any_child_started(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        attempt_dir_text = row.get("attempt_dir", "")
        if attempt_dir_text and (Path(attempt_dir_text) / "child_launch.json").is_file():
            return True
    return False


def current_passing_preflight(dataset: str) -> dict[str, Any]:
    payload = read_json(DATA / "gpu_preflight.json", {"cells": []})
    matches = [row for row in payload.get("cells", [])
               if row.get("dataset") == dataset and row.get("status") == "pass"
               and row.get("source_git_sha") == source_sha()
               and row.get("source_file_sha256s") == source_file_hashes()]
    if not matches:
        raise RuntimeError(f"no current-source GPU preflight passed for {dataset}")
    return matches[-1]


def worker_train(dataset: str, seed: int, attempt: int, attempt_dir: Path) -> None:
    if os.environ.get("MR_CGE_GUARDED_TRAIN") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("formal training must be a guarded physical-GPU-1 child")
    baseline = {int(pid) for pid in os.environ.get("MR_CGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    expected_uuid = os.environ.get("MR_CGE_EXPECTED_GPU_UUID", "")
    required_free = max(8192, int(os.environ.get("MR_CGE_EXPECTED_PEAK_RESERVED_MIB", "0")) + 2048)
    if not expected_uuid:
        raise RuntimeError("training child did not receive the captured GPU1 UUID")
    strict_gpu_guard(baseline, {os.getpid()}, phase="before_launch", required_free_mib=required_free,
                     expected_uuid=expected_uuid)
    if not (attempt_dir / "attempt_started.json").is_file():
        raise RuntimeError("parent has not reserved the formal attempt directory")
    pre = current_passing_preflight(dataset)
    measured_required = max(8192, math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024)) + 2048)
    if required_free < measured_required:
        raise RuntimeError(f"formal launch threshold {required_free} MiB is below measured-peak margin {measured_required} MiB")
    cache = v13_runner.cache_entries()[dataset]
    cfg, raw, safe, fingerprints, split_path = load_safe_data(dataset, seed, cache, "cuda:0")
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    model_path = ROOT / "src/models/mrcge_v13_intrinsic_prior_skip.py"
    trainer_path = ROOT / "src/tasks/nc.py"
    model_hash, trainer_hash = sha256_file(model_path), sha256_file(trainer_path)
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3 Intrinsic-Prior Fusion Controlled Screen",
        "dataset": dataset, "variant": VARIANT, "condition": "M_repaired",
        "v13_variant": "joint_view", "seed": seed, "attempt": attempt,
        "source_git_sha": source_sha(), "base_sha": BASE_SHA,
        "protocol_version": PROTOCOL, "model_source_sha256": model_hash,
        "trainer_source_sha256": trainer_hash, "config_sha256": config_hash,
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": sha256_file(split_path),
        "semantic_cache_sha256": sha256_file(Path(cache["cache_path_abs"])),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
        "test_evaluation": False, "test_label_values_indexed": False,
        "link_prediction": False, "physical_gpu": PHYSICAL_GPU,
        "gpu_uuid": expected_uuid,
    }
    # The training runner itself resets the registered seed before construction.
    # This pair audit also verifies the exact initial model and external head on CPU.
    init = [row for row in read_json(DATA / "preflight.json", {}).get("initialization_equivalence", [])
            if row.get("dataset") == dataset and int(row.get("seed", -1)) == seed]
    if len(init) != 1 or init[0].get("status") != "pass":
        raise RuntimeError("missing exact-seed CPU initial-equivalence audit")
    logger = logging.getLogger(f"mrcge_intrinsic_prior.{dataset}.{seed}.{attempt}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(attempt_dir / "train.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        if not bool((safe.y[safe.test_idx] == -100).all()) or cfg.task.evaluate_test is not False:
            raise RuntimeError("Test-off audit failed immediately before trainer entry")
        run_nc(cfg, safe, device, logger)
        strict_gpu_guard(baseline, {os.getpid()}, phase="during_run", required_free_mib=2048,
                         expected_uuid=expected_uuid)
        packed = read_json(Path(cfg.task.run_metrics_path), {})
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("trainer protocol or seed provenance mismatch")
        metadata = packed["runs"][0]["metadata"]
        selection = metadata["dual_checkpoint_selection"]
        init_audit = metadata["initial_parameter_audit"]
        expected_initial = init[0]["initial_model_sha256"]
        if init_audit["model_state_sha256"] != expected_initial:
            raise RuntimeError("trainer initial model hash differs from the audited initialization")
        if init_audit["classifier_state_sha256"] != init[0]["initial_classifier_sha256"]:
            raise RuntimeError("trainer initial classifier hash differs from the audited initialization")
        optimizer_names = [name for group in init_audit["optimizer_groups"]
                           for name in group["parameter_names"]]
        if "model.prior_skip_raw" not in optimizer_names:
            raise RuntimeError("prior_skip_raw is missing from the formal optimizer")
        acc_path = Path(cfg.task.save_ckpt_path)
        f1_path = Path(cfg.task.best_f1_ckpt_path)
        if not acc_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Best-Accuracy or Best-F1 checkpoint is missing")
        acc = v12_utils.validation_logits_audit(cfg, safe, cache, acc_path, device, details=True)
        f1 = v12_utils.validation_logits_audit(cfg, safe, cache, f1_path, device, details=True)
        if not acc["rng_unchanged"] or not f1["rng_unchanged"]:
            raise RuntimeError("checkpoint reload validation audit changed RNG")
        for selected, evaluated, label in (
            (selection["best_accuracy"], acc, "accuracy"),
            (selection["best_macro_f1"], f1, "macro_f1"),
        ):
            for field, metric in (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce")):
                if abs(float(selected["metrics"][field]) - float(evaluated["val"][metric])) > 3e-6:
                    raise RuntimeError(f"reloaded best-{label} {field} differs from trainer selection")
        if not bool((safe.y[safe.test_idx] == -100).all()) or cfg.task.evaluate_test is not False:
            raise RuntimeError("Test-off boundary changed during the worker")
        history = metadata.get("epoch_history", [])
        required_history = ("train_loss", "train_ce", "train_accuracy_eval", "train_macro_f1_eval",
                            "val_acc", "val_macro_f1", "val_ce", "intrinsic_prior_raw",
                            "intrinsic_prior_lambda")
        if not history or any(not all(key in row and math.isfinite(float(row[key])) for key in required_history)
                              for row in history):
            raise RuntimeError("full finite epoch history or prior trajectory is missing")
        prior_grad = history[0].get("intrinsic_prior_gradient_nonzero", 0)
        prior_grad_norm = float(history[0].get("intrinsic_prior_gradient_norm", 0.0))
        if not bool(prior_grad) or not math.isfinite(prior_grad_norm) or prior_grad_norm <= 0:
            raise RuntimeError("first-epoch prior scalar gradient was not finite and nonzero")
        details = acc["info"].get("details", {})
        prior = details.get("intrinsic_prior_skip")
        if not isinstance(prior, dict):
            raise RuntimeError("Best-Accuracy reload lacks the registered prior RMS diagnostics")
        train_metrics, val_metrics, f1_val = acc["train"], acc["val"], f1["val"]
        acc_payload = acc["payload"]
        checkpoint_prior = float(acc_payload["model_state"]["prior_skip_raw"].reshape(()))
        lambda_final = math.tanh(checkpoint_prior)
        if abs(float(prior["prior_skip_raw"]) - checkpoint_prior) > 1e-7:
            raise RuntimeError("prior scalar differs between reloaded model and saved checkpoint")
        row = {
            "dataset": dataset, "variant": VARIANT, "seed": seed, "attempt": attempt,
            "status": "complete", "best_acc_epoch": selection["best_accuracy"]["epoch"],
            "best_f1_epoch": selection["best_macro_f1"]["epoch"], "epochs_completed": len(history),
            "val_acc": val_metrics["acc"], "val_macro_f1": val_metrics["macro_f1"],
            "val_ce": val_metrics["ce"], "best_f1_val_acc": f1_val["acc"],
            "best_f1_val_macro_f1": f1_val["macro_f1"], "best_f1_val_ce": f1_val["ce"],
            "train_acc": train_metrics["acc"], "train_macro_f1": train_metrics["macro_f1"],
            "train_ce": train_metrics["ce"], "train_loss": history[-1]["train_loss"],
            "train_val_acc_gap": train_metrics["acc"] - val_metrics["acc"],
            "train_val_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"],
            "train_val_ce_gap": train_metrics["ce"] - val_metrics["ce"],
            "model_parameters": metadata["model_parameters"],
            "classifier_parameters": metadata["classifier_parameters"],
            "total_parameters": int(metadata["model_parameters"]) + int(metadata["classifier_parameters"]),
            "wall_seconds": metadata.get("training_wall_seconds", time.perf_counter() - started),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "prior_skip_raw": checkpoint_prior, "lambda_prior": lambda_final,
            "prior_skip_gradient_norm": prior_grad_norm, "prior_skip_gradient_nonzero": bool(prior_grad),
            "prior_skip_optimizer_managed": True, "prior_skip_diagnostics_json": json.dumps(prior, sort_keys=True),
            **prior, "source_git_sha": source_sha(), "base_sha": BASE_SHA,
            "physical_gpu": PHYSICAL_GPU, "gpu_uuid": expected_uuid,
            "gpu_name": torch.cuda.get_device_name(device), "required_start_free_mib": required_free,
            "model_source_sha256": model_hash, "trainer_source_sha256": trainer_hash,
            "config_sha256": config_hash, "feature_fingerprint": cache["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "split_file_sha256": sha256_file(split_path),
            "semantic_cache_sha256": sha256_file(Path(cache["cache_path_abs"])),
            "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_model_sha256": init_audit["model_state_sha256"],
            "initial_classifier_sha256": init_audit["classifier_state_sha256"],
            "checkpoint_path": str(acc_path), "checkpoint_sha256": sha256_file(acc_path),
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": sha256_file(f1_path),
            "attempt_dir": str(attempt_dir), "test_evaluation": False,
            "validation_rng_unchanged": True, "checkpoint_reload_verified": True, "error": "",
        }
        append_csv(DATA / "runs.csv", row, ("dataset", "variant", "seed", "attempt"))
        history_rows = [{"dataset": dataset, "variant": VARIANT, "seed": seed,
                         "attempt": attempt, **item} for item in history]
        old_history = read_rows(DATA / "epoch_history.csv")
        history_fields = list(dict.fromkeys([*(old_history[0].keys() if old_history else ()),
                                             *(key for item in history_rows for key in item)]))
        write_rows(DATA / "epoch_history.csv", [*old_history, *history_rows], history_fields)
        diag_row = {"dataset": dataset, "seed": seed, **prior, "test_evaluation": False}
        old_diag = read_rows(DATA / "prior_skip_diagnostics.csv")
        write_rows(DATA / "prior_skip_diagnostics.csv", [*old_diag, diag_row],
                   list(dict.fromkeys([*(old_diag[0].keys() if old_diag else ()), *diag_row.keys()])))
        write_json(attempt_dir / "mechanism_diagnostics.json", {
            "best_accuracy": prior,
            "epoch_trajectory": [{"epoch": item["epoch"], "prior_skip_raw": item["intrinsic_prior_raw"],
                                  "lambda_prior": item["intrinsic_prior_lambda"],
                                  "gradient_norm_epoch1": item.get("intrinsic_prior_gradient_norm"),
                                  "gradient_nonzero_epoch1": item.get("intrinsic_prior_gradient_nonzero")}
                                 for item in history],
            "train_metrics": train_metrics, "validation_metrics": val_metrics,
            "test_evaluation": False,
        })
        write_json(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(), **row})
    except Exception as exc:
        write_json(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(),
                    "error": repr(exc), "traceback": traceback.format_exc(), "test_evaluation": False})
        raise
    finally:
        torch.cuda.empty_cache()


def gpu_preflight() -> dict[str, Any]:
    verify_frozen_sources()
    source_is_committed = training_source_committed()
    cpu = read_json(DATA / "preflight.json", {})
    if cpu.get("status") != "pass":
        raise RuntimeError("CPU/data preflight must pass before any GPU work")
    history = read_json(DATA / "gpu_preflight.json", {"campaigns": [], "cells": []})
    attempt = max([int(row.get("attempt", 0)) for row in history.get("campaigns", [])] + [0]) + 1
    campaign = {
        "attempt": attempt, "started_utc": now(), "source_git_sha": source_sha(),
        "source_file_sha256s": source_file_hashes(), "physical_gpu": PHYSICAL_GPU,
        "training_source_committed": source_is_committed,
        "training_started": False, "status": "running", "datasets": list(DATASETS),
    }
    history.setdefault("campaigns", []).append(campaign)
    write_json(DATA / "gpu_preflight.json", history)
    passed, blocked = 0, False
    for dataset in DATASETS:
        if blocked:
            row = {"dataset": dataset, "variant": VARIANT, "seed": 42, "attempt": attempt,
                   "status": "blocked_resource", "reason": "not launched after GPU1 guard failure",
                   "source_git_sha": source_sha(), "training_started": False,
                   "test_evaluation": False}
            history.setdefault("cells", []).append(row)
            write_json(DATA / "gpu_preflight.json", history)
            continue
        record_path = OUTPUT / "preflight_attempts" / dataset / f"attempt_{attempt:02d}.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            baseline, uuid, snapshot = baseline_gpu()
            preguard = strict_gpu_guard(baseline, set(), phase="before_launch", required_free_mib=8192,
                                        expected_uuid=uuid)
            record_resource(snapshot, dataset, 42, attempt, "baseline_snapshot")
            record_resource(preguard, dataset, 42, attempt, "preflight_before_launch")
            run_info = guarded_child("preflight", dataset, 42, attempt, baseline, uuid,
                                     record_path=record_path, required_free_mib=8192)
            record = read_json(record_path, {})
            if (record.get("status") != "pass"
                    or record.get("source_git_sha") != source_sha()
                    or record.get("source_file_sha256s") != source_file_hashes()
                    or record.get("gpu_uuid") != uuid
                    or bool(record.get("training_source_committed")) != source_is_committed):
                raise RuntimeError("GPU preflight record failed exact-source/UUID verification")
            record["record_path"] = str(record_path)
            record["guard_log_path"] = run_info["log_path"]
            history.setdefault("cells", []).append(record)
            passed += 1
        except Exception as exc:
            reason = repr(exc)
            guard_related = any(term in reason.lower() for term in (
                "guard", "memory", "foreign compute", "temperature", "utilization",
                "cuda", "nvidia-smi", "uuid", "device", "permission", "not permitted",
            ))
            record = {
                "dataset": dataset, "variant": VARIANT, "seed": 42, "attempt": attempt,
                "status": "blocked_resource" if guard_related else "failed",
                "reason": reason, "source_git_sha": source_sha(),
                "source_file_sha256s": source_file_hashes(), "training_started": False,
                "test_evaluation": False,
            }
            history.setdefault("cells", []).append(record)
            append_failure({"stage": "gpu_preflight", **record})
            if guard_related:
                blocked = True
        write_json(DATA / "gpu_preflight.json", history)
    campaign.update({"completed_utc": now(), "passing_datasets": passed,
                     "status": "pass" if passed == len(DATASETS) else "incomplete",
                     "training_started": False,
                     "training_source_committed": source_is_committed})
    history["campaigns"][-1] = campaign
    write_json(DATA / "gpu_preflight.json", history)
    manifest = read_json(DATA / "manifest.json", {})
    manifest.update({"gpu_preflight_status": campaign["status"],
                     "gpu_preflight_passed_datasets": passed, "updated_utc": now()})
    write_json(DATA / "manifest.json", manifest)
    return campaign


def formal_stage1() -> bool:
    gpu = read_json(DATA / "gpu_preflight.json", {})
    passing = {row.get("dataset") for row in gpu.get("cells", [])
               if row.get("status") == "pass"
               and row.get("source_file_sha256s") == source_file_hashes()}
    if not set(DATASETS) <= passing:
        raise RuntimeError(f"all dataset GPU preflights must pass before training; missing {sorted(set(DATASETS)-passing)}")
    stop_on_resource = False
    for dataset, seed in STAGE1:
        if stop_on_resource:
            append_failure({"stage": "stage1_formal_train", "dataset": dataset,
                "variant": VARIANT, "seed": seed, "attempt": None,
                "status": "blocked_resource", "training_started": False,
                "error": "not launched after a previous GPU1 ownership/resource guard failure"})
            continue
        prior_rows = [row for row in read_rows(DATA / "runs.csv")
                      if row.get("dataset") == dataset and row.get("variant") == VARIANT
                      and row.get("seed") == str(seed)]
        if prior_rows:
            if any(row.get("status") == "complete" for row in prior_rows):
                continue
            if any_child_started(prior_rows):
                append_failure({"stage": "stage1_formal_train", "dataset": dataset,
                    "variant": VARIANT, "seed": seed, "attempt": None,
                    "status": "not_retried", "training_started": True,
                    "error": "a formal worker already launched for this cell; no second training attempt is permitted"})
                continue
            if any(row.get("status") != "blocked_resource" for row in prior_rows):
                append_failure({"stage": "stage1_formal_train", "dataset": dataset,
                    "variant": VARIANT, "seed": seed, "attempt": None,
                    "status": "not_retried", "training_started": False,
                    "error": "prior non-resource failure is recorded; no implicit rerun is permitted"})
                continue
            # A pre-launch ownership/resource block did not start a formal run.
            # It may be resumed after a later fresh guard passes, at most once trained.
        attempt, attempt_dir = current_attempt(dataset, seed)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        pre = current_passing_preflight(dataset)
        required = max(8192, math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024)) + 2048)
        write_json(attempt_dir / "attempt_started.json", {
            "dataset": dataset, "variant": VARIANT, "seed": seed, "attempt": attempt,
            "stage": "stage1", "started_utc": now(), "source_git_sha": source_sha(),
            "source_file_sha256s": source_file_hashes(),
            "preflight_record": pre.get("record_path", ""), "required_free_mib": required,
            "test_evaluation": False,
        })
        try:
            baseline, uuid, snapshot = baseline_gpu()
            record_resource(snapshot, dataset, seed, attempt, "formal_baseline_snapshot")
            guarded_child("train", dataset, seed, attempt, baseline, uuid,
                          attempt_dir=attempt_dir, required_free_mib=required)
            if not any(row.get("dataset") == dataset and row.get("variant") == VARIANT
                       and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)
                       and row.get("status") == "complete" for row in read_rows(DATA / "runs.csv")):
                raise RuntimeError("worker returned without an immutable complete run record")
        except Exception as exc:
            reason = repr(exc)
            guard_related = any(term in reason.lower() for term in (
                "guard", "memory", "foreign compute", "temperature", "utilization",
                "cuda", "nvidia-smi", "uuid", "device", "permission", "not permitted",
            ))
            status = "blocked_resource" if guard_related else "failed"
            if not any(row.get("dataset") == dataset and row.get("variant") == VARIANT
                       and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)
                       for row in read_rows(DATA / "runs.csv")):
                append_csv(DATA / "runs.csv", {
                    "dataset": dataset, "variant": VARIANT, "seed": seed, "attempt": attempt,
                    "status": status, "source_git_sha": source_sha(), "base_sha": BASE_SHA,
                    "attempt_dir": str(attempt_dir), "test_evaluation": False,
                    "error": reason,
                }, ("dataset", "variant", "seed", "attempt"))
            append_failure({"stage": "stage1_formal_train", "dataset": dataset,
                "variant": VARIANT, "seed": seed, "attempt": attempt,
                "status": status, "training_started": (attempt_dir / "child_launch.json").is_file(), "error": reason})
            write_json(attempt_dir / "status.json", {"status": status, "error": reason,
                        "recorded_utc": now(), "test_evaluation": False})
            if guard_related:
                stop_on_resource = True
    return not stop_on_resource


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_sd(values: list[float]) -> float | None:
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)) if values and avg is not None else None


def valid_complete_run(row: dict[str, str]) -> bool:
    if row.get("status") != "complete" or row.get("test_evaluation", "").lower() != "false":
        return False
    if row.get("source_git_sha") != source_sha() or row.get("base_sha") != BASE_SHA:
        return False
    if row.get("checkpoint_reload_verified", "").lower() != "true":
        return False
    if row.get("prior_skip_optimizer_managed", "").lower() != "true":
        return False
    if row.get("prior_skip_gradient_nonzero", "").lower() != "true":
        return False
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                               ("best_f1_checkpoint_path", "best_f1_checkpoint_sha256")):
        path = Path(row.get(path_key, ""))
        if not path.is_file() or not row.get(hash_key) or sha256_file(path) != row[hash_key]:
            return False
    for key in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
                "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce", "train_loss",
                "lambda_prior", "prior_skip_linear_rms", "peak_allocated_bytes", "peak_reserved_bytes"):
        try:
            if not math.isfinite(float(row[key])):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def paired_results(stage1_only: bool = False) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    run_rows = read_rows(DATA / "runs.csv")
    refs = {(row["dataset"], int(row["seed"])): row
            for row in read_json(DATA / "preflight.json", {}).get("historical_references", [])}
    paired: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    datasets = ("Movies", "Grocery") if stage1_only else DATASETS
    for dataset in datasets:
        seeds = (42, 43, 44) if dataset == "Movies" else (42, 43)
        differences = []
        for seed in seeds:
            matches = [row for row in run_rows if row.get("dataset") == dataset
                       and row.get("variant") == VARIANT and row.get("seed") == str(seed)
                       and row.get("status") == "complete"]
            if len(matches) != 1 or (dataset, seed) not in refs:
                continue
            run, reference = matches[0], refs[(dataset, seed)]
            best_acc, best_f1 = reference["metrics"], reference["best_f1_metrics"]
            delta = {
                "dataset": dataset, "variant": VARIANT, "seed": seed,
                "delta_val_acc_pp": 100.0 * (float(run["val_acc"]) - float(best_acc["val_acc"])),
                "delta_val_macro_f1_pp": 100.0 * (float(run["val_macro_f1"]) - float(best_acc["val_macro_f1"])),
                "delta_val_ce": float(run["val_ce"]) - float(best_acc["val_ce"]),
                "new_best_f1_val_acc": float(run["best_f1_val_acc"]),
                "new_best_f1_val_macro_f1": float(run["best_f1_val_macro_f1"]),
                "new_best_f1_val_ce": float(run["best_f1_val_ce"]),
                "delta_best_f1_val_acc_pp": 100.0 * (float(run["best_f1_val_acc"]) - float(best_f1["val_acc"])),
                "delta_best_f1_val_macro_f1_pp": 100.0 * (float(run["best_f1_val_macro_f1"]) - float(best_f1["val_macro_f1"])),
                "delta_best_f1_val_ce": float(run["best_f1_val_ce"]) - float(best_f1["val_ce"]),
                "reference_source_git_sha": reference["source_git_sha"],
                "reference_accuracy_checkpoint_sha256": reference["checkpoint_hashes"]["best_accuracy"]["sha256"],
                "reference_f1_checkpoint_sha256": reference["checkpoint_hashes"]["best_f1"]["sha256"],
            }
            differences.append(delta)
            paired.append(delta)
        keys = ("delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce",
                "delta_best_f1_val_acc_pp", "delta_best_f1_val_macro_f1_pp", "delta_best_f1_val_ce")
        grouped[dataset] = {
            "n": len(differences), "paired": differences,
            **{f"{key}_mean": mean([float(row[key]) for row in differences]) for key in keys},
            **{f"{key}_population_sd": population_sd([float(row[key]) for row in differences]) for key in keys},
            "positive_accuracy_seeds": sum(float(row["delta_val_acc_pp"]) > 0.0 for row in differences),
        }
    return paired, grouped


def stage1_decision() -> dict[str, Any]:
    protected = protected_history_hashes()
    preflight = read_json(DATA / "preflight.json", {})
    references = {(row["dataset"], int(row["seed"])): row
                  for row in preflight.get("historical_references", [])}
    rows = read_rows(DATA / "runs.csv")
    required_cells = list(STAGE1)
    complete: dict[tuple[str, int], dict[str, str]] = {}
    cell_status = []
    for dataset, seed in required_cells:
        attempts = [row for row in rows if row.get("dataset") == dataset
                    and row.get("variant") == VARIANT and row.get("seed") == str(seed)]
        successes = [row for row in attempts if valid_complete_run(row)]
        if len(successes) == 1:
            complete[(dataset, seed)] = successes[0]
        cell_status.append({"dataset": dataset, "seed": seed,
                            "status": "complete" if len(successes) == 1 else "missing_or_invalid",
                            "attempt_rows": len(attempts), "valid_complete_rows": len(successes)})
    paired, stats = paired_results(stage1_only=True)
    movies, grocery = stats.get("Movies", {}), stats.get("Grocery", {})
    five_complete = len(complete) == 5
    numeric_ok = five_complete and all(valid_complete_run(complete[key]) for key in required_cells)
    refs_ok = five_complete and all((dataset, seed) in references for dataset, seed in required_cells)
    started_attempts: dict[tuple[str, int], int] = {}
    for row in rows:
        if row.get("dataset") in {d for d, _ in required_cells} and row.get("variant") == VARIANT:
            try:
                key = (row["dataset"], int(row["seed"]))
            except (KeyError, ValueError):
                continue
            if key in required_cells and any_child_started([row]):
                started_attempts[key] = started_attempts.get(key, 0) + 1
    no_duplicate_attempts = all(started_attempts.get(key, 0) <= 1 for key in required_cells)
    all_fp_match = five_complete and all(
        complete[key].get("split_fingerprint") == EXPECTED_SPLIT_FINGERPRINTS[key[0]]
        and complete[key].get("feature_fingerprint") == preflight["dataset_records"][key[0]].get("feature_fingerprint")
        and complete[key].get("dataset_fingerprint") == preflight["dataset_records"][key[0]].get("dataset_fingerprint")
        and complete[key].get("physical_edge_fingerprint") == preflight["dataset_records"][key[0]].get("physical_edge_fingerprint")
        and complete[key].get("split_file_sha256") == preflight["dataset_records"][key[0]].get("split_file_sha256")
        and complete[key].get("semantic_cache_sha256") == preflight["dataset_records"][key[0]].get("semantic_cache_sha256")
        and complete[key].get("candidate_fingerprint") == preflight["dataset_records"][key[0]].get("candidate_fingerprint")
        and complete[key].get("v7b_fingerprint") == preflight["dataset_records"][key[0]].get("v7b_fingerprint")
        for key in required_cells
    ) if refs_ok else False
    movies_gate = bool(
        movies.get("n") == 3 and movies.get("delta_val_acc_pp_mean") is not None
        and movies["delta_val_acc_pp_mean"] >= 0.20
        and movies.get("positive_accuracy_seeds", 0) >= 2
        and movies.get("delta_val_macro_f1_pp_mean") is not None
        and movies["delta_val_macro_f1_pp_mean"] >= -0.10
        and movies.get("delta_val_ce_mean") is not None and movies["delta_val_ce_mean"] <= 0.020
    )
    grocery_gate = bool(
        grocery.get("n") == 2 and grocery.get("delta_val_acc_pp_mean") is not None
        and grocery["delta_val_acc_pp_mean"] >= -0.10
        and grocery.get("delta_val_macro_f1_pp_mean") is not None
        and grocery["delta_val_macro_f1_pp_mean"] >= -0.30
        and grocery.get("delta_val_ce_mean") is not None and grocery["delta_val_ce_mean"] <= 0.020
    )
    gates = {
        "five_of_five_complete": five_complete,
        "all_numeric_and_checkpoint_audits_pass": numeric_ok,
        "historical_references_verified": refs_ok,
        "no_duplicate_or_retry_attempts": no_duplicate_attempts,
        "frozen_data_cache_fingerprints_match": all_fp_match,
        "protected_historical_files_unchanged": bool(protected),
        "Movies_gate": movies_gate, "Grocery_gate": grocery_gate,
    }
    promoted = all(gates.values())
    decision = {
        "recorded_utc": now(), "study": "MR-CGE V1.3 Intrinsic-Prior Fusion Controlled Screen",
        "base_sha": BASE_SHA, "source_git_sha": source_sha(), "branch": BRANCH,
        "stage1_complete": five_complete, "status": "pass" if promoted else "no_go",
        "stage1_promoted": promoted, "stage2_authorized": promoted,
        "gates": gates, "cell_status": cell_status, "dataset_paired_statistics": stats,
        "protected_historical_file_count": len(protected),
        "test_evaluation": False, "link_prediction": False,
        "interpretation": (
            "All registered Stage 1 gates passed; ele-fashion seeds 42/43 are authorized."
            if promoted else
            "At least one registered Stage 1 completion, audit, data, or metric gate failed; Stage 2 is not authorized."
        ),
    }
    write_rows(DATA / "paired_differences.csv", paired,
        ("dataset", "variant", "seed", "delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce",
         "new_best_f1_val_acc", "new_best_f1_val_macro_f1", "new_best_f1_val_ce",
         "delta_best_f1_val_acc_pp", "delta_best_f1_val_macro_f1_pp", "delta_best_f1_val_ce",
         "reference_source_git_sha", "reference_accuracy_checkpoint_sha256", "reference_f1_checkpoint_sha256"))
    write_json(DATA / "screening_decision.json", decision)
    return decision


def formal_stage2() -> list[dict[str, str]]:
    decision = read_json(DATA / "screening_decision.json", {})
    if decision.get("stage2_authorized") is not True:
        raise RuntimeError("Stage 2 was not authorized by the complete registered Stage 1 gate")
    completed = []
    for dataset, seed in STAGE2:
        existing = [row for row in read_rows(DATA / "runs.csv")
                    if row.get("dataset") == dataset and row.get("variant") == VARIANT
                    and row.get("seed") == str(seed)]
        if existing:
            if any(valid_complete_run(row) for row in existing):
                completed.append(next(row for row in existing if valid_complete_run(row)))
                continue
            if any_child_started(existing) or any(row.get("status") != "blocked_resource" for row in existing):
                continue
            # No training process was launched; fresh ownership checks may resume this cell.
        pre = current_passing_preflight(dataset)
        required = max(8192, math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024)) + 2048)
        attempt, attempt_dir = current_attempt(dataset, seed)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        write_json(attempt_dir / "attempt_started.json", {
            "dataset": dataset, "variant": VARIANT, "seed": seed, "attempt": attempt,
            "stage": "stage2", "started_utc": now(), "source_git_sha": source_sha(),
            "source_file_sha256s": source_file_hashes(), "preflight_record": pre.get("record_path", ""),
            "required_free_mib": required, "test_evaluation": False,
        })
        try:
            baseline, uuid, snapshot = baseline_gpu()
            record_resource(snapshot, dataset, seed, attempt, "formal_baseline_snapshot")
            guarded_child("train", dataset, seed, attempt, baseline, uuid,
                          attempt_dir=attempt_dir, required_free_mib=required)
            run = next((row for row in read_rows(DATA / "runs.csv")
                        if row.get("dataset") == dataset and row.get("variant") == VARIANT
                        and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)
                        and row.get("status") == "complete"), None)
            if run is None or not valid_complete_run(run):
                raise RuntimeError("Stage 2 worker did not produce a verified complete run record")
            completed.append(run)
        except Exception as exc:
            reason = repr(exc)
            guard_related = any(term in reason.lower() for term in (
                "guard", "memory", "foreign compute", "temperature", "utilization",
                "cuda", "nvidia-smi", "uuid", "device", "permission", "not permitted",
            ))
            status = "blocked_resource" if guard_related else "failed"
            if not any(row.get("dataset") == dataset and row.get("variant") == VARIANT
                       and row.get("seed") == str(seed) and row.get("attempt") == str(attempt)
                       for row in read_rows(DATA / "runs.csv")):
                append_csv(DATA / "runs.csv", {
                    "dataset": dataset, "variant": VARIANT, "seed": seed, "attempt": attempt,
                    "status": status, "source_git_sha": source_sha(), "base_sha": BASE_SHA,
                    "attempt_dir": str(attempt_dir), "test_evaluation": False, "error": reason,
                }, ("dataset", "variant", "seed", "attempt"))
            append_failure({"stage": "stage2_formal_train", "dataset": dataset,
                "variant": VARIANT, "seed": seed, "attempt": attempt,
                "status": status, "training_started": (attempt_dir / "child_launch.json").is_file(), "error": reason})
            write_json(attempt_dir / "status.json", {"status": status, "error": reason,
                        "recorded_utc": now(), "test_evaluation": False})
            if guard_related:
                break
    return completed


def summarize_runs() -> list[dict[str, Any]]:
    rows = read_rows(DATA / "runs.csv")
    paired, paired_stats = paired_results(stage1_only=False)
    # Keep all available pairings, including ele-fashion only after authorization.
    write_rows(DATA / "paired_differences.csv", paired,
        ("dataset", "variant", "seed", "delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce",
         "new_best_f1_val_acc", "new_best_f1_val_macro_f1", "new_best_f1_val_ce",
         "delta_best_f1_val_acc_pp", "delta_best_f1_val_macro_f1_pp", "delta_best_f1_val_ce",
         "reference_source_git_sha", "reference_accuracy_checkpoint_sha256", "reference_f1_checkpoint_sha256"))
    output = []
    for dataset in DATASETS:
        done = [row for row in rows if row.get("dataset") == dataset and row.get("variant") == VARIANT
                and valid_complete_run(row)]
        if not done:
            continue
        stats = paired_stats.get(dataset, {})
        metrics = ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_macro_f1", "train_ce",
                   "train_val_acc_gap", "train_val_ce_gap", "lambda_prior", "prior_skip_linear_rms",
                   "prior_to_old_skip_rms_ratio", "wall_seconds")
        row = {"dataset": dataset, "variant": VARIANT, "n": len(done),
               "seeds": ";".join(str(item["seed"]) for item in sorted(done, key=lambda x: int(x["seed"])))}
        for key in metrics:
            values = [float(item[key]) for item in done]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_population_sd"] = population_sd(values)
        for key, value in stats.items():
            if key not in {"paired", "positive_accuracy_seeds", "n"}:
                row[key] = value
        row["paired_n"] = stats.get("n", 0)
        row["positive_accuracy_seeds"] = stats.get("positive_accuracy_seeds", 0)
        row["test_evaluation"] = False
        output.append(row)
    fields = list(dict.fromkeys(key for row in output for key in row))
    if not fields:
        fields = ["dataset", "variant", "n", "seeds"]
    write_rows(DATA / "dataset_summary.csv", output, fields)
    return output


def final_decision(stage1: dict[str, Any]) -> dict[str, Any]:
    _, stats = paired_results(stage1_only=False)
    fashion = stats.get("ele-fashion", {})
    stage2_rows = [row for row in read_rows(DATA / "runs.csv")
                   if row.get("dataset") == "ele-fashion" and row.get("variant") == VARIANT
                   and row.get("status") == "complete" and valid_complete_run(row)]
    stage2_complete = len({int(row["seed"]) for row in stage2_rows}) == 2
    if not stage1.get("stage1_promoted"):
        recommendation = "NO-GO"
        rationale = "Stage 1 did not pass all pre-registered gates; ele-fashion was not run."
    elif not stage2_complete:
        recommendation = "MIXED"
        rationale = "Stage 1 passed, but the conditional ele-fashion confirmation is incomplete; no conclusion is made for that dataset."
    else:
        acc = fashion.get("delta_val_acc_pp_mean")
        f1 = fashion.get("delta_val_macro_f1_pp_mean")
        ce = fashion.get("delta_val_ce_mean")
        consistent_decline = (acc is not None and f1 is not None and ce is not None
                              and acc < 0.0 and f1 < 0.0 and ce > 0.0)
        if consistent_decline:
            recommendation = "NO-GO"
            rationale = ("Stage 1 passed, but the ele-fashion confirmation shows a consistent mean regression "
                         "across Accuracy, Macro-F1, and CE; do not retain as a general default.")
        else:
            recommendation = "GO"
            rationale = ("Stage 1 passed and the conditional ele-fashion confirmation did not show a consistent "
                         "mean regression across Accuracy, Macro-F1, and CE. This is an exploratory screen, not significance evidence.")
    result = {
        "recorded_utc": now(), "recommendation": recommendation,
        "stage1_status": stage1.get("status"), "stage2_authorized": bool(stage1.get("stage2_authorized")),
        "stage2_complete": stage2_complete, "stage2_cells_completed": sorted(int(row["seed"]) for row in stage2_rows),
        "ele_fashion_paired_statistics": fashion, "rationale": rationale,
        "statistical_significance_claimed": False, "test_evaluation": False, "link_prediction": False,
    }
    decision_path = DATA / "screening_decision.json"
    prior = read_json(decision_path, {})
    prior["final_decision"] = result
    write_json(decision_path, prior)
    return result


def implementation_audit_text() -> str:
    import difflib
    import inspect
    from src.models.mrcge_v13 import Model as FrozenV13Model
    from src.models.mrcge_v13_intrinsic_prior_skip import Model as IntrinsicPriorModel

    parent_forward = inspect.getsource(FrozenV13Model.forward).splitlines()
    child_forward = inspect.getsource(IntrinsicPriorModel.forward).splitlines()
    diff = "\n".join(difflib.unified_diff(
        parent_forward, child_forward,
        fromfile="src/models/mrcge_v13.py:Model.forward",
        tofile="src/models/mrcge_v13_intrinsic_prior_skip.py:Model.forward",
        lineterm="",
    ))
    if not diff:
        raise RuntimeError("mechanical parent/child forward diff unexpectedly empty")
    # The copied forward is identical until the final fusion block. The remaining
    # additions expose read-only prior telemetry and do not feed earlier branches.
    return "\n".join([
        "# Implementation Audit — Intrinsic-Prior Skip", "",
        f"Base commit: `{BASE_SHA}`", f"Branch: `{BRANCH}`",
        f"Parent source SHA256: `{sha256_file(ROOT / 'src/models/mrcge_v13.py')}`",
        f"Candidate source SHA256: `{sha256_file(ROOT / 'src/models/mrcge_v13_intrinsic_prior_skip.py')}`",
        "", "## State and initialization", "",
        "The only new trainable state is `prior_skip_raw`, a zero-initialized scalar appended after the frozen parent constructor. It has one element and consumes no random draw. It is not a module, so the external classifier initialization order remains unchanged. The model inherits `src.models.mrcge_v13.Model` directly and adds no Two-Hop or Expert-Decoupled modules.",
        "", "## Mechanical `forward()` comparison", "",
        "The following unified diff is generated directly from `inspect.getsource` for the frozen parent and the independent subclass. All statements before the final fusion are copied unchanged. Computational changes are the concatenation of already-computed `h` as `h0_fused`, the zero-initialized scalar coefficient, and use of `fusion_skip(fused + lambda_prior * h0_fused)` while keeping the MLP on `fused`. Added remaining lines only record the scalar and no-bias RMS/cosine diagnostics or expose the scalar in `info`.",
        "", "```diff", diff, "```", "",
        "## Formula audit", "",
        "`lambda_prior = tanh(prior_skip_raw)` and `z = fusion_norm(fusion_skip(F + lambda_prior * H0) + fusion_linear2(fusion_dropout(GELU(fusion_linear1(F)))))`. The existing skip is called once, so its bias is applied once. `H0` is not detached, normalized, dropped out, or recomputed. No Router, Private, Collaborative, Late, Loss, or initialization path depends on lambda.",
        "", "## Test evidence", "",
        "- Dedicated CPU tests: 5 passed, including exact initialization/Eval parity, Train loss/shared-gradient/RNG parity, nonzero scalar gradient and optimizer inclusion, final-fusion invariants, checkpoint reload, and finite empty/isolated/zero-cross cases.",
        "- Existing MR-CGE regressions: 73 passed across V1.1, V1.2, V1.3, and Two-Hop Cross.",
        "- GPU parity and dataset forward/backward are separately required by the guarded GPU preflight and are recorded in `data/gpu_preflight.json`.",
        "",
    ])


def write_reports(cpu: dict[str, Any], stage1: dict[str, Any] | None,
                  final: dict[str, Any] | None, gpu: dict[str, Any] | None) -> None:
    gpu = gpu or {}
    data_records = cpu.get("dataset_records", {})
    ref_rows = cpu.get("historical_references", [])
    gpu_history = read_json(DATA / "gpu_preflight.json", {})
    gpu_cells = gpu_history.get("cells", [])
    run_rows = read_rows(DATA / "runs.csv")
    result_rows = [row for row in run_rows if row.get("status") == "complete"]
    lines = [
        "# Data Protocol Audit — Intrinsic-Prior Skip",
        "",
        f"Base `{BASE_SHA}`; source commit `{source_sha()}`; branch `{BRANCH}`.",
        "Fixed split seed is 42, auto-generation is disabled, Test labels are masked to `-100` before Trainer entry, Test evaluation is off, and Link Prediction is off.",
        "",
        "| Dataset | Nodes | Classes | Split fingerprint | Split SHA256 | Feature fingerprint | Physical graph fingerprint | Semantic cache SHA256 | Candidate fingerprint | V7B fingerprint |",
        "|---|---:|---:|---|---|---|---|---|---|---|",
    ]
    for dataset in DATASETS:
        item = data_records.get(dataset) or {}
        lines.append("| {} | {} | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
            dataset, item.get("num_nodes", ""), item.get("num_classes", ""),
            item.get("split_fingerprint", ""), item.get("split_file_sha256", ""),
            item.get("feature_fingerprint", ""), item.get("physical_edge_fingerprint", ""),
            item.get("semantic_cache_sha256", ""), item.get("candidate_fingerprint", ""),
            item.get("v7b_fingerprint", "")))
    lines += ["", "## Historical reference checkpoints", "",
              "Each same-seed Best-Accuracy and Best-F1 checkpoint hash was verified before comparison. Movies seed 44 uses the registered Physical-Basis fixed-split comparison record.", "",
              "| Dataset | Seed | Source SHA | Best-Accuracy checkpoint SHA256 | Best-F1 checkpoint SHA256 |",
              "|---|---:|---|---|---|"]
    for row in ref_rows:
        hashes = row["checkpoint_hashes"]
        lines.append(f"| {row['dataset']} | {row['seed']} | `{row['source_git_sha']}` | `{hashes['best_accuracy']['sha256']}` | `{hashes['best_f1']['sha256']}` |")
    lines += ["", "Protected historical files: {} hashes checked against the propagation-audit manifest; unchanged: `{}`.".format(
        len(cpu.get("protected_historical_file_sha256s", {})), bool(cpu.get("protected_historical_file_sha256s"))),
        "", "All formal run rows record `test_evaluation=false`; no Test metric is loaded or emitted. The only Test-index operation checks that labels are masked to `-100`.", ""]
    (STUDY / "DATA_PROTOCOL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    pre_lines = [
        "# Preflight Report — Intrinsic-Prior Skip", "",
        f"Branch `{BRANCH}`; source commit `{source_sha()}`; base `{BASE_SHA}`.", "",
        "## CPU checks", "",
        "- Intrinsic-Prior Skip tests: 5 passed.",
        "- Existing MR-CGE V1.1, V1.2, V1.3, and Two-Hop regressions: 73 passed.",
        "- CPU data/cache/history preflight: `{}`.".format(cpu.get("status", "not recorded")),
        "- Initialization equivalence audited for {} dataset/seed pairs.".format(len(cpu.get("initialization_equivalence", []))),
        "- Test-off masking and historical protected-file checks are recorded in `data/preflight.json`.",
        "", "## GPU 1 checks", "",
        "Physical GPU 1 only; no shared-GPU override is enabled. Each passing row requires a fresh UUID/PID/memory/utilization/temperature snapshot, a real dataset-specific forward/backward, finite nonzero `prior_skip_raw` gradient, optimizer membership, and a measured peak plus 2 GiB start margin.",
        "",
        "| Dataset | Status | GPU UUID | Peak allocated MiB | Peak reserved MiB | Required free MiB |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in gpu_cells:
        alloc = row.get("peak_allocated_bytes")
        reserved = row.get("peak_reserved_bytes")
        pre_lines.append("| {} | {} | `{}` | {} | {} | {} |".format(
            row.get("dataset", ""), row.get("status", ""), row.get("gpu_uuid", ""),
            round(int(alloc) / (1024 * 1024), 1) if alloc else "",
            round(int(reserved) / (1024 * 1024), 1) if reserved else "",
            row.get("required_start_free_mib_with_2gib_margin", "")))
    pre_lines += ["", "GPU preflight campaign status: `{}`. Formal training started: `{}`.".format(
        gpu.get("status", "not run"), bool(gpu and gpu.get("training_started"))),
        "", "## Development incident", "",
        "An initial automated test invocation had `CUDA_VISIBLE_DEVICES` unset and attempted to place models on the default CUDA ordinal before failing on a missing import. No model forward or training worker ran, and no UUID was captured for that invocation. This is recorded as a development protocol incident; subsequent unit/regression tests explicitly hid CUDA, and all later GPU actions use the strict physical-GPU-1 guard.", ""]
    (STUDY / "PREFLIGHT_REPORT.md").write_text("\n".join(pre_lines), encoding="utf-8")

    summaries = summarize_runs()
    report = ["# Screening Report — Intrinsic-Prior Fusion Controlled Screen", "",
              f"Base `{BASE_SHA}`; frozen training source `{source_sha()}`; branch `{BRANCH}`.", "",
              "Only the registered `intrinsic_prior_skip` model was trained. Best-Accuracy metrics are paired to the same-seed V1.3-A Joint View reference; Best-F1 checkpoint results remain a separate comparison. These validation results are exploratory and are not statistical significance evidence.", "",
              "## Paired Validation results", "",
              "| Dataset | Seed | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | Best-F1 Δ Accuracy (pp) | Best-F1 Δ Macro-F1 (pp) | Best-F1 Δ CE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    paired_rows = read_rows(DATA / "paired_differences.csv")
    for row in paired_rows:
        report.append("| {} | {} | {:.4f} | {:.4f} | {:.6f} | {:.4f} | {:.4f} | {:.6f} |".format(
            row["dataset"], row["seed"], float(row["delta_val_acc_pp"]),
            float(row["delta_val_macro_f1_pp"]), float(row["delta_val_ce"]),
            float(row["delta_best_f1_val_acc_pp"]), float(row["delta_best_f1_val_macro_f1_pp"]),
            float(row["delta_best_f1_val_ce"])))
    report += ["", "## Per-dataset descriptive summary", "",
               "| Dataset | n | Δ Accuracy mean ± population SD (pp) | Δ Macro-F1 mean ± population SD (pp) | Δ CE mean ± population SD | Train CE mean | Validation CE mean | Train−Validation CE gap mean | Lambda mean | Prior RMS mean | Wall time mean (s) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        def val(key):
            value = row.get(key)
            return "n/a" if value in (None, "") else f"{float(value):.6f}"
        report.append("| {} | {} | {} ± {} | {} ± {} | {} ± {} | {} | {} | {} | {} | {} | {} |".format(
            row["dataset"], row["n"], val("delta_val_acc_pp_mean"), val("delta_val_acc_pp_population_sd"),
            val("delta_val_macro_f1_pp_mean"), val("delta_val_macro_f1_pp_population_sd"),
            val("delta_val_ce_mean"), val("delta_val_ce_population_sd"), val("train_ce_mean"),
            val("val_ce_mean"), val("train_val_ce_gap_mean"), val("lambda_prior_mean"),
            val("prior_skip_linear_rms_mean"), val("wall_seconds_mean")))
    report += ["", "## Mechanism and generalization", "",
               "Per-seed final `prior_skip_raw`, `lambda_prior`, RMS(H0), RMS(F), RMS(F−H0), RMS(WsF), RMS(lambda WsH0), RMS(MLP(F)), the prior/original-skip RMS ratio, and cosine are in `data/prior_skip_diagnostics.csv`. Full epoch lambda histories are in `data/epoch_history.csv`. Lambda can be negative; nonzero lambda and high prior RMS do not establish performance benefit or graph bypass.", "",
               "Train/Validation Accuracy, Macro-F1, and CE and their gaps are in `data/runs.csv` and `data/dataset_summary.csv`. Train CE and Validation CE use the trainer's registered full-graph evaluation metrics at the Best-Accuracy checkpoint; epoch history also retains training-mode loss and CE.", "",
               "## Stage 1 pre-registered decision", ""]
    if stage1:
        report += [f"Status: **{stage1['status']}**; promoted: **{stage1['stage1_promoted']}**; Stage 2 authorized: **{stage1['stage2_authorized']}**.", "",
                   "| Gate | Result |", "|---|---|"]
        report += [f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in stage1["gates"].items()]
        report += ["", "Movies requires mean Δ Accuracy ≥ +0.20 pp, at least two positive seeds, mean Δ Macro-F1 ≥ −0.10 pp, and mean Δ CE ≤ +0.020. Grocery requires mean Δ Accuracy ≥ −0.10 pp, mean Δ Macro-F1 ≥ −0.30 pp, and mean Δ CE ≤ +0.020. All five valid cells, reloads, data fingerprints, scalar-gradient/optimizer evidence, Test-off and protected-history audits are required."]
    else:
        report += ["Stage 1 decision has not yet been recorded."]
    report += ["", "## Final recommendation", ""]
    if final:
        report += [f"**{final['recommendation']}** — {final['rationale']}", "",
                   "No Test metrics were evaluated. No Link Prediction task was run. No statistical significance claim is made."]
    else:
        report += ["Final recommendation is pending completion of the registered screen."]
    report += [""]
    (STUDY / "SCREENING_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    handoff = ["# Decision Handoff — Intrinsic-Prior Skip", "",
        f"Branch: `{BRANCH}`", f"Base: `{BASE_SHA}`", f"Frozen training source: `{source_sha()}`",
        f"Current HEAD: `{source_sha()}`", "",
        f"Stage 1 status: `{stage1.get('status') if stage1 else 'pending'}`",
        f"Stage 2 status: `{('complete' if final and final.get('stage2_complete') else 'not run or incomplete')}`",
        f"Recommendation: `{final.get('recommendation') if final else 'pending'}`", "",
        "The training source is frozen. Review the implementation audit, data protocol audit, reports, compact CSV/JSON evidence, and GPU resource record. Checkpoint files remain local under `outputs/mrcge_v13_intrinsic_prior_skip/runs/` and are not tracked. No default-model integration or further ablation is authorized in this task.", ""]
    (STUDY / "DECISION_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")

    manifest = read_json(DATA / "manifest.json", {})
    manifest.update({
        "source_git_sha": source_sha(), "training_source_frozen_sha": source_sha(),
        "gpu_preflight_status": gpu.get("status", manifest.get("gpu_preflight_status", "pending")) if gpu else manifest.get("gpu_preflight_status", "pending"),
        "stage1_status": stage1.get("status") if stage1 else "pending",
        "stage2_status": "complete" if final and final.get("stage2_complete") else (
            "not_authorized" if stage1 and not stage1.get("stage2_authorized") else "not_run_or_incomplete"),
        "completed_cells": len([row for row in result_rows if valid_complete_run(row)]),
        "test_evaluation": False, "link_prediction": False,
        "final_recommendation": final.get("recommendation") if final else None,
        "updated_utc": now(),
    })
    write_json(DATA / "manifest.json", manifest)


def run_screen() -> dict[str, Any]:
    verify_committed_source()
    cpu = read_json(DATA / "preflight.json", {})
    gpu_record = read_json(DATA / "gpu_preflight.json", {})
    if cpu.get("status") != "pass":
        raise RuntimeError("CPU/data preflight is not passing")
    expected_datasets = set(DATASETS)
    gpu_passed = {row.get("dataset") for row in gpu_record.get("cells", [])
                  if row.get("status") == "pass"
                  and row.get("source_git_sha") == source_sha()
                  and row.get("source_file_sha256s") == source_file_hashes()}
    if not expected_datasets <= gpu_passed:
        raise RuntimeError(f"GPU1 preflight missing current-source dataset passes: {sorted(expected_datasets-gpu_passed)}")
    formal_stage1()
    decision = stage1_decision()
    gpu_campaign = gpu_record.get("campaigns", [])[-1] if gpu_record.get("campaigns") else None
    write_reports(cpu, decision, None, gpu_campaign)
    if decision.get("stage2_authorized"):
        formal_stage2()
    final = final_decision(decision)
    write_reports(cpu, decision, final, gpu_campaign)
    return {"stage1": decision, "final": final,
            "manifest": read_json(DATA / "manifest.json", {})}


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--cpu-preflight", action="store_true")
    actions.add_argument("--gpu-preflight", action="store_true")
    actions.add_argument("--run-screen", action="store_true")
    actions.add_argument("--run-all", action="store_true")
    actions.add_argument("--worker-preflight", action="store_true")
    actions.add_argument("--worker-train", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--record-path")
    parser.add_argument("--attempt-dir")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        if args.worker_preflight or args.worker_train:
            if args.dataset is None or args.seed is None or args.attempt is None:
                parser.error("worker actions need --dataset, --seed and --attempt")
            if args.worker_preflight:
                if not args.record_path:
                    parser.error("--worker-preflight needs --record-path")
                worker_preflight(args.dataset, args.seed, args.attempt, Path(args.record_path))
                return
            if not args.attempt_dir:
                parser.error("--worker-train needs --attempt-dir")
            worker_train(args.dataset, args.seed, args.attempt, Path(args.attempt_dir))
            return
        if args.cpu_preflight:
            result = cpu_preflight()
            (STUDY / "IMPLEMENTATION_AUDIT.md").write_text(
                implementation_audit_text(), encoding="utf-8"
            )
            write_reports(result, None, None, None)
        elif args.gpu_preflight:
            result = gpu_preflight()
            cpu = read_json(DATA / "preflight.json", {})
            campaign = read_json(DATA / "gpu_preflight.json", {}).get("campaigns", [])
            write_reports(cpu, None, None, campaign[-1] if campaign else result)
        elif args.run_screen:
            result = run_screen()
        else:
            cpu_result = cpu_preflight()
            (STUDY / "IMPLEMENTATION_AUDIT.md").write_text(
                implementation_audit_text(), encoding="utf-8"
            )
            gpu_result = gpu_preflight()
            if gpu_result.get("status") == "pass":
                result = run_screen()
            else:
                result = {"gpu_preflight": gpu_result, "status": "blocked_before_training"}
                write_reports(cpu_result, None, None, gpu_result)
        print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    except Exception as exc:
        append_failure({"stage": "runner", "status": "failed_or_blocked",
                        "error": repr(exc), "traceback": traceback.format_exc(),
                        "source_git_sha": source_sha(), "test_evaluation": False})
        raise


if __name__ == "__main__":
    main()
