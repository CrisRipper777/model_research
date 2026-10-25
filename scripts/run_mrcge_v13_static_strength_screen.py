#!/usr/bin/env python3
"""Guarded MR-CGE V1.3 Static-Strength Validation screen.

All outputs are isolated under research/mrcge_v13_static_strength and
outputs/mrcge_v13_static_strength. Historical V1.2, V1.3-A, and fixed-split
Hybrid records are read-only. GPU work is launched in one-process children
under the physical-GPU ownership guard.
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
from scripts import run_mrcge_v13_screen as v13_runner
from scripts import run_mrcge_v12_confirmation as v12_utils
from src.data import load_mag_data
from src.data.loaders import resolve_path
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed


STUDY = ROOT / "research/mrcge_v13_static_strength"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_static_strength"
RUNS = OUTPUT / "runs"
BASE_SHA = "6d94c14d67e511aec34ff5fdcad3ce1a1cfd0701"
BASE_BRANCH = "exp/mrcge_v13_physical_basis_screen"
BRANCH = "exp/mrcge_v13_static_strength_screen"
PHYSICAL_GPU = 1
PROTOCOL = "unified_full_graph_nc_v1"
SPLIT_SEED = 42
EXPECTED_MOVIES_SPLIT = "2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d"
DATASETS = ("Movies", "Grocery", "ele-fashion")
V12_SELECTED = ROOT / "research/mrcge_v12_confirmation/data/combined_selected_runs.csv"
HYBRID_MODEL = "hybrid_cge_paper_v1"
V13_MODEL = "mrcge_v13"
STATIC_MODEL = "mrcge_v13_static_strength"
STATIC_CONDITION = "static_strength"

FROZEN_PATHS = (
    "src/models/mrcge_v13.py",
    "src/models/mrcge_v13_basis_screen.py",
    "src/models/mrcge_v12.py",
    "src/models/hybrid_cge_v11.py",
    "src/models/hybrid_cge_paper_v1.py",
    "src/models/cg_cge_v1.py",
    "src/tasks/nc.py",
    "configs/model/mrcge_v13.yaml",
    "configs/model/mrcge_v13_basis_screen.yaml",
    "configs/model/mrcge_v12.yaml",
    "configs/model/hybrid_cge_paper_v1.yaml",
    "scripts/run_mrcge_v13_screen.py",
    "scripts/run_mrcge_v13_physical_basis_screen.py",
    "scripts/run_mrcge_v12_confirmation.py",
    "scripts/run_hybrid_cge_paper_evidence_v1.py",
    "tests/test_mrcge_v13.py",
    "tests/test_mrcge_v13_physical_basis_screen.py",
)

TARGET_CELLS = (
    (STATIC_MODEL, STATIC_CONDITION, "Movies", 42),
    (STATIC_MODEL, STATIC_CONDITION, "Movies", 43),
    (STATIC_MODEL, STATIC_CONDITION, "Movies", 44),
    (STATIC_MODEL, STATIC_CONDITION, "Grocery", 42),
    (STATIC_MODEL, STATIC_CONDITION, "Grocery", 43),
    (STATIC_MODEL, STATIC_CONDITION, "ele-fashion", 42),
    (STATIC_MODEL, STATIC_CONDITION, "ele-fashion", 43),
)

RUN_FIELDS = (
    "model", "condition", "dataset", "seed", "attempt", "status", "source_git_sha",
    "model_source_sha256", "trainer_source_sha256", "config_sha256", "feature_fingerprint",
    "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256",
    "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint", "initial_model_sha256",
    "initial_classifier_sha256", "best_acc_epoch", "best_f1_epoch", "epochs_completed",
    "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
    "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce", "train_val_acc_gap",
    "train_val_macro_f1_gap", "train_val_ce_gap", "model_parameters", "classifier_parameters",
    "total_parameters", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "gpu_name",
    "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path",
    "best_f1_checkpoint_sha256", "attempt_dir", "config_snapshot_path", "run_metrics_path",
    "validation_rng_unchanged", "checkpoint_reload_verified", "test_evaluation", "error",
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path, default: Any):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_csv(path: Path, row: dict[str, Any], fields: tuple[str, ...], key_fields: tuple[str, ...]) -> None:
    rows = read_rows(path)
    key = tuple(str(row.get(name, "")) for name in key_fields)
    if any(tuple(old.get(name, "") for name in key_fields) == key for old in rows):
        raise RuntimeError(f"refusing to overwrite immutable attempt row {key}")
    all_fields = list(dict.fromkeys([*fields, *(name for old in rows for name in old), *row.keys()]))
    write_rows(path, [*rows, row], all_fields)


def append_failure(record: dict[str, Any]) -> None:
    path = DATA / "failures.json"
    payload = read_json(path, {"failures": []})
    payload.setdefault("failures", []).append({"time_utc": now(), "test_evaluation": False, **record})
    json_write(path, payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def git_file_sha(commit: str, relative: str) -> str:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                            check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def source_file_hashes() -> dict[str, str]:
    paths = [*FROZEN_PATHS,
             "src/models/mrcge_v13_static_strength.py",
             "scripts/run_mrcge_v13_static_strength_screen.py",
             "scripts/analyze_mrcge_v13_static_strength_screen.py",
             "configs/model/mrcge_v13_static_strength.yaml",
             "configs/study/mrcge_v13_static_strength_screen.yaml",
             "src/data/loaders.py", "src/data/splits.py", "src/data/graph_utils.py",
             "src/data/semantic_candidates_v7b.py", "src/data/types.py",
             "src/utils/seeds.py", "src/tasks/inference.py", "src/models/__init__.py",
             "configs/config.yaml", "configs/task/nc.yaml",
             "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml",
             "configs/dataset/ele-fashion.yaml", "tests/test_mrcge_v13_static_strength_screen.py"]
    return {relative: sha256_file(ROOT / relative) for relative in paths}


def verify_frozen_sources() -> dict[str, str]:
    actual = {relative: sha256_file(ROOT / relative) for relative in FROZEN_PATHS}
    base = {relative: git_file_sha(BASE_SHA, relative) for relative in FROZEN_PATHS}
    mismatches = {name: {"base": base[name], "worktree": actual[name]}
                  for name in FROZEN_PATHS if base[name] != actual[name]}
    if mismatches:
        raise RuntimeError(f"frozen source files differ from base commit: {mismatches}")
    return actual


def target_cells(*, models=None, conditions=None, datasets=None, seeds=None):
    cells = list(TARGET_CELLS)
    for name, selected in (("model", models), ("condition", conditions),
                           ("dataset", datasets), ("seed", seeds)):
        if selected is None:
            continue
        selected = {int(value) for value in selected} if name == "seed" else set(selected)
        cells = [cell for cell in cells if cell[{"model": 0, "condition": 1, "dataset": 2, "seed": 3}[name]] in selected]
    if not cells:
        raise ValueError("selection contains no authorized training cells")
    return cells


def cache_entries() -> dict[str, dict[str, Any]]:
    entries = v13_runner.cache_entries()
    missing = set(DATASETS) - set(entries)
    if missing:
        raise RuntimeError(f"required verified V7B.1 caches missing: {sorted(missing)}")
    return {name: entries[name] for name in DATASETS}


def fixed_split_path(dataset: str) -> Path:
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    frozen = OmegaConf.create({"paths": root.paths, "dataset": ds, "seed": SPLIT_SEED})
    OmegaConf.resolve(frozen)
    for key in ("nc_split_path", "node_split_path"):
        if key in frozen.dataset:
            return resolve_path(frozen.dataset[key])
    raise RuntimeError(f"{dataset}: project dataset config has no node-classification split path")


def build_cfg(model_name: str, condition: str, dataset: str, seed: int,
              device: str, cache: dict[str, Any]):
    if model_name != STATIC_MODEL or condition != STATIC_CONDITION:
        raise ValueError(f"unapproved Static-Strength cell {model_name}/{condition}")
    cfg = v13_runner.make_cfg(dataset, "joint_view", seed, device, cache)
    cfg.model = OmegaConf.load(ROOT / "configs/model/mrcge_v13_static_strength.yaml")
    cfg.model.name = STATIC_MODEL
    cfg.model.static_strength_variant = STATIC_CONDITION
    cfg.model.static_strength_enabled = True
    cfg.model.static_strength_init = 0.0
    cfg.model.semantic_cache_path = cache["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]

    split = fixed_split_path(dataset)
    cfg.dataset.name = dataset
    if "nc_split_path" in cfg.dataset:
        cfg.dataset.nc_split_path = str(split)
    elif "node_split_path" in cfg.dataset:
        cfg.dataset.node_split_path = str(split)
    else:
        raise RuntimeError(f"{dataset}: no supported fixed split config key")
    cfg.dataset.auto_generate_splits = False
    cfg.seed = int(seed)
    cfg.device = str(device)
    cfg.model.name = model_name
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0

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


def load_cell_data(model_name: str, condition: str, dataset: str, seed: int,
                   cache: dict[str, Any], device: str = "cpu"):
    cfg = build_cfg(model_name, condition, dataset, seed, device, cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = v12_utils.mask_test_labels(raw)
    split_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
    split_path = Path(split_text).resolve() if split_text else None
    expected_path = fixed_split_path(dataset)
    if split_path != expected_path or not split_path.is_file():
        raise RuntimeError(f"{dataset}/seed{seed}: loader did not use resolved frozen split {expected_path}")
    if cfg.dataset.auto_generate_splits is not False:
        raise RuntimeError("automatic split generation must be disabled")
    fp = v13_runner.data_utils.dataset_fingerprints(raw)
    split_fingerprint = fp["split_fingerprint"]
    expected = str(OmegaConf.load(ROOT / "configs/study/mrcge_v13_static_strength_screen.yaml").split_fingerprints[dataset])
    if split_fingerprint != expected:
        raise RuntimeError(f"{dataset} split fingerprint differs: {split_fingerprint} != {expected}")
    if bool((safe.y[raw.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
        raise RuntimeError("Test labels must be masked and Test evaluation disabled before Trainer")
    if raw.x.size(0) != raw.num_nodes or safe.y.size(0) != raw.num_nodes:
        raise RuntimeError("feature/label node order does not match graph nodes")
    if not torch.isfinite(raw.x).all():
        raise RuntimeError("input feature tensor contains non-finite values")
    return cfg, raw, safe, fp, split_path


def assert_fingerprints_match_v13(dataset: str, seed: int, fp: dict[str, Any],
                                  split_path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    baseline_seed = seed
    v13_rows = [row for row in read_rows(ROOT / "research/mrcge_v13/data/runs.csv")
                if row.get("dataset") == dataset and row.get("variant") == "joint_view"
                and int(row.get("seed", -1)) == baseline_seed and row.get("status") == "complete"]
    if len(v13_rows) != 1:
        raise RuntimeError(f"expected one frozen V1.3-A reference for {dataset}/seed{baseline_seed}")
    reference = v13_rows[0]
    expected_fields = {
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fp["features"],
        "physical_edge_fingerprint": fp["physical_edge_index"],
        "split_fingerprint": fp["split_fingerprint"],
        "split_file_sha256": sha256_file(split_path),
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
    }
    for field, actual in expected_fields.items():
        if reference.get(field) != actual:
            raise RuntimeError(f"{dataset}/seed{seed} {field} differs from V1.3-A reference")
    for field in ("dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256",
                  "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint"):
        if not reference.get(field):
            raise RuntimeError(f"{dataset}/seed{seed}: V1.3-A reference lacks {field}")
    return {"reference_campaign": "MR-CGE V1.3-A", "reference_seed": baseline_seed,
            "reference_source_git_sha": reference.get("source_git_sha"),
            **expected_fields, "split_file_path": str(split_path)}


def record_resource(snapshot: dict[str, Any], model: str, condition: str, dataset: str,
                    seed: int, attempt: int | None, phase: str,
                    extra: dict[str, Any] | None = None) -> None:
    gpu_index = PHYSICAL_GPU
    device = next(row for row in snapshot["devices"] if row["physical_index"] == gpu_index)
    pids = sorted(row["pid"] for row in snapshot["compute_processes"] if row["gpu_uuid"] == device["uuid"])
    row = {"model": model, "condition": condition, "dataset": dataset, "seed": seed,
           "attempt": attempt, "phase": phase, "captured_utc": snapshot.get("captured_utc"),
           "physical_gpu": gpu_index, "gpu_uuid": device["uuid"], "gpu_name": device["name"],
           "memory_free_mib": device["memory_free_mib"], "memory_used_mib": device["memory_used_mib"],
           "utilization_percent": device["utilization_percent"], "temperature_c": device["temperature_c"],
           "compute_pids": json.dumps(pids), "guard_status": snapshot.get("guard", {}).get("status"),
           "required_free_mib": snapshot.get("guard", {}).get("required_free_mib"),
           "test_evaluation": False}
    row.update(extra or {})
    path = DATA / "resource_summary.csv"
    old = read_rows(path)
    fields = list(dict.fromkeys([*(old[0].keys() if old else ()), *row.keys()]))
    write_rows(path, [*old, row], fields)


def build_model_and_cache(cfg, safe, cache, device):
    model = build_model(cfg, data_info(safe))
    if not bool(getattr(model, "requires_global_semantic_candidates", False)):
        raise RuntimeError("screen model must require the audited global V7B.1 semantic candidates")
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=cfg.dataset.name,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    return model.to(device)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def target_label(cell: tuple[str, str, str, int]) -> dict[str, Any]:
    model, condition, dataset, seed = cell
    return {"model": model, "condition": condition, "dataset": dataset, "seed": seed}


def v13_reference(dataset: str, seed: int) -> dict[str, str]:
    rows = [row for row in read_rows(ROOT / "research/mrcge_v13/data/runs.csv")
            if row.get("dataset") == dataset and row.get("variant") == "joint_view"
            and row.get("seed") == str(seed) and row.get("status") == "complete"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one V1.3-A reference for {dataset}/seed{seed}, found {len(rows)}")
    return rows[0]


def verify_checkpoint_record(row: dict[str, str], *, label: str) -> dict[str, Any]:
    records = []
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                               ("best_f1_checkpoint_path", "best_f1_checkpoint_sha256")):
        path_text, expected = row.get(path_key, ""), row.get(hash_key, "")
        if not path_text or not expected:
            raise RuntimeError(f"{label}: missing {path_key}/{hash_key}")
        path = Path(path_text)
        if not path.is_file():
            raise RuntimeError(f"{label}: checkpoint is unavailable: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label}: checkpoint hash mismatch: {path}")
        records.append({"path": str(path), "sha256": actual})
    return {"best_accuracy": records[0], "best_f1": records[1]}


def historical_references(cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    references: dict[str, Any] = {"v13_a": [], "v12_m_repaired": [], "hybrid_fixed_split": []}
    selected_v12 = read_rows(V12_SELECTED)
    v12_runs = read_rows(ROOT / "research/mrcge_v12_confirmation/data/runs.csv")
    v13_split_rows = []
    for dataset in DATASETS:
        seeds = (42, 43, 44) if dataset == "Movies" else (42, 43)
        for seed in seeds:
            v13 = v13_reference(dataset, seed)
            v13_hashes = verify_checkpoint_record(v13, label=f"V1.3-A {dataset}/seed{seed}")
            expected_split = str(OmegaConf.load(ROOT / "configs/study/mrcge_v13_static_strength_screen.yaml").split_fingerprints[dataset])
            if v13.get("split_fingerprint") != expected_split:
                raise RuntimeError(f"V1.3-A {dataset}/seed{seed} split differs from the frozen study split")
            for field, value in (("feature_fingerprint", cache[dataset]["feature_fingerprint"]),
                                 ("semantic_cache_sha256", sha256_file(cache[dataset]["cache_path_abs"])),
                                 ("candidate_fingerprint", cache[dataset]["candidate_fingerprint"]),
                                 ("v7b_fingerprint", cache[dataset]["v7b_fingerprint"])):
                if v13.get(field) and v13[field] != value:
                    raise RuntimeError(f"V1.3-A {dataset}/seed{seed} {field} does not match the audited cache")
            expected_split_sha = sha256_file(fixed_split_path(dataset))
            if v13.get("split_file_sha256") != expected_split_sha:
                raise RuntimeError(f"V1.3-A {dataset}/seed{seed} split-file SHA256 mismatch")
            references["v13_a"].append({
                "model": V13_MODEL, "condition": "joint_view", "dataset": dataset, "seed": seed,
                "status": "historical_reference", "source_git_sha": v13.get("source_git_sha"),
                "split_fingerprint": v13["split_fingerprint"], "metrics": {key: v13.get(key) for key in (
                    "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
                    "best_f1_val_ce", "train_acc", "train_ce", "best_acc_epoch", "best_f1_epoch",
                    "model_parameters", "total_parameters", "wall_seconds", "peak_reserved_bytes")},
                "checkpoints": v13_hashes,
            })
            v13_split_rows.append(v13)

            eligible = [row for row in selected_v12 if row.get("dataset") == dataset
                        and row.get("condition") == "M_repaired" and row.get("seed") == str(seed)
                        and row.get("status") == "complete"
                        and str(row.get("selected_for_combined", "")).lower() == "true"
                        and str(row.get("eligible_complete", "")).lower() == "true"]
            if len(eligible) != 1:
                raise RuntimeError(f"expected one selected V1.2 M_repaired reference for {dataset}/seed{seed}, got {len(eligible)}")
            v12 = eligible[0]
            if not (str(v12.get("checkpoint_accuracy_verified", "")).lower() == "true"
                    and str(v12.get("checkpoint_best_f1_verified", "")).lower() == "true"):
                raise RuntimeError(f"V1.2 {dataset}/seed{seed} selected record lacks dual checkpoint verification")
            v12_hashes = verify_checkpoint_record(v12, label=f"V1.2 M_repaired {dataset}/seed{seed}")
            if v12.get("split_fingerprint") != expected_split or v12.get("split_file_sha256") != expected_split_sha:
                raise RuntimeError(f"V1.2 {dataset}/seed{seed} split differs from the frozen study split")
            for field, value in (("feature_fingerprint", cache[dataset]["feature_fingerprint"]),
                                 ("semantic_cache_sha256", sha256_file(cache[dataset]["cache_path_abs"])),
                                 ("candidate_fingerprint", cache[dataset]["candidate_fingerprint"]),
                                 ("v7b_fingerprint", cache[dataset]["v7b_fingerprint"])):
                if v12.get(field) and v12[field] != value:
                    raise RuntimeError(f"V1.2 {dataset}/seed{seed} {field} does not match the audited cache")
            references["v12_m_repaired"].append({
                "model": "mrcge_v12", "condition": "M_repaired", "dataset": dataset, "seed": seed,
                "attempt": v12.get("attempt"), "source_git_sha": v12.get("source_git_sha"),
                "split_fingerprint": v12.get("split_fingerprint"),
                "metrics": {key: v12.get(key) for key in ("val_acc", "val_macro_f1", "val_ce",
                    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_ce",
                    "best_acc_epoch", "best_f1_epoch", "model_parameters", "total_parameters", "wall_seconds",
                    "peak_reserved_bytes")}, "checkpoints": v12_hashes,
            })

    split_ref = read_rows(ROOT / "research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv")
    prior_frozen = read_json(ROOT / "research/mrcge_v13_physical_basis/data/frozen_baseline.json", {})
    corrected_42 = prior_frozen.get("historical_references", {}).get("hybrid_corrected_seed42", {})
    for seed in (42, 43, 44):
        row = next((item for item in split_ref if item.get("model") == HYBRID_MODEL
                    and item.get("dataset") == "Movies" and item.get("seed") == str(seed)), None)
        if row is None or row.get("split_fingerprint") != EXPECTED_MOVIES_SPLIT:
            raise RuntimeError(f"missing fixed-split Hybrid H2 3-hop reference for Movies/seed{seed}")
        if seed == 42:
            checkpoint_data = corrected_42.get("checkpoints", {})
            checkpoints = {
                "best_accuracy": checkpoint_data.get("best_accuracy", {}),
                "best_f1": checkpoint_data.get("best_f1", {}),
            }
            source_commit = corrected_42.get("source_git_sha")
            condition = corrected_42.get("condition")
        else:
            checkpoints = {
                "best_accuracy": {"path": row.get("checkpoint_path"), "sha256": row.get("checkpoint_sha256")},
                "best_f1": {"path": row.get("best_f1_checkpoint_path"), "sha256": row.get("best_f1_checkpoint_sha256")},
            }
            source_commit = row.get("source_git_sha")
            condition = row.get("condition")
        checked = {}
        for label, item in checkpoints.items():
            path = Path(item.get("path") or "")
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                raise RuntimeError(f"fixed-split Hybrid Movies/seed{seed} {label} checkpoint failed hash audit")
            checked[label] = {"path": str(path), "sha256": item["sha256"]}
        references["hybrid_fixed_split"].append({
            "model": HYBRID_MODEL, "condition": "H2_physical_3hop", "dataset": "Movies", "seed": seed,
            "status": "historical_reference" if seed == 42 else row.get("status"),
            "source_git_sha": source_commit, "split_fingerprint": EXPECTED_MOVIES_SPLIT,
            "metrics": {key: row.get(key) for key in ("val_acc", "val_macro_f1", "val_ce",
                "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_ce",
                "best_acc_epoch", "best_f1_epoch", "total_parameters", "wall_seconds", "peak_reserved_bytes")},
            "checkpoints": checked,
        })
    return references


def protected_history_file_hashes() -> dict[str, str]:
    paths = []
    for relative in ("research/mrcge_v13", "research/mrcge_v13_physical_basis",
                     "research/mrcge_v12_confirmation", "research/hybrid_cge_paper_evidence_v1",
                     "outputs/mrcge_v12_screen", "outputs/mrcge_v12_confirmation",
                     "outputs/mrcge_v13_screen", "outputs/hybrid_cge_paper_evidence_v1",
                     "outputs/mrcge_v13_physical_basis"):
        directory = ROOT / relative
        if not directory.is_dir():
            raise RuntimeError(f"protected historical research directory is missing: {directory}")
        paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def verify_protected_history_file_hashes(expected: dict[str, str]) -> None:
    actual = protected_history_file_hashes()
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected))
        changed.extend(sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name]))
        raise RuntimeError(f"protected historical research files changed: {changed}")


def cpu_preflight() -> dict[str, Any]:
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT).returncode != 0:
        raise RuntimeError("current branch does not descend from the required Static-Strength base SHA")
    frozen = verify_frozen_sources()
    protected_hashes = protected_history_file_hashes()
    caches = cache_entries()
    study_cfg = OmegaConf.load(ROOT / "configs/study/mrcge_v13_static_strength_screen.yaml")
    dataset_records: dict[str, Any] = {}
    for dataset in DATASETS:
        cfg, raw, safe, fingerprints, split_path = load_cell_data(
            STATIC_MODEL, STATIC_CONDITION, dataset, SPLIT_SEED, caches[dataset], "cpu"
        )
        expected = str(study_cfg.split_fingerprints[dataset])
        if fingerprints["split_fingerprint"] != expected:
            raise RuntimeError(f"{dataset}: loaded split does not match the frozen study fingerprint")
        dataset_records[dataset] = {
            "split_path_resolved_by_project_loader": str(split_path),
            "split_file_sha256": sha256_file(split_path),
            "split_fingerprint": fingerprints["split_fingerprint"],
            "feature_fingerprint": caches[dataset]["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "semantic_cache_path": caches[dataset]["cache_path_abs"],
            "semantic_cache_sha256": sha256_file(caches[dataset]["cache_path_abs"]),
            "candidate_fingerprint": caches[dataset]["candidate_fingerprint"],
            "v7b_fingerprint": caches[dataset]["v7b_fingerprint"],
            "num_nodes": int(raw.num_nodes), "num_edges": int(raw.edge_index.size(1)),
            "num_classes": int(raw.num_classes), "text_dim": int(raw.x_t.size(1)),
            "visual_dim": int(raw.x_i.size(1)), "test_labels_masked_before_trainer": True,
            "test_evaluation": False,
        }
        del cfg, raw, safe

    target_records = []
    initialization_records = []
    for cell in TARGET_CELLS:
        model_name, condition, dataset, seed = cell
        cfg = build_cfg(model_name, condition, dataset, seed, "cpu", caches[dataset])
        if int(cfg.seed) != seed or cfg.dataset.auto_generate_splits is not False:
            raise RuntimeError(f"{target_label(cell)}: training seed or fixed split protocol drift")
        if cfg.task.evaluate_test is not False or bool(cfg.task.get("link_prediction", False)):
            raise RuntimeError(f"{target_label(cell)}: Test/LP must be disabled")
        _, raw, safe, fingerprints, split_path = load_cell_data(model_name, condition, dataset, seed,
                                                                 caches[dataset], "cpu")
        aligned = assert_fingerprints_match_v13(dataset, seed, fingerprints, split_path, caches[dataset])
        if fingerprints["split_fingerprint"] != dataset_records[dataset]["split_fingerprint"]:
            raise RuntimeError(f"{target_label(cell)}: target split differs from the dataset's seed42 split")
        if not bool((safe.y[raw.test_idx] == -100).all()):
            raise RuntimeError(f"{target_label(cell)}: Test labels were not masked before Trainer")
        target_records.append({**target_label(cell), "cfg_seed": seed,
            "split_fingerprint": fingerprints["split_fingerprint"], "auto_generate_splits": False,
            "test_evaluation": False, "link_prediction": False,
            "test_labels_masked": True, "baseline_alignment": aligned,
            "num_nodes": int(safe.num_nodes), "num_classes": int(safe.num_classes)})

        # Same seeded construction must preserve every public V1.3-A tensor and
        # the external NC head. The inherited Joint View is constructed in its
        # isolated RNG scope, then removed from the Static model parameter set.
        v13_cfg = v13_runner.make_cfg(dataset, "joint_view", seed, "cpu", caches[dataset])
        v13_cfg.model.name = V13_MODEL
        v13_model = None
        static_model = None
        try:
            set_seed(seed)
            v13_model = build_model(v13_cfg, data_info(safe))
            v13_model.load_semantic_cache(caches[dataset]["cache_path_abs"],
                expected_dataset=dataset, expected_feature_fingerprint=caches[dataset]["feature_fingerprint"],
                expected_candidate_fingerprint=caches[dataset]["candidate_fingerprint"],
                expected_v7b_fingerprint=caches[dataset]["v7b_fingerprint"])
            v13_head = nn.Linear(v13_model.out_dim, int(safe.num_classes))
            set_seed(seed)
            static_model = build_model(cfg, data_info(safe))
            static_model.load_semantic_cache(caches[dataset]["cache_path_abs"],
                expected_dataset=dataset, expected_feature_fingerprint=caches[dataset]["feature_fingerprint"],
                expected_candidate_fingerprint=caches[dataset]["candidate_fingerprint"],
                expected_v7b_fingerprint=caches[dataset]["v7b_fingerprint"])
            static_head = nn.Linear(static_model.out_dim, int(safe.num_classes))
            v13_state, static_state = v13_model.state_dict(), static_model.state_dict()
            public_v13 = {name for name in v13_state if not name.startswith("joint_view.")}
            public_static = set(static_state) - {"static_logits"}
            if public_v13 != public_static:
                raise RuntimeError(f"{dataset}/seed{seed}: Static and V1.3 public state keys differ")
            if any(not torch.equal(v13_state[name], static_state[name]) for name in public_v13):
                raise RuntimeError(f"{dataset}/seed{seed}: Static public initialization differs from V1.3-A")
            if not (torch.equal(v13_head.weight, static_head.weight)
                    and torch.equal(v13_head.bias, static_head.bias)):
                raise RuntimeError(f"{dataset}/seed{seed}: external NC classifier initialization differs")
            if hasattr(static_model, "joint_view") or static_model.static_logits.numel() != 4:
                raise RuntimeError("Static model must have four new scalars and no Joint View network")
            if not torch.equal(static_model.static_logits, torch.zeros_like(static_model.static_logits)):
                raise RuntimeError("Static logits must initialize to exact zero")
            if not torch.equal(static_model.static_strengths_all(), torch.ones((2, 2))):
                raise RuntimeError("Static strengths must initialize to exact one")
            initialization_records.append({"dataset": dataset, "seed": seed, "status": "pass",
                "public_state_keys_equal": True, "all_public_model_tensors_equal": True,
                "classifier_tensors_equal": True, "static_logits_zero": True,
                "initial_static_strengths_one": True, "static_parameter_count": 4,
                "model_sha256": module_sha256(static_model),
                "classifier_sha256": module_sha256(static_head)})
        finally:
            del v13_model, static_model, raw, safe

    references = historical_references(caches)
    old = read_json(DATA / "preflight.json", {})
    result = {
        "study": "MR-CGE V1.3 Static-Strength Calibration Screen", "stage": "cpu_preflight",
        "status": "pass", "recorded_utc": now(), "base_sha": BASE_SHA,
        "base_branch": BASE_BRANCH, "branch": BRANCH, "source_git_sha": source_sha(),
        "training_source_sha": source_sha(),
        "source_file_sha256s": source_file_hashes(), "frozen_source_file_sha256s": frozen,
        "physical_gpu": PHYSICAL_GPU, "protected_historical_file_sha256s": protected_hashes,
        "authorized_training_cells": [target_label(cell) for cell in TARGET_CELLS],
        "target_protocol_records": target_records, "dataset_records": dataset_records,
        "initialization_equivalence": initialization_records,
        "historical_references": references, "test_evaluation": False, "link_prediction": False,
        "gpu_preflight_cells": old.get("gpu_preflight_cells", []),
        "gpu_preflight_attempts": old.get("gpu_preflight_attempts", []),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / "runs.csv").exists():
        write_rows(DATA / "runs.csv", [], RUN_FIELDS)
    if not (DATA / "failures.json").exists():
        json_write(DATA / "failures.json", {"failures": []})
    csv_headers = {
        "paired_differences.csv": ("dataset", "seed", "model_a", "model_b", "delta_val_acc"),
        "dataset_summary.csv": ("dataset", "condition", "n", "seeds"),
        "strength_diagnostics.csv": ("model", "condition", "dataset", "seed", "static_strengths"),
        "resource_summary.csv": ("model", "condition", "dataset", "seed", "attempt", "phase", "physical_gpu",
                                 "gpu_uuid", "memory_free_mib", "guard_status", "test_evaluation"),
        "epoch_history.csv": ("model", "condition", "dataset", "seed", "attempt", "epoch"),
    }
    for filename, fields in csv_headers.items():
        if not (DATA / filename).exists():
            write_rows(DATA / filename, [], fields)
    json_write(DATA / "preflight.json", result)
    json_write(DATA / "frozen_baseline.json", {
        "base_sha": BASE_SHA, "base_branch": BASE_BRANCH, "branch": BRANCH,
        "source_git_sha": source_sha(), "training_source_sha": source_sha(),
        "source_file_sha256s": result["source_file_sha256s"],
        "frozen_source_file_sha256s": frozen, "dataset_records": dataset_records,
        "protected_historical_file_sha256s": protected_hashes,
        "historical_references": references, "test_evaluation": False, "link_prediction": False,
    })
    old_manifest = read_json(DATA / "manifest.json", {})
    json_write(DATA / "manifest.json", {**old_manifest,
        "study": result["study"], "branch": BRANCH, "base_sha": BASE_SHA,
        "base_branch": BASE_BRANCH, "physical_gpu": PHYSICAL_GPU,
        "authorized_training_cells": result["authorized_training_cells"],
        "reference_only_conditions": references, "cpu_preflight_status": "pass",
        "gpu_preflight_status": "pending", "gpu_preflight_passed": 0,
        "completed_cells": old_manifest.get("completed_cells", 0),
        "failed_attempts": old_manifest.get("failed_attempts", 0),
        "blocked_resource_attempts": old_manifest.get("blocked_resource_attempts", 0),
        "failed_cells": old_manifest.get("failed_cells", 0),
        "blocked_resource_cells": old_manifest.get("blocked_resource_cells", 0),
        "unstarted_cells": old_manifest.get("unstarted_cells", len(TARGET_CELLS)),
        "test_evaluation": False, "link_prediction": False, "updated_utc": now()})
    return result



def capture_guard_baseline(cell: tuple[str, str, str, int], attempt: int | None,
                           phase: str) -> tuple[set[int], dict[str, Any]]:
    snapshot = v13_runner.smi_snapshot()
    device = next((row for row in snapshot["devices"] if row["physical_index"] == PHYSICAL_GPU), None)
    if device is None:
        raise RuntimeError(f"authorized physical GPU {PHYSICAL_GPU} not found")
    pids = {row["pid"] for row in snapshot["compute_processes"] if row["gpu_uuid"] == device["uuid"]}
    snapshot["authorized_baseline_compute_pids"] = sorted(pids)
    snapshot["authorization_note"] = "Only physical GPU 1 is authorized; these fresh launch-time PIDs are treated as existing shared users."
    record_resource({**snapshot, "guard": {"status": "baseline", "required_free_mib": 8192}},
                    *cell[:3], cell[3], attempt, phase,
                    {"baseline_pids": json.dumps(sorted(pids))})
    return pids, snapshot


def assert_hybrid_same_node_live(model: nn.Module, info: dict[str, Any]) -> dict[str, Any]:
    details = info.get("details", {}).get("modalities", {})
    audit = {}
    for modality in ("text", "visual"):
        late = details.get(modality, {}).get("late_completion", {})
        support = int(late.get("same_node_support_nodes", 0))
        rms = float(late.get("same_node_candidate_rms", 0.0))
        source = float(late.get("same_node_source_weight", 0.0))
        if support <= 0 or not math.isfinite(rms) or rms <= 0 or not math.isfinite(source) or source <= 0:
            raise RuntimeError(f"corrected H2 SameNode path inactive for {modality}: {late}")
        audit[modality] = {"support_nodes": support, "candidate_rms": rms,
                           "source_weight": source}
    if str(getattr(model, "paper_condition", "")) != "H2_physical_3hop":
        raise RuntimeError("Hybrid worker did not load corrected H2_physical_3hop")
    return audit


def assert_no_p0_effective_coefficients(model: nn.Module, info: dict[str, Any] | None = None) -> dict[str, Any]:
    physical = model.physical_alpha().detach().cpu()
    semantic = model.semantic_alpha().detach().cpu()
    if not bool(torch.equal(physical[:, 0], torch.zeros_like(physical[:, 0]))):
        raise RuntimeError("NoP0 Physical alpha contains an order-zero coefficient")
    if not bool(torch.allclose(physical[:, 1:].abs().sum(-1), torch.ones(physical.size(0)), atol=1e-7, rtol=1e-7)):
        raise RuntimeError("NoP0 Physical orders 1-3 are not signed-L1 normalized")
    if bool((semantic[:, :2].abs().sum(-1) <= 0).any()) or bool((semantic[:, 2:] != 0).any()):
        raise RuntimeError("Semantic coefficients do not retain precisely orders 0-1")
    checked_detail_keys = []
    if info is not None:
        filters = info.get("details", {}).get("filters", {})
        for name, detail in filters.items():
            if name.endswith("_physical"):
                coefficients = torch.as_tensor(detail.get("coefficients"))
                if not torch.equal(coefficients[:, 0], torch.zeros_like(coefficients[:, 0])):
                    raise RuntimeError(f"physical coefficient diagnostics are not masked for {name}")
                checked_detail_keys.append(name)
        required = {"private_text_physical", "private_visual_physical",
                    "collaborative_text_physical", "collaborative_visual_physical"}
        if not required.issubset(set(checked_detail_keys)):
            raise RuntimeError(f"Physical/Collaborative diagnostics missing: {required - set(checked_detail_keys)}")
    return {"physical_alpha": physical.tolist(), "semantic_alpha": semantic.tolist(),
            "private_and_collaborative_masks_verified": bool(info is not None),
            "diagnostic_filter_keys": checked_detail_keys}


def worker_gpu_preflight(cell: tuple[str, str, str, int], attempt: int,
                         physical_gpu: int, record_path: Path) -> None:
    if os.environ.get("MR_CGE_GUARDED_PREFLIGHT") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("GPU preflight worker must be isolated and started by the ownership guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed")
    baseline = {int(pid) for pid in os.environ.get("MR_CGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    v13_runner.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    model_name, condition, dataset, seed = cell
    cache = cache_entries()[dataset]
    cfg, raw, safe, fingerprints, split_path = load_cell_data(model_name, condition, dataset, seed, cache, "cuda:0")
    device = torch.device("cuda:0")
    set_seed(seed)
    model = build_model_and_cache(cfg, safe, cache, device)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    initial_model_hash, initial_classifier_hash = module_sha256(model), module_sha256(classifier)
    x, edges, labels = safe.x.to(device), safe.edge_index.to(device), safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    if bool((labels[train_idx] < 0).any()):
        raise RuntimeError("invalid label entered training indices")
    optimizer = torch.optim.AdamW([*model.parameters(), *classifier.parameters()],
                                  lr=float(cfg.task.lr), weight_decay=float(cfg.task.weight_decay))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); classifier.train()
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux_loss, info = model(x, edges, return_details=False)
    main_ce = F.cross_entropy(classifier(z[train_idx]), labels[train_idx])
    loss, private_ce = _compose_full_graph_loss(
        model=model, info=info, main_ce=main_ce, aux_loss=aux_loss,
        aux_weight=float(cfg.task.loss.aux_weight),
        unimodal_weight=float(cfg.task.loss.unimodal_weight), labels=labels, train_idx=train_idx,
    )
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(z).all()):
        raise RuntimeError("single-step preflight produced non-finite output or loss")
    loss.backward()
    params = [*model.parameters(), *classifier.parameters()]
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params):
        raise RuntimeError("single-step preflight produced non-finite gradients")
    if hasattr(model, "capture_gradient_diagnostics"):
        gradient_diagnostics = model.capture_gradient_diagnostics()
        if not all(item["gradient_finite"] for item in gradient_diagnostics.values()):
            raise RuntimeError("model gradient diagnostics report non-finite gradients")
    else:
        gradient_diagnostics = {}
    static_grad = model.static_logits.grad
    if static_grad is None or not bool(torch.isfinite(static_grad).all()) or not bool(static_grad.abs().sum() > 0):
        raise RuntimeError("Static-Strength logits must receive a finite, nonzero task gradient")
    if hasattr(model, "joint_view") or any(name.startswith("joint_view.") for name, _ in model.named_parameters()):
        raise RuntimeError("Static-Strength must not retain the Joint View node-level network")
    torch.nn.utils.clip_grad_norm_(params, float(cfg.task.grad_clip), error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    train_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    train_peak_reserved = int(torch.cuda.max_memory_reserved(device))

    model.eval(); classifier.eval()
    torch.cuda.reset_peak_memory_stats(device)
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        z_eval, _, _, _, eval_info = model(x, edges, return_details=True)
    torch.cuda.synchronize(device)
    rng_unchanged = torch.equal(cpu_rng, torch.random.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    if not rng_unchanged:
        raise RuntimeError("eval diagnostic forward changed RNG state")
    mechanism = {
        "static_logits_gradient_norm_before_step": float(static_grad.detach().float().norm().cpu()),
        "static_strengths_before_step": model.static_strengths_all().detach().cpu().tolist(),
        "joint_view_module_absent": not hasattr(model, "joint_view"),
    }
    eval_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    eval_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    peak_allocated = max(train_peak_allocated, eval_peak_allocated)
    peak_reserved = max(train_peak_reserved, eval_peak_reserved)
    snapshot = v13_runner.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="during_run")
    required_free = math.ceil(peak_reserved / (1024 * 1024)) + 2048
    record = {
        **target_label(cell), "attempt": attempt, "status": "pass", "stage": "gpu_single_step",
        "physical_gpu": physical_gpu, "started_utc": now(), "source_git_sha": source_sha(),
        "source_file_sha256s": source_file_hashes(), "config_sha256": hashlib.sha256(
            OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest(),
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": sha256_file(split_path),
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "num_nodes": int(safe.num_nodes), "num_edges": int(safe.edge_index.size(1)),
        "test_labels_masked": bool((safe.y[safe.test_idx] == -100).all()),
        "test_evaluation": False, "loss": float(loss.detach().cpu()),
        "main_ce": float(main_ce.detach().cpu()), "private_ce": float(private_ce.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()), "all_gradients_finite": True,
        "gradient_diagnostics": gradient_diagnostics, "mechanism_diagnostics": mechanism,
        "validation_rng_unchanged": rng_unchanged, "initial_model_sha256": initial_model_hash,
        "initial_classifier_sha256": initial_classifier_hash,
        "train_peak_allocated_bytes": train_peak_allocated,
        "train_peak_reserved_bytes": train_peak_reserved,
        "eval_peak_allocated_bytes": eval_peak_allocated, "eval_peak_reserved_bytes": eval_peak_reserved,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "required_start_free_mib_with_2gib_margin": required_free,
        "gpu_name": torch.cuda.get_device_name(device), "guard_snapshot": snapshot,
        "completed_utc": now(),
    }
    json_write(record_path, record)
    del model, classifier, optimizer, raw, safe, z, z_eval, info, eval_info
    torch.cuda.empty_cache()


def run_guarded_worker(action: str, cell: tuple[str, str, str, int], attempt: int,
                       *, baseline_pids: set[int], record_path: Path | None = None,
                       attempt_dir: Path | None = None, required_free_mib: int = 8192) -> dict[str, Any]:
    model_name, condition, dataset, seed = cell
    phase_prefix = "gpu_preflight" if action == "preflight" else "formal_train"
    try:
        launch_snapshot = v13_runner.guard_gpu(PHYSICAL_GPU, baseline_pids, set(),
            phase="before_launch", required_free_mib=required_free_mib)
    except Exception as exc:
        try:
            blocked = v13_runner.smi_snapshot()
            blocked["guard"] = {"status": "blocked", "phase": "before_launch",
                                 "required_free_mib": required_free_mib, "reason": repr(exc)}
            record_resource(blocked, model_name, condition, dataset, seed, attempt,
                            f"{phase_prefix}_blocked_before_launch")
        except Exception:
            pass
        raise
    record_resource(launch_snapshot, model_name, condition, dataset, seed, attempt,
                    f"{phase_prefix}_before_launch")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    env["MR_CGE_BASELINE_GPU_PIDS"] = ",".join(map(str, sorted(baseline_pids)))
    env["MR_CGE_REQUIRED_FREE_MIB"] = str(required_free_mib)
    if action == "preflight":
        env["MR_CGE_GUARDED_PREFLIGHT"] = "1"
        if record_path is None:
            raise ValueError("GPU preflight child requires a record path")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-gpu-preflight",
                   "--model", model_name, "--condition", condition, "--dataset", dataset,
                   "--seed", str(seed), "--attempt", str(attempt), "--record-path", str(record_path),
                   "--physical-gpu", str(PHYSICAL_GPU)]
    elif action == "train":
        env["MR_CGE_GUARDED_TRAIN"] = "1"
        env["MR_CGE_EXPECTED_PEAK_RESERVED_MIB"] = str(max(0, required_free_mib - 2048))
        if attempt_dir is None:
            raise ValueError("training child requires a unique attempt directory")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-train",
                   "--model", model_name, "--condition", condition, "--dataset", dataset,
                   "--seed", str(seed), "--attempt", str(attempt), "--attempt-dir", str(attempt_dir),
                   "--physical-gpu", str(PHYSICAL_GPU)]
    else:
        raise ValueError(action)
    log_dir = OUTPUT / ("preflight_logs" if action == "preflight" else "guard_logs")
    log_path = log_dir / f"{model_name}_{condition}_{dataset}_seed{seed}_attempt{attempt:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(5)
            try:
                snap = v13_runner.guard_gpu(PHYSICAL_GPU, baseline_pids, {process.pid}, phase="during_run")
                record_resource(snap, model_name, condition, dataset, seed, attempt,
                                f"{phase_prefix}_during_run")
            except Exception as exc:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                append_failure({**target_label(cell), "attempt": attempt, "stage": "runtime_gpu_guard",
                    "action": action, "status": "blocked_resource", "child_pid": process.pid,
                    "log_path": str(log_path), "terminated_own_child_only": True,
                    "error": repr(exc), "training_started": action == "train"})
                raise RuntimeError(f"runtime GPU guard blocked {action}: {exc}") from exc
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        append_failure({**target_label(cell), "attempt": attempt, "stage": action,
            "status": "failed", "child_exit_code": process.returncode, "log_path": str(log_path),
            "tail": tail, "training_started": action == "train"})
        raise RuntimeError(f"{action} worker exited {process.returncode}; see {log_path}")
    return {"log_path": str(log_path), "launch_guard": launch_snapshot}


def gpu_preflight(cells: list[tuple[str, str, str, int]]) -> dict[str, Any]:
    assert_source_committed()
    payload = read_json(DATA / "preflight.json", {})
    if payload.get("status") != "pass" or payload.get("stage") != "cpu_preflight":
        raise RuntimeError("CPU preflight must pass before GPU preflight")
    attempts = payload.setdefault("gpu_preflight_attempts", [])
    passed = {(row.get("model"), row.get("condition"), row.get("dataset"), int(row.get("seed", -1)))
              for row in attempts if row.get("status") == "pass" and row.get("source_file_sha256s") == source_file_hashes()}
    for cell in cells:
        if cell in passed:
            continue
        attempt = 1 + sum(1 for row in attempts if tuple(row.get(k) for k in ("model", "condition", "dataset")) == cell[:3]
                          and int(row.get("seed", -1)) == cell[3])
        record_path = DATA / "gpu_preflight_attempts" / (
            f"{cell[0]}__{cell[1]}__{cell[2]}__seed{cell[3]}__attempt{attempt:02d}.json")
        try:
            baseline, baseline_snap = capture_guard_baseline(cell, attempt, "gpu_preflight_baseline")
            run_guarded_worker("preflight", cell, attempt, baseline_pids=baseline,
                               record_path=record_path, required_free_mib=8192)
            record = read_json(record_path, {})
            if record.get("status") != "pass":
                raise RuntimeError("GPU preflight worker did not write a passing record")
            record["baseline_snapshot"] = baseline_snap
            record["log_path"] = str(OUTPUT / "preflight_logs" /
                f"{cell[0]}_{cell[1]}_{cell[2]}_seed{cell[3]}_attempt{attempt:02d}.log")
            attempts.append(record)
        except Exception as exc:
            text = repr(exc)
            log_text = ""
            log_candidate = OUTPUT / "preflight_logs" / f"{cell[0]}_{cell[1]}_{cell[2]}_seed{cell[3]}_attempt{attempt:02d}.log"
            if log_candidate.is_file():
                log_text = log_candidate.read_text(encoding="utf-8", errors="replace")[-8000:]
            status = "blocked_resource" if any(token in (text + log_text).lower() for token in
                ("guard", "free memory", "foreign compute", "temperature", "utilization", "below")) else "failed"
            failed = {**target_label(cell), "attempt": attempt, "status": status,
                      "stage": "gpu_preflight", "error": text, "log_tail": log_text,
                      "record_path": str(record_path), "source_git_sha": source_sha(),
                      "source_file_sha256s": source_file_hashes(), "test_evaluation": False}
            attempts.append(failed)
            append_failure(failed)
            json_write(DATA / "preflight.json", payload)
            # A resource block ends this stage safely. Other cells remain unstarted.
            if status == "blocked_resource":
                break
        json_write(DATA / "preflight.json", payload)
    latest = {}
    for row in attempts:
        key = (row.get("model"), row.get("condition"), row.get("dataset"), int(row.get("seed", -1)))
        if key in cells:
            latest[key] = row
    payload["gpu_preflight_cells"] = [latest.get(cell, {**target_label(cell), "status": "unstarted"}) for cell in cells]
    payload["gpu_preflight_status"] = "pass" if all(row.get("status") == "pass" for row in payload["gpu_preflight_cells"]) else "incomplete"
    payload["recorded_utc"] = now()
    json_write(DATA / "preflight.json", payload)
    manifest = read_json(DATA / "manifest.json", {})
    manifest["gpu_preflight_status"] = payload["gpu_preflight_status"]
    manifest["gpu_preflight_passed"] = sum(row.get("status") == "pass" for row in payload["gpu_preflight_cells"])
    manifest["updated_utc"] = now()
    json_write(DATA / "manifest.json", manifest)
    return payload


def assert_source_committed() -> None:
    commit = source_sha()
    tracked = [*FROZEN_PATHS,
               "src/models/mrcge_v13_static_strength.py",
               "scripts/run_mrcge_v13_static_strength_screen.py",
               "scripts/analyze_mrcge_v13_static_strength_screen.py",
               "configs/model/mrcge_v13_static_strength.yaml",
               "configs/study/mrcge_v13_static_strength_screen.yaml",
               "tests/test_mrcge_v13_static_strength_screen.py",
               "src/data/loaders.py", "src/data/splits.py", "src/data/graph_utils.py",
               "src/data/semantic_candidates_v7b.py", "src/data/types.py",
               "src/utils/seeds.py", "src/tasks/inference.py", "src/models/__init__.py",
               "configs/config.yaml", "configs/task/nc.yaml",
               "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml",
               "configs/dataset/ele-fashion.yaml"]
    mismatches = {}
    for relative in tracked:
        committed = git_file_sha(commit, relative)
        current = sha256_file(ROOT / relative)
        if committed != current:
            mismatches[relative] = {"commit": committed, "worktree": current}
    if mismatches:
        raise RuntimeError(f"GPU work requires committed, source-frozen files: {mismatches}")


def diagnostic_forward(cfg, safe, cache, checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = hybrid_runner.get_model(cfg, safe, cache, device)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval(); classifier.eval()
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        z, _, _, aux, info = model(safe.x.to(device), safe.edge_index.to(device), return_details=True)
        logits = classifier(z)
    rng_ok = torch.equal(before_cpu, torch.random.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state(device))
    if not rng_ok:
        raise RuntimeError("checkpoint diagnostic forward changed validation RNG state")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(aux)):
        raise RuntimeError("checkpoint diagnostic forward returned non-finite values")
    return model, {"info": info, "aux_loss": float(aux.detach().cpu()),
                   "validation_rng_unchanged": rng_ok}, logits.detach().cpu()


def write_success_diagnostics(cell, attempt: int, cfg, safe, cache, checkpoint: Path,
                              device: torch.device, source: str) -> dict[str, Any]:
    model_name, condition, dataset, seed = cell
    model, result, _ = diagnostic_forward(cfg, safe, cache, checkpoint, device)
    info = result["info"]
    details = info.get("details", {})
    run_key = {"model": model_name, "condition": condition, "dataset": dataset,
               "seed": seed, "attempt": attempt, "source_git_sha": source}
    coeff_path = DATA / "coefficients_diagnostics.csv"
    coeff_rows = read_rows(coeff_path)
    coeff_record = {**run_key, "physical_alpha": "", "semantic_alpha": "",
                    "theta_raw": "", "diagnostic_source": "best_validation_accuracy_checkpoint"}
    physical_alpha = model.filter_bank.coefficients(model.physical_hops, model.filter_bank.theta.dtype)
    semantic_alpha = model.filter_bank.coefficients(model.semantic_hops, model.filter_bank.theta.dtype)
    strengths = model.static_strengths_all().detach().cpu()
    coeff_record.update({
        "physical_alpha": json.dumps(physical_alpha.detach().cpu().tolist(), separators=(",", ":")),
        "semantic_alpha": json.dumps(semantic_alpha.detach().cpu().tolist(), separators=(",", ":")),
        "theta_raw": json.dumps(model.filter_bank.theta.detach().cpu().tolist(), separators=(",", ":")),
        "static_logits": json.dumps(model.static_logits.detach().cpu().tolist(), separators=(",", ":")),
        "static_strengths": json.dumps(strengths.tolist(), separators=(",", ":")),
        "effective_physical_strength": json.dumps((model.rho[:, None] * model.static_strengths_all()[:, :1]).detach().cpu().tolist(), separators=(",", ":")),
        "effective_semantic_strength": json.dumps((model.gamma[:, None] * model.static_strengths_all()[:, 1:]).detach().cpu().tolist(), separators=(",", ":")),
    })
    coeff_rows.append(coeff_record)
    write_rows(coeff_path, coeff_rows, list(dict.fromkeys([*(coeff_rows[0].keys() if coeff_rows else ()), *coeff_record.keys()])))

    view = {**run_key, "physical_support_nodes": details.get("physical_support_nodes", ""),
            "private_rms": json.dumps(details.get("private_rms", {}), separators=(",", ":")),
            "physical_response_rms": json.dumps(details.get("physical_response_rms", {}), separators=(",", ":")),
            "semantic_response_rms": json.dumps(details.get("semantic_response_rms", {}), separators=(",", ":")),
            "early_physical_cross": json.dumps(details.get("early_physical_cross", {}), separators=(",", ":")),
            "late": json.dumps(details.get("late", {}), separators=(",", ":")),
            "filter_diagnostics": json.dumps(details.get("filters", {}), separators=(",", ":")),
            "mechanism": json.dumps(details.get("v13_mechanism", {}), separators=(",", ":")),
            "route_summary": json.dumps(details.get("routes", {}), separators=(",", ":")),
            "collaborative_route_summary": json.dumps(details.get("collaborative_routes", {}), separators=(",", ":")),
            "validation_rng_unchanged": result["validation_rng_unchanged"],
            "test_evaluation": False}
    view_path = DATA / "view_diagnostics.csv"
    previous = read_rows(view_path)
    write_rows(view_path, [*previous, view], list(dict.fromkeys([*(previous[0].keys() if previous else ()), *view.keys()])))
    del model
    torch.cuda.empty_cache()
    return {"info": info, "validation_rng_unchanged": result["validation_rng_unchanged"],
            "coefficients": coeff_record, "views": view}


def worker_train(cell: tuple[str, str, str, int], attempt: int,
                 attempt_dir: Path, physical_gpu: int) -> None:
    if os.environ.get("MR_CGE_GUARDED_TRAIN") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("formal worker must be launched by the isolated ownership guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed")
    model_name, condition, dataset, seed = cell
    baseline = {int(pid) for pid in os.environ.get("MR_CGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    expected_peak = int(os.environ.get("MR_CGE_EXPECTED_PEAK_RESERVED_MIB", "0"))
    required_start_free = int(os.environ.get("MR_CGE_REQUIRED_FREE_MIB", "8192"))
    v13_runner.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="before_launch",
                         required_free_mib=max(8192, required_start_free, expected_peak + 2048))
    if not (attempt_dir / "attempt_started.json").is_file():
        raise RuntimeError("parent did not reserve this unique attempt directory")
    preflight = read_json(DATA / "preflight.json", {})
    matches = [row for row in preflight.get("gpu_preflight_attempts", [])
               if (row.get("model"), row.get("condition"), row.get("dataset"), int(row.get("seed", -1))) == cell
               and row.get("status") == "pass" and row.get("source_file_sha256s") == source_file_hashes()]
    if not matches:
        raise RuntimeError("no passing GPU preflight for this exact cell and frozen source")
    cache = cache_entries()[dataset]
    cfg, raw, safe, fingerprints, split_path = load_cell_data(model_name, condition, dataset, seed, cache, "cuda:0")
    device = torch.device("cuda:0")
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    cfg.task.extended_checkpoint_metadata = {}
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    model_source = ROOT / "src/models" / f"{model_name}.py"
    trainer_source = ROOT / "src/tasks/nc.py"
    model_source_hash, trainer_hash = sha256_file(model_source), sha256_file(trainer_source)
    source = source_sha()
    set_seed(seed)
    initial_model_hash, initial_classifier_hash, dropout_audit = hybrid_runner.init_pair_hashes(
        cfg, safe, cache, seed
    )
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3 Static-Strength Calibration Screen",
        **target_label(cell), "attempt": attempt, "source_git_sha": source,
        "base_sha": BASE_SHA, "protocol_version": PROTOCOL,
        "model_source_sha256": model_source_hash, "trainer_source_sha256": trainer_hash,
        "config_sha256": config_hash, "config_snapshot_path": str(attempt_dir / "config_snapshot.yaml"),
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": sha256_file(split_path),
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
        "initial_model_sha256": initial_model_hash,
        "initial_classifier_sha256": initial_classifier_hash,
        "test_evaluation": False, "test_label_values_indexed": False,
        "link_prediction": False,
    }
    log_path = attempt_dir / "train.log"
    logger = logging.getLogger(f"mrcge_static_strength.{dataset}.{condition}.{seed}.{attempt}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        run_nc(cfg, safe, device, logger)
        v13_runner.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="during_run")
        packed = read_json(Path(cfg.task.run_metrics_path), {})
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("training run protocol/seed provenance mismatch")
        entry = packed["runs"][0]
        metadata = entry["metadata"]
        selection = metadata["dual_checkpoint_selection"]
        init_audit = metadata["initial_parameter_audit"]
        if init_audit["model_state_sha256"] != initial_model_hash or init_audit["classifier_state_sha256"] != initial_classifier_hash:
            raise RuntimeError("Trainer model or external classifier initialization differs from the audited pair")
        accuracy_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        if not accuracy_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Best-Accuracy or Best-F1 Validation checkpoint missing")
        acc_sha, f1_sha = sha256_file(accuracy_path), sha256_file(f1_path)
        unregistered = dropout_audit.get("unregistered_tensor_state", {})
        acc_audit = hybrid_runner.evaluate_checkpoint(cfg, safe, cache, accuracy_path, device,
            acc_sha, "best_accuracy", unregistered)
        f1_audit = hybrid_runner.evaluate_checkpoint(cfg, safe, cache, f1_path, device,
            f1_sha, "best_f1", unregistered)
        if not (acc_audit["validation_rng_unchanged"] and f1_audit["validation_rng_unchanged"]):
            raise RuntimeError("checkpoint Validation reload changed RNG state")
        if bool((safe.y[safe.test_idx] != -100).any()) or bool(cfg.task.evaluate_test):
            raise RuntimeError("Test label/evaluation safety invariant failed")
        history = metadata.get("epoch_history", [])
        if not history or any(not math.isfinite(float(epoch[key])) for epoch in history
                              for key in ("train_loss", "val_acc", "val_macro_f1", "val_ce")):
            raise RuntimeError("complete finite epoch history missing")
        details = write_success_diagnostics(cell, attempt, cfg, safe, cache, accuracy_path,
                                           device, source)
        train, val, f1_train, f1_val = acc_audit["train"], acc_audit["val"], f1_audit["train"], f1_audit["val"]
        wall = float(metadata.get("training_wall_seconds", time.perf_counter() - started))
        record = {
            "model": model_name, "condition": condition, "dataset": dataset, "seed": seed,
            "attempt": attempt, "status": "complete", "source_git_sha": source,
            "model_source_sha256": model_source_hash, "trainer_source_sha256": trainer_hash,
            "config_sha256": config_hash, "feature_fingerprint": cache["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": sha256_file(split_path),
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_model_sha256": initial_model_hash, "initial_classifier_sha256": initial_classifier_hash,
            "best_acc_epoch": int(selection["best_accuracy"]["epoch"]),
            "best_f1_epoch": int(selection["best_macro_f1"]["epoch"]),
            "epochs_completed": int(metadata.get("accuracy_stopping_horizon_epochs", len(history))),
            "val_acc": val["acc"], "val_macro_f1": val["macro_f1"], "val_ce": val["ce"],
            "best_f1_val_acc": f1_val["acc"], "best_f1_val_macro_f1": f1_val["macro_f1"],
            "best_f1_val_ce": f1_val["ce"], "train_acc": train["acc"],
            "train_macro_f1": train["macro_f1"], "train_ce": train["ce"],
            "train_val_acc_gap": train["acc"] - val["acc"],
            "train_val_macro_f1_gap": train["macro_f1"] - val["macro_f1"],
            "train_val_ce_gap": train["ce"] - val["ce"],
            "best_f1_train_acc": f1_train["acc"], "best_f1_train_macro_f1": f1_train["macro_f1"],
            "best_f1_train_ce": f1_train["ce"],
            "model_parameters": int(metadata["model_parameters"]),
            "classifier_parameters": int(metadata["classifier_parameters"]),
            "total_parameters": int(metadata["model_parameters"]) + int(metadata["classifier_parameters"]),
            "wall_seconds": wall, "peak_allocated_bytes": int(metadata.get("cuda_peak_allocated_bytes", 0)),
            "peak_reserved_bytes": int(metadata.get("cuda_peak_reserved_bytes", 0)),
            "gpu_name": torch.cuda.get_device_name(device), "checkpoint_path": str(accuracy_path),
            "checkpoint_sha256": acc_sha, "best_f1_checkpoint_path": str(f1_path),
            "best_f1_checkpoint_sha256": f1_sha, "attempt_dir": str(attempt_dir),
            "config_snapshot_path": str(attempt_dir / "config_snapshot.yaml"),
            "run_metrics_path": str(attempt_dir / "run_metrics.json"),
            "validation_rng_unchanged": True, "checkpoint_reload_verified": True,
            "test_evaluation": False, "error": "",
            "optimizer": metadata.get("optimizer", "AdamW"), "lr": float(cfg.task.lr),
            "weight_decay": float(cfg.task.weight_decay), "dropout": float(cfg.model.get("dropout", 0.2)),
            "late_dropout": float(cfg.model.get("late_cross_dropout", 0.15)),
            "private_auxiliary_ce_weight": float(cfg.task.loss.unimodal_weight),
            "balance_weight": float(cfg.model.get("balance_weight", 0.01)),
            "same_node_enabled": bool(cfg.model.get("same_node_enabled", True)),
            "mechanism_diagnostics_path": str(attempt_dir / "mechanism_diagnostics.json"),
        }
        json_write(attempt_dir / "mechanism_diagnostics.json", details)
        append_csv(DATA / "runs.csv", record, RUN_FIELDS,
                   ("model", "condition", "dataset", "seed", "attempt"))
        histories_path = DATA / "epoch_history.csv"
        old_history = read_rows(histories_path)
        history_rows = [{"model": model_name, "condition": condition, "dataset": dataset,
                         "seed": seed, "attempt": attempt, "source_git_sha": source, **epoch}
                        for epoch in history]
        write_rows(histories_path, [*old_history, *history_rows],
                   list(dict.fromkeys([*(old_history[0].keys() if old_history else ()),
                                       *(history_rows[0].keys() if history_rows else ())])))
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(),
            "checkpoint_reload_verified": True, "test_evaluation": False})
    except Exception:
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(),
            "error": traceback.format_exc(), "test_evaluation": False})
        raise
    finally:
        del raw, safe
        torch.cuda.empty_cache()


def next_training_attempt(cell: tuple[str, str, str, int]) -> tuple[int, Path]:
    model, condition, dataset, seed = cell
    rows = [row for row in read_rows(DATA / "runs.csv") if row.get("model") == model
            and row.get("condition") == condition and row.get("dataset") == dataset
            and row.get("seed") == str(seed)]
    attempt = max([int(row.get("attempt", 0) or 0) for row in rows] + [0]) + 1
    path = RUNS / dataset / condition / f"seed_{seed}" / f"attempt_{attempt:02d}"
    if path.exists():
        raise RuntimeError(f"refusing to reuse existing attempt directory {path}")
    return attempt, path


def record_noncomplete(cell: tuple[str, str, str, int], attempt: int, status: str,
                       stage: str, error: str, *, attempt_dir: Path | None = None) -> None:
    row = {key: "" for key in RUN_FIELDS}
    row.update({**target_label(cell), "attempt": attempt, "status": status,
                "source_git_sha": source_sha(), "test_evaluation": False,
                "attempt_dir": str(attempt_dir or ""), "error": f"{stage}: {error}"})
    append_csv(DATA / "runs.csv", row, RUN_FIELDS,
               ("model", "condition", "dataset", "seed", "attempt"))
    append_failure({**target_label(cell), "attempt": attempt, "status": status,
                    "stage": stage, "error": error, "attempt_dir": str(attempt_dir or ""),
                    "test_evaluation": False})


def run_screen(cells: list[tuple[str, str, str, int]]) -> dict[str, Any]:
    assert_source_committed()
    preflight = read_json(DATA / "preflight.json", {})
    current_gpu_rows = [row for row in preflight.get("gpu_preflight_attempts", [])
                        if row.get("source_file_sha256s") == source_file_hashes()]
    latest_gpu = {}
    for row in current_gpu_rows:
        key = (row.get("model"), row.get("condition"), row.get("dataset"), int(row.get("seed", -1)))
        latest_gpu[key] = row
    missing_preflight = [target_label(cell) for cell in TARGET_CELLS
                         if latest_gpu.get(cell, {}).get("status") != "pass"]
    if missing_preflight:
        raise RuntimeError(f"Stage 3 requires all seven current-source GPU preflights to pass: {missing_preflight}")
    gpu_rows = [row for row in preflight.get("gpu_preflight_attempts", [])
                if row.get("source_file_sha256s") == source_file_hashes()]
    gpu_by_cell = {}
    for row in gpu_rows:
        key = (row.get("model"), row.get("condition"), row.get("dataset"), int(row.get("seed", -1)))
        gpu_by_cell[key] = row
    for cell in cells:
        prior_rows = [row for row in read_rows(DATA / "runs.csv") if row.get("model") == cell[0]
                      and row.get("condition") == cell[1] and row.get("dataset") == cell[2]
                      and row.get("seed") == str(cell[3])]
        if any(row.get("status") == "complete" for row in prior_rows):
            continue
        attempt, attempt_dir = next_training_attempt(cell)
        pre = gpu_by_cell.get(cell)
        if pre is None or pre.get("status") != "pass":
            status = "blocked_resource" if pre and pre.get("status") == "blocked_resource" else "failed"
            reason = pre.get("error", "matching source-frozen GPU preflight has not passed") if pre else "missing passing GPU preflight"
            record_noncomplete(cell, attempt, status, "gpu_preflight_gate", str(reason), attempt_dir=attempt_dir)
            if status == "blocked_resource":
                break
            continue
        expected = int(math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024))) + 2048
        attempt_dir.mkdir(parents=True, exist_ok=False)
        json_write(attempt_dir / "attempt_started.json", {
            **target_label(cell), "attempt": attempt, "started_utc": now(),
            "source_git_sha": source_sha(), "source_file_sha256s": source_file_hashes(),
            "gpu_preflight_record": pre.get("record_path", ""),
            "expected_peak_reserved_mib": expected, "test_evaluation": False,
        })
        baseline_snap = None
        try:
            baseline, baseline_snap = capture_guard_baseline(cell, attempt, "formal_train_baseline")
            run_guarded_worker("train", cell, attempt, baseline_pids=baseline,
                attempt_dir=attempt_dir, required_free_mib=max(8192, expected))
            manifest = read_json(DATA / "manifest.json", {})
            manifest["completed_cells"] = len({(row.get("model"), row.get("condition"), row.get("dataset"), row.get("seed"))
                for row in read_rows(DATA / "runs.csv") if row.get("status") == "complete"})
            manifest["failed_attempts"] = sum(row.get("status") == "failed" for row in read_rows(DATA / "runs.csv"))
            manifest["blocked_resource_attempts"] = sum(row.get("status") == "blocked_resource" for row in read_rows(DATA / "runs.csv"))
            manifest["updated_utc"] = now()
            json_write(DATA / "manifest.json", manifest)
        except Exception as exc:
            text = repr(exc)
            status = "blocked_resource" if any(token in text.lower() for token in
                ("guard", "free memory", "foreign compute", "temperature", "utilization", "below")) else "failed"
            if not any(row.get("model") == cell[0] and row.get("condition") == cell[1]
                       and row.get("dataset") == cell[2] and row.get("seed") == str(cell[3])
                       and row.get("attempt") == str(attempt) for row in read_rows(DATA / "runs.csv")):
                record_noncomplete(cell, attempt, status, "formal_training", text, attempt_dir=attempt_dir)
            json_write(attempt_dir / "status.json", {"status": status, "failed_utc": now(),
                "error": text, "baseline_snapshot": baseline_snap,
                "test_evaluation": False})
            if status == "blocked_resource":
                break
    rows = read_rows(DATA / "runs.csv")
    manifest = read_json(DATA / "manifest.json", {})
    manifest.update({"completed_cells": len({(row.get("model"), row.get("condition"), row.get("dataset"), row.get("seed"))
                                               for row in rows if row.get("status") == "complete"}),
                     "failed_attempts": sum(row.get("status") == "failed" for row in rows),
                     "blocked_resource_attempts": sum(row.get("status") == "blocked_resource" for row in rows),
                     "unstarted_cells": max(0, len(TARGET_CELLS) - len({(r.get("model"), r.get("condition"), r.get("dataset"), r.get("seed")) for r in rows})),
                     "updated_utc": now()})
    json_write(DATA / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-preflight", action="store_true")
    modes.add_argument("--gpu-preflight", action="store_true")
    modes.add_argument("--run-screen", action="store_true")
    modes.add_argument("--worker-gpu-preflight", action="store_true")
    modes.add_argument("--worker-train", action="store_true")
    parser.add_argument("--model", choices=sorted({cell[0] for cell in TARGET_CELLS}), nargs="+")
    parser.add_argument("--condition", nargs="+")
    parser.add_argument("--dataset", choices=("Movies", "Grocery", "ele-fashion"), nargs="+")
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--attempt-dir")
    parser.add_argument("--record-path")
    parser.add_argument("--physical-gpu", type=int, default=PHYSICAL_GPU)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.physical_gpu != PHYSICAL_GPU:
        raise SystemExit(f"only authorized physical GPU {PHYSICAL_GPU} may be used")
    if args.worker_gpu_preflight or args.worker_train:
        required = (args.model, args.condition, args.dataset, args.seed, args.attempt)
        if any(value is None for value in required):
            raise SystemExit("worker requires --model --condition --dataset --seed and --attempt")
        if len(args.model) != 1 or len(args.condition) != 1 or len(args.dataset) != 1 or len(args.seed) != 1:
            raise SystemExit("worker accepts exactly one model/condition/dataset/seed")
        cell = (args.model[0], args.condition[0], args.dataset[0], args.seed[0])
        if cell not in TARGET_CELLS:
            raise SystemExit(f"worker cell is not authorized: {cell}")
        if args.worker_gpu_preflight:
            if not args.record_path:
                raise SystemExit("GPU preflight worker requires --record-path")
            worker_gpu_preflight(cell, args.attempt, PHYSICAL_GPU, Path(args.record_path))
        else:
            if not args.attempt_dir:
                raise SystemExit("training worker requires --attempt-dir")
            worker_train(cell, args.attempt, Path(args.attempt_dir), PHYSICAL_GPU)
        return
    cells = target_cells(models=args.model, conditions=args.condition,
                         datasets=args.dataset, seeds=args.seed)
    if args.cpu_preflight:
        result = cpu_preflight()
        print(json.dumps({"status": result["status"], "cells": len(result["authorized_training_cells"]),
                          "preflight": str(DATA / "preflight.json")}, ensure_ascii=False))
    elif args.gpu_preflight:
        result = gpu_preflight(cells)
        print(json.dumps({"gpu_preflight_status": result.get("gpu_preflight_status"),
                          "cells": result.get("gpu_preflight_cells", [])}, ensure_ascii=False))
    elif args.run_screen:
        result = run_screen(cells)
        print(json.dumps({"completed_cells": result.get("completed_cells"),
                          "failed_cells": result.get("failed_cells"),
                          "blocked_resource_cells": result.get("blocked_resource_cells"),
                          "unstarted_cells": result.get("unstarted_cells")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
