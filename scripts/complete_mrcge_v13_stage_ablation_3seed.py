"""Complete seed-43/44 MR-CGE V1.3-A stage ablations and audit Full checkpoints.

The formal encoder, ablation subclass and NC trainer are intentionally frozen.
All seeds are explicit arguments; this runner never changes another runner's
module-level seed or reuses an ablation checkpoint as initialization.
"""

from __future__ import annotations

import argparse
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
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

from scripts import run_mrcge_v12_confirmation as v12
from scripts import run_mrcge_v13_screen as v13
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed


BASE_SHA = "784a47044459abeb7feced4f337e753ff1a5207b"
BASE_BRANCH = "exp/mrcge_v13_stage_ablation_screen"
BRANCH = "exp/mrcge_v13_stage_ablation_3seed_completion"
STUDY = ROOT / "research/mrcge_v13_stage_ablation_3seed"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_stage_ablation_3seed"
FULL_RUNS = ROOT / "research/mrcge_v13_seed44_completion/data/runs.csv"
SEED42_ABLATIONS = ROOT / "research/mrcge_v13_stage_ablation/data/runs.csv"
DATASETS = ("Movies", "Grocery", "ele-fashion")
ABLATIONS = ("no_early_cross", "no_late_completion")
NEW_SEEDS = (43, 44)
ALL_SEEDS = (42, 43, 44)
PHYSICAL_GPU = 1
PROTOCOL = v13.PROTOCOL
MODEL_NAME = "mrcge_v13_nc_ablation"
FULL_MODEL_NAME = "mrcge_v13"

NEW_RUNS = DATA / "new_runs.csv"
ALL_RUNS = DATA / "all_ablation_runs.csv"
PAIRED = DATA / "paired_deltas.csv"
SUMMARY = DATA / "summary.csv"
GATE_DIAGNOSTICS = DATA / "full_gate_diagnostics.csv"
MANIFEST = DATA / "manifest.json"
PREFLIGHT = DATA / "preflight.json"

NEW_FIELDS = (
    "dataset", "seed", "ablation", "attempt", "status", "selection_checkpoint",
    "best_acc_epoch", "best_f1_epoch", "epochs_completed", "val_acc", "val_macro_f1", "val_ce",
    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce",
    "train_acc", "train_macro_f1", "train_ce", "wall_seconds",
    "peak_allocated_bytes", "peak_reserved_bytes", "gpu_name",
    "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "source_git_sha", "base_sha", "model_source_sha256", "ablation_source_sha256",
    "trainer_source_sha256", "config_sha256", "feature_fingerprint", "dataset_fingerprint",
    "physical_edge_fingerprint", "split_fingerprint", "split_file_path", "split_file_sha256",
    "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    "initial_model_sha256", "initial_classifier_sha256", "best_f1_rule",
    "test_labels_masked", "test_evaluation", "trained_from_scratch",
    "full_checkpoint_used_for_initialization", "checkpoint_reload_verified",
    "validation_rng_unchanged", "error",
)

GATE_FIELDS = (
    "dataset", "seed", "modality", "status", "checkpoint_path", "checkpoint_sha256",
    "checkpoint_reload_strict", "validation_metrics_match", "val_acc", "val_macro_f1", "val_ce",
    "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint",
    "semantic_cache_sha256", "beta", "early_cross_raw_rms", "early_cross_calibrated_rms",
    "early_cross_to_neighbor_rms", "private_physical_contribution_rms",
    "collaborative_physical_contribution_rms", "collaborative_vs_private_sP_abs_mean",
    "late_selector_semantic_share", "late_selector_same_node_share", "late_selector_none_share",
    "p_accept_mean", "gate_mean", "late_delta_ratio", "late_delta_rms",
    "relative_cap_fraction", "semantic_support_nodes", "semantic_support_coverage_all_nodes",
    "test_evaluation", "error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def file_hashes() -> dict[str, str]:
    names = (
        "src/models/mrcge_v13.py", "src/models/mrcge_v13_nc_ablation.py",
        "src/tasks/nc.py", "src/models/factory.py", "scripts/run_mrcge_v13_screen.py",
        "scripts/run_mrcge_v13_stage_ablation.py", "scripts/run_mrcge_v12_confirmation.py",
        "scripts/complete_mrcge_v13_stage_ablation_3seed.py",
        "configs/model/mrcge_v13.yaml", "configs/task/nc.yaml",
    )
    result = {name: sha256_file(ROOT / name) for name in names}
    for dataset in DATASETS:
        name = f"configs/dataset/{dataset}.yaml"
        result[name] = sha256_file(ROOT / name)
    return result


def verify_branch(*, committed: bool = True) -> None:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}; found {branch}")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT, check=True)
    if committed:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
        if status:
            raise RuntimeError("commit the runner before preflight/training; tracked source is dirty")


def ensure_layout() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not NEW_RUNS.exists():
        write_rows(NEW_RUNS, [], NEW_FIELDS)
    if not GATE_DIAGNOSTICS.exists():
        write_rows(GATE_DIAGNOSTICS, [], GATE_FIELDS)


def full_reference_rows() -> dict[tuple[str, int], dict[str, str]]:
    expected = {(d, s) for d in DATASETS for s in ALL_SEEDS}
    rows = [r for r in read_rows(FULL_RUNS)
            if r.get("model") == FULL_MODEL_NAME and r.get("condition") == "M_repaired"
            and r.get("variant") == "joint_view" and r.get("status") == "complete"
            and r.get("selection_checkpoint") == "best_validation_accuracy"]
    indexed: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]))
        if key in indexed:
            raise RuntimeError(f"duplicate Full reference row {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise RuntimeError(f"expected nine Full records; missing={expected-set(indexed)}, extra={set(indexed)-expected}")
    return indexed


def seed42_ablation_rows() -> dict[tuple[str, str], dict[str, str]]:
    expected = {(d, a) for d in DATASETS for a in ABLATIONS}
    rows = [r for r in read_rows(SEED42_ABLATIONS)
            if int(r.get("seed", -1)) == 42 and r.get("status") == "complete"]
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["ablation"])
        if key in indexed:
            raise RuntimeError(f"duplicate seed42 ablation row {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise RuntimeError(f"expected six seed42 ablations; missing={expected-set(indexed)}, extra={set(indexed)-expected}")
    return indexed


def verify_checkpoint_row(row: dict[str, str], *, full: bool) -> None:
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                               ("best_f1_checkpoint_path", "best_f1_checkpoint_sha256")):
        path = Path(row[path_key])
        if not path.is_file():
            raise RuntimeError(f"checkpoint missing: {path}")
        if sha256_file(path) != row[hash_key]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
    if row.get("checkpoint_reload_verified", "").lower() != "true":
        raise RuntimeError(f"historical checkpoint lacks reload verification: {row.get('dataset')}/{row.get('seed')}")
    if row.get("selection_checkpoint") != "best_validation_accuracy":
        raise RuntimeError("Full/seed42 reference is not a Best-Validation-Accuracy record")
    if row.get("test_evaluation", "").lower() != "false":
        raise RuntimeError("historical reference reports Test evaluation enabled")
    if row.get("test_labels_masked", "").lower() != "true":
        raise RuntimeError("historical reference does not affirm Test label masking")


def expected_fingerprints(full: dict[str, str]) -> dict[str, str]:
    return {key: full[key] for key in (
        "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint",
        "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
        "candidate_fingerprint", "v7b_fingerprint",
    )}


def assert_cfg_protocol(cfg, seed: int, *, ablation: str | None) -> None:
    expected = {
        "seed": seed, "num_runs": 1,
        "condition": "M_repaired", "variant": "joint_view", "sender_private_ratio": 0.0,
        "hidden_dim": 256, "physical_hops": 3, "semantic_hops": 1,
        "num_experts": 4, "top_k": 2, "semantic_knn_k": 8,
        "model_dropout": 0.2, "late_cross_dropout": 0.15,
        "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 300, "patience": 30, "early_stop_min_epoch": 30,
        "early_stop_min_delta": 1e-4, "private_ce_weight": 0.3,
        "balance_weight": 0.01, "aux_weight": 1.0, "evaluate_test": False,
    }
    actual = {
        "seed": int(cfg.seed), "num_runs": int(cfg.num_runs),
        "condition": str(cfg.model.condition), "variant": str(cfg.model.v13_variant),
        "sender_private_ratio": float(cfg.model.sender_private_ratio),
        "hidden_dim": int(cfg.model.hidden_dim), "physical_hops": int(cfg.model.physical_hops),
        "semantic_hops": int(cfg.model.semantic_hops), "num_experts": int(cfg.model.num_experts),
        "top_k": int(cfg.model.top_k), "semantic_knn_k": int(cfg.model.semantic_knn_k),
        "model_dropout": float(cfg.model.dropout), "late_cross_dropout": float(cfg.model.late_cross_dropout),
        "optimizer": str(cfg.task.optimizer).lower(), "lr": float(cfg.task.lr),
        "weight_decay": float(cfg.task.weight_decay), "epochs": int(cfg.task.epochs),
        "patience": int(cfg.task.patience), "early_stop_min_epoch": int(cfg.task.early_stop_min_epoch),
        "early_stop_min_delta": float(cfg.task.early_stop_min_delta),
        "private_ce_weight": float(cfg.task.loss.unimodal_weight),
        "balance_weight": float(cfg.model.balance_weight), "aux_weight": float(cfg.task.loss.aux_weight),
        "evaluate_test": bool(cfg.task.evaluate_test),
    }
    if actual != expected:
        raise RuntimeError(f"frozen configuration mismatch for seed{seed}/{ablation}: {actual}")
    if ablation and str(cfg.model.name) != MODEL_NAME:
        raise RuntimeError("ablation configuration selected the wrong model class")
    if not ablation and str(cfg.model.name) != FULL_MODEL_NAME:
        raise RuntimeError("formal configuration selected the wrong Full model class")


def assert_alignment_against_full(dataset: str, fingerprints: dict[str, Any], aligned: dict[str, Any],
                                  cache: dict[str, Any], full_rows: dict[tuple[str, int], dict[str, str]]) -> dict[str, Any]:
    reference = full_rows[(dataset, 42)]
    actual = {
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": aligned["split_file_sha256"],
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
    }
    mismatch = {k: {"full_seed42": reference.get(k), "current": v}
                for k, v in actual.items() if str(reference.get(k, "")) != str(v)}
    for seed in (43, 44):
        other = full_rows[(dataset, seed)]
        for key, value in actual.items():
            if str(other.get(key, "")) != str(value):
                mismatch[f"full_seed{seed}:{key}"] = {"historical": other.get(key), "current": value}
    if mismatch:
        raise RuntimeError(f"{dataset} data/split/cache mismatch against historical Full: {mismatch}")
    return {**actual, "split_file_path": aligned["split_file_path"]}


def initialization_hashes(cfg, data) -> dict[str, str]:
    set_seed(int(cfg.seed))
    model = build_model(cfg, v13.data_info(data))
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    return {
        "model": v12.state_hash(model),
        "classifier": v12.state_hash(classifier),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
    }


def init_config(dataset: str, seed: int, cache: dict[str, Any], *, ablation: str | None):
    cfg = v13.make_cfg(dataset, "joint_view", seed, "cpu", cache)
    cfg.model.name = MODEL_NAME if ablation else FULL_MODEL_NAME
    if ablation:
        cfg.model.ablation_kind = ablation
    assert_cfg_protocol(cfg, seed, ablation=ablation)
    return cfg


def cpu_preflight() -> None:
    verify_branch()
    ensure_layout()
    full = full_reference_rows()
    old = seed42_ablation_rows()
    for row in full.values():
        verify_checkpoint_row(row, full=True)
    for row in old.values():
        verify_checkpoint_row(row, full=False)
        reference = full[(row["dataset"], 42)]
        for key in ("feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint",
                    "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
                    "candidate_fingerprint", "v7b_fingerprint"):
            if row.get(key) != reference.get(key):
                raise RuntimeError(f"seed42 ablation {row['dataset']}/{row['ablation']} mismatches Full on {key}")

    caches = v13.cache_entries()
    data_records: dict[str, Any] = {}
    init_records: dict[str, Any] = {}
    for dataset in DATASETS:
        data_records[dataset] = {}
        for seed in NEW_SEEDS:
            cache = caches[dataset]
            cfg, safe, fingerprints, aligned = v13.load_safe_data(dataset, "joint_view", seed, cache, "cpu")
            assert_cfg_protocol(cfg, seed, ablation=None)
            if bool(cfg.task.evaluate_test) or bool((safe.y[safe.test_idx] != -100).any()):
                raise RuntimeError("Test evaluation must be disabled and labels masked before training")
            current = assert_alignment_against_full(dataset, fingerprints, aligned, cache, full)
            data_records[dataset][str(seed)] = current
            init_records.setdefault(dataset, {})[str(seed)] = {}
            full_ref = full[(dataset, seed)]
            for kind in ABLATIONS:
                cfg_init = init_config(dataset, seed, cache, ablation=kind)
                observed = initialization_hashes(cfg_init, safe)
                if full_ref.get("initial_model_sha256") and observed["model"] != full_ref["initial_model_sha256"]:
                    raise RuntimeError(f"{dataset}/seed{seed}/{kind}: model initialization differs from same-seed Full")
                if full_ref.get("initial_classifier_sha256") and observed["classifier"] != full_ref["initial_classifier_sha256"]:
                    raise RuntimeError(f"{dataset}/seed{seed}/{kind}: classifier initialization differs from same-seed Full")
                init_records[dataset][str(seed)][kind] = observed
            cfg_formal = init_config(dataset, seed, cache, ablation=None)
            formal = initialization_hashes(cfg_formal, safe)
            for kind in ABLATIONS:
                observed = init_records[dataset][str(seed)][kind]
                if observed["model"] != formal["model"] or observed["classifier"] != formal["classifier"]:
                    raise RuntimeError(f"{dataset}/seed{seed}/{kind}: ablation initialization differs from formal model")

    source_hashes = file_hashes()
    frozen_names = (
        "src/models/mrcge_v13.py", "src/models/mrcge_v13_nc_ablation.py", "src/tasks/nc.py",
        "src/models/factory.py", "scripts/run_mrcge_v13_stage_ablation.py",
    )
    for name in frozen_names:
        before = subprocess.run(["git", "show", f"{BASE_SHA}:{name}"], cwd=ROOT, check=True,
                                capture_output=True).stdout
        if hashlib.sha256(before).hexdigest() != source_hashes[name]:
            raise RuntimeError(f"frozen source differs from Base Commit: {name}")

    payload = {
        "status": "pass", "completed_utc": now(), "branch": BRANCH,
        "base_sha": BASE_SHA, "preflight_git_sha": source_sha(),
        "source_file_sha256s": source_hashes,
        "frozen_source_hashes_match_base": True,
        "full_reference_rows": len(full), "full_checkpoint_hashes_verified": len(full) * 2,
        "seed42_ablation_rows": len(old), "seed42_checkpoint_hashes_verified": len(old) * 2,
        "data_fingerprints": data_records,
        "initialization_compatibility": init_records,
        "formal_cells": [{"dataset": d, "ablation": a, "seed": s}
                         for d in DATASETS for a in ABLATIONS for s in NEW_SEEDS],
        "training_config": {
            "model": MODEL_NAME, "condition": "M_repaired", "variant": "joint_view",
            "sender_private_ratio": 0.0, "hidden_dim": 256, "physical_hops": 3,
            "semantic_hops": 1, "experts": 4, "top_k": 2, "semantic_knn_k": 8,
            "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
            "max_epochs": 300, "patience": 30, "early_stop_min_epoch": 30,
            "early_stop_min_delta": 0.0001, "private_ce_weight": 0.3,
            "balance_loss_weight": 0.01, "trainer_aux_weight": 1.0,
            "backbone_dropout": 0.2, "late_cross_dropout": 0.15,
            "best_checkpoint": "Best-Validation-Accuracy", "best_f1_checkpoint_saved_separately": True,
            "evaluate_test": False, "test_label_mask": -100,
            "training_seeds": list(NEW_SEEDS), "fixed_split_seed": 42,
        },
    }
    json_write(PREFLIGHT, payload)
    manifest = initial_manifest()
    manifest.update({
        "cpu_preflight_status": "pass", "cpu_preflight_utc": now(),
        "source_file_sha256s": source_hashes, "preflight_git_sha": source_sha(),
        "data_fingerprints": data_records, "initialization_compatibility": init_records,
        "full_references_verified": 9, "seed42_ablation_references_verified": 6,
        "pending_cells": payload["formal_cells"], "status": "preflight_passed",
    })
    json_write(MANIFEST, manifest)
    print(f"CPU PREFLIGHT PASS: Full={len(full)} old ablations={len(old)} new cells=12", flush=True)


def initial_manifest() -> dict[str, Any]:
    old = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    return {
        **old,
        "study": "MR-CGE V1.3-A three-seed stage ablation completion and Full adaptive behavior audit",
        "branch": BRANCH, "base_branch": BASE_BRANCH, "base_sha": BASE_SHA,
        "training_source_git_sha": source_sha(),
        "formal_cells": [{"dataset": d, "ablation": a, "seed": s}
                         for d in DATASETS for a in ABLATIONS for s in NEW_SEEDS],
        "historical_seed42_source": str(SEED42_ABLATIONS.relative_to(ROOT)),
        "full_reference_source": str(FULL_RUNS.relative_to(ROOT)),
        "test_evaluation": False, "link_prediction": False,
        "gpu": old.get("gpu", {"physical_gpu": PHYSICAL_GPU, "guard_bypassed_by_user": False}),
    }


def append_new_run(row: dict[str, Any]) -> None:
    rows = read_rows(NEW_RUNS)
    key = (str(row.get("dataset")), str(row.get("ablation")), str(row.get("seed")), str(row.get("attempt")))
    if any((r.get("dataset"), r.get("ablation"), r.get("seed"), r.get("attempt")) == key for r in rows):
        raise RuntimeError(f"refusing duplicate attempt record {key}")
    write_rows(NEW_RUNS, [*rows, row], NEW_FIELDS)


def baseline_gpu_pids(snapshot: dict[str, Any], physical_gpu: int) -> set[int]:
    selected = next((r for r in snapshot["devices"] if int(r["physical_index"]) == physical_gpu), None)
    if selected is None:
        raise RuntimeError(f"physical GPU {physical_gpu} not found")
    return {int(r["pid"]) for r in snapshot["compute_processes"] if r["gpu_uuid"] == selected["uuid"]}


def baseline_manifest(snapshot: dict[str, Any], physical_gpu: int, pids: set[int]) -> dict[str, Any]:
    selected = next(r for r in snapshot["devices"] if int(r["physical_index"]) == physical_gpu)
    return {
        "physical_gpu": physical_gpu, "name": selected["name"], "uuid": selected["uuid"],
        "launch_snapshot": selected, "baseline_compute_pids": sorted(pids),
        "guard_bypassed_by_user": False, "guard_bypass_scope": "none",
    }


def append_resource_event(manifest: dict[str, Any], *, action: str, unit: str,
                          phase: str, status: str, snapshot: dict[str, Any] | None = None,
                          error: str = "") -> None:
    manifest.setdefault("gpu_guard_events", []).append({
        "time_utc": now(), "action": action, "unit": unit, "phase": phase,
        "status": status, "snapshot": snapshot, "error": error,
    })


def guarded_child(action: str, args: list[str], *, dataset: str, seed: int,
                  physical_gpu: int, baseline_pids: set[int], log_path: Path,
                  manifest: dict[str, Any], required_free_mib: int = 8192) -> tuple[int, str | None]:
    label = f"{dataset}/seed{seed}/{action}"
    try:
        before = v13.guard_gpu(physical_gpu, baseline_pids, set(), phase="before_launch",
                              required_free_mib=required_free_mib)
        append_resource_event(manifest, action=action, unit=label, phase="before_launch",
                              status="pass", snapshot=before)
    except Exception as exc:
        error = repr(exc)
        append_resource_event(manifest, action=action, unit=label, phase="before_launch",
                              status="blocked_resource", error=error)
        return 125, error

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["MRCGE_3SEED_CHILD"] = "1"
    env["MRCGE_BASELINE_GPU_PIDS"] = ",".join(map(str, sorted(baseline_pids)))
    command = [sys.executable, str(Path(__file__).resolve()), action, *args,
               "--physical-gpu", str(physical_gpu)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    guard_error: str | None = None
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        manifest["active_worker"] = {"pid": process.pid, "unit": label, "action": action,
                                     "started_utc": now(), "log_path": str(log_path)}
        json_write(MANIFEST, manifest)
        while process.poll() is None:
            time.sleep(3)
            try:
                v13.guard_gpu(physical_gpu, baseline_pids, {process.pid}, phase="during_run")
            except Exception as exc:
                guard_error = repr(exc)
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                break
    returncode = process.returncode if process.returncode is not None else -1
    manifest.pop("active_worker", None)
    if guard_error:
        append_resource_event(manifest, action=action, unit=label, phase="during_run",
                              status="blocked_resource", error=guard_error)
        return returncode, guard_error
    if returncode != 0:
        return returncode, None
    try:
        after = v13.guard_gpu(physical_gpu, baseline_pids, set(), phase="during_run")
        append_resource_event(manifest, action=action, unit=label, phase="after_run",
                              status="pass", snapshot=after)
    except Exception as exc:
        error = repr(exc)
        append_resource_event(manifest, action=action, unit=label, phase="after_run",
                              status="blocked_resource", error=error)
        return 125, error
    json_write(MANIFEST, manifest)
    return returncode, None


def correctness_worker(physical_gpu: int) -> None:
    if os.environ.get("MRCGE_3SEED_CHILD") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("correctness worker must be launched by the isolated GPU guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed")
    baseline = {int(x) for x in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if x}
    v13.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    dataset, seed = "Movies", 43
    cache = v13.cache_entries()[dataset]
    cfg, safe, _, _ = v13.load_safe_data(dataset, "joint_view", seed, cache, "cuda:0")
    if cfg.seed != seed or cfg.task.evaluate_test is not False or bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError("correctness preflight seed/Test protocol mismatch")
    device = torch.device("cuda:0")
    y = safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    results = {}
    for kind in ABLATIONS:
        cfg = v13.make_cfg(dataset, "joint_view", seed, "cuda:0", cache)
        cfg.model.name = MODEL_NAME
        cfg.model.ablation_kind = kind
        assert_cfg_protocol(cfg, seed, ablation=kind)
        set_seed(seed)
        model = build_model(cfg, v13.data_info(safe)).to(device)
        model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
        classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
        model.train()
        classifier.train()
        z, _, _, aux_loss, info = model(safe.x.to(device), safe.edge_index.to(device), return_details=True)
        main_ce = nn.functional.cross_entropy(classifier(z[train_idx]), y[train_idx])
        loss, private_ce = _compose_full_graph_loss(
            model=model, info=info, main_ce=main_ce, aux_loss=aux_loss,
            aux_weight=float(cfg.task.loss.aux_weight),
            unimodal_weight=float(cfg.task.loss.unimodal_weight), labels=y, train_idx=train_idx,
        )
        if z.shape != (safe.num_nodes, int(cfg.model.hidden_dim)):
            raise RuntimeError(f"{kind}: unexpected output shape {tuple(z.shape)}")
        if not torch.isfinite(z).all() or not torch.isfinite(loss) or not torch.isfinite(private_ce):
            raise RuntimeError(f"{kind}: non-finite forward output or loss")
        loss.backward()
        grads = [p.grad for p in list(model.parameters()) + list(classifier.parameters()) if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"{kind}: missing or non-finite backward gradients")
        results[kind] = {
            "output_shape": list(z.shape), "loss": float(loss.detach().cpu()),
            "private_ce": float(private_ce.detach().cpu()), "balance_aux": float(aux_loss.detach().cpu()),
            "finite_gradient_tensors": len(grads), "all_gradients_finite": True,
            "weights_updated": False, "test_labels_masked": True, "evaluate_test": False,
        }
        del grads, info, z, loss, private_ce, main_ce, aux_loss, classifier, model
        torch.cuda.empty_cache()
    v13.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="during_run")
    json_write(DATA / "gpu_correctness.json", {
        "status": "pass", "dataset": dataset, "seed": seed,
        "physical_gpu": physical_gpu, "conditions": results,
        "completed_utc": now(), "test_evaluation": False,
    })


def config_for_cell(dataset: str, kind: str, seed: int, cache: dict[str, Any], device: str):
    cfg = v13.make_cfg(dataset, "joint_view", seed, device, cache)
    cfg.model.name = MODEL_NAME
    cfg.model.ablation_kind = kind
    cfg.model.v13_variant = "joint_view"
    cfg.model.condition = "M_repaired"
    cfg.model.sender_private_ratio = 0.0
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 1.0
    cfg.task.loss.unimodal_weight = 0.3
    cfg.task.record_extended_validation = True
    cfg.task.record_train_ce_separately = True
    cfg.task.save_ckpt_path = None
    cfg.task.best_f1_ckpt_path = None
    assert_cfg_protocol(cfg, seed, ablation=kind)
    return cfg


def worker_train(dataset: str, kind: str, seed: int, attempt: int,
                 attempt_dir_text: str, physical_gpu: int) -> None:
    if os.environ.get("MRCGE_3SEED_CHILD") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("training worker must be launched by the isolated shared-GPU guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed")
    baseline = {int(x) for x in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if x}
    v13.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    attempt_dir = Path(attempt_dir_text)
    if not (attempt_dir / "attempt_started.json").is_file():
        raise RuntimeError("parent did not reserve a unique attempt directory")

    cache = v13.cache_entries()[dataset]
    cfg, safe, fingerprints, aligned = v13.load_safe_data(dataset, "joint_view", seed, cache, "cuda:0")
    cfg = config_for_cell(dataset, kind, seed, cache, "cuda:0")
    if int(cfg.seed) != seed or int(cfg.num_runs) != 1:
        raise RuntimeError("worker configuration did not preserve the explicit seed")
    assert_cfg_protocol(cfg, seed, ablation=None)
    alignment = assert_alignment_against_full(dataset, fingerprints, aligned, cache, full_reference_rows())
    if cfg.task.evaluate_test is not False or bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError("Test protocol was not disabled/masked before Trainer entry")

    device = torch.device("cuda:0")
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    cfg.paths.output_root = str(OUTPUT / "hydra")
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    config_sha = hashlib.sha256(config_text.encode()).hexdigest()
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    source = source_sha()
    model_source = sha256_file(ROOT / "src/models/mrcge_v13.py")
    ablation_source = sha256_file(ROOT / "src/models/mrcge_v13_nc_ablation.py")
    trainer_source = sha256_file(ROOT / "src/tasks/nc.py")
    full = full_reference_rows()[(dataset, seed)]
    init_expected = json.loads(PREFLIGHT.read_text())["initialization_compatibility"][dataset][str(seed)][kind]
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3-A three-seed stage ablation completion",
        "dataset": dataset, "seed": seed, "attempt": attempt,
        "model": MODEL_NAME, "ablation_kind": kind, "condition": "M_repaired",
        "variant": "joint_view", "sender_private_ratio": 0.0,
        "base_sha": BASE_SHA, "source_git_sha": source,
        "model_source_sha256": model_source, "ablation_source_sha256": ablation_source,
        "task_source_sha256": trainer_source, "config_sha256": config_sha,
        "full_checkpoint_path": full["checkpoint_path"], "full_checkpoint_sha256": full["checkpoint_sha256"],
        "full_checkpoint_used_for_initialization": False,
        "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": alignment["split_file_sha256"],
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "initial_model_sha256_expected": init_expected["model"],
        "initial_classifier_sha256_expected": init_expected["classifier"],
        "test_evaluation": False, "test_labels_indexed": False, "link_prediction": False,
    }

    logger = logging.getLogger(f"stage_ablation_3seed.{dataset}.{kind}.{seed}.{attempt}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(attempt_dir / "train.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    try:
        started = time.perf_counter()
        run_nc(cfg, safe, device, logger)
        v13.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="during_run")
        packed = json.loads(Path(cfg.task.run_metrics_path).read_text(encoding="utf-8"))
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("trainer protocol or run-seed provenance mismatch")
        entry = packed["runs"][0]
        metadata = entry["metadata"]
        initial_audit = metadata.get("initial_parameter_audit", {})
        if initial_audit.get("model_state_sha256") != init_expected["model"]:
            raise RuntimeError("Trainer initial model state differs from explicit-seed preflight")
        if initial_audit.get("classifier_state_sha256") != init_expected["classifier"]:
            raise RuntimeError("Trainer classifier state differs from explicit-seed preflight")
        selections = metadata["dual_checkpoint_selection"]
        acc_sel, f1_sel = selections["best_accuracy"], selections["best_macro_f1"]
        acc_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        if not acc_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Best-Accuracy or Best-F1 checkpoint is missing")
        acc_sha, f1_sha = sha256_file(acc_path), sha256_file(f1_path)
        acc_audit = v12.validation_logits_audit(cfg, safe, cache, acc_path, device, details=False)
        f1_audit = v12.validation_logits_audit(cfg, safe, cache, f1_path, device, details=False)
        metric_names = (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce"))
        for selected, audit, label in ((acc_sel, acc_audit, "Best-Accuracy"), (f1_sel, f1_audit, "Best-F1")):
            for saved, recomputed in metric_names:
                if abs(float(selected["metrics"][saved]) - float(audit["val"][recomputed])) > 3e-6:
                    raise RuntimeError(f"reloaded {label} checkpoint {saved} differs from trainer selection")
        if not (acc_audit["rng_unchanged"] and f1_audit["rng_unchanged"]):
            raise RuntimeError("Validation checkpoint audits changed RNG state")
        if cfg.task.evaluate_test is not False or bool((safe.y[safe.test_idx] != -100).any()):
            raise RuntimeError("Test evaluation/masking protocol changed during training")
        history = metadata.get("epoch_history", [])
        keys = ("train_loss", "val_acc", "val_macro_f1", "val_ce")
        if not history or any(not all(k in h and math.isfinite(float(h[k])) for k in keys) for h in history):
            raise RuntimeError("epoch history missing required finite train/validation metrics")
        for key in ("val_acc", "val_macro_f1", "val_ce"):
            if not math.isfinite(float(acc_audit["val"][{"val_acc":"acc", "val_macro_f1":"macro_f1", "val_ce":"ce"}[key]])):
                raise RuntimeError(f"non-finite Best-Accuracy {key}")

        row = {
            "dataset": dataset, "seed": seed, "ablation": kind, "attempt": attempt,
            "status": "complete", "selection_checkpoint": "best_validation_accuracy",
            "best_acc_epoch": acc_sel["epoch"], "best_f1_epoch": f1_sel["epoch"],
            "epochs_completed": len(history),
            "val_acc": acc_audit["val"]["acc"], "val_macro_f1": acc_audit["val"]["macro_f1"],
            "val_ce": acc_audit["val"]["ce"],
            "best_f1_val_acc": f1_audit["val"]["acc"],
            "best_f1_val_macro_f1": f1_audit["val"]["macro_f1"],
            "best_f1_val_ce": f1_audit["val"]["ce"],
            "train_acc": acc_audit["train"]["acc"], "train_macro_f1": acc_audit["train"]["macro_f1"],
            "train_ce": acc_audit["train"]["ce"],
            "wall_seconds": metadata.get("training_wall_seconds", time.perf_counter() - started),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes", ""),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes", ""),
            "gpu_name": metadata.get("cuda_device_name", torch.cuda.get_device_name(device)),
            "checkpoint_path": str(acc_path), "checkpoint_sha256": acc_sha,
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": f1_sha,
            "source_git_sha": source, "base_sha": BASE_SHA,
            "model_source_sha256": model_source, "ablation_source_sha256": ablation_source,
            "trainer_source_sha256": trainer_source, "config_sha256": config_sha,
            "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "split_file_path": alignment["split_file_path"], "split_file_sha256": alignment["split_file_sha256"],
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_model_sha256": initial_audit["model_state_sha256"],
            "initial_classifier_sha256": initial_audit["classifier_state_sha256"],
            "best_f1_rule": f1_sel.get("rule", "highest Validation Macro-F1 through Accuracy stopping horizon"),
            "test_labels_masked": True, "test_evaluation": False,
            "trained_from_scratch": True, "full_checkpoint_used_for_initialization": False,
            "checkpoint_reload_verified": True,
            "validation_rng_unchanged": True, "error": "",
        }
        append_new_run(row)
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(), "run": row})
        print(f"COMPLETE {dataset}/{kind}/seed{seed}: val_acc={row['val_acc']:.6f} "
              f"val_f1={row['val_macro_f1']:.6f} val_ce={row['val_ce']:.6f}", flush=True)
    except Exception as exc:
        json_write(attempt_dir / "status.json", {
            "status": "failed", "failed_utc": now(), "error": repr(exc),
            "traceback": traceback.format_exc(), "dataset": dataset, "seed": seed, "ablation": kind,
        })
        raise
    finally:
        torch.cuda.empty_cache()


def audit_metric_match(saved: dict[str, Any], measured: dict[str, Any]) -> bool:
    return all(abs(float(saved[k]) - float(measured[m])) <= 3e-6 for k, m in (
        ("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce")))


def detail_value(container: dict[str, Any], key: str) -> Any:
    value = container.get(key, "not_available")
    return "not_available" if value is None else value


def full_audit_worker(dataset: str, seed: int, physical_gpu: int) -> None:
    if os.environ.get("MRCGE_3SEED_CHILD") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("Full checkpoint audit must be launched by the isolated GPU guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed")
    baseline = {int(x) for x in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if x}
    v13.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    cache = v13.cache_entries()[dataset]
    cfg, safe, fingerprints, aligned = v13.load_safe_data(dataset, "joint_view", seed, cache, "cuda:0")
    full = full_reference_rows()[(dataset, seed)]
    verify_checkpoint_row(full, full=True)
    alignment = assert_alignment_against_full(dataset, fingerprints, aligned, cache, full_reference_rows())
    checkpoint = Path(full["checkpoint_path"])
    if sha256_file(checkpoint) != full["checkpoint_sha256"]:
        raise RuntimeError(f"Full checkpoint SHA256 changed before audit: {checkpoint}")
    if cfg.task.evaluate_test is not False or bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError("Test protocol failed before Full behavior audit")
    # The official model is used here, never the ablation subclass.
    cfg.model.name = FULL_MODEL_NAME
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
    assert_cfg_protocol(cfg, seed, ablation=None)
    device = torch.device("cuda:0")
    audit = v12.validation_logits_audit(cfg, safe, cache, checkpoint, device, details=True)
    if not audit["rng_unchanged"]:
        raise RuntimeError("Full checkpoint validation audit changed RNG state")
    metrics_match = audit_metric_match(full, audit["val"])
    if not metrics_match:
        raise RuntimeError(f"Full checkpoint Validation metrics mismatch: saved={full}, measured={audit['val']}")
    details = audit["info"].get("details", {})
    early = details.get("early_physical_cross", {})
    late = details.get("late", {})
    mechanism = details.get("v13_mechanism", {})
    if not early or not late or not mechanism:
        raise RuntimeError("formal Full checkpoint omitted required V1.3 mechanism details")
    rows = []
    for modality in ("text", "visual"):
        e = early.get(modality, {})
        l = late.get(modality, {})
        selector = l.get("selector_share", [])
        selector_values = [selector[i] if len(selector) > i else "not_available" for i in range(3)]
        sem_nodes = l.get("semantic_support_nodes", "not_available")
        coverage = (float(sem_nodes) / float(safe.num_nodes)
                    if sem_nodes != "not_available" and int(safe.num_nodes) > 0 else "not_available")
        row = {
            "dataset": dataset, "seed": seed, "modality": modality, "status": "complete",
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": full["checkpoint_sha256"],
            "checkpoint_reload_strict": True, "validation_metrics_match": True,
            "val_acc": audit["val"]["acc"], "val_macro_f1": audit["val"]["macro_f1"],
            "val_ce": audit["val"]["ce"],
            "feature_fingerprint": cache["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "beta": detail_value(e, "beta"), "early_cross_raw_rms": detail_value(e, "raw_rms"),
            "early_cross_calibrated_rms": detail_value(e, "calibrated_rms"),
            "early_cross_to_neighbor_rms": detail_value(e, "early_cross_to_neighbor_rms"),
            "private_physical_contribution_rms": detail_value(
                mechanism, f"physical_private_contribution_rms_{modality}"),
            "collaborative_physical_contribution_rms": detail_value(
                mechanism, f"physical_collaborative_contribution_rms_{modality}"),
            "collaborative_vs_private_sP_abs_mean": detail_value(
                mechanism, f"collaborative_vs_private_sP_abs_mean_{modality}"),
            "late_selector_semantic_share": selector_values[0],
            "late_selector_same_node_share": selector_values[1],
            "late_selector_none_share": selector_values[2],
            "p_accept_mean": detail_value(l, "p_accept_mean"),
            "gate_mean": detail_value(l, "gate_mean"),
            "late_delta_ratio": detail_value(l, "late_delta_ratio"),
            "late_delta_rms": detail_value(l, "late_delta_rms"),
            "relative_cap_fraction": detail_value(l, "relative_cap_fraction"),
            "semantic_support_nodes": sem_nodes,
            "semantic_support_coverage_all_nodes": coverage,
            "test_evaluation": False, "error": "",
        }
        rows.append(row)
    v13.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="during_run")
    current = read_rows(GATE_DIAGNOSTICS)
    key = (dataset, str(seed))
    current = [r for r in current if (r.get("dataset"), r.get("seed")) != key]
    write_rows(GATE_DIAGNOSTICS, [*current, *rows], GATE_FIELDS)
    json_write(OUTPUT / dataset / f"seed_{seed}" / "full_audit_status.json", {
        "status": "complete", "dataset": dataset, "seed": seed, "checkpoint_sha256": full["checkpoint_sha256"],
        "strict_load": True, "validation_metrics_match": True, "test_evaluation": False,
        "fingerprints": alignment, "completed_utc": now(),
    })
    del audit, details
    torch.cuda.empty_cache()
    print(f"FULL AUDIT COMPLETE {dataset}/seed{seed}: strict-load and Validation metrics match", flush=True)


def worker_dispatch(args) -> None:
    if args.worker_train:
        worker_train(args.dataset, args.ablation, args.seed, args.attempt, args.attempt_dir, args.physical_gpu)
    elif args.worker_correctness:
        correctness_worker(args.physical_gpu)
    elif args.worker_full_audit:
        full_audit_worker(args.dataset, args.seed, args.physical_gpu)


def next_attempt(dataset: str, kind: str, seed: int) -> tuple[int, Path]:
    root = OUTPUT / dataset / kind / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    attempts = [int(p.name.split("_")[-1]) for p in root.glob("attempt_*") if p.is_dir()]
    attempt = max(attempts, default=0) + 1
    return attempt, root / f"attempt_{attempt:02d}"


def latest_new_runs() -> dict[tuple[str, str, int], dict[str, str]]:
    indexed = {}
    for row in read_rows(NEW_RUNS):
        key = (row["dataset"], row["ablation"], int(row["seed"]))
        if key not in indexed or int(row.get("attempt", 0)) > int(indexed[key].get("attempt", 0)):
            indexed[key] = row
    return indexed


def mark_blocked_or_failed(dataset: str, kind: str, seed: int, attempt: int,
                           status: str, reason: str) -> None:
    row = {field: "" for field in NEW_FIELDS}
    row.update({"dataset": dataset, "seed": seed, "ablation": kind, "attempt": attempt,
                "status": status, "selection_checkpoint": "best_validation_accuracy",
                "base_sha": BASE_SHA, "source_git_sha": source_sha(),
                "test_labels_masked": True, "test_evaluation": False,
                "trained_from_scratch": True, "full_checkpoint_used_for_initialization": False,
                "error": reason})
    append_new_run(row)


def update_run_manifest(manifest: dict[str, Any], *, blocked: bool = False) -> dict[str, Any]:
    latest = latest_new_runs()
    complete = [k for k, r in latest.items() if r.get("status") == "complete"]
    failed = [k for k, r in latest.items() if r.get("status") == "failed"]
    blocked_keys = [k for k, r in latest.items() if r.get("status") == "blocked_resource"]
    all_cells = [(d, a, s) for d in DATASETS for a in ABLATIONS for s in NEW_SEEDS]
    done = set(complete)
    pending = [
        {"dataset": d, "ablation": a, "seed": s}
        for d, a, s in all_cells if (d, a, s) not in done
    ]
    manifest.update({
        "completed_new_cells": len(complete), "failed_new_cells": len(failed),
        "blocked_resource_cells": len(blocked_keys), "pending_cells": pending,
        "successful_units": [f"{d}/seed{s}/{a}" for d, a, s in sorted(complete)],
        "failed_units": [f"{d}/seed{s}/{a}" for d, a, s in sorted(failed)],
        "blocked_resource_units": [f"{d}/seed{s}/{a}" for d, a, s in sorted(blocked_keys)],
        "new_run_attempt_rows": len(read_rows(NEW_RUNS)),
        "last_updated_utc": now(), "test_evaluation": False,
    })
    if blocked or blocked_keys:
        manifest["status"] = "blocked_resource"
    elif len(complete) == 12 and not failed:
        manifest["status"] = "training_complete"
    elif failed:
        manifest["status"] = "training_partial"
    else:
        manifest["status"] = "training_in_progress"
    return manifest


def append_gate_failure(dataset: str, seed: int, status: str, error: str) -> None:
    old = read_rows(GATE_DIAGNOSTICS)
    old = [r for r in old if not (r.get("dataset") == dataset and r.get("seed") == str(seed))]
    rows = []
    full = full_reference_rows()[(dataset, seed)]
    for modality in ("text", "visual"):
        row = {field: "not_available" for field in GATE_FIELDS}
        row.update({
            "dataset": dataset, "seed": seed, "modality": modality,
            "status": status, "checkpoint_path": full["checkpoint_path"],
            "checkpoint_sha256": full["checkpoint_sha256"],
            "checkpoint_reload_strict": False, "validation_metrics_match": False,
            "test_evaluation": False, "error": error,
        })
        rows.append(row)
    write_rows(GATE_DIAGNOSTICS, [*old, *rows], GATE_FIELDS)


def append_gate_blocked_for_pending() -> None:
    existing = {(r.get("dataset"), int(r.get("seed", 0))) for r in read_rows(GATE_DIAGNOSTICS)}
    old = read_rows(GATE_DIAGNOSTICS)
    for dataset in DATASETS:
        for seed in ALL_SEEDS:
            if (dataset, seed) not in existing:
                append_gate_failure(dataset, seed, "blocked_resource", "GPU guard stopped before this audit")
                old = read_rows(GATE_DIAGNOSTICS)


def run_gpu_preflight(physical_gpu: int) -> bool:
    verify_branch()
    ensure_layout()
    manifest = initial_manifest()
    try:
        snapshot = v13.smi_snapshot()
        pids = baseline_gpu_pids(snapshot, physical_gpu)
        manifest["gpu"] = baseline_manifest(snapshot, physical_gpu, pids)
        manifest["gpu_baseline_captured_utc"] = snapshot["captured_utc"]
        json_write(MANIFEST, manifest)
    except Exception as exc:
        manifest["gpu_preflight_status"] = "blocked_resource"
        manifest["gpu_preflight_error"] = repr(exc)
        manifest["status"] = "blocked_resource"
        append_resource_event(manifest, action="gpu_preflight", unit="Movies/seed43/correctness",
                              phase="snapshot", status="blocked_resource", error=repr(exc))
        json_write(MANIFEST, manifest)
        return False
    rc, guard_error = guarded_child(
        "--worker-correctness", [], dataset="Movies", seed=43,
        physical_gpu=physical_gpu, baseline_pids=pids,
        log_path=OUTPUT / "preflight" / "gpu_correctness.log", manifest=manifest,
    )
    if guard_error:
        manifest["gpu_preflight_status"] = "blocked_resource"
        manifest["gpu_preflight_error"] = guard_error
        manifest["status"] = "blocked_resource"
    elif rc != 0:
        manifest["gpu_preflight_status"] = "failed"
        manifest["gpu_preflight_error"] = f"correctness worker exit code {rc}"
        manifest["status"] = "preflight_failed"
    else:
        check = json.loads((DATA / "gpu_correctness.json").read_text())
        if check.get("status") != "pass":
            manifest["gpu_preflight_status"] = "failed"
            manifest["gpu_preflight_error"] = "correctness worker did not record pass"
            manifest["status"] = "preflight_failed"
        else:
            manifest["gpu_preflight_status"] = "pass"
            manifest["gpu_preflight_utc"] = now()
            manifest["status"] = "preflight_passed"
    manifest["gpu"] = manifest.get("gpu", baseline_manifest(snapshot, physical_gpu, pids))
    manifest["gpu"]["guard_bypassed_by_user"] = False
    manifest["gpu"]["guard_bypass_scope"] = "none"
    json_write(MANIFEST, manifest)
    print(f"GPU PREFLIGHT {manifest['gpu_preflight_status']}: {manifest.get('gpu_preflight_error', '')}", flush=True)
    return manifest["gpu_preflight_status"] == "pass"


def run_training(physical_gpu: int) -> None:
    verify_branch()
    ensure_layout()
    if not PREFLIGHT.is_file() or not MANIFEST.is_file():
        raise RuntimeError("CPU preflight must pass before formal training")
    manifest = json.loads(MANIFEST.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight.get("status") != "pass" or manifest.get("cpu_preflight_status") != "pass":
        raise RuntimeError("CPU preflight has not passed")
    if manifest.get("gpu_preflight_status") != "pass":
        raise RuntimeError("GPU correctness preflight must pass before formal training")
    if manifest.get("preflight_git_sha") != source_sha():
        raise RuntimeError("runner source changed after preflight; rerun preflights")
    if preflight.get("source_file_sha256s") != file_hashes():
        raise RuntimeError("source files changed after CPU preflight")

    try:
        snapshot = v13.smi_snapshot()
        pids = baseline_gpu_pids(snapshot, physical_gpu)
    except Exception as exc:
        manifest["gpu"] = {"physical_gpu": physical_gpu, "guard_bypassed_by_user": False,
                            "guard_bypass_scope": "none"}
        manifest["status"] = "blocked_resource"
        manifest["blocked_before_training"] = repr(exc)
        manifest["pending_cells"] = [{"dataset": d, "ablation": a, "seed": s}
                                     for d in DATASETS for a in ABLATIONS for s in NEW_SEEDS]
        append_resource_event(manifest, action="training", unit="all_pending",
                              phase="snapshot", status="blocked_resource", error=repr(exc))
        json_write(MANIFEST, manifest)
        finalize_outputs()
        return
    manifest["gpu"] = baseline_manifest(snapshot, physical_gpu, pids)
    manifest["training_started_utc"] = manifest.get("training_started_utc", now())
    cells = [(d, a, s) for d in DATASETS for a in ABLATIONS for s in NEW_SEEDS]
    manifest["requested_cells"] = [{"dataset": d, "ablation": a, "seed": s} for d, a, s in cells]
    json_write(MANIFEST, manifest)

    blocked = False
    for index, (dataset, kind, seed) in enumerate(cells):
        previous = latest_new_runs().get((dataset, kind, seed), {})
        if previous.get("status") == "complete":
            acc = Path(previous.get("checkpoint_path", ""))
            f1 = Path(previous.get("best_f1_checkpoint_path", ""))
            if (acc.is_file() and f1.is_file()
                    and sha256_file(acc) == previous.get("checkpoint_sha256")
                    and sha256_file(f1) == previous.get("best_f1_checkpoint_sha256")):
                print(f"SKIP verified {dataset}/{kind}/seed{seed}", flush=True)
                continue
        attempt, attempt_dir = next_attempt(dataset, kind, seed)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        json_write(attempt_dir / "attempt_started.json", {
            "started_utc": now(), "dataset": dataset, "ablation": kind, "seed": seed,
            "attempt": attempt, "source_git_sha": source_sha(), "base_sha": BASE_SHA,
            "test_evaluation": False, "full_checkpoint_used_for_initialization": False,
        })
        print(f"LAUNCH {dataset}/{kind}/seed{seed}/attempt{attempt:02d} gpu{physical_gpu} guarded=True", flush=True)
        rc, guard_error = guarded_child(
            "--worker-train", ["--dataset", dataset, "--ablation", kind, "--seed", str(seed),
                               "--attempt", str(attempt), "--attempt-dir", str(attempt_dir)],
            dataset=dataset, seed=seed, physical_gpu=physical_gpu, baseline_pids=pids,
            log_path=attempt_dir / "launcher.log", manifest=manifest,
        )
        if guard_error:
            mark_blocked_or_failed(dataset, kind, seed, attempt, "blocked_resource", guard_error)
            manifest = update_run_manifest(manifest, blocked=True)
            manifest["blocked_at_cell"] = {"dataset": dataset, "ablation": kind, "seed": seed}
            manifest["pending_cells"] = [
                {"dataset": d, "ablation": a, "seed": s}
                for d, a, s in cells[index + 1:]
            ]
            blocked = True
            json_write(MANIFEST, manifest)
            break
        if rc != 0:
            status_path = attempt_dir / "status.json"
            reason = json.loads(status_path.read_text()).get("error", f"worker exit code {rc}") if status_path.is_file() else f"worker exit code {rc}"
            mark_blocked_or_failed(dataset, kind, seed, attempt, "failed", reason)
            print(f"FAILED {dataset}/{kind}/seed{seed}: {reason}", flush=True)
        else:
            latest = latest_new_runs().get((dataset, kind, seed), {})
            if latest.get("status") != "complete":
                mark_blocked_or_failed(dataset, kind, seed, attempt, "failed", "worker exited without a complete result row")
        manifest = update_run_manifest(manifest)
        manifest["gpu"]["physical_gpu"] = physical_gpu
        json_write(MANIFEST, manifest)

    if not blocked:
        manifest = update_run_manifest(manifest)
        manifest["training_finished_utc"] = now()
        json_write(MANIFEST, manifest)
        run_full_gate_audits(physical_gpu, pids, manifest)
    finalize_outputs()


def run_full_gate_audits(physical_gpu: int, pids: set[int], manifest: dict[str, Any]) -> None:
    blocked = False
    for dataset in DATASETS:
        for seed in ALL_SEEDS:
            existing = [r for r in read_rows(GATE_DIAGNOSTICS)
                        if r.get("dataset") == dataset and r.get("seed") == str(seed)]
            if len(existing) == 2 and all(r.get("status") == "complete" for r in existing):
                full = full_reference_rows()[(dataset, seed)]
                if all(r.get("checkpoint_sha256") == full["checkpoint_sha256"]
                       and r.get("validation_metrics_match") == "True" for r in existing):
                    print(f"SKIP verified Full audit {dataset}/seed{seed}", flush=True)
                    continue
            print(f"AUDIT Full {dataset}/seed{seed} gpu{physical_gpu} guarded=True", flush=True)
            audit_log = OUTPUT / dataset / f"seed_{seed}" / "full_audit.log"
            rc, guard_error = guarded_child(
                "--worker-full-audit", ["--dataset", dataset, "--seed", str(seed)],
                dataset=dataset, seed=seed, physical_gpu=physical_gpu, baseline_pids=pids,
                log_path=audit_log, manifest=manifest,
            )
            if guard_error:
                append_gate_failure(dataset, seed, "blocked_resource", guard_error)
                manifest = update_run_manifest(manifest, blocked=True)
                manifest["full_audit_blocked_at"] = {"dataset": dataset, "seed": seed}
                blocked = True
                json_write(MANIFEST, manifest)
                break
            if rc != 0:
                status_path = OUTPUT / dataset / f"seed_{seed}" / "full_audit_status.json"
                reason = json.loads(status_path.read_text()).get("error", f"audit worker exit code {rc}") if status_path.is_file() else f"audit worker exit code {rc}"
                append_gate_failure(dataset, seed, "failed", reason)
                manifest.setdefault("full_audit_failures", []).append({"dataset": dataset, "seed": seed, "error": reason})
            manifest["full_gate_audits_completed"] = sum(
                1 for r in read_rows(GATE_DIAGNOSTICS)
                if r.get("modality") == "text" and r.get("status") == "complete"
            )
            json_write(MANIFEST, manifest)
        if blocked:
            break
    if blocked:
        append_gate_blocked_for_pending()
    manifest["full_gate_diagnostics_rows"] = len(read_rows(GATE_DIAGNOSTICS))
    manifest["full_gate_audits_completed"] = sum(
        1 for r in read_rows(GATE_DIAGNOSTICS)
        if r.get("modality") == "text" and r.get("status") == "complete"
    )
    manifest["full_gate_audits_failed"] = sum(
        1 for r in read_rows(GATE_DIAGNOSTICS)
        if r.get("modality") == "text" and r.get("status") == "failed"
    )
    if blocked:
        manifest["status"] = "blocked_resource"
    json_write(MANIFEST, manifest)


ALL_RUN_FIELDS = (
    "dataset", "seed", "ablation", "record_origin", "status", "selection_checkpoint",
    "best_acc_epoch", "best_f1_epoch", "epochs_completed", "val_acc", "val_macro_f1", "val_ce",
    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "checkpoint_path",
    "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "source_git_sha", "base_sha", "feature_fingerprint", "dataset_fingerprint",
    "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256",
    "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    "test_labels_masked", "test_evaluation", "checkpoint_reload_verified", "error",
)

PAIRED_FIELDS = (
    "dataset", "seed", "ablation", "status", "full_val_acc", "full_val_macro_f1", "full_val_ce",
    "ablation_val_acc", "ablation_val_macro_f1", "ablation_val_ce",
    "delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce",
    "full_checkpoint_path", "full_checkpoint_sha256", "ablation_checkpoint_path",
    "ablation_checkpoint_sha256", "pairing", "record_origin",
)

SUMMARY_FIELDS = (
    "dataset", "ablation", "n", "seeds",
    "full_val_acc_mean", "full_val_acc_sample_sd", "full_val_macro_f1_mean", "full_val_macro_f1_sample_sd",
    "full_val_ce_mean", "full_val_ce_sample_sd",
    "ablation_val_acc_mean", "ablation_val_acc_sample_sd",
    "ablation_val_macro_f1_mean", "ablation_val_macro_f1_sample_sd",
    "ablation_val_ce_mean", "ablation_val_ce_sample_sd",
    "delta_val_acc_pp_mean", "delta_val_acc_pp_sample_sd",
    "delta_val_macro_f1_pp_mean", "delta_val_macro_f1_pp_sample_sd",
    "delta_val_ce_mean", "delta_val_ce_sample_sd",
    "delta_acc_positive_seeds", "delta_acc_negative_seeds", "delta_acc_tie_seeds",
    "delta_f1_positive_seeds", "delta_f1_negative_seeds", "delta_f1_tie_seeds",
    "delta_ce_positive_seeds", "delta_ce_negative_seeds", "delta_ce_tie_seeds",
    "seed42_delta_acc_pp", "seed43_delta_acc_pp", "seed44_delta_acc_pp",
    "seed42_delta_f1_pp", "seed43_delta_f1_pp", "seed44_delta_f1_pp",
    "seed42_delta_ce", "seed43_delta_ce", "seed44_delta_ce",
)


def normalize_ablation_rows() -> list[dict[str, Any]]:
    old = seed42_ablation_rows()
    new = latest_new_runs()
    rows = []
    for (dataset, kind), row in old.items():
        rows.append({
            "dataset": dataset, "seed": 42, "ablation": kind,
            "record_origin": "historical_seed42", "status": row.get("status", "complete"),
            "selection_checkpoint": row.get("selection_checkpoint", "best_validation_accuracy"),
            "best_acc_epoch": row.get("best_acc_epoch", ""), "best_f1_epoch": row.get("best_f1_epoch", ""),
            "epochs_completed": row.get("epochs_completed", ""), "val_acc": row.get("val_acc", ""),
            "val_macro_f1": row.get("val_macro_f1", ""), "val_ce": row.get("val_ce", ""),
            "best_f1_val_acc": row.get("best_f1_val_acc", ""),
            "best_f1_val_macro_f1": row.get("best_f1_val_macro_f1", ""),
            "best_f1_val_ce": row.get("best_f1_val_ce", ""),
            "checkpoint_path": row.get("checkpoint_path", ""),
            "checkpoint_sha256": row.get("checkpoint_sha256", ""),
            "best_f1_checkpoint_path": row.get("best_f1_checkpoint_path", ""),
            "best_f1_checkpoint_sha256": row.get("best_f1_checkpoint_sha256", ""),
            "source_git_sha": row.get("source_git_sha", ""), "base_sha": row.get("base_sha", ""),
            "feature_fingerprint": row.get("feature_fingerprint", ""),
            "dataset_fingerprint": row.get("dataset_fingerprint", ""),
            "physical_edge_fingerprint": row.get("physical_edge_fingerprint", ""),
            "split_fingerprint": row.get("split_fingerprint", ""),
            "split_file_sha256": row.get("split_file_sha256", ""),
            "semantic_cache_sha256": row.get("semantic_cache_sha256", ""),
            "candidate_fingerprint": row.get("candidate_fingerprint", ""),
            "v7b_fingerprint": row.get("v7b_fingerprint", ""),
            "test_labels_masked": row.get("test_labels_masked", ""),
            "test_evaluation": row.get("test_evaluation", ""),
            "checkpoint_reload_verified": row.get("checkpoint_reload_verified", ""),
            "error": row.get("error", ""),
        })
    for row in new.values():
        rows.append({**row, "record_origin": "new_seed43_44"})
    rows.sort(key=lambda r: (DATASETS.index(r["dataset"]), ABLATIONS.index(r["ablation"]), int(r["seed"])))
    return rows


def make_paired_and_summary() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_rows = full_reference_rows()
    ablations = {(r["dataset"], r["ablation"], int(r["seed"])): r for r in normalize_ablation_rows()}
    paired: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kind in ABLATIONS:
            group = []
            for seed in ALL_SEEDS:
                full = full_rows[(dataset, seed)]
                ab = ablations.get((dataset, kind, seed), {})
                complete = ab.get("status") == "complete"
                values = {}
                if complete:
                    values = {
                        "full_val_acc": float(full["val_acc"]),
                        "full_val_macro_f1": float(full["val_macro_f1"]),
                        "full_val_ce": float(full["val_ce"]),
                        "ablation_val_acc": float(ab["val_acc"]),
                        "ablation_val_macro_f1": float(ab["val_macro_f1"]),
                        "ablation_val_ce": float(ab["val_ce"]),
                        "delta_val_acc_pp": 100 * (float(ab["val_acc"]) - float(full["val_acc"])),
                        "delta_val_macro_f1_pp": 100 * (float(ab["val_macro_f1"]) - float(full["val_macro_f1"])),
                        "delta_val_ce": float(ab["val_ce"]) - float(full["val_ce"]),
                    }
                    group.append(values)
                paired.append({
                    "dataset": dataset, "seed": seed, "ablation": kind,
                    "status": ab.get("status", "not_started"), **values,
                    "full_checkpoint_path": full["checkpoint_path"],
                    "full_checkpoint_sha256": full["checkpoint_sha256"],
                    "ablation_checkpoint_path": ab.get("checkpoint_path", ""),
                    "ablation_checkpoint_sha256": ab.get("checkpoint_sha256", ""),
                    "pairing": "same_dataset_same_seed", "record_origin": ab.get("record_origin", "not_available"),
                })
            if not group:
                continue
            row: dict[str, Any] = {
                "dataset": dataset, "ablation": kind, "n": len(group),
                "seeds": ";".join(str(seed) for seed in ALL_SEEDS
                                  if (dataset, kind, seed) in ablations
                                  and ablations[(dataset, kind, seed)].get("status") == "complete"),
            }
            metric_specs = (
                ("full_val_acc", "full_val_acc"), ("full_val_macro_f1", "full_val_macro_f1"),
                ("full_val_ce", "full_val_ce"),
                ("ablation_val_acc", "ablation_val_acc"), ("ablation_val_macro_f1", "ablation_val_macro_f1"),
                ("ablation_val_ce", "ablation_val_ce"),
                ("delta_val_acc_pp", "delta_val_acc_pp"),
                ("delta_val_macro_f1_pp", "delta_val_macro_f1_pp"), ("delta_val_ce", "delta_val_ce"),
            )
            for source, prefix in metric_specs:
                vals = [float(g[source]) for g in group]
                row[f"{prefix}_mean"] = statistics.mean(vals)
                row[f"{prefix}_sample_sd"] = statistics.stdev(vals) if len(vals) > 1 else "not_available"
            for metric, key in (("acc", "delta_val_acc_pp"), ("f1", "delta_val_macro_f1_pp"),
                                ("ce", "delta_val_ce")):
                values = [float(g[key]) for g in group]
                row[f"delta_{metric}_positive_seeds"] = sum(v > 0 for v in values)
                row[f"delta_{metric}_negative_seeds"] = sum(v < 0 for v in values)
                row[f"delta_{metric}_tie_seeds"] = sum(v == 0 for v in values)
                group_by_seed = {
                    int(p["seed"]): p for p in paired
                    if p.get("dataset") == dataset and p.get("ablation") == kind
                }
                for seed in ALL_SEEDS:
                    match = group_by_seed.get(seed, {})
                    row[f"seed{seed}_delta_{metric}{'_pp' if metric != 'ce' else ''}"] = match.get(key, "")
            summary.append(row)
    write_rows(ALL_RUNS, normalize_ablation_rows(), ALL_RUN_FIELDS)
    write_rows(PAIRED, paired, PAIRED_FIELDS)
    write_rows(SUMMARY, summary, SUMMARY_FIELDS)
    return paired, summary


def mean_field(rows: list[dict[str, str]], key: str) -> float | None:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return statistics.mean(values) if values else None


def fmt_mean_sd(mean: Any, sd: Any, *, scale: float = 1.0, digits: int = 3) -> str:
    if mean in (None, "") or sd in (None, "", "not_available"):
        return "not_available"
    return f"{float(mean)*scale:.{digits}f} ± {float(sd)*scale:.{digits}f}"


def delta_consistency_text(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 3:
        return "该组未完成三个训练种子。"
    harm = sum(float(r["delta_val_acc_pp"]) < 0 and float(r["delta_val_macro_f1_pp"]) < 0
               and float(r["delta_val_ce"]) > 0 for r in rows)
    benefit = sum(float(r["delta_val_acc_pp"]) > 0 and float(r["delta_val_macro_f1_pp"]) > 0
                  and float(r["delta_val_ce"]) < 0 for r in rows)
    if harm == 3:
        return "三个种子的 Acc/F1/CE 方向都支持完整模型；该数据集上移除模块的损害方向一致。"
    if benefit == 3:
        return "三个种子的 Acc/F1/CE 方向都支持移除该模块；该数据集上简化方向一致。"
    return f"种子方向混合（完整模型三项方向一致占 {harm}/3；消融三项方向一致占 {benefit}/3）。"


def gate_summary_rows() -> list[dict[str, Any]]:
    rows = read_rows(GATE_DIAGNOSTICS)
    complete = [r for r in rows if r.get("status") == "complete"]
    metrics = (
        "beta", "early_cross_raw_rms", "early_cross_calibrated_rms", "early_cross_to_neighbor_rms",
        "private_physical_contribution_rms", "collaborative_physical_contribution_rms",
        "collaborative_vs_private_sP_abs_mean", "late_selector_semantic_share",
        "late_selector_same_node_share", "late_selector_none_share", "p_accept_mean", "gate_mean",
        "late_delta_ratio", "late_delta_rms", "relative_cap_fraction", "semantic_support_coverage_all_nodes",
    )
    output = []
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            group = [r for r in complete if r.get("dataset") == dataset and r.get("modality") == modality]
            row: dict[str, Any] = {"dataset": dataset, "modality": modality, "n": len(group)}
            for key in metrics:
                value = mean_field(group, key)
                row[f"{key}_mean"] = value if value is not None else "not_available"
            output.append(row)
    return output


def report_text(manifest: dict[str, Any], paired: list[dict[str, Any]],
                summary: list[dict[str, Any]]) -> str:
    lines = [
        "# MR-CGE V1.3-A 三种子结构消融收敛与 Full 自适应行为审计", "",
        f"状态：**{manifest.get('status', 'unknown')}**。本轮新增 seed43/44 的 12 个消融训练，合并历史 seed42 六条消融；没有重训 Full 或 seed42，也没有运行 LP / Test。", "",
        "每个消融都与相同数据集、相同 seed 的 Full Best-Validation-Accuracy Checkpoint 配对。方向定义为 Ablation − Full；Acc/F1 差值为百分点，CE 为绝对差。均值的 SD 为样本 SD（ddof=1）。", "",
        "## 三种子配对汇总", "",
        "Acc/F1 以百分比及百分点表示；CE 保留原始交叉熵尺度。括号内为 seed42/43/44 的方向计数（正/负/持平）。", "",
        "| Dataset | Ablation | Full Acc / F1 / CE (mean ± SD) | Ablation Acc / F1 / CE (mean ± SD) | ΔAcc pp | ΔF1 pp | ΔCE | 方向计数 Acc / F1 / CE |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    summaries = {(r["dataset"], r["ablation"]): r for r in summary}
    for dataset in DATASETS:
        for kind in ABLATIONS:
            r = summaries.get((dataset, kind))
            if not r:
                lines.append(f"| {dataset} | {kind} | pending | pending | — | — | — | — |")
                continue
            full_text = (
                f"{fmt_mean_sd(r.get('full_val_acc_mean'), r.get('full_val_acc_sample_sd'), scale=100, digits=3)} / "
                f"{fmt_mean_sd(r.get('full_val_macro_f1_mean'), r.get('full_val_macro_f1_sample_sd'), scale=100, digits=3)} / "
                f"{fmt_mean_sd(r.get('full_val_ce_mean'), r.get('full_val_ce_sample_sd'), digits=4)}"
            )
            ab_text = (
                f"{fmt_mean_sd(r.get('ablation_val_acc_mean'), r.get('ablation_val_acc_sample_sd'), scale=100, digits=3)} / "
                f"{fmt_mean_sd(r.get('ablation_val_macro_f1_mean'), r.get('ablation_val_macro_f1_sample_sd'), scale=100, digits=3)} / "
                f"{fmt_mean_sd(r.get('ablation_val_ce_mean'), r.get('ablation_val_ce_sample_sd'), digits=4)}"
            )
            signs = " / ".join(
                f"{r[f'delta_{k}_positive_seeds']}/{r[f'delta_{k}_negative_seeds']}/{r[f'delta_{k}_tie_seeds']}"
                for k in ("acc", "f1", "ce")
            )
            lines.append(
                f"| {dataset} | {kind} | {full_text} | {ab_text} | "
                f"{fmt_mean_sd(r.get('delta_val_acc_pp_mean'), r.get('delta_val_acc_pp_sample_sd'), digits=3)} | "
                f"{fmt_mean_sd(r.get('delta_val_macro_f1_pp_mean'), r.get('delta_val_macro_f1_pp_sample_sd'), digits=3)} | "
                f"{fmt_mean_sd(r.get('delta_val_ce_mean'), r.get('delta_val_ce_sample_sd'), digits=5)} | {signs} |"
            )
    lines += ["", "### 各组逐种子方向与模块表现", ""]
    for kind, label in (("no_early_cross", "Early Cross"), ("no_late_completion", "Late Completion")):
        lines += [f"#### {label}", ""]
        for dataset in DATASETS:
            group = [r for r in paired if r.get("dataset") == dataset and r.get("ablation") == kind
                     and r.get("status") == "complete"]
            if not group:
                lines.append(f"- {dataset}: 配对记录尚未完整。")
                continue
            group.sort(key=lambda r: int(r["seed"]))
            detail = "; ".join(
                f"seed{r['seed']} ΔAcc {float(r['delta_val_acc_pp']):+.3f} pp, "
                f"ΔF1 {float(r['delta_val_macro_f1_pp']):+.3f} pp, ΔCE {float(r['delta_val_ce']):+.5f}"
                for r in group
            )
            lines.append(f"- {dataset}: {detail}. {delta_consistency_text(group)}")

    gate_rows = gate_summary_rows()
    complete_audits = sum(r.get("status") == "complete" for r in read_rows(GATE_DIAGNOSTICS))
    lines += ["", "## Full Checkpoint 自适应行为审计", "",
              f"已完成 {manifest.get('full_gate_audits_completed', 0)}/9 个 Full Checkpoint 审计（CSV 为每个 Checkpoint 的 Text/Visual 两行；当前完整模态行 {complete_audits}）。", "",
              "### Early Cross（每个 Dataset × Modality 为三个 seed 的均值）", "",
              "| Dataset | Modality | β | Raw RMS | Calibrated RMS | Cross / Neighbor RMS | Private Physical RMS | Collaborative Physical RMS | ΔsP abs mean |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in gate_rows:
        lines.append(
            f"| {r['dataset']} | {r['modality']} | {r['beta_mean']} | {r['early_cross_raw_rms_mean']} | "
            f"{r['early_cross_calibrated_rms_mean']} | {r['early_cross_to_neighbor_rms_mean']} | "
            f"{r['private_physical_contribution_rms_mean']} | {r['collaborative_physical_contribution_rms_mean']} | "
            f"{r['collaborative_vs_private_sP_abs_mean_mean']} |"
        )
    lines += ["", "### Late Completion", "",
              "| Dataset | Modality | Semantic / Same-node / None share | p_accept | Gate mean | Δ ratio | Δ RMS | Cap fraction | Semantic support coverage |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in gate_rows:
        lines.append(
            f"| {r['dataset']} | {r['modality']} | {r['late_selector_semantic_share_mean']} / "
            f"{r['late_selector_same_node_share_mean']} / {r['late_selector_none_share_mean']} | "
            f"{r['p_accept_mean_mean']} | {r['gate_mean_mean']} | {r['late_delta_ratio_mean']} | "
            f"{r['late_delta_rms_mean']} | {r['relative_cap_fraction_mean']} | "
            f"{r['semantic_support_coverage_all_nodes_mean']} |"
        )
    lines += [
        "", "### 行为解读边界", "",
        "- `early_cross_to_neighbor_rms` 接近 1 与实现中的 RMS 校准公式相符；校准按邻域参考 RMS 缩放，不代表模型学到更强的自适应能力。",
        "- `p_accept_mean` 单独较高不能证明 Selector/Gate 失效；应与 None share、Gate、残差幅度和性能配对结果一起看。",
        "- 较大的 Late Residual Ratio 只说明注入量级，不能单独说明信息有害。门控统计是描述性观察，不建立模块与性能变化之间的因果关系。",
        "- Semantic support coverage = semantic_support_nodes / 全图节点数；不使用标签，只描述图结构支持覆盖。没有从缺失值估算或填补字段。",
        "",
        "## 初步人工决策建议", "",
    ]
    recommendations = manifest.get("recommendations", {})
    for kind, label in (("no_early_cross", "Early Cross"), ("no_late_completion", "Late Completion")):
        recommendation = recommendations.get(kind, "DATASET_DEPENDENT_INCONCLUSIVE")
        lines.append(f"- {label}: **`{recommendation}`**。结合三数据集、三种子的方向与波动审阅；本报告不做显著性检验，也不自动触发结构删除。")
    lines += [
        "", "## 执行与协议核验", "",
        f"- 新增训练：{manifest.get('completed_new_cells', 0)}/12；失败 {manifest.get('failed_new_cells', 0)}；资源阻断 {manifest.get('blocked_resource_cells', 0)}；待执行 {len(manifest.get('pending_cells', []))}。",
        f"- Full 行为审计：{manifest.get('full_gate_audits_completed', 0)}/9；每个 Full checkpoint 均先核验 SHA256、同数据指纹、严格重载和历史 Validation 指标。",
        "- 训练 seed43/44；固定 seed42 Split；每个消融从与相同 seed Full 一致的随机初始化开始。Test 标签为 -100，`evaluate_test=false`；Best-Accuracy 是主结果，Best-F1 单独保存。",
        f"- GPU：{manifest.get('gpu', {}).get('physical_gpu', PHYSICAL_GPU)} ({manifest.get('gpu', {}).get('name', 'not_available')})；资源守卫启用；绕过：{manifest.get('gpu', {}).get('guard_bypassed_by_user', False)}。",
        f"- Base Commit：`{BASE_SHA}`；训练源码 Commit：`{manifest.get('training_source_git_sha', 'not_available')}`。Checkpoint 与日志留在服务器 `outputs/mrcge_v13_stage_ablation_3seed/`，没有纳入 Git。",
        "- 全程 NC；不含新的 Full 训练、LP、Test 指标、额外消融或超参数搜索。",
        "",
    ]
    return "\n".join(lines)


def finalize_outputs() -> dict[str, Any]:
    ensure_layout()
    paired, summary = make_paired_and_summary()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else initial_manifest()
    latest = latest_new_runs()
    completed = sum(r.get("status") == "complete" for r in latest.values())
    failed = sum(r.get("status") == "failed" for r in latest.values())
    blocked = sum(r.get("status") == "blocked_resource" for r in latest.values())
    gate_rows = read_rows(GATE_DIAGNOSTICS)
    gate_done = sum(r.get("modality") == "text" and r.get("status") == "complete" for r in gate_rows)
    gate_failed = sum(r.get("modality") == "text" and r.get("status") == "failed" for r in gate_rows)
    gate_blocked = sum(r.get("modality") == "text" and r.get("status") == "blocked_resource" for r in gate_rows)
    pending = [{"dataset": d, "ablation": a, "seed": s}
               for d in DATASETS for a in ABLATIONS for s in NEW_SEEDS
               if latest.get((d, a, s), {}).get("status") != "complete"]
    if blocked or gate_blocked or manifest.get("status") == "blocked_resource":
        status = "blocked_resource"
    elif completed == 12 and failed == 0 and gate_done == 9 and gate_failed == 0:
        status = "complete"
    elif completed == 12 and gate_done < 9:
        status = "audit_partial"
    else:
        status = "partial"
    manifest.update({
        "status": status, "completed_new_cells": completed, "failed_new_cells": failed,
        "blocked_resource_cells": blocked, "pending_cells": pending,
        "successful_units": [f"{d}/seed{s}/{a}" for (d, a, s), r in sorted(latest.items()) if r.get("status") == "complete"],
        "failed_units": [f"{d}/seed{s}/{a}" for (d, a, s), r in sorted(latest.items()) if r.get("status") == "failed"],
        "blocked_resource_units": [f"{d}/seed{s}/{a}" for (d, a, s), r in sorted(latest.items()) if r.get("status") == "blocked_resource"],
        "full_gate_audits_completed": gate_done, "full_gate_audits_failed": gate_failed,
        "full_gate_audits_blocked": gate_blocked, "full_gate_diagnostics_rows": len(gate_rows),
        "all_ablation_runs_rows": len(read_rows(ALL_RUNS)),
        "paired_delta_rows": sum(r.get("status") == "complete" for r in paired),
        "summary_rows": len(summary), "test_evaluation": False,
        "finished_utc": now() if status in ("complete", "blocked_resource", "partial") else "",
        "recommendations": manifest.get("recommendations", {
            "no_early_cross": "DATASET_DEPENDENT_INCONCLUSIVE",
            "no_late_completion": "DATASET_DEPENDENT_INCONCLUSIVE",
        }),
    })
    STUDY.joinpath("REPORT.md").write_text(report_text(manifest, paired, summary), encoding="utf-8")
    manifest["deliverable_sha256"] = {
        path.name: sha256_file(path) for path in (
            STUDY / "REPORT.md", NEW_RUNS, ALL_RUNS, PAIRED, SUMMARY, GATE_DIAGNOSTICS,
        ) if path.is_file()
    }
    manifest["last_updated_utc"] = now()
    json_write(MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--cpu-preflight", action="store_true")
    actions.add_argument("--gpu-preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--finalize", action="store_true")
    actions.add_argument("--worker-train", action="store_true")
    actions.add_argument("--worker-correctness", action="store_true")
    actions.add_argument("--worker-full-audit", action="store_true")
    parser.add_argument("--physical-gpu", type=int, default=PHYSICAL_GPU)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--seed", type=int, choices=ALL_SEEDS)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--attempt-dir")
    args = parser.parse_args()
    ensure_layout()
    if args.cpu_preflight:
        cpu_preflight()
    elif args.gpu_preflight:
        run_gpu_preflight(args.physical_gpu)
    elif args.run:
        run_training(args.physical_gpu)
    elif args.finalize:
        finalize_outputs()
    elif args.worker_correctness:
        correctness_worker(args.physical_gpu)
    elif args.worker_train:
        if not args.dataset or not args.ablation or args.seed not in NEW_SEEDS or args.attempt is None or not args.attempt_dir:
            parser.error("--worker-train requires dataset, ablation, seed43/44, attempt and attempt-dir")
        worker_train(args.dataset, args.ablation, args.seed, args.attempt, args.attempt_dir, args.physical_gpu)
    elif args.worker_full_audit:
        if not args.dataset or args.seed not in ALL_SEEDS:
            parser.error("--worker-full-audit requires dataset and seed42/43/44")
        full_audit_worker(args.dataset, args.seed, args.physical_gpu)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise











