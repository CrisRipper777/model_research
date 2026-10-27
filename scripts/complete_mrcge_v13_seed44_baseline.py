#!/usr/bin/env python3
"""Complete the two missing V1.3-A seed-44 NC baseline cells.

The original V1.3 runner remains untouched. This script reuses its frozen
configuration/data helpers and the existing V1.3 shared-GPU ownership guard.
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
from omegaconf import OmegaConf

from scripts import run_mrcge_v12_confirmation as v12_utils
from scripts import run_mrcge_v13_screen as v13
from scripts import run_mrcge_v13_two_hop_cross_screen as history_utils
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed


STUDY = ROOT / "research/mrcge_v13_seed44_completion"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_seed44_baseline_completion"
RUNS = OUTPUT / "runs"
BASE_SHA = "67bc4061617354dcde3ab99551de73263964af37"
BASE_BRANCH = "exp/mrcge_v13_view_global_router_ablation"
BRANCH = "exp/mrcge_v13_seed44_baseline_completion"
PHYSICAL_GPU = 1
DATASETS = ("Movies", "Grocery", "ele-fashion")
NEW_CELLS = (("Grocery", 44), ("ele-fashion", 44))
PROTOCOL = "unified_full_graph_nc_v1"
MODEL_NAME = "mrcge_v13"
CONDITION = "M_repaired"
VARIANT = "joint_view"
V12_SELECTED = ROOT / "research/mrcge_v12_confirmation/data/combined_selected_runs.csv"
HISTORICAL_FIELDS = (
    "dataset", "model", "condition", "variant", "seed", "record_origin", "source_record",
    "source_git_sha", "source_base_sha", "status", "selection_checkpoint", "selection_rule",
    "epochs_completed", "best_acc_epoch", "best_f1_epoch", "val_acc", "val_macro_f1", "val_ce",
    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1",
    "train_ce", "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap",
    "model_parameters", "classifier_parameters", "total_parameters", "metrics_source",
    "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint",
    "split_file_path", "split_file_sha256", "semantic_cache_sha256", "candidate_fingerprint",
    "v7b_fingerprint", "initial_model_sha256", "initial_classifier_sha256", "config_sha256",
    "test_evaluation", "test_labels_masked", "checkpoint_reload_verified", "validation_rng_unchanged",
    "protocol_deviation",
)
METRIC_NAMES = (
    "val_acc", "val_macro_f1", "val_ce", "train_acc", "train_macro_f1", "train_ce",
    "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap",
)
FROZEN_PATHS = (
    "src/models/mrcge_v13.py", "src/tasks/nc.py", "scripts/run_mrcge_v13_screen.py",
    "scripts/run_mrcge_v13_two_hop_cross_screen.py", "scripts/run_mrcge_v12_confirmation.py",
    "scripts/run_hybrid_cge_paper_evidence_v1.py", "configs/model/mrcge_v13.yaml",
    "configs/task/nc.yaml", "configs/config.yaml", "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml", "configs/dataset/ele-fashion.yaml", "src/data/loaders.py",
    "src/data/splits.py", "src/data/graph_utils.py", "src/data/semantic_candidates_v7b.py",
    "src/data/types.py", "src/utils/seeds.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                default=json_default), encoding="utf-8")
    temp.replace(path)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def git(*args: str, text: bool = True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=text)


def source_sha() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def current_branch() -> str:
    return git("branch", "--show-current").stdout.strip()


def file_hashes() -> dict[str, str]:
    names = [*FROZEN_PATHS, "scripts/complete_mrcge_v13_seed44_baseline.py"]
    return {name: sha256_file(ROOT / name) for name in names}


def verify_source_freeze() -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}, found {current_branch()}")
    if git("merge-base", "--is-ancestor", BASE_SHA, "HEAD").returncode:
        raise RuntimeError(f"HEAD does not descend from registered base {BASE_SHA}")
    for relative in FROZEN_PATHS:
        base = git("show", f"{BASE_SHA}:{relative}", text=False).stdout
        if hashlib.sha256(base).hexdigest() != sha256_file(ROOT / relative):
            raise RuntimeError(f"frozen model/trainer/config/data file differs from base: {relative}")


def assert_cfg(cfg, dataset: str, seed: int) -> dict[str, Any]:
    expected = {
        "model": {
            "name": MODEL_NAME, "condition": CONDITION, "v13_variant": VARIANT,
            "sender_private_ratio": 0.0, "physical_hops": 3, "semantic_hops": 1,
            "hidden_dim": 256, "num_experts": 4, "top_k": 2,
            "semantic_knn_k": 8, "dropout": 0.2, "late_cross_dropout": 0.15,
            "balance_weight": 0.01,
        },
        "task": {
            "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.0001,
            "epochs": 300, "patience": 30, "evaluate_test": False,
            "aux_weight": 1.0, "unimodal_weight": 0.3,
        },
    }
    for key, value in expected["model"].items():
        actual = cfg.model.get(key)
        if isinstance(value, float):
            ok = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            ok = actual == value
        if not ok:
            raise RuntimeError(f"{dataset}/seed{seed} model config {key}: expected {value!r}, got {actual!r}")
    for key, value in expected["task"].items():
        full_key = f"task.loss.{key}" if key in ("aux_weight", "unimodal_weight") else f"task.{key}"
        actual = cfg.task.loss.get(key) if key in ("aux_weight", "unimodal_weight") else cfg.task.get(key)
        if isinstance(value, float):
            ok = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            ok = actual == value
        if not ok:
            raise RuntimeError(f"{dataset}/seed{seed} config {full_key}: expected {value!r}, got {actual!r}")
    if int(cfg.seed) != seed or cfg.dataset.auto_generate_splits is not False:
        raise RuntimeError(f"{dataset}/seed{seed} must use training RNG seed {seed} and frozen splits")
    if bool(cfg.task.loss.get("class_balanced_main_ce", False)):
        raise RuntimeError(f"{dataset}/seed{seed}: class-balanced main CE is outside the V1.3-A protocol")
    if cfg.task.get("record_extended_validation") is not True or cfg.task.get("record_train_ce_separately") is not True:
        raise RuntimeError(f"{dataset}/seed{seed}: Train/Validation metrics are not configured for auditing")
    return expected


def cache_and_data(dataset: str, *, device: str):
    cache = v13.cache_entries()[dataset]
    cfg = v13.make_cfg(dataset, VARIANT, 44, device, cache)
    assert_cfg(cfg, dataset, 44)
    raw = load_mag_data(cfg, "nc", 44)
    safe = v12_utils.mask_test_labels(raw)
    fingerprints = v13.data_utils.dataset_fingerprints(raw)
    split_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
    if not split_text:
        raise RuntimeError(f"{dataset}: resolved frozen split path missing")
    split = Path(split_text).resolve()
    if not split.is_file():
        raise RuntimeError(f"{dataset}: frozen split missing: {split}")
    aligned = v13.assert_data_alignment(dataset, 44, safe, cache, fingerprints, sha256_file(split))
    refs = [row for row in read_rows(V12_SELECTED)
            if row.get("dataset") == dataset and row.get("condition") == CONDITION
            and row.get("seed") == "44" and row.get("selected_for_combined", "").lower() == "true"]
    if len(refs) != 1:
        raise RuntimeError(f"{dataset}: expected one V1.2 M_repaired seed44 data reference")
    ref = refs[0]
    if ref.get("split_file_sha256") and sha256_file(split) != ref["split_file_sha256"]:
        raise RuntimeError(f"{dataset}: split file SHA differs from V1.2 seed44 reference")
    if bool((safe.y[raw.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
        raise RuntimeError(f"{dataset}: test labels must be -100 and evaluate_test=false before Trainer")
    if not torch.equal(safe.y[raw.train_idx], raw.y[raw.train_idx]) or not torch.equal(safe.y[raw.val_idx], raw.y[raw.val_idx]):
        raise RuntimeError(f"{dataset}: masking modified Train or Validation labels")
    partition = torch.cat([raw.train_idx, raw.val_idx, raw.test_idx])
    if partition.numel() != raw.num_nodes or torch.unique(partition).numel() != raw.num_nodes:
        raise RuntimeError(f"{dataset}: Train/Validation/Test indices do not partition nodes")
    record = {
        "dataset": dataset, "seed": 44, "model": MODEL_NAME, "condition": CONDITION,
        "variant": VARIANT, "training_seed": 44, "split_seed": 42,
        "num_nodes": int(safe.num_nodes), "num_classes": int(safe.num_classes),
        "input_dim": int(safe.input_dim), "train_nodes": int(raw.train_idx.numel()),
        "validation_nodes": int(raw.val_idx.numel()), "test_nodes": int(raw.test_idx.numel()),
        "test_labels_masked": True, "test_evaluation": False,
        "split_file_path": str(split), "split_file_sha256": sha256_file(split),
        **aligned,
    }
    return cfg, raw, safe, cache, fingerprints, split, record


def data_info(data) -> dict[str, int]:
    return {"input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
            "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def init_pair(cfg, safe, cache, seed: int, device: torch.device):
    set_seed(seed)
    model = build_model(cfg, data_info(safe))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    return model, classifier, v12_utils.state_hash(model), v12_utils.state_hash(classifier)


def historical_rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in (42, 43, 44):
            if seed == 44 and dataset != "Movies":
                continue
            row = history_utils.historical_v13(dataset, seed)
            checkpoints = history_utils.verify_reference_checkpoints(row, f"{dataset}/seed{seed}")
            accuracy_path = Path(checkpoints["best_accuracy"]["path"])
            payload = torch.load(accuracy_path, map_location="cpu", weights_only=False)
            history = payload.get("run_metadata", {}).get("epoch_history", [])
            epoch = int(payload["epoch"])
            epoch_row = next((item for item in history if int(item.get("epoch", -1)) == epoch), None)
            if epoch_row is None:
                raise RuntimeError(f"{dataset}/seed{seed}: Best-Accuracy checkpoint has no matching epoch history")
            vals = {"val_acc": float(row["val_acc"]), "val_macro_f1": float(row["val_macro_f1"]),
                    "val_ce": float(row["val_ce"])}
            for key, source_key in (("val_acc", "val_acc"), ("val_macro_f1", "val_macro_f1"), ("val_ce", "val_ce")):
                if abs(float(epoch_row[source_key]) - vals[key]) > 3e-6:
                    raise RuntimeError(f"{dataset}/seed{seed}: checkpoint epoch history mismatches stored {key}")
            train = {
                "train_acc": float(epoch_row["train_accuracy_eval"]),
                "train_macro_f1": float(epoch_row["train_macro_f1_eval"]),
                "train_ce": float(epoch_row["train_ce_eval"]),
            }
            if not all(math.isfinite(value) for value in (*vals.values(), *train.values())):
                raise RuntimeError(f"{dataset}/seed{seed}: non-finite historical metrics")
            param = payload.get("run_metadata", {})
            best_f1 = {
                "best_f1_val_acc": float(row["best_f1_val_acc"]),
                "best_f1_val_macro_f1": float(row["best_f1_val_macro_f1"]),
                "best_f1_val_ce": float(row["best_f1_val_ce"]),
            }
            result.append({
                "dataset": dataset, "model": MODEL_NAME, "condition": CONDITION, "variant": VARIANT,
                "seed": seed, "record_origin": "reused_historical",
                "source_record": ("research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv"
                                  if dataset == "Movies" and seed == 44 else "research/mrcge_v13/data/runs.csv"),
                "source_git_sha": row.get("source_git_sha", ""), "source_base_sha": row.get("base_sha", ""),
                "status": "complete", "selection_checkpoint": "best_validation_accuracy",
                "selection_rule": "Validation Accuracy; min_delta=1e-4; patience=30",
                "epochs_completed": int(param.get("epochs_completed", len(history))),
                "best_acc_epoch": epoch, "best_f1_epoch": int(row["best_f1_epoch"]),
                **vals, **best_f1, **train,
                "train_val_acc_gap": train["train_acc"] - vals["val_acc"],
                "train_val_macro_f1_gap": train["train_macro_f1"] - vals["val_macro_f1"],
                "train_val_ce_gap": train["train_ce"] - vals["val_ce"],
                "model_parameters": int(param.get("model_parameters", 0)),
                "classifier_parameters": int(param.get("classifier_parameters", 0)),
                "total_parameters": int(param.get("model_parameters", 0)) + int(param.get("classifier_parameters", 0)),
                "metrics_source": "Best-Accuracy checkpoint run_metadata.epoch_history at checkpoint epoch",
                "checkpoint_path": checkpoints["best_accuracy"]["path"],
                "checkpoint_sha256": checkpoints["best_accuracy"]["sha256"],
                "best_f1_checkpoint_path": checkpoints["best_f1"]["path"],
                "best_f1_checkpoint_sha256": checkpoints["best_f1"]["sha256"],
                "feature_fingerprint": row.get("feature_fingerprint", ""),
                "dataset_fingerprint": row.get("dataset_fingerprint", ""),
                "physical_edge_fingerprint": row.get("physical_edge_fingerprint", ""),
                "split_fingerprint": row.get("split_fingerprint", ""),
                "split_file_path": str(history_utils.fixed_split_path(dataset)),
                "split_file_sha256": row.get("split_file_sha256", ""),
                "semantic_cache_sha256": row.get("semantic_cache_sha256", ""),
                "candidate_fingerprint": row.get("candidate_fingerprint", ""),
                "v7b_fingerprint": row.get("v7b_fingerprint", ""),
                "test_evaluation": False, "test_labels_masked": True,
                "checkpoint_reload_verified": True, "validation_rng_unchanged": row.get("validation_rng_unchanged", "true"),
                "protocol_deviation": "historical checkpoint reused; not retrained in this task",
            })
    if len(result) != 7 or len({(r["dataset"], r["seed"]) for r in result}) != 7:
        raise RuntimeError(f"expected 7 unique V1.3-A historical cells, got {len(result)}")
    return result


def cpu_preflight() -> dict[str, Any]:
    verify_source_freeze()
    histories = historical_rows()
    data_rows, init_rows = [], []
    for dataset, seed in NEW_CELLS:
        cfg, raw, safe, cache, fingerprints, split, row = cache_and_data(dataset, device="cpu")
        init = v12_utils.initial_state_audit(cfg, safe, cache, torch.device("cpu"))
        model, _, model_hash, classifier_hash = init_pair(cfg, safe, cache, seed, torch.device("cpu"))
        if model_hash != init["initial_model_sha256"] or classifier_hash != init["initial_classifier_sha256"]:
            raise RuntimeError(f"{dataset}/seed{seed}: model initialization audit is inconsistent")
        row["initial_model_sha256"] = model_hash
        row["initial_classifier_sha256"] = classifier_hash
        row["model_parameters"] = int(sum(p.numel() for p in model.parameters()))
        row["classifier_parameters"] = int(safe.num_classes * model.out_dim + safe.num_classes)
        data_rows.append(row)
        init_rows.append({"dataset": dataset, "seed": seed, **init,
                          "model_parameters": row["model_parameters"],
                          "classifier_parameters": row["classifier_parameters"]})
        del raw, safe, model
    result = {
        "status": "pass", "recorded_utc": now(), "branch": current_branch(),
        "source_git_sha": source_sha(), "base_sha": BASE_SHA, "source_file_sha256s": file_hashes(),
        "protocol": protocol_payload(), "new_cell_data": data_rows,
        "new_cell_initialization": init_rows, "historical_checkpoint_rows_verified": len(histories),
        "historical_cells": [{"dataset": r["dataset"], "seed": r["seed"],
                              "checkpoint_sha256": r["checkpoint_sha256"],
                              "best_f1_checkpoint_sha256": r["best_f1_checkpoint_sha256"]} for r in histories],
        "test_evaluation": False, "link_prediction": False,
    }
    json_write(DATA / "preflight.json", result)
    return result


def protocol_payload() -> dict[str, Any]:
    return {
        "model": MODEL_NAME, "condition": CONDITION, "variant": VARIANT,
        "sender_private_ratio": 0.0, "physical_hops": 3, "semantic_hops": 1,
        "num_experts": 4, "top_k": 2, "hidden_dim": 256,
        "semantic_cache": "V7B.1", "semantic_knn_k": 8,
        "private_ce_weight": 0.3, "balance_loss_weight": 0.01, "trainer_aux_weight": 1.0,
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
        "backbone_dropout": 0.2, "late_dropout": 0.15,
        "max_epochs": 300, "patience": 30, "early_stop_min_epoch": 30,
        "early_stop_min_delta": 0.0001, "split_seed": 42, "training_seeds": [44],
        "selection": "Best-Validation-Accuracy checkpoint; Best-F1 checkpoint saved separately",
        "evaluate_test": False, "test_label_mask": -100,
    }


def current_gpu_baseline() -> set[int]:
    snap = v13.smi_snapshot()
    device = next((item for item in snap["devices"] if item["physical_index"] == PHYSICAL_GPU), None)
    if device is None:
        raise RuntimeError(f"physical GPU {PHYSICAL_GPU} unavailable")
    return {int(app["pid"]) for app in snap["compute_processes"] if app["gpu_uuid"] == device["uuid"]}


def gpu_guard_bypassed() -> bool:
    return os.environ.get("MRCGE_SEED44_BASELINE_SKIP_GPU_GUARD") == "1"


def gpu_check(phase: str, baseline: set[int], allowed: set[int], required_free_mib: int) -> dict[str, Any]:
    if not gpu_guard_bypassed():
        return v13.guard_gpu(PHYSICAL_GPU, baseline, allowed, phase=phase,
                             required_free_mib=required_free_mib)
    snapshot = v13.smi_snapshot()
    snapshot["guard"] = {"status": "bypassed_by_user", "phase": phase,
                         "physical_gpu": PHYSICAL_GPU, "required_free_mib": required_free_mib}
    return snapshot


def guarded_child(action: str, dataset: str, baseline: set[int], *, required_free_mib: int = 8192) -> dict[str, Any]:
    launch = gpu_check("before_launch", baseline, set(), required_free_mib)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    env["MRCGE_SEED44_BASELINE_ACTION"] = action
    env["MRCGE_BASELINE_GPU_PIDS"] = ",".join(map(str, sorted(baseline)))
    if action == "preflight":
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-preflight", "--dataset", dataset]
    elif action == "train":
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-train", "--dataset", dataset]
    else:
        raise ValueError(action)
    log_path = OUTPUT / "guard_logs" / f"{action}_{dataset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(3)
            if not gpu_guard_bypassed():
                try:
                    v13.guard_gpu(PHYSICAL_GPU, baseline, {proc.pid}, phase="during_run", required_free_mib=2048)
                except Exception as exc:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait(timeout=10)
                    raise RuntimeError(f"shared-GPU guard stopped the owned {action} worker: {exc}") from exc
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"{action} worker exited {proc.returncode}; log={log_path}\n{tail}")
    return {"launch_guard": launch, "log_path": str(log_path)}


def worker_preflight(dataset: str) -> None:
    if os.environ.get("MRCGE_SEED44_BASELINE_ACTION") != "preflight" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("preflight must be launched through the physical GPU 1 guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or physical GPU isolation failed")
    baseline = {int(pid) for pid in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    gpu_check("before_launch", baseline, {os.getpid()}, 8192)
    cfg, raw, safe, cache, fingerprints, split, row = cache_and_data(dataset, device="cuda:0")
    device = torch.device("cuda:0")
    model, classifier, model_hash, classifier_hash = init_pair(cfg, safe, cache, 44, device)
    optimizer = torch.optim.AdamW([*model.parameters(), *classifier.parameters()], lr=0.001, weight_decay=0.0001)
    x, edge_index, labels = safe.x.to(device), safe.edge_index.to(device), safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.train(); classifier.train()
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux, info = model(x, edge_index, return_details=False)
    main_ce = nn.functional.cross_entropy(classifier(z[train_idx]), labels[train_idx])
    loss, private_ce = _compose_full_graph_loss(
        model=model, info=info, main_ce=main_ce, aux_loss=aux, aux_weight=1.0,
        unimodal_weight=0.3, labels=labels, train_idx=train_idx,
    )
    if not torch.isfinite(loss) or not torch.isfinite(z).all():
        raise RuntimeError("initialization preflight produced non-finite embedding/loss")
    loss.backward()
    params = [*model.parameters(), *classifier.parameters()]
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params):
        raise RuntimeError("initialization preflight produced non-finite gradients")
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    record = {
        **row, "status": "pass", "preflight_only": True, "recorded_utc": now(),
        "gpu_guard_status": "bypassed_by_user" if gpu_guard_bypassed() else "active",
        "physical_gpu": 1, "gpu_name": torch.cuda.get_device_name(device),
        "initial_model_sha256": model_hash, "initial_classifier_sha256": classifier_hash,
        "total_loss": float(loss.detach().cpu()), "main_ce": float(main_ce.detach().cpu()),
        "private_ce": float(private_ce.detach().cpu()), "balance_aux": float(aux.detach().cpu()),
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "test_evaluation": False, "test_labels_masked": bool((safe.y[safe.test_idx] == -100).all()),
    }
    if not all(math.isfinite(float(record[key])) for key in ("total_loss", "main_ce", "private_ce", "balance_aux")):
        raise RuntimeError("initialization preflight produced non-finite loss components")
    json_write(OUTPUT / "preflight" / f"{dataset}.json", record)
    del model, classifier, optimizer, raw, safe, z, info
    torch.cuda.empty_cache()


def worker_train(dataset: str) -> None:
    if os.environ.get("MRCGE_SEED44_BASELINE_ACTION") != "train" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("formal worker must be launched through the physical GPU 1 guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA device isolation failed")
    pre = read_json(OUTPUT / "preflight" / f"{dataset}.json", {})
    if pre.get("status") != "pass":
        raise RuntimeError(f"missing successful GPU initialization preflight for {dataset}")
    baseline = {int(pid) for pid in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if pid}
    required = max(8192, math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024)) + 2048)
    gpu_check("before_launch", baseline, {os.getpid()}, required)
    cfg, raw, safe, cache, fingerprints, split, row = cache_and_data(dataset, device="cuda:0")
    attempt_dir = RUNS / dataset / "joint_view" / "seed_44" / "attempt_01"
    if attempt_dir.exists():
        raise RuntimeError(f"refusing to reuse existing attempt directory {attempt_dir}")
    attempt_dir.mkdir(parents=True, exist_ok=False)
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    model_file = ROOT / "src/models/mrcge_v13.py"
    trainer_file = ROOT / "src/tasks/nc.py"
    src = source_sha()
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3-A Seed44 Baseline Completion", "dataset": dataset,
        "model": MODEL_NAME, "condition": CONDITION, "variant": VARIANT,
        "sender_private_ratio": 0.0, "seed": 44, "split_seed": 42,
        "source_git_sha": src, "base_sha": BASE_SHA, "protocol_version": PROTOCOL,
        "model_source_sha256": sha256_file(model_file), "trainer_source_sha256": sha256_file(trainer_file),
        "config_sha256": config_hash, "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": sha256_file(split),
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "test_evaluation": False, "test_label_values_indexed": False, "link_prediction": False,
    }
    _, _, init_model_hash, init_classifier_hash = init_pair(cfg, safe, cache, 44, torch.device("cuda:0"))
    logger = logging.getLogger(f"mrcge_seed44.{dataset}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(attempt_dir / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    json_write(attempt_dir / "attempt_started.json", {
        "dataset": dataset, "seed": 44, "split_seed": 42, "started_utc": now(),
        "source_git_sha": src, "base_sha": BASE_SHA, "config_sha256": config_hash,
        "physical_gpu": 1, "required_free_mib": required, "test_evaluation": False,
    })
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        run_nc(cfg, safe, device, logger)
        gpu_check("during_run", baseline, {os.getpid()}, 2048)
        packed = read_json(Path(cfg.task.run_metrics_path), {})
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [44]:
            raise RuntimeError("Trainer protocol or seed mismatch")
        metadata = packed["runs"][0]["metadata"]
        selection = metadata["dual_checkpoint_selection"]
        initial_audit = metadata["initial_parameter_audit"]
        if initial_audit["model_state_sha256"] != init_model_hash or initial_audit["classifier_state_sha256"] != init_classifier_hash:
            raise RuntimeError("Trainer initialization differs from the audited seed44 model/head")
        acc_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        if not acc_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Best-Accuracy or Best-F1 checkpoint missing")
        acc = v12_utils.validation_logits_audit(cfg, safe, cache, acc_path, device, details=False)
        f1 = v12_utils.validation_logits_audit(cfg, safe, cache, f1_path, device, details=False)
        if not (acc["rng_unchanged"] and f1["rng_unchanged"]):
            raise RuntimeError("checkpoint reload changed RNG state")
        for chosen, evaluated, label in ((selection["best_accuracy"], acc, "accuracy"),
                                         (selection["best_macro_f1"], f1, "macro-F1")):
            for field, key in (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce")):
                if abs(float(chosen["metrics"][field]) - float(evaluated["val"][key])) > 3e-6:
                    raise RuntimeError(f"reloaded Best-{label} {field} differs from the Trainer selection")
        history = metadata.get("epoch_history", [])
        required_history = ("train_loss", "train_accuracy_eval", "train_macro_f1_eval", "train_ce_eval",
                            "val_acc", "val_macro_f1", "val_ce")
        if not history or not all(all(field in item and math.isfinite(float(item[field])) for field in required_history)
                                  for item in history):
            raise RuntimeError("epoch history is missing required finite Train/Validation metrics")
        acc_epoch = int(selection["best_accuracy"]["epoch"])
        f1_epoch = int(selection["best_macro_f1"]["epoch"])
        if int(torch.load(acc_path, map_location="cpu", weights_only=False)["epoch"]) != acc_epoch:
            raise RuntimeError("Best-Accuracy path does not contain the Best-Accuracy epoch")
        if int(torch.load(f1_path, map_location="cpu", weights_only=False)["epoch"]) != f1_epoch:
            raise RuntimeError("Best-F1 path does not contain the Best-F1 epoch")
        # Recheck each selector from the recorded validation trajectory.
        acc_best = max(float(item["val_acc"]) for item in history)
        if abs(float(selection["best_accuracy"]["metrics"]["val_acc"]) - acc_best) > 1.1e-4:
            raise RuntimeError("Best-Accuracy selector does not match the validation trajectory")
        f1_best = max(float(item["val_macro_f1"]) for item in history)
        if abs(float(selection["best_macro_f1"]["metrics"]["val_macro_f1"]) - f1_best) > 1e-7:
            raise RuntimeError("Best-F1 selector does not match the validation trajectory")
        earliest_f1_epoch = min(int(item["epoch"]) for item in history if float(item["val_macro_f1"]) == f1_best)
        if f1_epoch != earliest_f1_epoch:
            raise RuntimeError("Best-F1 selector did not keep the earliest epoch on a tie")
        if bool((safe.y[safe.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
            raise RuntimeError("Test evaluation or label masking boundary failed")
        train, val, f1_val = acc["train"], acc["val"], f1["val"]
        record = {
            "dataset": dataset, "model": MODEL_NAME, "condition": CONDITION, "variant": VARIANT,
            "seed": 44, "record_origin": "new_training", "source_record": "this task; seed44 completion",
            "source_git_sha": src, "source_base_sha": BASE_SHA, "status": "complete",
            "selection_checkpoint": "best_validation_accuracy",
            "selection_rule": selection["best_accuracy"]["rule"], "epochs_completed": len(history),
            "best_acc_epoch": acc_epoch, "best_f1_epoch": f1_epoch,
            "val_acc": val["acc"], "val_macro_f1": val["macro_f1"], "val_ce": val["ce"],
            "best_f1_val_acc": f1_val["acc"], "best_f1_val_macro_f1": f1_val["macro_f1"],
            "best_f1_val_ce": f1_val["ce"], "train_acc": train["acc"],
            "train_macro_f1": train["macro_f1"], "train_ce": train["ce"],
            "train_val_acc_gap": train["acc"] - val["acc"],
            "train_val_macro_f1_gap": train["macro_f1"] - val["macro_f1"],
            "train_val_ce_gap": train["ce"] - val["ce"],
            "model_parameters": int(metadata["model_parameters"]),
            "classifier_parameters": int(metadata["classifier_parameters"]),
            "total_parameters": int(metadata["model_parameters"]) + int(metadata["classifier_parameters"]),
            "metrics_source": "Best-Accuracy checkpoint reloaded; Train/Validation labels only",
            "checkpoint_path": str(acc_path), "checkpoint_sha256": sha256_file(acc_path),
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": sha256_file(f1_path),
            "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"], "split_file_path": str(split),
            "split_file_sha256": sha256_file(split), "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_model_sha256": init_model_hash, "initial_classifier_sha256": init_classifier_hash,
            "config_sha256": config_hash, "test_evaluation": False, "test_labels_masked": True,
            "checkpoint_reload_verified": True, "validation_rng_unchanged": True,
            "wall_seconds": float(metadata.get("training_wall_seconds", time.perf_counter() - started)),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "gpu_name": metadata.get("cuda_device_name"), "attempt_dir": str(attempt_dir),
            "protocol_deviation": "none",
            "gpu_guard_status": "bypassed_by_user" if gpu_guard_bypassed() else "active",
        }
        if not all(math.isfinite(float(record[key])) for key in METRIC_NAMES):
            raise RuntimeError("Best-Accuracy metrics contain NaN/Inf")
        json_write(attempt_dir / "worker_result.json", record)
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(), **record})
    except Exception as exc:
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(),
                    "error": repr(exc), "traceback": traceback.format_exc(), "test_evaluation": False})
        raise
    finally:
        torch.cuda.empty_cache()


def run_gpu_preflights() -> None:
    pre = read_json(DATA / "preflight.json", {})
    if pre.get("status") != "pass" or pre.get("source_file_sha256s") != file_hashes():
        raise RuntimeError("current-source CPU/config/data preflight must pass first")
    results = []
    for dataset, _ in NEW_CELLS:
        baseline = current_gpu_baseline()
        try:
            outcome = guarded_child("preflight", dataset, baseline)
            record = read_json(OUTPUT / "preflight" / f"{dataset}.json", {})
            if record.get("status") != "pass":
                raise RuntimeError(f"{dataset}: GPU initialization check did not record pass")
            results.append({"dataset": dataset, "status": "pass", "preflight_only": True,
                            "guard": outcome, "peak_reserved_bytes": record["peak_reserved_bytes"]})
        except Exception as exc:
            results.append({"dataset": dataset, "status": "blocked_resource" if "guard" in repr(exc).lower() or "cuda" in repr(exc).lower() else "failed",
                            "error": repr(exc)})
            json_write(DATA / "gpu_preflight.json", {"status": "incomplete", "updated_utc": now(),
                        "source_git_sha": source_sha(), "cells": results, "training_started": False})
            raise
    json_write(DATA / "gpu_preflight.json", {"status": "pass", "completed_utc": now(),
                "source_git_sha": source_sha(), "physical_gpu": 1, "cells": results,
                "resource_guard_status": "bypassed_by_user" if gpu_guard_bypassed() else "active",
                "resource_guard_override_reason": "explicit user instruction" if gpu_guard_bypassed() else None,
                "training_started": False, "test_evaluation": False})


def run_formal_cells() -> None:
    gpu = read_json(DATA / "gpu_preflight.json", {})
    if gpu.get("status") != "pass" or len(gpu.get("cells", [])) != 2:
        raise RuntimeError("both GPU initialization preflights must pass before formal training")
    records: dict[str, Any] = read_json(DATA / "new_training_records.json", {}) or {}
    for dataset, _ in NEW_CELLS:
        if records.get(dataset, {}).get("status") == "complete":
            continue
        baseline = current_gpu_baseline()
        pre = read_json(OUTPUT / "preflight" / f"{dataset}.json", {})
        required = max(8192, math.ceil(int(pre["peak_reserved_bytes"]) / (1024 * 1024)) + 2048)
        try:
            outcome = guarded_child("train", dataset, baseline, required_free_mib=required)
            attempt = RUNS / dataset / "joint_view" / "seed_44" / "attempt_01"
            row = read_json(attempt / "worker_result.json", {})
            if row.get("status") != "complete":
                raise RuntimeError(f"{dataset}: worker produced no complete result")
            records[dataset] = {**row, "guard": outcome}
            json_write(DATA / "new_training_records.json", records)
        except Exception as exc:
            records[dataset] = {"dataset": dataset, "seed": 44,
                                "status": "blocked_resource" if "guard" in repr(exc).lower() or "memory" in repr(exc).lower() or "cuda" in repr(exc).lower() else "failed",
                                "error": repr(exc), "test_evaluation": False}
            json_write(DATA / "new_training_records.json", records)
            raise


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def finalize() -> None:
    verify_source_freeze()
    pre = read_json(DATA / "preflight.json", {})
    records = read_json(DATA / "new_training_records.json", {}) or {}
    if pre.get("status") != "pass" or not all(records.get(dataset, {}).get("status") == "complete" for dataset, _ in NEW_CELLS):
        raise RuntimeError("cannot finalize without both successful seed44 training records")
    histories = historical_rows()
    fresh = [records[dataset] for dataset, _ in NEW_CELLS]
    rows = histories + fresh
    expected = {(dataset, seed) for dataset in DATASETS for seed in (42, 43, 44)}
    actual = {(row["dataset"], int(row["seed"])) for row in rows}
    if len(rows) != 9 or actual != expected:
        raise RuntimeError(f"expected complete 3x3 baseline, got {sorted(actual)}")
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for key in ("feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint",
                    "split_fingerprint", "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint"):
            if len({row.get(key, "") for row in dataset_rows}) != 1 or not dataset_rows[0].get(key, ""):
                raise RuntimeError(f"{dataset}: seed42/43/44 disagree on {key}")
    write_rows(DATA / "runs.csv", rows, HISTORICAL_FIELDS)
    summary = []
    for dataset in DATASETS:
        cell_rows = [row for row in rows if row["dataset"] == dataset]
        if len(cell_rows) != 3:
            raise RuntimeError(f"{dataset}: expected seeds 42/43/44 exactly once")
        item: dict[str, Any] = {"dataset": dataset, "n": 3, "seeds": "42;43;44",
                                "model": MODEL_NAME, "condition": CONDITION, "variant": VARIANT}
        for metric in METRIC_NAMES:
            vals = [float(row[metric]) for row in cell_rows]
            item[f"{metric}_mean"] = sum(vals) / len(vals)
            item[f"{metric}_sd"] = sample_sd(vals)
        summary.append(item)
    summary_fields = ("dataset", "n", "seeds", "model", "condition", "variant",
                      *(name for metric in METRIC_NAMES for name in (f"{metric}_mean", f"{metric}_sd")))
    write_rows(DATA / "summary.csv", summary, summary_fields)
    all_finite = all(math.isfinite(float(row[key])) for row in rows for key in METRIC_NAMES)
    if not all_finite:
        raise RuntimeError("final table contains NaN/Inf")
    fingerprints = {f"{row['dataset']}/seed{row['seed']}": {
        key: row.get(key, "") for key in ("feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint",
                                            "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
                                            "candidate_fingerprint", "v7b_fingerprint")}
        for row in rows}
    gpu_preflight = read_json(DATA / "gpu_preflight.json", {})
    resource_guard_status = gpu_preflight.get("resource_guard_status", "active")
    manifest = {
        "study": "MR-CGE V1.3-A Main Model Seed44 Baseline Completion",
        "status": "complete", "created_utc": pre.get("recorded_utc"), "completed_utc": now(),
        "branch": BRANCH, "base_branch": BASE_BRANCH, "base_sha": BASE_SHA,
        "training_source_commit": pre.get("source_git_sha"),
        "training_source_file_sha256s": pre.get("source_file_sha256s"),
        "protocol": protocol_payload(), "new_training_units": [
            {"dataset": row["dataset"], "seed": row["seed"], "status": row["status"],
             "source_git_sha": row["source_git_sha"], "checkpoint_path": row["checkpoint_path"],
             "checkpoint_sha256": row["checkpoint_sha256"], "best_f1_checkpoint_path": row["best_f1_checkpoint_path"],
             "best_f1_checkpoint_sha256": row["best_f1_checkpoint_sha256"],
             "metrics": {name: row[name] for name in METRIC_NAMES},
             "best_acc_epoch": row["best_acc_epoch"], "best_f1_epoch": row["best_f1_epoch"],
             "protocol_deviation": row["protocol_deviation"]} for row in fresh],
        "reused_historical_units": [{"dataset": row["dataset"], "seed": row["seed"],
             "source_record": row["source_record"], "source_git_sha": row["source_git_sha"],
             "checkpoint_sha256": row["checkpoint_sha256"],
             "best_f1_checkpoint_sha256": row["best_f1_checkpoint_sha256"],
             "metrics_source": row["metrics_source"]} for row in histories],
        "data_and_cache_fingerprints": fingerprints,
        "gpu": {"physical_gpu": 1, "preflight": gpu_preflight},
        "resource_guard_status": resource_guard_status,
        "checkpoint_reloads_verified": True, "finite_metrics": True,
        "test_evaluation": False, "link_prediction": False,
    }
    json_write(DATA / "manifest.json", manifest)
    write_report(rows, summary)


def report_mean_sd(mean: float, sd: float, metric: str) -> str:
    if "acc" in metric or "f1" in metric:
        return f"{100.0 * mean:.2f} ± {100.0 * sd:.2f}%"
    return f"{mean:.4f} ± {sd:.4f}"


def write_report(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    by_dataset = {row["dataset"]: row for row in summary}
    lines = [
        "# MR-CGE V1.3-A 主模型 Seed44 基线补齐", "",
        "## 范围与协议", "",
        "本补充实验仅完成 Grocery/seed44 与 ele-fashion/seed44 两个节点分类单元。正式模型为 `mrcge_v13`、`M_repaired`、`joint_view`、`sender_private_ratio=0.0`；使用原 seed42 节点划分，训练随机种子为 44。原编码器、NC Trainer 与历史运行文件未修改。", "",
        "所有主结果来自 Best-Validation-Accuracy checkpoint。Best-F1 checkpoint 单独保存和核验，仅作为选择记录；它的指标未混入主结果。Private CE 权重为 0.3，Balance Loss 权重为 0.01，Trainer `aux_weight=1.0`。Test evaluation 关闭，进入 Trainer 前 Test 标签设为 `-100`。", "",
        "训练配置：Physical/Semantic hops=3/1，4 experts、Top-K=2、hidden=256，V7B.1 semantic KNN K=8，AdamW (LR=0.001, weight decay=0.0001)，dropout=0.2/0.15，300 epoch 上限、patience=30。", "",
        "## 九个单元", "",
        "Accuracy 与 Macro-F1 以百分比表示；Gap 定义为 Train − Validation。七条历史记录是复用既有 checkpoint 元数据，不是本次重训。", "",
        "| Dataset | Seed | 来源 | Val Acc | Val Macro-F1 | Val CE | Acc Gap | Macro-F1 Gap | CE Gap | Best Acc/F1 epoch |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for row in sorted((item for item in rows if item["dataset"] == dataset), key=lambda item: int(item["seed"])):
            origin = "新训练" if row["record_origin"] == "new_training" else "复用历史"
            lines.append(
                f"| {dataset} | {row['seed']} | {origin} | {100*float(row['val_acc']):.2f}% | "
                f"{100*float(row['val_macro_f1']):.2f}% | {float(row['val_ce']):.4f} | "
                f"{100*float(row['train_val_acc_gap']):+.2f} pp | "
                f"{100*float(row['train_val_macro_f1_gap']):+.2f} pp | "
                f"{float(row['train_val_ce_gap']):+.4f} | {row['best_acc_epoch']}/{row['best_f1_epoch']} |"
            )
    lines += ["", "## 各数据集 Mean ± SD", "", "SD 使用三个训练种子的样本标准差（ddof=1）。", "",
              "| Dataset | Val Acc | Val Macro-F1 | Val CE | Train Acc | Train Macro-F1 | Train CE | Acc Gap | Macro-F1 Gap | CE Gap |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        s = by_dataset[dataset]
        lines.append("| " + dataset + " | " + " | ".join(
            report_mean_sd(float(s[f"{metric}_mean"]), float(s[f"{metric}_sd"]), metric)
            for metric in METRIC_NAMES
        ) + " |")
    lines += [
        "", "## 来源、检查与偏差", "",
        "- 本次新训练：Grocery/44、ele-fashion/44；各自的配置快照、run metrics、Best-Accuracy 与 Best-F1 权重保存在服务器 `outputs/mrcge_v13_seed44_baseline_completion/`。权重不纳入 Git。",
        "- 复用历史：Movies/Grocery/ele-fashion 的 seed42/43，来源为 `research/mrcge_v13/data/runs.csv`；Movies/seed44 来自 `research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv`。历史 Train 指标从各自 Best-Accuracy checkpoint 的 `epoch_history` 对应 epoch 读取，并与存档 Validation 指标交叉核对。",
        "- seed44 Grocery 与 ele-fashion 的特征、物理边、固定 Split、V7B.1 Semantic Cache 指纹均与 V1.2 repaired 同 seed 参考记录一致；CPU 配置/数据/初始化检查和 GPU 单步前向反向预检通过。",
        ("- GPU 资源守卫按用户本轮明确指示绕过；此项仅为资源准入偏差，不改变训练/评估协议。"
         if any(row.get("gpu_guard_status") == "bypassed_by_user" for row in rows)
         else "- GPU 1 由仓库现有共享资源守卫检查。"),
        "- Test 关闭；Test 标签被掩码；checkpoint 重载后的 Train/Validation 指标与 Trainer 选择元数据一致；Best-Accuracy 与 Best-F1 分别核验；所有报告指标为有限数值。",
        "- 协议偏差：本次两个单元无已知偏差。七个历史单元按其原始研究记录复用；本报告不将它们表述为重新训练。",
        "- 详细来源、文件指纹、checkpoint SHA256 与每个新单元的运行状态见 `data/runs.csv` 和 `data/manifest.json`。", "",
    ]
    (STUDY / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-preflight", action="store_true")
    group.add_argument("--gpu-preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--worker-preflight", action="store_true")
    group.add_argument("--worker-train", action="store_true")
    parser.add_argument("--dataset", choices=("Grocery", "ele-fashion"))
    parser.add_argument("--bypass-gpu-guard", action="store_true",
                        help="explicitly skip the GPU resource guard and record the override")
    args = parser.parse_args()
    if args.bypass_gpu_guard:
        os.environ["MRCGE_SEED44_BASELINE_SKIP_GPU_GUARD"] = "1"
    DATA.mkdir(parents=True, exist_ok=True)
    if args.cpu_preflight:
        result = cpu_preflight()
        print(json.dumps({"status": result["status"], "new_cells": [x["dataset"] for x in result["new_cell_data"]],
                          "historical_verified": result["historical_checkpoint_rows_verified"]}, indent=2))
    elif args.gpu_preflight:
        run_gpu_preflights()
    elif args.run:
        run_formal_cells()
    elif args.finalize:
        finalize()
    elif args.worker_preflight:
        if not args.dataset:
            parser.error("--dataset required for worker")
        worker_preflight(args.dataset)
    elif args.worker_train:
        if not args.dataset:
            parser.error("--dataset required for worker")
        worker_train(args.dataset)


if __name__ == "__main__":
    main()
