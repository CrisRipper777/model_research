#!/usr/bin/env python3
"""Guarded per-cell NC5 and full-graph three-fold LP2 runner for MR-CGE V1.3-A."""

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
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import complete_mrcge_v13_stage_ablation_3seed as guard_impl  # noqa: E402
from scripts.mrcge_v13_nc5_lp2_utils import (  # noqa: E402
    BASE_SHA, DATA_DIR, HISTORICAL_NC_DATASETS, LP_DATASETS, NC_DATASETS,
    NEW_NC_DATASETS, OUTPUT, PROTOCOL_LP, SEEDS, build_config, config_snapshot,
    current_commit, csv_read, csv_write, json_write, loaded_lp_split_path, mask_test_labels,
    now_utc, sha256_file,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.inference import infer_all_embeddings  # noqa: E402
from src.tasks.mrcge_v13_full_graph_lp import (  # noqa: E402
    PROTOCOL_VERSION as LP_PROTOCOL_VERSION, run_full_graph_lp,
    validate_lp_split, audit_loaded_message_graph, train_only_physical_graph,
)
from src.utils.seeds import set_seed  # noqa: E402


NC_FIELDS = [
    "dataset", "seed", "status", "attempt", "record_origin", "selection", "best_epoch",
    "val_acc", "val_macro_f1", "val_ce", "train_acc", "train_ce", "checkpoint_path",
    "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "split_path", "split_sha256", "split_fingerprint", "feature_fingerprint",
    "physical_edge_fingerprint", "semantic_cache_path", "semantic_cache_sha256",
    "candidate_fingerprint", "v7b_fingerprint", "test_evaluation", "test_labels_masked",
    "checkpoint_reload_verified", "source_commit", "wall_seconds", "peak_memory_bytes",
    "gpu_name", "attempt_dir", "error",
]
LP_FIELDS = [
    "dataset", "seed", "status", "attempt", "protocol_version", "best_epoch",
    "val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10",
    "checkpoint_path", "checkpoint_sha256", "split_path", "split_sha256",
    "train_graph_fingerprint", "feature_fingerprint", "semantic_cache_path",
    "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    "fold_seed", "fold_count", "test_evaluation", "test_metrics", "checkpoint_reload_verified",
    "source_commit", "wall_seconds", "peak_memory_bytes", "gpu_name", "attempt_dir", "error",
]


def _load_fingerprints() -> dict[str, Any]:
    path = DATA_DIR / "fingerprints.json"
    if not path.is_file():
        raise FileNotFoundError(f"prepared data fingerprints are missing; run prepare script first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_record(prepared: dict[str, Any]) -> dict[str, Any]:
    path = Path(prepared["semantic_cache_path"])
    if not path.is_file() or sha256_file(path) != prepared["semantic_cache_sha256"]:
        raise RuntimeError(f"semantic cache missing or SHA256 changed: {path}")
    return {
        "cache_path_abs": str(path.resolve()),
        "feature_fingerprint": prepared["feature_fingerprint"],
        "candidate_fingerprint": prepared["candidate_fingerprint"],
        "v7b_fingerprint": prepared["v7b_fingerprint"],
    }


def _run_path(phase: str, dataset: str, seed: int, attempt: int) -> Path:
    return OUTPUT / phase / dataset / f"seed_{seed}" / f"attempt_{attempt:02d}"


def _next_attempt(phase: str, dataset: str, seed: int) -> tuple[int, Path]:
    parent = OUTPUT / phase / dataset / f"seed_{seed}"
    attempts = []
    if parent.is_dir():
        for path in parent.glob("attempt_*"):
            try:
                attempts.append(int(path.name.removeprefix("attempt_")))
            except ValueError:
                continue
    attempt = max(attempts, default=0) + 1
    return attempt, _run_path(phase, dataset, seed, attempt)


def _upsert_attempt_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    rows = csv_read(path)
    key = (row.get("dataset"), str(row.get("seed")), str(row.get("attempt")))
    if any((old.get("dataset"), old.get("seed"), old.get("attempt")) == key for old in rows):
        raise RuntimeError(f"immutable run record already exists: {key}")
    csv_write(path, [*rows, row], fields)


def _capture_gpu_baseline(physical_gpu: int) -> tuple[dict[str, Any], set[int]]:
    snapshot = guard_impl.smi_snapshot()
    selected = next((row for row in snapshot["devices"] if int(row["physical_index"]) == physical_gpu), None)
    if selected is None:
        raise RuntimeError(f"physical GPU {physical_gpu} not found")
    pids = {int(row["pid"]) for row in snapshot["compute_processes"] if row["gpu_uuid"] == selected["uuid"]}
    snapshot["authorized_baseline_compute_pids"] = sorted(pids)
    snapshot["guard_bypassed"] = False
    return snapshot, pids


def _guarded_child(
    *, phase: str, dataset: str, seed: int, physical_gpu: int,
    baseline_pids: set[int], attempt: int, attempt_dir: Path,
) -> tuple[int, str | None]:
    unit = f"{phase}/{dataset}/seed{seed}"
    before = guard_impl.guard_gpu(physical_gpu, baseline_pids, set(), phase="before_launch", required_free_mib=8192)
    if attempt_dir.exists():
        raise FileExistsError(f"refusing to reuse existing attempt directory {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    json_write(attempt_dir / "attempt_started.json", {
        "phase": phase, "dataset": dataset, "seed": seed, "attempt": attempt,
        "started_utc": now_utc(), "source_commit": current_commit(),
        "base_sha": BASE_SHA, "test_evaluation": False,
        "gpu_guard_before_launch": before, "gpu_guard_bypassed": False,
    })
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["MRCGE_NC5_LP2_WORKER"] = "1"
    env["MRCGE_BASELINE_GPU_PIDS"] = ",".join(map(str, sorted(baseline_pids)))
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--phase", phase, "--datasets", dataset,
        "--dataset", dataset, "--seed", str(seed), "--attempt", str(attempt),
        "--attempt-dir", str(attempt_dir), "--physical-gpu", str(physical_gpu),
    ]
    log_path = attempt_dir / "launcher.log"
    guard_error = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(3)
            try:
                guard_impl.guard_gpu(physical_gpu, baseline_pids, {process.pid}, phase="during_run")
            except Exception as exc:
                guard_error = repr(exc)
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                break
    rc = process.returncode if process.returncode is not None else -1
    try:
        after = guard_impl.guard_gpu(physical_gpu, baseline_pids, set(), phase="after_run", required_free_mib=8192)
        json_write(attempt_dir / "gpu_resource.json", {"before": before, "after": after, "guard_bypassed": False})
    except Exception as exc:
        guard_error = guard_error or f"after-run guard: {exc!r}"
        json_write(attempt_dir / "gpu_resource.json", {
            "before": before, "after": None, "after_guard_error": repr(exc), "guard_bypassed": False,
        })
    return rc, guard_error


def _verify_prepared_phase(phase: str, datasets: list[str]) -> dict[str, Any]:
    payload = _load_fingerprints()
    prep = payload.get("prepared_phases", {}).get(phase, {})
    if phase == "lp" and prep.get("blockers"):
        audit = DATA_DIR / "lp_split_preflight_audit.json"
        raise RuntimeError(
            f"LP execution is locked by official split conflicts for {prep['blockers']}; "
            f"see {audit}; no split repair or training was attempted"
        )
    if not prep.get("complete") or any(name not in payload.get("datasets", {}) for name in datasets):
        command = f"python scripts/prepare_mrcge_v13_nc5_lp2.py --phase {phase}"
        raise RuntimeError(f"phase {phase} data/cache preparation incomplete; run: {command}")
    return payload


def _completed_cell(path: Path, dataset: str, seed: int) -> dict[str, Any] | None:
    rows = csv_read(path)
    matches = [row for row in rows if row.get("dataset") == dataset and row.get("seed") == str(seed)
               and row.get("status") == "complete"]
    valid = []
    for row in matches:
        ckpt = Path(row.get("checkpoint_path", ""))
        if not ckpt.is_file() or sha256_file(ckpt) != row.get("checkpoint_sha256"):
            continue
        if row.get("checkpoint_reload_verified") != "True" or row.get("test_evaluation") != "False":
            continue
        if path == DATA_DIR / "nc_new_runs.csv":
            if row.get("selection") != "best_validation_accuracy" or row.get("test_labels_masked") != "True":
                continue
            f1_ckpt = Path(row.get("best_f1_checkpoint_path", ""))
            if not f1_ckpt.is_file() or sha256_file(f1_ckpt) != row.get("best_f1_checkpoint_sha256"):
                continue
        elif row.get("protocol_version") != LP_PROTOCOL_VERSION or row.get("test_metrics") != "not_evaluated":
            continue
        valid.append(row)
    if len(valid) > 1:
        raise RuntimeError(f"duplicate completed records exist for {dataset}/seed{seed}")
    return valid[0] if valid else None


def _verify_nc_gate() -> None:
    path = DATA_DIR / "nc_new_runs.csv"
    rows = csv_read(path)
    expected = {(dataset, seed) for dataset in NEW_NC_DATASETS for seed in SEEDS}
    complete = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (row.get("dataset"), int(row.get("seed", -1)))
        if key in expected:
            ckpt = Path(row.get("checkpoint_path", ""))
            if not ckpt.is_file() or sha256_file(ckpt) != row.get("checkpoint_sha256"):
                raise RuntimeError(f"NC completion gate found a missing or changed checkpoint: {key}")
            f1_ckpt = Path(row.get("best_f1_checkpoint_path", ""))
            if not f1_ckpt.is_file() or sha256_file(f1_ckpt) != row.get("best_f1_checkpoint_sha256"):
                raise RuntimeError(f"NC completion gate found a missing or changed Best-F1 checkpoint: {key}")
            if row.get("test_evaluation") != "False" or row.get("test_labels_masked") != "True":
                raise RuntimeError(f"NC completion gate found protocol mismatch: {key}")
            if row.get("selection") != "best_validation_accuracy" or row.get("checkpoint_reload_verified") != "True":
                raise RuntimeError(f"NC completion gate found invalid selection/reload status: {key}")
            if key in complete:
                raise RuntimeError(f"NC completion gate has duplicate successful rows: {key}")
            complete[key] = row
    if set(complete) != expected:
        missing = sorted(expected - set(complete))
        raise RuntimeError(f"LP is locked until all six new NC cells and checkpoint hashes pass; missing {missing}")


def _nc_worker(dataset: str, seed: int, attempt: int, attempt_dir: Path, physical_gpu: int) -> dict[str, Any]:
    if os.environ.get("MRCGE_NC5_LP2_WORKER") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("NC worker must be started by the isolated GPU guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA is unavailable or physical GPU isolation failed")
    baseline = {int(value) for value in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if value}
    guard_impl.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="worker_start", required_free_mib=8192)
    fingerprints = _load_fingerprints()
    prepared = fingerprints["datasets"][dataset]
    cache = _cache_record(prepared)
    cfg = build_config(dataset, "nc", seed)
    split_path = Path(prepared["split"]["path"]).resolve()
    cfg.dataset.nc_split_path = str(split_path)
    cfg.dataset.auto_generate_splits = False
    cfg.model.semantic_cache_path = cache["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    cfg.task.evaluate_test = False
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3-A NC5/LP2 expansion", "dataset": dataset, "seed": seed,
        "selection": "Best-Validation-Accuracy", "fixed_split_seed": 42,
        "fixed_split_path": str(split_path), "split_sha256": prepared["split"]["sha256"],
        "split_fingerprint": prepared["split"]["split_fingerprint"],
        "feature_fingerprint": prepared["feature_fingerprint"],
        "physical_edge_fingerprint": prepared["physical_edge_fingerprint"],
        "semantic_cache_sha256": prepared["semantic_cache_sha256"],
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "training_source_commit": current_commit(), "test_evaluation": False,
        "test_labels_masked": True, "protocol_version": cfg.task.protocol_version,
    }
    (attempt_dir / "config_snapshot.yaml").write_text(config_snapshot(cfg), encoding="utf-8")
    data = load_mag_data(cfg, "nc", seed)
    actual_split = Path(data.info["nc_split_path"]).resolve()
    if actual_split != split_path or sha256_file(actual_split) != prepared["split"]["sha256"]:
        raise RuntimeError("NC loader did not use the preverified fixed seed42 split")
    if data.y is None or data.train_idx is None or data.val_idx is None or data.test_idx is None:
        raise RuntimeError("NC loader omitted labels or a split partition")
    safe = mask_test_labels(data)
    if cfg.task.evaluate_test is not False or bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError("Test labels were not masked before Trainer entry")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    logger = logging.getLogger(f"nc5.{dataset}.{seed}.{attempt}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(attempt_dir / "train.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter); logger.addHandler(handler)
    started = time.perf_counter()
    try:
        from src.tasks.nc import run_nc
        run_nc(cfg, safe, device, logger)
        guard_impl.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="worker_after_train")
        packed = json.loads(Path(cfg.task.run_metrics_path).read_text(encoding="utf-8"))
        if packed.get("protocol_version") != PROTOCOL_NC or packed.get("run_seeds") != [seed]:
            raise RuntimeError("NC trainer returned an unexpected protocol or training seed")
        run_meta = packed["runs"][0]["metadata"]
        selection = run_meta["dual_checkpoint_selection"]["best_accuracy"]
        f1_selection = run_meta["dual_checkpoint_selection"]["best_macro_f1"]
        accuracy_path = Path(cfg.task.save_ckpt_path)
        f1_path = Path(cfg.task.best_f1_ckpt_path)
        if not accuracy_path.is_file() or not f1_path.is_file():
            raise RuntimeError("NC trainer did not save both Best-Accuracy and Best-F1 checkpoints")

        payload = torch.load(accuracy_path, map_location="cpu", weights_only=False)
        if payload.get("selection") != "best_val_accuracy" or int(payload.get("seed", -1)) != seed:
            raise RuntimeError("NC Best-Accuracy checkpoint metadata is invalid")
        model_info = payload["data_info"]
        model = build_model(cfg, model_info)
        model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
        model.load_state_dict(payload["model_state"], strict=True)
        classifier = nn.Linear(model.out_dim, int(data.num_classes))
        classifier.load_state_dict(payload["head_state"], strict=True)
        model = model.to(device).eval(); classifier = classifier.to(device).eval()
        with torch.no_grad():
            z = infer_all_embeddings(model, safe, device, True, int(cfg.task.inference_batch_size), "full")
            val_ids = safe.val_idx.to(device)
            y_val = safe.y[val_ids]
            logits = classifier(z[val_ids])
            measured = {
                "val_acc": float((logits.argmax(dim=1) == y_val).float().mean().item()),
                "val_macro_f1": float(f1_score(y_val.cpu().numpy(), logits.argmax(dim=1).cpu().numpy(), average="macro", zero_division=0)),
                "val_ce": float(nn.functional.cross_entropy(logits, y_val).item()),
            }
        saved = selection["metrics"]
        for key in ("val_acc", "val_macro_f1", "val_ce"):
            if not math.isfinite(measured[key]) or abs(float(saved[key]) - measured[key]) > 3e-6:
                raise RuntimeError(f"strict reload changed NC {key}: {saved.get(key)} vs {measured[key]}")
        if f1_selection.get("rule", "").find("Macro-F1") < 0:
            raise RuntimeError("Best-F1 checkpoint used a different selection rule")
        f1_payload = torch.load(f1_path, map_location="cpu", weights_only=False)
        if not str(f1_payload.get("selection", "")).startswith("best_val_macro_f1"):
            raise RuntimeError("Best-F1 checkpoint selection label is invalid")
        f1_model = build_model(cfg, model_info)
        f1_model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
        f1_model.load_state_dict(f1_payload["model_state"], strict=True)
        f1_head = nn.Linear(f1_model.out_dim, int(data.num_classes))
        f1_head.load_state_dict(f1_payload["head_state"], strict=True)
        f1_model = f1_model.to(device).eval(); f1_head = f1_head.to(device).eval()
        with torch.no_grad():
            f1_z = infer_all_embeddings(f1_model, safe, device, True, int(cfg.task.inference_batch_size), "full")
            f1_ids = safe.val_idx.to(device)
            f1_logits = f1_head(f1_z[f1_ids])
            f1_measured = {
                "val_acc": float((f1_logits.argmax(dim=1) == safe.y[f1_ids]).float().mean().item()),
                "val_macro_f1": float(f1_score(safe.y[f1_ids].cpu().numpy(), f1_logits.argmax(dim=1).cpu().numpy(), average="macro", zero_division=0)),
                "val_ce": float(nn.functional.cross_entropy(f1_logits, safe.y[f1_ids]).item()),
            }
        f1_saved = f1_payload["metrics"]
        for key in ("val_acc", "val_macro_f1", "val_ce"):
            if not math.isfinite(f1_measured[key]) or abs(float(f1_saved[key]) - f1_measured[key]) > 3e-6:
                raise RuntimeError(f"strict reload changed Best-F1 checkpoint {key}: {f1_saved.get(key)} vs {f1_measured[key]}")
        row = {
            "dataset": dataset, "seed": seed, "status": "complete", "attempt": attempt,
            "record_origin": "new_training", "selection": "best_validation_accuracy",
            "best_epoch": int(payload["epoch"]), **measured,
            "train_acc": selection["metrics"].get("train_accuracy_eval", ""),
            "train_ce": selection["metrics"].get("train_ce_eval", ""),
            "checkpoint_path": str(accuracy_path.resolve()), "checkpoint_sha256": sha256_file(accuracy_path),
            "best_f1_checkpoint_path": str(f1_path.resolve()), "best_f1_checkpoint_sha256": sha256_file(f1_path),
            "split_path": str(split_path), "split_sha256": prepared["split"]["sha256"],
            "split_fingerprint": prepared["split"]["split_fingerprint"],
            "feature_fingerprint": prepared["feature_fingerprint"],
            "physical_edge_fingerprint": prepared["physical_edge_fingerprint"],
            "semantic_cache_path": cache["cache_path_abs"], "semantic_cache_sha256": prepared["semantic_cache_sha256"],
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "test_evaluation": False, "test_labels_masked": True, "checkpoint_reload_verified": True,
            "source_commit": current_commit(), "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
            "gpu_name": torch.cuda.get_device_name(device), "attempt_dir": str(attempt_dir.resolve()), "error": "",
        }
        json_write(attempt_dir / "worker_result.json", row)
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now_utc(), "run": row})
        return row
    except Exception as exc:
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now_utc(), "error": repr(exc),
                                                   "traceback": traceback.format_exc(), "test_evaluation": False})
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close(); logger.removeHandler(handler)
        torch.cuda.empty_cache()


def _lp_worker(dataset: str, seed: int, attempt: int, attempt_dir: Path, physical_gpu: int) -> dict[str, Any]:
    if os.environ.get("MRCGE_NC5_LP2_WORKER") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("LP worker must be started by the isolated GPU guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA is unavailable or physical GPU isolation failed")
    baseline = {int(value) for value in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if value}
    guard_impl.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="worker_start", required_free_mib=8192)
    fingerprints = _load_fingerprints()
    prepared = fingerprints["datasets"][dataset]
    cache = _cache_record(prepared)
    cfg = build_config(dataset, "lp", seed)
    cfg.model.semantic_cache_path = cache["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    cfg.dataset.auto_generate_splits = False
    cfg.task.evaluate_test = False
    cfg.task.save_ckpt_path = str(attempt_dir / "best_validation_mrr.pt")
    data = load_mag_data(cfg, "lp", seed)
    if data.x_t is None or data.x_i is None:
        raise RuntimeError("MM-Graph LP loader did not expose Text/Visual feature views")
    from scripts.mrcge_v13_nc5_lp2_utils import assert_clip_feature_order
    assert_clip_feature_order(data.x, data.x_t, data.x_i)
    loaded_split_path = loaded_lp_split_path(data.info)
    if loaded_split_path is None or sha256_file(loaded_split_path) != prepared["split"]["sha256"]:
        raise RuntimeError("LP loader did not use the preverified official lp-edge-split.pt")
    split_audit = validate_lp_split(data.edge_split, data.num_nodes)
    train_graph = train_only_physical_graph(data.edge_split, data.num_nodes)
    graph_audit = audit_loaded_message_graph(data, train_graph, data.num_nodes)
    config_text = config_snapshot(cfg)
    cfg_sha = hashlib.sha256(config_text.encode()).hexdigest()
    model_cfg_sha = sha256_file(ROOT / "configs/model/mrcge_v13.yaml")
    model_source_sha = sha256_file(ROOT / "src/models/mrcge_v13.py")
    fingerprint_record = {
        "model": "mrcge_v13/M_repaired/joint_view/sender_private_ratio=0",
        "model_config_sha256": model_cfg_sha, "model_source_sha256": model_source_sha,
        "resolved_training_config_sha256": cfg_sha,
        "semantic_cache_path": cache["cache_path_abs"],
        "semantic_cache_sha256": prepared["semantic_cache_sha256"],
        "feature_fingerprint": prepared["feature_fingerprint"],
        "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
        "split_path": prepared["split"]["path"], "split_sha256": prepared["split"]["sha256"],
        "split_fingerprint": prepared["split"]["split_fingerprint"],
        "train_graph_fingerprint": prepared["physical_edge_fingerprint"],
        "train_graph_edge_index_sha256": __import__("scripts.mrcge_v13_nc5_lp2_utils", fromlist=["tensor_sha256"]).tensor_sha256(train_graph),
        "split_audit": split_audit, "loaded_train_graph_audit": graph_audit,
        "text_dim": int(data.x_t.size(1)), "visual_dim": int(data.x_i.size(1)),
        "feature_order": "global x[:, :512] Text; x[:, 512:1024] Visual",
        "semantic_candidate_source": "global node attributes only; train-only physical graph passed for provenance/overlap statistics",
    }
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    json_write(attempt_dir / "lp_protocol.json", {
        "protocol_version": LP_PROTOCOL_VERSION, "training_mode": "full graph; three-fold edge holdout",
        "optimizer": "Adam", "lr": 0.001, "weight_decay": 1e-5, "epochs": 150,
        "patience": 10, "early_stop_min_epoch": 20, "early_stop_min_delta": 1e-4,
        "grad_clip": 1.0, "validation_every_epochs": 2, "fold_count": 3, "fold_seed": 42,
        "projection_dim": 128, "decoder": "LinkPredictor MLP", "decoder_hidden_dim": 256,
        "decoder_layers": 3, "decoder_dropout": 0.02, "supervision_edge_batch_size": 2048,
        "loss": "BCEWithLogitsLoss", "balance_aux_weight": 1.0, "train_negative_ratio": 1,
        "model_semantic_knn_k": 8, "experts": 4, "top_k": 2,
        "evaluate_test": False, "test_metrics": "not_evaluated",
        "protocol_difference": "Full-Graph Three-Fold Edge-Holdout; not the sampled unified_sampled_lp_v1 protocol",
    })
    logger = logging.getLogger(f"lp2.{dataset}.{seed}.{attempt}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(attempt_dir / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        set_seed(seed)
        row = run_full_graph_lp(
            cfg, data, cache, device=device, logger=logger,
            checkpoint_path=str(cfg.task.save_ckpt_path), fingerprint_record=fingerprint_record,
            source_commit=current_commit(),
        )
        guard_impl.guard_gpu(physical_gpu, baseline, {os.getpid()}, phase="worker_after_train")
        row.update({
            "attempt": attempt, "split_path": prepared["split"]["path"],
            "split_sha256": prepared["split"]["sha256"],
            "semantic_cache_path": cache["cache_path_abs"],
            "semantic_cache_sha256": prepared["semantic_cache_sha256"],
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
            "gpu_name": torch.cuda.get_device_name(device),
            "attempt_dir": str(attempt_dir.resolve()), "error": "",
        })
        row["checkpoint_sha256"] = sha256_file(row["checkpoint_path"])
        json_write(attempt_dir / "worker_result.json", row)
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now_utc(), "run": row})
        return row
    except Exception as exc:
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now_utc(), "error": repr(exc),
                                                   "traceback": traceback.format_exc(), "test_evaluation": False})
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close(); logger.removeHandler(handler)
        torch.cuda.empty_cache()


def _worker_main(args) -> None:
    attempt_dir = Path(args.attempt_dir).resolve()
    if args.phase == "nc":
        row = _nc_worker(args.dataset, args.seed, args.attempt, attempt_dir, args.physical_gpu)
    else:
        row = _lp_worker(args.dataset, args.seed, args.attempt, attempt_dir, args.physical_gpu)
    print(f"COMPLETE {args.phase}/{args.dataset}/seed{args.seed}: {row.get('val_acc', row.get('val_mrr'))}", flush=True)


def run_phase(phase: str, datasets: list[str], seeds: list[int], physical_gpu: int) -> None:
    allowed_datasets = NEW_NC_DATASETS if phase == "nc" else list(LP_DATASETS)
    if set(datasets) - set(allowed_datasets):
        raise ValueError(f"phase {phase} supports only {allowed_datasets}; got {datasets}")
    if set(seeds) - set(SEEDS):
        raise ValueError(f"seeds must be selected from {SEEDS}")
    if not datasets or not seeds:
        raise ValueError("at least one dataset and seed are required")
    _verify_prepared_phase(phase, datasets)
    if phase == "lp":
        _verify_nc_gate()
    snapshot, baseline_pids = _capture_gpu_baseline(physical_gpu)
    json_write(OUTPUT / "gpu_baselines" / f"{phase}_baseline_{now_utc().replace(':', '').replace('-', '')}.json",
               {"snapshot": snapshot, "baseline_pids": sorted(baseline_pids), "physical_gpu": physical_gpu,
                "bypass": False, "guard_enabled": True})
    results = DATA_DIR / ("nc_new_runs.csv" if phase == "nc" else "lp_runs.csv")
    for dataset in datasets:
        for seed in seeds:
            complete = _completed_cell(results, dataset, seed)
            if complete:
                print(f"SKIP already complete {phase}/{dataset}/seed{seed}; checkpoint SHA256 verified", flush=True)
                continue
            attempt, attempt_dir = _next_attempt(phase, dataset, seed)
            label = f"{phase}/{dataset}/seed{seed}/attempt{attempt:02d}"
            print(f"START {label} gpu={physical_gpu} guard=enabled", flush=True)
            try:
                code, guard_error = _guarded_child(
                    phase=phase, dataset=dataset, seed=seed, physical_gpu=physical_gpu,
                    baseline_pids=baseline_pids, attempt=attempt, attempt_dir=attempt_dir,
                )
            except Exception as exc:
                code, guard_error = 125, repr(exc)
                attempt_dir.mkdir(parents=True, exist_ok=True)
                json_write(attempt_dir / "status.json", {"status": "blocked_resource", "error": guard_error,
                                                           "test_evaluation": False})
            worker_result_path = attempt_dir / "worker_result.json"
            if code == 0 and worker_result_path.is_file():
                row = json.loads(worker_result_path.read_text(encoding="utf-8"))
                if guard_error:
                    # Preserve a fully completed and verified result so a post-run resource
                    # snapshot failure cannot trigger an unnecessary duplicate training run.
                    row["error"] = f"completed; {guard_error}"
                    status_file = attempt_dir / "status.json"
                    status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
                    status["post_run_guard_warning"] = guard_error
                    json_write(status_file, status)
                _upsert_attempt_csv(results, row, NC_FIELDS if phase == "nc" else LP_FIELDS)
                warning = f" (post-run guard warning: {guard_error})" if guard_error else ""
                print(f"DONE {label}{warning}", flush=True)
                continue
            failure_status = "blocked_resource" if guard_error else "failed"
            status_file = attempt_dir / "status.json"
            detail = json.loads(status_file.read_text()) if status_file.is_file() else {}
            row = {
                "dataset": dataset, "seed": seed, "attempt": attempt, "status": failure_status,
                "record_origin": "new_training" if phase == "nc" else "new_training",
                "test_evaluation": False, "test_labels_masked": (phase == "nc"),
                "checkpoint_reload_verified": False, "source_commit": current_commit(),
                "attempt_dir": str(attempt_dir.resolve()),
                "error": guard_error or detail.get("error", f"worker exit code {code}"),
            }
            _upsert_attempt_csv(results, row, NC_FIELDS if phase == "nc" else LP_FIELDS)
            print(f"{failure_status.upper()} {label}: {row['error']}", flush=True)
            if failure_status == "blocked_resource":
                # Wait for the operator to re-run after the owner/resource guard is satisfied.
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("nc", "lp"), required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=(*NEW_NC_DATASETS, *LP_DATASETS), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.dataset or args.seed is None or args.attempt is None or not args.attempt_dir:
            parser.error("worker mode requires --dataset --seed --attempt --attempt-dir")
        _worker_main(args)
    else:
        datasets = args.datasets
        default = NEW_NC_DATASETS if args.phase == "nc" else LP_DATASETS
        if datasets == ["all"]:
            datasets = list(default)
        run_phase(args.phase, datasets, list(args.seeds), args.physical_gpu)


if __name__ == "__main__":
    main()
