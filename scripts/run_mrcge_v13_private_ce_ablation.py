#!/usr/bin/env python3
"""Small fixed-matrix V1.3-A Private Auxiliary CE ablation runner."""
from __future__ import annotations

import argparse
import copy
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

from scripts import run_mrcge_v12_confirmation as v12_utils
from scripts import run_mrcge_v13_screen as v13
from scripts import run_mrcge_v13_two_hop_cross_screen as history_utils
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed

STUDY = ROOT / "research/mrcge_v13_private_ce_ablation"
DATA_DIR = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_private_ce_ablation"
BRANCH = "exp/mrcge_v13_private_ce_ablation"
BASE_SHA = "a3bf807f6bb9c1330a5276acbee70836bb140086"
BASE_BRANCH = "exp/mrcge_v13_intrinsic_prior_skip_screen"
MODEL = "mrcge_v13"
CONDITION = "M_repaired"
VARIANT = "joint_view"
RUN_VARIANT = "no_private_ce"
PHYSICAL_GPU = 1
CELLS = (("Movies", 42), ("Movies", 43), ("Movies", 44), ("Grocery", 42), ("Grocery", 43))
RUN_FIELDS = (
    "dataset", "model", "condition", "variant", "seed", "attempt", "status",
    "source_git_sha", "unimodal_weight", "aux_weight", "balance_weight",
    "best_acc_epoch", "best_f1_epoch", "epochs_completed",
    "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce",
    "train_acc", "train_macro_f1", "train_ce", "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap",
    "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
    "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path", "best_f1_checkpoint_sha256",
    "attempt_dir", "config_sha256", "model_source_sha256", "trainer_source_sha256",
    "initial_model_sha256", "initial_classifier_sha256", "init_matches_history",
    "feature_fingerprint", "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256",
    "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    "physical_gpu", "gpu_uuid", "gpu_name", "gpu_free_before_mib", "gpu_util_before_pct",
    "gpu_temperature_before_c", "baseline_gpu_pids", "test_evaluation", "link_prediction",
    "checkpoint_reload_verified", "error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def source_hashes() -> dict[str, str]:
    paths = {
        "model": ROOT / "src/models/mrcge_v13.py",
        "trainer": ROOT / "src/tasks/nc.py",
        "model_config": ROOT / "configs/model/mrcge_v13.yaml",
        "task_config": ROOT / "configs/task/nc.yaml",
        "v13_config_builder": ROOT / "scripts/run_mrcge_v13_screen.py",
        "history_loader": ROOT / "scripts/run_mrcge_v13_two_hop_cross_screen.py",
        "checkpoint_audit": ROOT / "scripts/run_mrcge_v12_confirmation.py",
        "runner": Path(__file__).resolve(),
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def json_read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def cfg_paths_changed(before: Any, after: Any, prefix: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        result: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                result.append(path)
            else:
                result.extend(cfg_paths_changed(before[key], after[key], path))
        return result
    return [] if before == after else [prefix]


def build_cfg(dataset: str, seed: int, device: str, cache: dict[str, Any], *, zero_private_ce: bool):
    cfg = v13.make_cfg(dataset, VARIANT, seed, device, cache)
    if cfg.model.name != MODEL or cfg.model.condition != CONDITION or cfg.model.v13_variant != VARIANT:
        raise RuntimeError("configuration did not resolve to original V1.3-A Joint View")
    if float(cfg.model.sender_private_ratio) != 0.0:
        raise RuntimeError("sender_private_ratio must remain zero")
    cfg.task.evaluate_test = False
    if zero_private_ce:
        cfg.task.loss.unimodal_weight = 0.0
    return cfg


def assert_single_config_change(dataset: str, seed: int, cache: dict[str, Any], device: str) -> dict[str, Any]:
    historical_cfg = v13.make_cfg(dataset, VARIANT, seed, device, cache)
    candidate_cfg = copy.deepcopy(historical_cfg)
    candidate_cfg.task.loss.unimodal_weight = 0.0
    before = OmegaConf.to_container(historical_cfg, resolve=True)
    after = OmegaConf.to_container(candidate_cfg, resolve=True)
    changed = cfg_paths_changed(before, after)
    if changed != ["task.loss.unimodal_weight"]:
        raise RuntimeError(f"expected only unimodal_weight to change, found {changed}")
    return {"changed_paths": changed, "historical_unimodal_weight": float(historical_cfg.task.loss.unimodal_weight),
            "candidate_unimodal_weight": 0.0}


def load_model(cfg, data, cache, device: torch.device):
    model = build_model(cfg, v13.data_info(data))
    if model.__class__.__module__ != "src.models.mrcge_v13" or model.__class__.__name__ != "Model":
        raise RuntimeError(f"wrong model class loaded: {model.__class__.__module__}.{model.__class__.__name__}")
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    return model.to(device)


def smoke_test() -> dict[str, Any]:
    cache = v13.cache_entries()["Movies"]
    cfg = build_cfg("Movies", 42, "cpu", cache, zero_private_ce=False)
    assert_single_config_change("Movies", 42, cache, "cpu")
    cfg.task.loss.unimodal_weight = 0.0
    cfg, data, fingerprints, alignment = v13.load_safe_data("Movies", VARIANT, 42, cache, "cpu")
    cfg.task.loss.unimodal_weight = 0.0
    device = torch.device("cpu")
    set_seed(42)
    model = load_model(cfg, data, cache, device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    if not hasattr(model, "unimodal_heads") or set(model.unimodal_heads.keys()) != {"text", "visual"}:
        raise RuntimeError("Private classification heads must remain in the model")
    model.train(); classifier.train()
    z, _, _, aux_loss, info = model(data.x, data.edge_index, return_details=False)
    if not isinstance(info.get("unimodal_logits"), dict) or set(info["unimodal_logits"]) != {"text", "visual"}:
        raise RuntimeError("Private logits interface unexpectedly disappeared")
    main_ce = nn.functional.cross_entropy(classifier(z[data.train_idx]), data.y[data.train_idx])
    total_loss, private_ce = _compose_full_graph_loss(
        model=model, info=info, main_ce=main_ce, aux_loss=aux_loss, aux_weight=1.0,
        unimodal_weight=0.0, labels=data.y, train_idx=data.train_idx,
    )
    if private_ce is not None or not torch.isfinite(total_loss):
        raise RuntimeError("zero-weight loss path returned an invalid auxiliary CE result")
    total_loss.backward()
    grad_tensors = [p.grad for p in list(model.parameters()) + list(classifier.parameters()) if p.grad is not None]
    if not grad_tensors or any(not bool(torch.isfinite(g).all()) for g in grad_tensors):
        raise RuntimeError("CPU smoke test produced missing or non-finite main-model gradients")
    if any(p.grad is not None for p in model.unimodal_heads.parameters()):
        raise RuntimeError("unweighted Private classification heads unexpectedly received CE gradients")
    if not any(p.grad is not None and float(p.grad.norm()) > 0 for p in model.projectors.parameters()):
        raise RuntimeError("Private input projectors did not receive main-loss gradients")
    result = {
        "status": "pass", "completed_utc": now(), "device": "cpu", "dataset": "Movies", "seed": 42,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "unimodal_weight": float(cfg.task.loss.unimodal_weight), "private_heads_present": True,
        "private_logits_present": True, "private_heads_have_aux_ce_grad": False,
        "projector_main_loss_gradients_finite_nonzero": True,
        "loss_finite": True, "loss": float(total_loss.detach()),
        "all_existing_gradients_finite": True,
        "split_fingerprint": fingerprints["split_fingerprint"], "data_alignment": alignment,
        "test_labels_masked": bool((data.y[data.test_idx] == -100).all()), "test_evaluation": False,
    }
    del model, classifier, z, info, aux_loss, total_loss
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_write(OUTPUT / "smoke_test.json", result)
    return result


def historical_rows() -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for dataset, seed in CELLS:
        row = history_utils.historical_v13(dataset, seed)
        checkpoint_hashes = history_utils.verify_reference_checkpoints(row, f"V1.3-A {dataset}/seed{seed}")
        result[(dataset, seed)] = {"row": row, "checkpoint_hashes": checkpoint_hashes}
    return result


def protocol_preflight(cache_entries: dict[str, dict[str, Any]], references: dict[tuple[str, int], dict[str, Any]],
                       device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    configs = {f"{dataset}_seed{seed}": assert_single_config_change(dataset, seed, cache_entries[dataset], device)
               for dataset, seed in CELLS}
    data_records = {}
    for dataset in ("Movies", "Grocery"):
        seed = 42
        cache = cache_entries[dataset]
        cfg, data, fingerprints, alignment = v13.load_safe_data(dataset, VARIANT, seed, cache, device)
        cfg.task.loss.unimodal_weight = 0.0
        if cfg.dataset.auto_generate_splits is not False or cfg.task.evaluate_test is not False:
            raise RuntimeError("automatic split generation and Test evaluation must be disabled")
        if bool((data.y[data.test_idx] != -100).any()):
            raise RuntimeError("Test labels are not masked before Trainer entry")
        data_records[dataset] = {
            "num_nodes": int(data.num_nodes), "num_classes": int(data.num_classes),
            "split_seed": 42, "split_file_path": alignment["split_file_path"],
            "feature_fingerprint": alignment["feature_fingerprint"],
            "dataset_fingerprint": alignment["dataset_fingerprint"],
            "physical_edge_fingerprint": alignment["physical_edge_fingerprint"],
            "split_fingerprint": alignment["split_fingerprint"],
            "split_file_sha256": alignment["split_file_sha256"],
            "semantic_cache_path": cache["cache_path_abs"],
            "semantic_cache_sha256": alignment["semantic_cache_sha256"],
            "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"],
            "v7b_k": 8, "test_labels_masked": True,
        }
    for (dataset, seed), reference in references.items():
        h = reference["row"]
        record = data_records[dataset]
        for key, ref_key in (("dataset_fingerprint", "dataset_fingerprint"),
                             ("physical_edge_fingerprint", "physical_edge_fingerprint"),
                             ("split_fingerprint", "split_fingerprint"),
                             ("split_file_sha256", "split_file_sha256"),
                             ("semantic_cache_sha256", "semantic_cache_sha256"),
                             ("candidate_fingerprint", "candidate_fingerprint"),
                             ("v7b_fingerprint", "v7b_fingerprint")):
            old = h.get(ref_key)
            if old and old != record[key]:
                raise RuntimeError(f"historical/data mismatch for {dataset}/{seed}: {ref_key}")
    return configs, data_records


def run_worker(dataset: str, seed: int, attempt_dir_text: str) -> None:
    attempt_dir = Path(attempt_dir_text).resolve()
    try:
        if os.environ.get("MRCGE_PRIVATE_CE_GUARDED") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("worker must be launched under the physical GPU 1 guard")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("worker cannot see exactly the isolated physical GPU 1")
        baseline = {int(x) for x in os.environ.get("MRCGE_PRIVATE_CE_BASELINE_PIDS", "").split(",") if x}
        device = torch.device("cuda:0")
        v13.guard_gpu(PHYSICAL_GPU, baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192)
        if source_sha() != os.environ.get("MRCGE_PRIVATE_CE_SOURCE_SHA"):
            raise RuntimeError("training source SHA changed before worker start")
        cache = v13.cache_entries()[dataset]
        assert_single_config_change(dataset, seed, cache, "cuda:0")
        cfg, data, fingerprints, alignment = v13.load_safe_data(dataset, VARIANT, seed, cache, "cuda:0")
        cfg.task.loss.unimodal_weight = 0.0
        cfg.task.loss.aux_weight = 1.0
        cfg.model.balance_weight = 0.01
        if float(cfg.task.loss.unimodal_weight) != 0.0 or float(cfg.task.loss.aux_weight) != 1.0:
            raise RuntimeError("ablation loss weights did not resolve as registered")
        if cfg.dataset.auto_generate_splits is not False or cfg.task.evaluate_test is not False:
            raise RuntimeError("fixed split/Test-off configuration drifted")
        expected_init = history_utils.historical_v13(dataset, seed)
        cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
        cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
        cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
        model_path = ROOT / "src/models/mrcge_v13.py"
        trainer_path = ROOT / "src/tasks/nc.py"
        config_text = OmegaConf.to_yaml(cfg, resolve=True)
        (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
        config_hash = hashlib.sha256(config_text.encode()).hexdigest()
        cfg.task.extended_checkpoint_metadata = {
            "study": "MR-CGE V1.3-A Private Auxiliary CE ablation", "dataset": dataset,
            "condition": CONDITION, "variant": VARIANT, "run_variant": RUN_VARIANT,
            "seed": seed, "attempt": 1, "source_git_sha": source_sha(), "base_sha": BASE_SHA,
            "model_name": MODEL, "model_source_sha256": sha256_file(model_path),
            "trainer_source_sha256": sha256_file(trainer_path),
            "config_sha256": config_hash,
            "unimodal_weight": 0.0, "aux_weight": 1.0, "balance_weight": 0.01,
            "feature_fingerprint": cache["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "split_file_sha256": alignment["split_file_sha256"],
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"],
            "test_evaluation": False, "test_labels_masked": True, "link_prediction": False,
        }
        json_write(attempt_dir / "attempt_started.json", {
            "dataset": dataset, "seed": seed, "attempt": 1, "source_git_sha": source_sha(),
            "started_utc": now(), "physical_gpu": PHYSICAL_GPU, "test_evaluation": False,
        })
        init_audit = v12_utils.initial_state_audit(cfg, data, cache, device)
        if expected_init.get("initial_model_sha256") and init_audit["initial_model_sha256"] != expected_init["initial_model_sha256"]:
            raise RuntimeError("same-seed model initialization differs from historical V1.3-A")
        if expected_init.get("initial_classifier_sha256") and init_audit["initial_classifier_sha256"] != expected_init["initial_classifier_sha256"]:
            raise RuntimeError("same-seed external classifier initialization differs from historical V1.3-A")
        logger = logging.getLogger(f"private_ce.{dataset}.{seed}")
        logger.handlers.clear(); logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.FileHandler(attempt_dir / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt); logger.addHandler(handler)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        run_nc(cfg, data, device, logger)
        v13.guard_gpu(PHYSICAL_GPU, baseline, {os.getpid()}, phase="during_run")
        packed = json_read(Path(cfg.task.run_metrics_path), {})
        if packed.get("protocol_version") != str(cfg.task.protocol_version) or packed.get("run_seeds") != [seed]:
            raise RuntimeError("trainer protocol or seed provenance mismatch")
        metadata = packed["runs"][0]["metadata"]
        if float(metadata.get("unimodal_auxiliary_weight", math.nan)) != 0.0:
            raise RuntimeError("trainer metadata does not confirm unimodal weight zero")
        if metadata["initial_parameter_audit"]["model_state_sha256"] != init_audit["initial_model_sha256"]:
            raise RuntimeError("trainer model initialization hash differs from the audited initialization")
        if metadata["initial_parameter_audit"]["classifier_state_sha256"] != init_audit["initial_classifier_sha256"]:
            raise RuntimeError("trainer classifier initialization hash differs from the audited initialization")
        selection = metadata["dual_checkpoint_selection"]
        acc_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        if not acc_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Trainer did not write both Validation checkpoints")
        acc = v12_utils.validation_logits_audit(cfg, data, cache, acc_path, device, details=False)
        f1 = v12_utils.validation_logits_audit(cfg, data, cache, f1_path, device, details=False)
        for selected, audited, label in ((selection["best_accuracy"], acc, "Accuracy"),
                                         (selection["best_macro_f1"], f1, "F1")):
            for field, metric in (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce")):
                if abs(float(selected["metrics"][field]) - float(audited["val"][metric])) > 3e-6:
                    raise RuntimeError(f"reloaded Best-{label} checkpoint {field} differs from Trainer metadata")
        if not (acc["rng_unchanged"] and f1["rng_unchanged"]):
            raise RuntimeError("checkpoint reload validation changed RNG state")
        if bool((data.y[data.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
            raise RuntimeError("Test label/evaluation boundary failed")
        history = metadata.get("epoch_history", [])
        if not history or not all(all(k in row and math.isfinite(float(row[k]))
                                      for k in ("train_loss", "val_acc", "val_macro_f1", "val_ce")) for row in history):
            raise RuntimeError("finite epoch history is missing or incomplete")
        json_write(attempt_dir / "epoch_history.json", history)
        select_acc = selection["best_accuracy"]
        select_f1 = selection["best_macro_f1"]
        train_metrics, val_metrics = acc["train"], acc["val"]
        record = {
            "dataset": dataset, "model": MODEL, "condition": CONDITION, "variant": RUN_VARIANT,
            "seed": seed, "attempt": 1, "status": "complete", "source_git_sha": source_sha(),
            "unimodal_weight": 0.0, "aux_weight": 1.0, "balance_weight": 0.01,
            "best_acc_epoch": select_acc["epoch"], "best_f1_epoch": select_f1["epoch"],
            "epochs_completed": len(history), "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"], "val_ce": val_metrics["ce"],
            "best_f1_val_acc": f1["val"]["acc"], "best_f1_val_macro_f1": f1["val"]["macro_f1"],
            "best_f1_val_ce": f1["val"]["ce"], "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"], "train_ce": train_metrics["ce"],
            "train_val_acc_gap": train_metrics["acc"] - val_metrics["acc"],
            "train_val_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"],
            "train_val_ce_gap": train_metrics["ce"] - val_metrics["ce"],
            "wall_seconds": float(metadata.get("training_wall_seconds", time.perf_counter() - started)),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "checkpoint_path": str(acc_path), "checkpoint_sha256": sha256_file(acc_path),
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": sha256_file(f1_path),
            "attempt_dir": str(attempt_dir), "config_sha256": config_hash,
            "model_source_sha256": sha256_file(model_path), "trainer_source_sha256": sha256_file(trainer_path),
            "initial_model_sha256": init_audit["initial_model_sha256"],
            "initial_classifier_sha256": init_audit["initial_classifier_sha256"],
            "init_matches_history": True,
            "feature_fingerprint": cache["feature_fingerprint"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": alignment["split_file_sha256"],
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "physical_gpu": PHYSICAL_GPU, "gpu_uuid": os.environ.get("MRCGE_PRIVATE_CE_GPU_UUID", ""),
            "gpu_name": torch.cuda.get_device_name(device), "test_evaluation": False,
            "link_prediction": False, "checkpoint_reload_verified": True, "error": "",
        }
        json_write(attempt_dir / "worker_result.json", {"status": "complete", "run": record})
        torch.cuda.synchronize(device)
    except Exception as exc:
        json_write(attempt_dir / "worker_result.json", {
            "status": "failed", "dataset": dataset, "seed": seed, "source_git_sha": source_sha(),
            "error": repr(exc), "traceback": traceback.format_exc(), "test_evaluation": False,
        })
        raise


def baseline_gpu() -> tuple[dict[str, Any], set[int]]:
    snapshot = v13.smi_snapshot()
    gpu = next((x for x in snapshot["devices"] if int(x["physical_index"]) == PHYSICAL_GPU), None)
    if gpu is None:
        raise RuntimeError("physical GPU 1 is not visible to nvidia-smi")
    pids = {int(x["pid"]) for x in snapshot["compute_processes"] if x["gpu_uuid"] == gpu["uuid"]}
    return snapshot, pids


def guarded_run(dataset: str, seed: int, frozen_sha: str) -> dict[str, Any]:
    snapshot, baseline = baseline_gpu()
    pre = v13.guard_gpu(PHYSICAL_GPU, baseline, set(), phase="before_launch", required_free_mib=8192)
    gpu = next(x for x in pre["devices"] if int(x["physical_index"]) == PHYSICAL_GPU)
    attempt_dir = OUTPUT / "runs" / dataset / RUN_VARIANT / f"seed_{seed}" / "attempt_01"
    if attempt_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    json_write(attempt_dir / "attempt_started.json", {
        "dataset": dataset, "seed": seed, "attempt": 1, "started_utc": now(),
        "source_git_sha": frozen_sha, "physical_gpu": PHYSICAL_GPU,
        "gpu_uuid": gpu["uuid"], "baseline_gpu_pids": sorted(baseline),
        "required_free_mib": 8192, "guard_status": pre.get("guard", {}).get("status"),
        "test_evaluation": False,
    })
    env = os.environ.copy()
    env.pop("MR_CGE_SHARED_GPU_OVERRIDE", None)
    env["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    env["MRCGE_PRIVATE_CE_GUARDED"] = "1"
    env["MRCGE_PRIVATE_CE_BASELINE_PIDS"] = ",".join(map(str, sorted(baseline)))
    env["MRCGE_PRIVATE_CE_SOURCE_SHA"] = frozen_sha
    env["MRCGE_PRIVATE_CE_GPU_UUID"] = str(gpu["uuid"])
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--dataset", dataset,
           "--seed", str(seed), "--attempt-dir", str(attempt_dir)]
    log_path = attempt_dir / "guarded_worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        guard_error = None
        while proc.poll() is None:
            time.sleep(5)
            try:
                v13.guard_gpu(PHYSICAL_GPU, baseline, {proc.pid}, phase="during_run")
            except Exception as exc:
                guard_error = repr(exc)
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait(timeout=10)
                break
    payload = json_read(attempt_dir / "worker_result.json", {})
    if guard_error:
        raise RuntimeError(f"GPU ownership guard stopped only this task's worker: {guard_error}")
    if proc.returncode != 0 or payload.get("status") != "complete":
        raise RuntimeError(f"worker failed for {dataset}/seed{seed}; see {log_path}: {payload.get('error')}")
    record = payload["run"]
    record.update({
        "gpu_uuid": gpu["uuid"], "gpu_name": gpu["name"],
        "gpu_free_before_mib": gpu["memory_free_mib"],
        "gpu_util_before_pct": gpu["utilization_percent"],
        "gpu_temperature_before_c": gpu["temperature_c"],
        "baseline_gpu_pids": json.dumps(sorted(baseline)),
    })
    return record


def append_record(row: dict[str, Any]) -> None:
    path = DATA_DIR / "runs.csv"
    rows = read_rows(path)
    key = (row["dataset"], str(row["seed"]))
    if any((r.get("dataset"), r.get("seed")) == key for r in rows):
        raise RuntimeError(f"duplicate result row refused: {key}")
    write_rows(path, [*rows, row], RUN_FIELDS)


def summarize(references: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = read_rows(DATA_DIR / "runs.csv")
    paired = []
    grouped: dict[str, list[dict[str, Any]]] = {"Movies": [], "Grocery": []}
    for run in runs:
        dataset, seed = run["dataset"], int(run["seed"])
        ref = references[(dataset, seed)]["row"]
        old = {k: float(ref[k]) for k in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_ce")}
        new = {k: float(run[k]) for k in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_ce")}
        old["train_val_acc_gap"] = old["train_acc"] - old["val_acc"]
        old["train_val_ce_gap"] = old["train_ce"] - old["val_ce"]
        new["train_val_acc_gap"] = float(run["train_val_acc_gap"])
        new["train_val_ce_gap"] = float(run["train_val_ce_gap"])
        row = {"dataset": dataset, "variant": RUN_VARIANT, "seed": seed,
               "historical_source_git_sha": ref.get("source_git_sha", ""),
               "reference_checkpoint_sha256": ref.get("checkpoint_sha256", ""),
               "new_checkpoint_sha256": run["checkpoint_sha256"],
               "test_evaluation": False}
        row.update({"historical_val_acc": old["val_acc"], "new_val_acc": new["val_acc"],
                    "delta_val_acc_pp": 100.0 * (new["val_acc"] - old["val_acc"]),
                    "historical_val_macro_f1": old["val_macro_f1"], "new_val_macro_f1": new["val_macro_f1"],
                    "delta_val_macro_f1_pp": 100.0 * (new["val_macro_f1"] - old["val_macro_f1"]),
                    "historical_val_ce": old["val_ce"], "new_val_ce": new["val_ce"],
                    "delta_val_ce": new["val_ce"] - old["val_ce"]})
        for key in ("train_acc", "train_ce", "train_val_acc_gap", "train_val_ce_gap"):
            row[f"historical_{key}"] = old[key]
            row[f"new_{key}"] = new[key]
            row[f"delta_{key}"] = new[key] - old[key]
        paired.append(row); grouped[dataset].append({"run": run, "paired": row})
    pfields = list(paired[0]) if paired else []
    write_rows(DATA_DIR / "paired_differences.csv", paired, pfields)
    summary = []
    for dataset, entries in grouped.items():
        if not entries:
            continue
        item: dict[str, Any] = {"dataset": dataset, "n": len(entries), "seeds": ";".join(str(e["run"]["seed"]) for e in entries),
                                "variant": RUN_VARIANT, "test_evaluation": False}
        for metric in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_macro_f1", "train_ce",
                       "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap"):
            values = [float(e["run"][metric]) for e in entries]
            item[f"new_{metric}_mean"] = statistics.mean(values)
            item[f"new_{metric}_population_sd"] = statistics.pstdev(values)
        for metric in ("delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce",
                       "delta_train_acc", "delta_train_ce", "delta_train_val_acc_gap", "delta_train_val_ce_gap"):
            values = [float(e["paired"][metric]) for e in entries]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_population_sd"] = statistics.pstdev(values)
        summary.append(item)
    sfields = list(summary[0]) if summary else []
    write_rows(DATA_DIR / "summary.csv", summary, sfields)
    return paired, summary


def report(references: dict[tuple[str, int], dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    runs = read_rows(DATA_DIR / "runs.csv")
    lines = [
        "# Private Auxiliary CE Ablation Results", "",
        f"Branch `{BRANCH}`; base `{BASE_SHA}`; training source SHA `{source_sha()}`.", "",
        "Only `task.loss.unimodal_weight` changed (0.3 to 0.0). The model is the original `mrcge_v13.Model`, M_repaired / Joint View. Results are Validation-only and exploratory.", "",
        "## Per-seed paired Validation metrics", "",
        "| Dataset | Seed | New Acc | New Macro-F1 | New CE | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    paired = read_rows(DATA_DIR / "paired_differences.csv")
    for r in paired:
        lines.append(f"| {r['dataset']} | {r['seed']} | {100*float(r['new_val_acc']):.3f}% | {100*float(r['new_val_macro_f1']):.3f}% | {float(r['new_val_ce']):.5f} | {float(r['delta_val_acc_pp']):+.3f} | {float(r['delta_val_macro_f1_pp']):+.3f} | {float(r['delta_val_ce']):+.5f} |")
    lines += ["", "## Dataset summary", "", "| Dataset | n | New Acc mean ± SD | New F1 mean ± SD | New CE mean ± SD | Δ Acc mean ± SD (pp) | Δ F1 mean ± SD (pp) | Δ CE mean ± SD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['dataset']} | {s['n']} | {100*float(s['new_val_acc_mean']):.3f}% ± {100*float(s['new_val_acc_population_sd']):.3f} | {100*float(s['new_val_macro_f1_mean']):.3f}% ± {100*float(s['new_val_macro_f1_population_sd']):.3f} | {float(s['new_val_ce_mean']):.5f} ± {float(s['new_val_ce_population_sd']):.5f} | {float(s['delta_val_acc_pp_mean']):+.3f} ± {float(s['delta_val_acc_pp_population_sd']):.3f} | {float(s['delta_val_macro_f1_pp_mean']):+.3f} ± {float(s['delta_val_macro_f1_pp_population_sd']):.3f} | {float(s['delta_val_ce_mean']):+.5f} ± {float(s['delta_val_ce_population_sd']):.5f} |")
    lines += ["", "## Generalization", "", "Train−Validation gap is defined as Train minus Validation. Negative CE gap magnitude is the train/validation CE separation.", "", "| Dataset | Historical Train CE | New Train CE | Historical Val CE | New Val CE | Historical Acc gap | New Acc gap | Historical CE gap | New CE gap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        rr = [r for r in paired if r["dataset"] == dataset]
        if not rr: continue
        mean = lambda key: statistics.mean(float(x[key]) for x in rr)
        lines.append(f"| {dataset} | {mean('historical_train_ce'):.5f} | {mean('new_train_ce'):.5f} | {mean('historical_val_ce'):.5f} | {mean('new_val_ce'):.5f} | {100*mean('historical_train_val_acc_gap'):.2f} pp | {100*mean('new_train_val_acc_gap'):.2f} pp | {mean('historical_train_val_ce_gap'):.5f} | {mean('new_train_val_ce_gap'):.5f} |")
    lines += ["", "## Protocol and checkpoint audit", "", "Five cells ran serially with separate local checkpoint directories. Both Best-Accuracy and Best-F1 checkpoints were reloaded and compared to Trainer metadata. Test labels were masked before Trainer entry; `evaluate_test=false` and Link Prediction was not run. Fixed seed-42 splits and the V7B.1 K=8 cache were reused. See `data/manifest.json` for source/config/data hashes and `data/runs.csv` for checkpoint hashes, training times, resource snapshots, and per-run gaps.", "", "## Conclusion", "", "The decision label is assigned after all five cells complete. No additional unimodal weights or other ablations are part of this study.", ""]
    (STUDY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_all() -> None:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"wrong branch {branch}; expected {BRANCH}")
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip():
        raise RuntimeError("working tree must be clean after the training-source freeze commit")
    base = subprocess.run(["git", "merge-base", "HEAD", BASE_SHA], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    if base != BASE_SHA:
        raise RuntimeError(f"branch does not descend from required base {BASE_SHA}")
    frozen_sha = source_sha()
    frozen_hashes = source_hashes()
    for name in ("model", "trainer", "model_config", "task_config", "v13_config_builder", "history_loader", "checkpoint_audit", "runner"):
        if not frozen_hashes.get(name): raise RuntimeError(f"source hash missing: {name}")
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        raise RuntimeError("refusing to overwrite prior study evidence in data/")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    smoke = json_read(OUTPUT / "smoke_test.json", {})
    if smoke.get("status") != "pass" or smoke.get("device") != "cpu":
        raise RuntimeError("a passing CPU smoke record is required before formal training")
    if smoke.get("script_sha256") != source_hashes()["runner"]:
        raise RuntimeError("runner source changed after the CPU smoke test")
    cache = v13.cache_entries()
    refs = historical_rows()
    configs, data_records = protocol_preflight(cache, refs)
    references_manifest = {}
    for (dataset, seed), item in refs.items():
        row = item["row"]
        references_manifest[f"{dataset}_seed{seed}"] = {
            "source_git_sha": row.get("source_git_sha", ""),
            "metrics": {k: float(row[k]) for k in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_ce") if row.get(k) not in (None, "")},
            "initial_model_sha256": row.get("initial_model_sha256", ""),
            "initial_classifier_sha256": row.get("initial_classifier_sha256", ""),
            "checkpoint_hashes_verified": item["checkpoint_hashes"],
        }
    manifest = {
        "study": "MR-CGE V1.3-A Private Auxiliary CE Ablation", "branch": BRANCH,
        "base_sha": BASE_SHA, "base_branch": BASE_BRANCH, "training_source_sha": frozen_sha,
        "training_source_file_sha256s": frozen_hashes,
        "model": MODEL, "condition": CONDITION, "variant": VARIANT, "run_variant": RUN_VARIANT,
        "loss_weights": {"historical_unimodal_weight": 0.3, "new_unimodal_weight": 0.0,
                         "aux_weight": 1.0, "balance_weight": 0.01},
        "cpu_smoke_test": smoke,
        "configuration_diff_audit": configs, "dataset_records": data_records,
        "historical_references": references_manifest,
        "authorized_cells": [{"dataset": d, "seed": s, "variant": RUN_VARIANT} for d, s in CELLS],
        "completed_cells": [], "training_status": "not_started", "test_evaluation": False,
        "link_prediction": False, "physical_gpu": PHYSICAL_GPU, "started_utc": now(),
    }
    json_write(DATA_DIR / "manifest.json", manifest)
    try:
        for index, (dataset, seed) in enumerate(CELLS, start=1):
            if source_sha() != frozen_sha or source_hashes() != frozen_hashes:
                raise RuntimeError("training sources changed after freeze")
            manifest["training_status"] = "running"
            manifest["current_cell"] = {"dataset": dataset, "seed": seed, "ordinal": index, "total": len(CELLS)}
            json_write(DATA_DIR / "manifest.json", manifest)
            row = guarded_run(dataset, seed, frozen_sha)
            append_record(row)
            manifest["completed_cells"].append({"dataset": dataset, "seed": seed, "status": "complete"})
            manifest.pop("current_cell", None)
            json_write(DATA_DIR / "manifest.json", manifest)
            print(f"COMPLETE {dataset}/seed{seed}: val_acc={row['val_acc']:.6f} val_f1={row['val_macro_f1']:.6f} val_ce={row['val_ce']:.6f}", flush=True)
        if len(read_rows(DATA_DIR / "runs.csv")) != len(CELLS):
            raise RuntimeError("not all five formal run records exist")
        paired, summary = summarize(refs)
        manifest["training_status"] = "complete"
        manifest["completed_utc"] = now()
        manifest.pop("current_cell", None)
        json_write(DATA_DIR / "manifest.json", manifest)
        report(refs, summary)
    except Exception as exc:
        manifest["training_status"] = "failed_or_blocked"
        manifest["failure_utc"] = now()
        manifest["failure"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        json_write(DATA_DIR / "manifest.json", manifest)
        fail_path = DATA_DIR / "failures.json"
        failures = json_read(fail_path, {"failures": []})
        failures["failures"].append({"time_utc": now(), "error": repr(exc), "traceback": traceback.format_exc(),
                                      "source_git_sha": frozen_sha, "test_evaluation": False})
        json_write(fail_path, failures)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one CPU full-model forward/backward zero-weight smoke")
    parser.add_argument("--run-all", action="store_true", help="run the five frozen, serial GPU 1 cells")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=("Movies", "Grocery"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--attempt-dir")
    args = parser.parse_args()
    if args.smoke:
        result = smoke_test()
        result["script_sha256"] = source_hashes()["runner"]
        json_write(OUTPUT / "smoke_test.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.run_all:
        run_all()
    elif args.worker:
        if not args.dataset or args.seed is None or not args.attempt_dir:
            parser.error("--worker requires --dataset, --seed, --attempt-dir")
        run_worker(args.dataset, args.seed, args.attempt_dir)
    else:
        parser.error("choose --smoke or --run-all")


if __name__ == "__main__":
    main()
