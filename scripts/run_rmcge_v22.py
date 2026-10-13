#!/usr/bin/env python3
"""Run the frozen RM-CGE V2.2 validation-only paired study."""

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
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v21 import data_info, sha256_file, tensor_fingerprint, verify_cache_entries  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.inference import infer_all_embeddings  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels, run_nc  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


DATASETS = ("Movies", "Grocery", "ele-fashion")
CONDITIONS = ("M0", "M1", "M2", "M3", "M4")
SEEDS = (42, 43, 44)
PROTOCOL = "unified_full_graph_nc_v1"
STUDY = ROOT / "research/rmcge_v22"
DATA_DIR = STUDY / "data"
OUTPUT = ROOT / "outputs/rmcge_v22"
PREDICTION_CACHE: dict[tuple[str, int], torch.Tensor] = {}

RUN_FIELDS = [
    "dataset", "condition", "seed", "status", "best_accuracy_epoch", "best_f1_epoch",
    "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce",
    "train_acc", "train_macro_f1", "train_ce", "train_ce_eval", "train_val_acc_gap",
    "train_val_macro_f1_gap", "train_val_ce_gap", "model_parameters", "classifier_parameters", "wall_seconds",
    "peak_allocated_bytes", "peak_reserved_bytes", "best_accuracy_checkpoint", "best_f1_checkpoint",
    "best_accuracy_checkpoint_sha256", "best_f1_checkpoint_sha256", "run_metrics_path", "run_dir",
    "initial_model_sha256", "initial_classifier_sha256", "semantic_cache_sha256", "feature_fingerprint",
    "candidate_fingerprint", "v7b_fingerprint", "split_fingerprint", "test_evaluation", "error",
]
HISTORY_FIELDS = [
    "dataset", "condition", "seed", "epoch", "train_loss", "train_objective", "train_ce",
    "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce", "val_acc", "val_macro_f1",
    "epoch_wall_seconds", "learning_rate",
]
CLASS_FIELDS = [
    "dataset", "condition", "seed", "checkpoint", "class_id", "precision", "recall", "f1",
    "support", "predicted_count",
]
CONFUSION_FIELDS = ["dataset", "condition", "seed", "checkpoint", "true_class", "predicted_class", "count"]
DEGREE_FIELDS = [
    "dataset", "condition", "seed", "checkpoint", "degree_group", "node_count", "val_count",
    "accuracy", "macro_f1", "cross_entropy", "content_text_rms", "content_visual_rms",
]
TRANSITION_FIELDS = [
    "dataset", "condition", "seed", "reference_condition", "class_id", "correct_to_wrong",
    "wrong_to_correct", "unchanged_correct", "unchanged_wrong",
]
GATE_FIELDS = [
    "dataset", "condition", "seed", "modality", "mode", "edge_count", "channel_mean", "channel_std",
    "channel_correlation", "abs_ge_0_99_fraction", "receiver_query_roll_gate_rms",
]
MESSAGE_FIELDS = [
    "dataset", "condition", "seed", "modality", "u1_rms", "cross_rms", "cross_hat_rms",
    "knn_rms", "knn_hat_rms", "intra_delta_rms", "content_lowrank_rms", "content_residual_rms",
    "content_normalized_rms", "content_cross_cosine", "eta_weighted_rms_by_expert",
    "physical_profile_rms_by_expert", "combined_profile_rms_by_expert", "expert_output_rms_by_expert",
    "intrinsic_rms", "weighted_graph_correction_rms",
]
ROUTER_FIELDS = [
    "dataset", "condition", "seed", "modality", "router_entropy", "expert_selection_share",
    "expert_coefficients",
]
INTERVENTION_FIELDS = [
    "dataset", "condition", "seed", "intervention", "status", "val_acc", "val_macro_f1", "val_ce",
    "val_acc_delta", "val_macro_f1_delta", "logit_delta_rms", "prediction_flip_fraction",
]
RESOURCE_FIELDS = [
    "dataset", "condition", "seed", "wall_seconds", "epochs_completed", "mean_epoch_wall_seconds",
    "peak_allocated_bytes", "peak_reserved_bytes", "cuda_device_name", "parameter_count",
    "trainable_parameter_count", "content_path_active", "intra_selector_path_active",
    "content_parameter_count", "intra_selector_parameter_count",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _csv_value(value: Any):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    temporary.replace(path)


def append_rows(name: str, rows: list[dict[str, Any]], fields: list[str], key_fields: tuple[str, ...]) -> None:
    path = DATA_DIR / name
    old = read_rows(path)
    present = {tuple(row.get(key, "") for key in key_fields) for row in old}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key not in present:
            old.append(row)
            present.add(key)
    write_rows(path, old, fields)


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_update_rms(initial: dict[str, torch.Tensor], current: dict[str, torch.Tensor], prefix: str) -> float:
    squares, count = 0.0, 0
    for name, value in initial.items():
        if name.startswith(prefix) and name in current:
            diff = value.detach().cpu().float() - current[name].detach().cpu().float()
            squares += float(diff.square().sum())
            count += diff.numel()
    return math.sqrt(squares / max(count, 1))


def build_config(dataset: str, condition: str, seed: int, device: str, cache: dict[str, Any]):
    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create({
        "paths": root_cfg.paths,
        "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
        "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
        "model": OmegaConf.load(ROOT / "configs/model/rmcge_v22.yaml"),
        "logging": root_cfg.logging,
        "seed": int(seed),
        "num_runs": 1,
        "device": str(device),
    })
    cfg.dataset.name = dataset
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
    cfg.task.record_extended_validation = True
    cfg.task.record_train_ce_separately = True
    cfg.task.extended_checkpoint_metadata = {
        "study": "RM-CGE V2.2", "dataset": dataset, "condition": condition, "seed": int(seed),
        "official_train_val_split": True, "test_evaluation": False, "test_label_values_indexed": False,
    }
    cfg.paths.output_root = str(OUTPUT / "hydra")
    OmegaConf.resolve(cfg)
    return cfg


def mask_test_labels(data):
    """Pass a sentinel at Test indices so the trainer cannot read those labels."""
    safe = copy.copy(data)
    safe.y = data.y.clone()
    safe.y[data.test_idx] = -100
    return safe


def dataset_fingerprints(data) -> dict[str, Any]:
    # Labels enter only through the official Train and Validation indices.
    train_val_idx = torch.cat((data.train_idx, data.val_idx)).sort().values
    train_val_labels = data.y[train_val_idx]
    split = {
        "train_idx": tensor_fingerprint(data.train_idx),
        "val_idx": tensor_fingerprint(data.val_idx),
        "test_idx": tensor_fingerprint(data.test_idx),
        "train_val_labels": tensor_fingerprint(train_val_labels),
    }
    return {
        "features": tensor_fingerprint(data.x),
        "physical_edge_index": tensor_fingerprint(data.edge_index),
        "train_val_split": split,
        "split_fingerprint": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
    }


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"rmcge_v22.{path.parent.name}.{path.parent.parent.name}.{path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def environment_record() -> dict[str, Any]:
    import numpy

    gpu_snapshot: dict[str, Any] = {"available": False, "error": None, "devices": []}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_snapshot["available"] = result.returncode == 0
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                columns = [value.strip() for value in line.split(",")]
                if len(columns) == 7:
                    gpu_snapshot["devices"].append(
                        {
                            "physical_index": int(columns[0]),
                            "name": columns[1],
                            "uuid": columns[2],
                            "driver_version": columns[3],
                            "memory_used_mib": int(columns[4]),
                            "memory_total_mib": int(columns[5]),
                            "utilization_percent": int(columns[6]),
                        }
                    )
        else:
            gpu_snapshot["error"] = result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_snapshot["error"] = repr(exc)

    return {
        "timestamp_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "physical_gpu_snapshot": gpu_snapshot,
    }


def prepare_study(*, require_cuda: bool = True) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    caches = verify_cache_entries()
    env = environment_record()
    if require_cuda and not env["cuda_available"]:
        raise RuntimeError("CUDA is unavailable in the authorized yhf_env process")
    dataset_records = {}
    for dataset in DATASETS:
        cfg = build_config(dataset, "M0", 42, "cuda:0" if env["cuda_available"] else "cpu", caches[dataset])
        raw = load_mag_data(cfg, "nc", 42)
        fingerprint = dataset_fingerprints(raw)
        safe = mask_test_labels(raw)
        if not torch.equal(safe.y[raw.train_idx], raw.y[raw.train_idx]) or not torch.equal(safe.y[raw.val_idx], raw.y[raw.val_idx]):
            raise RuntimeError(f"Train/Validation labels changed while masking Test for {dataset}")
        if bool((safe.y[raw.test_idx] != -100).any()):
            raise RuntimeError(f"Test labels were not masked for {dataset}")
        record = {
            **{key: caches[dataset].get(key) for key in ("cache_path_abs", "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint")},
            "cache_sha256": sha256_file(Path(caches[dataset]["cache_path_abs"])),
            "num_nodes": raw.num_nodes,
            "num_edges": raw.num_edges,
            "num_train": int(raw.train_idx.numel()),
            "num_validation": int(raw.val_idx.numel()),
            "num_test_indices_masked": int(raw.test_idx.numel()),
            **fingerprint,
            "seed_split_fingerprints": {"42": fingerprint["split_fingerprint"]},
        }
        dataset_records[dataset] = record
        for other_seed in (43, 44):
            other_cfg = build_config(dataset, "M0", other_seed, "cuda:0" if env["cuda_available"] else "cpu", caches[dataset])
            other_data = load_mag_data(other_cfg, "nc", other_seed)
            other_fingerprint = dataset_fingerprints(other_data)
            if other_fingerprint["features"] != fingerprint["features"] or other_fingerprint["physical_edge_index"] != fingerprint["physical_edge_index"]:
                raise RuntimeError(f"feature/topology fingerprint differs across seeds for {dataset}/42 vs {other_seed}")
            record["seed_split_fingerprints"][str(other_seed)] = other_fingerprint["split_fingerprint"]
            del other_data
        del raw, safe
    manifest_path = DATA_DIR / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {
        **previous,
        "study": "RM-CGE V2.2 Receiver-Conditioned Cross-modal Message Experts",
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "protocol_version": PROTOCOL,
        "datasets": dataset_records,
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "formal_run_count": 45,
        "test_evaluation": False,
        "test_label_values_indexed": False,
        "environment": env,
        "last_preflight_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return caches, manifest


def load_model_checkpoint(cfg, data, cache, checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    set_seed(int(cfg.seed))
    model = build_model(cfg, data_info(data))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=cfg.dataset.name,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    initial_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    head = torch.nn.Linear(model.out_dim, int(data.num_classes))
    model.load_state_dict(payload["model_state"], strict=True)
    head.load_state_dict(payload["head_state"], strict=True)
    model, head = model.to(device), head.to(device)
    model.eval()
    head.eval()
    return payload, model, head, initial_model_state


def metric_bundle(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    num_classes: int,
    eval_labels: list[int],
) -> dict[str, Any]:
    indices = idx.detach().cpu().long()
    y = labels[indices].detach().cpu().long()
    if bool((y < 0).any()):
        raise RuntimeError("masked Test sentinel reached a measured Train/Validation split")
    selected = logits[indices]
    pred = selected.argmax(-1)
    return {
        "acc": float((pred == y).float().mean()),
        "macro_f1": float(f1_score(y.numpy(), pred.numpy(), labels=list(eval_labels), average="macro", zero_division=0)),
        "ce": float(torch.nn.functional.cross_entropy(selected, y)),
        "prediction": pred,
        "target": y,
        "logits": selected,
        "all_logits": logits,
    }


def _class_and_confusion_rows(dataset: str, condition: str, seed: int, checkpoint: str, metrics: dict[str, Any], num_classes: int):
    y, pred = metrics["target"].numpy(), metrics["prediction"].numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(num_classes)), zero_division=0
    )
    predicted = np.bincount(pred, minlength=num_classes)
    class_rows = [
        {"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint,
         "class_id": cls, "precision": float(precision[cls]), "recall": float(recall[cls]),
         "f1": float(f1[cls]), "support": int(support[cls]), "predicted_count": int(predicted[cls])}
        for cls in range(num_classes)
    ]
    matrix = confusion_matrix(y, pred, labels=list(range(num_classes)))
    confusion_rows = [
        {"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint,
         "true_class": i, "predicted_class": j, "count": int(matrix[i, j])}
        for i in range(num_classes) for j in range(num_classes)
    ]
    return class_rows, confusion_rows


def _degree_rows(dataset, condition, seed, checkpoint, degree, val_idx, y, pred, logits, content_nodes, num_classes):
    groups = (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), (">=8", 8, None))
    rows = []
    for name, lower, upper in groups:
        node_mask = degree >= lower if upper is None else (degree >= lower) & (degree <= upper)
        selected_ids = val_idx[node_mask[val_idx]]
        group_y = y[selected_ids].detach().cpu().long()
        if group_y.numel():
            group_logits = logits[selected_ids]
            group_pred = group_logits.argmax(-1)
            acc = float((group_y == group_pred).float().mean())
            f1 = float(f1_score(group_y.numpy(), group_pred.numpy(), labels=list(range(num_classes)), average="macro", zero_division=0))
            ce = float(torch.nn.functional.cross_entropy(group_logits, group_y))
        else:
            acc = f1 = ce = ""
        content_rms = []
        for modality in ("text", "visual"):
            value = content_nodes.get(modality)
            if value is not None and node_mask.any():
                content_rms.append(float(value[node_mask].float().square().mean().sqrt()))
            else:
                content_rms.append(0.0)
        rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint,
            "degree_group": name, "node_count": int(node_mask.sum()), "val_count": int(selected_ids.numel()),
            "accuracy": acc, "macro_f1": f1, "cross_entropy": ce,
            "content_text_rms": content_rms[0], "content_visual_rms": content_rms[1],
        })
    return rows


def _transition_rows(dataset, condition, seed, target, candidate_pred, reference_pred):
    rows = []
    all_classes = sorted(set(target.tolist()))
    labels = [None, *all_classes]
    for cls in labels:
        mask = torch.ones_like(target, dtype=torch.bool) if cls is None else target == cls
        ref_correct = reference_pred[mask] == target[mask]
        new_correct = candidate_pred[mask] == target[mask]
        rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "reference_condition": "M0",
            "class_id": "all" if cls is None else cls,
            "correct_to_wrong": int((ref_correct & ~new_correct).sum()),
            "wrong_to_correct": int((~ref_correct & new_correct).sum()),
            "unchanged_correct": int((ref_correct & new_correct).sum()),
            "unchanged_wrong": int((~ref_correct & ~new_correct).sum()),
        })
    return rows


def analyze_checkpoint(
    dataset: str, condition: str, seed: int, checkpoint_name: str, checkpoint_path: Path,
    cache: dict[str, Any], raw_data, safe_data, cfg, device: torch.device,
):
    payload, model, head, initial_state = load_model_checkpoint(cfg, safe_data, cache, checkpoint_path, device)
    model.eval()
    x_device = safe_data.x.to(device)
    edge_device = safe_data.edge_index.to(device)
    eval_labels = _resolve_nc_eval_labels(safe_data, include_test=False)
    model._capture_content_nodes = checkpoint_name == "best_accuracy"
    model._last_content_by_modality = {}
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    with torch.no_grad():
        z, _, _, _, info = model(x_device, edge_device, return_details=True)
        logits = head(z).detach().cpu()
    cpu_rng_unchanged = torch.equal(cpu_rng_before, torch.random.get_rng_state())
    cuda_rng_unchanged = cuda_rng_before is None or torch.equal(cuda_rng_before, torch.cuda.get_rng_state(device))
    if not cpu_rng_unchanged or not cuda_rng_unchanged:
        raise RuntimeError("eval-mode Train/Validation diagnostics changed CPU/CUDA RNG state")
    val = metric_bundle(logits, safe_data.y, safe_data.val_idx, int(safe_data.num_classes), eval_labels)
    train = metric_bundle(logits, safe_data.y, safe_data.train_idx, int(safe_data.num_classes), eval_labels)
    checkpoint_metrics = payload["metrics"]
    for key, checkpoint_key in (("acc", "val_acc"), ("macro_f1", "val_macro_f1"), ("ce", "val_ce")):
        if abs(val[key] - float(checkpoint_metrics[checkpoint_key])) > 1e-6:
            raise RuntimeError(f"reloaded {checkpoint_name} validation {key} mismatch: {val[key]}")
    if not all(math.isfinite(val[name]) and math.isfinite(train[name]) for name in ("acc", "macro_f1", "ce")):
        raise RuntimeError("checkpoint reload produced non-finite Train/Validation metrics")
    degree_src, degree_dst = model._normalized_operator(edge_device, safe_data.num_nodes, x_device.dtype)[:2]
    degree = torch.bincount(degree_dst, minlength=safe_data.num_nodes).cpu()
    class_rows, confusion_rows = _class_and_confusion_rows(dataset, condition, seed, checkpoint_name, val, int(safe_data.num_classes))
    content_nodes = {key: value.detach().cpu() for key, value in model._last_content_by_modality.items()}
    degree_rows = _degree_rows(
        dataset, condition, seed, checkpoint_name, degree, safe_data.val_idx, safe_data.y,
        val["prediction"], val["all_logits"], content_nodes, int(safe_data.num_classes),
    )
    output = {
        "payload": payload,
        "val": val,
        "train": train,
        "class_rows": class_rows,
        "confusion_rows": confusion_rows,
        "degree_rows": degree_rows,
        "predictions": val["prediction"].clone(),
        "details": info.get("details", {}),
        "rng_unchanged": cpu_rng_unchanged and cuda_rng_unchanged,
        "model": model,
        "head": head,
        "logits": logits,
        "initial_model_state": initial_state,
        "degree": degree,
    }
    return output


def _eval_intervention(model, head, x, edge, labels, val_idx, num_classes, eval_labels, device, intervention, **kwargs):
    model.set_interventions(**kwargs)
    with torch.no_grad():
        z = model(x, edge)[0]
        logits = head(z).detach().cpu()
    metrics = metric_bundle(logits, labels, val_idx, num_classes, eval_labels)
    return {
        "dataset": "", "condition": "", "seed": 0, "intervention": intervention, "status": "ok",
        "val_acc": metrics["acc"], "val_macro_f1": metrics["macro_f1"], "val_ce": metrics["ce"],
        "val_acc_delta": "", "val_macro_f1_delta": "", "logit_delta_rms": "", "prediction_flip_fraction": "",
        "_metrics": metrics,
    }


def frozen_interventions(dataset, condition, seed, result, safe_data, device):
    model, head = result["model"], result["head"]
    x, edge = safe_data.x.to(device), safe_data.edge_index.to(device)
    eval_labels = _resolve_nc_eval_labels(safe_data, include_test=False)
    base = result["val"]
    specifications = [("content_off", {"content_off": True})]
    if condition == "M4":
        specifications.append(("intra_selection_off", {"intra_selection_off": True}))
    specifications.extend((("cross_base_off", {"cross_off": True}), ("knn_off", {"knn_off": True})))
    if condition in {"M3", "M4"}:
        specifications.append(("receiver_query_roll", {"receiver_query_permutation": True}))
    rows = []
    for name, kwargs in specifications:
        item = _eval_intervention(
            model, head, x, edge, safe_data.y, safe_data.val_idx, int(safe_data.num_classes),
            eval_labels, device, name, **kwargs,
        )
        metrics = item.pop("_metrics")
        item.update({
            "dataset": dataset, "condition": condition, "seed": seed,
            "val_acc_delta": metrics["acc"] - base["acc"],
            "val_macro_f1_delta": metrics["macro_f1"] - base["macro_f1"],
            "logit_delta_rms": float((metrics["logits"] - base["logits"]).float().square().mean().sqrt()),
            "prediction_flip_fraction": float((metrics["prediction"] != base["prediction"]).float().mean()),
        })
        rows.append(item)
    model.set_interventions()
    return rows


def analyze_run(dataset, condition, seed, cache, cfg, raw_data, safe_data, paths, device):
    acc = analyze_checkpoint(dataset, condition, seed, "best_accuracy", paths["best_accuracy"], cache, raw_data, safe_data, cfg, device)
    f1 = analyze_checkpoint(dataset, condition, seed, "best_f1", paths["best_f1"], cache, raw_data, safe_data, cfg, device)
    if condition == "M0":
        PREDICTION_CACHE[(dataset, seed)] = acc["predictions"]
    elif (dataset, seed) not in PREDICTION_CACHE:
        # Resume-safe path: regenerate M0 predictions from its frozen checkpoint.
        base_cfg = build_config(dataset, "M0", seed, str(device), cache)
        base_path = OUTPUT / "runs" / dataset / "M0" / f"seed_{seed}" / "best_accuracy.pt"
        if not base_path.is_file():
            raise RuntimeError(f"paired M0 checkpoint missing for {dataset}/seed_{seed}")
        base_data = mask_test_labels(raw_data)
        baseline = analyze_checkpoint(dataset, "M0", seed, "best_accuracy", base_path, cache, raw_data, base_data, base_cfg, device)
        PREDICTION_CACHE[(dataset, seed)] = baseline["predictions"]
    transition_rows = [] if condition == "M0" else _transition_rows(
        dataset, condition, seed, acc["val"]["target"], acc["predictions"], PREDICTION_CACHE[(dataset, seed)]
    )
    intervention_rows = frozen_interventions(dataset, condition, seed, acc, safe_data, device)
    gate_rows = []
    message_rows = []
    router_rows = []
    if condition != "M0":
        gate_details = acc["model"].content_gate_diagnostics(safe_data.x.to(device), safe_data.edge_index.to(device))
        for modality, entry in gate_details.items():
            gate_rows.append({"dataset": dataset, "condition": condition, "seed": seed, "modality": modality, "mode": acc["model"].content_mode, **entry})
    details = acc["details"].get("modalities", {})
    for modality, entry in details.items():
        content = entry.get("content", {})
        message_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "modality": modality,
            "u1_rms": entry.get("u1_rms"), "cross_rms": entry.get("cross_rms"), "cross_hat_rms": entry.get("cross_hat_rms"),
            "knn_rms": entry.get("knn_rms"), "knn_hat_rms": entry.get("knn_hat_rms"),
            "intra_delta_rms": entry.get("intra_delta_rms"), "content_lowrank_rms": content.get("lowrank_rms", 0.0),
            "content_residual_rms": content.get("residual_rms", 0.0), "content_normalized_rms": content.get("normalized_residual_rms", 0.0),
            "content_cross_cosine": content.get("residual_cross_cosine", 0.0),
            "eta_weighted_rms_by_expert": content.get("eta_weighted_rms_by_expert", [0.0] * 4),
            "physical_profile_rms_by_expert": entry.get("physical_profile_rms_by_expert"),
            "combined_profile_rms_by_expert": entry.get("combined_profile_rms_by_expert"),
            "expert_output_rms_by_expert": entry.get("expert_output_rms_by_expert"),
            "intrinsic_rms": entry.get("intrinsic_rms"), "weighted_graph_correction_rms": entry.get("weighted_graph_correction_rms"),
        })
        router_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "modality": modality,
            "router_entropy": entry.get("router_entropy"), "expert_selection_share": entry.get("expert_selection_share"),
            "expert_coefficients": entry.get("expert_coefficients"),
        })
    return {
        "acc": acc, "f1": f1, "transition_rows": transition_rows, "intervention_rows": intervention_rows,
        "gate_rows": gate_rows, "message_rows": message_rows, "router_rows": router_rows,
        "class_rows": acc["class_rows"] + f1["class_rows"],
        "confusion_rows": acc["confusion_rows"] + f1["confusion_rows"],
        "degree_rows": acc["degree_rows"] + f1["degree_rows"],
    }


def _finite_history(history: list[dict[str, Any]]) -> bool:
    for row in history:
        for key, value in row.items():
            if key != "epoch" and isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return False
    return True


def _run_directory(dataset: str, condition: str, seed: int) -> Path:
    base = OUTPUT / "runs" / dataset / condition / f"seed_{seed}"
    if not base.exists():
        return base
    status_path = base / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "complete":
        raise FileExistsError(f"verified run already exists: {base}")
    attempt = OUTPUT / "attempts" / dataset / condition / f"seed_{seed}__{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
    return attempt


def run_one(dataset: str, condition: str, seed: int, device_name: str, cache: dict[str, Any]) -> dict[str, Any]:
    from src.tasks.nc import _module_state_sha256

    raw_data = load_mag_data(build_config(dataset, condition, seed, device_name, cache), "nc", seed)
    safe_data = mask_test_labels(raw_data)
    fingerprints = dataset_fingerprints(raw_data)
    cfg = build_config(dataset, condition, seed, device_name, cache)
    run_dir = _run_directory(dataset, condition, seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    run_name = f"seed_{seed}"
    best_accuracy_path = run_dir / "best_accuracy.pt"
    best_f1_path = run_dir / "best_f1.pt"
    metrics_path = run_dir / "run_metrics.json"
    cfg.task.save_ckpt_path = str(best_accuracy_path)
    cfg.task.best_f1_ckpt_path = str(best_f1_path)
    cfg.task.run_metrics_path = str(metrics_path)
    cfg.task.extended_checkpoint_metadata.update({
        "cache_sha256": sha256_file(Path(cache["cache_path_abs"])),
        "feature_fingerprint": cache["feature_fingerprint"],
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
        "dataset_fingerprints": fingerprints,
        "masked_test_index_count": int(raw_data.test_idx.numel()),
    })
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    atomic_json(run_dir / "status.json", {"status": "running", "dataset": dataset, "condition": condition, "seed": seed, "started_utc": utc_now(), "test_evaluation": False})
    logger = setup_logger(run_dir / "train.log")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        # Pass the masked copy to NC. Test label values are never transferred or indexed.
        run_nc(cfg, safe_data, device, logger)
        if not best_accuracy_path.is_file() or not best_f1_path.is_file() or not metrics_path.is_file():
            raise RuntimeError("training did not produce both selected checkpoints and run metrics")
        result_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if len(result_payload.get("runs", [])) != 1:
            raise RuntimeError("unexpected run_metrics structure")
        run = result_payload["runs"][0]
        metadata = run["metadata"]
        if run["seed"] != seed or result_payload["protocol_version"] != PROTOCOL:
            raise RuntimeError("run_metrics seed/protocol mismatch")
        if any(key.startswith("test_") for key in run.get("metrics", {})) or metadata.get("study_provenance", {}).get("test_evaluation") is not False:
            raise RuntimeError("Test evaluation or Test metric field detected")
        history = metadata.get("epoch_history", [])
        if not history or not _finite_history(history):
            raise RuntimeError("epoch history is empty or contains non-finite numeric values")
        first_gradients = metadata.get("gradient_diagnostics", {})
        if condition != "M0" and not first_gradients.get("content_gate.2.weight", {}).get("gradient_nonzero", False):
            raise RuntimeError("zero-initialized ContentGate final layer did not receive first-step task gradient")
        paths = {"best_accuracy": best_accuracy_path, "best_f1": best_f1_path}
        analyzed = analyze_run(dataset, condition, seed, cache, cfg, raw_data, safe_data, paths, device)
        acc, f1 = analyzed["acc"], analyzed["f1"]
        initial = metadata.get("initial_parameter_audit", {})
        if not initial.get("model_state_sha256") or not initial.get("classifier_state_sha256"):
            raise RuntimeError("missing shared-initialization hashes")
        parameter_total = int(metadata.get("model_parameters", 0)) + int(metadata.get("classifier_parameters", 0))
        run_row = {
            "dataset": dataset, "condition": condition, "seed": seed, "status": "complete",
            "best_accuracy_epoch": int(acc["payload"]["epoch"]), "best_f1_epoch": int(f1["payload"]["epoch"]),
            "val_acc": acc["val"]["acc"], "val_macro_f1": acc["val"]["macro_f1"], "val_ce": acc["val"]["ce"],
            "best_f1_val_acc": f1["val"]["acc"], "best_f1_val_macro_f1": f1["val"]["macro_f1"], "best_f1_val_ce": f1["val"]["ce"],
            "train_acc": acc["train"]["acc"], "train_macro_f1": acc["train"]["macro_f1"],
            "train_ce": acc["payload"]["metrics"]["train_ce"], "train_ce_eval": acc["train"]["ce"],
            "train_val_acc_gap": acc["train"]["acc"] - acc["val"]["acc"],
            "train_val_macro_f1_gap": acc["train"]["macro_f1"] - acc["val"]["macro_f1"],
            "train_val_ce_gap": acc["train"]["ce"] - acc["val"]["ce"],
            "model_parameters": int(metadata["model_parameters"]), "classifier_parameters": int(metadata["classifier_parameters"]),
            "wall_seconds": float(metadata["training_wall_seconds"]),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes", ""),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes", ""),
            "best_accuracy_checkpoint": str(best_accuracy_path), "best_f1_checkpoint": str(best_f1_path),
            "best_accuracy_checkpoint_sha256": sha256_file(best_accuracy_path),
            "best_f1_checkpoint_sha256": sha256_file(best_f1_path), "run_metrics_path": str(metrics_path), "run_dir": str(run_dir),
            "initial_model_sha256": initial["model_state_sha256"], "initial_classifier_sha256": initial["classifier_state_sha256"],
            "semantic_cache_sha256": sha256_file(Path(cache["cache_path_abs"])),
            "feature_fingerprint": cache["feature_fingerprint"], "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"], "split_fingerprint": fingerprints["split_fingerprint"],
            "test_evaluation": False, "error": "",
        }
        append_rows("runs.csv", [run_row], RUN_FIELDS, ("dataset", "condition", "seed"))
        history_rows = [{"dataset": dataset, "condition": condition, "seed": seed, **row} for row in history]
        append_rows("epoch_history.csv", history_rows, HISTORY_FIELDS, ("dataset", "condition", "seed", "epoch"))
        append_rows("classwise_metrics.csv", analyzed["class_rows"], CLASS_FIELDS, ("dataset", "condition", "seed", "checkpoint", "class_id"))
        append_rows("confusion_matrices.csv", analyzed["confusion_rows"], CONFUSION_FIELDS, ("dataset", "condition", "seed", "checkpoint", "true_class", "predicted_class"))
        append_rows("degree_group_metrics.csv", analyzed["degree_rows"], DEGREE_FIELDS, ("dataset", "condition", "seed", "checkpoint", "degree_group"))
        if analyzed["transition_rows"]:
            append_rows("prediction_transitions.csv", analyzed["transition_rows"], TRANSITION_FIELDS, ("dataset", "condition", "seed", "class_id"))
        if analyzed["gate_rows"]:
            append_rows("content_gate_diagnostics.csv", analyzed["gate_rows"], GATE_FIELDS, ("dataset", "condition", "seed", "modality"))
        append_rows("message_scale_diagnostics.csv", analyzed["message_rows"], MESSAGE_FIELDS, ("dataset", "condition", "seed", "modality"))
        append_rows("expert_router_diagnostics.csv", analyzed["router_rows"], ROUTER_FIELDS, ("dataset", "condition", "seed", "modality"))
        append_rows("frozen_interventions.csv", analyzed["intervention_rows"], INTERVENTION_FIELDS, ("dataset", "condition", "seed", "intervention"))
        resources = [{
            "dataset": dataset, "condition": condition, "seed": seed,
            "wall_seconds": metadata["training_wall_seconds"], "epochs_completed": metadata["epochs_completed"],
            "mean_epoch_wall_seconds": metadata["mean_epoch_wall_seconds"],
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "cuda_device_name": metadata.get("cuda_device_name"), "parameter_count": parameter_total,
            "trainable_parameter_count": sum(int(p.numel()) for p in analyzed["acc"]["model"].parameters() if p.requires_grad) + sum(int(p.numel()) for p in analyzed["acc"]["head"].parameters() if p.requires_grad),
            "content_path_active": condition != "M0",
            "intra_selector_path_active": condition == "M4",
            "content_parameter_count": sum(int(p.numel()) for name, p in analyzed["acc"]["model"].named_parameters() if name.startswith("content_")),
            "intra_selector_parameter_count": sum(int(p.numel()) for name, p in analyzed["acc"]["model"].named_parameters() if name.startswith(("query_projection.", "key_projection.", "edge_target_modality_embedding.", "edge_purpose_embedding.", "purpose_edge_scorer."))),
        }]
        append_rows("resource_profile.csv", resources, RESOURCE_FIELDS, ("dataset", "condition", "seed"))
        gradient_path = DATA_DIR / "gradient_audit.json"
        gradient = json.loads(gradient_path.read_text()) if gradient_path.exists() else {"runs": []}
        gkey = (dataset, condition, seed)
        if not any((r["dataset"], r["condition"], r["seed"]) == gkey for r in gradient["runs"]):
            gradient["runs"].append({"dataset": dataset, "condition": condition, "seed": seed, "first_step": first_gradients})
        atomic_json(gradient_path, gradient)
        check_path = DATA_DIR / "implementation_checks.json"
        checks = json.loads(check_path.read_text()) if check_path.exists() else {"formal_runs": []}
        if not any((r["dataset"], r["condition"], r["seed"]) == gkey for r in checks["formal_runs"]):
            checks["formal_runs"].append({
                "dataset": dataset, "condition": condition, "seed": seed,
                "initial_model_sha256": initial["model_state_sha256"],
                "initial_classifier_sha256": initial["classifier_state_sha256"],
                "validation_rng_unchanged": bool(acc["rng_unchanged"] and f1["rng_unchanged"]),
                "checkpoint_reload_verified": True,
                "checkpoint_val_metrics_match": True,
                "content_final_layer_first_step_gradient_nonzero": first_gradients.get("content_gate.2.weight", {}).get("gradient_nonzero", False),
                "test_evaluation": False, "test_label_values_indexed": False,
                "best_checkpoint_content_query_weight_update_rms": tensor_update_rms(acc["initial_model_state"], acc["payload"]["model_state"], "content_query."),
                "best_checkpoint_content_gate_output_weight_update_rms": tensor_update_rms(acc["initial_model_state"], acc["payload"]["model_state"], "content_gate.2.weight"),
            })
        atomic_json(check_path, checks)
        integrity_path = DATA_DIR / "artifact_integrity.json"
        integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {"checkpoints": []}
        for checkpoint_name, checkpoint_path in (("best_accuracy", best_accuracy_path), ("best_f1", best_f1_path)):
            record = {"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint_name, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path), "bytes": checkpoint_path.stat().st_size}
            if not any((r["dataset"], r["condition"], r["seed"], r["checkpoint"]) == (dataset, condition, seed, checkpoint_name) for r in integrity["checkpoints"]):
                integrity["checkpoints"].append(record)
        atomic_json(integrity_path, integrity)
        atomic_json(run_dir / "status.json", {"status": "complete", "dataset": dataset, "condition": condition, "seed": seed, "completed_utc": utc_now(), "test_evaluation": False, "test_label_values_indexed": False, "metrics": {"val_acc": run_row["val_acc"], "val_macro_f1": run_row["val_macro_f1"], "val_ce": run_row["val_ce"]}})
        print(f"COMPLETE {dataset}/{condition}/seed_{seed}: val_acc={run_row['val_acc']:.6f} val_f1={run_row['val_macro_f1']:.6f} epoch={run_row['best_accuracy_epoch']}", flush=True)
        return run_row
    except Exception as exc:
        record = {"dataset": dataset, "condition": condition, "seed": seed, "time_utc": utc_now(), "run_dir": str(run_dir), "error": repr(exc), "traceback": traceback.format_exc(), "test_evaluation": False}
        failures_path = DATA_DIR / "failures.json"
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else {"failures": []}
        failures["failures"].append(record)
        atomic_json(failures_path, failures)
        atomic_json(run_dir / "status.json", {"status": "failed", **record})
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        del safe_data, raw_data


def _assert_physical_gpu_idle(physical_gpu: int, max_unattributed_mib: int = 256) -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    snapshots: dict[int, tuple[str, int]] = {}
    for line in result.stdout.splitlines():
        index, uuid, used = (value.strip() for value in line.split(",", maxsplit=2))
        snapshots[int(index)] = (uuid, int(used))
    if physical_gpu not in snapshots:
        raise RuntimeError(f"physical GPU {physical_gpu} is not present in nvidia-smi output")
    selected_uuid, total_used_mib = snapshots[physical_gpu]
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    own_pid = os.getpid()
    own_used_mib = 0
    foreign_apps = []
    for line in apps.stdout.splitlines():
        uuid, pid_text, used_text = (value.strip() for value in line.split(",", maxsplit=2))
        if uuid != selected_uuid:
            continue
        try:
            pid, used = int(pid_text), int(used_text)
        except ValueError:
            pid, used = int(pid_text), 0
        if pid == own_pid:
            own_used_mib += used
        else:
            foreign_apps.append({"pid": pid, "used_memory_mib": used})
    if foreign_apps:
        raise RuntimeError(
            f"physical GPU {physical_gpu} has other compute processes {foreign_apps}; refusing to share it"
        )
    unattributed_mib = max(total_used_mib - own_used_mib, 0)
    if unattributed_mib > max_unattributed_mib:
        raise RuntimeError(
            f"physical GPU {physical_gpu} has {unattributed_mib} MiB not attributed to this process "
            f"(limit {max_unattributed_mib} MiB); refusing to share it"
        )


def run_batch(seeds: tuple[int, ...], physical_gpu: int) -> None:
    if seeds not in {(42,), (43, 44)}:
        raise ValueError("run seed 42 as its own integrity batch; run seeds 43/44 only after the batch gate")
    gate_path = DATA_DIR / "seed42_gate.json"
    if seeds == (43, 44):
        if not gate_path.is_file() or json.loads(gate_path.read_text()).get("status") != "pass":
            raise RuntimeError("seeds 43/44 are gated on an audited passing seed42_gate.json")
    caches, manifest = prepare_study(require_cuda=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {str(physical_gpu), f"{physical_gpu},"}:
        raise RuntimeError(f"launch must isolate physical GPU {physical_gpu}; CUDA_VISIBLE_DEVICES={visible!r}")
    device = torch.device("cuda:0")
    for seed in seeds:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                completed = read_rows(DATA_DIR / "runs.csv")
                if any(row["dataset"] == dataset and row["condition"] == condition and int(row["seed"]) == seed and row["status"] == "complete" for row in completed):
                    print(f"SKIP verified {dataset}/{condition}/seed_{seed}", flush=True)
                    continue
                try:
                    _assert_physical_gpu_idle(physical_gpu)
                except Exception as exc:
                    failure = {
                        "dataset": dataset, "condition": condition, "seed": seed,
                        "stage": "pre_run_gpu_guard", "training_started": False,
                        "time_utc": utc_now(), "error": repr(exc), "test_evaluation": False,
                    }
                    failure_path = DATA_DIR / "failures.json"
                    failures = json.loads(failure_path.read_text()) if failure_path.exists() else {"failures": []}
                    failures["failures"].append(failure)
                    atomic_json(failure_path, failures)
                    raise
                try:
                    run_one(dataset, condition, seed, str(device), caches[dataset])
                except FileExistsError as exc:
                    raise RuntimeError(str(exc)) from exc
        manifest = json.loads((DATA_DIR / "manifest.json").read_text())
        manifest.setdefault("completed_seed_batches", [])
        if seed not in manifest["completed_seed_batches"]:
            manifest["completed_seed_batches"].append(seed)
        manifest["last_batch_completed_utc"] = utc_now()
        atomic_json(DATA_DIR / "manifest.json", manifest)
    if seeds == (42,):
        completed = read_rows(DATA_DIR / "runs.csv")
        expected = {(dataset, condition, 42) for dataset in DATASETS for condition in CONDITIONS}
        actual = {_key for _key in ((row["dataset"], row["condition"], int(row["seed"])) for row in completed) if _key[2] == 42 and _key[0] in DATASETS and _key[1] in CONDITIONS}
        if actual != expected:
            raise RuntimeError(f"seed 42 gate missing run keys: {sorted(expected - actual)}")
        checks = json.loads((DATA_DIR / "implementation_checks.json").read_text())
        check_map = {(row["dataset"], row["condition"], row["seed"]): row for row in checks.get("formal_runs", [])}
        for dataset in DATASETS:
            arm_rows = [row for row in completed if row["dataset"] == dataset and int(row["seed"]) == 42]
            if len({row["initial_model_sha256"] for row in arm_rows}) != 1 or len({row["initial_classifier_sha256"] for row in arm_rows}) != 1 or len({row["split_fingerprint"] for row in arm_rows}) != 1:
                raise RuntimeError(f"seed 42 Dataset × arm pairing failed for {dataset}")
        for dataset in DATASETS:
            for condition in CONDITIONS:
                check = check_map.get((dataset, condition, 42))
                if check is None or not check.get("checkpoint_reload_verified") or not check.get("validation_rng_unchanged"):
                    raise RuntimeError(f"seed 42 checkpoint/RNG gate failed for {dataset}/{condition}")
                if condition != "M0" and not check.get("content_final_layer_first_step_gradient_nonzero"):
                    raise RuntimeError(f"seed 42 content gradient gate failed for {dataset}/{condition}")
        atomic_json(gate_path, {
            "status": "pass", "seed": 42, "completed_runs": len(expected), "generated_utc": utc_now(),
            "checks": ["finite train/validation histories", "all checkpoints reload and reproduce metrics", "shared initialization and split/cache pairing", "evaluation RNG unchanged", "M1-M4 final Gate layer receives task gradient", "Test evaluation disabled"],
            "performance_threshold_used": False,
        })
        print("SEED42_GATE pass: 15 verified runs; seed43/44 may proceed after report review", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int, default=1)
    args = parser.parse_args()
    run_batch(tuple(args.seeds), args.physical_gpu)


if __name__ == "__main__":
    main()
