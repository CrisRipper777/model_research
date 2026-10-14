#!/usr/bin/env python3
"""Run the frozen, validation-only MAG View-CGE V1 experiment matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
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

from scripts import run_rmcge_v22 as legacy  # noqa: E402
from scripts.preflight_mag_view_cge_v1 import config as build_config  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels, run_nc  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


DATASETS = legacy.DATASETS
CONDITIONS = ("A0_legacy", "A1_separate_intra", "A2_cross_source")
SEEDS = (42, 43, 44)
STUDY = ROOT / "research/mag_view_cge_v1"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mag_view_cge_v1"
PROTOCOL = legacy.PROTOCOL
IMPLEMENTATION_FREEZE_COMMIT = "f981a5f0164b1ed73c379724139e75c84abb0c98"
PREDICTIONS: dict[tuple[str, int], torch.Tensor] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default), encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    tmp.replace(path)


def append_rows(name: str, rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> None:
    path = DATA / name
    old = read_csv(path)
    present = {tuple(row.get(key, "") for key in key_fields) for row in old}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key not in present:
            old.append(row)
            present.add(key)
    write_csv(path, old)


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mask_test_labels(data):
    safe = copy.copy(data)
    safe.y = data.y.clone()
    safe.y[data.test_idx] = -100
    return safe


def prepare_study(physical_gpu: int) -> dict[str, dict[str, Any]]:
    DATA.mkdir(parents=True, exist_ok=True)
    if not (STUDY / "data/preflight.json").is_file():
        raise RuntimeError("real-data GPU preflight is required before formal runs")
    preflight = json.loads((STUDY / "data/preflight.json").read_text())
    if preflight.get("status") != "pass" or int(preflight.get("physical_gpu", -1)) != physical_gpu:
        raise RuntimeError("GPU preflight did not pass on the selected physical device")
    caches = legacy.verify_cache_entries()
    env = legacy.environment_record()
    if not env["cuda_available"] or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA unavailable or visible-device isolation failed")
    records: dict[str, Any] = {}
    for dataset in DATASETS:
        ref = None
        splits = {}
        for seed in SEEDS:
            cfg = build_config(dataset, "A0_legacy", seed, "mag_view_cge_v1", caches[dataset], "cuda:0")
            data = load_mag_data(cfg, "nc", seed)
            fp = legacy.dataset_fingerprints(data)
            safe = mask_test_labels(data)
            if bool((safe.y[data.test_idx] != -100).any()):
                raise RuntimeError(f"Test labels were not masked: {dataset}/seed{seed}")
            if not torch.equal(safe.y[data.train_idx], data.y[data.train_idx]) or not torch.equal(safe.y[data.val_idx], data.y[data.val_idx]):
                raise RuntimeError(f"masking changed Train/Validation labels: {dataset}/seed{seed}")
            if ref is not None and (fp["features"] != ref["features"] or fp["physical_edge_index"] != ref["physical_edge_index"]):
                raise RuntimeError(f"features/physical graph changed across seeds: {dataset}")
            ref = fp
            splits[str(seed)] = fp["split_fingerprint"]
            del data, safe
        records[dataset] = {
            "cache_path": caches[dataset]["cache_path_abs"],
            "cache_sha256": legacy.sha256_file(Path(caches[dataset]["cache_path_abs"])),
            "feature_fingerprint": caches[dataset]["feature_fingerprint"],
            "candidate_fingerprint": caches[dataset]["candidate_fingerprint"],
            "v7b_fingerprint": caches[dataset]["v7b_fingerprint"],
            "seed_split_fingerprints": splits,
            "features_fingerprint": ref["features"],
            "physical_edge_fingerprint": ref["physical_edge_index"],
        }
    previous = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
    manifest = {
        **previous,
        "study": "MAG View-Collaborative Experts V1",
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "implementation_freeze_commit": IMPLEMENTATION_FREEZE_COMMIT,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "protocol_version": PROTOCOL,
        "datasets": records,
        "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "formal_run_count": 27, "test_evaluation": False,
        "environment": env, "physical_gpu": physical_gpu, "preflight": preflight,
        "last_preflight_utc": utc_now(),
    }
    atomic_json(DATA / "manifest.json", manifest)
    return caches


def metric_bundle(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_classes: int, eval_labels: list[int]) -> dict[str, Any]:
    indices = idx.detach().cpu().long()
    y = labels[indices].detach().cpu().long()
    if bool((y < 0).any()):
        raise RuntimeError("masked Test sentinel reached Train/Validation metric calculation")
    selected = logits[indices]
    pred = selected.argmax(-1)
    return {
        "acc": float((pred == y).float().mean()),
        "macro_f1": float(f1_score(y.numpy(), pred.numpy(), labels=list(eval_labels), average="macro", zero_division=0)),
        "ce": float(torch.nn.functional.cross_entropy(selected, y)),
        "prediction": pred, "target": y, "logits": selected, "all_logits": logits,
    }


def load_checkpoint(dataset: str, condition: str, seed: int, checkpoint_path: Path, safe_data, cache, device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = build_config(dataset, condition, seed, "mag_view_cge_v1", cache, str(device))
    set_seed(seed)
    model = build_model(cfg, legacy.data_info(safe_data))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    head = torch.nn.Linear(model.out_dim, int(safe_data.num_classes))
    model.load_state_dict(payload["model_state"], strict=True)
    head.load_state_dict(payload["head_state"], strict=True)
    model, head = model.to(device).eval(), head.to(device).eval()
    return cfg, payload, model, head


def analyze_checkpoint(dataset: str, condition: str, seed: int, name: str, path: Path, safe_data, cache, device):
    cfg, payload, model, head = load_checkpoint(dataset, condition, seed, path, safe_data, cache, device)
    x, edge = safe_data.x.to(device), safe_data.edge_index.to(device)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    with torch.no_grad():
        z, _, _, _, info = model(x, edge, return_details=True)
        logits = head(z).detach().cpu()
    rng_unchanged = torch.equal(cpu_before, torch.random.get_rng_state())
    if cuda_before is not None:
        rng_unchanged = rng_unchanged and torch.equal(cuda_before, torch.cuda.get_rng_state(device))
    if not rng_unchanged:
        raise RuntimeError("checkpoint Train/Validation diagnostics changed CPU/CUDA RNG")
    labels = safe_data.y
    eval_labels = _resolve_nc_eval_labels(safe_data, include_test=False)
    val = metric_bundle(logits, labels, safe_data.val_idx, int(safe_data.num_classes), eval_labels)
    train = metric_bundle(logits, labels, safe_data.train_idx, int(safe_data.num_classes), eval_labels)
    for key, payload_key in (("acc", "val_acc"), ("macro_f1", "val_macro_f1"), ("ce", "val_ce")):
        if abs(val[key] - float(payload["metrics"][payload_key])) > 1e-6:
            raise RuntimeError(f"checkpoint {name} {key} did not reproduce stored validation metric")
    for subset in (val, train):
        if not all(math.isfinite(subset[key]) for key in ("acc", "macro_f1", "ce")):
            raise RuntimeError(f"non-finite reloaded Train/Validation metric: {dataset}/{condition}/{seed}")
    precision, recall, class_f1, support = precision_recall_fscore_support(
        val["target"].numpy(), val["prediction"].numpy(), labels=list(range(int(safe_data.num_classes))), zero_division=0
    )
    predicted = np.bincount(val["prediction"].numpy(), minlength=int(safe_data.num_classes))
    class_rows = [{
        "dataset": dataset, "condition": condition, "seed": seed, "checkpoint": name,
        "class_id": cls, "precision": float(precision[cls]), "recall": float(recall[cls]),
        "f1": float(class_f1[cls]), "support": int(support[cls]), "predicted_count": int(predicted[cls]),
    } for cls in range(int(safe_data.num_classes))]
    matrix = confusion_matrix(val["target"].numpy(), val["prediction"].numpy(), labels=list(range(int(safe_data.num_classes))))
    confusion_rows = [{
        "dataset": dataset, "condition": condition, "seed": seed, "checkpoint": name,
        "true_class": i, "predicted_class": j, "count": int(matrix[i, j]),
    } for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    if condition == "A0_legacy":
        src, dst, _, _, _ = model._normalized_operator(edge, safe_data.num_nodes, x.dtype)
        degree = torch.bincount(dst, minlength=safe_data.num_nodes).cpu()
    else:
        src, dst, _, _, _ = model._normalized_operator(edge, safe_data.num_nodes, x.dtype)
        degree = torch.bincount(dst, minlength=safe_data.num_nodes).cpu()
    degree_rows = []
    degree_groups = (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), (">=8", 8, None))
    for group, low, high in degree_groups:
        node_mask = degree == 0 if high == 0 else (degree >= low if high is None else (degree >= low) & (degree <= high))
        ids = safe_data.val_idx[node_mask[safe_data.val_idx]]
        y = safe_data.y[ids].long()
        if y.numel():
            pred = logits[ids].argmax(-1)
            values = {
                "accuracy": float((pred == y).float().mean()),
                "macro_f1": float(f1_score(y.numpy(), pred.numpy(), labels=list(range(int(safe_data.num_classes))), average="macro", zero_division=0)),
                "cross_entropy": float(torch.nn.functional.cross_entropy(logits[ids], y)),
            }
        else:
            values = {"accuracy": "", "macro_f1": "", "cross_entropy": ""}
        degree_rows.append({"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": name, "degree_group": group, "node_count": int(node_mask.sum()), "val_count": int(ids.numel()), **values})
    return {
        "payload": payload, "model": model, "head": head, "val": val, "train": train,
        "logits": logits, "predictions": val["prediction"], "details": info.get("details", {}),
        "rng_unchanged": rng_unchanged, "class_rows": class_rows,
        "confusion_rows": confusion_rows, "degree_rows": degree_rows,
    }


def _view_rows(dataset: str, condition: str, seed: int, checkpoint: str, details: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    views, routers = [], []
    modalities = details.get("modalities", {})
    for modality, entry in modalities.items():
        if condition == "A0_legacy":
            views.append({"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint, "modality": modality, "view": "physical", "input_rms": entry.get("u1_rms", ""), "calibrated_rms": entry.get("cross_hat_rms", ""), "expert_correction_rms": entry.get("weighted_graph_correction_rms", ""), "response_rms": "", "gamma": "", "weighted_contribution_rms": entry.get("weighted_graph_correction_rms", ""), "mask_nodes": "", "source_modality": "physical", "semantic_to_physical_ratio": ""})
            for view in ("same", "cross"):
                views.append({"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint, "modality": modality, "view": view, "input_rms": 0.0, "calibrated_rms": 0.0, "expert_correction_rms": 0.0, "response_rms": 0.0, "gamma": 0.0, "weighted_contribution_rms": 0.0, "mask_nodes": 0, "source_modality": "", "semantic_to_physical_ratio": 0.0})
            continue
        phys = {"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint, "modality": modality, "view": "physical", "input_rms": entry.get("physical_first_hop_rms"), "calibrated_rms": entry.get("physical_cross_calibrated_rms"), "expert_correction_rms": entry.get("physical_expert_correction_rms"), "response_rms": entry.get("physical_profile_rms"), "gamma": entry.get("physical_strength"), "weighted_contribution_rms": entry.get("physical_contribution_rms"), "mask_nodes": entry.get("physical_active_nodes"), "source_modality": "physical", "semantic_to_physical_ratio": entry.get("semantic_to_physical_ratio"), "expert_input_rms_by_expert": entry.get("physical_expert_input_rms_by_expert", []), "expert_output_rms_by_expert": entry.get("physical_expert_output_rms_by_expert", []), "expert_nonlinear_correction_rms_by_expert": entry.get("physical_expert_nonlinear_correction_rms_by_expert", [])}
        views.append(phys)
        for key, view in (("same_view", "same"), ("cross_view", "cross")):
            item = entry.get(key, {})
            views.append({"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint, "modality": modality, "view": view, "input_rms": item.get("input_rms", 0.0), "calibrated_rms": item.get("calibrated_rms", 0.0), "expert_correction_rms": item.get("expert_rms", 0.0), "response_rms": item.get("response_rms", 0.0), "gamma": item.get("gamma", 0.0), "weighted_contribution_rms": item.get("weighted_rms", 0.0), "mask_nodes": item.get("active_nodes", 0), "source_modality": item.get("source_modality", modality), "semantic_to_physical_ratio": entry.get("semantic_to_physical_ratio", 0.0), "expert_input_rms_by_expert": item.get("expert_input_rms_by_expert", []), "expert_output_rms_by_expert": item.get("expert_output_rms_by_expert", []), "expert_nonlinear_correction_rms_by_expert": item.get("expert_nonlinear_correction_rms_by_expert", [])})
            routers.append({"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint, "modality": modality, "view": view, "entropy": item.get("entropy", 0.0), "top2_selection_share": item.get("selection_share", [0, 0, 0, 0]), "exact_same_cross_top2_match": entry.get("same_cross_top2_exact_match_fraction", "") if view in ("same", "cross") else "", "mask_nodes": item.get("active_nodes", 0)})
        routers.append({"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": checkpoint, "modality": modality, "view": "physical", "entropy": entry.get("physical_route_entropy", 0.0), "top2_selection_share": entry.get("physical_selection_share", []), "exact_same_cross_top2_match": "", "mask_nodes": entry.get("physical_active_nodes", 0)})
    return views, routers


def _view_interventions(result, safe_data, dataset, condition, seed, device):
    if condition == "A0_legacy":
        return []
    model, head = result["model"], result["head"]
    baseline = result["val"]
    x, edge = safe_data.x.to(device), safe_data.edge_index.to(device)
    eval_labels = _resolve_nc_eval_labels(safe_data, include_test=False)
    specifications = [("same_view_off", {"same_off": True, "cross_source_off": False})]
    if condition == "A2_cross_source":
        specifications.append(("cross_source_view_off", {"same_off": False, "cross_source_off": True}))
        specifications.append(("all_semantic_views_off", {"same_off": True, "cross_source_off": True}))
    rows = []
    for name, kwargs in specifications:
        model.set_view_interventions(**kwargs)
        with torch.no_grad():
            logits = head(model(x, edge)[0]).detach().cpu()
        metrics = metric_bundle(logits, safe_data.y, safe_data.val_idx, int(safe_data.num_classes), eval_labels)
        rows.append({"dataset": dataset, "condition": condition, "seed": seed, "intervention": name, "status": "checkpoint_sensitivity_only", "val_acc": metrics["acc"], "val_macro_f1": metrics["macro_f1"], "val_ce": metrics["ce"], "val_acc_delta": metrics["acc"] - baseline["acc"], "val_macro_f1_delta": metrics["macro_f1"] - baseline["macro_f1"], "logit_delta_rms": float((logits[safe_data.val_idx] - result["logits"][safe_data.val_idx]).square().mean().sqrt()), "prediction_flip_fraction": float((metrics["prediction"] != baseline["prediction"]).float().mean())})
    model.set_view_interventions()
    return rows


def _baseline_accuracy_checkpoint(dataset: str, seed: int) -> Path:
    for row in read_csv(DATA / "runs.csv"):
        if row.get("dataset") == dataset and row.get("condition") == "A0_legacy" and int(row.get("seed", -1)) == seed and row.get("status") == "complete":
            return Path(row["best_accuracy_checkpoint"])
    raise RuntimeError(f"paired internal A0 run record missing for {dataset}/seed{seed}")


def _next_run_dir(dataset: str, condition: str, seed: int) -> Path:
    base = OUTPUT / "runs" / dataset / condition / f"seed_{seed}"
    if not base.exists():
        return base
    attempt = 2
    while base.with_name(f"seed_{seed}_attempt_{attempt}").exists():
        attempt += 1
    return base.with_name(f"seed_{seed}_attempt_{attempt}")


def run_one(dataset: str, condition: str, seed: int, cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = build_config(dataset, condition, seed, "mag_view_cge_v1", cache, str(device))
    raw = load_mag_data(cfg, "nc", seed)
    safe = mask_test_labels(raw)
    fingerprints = legacy.dataset_fingerprints(raw)
    run_dir = _next_run_dir(dataset, condition, seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    acc_path, f1_path, metrics_path = run_dir / "best_accuracy.pt", run_dir / "best_f1.pt", run_dir / "run_metrics.json"
    cfg.task.save_ckpt_path, cfg.task.best_f1_ckpt_path, cfg.task.run_metrics_path = str(acc_path), str(f1_path), str(metrics_path)
    cfg.task.extended_checkpoint_metadata.update({
        "study": "MAG View-Collaborative Experts V1", "dataset": dataset, "condition": condition,
        "seed": seed, "test_evaluation": False, "test_label_values_indexed": False,
        "masked_test_index_count": int(raw.test_idx.numel()), "dataset_fingerprints": fingerprints,
        "cache_sha256": legacy.sha256_file(Path(cache["cache_path_abs"])),
    })
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    atomic_json(run_dir / "status.json", {"status": "running", "dataset": dataset, "condition": condition, "seed": seed, "started_utc": utc_now(), "test_evaluation": False})
    logger = legacy.setup_logger(run_dir / "train.log")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        run_nc(cfg, safe, device, logger)
        if not acc_path.is_file() or not f1_path.is_file() or not metrics_path.is_file():
            raise RuntimeError("NC trainer did not write both checkpoints and run metrics")
        metrics_payload = json.loads(metrics_path.read_text())
        if len(metrics_payload.get("runs", [])) != 1 or metrics_payload.get("protocol_version") != PROTOCOL:
            raise RuntimeError("unexpected run_metrics structure or protocol")
        run = metrics_payload["runs"][0]
        meta = run["metadata"]
        if run["seed"] != seed or meta.get("study_provenance", {}).get("test_evaluation") is not False:
            raise RuntimeError("seed mismatch or Test evaluation enabled")
        if any(key.startswith("test_") for key in run.get("metrics", {})):
            raise RuntimeError("Test metrics found in trainer output")
        history = meta.get("epoch_history", [])
        if not history or not legacy._finite_history(history):
            raise RuntimeError("empty/non-finite train/validation history")
        grads = meta.get("gradient_diagnostics", {})
        if condition != "A0_legacy":
            for required in ("gamma_s_raw", "view_router.2.weight"):
                if not grads.get(required, {}).get("gradient_nonzero", False):
                    raise RuntimeError(f"first-step gradient missing/nonzero check failed: {required}")
            if condition == "A2_cross_source" and not grads.get("gamma_c_raw", {}).get("gradient_nonzero", False):
                raise RuntimeError("A2 gamma_C first-step task gradient is zero")

        acc = analyze_checkpoint(dataset, condition, seed, "best_accuracy", acc_path, safe, cache, device)
        f1 = analyze_checkpoint(dataset, condition, seed, "best_f1", f1_path, safe, cache, device)
        if condition == "A0_legacy":
            PREDICTIONS[(dataset, seed)] = acc["predictions"].clone()
        elif (dataset, seed) not in PREDICTIONS:
            base_path = _baseline_accuracy_checkpoint(dataset, seed)
            if not base_path.is_file():
                raise RuntimeError(f"paired internal A0 checkpoint missing for {dataset}/seed{seed}")
            baseline = analyze_checkpoint(dataset, "A0_legacy", seed, "best_accuracy", base_path, safe, cache, device)
            PREDICTIONS[(dataset, seed)] = baseline["predictions"].clone()

        initial = meta.get("initial_parameter_audit", {})
        if not initial.get("model_state_sha256") or not initial.get("classifier_state_sha256"):
            raise RuntimeError("missing initialized model/head hashes")
        params = int(meta.get("model_parameters", 0)) + int(meta.get("classifier_parameters", 0))
        peak_alloc = int(meta.get("cuda_peak_allocated_bytes", 0) or (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0))
        peak_reserved = int(meta.get("cuda_peak_reserved_bytes", 0) or (torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0))
        row = {
            "dataset": dataset, "condition": condition, "seed": seed, "status": "complete",
            "best_accuracy_epoch": int(acc["payload"]["epoch"]), "best_f1_epoch": int(f1["payload"]["epoch"]),
            "epochs_completed": int(meta["epochs_completed"]),
            "val_acc": acc["val"]["acc"], "val_macro_f1": acc["val"]["macro_f1"], "val_ce": acc["val"]["ce"],
            "best_f1_val_acc": f1["val"]["acc"], "best_f1_val_macro_f1": f1["val"]["macro_f1"], "best_f1_val_ce": f1["val"]["ce"],
            "best_f1_train_acc": f1["train"]["acc"], "best_f1_train_macro_f1": f1["train"]["macro_f1"], "best_f1_train_ce": f1["train"]["ce"],
            "best_f1_train_val_acc_gap": f1["train"]["acc"] - f1["val"]["acc"],
            "best_f1_train_val_macro_f1_gap": f1["train"]["macro_f1"] - f1["val"]["macro_f1"],
            "best_f1_train_val_ce_gap": f1["train"]["ce"] - f1["val"]["ce"],
            "train_acc": acc["train"]["acc"], "train_macro_f1": acc["train"]["macro_f1"],
            "train_ce": float(acc["payload"]["metrics"]["train_ce"]), "train_ce_eval": acc["train"]["ce"],
            "train_val_acc_gap": acc["train"]["acc"] - acc["val"]["acc"],
            "train_val_macro_f1_gap": acc["train"]["macro_f1"] - acc["val"]["macro_f1"],
            "train_val_ce_gap": acc["train"]["ce"] - acc["val"]["ce"],
            "model_parameters": int(meta["model_parameters"]), "classifier_parameters": int(meta["classifier_parameters"]),
            "wall_seconds": float(meta.get("training_wall_seconds", time.perf_counter() - started)),
            "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": peak_alloc / (1024 ** 3), "peak_reserved_gib": peak_reserved / (1024 ** 3),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "best_accuracy_checkpoint": str(acc_path), "best_f1_checkpoint": str(f1_path),
            "best_accuracy_checkpoint_sha256": legacy.sha256_file(acc_path), "best_f1_checkpoint_sha256": legacy.sha256_file(f1_path),
            "best_accuracy_checkpoint_bytes": acc_path.stat().st_size, "best_f1_checkpoint_bytes": f1_path.stat().st_size,
            "run_metrics_path": str(metrics_path), "run_dir": str(run_dir),
            "initial_model_sha256": initial["model_state_sha256"], "initial_classifier_sha256": initial["classifier_state_sha256"],
            "semantic_cache_sha256": legacy.sha256_file(Path(cache["cache_path_abs"])),
            "feature_fingerprint": cache["feature_fingerprint"], "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"], "split_fingerprint": fingerprints["split_fingerprint"],
            "validation_rng_unchanged": acc["rng_unchanged"] and f1["rng_unchanged"], "test_evaluation": False, "error": "",
        }
        append_rows("runs.csv", [row], ("dataset", "condition", "seed"))
        append_rows("epoch_history.csv", [{"dataset": dataset, "condition": condition, "seed": seed, **epoch} for epoch in history], ("dataset", "condition", "seed", "epoch"))
        append_rows("classwise_metrics.csv", acc["class_rows"] + f1["class_rows"], ("dataset", "condition", "seed", "checkpoint", "class_id"))
        append_rows("confusion_matrices.csv", acc["confusion_rows"] + f1["confusion_rows"], ("dataset", "condition", "seed", "checkpoint", "true_class", "predicted_class"))
        append_rows("degree_group_metrics.csv", acc["degree_rows"] + f1["degree_rows"], ("dataset", "condition", "seed", "checkpoint", "degree_group"))
        view_rows, router_rows = _view_rows(dataset, condition, seed, "best_accuracy", acc["details"])
        more_views, more_routers = _view_rows(dataset, condition, seed, "best_f1", f1["details"])
        append_rows("view_diagnostics.csv", view_rows + more_views, ("dataset", "condition", "seed", "checkpoint", "modality", "view"))
        append_rows("router_diagnostics.csv", router_rows + more_routers, ("dataset", "condition", "seed", "checkpoint", "modality", "view"))
        transition_rows = []
        if condition != "A0_legacy":
            target, candidate, reference = acc["val"]["target"], acc["predictions"], PREDICTIONS[(dataset, seed)]
            for class_id in ["all", *range(int(safe.num_classes))]:
                mask = torch.ones_like(target, dtype=torch.bool) if class_id == "all" else target == class_id
                ref_ok = reference[mask] == target[mask]
                new_ok = candidate[mask] == target[mask]
                transition_rows.append({"dataset": dataset, "condition": condition, "seed": seed, "reference_condition": "A0_legacy", "class_id": class_id, "correct_to_wrong": int((ref_ok & ~new_ok).sum()), "wrong_to_correct": int((~ref_ok & new_ok).sum()), "unchanged_correct": int((ref_ok & new_ok).sum()), "unchanged_wrong": int((~ref_ok & ~new_ok).sum())})
        if transition_rows:
            append_rows("prediction_transitions.csv", transition_rows, ("dataset", "condition", "seed", "class_id"))
        intervention_rows = _view_interventions(acc, safe, dataset, condition, seed, device)
        if intervention_rows:
            append_rows("frozen_interventions.csv", intervention_rows, ("dataset", "condition", "seed", "intervention"))
        resource = [{"dataset": dataset, "condition": condition, "seed": seed, "wall_seconds": row["wall_seconds"], "epochs_completed": row["epochs_completed"], "mean_epoch_wall_seconds": row["wall_seconds"] / max(row["epochs_completed"], 1), "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_reserved, "peak_allocated_gib": row["peak_allocated_gib"], "peak_reserved_gib": row["peak_reserved_gib"], "cuda_device_name": row["cuda_device_name"], "parameter_count": params, "trainable_parameter_count": params}]
        append_rows("resource_profile.csv", resource, ("dataset", "condition", "seed"))

        gradient_path = DATA / "gradient_audit.json"
        gradient = json.loads(gradient_path.read_text()) if gradient_path.exists() else {"runs": []}
        if not any((r["dataset"], r["condition"], int(r["seed"])) == (dataset, condition, seed) for r in gradient["runs"]):
            gradient["runs"].append({"dataset": dataset, "condition": condition, "seed": seed, "first_step": grads})
        atomic_json(gradient_path, gradient)
        integrity_path = DATA / "artifact_integrity.json"
        integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {"checkpoints": []}
        for name, path in (("best_accuracy", acc_path), ("best_f1", f1_path)):
            integrity["checkpoints"].append({"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": name, "path": str(path), "sha256": legacy.sha256_file(path), "bytes": path.stat().st_size})
        atomic_json(integrity_path, integrity)
        atomic_json(run_dir / "status.json", {"status": "complete", "dataset": dataset, "condition": condition, "seed": seed, "completed_utc": utc_now(), "test_evaluation": False, "test_label_values_indexed": False})
        print(f"COMPLETE {dataset}/{condition}/seed_{seed} acc={row['val_acc']:.6f} f1={row['val_macro_f1']:.6f} epoch={row['best_accuracy_epoch']} peak={row['peak_allocated_gib']:.2f} GiB", flush=True)
        del raw, safe, acc, f1
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return row
    except Exception as exc:
        failure = {"dataset": dataset, "condition": condition, "seed": seed, "run_dir": str(run_dir), "time_utc": utc_now(), "error": repr(exc), "traceback": traceback.format_exc(), "test_evaluation": False}
        path = DATA / "failures.json"
        old = json.loads(path.read_text()) if path.exists() else {"failures": []}
        old["failures"].append(failure)
        atomic_json(path, old)
        atomic_json(run_dir / "status.json", {"status": "failed", **failure})
        raise


def run_batch(seeds: tuple[int, ...], physical_gpu: int) -> None:
    if seeds not in {(42,), (43, 44)}:
        raise ValueError("use seed42 as the first gate, then seeds43/44")
    DATA.mkdir(parents=True, exist_ok=True)
    gate_path = DATA / "seed42_gate.json"
    if seeds == (43, 44) and (not gate_path.is_file() or json.loads(gate_path.read_text()).get("status") != "pass"):
        raise RuntimeError("seeds43/44 require a passing reviewed seed42 gate")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {str(physical_gpu), f"{physical_gpu},"}:
        raise RuntimeError(f"launch must isolate physical GPU {physical_gpu}; CUDA_VISIBLE_DEVICES={visible!r}")
    caches = prepare_study(physical_gpu)
    device = torch.device("cuda:0")
    for seed in seeds:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                prior = read_csv(DATA / "runs.csv")
                if any(row["dataset"] == dataset and row["condition"] == condition and int(row["seed"]) == seed and row["status"] == "complete" for row in prior):
                    print(f"SKIP verified {dataset}/{condition}/seed_{seed}", flush=True)
                    continue
                legacy._assert_physical_gpu_idle(physical_gpu)
                failure_path = DATA / "failures.json"
                failures_before = json.loads(failure_path.read_text()).get("failures", []) if failure_path.exists() else []
                try:
                    run_one(dataset, condition, seed, caches[dataset], device)
                except Exception as exc:
                    failures_after = json.loads(failure_path.read_text()).get("failures", []) if failure_path.exists() else []
                    if len(failures_after) == len(failures_before):
                        failure = {
                            "dataset": dataset, "condition": condition, "seed": seed,
                            "stage": "runner_setup", "training_started": False,
                            "time_utc": utc_now(), "error": repr(exc),
                            "traceback": traceback.format_exc(), "test_evaluation": False,
                        }
                        failures_after.append(failure)
                        atomic_json(failure_path, {"failures": failures_after})
                    raise
        manifest = json.loads((DATA / "manifest.json").read_text())
        manifest.setdefault("completed_seed_batches", [])
        if seed not in manifest["completed_seed_batches"]:
            manifest["completed_seed_batches"].append(seed)
        manifest["last_batch_completed_utc"] = utc_now()
        atomic_json(DATA / "manifest.json", manifest)
    if seeds == (42,):
        runs = read_csv(DATA / "runs.csv")
        expected = {(d, c, 42) for d in DATASETS for c in CONDITIONS}
        found = {(r["dataset"], r["condition"], int(r["seed"])) for r in runs if int(r["seed"]) == 42 and r["status"] == "complete"}
        if found != expected:
            raise RuntimeError(f"seed42 gate is incomplete: missing {sorted(expected-found)}")
        checks = {d: [r for r in runs if r["dataset"] == d and int(r["seed"]) == 42] for d in DATASETS}
        for dataset, rows in checks.items():
            if len({r["initial_model_sha256"] for r in rows}) != 1 or len({r["initial_classifier_sha256"] for r in rows}) != 1 or len({r["split_fingerprint"] for r in rows}) != 1:
                raise RuntimeError(f"matched initialization/split gate failed: {dataset}")
        for row in runs:
            if int(row["seed"]) != 42:
                continue
            if row["test_evaluation"] != "False" or row["validation_rng_unchanged"] != "True":
                raise RuntimeError(f"test/RNG gate failed: {row['dataset']}/{row['condition']}")
            for checkpoint in (row["best_accuracy_checkpoint"], row["best_f1_checkpoint"]):
                if not Path(checkpoint).is_file():
                    raise RuntimeError(f"checkpoint missing: {checkpoint}")
        for path in (DATA / "gradient_audit.json", DATA / "artifact_integrity.json"):
            if not path.is_file():
                raise RuntimeError(f"missing run audit {path.name}")
        atomic_json(gate_path, {"status": "pass", "seed": 42, "completed_runs": 9, "generated_utc": utc_now(), "checks": ["finite full-graph histories", "shared initialization/split/cache pairing", "first-step semantic gradients", "checkpoint reload and validation reproduction", "Test evaluation disabled", "CUDA memory and training-step preflight passed"], "performance_threshold_used": False})
        print("SEED42_GATE pass: 9 runs; continuing seeds 43/44 requires explicit second launch", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int, default=1)
    args = parser.parse_args()
    run_batch(tuple(args.seeds), args.physical_gpu)


if __name__ == "__main__":
    main()
