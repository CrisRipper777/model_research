#!/usr/bin/env python3
"""Execute frozen Hybrid-CGE V1.1 validation-only training batches."""

from __future__ import annotations

import argparse
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
from scripts.preflight_hybrid_cge_v11 import (  # noqa: E402
    CONDITIONS, DATASETS, DATA_DIR, SEEDS, STUDY, build_config, load_model, mask_test_labels, state_hash,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels, run_nc  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


OUTPUT = ROOT / "outputs/hybrid_cge_v11/runs"
PROTOCOL = legacy.PROTOCOL


def _git_revision(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


IMPLEMENTATION_FREEZE_COMMIT = _git_revision(
    "log", "--all", "-1", "--format=%H", "--grep=Hybrid-CGE V1.1 Implementation Freeze"
)
FORMAL_SOURCE_COMMIT = _git_revision("rev-parse", "HEAD")
if not IMPLEMENTATION_FREEZE_COMMIT:
    IMPLEMENTATION_FREEZE_COMMIT = "FREEZE_COMMIT_NOT_FOUND"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def append_csv(path: Path, rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> None:
    previous = read_csv(path)
    present = {tuple(row.get(k, "") for k in key_fields) for row in previous}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in key_fields)
        if key not in present:
            previous.append(row)
            present.add(key)
    fields = list(dict.fromkeys(k for row in previous for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(previous)
    temporary.replace(path)


def metric_bundle(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor,
                  num_classes: int, eval_labels: list[int]) -> dict[str, Any]:
    indices = idx.detach().cpu().long()
    target = labels[indices].detach().cpu().long()
    if bool((target < 0).any()):
        raise RuntimeError("masked Test sentinel reached Train/Validation metrics")
    selected = logits[indices].detach().cpu()
    prediction = selected.argmax(-1)
    return {
        "acc": float((prediction == target).float().mean()),
        "macro_f1": float(f1_score(target.numpy(), prediction.numpy(), labels=eval_labels, average="macro", zero_division=0)),
        "ce": float(torch.nn.functional.cross_entropy(selected, target)),
        "target": target, "prediction": prediction,
    }


def analyze_checkpoint(dataset: str, condition: str, seed: int, name: str,
                       checkpoint_path: Path, safe_data, cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("protocol_version") != PROTOCOL or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"checkpoint provenance mismatch: {dataset}/{condition}/seed{seed}/{name}")
    cfg = build_config(dataset, condition, seed, cache, str(device))
    set_seed(seed)
    model = load_model(cfg, safe_data, cache, device)
    head = torch.nn.Linear(model.out_dim, int(safe_data.num_classes)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()

    x = safe_data.x.to(device)
    edges = safe_data.edge_index.to(device)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    with torch.no_grad():
        z, _, _, _, info = model(x, edges, return_details=True)
        logits = head(z).detach().cpu()
    rng_unchanged = torch.equal(cpu_before, torch.random.get_rng_state())
    if cuda_before is not None:
        rng_unchanged &= torch.equal(cuda_before, torch.cuda.get_rng_state(device))
    if not rng_unchanged:
        raise RuntimeError(f"checkpoint evaluation changed RNG: {dataset}/{condition}/{seed}/{name}")

    eval_labels = _resolve_nc_eval_labels(safe_data, include_test=False)
    val = metric_bundle(logits, safe_data.y, safe_data.val_idx, int(safe_data.num_classes), eval_labels)
    train = metric_bundle(logits, safe_data.y, safe_data.train_idx, int(safe_data.num_classes), eval_labels)
    checkpoint_metrics = payload["metrics"]
    for key, metric_key in (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce")):
        if abs(float(checkpoint_metrics[key]) - val[metric_key]) > 1.0e-6:
            raise RuntimeError(f"reloaded checkpoint {key} mismatch: {dataset}/{condition}/{seed}/{name}")
    if any(not math.isfinite(metric[metric_key]) for metric in (val, train) for metric_key in ("acc", "macro_f1", "ce")):
        raise RuntimeError("non-finite Train/Validation checkpoint metric")

    class_rows: list[dict[str, Any]] = []
    precision, recall, class_f1, support = precision_recall_fscore_support(
        val["target"].numpy(), val["prediction"].numpy(), labels=list(range(int(safe_data.num_classes))), zero_division=0,
    )
    predicted_counts = np.bincount(val["prediction"].numpy(), minlength=int(safe_data.num_classes))
    for class_id in range(int(safe_data.num_classes)):
        class_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "checkpoint": name,
            "class_id": class_id, "precision": float(precision[class_id]), "recall": float(recall[class_id]),
            "f1": float(class_f1[class_id]), "support": int(support[class_id]),
            "predicted_count": int(predicted_counts[class_id]),
        })
    matrix = confusion_matrix(
        val["target"].numpy(), val["prediction"].numpy(), labels=list(range(int(safe_data.num_classes)))
    )
    confusion_rows = [
        {"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": name,
         "true_class": i, "predicted_class": j, "count": int(matrix[i, j])}
        for i in range(matrix.shape[0]) for j in range(matrix.shape[1])
    ]
    diagnostics = info.get("details", {}).get("modalities", {})
    diagnostic_rows = []
    for modality, values in diagnostics.items():
        row = {"dataset": dataset, "condition": condition, "seed": seed, "checkpoint": name, "modality": modality}
        for key, value in values.items():
            if isinstance(value, (str, bool, int, float)):
                row[key] = value
            elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                row[key] = json.dumps(value, separators=(",", ":"))
            elif isinstance(value, dict):
                row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        diagnostic_rows.append(row)
    early_cross = info.get("details", {}).get("early_physical_cross", {})
    for modality, values in early_cross.items():
        diagnostic_rows.append({
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "checkpoint": name,
            "modality": f"early_physical_cross_{modality}",
            **values,
        })

    return {
        "payload": payload, "val": val, "train": train, "logits": logits,
        "rng_unchanged": rng_unchanged, "class_rows": class_rows,
        "confusion_rows": confusion_rows, "diagnostic_rows": diagnostic_rows,
    }


def _next_run_dir(dataset: str, condition: str, seed: int) -> Path:
    root = OUTPUT / dataset / condition / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = root / f"attempt_{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1


def run_one(dataset: str, condition: str, seed: int, cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    cfg = build_config(dataset, condition, seed, cache, str(device))
    raw_data = load_mag_data(cfg, "nc", seed)
    safe_data = mask_test_labels(raw_data)
    fingerprints = legacy.dataset_fingerprints(raw_data)
    if bool((safe_data.y[raw_data.test_idx] != -100).any()):
        raise RuntimeError(f"Test labels are not masked: {dataset}/seed{seed}")
    if not torch.equal(safe_data.y[raw_data.train_idx], raw_data.y[raw_data.train_idx]):
        raise RuntimeError("Train labels changed while masking Test labels")

    run_dir = _next_run_dir(dataset, condition, seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    accuracy_path, f1_path, metrics_path = run_dir / "best_accuracy.pt", run_dir / "best_f1.pt", run_dir / "run_metrics.json"
    cfg.task.save_ckpt_path = str(accuracy_path)
    cfg.task.best_f1_ckpt_path = str(f1_path)
    cfg.task.run_metrics_path = str(metrics_path)
    cache_sha = legacy.sha256_file(Path(cache["cache_path_abs"]))
    cfg.task.extended_checkpoint_metadata.update({
        "study": "Hybrid-CGE V1.1", "dataset": dataset, "condition": condition, "seed": seed,
        "test_evaluation": False, "test_label_values_indexed": False,
        "masked_test_index_count": int(raw_data.test_idx.numel()),
        "dataset_fingerprints": fingerprints, "cache_sha256": cache_sha,
        "implementation_freeze_commit": IMPLEMENTATION_FREEZE_COMMIT,
        "formal_source_commit": FORMAL_SOURCE_COMMIT,
    })
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    atomic_json(run_dir / "status.json", {
        "status": "running", "dataset": dataset, "condition": condition, "seed": seed,
        "started_utc": utc_now(), "test_evaluation": False,
    })

    # Independent pre-training hash must agree with the peer condition under
    # the same Dataset x Seed; model construction precedes the external head.
    set_seed(seed)
    initial_model = load_model(cfg, safe_data, cache, torch.device("cpu"))
    initial_head = torch.nn.Linear(initial_model.out_dim, int(safe_data.num_classes))
    initial_model_hash, initial_head_hash = state_hash(initial_model), state_hash(initial_head)
    initial_total_model_parameters = sum(p.numel() for p in initial_model.parameters())
    initial_trainable_model_parameters = sum(p.numel() for p in initial_model.parameters() if p.requires_grad)
    if any("role_film" in name for name, _ in initial_model.named_parameters()):
        raise RuntimeError("Hybrid-CGE V1.1 must not register Role-FiLM parameters")
    del initial_model, initial_head

    logger = legacy.setup_logger(run_dir / "train.log")
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            # CUDA is lazily initialized. Availability/device-count checks do
            # not create a context, and resetting memory stats before the
            # first CUDA allocation can raise "Invalid device argument".
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        run_nc(cfg, safe_data, device, logger)
        if not accuracy_path.is_file() or not f1_path.is_file() or not metrics_path.is_file():
            raise RuntimeError("NC trainer did not save both validation checkpoints and run metrics")
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_payload.get("protocol_version") != PROTOCOL or len(metrics_payload.get("runs", [])) != 1:
            raise RuntimeError("unexpected run metrics protocol or number of runs")
        run = metrics_payload["runs"][0]
        metadata = run["metadata"]
        if int(run["seed"]) != seed or metadata.get("study_provenance", {}).get("test_evaluation") is not False:
            raise RuntimeError("seed mismatch or Test evaluation was enabled")
        if any(key.startswith("test_") for key in run.get("metrics", {})):
            raise RuntimeError("Test metrics were emitted by NC Trainer")
        if float(metadata.get("unimodal_auxiliary_weight", -1.0)) != 0.3:
            raise RuntimeError("NC Trainer did not apply the frozen 0.3 unimodal CE weight")
        history = metadata.get("epoch_history", [])
        if not history or not legacy._finite_history(history):
            raise RuntimeError("empty or non-finite training/validation history")
        if any("train_unimodal_ce" not in epoch or not math.isfinite(float(epoch["train_unimodal_ce"])) for epoch in history):
            raise RuntimeError("unimodal CE missing/non-finite in epoch history")
        if metadata["initial_parameter_audit"]["model_state_sha256"] != initial_model_hash:
            raise RuntimeError("NC Trainer initial model hash differs from pre-run public initialization")
        if metadata["initial_parameter_audit"]["classifier_state_sha256"] != initial_head_hash:
            raise RuntimeError("NC Trainer classifier hash differs from matched initial head")

        gradients = metadata.get("gradient_diagnostics")
        if not isinstance(gradients, dict):
            raise RuntimeError("NC Trainer did not persist first-step gradient diagnostics")
        required_gradients = (
            "physical_cross_coefficients",
            "cross_projections.visual_to_text.0.weight",
            "cross_projections.text_to_visual.0.weight",
            "node_router.3.weight", "view_router.2.weight", "gamma_s_raw",
            "query_projections.text.weight", "query_projections.visual.weight",
            "key_projections.text.weight", "key_projections.visual.weight",
            "source_selector.2.weight", "channel_gate.2.weight", "comp_router.3.weight",
            "unimodal_heads.text.weight", "unimodal_heads.visual.weight",
        )
        for name in required_gradients:
            if not gradients.get(name, {}).get("gradient_nonzero", False):
                raise RuntimeError(f"required first-step nonzero gradient missing: {name}")
        # A zero tau gradient is allowed when relative control never binds.

        accuracy = analyze_checkpoint(dataset, condition, seed, "best_accuracy", accuracy_path, safe_data, cache, device)
        best_f1 = analyze_checkpoint(dataset, condition, seed, "best_f1", f1_path, safe_data, cache, device)
        train_acc = accuracy["train"]
        val_acc = accuracy["val"]
        train_f1 = best_f1["train"]
        val_f1 = best_f1["val"]
        train_ce_mode = float(accuracy["payload"]["metrics"]["train_ce"])
        initial_audit = metadata["initial_parameter_audit"]
        final_audit = metadata["final_parameter_audit"]
        trainable_count = initial_trainable_model_parameters + int(metadata["classifier_parameters"])
        best_state = accuracy["payload"]["model_state"]
        b_norm = float(best_state["physical_cross_coefficients"].float().norm())
        tau_raw = best_state["tau_raw"].float()
        tau_values = 0.1 + 0.7 * torch.sigmoid(tau_raw)
        run_row = {
            "dataset": dataset, "condition": condition, "seed": seed, "status": "complete",
            "best_accuracy_epoch": int(accuracy["payload"]["epoch"]),
            "best_f1_epoch": int(best_f1["payload"]["epoch"]),
            "epochs_completed": int(metadata["epochs_completed"]),
            "val_acc": val_acc["acc"], "val_macro_f1": val_acc["macro_f1"], "val_ce": val_acc["ce"],
            "best_f1_val_acc": val_f1["acc"], "best_f1_val_macro_f1": val_f1["macro_f1"], "best_f1_val_ce": val_f1["ce"],
            "train_acc": train_acc["acc"], "train_macro_f1": train_acc["macro_f1"],
            "train_ce": train_ce_mode, "train_ce_eval": train_acc["ce"],
            "train_val_acc_gap": train_acc["acc"] - val_acc["acc"],
            "train_val_macro_f1_gap": train_acc["macro_f1"] - val_acc["macro_f1"],
            "train_val_ce_gap": train_acc["ce"] - val_acc["ce"],
            "best_f1_train_acc": train_f1["acc"], "best_f1_train_macro_f1": train_f1["macro_f1"],
            "best_f1_train_ce": train_f1["ce"],
            "best_f1_train_val_acc_gap": train_f1["acc"] - val_f1["acc"],
            "best_f1_train_val_macro_f1_gap": train_f1["macro_f1"] - val_f1["macro_f1"],
            "best_f1_train_val_ce_gap": train_f1["ce"] - val_f1["ce"],
            "model_parameters": int(metadata["model_parameters"]),
            "model_total_parameters": initial_total_model_parameters,
            "trainable_optimizer_parameters": trainable_count,
            "physical_cross_coefficients_l2": b_norm,
            "tau_text": float(tau_values[0]), "tau_visual": float(tau_values[1]),
            "late_cross_dropout_probability": float(cfg.model.late_cross_dropout),
            "wall_seconds": float(metadata.get("training_wall_seconds", time.perf_counter() - started)),
            "mean_epoch_wall_seconds": float(metadata.get("mean_epoch_wall_seconds", 0.0)),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "cuda_device_name": metadata.get("cuda_device_name", "CPU"),
            "best_accuracy_checkpoint": str(accuracy_path), "best_f1_checkpoint": str(f1_path),
            "best_accuracy_checkpoint_sha256": legacy.sha256_file(accuracy_path),
            "best_f1_checkpoint_sha256": legacy.sha256_file(f1_path),
            "best_accuracy_checkpoint_bytes": accuracy_path.stat().st_size,
            "best_f1_checkpoint_bytes": f1_path.stat().st_size,
            "run_metrics_path": str(metrics_path), "run_dir": str(run_dir),
            "initial_model_sha256": initial_model_hash,
            "initial_classifier_sha256": initial_head_hash,
            "semantic_cache_sha256": cache_sha,
            "feature_fingerprint": cache["feature_fingerprint"],
            "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "validation_rng_unchanged": accuracy["rng_unchanged"] and best_f1["rng_unchanged"],
            "test_evaluation": False,
            "implementation_freeze_commit": IMPLEMENTATION_FREEZE_COMMIT,
            "formal_source_commit": FORMAL_SOURCE_COMMIT,
            "error": "",
        }
        append_csv(DATA_DIR / "runs.csv", [run_row], ("dataset", "condition", "seed"))
        append_csv(DATA_DIR / "epoch_history.csv", [
            {"dataset": dataset, "condition": condition, "seed": seed, **entry}
            for entry in history
        ], ("dataset", "condition", "seed", "epoch"))
        append_csv(DATA_DIR / "classwise_metrics.csv", accuracy["class_rows"] + best_f1["class_rows"],
                   ("dataset", "condition", "seed", "checkpoint", "class_id"))
        append_csv(DATA_DIR / "confusion_matrices.csv", accuracy["confusion_rows"] + best_f1["confusion_rows"],
                   ("dataset", "condition", "seed", "checkpoint", "true_class", "predicted_class"))
        append_csv(DATA_DIR / "mechanism_diagnostics.csv", accuracy["diagnostic_rows"] + best_f1["diagnostic_rows"],
                   ("dataset", "condition", "seed", "checkpoint", "modality"))
        append_csv(DATA_DIR / "resource_profile.csv", [{
            "dataset": dataset, "condition": condition, "seed": seed,
            "wall_seconds": run_row["wall_seconds"], "epochs_completed": run_row["epochs_completed"],
            "mean_epoch_wall_seconds": run_row["mean_epoch_wall_seconds"],
            "peak_allocated_bytes": run_row["peak_allocated_bytes"],
            "peak_reserved_bytes": run_row["peak_reserved_bytes"],
            "model_parameters": run_row["model_parameters"], "model_total_parameters": initial_total_model_parameters,
            "trainable_optimizer_parameters": trainable_count,
            "physical_cross_coefficients_l2": b_norm,
            "tau_text": float(tau_values[0]), "tau_visual": float(tau_values[1]),
            "late_cross_dropout_probability": float(cfg.model.late_cross_dropout),
            "device": run_row["cuda_device_name"],
        }], ("dataset", "condition", "seed"))
        gradient_path = DATA_DIR / "gradient_audit.json"
        gradient_record = json.loads(gradient_path.read_text()) if gradient_path.exists() else {"runs": []}
        if not any((row["dataset"], row["condition"], int(row["seed"])) == (dataset, condition, seed) for row in gradient_record["runs"]):
            gradient_record["runs"].append({"dataset": dataset, "condition": condition, "seed": seed,
                                             "first_step": gradients, "final_parameter_audit": final_audit})
        atomic_json(gradient_path, gradient_record)
        integrity_path = DATA_DIR / "artifact_integrity.json"
        integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {"checkpoints": []}
        for checkpoint_name, path in (("best_accuracy", accuracy_path), ("best_f1", f1_path)):
            integrity["checkpoints"].append({
                "dataset": dataset, "condition": condition, "seed": seed,
                "checkpoint": checkpoint_name, "path": str(path),
                "sha256": legacy.sha256_file(path), "bytes": path.stat().st_size,
            })
        atomic_json(integrity_path, integrity)
        atomic_json(run_dir / "status.json", {
            "status": "complete", "dataset": dataset, "condition": condition, "seed": seed,
            "completed_utc": utc_now(), "test_evaluation": False,
            "test_label_values_indexed": False,
            "val_acc": run_row["val_acc"], "val_macro_f1": run_row["val_macro_f1"], "val_ce": run_row["val_ce"],
        })
        print(f"COMPLETE {dataset}/{condition}/seed_{seed}: val_acc={run_row['val_acc']:.6f} val_f1={run_row['val_macro_f1']:.6f} epoch={run_row['best_accuracy_epoch']}", flush=True)
        return run_row
    except Exception as exc:
        failures_path = DATA_DIR / "failures.json"
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else {"failures": []}
        record = {
            "dataset": dataset, "condition": condition, "seed": seed,
            "time_utc": utc_now(), "run_dir": str(run_dir), "error": repr(exc),
            "traceback": traceback.format_exc(), "training_started": metrics_path.exists(),
            "test_evaluation": False,
        }
        failures["failures"].append(record)
        atomic_json(failures_path, failures)
        atomic_json(run_dir / "status.json", {"status": "failed", **record})
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        del safe_data, raw_data


def _seed42_gate() -> None:
    rows = read_csv(DATA_DIR / "runs.csv")
    expected = {(dataset, condition, 42) for dataset in DATASETS for condition in CONDITIONS}
    found = {(row["dataset"], row["condition"], int(row["seed"])) for row in rows
             if int(row["seed"]) == 42 and row["status"] == "complete"}
    if found != expected:
        raise RuntimeError(f"seed42 integrity gate missing cells: {sorted(expected - found)}")
    for dataset in DATASETS:
        paired = [row for row in rows if row["dataset"] == dataset and int(row["seed"]) == 42]
        for field in (
            "initial_model_sha256", "initial_classifier_sha256", "split_fingerprint",
            "semantic_cache_sha256", "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint",
        ):
            if len({row[field] for row in paired}) != 1:
                raise RuntimeError(f"seed42 pairing check failed for {dataset}/{field}")
        for row in paired:
            if row["test_evaluation"] != "False" or row["validation_rng_unchanged"] != "True":
                raise RuntimeError(f"seed42 Test/RNG check failed for {dataset}/{row['condition']}")
            for name in ("best_accuracy_checkpoint", "best_f1_checkpoint"):
                if not Path(row[name]).is_file():
                    raise RuntimeError(f"seed42 checkpoint missing: {row[name]}")
    atomic_json(DATA_DIR / "seed42_gate.json", {
        "status": "pass", "seed": 42, "completed_runs": 6,
        "generated_utc": utc_now(), "performance_threshold_used": False,
        "checks": ["finite train/validation history", "matching initialization/split/cache",
                   "auxiliary and sparse-attention gradients", "checkpoint reload metric reproduction",
                   "Test labels masked/Test evaluation disabled", "full-graph GPU preflight passed"],
    })


def run_batch(seeds: tuple[int, ...], physical_gpu: int) -> None:
    if seeds not in {(42,), (43, 44)}:
        raise ValueError("run seed 42 as a separate integrity batch, then seeds 43/44")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(f"launch with CUDA_VISIBLE_DEVICES={physical_gpu}")
    preflight_path = DATA_DIR / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("GPU preflight must pass before formal runs")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "pass" or int(preflight.get("physical_gpu", -1)) != physical_gpu:
        raise RuntimeError("GPU preflight did not pass on the requested physical GPU")
    legacy._assert_physical_gpu_idle(physical_gpu)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    caches = legacy.verify_cache_entries()
    for seed in seeds:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                already = [row for row in read_csv(DATA_DIR / "runs.csv")
                           if (row.get("dataset"), row.get("condition"), row.get("seed"), row.get("status")) ==
                           (dataset, condition, str(seed), "complete")]
                if already:
                    continue
                legacy._assert_physical_gpu_idle(physical_gpu)
                run_one(dataset, condition, seed, caches[dataset], device)
        if seed == 42:
            _seed42_gate()
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "study": "Hybrid-CGE V1.1", "branch": "exp/hybrid_cge_v11",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "implementation_freeze_commit": IMPLEMENTATION_FREEZE_COMMIT,
        "formal_source_commit": FORMAL_SOURCE_COMMIT,
        "protocol_version": PROTOCOL, "conditions": list(CONDITIONS), "seeds": list(SEEDS),
        "formal_run_count": 18, "test_evaluation": False,
        "environment": legacy.environment_record(), "physical_gpu": physical_gpu,
        "last_batch_seeds": list(seeds), "last_batch_completed_utc": utc_now(),
    })
    atomic_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int, default=1)
    args = parser.parse_args()
    run_batch(tuple(args.seeds), args.physical_gpu)


if __name__ == "__main__":
    main()
