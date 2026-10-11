#!/usr/bin/env python3
"""Validation-only preflight and first 15-run PACE-MAG V1 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRANCH = "exp/pace_mag_v1_relation_view_experts"
BASE_SHA = "aa62807c113fde17db058d05d558d2635bccc90b"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = (
    "e0_source_agnostic",
    "e1_no_cross_modal",
    "e2_full",
    "e3_single_expert_control",
    "e4_static_routing",
)
SEED = 42
PROTOCOL = "unified_full_graph_nc_v1"
RESEARCH = ROOT / "research/pace_mag_v1"
DATA_DIR = RESEARCH / "data"
OUTPUT_ROOT = ROOT / "outputs/pace_mag_v1_pilot"
SUMMARY_PATH = ROOT / "research/psce_mag_v7b/data/candidate_graph_summary.json"


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _resolved_yaml(cfg) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)


def _build_config(dataset: str, variant: str, *, device: str, seed: int = SEED):
    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create(
        {
            "paths": root_cfg.paths,
            "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
            "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
            "model": OmegaConf.load(ROOT / "configs/model/pace_mag_v1.yaml"),
            "logging": root_cfg.logging,
            "seed": int(seed),
            "num_runs": 1,
            "device": str(device),
        }
    )
    cfg.dataset.name = dataset
    cfg.model.variant = variant
    cfg.task.name = "nc"
    cfg.task.protocol_version = PROTOCOL
    cfg.task.training_mode = "full_graph"
    cfg.task.optimizer = "adamw"
    cfg.task.lr = 1e-3
    cfg.task.weight_decay = 1e-4
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 1e-4
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.inference_mode = "full"
    cfg.task.inference_batch_size = 4096
    cfg.task.scheduler = None
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 1.0
    OmegaConf.resolve(cfg)
    return cfg


def _cache_entries() -> dict[str, dict[str, Any]]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {item["dataset"]: item for item in summary["datasets"] if item["dataset"] in DATASETS}


def _actual_feature_fingerprint(dataset: str, text: np.ndarray, visual: np.ndarray, sources: dict[str, Path]) -> str:
    from src.data.semantic_candidates import _json_fingerprint, _source_descriptor

    identity = {
        "dataset": dataset,
        "num_nodes": int(text.shape[0]),
        "text_dim": int(text.shape[1]),
        "visual_dim": int(visual.shape[1]),
        "text": _source_descriptor(sources["text"], np.ascontiguousarray(text, dtype=np.float32)),
        "visual": _source_descriptor(sources["visual"], np.ascontiguousarray(visual, dtype=np.float32)),
    }
    return _json_fingerprint(identity)


def _candidate_inputs(dataset: str):
    # The inherited helper deliberately reads topology/features only, without labels or splits.
    from scripts.run_psce_mag_v7b1_corrective import load_candidate_inputs

    return load_candidate_inputs(dataset)


def _verify_cache_entry(dataset: str, entry: dict[str, Any], *, save_audit: bool = True) -> dict[str, Any]:
    from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b

    text, visual, _physical, sources, info = _candidate_inputs(dataset)
    actual_feature_fp = _actual_feature_fingerprint(dataset, text, visual, sources)
    if actual_feature_fp != entry["feature_fingerprint"]:
        raise ValueError(f"{dataset}: current feature files do not match the frozen V7B feature fingerprint")
    cache_path = ROOT / entry["cache_path"]
    cache = load_semantic_candidate_cache_v7b(
        cache_path,
        expected_dataset=dataset,
        expected_feature_fingerprint=actual_feature_fp,
        expected_candidate_fingerprint=entry["candidate_fingerprint"],
        expected_v7b_fingerprint=entry["v7b_fingerprint"],
    )
    if int(cache["metadata"]["num_nodes"]) != int(info["num_nodes"]):
        raise ValueError(f"{dataset}: candidate cache node count differs from the source graph")
    audit = {
        "dataset": dataset,
        "cache_path": str(cache_path.relative_to(ROOT)),
        "feature_fingerprint": actual_feature_fp,
        "candidate_fingerprint": cache["metadata"]["candidate_fingerprint"],
        "v7b_fingerprint": cache["metadata"]["v7b_fingerprint"],
        "num_nodes": int(info["num_nodes"]),
        "unique_semantic_edges": int(cache["unique_edge_index"].size(1)),
        "directed_semantic_edges": int(cache["edge_index"].size(1)),
        "source_flags_count_0_1_2": {
            "text": {str(i): int((cache["source_flags"][:, 0] == i).sum()) for i in range(3)},
            "visual": {str(i): int((cache["source_flags"][:, 1] == i).sum()) for i in range(3)},
        },
        "label_free_cache_provenance": bool(cache["metadata"].get("candidate_phase_label_access") is False),
        "cache_contains_labels": bool(cache["metadata"].get("cache_contains_labels", False)),
    }
    if save_audit:
        write_json(DATA_DIR / "cache_audit.json", _merge_cache_audit(audit))
    return {**entry, "cache_path_abs": str(cache_path), "cache_audit": audit}


def _merge_cache_audit(row: dict[str, Any]) -> dict[str, Any]:
    path = DATA_DIR / "cache_audit.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"datasets": []}
    rows = {item["dataset"]: item for item in prior.get("datasets", [])}
    rows[row["dataset"]] = row
    return {"recorded_at_utc": now_utc(), "datasets": [rows[key] for key in DATASETS if key in rows]}


def _choose_device(requested: str) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("PACE-MAG preflight requires an available CUDA device")
    if requested != "auto":
        device = torch.device(requested)
        if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f"requested CUDA device is unavailable: {requested}")
        return str(device)
    candidates = []
    for index in range(torch.cuda.device_count()):
        free_bytes, _total = torch.cuda.mem_get_info(index)
        candidates.append((int(free_bytes), index))
    return f"cuda:{max(candidates)[1]}"


def _gpu_snapshot(device: str) -> dict[str, Any]:
    device_obj = torch.device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)
    props = torch.cuda.get_device_properties(device_obj)
    return {
        "device": device,
        "name": props.name,
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device_obj)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device_obj)),
    }


def _environment(device: str) -> dict[str, Any]:
    import faiss
    import torch_geometric

    return {
        "recorded_at_utc": now_utc(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "dgl": __import__("dgl").__version__,
        "faiss": faiss.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "selected_gpu": _gpu_snapshot(device),
        "nvidia_smi": _run_nvidia_smi(),
        "other_gpu_processes_stopped": False,
    }


def _run_nvidia_smi() -> str:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15, check=True)
        return result.stdout
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def _load_training_data(dataset: str, variant: str, device: str, cache_entry: dict[str, Any]):
    from src.data import load_mag_data

    cfg = _build_config(dataset, variant, device=device)
    data = load_mag_data(cfg, "nc", SEED)
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset}: NC loader did not provide both modalities")
    if data.num_nodes != int(cache_entry["num_nodes"] if "num_nodes" in cache_entry else data.num_nodes):
        raise ValueError(f"{dataset}: NC data node count and candidate cache disagree")
    return cfg, data


def _prepare_cfg_cache(cfg, dataset: str, variant: str, entry: dict[str, Any], run_dir: Path) -> str:
    cfg.model.semantic_cache_path = entry["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
    cfg.task.save_ckpt_path = str(run_dir / "best.pt")
    cfg.task.run_metrics_path = str(run_dir / "run_metrics.json")
    cfg.paths.output_root = str(run_dir / "hydra")
    cfg.dataset.name = dataset
    cfg.model.variant = variant
    return sha256_bytes(_resolved_yaml(cfg).encode("utf-8"))


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"pace_mag_v1.{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def run_preflight(dataset: str, device: str, entry: dict[str, Any]) -> dict[str, Any]:
    from src.models import build_model
    from src.tasks.common import build_optimizer
    from src.utils.seeds import set_seed

    cfg, data = _load_training_data(dataset, "e2_full", device, entry)
    run_dir = OUTPUT_ROOT / "preflight" / dataset
    checkpoint_path = run_dir / "preflight_checkpoint.pt"
    attempt = 2
    while checkpoint_path.exists():
        run_dir = OUTPUT_ROOT / "preflight" / f"{dataset}_attempt_{attempt:02d}"
        checkpoint_path = run_dir / "preflight_checkpoint.pt"
        attempt += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    config_sha = _prepare_cfg_cache(cfg, dataset, "e2_full", entry, run_dir)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }
    set_seed(SEED)
    model = build_model(cfg, info)
    model.load_semantic_cache(
        entry["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=entry["feature_fingerprint"],
        expected_candidate_fingerprint=entry["candidate_fingerprint"],
        expected_v7b_fingerprint=entry["v7b_fingerprint"],
    )
    model = model.to(device).train()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    labels = data.y[train_idx.cpu()].to(device)
    params_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    gpu_before = _gpu_snapshot(device)
    if gpu_before["free_bytes"] < 2 * 1024**3:
        raise RuntimeError(f"preflight requires >=2 GiB free on selected GPU; observed {gpu_before['free_bytes']} bytes")
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux_loss, _ = model(x, edge_index)
    loss = torch.nn.functional.cross_entropy(classifier(z[train_idx]), labels) + aux_loss
    loss.backward()
    gradient_audit = model.capture_gradient_diagnostics()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), 1.0, error_if_nonfinite=True)
    optimizer.step()
    update_groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        row = update_groups.setdefault(group, {"changed_parameter_tensors": 0, "changed_elements": 0, "max_abs_update": 0.0})
        before = params_before[name]
        difference = (parameter.detach() - before).abs()
        if bool((difference > 0).any()):
            row["changed_parameter_tensors"] += 1
            row["changed_elements"] += int((difference > 0).sum().item())
            row["max_abs_update"] = max(row["max_abs_update"], float(difference.max().cpu()))
    torch.save({"model_state": model.state_dict(), "head_state": classifier.state_dict(), "metrics": {"val_acc": 0.0, "val_macro_f1": 0.0}}, checkpoint_path)
    model.eval()
    classifier.eval()
    with torch.no_grad():
        expected = model(x, edge_index)[0]
    restored = build_model(cfg, info)
    restored.load_semantic_cache(
        entry["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=entry["feature_fingerprint"],
        expected_candidate_fingerprint=entry["candidate_fingerprint"],
        expected_v7b_fingerprint=entry["v7b_fingerprint"],
    )
    restored = restored.to(device).eval()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["model_state"])
    with torch.no_grad():
        observed = restored(x, edge_index)[0]
    torch.cuda.synchronize(torch.device(device))
    gpu_after = _gpu_snapshot(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(torch.device(device)))
    peak_reserved = int(torch.cuda.max_memory_reserved(torch.device(device)))
    passed_core_groups = ("projectors", "cross_modal", "experts", "router", "fusion")
    update_pass = all(
        any(
            row["changed_parameter_tensors"] > 0
            for name, row in update_groups.items()
            if name == group or (group == "fusion" and name in {"fusion", "fusion_norm", "skip"})
        )
        for group in passed_core_groups
    )
    summary = {
        "recorded_at_utc": now_utc(),
        "dataset": dataset,
        "variant": "e2_full",
        "seed": SEED,
        "protocol": PROTOCOL,
        "task_evaluate_test": False,
        "test_metrics_computed": False,
        "test_labels_used_in_loss_or_diagnostics": False,
        "device": device,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "one_complete_forward": bool(torch.isfinite(z).all()),
        "one_complete_backward": bool(torch.isfinite(loss) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())),
        "adamw_optimizer_step": True,
        "loss": float(loss.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
        "gradient_audit": gradient_audit,
        "parameter_update_audit": update_groups,
        "core_update_check_passed": bool(update_pass),
        "checkpoint_saved": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_reloaded_inference_max_abs_difference": float((expected - observed).abs().max().cpu()),
        "checkpoint_reload_consistent": bool(torch.allclose(expected, observed, atol=1e-6, rtol=1e-6)),
        "config_sha256": config_sha,
        "cache_fingerprints": {
            "feature": entry["feature_fingerprint"],
            "candidate": entry["candidate_fingerprint"],
            "v7b": entry["v7b_fingerprint"],
        },
    }
    summary["passed"] = bool(
        summary["one_complete_forward"]
        and summary["one_complete_backward"]
        and summary["core_update_check_passed"]
        and summary["checkpoint_reload_consistent"]
        and all(row["gradient_finite"] for row in gradient_audit.values())
    )
    write_json(DATA_DIR / "implementation_checks.json", summary)
    write_json(DATA_DIR / "gradient_audit.json", {"preflight": gradient_audit, "parameter_updates": update_groups, "passed": summary["passed"]})
    if not summary["passed"]:
        raise RuntimeError(f"PACE-MAG GPU preflight failed: {summary}")
    return summary


def _run_one(dataset: str, variant: str, device: str, entry: dict[str, Any]) -> dict[str, Any]:
    from src.tasks.nc import run_nc
    from src.data import load_mag_data

    run_dir = OUTPUT_ROOT / "runs" / dataset / f"seed_{SEED}" / variant
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing Pilot output: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_config(dataset, variant, device=device)
    config_sha = _prepare_cfg_cache(cfg, dataset, variant, entry, run_dir)
    (run_dir / "resolved_config.yaml").write_text(_resolved_yaml(cfg), encoding="utf-8")
    logger = _setup_logger(run_dir)
    data = load_mag_data(cfg, "nc", SEED)
    if data.num_nodes != int(entry["cache_audit"]["num_nodes"]):
        raise ValueError(f"{dataset}: NC training graph and semantic cache node count disagree")
    before = _gpu_snapshot(device)
    if before["free_bytes"] < 2 * 1024**3:
        raise RuntimeError(f"less than 2 GiB free on selected GPU before {dataset}/{variant}; no process was stopped")
    started = time.perf_counter()
    result = run_nc(cfg, data, torch.device(device), logger)
    elapsed = time.perf_counter() - started
    metrics_payload = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
    if len(metrics_payload.get("runs", [])) != 1:
        raise RuntimeError("NC runner did not save exactly one run row")
    run_row = metrics_payload["runs"][0]
    metrics = run_row["metrics"]
    metadata = run_row["metadata"]
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("NC runner did not save the selected Validation checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(checkpoint["metrics"]) != {"val_acc", "val_macro_f1"}:
        raise RuntimeError("evaluate_test=false run contains metrics other than Validation Accuracy/Macro-F1")
    if checkpoint.get("selection") != "best_val_accuracy":
        raise RuntimeError("checkpoint was not selected by Validation Accuracy")
    if bool(cfg.task.evaluate_test):
        raise RuntimeError("Test evaluation unexpectedly enabled")
    row = {
        "dataset": dataset,
        "seed": SEED,
        "variant": variant,
        "status": "completed",
        "commit_sha": _git("rev-parse", "HEAD"),
        "config_sha256": config_sha,
        "config_path": str((run_dir / "resolved_config.yaml").relative_to(ROOT)),
        "protocol": PROTOCOL,
        "evaluate_test": False,
        "cache_path": str(Path(entry["cache_path_abs"]).relative_to(ROOT)),
        "semantic_feature_fingerprint": entry["feature_fingerprint"],
        "semantic_candidate_fingerprint": entry["candidate_fingerprint"],
        "semantic_v7b_fingerprint": entry["v7b_fingerprint"],
        "val_acc": float(metrics["val_acc"]),
        "val_macro_f1": float(metrics["val_macro_f1"]),
        "best_epoch": int(metadata["best_epoch"]),
        "epochs_completed": int(metadata.get("epochs_completed", 0)),
        "training_seconds": float(metadata.get("training_wall_seconds", elapsed)),
        "wall_seconds_including_setup": float(elapsed),
        "model_parameters": int(metadata["model_parameters"]),
        "classifier_parameters": int(metadata["classifier_parameters"]),
        "cuda_peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
        "gpu_name": metadata.get("cuda_device_name"),
        "cuda_free_before_bytes": before["free_bytes"],
        "finite_metrics_and_parameters": bool(
            math.isfinite(float(metrics["val_acc"]))
            and math.isfinite(float(metrics["val_macro_f1"]))
            and all(torch.isfinite(value).all() for value in checkpoint["model_state"].values() if torch.is_floating_point(value))
        ),
        "gradient_diagnostics": json.dumps(metadata.get("gradient_diagnostics"), sort_keys=True),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "log_path": str((run_dir / "train.log").relative_to(ROOT)),
        "error": "",
    }
    if variant == "e3_single_expert_control":
        from src.models.pace_mag_v1 import _matched_single_bottleneck

        width, matched_target, actual_single = _matched_single_bottleneck(
            int(cfg.model.hidden_dim),
            int(cfg.model.expert_bottleneck),
            int(cfg.model.num_experts),
            int(cfg.model.router_hidden_dim),
            int(cfg.model.relation_embedding_dim),
            int(cfg.model.modality_embedding_dim),
        )
        row.update(
            {
                "e3_single_expert_bottleneck": width,
                "e3_matched_dynamic_parameter_target": matched_target,
                "e3_single_network_parameters": actual_single,
                "e3_parameter_difference": actual_single - matched_target,
            }
        )
    else:
        row.update(
            {
                "e3_single_expert_bottleneck": "",
                "e3_matched_dynamic_parameter_target": "",
                "e3_single_network_parameters": "",
                "e3_parameter_difference": "",
            }
        )
    if not row["finite_metrics_and_parameters"]:
        raise RuntimeError("non-finite Validation metric or model parameter found")
    if variant == "e2_full":
        from scripts.analyze_pace_mag_v1 import analyze_checkpoint

        diagnostic_path = DATA_DIR / "diagnostics" / f"{dataset}_seed_{SEED}_e2_full.json"
        row["diagnostic_path"] = str(diagnostic_path.relative_to(ROOT))
        analyze_checkpoint(
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            cache_entry=entry,
            device=device,
            output_path=diagnostic_path,
        )
    else:
        row["diagnostic_path"] = ""
    return row


def _write_rows(rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "seed", "variant", "status", "commit_sha", "config_sha256", "config_path", "protocol", "evaluate_test",
        "cache_path", "semantic_feature_fingerprint", "semantic_candidate_fingerprint", "semantic_v7b_fingerprint",
        "val_acc", "val_macro_f1", "best_epoch", "epochs_completed", "training_seconds", "wall_seconds_including_setup",
        "model_parameters", "classifier_parameters", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "gpu_name",
        "cuda_free_before_bytes", "finite_metrics_and_parameters", "gradient_diagnostics", "checkpoint_path", "log_path",
        "diagnostic_path", "error",
        "e3_single_expert_bottleneck", "e3_matched_dynamic_parameter_target", "e3_single_network_parameters", "e3_parameter_difference",
    ]
    with (DATA_DIR / "pilot_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    resources = [
        {key: row.get(key) for key in (
            "dataset", "seed", "variant", "status", "training_seconds", "wall_seconds_including_setup", "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes", "gpu_name", "cuda_free_before_bytes", "model_parameters", "classifier_parameters", "error",
        )}
        for row in rows
    ]
    with (DATA_DIR / "resource_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(resources[0]) if resources else ["dataset", "seed", "variant", "status"])
        writer.writeheader()
        writer.writerows(resources)


def _write_manifest(device: str, *, phase: str, environment: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    write_json(
        DATA_DIR / "manifest.json",
        {
            "branch": _git("branch", "--show-current"),
            "baseline_commit_sha": BASE_SHA,
            "commit_sha": _git("rev-parse", "HEAD"),
            "phase": phase,
            "protocol": PROTOCOL,
            "datasets": list(DATASETS),
            "variants": list(VARIANTS),
            "seed": SEED,
            "expected_pilot_runs": 15,
            "completed_runs": sum(row["status"] == "completed" for row in rows),
            "failed_runs": sum(row["status"] == "failed" for row in rows),
            "task_evaluate_test": False,
            "test_metrics_computed": False,
            "selected_device": device,
            "environment": environment,
            "outputs_root": str(OUTPUT_ROOT.relative_to(ROOT)),
            "recorded_at_utc": now_utc(),
        },
    )


def run_campaign(device: str, entries: dict[str, dict[str, Any]], environment: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _write_manifest(device, phase="pilot_running", environment=environment, rows=rows)
    for dataset in DATASETS:
        for variant in VARIANTS:
            print(f"\n=== PACE-MAG Pilot {dataset} / seed {SEED} / {variant} ===", flush=True)
            started = time.perf_counter()
            try:
                row = _run_one(dataset, variant, device, entries[dataset])
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "seed": SEED,
                    "variant": variant,
                    "status": "failed",
                    "commit_sha": _git("rev-parse", "HEAD"),
                    "config_sha256": "",
                    "protocol": PROTOCOL,
                    "evaluate_test": False,
                    "semantic_feature_fingerprint": entries[dataset]["feature_fingerprint"],
                    "semantic_candidate_fingerprint": entries[dataset]["candidate_fingerprint"],
                    "semantic_v7b_fingerprint": entries[dataset]["v7b_fingerprint"],
                    "wall_seconds_including_setup": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                }
                print(row["error"], file=sys.stderr, flush=True)
            rows.append(row)
            _write_rows(rows)
            _write_manifest(device, phase="pilot_running", environment=environment, rows=rows)
    _write_manifest(device, phase="pilot_complete" if len(rows) == 15 else "pilot_incomplete", environment=environment, rows=rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "pilot"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="Movies")
    parser.add_argument("--device", default="auto", help="CUDA device, e.g. cuda:1; auto selects the GPU with the most free memory")
    args = parser.parse_args()

    device = _choose_device(args.device)
    environment = _environment(device)
    if _git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}, found {_git('branch', '--show-current')}")
    entries = _cache_entries()
    if set(entries) != set(DATASETS):
        raise RuntimeError(f"V7B candidate manifest is incomplete: {sorted(entries)}")
    verified = {dataset: _verify_cache_entry(dataset, entries[dataset]) for dataset in DATASETS}
    for dataset, item in verified.items():
        verified[dataset] = {**entries[dataset], **item}
    write_json(DATA_DIR / "environment.json", environment)
    if args.phase == "preflight":
        selected = verified[args.dataset]
        check = run_preflight(args.dataset, device, selected)
        write_json(DATA_DIR / "manifest.json", {
            "branch": BRANCH,
            "baseline_commit_sha": BASE_SHA,
            "commit_sha": _git("rev-parse", "HEAD"),
            "phase": "preflight_passed" if check["passed"] else "preflight_failed",
            "protocol": PROTOCOL,
            "task_evaluate_test": False,
            "test_metrics_computed": False,
            "selected_device": device,
            "environment": environment,
            "recorded_at_utc": now_utc(),
        })
        print(json.dumps(check, indent=2))
        return
    if _git("rev-parse", "HEAD") == BASE_SHA:
        raise RuntimeError("Pilot requires an Implementation Freeze Commit before training")
    if (DATA_DIR / "pilot_runs.csv").exists() or OUTPUT_ROOT.joinpath("runs").exists():
        raise FileExistsError("Pilot outputs already exist; refusing to overwrite any prior run")
    rows = run_campaign(device, verified, environment)
    print(f"Pilot finished: {sum(r['status'] == 'completed' for r in rows)}/15 completed")


if __name__ == "__main__":
    main()
