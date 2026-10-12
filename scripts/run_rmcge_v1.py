#!/usr/bin/env python3
"""RM-CGE V1 engineering preflight and frozen three-dataset seed-42 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
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

BRANCH = "exp/rmcge_v1_modality_view_experts"
BASE_SHA = "e32eaeb676b3851cffc0067c549f42a748569e12"
FREEZE_MESSAGE = "RM-CGE V1 implementation freeze"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEED = 42
PROTOCOL = "unified_full_graph_nc_v1"
RESEARCH = ROOT / "research/rmcge_v1"
DATA_DIR = RESEARCH / "data"
OUTPUT_ROOT = ROOT / "outputs/rmcge_v1_pilot"
SUMMARY_PATH = ROOT / "research/psce_mag_v7b/data/candidate_graph_summary.json"
RELATED_TESTS = (
    "tests/test_rmcge_v1.py",
    "tests/test_pace_mag_v1.py",
    "tests/test_pace_mag_v1_audit.py",
    "tests/test_psce_mag_v7a.py",
    "tests/test_psce_mag_v7b.py",
    "tests/test_psce_mag_v7b1_corrective.py",
)
FINGERPRINT_FILES = (
    "configs/model/rmcge_v1.yaml",
    "research/rmcge_v1/DESIGN.md",
    "scripts/analyze_rmcge_v1.py",
    "scripts/run_rmcge_v1.py",
    "src/models/rmcge_v1.py",
    "src/tasks/nc.py",
    "tests/test_rmcge_v1.py",
)
SELECTED_PHYSICAL_GPU: int | None = None


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _resolved_yaml(cfg) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)


def _build_config(dataset: str, *, device: str, seed: int = SEED):
    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create(
        {
            "paths": root_cfg.paths,
            "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
            "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
            "model": OmegaConf.load(ROOT / "configs/model/rmcge_v1.yaml"),
            "logging": root_cfg.logging,
            "seed": int(seed),
            "num_runs": 1,
            "device": str(device),
        }
    )
    cfg.dataset.name = dataset
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
    OmegaConf.resolve(cfg)
    return cfg


def build_config(dataset: str, *, device: str, seed: int = SEED):
    """Public config builder shared with frozen-checkpoint diagnostics."""
    return _build_config(dataset, device=device, seed=seed)


def _candidate_inputs(dataset: str):
    # Read feature matrices and graph topology only, never labels or splits.
    from scripts.run_psce_mag_v7b1_corrective import load_candidate_inputs

    return load_candidate_inputs(dataset)


def _actual_feature_fingerprint(
    dataset: str,
    text: np.ndarray,
    visual: np.ndarray,
    sources: dict[str, Path],
) -> str:
    from src.data.semantic_candidates import _json_fingerprint, _source_descriptor

    identity = {
        "dataset": dataset,
        "num_nodes": int(text.shape[0]),
        "text_dim": int(text.shape[1]),
        "visual_dim": int(visual.shape[1]),
        "text": _source_descriptor(
            sources["text"], np.ascontiguousarray(text, dtype=np.float32)
        ),
        "visual": _source_descriptor(
            sources["visual"], np.ascontiguousarray(visual, dtype=np.float32)
        ),
    }
    return _json_fingerprint(identity)


def _cache_entries() -> dict[str, dict[str, Any]]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {
        item["dataset"]: item
        for item in summary["datasets"]
        if item["dataset"] in DATASETS
    }


def _verify_cache_entry(dataset: str, entry: dict[str, Any]) -> dict[str, Any]:
    from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b

    text, visual, _physical, sources, source_info = _candidate_inputs(dataset)
    observed_feature = _actual_feature_fingerprint(
        dataset, text, visual, sources
    )
    if observed_feature != entry["feature_fingerprint"]:
        raise ValueError(f"{dataset}: current feature arrays differ from the V7B cache")
    cache_path = ROOT / entry["cache_path"]
    cache = load_semantic_candidate_cache_v7b(
        cache_path,
        expected_dataset=dataset,
        expected_feature_fingerprint=observed_feature,
        expected_candidate_fingerprint=entry["candidate_fingerprint"],
        expected_v7b_fingerprint=entry["v7b_fingerprint"],
    )
    if int(cache["metadata"]["num_nodes"]) != int(source_info["num_nodes"]):
        raise ValueError(f"{dataset}: semantic cache node count mismatch")
    flags = cache["source_flags"]
    audit = {
        "dataset": dataset,
        "cache_path": str(cache_path.relative_to(ROOT)),
        "num_nodes": int(source_info["num_nodes"]),
        "physical_edges": int(source_info["num_edges_physical"]),
        "text_dim": int(source_info["text_dim"]),
        "visual_dim": int(source_info["visual_dim"]),
        "unique_union_edges": int(cache["unique_edge_index"].size(1)),
        "directed_union_edges": int(cache["edge_index"].size(1)),
        "text_semantic_undirected_edges": int((flags[:, 0] > 0).sum()),
        "visual_semantic_undirected_edges": int((flags[:, 1] > 0).sum()),
        "both_source_edges": int(((flags[:, 0] > 0) & (flags[:, 1] > 0)).sum()),
        "source_flags_count_0_1_2": {
            source: {
                str(value): int((flags[:, column] == value).sum())
                for value in range(3)
            }
            for column, source in ((0, "text"), (1, "visual"))
        },
        "candidate_phase_label_access": cache["metadata"].get(
            "candidate_phase_label_access"
        ),
        "cache_contains_labels": cache["metadata"].get("cache_contains_labels"),
        "feature_fingerprint": observed_feature,
        "candidate_fingerprint": cache["metadata"]["candidate_fingerprint"],
        "v7b_fingerprint": cache["metadata"]["v7b_fingerprint"],
    }
    return {
        **entry,
        "cache_path_abs": str(cache_path.resolve()),
        "cache_audit": audit,
    }


def _choose_device(requested: str) -> str:
    global SELECTED_PHYSICAL_GPU
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
        candidates = [
            (int(memory.strip()), int(index.strip()))
            for line in output.splitlines()
            if line.strip()
            for index, memory in (line.split(",", maxsplit=1),)
        ]
    except Exception as exc:
        raise RuntimeError(f"could not inspect GPUs through nvidia-smi: {exc}") from exc
    if not candidates:
        raise RuntimeError("nvidia-smi did not report any CUDA devices")
    if requested == "auto":
        physical_index = max(candidates)[1]
    else:
        requested_device = torch.device(requested)
        if requested_device.type != "cuda" or requested_device.index is None:
            raise ValueError(f"requested device must be auto or cuda:<physical-index>: {requested}")
        physical_index = int(requested_device.index)
        if physical_index not in {index for _free, index in candidates}:
            raise ValueError(f"requested physical CUDA device is unavailable: {requested}")

    # The shared seed helper calls manual_seed_all. Restrict visibility first so
    # it cannot initialize contexts on unrelated physical GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"failed to expose only selected physical GPU {physical_index} to PyTorch"
        )
    logical_index = torch.cuda.current_device()
    SELECTED_PHYSICAL_GPU = physical_index
    return str(torch.device("cuda", logical_index))


def _gpu_snapshot(device: str) -> dict[str, Any]:
    device_obj = torch.device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)
    properties = torch.cuda.get_device_properties(device_obj)
    return {
        "device": device,
        "physical_index": SELECTED_PHYSICAL_GPU,
        "name": properties.name,
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device_obj)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device_obj)),
    }


def _run_nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def _environment(device: str) -> dict[str, Any]:
    import dgl
    import faiss
    import torch_geometric

    return {
        "recorded_at_utc": now_utc(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "dgl": dgl.__version__,
        "faiss": faiss.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_physical_gpu_index": SELECTED_PHYSICAL_GPU,
        "selected_gpu": _gpu_snapshot(device),
        "nvidia_smi": _run_nvidia_smi(),
        "other_gpu_processes_stopped": False,
    }


def _load_training_data(dataset: str, device: str, entry: dict[str, Any]):
    from src.data import load_mag_data

    cfg = _build_config(dataset, device=device)
    cfg.model.semantic_cache_path = entry["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
    data = load_mag_data(cfg, "nc", SEED)
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset}: NC loader did not provide both modalities")
    if data.num_nodes != int(entry["cache_audit"]["num_nodes"]):
        raise ValueError(f"{dataset}: NC graph and semantic cache node count disagree")
    return cfg, data


def _prepare_cfg(cfg, run_dir: Path) -> str:
    cfg.task.save_ckpt_path = str(run_dir / "best.pt")
    cfg.task.run_metrics_path = str(run_dir / "run_metrics.json")
    cfg.paths.output_root = str(run_dir / "hydra")
    return hashlib.sha256(_resolved_yaml(cfg).encode("utf-8")).hexdigest()


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"rmcge_v1.{run_dir.name}")
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


def _group_name(name: str) -> str:
    if name.startswith("projectors."):
        return "projectors"
    if ".cross_modal." in name:
        return "cross_modal"
    if ".experts." in name:
        return "experts"
    if any(
        key in name
        for key in (
            ".router.",
            ".relation_embedding.",
            ".modality_embedding.",
            ".router_log_prior",
        )
    ):
        return "router"
    if name.startswith(("fusion.", "skip.", "fusion_norm.")):
        return "fusion"
    return "graph_normalization"


def _run_pytest() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *RELATED_TESTS]
    test_env = os.environ.copy()
    test_env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, env=test_env
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"(\d+) passed", output)
    evidence = {
        "command": command,
        "exit_code": int(result.returncode),
        "passed_tests": int(match.group(1)) if match else None,
        "output_tail": output[-12000:],
        "test_files": list(RELATED_TESTS),
    }
    if result.returncode != 0:
        raise RuntimeError(f"RM-CGE/PACE/V7 regression tests failed:\n{output[-8000:]}")
    return evidence


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in FINGERPRINT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_preflight(
    dataset: str,
    device: str,
    entry: dict[str, Any],
    *,
    test_evidence: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    from src.models import build_model
    from src.tasks.common import build_optimizer
    from src.utils.seeds import set_seed
    from src.utils.summary import count_parameters

    cfg, data = _load_training_data(dataset, device, entry)
    run_dir = OUTPUT_ROOT / "preflight" / dataset
    attempt = 1
    while run_dir.exists() and any(run_dir.iterdir()):
        attempt += 1
        run_dir = OUTPUT_ROOT / "preflight" / f"{dataset}_attempt_{attempt:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_sha = _prepare_cfg(cfg, run_dir)
    (run_dir / "resolved_config.yaml").write_text(
        _resolved_yaml(cfg), encoding="utf-8"
    )
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }
    set_seed(SEED)
    model = build_model(cfg, data_info)
    model.load_semantic_cache(
        entry["cache_path_abs"],
        expected_dataset=dataset,
        expected_feature_fingerprint=entry["feature_fingerprint"],
        expected_candidate_fingerprint=entry["candidate_fingerprint"],
        expected_v7b_fingerprint=entry["v7b_fingerprint"],
    )
    model = model.to(device).train()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = build_optimizer(
        list(model.parameters()) + list(classifier.parameters()), cfg, model=model
    )
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    train_labels = data.y[data.train_idx].to(device)
    before_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    gpu_before = _gpu_snapshot(device)
    if gpu_before["free_bytes"] < 8 * 1024**3:
        raise RuntimeError(
            "largest-graph preflight requires at least 8 GiB free on its selected GPU"
        )
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux_loss, _ = model(x, edge_index)
    loss = torch.nn.functional.cross_entropy(
        classifier(z[train_idx]), train_labels
    ) + aux_loss
    loss.backward()
    gradient_audit = model.capture_gradient_diagnostics()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(classifier.parameters()),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()

    updates: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        group = _group_name(name)
        row = updates.setdefault(
            group,
            {
                "changed_parameter_tensors": 0,
                "changed_elements": 0,
                "max_abs_update": 0.0,
            },
        )
        difference = (parameter.detach() - before_parameters[name]).abs()
        changed = difference > 0
        if bool(changed.any()):
            row["changed_parameter_tensors"] += 1
            row["changed_elements"] += int(changed.sum().item())
            row["max_abs_update"] = max(
                row["max_abs_update"], float(difference.max().cpu())
            )

    checkpoint_path = run_dir / "preflight_checkpoint.pt"
    torch.save(
        {
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "head_state": {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            },
        },
        checkpoint_path,
    )
    model.eval()
    classifier.eval()
    with torch.no_grad():
        expected = model(x, edge_index)[0]
        restored = build_model(cfg, data_info)
        restored.load_semantic_cache(
            entry["cache_path_abs"],
            expected_dataset=dataset,
            expected_feature_fingerprint=entry["feature_fingerprint"],
            expected_candidate_fingerprint=entry["candidate_fingerprint"],
            expected_v7b_fingerprint=entry["v7b_fingerprint"],
        )
        restored = restored.to(device).eval()
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        restored.load_state_dict(payload["model_state"], strict=True)
        restored_classifier = torch.nn.Linear(
            model.out_dim, int(data.num_classes)
        ).to(device)
        restored_classifier.load_state_dict(payload["head_state"], strict=True)
        observed = restored(x, edge_index)[0]
    torch.cuda.synchronize(torch.device(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(torch.device(device)))
    peak_reserved = int(torch.cuda.max_memory_reserved(torch.device(device)))
    gpu_after = _gpu_snapshot(device)
    core_groups = ("projectors", "cross_modal", "experts", "router", "fusion")
    gradients_pass = all(
        group in gradient_audit
        and gradient_audit[group]["gradient_finite"]
        and gradient_audit[group]["gradient_present"]
        == gradient_audit[group]["parameter_count"]
        and gradient_audit[group]["gradient_nonzero"] > 0
        and gradient_audit[group]["all_parameter_tensors_have_finite_nonzero_gradients"]
        and gradient_audit[group]["gradient_l2_norm"] > 0
        for group in core_groups
    )
    updates_pass = all(
        group in updates and updates[group]["changed_parameter_tensors"] > 0
        for group in core_groups
    )
    summary = {
        "recorded_at_utc": now_utc(),
        "dataset": dataset,
        "seed": SEED,
        "protocol": PROTOCOL,
        "source_revision_sha": _git("rev-parse", "HEAD"),
        "implementation_source_fingerprint_sha256": _source_fingerprint(),
        "config_sha256": config_sha,
        "config": {
            "hidden_dim": int(cfg.model.hidden_dim),
            "num_layers": int(cfg.model.num_layers),
            "num_experts": int(cfg.model.num_experts),
            "expert_bottleneck": int(cfg.model.expert_bottleneck),
            "dropout": float(cfg.model.dropout),
            "evaluate_test": bool(cfg.task.evaluate_test),
        },
        "parameters_including_classifier": count_parameters(model)
        + count_parameters(classifier),
        "device": device,
        "environment": environment,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "full_graph_forward_finite": bool(torch.isfinite(z).all()),
        "full_graph_backward_finite": bool(
            torch.isfinite(loss)
            and all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        ),
        "adamw_step_completed": True,
        "loss_on_train_indices_only": float(loss.detach().cpu()),
        "aux_loss_exact_zero": float(aux_loss.detach().cpu()) == 0.0,
        "gradient_audit": gradient_audit,
        "parameter_update_audit": updates,
        "core_gradient_check_passed": gradients_pass,
        "core_update_check_passed": updates_pass,
        "checkpoint_reload_max_abs_difference": float(
            (expected - observed).abs().max().cpu()
        ),
        "checkpoint_reload_consistent": bool(
            torch.allclose(expected, observed, atol=1.0e-6, rtol=1.0e-6)
        ),
        "evaluate_test": bool(cfg.task.evaluate_test),
        "test_metrics_computed": False,
        "test_labels_indexed": False,
        "cache_audit": entry["cache_audit"],
        "regression_tests": test_evidence,
        "preflight_checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }
    summary["passed"] = bool(
        summary["full_graph_forward_finite"]
        and summary["full_graph_backward_finite"]
        and summary["adamw_step_completed"]
        and summary["aux_loss_exact_zero"]
        and gradients_pass
        and updates_pass
        and summary["checkpoint_reload_consistent"]
        and not summary["evaluate_test"]
        and not summary["test_metrics_computed"]
        and not summary["test_labels_indexed"]
    )
    if not summary["passed"]:
        raise RuntimeError(f"RM-CGE GPU engineering preflight failed: {summary}")
    write_json(DATA_DIR / "implementation_checks.json", summary)
    write_json(
        DATA_DIR / "gradient_audit.json",
        {
            "dataset": dataset,
            "seed": SEED,
            "gradient_groups": gradient_audit,
            "parameter_updates": updates,
            "passed": summary["core_gradient_check_passed"]
            and summary["core_update_check_passed"],
        },
    )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            prepared = {}
            for key, value in row.items():
                prepared[key] = (
                    json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                )
            writer.writerow(prepared)


def _write_campaign_files(
    rows: list[dict[str, Any]],
    histories: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    resources: list[dict[str, Any]],
) -> None:
    _write_csv(
        DATA_DIR / "pilot_runs.csv",
        rows,
        [
            "dataset", "seed", "status", "commit_sha", "config_sha256", "protocol",
            "evaluate_test", "cache_path", "semantic_feature_fingerprint",
            "semantic_candidate_fingerprint", "semantic_v7b_fingerprint",
            "val_acc", "val_macro_f1", "best_epoch", "epochs_completed",
            "training_seconds", "wall_seconds_including_setup", "model_parameters",
            "classifier_parameters", "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes", "gpu_name", "finite_metrics_and_parameters",
            "checkpoint_path", "log_path", "diagnostic_path", "error",
        ],
    )
    _write_csv(
        DATA_DIR / "training_history.csv",
        histories,
        [
            "dataset", "seed", "epoch", "train_loss", "val_acc",
            "val_macro_f1", "epoch_wall_seconds", "learning_rate",
        ],
    )
    _write_csv(
        DATA_DIR / "model_diagnostics.csv",
        diagnostics,
        [
            "dataset", "seed", "layer", "modality", "relation", "support",
            "valid_nodes", "valid_node_fraction", "mean_neighbors_when_valid",
            "same_context_rms", "cross_context_rms",
            "cross_to_same_context_rms", "mean_joint_relation_mass",
            "mean_joint_expert_probabilities", "expert_output_rms",
            "expert_pairwise_cosine", "relation_marginal", "expert_use",
            "route_entropy_nats", "delta_rms", "relative_update_rms",
        ],
    )
    _write_csv(
        DATA_DIR / "resource_profile.csv",
        resources,
        [
            "dataset", "seed", "status", "training_seconds",
            "wall_seconds_including_setup", "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes", "cuda_free_before_bytes", "gpu_name",
            "error",
        ],
    )


def _write_manifest(
    *,
    phase: str,
    device: str,
    environment: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    freeze_sha = (
        _git("rev-parse", "HEAD") if not phase.startswith("preflight") else None
    )
    write_json(
        DATA_DIR / "manifest.json",
        {
            "branch": BRANCH,
            "baseline_commit_sha": BASE_SHA,
            "implementation_freeze_commit_sha": freeze_sha,
            "commit_sha": _git("rev-parse", "HEAD"),
            "phase": phase,
            "protocol": PROTOCOL,
            "datasets": list(DATASETS),
            "seed": SEED,
            "num_formal_trainings": 3,
            "epochs_max": 300,
            "optimizer": "AdamW",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "patience": 30,
            "early_stop_min_epoch": 30,
            "grad_clip": 1.0,
            "selection": "best_val_accuracy",
            "evaluate_test": False,
            "test_metrics_computed": False,
            "selected_device": device,
            "environment": environment,
            "completed_runs": [
                {
                    "dataset": row.get("dataset"),
                    "status": row.get("status"),
                    "val_acc": row.get("val_acc"),
                    "val_macro_f1": row.get("val_macro_f1"),
                    "best_epoch": row.get("best_epoch"),
                }
                for row in rows
            ],
            "recorded_at_utc": now_utc(),
        },
    )


def _run_one(
    dataset: str, device: str, entry: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.tasks.nc import run_nc

    run_dir = OUTPUT_ROOT / "runs" / dataset / f"seed_{SEED}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Pilot output: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _build_config(dataset, device=device)
    cfg.model.semantic_cache_path = entry["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
    cfg_sha = _prepare_cfg(cfg, run_dir)
    (run_dir / "resolved_config.yaml").write_text(
        _resolved_yaml(cfg), encoding="utf-8"
    )
    logger = _setup_logger(run_dir)
    from src.data import load_mag_data

    data = load_mag_data(cfg, "nc", SEED)
    if data.num_nodes != int(entry["cache_audit"]["num_nodes"]):
        raise ValueError(f"{dataset}: NC data and semantic cache node counts differ")
    before = _gpu_snapshot(device)
    if before["free_bytes"] < 4 * 1024**3:
        raise RuntimeError(
            f"{dataset}: less than 4 GiB free on selected GPU; no process was stopped"
        )
    started = time.perf_counter()
    run_nc(cfg, data, torch.device(device), logger)
    elapsed = time.perf_counter() - started
    metrics_payload = json.loads(
        (run_dir / "run_metrics.json").read_text(encoding="utf-8")
    )
    if len(metrics_payload.get("runs", [])) != 1:
        raise RuntimeError("NC runner did not produce exactly one run")
    run_row = metrics_payload["runs"][0]
    metrics = run_row["metrics"]
    metadata = run_row["metadata"]
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("NC runner did not save a best-Validation checkpoint")
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if set(metrics) != {"val_acc", "val_macro_f1"}:
        raise RuntimeError("evaluate_test=false run produced non-Validation metrics")
    if set(checkpoint_payload.get("metrics", {})) != {"val_acc", "val_macro_f1"}:
        raise RuntimeError("checkpoint contains metrics outside Validation")
    if checkpoint_payload.get("selection") != "best_val_accuracy":
        raise RuntimeError("checkpoint was not selected by Validation Accuracy")
    if bool(cfg.task.evaluate_test):
        raise RuntimeError("Test evaluation unexpectedly enabled")
    epoch_history = metadata.get("epoch_history")
    if not isinstance(epoch_history, list) or not epoch_history:
        raise RuntimeError("NC runner did not preserve exact per-epoch history")
    if len(epoch_history) != int(metadata.get("epochs_completed", -1)):
        raise RuntimeError("epoch history and completed-epoch count disagree")
    if not all(
        math.isfinite(float(metrics[key])) for key in ("val_acc", "val_macro_f1")
    ):
        raise FloatingPointError("non-finite Validation metric")
    if not all(
        torch.isfinite(value).all()
        for value in checkpoint_payload["model_state"].values()
        if torch.is_floating_point(value)
    ):
        raise FloatingPointError("selected checkpoint contains non-finite parameters")

    history_rows = [
        {"dataset": dataset, "seed": SEED, **item} for item in epoch_history
    ]
    row = {
        "dataset": dataset,
        "seed": SEED,
        "status": "completed",
        "commit_sha": _git("rev-parse", "HEAD"),
        "config_sha256": cfg_sha,
        "protocol": PROTOCOL,
        "evaluate_test": False,
        "cache_path": str(Path(entry["cache_path_abs"]).relative_to(ROOT)),
        "semantic_feature_fingerprint": entry["feature_fingerprint"],
        "semantic_candidate_fingerprint": entry["candidate_fingerprint"],
        "semantic_v7b_fingerprint": entry["v7b_fingerprint"],
        "val_acc": float(metrics["val_acc"]),
        "val_macro_f1": float(metrics["val_macro_f1"]),
        "best_epoch": int(metadata["best_epoch"]),
        "epochs_completed": int(metadata["epochs_completed"]),
        "training_seconds": float(metadata["training_wall_seconds"]),
        "wall_seconds_including_setup": float(elapsed),
        "model_parameters": int(metadata["model_parameters"]),
        "classifier_parameters": int(metadata["classifier_parameters"]),
        "cuda_peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
        "gpu_name": metadata.get("cuda_device_name"),
        "cuda_free_before_bytes": int(before["free_bytes"]),
        "finite_metrics_and_parameters": True,
        "gradient_diagnostics": metadata.get("gradient_diagnostics"),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "log_path": str((run_dir / "train.log").relative_to(ROOT)),
        "diagnostic_path": str(
            (
                DATA_DIR
                / "diagnostics"
                / f"{dataset}_seed_{SEED}.json"
            ).relative_to(ROOT)
        ),
        "error": "",
    }
    return row, history_rows


def _write_final_summary(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# RM-CGE V1 Pilot report",
        "",
        "The machine-readable run table is data/pilot_runs.csv; exact epoch metrics are in data/training_history.csv.",
        "",
        "| Dataset | Val Accuracy | Val Macro-F1 | Best Epoch |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") == "completed":
            lines.append(
                f"| {row['dataset']} | {row['val_acc']:.6f} | "
                f"{row['val_macro_f1']:.6f} | {row['best_epoch']} |"
            )
    lines.extend(
        [
            "",
            "Frozen relation/router, context-scale, residual, and branch-off diagnostics are in data/model_diagnostics.csv and data/diagnostics/.",
            "",
            "Interpretation and evidence limits are completed after inspecting all three selected checkpoints.",
        ]
    )
    (RESEARCH / "PILOT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_campaign(
    device: str,
    entries: dict[str, dict[str, Any]],
    environment: dict[str, Any],
) -> list[dict[str, Any]]:
    from scripts.analyze_rmcge_v1 import analyze_checkpoint

    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (DATA_DIR / "pilot_runs.csv").exists() or (OUTPUT_ROOT / "runs").exists():
        raise FileExistsError("RM-CGE Pilot outputs exist; refusing to overwrite them")
    _write_manifest(
        phase="pilot_running", device=device, environment=environment, rows=rows
    )
    for dataset in DATASETS:
        started = time.perf_counter()
        try:
            row, history_rows = _run_one(dataset, device, entries[dataset])
            diagnostic = analyze_checkpoint(
                dataset=dataset,
                checkpoint_path=row["checkpoint_path"],
                cache_entry=entries[dataset],
                device=device,
                output_path=DATA_DIR / "diagnostics" / f"{dataset}_seed_{SEED}.json",
            )
            row["diagnostic_path"] = str(
                (
                    DATA_DIR
                    / "diagnostics"
                    / f"{dataset}_seed_{SEED}.json"
                ).relative_to(ROOT)
            )
            diagnostics.extend(
                {
                    "dataset": dataset,
                    "seed": SEED,
                    **item,
                }
                for item in diagnostic["relation_diagnostics"]
            )
            histories.extend(history_rows)
            rows.append(row)
            resources.append(
                {
                    "dataset": dataset,
                    "seed": SEED,
                    "status": row["status"],
                    "training_seconds": row["training_seconds"],
                    "wall_seconds_including_setup": row[
                        "wall_seconds_including_setup"
                    ],
                    "cuda_peak_allocated_bytes": row[
                        "cuda_peak_allocated_bytes"
                    ],
                    "cuda_peak_reserved_bytes": row[
                        "cuda_peak_reserved_bytes"
                    ],
                    "cuda_free_before_bytes": row["cuda_free_before_bytes"],
                    "gpu_name": row["gpu_name"],
                    "error": "",
                }
            )
            _write_campaign_files(rows, histories, diagnostics, resources)
            _write_manifest(
                phase="pilot_running",
                device=device,
                environment=environment,
                rows=rows,
            )
            print(
                f"Completed {dataset}: Val Acc={row['val_acc']:.6f}, "
                f"Val Macro-F1={row['val_macro_f1']:.6f}, "
                f"best_epoch={row['best_epoch']}",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            failed = {
                "dataset": dataset,
                "seed": SEED,
                "status": "failed",
                "commit_sha": _git("rev-parse", "HEAD"),
                "protocol": PROTOCOL,
                "evaluate_test": False,
                "wall_seconds_including_setup": time.perf_counter() - started,
                "error": error,
            }
            rows.append(failed)
            resources.append(
                {
                    "dataset": dataset,
                    "seed": SEED,
                    "status": "failed",
                    "wall_seconds_including_setup": time.perf_counter() - started,
                    "error": error,
                }
            )
            _write_campaign_files(rows, histories, diagnostics, resources)
            _write_manifest(
                phase="pilot_failed",
                device=device,
                environment=environment,
                rows=rows,
            )
            raise
    _write_manifest(
        phase="pilot_complete",
        device=device,
        environment=environment,
        rows=rows,
    )
    _write_final_summary(rows)
    return rows


def _assert_provenance(phase: str) -> str | None:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}, found {branch}")
    if _git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError(f"branch ancestry does not include required base {BASE_SHA}")
    if phase == "preflight":
        if head != BASE_SHA:
            raise RuntimeError(
                f"implementation preflight must start from exact base SHA; found {head}"
            )
        return None
    freeze_sha = (
        _git("log", "--format=%H", "-1", f"--grep=^{FREEZE_MESSAGE}$") or None
    )
    if freeze_sha is None or head != freeze_sha:
        raise RuntimeError(
            f"Pilot requires HEAD at the implementation freeze commit; "
            f"head={head}, freeze={freeze_sha}"
        )
    dirty = _git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "Pilot requires a clean worktree at the implementation freeze commit"
        )
    return freeze_sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "pilot"), required=True)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="ele-fashion",
        help="largest graph is the default engineering preflight target",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="CUDA device such as cuda:1; auto chooses the device with most free memory",
    )
    args = parser.parse_args()
    phase = args.phase
    freeze_sha = _assert_provenance(phase)
    device = _choose_device(args.device)
    environment = _environment(device)
    entries = _cache_entries()
    if set(entries) != set(DATASETS):
        raise RuntimeError(f"V7B manifest is incomplete: {sorted(entries)}")
    verified = {
        dataset: _verify_cache_entry(dataset, entries[dataset])
        for dataset in DATASETS
    }
    if phase == "preflight":
        test_evidence = _run_pytest()
        result = run_preflight(
            args.dataset,
            device,
            verified[args.dataset],
            test_evidence=test_evidence,
            environment=environment,
        )
        write_json(
            DATA_DIR / "cache_audit.json",
            {
                "recorded_at_utc": now_utc(),
                "datasets": [
                    verified[dataset]["cache_audit"] for dataset in DATASETS
                ],
            },
        )
        _write_manifest(
            phase="preflight_passed" if result["passed"] else "preflight_failed",
            device=device,
            environment=environment,
            rows=[],
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return

    if freeze_sha is None:
        raise RuntimeError("Implementation Freeze commit was not found")
    if (DATA_DIR / "pilot_runs.csv").exists() or (OUTPUT_ROOT / "runs").exists():
        raise FileExistsError("Pilot outputs already exist; refusing a second campaign")
    write_json(DATA_DIR / "environment.json", environment)
    rows = run_campaign(device, verified, environment)
    print(f"Pilot finished: {len(rows)}/{len(DATASETS)} complete trainings")


if __name__ == "__main__":
    main()
