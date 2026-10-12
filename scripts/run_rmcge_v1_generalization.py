#!/usr/bin/env python3
"""Staged, isolated runner for the RM-CGE V1 generalization study."""

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
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_rmcge_v1 as pilot

DESIGN = ROOT / "configs/study/rmcge_v1_generalization.yaml"
RESEARCH = ROOT / "research/rmcge_v1_generalization"
DATA_DIR = RESEARCH / "data"
OUTPUT_ROOT = ROOT / "outputs/rmcge_v1_generalization/runs"
BASE_SHA = "7e4b3dedb9de419ecec4193a1c2e50efd80dc3ab"
BRANCH = "exp/rmcge_v1_generalization_study"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
TREATMENTS = {
    "O0": (0.20, 0.0001),
    "O1": (0.35, 0.0001),
    "O2": (0.20, 0.05),
    "O3": (0.35, 0.05),
}
RELATED_TESTS = (
    "tests/test_rmcge_v1_generalization.py",
    "tests/test_rmcge_v1.py",
    "tests/test_pace_mag_v1.py",
    "tests/test_pace_mag_v1_audit.py",
    "tests/test_psce_mag_v7a.py",
    "tests/test_psce_mag_v7b.py",
    "tests/test_psce_mag_v7b1_corrective.py",
)
SELECTED_PHYSICAL_GPU: int | None = None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_tensor_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _source_tree_hash() -> str:
    digest = hashlib.sha256()
    names = _git("ls-files", "-co", "--exclude-standard").splitlines()
    for relative in sorted(names):
        path = ROOT / relative
        if (
            not path.is_file()
            or relative.startswith("outputs/")
            or relative.startswith("research/rmcge_v1_generalization/data/")
        ):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_design() -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(DESIGN), resolve=True)


def _check_branch(*, require_clean: bool = False) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != BRANCH:
        raise RuntimeError(f"must run on {BRANCH}, currently on {branch}")
    if require_clean and _git("status", "--porcelain"):
        raise RuntimeError("the post-freeze matrix requires a clean worktree")
    return {"branch": branch, "head_sha": head, "worktree_clean": not bool(_git("status", "--porcelain"))}


def _choose_device() -> str:
    global SELECTED_PHYSICAL_GPU
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.splitlines()
    candidates = []
    for line in rows:
        if line.strip():
            index, free = line.split(",", 1)
            candidates.append((int(free.strip()), int(index.strip())))
    if not candidates:
        raise RuntimeError("nvidia-smi did not report a GPU")
    _free_mib, physical = max(candidates)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"failed to isolate physical GPU {physical}")
    SELECTED_PHYSICAL_GPU = physical
    return "cuda:0"


def _environment(device: str) -> dict[str, Any]:
    import numpy as np

    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    try:
        driver_version = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.splitlines()[SELECTED_PHYSICAL_GPU or 0].strip()
    except Exception:
        driver_version = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_physical_gpu": SELECTED_PHYSICAL_GPU,
        "gpu_name": props.name,
        "gpu_compute_capability": list(props.major_minor) if hasattr(props, "major_minor") else [props.major, props.minor],
        "gpu_total_bytes": int(total_bytes),
        "gpu_free_bytes_at_start": int(free_bytes),
        "gpu_allocated_bytes_at_start": int(torch.cuda.memory_allocated(device)),
        "gpu_reserved_bytes_at_start": int(torch.cuda.memory_reserved(device)),
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "determinism": {
            "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "training_randomness": {
            "seed_helper": "random.seed, numpy.random.seed, torch.manual_seed, torch.cuda.manual_seed_all",
            "dropout": "standard torch.nn.Dropout; CUDA RNG is reseeded once per run",
            "activation_checkpointing": True,
            "activation_checkpoint_preserve_rng_state": True,
            "extended_train_val_metrics": "CPU linear evaluation from cached eval embeddings; no extra model forward or RNG draw",
        },
        "nvidia_smi_snapshot": subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=20
        ).stdout,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_input_fingerprints(dataset: str, data, cfg) -> dict[str, Any]:
    split_key = "nc_split_path" if "nc_split_path" in cfg.dataset else "node_split_path"
    split_path = Path(str(cfg.dataset[split_key])).expanduser()
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    train_idx = data.train_idx.detach().cpu().long()
    val_idx = data.val_idx.detach().cpu().long()
    supervised_labels = torch.cat((data.y[data.train_idx], data.y[data.val_idx])).detach().cpu()
    semantic_cache_path = Path(str(cfg.model.semantic_cache_path)).expanduser()
    if not semantic_cache_path.is_absolute():
        semantic_cache_path = ROOT / semantic_cache_path
    return {
        "dataset": dataset,
        "protocol": str(cfg.task.protocol_version),
        "split_config_key": split_key,
        "official_nc_split_path": str(split_path),
        "official_nc_split_file_sha256": _file_sha256(split_path),
        "feature_tensor_sha256": _hash_tensor_state({"x": data.x}),
        "text_feature_tensor_sha256": _hash_tensor_state({"x_t": data.x_t}),
        "visual_feature_tensor_sha256": _hash_tensor_state({"x_i": data.x_i}),
        "physical_edge_index_sha256": _hash_tensor_state({"edge_index": data.edge_index}),
        "semantic_cache_file_sha256": _file_sha256(semantic_cache_path),
        "train_indices_sha256": _hash_tensor_state({"train_idx": train_idx}),
        "validation_indices_sha256": _hash_tensor_state({"val_idx": val_idx}),
        "train_validation_labels_sha256": _hash_tensor_state({"labels_train_val": supervised_labels}),
        "train_node_count": int(train_idx.numel()),
        "validation_node_count": int(val_idx.numel()),
        "test_labels_indexed": False,
    }


def _training_configuration_audit(cache_entries: dict[str, dict[str, Any]], device: str) -> dict[str, Any]:
    study_only_keys = [
        "task.record_extended_validation",
        "task.best_f1_ckpt_path",
        "task.extended_checkpoint_metadata",
        "task.run_metrics_path",
        "task.save_ckpt_path",
        "paths.output_root",
    ]
    controls = {
        "task.protocol_version", "task.training_mode", "task.optimizer", "task.lr",
        "task.weight_decay", "task.epochs", "task.patience", "task.early_stop_min_epoch",
        "task.early_stop_min_delta", "task.grad_clip", "task.eval_every", "task.scheduler",
        "task.evaluate_test", "task.inference_mode", "task.inference_batch_size",
        "task.loss.aux_weight", "model.name", "model.hidden_dim", "model.dropout",
        "model.num_layers", "model.num_experts", "model.expert_bottleneck",
        "model.router_hidden_dim", "model.residual_scale", "model.activation_checkpointing",
        "model.expert_chunk_size", "model.cross_modal_bottleneck",
        "model.relation_embedding_dim", "model.modality_embedding_dim",
        "model.router_correction_init_std", "model.physical_prior", "model.semantic_prior",
        "model.semantic_cache_path", "model.semantic_feature_fingerprint",
        "model.semantic_candidate_fingerprint", "model.semantic_v7b_fingerprint",
        "seed", "device",
    }
    records = {}
    for dataset in DATASETS:
        entry = cache_entries[dataset]
        old = pilot._build_config(dataset, device=device, seed=42)
        current = _build_study_config(dataset, "O0", 42, device, entry)
        old.model.semantic_cache_path = entry["cache_path_abs"]
        old.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
        old.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
        old.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
        old_values = OmegaConf.to_container(old, resolve=True)
        current_values = OmegaConf.to_container(current, resolve=True)
        def get_path(value, dotted):
            for part in dotted.split("."):
                value = value[part]
            return value
        dataset_controls = set(controls)
        dataset_config_match = old_values["dataset"] == current_values["dataset"]
        mismatches = {
            key: {"pilot": get_path(old_values, key), "study_o0": get_path(current_values, key)}
            for key in sorted(dataset_controls)
            if get_path(old_values, key) != get_path(current_values, key)
        }
        if mismatches or not dataset_config_match:
            raise RuntimeError(
                f"O0 configuration differs on frozen controls for {dataset}: "
                f"field_mismatches={mismatches}, dataset_configs_match={dataset_config_match}"
            )
        records[dataset] = {
            "frozen_controls_match": True,
            "compared_controls": sorted(dataset_controls),
            "dataset_configuration_identical": dataset_config_match,
            "expected_study_only_differences": [
                "task.record_extended_validation=true adds CPU-only train/validation metrics and dual-checkpoint bookkeeping",
                "task.best_f1_ckpt_path, task.run_metrics_path, task.save_ckpt_path and paths.output_root point to isolated study outputs",
                "task.extended_checkpoint_metadata records study/config/cache provenance",
            ],
            "non_frozen_helper_differences_documented": study_only_keys,
            "pilot_dropout": float(old.model.dropout),
            "study_o0_dropout": float(current.model.dropout),
            "pilot_weight_decay": float(old.task.weight_decay),
            "study_o0_weight_decay": float(current.task.weight_decay),
            "study_extended_validation": bool(current.task.record_extended_validation),
            "pilot_and_study_evaluate_test": [bool(old.task.evaluate_test), bool(current.task.evaluate_test)],
            "pilot_config_sha256": hashlib.sha256(
                OmegaConf.to_yaml(old, resolve=True, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "study_o0_config_sha256_before_run_output_paths": hashlib.sha256(
                OmegaConf.to_yaml(current, resolve=True, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
    return {"all_datasets_frozen_controls_match": True, "datasets": records}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _build_study_config(dataset: str, treatment: str, seed: int, device: str, entry: dict[str, Any]):
    cfg = pilot._build_config(dataset, device=device, seed=seed)
    dropout, weight_decay = TREATMENTS[treatment]
    cfg.model.dropout = float(dropout)
    cfg.task.weight_decay = float(weight_decay)
    cfg.task.record_extended_validation = True
    cfg.task.evaluate_test = False
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 1.0e-4
    cfg.task.lr = 1.0e-3
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.scheduler = None
    cfg.task.training_mode = "full_graph"
    cfg.task.protocol_version = "unified_full_graph_nc_v1"
    cfg.task.inference_mode = "full"
    cfg.model.semantic_cache_path = entry["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
    cfg.paths.output_root = str(OUTPUT_ROOT)
    return cfg


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def _initialization_audit(cache_entries: dict[str, dict[str, Any]], device: str) -> dict[str, Any]:
    from src.data import load_mag_data
    from src.models import build_model
    from src.utils.seeds import set_seed

    evidence: dict[str, Any] = {}
    dataset_fingerprints: dict[str, Any] = {}
    for dataset in DATASETS:
        entry = cache_entries[dataset]
        data_cfg = pilot._build_config(dataset, device=device, seed=42)
        data_cfg.model.semantic_cache_path = entry["cache_path_abs"]
        data_cfg.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
        data_cfg.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
        data_cfg.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
        data = load_mag_data(data_cfg, "nc", 42)
        info = _data_info(data)
        dataset_fingerprints[dataset] = _dataset_input_fingerprints(
            dataset, data, data_cfg
        )
        for seed in SEEDS:
            hashes = {}
            for treatment in TREATMENTS:
                cfg = _build_study_config(dataset, treatment, seed, device, entry)
                set_seed(seed)
                model = build_model(cfg, info)
                model.load_semantic_cache(
                    entry["cache_path_abs"],
                    expected_dataset=dataset,
                    expected_feature_fingerprint=entry["feature_fingerprint"],
                    expected_candidate_fingerprint=entry["candidate_fingerprint"],
                    expected_v7b_fingerprint=entry["v7b_fingerprint"],
                )
                classifier = torch.nn.Linear(model.out_dim, info["num_classes"])
                hashes[treatment] = {
                    "model_sha256": _hash_tensor_state(model.state_dict()),
                    "classifier_sha256": _hash_tensor_state(classifier.state_dict()),
                    "model_parameter_count": sum(p.numel() for p in model.parameters()),
                    "classifier_parameter_count": sum(p.numel() for p in classifier.parameters()),
                }
                del model, classifier
            unique_model = {item["model_sha256"] for item in hashes.values()}
            unique_head = {item["classifier_sha256"] for item in hashes.values()}
            evidence[f"{dataset}/seed_{seed}"] = {
                "treatments": hashes,
                "model_states_identical": len(unique_model) == 1,
                "classifier_states_identical": len(unique_head) == 1,
            }
            if len(unique_model) != 1 or len(unique_head) != 1:
                raise RuntimeError(f"initialization pairing failed for {dataset}, seed {seed}")
        del data
    return {"dataset_fingerprints": dataset_fingerprints, "pairs": evidence}


def _run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *RELATED_TESTS]
    test_env = os.environ.copy()
    test_env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=test_env)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError("study regression tests failed:\n" + output[-12000:])
    import re

    match = re.search(r"(\d+) passed", output)
    return {
        "command": command,
        "exit_code": result.returncode,
        "passed_tests": int(match.group(1)) if match else None,
        "files": list(RELATED_TESTS),
        "output_tail": output[-12000:],
    }


def preflight() -> dict[str, Any]:
    status = _check_branch()
    if status["head_sha"] != BASE_SHA:
        raise RuntimeError(f"preflight must start from pilot report SHA {BASE_SHA}, got {status['head_sha']}")
    origin_base_sha = _git("rev-parse", "refs/remotes/origin/exp/rmcge_v1_modality_view_experts")
    if origin_base_sha != BASE_SHA:
        raise RuntimeError(f"remote pilot branch changed or is unavailable: expected {BASE_SHA}, got {origin_base_sha}")
    design = _load_design()
    design_treatments = {
        name: (float(values["dropout"]), float(values["weight_decay"]))
        for name, values in design["treatments"].items()
    }
    if design_treatments != TREATMENTS:
        raise RuntimeError(f"study YAML and runner treatment matrix differ: {design_treatments} vs {TREATMENTS}")
    if design["datasets"] != list(DATASETS) or design["seeds"] != list(SEEDS):
        raise RuntimeError("study YAML dataset/seed matrix differs from runner constants")
    device = _choose_device()
    environment = _environment(device)
    cache_entries = {
        dataset: pilot._verify_cache_entry(dataset, pilot._cache_entries()[dataset])
        for dataset in DATASETS
    }
    original = {}
    for dataset in DATASETS:
        checkpoint = ROOT / "outputs/rmcge_v1_pilot/runs" / dataset / "seed_42/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        original[dataset] = {
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "diagnostic_path": str((ROOT / "research/rmcge_v1/data/diagnostics" / f"{dataset}_seed_42.json").relative_to(ROOT)),
        }
    tests = _run_tests()
    initialization = _initialization_audit(cache_entries, device)
    configuration_audit = _training_configuration_audit(cache_entries, device)
    parity_path = DATA_DIR / "o0_step_parity.json"
    if not parity_path.is_file():
        raise FileNotFoundError("the retained short-run O0 parity evidence is missing")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    short_parity_passed = bool(
        parity.get("initialization_identical")
        and parity.get("diagnostic_cpu_metrics_preserve_rng_states")
        and float(parity.get("first_step_parameter_max_abs_difference", float("inf"))) <= 1.0e-5
        and float(parity.get("second_step_parameter_max_abs_difference", float("inf"))) <= 1.0e-5
        and math.isclose(
            float(parity.get("first_step_loss_frozen", float("nan"))),
            float(parity.get("first_step_loss_study", float("inf"))),
            rel_tol=0.0,
            abs_tol=1.0e-5,
        )
        and math.isclose(
            float(parity.get("second_step_loss_frozen", float("nan"))),
            float(parity.get("second_step_loss_study", float("inf"))),
            rel_tol=0.0,
            abs_tol=1.0e-5,
        )
    )
    if not short_parity_passed:
        raise RuntimeError("retained short-run O0 parity evidence does not meet the revised numerical checks")
    checks_path = DATA_DIR / "implementation_checks.json"
    manifest_path = DATA_DIR / "manifest.json"
    previous_checks = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else {}
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prior_replay_record = previous_checks.get("historical_o0_replay", {})
    failures_path = DATA_DIR / "failures.json"
    previous_failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else []
    if tests.get("passed_tests") != 106:
        raise RuntimeError(f"unexpected regression test count: {tests.get('passed_tests')}")
    result = {
        "phase": "implementation_freeze_ready",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": status,
        "origin_baseline_branch_sha": origin_base_sha,
        "parent_pilot_sha": BASE_SHA,
        "environment": environment,
        "cache_audits": {name: item["cache_audit"] for name, item in cache_entries.items()},
        "original_checkpoints": original,
        "initialization_pairing": initialization,
        "training_configuration_audit": configuration_audit,
        "historical_o0_replay": {
            "classification": "historical_full_training_replay_not_exact",
            "formal_gate_blocking": False,
            "original_gate_record": previous_checks.get("o0_reproduction")
            or prior_replay_record.get("original_gate_record"),
            "previous_investigation_record": previous_checks.get("o0_investigation")
            or prior_replay_record.get("previous_investigation_record"),
            "short_run_parity": parity,
            "short_run_parity_passed": short_parity_passed,
            "old_attempts_and_failures_retained": True,
        },
        "test_evidence": tests,
        "evaluate_test": False,
        "test_labels_indexed": False,
        "formal_runs_planned": 36,
        "selected_physical_gpu_index": SELECTED_PHYSICAL_GPU,
    }
    old_statuses = previous_manifest.get("historical_planned_run_statuses") or [
        {key: item.get(key) for key in ("dataset", "treatment", "seed", "status")}
        for item in previous_manifest.get("planned_runs", [])
    ]
    _write_json(checks_path, result)
    manifest = {
        "study": "RM-CGE V1 Generalization & Optimization Study",
        "branch": BRANCH,
        "base_sha": BASE_SHA,
        "implementation_freeze_sha": None,
        "status": "implementation_freeze_ready",
        "planned_runs": [
            {"dataset": dataset, "treatment": treatment, "seed": seed, "status": "planned"}
            for dataset in DATASETS
            for treatment in TREATMENTS
            for seed in SEEDS
        ],
        "environment": environment,
        "training_configuration_audit": configuration_audit,
        "dataset_input_fingerprints": initialization["dataset_fingerprints"],
        "historical_o0_replay": result["historical_o0_replay"],
        "cache_fingerprints": {
            dataset: {k: cache_entries[dataset][k] for k in (
                "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint", "cache_path"
            )}
            for dataset in DATASETS
        },
        "original_checkpoints": original,
        "preflight_tests_passed": tests["passed_tests"],
        "runs": [],
        "failures": previous_failures,
        "historical_planned_run_statuses": old_statuses,
    }
    _write_json(manifest_path, manifest)
    return result


def _logger(run_dir: Path):
    import logging

    logger = logging.getLogger("rmcge_v1_generalization." + run_dir.as_posix().replace("/", "."))
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _audit_checkpoint_reloads(
    dataset: str,
    treatment: str,
    seed: int,
    cfg,
    data,
    entry: dict[str, Any],
    run_dir: Path,
    device: str,
    config_sha: str,
    run_metrics: dict[str, Any],
) -> dict[str, Any]:
    from src.models import build_model
    from src.tasks.inference import infer_all_embeddings
    from src.tasks.nc import (
        _evaluate_indices_from_embeddings,
        _evaluate_split,
        _resolve_nc_eval_labels,
    )

    expected_by_kind = {
        "best_acc": {
            "path": run_dir / "best_acc.pt",
            "selection": "best_val_accuracy",
            "epoch": int(run_metrics["metadata"]["best_epoch"]),
            "metrics": run_metrics["metadata"]["dual_checkpoint_selection"]["best_accuracy"]["metrics"],
        },
        "best_f1": {
            "path": run_dir / "best_f1.pt",
            "selection": "best_val_macro_f1_within_accuracy_stopping_horizon",
            "epoch": int(run_metrics["metadata"]["dual_checkpoint_selection"]["best_macro_f1"]["epoch"]),
            "metrics": run_metrics["metadata"]["dual_checkpoint_selection"]["best_macro_f1"]["metrics"],
        },
    }
    eval_labels = _resolve_nc_eval_labels(data, include_test=False)
    output: dict[str, Any] = {"passed": True, "test_labels_indexed": False, "checkpoints": {}}
    data_info = _data_info(data)
    for kind, expected in expected_by_kind.items():
        path = expected["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing {kind} checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != expected["selection"]:
            raise RuntimeError(f"{kind} selection metadata is incorrect")
        if int(checkpoint.get("epoch", -1)) != expected["epoch"]:
            raise RuntimeError(f"{kind} checkpoint epoch differs from run metadata")
        if checkpoint.get("metrics", {}).get("val_acc") is None or checkpoint.get("metrics", {}).get("val_ce") is None:
            raise RuntimeError(f"{kind} checkpoint is missing validation metrics")
        provenance = checkpoint.get("run_metadata", {}).get("study_provenance", {})
        if (
            provenance.get("config_sha256") != config_sha
            or provenance.get("dataset") != dataset
            or provenance.get("treatment") != treatment
            or int(provenance.get("seed", -1)) != seed
            or provenance.get("test_evaluation") is not False
        ):
            raise RuntimeError(f"{kind} checkpoint provenance mismatch")
        model = build_model(cfg, data_info)
        model.load_semantic_cache(
            entry["cache_path_abs"],
            expected_dataset=dataset,
            expected_feature_fingerprint=entry["feature_fingerprint"],
            expected_candidate_fingerprint=entry["candidate_fingerprint"],
            expected_v7b_fingerprint=entry["v7b_fingerprint"],
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        classifier = torch.nn.Linear(model.out_dim, int(data.num_classes))
        classifier.load_state_dict(checkpoint["head_state"], strict=True)
        model.to(device).eval()
        classifier.to(device).eval()
        with torch.no_grad():
            z = infer_all_embeddings(
                model,
                data,
                torch.device(device),
                True,
                int(cfg.task.inference_batch_size),
                str(cfg.task.inference_mode),
            )
            cls = _evaluate_split(
                classifier,
                z,
                data.y,
                data.val_idx,
                torch.device(device),
                int(cfg.task.inference_batch_size),
                eval_labels,
            )
            ce = _evaluate_indices_from_embeddings(
                classifier, z, data.y, data.val_idx,
                torch.device(device), int(cfg.task.inference_batch_size),
            )["cross_entropy"]
        observed = {"val_acc": cls["acc"], "val_macro_f1": cls["macro_f1"], "val_ce": ce}
        del model, classifier, z
        deltas = {
            key: float(observed[key]) - float(expected["metrics"][key])
            for key in observed
        }
        finite = all(math.isfinite(value) for value in observed.values())
        matched = (
            finite
            and abs(deltas["val_acc"]) <= 1.0e-6
            and abs(deltas["val_macro_f1"]) <= 1.0e-6
            and abs(deltas["val_ce"]) <= 1.0e-5
        )
        output["checkpoints"][kind] = {
            "checkpoint_sha256": _file_sha256(path),
            "selection": expected["selection"],
            "epoch": expected["epoch"],
            "reloaded_metrics": observed,
            "metric_delta_from_saved": deltas,
            "metrics_reproduced_within_tolerance": matched,
            "tolerance": {"val_acc": 1.0e-6, "val_macro_f1": 1.0e-6, "val_ce": 1.0e-5},
        }
        if not matched:
            output["passed"] = False
            raise RuntimeError(f"{kind} checkpoint failed validation replay: {output['checkpoints'][kind]}")
    output["config_sha256"] = config_sha
    output["cache_fingerprints"] = {
        "feature": entry["feature_fingerprint"],
        "candidate": entry["candidate_fingerprint"],
        "v7b": entry["v7b_fingerprint"],
    }
    return output


def _run_one(
    dataset: str,
    treatment: str,
    seed: int,
    entry: dict[str, Any],
    device: str,
    source_sha: str,
    worktree_hash: str,
    run_dir_override: Path | None = None,
) -> dict[str, Any]:
    from src.data import load_mag_data
    from src.tasks.nc import run_nc

    run_dir = run_dir_override or OUTPUT_ROOT / dataset / treatment / f"seed_{seed}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = _build_study_config(dataset, treatment, seed, device, entry)
    cfg.task.save_ckpt_path = str(run_dir / "best_acc.pt")
    cfg.task.best_f1_ckpt_path = str(run_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(run_dir / "run_metrics.json")
    cfg.paths.output_root = str(run_dir / "hydra")
    data_cfg = pilot._build_config(dataset, device=device, seed=42)
    data_cfg.model.semantic_cache_path = entry["cache_path_abs"]
    data_cfg.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
    data_cfg.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
    data_cfg.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
    data = load_mag_data(data_cfg, "nc", 42)
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset}: full multimodal data are unavailable")
    input_fingerprints = _dataset_input_fingerprints(dataset, data, data_cfg)
    config_without_provenance = OmegaConf.to_container(cfg, resolve=True)
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    config_sha = hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()
    provenance = {
        "study": "rmcge_v1_generalization",
        "dataset": dataset,
        "treatment": treatment,
        "seed": seed,
        "config_sha256": config_sha,
        "source_git_sha": source_sha,
        "source_worktree_sha256": worktree_hash,
        "cache_fingerprints": {
            "feature": entry["feature_fingerprint"],
            "candidate": entry["candidate_fingerprint"],
            "v7b": entry["v7b_fingerprint"],
        },
        "data_input_fingerprints": input_fingerprints,
        "resolved_training_config": config_without_provenance,
        "test_evaluation": False,
    }
    cfg.task.extended_checkpoint_metadata = provenance
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True), encoding="utf-8"
    )
    run_environment = _environment(device)
    started = time.perf_counter()
    logger = _logger(run_dir)
    logger.info("Study run %s / %s / seed %d", dataset, treatment, seed)
    logger.info("Config sha=%s | source sha=%s | device=%s", config_sha, source_sha, device)
    metrics = run_nc(cfg, data, torch.device(device), logger)
    elapsed = time.perf_counter() - started
    payload = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
    run = payload["runs"][0]
    metadata = run["metadata"]
    epoch_history = metadata.get("epoch_history", [])
    finite_history = bool(epoch_history) and all(
        math.isfinite(float(epoch_row[key]))
        for epoch_row in epoch_history
        for key in ("train_ce", "train_accuracy_eval", "val_ce", "val_acc", "val_macro_f1")
    )
    if not finite_history:
        raise FloatingPointError(f"non-finite or incomplete per-epoch history in {run_dir}")
    if "test_acc" in run.get("metrics", {}) or "test_macro_f1" in run.get("metrics", {}):
        raise RuntimeError("test metrics were unexpectedly present in a study run")
    checkpoint_audit = _audit_checkpoint_reloads(
        dataset, treatment, seed, cfg, data, entry, run_dir,
        device, config_sha, run,
    )
    if not checkpoint_audit.get("passed"):
        raise RuntimeError(f"checkpoint reload audit failed in {run_dir}")
    output = {
        "dataset": dataset,
        "treatment": treatment,
        "seed": seed,
        "status": "completed",
        "dropout": TREATMENTS[treatment][0],
        "weight_decay": TREATMENTS[treatment][1],
        "val_acc_best_acc": run["metrics"]["val_acc"],
        "val_macro_f1_best_acc": run["metrics"]["val_macro_f1"],
        "best_acc_epoch": metadata["best_epoch"],
        "val_acc_best_f1": metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["val_acc"],
        "val_macro_f1_best_f1": metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["val_macro_f1"],
        "best_f1_epoch": metadata["dual_checkpoint_selection"]["best_macro_f1"]["epoch"],
        "epochs_completed": metadata["epochs_completed"],
        "train_ce_best_acc": metadata["dual_checkpoint_selection"]["best_accuracy"]["metrics"]["train_ce"],
        "train_accuracy_eval_best_acc": metadata["dual_checkpoint_selection"]["best_accuracy"]["metrics"]["train_accuracy_eval"],
        "val_ce_best_acc": metadata["dual_checkpoint_selection"]["best_accuracy"]["metrics"]["val_ce"],
        "train_ce_minus_val_ce_best_acc": metadata["dual_checkpoint_selection"]["best_accuracy"]["metrics"]["train_ce"] - metadata["dual_checkpoint_selection"]["best_accuracy"]["metrics"]["val_ce"],
        "eval_train_acc_minus_val_acc_best_acc": metadata["dual_checkpoint_selection"]["best_accuracy"]["metrics"]["train_accuracy_eval"] - metadata["dual_checkpoint_selection"]["best_accuracy"]["metrics"]["val_acc"],
        "train_ce_best_f1": metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["train_ce"],
        "train_accuracy_eval_best_f1": metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["train_accuracy_eval"],
        "val_ce_best_f1": metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["val_ce"],
        "train_ce_minus_val_ce_best_f1": metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["train_ce"] - metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["val_ce"],
        "eval_train_acc_minus_val_acc_best_f1": metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["train_accuracy_eval"] - metadata["dual_checkpoint_selection"]["best_macro_f1"]["metrics"]["val_acc"],
        "training_seconds": metadata["training_wall_seconds"],
        "wall_seconds_run": elapsed,
        "model_parameters": metadata["model_parameters"],
        "classifier_parameters": metadata["classifier_parameters"],
        "optimizer_update_count": metadata["optimizer_update_count"],
        "initial_parameter_audit": metadata["initial_parameter_audit"],
        "final_parameter_audit": metadata["final_parameter_audit"],
        "initial_model_state_sha256": metadata["initial_parameter_audit"]["model_state_sha256"],
        "initial_classifier_state_sha256": metadata["initial_parameter_audit"]["classifier_state_sha256"],
        "gradient_diagnostics": metadata.get("gradient_diagnostics"),
        "checkpoint_audit": checkpoint_audit,
        "epoch_history_finite": finite_history,
        "data_input_fingerprints": input_fingerprints,
        "environment": run_environment,
        "cuda_peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
        "gpu_name": metadata.get("cuda_device_name"),
        "config_sha256": config_sha,
        "source_git_sha": source_sha,
        "source_worktree_sha256": worktree_hash,
        "feature_fingerprint": entry["feature_fingerprint"],
        "candidate_fingerprint": entry["candidate_fingerprint"],
        "v7b_fingerprint": entry["v7b_fingerprint"],
        "run_dir": str(run_dir.relative_to(ROOT)),
        "evaluate_test": False,
        "test_labels_indexed": False,
    }
    if not all(math.isfinite(float(output[key])) for key in (
        "val_acc_best_acc", "val_macro_f1_best_acc", "val_ce_best_acc", "val_acc_best_f1", "val_macro_f1_best_f1", "val_ce_best_f1"
    )):
        raise FloatingPointError(f"non-finite validation results in {run_dir}")
    return output


def _refresh_run_tables(manifest: dict[str, Any]) -> None:
    completed = manifest.get("runs", [])
    _write_csv(
        DATA_DIR / "runs.csv",
        completed,
        ["dataset", "treatment", "seed", "status", "dropout", "weight_decay", "val_acc_best_acc", "val_macro_f1_best_acc", "val_ce_best_acc", "train_ce_best_acc", "train_accuracy_eval_best_acc", "train_ce_minus_val_ce_best_acc", "eval_train_acc_minus_val_acc_best_acc", "best_acc_epoch", "val_acc_best_f1", "val_macro_f1_best_f1", "val_ce_best_f1", "train_ce_best_f1", "train_accuracy_eval_best_f1", "train_ce_minus_val_ce_best_f1", "eval_train_acc_minus_val_acc_best_f1", "best_f1_epoch", "epochs_completed", "training_seconds", "wall_seconds_run", "model_parameters", "classifier_parameters", "optimizer_update_count", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "gpu_name", "config_sha256", "source_git_sha", "source_worktree_sha256", "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint", "run_dir", "evaluate_test"],
    )
    histories = []
    for row in completed:
        metric_path = ROOT / row["run_dir"] / "run_metrics.json"
        if not metric_path.is_file():
            continue
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        for epoch_row in payload["runs"][0]["metadata"].get("epoch_history", []):
            histories.append({"dataset": row["dataset"], "treatment": row["treatment"], "seed": row["seed"], **epoch_row})
    _write_csv(
        DATA_DIR / "epoch_history.csv",
        histories,
        ["dataset", "treatment", "seed", "epoch", "train_loss", "train_ce", "train_accuracy_eval", "val_ce", "val_acc", "val_macro_f1", "epoch_wall_seconds", "learning_rate"],
    )
    _write_json(
        DATA_DIR / "gradient_and_parameter_audit.json",
        {
            "experimental_unit": "one seed-level training run",
            "runs": [
                {
                    "dataset": row["dataset"],
                    "treatment": row["treatment"],
                    "seed": row["seed"],
                    "source_git_sha": row["source_git_sha"],
                    "config_sha256": row["config_sha256"],
                    "optimizer_update_count": row["optimizer_update_count"],
                    "initial_parameter_audit": row["initial_parameter_audit"],
                    "final_parameter_audit": row["final_parameter_audit"],
                    "gradient_diagnostics": row.get("gradient_diagnostics"),
                }
                for row in completed
            ],
        },
    )


def _record_failure(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    manifest.setdefault("failures", []).append(record)
    _write_json(DATA_DIR / "failures.json", manifest["failures"])
    _write_json(DATA_DIR / "manifest.json", manifest)


def _verify_initialization_pairs(runs: list[dict[str, Any]], dataset: str, seed: int) -> dict[str, Any]:
    selected = [
        row for row in runs
        if row.get("dataset") == dataset and int(row.get("seed", -1)) == seed
        and row.get("status") == "completed"
    ]
    if len(selected) != len(TREATMENTS):
        raise RuntimeError(f"initialization audit needs all four treatments for {dataset}/seed_{seed}")
    model_hashes = {row["initial_model_state_sha256"] for row in selected}
    classifier_hashes = {row["initial_classifier_state_sha256"] for row in selected}
    if len(model_hashes) != 1 or len(classifier_hashes) != 1:
        raise RuntimeError(f"formal initialization pairing failed for {dataset}/seed_{seed}")
    return {
        "dataset": dataset,
        "seed": seed,
        "treatments": sorted(row["treatment"] for row in selected),
        "model_state_sha256": next(iter(model_hashes)),
        "classifier_state_sha256": next(iter(classifier_hashes)),
        "model_states_identical": True,
        "classifier_states_identical": True,
    }


def _dirty_paths() -> list[str]:
    return [line[3:] for line in _git("status", "--porcelain").splitlines() if len(line) >= 4]


def _check_freeze_worktree(*, allow_evidence_changes: bool) -> None:
    disallowed = [
        path for path in _dirty_paths()
        if not (allow_evidence_changes and path.startswith("research/rmcge_v1_generalization/data/"))
    ]
    if disallowed:
        raise RuntimeError(f"worktree has changes outside generated study evidence: {disallowed}")


def run_matrix(batch: str) -> dict[str, Any]:
    global OUTPUT_ROOT
    status = _check_branch(require_clean=False)
    checks = json.loads((DATA_DIR / "implementation_checks.json").read_text(encoding="utf-8"))
    if status["head_sha"] == BASE_SHA:
        raise RuntimeError("commit the Implementation Freeze before starting matrix runs")
    freeze_sha = status["head_sha"]
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if checks.get("phase") != "implementation_freeze_ready":
        raise RuntimeError("implementation preflight is not ready")
    if manifest.get("implementation_freeze_sha") not in (None, freeze_sha):
        raise RuntimeError("manifest freeze SHA differs from current HEAD")
    if batch not in {"seed42", "seeds43_44"}:
        raise ValueError("batch must be seed42 or seeds43_44")
    if batch == "seed42":
        if manifest.get("status") not in {"implementation_freeze_ready", "matrix_seed42_running"}:
            raise RuntimeError(f"seed-42 batch cannot start from manifest phase {manifest.get('status')}")
        _check_freeze_worktree(allow_evidence_changes=manifest.get("status") == "matrix_seed42_running")
        seeds = (42,)
    else:
        if manifest.get("status") != "seed42_batch_passed_review_pending":
            raise RuntimeError("seeds 43/44 require a completed and reviewed 12-run seed-42 batch")
        _check_freeze_worktree(allow_evidence_changes=True)
        seeds = (43, 44)
        first_batch = [
            row for row in manifest.get("runs", [])
            if int(row.get("seed", -1)) == 42 and row.get("status") == "completed"
        ]
        if len(first_batch) != 12:
            raise RuntimeError(f"expected 12 completed seed-42 runs, found {len(first_batch)}")
        for dataset in DATASETS:
            pair = _verify_initialization_pairs(manifest["runs"], dataset, 42)
            if not pair["model_states_identical"] or not pair["classifier_states_identical"]:
                raise RuntimeError(f"seed-42 paired initialization review failed for {dataset}")

    OUTPUT_ROOT = ROOT / "outputs/rmcge_v1_generalization/runs" / f"freeze_{freeze_sha[:10]}"
    manifest["implementation_freeze_sha"] = freeze_sha
    manifest["output_root"] = str(OUTPUT_ROOT.relative_to(ROOT))
    manifest["status"] = "matrix_seed42_running" if batch == "seed42" else "matrix_seeds43_44_running"
    manifest["runs"] = manifest.get("runs", [])
    manifest["failures"] = manifest.get("failures", [])
    _write_json(DATA_DIR / "manifest.json", manifest)
    try:
        device = _choose_device()
    except Exception as exc:
        failure = {
            "phase": "matrix",
            "status": "not_started_gpu_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "implementation_freeze_sha": freeze_sha,
        }
        _record_failure(manifest, failure)
        manifest["status"] = "matrix_not_started_gpu_unavailable"
        _write_json(DATA_DIR / "manifest.json", manifest)
        raise
    cache_entries = {dataset: pilot._verify_cache_entry(dataset, pilot._cache_entries()[dataset]) for dataset in DATASETS}
    source_tree_sha = _source_tree_hash()
    existing = {(row["dataset"], row["treatment"], int(row["seed"])) for row in manifest["runs"] if row.get("status") == "completed"}
    batch_keys = [
        (dataset, treatment, seed)
        for seed in seeds
        for dataset in DATASETS
        for treatment in TREATMENTS
    ]
    pair_audits = manifest.setdefault("formal_initialization_pairing", [])
    for dataset, treatment, seed in batch_keys:
        key = (dataset, treatment, seed)
        if key in existing:
            continue
        run_dir = OUTPUT_ROOT / dataset / treatment / f"seed_{seed}"
        if run_dir.exists():
            failure = {
                "phase": "matrix",
                "dataset": dataset,
                "treatment": treatment,
                "seed": seed,
                "status": "not_started_existing_directory",
                "error": f"run directory already exists; refusing overwrite: {run_dir}",
            }
            _record_failure(manifest, failure)
            raise FileExistsError(failure["error"])
        try:
            record = _run_one(dataset, treatment, seed, cache_entries[dataset], device, freeze_sha, source_tree_sha)
            manifest["runs"].append(record)
            existing.add(key)
            for planned in manifest["planned_runs"]:
                if (planned["dataset"], planned["treatment"], int(planned["seed"])) == key:
                    planned["status"] = "completed"
            same_seed = [
                row for row in manifest["runs"]
                if row.get("dataset") == dataset and int(row.get("seed", -1)) == seed
                and row.get("status") == "completed"
            ]
            if len(same_seed) == len(TREATMENTS):
                pair_audits.append(_verify_initialization_pairs(manifest["runs"], dataset, seed))
            _refresh_run_tables(manifest)
            _write_json(DATA_DIR / "manifest.json", manifest)
            print(f"COMPLETED {dataset}/{treatment}/seed_{seed}", flush=True)
        except Exception as exc:
            failure = {
                "phase": "matrix",
                "dataset": dataset,
                "treatment": treatment,
                "seed": seed,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "run_dir": str(run_dir.relative_to(ROOT)),
            }
            _record_failure(manifest, failure)
            raise
    if batch == "seed42":
        count = sum(1 for row in manifest["runs"] if int(row.get("seed", -1)) == 42 and row.get("status") == "completed")
        if count != 12:
            raise RuntimeError(f"seed-42 batch finished with {count}/12 completed runs")
        manifest["status"] = "seed42_batch_passed_review_pending"
    else:
        count = sum(1 for row in manifest["runs"] if row.get("status") == "completed")
        if count != 36:
            raise RuntimeError(f"matrix finished with {count}/36 completed runs")
        manifest["status"] = "matrix_completed"
        manifest["completed_runs"] = count
    _write_json(DATA_DIR / "manifest.json", manifest)
    return {"batch": batch, "status": manifest["status"], "completed_runs": len(manifest["runs"]), "output_root": str(OUTPUT_ROOT.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "matrix"), required=True)
    parser.add_argument("--batch", choices=("seed42", "seeds43_44"), default="seed42")
    args = parser.parse_args()
    if args.phase == "preflight":
        print(json.dumps(preflight(), indent=2, default=_json_default))
    else:
        print(json.dumps(run_matrix(args.batch), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
