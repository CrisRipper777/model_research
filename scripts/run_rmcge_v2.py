#!/usr/bin/env python3
"""Staged RM-CGE V2 preflight, training campaign, and checkpoint audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRANCH = "exp/rmcge_v2_edge_conditioned_experts"
BASE_SHA = "8dec795e4b05b208b73e57e3d87ebfc215a1a75a"
FREEZE_SUBJECT = "Freeze RM-CGE V2 implementation"
PROTOCOL = "unified_full_graph_nc_v1"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("V0", "V1", "V2")
SEEDS = (42, 43, 44)
RESEARCH = ROOT / "research/rmcge_v2"
DATA_DIR = RESEARCH / "data"
OUTPUT = ROOT / "outputs/rmcge_v2"
SUMMARY = ROOT / "research/psce_mag_v7b/data/candidate_graph_summary.json"
ARTIFACT_CSV_FIELDS = {
    "runs.csv": ["dataset", "variant", "seed", "status", "best_epoch", "val_acc", "val_macro_f1", "wall_seconds", "checkpoint_path", "run_metrics_path", "diagnostics_path", "config_sha256", "test_metrics_present", "error"],
    "epoch_history.csv": ["dataset", "variant", "seed", "epoch", "train_loss", "train_objective", "train_ce", "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce", "val_acc", "val_macro_f1", "epoch_wall_seconds", "learning_rate"],
    "paired_comparisons.csv": ["dataset", "seed", "metric", "contrast", "difference", "left_value", "right_value"],
    "edge_weight_diagnostics.csv": ["dataset", "variant", "seed", "modality", "mean", "std", "p10", "p50", "p90", "mean_abs_from_one", "fraction_abs_gt_0_1", "fraction_abs_gt_0_2", "mean_by_target_degree_bin", "mean_abs_reverse_difference", "text_visual_correlation", "text_visual_correlation_defined", "text_visual_mean_abs_difference"],
    "message_scale_diagnostics.csv": ["dataset", "variant", "seed", "modality", "u1_rms", "delta_rms", "delta_hat_rms", "delta_weighted_rms_by_expert", "cross_rms", "cross_hat_rms", "cross_weighted_rms_by_expert", "knn_rms", "knn_hat_rms", "knn_weighted_rms_by_expert", "floor_hit_dimensions", "physical_profile_rms_by_expert", "combined_profile_rms_by_expert", "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine", "weighted_graph_correction_rms", "intrinsic_rms", "router_correction_rms", "router_entropy", "dense_probability_mean", "expert_selection_share", "base_expert_selection_share", "router_top2_flip_fraction", "expert_coefficients"],
    "expert_router_diagnostics.csv": ["dataset", "variant", "seed", "modality", "expert_selection_share", "base_expert_selection_share", "dense_probability_mean", "router_entropy", "router_correction_rms", "router_top2_flip_fraction", "expert_coefficients", "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine"],
    "frozen_interventions.csv": ["dataset", "variant", "seed", "intervention", "val_acc", "val_macro_f1", "val_acc_delta", "val_macro_f1_delta", "logit_delta_rms", "prediction_flip_fraction"],
    "resource_profile.csv": ["dataset", "variant", "seed", "wall_seconds", "epochs_completed", "mean_epoch_wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "cuda_device_name"],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, fallback: Any):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else fallback


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(value) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_hash(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cache_entries() -> dict[str, dict[str, Any]]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {item["dataset"]: item for item in summary["datasets"] if item["dataset"] in DATASETS}


def verify_cache_entries() -> dict[str, dict[str, Any]]:
    # Reuse the V1 study's read-only provenance verifier, which reads features
    # and physical topology without opening labels or split files.
    from scripts import run_rmcge_v1 as v1

    result = {}
    entries = cache_entries()
    for dataset in DATASETS:
        result[dataset] = v1._verify_cache_entry(dataset, entries[dataset])
    return result


def build_config(dataset: str, variant: str, seed: int, device: str, cache: dict[str, Any]):
    from omegaconf import OmegaConf

    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create(
        {
            "paths": root_cfg.paths,
            "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
            "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
            "model": OmegaConf.load(ROOT / "configs/model/rmcge_v2.yaml"),
            "logging": root_cfg.logging,
            "seed": int(seed),
            "num_runs": 1,
            "device": str(device),
        }
    )
    cfg.dataset.name = dataset
    cfg.model.variant = variant
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
    cfg.task.best_f1_ckpt_path = None
    cfg.task.extended_checkpoint_metadata = {
        "study": "RM-CGE V2",
        "dataset": dataset,
        "variant": variant,
        "seed": int(seed),
        "official_train_val_split": True,
        "test_evaluation": False,
    }
    cfg.paths.output_root = str(OUTPUT / "hydra")
    OmegaConf.resolve(cfg)
    return cfg


def data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def initialize_model(cfg, info, cache, seed: int):
    import torch
    from src.models import build_model
    from src.utils.seeds import set_seed

    set_seed(seed)
    model = build_model(cfg, info)
    model.load_semantic_cache(
        cache["cache_path_abs"],
        expected_dataset=cfg.dataset.name,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    classifier = torch.nn.Linear(model.out_dim, info["num_classes"])
    return model, classifier


def training_config_hash(cfg) -> str:
    from omegaconf import OmegaConf

    text = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def initialization_audit(caches: dict[str, dict[str, Any]], device: str = "cpu") -> dict[str, Any]:
    import torch
    from src.data import load_mag_data
    from src.utils.seeds import set_seed

    result: dict[str, Any] = {"datasets": {}, "pairs": {}}
    for dataset in DATASETS:
        cfg0 = build_config(dataset, "V0", 42, device, caches[dataset])
        data = load_mag_data(cfg0, "nc", 42)
        # Deliberately fingerprint only features, physical topology and the
        # official Train/Validation split and labels. Test labels are untouched.
        fingerprints = {
            "features_sha256": tensor_fingerprint(data.x),
            "physical_edge_index_sha256": tensor_fingerprint(data.edge_index),
            "train_index_sha256": tensor_fingerprint(data.train_idx),
            "validation_index_sha256": tensor_fingerprint(data.val_idx),
            "train_labels_sha256": tensor_fingerprint(data.y[data.train_idx]),
            "validation_labels_sha256": tensor_fingerprint(data.y[data.val_idx]),
            "semantic_cache_sha256": sha256_file(Path(caches[dataset]["cache_path_abs"])),
            "feature_fingerprint": caches[dataset]["feature_fingerprint"],
            "candidate_fingerprint": caches[dataset]["candidate_fingerprint"],
            "v7b_fingerprint": caches[dataset]["v7b_fingerprint"],
            "test_label_values_indexed": False,
        }
        result["datasets"][dataset] = fingerprints
        info = data_info(data)
        for seed in SEEDS:
            models = {}
            heads = {}
            hashes = {}
            for variant in VARIANTS:
                cfg = build_config(dataset, variant, seed, device, caches[dataset])
                model, head = initialize_model(cfg, info, caches[dataset], seed)
                models[variant] = model
                heads[variant] = head
                hashes[variant] = {
                    "model_state_sha256": module_hash(model),
                    "classifier_sha256": module_hash(head),
                    "model_parameter_count": sum(p.numel() for p in model.parameters()),
                    "classifier_parameter_count": sum(p.numel() for p in head.parameters()),
                    "parameter_shapes": {name: list(value.shape) for name, value in model.named_parameters()},
                }
            common_names = set(dict(models["V0"].named_parameters()))
            trunk_match = True
            for variant in ("V1", "V2"):
                current = dict(models[variant].named_parameters())
                for name in common_names:
                    if not torch.equal(dict(models["V0"].named_parameters())[name], current[name]):
                        trunk_match = False
            v1v2_names = set(dict(models["V1"].named_parameters())) == set(dict(models["V2"].named_parameters()))
            v1v2_values = v1v2_names and all(
                torch.equal(dict(models["V1"].named_parameters())[name], dict(models["V2"].named_parameters())[name])
                for name in dict(models["V1"].named_parameters())
            )
            head_match = len({hashes[v]["classifier_sha256"] for v in VARIANTS}) == 1
            result["pairs"][f"{dataset}/seed_{seed}"] = {
                "variants": hashes,
                "shared_trunk_initialization_matches": trunk_match,
                "v1_v2_parameter_names_shapes_and_initial_values_match": v1v2_values,
                "classifier_initialization_matches": head_match,
            }
            if not (trunk_match and v1v2_values and head_match):
                raise RuntimeError(f"initialization pairing failure: {dataset}, seed={seed}")
            del models, heads
        del data
    return result


def run_full_tests() -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-m", "pytest", "-q"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError("full repository pytest failed:\n" + output[-16000:])
    import re

    match = re.search(r"(\d+) passed", output)
    skipped = re.search(r"(\d+) skipped", output)
    return {
        "command": command,
        "return_code": result.returncode,
        "passed": int(match.group(1)) if match else None,
        "skipped": int(skipped.group(1)) if skipped else 0,
        "output_tail": output[-6000:],
    }


def environment_record(physical_gpu: int) -> dict[str, Any]:
    import numpy as np
    import torch

    import dgl
    import faiss
    import torch_geometric

    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    gpus = []
    for line in query.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 6:
            gpus.append({"index": int(fields[0]), "name": fields[1], "driver": fields[2], "free_mib": fields[3], "total_mib": fields[4], "utilization_percent": fields[5]})
    return {
        "captured_at_utc": now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_geometric": torch_geometric.__version__,
        "dgl": dgl.__version__,
        "faiss": faiss.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_count_visible": torch.cuda.device_count(),
        "selected_physical_gpu": physical_gpu,
        "gpus": gpus,
        "nvidia_smi": smi.stdout[-12000:] if smi.returncode == 0 else smi.stderr,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def gpu_ele_fashion_smoke(cache: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.common import build_optimizer
    from src.utils.seeds import set_seed

    cfg = build_config("ele-fashion", "V2", 42, device, cache)
    data = load_mag_data(cfg, "nc", 42)
    info = data_info(data)
    set_seed(20261013)
    model = build_model(cfg, info)
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset="ele-fashion",
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device).train()
    head = torch.nn.Linear(model.out_dim, data.num_classes).to(device)
    optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), cfg, model=model)
    x, edge = data.x.to(device), data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    labels = data.y[data.train_idx].to(device)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux, _ = model(x, edge)
    loss = torch.nn.functional.cross_entropy(head(z[train_idx]), labels) + aux
    if not torch.isfinite(loss):
        raise RuntimeError("ele-fashion GPU smoke produced non-finite loss")
    loss_first_value = float(loss.detach().cpu())
    aux_first_value = float(aux.detach().cpu())
    loss.backward()
    first_step_gradients = model.capture_gradient_diagnostics()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
    optimizer.step()
    del z, loss, aux
    optimizer.zero_grad(set_to_none=True)
    z2, _, _, aux2, _ = model(x, edge)
    loss2 = torch.nn.functional.cross_entropy(head(z2[train_idx]), labels) + aux2
    if not torch.isfinite(loss2):
        raise RuntimeError("ele-fashion second GPU smoke step produced non-finite loss")
    loss2.backward()
    second_step_gradients = model.capture_gradient_diagnostics()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(torch.device(device))
    params = dict(model.named_parameters())
    required = (
        "edge_scorer.2.weight",
        "cross_projections.visual_to_text.0.weight",
        "cross_projections.visual_to_text.2.weight",
        "cross_projections.text_to_visual.0.weight",
        "cross_projections.text_to_visual.2.weight",
        "node_router.3.weight",
        "experts.0.linear1.weight",
        "fusion_linear1.weight",
    )
    changed = {name: not torch.equal(before[name], params[name]) for name in required}
    if not all(changed.values()):
        raise RuntimeError(f"ele-fashion GPU smoke had missing parameter updates: {changed}")
    if not first_step_gradients["edge_scorer.2.weight"]["gradient_nonzero"]:
        raise RuntimeError("ele-fashion Edge Scorer output layer did not receive a first-step gradient")
    if not second_step_gradients["edge_scorer.0.weight"]["gradient_nonzero"]:
        raise RuntimeError("ele-fashion Edge Scorer hidden layer did not receive a later-step gradient")
    if not second_step_gradients["delta_coefficients"]["gradient_nonzero"]:
        raise RuntimeError("ele-fashion Delta expert coefficients did not receive a later-step gradient")
    return {
        "dataset": "ele-fashion",
        "variant": "V2",
        "seed": 42,
        "loss": loss_first_value,
        "second_step_loss": float(loss2.detach().cpu()),
        "aux_loss_first_step": aux_first_value,
        "first_step_gradient_diagnostics": first_step_gradients,
        "second_step_gradient_diagnostics": second_step_gradients,
        "updated_parameter_groups": changed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device(device))),
        "device": torch.cuda.get_device_name(torch.device(device)),
    }


def current_freeze() -> str | None:
    value = git("log", "--format=%H", "-1", f"--grep=^{FREEZE_SUBJECT}$")
    return value or None


def preflight(physical_gpu: int) -> dict[str, Any]:
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != BRANCH or head != BASE_SHA:
        raise RuntimeError(f"preflight requires {BRANCH} at {BASE_SHA}; found {branch} {head}")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError("new V2 branch must descend from the specified Information Utility evidence commit")
    caches = verify_cache_entries()
    tests = run_full_tests()
    initialization = initialization_audit(caches)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"physical GPU {physical_gpu} was not isolated as the only visible CUDA device")
    gpu = environment_record(physical_gpu)
    selected_free = next((int(row["free_mib"]) for row in gpu["gpus"] if row["index"] == physical_gpu), -1)
    if selected_free <= 2048:
        raise RuntimeError(f"selected GPU has only {selected_free} MiB free; inspect active processes before the e-fashion smoke")
    smoke = gpu_ele_fashion_smoke(caches["ele-fashion"], "cuda:0")
    result = {
        "phase": "implementation_freeze_ready",
        "recorded_at_utc": now(),
        "branch": branch,
        "baseline_commit_sha": BASE_SHA,
        "head_at_preflight": head,
        "cache_audits": {name: item["cache_audit"] for name, item in caches.items()},
        "environment": gpu,
        "initialization_pairing": initialization,
        "rng_and_runtime_audit": {
            "formal_seeds": list(SEEDS),
            "seed_helper": "src.utils.seeds.set_seed seeds Python random, NumPy, torch CPU, and all visible CUDA RNGs before model construction",
            "added_module_rng_isolation": "V1/V2 added modules initialize inside torch.random.fork_rng(devices=[]); trunk and later classifier initialization remain paired",
            "dropout_probability": 0.2,
            "activation_checkpointing": "non-reentrant PyTorch checkpoint for expert MLPs, edge-score chunks, and node-router chunks; preserve_rng_state default is retained",
            "training_evaluation_rng": "Train/Val scorecards reuse the epoch's eval embeddings and a deterministic linear head; they add no model forward and no dropout draw",
            "test_label_access": "no Test-indexed label reads or Test evaluation",
            "cuda_determinism": "reported from runtime; not changed by this study",
            "tf32": "reported from runtime; not changed by this study",
            "sparse_cuda_note": "index_add/scatter accumulation may have backend-dependent floating-point ordering; no blanket CUDA attribution is made",
        },
        "regression_tests": tests,
        "ele_fashion_gpu_smoke": smoke,
        "test_evaluation": False,
        "test_label_values_indexed": False,
        "formal_runs_planned": 27,
        "freeze_commit_subject": FREEZE_SUBJECT,
    }
    initialize_artifact_files()
    write_json(DATA_DIR / "implementation_checks.json", result)
    manifest = {
        "study": "RM-CGE V2 Edge-Conditioned Collaborative Graph Experts",
        "branch": BRANCH,
        "baseline_commit_sha": BASE_SHA,
        "implementation_freeze_sha": None,
        "status": "implementation_freeze_ready",
        "protocol_version": PROTOCOL,
        "formal_runs_planned": 27,
        "planned_runs": [
            {"dataset": dataset, "variant": variant, "seed": seed, "status": "planned"}
            for dataset in DATASETS for variant in VARIANTS for seed in SEEDS
        ],
        "implementation_checks_sha256": sha256_file(DATA_DIR / "implementation_checks.json"),
        "created_at_utc": now(),
        "test_evaluation": False,
        "test_label_values_indexed": False,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    return result


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"rmcge_v2.{path.parent.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    if not new_file:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle), None)
        if existing_fields != fields:
            raise RuntimeError(
                f"CSV schema mismatch for {path}: expected {fields}, found {existing_fields}"
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False, default=json_default)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def ensure_csv_header(path: Path, fields: list[str]) -> None:
    """Create or update a header-only artifact; never reinterpret data rows."""
    if not path.exists():
        append_csv(path, [], fields)
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        existing_fields = next(reader, None)
        has_data = next(reader, None) is not None
    if existing_fields == fields:
        return
    if has_data:
        raise RuntimeError(
            f"Refusing to change CSV schema with existing data in {path}: "
            f"expected {fields}, found {existing_fields}"
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow(fields)


def initialize_artifact_files() -> None:
    for name, columns in ARTIFACT_CSV_FIELDS.items():
        path = DATA_DIR / name
        ensure_csv_header(path, columns)
    if not (DATA_DIR / "failures.json").exists():
        write_json(DATA_DIR / "failures.json", [])
    if not (DATA_DIR / "gradient_audit.json").exists():
        write_json(DATA_DIR / "gradient_audit.json", {"runs": []})


def append_failure(record: dict[str, Any]) -> None:
    failures = read_json(DATA_DIR / "failures.json", [])
    failures.append(record)
    write_json(DATA_DIR / "failures.json", failures)


def run_one(dataset: str, variant: str, seed: int, device: str, cache: dict[str, Any]) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf
    from src.data import load_mag_data
    from src.tasks.nc import run_nc

    run_dir = OUTPUT / "runs" / dataset / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(dataset, variant, seed, device, cache)
    cfg.task.save_ckpt_path = str(run_dir / "best.pt")
    cfg.task.best_f1_ckpt_path = str(run_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(run_dir / "run_metrics.json")
    cfg.paths.output_root = str(run_dir / "hydra")
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True), encoding="utf-8")
    data = load_mag_data(cfg, "nc", seed)
    logger = setup_logger(run_dir / "train.log")
    started = time.perf_counter()
    metrics = run_nc(cfg, data, torch.device(device), logger)
    torch.cuda.synchronize(torch.device(device)) if torch.device(device).type == "cuda" else None
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "status": "completed_training_pending_integrity_check",
        "val_acc": metrics["val_acc"][0],
        "val_acc_std": metrics["val_acc"][1],
        "val_macro_f1": metrics["val_macro_f1"][0],
        "val_macro_f1_std": metrics["val_macro_f1"][1],
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_path": str((run_dir / "best.pt").relative_to(ROOT)),
        "run_metrics_path": str((run_dir / "run_metrics.json").relative_to(ROOT)),
        "config_sha256": training_config_hash(cfg),
        "test_metrics_present": False,
    }


def _eval_val_metrics(head, z, data):
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels

    labels = _resolve_nc_eval_labels(data, include_test=False)
    return _evaluate_split(head, z, data.y, data.val_idx, z.device, 4096, labels)


@torch.no_grad()
def _baseline_v0_diagnostics(model, x, edge_index, z) -> dict[str, Any]:
    """Expose V0 scales from the parent helpers without changing its forward."""
    import torch

    inputs = model._split_modalities(x)
    priors = {name: model.projectors[name](inputs[name]) for name in ("text", "visual")}
    src, dst, norm, active, _ = model._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
    degree = torch.zeros(x.size(0), device=x.device, dtype=priors["text"].dtype)
    degree.index_add_(0, dst, torch.ones(dst.numel(), device=x.device, dtype=degree.dtype))
    alpha = model._effective_alpha(model.alpha_raw, model.eps)
    modalities: dict[str, Any] = {}
    outputs: dict[str, torch.Tensor] = {}
    for modality_index, modality in enumerate(("text", "visual")):
        prior = priors[modality]
        state = prior
        states = []
        for _ in range(model.physical_hops):
            state = model._propagate(state, src, dst, norm)
            state = state * active.to(state.dtype).unsqueeze(-1)
            states.append(state)
        normalized = torch.stack([model._active_rms_norm(value, active) for value in states], dim=0)
        profiles, expert_values = model._expert_profiles(alpha, normalized, model.experts, model.expert_feature_scale)
        route = model._physical_route(modality_index, prior, active)
        mixture = torch.einsum("nk,knd->nd", route["route_weights"], expert_values)
        correction = route["strength"].unsqueeze(-1) * mixture
        outputs[modality] = prior + correction
        modalities[modality] = {
            "u1_rms": model._scalar_rms(states[0], active),
            "delta_rms": 0.0,
            "delta_hat_rms": 0.0,
            "delta_weighted_rms_by_expert": [0.0] * model.num_experts,
            "cross_rms": 0.0,
            "cross_hat_rms": 0.0,
            "cross_weighted_rms_by_expert": [0.0] * model.num_experts,
            "knn_rms": 0.0,
            "knn_hat_rms": 0.0,
            "knn_weighted_rms_by_expert": [0.0] * model.num_experts,
            "floor_hit_dimensions": 0,
            "physical_profile_rms_by_expert": [model._scalar_rms(profiles[k], active) for k in range(model.num_experts)],
            "combined_profile_rms_by_expert": [model._scalar_rms(profiles[k], active) for k in range(model.num_experts)],
            "expert_output_rms": model._scalar_rms(expert_values, active),
            "expert_output_rms_by_expert": [
                model._scalar_rms(expert_values[k], active)
                for k in range(model.num_experts)
            ],
            "expert_output_pairwise_cosine": model._expert_pairwise_cosines(
                expert_values, active
            ),
            "weighted_graph_correction_rms": model._scalar_rms(correction, active),
            "intrinsic_rms": model._scalar_rms(prior, active),
            "router_correction_rms": 0.0,
            "router_entropy": model._route_entropy(route["dense_probs"], active),
            "dense_probability_mean": route["dense_probs"][active].mean(dim=0).cpu().tolist() if bool(active.any()) else [0.0] * model.num_experts,
            "expert_selection_share": route["selection_share"].cpu().tolist(),
            "base_expert_selection_share": route["selection_share"].cpu().tolist(),
            "router_top2_flip_fraction": 0.0,
            "expert_coefficients": {"delta": [0.0] * model.num_experts, "cross": [0.0] * model.num_experts, "knn": [0.0] * model.num_experts},
        }
    fusion_input = torch.cat((outputs["text"], outputs["visual"]), dim=-1)
    edge_weights = {}
    target_degree = degree[dst]
    ones = norm.new_ones(src.numel())
    quantile_values = torch.quantile(ones, ones.new_tensor([0.1, 0.5, 0.9])) if ones.numel() else ones.new_zeros(3)
    for modality in ("text", "visual"):
        bins = {}
        for name, mask in (("1_2", (target_degree >= 1) & (target_degree <= 2)), ("3_5", (target_degree >= 3) & (target_degree <= 5)), ("6_10", (target_degree >= 6) & (target_degree <= 10)), ("11_20", (target_degree >= 11) & (target_degree <= 20)), ("gt20", target_degree > 20)):
            if bool(mask.any()):
                bins[name] = 1.0
        edge_weights[modality] = {"mean": 1.0 if ones.numel() else 0.0, "std": 0.0, "p10": float(quantile_values[0]), "p50": float(quantile_values[1]), "p90": float(quantile_values[2]), "mean_abs_from_one": 0.0, "fraction_abs_gt_0_1": 0.0, "fraction_abs_gt_0_2": 0.0, "mean_by_target_degree_bin": bins, "mean_abs_reverse_difference": 0.0}
    edge_weights["text_visual_correlation"] = 0.0
    edge_weights["text_visual_correlation_defined"] = False
    edge_weights["text_visual_mean_abs_difference"] = 0.0
    return {
        "modalities": modalities,
        "edge_weights": edge_weights,
        "fusion_input_rms": model._scalar_rms(fusion_input),
        "fusion_input_mean": float(fusion_input.float().mean().cpu()),
        "fusion_input_std": float(fusion_input.float().std(unbiased=False).cpu()),
        "fused_output_rms": model._scalar_rms(z),
        "fused_output_mean": float(z.float().mean().cpu()),
        "fused_output_std": float(z.float().std(unbiased=False).cpu()),
    }


def diagnose_checkpoint(dataset: str, variant: str, seed: int, cache: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from src.data import load_mag_data
    from src.models import build_model

    run_dir = OUTPUT / "runs" / dataset / variant / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    metrics_path = run_dir / "run_metrics.json"
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"missing run checkpoint or metrics for {dataset}/{variant}/seed{seed}")
    cfg = build_config(dataset, variant, seed, device, cache)
    data = load_mag_data(cfg, "nc", seed)
    info = data_info(data)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "test_acc" in payload.get("metrics", {}) or "test_macro_f1" in payload.get("metrics", {}):
        raise RuntimeError("Test metrics unexpectedly exist in a validation-only checkpoint")
    best_f1_path = run_dir / "best_f1.pt"
    if not best_f1_path.is_file():
        raise FileNotFoundError(f"missing Best-Macro-F1 diagnostic checkpoint: {best_f1_path}")
    best_f1_payload = torch.load(best_f1_path, map_location="cpu", weights_only=False)
    if best_f1_payload.get("selection") != "best_val_macro_f1_within_accuracy_stopping_horizon":
        raise RuntimeError("Best-Macro-F1 checkpoint has an unexpected selection rule")
    if "test_acc" in best_f1_payload.get("metrics", {}) or "test_macro_f1" in best_f1_payload.get("metrics", {}):
        raise RuntimeError("Test metrics unexpectedly exist in the Best-Macro-F1 checkpoint")
    if any(not torch.isfinite(value).all() for value in payload["model_state"].values() if torch.is_floating_point(value)):
        raise RuntimeError("non-finite model value found in selected checkpoint")
    if any(not torch.isfinite(value).all() for value in payload["head_state"].values() if torch.is_floating_point(value)):
        raise RuntimeError("non-finite classifier value found in selected checkpoint")
    if any(not torch.isfinite(value).all() for value in best_f1_payload["model_state"].values() if torch.is_floating_point(value)):
        raise RuntimeError("non-finite model value found in Best-Macro-F1 checkpoint")
    model = build_model(cfg, info)
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    head = torch.nn.Linear(model.out_dim, info["num_classes"]).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    x, edge = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        open_z, _, _, _, open_info = model(x, edge, return_details=True)
    if variant == "V0":
        open_info["details"] = _baseline_v0_diagnostics(model, x, edge, open_z)
    if not torch.isfinite(open_z).all():
        raise RuntimeError("checkpoint produces non-finite full-graph embeddings")
    open_metrics = _eval_val_metrics(head, open_z, data)
    expected = payload["metrics"]
    if abs(open_metrics["acc"] - float(expected["val_acc"])) > 1e-8:
        raise RuntimeError(f"reloaded checkpoint Validation Accuracy mismatch: {open_metrics['acc']} vs {expected['val_acc']}")
    model.load_state_dict(best_f1_payload["model_state"], strict=True)
    head.load_state_dict(best_f1_payload["head_state"], strict=True)
    with torch.no_grad():
        best_f1_z = model(x, edge)[0]
    best_f1_metrics = _eval_val_metrics(head, best_f1_z, data)
    expected_f1 = best_f1_payload["metrics"]
    if abs(best_f1_metrics["acc"] - float(expected_f1["val_acc"])) > 1e-8:
        raise RuntimeError("reloaded Best-Macro-F1 checkpoint does not reproduce its Validation Accuracy")
    if abs(best_f1_metrics["macro_f1"] - float(expected_f1["val_macro_f1"])) > 1e-8:
        raise RuntimeError("reloaded Best-Macro-F1 checkpoint does not reproduce its Validation Macro-F1")
    del best_f1_z
    model.load_state_dict(payload["model_state"], strict=True)
    head.load_state_dict(payload["head_state"], strict=True)
    open_logits = head(open_z[data.val_idx.to(device)]).detach()
    intervention_rows = []
    interventions = (
        ("dynamic_edges_unit", {"force_unit_edges": True}),
        ("cross_context_off", {"cross_off": True}),
        ("knn_context_off", {"knn_off": True}),
        ("node_router_off", {"node_router_off": True}),
    )
    for name, settings in interventions:
        model.set_interventions(**settings)
        with torch.no_grad():
            altered_z = model(x, edge)[0]
        altered_metrics = _eval_val_metrics(head, altered_z, data)
        altered_logits = head(altered_z[data.val_idx.to(device)]).detach()
        intervention_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "intervention": name,
                "val_acc": altered_metrics["acc"],
                "val_macro_f1": altered_metrics["macro_f1"],
                "val_acc_delta": altered_metrics["acc"] - open_metrics["acc"],
                "val_macro_f1_delta": altered_metrics["macro_f1"] - open_metrics["macro_f1"],
                "logit_delta_rms": float((altered_logits - open_logits).float().square().mean().sqrt().cpu()),
                "prediction_flip_fraction": float((altered_logits.argmax(-1) != open_logits.argmax(-1)).float().mean().cpu()),
            }
        )
        del altered_z, altered_logits
    model.set_interventions()
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_record = run_metrics["runs"][0]
    history = run_record["metadata"].get("epoch_history", [])
    if not history:
        raise RuntimeError("extended Validation epoch history is absent")
    if any(not all(math_isfinite(row.get(key)) for key in ("train_loss", "val_acc", "val_macro_f1", "train_ce", "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce")) for row in history):
        raise RuntimeError("non-finite metric found in epoch history")
    if len(history) != int(run_record["metadata"]["epochs_completed"]):
        raise RuntimeError("epoch history length disagrees with trainer epoch count")
    gradient = run_record["metadata"].get("gradient_diagnostics") or {}
    update_norms = initial_vs_final_update_norms(dataset, variant, seed, cache, payload, info)
    for name, record in gradient.items():
        if record.get("gradient_present") and not record.get("gradient_finite", False):
            raise RuntimeError(f"non-finite gradient recorded for {name}")
    required_update_groups = {
        "V0": ("experts", "fusion_linear1", "projectors"),
        "V1": ("cross_projections", "node_router", "experts", "fusion_linear1", "projectors"),
        "V2": ("edge_scorer", "cross_projections", "node_router", "experts", "fusion_linear1", "projectors"),
    }[variant]
    missing_updates = [
        name for name in required_update_groups
        if name not in update_norms or float(update_norms[name]["l2_update_norm"]) <= 0.0
    ]
    if missing_updates:
        raise RuntimeError(f"required parameter groups did not update: {missing_updates}")
    if variant == "V2" and not gradient.get("edge_scorer.2.weight", {}).get("gradient_nonzero", False):
        raise RuntimeError("V2 Edge Scorer output layer has no nonzero first-step gradient")
    result = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "reloaded_checkpoint_matches_validation_accuracy": True,
        "best_epoch": int(payload["epoch"]),
        "best_f1_epoch": int(best_f1_payload["epoch"]),
        "best_f1_checkpoint_reloads": True,
        "best_f1_val_accuracy": best_f1_metrics["acc"],
        "best_f1_val_macro_f1": best_f1_metrics["macro_f1"],
        "val_accuracy": open_metrics["acc"],
        "val_macro_f1": open_metrics["macro_f1"],
        "epoch_count": len(history),
        "history": history,
        "gradient_diagnostics": gradient,
        "parameter_update_norms": update_norms,
        "mechanism_details": open_info.get("details", {}),
        "interventions": intervention_rows,
        "test_metrics_present": False,
    }
    write_json(run_dir / "checkpoint_diagnostics.json", result)
    del model, head, open_z, open_logits, data, payload, best_f1_payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def math_isfinite(value: Any) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def initial_vs_final_update_norms(dataset: str, variant: str, seed: int, cache: dict[str, Any], payload: dict[str, Any], info: dict[str, int]) -> dict[str, Any]:
    import torch
    from src.utils.seeds import set_seed

    cfg = build_config(dataset, variant, seed, "cpu", cache)
    model, head = initialize_model(cfg, info, cache, seed)
    final = payload["model_state"]
    groups: dict[str, dict[str, float | int]] = {}
    for name, parameter in model.named_parameters():
        if name not in final:
            continue
        delta = final[name].detach().cpu().float() - parameter.detach().cpu().float()
        top_level = name.split(".", 1)[0]
        record = groups.setdefault(top_level, {"tensors": 0, "updated_tensors": 0, "squared_norm": 0.0})
        record["tensors"] += 1
        norm = float(delta.norm())
        record["squared_norm"] += norm * norm
        if norm > 0:
            record["updated_tensors"] += 1
    for record in groups.values():
        record["l2_update_norm"] = record.pop("squared_norm") ** 0.5
    del model, head
    return groups


def update_manifest_run(row: dict[str, Any]) -> None:
    manifest_path = DATA_DIR / "manifest.json"
    manifest = read_json(manifest_path, {})
    if not manifest:
        manifest = {
            "study": "RM-CGE V2",
            "branch": BRANCH,
            "baseline_commit_sha": BASE_SHA,
            "planned_runs": [
                {"dataset": d, "variant": v, "seed": s, "status": "planned"}
                for d in DATASETS for v in VARIANTS for s in SEEDS
            ],
        }
    manifest["implementation_freeze_sha"] = current_freeze()
    target = (row["dataset"], row["variant"], int(row["seed"]))
    for planned in manifest["planned_runs"]:
        if (planned["dataset"], planned["variant"], int(planned["seed"])) == target:
            planned.update({key: value for key, value in row.items() if key != "epoch_history"})
    manifest["updated_at_utc"] = now()
    write_json(manifest_path, manifest)


def _require_freeze():
    branch, head, freeze = git("branch", "--show-current"), git("rev-parse", "HEAD"), current_freeze()
    if branch != BRANCH or freeze is None or head != freeze:
        raise RuntimeError(f"formal execution requires HEAD at {FREEZE_SUBJECT}; branch={branch}, head={head}, freeze={freeze}")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError("formal run branch ancestry differs from the required baseline")
    return freeze


def campaign(seeds: tuple[int, ...], device: str) -> list[dict[str, Any]]:
    _require_freeze()
    checks = read_json(DATA_DIR / "implementation_checks.json", {})
    verified_batches = {int(item["seed"]) for item in checks.get("seed_batch_verifications", [])}
    if any(seed > 42 for seed in seeds) and 42 not in verified_batches:
        raise RuntimeError("seeds 43/44 are gated on a passing seed-42 integrity review")
    if 44 in seeds and 43 not in verified_batches:
        raise RuntimeError("seed 44 is gated on a passing seed-43 integrity review")
    caches = verify_cache_entries()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for dataset in DATASETS:
            for variant in VARIANTS:
                key = f"{dataset}/{variant}/seed_{seed}"
                try:
                    row = run_one(dataset, variant, seed, device, caches[dataset])
                    diagnostics = diagnose_checkpoint(dataset, variant, seed, caches[dataset], device)
                    row["status"] = "completed_and_verified"
                    row["best_epoch"] = diagnostics["best_epoch"]
                    row["val_acc"] = diagnostics["val_accuracy"]
                    row["val_macro_f1"] = diagnostics["val_macro_f1"]
                    row["diagnostics_path"] = str((OUTPUT / "runs" / dataset / variant / f"seed_{seed}" / "checkpoint_diagnostics.json").relative_to(ROOT))
                    rows.append(row)
                    update_manifest_run(row)
                    append_csv(DATA_DIR / "runs.csv", [row], ["dataset", "variant", "seed", "status", "best_epoch", "val_acc", "val_macro_f1", "wall_seconds", "checkpoint_path", "run_metrics_path", "diagnostics_path", "config_sha256", "test_metrics_present", "error"])
                    all_history = [
                        {"dataset": dataset, "variant": variant, "seed": seed, **epoch}
                        for epoch in diagnostics["history"]
                    ]
                    append_csv(DATA_DIR / "epoch_history.csv", all_history, ["dataset", "variant", "seed", "epoch", "train_loss", "train_objective", "train_ce", "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce", "val_acc", "val_macro_f1", "epoch_wall_seconds", "learning_rate"])
                    append_csv(DATA_DIR / "frozen_interventions.csv", diagnostics["interventions"], ["dataset", "variant", "seed", "intervention", "val_acc", "val_macro_f1", "val_acc_delta", "val_macro_f1_delta", "logit_delta_rms", "prediction_flip_fraction"])
                    append_mechanism_rows(diagnostics)
                    append_gradient_resource_rows(row, diagnostics)
                except Exception as exc:
                    failure = {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "status": "failed",
                        "recorded_at_utc": now(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "existing_artifacts_retained": True,
                    }
                    append_failure(failure)
                    failed_row = {"dataset": dataset, "variant": variant, "seed": seed, "status": "failed", "error": failure["error"]}
                    rows.append(failed_row)
                    update_manifest_run(failed_row)
                    append_csv(DATA_DIR / "runs.csv", [failed_row], ["dataset", "variant", "seed", "status", "best_epoch", "val_acc", "val_macro_f1", "wall_seconds", "checkpoint_path", "run_metrics_path", "diagnostics_path", "config_sha256", "test_metrics_present", "error"])
                    raise
    return rows


def append_mechanism_rows(diagnostics: dict[str, Any]) -> None:
    edge = diagnostics["mechanism_details"].get("edge_weights", {})
    for modality in ("text", "visual"):
        if modality in edge:
            append_csv(DATA_DIR / "edge_weight_diagnostics.csv", [{"dataset": diagnostics["dataset"], "variant": diagnostics["variant"], "seed": diagnostics["seed"], "modality": modality, **edge[modality], "text_visual_correlation": edge.get("text_visual_correlation"), "text_visual_correlation_defined": edge.get("text_visual_correlation_defined"), "text_visual_mean_abs_difference": edge.get("text_visual_mean_abs_difference")}], ["dataset", "variant", "seed", "modality", "mean", "std", "p10", "p50", "p90", "mean_abs_from_one", "fraction_abs_gt_0_1", "fraction_abs_gt_0_2", "mean_by_target_degree_bin", "mean_abs_reverse_difference", "text_visual_correlation", "text_visual_correlation_defined", "text_visual_mean_abs_difference"])
    for modality, values in diagnostics["mechanism_details"].get("modalities", {}).items():
        row = {"dataset": diagnostics["dataset"], "variant": diagnostics["variant"], "seed": diagnostics["seed"], "modality": modality, **values}
        append_csv(DATA_DIR / "message_scale_diagnostics.csv", [row], ["dataset", "variant", "seed", "modality", "u1_rms", "delta_rms", "delta_hat_rms", "delta_weighted_rms_by_expert", "cross_rms", "cross_hat_rms", "cross_weighted_rms_by_expert", "knn_rms", "knn_hat_rms", "knn_weighted_rms_by_expert", "floor_hit_dimensions", "physical_profile_rms_by_expert", "combined_profile_rms_by_expert", "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine", "weighted_graph_correction_rms", "intrinsic_rms", "router_correction_rms", "router_entropy", "dense_probability_mean", "expert_selection_share", "base_expert_selection_share", "router_top2_flip_fraction", "expert_coefficients"])
        append_csv(DATA_DIR / "expert_router_diagnostics.csv", [{key: row.get(key) for key in ("dataset", "variant", "seed", "modality", "expert_selection_share", "base_expert_selection_share", "dense_probability_mean", "router_entropy", "router_correction_rms", "router_top2_flip_fraction", "expert_coefficients", "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine")}], ["dataset", "variant", "seed", "modality", "expert_selection_share", "base_expert_selection_share", "dense_probability_mean", "router_entropy", "router_correction_rms", "router_top2_flip_fraction", "expert_coefficients", "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine"])


def append_gradient_resource_rows(row: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    run_json = json.loads((ROOT / row["run_metrics_path"]).read_text(encoding="utf-8"))
    run_meta = run_json["runs"][0]["metadata"]
    gradients = diagnostics.get("gradient_diagnostics", {})
    existing = read_json(DATA_DIR / "gradient_audit.json", {"runs": []})
    existing["runs"].append({"dataset": row["dataset"], "variant": row["variant"], "seed": row["seed"], "gradients": gradients, "parameter_update_norms": diagnostics["parameter_update_norms"]})
    write_json(DATA_DIR / "gradient_audit.json", existing)
    append_csv(DATA_DIR / "resource_profile.csv", [{"dataset": row["dataset"], "variant": row["variant"], "seed": row["seed"], "wall_seconds": row["wall_seconds"], "epochs_completed": run_meta.get("epochs_completed"), "mean_epoch_wall_seconds": run_meta.get("mean_epoch_wall_seconds"), "peak_allocated_bytes": run_meta.get("cuda_peak_allocated_bytes"), "peak_reserved_bytes": run_meta.get("cuda_peak_reserved_bytes"), "cuda_device_name": run_meta.get("cuda_device_name")}], ["dataset", "variant", "seed", "wall_seconds", "epochs_completed", "mean_epoch_wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "cuda_device_name"])


def verify_batch(seed: int) -> dict[str, Any]:
    manifest = read_json(DATA_DIR / "manifest.json", {})
    rows = [item for item in manifest.get("planned_runs", []) if int(item["seed"]) == seed]
    expected = {(d, v) for d in DATASETS for v in VARIANTS}
    found = {(item["dataset"], item["variant"]) for item in rows if item.get("status") == "completed_and_verified"}
    if found != expected:
        raise RuntimeError(f"seed-{seed} verification incomplete: verified {sorted(found)}, expected {sorted(expected)}")
    records = [item for item in rows if item.get("status") == "completed_and_verified"]
    if any(item.get("test_metrics_present") for item in records):
        raise RuntimeError("Test metrics were found in the seed batch")
    result = {"seed": seed, "verified_runs": len(records), "all_histories_finite": True, "test_evaluation": False, "verified_at_utc": now()}
    checks = read_json(DATA_DIR / "implementation_checks.json", {})
    checks.setdefault("seed_batch_verifications", []).append(result)
    write_json(DATA_DIR / "implementation_checks.json", checks)
    if seed == 44:
        manifest["status"] = "all_27_runs_verified"
    manifest["updated_at_utc"] = now()
    write_json(DATA_DIR / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run-batch", "verify-batch", "run-all"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--gpu", type=int, default=1, help="physical GPU index to isolate")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    if args.mode == "preflight":
        try:
            result = preflight(args.gpu)
        except Exception as exc:
            initialize_artifact_files()
            append_failure(
                {
                    "phase": "preflight",
                    "recorded_at_utc": now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "existing_artifacts_retained": True,
                }
            )
            raise
        print(json.dumps({"status": result["phase"], "tests": result["regression_tests"].get("passed"), "gpu_smoke_peak_bytes": result["ele_fashion_gpu_smoke"]["peak_allocated_bytes"]}, indent=2))
        return
    if args.mode == "verify-batch":
        if args.seed is None:
            parser.error("verify-batch requires --seed")
        print(json.dumps(verify_batch(args.seed), indent=2))
        return
    if args.mode == "run-batch":
        if args.seed is None:
            parser.error("run-batch requires --seed")
        rows = campaign((args.seed,), "cuda:0")
    else:
        campaign((42,), "cuda:0")
        verify_batch(42)
        campaign((43,), "cuda:0")
        verify_batch(43)
        rows = campaign((44,), "cuda:0")
        verify_batch(44)
    print(json.dumps({"completed_rows": rows}, indent=2))


if __name__ == "__main__":
    main()
