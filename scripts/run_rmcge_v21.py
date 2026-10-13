#!/usr/bin/env python3
"""Preflight, run, and audit the RM-CGE V2.1 validation-only study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRANCH = "exp/rmcge_v21_message_purpose_selection"
BASE_SHA = "ba1fbbd100a0b218ec66853e29ea0cfee8cfa5b4"
FREEZE_SUBJECT = "Freeze RM-CGE V2.1 implementation"
PROTOCOL = "unified_full_graph_nc_v1"
DATASETS = ("Movies", "Grocery", "ele-fashion")
CONDITIONS = ("C0", "C1", "C2")
SEEDS = (42, 43, 44)
RESEARCH = ROOT / "research/rmcge_v21"
DATA_DIR = RESEARCH / "data"
OUTPUT = ROOT / "outputs/rmcge_v21"

CSV_FIELDS = {
    "runs.csv": [
        "dataset", "condition", "seed", "status", "best_accuracy_epoch", "best_f1_epoch",
        "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
        "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce", "train_val_acc_gap",
        "train_val_macro_f1_gap", "model_parameters", "classifier_parameters", "wall_seconds",
        "peak_allocated_bytes", "peak_reserved_bytes", "checkpoint_path", "best_f1_checkpoint_path",
        "run_metrics_path", "diagnostics_path", "config_sha256", "initial_model_sha256",
        "initial_classifier_sha256", "test_metrics_present", "error",
    ],
    "epoch_history.csv": [
        "dataset", "condition", "seed", "epoch", "train_loss", "train_objective", "train_ce",
        "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce", "val_acc",
        "val_macro_f1", "epoch_wall_seconds", "learning_rate",
    ],
    "paired_comparisons.csv": ["dataset", "seed", "metric", "contrast", "difference", "left_value", "right_value"],
    "edge_selection_diagnostics.csv": [
        "dataset", "condition", "seed", "target_modality", "message_purpose", "selection_enabled",
        "selection_status", "edge_count", "receiver_row_mass_error_max", "receiver_row_mass_error_mean",
        "degree_ge2_node_fraction", "degree_one_node_fraction", "relative_weight_mean",
        "relative_weight_mean_abs_deviation", "relative_weight_std",
        "relative_weight_p05", "relative_weight_p50", "relative_weight_p95",
        "within_receiver_relative_weight_std_mean_degree_ge2", "neighbor_entropy_physical_mean_degree_ge2",
        "neighbor_entropy_selected_mean_degree_ge2", "neighbor_entropy_selected_mean_degree1",
        "effective_neighbors_mean_degree_ge2", "effective_neighbors_mean_degree1",
        "kl_selected_to_physical_mean_degree_ge2", "js_selected_physical_mean_degree_ge2",
        "score_abs_ge_0_99_fraction", "score_abs_mean", "reverse_direction_selected_weight_abs_difference",
    ],
    "role_distribution_diagnostics.csv": [
        "dataset", "condition", "seed", "path_pair", "edge_probability_mean_abs_difference",
        "edge_probability_correlation", "receiver_js_mean_degree_ge2", "status",
    ],
    "message_scale_diagnostics.csv": [
        "dataset", "condition", "seed", "modality", "u1_rms", "delta_intra_rms", "delta_intra_hat_rms",
        "delta_intra_weighted_rms_by_expert", "cross_base_rms", "cross_base_hat_rms",
        "cross_base_weighted_rms_by_expert", "delta_cross_rms", "delta_cross_hat_rms",
        "delta_cross_weighted_rms_by_expert", "knn_rms", "knn_hat_rms", "knn_weighted_rms_by_expert",
        "floor_hit_dimensions", "physical_profile_rms_by_expert", "combined_profile_rms_by_expert",
        "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine",
        "weighted_graph_correction_rms", "intrinsic_rms", "router_correction_rms", "router_entropy",
        "dense_probability_mean", "expert_selection_share", "base_expert_selection_share",
        "router_top2_flip_fraction", "expert_coefficients",
    ],
    "expert_router_diagnostics.csv": [
        "dataset", "condition", "seed", "modality", "expert_selection_share", "base_expert_selection_share",
        "dense_probability_mean", "router_entropy", "router_correction_rms", "router_top2_flip_fraction",
        "expert_coefficients", "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine",
    ],
    "frozen_interventions.csv": [
        "dataset", "condition", "seed", "intervention", "status", "val_acc", "val_macro_f1",
        "val_acc_delta", "val_macro_f1_delta", "logit_delta_rms", "prediction_flip_fraction",
    ],
    "resource_profile.csv": [
        "dataset", "condition", "seed", "wall_seconds", "epochs_completed", "mean_epoch_wall_seconds",
        "peak_allocated_bytes", "peak_reserved_bytes", "cuda_device_name",
    ],
    "summary.csv": [
        "dataset", "condition", "metric", "mean", "sample_sd", "seed_42", "seed_43", "seed_44",
    ],
}
CSV_KEY_FIELDS = {
    "runs.csv": ("dataset", "condition", "seed"),
    "epoch_history.csv": ("dataset", "condition", "seed", "epoch"),
    "paired_comparisons.csv": ("dataset", "seed", "metric", "contrast"),
    "edge_selection_diagnostics.csv": ("dataset", "condition", "seed", "target_modality", "message_purpose"),
    "role_distribution_diagnostics.csv": ("dataset", "condition", "seed", "path_pair"),
    "message_scale_diagnostics.csv": ("dataset", "condition", "seed", "modality"),
    "expert_router_diagnostics.csv": ("dataset", "condition", "seed", "modality"),
    "frozen_interventions.csv": ("dataset", "condition", "seed", "intervention"),
    "resource_profile.csv": ("dataset", "condition", "seed"),
    "summary.csv": ("dataset", "condition", "metric"),
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
    from scripts.run_rmcge_v2 import cache_entries as v2_cache_entries

    return v2_cache_entries()


def verify_cache_entries() -> dict[str, dict[str, Any]]:
    from scripts.run_rmcge_v2 import verify_cache_entries as v2_verify_cache_entries

    return v2_verify_cache_entries()


def build_config(dataset: str, condition: str, seed: int, device: str, cache: dict[str, Any]):
    from omegaconf import OmegaConf

    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create(
        {
            "paths": root_cfg.paths,
            "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
            "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
            "model": OmegaConf.load(ROOT / "configs/model/rmcge_v21.yaml"),
            "logging": root_cfg.logging,
            "seed": int(seed),
            "num_runs": 1,
            "device": str(device),
        }
    )
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
    cfg.task.best_f1_ckpt_path = None
    cfg.task.extended_checkpoint_metadata = {
        "study": "RM-CGE V2.1",
        "dataset": dataset,
        "condition": condition,
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

    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()).hexdigest()


def initialization_audit(caches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import torch
    from src.data import load_mag_data

    result: dict[str, Any] = {"datasets": {}, "pairs": {}}
    for dataset in DATASETS:
        cfg0 = build_config(dataset, "C0", 42, "cpu", caches[dataset])
        data = load_mag_data(cfg0, "nc", 42)
        fingerprints = {
            "num_nodes": int(data.num_nodes),
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
            "official_test_index_values_accessed": False,
        }
        result["datasets"][dataset] = fingerprints
        info = data_info(data)
        for seed in SEEDS:
            models, heads, hashes = {}, {}, {}
            for condition in CONDITIONS:
                cfg = build_config(dataset, condition, seed, "cpu", caches[dataset])
                model, head = initialize_model(cfg, info, caches[dataset], seed)
                models[condition], heads[condition] = model, head
                hashes[condition] = {
                    "model_state_sha256": module_hash(model),
                    "classifier_sha256": module_hash(head),
                    "model_parameter_count": sum(p.numel() for p in model.parameters()),
                    "classifier_parameter_count": sum(p.numel() for p in head.parameters()),
                    "parameter_shapes": {name: list(value.shape) for name, value in model.named_parameters()},
                }
            common_names = set(dict(models["C0"].named_parameters()))
            common_match = all(
                torch.equal(dict(models["C0"].named_parameters())[name], dict(models[condition].named_parameters())[name])
                for condition in ("C1", "C2") for name in common_names
            )
            c1_params, c2_params = dict(models["C1"].named_parameters()), dict(models["C2"].named_parameters())
            new_names = set(c1_params) - common_names
            selector_pair_match = set(c1_params) == set(c2_params) and all(
                torch.equal(c1_params[name], c2_params[name]) for name in c1_params
            )
            head_match = len({hashes[c]["classifier_sha256"] for c in CONDITIONS}) == 1
            record = {
                "conditions": hashes,
                "shared_v1_parameter_initialization_matches_c0": common_match,
                "c1_c2_complete_parameter_initialization_matches": selector_pair_match,
                "c1_c2_added_parameter_names": sorted(new_names),
                "classifier_initialization_matches": head_match,
            }
            result["pairs"][f"{dataset}/seed_{seed}"] = record
            if not (common_match and selector_pair_match and head_match):
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
        raise RuntimeError("full repository pytest failed:\n" + output[-18000:])
    import re

    passed = re.search(r"(\d+) passed", output)
    skipped = re.search(r"(\d+) skipped", output)
    failed = re.search(r"(\d+) failed", output)
    return {
        "command": command,
        "return_code": result.returncode,
        "passed": int(passed.group(1)) if passed else 0,
        "skipped": int(skipped.group(1)) if skipped else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "output_tail": output[-6000:],
    }


def initialize_artifacts() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "failures.json").exists():
        write_json(DATA_DIR / "failures.json", {"failures": []})
    if not (DATA_DIR / "gradient_audit.json").exists():
        write_json(DATA_DIR / "gradient_audit.json", {"runs": []})
    for name, fields in CSV_FIELDS.items():
        path = DATA_DIR / name
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(fields)
        else:
            with path.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
            if header != fields:
                raise RuntimeError(f"artifact schema mismatch in {path}: {header} != {fields}")


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerow(fields)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != fields:
        raise RuntimeError(f"CSV schema mismatch for {path}: {header} != {fields}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))
    key_fields = CSV_KEY_FIELDS.get(path.name, ())
    def csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value, sort_keys=True, allow_nan=False, default=json_default)
        return value

    if key_fields:
        positions = {tuple(str(row.get(key, "")) for key in key_fields): index for index, row in enumerate(existing)}
        for row in rows:
            normalized = {key: csv_value(row.get(key, "")) for key in fields}
            key = tuple(str(normalized.get(field, "")) for field in key_fields)
            if key in positions:
                existing[positions[key]] = normalized
            else:
                positions[key] = len(existing)
                existing.append(normalized)
    else:
        existing.extend({key: csv_value(row.get(key, "")) for key in fields} for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(path)


def append_failure(record: dict[str, Any]) -> None:
    failures = read_json(DATA_DIR / "failures.json", {"failures": []})
    failures["failures"].append(record)
    write_json(DATA_DIR / "failures.json", failures)


def assert_gpu_capacity(physical_gpu: int, minimum_free_mib: int = 16000) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5 and int(fields[0]) == physical_gpu:
            snapshot = {
                "index": int(fields[0]), "name": fields[1], "free_mib": int(fields[2]),
                "total_mib": int(fields[3]), "utilization_percent": int(fields[4]),
            }
            if snapshot["free_mib"] < minimum_free_mib:
                raise RuntimeError(f"GPU {physical_gpu} has only {snapshot['free_mib']} MiB free; pause before starting a run")
            return snapshot
    raise RuntimeError(f"nvidia-smi did not report physical GPU {physical_gpu}")


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"rmcge_v21.{path.parent.name}.{path.parent.parent.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def gpu_ele_fashion_smoke(cache: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.common import build_optimizer
    from src.utils.seeds import set_seed

    from scripts.run_rmcge_v2 import environment_record

    outcomes = {}
    for condition in ("C1", "C2"):
        cfg = build_config("ele-fashion", condition, 42, device, cache)
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
        model.to(device).train()
        head = torch.nn.Linear(model.out_dim, data.num_classes).to(device)
        optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), cfg, model=model)
        x, edge = data.x.to(device), data.edge_index.to(device)
        train_idx = data.train_idx.to(device)
        labels = data.y[data.train_idx].to(device)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        losses, gradients = [], []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            z, _, _, aux, _ = model(x, edge)
            loss = torch.nn.functional.cross_entropy(head(z[train_idx]), labels) + aux
            if not torch.isfinite(loss):
                raise RuntimeError(f"ele-fashion {condition} smoke produced non-finite loss")
            losses.append(float(loss.detach().cpu()))
            loss.backward()
            gradients.append(model.capture_gradient_diagnostics())
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize(torch.device(device))
        names = dict(model.named_parameters())
        required = [
            "purpose_edge_scorer.2.weight", "cross_projections.visual_to_text.0.weight",
            "cross_projections.text_to_visual.0.weight", "node_router.3.weight", "fusion_linear1.weight",
        ]
        changed = {name: not torch.equal(before[name], names[name]) for name in required}
        if not all(changed.values()):
            raise RuntimeError(f"ele-fashion {condition} smoke has missing parameter updates: {changed}")
        if not gradients[0]["purpose_edge_scorer.2.weight"]["gradient_nonzero"]:
            raise RuntimeError(f"ele-fashion {condition} selector output layer has no first-step task gradient")
        if not gradients[1]["query_projection.weight"]["gradient_nonzero"]:
            raise RuntimeError(f"ele-fashion {condition} query projection has no later task gradient")
        if condition == "C2" and not gradients[1]["cross_delta_coefficients"]["gradient_nonzero"]:
            raise RuntimeError("ele-fashion C2 Cross Delta coefficients have no later task gradient")
        outcomes[condition] = {
            "losses": losses,
            "gradient_diagnostics": gradients,
            "updated_parameter_groups": changed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device(device))),
            "device": torch.cuda.get_device_name(torch.device(device)),
        }
        del model, head, data, x, edge
        torch.cuda.empty_cache()
    return {"dataset": "ele-fashion", "seed": 42, "conditions": outcomes}


def preflight(physical_gpu: int) -> dict[str, Any]:
    import torch
    from scripts.run_rmcge_v2 import environment_record

    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != BRANCH:
        raise RuntimeError(f"preflight requires branch {BRANCH}; found {branch}")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError("V2.1 branch must descend directly from the specified V2 evidence commit")
    caches = verify_cache_entries()
    tests = run_full_tests()
    initialization = initialization_audit(caches)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"physical GPU {physical_gpu} is not visible as the only CUDA device")
    environment = environment_record(physical_gpu)
    gpu_row = next((row for row in environment["gpus"] if row["index"] == physical_gpu), None)
    if not gpu_row or int(gpu_row["free_mib"]) < 16000:
        raise RuntimeError(f"selected RTX 3090 must have at least 16 GiB free for ele-fashion smoke; got {gpu_row}")
    smoke = gpu_ele_fashion_smoke(caches["ele-fashion"], "cuda:0")
    result = {
        "phase": "implementation_freeze_ready",
        "recorded_at_utc": now(),
        "branch": branch,
        "baseline_commit_sha": BASE_SHA,
        "head_at_preflight": head,
        "cache_audits": {name: item["cache_audit"] for name, item in caches.items()},
        "environment": environment,
        "initialization_pairing": initialization,
        "rng_runtime_protocol": {
            "seed_helper": "src.utils.seeds.set_seed seeds Python, NumPy, CPU Torch and visible CUDA RNGs before model construction",
            "added_module_rng_isolation": "V2 and V2.1 added modules initialize inside fork_rng scopes; classifier initialization remains paired",
            "dropout": 0.2,
            "activation_checkpointing": "non-reentrant per edge chunk and per shared expert, with PyTorch RNG preservation enabled",
            "train_val_rng": "extended Train/Validation scorecards reuse deterministic eval embeddings and do not invoke a stochastic model forward",
            "tf32_and_determinism": "reported from the runtime and not changed by this study",
            "sparse_reduction_note": "CUDA scatter/index_add floating point summation order may vary by backend; this is not assumed to explain all variation",
            "test_label_access": "no Test-indexed label reads and no Test metrics",
        },
        "regression_tests": tests,
        "ele_fashion_gpu_smoke": smoke,
        "test_evaluation": False,
        "test_label_values_indexed": False,
        "formal_runs_planned": 27,
        "freeze_commit_subject": FREEZE_SUBJECT,
    }
    initialize_artifacts()
    write_json(DATA_DIR / "implementation_checks.json", result)
    manifest = {
        "study": "RM-CGE V2.1 Message-Purpose Conditioned Selective Graph Experts",
        "branch": BRANCH,
        "baseline_commit_sha": BASE_SHA,
        "implementation_freeze_sha": None,
        "status": "implementation_freeze_ready",
        "protocol_version": PROTOCOL,
        "formal_runs_planned": 27,
        "planned_runs": [
            {"dataset": d, "condition": c, "seed": s, "status": "planned"}
            for d in DATASETS for c in CONDITIONS for s in SEEDS
        ],
        "implementation_checks_sha256": sha256_file(DATA_DIR / "implementation_checks.json"),
        "cache_feature_candidate_v7b_fingerprints": {
            d: {key: caches[d][key] for key in ("feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint")}
            for d in DATASETS
        },
        "created_at_utc": now(),
        "test_evaluation": False,
        "test_label_values_indexed": False,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    return result


def _update_manifest_run(row: dict[str, Any]) -> None:
    manifest = read_json(DATA_DIR / "manifest.json", {})
    key = (row["dataset"], row["condition"], int(row["seed"]))
    for index, item in enumerate(manifest.get("planned_runs", [])):
        if (item["dataset"], item["condition"], int(item["seed"])) == key:
            manifest["planned_runs"][index] = {**item, **row}
            break
    else:
        raise RuntimeError(f"run key missing from manifest: {key}")
    manifest["updated_at_utc"] = now()
    write_json(DATA_DIR / "manifest.json", manifest)


def _metrics_from_history(history: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    for item in history:
        if int(item["epoch"]) == int(epoch):
            return item
    raise RuntimeError(f"checkpoint epoch {epoch} not found in epoch history")


def run_one(dataset: str, condition: str, seed: int, device: str, cache: dict[str, Any]) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf
    from src.data import load_mag_data
    from src.tasks.nc import run_nc
    import shutil

    run_dir = OUTPUT / "runs" / dataset / condition / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        archive = OUTPUT / "attempt_history" / dataset / condition / f"seed_{seed}__{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(archive))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(dataset, condition, seed, device, cache)
    cfg.task.save_ckpt_path = str(run_dir / "best.pt")
    cfg.task.best_f1_ckpt_path = str(run_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(run_dir / "run_metrics.json")
    cfg.paths.output_root = str(run_dir / "hydra")
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True), encoding="utf-8")
    data = load_mag_data(cfg, "nc", seed)
    logger = setup_logger(run_dir / "train.log")
    started = time.perf_counter()
    metrics = run_nc(cfg, data, torch.device(device), logger)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    row = {
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "status": "completed_training_pending_integrity_check",
        "val_acc": metrics["val_acc"][0],
        "val_macro_f1": metrics["val_macro_f1"][0],
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_path": str((run_dir / "best.pt").relative_to(ROOT)),
        "best_f1_checkpoint_path": str((run_dir / "best_f1.pt").relative_to(ROOT)),
        "run_metrics_path": str((run_dir / "run_metrics.json").relative_to(ROOT)),
        "config_sha256": training_config_hash(cfg),
        "initial_model_sha256": "",
        "initial_classifier_sha256": "",
        "test_metrics_present": False,
        "peak_allocated_bytes": payload.get("metadata", {}).get("cuda_peak_allocated_bytes", ""),
        "peak_reserved_bytes": payload.get("metadata", {}).get("cuda_peak_reserved_bytes", ""),
    }
    del data
    return row


def _eval_split(head, z, data, split: str) -> dict[str, float]:
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels

    labels = _resolve_nc_eval_labels(data, include_test=False)
    index = data.train_idx if split == "train" else data.val_idx
    result = _evaluate_split(head, z, data.y, index, z.device, 4096, labels)
    logits = head(z[index.to(z.device)])
    result["cross_entropy"] = float(torch.nn.functional.cross_entropy(logits, data.y[index].to(z.device)).cpu())
    return result


def _baseline_selection(model, x, edge_index) -> dict[str, dict[str, Any]]:
    import torch
    from src.models.rmcge_v21 import Model as V21Model

    prior = model.projectors["text"](model._split_modalities(x)["text"])
    src, dst, physical, _, _ = model._normalized_operator(edge_index, x.size(0), prior.dtype)
    degree = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.long, device=x.device))
    zeros = torch.zeros_like(physical)
    result = {}
    for modality in ("text", "visual"):
        for purpose in ("intra", "cross"):
            result[f"{modality}_{purpose}"] = V21Model._selection_path_diagnostics(
                physical, physical, zeros, src, dst, degree
            )
    return result


def _update_norms(dataset: str, condition: str, seed: int, cache: dict[str, Any], payload: dict[str, Any], info: dict[str, int]) -> dict[str, Any]:
    import torch

    cfg = build_config(dataset, condition, seed, "cpu", cache)
    initial, _ = initialize_model(cfg, info, cache, seed)
    records: dict[str, dict[str, float]] = {}
    for name, before in initial.named_parameters():
        after = payload["model_state"][name].detach().cpu()
        delta = (after.float() - before.detach().cpu().float()).norm().item()
        group = name.split(".", 1)[0]
        records.setdefault(group, {"parameter_count": 0, "l2_update_norm": 0.0})
        records[group]["parameter_count"] += before.numel()
        records[group]["l2_update_norm"] += delta * delta
    for value in records.values():
        value["l2_update_norm"] = math.sqrt(value["l2_update_norm"])
    del initial
    return records


def _append_run_evidence(row: dict[str, Any], diagnostic: dict[str, Any], run_metadata: dict[str, Any]) -> None:
    dataset, condition, seed = row["dataset"], row["condition"], row["seed"]
    fields = CSV_FIELDS
    append_csv(DATA_DIR / "runs.csv", [row], fields["runs.csv"])
    append_csv(
        DATA_DIR / "epoch_history.csv",
        [{"dataset": dataset, "condition": condition, "seed": seed, **item} for item in diagnostic["history"]],
        fields["epoch_history.csv"],
    )
    append_csv(DATA_DIR / "frozen_interventions.csv", diagnostic["interventions"], fields["frozen_interventions.csv"])
    for path, values in diagnostic["edge_selection"].items():
        modality, purpose = path.split("_", 1)
        append_csv(
            DATA_DIR / "edge_selection_diagnostics.csv",
            [{"dataset": dataset, "condition": condition, "seed": seed, "target_modality": modality,
              "message_purpose": purpose, "selection_enabled": values["selection_enabled"],
              "selection_status": values["selection_status"], **values["statistics"]}],
            fields["edge_selection_diagnostics.csv"],
        )
    for pair, values in diagnostic["role_distribution_comparisons"].items():
        append_csv(DATA_DIR / "role_distribution_diagnostics.csv", [{
            "dataset": dataset, "condition": condition, "seed": seed, "path_pair": pair,
            "status": values.get("comparison_type", "measured"), **values,
        }], fields["role_distribution_diagnostics.csv"])
    for modality, values in diagnostic["mechanism_details"].get("modalities", {}).items():
        converted = {
            "delta_intra_rms": values.get("delta_rms"),
            "delta_intra_hat_rms": values.get("delta_hat_rms"),
            "delta_intra_weighted_rms_by_expert": values.get("delta_weighted_rms_by_expert"),
            "cross_base_rms": values.get("cross_rms"),
            "cross_base_hat_rms": values.get("cross_hat_rms"),
            "cross_base_weighted_rms_by_expert": values.get("cross_weighted_rms_by_expert"),
            "delta_cross_rms": values.get("cross_delta_rms", 0.0 if condition != "C0" else None),
            "delta_cross_hat_rms": values.get("cross_delta_hat_rms", 0.0 if condition != "C0" else None),
            "delta_cross_weighted_rms_by_expert": values.get("cross_delta_weighted_rms_by_expert", [0.0] * 4 if condition != "C0" else None),
            "knn_rms": values.get("knn_rms"),
            "knn_hat_rms": values.get("knn_hat_rms"),
            "knn_weighted_rms_by_expert": values.get("knn_weighted_rms_by_expert"),
        }
        record = {key: value for key, value in values.items() if key not in {
            "delta_rms", "delta_hat_rms", "delta_weighted_rms_by_expert", "cross_rms", "cross_hat_rms",
            "cross_weighted_rms_by_expert", "knn_rms", "knn_hat_rms", "knn_weighted_rms_by_expert",
        }}
        append_csv(DATA_DIR / "message_scale_diagnostics.csv", [{
            "dataset": dataset, "condition": condition, "seed": seed, "modality": modality,
            **record, **converted,
        }], fields["message_scale_diagnostics.csv"])
        append_csv(DATA_DIR / "expert_router_diagnostics.csv", [{
            "dataset": dataset, "condition": condition, "seed": seed, "modality": modality,
            **{key: values.get(key) for key in (
                "expert_selection_share", "base_expert_selection_share", "dense_probability_mean", "router_entropy",
                "router_correction_rms", "router_top2_flip_fraction", "expert_coefficients", "expert_output_rms",
                "expert_output_rms_by_expert", "expert_output_pairwise_cosine",
            )},
        }], fields["expert_router_diagnostics.csv"])
    gradient_audit = read_json(DATA_DIR / "gradient_audit.json", {"runs": []})
    gradient_record = {
        "dataset": dataset, "condition": condition, "seed": seed,
        "gradients": diagnostic["gradient_diagnostics"], "parameter_update_norms": diagnostic["parameter_update_norms"],
        "initial_parameter_audit": run_metadata.get("initial_parameter_audit"),
        "final_parameter_audit": run_metadata.get("final_parameter_audit"),
    }
    gradient_key = (dataset, condition, int(seed))
    prior = [item for item in gradient_audit["runs"] if (item["dataset"], item["condition"], int(item["seed"])) != gradient_key]
    prior.append(gradient_record)
    gradient_audit["runs"] = prior
    write_json(DATA_DIR / "gradient_audit.json", gradient_audit)
    append_csv(DATA_DIR / "resource_profile.csv", [{
        "dataset": dataset, "condition": condition, "seed": seed,
        "wall_seconds": row["wall_seconds"], "epochs_completed": run_metadata.get("epochs_completed"),
        "mean_epoch_wall_seconds": run_metadata.get("mean_epoch_wall_seconds"),
        "peak_allocated_bytes": run_metadata.get("cuda_peak_allocated_bytes"),
        "peak_reserved_bytes": run_metadata.get("cuda_peak_reserved_bytes"),
        "cuda_device_name": run_metadata.get("cuda_device_name"),
    }], fields["resource_profile.csv"])


@torch.no_grad()
def diagnose_checkpoint(dataset: str, condition: str, seed: int, cache: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from src.data import load_mag_data
    from src.models import build_model

    run_dir = OUTPUT / "runs" / dataset / condition / f"seed_{seed}"
    best_path, f1_path = run_dir / "best.pt", run_dir / "best_f1.pt"
    metrics_path = run_dir / "run_metrics.json"
    if not all(path.is_file() for path in (best_path, f1_path, metrics_path)):
        raise FileNotFoundError(f"missing checkpoint or metrics for {dataset}/{condition}/seed_{seed}")
    cfg = build_config(dataset, condition, seed, device, cache)
    data = load_mag_data(cfg, "nc", seed)
    info = data_info(data)
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    payload_f1 = torch.load(f1_path, map_location="cpu", weights_only=False)
    for name, candidate in (("Best Accuracy", payload), ("Best Macro-F1", payload_f1)):
        if any(key.startswith("test_") for key in candidate.get("metrics", {})):
            raise RuntimeError(f"{name} checkpoint unexpectedly contains Test metrics")
        for state_name in ("model_state", "head_state"):
            state = candidate[state_name]
            if any(not torch.isfinite(value).all() for value in state.values() if torch.is_floating_point(value)):
                raise RuntimeError(f"non-finite {state_name} tensor in {name} checkpoint")
    if payload_f1.get("selection") != "best_val_macro_f1_within_accuracy_stopping_horizon":
        raise RuntimeError("Best Macro-F1 checkpoint selection rule differs from the frozen NC protocol")
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
    z, _, _, _, model_info = model(x, edge, return_details=True)
    if not torch.isfinite(z).all():
        raise RuntimeError("reloaded checkpoint produces non-finite embeddings")
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels

    labels = _resolve_nc_eval_labels(data, include_test=False)
    val = _evaluate_split(head, z, data.y, data.val_idx, z.device, 4096, labels)
    val_logits = head(z[data.val_idx.to(device)]).detach()
    val["cross_entropy"] = float(torch.nn.functional.cross_entropy(val_logits, data.y[data.val_idx].to(device)).cpu())
    if abs(val["acc"] - float(payload["metrics"]["val_acc"])) > 1e-8:
        raise RuntimeError(f"reloaded Best Accuracy mismatch: {val['acc']} vs {payload['metrics']['val_acc']}")
    if abs(val["macro_f1"] - float(payload["metrics"]["val_macro_f1"])) > 1e-8:
        raise RuntimeError("reloaded Best Accuracy checkpoint does not reproduce Validation Macro-F1")
    history = json.loads(metrics_path.read_text(encoding="utf-8"))["runs"][0]["metadata"].get("epoch_history", [])
    if not history:
        raise RuntimeError("Train/Validation epoch history is absent")
    history_keys = ("train_loss", "val_acc", "val_macro_f1", "train_ce", "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce")
    if any(not all(value is not None and math.isfinite(float(value)) for value in (item.get(key) for key in history_keys)) for item in history):
        raise RuntimeError("non-finite Train/Validation metric in epoch history")
    if len(history) != int(json.loads(metrics_path.read_text(encoding="utf-8"))["runs"][0]["metadata"]["epochs_completed"]):
        raise RuntimeError("epoch history length differs from the training epoch count")
    f1_model = build_model(cfg, info)
    f1_model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    f1_model.load_state_dict(payload_f1["model_state"], strict=True)
    f1_model.to(device).eval()
    f1_head = torch.nn.Linear(model.out_dim, info["num_classes"]).to(device)
    f1_head.load_state_dict(payload_f1["head_state"], strict=True)
    z_f1 = f1_model(x, edge)[0]
    f1_metrics = _evaluate_split(f1_head, z_f1, data.y, data.val_idx, z_f1.device, 4096, labels)
    f1_logits = f1_head(z_f1[data.val_idx.to(device)])
    f1_metrics["cross_entropy"] = float(torch.nn.functional.cross_entropy(f1_logits, data.y[data.val_idx].to(device)).cpu())
    if abs(f1_metrics["acc"] - float(payload_f1["metrics"]["val_acc"])) > 1e-8:
        raise RuntimeError("Best Macro-F1 checkpoint does not reproduce Validation Accuracy")
    if abs(f1_metrics["macro_f1"] - float(payload_f1["metrics"]["val_macro_f1"])) > 1e-8:
        raise RuntimeError("Best Macro-F1 checkpoint does not reproduce Validation Macro-F1")

    # Cross-use the V2.1 path stats with a fixed-P baseline for disabled paths.
    edge_details = model_info.get("details", {}).get("edge_selection", {})
    baseline = _baseline_selection(model, x, edge)
    selection_records = {}
    for modality in ("text", "visual"):
        for purpose in ("intra", "cross"):
            key = f"{modality}_{purpose}"
            enabled = (purpose == "intra" and condition in ("C1", "C2")) or (purpose == "cross" and condition == "C2")
            selection_records[key] = {
                "selection_enabled": enabled,
                "selection_status": "measured" if enabled else "not_applicable_fixed_physical_reference",
                "statistics": edge_details.get(key, baseline[key]),
            }
    role_comparisons = model_info.get("details", {}).get("role_distribution_comparisons", {})
    if condition == "C0":
        role_comparisons = {}
    else:
        from src.models.rmcge_v21 import conserve_receiver_mass

        split_inputs = model._split_modalities(x)
        priors = {modality: model.projectors[modality](split_inputs[modality]) for modality in ("text", "visual")}
        src, dst, physical, active, _ = model._normalized_operator(edge, x.size(0), priors["text"].dtype)
        degree = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.float32, device=x.device))
        for modality_index, modality in enumerate(("text", "visual")):
            other = "visual" if modality == "text" else "text"
            messages = [("intra", priors[modality])]
            if condition == "C2":
                projected_sender = model.cross_projections[
                    "visual_to_text" if modality == "text" else "text_to_visual"
                ](priors[other])
                messages.append(("cross", projected_sender))
            receiver_mean = priors[modality][active].mean(dim=0, keepdim=True).expand_as(priors[modality])
            for purpose, sender_message in messages:
                purpose_index = 0 if purpose == "intra" else 1
                actual_score = model._score_message_edges(
                    priors[modality], sender_message, src, dst, degree, modality_index, purpose_index
                )
                mean_query_score = model._score_message_edges(
                    receiver_mean, sender_message, src, dst, degree, modality_index, purpose_index
                )
                actual_selected = conserve_receiver_mass(physical, actual_score, dst, x.size(0))
                mean_query_selected = conserve_receiver_mass(physical, mean_query_score, dst, x.size(0))
                role_comparisons[f"{modality}_{purpose}__receiver_global_mean_query"] = {
                    **model._compare_path_distributions(actual_selected, mean_query_selected, dst, degree.long()),
                    "comparison_type": "receiver_query_ablation",
                }

    train_metrics = _evaluate_split(head, z, data.y, data.train_idx, z.device, 4096, labels)
    train_logits = head(z[data.train_idx.to(device)])
    train_metrics["cross_entropy"] = float(torch.nn.functional.cross_entropy(train_logits, data.y[data.train_idx].to(device)).cpu())
    model.set_interventions()
    interventions = []
    settings = (
        ("intra_selection_forced_physical", {"intra_selection_off": True}, condition in ("C1", "C2")),
        ("cross_selection_forced_physical", {"cross_selection_off": True}, condition == "C2"),
        ("both_selections_forced_physical", {"intra_selection_off": True, "cross_selection_off": True}, condition in ("C1", "C2")),
        ("cross_base_context_off", {"cross_off": True}, True),
        ("knn_context_off", {"knn_off": True}, True),
    )
    for name, setting, applicable in settings:
        if not applicable:
            interventions.append({"dataset": dataset, "condition": condition, "seed": seed, "intervention": name, "status": "not_applicable"})
            continue
        model.set_interventions(**setting)
        altered_z = model(x, edge)[0]
        altered_metrics = _evaluate_split(head, altered_z, data.y, data.val_idx, altered_z.device, 4096, labels)
        altered_logits = head(altered_z[data.val_idx.to(device)])
        interventions.append({
            "dataset": dataset, "condition": condition, "seed": seed, "intervention": name, "status": "measured",
            "val_acc": altered_metrics["acc"], "val_macro_f1": altered_metrics["macro_f1"],
            "val_acc_delta": altered_metrics["acc"] - val["acc"],
            "val_macro_f1_delta": altered_metrics["macro_f1"] - val["macro_f1"],
            "logit_delta_rms": float((altered_logits - val_logits).float().square().mean().sqrt().cpu()),
            "prediction_flip_fraction": float((altered_logits.argmax(-1) != val_logits.argmax(-1)).float().mean().cpu()),
        })
    model.set_interventions()
    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_record = metrics_json["runs"][0]
    run_metrics = run_record["metrics"]
    if any(key.startswith("test_") for key in run_metrics):
        raise RuntimeError("run metrics contain Test fields despite evaluate_test=false")
    metadata = run_record["metadata"]
    gradient = metadata.get("gradient_diagnostics") or {}
    for name, value in gradient.items():
        if value.get("gradient_present") and not value.get("gradient_finite", False):
            raise RuntimeError(f"non-finite gradient recorded for {name}")
    update_norms = _update_norms(dataset, condition, seed, cache, payload, info)
    required = ["cross_projections", "node_router", "experts", "fusion_linear1", "projectors"]
    if condition in ("C1", "C2"):
        required.append("purpose_edge_scorer")
    missing = [name for name in required if name not in update_norms or update_norms[name]["l2_update_norm"] <= 0]
    if missing:
        raise RuntimeError(f"required trainable groups had no checkpoint update: {missing}")
    if condition in ("C1", "C2") and not gradient.get("purpose_edge_scorer.2.weight", {}).get("gradient_nonzero", False):
        raise RuntimeError("purpose-conditioned Edge Scorer output layer has no recorded task gradient")
    best_row = _metrics_from_history(history, payload["epoch"])
    best_f1_row = _metrics_from_history(history, payload_f1["epoch"])
    diagnostic = {
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "best_accuracy_epoch": int(payload["epoch"]),
        "best_f1_epoch": int(payload_f1["epoch"]),
        "val_metrics": val,
        "best_f1_metrics": f1_metrics,
        "train_metrics_at_best_accuracy": train_metrics,
        "best_accuracy_epoch_metrics": best_row,
        "best_f1_epoch_metrics": best_f1_row,
        "history": history,
        "gradient_diagnostics": gradient,
        "parameter_update_norms": update_norms,
        "mechanism_details": model_info.get("details", {}),
        "edge_selection": selection_records,
        "role_distribution_comparisons": role_comparisons,
        "interventions": interventions,
        "test_metrics_present": False,
        "test_label_values_indexed": False,
    }
    write_json(run_dir / "checkpoint_diagnostics.json", diagnostic)
    del model, f1_model, head, f1_head, data, z, z_f1, val_logits, f1_logits, payload, payload_f1
    return diagnostic


def _row_from_diagnostic(row: dict[str, Any], diagnostic: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    best = diagnostic["best_accuracy_epoch_metrics"]
    best_f1 = diagnostic["best_f1_epoch_metrics"]
    val, train = diagnostic["val_metrics"], diagnostic["train_metrics_at_best_accuracy"]
    return {
        **row,
        "status": "completed_and_verified",
        "best_accuracy_epoch": diagnostic["best_accuracy_epoch"],
        "best_f1_epoch": diagnostic["best_f1_epoch"],
        "val_acc": val["acc"],
        "val_macro_f1": val["macro_f1"],
        "val_ce": val["cross_entropy"],
        "best_f1_val_acc": diagnostic["best_f1_metrics"]["acc"],
        "best_f1_val_macro_f1": diagnostic["best_f1_metrics"]["macro_f1"],
        "best_f1_val_ce": diagnostic["best_f1_metrics"]["cross_entropy"],
        "train_acc": train["acc"],
        "train_macro_f1": train["macro_f1"],
        "train_ce": train["cross_entropy"],
        "train_val_acc_gap": train["acc"] - val["acc"],
        "train_val_macro_f1_gap": train["macro_f1"] - val["macro_f1"],
        "model_parameters": metadata.get("model_parameters"),
        "classifier_parameters": metadata.get("classifier_parameters"),
        "initial_model_sha256": (metadata.get("initial_parameter_audit") or {}).get("model_state_sha256", ""),
        "initial_classifier_sha256": (metadata.get("initial_parameter_audit") or {}).get("classifier_state_sha256", ""),
        "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
        "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
        "diagnostics_path": str((OUTPUT / "runs" / row["dataset"] / row["condition"] / f"seed_{row['seed']}" / "checkpoint_diagnostics.json").relative_to(ROOT)),
        "test_metrics_present": False,
    }


def campaign(seeds: tuple[int, ...], device: str) -> list[dict[str, Any]]:
    initialize_artifacts()
    caches = verify_cache_entries()
    manifest = read_json(DATA_DIR / "manifest.json", {})
    if manifest.get("implementation_freeze_sha") != git("rev-parse", "HEAD"):
        raise RuntimeError("formal runs require HEAD to equal the recorded Implementation Freeze SHA")
    rows = []
    for seed in seeds:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                existing = next((item for item in manifest["planned_runs"] if item["dataset"] == dataset and item["condition"] == condition and int(item["seed"]) == seed), {})
                if existing.get("status") == "completed_and_verified":
                    rows.append(existing)
                    continue
                physical_gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(",")[0])
                assert_gpu_capacity(physical_gpu)
                try:
                    row = run_one(dataset, condition, seed, device, caches[dataset])
                    diagnosis = diagnose_checkpoint(dataset, condition, seed, caches[dataset], device)
                    metrics_json = json.loads((ROOT / row["run_metrics_path"]).read_text(encoding="utf-8"))
                    metadata = metrics_json["runs"][0]["metadata"]
                    row = _row_from_diagnostic(row, diagnosis, metadata)
                    _append_run_evidence(row, diagnosis, metadata)
                    _update_manifest_run(row)
                    rows.append(row)
                except Exception as exc:
                    failure = {
                        "dataset": dataset, "condition": condition, "seed": seed, "status": "failed",
                        "recorded_at_utc": now(), "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(), "existing_artifacts_retained": True,
                    }
                    append_failure(failure)
                    failed = {"dataset": dataset, "condition": condition, "seed": seed, "status": "failed", "error": failure["error"]}
                    _update_manifest_run(failed)
                    append_csv(DATA_DIR / "runs.csv", [failed], CSV_FIELDS["runs.csv"])
                    raise
    return rows


def verify_batch(seed: int) -> dict[str, Any]:
    manifest = read_json(DATA_DIR / "manifest.json", {})
    rows = [item for item in manifest.get("planned_runs", []) if int(item["seed"]) == seed]
    expected = {(dataset, condition) for dataset in DATASETS for condition in CONDITIONS}
    found = {(item["dataset"], item["condition"]) for item in rows if item.get("status") == "completed_and_verified"}
    if found != expected:
        raise RuntimeError(f"seed-{seed} verification incomplete: {sorted(found)} != {sorted(expected)}")
    if any(item.get("test_metrics_present") for item in rows):
        raise RuntimeError("Test metrics found in a formal run")
    result = {"seed": seed, "verified_runs": len(found), "histories_finite": True, "test_evaluation": False, "verified_at_utc": now()}
    checks = read_json(DATA_DIR / "implementation_checks.json", {})
    checks.setdefault("seed_batch_verifications", []).append(result)
    write_json(DATA_DIR / "implementation_checks.json", checks)
    manifest["status"] = f"seed_{seed}_verified"
    if seed == 44:
        manifest["status"] = "all_27_runs_verified"
    manifest["updated_at_utc"] = now()
    write_json(DATA_DIR / "manifest.json", manifest)
    return result


def finalize_evidence() -> dict[str, Any]:
    from scripts.analyze_rmcge_v21 import analyze

    analysis = analyze()
    manifest = read_json(DATA_DIR / "manifest.json", {})
    manifest["status"] = "all_27_runs_verified"
    manifest["evidence_analysis"] = analysis
    manifest["updated_at_utc"] = now()
    write_json(DATA_DIR / "manifest.json", manifest)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run-batch", "verify-batch", "run-all", "finalize"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--gpu", type=int, default=1, help="physical GPU index to isolate")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    if args.mode == "preflight":
        try:
            result = preflight(args.gpu)
        except Exception as exc:
            initialize_artifacts()
            append_failure({
                "phase": "preflight", "recorded_at_utc": now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "existing_artifacts_retained": True,
            })
            raise
        print(json.dumps(result, indent=2))
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
        print(json.dumps({"completed_rows": len(rows), "seed": args.seed}, indent=2))
        return
    if args.mode == "run-all":
        for seed in SEEDS:
            rows = campaign((seed,), "cuda:0")
            print(json.dumps({"completed_rows": len(rows), "seed": seed}, indent=2), flush=True)
            verify_batch(seed)
        return
    print(json.dumps(finalize_evidence(), indent=2))


if __name__ == "__main__":
    main()
