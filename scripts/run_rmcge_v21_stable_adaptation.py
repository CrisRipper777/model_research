#!/usr/bin/env python3
"""Warm-start RM-CGE V2.1 W0/W1/W2 from matched historical C0 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v21 import (  # noqa: E402
    DATASETS,
    SEEDS,
    build_config,
    data_info,
    sha256_file,
    tensor_fingerprint,
    verify_cache_entries,
    write_json,
)
from src.research.rmcge_v21_stable_adaptation import (  # noqa: E402
    INITIAL_LOGIT_ATOL,
    SELECTOR_PREFIXES,
    apply_joint_trainability,
    apply_stage_trainability,
    compare_logits,
    finite_metrics,
    grad_norms_by_group,
    is_selector_name,
    is_stage1_selector_name,
    max_state_difference,
    metric_bundle,
    make_joint_optimizer,
    make_stage1_optimizer,
    module_sha256,
    parameter_snapshot,
    rms_update,
    selector_state_sha256,
    state_sha256,
)
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402

STUDY = ROOT / "research/rmcge_v21_stable_adaptation"
DATA_DIR = STUDY / "data"
OUTPUT = ROOT / "outputs/rmcge_v21_stable_adaptation"
OLD_RUNS = ROOT / "outputs/rmcge_v21/runs"
CONDITION_MAP = {"W0": "C0", "W1": "C1", "W2": "C2"}
WARM_EPOCHS = 12
JOINT_EPOCHS = 90
STAGE1_LR = 1.0e-3
JOINT_COMMON_LR = 1.0e-4
JOINT_SELECTOR_LR = 5.0e-4
JOINT_CLASSIFIER_LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
PATIENCE = 30
MIN_EPOCH = 20
MIN_DELTA = 1.0e-4
EVAL_BATCH_SIZE = 4096

RUN_FIELDS = [
    "dataset", "condition", "seed", "status", "source_checkpoint_sha256", "source_epoch",
    "selected_stage", "best_accuracy_epoch", "best_f1_epoch", "best_f1_stage", "val_acc", "val_macro_f1", "val_ce",
    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "t0_val_acc", "t0_val_macro_f1",
    "warm_end_val_acc", "warm_end_val_macro_f1", "train_acc", "train_macro_f1", "train_ce",
    "train_val_acc_gap", "train_val_macro_f1_gap", "epochs_completed", "joint_updates",
    "stage1_epochs", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "gpu_name",
    "model_parameters", "classifier_parameters", "result_path", "best_accuracy_checkpoint",
    "best_f1_checkpoint", "test_evaluation", "error",
]
HISTORY_FIELDS = [
    "dataset", "condition", "seed", "stage", "epoch", "train_ce", "train_objective",
    "train_accuracy", "train_macro_f1", "val_ce", "val_accuracy", "val_macro_f1",
    "selector_grad_norm", "scorer_grad_norm", "projector_grad_norm", "expert_grad_norm",
    "router_grad_norm", "fusion_grad_norm", "other_grad_norm", "selector_update_rms",
    "common_update_rms", "learning_rate_common", "learning_rate_selector", "learning_rate_classifier",
    "classifier_grad_norm", "classifier_update_rms", "wall_seconds",
    "cross_delta_coefficient_grad_norm", "cross_delta_coefficient_update_rms",
]
SELECTOR_FIELDS = [
    "dataset", "condition", "seed", "stage", "epoch", "path", "status", "edge_count",
    "score_abs_mean", "score_abs_ge_0_99_fraction", "relative_weight_mean_abs_deviation",
    "receiver_row_mass_error_max", "js_selected_physical_mean_degree_ge2",
    "effective_neighbors_mean_degree_ge2", "within_receiver_relative_weight_std_mean_degree_ge2",
    "neighbor_entropy_physical_mean_degree_ge2", "neighbor_entropy_selected_mean_degree_ge2",
    "distribution_pair", "distribution_pair_js_degree_ge2", "distribution_pair_edge_abs_difference",
    "gradient_norm", "parameter_update_rms",
]
EXPERT_FIELDS = [
    "dataset", "condition", "seed", "stage", "epoch", "modality", "intrinsic_rms", "u1_rms",
    "physical_profile_rms_by_expert", "delta_intra_rms", "delta_intra_hat_rms",
    "delta_intra_weighted_rms_by_expert", "cross_base_rms", "cross_base_hat_rms",
    "cross_base_weighted_rms_by_expert", "delta_cross_rms", "delta_cross_hat_rms",
    "delta_cross_weighted_rms_by_expert", "knn_rms", "knn_hat_rms", "knn_weighted_rms_by_expert",
    "expert_output_rms", "expert_output_rms_by_expert", "expert_output_pairwise_cosine",
    "router_correction_rms", "router_entropy", "dense_probability_mean", "expert_selection_share",
    "base_expert_selection_share", "router_top2_flip_fraction", "expert_coefficients",
]
TRADEOFF_FIELDS = [
    "dataset", "condition", "seed", "checkpoint", "selected_stage", "epoch", "train_accuracy",
    "train_macro_f1", "train_ce", "val_accuracy", "val_macro_f1", "val_ce", "train_val_acc_gap",
    "train_val_macro_f1_gap", "source_checkpoint_sha256", "checkpoint_sha256",
]
RESOURCE_FIELDS = [
    "dataset", "condition", "seed", "wall_seconds", "stage1_seconds", "joint_seconds",
    "epochs_completed", "joint_updates", "peak_allocated_bytes", "peak_reserved_bytes", "gpu_name",
]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default), encoding="utf-8")
    temp.replace(path)


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                out[field] = value
            writer.writerow(out)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checkpoint_source(dataset: str, seed: int) -> dict[str, Any]:
    manifest = json.loads((DATA_DIR / "source_checkpoints.json").read_text(encoding="utf-8"))
    if manifest.get("source_commit") != "218d6d230e5afc16e259c09777f6363de5e0c6cf":
        raise RuntimeError(f"historical V2.1 checkpoint manifest source commit mismatch: {manifest.get('source_commit')}")
    rows = [
        item for item in manifest["records"]
        if item["dataset"] == dataset and item["condition"] == "C0" and item["seed"] == seed
        and item["checkpoint"] == "best_accuracy"
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one manifest C0 Best Accuracy source for {dataset}/seed_{seed}, found {len(rows)}")
    record = rows[0]
    path = ROOT / record["path"]
    if not path.is_file() or sha256_file(path) != record["checkpoint_sha256"]:
        raise RuntimeError(f"historical C0 checkpoint missing or hash mismatch: {path}")
    config_path = ROOT / record["resolved_config_path"]
    if not config_path.is_file() or sha256_file(config_path) != record["resolved_config_sha256"]:
        raise RuntimeError(f"historical resolved config missing or hash mismatch: {config_path}")
    if record["run_metrics_sha256"] != sha256_file(OLD_RUNS / dataset / "C0" / f"seed_{seed}" / "run_metrics.json"):
        raise RuntimeError(f"historical run metrics hash mismatch for {dataset}/seed_{seed}")
    source_cfg = OmegaConf.load(config_path)
    if str(source_cfg.model.condition).upper() != "C0" or str(source_cfg.dataset.name) != dataset:
        raise RuntimeError(f"historical resolved config identifies a different model or data set: {config_path}")
    if bool(source_cfg.task.evaluate_test) or float(source_cfg.model.dropout) != 0.2:
        raise RuntimeError(f"historical source config changed Test/dropout protocol: {config_path}")
    return {**record, "absolute_path": str(path)}


def _load_payload(source: dict[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    payload = torch.load(source["absolute_path"], map_location="cpu", weights_only=False)
    provenance = payload.get("run_metadata", {}).get("study_provenance", {})
    expected_provenance = {
        "study": "RM-CGE V2.1", "dataset": dataset, "condition": "C0", "seed": seed,
        "official_train_val_split": True, "test_evaluation": False,
    }
    if payload.get("task") != "nc" or payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise RuntimeError(f"source task/protocol mismatch: {dataset}/seed_{seed}")
    if payload.get("selection") != "best_val_accuracy" or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"source checkpoint selection/seed mismatch: {dataset}/seed_{seed}")
    if provenance != expected_provenance:
        raise RuntimeError(f"source provenance mismatch: {provenance}")
    if any(key.startswith("test_") for key in payload.get("metrics", {})):
        raise RuntimeError("historical source checkpoint has Test metrics")
    if any(key in payload for key in ("optimizer_state", "scheduler_state", "optimizer", "scheduler")):
        raise RuntimeError("unexpected optimizer/scheduler state found; warm-start reset assumption must be revisited")
    return payload


def _construct_model(cfg, info, cache, seed: int, payload: dict[str, Any]):
    from src.models import build_model

    set_seed(seed)
    model = build_model(cfg, info)
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=cfg.dataset.name,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    base_state = payload["model_state"]
    if cfg.model.condition == "C0":
        model.load_state_dict(base_state, strict=True)
        missing, unexpected = [], []
    else:
        incompatible = model.load_state_dict(base_state, strict=False)
        missing, unexpected = list(incompatible.missing_keys), list(incompatible.unexpected_keys)
        expected_missing = [name for name in model.state_dict() if is_selector_name(name)]
        if set(missing) != set(expected_missing) or unexpected:
            raise RuntimeError(
                f"unexpected warm-start state mapping for {cfg.dataset.name}/{cfg.model.condition}: "
                f"missing={missing}, expected_added={expected_missing}, unexpected={unexpected}"
            )
    head = torch.nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"]))
    head.load_state_dict(payload["head_state"], strict=True)
    return model, head, {"missing_keys": missing, "unexpected_keys": unexpected}


def _assert_state_equal(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], *, label: str) -> float:
    if set(reference) != set(candidate):
        raise RuntimeError(f"{label} key sets differ")
    delta = max_state_difference(reference, candidate)
    if delta != 0.0:
        raise RuntimeError(f"{label} state differs; max absolute difference={delta}")
    return delta


@torch.no_grad()
def _eval_snapshot(model, classifier, data, device, eval_labels: list[int], *, details: bool = True):
    model.eval()
    classifier.eval()
    x, edge = data.x.to(device), data.edge_index.to(device)
    z, _, _, aux_loss, info = model(x, edge, return_details=details)
    if not torch.isfinite(z).all() or not torch.isfinite(aux_loss).all():
        raise FloatingPointError("nonfinite embedding or balance loss")
    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    y_train = data.y[data.train_idx].to(device)
    y_val = data.y[data.val_idx].to(device)
    train = metric_bundle(classifier(z[train_idx]), y_train, eval_labels)
    validation = metric_bundle(classifier(z[val_idx]), y_val, eval_labels)
    if not finite_metrics({"train": train, "validation": validation}):
        raise FloatingPointError("nonfinite evaluation metric")
    return {"train": train, "validation": validation}, info, z[val_idx].detach().float().cpu()


def _cpu_states(model, classifier) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
    )


def _candidate(stage: str, epoch: int, metrics: dict[str, Any], model, classifier) -> dict[str, Any]:
    model_state, head_state = _cpu_states(model, classifier)
    return {"stage": stage, "epoch": epoch, "metrics": metrics, "model_state": model_state, "head_state": head_state}


def _metric_match(actual: dict[str, Any], expected: dict[str, Any], label: str, tolerance: float = 2.0e-5) -> None:
    for metric in ("accuracy", "macro_f1", "cross_entropy"):
        expected_value = expected.get({"accuracy": "val_acc", "macro_f1": "val_macro_f1", "cross_entropy": "val_ce"}[metric])
        if expected_value is None:
            continue
        difference = abs(float(actual[metric]) - float(expected_value))
        if difference > tolerance:
            raise RuntimeError(f"{label} {metric} differs by {difference}: {actual[metric]} vs {expected_value}")


@torch.no_grad()
def audit_initial_triplet(dataset: str, seed: int, cache: dict[str, Any], *, device: str) -> dict[str, Any]:
    from src.data import load_mag_data

    source = checkpoint_source(dataset, seed)
    payload = _load_payload(source, dataset, seed)
    for key, expected in (
        ("semantic_feature_fingerprint", cache["feature_fingerprint"]),
        ("semantic_candidate_fingerprint", cache["candidate_fingerprint"]),
        ("semantic_v7b_fingerprint", cache["v7b_fingerprint"]),
    ):
        if payload["run_metadata"].get(key) != expected:
            raise RuntimeError(f"source cache fingerprint mismatch: {dataset}/seed_{seed}/{key}")
    cfg_data = build_config(dataset, "C0", seed, device, cache)
    data = load_mag_data(cfg_data, "nc", seed)
    if payload.get("data_info") != data_info(data):
        raise RuntimeError(f"source checkpoint data_info does not match loaded data: {dataset}/seed_{seed}")
    eval_labels = _resolve_nc_eval_labels(data, include_test=False)
    x, edge = data.x.to(device), data.edge_index.to(device)
    model_records: dict[str, Any] = {}
    loaded: dict[str, tuple[torch.nn.Module, torch.nn.Module]] = {}
    logits: dict[str, torch.Tensor] = {}
    public_states: dict[str, dict[str, torch.Tensor]] = {}
    selector_states: dict[str, dict[str, torch.Tensor]] = {}
    common_match = True
    classifier_match = True

    try:
        for warm_condition, model_condition in CONDITION_MAP.items():
            cfg = build_config(dataset, model_condition, seed, device, cache)
            model, classifier, key_audit = _construct_model(cfg, data_info(data), cache, seed, payload)
            if warm_condition != "W0":
                model.eval()
            model.to(device)
            classifier.to(device)
            model.eval()
            classifier.eval()
            state = model.state_dict()
            public = {name: tensor.detach().cpu().clone() for name, tensor in state.items() if not is_selector_name(name)}
            public_states[warm_condition] = public
            if warm_condition in ("W1", "W2"):
                selector_states[warm_condition] = {
                    name: tensor.detach().cpu().clone() for name, tensor in state.items() if is_selector_name(name)
                }
            rng_cpu_before = torch.get_rng_state().clone()
            rng_cuda_before = torch.cuda.get_rng_state(torch.device(device)).clone()
            metrics, info, val_logits = _eval_snapshot(model, classifier, data, device, eval_labels, details=True)
            rng_cpu_preserved = torch.equal(rng_cpu_before, torch.get_rng_state())
            rng_cuda_preserved = torch.equal(rng_cuda_before, torch.cuda.get_rng_state(torch.device(device)))
            if not (rng_cpu_preserved and rng_cuda_preserved):
                raise RuntimeError(f"t0 evaluation changed training RNG state for {warm_condition}")
            _metric_match(metrics["validation"], payload["metrics"], f"{dataset}/seed_{seed}/{warm_condition} t0")
            logits[warm_condition] = val_logits
            classifier_sha = module_sha256(classifier)
            model_records[warm_condition] = {
                "model_condition": model_condition,
                "state_mapping": key_audit,
                "public_model_sha256": state_sha256(public),
                "classifier_sha256": classifier_sha,
                "selector_sha256": selector_state_sha256(model) if warm_condition != "W0" else None,
                "t0_metrics": metrics,
                "t0_logits_vs_c0": None,
                "cpu_rng_preserved": rng_cpu_preserved,
                "cuda_rng_preserved": rng_cuda_preserved,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "initial_edge_selection": info.get("details", {}).get("edge_selection", {}),
            }
            loaded[warm_condition] = (model, classifier)

        for condition in ("W1", "W2"):
            common_match &= public_states["W0"].keys() == public_states[condition].keys()
            _assert_state_equal(public_states["W0"], public_states[condition], label=f"{warm_condition_label(condition)} inherited public state")
            classifier_match &= model_records["W0"]["classifier_sha256"] == model_records[condition]["classifier_sha256"]
        if not common_match or not classifier_match:
            raise RuntimeError("inherited model or classifier weights are not exactly matched")
        selector_pair = _assert_state_equal(selector_states["W1"], selector_states["W2"], label="W1/W2 paired selector initialization")

        reference_logits = logits["W0"]
        logit_checks = {}
        for condition in ("W1", "W2"):
            comparison = compare_logits(reference_logits, logits[condition])
            model_records[condition]["t0_logits_vs_c0"] = comparison
            logit_checks[condition] = comparison
            if not comparison["allclose"]:
                raise RuntimeError(f"t0 Validation logits differ beyond {INITIAL_LOGIT_ATOL}: {dataset}/seed_{seed}/{condition}: {comparison}")
        c1_c2 = compare_logits(logits["W1"], logits["W2"])
        if not c1_c2["allclose"]:
            raise RuntimeError(f"W1/W2 t0 logits differ: {dataset}/seed_{seed}: {c1_c2}")

        selector_zero: dict[str, Any] = {}
        for condition in ("W1", "W2"):
            model = loaded[condition][0]
            scorer_weight = model.purpose_edge_scorer[-1].weight.detach()
            scorer_bias = model.purpose_edge_scorer[-1].bias.detach()
            if torch.count_nonzero(scorer_weight).item() or torch.count_nonzero(scorer_bias).item():
                raise RuntimeError(f"selector output layer is not initialized to zero: {dataset}/seed_{seed}/{condition}")
            expected_paths = ("text_intra", "visual_intra") + (("text_cross", "visual_cross") if condition == "W2" else ())
            initial_paths = model_records[condition]["initial_edge_selection"]
            if set(initial_paths) != set(expected_paths):
                raise RuntimeError(f"unexpected initial selector paths for {condition}: {sorted(initial_paths)}")
            for path_name, diagnostic in initial_paths.items():
                if float(diagnostic.get("score_abs_mean", float("inf"))) > 1.0e-12:
                    raise RuntimeError(f"selector score is nonzero at t0: {condition}/{path_name}")
            selector_zero[condition] = {
                "scorer_last_weight_zero": True,
                "scorer_last_bias_zero": True,
                "score_abs_mean_by_path": {key: value.get("score_abs_mean") for key, value in initial_paths.items()},
                "logit_max_abs_from_c0": logit_checks[condition]["max_abs"],
                "zero_effect_within_tolerance": logit_checks[condition]["allclose"],
            }

        data_fingerprints = {
            "feature_sha256": tensor_fingerprint(data.x),
            "physical_edge_index_sha256": tensor_fingerprint(data.edge_index),
            "train_index_sha256": tensor_fingerprint(data.train_idx),
            "validation_index_sha256": tensor_fingerprint(data.val_idx),
            "train_labels_sha256": tensor_fingerprint(data.y[data.train_idx]),
            "validation_labels_sha256": tensor_fingerprint(data.y[data.val_idx]),
            "semantic_cache_sha256": sha256_file(Path(cache["cache_path_abs"])),
            "semantic_feature_fingerprint": cache["feature_fingerprint"],
            "semantic_candidate_fingerprint": cache["candidate_fingerprint"],
            "semantic_v7b_fingerprint": cache["v7b_fingerprint"],
            "official_test_index_values_accessed": False,
            "test_label_values_indexed": False,
        }
        result = {
            "dataset": dataset, "seed": seed, "source_checkpoint": source["path"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_checkpoint_bytes": source["checkpoint_bytes"],
            "source_best_accuracy_epoch": source["epoch"],
            "source_optimizer_state_present": False,
            "source_scheduler_state_present": False,
            "new_adamw_state_starts_empty": True,
            "data": data_fingerprints,
            "conditions": model_records,
            "public_initial_state_exact": common_match,
            "classifier_initial_state_exact": classifier_match,
            "new_w1_w2_parameter_state_exact": True,
            "selector_state_max_abs_difference": selector_pair,
            "selector_zero_effect": selector_zero,
            "t0_logit_tolerance": {"atol": INITIAL_LOGIT_ATOL, "rtol": INITIAL_LOGIT_ATOL},
            "t0_logits": {"W1_vs_W0": logit_checks["W1"], "W2_vs_W0": logit_checks["W2"], "W1_vs_W2": c1_c2},
            "test_evaluation": False,
            "status": "passed",
        }
        return result
    finally:
        for model, classifier in loaded.values():
            del model, classifier
        del data
        torch.cuda.empty_cache()


def warm_condition_label(condition: str) -> str:
    return {"W1": "W1", "W2": "W2"}.get(condition, condition)


def append_diagnostics(
    selector_rows: list[dict[str, Any]], expert_rows: list[dict[str, Any]],
    *, dataset: str, condition: str, seed: int, stage: str, epoch: int,
    info: dict[str, Any], selector_grads: dict[str, float] | None = None,
    selector_update: float | None = None,
) -> None:
    details = info.get("details", {})
    edge_selection = details.get("edge_selection", {})
    role_pairs = details.get("role_distribution_comparisons", {})
    if not edge_selection:
        selector_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "stage": stage, "epoch": epoch,
            "path": "all_selectors", "status": "not_applicable_baseline",
            "gradient_norm": (selector_grads or {}).get("selector"), "parameter_update_rms": selector_update,
        })
    for path, item in edge_selection.items():
        selector_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "stage": stage, "epoch": epoch,
            "path": path, "status": "measured", "edge_count": item.get("edge_count"),
            "score_abs_mean": item.get("score_abs_mean"),
            "score_abs_ge_0_99_fraction": item.get("score_abs_ge_0_99_fraction"),
            "relative_weight_mean_abs_deviation": item.get("relative_weight_mean_abs_deviation"),
            "receiver_row_mass_error_max": item.get("receiver_row_mass_error_max"),
            "js_selected_physical_mean_degree_ge2": item.get("js_selected_physical_mean_degree_ge2"),
            "effective_neighbors_mean_degree_ge2": item.get("effective_neighbors_mean_degree_ge2"),
            "within_receiver_relative_weight_std_mean_degree_ge2": item.get("within_receiver_relative_weight_std_mean_degree_ge2"),
            "neighbor_entropy_physical_mean_degree_ge2": item.get("neighbor_entropy_physical_mean_degree_ge2"),
            "neighbor_entropy_selected_mean_degree_ge2": item.get("neighbor_entropy_selected_mean_degree_ge2"),
            "gradient_norm": (selector_grads or {}).get("selector"), "parameter_update_rms": selector_update,
        })
    for pair, item in role_pairs.items():
        selector_rows.append({
            "dataset": dataset, "condition": condition, "seed": seed, "stage": stage, "epoch": epoch,
            "path": "distribution_pair", "status": item.get("comparison_type", "measured"),
            "distribution_pair": pair, "distribution_pair_js_degree_ge2": item.get("receiver_js_mean_degree_ge2"),
            "distribution_pair_edge_abs_difference": item.get("edge_probability_mean_abs_difference"),
            "gradient_norm": (selector_grads or {}).get("selector"), "parameter_update_rms": selector_update,
        })
    for modality, item in details.get("modalities", {}).items():
        expert_rows.append({"dataset": dataset, "condition": condition, "seed": seed, "stage": stage, "epoch": epoch, "modality": modality, **item})


def _save_progress(run_dir: Path, result: dict[str, Any]) -> None:
    atomic_json(run_dir / "run_progress.json", result)


def _zero_test_attestation(record: dict[str, Any]) -> None:
    record["test_evaluation"] = False
    record["test_label_values_indexed"] = False
    if any(str(key).startswith("test_") for key in record):
        raise RuntimeError("Test metric field present")


def run_one(dataset: str, condition: str, seed: int, cache: dict[str, Any], init_audit: dict[str, Any], *, device: str) -> dict[str, Any]:
    from src.data import load_mag_data

    warm_dir = OUTPUT / "runs" / dataset / condition / f"seed_{seed}"
    result_path = warm_dir / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed_and_verified":
            return previous
        raise RuntimeError(f"run output already exists but is incomplete; preserve and inspect it: {warm_dir}")
    warm_dir.mkdir(parents=True, exist_ok=False)
    warm_started = time.perf_counter()
    stage1_seconds = 0.0
    joint_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    source = checkpoint_source(dataset, seed)
    payload = _load_payload(source, dataset, seed)
    cfg = build_config(dataset, CONDITION_MAP[condition], seed, device, cache)
    cfg.model.dropout = 0.2
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 1.0
    cfg.task.grad_clip = GRAD_CLIP
    cfg.task.epochs = JOINT_EPOCHS
    cfg.task.patience = PATIENCE
    cfg.task.early_stop_min_epoch = MIN_EPOCH
    cfg.task.early_stop_min_delta = MIN_DELTA
    cfg.task.eval_every = 1
    cfg.task.scheduler = None
    cfg.task.inference_batch_size = EVAL_BATCH_SIZE
    cfg.paths.output_root = str(warm_dir / "hydra")
    atomic_json(warm_dir / "source_audit.json", source)
    (warm_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    data = load_mag_data(cfg, "nc", seed)
    eval_labels = _resolve_nc_eval_labels(data, include_test=False)
    model, classifier, state_mapping = _construct_model(cfg, data_info(data), cache, seed, payload)
    model.to(device)
    classifier.to(device)
    old_public_hash = module_sha256(model, exclude_selectors=True)
    if old_public_hash != init_audit["conditions"][condition]["public_model_sha256"]:
        raise RuntimeError("constructed public state differs from triplet initialization audit")
    if module_sha256(classifier) != init_audit["conditions"][condition]["classifier_sha256"]:
        raise RuntimeError("loaded classifier state differs from triplet initialization audit")
    if not math.isclose(float(getattr(model, "balance_weight", -1.0)), 0.01, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"V2.1 balance weight must remain 0.01, got {model.balance_weight}")

    history: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    stage_transition: dict[str, Any] = {"dataset": dataset, "condition": condition, "seed": seed, "source_sha256": source["checkpoint_sha256"]}
    _save_progress(warm_dir, {"status": "running", "stage": "t0", "dataset": dataset, "condition": condition, "seed": seed, "source_checkpoint_sha256": source["checkpoint_sha256"]})
    t0_metrics, t0_info, t0_logits = _eval_snapshot(model, classifier, data, device, eval_labels, details=True)
    _metric_match(t0_metrics["validation"], payload["metrics"], f"t0 checkpoint {dataset}/{condition}/{seed}")
    t0_model_state, t0_head_state = _cpu_states(model, classifier)
    best_accuracy = {"stage": "t0", "epoch": 0, "metrics": t0_metrics, "model_state": t0_model_state, "head_state": t0_head_state}
    best_f1 = best_accuracy
    stage_transition["t0"] = {"metrics": t0_metrics, "model_state_sha256": state_sha256(t0_model_state), "classifier_sha256": state_sha256(t0_head_state)}
    append_diagnostics(selector_rows, expert_rows, dataset=dataset, condition=condition, seed=seed, stage="t0", epoch=0, info=t0_info)
    stage1_selector_names: list[str] = []
    added_named = dict(model.named_parameters())
    stage1_selector_names = [name for name in added_named if is_stage1_selector_name(name, CONDITION_MAP[condition])]
    selector_named = [(name, param) for name, param in model.named_parameters() if is_selector_name(name)]
    stage1_named = [(name, added_named[name]) for name in stage1_selector_names]
    if condition in ("W1", "W2") and (not stage1_named or not any(name.startswith("purpose_edge_scorer.") for name, _ in stage1_named)):
        raise RuntimeError(f"expected trainable selector/scorer parameters missing for {condition}")
    stage1_selector_names = apply_stage_trainability(model, classifier, CONDITION_MAP[condition])
    model.eval()
    classifier.eval()
    common_hash_before = module_sha256(model, exclude_selectors=True)
    common_head_hash_before = module_sha256(classifier)
    selector_before = parameter_snapshot(stage1_named)
    scorer_before = state_sha256({name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name.startswith("purpose_edge_scorer.")}) if condition != "W0" else None
    stage_transition["stage1_start"] = {
        "common_model_sha256": common_hash_before, "classifier_sha256": common_head_hash_before,
        "selector_sha256": selector_state_sha256(model) if condition != "W0" else None,
        "model_training": model.training, "grad_enabled_for_forward": True,
        "trainable_names": stage1_selector_names, "optimizer_state_starts_empty": True,
    }
    warm_end_metrics = t0_metrics
    stage1_best_f1 = {"epoch": 0, "metrics": t0_metrics}
    if condition == "W0":
        history.append({
            "dataset": dataset, "condition": condition, "seed": seed, "stage": "selector", "epoch": 0,
            "train_ce": t0_metrics["train"]["cross_entropy"],
            "train_accuracy": t0_metrics["train"]["accuracy"], "train_macro_f1": t0_metrics["train"]["macro_f1"],
            "val_ce": t0_metrics["validation"]["cross_entropy"], "val_accuracy": t0_metrics["validation"]["accuracy"],
            "val_macro_f1": t0_metrics["validation"]["macro_f1"],
            "train_objective": t0_metrics["train"]["cross_entropy"] + float(t0_info.get("aux_loss", 0.0)),
            "selector_update_rms": 0.0,
            "common_update_rms": 0.0, "wall_seconds": 0.0,
        })
    else:
        selector_optimizer = make_stage1_optimizer(model, CONDITION_MAP[condition], learning_rate=STAGE1_LR, weight_decay=WEIGHT_DECAY)
        x = data.x.to(device)
        edge = data.edge_index.to(device)
        train_idx = data.train_idx.to(device)
        y_train = data.y[data.train_idx].to(device)
        stage1_started = time.perf_counter()
        for epoch in range(1, WARM_EPOCHS + 1):
            epoch_started = time.perf_counter()
            model.eval()
            classifier.eval()
            selector_optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                z, _, _, aux_loss, _ = model(x, edge, return_details=False)
                logits = classifier(z[train_idx])
                train_ce = F.cross_entropy(logits, y_train)
                objective = train_ce + aux_loss
                if not objective.requires_grad:
                    raise RuntimeError("Stage 1 eval-mode forward did not produce a differentiable objective")
                objective.backward()
            gradients = grad_norms_by_group(model.named_parameters())
            scorer_grad = math.sqrt(sum(float(parameter.grad.detach().double().square().sum().cpu()) for name, parameter in stage1_named if name.startswith("purpose_edge_scorer.") and parameter.grad is not None))
            cross_delta_grad = 0.0
            if condition == "W2":
                cross_parameter = dict(stage1_named)["cross_delta_coefficients"]
                if cross_parameter.grad is not None:
                    cross_delta_grad = float(cross_parameter.grad.detach().float().norm().cpu())
            if not math.isfinite(scorer_grad):
                raise FloatingPointError("nonfinite Stage 1 scorer gradient")
            if epoch == 1 and scorer_grad <= 0.0:
                raise RuntimeError(f"Stage 1 scorer did not receive a nonzero gradient at epoch {epoch}: {scorer_grad}")
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in stage1_named], max_norm=GRAD_CLIP, error_if_nonfinite=True)
            selector_optimizer.step()
            del z, logits
            metrics, info, _ = _eval_snapshot(model, classifier, data, device, eval_labels, details=True)
            if not finite_metrics(metrics):
                raise FloatingPointError("nonfinite Stage 1 metrics")
            selector_update = rms_update(selector_before, stage1_named)
            append_diagnostics(selector_rows, expert_rows, dataset=dataset, condition=condition, seed=seed, stage="selector", epoch=epoch, info=info, selector_grads=gradients, selector_update=selector_update)
            history.append({
                "dataset": dataset, "condition": condition, "seed": seed, "stage": "selector", "epoch": epoch,
                "train_ce": float(train_ce.detach().cpu()), "train_objective": float(objective.detach().cpu()),
                "train_accuracy": metrics["train"]["accuracy"], "train_macro_f1": metrics["train"]["macro_f1"],
                "val_ce": metrics["validation"]["cross_entropy"], "val_accuracy": metrics["validation"]["accuracy"],
                "val_macro_f1": metrics["validation"]["macro_f1"],
                "selector_grad_norm": gradients["selector"], "scorer_grad_norm": scorer_grad,
                "projector_grad_norm": gradients["projector"], "expert_grad_norm": gradients["experts"],
                "router_grad_norm": gradients["router"], "fusion_grad_norm": gradients["fusion"],
                "other_grad_norm": gradients["other"], "selector_update_rms": selector_update,
                "cross_delta_coefficient_grad_norm": cross_delta_grad,
                "cross_delta_coefficient_update_rms": rms_update(
                    {"cross_delta_coefficients": selector_before["cross_delta_coefficients"]},
                    [(name, parameter) for name, parameter in stage1_named if name == "cross_delta_coefficients"],
                ) if condition == "W2" else 0.0,
                "common_update_rms": 0.0, "learning_rate_common": 0.0,
                "learning_rate_selector": STAGE1_LR, "learning_rate_classifier": 0.0,
                "wall_seconds": time.perf_counter() - epoch_started,
            })
            if metrics["validation"]["macro_f1"] > stage1_best_f1["metrics"]["validation"]["macro_f1"]:
                stage1_best_f1 = {"epoch": epoch, "metrics": metrics}
            stage1_progress = {
                "stage": "selector", "epoch": epoch, "latest_metrics": metrics,
                "selector_gradient_norm": gradients["selector"], "scorer_gradient_norm": scorer_grad,
                "common_hash_unchanged_so_far": module_sha256(model, exclude_selectors=True) == common_hash_before,
                "history": history, "selector_diagnostics": selector_rows, "expert_diagnostics": expert_rows,
            }
            atomic_json(warm_dir / "stage1_progress.json", stage1_progress)
            _save_progress(warm_dir, {"status": "running", **stage1_progress, "stage_transition": stage_transition})
            del objective, train_ce, info
        stage1_seconds = time.perf_counter() - stage1_started
        common_hash_after = module_sha256(model, exclude_selectors=True)
        common_head_hash_after = module_sha256(classifier)
        if common_hash_after != common_hash_before or common_head_hash_after != common_head_hash_before:
            raise RuntimeError("Stage 1 changed frozen public C0 model or classifier state")
        scorer_after = state_sha256({name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name.startswith("purpose_edge_scorer.")})
        scorer_update = rms_update(selector_before, [(name, parameter) for name, parameter in stage1_named if name.startswith("purpose_edge_scorer.")])
        if scorer_before == scorer_after or scorer_update <= 0.0:
            raise RuntimeError("Stage 1 did not update the new purpose edge scorer")
        if condition == "W2" and "cross_delta_coefficients" not in stage1_selector_names:
            raise RuntimeError("C2 cross delta coefficient is absent from Stage 1 trainable selector set")
        cross_delta_updated = bool(condition == "W2" and any(
            not torch.equal(selector_before[name], dict(model.named_parameters())[name].detach().cpu())
            for name in stage1_selector_names if name == "cross_delta_coefficients"
        ))
        if condition == "W2" and (not cross_delta_updated or not any(
            row.get("cross_delta_coefficient_grad_norm", 0.0) > 0 for row in history if row["stage"] == "selector"
        )):
            raise RuntimeError("C2 Stage 1 cross_delta_coefficients did not receive a task gradient and update")
        warm_end_metrics, warm_info, _ = _eval_snapshot(model, classifier, data, device, eval_labels, details=True)
        append_diagnostics(selector_rows, expert_rows, dataset=dataset, condition=condition, seed=seed, stage="warm_end", epoch=WARM_EPOCHS, info=warm_info, selector_update=scorer_update)
        stage_transition["stage1_end"] = {
            "common_model_sha256": common_hash_after, "classifier_sha256": common_head_hash_after,
            "selector_sha256": selector_state_sha256(model), "scorer_sha256_before": scorer_before,
            "scorer_sha256_after": scorer_after, "scorer_update_rms": scorer_update,
            "common_model_unchanged": True, "classifier_unchanged": True,
            "gradient_enabled_eval_forward": True, "stage1_epochs": WARM_EPOCHS,
            "optimizer_state_entries": len(selector_optimizer.state), "metrics": warm_end_metrics,
            "exploratory_best_macro_f1_epoch": stage1_best_f1["epoch"],
            "exploratory_best_macro_f1": stage1_best_f1["metrics"]["validation"]["macro_f1"],
            "cross_delta_coefficients_updated": cross_delta_updated,
        }
        del x, edge, train_idx, y_train, selector_optimizer
    if condition == "W0":
        stage_transition["stage1_end"] = {
            "common_model_sha256": common_hash_before, "classifier_sha256": common_head_hash_before,
            "common_model_unchanged": True, "classifier_unchanged": True,
            "stage1_epochs": 0, "optimizer_state_entries": 0,
            "note": "W0 has no selector-only update; t0 is used as warm_end",
            "metrics": warm_end_metrics,
        }
    stage_transition["warm_end"] = {"metrics": warm_end_metrics, "main_checkpoint_candidate": False}

    # Stage 2 creates a fresh optimizer; no state is restored from the old source checkpoint.
    apply_joint_trainability(model, classifier)
    model_parameters = list(model.named_parameters())
    common_named = [(name, parameter) for name, parameter in model_parameters if not is_selector_name(name)]
    joint_selector_named = [(name, parameter) for name, parameter in model_parameters if is_selector_name(name)]
    optimizer = make_joint_optimizer(
        model, classifier, common_lr=JOINT_COMMON_LR, classifier_lr=JOINT_CLASSIFIER_LR,
        selector_lr=JOINT_SELECTOR_LR, weight_decay=WEIGHT_DECAY,
    )
    stage_transition["stage2_start"] = {
        "optimizer": "AdamW", "optimizer_state_entries_at_creation": len(optimizer.state),
        "optimizer_state_restored_from_source": False, "scheduler": None,
        "common_lr": JOINT_COMMON_LR, "classifier_lr": JOINT_CLASSIFIER_LR,
        "selector_lr": JOINT_SELECTOR_LR if joint_selector_named else None, "weight_decay": WEIGHT_DECAY,
        "common_state_sha256": module_sha256(model, exclude_selectors=True),
        "classifier_sha256": module_sha256(classifier),
        "maximum_common_updates": JOINT_EPOCHS,
    }
    # t0 and joint checkpoints only: Stage 1 is deliberately excluded.
    best_accuracy = {"stage": "t0", "epoch": 0, "metrics": t0_metrics, "model_state": t0_model_state, "head_state": t0_head_state}
    best_f1 = best_accuracy
    best_acc_value = t0_metrics["validation"]["accuracy"]
    best_f1_value = t0_metrics["validation"]["macro_f1"]
    patience_left = PATIENCE
    joint_updates = 0
    joint_started = time.perf_counter()
    best_acc_checkpoint_digest = state_sha256(t0_model_state)
    best_f1_checkpoint_digest = best_acc_checkpoint_digest
    for epoch in range(1, JOINT_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        before_common = parameter_snapshot(common_named)
        before_selector = parameter_snapshot(joint_selector_named)
        before_classifier = parameter_snapshot([(f"classifier.{name}", parameter) for name, parameter in classifier.named_parameters()])
        x = data.x.to(device)
        edge = data.edge_index.to(device)
        train_idx = data.train_idx.to(device)
        y_train = data.y[data.train_idx].to(device)
        z, _, _, aux_loss, _ = model(x, edge, return_details=False)
        train_logits = classifier(z[train_idx])
        train_ce = F.cross_entropy(train_logits, y_train)
        objective = train_ce + aux_loss
        if not torch.isfinite(objective):
            raise FloatingPointError(f"nonfinite joint objective at epoch {epoch}")
        objective.backward()
        gradients = grad_norms_by_group(model.named_parameters())
        classifier_grad = math.sqrt(sum(float(parameter.grad.detach().double().square().sum().cpu()) for parameter in classifier.parameters() if parameter.grad is not None))
        if not math.isfinite(classifier_grad):
            raise FloatingPointError("nonfinite classifier gradient")
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), max_norm=GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        joint_updates += 1
        common_update = rms_update(before_common, common_named)
        selector_update = rms_update(before_selector, joint_selector_named) if joint_selector_named else 0.0
        classifier_update = rms_update(before_classifier, [(f"classifier.{name}", parameter) for name, parameter in classifier.named_parameters()])
        del z, train_logits, objective
        metrics, info, _ = _eval_snapshot(model, classifier, data, device, eval_labels, details=True)
        if not finite_metrics(metrics):
            raise FloatingPointError(f"nonfinite joint metrics at epoch {epoch}")
        append_diagnostics(selector_rows, expert_rows, dataset=dataset, condition=condition, seed=seed, stage="joint", epoch=epoch, info=info, selector_grads=gradients, selector_update=selector_update)
        history.append({
            "dataset": dataset, "condition": condition, "seed": seed, "stage": "joint", "epoch": epoch,
            "train_ce": float(train_ce.detach().cpu()), "train_objective": float(train_ce.detach().cpu() + aux_loss.detach().cpu()),
            "train_accuracy": metrics["train"]["accuracy"], "train_macro_f1": metrics["train"]["macro_f1"],
            "val_ce": metrics["validation"]["cross_entropy"], "val_accuracy": metrics["validation"]["accuracy"],
            "val_macro_f1": metrics["validation"]["macro_f1"],
            "selector_grad_norm": gradients["selector"], "classifier_grad_norm": classifier_grad,
            "projector_grad_norm": gradients["projector"], "expert_grad_norm": gradients["experts"],
            "router_grad_norm": gradients["router"], "fusion_grad_norm": gradients["fusion"],
            "other_grad_norm": gradients["other"], "classifier_update_rms": classifier_update,
            "selector_update_rms": selector_update, "common_update_rms": common_update,
            "learning_rate_common": JOINT_COMMON_LR, "learning_rate_selector": JOINT_SELECTOR_LR if joint_selector_named else 0.0,
            "learning_rate_classifier": JOINT_CLASSIFIER_LR, "wall_seconds": time.perf_counter() - epoch_started,
        })
        val_acc, val_f1 = metrics["validation"]["accuracy"], metrics["validation"]["macro_f1"]
        if val_acc > best_acc_value + MIN_DELTA:
            best_acc_value = val_acc
            acc_model_snapshot, acc_head_snapshot = _cpu_states(model, classifier)
            best_accuracy = {"stage": "joint", "epoch": epoch, "metrics": metrics, "model_state": acc_model_snapshot, "head_state": acc_head_snapshot}
            best_acc_checkpoint_digest = state_sha256(best_accuracy["model_state"])
            patience_left = PATIENCE
        elif epoch >= MIN_EPOCH:
            patience_left -= 1
        if val_f1 > best_f1_value:
            best_f1_value = val_f1
            model_snapshot, head_snapshot = _cpu_states(model, classifier)
            best_f1 = {"stage": "joint", "epoch": epoch, "metrics": metrics, "model_state": model_snapshot, "head_state": head_snapshot}
            best_f1_checkpoint_digest = state_sha256(model_snapshot)
        checkpointed = {
            "dataset": dataset, "condition": condition, "seed": seed, "status": "running",
            "completed_joint_epochs": epoch, "joint_updates": joint_updates,
            "latest_metrics": metrics, "best_accuracy": {"stage": best_accuracy["stage"], "epoch": best_accuracy["epoch"], "metrics": best_accuracy["metrics"]},
            "best_f1": {"stage": best_f1["stage"], "epoch": best_f1["epoch"], "metrics": best_f1["metrics"]},
            "patience_left": patience_left, "history": history, "selector_diagnostics": selector_rows,
            "expert_diagnostics": expert_rows, "stage_transition": stage_transition,
        }
        _save_progress(warm_dir, checkpointed)
        if epoch >= MIN_EPOCH and patience_left <= 0:
            break
    joint_seconds = time.perf_counter() - joint_started
    joint_final_common_hash = module_sha256(model, exclude_selectors=True)

    # Restore each independently selected candidate and verify its Validation metrics before saving.
    accuracy_path = warm_dir / "best_accuracy.pt"
    f1_path = warm_dir / "best_f1.pt"
    common_metadata = {
        "study": "RM-CGE V2.1 Stable Adaptation", "dataset": dataset, "condition": condition,
        "model_condition": CONDITION_MAP[condition], "seed": seed,
        "source_checkpoint_path": source["path"], "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_checkpoint_epoch": source["epoch"], "source_optimizer_state_restored": False,
        "stage1_epochs": WARM_EPOCHS if condition != "W0" else 0,
        "stage2_joint_updates": joint_updates, "test_evaluation": False,
        "test_label_values_indexed": False, "model_state_mapping": state_mapping,
        "stage_transition_audit": stage_transition,
        "resolved_config_sha256": sha256_text(OmegaConf.to_yaml(cfg, resolve=True)),
    }
    best_records = []
    for name, candidate, path, selection in (
        ("best_accuracy", best_accuracy, accuracy_path, "best_validation_accuracy_from_t0_and_joint"),
        ("best_f1", best_f1, f1_path, "best_validation_macro_f1_from_t0_and_joint"),
    ):
        ckpt_payload = {
            "task": "nc", "protocol_version": "unified_full_graph_nc_v1",
            "selection": selection, "selected_stage": candidate["stage"], "epoch": candidate["epoch"],
            "metrics": {
                "train_ce": candidate["metrics"]["train"]["cross_entropy"],
                "train_accuracy": candidate["metrics"]["train"]["accuracy"],
                "train_macro_f1": candidate["metrics"]["train"]["macro_f1"],
                "val_ce": candidate["metrics"]["validation"]["cross_entropy"],
                "val_acc": candidate["metrics"]["validation"]["accuracy"],
                "val_macro_f1": candidate["metrics"]["validation"]["macro_f1"],
            },
            "run_metadata": common_metadata,
            "model_state": candidate["model_state"], "head_state": candidate["head_state"],
            "data_info": data_info(data),
            "data_fingerprints": init_audit["data"],
        }
        temp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt_payload, temp_path)
        temp_path.replace(path)
        persisted = torch.load(path, map_location="cpu", weights_only=False)
        if state_sha256(persisted["model_state"]) != state_sha256(candidate["model_state"]):
            raise RuntimeError(f"checkpoint file model state differs after save: {name}")
        if state_sha256(persisted["head_state"]) != state_sha256(candidate["head_state"]):
            raise RuntimeError(f"checkpoint file classifier state differs after save: {name}")
        if any(key.startswith("test_") for key in persisted.get("metrics", {})):
            raise RuntimeError("Test metrics present in saved warm-start checkpoint")
        model.load_state_dict(persisted["model_state"], strict=True)
        classifier.load_state_dict(persisted["head_state"], strict=True)
        reloaded, _, _ = _eval_snapshot(model, classifier, data, device, eval_labels, details=False)
        for metric in ("accuracy", "macro_f1", "cross_entropy"):
            if abs(reloaded["validation"][metric] - candidate["metrics"]["validation"][metric]) > 2.0e-6:
                raise RuntimeError(f"saved checkpoint reload mismatch: {name}/{metric}")
        if any(key.startswith("test_") for key in reloaded):
            raise RuntimeError("Test data present in selected checkpoint evaluation")
        best_records.append({
            "checkpoint": name, "selected_stage": candidate["stage"], "epoch": candidate["epoch"],
            "metrics": candidate["metrics"], "checkpoint_sha256": sha256_file(path),
            "path": str(path.relative_to(ROOT)),
        })

    total_wall = time.perf_counter() - warm_started
    peak_allocated = int(torch.cuda.max_memory_allocated(torch.device(device)))
    peak_reserved = int(torch.cuda.max_memory_reserved(torch.device(device)))
    acc_train, acc_val = best_accuracy["metrics"]["train"], best_accuracy["metrics"]["validation"]
    f1_val = best_f1["metrics"]["validation"]
    result = {
        "dataset": dataset, "condition": condition, "seed": seed,
        "model_condition": CONDITION_MAP[condition], "status": "completed_and_verified",
        "source_checkpoint": source["path"], "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_checkpoint_epoch": source["epoch"], "selected_stage": best_accuracy["stage"],
        "best_accuracy_epoch": best_accuracy["epoch"], "best_f1_epoch": best_f1["epoch"],
        "best_accuracy_metrics": best_accuracy["metrics"], "best_f1_metrics": best_f1["metrics"],
        "best_f1_stage": best_f1["stage"],
        "t0_metrics": t0_metrics, "warm_end_metrics": warm_end_metrics,
        "stage1_best_f1": stage1_best_f1, "stage1_transition": stage_transition.get("stage1_end"),
        "stage2_transition": stage_transition["stage2_start"],
        "epochs_completed": len([item for item in history if item["stage"] == "joint"]),
        "joint_updates": joint_updates, "stage1_epochs": WARM_EPOCHS if condition != "W0" else 0,
        "wall_seconds": total_wall, "stage1_seconds": stage1_seconds, "joint_seconds": joint_seconds,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classifier_parameters": sum(parameter.numel() for parameter in classifier.parameters()),
        "common_model_hash_after_stage1": stage_transition["stage1_end"]["common_model_sha256"],
        "common_model_hash_after_joint": joint_final_common_hash,
        "optimizer_fresh_start": True, "optimizer_state_entries_after_joint": len(optimizer.state),
        "checkpoint_hashes": {item["checkpoint"]: item["checkpoint_sha256"] for item in best_records},
        "checkpoint_paths": {item["checkpoint"]: item["path"] for item in best_records},
        "checkpoint_records": best_records,
        "history": history, "selector_diagnostics": selector_rows, "expert_diagnostics": expert_rows,
        "stage_transition_audit": stage_transition,
        "resolved_config_sha256": sha256_text(OmegaConf.to_yaml(cfg, resolve=True)),
        "test_evaluation": False, "test_label_values_indexed": False,
    }
    _zero_test_attestation(result)
    result["summary_metrics"] = {
        "train_accuracy": acc_train["accuracy"], "train_macro_f1": acc_train["macro_f1"], "train_ce": acc_train["cross_entropy"],
        "val_acc": acc_val["accuracy"], "val_macro_f1": acc_val["macro_f1"], "val_ce": acc_val["cross_entropy"],
        "best_f1_val_acc": f1_val["accuracy"], "best_f1_val_macro_f1": f1_val["macro_f1"], "best_f1_val_ce": f1_val["cross_entropy"],
        "train_val_acc_gap": acc_train["accuracy"] - acc_val["accuracy"],
        "train_val_macro_f1_gap": acc_train["macro_f1"] - acc_val["macro_f1"],
    }
    atomic_json(result_path, result)
    _save_progress(warm_dir, {"status": "completed_and_verified", "result_path": str(result_path.relative_to(ROOT))})
    del model, classifier, data, payload, optimizer
    torch.cuda.empty_cache()
    return result


def _load_completed_results() -> list[dict[str, Any]]:
    rows = []
    base = OUTPUT / "runs"
    if not base.exists():
        return rows
    for path in sorted(base.glob("*/*/seed_*/result.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") == "completed_and_verified":
            rows.append(item)
    return rows


def persist_tables(results: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_rows, history_rows, selector_rows, expert_rows, tradeoff_rows, resource_rows = [], [], [], [], [], []
    stage_audits = {}
    for result in results:
        metrics = result["summary_metrics"]
        stage_audits[f"{result['dataset']}/{result['condition']}/seed_{result['seed']}"] = result["stage_transition_audit"]
        run_rows.append({
            "dataset": result["dataset"], "condition": result["condition"], "seed": result["seed"], "status": result["status"],
            "source_checkpoint_sha256": result["source_checkpoint_sha256"], "source_epoch": result["source_checkpoint_epoch"],
            "selected_stage": result["selected_stage"], "best_accuracy_epoch": result["best_accuracy_epoch"], "best_f1_epoch": result["best_f1_epoch"],
            "best_f1_stage": result["best_f1_stage"],
            "val_acc": metrics["val_acc"], "val_macro_f1": metrics["val_macro_f1"], "val_ce": metrics["val_ce"],
            "best_f1_val_acc": metrics["best_f1_val_acc"], "best_f1_val_macro_f1": metrics["best_f1_val_macro_f1"], "best_f1_val_ce": metrics["best_f1_val_ce"],
            "t0_val_acc": result["t0_metrics"]["validation"]["accuracy"], "t0_val_macro_f1": result["t0_metrics"]["validation"]["macro_f1"],
            "warm_end_val_acc": result["warm_end_metrics"]["validation"]["accuracy"], "warm_end_val_macro_f1": result["warm_end_metrics"]["validation"]["macro_f1"],
            "train_acc": metrics["train_accuracy"], "train_macro_f1": metrics["train_macro_f1"], "train_ce": metrics["train_ce"],
            "train_val_acc_gap": metrics["train_val_acc_gap"], "train_val_macro_f1_gap": metrics["train_val_macro_f1_gap"],
            "epochs_completed": result["epochs_completed"], "joint_updates": result["joint_updates"], "stage1_epochs": result["stage1_epochs"],
            "wall_seconds": result["wall_seconds"], "peak_allocated_bytes": result["peak_allocated_bytes"], "peak_reserved_bytes": result["peak_reserved_bytes"],
            "gpu_name": result["gpu_name"], "model_parameters": result["model_parameters"], "classifier_parameters": result["classifier_parameters"],
            "result_path": str((OUTPUT / "runs" / result["dataset"] / result["condition"] / f"seed_{result['seed']}" / "result.json").relative_to(ROOT)),
            "best_accuracy_checkpoint": result["checkpoint_paths"]["best_accuracy"], "best_f1_checkpoint": result["checkpoint_paths"]["best_f1"],
            "test_evaluation": False, "error": "",
        })
        history_rows.extend(result["history"])
        selector_rows.extend(result["selector_diagnostics"])
        expert_rows.extend(result["expert_diagnostics"])
        for checkpoint in ("best_accuracy", "best_f1"):
            chosen = result["best_accuracy_metrics"] if checkpoint == "best_accuracy" else result["best_f1_metrics"]
            stage = result["selected_stage"] if checkpoint == "best_accuracy" else result["best_f1_stage"]
            epoch = result["best_accuracy_epoch"] if checkpoint == "best_accuracy" else result["best_f1_epoch"]
            tradeoff_rows.append({
                "dataset": result["dataset"], "condition": result["condition"], "seed": result["seed"],
                "checkpoint": checkpoint, "selected_stage": stage, "epoch": epoch,
                "train_accuracy": chosen["train"]["accuracy"], "train_macro_f1": chosen["train"]["macro_f1"], "train_ce": chosen["train"]["cross_entropy"],
                "val_accuracy": chosen["validation"]["accuracy"], "val_macro_f1": chosen["validation"]["macro_f1"], "val_ce": chosen["validation"]["cross_entropy"],
                "train_val_acc_gap": chosen["train"]["accuracy"] - chosen["validation"]["accuracy"],
                "train_val_macro_f1_gap": chosen["train"]["macro_f1"] - chosen["validation"]["macro_f1"],
                "source_checkpoint_sha256": result["source_checkpoint_sha256"],
                "checkpoint_sha256": result["checkpoint_hashes"][checkpoint],
            })
        resource_rows.append({
            "dataset": result["dataset"], "condition": result["condition"], "seed": result["seed"],
            "wall_seconds": result["wall_seconds"], "stage1_seconds": result["stage1_seconds"],
            "joint_seconds": result["joint_seconds"], "epochs_completed": result["epochs_completed"],
            "joint_updates": result["joint_updates"], "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"], "gpu_name": result["gpu_name"],
        })
    run_rows.sort(key=lambda row: (row["dataset"], int(row["seed"]), row["condition"]))
    write_rows(DATA_DIR / "runs.csv", run_rows, RUN_FIELDS)
    write_rows(DATA_DIR / "epoch_history.csv", history_rows, HISTORY_FIELDS)
    write_rows(DATA_DIR / "selector_diagnostics.csv", selector_rows, SELECTOR_FIELDS)
    write_rows(DATA_DIR / "expert_router_diagnostics.csv", expert_rows, EXPERT_FIELDS)
    write_rows(DATA_DIR / "checkpoint_tradeoffs.csv", tradeoff_rows, TRADEOFF_FIELDS)
    write_rows(DATA_DIR / "resource_profile.csv", resource_rows, RESOURCE_FIELDS)
    atomic_json(DATA_DIR / "stage_transition_audit.json", stage_audits)


def audit_and_run(seeds: tuple[int, ...], *, device: str, audit_only: bool = False) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this study")
    torch.cuda.set_device(torch.device(device))
    caches = verify_cache_entries()
    init_path = DATA_DIR / "initialization_audit.json"
    initial_audit = json.loads(init_path.read_text(encoding="utf-8")) if init_path.exists() else {"pairs": {}, "datasets": {}, "status": "in_progress"}
    for dataset in DATASETS:
        for seed in seeds:
            key = f"{dataset}/seed_{seed}"
            entry = initial_audit["pairs"].get(key)
            if not entry or entry.get("status") != "passed":
                entry = audit_initial_triplet(dataset, seed, caches[dataset], device=device)
                initial_audit["pairs"][key] = entry
                initial_audit["datasets"][dataset] = entry["data"]
                initial_audit["status"] = "in_progress"
                atomic_json(init_path, initial_audit)
    initial_audit["status"] = "passed" if all(initial_audit["pairs"].get(f"{dataset}/seed_{seed}", {}).get("status") == "passed" for dataset in DATASETS for seed in seeds) else "in_progress"
    atomic_json(init_path, initial_audit)
    if audit_only:
        print(json.dumps({"audit_only": True, "status": initial_audit["status"], "pairs_checked": [f"{dataset}/seed_{seed}" for dataset in DATASETS for seed in seeds]}, sort_keys=True), flush=True)
        return
    for seed in seeds:
        for dataset in DATASETS:
            for condition in ("W0", "W1", "W2"):
                try:
                    result = run_one(dataset, condition, seed, caches[dataset], initial_audit["pairs"][f"{dataset}/seed_{seed}"], device=device)
                except BaseException as exc:
                    append_failure(dataset, condition, seed, exc)
                    raise
                current = _load_completed_results()
                persist_tables(current)
                print(json.dumps({"completed": f"{dataset}/{condition}/seed_{seed}", "val_acc": result["summary_metrics"]["val_acc"], "val_macro_f1": result["summary_metrics"]["val_macro_f1"], "selected_stage": result["selected_stage"]}, sort_keys=True), flush=True)
        if seed == 42:
            seed42_results = [row for row in _load_completed_results() if int(row["seed"]) == 42]
            expected = len(DATASETS) * len(CONDITION_MAP)
            if len(seed42_results) != expected:
                raise RuntimeError(f"seed 42 audit gate expected {expected} completed runs, got {len(seed42_results)}")
            for item in seed42_results:
                if item["status"] != "completed_and_verified" or item["test_evaluation"] or item["peak_allocated_bytes"] <= 0:
                    raise RuntimeError(f"seed 42 batch gate failed: {item['dataset']}/{item['condition']}")
                for metric in (item["summary_metrics"]["val_acc"], item["summary_metrics"]["val_macro_f1"], item["summary_metrics"]["val_ce"]):
                    if not math.isfinite(metric):
                        raise RuntimeError("seed 42 batch has nonfinite Validation metrics")

    previous_checks = {}
    checks_path = DATA_DIR / "implementation_checks.json"
    if checks_path.is_file():
        previous_checks = json.loads(checks_path.read_text(encoding="utf-8"))
    implementation_checks = {
        **previous_checks,
        "initialization_audit": initial_audit,
        "formal_run_count": len(_load_completed_results()),
        "test_evaluation_disabled": True,
        "finalized": False,
    }
    atomic_json(DATA_DIR / "implementation_checks.json", implementation_checks)


def append_failure(dataset: str, condition: str, seed: int, exc: BaseException) -> None:
    path = DATA_DIR / "failures.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"failures": []}
    current.setdefault("failures", []).append({
        "dataset": dataset, "condition": condition, "seed": seed, "status": "failed",
        "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    atomic_json(path, current)


def parse_seeds(value: str) -> tuple[int, ...]:
    if value.lower() == "all":
        return tuple(SEEDS)
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or any(seed not in SEEDS for seed in parsed):
        raise argparse.ArgumentTypeError(f"seeds must be a subset of {SEEDS} or 'all'")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42", type=parse_seeds, help="42 for the gated first batch, or all")
    parser.add_argument("--device", default="cuda:0", help="use cuda:0 with CUDA_VISIBLE_DEVICES=1")
    parser.add_argument("--audit-only", action="store_true", help="verify t0 loading and paired initialization without training")
    args = parser.parse_args()
    try:
        audit_and_run(args.seeds, device=args.device, audit_only=args.audit_only)
    except BaseException as exc:
        failure_path = DATA_DIR / "failures.json"
        current = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {"failures": []}
        if not current.get("failures") or current["failures"][-1].get("error") != str(exc):
            append_failure("initialization", "audit", -1, exc)
        raise


if __name__ == "__main__":
    main()
