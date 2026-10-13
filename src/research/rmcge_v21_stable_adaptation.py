"""Shared utilities for the RM-CGE V2.1 stable-adaptation runner."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


SELECTOR_PREFIXES = (
    "query_projection.",
    "key_projection.",
    "edge_target_modality_embedding.",
    "edge_purpose_embedding.",
    "purpose_edge_scorer.",
    "cross_delta_coefficients",
)
SELECTOR_ONLY_PREFIXES = SELECTOR_PREFIXES[:-1]
INITIAL_LOGIT_ATOL = 5.0e-5
INITIAL_LOGIT_RTOL = 5.0e-5


def is_selector_name(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in SELECTOR_PREFIXES)


def is_stage1_selector_name(name: str, condition: str) -> bool:
    if any(name.startswith(prefix) for prefix in SELECTOR_ONLY_PREFIXES):
        return True
    return condition == "C2" and name == "cross_delta_coefficients"


def apply_stage_trainability(model: torch.nn.Module, classifier: torch.nn.Module, condition: str) -> list[str]:
    """Freeze inherited model/head parameters and expose only the Stage 1 selectors."""
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = is_stage1_selector_name(name, condition)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return trainable


def apply_joint_trainability(model: torch.nn.Module, classifier: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)


def make_stage1_optimizer(
    model: torch.nn.Module, condition: str, *, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if is_stage1_selector_name(name, condition)
    ]
    if not parameters:
        raise ValueError("Stage 1 selector optimizer cannot be empty")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer.state:
        raise AssertionError("new Stage 1 optimizer state must be empty")
    return optimizer


def make_joint_optimizer(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    *,
    common_lr: float,
    classifier_lr: float,
    selector_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    common = [parameter for name, parameter in model.named_parameters() if not is_selector_name(name)]
    selector = [parameter for name, parameter in model.named_parameters() if is_selector_name(name)]
    groups = [
        {"params": common, "lr": common_lr, "weight_decay": weight_decay, "group_name": "common_model"},
        {"params": list(classifier.parameters()), "lr": classifier_lr, "weight_decay": weight_decay, "group_name": "classifier"},
    ]
    if selector:
        groups.append({"params": selector, "lr": selector_lr, "weight_decay": weight_decay, "group_name": "selector"})
    optimizer = torch.optim.AdamW(groups)
    if optimizer.state:
        raise AssertionError("new Stage 2 optimizer state must be empty")
    return optimizer


def physical_degree_groups(degree: torch.Tensor) -> dict[str, torch.Tensor]:
    degree = degree.long()
    return {
        "degree_0": degree == 0,
        "degree_1": degree == 1,
        "degree_2_3": (degree >= 2) & (degree <= 3),
        "degree_4_7": (degree >= 4) & (degree <= 7),
        "degree_ge8": degree >= 8,
        "degree_lt2": degree < 2,
        "degree_ge2": degree >= 2,
    }


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_sha256(module: torch.nn.Module, *, exclude_selectors: bool = False) -> str:
    state = module.state_dict()
    if exclude_selectors:
        state = {name: value for name, value in state.items() if not is_selector_name(name)}
    return state_sha256(state)


def selector_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if is_selector_name(name)}


def selector_state_sha256(model: torch.nn.Module) -> str:
    return state_sha256(selector_state(model))


def max_state_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise ValueError("state dictionaries have different keys")
    maximum = 0.0
    for name in left:
        if left[name].shape != right[name].shape:
            raise ValueError(f"state tensor shape differs: {name}")
        if left[name].numel():
            maximum = max(maximum, float((left[name].detach().cpu() - right[name].detach().cpu()).abs().max()))
    return maximum


def metric_bundle(logits: torch.Tensor, labels: torch.Tensor, eval_labels: list[int]) -> dict[str, Any]:
    logits = logits.detach().float()
    labels = labels.detach().long()
    if logits.ndim != 2 or logits.size(0) != labels.numel():
        raise ValueError("logits and labels must align")
    if not torch.isfinite(logits).all() or not torch.isfinite(labels.float()).all():
        raise FloatingPointError("nonfinite logits or labels")
    ce = F.cross_entropy(logits, labels)
    predictions = logits.argmax(dim=-1)
    y_cpu, pred_cpu = labels.cpu().numpy(), predictions.cpu().numpy()
    return {
        "accuracy": float((predictions == labels).float().mean().cpu()),
        "macro_f1": float(f1_score(y_cpu, pred_cpu, labels=eval_labels, average="macro", zero_division=0)),
        "cross_entropy": float(ce.detach().cpu()),
        "count": int(labels.numel()),
    }


def finite_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_metrics(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_metrics(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError("validation logits have different shapes")
    delta = (reference.detach().float() - candidate.detach().float()).abs()
    max_abs = float(delta.max().cpu()) if delta.numel() else 0.0
    rms = float(delta.square().mean().sqrt().cpu()) if delta.numel() else 0.0
    return {
        "max_abs": max_abs,
        "rms": rms,
        "allclose": bool(torch.allclose(reference, candidate, atol=INITIAL_LOGIT_ATOL, rtol=INITIAL_LOGIT_RTOL)),
    }


def select_main_candidates(
    candidates: list[dict[str, Any]], *, min_delta: float = 1.0e-4
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select independent accuracy and macro-F1 states from t0 plus joint epochs."""
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    if candidates[0].get("stage") != "t0":
        raise ValueError("the first candidate must be the common t0 state")
    best_accuracy = candidates[0]
    best_f1 = candidates[0]
    for candidate in candidates[1:]:
        if candidate.get("stage") != "joint":
            raise ValueError("only joint epochs may follow t0 in the main candidate list")
        if float(candidate["metrics"]["validation"]["accuracy"]) > float(best_accuracy["metrics"]["validation"]["accuracy"]) + min_delta:
            best_accuracy = candidate
        if float(candidate["metrics"]["validation"]["macro_f1"]) > float(best_f1["metrics"]["validation"]["macro_f1"]):
            best_f1 = candidate
    return best_accuracy, best_f1


def grad_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().square().sum().cpu())
    return math.sqrt(squared)


def grad_norms_by_group(named_parameters) -> dict[str, float]:
    groups: dict[str, list[torch.nn.Parameter]] = {key: [] for key in ("selector", "projector", "experts", "router", "fusion", "other")}
    for name, parameter in named_parameters:
        if is_selector_name(name):
            group = "selector"
        elif name.startswith("projectors."):
            group = "projector"
        elif name.startswith("experts."):
            group = "experts"
        elif any(token in name for token in ("router", "selection_head", "strength_head")):
            group = "router"
        elif name.startswith("fusion_") or name.startswith("fusion_norm."):
            group = "fusion"
        else:
            group = "other"
        groups[group].append(parameter)
    return {name: grad_norm(parameters) for name, parameters in groups.items()}


def rms_update(before: dict[str, torch.Tensor], named_parameters) -> float:
    squared = 0.0
    count = 0
    for name, parameter in named_parameters:
        if name not in before:
            continue
        current = parameter.detach().float()
        delta = current - before[name].to(device=current.device, dtype=current.dtype)
        squared += float(delta.double().square().sum().cpu())
        count += delta.numel()
    return math.sqrt(squared / max(count, 1))


def parameter_snapshot(named_parameters) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in named_parameters}


def normalize_expert_diagnostic_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Map V2.1's detail keys to the stable-adaptation CSV schema."""
    aliases = {
        "delta_intra_rms": "delta_rms",
        "delta_intra_hat_rms": "delta_hat_rms",
        "delta_intra_weighted_rms_by_expert": "delta_weighted_rms_by_expert",
        "cross_base_rms": "cross_rms",
        "cross_base_hat_rms": "cross_hat_rms",
        "cross_base_weighted_rms_by_expert": "cross_weighted_rms_by_expert",
        "delta_cross_rms": "cross_delta_rms",
        "delta_cross_hat_rms": "cross_delta_hat_rms",
        "delta_cross_weighted_rms_by_expert": "cross_delta_weighted_rms_by_expert",
    }
    return {**record, **{target: record.get(source) for target, source in aliases.items()}}
