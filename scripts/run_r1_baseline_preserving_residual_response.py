from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m0_adaptive_propagation import load_m0_data, restrict_labels_to_train_val
from scripts.run_r0_mature_response_expert_prototype import (
    build_classifier,
    make_config as make_r0_config,
    model_data_info,
    tensor_hash,
)
from src.models.adaptive_prop_m0 import Model as M0Model, remove_self_messages as m0_remove_self_messages
from src.models.adaptive_prop_n1 import Model as N1Model
from src.models.mature_response_r0 import (
    NUM_EXPERTS,
    PROMPT_DIM,
    TOP_K,
    Model as R0Model,
    pairwise_prompt_cosines,
    prompt_orthogonality_loss,
    self_anchored_low_high,
)
from src.models.residual_response_r1 import Model, VARIANTS, parameter_counts
from src.tasks.common import build_optimizer, scheduler_step
from src.utils.seeds import set_seed


DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
CONTRASTS = (
    ("residual_generic", "smooth_base", "residual_generic-Smooth"),
    ("residual_protected", "smooth_base", "residual_protected-Smooth"),
    ("residual_protected", "residual_generic", "residual_protected-residual_generic"),
)
SOURCE_BRANCH = "exp/r0_mature_response_expert_prototype"
SOURCE_SHA = "d60db509e15e830a09a691665dcbe72435137477"
EXPERIMENT_BRANCH = "exp/r1_baseline_preserving_residual_response"
OUT_DIR = PROJECT_ROOT / "outputs" / "r1_baseline_preserving_residual_response"
RESEARCH_DIR = PROJECT_ROOT / "research" / "r1_baseline_preserving_residual_response"
ORTH_WEIGHT = 1e-3
SHUFFLE_SEEDS = (6101, 6102, 6103, 6104, 6105)
EPS = 1e-8


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_config(dataset: str, seed: int, variant: str, device: str,
                epochs: int | None = None):
    overrides = [
        f"dataset={dataset}", "task=nc", "model=residual_response_r1",
        f"model.variant={variant}", f"seed={seed}", "num_runs=1",
        f"device={device}", "task.evaluate_test=false",
    ]
    if epochs is not None:
        overrides.append(f"task.epochs={int(epochs)}")
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def check_provenance() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    source_local = subprocess.check_output(
        ["git", "rev-parse", SOURCE_BRANCH], cwd=PROJECT_ROOT, text=True
    ).strip()
    source_remote = subprocess.check_output(
        ["git", "rev-parse", f"origin/{SOURCE_BRANCH}"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if branch != EXPERIMENT_BRANCH or source_local != SOURCE_SHA or source_remote != SOURCE_SHA:
        raise RuntimeError(
            f"R1 provenance mismatch: branch={branch}, source={source_local}, "
            f"origin_source={source_remote}"
        )
    return {
        "source_branch": SOURCE_BRANCH,
        "source_sha": SOURCE_SHA,
        "remote_source_sha": source_remote,
        "experiment_branch": branch,
        "head_at_start": head,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item()) if left.numel() else 0.0


def _copy_projectors(source: nn.Module, target: Model) -> None:
    target.proj_t.load_state_dict(source.proj_t.state_dict())
    target.proj_v.load_state_dict(source.proj_v.state_dict())


def copy_m0_smooth_common(source: nn.Module, target: Model) -> None:
    _copy_projectors(source, target)
    for modality in range(2):
        target.smooth_transforms[modality].load_state_dict(source.w0[modality].state_dict())
        target.smooth_residual_norms[modality].load_state_dict(
            source.residual_norms[modality].state_dict()
        )
    target.fusion.load_state_dict(source.fusion.state_dict())


def copy_n1_smooth_common(source: nn.Module, target: Model) -> None:
    _copy_projectors(source, target)
    for modality in range(2):
        target.smooth_transforms[modality].load_state_dict(source.w_s[modality].state_dict())
        target.smooth_residual_norms[modality].load_state_dict(
            source.residual_norms[modality].state_dict()
        )
    target.fusion.load_state_dict(source.fusion.state_dict())


def copy_r0_smooth_common(source: R0Model, target: Model) -> None:
    _copy_projectors(source, target)
    for modality in range(2):
        target.smooth_transforms[modality].load_state_dict(
            source.low_transforms[modality].state_dict()
        )
        target.smooth_residual_norms[modality].load_state_dict(
            source.smooth_residual_norms[modality].state_dict()
        )
    target.fusion.load_state_dict(source.fusion.state_dict())


def copy_r0_response_weights(source: R0Model, target: Model) -> None:
    _copy_projectors(source, target)
    for name in (
        "low_transforms", "high_transforms", "low_response_norms", "high_response_norms",
        "cross_experts", "routers", "generic_composers", "protected_attention",
        "response_residual_norms", "post_ffns", "final_residual_norms",
    ):
        getattr(target, name).load_state_dict(getattr(source, name).state_dict())
    with torch.no_grad():
        for modality in range(2):
            target.cross_prompts[modality].copy_(source.cross_prompts[modality])
    target.fusion.load_state_dict(source.fusion.state_dict())


@torch.no_grad()
def smooth_compatibility_regression(dataset: str, seed: int, cfg, info: dict[str, int],
                                    data, device: torch.device) -> dict[str, Any]:
    """Map M0 UNI, N1 SmoothOnly, and R0 SmoothBase into the R1 Smooth path."""
    m0_cfg = make_r0_config(dataset, seed, "smooth_base", str(device))
    m0_cfg.model.name, m0_cfg.model.variant = "adaptive_prop_m0", "uniform"
    n1_cfg = make_r0_config(dataset, seed, "smooth_base", str(device))
    n1_cfg.model.name, n1_cfg.model.variant = "adaptive_prop_n1", "smooth_only"
    r0_cfg = make_r0_config(dataset, seed, "smooth_base", str(device))
    r1_cfg = copy.deepcopy(cfg)
    r1_cfg.model.variant = "smooth_base"

    set_seed(seed); m0 = M0Model(m0_cfg, info).to(device).eval()
    set_seed(seed); n1 = N1Model(n1_cfg, info).to(device).eval()
    set_seed(seed); r0 = R0Model(r0_cfg, info).to(device).eval()
    set_seed(seed); r1_m0 = Model(r1_cfg, info).to(device).eval()
    set_seed(seed); r1_n1 = Model(r1_cfg, info).to(device).eval()
    set_seed(seed); r1_r0 = Model(r1_cfg, info).to(device).eval()
    copy_m0_smooth_common(m0, r1_m0)
    copy_n1_smooth_common(n1, r1_n1)
    copy_r0_smooth_common(r0, r1_r0)

    x, edge_index = data.x.to(device), data.edge_index.to(device)
    m0_z, _, _, _, _ = m0(x, edge_index)
    n1_z, _, _, _, n1_aux = n1(x, edge_index, return_diagnostics=True)
    r0_z, _, _, _, r0_aux = r0(x, edge_index, return_diagnostics=True)
    z_m0, _, _, _, m0_aux = r1_m0(x, edge_index, return_diagnostics=True)
    z_n1, _, _, _, r1_n1_aux = r1_n1(x, edge_index, return_diagnostics=True)
    z_r0, _, _, _, r1_r0_aux = r1_r0(x, edge_index, return_diagnostics=True)

    # Reconstruct M0's raw mean and transformed message in its exact operation order.
    m0_x_t, m0_x_v = m0.split_modalities(x)
    m0_intrinsic = [m0.proj_t(m0_x_t), m0.proj_v(m0_x_v)]
    edge_index_nonself, src, dst = m0_remove_self_messages(edge_index)
    del edge_index_nonself
    degree = torch.zeros(x.size(0), dtype=torch.long, device=device)
    if dst.numel():
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.long))
    m0_neighbor_mean, m0_smooth_delta, m0_smooth_state = [], [], []
    for modality in range(2):
        raw_sum = m0_intrinsic[modality].new_zeros(m0_intrinsic[modality].shape)
        mapped_sum = m0_intrinsic[modality].new_zeros(m0_intrinsic[modality].shape)
        if src.numel():
            raw_sum.index_add_(0, dst, m0_intrinsic[modality][src])
            mapped_sum.index_add_(0, dst, m0.w0[modality](m0_intrinsic[modality][src]))
        denom = degree.clamp_min(1).to(raw_sum.dtype).unsqueeze(-1)
        mean = raw_sum / denom
        delta = mapped_sum / denom
        m0_neighbor_mean.append(mean)
        m0_smooth_delta.append(delta)
        m0_smooth_state.append(m0.residual_norms[modality](m0_intrinsic[modality] + delta))

    n1_neighbor_mean = []
    for modality in range(2):
        raw_sum = n1_aux["h0"][modality].new_zeros(n1_aux["h0"][modality].shape)
        if src.numel():
            raw_sum.index_add_(0, dst, n1_aux["h0"][modality][src])
        n1_neighbor_mean.append(
            raw_sum / degree.clamp_min(1).to(raw_sum.dtype).unsqueeze(-1)
        )

    comparisons: dict[str, float] = {}
    for modality, name in enumerate(("text", "visual")):
        comparisons[f"m0_projector_{name}"] = _max_abs(
            m0_intrinsic[modality], m0_aux["intrinsic"][modality]
        )
        comparisons[f"m0_neighbor_mean_{name}"] = _max_abs(
            m0_neighbor_mean[modality], m0_aux["neighbor_mean"][modality]
        )
        comparisons[f"m0_smooth_delta_{name}"] = _max_abs(
            m0_smooth_delta[modality], m0_aux["smooth_delta"][modality]
        )
        comparisons[f"m0_smooth_state_{name}"] = _max_abs(
            m0_smooth_state[modality], m0_aux["structural"][modality]
        )
        comparisons[f"n1_projector_{name}"] = _max_abs(
            n1_aux["h0"][modality], r1_n1_aux["intrinsic"][modality]
        )
        comparisons[f"n1_neighbor_mean_{name}"] = _max_abs(
            n1_neighbor_mean[modality], r1_n1_aux["neighbor_mean"][modality]
        )
        comparisons[f"n1_smooth_delta_{name}"] = _max_abs(
            n1_aux["contexts"][modality][0], r1_n1_aux["smooth_delta"][modality]
        )
        comparisons[f"n1_smooth_state_{name}"] = _max_abs(
            n1_aux["h_tilde"][modality], r1_n1_aux["structural"][modality]
        )
        comparisons[f"r0_projector_{name}"] = _max_abs(
            r0_aux["intrinsic"][modality], r1_r0_aux["intrinsic"][modality]
        )
        comparisons[f"r0_neighbor_mean_{name}"] = _max_abs(
            r0_aux["neighbor_mean"][modality], r1_r0_aux["neighbor_mean"][modality]
        )
        comparisons[f"r0_smooth_delta_{name}"] = _max_abs(
            r0_aux["smooth_delta"][modality], r1_r0_aux["smooth_delta"][modality]
        )
        comparisons[f"r0_smooth_state_{name}"] = _max_abs(
            r0_aux["structural"][modality], r1_r0_aux["structural"][modality]
        )
    comparisons.update({
        "m0_final_z": _max_abs(m0_z, z_m0),
        "n1_final_z": _max_abs(n1_z, z_n1),
        "r0_final_z": _max_abs(r0_z, z_r0),
    })
    tolerance = {"rtol": 1e-6, "atol": 1e-5}
    max_error = max(comparisons.values())
    result = {
        "dataset": dataset, "seed": int(seed), "device": str(device),
        "status": "passed" if max_error <= tolerance["atol"] else "failed",
        "max_abs_error": max_error, "tolerance": tolerance,
        "comparisons_max_abs_error": comparisons,
        "note": "Mapped Smooth paths agree within tolerance; CUDA scatter reductions may differ at floating-point roundoff.",
    }
    if result["status"] != "passed":
        raise AssertionError(f"R1 Smooth compatibility regression failed: {result}")
    return result


@torch.no_grad()
def response_branch_regression(dataset: str, seed: int, cfg, info: dict[str, int],
                               data, device: torch.device) -> dict[str, Any]:
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    comparisons: dict[str, float] = {}
    for r1_variant, r0_variant, suffix in (
        ("residual_generic", "bank_crossmoe_generic", "generic"),
        ("residual_protected", "bank_crossmoe_protected", "protected"),
    ):
        r0_cfg = make_r0_config(dataset, seed, r0_variant, str(device))
        r1_cfg = copy.deepcopy(cfg); r1_cfg.model.variant = r1_variant
        set_seed(seed); r0 = R0Model(r0_cfg, info).to(device).eval()
        set_seed(seed); r1 = Model(r1_cfg, info).to(device).eval()
        copy_r0_response_weights(r0, r1)
        _, _, r0_structural, _, a0 = r0(x, edge_index, return_diagnostics=True)
        _, _, _, _, a1 = r1(x, edge_index, return_diagnostics=True)
        for modality, name in enumerate(("text", "visual")):
            for key in ("low_raw", "high_raw", "low", "high", "cross",
                        "composition_response"):
                comparisons[f"{suffix}_{key}_{name}"] = _max_abs(
                    a0[key][modality], a1[key][modality]
                )
            comparisons[f"{suffix}_branch_structural_{name}"] = _max_abs(
                r0_structural[modality], a1["branch_structural"][modality]
            )
        if suffix == "protected":
            for modality, name in enumerate(("text", "visual")):
                comparisons[f"protected_attention_{name}"] = _max_abs(
                    a0["attention"][modality], a1["attention"][modality]
                )
    # CUDA scatter can change the last few bits of the raw high response; the
    # subsequent LayerNorm amplifies that small reduction-order difference.
    tolerance = 2e-5 if device.type == "cuda" else 1e-6
    maximum = max(comparisons.values())
    result = {
        "dataset": dataset, "seed": int(seed), "device": str(device),
        "status": "passed" if maximum <= tolerance else "failed",
        "max_abs_error": maximum, "tolerance": tolerance,
        "comparisons_max_abs_error": comparisons,
    }
    if result["status"] != "passed":
        raise AssertionError(f"R0 response branch regression failed: {result}")
    return result


def _parameter_summary(model: nn.Module, prefixes: tuple[str, ...]) -> dict[str, Any]:
    named = [(name, p) for name, p in model.named_parameters() if name.startswith(prefixes)]
    grads = [(name, p.grad.detach()) for name, p in named if p.grad is not None]
    norm = math.sqrt(sum(float(g.float().square().sum().item()) for _, g in grads)) if grads else 0.0
    return {
        "parameter_tensors": len(named),
        "gradient_tensors": len(grads),
        "parameters_with_gradient": sum(bool(torch.count_nonzero(g)) for _, g in grads),
        "finite": all(bool(torch.isfinite(g).all()) for _, g in grads),
        "norm": norm,
    }


GRADIENT_GROUPS = {
    "correction_projection": ("correction_transforms.",),
    "smooth_transforms": ("smooth_transforms.",),
    "bank_low_high": ("low_transforms.", "high_transforms.",
                      "low_response_norms.", "high_response_norms."),
    "cross_prompts": ("cross_prompts.",),
    "cross_experts": ("cross_experts.",),
    "routers": ("routers.",),
    "generic_composer": ("generic_composers.",),
    "protected_attention": ("protected_attention.",),
    "post_composer": ("response_residual_norms.", "post_ffns.", "final_residual_norms."),
    "fusion": ("fusion.",),
}


def gradient_bootstrap_audit(dataset: str, seed: int, cfg, info: dict[str, int], data,
                             device: torch.device) -> list[dict[str, Any]]:
    """Separate task and orthogonal gradients, then verify the two-step bootstrap."""
    splits = restrict_labels_to_train_val(data)
    train_idx = splits["train"].to(device)
    labels = data.y.to(device)
    train_y = labels[train_idx]
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    rows = []
    for variant in ("residual_generic", "residual_protected"):
        local_cfg = copy.deepcopy(cfg); local_cfg.model.variant = variant
        set_seed(seed)
        model = Model(local_cfg, info).to(device)
        classifier = build_classifier(model.out_dim, info["num_classes"], seed, device)
        set_seed(seed)
        optimizer = build_optimizer(
            list(model.parameters()) + list(classifier.parameters()), local_cfg, model=model
        )

        model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)
        z = model(x, edge_index)[0]
        task_loss = F.cross_entropy(classifier(z[train_idx]), train_y)
        task_loss.backward()
        step1_task = {name: _parameter_summary(model, prefixes)
                      for name, prefixes in GRADIENT_GROUPS.items()}
        step1_c_norm = float(sum(p.detach().float().square().sum().item()
                                 for p in model.correction_transforms.parameters()) ** 0.5)

        model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)
        z, _, _, orth_raw, _ = model(x, edge_index)
        orth_raw.backward()
        orth_only = {name: _parameter_summary(model, prefixes)
                     for name, prefixes in GRADIENT_GROUPS.items()}
        orth_loss_raw = float(orth_raw.detach().item())
        model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)

        # The actual first optimizer update uses task CE plus the frozen R0 orth term.
        z, _, _, orth_raw, _ = model(x, edge_index)
        task_loss = F.cross_entropy(classifier(z[train_idx]), train_y)
        (task_loss + model.orth_weight * orth_raw).backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()), 1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        post_step_c_norm = float(sum(p.detach().float().square().sum().item()
                                     for p in model.correction_transforms.parameters()) ** 0.5)

        model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)
        z = model(x, edge_index)[0]
        second_task_loss = F.cross_entropy(classifier(z[train_idx]), train_y)
        second_task_loss.backward()
        step2_task = {name: _parameter_summary(model, prefixes)
                      for name, prefixes in GRADIENT_GROUPS.items()}

        branch_prefix = "generic_composer" if variant == "residual_generic" else "protected_attention"
        inactive_prefix = "protected_attention" if variant == "residual_generic" else "generic_composer"
        if not step1_task["correction_projection"]["finite"] or \
                step1_task["correction_projection"]["norm"] <= 1e-12:
            raise AssertionError(f"R1 step 1 W_C task gradient missing for {variant}")
        for key in ("bank_low_high", "cross_experts", "routers", branch_prefix):
            if step1_task[key]["norm"] > 1e-10:
                raise AssertionError(f"R1 step 1 task gradient leaked upstream to {key}: {step1_task[key]}")
        if post_step_c_norm <= 0:
            raise AssertionError(f"R1 optimizer step did not move W_C for {variant}")
        for key in ("bank_low_high", "cross_experts", "routers", branch_prefix):
            group = step2_task[key]
            if not group["finite"] or group["norm"] <= 1e-12:
                raise AssertionError(f"R1 step 2 task gradient did not enter {key}: {group}")
        if step2_task[inactive_prefix]["gradient_tensors"] != 0:
            raise AssertionError(f"R1 inactive composer unexpectedly received gradients: {inactive_prefix}")
        if orth_only["cross_prompts"]["norm"] <= 0 or \
                orth_only["correction_projection"]["gradient_tensors"] != 0:
            raise AssertionError("R1 orth-only gradient isolation failed")

        rows.append({
            "dataset": dataset, "seed": int(seed), "variant": variant,
            "step1_correction_weight_norm_before": step1_c_norm,
            "step1_task_loss": float(task_loss.detach().item()),
            "step1_task_gradients": step1_task,
            "orth_loss_raw": orth_loss_raw,
            "orth_only_gradients": orth_only,
            "post_step_correction_weight_norm": post_step_c_norm,
            "step2_task_loss": float(second_task_loss.detach().item()),
            "step2_task_gradients": step2_task,
            "active_composer": branch_prefix,
            "inactive_composer": inactive_prefix,
            "checks_passed": True,
        })
        del model, classifier, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def audit_initialization(dataset: str, seed: int, info: dict[str, int], cfg) -> dict[str, Any]:
    states, classifiers, hashes, classifier_hashes, counts = {}, {}, {}, {}, {}
    for variant in VARIANTS:
        local = copy.deepcopy(cfg); local.model.variant = variant
        set_seed(seed)
        model = Model(local, info)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        states[variant] = state
        hashes[variant] = tensor_hash(state)
        counts[variant] = parameter_counts(model)
        classifiers[variant] = build_classifier(model.out_dim, info["num_classes"], seed).state_dict()
        classifiers[variant] = {key: value.detach().cpu().clone()
                                for key, value in classifiers[variant].items()}
        classifier_hashes[variant] = tensor_hash(classifiers[variant])
        counts[variant]["classifier_params"] = sum(v.numel() for v in classifiers[variant].values())
        counts[variant]["total_trainable_params"] = (
            counts[variant]["model_trainable"] + counts[variant]["classifier_params"]
        )
        del model

    comparisons = {}
    for left, right in ((VARIANTS[0], VARIANTS[1]), (VARIANTS[0], VARIANTS[2]),
                        (VARIANTS[1], VARIANTS[2])):
        comparisons[f"{left}__{right}"] = (
            states[left].keys() == states[right].keys()
            and all(torch.equal(states[left][key], states[right][key]) for key in states[left])
            and all(torch.equal(classifiers[left][key], classifiers[right][key])
                    for key in classifiers[left])
        )
    parameter_match = len({row["model_trainable"] for row in counts.values()}) == 1
    classifier_match = len({row["classifier_params"] for row in counts.values()}) == 1
    total_match = len({row["total_trainable_params"] for row in counts.values()}) == 1
    result = {
        "dataset": dataset, "seed": int(seed), "variant_hashes": hashes,
        "classifier_hashes": classifier_hashes,
        "pairwise_bitwise_equal": comparisons,
        "all_bitwise_equal": all(comparisons.values()),
        "exact_model_parameter_match": parameter_match,
        "exact_classifier_parameter_match": classifier_match,
        "exact_total_parameter_match": total_match,
        "parameter_counts": counts,
    }
    if not result["all_bitwise_equal"] or not parameter_match or not classifier_match or not total_match:
        raise AssertionError(f"R1 initialization/capacity fairness failed: {result}")
    return result


def initialization_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "dataset": audit["dataset"], "seed": audit["seed"], "variant": variant,
        "model_hash": audit["variant_hashes"][variant],
        "classifier_hash": audit["classifier_hashes"][variant],
        "model_params": counts["model_trainable"],
        "classifier_params": counts["classifier_params"],
        "total_params": counts["total_trainable_params"],
        "smooth_transform_params": counts["smooth_transforms"],
        "bank_low_transform_params": counts["bank_low_transforms"],
        "bank_high_transform_params": counts["bank_high_transforms"],
        "correction_transform_params": counts["correction_transforms"],
        "all_variants_bitwise_equal": audit["all_bitwise_equal"],
        "exact_total_parameter_match": audit["exact_total_parameter_match"],
    } for variant, counts in audit["parameter_counts"].items()]


def _rms_per_node(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=-1).sqrt()


def _cosine_per_node(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-8)


def _distribution(values: torch.Tensor | np.ndarray | list[float],
                  quantiles=(0.10, 0.25, 0.50, 0.75, 0.90, 0.99)) -> dict[str, float]:
    array = np.asarray(values.detach().float().cpu() if isinstance(values, torch.Tensor) else values,
                       dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                **{f"q{int(q * 100):02d}": float("nan") for q in quantiles}}
    return {
        "n": int(array.size), "mean": float(array.mean()), "std": float(array.std(ddof=0)),
        **{f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in quantiles},
    }


def collect_r1_diagnostics(model: Model, aux: dict[str, Any], val_idx: torch.Tensor,
                           dataset: str, seed: int, variant: str) -> dict[str, list[dict[str, Any]]]:
    correction_rows, direction_rows, response_rows, moe_rows, attention_rows = [], [], [], [], []
    for modality, modality_name in enumerate(("text", "visual")):
        scope = val_idx
        smooth = aux["smooth_state"][modality][scope]
        response_value = aux["residual"][modality]
        response = (response_value[scope] if response_value is not None
                    else torch.zeros_like(smooth))
        correction = aux["correction"][modality][scope]
        rms_smooth = _rms_per_node(smooth)
        rms_response = _rms_per_node(response)
        rms_correction = _rms_per_node(correction)
        ratio = rms_correction / (rms_smooth + EPS)
        for kind, values in (("smooth_rms", rms_smooth), ("residual_rms", rms_response),
                             ("correction_rms", rms_correction), ("correction_ratio", ratio)):
            stats = _distribution(values)
            correction_rows.append({
                "dataset": dataset, "seed": int(seed), "variant": variant,
                "modality": modality_name, "quantity": kind, **stats,
            })
        direction_row = {
            "dataset": dataset, "seed": int(seed), "variant": variant,
            "modality": modality_name,
        }
        for name, values in (
            ("cosine_correction_smooth", _cosine_per_node(correction, smooth)),
            ("cosine_correction_residual", _cosine_per_node(correction, response)),
        ):
            direction_row.update({f"{name}_{key}": value
                                  for key, value in _distribution(values).items()})
        direction_rows.append(direction_row)

        if variant != "smooth_base":
            for key, values in (("low_raw", aux["low_raw"]), ("high_raw", aux["high_raw"]),
                                ("low", aux["low"]), ("high", aux["high"]),
                                ("cross", aux["cross"]),
                                ("branch_structural", aux["branch_structural"])):
                if values[modality] is not None:
                    response_rows.append({
                        "dataset": dataset, "seed": int(seed), "variant": variant,
                        "modality": modality_name, "quantity": key,
                        **_distribution(_rms_per_node(values[modality][scope])),
                    })

        if aux["moe"]:
            route = aux["moe"][modality]
            idx = route["top_indices"][scope]
            weights = route["top_weights"][scope]
            counts_top1 = torch.bincount(idx[:, 0].detach().cpu(), minlength=NUM_EXPERTS)
            counts_top2 = torch.bincount(idx.reshape(-1).detach().cpu(), minlength=NUM_EXPERTS)
            prompt_cos = pairwise_prompt_cosines(model.cross_prompts[modality].detach())
            expert_outputs = route["expert_outputs"][scope].float()
            expert_pair_cos, expert_pair_dist = [], []
            for i in range(NUM_EXPERTS):
                for j in range(i + 1, NUM_EXPERTS):
                    expert_pair_cos.extend(_cosine_per_node(
                        expert_outputs[:, i], expert_outputs[:, j]
                    ).cpu().tolist())
                    distance = (expert_outputs[:, i] - expert_outputs[:, j]).norm(dim=-1)
                    scale = (expert_outputs[:, i].norm(dim=-1)
                             + expert_outputs[:, j].norm(dim=-1)).clamp_min(EPS)
                    expert_pair_dist.extend((distance / scale).cpu().tolist())
            moe_rows.append({
                "dataset": dataset, "seed": int(seed), "variant": variant,
                "modality": modality_name,
                "top1_fraction_by_expert": (counts_top1 / max(int(idx.size(0)), 1)).tolist(),
                "top2_inclusion_fraction_by_expert": (counts_top2 / max(int(idx.size(0)), 1)).tolist(),
                "largest_top1_share": float((counts_top1 / max(int(idx.size(0)), 1)).max()),
                "top_weight_distribution": _distribution(weights),
                "prompt_pair_cosine": _distribution(prompt_cos),
                "prompt_orth_loss": float(prompt_orthogonality_loss(
                    model.cross_prompts[modality].detach()
                ).item()),
                "expert_pair_cosine": _distribution(expert_pair_cos),
                "expert_pair_normalized_distance": _distribution(expert_pair_dist),
                "expert_output_rms": _distribution(_rms_per_node(expert_outputs.reshape(-1, 128))),
                "orth_loss_raw": float(aux["orth_loss"].item()),
            })

        if aux["attention"][modality] is not None:
            weights = aux["attention"][modality][scope].float().mean(dim=1)
            for token, token_name in enumerate(("L", "H", "X")):
                attention_rows.append({
                    "dataset": dataset, "seed": int(seed), "variant": variant,
                    "modality": modality_name, "token": token_name,
                    **_distribution(weights[:, token]),
                    "node_sd": float(weights[:, token].std(unbiased=False).item()),
                })
    return {
        "correction_scale": correction_rows,
        "correction_direction": direction_rows,
        "response": response_rows,
        "moe": moe_rows,
        "protected_attention": attention_rows,
    }


def metric_from_embeddings(classifier, z, val_idx, val_y, num_classes: int) -> dict[str, float]:
    logits = classifier(z[val_idx])
    ce = F.cross_entropy(logits, val_y)
    pred = logits.argmax(-1)
    return {
        "val_accuracy": float((pred == val_y).float().mean().item()),
        "val_macro_f1": float(f1_score(
            val_y.detach().cpu().numpy(), pred.detach().cpu().numpy(),
            labels=list(range(num_classes)), average="macro", zero_division=0,
        )),
        "val_ce": float(ce.item()),
    }


@torch.no_grad()
def evaluate(model: Model, classifier: nn.Module, x: torch.Tensor, edge_index: torch.Tensor,
             val_idx: torch.Tensor, val_y: torch.Tensor, num_classes: int,
             correction_off: bool = False,
             correction_override: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, float]:
    model.eval(); classifier.eval()
    z = model(x, edge_index, correction_off=correction_off,
              correction_override=correction_override)[0]
    return metric_from_embeddings(classifier, z, val_idx, val_y, num_classes)


@torch.no_grad()
def intervention_rows(model: Model, classifier: nn.Module, x: torch.Tensor,
                      edge_index: torch.Tensor, val_idx: torch.Tensor, val_y: torch.Tensor,
                      num_classes: int, base_metrics: dict[str, float],
                      aux: dict[str, Any], dataset: str, seed: int) -> list[dict[str, Any]]:
    rows = []

    def record(name: str, metrics: dict[str, float], repeat_seed: int | None = None):
        rows.append({
            "dataset": dataset, "seed": int(seed), "variant": model.variant,
            "intervention": name, "repeat_seed": repeat_seed,
            "delta_accuracy_pp": 100 * (metrics["val_accuracy"] - base_metrics["val_accuracy"]),
            "delta_macro_f1_pp": 100 * (metrics["val_macro_f1"] - base_metrics["val_macro_f1"]),
            "delta_ce": metrics["val_ce"] - base_metrics["val_ce"],
        })

    correction = tuple(value.detach().clone() for value in aux["correction"])
    identity = evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes,
                        correction_override=correction)
    if max(abs(identity[key] - base_metrics[key]) for key in base_metrics) > 1e-6:
        raise AssertionError("R1 correction override identity failed")
    record("identity", identity)
    record("correction_off", evaluate(
        model, classifier, x, edge_index, val_idx, val_y, num_classes,
        correction_off=True,
    ))

    if model.variant == "residual_protected":
        original_structural = aux["structural"]
        other_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        other_mask[val_idx] = False
        for repeat_seed in SHUFFLE_SEEDS:
            shuffled = [value.detach().clone() for value in correction]
            for modality in range(2):
                generator = torch.Generator(device="cpu").manual_seed(
                    repeat_seed + modality * 1009
                )
                permutation = torch.randperm(int(val_idx.numel()), generator=generator).to(val_idx.device)
                shuffled[modality][val_idx] = correction[modality][val_idx[permutation]]
            moved_z, _, moved_structural, _, _ = model(
                x, edge_index, correction_override=tuple(shuffled), return_diagnostics=False
            )
            if any(not torch.equal(original_structural[m][other_mask], moved_structural[m][other_mask])
                   for m in range(2)):
                raise AssertionError("validation correction shuffle changed non-validation structural rows")
            base_z = model(x, edge_index)[0]
            nonval_z_error = _max_abs(base_z[other_mask], moved_z[other_mask])
            if nonval_z_error > 1e-5:
                raise AssertionError(
                    "validation correction shuffle changed non-validation embeddings: "
                    f"max_abs_error={nonval_z_error}"
                )
            metrics = metric_from_embeddings(classifier, moved_z, val_idx, val_y, num_classes)
            record("correction_tuple_shuffle", metrics, repeat_seed)
    return rows


def initial_identity_audit(dataset: str, seed: int, cfg, info: dict[str, int],
                           data, device: torch.device) -> dict[str, Any]:
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    models = []
    for variant in VARIANTS:
        local = copy.deepcopy(cfg); local.model.variant = variant
        set_seed(seed)
        models.append(Model(local, info).to(device))
    errors: dict[str, float] = {}
    for mode in ("eval", "train"):
        outputs = []
        for model in models:
            model.eval() if mode == "eval" else model.train()
            set_seed(seed + 17003)
            outputs.append(model(x, edge_index, return_diagnostics=True))
        base = outputs[0]
        for i, variant in ((1, "residual_generic"), (2, "residual_protected")):
            errors[f"{mode}_{variant}_final_z"] = _max_abs(base[0], outputs[i][0])
            for modality, name in enumerate(("text", "visual")):
                errors[f"{mode}_{variant}_{name}_smooth_state"] = _max_abs(
                    base[4]["smooth_state"][modality], outputs[i][4]["smooth_state"][modality]
                )
                errors[f"{mode}_{variant}_{name}_structural"] = _max_abs(
                    base[2][modality], outputs[i][2][modality]
                )
                if torch.count_nonzero(models[i].correction_transforms[modality].weight):
                    raise AssertionError("R1 correction projection was not exactly zero at initialization")
    if any(value != 0.0 for value in errors.values()):
        raise AssertionError(f"R1 initial Smooth identity is not bitwise exact: {errors}")
    return {
        "dataset": dataset, "seed": int(seed), "status": "passed",
        "max_abs_error": max(errors.values(), default=0.0),
        "exact_zero_error": all(value == 0.0 for value in errors.values()),
        "comparisons_max_abs_error": errors,
    }


def _weight_norms(model: Model) -> list[float]:
    return [float(layer.weight.detach().float().norm().item())
            for layer in model.correction_transforms]


def run_one(dataset: str, seed: int, variant: str, device_name: str,
            out_dir: Path = OUT_DIR, epochs: int | None = None,
            save_checkpoint: bool = True, preloaded_data=None,
            init_audit: dict[str, Any] | None = None,
            smoke: bool = False) -> dict[str, Any]:
    cfg = make_config(dataset, seed, variant, device_name, epochs)
    data = preloaded_data if preloaded_data is not None else load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    info = model_data_info(data)
    if init_audit is None:
        init_audit = audit_initialization(dataset, seed, info, cfg)
    device = torch.device(device_name)
    set_seed(seed)
    model = Model(cfg, info).to(device)
    runtime_init_hash = tensor_hash(model.state_dict())
    if runtime_init_hash != init_audit["variant_hashes"][variant]:
        raise AssertionError(f"R1 runtime initialization differs for {dataset}/{seed}/{variant}")
    classifier = build_classifier(model.out_dim, info["num_classes"], seed, device)
    runtime_classifier_hash = tensor_hash(classifier.state_dict())
    if runtime_classifier_hash != init_audit["classifier_hashes"][variant]:
        raise AssertionError("R1 runtime classifier initialization differs from audit")
    # Match the R0 training stream after all model/classifier construction.
    set_seed(seed)
    optimizer = build_optimizer(
        list(model.parameters()) + list(classifier.parameters()), cfg, model=model
    )
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    train_idx = splits["train"].to(device)
    val_idx = splits["validation"].to(device)
    labels = data.y.to(device)
    train_y, val_y = labels[train_idx], labels[val_idx]
    if bool((train_y < 0).any()) or bool((val_y < 0).any()) or data.test_idx is not None:
        raise AssertionError("R1 must contain train/validation labels only and no test indices")
    if bool(cfg.task.evaluate_test):
        raise AssertionError("R1 requires task.evaluate_test=false")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_acc, best_epoch, patience_left = -1.0, 0, int(cfg.task.patience)
    best_model = best_classifier = best_metrics = None
    epoch_rows: list[dict[str, Any]] = []
    weight_rows = [{
        "dataset": dataset, "seed": int(seed), "variant": variant, "epoch": 0,
        "stage": "initial", "text_weight_frobenius": _weight_norms(model)[0],
        "visual_weight_frobenius": _weight_norms(model)[1],
    }]
    started = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, int(cfg.task.epochs) + 1):
        epochs_run = epoch
        model.train(); classifier.train()
        optimizer.zero_grad(set_to_none=True)
        z, _, _, orth_raw, _ = model(x, edge_index)
        logits = classifier(z[train_idx])
        task_loss = F.cross_entropy(logits, train_y)
        orth_weighted = model.orth_weight * orth_raw
        loss = task_loss + orth_weighted
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite R1 loss {dataset}/{seed}/{variant}/epoch{epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()),
            max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        norms = _weight_norms(model)
        weight_rows.append({
            "dataset": dataset, "seed": int(seed), "variant": variant, "epoch": epoch,
            "stage": "post_optimizer_step", "text_weight_frobenius": norms[0],
            "visual_weight_frobenius": norms[1],
        })
        epoch_rows.append({
            "epoch": epoch,
            "task_loss_ce": float(task_loss.detach().item()),
            "orth_loss_raw": float(orth_raw.detach().item()),
            "orth_loss_weighted": float(orth_weighted.detach().item()),
            "orth_to_task_ratio": float(
                (orth_weighted.detach() / task_loss.detach().clamp_min(EPS)).item()
            ),
            "text_correction_weight_frobenius": norms[0],
            "visual_correction_weight_frobenius": norms[1],
        })
        del z, logits, task_loss, orth_raw, orth_weighted, loss

        model.eval(); classifier.eval()
        with torch.no_grad():
            val_z = model(x, edge_index)[0]
            metrics = metric_from_embeddings(classifier, val_z, val_idx, val_y, info["num_classes"])
        if metrics["val_accuracy"] > best_acc + float(cfg.task.early_stop_min_delta):
            best_acc, best_epoch, best_metrics = metrics["val_accuracy"], epoch, metrics
            best_model = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_classifier = {key: value.detach().cpu().clone()
                               for key, value in classifier.state_dict().items()}
            patience_left = int(cfg.task.patience)
        elif epoch >= int(cfg.task.early_stop_min_epoch):
            patience_left -= 1
            if patience_left <= 0:
                break
        del val_z

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    if best_model is None or best_classifier is None or best_metrics is None:
        raise RuntimeError("R1 training ended without a validation-selected checkpoint")
    model.load_state_dict(best_model, strict=True)
    classifier.load_state_dict(best_classifier, strict=True)
    model.eval(); classifier.eval()
    with torch.no_grad():
        z, _, _, orth_raw_best, aux = model(x, edge_index, return_diagnostics=True)
        selected_metrics = metric_from_embeddings(
            classifier, z, val_idx, val_y, info["num_classes"]
        )
        complement_error = max(
            _max_abs(aux["low_raw"][m] + aux["high_raw"][m], aux["intrinsic"][m])
            for m in range(2)
        )
        if not bool(torch.isfinite(z).all()) or complement_error > 2e-6:
            raise AssertionError(f"R1 selected checkpoint numerical audit failed: {complement_error}")

    diagnostics = collect_r1_diagnostics(model, aux, val_idx, dataset, seed, variant)
    interventions = []
    intervention_seconds = 0.0
    if variant != "smooth_base":
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_start = time.perf_counter()
        interventions = intervention_rows(
            model, classifier, x, edge_index, val_idx, val_y, info["num_classes"],
            selected_metrics, aux, dataset, seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_seconds = time.perf_counter() - intervention_start
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    best_weight_norm = _weight_norms(model)
    weight_rows.append({
        "dataset": dataset, "seed": int(seed), "variant": variant, "epoch": best_epoch,
        "stage": "best_checkpoint", "text_weight_frobenius": best_weight_norm[0],
        "visual_weight_frobenius": best_weight_norm[1],
    })
    best_loss_row = next(row for row in epoch_rows if row["epoch"] == best_epoch)
    row = {
        "status": "completed", "dataset": dataset, "seed": int(seed), "variant": variant,
        "smoke": bool(smoke), "best_epoch": best_epoch, "epochs_run": epochs_run,
        **selected_metrics,
        "model_params": sum(int(p.numel()) for p in model.parameters()),
        "classifier_params": sum(int(p.numel()) for p in classifier.parameters()),
        "total_params": sum(int(p.numel()) for p in model.parameters())
                        + sum(int(p.numel()) for p in classifier.parameters()),
        "parameter_counts": parameter_counts(model),
        "peak_gpu_memory_bytes": peak_bytes,
        "training_time_sec": training_seconds,
        "intervention_time_sec": intervention_seconds,
        "orth_weight": model.orth_weight,
        "low_high_complement_max_error": complement_error,
        "best_epoch_task_loss_ce": best_loss_row["task_loss_ce"],
        "best_epoch_orth_loss_raw": float(orth_raw_best.item()),
        "best_epoch_orth_loss_weighted": float((model.orth_weight * orth_raw_best).item()),
        "best_epoch_orth_to_task_ratio": float(
            (model.orth_weight * orth_raw_best).item()
            / max(best_loss_row["task_loss_ce"], EPS)
        ),
        "epoch_history": epoch_rows,
        "correction_weight_norm_history": weight_rows,
        "diagnostics": diagnostics,
        "interventions": interventions,
        "initialization_audit": init_audit,
        "runtime_model_init_hash": runtime_init_hash,
        "runtime_classifier_init_hash": runtime_classifier_hash,
        "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False,
        "test_idx_attached": False,
        "test_labels_exposed": False,
        "link_prediction": False,
        "validation_label_indices_only": True,
        "data_info": info,
    }
    run_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    write_json(run_path, row)
    if save_checkpoint:
        checkpoint_path = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": best_model, "classifier_state": best_classifier,
            "data_info": info, "dataset": dataset, "seed": int(seed),
            "variant": variant, "best_epoch": best_epoch,
            "best_val_metrics": selected_metrics,
            "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        }, checkpoint_path)
        row["checkpoint"] = str(checkpoint_path.relative_to(PROJECT_ROOT))
        write_json(run_path, row)
    print(
        f"[R1 {'smoke' if smoke else 'done'}] {dataset}/{seed}/{variant} "
        f"best_epoch={best_epoch} acc={selected_metrics['val_accuracy']:.5f} "
        f"f1={selected_metrics['val_macro_f1']:.5f} ce={selected_metrics['val_ce']:.5f} "
        f"sec={training_seconds:.1f}", flush=True,
    )
    del model, classifier, optimizer, aux, z
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def run_smoke(device_name: str = "cuda:1") -> dict[str, Any]:
    out_dir = OUT_DIR / "smoke"
    cfg = make_config("Movies", 42, "smooth_base", device_name, epochs=1)
    data = load_m0_data(cfg, 42)
    splits = restrict_labels_to_train_val(data)
    info = model_data_info(data)
    device = torch.device(device_name)
    init_audit = audit_initialization("Movies", 42, info, cfg)
    smooth_regression = smooth_compatibility_regression(
        "Movies", 42, cfg, info, data, device
    )
    response_regression = response_branch_regression(
        "Movies", 42, cfg, info, data, device
    )
    initial_identity = initial_identity_audit("Movies", 42, cfg, info, data, device)

    # Parameter ownership audit: all three 128 x 128 transforms are independent.
    set_seed(42)
    ownership_model = Model(cfg, info)
    ownership = {
        "smooth_vs_low_distinct": all(
            ownership_model.smooth_transforms[m].weight.data_ptr()
            != ownership_model.bank_low_transforms[m].weight.data_ptr() for m in range(2)
        ),
        "smooth_vs_high_distinct": all(
            ownership_model.smooth_transforms[m].weight.data_ptr()
            != ownership_model.bank_high_transforms[m].weight.data_ptr() for m in range(2)
        ),
        "low_vs_high_distinct": all(
            ownership_model.bank_low_transforms[m].weight.data_ptr()
            != ownership_model.bank_high_transforms[m].weight.data_ptr() for m in range(2)
        ),
        "all_transforms_have_no_bias": all(
            layer.bias is None for layer in (
                *ownership_model.smooth_transforms,
                *ownership_model.bank_low_transforms,
                *ownership_model.bank_high_transforms,
                *ownership_model.correction_transforms,
            )
        ),
    }
    if not all(ownership.values()):
        raise AssertionError(f"R1 transform decoupling failed: {ownership}")

    gradient_rows = gradient_bootstrap_audit(
        "Movies", 42, cfg, info, data, device
    )
    short_runs = []
    for variant in VARIANTS:
        short_runs.append(run_one(
            "Movies", 42, variant, device_name, out_dir=out_dir, epochs=1,
            save_checkpoint=False, preloaded_data=data, init_audit=init_audit, smoke=True,
        ))
    audit = {
        "status": "passed",
        "source_sha": SOURCE_SHA,
        "dataset": "Movies", "seed": 42, "device": device_name,
        "initialization_audit": init_audit,
        "smooth_compatibility_regression": smooth_regression,
        "r0_response_branch_regression": response_regression,
        "initial_output_identity": initial_identity,
        "transform_ownership": ownership,
        "gradient_bootstrap_audit": gradient_rows,
        "short_gpu_runs": [
            {"variant": row["variant"], "status": row["status"],
             "best_epoch": row["best_epoch"], "val_accuracy": row["val_accuracy"],
             "val_macro_f1": row["val_macro_f1"], "val_ce": row["val_ce"]}
            for row in short_runs
        ],
        "evaluate_test": False, "test_idx_attached": False,
        "test_labels_exposed": False, "link_prediction": False,
    }
    write_json(out_dir / "smoke_audit.json", audit)
    write_json(RESEARCH_DIR / "data" / "smoke_audit.json", audit)
    write_csv(RESEARCH_DIR / "data" / "gradient_bootstrap_audit.csv", gradient_rows)
    del data, ownership_model
    return audit


def _read_existing_run(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unreadable"}
    return value if isinstance(value, dict) else {"status": "unreadable"}


def run_campaign(device_name: str = "cuda:1", retry_failed: bool = True) -> dict[str, Any]:
    provenance = check_provenance()
    smoke_path = RESEARCH_DIR / "data" / "smoke_audit.json"
    if smoke_path.is_file():
        smoke_audit = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke_audit.get("status") != "passed":
            raise RuntimeError("R1 smoke audit is not passed")
    else:
        smoke_audit = run_smoke(device_name)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    research_data = RESEARCH_DIR / "data"
    research_data.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []
    init_rows: list[dict[str, Any]] = []
    init_audits: dict[tuple[str, int], dict[str, Any]] = {}
    device = torch.device(device_name)
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg = make_config(dataset, seed, "smooth_base", device_name)
            data = load_m0_data(cfg, seed)
            splits = restrict_labels_to_train_val(data)
            if data.test_idx is not None or data.info.get("test_split_field_read", False):
                raise AssertionError("R1 dataset loader exposed test split metadata")
            info = model_data_info(data)
            audit = audit_initialization(dataset, seed, info, cfg)
            init_audits[(dataset, seed)] = audit
            init_rows.extend(initialization_rows(audit))
            for variant in VARIANTS:
                run_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                existing = _read_existing_run(run_path)
                if existing is not None and existing.get("status") == "completed":
                    records.append(existing)
                    continue
                if existing is not None:
                    if not retry_failed:
                        failures.append({
                            "dataset": dataset, "seed": seed, "variant": variant,
                            "status": "existing_record_not_retried",
                        })
                        continue
                    retries.append({
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "previous_status": existing.get("status", "unreadable"),
                        "retry_attempt": True,
                    })
                try:
                    result = run_one(
                        dataset, seed, variant, device_name, out_dir=out_dir,
                        preloaded_data=data, init_audit=audit, save_checkpoint=True,
                    )
                    records.append(result)
                except Exception as exc:
                    failure = {
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "status": "failed", "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "retry_attempt": existing is not None,
                    }
                    failures.append(failure)
                    write_json(out_dir / "failures" / dataset / f"seed_{seed}_{variant}.json", failure)
                    print(f"[R1 failed] {dataset}/{seed}/{variant}: {exc!r}", flush=True)
            del data
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    completed = sum(row.get("status") == "completed" for row in records)
    manifest = {
        **provenance,
        "datasets": list(DATASETS), "seeds": list(SEEDS), "variants": list(VARIANTS),
        "expected_runs": 27, "completed_runs": completed,
        "failures": failures, "retries": retries,
        "runtime_seconds": elapsed, "device": device_name,
        "epochs_override": None, "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False, "test_indices_attached": False,
        "test_labels_exposed": False, "link_prediction": False,
        "orthogonality_weight": ORTH_WEIGHT,
        "correction_shuffle_seeds": list(SHUFFLE_SEEDS),
        "smooth_compatibility_regression": smoke_audit["smooth_compatibility_regression"],
        "response_branch_regression": smoke_audit["r0_response_branch_regression"],
        "initial_output_identity": smoke_audit["initial_output_identity"],
        "transform_ownership": smoke_audit["transform_ownership"],
        "gradient_bootstrap_audit": smoke_audit["gradient_bootstrap_audit"],
        "training_config": OmegaConf.to_container(
            make_config("Movies", 42, "smooth_base", device_name).task, resolve=True
        ),
        "training_seconds_sum": sum(float(row.get("training_time_sec", 0)) for row in records),
        "intervention_seconds_sum": sum(float(row.get("intervention_time_sec", 0))
                                         for row in records),
        "peak_gpu_memory_bytes_max": max(
            [int(row.get("peak_gpu_memory_bytes", 0)) for row in records] or [0]
        ),
    }
    write_json(out_dir / "campaign_manifest.json", manifest)
    write_json(out_dir / "failures.json", failures)
    write_json(RESEARCH_DIR / "run_manifest.json", manifest)
    write_csv(research_data / "parameter_init_audit.csv", init_rows)
    if records:
        from scripts.analyze_r1_baseline_preserving_residual_response import analyze_campaign
        analysis = analyze_campaign(out_dir, RESEARCH_DIR, manifest)
        manifest["final_label"] = analysis["final_label"]
        write_json(out_dir / "campaign_manifest.json", manifest)
        write_json(RESEARCH_DIR / "run_manifest.json", manifest)
    print(
        f"[R1 campaign] completed={completed}/27 failed={len(failures)} "
        f"retries={len(retries)} elapsed={elapsed:.1f}s device={device_name}",
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--retry-failed", action="store_true", default=True)
    parser.add_argument("--no-retry-failed", action="store_false", dest="retry_failed")
    args = parser.parse_args()
    if args.smoke == args.campaign:
        parser.error("choose exactly one of --smoke or --campaign")
    if args.smoke:
        result = run_smoke(args.device)
        print(f"[R1 smoke] {result['status']} device={args.device}", flush=True)
    else:
        run_campaign(args.device, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
