from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
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

from scripts.run_m0_adaptive_propagation import (
    load_m0_data,
    restrict_labels_to_train_val,
)
from src.models.adaptive_prop_m0 import Model as M0Model
from src.models.adaptive_prop_n1 import Model as N1Model
from src.models.mature_response_r0 import (
    NUM_EXPERTS,
    PROMPT_DIM,
    TOP_K,
    VARIANTS,
    Model,
    copy_m0_uniform_common_weights,
    copy_n1_smooth_common_weights,
    pairwise_prompt_cosines,
    parameter_counts,
    prompt_orthogonality_loss,
    remove_self_messages,
)
from src.tasks.common import build_optimizer, scheduler_step
from src.utils.seeds import set_seed


DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS_ORDERED = VARIANTS
SOURCE_BRANCH = "exp/n1_recipient_function_strength_mixer"
SOURCE_SHA = "2fbe1f14cda0d33d72f8c8468ef9068af28f6217"
OUT_DIR = PROJECT_ROOT / "outputs" / "r0_mature_response_expert_prototype"
RESEARCH_DIR = PROJECT_ROOT / "research" / "r0_mature_response_expert_prototype"
ORTH_WEIGHT = 1e-3
ROUTER_SHUFFLE_SEEDS = (5101, 5102, 5103, 5104, 5105)
EPS = 1e-8


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temp.replace(path)


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
        f"dataset={dataset}", "task=nc", "model=mature_response_r0",
        f"model.variant={variant}", f"seed={seed}", "num_runs=1",
        f"device={device}", "task.evaluate_test=false",
    ]
    if epochs is not None:
        overrides.append(f"task.epochs={int(epochs)}")
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def model_data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_classifier(out_dim: int, num_classes: int, seed: int,
                     device: torch.device | str = "cpu") -> nn.Linear:
    torch.manual_seed(int(seed) + 1907)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 1907)
    return nn.Linear(int(out_dim), int(num_classes)).to(resolved_device)


def _prompt_init_summary(model: Model) -> list[dict[str, Any]]:
    output = []
    for modality, prompts in enumerate(model.cross_prompts):
        cosine = pairwise_prompt_cosines(prompts.detach())
        # Prompts are small; store the three unordered pair values explicitly.
        normed = F.normalize(prompts.detach(), dim=-1)
        pair_values = [float((normed[i] * normed[j]).sum().item())
                       for i, j in ((0, 1), (0, 2), (1, 2))]
        output.append({
            "modality": "text" if modality == 0 else "visual",
            "pairwise_cosines": pair_values,
            "orth_loss": float(cosine.exp().mean().item()),
            "prompt_norms": [float(value) for value in prompts.detach().norm(dim=-1).cpu()],
        })
    return output


def audit_initialization(dataset: str, seed: int, data_info: dict[str, int], cfg) -> dict[str, Any]:
    states, classifiers, hashes, classifier_hashes, counts = {}, {}, {}, {}, {}
    initial_prompts = None
    for variant in VARIANTS_ORDERED:
        set_seed(seed)
        local = copy.deepcopy(cfg); local.model.variant = variant
        model = Model(local, data_info)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        states[variant] = state; hashes[variant] = tensor_hash(state)
        if initial_prompts is None:
            initial_prompts = _prompt_init_summary(model)
        counts[variant] = parameter_counts(model)
        classifier = build_classifier(model.out_dim, data_info["num_classes"], seed)
        classifier_state = {key: value.detach().cpu().clone()
                            for key, value in classifier.state_dict().items()}
        classifiers[variant] = classifier_state
        classifier_hashes[variant] = tensor_hash(classifier_state)
        counts[variant]["classifier_params"] = sum(int(p.numel()) for p in classifier.parameters())
        counts[variant]["total_trainable_params"] = (
            counts[variant]["model_trainable"]
            + sum(int(p.numel()) for p in classifier.parameters() if p.requires_grad)
        )
        del model, classifier

    pairwise = {}
    for i, left in enumerate(VARIANTS_ORDERED):
        for right in VARIANTS_ORDERED[i + 1:]:
            pairwise[f"{left}__{right}"] = (
                states[left].keys() == states[right].keys()
                and all(torch.equal(states[left][name], states[right][name]) for name in states[left])
                and classifiers[left].keys() == classifiers[right].keys()
                and all(torch.equal(classifiers[left][name], classifiers[right][name])
                        for name in classifiers[left])
            )
    model_counts = {row["model_trainable"] for row in counts.values()}
    total_counts = {row["total_trainable_params"] for row in counts.values()}
    classifier_counts = {row["classifier_params"] for row in counts.values()}
    audit = {
        "dataset": dataset, "seed": int(seed), "variant_hashes": hashes,
        "classifier_hashes": classifier_hashes, "pairwise_bitwise_equal": pairwise,
        "all_bitwise_equal": all(pairwise.values()),
        "exact_model_parameter_match": len(model_counts) == 1,
        "exact_classifier_parameter_match": len(classifier_counts) == 1,
        "exact_total_parameter_match": len(total_counts) == 1,
        "parameter_counts": counts, "initial_prompt_summary": initial_prompts,
    }
    if not audit["all_bitwise_equal"] or not audit["exact_model_parameter_match"] \
            or not audit["exact_classifier_parameter_match"] or not audit["exact_total_parameter_match"]:
        raise AssertionError(f"R0 variants failed initialization/capacity fairness: {audit}")
    return audit


def initialization_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, count in audit["parameter_counts"].items():
        rows.append({
            "dataset": audit["dataset"], "seed": audit["seed"], "variant": variant,
            "model_hash": audit["variant_hashes"][variant],
            "classifier_hash": audit["classifier_hashes"][variant],
            "model_params": count["model_trainable"],
            "classifier_params": count["classifier_params"],
            "total_params": count["total_trainable_params"],
            "generic_first_stage_both_modalities": count["generic_first_stage_both_modalities"],
            "protected_mha_both_modalities": count["protected_mha_both_modalities"],
            "common_post_composer_both_modalities": count["common_post_composer_both_modalities"],
            "generic_active_composer_both_modalities": count["generic_active_composer_both_modalities"],
            "protected_active_composer_both_modalities": count["protected_active_composer_both_modalities"],
            "all_variants_bitwise_equal": audit["all_bitwise_equal"],
            "exact_total_parameter_match": audit["exact_total_parameter_match"],
        })
    return rows


def check_provenance() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    remote_ref = subprocess.check_output(
        ["git", "rev-parse", f"refs/remotes/origin/{SOURCE_BRANCH}"], cwd=PROJECT_ROOT, text=True
    ).strip()
    expected_branch = "exp/r0_mature_response_expert_prototype"
    if head != SOURCE_SHA or remote_ref != SOURCE_SHA or branch != expected_branch:
        raise RuntimeError(
            f"R0 provenance mismatch: HEAD={head}, origin/{SOURCE_BRANCH}={remote_ref}, branch={branch}"
        )
    return {"source_branch": SOURCE_BRANCH, "source_sha": SOURCE_SHA,
            "remote_source_sha": remote_ref, "experiment_branch": branch, "head_at_start": head}


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item()) if left.numel() else 0.0


@torch.no_grad()
def smooth_compatibility_regression(dataset: str, seed: int, cfg, data, device: torch.device) -> dict[str, Any]:
    """Compare mapped M0 UNI, N1 SmoothOnly, and R0 SmoothBase common paths."""
    info = model_data_info(data)
    m0_cfg = make_config(dataset, seed, "smooth_base", str(device))
    m0_cfg.model.name = "adaptive_prop_m0"; m0_cfg.model.variant = "uniform"
    n1_cfg = make_config(dataset, seed, "smooth_base", str(device))
    n1_cfg.model.name = "adaptive_prop_n1"; n1_cfg.model.variant = "smooth_only"
    r0_cfg = copy.deepcopy(cfg); r0_cfg.model.variant = "smooth_base"
    set_seed(seed); m0 = M0Model(m0_cfg, info).to(device).eval()
    set_seed(seed); n1 = N1Model(n1_cfg, info).to(device).eval()
    set_seed(seed); r0 = Model(r0_cfg, info).to(device).eval()
    copy_m0_uniform_common_weights(m0, r0)

    x, edge_index = data.x.to(device), data.edge_index.to(device)
    src, dst = remove_self_messages(edge_index)[1:]
    m0_z = m0(x, edge_index)[0]
    n1_z, _, _, _, n1_aux = n1(x, edge_index, return_diagnostics=True)
    r0_z, _, _, _, r0_aux = r0(x, edge_index, return_diagnostics=True)

    # Reconstruct M0 intermediate states using the exact edge-message order.
    x_t, x_v = m0.split_modalities(x)
    m0_i = [m0.proj_t(x_t), m0.proj_v(x_v)]
    degree = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    if dst.numel():
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.long))
    m0_neighbor_mean, m0_smooth_delta, m0_structural = [], [], []
    for modality in range(2):
        aggregate = m0_i[modality].new_zeros(m0_i[modality].shape)
        if src.numel():
            aggregate.index_add_(0, dst, m0.w0[modality](m0_i[modality][src]))
        delta = aggregate / degree.clamp_min(1).to(aggregate.dtype).unsqueeze(-1)
        low_sum = m0_i[modality].new_zeros(m0_i[modality].shape)
        if src.numel():
            low_sum.index_add_(0, dst, m0_i[modality][src])
        neighbor_mean = low_sum / degree.clamp_min(1).to(low_sum.dtype).unsqueeze(-1)
        m0_neighbor_mean.append(neighbor_mean)
        m0_smooth_delta.append(delta)
        m0_structural.append(m0.residual_norms[modality](m0_i[modality] + delta))

    # N1 owns the same initial common pathway. Map it separately to audit both baselines.
    r0_n1 = Model(r0_cfg, info).to(device).eval()
    copy_n1_smooth_common_weights(n1, r0_n1)
    r0_n1_z, _, _, _, r0_n1_aux = r0_n1(x, edge_index, return_diagnostics=True)

    comparisons = {
        "m0_h0_text": _max_abs(m0_i[0], r0_aux["intrinsic"][0]),
        "m0_h0_visual": _max_abs(m0_i[1], r0_aux["intrinsic"][1]),
        "m0_neighbor_mean_text": _max_abs(m0_neighbor_mean[0], r0_aux["neighbor_mean"][0]),
        "m0_neighbor_mean_visual": _max_abs(m0_neighbor_mean[1], r0_aux["neighbor_mean"][1]),
        "m0_smooth_context_text": _max_abs(m0_smooth_delta[0], r0_aux["smooth_delta"][0]),
        "m0_smooth_context_visual": _max_abs(m0_smooth_delta[1], r0_aux["smooth_delta"][1]),
        "m0_structural_text": _max_abs(m0_structural[0], r0_aux["structural"][0]),
        "m0_structural_visual": _max_abs(m0_structural[1], r0_aux["structural"][1]),
        "m0_final_z": _max_abs(m0_z, r0_z),
        "n1_h0_text": _max_abs(n1_aux["h0"][0], r0_n1_aux["intrinsic"][0]),
        "n1_h0_visual": _max_abs(n1_aux["h0"][1], r0_n1_aux["intrinsic"][1]),
        "n1_smooth_context_text": _max_abs(n1_aux["contexts"][0][0], r0_n1_aux["smooth_delta"][0]),
        "n1_smooth_context_visual": _max_abs(n1_aux["contexts"][1][0], r0_n1_aux["smooth_delta"][1]),
        "n1_structural_text": _max_abs(n1_aux["h_tilde"][0], r0_n1_aux["structural"][0]),
        "n1_structural_visual": _max_abs(n1_aux["h_tilde"][1], r0_n1_aux["structural"][1]),
        "n1_final_z": _max_abs(n1_z, r0_n1_z),
    }
    tolerance = {"rtol": 1e-6, "atol": 1e-5}
    max_error = max(comparisons.values())
    result = {
        "dataset": dataset, "seed": int(seed), "device": str(device),
        "status": "passed" if max_error <= tolerance["atol"] else "failed",
        "max_abs_error": max_error, "tolerance": tolerance,
        "comparisons_max_abs_error": comparisons,
        "note": "Linear transform and neighbor mean commute mathematically; CUDA scatter order introduces small floating-point differences.",
    }
    del m0, n1, r0, r0_n1
    if result["status"] != "passed":
        raise AssertionError(f"R0 SmoothBase compatibility regression failed: {result}")
    return result


def _distribution(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: float("nan") for key in ("mean", "std", "median", "q10", "q25", "q75", "q90", "q99")}
    q10, q25, median, q75, q90, q99 = np.quantile(array, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "median": float(median),
            "q10": float(q10), "q25": float(q25), "q75": float(q75), "q90": float(q90), "q99": float(q99)}


def _rms_per_node(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=-1).sqrt()


def _cosine_per_node(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=EPS)


def _diag_row(meta: dict[str, Any], kind: str, modality: str, name: str,
              values: torch.Tensor, **extra) -> dict[str, Any]:
    stats = _distribution(values.detach().cpu().numpy())
    return {**meta, "modality": modality, "kind": kind, "name": name, **stats, **extra}


def collect_diagnostics(model: Model, aux: dict[str, Any], val_idx: torch.Tensor,
                        dataset: str, seed: int, variant: str,
                        initial_prompt_summary: list[dict[str, Any]]) -> dict[str, Any]:
    meta = {"dataset": dataset, "seed": int(seed), "variant": variant,
            "scope": "validation_targets"}
    response_scale, response_distinctness, moe_usage, routing_nodes = [], [], [], []
    prompt_rows, expert_distinctness, cross_response = [], [], []
    protected_attention, attention_difference = [], []
    val_cpu = val_idx.detach().cpu().long()
    for modality, modality_name in enumerate(("text", "visual")):
        i = aux["intrinsic"][modality][val_idx]
        lraw = aux["low_raw"][modality][val_idx]
        hraw = aux["high_raw"][modality][val_idx]
        low = aux["low"][modality][val_idx]
        high = aux["high"][modality][val_idx]
        for name, value in (("RMS_I", i), ("RMS_Lraw", lraw), ("RMS_Hraw", hraw),
                            ("RMS_L", low), ("RMS_H", high)):
            response_scale.append(_diag_row(meta, "rms", modality_name, name, _rms_per_node(value)))
        for left_name, right_name, left, right in (
            ("I", "L", i, low), ("I", "H", i, high), ("L", "H", low, high),
            ("Lraw", "Hraw", lraw, hraw),
        ):
            response_distinctness.append(_diag_row(
                meta, "cosine", modality_name, f"{left_name}_vs_{right_name}",
                _cosine_per_node(left, right),
            ))

        prompt = model.cross_prompts[modality].detach()
        initial = initial_prompt_summary[modality]
        normed_prompt = F.normalize(prompt, dim=-1)
        final_pairs = [float((normed_prompt[a] * normed_prompt[b]).sum().item())
                       for a, b in ((0, 1), (0, 2), (1, 2))]
        prompt_rows.append({
            **meta, "modality": modality_name,
            "initial_cos_01": initial["pairwise_cosines"][0],
            "initial_cos_02": initial["pairwise_cosines"][1],
            "initial_cos_12": initial["pairwise_cosines"][2],
            "final_cos_01": final_pairs[0], "final_cos_02": final_pairs[1],
            "final_cos_12": final_pairs[2],
            "initial_orth_loss": initial["orth_loss"],
            "final_orth_loss": float(prompt_orthogonality_loss(prompt).item()),
            "prompt_norm_0": float(prompt[0].norm().item()),
            "prompt_norm_1": float(prompt[1].norm().item()),
            "prompt_norm_2": float(prompt[2].norm().item()),
        })

        if aux["cross_active"]:
            info = aux["moe"][modality]
            logits = info["router_logits"][val_idx].detach()
            indices = info["top_indices"][val_idx].detach()
            weights = info["top_weights"][val_idx].detach()
            full_prob = torch.softmax(logits, dim=-1)
            top1_counts = torch.bincount(indices[:, 0].cpu(), minlength=NUM_EXPERTS).float()
            top2_counts = torch.bincount(indices.reshape(-1).cpu(), minlength=NUM_EXPERTS).float()
            top1_usage = top1_counts / max(int(indices.size(0)), 1)
            top2_inclusion = top2_counts / max(int(indices.size(0)), 1)
            selected_entropy = -(weights.clamp_min(EPS) * weights.clamp_min(EPS).log()).sum(-1)
            full_entropy = -(full_prob.clamp_min(EPS) * full_prob.clamp_min(EPS).log()).sum(-1)
            usage = {
                **meta, "modality": modality_name,
                "n_validation_nodes": int(indices.size(0)),
                "top1_usage_expert0": float(top1_usage[0]),
                "top1_usage_expert1": float(top1_usage[1]),
                "top1_usage_expert2": float(top1_usage[2]),
                "top2_inclusion_expert0": float(top2_inclusion[0]),
                "top2_inclusion_expert1": float(top2_inclusion[1]),
                "top2_inclusion_expert2": float(top2_inclusion[2]),
                "max_expert_top1_share": float(top1_usage.max()),
                "selected_weight_mean_0": float(weights[:, 0].mean()),
                "selected_weight_mean_1": float(weights[:, 1].mean()),
                "selected_entropy_mean": float(selected_entropy.mean()),
                "full_router_entropy_mean": float(full_entropy.mean()),
                "router_logit_mean_0": float(logits[:, 0].mean()),
                "router_logit_mean_1": float(logits[:, 1].mean()),
                "router_logit_mean_2": float(logits[:, 2].mean()),
                "router_logit_std_0": float(logits[:, 0].std(unbiased=False)),
                "router_logit_std_1": float(logits[:, 1].std(unbiased=False)),
                "router_logit_std_2": float(logits[:, 2].std(unbiased=False)),
            }
            moe_usage.append(usage)
            for local_row, global_node in enumerate(val_cpu.tolist()):
                routing_nodes.append({
                    **meta, "modality": modality_name, "node_id": int(global_node),
                    "router_logit_0": float(logits[local_row, 0]),
                    "router_logit_1": float(logits[local_row, 1]),
                    "router_logit_2": float(logits[local_row, 2]),
                    "top1_expert": int(indices[local_row, 0]),
                    "top2_expert": int(indices[local_row, 1]),
                    "top1_weight": float(weights[local_row, 0]),
                    "top2_weight": float(weights[local_row, 1]),
                })

            experts = info["expert_outputs"][val_idx].detach()
            for expert_idx in range(NUM_EXPERTS):
                expert_distinctness.append(_diag_row(
                    meta, "expert_rms", modality_name, f"expert_{expert_idx}",
                    _rms_per_node(experts[:, expert_idx]),
                ))
            for left_idx, right_idx in ((0, 1), (0, 2), (1, 2)):
                left, right = experts[:, left_idx], experts[:, right_idx]
                norm_den = (left.float().norm(dim=-1) + right.float().norm(dim=-1)).clamp_min(EPS) * 0.5
                expert_distinctness.append(_diag_row(
                    meta, "expert_cosine", modality_name, f"expert_{left_idx}_vs_{right_idx}",
                    _cosine_per_node(left, right),
                ))
                expert_distinctness.append(_diag_row(
                    meta, "expert_normalized_l2", modality_name, f"expert_{left_idx}_vs_{right_idx}",
                    (left.float() - right.float()).norm(dim=-1) / norm_den,
                ))

            xcross = aux["cross"][modality][val_idx]
            x_rms = _rms_per_node(xcross)
            low_rms = _rms_per_node(low).clamp_min(EPS)
            high_rms = _rms_per_node(high).clamp_min(EPS)
            cross_response.append({
                **meta, "modality": modality_name,
                **{f"X_rms_{k}": v for k, v in _distribution(x_rms.cpu().numpy()).items()},
                **{f"X_over_L_{k}": v for k, v in _distribution((x_rms / low_rms).cpu().numpy()).items()},
                **{f"X_over_H_{k}": v for k, v in _distribution((x_rms / high_rms).cpu().numpy()).items()},
                "cos_X_L_median": float(_cosine_per_node(xcross, low).median()),
                "cos_X_H_median": float(_cosine_per_node(xcross, high).median()),
                "cos_X_I_median": float(_cosine_per_node(xcross, i).median()),
                "cos_X_L_q25": float(torch.quantile(_cosine_per_node(xcross, low), 0.25)),
                "cos_X_L_q75": float(torch.quantile(_cosine_per_node(xcross, low), 0.75)),
                "cos_X_H_q25": float(torch.quantile(_cosine_per_node(xcross, high), 0.25)),
                "cos_X_H_q75": float(torch.quantile(_cosine_per_node(xcross, high), 0.75)),
                "cos_X_I_q25": float(torch.quantile(_cosine_per_node(xcross, i), 0.25)),
                "cos_X_I_q75": float(torch.quantile(_cosine_per_node(xcross, i), 0.75)),
            })

        if variant == "bank_crossmoe_protected":
            weights = aux["attention"][modality][val_idx].detach()  # [n, heads, 3]
            head_mean = weights.mean(dim=1)
            node_entropy = -(head_mean.clamp_min(EPS) * head_mean.clamp_min(EPS).log()).sum(-1)
            row: dict[str, Any] = {**meta, "modality": modality_name,
                                   "attention_entropy_mean": float(node_entropy.mean()),
                                   "attention_entropy_std": float(node_entropy.std(unbiased=False))}
            for token_idx, token in enumerate(("L", "H", "X")):
                vals = head_mean[:, token_idx]
                for stat, value in _distribution(vals.cpu().numpy()).items():
                    row[f"attention_{token}_{stat}"] = value
                row[f"attention_{token}_node_std"] = float(vals.std(unbiased=False))
            protected_attention.append(row)

    if variant == "bank_crossmoe_protected":
        text_weights = aux["attention"][0][val_idx].detach().mean(dim=1)
        visual_weights = aux["attention"][1][val_idx].detach().mean(dim=1)
        row = {**meta, "n_validation_nodes": int(val_idx.numel()),
               "attention_text_visual_l1_mean": float((text_weights - visual_weights).abs().sum(-1).mean())}
        for token_idx, token in enumerate(("L", "H", "X")):
            left, right = text_weights[:, token_idx], visual_weights[:, token_idx]
            row[f"attention_{token}_text_mean"] = float(left.mean())
            row[f"attention_{token}_visual_mean"] = float(right.mean())
            row[f"attention_{token}_text_visual_l1"] = float((left - right).abs().mean())
            if left.std(unbiased=False) > 0 and right.std(unbiased=False) > 0:
                row[f"attention_{token}_text_visual_corr"] = float(torch.corrcoef(torch.stack([left, right]))[0, 1])
            else:
                row[f"attention_{token}_text_visual_corr"] = float("nan")
        attention_difference.append(row)

    return {
        "response_scale": response_scale,
        "response_distinctness": response_distinctness,
        "moe_usage": moe_usage,
        "routing_node_rows": routing_nodes,
        "prompt_specialization": prompt_rows if aux["cross_active"] else [],
        "expert_distinctness": expert_distinctness,
        "cross_response": cross_response,
        "protected_attention": protected_attention,
        "modality_attention_difference": attention_difference,
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
def evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes,
             response_overrides=None, routing_override=None) -> dict[str, float]:
    model.eval(); classifier.eval()
    z = model(x, edge_index, response_overrides=response_overrides,
              routing_override=routing_override)[0]
    return metric_from_embeddings(classifier, z, val_idx, val_y, num_classes)


def _write_router_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _initial_model_hash(model: Model) -> str:
    return tensor_hash({key: value for key, value in model.state_dict().items()})


def _gradient_audit(model: nn.Module) -> dict[str, dict[str, Any]]:
    groups = {
        "projectors": ("proj_t.", "proj_v."),
        "low_high_bank": ("low_transforms.", "high_transforms.",
                          "low_response_norms.", "high_response_norms."),
        "cross_prompts": ("cross_prompts.",),
        "cross_experts": ("cross_experts.",),
        "routers": ("routers.",),
        "generic_composer": ("generic_composers.",),
        "protected_attention": ("protected_attention.",),
        "post_composer": ("response_residual_norms.", "post_ffns.", "final_residual_norms."),
        "smooth_residual": ("smooth_residual_norms.",),
        "fusion": ("fusion.",),
    }
    result = {}
    for group, prefixes in groups.items():
        params = [(name, parameter) for name, parameter in model.named_parameters()
                  if name.startswith(prefixes)]
        grads = [parameter.grad.detach() for _, parameter in params if parameter.grad is not None]
        result[group] = {
            "parameter_tensors": len(params), "gradient_tensors": len(grads),
            "finite": all(bool(torch.isfinite(grad).all()) for grad in grads),
            "norm": float(torch.sqrt(sum(grad.float().square().sum() for grad in grads)).item()) if grads else 0.0,
        }
    for modality in range(2):
        for expert in range(NUM_EXPERTS):
            prefix = f"cross_experts.{modality}.{expert}."
            params = [(name, parameter) for name, parameter in model.named_parameters()
                      if name.startswith(prefix)]
            grads = [parameter.grad.detach() for _, parameter in params if parameter.grad is not None]
            result[f"cross_expert_modality{modality}_expert{expert}"] = {
                "parameter_tensors": len(params), "gradient_tensors": len(grads),
                "has_gradient": bool(grads),
                "finite": all(bool(torch.isfinite(grad).all()) for grad in grads),
                "norm": float(torch.sqrt(sum(grad.float().square().sum() for grad in grads)).item()) if grads else 0.0,
            }
    return result


@torch.no_grad()
def initial_router_usage(dataset: str, seed: int, cfg, data, val_idx: torch.Tensor,
                         device: torch.device) -> dict[str, Any]:
    """Audit non-tied node-level Top-2 selection before any optimizer update."""
    local_cfg = copy.deepcopy(cfg)
    local_cfg.model.variant = "bank_crossmoe_generic"
    set_seed(seed)
    model = Model(local_cfg, model_data_info(data)).to(device).eval()
    _, _, _, _, aux = model(data.x.to(device), data.edge_index.to(device), return_diagnostics=True)
    rows = []
    for modality, item in enumerate(aux["moe"]):
        logits = item["router_logits"][val_idx]
        indices = item["top_indices"][val_idx]
        unique_pairs = sorted({tuple(map(int, pair)) for pair in indices.detach().cpu().tolist()})
        top1 = torch.bincount(indices[:, 0].detach().cpu(), minlength=NUM_EXPERTS)
        top2 = torch.bincount(indices.reshape(-1).detach().cpu(), minlength=NUM_EXPERTS)
        final_weight = model.routers[modality][-1].weight.detach()
        sorted_logits = logits.sort(dim=-1).values
        min_top_gap = float((sorted_logits[:, -1] - sorted_logits[:, -2]).abs().min().item())
        if not bool(torch.isfinite(logits).all()) or float(logits.std(unbiased=False)) == 0.0:
            raise AssertionError("R0 initial router logits are constant or non-finite")
        if not bool(torch.isfinite(final_weight).all()) or torch.count_nonzero(final_weight) == 0:
            raise AssertionError("R0 router output layer has a zero or non-finite initialization")
        if bool(((sorted_logits[:, -1] - sorted_logits[:, -2]) == 0).any()):
            raise AssertionError("R0 initial router has exact Top-2 ties on validation nodes")
        rows.append({
            "modality": "text" if modality == 0 else "visual",
            "router_final_weight_std": float(final_weight.std(unbiased=False).item()),
            "router_logit_std": float(logits.std(unbiased=False).item()),
            "minimum_top2_boundary_gap": min_top_gap,
            "unique_selected_pairs": unique_pairs,
            "top1_counts": [int(value) for value in top1.tolist()],
            "top2_inclusion_counts": [int(value) for value in top2.tolist()],
            "top1_usage_fraction": [float(value / max(int(indices.size(0)), 1)) for value in top1.tolist()],
            "top2_inclusion_fraction": [float(value / max(int(indices.size(0)), 1)) for value in top2.tolist()],
        })
    del model, aux
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"dataset": dataset, "seed": int(seed), "scope": "validation_nodes_before_training", "modalities": rows}


def _intervention_rows(model: Model, classifier: nn.Module, x: torch.Tensor,
                       edge_index: torch.Tensor, val_idx: torch.Tensor, val_y: torch.Tensor,
                       num_classes: int, base_metrics: dict[str, float],
                       aux: dict[str, Any], dataset: str, seed: int) -> list[dict[str, Any]]:
    output = []

    def record(name: str, metrics: dict[str, float], repeat_seed: int | None = None):
        output.append({
            "dataset": dataset, "seed": int(seed), "variant": "bank_crossmoe_protected",
            "intervention": name, "repeat_seed": repeat_seed,
            "delta_accuracy_pp": 100.0 * (metrics["val_accuracy"] - base_metrics["val_accuracy"]),
            "delta_macro_f1_pp": 100.0 * (metrics["val_macro_f1"] - base_metrics["val_macro_f1"]),
            "delta_ce": metrics["val_ce"] - base_metrics["val_ce"],
        })

    # Identity overrides verify that diagnostic injection preserves the checkpoint.
    identity_low = tuple(value.detach().clone() for value in aux["low"])
    identity_high = tuple(value.detach().clone() for value in aux["high"])
    identity_cross = tuple(value.detach().clone() for value in aux["cross"])
    identity_routes = tuple((item["top_indices"].detach().clone(), item["top_weights"].detach().clone())
                            for item in aux["moe"])
    identity = evaluate(
        model, classifier, x, edge_index, val_idx, val_y, num_classes,
        response_overrides={"L": identity_low, "H": identity_high, "X": identity_cross},
        routing_override=identity_routes,
    )
    identity_error = max(abs(identity[key] - base_metrics[key]) for key in base_metrics)
    if identity_error > 1e-6:
        raise AssertionError(f"R0 control-override identity failed: {identity_error}")
    record("identity", identity)

    x_zero = [value.detach().clone() for value in aux["cross"]]
    for value in x_zero:
        value[val_idx] = 0
    record("X_off", evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes,
                              response_overrides={"X": tuple(x_zero)}))
    for name, key in (("H_off", "high"), ("L_off", "low")):
        values = [value.detach().clone() for value in aux[key]]
        for value in values:
            value[val_idx] = 0
        record(name, evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes,
                              response_overrides={"H" if key == "high" else "L": tuple(values)}))

    for repeat_seed in ROUTER_SHUFFLE_SEEDS:
        overrides = []
        for modality, item in enumerate(aux["moe"]):
            generator = torch.Generator(device="cpu").manual_seed(repeat_seed + modality * 1009)
            permutation = torch.randperm(int(val_idx.numel()), generator=generator).to(val_idx.device)
            indices = item["top_indices"].detach().clone()
            weights = item["top_weights"].detach().clone()
            indices[val_idx] = item["top_indices"].detach()[val_idx[permutation]]
            weights[val_idx] = item["top_weights"].detach()[val_idx[permutation]]
            overrides.append((indices, weights))
        record("router_tuple_shuffle", evaluate(
            model, classifier, x, edge_index, val_idx, val_y, num_classes,
            routing_override=tuple(overrides),
        ), repeat_seed)
    return output


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
    runtime_init_hash = _initial_model_hash(model)
    if runtime_init_hash != init_audit["variant_hashes"][variant]:
        raise AssertionError(f"R0 runtime model initialization differs from audit for {dataset}/{seed}/{variant}")
    classifier = build_classifier(model.out_dim, info["num_classes"], seed, device)
    runtime_classifier_hash = tensor_hash(classifier.state_dict())
    if runtime_classifier_hash != init_audit["classifier_hashes"][variant]:
        raise AssertionError("R0 runtime classifier initialization differs from audit")
    # Added module construction must not perturb the classifier or training stream.
    set_seed(seed)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    train_idx, val_idx = splits["train"].to(device), splits["validation"].to(device)
    labels = data.y.to(device)
    train_y, val_y = labels[train_idx], labels[val_idx]
    if bool((train_y < 0).any()) or bool((val_y < 0).any()) or data.test_idx is not None:
        raise AssertionError("R0 train/validation labels are invalid or test indices are attached")
    if bool(cfg.task.evaluate_test):
        raise AssertionError("R0 requires task.evaluate_test=false")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_acc, best_epoch = -1.0, 0
    best_model = best_classifier = best_metrics = None
    patience = int(cfg.task.patience); patience_left = patience
    epoch_rows, first_gradients = [], {}
    started = time.perf_counter(); epochs_run = 0
    for epoch in range(1, int(cfg.task.epochs) + 1):
        epochs_run = epoch
        model.train(); classifier.train(); optimizer.zero_grad(set_to_none=True)
        z, _, _, orth_raw, _ = model(x, edge_index, return_diagnostics=False)
        # The fixed R0 auxiliary coefficient applies only to the two CrossMoE variants.
        logits = classifier(z[train_idx])
        task_loss = F.cross_entropy(logits, train_y)
        orth_weighted = model.orth_weight * orth_raw
        loss = task_loss + orth_weighted
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite R0 loss at {dataset}/{seed}/{variant}/epoch{epoch}")
        loss.backward()
        if epoch == 1:
            first_gradients = _gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()),
            max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True,
        )
        optimizer.step(); scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        epoch_rows.append({
            "epoch": epoch, "task_loss_ce": float(task_loss.detach()),
            "orth_loss_raw": float(orth_raw.detach()), "orth_loss_weighted": float(orth_weighted.detach()),
            "orth_to_task_ratio": float((orth_weighted.detach() / task_loss.detach().clamp_min(EPS))),
        })
        del z, logits, task_loss, orth_raw, orth_weighted, loss

        model.eval(); classifier.eval()
        with torch.no_grad():
            val_z = model(x, edge_index, return_diagnostics=False)[0]
            metrics = metric_from_embeddings(classifier, val_z, val_idx, val_y, info["num_classes"])
        if metrics["val_accuracy"] > best_acc + float(cfg.task.early_stop_min_delta):
            best_acc, best_epoch, best_metrics = metrics["val_accuracy"], epoch, metrics
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_classifier = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            patience_left = patience
        elif epoch >= int(cfg.task.early_stop_min_epoch):
            patience_left -= 1
            if patience_left <= 0:
                break
        del val_z
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    if best_model is None or best_classifier is None or best_metrics is None:
        raise RuntimeError("R0 training ended without a validation-selected checkpoint")
    model.load_state_dict(best_model, strict=True); classifier.load_state_dict(best_classifier, strict=True)
    model.eval(); classifier.eval()

    with torch.no_grad():
        z, _, _, orth_raw_best, aux = model(x, edge_index, return_diagnostics=True)
        selected_metrics = metric_from_embeddings(classifier, z, val_idx, val_y, info["num_classes"])
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("R0 selected checkpoint embeddings are non-finite")
        response_complement_max_error = 0.0
        for modality in range(2):
            identity_error = (aux["low_raw"][modality] + aux["high_raw"][modality]
                              - aux["intrinsic"][modality]).abs().max().item()
            if not math.isfinite(identity_error) or identity_error > 2e-6:
                raise AssertionError(f"R0 low/high complement identity failed: {identity_error}")
            response_complement_max_error = max(response_complement_max_error, identity_error)
        for module_info in aux["moe"]:
            if not bool(torch.isfinite(module_info["router_logits"]).all()):
                raise FloatingPointError("R0 router logits are non-finite")
            if not bool(torch.isfinite(module_info["top_weights"]).all()):
                raise FloatingPointError("R0 selected router weights are non-finite")
    diagnostics = collect_diagnostics(
        model, aux, val_idx, dataset, seed, variant, init_audit["initial_prompt_summary"]
    )
    diagnostic_summaries = {key: value for key, value in diagnostics.items() if key != "routing_node_rows"}
    routing_rows = diagnostics["routing_node_rows"]
    if routing_rows:
        route_path = out_dir / "router_nodes" / dataset / f"seed_{seed}_{variant}.csv.gz"
        _write_router_rows(route_path, routing_rows)
    intervention_rows = []
    intervention_seconds = 0.0
    if variant == "bank_crossmoe_protected":
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_start = time.perf_counter()
        intervention_rows = _intervention_rows(
            model, classifier, x, edge_index, val_idx, val_y, info["num_classes"],
            selected_metrics, aux, dataset, seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_seconds = time.perf_counter() - intervention_start
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    orth_weighted_best = model.orth_weight * orth_raw_best
    best_epoch_loss = next(item for item in epoch_rows if item["epoch"] == best_epoch)
    row = {
        "status": "completed", "dataset": dataset, "seed": int(seed), "variant": variant,
        "smoke": bool(smoke), "best_epoch": best_epoch, "epochs_run": epochs_run,
        **selected_metrics,
        "model_params": sum(int(p.numel()) for p in model.parameters()),
        "classifier_params": sum(int(p.numel()) for p in classifier.parameters()),
        "total_params": sum(int(p.numel()) for p in model.parameters())
                         + sum(int(p.numel()) for p in classifier.parameters()),
        "active_capacity": parameter_counts(model),
        "peak_gpu_memory_bytes": peak_bytes,
        "training_time_sec": training_seconds, "intervention_time_sec": intervention_seconds,
        "orth_weight": model.orth_weight,
        "response_complement_max_error": response_complement_max_error,
        "best_epoch_task_loss_ce": best_epoch_loss["task_loss_ce"],
        "best_epoch_orth_loss_raw": float(orth_raw_best),
        "best_epoch_orth_loss_weighted": float(orth_weighted_best),
        "best_epoch_orth_to_task_ratio": float(orth_weighted_best / max(best_epoch_loss["task_loss_ce"], EPS)),
        "orth_loss_by_epoch": epoch_rows,
        "gradient_audit_epoch1": first_gradients,
        "diagnostics": diagnostic_summaries, "interventions": intervention_rows,
        "initialization_audit": init_audit,
        "runtime_model_init_hash": runtime_init_hash,
        "runtime_classifier_init_hash": runtime_classifier_hash,
        "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        "test_idx_attached": False, "test_labels_exposed": False, "link_prediction": False,
        "validation_label_indices_only": True, "data_info": info,
    }
    run_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    write_json(run_path, row)
    if save_checkpoint:
        checkpoint_path = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": best_model, "classifier_state": best_classifier,
            "data_info": info, "dataset": dataset, "seed": int(seed), "variant": variant,
            "best_epoch": best_epoch, "best_val_metrics": selected_metrics,
            "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        }, checkpoint_path)
        row["checkpoint"] = str(checkpoint_path.relative_to(PROJECT_ROOT))
        write_json(run_path, row)
    print(f"[R0 done] {dataset}/{seed}/{variant} epoch={best_epoch} "
          f"acc={selected_metrics['val_accuracy']:.4f} f1={selected_metrics['val_macro_f1']:.4f} "
          f"ce={selected_metrics['val_ce']:.4f} train={training_seconds:.1f}s "
          f"interventions={intervention_seconds:.1f}s peak={peak_bytes / 1024**3:.2f}GiB", flush=True)
    del z, aux, data, x, edge_index, model, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def run_smoke(device_name: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    dataset, seed = "Movies", 42
    cfg = make_config(dataset, seed, "smooth_base", device_name, epochs=1)
    data = load_m0_data(cfg, seed); splits = restrict_labels_to_train_val(data)
    info = model_data_info(data)
    audit = audit_initialization(dataset, seed, info, cfg)
    device = torch.device(device_name)
    regression = smooth_compatibility_regression(dataset, seed, cfg, data, device)
    router_init = initial_router_usage(dataset, seed, cfg, data,
                                       splits["validation"].to(device), device)
    smoke_rows = []
    for variant in VARIANTS_ORDERED:
        row = run_one(dataset, seed, variant, device_name, out_dir=out_dir / "smoke_artifacts",
                      epochs=1, save_checkpoint=True, preloaded_data=data,
                      init_audit=audit, smoke=True)
        expected_active = {"smooth_base": {"projectors", "smooth_residual", "fusion"},
                           "bank_generic": {"projectors", "low_high_bank", "generic_composer", "post_composer", "fusion"},
                           "bank_crossmoe_generic": {"projectors", "low_high_bank", "cross_prompts", "cross_experts", "routers", "generic_composer", "post_composer", "fusion"},
                           "bank_crossmoe_protected": {"projectors", "low_high_bank", "cross_prompts", "cross_experts", "routers", "protected_attention", "post_composer", "fusion"}}[variant]
        gradients = row["gradient_audit_epoch1"]
        for group in expected_active:
            item = gradients[group]
            if item["gradient_tensors"] < 1 or not item["finite"] or item["norm"] <= 0:
                raise AssertionError(f"R0 {variant} active gradient group failed: {group}={item}")
        if variant in {"bank_crossmoe_generic", "bank_crossmoe_protected"}:
            for modality in range(2):
                for expert in range(NUM_EXPERTS):
                    item = gradients[f"cross_expert_modality{modality}_expert{expert}"]
                    if not item["finite"]:
                        raise AssertionError(f"R0 expert gradient is non-finite: {item}")
        if variant in {"bank_crossmoe_generic", "bank_crossmoe_protected"}:
            top1_used = set()
            for item in row["diagnostics"]["moe_usage"]:
                top1_used.update(i for i in range(NUM_EXPERTS)
                                 if item[f"top1_usage_expert{i}"] > 0)
            if not top1_used:
                raise AssertionError("R0 CrossMoE smoke selected no expert")
        smoke_rows.append({
            "variant": variant, "status": row["status"],
            "gradient_audit_epoch1": gradients,
            "orth_loss": row["best_epoch_orth_loss_raw"],
            "active_capacity": row["active_capacity"],
        })
    result = {
        "dataset": dataset, "seed": seed, "device": device_name,
        "initialization_fairness": audit,
        "smooth_m0_n1_regression": regression,
        "initial_router_usage": router_init,
        "four_variant_one_epoch_training_smoke": smoke_rows,
        "status": "passed", "evaluate_test": False,
    }
    write_json(out_dir / "smoke" / "movies42.json", result)
    return result


def run_campaign(args) -> None:
    provenance = check_provenance()
    if args.smoke:
        run_smoke(args.device, Path(args.out_dir)); return
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_records, errors, retries, capacity_rows, init_rows = [], [], [], [], []
    regression = None; started = time.perf_counter()
    if args.regression:
        cfg = make_config("Movies", 42, "smooth_base", args.device)
        data = load_m0_data(cfg, 42); restrict_labels_to_train_val(data)
        regression = smooth_compatibility_regression("Movies", 42, cfg, data, torch.device(args.device))
        write_json(out_dir / "regression" / "smooth_m0_n1_movies42.json", regression)
        del data
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg = make_config(dataset, seed, "smooth_base", args.device, args.epochs)
            data = load_m0_data(cfg, seed); restrict_labels_to_train_val(data)
            info = model_data_info(data)
            audit = audit_initialization(dataset, seed, info, cfg)
            init_rows.extend(initialization_rows(audit))
            for variant in VARIANTS_ORDERED:
                record_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if record_path.exists():
                    try:
                        existing = json.loads(record_path.read_text(encoding="utf-8"))
                        if existing.get("status") == "completed" and not existing.get("smoke", False):
                            run_records.append(existing)
                            capacity_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                                 **existing["active_capacity"]})
                            continue
                        if not args.retry_failed:
                            errors.append({"dataset": dataset, "seed": seed, "variant": variant,
                                           "status": "existing_noncompleted_record_not_retried"})
                            continue
                        retries.append({"dataset": dataset, "seed": seed, "variant": variant,
                                        "previous_status": existing.get("status", "unknown"),
                                        "retry_attempt": True})
                    except Exception:
                        if not args.retry_failed:
                            errors.append({"dataset": dataset, "seed": seed, "variant": variant,
                                           "status": "existing_record_unreadable_not_retried"})
                            continue
                        retries.append({"dataset": dataset, "seed": seed, "variant": variant,
                                        "previous_status": "unreadable_record", "retry_attempt": True})
                try:
                    result = run_one(dataset, seed, variant, args.device, out_dir=out_dir,
                                     epochs=args.epochs, save_checkpoint=True,
                                     preloaded_data=data, init_audit=audit, smoke=False)
                    run_records.append(result)
                    capacity_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                         **result["active_capacity"]})
                except Exception as exc:
                    failure = {"dataset": dataset, "seed": seed, "variant": variant,
                               "status": "failed", "error": repr(exc),
                               "traceback": traceback.format_exc(),
                               "retry_attempt": bool(args.retry_failed)}
                    errors.append(failure)
                    write_json(out_dir / "failures" / dataset / f"seed_{seed}_{variant}.json", failure)
                    print(f"[R0 failed] {dataset}/{seed}/{variant}: {exc!r}", flush=True)
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    manifest = {
        **provenance, "datasets": list(DATASETS), "seeds": list(SEEDS),
        "variants": list(VARIANTS_ORDERED), "expected_runs": 36,
        "completed_runs": sum(row.get("status") == "completed" for row in run_records),
        "failures": errors, "retries": retries, "retry_failed_requested": bool(args.retry_failed),
        "runtime_seconds": elapsed, "device": args.device,
        "epochs_override": args.epochs, "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False, "test_labels_exposed": False, "link_prediction": False,
        "orthogonality_weight": ORTH_WEIGHT, "router_tuple_shuffle_seeds": list(ROUTER_SHUFFLE_SEEDS),
        "smooth_compatibility_regression": regression,
        "training_config": OmegaConf.to_container(make_config("Movies", 42, "smooth_base", args.device).task,
                                                    resolve=True),
    }
    write_json(out_dir / "campaign_manifest.json", manifest)
    write_json(out_dir / "failures.json", errors)
    write_csv(RESEARCH_DIR / "data" / "parameter_init_audit.csv", init_rows)
    write_csv(RESEARCH_DIR / "data" / "active_capacity_audit.csv", capacity_rows)
    if run_records:
        from scripts.analyze_r0_mature_response_expert_prototype import analyze_campaign
        analyze_campaign(out_dir, RESEARCH_DIR, manifest)
    print(f"[R0 campaign] completed={manifest['completed_runs']}/36 failed={len(errors)} "
          f"elapsed={elapsed:.1f}s device={args.device}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Explicitly retry prior failed/unreadable records")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    if not args.smoke and not args.campaign:
        parser.error("choose --smoke or --campaign")
    run_campaign(args)


if __name__ == "__main__":
    main()
