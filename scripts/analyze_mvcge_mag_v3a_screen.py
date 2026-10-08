#!/usr/bin/env python3
"""Analyze V3A validation-selected checkpoints without loading labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "C0_raw",
    "C1_effective",
    "C2_dual_shared_profile",
    "C3_dual_context_profile",
)
PAIRS = (
    ("C1_effective", "C0_raw"),
    ("C2_dual_shared_profile", "C0_raw"),
    ("C2_dual_shared_profile", "C1_effective"),
    ("C3_dual_context_profile", "C2_dual_shared_profile"),
    ("C3_dual_context_profile", "C0_raw"),
)
EXPERT_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v3a_effective_context_screen"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v3a_effective_context_screen"
DATA_ROOT = RESEARCH_ROOT / "data"
V22_ROOT = ROOT / "research" / "mvcge_mag_v22_structure_grounded_router"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, numbers.Real):
        return math.isfinite(float(value))
    # CSV identifier/string fields (dataset, variant, index lists) are not
    # numeric measurements and should remain intact.
    return isinstance(value, str)


def _stats(value: torch.Tensor) -> dict[str, float | None]:
    value = value.detach().reshape(-1).float()
    if value.numel() == 0:
        return {k: None for k in ("mean", "std", "p10", "p50", "p90")}
    quantiles = torch.quantile(value, torch.tensor([0.1, 0.5, 0.9], device=value.device))
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def _rms(value: torch.Tensor, active: torch.Tensor | None = None) -> float:
    if active is not None:
        value = value[active]
    if value.numel() == 0:
        return 0.0
    return float(value.float().square().mean().sqrt().item())


def _condition_number(value: torch.Tensor) -> float | None:
    condition = float(torch.linalg.cond(value).item())
    return condition if math.isfinite(condition) else None


def _cosine_flat(a: torch.Tensor, b: torch.Tensor, active: torch.Tensor | None = None) -> float | None:
    if active is not None:
        a, b = a[active], b[active]
    if a.numel() == 0:
        return None
    value = F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0, eps=1.0e-8)
    return float(value.clamp(-1.0, 1.0).item())


def _cosine_mean_node(a: torch.Tensor, b: torch.Tensor, active: torch.Tensor) -> float | None:
    if not bool(active.any()):
        return None
    value = F.cosine_similarity(a[active].float(), b[active].float(), dim=-1, eps=1.0e-8)
    return float(value.clamp(-1.0, 1.0).mean().item())


def _compose_dataset_config(dataset: str, seed: int):
    # This composes paths/config only. The imported loader below reads features
    # and physical edges, and deliberately does not touch labels or splits.
    from scripts.analyze_sosb_mag_v15_basis_screen import _compose_config

    return _compose_config(dataset, seed)


def load_features_and_edges(dataset: str, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    from scripts.analyze_sosb_mag_v15_basis_screen import load_features_and_edges as load

    return load(dataset, seed)


def _feature_dims(dataset: str, seed: int) -> tuple[int, int, int]:
    import numpy as np

    cfg = _compose_dataset_config(dataset, seed)
    ds = cfg.dataset
    if str(ds.source).lower() == "magb":
        text_dim = int(np.load(str(ds.text_feat_path), mmap_mode="r", allow_pickle=False).shape[1])
        visual_dim = int(np.load(str(ds.image_feat_path), mmap_mode="r", allow_pickle=False).shape[1])
    else:
        text_dim, visual_dim = int(ds.text_dim), int(ds.visual_dim)
    return text_dim + visual_dim, text_dim, visual_dim


def _model_config(variant: str):
    cfg = OmegaConf.load(ROOT / "configs" / "model" / "mvcge_mag_v3a.yaml")
    cfg.variant = variant
    return OmegaConf.create({"model": OmegaConf.to_container(cfg, resolve=True)})


def operator_diagnostics(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    dataset: str,
    seed: int,
    model,
    info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Summarize sparse raw/effective edge vectors and modality discrepancy."""
    if info is not None:
        operators = info["operators"]
        raw_weight = info["details"]["text"]["raw_operator_weight"]
        src, dst, _, _, _ = model._normalized_operator(edge_index, x.size(0), x.dtype)
    else:
        inputs = model._split_modalities(x)
        src, dst, raw_weight, _, _ = model._normalized_operator(edge_index, x.size(0), x.dtype)
        operators = {
            modality: model._effective_operator(inputs[modality], src, dst)
            for modality in ("text", "visual")
        }
    text_eff = operators["text"]["norm_weight"]
    visual_eff = operators["visual"]["norm_weight"]
    denominator = torch.linalg.vector_norm(raw_weight) + model.eps
    tv_delta = torch.linalg.vector_norm(text_eff - visual_eff)
    tv_cos = F.cosine_similarity(text_eff.float(), visual_eff.float(), dim=0, eps=model.eps).clamp(-1.0, 1.0)
    rows = []
    for modality in ("text", "visual"):
        op = operators[modality]
        sim = _stats(op["semantic_similarity"])
        degree = _stats(op["effective_degree"])
        eff_weight = op["norm_weight"]
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "semantic_similarity_mean": sim["mean"],
                "semantic_similarity_std": sim["std"],
                "semantic_similarity_p10": sim["p10"],
                "semantic_similarity_p50": sim["p50"],
                "semantic_similarity_p90": sim["p90"],
                "fraction_lt_025": float((op["semantic_similarity"] < 0.25).float().mean().item()) if src.numel() else 0.0,
                "fraction_lt_050": float((op["semantic_similarity"] < 0.50).float().mean().item()) if src.numel() else 0.0,
                "fraction_gt_075": float((op["semantic_similarity"] > 0.75).float().mean().item()) if src.numel() else 0.0,
                "raw_norm_weight_l2": float(torch.linalg.vector_norm(raw_weight).item()),
                "effective_norm_weight_l2": float(torch.linalg.vector_norm(eff_weight).item()),
                "operator_weight_delta_l2": float((torch.linalg.vector_norm(eff_weight - raw_weight) / denominator).item()),
                "operator_weight_cosine": _cosine_flat(raw_weight, eff_weight),
                "effective_degree_mean": degree["mean"],
                "effective_degree_std": degree["std"],
                "effective_degree_p10": degree["p10"],
                "effective_degree_p50": degree["p50"],
                "effective_degree_p90": degree["p90"],
                "text_visual_eff_operator_delta_l2": float(tv_delta.item()),
                "text_visual_eff_operator_relative_delta_l2": float((tv_delta / denominator).item()),
                "text_visual_eff_operator_cosine": float(tv_cos.item()),
                "physical_edge_count": int(src.numel()),
            }
        )
    return rows


def run_label_free_preflight(device: str = "cuda:0") -> list[dict[str, Any]]:
    from src.models.mvcge_mag_v3a import Model

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        x, edge = load_features_and_edges(dataset, 42)
        input_dim, text_dim, visual_dim = _feature_dims(dataset, 42)
        if input_dim != x.size(1):
            raise ValueError(f"Feature dimensions disagree for {dataset}: {input_dim} != {x.size(1)}")
        data_info = {
            "input_dim": input_dim,
            "text_dim": text_dim,
            "visual_dim": visual_dim,
            "num_nodes": int(x.size(0)),
            "num_classes": int(_compose_dataset_config(dataset, 42).dataset.num_classes),
        }
        model = Model(_model_config("C0_raw"), data_info).to(device).eval()
        x, edge = x.to(device), edge.to(device)
        rows.extend(operator_diagnostics(x, edge, dataset=dataset, seed=42, model=model))
        del model, x, edge
        if "cuda" in device:
            torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("Label-free preflight produced no operator records")
    return rows


def preflight_is_degenerate(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        float(row["operator_weight_delta_l2"]) <= 1.0e-6 for row in rows
    )


def _checkpoint_paths(row: dict[str, Any], *, root: Path, output_root: Path):
    run_dir = output_root / str(row.get("mode", "formal")) / "runs" / row["dataset"] / f"seed_{int(row['seed'])}" / row["variant"]
    checkpoint_path = root / row["checkpoint_path"]
    return run_dir, checkpoint_path


@torch.no_grad()
def audit_checkpoint(
    row: dict[str, Any],
    *,
    root: Path = ROOT,
    output_root: Path = OUTPUT_ROOT,
    device_name: str = "cuda:0",
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Inspect one selected checkpoint using features, graph, weights and metadata only."""
    from src.models.mvcge_mag_v3a import Model

    dataset, seed, variant = row["dataset"], int(row["seed"]), row["variant"]
    run_dir, checkpoint_path = _checkpoint_paths(row, root=root, output_root=output_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    if cfg.task.get("evaluate_test", True) is not False:
        raise AssertionError(f"Test evaluation was not disabled in {run_dir}")
    cfg.model.variant = variant
    model = Model(cfg, checkpoint["data_info"]).to(device_name).eval()
    model.load_state_dict(checkpoint["model_state"])
    if cached_data is None:
        cached_data = load_features_and_edges(dataset, seed)
    x_cpu, edge_cpu = cached_data
    x, edge = x_cpu.to(device_name), edge_cpu.to(device_name)
    z, _, _, aux, info = model(x, edge, return_details=True)
    test_metrics_present = any(str(key).lower().startswith("test") for key in row["metrics"])
    if test_metrics_present:
        raise AssertionError(f"Test metrics found in run row {dataset}/{seed}/{variant}")

    active = info["active_nodes"]
    max_static_strength_node_difference = 0.0
    finite_keys = (
        "prior", "raw_trajectory", "effective_trajectory", "alpha_raw", "alpha_effective",
        "raw_profiles", "effective_shared_profiles", "effective_context_profiles",
        "mixed_profiles", "expert_outputs", "mixture", "scaled_mixture", "output",
        "semantic_similarity", "semantic_weight", "effective_degree", "selection_logits",
        "dense_probs", "route_weights", "strength",
    )
    for modality in ("text", "visual"):
        item = info["details"][modality]
        for key in finite_keys:
            if not torch.isfinite(item[key]).all():
                raise FloatingPointError(f"Non-finite {key}: {dataset}/{seed}/{variant}/{modality}")
        if bool((item["semantic_weight"] < 0).any() or (item["semantic_weight"] > 1).any()):
            raise AssertionError("Semantic edge weight outside [0,1]")
        route = info["router"][modality]
        if not torch.equal(route["selection_logits"], route["selection_logits"][:1].expand_as(route["selection_logits"])):
            raise AssertionError("V3A route is not modality-static")
        if not torch.equal(
            route["strength_logit"],
            route["strength_logit"][:1].expand_as(route["strength_logit"]),
        ):
            raise AssertionError("V3A strength logits are not modality-static")
        strength_node_difference = float(
            (route["strength"] - route["strength"][:1]).abs().max().item()
        )
        max_static_strength_node_difference = max(
            max_static_strength_node_difference, strength_node_difference
        )
        if not torch.allclose(
            route["strength"],
            route["strength"][:1].expand_as(route["strength"]),
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise AssertionError("V3A strength is not modality-static")
        if not torch.equal((route["route_weights"] > 0).sum(-1), torch.full((x.size(0),), 2, device=x.device)):
            raise AssertionError("V3A Top-2 route does not select exactly two experts")
        if not torch.allclose(route["route_weights"].sum(-1), torch.ones(x.size(0), device=x.device), atol=1.0e-6, rtol=0.0):
            raise AssertionError("V3A Top-2 weights do not sum to one")

    op_rows = operator_diagnostics(x, edge, dataset=dataset, seed=seed, model=model, info=info)
    trajectory_rows = []
    for modality in ("text", "visual"):
        item = info["details"][modality]
        record: dict[str, Any] = {"dataset": dataset, "seed": seed, "variant": variant, "modality": modality}
        for hop in range(4):
            raw, effective = item["raw_trajectory"][hop], item["effective_trajectory"][hop]
            record[f"raw_eff_flattened_cosine_{hop + 1}"] = _cosine_flat(raw, effective, active)
            record[f"raw_eff_relative_rms_delta_{hop + 1}"] = _rms(effective[active] - raw[active]) / (_rms(raw, active) + model.eps)
            record[f"raw_rms_{hop + 1}"] = _rms(raw, active)
            record[f"effective_rms_{hop + 1}"] = _rms(effective, active)
        trajectory_rows.append(record)

    novelty_rows = []
    if variant == "C0_raw":
        for modality in ("text", "visual"):
            raw = info["details"][modality]["raw_trajectory"][:, active].float().reshape(4, -1)
            eff = info["details"][modality]["effective_trajectory"][:, active].float().reshape(4, -1)
            g_raw, g_eff = raw @ raw.T, eff @ eff.T
            cross = raw @ eff.T
            coef = torch.linalg.pinv(g_raw) @ cross
            residual_sq = (eff * eff).sum() - 2.0 * torch.trace(coef.T @ cross) + torch.trace(coef.T @ g_raw @ coef)
            coef_reverse = torch.linalg.pinv(g_eff) @ cross.T
            raw_residual_sq = (raw * raw).sum() - 2.0 * torch.trace(coef_reverse.T @ cross.T) + torch.trace(coef_reverse.T @ g_eff @ coef_reverse)
            novelty_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "effective_outside_raw_span_ratio": float(torch.sqrt(residual_sq.clamp_min(0.0)).item() / (torch.linalg.vector_norm(eff).item() + model.eps)),
                    "raw_outside_effective_span_ratio": float(torch.sqrt(raw_residual_sq.clamp_min(0.0)).item() / (torch.linalg.vector_norm(raw).item() + model.eps)),
                    "raw_gram_condition": _condition_number(g_raw),
                    "effective_gram_condition": _condition_number(g_eff),
                }
            )

    profile_rows = []
    similarity_rows = []
    context_rows = []
    routing_rows = []
    strength_rows = []
    modality_pair = {}
    for modality in ("text", "visual"):
        route = info["router"][modality]
        indices = tuple(sorted(int(v) for v in route["top_indices"][0].tolist()))
        modality_pair[modality] = indices
    used_union = set(modality_pair["text"]) | set(modality_pair["visual"])
    pair_same = modality_pair["text"] == modality_pair["visual"]
    dead_count = model.num_experts - len(used_union)
    alpha0 = torch.tensor(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        dtype=model.alpha_raw.dtype,
        device=model.alpha_raw.device,
    ) * 0.5
    alpha_raw = model._effective_alpha(model.alpha_raw, model.eps)
    alpha_effective = model._effective_alpha(model.alpha_effective_raw, model.eps)
    for a, b in EXPERT_PAIRS:
        similarity_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": "shared", "kind": "raw_alpha_pair", "expert_a": a, "expert_b": b, "flattened_cosine": _cosine_flat(alpha_raw[a], alpha_raw[b]), "mean_node_cosine": None})
        if variant == "C3_dual_context_profile":
            similarity_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": "shared", "kind": "effective_alpha_pair", "expert_a": a, "expert_b": b, "flattened_cosine": _cosine_flat(alpha_effective[a], alpha_effective[b]), "mean_node_cosine": None})

    for expert_id in range(model.num_experts):
        raw_alpha_row = model.alpha_raw.detach()[expert_id]
        effective_alpha_row = model.alpha_effective_raw.detach()[expert_id]
        profile_record = {
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            "expert_id": expert_id,
            **{f"alpha_raw_{k + 1}": float(raw_alpha_row[k].item()) for k in range(4)},
            "raw_alpha_drift_from_init": float(torch.linalg.vector_norm(raw_alpha_row - alpha0[expert_id]).item()),
            "raw_eff_alpha_cosine": _cosine_flat(alpha_raw[expert_id], alpha_effective[expert_id]) if variant == "C3_dual_context_profile" else None,
            **{f"alpha_effective_raw_{k + 1}": float(effective_alpha_row[k].item()) if variant == "C3_dual_context_profile" else None for k in range(4)},
            "eff_alpha_drift_from_init": float(torch.linalg.vector_norm(effective_alpha_row - alpha0[expert_id]).item()) if variant == "C3_dual_context_profile" else None,
        }
        profile_rows.append(profile_record)
        if variant in {"C2_dual_shared_profile", "C3_dual_context_profile"}:
            mix = torch.sigmoid(model.context_mix_raw[expert_id])
            context_record: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "expert_id": expert_id,
                "lambda": float(mix.item()),
            }
            for modality in ("text", "visual"):
                item = info["details"][modality]
                raw_prof = item["raw_profiles"][expert_id][active]
                eff_prof = (item["effective_shared_profiles"] if variant == "C2_dual_shared_profile" else item["effective_context_profiles"])[expert_id][active]
                mixed = item["mixed_profiles"][expert_id][active]
                lam = float(mix.item())
                context_record.update(
                    {
                        f"{modality}_raw_profile_rms": _rms(raw_prof),
                        f"{modality}_effective_profile_rms": _rms(eff_prof),
                        f"{modality}_raw_effective_profile_cosine": _cosine_flat(raw_prof, eff_prof),
                        f"{modality}_raw_contribution_rms": _rms((1.0 - lam) * raw_prof),
                        f"{modality}_effective_contribution_rms": _rms(lam * eff_prof),
                        f"{modality}_mixed_profile_rms": _rms(mixed),
                    }
                )
            context_rows.append(context_record)

    for modality in ("text", "visual"):
        item = info["details"][modality]
        route = info["router"][modality]
        selected = tuple(sorted(int(v) for v in route["top_indices"][0].tolist()))
        routing_record: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            "modality": modality,
            "selected_expert_pair": f"{selected[0]}-{selected[1]}",
            "text_visual_selected_pair_same": pair_same,
            "union_experts_used": len(used_union),
            "union_expert_indices": ",".join(str(v) for v in sorted(used_union)),
            "dead_expert_slots_across_modalities": dead_count,
            "dead_expert_indices": ",".join(str(v) for v in sorted(set(range(model.num_experts)) - used_union)),
            "strength": float(route["strength"][0].item()),
        }
        for expert_id in range(model.num_experts):
            routing_record[f"top2_weight_expert_{expert_id}"] = float(route["route_weights"][0, expert_id].item())
            routing_record[f"dense_prob_expert_{expert_id}"] = float(route["dense_probs"][0, expert_id].item())
        routing_rows.append(routing_record)
        for a, b in EXPERT_PAIRS:
            similarity_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "modality": modality,
                    "kind": "expert_output",
                    "expert_a": a,
                    "expert_b": b,
                    "flattened_cosine": _cosine_flat(item["expert_outputs"][a], item["expert_outputs"][b], active),
                    "mean_node_cosine": _cosine_mean_node(item["expert_outputs"][a], item["expert_outputs"][b], active),
                }
            )
        if variant in {"C2_dual_shared_profile", "C3_dual_context_profile"}:
            effective_profiles = item["effective_shared_profiles"] if variant == "C2_dual_shared_profile" else item["effective_context_profiles"]
            for expert_id in range(model.num_experts):
                similarity_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "modality": modality,
                        "kind": "raw_effective_profile",
                        "expert_a": expert_id,
                        "expert_b": expert_id,
                        "flattened_cosine": _cosine_flat(item["raw_profiles"][expert_id], effective_profiles[expert_id], active),
                        "mean_node_cosine": _cosine_mean_node(item["raw_profiles"][expert_id], effective_profiles[expert_id], active),
                    }
                )
        correction_ratio = _rms(item["scaled_mixture"], active) / (_rms(item["prior"], active) + model.eps)
        strength_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "modality": modality,
                "prior_rms": _rms(item["prior"], active),
                "unscaled_expert_mixture_rms": _rms(item["mixture"], active),
                "scaled_correction_rms": _rms(item["scaled_mixture"], active),
                "scaled_correction_to_prior_rms_ratio": correction_ratio,
                "strength": float(route["strength"][0].item()),
            }
        )

    result: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "finite": bool(torch.isfinite(z).all() and torch.isfinite(aux)),
        "test_metrics_present": test_metrics_present,
        "task_evaluate_test": False,
        "active_node_count": int(active.sum().item()),
        "input_self_loops_removed": int(info["input_self_loops_removed"]),
        "parameter_count_model": int(row["metadata"]["model_parameters"]),
        "parameter_count_classifier": int(row["metadata"]["classifier_parameters"]),
        "static_routing_verified": True,
        "max_static_strength_node_difference": max_static_strength_node_difference,
        "top2_verified": True,
        "operator_rows": op_rows,
        "trajectory_rows": trajectory_rows,
        "novelty_rows": novelty_rows,
        "profile_rows": profile_rows,
        "similarity_rows": similarity_rows,
        "context_rows": context_rows,
        "routing_rows": routing_rows,
        "strength_rows": strength_rows,
        "legacy_modules_frozen": all(
            not p.requires_grad
            for p in (
                *model.evidence_scalar_projector.parameters(),
                *model.evidence_norm.parameters(),
                *model.expert_query_proj.parameters(),
                *model.expert_key_proj.parameters(),
            )
        )
        and not model.expert_key_embedding.requires_grad
        and not model.node_residual_raw.requires_grad
        and not model.compatibility_residual_raw.requires_grad,
        "strength_head_active": any(p.requires_grad for p in model.strength_head.parameters()),
        "context_mix_no_weight_decay": "context_mix_raw" in model.no_weight_decay_parameter_names,
    }
    if not result["finite"] or not result["legacy_modules_frozen"]:
        raise AssertionError(f"V3A selected checkpoint audit failed: {dataset}/{seed}/{variant}")
    del model, x, edge, checkpoint
    if "cuda" in device_name:
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def audit_c0_compatible_checkpoint(
    row: dict[str, Any],
    *,
    root: Path = ROOT,
    output_root: Path = OUTPUT_ROOT,
    device_name: str = "cuda:0",
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Compare a C0 checkpoint's shared state through V2.2 and V3A forwards."""
    from src.models.mvcge_mag_v22 import Model as V22Model
    from src.models.mvcge_mag_v3a import Model as V3AModel

    if row["variant"] != "C0_raw":
        raise ValueError("C0 compatibility audit only accepts C0_raw")
    run_dir, checkpoint_path = _checkpoint_paths(row, root=root, output_root=output_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    cfg.model.variant = "C0_raw"
    data_info = checkpoint["data_info"]
    v22_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    v22_cfg.model.name = "mvcge_mag_v22"
    v22_cfg.model.variant = "R0_modality_static"
    old_template = V22Model(v22_cfg, data_info)
    v22_state_keys = old_template.state_dict()
    state = checkpoint["model_state"]
    missing = sorted(set(v22_state_keys) - set(state))
    shape_mismatches = sorted(
        key for key, value in v22_state_keys.items()
        if key in state and tuple(value.shape) != tuple(state[key].shape)
    )
    if missing or shape_mismatches:
        raise AssertionError(
            f"C0 checkpoint is not V2.2-compatible: missing={missing}, shapes={shape_mismatches}"
        )
    common_state = {key: state[key] for key in v22_state_keys}
    # Compare on CPU to avoid sub-ULP CUDA scatter/reduction differences being
    # mistaken for a C0 implementation mismatch.
    audit_device = "cpu"
    v22 = V22Model(v22_cfg, data_info)
    v22.load_state_dict(common_state, strict=True)
    v3 = V3AModel(cfg, data_info)
    v3.load_state_dict(common_state, strict=False)
    v22, v3 = v22.to(audit_device).eval(), v3.to(audit_device).eval()
    if cached_data is None:
        cached_data = load_features_and_edges(row["dataset"], int(row["seed"]))
    x_cpu, edge_cpu = cached_data
    x, edge = x_cpu.to(audit_device), edge_cpu.to(audit_device)
    old_z, _, _, old_aux, _ = v22(x, edge)
    new_z, _, _, new_aux, _ = v3(x, edge)
    max_diff = float((old_z - new_z).abs().max().item())
    aux_diff = float((old_aux - new_aux).abs().item())
    exact = torch.equal(old_z, new_z) and torch.equal(old_aux, new_aux)
    del old_template, v22, v3, checkpoint
    if not exact:
        raise AssertionError(
            f"C0 checkpoint differs from compatible V2.2 forward: z={max_diff}, aux={aux_diff}"
        )
    return {
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "v22_compatible": True,
        "v22_state_missing_keys": len(missing),
        "v22_state_shape_mismatches": len(shape_mismatches),
        "audit_device": audit_device,
        "state_output_exact": exact,
        "z_max_abs_diff": max_diff,
        "aux_abs_diff": aux_diff,
    }


def _stable_positive(item: dict[str, Any]) -> bool:
    return (
        item["overall_delta_accuracy_pp"] > 0
        and item["overall_delta_macro_f1_pp"] > 0
        and item["positive_accuracy_pairs"] >= 6
        and sum(item[f"{dataset}_delta_accuracy_pp_mean"] > 0 for dataset in DATASETS) >= 2
    )


def _approximately_equal(item: dict[str, Any]) -> bool:
    return abs(item["overall_delta_accuracy_pp"]) <= 0.15 and abs(item["overall_delta_macro_f1_pp"]) <= 0.50


def campaign_summaries(rows: list[dict[str, Any]]):
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)
    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = grouped[(dataset, variant)]
            for key in ("val_acc", "val_macro_f1"):
                values = [float(row["metrics"][key]) for row in group]
                mean = statistics.fmean(values)
                std = statistics.pstdev(values) if len(values) > 1 else 0.0
                if key == "val_acc":
                    acc_mean, acc_std = mean, std
                else:
                    f1_mean, f1_std = mean, std
            summary.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "num_runs": len(group),
                    "val_acc_mean": acc_mean,
                    "val_acc_std": acc_std,
                    "val_macro_f1_mean": f1_mean,
                    "val_macro_f1_std": f1_std,
                    "mean_best_epoch": statistics.fmean(float(row["metadata"]["best_epoch"]) for row in group),
                    "model_trainable_params": group[0]["metadata"]["model_parameters"],
                    "classifier_params": group[0]["metadata"]["classifier_parameters"],
                    "total_trainable_params": group[0]["metadata"]["model_parameters"] + group[0]["metadata"]["classifier_parameters"],
                }
            )
    return summary


def paired_comparisons(rows: list[dict[str, Any]]):
    keyed = {(row["dataset"], int(row["seed"]), row["variant"]): row for row in rows}
    output = []
    for left, right in PAIRS:
        acc, f1 = [], []
        record: dict[str, Any] = {"comparison": f"{left} - {right}", "n_paired_runs": 0}
        for dataset in DATASETS:
            da, df = [], []
            for seed in SEEDS:
                a = keyed[(dataset, seed, left)]["metrics"]
                b = keyed[(dataset, seed, right)]["metrics"]
                da.append((float(a["val_acc"]) - float(b["val_acc"])) * 100.0)
                df.append((float(a["val_macro_f1"]) - float(b["val_macro_f1"])) * 100.0)
            acc.extend(da)
            f1.extend(df)
            record[f"{dataset}_delta_accuracy_pp_mean"] = statistics.fmean(da)
            record[f"{dataset}_delta_macro_f1_pp_mean"] = statistics.fmean(df)
            record[f"{dataset}_positive_accuracy_seeds"] = sum(value > 0 for value in da)
            record[f"{dataset}_positive_macro_f1_seeds"] = sum(value > 0 for value in df)
        record.update(
            {
                "overall_delta_accuracy_pp": statistics.fmean(acc),
                "overall_delta_macro_f1_pp": statistics.fmean(f1),
                "positive_accuracy_pairs": sum(value > 0 for value in acc),
                "positive_macro_f1_pairs": sum(value > 0 for value in f1),
                "n_paired_runs": len(acc),
            }
        )
        output.append(record)
    return output


@torch.no_grad()
def historical_c0_audit(
    new_row: dict[str, Any],
    old_row: dict[str, Any],
    *,
    root: Path,
    output_root: Path,
    device: str,
    cached_data: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    """Compare historical V2.2 R0 metrics and verify V3A C0 state/output compatibility."""
    from src.models.mvcge_mag_v22 import Model as V22Model
    from src.models.mvcge_mag_v3a import Model as V3AModel

    new_dir, new_path = _checkpoint_paths(new_row, root=root, output_root=output_root)
    old_dir, old_path = _checkpoint_paths(old_row, root=root, output_root=root / "outputs" / "mvcge_mag_v22_structure_grounded_router")
    old_checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    new_checkpoint = torch.load(new_path, map_location="cpu", weights_only=False)
    old_cfg = OmegaConf.load(old_dir / "hydra" / ".hydra" / "config.yaml")
    old_cfg.model.name = "mvcge_mag_v22"
    old_cfg.model.variant = "R0_modality_static"
    new_cfg = OmegaConf.load(new_dir / "hydra" / ".hydra" / "config.yaml")
    new_cfg.model.variant = "C0_raw"
    old_state = old_checkpoint["model_state"]
    new_state = new_checkpoint["model_state"]
    data_info = new_checkpoint["data_info"]
    old_template = V22Model(old_cfg, data_info)
    v22_keys = old_template.state_dict()
    missing = sorted(set(v22_keys) - set(new_state))
    shape_mismatches = sorted(key for key in v22_keys if key in new_state and tuple(v22_keys[key].shape) != tuple(new_state[key].shape))
    if missing or shape_mismatches:
        raise AssertionError(f"V3A C0 checkpoint is not V2.2-compatible: missing={missing}, shapes={shape_mismatches}")

    x_cpu, edge_cpu = cached_data
    # CPU execution makes exact state/output compatibility independent of
    # GPU scatter accumulation order across otherwise equivalent forwards.
    audit_device = "cpu"
    x, edge = x_cpu.to(audit_device), edge_cpu.to(audit_device)
    comparisons = {}
    for label, state in (("historical_r0", old_state), ("new_c0", new_state)):
        v22 = V22Model(old_cfg, data_info)
        common_state = {key: state[key] for key in v22.state_dict()}
        v22.load_state_dict(common_state, strict=True)
        v3cfg = OmegaConf.create(OmegaConf.to_container(new_cfg, resolve=True))
        v3cfg.model.variant = "C0_raw"
        v3 = V3AModel(v3cfg, data_info)
        v3.load_state_dict(common_state, strict=False)
        v22, v3 = v22.to(audit_device).eval(), v3.to(audit_device).eval()
        z_old, _, _, aux_old, _ = v22(x, edge)
        z_new, _, _, aux_new, _ = v3(x, edge)
        comparisons[label] = {
            "z_max_abs_diff": float((z_old - z_new).abs().max().item()),
            "aux_abs_diff": float((aux_old - aux_new).abs().item()),
            "exact": bool(torch.equal(z_old, z_new) and torch.equal(aux_old, aux_new)),
            "allclose_1e7": bool(
                torch.allclose(z_old, z_new, atol=1.0e-7, rtol=0.0)
                and torch.allclose(aux_old, aux_new, atol=1.0e-7, rtol=0.0)
            ),
        }
        del v22, v3
    return {
        "dataset": new_row["dataset"],
        "seed": int(new_row["seed"]),
        "old_R0_val_acc": float(old_row["metrics"]["val_acc"]),
        "old_R0_val_macro_f1": float(old_row["metrics"]["val_macro_f1"]),
        "new_C0_val_acc": float(new_row["metrics"]["val_acc"]),
        "new_C0_val_macro_f1": float(new_row["metrics"]["val_macro_f1"]),
        "C0_minus_old_R0_acc_pp": (float(new_row["metrics"]["val_acc"]) - float(old_row["metrics"]["val_acc"])) * 100.0,
        "C0_minus_old_R0_f1_pp": (float(new_row["metrics"]["val_macro_f1"]) - float(old_row["metrics"]["val_macro_f1"])) * 100.0,
        "old_R0_best_epoch": int(old_row["metadata"]["best_epoch"]),
        "new_C0_best_epoch": int(new_row["metadata"]["best_epoch"]),
        "best_epoch_difference": int(new_row["metadata"]["best_epoch"]) - int(old_row["metadata"]["best_epoch"]),
        "v22_missing_state_keys": len(missing),
        "v22_state_shape_mismatches": len(shape_mismatches),
        "historical_r0_state_output_max_abs_diff": comparisons["historical_r0"]["z_max_abs_diff"],
        "historical_r0_state_aux_abs_diff": comparisons["historical_r0"]["aux_abs_diff"],
        "historical_r0_state_output_exact": comparisons["historical_r0"]["exact"],
        "historical_r0_state_output_allclose_1e7": comparisons["historical_r0"]["allclose_1e7"],
        "new_c0_state_output_max_abs_diff": comparisons["new_c0"]["z_max_abs_diff"],
        "new_c0_state_aux_abs_diff": comparisons["new_c0"]["aux_abs_diff"],
        "new_c0_state_output_exact": comparisons["new_c0"]["exact"],
        "new_c0_state_output_allclose_1e7": comparisons["new_c0"]["allclose_1e7"],
    }


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _report_table(summary, paired, operator_rows, trajectory_rows, novelty_rows, context_rows, profiles, similarities, routing, strengths, historical):
    pair_map = {row["comparison"]: row for row in paired}
    report = ["# MvCGE-MAG V3A: Effective Structural Action-Space Screen", "", "## Protocol and provenance", ""]
    report.extend(
        [
            "- Branch `exp/mvcge_mag_v3a_effective_context_screen`; parent `bb52895da5863c737fc47c1f88637be12a069445`; validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; C0–C3; 36 runs.",
            "- C0 is the V2.2 R0 modality-static selection and strength controller. The raw and effective contexts use modality-static Top-2 selection and the same static strength; no node-conditioned router is active.",
            "- Semantic edge weights use only detached pretrained raw modality features on existing physical edges. No labels were used in operator or selected-checkpoint audits.",
            "- `task.evaluate_test=false`; checkpoints are selected by Validation Accuracy. No Test metrics, HPO, LP, significance testing, or V3B implementation.",
            "- Model/config were frozen before the formal campaign. Formal campaign provenance and selected-checkpoint audits are recorded in `data/campaign_manifest.json`.",
            "",
            "## Validation results",
            "",
            "Run-level means and population standard deviations; paired deltas are descriptive percentage points across matched dataset-seed runs.",
            "",
            "| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Trainable parameters (model + head) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        report.append(
            f"| {row['dataset']} | {row['variant']} | {100*row['val_acc_mean']:.2f}% ± {100*row['val_acc_std']:.2f}% | {100*row['val_macro_f1_mean']:.2f}% ± {100*row['val_macro_f1_std']:.2f}% | {row['mean_best_epoch']:.1f} | {row['model_trainable_params']:,} + {row['classifier_params']:,} |"
        )
    report.extend(["", "## Primary paired comparisons", "", "| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Accuracy / F1) |", "|---|---:|---:|---:|"])
    pretty = (
        ("C1_effective - C0_raw", "C1 − C0: effective-only vs raw-only"),
        ("C2_dual_shared_profile - C0_raw", "C2 − C0: dual shared-profile vs raw-only"),
        ("C2_dual_shared_profile - C1_effective", "C2 − C1: raw context added to effective-only"),
        ("C3_dual_context_profile - C2_dual_shared_profile", "C3 − C2: context-specific effective hop profile"),
        ("C3_dual_context_profile - C0_raw", "C3 − C0: full action-space expansion vs raw-only"),
    )
    for key, label in pretty:
        row = pair_map[key]
        report.append(f"| {label} | {row['overall_delta_accuracy_pp']:+.3f} pp | {row['overall_delta_macro_f1_pp']:+.3f} pp | {row['positive_accuracy_pairs']}/9 / {row['positive_macro_f1_pairs']}/9 |")
    report.extend(["", "Per-dataset deltas and positive-seed counts are in `data/paired_comparisons.csv`. No significance testing was performed.", "", "## Semantic edge and operator diagnostics", ""])
    if operator_rows:
        report.extend(["| Dataset | Modality | Mean semantic cosine | p10–p90 | Mean relative raw/effective operator ΔL2 | Text/Visual effective ΔL2 |", "|---|---|---:|---:|---:|---:|"])
        for dataset in DATASETS:
            for modality in ("text", "visual"):
                rows = [r for r in operator_rows if r["dataset"] == dataset and r["modality"] == modality]
                report.append(
                    f"| {dataset} | {modality} | {_mean(r['semantic_similarity_mean'] for r in rows):.4f} | {_mean(r['semantic_similarity_p10'] for r in rows):.4f}–{_mean(r['semantic_similarity_p90'] for r in rows):.4f} | {_mean(r['operator_weight_delta_l2'] for r in rows):.4f} | {_mean(r['text_visual_eff_operator_delta_l2'] for r in rows):.4f} |"
                )
    report.extend(["", "The edge-score cosine distribution, threshold fractions, normalized raw/effective sparse edge-vector norms/cosines, effective-degree quantiles, and text/visual operator discrepancies are recorded in `data/operator_diagnostics.csv`. No dense adjacency was constructed.", "", "## Trajectory and action-space novelty", ""])
    report.extend(["| Variant | Modality | Hop | Raw/effective cosine | Relative RMS delta | Raw RMS | Effective RMS |", "|---|---|---:|---:|---:|---:|---:|"])
    for variant in VARIANTS:
        for modality in ("text", "visual"):
            selected = [r for r in trajectory_rows if r["variant"] == variant and r["modality"] == modality]
            for hop in range(1, 5):
                report.append(
                    f"| {variant} | {modality} | {hop} | {_mean(r[f'raw_eff_flattened_cosine_{hop}'] for r in selected):.4f} | {_mean(r[f'raw_eff_relative_rms_delta_{hop}'] for r in selected):.4f} | {_mean(r[f'raw_rms_{hop}'] for r in selected):.4f} | {_mean(r[f'effective_rms_{hop}'] for r in selected):.4f} |"
                )
    report.extend(["", "`action_space_novelty.csv` uses the C0-selected projector state to isolate operator-context span novelty. It reports effective-outside-raw and raw-outside-effective ratios plus Gram condition diagnostics, using 4×4 Gram identities rather than explicit projection residuals.", ""])
    for modality in ("text", "visual"):
        selected = [r for r in novelty_rows if r["modality"] == modality]
        report.append(f"- {modality}: mean effective-outside-raw-span ratio {_mean(r['effective_outside_raw_span_ratio'] for r in selected):.4f}; mean raw-outside-effective-span ratio {_mean(r['raw_outside_effective_span_ratio'] for r in selected):.4f}.")
    report.extend(["", "## Context mixing, alpha profiles, and functional experts", ""])
    for variant in ("C2_dual_shared_profile", "C3_dual_context_profile"):
        selected = [r for r in context_rows if r["variant"] == variant]
        report.append(f"- {variant} learned λ mean/range by expert: " + "; ".join(
            f"E{expert}: {_mean(r['lambda'] for r in selected if r['expert_id']==expert):.4f} [{min(r['lambda'] for r in selected if r['expert_id']==expert):.4f}, {max(r['lambda'] for r in selected if r['expert_id']==expert):.4f}]"
            for expert in range(4)
        ) + ".")
    c3_profiles = [r for r in profiles if r["variant"] == "C3_dual_context_profile"]
    report.append(f"- C3 raw/effective alpha cosine mean: {_mean(r['raw_eff_alpha_cosine'] for r in c3_profiles):.4f}; raw-alpha drift from initialization: {_mean(r['raw_alpha_drift_from_init'] for r in c3_profiles):.4f}; effective-alpha drift: {_mean(r['eff_alpha_drift_from_init'] for r in c3_profiles):.4f}.")
    output_cos = [r for r in similarities if r["kind"] == "expert_output"]
    raw_eff_cos = [r for r in similarities if r["kind"] == "raw_effective_profile"]
    report.append(f"- Expert output cosine: flattened {_mean(r['flattened_cosine'] for r in output_cos):.4f}; mean-node {_mean(r['mean_node_cosine'] for r in output_cos):.4f}. C2/C3 raw/effective pre-transform profile cosine mean: {_mean(r['flattened_cosine'] for r in raw_eff_cos):.4f}.")
    report.append("- Full per-expert λ, RMS contributions, alpha values/drifts, within-bank alpha cosine, and output/profile pair similarities are in `data/context_mix_diagnostics.csv`, `data/expert_profiles.csv`, and `data/expert_similarity.csv`.")
    report.extend(["", "## Static routing and correction scale", "", "| Variant | Most common Text pair | Most common Visual pair | Text/Visual same-pair rate | Mean expert union | Runs with dead expert slot |", "|---|---|---|---:|---:|---:|"])
    for variant in VARIANTS:
        group = [r for r in routing if r["variant"] == variant]
        pair_counts = {m: Counter(r["selected_expert_pair"] for r in group if r["modality"] == m) for m in ("text", "visual")}
        top_text = pair_counts["text"].most_common(1)[0][0]
        top_visual = pair_counts["visual"].most_common(1)[0][0]
        checkpoints = {(r["dataset"], r["seed"]): r for r in group if r["modality"] == "text"}
        same_rate = sum(bool(r["text_visual_selected_pair_same"]) for r in checkpoints.values()) / max(len(checkpoints), 1)
        report.append(f"| {variant} | {top_text} ({pair_counts['text'][top_text]}/9) | {top_visual} ({pair_counts['visual'][top_visual]}/9) | {same_rate:.3f} | {_mean(r['union_experts_used'] for r in group):.3f}/4 | {sum(r['dead_expert_slots_across_modalities'] > 0 for r in checkpoints.values())}/9 |")
    for variant in VARIANTS:
        group = [r for r in strengths if r["variant"] == variant]
        report.append(f"- {variant} correction RMS/prior RMS ratio mean: {_mean(r['scaled_correction_to_prior_rms_ratio'] for r in group):.4f}; scaled correction RMS {_mean(r['scaled_correction_rms'] for r in group):.4f}; prior RMS {_mean(r['prior_rms'] for r in group):.4f}.")
    report.extend(["", "Static unused experts are sparse global compositions, not node-router collapse. Per-checkpoint routes, dense probabilities, Top-2 weights, strength and correction RMS are in `data/routing_diagnostics.csv` and `data/strength_diagnostics.csv`.", "", "## Historical C0 regression against V2.2 R0", ""])
    old_acc_deltas = [float(r["C0_minus_old_R0_acc_pp"]) for r in historical]
    old_f1_deltas = [float(r["C0_minus_old_R0_f1_pp"]) for r in historical]

    def direction(values):
        if all(value > 0 for value in values):
            return "all positive"
        if all(value < 0 for value in values):
            return "all negative"
        return "mixed or zero"

    report.append(
        f"Across nine matched dataset-seed runs, C0−old R0 mean deltas were "
        f"Accuracy {_mean(old_acc_deltas):+.3f} pp ({sum(v > 0 for v in old_acc_deltas)}/9 positive; "
        f"range {min(old_acc_deltas):+.3f} to {max(old_acc_deltas):+.3f}; direction {direction(old_acc_deltas)}) "
        f"and Macro-F1 {_mean(old_f1_deltas):+.3f} pp ({sum(v > 0 for v in old_f1_deltas)}/9 positive; "
        f"range {min(old_f1_deltas):+.3f} to {max(old_f1_deltas):+.3f}; direction {direction(old_f1_deltas)}); "
        f"best-epoch difference mean {_mean(r['best_epoch_difference'] for r in historical):+.2f}. "
        f"The mixed-or-zero directions show no uniform signed shift; this is descriptive and not a significance claim. "
        f"Compatible historical/new checkpoint state-to-output audits exact: "
        f"{sum(bool(r['historical_r0_state_output_exact']) for r in historical)}/9 historical states and "
        f"{sum(bool(r['new_c0_state_output_exact']) for r in historical)}/9 new C0 states. "
        f"At atol=1e-7, allclose counts were "
        f"{sum(bool(r['historical_r0_state_output_allclose_1e7']) for r in historical)}/9 historical and "
        f"{sum(bool(r['new_c0_state_output_allclose_1e7']) for r in historical)}/9 new C0; "
        f"maximum z differences were "
        f"{max(float(r['historical_r0_state_output_max_abs_diff']) for r in historical):.3g} historical and "
        f"{max(float(r['new_c0_state_output_max_abs_diff']) for r in historical):.3g} new C0. "
        f"Any failed exact/allclose pair is reported in the CSV rather than suppressing the analysis. "
        f"See `data/historical_c0_regression.csv` for each matched pair and maximum output difference."
    )
    report.extend(["", "## Frozen interpretation rules and decision map", "", "Stable-positive means positive overall Accuracy and Macro-F1 deltas, at least 6/9 positive Accuracy pairs, and positive Accuracy mean on at least 2/3 datasets. Approximately equal means |Δ Accuracy| ≤ 0.15 pp and |Δ Macro-F1| ≤ 0.50 pp. These are descriptive labels, not significance or equivalence tests.", ""])
    c1 = pair_map["C1_effective - C0_raw"]
    c2 = pair_map["C2_dual_shared_profile - C0_raw"]
    c2_c1 = pair_map["C2_dual_shared_profile - C1_effective"]
    c3_c2 = pair_map["C3_dual_context_profile - C2_dual_shared_profile"]
    c3_c0 = pair_map["C3_dual_context_profile - C0_raw"]
    s1, s2, s3, s5 = map(_stable_positive, (c1, c2, c3_c2, c3_c0))
    eq32 = _approximately_equal(c3_c2)
    eq20 = _approximately_equal(c2)
    eq30 = _approximately_equal(c3_c0)
    report.append(f"- **A. {'Observed' if s1 else 'Not observed'}.** C1−C0 is {c1['overall_delta_accuracy_pp']:+.3f}/{c1['overall_delta_macro_f1_pp']:+.3f} pp Accuracy/Macro-F1 with {c1['positive_accuracy_pairs']}/9 positive Accuracy pairs. {'Effective-only is stable-positive against raw-only.' if s1 else 'Effective-only does not meet the stable-positive gate as a standalone replacement.'}")
    b_obs = (not s1) and _stable_positive(c2)
    report.append(f"- **B. {'Observed' if b_obs else 'Not observed'}.** C1−C0 {'does not pass' if not s1 else 'passes'} stable-positive; C2−C0 is {c2['overall_delta_accuracy_pp']:+.3f}/{c2['overall_delta_macro_f1_pp']:+.3f} pp. {'The dual context is stable-positive while effective-only is not; interpret it as complementary raw/effective utility, not replacement.' if b_obs else 'The rule requiring C1 not to pass and C2 to pass is not met.'}")
    c_obs = _stable_positive(c2) and eq32
    report.append(f"- **C. {'Observed' if c_obs else 'Not observed'}.** C2−C0 stable-positive={_stable_positive(c2)}; C3−C2 approximate-equal={eq32}. {'The dual context helps and shared hop profiles suffice; prefer the simpler C2.' if c_obs else 'Both parts of the C2-positive/C3-near-equal pattern are not satisfied together.'}")
    d_obs = s3
    report.append(f"- **D. {'Observed' if d_obs else 'Not observed'}.** C3−C2 stable-positive={d_obs}; {'different structural contexts support context-specific multi-hop profiles.' if d_obs else 'context-specific multi-hop profiles are not supported by a stable-positive C3−C2 result.'}")
    e_obs = eq20 and eq30 and max(float(r['operator_weight_delta_l2']) for r in operator_rows) <= 1.0e-6 and max(float(r['effective_outside_raw_span_ratio']) for r in novelty_rows) <= 1.0e-3
    report.append(f"- **E. {'Observed' if e_obs else 'Not observed'}.** C2/C3 near-equality to C0={eq20 and eq30}; maximum operator relative ΔL2={max(float(r['operator_weight_delta_l2']) for r in operator_rows):.4g}; maximum C0 effective-outside-raw span ratio={max(float(r['effective_outside_raw_span_ratio']) for r in novelty_rows):.4g}. {'This points to insufficient new structural action from this CAMPA-style weighting without rejecting the broader effective-context principle.' if e_obs else 'The full near-equal/near-zero operator-and-span pattern is not present.'}")
    f_obs = (not _stable_positive(c2)) and (not _stable_positive(c3_c0)) and max(float(r['operator_weight_delta_l2']) for r in operator_rows) > 1.0e-3 and max(float(r['effective_outside_raw_span_ratio']) for r in novelty_rows) > 1.0e-2
    report.append(f"- **F. {'Observed' if f_obs else 'Not observed'}.** C2/C3 stable-positive against C0={_stable_positive(c2)}/{_stable_positive(c3_c0)}; maximum operator relative ΔL2={max(float(r['operator_weight_delta_l2']) for r in operator_rows):.4f}; maximum span novelty={max(float(r['effective_outside_raw_span_ratio']) for r in novelty_rows):.4f}. {'The contexts differ, but this NC screen does not show task need; do not respond by sharpening similarity.' if f_obs else 'This combined different-context/no-stable-gain pattern is not established under the descriptive screen.'}")
    lambdas = [float(r["lambda"]) for r in context_rows]
    g_obs = bool(lambdas) and all(value <= 0.05 for value in lambdas)
    report.append(f"- **G. {'Observed' if g_obs else 'Not observed'}.** C2/C3 learned λ range is {min(lambdas):.4f}–{max(lambdas):.4f} (initial 0.10); {'all values are ≤0.05, indicating preference for Raw context.' if g_obs else 'not all learned values are near zero under the frozen ≤0.05 descriptive flag.'}")
    lambda_checkpoint_spreads = []
    by_lambda_key: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for r in context_rows:
        by_lambda_key[(r["dataset"], int(r["seed"]), r["variant"])].append(float(r["lambda"]))
    lambda_checkpoint_spreads = [max(v) - min(v) for v in by_lambda_key.values() if v]
    h_obs = bool(lambda_checkpoint_spreads) and max(lambda_checkpoint_spreads) > 0.05 and (_stable_positive(c2) or _stable_positive(c3_c0))
    report.append(f"- **H. {'Observed' if h_obs else 'Not observed'}.** Maximum within-checkpoint expert λ range={max(lambda_checkpoint_spreads) if lambda_checkpoint_spreads else 0.0:.4f}; positive C2/C3-vs-C0 gate={_stable_positive(c2) or _stable_positive(c3_c0)}. {'This supports shared-dictionary structural-context specialists.' if h_obs else 'The rule of differentiated expert mixing plus positive performance is not established.'}")
    c3_alpha_cos = [float(r["raw_eff_alpha_cosine"]) for r in c3_profiles if r["raw_eff_alpha_cosine"] is not None]
    c3_alpha_drift = [float(r["eff_alpha_drift_from_init"]) for r in c3_profiles if r["eff_alpha_drift_from_init"] is not None]
    i_obs = s3 and bool(c3_alpha_cos) and _mean(c3_alpha_cos) < 0.99 and _mean(c3_alpha_drift) > 1.0e-3
    report.append(f"- **I. {'Observed' if i_obs else 'Not observed'}.** C3−C2 stable-positive={s3}; mean raw/effective alpha cosine={_mean(c3_alpha_cos):.4f}; mean effective-alpha drift={_mean(c3_alpha_drift):.4f}. {'The evidence supports context-specific multi-hop response.' if i_obs else 'Performance and alpha separation do not jointly establish context-specific multi-hop response.'}")
    report.extend(["", "## V3B gate and boundary", "", f"C2 stable-positive vs C0: {_stable_positive(c2)}; C3 stable-positive vs C0: {_stable_positive(c3_c0)}. {'The V3B gate is met for a later separately authorized study. This campaign still stops here.' if (_stable_positive(c2) or _stable_positive(c3_c0)) else 'The V3B gate is not met; stop effective-context screen work here.'}", "", "No V3B or cross-modal operator was implemented. The observed comparisons do not establish statistical significance or causal mechanism.", "", "## Reproducibility artifacts", "", "- `data/environment.json`, `data/preflight_operator_diagnostics.csv`, `data/preflight_summary.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`", "- `data/summary.csv`, `data/paired_comparisons.csv`, `data/operator_diagnostics.csv`, `data/trajectory_diagnostics.csv`, `data/action_space_novelty.csv`", "- `data/context_mix_diagnostics.csv`, `data/routing_diagnostics.csv`, `data/strength_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/historical_c0_regression.csv`", "- Checkpoints, logs, raw features, raw edge tensors and caches remain under ignored server-local `outputs/`."])
    return "\n".join(report) + "\n"


def analyze(device: str = "cuda:0") -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[MvCGE-MAG V3A analysis] {device} unavailable; falling back to CPU for read-only audits",
            flush=True,
        )
        device = "cpu"
    # Limit CPU thread fan-out for the sparse full-graph audits; this keeps
    # repeated index_add propagation predictable on large feature tables.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rows = json.loads((DATA_ROOT / "run_rows.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA_ROOT / "campaign_manifest.json").read_text(encoding="utf-8"))
    keys = {(r["dataset"], int(r["seed"]), r["variant"]) for r in rows if r.get("status") in {"completed", "reused"}}
    if len(rows) != 36 or len(keys) != 36 or manifest.get("completed_runs") != 36:
        raise RuntimeError("Analysis requires 36 unique completed runs")
    if manifest.get("task_evaluate_test") is not False:
        raise RuntimeError("Analysis requires task.evaluate_test=false")
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    audit_results = []
    for idx, row in enumerate(rows, start=1):
        if row["dataset"] not in cache:
            cache[row["dataset"]] = load_features_and_edges(row["dataset"], int(row["seed"]))
        audit = audit_checkpoint(row, root=ROOT, output_root=OUTPUT_ROOT, device_name=device, cached_data=cache[row["dataset"]])
        if audit["test_metrics_present"] or audit["task_evaluate_test"] is not False:
            raise AssertionError("Test isolation audit failed")
        audit_results.append(audit)
        print(f"[MvCGE-MAG V3A audit] {idx}/36 {row['dataset']} seed={row['seed']} {row['variant']}", flush=True)

    summary = campaign_summaries(rows)
    paired = paired_comparisons(rows)
    operator_rows = [r for a in audit_results if a["variant"] == "C0_raw" for r in a["operator_rows"]]
    trajectory_rows = [r for a in audit_results for r in a["trajectory_rows"]]
    novelty_rows = [r for a in audit_results if a["variant"] == "C0_raw" for r in a["novelty_rows"]]
    context_rows = [r for a in audit_results for r in a["context_rows"]]
    profile_rows = [r for a in audit_results for r in a["profile_rows"]]
    similarity_rows = [r for a in audit_results for r in a["similarity_rows"]]
    routing_rows = [r for a in audit_results for r in a["routing_rows"]]
    strength_rows = [r for a in audit_results for r in a["strength_rows"]]

    old_rows = json.loads((V22_ROOT / "data" / "run_rows.json").read_text(encoding="utf-8"))
    old_map = {(r["dataset"], int(r["seed"])): r for r in old_rows if r["variant"] == "R0_modality_static"}
    new_map = {(r["dataset"], int(r["seed"])): r for r in rows if r["variant"] == "C0_raw"}
    historical = []
    for dataset in DATASETS:
        for seed in SEEDS:
            if dataset not in cache:
                cache[dataset] = load_features_and_edges(dataset, seed)
            historical.append(
                historical_c0_audit(
                    new_map[(dataset, seed)], old_map[(dataset, seed)], root=ROOT,
                    output_root=OUTPUT_ROOT, device=device, cached_data=cache[dataset]
                )
            )
    if len(historical) != 9 or any(
        r["v22_missing_state_keys"] or r["v22_state_shape_mismatches"] for r in historical
    ):
        raise AssertionError("C0 historical checkpoint state compatibility failed")

    # All output tables are finite-or-blank before they are written.
    tables = {
        "summary.csv": summary,
        "paired_comparisons.csv": paired,
        "operator_diagnostics.csv": operator_rows,
        "trajectory_diagnostics.csv": trajectory_rows,
        "action_space_novelty.csv": novelty_rows,
        "context_mix_diagnostics.csv": context_rows,
        "routing_diagnostics.csv": routing_rows,
        "strength_diagnostics.csv": strength_rows,
        "expert_profiles.csv": profile_rows,
        "expert_similarity.csv": similarity_rows,
        "historical_c0_regression.csv": historical,
    }
    for filename, table in tables.items():
        for row in table:
            for value in row.values():
                if not _finite_float(value):
                    raise FloatingPointError(f"Non-finite value in {filename}: {row}")
        write_csv(DATA_ROOT / filename, table)

    manifest.update(
        {
            "task_evaluate_test": False,
            "test_evaluation_verified_false": True,
            "test_metrics_absent": all(not a["test_metrics_present"] for a in audit_results),
            "label_free_checkpoint_audit": True,
            "selected_checkpoint_audits": len(audit_results),
            "all_selected_checkpoints_finite": all(a["finite"] for a in audit_results),
            "all_static_routing_verified": all(a["static_routing_verified"] for a in audit_results),
            "max_static_strength_node_difference": max(
                a["max_static_strength_node_difference"] for a in audit_results
            ),
            "all_top2_verified": all(a["top2_verified"] for a in audit_results),
            "legacy_router_evidence_compatibility_modules_frozen": all(a["legacy_modules_frozen"] for a in audit_results),
            "c0_historical_state_output_audits_exact": all(
                r["historical_r0_state_output_exact"] and r["new_c0_state_output_exact"]
                for r in historical
            ),
            "c0_historical_exact_pairs": sum(
                bool(r["historical_r0_state_output_exact"] and r["new_c0_state_output_exact"])
                for r in historical
            ),
            "c0_historical_allclose_1e7_pairs": sum(
                bool(r["historical_r0_state_output_allclose_1e7"] and r["new_c0_state_output_allclose_1e7"])
                for r in historical
            ),
            "analysis_device": device,
        }
    )
    write_json(DATA_ROOT / "campaign_manifest.json", manifest)
    (RESEARCH_ROOT / "REPORT.md").write_text(
        _report_table(summary, paired, operator_rows, trajectory_rows, novelty_rows, context_rows, profile_rows, similarity_rows, routing_rows, strength_rows, historical),
        encoding="utf-8",
    )
    print("[MvCGE-MAG V3A analysis] wrote selected-checkpoint diagnostics and REPORT.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight = run_label_free_preflight(args.device)
        write_csv(DATA_ROOT / "preflight_operator_diagnostics.csv", preflight)
        degenerate = preflight_is_degenerate(preflight)
        print(
            f"[MvCGE-MAG V3A preflight] {len(preflight)} modality records; "
            f"degenerate={degenerate}",
            flush=True,
        )
        if degenerate:
            raise SystemExit(2)
    else:
        analyze(args.device)
