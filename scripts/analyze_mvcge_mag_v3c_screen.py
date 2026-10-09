#!/usr/bin/env python3
"""Label-free role preflight and validation-selected V3C checkpoint analysis."""

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
VARIANTS = ("F0_raw", "F1_support", "F2_role_dual_smooth", "F3_role_functional")
PAIRS = (
    ("F1_support", "F0_raw"),
    ("F2_role_dual_smooth", "F1_support"),
    ("F2_role_dual_smooth", "F0_raw"),
    ("F3_role_functional", "F2_role_dual_smooth"),
    ("F3_role_functional", "F1_support"),
    ("F3_role_functional", "F0_raw"),
)
EXPERT_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v3c_functional_role_screen"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v3c_functional_role_screen"
DATA_ROOT = RESEARCH_ROOT / "data"
V3A_ROOT = ROOT / "research" / "mvcge_mag_v3a_effective_context_screen"
V3A_OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v3a_effective_context_screen"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _finite_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, numbers.Real):
        return math.isfinite(float(value))
    return isinstance(value, str)


def _stats(value: torch.Tensor) -> dict[str, float | None]:
    value = value.detach().reshape(-1).float()
    if not value.numel():
        return {name: None for name in ("mean", "std", "p10", "p50", "p90")}
    q = torch.quantile(value, torch.tensor([0.1, 0.5, 0.9], device=value.device))
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "p10": float(q[0].item()),
        "p50": float(q[1].item()),
        "p90": float(q[2].item()),
    }


def _rms(value: torch.Tensor, active: torch.Tensor | None = None) -> float:
    if active is not None:
        value = value[active]
    if not value.numel():
        return 0.0
    return float(value.float().square().mean().sqrt().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if not a.numel() or not b.numel():
        return None
    value = F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0, eps=1.0e-8)
    return float(value.clamp(-1.0, 1.0).item())


def _mean(values) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def _compose_dataset_config(dataset: str, seed: int):
    from scripts.analyze_mvcge_mag_v3a_screen import _compose_dataset_config as compose

    return compose(dataset, seed)


def load_features_and_edges(dataset: str, seed: int = 42):
    from scripts.analyze_mvcge_mag_v3a_screen import load_features_and_edges as load

    return load(dataset, seed)


def _feature_dims(dataset: str, seed: int):
    from scripts.analyze_mvcge_mag_v3a_screen import _feature_dims as dims

    return dims(dataset, seed)


def _model_config(variant: str):
    cfg = OmegaConf.load(ROOT / "configs" / "model" / "mvcge_mag_v3c.yaml")
    cfg.variant = variant
    return OmegaConf.create({"model": OmegaConf.to_container(cfg, resolve=True)})


def _data_info(dataset: str, seed: int, x: torch.Tensor) -> dict[str, int]:
    cfg = _compose_dataset_config(dataset, seed)
    input_dim, text_dim, visual_dim = _feature_dims(dataset, seed)
    if input_dim != x.size(1):
        raise ValueError(f"Feature dimensions disagree for {dataset}: {input_dim} != {x.size(1)}")
    return {
        "input_dim": input_dim,
        "text_dim": text_dim,
        "visual_dim": visual_dim,
        "num_nodes": int(x.size(0)),
        "num_classes": int(cfg.dataset.num_classes),
    }


def _cross_role_fields(text: dict[str, torch.Tensor], visual: dict[str, torch.Tensor]):
    ts, vs = text["supportive_mask"], visual["supportive_mask"]
    td, vd = ~ts, ~vs
    total = max(ts.numel(), 1)
    pairs = {
        "fraction_support_text_support_visual": ts & vs,
        "fraction_support_text_discrepant_visual": ts & vd,
        "fraction_discrepant_text_support_visual": td & vs,
        "fraction_discrepant_text_discrepant_visual": td & vd,
    }
    union_support = ts | vs
    union_disc = td | vd
    support_jaccard = float((ts & vs).sum().item()) / max(int(union_support.sum().item()), 1)
    disc_jaccard = float((td & vd).sum().item()) / max(int(union_disc.sum().item()), 1)
    result = {name: float(mask.sum().item()) / total for name, mask in pairs.items()}
    result.update(
        {
            "role_disagreement_fraction": float((ts ^ vs).sum().item()) / total,
            "support_mask_jaccard_text_visual": support_jaccard,
            "discrepant_mask_jaccard_text_visual": disc_jaccard,
        }
    )
    return result


@torch.no_grad()
def role_preflight(device: str = "cuda:0") -> list[dict[str, Any]]:
    from src.models.mvcge_mag_v3c import Model

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested preflight device {device} is unavailable")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rows = []
    for dataset in DATASETS:
        x_cpu, edge_cpu = load_features_and_edges(dataset, 42)
        info = _data_info(dataset, 42, x_cpu)
        model = Model(_model_config("F0_raw"), info).to(device).eval()
        x, edge = x_cpu.to(device), edge_cpu.to(device)
        inputs = model._split_modalities(x)
        src, dst, raw_norm, physical_active, loops = model._normalized_operator(
            edge, x.size(0), x.dtype
        )
        roles = {}
        for modality in ("text", "visual"):
            roles[modality] = model._role_partition(inputs[modality], src, dst, raw_norm)
        cross = _cross_role_fields(roles["text"], roles["visual"])
        n_active = max(int(physical_active.sum().item()), 1)
        raw_norm_l2 = torch.linalg.vector_norm(raw_norm)
        for modality in ("text", "visual"):
            role = roles[modality]
            edge_n = int(src.numel())
            cosine_stats = _stats(role["semantic_cosine"])
            mu_stats = _stats(role["local_mean"][physical_active])
            score_stats = _stats(role["role_score"])
            support_count = int(role["supportive_mask"].sum().item())
            discrepant_count = int(role["discrepant_mask"].sum().item())
            support_nodes = int((role["support_active"] & physical_active).sum().item())
            discrepant_nodes = int((role["discrepant_active"] & physical_active).sum().item())
            row = {
                "dataset": dataset,
                "seed": "fixed",
                "modality": modality,
                "physical_edge_count": edge_n,
                "physical_active_node_count": int(physical_active.sum().item()),
                "input_self_loops_removed": loops,
                "semantic_cosine_mean": cosine_stats["mean"],
                "semantic_cosine_std": cosine_stats["std"],
                "semantic_cosine_p10": cosine_stats["p10"],
                "semantic_cosine_p50": cosine_stats["p50"],
                "semantic_cosine_p90": cosine_stats["p90"],
                "local_mu_mean_active_nodes": mu_stats["mean"],
                "local_mu_std_active_nodes": mu_stats["std"],
                "local_mu_p10_active_nodes": mu_stats["p10"],
                "local_mu_p50_active_nodes": mu_stats["p50"],
                "local_mu_p90_active_nodes": mu_stats["p90"],
                "role_score_mean": score_stats["mean"],
                "role_score_std": score_stats["std"],
                "role_score_p10": score_stats["p10"],
                "role_score_p50": score_stats["p50"],
                "role_score_p90": score_stats["p90"],
                "support_edge_count": support_count,
                "discrepant_edge_count": discrepant_count,
                "support_edge_fraction": support_count / max(edge_n, 1),
                "discrepant_edge_fraction": discrepant_count / max(edge_n, 1),
                "support_node_coverage": support_nodes / n_active,
                "discrepant_node_coverage": discrepant_nodes / n_active,
                "support_weight_l2_fraction_of_raw": float(torch.linalg.vector_norm(role["support_norm_weight"]).item() / (raw_norm_l2.item() + model.eps)),
                "discrepant_weight_l2_fraction_of_raw": float(torch.linalg.vector_norm(role["discrepant_norm_weight"]).item() / (raw_norm_l2.item() + model.eps)),
                "partition_weight_max_abs_error": float(role["partition_max_abs_error"].item()),
                **cross,
            }
            rows.append(row)
            role.pop("semantic_features", None)
        del model, x, edge, inputs, roles
        if "cuda" in device:
            torch.cuda.empty_cache()
    return rows


def _run_directory(row: dict[str, Any], root: Path, output_root: Path):
    mode = str(row.get("mode", "formal"))
    run_dir = output_root / mode / "runs" / row["dataset"] / f"seed_{int(row['seed'])}" / row["variant"]
    return run_dir, root / row["checkpoint_path"]


def _assert_finite_tensors(values: dict[str, torch.Tensor], where: str) -> None:
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {name} in {where}")


def _action_span_ratio(basis: torch.Tensor, target: torch.Tensor, eps: float) -> tuple[float, float | None]:
    """Projection residual from small Gram matrices; never forms a node-space projector."""
    if not basis.numel() or not target.numel():
        return 0.0, None
    gram = basis @ basis.T
    cross = basis @ target.T
    coef = torch.linalg.pinv(gram) @ cross
    residual_sq = (
        target.square().sum()
        - 2.0 * torch.trace(coef.T @ cross)
        + torch.trace(coef.T @ gram @ coef)
    ).clamp_min(0.0)
    ratio = float(torch.sqrt(residual_sq).item() / (torch.linalg.vector_norm(target).item() + eps))
    condition = float(torch.linalg.cond(gram).item())
    return ratio, condition if math.isfinite(condition) else None


def _trajectory_cos_delta(a: torch.Tensor, b: torch.Tensor, active: torch.Tensor, eps: float):
    aa, bb = a[active], b[active]
    return _cosine(aa, bb), _rms(bb - aa) / (_rms(aa) + eps)


def _checkpoint_paths(row, *, root=ROOT, output_root=OUTPUT_ROOT):
    return _run_directory(row, root, output_root)


@torch.no_grad()
def audit_checkpoint(
    row: dict[str, Any],
    *,
    root: Path = ROOT,
    output_root: Path = OUTPUT_ROOT,
    device_name: str = "cuda:0",
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    from src.models.mvcge_mag_v3c import Model

    dataset, seed, variant = row["dataset"], int(row["seed"]), row["variant"]
    run_dir, checkpoint_path = _checkpoint_paths(row, root=root, output_root=output_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    if cfg.task.get("evaluate_test", True) is not False:
        raise AssertionError(f"Test evaluation enabled in {run_dir}")
    cfg.model.variant = variant
    model = Model(cfg, checkpoint["data_info"]).to(device_name).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if cached_data is None:
        cached_data = load_features_and_edges(dataset, seed)
    x_cpu, edge_cpu = cached_data
    x, edge = x_cpu.to(device_name), edge_cpu.to(device_name)
    z, _, _, aux, info = model(x, edge, return_details=True)
    test_present = any(str(key).lower().startswith("test") for key in row.get("metrics", {}))
    if test_present:
        raise AssertionError(f"Test metrics found in {dataset}/{seed}/{variant}")

    active = info["active_nodes"]
    role_partition_valid = True
    role_trajectories_finite = True
    max_partition_error = 0.0
    role_rows = {}
    for modality in ("text", "visual"):
        item = info["details"][modality]
        _assert_finite_tensors(
            {
                key: item[key]
                for key in (
                    "prior", "raw_trajectory", "support_trajectory",
                    "discrepant_smooth_trajectory", "discrepant_signed_trajectory",
                    "raw_profiles", "support_profiles", "discrepant_smooth_profiles",
                    "discrepant_signed_profiles", "expert_inputs", "expert_outputs",
                    "mixture", "scaled_mixture", "semantic_cosine", "local_mean",
                    "role_score", "support_norm_weight", "discrepant_norm_weight",
                    "selection_logits", "route_weights", "strength",
                )
            },
            f"{dataset}/{seed}/{variant}/{modality}",
        )
        if not torch.equal(item["supportive_mask"] | item["discrepant_mask"], torch.ones_like(item["supportive_mask"])):
            role_partition_valid = False
        if torch.any(item["supportive_mask"] & item["discrepant_mask"]):
            role_partition_valid = False
        error = float(item["partition_max_abs_error"].item())
        max_partition_error = max(max_partition_error, error)
        if error > 1.0e-7:
            role_partition_valid = False
        if not torch.allclose(
            item["support_norm_weight"] + item["discrepant_norm_weight"],
            item["raw_operator_weight"], atol=1.0e-7, rtol=0.0,
        ):
            role_partition_valid = False
        for key in ("support_trajectory", "discrepant_smooth_trajectory", "discrepant_signed_trajectory"):
            role_trajectories_finite &= bool(torch.isfinite(item[key]).all())
        role_rows[modality] = item

    routing_rows = []
    strength_rows = []
    profile_rows = []
    similarity_rows = []
    role_profile_rows = []
    trajectory_rows = []
    novelty_rows = []
    chosen_pairs = {}
    active_nodes = max(int(active.sum().item()), 1)
    alpha_init = torch.tensor(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        dtype=model.alpha_raw.dtype, device=model.alpha_raw.device,
    ) * 0.5
    alpha = model._effective_alpha(model.alpha_raw, model.eps)
    mix_values = torch.sigmoid(model.role_mix_raw)

    # Per-checkpoint action-span novelty for all requested role spaces.
    for modality in ("text", "visual"):
        item = role_rows[modality]
        raw = item["raw_trajectory"][:, active].float().reshape(4, -1)
        support = item["support_trajectory"][:, active].float().reshape(4, -1)
        disc_smooth = item["discrepant_smooth_trajectory"][:, active].float().reshape(4, -1)
        disc_signed = item["discrepant_signed_trajectory"][:, active].float().reshape(4, -1)
        dual_smooth = torch.cat([support, disc_smooth], dim=0)
        functional = torch.cat([support, disc_signed], dim=0)
        values = {}
        for name, basis, target in (
            ("support_outside_raw", raw, support),
            ("disc_smooth_outside_raw", raw, disc_smooth),
            ("disc_signed_outside_raw", raw, disc_signed),
            ("dual_smooth_union_outside_raw", raw, dual_smooth),
            ("functional_union_outside_raw", raw, functional),
            ("disc_signed_outside_dual_smooth", dual_smooth, disc_signed),
        ):
            ratio, condition = _action_span_ratio(basis, target, model.eps)
            values[f"{name}_span_ratio"] = ratio
            values[f"{name}_gram_condition"] = condition
        novelty_rows.append(
            {"dataset": dataset, "seed": seed, "variant": variant, "modality": modality, **values}
        )

    for modality in ("text", "visual"):
        item = role_rows[modality]
        route = info["router"][modality]
        pair = tuple(sorted(int(value) for value in route["top_indices"][0].tolist()))
        chosen_pairs[modality] = pair
        # Compare every action family at each hop on the same physical-active nodes.
        comparisons = (
            ("raw_vs_support", "raw_trajectory", "support_trajectory"),
            ("raw_vs_disc_smooth", "raw_trajectory", "discrepant_smooth_trajectory"),
            ("raw_vs_disc_signed", "raw_trajectory", "discrepant_signed_trajectory"),
            ("support_vs_disc_smooth", "support_trajectory", "discrepant_smooth_trajectory"),
            ("support_vs_disc_signed", "support_trajectory", "discrepant_signed_trajectory"),
            ("disc_smooth_vs_disc_signed", "discrepant_smooth_trajectory", "discrepant_signed_trajectory"),
        )
        trajectory = {"dataset": dataset, "seed": seed, "variant": variant, "modality": modality}
        for hop in range(4):
            for name, akey, bkey in comparisons:
                cosine, delta = _trajectory_cos_delta(
                    item[akey][hop], item[bkey][hop], active, model.eps
                )
                trajectory[f"{name}_cosine_{hop + 1}"] = cosine
                trajectory[f"{name}_relative_rms_delta_{hop + 1}"] = delta
            for name, key in (
                ("raw", "raw_trajectory"),
                ("support", "support_trajectory"),
                ("disc_smooth", "discrepant_smooth_trajectory"),
                ("disc_signed", "discrepant_signed_trajectory"),
            ):
                trajectory[f"{name}_rms_{hop + 1}"] = _rms(item[key][hop], active)
        trajectory_rows.append(trajectory)

        selected = pair
        rec = {
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            "modality": modality,
            "selected_expert_pair": f"{selected[0]}-{selected[1]}",
            "strength": float(route["strength"][0].item()),
            "same_pair_text_visual": None,
            "union_experts_text_visual": None,
            "dead_expert_slots_text_visual": None,
        }
        for expert in range(4):
            rec[f"top2_weight_expert_{expert}"] = float(route["route_weights"][0, expert].item())
            rec[f"dense_prob_expert_{expert}"] = float(route["dense_probs"][0, expert].item())
        routing_rows.append(rec)

        for expert in range(4):
            raw_param = model.alpha_raw.detach()[expert]
            prof = {
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "expert_id": expert,
                "alpha_drift_from_init": float(torch.linalg.vector_norm(raw_param - alpha_init[expert]).item()),
                "alpha_normalized_drift_from_init": float(torch.linalg.vector_norm(alpha[expert] - alpha_init[expert] / (alpha_init[expert].norm() + model.eps)).item()),
            }
            for order in range(4):
                prof[f"alpha_raw_{order + 1}"] = float(raw_param[order].item())
                prof[f"alpha_normalized_{order + 1}"] = float(alpha[expert, order].item())
            profile_rows.append(prof)

            support_p = item["support_profiles"][expert]
            smooth_p = item["discrepant_smooth_profiles"][expert]
            signed_p = item["discrepant_signed_profiles"][expert]
            lam = float(mix_values[expert].item())
            role_profile_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "modality": modality,
                    "expert_id": expert,
                    "support_profile_rms": _rms(support_p, active),
                    "disc_smooth_profile_rms": _rms(smooth_p, active),
                    "disc_signed_profile_rms": _rms(signed_p, active),
                    "support_disc_smooth_cosine": _cosine(support_p[active], smooth_p[active]),
                    "support_disc_smooth_relative_rms_delta": _rms(smooth_p[active] - support_p[active]) / (_rms(support_p[active]) + model.eps),
                    "support_disc_signed_cosine": _cosine(support_p[active], signed_p[active]),
                    "support_disc_signed_relative_rms_delta": _rms(signed_p[active] - support_p[active]) / (_rms(support_p[active]) + model.eps),
                    "disc_smooth_disc_signed_cosine": _cosine(smooth_p[active], signed_p[active]),
                    "disc_smooth_disc_signed_relative_rms_delta": _rms(signed_p[active] - smooth_p[active]) / (_rms(smooth_p[active]) + model.eps),
                    "lambda": lam if variant in {"F2_role_dual_smooth", "F3_role_functional"} else None,
                    "support_contribution_rms": _rms((1.0 - lam) * support_p, active) if variant in {"F2_role_dual_smooth", "F3_role_functional"} else None,
                    "discrepant_contribution_rms": _rms(lam * (smooth_p if variant == "F2_role_dual_smooth" else signed_p), active) if variant in {"F2_role_dual_smooth", "F3_role_functional"} else None,
                    "mixed_profile_rms": _rms(item["expert_inputs"][expert], active),
                }
            )

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
                    "flattened_cosine": _cosine(item["expert_outputs"][a][active], item["expert_outputs"][b][active]),
                    "mean_node_cosine": float(F.cosine_similarity(item["expert_outputs"][a][active].float(), item["expert_outputs"][b][active].float(), dim=-1, eps=model.eps).clamp(-1.0, 1.0).mean().item()),
                }
            )

        prior = item["prior"]
        scaled = item["scaled_mixture"]
        mixture = item["mixture"]
        strength_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "variant": variant,
                "modality": modality,
                "prior_rms": _rms(prior, active),
                "unscaled_expert_mixture_rms": _rms(mixture, active),
                "scaled_correction_rms": _rms(scaled, active),
                "correction_to_prior_rms_ratio": _rms(scaled, active) / (_rms(prior, active) + model.eps),
                "scaled_correction_prior_cosine": _cosine(scaled[active], prior[active]),
                "strength": float(route["strength"][0].item()),
            }
        )

    same = chosen_pairs["text"] == chosen_pairs["visual"]
    union = set(chosen_pairs["text"]) | set(chosen_pairs["visual"])
    dead = sorted(set(range(4)) - union)
    for record in routing_rows:
        record["same_pair_text_visual"] = same
        record["union_experts_text_visual"] = len(union)
        record["dead_expert_slots_text_visual"] = len(dead)
        record["dead_expert_indices_text_visual"] = ",".join(map(str, dead))

    finite = bool(torch.isfinite(z).all() and torch.isfinite(aux))
    routes_valid = True
    max_strength_delta = 0.0
    for modality in ("text", "visual"):
        route = info["router"][modality]
        routes_valid &= bool(
            torch.equal(
                route["selection_logits"],
                route["selection_logits"][:1].expand_as(route["selection_logits"]),
            )
        )
        routes_valid &= bool(
            torch.equal(
                route["strength_logit"],
                route["strength_logit"][:1].expand_as(route["strength_logit"]),
            )
        )
        max_strength_delta = max(
            max_strength_delta,
            float((route["strength"] - route["strength"][:1]).abs().max().item()),
        )
        routes_valid &= bool(torch.equal((route["route_weights"] > 0).sum(-1), torch.full((x.size(0),), 2, device=x.device)))
        routes_valid &= bool(torch.allclose(route["route_weights"].sum(-1), torch.ones(x.size(0), device=x.device), atol=1.0e-6, rtol=0.0))

    result = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "finite": finite,
        "test_metrics_present": test_present,
        "task_evaluate_test": False,
        "role_partition_valid": role_partition_valid,
        "role_partition_max_abs_error": max_partition_error,
        "role_trajectories_finite": role_trajectories_finite,
        "static_routing_verified": routes_valid and max_strength_delta <= 1.0e-7,
        "max_static_strength_node_difference": max_strength_delta,
        "top2_verified": routes_valid,
        "role_mix_values": [float(value) for value in mix_values.detach().cpu().tolist()],
        "parameter_count_model": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count_model": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_count_classifier": int(row["metadata"]["classifier_parameters"]),
        "trajectory_rows": trajectory_rows,
        "novelty_rows": novelty_rows,
        "role_profile_rows": role_profile_rows,
        "routing_rows": routing_rows,
        "strength_rows": strength_rows,
        "profile_rows": profile_rows,
        "similarity_rows": similarity_rows,
    }
    if not finite or not role_partition_valid or not role_trajectories_finite or not routes_valid:
        raise AssertionError(f"Selected checkpoint audit failed: {dataset}/{seed}/{variant}")
    del model, x, edge, z, checkpoint
    if "cuda" in device_name:
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def audit_f0_implementation(
    row: dict[str, Any],
    *,
    device_name: str = "cuda:0",
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Compare one F0 checkpoint through V3C F0 and V3A C0 implementations."""
    from src.models.mvcge_mag_v3a import Model as V3AModel
    from src.models.mvcge_mag_v3c import Model as V3CModel

    if row["variant"] != "F0_raw":
        raise ValueError("F0 implementation audit only accepts F0_raw")
    run_dir, checkpoint_path = _checkpoint_paths(
        row, root=ROOT, output_root=OUTPUT_ROOT
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    cfg.model.variant = "F0_raw"
    v3a_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    v3a_cfg.model.name = "mvcge_mag_v3a"
    v3a_cfg.model.variant = "C0_raw"
    shared_state = {
        key: value for key, value in checkpoint["model_state"].items()
        if key != "role_mix_raw"
    }
    v3c = V3CModel(cfg, checkpoint["data_info"]).to("cpu").eval()
    v3c.load_state_dict(checkpoint["model_state"], strict=True)
    v3a = V3AModel(v3a_cfg, checkpoint["data_info"]).to("cpu").eval()
    incompatible = v3a.load_state_dict(shared_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError("F0 V3C/V3A shared state keys differ")
    if cached_data is None:
        cached_data = load_features_and_edges(row["dataset"], int(row["seed"]))
    x_cpu, edge_cpu = cached_data
    z0, _, _, aux0, _ = v3a(x_cpu, edge_cpu)
    z1, _, _, aux1, _ = v3c(x_cpu, edge_cpu)
    z_diff = float((z0 - z1).abs().max().item())
    aux_diff = float((aux0 - aux1).abs().max().item())
    return {
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "audit_device": "cpu",
        "state_keys_compatible": True,
        "forward_exact": torch.equal(z0, z1) and torch.equal(aux0, aux1),
        "forward_allclose_1e7": torch.allclose(z0, z1, atol=1.0e-7, rtol=0.0) and torch.allclose(aux0, aux1, atol=1.0e-7, rtol=0.0),
        "z_max_abs_difference": z_diff,
        "aux_max_abs_difference": aux_diff,
    }


@torch.no_grad()
def audit_f0_compatible_checkpoint(
    row: dict[str, Any],
    *,
    device_name: str = "cuda:0",
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    from src.models.mvcge_mag_v3a import Model as V3AModel
    from src.models.mvcge_mag_v3c import Model as V3CModel

    dataset, seed = row["dataset"], int(row["seed"])
    old_path = V3A_ROOT / "data" / "run_rows.json"
    old_rows = json.loads(old_path.read_text(encoding="utf-8"))
    old_row = next(
        item for item in old_rows
        if item["dataset"] == dataset and int(item["seed"]) == seed and item["variant"] == "C0_raw"
    )
    old_run_dir, old_ckpt_path = _run_directory(old_row, ROOT, V3A_OUTPUT_ROOT)
    new_run_dir, new_ckpt_path = _run_directory(row, ROOT, OUTPUT_ROOT)
    old_ckpt = torch.load(old_ckpt_path, map_location="cpu", weights_only=False)
    new_ckpt = torch.load(new_ckpt_path, map_location="cpu", weights_only=False)
    old_cfg = OmegaConf.load(old_run_dir / "hydra" / ".hydra" / "config.yaml")
    new_cfg = OmegaConf.load(new_run_dir / "hydra" / ".hydra" / "config.yaml")
    old_cfg.model.variant = "C0_raw"
    new_cfg.model.variant = "F0_raw"
    old_model = V3AModel(old_cfg, old_ckpt["data_info"]).to(device_name).eval()
    new_model = V3CModel(new_cfg, new_ckpt["data_info"]).to(device_name).eval()
    old_state = old_ckpt["model_state"]
    new_state = new_ckpt["model_state"]
    new_for_old = {key: value for key, value in new_state.items() if key != "role_mix_raw"}
    old_keys, new_keys = set(old_state), set(new_for_old)
    missing = sorted(old_keys - new_keys)
    extra = sorted(new_keys - old_keys)
    shape_mismatches = sorted(
        key for key in old_keys & new_keys if old_state[key].shape != new_for_old[key].shape
    )
    loadable = {key: new_for_old[key] for key in old_keys & new_keys if key not in shape_mismatches}
    old_model.load_state_dict(old_state, strict=True)
    new_model.load_state_dict(new_state, strict=True)
    exact_state = not missing and not extra and not shape_mismatches and all(
        torch.equal(old_state[key], new_for_old[key]) for key in old_keys
    )
    max_state_diff = max(
        (float((old_state[key].float() - new_for_old[key].float()).abs().max().item()) for key in loadable),
        default=0.0,
    )
    if cached_data is None:
        cached_data = load_features_and_edges(dataset, seed)
    x_cpu, edge_cpu = cached_data
    x, edge = x_cpu.to(device_name), edge_cpu.to(device_name)
    z_old, _, _, aux_old, _ = old_model(x, edge)
    z_new, _, _, aux_new, _ = new_model(x, edge)
    z_diff = float((z_old - z_new).abs().max().item())
    aux_diff = float((aux_old - aux_new).abs().max().item())
    return {
        "dataset": dataset,
        "seed": seed,
        "old_val_acc": float(old_row["metrics"]["val_acc"]),
        "new_val_acc": float(row["metrics"]["val_acc"]),
        "C0_minus_F0_acc_pp": 100.0 * (float(row["metrics"]["val_acc"]) - float(old_row["metrics"]["val_acc"])),
        "old_val_macro_f1": float(old_row["metrics"]["val_macro_f1"]),
        "new_val_macro_f1": float(row["metrics"]["val_macro_f1"]),
        "C0_minus_F0_macro_f1_pp": 100.0 * (float(row["metrics"]["val_macro_f1"]) - float(old_row["metrics"]["val_macro_f1"])),
        "old_best_epoch": int(old_row["metadata"]["best_epoch"]),
        "new_best_epoch": int(row["metadata"]["best_epoch"]),
        "best_epoch_difference": int(row["metadata"]["best_epoch"]) - int(old_row["metadata"]["best_epoch"]),
        "missing_shared_state_keys": len(missing),
        "extra_shared_state_keys": len(extra),
        "shared_state_shape_mismatches": len(shape_mismatches),
        "shared_state_exact": exact_state,
        "shared_state_max_abs_difference": max_state_diff,
        "forward_z_exact": torch.equal(z_old, z_new),
        "forward_aux_exact": torch.equal(aux_old, aux_new),
        "forward_allclose_1e7": torch.allclose(z_old, z_new, atol=1.0e-7, rtol=0.0) and torch.allclose(aux_old, aux_new, atol=1.0e-7, rtol=0.0),
        "forward_z_max_abs_difference": z_diff,
        "forward_aux_max_abs_difference": aux_diff,
    }


def campaign_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            groups.append((dataset, variant, [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]))
    for variant in VARIANTS:
        groups.append(("ALL", variant, [r for r in rows if r["variant"] == variant]))
    for dataset, variant, group in groups:
        for metric in ("val_acc", "val_macro_f1"):
            values = [float(r["metrics"][metric]) for r in group]
            result.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "population_std": statistics.pstdev(values),
                    "n": len(values),
                }
            )
        result.append(
            {
                "dataset": dataset,
                "variant": variant,
                "metric": "best_epoch",
                "mean": statistics.fmean(float(r["metadata"]["best_epoch"]) for r in group),
                "population_std": statistics.pstdev(float(r["metadata"]["best_epoch"]) for r in group),
                "n": len(group),
            }
        )
        result.append(
            {
                "dataset": dataset,
                "variant": variant,
                "metric": "model_parameters",
                "mean": statistics.fmean(float(r["metadata"]["model_parameters"]) for r in group),
                "population_std": 0.0,
                "n": len(group),
            }
        )
    return result


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in rows}
    output = []
    for candidate, baseline in PAIRS:
        acc, f1 = [], []
        record: dict[str, Any] = {
            "comparison": f"{candidate} - {baseline}",
            "n_paired_runs": 9,
        }
        for dataset in DATASETS:
            ds_acc, ds_f1 = [], []
            for seed in SEEDS:
                new = keyed[(dataset, seed, candidate)]["metrics"]
                old = keyed[(dataset, seed, baseline)]["metrics"]
                da = 100.0 * (float(new["val_acc"]) - float(old["val_acc"]))
                df = 100.0 * (float(new["val_macro_f1"]) - float(old["val_macro_f1"]))
                acc.append(da)
                f1.append(df)
                ds_acc.append(da)
                ds_f1.append(df)
            record[f"{dataset}_delta_accuracy_pp_mean"] = statistics.fmean(ds_acc)
            record[f"{dataset}_delta_macro_f1_pp_mean"] = statistics.fmean(ds_f1)
            record[f"{dataset}_positive_accuracy_seeds"] = sum(v > 0 for v in ds_acc)
            record[f"{dataset}_positive_macro_f1_seeds"] = sum(v > 0 for v in ds_f1)
        record["overall_delta_accuracy_pp"] = statistics.fmean(acc)
        record["overall_delta_macro_f1_pp"] = statistics.fmean(f1)
        record["positive_accuracy_pairs"] = sum(v > 0 for v in acc)
        record["positive_macro_f1_pairs"] = sum(v > 0 for v in f1)
        output.append(record)
    return output


def _stable_positive(row: dict[str, Any]) -> bool:
    return (
        float(row["overall_delta_accuracy_pp"]) > 0
        and float(row["overall_delta_macro_f1_pp"]) > 0
        and int(row["positive_accuracy_pairs"]) >= 6
        and sum(float(row[f"{d}_delta_accuracy_pp_mean"]) > 0 for d in DATASETS) >= 2
    )


def _approximately_equal(row: dict[str, Any]) -> bool:
    return (
        abs(float(row["overall_delta_accuracy_pp"])) <= 0.15
        and abs(float(row["overall_delta_macro_f1_pp"])) <= 0.50
    )


def _role_summary(rows, column):
    output = []
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            vals = [float(r[column]) for r in rows if r["dataset"] == dataset and r["modality"] == modality]
            output.append({"dataset": dataset, "modality": modality, "mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)})
    return output


def _report(summary, paired, roles, trajectories, novelty, role_profiles, profiles, similarities, routing, strengths, historical):
    pair_map = {row["comparison"]: row for row in paired}
    lines = [
        "# MvCGE-MAG V3C: Functional Role Action-Space Screen",
        "",
        "## Protocol and provenance",
        "",
        "- Parent `075b12b497d619d0b86d25f264fa97402ac74778`; validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; four variants; 36 runs.",
        "- F0 follows V3A C0 / V2.2 R0 construction and output. Every variant uses modality-static Top-2 selection and strength, four shared experts, and one shared alpha profile.",
        "- Supportive and Discrepant roles use detached raw modality features and observed physical edges only. Role edge weights are masks of the raw normalized operator; no channel-specific renormalization or new edges.",
        "- `task.evaluate_test=false`; checkpoints selected by Validation Accuracy. No Test metrics, labels in preflight, HPO, LP, significance testing, learned edge router, node routing, or cross-modal operators.",
        "- Discrepant is a fixed local-relative semantic-consistency label; it does not assert ground-truth heterophily.",
        "",
        "## Validation results",
        "",
        "Run-level means ± population standard deviations; paired differences below are descriptive percentage points.",
        "",
        "| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Model parameters |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in (*DATASETS, "ALL"):
        for variant in VARIANTS:
            def get(metric):
                return next(r for r in summary if r["dataset"] == dataset and r["variant"] == variant and r["metric"] == metric)
            a, f, epoch, params = get("val_acc"), get("val_macro_f1"), get("best_epoch"), get("model_parameters")
            lines.append(f"| {dataset} | {variant} | {100*a['mean']:.2f}% ± {100*a['population_std']:.2f}% | {100*f['mean']:.2f}% ± {100*f['population_std']:.2f}% | {epoch['mean']:.1f} | {params['mean']:.0f} |")

    lines.extend(["", "## Required paired comparisons", "", "| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Accuracy / Macro-F1) |", "|---|---:|---:|---:|"])
    labels = {
        "F1_support - F0_raw": "F1 − F0: Supportive-only smoothing vs raw physical context",
        "F2_role_dual_smooth - F1_support": "F2 − F1: add Discrepant ordinary smoothing",
        "F2_role_dual_smooth - F0_raw": "F2 − F0: dual-role smoothing vs raw context",
        "F3_role_functional - F2_role_dual_smooth": "F3 − F2: signed difference vs Discrepant smoothing",
        "F3_role_functional - F1_support": "F3 − F1: functional role vs Supportive-only",
        "F3_role_functional - F0_raw": "F3 − F0: full functional-role action space vs raw",
    }
    for comparison, label in labels.items():
        row = pair_map[comparison]
        lines.append(f"| {label} | {float(row['overall_delta_accuracy_pp']):+.3f} pp | {float(row['overall_delta_macro_f1_pp']):+.3f} pp | {row['positive_accuracy_pairs']}/9 / {row['positive_macro_f1_pairs']}/9 |")
    lines.extend(["", "Per-dataset means and positive-seed counts are in `data/paired_comparisons.csv`; no significance testing was performed.", "", "## Fixed role partition and cross-modal diagnostics", "", "Statistics below are from raw modality features and physical edges only. Local-μ summaries use physical-active nodes; node coverage is the fraction of physical-active nodes receiving at least one incoming edge in that role.", "", "| Dataset | Modality | Support edges | Discrepant edges | Support node coverage | Discrepant node coverage | Support/raw weight L2 | Discrepant/raw weight L2 |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in roles:
        lines.append(f"| {r['dataset']} | {r['modality']} | {r['support_edge_fraction']:.3f} | {r['discrepant_edge_fraction']:.3f} | {r['support_node_coverage']:.3f} | {r['discrepant_node_coverage']:.3f} | {r['support_weight_l2_fraction_of_raw']:.3f} | {r['discrepant_weight_l2_fraction_of_raw']:.3f} |")
    lines.extend(["", "The same-edge Text/Visual role contingency, disagreement fraction, and Supportive/Discrepant mask Jaccard values are recorded in `data/role_partition_diagnostics.csv`. Partition maximum absolute error is checked against 1e-7. Role score, semantic cosine and local μ quantiles are in the same file.", "", "## Trajectory comparisons", "", "For each hop, cosine is computed on physical-active nodes; relative RMS delta uses the first named action family as denominator. Full per-checkpoint values are in `data/trajectory_diagnostics.csv`.", ""])
    for variant in VARIANTS:
        for modality in ("text", "visual"):
            group = [r for r in trajectories if r["variant"] == variant and r["modality"] == modality]
            if not group:
                continue
            cos = [statistics.fmean(float(r[f"{name}_cosine_{hop}"]) for r in group) for name in ("raw_vs_support", "raw_vs_disc_smooth", "raw_vs_disc_signed", "support_vs_disc_smooth", "support_vs_disc_signed", "disc_smooth_vs_disc_signed") for hop in range(1, 5)]
            delta = [statistics.fmean(float(r[f"{name}_relative_rms_delta_{hop}"]) for r in group) for name in ("raw_vs_support", "raw_vs_disc_smooth", "raw_vs_disc_signed", "support_vs_disc_smooth", "support_vs_disc_signed", "disc_smooth_vs_disc_signed") for hop in range(1, 5)]
            lines.append(f"- {variant}/{modality}: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `{[round(v, 4) for v in cos]}`; corresponding relative RMS-delta vectors: `{[round(v, 4) for v in delta]}`.")

    lines.extend(["", "## Action-space span novelty", "", "Ratios use flattened selected-checkpoint trajectories and small Gram/pseudoinverse identities; no large projection matrix is formed. Ratios are summarized across nine dataset-seed checkpoints by variant and modality.", "", "| Variant | Modality | support outside raw | discrepant smoothing outside raw | discrepant signed outside raw | dual-smooth union outside raw | functional union outside raw | signed outside dual-smooth |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    novelty_fields = ("support_outside_raw_span_ratio", "disc_smooth_outside_raw_span_ratio", "disc_signed_outside_raw_span_ratio", "dual_smooth_union_outside_raw_span_ratio", "functional_union_outside_raw_span_ratio", "disc_signed_outside_dual_smooth_span_ratio")
    for variant in VARIANTS:
        for modality in ("text", "visual"):
            group = [r for r in novelty if r["variant"] == variant and r["modality"] == modality]
            means = [statistics.fmean(float(r[name]) for r in group) for name in novelty_fields]
            lines.append("| " + variant + " | " + modality + " | " + " | ".join(f"{v:.4f}" for v in means) + " |")
    lines.append("Per-checkpoint novelty and Gram condition numbers for all variants are in `data/action_space_novelty.csv`.")

    lines.extend(["", "## Role profiles, λ, experts and correction scale", ""])
    for variant in ("F2_role_dual_smooth", "F3_role_functional"):
        group = [r for r in role_profiles if r["variant"] == variant and r["modality"] == "text"]
        lines.append(f"- {variant} λ mean/range by expert: " + "; ".join(f"E{e}: {_mean(r['lambda'] for r in group if int(r['expert_id'])==e):.4f} [{min(float(r['lambda']) for r in group if int(r['expert_id'])==e):.4f}, {max(float(r['lambda']) for r in group if int(r['expert_id'])==e):.4f}]" for e in range(4)) + ". Text/visual share the same λ parameter.")
    for variant in VARIANTS:
        group = [r for r in strengths if r["variant"] == variant]
        lines.append(f"- {variant} correction/prior RMS ratio mean {_mean(r['correction_to_prior_rms_ratio'] for r in group):.4f}; correction RMS {_mean(r['scaled_correction_rms'] for r in group):.4f}; prior RMS {_mean(r['prior_rms'] for r in group):.4f}; scaled-correction/prior cosine {_mean(r['scaled_correction_prior_cosine'] for r in group):.4f}.")
    output = [r for r in similarities if r["kind"] == "expert_output"]
    lines.append(f"- Shared expert output pair cosine: flattened mean {_mean(r['flattened_cosine'] for r in output):.4f}; mean-node mean {_mean(r['mean_node_cosine'] for r in output):.4f}. Alpha values/drifts, per-profile channel cosine and RMS, and per-pair similarities are in `data/expert_profiles.csv`, `data/role_profile_diagnostics.csv`, and `data/expert_similarity.csv`.")

    lines.extend(["", "## Static routing", "", "| Variant | Most common Text pair | Most common Visual pair | Same-pair rate | Mean expert union | Runs with a dead slot |", "|---|---|---|---:|---:|---:|"])
    for variant in VARIANTS:
        group = [r for r in routing if r["variant"] == variant and r["modality"] == "text"]
        text_counts = Counter(r["selected_expert_pair"] for r in group)
        visual_group = [r for r in routing if r["variant"] == variant and r["modality"] == "visual"]
        visual_counts = Counter(r["selected_expert_pair"] for r in visual_group)
        top_t, top_v = text_counts.most_common(1)[0], visual_counts.most_common(1)[0]
        lines.append(f"| {variant} | {top_t[0]} ({top_t[1]}/9) | {top_v[0]} ({top_v[1]}/9) | {_mean(float(r['same_pair_text_visual']) for r in group):.3f} | {_mean(r['union_experts_text_visual'] for r in group):.3f}/4 | {sum(int(r['dead_expert_slots_text_visual'])>0 for r in group)}/9 |")
    lines.append("Static unused experts are sparse global compositions, not node-router collapse; per-checkpoint Top-2 weights, dense probabilities, and strength are in `data/routing_diagnostics.csv`.")

    hist_acc = [float(r["C0_minus_F0_acc_pp"]) for r in historical]
    hist_f1 = [float(r["C0_minus_F0_macro_f1_pp"]) for r in historical]
    lines.extend(["", "## Historical F0 comparison against V3A C0", "", f"Across nine matched runs, F0−C0 mean deltas were Accuracy {statistics.fmean(hist_acc):+.3f} pp ({sum(v>0 for v in hist_acc)}/9 positive; range {min(hist_acc):+.3f} to {max(hist_acc):+.3f}) and Macro-F1 {statistics.fmean(hist_f1):+.3f} pp ({sum(v>0 for v in hist_f1)}/9 positive; range {min(hist_f1):+.3f} to {max(hist_f1):+.3f}). Mean best-epoch difference was {statistics.fmean(float(r['best_epoch_difference']) for r in historical):+.2f}. Shared state exact: {sum(bool(r['shared_state_exact']) for r in historical)}/9; forward allclose at 1e-7: {sum(bool(r['forward_allclose_1e7']) for r in historical)}/9. Maximum z/aux differences were {max(float(r['forward_z_max_abs_difference']) for r in historical):.3g}/{max(float(r['forward_aux_max_abs_difference']) for r in historical):.3g}. See `data/historical_f0_regression.csv` for each matched checkpoint."])

    c1 = pair_map["F1_support - F0_raw"]
    c2f1 = pair_map["F2_role_dual_smooth - F1_support"]
    c2 = pair_map["F2_role_dual_smooth - F0_raw"]
    c3c2 = pair_map["F3_role_functional - F2_role_dual_smooth"]
    c3f1 = pair_map["F3_role_functional - F1_support"]
    c3 = pair_map["F3_role_functional - F0_raw"]
    nmean = {name: _mean(r[name] for r in novelty if r["variant"] == "F3_role_functional") for name in novelty_fields}
    disagreement = _mean(r["role_disagreement_fraction"] for r in roles)
    novelty_near_zero = nmean["functional_union_outside_raw_span_ratio"] <= 1.0e-3
    high_disagreement = disagreement is not None and disagreement > 0.5
    lines.extend(["", "## Frozen descriptive interpretation and next-stage gate", "", "Stable-positive: overall Accuracy and Macro-F1 deltas both >0, at least 6/9 positive Accuracy pairs, and positive Accuracy means on at least 2/3 datasets. Approximately equal: absolute Accuracy delta ≤0.15 pp and absolute Macro-F1 delta ≤0.50 pp. For interpretation labels only, functional novelty ≤1e-3 is treated as near-zero; cross-modal disagreement >0.5 means a majority of physical edges change role. These are descriptive cutoffs, not statistical tests or tuning targets.", ""])
    s1, s2, s3, s5, s6 = map(_stable_positive, (c1, c2, c3c2, c3f1, c3))
    lines.append(f"- **A: {'Observed' if s1 else 'Not observed'}.** F1−F0 stable-positive={s1}; Supportive-only smoothing {'passes' if s1 else 'does not pass'} the stable-positive gate.")
    b = float(c1["overall_delta_accuracy_pp"]) < 0 and float(c1["overall_delta_macro_f1_pp"]) < 0 and _approximately_equal(c2)
    lines.append(f"- **B: {'Observed' if b else 'Not observed'}.** F1−F0 is below zero on both overall metrics={float(c1['overall_delta_accuracy_pp'])<0 and float(c1['overall_delta_macro_f1_pp'])<0}; F2≈F0={_approximately_equal(c2)}.")
    c = s3 and s5
    lines.append(f"- **C: {'Observed' if c else 'Not observed'}.** F3−F2 stable-positive={s3}; F3−F0 stable-positive={s5}.")
    d = s3 and not s5
    lines.append(f"- **D: {'Observed' if d else 'Not observed'}.** F3−F2 stable-positive={s3}; F3−F0 stable-positive={s5}. If observed, prefer testing Raw + functional-role residual before node routing.")
    e = _stable_positive(c2) and (_approximately_equal(c3c2) or (float(c3c2['overall_delta_accuracy_pp']) < 0 and float(c3c2['overall_delta_macro_f1_pp']) < 0))
    lines.append(f"- **E: {'Observed' if e else 'Not observed'}.** F2−F0 stable-positive={_stable_positive(c2)}; F3−F2 approximately equal or below on both metrics={_approximately_equal(c3c2) or (float(c3c2['overall_delta_accuracy_pp'])<0 and float(c3c2['overall_delta_macro_f1_pp'])<0)}.")
    f = not _stable_positive(c2) and not _stable_positive(c3) and not novelty_near_zero
    lines.append(f"- **F: {'Observed' if f else 'Not observed'}.** F2/F3 stable-positive vs F0={_stable_positive(c2)}/{_stable_positive(c3)}; mean functional-union novelty={nmean['functional_union_outside_raw_span_ratio']:.4f}. If observed, favor learned role assignment over static threshold changes.")
    lines.append(f"- **G: {'Observed' if novelty_near_zero else 'Not observed'}.** Mean functional-union outside-raw-span ratio={nmean['functional_union_outside_raw_span_ratio']:.4f}; descriptive near-zero cutoff=0.001.")
    h = s5 and high_disagreement
    lines.append(f"- **H: {'Observed' if h else 'Not observed'}.** F3−F0 stable-positive={s5}; mean Text/Visual role disagreement={disagreement:.4f}; majority-disagreement flag={high_disagreement}.")
    gate = s5 and not novelty_near_zero
    lines.extend(["", "## Next-stage decision", "", f"F3 vs F0 stable-positive: {s5}; functional action novelty near-zero: {novelty_near_zero}. The next-stage gate is {'met' if gate else 'not met'}. Stop this screen here; do not automatically start node-conditioned routing, role learning, or context augmentation.", "", "No ground-truth homophily/heterophily claim, causal claim, or significance claim is made.", "", "## Reproducibility artifacts", "", "- `data/environment.json`, `data/preflight_role_diagnostics.csv`, `data/preflight_summary.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`", "- `data/summary.csv`, `data/paired_comparisons.csv`, `data/role_partition_diagnostics.csv`, `data/trajectory_diagnostics.csv`, `data/action_space_novelty.csv`", "- `data/role_profile_diagnostics.csv`, `data/routing_diagnostics.csv`, `data/strength_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/historical_f0_regression.csv`", "- Checkpoints, training/Hydra logs, raw features and role masks remain under ignored server-local `outputs/`."])
    return "\n".join(lines) + "\n"


@torch.no_grad()
def analyze(device: str = "cuda:0") -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[V3C analysis] {device} unavailable; using CPU for read-only audits", flush=True)
        device = "cpu"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rows = json.loads((DATA_ROOT / "run_rows.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA_ROOT / "campaign_manifest.json").read_text(encoding="utf-8"))
    keys = {(r["dataset"], int(r["seed"]), r["variant"]) for r in rows if r.get("status") in {"completed", "reused"}}
    if len(rows) != 36 or len(keys) != 36 or manifest.get("completed_runs") != 36:
        raise RuntimeError("Analysis requires 36 unique completed runs")
    if manifest.get("task_evaluate_test") is not False:
        raise RuntimeError("Analysis requires task.evaluate_test=false")
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    audits = []
    for idx, row in enumerate(rows, 1):
        if row["dataset"] not in cache:
            cache[row["dataset"]] = load_features_and_edges(row["dataset"], int(row["seed"]))
        audit = audit_checkpoint(row, device_name=device, cached_data=cache[row["dataset"]])
        if audit["test_metrics_present"] or audit["task_evaluate_test"] is not False:
            raise AssertionError("Test isolation audit failed")
        audits.append(audit)
        print(f"[V3C audit] {idx}/36 {row['dataset']} seed={row['seed']} {row['variant']}", flush=True)

    summary = campaign_summaries(rows)
    paired = paired_comparisons(rows)
    trajectory_rows = [r for a in audits for r in a["trajectory_rows"]]
    novelty_rows = [r for a in audits for r in a["novelty_rows"]]
    role_profile_rows = [r for a in audits for r in a["role_profile_rows"]]
    routing_rows = [r for a in audits for r in a["routing_rows"]]
    strength_rows = [r for a in audits for r in a["strength_rows"]]
    profile_rows = [r for a in audits for r in a["profile_rows"]]
    similarity_rows = [r for a in audits for r in a["similarity_rows"]]

    old_rows = json.loads((V3A_ROOT / "data" / "run_rows.json").read_text(encoding="utf-8"))
    old_map = {(r["dataset"], int(r["seed"])): r for r in old_rows if r["variant"] == "C0_raw"}
    new_map = {(r["dataset"], int(r["seed"])): r for r in rows if r["variant"] == "F0_raw"}
    historical = []
    for dataset in DATASETS:
        for seed in SEEDS:
            if dataset not in cache:
                cache[dataset] = load_features_and_edges(dataset, seed)
            historical.append(
                audit_f0_compatible_checkpoint(
                    new_map[(dataset, seed)], device_name=device, cached_data=cache[dataset]
                )
            )
    if len(historical) != 9 or any(
        r["missing_shared_state_keys"] or r["extra_shared_state_keys"] or r["shared_state_shape_mismatches"]
        for r in historical
    ):
        raise AssertionError("F0/V3A C0 checkpoint shared-state compatibility failed")

    roles = role_preflight(device=device)
    for role_row in roles:
        if int(role_row["support_edge_count"]) == 0 or int(role_row["discrepant_edge_count"]) == 0 or float(role_row["partition_weight_max_abs_error"]) > 1.0e-7:
            raise AssertionError("Role partition diagnostic failed")

    tables = {
        "summary.csv": summary,
        "paired_comparisons.csv": paired,
        "role_partition_diagnostics.csv": roles,
        "trajectory_diagnostics.csv": trajectory_rows,
        "action_space_novelty.csv": novelty_rows,
        "role_profile_diagnostics.csv": role_profile_rows,
        "routing_diagnostics.csv": routing_rows,
        "strength_diagnostics.csv": strength_rows,
        "expert_profiles.csv": profile_rows,
        "expert_similarity.csv": similarity_rows,
        "historical_f0_regression.csv": historical,
    }
    for filename, table in tables.items():
        for row in table:
            for value in row.values():
                if not _finite_value(value):
                    raise FloatingPointError(f"Non-finite value in {filename}: {row}")
        write_csv(DATA_ROOT / filename, table)

    report = _report(summary, paired, roles, trajectory_rows, novelty_rows, role_profile_rows, profile_rows, similarity_rows, routing_rows, strength_rows, historical)
    (RESEARCH_ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    manifest.update(
        {
            "task_evaluate_test": False,
            "test_evaluation_verified_false": True,
            "test_metrics_absent": all(not a["test_metrics_present"] for a in audits),
            "selected_checkpoint_audits": len(audits),
            "all_selected_checkpoints_finite": all(a["finite"] for a in audits),
            "all_role_partitions_valid": all(a["role_partition_valid"] for a in audits),
            "all_role_trajectories_finite": all(a["role_trajectories_finite"] for a in audits),
            "all_static_routing_verified": all(a["static_routing_verified"] for a in audits),
            "all_top2_verified": all(a["top2_verified"] for a in audits),
            "historical_f0_forward_allclose_1e7_pairs": sum(bool(r["forward_allclose_1e7"]) for r in historical),
            "analysis_device": device,
            "artifacts_complete": True,
        }
    )
    write_json(DATA_ROOT / "campaign_manifest.json", manifest)
    print("[V3C analysis] wrote diagnostics and REPORT.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        rows = role_preflight(args.device)
        write_csv(DATA_ROOT / "preflight_role_diagnostics.csv", rows)
        print(f"[V3C role preflight] wrote {len(rows)} label-free dataset/modality records", flush=True)
    else:
        analyze(args.device)
