#!/usr/bin/env python3
"""Audit V2.2b validation-selected checkpoints without reading labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
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
    "D0_static",
    "D1_free_node",
    "D2_struct_free",
    "D3_struct_residual",
)
PAIRS = (
    ("D1_free_node", "D0_static"),
    ("D2_struct_free", "D1_free_node"),
    ("D2_struct_free", "D0_static"),
    ("D3_struct_residual", "D2_struct_free"),
    ("D3_struct_residual", "D0_static"),
)
EXPERT_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PAIR_TO_ID = {pair: index for index, pair in enumerate(EXPERT_PAIRS)}
PAIR_NAMES = tuple(f"{a}{b}" for a, b in EXPERT_PAIRS)
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v22b_strength_decoupling"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v22b_strength_decoupling"
DATA_ROOT = RESEARCH_ROOT / "data"
V22_RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v22_structure_grounded_router"
V22_DATA_ROOT = V22_RESEARCH_ROOT / "data"


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
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stats(values: torch.Tensor) -> dict[str, float | None]:
    values = values.detach().reshape(-1).float()
    if values.numel() == 0:
        return {name: None for name in ("mean", "std", "p10", "p50", "p90")}
    qs = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p10": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p90": float(qs[2].item()),
    }


def active_rms(value: torch.Tensor, active: torch.Tensor) -> float | None:
    if not bool(active.any()):
        return None
    return float(value[active].float().square().mean().sqrt().item())


def js_per_node(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    p, q = p.float().clamp_min(eps), q.float().clamp_min(eps)
    midpoint = 0.5 * (p + q)
    value = 0.5 * (p * (p.log() - midpoint.log())).sum(dim=-1) + 0.5 * (
        q * (q.log() - midpoint.log())
    ).sum(dim=-1)
    return value.clamp_min(0.0)


def cosine_record(a: torch.Tensor, b: torch.Tensor, active: torch.Tensor):
    if not bool(active.any()):
        return {"flattened_cosine": None, "mean_node_cosine": None}
    aa, bb = a[active].float(), b[active].float()
    return {
        "flattened_cosine": float(
            F.cosine_similarity(aa.reshape(-1), bb.reshape(-1), dim=0, eps=1.0e-8).item()
        ),
        "mean_node_cosine": float(
            F.cosine_similarity(aa, bb, dim=-1, eps=1.0e-8).mean().item()
        ),
    }


def pair_summary(route: torch.Tensor, active: torch.Tensor) -> dict[str, Any]:
    if not bool(active.any()):
        return {
            **{f"pair_count_{name}": 0 for name in PAIR_NAMES},
            **{f"pair_fraction_{name}": None for name in PAIR_NAMES},
            "num_unique_pairs": 0,
            "dominant_pair": None,
            "dominant_pair_fraction": None,
            "pair_entropy": None,
            "normalized_pair_entropy": None,
        }
    top = torch.topk(route[active], k=2, dim=-1).indices.sort(dim=-1).values
    pair_id = torch.empty(top.size(0), dtype=torch.long, device=top.device)
    for pair, index in PAIR_TO_ID.items():
        pair_id[(top[:, 0] == pair[0]) & (top[:, 1] == pair[1])] = index
    counts = torch.bincount(pair_id, minlength=6)
    fractions = counts.float() / counts.sum()
    nonzero = fractions > 0
    entropy = (-(fractions[nonzero] * fractions[nonzero].log()).sum()).clamp_min(0.0)
    dominant = int(torch.argmax(counts).item())
    return {
        **{f"pair_count_{name}": int(counts[i].item()) for i, name in enumerate(PAIR_NAMES)},
        **{f"pair_fraction_{name}": float(fractions[i].item()) for i, name in enumerate(PAIR_NAMES)},
        "num_unique_pairs": int(nonzero.sum().item()),
        "dominant_pair": PAIR_NAMES[dominant],
        "dominant_pair_fraction": float(fractions[dominant].item()),
        "pair_entropy": float(entropy.item()),
        "normalized_pair_entropy": float((entropy / math.log(6)).item()),
    }


def pair_change_fraction(left_logits: torch.Tensor, right_logits: torch.Tensor, active: torch.Tensor):
    if not bool(active.any()):
        return None
    left = torch.topk(left_logits[active], k=2, dim=-1).indices.sort(dim=-1).values
    right = torch.topk(right_logits[active], k=2, dim=-1).indices.sort(dim=-1).values
    return float((left != right).any(dim=-1).float().mean().item())


@torch.no_grad()
def audit_checkpoint(
    row: dict[str, Any],
    *,
    root: Path = ROOT,
    output_root: Path = OUTPUT_ROOT,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Audit a validation-selected checkpoint using features, edges, weights and metadata only."""
    from scripts.analyze_sosb_mag_v15_basis_screen import load_features_and_edges
    from src.models.mvcge_mag_v22b import Model

    dataset, seed, variant = row["dataset"], int(row["seed"]), row["variant"]
    mode = str(row.get("mode", "formal"))
    run_dir = output_root / mode / "runs" / dataset / f"seed_{seed}" / variant
    checkpoint = torch.load(root / row["checkpoint_path"], map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    cfg.model.variant = variant
    model = Model(cfg, checkpoint["data_info"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(device_name)
    model = model.to(device).eval()
    x_cpu, edge_cpu = load_features_and_edges(dataset, seed)
    x, edge = x_cpu.to(model.direct_strength_raw.device), edge_cpu.to(model.direct_strength_raw.device)
    z, _, _, aux, info = model(x, edge, return_details=True)
    if not torch.isfinite(z).all() or not torch.isfinite(aux):
        raise FloatingPointError(f"Non-finite output at {dataset}/{seed}/{variant}")

    active = info["active_nodes"]
    legacy_frozen = all(not p.requires_grad for p in model.strength_head.parameters())
    compat_frozen = all(
        not p.requires_grad
        for p in [model.expert_key_embedding, *model.expert_query_proj.parameters(), *model.expert_key_proj.parameters(), model.compatibility_residual_raw]
    )
    result: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "finite": True,
        "test_metrics_present": any(str(key).lower().startswith("test") for key in row["metrics"]),
        "active_node_count": int(active.sum().item()),
        "input_self_loops_removed": int(info["input_self_loops_removed"]),
        "parameter_count_model": int(row["metadata"]["model_parameters"]),
        "parameter_count_classifier": int(row["metadata"]["classifier_parameters"]),
        "alpha": info["details"]["text"]["alpha"].detach().cpu().tolist(),
        "modalities": {},
        "pair_diagnostics": {},
        "alpha_pairwise": [],
        "expert_function_pairs": [],
        "evidence_finite": True,
        "reliability_in_range": True,
        "legacy_strength_head_frozen": legacy_frozen,
        "compatibility_modules_frozen": compat_frozen,
        "direct_strength_finite": bool(torch.isfinite(model.direct_strength_raw).all()),
        "eta": {},
        "evidence_diagnostics": {},
        "static_routing_verified": True,
        "free_structure_logits_verified": True,
        "residual_logits_verified": True,
    }
    if not legacy_frozen or not compat_frozen:
        raise AssertionError("Legacy strength/compatibility parameters were not frozen")
    if not result["direct_strength_finite"]:
        raise FloatingPointError("Non-finite direct strength parameters")

    balances = []
    for modality_index, modality in enumerate(("text", "visual")):
        item = info["details"][modality]
        route = info["router"][modality]
        for key in (
            "prior", "basis", "alpha", "expert_inputs", "expert_outputs", "mixture",
            "scaled_mixture", "output", "selection_logits", "route_weights", "strength",
            "dense_probs", "scalar_evidence", "L_rel", "EVID", "reliability", "mu",
            "sigma", "degree_norm", "hop_cos", "hop_disp",
        ):
            if not torch.isfinite(item[key]).all():
                result["evidence_finite"] = False
                raise FloatingPointError(f"Non-finite {key}: {dataset}/{seed}/{variant}/{modality}")
        for key in ("selection_logits", "static_logits", "structure_free_logits", "eta", "strength", "strength_logit", "strength_raw"):
            if not torch.isfinite(route[key]).all():
                raise FloatingPointError(f"Non-finite router {key}: {dataset}/{seed}/{variant}/{modality}")
        if "compatibility" in route or "compatibility_logits" in route:
            raise AssertionError("Compatibility logits must not be part of V2.2b model behavior")

        # Verify the variant's only intended selection-path difference.
        if variant == "D0_static":
            static = route["selection_logits"][:1].expand_as(route["selection_logits"])
            result["static_routing_verified"] &= torch.equal(route["selection_logits"], static)
        elif variant == "D2_struct_free":
            result["free_structure_logits_verified"] &= torch.equal(route["selection_logits"], route["structure_free_logits"])
        elif variant == "D3_struct_residual":
            expected = route["static_logits"] + route["eta"] * (route["structure_free_logits"] - route["static_logits"])
            result["residual_logits_verified"] &= torch.equal(route["selection_logits"], expected)
        if variant == "D0_static" and not result["static_routing_verified"]:
            raise AssertionError("D0 static selection logits vary by node")
        if variant == "D2_struct_free" and not result["free_structure_logits_verified"]:
            raise AssertionError("D2 selection logits are not structure-grounded free logits")
        if variant == "D3_struct_residual" and not result["residual_logits_verified"]:
            raise AssertionError("D3 selection logits do not match static-centered eta interpolation")

        reliability = item["reliability"]
        if reliability.numel() and not bool(((reliability >= 0) & (reliability <= 1)).all()):
            result["reliability_in_range"] = False
            raise AssertionError("Edge reliability escaped [0,1]")
        if bool((~active).any()):
            if torch.count_nonzero(item["basis"][:, ~active]) or not torch.equal(item["output"][~active], item["prior"][~active]):
                raise AssertionError(f"Isolated-node output invariant failed: {dataset}/{variant}")
            if torch.count_nonzero(item["expert_inputs"][:, ~active]) or torch.count_nonzero(item["expert_outputs"][:, ~active]) or torch.count_nonzero(item["scaled_mixture"][~active]):
                raise AssertionError("Isolated nodes received structural expert corrections")
            for key in ("mu", "sigma", "degree_norm", "hop_cos", "hop_disp", "scalar_evidence", "L_rel", "EVID"):
                if torch.count_nonzero(item[key][~active]):
                    raise AssertionError(f"Isolated-node evidence {key} is nonzero")

        if bool(active.any()):
            selected_count = route["selected_mask"][active].sum(dim=-1)
            if not torch.equal(selected_count, torch.full_like(selected_count, 2)):
                raise AssertionError("Top-2 selection count changed")
            if not torch.allclose(route["route_weights"][active].sum(-1), torch.ones_like(selected_count, dtype=torch.float32), atol=1e-6):
                raise AssertionError("Top-2 weights do not sum to one")
            importance = route["dense_probs"][active].mean(dim=0)
            share = route["selected_mask"][active].float().mean(dim=0) / 2
            balance = model.num_experts * (importance * share).sum()
        else:
            importance = item["prior"].new_zeros((4,))
            share = importance
            balance = item["prior"].new_zeros(())
        if not torch.allclose(route["importance"], importance, atol=1e-7) or not torch.allclose(route["selection_share"], share, atol=1e-7):
            raise AssertionError("Active-only final-logit load balance changed")
        balances.append(balance)

        route_weights = route["route_weights"]
        mean_route = route_weights[active].mean(dim=0, keepdim=True) if bool(active.any()) else route_weights.new_zeros((1, 4))
        static_route = route["static_route_weights"]
        route_to_mean = stats(js_per_node(route_weights[active], mean_route)) if bool(active.any()) else stats(route_weights.new_empty((0,)))
        route_to_static = stats(js_per_node(route_weights[active], static_route[active])) if bool(active.any()) else stats(route_weights.new_empty((0,)))
        pair = pair_summary(route_weights, active)
        pair["route_to_modality_mean_js"] = route_to_mean
        pair["route_to_static_js"] = route_to_static
        pair["free_to_final_pair_change_fraction"] = pair_change_fraction(route["structure_free_logits"], route["selection_logits"], active) if variant == "D3_struct_residual" else None
        pair["static_to_final_pair_change_fraction"] = pair_change_fraction(route["static_logits"], route["selection_logits"], active) if variant == "D3_struct_residual" else None
        result["pair_diagnostics"][modality] = pair

        strength = route["strength"]
        strength_stats = stats(strength[active]) if bool(active.any()) else stats(strength.new_empty((0,)))
        node_strength_std = float((strength - strength[:1]).abs().max().item()) if strength.numel() else 0.0
        if node_strength_std != 0.0:
            raise AssertionError(f"Strength varies across nodes: {dataset}/{seed}/{variant}/{modality}")
        raw_strength = float(route["strength_raw"].item())
        effective_strength = float(strength[0].item()) if strength.numel() else float(torch.sigmoid(model.direct_strength_raw[modality_index]).item())
        expected_strength = float(torch.sigmoid(model.direct_strength_raw[modality_index]).item())
        if not math.isclose(effective_strength, expected_strength, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError("Effective strength differs from direct modality scalar")
        prior_rms = active_rms(item["prior"], active)
        mixture_rms = active_rms(item["mixture"], active)
        correction_rms = active_rms(item["scaled_mixture"], active)
        result["modalities"][modality] = {
            "selection_share": share.detach().cpu().tolist(),
            "dense_importance": importance.detach().cpu().tolist(),
            "num_experts_positive_share": int((share > 0).sum().item()),
            "max_min_load_ratio": float(share[share > 0].max() / share[share > 0].min()) if bool((share > 0).any()) else None,
            "routing_entropy": stats(-(route["dense_probs"][active].clamp_min(1e-12) * route["dense_probs"][active].clamp_min(1e-12).log()).sum(-1)) if bool(active.any()) else stats(route["dense_probs"].new_empty((0,))),
            "top1_top2_logit_margin": stats(route["selection_logits"][active].topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)) if bool(active.any()) else stats(route["selection_logits"].new_empty((0,))),
            "strength": strength_stats,
            "direct_strength_raw": raw_strength,
            "effective_strength": effective_strength,
            "node_strength_std": node_strength_std,
            "legacy_strength_head_frozen": legacy_frozen,
            "strength_static": node_strength_std == 0.0,
            "strength_fraction_gt_0_9": float((strength[active] > 0.9).float().mean()) if bool(active.any()) else None,
            "prior_rms": prior_rms,
            "unscaled_expert_mixture_rms": mixture_rms,
            "scaled_moe_correction_rms": correction_rms,
            "scaled_correction_to_prior_rms_ratio": correction_rms / prior_rms if correction_rms is not None and prior_rms else None,
            "eta": float(route["eta"].item()),
            "route_to_static_js": route_to_static,
            "structure_free_to_static_js": stats(js_per_node(route["structure_free_route_weights"][active], static_route[active])) if bool(active.any()) else stats(route_weights.new_empty((0,))),
        }
        result["eta"][modality] = float(route["eta"].item()) if variant == "D3_struct_residual" else None

        if variant in {"D2_struct_free", "D3_struct_residual"}:
            result["evidence_diagnostics"][modality] = {
                "reliability_mu": stats(item["mu"][active]),
                "reliability_sigma": stats(item["sigma"][active]),
                "degree_norm": stats(item["degree_norm"][active]),
                "hop_cos": [stats(item["hop_cos"][active, k]) for k in range(4)],
                "hop_disp": [stats(item["hop_disp"][active, k]) for k in range(4)],
            }
        else:
            result["evidence_diagnostics"][modality] = None

    expected_aux = model.balance_weight * 0.5 * (balances[0] + balances[1])
    if not torch.allclose(aux, expected_aux, atol=1e-7, rtol=1e-6):
        raise AssertionError("Returned auxiliary loss differs from the V2.2 active-only balance formula")
    text_share = result["modalities"]["text"]["selection_share"]
    visual_share = result["modalities"]["visual"]["selection_share"]
    used_union = [i for i in range(4) if text_share[i] > 0 or visual_share[i] > 0]
    dead_both = [i for i in range(4) if text_share[i] == 0 and visual_share[i] == 0]
    result["num_experts_used_union"] = len(used_union)
    result["num_experts_dead_both_modalities"] = len(dead_both)
    result["expert_union_used_indices"] = used_union
    result["expert_dead_both_indices"] = dead_both

    alpha = info["details"]["text"]["alpha"].detach()
    for first, second in EXPERT_PAIRS:
        result["alpha_pairwise"].append({"expert_a": first, "expert_b": second, "cosine": float(F.cosine_similarity(alpha[first], alpha[second], dim=0).item())})
    for modality in ("text", "visual"):
        expert_values = info["details"][modality]["expert_outputs"]
        for first, second in EXPERT_PAIRS:
            result["expert_function_pairs"].append({"modality": modality, "expert_a": first, "expert_b": second, **cosine_record(expert_values[first], expert_values[second], active)})
    flat_cos = [r["flattened_cosine"] for r in result["expert_function_pairs"] if r["flattened_cosine"] is not None]
    node_cos = [r["mean_node_cosine"] for r in result["expert_function_pairs"] if r["mean_node_cosine"] is not None]
    result["expert_similarity_checkpoint_summary"] = {
        "flattened_mean": statistics.fmean(flat_cos) if flat_cos else None,
        "flattened_min": min(flat_cos) if flat_cos else None,
        "flattened_max": max(flat_cos) if flat_cos else None,
        "mean_node_mean": statistics.fmean(node_cos) if node_cos else None,
        "mean_node_min": min(node_cos) if node_cos else None,
        "mean_node_max": max(node_cos) if node_cos else None,
    }
    result["router_evidence_finite"] = result["evidence_finite"] and result["reliability_in_range"]
    result["strength_static"] = all(values["strength_static"] for values in result["modalities"].values())
    if device.type == "cuda":
        torch.cuda.empty_cache()
    del model, x_cpu, edge_cpu, x, edge, z, info
    return result

def campaign_summaries(rows: list[dict[str, Any]]):
    by_key = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in rows}
    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            if not group:
                continue
            models = {int(r["metadata"]["model_parameters"]) for r in group}
            heads = {int(r["metadata"]["classifier_parameters"]) for r in group}
            if len(models) != 1 or len(heads) != 1:
                raise ValueError(f"Parameter count changed within {dataset}/{variant}")
            acc = [float(r["metrics"]["val_acc"]) for r in group]
            f1 = [float(r["metrics"]["val_macro_f1"]) for r in group]
            epochs = [int(r["metadata"]["best_epoch"]) for r in group]
            summary.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "num_runs": len(group),
                    "val_acc_mean": statistics.fmean(acc),
                    "val_acc_std": statistics.pstdev(acc),
                    "val_macro_f1_mean": statistics.fmean(f1),
                    "val_macro_f1_std": statistics.pstdev(f1),
                    "mean_best_epoch": statistics.fmean(epochs),
                    "model_trainable_params": models.pop(),
                    "classifier_params": heads.pop(),
                    "total_trainable_params": int(group[0]["metadata"]["model_parameters"]) + int(group[0]["metadata"]["classifier_parameters"]),
                }
            )
    paired = []
    for candidate, reference in PAIRS:
        pairs = [
            (ds, seed, by_key[(ds, seed, candidate)], by_key[(ds, seed, reference)])
            for ds in DATASETS
            for seed in SEEDS
            if (ds, seed, candidate) in by_key and (ds, seed, reference) in by_key
        ]
        if not pairs:
            continue
        da = [float(a["metrics"]["val_acc"]) - float(b["metrics"]["val_acc"]) for _, _, a, b in pairs]
        df = [float(a["metrics"]["val_macro_f1"]) - float(b["metrics"]["val_macro_f1"]) for _, _, a, b in pairs]
        item: dict[str, Any] = {
            "comparison": f"{candidate} - {reference}",
            "overall_delta_accuracy_pp": 100 * statistics.fmean(da),
            "overall_delta_macro_f1_pp": 100 * statistics.fmean(df),
            "positive_accuracy_pairs": f"{sum(v > 0 for v in da)}/{len(da)}",
            "positive_macro_f1_pairs": f"{sum(v > 0 for v in df)}/{len(df)}",
            "n_paired_runs": len(pairs),
        }
        for dataset in DATASETS:
            subset = [(a, b) for ds, _, a, b in pairs if ds == dataset]
            dsa = [float(a["metrics"]["val_acc"]) - float(b["metrics"]["val_acc"]) for a, b in subset]
            dsf = [float(a["metrics"]["val_macro_f1"]) - float(b["metrics"]["val_macro_f1"]) for a, b in subset]
            item[f"{dataset}_delta_accuracy_pp_mean"] = 100 * statistics.fmean(dsa) if dsa else None
            item[f"{dataset}_delta_macro_f1_pp_mean"] = 100 * statistics.fmean(dsf) if dsf else None
            item[f"{dataset}_positive_accuracy_seeds"] = f"{sum(v > 0 for v in dsa)}/{len(dsa)}" if dsa else None
            item[f"{dataset}_positive_macro_f1_seeds"] = f"{sum(v > 0 for v in dsf)}/{len(dsf)}" if dsf else None
        paired.append(item)
    return summary, paired


def diagnostic_tables(audits: list[dict[str, Any]]):
    routing_rows, pair_rows, evidence_rows = [], [], []
    profile_rows, similarity_rows, strength_rows = [], [], []
    for audit in audits:
        dataset, seed, variant = audit["dataset"], audit["seed"], audit["variant"]
        for expert, coefficients in enumerate(audit["alpha"]):
            profile_rows.append({
                "dataset": dataset, "seed": seed, "variant": variant, "expert": expert,
                **{f"alpha_{k + 1}": float(value) for k, value in enumerate(coefficients)},
                "row_norm": math.sqrt(sum(float(v) ** 2 for v in coefficients)),
            })
        for pair in audit["alpha_pairwise"]:
            similarity_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "kind": "alpha_profile", "modality": "shared", **pair})
        for pair in audit["expert_function_pairs"]:
            similarity_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "kind": "expert_output", **pair})
        similarity_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "kind": "expert_output_checkpoint_summary", "modality": "both", **audit["expert_similarity_checkpoint_summary"]})

        for modality, values in audit["modalities"].items():
            route_row = {
                "dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                "num_experts_positive_share": values["num_experts_positive_share"],
                "max_min_positive_load_ratio": values["max_min_load_ratio"],
                "num_experts_used_union": audit["num_experts_used_union"],
                "num_experts_dead_both_modalities": audit["num_experts_dead_both_modalities"],
                "expert_union_used_indices": ";".join(map(str, audit["expert_union_used_indices"])),
                "expert_dead_both_indices": ";".join(map(str, audit["expert_dead_both_indices"])),
            }
            for expert in range(4):
                route_row[f"selection_share_expert_{expert}"] = values["selection_share"][expert]
                route_row[f"dense_importance_expert_{expert}"] = values["dense_importance"][expert]
            for prefix, metric in (
                ("route_to_modality_mean_js", audit["pair_diagnostics"][modality]["route_to_modality_mean_js"]),
                ("route_to_static_js", values["route_to_static_js"]),
                ("routing_entropy", values["routing_entropy"]),
                ("top1_top2_logit_margin", values["top1_top2_logit_margin"]),
            ):
                for stat_name, stat_value in metric.items():
                    route_row[f"{prefix}_{stat_name}"] = stat_value
            routing_rows.append(route_row)
            pair_rows.append({"dataset": dataset, "seed": seed, "variant": variant, "modality": modality, **audit["pair_diagnostics"][modality]})
            strength_rows.append({
                "dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
                "direct_strength_raw": values["direct_strength_raw"],
                "effective_strength": values["effective_strength"],
                "node_strength_std": values["node_strength_std"],
                "prior_rms": values["prior_rms"],
                "unscaled_expert_mixture_rms": values["unscaled_expert_mixture_rms"],
                "scaled_correction_rms": values["scaled_moe_correction_rms"],
                "scaled_correction_to_prior_rms_ratio": values["scaled_correction_to_prior_rms_ratio"],
                "legacy_strength_head_frozen": values["legacy_strength_head_frozen"],
            })
            evidence = audit["evidence_diagnostics"][modality]
            if evidence is not None:
                evidence_row = {"dataset": dataset, "seed": seed, "variant": variant, "modality": modality}
                for prefix in ("reliability_mu", "reliability_sigma", "degree_norm"):
                    for stat_name, stat_value in evidence[prefix].items():
                        evidence_row[f"{prefix}_{stat_name}"] = stat_value
                for order in range(4):
                    for prefix in ("hop_cos", "hop_disp"):
                        for stat_name, stat_value in evidence[prefix][order].items():
                            evidence_row[f"{prefix}_{order + 1}_{stat_name}"] = stat_value
                evidence_rows.append(evidence_row)
    return routing_rows, pair_rows, evidence_rows, profile_rows, similarity_rows, strength_rows

def _comparison_map(paired):
    return {item["comparison"]: item for item in paired}


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _stable_positive(item):
    if not item or item["n_paired_runs"] != 9:
        return False
    dataset_means = [item[f"{ds}_delta_accuracy_pp_mean"] for ds in DATASETS]
    return (
        item["overall_delta_accuracy_pp"] > 0
        and item["overall_delta_macro_f1_pp"] > 0
        and int(item["positive_accuracy_pairs"].split("/")[0]) >= 6
        and sum(value is not None and value > 0 for value in dataset_means) >= 2
    )


def make_repair_comparison(rows, audits):
    old_runs_path = V22_DATA_ROOT / "run_rows.json"
    old_strength_path = V22_DATA_ROOT / "strength_diagnostics.csv"
    old_runs = json.loads(old_runs_path.read_text(encoding="utf-8"))
    old_run_map = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in old_runs if r["variant"] in {"R0_modality_static", "R1_free_node"}}
    old_strength = {}
    with old_strength_path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["variant"] in {"R0_modality_static", "R1_free_node"}:
                old_strength[(r["dataset"], int(r["seed"]), r["variant"], r["modality"])] = float(r["strength_mean"])
    new_run_map = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in rows if r["variant"] in {"D0_static", "D1_free_node"}}
    new_audit_map = {(a["dataset"], int(a["seed"]), a["variant"]): a for a in audits if a["variant"] in {"D0_static", "D1_free_node"}}
    result = []
    for dataset in DATASETS:
        for seed in SEEDS:
            key = (dataset, seed)
            old_r0 = old_run_map[(*key, "R0_modality_static")]
            old_r1 = old_run_map[(*key, "R1_free_node")]
            new_d0 = new_run_map[(*key, "D0_static")]
            new_d1 = new_run_map[(*key, "D1_free_node")]
            old_acc = float(old_r1["metrics"]["val_acc"]) - float(old_r0["metrics"]["val_acc"])
            old_f1 = float(old_r1["metrics"]["val_macro_f1"]) - float(old_r0["metrics"]["val_macro_f1"])
            new_acc = float(new_d1["metrics"]["val_acc"]) - float(new_d0["metrics"]["val_acc"])
            new_f1 = float(new_d1["metrics"]["val_macro_f1"]) - float(new_d0["metrics"]["val_macro_f1"])
            row = {
                "dataset": dataset,
                "seed": seed,
                "old_R1_minus_R0_acc": old_acc,
                "old_R1_minus_R0_f1": old_f1,
                "new_D1_minus_D0_acc": new_acc,
                "new_D1_minus_D0_f1": new_f1,
                "contrast_change_acc": new_acc - old_acc,
                "contrast_change_f1": new_f1 - old_f1,
            }
            for old_name, new_name in (("R0_modality_static", "D0_static"), ("R1_free_node", "D1_free_node")):
                old_short = "R0" if old_name.startswith("R0") else "R1"
                new_short = "D0" if new_name.startswith("D0") else "D1"
                old_values = [old_strength[(*key, old_name, modality)] for modality in ("text", "visual")]
                new_audit = new_audit_map[(*key, new_name)]
                for modality, old_value in zip(("text", "visual"), old_values):
                    row[f"old_{old_short}_strength_{modality}"] = old_value
                    row[f"new_{new_short}_strength_{modality}"] = new_audit["modalities"][modality]["effective_strength"]
            result.append(row)
    return result


def _approximately_equal(item):
    return abs(item["overall_delta_accuracy_pp"]) <= 0.15 and abs(item["overall_delta_macro_f1_pp"]) <= 0.50


def _delta_positive(item):
    return item["overall_delta_accuracy_pp"] > 0 and item["overall_delta_macro_f1_pp"] > 0


def make_report(rows, summary, paired, audits, manifest, repair_rows) -> str:
    pairs = _comparison_map(paired)
    labels = (
        ("D1_free_node - D0_static", "D1 − D0: free node selection after strength decoupling"),
        ("D2_struct_free - D1_free_node", "D2 − D1: structural evidence added to free routing"),
        ("D3_struct_residual - D2_struct_free", "D3 − D2: static-centered residualization"),
        ("D2_struct_free - D0_static", "D2 − D0: structure-grounded free routing vs static selection"),
        ("D3_struct_residual - D0_static", "D3 − D0: structure-grounded residual routing vs static selection"),
    )
    summary_table = "| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Trainable params (model + head) |\n|---|---|---:|---:|---:|---:|\n"
    for item in summary:
        summary_table += f"| {item['dataset']} | {item['variant']} | {100*item['val_acc_mean']:.2f}% ± {100*item['val_acc_std']:.2f}% | {100*item['val_macro_f1_mean']:.2f}% ± {100*item['val_macro_f1_std']:.2f}% | {item['mean_best_epoch']:.1f} | {item['model_trainable_params']:,} + {item['classifier_params']:,} |\n"
    paired_table = "| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Acc / F1) |\n|---|---:|---:|---:|\n"
    for key, title in labels:
        item = pairs[key]
        paired_table += f"| {title} | {item['overall_delta_accuracy_pp']:+.2f} pp | {item['overall_delta_macro_f1_pp']:+.2f} pp | {item['positive_accuracy_pairs']} / {item['positive_macro_f1_pairs']} |\n"
    dataset_table = "| Comparison | Dataset | Δ Accuracy mean (positive seeds) | Δ Macro-F1 mean (positive seeds) |\n|---|---|---:|---:|\n"
    for key, title in labels:
        item = pairs[key]
        for dataset in DATASETS:
            dataset_table += f"| {title} | {dataset} | {item[f'{dataset}_delta_accuracy_pp_mean']:+.2f} pp ({item[f'{dataset}_positive_accuracy_seeds']}) | {item[f'{dataset}_delta_macro_f1_pp_mean']:+.2f} pp ({item[f'{dataset}_positive_macro_f1_seeds']}) |\n"

    by_variant = {v: [a for a in audits if a["variant"] == v] for v in VARIANTS}
    mean_metric = lambda variant, fn: _mean(fn(a, modality) for a in by_variant[variant] for modality in ("text", "visual"))
    pair_entropy = {v: mean_metric(v, lambda a, m: a["pair_diagnostics"][m]["normalized_pair_entropy"]) for v in VARIANTS}
    route_mean_js = {v: mean_metric(v, lambda a, m: a["pair_diagnostics"][m]["route_to_modality_mean_js"]["mean"]) for v in VARIANTS}
    route_static_js = {v: mean_metric(v, lambda a, m: a["modalities"][m]["route_to_static_js"]["mean"]) for v in VARIANTS}
    strength_means = {v: {m: _mean(a["modalities"][m]["effective_strength"] for a in by_variant[v]) for m in ("text", "visual")} for v in VARIANTS}
    dead_slot_total = {v: sum(a["num_experts_dead_both_modalities"] for a in by_variant[v]) for v in VARIANTS}
    dead_run_count = {v: sum(a["num_experts_dead_both_modalities"] > 0 for a in by_variant[v]) for v in VARIANTS}
    dead_dist = {v: dict(sorted(Counter(a["num_experts_dead_both_modalities"] for a in by_variant[v]).items())) for v in VARIANTS}
    used_union = {v: _mean(a["num_experts_used_union"] for a in by_variant[v]) for v in VARIANTS}
    evidence_audits = [a for a in audits if a["variant"] in {"D2_struct_free", "D3_struct_residual"}]
    evidence_mu = _mean(a["evidence_diagnostics"][m]["reliability_mu"]["mean"] for a in evidence_audits for m in ("text", "visual"))
    evidence_sigma = _mean(a["evidence_diagnostics"][m]["reliability_sigma"]["mean"] for a in evidence_audits for m in ("text", "visual"))
    evidence_degree = _mean(a["evidence_diagnostics"][m]["degree_norm"]["std"] for a in evidence_audits for m in ("text", "visual"))
    hop_cos_mean = _mean(a["evidence_diagnostics"][m]["hop_cos"][k]["std"] for a in evidence_audits for m in ("text", "visual") for k in range(4))
    hop_disp_mean = _mean(a["evidence_diagnostics"][m]["hop_disp"][k]["std"] for a in evidence_audits for m in ("text", "visual") for k in range(4))
    eta_means = {m: _mean(a["modalities"][m]["eta"] for a in by_variant["D3_struct_residual"]) for m in ("text", "visual")}
    free_pair_change = {m: _mean(a["pair_diagnostics"][m]["free_to_final_pair_change_fraction"] for a in by_variant["D3_struct_residual"]) for m in ("text", "visual")}
    static_pair_change = {m: _mean(a["pair_diagnostics"][m]["static_to_final_pair_change_fraction"] for a in by_variant["D3_struct_residual"]) for m in ("text", "visual")}
    alpha_cos = [p["cosine"] for a in audits for p in a["alpha_pairwise"]]
    functional_flat = [p["flattened_cosine"] for a in audits for p in a["expert_function_pairs"] if p["flattened_cosine"] is not None]
    functional_node = [p["mean_node_cosine"] for a in audits for p in a["expert_function_pairs"] if p["mean_node_cosine"] is not None]
    d1_d0 = pairs["D1_free_node - D0_static"]
    d2_d1 = pairs["D2_struct_free - D1_free_node"]
    d2_d0 = pairs["D2_struct_free - D0_static"]
    d3_d2 = pairs["D3_struct_residual - D2_struct_free"]
    d3_d0 = pairs["D3_struct_residual - D0_static"]
    a_observed = _stable_positive(d1_d0)
    b_observed = _approximately_equal(d1_d0) or d1_d0["overall_delta_accuracy_pp"] <= 0 or d1_d0["overall_delta_macro_f1_pp"] <= 0
    c_observed = _delta_positive(d2_d1)
    d_observed = _approximately_equal(d2_d1)
    e_observed = d3_d2["overall_delta_accuracy_pp"] < 0 and d3_d2["overall_delta_macro_f1_pp"] < 0
    f_observed = _delta_positive(d3_d2)
    g_observed = not (_stable_positive(d1_d0) or _stable_positive(d2_d0))
    meanings = [
        ("A", a_observed, "D1−D0 meets the frozen stable-positive rule after strength is decoupled; node-specific expert selection retains a stable positive signal."),
        ("B", b_observed, "D1−D0 is approximately equal under the frozen descriptive band or at least one headline delta is non-positive; the old node-routing signal is not consistently retained after decoupling."),
        ("C", c_observed, "D2−D1 is positive on both headline metrics; structure-grounded evidence adds descriptive value to the free-router parameterization."),
        ("D", d_observed, "D2−D1 falls within the frozen approximate-equality band; current structural observation provides no distinguishable task value in this screen."),
        ("E", e_observed, "D3 is below D2 on both headline metrics; static-centered residualization is not supported under hard Top-2 routing."),
        ("F", f_observed, f"D3 is positive over D2 on both headline metrics; interpret this alongside mean eta Text/Visual {eta_means['text']:.4f}/{eta_means['visual']:.4f} and pair-change fractions (free→final {free_pair_change['text']:.4f}/{free_pair_change['visual']:.4f}; static→final {static_pair_change['text']:.4f}/{static_pair_change['visual']:.4f})."),
        ("G", g_observed, "Neither D1 nor D2 passes the stable-positive rule versus D0; conclude router-centric optimization and move the next research question to modality-conditioned effective structural contexts for the shared expert action space."),
    ]
    interpretation = "\n".join(f"- **{letter}. {'Observed' if observed else 'Not observed'}.** {description}" for letter, observed, description in meanings)
    routing_table = "| Variant | Normalized Top-2 pair entropy | Route-to-mean JS | Route-to-static JS | Mean experts in Text ∪ Visual | Dead slots | Runs with ≥1 dead slot |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for variant in VARIANTS:
        routing_table += f"| {variant} | {pair_entropy[variant]:.4f} | {route_mean_js[variant]:.6f} | {route_static_js[variant]:.6f} | {used_union[variant]:.3f}/4 | {dead_slot_total[variant]}/9 | {dead_run_count[variant]}/9 |\n"
    strength_table = "| Variant | Direct strength Text | Direct strength Visual | Max node-strength std |\n|---|---:|---:|---:|\n"
    for variant in VARIANTS:
        max_std = max(a["modalities"][m]["node_strength_std"] for a in by_variant[variant] for m in ("text", "visual"))
        strength_table += f"| {variant} | {strength_means[variant]['text']:.6f} | {strength_means[variant]['visual']:.6f} | {max_std:.1e} |\n"
    repair_keys = ("old_R1_minus_R0_acc", "old_R1_minus_R0_f1", "new_D1_minus_D0_acc", "new_D1_minus_D0_f1", "contrast_change_acc", "contrast_change_f1")
    repair_means = {key: 100 * statistics.fmean(float(r[key]) for r in repair_rows) for key in repair_keys}
    repair_table = "| Dataset | Seed | Old R1−R0 Acc / F1 | New D1−D0 Acc / F1 | Contrast change Acc / F1 |\n|---|---:|---:|---:|---:|\n"
    for r in repair_rows:
        repair_table += f"| {r['dataset']} | {r['seed']} | {100*r['old_R1_minus_R0_acc']:+.3f} / {100*r['old_R1_minus_R0_f1']:+.3f} pp | {100*r['new_D1_minus_D0_acc']:+.3f} / {100*r['new_D1_minus_D0_f1']:+.3f} pp | {100*r['contrast_change_acc']:+.3f} / {100*r['contrast_change_f1']:+.3f} pp |\n"
    repair_strength_means = {}
    for variant in ("R0", "R1", "D0", "D1"):
        for modality in ("text", "visual"):
            repair_strength_means[f"{variant}_{modality}"] = statistics.fmean(float(r[f"old_{variant}_strength_{modality}"] if variant.startswith("R") else r[f"new_{variant}_strength_{modality}"]) for r in repair_rows)

    json_means = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        json_means[variant] = {"val_accuracy": statistics.fmean(float(r["metrics"]["val_acc"]) for r in subset), "val_macro_f1": statistics.fmean(float(r["metrics"]["val_macro_f1"]) for r in subset)}
    device = manifest.get("device", "unknown")
    fail_count = len([f for f in manifest.get("failures", []) if not f.get("resolved", False)])
    return f"""# MvCGE-MAG V2.2b: Strength-Decoupled Routing Repair

## Protocol and provenance

- Branch: `{manifest['provenance']['branch']}`; parent: `{manifest['provenance']['parent_commit_sha']}`; freeze commit: `{manifest['provenance']['freeze_commit_sha']}`.
- Validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; D0–D3; {len(rows)}/36 completed.
- Each training command set `task.evaluate_test=false`; Validation Accuracy selected checkpoints. Run metrics have no Test keys. Checkpoint audits used only features, graph edges, weights and selected metadata; no labels were read.
- No HPO, significance test, LP run or post-freeze model/config edit. Device `{device}`; unresolved training failures: {fail_count}.
- Freeze SHA: `{manifest['provenance']['freeze_commit_sha']}`; formal campaign HEAD matched this SHA and started with a clean worktree.

## Validation results

Run-level means and population standard deviations; paired deltas are descriptive percentage points across matched dataset-seed runs.

{summary_table}
Equally weighted run means:

```json
{json.dumps(json_means, indent=2)}
```

## Required paired comparisons

{paired_table}
Per-dataset mean deltas and positive seed counts:

{dataset_table}
No significance testing was performed.

## Strength decoupling and router diagnostics

{strength_table}

- For each checkpoint/modality, `node_strength_std` is computed as the maximum absolute deviation from the broadcast scalar and is exactly zero. The effective strength depends only on `direct_strength_raw[m]`; the legacy `strength_head` and compatibility modules were frozen.
- D3 mean eta Text/Visual is `{eta_means['text']:.6f}` / `{eta_means['visual']:.6f}`. D3 free→final hard-Top-2 pair-change fractions are `{free_pair_change['text']:.4f}` / `{free_pair_change['visual']:.4f}`; static→final are `{static_pair_change['text']:.4f}` / `{static_pair_change['visual']:.4f}`.
- Reliability mean/sigma, degree variation, and hop-cosine/displacement variation for D2/D3 are `{evidence_mu:.4f}`, `{evidence_sigma:.4f}`, `{evidence_degree:.4f}`, `{hop_cos_mean:.4f}`, `{hop_disp_mean:.4f}`. Per-order quantiles are in `data/evidence_diagnostics.csv`.
- Alpha profile mean pairwise cosine is `{statistics.fmean(alpha_cos):.4f}`; expert-output mean-node cosine is `{statistics.fmean(functional_node):.4f}`; flattened cosine is `{statistics.fmean(functional_flat):.4f}`.
- Routing entropy, route-to-modality-mean JS and route-to-static JS are summarized below. Dead slots count experts with zero selection in both modalities; affected run-checkpoints are counted separately.

{routing_table}

## Historical V2.2 strength-coupling comparison

The table uses only committed V2.2 R0/R1 run metrics and their per-modality strength diagnostics, paired with new D0/D1 results. The contrast change is a descriptive arithmetic comparison, not a causal attribution.

{repair_table}

Across nine matched dataset-seed runs, old R1−R0 means were Accuracy `{repair_means['old_R1_minus_R0_acc']:+.3f}` pp / Macro-F1 `{repair_means['old_R1_minus_R0_f1']:+.3f}` pp; new D1−D0 means were `{repair_means['new_D1_minus_D0_acc']:+.3f}` / `{repair_means['new_D1_minus_D0_f1']:+.3f}` pp. The arithmetic contrast change was `{repair_means['contrast_change_acc']:+.3f}` / `{repair_means['contrast_change_f1']:+.3f}` pp. Mean old/new strengths by modality are R0 `{repair_strength_means['R0_text']:.4f}/{repair_strength_means['R0_visual']:.4f}`, R1 `{repair_strength_means['R1_text']:.4f}/{repair_strength_means['R1_visual']:.4f}`, D0 `{repair_strength_means['D0_text']:.4f}/{repair_strength_means['D0_visual']:.4f}`, D1 `{repair_strength_means['D1_text']:.4f}/{repair_strength_means['D1_visual']:.4f}` (Text/Visual).

## Interpretation map

Interpretation thresholds were fixed in `README.md`: approximate equality means `|Δ Accuracy| <= 0.15 pp` and `|Δ Macro-F1| <= 0.50 pp`; stable-positive means both overall deltas positive, at least 6/9 positive Accuracy pairs, and positive Accuracy means on at least 2/3 datasets. These are descriptive rules, not inferential tests.

{interpretation}

## Post-screen decision gate

This screen ends here regardless of result; it does not automatically launch V3. If neither D1 nor D2 is stable-positive versus D0, the next research question is: **Can modality-conditioned effective structural contexts create a better shared expert action space for MAG?** If either passes, retain node routing as a future candidate only after validating that effective structural action space.

- No compatibility routing, expert keys, extra experts, changed Top-K, topology reweighting, cross-modal input, HPO, LP, Test evaluation, or significance testing were introduced.
- Paired outcomes and diagnostics are descriptive; they do not establish causal mechanisms.

## Reproducibility artifacts

- `data/environment.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/routing_diagnostics.csv`, `data/pair_diagnostics.csv`
- `data/evidence_diagnostics.csv`, `data/strength_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/repair_comparison.csv`
- Model checkpoints, training/Hydra logs, raw node tensors, embeddings and caches remain under ignored server-local `outputs/`.
"""

def analyze(device: str) -> None:
    rows = json.loads((DATA_ROOT / "run_rows.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA_ROOT / "campaign_manifest.json").read_text(encoding="utf-8"))
    if len(rows) != 36 or manifest.get("completed_runs") != 36 or manifest.get("task_evaluate_test") is not False:
        raise RuntimeError("Campaign record is incomplete or not validation-only")
    keys = {(row["dataset"], int(row["seed"]), row["variant"]) for row in rows}
    expected = {(ds, seed, variant) for ds in DATASETS for seed in SEEDS for variant in VARIANTS}
    if keys != expected:
        raise RuntimeError("Campaign run grid does not match the frozen 36-run design")
    audits = []
    for index, row in enumerate(rows, start=1):
        print(f"[MvCGE-MAG V2.2b audit] {index}/36 {row['dataset']} seed={row['seed']} {row['variant']}", flush=True)
        if row.get("status") not in {"completed", "reused"}:
            raise RuntimeError(f"Incomplete run row: {row}")
        if any(str(key).lower().startswith("test") for key in row["metrics"]):
            raise RuntimeError("Test metric key found in validation-only run record")
        audit = audit_checkpoint(row, root=ROOT, output_root=OUTPUT_ROOT, device_name=device)
        required = ("finite", "strength_static", "legacy_strength_head_frozen", "compatibility_modules_frozen", "direct_strength_finite", "router_evidence_finite")
        if audit["test_metrics_present"] or not all(audit[key] for key in required):
            raise RuntimeError(f"Selected-checkpoint audit failed: {audit['dataset']}/{audit['variant']}")
        if row["variant"] == "D0_static" and not audit["static_routing_verified"]:
            raise AssertionError("D0 static routing audit failed")
        if row["variant"] == "D2_struct_free" and not audit["free_structure_logits_verified"]:
            raise AssertionError("D2 free structural routing audit failed")
        if row["variant"] == "D3_struct_residual" and not audit["residual_logits_verified"]:
            raise AssertionError("D3 residual routing audit failed")
        audits.append(audit)
        del audit

    summary, paired = campaign_summaries(rows)
    routing, pair, evidence, profiles, similarities, strengths = diagnostic_tables(audits)
    repair = make_repair_comparison(rows, audits)
    write_csv(DATA_ROOT / "summary.csv", summary)
    write_csv(DATA_ROOT / "paired_comparisons.csv", paired)
    for name, table in (
        ("routing_diagnostics.csv", routing),
        ("pair_diagnostics.csv", pair),
        ("evidence_diagnostics.csv", evidence),
        ("expert_profiles.csv", profiles),
        ("expert_similarity.csv", similarities),
        ("strength_diagnostics.csv", strengths),
        ("repair_comparison.csv", repair),
    ):
        write_csv(DATA_ROOT / name, table)
    manifest["selected_checkpoint_audits"] = len(audits)
    manifest["test_evaluation_verified_false"] = True
    manifest["test_metrics_absent"] = True
    manifest["label_free_checkpoint_audit"] = True
    manifest["strength_decoupled_verified"] = True
    manifest["strength_static_verified"] = all(r["node_strength_std"] == 0.0 for r in strengths)
    manifest["legacy_strength_head_frozen_verified"] = all(r["legacy_strength_head_frozen"] for r in strengths)
    manifest["compatibility_modules_frozen_verified"] = all(a["compatibility_modules_frozen"] for a in audits)
    manifest["diagnostic_device"] = device
    write_json(DATA_ROOT / "campaign_manifest.json", manifest)
    (RESEARCH_ROOT / "REPORT.md").write_text(
        make_report(rows, summary, paired, audits, manifest, repair), encoding="utf-8"
    )
    print("[MvCGE-MAG V2.2b analysis] wrote the tracked tables, historical repair comparison and REPORT.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    analyze(args.device)
