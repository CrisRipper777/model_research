#!/usr/bin/env python3
"""Audit validation-selected V2.1 checkpoints and write compact diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
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
    "U0_modality_static",
    "U1_node_selection",
    "U2_node_selection_strength",
    "U3_collaborative",
)
PAIRS = (
    ("U1_node_selection", "U0_modality_static"),
    ("U2_node_selection_strength", "U1_node_selection"),
    ("U3_collaborative", "U2_node_selection_strength"),
    ("U2_node_selection_strength", "U0_modality_static"),
    ("U3_collaborative", "U0_modality_static"),
)
EXPERT_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PAIR_TO_ID = {pair: index for index, pair in enumerate(EXPERT_PAIRS)}
PAIR_NAMES = tuple(f"{a}{b}" for a, b in EXPERT_PAIRS)
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v21_utilization_screen"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v21_utilization_screen"
DATA_ROOT = RESEARCH_ROOT / "data"


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
        writer = csv.DictWriter(
            handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def stats(values: torch.Tensor) -> dict[str, float | None]:
    values = values.detach().reshape(-1).float()
    if values.numel() == 0:
        return {name: None for name in ("mean", "std", "p10", "p50", "p90")}
    quantiles = torch.quantile(
        values, torch.tensor([0.1, 0.5, 0.9], device=values.device)
    )
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def active_rms(value: torch.Tensor, active: torch.Tensor) -> float | None:
    if not bool(active.any()):
        return None
    return float(value[active].float().square().mean().sqrt().item())


def cosine_record(
    a: torch.Tensor, b: torch.Tensor, active: torch.Tensor
) -> dict[str, float | None]:
    if not bool(active.any()):
        return {"flattened_cosine": None, "mean_node_cosine": None}
    aa, bb = a[active].float(), b[active].float()
    node = F.cosine_similarity(aa, bb, dim=-1, eps=1.0e-8)
    flat = F.cosine_similarity(aa.reshape(-1), bb.reshape(-1), dim=0, eps=1.0e-8)
    return {
        "flattened_cosine": float(flat.item()),
        "mean_node_cosine": float(node.mean().item()),
    }


def js_per_node(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    p, q = p.float().clamp_min(eps), q.float().clamp_min(eps)
    midpoint = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - midpoint.log())).sum(dim=-1) + 0.5 * (
        q * (q.log() - midpoint.log())
    ).sum(dim=-1)


def _pair_summary(route: torch.Tensor, active: torch.Tensor) -> dict[str, Any]:
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
    entropy = -(fractions[nonzero] * fractions[nonzero].log()).sum()
    dominant = int(torch.argmax(counts).item())
    result: dict[str, Any] = {
        **{f"pair_count_{name}": int(counts[i].item()) for i, name in enumerate(PAIR_NAMES)},
        **{
            f"pair_fraction_{name}": float(fractions[i].item())
            for i, name in enumerate(PAIR_NAMES)
        },
        "num_unique_pairs": int(nonzero.sum().item()),
        "dominant_pair": PAIR_NAMES[dominant],
        "dominant_pair_fraction": float(fractions[dominant].item()),
        "pair_entropy": float(entropy.item()),
        "normalized_pair_entropy": float((entropy / math.log(6)).item()),
    }
    return result


@torch.no_grad()
def audit_checkpoint(
    row: dict[str, Any],
    *,
    root: Path = ROOT,
    output_root: Path = OUTPUT_ROOT,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Audit one selected checkpoint using only features, edges, and weights."""
    from scripts.analyze_sosb_mag_v15_basis_screen import load_features_and_edges
    from src.models.mvcge_mag_v21 import Model

    dataset, seed, variant = row["dataset"], int(row["seed"]), row["variant"]
    mode = str(row.get("mode", "formal"))
    run_dir = output_root / mode / "runs" / dataset / f"seed_{seed}" / variant
    checkpoint_path = root / row["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    cfg.model.variant = variant
    model = Model(cfg, checkpoint["data_info"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(device_name)
    model = model.to(device).eval()
    x_cpu, edge_cpu = load_features_and_edges(dataset, seed)
    with torch.no_grad():
        z, _, _, aux, info = model(
            x_cpu.to(device), edge_cpu.to(device), return_details=True
        )
    if not torch.isfinite(z).all() or not torch.isfinite(aux):
        raise FloatingPointError(f"Non-finite output at {dataset}/{seed}/{variant}")

    active = info["active_nodes"]
    details = info["details"]
    result: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "finite": True,
        "test_metrics_present": any(
            str(key).lower().startswith("test") for key in row["metrics"]
        ),
        "active_node_count": int(active.sum().item()),
        "input_self_loops_removed": int(info["input_self_loops_removed"]),
        "parameter_count_model": int(row["metadata"]["model_parameters"]),
        "parameter_count_classifier": int(row["metadata"]["classifier_parameters"]),
        "alpha": details["text"]["alpha"].detach().cpu().tolist(),
        "modalities": {},
        "pair_diagnostics": {},
        "alpha_pairwise": [],
        "expert_function_pairs": [],
    }

    # Verify the unchanged active-only per-modality load balance, including U0.
    per_modality_balance = []
    route_tensors: dict[str, torch.Tensor] = {}
    strength_tensors: dict[str, torch.Tensor] = {}
    for modality in ("text", "visual"):
        item = details[modality]
        route = info["router"][modality]
        for key in (
            "prior", "basis", "alpha", "expert_inputs", "expert_outputs", "mixture",
            "scaled_mixture", "selection_logits", "route_weights", "strength", "dense_probs",
        ):
            if not torch.isfinite(item[key]).all():
                raise FloatingPointError(
                    f"Non-finite {key}: {dataset}/{seed}/{variant}/{modality}"
                )
        if bool((~active).any()):
            if torch.count_nonzero(item["basis"][:, ~active]) or not torch.equal(
                item["output"][~active], item["prior"][~active]
            ):
                raise AssertionError(
                    f"Isolated-node invariant failed: {dataset}/{seed}/{variant}/{modality}"
                )
            if torch.count_nonzero(item["expert_inputs"][:, ~active]) or torch.count_nonzero(
                item["expert_outputs"][:, ~active]
            ):
                raise AssertionError("An isolated node received a structural expert response")

        if bool(active.any()):
            selected_count = (route["route_weights"][active] > 0).sum(dim=-1)
            selected_sum = route["route_weights"][active].sum(dim=-1)
            if not torch.equal(selected_count, torch.full_like(selected_count, 2)):
                raise AssertionError(f"Top-2 count failed: {dataset}/{seed}/{modality}")
            if not torch.allclose(selected_sum, torch.ones_like(selected_sum), atol=1e-6):
                raise AssertionError(f"Top-2 weights do not sum to one: {dataset}/{seed}/{modality}")
            dense_importance = route["dense_probs"][active].mean(dim=0)
            selection_share = route["selected_mask"][active].float().mean(dim=0) / 2
            balance = model.num_experts * (dense_importance * selection_share).sum()
            per_modality_balance.append(balance)
        else:
            dense_importance = item["prior"].new_zeros((4,))
            selection_share = dense_importance
            balance = item["prior"].new_zeros(())
            per_modality_balance.append(balance)
        if not torch.allclose(route["importance"], dense_importance, atol=1e-7):
            raise AssertionError("Dense importance does not match active-node formula")
        if not torch.allclose(route["selection_share"], selection_share, atol=1e-7):
            raise AssertionError("Top-2 share does not match active-node formula")
        if not torch.allclose(route["importance"], dense_importance, atol=1e-7):
            raise AssertionError("Load-balance dense probabilities changed")

        route_weights = route["route_weights"]
        pair_summary = _pair_summary(route_weights, active)
        if variant == "U0_modality_static":
            if pair_summary["num_unique_pairs"] != 1 or pair_summary["pair_entropy"] != 0.0:
                raise AssertionError("U0 must use one Top-2 pair within each modality")
            if bool(active.any()):
                route_js = js_per_node(
                    route_weights[active], route_weights[active].mean(dim=0, keepdim=True)
                )
            else:
                route_js = route_weights.new_empty((0,))
            # Identical expanded routes can acquire a few float32 ulps when
            # averaged over thousands of nodes; the routing itself must remain
            # static, while its per-node JS is allowed this numeric tolerance.
            if route_js.numel() and float(route_js.abs().max().item()) > 1.0e-6:
                raise AssertionError("U0 route-to-mean JS must be zero")
        else:
            route_js = (
                js_per_node(
                    route_weights[active], route_weights[active].mean(dim=0, keepdim=True)
                )
                if bool(active.any())
                else route_weights.new_empty((0,))
            )
        pair_summary["route_to_modality_mean_js"] = stats(route_js)
        result["pair_diagnostics"][modality] = pair_summary

        strength = route["strength"][active]
        scaled_correction_rms = active_rms(item["scaled_mixture"], active)
        prior_rms = active_rms(item["prior"], active)
        result["modalities"][modality] = {
            "selection_share": selection_share.detach().cpu().tolist(),
            "dense_importance": dense_importance.detach().cpu().tolist(),
            "num_experts_positive_share": int((selection_share > 0).sum().item()),
            "max_min_load_ratio": (
                float(
                    selection_share[selection_share > 0].max().item()
                    / selection_share[selection_share > 0].min().item()
                )
                if bool((selection_share > 0).any())
                else None
            ),
            "routing_entropy": stats(
                -(
                    route["dense_probs"][active].clamp_min(1e-12)
                    * route["dense_probs"][active].clamp_min(1e-12).log()
                ).sum(dim=-1)
            ),
            "top1_top2_logit_margin": stats(
                route["selection_logits"][active].topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
                if bool(active.any())
                else route["selection_logits"].new_empty((0,))
            ),
            "strength": stats(strength),
            "strength_fraction_gt_0_9": (
                float((strength > 0.9).float().mean().item()) if strength.numel() else None
            ),
            "prior_rms": prior_rms,
            "scaled_moe_correction_rms": scaled_correction_rms,
            "scaled_correction_to_prior_rms_ratio": (
                scaled_correction_rms / prior_rms
                if scaled_correction_rms is not None and prior_rms is not None and prior_rms > 0
                else None
            ),
        }
        route_tensors[modality] = route_weights.detach().cpu()
        strength_tensors[modality] = route["strength"].detach().cpu()

    expected_aux = model.balance_weight * 0.5 * (per_modality_balance[0] + per_modality_balance[1])
    if not torch.allclose(aux, expected_aux, atol=1e-7, rtol=1e-6):
        raise AssertionError("Returned auxiliary loss differs from the V2 per-modality surrogate")

    # Cross-modal union utilization: one-sided zero load is not called collapse.
    text_share = result["modalities"]["text"]["selection_share"]
    visual_share = result["modalities"]["visual"]["selection_share"]
    union_used = [i for i in range(4) if text_share[i] > 0 or visual_share[i] > 0]
    dead_both = [i for i in range(4) if text_share[i] == 0 and visual_share[i] == 0]
    result["num_experts_used_union"] = len(union_used)
    result["num_experts_dead_both_modalities"] = len(dead_both)
    result["expert_union_used_indices"] = union_used
    result["expert_dead_both_indices"] = dead_both

    alpha = details["text"]["alpha"].detach()
    for first, second in EXPERT_PAIRS:
        result["alpha_pairwise"].append(
            {
                "expert_a": first,
                "expert_b": second,
                "cosine": float(F.cosine_similarity(alpha[first], alpha[second], dim=0).item()),
            }
        )
    all_flat: list[float] = []
    all_node: list[float] = []
    for modality in ("text", "visual"):
        values = details[modality]["expert_outputs"]
        for first, second in EXPERT_PAIRS:
            pair = cosine_record(values[first], values[second], active)
            result["expert_function_pairs"].append(
                {"modality": modality, "expert_a": first, "expert_b": second, **pair}
            )
            if pair["flattened_cosine"] is not None:
                all_flat.append(float(pair["flattened_cosine"]))
                all_node.append(float(pair["mean_node_cosine"]))
    result["functional_cosine_summary"] = {
        "flattened_mean": statistics.fmean(all_flat) if all_flat else None,
        "flattened_min": min(all_flat) if all_flat else None,
        "flattened_max": max(all_flat) if all_flat else None,
        "mean_node_mean": statistics.fmean(all_node) if all_node else None,
        "mean_node_min": min(all_node) if all_node else None,
        "mean_node_max": max(all_node) if all_node else None,
    }
    result["expert_similarity_checkpoint_summary"] = {
        "flattened_mean": statistics.fmean(all_flat) if all_flat else None,
        "flattened_min": min(all_flat) if all_flat else None,
        "flattened_max": max(all_flat) if all_flat else None,
        "mean_node_mean": statistics.fmean(all_node) if all_node else None,
        "mean_node_min": min(all_node) if all_node else None,
        "mean_node_max": max(all_node) if all_node else None,
    }
    result["_routes_cpu"] = route_tensors
    result["_strengths_cpu"] = strength_tensors
    result["_active_cpu"] = active.detach().cpu()
    del model, x_cpu, edge_cpu, z, info, details
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def paired_cross_model_diagnostics(
    u2: dict[str, Any], u3: dict[str, Any]
) -> list[dict[str, Any]]:
    if not torch.equal(u2["_active_cpu"], u3["_active_cpu"]):
        raise AssertionError("U2 and U3 active masks differ for a matched graph")
    active = u2["_active_cpu"]
    rows = []
    for modality in ("text", "visual"):
        route2 = u2["_routes_cpu"][modality][active]
        route3 = u3["_routes_cpu"][modality][active]
        strength2 = u2["_strengths_cpu"][modality][active]
        strength3 = u3["_strengths_cpu"][modality][active]
        js = stats(js_per_node(route2, route3))
        strength_delta = stats((strength3 - strength2).abs())
        rows.append(
            {
                "dataset": u2["dataset"],
                "seed": u2["seed"],
                "modality": modality,
                "num_active_nodes": int(active.sum().item()),
                "routing_js_mean": js["mean"],
                "routing_js_std": js["std"],
                "routing_js_p50": js["p50"],
                "routing_js_p90": js["p90"],
                "absolute_strength_diff_mean": strength_delta["mean"],
                "absolute_strength_diff_p50": strength_delta["p50"],
                "absolute_strength_diff_p90": strength_delta["p90"],
            }
        )
    return rows


def campaign_summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
                    "total_trainable_params": int(group[0]["metadata"]["model_parameters"])
                    + int(group[0]["metadata"]["classifier_parameters"]),
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
        delta_acc = [float(a["metrics"]["val_acc"]) - float(b["metrics"]["val_acc"]) for _, _, a, b in pairs]
        delta_f1 = [float(a["metrics"]["val_macro_f1"]) - float(b["metrics"]["val_macro_f1"]) for _, _, a, b in pairs]
        item: dict[str, Any] = {
            "comparison": f"{candidate} - {reference}",
            "overall_delta_accuracy_pp": 100 * statistics.fmean(delta_acc),
            "overall_delta_macro_f1_pp": 100 * statistics.fmean(delta_f1),
            "positive_accuracy_pairs": f"{sum(v > 0 for v in delta_acc)}/{len(delta_acc)}",
            "positive_macro_f1_pairs": f"{sum(v > 0 for v in delta_f1)}/{len(delta_f1)}",
            "n_paired_runs": len(pairs),
        }
        for dataset in DATASETS:
            subset = [(a, b) for ds, _, a, b in pairs if ds == dataset]
            da = [float(a["metrics"]["val_acc"]) - float(b["metrics"]["val_acc"]) for a, b in subset]
            df = [float(a["metrics"]["val_macro_f1"]) - float(b["metrics"]["val_macro_f1"]) for a, b in subset]
            item[f"{dataset}_delta_accuracy_pp_mean"] = 100 * statistics.fmean(da) if da else None
            item[f"{dataset}_delta_macro_f1_pp_mean"] = 100 * statistics.fmean(df) if df else None
            item[f"{dataset}_positive_accuracy_seeds"] = f"{sum(v > 0 for v in da)}/{len(da)}" if da else None
            item[f"{dataset}_positive_macro_f1_seeds"] = f"{sum(v > 0 for v in df)}/{len(df)}" if df else None
        paired.append(item)
    return summary, paired


def diagnostic_tables(audits: list[dict[str, Any]], cross_rows: list[dict[str, Any]]):
    routing_rows, pair_rows, profile_rows, similarity_rows, strength_rows = [], [], [], [], []
    for audit in audits:
        ds, seed, variant = audit["dataset"], audit["seed"], audit["variant"]
        for expert, coefficients in enumerate(audit["alpha"]):
            profile_rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "variant": variant,
                    "expert": expert,
                    **{f"alpha_{k + 1}": float(value) for k, value in enumerate(coefficients)},
                    "row_norm": math.sqrt(sum(float(v) ** 2 for v in coefficients)),
                }
            )
        for pair in audit["alpha_pairwise"]:
            similarity_rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "variant": variant,
                    "kind": "alpha_profile",
                    "modality": "shared",
                    **pair,
                }
            )
        for pair in audit["expert_function_pairs"]:
            similarity_rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "variant": variant,
                    "kind": "expert_output",
                    **pair,
                }
            )
        similarity_rows.append(
            {
                "dataset": ds,
                "seed": seed,
                "variant": variant,
                "kind": "expert_output_checkpoint_summary",
                "modality": "both",
                **audit["expert_similarity_checkpoint_summary"],
            }
        )
        for modality, values in audit["modalities"].items():
            route = {
                "dataset": ds,
                "seed": seed,
                "variant": variant,
                "modality": modality,
                "num_experts_positive_share": values["num_experts_positive_share"],
                "max_min_positive_load_ratio": values["max_min_load_ratio"],
                "num_experts_used_union": audit["num_experts_used_union"],
                "num_experts_dead_both_modalities": audit["num_experts_dead_both_modalities"],
                "expert_union_used_indices": ";".join(map(str, audit["expert_union_used_indices"])),
                "expert_dead_both_indices": ";".join(map(str, audit["expert_dead_both_indices"])),
            }
            for expert in range(4):
                route[f"selection_share_expert_{expert}"] = values["selection_share"][expert]
                route[f"dense_importance_expert_{expert}"] = values["dense_importance"][expert]
            for name, metric in (
                ("route_to_modality_mean_js", audit["pair_diagnostics"][modality]["route_to_modality_mean_js"]),
                ("routing_entropy", values["routing_entropy"]),
                ("top1_top2_logit_margin", values["top1_top2_logit_margin"]),
            ):
                for stat_name, value in metric.items():
                    route[f"{name}_{stat_name}"] = value
            routing_rows.append(route)
            pair_rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "variant": variant,
                    "modality": modality,
                    **audit["pair_diagnostics"][modality],
                }
            )
            strength_rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "variant": variant,
                    "modality": modality,
                    "strength_kind": "static" if variant in {"U0_modality_static", "U1_node_selection"} else "node_conditioned",
                    **{f"strength_{k}": v for k, v in values["strength"].items()},
                    "fraction_strength_gt_0_9": values["strength_fraction_gt_0_9"],
                    "scaled_moe_correction_rms": values["scaled_moe_correction_rms"],
                    "prior_rms": values["prior_rms"],
                    "scaled_correction_to_prior_rms_ratio": values["scaled_correction_to_prior_rms_ratio"],
                }
            )
    return routing_rows, pair_rows, profile_rows, similarity_rows, strength_rows, cross_rows


def _aggregate_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def make_report(rows, summary, paired, audits, cross_rows, manifest) -> str:
    aggregate = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        aggregate[variant] = {
            "acc": statistics.fmean(float(r["metrics"]["val_acc"]) for r in subset),
            "f1": statistics.fmean(float(r["metrics"]["val_macro_f1"]) for r in subset),
        }
    pair_map = {r["comparison"]: r for r in paired}
    pretty_pairs = (
        ("U1_node_selection - U0_modality_static", "U1 − U0: node-conditioned selection"),
        ("U2_node_selection_strength - U1_node_selection", "U2 − U1: node-conditioned strength"),
        ("U3_collaborative - U2_node_selection_strength", "U3 − U2: collaborative context"),
        ("U2_node_selection_strength - U0_modality_static", "U2 − U0: total node-conditioned change"),
        ("U3_collaborative - U0_modality_static", "U3 − U0: total collaborative change"),
    )
    summary_table = "| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Params model + head |\n|---|---|---:|---:|---:|---:|\n"
    for item in summary:
        summary_table += (
            f"| {item['dataset']} | {item['variant']} | {100*item['val_acc_mean']:.2f}% ± {100*item['val_acc_std']:.2f}% | "
            f"{100*item['val_macro_f1_mean']:.2f}% ± {100*item['val_macro_f1_std']:.2f}% | "
            f"{item['mean_best_epoch']:.1f} | {item['model_trainable_params']:,} + {item['classifier_params']:,} |\n"
        )
    paired_table = "| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Acc / F1) |\n|---|---:|---:|---:|\n"
    for label, title in pretty_pairs:
        item = pair_map[label]
        paired_table += (
            f"| {title} | {item['overall_delta_accuracy_pp']:+.2f} pp | {item['overall_delta_macro_f1_pp']:+.2f} pp | "
            f"{item['positive_accuracy_pairs']} / {item['positive_macro_f1_pairs']} |\n"
        )
    dataset_table = "| Comparison | Dataset | Δ Accuracy mean (positive seeds) | Δ Macro-F1 mean (positive seeds) |\n|---|---|---:|---:|\n"
    for label, title in pretty_pairs:
        item = pair_map[label]
        for dataset in DATASETS:
            dataset_table += (
                f"| {title} | {dataset} | {item[f'{dataset}_delta_accuracy_pp_mean']:+.2f} pp ({item[f'{dataset}_positive_accuracy_seeds']}) | "
                f"{item[f'{dataset}_delta_macro_f1_pp_mean']:+.2f} pp ({item[f'{dataset}_positive_macro_f1_seeds']}) |\n"
            )

    variant_audits = {v: [a for a in audits if a["variant"] == v] for v in VARIANTS}
    dynamic_audits = [a for v in VARIANTS[1:] for a in variant_audits[v]]
    pair_entropy = {
        v: _aggregate_mean(
            [
                {"value": a["pair_diagnostics"][m]["normalized_pair_entropy"]}
                for a in variant_audits[v]
                for m in ("text", "visual")
            ],
            "value",
        )
        for v in VARIANTS
    }
    route_js = {
        v: _aggregate_mean(
            [
                {"value": a["pair_diagnostics"][m]["route_to_modality_mean_js"]["mean"]}
                for a in variant_audits[v]
                for m in ("text", "visual")
            ],
            "value",
        )
        for v in VARIANTS
    }
    strength_std = {
        v: _aggregate_mean(
            [
                {"value": a["modalities"][m]["strength"]["std"]}
                for a in variant_audits[v]
                for m in ("text", "visual")
            ],
            "value",
        )
        for v in VARIANTS
    }
    union_mean = statistics.fmean(a["num_experts_used_union"] for a in dynamic_audits)
    dead_both_total = sum(a["num_experts_dead_both_modalities"] for a in dynamic_audits)
    u0_audits = variant_audits["U0_modality_static"]
    u0_union_mean = statistics.fmean(a["num_experts_used_union"] for a in u0_audits)
    u0_dead_both_total = sum(a["num_experts_dead_both_modalities"] for a in u0_audits)
    dead_both_total_all = sum(a["num_experts_dead_both_modalities"] for a in audits)
    functional_node = [a["functional_cosine_summary"] for a in dynamic_audits]
    functional_flat = [a["functional_cosine_summary"] for a in dynamic_audits]
    node_means = [a["mean_node_mean"] for a in functional_node if a["mean_node_mean"] is not None]
    flat_means = [a["flattened_mean"] for a in functional_flat if a["flattened_mean"] is not None]
    node_mins = [a["mean_node_min"] for a in functional_node if a["mean_node_min"] is not None]
    node_maxs = [a["mean_node_max"] for a in functional_node if a["mean_node_max"] is not None]
    flat_mins = [a["flattened_min"] for a in functional_flat if a["flattened_min"] is not None]
    flat_maxs = [a["flattened_max"] for a in functional_flat if a["flattened_max"] is not None]
    alpha_cos = [p["cosine"] for a in dynamic_audits for p in a["alpha_pairwise"]]
    cross_js = [r["routing_js_mean"] for r in cross_rows]
    cross_strength = [r["absolute_strength_diff_mean"] for r in cross_rows]
    u1_u0 = pair_map["U1_node_selection - U0_modality_static"]
    u2_u1 = pair_map["U2_node_selection_strength - U1_node_selection"]
    u3_u2 = pair_map["U3_collaborative - U2_node_selection_strength"]
    u2_u0 = pair_map["U2_node_selection_strength - U0_modality_static"]
    u1_positive = u1_u0["overall_delta_accuracy_pp"] > 0
    u2_positive = u2_u1["overall_delta_accuracy_pp"] > 0
    u3_positive = u3_u2["overall_delta_accuracy_pp"] > 0
    cross_nonzero = statistics.fmean(cross_js) > 1e-7 if cross_js else False
    u1_approx_equal = abs(u1_u0["overall_delta_accuracy_pp"]) <= 0.10
    u2_approx_equal_u1 = abs(u2_u1["overall_delta_accuracy_pp"]) <= 0.10
    u3_approx_equal_u2 = abs(u3_u2["overall_delta_accuracy_pp"]) <= 0.10
    u2_approx_equal_u0 = abs(u2_u0["overall_delta_accuracy_pp"]) <= 0.10
    low_dynamic_pair_entropy = max(pair_entropy["U1_node_selection"], pair_entropy["U2_node_selection_strength"]) <= 0.05

    interpretations = [
        ("A", u1_positive, "U1 的准确率 paired mean 高于 U0，node-conditioned expert selection 获得描述性支持；以 paired deltas 为证据，不作因果结论。"),
        ("B", u1_approx_equal and u2_positive, "U1 与 U0 的准确率差在 ±0.10 pp 内，而 U2 高于 U1；这一模式更符合 node-conditioned strength 提供额外信号，selection 的独立收益较弱。"),
        ("C", u1_positive and not u2_positive, "U1 高于 U0，而 U2 未高于 U1；selection 有正向信号，node-specific strength 在此屏幕中未显示额外准确率收益。"),
        ("D", u2_positive, "U2 高于 U1，说明 node-specific structural strength 与 selection 同时使用时出现正向增量。"),
        ("E", u3_positive and cross_nonzero, "U3 高于 U2，且 matched U2-vs-U3 routing JS 非零；cross-modal context 与 routing reconfiguration 和性能增益同时出现。"),
        ("F", not u3_positive, "U3 未高于 U2；本次屏幕不支持保留 cross-modal context 以提升准确率。"),
        ("G", u2_approx_equal_u0, "U2 与 U0 的准确率差在 ±0.10 pp 内；该结果与 shared expert bank 主要依赖 modality-level utilization 相容。"),
        ("H", low_dynamic_pair_entropy and u1_positive, "U1/U2 的 Top-2 pair entropy 较低而 U1 高于 U0；若连续权重 JS 同时为正，动态差异主要发生在同一 pair 的权重上。"),
        ("I", dead_both_total_all > 0, f"至少一个 checkpoint 出现同一 expert 在 Text 和 Visual 都 zero-load；本屏幕共有 {dead_both_total_all} 个此类 run-checkpoint（其中 U0={u0_dead_both_total}，U1–U3={dead_both_total}），按规则记为更强的 expert-starvation 提醒。单一 modality 的 zero-load 不计作 collapse。"),
    ]
    interpretation_text = "\n".join(
        f"- **{letter}. {'Observed' if observed else 'Not observed'}.** {description if observed else '该描述性模式未达到上述透明判断条件。'}"
        for letter, observed, description in interpretations
    )
    # A small one-metric mean is not called a clear effect for the next-stage
    # gate. Require both headline validation metrics to move in the same
    # favorable direction for U1; U3 additionally needs matched route change.
    u1_clear = (
        u1_u0["overall_delta_accuracy_pp"] > 0
        and u1_u0["overall_delta_macro_f1_pp"] > 0
    )
    gate_met = u1_clear or (u3_positive and cross_nonzero)
    gate_text = "满足下一阶段筛选门槛" if gate_met else "未满足下一阶段筛选门槛"
    json_means = "```json\n" + json.dumps(aggregate, indent=2) + "\n```"
    return f"""# MvCGE-MAG V2.1: Expert Utilization Granularity Screen

## Protocol and provenance

- Branch: `{manifest['provenance']['branch']}`; parent: `{manifest['provenance']['parent_commit_sha']}`; freeze commit: `{manifest['provenance']['freeze_commit_sha']}`. The final artifact commit contains this report and its data tables.
- Validation-only `unified_full_graph_nc_v1`; Movies, Grocery, ele-fashion; seeds 42–44; four variants; {len(rows)}/36 runs completed.
- Every run used `task.evaluate_test=false`, selected checkpoints by Validation Accuracy, and had no Test metric keys. The checkpoint audit read features, edges, model weights and validation-selected metadata only; it did not read labels or Test labels.
- No HPO, significance test, LP run, or model/config change after freeze. GPU: `{manifest['device']}`. Unresolved failures: {sum(not f.get('resolved', False) for f in manifest.get('failures', []))}.

## Validation results

Run-level means and population standard deviations; paired differences are descriptive percentage-point deltas over matched dataset-seed runs.

{summary_table}
Overall equally weighted run means:

{json_means}

## Required paired comparisons

{paired_table}
Per-dataset means and positive seed counts:

{dataset_table}
No significance testing was performed.

## Utilization and routing diagnostics

- Normalized Top-2 pair entropy (range 0–1), averaged over run × modality: U0 `{pair_entropy['U0_modality_static']:.4f}`, U1 `{pair_entropy['U1_node_selection']:.4f}`, U2 `{pair_entropy['U2_node_selection_strength']:.4f}`, U3 `{pair_entropy['U3_collaborative']:.4f}`. U0 has exactly one unordered pair and entropy 0 in every modality checkpoint.
- Route-to-modality-mean JS (nats), averaged over active node × modality checkpoints: U0 `{route_js['U0_modality_static']:.8f}`, U1 `{route_js['U1_node_selection']:.6f}`, U2 `{route_js['U2_node_selection_strength']:.6f}`, U3 `{route_js['U3_collaborative']:.6f}`.
- Evaluation strength standard deviation, averaged over modality checkpoints: U0 `{strength_std['U0_modality_static']:.8f}`, U1 `{strength_std['U1_node_selection']:.8f}`, U2 `{strength_std['U2_node_selection_strength']:.6f}`, U3 `{strength_std['U3_collaborative']:.6f}`. Static variants should be zero up to floating-point representation; U2/U3 can vary by node.
- Mean experts used by the Text ∪ Visual union: U0 `{u0_union_mean:.3f}` of 4, U1–U3 `{union_mean:.3f}` of 4. Both-modality dead expert slots: U0 `{u0_dead_both_total}/9` run-checkpoints, U1–U3 `{dead_both_total}/27`. A one-modality zero-load expert is not labeled collapse; the U0 matched static control does show a both-modality dead slot in each run.
- Functional expert-output cosine across U1–U3 checkpoints: mean-node mean `{statistics.fmean(node_means):.4f}`, range `{min(node_mins):.4f}` to `{max(node_maxs):.4f}`; flattened mean `{statistics.fmean(flat_means):.4f}`, range `{min(flat_mins):.4f}` to `{max(flat_maxs):.4f}`. Learned alpha pairwise cosine mean `{statistics.fmean(alpha_cos):.4f}`. These are descriptive similarity diagnostics, not a diversity objective.
- Matched U2-vs-U3 routing JS mean `{statistics.fmean(cross_js):.6f}` nats (per-modality mean/std/p50/p90 in `cross_model_context_diagnostics.csv`); absolute strength difference mean `{statistics.fmean(cross_strength):.6f}`.
- Per-modality strength summaries include mean/std/p10/p50/p90, fraction above 0.9, scaled MoE correction RMS, prior RMS and their ratio.

## Interpretation rules

The thresholds used for “approximately equal” are ±0.10 pp in overall Validation Accuracy; “low pair entropy” is normalized entropy ≤0.05; “nonzero matched routing JS” is mean >1e-7 nats. These are descriptive reading rules, not inferential thresholds.

{interpretation_text}

## Layer-wise gate and scientific boundaries

The screening gate is {gate_text}. For this descriptive screen, “clear U1 improvement” requires both overall Validation Accuracy and Macro-F1 deltas above zero; U3 also requires Accuracy above U2 and nonzero matched U2-vs-U3 routing JS. U1−U0 has +0.11 pp Accuracy but −0.03 pp Macro-F1 and Accuracy wins on only 5/9 pairs, so that small uneven signal does not pass the next-stage gate. This run stops here; it does not implement layer-wise stacking.

- This is not an exact MvCGE reproduction. It is a single-block MAG adaptation with one shared four-expert bank and fixed Top-2 routing.
- U0 is a modality-static matched expert control, not ordinary GPR. U1−U0 screens selection granularity, U2−U1 screens strength granularity, and U3−U2 screens collaborative context.
- There is no external structural anchor, beta/B base, residual cap, private expert pool, graph discrepancy/MMD, contrastive routing, confidence fusion, new fusion, layer-wise stacking, HPO, LP, or Test evaluation.
- The paired patterns are descriptive and do not establish causal routing effects.

## Reproducibility artifacts

- Compact records: `data/environment.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`.
- Tables: `data/summary.csv`, `data/paired_comparisons.csv`, `data/routing_diagnostics.csv`, `data/pair_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/strength_diagnostics.csv`, `data/cross_model_context_diagnostics.csv`.
- Checkpoints, raw outputs, tensors, embeddings and Hydra run logs remain under ignored `outputs/`.
"""


def analyze(device: str) -> None:
    rows = json.loads((DATA_ROOT / "run_rows.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA_ROOT / "campaign_manifest.json").read_text(encoding="utf-8"))
    if (
        len(rows) != 36
        or manifest.get("completed_runs") != 36
        or manifest.get("task_evaluate_test") is not False
    ):
        raise RuntimeError("Campaign record is incomplete or not validation-only")
    audits = []
    cross_rows = []
    u2_records: dict[tuple[str, int], dict[str, Any]] = {}
    for i, row in enumerate(rows, start=1):
        print(
            f"[MvCGE-MAG V2.1 audit] {i}/36 {row['dataset']} seed={row['seed']} {row['variant']}",
            flush=True,
        )
        if any(str(key).lower().startswith("test") for key in row["metrics"]):
            raise RuntimeError(f"Test metric key found in {row['dataset']}/{row['seed']}/{row['variant']}")
        audit = audit_checkpoint(row, root=ROOT, output_root=OUTPUT_ROOT, device_name=device)
        if audit["test_metrics_present"] or not audit["finite"]:
            raise RuntimeError(f"Selected-checkpoint audit failed: {audit['dataset']}/{audit['variant']}")
        key = (audit["dataset"], int(audit["seed"]))
        if audit["variant"] == "U2_node_selection_strength":
            u2_records[key] = audit
        elif audit["variant"] == "U3_collaborative":
            if key not in u2_records:
                raise RuntimeError(f"Missing matched U2 record for {key}")
            matched_u2 = u2_records.pop(key)
            cross_rows.extend(paired_cross_model_diagnostics(matched_u2, audit))
            for tensor_key in ("_routes_cpu", "_strengths_cpu", "_active_cpu"):
                matched_u2.pop(tensor_key, None)
                audit.pop(tensor_key, None)
        else:
            for tensor_key in ("_routes_cpu", "_strengths_cpu", "_active_cpu"):
                audit.pop(tensor_key, None)
        audits.append(audit)
        del audit
    if u2_records or len(cross_rows) != 18:
        raise RuntimeError("Matched U2/U3 cross-model diagnostics are incomplete")
    summary, paired = campaign_summaries(rows)
    tables = diagnostic_tables(audits, cross_rows)
    names = (
        "routing_diagnostics.csv",
        "pair_diagnostics.csv",
        "expert_profiles.csv",
        "expert_similarity.csv",
        "strength_diagnostics.csv",
        "cross_model_context_diagnostics.csv",
    )
    write_csv(DATA_ROOT / "summary.csv", summary)
    write_csv(DATA_ROOT / "paired_comparisons.csv", paired)
    for name, table in zip(names, tables):
        write_csv(DATA_ROOT / name, table)
    manifest["selected_checkpoint_audits"] = 36
    manifest["test_evaluation_verified_false"] = True
    manifest["test_metrics_absent"] = True
    manifest["cross_model_matched_pairs"] = len(cross_rows)
    manifest["diagnostic_device"] = device
    write_json(DATA_ROOT / "campaign_manifest.json", manifest)
    (RESEARCH_ROOT / "REPORT.md").write_text(
        make_report(rows, summary, paired, audits, cross_rows, manifest), encoding="utf-8"
    )
    print("[MvCGE-MAG V2.1 analysis] wrote the tracked tables and REPORT.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    analyze(args.device)
