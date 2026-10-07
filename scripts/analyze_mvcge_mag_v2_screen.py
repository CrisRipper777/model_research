#!/usr/bin/env python3
"""Create validation-selected, label-free MvCGE-MAG V2 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
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
VARIANTS = ("M0_anchor", "M1_direct_moe", "M2_anchored_moe", "M3_collaborative_moe")
PAIRS = (
    ("M1_direct_moe", "M0_anchor"),
    ("M2_anchored_moe", "M0_anchor"),
    ("M2_anchored_moe", "M1_direct_moe"),
    ("M3_collaborative_moe", "M2_anchored_moe"),
    ("M3_collaborative_moe", "M0_anchor"),
)
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v2_anchored_experts"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v2_anchored_experts"
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
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stats(values: torch.Tensor) -> dict[str, float | None]:
    values = values.detach().reshape(-1).float()
    if values.numel() == 0:
        return {name: None for name in ("mean", "std", "p10", "p50", "p90")}
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
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


def cosine_record(a: torch.Tensor, b: torch.Tensor, active: torch.Tensor) -> dict[str, float | None]:
    if not bool(active.any()):
        return {"flattened_cosine": None, "mean_node_cosine": None}
    aa, bb = a[active].float(), b[active].float()
    node = F.cosine_similarity(aa, bb, dim=-1, eps=1.0e-8)
    flat = F.cosine_similarity(aa.reshape(-1), bb.reshape(-1), dim=0, eps=1.0e-8)
    return {"flattened_cosine": float(flat.item()), "mean_node_cosine": float(node.mean().item())}


def js_per_node(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    p, q = p.float().clamp_min(eps), q.float().clamp_min(eps)
    midpoint = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - midpoint.log())).sum(dim=-1) + 0.5 * (
        q * (q.log() - midpoint.log())
    ).sum(dim=-1)


@torch.no_grad()
def audit_checkpoint(
    row: dict[str, Any], *, root: Path = ROOT, output_root: Path = OUTPUT_ROOT, device_name: str = "cuda:0"
) -> dict[str, Any]:
    """Inspect one selected checkpoint without loading or indexing label fields."""
    from scripts.analyze_sosb_mag_v15_basis_screen import load_features_and_edges
    from src.models.mvcge_mag_v2 import Model

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
        z, _, _, aux, info = model(x_cpu.to(device), edge_cpu.to(device), return_details=True)
    if not torch.isfinite(z).all() or not torch.isfinite(aux):
        raise FloatingPointError(f"Non-finite embedding or aux loss at {dataset}/{seed}/{variant}")
    active = info["active_nodes"]
    test_metrics_present = any(str(key).lower().startswith("test") for key in row["metrics"])
    result: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "finite": True,
        "test_metrics_present": test_metrics_present,
        "embedding_finite": bool(torch.isfinite(z).all()),
        "aux_loss_finite": bool(torch.isfinite(aux)),
        "active_node_count": int(active.sum().item()),
        "input_self_loops_removed": info["input_self_loops_removed"],
        "parameter_count_model": int(row["metadata"]["model_parameters"]),
        "parameter_count_classifier": int(row["metadata"]["classifier_parameters"]),
        "moe_active": variant != "M0_anchor",
        "top2_exact_active": variant == "M0_anchor",
        "load_balance_formula_consistent": variant == "M0_anchor",
        "alpha": info["details"]["text"]["alpha"].detach().cpu().tolist(),
        "modalities": {},
        "alpha_pairwise": [],
        "expert_function_pairs": [],
        "routing_js": None,
    }
    details = info["details"]
    if variant == "M0_anchor":
        if float(aux.item()) != 0.0:
            raise AssertionError("M0 aux_loss must be exactly zero")
    else:
        expected_balance = 0.5 * sum(
            model.num_experts
            * (info["router"][modality]["importance"] * info["router"][modality]["selection_share"]).sum()
            for modality in ("text", "visual")
        )
        expected_aux = model.balance_weight * expected_balance
        if not torch.allclose(aux, expected_aux, atol=1e-7, rtol=1e-6):
            raise AssertionError("Returned auxiliary loss does not match the per-modality load-balance surrogate")
        result["load_balance_formula_consistent"] = True
        result["top2_exact_active"] = True
    # Check protected isolated-node and exact-zero structural invariants.
    for modality in ("text", "visual"):
        item = details[modality]
        for key in (
            "prior", "basis", "beta", "base", "base_scaled", "alpha",
            "expert_inputs", "expert_outputs", "mixture", "scaled_mixture",
            "selection_logits", "route_weights", "strength", "dense_probs",
        ):
            if not torch.isfinite(item[key]).all():
                raise FloatingPointError(f"Non-finite {key}: {dataset}/{seed}/{variant}/{modality}")
        if bool((~active).any()):
            if torch.count_nonzero(item["basis"][:, ~active]) or not torch.equal(
                item["output"][~active], item["prior"][~active]
            ):
                raise AssertionError(f"Isolated structural invariant failed: {dataset}/{seed}/{variant}/{modality}")
            if torch.count_nonzero(item["expert_inputs"][:, ~active]) or torch.count_nonzero(
                item["expert_outputs"][:, ~active]
            ):
                raise AssertionError(f"Isolated expert output was nonzero: {dataset}/{seed}/{variant}/{modality}")

        route = info["router"][modality]
        if variant != "M0_anchor" and bool(active.any()):
            selected_sum = route["route_weights"][active].sum(dim=-1)
            selected_count = (route["route_weights"][active] > 0).sum(dim=-1)
            if not torch.allclose(selected_sum, torch.ones_like(selected_sum), atol=1e-6):
                raise AssertionError(f"Sparse routing weights do not sum to one: {dataset}/{seed}/{modality}")
            if not torch.equal(selected_count, torch.full_like(selected_count, 2)):
                raise AssertionError(f"Top-2 routing is not exact: {dataset}/{seed}/{modality}")
        elif variant != "M0_anchor":
            result["top2_exact_active"] = False
        probs = route["dense_probs"][active]
        logits = route["selection_logits"][active]
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        margin = logits.topk(2, dim=-1).values
        margin = margin[:, 0] - margin[:, 1]
        selected_share = route["selected_mask"][active].float().mean(dim=0) / 2 if bool(active.any()) else torch.zeros(4, device=device)
        importance = route["dense_probs"][active].mean(dim=0) if bool(active.any()) else torch.zeros(4, device=device)
        positive_loads = selected_share[selected_share > 0]
        load_ratio = (
            float(positive_loads.max().item() / positive_loads.min().item())
            if positive_loads.numel()
            else None
        )
        strength = route["strength"][active]
        rho = model.residual_max * strength if variant in {"M2_anchored_moe", "M3_collaborative_moe"} else None
        if variant == "M1_direct_moe":
            gate_summary = stats(strength)
            saturation_fraction = None
        elif rho is not None:
            gate_summary = stats(rho)
            saturation_fraction = float((rho > 0.9 * model.residual_max).float().mean().item()) if rho.numel() else None
        else:
            gate_summary = stats(torch.sigmoid(model.gamma_base).reshape(1).expand(int(active.sum().item())))
            saturation_fraction = None
        prior_rms = active_rms(item["prior"], active)
        base_rms = active_rms(item["base"], active)
        base_scaled_rms = active_rms(item["base_scaled"], active)
        expert_scaled_rms = active_rms(item["scaled_mixture"], active)
        expert_to_base = (
            expert_scaled_rms / base_scaled_rms
            if variant in {"M2_anchored_moe", "M3_collaborative_moe"}
            and base_scaled_rms is not None and base_scaled_rms > 0
            else None
        )
        result["modalities"][modality] = {
            "selection_share": selected_share.detach().cpu().tolist(),
            "dense_importance": importance.detach().cpu().tolist(),
            "num_experts_positive_share": int((selected_share > 0).sum().item()),
            "max_min_load_ratio": load_ratio,
            "routing_entropy": stats(entropy),
            "top1_top2_logit_margin": stats(margin),
            "strength": gate_summary,
            "rho_saturation_fraction": saturation_fraction,
            "prior_rms": prior_rms,
            "base_response_rms": base_rms if variant in {"M0_anchor", "M2_anchored_moe", "M3_collaborative_moe"} else None,
            "scaled_base_correction_rms": base_scaled_rms if variant in {"M0_anchor", "M2_anchored_moe", "M3_collaborative_moe"} else None,
            "scaled_direct_moe_rms": expert_scaled_rms if variant == "M1_direct_moe" else None,
            "scaled_expert_residual_rms": expert_scaled_rms if variant in {"M2_anchored_moe", "M3_collaborative_moe"} else None,
            "expert_residual_to_base_ratio": expert_to_base,
            "route_weights": route["route_weights"].detach(),
            "expert_outputs": item["expert_outputs"].detach(),
        }

    # One shared alpha bank, and per modality functional expert comparisons.
    alpha = details["text"]["alpha"].detach()
    for first in range(4):
        for second in range(first + 1, 4):
            result["alpha_pairwise"].append(
                {
                    "expert_a": first,
                    "expert_b": second,
                    "cosine": float(F.cosine_similarity(alpha[first], alpha[second], dim=0).item()),
                }
            )
    all_flat, all_node = [], []
    for modality in ("text", "visual"):
        expert_outputs = details[modality]["expert_outputs"]
        for first in range(4):
            for second in range(first + 1, 4):
                record = cosine_record(expert_outputs[first], expert_outputs[second], active)
                all_flat.append(record["flattened_cosine"])
                all_node.append(record["mean_node_cosine"])
                result["expert_function_pairs"].append(
                    {"modality": modality, "expert_a": first, "expert_b": second, **record}
                )
    result["functional_cosine_summary"] = {
        "flattened_mean": statistics.fmean(v for v in all_flat if v is not None) if any(v is not None for v in all_flat) else None,
        "flattened_min": min((v for v in all_flat if v is not None), default=None),
        "flattened_max": max((v for v in all_flat if v is not None), default=None),
        "mean_node_mean": statistics.fmean(v for v in all_node if v is not None) if any(v is not None for v in all_node) else None,
        "mean_node_min": min((v for v in all_node if v is not None), default=None),
        "mean_node_max": max((v for v in all_node if v is not None), default=None),
    }
    if variant in {"M2_anchored_moe", "M3_collaborative_moe"} and bool(active.any()):
        result["routing_js"] = stats(
            js_per_node(
                result["modalities"]["text"]["route_weights"][active],
                result["modalities"]["visual"]["route_weights"][active],
            )
        )
    # Keep the serialized diagnostic compact: no node tensors are retained.
    for modality in result["modalities"]:
        result["modalities"][modality].pop("route_weights")
        result["modalities"][modality].pop("expert_outputs")
    del model, x_cpu, edge_cpu, z, info, details
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def campaign_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    by_key = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in rows}
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            if not group:
                continue
            acc = [float(r["metrics"]["val_acc"]) for r in group]
            f1 = [float(r["metrics"]["val_macro_f1"]) for r in group]
            epochs = [int(r["metadata"]["best_epoch"]) for r in group]
            models = {int(r["metadata"]["model_parameters"]) for r in group}
            heads = {int(r["metadata"]["classifier_parameters"]) for r in group}
            if len(models) != 1 or len(heads) != 1:
                raise ValueError(f"Parameter counts changed within {dataset}/{variant}")
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
    paired: list[dict[str, Any]] = []
    for candidate, reference in PAIRS:
        pairs = [
            (dataset, seed, by_key[(dataset, seed, candidate)], by_key[(dataset, seed, reference)])
            for dataset in DATASETS
            for seed in SEEDS
            if (dataset, seed, candidate) in by_key and (dataset, seed, reference) in by_key
        ]
        if not pairs:
            continue
        acc_delta = [float(a["metrics"]["val_acc"]) - float(b["metrics"]["val_acc"]) for _, _, a, b in pairs]
        f1_delta = [float(a["metrics"]["val_macro_f1"]) - float(b["metrics"]["val_macro_f1"]) for _, _, a, b in pairs]
        base = {
            "comparison": f"{candidate} - {reference}",
            "overall_delta_accuracy_pp": 100.0 * statistics.fmean(acc_delta),
            "overall_delta_macro_f1_pp": 100.0 * statistics.fmean(f1_delta),
            "positive_accuracy_pairs": f"{sum(v > 0 for v in acc_delta)}/{len(acc_delta)}",
            "positive_macro_f1_pairs": f"{sum(v > 0 for v in f1_delta)}/{len(f1_delta)}",
            "n_paired_runs": len(pairs),
        }
        for dataset in DATASETS:
            subset = [(a, b) for d, _, a, b in pairs if d == dataset]
            da = [float(a["metrics"]["val_acc"]) - float(b["metrics"]["val_acc"]) for a, b in subset]
            df = [float(a["metrics"]["val_macro_f1"]) - float(b["metrics"]["val_macro_f1"]) for a, b in subset]
            base[f"{dataset}_delta_accuracy_pp_mean"] = 100.0 * statistics.fmean(da) if da else None
            base[f"{dataset}_delta_macro_f1_pp_mean"] = 100.0 * statistics.fmean(df) if df else None
            base[f"{dataset}_positive_accuracy_seeds"] = f"{sum(v > 0 for v in da)}/{len(da)}" if da else None
            base[f"{dataset}_positive_macro_f1_seeds"] = f"{sum(v > 0 for v in df)}/{len(df)}" if df else None
        paired.append(base)
    return summary, paired


def diagnostic_tables(audits: list[dict[str, Any]]):
    routing_rows, profile_rows, similarity_rows, strength_rows = [], [], [], []
    for audit in audits:
        ds, seed, variant = audit["dataset"], audit["seed"], audit["variant"]
        if variant != "M0_anchor":
            for expert, coefficients in enumerate(audit["alpha"], start=0):
                profile_rows.append(
                    {
                        "dataset": ds, "seed": seed, "variant": variant, "expert": expert,
                        **{f"alpha_{k+1}": float(value) for k, value in enumerate(coefficients)},
                        "row_norm": math.sqrt(sum(float(v) ** 2 for v in coefficients)),
                    }
                )
            for pair in audit["alpha_pairwise"]:
                similarity_rows.append(
                    {"dataset": ds, "seed": seed, "variant": variant, "kind": "alpha_profile", "modality": "shared", **pair}
                )
            for pair in audit["expert_function_pairs"]:
                similarity_rows.append(
                    {"dataset": ds, "seed": seed, "variant": variant, "kind": "expert_output", **pair}
                )
        for modality, values in audit["modalities"].items():
            load = values["selection_share"]
            importance = values["dense_importance"]
            routing_row = {
                "dataset": ds, "seed": seed, "variant": variant, "modality": modality,
                "moe_active": variant != "M0_anchor",
                "num_experts_positive_share": values["num_experts_positive_share"],
                "max_min_load_ratio": values["max_min_load_ratio"],
            }
            for expert in range(4):
                routing_row[f"selection_share_expert_{expert}"] = load[expert]
                routing_row[f"dense_importance_expert_{expert}"] = importance[expert]
            for name, metric in (("entropy", values["routing_entropy"]), ("margin", values["top1_top2_logit_margin"])):
                for stat_name, stat_value in metric.items():
                    routing_row[f"{name}_{stat_name}"] = stat_value
            routing_rows.append(routing_row)
            strength_rows.append(
                {
                    "dataset": ds, "seed": seed, "variant": variant, "modality": modality,
                    "strength_kind": "direct_gate" if variant == "M1_direct_moe" else (
                        "rho" if variant in {"M2_anchored_moe", "M3_collaborative_moe"} else "base_gate"
                    ),
                    **{f"strength_{k}": v for k, v in values["strength"].items()},
                    "rho_saturation_fraction": values["rho_saturation_fraction"],
                    "prior_rms": values["prior_rms"],
                    "base_response_rms": values["base_response_rms"],
                    "scaled_base_correction_rms": values["scaled_base_correction_rms"],
                    "scaled_direct_moe_rms": values["scaled_direct_moe_rms"],
                    "scaled_expert_residual_rms": values["scaled_expert_residual_rms"],
                    "expert_residual_to_base_ratio": values["expert_residual_to_base_ratio"],
                }
            )
    return routing_rows, profile_rows, similarity_rows, strength_rows


def make_report(rows, summary, paired, audits, manifest) -> str:
    aggregate: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        vrows = [r for r in rows if r["variant"] == variant]
        aggregate[variant] = {
            "acc": statistics.fmean(float(r["metrics"]["val_acc"]) for r in vrows),
            "f1": statistics.fmean(float(r["metrics"]["val_macro_f1"]) for r in vrows),
        }
    pair_map = {p["comparison"]: p for p in paired}
    table = "| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Mean best epoch | Trainable params (model + head) |\n|---|---|---:|---:|---:|---:|\n"
    for item in summary:
        table += (
            f"| {item['dataset']} | {item['variant']} | {100*item['val_acc_mean']:.2f}% ± {100*item['val_acc_std']:.2f}% | "
            f"{100*item['val_macro_f1_mean']:.2f}% ± {100*item['val_macro_f1_std']:.2f}% | "
            f"{item['mean_best_epoch']:.1f} | {item['model_trainable_params']:,} + {item['classifier_params']:,} |\n"
        )
    paired_table = "| Comparison | Δ Accuracy | Δ Macro-F1 | Positive accuracy pairs | Positive Macro-F1 pairs |\n|---|---:|---:|---:|---:|\n"
    for item in paired:
        paired_table += (
            f"| {item['comparison']} | {item['overall_delta_accuracy_pp']:+.2f} pp | {item['overall_delta_macro_f1_pp']:+.2f} pp | "
            f"{item['positive_accuracy_pairs']} | {item['positive_macro_f1_pairs']} |\n"
        )
    dataset_pair_rows = []
    for item in paired:
        for dataset in DATASETS:
            dataset_pair_rows.append(
                f"| {item['comparison']} | {dataset} | {item[f'{dataset}_delta_accuracy_pp_mean']:+.2f} pp "
                f"({item[f'{dataset}_positive_accuracy_seeds']}) | {item[f'{dataset}_delta_macro_f1_pp_mean']:+.2f} pp "
                f"({item[f'{dataset}_positive_macro_f1_seeds']}) |"
            )
    data_pair_table = "| Comparison | Dataset | Δ Accuracy mean (positive seeds) | Δ Macro-F1 mean (positive seeds) |\n|---|---|---:|---:|\n" + "\n".join(dataset_pair_rows) + "\n"
    m2m0 = pair_map["M2_anchored_moe - M0_anchor"]
    m2m1 = pair_map["M2_anchored_moe - M1_direct_moe"]
    m3m2 = pair_map["M3_collaborative_moe - M2_anchored_moe"]
    mean_a = [a for a in audits if a["variant"] in {"M1_direct_moe", "M2_anchored_moe", "M3_collaborative_moe"}]
    mean_functional = statistics.fmean(a["functional_cosine_summary"]["mean_node_mean"] for a in mean_a)
    mean_functional_flat = statistics.fmean(a["functional_cosine_summary"]["flattened_mean"] for a in mean_a)
    min_functional_node = min(a["functional_cosine_summary"]["mean_node_min"] for a in mean_a)
    max_functional_node = max(a["functional_cosine_summary"]["mean_node_max"] for a in mean_a)
    min_functional_flat = min(a["functional_cosine_summary"]["flattened_min"] for a in mean_a)
    max_functional_flat = max(a["functional_cosine_summary"]["flattened_max"] for a in mean_a)
    mean_alpha_cos = statistics.fmean(p["cosine"] for a in mean_a for p in a["alpha_pairwise"])
    moes = [a for a in audits if a["variant"] != "M0_anchor"]
    experts_all_used = sum(
        1 for a in moes for m in a["modalities"].values() if m["num_experts_positive_share"] == 4
    )
    load_count = sum(len(a["modalities"]) for a in moes)
    load_ratios = [
        a["modalities"][m]["max_min_load_ratio"]
        for a in moes for m in ("text", "visual")
        if a["modalities"][m]["max_min_load_ratio"] is not None
    ]
    m2m3 = [a for a in audits if a["variant"] in {"M2_anchored_moe", "M3_collaborative_moe"}]
    entropy_avg = statistics.fmean(
        a["modalities"][m]["routing_entropy"]["mean"] for a in moes for m in ("text", "visual")
    )
    entropy_std_avg = statistics.fmean(
        a["modalities"][m]["routing_entropy"]["std"] for a in moes for m in ("text", "visual")
    )
    margin_avg = statistics.fmean(
        a["modalities"][m]["top1_top2_logit_margin"]["mean"] for a in moes for m in ("text", "visual")
    )
    margin_std_avg = statistics.fmean(
        a["modalities"][m]["top1_top2_logit_margin"]["std"] for a in moes for m in ("text", "visual")
    )
    rho_fractions = [
        a["modalities"][m]["rho_saturation_fraction"]
        for a in m2m3 for m in ("text", "visual")
        if a["modalities"][m]["rho_saturation_fraction"] is not None
    ]
    rho_avg = statistics.fmean(rho_fractions) if rho_fractions else None
    ratios = [
        a["modalities"][m]["expert_residual_to_base_ratio"]
        for a in m2m3 for m in ("text", "visual")
        if a["modalities"][m]["expert_residual_to_base_ratio"] is not None
    ]
    ratio_avg = statistics.fmean(ratios) if ratios else None
    alpha_diverse = mean_alpha_cos < 0.95
    function_homogenized = mean_functional > 0.90
    load_healthy = experts_all_used == load_count
    m2_supports_anchor = m2m0["overall_delta_accuracy_pp"] > 0
    m1_lower = pair_map["M1_direct_moe - M0_anchor"]["overall_delta_accuracy_pp"] < 0
    m2_over_m1 = m2m1["overall_delta_accuracy_pp"] > 0
    m1_over_m0 = not m1_lower
    m2_under_m1 = not m2_over_m1
    m3_over_m2 = m3m2["overall_delta_accuracy_pp"] > 0
    m3_routing_differs = any(
        a["routing_js"] is not None and a["routing_js"]["mean"] > 0.0 for a in m2m3
    )
    m2m3_mean_js = statistics.fmean(a["routing_js"]["mean"] for a in m2m3 if a["routing_js"] is not None)
    direct_text = "```json\n" + json.dumps(aggregate, indent=2) + "\n```"
    scenarios = [
        ("A", m2_supports_anchor and not function_homogenized and load_healthy,
         "M2 exceeds M0 with differentiated expert outputs and healthy observed load. This supports anchored shared-expert specialization for a next-stage layer-wise/stronger expert study."),
        ("B", m1_lower and m2_over_m1 and m2m0["overall_delta_accuracy_pp"] >= 0,
         "M1 is below M0 while M2 improves on M1 and is near or above M0. This is consistent with direct dynamic MoE disrupting stable structural processing and the anchor helping; whether experts add value is still determined by M2−M0."),
        ("C", m1_over_m0 and m2_under_m1 and rho_avg is not None and rho_avg > 0.5,
         "M1 exceeds M0, M2 is below M1, and many residual strengths approach the cap. This can indicate that the bounded residual limits expert capacity; it does not by itself reject MoE."),
        ("D", m3_over_m2 and m3_routing_differs,
         "M3 exceeds M2 and their text/visual sparse routing distributions differ. This supports independent value from the DMCAR-inspired public cross-view routing context in this screen."),
        ("E", not m3_over_m2,
         "M3 is no better than M2 on overall paired Validation Accuracy. The current lightweight cross-modal routing context has no observed gain here and need not be retained."),
        ("F", alpha_diverse and function_homogenized,
         "Learned alpha profiles differ while expert output cosine remains high. This is consistent with functional expert homogenization; a follow-up should first study specialization/diversity rather than add router complexity."),
        ("G", not load_healthy,
         "At least one MoE modality-checkpoint does not use all four experts. Review the observed shares and max/min ratios before interpreting the result as evidence against shared experts; the balance objective/weight may need a dedicated study."),
        ("H", entropy_avg / math.log(4) > 0.9 and margin_avg < 0.1 and aggregate["M3_collaborative_moe"]["acc"] < aggregate["M0_anchor"]["acc"],
         "Routing entropy is near the four-way uniform maximum, mean Top1−Top2 margin is small, and M3 underperforms M0. This is consistent with routing uncertainty; guidance/confidence mechanisms belong to a separate future study."),
    ]
    scenario_text = "\n".join(
        f"- **{label}. {'Observed' if condition else 'Not triggered'}.** {description if condition else 'The specified joint pattern was not observed in these descriptive summaries.'}"
        for label, condition, description in scenarios
    )
    return f"""# MvCGE-MAG V2: Anchored Collaborative Structural Experts

## Protocol and provenance

- Branch: `{manifest['provenance']['branch']}`; parent: `{manifest['provenance']['parent_commit_sha']}`; freeze commit: `{manifest['provenance']['freeze_commit_sha']}`.
- Validation-only protocol: `unified_full_graph_nc_v1`; datasets Movies, Grocery, ele-fashion; seeds 42, 43, 44; four fixed variants; {len(rows)}/36 completed.
- `task.evaluate_test=false` for every run. Metrics JSON contains no Test metrics; diagnostics loaded features, graph edges, model weights and validation-selected metadata without reading label fields or indexing Test labels. The standard data loader materializes the full label vector as allowed by the protocol.
- Checkpoint selection used Validation Accuracy. No HPO, significance test, intervention, LP task, or model/config change occurred after freeze.
- GPU: `{manifest['device']}`. Failures recorded: {sum(not item.get('resolved', False) for item in manifest.get('failures', []))} unresolved.

## Validation results

Values are run-level means and population standard deviations. Paired deltas are descriptive percentage points across matched dataset-seed runs.

{table}
Overall equally weighted run means:

{direct_text}

## Paired comparisons

{paired_table}
Dataset means and positive seed counts:

{data_pair_table}
No inferential or significance testing was performed.

## Selected-checkpoint mechanism diagnostics

- Expert profiles were analyzed for M1/M2/M3: learned alpha pairwise cosine mean `{mean_alpha_cos:.4f}`. Hadamard is only the initialization.
- Across {load_count} MoE modality/checkpoint records, all four experts had positive selection share in `{experts_all_used}/{load_count}` records; `{load_count-experts_all_used}` records had at least one zero-load expert. The median max/min ratio across positive loads was `{statistics.median(load_ratios):.3f}`.
- Functional expert output cosine: mean-node mean `{mean_functional:.4f}`, pairwise range `{min_functional_node:.4f}` to `{max_functional_node:.4f}`; flattened mean `{mean_functional_flat:.4f}`, range `{min_functional_flat:.4f}` to `{max_functional_flat:.4f}`.
- Dense routing entropy mean ± mean within-checkpoint std: `{entropy_avg:.4f} ± {entropy_std_avg:.4f}` nats (maximum for four experts is `{math.log(4):.4f}`); Top1−Top2 logit margin mean ± mean within-checkpoint std: `{margin_avg:.4f} ± {margin_std_avg:.4f}`.
- Mean fraction with `rho > 0.9*rho_max` across M2/M3 modality-checkpoints: `{rho_avg:.4f}`.
- Mean scaled expert residual/base correction RMS ratio across M2/M3: `{ratio_avg:.4f}`.
- M2/M3 text-vs-visual sparse routing JS mean across selected checkpoints: `{m2m3_mean_js:.4f}` nats.

Detailed per-run values are in `data/routing_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, and `data/strength_diagnostics.csv`.

## Interpretation map

The conditions below are applied as descriptive patterns; they do not establish causal mechanisms.

{scenario_text}

## Scientific boundaries

- This is an MvCGE-skeleton MAG-specific adaptation, not an exact MvCGE reproduction.
- It implements one collaborative expert block, not the full layer-wise architecture.
- Load balancing is an MvCGE-inspired per-modality surrogate, not an exact reproduction of Eq. (11).
- There is no graph discrepancy/MMD, DMCAR private expert pool, C2GMoE contrastive routing or confidence fusion, edge routing, topology learning, cross-modal attention, or modality transformer.
- Hadamard rows initialize trainable profiles; they do not define fixed expert semantics. RawPoly supplies a structural response substrate and is not described as a GPR model.
- No expert is assigned a fixed frequency band. Routing diagnostics do not support causal claims about node routing.

## Reproducibility artifacts

- `data/environment.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/routing_diagnostics.csv`
- `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/strength_diagnostics.csv`
- Raw outputs, checkpoints, and Hydra logs remain in the ignored `outputs/` tree and are not committed.
"""


def analyze(device: str) -> None:
    rows = json.loads((DATA_ROOT / "run_rows.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA_ROOT / "campaign_manifest.json").read_text(encoding="utf-8"))
    if len(rows) != 36 or manifest.get("completed_runs") != 36 or manifest.get("task_evaluate_test") is not False:
        raise RuntimeError("Campaign record is incomplete or not validation-only")
    audits = []
    for i, row in enumerate(rows, start=1):
        print(f"[MvCGE-MAG V2 audit] {i}/36 {row['dataset']} seed={row['seed']} {row['variant']}", flush=True)
        if any(str(key).lower().startswith("test") for key in row["metrics"]):
            raise RuntimeError(f"Test metrics appeared in {row['dataset']}/{row['seed']}/{row['variant']}")
        audit = audit_checkpoint(row, root=ROOT, output_root=OUTPUT_ROOT, device_name=device)
        if audit["test_metrics_present"] or not audit["finite"]:
            raise RuntimeError(f"Selected-checkpoint audit failed: {audit}")
        audits.append(audit)
    summary, paired = campaign_summaries(rows)
    routing, profiles, similarity, strength = diagnostic_tables(audits)
    write_csv(DATA_ROOT / "summary.csv", summary)
    write_csv(DATA_ROOT / "paired_comparisons.csv", paired)
    write_csv(DATA_ROOT / "routing_diagnostics.csv", routing)
    write_csv(DATA_ROOT / "expert_profiles.csv", profiles)
    write_csv(DATA_ROOT / "expert_similarity.csv", similarity)
    write_csv(DATA_ROOT / "strength_diagnostics.csv", strength)
    manifest["selected_checkpoint_audits"] = 36
    manifest["test_evaluation_verified_false"] = True
    manifest["test_metrics_absent"] = True
    manifest["diagnostic_device"] = device
    write_json(DATA_ROOT / "campaign_manifest.json", manifest)
    (RESEARCH_ROOT / "REPORT.md").write_text(make_report(rows, summary, paired, audits, manifest), encoding="utf-8")
    print("[MvCGE-MAG V2 analysis] wrote tracked campaign tables and REPORT.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    analyze(args.device)
