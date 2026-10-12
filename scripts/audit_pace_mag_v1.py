#!/usr/bin/env python3
"""Audit the frozen PACE-MAG V1 E2 checkpoints without retraining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "research/pace_mag_v1_mechanism_audit/data"
PILOT_DIR = ROOT / "research/pace_mag_v1/data"
PILOT_RUNS = ROOT / "outputs/pace_mag_v1_pilot/runs"
DATASETS = ("Movies", "Grocery", "ele-fashion")
FREEZE_SHA = "739f340bb6858b7280dede1dfaaf37740c2de3fa"
PILOT_SHA = "d4094d3f003826d71a496495e27d8998f8f594ec"
AUDIT_PERMUTATION_SEED = 20261012
TOL = 1e-5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.pace_mag_v1_audit import (  # noqa: E402
    audit_forward,
    context_comparison,
    jensen_shannon,
    node_norms,
    summarize_values,
    tensor_sha256,
    validation_metrics,
)
from src.models.pace_mag_v1 import _MODALITIES, row_quality_aggregate  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["dataset", "metric"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _cuda_snapshot(device: torch.device) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": props.name,
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    import importlib.metadata

    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15, check=True).stdout
    except Exception as exc:
        smi = f"unavailable: {type(exc).__name__}: {exc}"
    versions = {}
    for distribution in ("torch-geometric", "dgl", "faiss-cpu", "faiss-gpu"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "selected_device": _cuda_snapshot(device),
        "package_versions": versions,
        "nvidia_smi_snapshot": smi,
    }


def _source_row(rows: list[dict[str, Any]], dataset: str, family: str, metric: str, values, *, view: str = "", modality: str = "", condition: str = "", mask=None, unit: str = "", notes: str = "") -> None:
    row = {
        "dataset": dataset,
        "family": family,
        "view": view,
        "modality": modality,
        "condition": condition,
        "metric": metric,
        "unit": unit,
        "notes": notes,
    }
    row.update(summarize_values(values, mask))
    rows.append(row)


def _directed_codes(unique: torch.Tensor, num_nodes: int) -> np.ndarray:
    array = unique.detach().cpu().numpy().astype(np.int64, copy=False)
    if array.shape[1] == 0:
        return np.empty((0,), dtype=np.int64)
    forward = array[0] * num_nodes + array[1]
    reverse = array[1] * num_nodes + array[0]
    return np.sort(np.concatenate((forward, reverse)))


def _edge_overlap(left: np.ndarray, right: np.ndarray, num_nodes: int) -> dict[str, Any]:
    shared = np.intersect1d(left, right, assume_unique=True)
    left_degree = np.bincount(left // num_nodes, minlength=num_nodes)
    right_degree = np.bincount(right // num_nodes, minlength=num_nodes)
    shared_degree = np.bincount(shared // num_nodes, minlength=num_nodes)
    union_degree = left_degree + right_degree - shared_degree
    valid = union_degree > 0
    jaccard = np.zeros(num_nodes, dtype=np.float64)
    jaccard[valid] = shared_degree[valid] / union_degree[valid]
    return {
        "shared_directed_neighbor_pairs": int(shared.size),
        "left_directed_neighbor_pairs": int(left.size),
        "right_directed_neighbor_pairs": int(right.size),
        "global_jaccard": float(shared.size / max(left.size + right.size - shared.size, 1)),
        "node_jaccard": jaccard,
        "left_degree": left_degree,
        "right_degree": right_degree,
        "shared_degree": shared_degree,
        "valid_nodes": valid,
    }


def _effective_neighbors(operator: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    row_sum_squares = torch.zeros_like(q)
    if operator._nnz():
        row_sum_squares.index_add_(0, operator.indices()[0], operator.values().square())
    result = torch.zeros_like(q)
    valid = q > 0
    result[valid] = q[valid].square() / row_sum_squares[valid].clamp_min(1e-20)
    return result


def _source_graph_diagnostics(dataset: str, model, cache: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_nodes = model.num_nodes
    flags = cache["source_flags"].to(dtype=torch.uint8, device="cpu")
    unique = cache["unique_edge_index"].long().cpu()
    text_present = flags[:, 0] > 0
    visual_present = flags[:, 1] > 0
    categories = {
        "text_only": text_present & ~visual_present,
        "visual_only": visual_present & ~text_present,
        "both_present": text_present & visual_present,
    }
    total = max(int(flags.size(0)), 1)
    for category, selected in categories.items():
        count = int(selected.sum())
        rows.append({
            "dataset": dataset, "family": "edge_source_categories", "metric": category,
            "count": count, "fraction_of_unique_semantic_edges": count / total,
            "notes": "Unique undirected edges; Both means each source count is >0. A raw count of 2 is two proposals from that modality, not joint provenance.",
        })
    for modality_id, modality in enumerate(("text", "visual")):
        for count_value in (0, 1, 2):
            count = int((flags[:, modality_id] == count_value).sum())
            rows.append({
                "dataset": dataset, "family": "raw_source_proposal_counts",
                "view": modality, "metric": f"raw_count_{count_value}", "count": count,
                "fraction_of_unique_semantic_edges": count / total,
                "notes": "Marginal of this modality's raw proposal count; 2 does not mean both modalities proposed the edge.",
            })

    p0, q0 = model._physical_operator, model._physical_quality
    pt, pv, pu = model.semantic_pt, model.semantic_pv, model.semantic_pu
    q_text, q_visual, q_union = model.semantic_q_text, model.semantic_q_visual, model.semantic_q_union
    qualities = {"physical": q0, "text": q_text, "visual": q_visual, "semantic_union": q_union}
    operators = {"physical": p0, "text": pt, "visual": pv, "semantic_union": pu}
    for view, q in qualities.items():
        q_cpu = q.detach().float().cpu()
        _source_row(rows, dataset, "node_quality", "row_quality_q", q_cpu, view=view, unit="operator row mass", notes="Includes zero-quality nodes.")
        _source_row(rows, dataset, "node_coverage", "covered_indicator", (q_cpu > 0).float(), view=view, unit="fraction", notes="Mean is the effective node coverage fraction.")
        if operators[view]._nnz():
            operator = operators[view].coalesce()
            positive_rows = operator.indices()[0][operator.values() > 0]
            support_counts = torch.bincount(positive_rows, minlength=num_nodes)
        else:
            support_counts = torch.zeros(num_nodes, device=q.device, dtype=torch.long)
        _source_row(rows, dataset, "neighbor_counts", "distinct_neighbor_count", support_counts.detach().cpu(), view=view, unit="neighbors/node", notes="Counts positive sparse support entries only; explicit zero weights are excluded; zero degree remains included.")
        _source_row(rows, dataset, "neighbor_counts", "weighted_effective_neighbor_count", _effective_neighbors(operators[view], q).detach().cpu(), view=view, unit="effective neighbors/node", notes="Participation ratio q^2/sum_j w_ij^2 of row-normalized weights; zero degree is 0.")
    q_total = q_text + q_visual
    valid_semantic = q_total > 0
    source_fraction = q_text / q_total.clamp_min(model.eps)
    _source_row(rows, dataset, "node_quality", "text_fraction_of_semantic_quality", source_fraction.detach().cpu(), mask=valid_semantic.detach().cpu(), unit="fraction", notes="Defined only for nodes with at least one semantic source neighbor.")

    from src.models.pace_mag_v1 import canonical_undirected_edges
    physical_unique = canonical_undirected_edges(model._physical_edge_reference, num_nodes)
    text_unique = unique[:, text_present]
    visual_unique = unique[:, visual_present]
    union_unique = unique
    supports = {
        "physical": _directed_codes(physical_unique, num_nodes),
        "text": _directed_codes(text_unique, num_nodes),
        "visual": _directed_codes(visual_unique, num_nodes),
        "semantic_union": _directed_codes(union_unique, num_nodes),
    }
    for left_name, right_name in (("text", "visual"), ("physical", "text"), ("physical", "visual"), ("physical", "semantic_union")):
        overlap = _edge_overlap(supports[left_name], supports[right_name], num_nodes)
        rows.append({
            "dataset": dataset, "family": "global_support_overlap",
            "view": left_name, "modality": right_name, "metric": "directed_neighbor_support_jaccard",
            "count": overlap["shared_directed_neighbor_pairs"], "value": overlap["global_jaccard"],
            "notes": "Global Jaccard over directed receiver-neighbor support pairs; reverse directions are included symmetrically.",
        })
        _source_row(rows, dataset, "node_support_overlap", "neighbor_jaccard", overlap["node_jaccard"],
                    view=left_name, modality=right_name, mask=overlap["valid_nodes"], unit="Jaccard",
                    notes="Per receiver node; nodes with no neighbor in either set are excluded, one-sided support has Jaccard 0.")

    # Operator reconstruction and row-quality mean identities for one fixed H.
    shared_messages: dict[str, dict[str, torch.Tensor]] = {}
    for modality in _MODALITIES:
        h = model.projectors[modality](model._audit_raw_features[modality])
        ut = row_quality_aggregate(pt, h, q_text, model.eps)
        uv = row_quality_aggregate(pv, h, q_visual, model.eps)
        pu_h = torch.sparse.mm(pu, h)
        weighted = q_text.unsqueeze(-1) * ut + q_visual.unsqueeze(-1) * uv
        valid = q_union > 0
        left_mean = weighted / q_union.clamp_min(model.eps).unsqueeze(-1)
        right_mean = pu_h / q_union.clamp_min(model.eps).unsqueeze(-1)
        diff1 = (weighted - pu_h).abs()
        diff2 = (left_mean - right_mean).abs()[valid]
        rows.append({
            "dataset": dataset, "family": "linear_reconstruction", "modality": modality,
            "metric": "max_abs_q_weighted_context_minus_PU_H", "value": float(diff1.max().cpu()) if diff1.numel() else 0.0,
            "notes": "Fixed same-modality H; measured before LayerNorm, experts, and Router.",
        })
        rows.append({
            "dataset": dataset, "family": "linear_reconstruction", "modality": modality,
            "metric": "max_abs_reconstructed_row_mean_minus_PU_row_mean", "value": float(diff2.max().cpu()) if diff2.numel() else 0.0,
            "notes": "Only nodes with q_union>0; checks (qT*UT+qV*UV)/qU = PU@H/qU.",
        })
    from src.models.pace_mag_v1 import build_semantic_operators
    alternate_weight_views = build_semantic_operators(unique, flags, num_nodes, fixed_edge_weight=0.9)
    unit_h = model.projectors["text"](model._audit_raw_features["text"])
    alternate_text_context = row_quality_aggregate(alternate_weight_views["pt"].to(unit_h.device), unit_h, alternate_weight_views["q_text"].to(unit_h.device), model.eps)
    scaled_context = row_quality_aggregate(pt, unit_h, q_text, model.eps)
    logq_change = (torch.log1p(alternate_weight_views["q_union"].to(q_union.device)) - torch.log1p(q_union)).abs()
    rows.append({
        "dataset": dataset, "family": "fixed_edge_weight_scale", "metric": "max_abs_text_context_0_8_vs_0_9",
        "value": float((scaled_context - alternate_text_context).abs().max().cpu()),
        "notes": "A common scalar on a single view cancels in P@H/q, up to floating-point tolerance.",
    })
    rows.append({
        "dataset": dataset, "family": "fixed_edge_weight_scale", "metric": "mean_abs_log1p_q_change_0_8_vs_0_9",
        "value": float(logq_change.mean().cpu()),
        "notes": "Although it cancels from the row mean, changing the common constant changes absolute q and therefore the Router log1p(q) input.",
    })
    return rows


def _context_rows(dataset: str, trace_on, trace_off, q_text: torch.Tensor, q_visual: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    common_valid = (q_text > 0) & (q_visual > 0)
    for modality in _MODALITIES:
        on_text = trace_on.aggregates[("text", modality)]
        on_visual = trace_on.aggregates[("visual", modality)]
        off_text = trace_off.aggregates[("text", modality)]
        off_visual = trace_off.aggregates[("visual", modality)]
        for condition, left, right in (("cross_on", on_text, on_visual), ("cross_off", off_text, off_visual)):
            comp = context_comparison(left, right, common_valid)
            for metric in ("cosine", "normalized_l2"):
                summary = comp[metric]
                rows.append({
                    "dataset": dataset, "family": "source_context_difference", "condition": condition,
                    "modality": modality, "metric": metric, **summary,
                    "notes": "Text-source versus Visual-source U at the same target modality; only nodes with both q>0 are eligible. Cosine excludes zero vectors; normalized L2 divides by max node norm.",
                })
            for source, rms_key in (("text", "left_rms"), ("visual", "right_rms")):
                rows.append({
                    "dataset": dataset, "family": "source_context_rms", "condition": condition,
                    "view": source, "modality": modality, "metric": "aggregate_context_rms",
                    "value": comp[rms_key], "n": comp["eligible_nodes"],
                    "notes": "RMS over nodes eligible in both source views, for paired comparison.",
                })
            rows.append({
                "dataset": dataset, "family": "source_context_difference", "condition": condition,
                "modality": modality, "metric": "difference_rms", "value": comp["difference_rms"],
                "n": comp["eligible_nodes"],
                "notes": "RMS of U_text_source^m-U_visual_source^m over nodes with both source views valid.",
            })
            rows.append({
                "dataset": dataset, "family": "source_context_difference", "condition": condition,
                "modality": modality, "metric": "large_difference_fraction_rel_l2_gt_0_1",
                "value": comp["large_difference_fraction_rel_l2_gt_0_1"], "n": comp["eligible_nodes"],
                "notes": "Large means normalized nodewise L2 >0.1; denominator is max(norm(left),norm(right),eps).",
            })
            rows.append({
                "dataset": dataset, "family": "source_context_difference", "condition": condition,
                "modality": modality, "metric": "small_difference_fraction_rel_l2_le_0_01",
                "value": comp["small_difference_fraction_rel_l2_le_0_01"], "n": comp["eligible_nodes"],
                "notes": "Small means normalized nodewise L2 <=0.01; denominator is max(norm(left),norm(right),eps).",
            })

        injection_ratio = trace_on.messages[("text", modality)] - trace_off.messages[("text", modality)]
        injection_visual = trace_on.messages[("visual", modality)] - trace_off.messages[("visual", modality)]
        for source, on, off, q in (("text", on_text, off_text, q_text), ("visual", on_visual, off_visual, q_visual)):
            delta = on - off
            off_norm = node_norms(off)
            delta_norm = node_norms(delta)
            relative = delta_norm / off_norm.clamp_min(1e-8)
            valid = q > 0
            target_message_delta = injection_ratio if source == "text" else injection_visual
            source_on_rms = float(on.detach().float().square().mean().sqrt().cpu())
            source_off_rms = float(off.detach().float().square().mean().sqrt().cpu())
            delta_rms = float(delta.detach().float().square().mean().sqrt().cpu())
            source_ratio = float((delta_rms / max(source_off_rms, 1e-8)))
            # The message-level ratio is filled below from the corresponding source/target message.
            cross_rows.append({
                "dataset": dataset, "source": source, "target_modality": modality,
                "aggregate_on_rms": source_on_rms, "aggregate_off_rms": source_off_rms,
                "aggregate_delta_rms": delta_rms,
                "aggregate_relative_change_rms": source_ratio,
                "aggregate_node_relative_change_mean": float(relative[valid].mean().cpu()) if bool(valid.any()) else 0.0,
                "aggregate_node_relative_change_q50": float(torch.quantile(relative[valid], 0.5).cpu()) if bool(valid.any()) else 0.0,
                "aggregate_node_relative_change_q90": float(torch.quantile(relative[valid], 0.9).cpu()) if bool(valid.any()) else 0.0,
                "aggregate_node_relative_change_q99": float(torch.quantile(relative[valid], 0.99).cpu()) if bool(valid.any()) else 0.0,
                "aggregation_to_message_change_ratio": None,
                "aggregation_attenuation_ge_50pct": None,
                "valid_nodes": int(valid.sum().item()),
                "notes": "Cross-off is the same E2 checkpoint with only rho*Phi injection set to zero; no retraining.",
            })
    return rows, cross_rows


def _fill_cross_modal_strength(dataset: str, model, trace_on, trace_off, cross_rows: list[dict[str, Any]]) -> None:
    h = {m: model.projectors[m](model._audit_raw_features[m]) for m in _MODALITIES}
    phi = {
        "text": model.cross_modal["visual_to_text"](h["visual"]),
        "visual": model.cross_modal["text_to_visual"](h["text"]),
    }
    source_index = {"text": 0, "visual": 1}
    modality_index = {"text": 0, "visual": 1}
    for row in cross_rows:
        source, modality = row["source"], row["target_modality"]
        rho = float(model.rho[source_index[source], modality_index[modality]].cpu())
        phi_target = phi["text" if modality == "text" else "visual"]
        injection = rho * phi_target
        target = h[modality]
        msg_on = trace_on.messages[(source, modality)]
        valid = (model.semantic_q_text if source == "text" else model.semantic_q_visual) > 0
        node_ratio = node_norms(injection) / node_norms(target).clamp_min(1e-8)
        summary = summarize_values(node_ratio, valid)
        injection_rms = float(injection.detach().float().square().mean().sqrt().cpu())
        target_rms = float(target.detach().float().square().mean().sqrt().cpu())
        message_delta_rms = float((msg_on - target).detach().float().square().mean().sqrt().cpu())
        msg_ratio = message_delta_rms / max(target_rms, 1e-8)
        aggregate_ratio = float(row["aggregate_relative_change_rms"])
        row.update({
            "rho": rho,
            "rho_effective_derivative": rho * (1.0 - rho / model.rho_max),
            "target_h_rms": target_rms,
            "phi_other_rms": float(phi_target.detach().float().square().mean().sqrt().cpu()),
            "rho_phi_rms": injection_rms,
            "rho_phi_to_target_rms_ratio": injection_rms / max(target_rms, 1e-8),
            "send_message_rms": float(msg_on.detach().float().square().mean().sqrt().cpu()),
            "send_message_change_rms": message_delta_rms,
            "send_message_relative_change_rms": msg_ratio,
            "send_node_relative_change_mean": summary["mean"],
            "send_node_relative_change_q10": summary["q10"],
            "send_node_relative_change_q50": summary["median"],
            "send_node_relative_change_q90": summary["q90"],
            "send_node_relative_change_q99": summary["q99"],
            "send_node_relative_change_max": summary["max"],
            "aggregation_to_message_change_ratio": aggregate_ratio / max(msg_ratio, 1e-8),
            "aggregation_attenuation_ge_50pct": bool(aggregate_ratio / max(msg_ratio, 1e-8) < 0.5),
        })


def _router_rows(dataset: str, trace, qualities: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for relation in ("physical", "text", "visual"):
        valid = qualities[relation] > 0
        for modality in _MODALITIES:
            route = trace.routes[(relation, modality)].float()
            selected = route[valid]
            distributions[(relation, modality)] = (route, valid)
            if selected.numel():
                mean = selected.mean(dim=0)
                std = selected.std(dim=0, unbiased=False)
                entropy = -(selected.clamp_min(1e-8) * selected.clamp_min(1e-8).log()).sum(dim=-1)
                top1 = torch.bincount(selected.argmax(dim=-1), minlength=3).float() / selected.size(0)
            else:
                mean = std = entropy = route.new_zeros((3,))
                top1 = route.new_zeros((3,))
            rows.append({
                "dataset": dataset, "kind": "view_summary", "relation": relation, "modality": modality,
                "valid_nodes": int(valid.sum().item()),
                "mean_p0": float(mean[0].cpu()), "mean_p1": float(mean[1].cpu()), "mean_p2": float(mean[2].cpu()),
                "node_std_p0": float(std[0].cpu()), "node_std_p1": float(std[1].cpu()), "node_std_p2": float(std[2].cpu()),
                "entropy_mean_nats": float(entropy.mean().cpu()) if entropy.numel() else 0.0,
                "top1_fraction_e0": float(top1[0].cpu()), "top1_fraction_e1": float(top1[1].cpu()), "top1_fraction_e2": float(top1[2].cpu()),
                "notes": "Router probabilities are original E2 per-node values; invalid source rows are excluded and have probability 0.",
            })
    for modality in _MODALITIES:
        for left, right in (("text", "visual"), ("physical", "text"), ("physical", "visual")):
            lroute, lvalid = distributions[(left, modality)]
            rroute, rvalid = distributions[(right, modality)]
            common = lvalid & rvalid
            js = jensen_shannon(lroute, rroute)
            rows.append({
                "dataset": dataset, "kind": "same_node_view_comparison", "relation": f"{left}_vs_{right}",
                "modality": modality, "valid_nodes": int(common.sum().item()),
                "mean_route_js_nats": float(js[common].mean().cpu()) if bool(common.any()) else None,
                "mean_route_l1": float((lroute[common] - rroute[common]).abs().sum(dim=-1).mean().cpu()) if bool(common.any()) else None,
                "notes": "Same-node comparison restricted to nodes valid in both views; JS is in nats.",
            })
    for relation in ("physical", "text", "visual"):
        left, lvalid = distributions[(relation, "text")]
        right, rvalid = distributions[(relation, "visual")]
        common = lvalid & rvalid
        js = jensen_shannon(left, right)
        rows.append({
            "dataset": dataset, "kind": "same_node_modality_comparison", "relation": relation,
            "modality": "text_vs_visual", "valid_nodes": int(common.sum().item()),
            "mean_route_js_nats": float(js[common].mean().cpu()) if bool(common.any()) else None,
            "mean_route_l1": float((left[common] - right[common]).abs().sum(dim=-1).mean().cpu()) if bool(common.any()) else None,
            "notes": "Same-node comparison across target modalities, restricted to nodes valid in both.",
        })
    return rows


def _expert_rows(dataset: str, trace) -> list[dict[str, Any]]:
    rows = []
    for summary in trace.expert_statistics:
        message = max(float(summary["message_rms"]), 1e-8)
        cos = summary["expert_pairwise_cosine"]
        rows.append({
            "dataset": dataset,
            **summary,
            "expert0_expert1_cosine": cos[0] if len(cos) > 0 else None,
            "expert0_expert2_cosine": cos[1] if len(cos) > 1 else None,
            "expert1_expert2_cosine": cos[2] if len(cos) > 2 else None,
            "weighted_response_to_message_rms_ratio": summary["weighted_response_rms"] / message,
            "notes": "Output direction differences do not establish task-functional complementarity.",
        })
    return rows


def _parameter_tensor_audit(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    array = tensor.float().numpy()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return {
        "shape": list(tensor.shape),
        "parameter_count": int(tensor.numel()),
        "sha256": digest.hexdigest(),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "l2_norm": float(np.linalg.norm(array)) if array.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0,
    }


def _trained_parameter_audit(dataset: str, model) -> dict[str, Any]:
    rho = model.rho.detach().float().cpu()
    rho_raw = model.rho_raw.detach().float().cpu()
    beta_phy = float(model.beta_phy.detach().cpu())
    beta_sem = float(model.beta_sem.detach().cpu())
    beta_phy_raw = float(model.beta_phy_raw.detach().cpu())
    beta_sem_raw = float(model.beta_sem_raw.detach().cpu())
    router_tensors = []
    for scope, module in (
        ("router", model.router),
        ("relation_embedding", model.relation_embedding),
        ("modality_embedding", model.modality_embedding),
    ):
        for name, parameter in module.named_parameters():
            router_tensors.append({
                "name": f"{scope}.{name}",
                **_parameter_tensor_audit(parameter),
            })
    return {
        "dataset": dataset,
        "checkpoint_seed": 42,
        "semantic_edge_weight_constant": model._config_semantic_edge_weight,
        "bounded_scalars": {
            "rho": {
                "source_rows": ["text", "visual"], "target_columns": ["text", "visual"],
                "raw": rho_raw.tolist(), "effective": rho.tolist(),
                "effective_derivative_wrt_raw": (rho * (1.0 - rho / model.rho_max)).tolist(),
                "maximum": model.rho_max,
            },
            "beta_phy": {
                "raw": beta_phy_raw, "effective": beta_phy,
                "effective_derivative_wrt_raw": beta_phy * (1.0 - beta_phy / model.beta_max),
                "maximum": model.beta_max,
            },
            "beta_sem": {
                "raw": beta_sem_raw, "effective": beta_sem,
                "effective_derivative_wrt_raw": beta_sem * (1.0 - beta_sem / model.beta_max),
                "maximum": model.beta_max,
            },
            "gamma_fixed": model.gamma,
        },
        "router": {
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.router.parameters()),
            "conditioning_embedding_parameter_count": sum(
                parameter.numel() for parameter in model.relation_embedding.parameters()
            ) + sum(parameter.numel() for parameter in model.modality_embedding.parameters()),
            "trained_parameter_tensors": router_tensors,
            "summary_note": "Per-tensor summaries and SHA-256 bind these measurements to the frozen checkpoint without duplicating checkpoint weights.",
        },
    }


def _intervention_row(dataset: str, name: str, result, baseline, metric, restore, *, mode: str, seed: int) -> dict[str, Any]:
    z_delta = result.embedding - baseline.embedding
    logit_delta = result.logits - baseline.logits
    base_metrics = metric
    current, _class_rows = current_metrics = validation_metrics(
        result.logits, base_metrics["_labels"], base_metrics["_train_idx"], base_metrics["_val_idx"], base_metrics["_num_classes"]
    )
    valid_idx = base_metrics["_val_idx"].to(result.logits.device)
    flip = (result.logits[valid_idx].argmax(dim=-1) != baseline.logits[valid_idx].argmax(dim=-1)).float().mean()
    return {
        "dataset": dataset,
        "intervention": name,
        "route_mode": mode,
        "permutation_seed": seed if mode == "node_permuted" else "",
        "val_ce": current["val_ce"], "val_acc": current["val_acc"], "val_macro_f1": current["val_macro_f1"],
        "embedding_delta_rms": float(z_delta.detach().float().square().mean().sqrt().cpu()),
        "embedding_rms": float(result.embedding.detach().float().square().mean().sqrt().cpu()),
        "embedding_relative_delta_rms": float(z_delta.detach().float().square().mean().sqrt().cpu() / baseline.embedding.detach().float().square().mean().sqrt().clamp_min(1e-8).cpu()),
        "logit_delta_rms": float(logit_delta.detach().float().square().mean().sqrt().cpu()),
        "logit_rms": float(result.logits.detach().float().square().mean().sqrt().cpu()),
        "logit_relative_delta_rms": float(logit_delta.detach().float().square().mean().sqrt().cpu() / baseline.logits.detach().float().square().mean().sqrt().clamp_min(1e-8).cpu()),
        "validation_prediction_flip_fraction": float(flip.cpu()),
        "delta_val_ce": current["val_ce"] - metric["val_ce"],
        "delta_val_acc": current["val_acc"] - metric["val_acc"],
        "delta_val_macro_f1": current["val_macro_f1"] - metric["val_macro_f1"],
        "restore_max_abs_embedding_difference": float((restore.embedding - baseline.embedding).abs().max().cpu()),
        "restore_max_abs_logit_difference": float((restore.logits - baseline.logits).abs().max().cpu()),
        "validation_labels_only": True,
        "test_labels_accessed": False,
        "test_metrics_computed": False,
        "notes": "Frozen-checkpoint intervention; no weights, optimizer, norm state, or training mode changed.",
    }


def _initialization_audit(dataset: str, cfg_e2, cfg_e3, data_info: dict[str, int], seed: int = 42) -> dict[str, Any]:
    from src.models import build_model

    common = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        e2 = build_model(cfg_e2, data_info)
        torch.manual_seed(seed)
        e3 = build_model(cfg_e3, data_info)
    p2 = dict(e2.named_parameters())
    p3 = dict(e3.named_parameters())
    rows = []
    for name in sorted(p2.keys() & p3.keys()):
        left, right = p2[name].detach().cpu(), p3[name].detach().cpu()
        delta = (left - right).abs()
        root = name.split(".", 1)[0]
        if root in {"fusion", "skip"}:
            group = "post-router randomized common module"
        elif root in {"message_norm", "aggregate_norm", "mix_norm", "physical_norm", "fusion_norm"}:
            group = "LayerNorm common module"
        elif root in {"projectors", "cross_modal", "rho_raw"}:
            group = "pre-expert common module"
        elif root in {"beta_phy_raw", "beta_sem_raw", "gamma"}:
            group = "scalar common module"
        else:
            group = "other common module"
        rows.append({
            "name": name, "group": group, "parameter_count": int(left.numel()),
            "exact_equal": bool(torch.equal(left, right)), "max_abs_difference": float(delta.max()) if delta.numel() else 0.0,
        })
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = grouped.setdefault(row["group"], {"tensor_count": 0, "parameter_count": 0, "exact_equal_tensors": 0, "different_tensors": 0, "max_abs_difference": 0.0})
        target["tensor_count"] += 1
        target["parameter_count"] += row["parameter_count"]
        target["exact_equal_tensors"] += int(row["exact_equal"])
        target["different_tensors"] += int(not row["exact_equal"])
        target["max_abs_difference"] = max(target["max_abs_difference"], row["max_abs_difference"])

    e2_expert = sum(p.numel() for p in e2.experts.parameters())
    e2_router = sum(p.numel() for p in e2.router.parameters()) + sum(p.numel() for p in e2.relation_embedding.parameters()) + sum(p.numel() for p in e2.modality_embedding.parameters())
    e3_single = sum(p.numel() for p in e3.single_expert.parameters())
    d, b, k, rh, re, me = e2.hidden_dim, e2.expert_bottleneck, e2.num_experts, e2.router_hidden_dim, e2.relation_embedding_dim, e2.modality_embedding_dim
    expert_mac_per_call = (4 * d) * b + b * d
    router_in = 4 * d + re + me + 1
    router_mac_per_call = router_in * rh + rh * k
    e2_mac_per_node = 6 * (k * expert_mac_per_call + router_mac_per_call)
    e3_mac_per_node = 6 * ((4 * d) * e3.single_expert_bottleneck + e3.single_expert_bottleneck * d)
    common.update({
        "dataset": dataset,
        "seed": seed,
        "construction": "CPU model construction only; no checkpoint weights loaded and no training/optimizer step",
        "common_parameter_tensors": rows,
        "group_summary": grouped,
        "e2_processing_parameters": {"experts": e2_expert, "router_and_embeddings": e2_router, "total": e2_expert + e2_router},
        "e3_single_processing_parameters": e3_single,
        "e3_parameter_difference": e3_single - (e2_expert + e2_router),
        "e2_approx_linear_macs_per_node_for_six_view_modality_calls": e2_mac_per_node,
        "e3_approx_linear_macs_per_node_for_six_view_modality_calls": e3_mac_per_node,
        "e3_relative_mac_difference": (e3_mac_per_node - e2_mac_per_node) / e2_mac_per_node,
        "cost_scope_note": "Counts multiply-accumulates in expert and router Linear layers only; excludes activations, sparse propagation, normalization, memory traffic, and any hardware-dependent dispatch cost.",
    })
    return common


def _training_log_review(dataset: str, checkpoint: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import re

    pattern = re.compile(r"Epoch (\d+) \| Train Loss ([\d.]+) \| Val Acc ([\d.]+) \| Val Macro-F1 ([\d.]+)")
    records = []
    for line in (run_dir / "train.log").read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            records.append({"epoch": int(match.group(1)), "train_loss": float(match.group(2)), "val_acc_pct": float(match.group(3)), "val_macro_f1_pct": float(match.group(4))})
    best = max(records, key=lambda row: row["val_acc_pct"])
    last = records[-1]
    return {
        "dataset": dataset,
        "epochs_logged": len(records),
        "checkpoint_best_epoch": int(checkpoint["epoch"]),
        "logged_best_accuracy_epoch": best["epoch"],
        "first_epoch": records[0], "best_accuracy_epoch": best, "last_epoch": last,
        "minimum_train_loss": min(row["train_loss"] for row in records),
        "early_stop_line": next((line for line in reversed((run_dir / "train.log").read_text(encoding="utf-8").splitlines()) if "Early stopping" in line), None),
        "interpretation_limit": "The log contains train loss and validation metrics, but no train Accuracy/Macro-F1, per-epoch gradients, optimizer state, or intermediate weights.",
    }


def _checkpoint_and_data(dataset: str, device: torch.device) -> dict[str, Any]:
    from scripts import run_pace_mag_v1 as pilot
    from src.data import load_mag_data
    from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b
    from src.models import build_model

    run_dir = PILOT_RUNS / dataset / "seed_42" / "e2_full"
    checkpoint_path = run_dir / "best.pt"
    config_path = run_dir / "resolved_config.yaml"
    if not checkpoint_path.is_file() or not config_path.is_file() or not (run_dir / "train.log").is_file():
        raise FileNotFoundError(f"required E2 checkpoint/config/log missing under {run_dir}")
    pilot_row = next(row for row in _read_csv(PILOT_DIR / "pilot_runs.csv") if row["dataset"] == dataset and row["variant"] == "e2_full")
    if pilot_row["status"] != "completed" or pilot_row["commit_sha"] != FREEZE_SHA or pilot_row["evaluate_test"] != "False":
        raise ValueError(f"{dataset}: source Pilot row is failed, Test-enabled, or from an unexpected code commit")
    checkpoint_hash = sha256_file(checkpoint_path)
    config_hash = sha256_file(config_path)
    if config_hash != pilot_row["config_sha256"] or pilot_row["config_path"] != str(config_path.relative_to(ROOT)):
        raise ValueError(f"{dataset}: resolved config SHA does not match the frozen Pilot CSV")
    cfg = OmegaConf.load(config_path)
    if str(cfg.dataset.name) != dataset or str(cfg.model.variant) != "e2_full" or bool(cfg.task.evaluate_test):
        raise ValueError(f"{dataset}: saved config identity or Test-disabled setting does not match E2 Pilot")
    if str(cfg.task.protocol_version) != "unified_full_graph_nc_v1" or int(cfg.seed) != 42:
        raise ValueError(f"{dataset}: saved config protocol/seed mismatch")

    # This loader materializes the full y tensor. The audit records that fact,
    # then indexes only train/validation labels and never touches test_idx.
    data = load_mag_data(cfg, "nc", 42)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "nc" or checkpoint.get("protocol_version") != "unified_full_graph_nc_v1" or checkpoint.get("seed") != 42 or checkpoint.get("selection") != "best_val_accuracy":
        raise ValueError(f"{dataset}: checkpoint metadata does not identify the requested best-validation E2 checkpoint")
    if checkpoint.get("data_info") != info or int(checkpoint.get("epoch", -1)) != int(pilot_row["best_epoch"]):
        raise ValueError(f"{dataset}: checkpoint dataset shape or best epoch does not match the Pilot record")
    expected_metrics = {"val_acc": float(pilot_row["val_acc"]), "val_macro_f1": float(pilot_row["val_macro_f1"])}
    if set(checkpoint.get("metrics", {})) != set(expected_metrics) or any(abs(float(checkpoint["metrics"][key]) - value) > 1e-12 for key, value in expected_metrics.items()):
        raise ValueError(f"{dataset}: checkpoint validation metrics do not match the original Pilot CSV")
    if checkpoint.get("run_metadata", {}).get("optimizer") != "AdamW":
        raise ValueError(f"{dataset}: expected the frozen AdamW Pilot checkpoint")
    run_meta = checkpoint["run_metadata"]
    if int(run_meta.get("model_parameters", -1)) != int(pilot_row["model_parameters"]) or int(run_meta.get("classifier_parameters", -1)) != int(pilot_row["classifier_parameters"]):
        raise ValueError(f"{dataset}: checkpoint parameter counts do not match the Pilot record")

    entry = pilot._cache_entries()[dataset]
    verified = pilot._verify_cache_entry(dataset, entry, save_audit=False)
    cache_path = Path(verified["cache_path_abs"])
    cache = load_semantic_candidate_cache_v7b(
        cache_path,
        expected_dataset=dataset,
        expected_feature_fingerprint=pilot_row["semantic_feature_fingerprint"],
        expected_candidate_fingerprint=pilot_row["semantic_candidate_fingerprint"],
        expected_v7b_fingerprint=pilot_row["semantic_v7b_fingerprint"],
    )
    for key, cache_key in (("semantic_feature_fingerprint", "feature"), ("semantic_candidate_fingerprint", "candidate"), ("semantic_v7b_fingerprint", "v7b")):
        if run_meta.get(key) != pilot_row[f"semantic_{cache_key}_fingerprint"]:
            raise ValueError(f"{dataset}: checkpoint {key} does not match the frozen Pilot/cache provenance")
    model = build_model(cfg, info)
    model.load_semantic_cache(
        str(cache_path), expected_dataset=dataset,
        expected_feature_fingerprint=pilot_row["semantic_feature_fingerprint"],
        expected_candidate_fingerprint=pilot_row["semantic_candidate_fingerprint"],
        expected_v7b_fingerprint=pilot_row["semantic_v7b_fingerprint"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes))
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    model = model.to(device).eval()
    classifier = classifier.to(device).eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    model._audit_raw_features = {
        "text": x[:, : model.text_dim],
        "visual": x[:, model.text_dim : model.text_dim + model.visual_dim],
    }
    metadata = {
        "dataset": dataset,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": config_hash,
        "cache_path": str(cache_path.relative_to(ROOT)),
        "cache_fingerprints": {
            "feature": cache["metadata"]["feature_fingerprint"],
            "candidate": cache["metadata"]["candidate_fingerprint"],
            "v7b": cache["metadata"]["v7b_fingerprint"],
        },
        "data_info": info,
        "labels_loaded_in_memory": True,
        "labels_accessed_for_audit": ["train_idx", "val_idx"],
        "test_label_values_accessed": False,
        "test_metrics_computed": False,
        "split_path": data.info.get("nc_split_path", data.info.get("node_split_path", "")),
        "checkpoint_metrics": checkpoint["metrics"],
        "pilot_csv_metrics": expected_metrics,
    }
    return {"cfg": cfg, "data": data, "model": model, "classifier": classifier, "x": x, "edge_index": edge_index,
            "checkpoint": checkpoint, "cache": cache, "metadata": metadata, "run_dir": run_dir}


def audit_dataset(dataset: str, device: torch.device, min_free_bytes: int) -> dict[str, Any]:
    free, _total = torch.cuda.mem_get_info(device)
    if free < min_free_bytes:
        raise RuntimeError(f"{device}: only {free} free bytes; audit requires {min_free_bytes}; no process was stopped")
    started = time.perf_counter()
    device_before = _cuda_snapshot(device)
    loaded = _checkpoint_and_data(dataset, device)
    data, model, classifier = loaded["data"], loaded["model"], loaded["classifier"]
    x, edge_index = loaded["x"], loaded["edge_index"]
    if model.training or classifier.training:
        raise RuntimeError("model/classifier failed to remain in eval mode")
    state_before = tensor_sha256((model, classifier))
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        original_z = model(x, edge_index)[0]
        original_logits = classifier(original_z)
        original_metrics, original_class_rows = validation_metrics(
            original_logits, data.y, data.train_idx, data.val_idx, int(data.num_classes)
        )
        baseline = audit_forward(model, x, edge_index, classifier, capture=True)
        parity = {
            **loaded["metadata"],
            "original_forward_audit_forward_max_abs_embedding_difference": float((original_z - baseline.embedding).abs().max().cpu()),
            "original_forward_audit_forward_max_abs_logit_difference": float((original_logits - baseline.logits).abs().max().cpu()),
            "val_acc_reproduced": original_metrics["val_acc"],
            "val_macro_f1_reproduced": original_metrics["val_macro_f1"],
            "val_ce_reproduced": original_metrics["val_ce"],
            "val_acc_difference_from_pilot": original_metrics["val_acc"] - float(loaded["checkpoint"]["metrics"]["val_acc"]),
            "val_macro_f1_difference_from_pilot": original_metrics["val_macro_f1"] - float(loaded["checkpoint"]["metrics"]["val_macro_f1"]),
            "parity_tolerance": TOL,
        }
        parity["passed"] = bool(
            parity["original_forward_audit_forward_max_abs_embedding_difference"] <= TOL
            and parity["original_forward_audit_forward_max_abs_logit_difference"] <= TOL
            and abs(parity["val_acc_difference_from_pilot"]) <= TOL
            and abs(parity["val_macro_f1_difference_from_pilot"]) <= TOL
        )
        if not parity["passed"]:
            raise RuntimeError(f"{dataset}: original/Audit forward or Validation metric parity failed; interventions were not run")

        # Keep only scalar baseline metrics plus route summaries after the source
        # context comparisons; full node tensors are not written to disk.
        baseline_metrics, baseline_class_rows = original_metrics, original_class_rows
        q = {"physical": model._physical_quality, "text": model.semantic_q_text, "visual": model.semantic_q_visual}
        source_rows = _source_graph_diagnostics(dataset, model, loaded["cache"])
        trace_off = audit_forward(model, x, edge_index, classifier, cross_modal=False, capture=True)
        context_rows, cross_rows = _context_rows(dataset, baseline, trace_off, q["text"], q["visual"])
        _fill_cross_modal_strength(dataset, model, baseline, trace_off, cross_rows)
        source_rows.extend(context_rows)
        router_rows = _router_rows(dataset, baseline, q)
        expert_rows = _expert_rows(dataset, baseline)
        branch_rows = []
        for name in ("physical", "semantic", "intrinsic_skip"):
            key_in = {"physical": "physical_response_rms", "semantic": "semantic_response_rms", "intrinsic_skip": "intrinsic_skip_input_rms"}[name]
            key_weighted = {"physical": "physical_weighted_rms", "semantic": "semantic_weighted_rms", "intrinsic_skip": "intrinsic_skip_projected_rms"}[name]
            branch_rows.append({"dataset": dataset, "branch": name, "unweighted_input_rms": baseline.branch_statistics[key_in],
                                "weighted_or_projected_input_rms": baseline.branch_statistics[key_weighted],
                                "fusion_projected_rms": baseline.branch_statistics["fusion_projected_rms"],
                                "final_embedding_rms": baseline.branch_statistics["final_embedding_rms"],
                                "notes": "These amplitudes precede final LayerNorm and must be paired with frozen branch-off logit results."})

        intervention_specs = [
            ("R1_uniform_routing", {"router_mode": "uniform"}, "uniform"),
            ("R2_view_mean_routing", {"router_mode": "view_mean"}, "view_mean"),
            ("R3_node_permuted_routing", {"router_mode": "node_permuted"}, "node_permuted"),
            ("cross_modal_off", {"cross_modal": False}, "original"),
            ("semantic_response_off", {"semantic_response": False}, "original"),
            ("physical_response_off", {"physical_response": False}, "original"),
            ("intrinsic_skip_off", {"intrinsic_skip": False}, "original"),
        ]
        interventions = [{
            "dataset": dataset, "intervention": "R0_original", "route_mode": "original", "permutation_seed": "",
            "val_ce": baseline_metrics["val_ce"], "val_acc": baseline_metrics["val_acc"], "val_macro_f1": baseline_metrics["val_macro_f1"],
            "embedding_delta_rms": 0.0, "embedding_rms": float(baseline.embedding.float().square().mean().sqrt().cpu()), "embedding_relative_delta_rms": 0.0,
            "logit_delta_rms": 0.0, "logit_rms": float(baseline.logits.float().square().mean().sqrt().cpu()), "logit_relative_delta_rms": 0.0,
            "validation_prediction_flip_fraction": 0.0, "delta_val_ce": 0.0, "delta_val_acc": 0.0, "delta_val_macro_f1": 0.0,
            "restore_max_abs_embedding_difference": 0.0, "restore_max_abs_logit_difference": 0.0,
            "validation_labels_only": True, "test_labels_accessed": False, "test_metrics_computed": False,
            "notes": "R0 is the original frozen E2 model and classifier.",
        }]
        class_rows = []
        for item in baseline_class_rows:
            class_rows.append({"dataset": dataset, "intervention": "R0_original", **item})
        for name, kwargs, route_mode in intervention_specs:
            result = audit_forward(
                model, x, edge_index, classifier, capture=False,
                router_seed=AUDIT_PERMUTATION_SEED, **kwargs,
            )
            restored_z = model(x, edge_index)[0]
            restored_logits = classifier(restored_z)
            restore = type(baseline)(restored_z, restored_logits, {}, {}, {}, {}, [], {})
            metric_context = {
                "val_ce": baseline_metrics["val_ce"],
                "val_acc": baseline_metrics["val_acc"],
                "val_macro_f1": baseline_metrics["val_macro_f1"],
                "_labels": data.y, "_train_idx": data.train_idx, "_val_idx": data.val_idx, "_num_classes": int(data.num_classes),
            }
            intervention_row = _intervention_row(dataset, name, result, baseline, metric_context, restore, mode=route_mode, seed=AUDIT_PERMUTATION_SEED)
            if intervention_row["restore_max_abs_embedding_difference"] > TOL or intervention_row["restore_max_abs_logit_difference"] > TOL:
                raise RuntimeError(f"{dataset}/{name}: the original forward did not restore within tolerance")
            interventions.append(intervention_row)
            if name.endswith("response_off") or name == "intrinsic_skip_off":
                branch_name = {
                    "physical_response_off": "physical",
                    "semantic_response_off": "semantic",
                    "intrinsic_skip_off": "intrinsic_skip",
                }[name]
                branch_row = next(item for item in branch_rows if item["branch"] == branch_name)
                branch_row.update({
                    "off_embedding_delta_rms": intervention_row["embedding_delta_rms"],
                    "off_logit_delta_rms": intervention_row["logit_delta_rms"],
                    "off_val_ce_delta": intervention_row["delta_val_ce"],
                    "off_val_acc_delta": intervention_row["delta_val_acc"],
                    "off_val_macro_f1_delta": intervention_row["delta_val_macro_f1"],
                    "off_prediction_flip_fraction": intervention_row["validation_prediction_flip_fraction"],
                })
            current_metrics, current_class_rows = validation_metrics(
                result.logits, data.y, data.train_idx, data.val_idx, int(data.num_classes)
            )
            for item in current_class_rows:
                class_rows.append({"dataset": dataset, "intervention": name, **item})

        state_after = tensor_sha256((model, classifier))
        if state_after != state_before:
            raise RuntimeError(f"{dataset}: checkpoint parameters/classifier state changed during frozen audit")
        if model.training or classifier.training:
            raise RuntimeError(f"{dataset}: model or classifier left eval mode during audit")

        # Compare deterministic E2/E3 parameter initialization for the exact
        # dataset dimensions and seed; never load these models into training.
        pilot_rows = _read_csv(PILOT_DIR / "pilot_runs.csv")
        config_e3 = Path(next(r["config_path"] for r in pilot_rows if r["dataset"] == dataset and r["variant"] == "e3_single_expert_control"))
        cfg_e3 = OmegaConf.load(ROOT / config_e3)
        init = _initialization_audit(dataset, loaded["cfg"], cfg_e3, parity["data_info"], seed=42)
        parity["state_sha256_before_and_after_equal"] = state_before == state_after
        parity["model_eval_state_preserved"] = not model.training and not classifier.training
        parity["model_source_commit"] = FREEZE_SHA
        parity["pilot_report_commit"] = PILOT_SHA

        peak = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        result = {
            "dataset": dataset,
            "metadata": loaded["metadata"],
            "forward_parity": parity,
            "source_context_rows": source_rows,
            "cross_modal_rows": cross_rows,
            "router_rows": router_rows,
            "expert_rows": expert_rows,
            "branch_rows": branch_rows,
            "intervention_rows": interventions,
            "class_rows": class_rows,
            "initialization": init,
            "trained_parameters": _trained_parameter_audit(dataset, model),
            "optimization_log_review": _training_log_review(dataset, loaded["checkpoint"], loaded["run_dir"]),
            "resources": {"device_before": device_before, "device_after": _cuda_snapshot(device), "peak_allocated_bytes": peak, "peak_reserved_bytes": peak_reserved, "elapsed_seconds": time.perf_counter() - started},
        }
        # Drop all graph-sized audit tensors before loading the next dataset.
        del baseline, trace_off, original_z, original_logits
        del x, edge_index
        return result


def _optimization_evidence(results: list[dict[str, Any]]) -> dict[str, Any]:
    pilot = _read_csv(PILOT_DIR / "pilot_runs.csv")
    gradient_audit = json.loads((PILOT_DIR / "gradient_audit.json").read_text(encoding="utf-8"))
    e2_rows = []
    for row in pilot:
        if row["variant"] != "e2_full":
            continue
        groups = json.loads(row["gradient_diagnostics"])
        reduced = {}
        for key in ("cross_modal", "rho_raw", "beta_phy_raw", "beta_sem_raw", "experts", "router", "fusion", "skip"):
            item = groups.get(key, {})
            n = int(item.get("parameter_count", 0))
            norm = float(item.get("gradient_l2_norm", 0.0))
            reduced[key] = {
                "parameter_count": n,
                "gradient_l2_norm": norm,
                "gradient_rms_from_l2": norm / math.sqrt(n) if n else None,
                "gradient_present": item.get("gradient_present"),
                "gradient_finite": item.get("gradient_finite"),
            }
        e2_rows.append({"dataset": row["dataset"], "gradient_timing": "epoch 1, before global norm clipping", "groups": reduced})
    return {
        "trained_checkpoint_parameters": [result["trained_parameters"] for result in results],
        "pilot_e2_epoch1_gradient_summaries": e2_rows,
        "real_graph_preflight_gradient_and_update_audit": gradient_audit,
        "training_logs": [
            _training_log_review(ds, torch.load(PILOT_RUNS / ds / "seed_42/e2_full/best.pt", map_location="cpu", weights_only=False), PILOT_RUNS / ds / "seed_42/e2_full")
            for ds in DATASETS
        ],
        "rho_parameterization": {
            "form": "rho = rho_max * sigmoid(rho_raw)",
            "rho_max": 0.25,
            "pilot_initial_rho": 0.1,
            "effective_derivative_at_initial": 0.1 * (1.0 - 0.1 / 0.25),
            "interpretation": "The initial derivative is about 0.06; use each frozen checkpoint's measured rho and derivative to assess saturation. A small raw-gradient L2 alone cannot establish an optimization defect; E2 records are first-epoch summaries and AdamW's normalized update also depends on its moment state.",
        },
        "cross_modal_initialization": {
            "module": "CrossModalBottleneck.second.weight",
            "initialization": "Normal(0, 0.02); bias zero",
            "interpretation": "The output layer is deliberately small and is multiplied by rho. Whether that combination suppresses messages must be judged from measured Phi/injection/context ratios and frozen interventions, not initialization alone.",
        },
        "trajectory_limit": "No per-epoch gradients, parameter checkpoints, optimizer state, or training Accuracy are stored; only epoch-1 gradient summaries, logs, and each selected best Validation checkpoint are available.",
        "gradient_norm_limit": "L2 norms across groups have different parameter counts and scales. RMS-from-L2 is supplied descriptively; neither is an importance ranking or a direct comparison of AdamW update magnitude.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("frozen PACE-MAG audit requires a CUDA device")
    if device.index is None or device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device is unavailable: {device}")
    if _git("rev-parse", "HEAD^") != PILOT_SHA:
        raise RuntimeError(f"audit must run from an Instrumentation commit whose direct parent is {PILOT_SHA}; current parent={_git('rev-parse','HEAD^')}")
    if _git("branch", "--show-current") != "exp/pace_mag_v1_mechanism_audit":
        raise RuntimeError("audit must run on exp/pace_mag_v1_mechanism_audit")
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit evidence at {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    environment = _runtime_environment(device)
    min_free = int(args.min_free_gib * 1024**3)
    results = []
    failures = []
    for dataset in args.datasets:
        try:
            print(f"[audit] {dataset} on {device}", flush=True)
            results.append(audit_dataset(dataset, device, min_free))
            print(f"[audit] {dataset} parity and interventions complete", flush=True)
        except Exception as exc:
            failures.append({"dataset": dataset, "error_type": type(exc).__name__, "error": str(exc)})
            _write_json(DATA_DIR / "failures.json", failures)
            # A failed reproduction invalidates interpretation for this run;
            # stop before another dataset can obscure the cause.
            raise

    source_rows = [row for result in results for row in result["source_context_rows"]]
    cross_rows = [row for result in results for row in result["cross_modal_rows"]]
    router_rows = [row for result in results for row in result["router_rows"]]
    expert_rows = [row for result in results for row in result["expert_rows"]]
    branch_rows = [row for result in results for row in result["branch_rows"]]
    intervention_rows = [row for result in results for row in result["intervention_rows"]]
    class_rows = [row for result in results for row in result["class_rows"]]
    _write_csv(DATA_DIR / "source_context_stats.csv", source_rows)
    _write_csv(DATA_DIR / "cross_modal_strength.csv", cross_rows)
    _write_csv(DATA_DIR / "router_diagnostics.csv", router_rows)
    _write_csv(DATA_DIR / "expert_diagnostics.csv", expert_rows)
    _write_csv(DATA_DIR / "frozen_interventions.csv", intervention_rows)
    _write_csv(DATA_DIR / "branch_contributions.csv", branch_rows)
    _write_csv(DATA_DIR / "validation_class_metrics.csv", class_rows)
    _write_json(DATA_DIR / "forward_parity.json", {r["dataset"]: r["forward_parity"] for r in results})
    _write_json(DATA_DIR / "initialization_audit.json", {r["dataset"]: r["initialization"] for r in results})
    _write_json(DATA_DIR / "optimization_evidence.json", _optimization_evidence(results))
    _write_json(DATA_DIR / "failures.json", failures)
    _write_json(DATA_DIR / "audit_manifest.json", {
        "audit_branch": _git("branch", "--show-current"),
        "instrumentation_commit": _git("rev-parse", "HEAD"),
        "required_v1_freeze_commit": FREEZE_SHA,
        "required_v1_pilot_report_commit": PILOT_SHA,
        "datasets": [r["dataset"] for r in results],
        "checkpoint_hashes": {r["dataset"]: r["metadata"]["checkpoint_sha256"] for r in results},
        "selected_device": _cuda_snapshot(device),
        "environment": environment,
        "router_permutation_seed": AUDIT_PERMUTATION_SEED,
        "inference_mode": "torch.inference_mode",
        "optimizer_created_or_called": False,
        "training_performed": False,
        "test_labels_accessed": False,
        "test_metrics_computed": False,
        "label_tensor_note": "The repository MAG NC loader materializes its full label tensor. Audit metrics index train/validation labels only; Test label values and test_idx are not accessed.",
        "failures": failures,
    })
    print(f"Audit evidence written for {len(results)}/{len(args.datasets)} datasets; failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
