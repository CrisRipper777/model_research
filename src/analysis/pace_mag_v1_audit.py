"""Frozen-checkpoint instrumentation for PACE-MAG V1.

This module calls the V1 modules and operators without editing their weights or
definition.  It provides an audit-only forward with explicit, local switches
for the frozen interventions requested in the mechanism audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.models.pace_mag_v1 import _MODALITIES, _RELATIONS, row_quality_aggregate


@dataclass
class AuditForward:
    embedding: torch.Tensor
    logits: torch.Tensor | None
    aggregates: dict[tuple[str, str], torch.Tensor]
    messages: dict[tuple[str, str], torch.Tensor]
    routes: dict[tuple[str, str], torch.Tensor]
    responses: dict[tuple[str, str], torch.Tensor]
    expert_statistics: list[dict[str, Any]]
    branch_statistics: dict[str, float]


def _rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    selected = value if mask is None else value[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.detach().float().square().mean().sqrt().cpu())


def _route_override(
    route: torch.Tensor,
    valid: torch.Tensor,
    mode: str,
    *,
    relation_id: int,
    modality_id: int,
    seed: int,
) -> torch.Tensor:
    if mode == "original":
        return route
    result = torch.zeros_like(route)
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() == 0:
        return result
    if mode == "uniform":
        result[indices] = 1.0 / route.size(1)
    elif mode == "view_mean":
        result[indices] = route[indices].mean(dim=0, keepdim=True)
    elif mode == "node_permuted":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1009 * relation_id + 9176 * modality_id)
        permutation = torch.randperm(indices.numel(), generator=generator).to(indices.device)
        result[indices] = route[indices[permutation]]
    else:
        raise ValueError(f"unknown route intervention {mode!r}")
    return result


def audit_forward(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    classifier=None,
    *,
    cross_modal: bool = True,
    router_mode: str = "original",
    router_seed: int = 20261012,
    physical_response: bool = True,
    semantic_response: bool = True,
    intrinsic_skip: bool = True,
    capture: bool = False,
) -> AuditForward:
    """Run an eval-only V1-equation forward with independent frozen gates.

    The route modes are ``original``, ``uniform``, ``view_mean``, and
    ``node_permuted``.  All gates are local variables; no parameter, module
    mode, normalization state, optimizer, or graph cache is replaced.
    """
    if model.training:
        raise RuntimeError("frozen audit requires model.eval()")
    if torch.is_grad_enabled():
        raise RuntimeError("frozen audit requires torch.no_grad() or inference_mode()")
    if model.variant != "e2_full":
        raise ValueError("mechanism interventions require an E2 Full checkpoint")

    model._validate_input_features_once(x)
    p0, q0 = model._get_physical_operator(edge_index.to(device=x.device, dtype=torch.long), x.dtype)
    if model.semantic_pt.device != x.device or p0.device != x.device:
        raise RuntimeError("audit graph operators and features must share a device")
    operators = {"physical": p0, "text": model.semantic_pt, "visual": model.semantic_pv}
    qualities = {"physical": q0, "text": model.semantic_q_text, "visual": model.semantic_q_visual}

    raw = {
        "text": x[:, : model.text_dim],
        "visual": x[:, model.text_dim : model.text_dim + model.visual_dim],
    }
    h = {name: model.projectors[name](raw[name]) for name in _MODALITIES}
    if cross_modal:
        phi = {
            "text": model.cross_modal["visual_to_text"](h["visual"]),
            "visual": model.cross_modal["text_to_visual"](h["text"]),
        }
    else:
        phi = {}

    messages: dict[tuple[str, str], torch.Tensor] = {}
    aggregates: dict[tuple[str, str], torch.Tensor] = {}
    routes: dict[tuple[str, str], torch.Tensor] = {}
    responses: dict[tuple[str, str], torch.Tensor] = {}
    expert_statistics: list[dict[str, Any]] = []
    for modality_id, modality in enumerate(_MODALITIES):
        messages[("physical", modality)] = h[modality]
        for relation, relation_index in (("text", 0), ("visual", 1)):
            if cross_modal:
                messages[(relation, modality)] = model.compose_cross_modal_message(
                    h[modality], phi["text" if modality == "text" else "visual"], model.rho[relation_index, modality_id]
                )
            else:
                messages[(relation, modality)] = h[modality]

    for relation_id, relation in enumerate(_RELATIONS):
        q = qualities[relation]
        valid = q > 0
        for modality_id, modality in enumerate(_MODALITIES):
            aggregate = row_quality_aggregate(operators[relation], messages[(relation, modality)], q, model.eps)
            h_norm = model.message_norm(h[modality])
            u_norm = model.aggregate_norm(aggregate)
            diff = h_norm - u_norm
            expert_input = torch.cat((h_norm, u_norm, diff.abs(), h_norm * u_norm), dim=-1)
            expert_outputs = [model._run_expert(expert, expert_input) for expert in model.experts]
            route = model._router_weights(h_norm, u_norm, q, valid, relation_id, modality_id)
            route = _route_override(
                route, valid, router_mode, relation_id=relation_id, modality_id=modality_id, seed=router_seed
            )
            stacked = torch.stack(expert_outputs, dim=1)
            response = (stacked * route.unsqueeze(-1)).sum(dim=1)
            response = response * valid.to(response.dtype).unsqueeze(-1)
            aggregates[(relation, modality)] = aggregate
            routes[(relation, modality)] = route
            responses[(relation, modality)] = response
            if capture:
                expert_float = stacked[valid].float()
                cosines: list[float] = []
                if expert_float.numel():
                    normalized = F.normalize(expert_float, dim=-1, eps=model.eps)
                    for left in range(normalized.size(1)):
                        for right in range(left + 1, normalized.size(1)):
                            cosines.append(float((normalized[:, left] * normalized[:, right]).sum(dim=-1).mean().cpu()))
                expert_statistics.append(
                    {
                        "relation": relation,
                        "modality": modality,
                        "valid_nodes": int(valid.sum().item()),
                        "message_rms": _rms(messages[(relation, modality)]),
                        "aggregate_rms": _rms(aggregate, valid),
                        "expert_rms": [_rms(stacked[:, k], valid) for k in range(stacked.size(1))],
                        "expert_pairwise_cosine": cosines,
                        "weighted_response_rms": _rms(response, valid),
                    }
                )

    q_text, q_visual = model.semantic_q_text, model.semantic_q_visual
    q_semantic = q_text + q_visual
    semantic_valid = q_semantic > 0
    mixed: dict[str, torch.Tensor] = {}
    physical: dict[str, torch.Tensor] = {}
    semantic: dict[str, torch.Tensor] = {}
    for modality in _MODALITIES:
        physical[modality] = responses[("physical", modality)]
        semantic_numerator = (
            q_text.unsqueeze(-1) * responses[("text", modality)]
            + q_visual.unsqueeze(-1) * responses[("visual", modality)]
        )
        semantic[modality] = semantic_numerator / q_semantic.clamp_min(model.eps).unsqueeze(-1)
        semantic[modality] = semantic[modality] * semantic_valid.to(x.dtype).unsqueeze(-1)
        physical_term = model.beta_phy * physical[modality] if physical_response else torch.zeros_like(physical[modality])
        semantic_term = model.beta_sem * semantic[modality] if semantic_response else torch.zeros_like(semantic[modality])
        mixed[modality] = model.mix_norm[modality](h[modality] + physical_term + semantic_term)
    q_state = {
        modality: model.physical_norm[modality](mixed[modality] + model.gamma * torch.sparse.mm(p0, mixed[modality]))
        for modality in _MODALITIES
    }
    joined = torch.cat(
        (q_state["text"], q_state["visual"], (q_state["text"] - q_state["visual"]).abs(), q_state["text"] * q_state["visual"]),
        dim=-1,
    )
    skip_input = torch.cat((h["text"], h["visual"]), dim=-1)
    fused = model.fusion(joined)
    skip_value = model.skip(skip_input)
    z = model.fusion_norm(fused + (skip_value if intrinsic_skip else torch.zeros_like(skip_value)))
    logits = classifier(z) if classifier is not None else None
    branch_statistics = {}
    if capture:
        branch_statistics = {
            "physical_response_rms": _rms(torch.cat([physical[m] for m in _MODALITIES], dim=0)),
            "physical_weighted_rms": _rms(torch.cat([model.beta_phy * physical[m] for m in _MODALITIES], dim=0)),
            "semantic_response_rms": _rms(torch.cat([semantic[m] for m in _MODALITIES], dim=0)),
            "semantic_weighted_rms": _rms(torch.cat([model.beta_sem * semantic[m] for m in _MODALITIES], dim=0)),
            "intrinsic_skip_input_rms": _rms(skip_input),
            "intrinsic_skip_projected_rms": _rms(skip_value),
            "fusion_projected_rms": _rms(fused),
            "final_embedding_rms": _rms(z),
        }
    return AuditForward(
        embedding=z,
        logits=logits,
        aggregates=aggregates if capture else {},
        messages=messages if capture else {},
        routes=routes if capture else {},
        responses=responses if capture else {},
        expert_statistics=expert_statistics,
        branch_statistics=branch_statistics,
    )


def summarize_values(values: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray | None = None) -> dict[str, Any]:
    """Finite-value distribution summary with explicit n and empty handling."""
    if isinstance(values, torch.Tensor):
        array = values.detach().float().reshape(-1).cpu().numpy()
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            keep = mask.detach().bool().reshape(-1).cpu().numpy()
        else:
            keep = np.asarray(mask, dtype=bool).reshape(-1)
        if keep.size != array.size:
            raise ValueError("summary mask size does not match the values")
        array = array[keep]
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "q10": None, "median": None, "q90": None, "q99": None, "max": None}
    quantiles = np.quantile(array, [0.1, 0.5, 0.9, 0.99])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q90": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "max": float(array.max()),
    }


def node_norms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().norm(dim=-1)


def context_comparison(left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor, eps: float = 1e-8) -> dict[str, Any]:
    """Compare contexts; cosine is defined only where both vectors are nonzero."""
    left_norm = node_norms(left)
    right_norm = node_norms(right)
    delta_norm = node_norms(left - right)
    eligible = valid.bool()
    cosine_valid = eligible & (left_norm > eps) & (right_norm > eps)
    cosine = F.cosine_similarity(left.float(), right.float(), dim=-1, eps=eps)
    scale = torch.maximum(left_norm, right_norm).clamp_min(eps)
    relative = delta_norm / scale
    return {
        "eligible_nodes": int(eligible.sum().item()),
        "cosine_valid_nodes": int(cosine_valid.sum().item()),
        "cosine": summarize_values(cosine, cosine_valid),
        "normalized_l2": summarize_values(relative, eligible),
        "left_rms": _rms(left, eligible),
        "right_rms": _rms(right, eligible),
        "difference_rms": _rms(left - right, eligible),
        "large_difference_fraction_rel_l2_gt_0_1": float((relative[eligible] > 0.1).float().mean().item()) if bool(eligible.any()) else 0.0,
        "small_difference_fraction_rel_l2_le_0_01": float((relative[eligible] <= 0.01).float().mean().item()) if bool(eligible.any()) else 0.0,
    }


def jensen_shannon(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-row Jensen-Shannon divergence in nats."""
    p = left.float().clamp_min(eps)
    q = right.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)


def validation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute NC metrics using only train and validation labels.

    MAGData contains a full y tensor, but this function never indexes
    ``test_idx`` or reads a test-label value. The label set mirrors the pilot
    runner's ``_resolve_nc_eval_labels(include_test=False)`` behavior.
    """
    train_val_idx = torch.cat((train_idx.reshape(-1), val_idx.reshape(-1))).to(labels.device)
    known = labels[train_val_idx]
    eval_labels = sorted(int(v) for v in torch.unique(known[(known >= 0) & (known < num_classes)]).cpu().tolist())
    if not eval_labels:
        raise ValueError("train/validation labels contain no valid class IDs")
    idx = val_idx.detach().cpu().long()
    target = labels[idx].detach().cpu().long()
    pred = logits[idx.to(logits.device)].argmax(dim=-1).detach().cpu().long()
    result = {
        "val_ce": float(F.cross_entropy(logits[idx.to(logits.device)], target.to(logits.device)).cpu()),
        "val_acc": float((pred == target).float().mean()),
    }
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    result["val_macro_f1"] = float(
        f1_score(target.numpy(), pred.numpy(), labels=eval_labels, average="macro", zero_division=0)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        target.numpy(), pred.numpy(), labels=eval_labels, zero_division=0
    )
    predicted_counts = torch.bincount(pred, minlength=num_classes)
    class_rows = [
        {
            "class_id": class_id,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "validation_prediction_count": int(predicted_counts[class_id]),
            "validation_prediction_fraction": float(predicted_counts[class_id] / max(int(pred.numel()), 1)),
        }
        for i, class_id in enumerate(eval_labels)
    ]
    result["eval_label_ids_train_val_only"] = eval_labels
    return result, class_rows


def tensor_sha256(module) -> str:
    """Hash persistent model/classifier state for before/after immutability checks."""
    import hashlib

    digest = hashlib.sha256()
    for prefix, state in (("model", module[0].state_dict()), ("classifier", module[1].state_dict())):
        digest.update(prefix.encode("utf-8"))
        for name, tensor in sorted(state.items()):
            digest.update(name.encode("utf-8"))
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()
