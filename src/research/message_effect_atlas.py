from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .message_effect_replay import (
    FrozenHostCache,
    estimator_receiver_split,
    replay_local_basis_changes,
    singleton_basis_delta,
)
from .message_effect_estimator import parameter_free_pair

MODALITIES = ("text", "visual")
SCALAR_FEATURE_NAMES = (
    "normalized_edge_weight",
    "log1p_receiver_degree",
    "log1p_sender_degree",
    "semantic_cosine_text",
    "semantic_cosine_visual",
    "role_score_text",
    "role_score_visual",
    "active_modality_support",
    "active_modality_router_cosine",
    "active_modality_router_reliability",
    "raw_message_rms",
    "message_to_state_rms_ratio",
    *(f"hop_{k}" for k in range(1, 5)),
    "modality_text",
    "modality_visual",
)


def select_receiver_edges(
    cache: FrozenHostCache,
    receivers: Sequence[int] | torch.Tensor,
    *,
    max_edges: int = 4,
    seed: int = 2027,
) -> tuple[list[int], dict[int, float]]:
    chosen: list[int] = []
    probabilities: dict[int, float] = {}
    dst = cache.dst.detach().cpu()
    edge_order = torch.argsort(dst, stable=True)
    counts = torch.bincount(dst, minlength=cache.indegree.numel())
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    for receiver in sorted({int(x) for x in torch.as_tensor(receivers).reshape(-1).tolist()}):
        start, end = int(offsets[receiver]), int(offsets[receiver + 1])
        incoming = edge_order[start:end]
        degree = int(incoming.numel())
        probability = 1.0 if degree <= max_edges else max_edges / degree
        if degree > max_edges:
            stable_seed = int.from_bytes(
                hashlib.sha256(f"{cache.dataset}:{receiver}:{seed}".encode("utf-8")).digest()[:8],
                "little",
            ) & ((1 << 63) - 1)
            generator = torch.Generator().manual_seed(stable_seed)
            incoming = incoming[torch.randperm(degree, generator=generator)[:max_edges]].sort().values
        chosen.extend(int(x) for x in incoming.tolist())
        probabilities[receiver] = probability
    return sorted(chosen), probabilities


def _previous_state(cache: FrozenHostCache, modality: str, hop: int) -> torch.Tensor:
    return cache.prior[modality] if hop == 1 else cache.raw_states[modality][hop - 2]


def _rms_rows(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean(dim=-1).sqrt()


def _local_logits(
    cache: FrozenHostCache,
    receivers: torch.Tensor,
    changes: list[dict[str, Any]],
    labels: torch.Tensor,
    baseline_loss: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for start in range(0, receivers.numel(), batch_size):
        end = min(start + batch_size, receivers.numel())
        chunk_changes = [
            {"modality": c["modality"], "hop": c["hop"], "delta": c["delta"][start:end]}
            for c in changes
        ]
        replay = replay_local_basis_changes(cache, receivers[start:end], chunk_changes)
        loss = F.cross_entropy(
            replay["logits"], labels[start:end], reduction="none"
        ) - baseline_loss[start:end]
        pieces.append(loss.detach())
    return torch.cat(pieces) if pieces else receivers.new_empty((0,), dtype=torch.float32)


def _replay_edge_bundle(
    cache: FrozenHostCache,
    edge_ids: torch.Tensor,
    receivers: torch.Tensor,
    specs: Sequence[tuple[str, int]],
    labels: torch.Tensor,
    baseline_loss: torch.Tensor,
    *,
    compensated: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    changes: list[dict[str, Any]] = []
    mass = cache.receiver_mass.index_select(0, receivers)
    coefficient = cache.edge_weight.index_select(0, edge_ids)
    retained = mass - coefficient
    feasible = retained > 1.0e-12
    scale = torch.where(feasible, mass / retained.clamp_min(1.0e-12), torch.ones_like(mass))
    for modality, hop in specs:
        previous = _previous_state(cache, modality, hop)
        message = coefficient.unsqueeze(-1) * previous.index_select(
            0, cache.src.index_select(0, edge_ids)
        )
        if compensated:
            aggregate = cache.raw_states[modality][hop - 1].index_select(0, receivers)
            new_aggregate = scale.unsqueeze(-1) * (aggregate - message)
            delta = (new_aggregate - aggregate) / cache.denominators[modality][hop - 1]
        else:
            delta = singleton_basis_delta(message, cache.denominators[modality][hop - 1])
        changes.append({"modality": modality, "hop": hop, "delta": delta})
    if compensated:
        keep = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        if not keep.numel():
            return cache.logits.new_full((edge_ids.numel(),), float("nan")), feasible
        subchanges = [
            {"modality": c["modality"], "hop": c["hop"], "delta": c["delta"].index_select(0, keep)}
            for c in changes
        ]
        effect = _local_logits(
            cache,
            receivers.index_select(0, keep),
            subchanges,
            labels.index_select(0, keep),
            baseline_loss.index_select(0, keep),
        )
        result = effect.new_full((edge_ids.numel(),), float("nan"))
        result[keep] = effect
        return result, feasible
    return (
        _local_logits(cache, receivers, changes, labels, baseline_loss),
        feasible,
    )


@torch.no_grad()
def generate_message_effect_atlas(
    cache: FrozenHostCache,
    edge_ids: Sequence[int] | torch.Tensor,
    train_labels_by_node: torch.Tensor,
    receiver_sample_probability: dict[int, float],
    edge_sample_probability: dict[int, float],
    *,
    host_seed: int = 42,
    replay_batch_size: int = 1024,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Generate train-receiver exposed-message singleton and bundle effects."""
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long, device=cache.device).reshape(-1)
    if not edge_ids.numel():
        raise ValueError("the atlas needs at least one sampled directed edge")
    receivers_all = cache.dst.index_select(0, edge_ids)
    senders_all = cache.src.index_select(0, edge_ids)
    labels_all = train_labels_by_node.to(cache.device).index_select(0, receivers_all)
    if bool((labels_all < 0).any()):
        raise ValueError("all selected Atlas receivers must have training labels")
    base_logits_all = cache.logits.index_select(0, receivers_all)
    base_loss_all = F.cross_entropy(base_logits_all, labels_all, reduction="none")
    probabilities_all = cache.probabilities.index_select(0, receivers_all)
    sorted_probs = probabilities_all.sort(dim=-1, descending=True).values
    entropy_all = -(probabilities_all * probabilities_all.clamp_min(1.0e-12).log()).sum(dim=-1)
    max_probability_all = sorted_probs[:, 0]
    margin_all = sorted_probs[:, 0] - sorted_probs[:, 1]

    singleton_rows: list[dict[str, Any]] = []
    scalar_features: list[torch.Tensor] = []
    pair_text_features: list[torch.Tensor] = []
    pair_visual_features: list[torch.Tensor] = []
    host_features: list[torch.Tensor] = []
    delete_targets: list[torch.Tensor] = []
    comp_targets: list[torch.Tensor] = []
    row_receivers: list[torch.Tensor] = []
    row_edge_indices: list[torch.Tensor] = []
    row_modality_codes: list[torch.Tensor] = []
    single_effects: dict[tuple[int, str, int], tuple[float, float | None]] = {}

    src_nodes = senders_all
    receiver_nodes = receivers_all
    edge_weight = cache.edge_weight.index_select(0, edge_ids)
    receiver_degree = cache.indegree.index_select(0, receiver_nodes)
    sender_degree = cache.indegree.index_select(0, src_nodes)
    mass = cache.receiver_mass.index_select(0, receiver_nodes)
    retained_mass = mass - edge_weight
    feasible = retained_mass > 1.0e-12
    comp_scale = torch.where(
        feasible, mass / retained_mass.clamp_min(1.0e-12), torch.full_like(mass, float("nan"))
    )

    for modality_index, modality in enumerate(MODALITIES):
        for hop in range(1, 5):
            previous = _previous_state(cache, modality, hop)
            message = edge_weight.unsqueeze(-1) * previous.index_select(0, src_nodes)
            receiver_state = cache.raw_states[modality][hop - 1].index_select(0, receiver_nodes)
            denominator = cache.denominators[modality][hop - 1]
            deletion_delta_basis = singleton_basis_delta(message, denominator)
            compensated_state = comp_scale.nan_to_num(nan=1.0).unsqueeze(-1) * (
                receiver_state - message
            )
            compensated_delta_basis = (compensated_state - receiver_state) / denominator

            delete_effect = torch.empty(edge_ids.numel(), device=cache.device, dtype=cache.logits.dtype)
            for start in range(0, edge_ids.numel(), replay_batch_size):
                end = min(start + replay_batch_size, edge_ids.numel())
                replay = replay_local_basis_changes(
                    cache,
                    receiver_nodes[start:end],
                    [{"modality": modality, "hop": hop, "delta": deletion_delta_basis[start:end]}],
                )
                delete_effect[start:end] = F.cross_entropy(
                    replay["logits"], labels_all[start:end], reduction="none"
                ) - base_loss_all[start:end]

            comp_effect = torch.full_like(delete_effect, float("nan"))
            feasible_ids = torch.nonzero(feasible, as_tuple=False).reshape(-1)
            for start in range(0, feasible_ids.numel(), replay_batch_size):
                chosen = feasible_ids[start : start + replay_batch_size]
                replay = replay_local_basis_changes(
                    cache,
                    receiver_nodes.index_select(0, chosen),
                    [{"modality": modality, "hop": hop,
                      "delta": compensated_delta_basis.index_select(0, chosen)}],
                )
                comp_effect[chosen] = F.cross_entropy(
                    replay["logits"], labels_all.index_select(0, chosen), reduction="none"
                ) - base_loss_all.index_select(0, chosen)

            semantic_text = cache.semantic_cosine["text"].index_select(0, edge_ids)
            semantic_visual = cache.semantic_cosine["visual"].index_select(0, edge_ids)
            role_text = cache.role_score["text"].index_select(0, edge_ids)
            role_visual = cache.role_score["visual"].index_select(0, edge_ids)
            support_text = cache.support_flag["text"].index_select(0, edge_ids)
            support_visual = cache.support_flag["visual"].index_select(0, edge_ids)
            active_support = (support_text if modality == "text" else support_visual).float()
            router_cos = cache.router_cosine[modality].index_select(0, edge_ids)
            router_reliability = cache.router_reliability[modality].index_select(0, edge_ids)
            message_rms = _rms_rows(message)
            state_rms = _rms_rows(receiver_state)
            message_ratio = message_rms / state_rms.clamp_min(1.0e-12)
            log_degrees = torch.stack(
                (receiver_degree.float().log1p(), sender_degree.float().log1p()), dim=-1
            )
            hop_one_hot = F.one_hot(
                torch.full((edge_ids.numel(),), hop - 1, device=cache.device), num_classes=4
            ).float()
            modality_one_hot = F.one_hot(
                torch.full((edge_ids.numel(),), modality_index, device=cache.device), num_classes=2
            ).float()
            scalar = torch.cat(
                (
                    edge_weight[:, None],
                    log_degrees,
                    semantic_text[:, None],
                    semantic_visual[:, None],
                    role_text[:, None],
                    role_visual[:, None],
                    active_support[:, None],
                    router_cos[:, None],
                    router_reliability[:, None],
                    message_rms[:, None],
                    message_ratio[:, None],
                    hop_one_hot,
                    modality_one_hot,
                ),
                dim=-1,
            )
            pair_text = parameter_free_pair(
                _previous_state(cache, "text", hop).index_select(0, receiver_nodes),
                _previous_state(cache, "text", hop).index_select(0, src_nodes),
            )
            pair_visual = parameter_free_pair(
                _previous_state(cache, "visual", hop).index_select(0, receiver_nodes),
                _previous_state(cache, "visual", hop).index_select(0, src_nodes),
            )
            active_mixture = cache.raw_mixture[modality].index_select(0, receiver_nodes)
            host = torch.cat(
                (
                    cache.z.index_select(0, receiver_nodes),
                    active_mixture,
                    base_logits_all,
                    entropy_all[:, None],
                    max_probability_all[:, None],
                    margin_all[:, None],
                ),
                dim=-1,
            )
            edge_weight_cpu = edge_weight.detach().cpu().tolist()
            receiver_degree_cpu = receiver_degree.detach().cpu().tolist()
            sender_degree_cpu = sender_degree.detach().cpu().tolist()
            sem_text_cpu = semantic_text.detach().cpu().tolist()
            sem_visual_cpu = semantic_visual.detach().cpu().tolist()
            role_text_cpu = role_text.detach().cpu().tolist()
            role_visual_cpu = role_visual.detach().cpu().tolist()
            support_text_cpu = support_text.detach().cpu().tolist()
            support_visual_cpu = support_visual.detach().cpu().tolist()
            router_cos_cpu = router_cos.detach().cpu().tolist()
            router_rel_cpu = router_reliability.detach().cpu().tolist()
            message_rms_cpu = message_rms.detach().cpu().tolist()
            state_rms_cpu = state_rms.detach().cpu().tolist()
            message_ratio_cpu = message_ratio.detach().cpu().tolist()
            base_loss_cpu = base_loss_all.detach().cpu().tolist()
            entropy_cpu = entropy_all.detach().cpu().tolist()
            max_prob_cpu = max_probability_all.detach().cpu().tolist()
            margin_cpu = margin_all.detach().cpu().tolist()
            delete_cpu = delete_effect.detach().cpu().tolist()
            comp_cpu = comp_effect.detach().cpu().tolist()
            scale_cpu = comp_scale.detach().cpu().tolist()
            feasible_cpu = feasible.detach().cpu().tolist()
            recv_cpu = receiver_nodes.detach().cpu().tolist()
            send_cpu = src_nodes.detach().cpu().tolist()
            eid_cpu = edge_ids.detach().cpu().tolist()
            for i, (receiver, sender, eid) in enumerate(zip(recv_cpu, send_cpu, eid_cpu)):
                key = (int(eid), modality, hop)
                singleton_rows.append(
                    {
                        "dataset": cache.dataset,
                        "host_seed": host_seed,
                        "receiver_id": int(receiver),
                        "sender_id": int(sender),
                        "directed_edge_index": int(eid),
                        "modality": modality,
                        "hop": hop,
                        "normalized_edge_weight": float(edge_weight_cpu[i]),
                        "receiver_indegree": int(receiver_degree_cpu[i]),
                        "sender_indegree": int(sender_degree_cpu[i]),
                        "receiver_sample_probability": float(receiver_sample_probability[int(receiver)]),
                        "edge_sample_probability": float(edge_sample_probability[int(receiver)]),
                        "semantic_cosine_text": float(sem_text_cpu[i]),
                        "semantic_cosine_visual": float(sem_visual_cpu[i]),
                        "role_score_text": float(role_text_cpu[i]),
                        "role_score_visual": float(role_visual_cpu[i]),
                        "support_flag_text": bool(support_text_cpu[i]),
                        "support_flag_visual": bool(support_visual_cpu[i]),
                        "router_cosine_active_modality": float(router_cos_cpu[i]),
                        "router_reliability_active_modality": float(router_rel_cpu[i]),
                        "raw_message_rms": float(message_rms_cpu[i]),
                        "raw_receiver_hop_state_rms": float(state_rms_cpu[i]),
                        "message_to_state_rms_ratio": float(message_ratio_cpu[i]),
                        "baseline_loss": float(base_loss_cpu[i]),
                        "baseline_entropy": float(entropy_cpu[i]),
                        "baseline_max_prob": float(max_prob_cpu[i]),
                        "baseline_margin": float(margin_cpu[i]),
                        "delta_delete": float(delete_cpu[i]),
                        "compensation_feasible": bool(feasible_cpu[i]),
                        "comp_scale": float(scale_cpu[i]) if feasible_cpu[i] else None,
                        "delta_comp": float(comp_cpu[i]) if feasible_cpu[i] else None,
                    }
                )
                single_effects[key] = (
                    float(delete_cpu[i]),
                    float(comp_cpu[i]) if feasible_cpu[i] else None,
                )
            scalar_features.append(scalar.detach().cpu().float())
            pair_text_features.append(pair_text.detach().cpu().float())
            pair_visual_features.append(pair_visual.detach().cpu().float())
            host_features.append(host.detach().cpu().float())
            delete_targets.append(delete_effect.detach().cpu().float())
            comp_targets.append(comp_effect.detach().cpu().float())
            row_receivers.append(receiver_nodes.detach().cpu())
            row_edge_indices.append(edge_ids.detach().cpu())
            row_modality_codes.append(
                torch.full((edge_ids.numel(),), modality_index, dtype=torch.int8)
            )

    baseline_loss = base_loss_all
    bundle_rows: list[dict[str, Any]] = []
    bundle_specs: list[tuple[str, list[tuple[str, int]]]] = []
    for hop in range(1, 5):
        bundle_specs.append((f"same_hop_text_visual_k{hop}", [("text", hop), ("visual", hop)]))
    for modality in MODALITIES:
        bundle_specs.append((f"{modality}_all_hops", [(modality, hop) for hop in range(1, 5)]))
    bundle_specs.append(("full_exposed_edge_8_messages", [
        (modality, hop) for modality in MODALITIES for hop in range(1, 5)
    ]))
    for bundle_type, specs in bundle_specs:
        delete_bundle, bundle_feasible = _replay_edge_bundle(
            cache, edge_ids, receivers_all, specs, labels_all, baseline_loss, compensated=False
        )
        comp_bundle, _ = _replay_edge_bundle(
            cache, edge_ids, receivers_all, specs, labels_all, baseline_loss, compensated=True
        )
        delete_bundle_cpu = delete_bundle.detach().cpu().tolist()
        comp_bundle_cpu = comp_bundle.detach().cpu().tolist()
        feasible_cpu = bundle_feasible.detach().cpu().tolist()
        bundle_receiver_cpu = receivers_all.detach().cpu().tolist()
        bundle_sender_cpu = senders_all.detach().cpu().tolist()
        bundle_edge_cpu = edge_ids.detach().cpu().tolist()
        for i, eid in enumerate(bundle_edge_cpu):
            components = [single_effects[(int(eid), modality, hop)] for modality, hop in specs]
            delete_component_values = [item[0] for item in components]
            comp_component_values = [item[1] for item in components]
            delete_interaction = float(delete_bundle_cpu[i]) - sum(delete_component_values)
            delete_relative = abs(delete_interaction) / (
                sum(abs(v) for v in delete_component_values) + 1.0e-12
            )
            if feasible_cpu[i]:
                comp_interaction = float(comp_bundle_cpu[i]) - sum(
                    float(v) for v in comp_component_values if v is not None
                )
                comp_relative = abs(comp_interaction) / (
                    sum(abs(float(v)) for v in comp_component_values if v is not None) + 1.0e-12
                )
            else:
                comp_interaction = float("nan")
                comp_relative = float("nan")
            bundle_rows.append(
                {
                    "dataset": cache.dataset,
                    "host_seed": host_seed,
                    "receiver_id": int(bundle_receiver_cpu[i]),
                    "sender_id": int(bundle_sender_cpu[i]),
                    "directed_edge_index": int(eid),
                    "bundle_type": bundle_type,
                    "component_count": len(specs),
                    "delta_bundle_delete": float(delete_bundle_cpu[i]),
                    "delete_component_sum": sum(delete_component_values),
                    "interaction_delete": delete_interaction,
                    "relative_interaction_delete": delete_relative,
                    "compensation_feasible": bool(feasible_cpu[i]),
                    "delta_bundle_comp": float(comp_bundle_cpu[i]) if feasible_cpu[i] else None,
                    "comp_component_sum": (
                        sum(float(v) for v in comp_component_values if v is not None)
                        if feasible_cpu[i] else None
                    ),
                    "interaction_comp": comp_interaction if feasible_cpu[i] else None,
                    "relative_interaction_comp": comp_relative if feasible_cpu[i] else None,
                }
            )

    features = {
        "scalar": torch.cat(scalar_features),
        "pair_text": torch.cat(pair_text_features),
        "pair_visual": torch.cat(pair_visual_features),
        "host": torch.cat(host_features),
        "target_delete": torch.cat(delete_targets),
        "target_comp": torch.cat(comp_targets),
        "receiver_id": torch.cat(row_receivers).long(),
        "edge_index": torch.cat(row_edge_indices).long(),
        "modality_code": torch.cat(row_modality_codes).long(),
    }
    if any(
        not bool(torch.isfinite(value).all())
        for key, value in features.items()
        if key != "target_comp" and value.is_floating_point()
    ) or bool(torch.isinf(features["target_comp"]).any()):
        raise FloatingPointError("non-finite values found in finite Atlas feature/target entries")
    return singleton_rows, bundle_rows, features


def _safe_spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    if len(a) < 2 or len(b) < 2:
        return float("nan")
    value = spearmanr(a, b).statistic
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def _safe_pearson(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import pearsonr

    if len(a) < 2 or len(b) < 2:
        return float("nan")
    value = pearsonr(a, b).statistic
    return float(value) if math.isfinite(float(value)) else float("nan")


def effect_diagnostics(
    singleton_rows: list[dict[str, Any]], bundle_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Descriptive distributions, heterogeneity, compensation, and bundle summaries."""
    distribution: list[dict[str, Any]] = []
    heterogeneity: list[dict[str, Any]] = []
    compensation: list[dict[str, Any]] = []
    bundle_summary: list[dict[str, Any]] = []
    heuristic: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in singleton_rows:
        groups[(row["modality"], int(row["hop"]), row["receiver_id"])].append(row)
    for dataset in sorted({r["dataset"] for r in singleton_rows}):
        data = [r for r in singleton_rows if r["dataset"] == dataset]
        for modality in MODALITIES:
            for hop in range(1, 5):
                subset = [r for r in data if r["modality"] == modality and r["hop"] == hop]
                for target_name, field in (("delete", "delta_delete"), ("comp", "delta_comp")):
                    values = [float(r[field]) for r in subset if r[field] is not None and math.isfinite(float(r[field]))]
                    if values:
                        tensor = torch.tensor(values, dtype=torch.float64)
                        quantiles = torch.quantile(tensor, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)).tolist()
                        distribution.append(
                            {
                                "dataset": dataset,
                                "modality": modality,
                                "hop": hop,
                                "target": target_name,
                                "n": len(values),
                                "mean": float(tensor.mean()),
                                "std": float(tensor.std(unbiased=False)),
                                "p05": quantiles[0],
                                "p25": quantiles[1],
                                "p50": quantiles[2],
                                "p75": quantiles[3],
                                "p95": quantiles[4],
                                "median_abs": float(tensor.abs().median()),
                                "negative_fraction": float((tensor < 0).float().mean()),
                            }
                        )
                cue_fields = (
                    "semantic_cosine_text",
                    "semantic_cosine_visual",
                    "role_score_text",
                    "role_score_visual",
                    "support_flag_text",
                    "support_flag_visual",
                    "router_cosine_active_modality",
                    "router_reliability_active_modality",
                    "normalized_edge_weight",
                    "receiver_indegree",
                    "raw_message_rms",
                    "message_to_state_rms_ratio",
                )
                for cue in cue_fields:
                    cue_values = [float(r[cue]) for r in subset]
                    delete_values = [float(r["delta_delete"]) for r in subset]
                    comp_pairs = [
                        (float(r[cue]), float(r["delta_comp"]))
                        for r in subset if r["delta_comp"] is not None
                    ]
                    heuristic.append(
                        {
                            "dataset": dataset,
                            "modality": modality,
                            "hop": hop,
                            "cue": cue,
                            "n_delete": len(delete_values),
                            "spearman_delete": _safe_spearman(cue_values, delete_values),
                            "spearman_comp": _safe_spearman(
                                [x for x, _ in comp_pairs], [y for _, y in comp_pairs]
                            ),
                        }
                    )
                sub_support = [
                    r for r in subset
                    if bool(r["support_flag_text"] if modality == "text" else r["support_flag_visual"])
                ]
                sub_discrepant = [
                    r for r in subset
                    if not bool(r["support_flag_text"] if modality == "text" else r["support_flag_visual"])
                ]
                heuristic.append(
                    {
                        "dataset": dataset,
                        "modality": modality,
                        "hop": hop,
                        "cue": "P(delta_delete<0 | role_partition)",
                        "support_negative_fraction": (
                            sum(float(r["delta_delete"]) < 0 for r in sub_support) / len(sub_support)
                            if sub_support else float("nan")
                        ),
                        "discrepant_negative_fraction": (
                            sum(float(r["delta_delete"]) < 0 for r in sub_discrepant) / len(sub_discrepant)
                            if sub_discrepant else float("nan")
                        ),
                        "overall_negative_fraction": (
                            sum(float(r["delta_delete"]) < 0 for r in subset) / len(subset)
                            if subset else float("nan")
                        ),
                    }
                )

        for modality in MODALITIES:
            for hop in range(1, 5):
                receiver_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for r in data:
                    if r["modality"] == modality and r["hop"] == hop:
                        receiver_groups[int(r["receiver_id"])].append(r)
                group_std, group_range, pair_abs = [], [], []
                for rows in receiver_groups.values():
                    vals = [float(x["delta_delete"]) for x in rows]
                    if len(vals) < 2:
                        continue
                    group_std.append(float(torch.tensor(vals).std(unbiased=False)))
                    group_range.append(max(vals) - min(vals))
                    pair_abs.extend(abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals)))
                heterogeneity.append(
                    {
                        "dataset": dataset,
                        "diagnostic": "within_receiver_edge_effects",
                        "modality": modality,
                        "hop": hop,
                        "groups_n": len(group_std),
                        "mean_within_group_std": float(torch.tensor(group_std).mean()) if group_std else float("nan"),
                        "mean_within_group_range": float(torch.tensor(group_range).mean()) if group_range else float("nan"),
                        "median_pairwise_abs_difference": float(torch.tensor(pair_abs).median()) if pair_abs else float("nan"),
                    }
                )

        edge_hop: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        edge_mod: dict[tuple[int, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
        receiver_effects: dict[int, list[float]] = defaultdict(list)
        for r in data:
            edge_hop[(r["receiver_id"], r["directed_edge_index"], r["hop"])][r["modality"]] = r
            edge_mod[(r["receiver_id"], r["directed_edge_index"], r["modality"])][r["hop"]] = r
            receiver_effects[int(r["receiver_id"])].append(float(r["delta_delete"]))
        paired = [pair for pair in edge_hop.values() if "text" in pair and "visual" in pair]
        text_effect = [float(pair["text"]["delta_delete"]) for pair in paired]
        visual_effect = [float(pair["visual"]["delta_delete"]) for pair in paired]
        heterogeneity.append(
            {
                "dataset": dataset,
                "diagnostic": "text_visual_effect_difference",
                "groups_n": len(paired),
                "mean_text_minus_visual": (
                    sum(a - b for a, b in zip(text_effect, visual_effect)) / len(paired) if paired else float("nan")
                ),
                "median_abs_text_visual_difference": (
                    float(torch.tensor([abs(a - b) for a, b in zip(text_effect, visual_effect)]).median())
                    if paired else float("nan")
                ),
                "sign_disagreement_fraction": (
                    sum((a < 0) != (b < 0) for a, b in zip(text_effect, visual_effect)) / len(paired)
                    if paired else float("nan")
                ),
                "text_visual_spearman": _safe_spearman(text_effect, visual_effect),
                "text_visual_pearson": _safe_pearson(text_effect, visual_effect),
            }
        )
        hop_stds, mixed_sign, rank_values = [], [], []
        for hops in edge_mod.values():
            if len(hops) < 2:
                continue
            ordered = [hops[h]["delta_delete"] for h in sorted(hops)]
            hop_stds.append(float(torch.tensor(ordered).std(unbiased=False)))
            mixed_sign.append(min(ordered) < 0 < max(ordered))
            rank_values.append(_safe_spearman(list(range(len(ordered))), ordered))
        heterogeneity.append(
            {
                "dataset": dataset,
                "diagnostic": "hop_effect_heterogeneity",
                "groups_n": len(hop_stds),
                "mean_effect_std_across_hops": sum(hop_stds) / len(hop_stds) if hop_stds else float("nan"),
                "mixed_sign_fraction": sum(mixed_sign) / len(mixed_sign) if mixed_sign else float("nan"),
                "mean_hop_order_spearman": (
                    sum(v for v in rank_values if math.isfinite(v))
                    / sum(math.isfinite(v) for v in rank_values)
                    if any(math.isfinite(v) for v in rank_values) else float("nan")
                ),
            }
        )
        receiver_means = [sum(v) / len(v) for v in receiver_effects.values() if v]
        receiver_stds = [float(torch.tensor(v).std(unbiased=False)) for v in receiver_effects.values() if v]
        heterogeneity.append(
            {
                "dataset": dataset,
                "diagnostic": "receiver_effect_heterogeneity",
                "groups_n": len(receiver_means),
                "receiver_mean_effect_mean": sum(receiver_means) / len(receiver_means) if receiver_means else float("nan"),
                "receiver_mean_effect_std": float(torch.tensor(receiver_means).std(unbiased=False)) if receiver_means else float("nan"),
                "within_receiver_effect_std_mean": sum(receiver_stds) / len(receiver_stds) if receiver_stds else float("nan"),
            }
        )
        for target_delete, target_comp in (("delta_delete", "delta_comp"),):
            feasible_rows = [r for r in data if r[target_comp] is not None]
            x = [float(r[target_delete]) for r in feasible_rows]
            y = [float(r[target_comp]) for r in feasible_rows]
            receiver_local = []
            local_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for r in feasible_rows:
                local_groups[int(r["receiver_id"])].append(r)
            for group in local_groups.values():
                if len(group) >= 2:
                    receiver_local.append(_safe_spearman(
                        [float(r[target_delete]) for r in group],
                        [float(r[target_comp]) for r in group],
                    ))
            compensation.append(
                {
                    "dataset": dataset,
                    "n_feasible": len(feasible_rows),
                    "n_total": len(data),
                    "feasible_fraction": len(feasible_rows) / len(data) if data else float("nan"),
                    "spearman_delete_comp": _safe_spearman(x, y),
                    "pearson_delete_comp": _safe_pearson(x, y),
                    "sign_flip_fraction": (
                        sum((a < 0) != (b < 0) for a, b in zip(x, y)) / len(x) if x else float("nan")
                    ),
                    "median_effect_magnitude_ratio_comp_over_delete": (
                        float(torch.tensor([abs(b) / max(abs(a), 1.0e-12) for a, b in zip(x, y)]).median())
                        if x else float("nan")
                    ),
                    "receiver_local_ranking_agreement_mean": (
                        sum(v for v in receiver_local if math.isfinite(v))
                        / sum(math.isfinite(v) for v in receiver_local)
                        if any(math.isfinite(v) for v in receiver_local) else float("nan")
                    ),
                    "receiver_local_groups_n": len(receiver_local),
                }
            )

        for bundle_type in sorted({r["bundle_type"] for r in bundle_rows if r["dataset"] == dataset}):
            subset = [r for r in bundle_rows if r["dataset"] == dataset and r["bundle_type"] == bundle_type]
            for target in ("delete", "comp"):
                field = "interaction_" + target
                values = [
                    float(r[field]) for r in subset
                    if r[field] is not None and math.isfinite(float(r[field]))
                ]
                relatives = [
                    float(r["relative_interaction_" + target]) for r in subset
                    if r["relative_interaction_" + target] is not None
                    and math.isfinite(float(r["relative_interaction_" + target]))
                ]
                bundle_summary.append(
                    {
                        "dataset": dataset,
                        "bundle_type": bundle_type,
                        "target": target,
                        "n": len(values),
                        "interaction_mean": sum(values) / len(values) if values else float("nan"),
                        "interaction_std": float(torch.tensor(values).std(unbiased=False)) if values else float("nan"),
                        "median_abs_interaction": float(torch.tensor(values).abs().median()) if values else float("nan"),
                        "relative_interaction_median": float(torch.tensor(relatives).median()) if relatives else float("nan"),
                    }
                )
    return {
        "effect_distribution": distribution,
        "heterogeneity_diagnostics": heterogeneity,
        "compensation_diagnostics": compensation,
        "bundle_diagnostics": bundle_summary,
        "heuristic_alignment": heuristic,
    }


def make_estimator_split(features: dict[str, torch.Tensor], dataset: str) -> dict[str, set[int]]:
    return estimator_receiver_split(dataset, features["receiver_id"], seed=2027)
