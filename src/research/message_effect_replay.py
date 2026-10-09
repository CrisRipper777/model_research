from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class FrozenHostCache:
    """Receiver-local replay state from one frozen V4A R0 host."""

    dataset: str
    model: torch.nn.Module
    classifier: torch.nn.Module
    src: torch.Tensor
    dst: torch.Tensor
    edge_weight: torch.Tensor
    active: torch.Tensor
    indegree: torch.Tensor
    prior: dict[str, torch.Tensor]
    raw_states: dict[str, torch.Tensor]
    raw_basis: dict[str, torch.Tensor]
    denominators: dict[str, torch.Tensor]
    alpha: torch.Tensor
    route_weights: dict[str, torch.Tensor]
    strength: dict[str, torch.Tensor]
    modality_outputs: dict[str, torch.Tensor]
    raw_mixture: dict[str, torch.Tensor]
    z: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    semantic_cosine: dict[str, torch.Tensor]
    role_score: dict[str, torch.Tensor]
    support_flag: dict[str, torch.Tensor]
    router_cosine: dict[str, torch.Tensor]
    router_reliability: dict[str, torch.Tensor]
    receiver_mass: torch.Tensor
    device: torch.device


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def build_frozen_host_cache(
    dataset: str,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: str | torch.device,
) -> FrozenHostCache:
    """Run a frozen host once and retain only graph-sized tensors needed by replay."""
    device = torch.device(device)
    model = model.to(device).eval()
    classifier = classifier.to(device).eval()
    x = x.to(device)
    edge_index = edge_index.to(device)
    with torch.no_grad():
        z, _, _, _, info = model(x, edge_index, return_details=True)
        logits = classifier(z)
        probabilities = logits.softmax(dim=-1)
        src, dst, edge_weight, active, _ = model._normalized_operator(
            edge_index, x.size(0), x.dtype
        )
        indegree = torch.bincount(dst, minlength=x.size(0))
        prior: dict[str, torch.Tensor] = {}
        raw_states: dict[str, torch.Tensor] = {}
        raw_basis: dict[str, torch.Tensor] = {}
        denominators: dict[str, torch.Tensor] = {}
        route_weights: dict[str, torch.Tensor] = {}
        strength: dict[str, torch.Tensor] = {}
        modality_outputs: dict[str, torch.Tensor] = {}
        raw_mixture: dict[str, torch.Tensor] = {}
        semantic_cosine: dict[str, torch.Tensor] = {}
        role_score: dict[str, torch.Tensor] = {}
        support_flag: dict[str, torch.Tensor] = {}
        router_cosine: dict[str, torch.Tensor] = {}
        router_reliability: dict[str, torch.Tensor] = {}
        receiver_mass = edge_weight.new_zeros((x.size(0),)).index_add(0, dst, edge_weight)
        for modality in ("text", "visual"):
            item = info["details"][modality]
            role = info["role_channels"][modality]
            prior[modality] = item["prior"].detach()
            raw_states[modality] = item["raw_states"].detach()
            raw_basis[modality] = item["raw_trajectory"].detach()
            denominators[modality] = item["raw_denominators"].detach()
            route_weights[modality] = info["router"][modality]["route_weights"].detach()
            strength[modality] = info["router"][modality]["strength"].detach()
            modality_outputs[modality] = item["output"].detach()
            raw_mixture[modality] = item["raw_mixture"].detach()
            semantic_cosine[modality] = role["semantic_cosine"].detach()
            role_score[modality] = role["role_score"].detach()
            support_flag[modality] = role["supportive_mask"].detach()
            router_state = model.router_norms[modality](
                model.router_projectors[modality](item["prior"])
            )
            norm_sq = router_state.square().sum(dim=-1, keepdim=True)
            safe_norm_sq = torch.where(norm_sq > 0, norm_sq, torch.ones_like(norm_sq))
            norm = torch.where(norm_sq > 0, safe_norm_sq.sqrt(), torch.zeros_like(norm_sq))
            normalized = router_state / (norm + model.eps)
            cos = (normalized[src] * normalized[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            router_cosine[modality] = cos.detach()
            router_reliability[modality] = (0.5 * (1.0 + cos)).clamp(0.0, 1.0).detach()

        del info
    return FrozenHostCache(
        dataset=dataset,
        model=model,
        classifier=classifier,
        src=src,
        dst=dst,
        edge_weight=edge_weight,
        active=active,
        indegree=indegree,
        prior=prior,
        raw_states=raw_states,
        raw_basis=raw_basis,
        denominators=denominators,
        alpha=model._effective_alpha(model.alpha_raw, model.eps).detach(),
        route_weights=route_weights,
        strength=strength,
        modality_outputs=modality_outputs,
        raw_mixture=raw_mixture,
        z=z.detach(),
        logits=logits.detach(),
        probabilities=probabilities.detach(),
        semantic_cosine=semantic_cosine,
        role_score=role_score,
        support_flag=support_flag,
        router_cosine=router_cosine,
        router_reliability=router_reliability,
        receiver_mass=receiver_mass.detach(),
        device=device,
    )


def _replay_modality(
    cache: FrozenHostCache, modality: str, receivers: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    model = cache.model
    profiles = torch.einsum("ek,bkd->bed", cache.alpha, basis)
    experts = torch.stack(
        [module(profiles[:, idx]) for idx, module in enumerate(model.experts)], dim=1
    )
    experts = profiles + model.expert_feature_scale * experts
    route = cache.route_weights[modality].index_select(0, receivers)
    strength = cache.strength[modality].index_select(0, receivers).reshape(-1, 1)
    mixture = (route.unsqueeze(-1) * experts).sum(dim=1)
    prior = cache.prior[modality].index_select(0, receivers)
    return prior + strength * mixture


def replay_local_basis_changes(
    cache: FrozenHostCache,
    receivers: torch.Tensor,
    changes: Sequence[dict[str, Any]] = (),
) -> dict[str, torch.Tensor]:
    """Replay selected receiver-local basis changes without rerunning propagation."""
    receivers = torch.as_tensor(receivers, dtype=torch.long, device=cache.device).reshape(-1)
    if not receivers.numel():
        raise ValueError("receiver batch must be non-empty")
    bases: dict[str, torch.Tensor] = {}
    for modality in ("text", "visual"):
        bases[modality] = cache.raw_basis[modality].index_select(1, receivers).transpose(0, 1).clone()
    for change in changes:
        modality = str(change["modality"])
        hop = int(change["hop"])
        if modality not in bases or hop < 1 or hop > 4:
            raise ValueError("change must specify modality text/visual and hop in 1..4")
        delta = torch.as_tensor(change["delta"], dtype=bases[modality].dtype, device=cache.device)
        if delta.shape != (receivers.numel(), bases[modality].size(-1)):
            raise ValueError("basis delta must have shape [batch, hidden_dim]")
        bases[modality][:, hop - 1] += delta

    modality_outputs = {
        modality: _replay_modality(cache, modality, receivers, bases[modality])
        for modality in ("text", "visual")
    }
    fused = torch.cat((modality_outputs["text"], modality_outputs["visual"]), dim=-1)
    z = cache.model.fusion_norm(
        cache.model.fusion_skip(fused)
        + cache.model.fusion_linear2(
            cache.model.fusion_dropout(F.gelu(cache.model.fusion_linear1(fused)))
        )
    )
    logits = cache.classifier(z)
    return {"modality_outputs": modality_outputs, "z": z, "logits": logits, "basis": bases}


def replay_singleton(
    cache: FrozenHostCache,
    receivers: torch.Tensor,
    modality: str,
    hop: int,
    delta_basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return replay_local_basis_changes(
        cache,
        receivers,
        [{"modality": modality, "hop": hop, "delta": delta_basis}],
    )


def replay_bundle(
    cache: FrozenHostCache,
    receivers: torch.Tensor,
    changes: Sequence[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    return replay_local_basis_changes(cache, receivers, changes)


def receiver_degree_quartile_sample(
    train_idx: torch.Tensor,
    indegree: torch.Tensor,
    *,
    max_receivers: int = 1500,
    seed: int = 2027,
) -> tuple[torch.Tensor, dict[int, float], dict[int, int]]:
    """Deterministic degree-stratified sample and its inclusion probabilities."""
    train_idx = torch.as_tensor(train_idx, dtype=torch.long).cpu().unique(sorted=True)
    indegree = torch.as_tensor(indegree, dtype=torch.long).cpu()
    candidates = train_idx[indegree[train_idx] > 0]
    if candidates.numel() == 0:
        return candidates, {}, {}
    order = torch.argsort(indegree[candidates], stable=True)
    ranked = candidates[order]
    strata = [part for part in torch.tensor_split(ranked, 4) if part.numel()]
    cap = min(int(max_receivers), ranked.numel())
    quotas = [cap // len(strata)] * len(strata)
    for i in range(cap % len(strata)):
        quotas[i] += 1
    selected: list[torch.Tensor] = []
    probabilities: dict[int, float] = {}
    counts: dict[int, int] = {}
    for stratum_id, (stratum, quota) in enumerate(zip(strata, quotas)):
        n = int(stratum.numel())
        take = min(n, quota)
        if take == n:
            chosen = stratum
        else:
            generator = torch.Generator().manual_seed(int(seed) + stratum_id)
            chosen = stratum[torch.randperm(n, generator=generator)[:take]]
        selected.append(chosen)
        prob = take / n
        for receiver in chosen.tolist():
            probabilities[int(receiver)] = prob
            counts[int(receiver)] = n
    return torch.cat(selected).sort().values, probabilities, counts


def sample_receiver_edges(
    dataset: str,
    receiver: int,
    dst: torch.Tensor,
    *,
    max_edges: int = 4,
    seed: int = 2027,
) -> tuple[torch.Tensor, float]:
    """Uniform deterministic without-replacement sample of incoming edges."""
    edge_ids = torch.nonzero(dst.cpu() == int(receiver), as_tuple=False).reshape(-1)
    degree = int(edge_ids.numel())
    if degree <= max_edges:
        return edge_ids, 1.0
    generator = torch.Generator().manual_seed(_stable_seed(f"{dataset}:{receiver}:{seed}"))
    chosen = edge_ids[torch.randperm(degree, generator=generator)[:max_edges]].sort().values
    return chosen, max_edges / degree


def estimator_receiver_split(
    dataset: str, receiver_ids: Sequence[int] | torch.Tensor, seed: int = 2027
) -> dict[str, set[int]]:
    """Stable hash split of sampled training receivers into disjoint partitions."""
    split = {"EstimatorTrain": set(), "EstimatorVal": set(), "EstimatorHoldout": set()}
    for receiver in sorted({int(x) for x in torch.as_tensor(receiver_ids).reshape(-1).tolist()}):
        value = _stable_seed(f"{dataset}:{receiver}:{seed}") % 100
        name = "EstimatorTrain" if value < 70 else ("EstimatorVal" if value < 85 else "EstimatorHoldout")
        split[name].add(receiver)
    return split


def singleton_basis_delta(
    message: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    try:
        result_shape = torch.broadcast_shapes(message.shape, denominator.shape)
    except RuntimeError as exc:
        raise ValueError("message and frozen denominator must be broadcast compatible") from exc
    if tuple(result_shape) != tuple(message.shape):
        raise ValueError("broadcasting the denominator would change the message batch shape")
    return -message / denominator


def compensated_basis_delta(
    raw_state: torch.Tensor,
    message: torch.Tensor,
    denominator: torch.Tensor,
    receiver_mass: float,
    edge_weight: float,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor | None, float | None, float | None, bool]:
    retained = float(receiver_mass) - float(edge_weight)
    if retained <= eps:
        return None, None, None, False
    scale = float(receiver_mass) / retained
    compensated = scale * (raw_state - message)
    delta = (compensated - raw_state) / denominator
    retained_mass = scale * retained
    return delta, scale, retained_mass, True


def loss_delta(logits: torch.Tensor, labels: torch.Tensor, baseline_loss: torch.Tensor) -> torch.Tensor:
    """Return singleton deletion effect: modified loss minus baseline loss."""
    return per_receiver_cross_entropy_delta(logits, labels, baseline_loss)


def per_receiver_cross_entropy_delta(
    logits: torch.Tensor, labels: torch.Tensor, baseline_loss: torch.Tensor
) -> torch.Tensor:
    labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device).reshape(-1)
    baseline_loss = torch.as_tensor(baseline_loss, dtype=logits.dtype, device=logits.device).reshape(-1)
    if logits.shape[0] != labels.numel() or baseline_loss.shape != labels.shape:
        raise ValueError("logits, labels, and baseline losses must have matching batch sizes")
    return F.cross_entropy(logits, labels, reduction="none") - baseline_loss


def bundle_interaction(bundle_delta: float, component_deltas: Sequence[float], eps: float = 1.0e-12) -> tuple[float, float]:
    interaction = float(bundle_delta) - sum(float(x) for x in component_deltas)
    relative = abs(interaction) / (sum(abs(float(x)) for x in component_deltas) + eps)
    return interaction, relative
