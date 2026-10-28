"""Independent full-graph, three-fold edge-holdout LP adapter for MR-CGE V1.3-A.

This deliberately does not opt the encoder into the repository's sampled LP
trainer. Each fold runs one global encoder forward on a Train-only physical
graph with the supervised fold removed, then accumulates decoder gradients
through a detached embedding leaf and backpropagates them into the encoder.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn.functional as F

from src.data import EdgeSplit, MAGData
from src.data.graph_utils import canonicalize_edges, edge_dict_to_index
from src.models import LinkPredictor, build_model
from src.tasks.common import clone_state_dict
from src.tasks.inference import infer_all_embeddings
from src.tasks.lp import _build_forbidden_edge_keys, _evaluate_split, _sample_filtered_negative_targets


PROTOCOL_VERSION = "mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is"
FOLD_COUNT = 3
FOLD_SEED = 42


class LPSplitConflictError(ValueError):
    """Raised when an official LP positive split violates the frozen leakage contract."""

    def __init__(self, message: str, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int, name: str, *, allow_self_loops: bool = False) -> torch.Tensor:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu").contiguous()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError(f"{name} contains out-of-range node IDs outside [0,{num_nodes})")
    if not allow_self_loops and edge_index.numel() and bool((edge_index[0] == edge_index[1]).any()):
        raise ValueError(f"{name} contains self-loop positive edges")
    return edge_index


def _edge_codes(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return edges[0].long() * int(num_nodes) + edges[1].long()


def _canonical_codes(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    canonical = canonicalize_edges(edges)
    return _edge_codes(canonical.t().contiguous(), num_nodes)


def validate_lp_split(
    edge_split: EdgeSplit, num_nodes: int, *, allow_official_conflicts: bool = False,
) -> dict[str, Any]:
    """Validate positive partitions and fixed Validation candidates, without reading Test negatives."""
    raw: dict[str, torch.Tensor] = {}
    canonical: dict[str, torch.Tensor] = {}
    duplicate_counts: dict[str, int] = {}
    for part in ("train", "valid", "test"):
        edges = _validate_edge_index(
            edge_dict_to_index(getattr(edge_split, part)), num_nodes, f"{part} positives", allow_self_loops=True,
        )
        raw[part] = edges
        pairs = torch.stack((torch.minimum(edges[0], edges[1]), torch.maximum(edges[0], edges[1])), dim=1)
        unique = torch.unique(pairs, dim=0)
        duplicate_counts[part] = int(edges.size(1) - unique.size(0))
        canonical[part] = unique
        if unique.numel() == 0:
            raise ValueError(f"LP split has no {part} positive edges")

    sets = {part: set(map(tuple, edges.tolist())) for part, edges in canonical.items()}
    overlaps = {
        "train_valid": len(sets["train"] & sets["valid"]),
        "train_test": len(sets["train"] & sets["test"]),
        "valid_test": len(sets["valid"] & sets["test"]),
    }
    self_loop_counts = {part: int((edges[0] == edges[1]).sum()) for part, edges in raw.items()}

    # Only the fixed Validation negatives are checked/used by this adapter.
    valid = edge_split.valid
    if "target_node_neg" not in valid:
        raise KeyError("official LP split is missing fixed Validation target_node_neg")
    valid_neg = torch.as_tensor(valid["target_node_neg"], dtype=torch.long, device="cpu")
    valid_src = torch.as_tensor(valid["source_node"], dtype=torch.long, device="cpu")
    if valid_neg.ndim != 2 or valid_neg.size(0) != valid_src.numel():
        raise ValueError("Validation negatives must have shape [num_valid_positive, num_negatives]")
    if valid_neg.numel() and (int(valid_neg.min()) < 0 or int(valid_neg.max()) >= num_nodes):
        raise ValueError("Validation negatives contain an out-of-range node ID")
    if bool((valid_neg == valid_src.view(-1, 1)).any()):
        raise ValueError("Validation negatives include a self-pair")

    audit = {
        "num_nodes": int(num_nodes),
        "positive_counts_raw": {key: int(value.size(1)) for key, value in raw.items()},
        "positive_counts_undirected_unique": {key: int(value.size(0)) for key, value in canonical.items()},
        "duplicate_positive_count_within_partition": duplicate_counts,
        "positive_partition_overlap_counts": overlaps,
        "self_loop_positive_counts": self_loop_counts,
        "validation_negative_shape": list(valid_neg.shape),
        "validation_negative_evaluation_only": True,
        "test_negative_candidates_read": False,
    }
    issues = []
    if any(self_loop_counts.values()):
        issues.append(f"self-loop positives={self_loop_counts}")
    if any(overlaps.values()):
        issues.append(f"undirected positive partitions overlap={overlaps}")
    if issues and not allow_official_conflicts:
        audit["protocol_conflict"] = True
        audit["conflicts"] = issues
        raise LPSplitConflictError("official LP split conflicts with the Train-only edge-holdout contract: " + "; ".join(issues), audit)
    audit["protocol_conflict"] = bool(issues)
    audit["conflicts"] = issues
    audit["official_split_conflicts_accepted_as_is"] = bool(issues and allow_official_conflicts)
    audit["leakage_safe_partition_claim"] = not bool(issues)
    return audit


def train_only_physical_graph(
    edge_split: EdgeSplit, num_nodes: int, *, allow_official_conflicts: bool = False,
) -> torch.Tensor:
    """Build a unique bidirectional physical graph from Train positives only."""
    train = _validate_edge_index(
        edge_dict_to_index(edge_split.train), num_nodes, "Train positives", allow_self_loops=True,
    )
    canonical = canonicalize_edges(train)
    if canonical.numel() == 0:
        raise ValueError("Train positive edge set is empty")
    directed = torch.cat((canonical.t(), canonical.flip(1).t()), dim=1)
    directed = torch.unique(directed.t().contiguous(), dim=0).t().contiguous()

    forbidden = torch.cat(
        [
            _validate_edge_index(edge_dict_to_index(edge_split.valid), num_nodes, "Validation positives", allow_self_loops=True),
            _validate_edge_index(edge_dict_to_index(edge_split.test), num_nodes, "Test positives", allow_self_loops=True),
        ], dim=1,
    )
    forbidden_codes = torch.unique(torch.cat((_edge_codes(forbidden, num_nodes), _edge_codes(forbidden.flip(0), num_nodes)))).sort().values
    message_codes = _edge_codes(directed, num_nodes)
    positions = torch.searchsorted(forbidden_codes, message_codes)
    hit = positions < forbidden_codes.numel()
    if bool(hit.any()):
        exact = torch.zeros_like(hit)
        exact[hit] = forbidden_codes[positions[hit]] == message_codes[hit]
        hit = exact
    if bool(hit.any()) and not allow_official_conflicts:
        raise ValueError("Train-only physical graph contains a Validation/Test positive edge direction")
    return directed


def _unique_undirected_positives(edges: torch.Tensor, num_nodes: int, name: str) -> torch.Tensor:
    """Canonicalize positive endpoint pairs while retaining official self-loop rows."""
    edges = _validate_edge_index(edges, num_nodes, name, allow_self_loops=True)
    pairs = torch.stack((torch.minimum(edges[0], edges[1]), torch.maximum(edges[0], edges[1])), dim=1)
    return torch.unique(pairs, dim=0)


def deterministic_train_folds(
    train_edges: torch.Tensor,
    *,
    num_nodes: int,
    fold_count: int = FOLD_COUNT,
    fold_seed: int = FOLD_SEED,
) -> list[torch.Tensor]:
    """Assign unique undirected positives once each; duplicate orientations share one fold."""
    if fold_count < 2:
        raise ValueError("fold_count must be >= 2")
    edges = _unique_undirected_positives(train_edges, num_nodes, "Train positives")
    if edges.size(0) < fold_count:
        raise ValueError(f"need at least {fold_count} unique Train positives")
    generator = torch.Generator(device="cpu").manual_seed(int(fold_seed))
    permutation = torch.randperm(edges.size(0), generator=generator)
    index_folds = torch.tensor_split(permutation, fold_count)
    folds = [edges[index].t().contiguous() for index in index_folds]
    seen: set[tuple[int, int]] = set()
    for fold in folds:
        for edge in fold.t().tolist():
            key = tuple(edge)
            if key in seen:
                raise RuntimeError(f"Train positive assigned to more than one fold: {key}")
            seen.add(key)
    if len(seen) != edges.size(0):
        raise RuntimeError("three-fold assignment did not cover every unique Train positive exactly once")
    return folds


def build_fold_message_graphs(
    train_edges: torch.Tensor,
    folds: list[torch.Tensor],
    *,
    num_nodes: int,
) -> list[torch.Tensor]:
    """Remove current fold positives and their reverse directions from Train-only propagation."""
    train_unique = _unique_undirected_positives(train_edges, num_nodes, "Train positives")
    # The repository's graph preprocessing removes self-loops from message passing.
    train_unique = train_unique[train_unique[:, 0] != train_unique[:, 1]]
    train_graph = torch.cat((train_unique.t(), train_unique.flip(1).t()), dim=1)
    train_graph = torch.unique(train_graph.t().contiguous(), dim=0).t().contiguous()
    graphs = []
    for fold_id, fold in enumerate(folds):
        holdout = _unique_undirected_positives(fold, num_nodes, f"fold {fold_id} positives")
        holdout_codes = _canonical_codes(holdout.t().contiguous(), num_nodes)
        graph_canonical = canonicalize_edges(train_graph)
        keep = ~torch.isin(_edge_codes(graph_canonical.t().contiguous(), num_nodes), holdout_codes)
        remaining = graph_canonical[keep]
        directed = torch.cat((remaining.t(), remaining.flip(1).t()), dim=1)
        directed = torch.unique(directed.t().contiguous(), dim=0).t().contiguous()
        message_codes = _edge_codes(directed, num_nodes)
        fold_directions = torch.cat((
            _edge_codes(holdout.t().contiguous(), num_nodes),
            _edge_codes(holdout.flip(1).t().contiguous(), num_nodes),
        ))
        sorted_codes = message_codes.sort().values
        position = torch.searchsorted(sorted_codes, fold_directions)
        present = position < sorted_codes.numel()
        if bool(present.any()):
            exact = torch.zeros_like(present)
            exact[present] = sorted_codes[position[present]] == fold_directions[present]
            present = exact
        if bool(present.any()):
            raise RuntimeError(f"fold {fold_id} supervised positive leaks into its physical message graph")
        graphs.append(directed)
    return graphs


def audit_loaded_message_graph(data: MAGData, train_graph: torch.Tensor, num_nodes: int) -> dict[str, Any]:
    loaded = torch.unique(data.edge_index.cpu().long().t().contiguous(), dim=0)
    expected = torch.unique(train_graph.cpu().long().t().contiguous(), dim=0)
    n = int(num_nodes)
    if not torch.equal(torch.sort(_edge_codes(loaded.t().contiguous(), n)).values,
                       torch.sort(_edge_codes(expected.t().contiguous(), n)).values):
        raise ValueError("loader's LP physical graph differs from the verified Train-only graph")
    return {"loaded_message_edges": int(data.edge_index.size(1)), "unique_train_only_directed_edges": int(expected.size(0))}


def _finite_gradients(parameters, label: str) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        raise RuntimeError(f"{label}: no gradients were produced")
    if not all(bool(torch.isfinite(grad).all()) for grad in grads):
        raise FloatingPointError(f"{label}: non-finite gradients")


def backward_fold_with_leaf(
    *,
    model,
    projection,
    decoder: LinkPredictor,
    z: torch.Tensor,
    aux_loss: torch.Tensor,
    supervision_edges: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    aux_weight: float,
    grad_clip: float,
    shuffle_generator: torch.Generator,
) -> dict[str, float]:
    """Backpropagate a fold mean BCE once plus one weighted encoder auxiliary loss."""
    if supervision_edges.ndim != 2 or supervision_edges.shape[0] != 2:
        raise ValueError("supervision_edges must have shape [2, N]")
    if labels.ndim != 1 or labels.numel() != supervision_edges.size(1):
        raise ValueError("labels must align with supervision edge columns")
    total = int(labels.numel())
    if total == 0:
        raise ValueError("fold supervision cannot be empty")
    optimizer.zero_grad(set_to_none=True)
    z_leaf = z.detach().requires_grad_(True)
    order = torch.randperm(total, generator=shuffle_generator)
    edge_order = supervision_edges[:, order].to(z.device)
    label_order = labels[order].to(z.device)
    bce_sum_value = 0.0
    for start in range(0, total, int(batch_size)):
        end = min(start + int(batch_size), total)
        projection_z = projection(z_leaf) if projection is not None else z_leaf
        edge_batch = edge_order[:, start:end]
        logits = decoder.score_pairs(projection_z[edge_batch[0]], projection_z[edge_batch[1]])
        loss_sum = F.binary_cross_entropy_with_logits(logits, label_order[start:end], reduction="sum")
        (loss_sum / float(total)).backward()
        bce_sum_value += float(loss_sum.detach().item())
        del projection_z, logits, loss_sum

    if z_leaf.grad is None:
        raise RuntimeError("detached embedding leaf did not collect Decoder gradients")
    if not bool(torch.isfinite(z_leaf.grad).all()):
        raise FloatingPointError("non-finite detached embedding gradient")
    encoder_grad = z_leaf.grad.detach()
    if aux_loss.requires_grad:
        torch.autograd.backward((z, aux_loss), (encoder_grad, torch.ones_like(aux_loss) * float(aux_weight)))
    else:
        torch.autograd.backward(z, encoder_grad)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    _finite_gradients(parameters, "full-graph fold backward")
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=float(grad_clip), error_if_nonfinite=True)
    optimizer.step()
    return {"bce": bce_sum_value / total, "aux": float(aux_loss.detach().item()), "supervision_examples": total}


def _model_and_heads(cfg, data: MAGData, cache: dict[str, Any], device: torch.device):
    info = {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes or 0),
        "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0,
    }
    model = build_model(cfg, info)
    if getattr(model, "supports_link_prediction", True) is not False:
        raise RuntimeError("LP adapter expects the frozen MR-CGE V1.3 NC-only capability declaration")
    if not bool(getattr(model, "requires_global_semantic_candidates", False)):
        raise RuntimeError("MR-CGE V1.3 global semantic candidate contract is missing")
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device)
    projection_dim = int(cfg.task.decoder.proj_dim)
    projection = torch.nn.Linear(model.out_dim, projection_dim).to(device)
    decoder = LinkPredictor(
        in_dim=projection_dim, hidden_dim=int(cfg.task.decoder.hidden_dim),
        num_layers=int(cfg.task.decoder.num_layers), dropout=float(cfg.task.decoder.dropout),
    ).to(device)
    return model, projection, decoder, info


def run_full_graph_lp(
    cfg,
    data: MAGData,
    cache: dict[str, Any],
    *,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_path: str,
    fingerprint_record: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Train one LP seed; Test metrics and Test negative candidates are never evaluated."""
    if data.edge_split is None:
        raise ValueError("LP input is missing the official edge split")
    if bool(cfg.task.get("evaluate_test", True)):
        raise ValueError("this adapter permanently disables Test evaluation")
    if str(cfg.task.protocol_version) != PROTOCOL_VERSION:
        raise ValueError("unexpected LP protocol version")
    seed = int(cfg.seed)
    allow_conflicts = bool(cfg.task.get("allow_official_split_conflicts", False))
    seed_split = validate_lp_split(data.edge_split, data.num_nodes, allow_official_conflicts=allow_conflicts)
    train_pos = edge_dict_to_index(data.edge_split.train).cpu()
    folds = deterministic_train_folds(train_pos, num_nodes=data.num_nodes)
    fold_graphs = build_fold_message_graphs(train_pos, folds, num_nodes=data.num_nodes)
    validation_graph = train_only_physical_graph(
        data.edge_split, data.num_nodes, allow_official_conflicts=allow_conflicts,
    )
    # The fixed loader graph is checked against Train positives; no held-out positive may be present.
    audit_loaded_message_graph(data, validation_graph, data.num_nodes)
    num_unique = sum(int(fold.size(1)) for fold in folds)
    fold_counts = [int(fold.size(1)) for fold in folds]
    if sum(fold_counts) != num_unique:
        raise RuntimeError("fold edge counts do not sum to unique Train positives")

    undirected = bool(data.edge_split.metadata.get("undirected", cfg.dataset.get("make_undirected", True)))
    forbidden_keys = _build_forbidden_edge_keys(data.edge_split, data.num_nodes, undirected=undirected)
    model, projection, decoder, data_info = _model_and_heads(cfg, data, cache, device)
    parameters = list(model.parameters()) + list(projection.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(cfg.task.lr), weight_decay=float(cfg.task.weight_decay))
    x = data.x.to(device)
    train_fold_device = [fold.to(device) for fold in folds]
    fold_graph_device = [graph.to(device) for graph in fold_graphs]
    aux_weight = float(cfg.task.loss.aux_weight)
    epochs = int(cfg.task.epochs)
    patience_limit = int(cfg.task.patience)
    min_epoch = int(cfg.task.early_stop_min_epoch)
    min_delta = float(cfg.task.early_stop_min_delta)
    eval_every = int(cfg.task.eval_every)
    batch_size = int(cfg.task.supervision_edge_batch_size)
    grad_clip = float(cfg.task.grad_clip)
    best_mrr = -1.0
    best_epoch: int | None = None
    best_metrics: dict[str, float] = {}
    best_states: dict[str, Any] | None = None
    patience_left = patience_limit
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train(); projection.train(); decoder.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        fold_losses = []
        for fold_id, (supervised_pos, message_graph) in enumerate(zip(train_fold_device, fold_graph_device)):
            gen = torch.Generator(device="cpu").manual_seed(seed + 10_000 + epoch * FOLD_COUNT + fold_id)
            neg = _sample_filtered_negative_targets(
                supervised_pos[0].cpu(), data.num_nodes, int(cfg.task.num_train_neg), forbidden_keys, gen,
            )
            neg_edges = torch.stack(
                [supervised_pos[0].cpu().repeat_interleave(int(cfg.task.num_train_neg)), neg.reshape(-1)], dim=0,
            )
            all_edges = torch.cat((supervised_pos.cpu(), neg_edges), dim=1)
            labels = torch.cat((torch.ones(supervised_pos.size(1)), torch.zeros(neg_edges.size(1)))).float()
            if bool((labels[supervised_pos.size(1):] != 0).any()):
                raise RuntimeError("negative sample labels are malformed")
            # The reused negative sampler rejects all known Train/Validation/Test positives,
            # but only its returned Train negatives enter supervision.
            optimizer.zero_grad(set_to_none=True)
            z, _, _, aux_loss, _ = model(x, message_graph)
            shuffle_gen = torch.Generator(device="cpu").manual_seed(seed + 20_000 + epoch * FOLD_COUNT + fold_id)
            fold_stats = backward_fold_with_leaf(
                model=model, projection=projection, decoder=decoder, z=z, aux_loss=aux_loss,
                supervision_edges=all_edges, labels=labels, optimizer=optimizer,
                batch_size=batch_size, aux_weight=aux_weight, grad_clip=grad_clip,
                shuffle_generator=shuffle_gen,
            )
            fold_losses.append(fold_stats["bce"] + aux_weight * fold_stats["aux"])
            del z, aux_loss
        train_loss = sum(fold_losses) / len(fold_losses)
        history.append({"epoch": epoch, "train_fold_objective_mean": train_loss})

        if epoch % eval_every:
            logger.info("Epoch %03d | Train Fold Objective %.5f", epoch, train_loss)
            continue
        model.eval(); projection.eval(); decoder.eval()
        with torch.no_grad():
            z = infer_all_embeddings(model, data, device, True, int(cfg.task.inference_batch_size), "full")
            z = projection(z.to(device)).detach().cpu()
            val_metrics = _evaluate_split(
                z, decoder, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size),
            )
        if not all(torch.isfinite(torch.tensor(value)) for value in val_metrics.values()):
            raise FloatingPointError(f"non-finite Validation metric at epoch {epoch}: {val_metrics}")
        history[-1].update({f"val_{key.replace('@', '_at_')}": float(value) for key, value in val_metrics.items()})
        logger.info("Epoch %03d | Train Fold Objective %.5f | Val MRR %.6f | Hits@1 %.6f | Hits@3 %.6f | Hits@10 %.6f",
                    epoch, train_loss, val_metrics["mrr"], val_metrics["hits@1"], val_metrics["hits@3"], val_metrics["hits@10"])
        if val_metrics["mrr"] > best_mrr + min_delta:
            best_mrr = float(val_metrics["mrr"])
            best_epoch = epoch
            best_metrics = {f"val_{key.replace('@', '_at_')}": float(value) for key, value in val_metrics.items()}
            best_states = {
                "model_state": clone_state_dict(model),
                "projection_state": clone_state_dict(projection),
                "decoder_state": clone_state_dict(decoder),
            }
            patience_left = patience_limit
        elif epoch >= min_epoch:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %03d", epoch)
                break

    if best_states is None or best_epoch is None:
        raise RuntimeError("LP run produced no Best-Validation-MRR checkpoint")
    model.load_state_dict(best_states["model_state"], strict=True)
    projection.load_state_dict(best_states["projection_state"], strict=True)
    decoder.load_state_dict(best_states["decoder_state"], strict=True)
    model.eval(); projection.eval(); decoder.eval()
    with torch.no_grad():
        z = infer_all_embeddings(model, data, device, True, int(cfg.task.inference_batch_size), "full")
        z = projection(z.to(device)).detach().cpu()
        reloaded_metrics = _evaluate_split(
            z, decoder, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size),
        )
    for key, value in best_metrics.items():
        metric_key = key.removeprefix("val_").replace("_at_", "@")
        if abs(float(value) - float(reloaded_metrics[metric_key])) > 3e-6:
            raise RuntimeError(f"reloaded LP checkpoint Validation {key} changed: {value} vs {reloaded_metrics[metric_key]}")

    metadata = {
        "protocol_version": PROTOCOL_VERSION, "model": "mrcge_v13", "condition": "M_repaired",
        "variant": "joint_view", "sender_private_ratio": 0.0, "seed": seed,
        "best_epoch": best_epoch, "selection": "best_validation_mrr", "validation_metrics": best_metrics,
        "checkpoint_reload_validation_metrics": {f"val_{k.replace('@', '_at_')}": float(v) for k, v in reloaded_metrics.items()},
        "fold_count": FOLD_COUNT, "fold_seed": FOLD_SEED, "fold_unique_positive_counts": fold_counts,
        "fold_graph_edge_counts_directed": [int(g.size(1)) for g in fold_graphs],
        "validation_graph_edge_count_directed": int(validation_graph.size(1)),
        "split_audit": seed_split, "fingerprints": copy.deepcopy(fingerprint_record),
        "training_source_commit": source_commit, "train_negative_ratio": int(cfg.task.num_train_neg),
        "train_negative_filter_scope": "all known positive Train/Validation/Test edges are used only to reject false-negative candidates; Test negative candidate lists and Test rankings are not accessed",
        "validation_message_graph": "full unique Train-positive physical graph; official Train/Validation/Test overlaps are retained when present",
        "fold_message_graph": "Train positives excluding current supervision fold and reverse directions; official partition overlaps are retained",
        "protocol_deviation": (
            "Official lp-edge-split.pt used unchanged despite self-loop positives and cross-partition positive overlap; "
            "Validation positives overlapping Train may appear in message passing and training supervision. "
            "Train self-loop positives remain in decoder supervision but, following repository preprocessing, are absent from message passing."
            if seed_split["protocol_conflict"] else "none"
        ),
        "official_split_conflicts_accepted_as_is": bool(seed_split["protocol_conflict"]),
        "model_semantic_knn": "global node-attribute V7B.1 K=8; no label-derived candidates",
        "test_evaluation": False, "test_metrics": "not_evaluated", "checkpoint_reload_verified": False,
        "optimizer": "Adam", "lr": float(cfg.task.lr), "weight_decay": float(cfg.task.weight_decay),
        "epoch_history": history,
    }
    checkpoint = {
        "task": "lp", "protocol_version": PROTOCOL_VERSION, "seed": seed,
        "selection": "best_validation_mrr", "epoch": best_epoch, "metrics": best_metrics,
        **best_states, "data_info": data_info, "metadata": metadata,
    }
    torch.save(checkpoint, checkpoint_path)
    serialized = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_model, reload_projection, reload_decoder, _ = _model_and_heads(cfg, data, cache, device)
    reload_model.load_state_dict(serialized["model_state"], strict=True)
    reload_projection.load_state_dict(serialized["projection_state"], strict=True)
    reload_decoder.load_state_dict(serialized["decoder_state"], strict=True)
    reload_model.eval(); reload_projection.eval(); reload_decoder.eval()
    with torch.no_grad():
        reload_z = infer_all_embeddings(reload_model, data, device, True, int(cfg.task.inference_batch_size), "full")
        reload_z = reload_projection(reload_z.to(device)).detach().cpu()
        disk_metrics = _evaluate_split(
            reload_z, reload_decoder, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size),
        )
    for key, value in best_metrics.items():
        metric_key = key.removeprefix("val_").replace("_at_", "@")
        if abs(float(value) - float(disk_metrics[metric_key])) > 3e-6:
            raise RuntimeError(f"strictly reloaded LP checkpoint Validation {key} changed: {value} vs {disk_metrics[metric_key]}")
    metadata["checkpoint_reload_verified"] = True
    checkpoint["metadata"] = metadata
    torch.save(checkpoint, checkpoint_path)
    return {
        "dataset": str(data.name), "seed": seed, "protocol_version": PROTOCOL_VERSION,
        "status": "complete", "best_epoch": best_epoch, **best_metrics,
        "checkpoint_reload_verified": True, "test_evaluation": False, "test_metrics": "not_evaluated",
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": "",
        "semantic_cache_sha256": fingerprint_record.get("semantic_cache_sha256", ""),
        "split_sha256": fingerprint_record.get("split_sha256", ""),
        "train_graph_fingerprint": fingerprint_record.get("train_graph_fingerprint", ""),
        "feature_fingerprint": fingerprint_record.get("feature_fingerprint", ""),
        "candidate_fingerprint": fingerprint_record.get("candidate_fingerprint", ""),
        "v7b_fingerprint": fingerprint_record.get("v7b_fingerprint", ""),
        "fold_seed": FOLD_SEED, "fold_count": FOLD_COUNT, "source_commit": source_commit,
        "train_negative_filter_scope": metadata["train_negative_filter_scope"],
        "split_protocol_deviation": metadata["protocol_deviation"],
        "split_conflict_summary": seed_split["positive_partition_overlap_counts"],
    }
