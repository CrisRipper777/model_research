from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from src.analysis.mechanism_discovery import cosine_flat, linear_cka
from src.analysis.problem_deep_dive import normalized_physical_operator
from src.analysis.r3mag_response_core import (
    GlobalDualBranchResponseHost,
    R3MAGHostConfig,
    action_names,
    make_action_candidates,
    normalized_response_directions,
)
from src.data import load_mag_data
from src.utils.seeds import set_seed


ROOT = Path(__file__).resolve().parents[2]
PARTITION_SEED = 20261006
BOOTSTRAP_REPEATS = 2000
CONTROL_SEEDS = [20261006 + 104729 * i for i in range(20)]
DATA_SEED = 42
EPSILON_VALUES = (0.1, 0.2)
DEFAULT_HOST_SEEDS = (42, 43, 44)
DEGREE_BUCKETS = 4
ROBUST_MARGIN_THRESHOLD = 0.01
PRACTICAL_CE_EXCESS_THRESHOLD = 0.01


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_sha256(index: torch.Tensor) -> str:
    value = torch.as_tensor(index, dtype=torch.int64).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _allocation_counts(class_count: int) -> tuple[int, int, int]:
    """Hamilton allocation for HostTrain/ResponseTrain/Audit, keeping host nonempty."""
    if class_count <= 0:
        return 0, 0, 0
    ratios = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
    quotas = class_count * ratios
    counts = np.floor(quotas).astype(np.int64)
    remainder = class_count - int(counts.sum())
    priority = sorted(range(3), key=lambda i: (-float(quotas[i] - counts[i]), i))
    for i in priority[:remainder]:
        counts[i] += 1
    if counts[0] == 0:
        donor = int(np.argmax(counts[1:]) + 1)
        if counts[donor] > 0:
            counts[donor] -= 1
            counts[0] += 1
    return tuple(int(x) for x in counts)


def stratified_internal_partition(
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    seed: int = PARTITION_SEED,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Partition only original train indices; labels are used solely as strata."""
    ids = torch.as_tensor(train_idx, dtype=torch.long).detach().cpu().flatten()
    y = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    classes = y[ids]
    if bool((classes < 0).any()):
        raise ValueError("Original train_idx contains an unlabeled class; stratified H1 split is undefined")
    generator = np.random.default_rng(int(seed))
    pieces: dict[str, list[np.ndarray]] = {"host_train": [], "response_train": [], "audit": []}
    class_allocations: dict[str, dict[str, int]] = {}
    fallback_classes: list[int] = []
    for class_id in sorted(int(x) for x in torch.unique(classes).tolist()):
        class_ids = ids[classes == class_id].numpy()
        shuffled = generator.permutation(class_ids)
        n_host, n_response, n_audit = _allocation_counts(len(class_ids))
        if len(class_ids) < 10:
            fallback_classes.append(class_id)
        pieces["host_train"].append(shuffled[:n_host])
        pieces["response_train"].append(shuffled[n_host : n_host + n_response])
        pieces["audit"].append(shuffled[n_host + n_response :])
        class_allocations[str(class_id)] = {
            "original_train": int(len(class_ids)),
            "host_train": n_host,
            "response_train": n_response,
            "audit": n_audit,
            "small_class_fallback": len(class_ids) < 10,
        }
    partitions = {
        key: torch.as_tensor(np.concatenate(values) if values else [], dtype=torch.long).sort().values
        for key, values in pieces.items()
    }
    expected = set(ids.tolist())
    actual = set(torch.cat(list(partitions.values())).tolist())
    if expected != actual:
        raise AssertionError("Internal partition does not exactly cover original train_idx")
    metadata = {
        "partition_seed": int(seed),
        "method": "per-class deterministic NumPy Generator permutation + Hamilton largest remainder (80/10/10); host count at least one",
        "class_allocations": class_allocations,
        "small_class_fallback_classes": fallback_classes,
        "fallback_rule": "classes with fewer than 10 original-train examples use the same Hamilton allocation; HostTrain is guaranteed at least one, other pieces may be empty",
    }
    return partitions, metadata


def _class_histogram(labels: torch.Tensor, indices: torch.Tensor) -> dict[str, int]:
    selected = torch.as_tensor(labels, dtype=torch.long).detach().cpu()[indices.cpu()]
    counts = Counter(int(x) for x in selected.tolist())
    return {str(k): int(counts[k]) for k in sorted(counts)}


def _partition_histogram_from_allocation(
    allocation: dict[str, Any], partition: str
) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, values in allocation["class_allocations"].items():
        count = int(values[partition])
        if count:
            result[label] = count
    return result


def assert_disjoint_splits(splits: dict[str, torch.Tensor]) -> None:
    names = list(splits)
    sets = {name: set(torch.as_tensor(idx).cpu().tolist()) for name, idx in splits.items()}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sets[left].intersection(sets[right])
            if overlap:
                raise AssertionError(f"Split overlap between {left} and {right}: {len(overlap)} indices")


def _compose_dataset(dataset: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[f"dataset={dataset}", "task=nc", "model=mlp", f"seed={DATA_SEED}", "num_runs=1"],
        )
    return cfg


def load_fixed_data(dataset: str):
    cfg = _compose_dataset(dataset)
    data = load_mag_data(cfg, "nc", seed=DATA_SEED)
    if data.train_idx is None or data.val_idx is None or data.test_idx is None or data.y is None:
        raise ValueError(f"{dataset}: NC data is missing labels or split indices")
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset}: separate text/visual features are required")
    split_source = data.info.get("nc_split_path", data.info.get("node_split_path", "unknown"))
    return data, Path(split_source), cfg


def _operator(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    return normalized_physical_operator(edge_index, num_nodes, device=device).coalesce()


def _validate_label_boundary(
    data,
    partitions: dict[str, torch.Tensor],
    split_hashes: dict[str, str],
) -> None:
    assert_disjoint_splits(
        {
            "HostTrain": partitions["host_train"],
            "ResponseTrain": partitions["response_train"],
            "Audit": partitions["audit"],
            "Val": data.val_idx,
            "Test": data.test_idx,
        }
    )
    if "Test" not in split_hashes:
        raise AssertionError("Test index metadata must be preserved")


def _train_host(
    data,
    operator: torch.Tensor,
    partitions: dict[str, torch.Tensor],
    host_seed: int,
    device: torch.device,
    output_path: Path,
    *,
    max_epochs: int = 1000,
    patience: int = 100,
    min_epoch: int = 30,
    min_delta: float = 1e-4,
) -> tuple[GlobalDualBranchResponseHost, dict[str, Any]]:
    set_seed(int(host_seed))
    config = R3MAGHostConfig()
    model = GlobalDualBranchResponseHost(
        int(data.x_t.size(1)), int(data.x_i.size(1)), int(data.num_classes), config
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    x_text = data.x_t.to(device)
    x_visual = data.x_i.to(device)
    train_idx = partitions["host_train"].to(device)
    val_idx = data.val_idx.to(device)
    # Only these two label slices are made available to host fitting/selection.
    host_labels = data.y[partitions["host_train"]].long().to(device)
    val_labels = data.y[data.val_idx].long().to(device)
    best_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val: dict[str, float] | None = None
    patience_left = int(patience)

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        forward_output = model(x_text, x_visual, operator)
        logits = forward_output[0]
        loss = F.cross_entropy(logits[train_idx], host_labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite HostTrain CE at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        del forward_output, logits, loss

        model.eval()
        with torch.no_grad():
            val_logits, _, _, _, _ = model(x_text, x_visual, operator)
            selected = val_logits[val_idx]
            val_loss = F.cross_entropy(selected, val_labels)
            predictions = selected.argmax(dim=-1)
            val_acc = float((predictions == val_labels).float().mean().item())
            val_f1 = float(
                f1_score(
                    val_labels.detach().cpu().numpy(),
                    predictions.detach().cpu().numpy(),
                    labels=list(range(int(data.num_classes))),
                    average="macro",
                    zero_division=0,
                )
            )
            val_ce = float(val_loss.item())
        del val_logits, selected

        if val_acc > best_acc + float(min_delta):
            best_acc = val_acc
            best_epoch = epoch
            best_val = {"val_accuracy": val_acc, "val_macro_f1": val_f1, "val_ce": val_ce}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(patience)
            if epoch == 1 or epoch % 25 == 0:
                print(
                    f"[{data.name} seed={host_seed}] epoch={epoch} val_acc={val_acc:.5f} val_macro_f1={val_f1:.5f} val_ce={val_ce:.5f}",
                    flush=True,
                )
        else:
            if epoch >= int(min_epoch):
                patience_left -= 1
            if epoch % 25 == 0:
                print(
                    f"[{data.name} seed={host_seed}] epoch={epoch} val_acc={val_acc:.5f} best={best_acc:.5f} patience_left={patience_left}",
                    flush=True,
                )
            if epoch >= int(min_epoch) and patience_left <= 0:
                break

    if best_state is None or best_val is None:
        raise RuntimeError("Host produced no validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        logits, states_text, states_visual, global_text, global_visual = model(
            x_text, x_visual, operator
        )
        # Reloaded best checkpoint is rechecked against its stored validation selection.
        selected = logits[val_idx]
        checkpoint_val = {
            "val_accuracy": float((selected.argmax(-1) == val_labels).float().mean().item()),
            "val_ce": float(F.cross_entropy(selected, val_labels).item()),
        }
    run = {
        "host_seed": int(host_seed),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "selection": "best original-val accuracy; improvements must exceed main NC early_stop_min_delta=1e-4, and sub-threshold changes retain the earlier checkpoint",
        "early_stopping": {
            "min_epoch": int(min_epoch),
            "min_delta": float(min_delta),
            "patience": int(patience),
        },
        **best_val,
        "restored_checkpoint_val_accuracy": checkpoint_val["val_accuracy"],
        "restored_checkpoint_val_ce": checkpoint_val["val_ce"],
        "learned_gamma_text": model.gamma_text.detach().cpu().tolist(),
        "learned_gamma_visual": model.gamma_visual.detach().cpu().tolist(),
        "ppr_initialization": {
            "formula": "gamma_k = alpha * (1-alpha)^k",
            "alpha_restart": config.ppr_restart,
            "max_order": config.max_order,
            "initial": config.gamma_init().tolist(),
            "truncated_sum": float(config.gamma_init().sum().item()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": data.name,
            "host_seed": int(host_seed),
            "selection": run["selection"],
            "best_epoch": best_epoch,
            "metrics": best_val,
            "model_state": best_state,
            "config": asdict(config),
            "learned_gamma_text": run["learned_gamma_text"],
            "learned_gamma_visual": run["learned_gamma_visual"],
        },
        output_path,
    )
    run["checkpoint_path"] = str(output_path)
    run["checkpoint_bytes"] = output_path.stat().st_size
    # Keep the frozen graph states on the returned model's next audit call; avoid retaining logits.
    del logits, states_text, states_visual, global_text, global_visual, selected
    return model, run


def _basis_diagnostics(
    states: dict[str, list[torch.Tensor]],
    gamma: dict[str, torch.Tensor],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for modality in ("text", "visual"):
        values = [state.detach().float().cpu() for state in states[modality]]
        pairs = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                pairs.append(
                    {
                        "order_i": i,
                        "order_j": j,
                        "linear_cka": linear_cka(values[i], values[j]),
                        "flattened_cosine": cosine_flat(values[i], values[j]),
                    }
                )
        out[modality] = {
            "mean_norm_by_order": [float(x.norm(dim=-1).mean().item()) for x in values],
            "pairwise_similarity": pairs,
            "learned_gamma": gamma[modality].detach().cpu().tolist(),
        }
    return out


def _degree_and_prediction_buckets(
    edge_index: torch.Tensor, num_nodes: int, baseline_predictions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = edge_index[0].detach().cpu().long()
    degree = torch.bincount(src, minlength=num_nodes).float()
    cuts = torch.quantile(degree, torch.tensor([0.25, 0.5, 0.75]))
    degree_bucket = torch.bucketize(degree, cuts, right=False).long()
    return degree, degree_bucket, baseline_predictions.detach().cpu().long()


def _bucket_key(degree_bucket: torch.Tensor, predictions: torch.Tensor, node: int, tier: str):
    if tier == "joint":
        return int(degree_bucket[node]), int(predictions[node])
    if tier == "degree":
        return int(degree_bucket[node])
    return 0


def _assign_donors(
    targets: list[int], pool: np.ndarray, rng: np.random.Generator
) -> tuple[dict[int, int], int]:
    """Random one-to-one assignment where possible, always avoiding self when possible."""
    targets = [int(x) for x in targets]
    candidates = [int(x) for x in pool.tolist()]
    if not targets or not candidates:
        return {}, 0
    if len(candidates) >= len(targets):
        for _ in range(100):
            shuffled = rng.permutation(candidates).tolist()
            assigned = dict(zip(targets, shuffled[: len(targets)], strict=True))
            if all(assigned[t] != t for t in targets):
                return assigned, 0
        # A cyclic shift of a random pool order is a guaranteed derangement when
        # the target set is exactly that pool and has size >= 2.
        if len(candidates) >= 2 and set(targets) == set(candidates):
            perm = rng.permutation(candidates).tolist()
            shift = int(rng.integers(1, len(perm)))
            return dict(zip(perm, perm[shift:] + perm[:shift], strict=True)), 0
    assigned: dict[int, int] = {}
    used: set[int] = set()
    fallback_count = 0
    order = rng.permutation(targets).tolist()
    for target in order:
        available = [x for x in candidates if x != target and x not in used]
        reused = False
        if not available:
            available = [x for x in candidates if x != target]
            reused = True
        if not available:
            # Singleton pools are the only case where j != i cannot be met.
            available = candidates
            reused = True
        donor = int(available[int(rng.integers(0, len(available)))])
        assigned[target] = donor
        used.add(donor)
        if reused:
            fallback_count += 1
    duplicate_count = len(assigned) - len(set(assigned.values()))
    self_count = sum(1 for target, donor in assigned.items() if target == donor)
    return assigned, fallback_count + duplicate_count + self_count


def matched_control_donors(
    audit_idx: torch.Tensor,
    degree_bucket: torch.Tensor,
    predicted_class: torch.Tensor,
    num_nodes: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Return donor ids [audit nodes, modalities, orders] with coarse matched buckets."""
    audit = audit_idx.detach().cpu().long().tolist()
    all_nodes = np.arange(num_nodes, dtype=np.int64)
    joint_pool_map: dict[tuple[int, int], np.ndarray] = {}
    degree_pool_map: dict[int, np.ndarray] = {}
    joint_lists: dict[tuple[int, int], list[int]] = defaultdict(list)
    degree_lists: dict[int, list[int]] = defaultdict(list)
    for node in all_nodes.tolist():
        joint_lists[_bucket_key(degree_bucket, predicted_class, node, "joint")].append(node)
        degree_lists[_bucket_key(degree_bucket, predicted_class, node, "degree")].append(node)
    joint_pool_map = {key: np.asarray(value, dtype=np.int64) for key, value in joint_lists.items()}
    degree_pool_map = {key: np.asarray(value, dtype=np.int64) for key, value in degree_lists.items()}
    donors = torch.empty((len(audit), 2, 4), dtype=torch.long)
    counts: Counter[str] = Counter()
    # Two modalities and four response orders each use independently seeded shuffles.
    for modality in range(2):
        for order in range(4):
            rng = np.random.default_rng(int(seed) + 1009 * modality + 9176 * order)
            strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
            for target in audit:
                joint_key = _bucket_key(degree_bucket, predicted_class, target, "joint")
                joint_pool = joint_pool_map.get(joint_key, np.empty(0, dtype=np.int64))
                if len(joint_pool) >= 2:
                    tier = "joint"
                    key = joint_key
                else:
                    degree_key = _bucket_key(degree_bucket, predicted_class, target, "degree")
                    degree_pool = degree_pool_map.get(degree_key, np.empty(0, dtype=np.int64))
                    if len(degree_pool) >= 2:
                        tier = "degree"
                        key = degree_key
                    else:
                        tier = "global"
                        key = (0,)
                strata[(tier, key)].append(target)

            donor_by_target: dict[int, int] = {}
            for (tier, key), targets in strata.items():
                if tier == "joint":
                    pool = joint_pool_map[key]
                elif tier == "degree":
                    pool = degree_pool_map[key]
                else:
                    pool = all_nodes
                mapping, repeated_or_self = _assign_donors(targets, pool, rng)
                donor_by_target.update(mapping)
                counts[tier] += len(targets)
                counts["donor_reuse_or_singleton"] += repeated_or_self
            for row, target in enumerate(audit):
                donors[row, modality, order] = int(donor_by_target[target])
    if donors.shape != (len(audit), 2, 4):
        raise AssertionError("Matched control donor array shape changed")
    self_count = int((donors == torch.tensor(audit)[:, None, None]).sum().item())
    counts["self_donor_count"] += self_count
    return donors, dict(counts)


def _ce_utility(logits: torch.Tensor, labels: torch.Tensor, baseline_ce: torch.Tensor) -> torch.Tensor:
    candidate_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    return baseline_ce - candidate_ce


@torch.no_grad()
def evaluate_single_action_utilities(
    model: GlobalDualBranchResponseHost,
    candidates_text: torch.Tensor,
    candidates_visual: torch.Tensor,
    global_text: torch.Tensor,
    global_visual: torch.Tensor,
    labels: torch.Tensor,
    baseline_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(labels.numel(), device=labels.device)
    baseline_ce = F.cross_entropy(baseline_logits, labels, reduction="none")
    text_logits = model.classify_responses(
        candidates_text, global_visual[:, None, :].expand_as(candidates_text)
    )
    visual_logits = model.classify_responses(
        global_text[:, None, :].expand_as(candidates_visual), candidates_visual
    )
    labels_actions = labels[:, None].expand(-1, candidates_text.size(1))
    utility_text = _ce_utility(text_logits, labels_actions, baseline_ce[:, None])
    utility_visual = _ce_utility(visual_logits, labels_actions, baseline_ce[:, None])
    return utility_text, utility_visual, text_logits, visual_logits


@torch.no_grad()
def evaluate_joint_action_utilities(
    model: GlobalDualBranchResponseHost,
    candidates_text: torch.Tensor,
    candidates_visual: torch.Tensor,
    global_text: torch.Tensor,
    global_visual: torch.Tensor,
    labels: torch.Tensor,
    baseline_logits: torch.Tensor,
    *,
    node_batch_size: int = 512,
) -> torch.Tensor:
    n, n_actions, hidden = candidates_text.shape
    if candidates_visual.shape != (n, n_actions, hidden):
        raise ValueError("Text and visual action candidate shapes must match")
    baseline_ce = F.cross_entropy(baseline_logits, labels, reduction="none")
    utilities = torch.empty((n, n_actions, n_actions), device=labels.device, dtype=baseline_logits.dtype)
    for start in range(0, n, int(node_batch_size)):
        end = min(start + int(node_batch_size), n)
        ct = candidates_text[start:end]
        cv = candidates_visual[start:end]
        b = end - start
        text = ct[:, :, None, :].expand(b, n_actions, n_actions, hidden)
        visual = cv[:, None, :, :].expand(b, n_actions, n_actions, hidden)
        logits = model.classify_responses(text, visual)
        repeated_labels = labels[start:end, None, None].expand(b, n_actions, n_actions)
        utilities[start:end] = _ce_utility(
            logits,
            repeated_labels,
            baseline_ce[start:end, None, None],
        )
    return utilities


def _bootstrap_ci(values: np.ndarray, seed: int, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(len(x), np.full(len(x), 1.0 / len(x)), size=int(repeats)).astype(np.float64)
    estimates = weights @ x / len(x)
    return {
        "mean": float(x.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_repeats": int(repeats),
    }


def _bootstrap_node_headroom_excess(
    structural: np.ndarray,
    controls: np.ndarray,
    seed: int,
    repeats: int,
) -> dict[str, float]:
    n = structural.shape[0]
    if controls.ndim != 3 or controls.shape[1:] != structural.shape:
        raise ValueError("control utilities must have shape [repeats, nodes, actions]")
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=int(repeats)).astype(np.float64)
    structural_means = weights @ structural / n
    structural_oracles = weights @ structural.max(axis=1) / n
    structural_headroom = structural_oracles - structural_means.max(axis=1)
    control_heads = np.empty((int(repeats), controls.shape[0]), dtype=np.float64)
    for r in range(controls.shape[0]):
        means = weights @ controls[r] / n
        oracle = weights @ controls[r].max(axis=1) / n
        control_heads[:, r] = oracle - means.max(axis=1)
    estimates = structural_headroom - control_heads.mean(axis=1)
    full_struct = structural.max(axis=1).mean() - structural.mean(axis=0).max()
    full_controls = np.asarray(
        [ctrl.max(axis=1).mean() - ctrl.mean(axis=0).max() for ctrl in controls], dtype=np.float64
    ).mean()
    return {
        "mean": float(full_struct - full_controls),
        "structural_headroom": float(full_struct),
        "shuffled_headroom_mean": float(full_controls),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_repeats": int(repeats),
    }


def _paired_mean_summary(structural: np.ndarray, controls: np.ndarray, seed: int, repeats: int) -> dict[str, Any]:
    # controls is [control_repeats, audit_nodes].
    paired_difference = structural - controls.mean(axis=0)
    return _bootstrap_ci(paired_difference, seed, repeats)


def _preferred_action_summary(utilities: np.ndarray, names: list[str]) -> dict[str, Any]:
    order = np.argsort(utilities, axis=1)
    best = order[:, -1]
    second = order[:, -2]
    margins = utilities[np.arange(len(utilities)), best] - utilities[np.arange(len(utilities)), second]
    counts = Counter(names[int(x)] for x in best.tolist())
    return {
        "positive_best_fraction": float((utilities.max(axis=1) > 0.0).mean()),
        "top1_top2_margin_mean": float(margins.mean()),
        "top1_top2_margin_median": float(np.median(margins)),
        "preferred_action_counts": {name: int(counts.get(name, 0)) for name in names},
        "preferred_action_fraction": {name: float(counts.get(name, 0) / len(best)) for name in names},
        "best_action_index": best,
        "margin": margins,
    }


def _utility_profile_spearman(text_u: np.ndarray, visual_u: np.ndarray) -> dict[str, float | int | None]:
    values = []
    for text_row, visual_row in zip(text_u, visual_u, strict=True):
        if np.ptp(text_row) <= 1e-12 or np.ptp(visual_row) <= 1e-12:
            continue
        value = spearmanr(text_row, visual_row).statistic
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"n_nonconstant_profiles": 0, "mean": None, "median": None}
    return {"n_nonconstant_profiles": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}


def _match_metrics(utilities: np.ndarray) -> dict[str, float]:
    per_node_oracle = utilities.max(axis=1)
    global_mean = utilities.mean(axis=0)
    headroom = float(per_node_oracle.mean() - global_mean.max())
    return {
        "node_oracle_mean": float(per_node_oracle.mean()),
        "best_global_action_mean": float(global_mean.max()),
        "node_specific_headroom": headroom,
    }


def _joint_action_summary(joint_u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n = joint_u.shape[0]
    separate_indices = joint_u.reshape(n, -1).argmax(axis=1)
    best_separate = joint_u.reshape(n, -1).max(axis=1)
    shared = np.diagonal(joint_u, axis1=1, axis2=2)
    shared_indices = shared.argmax(axis=1)
    best_shared = shared.max(axis=1)
    text_action = separate_indices // joint_u.shape[2]
    visual_action = separate_indices % joint_u.shape[2]
    text_changes = text_action != 0
    visual_changes = visual_action != 0
    same = text_action == visual_action
    metrics = {
        "separate_headroom": float((best_separate - best_shared).mean()),
        "best_joint_only_text_fraction": float((text_changes & ~visual_changes).mean()),
        "best_joint_only_visual_fraction": float((visual_changes & ~text_changes).mean()),
        "best_joint_both_changed_different_fraction": float((text_changes & visual_changes & ~same).mean()),
        "best_joint_same_action_including_noop_fraction": float(same.mean()),
        "best_joint_both_changed_same_nonnoop_fraction": float((text_changes & visual_changes & same).mean()),
        "best_joint_noop_noop_fraction": float(((text_action == 0) & (visual_action == 0)).mean()),
        "best_shared_action_counts": {
            str(i): int((shared_indices == i).sum()) for i in range(joint_u.shape[1])
        },
    }
    return best_separate, best_shared, best_separate - best_shared, metrics


def _node_bootstrap_headroom_excess(
    structural_gap: np.ndarray,
    shuffled_gap: np.ndarray,
    seed: int,
    repeats: int,
) -> dict[str, float]:
    # Paired target-level separate-minus-shared gaps, averaged over controls per target.
    return _paired_mean_summary(structural_gap, shuffled_gap, seed, repeats)


def _vectorized_bruteforce_qa(
    model: GlobalDualBranchResponseHost,
    cand_text: torch.Tensor,
    cand_visual: torch.Tensor,
    global_text: torch.Tensor,
    global_visual: torch.Tensor,
    labels: torch.Tensor,
    baseline_logits: torch.Tensor,
    utility_text: torch.Tensor,
    utility_visual: torch.Tensor,
    joint_utility: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    n = int(labels.numel())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sampled = torch.randperm(n, generator=generator)[: min(10, n)].to(labels.device)
    action_ids = sorted(set([0, 1, cand_text.size(1) // 2, cand_text.size(1) - 1]))
    joint_pairs = [(0, 0), (1, cand_text.size(1) - 1), (cand_text.size(1) // 2, cand_text.size(1) // 2), (cand_text.size(1) - 1, 1)]
    max_logit_error = 0.0
    max_ce_error = 0.0
    max_joint_logit_error = 0.0
    max_joint_ce_error = 0.0
    with torch.no_grad():
        for action in action_ids:
            batch_t = model.classify_responses(
                cand_text[sampled, action], global_visual[sampled]
            )
            batch_v = model.classify_responses(
                global_text[sampled], cand_visual[sampled, action]
            )
            for pos, node_tensor in enumerate(sampled):
                node = int(node_tensor.item())
                brute_t = model.classify_responses(cand_text[node, action], global_visual[node])
                brute_v = model.classify_responses(global_text[node], cand_visual[node, action])
                max_logit_error = max(
                    max_logit_error,
                    float((batch_t[pos] - brute_t).abs().max().item()),
                    float((batch_v[pos] - brute_v).abs().max().item()),
                )
        for node_tensor in sampled:
            node = int(node_tensor.item())
            for action in action_ids:
                logits_t = model.classify_responses(cand_text[node, action], global_visual[node])
                logits_v = model.classify_responses(global_text[node], cand_visual[node, action])
                ce_base = F.cross_entropy(baseline_logits[node : node + 1], labels[node : node + 1])
                for logits, expected_u in (
                    (logits_t, utility_text[node, action]),
                    (logits_v, utility_visual[node, action]),
                ):
                    actual_ce = F.cross_entropy(logits.reshape(1, -1), labels[node : node + 1])
                    max_ce_error = max(max_ce_error, abs(float((ce_base - actual_ce - expected_u).item())))
            for at, av in joint_pairs:
                brute = model.classify_responses(cand_text[node, at], cand_visual[node, av])
                brute_u = F.cross_entropy(baseline_logits[node : node + 1], labels[node : node + 1]) - F.cross_entropy(
                    brute.reshape(1, -1), labels[node : node + 1]
                )
                max_joint_ce_error = max(max_joint_ce_error, abs(float((brute_u - joint_utility[node, at, av]).item())))
    # The scalar CE checks above validate utility parity; compare a deterministic batched
    # candidate subset directly to row-wise inference logits as a separate vectorization check.
    qa_nodes = sampled[: min(10, sampled.numel())]
    batch_pair_logits = []
    for at, av in joint_pairs:
        batch_pair_logits.append(
            model.classify_responses(cand_text[qa_nodes, at], cand_visual[qa_nodes, av])
        )
    batched_joint = torch.stack(batch_pair_logits, dim=1)
    brute_joint = _exact_joint_logits_for_pairs(model, cand_text, cand_visual, qa_nodes, joint_pairs)
    brute_joint = brute_joint.reshape(qa_nodes.numel(), len(joint_pairs), -1)
    max_joint_logit_error = float((batched_joint - brute_joint).abs().max().item())
    return {
        "sampled_nodes": [int(x) for x in sampled.detach().cpu().tolist()],
        "single_action_ids_checked": action_ids,
        "joint_action_pairs_checked": [list(x) for x in joint_pairs],
        "max_vectorized_vs_bruteforce_logit_abs_error": max_logit_error,
        "max_vectorized_vs_bruteforce_ce_abs_error": max(max_ce_error, max_joint_ce_error),
        "max_joint_vectorized_vs_bruteforce_logit_abs_error": max_joint_logit_error,
        "max_joint_vectorized_vs_bruteforce_ce_abs_error": max_joint_ce_error,
    }


def _exact_joint_logits_for_pairs(
    model: GlobalDualBranchResponseHost,
    cand_text: torch.Tensor,
    cand_visual: torch.Tensor,
    nodes: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    values = []
    for node in nodes.tolist():
        for at, av in pairs:
            values.append(model.classify_responses(cand_text[node, at], cand_visual[node, av]))
    return torch.stack(values, dim=0)


def _direction_candidates(
    global_response: torch.Tensor,
    states: list[torch.Tensor],
    audit_idx: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    directions, raw_norm = normalized_response_directions(states, global_response)
    counts = int((raw_norm <= 1e-12).sum().item())
    local_global = global_response[audit_idx]
    local_directions = directions[audit_idx]
    candidates = make_action_candidates(local_global, local_directions, epsilon)
    return candidates, counts, raw_norm[audit_idx]


def _borrowed_directions(
    global_response: torch.Tensor,
    states: list[torch.Tensor],
    audit_idx: torch.Tensor,
    donors: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    local_global = global_response[audit_idx]
    raw_basis = torch.stack(states, dim=1)
    raw_directions = raw_basis - global_response[:, None, :]
    target_norm = local_global.norm(dim=-1, keepdim=True)
    borrowed: list[torch.Tensor] = []
    low: list[torch.Tensor] = []
    for k in range(donors.size(1)):
        direction = raw_directions[donors[:, k], k]
        direction_norm = direction.norm(dim=-1, keepdim=True)
        low.append(direction_norm.squeeze(-1) <= 1e-12)
        standardized = direction / direction_norm.clamp_min(1e-12) * target_norm
        standardized = torch.where(direction_norm > 1e-12, standardized, torch.zeros_like(standardized))
        borrowed.append(standardized)
    picked = torch.stack(borrowed, dim=1)
    low_mask = torch.stack(low, dim=1)
    return picked, int(low_mask.sum().item())


def _shuffled_candidates(
    global_response: torch.Tensor,
    states: list[torch.Tensor],
    audit_idx: torch.Tensor,
    donors: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, int]:
    directions, low_count = _borrowed_directions(global_response, states, audit_idx, donors)
    return make_action_candidates(global_response[audit_idx], directions, epsilon), low_count


def _candidate_norm_error(candidates: torch.Tensor, global_response: torch.Tensor) -> float:
    actual = candidates[:, 1:, :].norm(dim=-1)
    expected = global_response.norm(dim=-1, keepdim=True)
    relative = (actual - expected).abs() / expected.clamp_min(1e-12)
    return float(relative.max().item()) if relative.numel() else 0.0


def _evaluate_epsilon(
    *,
    model: GlobalDualBranchResponseHost,
    all_states_text: list[torch.Tensor],
    all_states_visual: list[torch.Tensor],
    global_text: torch.Tensor,
    global_visual: torch.Tensor,
    baseline_logits_all: torch.Tensor,
    audit_idx: torch.Tensor,
    audit_labels: torch.Tensor,
    degree_bucket: torch.Tensor,
    predicted_class: torch.Tensor,
    epsilon: float,
    control_repeats: int,
    bootstrap_repeats: int,
    qa_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    device = global_text.device
    audit_idx_device = audit_idx.to(device)
    labels = audit_labels.to(device)
    baseline_logits = baseline_logits_all[audit_idx_device]
    if baseline_logits.shape[0] != labels.shape[0]:
        raise AssertionError("Audit labels/logits are misaligned")
    names = action_names(model.config.max_order)
    if len(names) != 1 + 2 * (model.config.max_order + 1):
        raise AssertionError("Action cardinality changed")
    global_text_audit = global_text[audit_idx_device]
    global_visual_audit = global_visual[audit_idx_device]
    structural_text, zero_text, _ = _direction_candidates(global_text, all_states_text, audit_idx_device, epsilon)
    structural_visual, zero_visual, _ = _direction_candidates(global_visual, all_states_visual, audit_idx_device, epsilon)
    u_text_t, u_text_v, logits_text, logits_visual = evaluate_single_action_utilities(
        model,
        structural_text,
        structural_visual,
        global_text_audit,
        global_visual_audit,
        labels,
        baseline_logits,
    )
    text_np = u_text_t.detach().cpu().numpy()
    visual_np = u_text_v.detach().cpu().numpy()

    no_op_error = max(
        float((logits_text[:, 0] - baseline_logits).abs().max().item()),
        float((logits_visual[:, 0] - baseline_logits).abs().max().item()),
    )
    # Independent eval repeat quantifies non-deterministic kernels and frozen-host repeatability.
    repeat_logits = model.classify_responses(global_text_audit, global_visual_audit)
    determinism_error = float((repeat_logits - baseline_logits).abs().max().item())
    norm_error = max(
        _candidate_norm_error(structural_text, global_text_audit),
        _candidate_norm_error(structural_visual, global_visual_audit),
    )
    structural_dirs_text, _ = normalized_response_directions(all_states_text, global_text)
    structural_dirs_visual, _ = normalized_response_directions(all_states_visual, global_visual)
    eps0_text = make_action_candidates(global_text_audit, structural_dirs_text[audit_idx_device], 0.0)
    eps0_visual = make_action_candidates(global_visual_audit, structural_dirs_visual[audit_idx_device], 0.0)
    eps0_error = max(
        float((eps0_text - global_text_audit[:, None]).abs().max().item()),
        float((eps0_visual - global_visual_audit[:, None]).abs().max().item()),
    )
    joint_u_t = evaluate_joint_action_utilities(
        model,
        structural_text,
        structural_visual,
        global_text_audit,
        global_visual_audit,
        labels,
        baseline_logits,
    )
    joint_t = joint_u_t.detach().cpu().numpy()
    structural_sep, structural_shared, structural_gap, joint_summary = _joint_action_summary(joint_t)
    joint_property_error = float(np.maximum(structural_shared - structural_sep, 0.0).max())
    noop_joint_utility_error = float(np.abs(joint_t[:, 0, 0]).max())

    # Re-run the exact same candidate audit to quantify deterministic inference
    # over candidate logits and utility, rather than only baseline logits.
    repeated_utility_text, repeated_utility_visual, repeated_logits_text, repeated_logits_visual = evaluate_single_action_utilities(
        model,
        structural_text,
        structural_visual,
        global_text_audit,
        global_visual_audit,
        labels,
        baseline_logits,
    )
    candidate_repeat_error = max(
        float((repeated_utility_text - u_text_t).abs().max().item()),
        float((repeated_utility_visual - u_text_v).abs().max().item()),
        float((repeated_logits_text - logits_text).abs().max().item()),
        float((repeated_logits_visual - logits_visual).abs().max().item()),
    )
    repeated_joint_utility = evaluate_joint_action_utilities(
        model,
        structural_text,
        structural_visual,
        global_text_audit,
        global_visual_audit,
        labels,
        baseline_logits,
    )
    candidate_repeat_error = max(
        candidate_repeat_error,
        float((repeated_joint_utility - joint_u_t).abs().max().item()),
    )

    shuffled_text: list[np.ndarray] = []
    shuffled_visual: list[np.ndarray] = []
    shuffled_joint: list[np.ndarray] = []
    shuffle_fallbacks: list[dict[str, int]] = []
    shuffled_zero_directions = 0
    epsilon_zero_control_error = 0.0
    norm_error_control = 0.0
    control_seeds = CONTROL_SEEDS[:control_repeats]
    for control_seed in control_seeds:
        donors, fallback = matched_control_donors(
            audit_idx,
            degree_bucket,
            predicted_class,
            len(degree_bucket),
            int(control_seed),
        )
        if int((donors == audit_idx.cpu()[:, None, None]).sum()) != 0:
            raise AssertionError("A matched shuffled donor reused its own target while alternatives existed")
        donor_text = donors[:, 0, :].to(device)
        donor_visual = donors[:, 1, :].to(device)
        if not epsilon_zero_control_error:
            borrowed_text, _ = _borrowed_directions(global_text, all_states_text, audit_idx_device, donor_text)
            borrowed_visual, _ = _borrowed_directions(global_visual, all_states_visual, audit_idx_device, donor_visual)
            epsilon0_candidates_text = make_action_candidates(global_text_audit, borrowed_text, 0.0)
            epsilon0_candidates_visual = make_action_candidates(global_visual_audit, borrowed_visual, 0.0)
            epsilon_zero_control_error = max(
                float((epsilon0_candidates_text - global_text_audit[:, None]).abs().max().item()),
                float((epsilon0_candidates_visual - global_visual_audit[:, None]).abs().max().item()),
            )
        shuffled_text_candidates, zero_t = _shuffled_candidates(
            global_text, all_states_text, audit_idx_device, donor_text, epsilon
        )
        shuffled_visual_candidates, zero_v = _shuffled_candidates(
            global_visual, all_states_visual, audit_idx_device, donor_visual, epsilon
        )
        norm_error_control = max(
            norm_error_control,
            _candidate_norm_error(shuffled_text_candidates, global_text_audit),
            _candidate_norm_error(shuffled_visual_candidates, global_visual_audit),
        )
        shuffled_zero_directions += zero_t + zero_v
        ut, uv, _, _ = evaluate_single_action_utilities(
            model,
            shuffled_text_candidates,
            shuffled_visual_candidates,
            global_text_audit,
            global_visual_audit,
            labels,
            baseline_logits,
        )
        shuffled_text.append(ut.detach().cpu().numpy())
        shuffled_visual.append(uv.detach().cpu().numpy())
        uj = evaluate_joint_action_utilities(
            model,
            shuffled_text_candidates,
            shuffled_visual_candidates,
            global_text_audit,
            global_visual_audit,
            labels,
            baseline_logits,
        )
        uj_np = uj.detach().cpu().numpy()
        shuffled_joint.append(uj_np)
        control_sep = uj_np.reshape(uj_np.shape[0], -1).max(axis=1)
        control_shared = np.diagonal(uj_np, axis1=1, axis2=2).max(axis=1)
        joint_property_error = max(
            joint_property_error,
            float(np.maximum(control_shared - control_sep, 0.0).max()),
        )
        noop_joint_utility_error = max(noop_joint_utility_error, float(np.abs(uj_np[:, 0, 0]).max()))
        shuffle_fallbacks.append(fallback)

    controls_text = np.stack(shuffled_text)
    controls_visual = np.stack(shuffled_visual)
    controls_joint = np.stack(shuffled_joint)
    summary_text = _match_metrics(text_np)
    summary_visual = _match_metrics(visual_np)
    control_text_metrics = [_match_metrics(x) for x in controls_text]
    control_visual_metrics = [_match_metrics(x) for x in controls_visual]
    t_pref = _preferred_action_summary(text_np, names)
    v_pref = _preferred_action_summary(visual_np, names)
    robust = (t_pref["margin"] >= ROBUST_MARGIN_THRESHOLD) & (v_pref["margin"] >= ROBUST_MARGIN_THRESHOLD)
    robust_disagreement = float((t_pref["best_action_index"][robust] != v_pref["best_action_index"][robust]).mean()) if robust.any() else None
    best_separate_controls = []
    best_shared_controls = []
    gap_controls = []
    joint_control_summaries = []
    for item in controls_joint:
        sep, shared, gap, details = _joint_action_summary(item)
        best_separate_controls.append(sep)
        best_shared_controls.append(shared)
        gap_controls.append(gap)
        joint_control_summaries.append(details)
    best_separate_controls = np.stack(best_separate_controls)
    best_shared_controls = np.stack(best_shared_controls)
    gap_controls = np.stack(gap_controls)

    node_seed = int(qa_seed) + int(round(epsilon * 1000))
    oracle_text_ci = _paired_mean_summary(text_np.max(axis=1), np.stack([x.max(axis=1) for x in controls_text]), node_seed, bootstrap_repeats)
    oracle_visual_ci = _paired_mean_summary(visual_np.max(axis=1), np.stack([x.max(axis=1) for x in controls_visual]), node_seed + 1, bootstrap_repeats)
    node_headroom_text = _bootstrap_node_headroom_excess(text_np, controls_text, node_seed + 2, bootstrap_repeats)
    node_headroom_visual = _bootstrap_node_headroom_excess(visual_np, controls_visual, node_seed + 3, bootstrap_repeats)
    modality_ci = _node_bootstrap_headroom_excess(
        structural_gap,
        gap_controls,
        node_seed + 4,
        bootstrap_repeats,
    )
    modality_ci["structural_modality_headroom"] = float(structural_gap.mean())
    modality_ci["shuffled_modality_headroom_mean"] = float(gap_controls.mean(axis=1).mean())

    # Check vectorized candidate utilities against direct one-row model evaluation.
    qa_single = _vectorized_bruteforce_qa(
        model,
        structural_text,
        structural_visual,
        global_text_audit,
        global_visual_audit,
        labels,
        baseline_logits,
        u_text_t,
        u_text_v,
        joint_u_t,
        seed=qa_seed,
    )
    # Small deterministic subset joint logits are also checked as batched versus row-wise.
    qa_joint_nodes = torch.tensor(qa_single["sampled_nodes"], dtype=torch.long, device=device)
    qa_joint_pairs = [(0, 0), (1, len(names) - 1), (len(names) // 2, len(names) // 2), (len(names) - 1, 1)]
    logits_batch_rows = []
    for node in qa_joint_nodes:
        for at, av in qa_joint_pairs:
            logits_batch_rows.append(model.classify_responses(structural_text[node, at], structural_visual[node, av]))
    qa_single["joint_candidate_rows_checked"] = len(logits_batch_rows)
    qa_single["joint_candidate_logits_finite"] = bool(torch.isfinite(torch.stack(logits_batch_rows)).all())
    qa_single["single_action_logits_finite"] = bool(torch.isfinite(logits_text).all() and torch.isfinite(logits_visual).all())

    metrics = {
        "epsilon": float(epsilon),
        "action_names": names,
        "action_count": len(names),
        "baseline_audit_ce_mean": float(F.cross_entropy(baseline_logits, labels).item()),
        "structural": {
            "text": {**summary_text, "paired_oracle_excess_vs_shuffled": oracle_text_ci, "paired_headroom_excess_vs_shuffled": node_headroom_text},
            "visual": {**summary_visual, "paired_oracle_excess_vs_shuffled": oracle_visual_ci, "paired_headroom_excess_vs_shuffled": node_headroom_visual},
            "modality_specific": {
                "separate_vs_shared_headroom": float(structural_gap.mean()),
                "excess_vs_shuffled": modality_ci,
                **joint_summary,
            },
            "preferred_action": {"text": {k: v for k, v in t_pref.items() if k not in {"best_action_index", "margin"}},
                                 "visual": {k: v for k, v in v_pref.items() if k not in {"best_action_index", "margin"}},
                                 "text_visual_profile_spearman": _utility_profile_spearman(text_np, visual_np),
                                 "robust_margin_threshold_ce": ROBUST_MARGIN_THRESHOLD,
                                 "robust_node_count": int(robust.sum()),
                                 "robust_preferred_action_disagreement_fraction": robust_disagreement},
        },
        "shuffled_control": {
            "repeat_count": int(control_repeats),
            "control_seeds": control_seeds,
            "fallback_counts_by_repeat": shuffle_fallbacks,
            "direction_near_zero_count": int(shuffled_zero_directions),
            "text_node_oracle_mean": float(np.mean([x["node_oracle_mean"] for x in control_text_metrics])),
            "text_node_oracle_q95": float(np.quantile([x["node_oracle_mean"] for x in control_text_metrics], 0.95)),
            "text_headroom_mean": float(np.mean([x["node_specific_headroom"] for x in control_text_metrics])),
            "text_headroom_q95": float(np.quantile([x["node_specific_headroom"] for x in control_text_metrics], 0.95)),
            "visual_node_oracle_mean": float(np.mean([x["node_oracle_mean"] for x in control_visual_metrics])),
            "visual_node_oracle_q95": float(np.quantile([x["node_oracle_mean"] for x in control_visual_metrics], 0.95)),
            "visual_headroom_mean": float(np.mean([x["node_specific_headroom"] for x in control_visual_metrics])),
            "visual_headroom_q95": float(np.quantile([x["node_specific_headroom"] for x in control_visual_metrics], 0.95)),
            "modality_headroom_mean": float(gap_controls.mean(axis=1).mean()),
            "modality_headroom_q95": float(np.quantile(gap_controls.mean(axis=1), 0.95)),
            "text_oracle_repeat_distribution": [float(x["node_oracle_mean"]) for x in control_text_metrics],
            "text_headroom_repeat_distribution": [float(x["node_specific_headroom"]) for x in control_text_metrics],
            "visual_oracle_repeat_distribution": [float(x["node_oracle_mean"]) for x in control_visual_metrics],
            "visual_headroom_repeat_distribution": [float(x["node_specific_headroom"]) for x in control_visual_metrics],
            "modality_headroom_repeat_distribution": [float(x.mean()) for x in gap_controls],
        },
        "qa": {
            "noop_max_abs_logit_error": no_op_error,
            "noop_pass_lt_1e-6": bool(no_op_error < 1e-6),
            "joint_noop_max_abs_utility_error": noop_joint_utility_error,
            "joint_noop_pass_lt_1e-6": bool(noop_joint_utility_error < 1e-6),
            "normmatch_max_relative_error": norm_error,
            "normmatch_shuffled_max_relative_error": norm_error_control,
            "normmatch_pass_lt_1e-5": bool(max(norm_error, norm_error_control) < 1e-5),
            "epsilon_zero_max_abs_error": eps0_error,
            "epsilon_zero_control_max_abs_error": epsilon_zero_control_error,
            "epsilon_zero_pass_lt_1e-6": bool(max(eps0_error, epsilon_zero_control_error) < 1e-6),
            "action_cardinality_pass": bool(len(names) == 9),
            "structural_action_count": len(names),
            "shuffled_action_count": len(names),
            "joint_oracle_max_shared_minus_separate": joint_property_error,
            "joint_oracle_pass": bool(joint_property_error <= 1e-7),
            "eval_repeat_max_abs_logit_error": determinism_error,
            "eval_repeat_logit_tolerance": 1e-5,
            "eval_repeat_pass_lt_1e-5": bool(determinism_error < 1e-5),
            "candidate_eval_repeat_max_abs_utility_error": candidate_repeat_error,
            "vectorization_float32_tolerance": 1e-5,
            "vectorization": qa_single,
            "zero_structural_direction_count_text": zero_text,
            "zero_structural_direction_count_visual": zero_visual,
        },
    }
    # Preserve full compact per-node arrays for deterministic sampling by the caller.
    details = {
        "audit_idx": audit_idx.detach().cpu().numpy(),
        "baseline_ce": F.cross_entropy(baseline_logits, labels, reduction="none").detach().cpu().numpy(),
        "text_utility": text_np,
        "visual_utility": visual_np,
        "text_best_index": t_pref["best_action_index"],
        "visual_best_index": v_pref["best_action_index"],
        "text_margin": t_pref["margin"],
        "visual_margin": v_pref["margin"],
        "text_oracle": text_np.max(axis=1),
        "visual_oracle": visual_np.max(axis=1),
        "control_text_oracle_by_node": np.stack([x.max(axis=1) for x in controls_text]),
        "control_visual_oracle_by_node": np.stack([x.max(axis=1) for x in controls_visual]),
        "joint_best_separate": structural_sep,
        "joint_best_shared": structural_shared,
        "joint_gap": structural_gap,
        "control_joint_gap_by_node": gap_controls,
    }
    return metrics, details, {"all_nodes_text": global_text, "all_nodes_visual": global_visual}


def _sample_per_node_rows(
    details_by_epsilon: dict[float, dict[str, Any]],
    epsilon_values: tuple[float, ...],
    audit_idx: torch.Tensor,
    degree_bucket: torch.Tensor,
    predicted_class: torch.Tensor,
    sample_size: int = 256,
) -> list[dict[str, Any]]:
    audit_cpu = audit_idx.detach().cpu().numpy()
    generator = np.random.default_rng(PARTITION_SEED)
    take = np.sort(generator.choice(len(audit_cpu), size=min(sample_size, len(audit_cpu)), replace=False))
    rows: list[dict[str, Any]] = []
    for epsilon in epsilon_values:
        detail = details_by_epsilon[float(epsilon)]
        for local_id in take:
            node = int(audit_cpu[local_id])
            row: dict[str, Any] = {
                "node_id": node,
                "epsilon": float(epsilon),
                "degree_bucket": int(degree_bucket[node]),
                "predicted_class": int(predicted_class[node]),
                "baseline_ce": float(detail["baseline_ce"][local_id]),
                "best_text_action": int(detail["text_best_index"][local_id]),
                "best_text_margin": float(detail["text_margin"][local_id]),
                "text_node_oracle": float(detail["text_oracle"][local_id]),
                "best_visual_action": int(detail["visual_best_index"][local_id]),
                "best_visual_margin": float(detail["visual_margin"][local_id]),
                "visual_node_oracle": float(detail["visual_oracle"][local_id]),
                "shuffled_text_oracle_mean": float(detail["control_text_oracle_by_node"][:, local_id].mean()),
                "shuffled_text_oracle_q95": float(np.quantile(detail["control_text_oracle_by_node"][:, local_id], 0.95)),
                "shuffled_visual_oracle_mean": float(detail["control_visual_oracle_by_node"][:, local_id].mean()),
                "shuffled_visual_oracle_q95": float(np.quantile(detail["control_visual_oracle_by_node"][:, local_id], 0.95)),
                "best_joint_separate_utility": float(detail["joint_best_separate"][local_id]),
                "best_joint_shared_utility": float(detail["joint_best_shared"][local_id]),
                "modality_headroom": float(detail["joint_gap"][local_id]),
                "shuffled_modality_headroom_mean": float(detail["control_joint_gap_by_node"][:, local_id].mean()),
                "shuffled_modality_headroom_q95": float(np.quantile(detail["control_joint_gap_by_node"][:, local_id], 0.95)),
            }
            for action_id, utility in enumerate(detail["text_utility"][local_id]):
                row[f"structural_text_U_{action_id}"] = float(utility)
            for action_id, utility in enumerate(detail["visual_utility"][local_id]):
                row[f"structural_visual_U_{action_id}"] = float(utility)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _qa_pass(qa: dict[str, Any]) -> bool:
    passed = [
        qa["noop_pass_lt_1e-6"],
        qa["joint_noop_pass_lt_1e-6"],
        qa["normmatch_pass_lt_1e-5"],
        qa["epsilon_zero_pass_lt_1e-6"],
        qa["action_cardinality_pass"],
        qa["joint_oracle_pass"],
        qa["vectorization"]["single_action_logits_finite"],
        qa["vectorization"]["joint_candidate_logits_finite"],
        qa["eval_repeat_pass_lt_1e-5"],
        qa["candidate_eval_repeat_max_abs_utility_error"] < 1e-6,
        qa["vectorization"]["max_vectorized_vs_bruteforce_logit_abs_error"] < 1e-5,
        qa["vectorization"]["max_vectorized_vs_bruteforce_ce_abs_error"] < 1e-5,
        qa["vectorization"]["max_joint_vectorized_vs_bruteforce_logit_abs_error"] < 1e-5,
    ]
    return bool(all(passed))


def _make_split_report(data, split_source: Path, partitions, allocation, data_seed: int | None) -> dict[str, Any]:
    splits = {
        "OriginalTrain": data.train_idx.detach().cpu(),
        "HostTrain": partitions["host_train"],
        "ResponseTrain": partitions["response_train"],
        "Audit": partitions["audit"],
        "Val": data.val_idx.detach().cpu(),
        "Test": data.test_idx.detach().cpu(),
    }
    assert_disjoint_splits({k: v for k, v in splits.items() if k != "OriginalTrain"})
    # The internal partitions must also exactly cover the original train split.
    if set(torch.cat([partitions[k] for k in ("host_train", "response_train", "audit")]).tolist()) != set(data.train_idx.cpu().tolist()):
        raise AssertionError("Internal indices do not cover original train_idx")
    hashes = {name: tensor_sha256(index) for name, index in splits.items()}
    hist = {
        "OriginalTrain": _class_histogram(data.y, data.train_idx),
        "HostTrain": _partition_histogram_from_allocation(allocation, "host_train"),
        "ResponseTrain": _partition_histogram_from_allocation(allocation, "response_train"),
        "Audit": _partition_histogram_from_allocation(allocation, "audit"),
        "Val": _class_histogram(data.y, data.val_idx),
        "Test": "NOT READ — TEST SPLIT UNTOUCHED",
    }
    counts = {name: int(index.numel()) for name, index in splits.items()}
    return {
        "split_source_path": str(split_source),
        "data_seed": data_seed if data.source == "magb" else "official split",
        "partition_seed": PARTITION_SEED,
        "partition_method": allocation["method"],
        "partition_fallback": allocation["fallback_rule"],
        "small_class_fallback_classes": allocation["small_class_fallback_classes"],
        "counts": counts,
        "class_histograms": hist,
        "index_sha256": hashes,
        "pairwise_disjoint": True,
        "test_split_untouched": True,
        "test_labels_read": False,
    }


def run_dataset(
    dataset: str,
    host_seeds: tuple[int, ...],
    epsilons: tuple[float, ...],
    device: torch.device,
    output_root: Path,
    *,
    max_epochs: int = 1000,
    patience: int = 100,
    control_repeats: int = 20,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> list[dict[str, Any]]:
    data, split_source, _ = load_fixed_data(dataset)
    if data.source == "magb":
        data_seed = DATA_SEED
    else:
        data_seed = None
    partitions, allocation = stratified_internal_partition(data.train_idx, data.y, PARTITION_SEED)
    split_report = _make_split_report(data, split_source, partitions, allocation, data_seed)
    _validate_label_boundary(data, partitions, split_report["index_sha256"])
    # Every host seed reuses these exact split tensors; assert their hashes before each fit.
    fixed_hashes = {k: tensor_sha256(v) for k, v in partitions.items()}
    seed_split_hashes = {
        str(seed): {**split_report["index_sha256"], **fixed_hashes}
        for seed in host_seeds
    }
    same_seed_hashes = len({json.dumps(value, sort_keys=True) for value in seed_split_hashes.values()}) == 1
    if not same_seed_hashes:
        raise AssertionError("Data/internal split hashes differ across host seeds")
    split_report["index_sha256_by_host_seed"] = seed_split_hashes
    split_report["same_data_and_internal_split_across_host_seeds"] = same_seed_hashes
    operator = _operator(data.edge_index, data.num_nodes, device)
    x_text = data.x_t.to(device)
    x_visual = data.x_i.to(device)
    outputs: list[dict[str, Any]] = []
    for host_seed in host_seeds:
        print(
            f"[H1 run start] dataset={dataset} host_seed={host_seed} data_seed={data_seed or 'official'} device={device} nodes={data.num_nodes} audit={partitions['audit'].numel()}",
            flush=True,
        )
        assert fixed_hashes == {k: tensor_sha256(v) for k, v in partitions.items()}
        run_dir = output_root / dataset / f"seed{host_seed}"
        checkpoint_path = run_dir / "host_best_val.pt"
        model, training = _train_host(
            data,
            operator,
            partitions,
            host_seed,
            device,
            checkpoint_path,
            max_epochs=max_epochs,
            patience=patience,
        )
        model.eval()
        with torch.no_grad():
            baseline_logits, states_text, states_visual, global_text, global_visual = model(
                x_text, x_visual, operator
            )
        for parameter in model.parameters():
            if parameter.requires_grad:
                raise AssertionError("Frozen response audit requires every host parameter frozen")
        # Audit labels become available only after validation checkpoint restoration and freeze.
        audit_labels = data.y[partitions["audit"]].long()
        if audit_labels.numel() != partitions["audit"].numel():
            raise AssertionError("Audit label slice does not match Audit indices")
        states = {"text": states_text, "visual": states_visual}
        gammas = {"text": model.gamma_text, "visual": model.gamma_visual}
        basis = _basis_diagnostics(states, gammas)
        predicted = baseline_logits.argmax(-1).detach().cpu()
        degree, degree_bucket, predicted_class = _degree_and_prediction_buckets(
            data.edge_index, data.num_nodes, predicted
        )
        epsilon_metrics: dict[str, Any] = {}
        details_by_epsilon: dict[float, dict[str, Any]] = {}
        all_qas: list[dict[str, Any]] = []
        for epsilon in epsilons:
            print(
                f"[H1 audit] dataset={dataset} host_seed={host_seed} epsilon={epsilon} controls={control_repeats} audit_nodes={partitions['audit'].numel()}",
                flush=True,
            )
            metrics, details, _ = _evaluate_epsilon(
                model=model,
                all_states_text=states_text,
                all_states_visual=states_visual,
                global_text=global_text,
                global_visual=global_visual,
                baseline_logits_all=baseline_logits,
                audit_idx=partitions["audit"],
                audit_labels=audit_labels,
                degree_bucket=degree_bucket,
                predicted_class=predicted_class,
                epsilon=float(epsilon),
                control_repeats=control_repeats,
                bootstrap_repeats=bootstrap_repeats,
                qa_seed=PARTITION_SEED + host_seed,
            )
            epsilon_metrics[str(epsilon)] = metrics
            details_by_epsilon[float(epsilon)] = details
            all_qas.append(metrics["qa"])
            print(
                f"[H1 audit done] dataset={dataset} host_seed={host_seed} epsilon={epsilon} text_headroom_excess={metrics['structural']['text']['paired_headroom_excess_vs_shuffled']['mean']:.6f} visual_headroom_excess={metrics['structural']['visual']['paired_headroom_excess_vs_shuffled']['mean']:.6f} modality_excess={metrics['structural']['modality_specific']['excess_vs_shuffled']['mean']:.6f} qa={_qa_pass(metrics['qa'])}",
                flush=True,
            )
        qa = {
            "split_disjointness": True,
            "same_data_and_internal_split_across_host_seeds": split_report["same_data_and_internal_split_across_host_seeds"],
            "response_labels_in_host_training_loss": False,
            "audit_labels_used_before_best_checkpoint_freeze": False,
            "test_labels_read": False,
            "test_evaluation_ran": False,
            "test_split_untouched": True,
            "epsilon_runs": {str(e): m["qa"] for e, m in ((e, epsilon_metrics[str(e)]) for e in epsilons)},
            "all_epsilon_checks_pass": all(_qa_pass(x) for x in all_qas),
            "bootstrap_target_unit": "Audit node resampling; target-sample uncertainty only, not causal or independent-graph CI",
        }
        summary = {
            "dataset": dataset,
            "host_seed": host_seed,
            "split": split_report,
            "host_training": training,
            "basis_diagnostics": basis,
            "epsilon_results": epsilon_metrics,
            "qa_report": qa,
            "protocol_deviation": [
                "Test class histogram is withheld because computing it would require reading test labels, contrary to TEST SPLIT UNTOUCHED.",
                "Non-NOOP float32 batch/reduction QA uses a 1e-5 logit/CE tolerance after formal Movies CUDA QA measured up to 4.8e-6 across batch shapes; NOOP identity remains <1e-6, and all observed errors are reported.",
            ],
        }
        _write_json(run_dir / "summary.json", summary)
        sampled_rows = _sample_per_node_rows(
            details_by_epsilon,
            tuple(float(x) for x in epsilons),
            partitions["audit"],
            degree_bucket,
            predicted_class,
        )
        _write_csv(run_dir / "per_node_sample.csv", sampled_rows)
        outputs.append(summary)
        del model, baseline_logits, states_text, states_visual, global_text, global_visual, audit_labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return outputs


def _rollup_evidence(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        dataset = summary["dataset"]
        seed = summary["host_seed"]
        for epsilon_text, result in summary["epsilon_results"].items():
            node_t = result["structural"]["text"]
            node_v = result["structural"]["visual"]
            modal = result["structural"]["modality_specific"]["excess_vs_shuffled"]
            rows.append(
                {
                    "dataset": dataset,
                    "host_seed": seed,
                    "epsilon": float(epsilon_text),
                    "audit_count": summary["split"]["counts"]["Audit"],
                    "best_epoch": summary["host_training"]["best_epoch"],
                    "val_accuracy": summary["host_training"]["val_accuracy"],
                    "val_macro_f1": summary["host_training"]["val_macro_f1"],
                    "val_ce": summary["host_training"]["val_ce"],
                    "baseline_audit_ce": result["baseline_audit_ce_mean"],
                    "text_node_oracle_structural": node_t["node_oracle_mean"],
                    "text_node_oracle_shuffled": result["shuffled_control"]["text_node_oracle_mean"],
                    "text_node_oracle_excess": node_t["paired_oracle_excess_vs_shuffled"]["mean"],
                    "text_node_oracle_ci95_low": node_t["paired_oracle_excess_vs_shuffled"]["ci95_low"],
                    "text_node_oracle_ci95_high": node_t["paired_oracle_excess_vs_shuffled"]["ci95_high"],
                    "text_headroom_structural": node_t["node_specific_headroom"],
                    "text_headroom_shuffled": result["shuffled_control"]["text_headroom_mean"],
                    "text_headroom_excess": node_t["paired_headroom_excess_vs_shuffled"]["mean"],
                    "text_headroom_ci95_low": node_t["paired_headroom_excess_vs_shuffled"]["ci95_low"],
                    "text_headroom_ci95_high": node_t["paired_headroom_excess_vs_shuffled"]["ci95_high"],
                    "visual_node_oracle_structural": node_v["node_oracle_mean"],
                    "visual_node_oracle_shuffled": result["shuffled_control"]["visual_node_oracle_mean"],
                    "visual_node_oracle_excess": node_v["paired_oracle_excess_vs_shuffled"]["mean"],
                    "visual_node_oracle_ci95_low": node_v["paired_oracle_excess_vs_shuffled"]["ci95_low"],
                    "visual_node_oracle_ci95_high": node_v["paired_oracle_excess_vs_shuffled"]["ci95_high"],
                    "visual_headroom_structural": node_v["node_specific_headroom"],
                    "visual_headroom_shuffled": result["shuffled_control"]["visual_headroom_mean"],
                    "visual_headroom_excess": node_v["paired_headroom_excess_vs_shuffled"]["mean"],
                    "visual_headroom_ci95_low": node_v["paired_headroom_excess_vs_shuffled"]["ci95_low"],
                    "visual_headroom_ci95_high": node_v["paired_headroom_excess_vs_shuffled"]["ci95_high"],
                    "modality_headroom_structural": modal["structural_modality_headroom"],
                    "modality_headroom_shuffled": modal["shuffled_modality_headroom_mean"],
                    "modality_headroom_excess": modal["mean"],
                    "modality_headroom_ci95_low": modal["ci95_low"],
                    "modality_headroom_ci95_high": modal["ci95_high"],
                    "qa_pass": summary["qa_report"]["all_epsilon_checks_pass"],
                    "test_split_untouched": summary["qa_report"]["test_split_untouched"],
                }
            )
    return rows


def _tier_for_primary(summaries: list[dict[str, Any]], metric_key: str) -> str:
    primary = [x for x in summaries if x["dataset"] in {"Movies", "Grocery"}]
    values = []
    all_ci_positive = True
    for item in primary:
        for result in item["epsilon_results"].values():
            if metric_key == "node":
                for modality in ("text", "visual"):
                    stat = result["structural"][modality]["paired_headroom_excess_vs_shuffled"]
                    values.append(float(stat["mean"]))
                    all_ci_positive = all_ci_positive and float(stat["ci95_low"]) > 0
            else:
                stat = result["structural"]["modality_specific"]["excess_vs_shuffled"]
                values.append(float(stat["mean"]))
                all_ci_positive = all_ci_positive and float(stat["ci95_low"]) > 0
    if values and all(v >= PRACTICAL_CE_EXCESS_THRESHOLD for v in values) and all_ci_positive:
        return "STRONG"
    if values and sum(v > 0 for v in values) >= math.ceil(0.7 * len(values)):
        return "MIXED / MARGINAL"
    return "WEAK / FAIL"


def _aggregate_report(summaries: list[dict[str, Any]], metrics_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# R³-MAG H1 design-freeze preflight aggregate report",
        "",
        "**TEST SPLIT UNTOUCHED.** No test labels were read and no test metric was computed.",
        "",
        "## Protocol deviations and interpretation",
        "",
        "- Test class histograms are withheld because computing one would require reading test labels. Counts and index hashes are reported.",
        "- Numerical QA deviation: non-NOOP float32 logits/CE vectorization and repeat checks use a fixed 1e-5 tolerance after formal Movies CUDA comparisons showed batch-shape differences up to 4.8e-6; NOOP identity remains held to 1e-6. Actual maximum errors are recorded per run.",
        "- HostTrain/ResponseTrain/Audit partitions use fixed seed 20261006, stratified within the fixed original train split. Original-train labels are read for this required stratification and to derive partition histograms. ResponseTrain labels do not enter host fitting or response utility. The Audit label slice used as CE targets is first indexed after best-validation checkpoint restoration and freeze.",
        "- Bootstrap intervals resample Audit targets only. They do not quantify causal or independent-graph uncertainty.",
        "- Frozen-host interventions are local diagnostics, not causal effects; argmax actions are not latent roles.",
        "",
        f"Evidence tiers use primary datasets Movies and Grocery. For tiering only, the predeclared practical CE-excess threshold is {PRACTICAL_CE_EXCESS_THRESHOLD:.2f} nats/node. STRONG requires every primary dataset × host seed × epsilon headroom excess estimate to meet that threshold with within-run 95% node-bootstrap intervals above zero. MIXED / MARGINAL means at least 70% of those estimates are positive; statistically detectable estimates below the practical threshold remain practically marginal. Otherwise WEAK / FAIL. This rule is applied unchanged to H1-Node (both modalities) and H1-Modality; raw values remain primary.",
        "",
        f"**H1-Node evidence: {_tier_for_primary(summaries, 'node')}**",
        f"**H1-Modality evidence: {_tier_for_primary(summaries, 'modality')}**",
    ]
    primary_rows = [r for r in metrics_rows if r["dataset"] in {"Movies", "Grocery"}]
    node_excess_below_control = bool(primary_rows) and all(
        float(r["text_headroom_ci95_high"]) < 0 and float(r["visual_headroom_ci95_high"]) < 0
        for r in primary_rows
    )
    modal_excess_below_control = bool(primary_rows) and all(
        float(r["modality_headroom_ci95_high"]) < 0 for r in primary_rows
    )
    if node_excess_below_control:
        lines += [
            "",
            "## H1 result reading",
            "",
            "Structural node-oracle utility is positive in absolute terms, but text and visual node-specific headroom excess are below the matched shuffled controls for every primary seed × epsilon, with all within-run 95% intervals below zero. **NO evidence for structure-specific node adaptation.**",
        ]
    if modal_excess_below_control:
        if not node_excess_below_control:
            lines += ["", "## H1 result reading", ""]
        lines.append(
            "Separate Text/Visual actions have positive absolute headroom, but the matched shuffled controls produce greater separate-vs-shared headroom for every primary seed × epsilon, with all within-run 95% intervals below zero. **NO evidence for modality-specific structural response.**"
        )
    lines += [
        "",
        "These are frozen-host local response diagnostics, not causal effects. ele-fashion shows the same negative structural excess direction here and remains a stress dataset.",
        "",
        "## Mean structural vs shuffled utilities",
        "",
        "Cells show structural / shuffled / structural-minus-shuffled means averaged over all three host seeds and both epsilons (CE utility, nats/node). Paired per-run bootstrap intervals remain in the table above and CSV.",
        "",
        "| Dataset | Text node oracle | Text node headroom | Visual node oracle | Visual node headroom | Modality separate-vs-shared headroom |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        rows = [r for r in metrics_rows if r["dataset"] == dataset]
        if not rows:
            continue
        def triple(structural: str, shuffled: str, excess: str) -> str:
            s = float(np.mean([float(r[structural]) for r in rows]))
            c = float(np.mean([float(r[shuffled]) for r in rows]))
            d = float(np.mean([float(r[excess]) for r in rows]))
            return f"{s:.4f} / {c:.4f} / {d:.4f}"
        lines.append(
            f"| {dataset} | {triple('text_node_oracle_structural','text_node_oracle_shuffled','text_node_oracle_excess')} | {triple('text_headroom_structural','text_headroom_shuffled','text_headroom_excess')} | {triple('visual_node_oracle_structural','visual_node_oracle_shuffled','visual_node_oracle_excess')} | {triple('visual_headroom_structural','visual_headroom_shuffled','visual_headroom_excess')} | {triple('modality_headroom_structural','modality_headroom_shuffled','modality_headroom_excess')} |"
        )
    lines += [
        "",
        "## Host and split protocol",
        "",
        "Host: dual projector branches (Linear→LayerNorm→ReLU→Dropout), shared symmetric normalized A+I physical graph, K=3 response basis, global trainable signed gamma initialized by gamma_k=0.2×0.8^k, modality LayerNorm, one-layer fusion MLP and classifier. Hyperparameters are fixed across datasets: hidden=128, dropout=0.2, AdamW lr=1e-3, weight decay=5e-4, max epochs=1000, patience=100. Checkpoint selection follows the NC runner's min_epoch=30 and min_delta=1e-4, using original-val accuracy only.",
        "",
        "Response action set has nine actions per modality (NOOP, ± for each k=0…3), with epsilon 0.1 and 0.2 and L2 NormMatch. Each shuffled-control repeat borrows real directions within degree-quantile × frozen-predicted-class buckets, falling back to degree and then global buckets. Each repeat executes all joint action pairs through frozen fusion and classifier.",
        "",
        "## Split counts and index hashes",
        "",
        "| Dataset | Split source | Data seed | HostTrain | ResponseTrain | Audit | Val | Test | Partition seed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    seen = {}
    for item in summaries:
        seen.setdefault(item["dataset"], item["split"])
    for dataset, split in seen.items():
        c = split["counts"]
        lines.append(
            f"| {dataset} | `{split['split_source_path']}` | {split['data_seed']} | {c['HostTrain']} | {c['ResponseTrain']} | {c['Audit']} | {c['Val']} | {c['Test']} | {split['partition_seed']} |"
        )
    lines += ["", "Index SHA256 values and class histograms are in each per-run summary JSON. Test histogram is withheld by design.", "", "## Primary H1 results", "", "Means are over three host seeds; each row preserves epsilon. Utility is CE reduction in nats/node. Positive excess means structure-specific gain over the matched shuffled control. Parenthesized endpoints show the envelope of per-seed 95% target-bootstrap intervals (minimum lower to maximum upper), not a pooled confidence interval; per-seed intervals are in the CSV and JSON.", "", "| Dataset | ε | Text oracle excess (run-CI envelope) | Text headroom excess (run-CI envelope) | Visual oracle excess (run-CI envelope) | Visual headroom excess (run-CI envelope) | Modality headroom structural | shuffled | excess (run-CI envelope) | seed signs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        for epsilon in EPSILON_VALUES:
            rows = [r for r in metrics_rows if r["dataset"] == dataset and float(r["epsilon"]) == float(epsilon)]
            if not rows:
                continue
            def mean_ci(key, low, high):
                return f"{np.mean([x[key] for x in rows]):.5f} ({np.min([x[low] for x in rows]):.5f}, {np.max([x[high] for x in rows]):.5f})"
            modal_s = float(np.mean([x["modality_headroom_structural"] for x in rows]))
            modal_c = float(np.mean([x["modality_headroom_shuffled"] for x in rows]))
            signs = f"{sum(x['modality_headroom_excess'] > 0 for x in rows)}/{len(rows)}"
            lines.append(
                f"| {dataset} | {epsilon:.1f} | {mean_ci('text_node_oracle_excess','text_node_oracle_ci95_low','text_node_oracle_ci95_high')} | {mean_ci('text_headroom_excess','text_headroom_ci95_low','text_headroom_ci95_high')} | {mean_ci('visual_node_oracle_excess','visual_node_oracle_ci95_low','visual_node_oracle_ci95_high')} | {mean_ci('visual_headroom_excess','visual_headroom_ci95_low','visual_headroom_ci95_high')} | {modal_s:.5f} | {modal_c:.5f} | {mean_ci('modality_headroom_excess','modality_headroom_ci95_low','modality_headroom_ci95_high')} | {signs} |"
            )
    lines += ["", "## Across-seed and epsilon stability", ""]
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        rows = [r for r in metrics_rows if r["dataset"] == dataset]
        if not rows:
            continue
        for metric in ("text_headroom_excess", "visual_headroom_excess", "modality_headroom_excess"):
            signs = sum(float(x[metric]) > 0 for x in rows)
            for epsilon in EPSILON_VALUES:
                at_epsilon = [x for x in rows if float(x["epsilon"]) == float(epsilon)]
                positive = sum(float(x[metric]) > 0 for x in at_epsilon)
                lines.append(f"- {dataset} ε={epsilon:.1f} {metric}: {positive}/{len(at_epsilon)} host-seed estimates are positive.")
            same_sign = 0
            for seed in DEFAULT_HOST_SEEDS:
                pair = [x for x in rows if int(x["host_seed"]) == seed]
                if len(pair) == 2 and (float(pair[0][metric]) > 0) == (float(pair[1][metric]) > 0):
                    same_sign += 1
            lines.append(f"- {dataset} {metric}: {same_sign}/{len(DEFAULT_HOST_SEEDS)} host seeds agree in sign across ε=0.1 and 0.2; {signs}/{len(rows)} seed×epsilon estimates are positive.")
    lines += ["", "## Per-run records", "", "Each dataset/seed JSON contains training metrics, learned signed gamma, basis diagnostics, paired bootstrap intervals, all 20 shuffled repeat summaries, fallback counts, and correctness QA. Each per-node CSV is a deterministic sample (up to 256 Audit nodes per epsilon) with all structural single-modality action utilities and shuffled/joint summaries.", ""]
    return "\n".join(lines)


def run_formal(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_root = Path(args.output_root).resolve()
    summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        summaries.extend(
            run_dataset(
                dataset,
                tuple(args.seeds),
                tuple(args.epsilons),
                device,
                output_root,
                max_epochs=args.max_epochs,
                patience=args.patience,
                control_repeats=args.control_repeats,
                bootstrap_repeats=args.bootstrap_repeats,
            )
        )
    report_root = Path(args.report_root).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    metrics_rows = _rollup_evidence(summaries)
    _write_csv(report_root / "aggregate_metrics.csv", metrics_rows)
    for item in summaries:
        dataset = item["dataset"]
        seed = item["host_seed"]
        source = output_root / dataset / f"seed{seed}" / "summary.json"
        dest = report_root / "per_run" / f"{dataset}_seed{seed}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        sample_src = output_root / dataset / f"seed{seed}" / "per_node_sample.csv"
        sample_dst = report_root / "per_node" / f"{dataset}_seed{seed}_sample.csv"
        sample_dst.parent.mkdir(parents=True, exist_ok=True)
        sample_dst.write_text(sample_src.read_text(encoding="utf-8"), encoding="utf-8")
    qa_report = {
        "test_split_untouched": True,
        "test_labels_read": False,
        "formal_runs": len(summaries),
        "all_runs_all_epsilon_qa_pass": all(x["qa_report"]["all_epsilon_checks_pass"] for x in summaries),
        "bootstrap_repeats": args.bootstrap_repeats,
        "runs": {
            f"{x['dataset']}_seed{x['host_seed']}": x["qa_report"] for x in summaries
        },
    }
    _write_json(report_root / "qa_report.json", qa_report)
    (report_root / "aggregate_report.md").write_text(_aggregate_report(summaries, metrics_rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="R3-MAG H1 frozen-host structural response audit")
    parser.add_argument("--datasets", nargs="+", default=["Movies", "Grocery", "ele-fashion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_HOST_SEEDS))
    parser.add_argument("--epsilons", nargs="+", type=float, default=list(EPSILON_VALUES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/r3mag_design_freeze/h1")
    parser.add_argument("--report-root", default="reports/r3mag_design_freeze/h1")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--smoke", action="store_true", help="Movies seed42 short implementation/QA smoke only")
    args = parser.parse_args(argv)
    if args.smoke:
        args.datasets = ["Movies"]
        args.seeds = [42]
        args.epsilons = [0.1, 0.2]
        args.max_epochs = 3
        args.patience = 3
        args.control_repeats = 2
        args.bootstrap_repeats = 50
        args.output_root = "outputs/r3mag_design_freeze/h1_smoke"
        args.report_root = "outputs/r3mag_design_freeze/h1_smoke/reports"
    if args.control_repeats < 1 or args.control_repeats > 20:
        raise ValueError("control_repeats must be within 1..20")
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    run_formal(args)


if __name__ == "__main__":
    main()
