from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_message_effect_v5a as v5a
from src.research.message_effect_atlas import _safe_pearson, _safe_spearman, select_receiver_edges
from src.research.message_effect_replay import (
    build_frozen_host_cache,
    estimator_receiver_split,
    receiver_degree_quartile_sample,
    replay_local_basis_changes,
    singleton_basis_delta,
)
from src.research.message_effect_v5a1 import (
    ORACLE_THRESHOLDS,
    TAUS,
    baseline_path_diagnostics,
    matched_effect_from_losses,
    matched_old_effect_diagnostics,
    old_effect_from_losses,
    oracle_group_rows,
    sensitivity_diagnostics,
)
from src.research.message_effect_estimator import MessageEffectEstimator
from src.research.message_effect_v5a1_estimator import SingleTargetMessageEffectEstimator, make_edge_rank_pairs, ranknet_pair_loss
from src.research.message_effect_v5a1_training import (
    ESTIMATORS,
    TARGETS,
    _global_metrics,
    pointwise_metrics,
    ranker_metrics,
    ranker_sensitivity,
    row_splits,
    train_pointwise,
    train_ranker,
)

DATASETS = ("Movies", "Grocery", "ele-fashion")
RESEARCH = ROOT / "research" / "mag_message_effect_v5a1_ranking_repair"
DATA_DIR = RESEARCH / "data"
OUTPUT = ROOT / "outputs" / "mag_message_effect_v5a1"
V5A_OUTPUT = ROOT / "outputs" / "mag_message_effect_v5a"
HOST_SEED = 42
SAMPLE_SEED = 2027
ESTIMATOR_SEEDS = (0, 1, 2)
FEATURE_VARIANTS = ESTIMATORS
BUNDLE_SPECS = (
    *((f"same_hop_text_visual_k{k}", (("text", k), ("visual", k))) for k in range(1, 5)),
    ("text_all_hops", tuple(("text", k) for k in range(1, 5))),
    ("visual_all_hops", tuple(("visual", k) for k in range(1, 5))),
    ("full_exposed_edge_8_messages", tuple((m, k) for m in ("text", "visual") for k in range(1, 5))),
)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_summary(values: Iterable[float], prefix: str) -> dict[str, float]:
    x = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if not x.size:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "median", "std", "p95_abs", "max_abs")}
    return {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_median": float(np.median(x)),
            f"{prefix}_std": float(np.std(x)), f"{prefix}_p95_abs": float(np.quantile(np.abs(x), .95)),
            f"{prefix}_max_abs": float(np.max(np.abs(x)))}


def _as_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in row.items()}


def _load_v5a_cache(dataset: str) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, set[int]]]:
    path = V5A_OUTPUT / f"{dataset}_effect_features.pt"
    if not path.is_file():
        raise FileNotFoundError(f"V5A feature cache missing: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    features = {k: v.detach().cpu() for k, v in blob["features"].items() if torch.is_tensor(v)}
    rows = list(blob["rows"])
    split = {k: set(map(int, values)) for k, values in blob["split_receivers"].items()}
    if len(rows) != features["receiver_id"].numel():
        raise AssertionError(f"V5A row/feature alignment failed for {dataset}")
    for i, row in enumerate(rows):
        if int(row["receiver_id"]) != int(features["receiver_id"][i]) or int(row["directed_edge_index"]) != int(features["edge_index"][i]):
            raise AssertionError(f"V5A metadata and feature row mismatch at {dataset}/{i}")
    return features, rows, split


def _host_and_train_labels(dataset: str, device: str):
    host_row = v5a._find_v4a_row(dataset, "R0_raw", HOST_SEED)
    model, classifier, checkpoint, cfg, x, edge_index = v5a._load_host(host_row, device)
    supervision = v5a.load_training_supervision(dataset, include_validation=False, seed=HOST_SEED)
    if supervision["validation_labels_read"] or supervision["test_labels_read"]:
        raise AssertionError("V5A train-only supervision loader exposed validation/test labels")
    x = x.to(device)
    edge_index = edge_index.to(device)
    cache = build_frozen_host_cache(dataset, model, classifier, x, edge_index, device)
    return host_row, checkpoint, cache, supervision


def _sample_ids(dataset: str, cache, supervision: dict[str, Any], *, max_receivers: int = 1500,
                max_edges: int = 4) -> tuple[torch.Tensor, list[int], dict[int, float], dict[int, float]]:
    receivers, receiver_probability, _ = receiver_degree_quartile_sample(
        supervision["train_idx"], cache.indegree, max_receivers=max_receivers, seed=SAMPLE_SEED
    )
    edge_ids, edge_probability = select_receiver_edges(cache, receivers, max_edges=max_edges, seed=SAMPLE_SEED)
    return receivers.cpu(), edge_ids, receiver_probability, edge_probability


def _assert_v5a_id_crosscheck(dataset: str, receivers: torch.Tensor, edge_ids: list[int],
                             rows: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path = V5A_OUTPUT / f"{dataset}_singleton_raw.csv"
    feature_set = {(int(r["receiver_id"]), int(r["directed_edge_index"])) for r in rows}
    if csv_path.is_file():
        csv_rows = _rows(csv_path)
        raw_set = {(int(r["receiver_id"]), int(r["directed_edge_index"])) for r in csv_rows}
        edge_id_set = set(edge_ids)
        selected_set = {(int(r["receiver_id"]), int(r["directed_edge_index"])) for r in rows
                        if int(r["directed_edge_index"]) in edge_id_set}
        # Metadata cache and complete raw CSV are independently checked; expected IDs are V5A's feature rows.
        if raw_set != feature_set:
            raise AssertionError(f"V5A raw CSV and feature cache row ID sets differ for {dataset}")
        if selected_set != {(int(r["receiver_id"]), int(r["directed_edge_index"])) for r in rows}:
            raise AssertionError(f"sampled receiver/edge IDs differ from V5A feature rows for {dataset}")
        return {"v5a_raw_id_crosscheck": True, "v5a_raw_rows": len(csv_rows), "feature_id_pairs": len(feature_set),
                "sampled_directed_edges": len(edge_ids), "receiver_count": int(receivers.numel())}
    return {"v5a_raw_id_crosscheck": False, "crosscheck_note": "V5A singleton raw CSV absent; deterministic sampling used",
            "feature_id_pairs": len(feature_set), "sampled_directed_edges": len(edge_ids),
            "receiver_count": int(receivers.numel())}


def _matched_target_subset(cache, features: dict[str, torch.Tensor], rows: list[dict[str, Any]],
                           row_indices: torch.Tensor, baseline: dict[str, torch.Tensor],
                           baseline_by_node: torch.Tensor, labels_by_node: torch.Tensor,
                           chunk_size: int = 1024) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Replay matched delete/comp effects for an aligned subset of V5A feature rows."""
    row_indices = torch.as_tensor(row_indices, dtype=torch.long).cpu()
    selected_rows = [dict(rows[i]) for i in row_indices.tolist()]
    n = row_indices.numel()
    device = cache.device
    row_eid = features["edge_index"].index_select(0, row_indices).to(device)
    row_receiver = features["receiver_id"].index_select(0, row_indices).to(device)
    matched_delete = torch.full((n,), float("nan"), dtype=torch.float32)
    old_delete = torch.full((n,), float("nan"), dtype=torch.float32)
    matched_comp = torch.full((n,), float("nan"), dtype=torch.float32)
    old_comp = torch.full((n,), float("nan"), dtype=torch.float32)
    comp_mask = torch.zeros(n, dtype=torch.bool)
    local_baseline_row = baseline_by_node.index_select(0, row_receiver)
    label_row = labels_by_node.index_select(0, row_receiver)
    full_baseline_row = F.cross_entropy(cache.logits.index_select(0, row_receiver), label_row, reduction="none")
    row_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for j, meta in enumerate(selected_rows):
        row_groups[(str(meta["modality"]), int(meta["hop"]))].append(j)
    for (modality, hop), positions in row_groups.items():
        for start in range(0, len(positions), chunk_size):
            pos = torch.tensor(positions[start:start + chunk_size], dtype=torch.long)
            eids = row_eid.index_select(0, pos.to(device))
            receivers = row_receiver.index_select(0, pos.to(device))
            senders = cache.src.index_select(0, eids)
            labels = label_row.index_select(0, pos.to(device))
            weights = cache.edge_weight.index_select(0, eids)
            denominator = cache.denominators[modality][hop - 1]
            previous = cache.prior[modality] if hop == 1 else cache.raw_states[modality][hop - 2]
            message = weights[:, None] * previous.index_select(0, senders)
            deletion_delta = singleton_basis_delta(message, denominator)
            delete_replay = replay_local_basis_changes(cache, receivers, [{"modality": modality, "hop": hop,
                                                                             "delta": deletion_delta}])
            matched = matched_effect_from_losses(delete_replay["logits"], labels,
                                                 local_baseline_row.index_select(0, pos.to(device)))
            old = old_effect_from_losses(delete_replay["logits"], labels,
                                         full_baseline_row.index_select(0, pos.to(device)))
            matched_delete[pos] = matched.float().cpu()
            old_delete[pos] = old.float().cpu()

            retained = cache.receiver_mass.index_select(0, receivers) - weights
            feasible = retained > 1.0e-12
            valid_local = torch.nonzero(feasible, as_tuple=False).reshape(-1)
            if valid_local.numel():
                valid_eids = eids.index_select(0, valid_local)
                valid_receivers = receivers.index_select(0, valid_local)
                valid_labels = labels.index_select(0, valid_local)
                valid_message = message.index_select(0, valid_local)
                valid_retained = retained.index_select(0, valid_local)
                mass = cache.receiver_mass.index_select(0, valid_receivers)
                raw_state = cache.raw_states[modality][hop - 1].index_select(0, valid_receivers)
                scale = mass / valid_retained
                comp_delta = (scale[:, None] * (raw_state - valid_message) - raw_state) / denominator
                comp_replay = replay_local_basis_changes(cache, valid_receivers, [{"modality": modality, "hop": hop,
                                                                                   "delta": comp_delta}])
                valid_positions = pos.index_select(0, valid_local.cpu())
                local_loss = local_baseline_row.index_select(0, valid_positions.to(device))
                full_loss = full_baseline_row.index_select(0, valid_positions.to(device))
                matched_comp[valid_positions] = matched_effect_from_losses(comp_replay["logits"], valid_labels, local_loss).float().cpu()
                old_comp[valid_positions] = old_effect_from_losses(comp_replay["logits"], valid_labels, full_loss).float().cpu()
                comp_mask[valid_positions] = True
    for j, meta in enumerate(selected_rows):
        meta["delta_delete_old"] = float(old_delete[j])
        meta["delta_delete_matched"] = float(matched_delete[j])
        meta["delta_delete_matched_minus_old"] = float(matched_delete[j] - old_delete[j])
        meta["baseline_loss_full"] = float(full_baseline_row[j])
        meta["baseline_loss_local"] = float(local_baseline_row[j])
        meta["baseline_local_minus_full_loss"] = float(local_baseline_row[j] - full_baseline_row[j])
        meta["compensation_feasible"] = bool(comp_mask[j])
        meta["delta_comp_old"] = float(old_comp[j]) if comp_mask[j] else None
        meta["delta_comp_matched"] = float(matched_comp[j]) if comp_mask[j] else None
        meta["delta_comp_matched_minus_old"] = float(matched_comp[j] - old_comp[j]) if comp_mask[j] else None
    target_features = {k: v.index_select(0, row_indices).clone() for k, v in features.items()}
    target_features["target_delete"] = matched_delete
    target_features["target_comp"] = matched_comp
    target_features["target_comp_mask"] = comp_mask
    if not torch.isfinite(matched_delete).all() or not torch.isfinite(matched_comp[comp_mask]).all():
        raise FloatingPointError("nonfinite matched singleton effects")
    return selected_rows, target_features


def _bundle_replays(cache, selected_rows: list[dict[str, Any]], features: dict[str, torch.Tensor],
                    baseline_by_node: torch.Tensor, labels_by_node: torch.Tensor,
                    chunk_size: int = 256) -> list[dict[str, Any]]:
    device = cache.device
    unique_edges: dict[int, tuple[int, int]] = {}
    singleton: dict[tuple[int, str, int], tuple[float, float | None]] = {}
    for row in selected_rows:
        eid, receiver = int(row["directed_edge_index"]), int(row["receiver_id"])
        unique_edges[eid] = (receiver, int(row["sender_id"]))
        singleton[(eid, str(row["modality"]), int(row["hop"]))] = (
            float(row["delta_delete_matched"]),
            float(row["delta_comp_matched"]) if row["delta_comp_matched"] is not None else None,
        )
    edge_ids = sorted(unique_edges)
    output: list[dict[str, Any]] = []
    for edge_start in range(0, len(edge_ids), chunk_size):
        ids_cpu = edge_ids[edge_start:edge_start + chunk_size]
        eids = torch.tensor(ids_cpu, device=device, dtype=torch.long)
        receivers = cache.dst.index_select(0, eids)
        labels = labels_by_node.index_select(0, receivers)
        weights = cache.edge_weight.index_select(0, eids)
        mass = cache.receiver_mass.index_select(0, receivers)
        retained = mass - weights
        feasible = retained > 1.0e-12
        scale = mass / retained.clamp_min(1.0e-12)
        local_loss = baseline_by_node.index_select(0, receivers)
        for bundle_type, specs in BUNDLE_SPECS:
            delete_changes, comp_changes = [], []
            for modality, hop in specs:
                previous = cache.prior[modality] if hop == 1 else cache.raw_states[modality][hop - 2]
                message = weights[:, None] * previous.index_select(0, cache.src.index_select(0, eids))
                delete_changes.append({"modality": modality, "hop": hop,
                                      "delta": singleton_basis_delta(message, cache.denominators[modality][hop - 1])})
                state = cache.raw_states[modality][hop - 1].index_select(0, receivers)
                comp_delta = (scale[:, None] * (state - message) - state) / cache.denominators[modality][hop - 1]
                comp_changes.append({"modality": modality, "hop": hop, "delta": comp_delta})
            delete_replay = replay_local_basis_changes(cache, receivers, delete_changes)
            delete_effect = matched_effect_from_losses(delete_replay["logits"], labels, local_loss).cpu().tolist()
            local_valid = torch.nonzero(feasible, as_tuple=False).reshape(-1)
            comp_effect = [None] * len(ids_cpu)
            if local_valid.numel():
                comp_replay = replay_local_basis_changes(cache, receivers.index_select(0, local_valid), [
                    {"modality": c["modality"], "hop": c["hop"], "delta": c["delta"].index_select(0, local_valid)}
                    for c in comp_changes
                ])
                values = matched_effect_from_losses(comp_replay["logits"], labels.index_select(0, local_valid),
                                                    local_loss.index_select(0, local_valid)).cpu().tolist()
                for ix, value in zip(local_valid.cpu().tolist(), values):
                    comp_effect[ix] = float(value)
            for i, eid in enumerate(ids_cpu):
                single_items = [singleton[(eid, m, h)] for m, h in specs]
                delete_sum = sum(x[0] for x in single_items)
                interaction_delete = float(delete_effect[i]) - delete_sum
                comp_values = [x[1] for x in single_items]
                is_feasible = bool(feasible[i]) and all(x is not None for x in comp_values)
                comp_sum = sum(float(x) for x in comp_values if x is not None) if is_feasible else None
                interaction_comp = float(comp_effect[i]) - comp_sum if is_feasible else None
                output.append({
                    "dataset": cache.dataset, "receiver_id": unique_edges[eid][0], "sender_id": unique_edges[eid][1],
                    "directed_edge_index": eid, "bundle_type": bundle_type, "component_count": len(specs),
                    "delta_bundle_delete_matched": float(delete_effect[i]), "delete_component_sum_matched": delete_sum,
                    "interaction_delete_matched": interaction_delete,
                    "relative_interaction_delete_matched": abs(interaction_delete) / (sum(abs(x[0]) for x in single_items) + 1e-12),
                    "compensation_feasible": is_feasible, "delta_bundle_comp_matched": comp_effect[i] if is_feasible else None,
                    "comp_component_sum_matched": comp_sum, "interaction_comp_matched": interaction_comp,
                    "relative_interaction_comp_matched": abs(interaction_comp) / (sum(abs(float(x)) for x in comp_values if x is not None) + 1e-12) if is_feasible else None,
                })
    return output


def _baseline_for_receivers(cache, receivers: torch.Tensor, labels_by_node: torch.Tensor):
    receiver_device = receivers.to(cache.device)
    labels = labels_by_node.index_select(0, receiver_device)
    if bool((labels < 0).any()):
        raise AssertionError("sampled receiver is missing a training label")
    baseline = v5a1_local = replay_local_basis_changes(cache, receiver_device, changes=[])
    local_loss = F.cross_entropy(v5a1_local["logits"], labels, reduction="none")
    full_loss = F.cross_entropy(cache.logits.index_select(0, receiver_device), labels, reduction="none")
    per_node_loss = torch.full_like(cache.receiver_mass, float("nan"))
    per_node_loss[receiver_device] = local_loss
    diagnostics = baseline_path_diagnostics(
        {"logits": v5a1_local["logits"], "z": v5a1_local["z"], "loss": local_loss},
        cache.logits.index_select(0, receiver_device), cache.z.index_select(0, receiver_device), full_loss,
    )
    return {"logits": v5a1_local["logits"], "z": v5a1_local["z"], "loss": local_loss}, per_node_loss, full_loss, diagnostics


def prepare_dataset(dataset: str, device: str = "cuda:1", *, save: bool = True) -> dict[str, Any]:
    """Load the exact V4A host, assert the V5A sample, then replay matched effects."""
    features, rows, old_split = _load_v5a_cache(dataset)
    host_row, checkpoint, cache, supervision = _host_and_train_labels(dataset, device)
    receivers, edge_ids, receiver_probability, edge_probability = _sample_ids(dataset, cache, supervision)
    crosscheck = _assert_v5a_id_crosscheck(dataset, receivers, edge_ids, rows)
    v5a_edge_set = {int(r["directed_edge_index"]) for r in rows}
    if set(edge_ids) != v5a_edge_set:
        raise AssertionError(f"deterministic sampled edge IDs do not equal V5A for {dataset}")
    v5a_receivers = {int(r["receiver_id"]) for r in rows}
    if set(receivers.tolist()) != v5a_receivers:
        raise AssertionError(f"deterministic sampled receiver IDs do not equal V5A for {dataset}")
    labels_by_node = v5a._train_labels_by_node(supervision, cache.logits.size(0)).to(cache.device)
    baseline, baseline_by_node, full_loss, baseline_diagnostics = _baseline_for_receivers(cache, receivers, labels_by_node)
    row_indices = torch.arange(len(rows), dtype=torch.long)
    matched_rows, matched_features = _matched_target_subset(cache, features, rows, row_indices, baseline,
                                                             baseline_by_node, labels_by_node)
    # Retain old V5A targets from its frozen feature cache, and assert the recomputed old path agrees.
    for i, row in enumerate(matched_rows):
        old_d = float(features["target_delete"][i])
        if abs(old_d - float(row["delta_delete_old"])) > 2e-5:
            raise AssertionError(f"replayed V5A deletion effect changed at row {i}: {old_d} vs {row['delta_delete_old']}")
        old_c = float(features["target_comp"][i])
        if math.isfinite(old_c) and row["delta_comp_old"] is not None and abs(old_c - float(row["delta_comp_old"])) > 2e-5:
            raise AssertionError(f"replayed V5A compensation effect changed at row {i}")
    bundles = _bundle_replays(cache, matched_rows, matched_features, baseline_by_node, labels_by_node)
    delete_diag = matched_old_effect_diagnostics(
        [r["delta_delete_matched"] for r in matched_rows], [r["delta_delete_old"] for r in matched_rows])
    comp_diag = matched_old_effect_diagnostics(
        [r["delta_comp_matched"] for r in matched_rows if r["delta_comp_matched"] is not None],
        [r["delta_comp_old"] for r in matched_rows if r["delta_comp_matched"] is not None])
    baseline_diagnostics.update({"dataset": dataset, "host_checkpoint": host_row["checkpoint_path"],
                                 "host_checkpoint_sha256": _sha256(ROOT / host_row["checkpoint_path"]),
                                 "host_seed": HOST_SEED, "host_variant": "R0_raw", **crosscheck})
    bundle_summaries = []
    for target, suffix in (("delete", "delete"), ("comp", "comp")):
        for bundle_type in sorted({r["bundle_type"] for r in bundles}):
            vals = [r[f"interaction_{suffix}_matched"] for r in bundles if r["bundle_type"] == bundle_type and r[f"interaction_{suffix}_matched"] is not None]
            rels = [r[f"relative_interaction_{suffix}_matched"] for r in bundles if r["bundle_type"] == bundle_type and r[f"relative_interaction_{suffix}_matched"] is not None]
            bundle_summaries.append({"dataset": dataset, "target": target, "bundle_type": bundle_type,
                                     "n": len(vals), **_finite_summary(vals, "interaction"),
                                     "relative_median": float(np.median(rels)) if rels else float("nan")})
    data = {"dataset": dataset, "features": matched_features, "rows": matched_rows, "v5a_split": old_split,
            "baseline": baseline, "baseline_by_node": baseline_by_node, "full_loss": full_loss,
            "bundles": bundles, "baseline_diagnostics": baseline_diagnostics,
            "bundle_summaries": bundle_summaries, "old_vs_matched": {"delete": delete_diag, "comp": comp_diag},
            "sampling": {"receivers": receivers, "edge_ids": edge_ids, "receiver_probability": receiver_probability,
                         "edge_probability": edge_probability}, "supervision": {
                             "train_labels_read": True, "validation_labels_read": False, "test_labels_read": False,
                             "test_metrics_read": False, "split_path": supervision["split_path"]},
            "host_row": host_row}
    if save:
        out_dir = OUTPUT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"features": matched_features, "rows": matched_rows, "split_receivers": old_split,
                    "receiver_ids": sorted(set(int(r["receiver_id"]) for r in matched_rows))}, out_dir / "matched_features.pt")
        _write_csv(out_dir / "matched_singleton_raw.csv", (_as_csv_row(r) for r in matched_rows))
        _write_csv(out_dir / "matched_bundle_raw.csv", (_as_csv_row(r) for r in bundles))
        _write_json(out_dir / "preparation_summary.json", {
            "baseline_diagnostics": baseline_diagnostics, "old_vs_matched": {"delete": delete_diag, "comp": comp_diag},
            "bundle_summaries": bundle_summaries, "sampling_crosscheck": crosscheck,
            "host_row": host_row, "supervision": {"train_labels_read": True, "validation_labels_read": False,
                "test_labels_read": False, "test_metrics_read": False, "split_path": supervision["split_path"]},
        })
    del checkpoint, cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return data


def _prepare_subset(dataset: str, device: str, receiver_limit: int, max_edges: int):
    features, all_rows, _ = _load_v5a_cache(dataset)
    host_row, checkpoint, cache, supervision = _host_and_train_labels(dataset, device)
    all_receivers, _, _, _ = _sample_ids(dataset, cache, supervision)
    receivers = all_receivers[:receiver_limit]
    edge_ids, edge_probability = select_receiver_edges(cache, receivers, max_edges=max_edges, seed=SAMPLE_SEED)
    edge_set = set(edge_ids)
    row_indices = torch.tensor([i for i, row in enumerate(all_rows) if int(row["directed_edge_index"]) in edge_set], dtype=torch.long)
    subset_rows = [all_rows[i] for i in row_indices.tolist()]
    selected_receivers = torch.tensor(sorted({int(r["receiver_id"]) for r in subset_rows}), dtype=torch.long)
    labels_by_node = v5a._train_labels_by_node(supervision, cache.logits.size(0)).to(cache.device)
    baseline, baseline_by_node, full_loss, baseline_diag = _baseline_for_receivers(cache, selected_receivers, labels_by_node)
    matched_rows, matched_features = _matched_target_subset(cache, features, all_rows, row_indices, baseline,
                                                             baseline_by_node, labels_by_node)
    bundles = _bundle_replays(cache, matched_rows, matched_features, baseline_by_node, labels_by_node)
    comp_ok = all(math.isfinite(float(r["delta_comp_matched"])) for r in matched_rows if r["compensation_feasible"])
    zero_change = replay_local_basis_changes(cache, selected_receivers.to(cache.device), changes=[])
    zero_loss = F.cross_entropy(zero_change["logits"], labels_by_node.index_select(0, selected_receivers.to(cache.device)), reduction="none")
    preflight = {"dataset": dataset, "host_checkpoint": host_row["checkpoint_path"], "host_seed": 42,
                 "train_labels_read": True, "validation_labels_read": False, "test_labels_read": False, "test_metrics_read": False,
                 "receiver_count": len(selected_receivers), "edge_count": len(edge_ids), "max_edges_per_receiver": max_edges,
                 "singleton_rows": len(matched_rows), "bundle_rows": len(bundles),
                 "baseline_reconstruction": baseline_diag,
                 "zero_change_delta_max_abs": float((zero_loss - baseline["loss"]).abs().max()),
                 "delete_finite": bool(torch.isfinite(matched_features["target_delete"]).all()),
                 "comp_feasible_finite": comp_ok,
                 "compensation_feasible_rows": int(matched_features["target_comp_mask"].sum()),
                 "sensitivity_rows": sum(len(sensitivity_diagnostics(matched_rows, target)) for target in TARGETS),
                 "oracle_edge_groups": sum(sum(1 for r in oracle_group_rows(matched_rows, target)[0] if r["granularity"] == "edge") for target in TARGETS),
                 "bundle_interactions_finite": all(math.isfinite(float(r["interaction_delete_matched"])) for r in bundles),
                 "edge_pair_count_delete": int(make_edge_rank_pairs(matched_rows, matched_features["target_delete"])["left"].numel()),
                 "pointwise_feature_forward_finite": True, "ranker_score_forward_finite": True,
                 "sample_ids_crosschecked": False,
                 "finite": bool(torch.isfinite(matched_features["target_delete"]).all() and comp_ok and torch.isfinite(zero_loss).all())}
    # Exercise pointwise and scalar-ranker feature paths before the training smoke.
    for estimator in ESTIMATORS:
        if estimator == "E0_heuristic":
            args = {"scalar_features": matched_features["scalar"][:2]}
        elif estimator == "E1_unimodal_pair":
            pair = torch.where(matched_features["modality_code"][:2, None] == 0,
                               matched_features["pair_text"][:2], matched_features["pair_visual"][:2])
            args = {"scalar_features": matched_features["scalar"][:2], "pair_active": pair}
        else:
            args = {"scalar_features": matched_features["scalar"][:2], "pair_text": matched_features["pair_text"][:2],
                    "pair_visual": matched_features["pair_visual"][:2]}
            if estimator == "E3_host_context": args["host_context"] = matched_features["host"][:2]
        multi = MessageEffectEstimator(estimator, matched_features["scalar"].size(1),
                                       host_dim=matched_features["host"].size(1) if estimator == "E3_host_context" else None)
        single = SingleTargetMessageEffectEstimator(estimator, matched_features["scalar"].size(1),
                                                    host_dim=matched_features["host"].size(1) if estimator == "E3_host_context" else None)
        if not all(torch.isfinite(t).all() for t in args.values()):
            raise FloatingPointError("preflight feature path contains nonfinite values")
        multi_output = multi(**args)
        single_output = single(**args)
        if multi_output.shape != (2, 2) or single_output.shape != (2, 1) or not torch.isfinite(multi_output).all() or not torch.isfinite(single_output).all():
            raise FloatingPointError("preflight pointwise/ranker forward failed")
        pair_pred = single_output[:1].reshape(1)
        pair_loss = ranknet_pair_loss(pair_pred, pair_pred + 1, torch.ones_like(pair_pred)).mean()
        if not torch.isfinite(pair_loss):
            raise FloatingPointError("preflight RankNet loss is nonfinite")
        del multi, single, multi_output, single_output
    del checkpoint, cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return preflight


def _save_preflight(device: str) -> dict[str, Any]:
    result = _prepare_subset("Movies", device, receiver_limit=10, max_edges=2)
    result["protocol"] = "Movies R0 seed42; ten sampled train receivers; at most two incoming edges per receiver"
    result["requirements_passed"] = bool(result["finite"] and result["zero_change_delta_max_abs"] <= 1e-6 and result["delete_finite"] and result["comp_feasible_finite"])
    _write_json(DATA_DIR / "preflight_summary.json", result)
    return result


def _slice_for_receivers(features: dict[str, torch.Tensor], rows: list[dict[str, Any]], receivers: set[int]):
    idx = torch.tensor([i for i, r in enumerate(rows) if int(r["receiver_id"]) in receivers], dtype=torch.long)
    selected = {k: v.index_select(0, idx) for k, v in features.items()}
    return selected, [rows[i] for i in idx.tolist()]


def _smoke(device: str) -> dict[str, Any]:
    # Use a deterministic receiver subset large enough for disjoint train/val/holdout pairs.
    full_features, all_rows, _ = _load_v5a_cache("Movies")
    receiver_values = sorted(set(int(x) for x in full_features["receiver_id"].tolist()))[:240]
    subset_features, subset_rows = _slice_for_receivers(full_features, all_rows, set(receiver_values))
    # The smoke also verifies matched targets, generated on this smaller set using the same frozen replay code.
    host_row, checkpoint, cache, supervision = _host_and_train_labels("Movies", device)
    receiver_ids = torch.tensor(receiver_values, dtype=torch.long)
    labels_by_node = v5a._train_labels_by_node(supervision, cache.logits.size(0)).to(cache.device)
    baseline, baseline_by_node, _, _ = _baseline_for_receivers(cache, receiver_ids, labels_by_node)
    original_indices = torch.tensor([i for i, r in enumerate(all_rows) if int(r["receiver_id"]) in set(receiver_values)], dtype=torch.long)
    subset_rows, subset_features = _matched_target_subset(cache, full_features, all_rows, original_indices,
                                                            baseline, baseline_by_node, labels_by_node)
    del checkpoint, cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary = {"dataset": "Movies", "receiver_count": len(receiver_values), "row_count": len(subset_rows),
               "device": device, "max_epochs": 3, "patience": 10, "expected_runs": 20, "completed_runs": 0,
               "training_runs": [], "validation_selection": "EstimatorVal receiver-equal Huber or RankNet loss",
               "holdout_used_for_selection": False, "finite": True}
    for estimator in ESTIMATORS:
        result = train_pointwise("Movies", estimator, 0, "M", subset_features, subset_rows,
                                 device=device, max_epochs=3, patience=10)
        summary["training_runs"].append({"model": "M", "estimator": estimator, "target": "both",
                                         "best_epoch": result["best_epoch"], "best_validation_loss": result["best_validation_loss"]})
        summary["completed_runs"] += 1
        for target in TARGETS:
            result = train_pointwise("Movies", estimator, 0, "S", subset_features, subset_rows,
                                     target_name=target, device=device, max_epochs=3, patience=10)
            summary["training_runs"].append({"model": "S", "estimator": estimator, "target": target,
                                             "best_epoch": result["best_epoch"], "best_validation_loss": result["best_validation_loss"]})
            summary["completed_runs"] += 1
            result = train_ranker("Movies", estimator, target, 0, subset_features, subset_rows,
                                  device=device, max_epochs=3, patience=10)
            summary["training_runs"].append({"model": "Ranker", "estimator": estimator, "target": target,
                                             "best_epoch": result["best_epoch"], "best_validation_loss": result["best_validation_loss"]})
            summary["completed_runs"] += 1
        del result
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary["finite"] = summary["completed_runs"] == 20 and all(math.isfinite(float(r["best_validation_loss"])) for r in summary["training_runs"])
    summary["sample_note"] = "240 lowest-ID receivers from exact V5A sample; matched labels/replays only; heldout used only in smoke finite checks"
    _write_json(DATA_DIR / "smoke_summary.json", summary)
    return summary


def _write_dataset_metric_csv(path: Path, existing: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    _write_csv(path, existing + rows)


def _quantile_boundaries(values: torch.Tensor) -> list[float]:
    x = values.detach().cpu().float()
    x = x[torch.isfinite(x)].abs()
    if not x.numel(): return [float("nan")] * 3
    return [float(v) for v in torch.quantile(x, torch.tensor([.25, .5, .75])).tolist()]


def _effect_strata(rows: list[dict[str, Any]], features: dict[str, torch.Tensor], predictions: dict[str, list[float]],
                   train_receivers: set[int], holdout_receivers: set[int], dataset: str,
                   model: str, estimator: str, seed: int, target: str) -> list[dict[str, Any]]:
    target_values = features[f"target_{target}"]
    train_idx = torch.tensor([i for i, r in enumerate(rows) if int(r["receiver_id"]) in train_receivers and math.isfinite(float(target_values[i]))], dtype=torch.long)
    hold_idx = [i for i, r in enumerate(rows) if int(r["receiver_id"]) in holdout_receivers and math.isfinite(float(target_values[i]))]
    cuts = _quantile_boundaries(target_values.index_select(0, train_idx))
    labels = ("Q0-Q25", "Q25-Q50", "Q50-Q75", "Q75-Q100")
    out = []
    pred = predictions[target]
    for i, idx in enumerate(hold_idx):
        val = abs(float(target_values[idx]))
        stratum = int(np.searchsorted(cuts, val, side="right"))
        # Preserve assignment of exact cutoff ties into the upper adjacent stratum.
        rows[idx]["_magnitude_stratum"] = labels[min(stratum, 3)]
    for stratum, stratum_name in enumerate(labels):
        indices = [i for i in hold_idx if rows[i].get("_magnitude_stratum") == stratum_name]
        truths = [float(target_values[i]) for i in indices]
        preds = [float(pred[i]) for i in indices]
        metrics = _global_metrics(truths, preds)
        edge_pairs, modality_orders, hop_pairs = [], [], []
        edge_groups: dict[tuple[int, str, int], list[int]] = defaultdict(list)
        modality_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        hop_groups: dict[tuple[int, int, str], list[int]] = defaultdict(list)
        for idx in indices:
            row = rows[idx]
            edge_groups[(int(row["receiver_id"]), str(row["modality"]), int(row["hop"]))].append(idx)
            modality_groups[(int(row["receiver_id"]), int(row["directed_edge_index"]), int(row["hop"]))].append(idx)
            hop_groups[(int(row["receiver_id"]), int(row["directed_edge_index"]), str(row["modality"]))].append(idx)
        for group in edge_groups.values():
            for a, b in __import__("itertools").combinations(group, 2):
                td = float(target_values[a]) - float(target_values[b])
                if td != 0:
                    pd = float(pred[a]) - float(pred[b])
                    edge_pairs.append(.5 if pd == 0 else float((td < 0) == (pd < 0)))
        for group in modality_groups.values():
            if len(group) == 2:
                a, b = group
                td, pd = float(target_values[a]) - float(target_values[b]), float(pred[a]) - float(pred[b])
                modality_orders.append(1.0 if td == pd == 0 else (.5 if td == 0 or pd == 0 else float((td < 0) == (pd < 0))))
        for group in hop_groups.values():
            if len(group) == 4:
                for a, b in __import__("itertools").combinations(group, 2):
                    td, pd = float(target_values[a]) - float(target_values[b]), float(pred[a]) - float(pred[b])
                    if td != 0:
                        hop_pairs.append(.5 if pd == 0 else float((td < 0) == (pd < 0)))
        out.append({"dataset": dataset, "target": target, "model": model, "estimator": estimator, "seed": seed,
                    "stratum_type": "absolute_effect", "stratum": stratum, "stratum_name": stratum_name,
                    "train_q25": cuts[0], "train_q50": cuts[1], "train_q75": cuts[2],
                    "holdout_rows": len(indices), "mae": metrics["mae"], "rmse": metrics["rmse"],
                    "r2": metrics["r2"], "spearman": metrics["spearman"], "harmful_auroc": metrics["harmful_auroc"],
                    "harmful_auprc": metrics["harmful_auprc"], "harmful_prevalence": metrics["harmful_prevalence"],
                    "edge_pair_count": len(edge_pairs), "edge_pairwise_accuracy": float(np.mean(edge_pairs)) if edge_pairs else float("nan"),
                    "modality_ordering_accuracy": float(np.mean(modality_orders)) if modality_orders else float("nan"),
                    "hop_pair_count": len(hop_pairs), "hop_pairwise_accuracy": float(np.mean(hop_pairs)) if hop_pairs else float("nan")})
    return out


def _pair_gap_strata(rows: list[dict[str, Any]], features: dict[str, torch.Tensor], prediction: list[float],
                     train_receivers: set[int], holdout_receivers: set[int], dataset: str,
                     model: str, estimator: str, seed: int, target: str) -> list[dict[str, Any]]:
    values = features[f"target_{target}"].float()
    def collect(receiver_set: set[int]):
        groups: dict[tuple[int, str, int], list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            if int(row["receiver_id"]) in receiver_set and math.isfinite(float(values[i])):
                groups[(int(row["receiver_id"]), str(row["modality"]), int(row["hop"]))].append(i)
        result = []
        for members in groups.values():
            for a, b in __import__("itertools").combinations(members, 2):
                gap = abs(float(values[a]) - float(values[b]))
                if gap > 0: result.append((gap, a, b))
        return result
    train_pairs, hold_pairs = collect(train_receivers), collect(holdout_receivers)
    cuts = np.quantile([x[0] for x in train_pairs], [.25, .5, .75]).tolist() if train_pairs else [float("nan")] * 3
    out = []
    for group_id, label in enumerate(("Q0-Q25", "Q25-Q50", "Q50-Q75", "Q75-Q100")):
        selected = []
        for gap, a, b in hold_pairs:
            bucket = int(np.searchsorted(cuts, gap, side="right"))
            if min(bucket, 3) == group_id: selected.append((gap, a, b))
        correctness = [.5 if prediction[a] == prediction[b] else float((float(values[a]) < float(values[b])) == (prediction[a] < prediction[b]))
                       for _, a, b in selected]
        out.append({"dataset": dataset, "target": target, "model": model, "estimator": estimator, "seed": seed,
                    "stratum_type": "edge_pair_effect_gap", "stratum": group_id, "stratum_name": label,
                    "train_q25": cuts[0], "train_q50": cuts[1], "train_q75": cuts[2], "holdout_pairs": len(selected),
                    "edge_pairwise_accuracy": float(np.mean(correctness)) if correctness else float("nan")})
    return out


def _write_formal_artifacts(dataset_data: dict[str, Any], formal: dict[str, list[dict[str, Any]]]) -> None:
    dataset = dataset_data["dataset"]
    for name, rows in formal.items():
        path = DATA_DIR / f"{name}.csv"
        existing = _rows(path) if path.exists() and path.stat().st_size else []
        # Replacing this dataset's rows lets a campaign resume after interruption.
        existing = [r for r in existing if r.get("dataset") != dataset]
        _write_dataset_metric_csv(path, existing, rows)


def run_formal(device: str = "cuda:1", datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    all_data_summary, all_manifest_ds = [], []
    for dataset in datasets:
        prepared_path = OUTPUT / dataset / "matched_features.pt"
        if prepared_path.is_file():
            blob = torch.load(prepared_path, map_location="cpu", weights_only=False)
            features, rows = blob["features"], blob["rows"]
            # Formal target cache is accepted only after each row has explicit matched targets.
            if not rows or "delta_delete_matched" not in rows[0]:
                raise AssertionError(f"invalid prepared matched cache: {prepared_path}")
            data = {"dataset": dataset, "features": features, "rows": rows,
                    "v5a_split": blob.get("split_receivers")}
            if data["v5a_split"] is None:
                _, _, data["v5a_split"] = _load_v5a_cache(dataset)
            bundle_path = OUTPUT / dataset / "matched_bundle_raw.csv"
            bundle_rows = _rows(bundle_path) if bundle_path.is_file() else []
            data["bundles"] = bundle_rows
            summary_path = OUTPUT / dataset / "preparation_summary.json"
            if summary_path.is_file():
                data.update(json.loads(summary_path.read_text(encoding="utf-8")))
        else:
            print(f"V5A.1 {dataset}: preparing matched frozen-host effects", flush=True)
            data = prepare_dataset(dataset, device, save=True)
        features, rows = data["features"], data["rows"]
        split_indices, split_sets = row_splits(features, dataset)
        if {k: set(v) for k, v in split_sets.items()} != {k: set(map(int, v)) for k, v in data["v5a_split"].items()}:
            raise AssertionError(f"EstimatorTrain/Val/Holdout receiver split differs from V5A for {dataset}")
        formal = defaultdict(list)
        diagnostics = [dict(data.get("baseline_diagnostics", {}), dataset=dataset, diagnostic="baseline_local_vs_full")]
        for target in TARGETS:
            old = [r[f"delta_{target}_old"] for r in rows if r[f"delta_{target}_old"] is not None]
            matched = [r[f"delta_{target}_matched"] for r in rows if r[f"delta_{target}_matched"] is not None]
            diagnostic = matched_old_effect_diagnostics(matched, old)
            diagnostics.append({"dataset": dataset, "target": target, "diagnostic": "matched_vs_old_effect", **diagnostic})
            diagnostics.append({"dataset": dataset, "target": target, "diagnostic": "matched_minus_old_distribution",
                                **_finite_summary([r[f"delta_{target}_matched_minus_old"] for r in rows if r[f"delta_{target}_matched_minus_old"] is not None], "difference")})
            for sens in sensitivity_diagnostics(rows, target):
                formal["matched_effect_sensitivity"].append({"dataset": dataset, **sens})
            for oracle, summary in [oracle_group_rows(rows, target)]:
                formal["oracle_leverage"].extend({"dataset": dataset, **r} for r in summary)
            for bundle_type in sorted({str(r["bundle_type"]) for r in data.get("bundles", [])}):
                suffix = target
                bvalues = [float(r[f"interaction_{suffix}_matched"]) for r in data.get("bundles", [])
                           if r["bundle_type"] == bundle_type and r.get(f"interaction_{suffix}_matched") not in (None, "")]
                if bvalues:
                    diagnostics.append({"dataset": dataset, "target": target, "diagnostic": "bundle_interaction",
                                        "bundle_type": bundle_type, **_finite_summary(bvalues, "interaction")})
        formal["matched_effect_diagnostics"] = diagnostics
        # Train-only-derived effect-gap quartiles and stratum edges are part of the fixed protocol.
        stratum_rows = []
        point_metrics = []
        point_groups = []
        point_selection = []
        rank_metrics = []
        rank_sensitivity = []
        rank_selection = []
        for estimator in FEATURE_VARIANTS:
            for seed in ESTIMATOR_SEEDS:
                print(f"V5A.1 {dataset}: {estimator} seed={seed} MatchedMultiTarget", flush=True)
                ckpt_dir = OUTPUT / dataset / "checkpoints"
                m = train_pointwise(dataset, estimator, seed, "M", features, rows, device=device,
                                    checkpoint_path=str(ckpt_dir / f"{estimator}_M_seed{seed}.pt"))
                for target in TARGETS:
                    target_vec = features[f"target_{target}"].tolist()
                    hold_idx = m["holdout_indices"].tolist()
                    # M result predictions correspond to row indices in EstimatorHoldout order.
                    m_pred = [float("nan")] * len(rows)
                    for i, value in zip(hold_idx, m["predictions"][target]): m_pred[i] = value
                    meta = {"dataset": dataset, "model": "M", "estimator": estimator, "seed": seed, "target": target}
                    metric, group_rows, selection = pointwise_metrics(rows, target_vec, m_pred, **meta)
                    metric.update({"best_epoch": m["best_epoch"], "best_validation_loss_receiver_equal_huber": m["best_validation_loss"]})
                    point_metrics.append(metric); point_groups.extend(group_rows); point_selection.extend(selection)
                    formal["ranker_sensitivity"].extend(ranker_sensitivity(
                        rows, target_vec, m_pred, TAUS, dataset=dataset, model="M", estimator=estimator,
                        seed=seed, target=target))
                    stratum_rows.extend(_effect_strata(rows, features, {target: m_pred}, split_sets["EstimatorTrain"], split_sets["EstimatorHoldout"], dataset, "M", estimator, seed, target))
                    stratum_rows.extend(_pair_gap_strata(rows, features, m_pred, split_sets["EstimatorTrain"], split_sets["EstimatorHoldout"], dataset, "M", estimator, seed, target))
                    s = train_pointwise(dataset, estimator, seed, "S", features, rows, target_name=target, device=device,
                                        checkpoint_path=str(ckpt_dir / f"{estimator}_S_{target}_seed{seed}.pt"))
                    print(f"V5A.1 {dataset}: {estimator} seed={seed} SingleTarget/{target}", flush=True)
                    s_pred = [float("nan")] * len(rows)
                    for i, value in zip(s["holdout_indices"].tolist(), s["predictions"][target]): s_pred[i] = value
                    smeta = {"dataset": dataset, "model": "S", "estimator": estimator, "seed": seed, "target": target}
                    smetric, sgroups, sselection = pointwise_metrics(rows, target_vec, s_pred, **smeta)
                    smetric.update({"best_epoch": s["best_epoch"], "best_validation_loss_receiver_equal_huber": s["best_validation_loss"]})
                    point_metrics.append(smetric); point_groups.extend(sgroups); point_selection.extend(sselection)
                    formal["ranker_sensitivity"].extend(ranker_sensitivity(
                        rows, target_vec, s_pred, TAUS, dataset=dataset, model="S", estimator=estimator,
                        seed=seed, target=target))
                    stratum_rows.extend(_effect_strata(rows, features, {target: s_pred}, split_sets["EstimatorTrain"], split_sets["EstimatorHoldout"], dataset, "S", estimator, seed, target))
                    stratum_rows.extend(_pair_gap_strata(rows, features, s_pred, split_sets["EstimatorTrain"], split_sets["EstimatorHoldout"], dataset, "S", estimator, seed, target))
                    rank = train_ranker(dataset, estimator, target, seed, features, rows, device=device,
                                        checkpoint_path=str(ckpt_dir / f"{estimator}_Rank_{target}_seed{seed}.pt"))
                    print(f"V5A.1 {dataset}: {estimator} seed={seed} Ranker/{target}", flush=True)
                    rank_pred = rank["predictions"]
                    rmeta = {"dataset": dataset, "model": "Ranker", "estimator": estimator, "seed": seed, "target": target}
                    rmetric, _ = ranker_metrics(rows, target_vec, rank_pred, **rmeta)
                    rmetric.update({"best_epoch": rank["best_epoch"], "best_validation_loss_receiver_group_ranknet": rank["best_validation_loss"],
                                    "train_pair_count": rank["train_pair_count"], "train_pair_group_count": rank["train_pair_group_count"]})
                    rank_metrics.append(rmetric)
                    rank_sensitivity.extend(ranker_sensitivity(rows, target_vec, rank_pred, TAUS, **rmeta))
                    rank_selection.extend(_selection_for_ranker(rows, rank_pred, target, **rmeta)[0])
                    # Ranker scores are not calibrated globally. Magnitude stratification is edge-pair-only.
                    stratum_rows.extend(_pair_gap_strata(rows, features, rank_pred, split_sets["EstimatorTrain"], split_sets["EstimatorHoldout"], dataset, "Ranker", estimator, seed, target))
                    del s, rank
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                del m
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        comparisons = _pointwise_comparisons(dataset, point_metrics)
        rank_comparisons = _ranker_comparisons(dataset, point_selection, rank_selection)
        granularity = _granularity_summary(dataset, TARGETS, formal["oracle_leverage"], point_metrics, rank_metrics)
        dataset_summary = _dataset_summary(dataset, rows, point_metrics, rank_metrics, rank_selection, formal["oracle_leverage"])
        formal["pointwise_metrics"] = point_metrics
        formal["pointwise_group_metrics"] = point_groups
        formal["pointwise_comparisons"] = comparisons
        formal["ranker_metrics"] = rank_metrics
        formal["ranker_sensitivity"] = rank_sensitivity
        formal["ranker_selection_metrics"] = point_selection + rank_selection
        formal["ranker_comparisons"] = rank_comparisons
        formal["effect_magnitude_strata"] = stratum_rows
        formal["granularity_summary"] = granularity
        formal["dataset_summary"] = dataset_summary
        formal["audit_sample"] = _audit_sample(rows, dataset)
        _write_formal_artifacts(data, formal)
        all_data_summary.extend(dataset_summary)
        all_manifest_ds.append({"dataset": dataset, "receivers": len(split_sets["EstimatorTrain"]) + len(split_sets["EstimatorVal"]) + len(split_sets["EstimatorHoldout"]),
                                "rows": len(rows), "matched_comp_feasible_rows": int(features["target_comp_mask"].sum()),
                                "estimator_runs_completed": 60, "host_checkpoint": data.get("host_row", {}).get("checkpoint_path", "V4A R0_raw seed42"),
                                "sample_crosscheck": data.get("baseline_diagnostics", {}).get("v5a_raw_id_crosscheck", True)})
        _write_json(OUTPUT / "formal_progress.json", {"completed_datasets": [x["dataset"] for x in all_manifest_ds],
                                                        "dataset_runs": all_manifest_ds, "updated_at_utc": _utc_now()})
    return {"datasets": all_manifest_ds, "summary_rows": all_data_summary}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _selection_for_ranker(rows, predictions, target_name, **meta):
    from src.research.message_effect_v5a1 import selection_group_rows
    meta = dict(meta)
    meta.pop("target", None)
    return selection_group_rows(rows, predictions, target_name, model_name="Ranker", dataset=meta["dataset"],
                                estimator=meta["estimator"], seed=int(meta["seed"]))


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    return (float(vals.mean()), float(vals.std())) if vals.size else (float("nan"), float("nan"))


def _pointwise_comparisons(dataset: str, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("mae", "rmse", "spearman", "edge_pairwise_accuracy", "edge_top1_agreement", "edge_top1_excess",
              "modality_ordering_accuracy", "hop_pairwise_accuracy")
    lookup = {(r["model"], r["estimator"], r["target"], int(r["seed"])): r for r in metrics}
    output = []
    legacy_rows = _rows(RESEARCH.parent / "mag_message_effect_v5a_characterization" / "data" / "estimator_metrics.csv")
    legacy = {(r["estimator"], r["target"], int(r["seed"])): r for r in legacy_rows if r["dataset"] == dataset}
    for estimator in ESTIMATORS:
        for target in TARGETS:
            comparisons = (("S", "M", "matched target-sharing control"),
                           ("M", "V5A_MultiTarget_old_effect", "legacy target differs: old full-host baseline versus matched local baseline"),
                           ("S", "V5A_MultiTarget_old_effect", "legacy target differs: old full-host baseline versus matched local baseline"))
            for left, right, note in comparisons:
                for field in fields:
                    diffs = []
                    for seed in ESTIMATOR_SEEDS:
                        a = lookup.get((left, estimator, target, seed), {}).get(field)
                        if right == "V5A_MultiTarget_old_effect":
                            b = legacy.get((estimator, target, seed), {}).get(field)
                        else:
                            b = lookup.get((right, estimator, target, seed), {}).get(field)
                        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b)):
                            diffs.append(float(a) - float(b))
                    output.append({"dataset": dataset, "estimator": estimator, "target": target,
                                   "comparison": f"{left}-{right}", "metric": field, "n_seeds": len(diffs),
                                   "difference_mean": float(np.mean(diffs)) if diffs else float("nan"),
                                   "difference_std": float(np.std(diffs)) if diffs else float("nan"),
                                   "interpretation_note": note})
    ladder = (("E1_unimodal_pair", "E0_heuristic"), ("E2_multimodal_pair", "E1_unimodal_pair"),
              ("E3_host_context", "E2_multimodal_pair"))
    for model in ("M", "S"):
        for left, right in ladder:
            for target in TARGETS:
                for field in fields:
                    diffs = []
                    for seed in ESTIMATOR_SEEDS:
                        a = lookup.get((model, left, target, seed), {}).get(field)
                        b = lookup.get((model, right, target, seed), {}).get(field)
                        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b)):
                            diffs.append(float(a) - float(b))
                    output.append({"dataset": dataset, "model": model, "target": target,
                                   "estimator": f"{left}-{right}", "comparison": f"{left}-{right}", "metric": field,
                                   "n_seeds": len(diffs), "difference_mean": float(np.mean(diffs)) if diffs else float("nan"),
                                   "difference_std": float(np.std(diffs)) if diffs else float("nan"),
                                   "interpretation_note": "Fixed V5A feature-ladder comparison; post-hoc descriptive only."})
    return output


def _ranker_comparisons(dataset: str, point_selection: list[dict[str, Any]], rank_selection: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("edge_pairwise_accuracy", "top1_agreement", "top1_excess", "model_gain_vs_random", "oracle_regret")
    def aggregate(source, model):
        groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in source:
            if row.get("model") == model:
                groups[(row["estimator"], row["target"], int(row["seed"]))].append(row)
        out = {}
        for key, group in groups.items():
            for field in fields:
                out[(*key, field)] = float(np.mean([float(r[field]) for r in group])) if group else float("nan")
        return out
    left, right = aggregate(rank_selection, "Ranker"), aggregate(point_selection, "S")
    out = []
    for estimator in ESTIMATORS:
        for target in TARGETS:
            for field in fields:
                deltas = [left.get((estimator, target, seed, field), float("nan")) - right.get((estimator, target, seed, field), float("nan"))
                          for seed in ESTIMATOR_SEEDS]
                vals = [x for x in deltas if math.isfinite(x)]
                out.append({"dataset": dataset, "comparison": "Ranker-SingleTarget", "estimator": estimator, "target": target,
                            "metric": field, "n_seeds": len(vals), "ranker_minus_pointwise_mean": float(np.mean(vals)) if vals else float("nan"),
                            "ranker_minus_pointwise_std": float(np.std(vals)) if vals else float("nan")})
    return out


def _granularity_summary(dataset: str, targets: Iterable[str], oracle_rows: list[dict[str, Any]],
                         point_metrics: list[dict[str, Any]], rank_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for target in targets:
        for granularity in ("edge", "modality", "hop"):
            oracles = [r for r in oracle_rows if r["target"] == target and r["granularity"] == granularity]
            if granularity == "edge":
                point_field, rank_field = "edge_pairwise_accuracy", "edge_pairwise_accuracy"
            elif granularity == "modality":
                point_field, rank_field = "modality_ordering_accuracy", None
            else:
                point_field, rank_field = "hop_pairwise_accuracy", None
            pvalues = [float(r[point_field]) for r in point_metrics if r["target"] == target and math.isfinite(float(r.get(point_field, float("nan"))))]
            rvalues = [float(r[rank_field]) for r in rank_metrics if r["target"] == target and math.isfinite(float(r.get(rank_field, float("nan"))))] if rank_field else []
            out.append({"dataset": dataset, "target": target, "granularity": granularity,
                        "oracle_gain_mean": float(np.mean([float(r["oracle_gain_vs_random_mean"]) for r in oracles])) if oracles else float("nan"),
                        "best_pointwise_accuracy_posthoc": max(pvalues) if pvalues else float("nan"),
                        "best_pointwise_excess_over_chance_posthoc": max(pvalues) - .5 if pvalues else float("nan"),
                        "best_ranker_accuracy_posthoc": max(rvalues) if rvalues else float("nan"),
                        "best_ranker_excess_over_chance_posthoc": max(rvalues) - .5 if rvalues else float("nan"),
                        "selection_note": "best values are post-hoc descriptive summaries across fixed variants/seeds"})
    return out


def _dataset_summary(dataset: str, rows: list[dict[str, Any]], point_metrics: list[dict[str, Any]],
                     rank_metrics: list[dict[str, Any]], selection: list[dict[str, Any]], oracle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in TARGETS:
        p = [r for r in point_metrics if r["target"] == target]
        rk = [r for r in rank_metrics if r["target"] == target]
        sel = [r for r in selection if r.get("target") == target and r.get("model") == "Ranker"]
        lev = [r for r in oracle_rows if r["target"] == target]
        output.append({"dataset": dataset, "target": target, "matched_rows": sum(r.get(f"delta_{target}_matched") is not None for r in rows),
                       "matched_mean": float(np.mean([r[f"delta_{target}_matched"] for r in rows if r.get(f"delta_{target}_matched") is not None])),
                       "old_matched_spearman": _safe_corr([r[f"delta_{target}_old"] for r in rows if r.get(f"delta_{target}_matched") is not None],
                                                           [r[f"delta_{target}_matched"] for r in rows if r.get(f"delta_{target}_matched") is not None], rank=True),
                       "pointwise_M_edge_pairwise_mean": _safe_mean([r["edge_pairwise_accuracy"] for r in p if r.get("model") == "M"]),
                       "pointwise_S_edge_pairwise_mean": _safe_mean([r["edge_pairwise_accuracy"] for r in p if r.get("model") == "S"]),
                       "ranker_edge_pairwise_mean": _safe_mean([r["edge_pairwise_accuracy"] for r in rk]),
                       "ranker_top1_excess_mean": _safe_mean([r["edge_top1_excess_over_chance"] for r in rk]),
                       "ranker_model_gain_vs_random_mean": _safe_mean([r["model_gain_vs_random"] for r in sel]),
                       "ranker_oracle_regret_mean": _safe_mean([r["oracle_regret"] for r in sel]),
                       "edge_oracle_gain_mean": _safe_mean([r["oracle_gain_vs_random_mean"] for r in lev if r["granularity"] == "edge"]),
                       "modality_oracle_gain_mean": _safe_mean([r["oracle_gain_vs_random_mean"] for r in lev if r["granularity"] == "modality"]),
                       "hop_oracle_gain_mean": _safe_mean([r["oracle_gain_vs_random_mean"] for r in lev if r["granularity"] == "hop"])})
    return output


def _safe_mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_corr(left: Iterable[Any], right: Iterable[Any], rank: bool = False) -> float:
    l, r = list(left), list(right)
    return _safe_spearman(l, r) if rank else _safe_pearson(l, r)


def _audit_sample(rows: list[dict[str, Any]], dataset: str, limit: int = 500) -> list[dict[str, Any]]:
    if len(rows) <= limit: chosen = list(range(len(rows)))
    else:
        generator = np.random.default_rng(2027)
        chosen = sorted(generator.choice(len(rows), size=limit, replace=False).tolist())
    fields = ("receiver_id", "sender_id", "directed_edge_index", "modality", "hop", "delta_delete_old",
              "delta_delete_matched", "delta_comp_old", "delta_comp_matched", "compensation_feasible")
    return [{"dataset": dataset, **{k: rows[i].get(k) for k in fields}} for i in chosen]


def _environment(device: str) -> dict[str, Any]:
    cuda_mem = {}
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(idx)
            cuda_mem[str(idx)] = {"name": torch.cuda.get_device_name(idx), "total_bytes": int(total), "free_bytes": int(free),
                                  "used_bytes": int(total - free)}
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(), "requested_device": device,
            "cuda_device_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") and torch.cuda.is_available() else "CPU",
            "cuda_devices": torch.cuda.device_count(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "effect_target_labels": "train_only",
            "validation_labels_read": False, "test_labels_read": False, "test_metrics_read": False,
            "cuda_memory_at_stage_start": cuda_mem}


def _finalize_report() -> None:
    dataset_rows = _rows(DATA_DIR / "dataset_summary.csv") if (DATA_DIR / "dataset_summary.csv").is_file() else []
    diagnostics = _rows(DATA_DIR / "matched_effect_diagnostics.csv") if (DATA_DIR / "matched_effect_diagnostics.csv").is_file() else []
    sensitivity = _rows(DATA_DIR / "matched_effect_sensitivity.csv") if (DATA_DIR / "matched_effect_sensitivity.csv").is_file() else []
    oracle = _rows(DATA_DIR / "oracle_leverage.csv") if (DATA_DIR / "oracle_leverage.csv").is_file() else []
    pointwise = _rows(DATA_DIR / "pointwise_metrics.csv") if (DATA_DIR / "pointwise_metrics.csv").is_file() else []
    pointwise_cmp = _rows(DATA_DIR / "pointwise_comparisons.csv") if (DATA_DIR / "pointwise_comparisons.csv").is_file() else []
    ranker_cmp = _rows(DATA_DIR / "ranker_comparisons.csv") if (DATA_DIR / "ranker_comparisons.csv").is_file() else []
    def f(value: Any, digits: int = 4) -> str:
        try:
            value = float(value)
            return f"{value:.{digits}g}" if math.isfinite(value) else "NA"
        except (ValueError, TypeError):
            return "NA"
    txt = ["# V5A.1 Matched-Effect & Receiver-Local Ranking Repair", "",
           "## Protocol", "",
           "All reported effects are frozen-host exposed-message replay contrasts using the same local zero-change replay as their baseline. The V4A R0_raw seed-42 hosts, V5A receiver/edge sample, E0–E3 features, and train/validation/holdout receiver split are retained. No new MAG host or controller is trained.", "",
           "Only training receiver indices and training labels enter effect construction and estimator fitting. EstimatorVal receivers select checkpoints through receiver-equal Huber or RankNet loss; EstimatorHoldout receivers are used for final descriptive metrics. Validation labels, test labels, and test metrics were not read.", "",
           "## Main results", "",
           "| Dataset | Target | Matched−old correlation | M edge pairwise | S edge pairwise | Ranker pairwise | Ranker top1 excess | Ranker gain vs random | Ranker oracle regret | Edge oracle gain | Modality oracle gain | Hop oracle gain |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in dataset_rows:
        txt.append(f"| {r['dataset']} | {r['target']} | {f(r.get('old_matched_spearman'))} | {f(r.get('pointwise_M_edge_pairwise_mean'))} | {f(r.get('pointwise_S_edge_pairwise_mean'))} | {f(r.get('ranker_edge_pairwise_mean'))} | {f(r.get('ranker_top1_excess_mean'))} | {f(r.get('ranker_model_gain_vs_random_mean'), 3)} | {f(r.get('ranker_oracle_regret_mean'), 3)} | {f(r.get('edge_oracle_gain_mean'), 3)} | {f(r.get('modality_oracle_gain_mean'), 3)} | {f(r.get('hop_oracle_gain_mean'), 3)} |")
    txt.extend(["", "## Q1. Matched baseline effect", "",
                "| Dataset | Target | Matched−old Spearman | Median absolute difference | P95 absolute difference | Sign disagreement |", "|---|---|---:|---:|---:|---:|"])
    for r in diagnostics:
        if r.get("diagnostic") == "matched_vs_old_effect":
            txt.append(f"| {r.get('dataset')} | {r.get('target')} | {f(r.get('old_matched_spearman'))} | {f(r.get('matched_minus_old_median_abs'))} | {f(r.get('matched_minus_old_p95_abs'))} | {f(r.get('old_matched_sign_disagreement_fraction'))} |")
    txt.extend(["", "## Q2–Q3. Magnitude-aware sign sensitivity", "",
                "| Dataset | Target | τ | Text/Visual eligible fraction | Text/Visual sign disagreement | Hop strong mixed-sign fraction |", "|---|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for target in TARGETS:
            for tau in TAUS:
                tv = next((r for r in sensitivity if r.get("dataset") == dataset and r.get("target") == target and r.get("diagnostic") == "text_visual_sign" and float(r.get("tau", -1)) == tau), {})
                hop = next((r for r in sensitivity if r.get("dataset") == dataset and r.get("target") == target and r.get("diagnostic") == "hop_robust_sign" and float(r.get("tau", -1)) == tau), {})
                txt.append(f"| {dataset} | {target} | {tau:g} | {f(tv.get('eligible_fraction'))} | {f(tv.get('sign_disagreement_fraction'))} | {f(hop.get('strong_mixed_sign_fraction'))} |")
    txt.extend(["", "## Q4. Oracle leverage by action granularity", "",
                "| Dataset | Target | Granularity | Groups | Oracle gain mean | Median | P90 | Fraction > 1e−4 |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in oracle:
        txt.append(f"| {r.get('dataset')} | {r.get('target')} | {r.get('granularity')} | {r.get('groups_n')} | {f(r.get('oracle_gain_vs_random_mean'), 3)} | {f(r.get('oracle_gain_vs_random_median'), 3)} | {f(r.get('oracle_gain_vs_random_p90'), 3)} | {f(r.get('fraction_gain_gt_0.0001'))} |")
    txt.extend(["", "## Q5–Q6. Estimator and ranking comparisons", "",
                "MatchedSingleTarget−MatchedMultiTarget is paired by dataset, target, feature family, and seed. Ranker−SingleTarget is paired on the same comparison cells. Means and population standard deviations across estimator seeds are in `pointwise_comparisons.csv` and `ranker_comparisons.csv`. The V5A old-target comparisons are also included with a warning that the effect baseline changed.", "",
                "Holdout global and receiver-local pointwise metrics for every E0–E3 variant/seed are in `pointwise_metrics.csv`. Ranker metrics are restricted to within-group edge ranking in `ranker_metrics.csv`.", "",
                "## Q7–Q9. Selection quality and target differences", "",
                "Per-holdout-group selected effect, random expected effect, model gain, oracle gain, and oracle regret are reported in `ranker_selection_metrics.csv` for pointwise M/S and Ranker selections. The target-specific summaries above preserve Delete and Comp separately; compare these rows to assess their predictability and selection differences.", "",
                "## Q10. Interpretation", "",
                "All quantities describe frozen-host exposed-message replay contrasts. Oracle gain measures available selection leverage under a granularity, while model gain and regret measure what the fixed estimators selected. The reported results are descriptive, with no significance tests and no causal claim. A fine edge policy is supported only if its holdout edge-local ranking and selected-effect gain improve consistently; coarse modality/hop leverage alone does not establish that those actions are predictable.", "",
                "The sensitivity thresholds and effect/pair-gap quartile boundaries are diagnostics only. Quartile cutoffs are fitted from EstimatorTrain receivers and applied to EstimatorHoldout. The Ranker has no cross-group score calibration, so global Spearman, modality ordering, and hop ordering are not reported for it.", "",
                "See the CSV files for all fixed seeds and variants, bundle interactions, group-level selection records, and the at-most-500-row audit sample.", ""])
    (RESEARCH / "REPORT.md").write_text("\n".join(txt), encoding="utf-8")
    readme = """# V5A.1 experiment package\n\nThis package repairs the V5A target baseline with a matched local zero-change replay and compares receiver-equal matched multi-target/single-target pointwise models with a receiver-local RankNet scorer. The host, sample, feature ladder, and receiver split are inherited from V5A.\n\n- `REPORT.md`: protocol, aggregate results, and interpretation boundary.\n- `data/`: committed summaries, audits, diagnostics, and campaign records.\n- Full singleton and bundle rows, high-dimensional feature tensors, checkpoints, and logs remain under ignored `outputs/mag_message_effect_v5a1/`.\n\nAll effects remain frozen-host exposed-message replay contrasts. The receiver-local ranker is evaluated only within `(receiver, modality, hop)` candidate sets.\n"""
    (RESEARCH / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "smoke", "prepare", "formal", "all"), default="preflight")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    RESEARCH.mkdir(parents=True, exist_ok=True); DATA_DIR.mkdir(parents=True, exist_ok=True); OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(DATA_DIR / "environment.json", _environment(args.device))
    if args.stage in {"preflight", "all"}:
        preflight = _save_preflight(args.device)
        if not preflight["requirements_passed"]: raise RuntimeError("V5A.1 preflight failed")
    if args.stage in {"smoke", "all"}:
        smoke = _smoke(args.device)
        if not smoke["finite"]: raise RuntimeError("V5A.1 estimator smoke failed")
    if args.stage == "prepare":
        for dataset in args.datasets: prepare_dataset(dataset, args.device, save=True)
    if args.stage == "formal":
        manifest = run_formal(args.device, tuple(args.datasets))
        _write_json(DATA_DIR / "campaign_manifest.json", {
            "experiment": "V5A.1: Matched-Effect & Receiver-Local Ranking Repair",
            "branch": "exp/mag_message_effect_v5a1_ranking_repair", "parent_sha": "5b36b3dee2a7198f178b61e01b53fac255ee3651",
            "formal_freeze_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "host": "V4A R0_raw seed42, existing selected checkpoints, no retraining",
            "datasets": list(args.datasets), "sampling_max_receivers": 1500, "sampling_max_edges": 4,
            "sample_seed": SAMPLE_SEED, "estimator_seeds": list(ESTIMATOR_SEEDS),
            "runs_expected": 180, "runs_completed": 60 * len(args.datasets),
            "effect_target_labels": "train_only", "validation_labels_read": False,
            "test_labels_read": False, "test_metrics_read": False,
            "holdout_checkpoint_selection": False, "hpo": False, "significance_testing": False,
            "device": args.device, "dataset_runs": manifest["datasets"], "completed_at_utc": _utc_now()})
        _finalize_report()
    if args.stage in {"preflight", "smoke", "all"} and (DATA_DIR / "preflight_summary.json").exists():
        _write_json(DATA_DIR / "campaign_manifest.json", {
            "experiment": "V5A.1: Matched-Effect & Receiver-Local Ranking Repair",
            "branch": "exp/mag_message_effect_v5a1_ranking_repair", "parent_sha": "5b36b3dee2a7198f178b61e01b53fac255ee3651",
            "effect_target_labels": "train_only", "validation_labels_read": False,
            "test_labels_read": False, "test_metrics_read": False, "preflight_completed": True,
            "smoke_completed": (DATA_DIR / "smoke_summary.json").exists(), "formal_completed": False})


if __name__ == "__main__":
    main()
