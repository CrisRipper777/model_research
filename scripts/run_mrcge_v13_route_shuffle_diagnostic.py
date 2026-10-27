#!/usr/bin/env python3
"""Read-only frozen-checkpoint expert diversity and Router-Shuffle diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts import run_mrcge_v12_confirmation as v12
from scripts import run_mrcge_v13_screen as v13
from scripts import run_mrcge_v13_two_hop_cross_screen as v13_history
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _resolve_nc_eval_labels


STUDY = ROOT / "research/mrcge_v13_route_shuffle_diagnostic"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_route_shuffle_diagnostic"
BASE_SHA = "f24192b1233f09edfb22ccfe52a64333dbc463b4"
BASE_BRANCH = "exp/mrcge_v13_seed44_baseline_completion"
BRANCH = "exp/mrcge_v13_route_shuffle_diagnostic"
DATASETS = ("Movies", "Grocery")
SEED = 42
PERMUTATION_SEEDS = (1001, 1002, 1003)
MODEL_NAME = "mrcge_v13"
CONDITION = "M_repaired"
VARIANT = "joint_view"
EXPECTED_VIEWS = (
    ("private_text_physical", 0, 0, "physical"),
    ("private_text_semantic", 0, 1, "semantic"),
    ("private_visual_physical", 1, 0, "physical"),
    ("private_visual_semantic", 1, 1, "semantic"),
    ("collaborative_text_physical", 0, 0, "physical"),
    ("collaborative_visual_physical", 1, 0, "physical"),
)
CONDITIONS = {
    "shuffle_private_physical": ("private_text_physical", "private_visual_physical"),
    "shuffle_private_semantic": ("private_text_semantic", "private_visual_semantic"),
    "shuffle_collaborative_physical": ("collaborative_text_physical", "collaborative_visual_physical"),
    "shuffle_all_six": tuple(item[0] for item in EXPECTED_VIEWS),
}
TRAIN_RUNS = STUDY.parent / "mrcge_v13_seed44_completion/data/runs.csv"
V12_SELECTED = ROOT / "research/mrcge_v12_confirmation/data/combined_selected_runs.csv"
HISTORY_REPORT = ROOT / "research/mrcge_v13_view_global_router_ablation/RESULTS.md"
REQUIRED_FILES = (
    "src/models/mrcge_v13.py", "src/tasks/nc.py", "src/tasks/inference.py",
    "scripts/run_mrcge_v12_confirmation.py", "scripts/run_mrcge_v13_screen.py",
    "scripts/run_mrcge_v13_two_hop_cross_screen.py", "scripts/complete_mrcge_v13_seed44_baseline.py",
    "configs/model/mrcge_v13.yaml", "configs/task/nc.yaml", "configs/config.yaml",
    "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml", "src/data/loaders.py",
    "src/data/splits.py", "src/data/graph_utils.py", "src/data/semantic_candidates_v7b.py",
    "src/data/types.py", "src/utils/seeds.py",
    "research/mrcge_v13_seed44_completion/data/runs.csv",
    "research/mrcge_v13_seed44_completion/data/summary.csv",
    "research/mrcge_v13_view_global_router_ablation/RESULTS.md",
)

BASELINE_FIELDS = (
    "dataset", "seed", "model", "condition", "variant", "checkpoint_path", "checkpoint_sha256",
    "checkpoint_epoch", "val_nodes", "val_acc", "val_macro_f1", "val_ce",
    "validation_predictions_sha256", "validation_logits_sha256", "validation_class_counts_json",
    "history_val_acc", "history_val_macro_f1", "history_val_ce", "no_hook_hook_equal",
    "hook_control_equal", "hook_call_order_pass", "support_masks_pass", "model_weights_unchanged",
    "checkpoint_unchanged", "rng_unchanged", "test_labels_masked", "test_evaluation",
    "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint",
    "split_file_sha256", "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
)
INTERVENTION_FIELDS = (
    "dataset", "seed", "condition", "permutation_seed", "status", "target_views_json",
    "effective_nodes_json", "val_acc", "val_macro_f1", "val_ce",
    "delta_acc_pp", "delta_macro_f1_pp", "delta_ce",
    "validation_prediction_changed_fraction", "validation_logits_mean_abs_change",
    "router_checks_pass", "target_router_checks_json", "non_target_router_checks_pass",
    "hook_call_order_pass", "hook_calls", "test_evaluation", "checkpoint_sha256",
)
CONDITION_FIELDS = (
    "dataset", "condition", "n_permutations", "permutation_seeds", "status",
    "delta_acc_pp_mean", "delta_acc_pp_min", "delta_acc_pp_max", "delta_acc_direction",
    "delta_macro_f1_pp_mean", "delta_macro_f1_pp_min", "delta_macro_f1_pp_max", "delta_macro_f1_direction",
    "delta_ce_mean", "delta_ce_min", "delta_ce_max", "delta_ce_direction",
    "prediction_changed_fraction_mean", "prediction_changed_fraction_min", "prediction_changed_fraction_max",
    "logits_mean_abs_change_mean", "logits_mean_abs_change_min", "logits_mean_abs_change_max",
)
EXPERT_FIELDS = (
    "dataset", "seed", "view", "support_nodes", "max_order", "expert_a", "expert_b",
    "expert_a_output_rms", "expert_b_output_rms", "mean_node_cosine", "flattened_cosine",
    "normalized_rms_difference", "expert_outputs_masked_outside_support",
)
ROUTER_FIELDS = (
    "dataset", "seed", "view", "modality_id", "relation_id", "relation", "support_nodes",
    "importance_0", "importance_1", "importance_2", "importance_3",
    "selection_share_0", "selection_share_1", "selection_share_2", "selection_share_3",
    "top2_weight_mean_0", "top2_weight_mean_1", "top2_weight_mean_2", "top2_weight_mean_3",
    "top2_count_0", "top2_count_1", "top2_count_2", "top2_count_3",
    "top2_combination_count", "top2_combinations_json",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                    default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git(*args: str, text: bool = True):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text)


def source_sha() -> str:
    return git("rev-parse", "HEAD").stdout.strip()


def file_hashes() -> dict[str, str]:
    paths = [*REQUIRED_FILES, "scripts/run_mrcge_v13_route_shuffle_diagnostic.py"]
    return {path: sha256_file(ROOT / path) for path in paths}


def verify_base() -> None:
    if git("branch", "--show-current").stdout.strip() != BRANCH:
        raise RuntimeError(f"expected new diagnostic branch {BRANCH}")
    if git("merge-base", "--is-ancestor", BASE_SHA, "HEAD").returncode != 0:
        raise RuntimeError(f"working branch does not descend from {BASE_SHA}")
    for path in REQUIRED_FILES:
        base_blob = git("show", f"{BASE_SHA}:{path}", text=False).stdout
        if hashlib.sha256(base_blob).hexdigest() != sha256_file(ROOT / path):
            raise RuntimeError(f"frozen model/trainer/history source differs from base: {path}")


def data_info(data) -> dict[str, int]:
    return {"input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
            "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def expected_cfg(dataset: str, cache: dict[str, Any]):
    cfg = v13.make_cfg(dataset, VARIANT, SEED, "cpu", cache)
    required = {
        "name": MODEL_NAME, "condition": CONDITION, "v13_variant": VARIANT,
        "sender_private_ratio": 0.0, "physical_hops": 3, "semantic_hops": 1,
        "hidden_dim": 256, "num_experts": 4, "top_k": 2, "semantic_knn_k": 8,
        "dropout": 0.2, "late_cross_dropout": 0.15,
    }
    for name, expected in required.items():
        actual = cfg.model.get(name)
        if actual != expected and not (isinstance(expected, float) and math.isclose(float(actual), expected, abs_tol=1e-12)):
            raise RuntimeError(f"{dataset}: config {name} expected {expected!r}, got {actual!r}")
    if cfg.task.evaluate_test is not False or cfg.dataset.auto_generate_splits is not False:
        raise RuntimeError(f"{dataset}: Test evaluation or fixed-split protocol is not disabled/frozen")
    return cfg


def load_context(dataset: str, device: torch.device):
    rows = [row for row in read_rows(TRAIN_RUNS)
            if row.get("dataset") == dataset and row.get("variant") == VARIANT
            and row.get("seed") == str(SEED) and row.get("status") == "complete"]
    if len(rows) != 1:
        raise RuntimeError(f"{dataset}/seed42: expected one completed V1.3-A run row; found {len(rows)}")
    run_row = rows[0]
    checkpoint_path = Path(run_row["checkpoint_path"])
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else ""
    if not checkpoint_sha or checkpoint_sha != run_row["checkpoint_sha256"]:
        raise RuntimeError(f"{dataset}/seed42: checkpoint is absent or hash differs from runs.csv")
    cache = v13.cache_entries()[dataset]
    cfg = expected_cfg(dataset, cache)
    raw = load_mag_data(cfg, "nc", SEED)
    safe = v12.mask_test_labels(raw)
    fingerprint = v13.data_utils.dataset_fingerprints(raw)
    split_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
    if not split_text:
        raise RuntimeError(f"{dataset}: loader omitted the fixed split path")
    split_path = Path(split_text).resolve()
    if not split_path.is_file():
        raise RuntimeError(f"{dataset}: fixed split missing: {split_path}")
    split_sha = sha256_file(split_path)
    alignment = v13.assert_data_alignment(dataset, SEED, safe, cache, fingerprint, split_sha)
    checks = {
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprint["features"],
        "physical_edge_fingerprint": fingerprint["physical_edge_index"],
        "split_fingerprint": fingerprint["split_fingerprint"],
        "split_file_sha256": split_sha,
        "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
    }
    mismatch = {key: (run_row.get(key), value) for key, value in checks.items()
                if run_row.get(key) != value}
    if mismatch:
        raise RuntimeError(f"{dataset}: saved checkpoint fingerprints do not match current data/cache: {mismatch}")
    if run_row.get("split_file_path") and Path(run_row["split_file_path"]).resolve() != split_path:
        raise RuntimeError(f"{dataset}: split path differs from checkpoint run record")
    if cfg.task.evaluate_test is not False or bool((safe.y[raw.test_idx] != -100).any()):
        raise RuntimeError(f"{dataset}: Test evaluation must be off and Test labels masked")
    if not torch.isfinite(safe.x).all():
        raise RuntimeError(f"{dataset}: non-finite input feature")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("seed") != SEED:
        raise RuntimeError(f"{dataset}: checkpoint is not seed42 NC")
    if payload.get("protocol_version") != "unified_full_graph_nc_v1" or payload.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"{dataset}: checkpoint protocol or selection is not V1.3-A Best-Accuracy")
    provenance = payload.get("run_metadata", {}).get("study_provenance", {})
    if (provenance.get("dataset") != dataset or provenance.get("variant") != VARIANT
            or provenance.get("backbone_condition") != CONDITION
            or float(provenance.get("sender_private_ratio", -1.0)) != 0.0):
        raise RuntimeError(f"{dataset}: checkpoint model provenance is not the formal V1.3-A joint_view")
    current_model_source_sha = sha256_file(ROOT / "src/models/mrcge_v13.py")
    current_trainer_source_sha = sha256_file(ROOT / "src/tasks/nc.py")
    if provenance.get("task_source_sha256") != current_trainer_source_sha:
        raise RuntimeError(f"{dataset}: NC Trainer source differs from the historical checkpoint provenance")
    if run_row.get("config_sha256") != provenance.get("config_sha256"):
        raise RuntimeError(f"{dataset}: run-row and checkpoint config hashes differ")

    model = build_model(cfg, data_info(safe))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes))
    classifier.load_state_dict(payload["head_state"], strict=True)
    if model.num_experts != 4 or model.top_k != 2:
        raise RuntimeError(f"{dataset}: loaded architecture is not 4-expert Top-2")
    model = model.to(device).eval()
    classifier = classifier.to(device).eval()
    if any(parameter.dtype != torch.float32 for parameter in (*model.parameters(), *classifier.parameters())):
        raise RuntimeError(f"{dataset}: diagnostic must use FP32")
    expected_masks = build_expected_supports(model, safe.edge_index.to(device), int(safe.num_nodes), device)
    loaded_model_hash = state_hash(model)
    loaded_classifier_hash = state_hash(classifier)
    record = {
        "dataset": dataset, "seed": SEED, "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_selection": payload["selection"], "checkpoint_source_git_sha": provenance.get("source_git_sha"),
        "checkpoint_model_source_sha256": provenance.get("model_source_sha256"),
        "current_model_source_sha256": current_model_source_sha,
        "historical_model_source_matches_current": provenance.get("model_source_sha256") == current_model_source_sha,
        "protocol_version": payload["protocol_version"], "data_info": data_info(safe),
        "split_file_path": str(split_path), "alignment": alignment, "fingerprints": checks,
        "test_labels_masked": bool((safe.y[raw.test_idx] == -100).all()),
        "test_evaluation": False,
    }
    result = {"dataset": dataset, "cfg": cfg, "raw": raw, "data": safe, "cache": cache,
              "run_row": run_row, "payload": payload, "model": model, "classifier": classifier,
              "record": record, "expected_masks": expected_masks,
              "loaded_model_hash": loaded_model_hash, "loaded_classifier_hash": loaded_classifier_hash}
    return result


def build_expected_supports(model, physical_input: torch.Tensor, num_nodes: int,
                            device: torch.device) -> dict[tuple[int, int], torch.Tensor]:
    physical_edges, _, physical_support, _ = model._physical_operator(
        physical_input, num_nodes, torch.float32
    )
    del physical_edges
    expected: dict[tuple[int, int], torch.Tensor] = {}
    for modality_id, modality in enumerate(("text", "visual")):
        expected[(modality_id, 0)] = physical_support
        sem_edges = getattr(model, f"semantic_{modality}_edge_index")
        expected[(modality_id, 1)] = model._support(sem_edges, num_nodes, device)
    return expected


def state_hash(module: nn.Module) -> str:
    return v12.state_hash(module)


def rng_snapshot(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    return (torch.random.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None)


def rng_unchanged(before, device: torch.device) -> bool:
    after = rng_snapshot(device)
    return torch.equal(before[0], after[0]) and (device.type != "cuda" or torch.equal(before[1], after[1]))


def route_stats(logits: torch.Tensor, support: torch.Tensor) -> dict[str, Any]:
    active = logits[support].float()
    if active.numel() == 0:
        raise RuntimeError("a routed view has no supported nodes")
    dense = torch.softmax(active, dim=-1)
    top_logits, top_ids = torch.topk(active, k=2, dim=-1)
    top_weights = torch.softmax(top_logits, dim=-1)
    weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights)
    selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
    importance = dense.mean(dim=0)
    share = selected.float().mean(dim=0) / 2.0
    weight_mean = weights.mean(dim=0)
    counts = selected.sum(dim=0)
    combos = torch.sort(top_ids, dim=-1).values.detach().cpu().numpy()
    unique, combo_counts = np.unique(combos, axis=0, return_counts=True)
    return {
        "support_nodes": int(active.size(0)),
        "importance": importance.detach().cpu().tolist(),
        "selection_share": share.detach().cpu().tolist(),
        "top2_weight_mean": weight_mean.detach().cpu().tolist(),
        "top2_counts": counts.detach().cpu().tolist(),
        "top2_combinations": [
            {"experts": [int(x) for x in pair], "nodes": int(count)}
            for pair, count in zip(unique.tolist(), combo_counts.tolist())
        ],
        "top2_combination_count": int(len(unique)),
    }


def row_multiset_equal(before: torch.Tensor, after: torch.Tensor, active: torch.Tensor) -> bool:
    left = before[active].detach().float().cpu().contiguous().numpy()
    right = after[active].detach().float().cpu().contiguous().numpy()
    lu, lc = np.unique(left, axis=0, return_counts=True)
    ru, rc = np.unique(right, axis=0, return_counts=True)
    return np.array_equal(lu, ru) and np.array_equal(lc, rc)


def make_derangement(size: int, generator: torch.Generator) -> torch.Tensor:
    if size < 2:
        raise RuntimeError(f"cannot shuffle only {size} supported node")
    order = torch.randperm(size, generator=generator, device="cpu")
    perm = torch.empty_like(order)
    perm[order] = torch.roll(order, shifts=1)
    if bool((perm == torch.arange(size)).any()):
        raise RuntimeError("constructed shuffle unexpectedly contains fixed points")
    return perm


class HookCapture:
    def __init__(self, model, expected_masks: dict[tuple[int, int], torch.Tensor],
                 targets: set[str] | None = None, permutation_seed: int | None = None,
                 *, capture_experts: bool = False):
        self.model = model
        self.expected_masks = expected_masks
        self.targets = targets or set()
        self.permutation_seed = permutation_seed
        self.capture_experts = capture_experts
        self.bank_calls: list[dict[str, Any]] = []
        self.route_calls: list[dict[str, Any]] = []
        self.handles = []

    def _expert_statistics(self, outputs: torch.Tensor, support: torch.Tensor) -> list[dict[str, Any]]:
        if outputs.ndim != 3 or outputs.size(0) != 4:
            raise RuntimeError(f"expected final four-expert outputs [4,N,H], got {tuple(outputs.shape)}")
        if not bool(torch.isfinite(outputs).all()):
            raise RuntimeError("SharedExpertBank produced NaN/Inf final expert outputs")
        n, width = int(outputs.size(1)), int(outputs.size(2))
        active_count = int(support.sum().item())
        if active_count == 0:
            raise RuntimeError("shared expert call has an empty support mask")
        sq = torch.zeros(4, dtype=torch.float64, device=outputs.device)
        pair_dot = torch.zeros((4, 4), dtype=torch.float64, device=outputs.device)
        pair_norm_a = torch.zeros((4, 4), dtype=torch.float64, device=outputs.device)
        pair_norm_b = torch.zeros((4, 4), dtype=torch.float64, device=outputs.device)
        pair_cos_sum = torch.zeros((4, 4), dtype=torch.float64, device=outputs.device)
        pair_cos_count = 0
        pair_diff_sq = torch.zeros((4, 4), dtype=torch.float64, device=outputs.device)
        chunk = 8192
        masked_outside = True
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            local_mask = support[start:stop]
            block = outputs[:, start:stop]
            if bool((~local_mask).any()) and bool((block[:, ~local_mask] != 0).any()):
                masked_outside = False
            if not bool(local_mask.any()):
                continue
            values = block[:, local_mask].float()
            sq += values.square().sum(dim=(1, 2), dtype=torch.float64)
            pair_cos_count += int(values.size(1))
            for a in range(4):
                va = values[a]
                for b in range(a + 1, 4):
                    vb = values[b]
                    dot_node = (va * vb).sum(dim=-1)
                    norm_a_node = va.square().sum(dim=-1).sqrt()
                    norm_b_node = vb.square().sum(dim=-1).sqrt()
                    cosine = dot_node / (norm_a_node * norm_b_node).clamp_min(1.0e-12)
                    pair_cos_sum[a, b] += cosine.double().sum()
                    pair_dot[a, b] += dot_node.double().sum()
                    pair_norm_a[a, b] += va.square().sum(dtype=torch.float64)
                    pair_norm_b[a, b] += vb.square().sum(dtype=torch.float64)
                    pair_diff_sq[a, b] += (va - vb).square().sum(dtype=torch.float64)
        rms = (sq / float(active_count * width)).sqrt()
        rows = []
        rms_cpu = rms.detach().cpu().tolist()
        for a in range(4):
            for b in range(a + 1, 4):
                node_cos = float((pair_cos_sum[a, b] / pair_cos_count).detach().cpu())
                flat_cos = float((pair_dot[a, b] / (pair_norm_a[a, b] * pair_norm_b[a, b]).sqrt().clamp_min(1.0e-12)).detach().cpu())
                diff_rms = float((pair_diff_sq[a, b] / float(active_count * width)).sqrt().detach().cpu())
                norm_diff = diff_rms / max((rms_cpu[a] + rms_cpu[b]) * 0.5, 1.0e-12)
                rows.append({
                    "expert_a": a, "expert_b": b,
                    "expert_a_output_rms": float(rms_cpu[a]), "expert_b_output_rms": float(rms_cpu[b]),
                    "mean_node_cosine": node_cos, "flattened_cosine": flat_cos,
                    "normalized_rms_difference": norm_diff,
                    "expert_outputs_masked_outside_support": masked_outside,
                })
        if not masked_outside:
            raise RuntimeError("SharedExpertBank returned non-zero expert outputs outside Support Mask")
        return rows

    def _bank_hook(self, module, inputs, output):
        if len(inputs) != 3 or not isinstance(output, tuple) or len(output) != 4:
            raise RuntimeError("unexpected SharedExpertBank hook signature/output")
        support = inputs[1]
        max_order = int(inputs[2])
        if support.ndim != 1 or support.dtype != torch.bool:
            raise RuntimeError("SharedExpertBank support must be a one-dimensional bool mask")
        expert_stats = self._expert_statistics(output[0], support) if self.capture_experts else None
        self.bank_calls.append({"support": support.detach().clone(), "max_order": max_order,
                                "expert_stats": expert_stats})
        return None

    def _router_hook(self, module, inputs, output):
        index = len(self.route_calls)
        if index >= len(EXPECTED_VIEWS) or index >= len(self.bank_calls):
            raise RuntimeError("Router call arrived without the expected preceding filter_bank call")
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.size(1) != 4:
            raise RuntimeError("GraphAwareRouter must return [nodes,4] logits")
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("GraphAwareRouter produced NaN/Inf logits")
        _, expected_mid, expected_rid, relation = EXPECTED_VIEWS[index]
        current, first, cross, modality_id, relation_id = inputs
        modality_id, relation_id = int(modality_id), int(relation_id)
        if (modality_id, relation_id) != (expected_mid, expected_rid):
            raise RuntimeError(f"call {index}: actual router IDs {(modality_id, relation_id)} differ from source order {(expected_mid, expected_rid)}")
        view = EXPECTED_VIEWS[index][0]
        bank = self.bank_calls[index]
        if bank["support"].numel() != output.size(0) or current.size(0) != output.size(0):
            raise RuntimeError(f"{view}: logits/support/global node count mismatch")
        expected_support = self.expected_masks[(modality_id, relation_id)]
        support_matches = torch.equal(bank["support"], expected_support)
        expected_order = 3 if relation == "physical" else 1
        if bank["max_order"] != expected_order or not support_matches:
            raise RuntimeError(f"{view}: max-order/support mask differs from independently derived graph support")
        original = output
        support = bank["support"]
        before_stats = route_stats(original, support)
        row = {
            "view": view, "modality_id": modality_id, "relation_id": relation_id,
            "relation": relation, "support_nodes": int(support.sum().item()),
            "support_matches_graph": support_matches, "max_order": bank["max_order"],
            "before_stats": before_stats, "modified": False,
        }
        returned = original
        if view in self.targets:
            n_active = int(support.sum().item())
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.permutation_seed) * 1009 + index)
            permutation = make_derangement(n_active, generator)
            active = support.nonzero(as_tuple=False).flatten()
            shuffled = original.clone()
            shuffled[active] = original[active[permutation.to(active.device)]]
            row["permutation_fixed_points"] = int((permutation == torch.arange(n_active)).sum().item())
            row["permuted_rows"] = n_active
            row["changed_logit_rows"] = int((shuffled[active] != original[active]).any(dim=-1).sum().item())
            row["row_multiset_equal"] = row_multiset_equal(original, shuffled, support)
            after_stats = route_stats(shuffled, support)
            row["after_stats"] = after_stats
            row["effective_nodes"] = n_active
            row["checks"] = {
                "effective_node_count_equal": before_stats["support_nodes"] == after_stats["support_nodes"],
                "logit_row_multiset_equal": row["row_multiset_equal"],
                "nontrivial_derangement": row["permutation_fixed_points"] == 0 and row["changed_logit_rows"] > 0,
                "unsupported_logits_unchanged": torch.equal(shuffled[~support], original[~support]),
                "top2_selection_counts_equal": before_stats["top2_counts"] == after_stats["top2_counts"],
                "top2_combinations_equal": before_stats["top2_combinations"] == after_stats["top2_combinations"],
                "softmax_importance_equal": np.allclose(before_stats["importance"], after_stats["importance"], rtol=0, atol=1e-7),
                "top2_weight_stats_equal": np.allclose(before_stats["top2_weight_mean"], after_stats["top2_weight_mean"], rtol=0, atol=1e-7),
                "selection_share_equal": np.allclose(before_stats["selection_share"], after_stats["selection_share"], rtol=0, atol=1e-7),
            }
            row["modified"] = True
            if not all(row["checks"].values()):
                raise RuntimeError(f"{view}: route-distribution preservation check failed: {row['checks']}")
            returned = shuffled
        else:
            row["checks"] = {"non_target_logits_unchanged": torch.equal(original, returned)}
            if not row["checks"]["non_target_logits_unchanged"]:
                raise RuntimeError(f"{view}: non-target Router logits changed")
        self.route_calls.append(row)
        return returned

    def __enter__(self):
        self.handles.append(self.model.filter_bank.register_forward_hook(self._bank_hook))
        self.handles.append(self.model.router.register_forward_hook(self._router_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def validate_calls(self) -> None:
        if len(self.bank_calls) != 6 or len(self.route_calls) != 6:
            raise RuntimeError(f"expected exactly six filter/router calls, got {len(self.bank_calls)}/{len(self.route_calls)}")
        actual = [(row["view"], row["modality_id"], row["relation_id"]) for row in self.route_calls]
        expected = [(name, mid, rid) for name, mid, rid, _ in EXPECTED_VIEWS]
        if actual != expected:
            raise RuntimeError(f"six-call order differs from source: {actual}")
        if not all(row["support_matches_graph"] for row in self.route_calls):
            raise RuntimeError("at least one Support Mask did not match its graph-derived support")


def run_forward(context: dict[str, Any], device: torch.device, capture: HookCapture | None = None):
    model, classifier = context["model"], context["classifier"]
    data = context["data"]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    rng_before = rng_snapshot(device)
    if capture is None:
        with torch.no_grad():
            z, _, _, _, _ = model(x, edge_index, return_details=False)
            logits = classifier(z)
    else:
        with capture:
            with torch.no_grad():
                z, _, _, _, _ = model(x, edge_index, return_details=False)
                logits = classifier(z)
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("diagnostic forward produced NaN/Inf embeddings or logits")
    if capture is not None:
        capture.validate_calls()
    if not rng_unchanged(rng_before, device):
        raise RuntimeError("diagnostic forward changed the model inference RNG state")
    val_idx = data.val_idx.detach().cpu().long()
    val_logits = logits.detach()[val_idx.to(logits.device)].float().cpu()
    labels = data.y.detach().cpu()
    val_labels = labels[val_idx]
    f1_labels = _resolve_nc_eval_labels(data, include_test=False)
    metrics = v12.metric_bundle(val_logits, val_labels, torch.arange(val_idx.numel()), f1_labels)
    predictions = val_logits.argmax(dim=-1)
    del z, logits, x, edge_index
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return metrics, val_logits, predictions, (capture.route_calls if capture else None)


def route_csv_row(dataset: str, seed: int, route: dict[str, Any]) -> dict[str, Any]:
    result = {"dataset": dataset, "seed": seed, "view": route["view"],
              "modality_id": route["modality_id"], "relation_id": route["relation_id"],
              "relation": route["relation"], "support_nodes": route["support_nodes"],
              "top2_combination_count": route["top2_combination_count"],
              "top2_combinations_json": json.dumps(route["top2_combinations"], separators=(",", ":"))}
    for name, values in (("importance", route["importance"]),
                         ("selection_share", route["selection_share"]),
                         ("top2_weight_mean", route["top2_weight_mean"]),
                         ("top2_count", route["top2_counts"])):
        for expert, value in enumerate(values):
            result[f"{name}_{expert}"] = value
    return result


def prediction_logits_hash(logits: torch.Tensor) -> str:
    return hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def baseline_rows(context: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = context["dataset"]
    model, classifier = context["model"], context["classifier"]
    model_hash = state_hash(model)
    classifier_hash = state_hash(classifier)
    checkpoint_hash = sha256_file(context["record"]["checkpoint_path"])
    plain_metrics, plain_logits, plain_preds, _ = run_forward(context, device)
    tolerance = 3e-6
    historical = context["run_row"]
    reproduced = all(abs(plain_metrics[key] - float(historical[source])) <= tolerance for key, source in (
        ("acc", "val_acc"), ("macro_f1", "val_macro_f1"), ("ce", "val_ce")))
    if not reproduced:
        raise RuntimeError(f"{dataset}: frozen checkpoint did not reproduce historical Validation metrics: {plain_metrics}")
    hooked = HookCapture(model, context["expected_masks"], capture_experts=True)
    hooked_metrics, hooked_logits, hooked_preds, routes = run_forward(context, device, hooked)
    hook_control_equal = (all(abs(plain_metrics[key] - hooked_metrics[key]) <= tolerance for key in plain_metrics)
                          and torch.equal(plain_preds, hooked_preds)
                          and torch.allclose(plain_logits, hooked_logits, rtol=0, atol=tolerance))
    if not hook_control_equal:
        raise RuntimeError(f"{dataset}: pass-through Hooks changed baseline outputs")
    if state_hash(model) != model_hash or state_hash(classifier) != classifier_hash:
        raise RuntimeError(f"{dataset}: baseline Hook/forward modified model or classifier parameters")
    if sha256_file(context["record"]["checkpoint_path"]) != checkpoint_hash:
        raise RuntimeError(f"{dataset}: checkpoint changed during inference")
    expert_rows = []
    router_rows = []
    for call, route in zip(hooked.bank_calls, routes):
        for pair in call["expert_stats"]:
            expert_rows.append({"dataset": dataset, "seed": SEED, "view": route["view"],
                                "support_nodes": route["support_nodes"], "max_order": call["max_order"], **pair})
        router_rows.append(route_csv_row(dataset, SEED, {**route, **route["before_stats"]}))
    fp = context["record"]["fingerprints"]
    record = {
        "dataset": dataset, "seed": SEED, "model": MODEL_NAME, "condition": CONDITION,
        "variant": VARIANT, "checkpoint_path": context["record"]["checkpoint_path"],
        "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": context["record"]["checkpoint_epoch"],
        "val_nodes": int(plain_preds.numel()), "val_acc": plain_metrics["acc"],
        "val_macro_f1": plain_metrics["macro_f1"], "val_ce": plain_metrics["ce"],
        "validation_predictions_sha256": prediction_logits_hash(plain_preds),
        "validation_logits_sha256": prediction_logits_hash(plain_logits),
        "validation_class_counts_json": json.dumps(torch.bincount(
            plain_preds, minlength=int(context["data"].num_classes)).tolist(), separators=(",", ":")),
        "history_val_acc": float(historical["val_acc"]), "history_val_macro_f1": float(historical["val_macro_f1"]),
        "history_val_ce": float(historical["val_ce"]), "no_hook_hook_equal": hook_control_equal,
        "hook_control_equal": hook_control_equal, "hook_call_order_pass": True, "support_masks_pass": True,
        "model_weights_unchanged": True, "checkpoint_unchanged": True, "rng_unchanged": True,
        "test_labels_masked": True, "test_evaluation": False, **fp,
    }
    context["baseline_val_logits"] = plain_logits
    context["baseline_val_predictions"] = plain_preds
    return record, expert_rows, router_rows


def run_intervention(context: dict[str, Any], device: torch.device,
                     condition: str, permutation_seed: int,
                     baseline: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = set(CONDITIONS[condition])
    capture = HookCapture(context["model"], context["expected_masks"], targets,
                          permutation_seed, capture_experts=False)
    metrics, logits, predictions, route_calls = run_forward(context, device, capture)
    base_logits = context["baseline_val_logits"]
    base_predictions = context["baseline_val_predictions"]
    if logits.shape != base_logits.shape or predictions.shape != base_predictions.shape:
        raise RuntimeError("Validation prediction/logit shape changed during router intervention")
    target_audits = []
    targets_seen = set()
    target_nodes = {}
    for route in route_calls:
        view = route["view"]
        if route["modified"]:
            targets_seen.add(view)
            target_nodes[view] = route["effective_nodes"]
            target_audits.append({"view": view, "support_nodes": route["effective_nodes"],
                                  "checks": route["checks"], "fixed_points": route["permutation_fixed_points"],
                                  "changed_logit_rows": route["changed_logit_rows"],
                                  "row_multiset_equal": route["row_multiset_equal"]})
    if targets_seen != targets:
        raise RuntimeError(f"{condition}: intended target views differ from actual hooks: {targets_seen} vs {targets}")
    non_target_ok = all(all(value.values()) for route in route_calls if not route["modified"]
                        for value in [route["checks"]])
    if not non_target_ok:
        raise RuntimeError(f"{condition}: at least one non-target Router output changed")
    checks_ok = all(all(audit["checks"].values()) for audit in target_audits)
    if not checks_ok:
        raise RuntimeError(f"{condition}: route-distribution checks failed")
    row = {
        "dataset": context["dataset"], "seed": SEED, "condition": condition,
        "permutation_seed": permutation_seed, "status": "complete",
        "target_views_json": json.dumps(sorted(targets), separators=(",", ":")),
        "effective_nodes_json": json.dumps(target_nodes, sort_keys=True, separators=(",", ":")),
        "val_acc": metrics["acc"], "val_macro_f1": metrics["macro_f1"], "val_ce": metrics["ce"],
        "delta_acc_pp": (metrics["acc"] - float(baseline["val_acc"])) * 100.0,
        "delta_macro_f1_pp": (metrics["macro_f1"] - float(baseline["val_macro_f1"])) * 100.0,
        "delta_ce": metrics["ce"] - float(baseline["val_ce"]),
        "validation_prediction_changed_fraction": float((predictions != base_predictions).float().mean()),
        "validation_logits_mean_abs_change": float((logits - base_logits).abs().mean()),
        "router_checks_pass": checks_ok, "target_router_checks_json": json.dumps(target_audits, sort_keys=True, separators=(",", ":")),
        "non_target_router_checks_pass": non_target_ok, "hook_call_order_pass": True,
        "hook_calls": len(route_calls), "test_evaluation": False,
        "checkpoint_sha256": context["record"]["checkpoint_sha256"],
    }
    if not all(math.isfinite(float(row[key])) for key in
               ("delta_acc_pp", "delta_macro_f1_pp", "delta_ce",
                "validation_prediction_changed_fraction", "validation_logits_mean_abs_change")):
        raise RuntimeError(f"{condition}: intervention metrics include NaN/Inf")
    return row, {"metrics": metrics, "route_calls": route_calls}


def direction(values: list[float], *, higher_is_better: bool = True) -> str:
    mean = sum(values) / len(values)
    if abs(mean) < 1e-10:
        return "unchanged"
    if higher_is_better:
        return "improved" if mean > 0 else "worsened"
    return "improved" if mean < 0 else "worsened"


def summarize_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            cells = [row for row in rows if row["dataset"] == dataset and row["condition"] == condition
                     and row["status"] == "complete"]
            if not cells:
                output.append({"dataset": dataset, "condition": condition, "n_permutations": 0,
                               "permutation_seeds": "", "status": "not_run"})
                continue
            item: dict[str, Any] = {"dataset": dataset, "condition": condition,
                                    "n_permutations": len(cells),
                                    "permutation_seeds": ";".join(str(x) for x in PERMUTATION_SEEDS if any(int(c["permutation_seed"]) == x for c in cells)),
                                    "status": "complete" if len(cells) == 3 else "partial"}
            for metric, higher in (("delta_acc_pp", True), ("delta_macro_f1_pp", True),
                                   ("delta_ce", False), ("validation_prediction_changed_fraction", None),
                                   ("validation_logits_mean_abs_change", None)):
                values = [float(row[metric]) for row in cells]
                item[f"{metric}_mean"] = sum(values) / len(values)
                item[f"{metric}_min"] = min(values)
                item[f"{metric}_max"] = max(values)
                if higher is not None:
                    item[f"{metric}_direction"] = direction(values, higher_is_better=higher)
            output.append(item)
    return output


def recommendation(summary: list[dict[str, Any]]) -> str:
    if len(summary) != 8 or any(row.get("status") != "complete" for row in summary):
        return "INCONCLUSIVE"
    negligible = all(
        abs(float(row["delta_acc_pp_mean"])) < 0.2
        and abs(float(row["delta_macro_f1_pp_mean"])) < 0.2
        and abs(float(row["delta_ce_mean"])) < 0.01
        and float(row["validation_prediction_changed_fraction_mean"]) < 0.01
        for row in summary
    )
    if negligible:
        return "OPTIONAL_SIMPLIFICATION"
    materially_degraded = any(
        float(row["delta_acc_pp_mean"]) <= -0.5
        or float(row["delta_macro_f1_pp_mean"]) <= -0.5
        or float(row["delta_ce_mean"]) >= 0.02
        for row in summary
    )
    return "KEEP_CURRENT_ROUTER" if materially_degraded else "INCONCLUSIVE"


def protocol_payload() -> dict[str, Any]:
    return {
        "checkpoint_seed": SEED, "datasets": list(DATASETS), "model": MODEL_NAME,
        "condition": CONDITION, "variant": VARIANT, "sender_private_ratio": 0.0,
        "physical_hops": 3, "semantic_hops": 1, "shared_experts": 4, "top_k": 2,
        "conditions": list(CONDITIONS), "permutation_seeds": list(PERMUTATION_SEEDS),
        "planned_baselines": 2, "planned_interventions": 24,
        "planned_full_forwards": 26, "additional_pass_through_hook_controls": 2,
        "shuffle": "within-view permutation of supported-node Router-logit rows only",
        "device_dtype": "FP32", "model_mode": "eval", "gradient_mode": "no_grad",
        "test_evaluation": False, "test_label_mask": -100,
    }


def initial_manifest() -> dict[str, Any]:
    return {
        "study": "MR-CGE V1.3-A Frozen Router-Shuffle & Expert Diversity Diagnostic",
        "status": "preflight_pending", "recorded_utc": now(), "branch": BRANCH,
        "base_branch": BASE_BRANCH, "base_sha": BASE_SHA, "execution_sha": source_sha(),
        "source_file_sha256s": file_hashes(), "protocol": protocol_payload(),
        "gpu": {"status": "not_checked", "attempts": []},
        "checkpoints": {}, "datasets": {}, "successful_units": [], "failed_units": [],
        "unstarted_units": [f"{d}/seed42/{unit}" for d in DATASETS for unit in
                             (["baseline"] + [f"{c}/perm{s}" for c in CONDITIONS for s in PERMUTATION_SEEDS])],
        "test_evaluation": False, "link_prediction": False,
    }


def preflight() -> dict[str, Any]:
    verify_base()
    DATA.mkdir(parents=True, exist_ok=True)
    for filename, fields in (("baseline.csv", BASELINE_FIELDS), ("interventions.csv", INTERVENTION_FIELDS),
                             ("condition_summary.csv", CONDITION_FIELDS), ("expert_diversity.csv", EXPERT_FIELDS),
                             ("router_summary.csv", ROUTER_FIELDS)):
        write_rows(DATA / filename, [], fields)
    manifest = initial_manifest()
    devices = {}
    for dataset in DATASETS:
        context = load_context(dataset, torch.device("cpu"))
        model, classifier = context["model"], context["classifier"]
        devices[dataset] = {**context["record"],
                            "model_parameters": sum(p.numel() for p in model.parameters()),
                            "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
                            "model_type": type(model).__name__, "shared_expert_objects": len({id(e) for e in model.filter_bank.experts}),
                            "router_output_dim": int(model.router.net[-1].out_features),
                            "max_orders_from_config": {"physical": int(model.physical_hops), "semantic": int(model.semantic_hops)}}
        del context
    manifest.update({"status": "preflight_passed", "preflight_utc": now(),
                     "preflight_source_sha": source_sha(), "preflight_source_file_sha256s": file_hashes(),
                     "checkpoints": devices,
                     "source_call_order_audit": [
                         {"index": i, "view": name, "modality_id": mid, "relation_id": rid,
                          "relation": relation, "filter_bank_precedes_router": True}
                         for i, (name, mid, rid, relation) in enumerate(EXPECTED_VIEWS)],
                     "hook_removal": "try/finally in each hooked forward",
                     "test_evaluation": False, "test_label_mask": -100})
    json_write(DATA / "manifest.json", manifest)
    return manifest


def append_progress(dataset: str, baseline_rows_new: list[dict[str, Any]],
                    expert_rows_new: list[dict[str, Any]], router_rows_new: list[dict[str, Any]],
                    intervention_rows_new: list[dict[str, Any]]) -> None:
    write_rows(DATA / "baseline.csv", baseline_rows_new, BASELINE_FIELDS)
    write_rows(DATA / "expert_diversity.csv", expert_rows_new, EXPERT_FIELDS)
    write_rows(DATA / "router_summary.csv", router_rows_new, ROUTER_FIELDS)
    write_rows(DATA / "interventions.csv", intervention_rows_new, INTERVENTION_FIELDS)


def build_report(manifest: dict[str, Any], baseline_rows_all: list[dict[str, Any]],
                 intervention_rows_all: list[dict[str, Any]], summary_rows: list[dict[str, Any]],
                 expert_rows_all: list[dict[str, Any]]) -> None:
    rec = recommendation(summary_rows)
    lines = [
        "# MR-CGE V1.3-A Frozen Router-Shuffle 与专家差异诊断", "",
        "## 范围与完整性", "",
        f"状态：**{manifest.get('status')}**。只分析 Movies/Grocery seed42 的既有 `mrcge_v13 / M_repaired / joint_view` Best-Validation-Accuracy checkpoint。未重新训练、未执行 Test 指标、未运行 LP。", "",
        "计划包含 2 个无干预 baseline 与 24 个 Router-Shuffle 推理；另有每个数据集一次 pass-through Hook 控制，用来验证 Hook 不改变 baseline。所有干预仅置换同一视图内 Support=True 节点的四维 Router-logit 行；每个视图使用独立 CPU Generator。", "",
        "## Baseline 复现", "",
        "| Dataset | Val Acc | Macro-F1 | CE | 历史差异最大值 | Hook 控制一致 |", "|---|---:|---:|---:|---:|---|",
    ]
    for row in baseline_rows_all:
        diffs = [abs(float(row[k]) - float(row[h])) for k, h in
                 (("val_acc", "history_val_acc"), ("val_macro_f1", "history_val_macro_f1"), ("val_ce", "history_val_ce"))]
        lines.append(f"| {row['dataset']} | {100*float(row['val_acc']):.3f}% | {100*float(row['val_macro_f1']):.3f}% | {float(row['val_ce']):.5f} | {max(diffs):.2g} | {row['hook_control_equal']} |")
    lines += ["", "## 专家最终输出差异", "",
              "以下输出 RMS 差异使用 `RMS(Ea−Eb) / mean(RMS(Ea), RMS(Eb))`；cosine 只统计 Support=True 节点。Flattened cosine 在所有有效节点与通道上展平。它们描述表示差异，不赋予专家明确物理/频率语义。", "",
              "| Dataset | Mean pair node cosine | Mean flattened cosine | Mean normalized RMS difference |", "|---|---:|---:|---:|"]
    for dataset in DATASETS:
        rows = [r for r in expert_rows_all if r["dataset"] == dataset]
        if rows:
            lines.append(f"| {dataset} | {np.mean([float(r['mean_node_cosine']) for r in rows]):.4f} | {np.mean([float(r['flattened_cosine']) for r in rows]):.4f} | {np.mean([float(r['normalized_rms_difference']) for r in rows]):.4f} |")
        else:
            lines.append(f"| {dataset} | 未运行 | 未运行 | 未运行 |")
    lines += ["", "每视图的专家 RMS 与六对逐节点/全局 cosine、归一化 RMS 差异详见 `data/expert_diversity.csv`。路由选择份额、Top-2 组合数与 importance 见 `data/router_summary.csv`。", "",
              "## 冻结 Router-Shuffle 变化", "",
              "| Dataset | Condition | ΔAcc mean [min,max] pp | ΔMacro-F1 mean [min,max] pp | ΔCE mean [min,max] | 预测改变率 mean | Logit MAE mean |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        if row.get("n_permutations", 0):
            lines.append(f"| {row['dataset']} | {row['condition']} | {float(row['delta_acc_pp_mean']):+.3f} [{float(row['delta_acc_pp_min']):+.3f}, {float(row['delta_acc_pp_max']):+.3f}] | {float(row['delta_macro_f1_pp_mean']):+.3f} [{float(row['delta_macro_f1_pp_min']):+.3f}, {float(row['delta_macro_f1_pp_max']):+.3f}] | {float(row['delta_ce_mean']):+.5f} [{float(row['delta_ce_min']):+.5f}, {float(row['delta_ce_max']):+.5f}] | {100*float(row['validation_prediction_changed_fraction_mean']):.2f}% | {float(row['logits_mean_abs_change_mean']):.5f} |")
        else:
            lines.append(f"| {row['dataset']} | {row['condition']} | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |")
    lines += ["", "## 解释与建议", "",
              "专家输出是否不同、冻结节点路由是否影响验证预测、以及是否保留节点 Router 是三个不同判断。Shuffle 是对已训练 checkpoint 的推理敏感性诊断，不是重训静态 Router、因果识别、显著性检验或 Test 泛化证据。", "",
              f"建议：**`{rec}`**。此建议仅供下一轮人工研究判断；本任务不据此修改模型。", "",
              "## 核验与偏差", "",
              "- Baseline 与历史 runs.csv 的 Accuracy/Macro-F1/CE 容差为 3e-6；先做无 Hook，再做 pass-through Hook 对照。Baseline 不匹配的数据集不会继续 Shuffle。",
              "- 每次 Hook 都检查六次调用顺序、实际 modality/relation ID、Physical/Semantic Support Mask 与 hop；Hook 在 `finally` 中移除。",
              "- 每个目标视图检查非恒等节点行置换、Logit 行多重集不变、Top-2 专家计数/组合不变、softmax importance 与 Top-2 权重统计不变；非目标视图 logits 必须逐元素不变。",
              "- checkpoint/model/classifier 哈希与 RNG 在诊断前后核验；Test 标签为 -100，Test evaluation=false。", "",
              f"实际 GPU：{manifest.get('gpu', {}).get('selected_physical_gpu', '未运行/未通过守卫')}。资源守卫状态：{manifest.get('gpu', {}).get('status', 'not checked')}。执行 Commit：`{manifest.get('execution_sha', '')}`。", "",
              "单元完成/失败/未启动数量及其原因见 `data/manifest.json`；干预逐排列数据见 `data/interventions.csv`。", ""]
    (STUDY / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_outputs(manifest: dict[str, Any], baseline: list[dict[str, Any]],
                     interventions: list[dict[str, Any]], experts: list[dict[str, Any]],
                     routers: list[dict[str, Any]]) -> None:
    summary = summarize_conditions(interventions)
    write_rows(DATA / "baseline.csv", baseline, BASELINE_FIELDS)
    write_rows(DATA / "interventions.csv", interventions, INTERVENTION_FIELDS)
    write_rows(DATA / "condition_summary.csv", summary, CONDITION_FIELDS)
    write_rows(DATA / "expert_diversity.csv", experts, EXPERT_FIELDS)
    write_rows(DATA / "router_summary.csv", routers, ROUTER_FIELDS)
    manifest["condition_summary"] = summary
    manifest["recommendation"] = recommendation(summary)
    manifest["successful_units"] = [f"{r['dataset']}/seed42/baseline" for r in baseline]
    manifest["successful_units"] += [f"{r['dataset']}/seed42/{r['condition']}/perm{r['permutation_seed']}"
                                     for r in interventions if r.get("status") == "complete"]
    planned = [f"{d}/seed42/{unit}" for d in DATASETS for unit in
               (["baseline"] + [f"{c}/perm{s}" for c in CONDITIONS for s in PERMUTATION_SEEDS])]
    done = set(manifest["successful_units"]) | set(manifest.get("failed_units", []))
    manifest["unstarted_units"] = [unit for unit in planned if unit not in done]
    completed = len(baseline) == 2 and len(interventions) == 24 and all(r.get("status") == "complete" for r in interventions)
    if completed:
        manifest["status"] = "complete"
    elif manifest.get("status") != "blocked_resource":
        manifest["status"] = "partial" if baseline or interventions else "failed"
    manifest["completed_utc"] = now()
    manifest["planned_unit_counts"] = {"baselines": 2, "interventions": 24, "hook_controls": 2,
                                        "formal_full_forwards": 26, "hook_controls_excluded_from_formal_count": True}
    manifest["test_evaluation"] = False
    manifest["link_prediction"] = False
    json_write(DATA / "manifest.json", manifest)
    build_report(manifest, baseline, interventions, summary, experts)


def compact_gpu(snapshot: dict[str, Any], physical: int) -> dict[str, Any]:
    device = next(row for row in snapshot["devices"] if row["physical_index"] == physical)
    guard = snapshot.get("guard", {})
    return {"captured_utc": snapshot.get("captured_utc"),
            "device": {k: device[k] for k in ("physical_index", "name", "memory_free_mib", "memory_used_mib", "memory_total_mib", "utilization_percent", "temperature_c")},
            "guard": {k: guard.get(k) for k in ("status", "phase", "physical_gpu", "required_free_mib")}}


def baseline_pids(snapshot: dict[str, Any], physical: int) -> set[int]:
    device = next(row for row in snapshot["devices"] if row["physical_index"] == physical)
    return {int(row["pid"]) for row in snapshot["compute_processes"] if row["gpu_uuid"] == device["uuid"]}


def record_empty_outputs() -> None:
    for filename, fields in (("baseline.csv", BASELINE_FIELDS), ("interventions.csv", INTERVENTION_FIELDS),
                             ("condition_summary.csv", CONDITION_FIELDS), ("expert_diversity.csv", EXPERT_FIELDS),
                             ("router_summary.csv", ROUTER_FIELDS)):
        if not (DATA / filename).exists():
            write_rows(DATA / filename, [], fields)


def run_parent() -> None:
    verify_base()
    manifest = read_json(DATA / "manifest.json", {})
    if manifest.get("status") != "preflight_passed" or manifest.get("preflight_source_sha") != source_sha():
        raise RuntimeError("a current-commit CPU preflight must pass before the guarded GPU run")
    if manifest.get("preflight_source_file_sha256s") != file_hashes():
        raise RuntimeError("source files changed after preflight")
    snapshot = v13.smi_snapshot()
    attempts = []
    choices = sorted(snapshot["devices"], key=lambda row: (row["utilization_percent"], -row["memory_free_mib"]))
    selected_gpu, selected_baseline, launch_guard = None, None, None
    for item in choices:
        physical = int(item["physical_index"])
        base = baseline_pids(snapshot, physical)
        try:
            guarded = v13.guard_gpu(physical, base, set(), phase="before_launch", required_free_mib=8192)
            attempts.append(compact_gpu(guarded, physical))
            selected_gpu, selected_baseline, launch_guard = physical, base, guarded
            break
        except Exception as exc:
            attempts.append({**compact_gpu(snapshot, physical), "guard": {"status": "blocked", "reason": safe_guard_reason(exc)}})
    manifest.setdefault("gpu", {}).update({"attempts": attempts, "physical_gpu_candidates": [x["device"]["physical_index"] for x in attempts]})
    if selected_gpu is None:
        manifest["status"] = "blocked_resource"
        manifest["gpu"].update({"status": "blocked_resource", "reason": "No GPU passed the repository ownership/resource guard before launch."})
        manifest["gpu_guard_blocked_utc"] = now()
        finalize_outputs(manifest, [], [], [], [])
        return
    manifest["gpu"].update({"status": "guard_passed", "selected_physical_gpu": selected_gpu,
                            "launch": compact_gpu(launch_guard, selected_gpu)})
    json_write(DATA / "manifest.json", manifest)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    env["MRCGE_ROUTE_SHUFFLE_PHYSICAL_GPU"] = str(selected_gpu)
    env["MRCGE_ROUTE_SHUFFLE_BASELINE_PIDS"] = ",".join(map(str, sorted(selected_baseline)))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    log_path = OUTPUT / "diagnostic.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        manifest["gpu"]["worker_pid"] = process.pid
        json_write(DATA / "manifest.json", manifest)
        guard_error = None
        while process.poll() is None:
            time.sleep(3)
            try:
                v13.guard_gpu(selected_gpu, selected_baseline, {process.pid}, phase="during_run", required_free_mib=2048)
            except Exception as exc:
                guard_error = safe_guard_reason(exc)
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
                break
    if guard_error:
        manifest = read_json(DATA / "manifest.json", manifest)
        manifest["status"] = "blocked_resource"
        manifest["gpu"]["runtime_guard_failure"] = guard_error
        manifest["gpu"]["runtime_guard_failed_utc"] = now()
        json_write(DATA / "manifest.json", manifest)
        return
    if process.returncode != 0:
        manifest = read_json(DATA / "manifest.json", manifest)
        manifest["status"] = "failed"
        manifest.setdefault("failed_units", []).append({"unit": "worker", "reason": f"exit_{process.returncode}"})
        manifest["worker_log"] = str(log_path)
        json_write(DATA / "manifest.json", manifest)


def safe_guard_reason(exc: Exception) -> str:
    message = str(exc)
    if "foreign compute PIDs" in message:
        return "ownership_guard_detected_new_external_compute_process"
    if "below" in message and "MiB" in message:
        return "free_memory_below_guard_threshold"
    if "utilization" in message:
        return "baseline_utilization_at_or_above_90_percent"
    if "temperature" in message:
        return "temperature_at_or_above_guard_threshold"
    return "resource_guard_rejected_launch"


def worker() -> None:
    physical = int(os.environ.get("MRCGE_ROUTE_SHUFFLE_PHYSICAL_GPU", "-1"))
    if physical < 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("diagnostic worker was not isolated to a guard-approved physical GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA visibility/isolation failed")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    baseline = {int(pid) for pid in os.environ.get("MRCGE_ROUTE_SHUFFLE_BASELINE_PIDS", "").split(",") if pid}
    v13.guard_gpu(physical, baseline, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    contexts, baseline_rows_all, intervention_rows_all, expert_rows_all, router_rows_all = {}, [], [], [], []
    manifest = read_json(DATA / "manifest.json", {})
    manifest["execution_started_utc"] = now()
    manifest["gpu"]["worker_device_name"] = torch.cuda.get_device_name(device)
    manifest["datasets"] = {}
    json_write(DATA / "manifest.json", manifest)
    for dataset in DATASETS:
        unit = f"{dataset}/seed42/baseline"
        context = None
        try:
            context = load_context(dataset, device)
            contexts[dataset] = context
            record, expert_rows, router_rows = baseline_rows(context, device)
            baseline_rows_all.append(record)
            expert_rows_all.extend(expert_rows)
            router_rows_all.extend(router_rows)
            manifest = read_json(DATA / "manifest.json", manifest)
            manifest["datasets"][dataset] = {
                "status": "baseline_passed", "baseline_reproduced": True,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "hook_control_equal": record["hook_control_equal"],
                "hook_call_order_pass": record["hook_call_order_pass"],
                "support_masks_pass": record["support_masks_pass"],
                "baseline_metrics": {k: record[k] for k in ("val_acc", "val_macro_f1", "val_ce")},
            }
            manifest["successful_units"].append(unit)
            if unit in manifest.get("unstarted_units", []):
                manifest["unstarted_units"].remove(unit)
            json_write(DATA / "manifest.json", manifest)
            append_progress(dataset, baseline_rows_all, expert_rows_all, router_rows_all, intervention_rows_all)
        except Exception as exc:
            manifest = read_json(DATA / "manifest.json", manifest)
            manifest["failed_units"].append({"unit": unit, "reason": str(exc), "traceback": traceback.format_exc()})
            manifest["datasets"][dataset] = {"status": "baseline_failed", "baseline_reproduced": False,
                                              "error": str(exc)}
            if unit in manifest.get("unstarted_units", []):
                manifest["unstarted_units"].remove(unit)
            json_write(DATA / "manifest.json", manifest)
            finalize_outputs(manifest, baseline_rows_all, intervention_rows_all, expert_rows_all, router_rows_all)
            continue

        for condition, _targets in CONDITIONS.items():
            for perm_seed in PERMUTATION_SEEDS:
                unit = f"{dataset}/seed42/{condition}/perm{perm_seed}"
                try:
                    record, _ = run_intervention(context, device, condition, perm_seed, baseline_rows_all[-1])
                    intervention_rows_all.append(record)
                    manifest = read_json(DATA / "manifest.json", manifest)
                    manifest["successful_units"].append(unit)
                    if unit in manifest.get("unstarted_units", []):
                        manifest["unstarted_units"].remove(unit)
                    json_write(DATA / "manifest.json", manifest)
                    append_progress(dataset, baseline_rows_all, expert_rows_all, router_rows_all, intervention_rows_all)
                except Exception as exc:
                    manifest = read_json(DATA / "manifest.json", manifest)
                    manifest["failed_units"].append({"unit": unit, "reason": str(exc), "traceback": traceback.format_exc()})
                    if unit in manifest.get("unstarted_units", []):
                        manifest["unstarted_units"].remove(unit)
                    json_write(DATA / "manifest.json", manifest)
                    finalize_outputs(manifest, baseline_rows_all, intervention_rows_all, expert_rows_all, router_rows_all)
                    return
        model_hash_after = state_hash(context["model"])
        classifier_hash_after = state_hash(context["classifier"])
        checkpoint_hash_after = sha256_file(context["record"]["checkpoint_path"])
        if (model_hash_after != context["loaded_model_hash"] or classifier_hash_after != context["loaded_classifier_hash"]
                or checkpoint_hash_after != context["record"]["checkpoint_sha256"]):
            raise RuntimeError(f"{dataset}: final read-only model/checkpoint hash audit failed")
        if any(parameter.grad is not None for parameter in (*context["model"].parameters(), *context["classifier"].parameters())):
            raise RuntimeError(f"{dataset}: a parameter gradient exists despite no_grad inference")
        manifest = read_json(DATA / "manifest.json", manifest)
        manifest["datasets"][dataset].update({"model_weights_unchanged": True,
                                              "classifier_weights_unchanged": True,
                                              "checkpoint_unchanged": True,
                                              "parameter_gradients_absent": True})
        json_write(DATA / "manifest.json", manifest)
        del contexts[dataset]
        del context
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest = read_json(DATA / "manifest.json", manifest)
    finalize_outputs(manifest, baseline_rows_all, intervention_rows_all, expert_rows_all, router_rows_all)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        result = preflight()
        print(json.dumps({"status": result["status"], "checkpoints": list(result["checkpoints"])}, indent=2))
    elif args.run:
        run_parent()
    elif args.worker:
        worker()


if __name__ == "__main__":
    main()
