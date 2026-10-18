#!/usr/bin/env python3
"""Auditable Hybrid-CGE optimization, generalization and baseline campaign.

Formal training is orchestrated as one isolated process per cell. The parent
process runs the existing physical-GPU ownership guard before every child.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
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
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score

from scripts import run_rmcge_v22 as gpu_guard
from scripts.preflight_hybrid_cge_v11 import load_model as load_hybrid_model, mask_test_labels
from src.data import load_mag_data
from src.data.graph_utils import preprocess_edge_index
from src.data.loaders import resolve_path
from src.data.semantic_candidates_v7b import (
    build_semantic_candidate_cache_v7b,
    load_semantic_candidate_cache_v7b,
)
from src.models import build_model
from src.tasks.nc import _resolve_nc_eval_labels, run_nc
from src.utils.seeds import set_seed
from scripts import run_hybrid_cge_validation_expansion_v1 as prior_runner


STUDY = ROOT / "research/hybrid_cge_paper_evidence_v1"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/hybrid_cge_paper_evidence_v1"
RUNS = OUTPUT / "runs"
SEMANTIC_CACHE = ROOT / "outputs/hybrid_cge_validation_expansion_v1/semantic_cache"
CONFIG_PATH = ROOT / "configs/study/hybrid_cge_paper_evidence_v1.yaml"
DATASETS = ("Movies", "Grocery", "ele-fashion", "Toys", "Reddit-S")
OPT_DATASETS = ("Movies", "Grocery", "ele-fashion")
GEN_DATASETS = ("Toys", "Reddit-S")
SEEDS = (42, 43, 44)
CONDITIONS = ("A1_no_early_physical_cross", "A2_no_late_completion", "A3_no_moe", "A4_no_same_node")
HOP_CONDITION = "H2_physical_3hop"
FASHION_CONDITIONS = ("F1_fashion_dropout_aux", "F2_fashion_class_balanced_4hop", "F4_fashion_class_balanced_3hop")
CONFIG_CONDITIONS = ("H2_full", *CONDITIONS, HOP_CONDITION, *FASHION_CONDITIONS)
# A3 deliberately removes these complete shared-MoE/router trees.  Paired
# ablation initialization therefore audits the exact intersection of the H2
# public parameter state, while retaining the full-model hash separately.
NO_MOE_PARAMETER_PREFIXES = frozenset({
    "experts", "context_encoder", "router_projectors", "router_norms",
    "modality_embedding", "selection_head", "strength_head", "node_router",
    "node_modality_embedding", "view_router", "view_type_embedding",
    "comp_router", "comp_modality_embedding", "physical_cross_coefficients",
    "interaction",
})
BASELINES = ("gcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
REUSED_BASELINES = ("sage", "mmgcn", "rolemag")
ALL_BASELINES = (*BASELINES, *REUSED_BASELINES)
PAPER_MODEL = "hybrid_cge_paper_v1"
PROTOCOL = "unified_full_graph_nc_v1"
MODEL_CONFIGS = {
    PAPER_MODEL: PAPER_MODEL,
    "hybrid_cge_v11": "hybrid_cge_v11",
    "mlp": "mlp",  # utility path for split audit only; not part of the baseline matrix
    "gcn": "gcn",
    "sage": "sage",
    "mmgcn": "mmgcn",
    "mgat": "mgat",
    "dip": "dip",
    "dgf": "dgf",
    "dmgc": "dmgc",
    "lgmrec": "lgmrec",
    "rolemag": "rolemag",
}
RUN_FIELDS = (
    "dataset", "model", "condition", "seed", "attempt", "status", "best_accuracy_epoch",
    "best_f1_epoch", "epochs_completed", "val_acc", "val_macro_f1", "val_ce",
    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc",
    "train_macro_f1", "train_ce", "train_ce_eval", "train_val_acc_gap",
    "train_val_macro_f1_gap", "train_val_ce_gap", "model_parameters", "classifier_parameters",
    "optimizer", "lr", "weight_decay", "dropout", "private_auxiliary_ce",
    "class_balanced_main_ce", "main_ce_class_weights", "paper_condition",
    "reused_from_branch", "reused_source_git_sha",
    "model_dropout_modules", "parameter_count", "wall_seconds", "peak_allocated_bytes",
    "peak_reserved_bytes", "gpu_name", "config_sha256", "source_git_sha", "feature_fingerprint",
    "raw_feature_tensor_fingerprint", "physical_edge_fingerprint", "split_fingerprint", "split_file_sha256", "cache_path",
    "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint", "initial_model_sha256",
    "initial_classifier_sha256", "validation_rng_unchanged", "test_evaluation",
    "config_snapshot_path", "model_source_path", "model_source_sha256", "task_source_sha256",
    "best_accuracy_checkpoint", "best_accuracy_checkpoint_sha256", "best_f1_checkpoint",
    "best_f1_checkpoint_sha256", "run_metrics_path", "run_dir", "error",
    "initial_shared_model_sha256",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default), encoding="utf-8")
    tmp.replace(path)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def append_csv(path: Path, row: dict[str, Any], key: tuple[str, ...]) -> None:
    rows = csv_rows(path)
    row_key = tuple(str(row.get(field, "")) for field in key)
    if any(tuple(old.get(field, "") for field in key) == row_key for old in rows):
        return
    fields = list(dict.fromkeys([*RUN_FIELDS, *(k for old in rows for k in old), *row.keys()]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)
    tmp.replace(path)


def append_failure(record: dict[str, Any]) -> None:
    path = DATA / "failures.json"
    payload = json.loads(path.read_text()) if path.is_file() else {"failures": []}
    payload["failures"].append({"time_utc": now(), "training_started": False,
                                "test_evaluation": False, **record})
    json_write(path, payload)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def module_hash(module: torch.nn.Module, *, excluded_prefixes: frozenset[str] = frozenset()) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name.split(".", maxsplit=1)[0] in excluded_prefixes:
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def study_config() -> DictConfig:
    return OmegaConf.load(CONFIG_PATH)


def build_cfg(dataset: str, model_name: str, seed: int, device: str,
              condition: str | None = None, cache: dict[str, Any] | None = None) -> DictConfig:
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    task = OmegaConf.load(ROOT / "configs/task/nc.yaml")
    model = OmegaConf.load(ROOT / f"configs/model/{MODEL_CONFIGS[model_name]}.yaml")
    cfg = OmegaConf.create({
        "paths": root.paths, "dataset": ds, "task": task, "model": model,
        "logging": root.logging, "seed": int(seed), "num_runs": 1, "device": str(device),
    })
    cfg.dataset.name = dataset
    cfg.model.name = model_name
    cfg.task.name = "nc"
    cfg.task.protocol_version = PROTOCOL
    cfg.task.training_mode = "full_graph"
    cfg.task.optimizer = "adamw"
    cfg.task.lr = 1e-3
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 1e-4
    cfg.task.grad_clip = 1.0
    cfg.task.eval_every = 1
    cfg.task.inference_mode = "full"
    cfg.task.inference_batch_size = 4096
    cfg.task.scheduler = None
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 1.0
    cfg.task.loss.unimodal_weight = 0.0
    cfg.task.record_extended_validation = True
    cfg.task.record_train_ce_separately = True
    if model_name == PAPER_MODEL:
        if condition not in CONFIG_CONDITIONS:
            raise ValueError(f"formal paper-evidence Hybrid-CGE training requires a candidate condition, got {condition}")
        study = study_config()
        cond = study.conditions.get(condition)
        fashion_cond = study.fashion_candidates.get(condition)
        if cond is None and fashion_cond is None:
            raise ValueError(f"condition {condition} has no formal configuration")
        if cond is None:
            base_cond = study.conditions["H2_full"]
            physical_hops = int(fashion_cond.get("physical_hops", base_cond.physical_hops))
            dropout = float(fashion_cond.get("dropout", base_cond.dropout))
            aux_ce = float(fashion_cond.get("private_auxiliary_ce_weight", base_cond.auxiliary_ce_weight))
        else:
            physical_hops = int(cond.get("physical_hops", 4))
            dropout = float(cond.get("dropout", 0.2))
            aux_ce = float(cond.get("auxiliary_ce_weight", 0.3))
        cfg.model.paper_condition = condition
        cfg.model.condition = "H2_hybrid_plus"
        cfg.model.physical_hops = physical_hops
        cfg.model.dropout = dropout
        cfg.model.late_cross_dropout = 0.15
        cfg.task.weight_decay = 1e-4
        cfg.task.loss.unimodal_weight = aux_ce
        cfg.task.loss.class_balanced_main_ce = bool(condition in {
            "F2_fashion_class_balanced_4hop", "F4_fashion_class_balanced_3hop"
        })
        if cache is None:
            raise ValueError(f"Hybrid-CGE requires audited semantic cache for {dataset}")
        cfg.model.semantic_cache_path = str(cache["cache_path_abs"])
        cfg.model.semantic_feature_fingerprint = str(cache["feature_fingerprint"])
        cfg.model.semantic_candidate_fingerprint = str(cache["candidate_fingerprint"])
        cfg.model.semantic_v7b_fingerprint = str(cache["v7b_fingerprint"])
    else:
        cfg.task.weight_decay = 1e-4
        cfg.task.loss.class_balanced_main_ce = False
    cfg.paths.output_root = str(OUTPUT / "hydra")
    cfg.task.extended_checkpoint_metadata = {
        "study": "Hybrid-CGE paper evidence V1",
        "dataset": dataset, "model": model_name, "condition": condition,
        "seed": int(seed), "test_evaluation": False,
        "test_label_values_indexed": False,
    }
    OmegaConf.resolve(cfg)
    return cfg


def info_for(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def current_cache(dataset: str) -> dict[str, Any]:
    if dataset in OPT_DATASETS:
        cache_entries = gpu_guard.verify_cache_entries()
        entry = cache_entries[dataset]
        return {
            "cache_path_abs": entry["cache_path_abs"],
            "feature_fingerprint": entry["feature_fingerprint"],
            "candidate_fingerprint": entry["candidate_fingerprint"],
            "v7b_fingerprint": entry["v7b_fingerprint"],
        }
    candidates = sorted(SEMANTIC_CACHE.glob(f"{dataset}_*_v7b.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one expansion V7B cache for {dataset}, found {len(candidates)}")
    payload = load_semantic_candidate_cache_v7b(candidates[0], expected_dataset=dataset)
    metadata = payload["metadata"]
    return {
        "cache_path_abs": str(candidates[0].resolve()),
        "feature_fingerprint": metadata["feature_fingerprint"],
        "candidate_fingerprint": metadata["candidate_fingerprint"],
        "v7b_fingerprint": metadata["v7b_fingerprint"],
    }


def dataset_paths(dataset: str) -> dict[str, Path]:
    ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    paths = {"dataset_config": ROOT / f"configs/dataset/{dataset}.yaml"}
    if dataset == "ele-fashion":
        temp = OmegaConf.create({"paths": root_cfg.paths, "dataset": ds})
        OmegaConf.resolve(temp)
        resolved_ds = temp.dataset
        paths.update({key: resolve_path(resolved_ds[key]) for key in ("joint_feat_path", "edge_path", "label_path", "node_split_path")})
    else:
        temp = OmegaConf.create({"paths": root_cfg.paths, "dataset": ds, "seed": 42})
        OmegaConf.resolve(temp)
        resolved_ds = temp.dataset
        paths.update({"graph_path": resolve_path(resolved_ds.graph_path), "text_feat_path": resolve_path(resolved_ds.text_feat_path),
                      "image_feat_path": resolve_path(resolved_ds.image_feat_path),
                      "split_template": resolve_path(str(temp.dataset.nc_split_path).replace("42", "{seed}"))})
    return paths


def source_data_audit(datasets: tuple[str, ...] = DATASETS, *, build_new_cache: bool = False) -> dict[str, Any]:
    from src.data.semantic_candidates import sha256_file as feature_file_sha

    DATA.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for dataset in datasets:
        cache = None
        if dataset in OPT_DATASETS:
            cache = current_cache(dataset)
        elif build_new_cache:
            cache = build_new_dataset_cache(dataset)
        elif list(SEMANTIC_CACHE.glob(f"{dataset}_*_v7b.pt")):
            cache = current_cache(dataset)

        cfg = build_cfg(dataset, PAPER_MODEL, 42, "cpu", condition="H2_full", cache=cache) if cache else None
        if cfg is None:
            # Data loading is independent of model semantics; use the dataset's canonical config.
            root = OmegaConf.load(ROOT / "configs/config.yaml")
            ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
            task = OmegaConf.load(ROOT / "configs/task/nc.yaml")
            cfg = OmegaConf.create({"paths": root.paths, "dataset": ds, "task": task, "model": OmegaConf.load(ROOT / "configs/model/mlp.yaml"), "seed": 42, "num_runs": 1, "device": "cpu"})
            cfg.dataset.name = dataset
            OmegaConf.resolve(cfg)
        raw = load_mag_data(cfg, "nc", 42)
        paths = dataset_paths(dataset)
        source_files = {}
        for name, path in paths.items():
            if name == "split_template":
                continue
            source_files[name] = {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None,
                                  "sha256": feature_file_sha(path) if path.is_file() else None}
        split_info = {}
        split_template = paths.get("split_template")
        if split_template:
            for seed in SEEDS:
                split_path = Path(str(split_template).format(seed=seed))
                # Existing split path is persisted by the project loader.
                split_info[str(seed)] = {"path": str(split_path), "exists": split_path.is_file(),
                                         "sha256": feature_file_sha(split_path) if split_path.is_file() else None,
                                         "source": "reused_project_split" if split_path.is_file() else "generated_by_src.data.splits"}
        else:
            p = paths["node_split_path"]
            split_info = {str(seed): {"path": str(p), "exists": p.is_file(), "sha256": feature_file_sha(p) if p.is_file() else None,
                                      "source": "fixed_ele-fashion_split.pt"} for seed in SEEDS}
        x_t, x_i = raw.x_t, raw.x_i
        if x_t is None or x_i is None or x_t.shape[0] != raw.num_nodes or x_i.shape[0] != raw.num_nodes:
            raise RuntimeError(f"{dataset} modality tensors failed node-index alignment")
        if not torch.isfinite(x_t).all() or not torch.isfinite(x_i).all():
            raise RuntimeError(f"{dataset} has non-finite configured modality features")
        if any(bool(((raw.y[index] < 0) | (raw.y[index] >= int(raw.num_classes))).any()) for index in (raw.train_idx, raw.val_idx)):
            raise RuntimeError(f"{dataset} Train/Validation has a missing class label")
        # Never serialize, summarize, or inspect Test label values in this audit.
        split_fingerprints = {}
        if dataset != "ele-fashion":
            for seed in SEEDS:
                seed_cfg = build_cfg(dataset, "mlp", seed, "cpu")
                split_data = load_mag_data(seed_cfg, "nc", seed)
                split_fingerprints[str(seed)] = {
                    "train": tensor_hash(split_data.train_idx),
                    "validation": tensor_hash(split_data.val_idx),
                    "test_indices_only": tensor_hash(split_data.test_idx),
                }
                del split_data
        else:
            split_fingerprints = {str(seed): {"fixed_split_file_sha256": split_info[str(seed)]["sha256"]} for seed in SEEDS}
        entries[dataset] = {
            "num_nodes": int(raw.num_nodes), "num_edges_directed_after_preprocessing": int(raw.edge_index.size(1)),
            "num_classes_configured": int(raw.num_classes), "feature_dims": {"text": int(x_t.shape[1]), "visual": int(x_i.shape[1])},
            "nonzero_feature_rows": {"text": int((x_t.abs().sum(1) > 0).sum()), "visual": int((x_i.abs().sum(1) > 0).sum())},
            "feature_fingerprint": tensor_hash(raw.x), "physical_edge_fingerprint": tensor_hash(raw.edge_index),
            "train_count": int(raw.train_idx.numel()), "validation_count": int(raw.val_idx.numel()),
            "masked_test_count": int(raw.test_idx.numel()), "split_fingerprints": split_fingerprints,
            "source_files": source_files, "split_files": split_info,
            "text_visual_alignment": True, "train_val_labels_valid": True,
            "test_label_values_serialized": False,
            "semantic_cache": ({"path": cache["cache_path_abs"], "sha256": feature_file_sha(Path(cache["cache_path_abs"])),
                                "feature_fingerprint": cache["feature_fingerprint"],
                                "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"]}
                               if cache else None),
        }
        del raw
    result = {"created_utc": now(), "protocol_version": PROTOCOL, "datasets": entries,
              "configured_feature_sources": "RoBERTa text + OpenAI CLIP image for MAGB; fixed CLIP joint tensor split into text/visual for ele-fashion",
              "test_evaluation": False, "test_labels_inspected": False}
    json_write(DATA / "data_audit.json", result)
    return result


def build_new_dataset_cache(dataset: str) -> dict[str, Any]:
    if dataset not in GEN_DATASETS:
        raise ValueError("only Toys and Reddit-S require newly built cache")
    ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create({"paths": root.paths, "dataset": ds, "seed": 42})
    OmegaConf.resolve(cfg)
    ds = cfg.dataset
    text_path, visual_path, graph_path = map(resolve_path, (ds.text_feat_path, ds.image_feat_path, ds.graph_path))
    import dgl
    graphs, _ = dgl.load_graphs(str(graph_path))
    if len(graphs) != 1:
        raise RuntimeError(f"{dataset}: expected exactly one DGL graph")
    graph = graphs[0]
    src, dst = graph.edges()
    physical = preprocess_edge_index(torch.stack((src.long(), dst.long()), dim=0), int(graph.num_nodes()),
                                     make_undirected=bool(ds.get("make_undirected", True)), with_self_loops=False)
    text = np.load(text_path, mmap_mode="r", allow_pickle=False)
    visual = np.load(visual_path, mmap_mode="r", allow_pickle=False)
    if text.shape[0] != graph.num_nodes() or visual.shape[0] != graph.num_nodes():
        raise RuntimeError(f"{dataset}: specified RoBERTa/CLIP feature node count does not match graph")
    if (text.shape[1], visual.shape[1]) == (0, 0):
        raise RuntimeError("empty configured modality")
    path, metadata = build_semantic_candidate_cache_v7b(
        dataset=dataset, text_features=text, visual_features=visual,
        physical_edge_index=physical, cache_dir=SEMANTIC_CACHE,
        text_source=text_path, visual_source=visual_path, top_k=8, seed=2026, thread_count=8,
    )
    payload = load_semantic_candidate_cache_v7b(
        path, expected_dataset=dataset, expected_feature_fingerprint=metadata["feature_fingerprint"],
        expected_candidate_fingerprint=metadata["candidate_fingerprint"], expected_v7b_fingerprint=metadata["v7b_fingerprint"],
    )
    if int(payload["metadata"]["num_nodes"]) != int(graph.num_nodes()) or payload["metadata"].get("candidate_phase_label_access") is not False:
        raise RuntimeError(f"{dataset}: semantic cache provenance/node-count audit failed")
    return {"cache_path_abs": str(path.resolve()), "feature_fingerprint": metadata["feature_fingerprint"],
            "candidate_fingerprint": metadata["candidate_fingerprint"], "v7b_fingerprint": metadata["v7b_fingerprint"]}


def model_dropout_audit(model: torch.nn.Module, model_name: str, expected: float) -> dict[str, Any]:
    layers = {name: float(module.p) for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout)}
    if model_name in {"hybrid_cge_v11", PAPER_MODEL}:
        if not layers:
            raise RuntimeError("Hybrid-CGE exposes no nn.Dropout modules for the configured main dropout")
        wrong = {name: p for name, p in layers.items() if not math.isclose(p, expected, abs_tol=1e-12)}
        if wrong:
            raise RuntimeError(f"Hybrid-CGE configured dropout not applied in all nn.Dropout paths: {wrong}")
        if not math.isclose(float(model.cross_dropout_p), .15, abs_tol=1e-12):
            raise RuntimeError("fixed late Cross Dropout unexpectedly changed")
    return {"nn_dropout_modules": layers,
            "custom_late_cross_dropout": float(getattr(model, "cross_dropout_p", 0.0)) if model_name in {"hybrid_cge_v11", PAPER_MODEL} else None}


def get_model(cfg: DictConfig, data, cache: dict[str, Any] | None, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg, info_for(data))
    if bool(getattr(model, "requires_global_semantic_candidates", False)):
        if cache is None:
            raise RuntimeError("semantic cache required by model but absent")
        model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset=cfg.dataset.name,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
    return model.to(device)


def init_pair_hashes(cfg, data, cache, seed: int) -> tuple[str, str, dict[str, Any]]:
    set_seed(seed)
    model = get_model(cfg, data, cache, torch.device("cpu"))
    head = torch.nn.Linear(model.out_dim, int(data.num_classes))
    audit = model_dropout_audit(model, str(cfg.model.name), float(cfg.model.get("dropout", 0.0)))
    audit["unregistered_tensor_state"] = {
        name: tensor_hash(value)
        for name in ("id_embedding",)
        if isinstance((value := getattr(model, name, None)), torch.Tensor) and name not in model.state_dict()
    }
    audit["initial_shared_model_sha256"] = module_hash(
        model,
        excluded_prefixes=(NO_MOE_PARAMETER_PREFIXES if str(cfg.model.name) == PAPER_MODEL else frozenset()),
    )
    hashes = (module_hash(model), module_hash(head), audit)
    del model, head
    return hashes


def prep_model_and_cache(dataset: str, model_name: str) -> dict[str, Any] | None:
    return current_cache(dataset) if model_name in {"hybrid_cge_v11", PAPER_MODEL} else None


def metric_bundle(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor,
                  classes: int, eval_labels: list[int]) -> dict[str, float]:
    index = idx.detach().cpu().long()
    targets = labels[index].detach().cpu().long()
    if bool((targets < 0).any()):
        raise RuntimeError("masked Test sentinel reached Train/Validation evaluation")
    selected = logits[index].detach().cpu().float()
    return {
        "acc": float((selected.argmax(-1) == targets).float().mean()),
        "macro_f1": float(f1_score(targets.numpy(), selected.argmax(-1).numpy(), labels=eval_labels,
                                    average="macro", zero_division=0)),
        "ce": float(torch.nn.functional.cross_entropy(selected, targets)),
    }


def evaluate_checkpoint(cfg: DictConfig, data, cache: dict[str, Any] | None, path: Path,
                        device: torch.device, expected_sha: str, name: str,
                        expected_unregistered_state: dict[str, str] | None = None) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"checkpoint hash changed before audit: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    set_seed(int(cfg.seed))
    model = get_model(cfg, data, cache, device)
    if expected_unregistered_state:
        for state_name, expected_hash in expected_unregistered_state.items():
            state_value = getattr(model, state_name, None)
            if not isinstance(state_value, torch.Tensor) or tensor_hash(state_value) != expected_hash:
                raise RuntimeError(f"could not reconstruct unregistered model tensor {state_name} from recorded seed/config")
    head = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval(); head.eval()
    x = data.x.to(device)
    graph = data.edge_index.to(device) if str(cfg.model.name) != "mlp" else None
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()
    with torch.no_grad():
        z, _, _, _, _ = model(x, graph)
        logits = head(z).detach().cpu()
    rng_ok = torch.equal(cpu_before, torch.random.get_rng_state()) and torch.equal(cuda_before, torch.cuda.get_rng_state(device))
    if not rng_ok:
        raise RuntimeError(f"checkpoint validation evaluation changed RNG state: {name}")
    labels_for_f1 = _resolve_nc_eval_labels(data, include_test=False)
    train = metric_bundle(logits, data.y, data.train_idx, int(data.num_classes), labels_for_f1)
    val = metric_bundle(logits, data.y, data.val_idx, int(data.num_classes), labels_for_f1)
    saved = payload.get("metrics", {})
    for key, value in (("val_acc", val["acc"]), ("val_macro_f1", val["macro_f1"]), ("val_ce", val["ce"])):
        if key in saved and abs(float(saved[key]) - value) > 2e-6:
            raise RuntimeError(f"reloaded {name} checkpoint {key} differs: saved={saved[key]} recalculated={value}")
    if not all(math.isfinite(v) for metric in (train, val) for v in metric.values()):
        raise RuntimeError(f"non-finite checkpoint metrics in {name}")
    del z, logits, model, head
    torch.cuda.empty_cache()
    return {"train": train, "val": val, "validation_rng_unchanged": rng_ok, "payload": payload}


def worker_preflight(dataset: str, model_name: str, condition: str | None, seed: int,
                     physical_gpu: int) -> dict[str, Any]:
    if os.environ.get("HYBRID_EXP_GUARDED_CELL") != "1":
        raise RuntimeError("preflight worker must be launched by the GPU-guarded parent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("worker CUDA visibility does not isolate selected physical GPU")
    gpu_guard._assert_physical_gpu_idle(physical_gpu)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected one visible CUDA device")
    cache = prep_model_and_cache(dataset, model_name)
    cfg = build_cfg(dataset, model_name, seed, str(device), condition, cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = mask_test_labels(raw)
    if bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError("Test labels were not masked before preflight")
    if safe.x.size(0) != safe.num_nodes or safe.y.size(0) != safe.num_nodes or safe.edge_index.numel() and int(safe.edge_index.max()) >= safe.num_nodes:
        raise RuntimeError("node/index alignment check failed")
    set_seed(seed)
    model = get_model(cfg, safe, cache, device)
    head = torch.nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    dropout = model_dropout_audit(model, model_name, float(cfg.model.get("dropout", 0.0)))
    x, edges, labels = safe.x.to(device), safe.edge_index.to(device), safe.y.to(device)
    model.train(); head.train()
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    z, _, _, aux_loss, forward_info = model(x, edges if model_name != "mlp" else None)
    from src.tasks.nc import _training_class_weights
    train_indices = safe.train_idx.to(device)
    class_weights = _training_class_weights(
        labels, train_indices, int(safe.num_classes),
        enabled=bool(cfg.task.loss.get("class_balanced_main_ce", False)),
    )
    loss = torch.nn.functional.cross_entropy(
        head(z[train_indices]), labels[train_indices], weight=class_weights
    )
    if float(cfg.task.loss.unimodal_weight) > 0:
        # Current Hybrid-CGE private auxiliary CE path is checked exactly as in the task trainer.
        from src.tasks.nc import _compose_full_graph_loss
        loss, _ = _compose_full_graph_loss(model=model, info=forward_info, main_ce=loss, aux_loss=aux_loss,
                                           aux_weight=float(cfg.task.loss.aux_weight),
                                           unimodal_weight=float(cfg.task.loss.unimodal_weight),
                                           labels=labels, train_idx=safe.train_idx.to(device))
    else:
        loss = loss + float(cfg.task.loss.aux_weight) * aux_loss
    loss.backward()
    finite_grad = all(p.grad is None or torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(head.parameters()))
    if not torch.isfinite(z).all() or not torch.isfinite(loss) or not finite_grad:
        raise RuntimeError("full-graph preflight produced non-finite embeddings/loss/gradients")
    before_model, before_head = module_hash(model), module_hash(head)
    checkpoint_path = OUTPUT / "preflight" / f"{dataset}_{model_name}_{condition or 'baseline'}_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "head": head.state_dict(), "num_nodes": safe.num_nodes}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    if module_hash(model) != before_model or module_hash(head) != before_head:
        raise RuntimeError("preflight checkpoint round-trip failed")
    record = {
        "dataset": dataset, "model": model_name, "condition": condition, "seed": seed,
        "status": "pass", "num_nodes": safe.num_nodes, "num_edges": int(safe.edge_index.size(1)),
        "node_index_alignment": True, "test_labels_masked": True, "test_evaluation": False,
        "forward_finite": True, "loss_finite": True, "gradients_finite": True,
        "checkpoint_roundtrip": True, "dropout_audit": dropout,
        "loss": float(loss.detach().cpu()), "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu_name": torch.cuda.get_device_name(device), "created_utc": now(),
    }
    path = DATA / "preflight.json"
    full = json.loads(path.read_text()) if path.is_file() else {"runs": []}
    key = (dataset, model_name, condition or "", seed)
    full["runs"] = [r for r in full["runs"] if (r["dataset"], r["model"], r.get("condition") or "", int(r["seed"])) != key]
    full["runs"].append(record)
    json_write(path, full)
    checkpoint_path.unlink(missing_ok=True)
    del model, head, safe, raw
    torch.cuda.empty_cache()
    return record


def worker_train(dataset: str, model_name: str, condition: str | None, seed: int,
                 physical_gpu: int) -> None:
    if os.environ.get("HYBRID_EXP_GUARDED_CELL") != "1":
        raise RuntimeError("training worker must be launched by the GPU-guarded parent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("worker CUDA visibility does not isolate selected physical GPU")
    gpu_guard._assert_physical_gpu_idle(physical_gpu)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected one visible CUDA device")
    cache = prep_model_and_cache(dataset, model_name)
    cfg = build_cfg(dataset, model_name, seed, str(device), condition, cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = mask_test_labels(raw)
    if bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError("Test labels were not masked before Trainer")

    set_seed(seed)
    initial_model, initial_head, dropouts = init_pair_hashes(cfg, safe, cache, seed)
    run_dir = RUNS / dataset / (condition or model_name) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = sorted(run_dir.glob("attempt_*"))
    attempt = len(attempts) + 1
    attempt_dir = run_dir / f"attempt_{attempt:02d}"
    attempt_dir.mkdir(parents=True)
    accuracy_path = attempt_dir / "best_accuracy.pt"
    f1_path = attempt_dir / "best_f1.pt"
    metrics_path = attempt_dir / "run_metrics.json"
    cfg.task.save_ckpt_path = str(accuracy_path)
    cfg.task.best_f1_ckpt_path = str(f1_path) if bool(cfg.task.record_extended_validation) else None
    cfg.task.run_metrics_path = str(metrics_path)
    fingerprints = gpu_guard.dataset_fingerprints(safe)
    config_snapshot = OmegaConf.to_yaml(cfg, resolve=True)
    config_snapshot_path = attempt_dir / "config_snapshot.yaml"
    config_snapshot_path.write_text(config_snapshot, encoding="utf-8")
    model_source_path = ROOT / "src/models" / f"{model_name}.py"
    task_source_path = ROOT / "src/tasks/nc.py"
    cfg.task.extended_checkpoint_metadata = {
        "study": "Hybrid-CGE validation expansion V1", "dataset": dataset,
        "model": model_name, "condition": condition, "seed": seed,
        "config_sha256": hashlib.sha256(config_snapshot.encode()).hexdigest(),
        "config_snapshot_path": str(config_snapshot_path),
        "model_source_path": str(model_source_path.resolve()),
        "model_source_sha256": sha256_file(model_source_path),
        "task_source_sha256": sha256_file(task_source_path),
        "source_git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "feature_fingerprint": cache["feature_fingerprint"] if cache else fingerprints["features"],
        "raw_feature_tensor_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "test_evaluation": False, "test_label_values_indexed": False,
    }
    started = time.perf_counter()
    logger = logging.getLogger(f"hybrid_exp.{dataset}.{model_name}.{condition}.{seed}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(attempt_dir / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    try:
        run_nc(cfg, safe, device, logger)
        packed = json.loads(metrics_path.read_text())
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("training metrics provenance mismatch")
        entry = packed["runs"][0]
        metadata = entry["metadata"]
        history = metadata.get("epoch_history", [])
        if model_name == PAPER_MODEL and (not history or any(not math.isfinite(float(epoch[k])) for epoch in history for k in ("train_loss", "val_acc", "val_macro_f1"))):
            raise RuntimeError("paper-evidence Hybrid-CGE train/validation history missing or non-finite")
        for value in entry["metrics"].values():
            if not math.isfinite(float(value)):
                raise RuntimeError("non-finite output metric")
        if not accuracy_path.is_file():
            raise RuntimeError("Best-Accuracy checkpoint was not saved")
        acc_sha = sha256_file(accuracy_path)
        acc_audit = evaluate_checkpoint(cfg, safe, cache, accuracy_path, device, acc_sha, "best_accuracy", dropouts["unregistered_tensor_state"])
        f1_audit = None
        f1_sha = ""
        if f1_path.is_file():
            f1_sha = sha256_file(f1_path)
            f1_audit = evaluate_checkpoint(cfg, safe, cache, f1_path, device, f1_sha, "best_f1", dropouts["unregistered_tensor_state"])
        best_acc_payload = acc_audit["payload"]
        val = acc_audit["val"]; train = acc_audit["train"]
        best_f1_val = f1_audit["val"] if f1_audit else {}
        best_f1_train = f1_audit["train"] if f1_audit else {}
        logged_epochs = sum(1 for line in (attempt_dir / "train.log").read_text(encoding="utf-8").splitlines()
                            if "Epoch " in line and "Train Loss" in line)
        cache_sha = sha256_file(Path(cache["cache_path_abs"])) if cache else ""
        split_fp = gpu_guard.dataset_fingerprints(raw)["split_fingerprint"]
        paths = dataset_paths(dataset)
        split_path = paths.get("node_split_path") if dataset == "ele-fashion" else Path(str(paths["split_template"]).format(seed=seed))
        dropout_expected = float(cfg.model.get("dropout", 0.0))
        run_row = {
            "dataset": dataset, "model": model_name, "condition": condition or model_name, "seed": seed,
            "attempt": attempt, "status": "complete", "best_accuracy_epoch": int(best_acc_payload["epoch"]),
            "best_f1_epoch": int(f1_audit["payload"]["epoch"]) if f1_audit else "",
            "epochs_completed": int(metadata.get("epochs_completed", metadata.get("accuracy_stopping_horizon_epochs", logged_epochs))),
            "val_acc": val["acc"], "val_macro_f1": val["macro_f1"], "val_ce": val["ce"],
            "best_f1_val_acc": best_f1_val.get("acc", ""), "best_f1_val_macro_f1": best_f1_val.get("macro_f1", ""),
            "best_f1_val_ce": best_f1_val.get("ce", ""), "train_acc": train["acc"],
            "train_macro_f1": train["macro_f1"], "train_ce": best_acc_payload.get("metrics", {}).get("train_ce", train["ce"]),
            "train_ce_eval": train["ce"], "train_val_acc_gap": train["acc"] - val["acc"],
            "train_val_macro_f1_gap": train["macro_f1"] - val["macro_f1"], "train_val_ce_gap": train["ce"] - val["ce"],
            "best_f1_train_acc": best_f1_train.get("acc", ""), "best_f1_train_macro_f1": best_f1_train.get("macro_f1", ""),
            "best_f1_train_ce": best_f1_train.get("ce", ""),
            "model_parameters": metadata.get("model_parameters"), "classifier_parameters": metadata.get("classifier_parameters"),
            "optimizer": metadata.get("optimizer", "AdamW"), "lr": float(cfg.task.lr),
            "weight_decay": float(cfg.task.weight_decay), "dropout": dropout_expected,
            "private_auxiliary_ce": float(cfg.task.loss.unimodal_weight),
            "class_balanced_main_ce": bool(cfg.task.loss.get("class_balanced_main_ce", False)),
            "main_ce_class_weights": json.dumps(metadata.get("main_ce_class_weights"), separators=(",", ":")),
            "paper_condition": condition if model_name == PAPER_MODEL else "",
            "config_snapshot_path": str(config_snapshot_path),
            "model_source_path": str(model_source_path.resolve()),
            "model_source_sha256": cfg.task.extended_checkpoint_metadata["model_source_sha256"],
            "task_source_sha256": cfg.task.extended_checkpoint_metadata["task_source_sha256"],
            "model_dropout_modules": json.dumps(dropouts["nn_dropout_modules"], sort_keys=True),
            "unregistered_tensor_state": json.dumps(dropouts["unregistered_tensor_state"], sort_keys=True),
            "parameter_count": int(metadata.get("model_parameters", 0)) + int(metadata.get("classifier_parameters", 0)),
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes", int(torch.cuda.max_memory_allocated(device))),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes", int(torch.cuda.max_memory_reserved(device))),
            "gpu_name": torch.cuda.get_device_name(device),
            "config_sha256": cfg.task.extended_checkpoint_metadata["config_sha256"],
            "source_git_sha": cfg.task.extended_checkpoint_metadata["source_git_sha"],
            "feature_fingerprint": cfg.task.extended_checkpoint_metadata["feature_fingerprint"],
            "raw_feature_tensor_fingerprint": cfg.task.extended_checkpoint_metadata["raw_feature_tensor_fingerprint"],
            "physical_edge_fingerprint": cfg.task.extended_checkpoint_metadata["physical_edge_fingerprint"],
            "split_fingerprint": split_fp, "split_file_sha256": sha256_file(split_path) if split_path and split_path.is_file() else "generated_during_loader",
            "cache_path": cache["cache_path_abs"] if cache else "", "semantic_cache_sha256": cache_sha,
            "candidate_fingerprint": cache["candidate_fingerprint"] if cache else "",
            "v7b_fingerprint": cache["v7b_fingerprint"] if cache else "",
            "initial_model_sha256": initial_model, "initial_classifier_sha256": initial_head,
            "initial_shared_model_sha256": dropouts["initial_shared_model_sha256"],
            "validation_rng_unchanged": bool(acc_audit["validation_rng_unchanged"] and (f1_audit is None or f1_audit["validation_rng_unchanged"])),
            "test_evaluation": False, "best_accuracy_checkpoint": str(accuracy_path),
            "best_accuracy_checkpoint_sha256": acc_sha, "best_f1_checkpoint": str(f1_path) if f1_audit else "",
            "best_f1_checkpoint_sha256": f1_sha, "run_metrics_path": str(metrics_path), "run_dir": str(attempt_dir), "error": "",
        }
        if model_name in {"hybrid_cge_v11", PAPER_MODEL}:
            if metadata["initial_parameter_audit"]["model_state_sha256"] != initial_model or metadata["initial_parameter_audit"]["classifier_state_sha256"] != initial_head:
                raise RuntimeError("NC trainer initial weights differ from the pre-recorded paired initialization")
        if run_row["test_evaluation"] is not False or not run_row["validation_rng_unchanged"]:
            raise RuntimeError("Test/RNG audit did not pass")
        append_csv(DATA / "runs.csv", run_row, ("dataset", "model", "condition", "seed", "attempt"))
        if history:
            history_rows_path = DATA / "epoch_history.csv"
            previous = csv_rows(history_rows_path)
            for item in history:
                previous.append({"dataset": dataset, "model": model_name, "condition": condition or model_name, "seed": seed, **item})
            fields = list(dict.fromkeys([*(previous[0].keys() if previous else []), *(k for row in previous for k in row)]))
            history_rows_path.parent.mkdir(parents=True, exist_ok=True)
            with history_rows_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
                writer.writeheader(); writer.writerows(previous)
        json_write(attempt_dir / "status.json", {"status": "complete", **run_row, "completed_utc": now()})
        print(f"COMPLETE {dataset}/{condition or model_name}/seed{seed}: val_acc={val['acc']:.6f}, val_f1={val['macro_f1']:.6f}", flush=True)
    except Exception as exc:
        failure = {"dataset": dataset, "model": model_name, "condition": condition, "seed": seed,
                   "attempt": attempt, "time_utc": now(), "run_dir": str(attempt_dir),
                   "training_started": metrics_path.is_file(), "error": repr(exc),
                   "traceback": traceback.format_exc(), "test_evaluation": False}
        path = DATA / "failures.json"
        obj = json.loads(path.read_text()) if path.is_file() else {"failures": []}
        obj["failures"].append(failure); json_write(path, obj)
        json_write(attempt_dir / "status.json", {"status": "failed", **failure})
        raise
    finally:
        torch.cuda.empty_cache()
        del safe, raw


def worker_command(args: argparse.Namespace) -> int:
    try:
        if args.action == "worker-preflight":
            worker_preflight(args.dataset, args.model, args.condition, args.seed, args.physical_gpu)
        elif args.action == "worker-train":
            worker_train(args.dataset, args.model, args.condition, args.seed, args.physical_gpu)
        else:
            raise ValueError(args.action)
    except Exception as exc:
        traceback.print_exc()
        if args.action == "worker-preflight":
            append_failure({"stage": "gpu_preflight", "dataset": args.dataset, "model": args.model,
                            "condition": args.condition, "seed": args.seed,
                            "physical_gpu": args.physical_gpu, "error": repr(exc),
                            "traceback": traceback.format_exc()})
        elif args.action == "worker-train":
            failure_path = DATA / "failures.json"
            known = json.loads(failure_path.read_text()).get("failures", []) if failure_path.is_file() else []
            already_recorded = any(
                item.get("dataset") == args.dataset and item.get("model") == args.model
                and item.get("condition") == args.condition and int(item.get("seed", -1)) == args.seed
                for item in known
            )
            if not already_recorded:
                append_failure({"stage": "training_worker_entry", "dataset": args.dataset, "model": args.model,
                                "condition": args.condition, "seed": args.seed,
                                "physical_gpu": args.physical_gpu, "error": repr(exc),
                                "traceback": traceback.format_exc()})
        return 1
    return 0


def guarded_child(action: str, dataset: str, model_name: str, condition: str | None,
                  seed: int, physical_gpu: int) -> None:
    try:
        gpu_guard._assert_physical_gpu_idle(physical_gpu)
    except Exception as exc:
        append_failure({"stage": "pre_run_gpu_guard", "action": action, "dataset": dataset,
                        "model": model_name, "condition": condition, "seed": seed,
                        "physical_gpu": physical_gpu, "error": repr(exc)})
        raise
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["HYBRID_EXP_GUARDED_CELL"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--action", action,
               "--dataset", dataset, "--model", model_name, "--seed", str(seed),
               "--physical-gpu", str(physical_gpu)]
    if condition:
        command += ["--condition", condition]
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"guarded {action} failed for {dataset}/{model_name}/{condition}/seed{seed} (exit {completed.returncode})")


def load_completed() -> set[tuple[str, str, str, int]]:
    return {(r["dataset"], r["model"], r["condition"], int(r["seed"])) for r in csv_rows(DATA / "runs.csv") if r.get("status") == "complete"}



def environment_record() -> dict[str, Any]:
    import platform
    import sklearn
    import dgl
    import faiss
    try:
        import torch_geometric
        pyg = torch_geometric.__version__
    except Exception:
        pyg = None
    snapshot = gpu_guard.environment_record()
    return {**snapshot, "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "dgl": dgl.__version__, "faiss": faiss.__version__,
            "torch_geometric": pyg, "recorded_utc": now()}


def audit_reused_baselines() -> dict[str, Any]:
    """Validate the only three inherited baseline families before reuse."""
    parent_csv = ROOT / "research/hybrid_cge_validation_expansion_v1/data/runs.csv"
    parent_rows = csv_rows(parent_csv)
    source_commit = "2f585c745fa88b55a87f07eb3260a2b49b133f0c"
    expected = {
        (dataset, model, seed)
        for dataset in DATASETS for model in REUSED_BASELINES for seed in SEEDS
    }
    selected = [
        row for row in parent_rows
        if row.get("dataset") in DATASETS and row.get("model") in REUSED_BASELINES
        and row.get("condition") == row.get("model") and int(row.get("seed", -1)) in SEEDS
    ]
    observed = [(row["dataset"], row["model"], int(row["seed"])) for row in selected]
    if len(observed) != 45 or set(observed) != expected:
        raise RuntimeError(f"inherited SAGE/MMGCN/RoleMAG matrix is incomplete or duplicated: {len(observed)}/45")

    checks = []
    imported = []
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg = build_cfg(dataset, "sage", seed, "cpu")
            raw = load_mag_data(cfg, "nc", seed)
            safe = mask_test_labels(raw)
            fingerprints = gpu_guard.dataset_fingerprints(raw)
            for model_name in REUSED_BASELINES:
                row = next(r for r in selected if r["dataset"] == dataset and r["model"] == model_name and int(r["seed"]) == seed)
                metrics_path = Path(row["run_metrics_path"])
                checkpoint_path = Path(row["best_accuracy_checkpoint"])
                if row.get("status") != "complete" or row.get("source_git_sha") != source_commit:
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited status/source commit mismatch")
                if row.get("test_evaluation", "").lower() != "false":
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited row does not prove Test-off")
                if not metrics_path.is_file() or not checkpoint_path.is_file():
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: missing metrics/checkpoint artifact")
                if sha256_file(checkpoint_path) != row.get("best_accuracy_checkpoint_sha256"):
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited checkpoint SHA mismatch")
                packed = json.loads(metrics_path.read_text(encoding="utf-8"))
                if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited training protocol/seed mismatch")
                entry = packed["runs"][0]
                provenance = entry["metadata"].get("study_provenance", {})
                if provenance.get("source_git_sha") != source_commit or provenance.get("test_evaluation") is not False:
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited metadata provenance/Test audit mismatch")
                for field, actual in (
                    ("feature_fingerprint", fingerprints["features"]),
                    ("raw_feature_tensor_fingerprint", fingerprints["features"]),
                    ("physical_edge_fingerprint", fingerprints["physical_edge_index"]),
                    ("split_fingerprint", fingerprints["split_fingerprint"]),
                ):
                    if row.get(field) != actual or (
                        provenance.get(field) is not None and provenance.get(field) != actual
                    ):
                        raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited {field} mismatch")
                set_seed(seed)
                model = get_model(build_cfg(dataset, model_name, seed, "cpu"), safe, None, torch.device("cpu"))
                head = torch.nn.Linear(model.out_dim, int(safe.num_classes))
                initial_model, initial_head, extra = init_pair_hashes(
                    build_cfg(dataset, model_name, seed, "cpu"), safe, None, seed
                )
                if initial_model != row.get("initial_model_sha256") or initial_head != row.get("initial_classifier_sha256"):
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited model/head initialization mismatch")
                try:
                    saved_unregistered = json.loads(row.get("unregistered_tensor_state") or "{}")
                except json.JSONDecodeError:
                    saved_unregistered = {}
                if extra["unregistered_tensor_state"] != saved_unregistered:
                    raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: inherited unregistered ID state mismatch")
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model.load_state_dict(payload["model_state"], strict=True)
                head.load_state_dict(payload["head_state"], strict=True)
                val = entry["metrics"]
                for csv_key, metric_key in (("val_acc", "val_acc"), ("val_macro_f1", "val_macro_f1")):
                    if abs(float(row[csv_key]) - float(val[metric_key])) > 1e-10:
                        raise RuntimeError(f"{dataset}/{model_name}/seed{seed}: CSV/JSON {csv_key} mismatch")
                checks.append({
                    "dataset": dataset, "model": model_name, "seed": seed,
                    "status": "pass", "source_git_sha": source_commit,
                    "feature_fingerprint": fingerprints["features"],
                    "physical_edge_fingerprint": fingerprints["physical_edge_index"],
                    "split_fingerprint": fingerprints["split_fingerprint"],
                    "initial_model_sha256": initial_model,
                    "initial_classifier_sha256": initial_head,
                    "checkpoint_sha256": row["best_accuracy_checkpoint_sha256"],
                    "test_evaluation": False,
                    "split_fingerprint_source": "inherited run CSV row, recomputed from the current loader split",
                })
                imported.append({
                    **row, "condition": model_name, "attempt": "reused_parent",
                    "paper_condition": "", "reused_from_branch": "exp/hybrid_cge_validation_expansion_v1",
                    "reused_source_git_sha": source_commit, "reused": True,
                })
                del model, head
            del raw, safe

    result = {"status": "pass", "checked_rows": len(checks), "rows": checks,
              "source_branch": "exp/hybrid_cge_validation_expansion_v1", "source_commit": source_commit,
              "test_used": False, "created_utc": now()}
    json_write(DATA / "baseline_reuse_audit.json", result)
    for row in imported:
        append_csv(DATA / "runs.csv", row, ("dataset", "model", "condition", "seed", "attempt"))
    return result


def audit_legacy_h2_checkpoint_compatibility() -> dict[str, Any]:
    """Strict-load one untouched historical 4-hop H2 checkpoint per dataset."""
    legacy_paths = (
        ROOT / "research/hybrid_cge_v11/data/runs.csv",
        ROOT / "research/hybrid_cge_validation_expansion_v1/data/runs.csv",
    )
    legacy_rows = [row for path in legacy_paths for row in csv_rows(path)]
    checked = []
    for dataset in DATASETS:
        candidates = [
            row for row in legacy_rows
            if row.get("dataset") == dataset and row.get("status") == "complete"
            and (row.get("condition") in {"H2_hybrid_plus", "H2-Frozen"})
            and int(row.get("seed", -1)) == 42
        ]
        if not candidates:
            raise RuntimeError(f"no historical H2-Frozen 4-hop checkpoint found for {dataset}")
        row = candidates[0]
        checkpoint = Path(row["best_accuracy_checkpoint"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != row.get("best_accuracy_checkpoint_sha256"):
            raise RuntimeError(f"historical H2 checkpoint integrity failed for {dataset}")
        cache = current_cache(dataset)
        cfg = build_cfg(dataset, PAPER_MODEL, 42, "cpu", "H2_full", cache)
        raw = load_mag_data(cfg, "nc", 42)
        safe = mask_test_labels(raw)
        set_seed(42)
        model = get_model(cfg, safe, cache, torch.device("cpu"))
        head = torch.nn.Linear(model.out_dim, int(safe.num_classes))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        head.load_state_dict(payload["head_state"], strict=True)
        checked.append({
            "dataset": dataset, "seed": 42,
            "legacy_condition": row["condition"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": row["best_accuracy_checkpoint_sha256"],
            "strict_model_and_head_load": True,
            "test_evaluation": False,
        })
        del model, head, raw, safe
    result = {"status": "pass", "checked_rows": len(checked), "datasets": checked,
              "preserved_checkpoint_files": True, "test_evaluation": False, "created_utc": now()}
    json_write(DATA / "legacy_h2_checkpoint_compatibility.json", result)
    return result


def preflight_stage(stage: str, physical_gpu: int) -> None:
    if stage == "ablations":
        cells = [(d, PAPER_MODEL, c) for d in DATASETS for c in CONDITIONS]
    elif stage == "hop":
        cells = [(d, PAPER_MODEL, HOP_CONDITION) for d in DATASETS]
    elif stage == "baselines":
        cells = [(d, model, None) for d in DATASETS for model in BASELINES]
    elif stage == "fashion":
        cells = [("ele-fashion", PAPER_MODEL, c) for c in FASHION_CONDITIONS]
    else:
        raise ValueError(f"unknown preflight stage {stage}")
    prior_records = json.loads((DATA / "preflight.json").read_text()).get("runs", [])
    already_passed = {
        (r["dataset"], r["model"], r.get("condition") or "", int(r["seed"]))
        for r in prior_records if r.get("status") == "pass"
    }
    pending = [
        cell for cell in cells
        if (cell[0], cell[1], cell[2] or "", 42) not in already_passed
    ]
    for dataset, model_name, condition in pending:
        guarded_child("worker-preflight", dataset, model_name, condition, 42, physical_gpu)
    records = json.loads((DATA / "preflight.json").read_text()).get("runs", [])
    wanted = {(d, m, c or "", 42) for d, m, c in cells}
    present = {(r["dataset"], r["model"], r.get("condition") or "", int(r["seed"]))
               for r in records if r.get("status") == "pass"}
    if not wanted.issubset(present):
        raise RuntimeError(f"preflight stage {stage} incomplete: {len(wanted & present)}/{len(wanted)}")
    json_write(DATA / f"preflight_{stage}_gate.json", {
        "status": "pass", "stage": stage, "cells": len(wanted), "test_evaluation": False,
        "created_utc": now(),
    })


def training_cells(stage: str):
    if stage == "ablations":
        return [(d, PAPER_MODEL, c, s) for d in DATASETS for c in CONDITIONS for s in SEEDS]
    if stage == "hop":
        return [(d, PAPER_MODEL, HOP_CONDITION, s) for d in DATASETS for s in SEEDS]
    if stage == "baselines":
        return [(d, m, None, s) for d in DATASETS for m in BASELINES for s in SEEDS]
    if stage == "fashion":
        return [("ele-fashion", PAPER_MODEL, c, s) for c in FASHION_CONDITIONS for s in SEEDS]
    raise ValueError(f"unknown training stage {stage}")


def run_training_stage(stage: str, physical_gpu: int) -> None:
    gate_path = DATA / f"preflight_{stage}_gate.json"
    if not gate_path.is_file() or json.loads(gate_path.read_text()).get("status") != "pass":
        raise RuntimeError(f"training stage {stage} requires its successful real-data preflight gate")
    completed = load_completed()
    for dataset, model_name, condition, seed in training_cells(stage):
        key = (dataset, model_name, condition or model_name, seed)
        if key in completed:
            continue
        guarded_child("worker-train", dataset, model_name, condition, seed, physical_gpu)
        completed.add(key)


def verify_stage(stage: str) -> dict[str, Any]:
    cells = training_cells(stage)
    rows = csv_rows(DATA / "runs.csv")
    by_key = {}
    for row in rows:
        if row.get("status") == "complete":
            by_key[(row["dataset"], row["model"], row["condition"], int(row["seed"]))] = row
    selected = []
    for dataset, model_name, condition, seed in cells:
        key = (dataset, model_name, condition or model_name, seed)
        if key not in by_key:
            raise RuntimeError(f"stage {stage} missing complete cell {key}")
        row = by_key[key]
        if str(row.get("test_evaluation")).lower() != "false":
            raise RuntimeError(f"stage {stage} Test-off audit failed for {key}")
        for field in ("val_acc", "val_macro_f1", "val_ce"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"stage {stage} has non-finite {field} in {key}")
        checkpoint = Path(row["best_accuracy_checkpoint"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != row["best_accuracy_checkpoint_sha256"]:
            raise RuntimeError(f"stage {stage} checkpoint integrity failed for {key}")
        if row.get("validation_rng_unchanged", "").lower() != "true":
            raise RuntimeError(f"stage {stage} Validation RNG audit failed for {key}")
        selected.append(row)
    if stage == "ablations":
        for dataset in DATASETS:
            for seed in SEEDS:
                pair = [r for r in selected if r["dataset"] == dataset and int(r["seed"]) == seed]
                if len({r["initial_classifier_sha256"] for r in pair}) != 1:
                    raise RuntimeError(f"ablation classifier initialization is not matched for {dataset}/seed{seed}")
                if len({r.get("initial_shared_model_sha256", "") for r in pair}) != 1:
                    raise RuntimeError(f"ablation shared-parameter initialization is not matched for {dataset}/seed{seed}")
    result = {"status": "pass", "stage": stage, "expected": len(cells), "completed": len(selected),
              "test_evaluation": False, "verified_utc": now(),
              "checkpoint_hashes": [r["best_accuracy_checkpoint_sha256"] for r in selected]}
    json_write(DATA / f"{stage}_stage_gate.json", result)
    return result


def parent_run(args: argparse.Namespace) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        source_data_audit(DATASETS, build_new_cache=False)
        audit_legacy_h2_checkpoint_compatibility()
        audit_reused_baselines()
        json_write(DATA / "environment.json", environment_record())
        return
    if args.action.startswith("preflight-"):
        preflight_stage(args.action.removeprefix("preflight-"), args.physical_gpu)
        return
    if args.action.startswith("train-"):
        stage = args.action.removeprefix("train-")
        run_training_stage(stage, args.physical_gpu)
        verify_stage(stage)
        return
    raise ValueError(args.action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("prepare", "preflight-ablations", "preflight-hop", "preflight-baselines", "preflight-fashion", "train-ablations", "train-hop", "train-baselines", "train-fashion", "worker-preflight", "worker-train"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODEL_CONFIGS)
    parser.add_argument("--condition", choices=CONFIG_CONDITIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", choices=SEEDS)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS)
    parser.add_argument("--models", nargs="*", choices=(PAPER_MODEL, *BASELINES))
    parser.add_argument("--physical-gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action.startswith("worker-"):
        code = worker_command(args)
        raise SystemExit(code)
    parent_run(args)


if __name__ == "__main__":
    main()
