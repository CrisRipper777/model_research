#!/usr/bin/env python3
"""Build, audit, preflight, and run the frozen V7B validation-only NC campaign."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRANCH = "exp/v7b_provenance_relative_semantic_edges"
BASE_SHA = "bcf0d8328642dc6747f8f94f42e56b1717555d64"
PROTOCOL = "unified_full_graph_nc_v1"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "b0_v7a_original",
    "b1_fixed_knn",
    "b2_random_fixed",
    "b3_relative_shared",
    "b4_relative_modality",
)
FREEZE_MESSAGE = "Freeze PSCE-MAG V7B provenance-aware relative semantic edges"
OUTPUT_ROOT = ROOT / "outputs/psce_mag_v7b"
RESEARCH_ROOT = ROOT / "research/psce_mag_v7b"
DATA_ROOT = RESEARCH_ROOT / "data"
FORMAL_ARTIFACTS = {
    "REPORT.md",
    "README.md",
    "data/campaign_manifest.json",
    "data/run_rows.json",
    "data/failures.json",
    "data/summary.csv",
    "data/paired_comparisons.csv",
    "data/semantic_diagnostics.csv",
    "data/resource_profile.json",
    "data/resource_profile.csv",
    "data/random_graph_summary.json",
}
FINGERPRINT_FILES = (
    "configs/config.yaml",
    "configs/model/psce_mag_v7b.yaml",
    "configs/task/nc.yaml",
    "configs/dataset/Movies.yaml",
    "configs/dataset/Grocery.yaml",
    "configs/dataset/ele-fashion.yaml",
    "src/data/semantic_candidates.py",
    "src/data/semantic_candidates_v7b.py",
    "src/data/loaders.py",
    "src/data/graph_utils.py",
    "src/models/psce_mag_v7b.py",
    "src/models/psce_mag_v7a.py",
    "src/tasks/nc.py",
    "src/tasks/common.py",
    "src/tasks/inference.py",
    "src/tasks/lp.py",
    "src/utils/seeds.py",
    "scripts/run_psce_mag_v7b.py",
    "scripts/analyze_psce_mag_v7b.py",
    "tests/test_psce_mag_v7a.py",
    "tests/test_psce_mag_v7b.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dirty_paths() -> set[str]:
    output = git("status", "--porcelain", "--untracked-files=all")
    return {line[3:].split(" -> ")[-1] for line in output.splitlines()}


def _configuration_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"missing fingerprint input: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def find_freeze_sha() -> str | None:
    return git("log", "--format=%H", "-1", f"--grep=^{FREEZE_MESSAGE}$") or None


def provenance(mode: str, *, resume: bool = False) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}, found {branch}")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError(f"experiment ancestry must include {BASE_SHA}")
    dirty = _dirty_paths()
    freeze = find_freeze_sha()
    if mode in {"campaign", "resume"}:
        if freeze is None or head != freeze:
            raise RuntimeError(
                f"formal runs require HEAD at the V7B freeze commit; head={head}, freeze={freeze}"
            )
        if not resume and dirty:
            raise RuntimeError(f"formal campaign requires a clean worktree: {sorted(dirty)}")
        if resume:
            allowed = {f"research/psce_mag_v7b/{path}" for path in FORMAL_ARTIFACTS}
            unexpected = dirty - allowed
            if unexpected:
                raise RuntimeError(f"resume has unexpected modified paths: {sorted(unexpected)}")
    return {
        "branch": branch,
        "baseline_commit_sha": BASE_SHA,
        "head_at_start": head,
        "freeze_commit_sha": freeze,
        "worktree_clean_at_start": not dirty,
        "configuration_fingerprint_sha256": _configuration_fingerprint(),
        "started_at_utc": now(),
    }


def _root_and_dataset_configs(dataset: str):
    from omegaconf import OmegaConf

    if dataset not in DATASETS:
        raise ValueError(f"unsupported V7B dataset {dataset!r}")
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    cfg = OmegaConf.create({"paths": root.paths, "dataset": ds, "seed": 42})
    OmegaConf.resolve(cfg)
    return cfg


def load_candidate_inputs(dataset: str):
    """Read only features and topology; never fetch graph labels or split files."""
    import numpy as np
    import torch

    from src.data.graph_utils import ensure_edge_index, preprocess_edge_index
    from src.data.loaders import resolve_path

    cfg = _root_and_dataset_configs(dataset)
    ds = cfg.dataset
    if dataset in {"Movies", "Grocery"}:
        import dgl

        graph_path = resolve_path(ds.graph_path)
        graphs, _ = dgl.load_graphs(str(graph_path))
        if not graphs:
            raise ValueError(f"no DGL graph found in {graph_path}")
        graph = graphs[0]
        src, dst = graph.edges()
        edge_raw = torch.stack([src.long(), dst.long()], dim=0).contiguous()
        num_nodes = int(graph.num_nodes())
        text_path = resolve_path(ds.text_feat_path)
        visual_path = resolve_path(ds.image_feat_path)
        text = np.load(text_path, mmap_mode="r", allow_pickle=False)
        visual = np.load(visual_path, mmap_mode="r", allow_pickle=False)
        if text.shape[0] != num_nodes or visual.shape[0] != num_nodes:
            raise ValueError(f"{dataset} feature rows do not match DGL graph nodes")
        feature_sources = {"text": text_path, "visual": visual_path}
        text_dim, visual_dim = int(text.shape[1]), int(visual.shape[1])
    else:
        root = resolve_path(ds.root)
        joint_path = resolve_path(ds.joint_feat_path)
        edge_path = resolve_path(ds.edge_path)
        features = torch.load(joint_path, map_location="cpu", weights_only=False).float()
        edge_payload = torch.load(edge_path, map_location="cpu", weights_only=False)
        edge_raw = ensure_edge_index(edge_payload)
        num_nodes = int(features.size(0))
        text_dim, visual_dim = int(ds.text_dim), int(ds.visual_dim)
        if text_dim + visual_dim != features.size(1):
            raise ValueError("ele-fashion modality split does not match clip feature width")
        text = features[:, :text_dim].numpy()
        visual = features[:, text_dim : text_dim + visual_dim].numpy()
        feature_sources = {"text": joint_path, "visual": joint_path}
        del root, features

    edge_index = preprocess_edge_index(
        edge_raw,
        num_nodes,
        make_undirected=bool(ds.get("make_undirected", True)),
        with_self_loops=False,
    )
    info = {
        "dataset": dataset,
        "num_nodes": num_nodes,
        "num_edges_raw": int(edge_raw.size(1)),
        "num_edges_physical": int(edge_index.size(1)),
        "text_dim": text_dim,
        "visual_dim": visual_dim,
        "input_dim": text_dim + visual_dim,
        "labels_or_splits_loaded": False,
        "candidate_phase_label_access": False,
    }
    return text, visual, edge_index, feature_sources, info


def build_candidates(datasets: tuple[str, ...] = DATASETS) -> None:
    import torch
    from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b

    prov = provenance("build-candidates")
    cache_root = OUTPUT_ROOT / "semantic_cache"
    old_summary = read_json(ROOT / "research/psce_mag_v7a/data/candidate_graph_summary.json")
    old_entries = {row["dataset"]: row for row in old_summary["datasets"]}
    records = []
    for dataset in datasets:
        started = time.perf_counter()
        text, visual, physical, sources, info = load_candidate_inputs(dataset)
        path, metadata = build_semantic_candidate_cache_v7b(
            dataset=dataset,
            text_features=text,
            visual_features=visual,
            physical_edge_index=physical,
            cache_dir=cache_root,
            text_source=sources["text"],
            visual_source=sources["visual"],
            top_k=8,
            seed=2026,
            thread_count=8,
        )
        old_entry = old_entries[dataset]
        old_path = ROOT / old_entry["cache_path"]
        old_cache = torch.load(old_path, map_location="cpu", weights_only=False)
        new_cache = torch.load(path, map_location="cpu", weights_only=False)
        exact = {
            key: bool(torch.equal(old_cache[key], new_cache[key]))
            for key in ("edge_index", "unique_edge_index", "degree")
        }
        if not all(exact.values()) or metadata["candidate_fingerprint"] != old_entry["candidate_fingerprint"]:
            raise RuntimeError(f"{dataset}: V7B base candidate graph differs from frozen V7A: {exact}")
        row = {
            **info,
            "cache_path": str(path.relative_to(ROOT)),
            "feature_fingerprint": metadata["feature_fingerprint"],
            "candidate_fingerprint": metadata["candidate_fingerprint"],
            "v7a_candidate_fingerprint": metadata["v7a_candidate_fingerprint"],
            "v7b_fingerprint": metadata["v7b_fingerprint"],
            "v7a_candidate_fingerprint_expected": old_entry["candidate_fingerprint"],
            "tensor_equal_to_v7a": exact,
            "source_counts": metadata["source_counts"],
            "undirected_candidate_edge_count": metadata["undirected_candidate_edge_count"],
            "directed_candidate_edge_count": metadata["directed_candidate_edge_count"],
            "candidate_nonphysical_ratio": metadata["candidate_nonphysical_ratio"],
            "candidate_physical_overlap_ratio": metadata["candidate_physical_overlap_ratio"],
            "text_visual_overlap_jaccard": metadata["text_visual_overlap_jaccard"],
            "isolated_node_count": metadata["isolated_node_count"],
            "covered_node_count": metadata["covered_node_count"],
            "degree_quantiles": metadata["degree_quantiles"],
            "similarity_quantiles": metadata["similarity_quantiles"],
            "retrieval_recall_sample": metadata["retrieval_recall_sample"],
            "index_provenance": metadata["index_provenance"],
            "construction_seconds": metadata["construction_seconds"],
            "wall_seconds": time.perf_counter() - started,
            "cache_contains_labels": False,
        }
        records.append(row)
        print(
            f"[V7B candidates] {dataset}: nodes={info['num_nodes']} "
            f"undirected={row['undirected_candidate_edge_count']} "
            f"nonphysical={row['candidate_nonphysical_ratio']:.4f} "
            f"cache={path}",
            flush=True,
        )
    summary = {
        "rule_version": "psce_mag_v7b_source_flags_relative_edges_v1",
        "top_k_per_modality": 8,
        "task_evaluate_test": False,
        "candidate_phase_label_access": False,
        "datasets": records,
        "provenance": prov,
        "created_at_utc": now(),
    }
    write_json(DATA_ROOT / "candidate_graph_summary.json", summary)


def build_random_graphs() -> None:
    from src.data.semantic_candidates_v7b import write_random_candidate_cache_v7b

    prov = provenance("build-random-graphs")
    summary = {
        "algorithm": "simple undirected degree-preserving double-edge swaps",
        "swaps_per_edge": 3,
        "training_seed_to_topology_seed": {"42": 2026, "43": 2027, "44": 2028},
        "graphs": [],
        "provenance": prov,
    }
    base = {row["dataset"]: row for row in read_json(DATA_ROOT / "candidate_graph_summary.json")["datasets"]}
    random_root = OUTPUT_ROOT / "random_cache"
    for dataset in DATASETS:
        for train_seed, graph_seed in zip(SEEDS, (2026, 2027, 2028)):
            out = random_root / f"{dataset}_training_seed_{train_seed}_topology_{graph_seed}.pt"
            started = time.perf_counter()
            result = write_random_candidate_cache_v7b(
                ROOT / base[dataset]["cache_path"], out, seed=graph_seed, swaps_per_edge=3
            )
            record = {
                "dataset": dataset,
                "training_seed": train_seed,
                **result,
                "source_candidate_fingerprint": base[dataset]["candidate_fingerprint"],
                "feature_fingerprint": base[dataset]["feature_fingerprint"],
                "wall_seconds": time.perf_counter() - started,
                "cache_path": str(out.relative_to(ROOT)),
            }
            summary["graphs"].append(record)
            write_json(DATA_ROOT / "random_graph_summary.json", summary)
            print(f"[V7B random graph] {dataset} seed={train_seed}: swaps={record['successful_swaps']} changed={record['changed_edge_count']}", flush=True)
    for dataset in DATASETS:
        selected = [row for row in summary["graphs"] if row["dataset"] == dataset]
        fingerprints = {row["candidate_fingerprint"] for row in selected}
        if len(selected) != 3 or len(fingerprints) != 3:
            raise RuntimeError(f"{dataset}: seed-specific random graphs are not three distinct topologies")
        if any(row["candidate_fingerprint"] == row["source_candidate_fingerprint"] for row in selected):
            raise RuntimeError(f"{dataset}: random rewiring did not change the base topology")
    summary["completed_at_utc"] = now()
    write_json(DATA_ROOT / "random_graph_summary.json", summary)


def _nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15, check=True
        )
        return result.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def _gpu_query() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device: str) -> dict[str, Any]:
    import torch
    import torch_geometric
    import faiss

    device_obj = torch.device(device)
    selected = None
    if device_obj.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        props = torch.cuda.get_device_properties(index)
        selected = {
            "device_index": index,
            "device_name": props.name,
            "total_memory_bytes": int(total_bytes),
            "free_memory_bytes": int(free_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        }
    return {
        "recorded_at_utc": now(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "faiss": faiss.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": device,
        "selected_device": selected,
        "nvidia_smi_snapshot": _nvidia_smi(),
        "gpu_query": _gpu_query(),
        "other_processes_terminated": False,
    }


def _experiment_cfg(dataset: str, seed: int, variant: str):
    from omegaconf import OmegaConf

    root = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg = OmegaConf.create(
        {
            "paths": root.paths,
            "dataset": OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml"),
            "task": OmegaConf.load(ROOT / "configs/task/nc.yaml"),
            "model": OmegaConf.load(ROOT / "configs/model/psce_mag_v7b.yaml"),
            "seed": int(seed),
            "num_runs": 1,
            "device": "cpu",
        }
    )
    cfg.model.variant = variant
    cfg.task.evaluate_test = False
    cfg.task.training_mode = "full_graph"
    OmegaConf.resolve(cfg)
    return cfg


def _candidate_entry(dataset: str, variant: str | None = None, seed: int | None = None) -> dict[str, Any]:
    summary = read_json(DATA_ROOT / "candidate_graph_summary.json")
    for item in summary["datasets"]:
        if item["dataset"] == dataset:
            if variant == "b2_random_fixed":
                if seed is None:
                    raise ValueError("B2 candidate lookup requires the training seed")
                random_summary = read_json(DATA_ROOT / "random_graph_summary.json")
                for random_item in random_summary["graphs"]:
                    if random_item["dataset"] == dataset and int(random_item["training_seed"]) == int(seed):
                        return {
                            **item,
                            "cache_path": random_item["cache_path"],
                            "candidate_fingerprint": random_item["candidate_fingerprint"],
                            "v7b_fingerprint": random_item.get("v7b_fingerprint"),
                            "random_topology_seed": random_item["seed"],
                        }
                raise KeyError(f"B2 random candidate graph is missing for {dataset} seed {seed}")
            return item
    raise KeyError(f"semantic candidate cache not present for {dataset}")


def run_raw_regression(device: str = "cuda:1") -> dict[str, Any]:
    """Compare V7A A2 and V7B B0 with the same weights on Movies full graph."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from src.models.psce_mag_v7a import Model as V7AModel
    from src.models.psce_mag_v7b import Model as V7BModel

    prov = provenance("raw-regression")
    checkpoint_path = ROOT / "outputs/psce_mag_v7a/formal/runs/Movies/seed_42/a2_conditional_dual/best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen V7A A2 Movies checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    text, visual, physical, _, data_info = load_candidate_inputs("Movies")
    x = torch.from_numpy(np.concatenate([np.asarray(text, dtype=np.float32), np.asarray(visual, dtype=np.float32)], axis=1).copy())
    model_info = {"input_dim": data_info["input_dim"], "text_dim": data_info["text_dim"], "visual_dim": data_info["visual_dim"], "num_nodes": data_info["num_nodes"]}
    v7a_cfg = OmegaConf.create({"model": OmegaConf.load(ROOT / "configs/model/psce_mag_v7a.yaml")})
    v7a_cfg.model.variant = "a2_conditional_dual"
    v7b_cfg = _experiment_cfg("Movies", 42, "b0_v7a_original")
    reference = V7AModel(v7a_cfg, model_info)
    candidate = V7BModel(v7b_cfg, model_info)
    old_cache = ROOT / _candidate_entry("Movies")["cache_path"]
    new_cache = ROOT / _candidate_entry("Movies", "b0_v7a_original", 42)["cache_path"]
    reference.load_semantic_cache(str(old_cache), expected_dataset="Movies")
    candidate.load_semantic_cache(str(new_cache), expected_dataset="Movies")
    reference.load_state_dict(checkpoint["model_state"], strict=True)
    candidate.load_state_dict(checkpoint["model_state"], strict=True)
    device_obj = torch.device(device)
    reference.to(device_obj).eval(); candidate.to(device_obj).eval()
    edge = physical.to(device_obj)
    x = x.to(device_obj)
    with torch.no_grad():
        old = reference(x, edge, return_details=True)
        new = candidate(x, edge, return_details=True)
    z_abs = float((old[0] - new[0]).abs().max().cpu())
    aux_abs = float((old[3] - new[3]).abs().max().cpu())
    old_weight = old[4]["details"]["edge_weight"]
    new_weight = new[4]["details"]["edge_weight"]
    passed = z_abs <= 2e-6 and aux_abs <= 2e-6 and old_weight == new_weight
    summary = {
        "passed": passed,
        "regression": "same V7A A2 checkpoint and real Movies full graph: V7A A2 vs V7B B0",
        "dataset": "Movies",
        "num_nodes": data_info["num_nodes"],
        "candidate_fingerprint_v7a": reference.semantic_candidate_fingerprint,
        "candidate_fingerprint_v7b": candidate.semantic_candidate_fingerprint,
        "weights_loaded_strictly": True,
        "max_abs_embedding_error": z_abs,
        "max_abs_auxiliary_loss_error": aux_abs,
        "semantic_edge_weight_summary_equal": old_weight == new_weight,
        "tolerance": {"atol": 2e-6, "rtol": 2e-5},
        "device": device,
        "test_metrics_generated": False,
        "provenance": prov,
        "recorded_at_utc": now(),
    }
    write_json(DATA_ROOT / "raw_regression_summary.json", summary)
    if not passed:
        raise RuntimeError(f"V7A/B0 real-graph regression failed: {summary}")
    return summary

def run_preflight(device: str) -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.common import build_optimizer
    from src.tasks.nc import _resolve_nc_eval_labels
    from src.utils.seeds import set_seed

    prov = provenance("preflight")
    device_obj = torch.device(device)
    env = environment(device)
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "task_evaluate_test": False,
        "validation_only": True,
        "preflight_target_policy": "real train-split labels for one gradient step; full-graph validation only",
        "device": device,
        "expected_datasets": list(DATASETS),
        "datasets": [],
        "environment": env,
        "passed": False,
        "provenance": prov,
    }
    write_json(DATA_ROOT / "environment.json", env)
    write_json(DATA_ROOT / "preflight_summary.json", summary)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        summary["failure"] = "requested CUDA device is unavailable; GPU preflight did not run"
        write_json(DATA_ROOT / "preflight_summary.json", summary)
        raise RuntimeError(summary["failure"])
    torch.cuda.set_device(device_obj)
    for dataset in DATASETS:
        row: dict[str, Any] = {"dataset": dataset, "status": "running"}
        row["gpu_snapshot_before"] = _gpu_query()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)
        row["cuda_free_before_bytes"] = int(free_bytes)
        row["cuda_total_bytes"] = int(total_bytes)
        if free_bytes < 2 * 1024**3:
            row["status"] = "insufficient_free_memory_before_run"
            row["passed"] = False
            summary["datasets"].append(row)
            summary["failure"] = {
                "dataset": dataset,
                "reason": "less than 2 GiB free before preflight; no process was stopped",
            }
            write_json(DATA_ROOT / "preflight_summary.json", summary)
            raise RuntimeError(summary["failure"]["reason"])

        started = time.perf_counter()
        cfg = _experiment_cfg(dataset, 42, "b4_relative_modality")
        cfg.model.semantic_cache_path = str(ROOT / _candidate_entry(dataset)["cache_path"])
        cfg.model.semantic_feature_fingerprint = _candidate_entry(dataset)["feature_fingerprint"]
        cfg.model.semantic_candidate_fingerprint = _candidate_entry(dataset)["candidate_fingerprint"]
        cfg.model.semantic_v7b_fingerprint = _candidate_entry(dataset)["v7b_fingerprint"]
        data = load_mag_data(cfg, "nc", 42)
        data_info = {
            "input_dim": data.input_dim,
            "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1)),
        }
        row.update(
            {
                **data_info,
                "directed_physical_edges": int(data.edge_index.size(1)),
                "train_nodes": int(data.train_idx.numel()),
                "validation_nodes": int(data.val_idx.numel()),
                "test_metrics_generated": False,
                "labels_or_splits_loaded_for_preflight": True,
            }
        )
        set_seed(42)
        model = build_model(cfg, data_info)
        model.load_semantic_cache(
            str(cfg.model.semantic_cache_path),
            expected_dataset=dataset,
            expected_feature_fingerprint=str(cfg.model.semantic_feature_fingerprint),
            expected_candidate_fingerprint=str(cfg.model.semantic_candidate_fingerprint),
            expected_v7b_fingerprint=str(cfg.model.semantic_v7b_fingerprint),
        )
        model = model.to(device_obj).train()
        classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device_obj)
        model.profile_forward = True
        optimizer = build_optimizer(
            list(model.parameters()) + list(classifier.parameters()), cfg, model=model
        )
        x_all = data.x.to(device_obj)
        edge_index = data.edge_index.to(device_obj)
        train_idx = data.train_idx.to(device_obj)
        labels_train = data.y[data.train_idx].to(device_obj)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device_obj)
        train_started = time.perf_counter()
        try:
            z, _, _, aux_loss, _ = model(x_all, edge_index)
            logits = classifier(z.index_select(0, train_idx))
            loss = nn.functional.cross_entropy(logits, labels_train) + aux_loss
            loss.backward()
            gradient_diagnostics = model.capture_gradient_diagnostics()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=float(cfg.task.grad_clip),
                error_if_nonfinite=True,
            )
            optimizer.step()
            torch.cuda.synchronize(device_obj)
            train_seconds = time.perf_counter() - train_started
            train_loss = float(loss.detach().cpu())
            forward_timing = dict(model._last_timing)
            del z, logits, loss, aux_loss

            model.eval()
            classifier.eval()
            validation_started = time.perf_counter()
            with torch.no_grad():
                z_validation = model(x_all, edge_index)[0].detach().cpu()
                pred = classifier(z_validation[data.val_idx].to(device_obj)).argmax(-1).cpu()
            torch.cuda.synchronize(device_obj)
            validation_seconds = time.perf_counter() - validation_started
            target = data.y[data.val_idx].cpu()
            validation = {
                "val_acc": float((pred == target).float().mean()),
                "val_macro_f1": float(
                    f1_score(
                        target.numpy(),
                        pred.numpy(),
                        labels=_resolve_nc_eval_labels(data, include_test=False),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
            gradients_ok = all(
                gradient_diagnostics[name]["gradient_finite"]
                and gradient_diagnostics[name]["gradient_nonzero"]
                for name in (
                    "relation_projections.text.weight",
                    "relation_projections.visual.weight",
                    "relation_scorer.0.weight",
                    "relation_scorer.2.weight",
                    "relation_scorer.output_text",
                    "relation_scorer.output_visual",
                )
            )
            finite = bool(torch.isfinite(z_validation).all()) and all(
                math.isfinite(float(value)) for value in validation.values()
            ) and math.isfinite(train_loss)
            passed = finite and gradients_ok
            row.update(
                {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "training_loss": train_loss,
                    "validation_metrics": validation,
                    "training_step_seconds": train_seconds,
                    "validation_inference_seconds": validation_seconds,
                    "relation_scoring_seconds": forward_timing.get("relation_scoring_seconds"),
                    "semantic_propagation_seconds": forward_timing.get("semantic_propagation_seconds"),
                    "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
                    "gradient_diagnostics": gradient_diagnostics,
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)),
                    "cuda_free_after_bytes": int(torch.cuda.mem_get_info(device_obj)[0]),
                    "gpu_snapshot_after": _gpu_query(),
                    "wall_seconds": time.perf_counter() - started,
                    "test_metric_keys_present": False,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            row.update(
                {
                    "status": "oom",
                    "passed": False,
                    "error": str(exc),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)),
                    "gpu_snapshot_after": _gpu_query(),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        summary["datasets"].append(row)
        write_json(DATA_ROOT / "preflight_summary.json", summary)
        print(
            f"[V7B preflight] {dataset}: {row['status']} "
            f"peak_alloc={row.get('cuda_peak_allocated_bytes')}",
            flush=True,
        )
        del model, classifier, data, x_all, edge_index, train_idx, labels_train
        torch.cuda.empty_cache()
        if not row.get("passed", False):
            summary["failure"] = {"dataset": dataset, "reason": row.get("error", row["status"])}
            write_json(DATA_ROOT / "preflight_summary.json", summary)
            raise RuntimeError(f"V7B preflight failed on {dataset}: {summary['failure']['reason']}")
    summary["passed"] = len(summary["datasets"]) == len(DATASETS) and all(
        item.get("passed") for item in summary["datasets"]
    )
    summary["completed_at_utc"] = now()
    write_json(DATA_ROOT / "preflight_summary.json", summary)
    write_json(DATA_ROOT / "environment.json", env)


def run_tests() -> None:
    prov = provenance("tests")
    log_path = OUTPUT_ROOT / "test_suite.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q"]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        exit_code = process.wait()
    output = log_path.read_text(encoding="utf-8", errors="replace")
    summary = {
        "command": command,
        "exit_code": exit_code,
        "passed": exit_code == 0,
        "elapsed_seconds": time.perf_counter() - started,
        "output_tail": output[-12000:],
        "provenance": prov,
        "recorded_at_utc": now(),
    }
    write_json(DATA_ROOT / "test_summary.json", summary)
    if exit_code != 0:
        raise RuntimeError(f"full pytest suite failed with exit code {exit_code}")


def validate_run(metrics_path: Path, checkpoint_path: Path, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing metrics/checkpoint: {metrics_path} / {checkpoint_path}")
    payload = read_json(metrics_path)
    if payload.get("protocol_version") != PROTOCOL or len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"wrong protocol or run count in {metrics_path}")
    run = payload["runs"][0]
    metrics = dict(run.get("metrics", {}))
    if any(str(key).lower().startswith("test") for key in metrics):
        raise RuntimeError(f"Test metric key present in {metrics_path}")
    for key in ("val_acc", "val_macro_f1"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise FloatingPointError(f"missing or nonfinite {key} in {metrics_path}")
    metadata = dict(run.get("metadata", {}))
    if int(metadata.get("best_epoch", 0)) < 1:
        raise RuntimeError(f"no Validation Accuracy-selected checkpoint in {metrics_path}")
    config_path = metrics_path.parent / "hydra/.hydra/config.yaml"
    if config_path.is_file():
        resolved_cfg = OmegaConf.load(config_path)
        if bool(resolved_cfg.task.get("evaluate_test", True)):
            raise RuntimeError(f"Hydra config has evaluate_test=true: {config_path}")
        if str(resolved_cfg.model.variant) != variant:
            raise RuntimeError(f"wrong model variant in Hydra config: {config_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(checkpoint.get("metrics", {})) != {"val_acc", "val_macro_f1"}:
        raise RuntimeError(f"checkpoint metric keys are not validation-only: {checkpoint_path}")
    if checkpoint.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"checkpoint was not selected by Validation Accuracy: {checkpoint_path}")
    candidate = _candidate_entry(dataset, variant, seed)
    expected_fp = candidate["candidate_fingerprint"]
    observed_bytes = checkpoint["model_state"]["semantic_fingerprint_state"].tolist()
    observed_fp = bytes(observed_bytes).hex()
    if observed_fp != expected_fp:
        raise RuntimeError(f"checkpoint candidate fingerprint mismatch: {checkpoint_path}")
    expected_feature_fp = candidate["feature_fingerprint"]
    observed_feature_fp = bytes(
        checkpoint["model_state"]["semantic_feature_fingerprint_state"].tolist()
    ).hex()
    if observed_feature_fp != expected_feature_fp:
        raise RuntimeError(f"checkpoint feature fingerprint mismatch: {checkpoint_path}")
    if checkpoint.get("run_metadata", {}).get("semantic_v7b_fingerprint") != candidate["v7b_fingerprint"]:
        raise RuntimeError(f"checkpoint V7B provenance fingerprint mismatch: {checkpoint_path}")
    return {"metrics": metrics, "metadata": metadata, "checkpoint": checkpoint}


def run_one(
    dataset: str,
    seed: int,
    variant: str,
    *,
    device: str,
    mode: str,
    epochs: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / mode / "runs" / dataset / f"seed_{seed}" / variant
    metrics_path = run_dir / "run_metrics.json"
    checkpoint_path = run_dir / "best.pt"
    if resume and metrics_path.is_file() and checkpoint_path.is_file():
        checked = validate_run(metrics_path, checkpoint_path, variant, dataset, seed)
        status = "reused"
    else:
        if not resume and (metrics_path.exists() or checkpoint_path.exists()):
            raise RuntimeError(f"existing run files require --resume: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "src.main",
            f"dataset={dataset}",
            "task=nc",
            "model=psce_mag_v7b",
            f"model.variant={variant}",
            f"seed={seed}",
            "num_runs=1",
            f"device={device}",
            "task.evaluate_test=false",
            "task.training_mode=full_graph",
            "task.inference_mode=full",
            f"task.run_metrics_path={metrics_path}",
            f"task.save_ckpt_path={checkpoint_path}",
            f"hydra.run.dir={run_dir / 'hydra'}",
        ]
        candidate = _candidate_entry(dataset, variant, seed)
        command.extend(
            [
                f"model.semantic_cache_path={ROOT / candidate['cache_path']}",
                f"model.semantic_feature_fingerprint={candidate['feature_fingerprint']}",
                f"model.semantic_candidate_fingerprint={candidate['candidate_fingerprint']}",
                f"model.semantic_v7b_fingerprint={candidate['v7b_fingerprint']}",
            ]
        )
        if epochs is not None:
            command.extend(
                [
                    f"task.epochs={epochs}",
                    "task.early_stop_min_epoch=1",
                    "task.patience=1",
                    "task.early_stop_min_delta=0.0",
                ]
            )
        log_path = run_dir / "training.log"
        print(f"[V7B {mode}] {dataset} seed={seed} {variant} device={device}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
            return_code = process.wait()
        if return_code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-10000:]
            raise RuntimeError(f"training exited {return_code}; log tail:\n{tail}")
        checked = validate_run(metrics_path, checkpoint_path, variant, dataset, seed)
        status = "completed"
    metadata = checked["metadata"]
    return {
        "status": status,
        "mode": mode,
        "run_id": f"{dataset}__seed{int(seed)}__{variant}",
        "dataset": dataset,
        "seed": int(seed),
        "variant": variant,
        "metrics": checked["metrics"],
        "metadata": metadata,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "semantic_checkpoint_diagnostics": metadata.get("semantic_checkpoint_diagnostics"),
    }


def _audit_smoke(row: dict[str, Any], dataset: str, variant: str) -> dict[str, Any]:
    metrics = row["metrics"]
    return {
        "validation_metrics_finite": all(
            math.isfinite(float(metrics[key])) for key in ("val_acc", "val_macro_f1")
        ),
        "test_metrics_absent": not any(key.lower().startswith("test") for key in metrics),
        "best_epoch": row["metadata"].get("best_epoch"),
        "model_parameters": row["metadata"].get("model_parameters"),
        "semantic_diagnostics_present": row.get("semantic_checkpoint_diagnostics") is not None,
        "candidate_fingerprint": (
            _candidate_entry(dataset, variant, int(row["seed"]))["candidate_fingerprint"]
        ),
    }


def run_smoke(device: str) -> None:
    prov = provenance("smoke")
    preflight = read_json(DATA_ROOT / "preflight_summary.json")
    tests = read_json(DATA_ROOT / "test_summary.json")
    if not preflight.get("passed") or not tests.get("passed"):
        raise RuntimeError("smoke requires passing preflight and full pytest")
    rows, failures = [], []
    for variant in VARIANTS:
        try:
            row = run_one("Movies", 42, variant, device=device, mode="smoke", epochs=1)
            row["audit"] = _audit_smoke(row, "Movies", variant)
            rows.append(row)
        except Exception as exc:
            failures.append({"variant": variant, "error": repr(exc)})
        write_json(
            DATA_ROOT / "smoke_summary.json",
            {
                "protocol": PROTOCOL,
                "task_evaluate_test": False,
                "dataset": "Movies",
                "seed": 42,
                "epochs": 1,
                "expected_runs": len(VARIANTS),
                "completed_runs": len(rows),
                "failures": failures,
                "rows": rows,
                "provenance": prov,
                "updated_at_utc": now(),
            },
        )
        print(f"[V7B smoke] progress={len(rows)}/{len(VARIANTS)} failures={len(failures)}", flush=True)
        if failures:
            raise RuntimeError(f"V7B smoke failed: {failures[-1]}")
    passed = len(rows) == len(VARIANTS) and all(
        row["audit"]["validation_metrics_finite"]
        and row["audit"]["test_metrics_absent"]
        and row["audit"]["semantic_diagnostics_present"]
        for row in rows
    )
    final = read_json(DATA_ROOT / "smoke_summary.json")
    final["audit_passed"] = passed
    final["checks"] = {
        "all_variants_complete": len(rows) == len(VARIANTS),
        "validation_metrics_finite": passed,
        "test_metrics_absent": passed,
        "no_failures": not failures,
    }
    final["completed_at_utc"] = now()
    write_json(DATA_ROOT / "smoke_summary.json", final)
    if not passed:
        raise RuntimeError("V7B smoke audit failed")


def run_campaign(device: str, *, resume: bool) -> None:
    import torch

    mode = "resume" if resume else "campaign"
    prov = provenance(mode, resume=resume)
    preflight = read_json(DATA_ROOT / "preflight_summary.json")
    smoke = read_json(DATA_ROOT / "smoke_summary.json")
    tests = read_json(DATA_ROOT / "test_summary.json")
    regression = read_json(DATA_ROOT / "raw_regression_summary.json")
    if not preflight.get("passed") or not smoke.get("audit_passed"):
        raise RuntimeError("campaign requires passing GPU preflight and all-variant smoke")
    if not tests.get("passed") or tests.get("exit_code") != 0:
        raise RuntimeError("campaign requires a passing full repository pytest record")
    if not regression.get("passed"):
        raise RuntimeError("campaign requires passing historical V4A real-graph regression")
    expected_fp = prov["configuration_fingerprint_sha256"]
    for label, item in (("preflight", preflight), ("smoke", smoke), ("tests", tests), ("raw-regression", regression)):
        recorded = item.get("provenance", {}).get("configuration_fingerprint_sha256")
        if recorded != expected_fp:
            raise RuntimeError(f"{label} used implementation fingerprint {recorded}, expected {expected_fp}")

    rows_path = DATA_ROOT / "run_rows.json"
    manifest_path = DATA_ROOT / "campaign_manifest.json"
    failures_path = DATA_ROOT / "failures.json"
    if not resume and (rows_path.exists() or manifest_path.exists()):
        raise RuntimeError("formal campaign artifacts already exist; use --mode resume")
    rows = read_json(rows_path) if resume and rows_path.is_file() else []
    failures = read_json(failures_path) if resume and failures_path.is_file() else []
    old_manifest = read_json(manifest_path) if resume and manifest_path.is_file() else None
    if old_manifest:
        old_prov = old_manifest.get("provenance", {})
        if old_prov.get("freeze_commit_sha") != prov.get("freeze_commit_sha"):
            raise RuntimeError("resume freeze SHA mismatch")
        if old_prov.get("configuration_fingerprint_sha256") != expected_fp:
            raise RuntimeError("resume implementation fingerprint mismatch")
    row_keys = [(row.get("dataset"), int(row.get("seed", -1)), row.get("variant")) for row in rows]
    run_ids = [row.get("run_id") for row in rows]
    if len(set(row_keys)) != len(row_keys) or len(set(run_ids)) != len(run_ids):
        raise RuntimeError("formal run rows contain duplicate keys or run IDs")
    for row in rows:
        if row.get("freeze_commit_sha") != prov["freeze_commit_sha"] or row.get("configuration_fingerprint_sha256") != expected_fp:
            raise RuntimeError("formal run row has a mismatched freeze SHA or implementation fingerprint")
    completed = {
        (row["dataset"], int(row["seed"]), row["variant"])
        for row in rows
        if row.get("status") in {"completed", "reused"}
    }
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                key = (dataset, seed, variant)
                if key in completed:
                    continue
                try:
                    row = run_one(
                        dataset,
                        seed,
                        variant,
                        device=device,
                        mode="formal",
                        resume=resume,
                    )
                    row["freeze_commit_sha"] = prov["freeze_commit_sha"]
                    row["configuration_fingerprint_sha256"] = expected_fp
                    if row.get("run_id") != f"{dataset}__seed{seed}__{variant}":
                        raise RuntimeError("formal run ID does not uniquely identify dataset/seed/variant")
                    rows.append(row)
                    completed.add(key)
                    for failure in failures:
                        if (
                            failure.get("dataset"),
                            int(failure.get("seed", -1)),
                            failure.get("variant"),
                        ) == key:
                            failure["resolved"] = True
                            failure["resolved_at_utc"] = now()
                except Exception as exc:
                    message = repr(exc)
                    failures.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "variant": variant,
                            "error": message,
                            "oom": "out of memory" in message.lower() or "cuda oom" in message.lower(),
                            "resolved": False,
                            "at_utc": now(),
                        }
                    )
                    write_json(failures_path, failures)
                    write_json(rows_path, rows)
                    write_json(
                        manifest_path,
                        {
                            "protocol": PROTOCOL,
                            "task_evaluate_test": False,
                            "expected_runs": 45,
                            "completed_runs": len(completed),
                            "datasets": DATASETS,
                            "seeds": SEEDS,
                            "variants": VARIANTS,
                            "failures": failures,
                            "provenance": prov,
                            "updated_at_utc": now(),
                            "test_metric_keys_present": False,
                        },
                    )
                    raise RuntimeError(f"formal run failed for {key}: {message}") from exc
                write_json(rows_path, rows)
                write_json(failures_path, failures)
                write_json(
                    manifest_path,
                    {
                        "protocol": PROTOCOL,
                        "task_evaluate_test": False,
                        "expected_runs": 45,
                        "completed_runs": len(completed),
                        "datasets": DATASETS,
                        "seeds": SEEDS,
                        "variants": VARIANTS,
                        "failures": failures,
                        "provenance": prov,
                        "updated_at_utc": now(),
                        "test_metric_keys_present": False,
                    },
                )
                print(f"[V7B campaign] progress={len(completed)}/45", flush=True)
    if len(completed) != 45 or any(not item.get("resolved", False) for item in failures):
        raise RuntimeError(f"campaign incomplete: completed={len(completed)}/45")
    if any(
        any(str(key).lower().startswith("test") for key in row["metrics"])
        for row in rows
    ):
        raise RuntimeError("Test metric key found in formal campaign rows")
    subprocess.run(
        [sys.executable, "scripts/analyze_psce_mag_v7b.py"], cwd=ROOT, check=True
    )
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "status": "complete",
            "completed_runs": 45,
            "completed_at_utc": now(),
            "test_metric_keys_present": False,
            "nvidia_smi_at_completion": _nvidia_smi(),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        }
    )
    write_json(manifest_path, manifest)
    print("[V7B campaign] all 45 validation-only runs completed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("build-candidates", "build-random-graphs", "raw-regression", "tests", "preflight", "smoke", "campaign", "resume"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    args = parser.parse_args()
    if args.mode == "build-candidates":
        build_candidates(tuple(args.dataset) if args.dataset else DATASETS)
    elif args.mode == "build-random-graphs":
        build_random_graphs()
    elif args.mode == "raw-regression":
        run_raw_regression(args.device)
    elif args.mode == "tests":
        run_tests()
    elif args.mode == "preflight":
        run_preflight(args.device)
    elif args.mode == "smoke":
        run_smoke(args.device)
    elif args.mode == "campaign":
        run_campaign(args.device, resume=False)
    else:
        run_campaign(args.device, resume=True)


if __name__ == "__main__":
    main()
