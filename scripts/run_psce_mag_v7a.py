#!/usr/bin/env python3
"""Build, preflight, smoke, and run the frozen V7A validation-only NC campaign."""

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

BRANCH = "exp/v7a_physical_semantic_collaborative_experts"
BASE_SHA = "af848e6c241994d34827fab8d37611159d167965"
PROTOCOL = "unified_full_graph_nc_v1"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("a0_raw", "a1_static_dual", "a2_conditional_dual")
FREEZE_MESSAGE = "Freeze PSCE-MAG V7A physical-semantic collaborative experts"
OUTPUT_ROOT = ROOT / "outputs/psce_mag_v7a"
RESEARCH_ROOT = ROOT / "research/psce_mag_v7a"
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
}
FINGERPRINT_FILES = (
    "configs/model/psce_mag_v7a.yaml",
    "configs/task/nc.yaml",
    "src/data/semantic_candidates.py",
    "src/models/psce_mag_v7a.py",
    "src/tasks/nc.py",
    "src/tasks/lp.py",
    "scripts/run_psce_mag_v7a.py",
    "scripts/analyze_psce_mag_v7a.py",
    "tests/test_psce_mag_v7a.py",
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
                f"formal runs require HEAD at the V7A freeze commit; head={head}, freeze={freeze}"
            )
        if not resume and dirty:
            raise RuntimeError(f"formal campaign requires a clean worktree: {sorted(dirty)}")
        if resume:
            allowed = {f"research/psce_mag_v7a/{path}" for path in FORMAL_ARTIFACTS}
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
        raise ValueError(f"unsupported V7A dataset {dataset!r}")
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
    from src.data.semantic_candidates import build_semantic_candidate_cache

    prov = provenance("build-candidates")
    cache_root = OUTPUT_ROOT / "semantic_cache"
    records = []
    for dataset in datasets:
        started = time.perf_counter()
        text, visual, physical, sources, info = load_candidate_inputs(dataset)
        path, metadata = build_semantic_candidate_cache(
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
        row = {
            **info,
            "cache_path": str(path.relative_to(ROOT)),
            "feature_fingerprint": metadata["feature_fingerprint"],
            "candidate_fingerprint": metadata["candidate_fingerprint"],
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
            f"[V7A candidates] {dataset}: nodes={info['num_nodes']} "
            f"undirected={row['undirected_candidate_edge_count']} "
            f"nonphysical={row['candidate_nonphysical_ratio']:.4f} "
            f"cache={path}",
            flush=True,
        )
    summary = {
        "rule_version": "psce_mag_v7a_mean_center_cosine_union_sym_v1",
        "top_k_per_modality": 8,
        "task_evaluate_test": False,
        "candidate_phase_label_access": False,
        "datasets": records,
        "provenance": prov,
        "created_at_utc": now(),
    }
    write_json(DATA_ROOT / "candidate_graph_summary.json", summary)


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
            "model": OmegaConf.load(ROOT / "configs/model/psce_mag_v7a.yaml"),
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


def _candidate_entry(dataset: str) -> dict[str, Any]:
    summary = read_json(DATA_ROOT / "candidate_graph_summary.json")
    for item in summary["datasets"]:
        if item["dataset"] == dataset:
            return item
    raise KeyError(f"semantic candidate cache not present for {dataset}")


def run_raw_regression(device: str = "cpu") -> dict[str, Any]:
    """Map historical V4A R0 weights and compare full-graph output on Movies."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from src.models.mvcge_mag_v4a import Model as V4AModel
    from src.models.psce_mag_v7a import Model as V7AModel

    checkpoint_path = (
        ROOT
        / "outputs/mvcge_mag_v4a_raw_anchored_residual_screen/formal/runs/Movies/seed_42/R0_raw/best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"historical V4A R0 checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    text, visual, edge_cpu, _, data_info = load_candidate_inputs("Movies")
    x_cpu = torch.from_numpy(
        np.concatenate(
            [np.asarray(text, dtype=np.float32), np.asarray(visual, dtype=np.float32)],
            axis=-1,
        ).copy()
    )
    if tuple(x_cpu.shape) != (int(data_info["num_nodes"]), int(data_info["input_dim"])):
        raise ValueError("Movies feature matrix has an unexpected shape")
    info = dict(checkpoint["data_info"])
    if info["num_nodes"] != data_info["num_nodes"]:
        raise ValueError("historical V4A checkpoint and Movies graph node counts differ")

    v4_cfg = OmegaConf.load(ROOT / "configs/model/mvcge_mag_v4a.yaml")
    v4_cfg.variant = "R0_raw"
    reference = V4AModel(OmegaConf.create({"model": v4_cfg}), info)
    reference.load_state_dict(checkpoint["model_state"], strict=True)
    reference.to(device).eval()
    with torch.no_grad():
        z_reference, _, _, aux_reference, _ = reference(
            x_cpu.to(device), edge_cpu.to(device)
        )
        z_reference = z_reference.detach().cpu()
        aux_reference = aux_reference.detach().cpu()
    del reference
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    v7_cfg = OmegaConf.load(ROOT / "configs/model/psce_mag_v7a.yaml")
    v7_cfg.variant = "a0_raw"
    candidate = V7AModel(OmegaConf.create({"model": v7_cfg}), info)
    old_state = checkpoint["model_state"]
    new_state = candidate.state_dict()
    common = {
        key: value
        for key, value in old_state.items()
        if key in new_state and value.shape == new_state[key].shape
    }
    if len(common) < 40:
        raise RuntimeError(f"only {len(common)} historical common-trunk weights could be mapped")
    incompatible = candidate.load_state_dict(common, strict=False)
    candidate.to(device).eval()
    with torch.no_grad():
        z_candidate, _, _, aux_candidate, _ = candidate(
            x_cpu.to(device), edge_cpu.to(device)
        )
        z_candidate = z_candidate.detach().cpu()
        aux_candidate = aux_candidate.detach().cpu()
    absolute = (z_candidate - z_reference).abs()
    relative = absolute / z_reference.abs().clamp_min(1e-8)
    max_abs = float(absolute.max())
    max_rel = float(relative.max())
    aux_abs = float((aux_candidate - aux_reference).abs().max())
    passed = torch.allclose(z_candidate, z_reference, atol=2e-6, rtol=2e-5) and torch.allclose(
        aux_candidate, aux_reference, atol=2e-6, rtol=2e-5
    )
    result = {
        "dataset": "Movies",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "device": device,
        "nodes": data_info["num_nodes"],
        "directed_physical_edges": int(edge_cpu.size(1)),
        "mapped_common_weight_count": len(common),
        "missing_v7_state_keys_after_map": list(incompatible.missing_keys),
        "unexpected_v4_state_keys_after_map": list(incompatible.unexpected_keys),
        "max_absolute_z_error": max_abs,
        "max_relative_z_error": max_rel,
        "max_aux_loss_error": aux_abs,
        "tolerance": {"atol": 2e-6, "rtol": 2e-5},
        "passed": bool(passed),
        "recorded_at_utc": now(),
    }
    write_json(DATA_ROOT / "raw_regression_summary.json", result)
    if not passed:
        raise RuntimeError(f"V4A R0 historical-weight regression failed: {result}")
    print(f"[V7A raw regression] Movies max_abs={max_abs:.3g}, max_rel={max_rel:.3g}", flush=True)
    return result


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
        cfg = _experiment_cfg(dataset, 42, "a2_conditional_dual")
        cfg.model.semantic_cache_path = str(ROOT / _candidate_entry(dataset)["cache_path"])
        cfg.model.semantic_feature_fingerprint = _candidate_entry(dataset)["feature_fingerprint"]
        cfg.model.semantic_candidate_fingerprint = _candidate_entry(dataset)["candidate_fingerprint"]
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
                    "model_parameters": sum(p.numel() for p in model.parameters()),
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
            f"[V7A preflight] {dataset}: {row['status']} "
            f"peak_alloc={row.get('cuda_peak_allocated_bytes')}",
            flush=True,
        )
        del model, classifier, data, x_all, edge_index, train_idx, labels_train
        torch.cuda.empty_cache()
        if not row.get("passed", False):
            summary["failure"] = {"dataset": dataset, "reason": row.get("error", row["status"])}
            write_json(DATA_ROOT / "preflight_summary.json", summary)
            raise RuntimeError(f"V7A preflight failed on {dataset}: {summary['failure']['reason']}")
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


def validate_run(metrics_path: Path, checkpoint_path: Path, variant: str, dataset: str) -> dict[str, Any]:
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
    candidate = _candidate_entry(dataset)
    expected_fp = candidate["candidate_fingerprint"] if variant != "a0_raw" else "0" * 64
    observed_bytes = checkpoint["model_state"]["semantic_fingerprint_state"].tolist()
    observed_fp = bytes(observed_bytes).hex()
    if observed_fp != expected_fp:
        raise RuntimeError(f"checkpoint candidate fingerprint mismatch: {checkpoint_path}")
    expected_feature_fp = (
        _candidate_entry(dataset)["feature_fingerprint"] if variant != "a0_raw" else "0" * 64
    )
    observed_feature_fp = bytes(
        checkpoint["model_state"]["semantic_feature_fingerprint_state"].tolist()
    ).hex()
    if observed_feature_fp != expected_feature_fp:
        raise RuntimeError(f"checkpoint feature fingerprint mismatch: {checkpoint_path}")
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
        checked = validate_run(metrics_path, checkpoint_path, variant, dataset)
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
            "model=psce_mag_v7a",
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
        if variant != "a0_raw":
            candidate = _candidate_entry(dataset)
            command.extend(
                [
                    f"model.semantic_cache_path={ROOT / candidate['cache_path']}",
                    f"model.semantic_feature_fingerprint={candidate['feature_fingerprint']}",
                    f"model.semantic_candidate_fingerprint={candidate['candidate_fingerprint']}",
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
        print(f"[V7A {mode}] {dataset} seed={seed} {variant} device={device}", flush=True)
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
        checked = validate_run(metrics_path, checkpoint_path, variant, dataset)
        status = "completed"
    metadata = checked["metadata"]
    return {
        "status": status,
        "mode": mode,
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
        "semantic_diagnostics_present": (
            variant == "a0_raw" or row.get("semantic_checkpoint_diagnostics") is not None
        ),
        "candidate_fingerprint": (
            None if variant == "a0_raw" else _candidate_entry(dataset)["candidate_fingerprint"]
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
                "expected_runs": 3,
                "completed_runs": len(rows),
                "failures": failures,
                "rows": rows,
                "provenance": prov,
                "updated_at_utc": now(),
            },
        )
        print(f"[V7A smoke] progress={len(rows)}/3 failures={len(failures)}", flush=True)
        if failures:
            raise RuntimeError(f"V7A smoke failed: {failures[-1]}")
    passed = len(rows) == 3 and all(
        row["audit"]["validation_metrics_finite"]
        and row["audit"]["test_metrics_absent"]
        and row["audit"]["semantic_diagnostics_present"]
        for row in rows
    )
    final = read_json(DATA_ROOT / "smoke_summary.json")
    final["audit_passed"] = passed
    final["checks"] = {
        "all_variants_complete": len(rows) == 3,
        "validation_metrics_finite": passed,
        "test_metrics_absent": passed,
        "no_failures": not failures,
    }
    final["completed_at_utc"] = now()
    write_json(DATA_ROOT / "smoke_summary.json", final)
    if not passed:
        raise RuntimeError("V7A smoke audit failed")


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
    for label, item in (("preflight", preflight), ("smoke", smoke), ("tests", tests)):
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
                            "expected_runs": 27,
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
                        "expected_runs": 27,
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
                print(f"[V7A campaign] progress={len(completed)}/27", flush=True)
    if len(completed) != 27 or any(not item.get("resolved", False) for item in failures):
        raise RuntimeError(f"campaign incomplete: completed={len(completed)}/27")
    if any(
        any(str(key).lower().startswith("test") for key in row["metrics"])
        for row in rows
    ):
        raise RuntimeError("Test metric key found in formal campaign rows")
    subprocess.run(
        [sys.executable, "scripts/analyze_psce_mag_v7a.py"], cwd=ROOT, check=True
    )
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "status": "complete",
            "completed_runs": 27,
            "completed_at_utc": now(),
            "test_metric_keys_present": False,
            "nvidia_smi_at_completion": _nvidia_smi(),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        }
    )
    write_json(manifest_path, manifest)
    print("[V7A campaign] all 27 validation-only runs completed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("build-candidates", "raw-regression", "tests", "preflight", "smoke", "campaign", "resume"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    args = parser.parse_args()
    if args.mode == "build-candidates":
        build_candidates(tuple(args.dataset) if args.dataset else DATASETS)
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
