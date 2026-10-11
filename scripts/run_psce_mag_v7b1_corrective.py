#!/usr/bin/env python3
"""Audit, preflight, and run the corrected V7B.1 validation-only NC campaign."""

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

BRANCH = "exp/v7b1_corrective_replication"
BASE_SHA = "75157f052a50c60b791566b2c91207e372956d7b"
PROTOCOL = "unified_full_graph_nc_v1"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "r0_raw",
    "b0_v7a_original",
    "b1_fixed_knn",
    "b2_random_fixed",
    "b3_relative_shared",
    "b4_relative_modality",
)
FREEZE_MESSAGE = "Freeze V7B.1 corrective replication and training parity checks"
OUTPUT_ROOT = ROOT / "outputs/psce_mag_v7b1_corrective"
RESEARCH_ROOT = ROOT / "research/psce_mag_v7b1_corrective"
DATA_ROOT = RESEARCH_ROOT / "data"
FORMAL_ARTIFACTS = {
    "REPORT.md",
    "README.md",
    "data/run_rows.json",
    "data/failures.json",
    "data/summary.csv",
    "data/paired_comparisons.csv",
    "data/semantic_diagnostics.csv",
    "data/resource_profile.json",
    "data/resource_profile.csv",
    "data/campaign_manifest.json",
    "data/random_graph_summary.json",
}
FINGERPRINT_FILES = (
    "configs/config.yaml",
    "configs/model/psce_mag_v7a.yaml",
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
    "scripts/run_psce_mag_v7b1_corrective.py",
    "scripts/analyze_psce_mag_v7b1_corrective.py",
    "tests/test_psce_mag_v7a.py",
    "tests/test_psce_mag_v7b.py",
    "tests/test_psce_mag_v7b1_corrective.py",
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
            allowed = {f"research/psce_mag_v7b1_corrective/{path}" for path in FORMAL_ARTIFACTS}
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
            "model": OmegaConf.load(
                ROOT
                / (
                    "configs/model/psce_mag_v7a.yaml"
                    if variant == "r0_raw"
                    else "configs/model/psce_mag_v7b.yaml"
                )
            ),
            "seed": int(seed),
            "num_runs": 1,
            "device": "cpu",
        }
    )
    cfg.model.variant = "a0_raw" if variant == "r0_raw" else variant
    cfg.task.evaluate_test = False
    cfg.task.training_mode = "full_graph"
    OmegaConf.resolve(cfg)
    return cfg


def _candidate_entry(dataset: str, variant: str | None = None, seed: int | None = None) -> dict[str, Any]:
    if variant == "r0_raw":
        return {
            "dataset": dataset,
            "semantic_candidate_required": False,
            "variant": "a0_raw",
        }
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


def audit_existing_caches() -> dict[str, Any]:
    """Verify and reuse the V7B cache tensors without regenerating topology."""
    import numpy as np
    import torch

    from src.data.semantic_candidates import (
        _json_fingerprint,
        _source_descriptor,
        load_semantic_candidate_cache,
    )
    from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b

    previous_root = ROOT / "research/psce_mag_v7b/data"
    candidate_source = read_json(previous_root / "candidate_graph_summary.json")
    random_source = read_json(previous_root / "random_graph_summary.json")
    v7a_source = read_json(ROOT / "research/psce_mag_v7a/data/candidate_graph_summary.json")
    v7a_rows = {item["dataset"]: item for item in v7a_source["datasets"]}
    candidate_rows = []
    random_rows = []
    graph_stats = []
    physical_codes_by_dataset: dict[str, torch.Tensor] = {}

    for item in candidate_source["datasets"]:
        dataset = item["dataset"]
        if dataset not in DATASETS:
            continue
        text, visual, physical, feature_sources, info = load_candidate_inputs(dataset)
        text = np.ascontiguousarray(text, dtype=np.float32)
        visual = np.ascontiguousarray(visual, dtype=np.float32)
        source_identity = {
            "dataset": dataset,
            "num_nodes": int(text.shape[0]),
            "text_dim": int(text.shape[1]),
            "visual_dim": int(visual.shape[1]),
            "text": _source_descriptor(feature_sources["text"], text),
            "visual": _source_descriptor(feature_sources["visual"], visual),
        }
        actual_feature_fp = _json_fingerprint(source_identity)
        cache_path = ROOT / item["cache_path"]
        cached = load_semantic_candidate_cache_v7b(
            cache_path,
            expected_dataset=dataset,
            expected_feature_fingerprint=actual_feature_fp,
            expected_candidate_fingerprint=item["candidate_fingerprint"],
            expected_v7b_fingerprint=item["v7b_fingerprint"],
        )
        v7a_item = v7a_rows[dataset]
        legacy = load_semantic_candidate_cache(
            ROOT / v7a_item["cache_path"],
            expected_dataset=dataset,
            expected_feature_fingerprint=actual_feature_fp,
            expected_candidate_fingerprint=item["candidate_fingerprint"],
        )
        exact = {
            key: bool(torch.equal(cached[key], legacy[key]))
            for key in ("edge_index", "unique_edge_index", "degree")
        }
        if not all(exact.values()):
            raise RuntimeError(f"{dataset}: reused V7B KNN tensors do not exactly match V7A: {exact}")
        flags = cached["source_flags"]
        flag_counts = {
            "text_only": int(((flags[:, 0] > 0) & (flags[:, 1] == 0)).sum()),
            "visual_only": int(((flags[:, 0] == 0) & (flags[:, 1] > 0)).sum()),
            "both": int(((flags[:, 0] > 0) & (flags[:, 1] > 0)).sum()),
        }
        if flag_counts != {key: int(item["source_counts"][key]) for key in flag_counts}:
            raise RuntimeError(f"{dataset}: raw source-flag counts disagree with summary")
        unique = cached["unique_edge_index"]
        physical_unique = torch.sort(physical, dim=0).values
        physical_unique = physical_unique[:, physical_unique[0] < physical_unique[1]
        ]
        physical_codes = torch.unique(physical_unique[0] * info["num_nodes"] + physical_unique[1])
        physical_codes_by_dataset[dataset] = physical_codes
        candidate_codes = unique[0] * info["num_nodes"] + unique[1]
        overlap = int(torch.isin(candidate_codes, physical_codes).sum())
        overlap_ratio = overlap / max(int(unique.size(1)), 1)
        if not math.isclose(overlap_ratio, float(item["candidate_physical_overlap_ratio"]), abs_tol=1e-12):
            raise RuntimeError(f"{dataset}: physical overlap ratio changed")
        row = {
            "dataset": dataset,
            "feature_fingerprint_expected": item["feature_fingerprint"],
            "feature_fingerprint_recomputed": actual_feature_fp,
            "candidate_fingerprint": item["candidate_fingerprint"],
            "v7b_source_fingerprint": item["v7b_fingerprint"],
            "candidate_tensors_equal_to_v7a": exact,
            "source_flag_counts_recomputed": flag_counts,
            "undirected_edges": int(unique.size(1)),
            "directed_edges": int(cached["edge_index"].size(1)),
            "self_loops": int((cached["edge_index"][0] == cached["edge_index"][1]).sum()),
            "duplicate_undirected_edges": int(unique.size(1) - torch.unique(candidate_codes).numel()),
            "physical_overlap_count": overlap,
            "physical_overlap_ratio": overlap_ratio,
            "cache_contains_labels": bool(cached["metadata"].get("cache_contains_labels", False)),
            "candidate_phase_label_access": bool(item.get("candidate_phase_label_access", False)),
            "cache_path_reused": str(cache_path.relative_to(ROOT)),
        }
        if row["self_loops"] or row["duplicate_undirected_edges"] or row["cache_contains_labels"] or row["candidate_phase_label_access"]:
            raise RuntimeError(f"{dataset}: candidate cache integrity/privacy audit failed: {row}")
        candidate_rows.append(row)
        candidate_copy = dict(item)
        candidate_copy.update(
            {
                **info,
                "feature_fingerprint_recomputed": actual_feature_fp,
                "tensor_equal_to_v7a": exact,
                "source_counts_recomputed": flag_counts,
                "cache_path": str(cache_path.relative_to(ROOT)),
                "cache_contains_labels": False,
                "candidate_phase_label_access": False,
            }
        )
        graph_stats.append(candidate_copy)

    random_fingerprints: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    expected_topology_seed = {42: 2026, 43: 2027, 44: 2028}
    for item in random_source["graphs"]:
        dataset = item["dataset"]
        if dataset not in DATASETS:
            continue
        base_item = next(row for row in graph_stats if row["dataset"] == dataset)
        base = load_semantic_candidate_cache_v7b(
            ROOT / base_item["cache_path"], expected_dataset=dataset
        )
        path = ROOT / item["cache_path"]
        randomized = load_semantic_candidate_cache_v7b(
            path,
            expected_dataset=dataset,
            expected_feature_fingerprint=base_item["feature_fingerprint"],
            expected_candidate_fingerprint=item["candidate_fingerprint"],
            expected_v7b_fingerprint=item["v7b_fingerprint"],
        )
        n = int(randomized["metadata"]["num_nodes"])
        unique = randomized["unique_edge_index"]
        codes = unique[0] * n + unique[1]
        train_seed = int(item["training_seed"])
        topology_seed = int(item["seed"])
        if expected_topology_seed.get(train_seed) != topology_seed:
            raise RuntimeError(f"{dataset}: B2 seed map mismatch for training seed {train_seed}")
        if (
            not torch.equal(randomized["degree"], base["degree"])
            or unique.size(1) != base["unique_edge_index"].size(1)
            or torch.any(unique[0] == unique[1])
            or torch.unique(codes).numel() != codes.numel()
            or not torch.equal(randomized["edge_index"][:, : unique.size(1)], unique)
            or not torch.equal(randomized["edge_index"][:, unique.size(1) :], unique.flip(0))
            or bool(randomized["source_flags"].any())
        ):
            raise RuntimeError(f"{dataset} seed {item['training_seed']}: B2 topology audit failed")
        if item["source_candidate_fingerprint"] != base_item["candidate_fingerprint"]:
            raise RuntimeError(f"{dataset}: B2 source KNN fingerprint mismatch")
        physical_overlap_count = int(torch.isin(codes, physical_codes_by_dataset[dataset]).sum())
        physical_overlap_ratio = physical_overlap_count / max(int(unique.size(1)), 1)
        historical_overlap = item.get("physical_overlap_audit", {})
        if historical_overlap and (
            physical_overlap_count != int(historical_overlap.get("random_candidate_physical_overlap_count", -1))
            or not math.isclose(physical_overlap_ratio, float(historical_overlap.get("random_candidate_physical_overlap_ratio", -1)), abs_tol=1e-12)
        ):
            raise RuntimeError(f"{dataset} seed {train_seed}: recomputed B2/physical overlap differs from frozen audit")
        random_fingerprints[dataset].add(item["candidate_fingerprint"])
        random_rows.append(
            {
                "dataset": dataset,
                "training_seed": train_seed,
                "topology_seed": topology_seed,
                "candidate_fingerprint": item["candidate_fingerprint"],
                "v7b_fingerprint": item["v7b_fingerprint"],
                "source_candidate_fingerprint": item["source_candidate_fingerprint"],
                "edge_count": int(unique.size(1)),
                "degree_sequence_exact": True,
                "self_loops": 0,
                "duplicate_edges": 0,
                "physical_overlap_audit": {
                    **historical_overlap,
                    "random_candidate_physical_overlap_count_recomputed": physical_overlap_count,
                    "random_candidate_physical_overlap_ratio_recomputed": physical_overlap_ratio,
                },
                "cache_path_reused": str(path.relative_to(ROOT)),
            }
        )
    if any(len(random_fingerprints[dataset]) != 3 for dataset in DATASETS):
        raise RuntimeError("B2 cached topology fingerprints do not cover three distinct seed maps")

    provenance_record = provenance("audit")
    candidate_summary = dict(candidate_source)
    candidate_summary.update(
        {
            "datasets": graph_stats,
            "provenance": provenance_record,
            "cache_policy": "reused V7B caches; no topology regeneration",
            "audited_at_utc": now(),
        }
    )
    random_summary = dict(random_source)
    random_summary.update(
        {
            "graphs": [
                {
                    **old,
                    "cache_path": str((ROOT / old["cache_path"]).relative_to(ROOT)),
                }
                for old in random_source["graphs"]
            ],
            "provenance": provenance_record,
            "cache_policy": "reused exact V7B seed-mapped topology caches",
            "audited_at_utc": now(),
        }
    )
    write_json(DATA_ROOT / "candidate_graph_summary.json", candidate_summary)
    write_json(DATA_ROOT / "random_graph_summary.json", random_summary)
    audit = {
        "passed": True,
        "datasets": candidate_rows,
        "random_graphs": random_rows,
        "training_seed_to_topology_seed": {"42": 2026, "43": 2027, "44": 2028},
        "label_access_during_candidate_audit": False,
        "provenance": provenance_record,
        "completed_at_utc": now(),
    }
    write_json(DATA_ROOT / "candidate_cache_audit.json", audit)
    write_json(DATA_ROOT / "random_graph_audit.json", {"passed": True, "graphs": random_rows, "provenance": provenance_record})
    return audit


def audit_parameters() -> dict[str, Any]:
    from omegaconf import OmegaConf
    from src.models.psce_mag_v7a import Model as V7AModel
    from src.models.psce_mag_v7b import Model as V7BModel

    movie = next(
        item for item in read_json(DATA_ROOT / "candidate_graph_summary.json")["datasets"]
        if item["dataset"] == "Movies"
    )
    info = {
        "input_dim": int(movie["input_dim"]),
        "text_dim": int(movie["text_dim"]),
        "visual_dim": int(movie["visual_dim"]),
        "num_nodes": int(movie["num_nodes"]),
        "num_classes": 20,
    }
    cfg_a = OmegaConf.create({"model": OmegaConf.load(ROOT / "configs/model/psce_mag_v7a.yaml")})
    cfg_a.model.variant = "a2_conditional_dual"
    a2 = V7AModel(cfg_a, info)
    cfg_b = OmegaConf.create({"model": OmegaConf.load(ROOT / "configs/model/psce_mag_v7b.yaml")})
    all_models: dict[str, Any] = {}
    models = {"a2_conditional_dual": a2}
    for variant in ("b0_v7a_original", "b1_fixed_knn", "b2_random_fixed", "b3_relative_shared", "b4_relative_modality"):
        variant_cfg = OmegaConf.create({"model": OmegaConf.create(OmegaConf.to_container(cfg_b.model, resolve=True))})
        variant_cfg.model.variant = variant
        models[variant] = V7BModel(variant_cfg, info)
    a2_params = dict(a2.named_parameters())
    for variant, model in models.items():
        params = dict(model.named_parameters())
        if variant in {"b0_v7a_original", "b1_fixed_knn", "b2_random_fixed"}:
            if params.keys() != a2_params.keys() or any(params[name].shape != a2_params[name].shape for name in a2_params):
                raise RuntimeError(f"{variant}: public parameter names/shapes differ from V7A A2")
            public_match = True
        else:
            shared_names = {
                name for name in params.keys() & a2_params.keys()
                if not name.startswith("relation_scorer.")
            }
            public_match = (
                {name for name in params if not name.startswith("relation_scorer.")}
                == {name for name in a2_params if not name.startswith("relation_scorer.")}
                and all(params[name].shape == a2_params[name].shape for name in shared_names)
            )
            if not public_match:
                raise RuntimeError(f"{variant}: shared trunk parameter names/shapes changed")
        all_models[variant] = {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "relation_trainability": {
                name: bool(parameter.requires_grad)
                for name, parameter in params.items()
                if name.startswith(("relation_projections.", "relation_scorer."))
            },
            "shared_trunk_parameter_names_shapes_match_v7a_a2": public_match,
            "relation_scorer_parameter_shapes": {name: list(parameter.shape) for name, parameter in params.items() if name.startswith("relation_scorer.")},
        }
    b0 = models["b0_v7a_original"]
    old_b0_trainable = all_models["b0_v7a_original"]["trainable_parameter_count"] - sum(
        p.numel() for name, p in b0.named_parameters() if name.startswith(("relation_projections.", "relation_scorer."))
    )
    delta = all_models["b0_v7a_original"]["trainable_parameter_count"] - old_b0_trainable
    if delta != 24_705 or all_models["b0_v7a_original"]["trainable_parameter_count"] != all_models["a2_conditional_dual"]["trainable_parameter_count"]:
        raise RuntimeError(f"unexpected corrected B0 trainable parameter delta: {delta}")
    result = {
        "passed": True,
        "models": all_models,
        "old_buggy_b0_trainable_parameter_count": old_b0_trainable,
        "corrected_b0_trainable_parameter_count": all_models["b0_v7a_original"]["trainable_parameter_count"],
        "trainable_parameter_delta": delta,
        "relation_parameter_count": delta,
        "v7a_a2_b0_parameter_names_shapes_equal": True,
        "provenance": provenance("audit"),
        "completed_at_utc": now(),
    }
    write_json(DATA_ROOT / "parameter_grad_audit.json", result)
    return result


def run_training_parity(device: str) -> dict[str, Any]:
    """Compare V7A A2 and repaired B0 through a real Movies train step."""
    import torch
    import torch.nn as nn
    from src.data import load_mag_data
    from src.models.psce_mag_v7a import Model as V7AModel
    from src.models.psce_mag_v7b import Model as V7BModel
    from src.tasks.common import build_optimizer
    from src.utils.seeds import set_seed

    prov = provenance("training-parity")
    device_obj = torch.device(device)
    env = environment(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-graph training parity requires CUDA")
    torch.cuda.set_device(device_obj)
    cfg_a = _experiment_cfg("Movies", 42, "b0_v7a_original")
    cfg_b = _experiment_cfg("Movies", 42, "b0_v7a_original")
    cfg_a.model.name = "psce_mag_v7a"
    cfg_a.model.variant = "a2_conditional_dual"
    data = load_mag_data(cfg_a, "nc", 42)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }
    candidate = _candidate_entry("Movies", "b0_v7a_original", 42)
    for cfg in (cfg_a, cfg_b):
        cfg.model.semantic_cache_path = str(ROOT / candidate["cache_path"])
        cfg.model.semantic_feature_fingerprint = candidate["feature_fingerprint"]
        cfg.model.semantic_candidate_fingerprint = candidate["candidate_fingerprint"]
        if "semantic_v7b_fingerprint" in cfg.model:
            cfg.model.semantic_v7b_fingerprint = candidate["v7b_fingerprint"]

    set_seed(91042)
    reference = V7AModel(cfg_a, info)
    head_reference = nn.Linear(reference.out_dim, int(data.num_classes))
    set_seed(91042)
    candidate_model = V7BModel(cfg_b, info)
    head_candidate = nn.Linear(candidate_model.out_dim, int(data.num_classes))
    reference.load_semantic_cache(
        cfg_a.model.semantic_cache_path,
        expected_dataset="Movies",
        expected_feature_fingerprint=candidate["feature_fingerprint"],
        expected_candidate_fingerprint=candidate["candidate_fingerprint"],
    )
    candidate_model.load_semantic_cache(
        cfg_b.model.semantic_cache_path,
        expected_dataset="Movies",
        expected_feature_fingerprint=candidate["feature_fingerprint"],
        expected_candidate_fingerprint=candidate["candidate_fingerprint"],
        expected_v7b_fingerprint=candidate["v7b_fingerprint"],
    )
    init_param_diff = max(
        float((left - dict(candidate_model.named_parameters())[name].detach().cpu()).abs().max())
        for name, left in reference.named_parameters()
    )
    head_init_diff = max(
        float((left - right).abs().max())
        for left, right in zip(head_reference.parameters(), head_candidate.parameters())
    )
    reference_names = {name for name, parameter in reference.named_parameters() if parameter.requires_grad}
    candidate_names = {name for name, parameter in candidate_model.named_parameters() if parameter.requires_grad}
    trainable_set_delta = {
        "v7a_only": sorted(reference_names - candidate_names),
        "b0_only": sorted(candidate_names - reference_names),
    }
    reference = reference.to(device_obj).train()
    candidate_model = candidate_model.to(device_obj).train()
    head_reference = head_reference.to(device_obj).train()
    head_candidate = head_candidate.to(device_obj).train()
    reference.profile_forward = candidate_model.profile_forward = True
    x = data.x.to(device_obj)
    edge = data.edge_index.to(device_obj)
    train_idx = data.train_idx.to(device_obj)
    labels = data.y[data.train_idx].to(device_obj)
    optimizer_a = build_optimizer(list(reference.parameters()) + list(head_reference.parameters()), cfg_a, model=reference)
    optimizer_b = build_optimizer(list(candidate_model.parameters()) + list(head_candidate.parameters()), cfg_b, model=candidate_model)
    for left, right in zip(optimizer_a.param_groups, optimizer_b.param_groups):
        if {k: v for k, v in left.items() if k != "params"} != {k: v for k, v in right.items() if k != "params"}:
            raise RuntimeError("V7A/B0 optimizer hyperparameters differ")

    def forward_loss(model, head):
        z, _, _, aux, _ = model(x, edge)
        logits = head(z.index_select(0, train_idx))
        ce = nn.functional.cross_entropy(logits, labels)
        return z, logits, ce, aux, ce + float(cfg_a.task.loss.aux_weight) * aux

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device_obj)
    optimizer_a.zero_grad(set_to_none=True)
    optimizer_b.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device_obj)
    old = forward_loss(reference, head_reference)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device_obj)
    new = forward_loss(candidate_model, head_candidate)
    embedding_error = float((old[0] - new[0]).abs().max().detach().cpu())
    logits_error = float((old[1] - new[1]).abs().max().detach().cpu())
    ce_error = abs(float(old[2].detach().cpu()) - float(new[2].detach().cpu()))
    auxiliary_error = abs(float(old[3].detach().cpu()) - float(new[3].detach().cpu()))
    total_loss_error = abs(float(old[4].detach().cpu()) - float(new[4].detach().detach().cpu()))
    old[4].backward()
    new[4].backward()
    gradients_a, gradients_b = dict(reference.named_parameters()), dict(candidate_model.named_parameters())
    gradient_errors = []
    for name in gradients_a:
        left, right = gradients_a[name].grad, gradients_b[name].grad
        if (left is None) != (right is None):
            raise RuntimeError(f"gradient presence mismatch for {name}")
        if left is not None:
            gradient_errors.append(float((left - right).abs().max().detach().cpu()))
    for name in ("weight", "bias"):
        left = getattr(head_reference, name).grad
        right = getattr(head_candidate, name).grad
        if (left is None) != (right is None):
            raise RuntimeError(f"classifier gradient presence mismatch for {name}")
        if left is not None:
            gradient_errors.append(float((left - right).abs().max().detach().cpu()))
    gradient_error = max(gradient_errors, default=0.0)
    relation_names = [name for name in gradients_b if name.startswith(("relation_projections.", "relation_scorer."))]
    relation_gradients = {
        name: {
            "finite": bool(gradients_b[name].grad is not None and torch.isfinite(gradients_b[name].grad).all()),
            "norm": float(gradients_b[name].grad.detach().float().norm().cpu()) if gradients_b[name].grad is not None else 0.0,
            "nonzero": bool(gradients_b[name].grad is not None and torch.count_nonzero(gradients_b[name].grad).item() > 0),
        }
        for name in relation_names
    }
    for name, diagnostic in relation_gradients.items():
        if not diagnostic["finite"] or not diagnostic["nonzero"]:
            raise RuntimeError(f"real Movies B0 relation gradient is absent/zero: {name} {diagnostic}")
    grad_diag = candidate_model.capture_gradient_diagnostics()
    before = {name: parameter.detach().clone() for name, parameter in candidate_model.named_parameters() if name.startswith(("relation_projections.", "relation_scorer."))}
    torch.nn.utils.clip_grad_norm_(list(reference.parameters()) + list(head_reference.parameters()), float(cfg_a.task.grad_clip), error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(list(candidate_model.parameters()) + list(head_candidate.parameters()), float(cfg_b.task.grad_clip), error_if_nonfinite=True)
    optimizer_a.step(); optimizer_b.step()
    update_diffs = [
        float((left.detach() - dict(candidate_model.named_parameters())[name].detach()).abs().max().cpu())
        for name, left in reference.named_parameters()
    ]
    update_diffs.extend(float((left.detach() - right.detach()).abs().max().cpu()) for left, right in zip(head_reference.parameters(), head_candidate.parameters()))
    relation_updates = {
        name: {
            "l1_update": float((dict(candidate_model.named_parameters())[name].detach() - before[name]).abs().sum().cpu()),
            "max_abs_update": float((dict(candidate_model.named_parameters())[name].detach() - before[name]).abs().max().cpu()),
        }
        for name in before
    }
    passed = (
        init_param_diff == 0.0
        and head_init_diff == 0.0
        and not trainable_set_delta["v7a_only"]
        and not trainable_set_delta["b0_only"]
        and embedding_error <= 2e-5
        and logits_error <= 2e-5
        and total_loss_error <= 2e-6
        and gradient_error <= 2e-5
        and max(update_diffs, default=0.0) <= 2e-5
        and all(value["l1_update"] > 0 for value in relation_updates.values())
    )
    summary = {
        "passed": passed,
        "regression": "V7A A2 vs corrected V7B B0 real Movies full-graph train-mode forward/backward/AdamW update",
        "dataset": "Movies",
        "num_nodes": int(data.num_nodes),
        "train_nodes": int(data.train_idx.numel()),
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "source_fingerprint": candidate["v7b_fingerprint"],
        "same_initialization": {"model_max_abs_difference": init_param_diff, "classifier_max_abs_difference": head_init_diff},
        "trainable_parameter_set_difference": trainable_set_delta,
        "trainable_parameters": int(sum(p.numel() for p in candidate_model.parameters() if p.requires_grad)),
        "embedding_max_abs_error": embedding_error,
        "logits_max_abs_error": logits_error,
        "cross_entropy_abs_error": ce_error,
        "auxiliary_loss_abs_error": auxiliary_error,
        "total_loss_abs_error": total_loss_error,
        "public_gradient_max_abs_error": gradient_error,
        "relation_gradients": relation_gradients,
        "gradient_diagnostics": grad_diag,
        "adamw_update_max_abs_difference": max(update_diffs, default=0.0),
        "relation_parameter_updates": relation_updates,
        "tolerance": {"atol": 2e-5, "rtol": 2e-5, "loss_atol": 2e-6},
        "device": device,
        "gpu": torch.cuda.get_device_name(device_obj),
        "test_metrics_generated": False,
        "provenance": prov,
        "recorded_at_utc": now(),
    }
    write_json(DATA_ROOT / "training_parity_audit.json", summary)
    if not passed:
        raise RuntimeError(f"real-graph V7A/B0 train parity failed: {summary}")
    return summary

def run_preflight_v71(device: str) -> None:
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
    summary = {
        "protocol": PROTOCOL,
        "task_evaluate_test": False,
        "validation_only": True,
        "preflight_variants": ["r0_raw", "b0_v7a_original", "b3_relative_shared", "b4_relative_modality"],
        "device": device,
        "datasets": [],
        "environment": env,
        "passed": False,
        "provenance": prov,
    }
    write_json(DATA_ROOT / "environment.json", env)
    write_json(DATA_ROOT / "preflight_summary.json", summary)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V7B.1 GPU preflight requires an available CUDA device")
    torch.cuda.set_device(device_obj)
    variants = summary["preflight_variants"]
    for dataset in DATASETS:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_obj)
        if free_bytes < 2 * 1024**3:
            raise RuntimeError(f"{dataset}: less than 2 GiB free on target GPU; no process was stopped")
        dataset_cfg = _experiment_cfg(dataset, 42, "r0_raw")
        data = load_mag_data(dataset_cfg, "nc", 42)
        info = {
            "input_dim": data.input_dim,
            "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1)),
        }
        for variant in variants:
            row: dict[str, Any] = {
                "dataset": dataset,
                "variant": variant,
                "status": "running",
                "gpu_snapshot_before": _gpu_query(),
                "cuda_free_before_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
                **info,
                "train_nodes": int(data.train_idx.numel()),
                "validation_nodes": int(data.val_idx.numel()),
                "test_metrics_generated": False,
            }
            started = time.perf_counter()
            cfg = _experiment_cfg(dataset, 42, variant)
            cache_entry = _candidate_entry(dataset, variant, 42)
            if variant != "r0_raw":
                cfg.model.semantic_cache_path = str(ROOT / cache_entry["cache_path"])
                cfg.model.semantic_feature_fingerprint = cache_entry["feature_fingerprint"]
                cfg.model.semantic_candidate_fingerprint = cache_entry["candidate_fingerprint"]
                cfg.model.semantic_v7b_fingerprint = cache_entry["v7b_fingerprint"]
                row["candidate_fingerprint"] = cache_entry["candidate_fingerprint"]
            set_seed(42042)
            model = build_model(cfg, info)
            if variant != "r0_raw":
                model.load_semantic_cache(
                    str(cfg.model.semantic_cache_path),
                    expected_dataset=dataset,
                    expected_feature_fingerprint=cache_entry["feature_fingerprint"],
                    expected_candidate_fingerprint=cache_entry["candidate_fingerprint"],
                    expected_v7b_fingerprint=cache_entry["v7b_fingerprint"],
                )
            model = model.to(device_obj).train()
            head = nn.Linear(model.out_dim, int(data.num_classes)).to(device_obj)
            model.profile_forward = True
            optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), cfg, model=model)
            x_all, edge_index = data.x.to(device_obj), data.edge_index.to(device_obj)
            train_idx = data.train_idx.to(device_obj)
            labels_train = data.y[data.train_idx].to(device_obj)
            tracked = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
                if name.startswith(("relation_projections.", "relation_scorer."))
            }
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device_obj)
            train_started = time.perf_counter()
            try:
                z, _, _, aux_loss, _ = model(x_all, edge_index)
                logits = head(z.index_select(0, train_idx))
                loss = nn.functional.cross_entropy(logits, labels_train) + float(cfg.task.loss.aux_weight) * aux_loss
                loss.backward()
                gradients = {
                    name: {
                        "present": parameter.grad is not None,
                        "finite": bool(parameter.grad is None or torch.isfinite(parameter.grad).all()),
                        "norm": float(parameter.grad.detach().float().norm().cpu()) if parameter.grad is not None else 0.0,
                        "nonzero": bool(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0),
                    }
                    for name, parameter in model.named_parameters()
                    if name.startswith(("relation_projections.", "relation_scorer."))
                }
                gradient_diagnostics = getattr(model, "capture_gradient_diagnostics", lambda: {})()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), float(cfg.task.grad_clip), error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(device_obj)
                train_seconds = time.perf_counter() - train_started
                relation_updates = {
                    name: float((dict(model.named_parameters())[name].detach() - value).abs().sum().cpu())
                    for name, value in tracked.items()
                }
                forward_timing = dict(getattr(model, "_last_timing", {}))
                train_loss = float(loss.detach().cpu())
                del z, logits, loss, aux_loss
                model.eval(); head.eval()
                validation_started = time.perf_counter()
                with torch.no_grad():
                    z_val = model(x_all, edge_index)[0]
                    pred = head(z_val.index_select(0, data.val_idx.to(device_obj))).argmax(-1).cpu()
                torch.cuda.synchronize(device_obj)
                validation_seconds = time.perf_counter() - validation_started
                target = data.y[data.val_idx].cpu()
                validation = {
                    "val_acc": float((pred == target).float().mean()),
                    "val_macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=_resolve_nc_eval_labels(data, include_test=False), average="macro", zero_division=0)),
                }
                if variant in {"b0_v7a_original", "b3_relative_shared", "b4_relative_modality"}:
                    gradients_ok = bool(gradients) and all(item["present"] and item["finite"] and item["nonzero"] for item in gradients.values()) and bool(relation_updates) and all(value > 0 for value in relation_updates.values())
                elif variant in {"b1_fixed_knn", "b2_random_fixed"}:
                    gradients_ok = all(not item["present"] for item in gradients.values()) and all(value == 0.0 for value in relation_updates.values())
                else:
                    gradients_ok = all(not item["present"] for item in gradients.values()) and all(value == 0.0 for value in relation_updates.values())
                finite = bool(torch.isfinite(z_val).all()) and math.isfinite(train_loss) and all(math.isfinite(float(value)) for value in validation.values())
                row.update(
                    {
                        "status": "passed" if finite and gradients_ok else "failed",
                        "passed": finite and gradients_ok,
                        "training_loss": train_loss,
                        "validation_metrics": validation,
                        "training_step_seconds": train_seconds,
                        "validation_inference_seconds": validation_seconds,
                        "relation_scoring_seconds": forward_timing.get("relation_scoring_seconds"),
                        "semantic_propagation_seconds": forward_timing.get("semantic_propagation_seconds"),
                        "model_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                        "classifier_parameters": sum(parameter.numel() for parameter in head.parameters()),
                        "relation_gradients": gradients,
                        "relation_parameter_updates_l1": relation_updates,
                        "gradient_diagnostics": gradient_diagnostics,
                        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
                        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)),
                        "gpu_snapshot_after": _gpu_query(),
                        "wall_seconds": time.perf_counter() - started,
                        "test_metric_keys_present": False,
                    }
                )
                del z_val, pred, model, head, optimizer, x_all, edge_index
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row.update({"status": "oom", "passed": False, "error": str(exc), "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)), "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)), "gpu_snapshot_after": _gpu_query(), "wall_seconds": time.perf_counter() - started})
            summary["datasets"].append(row)
            write_json(DATA_ROOT / "preflight_summary.json", summary)
            print(f"[V7B.1 preflight] {dataset} {variant}: {row['status']} peak_alloc={row.get('cuda_peak_allocated_bytes')}", flush=True)
            if not row.get("passed"):
                summary["failure"] = {"dataset": dataset, "variant": variant, "reason": row.get("error", "nonfinite result or gradient/update audit failure")}
                write_json(DATA_ROOT / "preflight_summary.json", summary)
                raise RuntimeError(f"V7B.1 preflight failed: {summary['failure']}")
        del data
        torch.cuda.empty_cache()
    summary["passed"] = len(summary["datasets"]) == len(DATASETS) * len(variants) and all(row["passed"] for row in summary["datasets"])
    summary["completed_at_utc"] = now()
    write_json(DATA_ROOT / "preflight_summary.json", summary)


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
    expected_model_variant = "a0_raw" if variant == "r0_raw" else variant
    if config_path.is_file():
        resolved_cfg = OmegaConf.load(config_path)
        if bool(resolved_cfg.task.get("evaluate_test", True)):
            raise RuntimeError(f"Hydra config has evaluate_test=true: {config_path}")
        if str(resolved_cfg.model.variant) != expected_model_variant:
            raise RuntimeError(f"wrong model variant in Hydra config: {config_path}")
        expected_model_name = "psce_mag_v7a" if variant == "r0_raw" else "psce_mag_v7b"
        if str(resolved_cfg.model.name) != expected_model_name:
            raise RuntimeError(f"wrong model implementation in Hydra config: {config_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(checkpoint.get("metrics", {})) != {"val_acc", "val_macro_f1"}:
        raise RuntimeError(f"checkpoint metric keys are not validation-only: {checkpoint_path}")
    if checkpoint.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"checkpoint was not selected by Validation Accuracy: {checkpoint_path}")
    if variant != "r0_raw":
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
    if variant == "b0_v7a_original":
        diagnostics = metadata.get("gradient_diagnostics", {})
        relation_keys = [name for name in diagnostics if name.startswith(("relation_projections.", "relation_scorer."))]
        if not relation_keys or any(not diagnostics[name].get("gradient_present") or not diagnostics[name].get("gradient_nonzero") or not diagnostics[name].get("gradient_finite") for name in relation_keys):
            raise RuntimeError(f"corrected B0 checkpoint lacks nonzero relation-gradient evidence: {checkpoint_path}")
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
        model_name = "psce_mag_v7a" if variant == "r0_raw" else "psce_mag_v7b"
        model_variant = "a0_raw" if variant == "r0_raw" else variant
        command = [
            sys.executable,
            "-m",
            "src.main",
            f"dataset={dataset}",
            "task=nc",
            f"model={model_name}",
            f"model.variant={model_variant}",
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
        if variant != "r0_raw":
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
        print(f"[V7B.1 {mode}] {dataset} seed={seed} {variant} device={device}", flush=True)
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
    result = {
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
    if variant != "r0_raw":
        candidate = _candidate_entry(dataset, variant, seed)
        result["candidate_fingerprint"] = candidate["candidate_fingerprint"]
        result["feature_fingerprint"] = candidate["feature_fingerprint"]
        result["v7b_fingerprint"] = candidate["v7b_fingerprint"]
    else:
        result["candidate_fingerprint"] = None
        result["feature_fingerprint"] = None
        result["v7b_fingerprint"] = None
    return result


def _audit_smoke(row: dict[str, Any], dataset: str, variant: str) -> dict[str, Any]:
    metrics = row["metrics"]
    result = {
        "validation_metrics_finite": all(
            math.isfinite(float(metrics[key])) for key in ("val_acc", "val_macro_f1")
        ),
        "test_metrics_absent": not any(key.lower().startswith("test") for key in metrics),
        "best_epoch": row["metadata"].get("best_epoch"),
        "model_parameters": row["metadata"].get("model_parameters"),
        "semantic_diagnostics_present": row.get("semantic_checkpoint_diagnostics") is not None,
    }
    if variant != "r0_raw":
        result["candidate_fingerprint"] = _candidate_entry(dataset, variant, int(row["seed"]))["candidate_fingerprint"]
    gradients = row["metadata"].get("gradient_diagnostics", {})
    relation = {name: value for name, value in gradients.items() if name.startswith(("relation_projections.", "relation_scorer."))}
    if variant == "r0_raw":
        result["raw_relation_gradients_absent"] = all(not value.get("gradient_present") for value in relation.values())
    elif variant == "b0_v7a_original":
        result["b0_relation_gradients_nonzero"] = bool(relation) and all(value.get("gradient_present") and value.get("gradient_nonzero") and value.get("gradient_finite") for value in relation.values())
    elif variant in {"b1_fixed_knn", "b2_random_fixed"}:
        result["fixed_relation_gradients_absent"] = all(not value.get("gradient_present") for value in relation.values())
    else:
        result["relation_gradients_nonzero"] = all(value.get("gradient_nonzero") and value.get("gradient_finite") for value in relation.values())
    return result


def run_smoke(device: str) -> None:
    prov = provenance("smoke")
    preflight = read_json(DATA_ROOT / "preflight_summary.json")
    tests = read_json(DATA_ROOT / "test_summary.json")
    parity = read_json(DATA_ROOT / "training_parity_audit.json")
    if not preflight.get("passed") or not tests.get("passed") or not parity.get("passed"):
        raise RuntimeError("smoke requires passing GPU preflight, real-graph parity, and full pytest")
    rows, failures = [], []
    for variant in VARIANTS:
        try:
            row = run_one("Movies", 42, variant, device=device, mode="smoke", epochs=1, resume=True)
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
        print(f"[V7B.1 smoke] progress={len(rows)}/{len(VARIANTS)} failures={len(failures)}", flush=True)
        if failures:
            raise RuntimeError(f"V7B smoke failed: {failures[-1]}")
    passed = len(rows) == len(VARIANTS) and all(
        row["audit"]["validation_metrics_finite"]
        and row["audit"]["test_metrics_absent"]
        and (row["variant"] == "r0_raw" or row["audit"]["semantic_diagnostics_present"])
        and all(value for key, value in row["audit"].items() if key.endswith("nonzero") or key.endswith("absent"))
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
        raise RuntimeError("V7B.1 smoke audit failed")


def run_campaign(device: str, *, resume: bool) -> None:
    import torch

    mode = "resume" if resume else "campaign"
    prov = provenance(mode, resume=resume)
    preflight = read_json(DATA_ROOT / "preflight_summary.json")
    smoke = read_json(DATA_ROOT / "smoke_summary.json")
    tests = read_json(DATA_ROOT / "test_summary.json")
    parity = read_json(DATA_ROOT / "training_parity_audit.json")
    cache_audit = read_json(DATA_ROOT / "candidate_cache_audit.json")
    parameter_audit = read_json(DATA_ROOT / "parameter_grad_audit.json")
    if not preflight.get("passed") or not smoke.get("audit_passed"):
        raise RuntimeError("campaign requires passing GPU preflight and all-variant smoke")
    if not tests.get("passed") or tests.get("exit_code") != 0:
        raise RuntimeError("campaign requires a passing full repository pytest record")
    if not parity.get("passed") or not cache_audit.get("passed") or not parameter_audit.get("passed"):
        raise RuntimeError("campaign requires passing real-graph parity, parameter audit, and cache audit")
    expected_fp = prov["configuration_fingerprint_sha256"]
    for label, item in (("preflight", preflight), ("smoke", smoke), ("tests", tests), ("training-parity", parity), ("cache-audit", cache_audit), ("parameter-audit", parameter_audit)):
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
                            "expected_runs": 54,
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
                        "expected_runs": 54,
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
                print(f"[V7B.1 campaign] progress={len(completed)}/54", flush=True)
    if len(completed) != 54 or any(not item.get("resolved", False) for item in failures):
        raise RuntimeError(f"campaign incomplete: completed={len(completed)}/54")
    if any(
        any(str(key).lower().startswith("test") for key in row["metrics"])
        for row in rows
    ):
        raise RuntimeError("Test metric key found in formal campaign rows")
    subprocess.run(
        [sys.executable, "scripts/analyze_psce_mag_v7b1_corrective.py"], cwd=ROOT, check=True
    )
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "status": "complete",
            "completed_runs": 54,
            "completed_at_utc": now(),
            "test_metric_keys_present": False,
            "nvidia_smi_at_completion": _nvidia_smi(),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        }
    )
    write_json(manifest_path, manifest)
    print("[V7B.1 campaign] all 54 validation-only runs completed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("audit", "training-parity", "tests", "preflight", "smoke", "campaign", "resume"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    args = parser.parse_args()
    if args.mode == "audit":
        cache = audit_existing_caches()
        parameters = audit_parameters()
        print(f"[V7B.1 audit] cache={cache['passed']} parameters={parameters['passed']}", flush=True)
    elif args.mode == "training-parity":
        run_training_parity(args.device)
    elif args.mode == "tests":
        run_tests()
    elif args.mode == "preflight":
        run_preflight_v71(args.device)
    elif args.mode == "smoke":
        run_smoke(args.device)
    elif args.mode == "campaign":
        run_campaign(args.device, resume=False)
    else:
        run_campaign(args.device, resume=True)


if __name__ == "__main__":
    main()
