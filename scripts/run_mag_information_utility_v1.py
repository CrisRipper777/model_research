#!/usr/bin/env python3
"""Staged runner for MAG Information Utility V1.

The graph-audit stage reads only features and topology. The training stage is
full-graph NC, validation-only, and refuses to run unless an idle CUDA device is
available and the implementation-freeze commit is clean.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph_utils import canonicalize_edges  # noqa: E402
from src.data.loaders import load_mag_data, resolve_path  # noqa: E402
from src.data.semantic_candidates import (  # noqa: E402
    _array_fingerprint,
    _tensor_fingerprint,
    sha256_file,
)
from src.data.semantic_candidates_v7b import (  # noqa: E402
    degree_preserving_double_edge_swap,
    load_semantic_candidate_cache_v7b,
)
from src.models.mag_information_probe_v1 import (  # noqa: E402
    Model,
    SLOT_NAMES,
    TREATMENTS,
    build_row_mean_operator,
    canonical_undirected_edges,
    semantic_edges_from_flags,
)

DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
CONDITIONS = TREATMENTS
BRANCH = "exp/mag_information_utility_v1"
BASELINE_SHA = "4cd655a411a6c0ab12e067285cd2202208b0ab18"
STUDY_YAML = ROOT / "configs/study/mag_information_utility_v1.yaml"
RESEARCH = ROOT / "research/mag_information_utility_v1"
DATA_DIR = RESEARCH / "data"
OUTPUT_ROOT = ROOT / "outputs/mag_information_utility_v1"
RUN_ROOT = OUTPUT_ROOT / "runs"
CACHE_ROOT = OUTPUT_ROOT / "cache"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    tmp.replace(path)


def state_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def graph_array_sha(edge_index: torch.Tensor) -> str:
    edges = canonical_undirected_edges(edge_index, int(edge_index.max().item()) + 1 if edge_index.numel() else 0)
    return hashlib.sha256(edges.t().contiguous().numpy().tobytes()).hexdigest()


def compose_dataset_config(dataset: str, seed: int):
    from hydra import compose, initialize_config_dir

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=mag_information_probe_v1",
                f"seed={int(seed)}",
                "num_runs=1",
                "task.evaluate_test=false",
                "task.scheduler=null",
            ],
        )
    return cfg


def nc_split_path(cfg) -> Path:
    key = "nc_split_path" if "nc_split_path" in cfg.dataset else "node_split_path"
    if key == "nc_split_path":
        path = Path(str(cfg.dataset[key]))
    else:
        path = resolve_path(cfg.dataset[key])
    return resolve_path(path)


def _candidate_cache_path(dataset: str) -> Path:
    candidates = sorted((ROOT / "outputs/psce_mag_v7b/semantic_cache").glob(f"{dataset}_*_v7b.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one existing V7B candidate cache for {dataset}, found {len(candidates)}"
        )
    return candidates[0]


def _source_graph_only(dataset: str):
    # This existing V7B helper deliberately reads features and graph structure,
    # never labels or split files.
    from scripts.run_psce_mag_v7b1_corrective import load_candidate_inputs

    return load_candidate_inputs(dataset)


def _array32(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def _edge_pairs(edge_index: torch.Tensor) -> np.ndarray:
    unique = canonical_undirected_edges(edge_index, int(edge_index.max().item()) + 1 if edge_index.numel() else 0)
    return unique.t().contiguous().numpy()


def _feature_fingerprint_audit(dataset: str, text, visual, sources, metadata) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, feature in (("text", text), ("visual", visual)):
        array = _array32(feature)
        expected = metadata["feature_identity"][name]
        observed_array = _array_fingerprint(array)
        source = Path(sources[name]).resolve()
        observed_file = file_sha256(source)
        if observed_array != expected["array_sha256"]:
            raise ValueError(f"{dataset} {name} features do not match the V7B cache array fingerprint")
        if observed_file != expected["file_sha256"]:
            raise ValueError(f"{dataset} {name} feature file does not match the V7B cache file fingerprint")
        entries[name] = {
            "path": str(source),
            "file_sha256": observed_file,
            "array_sha256": observed_array,
            "shape": list(array.shape),
        }
    return entries


def _stable_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _edge_sha(edges: torch.Tensor) -> str:
    pairs = edges.detach().cpu().long().t().contiguous().numpy().astype(np.int64, copy=False)
    return hashlib.sha256(pairs.tobytes()).hexdigest()


def _random_graph_cache(dataset: str, modality: str, unique_edges: torch.Tensor, num_nodes: int, seed: int, source_fp: str):
    path = CACHE_ROOT / f"{dataset}_random_{modality}.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("metadata", {})
        if (
            meta.get("dataset") != dataset
            or meta.get("modality") != modality
            or int(meta.get("num_nodes", -1)) != int(num_nodes)
            or int(meta.get("seed", -1)) != int(seed)
            or meta.get("source_view_edge_sha256") != source_fp
        ):
            raise ValueError(f"random graph cache provenance mismatch: {path}")
        expected_metadata_fp = _stable_json_sha(
            {key: value for key, value in meta.items() if key != "cache_fingerprint"}
        )
        if expected_metadata_fp != meta.get("cache_fingerprint"):
            raise ValueError(f"random graph cache metadata checksum mismatch: {path}")
        randomized = payload["unique_edge_index"].long().contiguous()
        degree = payload["degree"].long().contiguous()
        computed = torch.bincount(randomized.reshape(-1), minlength=num_nodes)
        if not torch.equal(computed, degree):
            raise ValueError(f"random graph cache degree tensor mismatch: {path}")
        if randomized.numel() and (
            bool((randomized[0] >= randomized[1]).any())
            or (randomized[0] * num_nodes + randomized[1]).unique().numel() != randomized.size(1)
        ):
            raise ValueError(f"random graph cache is not simple canonical undirected: {path}")
        observed_fp = _edge_sha(randomized)
        if observed_fp != meta.get("random_edge_sha256"):
            raise ValueError(f"random graph cache fingerprint mismatch: {path}")
        return randomized, degree, meta, path

    randomized, swap_summary = degree_preserving_double_edge_swap(
        unique_edges, num_nodes, seed=seed, swaps_per_edge=3
    )
    degree = torch.bincount(randomized.reshape(-1), minlength=num_nodes).long()
    source_degree = torch.bincount(unique_edges.reshape(-1), minlength=num_nodes).long()
    if not torch.equal(degree, source_degree):
        raise AssertionError("degree-preserving rewire changed the input degree sequence")
    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "modality": modality,
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "source_view_edge_sha256": source_fp,
        "source_edge_count": int(unique_edges.size(1)),
        "random_edge_count": int(randomized.size(1)),
        "random_edge_sha256": _edge_sha(randomized),
        "source_degree_sha256": hashlib.sha256(source_degree.numpy().tobytes()).hexdigest(),
        "random_degree_sha256": hashlib.sha256(degree.numpy().tobytes()).hexdigest(),
        "swap_audit": swap_summary,
        "label_access": False,
        "training_seed_independent": True,
    }
    metadata["cache_fingerprint"] = _stable_json_sha(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"metadata": metadata, "unique_edge_index": randomized, "degree": degree},
        temporary,
    )
    temporary.replace(path)
    return randomized, degree, metadata, path


def _component_stats(edges: np.ndarray, num_nodes: int) -> tuple[int, float]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if edges.size == 0:
        return int(num_nodes), (1.0 / max(num_nodes, 1))
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    count, labels = connected_components(graph, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=count)
    return int(count), float(sizes.max(initial=0) / max(num_nodes, 1))


def _quantile_record(values: np.ndarray) -> dict[str, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"count": 0}
    out: dict[str, float] = {"count": int(clean.size), "mean": float(clean.mean()), "std": float(clean.std(ddof=0))}
    for q in (0, 5, 25, 50, 75, 95, 100):
        out[f"q{q}"] = float(np.quantile(clean, q / 100.0))
    return out


def _neighbor_attribute_stats(features: np.ndarray, edges: np.ndarray, degree: np.ndarray):
    n, dim = features.shape
    if not len(edges):
        return {
            "edge_cosine": _quantile_record(np.empty(0)),
            "edge_cosine_directed_values": np.empty(0, dtype=np.float32),
            "self_to_neighbor_mean_cosine": _quantile_record(np.empty(0)),
            "neighborhood_variance_per_dimension": _quantile_record(np.empty(0)),
            "covered_nodes": 0,
            "zero_neighbor_mean_nodes": 0,
        }
    edge_cos = np.empty(len(edges), dtype=np.float32)
    sums = np.zeros((n, dim), dtype=np.float32)
    row_norm_sq = np.einsum("ij,ij->i", features, features, dtype=np.float32, optimize=False)
    neighbor_sq_sum = np.zeros(n, dtype=np.float64)
    chunk_size = 32768
    for start in range(0, len(edges), chunk_size):
        stop = min(start + chunk_size, len(edges))
        left = edges[start:stop, 0]
        right = edges[start:stop, 1]
        edge_cos[start:stop] = np.einsum("ij,ij->i", features[left], features[right], optimize=False)
        np.add.at(sums, left, features[right])
        np.add.at(sums, right, features[left])
        np.add.at(neighbor_sq_sum, left, row_norm_sq[right])
        np.add.at(neighbor_sq_sum, right, row_norm_sq[left])
    mean = np.zeros_like(sums)
    covered = degree > 0
    mean[covered] = sums[covered] / degree[covered, None]
    mean_norm = np.linalg.norm(mean, axis=1)
    self_cos = np.zeros(n, dtype=np.float64)
    valid = covered & (mean_norm > 1.0e-12)
    self_cos[valid] = np.einsum(
        "ij,ij->i", features[valid], mean[valid], dtype=np.float64, optimize=False
    ) / mean_norm[valid]
    mean_sq = np.zeros(n, dtype=np.float64)
    mean_sq[covered] = neighbor_sq_sum[covered] / degree[covered]
    variance = np.maximum(mean_sq - np.sum(mean * mean, axis=1), 0.0) / max(dim, 1)
    return {
        "edge_cosine": _quantile_record(edge_cos),
        "edge_cosine_directed_values": np.concatenate((edge_cos, edge_cos)),
        "self_to_neighbor_mean_cosine": _quantile_record(self_cos[covered]),
        "neighborhood_variance_per_dimension": _quantile_record(variance[covered]),
        "covered_nodes": int(covered.sum()),
        "zero_neighbor_mean_nodes": int((covered & ~valid).sum()),
    }


def _degree_stats(edges: np.ndarray, num_nodes: int) -> tuple[np.ndarray, dict[str, Any]]:
    degree = np.bincount(edges.reshape(-1), minlength=num_nodes).astype(np.int64, copy=False)
    components, giant_share = _component_stats(edges, num_nodes)
    stats: dict[str, Any] = {
        "num_nodes": int(num_nodes),
        "undirected_edges": int(len(edges)),
        "directed_edges": int(2 * len(edges)),
        "mean_degree": float(degree.mean()) if num_nodes else 0.0,
        "median_degree": float(np.median(degree)) if num_nodes else 0.0,
        "isolated_nodes": int((degree == 0).sum()),
        "isolated_fraction": float((degree == 0).mean()) if num_nodes else 0.0,
        "connected_components_including_isolates": components,
        "largest_component_fraction": giant_share,
    }
    for q in (0, 10, 25, 50, 75, 90, 99, 100):
        stats[f"degree_q{q}"] = float(np.quantile(degree, q / 100.0)) if num_nodes else 0.0
    return degree, stats


def _graph_stats_rows(dataset: str, text: np.ndarray, visual: np.ndarray, physical_edges: np.ndarray,
                      text_edges: np.ndarray, visual_edges: np.ndarray, cache_meta: dict[str, Any]):
    text_norm, _ = __import__("src.data.semantic_candidates", fromlist=["mean_center_l2_normalize"]).mean_center_l2_normalize(text)
    visual_norm, _ = __import__("src.data.semantic_candidates", fromlist=["mean_center_l2_normalize"]).mean_center_l2_normalize(visual)
    n = int(text.shape[0])
    physical_degree, physical_meta = _degree_stats(physical_edges, n)
    text_degree, text_meta = _degree_stats(text_edges, n)
    visual_degree, visual_meta = _degree_stats(visual_edges, n)
    edge_sets = {
        "physical": physical_edges,
        "text_knn": text_edges,
        "visual_knn": visual_edges,
    }
    code_sets = {
        key: edges[:, 0] * n + edges[:, 1]
        for key, edges in edge_sets.items()
    }
    rows = []
    for relation, edges, degree, base_stats in (
        ("physical", physical_edges, physical_degree, physical_meta),
        ("text_knn", text_edges, text_degree, text_meta),
        ("visual_knn", visual_edges, visual_degree, visual_meta),
    ):
        row: dict[str, Any] = {"dataset": dataset, "relation": relation, **base_stats}
        for other in ("text_knn", "visual_knn"):
            overlap = int(np.intersect1d(code_sets[relation], code_sets[other], assume_unique=True).size)
            row[f"overlap_{other}_edge_count"] = overlap
            row[f"overlap_{other}_fraction_of_relation"] = overlap / max(len(edges), 1)
        physical_overlap = int(np.intersect1d(code_sets[relation], code_sets["physical"], assume_unique=True).size)
        row["physical_overlap_edge_count"] = physical_overlap
        row["physical_overlap_fraction"] = physical_overlap / max(len(edges), 1)
        for modality, features in (("text", text_norm), ("visual", visual_norm)):
            attr = _neighbor_attribute_stats(features, edges, degree)
            for measure in ("edge_cosine", "self_to_neighbor_mean_cosine", "neighborhood_variance_per_dimension"):
                summary = attr[measure]
                row[f"{modality}_{measure}_mean"] = summary.get("mean")
                row[f"{modality}_{measure}_q05"] = summary.get("q5")
                row[f"{modality}_{measure}_median"] = summary.get("q50")
                row[f"{modality}_{measure}_q95"] = summary.get("q95")
            row[f"{modality}_covered_nodes"] = attr["covered_nodes"]
            row[f"{modality}_zero_neighbor_mean_nodes"] = attr["zero_neighbor_mean_nodes"]
            if len(edges):
                src = np.concatenate((edges[:, 0], edges[:, 1]))
                cosine = attr["edge_cosine_directed_values"]
                low_cut = float(np.quantile(physical_degree, 0.25))
                high_cut = float(np.quantile(physical_degree, 0.75))
                high_mask = physical_degree[src] >= high_cut
                row[f"{modality}_edge_cosine_low_physical_degree_mean"] = float(cosine[physical_degree[src] <= low_cut].mean()) if np.any(physical_degree[src] <= low_cut) else None
                row[f"{modality}_edge_cosine_high_physical_degree_mean"] = float(cosine[high_mask].mean()) if np.any(high_mask) else None
                row[f"{modality}_low_degree_node_count"] = int((physical_degree <= low_cut).sum())
                row[f"{modality}_high_degree_node_count"] = int((physical_degree >= high_cut).sum())
        rows.append(row)
    return rows, {"physical": physical_meta, "text_knn": text_meta, "visual_knn": visual_meta}


def _node_self_neighbor_cosine(features: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pre-label node-to-neighbor-mean cosine and relation degree."""
    num_nodes, dim = features.shape
    degree = np.bincount(edges.reshape(-1), minlength=num_nodes).astype(np.int64, copy=False)
    sums = np.zeros((num_nodes, dim), dtype=np.float32)
    for start in range(0, len(edges), 32768):
        block = edges[start : start + 32768]
        left, right = block[:, 0], block[:, 1]
        np.add.at(sums, left, features[right])
        np.add.at(sums, right, features[left])
    mean = np.zeros_like(sums)
    covered = degree > 0
    mean[covered] = sums[covered] / degree[covered, None]
    mean_norm = np.linalg.norm(mean, axis=1)
    valid = covered & (mean_norm > 1e-12)
    cosine = np.full(num_nodes, np.nan, dtype=np.float32)
    cosine[valid] = (
        np.einsum("ij,ij->i", features[valid], mean[valid], optimize=False)
        / mean_norm[valid]
    )
    return cosine, degree


def _quantile_group_codes(values: np.ndarray, key: str) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.isfinite(values)
    codes = np.full(values.shape, -1, dtype=np.int8)
    if not bool(valid.any()):
        return codes, {"feature": key, "cut_points": [], "groups": [], "missing_code": -1}
    cuts = np.unique(np.quantile(values[valid].astype(np.float64), [0.2, 0.4, 0.6, 0.8]))
    codes[valid] = np.searchsorted(cuts, values[valid], side="right").astype(np.int8)
    group_names = [f"Q{i+1}" for i in range(len(cuts) + 1)]
    counts = [int((codes == i).sum()) for i in range(len(group_names))]
    missing_count = int((~valid).sum())
    metadata = {
        "feature": key,
        "definition": "20/40/60/80 percentiles computed over all nodes before label grouping",
        "cut_points": [float(x) for x in cuts.tolist()],
        "groups": group_names,
        "group_counts_all_nodes": counts,
        "missing_or_no_neighbor_code": len(group_names),
        "missing_or_no_neighbor_count": missing_count,
        "labels_used_to_define_groups": False,
    }
    if missing_count:
        codes[~valid] = len(group_names)
    return codes, metadata


def _build_node_groups(dataset: str, bundle: dict[str, Any]) -> dict[str, Any]:
    n = int(bundle["input_info"]["num_nodes"])
    physical = bundle["physical_edges"]
    text_edges = bundle["text_unique"].t().contiguous().numpy()
    visual_edges = bundle["visual_unique"].t().contiguous().numpy()
    from src.data.semantic_candidates import mean_center_l2_normalize

    text_norm, _ = mean_center_l2_normalize(bundle["text"])
    text_physical, physical_degree = _node_self_neighbor_cosine(text_norm, physical)
    text_knn, text_degree = _node_self_neighbor_cosine(text_norm, text_edges)
    del text_norm
    visual_norm, _ = mean_center_l2_normalize(bundle["visual"])
    visual_physical, visual_physical_degree = _node_self_neighbor_cosine(visual_norm, physical)
    visual_knn, visual_degree = _node_self_neighbor_cosine(visual_norm, visual_edges)
    del visual_norm
    if not np.array_equal(physical_degree, visual_physical_degree):
        raise AssertionError("Text/Visual physical-neighbor feature groups use different physical degrees")
    values = {
        "physical_degree_q": physical_degree.astype(np.float32),
        "text_knn_degree_q": text_degree.astype(np.float32),
        "visual_knn_degree_q": visual_degree.astype(np.float32),
        "physical_neighbor_text_consistency_q": text_physical,
        "physical_neighbor_visual_consistency_q": visual_physical,
        "text_knn_neighbor_text_consistency_q": text_knn,
        "visual_knn_neighbor_visual_consistency_q": visual_knn,
    }
    code_arrays: dict[str, np.ndarray] = {}
    definitions: dict[str, Any] = {}
    for key, feature in values.items():
        code_arrays[key], definitions[key] = _quantile_group_codes(feature, key)
    group_dir = OUTPUT_ROOT / "node_groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / f"{dataset}.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **code_arrays)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "node_count": n,
        "groups": definitions,
        "labels_used": False,
    }


def _graph_bundle(dataset: str, *, build_random: bool, compute_stats: bool = True):
    text, visual, physical_edge_index, sources, info = _source_graph_only(dataset)
    text_np, visual_np = _array32(text), _array32(visual)
    num_nodes = int(info["num_nodes"])
    if text_np.shape[0] != num_nodes or visual_np.shape[0] != num_nodes:
        raise ValueError(f"{dataset}: feature rows do not align with graph node IDs")
    cache_path = _candidate_cache_path(dataset)
    cache = load_semantic_candidate_cache_v7b(cache_path, expected_dataset=dataset)
    meta = cache["metadata"]
    if int(meta["num_nodes"]) != num_nodes or int(meta["text_dim"]) != text_np.shape[1] or int(meta["visual_dim"]) != visual_np.shape[1]:
        raise ValueError(f"{dataset}: V7B cache node/dimension mismatch")
    feature_records = _feature_fingerprint_audit(dataset, text_np, visual_np, sources, meta)
    physical_unique = canonical_undirected_edges(physical_edge_index, num_nodes)
    physical_pairs = physical_unique.t().contiguous().numpy()
    physical_sha = hashlib.sha256(np.ascontiguousarray(physical_pairs, dtype=np.int64).view(np.uint8).tobytes()).hexdigest()
    if physical_sha != meta.get("physical_edge_fingerprint"):
        raise ValueError(f"{dataset}: physical graph fingerprint differs from V7B cache provenance")

    source_flags = cache["source_flags"].cpu()
    unique = cache["unique_edge_index"].cpu()
    text_unique = semantic_edges_from_flags(unique, source_flags, 0)
    visual_unique = semantic_edges_from_flags(unique, source_flags, 1)
    p_physical, physical_degree = build_row_mean_operator(physical_unique, num_nodes)
    p_text, text_degree = build_row_mean_operator(text_unique, num_nodes)
    p_visual, visual_degree = build_row_mean_operator(visual_unique, num_nodes)
    text_fp, visual_fp = _edge_sha(text_unique), _edge_sha(visual_unique)
    random_entries = {}
    random_edges = {}
    topology_base = {"Movies": 2026, "Grocery": 2027, "ele-fashion": 2028}[dataset]
    if build_random:
        for modality, edges, fp, offset in (
            ("text", text_unique, text_fp, 0),
            ("visual", visual_unique, visual_fp, 1),
        ):
            seeded = topology_base * 2 + offset
            randomized, degree, rmeta, rpath = _random_graph_cache(
                dataset, modality, edges, num_nodes, seeded, fp
            )
            random_edges[modality] = randomized
            random_entries[modality] = {**rmeta, "path": str(rpath), "file_sha256": file_sha256(rpath)}
    else:
        for modality, fp, source_edges in (("text", text_fp, text_unique), ("visual", visual_fp, visual_unique)):
            rpath = CACHE_ROOT / f"{dataset}_random_{modality}.pt"
            payload = torch.load(rpath, map_location="cpu", weights_only=False)
            rmeta = payload["metadata"]
            expected_seed = topology_base * 2 + int(modality == "visual")
            expected_metadata_fp = _stable_json_sha(
                {key: value for key, value in rmeta.items() if key != "cache_fingerprint"}
            )
            if (
                rmeta.get("source_view_edge_sha256") != fp
                or int(rmeta.get("seed", -1)) != expected_seed
                or rmeta.get("dataset") != dataset
                or rmeta.get("modality") != modality
                or int(rmeta.get("num_nodes", -1)) != num_nodes
                or rmeta.get("cache_fingerprint") != expected_metadata_fp
            ):
                raise ValueError(f"{dataset} {modality} random graph cache provenance/checksum mismatch")
            rand_edges = payload["unique_edge_index"].long().contiguous()
            if rand_edges.ndim != 2 or rand_edges.size(0) != 2:
                raise ValueError(f"{dataset} {modality} random graph cache has invalid edge shape")
            source_degree = torch.bincount(source_edges.reshape(-1), minlength=num_nodes)
            random_degree = torch.bincount(rand_edges.reshape(-1), minlength=num_nodes)
            if not torch.equal(source_degree, random_degree) or rand_edges.size(1) != source_edges.size(1):
                raise ValueError(f"{dataset} {modality} random cache no longer matches degree/edge count")
            if rand_edges.numel() and (
                bool((rand_edges[0] >= rand_edges[1]).any())
                or (rand_edges[0] * num_nodes + rand_edges[1]).unique().numel() != rand_edges.size(1)
            ):
                raise ValueError(f"{dataset} {modality} random graph is not simple canonical undirected")
            if _edge_sha(rand_edges) != rmeta.get("random_edge_sha256"):
                raise ValueError(f"{dataset} {modality} random cache checksum mismatch")
            if hashlib.sha256(source_degree.numpy().tobytes()).hexdigest() != rmeta.get("source_degree_sha256"):
                raise ValueError(f"{dataset} {modality} random cache source-degree checksum mismatch")
            if hashlib.sha256(random_degree.numpy().tobytes()).hexdigest() != rmeta.get("random_degree_sha256"):
                raise ValueError(f"{dataset} {modality} random cache degree checksum mismatch")
            random_edges[modality] = rand_edges
            random_entries[modality] = {**rmeta, "path": str(rpath), "file_sha256": file_sha256(rpath)}

    if compute_stats:
        row_stats, meta_stats = _graph_stats_rows(
            dataset, text_np, visual_np, physical_pairs,
            text_unique.t().contiguous().numpy(), visual_unique.t().contiguous().numpy(), meta,
        )
    else:
        row_stats, meta_stats = [], {}
    codes = {
        name: set((edges[:, 0] * num_nodes + edges[:, 1]).tolist())
        for name, edges in (
            ("physical", physical_pairs),
            ("text", text_unique.t().contiguous().numpy()),
            ("visual", visual_unique.t().contiguous().numpy()),
        )
    }
    source_summary = meta["source_counts"]
    overlap_tv = len(codes["text"] & codes["visual"])
    union_tv = len(codes["text"] | codes["visual"])
    random_audit = {
        "dataset": dataset,
        "num_nodes": num_nodes,
        "source_cache": str(cache_path),
        "source_cache_sha256": file_sha256(cache_path),
        "source_feature_fingerprint": meta["feature_fingerprint"],
        "source_candidate_fingerprint": meta["candidate_fingerprint"],
        "source_v7b_fingerprint": meta["v7b_fingerprint"],
        "physical_edge_fingerprint": physical_sha,
        "knn_k": int(meta["top_k"]),
        "feature_sources": feature_records,
        "candidate_proposals": source_summary,
        "text_visual_overlap_edges": int(overlap_tv),
        "text_visual_jaccard": float(overlap_tv / max(union_tv, 1)),
        "physical_overlap_text": int(len(codes["physical"] & codes["text"])),
        "physical_overlap_visual": int(len(codes["physical"] & codes["visual"])),
        "random_topology_seed_base": int(topology_base),
        "same_topology_for_all_training_seeds": True,
        "label_access_during_candidate_and_null_construction": False,
        "views": {},
        "graph_stats": meta_stats,
    }
    for modality in ("text", "visual"):
        source_set = codes[modality]
        random_set = set(
            (random_edges[modality][0] * num_nodes + random_edges[modality][1]).tolist()
        )
        random_meta = random_entries[modality]
        source_degree = torch.bincount(
            (text_unique if modality == "text" else visual_unique).reshape(-1), minlength=num_nodes
        )
        random_degree = torch.bincount(random_edges[modality].reshape(-1), minlength=num_nodes)
        if len(source_set) != len(random_set) or not torch.equal(source_degree, random_degree):
            raise AssertionError(f"{dataset} {modality} random null does not match source count/degree")
        source_edges_np = (text_unique if modality == "text" else visual_unique).t().contiguous().numpy()
        random_edges_np = random_edges[modality].t().contiguous().numpy()
        _source_degree_np, source_connectivity = _degree_stats(source_edges_np, num_nodes)
        _random_degree_np, random_connectivity = _degree_stats(random_edges_np, num_nodes)
        random_audit["views"][modality] = {
            "source_edge_count": len(source_set),
            "random_edge_count": len(random_set),
            "source_degree_sha256": hashlib.sha256(source_degree.numpy().tobytes()).hexdigest(),
            "random_degree_sha256": hashlib.sha256(random_degree.numpy().tobytes()).hexdigest(),
            "degree_sequence_exact": torch.equal(source_degree, random_degree),
            "same_active_node_set": torch.equal(source_degree > 0, random_degree > 0),
            "source_is_simple_loop_free_graph": bool(
                len(source_set) == text_unique.size(1) if modality == "text" else len(source_set) == visual_unique.size(1)
            ),
            "random_is_simple_loop_free_graph": bool(
                random_edges[modality].size(1) == len(random_set)
                and (not random_edges[modality].numel() or bool((random_edges[modality][0] < random_edges[modality][1]).all()))
            ),
            "source_connectivity": source_connectivity,
            "random_connectivity": random_connectivity,
            "source_random_edge_overlap": len(source_set & random_set),
            "source_random_jaccard": len(source_set & random_set) / max(len(source_set | random_set), 1),
            "source_physical_overlap": len(source_set & codes["physical"]),
            "random_physical_overlap": len(random_set & codes["physical"]),
            "cache": random_meta,
            "swap_audit": random_meta["swap_audit"],
        }
    return {
        "text": text_np,
        "visual": visual_np,
        "physical_edge_index": physical_edge_index,
        "physical_edges": physical_pairs,
        "cache": cache,
        "text_unique": text_unique,
        "visual_unique": visual_unique,
        "random_edges": random_edges,
        "operators": {
            "physical": p_physical,
            "text": p_text,
            "visual": p_visual,
            "random_text": build_row_mean_operator(random_edges["text"], num_nodes)[0],
            "random_visual": build_row_mean_operator(random_edges["visual"], num_nodes)[0],
        },
        "degrees": {
            "physical": physical_degree,
            "text": text_degree,
            "visual": visual_degree,
        },
        "random_audit": random_audit,
        "stats_rows": row_stats,
        "input_info": info,
    }


def _train_only_label_stats(dataset: str, seed: int, edges_by_relation: dict[str, np.ndarray],
                            num_nodes: int, num_classes: int) -> list[dict[str, Any]]:
    cfg = compose_dataset_config(dataset, seed)
    split_path = nc_split_path(cfg)
    if not split_path.is_file():
        raise FileNotFoundError(f"official NC split required; will not generate a replacement: {split_path}")
    data = load_mag_data(cfg, "nc", seed)
    train = data.train_idx.detach().cpu().long().numpy()
    labels = data.y.detach().cpu()
    train_labels = labels[torch.from_numpy(train)].numpy()
    valid_label_mask = (train_labels >= 0) & (train_labels < num_classes)
    train = train[valid_label_mask]
    train_labels = train_labels[valid_label_mask]
    if len(train) != len(np.unique(train)):
        raise ValueError(f"{dataset} seed {seed}: duplicate training node indices")
    class_count = np.bincount(train_labels, minlength=num_classes).astype(np.int64)
    marginal = class_count / max(int(class_count.sum()), 1)
    random_agreement = float(np.sum(marginal**2))
    is_train = np.zeros(num_nodes, dtype=bool)
    is_train[train] = True
    y_lookup = np.full(num_nodes, -1, dtype=np.int64)
    y_lookup[train] = train_labels
    rows: list[dict[str, Any]] = []
    for relation, edges in edges_by_relation.items():
        keep = is_train[edges[:, 0]] & is_train[edges[:, 1]]
        tt = edges[keep]
        y0, y1 = y_lookup[tt[:, 0]], y_lookup[tt[:, 1]]
        label_pairs = np.stack((np.minimum(y0, y1), np.maximum(y0, y1)), axis=1) if len(tt) else np.empty((0, 2), dtype=np.int64)
        unique_pairs, pair_counts = np.unique(label_pairs, axis=0, return_counts=True) if len(tt) else (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64))
        observed_endpoints = np.unique(tt.reshape(-1)) if len(tt) else np.empty(0, dtype=np.int64)
        homophily = float(np.mean(y0 == y1)) if len(tt) else None
        for pair, count in zip(unique_pairs.tolist(), pair_counts.tolist()):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "relation": relation,
                    "record_type": "unordered_class_transition",
                    "class_a": int(pair[0]),
                    "class_b": int(pair[1]),
                    "edge_count": int(count),
                    "train_train_undirected_edge_count": int(len(tt)),
                    "train_endpoint_coverage": int(len(observed_endpoints)),
                    "train_node_count": int(len(train)),
                    "train_endpoint_coverage_fraction": float(len(observed_endpoints) / max(len(train), 1)),
                    "edge_homophily": homophily,
                    "random_label_marginal_agreement": random_agreement,
                    "label_access_scope": "train_labels_only",
                    "used_to_build_or_filter_graph": False,
                }
            )
        if not len(unique_pairs):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "relation": relation,
                    "record_type": "summary_no_train_train_edges",
                    "class_a": "",
                    "class_b": "",
                    "edge_count": 0,
                    "train_train_undirected_edge_count": 0,
                    "train_endpoint_coverage": 0,
                    "train_node_count": int(len(train)),
                    "train_endpoint_coverage_fraction": 0.0,
                    "edge_homophily": None,
                    "random_label_marginal_agreement": random_agreement,
                    "label_access_scope": "train_labels_only",
                    "used_to_build_or_filter_graph": False,
                }
            )
    return rows


def run_graph_audit() -> None:
    graph_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    null_audits: dict[str, Any] = {}
    data_manifest: dict[str, Any] = {}
    node_group_definitions: dict[str, Any] = {}
    for dataset in DATASETS:
        print(f"[graph-audit] {dataset}: loading features/topology and checking V7B cache", flush=True)
        bundle = _graph_bundle(dataset, build_random=True)
        graph_rows.extend(bundle["stats_rows"])
        null_audits[dataset] = bundle["random_audit"]
        node_group_definitions[dataset] = _build_node_groups(dataset, bundle)
        n = int(bundle["input_info"]["num_nodes"])
        physical = bundle["physical_edges"]
        text_edges = bundle["text_unique"].t().contiguous().numpy()
        visual_edges = bundle["visual_unique"].t().contiguous().numpy()
        edges = {"physical": physical, "text_knn": text_edges, "visual_knn": visual_edges}
        data_manifest[dataset] = {
            "input_info": bundle["input_info"],
            "source_cache": bundle["random_audit"]["source_cache"],
            "source_cache_sha256": bundle["random_audit"]["source_cache_sha256"],
            "feature_fingerprint": bundle["random_audit"]["source_feature_fingerprint"],
            "candidate_fingerprint": bundle["random_audit"]["source_candidate_fingerprint"],
            "v7b_fingerprint": bundle["random_audit"]["source_v7b_fingerprint"],
            "physical_edge_fingerprint": bundle["random_audit"]["physical_edge_fingerprint"],
        }
        num_classes = 20 if dataset in {"Movies", "Grocery"} else 12
        for seed in SEEDS:
            print(f"[train-only relation audit] {dataset} seed {seed}", flush=True)
            label_rows.extend(_train_only_label_stats(dataset, seed, edges, n, num_classes))
    graph_fields = sorted({key for row in graph_rows for key in row})
    label_fields = [
        "dataset", "seed", "relation", "record_type", "class_a", "class_b",
        "edge_count", "train_train_undirected_edge_count", "train_endpoint_coverage",
        "train_node_count", "train_endpoint_coverage_fraction", "edge_homophily",
        "random_label_marginal_agreement", "label_access_scope", "used_to_build_or_filter_graph",
    ]
    csv_write(DATA_DIR / "graph_structure_stats.csv", graph_rows, graph_fields)
    csv_write(DATA_DIR / "train_only_label_relation_stats.csv", label_rows, label_fields)
    json_write(DATA_DIR / "random_graph_audit.json", {
        "schema_version": 1,
        "method": "independent_text_visual_undirected_double_edge_swaps",
        "swaps_per_edge": 3,
        "labels_used": False,
        "same_random_graph_across_training_seeds": True,
        "datasets": null_audits,
    })
    json_write(DATA_DIR / "pre_run_data_manifest.json", {
        "baseline_commit": BASELINE_SHA,
        "branch": BRANCH,
        "protocol": "unified_full_graph_nc_v1",
        "datasets": data_manifest,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "test_evaluation": False,
        "candidate_and_random_graph_construction_label_access": False,
    })
    json_write(DATA_DIR / "node_group_definitions.json", {
        "groups_computed_before_model_training": True,
        "labels_used_to_define_groups": False,
        "datasets": node_group_definitions,
    })
    _write_graph_signal_report(bundle_by_dataset=null_audits)


def _write_graph_signal_report(bundle_by_dataset: dict[str, Any]) -> None:
    import csv as csv_module

    stats = list(csv_module.DictReader((DATA_DIR / "graph_structure_stats.csv").open(encoding="utf-8")))
    lines = [
        "# Unlabeled graph and neighborhood-attribute audit",
        "",
        "Physical and semantic structure was audited from node features, graph topology, and V7B cache provenance. KNN construction and random rewiring did not load labels. The separate homophily table uses Train labels only for descriptive counts; it does not influence graph construction or scoring.",
        "",
        "## Graph structure",
        "",
        "| Dataset | Relation | Nodes | Undirected edges | Mean degree | Median degree | Isolated % | Components | Largest-component % | Physical edge overlap % | Text/Visual overlap % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        try:
            overlap_tv = float(row.get("overlap_visual_knn_fraction_of_relation") or 0.0) if row["relation"] == "text_knn" else (float(row.get("overlap_text_knn_fraction_of_relation") or 0.0) if row["relation"] == "visual_knn" else 0.0)
            lines.append(
                f"| {row['dataset']} | {row['relation']} | {int(float(row['num_nodes']))} | {int(float(row['undirected_edges']))} | {float(row['mean_degree']):.3f} | {float(row['median_degree']):.1f} | {100*float(row['isolated_fraction']):.2f} | {int(float(row['connected_components_including_isolates']))} | {100*float(row['largest_component_fraction']):.2f} | {100*float(row['physical_overlap_fraction']):.2f} | {100*overlap_tv:.2f} |"
            )
        except (KeyError, ValueError, TypeError):
            continue
    lines.extend([
        "",
        "## KNN source proposals",
        "",
        "Source flags are reported over the deduplicated undirected candidate union before modality-specific propagation. The two view counts include `Both` edges, so an edge supported by both modalities remains in both KNN views.",
        "",
        "| Dataset | Text-only | Visual-only | Both | Candidate union | Text proposals (%) | Visual proposals (%) | Both (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in sorted(bundle_by_dataset):
        counts = bundle_by_dataset[dataset]["candidate_proposals"]
        text_only = int(counts["text_only"])
        visual_only = int(counts["visual_only"])
        both = int(counts["both"])
        union = text_only + visual_only + both
        lines.append(
            f"| {dataset} | {text_only} | {visual_only} | {both} | {union} | "
            f"{100*(text_only+both)/max(union, 1):.2f} | {100*(visual_only+both)/max(union, 1):.2f} | {100*both/max(union, 1):.2f} |"
        )
    lines.extend([
        "",
        "## Degree-matched random semantic controls",
        "",
        "Each view is rewired separately and keeps the exact degree sequence, active node set, and edge count. The following structural quantities were recorded to expose properties that degree matching does not control.",
        "",
        "| Dataset | View | Edges | Exact degree match | Real components | Random components | Source/null Jaccard | Real physical overlap | Random physical overlap |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for dataset in sorted(bundle_by_dataset):
        views = bundle_by_dataset[dataset]["views"]
        for modality in ("text", "visual"):
            view = views[modality]
            lines.append(
                f"| {dataset} | {modality} | {view['source_edge_count']} | {view['degree_sequence_exact']} | "
                f"{view['source_connectivity']['connected_components_including_isolates']} | "
                f"{view['random_connectivity']['connected_components_including_isolates']} | "
                f"{view['source_random_jaccard']:.5f} | {view['source_physical_overlap']} | {view['random_physical_overlap']} |"
            )
    lines.extend([
        "",
        "## Neighbor feature similarity and dispersion",
        "",
        "`graph_structure_stats.csv` reports Text and Visual edge-cosine distributions for each relation, node-to-neighbor-mean cosine distributions, per-dimension neighborhood variance, and edge cosine means for nodes in the low/high physical-degree quartiles. Similarities use the same mean-centered, row-L2-normalized feature preparation as the V7B KNN cache. These describe feature geometry, not predictive utility.",
        "",
        "## Train-only label relation statistics",
        "",
        "`train_only_label_relation_stats.csv` records Train–Train edge count, endpoint coverage, observed unordered class transitions, edge homophily, and the Train class-marginal random-agreement reference. Validation and Test labels are not used. Homophily is descriptive structure-label alignment and is not a prediction-gain estimate.",
        "",
        "## KNN provenance and random semantic controls",
        "",
        "Random Text and Visual views are rewired independently by valid degree-preserving double-edge swaps; they preserve edge count and the exact per-node degree sequence, have no loops or duplicate edges, and are fixed across all training seeds. The older V7B union random graph is not used as either modality-specific null.",
        "",
        "A single random topology per dataset and source is one null sample, not a distribution over random graphs. Degree matching does not guarantee matching component count, clustering, or overlap with the physical graph.",
        "",
    ])
    (RESEARCH / "GRAPH_SIGNAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def make_train_val_label_view(labels: torch.Tensor, train_idx: torch.Tensor, val_idx: torch.Tensor) -> torch.Tensor:
    """Return labels visible to training/validation; never index a Test target."""
    train_idx = train_idx.detach().cpu().long().reshape(-1)
    val_idx = val_idx.detach().cpu().long().reshape(-1)
    if train_idx.numel() != torch.unique(train_idx).numel() or val_idx.numel() != torch.unique(val_idx).numel():
        raise ValueError("Train or Validation indices contain duplicates")
    if torch.isin(train_idx, val_idx).any():
        raise ValueError("Train and Validation indices overlap")
    source = labels.detach()
    visible = torch.full((int(source.numel()),), -100, dtype=torch.long)
    # Only Train and Validation label indices are dereferenced here.
    visible[train_idx] = source[train_idx.to(source.device)].detach().cpu().long()
    visible[val_idx] = source[val_idx.to(source.device)].detach().cpu().long()
    return visible


def evaluate_logits(logits: torch.Tensor, visible_labels: torch.Tensor, idx: torch.Tensor, num_classes: int) -> dict[str, float]:
    indices = idx.detach().to(device=logits.device, dtype=torch.long)
    targets = visible_labels.to(device=logits.device, dtype=torch.long)[indices]
    if targets.numel() == 0 or bool((targets == -100).any()):
        raise ValueError("metric indices must select nonempty Train/Validation labels only")
    if bool(((targets < 0) | (targets >= int(num_classes))).any()):
        raise ValueError("visible labels are outside configured classes")
    loss = F.cross_entropy(logits.index_select(0, indices), targets)
    predictions = logits.index_select(0, indices).argmax(dim=-1)
    acc = (predictions == targets).float().mean()
    f1 = f1_score(
        targets.detach().cpu().numpy(),
        predictions.detach().cpu().numpy(),
        labels=list(range(int(num_classes))),
        average="macro",
        zero_division=0,
    )
    return {
        "cross_entropy": float(loss.detach().cpu()),
        "accuracy": float(acc.detach().cpu()),
        "macro_f1": float(f1),
    }


def _rng_state_snapshot(device: torch.device | None = None) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": (
            [torch.cuda.get_rng_state(device).clone()]
            if device is not None and device.type == "cuda" and torch.cuda.is_available()
            else []
        ),
    }


def _assert_rng_unchanged(before: dict[str, Any], label: str, device: torch.device | None = None) -> None:
    after = _rng_state_snapshot(device)
    if before["python"] != after["python"]:
        raise AssertionError(f"{label} changed Python RNG state")
    before_numpy, after_numpy = before["numpy"], after["numpy"]
    if (
        before_numpy[0] != after_numpy[0]
        or not np.array_equal(before_numpy[1], after_numpy[1])
        or before_numpy[2:] != after_numpy[2:]
    ):
        raise AssertionError(f"{label} changed NumPy RNG state")
    if not torch.equal(before["cpu"], after["cpu"]):
        raise AssertionError(f"{label} changed CPU Torch RNG state")
    if len(before["cuda"]) != len(after["cuda"]) or any(
        not torch.equal(left, right) for left, right in zip(before["cuda"], after["cuda"])
    ):
        raise AssertionError(f"{label} changed CUDA Torch RNG state")


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(int(seed))


def _cuda_device() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "formal runs require CUDA, but torch.cuda.is_available() is false or there are no visible GPUs"
        )
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {query.stderr.strip()}")
    active_pids_by_uuid: dict[str, set[int]] = {}
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if processes.returncode == 0:
        for line in processes.stdout.splitlines():
            try:
                uuid, pid = [part.strip() for part in line.split(",", 1)]
                active_pids_by_uuid.setdefault(uuid, set()).add(int(pid))
            except (ValueError, TypeError):
                pass
    candidates = []
    gpu_rows = []
    for line in query.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 7:
            continue
        index, uuid, name, total, free, utilization, driver = cells
        row = {
            "index": int(index), "uuid": uuid, "name": name,
            "memory_total_mib": int(total), "memory_free_mib": int(free),
            "utilization_percent": int(utilization), "driver_version": driver,
        }
        gpu_rows.append(row)
        if (
            (int(free) / max(int(total), 1)) >= 0.80
            and int(utilization) <= 10
            and not active_pids_by_uuid.get(uuid)
        ):
            candidates.append(row)
    if not candidates:
        raise RuntimeError(
            "no idle GPU has >=80% free VRAM and <=10% utilization; refusing to interfere with other GPU work"
        )
    selected = max(candidates, key=lambda row: row["memory_free_mib"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() not in {"", "-1"}:
        visible_tokens = [item.strip() for item in visible.split(",")]
        physical_tokens = {str(selected["index"]), selected["uuid"]}
        logical_index = next(
            (i for i, token in enumerate(visible_tokens) if token in physical_tokens),
            None,
        )
        if logical_index is None:
            raise RuntimeError("selected idle physical GPU is not in CUDA_VISIBLE_DEVICES")
    else:
        logical_index = selected["index"]
    if logical_index >= torch.cuda.device_count():
        raise RuntimeError("selected nvidia-smi GPU does not map to a visible PyTorch CUDA device")
    device = torch.device(f"cuda:{logical_index}")
    try:
        name = torch.cuda.get_device_name(device)
    except Exception as exc:
        raise RuntimeError(f"selected nvidia-smi device {selected['index']} is not visible to PyTorch") from exc
    selected["torch_device_name"] = name
    selected["torch_device_index"] = int(logical_index)
    selected["other_compute_pids"] = sorted(active_pids_by_uuid.get(selected["uuid"], set()))
    return device, {
        "selected_gpu": selected,
        "visible_gpu_rows": gpu_rows,
        "compute_pids_by_uuid": {key: sorted(value) for key, value in active_pids_by_uuid.items()},
    }


def _environment_record(gpu_info: dict[str, Any]) -> dict[str, Any]:
    import platform

    import numpy as np

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "selected_gpu": gpu_info["selected_gpu"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "determinism": {
            "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "randomness": {
            "python_seeded": True,
            "numpy_seeded": True,
            "torch_cpu_seeded_per_run": True,
            "only_selected_cuda_device_seeded": True,
            "dropout": "torch.nn.Dropout p=0.20 in training; disabled during metrics",
            "activation_checkpointing": False,
            "train_validation_metrics_preserve_rng": True,
        },
    }


def _run_provenance(dataset: str, seed: int, cfg, bundle: dict[str, Any], data) -> dict[str, Any]:
    split_path = nc_split_path(cfg)
    if not split_path.is_file():
        raise FileNotFoundError(f"official split is missing; no split generation allowed: {split_path}")
    cache_meta = bundle["candidate_cache"]["metadata"]
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    return {
        "branch": git("branch", "--show-current"),
        "source_commit": git("rev-parse", "HEAD"),
        "dataset": dataset,
        "seed": int(seed),
        "protocol": "unified_full_graph_nc_v1",
        "split_path": str(split_path),
        "split_sha256": file_sha256(split_path),
        "resolved_config_sha256": _stable_json_sha(resolved_config),
        "node_count": int(data.num_nodes),
        "physical_directed_edge_count": int(data.edge_index.size(1)),
        "feature_shape_text": list(data.x_t.shape),
        "feature_shape_visual": list(data.x_i.shape),
        "text_array_sha256": _array_fingerprint(_array32(data.x_t)),
        "visual_array_sha256": _array_fingerprint(_array32(data.x_i)),
        "semantic_cache_path": str(bundle["candidate_cache"]["metadata"].get("cache_path", _candidate_cache_path(dataset))),
        "semantic_cache_sha256": file_sha256(_candidate_cache_path(dataset)),
        "feature_fingerprint": cache_meta["feature_fingerprint"],
        "candidate_fingerprint": cache_meta["candidate_fingerprint"],
        "v7b_fingerprint": cache_meta["v7b_fingerprint"],
        "random_text_cache_sha256": file_sha256(CACHE_ROOT / f"{dataset}_random_text.pt"),
        "random_visual_cache_sha256": file_sha256(CACHE_ROOT / f"{dataset}_random_visual.pt"),
        "evaluate_test": False,
        "test_labels_indexed": False,
    }


def _checkpoint_save(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return file_sha256(path)


def _checkpoint_payload(model, classifier, dataset, seed, condition, selection, epoch, metrics, provenance, parameter_audit):
    return {
        "task": "nc",
        "protocol_version": "unified_full_graph_nc_v1",
        "dataset": dataset,
        "seed": int(seed),
        "condition": condition,
        "selection": selection,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "head_state": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
        "run_provenance": provenance,
        "parameter_audit": parameter_audit,
    }


def _module_from_checkpoint(checkpoint, data_info, condition, model_cfg, device):
    model = Model(model_cfg, data_info, treatment=condition).to(device)
    classifier = torch.nn.Linear(model.out_dim, int(data_info["num_classes"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    model.eval()
    classifier.eval()
    return model, classifier


def _evaluate_state(model, classifier, operators, x_text, x_visual, visible_y, train_idx, val_idx, num_classes, device):
    before = _rng_state_snapshot(device)
    model.eval()
    classifier.eval()
    with torch.no_grad():
        fused = model(
            x_text, x_visual, operators["physical"], operators["text"], operators["visual"]
        )
        logits = classifier(fused)
        train_metrics = evaluate_logits(logits, visible_y, train_idx, num_classes)
        val_metrics = evaluate_logits(logits, visible_y, val_idx, num_classes)
        val_idx_dev = val_idx.to(device=device, dtype=torch.long)
        val_targets = visible_y.to(device=device, dtype=torch.long)[val_idx_dev]
        val_logits = logits[val_idx_dev]
        val_preds = val_logits.argmax(dim=-1)
        per_node_ce = F.cross_entropy(val_logits, val_targets, reduction="none")
    _assert_rng_unchanged(before, "evaluation-mode Train/Validation diagnostics", device)
    return train_metrics, val_metrics, val_idx_dev.detach().cpu().numpy(), val_targets.detach().cpu().numpy(), val_preds.detach().cpu().numpy(), per_node_ce.detach().cpu().numpy()


def _check_finite_metrics(metrics: dict[str, float], where: str) -> None:
    for name, value in metrics.items():
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite {where} metric {name}={value}")


def _prepare_dataset_seed(dataset: str, seed: int, device: torch.device) -> dict[str, Any]:
    cfg = compose_dataset_config(dataset, seed)
    split_path = nc_split_path(cfg)
    if not split_path.is_file():
        raise FileNotFoundError(f"formal run requires existing official split: {split_path}")
    data = load_mag_data(cfg, "nc", seed)
    if data.x_t is None or data.x_i is None or data.y is None or data.train_idx is None or data.val_idx is None:
        raise ValueError("NC dataset is missing a modality, labels, Train, or Validation indices")
    if bool(cfg.task.evaluate_test):
        raise ValueError("Test evaluation must remain disabled")
    visible_y = make_train_val_label_view(data.y, data.train_idx, data.val_idx)
    # The runner never indexes the dataset's Test labels and does not pass Test
    # indices to the training/metric functions.
    data.test_idx = None
    data.y = visible_y
    num_classes = int(data.num_classes)
    if bool(((visible_y[data.train_idx] < 0) | (visible_y[data.train_idx] >= num_classes)).any()):
        raise ValueError("training split contains a missing or out-of-range label")
    if bool(((visible_y[data.val_idx] < 0) | (visible_y[data.val_idx] >= num_classes)).any()):
        raise ValueError("validation split contains a missing or out-of-range label")

    bundle = _graph_bundle(dataset, build_random=False, compute_stats=False)
    if int(bundle["input_info"]["num_nodes"]) != int(data.num_nodes):
        raise ValueError("candidate graph and NC loader use different node counts")
    if _array_fingerprint(_array32(data.x_t)) != _array_fingerprint(bundle["text"]):
        raise ValueError("NC Text features do not align with frozen V7B candidate source")
    if _array_fingerprint(_array32(data.x_i)) != _array_fingerprint(bundle["visual"]):
        raise ValueError("NC Visual features do not align with frozen V7B candidate source")
    physical_pairs = canonical_undirected_edges(data.edge_index, int(data.num_nodes)).t().contiguous().numpy()
    if not np.array_equal(physical_pairs, bundle["physical_edges"]):
        raise ValueError("NC physical graph does not align with frozen candidate graph")
    operators_device = {
        name: operator.to(device)
        for name, operator in bundle["operators"].items()
    }
    x_text = data.x_t.to(device)
    x_visual = data.x_i.to(device)
    labels_device = visible_y.to(device)
    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)
    data_info = {
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
        "num_nodes": int(data.num_nodes),
        "num_classes": num_classes,
    }
    # Fingerprints have been checked; do not retain the second full feature copy.
    bundle["text"] = None
    bundle["visual"] = None
    return {
        "cfg": cfg,
        "data": data,
        "bundle": bundle,
        "operators": operators_device,
        "x_text": x_text,
        "x_visual": x_visual,
        "visible_y": labels_device,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "data_info": data_info,
    }


def train_one(dataset: str, seed: int, condition: str, device: torch.device, gpu_info: dict[str, Any], prepared: dict[str, Any], *, overwrite: bool = False):
    if condition not in CONDITIONS:
        raise ValueError(f"invalid condition {condition}")
    cfg = prepared["cfg"]
    data = prepared["data"]
    bundle = prepared["bundle"]
    operators = prepared["operators"]
    x_text = prepared["x_text"]
    x_visual = prepared["x_visual"]
    labels_device = prepared["visible_y"]
    train_idx = prepared["train_idx"]
    val_idx = prepared["val_idx"]
    data_info = prepared["data_info"]
    num_classes = int(data_info["num_classes"])
    if condition == "I4":
        op_device = {
            "physical": operators["physical"],
            "text": operators["random_text"],
            "visual": operators["random_visual"],
        }
    else:
        op_device = {
            "physical": operators["physical"],
            "text": operators["text"],
            "visual": operators["visual"],
        }

    run_dir = RUN_ROOT / dataset
    checkpoint_dir = OUTPUT_ROOT / "checkpoints" / dataset
    stem = f"{dataset}_{condition}_seed_{seed}"
    metrics_path = run_dir / f"{stem}.json"
    history_path = run_dir / f"{stem}_epoch_history.csv"
    node_path = OUTPUT_ROOT / "validation_predictions" / f"{stem}.npz"
    best_acc_path = checkpoint_dir / f"{stem}_best_acc.pt"
    best_f1_path = checkpoint_dir / f"{stem}_best_f1.pt"
    if not overwrite and metrics_path.exists() and best_acc_path.exists() and best_f1_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        if saved.get("source_commit") == git("rev-parse", "HEAD") and saved.get("status") == "complete":
            print(f"[skip completed] {dataset} {condition} seed {seed}", flush=True)
            return saved
        raise RuntimeError(f"existing run output does not match frozen source; inspect manually: {metrics_path}")
    if overwrite:
        for path in (metrics_path, history_path, node_path, best_acc_path, best_f1_path):
            if path.exists():
                path.unlink()

    _set_seed(seed, device)
    model_cfg = cfg.model
    model = Model(model_cfg, data_info, treatment=condition).to(device)
    classifier = torch.nn.Linear(model.out_dim, num_classes).to(device)
    initial_model_sha = state_sha(model)
    initial_head_sha = state_sha(classifier)
    initial_joint_sha = hashlib.sha256((initial_model_sha + initial_head_sha).encode()).hexdigest()
    param_count = sum(parameter.numel() for parameter in model.parameters())
    classifier_count = sum(parameter.numel() for parameter in classifier.parameters())
    parameter_audit = {
        "model_parameter_count": int(param_count),
        "classifier_parameter_count": int(classifier_count),
        "total_parameter_count": int(param_count + classifier_count),
        "model_initial_sha256": initial_model_sha,
        "classifier_initial_sha256": initial_head_sha,
        "joint_initial_sha256": initial_joint_sha,
        "active_context_slots": list(model.active_slots),
        "context_slot_names_per_modality": list(SLOT_NAMES),
        "context_slot_gradient_norms": {},
        "bias_free_context_processor": True,
    }
    provenance = _run_provenance(dataset, seed, cfg, bundle, data)
    provenance["gpu"] = gpu_info
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    node_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=float(cfg.task.lr),
        weight_decay=float(cfg.task.weight_decay),
    )
    max_epochs = int(cfg.task.epochs)
    patience = int(cfg.task.patience)
    min_epoch = int(cfg.task.early_stop_min_epoch)
    min_delta = float(cfg.task.early_stop_min_delta)
    grad_clip = float(cfg.task.grad_clip)
    best_acc = -1.0
    best_f1 = -1.0
    best_acc_epoch = 0
    best_f1_epoch = 0
    best_acc_payload = None
    best_f1_payload = None
    best_acc_metrics = None
    best_f1_metrics = None
    best_acc_node = None
    bad_epochs = 0
    epoch_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    epochs_completed = 0
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        fused = model(
            x_text, x_visual, op_device["physical"], op_device["text"], op_device["visual"]
        )
        logits = classifier(fused.index_select(0, train_idx))
        train_targets = labels_device.index_select(0, train_idx)
        training_loss = F.cross_entropy(logits, train_targets)
        if not bool(torch.isfinite(training_loss)):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        training_loss.backward()
        if epoch == 1:
            grad = model.context_processors["text"].linear1.weight.grad
            if grad is None:
                raise RuntimeError("context processor did not receive a gradient tensor")
            blocks = grad.view(model.hidden_dim, 6, model.hidden_dim)
            norms = blocks.norm(dim=(0, 2)).detach().cpu().tolist()
            parameter_audit["context_slot_gradient_norms"]["text"] = [float(x) for x in norms]
            grad_v = model.context_processors["visual"].linear1.weight.grad
            blocks_v = grad_v.view(model.hidden_dim, 6, model.hidden_dim)
            parameter_audit["context_slot_gradient_norms"]["visual"] = [
                float(x) for x in blocks_v.norm(dim=(0, 2)).detach().cpu().tolist()
            ]
            active = set(model.active_slots)
            for modality in ("text", "visual"):
                observed = parameter_audit["context_slot_gradient_norms"][modality]
                disabled_nonzero = [
                    SLOT_NAMES[index]
                    for index, (slot, magnitude) in enumerate(zip(SLOT_NAMES, observed))
                    if slot not in active and magnitude != 0.0
                ]
                if disabled_nonzero:
                    raise AssertionError(
                        f"disabled {modality} context slots received input-related gradients: {disabled_nonzero}"
                    )
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()),
            max_norm=grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        epochs_completed = epoch
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start

        train_metrics, val_metrics, val_ids, val_targets, val_preds, val_node_ce = _evaluate_state(
            model, classifier, op_device, x_text, x_visual, labels_device,
            train_idx, val_idx, num_classes, device,
        )
        _check_finite_metrics(train_metrics, "Train")
        _check_finite_metrics(val_metrics, "Validation")
        row = {
            "dataset": dataset,
            "condition": condition,
            "seed": int(seed),
            "epoch": epoch,
            "training_mode_stochastic_ce_pre_update": float(training_loss.detach().cpu()),
            "train_ce_eval_after_update": train_metrics["cross_entropy"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_ce": val_metrics["cross_entropy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_val_accuracy_gap": train_metrics["accuracy"] - val_metrics["accuracy"],
            "epoch_wall_seconds": epoch_seconds,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "all_metrics_finite": True,
        }
        epoch_rows.append(row)

        if val_metrics["accuracy"] > best_acc + min_delta:
            best_acc = val_metrics["accuracy"]
            best_acc_epoch = epoch
            best_acc_metrics = {"train": dict(train_metrics), "validation": dict(val_metrics)}
            best_acc_payload = _checkpoint_payload(
                model, classifier, dataset, seed, condition, "best_validation_accuracy",
                epoch, best_acc_metrics, provenance, parameter_audit,
            )
            best_acc_node = (val_ids.copy(), val_targets.copy(), val_preds.copy(), val_node_ce.copy())
            bad_epochs = 0
        else:
            if epoch >= min_epoch:
                bad_epochs += 1
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_f1_epoch = epoch
            best_f1_metrics = {"train": dict(train_metrics), "validation": dict(val_metrics)}
            best_f1_payload = _checkpoint_payload(
                model, classifier, dataset, seed, condition,
                "best_validation_macro_f1_within_accuracy_stopping_horizon",
                epoch, best_f1_metrics, provenance, parameter_audit,
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{dataset} {condition} seed={seed}] epoch={epoch} "
                f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f} val_ce={val_metrics['cross_entropy']:.4f}",
                flush=True,
            )
        if epoch >= min_epoch and bad_epochs >= patience:
            break

    if best_acc_payload is None or best_f1_payload is None or best_acc_node is None:
        raise RuntimeError("training finished without both selected checkpoints")
    duration = time.perf_counter() - started
    acc_sha = _checkpoint_save(best_acc_path, best_acc_payload)
    f1_sha = _checkpoint_save(best_f1_path, best_f1_payload)
    loaded_acc_payload = torch.load(best_acc_path, map_location="cpu", weights_only=False)
    model_acc, head_acc = _module_from_checkpoint(
        loaded_acc_payload, data_info, condition, model_cfg, device
    )
    replay_train, replay_val, replay_ids, replay_y, replay_pred, replay_ce = _evaluate_state(
        model_acc, head_acc, op_device, x_text, x_visual, labels_device,
        train_idx, val_idx, num_classes, device,
    )
    if not np.array_equal(replay_ids, best_acc_node[0]) or not np.array_equal(replay_y, best_acc_node[1]):
        raise AssertionError("Best-Accuracy checkpoint validation node order changed on reload")
    if not np.array_equal(replay_pred, best_acc_node[2]) or not np.allclose(replay_ce, best_acc_node[3], atol=2e-6, rtol=0):
        raise AssertionError("Best-Accuracy checkpoint did not reproduce Validation predictions/loss")
    for metric in ("cross_entropy", "accuracy", "macro_f1"):
        if not math.isclose(replay_val[metric], best_acc_metrics["validation"][metric], rel_tol=0, abs_tol=2e-6):
            raise AssertionError(f"Best-Accuracy checkpoint {metric} replay mismatch")

    loaded_f1_payload = torch.load(best_f1_path, map_location="cpu", weights_only=False)
    model_f1, head_f1 = _module_from_checkpoint(
        loaded_f1_payload, data_info, condition, model_cfg, device
    )
    _f1_train, replay_f1_val, _f1_ids, _f1_y, _f1_pred, _f1_ce = _evaluate_state(
        model_f1, head_f1, op_device, x_text, x_visual, labels_device,
        train_idx, val_idx, num_classes, device,
    )
    for metric in ("cross_entropy", "accuracy", "macro_f1"):
        if not math.isclose(replay_f1_val[metric], best_f1_metrics["validation"][metric], rel_tol=0, abs_tol=2e-6):
            raise AssertionError(f"Best-F1 checkpoint {metric} replay mismatch")

    # Validation-only, local node predictions are ignored by Git. They are
    # consumed only by the post-run aggregate node-group analysis.
    node_tmp = node_path.with_suffix(".npz.tmp")
    with node_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            validation_node_ids=best_acc_node[0],
            validation_labels=best_acc_node[1],
            predictions=best_acc_node[2],
            per_node_cross_entropy=best_acc_node[3],
        )
    node_tmp.replace(node_path)
    csv_write(history_path, epoch_rows, list(epoch_rows[0].keys()))
    row = {
        "status": "complete",
        "dataset": dataset,
        "condition": condition,
        "seed": int(seed),
        "source_commit": git("rev-parse", "HEAD"),
        "baseline_commit": BASELINE_SHA,
        "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False,
        "test_labels_indexed": False,
        "best_accuracy_epoch": int(best_acc_epoch),
        "best_f1_epoch": int(best_f1_epoch),
        "epochs_completed": int(epochs_completed),
        "best_accuracy_checkpoint_metrics": best_acc_metrics,
        "best_f1_checkpoint_metrics": best_f1_metrics,
        "best_accuracy_checkpoint_path": str(best_acc_path.relative_to(ROOT)),
        "best_accuracy_checkpoint_sha256": acc_sha,
        "best_f1_checkpoint_path": str(best_f1_path.relative_to(ROOT)),
        "best_f1_checkpoint_sha256": f1_sha,
        "validation_predictions_path_local": str(node_path.relative_to(ROOT)),
        "validation_predictions_sha256": file_sha256(node_path),
        "best_accuracy_reload_reproduced": True,
        "best_f1_reload_reproduced": True,
        "best_accuracy_reload_metrics": {"train": replay_train, "validation": replay_val},
        "best_f1_reload_metrics": {"validation": replay_f1_val},
        "parameter_audit": parameter_audit,
        "total_parameter_count": int(param_count + classifier_count),
        "training_settings": {
            "protocol": "unified_full_graph_nc_v1",
            "optimizer": "AdamW",
            "learning_rate": float(cfg.task.lr),
            "weight_decay": float(cfg.task.weight_decay),
            "dropout": float(cfg.model.dropout),
            "epochs_max": int(cfg.task.epochs),
            "patience": int(cfg.task.patience),
            "early_stop_min_epoch": int(cfg.task.early_stop_min_epoch),
            "early_stop_min_delta": float(cfg.task.early_stop_min_delta),
            "grad_clip": float(cfg.task.grad_clip),
            "eval_every": int(cfg.task.eval_every),
            "scheduler": None,
            "evaluate_test": False,
            "macro_f1_classes": list(range(num_classes)),
        },
        "training_seconds": float(duration),
        "mean_epoch_seconds": float(duration / max(epochs_completed, 1)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu": gpu_info["selected_gpu"],
        "environment": _environment_record(gpu_info),
        "provenance": provenance,
    }
    json_write(metrics_path, row)
    return row


def _training_source_preflight() -> tuple[str, dict[str, Any]]:
    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"formal run requires branch {BRANCH}, current branch is {branch}")
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError("formal runs require a clean Implementation Freeze worktree")
    head = git("rev-parse", "HEAD")
    if head == BASELINE_SHA:
        raise RuntimeError("Implementation Freeze commit is required before formal runs")
    return head, {"branch": branch, "source_commit": head, "worktree_clean": True}


def run_seed(seed: int, *, overwrite: bool = False) -> None:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    _freeze_sha, _source = _training_source_preflight()
    device, gpu_info = _cuda_device()
    print(f"Selected {device}: {gpu_info['selected_gpu']}", flush=True)
    failures_path = OUTPUT_ROOT / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    for dataset in DATASETS:
        print(f"=== Dataset {dataset} seed {seed}: I0-I4 ===", flush=True)
        print(f"[prepare] {dataset} seed {seed}: validating data/cache fingerprints", flush=True)
        prepared = _prepare_dataset_seed(dataset, seed, device)
        initial_hashes: dict[str, tuple[str, str]] = {}
        for condition in CONDITIONS:
            try:
                result = train_one(
                    dataset, seed, condition, device, gpu_info, prepared, overwrite=overwrite
                )
                pair = (
                    result["parameter_audit"]["model_initial_sha256"],
                    result["parameter_audit"]["classifier_initial_sha256"],
                )
                if condition in initial_hashes and initial_hashes[condition] != pair:
                    raise AssertionError("repeated condition initialization changed")
                if initial_hashes and any(pair != value for value in initial_hashes.values()):
                    raise AssertionError(f"{dataset} seed {seed} initial weights differ across treatments")
                initial_hashes[condition] = pair
            except Exception as exc:
                failure = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "condition": condition,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "source_commit": git("rev-parse", "HEAD"),
                    "time_unix": time.time(),
                }
                failures.append(failure)
                json_write(failures_path, failures)
                raise
        if len(set(initial_hashes.values())) != 1 or len(initial_hashes) != len(CONDITIONS):
            raise AssertionError(f"{dataset} seed {seed} treatment initialization audit failed")


def verify_batch(seed: int) -> dict[str, Any]:
    expected = {(ds, condition, seed) for ds in DATASETS for condition in CONDITIONS}
    rows = []
    errors = []
    init_by_block: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for dataset, condition, run_seed in sorted(expected):
        path = RUN_ROOT / dataset / f"{dataset}_{condition}_seed_{run_seed}.json"
        if not path.is_file():
            errors.append(f"missing run metrics: {path}")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("evaluate_test") is not False or result.get("test_labels_indexed") is not False:
            errors.append(f"run status/Test audit failed: {path}")
        if not result.get("best_accuracy_reload_reproduced") or not result.get("best_f1_reload_reproduced"):
            errors.append(f"checkpoint replay failed: {path}")
        _check_finite_metrics(result["best_accuracy_checkpoint_metrics"]["validation"], str(path))
        _check_finite_metrics(result["best_f1_checkpoint_metrics"]["validation"], str(path))
        for ckpt_key in ("best_accuracy_checkpoint_path", "best_f1_checkpoint_path"):
            checkpoint = ROOT / result[ckpt_key]
            if not checkpoint.is_file():
                errors.append(f"missing checkpoint: {checkpoint}")
        block = (dataset, seed)
        init_by_block.setdefault(block, set()).add((
            result["parameter_audit"]["model_initial_sha256"],
            result["parameter_audit"]["classifier_initial_sha256"],
        ))
        rows.append(result)
    for block, hashes in init_by_block.items():
        if len(hashes) != 1:
            errors.append(f"paired initialization mismatch: {block}")
    audit = {
        "seed": int(seed),
        "expected_runs": len(expected),
        "observed_runs": len(rows),
        "all_finite": not errors,
        "no_test_evaluation": all(row.get("evaluate_test") is False for row in rows),
        "all_checkpoints_reloaded": all(row.get("best_accuracy_reload_reproduced") and row.get("best_f1_reload_reproduced") for row in rows),
        "paired_initializations_match": all(len(value) == 1 for value in init_by_block.values()),
        "errors": errors,
    }
    json_write(OUTPUT_ROOT / f"seed_{seed}_batch_audit.json", audit)
    if errors or len(rows) != len(expected):
        raise RuntimeError(f"Seed {seed} batch gate failed: {errors}")
    return audit


def _write_failures_snapshot() -> None:
    path = OUTPUT_ROOT / "failures.json"
    failures = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    json_write(DATA_DIR / "failures.json", failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("graph-audit", "train-seed", "verify-batch"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "graph-audit":
        run_graph_audit()
        _write_failures_snapshot()
    elif args.stage == "train-seed":
        if args.seed is None:
            parser.error("--seed is required for train-seed")
        run_seed(args.seed, overwrite=args.overwrite)
    else:
        if args.seed is None:
            parser.error("--seed is required for verify-batch")
        print(json.dumps(verify_batch(args.seed), indent=2))


if __name__ == "__main__":
    main()
