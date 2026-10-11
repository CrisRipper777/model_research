"""V7B provenance-aware semantic candidate caches and degree-preserving nulls.

The base topology intentionally follows V7A's exact feature preparation,
FAISS search, stable tie-breaking, and edge tensor layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.semantic_candidates import (
    CACHE_SCHEMA_VERSION,
    EXACT_INDEX_NODE_LIMIT,
    FAISS_THREADS,
    INDEX_SEED,
    RULE_VERSION,
    _array_fingerprint,
    _faiss_knn,
    _json_fingerprint,
    _sorted_unique_pairs,
    _sample_recall,
    _tensor_fingerprint,
    canonical_undirected_edges,
    load_semantic_candidate_cache,
    mean_center_l2_normalize,
)

V7B_RULE_VERSION = "psce_mag_v7b_source_flags_relative_edges_v1"


def _source_edge_counts(query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count raw directed KNN proposals per canonical edge, capped at two."""
    n, k = query_ids.shape
    query = np.repeat(np.arange(n, dtype=np.int64), k)
    target = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    pairs = np.sort(np.stack((query, target), axis=1), axis=1)
    valid = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[valid]
    codes = pairs[:, 0] * n + pairs[:, 1]
    unique, counts = np.unique(codes, return_counts=True)
    return unique.astype(np.int64, copy=False), np.minimum(counts, 2).astype(np.uint8)


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_semantic_candidate_cache_v7b(
    *,
    dataset: str,
    text_features: np.ndarray | torch.Tensor,
    visual_features: np.ndarray | torch.Tensor,
    physical_edge_index: torch.Tensor | np.ndarray,
    cache_dir: str | Path,
    text_source: str | Path | None = None,
    visual_source: str | Path | None = None,
    top_k: int = 8,
    seed: int = INDEX_SEED,
    thread_count: int = FAISS_THREADS,
) -> tuple[Path, dict[str, Any]]:
    """Build V7A-identical candidate topology and raw-query provenance flags."""
    if int(top_k) != 8:
        raise ValueError("V7B freezes both modality searches at K=8")
    if thread_count < 1:
        raise ValueError("FAISS thread_count must be positive")
    start_time = time.perf_counter()
    text_raw = np.ascontiguousarray(
        text_features.detach().cpu().numpy() if isinstance(text_features, torch.Tensor) else text_features,
        dtype=np.float32,
    )
    visual_raw = np.ascontiguousarray(
        visual_features.detach().cpu().numpy() if isinstance(visual_features, torch.Tensor) else visual_features,
        dtype=np.float32,
    )
    if text_raw.ndim != 2 or visual_raw.ndim != 2 or text_raw.shape[0] != visual_raw.shape[0]:
        raise ValueError("Text and Visual features must be matrices with the same node count")
    n = int(text_raw.shape[0])
    if n < 2:
        raise ValueError("semantic candidate construction needs at least two nodes")

    # This identity definition is copied from the frozen V7A builder.
    def source_meta(source, values):
        if source is None:
            fp = _array_fingerprint(values)
            return {"identifier": "in_memory_array", "file_sha256": fp, "array_sha256": fp, "file_bytes": int(values.nbytes)}
        from src.data.semantic_candidates import _source_descriptor
        return _source_descriptor(source, values)

    feature_identity = {
        "dataset": str(dataset),
        "num_nodes": n,
        "text_dim": int(text_raw.shape[1]),
        "visual_dim": int(visual_raw.shape[1]),
        "text": source_meta(text_source, text_raw),
        "visual": source_meta(visual_source, visual_raw),
    }
    feature_fingerprint = _digest_json(feature_identity)

    text_norm, text_norm_stats = mean_center_l2_normalize(text_raw)
    text_ids, text_sims, text_params = _faiss_knn(
        text_norm, k=8, seed=seed, thread_count=thread_count
    )
    text_recall = (
        _sample_recall(text_norm, text_ids, k=8, seed=seed)
        if not text_params["exact"]
        else {"sample_count": 0, "recall_at_k_mean": 1.0, "recall_at_k_min": 1.0}
    )
    text_codes, text_counts = _source_edge_counts(text_ids)
    text_proposed = _sorted_unique_pairs(
        np.stack(
            [np.repeat(np.arange(n, dtype=np.int64), 8), text_ids.reshape(-1)], axis=1
        ),
        n,
    )
    del text_norm

    visual_norm, visual_norm_stats = mean_center_l2_normalize(visual_raw)
    visual_ids, visual_sims, visual_params = _faiss_knn(
        visual_norm, k=8, seed=seed, thread_count=thread_count
    )
    visual_recall = (
        _sample_recall(visual_norm, visual_ids, k=8, seed=seed)
        if not visual_params["exact"]
        else {"sample_count": 0, "recall_at_k_mean": 1.0, "recall_at_k_min": 1.0}
    )
    visual_codes, visual_counts = _source_edge_counts(visual_ids)
    visual_proposed = _sorted_unique_pairs(
        np.stack(
            [np.repeat(np.arange(n, dtype=np.int64), 8), visual_ids.reshape(-1)], axis=1
        ),
        n,
    )
    del visual_norm

    union = _sorted_unique_pairs(np.concatenate((text_proposed, visual_proposed), axis=0), n)
    edge_count = int(union.shape[0])
    edge_codes = union[:, 0] * n + union[:, 1]
    p_text = np.zeros(edge_count, dtype=np.uint8)
    p_visual = np.zeros(edge_count, dtype=np.uint8)
    text_positions = np.searchsorted(edge_codes, text_codes)
    p_text[text_positions] = text_counts
    visual_positions = np.searchsorted(edge_codes, visual_codes)
    p_visual[visual_positions] = visual_counts
    if np.any((p_text + p_visual) == 0):
        raise RuntimeError("candidate union contains an edge without a recorded KNN source")
    source_flags = torch.from_numpy(np.stack((p_text, p_visual), axis=1).copy()).to(torch.uint8)

    symmetric = np.concatenate((union, union[:, ::-1]), axis=0)
    edge_index = torch.from_numpy(symmetric.T.copy()).long().contiguous()
    unique_edge_index = torch.from_numpy(union.T.copy()).long().contiguous()
    degree_np = np.bincount(union.reshape(-1), minlength=n).astype(np.int64, copy=False)
    degree = torch.from_numpy(degree_np.copy()).long().contiguous()
    physical = canonical_undirected_edges(physical_edge_index, n)
    physical_fp = hashlib.sha256(np.ascontiguousarray(physical, dtype=np.int64).view(np.uint8).tobytes()).hexdigest()
    physical_codes = physical[:, 0] * n + physical[:, 1]
    physical_overlap = int(np.intersect1d(edge_codes, physical_codes, assume_unique=True).size)
    text_edge_codes = text_proposed[:, 0] * n + text_proposed[:, 1]
    visual_edge_codes = visual_proposed[:, 0] * n + visual_proposed[:, 1]
    shared_count = int(np.intersect1d(text_edge_codes, visual_edge_codes, assume_unique=True).size)
    jaccard = shared_count / max(len(text_edge_codes) + len(visual_edge_codes) - shared_count, 1)
    params = {
        "text": text_params,
        "visual": visual_params,
        "faiss_version": text_params["faiss_version"],
        "thread_count": int(thread_count),
        "random_seed": int(seed),
    }
    metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": str(dataset),
        "num_nodes": n,
        "text_dim": int(text_raw.shape[1]),
        "visual_dim": int(visual_raw.shape[1]),
        "rule_version": RULE_VERSION,
        "v7b_rule_version": V7B_RULE_VERSION,
        "feature_identity": feature_identity,
        "feature_fingerprint": feature_fingerprint,
        "top_k": 8,
        "candidate_sources": ["text_knn", "visual_knn"],
        "symmetrization": "union_then_canonical_deduplicate_then_add_reverse",
        "index_provenance": params,
        "text_normalization": text_norm_stats,
        "visual_normalization": visual_norm_stats,
        "directed_candidate_edge_count": int(edge_index.size(1)),
        "undirected_candidate_edge_count": edge_count,
        "text_undirected_edge_count": int(len(text_proposed)),
        "visual_undirected_edge_count": int(len(visual_proposed)),
        "text_visual_overlap_count": shared_count,
        "text_visual_overlap_jaccard": float(jaccard),
        "physical_edge_count": int(len(physical)),
        "physical_edge_fingerprint": physical_fp,
        "candidate_physical_overlap_count": physical_overlap,
        "candidate_physical_overlap_ratio": float(physical_overlap / max(edge_count, 1)),
        "candidate_nonphysical_ratio": float(1.0 - physical_overlap / max(edge_count, 1)),
        "isolated_node_count": int((degree_np == 0).sum()),
        "covered_node_count": int((degree_np > 0).sum()),
        "degree_quantiles": {str(q): float(np.quantile(degree_np, q / 100.0)) for q in (0, 10, 25, 50, 75, 90, 99, 100)},
        "similarity_quantiles": {
            modality: {
                str(q): float(np.quantile(values, q / 100.0))
                for q in (0, 10, 25, 50, 75, 90, 99, 100)
            }
            for modality, values in (("text", text_sims.reshape(-1)), ("visual", visual_sims.reshape(-1)))
        },
        "retrieval_recall_sample": {"text": text_recall, "visual": visual_recall},
        "source_counts": {
            "text_only": int(((p_text > 0) & (p_visual == 0)).sum()),
            "visual_only": int(((p_text == 0) & (p_visual > 0)).sum()),
            "both": int(((p_text > 0) & (p_visual > 0)).sum()),
            "text_proposals_before_symmetrization": int(8 * n),
            "visual_proposals_before_symmetrization": int(8 * n),
            "text_count_0_1_2": {str(i): int((p_text == i).sum()) for i in range(3)},
            "visual_count_0_1_2": {str(i): int((p_visual == i).sum()) for i in range(3)},
        },
        "cache_contains_labels": False,
        "candidate_phase_label_access": False,
        "construction_seconds": 0.0,
    }
    legacy_fp = _tensor_fingerprint(edge_index, degree, unique_edge_index)
    metadata["candidate_fingerprint"] = legacy_fp
    metadata["v7a_candidate_fingerprint"] = legacy_fp
    metadata["v7b_fingerprint"] = _digest_json(
        {
            **{k: v for k, v in metadata.items() if k != "construction_seconds"},
            "source_flags_sha256": hashlib.sha256(source_flags.numpy().tobytes()).hexdigest(),
        }
    )
    metadata["construction_seconds"] = time.perf_counter() - start_time
    metadata["cache_fingerprint"] = _json_fingerprint(
        {key: value for key, value in metadata.items() if key != "construction_seconds"}
    )
    path = Path(cache_dir) / f"{dataset}_{feature_fingerprint[:12]}_{legacy_fp[:12]}_v7b.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "edge_index": edge_index,
        "unique_edge_index": unique_edge_index,
        "degree": degree,
        "source_flags": source_flags,
        "text_similarity": torch.from_numpy(text_sims.reshape(-1).copy()),
        "visual_similarity": torch.from_numpy(visual_sims.reshape(-1).copy()),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    # Ensure the graph tensors pass the unchanged V7A loader contract.
    load_semantic_candidate_cache(path, expected_dataset=dataset, expected_candidate_fingerprint=legacy_fp)
    return path, metadata


def load_semantic_candidate_cache_v7b(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_feature_fingerprint: str | None = None,
    expected_candidate_fingerprint: str | None = None,
    expected_v7b_fingerprint: str | None = None,
) -> dict[str, Any]:
    base = load_semantic_candidate_cache(
        path,
        expected_dataset=expected_dataset,
        expected_feature_fingerprint=expected_feature_fingerprint,
        expected_candidate_fingerprint=expected_candidate_fingerprint,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata.get("v7b_rule_version") != V7B_RULE_VERSION:
        raise ValueError("semantic cache V7B rule version mismatch")
    flags = payload.get("source_flags")
    e = base["unique_edge_index"].size(1)
    if not isinstance(flags, torch.Tensor) or flags.dtype != torch.uint8 or tuple(flags.shape) != (e, 2):
        raise ValueError("V7B source flags must have uint8 shape [unique_edges,2]")
    if flags.numel() and int(flags.max()) > 2:
        raise ValueError("V7B source counts must lie in {0,1,2}")
    if not metadata.get("random_graph") and bool((flags.sum(dim=1) == 0).any()):
        raise ValueError("every semantic edge must have at least one recorded source")
    complete_fp = metadata.get("v7b_fingerprint")
    if expected_v7b_fingerprint is not None and complete_fp != expected_v7b_fingerprint:
        raise ValueError("semantic V7B source fingerprint mismatch")
    observed_source_fp = hashlib.sha256(flags.contiguous().numpy().tobytes()).hexdigest()
    expected_complete = _digest_json(
        {
            **{k: v for k, v in metadata.items() if k not in {"construction_seconds", "cache_fingerprint", "v7b_fingerprint"}},
            "source_flags_sha256": observed_source_fp,
        }
    )
    if expected_complete != complete_fp:
        raise ValueError("semantic V7B fingerprint does not match cache contents")
    base["source_flags"] = flags
    base["metadata"] = metadata
    if "text_similarity" in payload:
        base["text_similarity"] = payload["text_similarity"]
        base["visual_similarity"] = payload["visual_similarity"]
    return base


def degree_preserving_double_edge_swap(
    unique_edge_index: torch.Tensor | np.ndarray,
    num_nodes: int,
    *,
    seed: int,
    swaps_per_edge: int = 3,
    max_attempt_factor: int = 50,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rewire a simple undirected graph using valid 2-switches only."""
    edges = np.asarray(
        unique_edge_index.detach().cpu().numpy() if isinstance(unique_edge_index, torch.Tensor) else unique_edge_index,
        dtype=np.int64,
    )
    if edges.ndim != 2:
        raise ValueError("unique_edge_index must be rank two")
    if edges.shape[0] == 2:
        edges = edges.T
    if edges.shape[1] != 2:
        raise ValueError("unique_edge_index must have shape [2,E] or [E,2]")
    edges = np.sort(edges.copy(), axis=1)
    if len(edges) and (np.any(edges[:, 0] == edges[:, 1]) or np.any(edges < 0) or np.any(edges >= num_nodes)):
        raise ValueError("input graph contains loops or invalid node IDs")
    codes = edges[:, 0] * int(num_nodes) + edges[:, 1]
    if np.unique(codes).size != len(edges):
        raise ValueError("input graph must have unique canonical edges")
    before_degree = np.bincount(edges.reshape(-1), minlength=num_nodes)
    e = int(len(edges))
    target = int(swaps_per_edge) * e
    if target < 1:
        raise ValueError("swap target must be positive")
    if e < 2:
        raise RuntimeError("cannot rewire a graph with fewer than two edges")
    original_codes = set(int(code) for code in codes.tolist())
    present = set(original_codes)
    rng = np.random.default_rng(int(seed))
    successes = attempts = 0
    limit = target * int(max_attempt_factor)
    while successes < target and attempts < limit:
        attempts += 1
        idx = rng.integers(0, e, size=2)
        i, j = int(idx[0]), int(idx[1])
        if i == j:
            continue
        a, b = map(int, edges[i])
        c, d = map(int, edges[j])
        if rng.integers(0, 2):
            a, b = b, a
        if rng.integers(0, 2):
            c, d = d, c
        if rng.integers(0, 2):
            new_one, new_two = (min(a, c), max(a, c)), (min(b, d), max(b, d))
        else:
            new_one, new_two = (min(a, d), max(a, d)), (min(b, c), max(b, c))
        if new_one[0] == new_one[1] or new_two[0] == new_two[1] or new_one == new_two:
            continue
        old_one = (min(int(edges[i, 0]), int(edges[i, 1])), max(int(edges[i, 0]), int(edges[i, 1])))
        old_two = (min(int(edges[j, 0]), int(edges[j, 1])), max(int(edges[j, 0]), int(edges[j, 1])))
        old_codes = {old_one[0] * num_nodes + old_one[1], old_two[0] * num_nodes + old_two[1]}
        code_one = new_one[0] * num_nodes + new_one[1]
        code_two = new_two[0] * num_nodes + new_two[1]
        if code_one == code_two or code_one in present or code_two in present:
            continue
        for code in old_codes:
            present.remove(code)
        present.add(code_one)
        present.add(code_two)
        edges[i] = new_one
        edges[j] = new_two
        successes += 1
    if successes != target:
        raise RuntimeError(f"double-edge swaps stalled: {successes}/{target} after {attempts} attempts")
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    after_degree = np.bincount(edges.reshape(-1), minlength=num_nodes)
    if len(edges) != e or not np.array_equal(before_degree, after_degree):
        raise AssertionError("double-edge swaps changed edge count or degree sequence")
    if len(np.unique(edges[:, 0] * num_nodes + edges[:, 1])) != e:
        raise AssertionError("double-edge swaps created duplicate edges")
    summary = {
        "seed": int(seed),
        "algorithm": "canonical undirected double-edge swaps with rejection of loops/duplicates",
        "swaps_per_edge": int(swaps_per_edge),
        "successful_swaps": successes,
        "attempts": attempts,
        "acceptance_rate": successes / max(attempts, 1),
        "edge_count_before": e,
        "edge_count_after": int(len(edges)),
        "degree_sequence_exact": True,
        "degree_sequence_sha256": hashlib.sha256(np.ascontiguousarray(after_degree, dtype=np.int64).view(np.uint8).tobytes()).hexdigest(),
        "changed_edge_count": int(len(original_codes - present)),
        "max_degree_difference": int(np.abs(before_degree - after_degree).max(initial=0)),
    }
    return torch.from_numpy(edges.T.copy()).long().contiguous(), summary


def write_random_candidate_cache_v7b(
    base_cache_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    swaps_per_edge: int = 3,
) -> dict[str, Any]:
    base = load_semantic_candidate_cache_v7b(base_cache_path)
    meta = dict(base["metadata"])
    swapped, summary = degree_preserving_double_edge_swap(
        base["unique_edge_index"], int(meta["num_nodes"]), seed=seed, swaps_per_edge=swaps_per_edge
    )
    n = int(meta["num_nodes"])
    unique = swapped
    forward = unique
    reverse = unique.flip(0)
    edge_index = torch.cat((forward, reverse), dim=1).contiguous()
    degree = torch.bincount(edge_index[0], minlength=n).long()
    random_fp = _tensor_fingerprint(edge_index, degree, unique)
    meta["candidate_fingerprint"] = random_fp
    meta["source_counts"] = {
        "source_flags_available": False,
        "randomized_candidate_edge_count": int(unique.size(1)),
    }
    meta["random_graph"] = {**summary, "source_candidate_fingerprint": base["metadata"]["candidate_fingerprint"]}
    meta["v7b_fingerprint"] = _digest_json(
        {
            **{k: v for k, v in meta.items() if k not in {"construction_seconds", "cache_fingerprint", "v7b_fingerprint"}},
            "source_flags_sha256": hashlib.sha256(bytes(len(unique.T) * 2)).hexdigest(),
        }
    )
    meta.pop("cache_fingerprint", None)
    meta["cache_fingerprint"] = _json_fingerprint(
        {k: v for k, v in meta.items() if k != "construction_seconds"}
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": meta,
        "edge_index": edge_index,
        "unique_edge_index": unique,
        "degree": degree,
        "source_flags": torch.zeros((unique.size(1), 2), dtype=torch.uint8),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out)
    # The null graph has no source flags; it is loaded only by B2, whose scorer is disabled.
    from src.data.semantic_candidates import load_semantic_candidate_cache as load_v7a
    load_v7a(out, expected_candidate_fingerprint=random_fp)
    return {
        "path": str(out),
        "candidate_fingerprint": random_fp,
        "v7b_fingerprint": meta["v7b_fingerprint"],
        **summary,
    }
