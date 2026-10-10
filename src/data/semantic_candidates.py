"""Label-free FAISS construction and integrity checks for V7A candidates."""

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


RULE_VERSION = "psce_mag_v7a_mean_center_cosine_union_sym_v1"
CACHE_SCHEMA_VERSION = 1
INDEX_SEED = 2026
FAISS_THREADS = 8
EXACT_INDEX_NODE_LIMIT = 25_000


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(contiguous.view(np.uint8).tobytes())


def mean_center_l2_normalize(features: np.ndarray | torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    """Mean-center each modality globally and return contiguous unit rows."""
    if isinstance(features, torch.Tensor):
        values = features.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must be a nonempty [num_nodes, dim] matrix")
    if not np.isfinite(values).all():
        raise ValueError("features contain NaN or Inf")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = np.ascontiguousarray(values - mean[None, :], dtype=np.float32)
    norms = np.linalg.norm(centered, axis=1)
    nonzero = norms > 1.0e-12
    centered[nonzero] /= norms[nonzero, None]
    centered[~nonzero] = 0.0
    summary = {
        "feature_mean_l2": float(np.linalg.norm(mean)),
        "zero_centered_rows": int((~nonzero).sum()),
        "nonzero_centered_rows": int(nonzero.sum()),
    }
    return centered, summary


def canonical_undirected_edges(edge_index: torch.Tensor | np.ndarray, num_nodes: int) -> np.ndarray:
    """Return sorted, unique, non-self `[E,2]` pairs with `u < v`."""
    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.detach().to(device="cpu", dtype=torch.int64).numpy()
    else:
        edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError("edge_index must be rank two")
    if edges.shape[0] == 2:
        edges = edges.T
    if edges.shape[1] != 2:
        raise ValueError("edge_index must have shape [2,E] or [E,2]")
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if int(edges.min()) < 0 or int(edges.max()) >= int(num_nodes):
        raise ValueError("edge_index contains an out-of-range node ID")
    pairs = np.sort(edges, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    return _sorted_unique_pairs(pairs, num_nodes)


def _sorted_unique_pairs(pairs: np.ndarray, num_nodes: int) -> np.ndarray:
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    codes = pairs[:, 0] * int(num_nodes) + pairs[:, 1]
    unique_codes = np.unique(codes)
    return np.stack(
        [unique_codes // int(num_nodes), unique_codes % int(num_nodes)], axis=1
    ).astype(np.int64, copy=False)


def _faiss_knn(
    normalized: np.ndarray,
    *,
    k: int,
    seed: int,
    thread_count: int,
    exact_node_limit: int = EXACT_INDEX_NODE_LIMIT,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "V7A candidate construction requires FAISS; install a compatible faiss-cpu "
            "package in yhf_env before building large MAG graphs"
        ) from exc

    num_nodes, dim = normalized.shape
    if num_nodes < 2:
        raise ValueError("KNN retrieval requires at least two nodes")
    k_eff = min(int(k), num_nodes - 1)
    if k_eff < 1:
        raise ValueError("K must be positive")
    faiss.omp_set_num_threads(int(thread_count))
    use_exact = num_nodes <= int(exact_node_limit)
    if use_exact:
        index = faiss.IndexFlatIP(int(dim))
        params: dict[str, Any] = {
            "index_type": "IndexFlatIP",
            "exact": True,
            "nlist": None,
            "nprobe": None,
            "seed": int(seed),
        }
    else:
        nlist = min(2048, max(64, int(round(2.0 * math.sqrt(num_nodes)))))
        nlist = min(nlist, max(1, num_nodes // 40))
        nlist = max(1, nlist)
        quantizer = faiss.IndexFlatIP(int(dim))
        index = faiss.IndexIVFFlat(
            quantizer, int(dim), int(nlist), faiss.METRIC_INNER_PRODUCT
        )
        index.cp.seed = int(seed)
        index.cp.niter = 20
        index.cp.nredo = 1
        index.cp.min_points_per_centroid = 1
        index.train(normalized)
        nprobe = min(int(nlist), max(16, int(math.ceil(2.0 * math.sqrt(nlist)))))
        index.nprobe = int(nprobe)
        params = {
            "index_type": "IndexIVFFlat",
            "exact": False,
            "nlist": int(nlist),
            "nprobe": int(nprobe),
            "training_iterations": 20,
            "training_seed": int(seed),
        }
    index.add(normalized)

    candidate_ids = np.empty((num_nodes, k_eff), dtype=np.int64)
    candidate_sims = np.empty((num_nodes, k_eff), dtype=np.float32)
    batch_size = 4096 if not use_exact else max(64, min(1024, 16_000_000 // num_nodes))
    for start in range(0, num_nodes, batch_size):
        stop = min(start + batch_size, num_nodes)
        sims, ids = index.search(normalized[start:stop], min(num_nodes, k_eff + 1))
        for local, query_id in enumerate(range(start, stop)):
            valid = (ids[local] >= 0) & (ids[local] != query_id)
            row_ids = ids[local][valid]
            row_sims = sims[local][valid]
            # Stable tie-breaking by ascending global node ID.
            order = np.lexsort((row_ids, -row_sims))[:k_eff]
            if len(order) != k_eff:
                raise RuntimeError(
                    f"FAISS returned only {len(order)} non-self neighbors for node {query_id}"
                )
            candidate_ids[query_id] = row_ids[order]
            candidate_sims[query_id] = row_sims[order]

    params.update(
        {
            "faiss_version": str(getattr(faiss, "__version__", "unknown")),
            "thread_count": int(thread_count),
            "batch_size": int(batch_size),
            "k": int(k_eff),
            "dimension": int(dim),
        }
    )
    return candidate_ids, candidate_sims, params


def _sample_recall(
    normalized: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    k: int,
    seed: int,
    sample_count: int = 256,
) -> dict[str, Any]:
    """Compare approximate neighbors with exact feature-only queries on a fixed sample."""
    num_nodes = normalized.shape[0]
    sample_count = min(int(sample_count), num_nodes)
    if sample_count == 0:
        return {"sample_count": 0, "recall_at_k": None}
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS became unavailable during recall audit") from exc
    rng = np.random.default_rng(int(seed))
    sample_ids = np.sort(rng.choice(num_nodes, size=sample_count, replace=False))
    exact = faiss.IndexFlatIP(int(normalized.shape[1]))
    exact.add(normalized)
    _, indices = exact.search(normalized[sample_ids], min(num_nodes, k + 1))
    recalls: list[float] = []
    for row, query_id in enumerate(sample_ids.tolist()):
        exact_ids = [int(i) for i in indices[row] if int(i) != query_id][:k]
        approximate_ids = set(int(i) for i in candidate_ids[query_id, :k])
        recalls.append(len(set(exact_ids) & approximate_ids) / max(len(exact_ids), 1))
    return {
        "sample_count": int(sample_count),
        "sample_seed": int(seed),
        "recall_at_k_mean": float(np.mean(recalls)),
        "recall_at_k_min": float(np.min(recalls)),
    }


def _source_descriptor(source: str | Path | None, array: np.ndarray) -> dict[str, Any]:
    if source is None:
        return {
            "identifier": "in_memory_array",
            "file_sha256": _array_fingerprint(array),
            "array_sha256": _array_fingerprint(array),
            "file_bytes": int(array.nbytes),
        }
    path = Path(source).resolve()
    return {
        "identifier": str(path),
        "file_sha256": sha256_file(path),
        "array_sha256": _array_fingerprint(array),
        "file_bytes": int(path.stat().st_size),
    }


def _tensor_fingerprint(
    edge_index: torch.Tensor,
    degree: torch.Tensor,
    unique_edge_index: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(edge_index.detach().cpu().contiguous().numpy().tobytes())
    digest.update(unique_edge_index.detach().cpu().contiguous().numpy().tobytes())
    digest.update(degree.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _json_fingerprint(value: dict[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def build_semantic_candidate_cache(
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
    """Build a cache containing a symmetric candidate graph and fixed degrees."""
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
    num_nodes = int(text_raw.shape[0])
    if num_nodes < 2:
        raise ValueError("semantic candidate construction needs at least two nodes")
    if int(top_k) != 8:
        raise ValueError("V7A freezes Text and Visual KNN K=8")
    if int(thread_count) < 1:
        raise ValueError("FAISS thread_count must be positive")

    started = time.perf_counter()
    text_source_meta = _source_descriptor(text_source, text_raw)
    visual_source_meta = _source_descriptor(visual_source, visual_raw)
    feature_identity = {
        "dataset": str(dataset),
        "num_nodes": num_nodes,
        "text_dim": int(text_raw.shape[1]),
        "visual_dim": int(visual_raw.shape[1]),
        "text": text_source_meta,
        "visual": visual_source_meta,
    }
    feature_fingerprint = _json_fingerprint(feature_identity)

    text_norm, text_norm_summary = mean_center_l2_normalize(text_raw)
    text_ids, text_sims, text_index_params = _faiss_knn(
        text_norm, k=top_k, seed=seed, thread_count=thread_count
    )
    text_recall = (
        _sample_recall(text_norm, text_ids, k=top_k, seed=seed)
        if not text_index_params["exact"]
        else {"sample_count": 0, "recall_at_k_mean": 1.0, "recall_at_k_min": 1.0}
    )
    text_directed = np.stack(
        [np.repeat(np.arange(num_nodes, dtype=np.int64), top_k), text_ids.reshape(-1)],
        axis=1,
    )
    text_undirected = _sorted_unique_pairs(text_directed, num_nodes)
    text_similarity = text_sims.reshape(-1)
    del text_norm, text_ids, text_sims, text_directed

    visual_norm, visual_norm_summary = mean_center_l2_normalize(visual_raw)
    visual_ids, visual_sims, visual_index_params = _faiss_knn(
        visual_norm, k=top_k, seed=seed, thread_count=thread_count
    )
    visual_recall = (
        _sample_recall(visual_norm, visual_ids, k=top_k, seed=seed)
        if not visual_index_params["exact"]
        else {"sample_count": 0, "recall_at_k_mean": 1.0, "recall_at_k_min": 1.0}
    )
    visual_directed = np.stack(
        [np.repeat(np.arange(num_nodes, dtype=np.int64), top_k), visual_ids.reshape(-1)],
        axis=1,
    )
    visual_undirected = _sorted_unique_pairs(visual_directed, num_nodes)
    visual_similarity = visual_sims.reshape(-1)
    del visual_norm, visual_ids, visual_sims, visual_directed

    union_undirected = _sorted_unique_pairs(
        np.concatenate([text_undirected, visual_undirected], axis=0), num_nodes
    )
    reverse = union_undirected[:, ::-1]
    symmetric_pairs = np.concatenate([union_undirected, reverse], axis=0)
    # Keep the deterministic layout `[u->v for u<v] + [v->u]`, aligned with
    # unique_edge_index, so the model can reuse this cache without rebuilding E.
    edge_index = torch.from_numpy(symmetric_pairs.T.copy()).long().contiguous()
    unique_edge_index = torch.from_numpy(union_undirected.T.copy()).long().contiguous()
    degree_np = np.bincount(
        union_undirected.reshape(-1), minlength=num_nodes
    ).astype(np.int64, copy=False)
    degree = torch.from_numpy(degree_np.copy()).long().contiguous()
    physical = canonical_undirected_edges(physical_edge_index, num_nodes)
    physical_edge_fingerprint = _sha256_bytes(
        np.ascontiguousarray(physical, dtype=np.int64).view(np.uint8).tobytes()
    )
    physical_codes = physical[:, 0] * num_nodes + physical[:, 1]
    candidate_codes = union_undirected[:, 0] * num_nodes + union_undirected[:, 1]
    physical_overlap = int(
        np.intersect1d(candidate_codes, physical_codes, assume_unique=True).size
    )
    text_codes = text_undirected[:, 0] * num_nodes + text_undirected[:, 1]
    visual_codes = visual_undirected[:, 0] * num_nodes + visual_undirected[:, 1]
    modality_intersection = int(
        np.intersect1d(text_codes, visual_codes, assume_unique=True).size
    )
    modality_union = len(text_codes) + len(visual_codes) - modality_intersection

    index_provenance = {
        "text": text_index_params,
        "visual": visual_index_params,
        "faiss_version": text_index_params["faiss_version"],
        "thread_count": int(thread_count),
        "random_seed": int(seed),
    }
    cache_metadata: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": str(dataset),
        "num_nodes": num_nodes,
        "text_dim": int(text_raw.shape[1]),
        "visual_dim": int(visual_raw.shape[1]),
        "rule_version": RULE_VERSION,
        "feature_identity": feature_identity,
        "feature_fingerprint": feature_fingerprint,
        "top_k": int(top_k),
        "candidate_sources": ["text_knn", "visual_knn"],
        "symmetrization": "union_then_canonical_deduplicate_then_add_reverse",
        "index_provenance": index_provenance,
        "text_normalization": text_norm_summary,
        "visual_normalization": visual_norm_summary,
        "directed_candidate_edge_count": int(edge_index.size(1)),
        "undirected_candidate_edge_count": int(unique_edge_index.size(1)),
        "text_undirected_edge_count": int(len(text_undirected)),
        "visual_undirected_edge_count": int(len(visual_undirected)),
        "text_visual_overlap_count": int(modality_intersection),
        "text_visual_overlap_jaccard": float(
            modality_intersection / max(modality_union, 1)
        ),
        "physical_edge_count": int(len(physical)),
        "physical_edge_fingerprint": physical_edge_fingerprint,
        "candidate_physical_overlap_count": int(physical_overlap),
        "candidate_physical_overlap_ratio": float(
            physical_overlap / max(len(union_undirected), 1)
        ),
        "candidate_nonphysical_ratio": float(
            1.0 - physical_overlap / max(len(union_undirected), 1)
        ),
        "isolated_node_count": int((degree_np == 0).sum()),
        "covered_node_count": int((degree_np > 0).sum()),
        "degree_quantiles": {
            str(q): float(np.quantile(degree_np, q / 100.0))
            for q in (0, 10, 25, 50, 75, 90, 99, 100)
        },
        "similarity_quantiles": {
            modality: {
                str(q): float(np.quantile(values, q / 100.0))
                for q in (0, 10, 25, 50, 75, 90, 99, 100)
            }
            for modality, values in (
                ("text", text_similarity),
                ("visual", visual_similarity),
            )
        },
        "retrieval_recall_sample": {"text": text_recall, "visual": visual_recall},
        "construction_seconds": float(time.perf_counter() - started),
    }
    cache_metadata["candidate_fingerprint"] = _tensor_fingerprint(
        edge_index, degree, unique_edge_index
    )
    cache_metadata["cache_fingerprint"] = _json_fingerprint(
        {key: value for key, value in cache_metadata.items() if key != "construction_seconds"}
    )
    cache_name = f"{dataset}_{feature_fingerprint[:12]}_{cache_metadata['candidate_fingerprint'][:12]}.pt"
    cache_path = Path(cache_dir) / cache_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {
            "metadata": cache_metadata,
            "edge_index": edge_index,
            "unique_edge_index": unique_edge_index,
            "degree": degree,
        },
        temporary,
    )
    os.replace(temporary, cache_path)
    return cache_path, cache_metadata


def load_semantic_candidate_cache(
    cache_path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_feature_fingerprint: str | None = None,
    expected_candidate_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Load a CPU candidate cache and validate its self-fingerprint and provenance."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"metadata", "edge_index", "unique_edge_index", "degree"} <= payload.keys():
        raise ValueError(f"invalid PSCE semantic cache payload: {cache_path}")
    metadata = payload["metadata"]
    if int(metadata.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError("semantic cache schema version mismatch")
    if metadata.get("rule_version") != RULE_VERSION:
        raise ValueError("semantic candidate rule version mismatch")
    if int(metadata.get("top_k", -1)) != 8:
        raise ValueError("V7A semantic cache must use K=8 for both modalities")
    if metadata.get("candidate_sources") != ["text_knn", "visual_knn"]:
        raise ValueError("semantic cache candidate sources do not match the V7A rule")
    if metadata.get("symmetrization") != "union_then_canonical_deduplicate_then_add_reverse":
        raise ValueError("semantic cache symmetrization rule does not match V7A")
    if _json_fingerprint(metadata.get("feature_identity", {})) != metadata.get(
        "feature_fingerprint"
    ):
        raise ValueError("semantic cache feature metadata fingerprint mismatch")
    if expected_dataset is not None and metadata.get("dataset") != expected_dataset:
        raise ValueError(
            f"semantic cache dataset mismatch: expected {expected_dataset}, got {metadata.get('dataset')}"
        )
    if (
        expected_feature_fingerprint is not None
        and metadata.get("feature_fingerprint") != expected_feature_fingerprint
    ):
        raise ValueError("semantic cache input feature fingerprint mismatch")
    if (
        expected_candidate_fingerprint is not None
        and metadata.get("candidate_fingerprint") != expected_candidate_fingerprint
    ):
        raise ValueError("semantic cache candidate graph fingerprint mismatch")
    edge_index = payload["edge_index"].to(dtype=torch.long).contiguous()
    unique_edge_index = payload["unique_edge_index"].to(dtype=torch.long).contiguous()
    degree = payload["degree"].to(dtype=torch.long).contiguous()
    num_nodes = int(metadata["num_nodes"])
    if edge_index.shape[0] != 2 or unique_edge_index.shape[0] != 2:
        raise ValueError("semantic cache edge indices must have shape [2,E]")
    if unique_edge_index.numel():
        if bool((unique_edge_index[0] >= unique_edge_index[1]).any()):
            raise ValueError("semantic cache unique pairs must be canonical u<v edges")
        unique_codes = unique_edge_index[0] * num_nodes + unique_edge_index[1]
        if not bool((unique_codes[1:] > unique_codes[:-1]).all()):
            raise ValueError("semantic cache unique pairs must be sorted and deduplicated")
    if degree.shape != (num_nodes,):
        raise ValueError("semantic cache candidate degree has the wrong shape")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("semantic cache contains an out-of-range node ID")
    if edge_index.numel() and bool((edge_index[0] == edge_index[1]).any()):
        raise ValueError("semantic cache contains self-loops")
    computed_candidate_fingerprint = _tensor_fingerprint(
        edge_index, degree, unique_edge_index
    )
    if computed_candidate_fingerprint != metadata.get("candidate_fingerprint"):
        raise ValueError("semantic cache tensor fingerprint mismatch")
    if edge_index.size(1) != int(metadata["directed_candidate_edge_count"]):
        raise ValueError("semantic cache edge count disagrees with metadata")
    unique_count = unique_edge_index.size(1)
    if edge_index.size(1) != 2 * unique_count:
        raise ValueError("symmetric semantic graph must contain two directions per pair")
    if not torch.equal(edge_index[:, :unique_count], unique_edge_index):
        raise ValueError("semantic cache forward directions do not align with unique pairs")
    if not torch.equal(edge_index[:, unique_count:], unique_edge_index.flip(0)):
        raise ValueError("semantic cache reverse directions do not align with unique pairs")
    if int(degree.sum()) != edge_index.size(1):
        raise ValueError("symmetric semantic degree sum disagrees with directed edge count")
    observed_degree = torch.bincount(edge_index[0], minlength=num_nodes)
    if not torch.equal(observed_degree, degree):
        raise ValueError("semantic candidate degree does not match edge index")
    expected_metadata_fingerprint = _json_fingerprint(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"construction_seconds", "cache_fingerprint"}
        }
    )
    if expected_metadata_fingerprint != metadata.get("cache_fingerprint"):
        raise ValueError("semantic cache metadata fingerprint mismatch")
    return {
        "metadata": metadata,
        "edge_index": edge_index,
        "unique_edge_index": unique_edge_index,
        "degree": degree,
    }
