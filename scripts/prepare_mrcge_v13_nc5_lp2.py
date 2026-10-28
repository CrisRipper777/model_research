#!/usr/bin/env python3
"""Prepare fixed NC splits and verified V7B.1 caches for MR-CGE V1.3-A NC5/LP2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mrcge_v13_nc5_lp2_utils import (  # noqa: E402
    DATA_DIR, LP_DATASETS, NC_DATASETS, OUTPUT, build_config, json_write,
    sha256_file, tensor_sha256,
)
from src.data.graph_utils import canonicalize_edges, ensure_edge_index, preprocess_edge_index  # noqa: E402
from src.data.loaders import resolve_path  # noqa: E402
from src.data.semantic_candidates import _source_descriptor  # noqa: E402
from src.data.semantic_candidates_v7b import (  # noqa: E402
    _digest_json, build_semantic_candidate_cache_v7b, load_semantic_candidate_cache_v7b,
)
from src.data.splits import generate_nc_split  # noqa: E402
from src.tasks.mrcge_v13_full_graph_lp import train_only_physical_graph, validate_lp_split  # noqa: E402
from src.tasks.mrcge_v13_full_graph_lp import LPSplitConflictError  # noqa: E402


def _root_cfg(dataset: str):
    root = OmegaConf.load(ROOT / "configs/config.yaml")
    ds = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    cfg = OmegaConf.create({"paths": root.paths, "dataset": ds, "seed": 42})
    OmegaConf.resolve(cfg)
    return cfg


def _load_feature_graph(dataset: str, phase: str) -> tuple[np.ndarray, np.ndarray, torch.Tensor, dict[str, Any]]:
    cfg = _root_cfg(dataset)
    ds = cfg.dataset
    if phase == "nc":
        if str(ds.source).lower() == "magb":
            import dgl

            graph_path = resolve_path(ds.graph_path)
            graphs, _ = dgl.load_graphs(str(graph_path))
            if not graphs:
                raise ValueError(f"{dataset}: no graph stored at {graph_path}")
            graph = graphs[0]
            source, target = graph.edges()
            edge_index = torch.stack((source.long(), target.long()), dim=0).contiguous()
            num_nodes = int(graph.num_nodes())
            text_path, visual_path = resolve_path(ds.text_feat_path), resolve_path(ds.image_feat_path)
            text = np.load(text_path, allow_pickle=False, mmap_mode="r")
            visual = np.load(visual_path, allow_pickle=False, mmap_mode="r")
            sources = {"text": text_path, "visual": visual_path}
            if text.shape[0] != num_nodes or visual.shape[0] != num_nodes:
                raise ValueError(f"{dataset}: feature row count does not equal graph node count")
            source_name = "magb"
        else:
            feature_path = resolve_path(ds.joint_feat_path)
            edge_path = resolve_path(ds.edge_path)
            features = torch.load(feature_path, map_location="cpu", weights_only=False).float().contiguous()
            edge_index = ensure_edge_index(torch.load(edge_path, map_location="cpu", weights_only=False))
            num_nodes = int(features.size(0))
            text_dim, visual_dim = int(ds.text_dim), int(ds.visual_dim)
            if features.ndim != 2 or features.size(1) != text_dim + visual_dim:
                raise ValueError(f"{dataset}: joint feature dimensions differ from the configured modality split")
            text = features[:, :text_dim].numpy()
            visual = features[:, text_dim:text_dim + visual_dim].numpy()
            sources = {"text": feature_path, "visual": feature_path}
            source_name = "mmgraph"
        physical = preprocess_edge_index(edge_index, num_nodes, bool(ds.get("make_undirected", True)), False)
        attrs = {"num_nodes": num_nodes, "raw_edges": int(edge_index.size(1)),
                 "text_source": str(sources["text"]), "visual_source": str(sources["visual"]),
                 "source": source_name, "split_path": str(_fixed_nc_split_path(dataset))}
    else:
        cfg_lp = build_config(dataset, "lp", 42)
        feature_path = resolve_path(cfg_lp.dataset.joint_feat_path)
        features = torch.load(feature_path, map_location="cpu", weights_only=False).float().contiguous()
        text_dim = int(cfg_lp.dataset.text_dim)
        visual_dim = int(cfg_lp.dataset.visual_dim)
        if features.ndim != 2 or features.size(1) != text_dim + visual_dim:
            raise ValueError(f"{dataset}: joint feature width does not match text_dim+visual_dim")
        if (text_dim, visual_dim) != (512, 512):
            raise ValueError(f"{dataset}: expected the configured CLIP 512/512 modality split")
        text = features[:, :text_dim].numpy()
        visual = features[:, text_dim:text_dim + visual_dim].numpy()
        sources = {"text": feature_path, "visual": feature_path}
        split_path = resolve_path(cfg_lp.dataset.edge_split_path)
        if not split_path.is_file():
            raise FileNotFoundError(f"official LP split missing: {split_path}")
        raw = torch.load(split_path, map_location="cpu", weights_only=False)
        from src.data.loaders import _normalize_split_dict
        from src.data.graph_utils import edge_dict_to_index
        edge_split = _normalize_split_dict(raw)
        num_nodes = int(features.size(0))
        audit = validate_lp_split(edge_split, num_nodes, allow_official_conflicts=True)
        physical = train_only_physical_graph(edge_split, num_nodes, allow_official_conflicts=True)
        audit["train_only_graph_source"] = "official lp-edge-split.pt train positives only"
        attrs = {"num_nodes": num_nodes, "raw_edges": int(edge_dict_to_index(edge_split.train).size(1)),
                 "text_source": str(feature_path), "visual_source": str(feature_path), "source": "mmgraph",
                 "split_path": str(split_path), "split_sha256": sha256_file(split_path),
                 "edge_split_audit": audit, "feature_path": str(feature_path)}
        if int(text.shape[1]) != 512 or int(visual.shape[1]) != 512:
            raise ValueError(f"{dataset}: loaded CLIP Text/Visual order must be 512/512")
        # Keep these references out of cache construction; neither Test negatives nor labels are used.
        del raw, edge_split, features
    if text.ndim != 2 or visual.ndim != 2 or text.shape[0] != visual.shape[0] or text.shape[0] != num_nodes:
        raise ValueError(f"{dataset}: invalid Text/Visual arrays {text.shape}/{visual.shape}")
    physical = ensure_edge_index(physical).cpu().long().contiguous()
    if physical.numel() and (int(physical.min()) < 0 or int(physical.max()) >= num_nodes):
        raise ValueError(f"{dataset}: physical graph contains invalid node IDs")
    return np.asarray(text, dtype=np.float32), np.asarray(visual, dtype=np.float32), physical, attrs | {"sources": sources}


def _fixed_nc_split_path(dataset: str) -> Path:
    cfg = build_config(dataset, "nc", 42)
    key = "nc_split_path" if "nc_split_path" in cfg.dataset else "node_split_path"
    return resolve_path(cfg.dataset[key])


def _validate_nc_split(path: Path, num_nodes: int, *, split_seed: int | None = 42,
                       require_full_coverage: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"fixed seed42 NC split is missing: {path}")
    split = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(split, dict):
        raise TypeError(f"NC split must be a dict: {path}")
    names = ("train_idx", "val_idx", "test_idx")
    values = {name: torch.as_tensor(split[name], dtype=torch.long).flatten() for name in names}
    for name, ids in values.items():
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= num_nodes):
            raise ValueError(f"{path}: {name} contains an out-of-range node ID")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError(f"{path}: {name} contains duplicate node IDs")
    a, b, c = (set(values[name].tolist()) for name in names)
    overlap = {"train_val": len(a & b), "train_test": len(a & c), "val_test": len(b & c)}
    if any(overlap.values()):
        raise ValueError(f"{path}: NC partitions overlap: {overlap}")
    covered = a | b | c
    if require_full_coverage and covered != set(range(num_nodes)):
        raise ValueError(f"{path}: NC partitions do not cover exactly all nodes")
    recorded_seed = split.get("seed")
    if split_seed is not None and recorded_seed is not None and int(recorded_seed) != split_seed:
        raise ValueError(f"{path}: expected fixed seed{split_seed} split; payload says seed={recorded_seed}")
    parts = {name: tensor_sha256(torch.sort(ids).values) for name, ids in values.items()}
    fp = _digest_json({"num_nodes": num_nodes, "parts": parts, "split_seed": split_seed,
                       "full_node_coverage": bool(require_full_coverage)})
    return {
        "path": str(path), "sha256": sha256_file(path), "split_fingerprint": fp,
        "split_seed": split_seed if split_seed is not None else "fixed_official",
        "counts": {name: int(ids.numel()) for name, ids in values.items()},
        "partition_fingerprints": parts, "partition_overlap_counts": overlap,
        "covers_all_nodes": covered == set(range(num_nodes)),
    }


def _create_fixed_nc_split(dataset: str, *, allow_create: bool) -> dict[str, Any]:
    path = _fixed_nc_split_path(dataset)
    cfg = _root_cfg(dataset)
    if str(cfg.dataset.source).lower() == "magb":
        # Obtain node count without inspecting graph labels on the cache preparation path.
        import dgl
        graph_path = resolve_path(cfg.dataset.graph_path)
        graphs, _ = dgl.load_graphs(str(graph_path))
        if not graphs:
            raise ValueError(f"no DGL graph stored at {graph_path}")
        graph = graphs[0]
        num_nodes = int(graph.num_nodes())
    else:
        feature_path = resolve_path(cfg.dataset.joint_feat_path)
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
        num_nodes = int(torch.as_tensor(features).size(0))
    if not path.is_file():
        if not allow_create:
            raise FileNotFoundError(
                f"fixed seed42 split not found: {path}; rerun explicitly with --allow-create-fixed-split"
            )
        if str(cfg.dataset.source).lower() != "magb":
            raise RuntimeError(f"{dataset} uses an existing official node split; this tool will not regenerate it")
        labels = graph.ndata.get("label")
        if labels is None:
            raise KeyError(f"{graph_path} does not contain ndata['label']; cannot create official NC split")
        generated = generate_nc_split(
            labels.long().cpu(), float(cfg.dataset.nc_train_ratio), float(cfg.dataset.nc_val_ratio), 42,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(generated, temp)
        temp.replace(path)
    split_is_seeded = "nc_split_path" in cfg.dataset
    return _validate_nc_split(
        path, num_nodes, split_seed=42 if split_is_seeded else None,
        require_full_coverage=split_is_seeded,
    )


def _feature_fingerprint(dataset: str, text: np.ndarray, visual: np.ndarray, sources: dict[str, Path]) -> str:
    identity = {
        "dataset": dataset, "num_nodes": int(text.shape[0]),
        "text_dim": int(text.shape[1]), "visual_dim": int(visual.shape[1]),
        "text": _source_descriptor(sources["text"], text),
        "visual": _source_descriptor(sources["visual"], visual),
    }
    return _digest_json(identity)


def _physical_fingerprint(physical: torch.Tensor, num_nodes: int) -> tuple[str, int]:
    from src.data.semantic_candidates import canonical_undirected_edges
    edges = canonical_undirected_edges(physical, num_nodes)
    digest = hashlib.sha256(np.ascontiguousarray(edges, dtype=np.int64).view(np.uint8).tobytes()).hexdigest()
    return digest, int(len(edges))


def _existing_cache_paths(dataset: str) -> list[Path]:
    directories = (
        ROOT / "outputs/psce_mag_v7b/semantic_cache",
        ROOT / "outputs/hybrid_cge_validation_expansion_v1/semantic_cache",
        OUTPUT / "semantic_cache",
    )
    return sorted({path for directory in directories if directory.is_dir()
                   for path in directory.glob(f"{dataset}_*_v7b.pt")})


def _reuse_matching_cache(dataset: str, feature_fp: str, physical_fp: str) -> tuple[Path, dict[str, Any]] | None:
    for path in _existing_cache_paths(dataset):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {})
            if metadata.get("dataset") != dataset or int(metadata.get("top_k", -1)) != 8:
                continue
            if metadata.get("feature_fingerprint") != feature_fp:
                continue
            if metadata.get("physical_edge_fingerprint") != physical_fp:
                continue
            if metadata.get("candidate_phase_label_access") is not False or metadata.get("cache_contains_labels") is not False:
                continue
            cache = load_semantic_candidate_cache_v7b(
                path, expected_dataset=dataset, expected_feature_fingerprint=feature_fp,
                expected_candidate_fingerprint=metadata.get("candidate_fingerprint"),
                expected_v7b_fingerprint=metadata.get("v7b_fingerprint"),
            )
            return path, cache["metadata"]
        except Exception:
            # A corrupt or mismatched cache is never reused. The V7B builder will create a
            # separate graph-fingerprinted path, so existing research artifacts are untouched.
            continue
    return None


def _prepare_one(dataset: str, phase: str, allow_create_split: bool) -> dict[str, Any]:
    text, visual, physical, attrs = _load_feature_graph(dataset, phase)
    num_nodes = int(text.shape[0])
    split_record = _create_fixed_nc_split(dataset, allow_create=allow_create_split) if phase == "nc" else {
        "path": attrs["split_path"], "sha256": attrs["split_sha256"],
        "split_fingerprint": attrs["split_sha256"], "official": True,
        "audit": attrs["edge_split_audit"],
    }
    feature_fp = _feature_fingerprint(dataset, text, visual, attrs["sources"])
    physical_fp, physical_unique = _physical_fingerprint(physical, num_nodes)
    reuse = _reuse_matching_cache(dataset, feature_fp, physical_fp)
    if reuse is None:
        cache_dir = OUTPUT / "semantic_cache" / f"{phase}_{dataset}_{physical_fp[:12]}"
        cache_path, metadata = build_semantic_candidate_cache_v7b(
            dataset=dataset, text_features=text, visual_features=visual,
            physical_edge_index=physical, cache_dir=cache_dir,
            text_source=attrs["sources"]["text"], visual_source=attrs["sources"]["visual"],
            top_k=8,
        )
        cache_origin = "built_v7b1"
    else:
        cache_path, metadata = reuse
        cache_origin = "reused_hash_verified"
    if metadata.get("feature_fingerprint") != feature_fp or metadata.get("physical_edge_fingerprint") != physical_fp:
        raise RuntimeError(f"{dataset}: prepared semantic cache does not match current feature/graph fingerprints")
    cache_loaded = load_semantic_candidate_cache_v7b(
        cache_path, expected_dataset=dataset, expected_feature_fingerprint=feature_fp,
        expected_candidate_fingerprint=metadata["candidate_fingerprint"],
        expected_v7b_fingerprint=metadata["v7b_fingerprint"],
    )
    file_bytes = cache_path.stat().st_size
    record = {
        "dataset": dataset, "phase": phase, "source": attrs["source"],
        "num_nodes": num_nodes, "physical_edge_count_directed": int(physical.size(1)),
        "physical_edge_count_undirected_unique": physical_unique,
        "text_dim": int(text.shape[1]), "visual_dim": int(visual.shape[1]),
        "input_dim": int(text.shape[1] + visual.shape[1]),
        "feature_source_text": attrs["text_source"], "feature_source_visual": attrs["visual_source"],
        "feature_file_sha256_text": sha256_file(attrs["sources"]["text"]),
        "feature_file_sha256_visual": sha256_file(attrs["sources"]["visual"]),
        "feature_fingerprint": feature_fp, "physical_edge_fingerprint": physical_fp,
        "split": split_record, "semantic_cache_path": str(cache_path.resolve()),
        "semantic_cache_sha256": sha256_file(cache_path), "semantic_cache_bytes": file_bytes,
        "cache_origin": cache_origin, "candidate_fingerprint": metadata["candidate_fingerprint"],
        "v7b_fingerprint": metadata["v7b_fingerprint"], "semantic_knn_k": 8,
        "candidate_phase_label_access": metadata.get("candidate_phase_label_access"),
        "cache_contains_labels": metadata.get("cache_contains_labels"),
        "semantic_edges_undirected_unique": int(cache_loaded["unique_edge_index"].size(1)),
        "semantic_edges_directed": int(cache_loaded["edge_index"].size(1)),
        "cache_metadata": metadata,
    }
    if phase == "lp":
        record.update({
            "semantic_relation_source": "global node attributes only; candidate builder sees Train-positive graph only for overlap/provenance statistics",
            "validation_test_edges_used_for_semantics": False,
            "lp_train_graph_fingerprint": physical_fp,
        })
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("nc", "lp"), required=True)
    parser.add_argument("--allow-create-fixed-split", action="store_true",
                        help="explicitly permit generating a missing NC seed42 split with the existing official generator")
    args = parser.parse_args()
    datasets = NC_DATASETS if args.phase == "nc" else LP_DATASETS
    if args.allow_create_fixed_split and args.phase != "nc":
        parser.error("--allow-create-fixed-split is available only for --phase nc")
    data_file = DATA_DIR / "fingerprints.json"
    payload = json.loads(data_file.read_text()) if data_file.is_file() else {
        "schema_version": 1, "base_sha": "4bb7253ac9dffeca57686771c139002d6e84a19f",
        "v7b_rule": "V7B.1 K=8, source-flagged shared union; candidate builder label-free",
        "datasets": {}, "prepared_phases": {},
    }
    phase_records = {}
    lp_blockers = {}
    for dataset in datasets:
        try:
            phase_records[dataset] = _prepare_one(dataset, args.phase, args.allow_create_fixed_split)
            payload["datasets"][dataset] = phase_records[dataset]
            print(f"PREPARED {args.phase}/{dataset}: nodes={phase_records[dataset]['num_nodes']} "
                  f"physical={phase_records[dataset]['physical_edge_count_directed']} "
                  f"semantic={phase_records[dataset]['semantic_edges_directed']} "
                  f"cache={phase_records[dataset]['cache_origin']}", flush=True)
        except LPSplitConflictError as exc:
            lp_cfg = build_config(dataset, "lp", 42)
            split_path = resolve_path(lp_cfg.dataset.edge_split_path)
            lp_blockers[dataset] = {
                "dataset": dataset, "split_path": str(split_path), "split_sha256": sha256_file(split_path),
                "protocol_conflict": True, "audit": exc.audit, "error": str(exc),
                "cache_build_skipped": True,
            }
            print(f"BLOCKED lp/{dataset}: {exc}", flush=True)
            continue
    if args.phase == "lp":
        audits = {}
        for dataset in datasets:
            record = phase_records.get(dataset)
            if record is None:
                continue
            audits[dataset] = {
                "split_path": record["split"]["path"],
                "split_sha256": record["split"]["sha256"],
                "audit": record["split"]["audit"],
            }
        has_conflicts = any(item["audit"].get("protocol_conflict") for item in audits.values())
        payload["prepared_phases"][args.phase] = {
            "completed_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "datasets": list(datasets), "complete": len(phase_records) == len(datasets),
            "protocol_version": "mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is",
            "official_split_conflicts_accepted_as_is": has_conflicts,
            "blockers": sorted(lp_blockers),
            "reason": "official split files were used unchanged; overlap/self-loop effects are documented as a protocol deviation",
        }
        json_write(DATA_DIR / "lp_split_preflight_audit.json", {
            "protocol_version": "mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is",
            "status": "official_split_as_is_conflicts_accepted" if has_conflicts else "official_split_as_is_no_conflicts",
            "test_metrics": "not_evaluated", "split_files_modified": False,
            "datasets": audits,
            "interpretation": (
                "Validation/Test positives overlapping Train positives are retained as supplied. Validation can therefore "
                "score edges seen during Train supervision or Train message passing; results are not leakage-safe and are "
                "reported as official-split-as-is protocol-deviation results. Self-loop positives remain in edge supervision "
                "and are omitted from message passing by the repository's standard graph preprocessing."
                if has_conflicts else "No split conflicts were detected."
            ),
        })
    else:
        payload["prepared_phases"][args.phase] = {
            "completed_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "datasets": list(datasets), "complete": True,
        }
    json_write(data_file, payload)
    if lp_blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
