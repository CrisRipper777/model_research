"""Shared setup and provenance helpers for the final NC5/LP2 runners."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from src.data.semantic_candidates import canonical_undirected_edges


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "research/mrcge_v13_nc5_lp2/data/fingerprints.json"
OUTPUT_ROOT = ROOT / "outputs/mrcge_final_benchmark"
SEEDS = (42, 43, 44)
NC_DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
LP_DATASETS = ("sports-copurchase", "cloth-copurchase")
MODEL_VERSION = "MR-CGE Final (V1.3-A Full)"
SOURCE_COMMIT = "4177ec8aec4e05ad513c0357f36da7b926a5c040"
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def json_write(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_prepared_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"prepared V7B.1 fingerprint manifest is missing: {MANIFEST_PATH}")
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not payload.get("prepared_phases", {}).get("nc", {}).get("complete"):
        raise RuntimeError("NC data/cache preparation is not marked complete")
    if not payload.get("prepared_phases", {}).get("lp", {}).get("complete"):
        raise RuntimeError("LP data/cache preparation is not marked complete")
    return payload


def _load_yaml(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return OmegaConf.load(path)


def build_config(phase: str, dataset: str, seed: int, device: str, record: dict[str, Any]):
    """Compose an explicit config and pin its split path to the prepared record."""
    if phase not in {"nc", "lp"}:
        raise ValueError(f"unsupported phase {phase!r}")
    cfg = _load_yaml(ROOT / "configs/config.yaml")
    cfg.dataset = _load_yaml(ROOT / f"configs/dataset/{dataset}.yaml")
    cfg.model = _load_yaml(ROOT / "configs/model/mrcge_final.yaml")
    if phase == "lp":
        adapter = _load_yaml(ROOT / "configs/model/mrcge_final_sampled.yaml")
        if "defaults" in adapter:
            del adapter["defaults"]
        cfg.model = OmegaConf.merge(cfg.model, adapter)
    cfg.task = _load_yaml(ROOT / f"configs/task/mrcge_final_{phase}_benchmark.yaml")
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.device = str(device)
    cfg.paths.output_root = str(OUTPUT_ROOT)

    # Resolve dataset feature/graph paths while the complete root config is attached.
    OmegaConf.resolve(cfg.dataset)
    split = record["split"]
    split_path = str(Path(split["path"]).expanduser().resolve())
    if phase == "nc":
        split_key = "node_split_path" if str(cfg.dataset.source).lower() == "mmgraph" else "nc_split_path"
        cfg.dataset[split_key] = split_path
    else:
        cfg.dataset.edge_split_path = split_path
    cfg.dataset.auto_generate_splits = False

    cfg.model.semantic_cache_path = str(Path(record["semantic_cache_path"]).resolve())
    cfg.model.semantic_feature_fingerprint = str(record["feature_fingerprint"])
    cfg.model.semantic_candidate_fingerprint = str(record["candidate_fingerprint"])
    cfg.model.semantic_v7b_fingerprint = str(record["v7b_fingerprint"])
    cfg.task.evaluate_test = False
    return cfg


def resolved_config_yaml(cfg) -> str:
    snapshot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if "hydra" in snapshot:
        del snapshot["hydra"]
    OmegaConf.resolve(snapshot)
    return OmegaConf.to_yaml(snapshot, resolve=True)


def verify_prepared_inputs(cfg, record: dict[str, Any]) -> dict[str, Any]:
    split_path = Path(cfg.dataset.node_split_path if cfg.task.name == "nc" and cfg.dataset.source == "mmgraph"
                      else cfg.dataset.nc_split_path if cfg.task.name == "nc"
                      else cfg.dataset.edge_split_path)
    expected_split_sha = str(record["split"]["sha256"])
    if not split_path.is_file() or sha256_file(split_path) != expected_split_sha:
        raise RuntimeError(f"official/fixed split missing or changed: {split_path}")

    if str(cfg.dataset.source).lower() == "magb":
        feature_paths = [
            (Path(cfg.dataset.text_feat_path), str(record["feature_file_sha256_text"])),
            (Path(cfg.dataset.image_feat_path), str(record["feature_file_sha256_visual"])),
        ]
        source_path = Path(cfg.dataset.graph_path)
    else:
        feature_paths = [(Path(cfg.dataset.joint_feat_path), str(record["feature_file_sha256_text"]))]
        source_path = Path(cfg.dataset.edge_path if cfg.task.name == "nc" else cfg.dataset.edge_split_path)

    for feature_path, expected in feature_paths:
        if not feature_path.is_file() or sha256_file(feature_path) != expected:
            raise RuntimeError(f"feature file missing or fingerprint changed: {feature_path}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if cfg.task.name == "nc" and str(cfg.dataset.source).lower() == "mmgraph":
        label_path = Path(cfg.dataset.label_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)

    cache_path = Path(cfg.model.semantic_cache_path)
    if not cache_path.is_file() or sha256_file(cache_path) != str(record["semantic_cache_sha256"]):
        raise RuntimeError(f"V7B.1 semantic cache missing or SHA256 changed: {cache_path}")
    metadata = record.get("cache_metadata", {})
    if metadata.get("cache_contains_labels") is not False or metadata.get("candidate_phase_label_access") is not False:
        raise RuntimeError("prepared semantic cache provenance is not label-free")
    if int(record.get("semantic_knn_k", -1)) != 8:
        raise RuntimeError("prepared semantic cache does not use frozen K=8")
    if cfg.task.name == "lp" and record.get("validation_test_edges_used_for_semantics") is not False:
        raise RuntimeError("LP semantic cache provenance includes Validation/Test edges")
    return {
        "split_path": str(split_path),
        "split_sha256": expected_split_sha,
        "split_fingerprint": str(record["split"].get("split_fingerprint", expected_split_sha)),
        "feature_fingerprint": str(record["feature_fingerprint"]),
        "feature_files": [{"path": str(path), "sha256": expected} for path, expected in feature_paths],
        "physical_edge_fingerprint_expected": str(record["physical_edge_fingerprint"]),
        "semantic_cache_path": str(cache_path),
        "semantic_cache_sha256": str(record["semantic_cache_sha256"]),
        "candidate_fingerprint": str(record["candidate_fingerprint"]),
        "v7b_fingerprint": str(record["v7b_fingerprint"]),
    }


def verify_loaded_physical_graph(data, record: dict[str, Any]) -> str:
    canonical = canonical_undirected_edges(data.edge_index, data.num_nodes)
    digest = hashlib.sha256(np.ascontiguousarray(canonical, dtype=np.int64).view(np.uint8).tobytes()).hexdigest()
    expected = str(record["physical_edge_fingerprint"])
    if digest != expected:
        raise RuntimeError(f"loaded Train/NC physical graph fingerprint mismatch: expected {expected}, got {digest}")
    return digest


def next_attempt_dir(phase: str, dataset: str, seed: int) -> tuple[Path, int, Path | None]:
    parent = OUTPUT_ROOT / phase / dataset / f"seed_{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    complete = []
    attempt_ids = []
    uncertain_test_attempts = []

    def _valid_saved_result(path: Path, status: dict[str, Any]) -> bool:
        result_path = path / "result.json"
        if not result_path.is_file():
            return False
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        test_count = result.get("test_evaluation_count", result.get("test", {}).get("evaluation_count"))
        checkpoint = Path(result.get("checkpoint_path", ""))
        return (
            test_count == 1
            and result.get("test_evaluated_after_checkpoint_freeze_and_reload") is True
            and checkpoint.is_file()
            and sha256_file(checkpoint) == result.get("checkpoint_sha256")
            and status.get("checkpoint_sha256") == result.get("checkpoint_sha256")
        )

    for path in sorted(parent.glob("attempt_*")):
        try:
            attempt_ids.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
        status_path = path / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("status") == "complete":
            if not _valid_saved_result(path, status):
                raise RuntimeError(f"completed result/checkpoint integrity failed in {path}")
            complete.append(path)
            continue
        if status.get("test_evaluation_started"):
            if _valid_saved_result(path, status):
                status.update({"status": "complete", "recovered_after_result_write": True})
                json_write(status_path, status)
                complete.append(path)
            else:
                uncertain_test_attempts.append(path)
            continue
    if len(complete) > 1:
        raise RuntimeError(f"multiple complete attempts exist for {phase}/{dataset}/seed{seed}")
    if complete:
        return parent, 0, complete[0]
    if uncertain_test_attempts:
        raise RuntimeError(
            f"Test evaluation may already have run for {phase}/{dataset}/seed{seed} in "
            f"{uncertain_test_attempts[0]}; refusing an automatic second Test evaluation"
        )
    attempt = max(attempt_ids, default=0) + 1
    output_dir = parent / f"attempt_{attempt:02d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return parent, attempt, output_dir


def create_logger(output_dir: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def data_info(data) -> dict[str, Any]:
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
