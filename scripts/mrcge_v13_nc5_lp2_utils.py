"""Shared configuration, integrity, and serialization helpers for NC5/LP2."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/mrcge_v13_nc5_lp2"
DATA_DIR = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_nc5_lp2"
BASE_SHA = "4bb7253ac9dffeca57686771c139002d6e84a19f"
BASE_BRANCH = "exp/mrcge_v13_stage_ablation_3seed_completion"
BRANCH = "exp/mrcge_v13_nc5_lp2_fullgraph"
NC_DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
LP_DATASETS = ("sports-copurchase", "cloth-copurchase")
SEEDS = (42, 43, 44)
NEW_NC_DATASETS = ("Toys", "Reddit-S")
HISTORICAL_NC_DATASETS = ("Movies", "Grocery", "ele-fashion")
PROTOCOL_NC = "unified_full_graph_nc_v1"
PROTOCOL_LP = "mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def json_write(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default), encoding="utf-8")
    temp.replace(path)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def csv_read(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def current_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def build_config(dataset: str, phase: str, seed: int):
    if phase not in {"nc", "lp"}:
        raise ValueError(f"phase must be nc|lp, got {phase!r}")
    task_name = phase
    root_cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    dataset_cfg = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    task_cfg = OmegaConf.load(ROOT / f"configs/task/{task_name}.yaml")
    model_cfg = OmegaConf.load(ROOT / "configs/model/mrcge_v13.yaml")
    cfg = OmegaConf.create({
        "paths": root_cfg.paths, "dataset": dataset_cfg, "task": task_cfg,
        "model": model_cfg, "logging": root_cfg.logging,
        "seed": 42, "num_runs": 1, "device": "cpu",
    })
    cfg.dataset.name = dataset
    cfg.task.name = task_name
    cfg.model.name = "mrcge_v13"
    cfg.model.condition = "M_repaired"
    cfg.model.v13_variant = "joint_view"
    cfg.model.sender_private_ratio = 0.0
    # Resolve the interpolated NC Split while the config seed is fixed at 42,
    # then set the independent model RNG seed without changing the split path.
    if phase == "nc":
        cfg.dataset.auto_generate_splits = False
        OmegaConf.resolve(cfg)
        split_key = "nc_split_path" if "nc_split_path" in cfg.dataset else "node_split_path"
        split_path = str(cfg.dataset[split_key])
        cfg.dataset[split_key] = split_path
    cfg.seed = int(seed)
    if phase == "nc":
        cfg.task.protocol_version = PROTOCOL_NC
        cfg.task.training_mode = "full_graph"
        cfg.task.optimizer = "adamw"
        cfg.task.lr = 1.0e-3
        cfg.task.weight_decay = 1.0e-4
        cfg.task.epochs = 300
        cfg.task.patience = 30
        cfg.task.early_stop_min_epoch = 30
        cfg.task.early_stop_min_delta = 1.0e-4
        cfg.task.grad_clip = 1.0
        cfg.task.eval_every = 1
        cfg.task.inference_mode = "full"
        cfg.task.inference_batch_size = 4096
        cfg.task.scheduler = None
        cfg.task.evaluate_test = False
        cfg.task.loss.aux_weight = 1.0
        cfg.task.loss.unimodal_weight = 0.3
        cfg.task.save_ckpt_path = None
        cfg.task.best_f1_ckpt_path = None
        cfg.task.record_extended_validation = True
        cfg.task.record_train_ce_separately = True
    else:
        cfg.dataset.make_undirected = True
        cfg.dataset.add_self_loops = False
        cfg.task.protocol_version = PROTOCOL_LP
        cfg.task.training_mode = "full_graph_three_fold_edge_holdout"
        cfg.task.allow_official_split_conflicts = True
        cfg.task.optimizer = "adam"
        cfg.task.lr = 1.0e-3
        cfg.task.weight_decay = 1.0e-5
        cfg.task.epochs = 150
        cfg.task.patience = 10
        cfg.task.early_stop_min_epoch = 20
        cfg.task.early_stop_min_delta = 1.0e-4
        cfg.task.grad_clip = 1.0
        cfg.task.eval_every = 2
        cfg.task.inference_mode = "full"
        cfg.task.inference_batch_size = 2048
        cfg.task.eval_edge_batch_size = 2048
        cfg.task.supervision_edge_batch_size = 2048
        cfg.task.num_train_neg = 1
        cfg.task.loss.aux_weight = 1.0
        cfg.task.decoder.name = "mlp"
        cfg.task.decoder.proj_dim = 128
        cfg.task.decoder.hidden_dim = 256
        cfg.task.decoder.num_layers = 3
        cfg.task.decoder.dropout = 0.02
        cfg.task.evaluate_test = False
        cfg.task.save_ckpt_path = None
        cfg.model.semantic_knn_k = 8
    cfg.paths.output_root = str(OUTPUT)
    OmegaConf.resolve(cfg)
    return cfg


def config_snapshot(cfg) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)


def mask_test_labels(data):
    """Return a shallow data copy whose Test target labels are inaccessible to the trainer."""
    import copy

    if data.y is None or data.test_idx is None:
        raise ValueError("NC data must contain labels and test_idx before masking")
    safe = copy.copy(data)
    safe.y = data.y.clone()
    safe.y[safe.test_idx] = -100
    if safe.train_idx is not None and not torch.equal(safe.y[safe.train_idx], data.y[safe.train_idx]):
        raise RuntimeError("Test masking changed Train labels")
    if safe.val_idx is not None and not torch.equal(safe.y[safe.val_idx], data.y[safe.val_idx]):
        raise RuntimeError("Test masking changed Validation labels")
    if bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError("Test labels remain visible after masking")
    return safe


def assert_clip_feature_order(x: torch.Tensor, x_t: torch.Tensor, x_i: torch.Tensor) -> None:
    """Enforce the MM-Graph convention x=[Text(0:512), Visual(512:1024)]."""
    if x.ndim != 2 or x_t.ndim != 2 or x_i.ndim != 2:
        raise ValueError("CLIP joint and modality features must be rank-2 matrices")
    if x.shape[1] != 1024 or x_t.shape != (x.shape[0], 512) or x_i.shape != (x.shape[0], 512):
        raise ValueError(f"expected joint/Text/Visual widths 1024/512/512, got {x.shape}/{x_t.shape}/{x_i.shape}")
    if not torch.equal(x[:, :512], x_t) or not torch.equal(x[:, 512:], x_i):
        raise ValueError("MM-Graph CLIP feature order mismatch; expected Text first and Visual second")


def loaded_lp_split_path(info: dict[str, Any]) -> str | None:
    """Normalize loader metadata names for MAGB and MM-Graph LP splits."""
    value = info.get("lp_split_path") or info.get("edge_split_path")
    return str(value) if value else None
