#!/usr/bin/env python3
"""Default-deny final Test evaluator for an independently approved model lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from scripts import run_hybrid_cge_paper_evidence_v1 as study
from scripts import run_hybrid_cge_validation_expansion_v1 as prior_study
from src.data import load_mag_data
from src.models import build_model
from src.utils.seeds import set_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "research/hybrid_cge_paper_evidence_v1/FINAL_MODEL_LOCK.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authorize(lock_path: Path, dataset: str, *, explicit_final_test: bool) -> dict:
    """Check every authorization and immutable artifact before loading data."""
    if not explicit_final_test:
        raise PermissionError("final Test is disabled; pass --final-test after independent approval")
    if not lock_path.is_file():
        raise PermissionError(f"approved final model lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "approved":
        raise PermissionError(f"final model lock status is {lock.get('status')!r}; approved is required")
    entry = lock.get("models", {}).get(dataset)
    if not isinstance(entry, dict):
        raise PermissionError(f"no approved locked checkpoint for dataset {dataset}")
    required = (
        "model_name", "condition", "seed", "source_git_sha", "config_snapshot_path",
        "config_sha256", "model_source_path", "model_source_sha256", "checkpoint_path",
        "checkpoint_sha256",
    )
    missing = [key for key in required if key not in entry]
    if missing:
        raise PermissionError(f"model lock is incomplete for {dataset}: {missing}")
    config_path = Path(entry["config_snapshot_path"])
    source_path = Path(entry["model_source_path"])
    checkpoint_path = Path(entry["checkpoint_path"])
    for label, path, expected_hash in (
        ("config", config_path, entry["config_sha256"]),
        ("model source", source_path, entry["model_source_sha256"]),
        ("checkpoint", checkpoint_path, entry["checkpoint_sha256"]),
    ):
        if not path.is_file():
            raise PermissionError(f"locked {label} artifact is missing: {path}")
        if sha256_file(path) != expected_hash:
            raise PermissionError(f"locked {label} SHA256 mismatch: {path}")
    try:
        snapshot = OmegaConf.load(config_path)
    except Exception as exc:
        raise PermissionError(f"locked configuration snapshot cannot be parsed: {config_path}") from exc
    if str(snapshot.model.name) != str(entry["model_name"]):
        raise PermissionError("locked configuration model name differs from the approval record")
    if int(snapshot.seed) != int(entry["seed"]):
        raise PermissionError("locked configuration seed differs from the approval record")
    if str(entry["model_name"]) == study.PAPER_MODEL:
        if str(snapshot.model.paper_condition) != str(entry["condition"]):
            raise PermissionError("locked Hybrid-CGE condition differs from the approval record")
    return {"lock": lock, "entry": entry}


def _build_locked_cfg(entry: dict, dataset: str, cache: dict | None):
    model_name = str(entry["model_name"])
    condition = str(entry["condition"])
    seed = int(entry["seed"])
    if model_name == "hybrid_cge_v11":
        cfg = prior_study.build_cfg(dataset, model_name, seed, "cpu", condition, cache)
    else:
        cfg = study.build_cfg(dataset, model_name, seed, "cpu", condition, cache)
    return cfg


def run_locked_test(lock_path: Path, dataset: str, *, explicit_final_test: bool) -> dict:
    permission = authorize(lock_path, dataset, explicit_final_test=explicit_final_test)
    entry = permission["entry"]
    seed = int(entry["seed"])
    model_name = str(entry["model_name"])
    cache = study.prep_model_and_cache(dataset, model_name)
    cfg = _build_locked_cfg(entry, dataset, cache)
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)

    # Loading labels is deliberately after the explicit flag and approved-lock
    # checks. This process never constructs an optimizer or calls a trainer.
    raw = load_mag_data(cfg, "nc", seed)
    data = raw
    if cache is None:
        cache = None
    set_seed(seed)
    model = build_model(cfg, study.info_for(data))
    if bool(getattr(model, "requires_global_semantic_candidates", False)):
        if cache is None:
            raise RuntimeError("locked model requires a semantic cache")
        model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
    head = torch.nn.Linear(model.out_dim, int(data.num_classes))
    payload = torch.load(entry["checkpoint_path"], map_location="cpu", weights_only=False)
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError("locked checkpoint seed differs from the approved model lock")
    provenance = payload.get("run_metadata", {}).get("study_provenance", {})
    if provenance.get("source_git_sha") != entry["source_git_sha"]:
        raise RuntimeError("locked checkpoint source commit differs from the approved model lock")
    expected_training_config = entry.get("checkpoint_config_sha256")
    if expected_training_config and provenance.get("config_sha256") != expected_training_config:
        raise RuntimeError("locked checkpoint training configuration differs from the approved model lock")
    model.load_state_dict(payload["model_state"], strict=True)
    head.load_state_dict(payload["head_state"], strict=True)
    model, head = model.to(device).eval(), head.to(device).eval()
    x = data.x.to(device)
    edges = data.edge_index.to(device)
    with torch.inference_mode():
        z, _, _, _, _ = model(x, edges)
        logits = head(z).detach().cpu().float()
    indices = data.test_idx.detach().cpu().long()
    target = data.y[indices].detach().cpu().long()
    if target.numel() == 0 or bool((target < 0).any()):
        raise RuntimeError("final Test split is empty or contains masked labels")
    prediction = logits[indices].argmax(dim=-1)
    f1_labels = list(range(int(data.num_classes)))
    metrics = {
        "accuracy": float((prediction == target).float().mean()),
        "macro_f1": float(f1_score(target.numpy(), prediction.numpy(), labels=f1_labels,
                                    average="macro", zero_division=0)),
        "f1_label_set": f1_labels,
    }
    if not all(torch.isfinite(torch.tensor(value)) for key, value in metrics.items() if key != "f1_label_set"):
        raise RuntimeError("final Test metrics are non-finite")
    fingerprints = study.gpu_guard.dataset_fingerprints(raw)
    if entry.get("feature_fingerprint") and entry["feature_fingerprint"] != fingerprints["features"]:
        raise RuntimeError("locked model feature fingerprint differs from loaded data")
    if entry.get("physical_edge_fingerprint") and entry["physical_edge_fingerprint"] != fingerprints["physical_edge_index"]:
        raise RuntimeError("locked model physical graph fingerprint differs from loaded data")

    record = {
        "dataset": dataset, "model_name": model_name, "condition": entry["condition"], "seed": seed,
        "source_git_sha": entry["source_git_sha"], "config_sha256": entry["config_sha256"],
        "checkpoint_sha256": entry["checkpoint_sha256"], "feature_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"], "metrics": metrics,
        "test_evaluation": True, "training_performed": False,
        "approved_lock_sha256": sha256_file(lock_path),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output = ROOT / "research/hybrid_cge_paper_evidence_v1/data/final_test_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(output.read_text()) if output.is_file() else {"evaluations": []}
    previous["evaluations"].append(record)
    output.write_text(json.dumps(previous, indent=2, sort_keys=True), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-test", action="store_true", help="explicitly authorize final Test evaluation")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--dataset", required=True, choices=study.DATASETS)
    args = parser.parse_args()
    result = run_locked_test(args.lock, args.dataset, explicit_final_test=args.final_test)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
