#!/usr/bin/env python3
"""CPU/data/cache/initialization preflight for the MR-CGE V1.1 pilot."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import run_mrcge_v11_pilot as pilot
from scripts import run_rmcge_v22 as legacy
from src.data import load_mag_data
from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b
from src.models import build_model
from src.utils.seeds import set_seed


DATA = ROOT / "research/mrcge_v11/data"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def module_hash(module: torch.nn.Module) -> str:
    return pilot.state_hash(module)


def append_failure(record: dict[str, Any]) -> None:
    path = DATA / "failures.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"failures": []}
    payload.setdefault("failures", []).append({"time_utc": now(), "test_evaluation": False, **record})
    pilot.json_write(path, payload)


def run_unit_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_mrcge_v11.py"]
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    record = {
        "command": "PYTHONPATH=. " + " ".join(command), "returncode": result.returncode,
        "wall_seconds": time.perf_counter() - start,
        "stdout": result.stdout, "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
    if result.returncode != 0:
        raise RuntimeError("MR-CGE focused pytest suite failed; see preflight attempt and pytest output")
    return record


def _common_audit(dataset: str, condition: str, seed: int, data, cache, cfg) -> dict[str, Any]:
    set_seed(seed)
    model = build_model(cfg, pilot.model_info(data))
    cache_meta = model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    head = torch.nn.Linear(model.out_dim, int(data.num_classes))
    return {
        "condition": condition,
        "model_state_sha256": module_hash(model),
        "common_state_sha256": pilot.state_hash(model, pilot.COMMON_PREFIXES),
        "classifier_state_sha256": module_hash(head),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
        "semantic_feature_fingerprint": cache_meta["feature_fingerprint"],
        "semantic_candidate_fingerprint": cache_meta["candidate_fingerprint"],
        "v7b_fingerprint": cache_meta["v7b_fingerprint"],
        "source_flags_shape": list(model.semantic_source_flags_v7b.shape),
        "text_semantic_edges": int(model.semantic_text_edge_index.size(1)),
        "visual_semantic_edges": int(model.semantic_visual_edge_index.size(1)),
    }


def cpu_preflight(*, reuse_focused_tests: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    started_utc = now()
    old = json.loads((DATA / "preflight.json").read_text(encoding="utf-8")) if (DATA / "preflight.json").is_file() else {}
    if reuse_focused_tests:
        source_attempt = next(
            (
                attempt for attempt in reversed(old.get("cpu_attempts", []))
                if attempt.get("status") == "pass"
                and attempt.get("unit_tests", {}).get("passed") is True
            ),
            None,
        )
        if source_attempt is None:
            raise RuntimeError("no prior passing focused test result is available to reuse")
        test_record = dict(source_attempt["unit_tests"])
        test_record.update({
            "reused": True,
            "reused_from_cpu_attempt": source_attempt.get("attempt"),
            "reuse_basis": "MR-CGE model, NC task runner, and focused test sources are unchanged; this update adds Toys data/cache coverage.",
        })
    else:
        test_record = run_unit_tests()
    caches = pilot.cache_entries()
    records = {}
    for dataset in pilot.DATASETS:
        cache = caches[dataset]
        cfg = pilot.build_cfg(dataset, "M", pilot.SEED, "cpu", cache)
        raw = load_mag_data(cfg, "nc", pilot.SEED)
        safe = pilot.mask_test_labels(raw)
        if bool((safe.y[raw.test_idx] != -100).any()):
            raise RuntimeError(f"{dataset}: Test labels not masked")
        if not torch.equal(safe.x, raw.x) or not torch.equal(safe.y[raw.train_idx], raw.y[raw.train_idx]) or not torch.equal(safe.y[raw.val_idx], raw.y[raw.val_idx]):
            raise RuntimeError(f"{dataset}: masking changed non-Test inputs/labels")
        if safe.x.size(0) != safe.num_nodes or safe.edge_index.numel() and int(safe.edge_index.max()) >= safe.num_nodes:
            raise RuntimeError(f"{dataset}: global node alignment failed")
        fingerprint = legacy.dataset_fingerprints(raw)
        cache_payload = load_semantic_candidate_cache_v7b(
            cache["cache_path_abs"], expected_dataset=dataset,
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
        if cache_payload["source_flags"].shape[1] != 2:
            raise RuntimeError(f"{dataset}: V7B source_flags must contain Text and Visual columns")
        if int(safe.x_t.size(1) + safe.x_i.size(1)) != int(safe.input_dim):
            raise RuntimeError(f"{dataset}: modality feature dimensions do not match global input")
        by_condition = []
        for condition in pilot.CONDITIONS:
            condition_cfg = pilot.build_cfg(dataset, condition, pilot.SEED, "cpu", cache)
            by_condition.append(_common_audit(dataset, condition, pilot.SEED, safe, cache, condition_cfg))
        common_hashes = {item["common_state_sha256"] for item in by_condition}
        classifier_hashes = {item["classifier_state_sha256"] for item in by_condition}
        if len(common_hashes) != 1 or len(classifier_hashes) != 1:
            raise RuntimeError(f"{dataset}: S/F/M common initialization or external classifier initialization mismatch")
        records[dataset] = {
            "num_nodes": int(safe.num_nodes), "num_edges": int(safe.edge_index.size(1)),
            "input_dim": int(safe.input_dim), "text_dim": int(safe.x_t.size(1)),
            "visual_dim": int(safe.x_i.size(1)), "num_classes": int(safe.num_classes),
            "test_labels_masked": True, "test_evaluation": False,
            "features_fingerprint": fingerprint["features"],
            "physical_edge_fingerprint": fingerprint["physical_edge_index"],
            "split_fingerprint": fingerprint["split_fingerprint"],
            "semantic_cache_sha256": pilot.sha256_file(Path(cache["cache_path_abs"])),
            "semantic_feature_fingerprint": cache["feature_fingerprint"],
            "semantic_candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"],
            "v7b_source_flags_shape": list(cache_payload["source_flags"].shape),
            "conditions": by_condition,
            "common_initialization_sha256": by_condition[0]["common_state_sha256"],
            "classifier_initialization_sha256": by_condition[0]["classifier_state_sha256"],
        }
    environment = {
        "recorded_utc": now(), "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_available_in_process": torch.cuda.is_available(),
        "cuda_device_count_in_process": torch.cuda.device_count(),
        "nvidia_smi": legacy.environment_record().get("physical_gpu_snapshot"),
        "source_git_sha": pilot.source_sha(),
    }
    attempt = {
        "attempt": len(old.get("cpu_attempts", [])) + 1,
        "started_utc": started_utc,
        "completed_utc": now(), "status": "pass", "unit_tests": test_record,
        "dataset_audits": records, "elapsed_seconds": time.perf_counter() - started,
        "source_git_sha": pilot.source_sha(), "test_evaluation": False,
    }
    old.setdefault("cpu_attempts", []).append(attempt)
    old.update({
        "schema_version": 1, "study": "MR-CGE V1.1", "cpu_status": "pass",
        "cpu_completed_utc": now(), "datasets": records, "source_git_sha": pilot.source_sha(),
        "test_evaluation": False, "test_labels_masked_before_trainer": True,
    })
    pilot.json_write(DATA / "preflight.json", old)
    pilot.json_write(DATA / "environment.json", environment)
    return attempt


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-focused-tests", action="store_true")
    args = parser.parse_args()
    try:
        record = cpu_preflight(reuse_focused_tests=args.reuse_focused_tests)
    except Exception as exc:
        path = DATA / "preflight.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        attempt = {
            "attempt": len(old.get("cpu_attempts", [])) + 1, "completed_utc": now(), "status": "fail",
            "error": repr(exc), "traceback": traceback.format_exc(),
            "source_git_sha": pilot.source_sha(), "test_evaluation": False,
        }
        old.setdefault("cpu_attempts", []).append(attempt)
        old["cpu_status"] = "fail"
        pilot.json_write(path, old)
        append_failure({"stage": "cpu_preflight", **attempt})
        raise
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
