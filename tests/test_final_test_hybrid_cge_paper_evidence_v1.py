from __future__ import annotations

import hashlib
import json

import pytest

from scripts.run_final_test_hybrid_cge_paper_evidence_v1 import authorize


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_test_runner_denies_by_default_before_reading_lock(tmp_path):
    with pytest.raises(PermissionError, match="disabled"):
        authorize(tmp_path / "does-not-exist.json", "Movies", explicit_final_test=False)


def test_final_test_runner_rejects_pending_review_lock(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"status": "pending_review", "models": {}}))
    with pytest.raises(PermissionError, match="approved is required"):
        authorize(lock, "Movies", explicit_final_test=True)


def test_final_test_runner_validates_all_locked_artifact_hashes(tmp_path):
    config = tmp_path / "config.yaml"
    source = tmp_path / "model.py"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("model:\n  name: example\n  paper_condition: base\nseed: 42\n")
    source.write_text("class Model: pass\n")
    checkpoint.write_bytes(b"checkpoint")
    entry = {
        "model_name": "example", "condition": "base", "seed": 42,
        "source_git_sha": "a" * 40, "config_snapshot_path": str(config),
        "config_sha256": _sha(config), "model_source_path": str(source),
        "model_source_sha256": _sha(source), "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
    }
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"status": "approved", "models": {"Movies": entry}}))
    assert authorize(lock, "Movies", explicit_final_test=True)["entry"] == entry
    checkpoint.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="checkpoint SHA256 mismatch"):
        authorize(lock, "Movies", explicit_final_test=True)
