#!/usr/bin/env python3
"""One-update parity check between the frozen source and study instrumentation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_SHA = "8b94d16c3b2a66b658419c7f79e0de688b51a23a"
OUT = ROOT / "research/rmcge_v1_generalization/data/o0_step_parity.json"


def _load_frozen_model_class():
    source = subprocess.run(
        ["git", "show", f"{BASE_SHA}:src/models/rmcge_v1.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    module = types.ModuleType("rmcge_v1_freeze_8b94d16")
    module.__file__ = f"{BASE_SHA}:src/models/rmcge_v1.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.Model, hashlib.sha256(source.encode("utf-8")).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return torch.equal(left["cpu"], right["cpu"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])
    )


def _state_max_abs(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(
        float((left[name].detach().cpu() - right[name].detach().cpu()).abs().max())
        for name in left
    )


def _train_step(model, classifier, optimizer, x, edges, labels, train_idx) -> float:
    model.train()
    classifier.train()
    optimizer.zero_grad(set_to_none=True)
    z = model(x, edges)[0]
    loss = torch.nn.functional.cross_entropy(classifier(z[train_idx]), labels[train_idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(classifier.parameters()),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return float(loss.detach().cpu())


def _run() -> None:
    from scripts.run_rmcge_v1 import _build_config, _cache_entries, _verify_cache_entry
    from src.data import load_mag_data
    from src.models.rmcge_v1 import Model as StudyModel
    from src.tasks.common import build_optimizer, clone_state_dict
    from src.tasks.inference import infer_all_embeddings
    from src.tasks.nc import _evaluate_indices_from_embeddings, _evaluate_split
    from src.utils.seeds import set_seed

    # Select the least-used physical device before the first CUDA initialization.
    import os

    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    physical = max(
        (int(free.strip()), int(index.strip()))
        for line in query.splitlines()
        if line.strip()
        for index, free in (line.split(",", 1),)
    )[1]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"failed to expose selected physical GPU {physical}")
    device = "cuda:0"
    old_model_class, frozen_model_sha = _load_frozen_model_class()

    cache = _verify_cache_entry("Movies", _cache_entries()["Movies"])
    cfg = _build_config("Movies", device=device, seed=42)
    cfg.task.evaluate_test = False
    cfg.task.record_extended_validation = True
    cfg.task.save_ckpt_path = str(ROOT / "outputs/rmcge_v1_generalization/step_parity/best_acc.pt")
    cfg.task.best_f1_ckpt_path = str(ROOT / "outputs/rmcge_v1_generalization/step_parity/best_f1.pt")
    cfg.model.semantic_cache_path = cache["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    data = load_mag_data(cfg, "nc", 42)
    info = {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }

    # Match construction-time initialization and classifier initialization.
    set_seed(42)
    frozen_model = old_model_class(cfg, info)
    frozen_model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset="Movies",
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    frozen_head = torch.nn.Linear(frozen_model.out_dim, info["num_classes"])
    frozen_state = clone_state_dict(frozen_model)
    frozen_head_state = clone_state_dict(frozen_head)

    set_seed(42)
    study_model = StudyModel(cfg, info)
    study_model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset="Movies",
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    study_head = torch.nn.Linear(study_model.out_dim, info["num_classes"])
    study_state = clone_state_dict(study_model)
    study_head_state = clone_state_dict(study_head)
    initialization_match = (
        _state_max_abs(frozen_state, study_state) == 0.0
        and _state_max_abs(frozen_head_state, study_head_state) == 0.0
    )
    frozen_model.load_state_dict(frozen_state, strict=True)
    frozen_head.load_state_dict(frozen_head_state, strict=True)
    study_model.load_state_dict(frozen_state, strict=True)
    study_head.load_state_dict(frozen_head_state, strict=True)

    frozen_model.to(device)
    frozen_head.to(device)
    study_model.to(device)
    study_head.to(device)
    frozen_opt = build_optimizer(list(frozen_model.parameters()) + list(frozen_head.parameters()), cfg, model=frozen_model)
    study_opt = build_optimizer(list(study_model.parameters()) + list(study_head.parameters()), cfg, model=study_model)
    x = data.x.to(device)
    edges = data.edge_index.to(device)
    labels = data.y.to(device)
    train_idx = data.train_idx.to(device)

    set_seed(8127)
    frozen_loss_1 = _train_step(frozen_model, frozen_head, frozen_opt, x, edges, labels, train_idx)
    set_seed(8127)
    study_loss_1 = _train_step(study_model, study_head, study_opt, x, edges, labels, train_idx)
    first_step_state_diff = max(
        _state_max_abs(frozen_model.state_dict(), study_model.state_dict()),
        _state_max_abs(frozen_head.state_dict(), study_head.state_dict()),
    )
    rng_after_step = _rng_state()

    # Replay the original evaluation path in both arms, then add the CPU-only
    # extended metrics and checkpoint snapshot copies to the study arm.
    eval_rng_start = _rng_state()
    frozen_model.eval()
    frozen_head.eval()
    with torch.no_grad():
        frozen_z = infer_all_embeddings(frozen_model, data, torch.device(device), True, 4096, "full")
        frozen_val = _evaluate_split(frozen_head, frozen_z, data.y, data.val_idx, torch.device(device), 4096, list(range(info["num_classes"])))
    rng_after_frozen_eval = _rng_state()

    _restore_rng(eval_rng_start)
    study_model.eval()
    study_head.eval()
    with torch.no_grad():
        study_z = infer_all_embeddings(study_model, data, torch.device(device), True, 4096, "full")
        study_val = _evaluate_split(study_head, study_z, data.y, data.val_idx, torch.device(device), 4096, list(range(info["num_classes"])))
        _evaluate_indices_from_embeddings(study_head, study_z, data.y, data.train_idx, torch.device(device), 4096)
        _evaluate_indices_from_embeddings(study_head, study_z, data.y, data.val_idx, torch.device(device), 4096)
    _study_best_snapshot = clone_state_dict(study_model)
    _study_head_snapshot = clone_state_dict(study_head)
    rng_after_study_eval = _rng_state()
    diagnostic_rng_preserved = _rng_equal(rng_after_frozen_eval, rng_after_study_eval)
    _restore_rng(rng_after_frozen_eval)

    frozen_model.train()
    frozen_head.train()
    frozen_loss_2 = _train_step(frozen_model, frozen_head, frozen_opt, x, edges, labels, train_idx)
    _restore_rng(rng_after_frozen_eval)
    study_model.train()
    study_head.train()
    study_loss_2 = _train_step(study_model, study_head, study_opt, x, edges, labels, train_idx)
    second_step_state_diff = max(
        _state_max_abs(frozen_model.state_dict(), study_model.state_dict()),
        _state_max_abs(frozen_head.state_dict(), study_head.state_dict()),
    )

    result = {
        "passed": bool(initialization_match and diagnostic_rng_preserved and first_step_state_diff <= 1e-5 and second_step_state_diff <= 1e-5),
        "dataset": "Movies",
        "seed": 42,
        "frozen_reference_sha": BASE_SHA,
        "frozen_model_source_sha256": frozen_model_sha,
        "study_model_source_sha256": hashlib.sha256(Path(ROOT / "src/models/rmcge_v1.py").read_bytes()).hexdigest(),
        "device": device,
        "selected_physical_gpu": physical,
        "initialization_identical": initialization_match,
        "first_step_loss_frozen": frozen_loss_1,
        "first_step_loss_study": study_loss_1,
        "first_step_parameter_max_abs_difference": first_step_state_diff,
        "diagnostic_cpu_metrics_preserve_rng_states": diagnostic_rng_preserved,
        "baseline_validation_metrics": frozen_val,
        "study_validation_metrics": study_val,
        "second_step_loss_frozen": frozen_loss_2,
        "second_step_loss_study": study_loss_2,
        "second_step_parameter_max_abs_difference": second_step_state_diff,
        "updates_performed": 2,
        "test_evaluation": False,
        "test_labels_indexed": False,
        "interpretation": "Two-update numerical parity diagnostic only; not a formal run. Repeated CUDA sparse aggregation may create small floating-point differences despite equal RNG states.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        _run()
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "passed": False,
            "phase": "execution_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "test_evaluation": False,
        }
        OUT.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
