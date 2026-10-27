#!/usr/bin/env python3
"""Serial fixed-matrix MR-CGE V1.3-A View-Global Router ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
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

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts import run_mrcge_v12_confirmation as v12
from scripts import run_mrcge_v13_screen as v13
from scripts import run_mrcge_v13_two_hop_cross_screen as history_utils
from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc
from src.utils.seeds import set_seed

STUDY = ROOT / "research/mrcge_v13_view_global_router_ablation"
DATA_DIR = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_view_global_router_ablation"
MODEL = "mrcge_v13_view_global_router"
BASE_MODEL = "mrcge_v13"
CONDITION = "M_repaired"
VARIANT = "joint_view"
PHYSICAL_GPU = 1
BASE_SHA = "4ec2e83b630adf6a7752b33f1ffd9e988392638b"
BASE_BRANCH = "exp/mrcge_v13_private_ce_ablation"
BRANCH = "exp/mrcge_v13_view_global_router_ablation"
PROTOCOL = "unified_full_graph_nc_v1"
CELLS = (("Movies", 42), ("Movies", 43), ("Movies", 44), ("Grocery", 42), ("Grocery", 43))
VIEWS = ("private_text_physical", "private_text_semantic", "private_visual_physical",
         "private_visual_semantic", "collaborative_text_physical", "collaborative_visual_physical")
RUN_FIELDS = (
    "dataset", "model", "condition", "variant", "seed", "status", "source_git_sha",
    "unimodal_weight", "aux_weight", "balance_weight", "best_acc_epoch", "best_f1_epoch",
    "epochs_completed", "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc",
    "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce",
    "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap",
    "mean_train_balance_raw", "mean_train_balance_scaled", "mean_val_balance_raw",
    "mean_val_balance_scaled", "best_acc_balance_raw", "best_acc_balance_scaled",
    "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "physical_gpu", "gpu_uuid",
    "gpu_name", "gpu_free_before_mib", "gpu_util_before_pct", "gpu_temperature_before_c",
    "gpu_processes_before", "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path",
    "best_f1_checkpoint_sha256", "checkpoint_reload_verified", "test_evaluation", "link_prediction",
    "initial_model_sha256", "initial_classifier_sha256", "init_matches_history",
    "model_parameters", "classifier_parameters", "total_parameters", "config_sha256",
    "model_source_sha256", "trainer_source_sha256", "runner_source_sha256",
    "feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint",
    "split_file_sha256", "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    "attempt_dir", "error",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def source_hashes() -> dict[str, str]:
    paths = {
        "model": ROOT / "src/models/mrcge_v13_view_global_router.py",
        "parent_model": ROOT / "src/models/mrcge_v13.py",
        "trainer": ROOT / "src/tasks/nc.py",
        "model_config": ROOT / "configs/model/mrcge_v13_view_global_router.yaml",
        "base_model_config": ROOT / "configs/model/mrcge_v13.yaml",
        "task_config": ROOT / "configs/task/nc.yaml",
        "v13_config_builder": ROOT / "scripts/run_mrcge_v13_screen.py",
        "history_loader": ROOT / "scripts/run_mrcge_v13_two_hop_cross_screen.py",
        "checkpoint_audit": ROOT / "scripts/run_mrcge_v12_confirmation.py",
        "runner": Path(__file__).resolve(),
        "tests": ROOT / "tests/test_mrcge_v13_view_global_router.py",
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False,
                               default=lambda value: str(value)), encoding="utf-8")
    temp.replace(path)


def json_read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def cfg_model_container(cfg) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.model, resolve=True)


def candidate_model_config(cache: dict[str, Any]):
    model_cfg = OmegaConf.load(ROOT / f"configs/model/{MODEL}.yaml")
    model_cfg.semantic_cache_path = cache["cache_path_abs"]
    model_cfg.semantic_feature_fingerprint = cache["feature_fingerprint"]
    model_cfg.semantic_candidate_fingerprint = cache["candidate_fingerprint"]
    model_cfg.semantic_v7b_fingerprint = cache["v7b_fingerprint"]
    return model_cfg


def assert_config_and_data_protocol(dataset: str, seed: int, cache: dict[str, Any], device: str):
    base_cfg = v13.make_cfg(dataset, VARIANT, seed, device, cache)
    candidate_cfg = v13.make_cfg(dataset, VARIANT, seed, device, cache)
    candidate_cfg.model = candidate_model_config(cache)
    base_model, candidate_model = cfg_model_container(base_cfg), cfg_model_container(candidate_cfg)
    changed = [key for key in sorted(set(base_model) | set(candidate_model))
               if base_model.get(key) != candidate_model.get(key)]
    if changed != ["name"]:
        raise RuntimeError(f"only model.name may differ from V1.3-A, found model changes: {changed}")
    if OmegaConf.to_container(base_cfg.task, resolve=True) != OmegaConf.to_container(candidate_cfg.task, resolve=True):
        raise RuntimeError("candidate task settings differ from the original V1.3-A config")
    if float(candidate_cfg.task.loss.unimodal_weight) != 0.3:
        raise RuntimeError("the original V1.3-A Private CE weight must remain 0.3")
    if candidate_cfg.dataset.auto_generate_splits is not False or candidate_cfg.task.evaluate_test is not False:
        raise RuntimeError("fixed splits and Test-off are required")
    if float(candidate_cfg.model.balance_weight) != 0.01:
        raise RuntimeError("Balance Loss weight must remain 0.01")
    return {"changed_model_fields": changed, "unimodal_weight": 0.3,
            "aux_weight": float(candidate_cfg.task.loss.aux_weight), "balance_weight": 0.01}


def historical_references() -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for dataset, seed in CELLS:
        row = history_utils.historical_v13(dataset, seed)
        checkpoint_hashes = history_utils.verify_reference_checkpoints(row, f"V1.3-A {dataset}/seed{seed}")
        result[(dataset, seed)] = {"row": row, "checkpoint_hashes": checkpoint_hashes}
    return result


def data_for_run(dataset: str, seed: int, cache: dict[str, Any], device: str):
    # Reuse the established loader/alignment audit; labels are Test-masked before
    # the Trainer receives this object.
    cfg, data, fingerprints, alignment = v13.load_safe_data(dataset, VARIANT, seed, cache, device)
    cfg.model = candidate_model_config(cache)
    assert_config_and_data_protocol(dataset, seed, cache, device)
    if bool((data.y[data.test_idx] != -100).any()):
        raise RuntimeError("Test labels are not masked before trainer entry")
    return cfg, data, fingerprints, alignment


def route_summary_rows(dataset: str, seed: int, audit: dict[str, Any], *,
                       balance_raw: float, balance_scaled: float, router_kind: str) -> list[dict[str, Any]]:
    details = audit["info"].get("details", {})
    routes = details.get("routes", {})
    routes = {**routes, **{f"collaborative_{name}_physical": value
                           for name, value in details.get("collaborative_routes", {}).items()}}
    rows = []
    for view in VIEWS:
        route = routes.get(view)
        if route is None:
            raise RuntimeError(f"checkpoint audit omitted route telemetry for {view}")
        probs = [float(value) for value in route["dense_importance"]]
        shares = [float(value) for value in route["top2_selection_share"]]
        selected = [index for index, value in enumerate(shares) if value > 1.0e-6]
        if not selected:
            raise RuntimeError(f"{view}: route telemetry reports no selected experts")
        if router_kind == "view_global":
            if len(selected) != 2 or any(abs(shares[index] - 0.5) > 1.0e-6 for index in selected):
                raise RuntimeError(f"{view}: expected exactly two globally selected experts, got {selected}/{shares}")
            weights = [probs[index] / sum(probs[item] for item in selected) for index in selected]
        else:
            # Existing summary telemetry aggregates node-conditioned selections;
            # it does not retain per-node mixture weights.
            weights = []
        rows.append({
            "dataset": dataset, "seed": seed, "router": router_kind, "view": view,
            "selected_expert_ids": json.dumps(selected),
            "view_global_top2_expert_ids": json.dumps(selected if router_kind == "view_global" else []),
            "view_global_top2_mixture_weights": json.dumps(weights),
            "expert_probabilities": json.dumps(probs), "expert_selection_shares": json.dumps(shares),
            "dense_entropy": route["dense_entropy"], "selected_weight_entropy": route["selected_weight_entropy"],
            "active_nodes": route["active_nodes"], "balance_raw": balance_raw,
            "balance_scaled": balance_scaled, "test_evaluation": False,
        })
    return rows


def gpu_snapshot_for_one() -> dict[str, Any]:
    snapshot = v13.smi_snapshot()
    selected = next((row for row in snapshot["devices"] if int(row["physical_index"]) == PHYSICAL_GPU), None)
    if selected is None:
        raise RuntimeError("physical GPU 1 is not visible")
    if int(selected["memory_free_mib"]) < 8192:
        raise RuntimeError(f"GPU 1 has less than 8192 MiB free: {selected['memory_free_mib']} MiB")
    return {"snapshot": snapshot, "device": selected,
            "processes": [row for row in snapshot["compute_processes"] if row["gpu_uuid"] == selected["uuid"]]}


def assert_view_global_summaries(summaries: list[dict[str, Any]]) -> None:
    if len(summaries) != 6:
        raise RuntimeError(f"expected telemetry for six independently routed views, found {len(summaries)}")
    for summary in summaries:
        share = [float(value) for value in summary["top2_selection_share"]]
        chosen = [value for value in share if value > 1.0e-6]
        if len(chosen) != 2 or any(abs(value - 0.5) > 1.0e-6 for value in chosen):
            raise RuntimeError(f"route is not View-Global Top-2: {summary}")


def gpu_smoke() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError("GPU smoke must see only CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU smoke cannot see the isolated physical GPU 1")
    snapshot = gpu_snapshot_for_one()
    cache = v13.cache_entries()["Movies"]
    cfg, data, fingerprints, alignment = data_for_run("Movies", 42, cache, "cuda:0")
    device = torch.device("cuda:0")
    set_seed(90042)
    model = build_model(cfg, v13.data_info(data)).to(device)
    model.load_semantic_cache(cache["cache_path_abs"], expected_dataset="Movies",
                              expected_feature_fingerprint=cache["feature_fingerprint"],
                              expected_candidate_fingerprint=cache["candidate_fingerprint"],
                              expected_v7b_fingerprint=cache["v7b_fingerprint"])
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    data_x, edge_index, labels = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    train_idx = data.train_idx.to(device)
    model.train(); classifier.train()
    model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)
    output, _, _, aux, info = model(data_x, edge_index, return_details=False)
    main_ce = nn.functional.cross_entropy(classifier(output[train_idx]), labels[train_idx])
    loss, _ = _compose_full_graph_loss(model=model, info=info, main_ce=main_ce, aux_loss=aux,
                                       aux_weight=1.0, unimodal_weight=0.3,
                                       labels=labels, train_idx=train_idx)
    if not torch.isfinite(loss):
        raise RuntimeError("GPU smoke produced a non-finite Train loss")
    loss.backward()
    if not any(p.grad is not None and float(p.grad.norm()) > 0 for p in model.router.parameters()):
        raise RuntimeError("GPU smoke did not propagate task gradients through the router mean")
    if not any(p.grad is not None and float(p.grad.norm()) > 0 for p in model.filter_bank.experts.parameters()):
        raise RuntimeError("GPU smoke did not propagate task gradients into selected experts")
    train_summaries = [model._route_summary(item["route"], item["support"])
                       for item in model._last_route_snapshot.values()]
    assert_view_global_summaries(train_summaries)
    model.eval(); classifier.eval()
    with torch.no_grad():
        _, _, _, _, eval_info = model(data_x, edge_index, return_details=True)
    details = eval_info["details"]
    eval_summaries = list(details["routes"].values()) + list(details["collaborative_routes"].values())
    assert_view_global_summaries(eval_summaries)
    if bool((data.y[data.test_idx] != -100).any()):
        raise RuntimeError("GPU smoke found an unmasked Test label")
    torch.cuda.synchronize(device)
    result = {
        "status": "pass", "completed_utc": now(), "device": "physical GPU 1",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(ROOT / f"src/models/{MODEL}.py"),
        "gpu_uuid": snapshot["device"]["uuid"], "gpu_name": snapshot["device"]["name"],
        "gpu_free_before_mib": snapshot["device"]["memory_free_mib"],
        "baseline_processes": snapshot["processes"], "dataset": "Movies", "seed": 42,
        "unimodal_weight": float(cfg.task.loss.unimodal_weight),
        "train_low_memory_view_global": True, "eval_collect_view_global": True,
        "six_train_views_verified": len(train_summaries), "six_eval_views_verified": len(eval_summaries),
        "router_gradient_nonzero": True, "expert_gradient_nonzero": True,
        "train_loss": float(loss.detach()), "test_labels_masked": True,
        "test_evaluation": False, "link_prediction": False,
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": alignment["split_file_sha256"],
    }
    json_write(DATA_DIR / "gpu_smoke.json", result)
    return result


def worker(dataset: str, seed: int, attempt_text: str) -> None:
    attempt = Path(attempt_text).resolve()
    started = time.perf_counter()
    try:
        if os.environ.get("MRCGE_VIEW_GLOBAL_WORKER") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
            raise RuntimeError("worker must use CUDA_VISIBLE_DEVICES=1")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("worker does not see exactly physical GPU 1")
        expected_source = os.environ["MRCGE_VIEW_GLOBAL_SOURCE_SHA"]
        if source_sha() != expected_source:
            raise RuntimeError("training source HEAD changed after freeze")
        gpu = gpu_snapshot_for_one()
        device = torch.device("cuda:0")
        cache = v13.cache_entries()[dataset]
        cfg, data, fingerprints, alignment = data_for_run(dataset, seed, cache, "cuda:0")
        if not (cfg.task.evaluate_test is False and cfg.dataset.auto_generate_splits is False):
            raise RuntimeError("Test-off/fixed-split protocol drifted")
        if float(cfg.task.loss.unimodal_weight) != 0.3 or float(cfg.task.loss.aux_weight) != 1.0:
            raise RuntimeError("V1.3-A loss weights drifted from the frozen protocol")
        if bool((data.y[data.test_idx] != -100).any()):
            raise RuntimeError("Test labels were not masked before training")
        ref = history_utils.historical_v13(dataset, seed)
        ref_hashes = history_utils.verify_reference_checkpoints(ref, f"V1.3-A {dataset}/seed{seed}")
        init = v12.initial_state_audit(cfg, data, cache, device)
        for key, ref_key in (("initial_model_sha256", "initial_model_sha256"),
                             ("initial_classifier_sha256", "initial_classifier_sha256")):
            expected = ref.get(ref_key, "")
            if expected and init[key] != expected:
                raise RuntimeError(f"same-seed initialization differs from historical V1.3-A: {key}")

        base_cfg = cfg.copy()
        base_cfg.model.name = BASE_MODEL
        historical_audit = v12.validation_logits_audit(
            base_cfg, data, cache, Path(ref["checkpoint_path"]), device, details=True
        )
        if not historical_audit["rng_unchanged"]:
            raise RuntimeError("historical read-only validation audit changed RNG state")
        old_details = historical_audit["info"].get("details", {})
        old_raw, old_scaled = float(old_details["balance_loss"]), float(old_details["aux_loss"])
        old_route_rows = route_summary_rows(dataset, seed, historical_audit, balance_raw=old_raw,
                                            balance_scaled=old_scaled, router_kind="node_conditioned")

        cfg.task.save_ckpt_path = str(attempt / "best_accuracy.pt")
        cfg.task.best_f1_ckpt_path = str(attempt / "best_f1.pt")
        cfg.task.run_metrics_path = str(attempt / "run_metrics.json")
        config_text = OmegaConf.to_yaml(cfg, resolve=True)
        (attempt / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
        config_hash = hashlib.sha256(config_text.encode()).hexdigest()
        model_path = ROOT / f"src/models/{MODEL}.py"
        trainer_path = ROOT / "src/tasks/nc.py"
        cfg.task.extended_checkpoint_metadata = {
            "study": "MR-CGE V1.3-A Node-Conditioned vs View-Global Router Ablation",
            "dataset": dataset, "condition": CONDITION, "variant": VARIANT, "seed": seed,
            "source_git_sha": expected_source, "base_sha": BASE_SHA, "model_name": MODEL,
            "model_source_sha256": sha256_file(model_path), "trainer_source_sha256": sha256_file(trainer_path),
            "config_sha256": config_hash, "unimodal_weight": 0.3, "aux_weight": 1.0,
            "balance_weight": 0.01, "feature_fingerprint": cache["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "split_file_sha256": alignment["split_file_sha256"],
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"],
            "v7b_fingerprint": cache["v7b_fingerprint"], "test_evaluation": False,
            "test_labels_masked": True, "link_prediction": False,
            "historical_best_accuracy_checkpoint_sha256": ref_hashes["best_accuracy"]["sha256"],
        }
        set_seed(seed)
        model = build_model(cfg, v13.data_info(data)).to(device)
        model.load_semantic_cache(cache["cache_path_abs"], expected_dataset=dataset,
                                  expected_feature_fingerprint=cache["feature_fingerprint"],
                                  expected_candidate_fingerprint=cache["candidate_fingerprint"],
                                  expected_v7b_fingerprint=cache["v7b_fingerprint"])
        classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
        model_state_hash = v12.state_hash(model)
        classifier_state_hash = v12.state_hash(classifier)
        if model_state_hash != init["initial_model_sha256"] or classifier_state_hash != init["initial_classifier_sha256"]:
            raise RuntimeError("Trainer initialization differs from the separately audited initialization")
        parameter_count = sum(p.numel() for p in model.parameters())
        cfg.task.extended_checkpoint_metadata["model_parameters"] = parameter_count
        logger = logging.getLogger(f"view_global.{dataset}.{seed}")
        logger.handlers.clear(); logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.FileHandler(attempt / "train.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt); logger.addHandler(handler)
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        run_nc(cfg, data, device, logger)
        packed = json.loads(Path(cfg.task.run_metrics_path).read_text(encoding="utf-8"))
        if packed.get("protocol_version") != PROTOCOL or packed.get("run_seeds") != [seed]:
            raise RuntimeError("Trainer protocol or seed provenance mismatch")
        metadata = packed["runs"][0]["metadata"]
        if metadata["initial_parameter_audit"]["model_state_sha256"] != model_state_hash:
            raise RuntimeError("Trainer initial model hash differs from preflight")
        if metadata["initial_parameter_audit"]["classifier_state_sha256"] != classifier_state_hash:
            raise RuntimeError("Trainer initial classifier hash differs from preflight")
        if float(metadata.get("unimodal_auxiliary_weight", math.nan)) != 0.3:
            raise RuntimeError("Trainer did not confirm Private CE weight 0.3")
        select_acc, select_f1 = metadata["dual_checkpoint_selection"]["best_accuracy"], metadata["dual_checkpoint_selection"]["best_macro_f1"]
        acc_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        if not acc_path.is_file() or not f1_path.is_file():
            raise RuntimeError("Best-Accuracy or Best-F1 checkpoint is missing")
        acc = v12.validation_logits_audit(cfg, data, cache, acc_path, device, details=True)
        f1 = v12.validation_logits_audit(cfg, data, cache, f1_path, device, details=True)
        metric_map = (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce"))
        for selection, audit, name in ((select_acc, acc, "Best-Accuracy"), (select_f1, f1, "Best-F1")):
            for field, metric in metric_map:
                if abs(float(selection["metrics"][field]) - float(audit["val"][metric])) > 3e-6:
                    raise RuntimeError(f"reloaded {name} checkpoint {field} differs from Trainer metrics")
        if not (acc["rng_unchanged"] and f1["rng_unchanged"]):
            raise RuntimeError("checkpoint reload validation changed RNG state")
        if cfg.task.evaluate_test is not False or bool((data.y[data.test_idx] != -100).any()):
            raise RuntimeError("Test-off boundary failed")
        history = metadata.get("epoch_history", [])
        if not history or not all(all(k in row and math.isfinite(float(row[k])) for k in
                                      ("train_loss", "val_acc", "val_macro_f1", "val_ce")) for row in history):
            raise RuntimeError("finite epoch history is missing or incomplete")
        json_write(attempt / "epoch_history.json", history)
        train_metrics, val_metrics = acc["train"], acc["val"]
        details = acc["info"].get("details", {})
        raw_balance, scaled_balance = float(details["balance_loss"]), float(details["aux_loss"])
        global_route_rows = route_summary_rows(dataset, seed, acc, balance_raw=raw_balance,
                                               balance_scaled=scaled_balance, router_kind="view_global")
        train_raw = [float(row["train_balance_loss_raw"]) for row in history if "train_balance_loss_raw" in row]
        train_scaled = [float(row["train_aux_loss_scaled"]) for row in history if "train_aux_loss_scaled" in row]
        val_raw = [float(row["balance_loss_raw"]) for row in history if "balance_loss_raw" in row]
        val_scaled = [float(row["aux_loss_scaled"]) for row in history if "aux_loss_scaled" in row]
        record = {
            "dataset": dataset, "model": MODEL, "condition": CONDITION, "variant": VARIANT,
            "seed": seed, "status": "complete", "source_git_sha": expected_source,
            "unimodal_weight": 0.3, "aux_weight": 1.0, "balance_weight": 0.01,
            "best_acc_epoch": select_acc["epoch"], "best_f1_epoch": select_f1["epoch"],
            "epochs_completed": len(history), "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"], "val_ce": val_metrics["ce"],
            "best_f1_val_acc": f1["val"]["acc"], "best_f1_val_macro_f1": f1["val"]["macro_f1"],
            "best_f1_val_ce": f1["val"]["ce"], "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"], "train_ce": train_metrics["ce"],
            "train_val_acc_gap": train_metrics["acc"] - val_metrics["acc"],
            "train_val_macro_f1_gap": train_metrics["macro_f1"] - val_metrics["macro_f1"],
            "train_val_ce_gap": train_metrics["ce"] - val_metrics["ce"],
            "mean_train_balance_raw": statistics.mean(train_raw) if train_raw else "",
            "mean_train_balance_scaled": statistics.mean(train_scaled) if train_scaled else "",
            "mean_val_balance_raw": statistics.mean(val_raw) if val_raw else "",
            "mean_val_balance_scaled": statistics.mean(val_scaled) if val_scaled else "",
            "best_acc_balance_raw": raw_balance, "best_acc_balance_scaled": scaled_balance,
            "wall_seconds": float(metadata.get("training_wall_seconds", time.perf_counter() - started)),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "physical_gpu": PHYSICAL_GPU, "gpu_uuid": gpu["device"]["uuid"],
            "gpu_name": torch.cuda.get_device_name(device), "gpu_free_before_mib": gpu["device"]["memory_free_mib"],
            "gpu_util_before_pct": gpu["device"]["utilization_percent"],
            "gpu_temperature_before_c": gpu["device"]["temperature_c"],
            "gpu_processes_before": json.dumps(gpu["processes"], sort_keys=True),
            "checkpoint_path": str(acc_path), "checkpoint_sha256": sha256_file(acc_path),
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": sha256_file(f1_path),
            "checkpoint_reload_verified": True, "test_evaluation": False, "link_prediction": False,
            "initial_model_sha256": model_state_hash, "initial_classifier_sha256": classifier_state_hash,
            "init_matches_history": True, "model_parameters": parameter_count,
            "classifier_parameters": sum(p.numel() for p in classifier.parameters()),
            "total_parameters": parameter_count + sum(p.numel() for p in classifier.parameters()),
            "config_sha256": config_hash, "model_source_sha256": sha256_file(model_path),
            "trainer_source_sha256": sha256_file(trainer_path), "runner_source_sha256": sha256_file(Path(__file__)),
            "feature_fingerprint": cache["feature_fingerprint"], "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"], "split_file_sha256": alignment["split_file_sha256"],
            "semantic_cache_sha256": sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "attempt_dir": str(attempt), "error": "",
        }
        json_write(attempt / "worker_result.json", {"status": "complete", "run": record,
                                                     "historical_router_rows": old_route_rows,
                                                     "view_global_router_rows": global_route_rows,
                                                     "historical_balance_raw": old_raw,
                                                     "historical_balance_scaled": old_scaled,
                                                     "historical_checkpoint_hashes": ref_hashes})
    except Exception as exc:
        json_write(attempt / "worker_result.json", {"status": "failed", "dataset": dataset, "seed": seed,
                                                     "source_git_sha": source_sha(), "error": repr(exc),
                                                     "traceback": traceback.format_exc(), "test_evaluation": False})
        raise


def run_one(dataset: str, seed: int, frozen_sha: str) -> dict[str, Any]:
    gpu = gpu_snapshot_for_one()
    attempt = OUTPUT / "runs" / dataset / f"seed_{seed}"
    if attempt.exists():
        raise RuntimeError(f"refusing to overwrite existing attempt directory: {attempt}")
    attempt.mkdir(parents=True)
    json_write(attempt / "attempt_started.json", {
        "dataset": dataset, "seed": seed, "source_git_sha": frozen_sha, "started_utc": now(),
        "physical_gpu": PHYSICAL_GPU, "gpu_uuid": gpu["device"]["uuid"],
        "gpu_free_before_mib": gpu["device"]["memory_free_mib"],
        "gpu_processes_before": gpu["processes"], "test_evaluation": False,
    })
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    env["MRCGE_VIEW_GLOBAL_WORKER"] = "1"
    env["MRCGE_VIEW_GLOBAL_SOURCE_SHA"] = frozen_sha
    env.pop("MR_CGE_SHARED_GPU_OVERRIDE", None)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--dataset", dataset,
               "--seed", str(seed), "--attempt-dir", str(attempt)]
    log_path = attempt / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(15)
        code = proc.wait()
    result = json_read(attempt / "worker_result.json", {})
    if code != 0 or result.get("status") != "complete":
        raise RuntimeError(f"worker failed for {dataset}/seed{seed}; see {log_path}: {result.get('error')}")
    record = result["run"]
    for name in ("historical_router_rows", "view_global_router_rows"):
        path = DATA_DIR / "router_summary.csv"
        rows = read_rows(path)
        fields = list(dict.fromkeys([*(rows[0].keys() if rows else ()), *(key for row in result[name] for key in row)]))
        write_rows(path, [*rows, *result[name]], fields)
    record["historical_balance_raw"] = result["historical_balance_raw"]
    record["historical_balance_scaled"] = result["historical_balance_scaled"]
    record["historical_best_accuracy_checkpoint_sha256"] = result["historical_checkpoint_hashes"]["best_accuracy"]["sha256"]
    append_record(record)
    return record


def append_record(row: dict[str, Any]) -> None:
    path = DATA_DIR / "runs.csv"
    old = read_rows(path)
    key = (row["dataset"], str(row["seed"]))
    if any((item.get("dataset"), item.get("seed")) == key for item in old):
        raise RuntimeError(f"duplicate run record refused: {key}")
    fields = list(dict.fromkeys([*RUN_FIELDS, *(key for item in old for key in item), *row.keys()]))
    write_rows(path, [*old, row], fields)


def summarize(references: dict[tuple[str, int], dict[str, Any]]):
    runs = read_rows(DATA_DIR / "runs.csv")
    paired, grouped = [], {"Movies": [], "Grocery": []}
    for run in runs:
        dataset, seed = run["dataset"], int(run["seed"])
        ref = references[(dataset, seed)]["row"]
        row = {"dataset": dataset, "seed": seed, "new_checkpoint_sha256": run["checkpoint_sha256"],
               "historical_checkpoint_sha256": run["historical_best_accuracy_checkpoint_sha256"],
               "test_evaluation": False}
        for old, new, delta in (("val_acc", "val_acc", "delta_val_acc_pp"),
                                ("val_macro_f1", "val_macro_f1", "delta_val_macro_f1_pp")):
            old_value, new_value = float(ref[old]), float(run[new])
            row[f"historical_{old}"] = old_value; row[f"new_{new}"] = new_value
            row[delta] = 100.0 * (new_value - old_value)
        old_ce, new_ce = float(ref["val_ce"]), float(run["val_ce"])
        row.update({"historical_val_ce": old_ce, "new_val_ce": new_ce, "delta_val_ce": new_ce - old_ce})
        for metric in ("train_acc", "train_macro_f1", "train_ce", "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap"):
            old_value = float(ref[metric]) if ref.get(metric) not in (None, "") else ""
            new_value = float(run[metric])
            row[f"historical_{metric}"] = old_value; row[f"new_{metric}"] = new_value
            row[f"delta_{metric}"] = new_value - old_value if old_value != "" else ""
        paired.append(row); grouped[dataset].append((run, row))
    write_rows(DATA_DIR / "paired_differences.csv", paired, list(paired[0]) if paired else [])
    summaries = []
    for dataset, entries in grouped.items():
        if not entries:
            continue
        item = {"dataset": dataset, "n": len(entries), "seeds": ";".join(str(run["seed"]) for run, _ in entries)}
        for metric in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_macro_f1", "train_ce",
                       "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap"):
            values = [float(run[metric]) for run, _ in entries]
            item[f"new_{metric}_mean"] = statistics.mean(values)
            item[f"new_{metric}_population_sd"] = statistics.pstdev(values)
        for metric in ("delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce",
                       "delta_train_acc", "delta_train_macro_f1", "delta_train_ce",
                       "delta_train_val_acc_gap", "delta_train_val_macro_f1_gap", "delta_train_val_ce_gap"):
            values = [float(row[metric]) for _, row in entries if row[metric] != ""]
            if values:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_population_sd"] = statistics.pstdev(values)
        summaries.append(item)
    write_rows(DATA_DIR / "summary.csv", summaries, list(summaries[0]) if summaries else [])
    return paired, summaries


def decision_label(paired: list[dict[str, Any]]) -> tuple[str, str]:
    by_dataset = {dataset: [row for row in paired if row["dataset"] == dataset]
                  for dataset in ("Movies", "Grocery")}
    means = {dataset: {metric: statistics.mean(float(row[metric]) for row in rows)
                       for metric in ("delta_val_acc_pp", "delta_val_macro_f1_pp", "delta_val_ce")}
             for dataset, rows in by_dataset.items() if rows}
    if len(means) != 2:
        return "NEAR_TIE", "Incomplete dataset coverage; no mechanism conclusion."
    m, g = means["Movies"], means["Grocery"]
    if (m["delta_val_acc_pp"] > 0 > g["delta_val_acc_pp"] or
            g["delta_val_acc_pp"] > 0 > m["delta_val_acc_pp"]):
        return "MIXED", "The paired Accuracy direction differs by dataset."
    if all(v["delta_val_acc_pp"] < 0 for v in means.values()) and all(v["delta_val_ce"] >= 0 for v in means.values()):
        return "SUPPORT_NODE_ROUTING", "Node-Conditioned Router is favored on both datasets without a CE contradiction."
    if all(v["delta_val_acc_pp"] >= 0 and v["delta_val_macro_f1_pp"] >= -0.5 and v["delta_val_ce"] <= 0.01
           for v in means.values()):
        return "GLOBAL_COMPETITIVE", "View-Global is competitive on both datasets without a material F1/CE regression."
    if any((v["delta_val_acc_pp"] < 0 and v["delta_val_macro_f1_pp"] > 0.5 and v["delta_val_ce"] < 0) or
           (v["delta_val_acc_pp"] > 0 and (v["delta_val_macro_f1_pp"] < -0.5 or v["delta_val_ce"] > 0.01))
           for v in means.values()):
        return "MIXED", "Accuracy, Macro-F1, or CE show a material within-dataset conflict."
    return "NEAR_TIE", "Effects are small, seed directions vary, or metrics do not establish a clear winner."


def report(paired: list[dict[str, Any]], summaries: list[dict[str, Any]], decision: tuple[str, str]) -> None:
    runs = read_rows(DATA_DIR / "runs.csv")
    lines = [
        "# Node-Conditioned vs View-Global Router Ablation Results", "",
        f"Branch `{BRANCH}`; base `{BASE_SHA}`; frozen training source `{source_sha()}`.", "",
        "This full-mechanism ablation changes only the router granularity. The original V1.3-A protocol is retained, including Private CE weight 0.3 and Balance Loss weight 0.01. Results use Validation only; Test labels were masked before training.", "",
        "## Paired Best-Validation-Accuracy checkpoints", "",
        "| Dataset | Seed | Global Acc | Global Macro-F1 | Global CE | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE | Train−Val Acc gap (pp) | Train−Val CE gap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        run = next(item for item in runs if item["dataset"] == row["dataset"] and item["seed"] == str(row["seed"]))
        lines.append(f"| {row['dataset']} | {row['seed']} | {100*float(run['val_acc']):.3f}% | {100*float(run['val_macro_f1']):.3f}% | {float(run['val_ce']):.5f} | {float(row['delta_val_acc_pp']):+.3f} | {float(row['delta_val_macro_f1_pp']):+.3f} | {float(row['delta_val_ce']):+.5f} | {100*float(run['train_val_acc_gap']):+.2f} | {float(run['train_val_ce_gap']):+.5f} |")
    lines += ["", "## Dataset summaries", "", "| Dataset | n | Global Acc mean ± SD | Global Macro-F1 mean ± SD | Global CE mean ± SD | Δ Acc mean ± SD (pp) | Δ Macro-F1 mean ± SD (pp) | Δ CE mean ± SD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['n']} | {100*float(row['new_val_acc_mean']):.3f}% ± {100*float(row['new_val_acc_population_sd']):.3f} | {100*float(row['new_val_macro_f1_mean']):.3f}% ± {100*float(row['new_val_macro_f1_population_sd']):.3f} | {float(row['new_val_ce_mean']):.5f} ± {float(row['new_val_ce_population_sd']):.5f} | {float(row['delta_val_acc_pp_mean']):+.3f} ± {float(row['delta_val_acc_pp_population_sd']):.3f} | {float(row['delta_val_macro_f1_pp_mean']):+.3f} ± {float(row['delta_val_macro_f1_pp_population_sd']):.3f} | {float(row['delta_val_ce_mean']):+.5f} ± {float(row['delta_val_ce_population_sd']):.5f} |")
    router_rows = read_rows(DATA_DIR / "router_summary.csv")
    lines += ["", "## Route and Balance Loss audit", "", "For each Best-Accuracy checkpoint, `data/router_summary.csv` records the six independent view calls for both the historical node-conditioned checkpoint and the new View-Global checkpoint. The View-Global rows contain exact shared Top-2 IDs and mixture weights. Historical rows use existing aggregate telemetry: selected-expert union, four-expert mean probabilities, selection shares, entropy, and raw/scaled Balance Loss; the old telemetry does not retain per-node mixture weights.", ""]
    for run in runs:
        d, seed = run["dataset"], run["seed"]
        lines.append(f"### {d}, seed {seed}")
        lines.append("")
        lines.append(f"Historical raw/scaled Balance: {float(run['historical_balance_raw']):.6f} / {float(run['historical_balance_scaled']):.6f}; View-Global: {float(run['best_acc_balance_raw']):.6f} / {float(run['best_acc_balance_scaled']):.6f}.")
        for router in ("node_conditioned", "view_global"):
            selected = [row for row in router_rows if row["dataset"] == d and row["seed"] == seed and row["router"] == router]
            lines.append("")
            lines.append(f"{router.replace('_', ' ').title()} expert routing by view:")
            for row in selected:
                if router == "view_global":
                    lines.append(f"- `{row['view']}`: Top-2 experts {row['view_global_top2_expert_ids']}, weights {row['view_global_top2_mixture_weights']}; shares {row['expert_selection_shares']}; entropy {float(row['dense_entropy']):.4f}.")
                else:
                    lines.append(f"- `{row['view']}`: selected-expert union {row['selected_expert_ids']}; shares {row['expert_selection_shares']}; entropy {float(row['dense_entropy']):.4f}.")
        lines.append("")
    lines += ["## Interpretation", "", f"**{decision[0]}** — {decision[1]}", "",
              "Balance Loss is coupled to route granularity because each View-Global call uses only two experts. This is a practical whole-router ablation, not a strict causal isolation of node adaptation from balancing. The transductive full-graph route mean uses unlabeled node features and does not establish inductive online behavior. No expert frequency, class, or modality specialization claim is made.", "",
              "## Protocol and reproducibility", "",
              "Five serial runs used physical GPU 1, fixed seed-42 splits and the existing V7B.1 K=8 cache. AdamW, 1e-3 LR, 1e-4 weight decay, 300-epoch cap, patience 30, minimum epoch 30, min delta 1e-4, clip 1.0, dropout 0.2, fusion dropout 0.2, late dropout 0.15, no scheduler, Private CE 0.3 and Balance Loss 0.01 were retained. Both Best-Accuracy and Best-F1 checkpoints were reloaded and verified. Test evaluation and Link Prediction were off. See `data/manifest.json`, `data/runs.csv`, and `data/router_summary.csv` for hashes and per-run provenance.", ""]
    (STUDY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_all() -> None:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"wrong branch {branch}; expected {BRANCH}")
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                      capture_output=True, text=True).stdout.strip():
        raise RuntimeError("working tree must be clean after the source-freeze commit")
    if subprocess.run(["git", "merge-base", "HEAD", BASE_SHA], cwd=ROOT, check=True,
                      capture_output=True, text=True).stdout.strip() != BASE_SHA:
        raise RuntimeError("branch is not based on the required V1.3-A parent commit")
    frozen_sha, frozen_hashes = source_sha(), source_hashes()
    smoke = json_read(DATA_DIR / "gpu_smoke.json", {})
    if smoke.get("status") != "pass" or smoke.get("gpu_uuid") == "":
        raise RuntimeError("a passing GPU 1 Train/Eval smoke test is required")
    if smoke.get("runner_source_sha256") != frozen_hashes["runner"] or smoke.get("model_source_sha256") != frozen_hashes["model"]:
        raise RuntimeError("GPU smoke evidence was produced by different model/runner source")
    if DATA_DIR.exists() and any(path.name != "gpu_smoke.json" for path in DATA_DIR.iterdir()):
        raise RuntimeError("refusing to overwrite existing study data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    caches = v13.cache_entries()
    refs = historical_references()
    config_audit, dataset_records, model_infos = {}, {}, {}
    for dataset, seed in CELLS:
        config_audit[f"{dataset}_seed{seed}"] = assert_config_and_data_protocol(dataset, seed, caches[dataset], "cpu")
        cfg, data, fingerprints, alignment = v13.load_safe_data(dataset, VARIANT, seed, caches[dataset], "cpu")
        model_infos[dataset] = v13.data_info(data)
        if bool((data.y[data.test_idx] != -100).any()):
            raise RuntimeError(f"{dataset}/seed{seed}: Test labels are not masked")
        dataset_records[f"{dataset}_seed{seed}"] = {
            "split_seed": 42, "feature_fingerprint": caches[dataset]["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "split_file_sha256": alignment["split_file_sha256"],
            "semantic_cache_sha256": sha256_file(caches[dataset]["cache_path_abs"]),
            "candidate_fingerprint": caches[dataset]["candidate_fingerprint"],
            "v7b_fingerprint": caches[dataset]["v7b_fingerprint"], "v7b_k": 8,
            "test_labels_masked": True,
        }
    reference_manifest = {}
    for (dataset, seed), item in refs.items():
        row = item["row"]
        current = dataset_records[f"{dataset}_seed{seed}"]
        for field in ("feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint",
                      "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
                      "candidate_fingerprint", "v7b_fingerprint"):
            if row.get(field) and row[field] != current[field]:
                raise RuntimeError(f"historical/data mismatch for {dataset}/seed{seed}: {field}")
        reference_manifest[f"{dataset}_seed{seed}"] = {
            "source_git_sha": row.get("source_git_sha", ""),
            "metrics": {key: float(row[key]) for key in ("val_acc", "val_macro_f1", "val_ce", "train_acc", "train_macro_f1", "train_ce") if row.get(key) not in (None, "")},
            "initial_model_sha256": row.get("initial_model_sha256", ""),
            "initial_classifier_sha256": row.get("initial_classifier_sha256", ""),
            "checkpoint_hashes_verified": item["checkpoint_hashes"],
        }
    gpu = gpu_snapshot_for_one()
    manifest = {
        "study": "MR-CGE V1.3-A Node-Conditioned vs View-Global Router Ablation",
        "branch": BRANCH, "base_branch": BASE_BRANCH, "base_sha": BASE_SHA,
        "training_source_sha": frozen_sha, "source_file_sha256s": frozen_hashes,
        "model": MODEL, "condition": CONDITION, "v13_variant": VARIANT,
        "loss_weights": {"unimodal_weight": 0.3, "aux_weight": 1.0, "balance_weight": 0.01},
        "configuration_audit": config_audit, "dataset_records": dataset_records,
        "historical_references": reference_manifest, "gpu_smoke_test": smoke,
        "model_parameter_count": None, "base_parameter_count": None,
        "authorized_cells": [{"dataset": d, "seed": s} for d, s in CELLS],
        "completed_cells": [], "training_status": "not_started", "test_evaluation": False,
        "link_prediction": False, "physical_gpu": PHYSICAL_GPU,
        "gpu_preflight": {"uuid": gpu["device"]["uuid"], "free_mib": gpu["device"]["memory_free_mib"],
                          "processes": gpu["processes"]}, "started_utc": now(),
    }
    json_write(DATA_DIR / "manifest.json", manifest)
    try:
        # Record parameter equality at a fixed seed before any formal run.
        cache = caches["Movies"]
        base_cfg = v13.make_cfg("Movies", VARIANT, 42, "cpu", cache)
        global_cfg = v13.make_cfg("Movies", VARIANT, 42, "cpu", cache)
        global_cfg.model = candidate_model_config(cache)
        set_seed(42); base_model = build_model(base_cfg, model_infos["Movies"])
        set_seed(42); global_model = build_model(global_cfg, model_infos["Movies"])
        if set(base_model.state_dict()) != set(global_model.state_dict()) or any(
                not torch.equal(value, global_model.state_dict()[name]) for name, value in base_model.state_dict().items()):
            raise RuntimeError("initial model State Dict differs between node and view-global conditions")
        base_count, global_count = sum(p.numel() for p in base_model.parameters()), sum(p.numel() for p in global_model.parameters())
        if base_count != global_count:
            raise RuntimeError("model parameter counts differ")
        manifest["base_parameter_count"] = base_count
        manifest["model_parameter_count"] = global_count
        manifest["parameter_counts_equal"] = base_count == global_count
        json_write(DATA_DIR / "manifest.json", manifest)
        del base_model, global_model
        for index, (dataset, seed) in enumerate(CELLS, start=1):
            if source_sha() != frozen_sha or source_hashes() != frozen_hashes:
                raise RuntimeError("training source changed after source freeze")
            manifest["training_status"] = "running"
            manifest["current_cell"] = {"dataset": dataset, "seed": seed, "ordinal": index, "total": len(CELLS)}
            json_write(DATA_DIR / "manifest.json", manifest)
            row = run_one(dataset, seed, frozen_sha)
            manifest["completed_cells"].append({"dataset": dataset, "seed": seed, "status": "complete",
                                                 "checkpoint_sha256": row["checkpoint_sha256"]})
            manifest.pop("current_cell", None)
            json_write(DATA_DIR / "manifest.json", manifest)
            print(f"COMPLETE {dataset}/seed{seed}: val_acc={row['val_acc']:.6f} val_f1={row['val_macro_f1']:.6f} val_ce={row['val_ce']:.6f}", flush=True)
        if len(read_rows(DATA_DIR / "runs.csv")) != len(CELLS):
            raise RuntimeError("not all five formal run rows exist")
        paired, summaries = summarize(refs)
        decision = decision_label(paired)
        manifest.update({"training_status": "complete", "completed_utc": now(),
                         "decision": decision[0], "decision_reason": decision[1],
                         "final_metrics": read_rows(DATA_DIR / "runs.csv"),
                         "parameter_counts_equal": base_count == global_count})
        manifest.pop("current_cell", None)
        json_write(DATA_DIR / "manifest.json", manifest)
        report(paired, summaries, decision)
    except Exception as exc:
        manifest["training_status"] = "failed_or_blocked"
        manifest["failure_utc"] = now(); manifest["failure"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        json_write(DATA_DIR / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=("Movies", "Grocery"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--attempt-dir")
    args = parser.parse_args()
    if args.gpu_smoke:
        result = gpu_smoke()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.run_all:
        run_all()
    elif args.worker:
        if not args.dataset or args.seed is None or not args.attempt_dir:
            parser.error("--worker requires --dataset, --seed, and --attempt-dir")
        worker(args.dataset, args.seed, args.attempt_dir)
    else:
        parser.error("choose --gpu-smoke or --run-all")


if __name__ == "__main__":
    main()
