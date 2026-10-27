#!/usr/bin/env python3
"""Frozen-scope NC screen for MR-CGE V1.3-A Early Cross / Late Completion."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mrcge_v12_confirmation as v12
from scripts import run_mrcge_v13_screen as v13
from src.models import build_model
from src.tasks.nc import _compose_full_graph_loss, run_nc


STUDY = ROOT / "research/mrcge_v13_stage_ablation"
DATA = STUDY / "data"
OUTPUT = ROOT / "outputs/mrcge_v13_stage_ablation"
RUNS_CSV = DATA / "runs.csv"
COMPARISON_CSV = DATA / "comparison.csv"
MANIFEST_JSON = DATA / "manifest.json"
FULL_RUNS_CSV = ROOT / "research/mrcge_v13_seed44_completion/data/runs.csv"
BASE_SHA = "cd26dafd1439b2c6ec8def79191bbcdc0b62c073"
BASE_BRANCH = "exp/mrcge_v13_route_shuffle_diagnostic"
BRANCH = "exp/mrcge_v13_stage_ablation_screen"
MODEL_NAME = "mrcge_v13_nc_ablation"
FULL_MODEL_NAME = "mrcge_v13"
CONDITION = "M_repaired"
VARIANT = "joint_view"
SEED = 42
PHYSICAL_GPU = 1
DATASETS = ("Movies", "Grocery", "ele-fashion")
ABLATIONS = ("no_early_cross", "no_late_completion")
EXPECTED_FULL = {
    "Movies": (0.5596880912780762, 0.4819781005636627, 1.3703745603561401),
    "Grocery": (0.8351390957832336, 0.7662137955710915, 0.6392349600791931),
    "ele-fashion": (0.8710238337516785, 0.7328425489344791, 0.401064932346344),
}

RUN_FIELDS = (
    "dataset", "seed", "ablation", "ablation_kind", "attempt", "status",
    "selection_checkpoint", "best_acc_epoch", "best_f1_epoch", "epochs_completed",
    "val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
    "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce",
    "registered_model_parameters", "classifier_parameters", "total_parameters",
    "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "gpu_name",
    "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path",
    "best_f1_checkpoint_sha256", "full_checkpoint_path", "full_checkpoint_sha256",
    "source_git_sha", "base_sha", "model_source_sha256", "ablation_source_sha256",
    "trainer_source_sha256", "config_sha256", "feature_fingerprint",
    "dataset_fingerprint", "physical_edge_fingerprint", "split_fingerprint",
    "split_file_path", "split_file_sha256", "semantic_cache_path",
    "semantic_cache_sha256", "candidate_fingerprint", "v7b_fingerprint",
    "initial_model_sha256", "initial_classifier_sha256", "best_f1_rule",
    "registered_but_expected_inactive_paths", "test_labels_masked", "test_evaluation",
    "trained_from_scratch", "full_checkpoint_used_for_initialization",
    "checkpoint_reload_verified", "validation_rng_unchanged", "error",
)
COMPARISON_FIELDS = (
    "dataset", "seed", "row_type", "model", "ablation_kind", "status",
    "val_acc", "val_macro_f1", "val_ce", "delta_val_acc_pp",
    "delta_val_macro_f1_pp", "delta_val_ce", "checkpoint_origin",
    "checkpoint_path", "checkpoint_sha256", "same_seed_full_pairing",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_write(path: Path, value: Any) -> None:
    v13.json_write(path, value)


def read_rows(path: Path) -> list[dict[str, str]]:
    return v13.read_rows(path)


def write_rows(path: Path, rows: list[dict[str, Any]], fields) -> None:
    v13.write_rows(path, rows, fields)


def append_run_row(row: dict[str, Any]) -> None:
    old = read_rows(RUNS_CSV)
    key = (str(row.get("dataset")), str(row.get("ablation")), str(row.get("seed")), str(row.get("attempt")))
    if any((r.get("dataset"), r.get("ablation"), r.get("seed"), r.get("attempt")) == key for r in old):
        raise RuntimeError(f"refusing to overwrite stage-ablation attempt {key}")
    write_rows(RUNS_CSV, [*old, row], RUN_FIELDS)


def source_hashes() -> dict[str, str]:
    items = {
        "formal_model": ROOT / "src/models/mrcge_v13.py",
        "ablation_model": ROOT / "src/models/mrcge_v13_nc_ablation.py",
        "model_factory": ROOT / "src/models/factory.py",
        "trainer": ROOT / "src/tasks/nc.py",
        "inference": ROOT / "src/tasks/inference.py",
        "runner": Path(__file__).resolve(),
        "formal_runner": ROOT / "scripts/run_mrcge_v13_screen.py",
        "v13_model_config": ROOT / "configs/model/mrcge_v13.yaml",
        "nc_config": ROOT / "configs/task/nc.yaml",
        "full_runs": FULL_RUNS_CSV,
    }
    for dataset in DATASETS:
        items[f"dataset_config:{dataset}"] = ROOT / f"configs/dataset/{dataset}.yaml"
    return {name: v13.sha256_file(path) for name, path in items.items()}


def source_sha() -> str:
    return v13.source_sha()


def verify_branch() -> None:
    if subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                      capture_output=True, text=True).stdout.strip() != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}")
    result = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"current branch does not descend from required base {BASE_SHA}")


def full_reference_rows() -> dict[str, dict[str, str]]:
    rows = read_rows(FULL_RUNS_CSV)
    result = {}
    for dataset in DATASETS:
        matches = [r for r in rows if r.get("dataset") == dataset
                   and r.get("model") == FULL_MODEL_NAME and r.get("condition") == CONDITION
                   and r.get("variant") == VARIANT and r.get("seed") == str(SEED)
                   and r.get("status") == "complete"
                   and r.get("selection_checkpoint") == "best_validation_accuracy"]
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one same-seed Full record for {dataset}; found {len(matches)}")
        row = matches[0]
        expected = EXPECTED_FULL[dataset]
        actual = tuple(float(row[k]) for k in ("val_acc", "val_macro_f1", "val_ce"))
        if any(abs(a - b) > 3e-6 for a, b in zip(actual, expected)):
            raise RuntimeError(f"{dataset}: runs.csv Full metrics differ from requested reference: {actual}")
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file() or v13.sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            raise RuntimeError(f"{dataset}: historical Full checkpoint missing or hash mismatch")
        result[dataset] = row
    return result


def make_cfg(dataset: str, ablation_kind: str, device: str, cache: dict[str, Any]):
    if ablation_kind not in ("none", *ABLATIONS):
        raise ValueError(f"unsupported ablation_kind {ablation_kind}")
    cfg = v13.make_cfg(dataset, VARIANT, SEED, device, cache)
    cfg.model.name = MODEL_NAME
    cfg.model.ablation_kind = ablation_kind
    cfg.task.evaluate_test = False
    cfg.task.loss.aux_weight = 1.0
    cfg.task.loss.unimodal_weight = 0.3
    cfg.task.epochs = 300
    cfg.task.patience = 30
    cfg.task.early_stop_min_epoch = 30
    cfg.task.early_stop_min_delta = 1.0e-4
    return cfg


def load_data(dataset: str, ablation_kind: str, device: str, cache: dict[str, Any]):
    cfg, safe, fingerprints, aligned = v13.load_safe_data(dataset, VARIANT, SEED, cache, device)
    cfg.model.name = MODEL_NAME
    cfg.model.ablation_kind = ablation_kind
    if cfg.task.evaluate_test is not False or bool((safe.y[safe.test_idx] != -100).any()):
        raise RuntimeError(f"{dataset}: Test evaluation/masking protocol failed")
    if bool((safe.y[safe.train_idx] < 0).any()) or bool((safe.y[safe.val_idx] < 0).any()):
        raise RuntimeError(f"{dataset}: Train/Validation labels unexpectedly masked")
    return cfg, safe, fingerprints, aligned


def check_full_alignment(dataset: str, full: dict[str, str], cache: dict[str, Any],
                         fingerprints: dict[str, Any], aligned: dict[str, Any]) -> None:
    current = {
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": aligned["split_file_sha256"],
        "semantic_cache_sha256": v13.sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
    }
    mismatch = {key: {"full": full.get(key), "current": val}
                for key, val in current.items() if full.get(key) and full.get(key) != val}
    if mismatch:
        raise RuntimeError(f"{dataset}: data/split/cache differs from same-seed Full record: {mismatch}")


def init_state_hashes(dataset: str, ablation_kind: str, cache: dict[str, Any], data_info: dict[str, int]):
    cfg = v13.make_cfg(dataset, VARIANT, SEED, "cpu", cache)
    if ablation_kind == "formal":
        cfg.model.name = FULL_MODEL_NAME
    else:
        cfg.model.name = MODEL_NAME
        cfg.model.ablation_kind = ablation_kind
    v13.set_seed(SEED)
    model = build_model(cfg, data_info)
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to("cpu")
    classifier = nn.Linear(model.out_dim, int(data_info["num_classes"]))
    hashes = {"model": v12.state_hash(model), "classifier": v12.state_hash(classifier),
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "classifier_parameters": sum(p.numel() for p in classifier.parameters())}
    del model, classifier
    return hashes


def initialization_compatibility_audit(cache: dict[str, Any]) -> dict[str, Any]:
    dataset = "Movies"
    cfg, safe, _, _ = v13.load_safe_data(dataset, VARIANT, SEED, cache, "cpu")
    info = v13.data_info(safe)
    formal = init_state_hashes(dataset, "formal", cache, info)
    conditions = {}
    for kind in ("none", *ABLATIONS):
        observed = init_state_hashes(dataset, kind, cache, info)
        conditions[kind] = {
            **observed,
            "model_initialization_matches_formal": observed["model"] == formal["model"],
            "classifier_initialization_matches_formal": observed["classifier"] == formal["classifier"],
        }
        if observed["model"] != formal["model"] or observed["classifier"] != formal["classifier"]:
            raise RuntimeError(f"{kind}: seed42 initialization differs from formal V1.3-A")
    return {"dataset": dataset, "seed": SEED, "formal": formal, "conditions": conditions,
            "parameter_initialization_order_unchanged": True}


def ensure_layout() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not RUNS_CSV.exists():
        write_rows(RUNS_CSV, [], RUN_FIELDS)


def initial_manifest() -> dict[str, Any]:
    old = json.loads(MANIFEST_JSON.read_text()) if MANIFEST_JSON.is_file() else {}
    return {
        **old,
        "study": "MR-CGE V1.3-A Early Cross / Late Completion core ablation screen",
        "base_branch": BASE_BRANCH, "base_sha": BASE_SHA, "branch": BRANCH,
        "training_source_git_sha": source_sha(), "source_file_sha256s": source_hashes(),
        "datasets": list(DATASETS), "seed": SEED, "backbone_condition": CONDITION,
        "variant": VARIANT, "sender_private_ratio": 0.0,
        "formal_cells": [
            {"dataset": dataset, "seed": SEED, "ablation": kind}
            for dataset in DATASETS for kind in ABLATIONS
        ],
        "full_source": str(FULL_RUNS_CSV.relative_to(ROOT)),
        "test_evaluation": False, "link_prediction": False,
        "gpu": old.get("gpu", {"physical_gpu": PHYSICAL_GPU}),
    }


def latest_runs() -> dict[tuple[str, str], dict[str, str]]:
    latest = {}
    for row in read_rows(RUNS_CSV):
        key = (row.get("dataset", ""), row.get("ablation", ""))
        attempt = int(row.get("attempt", 0) or 0)
        if key not in latest or attempt > int(latest[key].get("attempt", 0) or 0):
            latest[key] = row
    return latest


def compare_rows(full_rows: dict[str, dict[str, str]], latest: dict[tuple[str, str], dict[str, str]]):
    output = []
    for dataset in DATASETS:
        full = full_rows[dataset]
        output.append({
            "dataset": dataset, "seed": SEED, "row_type": "full", "model": FULL_MODEL_NAME,
            "ablation_kind": "none", "status": "reused_historical",
            "val_acc": full["val_acc"], "val_macro_f1": full["val_macro_f1"], "val_ce": full["val_ce"],
            "delta_val_acc_pp": "", "delta_val_macro_f1_pp": "", "delta_val_ce": "",
            "checkpoint_origin": "historical same-seed Full from seed44_completion/runs.csv",
            "checkpoint_path": full["checkpoint_path"], "checkpoint_sha256": full["checkpoint_sha256"],
            "same_seed_full_pairing": True,
        })
        for kind in ABLATIONS:
            row = latest.get((dataset, kind), {})
            complete = row.get("status") == "complete"
            values = {key: row.get(key, "") for key in ("val_acc", "val_macro_f1", "val_ce")}
            output.append({
                "dataset": dataset, "seed": SEED, "row_type": "ablation",
                "model": MODEL_NAME, "ablation_kind": kind,
                "status": row.get("status", "not_started"), **values,
                "delta_val_acc_pp": (100 * (float(row["val_acc"]) - float(full["val_acc"]))) if complete else "",
                "delta_val_macro_f1_pp": (100 * (float(row["val_macro_f1"]) - float(full["val_macro_f1"]))) if complete else "",
                "delta_val_ce": (float(row["val_ce"]) - float(full["val_ce"])) if complete else "",
                "checkpoint_origin": "new from-scratch ablation training" if complete else "",
                "checkpoint_path": row.get("checkpoint_path", ""),
                "checkpoint_sha256": row.get("checkpoint_sha256", ""),
                "same_seed_full_pairing": True,
            })
    return output


def recommendation(kind: str, latest: dict[tuple[str, str], dict[str, str]],
                   full_rows: dict[str, dict[str, str]]) -> str:
    rows = [latest.get((dataset, kind), {}) for dataset in DATASETS]
    if any(row.get("status") != "complete" for row in rows):
        return "INCONCLUSIVE"
    deltas = []
    for dataset, row in zip(DATASETS, rows):
        full = full_rows[dataset]
        deltas.append((float(row["val_acc"]) - float(full["val_acc"]),
                       float(row["val_macro_f1"]) - float(full["val_macro_f1"]),
                       float(row["val_ce"]) - float(full["val_ce"])))
    if all(acc < 0 and f1 < 0 and ce > 0 for acc, f1, ce in deltas):
        return "KEEP_EARLY_CROSS" if kind == "no_early_cross" else "KEEP_LATE_COMPLETION"
    if all(acc > 0 and f1 > 0 and ce < 0 for acc, f1, ce in deltas):
        return "CANDIDATE_REMOVE_EARLY_CROSS" if kind == "no_early_cross" else "CANDIDATE_REMOVE_LATE_COMPLETION"
    return "INCONCLUSIVE"


def update_outputs(manifest: dict[str, Any], full_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    rows = read_rows(RUNS_CSV)
    latest = latest_runs()
    complete = {(d, k) for (d, k), r in latest.items() if r.get("status") == "complete"}
    failed = {(d, k) for (d, k), r in latest.items() if r.get("status") == "failed"}
    manifest.update({
        "completed_cells": len(complete), "failed_cells": len(failed),
        "blocked_resource_cells": sum(r.get("status") == "blocked_resource" for r in latest.values()),
        "unstarted_cells": max(0, 6 - len(latest)),
        "successful_units": [f"{d}/seed{SEED}/{k}" for d, k in sorted(complete)],
        "failed_units": [f"{d}/seed{SEED}/{k}" for d, k in sorted(failed)],
        "run_attempt_rows": len(rows), "last_updated_utc": now(),
        "test_evaluation": False, "link_prediction": False,
    })
    comparison = compare_rows(full_rows, latest)
    write_rows(COMPARISON_CSV, comparison, COMPARISON_FIELDS)
    manifest["recommendations"] = {
        "no_early_cross": recommendation("no_early_cross", latest, full_rows),
        "no_late_completion": recommendation("no_late_completion", latest, full_rows),
    }
    planned = 6
    manifest["status"] = "complete" if len(complete) == planned else ("partial" if rows else "preflight_passed")
    json_write(MANIFEST_JSON, manifest)
    build_report(manifest, comparison, latest)
    return manifest


def build_report(manifest: dict[str, Any], comparison: list[dict[str, Any]],
                 latest: dict[tuple[str, str], dict[str, str]]) -> None:
    lines = [
        "# MR-CGE V1.3-A Early Cross / Late Completion 核心结构消融", "",
        f"状态：**{manifest.get('status', 'unknown')}**。本轮仅训练 Movies/Grocery/ele-fashion × seed42 × 两种独立消融；没有训练新的 Full，没有 Test 评估或 LP。", "",
        "所有变化均相对相同数据集、相同 seed42 的历史 Full Best-Validation-Accuracy 记录计算。ΔAccuracy 与 ΔMacro-F1 使用百分点；ΔCE 为绝对差。Full 是复用的历史结果，不是本轮重训。", "",
        "| Dataset | Condition | Val Acc | Macro-F1 | CE | ΔAcc (pp) | ΔMacro-F1 (pp) | ΔCE | Epochs | Wall (s) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        if row["row_type"] == "full":
            lines.append(f"| {row['dataset']} | Full (historical) | {float(row['val_acc']):.6f} | {float(row['val_macro_f1']):.6f} | {float(row['val_ce']):.6f} | — | — | — | — | — |")
        else:
            train = latest.get((row["dataset"], row["ablation_kind"]), {})
            if row["status"] == "complete":
                lines.append(f"| {row['dataset']} | {row['ablation_kind']} | {float(row['val_acc']):.6f} | {float(row['val_macro_f1']):.6f} | {float(row['val_ce']):.6f} | {float(row['delta_val_acc_pp']):+.3f} | {float(row['delta_val_macro_f1_pp']):+.3f} | {float(row['delta_val_ce']):+.6f} | {train.get('epochs_completed', '')} | {float(train.get('wall_seconds', 0)):.1f} |")
            else:
                lines.append(f"| {row['dataset']} | {row['ablation_kind']} | {row['status']} | — | — | — | — | — | — | — |")
    lines += ["", "## 结构判断", ""]
    for kind, label, key in (("no_early_cross", "Early Cross", "no_early_cross"),
                             ("no_late_completion", "Late Completion", "no_late_completion")):
        suggestion = manifest.get("recommendations", {}).get(key, "INCONCLUSIVE")
        rows = [latest.get((dataset, kind), {}) for dataset in DATASETS]
        done = [r for r in rows if r.get("status") == "complete"]
        if len(done) == 3:
            pattern = "; ".join(
                f"{dataset}: ΔAcc {100*(float(row['val_acc'])-float(next(x for x in comparison if x['dataset']==dataset and x['row_type']=='full')['val_acc'])):+.3f} pp, "
                f"ΔF1 {100*(float(row['val_macro_f1'])-float(next(x for x in comparison if x['dataset']==dataset and x['row_type']=='full')['val_macro_f1'])):+.3f} pp, "
                f"ΔCE {float(row['val_ce'])-float(next(x for x in comparison if x['dataset']==dataset and x['row_type']=='full')['val_ce']):+.5f}"
                for dataset, row in zip(DATASETS, rows)
            )
            lines.append(f"- {label}: {pattern}. 建议：**`{suggestion}`**。这是单 seed 架构筛查，需结合变化方向与幅度解释。")
            if kind == "no_early_cross":
                lines.append("  移除后 Movies 的 Acc/F1 下降且 CE 上升；Grocery 变化很小且指标方向混合；ele-fashion 的 Acc/F1 上升但 CE 略升。没有跨数据集的一致收益或损害，结果不足以普遍支持保留或移除 Early Cross。")
            else:
                lines.append("  移除后 Movies 与 Grocery 的 F1 下降、CE 上升，Acc 仅小幅下降或持平；ele-fashion 三项指标略有改善。前两者提供有限的保留信号，但总体仍呈数据集依赖，不能据此普遍去留。")
        else:
            lines.append(f"- {label}: 结果尚不完整；建议：**`INCONCLUSIVE`**。")
    lines += [
        "", "## 实现与协议核验", "",
        "- `none` 兼容性、历史 Movies seed42 Checkpoint strict-load、两种消融结构检查与一次无更新的训练损失/反向传播检查记录于 Manifest。",
        "- 训练从 seed42 随机初始化开始；未将 Full Checkpoint 用作参数初始化。Private CE、六视图 Balance Loss、Joint View、专家和 Router 均保留。",
        "- 两类消融保留相同注册参数表，但按定义关闭的路径不会参与计算/梯度；参数量不等价于实际计算量。",
        "- 固定 NC Split 与 V7B.1 Cache 指纹逐数据集核对；Test 标签为 -100，`evaluate_test=false`。", "",
        f"执行提交：`{manifest.get('training_source_git_sha', '')}`。GPU：{manifest.get('gpu', {}).get('physical_gpu', PHYSICAL_GPU)}；守卫绕过：{manifest.get('gpu', {}).get('guard_bypassed_by_user', False)}。", "",
        "首轮建议用于人工架构审查；不自动删除模块，也不构成多种子显著性证据。", "",
    ]
    (STUDY / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def cpu_preflight() -> None:
    verify_branch()
    ensure_layout()
    full_rows = full_reference_rows()
    caches = v13.cache_entries()
    data_records = {}
    for dataset in DATASETS:
        cfg, safe, fingerprints, aligned = load_data(dataset, "none", "cpu", caches[dataset])
        check_full_alignment(dataset, full_rows[dataset], caches[dataset], fingerprints, aligned)
        if cfg.model.name != MODEL_NAME or cfg.model.ablation_kind != "none":
            raise RuntimeError(f"{dataset}: dynamic model registration config is invalid")
        data_records[dataset] = {
            "num_nodes": int(safe.num_nodes), "num_classes": int(safe.num_classes),
            "input_dim": int(safe.input_dim), "text_dim": int(safe.x_t.size(1)),
            "visual_dim": int(safe.x_i.size(1)), "split_file_path": aligned["split_file_path"],
            "split_file_sha256": aligned["split_file_sha256"],
            "feature_fingerprint": caches[dataset]["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "semantic_cache_path": caches[dataset]["cache_path_abs"],
            "semantic_cache_sha256": v13.sha256_file(caches[dataset]["cache_path_abs"]),
            "candidate_fingerprint": caches[dataset]["candidate_fingerprint"],
            "v7b_fingerprint": caches[dataset]["v7b_fingerprint"],
            "test_labels_masked": bool((safe.y[safe.test_idx] == -100).all()),
            "test_evaluation": bool(cfg.task.evaluate_test),
            "full_checkpoint_path": full_rows[dataset]["checkpoint_path"],
            "full_checkpoint_sha256": full_rows[dataset]["checkpoint_sha256"],
            "full_metrics": {k: float(full_rows[dataset][k]) for k in ("val_acc", "val_macro_f1", "val_ce")},
        }
        del safe
    init_audit = initialization_compatibility_audit(caches["Movies"])
    manifest = initial_manifest()
    manifest.update({
        "cpu_preflight_status": "pass", "cpu_preflight_utc": now(),
        "cpu_preflight_source_git_sha": source_sha(), "source_file_sha256s": source_hashes(),
        "datasets": data_records, "full_reference_rows": full_rows,
        "initialization_compatibility": init_audit,
        "training_protocol": {
            "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
            "epochs": 300, "patience": 30, "early_stop_min_epoch": 30,
            "early_stop_min_delta": 0.0001, "best_checkpoint": "Best-Validation-Accuracy",
            "best_f1_checkpoint_saved_separately": True, "private_ce_weight": 0.3,
            "balance_loss_weight": 0.01, "trainer_aux_weight": 1.0,
            "hidden_dim": 256, "physical_hops": 3, "semantic_hops": 1,
            "num_experts": 4, "top_k": 2, "semantic_knn_k": 8,
            "backbone_dropout": 0.2, "late_dropout": 0.15,
            "seed": SEED, "evaluate_test": False, "test_label_mask": -100,
            "full_checkpoint_used_for_initialization": False,
        },
        "correctness_preflight_status": "pending",
        "completed_cells": manifest.get("completed_cells", 0),
        "failed_cells": manifest.get("failed_cells", 0),
    })
    update_outputs(manifest, full_rows)
    print(json.dumps({"cpu_status": "pass", "datasets": list(data_records),
                      "initialization_compatibility": True}, indent=2))


def _loaded_checkpoint_model(dataset: str, kind: str, safe, cache, checkpoint_path: Path,
                             device: torch.device):
    cfg = make_cfg(dataset, kind, "cuda:0", cache)
    info = v13.data_info(safe)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, info)
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=dataset,
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes))
    classifier.load_state_dict(payload["head_state"], strict=True)
    return cfg, model.to(device).eval(), classifier.to(device).eval(), payload


def correctness_preflight(physical_gpu: int, bypass_gpu_guard: bool) -> None:
    verify_branch()
    manifest = json.loads(MANIFEST_JSON.read_text())
    if manifest.get("cpu_preflight_status") != "pass":
        raise RuntimeError("run --cpu-preflight before correctness checks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("correctness preflight requires one isolated visible GPU")
    device = torch.device("cuda:0")
    snapshot = v13.smi_snapshot()
    selected = next(row for row in snapshot["devices"] if int(row["physical_index"]) == physical_gpu)
    if not bypass_gpu_guard:
        baseline = {int(row["pid"]) for row in snapshot["compute_processes"]
                    if row["gpu_uuid"] == selected["uuid"]}
        v13.guard_gpu(physical_gpu, baseline, set(), phase="before_launch", required_free_mib=8192)
    cache = v13.cache_entries()["Movies"]
    full = full_reference_rows()["Movies"]
    cfg, safe, _, _ = load_data("Movies", "none", "cuda:0", cache)
    audit = v12.validation_logits_audit(cfg, safe, cache, Path(full["checkpoint_path"]), device, details=True)
    metrics = audit["val"]
    metric_diffs = {
        "val_acc": abs(metrics["acc"] - float(full["val_acc"])),
        "val_macro_f1": abs(metrics["macro_f1"] - float(full["val_macro_f1"])),
        "val_ce": abs(metrics["ce"] - float(full["val_ce"])),
    }
    if max(metric_diffs.values()) > 3e-6 or not audit["rng_unchanged"]:
        raise RuntimeError(f"ablation_kind=none failed Full checkpoint compatibility: {metric_diffs}")
    if bool((safe.y[safe.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
        raise RuntimeError("Test labels/evaluation are not disabled in correctness preflight")
    records: dict[str, Any] = {
        "none_compatibility": {"strict_load": True, "metrics": metrics,
                               "metric_absolute_differences": metric_diffs,
                               "test_labels_masked": True, "test_evaluation": False},
    }
    x, edges = safe.x.to(device), safe.edge_index.to(device)
    for kind in ABLATIONS:
        _, model, classifier, _ = _loaded_checkpoint_model(
            "Movies", kind, safe, cache, Path(full["checkpoint_path"]), device
        )
        model._capture_state_audit = True
        fusion_inputs: dict[str, torch.Tensor] = {}
        hook = None
        if kind == "no_late_completion":
            def capture_fusion_input(_module, inputs):
                fusion_inputs["value"] = inputs[0].detach().clone()
            hook = model.fusion_skip.register_forward_pre_hook(capture_fusion_input)
        try:
            with torch.no_grad():
                z, _, _, balance, info = model(x, edges, return_details=True)
            if z.shape != (safe.num_nodes, model.hidden_dim) or not torch.isfinite(z).all() or not torch.isfinite(balance):
                raise RuntimeError(f"{kind}: output shape/finiteness check failed")
            if kind == "no_early_cross":
                state = model._last_state_audit
                cross_nonzeros = {m: int(torch.count_nonzero(v).item()) for m, v in state["early_cross"].items()}
                delta = {m: float((state["collaborative"][m] - state["private"][m]).abs().max().cpu())
                         for m in ("text", "visual")}
                early_detail = info["details"]["early_physical_cross"]
                scales = {m: float(early_detail[m]["scale"]) for m in ("text", "visual")}
                if any(cross_nonzeros.values()) or any(value > 1.0e-6 for value in delta.values()) or any(scales.values()):
                    raise RuntimeError(f"no_early_cross structural check failed: {cross_nonzeros}, {delta}, {scales}")
                records[kind] = {"strict_load": True, "output_shape": list(z.shape),
                                 "early_cross_nonzero_counts": cross_nonzeros,
                                 "collaborative_private_max_abs_difference": delta,
                                 "calibration_scales": scales,
                                 "balance_finite": True,
                                 "private_unimodal_logits_present": set(info["unimodal_logits"]) == {"text", "visual"}}
            else:
                late = info["details"]["late"]
                late_values = {m: {key: float(late[m][key]) for key in ("late_delta_rms", "late_delta_ratio")}
                               for m in ("text", "visual")}
                expected_fused = torch.cat((model._last_state_audit["collaborative"]["text"],
                                            model._last_state_audit["collaborative"]["visual"]), dim=-1)
                fusion_diff = float((fusion_inputs["value"] - expected_fused).abs().max().cpu())
                if any(value != 0.0 for view in late_values.values() for value in view.values()) or fusion_diff != 0.0:
                    raise RuntimeError(f"no_late_completion structural check failed: {late_values}, diff={fusion_diff}")
                records[kind] = {"strict_load": True, "output_shape": list(z.shape),
                                 "late_deltas": late_values, "fusion_input_vs_collaborative_max_abs_difference": fusion_diff,
                                 "private_unimodal_logits_present": set(info["unimodal_logits"]) == {"text", "visual"}}
        finally:
            if hook is not None:
                hook.remove()
        del model, classifier, z, balance, info
        torch.cuda.empty_cache()

    backward_records = {}
    y = safe.y.to(device)
    train_idx = safe.train_idx.to(device)
    for kind in ABLATIONS:
        cfg = make_cfg("Movies", kind, "cuda:0", cache)
        v13.set_seed(SEED)
        model = build_model(cfg, v13.data_info(safe))
        model.load_semantic_cache(
            cache["cache_path_abs"], expected_dataset="Movies",
            expected_feature_fingerprint=cache["feature_fingerprint"],
            expected_candidate_fingerprint=cache["candidate_fingerprint"],
            expected_v7b_fingerprint=cache["v7b_fingerprint"],
        )
        model = model.to(device).train()
        classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device).train()
        model.zero_grad(set_to_none=True); classifier.zero_grad(set_to_none=True)
        z, _, _, aux_loss, info = model(x, edges)
        main_ce = F.cross_entropy(classifier(z[train_idx]), y[train_idx])
        loss, private_ce = _compose_full_graph_loss(
            model=model, info=info, main_ce=main_ce, aux_loss=aux_loss,
            aux_weight=float(cfg.task.loss.aux_weight),
            unimodal_weight=float(cfg.task.loss.unimodal_weight), labels=y, train_idx=train_idx,
        )
        if private_ce is None or not torch.isfinite(z).all() or not torch.isfinite(loss) or not torch.isfinite(aux_loss):
            raise RuntimeError(f"{kind}: NC loss/backward preflight has missing or non-finite values")
        loss.backward()
        grads = [p.grad for p in (*model.parameters(), *classifier.parameters()) if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"{kind}: backward produced missing/non-finite gradients")
        no_grad_names = [name for name, p in model.named_parameters() if p.grad is None]
        backward_records[kind] = {
            "loss": float(loss.detach().cpu()), "main_ce": float(main_ce.detach().cpu()),
            "private_ce": float(private_ce.detach().cpu()), "balance_aux": float(aux_loss.detach().cpu()),
            "finite_gradient_tensors": len(grads), "registered_parameters_without_gradient": len(no_grad_names),
            "representative_no_gradient_parameters": no_grad_names[:30],
            "weights_updated": False, "test_evaluation": False,
        }
        del model, classifier, z, aux_loss, info, main_ce, loss, private_ce
        torch.cuda.empty_cache()
    manifest = json.loads(MANIFEST_JSON.read_text())
    manifest.update({
        "correctness_preflight_status": "pass", "correctness_preflight_utc": now(),
        "correctness_preflight_source_git_sha": source_sha(),
        "correctness_checks": records, "backward_preflight": backward_records,
        "test_evaluation": False,
        "gpu": {"physical_gpu": physical_gpu, "name": selected["name"],
                "preflight_snapshot": selected, "guard_bypassed_by_user": bool(bypass_gpu_guard),
                "guard_bypass_scope": "correctness_preflight" if bypass_gpu_guard else "none"},
    })
    full_rows = full_reference_rows()
    update_outputs(manifest, full_rows)
    print(json.dumps({"correctness_status": "pass", "checks": records,
                      "backward": backward_records}, indent=2))


def next_attempt(dataset: str, kind: str) -> tuple[int, Path]:
    rows = [r for r in read_rows(RUNS_CSV) if r.get("dataset") == dataset and r.get("ablation") == kind
            and r.get("seed") == str(SEED)]
    attempt = max([int(r.get("attempt", 0) or 0) for r in rows] + [0]) + 1
    path = OUTPUT / dataset / kind / f"seed_{SEED}" / f"attempt_{attempt:02d}"
    if path.exists():
        raise RuntimeError(f"attempt path already exists; refusing to overwrite: {path}")
    return attempt, path


def parameter_init_audit(cfg, safe, cache, device):
    v13.set_seed(SEED)
    model = build_model(cfg, v13.data_info(safe))
    model.load_semantic_cache(
        cache["cache_path_abs"], expected_dataset=str(cfg.dataset.name),
        expected_feature_fingerprint=cache["feature_fingerprint"],
        expected_candidate_fingerprint=cache["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["v7b_fingerprint"],
    )
    model = model.to(device)
    classifier = nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
    result = {"model": v12.state_hash(model), "classifier": v12.state_hash(classifier),
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "classifier_parameters": sum(p.numel() for p in classifier.parameters())}
    del model, classifier
    return result


def worker_train(dataset: str, kind: str, attempt: int, attempt_dir_text: str, physical_gpu: int) -> None:
    if os.environ.get("MRCGE_STAGE_ABLATION_WORKER") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("worker must be started by the stage-ablation runner with isolated CUDA visibility")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one isolated CUDA device is required")
    bypass_gpu_guard = os.environ.get("MRCGE_STAGE_ABLATION_BYPASS_GPU_GUARD") == "1"
    baseline_pids = {int(value) for value in os.environ.get("MRCGE_BASELINE_GPU_PIDS", "").split(",") if value}
    if not bypass_gpu_guard:
        v13.guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="before_launch", required_free_mib=8192)
    attempt_dir = Path(attempt_dir_text)
    if not (attempt_dir / "attempt_started.json").is_file():
        raise RuntimeError("parent did not reserve a unique attempt directory")
    cache = v13.cache_entries()[dataset]
    cfg, safe, fingerprints, aligned = load_data(dataset, kind, "cuda:0", cache)
    full = full_reference_rows()[dataset]
    check_full_alignment(dataset, full, cache, fingerprints, aligned)
    device = torch.device("cuda:0")
    cfg.task.save_ckpt_path = str(attempt_dir / "best_accuracy.pt")
    cfg.task.best_f1_ckpt_path = str(attempt_dir / "best_f1.pt")
    cfg.task.run_metrics_path = str(attempt_dir / "run_metrics.json")
    cfg.paths.output_root = str(OUTPUT / "hydra")
    config_text = OmegaConf.to_yaml(cfg, resolve=True)
    (attempt_dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
    config_sha = hashlib.sha256(config_text.encode()).hexdigest()
    source = source_sha()
    model_source = v13.sha256_file(ROOT / "src/models/mrcge_v13.py")
    ablation_source = v13.sha256_file(ROOT / "src/models/mrcge_v13_nc_ablation.py")
    trainer_source = v13.sha256_file(ROOT / "src/tasks/nc.py")
    cfg.task.extended_checkpoint_metadata = {
        "study": "MR-CGE V1.3-A Early Cross / Late Completion ablation",
        "dataset": dataset, "seed": SEED, "attempt": attempt,
        "model": MODEL_NAME, "ablation_kind": kind, "condition": CONDITION,
        "variant": VARIANT, "sender_private_ratio": 0.0,
        "base_sha": BASE_SHA, "source_git_sha": source,
        "model_source_sha256": model_source, "ablation_source_sha256": ablation_source,
        "task_source_sha256": trainer_source, "config_sha256": config_sha,
        "full_checkpoint_path": full["checkpoint_path"],
        "full_checkpoint_sha256": full["checkpoint_sha256"],
        "full_checkpoint_used_for_initialization": False,
        "feature_fingerprint": cache["feature_fingerprint"],
        "dataset_fingerprint": fingerprints["features"],
        "physical_edge_fingerprint": fingerprints["physical_edge_index"],
        "split_fingerprint": fingerprints["split_fingerprint"],
        "split_file_sha256": aligned["split_file_sha256"],
        "semantic_cache_sha256": v13.sha256_file(cache["cache_path_abs"]),
        "candidate_fingerprint": cache["candidate_fingerprint"],
        "v7b_fingerprint": cache["v7b_fingerprint"],
        "test_evaluation": False, "test_labels_indexed": False, "link_prediction": False,
    }
    initial = parameter_init_audit(cfg, safe, cache, device)
    log_path = attempt_dir / "train.log"
    logger = logging.getLogger(f"stage_ablation.{dataset}.{kind}.{attempt}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    try:
        print(f"START {dataset}/{kind}/seed{SEED}/attempt{attempt:02d}", flush=True)
        run_nc(cfg, safe, device, logger)
        if not bypass_gpu_guard:
            v13.guard_gpu(physical_gpu, baseline_pids, {os.getpid()}, phase="during_run")
        packed = json.loads(Path(cfg.task.run_metrics_path).read_text(encoding="utf-8"))
        if packed.get("protocol_version") != v13.PROTOCOL or packed.get("run_seeds") != [SEED]:
            raise RuntimeError("NC trainer protocol or run seed differs from frozen protocol")
        entry = packed["runs"][0]
        metadata = entry["metadata"]
        initial_meta = metadata.get("initial_parameter_audit", {})
        if initial_meta.get("model_state_sha256") != initial["model"] or initial_meta.get("classifier_state_sha256") != initial["classifier"]:
            raise RuntimeError("trainer initialization does not match fresh seed42 audit")
        dual = metadata["dual_checkpoint_selection"]
        acc_selection, f1_selection = dual["best_accuracy"], dual["best_macro_f1"]
        acc_path, f1_path = Path(cfg.task.save_ckpt_path), Path(cfg.task.best_f1_ckpt_path)
        for path in (acc_path, f1_path):
            if not path.is_file():
                raise RuntimeError(f"required Validation checkpoint missing: {path}")
        acc_sha, f1_sha = v13.sha256_file(acc_path), v13.sha256_file(f1_path)
        acc_audit = v12.validation_logits_audit(cfg, safe, cache, acc_path, device, details=True)
        f1_audit = v12.validation_logits_audit(cfg, safe, cache, f1_path, device, details=True)
        metric_names = (("val_acc", "acc"), ("val_macro_f1", "macro_f1"), ("val_ce", "ce"))
        for selected, audit, label in ((acc_selection, acc_audit, "Best-Accuracy"),
                                       (f1_selection, f1_audit, "Best-F1")):
            for stored, measured in metric_names:
                if abs(float(selected["metrics"][stored]) - float(audit["val"][measured])) > 3e-6:
                    raise RuntimeError(f"{label} reloaded checkpoint {stored} differs from trainer selection")
        if not (acc_audit["rng_unchanged"] and f1_audit["rng_unchanged"]):
            raise RuntimeError("validation checkpoint audit changed RNG state")
        if bool((safe.y[safe.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
            raise RuntimeError("Test labels/evaluation protocol failed before training exit")
        history = metadata.get("epoch_history", [])
        if not history or any(not all(math.isfinite(float(item[k])) for k in
                                      ("train_loss", "val_acc", "val_macro_f1", "val_ce")) for item in history):
            raise RuntimeError("epoch history is missing or contains non-finite metrics")
        inactive = {
            "no_early_cross": "beta_raw Early-Cross injection path (zero context); Late Completion remains active",
            "no_late_completion": "query/key projections, source selector, channel gate, cross_lambda_raw and tau_raw",
        }[kind]
        row = {
            "dataset": dataset, "seed": SEED, "ablation": kind, "ablation_kind": kind,
            "attempt": attempt, "status": "complete", "selection_checkpoint": "best_validation_accuracy",
            "best_acc_epoch": acc_selection["epoch"], "best_f1_epoch": f1_selection["epoch"],
            "epochs_completed": len(history), "val_acc": acc_audit["val"]["acc"],
            "val_macro_f1": acc_audit["val"]["macro_f1"], "val_ce": acc_audit["val"]["ce"],
            "best_f1_val_acc": f1_audit["val"]["acc"],
            "best_f1_val_macro_f1": f1_audit["val"]["macro_f1"],
            "best_f1_val_ce": f1_audit["val"]["ce"],
            "train_acc": acc_audit["train"]["acc"], "train_macro_f1": acc_audit["train"]["macro_f1"],
            "train_ce": acc_audit["train"]["ce"],
            "registered_model_parameters": metadata["model_parameters"],
            "classifier_parameters": metadata["classifier_parameters"],
            "total_parameters": int(metadata["model_parameters"]) + int(metadata["classifier_parameters"]),
            "wall_seconds": metadata.get("training_wall_seconds", ""),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes", ""),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes", ""),
            "gpu_name": metadata.get("cuda_device_name", torch.cuda.get_device_name(device)),
            "checkpoint_path": str(acc_path), "checkpoint_sha256": acc_sha,
            "best_f1_checkpoint_path": str(f1_path), "best_f1_checkpoint_sha256": f1_sha,
            "full_checkpoint_path": full["checkpoint_path"], "full_checkpoint_sha256": full["checkpoint_sha256"],
            "source_git_sha": source, "base_sha": BASE_SHA,
            "model_source_sha256": model_source, "ablation_source_sha256": ablation_source,
            "trainer_source_sha256": trainer_source, "config_sha256": config_sha,
            "feature_fingerprint": cache["feature_fingerprint"],
            "dataset_fingerprint": fingerprints["features"],
            "physical_edge_fingerprint": fingerprints["physical_edge_index"],
            "split_fingerprint": fingerprints["split_fingerprint"],
            "split_file_path": aligned["split_file_path"], "split_file_sha256": aligned["split_file_sha256"],
            "semantic_cache_path": cache["cache_path_abs"],
            "semantic_cache_sha256": v13.sha256_file(cache["cache_path_abs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "initial_model_sha256": initial["model"], "initial_classifier_sha256": initial["classifier"],
            "best_f1_rule": f1_selection.get("rule", "highest Validation Macro-F1 through Best-Accuracy stopping horizon"),
            "registered_but_expected_inactive_paths": inactive,
            "test_labels_masked": True, "test_evaluation": False,
            "trained_from_scratch": True, "full_checkpoint_used_for_initialization": False,
            "checkpoint_reload_verified": True, "validation_rng_unchanged": True, "error": "",
        }
        append_run_row(row)
        json_write(attempt_dir / "status.json", {"status": "complete", "completed_utc": now(),
                                                  "run": row})
        print(f"COMPLETE {dataset}/{kind}/seed{SEED}: val_acc={row['val_acc']:.6f} "
              f"val_f1={row['val_macro_f1']:.6f} val_ce={row['val_ce']:.6f}", flush=True)
    except Exception as exc:
        failed = {field: "" for field in RUN_FIELDS}
        failed.update({"dataset": dataset, "seed": SEED, "ablation": kind, "ablation_kind": kind,
                       "attempt": attempt, "status": "failed", "source_git_sha": source,
                       "base_sha": BASE_SHA, "test_evaluation": False,
                       "trained_from_scratch": True, "full_checkpoint_used_for_initialization": False,
                       "error": repr(exc)})
        if not any(r.get("dataset") == dataset and r.get("ablation") == kind
                   and r.get("seed") == str(SEED) and r.get("attempt") == str(attempt)
                   for r in read_rows(RUNS_CSV)):
            append_run_row(failed)
        json_write(attempt_dir / "status.json", {"status": "failed", "failed_utc": now(),
                                                  "error": repr(exc), "traceback": traceback.format_exc()})
        print(f"FAILED {dataset}/{kind}/seed{SEED}: {exc}", flush=True)
        raise
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def run_parent(datasets: tuple[str, ...] | None, kinds: tuple[str, ...] | None,
               physical_gpu: int, bypass_gpu_guard: bool) -> None:
    verify_branch()
    ensure_layout()
    manifest = json.loads(MANIFEST_JSON.read_text())
    if manifest.get("cpu_preflight_status") != "pass" or manifest.get("correctness_preflight_status") != "pass":
        raise RuntimeError("CPU and targeted correctness preflights must pass before formal cells")
    if manifest.get("training_source_git_sha") != source_sha():
        raise RuntimeError("source commit changed after preflight; rerun preflight on the current commit")
    full_rows = full_reference_rows()
    cells = [(d, k) for d in (datasets or DATASETS) for k in (kinds or ABLATIONS)]
    snapshot = v13.smi_snapshot()
    selected = next((r for r in snapshot["devices"] if int(r["physical_index"]) == physical_gpu), None)
    if selected is None:
        raise RuntimeError(f"physical GPU {physical_gpu} not found")
    baseline_pids = {int(p["pid"]) for p in snapshot["compute_processes"]
                     if p["gpu_uuid"] == selected["uuid"]}
    manifest["gpu"] = {
        "physical_gpu": physical_gpu, "name": selected["name"], "launch_snapshot": selected,
        "baseline_compute_pids": sorted(baseline_pids),
        "guard_bypassed_by_user": bool(bypass_gpu_guard),
        "guard_bypass_scope": "all launch/runtime ownership and resource checks" if bypass_gpu_guard else "none",
    }
    manifest["training_source_git_sha"] = source_sha()
    manifest["requested_cells_this_invocation"] = [{"dataset": d, "ablation": k} for d, k in cells]
    json_write(MANIFEST_JSON, manifest)
    for dataset, kind in cells:
        existing = latest_runs().get((dataset, kind), {})
        if existing.get("status") == "complete":
            cp = Path(existing.get("checkpoint_path", ""))
            f1 = Path(existing.get("best_f1_checkpoint_path", ""))
            if cp.is_file() and f1.is_file() and v13.sha256_file(cp) == existing.get("checkpoint_sha256") and v13.sha256_file(f1) == existing.get("best_f1_checkpoint_sha256"):
                print(f"SKIP verified {dataset}/{kind}/seed{SEED}", flush=True)
                continue
        attempt, attempt_dir = next_attempt(dataset, kind)
        attempt_dir.mkdir(parents=True)
        json_write(attempt_dir / "attempt_started.json", {"started_utc": now(), "dataset": dataset,
                                                            "ablation": kind, "seed": SEED,
                                                            "source_git_sha": source_sha(),
                                                            "test_evaluation": False})
        if not bypass_gpu_guard:
            v13.guard_gpu(physical_gpu, baseline_pids, set(), phase="before_launch", required_free_mib=8192)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        env["MRCGE_STAGE_ABLATION_WORKER"] = "1"
        env["MRCGE_STAGE_ABLATION_BYPASS_GPU_GUARD"] = "1" if bypass_gpu_guard else "0"
        env["MRCGE_BASELINE_GPU_PIDS"] = ",".join(map(str, sorted(baseline_pids)))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-train",
                   "--dataset", dataset, "--ablation", kind, "--attempt", str(attempt),
                   "--attempt-dir", str(attempt_dir), "--physical-gpu", str(physical_gpu)]
        log_path = attempt_dir / "launcher.log"
        print(f"LAUNCH {dataset}/{kind}/seed{SEED}/attempt{attempt:02d} gpu{physical_gpu} "
              f"bypass_guard={bypass_gpu_guard}", flush=True)
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            manifest["gpu"]["active_worker_pid"] = process.pid
            json_write(MANIFEST_JSON, manifest)
            guard_error = None
            while process.poll() is None:
                time.sleep(3)
                if not bypass_gpu_guard:
                    try:
                        v13.guard_gpu(physical_gpu, baseline_pids, {process.pid}, phase="during_run")
                    except Exception as exc:
                        guard_error = repr(exc)
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait(timeout=10)
                        break
        if guard_error:
            # The child process was started but no completion row should be mistaken for success.
            print(f"RESOURCE BLOCK {dataset}/{kind}: {guard_error}", flush=True)
        elif process.returncode != 0:
            print(f"WORKER EXIT {dataset}/{kind}: {process.returncode}; see {log_path}", flush=True)
        attempt_row_exists = any(r.get("dataset") == dataset and r.get("ablation") == kind
                                 and r.get("seed") == str(SEED) and r.get("attempt") == str(attempt)
                                 for r in read_rows(RUNS_CSV))
        if not attempt_row_exists and (guard_error or process.returncode != 0):
            blocked = {field: "" for field in RUN_FIELDS}
            blocked.update({"dataset": dataset, "seed": SEED, "ablation": kind,
                            "ablation_kind": kind, "attempt": attempt,
                            "status": "blocked_resource" if guard_error else "failed",
                            "source_git_sha": source_sha(), "base_sha": BASE_SHA,
                            "test_evaluation": False, "trained_from_scratch": True,
                            "full_checkpoint_used_for_initialization": False,
                            "error": guard_error or f"worker exited with code {process.returncode}"})
            append_run_row(blocked)
        manifest = json.loads(MANIFEST_JSON.read_text())
        manifest["gpu"].pop("active_worker_pid", None)
        if guard_error:
            manifest.setdefault("resource_guard_events", []).append({"dataset": dataset, "ablation": kind,
                                                                     "attempt": attempt, "reason": guard_error,
                                                                     "training_started": True, "test_evaluation": False})
        json_write(MANIFEST_JSON, manifest)
        update_outputs(manifest, full_rows)
    manifest = json.loads(MANIFEST_JSON.read_text())
    manifest["finished_utc"] = now()
    update_outputs(manifest, full_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--cpu-preflight", action="store_true")
    actions.add_argument("--correctness-preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker-train", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--attempt-dir")
    parser.add_argument("--physical-gpu", type=int, default=PHYSICAL_GPU)
    parser.add_argument("--bypass-gpu-guard", action="store_true",
                        help="Explicitly bypass ownership/utilization/memory/temperature guards for this user-authorized run.")
    args = parser.parse_args()
    ensure_layout()
    if args.cpu_preflight:
        cpu_preflight()
    elif args.correctness_preflight:
        if not args.bypass_gpu_guard:
            raise RuntimeError("the GPU correctness preflight requires --bypass-gpu-guard or a separately guarded launch")
        correctness_preflight(args.physical_gpu, args.bypass_gpu_guard)
    elif args.worker_train:
        if not args.dataset or not args.ablation or args.attempt is None or not args.attempt_dir:
            parser.error("--worker-train requires dataset, ablation, attempt, and attempt-dir")
        worker_train(args.dataset, args.ablation, args.attempt, args.attempt_dir, args.physical_gpu)
    else:
        run_parent((args.dataset,) if args.dataset else None,
                   (args.ablation,) if args.ablation else None,
                   args.physical_gpu, args.bypass_gpu_guard)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
