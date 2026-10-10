#!/usr/bin/env python3
"""Preflight, smoke, and run the frozen V6A validation-only NC campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRANCH = "exp/mace_mag_v6a_expert_collaboration"
PARENT_SHA = "0c66c6d65ed5da9e5850b7cfeff2454074e52a3b"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("a0_static", "a1_node", "a2_cross", "a3_intra")
PROTOCOL = "unified_full_graph_nc_v1"
FREEZE_MESSAGE = "Freeze MACE-MAG V6A expert collaboration implementation"
OUTPUT_ROOT = ROOT / "outputs" / "mace_mag_v6a_expert_collaboration"
RESEARCH_ROOT = ROOT / "research" / "mace_mag_v6a_expert_collaboration"
DATA_ROOT = RESEARCH_ROOT / "data"
FORMAL_ARTIFACTS = {
    "REPORT.md",
    "README.md",
    "data/campaign_manifest.json",
    "data/run_rows.json",
    "data/summary.csv",
    "data/paired_comparisons.csv",
    "data/resource_profile.json",
    "data/collaboration_diagnostics.csv",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dirty_paths() -> set[str]:
    output = git("status", "--porcelain", "--untracked-files=all")
    return {line[3:].split(" -> ")[-1] for line in output.splitlines()}


def find_freeze_sha() -> str | None:
    return git("log", "--format=%H", "-1", f"--grep=^{FREEZE_MESSAGE}$") or None


def _fingerprint() -> str:
    files = (
        "configs/model/mace_mag_v6a.yaml",
        "configs/task/nc.yaml",
        "src/models/mace_mag_v6a.py",
        "src/tasks/nc.py",
        "scripts/run_mace_mag_v6a.py",
        "scripts/analyze_mace_mag_v6a.py",
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def provenance(mode: str, resume: bool = False) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    dirty = _dirty_paths()
    if branch != BRANCH:
        raise RuntimeError(f"expected branch {BRANCH}, found {branch}")
    if git("merge-base", PARENT_SHA, "HEAD") != PARENT_SHA:
        raise RuntimeError(f"experiment ancestry must include {PARENT_SHA}")
    freeze = find_freeze_sha()
    if mode == "campaign":
        if freeze is None or head != freeze:
            raise RuntimeError(
                f"formal campaign requires HEAD at freeze commit; head={head}, freeze={freeze}"
            )
        if not resume and dirty:
            raise RuntimeError(
                f"formal campaign requires a clean worktree, found {sorted(dirty)}"
            )
        if resume:
            allowed = {
                f"research/mace_mag_v6a_expert_collaboration/{path}"
                for path in FORMAL_ARTIFACTS
            }
            unexpected = dirty - allowed
            if unexpected:
                raise RuntimeError(
                    f"resume has unexpected modified paths: {sorted(unexpected)}"
                )
    return {
        "branch": branch,
        "parent_commit_sha": PARENT_SHA,
        "head_at_start": head,
        "freeze_commit_sha": freeze,
        "worktree_clean_at_start": not bool(dirty),
        "configuration_fingerprint_sha256": _fingerprint(),
        "started_at_utc": now(),
    }


def _nvidia_smi() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def environment(device: str) -> dict[str, Any]:
    import torch
    import torch_geometric

    device_obj = torch.device(device)
    cuda_properties = None
    if device_obj.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        props = torch.cuda.get_device_properties(index)
        cuda_properties = {
            "device_index": index,
            "device_name": props.name,
            "total_memory_bytes": int(total_bytes),
            "free_memory_bytes": int(free_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        }
    return {
        "recorded_at_utc": now(),
        "python": sys.version,
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": device,
        "selected_device": cuda_properties,
        "protocol": PROTOCOL,
        "nvidia_smi_snapshot": _nvidia_smi(),
        "other_processes_terminated": False,
    }


def load_preflight_dataset(dataset: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load only modality features and the physical graph; no labels or splits."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from src.data.graph_utils import ensure_edge_index, preprocess_edge_index
    from src.data.loaders import _load_dgl_graph, resolve_path

    root_cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    data_root = Path(str(root_cfg.paths.data_root))
    ds_cfg = OmegaConf.load(ROOT / "configs" / "dataset" / f"{dataset}.yaml")
    composed = OmegaConf.create(
        {
            "paths": root_cfg.paths,
            "dataset": ds_cfg,
            "seed": 42,
        }
    )
    OmegaConf.resolve(composed)
    ds_cfg = composed.dataset
    if dataset in {"Movies", "Grocery"}:
        edge_raw, _labels_unused, num_nodes = _load_dgl_graph(resolve_path(ds_cfg.graph_path))
        text = np.load(resolve_path(ds_cfg.text_feat_path), allow_pickle=False).astype(
            np.float32, copy=False
        )
        visual = np.load(resolve_path(ds_cfg.image_feat_path), allow_pickle=False).astype(
            np.float32, copy=False
        )
        if text.shape[0] != num_nodes or visual.shape[0] != num_nodes:
            raise ValueError(f"{dataset} feature/node counts do not match")
        x_text = torch.from_numpy(np.array(text, copy=True))
        x_visual = torch.from_numpy(np.array(visual, copy=True))
        x = torch.cat([x_text, x_visual], dim=-1).contiguous()
        text_dim, visual_dim = int(text.shape[1]), int(visual.shape[1])
    elif dataset == "ele-fashion":
        fashion_root = data_root / "ele-fashion"
        x = torch.load(
            fashion_root / "clip_feat.pt", map_location="cpu", weights_only=False
        ).float().contiguous()
        edge_payload = torch.load(
            fashion_root / "nc_edges-nodeid.pt",
            map_location="cpu",
            weights_only=False,
        )
        edge_raw = ensure_edge_index(edge_payload)
        num_nodes = int(x.size(0))
        text_dim, visual_dim = int(ds_cfg.text_dim), int(ds_cfg.visual_dim)
        if text_dim + visual_dim != x.size(1):
            raise ValueError("ele-fashion configured modality dimensions mismatch features")
    else:
        raise ValueError(f"unsupported preflight dataset: {dataset}")

    edge_index = preprocess_edge_index(
        edge_raw,
        int(num_nodes),
        make_undirected=bool(ds_cfg.get("make_undirected", True)),
        with_self_loops=False,
    )
    info = {
        "dataset": dataset,
        "num_nodes": int(num_nodes),
        "num_edges_raw": int(edge_raw.size(1)),
        "num_edges_physical": int(edge_index.size(1)),
        "input_dim": int(x.size(1)),
        "text_dim": text_dim,
        "visual_dim": visual_dim,
        "num_classes": int(ds_cfg.num_classes),
        "feature_bytes_cpu": int(x.numel() * x.element_size()),
        "labels_or_splits_loaded": False,
        "self_loops_in_propagation": False,
    }
    return x, edge_index, info


def run_preflight(device: str) -> None:
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf

    from src.models.mace_mag_v6a import Model

    prov = provenance("preflight")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "task_evaluate_test": False,
        "preflight_label_policy": "no labels or split indices loaded; synthetic targets only for task-loss backward",
        "device": device,
        "expected_datasets": list(DATASETS),
        "datasets": [],
        "environment": environment(device),
        "passed": False,
        "provenance": prov,
    }
    write_json(DATA_ROOT / "preflight_summary.json", summary)
    for dataset in DATASETS:
        start = time.perf_counter()
        x_cpu, edge_cpu, data_info = load_preflight_dataset(dataset)
        row: dict[str, Any] = dict(data_info)
        row["forward_backward"] = {"status": "not_run"}
        x = x_cpu.to(device_obj)
        edge_index = edge_cpu.to(device_obj)
        del x_cpu, edge_cpu
        torch.manual_seed(42)
        cfg = OmegaConf.load(ROOT / "configs" / "model" / "mace_mag_v6a.yaml")
        model_cfg = OmegaConf.create({"model": cfg})
        model = Model(model_cfg, data_info).to(device_obj).train()
        head = nn.Linear(model.out_dim, int(data_info["num_classes"])).to(device_obj)
        if device_obj.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_obj)
        train_count = max(1, int(0.6 * x.size(0)))
        synthetic_labels = torch.randint(
            int(data_info["num_classes"]), (train_count,), device=device_obj
        )
        run_started = time.perf_counter()
        try:
            z, _, _, aux_loss, _ = model(x, edge_index)
            logits = head(z[:train_count])
            loss = nn.functional.cross_entropy(logits, synthetic_labels) + aux_loss
            loss.backward()
            gradient_diagnostics = model.capture_gradient_diagnostics()
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            finite = bool(torch.isfinite(z).all() and torch.isfinite(loss))
            attention_gradients = [
                value["gradient_nonzero"]
                for key, value in gradient_diagnostics.items()
                if ".key.weight" in key or ".value.weight" in key
            ]
            passed = finite and bool(attention_gradients) and all(attention_gradients)
            row["forward_backward"] = {
                "status": "passed" if passed else "failed",
                "synthetic_loss_finite": finite,
                "attention_key_value_gradients_nonzero": all(attention_gradients),
                "gradient_diagnostics": gradient_diagnostics,
                "elapsed_seconds": float(time.perf_counter() - run_started),
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device_obj))
                    if device_obj.type == "cuda"
                    else None
                ),
                "cuda_peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(device_obj))
                    if device_obj.type == "cuda"
                    else None
                ),
            }
            row["passed"] = passed
            del z, logits, loss
        except torch.cuda.OutOfMemoryError as exc:
            row["passed"] = False
            row["forward_backward"] = {
                "status": "oom",
                "error": str(exc),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)),
            }
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            row["preflight_wall_seconds"] = float(time.perf_counter() - start)
            summary["datasets"].append(row)
            write_json(DATA_ROOT / "preflight_summary.json", summary)
            print(
                f"[V6A preflight] {dataset}: nodes={data_info['num_nodes']} "
                f"edges={data_info['num_edges_physical']} "
                f"forward_backward={row['forward_backward']['status']}",
                flush=True,
            )
            del model, head, x, edge_index, synthetic_labels
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
        if not row.get("passed", False):
            summary["passed"] = False
            summary["failure"] = {"dataset": dataset, **row["forward_backward"]}
            write_json(DATA_ROOT / "preflight_summary.json", summary)
            raise SystemExit(2)
    summary["passed"] = len(summary["datasets"]) == len(DATASETS) and all(
        row["passed"] for row in summary["datasets"]
    )
    write_json(DATA_ROOT / "preflight_summary.json", summary)
    write_json(DATA_ROOT / "environment.json", summary["environment"])
    if not summary["passed"]:
        raise SystemExit(2)


def validate_run(metrics_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"missing metrics/checkpoint: {metrics_path} / {checkpoint_path}"
        )
    payload = read_json(metrics_path)
    if payload.get("protocol_version") != PROTOCOL:
        raise RuntimeError(f"wrong protocol in {metrics_path}")
    if len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"expected one run in {metrics_path}")
    run = payload["runs"][0]
    metrics = dict(run.get("metrics", {}))
    if any(str(key).lower().startswith("test") for key in metrics):
        raise RuntimeError(f"Test metric key present in {metrics_path}")
    for key in ("val_acc", "val_macro_f1"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise FloatingPointError(f"missing/non-finite {key} in {metrics_path}")
    metadata = dict(run.get("metadata", {}))
    if int(metadata.get("best_epoch", 0)) < 1:
        raise RuntimeError(f"no Validation Accuracy-selected epoch in {metrics_path}")
    hydra_cfg_path = metrics_path.parent / "hydra" / ".hydra" / "config.yaml"
    if hydra_cfg_path.is_file():
        cfg = OmegaConf.load(hydra_cfg_path)
        if bool(cfg.task.get("evaluate_test", True)):
            raise RuntimeError(f"Hydra config has evaluate_test=true: {hydra_cfg_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metrics = dict(checkpoint.get("metrics", {}))
    if any(str(key).lower().startswith("test") for key in checkpoint_metrics):
        raise RuntimeError(f"Test metric key present in checkpoint {checkpoint_path}")
    if set(checkpoint_metrics) != {"val_acc", "val_macro_f1"}:
        raise RuntimeError(f"unexpected checkpoint metric keys in {checkpoint_path}")
    if checkpoint.get("selection") != "best_val_accuracy":
        raise RuntimeError(f"checkpoint was not selected by Validation Accuracy: {checkpoint_path}")
    if not checkpoint.get("model_state") or not checkpoint.get("head_state"):
        raise RuntimeError(f"checkpoint state is incomplete: {checkpoint_path}")
    return {
        "metrics": metrics,
        "metadata": metadata,
        "checkpoint_metrics": checkpoint_metrics,
    }


def run_one(
    dataset: str,
    seed: int,
    variant: str,
    *,
    device: str,
    mode: str,
    epochs: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / mode / "runs" / dataset / f"seed_{seed}" / variant
    metrics_path = run_dir / "run_metrics.json"
    checkpoint_path = run_dir / "best.pt"
    if resume and metrics_path.is_file() and checkpoint_path.is_file():
        checked = validate_run(metrics_path, checkpoint_path)
        status = "reused"
    else:
        if not resume and (metrics_path.exists() or checkpoint_path.exists()):
            raise RuntimeError(
                f"existing run files require --resume: {run_dir.relative_to(ROOT)}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "src.main",
            f"dataset={dataset}",
            "task=nc",
            "model=mace_mag_v6a",
            f"model.variant={variant}",
            f"seed={seed}",
            "num_runs=1",
            f"device={device}",
            "task.evaluate_test=false",
            f"task.run_metrics_path={metrics_path}",
            f"task.save_ckpt_path={checkpoint_path}",
            f"hydra.run.dir={run_dir / 'hydra'}",
        ]
        if epochs is not None:
            command.extend(
                [
                    f"task.epochs={epochs}",
                    "task.early_stop_min_epoch=1",
                    "task.patience=2",
                    "task.early_stop_min_delta=0.0",
                ]
            )
        log_path = run_dir / "training.log"
        print(
            f"[V6A {mode}] {dataset} seed={seed} {variant} device={device}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
            return_code = process.wait()
        if return_code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(f"training exited {return_code}; log tail:\n{tail}")
        checked = validate_run(metrics_path, checkpoint_path)
        status = "completed"
    return {
        "status": status,
        "mode": mode,
        "dataset": dataset,
        "seed": int(seed),
        "variant": variant,
        **checked,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }


def _audit_smoke_row(row: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    from src.models.mace_mag_v6a import Model

    x_cpu, edge_cpu, data_info = load_preflight_dataset(row["dataset"])
    checkpoint_path = ROOT / row["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(ROOT / "configs" / "model" / "mace_mag_v6a.yaml")
    cfg.variant = row["variant"]
    model = Model(OmegaConf.create({"model": cfg}), data_info)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    with torch.no_grad():
        _, _, _, aux_loss, info = model(
            x_cpu.to(device), edge_cpu.to(device), return_details=True
        )
    if not torch.isfinite(aux_loss):
        raise FloatingPointError("smoke checkpoint has a non-finite auxiliary loss")
    modalities: dict[str, Any] = {}
    for modality in ("text", "visual"):
        route = info["details"]["routes"][modality]
        if route["top_indices"].size(1) != 2:
            raise RuntimeError("Top-2 route has the wrong width")
        if not torch.equal(
            route["top_indices"][:, 0] != route["top_indices"][:, 1],
            torch.ones(route["top_indices"].size(0), dtype=torch.bool, device=device),
        ):
            raise RuntimeError("Top-2 route contains duplicate experts")
        if not torch.isfinite(route["dense_probs"]).all():
            raise FloatingPointError("router probabilities are non-finite")
        modalities[modality] = {
            "top2_weights_sum_max_error": float(
                (route["top_weights"].sum(-1) - 1.0).abs().max().cpu()
            ),
            "mean_dense_probability": route["dense_probs"].mean(0).cpu().tolist(),
        }
    if row["variant"] == "a0_static":
        for modality in ("text", "visual"):
            logits = info["details"]["routes"][modality]["selection_logits"]
            if not torch.equal(logits, logits[:1].expand_as(logits)):
                raise RuntimeError("A0 route is not static")
    if row["variant"] in {"a2_cross", "a3_intra"}:
        collab = info["details"]["collaboration"]
        expected = {
            "a2_cross": {"text": "visual", "visual": "text"},
            "a3_intra": {"text": "text", "visual": "visual"},
        }[row["variant"]]
        for modality in ("text", "visual"):
            item = collab[modality]
            if item["source_modality"] != expected[modality]:
                raise RuntimeError("collaboration source mode does not match variant")
            if not torch.isfinite(item["attention"]).all():
                raise FloatingPointError("attention weights are non-finite")
            modalities[modality]["mean_null_attention"] = float(
                item["attention"][:, :, -1].mean().cpu()
            )
    return {
        "finite_checkpoint_metrics": all(
            math.isfinite(float(value)) for value in row["metrics"].values()
        ),
        "test_metrics_absent": not any(
            str(key).lower().startswith("test") for key in row["metrics"]
        ),
        "modalities": modalities,
        "parameters": row["metadata"].get("model_parameters"),
        "classifier_parameters": row["metadata"].get("classifier_parameters"),
    }


def run_smoke(device: str) -> None:
    prov = provenance("smoke")
    preflight = read_json(DATA_ROOT / "preflight_summary.json")
    if not preflight.get("passed"):
        raise RuntimeError("smoke requires a passing preflight")
    rows = []
    failures = []
    for variant in VARIANTS:
        try:
            row = run_one(
                "Movies", 42, variant, device=device, mode="smoke_final", epochs=1
            )
            row["audit"] = _audit_smoke_row(row, device)
            rows.append(row)
        except Exception as exc:
            failures.append({"variant": variant, "error": repr(exc)})
        write_json(
            DATA_ROOT / "smoke_summary.json",
            {
                "protocol": PROTOCOL,
                "task_evaluate_test": False,
                "dataset": "Movies",
                "seed": 42,
                "epochs": 1,
                "expected_runs": 4,
                "completed_runs": len(rows),
                "failures": failures,
                "rows": rows,
                "audit_passed": False,
                "provenance": prov,
            },
        )
        print(
            f"[V6A smoke] progress={len(rows)}/4 failures={len(failures)}",
            flush=True,
        )
    passed = len(rows) == 4 and not failures and all(
        item["audit"]["finite_checkpoint_metrics"]
        and item["audit"]["test_metrics_absent"]
        for item in rows
    )
    summary = {
        "protocol": PROTOCOL,
        "task_evaluate_test": False,
        "dataset": "Movies",
        "seed": 42,
        "epochs": 1,
        "device": device,
        "expected_runs": 4,
        "completed_runs": len(rows),
        "failures": failures,
        "rows": rows,
        "checks": {
            "four_variants_complete": len(rows) == 4,
            "all_validation_metrics_finite": len(rows) == 4
            and all(item["audit"]["finite_checkpoint_metrics"] for item in rows),
            "test_metrics_absent": len(rows) == 4
            and all(item["audit"]["test_metrics_absent"] for item in rows),
            "no_failures": not failures,
        },
        "audit_passed": passed,
        "provenance": prov,
    }
    write_json(DATA_ROOT / "smoke_summary.json", summary)
    print(f"[V6A smoke] completed={len(rows)}/4 audit_passed={passed}", flush=True)
    if not passed:
        raise SystemExit(1)


def run_campaign(device: str, resume: bool) -> None:
    import torch

    prov = provenance("campaign", resume=resume)
    preflight = read_json(DATA_ROOT / "preflight_summary.json")
    smoke = read_json(DATA_ROOT / "smoke_summary.json")
    tests = read_json(DATA_ROOT / "test_summary.json")
    if not preflight.get("passed") or not smoke.get("audit_passed"):
        raise RuntimeError("campaign requires passing preflight and audited 4/4 smoke")
    if not tests.get("passed") or tests.get("full_suite_exit_code") != 0:
        raise RuntimeError("campaign requires a passing full repository pytest record")
    expected_fingerprint = prov["configuration_fingerprint_sha256"]
    for label, artifact in (("preflight", preflight), ("smoke", smoke), ("full pytest", tests)):
        recorded_fingerprint = artifact.get("provenance", {}).get(
            "configuration_fingerprint_sha256"
        )
        if recorded_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"{label} artifact fingerprint {recorded_fingerprint} differs from frozen configuration {expected_fingerprint}"
            )
    rows_path = DATA_ROOT / "run_rows.json"
    manifest_path = DATA_ROOT / "campaign_manifest.json"
    failures_path = OUTPUT_ROOT / "formal" / "failures.json"
    if not resume and (rows_path.exists() or manifest_path.exists()):
        raise RuntimeError("formal artifacts exist; pass --resume to continue safely")
    rows = read_json(rows_path) if resume and rows_path.is_file() else []
    failures = read_json(failures_path) if resume and failures_path.is_file() else []
    existing_manifest = read_json(manifest_path) if resume and manifest_path.is_file() else None
    if existing_manifest:
        old_provenance = existing_manifest.get("provenance", {})
        if old_provenance.get("freeze_commit_sha") != prov.get("freeze_commit_sha"):
            raise RuntimeError("resume manifest freeze SHA differs from current HEAD")
        if (
            old_provenance.get("configuration_fingerprint_sha256")
            != prov.get("configuration_fingerprint_sha256")
        ):
            raise RuntimeError("resume manifest configuration fingerprint changed")
    completed = {
        (item["dataset"], int(item["seed"]), item["variant"])
        for item in rows
        if item.get("status") in {"completed", "reused"}
    }
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                key = (dataset, seed, variant)
                if key in completed:
                    continue
                try:
                    row = run_one(
                        dataset,
                        seed,
                        variant,
                        device=device,
                        mode="formal",
                        resume=resume,
                    )
                    row["freeze_commit_sha"] = prov["freeze_commit_sha"]
                    rows.append(row)
                    completed.add(key)
                    for failure in failures:
                        if (
                            failure.get("dataset"),
                            int(failure.get("seed", -1)),
                            failure.get("variant"),
                        ) == key:
                            failure["resolved"] = True
                            failure["resolved_at_utc"] = now()
                except Exception as exc:
                    message = repr(exc)
                    failures.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "variant": variant,
                            "error": message,
                            "oom": any(
                                term in message.lower()
                                for term in ("out of memory", "cuda oom")
                            ),
                            "nonfinite": any(
                                term in message.lower()
                                for term in ("non-finite", "nan", "inf")
                            ),
                            "resolved": False,
                            "at_utc": now(),
                        }
                    )
                    failures_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(failures_path, failures)
                    print(f"[V6A failure] {failures[-1]}", flush=True)
                    # Any failed formal cell pauses the frozen campaign. The
                    # recorded manifest can be resumed after review/repair.
                    write_json(rows_path, rows)
                    write_json(
                        manifest_path,
                        {
                            "protocol": PROTOCOL,
                            "task_evaluate_test": False,
                            "expected_runs": 36,
                            "completed_runs": len(completed),
                            "device": device,
                            "datasets": DATASETS,
                            "seeds": SEEDS,
                            "variants": VARIANTS,
                            "failures": failures,
                            "provenance": prov,
                            "updated_at_utc": now(),
                        },
                    )
                    raise SystemExit(1)
                write_json(rows_path, rows)
                manifest = {
                    "protocol": PROTOCOL,
                    "task_evaluate_test": False,
                    "expected_runs": 36,
                    "completed_runs": len(completed),
                    "device": device,
                    "datasets": DATASETS,
                    "seeds": SEEDS,
                    "variants": VARIANTS,
                    "failures": failures,
                    "provenance": prov,
                    "updated_at_utc": now(),
                    "test_metric_keys_present": any(
                        any(str(key).lower().startswith("test") for key in r["metrics"])
                        for r in rows
                    ),
                }
                write_json(manifest_path, manifest)
                write_json(failures_path, failures)
                print(
                    f"[V6A campaign] progress={len(completed)}/36",
                    flush=True,
                )
    if len(completed) != 36 or any(not f.get("resolved", False) for f in failures):
        raise SystemExit(1)
    if any(
        any(str(key).lower().startswith("test") for key in row["metrics"])
        for row in rows
    ):
        raise RuntimeError("campaign contains a Test metric key")
    print("[V6A campaign] all 36 runs completed; generating summaries", flush=True)
    subprocess.run(
        [sys.executable, "scripts/analyze_mace_mag_v6a.py", "--device", device],
        cwd=ROOT,
        check=True,
    )
    final_manifest = read_json(manifest_path)
    final_manifest["completed_runs"] = len(completed)
    final_manifest["status"] = "complete"
    final_manifest["completed_at_utc"] = now()
    final_manifest["test_metric_keys_present"] = False
    final_manifest["nvidia_smi_at_completion"] = _nvidia_smi()
    write_json(manifest_path, final_manifest)
    print(
        f"[V6A campaign] completed={len(completed)}/36; "
        f"GPU={torch.cuda.get_device_name(torch.device(device)) if 'cuda' in device else 'CPU'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "campaign"), required=True
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        run_preflight(args.device)
    elif args.mode == "smoke":
        run_smoke(args.device)
    else:
        run_campaign(args.device, args.resume)


if __name__ == "__main__":
    main()
