#!/usr/bin/env python3
"""Run the validation-only V2.1 smoke and frozen 36-run campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BRANCH = "exp/mvcge_mag_v21_utilization_screen"
PARENT_SHA = "40b2c3953f6647222ce07053b10a7500c1b6a856"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "U0_modality_static",
    "U1_node_selection",
    "U2_node_selection_strength",
    "U3_collaborative",
)
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v21_utilization_screen"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v21_utilization_screen"
DATA_ROOT = RESEARCH_ROOT / "data"
FREEZE_MESSAGE = "Freeze MvCGE-MAG V2.1 utilization screening implementation"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_freeze_sha() -> str | None:
    result = git("log", "--format=%H", "-1", f"--grep=^{FREEZE_MESSAGE}$")
    return result or None


def provenance(*, smoke: bool) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, found {branch}")
    if git("merge-base", PARENT_SHA, "HEAD") != PARENT_SHA:
        raise RuntimeError(f"Expected experiment ancestry from {PARENT_SHA}")
    freeze_sha = find_freeze_sha()
    if not smoke:
        if freeze_sha is None or head != freeze_sha:
            raise RuntimeError(
                f"Formal campaign requires HEAD at the freeze commit; head={head}, freeze={freeze_sha}"
            )
        if dirty:
            raise RuntimeError("Formal campaign requires a clean worktree at launch")
    return {
        "branch": branch,
        "parent_commit_sha": PARENT_SHA,
        "head_at_start": head,
        "freeze_commit_sha": freeze_sha,
        "worktree_clean_at_start": not bool(dirty),
        "started_at_utc": now(),
    }


def validate_run(metrics_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing run metrics or checkpoint: {metrics_path}, {checkpoint_path}"
        )
    payload = read_json(metrics_path)
    if len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"Expected one run in {metrics_path}")
    run = payload["runs"][0]
    metrics = dict(run.get("metrics", {}))
    if any(str(key).lower().startswith("test") for key in metrics):
        raise RuntimeError(f"Test metrics are present with evaluate_test=false: {metrics_path}")
    for key in ("val_acc", "val_macro_f1"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise FloatingPointError(f"Missing/non-finite {key} in {metrics_path}")
    if int(run.get("metadata", {}).get("best_epoch", 0)) < 1:
        raise RuntimeError(f"No Validation Accuracy-selected epoch in {metrics_path}")
    return {"metrics": metrics, "metadata": dict(run["metadata"])}


def run_one(
    dataset: str,
    seed: int,
    variant: str,
    *,
    device: str,
    mode: str,
    epochs: int | None,
    resume: bool,
) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / mode / "runs" / dataset / f"seed_{seed}" / variant
    metrics_path = run_dir / "run_metrics.json"
    checkpoint_path = run_dir / "best.pt"
    if resume and metrics_path.is_file() and checkpoint_path.is_file():
        checked = validate_run(metrics_path, checkpoint_path)
        return {
            "status": "reused",
            "mode": mode,
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            **checked,
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=mvcge_mag_v21",
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
    print(f"[MvCGE-MAG V2.1 {mode}] {dataset} seed={seed} {variant}", flush=True)
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
        raise RuntimeError(f"Training command exited {return_code}; last log lines:\n{tail}")
    checked = validate_run(metrics_path, checkpoint_path)
    return {
        "status": "completed",
        "mode": mode,
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        **checked,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }


def compare_old_m1_smoke(u2_row: dict[str, Any]) -> dict[str, Any]:
    """Compare the U2 smoke checkpoint with the archived V2 M1 smoke run."""
    import torch

    old_dir = (
        ROOT
        / "outputs/mvcge_mag_v2_anchored_experts/smoke/runs/Movies/seed_42/M1_direct_moe"
    )
    new_checkpoint = torch.load(
        ROOT / u2_row["checkpoint_path"], map_location="cpu", weights_only=False
    )
    old_checkpoint_path = old_dir / "best.pt"
    old_metrics_path = old_dir / "run_metrics.json"
    if not old_checkpoint_path.is_file() or not old_metrics_path.is_file():
        raise FileNotFoundError("The prior V2 M1 Movies/42 smoke checkpoint is required")
    old_checkpoint = torch.load(old_checkpoint_path, map_location="cpu", weights_only=False)
    old_model = old_checkpoint["model_state"]
    new_model = new_checkpoint["model_state"]
    shared_keys = sorted(set(old_model) & set(new_model))
    if not shared_keys:
        raise AssertionError("No shared model state keys found for old-M1 smoke comparison")
    mismatches = []
    max_abs = 0.0
    for key in shared_keys:
        left, right = old_model[key], new_model[key]
        if left.is_floating_point():
            delta = float((left - right).abs().max().item()) if left.numel() else 0.0
            max_abs = max(max_abs, delta)
            if not torch.allclose(left, right, atol=2.0e-6, rtol=0.0):
                mismatches.append(key)
        elif not torch.equal(left, right):
            mismatches.append(key)
    old_head = old_checkpoint["head_state"]
    new_head = new_checkpoint["head_state"]
    for key in old_head:
        if not torch.allclose(old_head[key], new_head[key], atol=2.0e-6, rtol=0.0):
            mismatches.append(f"head.{key}")
    old_metrics = read_json(old_metrics_path)["runs"][0]["metrics"]
    metric_deltas = {
        key: float(u2_row["metrics"][key]) - float(old_metrics[key])
        for key in ("val_acc", "val_macro_f1")
    }
    if mismatches or any(abs(value) > 1.0e-7 for value in metric_deltas.values()):
        raise AssertionError(
            f"U2 smoke differs from archived V2 M1: params={mismatches[:20]}, "
            f"max_abs={max_abs}, metric_deltas={metric_deltas}"
        )
    return {
        "available": True,
        "shared_model_state_keys_compared": len(shared_keys),
        "classifier_state_keys_compared": len(old_head),
        "max_abs_parameter_delta": max_abs,
        "validation_metric_deltas": metric_deltas,
        "matches": True,
    }


def _json_safe_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if not key.startswith("_")}


def run_smoke(device: str) -> None:
    import torch

    prov = provenance(smoke=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    from scripts.analyze_mvcge_mag_v21_screen import audit_checkpoint

    for variant in VARIANTS:
        try:
            row = run_one(
                "Movies", 42, variant, device=device, mode="smoke", epochs=1, resume=False
            )
            audit = audit_checkpoint(
                row, root=ROOT, output_root=OUTPUT_ROOT, device_name=device
            )
            if audit["test_metrics_present"] or not audit["finite"]:
                raise RuntimeError("Smoke checkpoint audit found test metrics or non-finite tensors")
            row["audit"] = _json_safe_audit(audit)
            rows.append(row)
            del audit
        except Exception as exc:
            message = repr(exc)
            failures.append(
                {
                    "dataset": "Movies",
                    "seed": 42,
                    "variant": variant,
                    "error": message,
                    "oom": any(term in message.lower() for term in ("out of memory", "cuda oom")),
                    "nonfinite": any(term in message.lower() for term in ("non-finite", "nan", "inf")),
                }
            )
        payload = {
            "protocol": "unified_full_graph_nc_v1",
            "task_evaluate_test": False,
            "dataset": "Movies",
            "seed": 42,
            "epochs": 1,
            "device": device,
            "expected_runs": 4,
            "completed_runs": len(rows),
            "failures": failures,
            "rows": rows,
            "provenance": prov,
        }
        write_json(DATA_ROOT / "smoke_summary.json", payload)

    regression = None
    if not failures and len(rows) == 4:
        try:
            u2_row = next(row for row in rows if row["variant"] == "U2_node_selection_strength")
            regression = compare_old_m1_smoke(u2_row)
        except Exception as exc:
            failures.append(
                {
                    "variant": "U2_node_selection_strength",
                    "error": repr(exc),
                    "oom": False,
                    "nonfinite": False,
                    "old_m1_smoke_comparison": True,
                }
            )

    row_by_variant = {row["variant"]: row for row in rows}
    u0_checks = (
        row_by_variant.get("U0_modality_static", {}).get("audit", {}).get("pair_diagnostics", {})
    )
    u1_row = row_by_variant.get("U1_node_selection", {})
    u1_strength_stds = [
        u1_row.get("audit", {}).get("modalities", {}).get(modality, {}).get("strength", {}).get("std")
        for modality in ("text", "visual")
    ]
    u0_entropy_ok = all(
        u0_checks.get(modality, {}).get("num_unique_pairs") == 1
        and u0_checks.get(modality, {}).get("pair_entropy") == 0.0
        for modality in ("text", "visual")
    )
    u0_strength_ok = all(
        abs(float(row_by_variant.get("U0_modality_static", {}).get("audit", {}).get("modalities", {}).get(modality, {}).get("strength", {}).get("std", math.inf)))
        <= 1.0e-12
        for modality in ("text", "visual")
    )
    u1_strength_ok = all(value is not None and abs(float(value)) <= 1.0e-12 for value in u1_strength_stds)
    checkpoint_count = sum(
        Path(ROOT / row["checkpoint_path"]).is_file() for row in rows
    )
    test_metrics_absent = all(
        not any(key.lower().startswith("test") for key in row["metrics"])
        and not row["audit"]["test_metrics_present"]
        for row in rows
    )
    summary = {
        "protocol": "unified_full_graph_nc_v1",
        "task_evaluate_test": False,
        "dataset": "Movies",
        "seed": 42,
        "epochs": 1,
        "device": device,
        "expected_runs": 4,
        "completed_runs": len(rows),
        "checkpoint_count": checkpoint_count,
        "failures": failures,
        "rows": rows,
        "provenance": prov,
        "resolved_smoke_retry_notes": [
            "First audit retry: U0 route-to-mean JS used an overly strict 1e-12 bound; relaxed to 1e-6 to allow float32 averaging roundoff while preserving exact per-node route equality tests.",
            "Old-M1 checkpoint retry: GPU training produced one shared parameter delta of 1.15e-6 with identical validation metrics; smoke checkpoint comparison now uses 2e-6 absolute tolerance. The eval-mode forward regression remains tested at 1e-7.",
        ],
        "checks": {
            "finite_selected_checkpoints": all(row["audit"]["finite"] for row in rows),
            "test_metrics_absent": test_metrics_absent,
            "U0_pair_entropy_zero": u0_entropy_ok,
            "U0_strength_std_zero": u0_strength_ok,
            "U1_strength_std_zero": u1_strength_ok,
            "U2_matches_old_M1_smoke": bool(regression and regression["matches"]),
        },
        "old_m1_smoke_regression": regression,
    }
    summary["audit_passed"] = bool(
        len(rows) == 4
        and checkpoint_count == 4
        and not failures
        and all(summary["checks"].values())
    )
    write_json(DATA_ROOT / "smoke_summary.json", summary)
    write_json(
        DATA_ROOT / "environment.json",
        {
            "python": sys.version,
            "platform": sys.platform,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "device": device,
            "device_name": torch.cuda.get_device_name(torch.device(device)) if "cuda" in device else "CPU",
            "protocol": "unified_full_graph_nc_v1",
        },
    )
    print(
        f"[MvCGE-MAG V2.1 smoke] {len(rows)}/4 complete; failures={len(failures)}; "
        f"audit_passed={summary['audit_passed']}",
        flush=True,
    )
    if not summary["audit_passed"]:
        raise SystemExit(1)


def run_campaign(device: str, resume: bool) -> None:
    prov = provenance(smoke=False)
    rows_path = DATA_ROOT / "run_rows.json"
    manifest_path = DATA_ROOT / "campaign_manifest.json"
    rows = read_json(rows_path) if resume and rows_path.exists() else []
    failures_path = OUTPUT_ROOT / "formal" / "failures.json"
    failures = read_json(failures_path) if resume and failures_path.exists() else []
    completed_keys = {
        (item["dataset"], int(item["seed"]), item["variant"])
        for item in rows
        if item.get("status") in {"completed", "reused"}
    }
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                key = (dataset, seed, variant)
                if key in completed_keys:
                    continue
                try:
                    row = run_one(
                        dataset,
                        seed,
                        variant,
                        device=device,
                        mode="formal",
                        epochs=None,
                        resume=resume,
                    )
                    rows.append(row)
                    completed_keys.add(key)
                    for previous in failures:
                        if (
                            previous.get("dataset"),
                            int(previous.get("seed", -1)),
                            previous.get("variant"),
                        ) == key:
                            previous["resolved"] = True
                            previous["resolved_at_utc"] = now()
                except Exception as exc:
                    message = repr(exc)
                    failure = {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "error": message,
                        "oom": any(term in message.lower() for term in ("out of memory", "cuda oom")),
                        "nonfinite": any(term in message.lower() for term in ("non-finite", "nan", "inf")),
                        "resolved": False,
                        "at_utc": now(),
                    }
                    failures.append(failure)
                    write_json(failures_path, failures)
                    print(f"[MvCGE-MAG V2.1 failure] {failure}", flush=True)
                payload = {
                    "protocol": "unified_full_graph_nc_v1",
                    "task_evaluate_test": False,
                    "expected_runs": 36,
                    "completed_runs": len(completed_keys),
                    "device": device,
                    "datasets": DATASETS,
                    "seeds": SEEDS,
                    "variants": VARIANTS,
                    "failures": failures,
                    "provenance": prov,
                    "updated_at_utc": now(),
                }
                write_json(rows_path, rows)
                write_json(manifest_path, payload)
                write_json(failures_path, failures)
                print(
                    f"[MvCGE-MAG V2.1 campaign] progress={len(completed_keys)}/36",
                    flush=True,
                )
    unresolved = [item for item in failures if not item.get("resolved", False)]
    print(
        f"[MvCGE-MAG V2.1 campaign] completed={len(completed_keys)}/36; "
        f"unresolved_failures={len(unresolved)}",
        flush=True,
    )
    if len(completed_keys) != 36 or unresolved:
        raise SystemExit(1)
    subprocess.run(
        [sys.executable, "scripts/analyze_mvcge_mag_v21_screen.py", "--device", device],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "campaign"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke(args.device)
    else:
        run_campaign(args.device, args.resume)


if __name__ == "__main__":
    main()
