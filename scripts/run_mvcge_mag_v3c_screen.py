#!/usr/bin/env python3
"""Run the V3C label-free preflight, smoke, and frozen NC campaign."""

from __future__ import annotations

import argparse
import csv
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

BRANCH = "exp/mvcge_mag_v3c_functional_role_screen"
PARENT_SHA = "075b12b497d619d0b86d25f264fa97402ac74778"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("F0_raw", "F1_support", "F2_role_dual_smooth", "F3_role_functional")
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v3c_functional_role_screen"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v3c_functional_role_screen"
DATA_ROOT = RESEARCH_ROOT / "data"
FREEZE_MESSAGE = "Freeze MvCGE-MAG V3C functional role action-space screen"
FORMAL_ARTIFACTS = {
    "REPORT.md",
    "data/campaign_manifest.json",
    "data/run_rows.json",
    "data/summary.csv",
    "data/paired_comparisons.csv",
    "data/role_partition_diagnostics.csv",
    "data/trajectory_diagnostics.csv",
    "data/action_space_novelty.csv",
    "data/role_profile_diagnostics.csv",
    "data/routing_diagnostics.csv",
    "data/strength_diagnostics.csv",
    "data/expert_profiles.csv",
    "data/expert_similarity.csv",
    "data/historical_f0_regression.csv",
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
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_freeze_sha() -> str | None:
    value = git("log", "--format=%H", "-1", f"--grep=^{FREEZE_MESSAGE}$")
    return value or None


def _dirty_paths() -> set[str]:
    dirty = git("status", "--porcelain", "--untracked-files=all")
    return {line[3:].split(" -> ")[-1] for line in dirty.splitlines()}


def _preflight_records() -> list[dict[str, str]]:
    path = DATA_ROOT / "preflight_role_diagnostics.csv"
    if not path.is_file():
        raise RuntimeError("Run the label-free role preflight before smoke or campaign")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise RuntimeError(f"Expected six dataset/modality role records, found {len(rows)}")
    return rows


def _validate_preflight_rows(rows: list[dict[str, str]]) -> None:
    for row in rows:
        dataset, modality = row["dataset"], row["modality"]
        support_edges = int(row["support_edge_count"])
        discrepant_edges = int(row["discrepant_edge_count"])
        error = float(row["partition_weight_max_abs_error"])
        if support_edges == 0 or discrepant_edges == 0:
            raise RuntimeError(
                f"Implementation-level empty role channel: {dataset}/{modality}; "
                f"support={support_edges}, discrepant={discrepant_edges}"
            )
        if not math.isfinite(error) or error > 1.0e-7:
            raise RuntimeError(f"Role partition identity failed for {dataset}/{modality}: {error}")
        for key, value in row.items():
            if key in {"dataset", "modality"} or value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise FloatingPointError(f"Non-finite preflight value {dataset}/{modality}/{key}")


def _require_preflight_and_smoke() -> None:
    _validate_preflight_rows(_preflight_records())
    summary_path = DATA_ROOT / "smoke_summary.json"
    if not summary_path.is_file() or not read_json(summary_path).get("audit_passed", False):
        raise RuntimeError("Formal campaign requires a passing four-variant smoke summary")


def provenance(*, mode: str, resume: bool = False) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    dirty = _dirty_paths()
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, found {branch}")
    if git("merge-base", PARENT_SHA, "HEAD") != PARENT_SHA:
        raise RuntimeError(f"Expected experiment ancestry from {PARENT_SHA}")
    freeze_sha = find_freeze_sha()
    if mode == "campaign":
        if freeze_sha is None or head != freeze_sha:
            raise RuntimeError(
                f"Formal campaign requires HEAD at freeze commit; head={head}, freeze={freeze_sha}"
            )
        if dirty and not resume:
            raise RuntimeError("Formal campaign must start with a clean worktree")
        if dirty and resume:
            allowed = {
                f"research/mvcge_mag_v3c_functional_role_screen/{path}"
                for path in FORMAL_ARTIFACTS
            }
            unexpected = dirty - allowed
            if unexpected:
                raise RuntimeError(f"Resume found non-artifact modifications: {sorted(unexpected)}")
        _require_preflight_and_smoke()
    return {
        "branch": branch,
        "parent_commit_sha": PARENT_SHA,
        "head_at_start": head,
        "freeze_commit_sha": freeze_sha,
        "worktree_clean_at_start": not dirty,
        "started_at_utc": now(),
    }


def _environment(device: str) -> dict[str, Any]:
    import torch

    return {
        "python": sys.version,
        "platform": sys.platform,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": device,
        "device_name": torch.cuda.get_device_name(torch.device(device))
        if "cuda" in device
        else "CPU",
        "protocol": "unified_full_graph_nc_v1",
    }


def run_preflight(device: str) -> None:
    from scripts.analyze_mvcge_mag_v3c_screen import (
        role_preflight,
        write_csv,
    )

    prov = provenance(mode="preflight")
    rows = role_preflight(device=device)
    _validate_preflight_rows(rows)
    write_csv(DATA_ROOT / "preflight_role_diagnostics.csv", rows)
    write_json(DATA_ROOT / "environment.json", _environment(device))
    write_json(
        DATA_ROOT / "preflight_summary.json",
        {
            "label_free": True,
            "labels_or_splits_read": False,
            "datasets": DATASETS,
            "modality_records": len(rows),
            "implementation_degeneracy": False,
            "partition_identity_max_abs_error": max(
                float(row["partition_weight_max_abs_error"]) for row in rows
            ),
            "provenance": prov,
        },
    )
    print(
        f"[MvCGE-MAG V3C preflight] {len(rows)} records; "
        f"max_partition_error={max(float(r['partition_weight_max_abs_error']) for r in rows):.3g}; "
        "implementation_degeneracy=False",
        flush=True,
    )


def validate_run(metrics_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing metrics/checkpoint: {metrics_path}, {checkpoint_path}")
    payload = read_json(metrics_path)
    if len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"Expected one run in {metrics_path}")
    run = payload["runs"][0]
    metrics = dict(run.get("metrics", {}))
    if any(str(key).lower().startswith("test") for key in metrics):
        raise RuntimeError(f"Test metrics present despite evaluate_test=false: {metrics_path}")
    for key in ("val_acc", "val_macro_f1"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise FloatingPointError(f"Missing/non-finite {key} in {metrics_path}")
    if int(run.get("metadata", {}).get("best_epoch", 0)) < 1:
        raise RuntimeError(f"No Validation Accuracy-selected checkpoint in {metrics_path}")
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
            "status": "reused", "mode": mode, "dataset": dataset,
            "seed": seed, "variant": variant, **checked,
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=mvcge_mag_v3c", f"model.variant={variant}", f"seed={seed}",
        "num_runs=1", f"device={device}", "task.evaluate_test=false",
        f"task.run_metrics_path={metrics_path}",
        f"task.save_ckpt_path={checkpoint_path}", f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    if epochs is not None:
        command.extend(
            [
                f"task.epochs={epochs}", "task.early_stop_min_epoch=1",
                "task.patience=2", "task.early_stop_min_delta=0.0",
            ]
        )
    log_path = run_dir / "training.log"
    print(f"[MvCGE-MAG V3C {mode}] {dataset} seed={seed} {variant}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"Training exited {return_code}; log tail:\n{tail}")
    checked = validate_run(metrics_path, checkpoint_path)
    return {
        "status": "completed", "mode": mode, "dataset": dataset,
        "seed": seed, "variant": variant, **checked,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }


def _audit_payload(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if not key.startswith("_")}


def run_smoke(device: str) -> None:
    import torch

    from scripts.analyze_mvcge_mag_v3c_screen import (
        audit_f0_implementation,
        audit_checkpoint,
        load_features_and_edges,
    )

    prov = provenance(mode="smoke")
    rows = _preflight_records()
    _validate_preflight_rows(rows)
    cached_data = load_features_and_edges("Movies", 42)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for variant in VARIANTS:
        try:
            row = run_one(
                "Movies", 42, variant, device=device, mode="smoke", epochs=1, resume=False
            )
            audit = audit_checkpoint(
                row, device_name=device, cached_data=cached_data
            )
            regression = None
            if variant == "F0_raw":
                regression = audit_f0_implementation(
                    row, device_name=device, cached_data=cached_data
                )
                if not regression["forward_allclose_1e7"]:
                    raise RuntimeError("F0 selected checkpoint failed V3A C0 compatible forward audit")
            if audit["test_metrics_present"] or audit["task_evaluate_test"] is not False:
                raise RuntimeError("Smoke found Test metrics or task.evaluate_test enabled")
            if not all(audit[key] for key in ("finite", "role_partition_valid", "static_routing_verified", "top2_verified")):
                raise RuntimeError("Smoke selected-checkpoint audit failed")
            row["audit"] = _audit_payload(audit)
            row["f0_compatible_regression"] = regression
            results.append(row)
        except Exception as exc:
            message = repr(exc)
            lower = message.lower()
            failures.append(
                {
                    "dataset": "Movies", "seed": 42, "variant": variant,
                    "error": message,
                    "oom": any(token in lower for token in ("out of memory", "cuda oom")),
                    "nonfinite": any(token in lower for token in ("non-finite", "nan", "inf")),
                }
            )
        write_json(
            DATA_ROOT / "smoke_summary.json",
            {
                "protocol": "unified_full_graph_nc_v1",
                "task_evaluate_test": False,
                "dataset": "Movies", "seed": 42, "epochs": 1, "device": device,
                "expected_runs": 4, "completed_runs": len(results),
                "checkpoint_count": sum(Path(ROOT / r["checkpoint_path"]).is_file() for r in results),
                "failures": failures, "rows": results, "provenance": prov,
                "audit_passed": False,
            },
        )
        print(f"[MvCGE-MAG V3C smoke] progress={len(results)}/4 failures={len(failures)}", flush=True)

    checks = {
        "four_runs_complete": len(results) == 4,
        "all_checkpoints_finite": len(results) == 4 and all(r["audit"]["finite"] for r in results),
        "test_metrics_absent": len(results) == 4 and all(not r["audit"]["test_metrics_present"] for r in results),
        "evaluate_test_false": True,
        "role_partition_and_trajectories_finite": len(results) == 4 and all(r["audit"]["role_partition_valid"] and r["audit"]["role_trajectories_finite"] for r in results),
        "static_top2_verified": len(results) == 4 and all(r["audit"]["static_routing_verified"] and r["audit"]["top2_verified"] for r in results),
        "f0_v3a_forward_compatible": any(r["variant"] == "F0_raw" and r["f0_compatible_regression"]["forward_allclose_1e7"] for r in results),
        "dual_role_mix_finite": all(all(math.isfinite(float(v)) for v in r["audit"]["role_mix_values"]) for r in results if r["variant"] in {"F2_role_dual_smooth", "F3_role_functional"}),
        "no_oom_or_nonfinite_failures": not any(f["oom"] or f["nonfinite"] for f in failures),
    }
    summary = {
        "protocol": "unified_full_graph_nc_v1",
        "task_evaluate_test": False,
        "dataset": "Movies", "seed": 42, "epochs": 1, "device": device,
        "expected_runs": 4, "completed_runs": len(results),
        "checkpoint_count": sum(Path(ROOT / r["checkpoint_path"]).is_file() for r in results),
        "failures": failures, "rows": results, "checks": checks,
        "audit_passed": len(results) == 4 and not failures and all(checks.values()),
        "provenance": prov,
    }
    write_json(DATA_ROOT / "smoke_summary.json", summary)
    write_json(DATA_ROOT / "environment.json", _environment(device))
    print(
        f"[MvCGE-MAG V3C smoke] {len(results)}/4 complete; "
        f"failures={len(failures)}; audit_passed={summary['audit_passed']}",
        flush=True,
    )
    del cached_data
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.empty_cache()
    if not summary["audit_passed"]:
        raise SystemExit(1)


def run_campaign(device: str, resume: bool) -> None:
    prov = provenance(mode="campaign", resume=resume)
    rows_path = DATA_ROOT / "run_rows.json"
    manifest_path = DATA_ROOT / "campaign_manifest.json"
    rows = read_json(rows_path) if resume and rows_path.is_file() else []
    failures_path = OUTPUT_ROOT / "formal" / "failures.json"
    failures = read_json(failures_path) if resume and failures_path.is_file() else []
    completed = {
        (item["dataset"], int(item["seed"]), item["variant"])
        for item in rows if item.get("status") in {"completed", "reused"}
    }
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                key = (dataset, seed, variant)
                if key in completed:
                    continue
                try:
                    row = run_one(
                        dataset, seed, variant, device=device, mode="formal",
                        epochs=None, resume=resume,
                    )
                    rows.append(row)
                    completed.add(key)
                    for previous in failures:
                        if (previous.get("dataset"), int(previous.get("seed", -1)), previous.get("variant")) == key:
                            previous["resolved"] = True
                            previous["resolved_at_utc"] = now()
                except Exception as exc:
                    message = repr(exc)
                    lower = message.lower()
                    failures.append(
                        {
                            "dataset": dataset, "seed": seed, "variant": variant,
                            "error": message,
                            "oom": any(token in lower for token in ("out of memory", "cuda oom")),
                            "nonfinite": any(token in lower for token in ("non-finite", "nan", "inf")),
                            "resolved": False, "at_utc": now(),
                        }
                    )
                    print(f"[MvCGE-MAG V3C failure] {failures[-1]}", flush=True)
                payload = {
                    "protocol": "unified_full_graph_nc_v1",
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
                }
                write_json(rows_path, rows)
                write_json(manifest_path, payload)
                write_json(failures_path, failures)
                print(f"[MvCGE-MAG V3C campaign] progress={len(completed)}/36", flush=True)
    unresolved = [entry for entry in failures if not entry.get("resolved", False)]
    print(
        f"[MvCGE-MAG V3C campaign] completed={len(completed)}/36; "
        f"unresolved_failures={len(unresolved)}",
        flush=True,
    )
    if len(completed) != 36 or unresolved:
        raise SystemExit(1)
    subprocess.run(
        [sys.executable, "scripts/analyze_mvcge_mag_v3c_screen.py", "--device", device],
        cwd=ROOT, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "smoke", "campaign"), required=True)
    parser.add_argument("--device", default="cuda:0")
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
