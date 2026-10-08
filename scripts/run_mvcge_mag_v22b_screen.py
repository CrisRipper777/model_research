#!/usr/bin/env python3
"""Run the validation-only V2.2b smoke and frozen 36-run campaign."""

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

BRANCH = "exp/mvcge_mag_v22b_strength_decoupling"
PARENT_SHA = "d1d7bbd4b795abd014cec6fdc194dc056f6d058c"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "D0_static",
    "D1_free_node",
    "D2_struct_free",
    "D3_struct_residual",
)
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v22b_strength_decoupling"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v22b_strength_decoupling"
DATA_ROOT = RESEARCH_ROOT / "data"
FREEZE_MESSAGE = "Freeze MvCGE-MAG V2.2b strength-decoupled routing repair implementation"


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


def provenance(*, smoke: bool, resume: bool = False) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, found {branch}")
    if git("merge-base", PARENT_SHA, "HEAD") != PARENT_SHA:
        raise RuntimeError(f"Expected experiment ancestry from {PARENT_SHA}")
    freeze_sha = find_freeze_sha()
    if not smoke:
        if freeze_sha is None or head != freeze_sha:
            raise RuntimeError(
                f"Formal campaign requires HEAD at freeze commit; head={head}, freeze={freeze_sha}"
            )
        if dirty and not resume:
            raise RuntimeError("Formal campaign requires a clean worktree at launch")
        if dirty and resume:
            artifact_root = "research/mvcge_mag_v22b_strength_decoupling/"
            allowed = {artifact_root + "REPORT.md"}
            allowed.update(
                artifact_root + "data/" + name
                for name in (
                    "run_rows.json", "campaign_manifest.json", "summary.csv",
                    "paired_comparisons.csv", "routing_diagnostics.csv",
                    "pair_diagnostics.csv", "evidence_diagnostics.csv",
                    "strength_diagnostics.csv", "expert_profiles.csv",
                    "expert_similarity.csv", "repair_comparison.csv",
                )
            )
            dirty_paths = {line[3:].split(" -> ")[-1] for line in dirty.splitlines()}
            if not dirty_paths <= allowed:
                raise RuntimeError(f"Resume found unexpected modified paths: {sorted(dirty_paths - allowed)}")
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
        raise FileNotFoundError(f"Missing run metrics or checkpoint: {metrics_path}, {checkpoint_path}")
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
        "model=mvcge_mag_v22b",
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
    print(f"[MvCGE-MAG V2.2b {mode}] {dataset} seed={seed} {variant}", flush=True)
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


def _json_safe_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if not key.startswith("_")}


def run_smoke(device: str) -> None:
    import torch

    from scripts.analyze_mvcge_mag_v22b_screen import audit_checkpoint

    prov = provenance(smoke=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for variant in VARIANTS:
        try:
            row = run_one(
                "Movies", 42, variant, device=device, mode="smoke", epochs=1, resume=False
            )
            audit = audit_checkpoint(row, root=ROOT, output_root=OUTPUT_ROOT, device_name=device)
            if audit["test_metrics_present"] or not audit["finite"]:
                raise RuntimeError("Smoke audit found Test metrics or non-finite tensors")
            if not audit["legacy_strength_head_frozen"] or not audit["direct_strength_finite"]:
                raise RuntimeError("Smoke audit found an invalid strength parameter state")
            if variant == "D0_static" and not audit["static_routing_verified"]:
                raise RuntimeError("D0 did not produce modality-static routing")
            if variant == "D2_struct_free" and not audit["free_structure_logits_verified"]:
                raise RuntimeError("D2 selection logits differ from structure-free logits")
            if variant == "D3_struct_residual" and not audit["residual_logits_verified"]:
                raise RuntimeError("D3 eta interpolation audit failed")
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
        write_json(
            DATA_ROOT / "smoke_summary.json",
            {
                "protocol": "unified_full_graph_nc_v1",
                "task_evaluate_test": False,
                "dataset": "Movies",
                "seed": 42,
                "epochs": 1,
                "device": device,
                "expected_runs": 4,
                "completed_runs": len(rows),
                "checkpoint_count": sum(Path(ROOT / r["checkpoint_path"]).is_file() for r in rows),
                "failures": failures,
                "resolved_retry_notes": [],
                "rows": rows,
                "provenance": prov,
            },
        )
        print(f"[MvCGE-MAG V2.2b smoke] progress={len(rows)}/4 failures={len(failures)}", flush=True)

    checks = {
        "finite_selected_checkpoints": len(rows) == 4 and all(r["audit"]["finite"] for r in rows),
        "test_metrics_absent": len(rows) == 4
        and all(not r["audit"]["test_metrics_present"] for r in rows),
        "router_evidence_finite": len(rows) == 4
        and all(r["audit"]["router_evidence_finite"] for r in rows),
        "strength_is_static_and_decoupled": len(rows) == 4
        and all(r["audit"]["strength_static"] and r["audit"]["legacy_strength_head_frozen"] for r in rows),
        "D0_static_routing": any(r["variant"] == "D0_static" and r["audit"]["static_routing_verified"] for r in rows),
        "D2_free_structure_logits": any(r["variant"] == "D2_struct_free" and r["audit"]["free_structure_logits_verified"] for r in rows),
        "D3_eta_residual": any(r["variant"] == "D3_struct_residual" and r["audit"]["residual_logits_verified"] for r in rows),
    }
    summary = read_json(DATA_ROOT / "smoke_summary.json")
    summary["diagnostics"] = {
        r["variant"]: {
            "strength_text": r["audit"]["modalities"]["text"]["strength"]["mean"],
            "strength_visual": r["audit"]["modalities"]["visual"]["strength"]["mean"],
            "pair_entropy_text": r["audit"]["pair_diagnostics"]["text"]["normalized_pair_entropy"],
            "pair_entropy_visual": r["audit"]["pair_diagnostics"]["visual"]["normalized_pair_entropy"],
            "route_to_mean_js_text": r["audit"]["pair_diagnostics"]["text"]["route_to_modality_mean_js"]["mean"],
            "route_to_mean_js_visual": r["audit"]["pair_diagnostics"]["visual"]["route_to_modality_mean_js"]["mean"],
            "route_to_static_js_text": r["audit"]["modalities"]["text"]["route_to_static_js"]["mean"],
            "route_to_static_js_visual": r["audit"]["modalities"]["visual"]["route_to_static_js"]["mean"],
            "evidence_text": r["audit"]["evidence_diagnostics"]["text"],
            "evidence_visual": r["audit"]["evidence_diagnostics"]["visual"],
            "eta_text": r["audit"]["eta"]["text"],
            "eta_visual": r["audit"]["eta"]["visual"],
        }
        for r in rows
    }
    summary["checks"] = checks
    summary["audit_passed"] = len(rows) == 4 and not failures and all(checks.values())
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
            "device_name": torch.cuda.get_device_name(torch.device(device))
            if "cuda" in device
            else "CPU",
            "protocol": "unified_full_graph_nc_v1",
        },
    )
    print(
        f"[MvCGE-MAG V2.2b smoke] {len(rows)}/4 complete; failures={len(failures)}; "
        f"audit_passed={summary['audit_passed']}",
        flush=True,
    )
    if not summary["audit_passed"]:
        raise SystemExit(1)


def run_campaign(device: str, resume: bool) -> None:
    prov = provenance(smoke=False, resume=resume)
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
                    failures.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "variant": variant,
                            "error": message,
                            "oom": any(term in message.lower() for term in ("out of memory", "cuda oom")),
                            "nonfinite": any(term in message.lower() for term in ("non-finite", "nan", "inf")),
                            "resolved": False,
                            "at_utc": now(),
                        }
                    )
                    write_json(failures_path, failures)
                    print(f"[MvCGE-MAG V2.2b failure] {failures[-1]}", flush=True)
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
                print(f"[MvCGE-MAG V2.2b campaign] progress={len(completed_keys)}/36", flush=True)
    unresolved = [item for item in failures if not item.get("resolved", False)]
    print(
        f"[MvCGE-MAG V2.2b campaign] completed={len(completed_keys)}/36; "
        f"unresolved_failures={len(unresolved)}",
        flush=True,
    )
    if len(completed_keys) != 36 or unresolved:
        raise SystemExit(1)
    subprocess.run(
        [sys.executable, "scripts/analyze_mvcge_mag_v22b_screen.py", "--device", device],
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
