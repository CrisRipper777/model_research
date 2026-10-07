#!/usr/bin/env python3
"""Run the validation-only MvCGE-MAG V2 smoke or frozen campaign."""

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

BRANCH = "exp/mvcge_mag_v2_anchored_experts"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    "M0_anchor",
    "M1_direct_moe",
    "M2_anchored_moe",
    "M3_collaborative_moe",
)
OUTPUT_ROOT = ROOT / "outputs" / "mvcge_mag_v2_anchored_experts"
RESEARCH_ROOT = ROOT / "research" / "mvcge_mag_v2_anchored_experts"
DATA_ROOT = RESEARCH_ROOT / "data"
FREEZE_MESSAGE = "Freeze MvCGE-MAG V2 anchored expert screening implementation"


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
    freeze_sha = find_freeze_sha()
    if not smoke:
        if freeze_sha is None or head != freeze_sha:
            raise RuntimeError(f"Formal campaign requires HEAD at freeze commit; head={head}, freeze={freeze_sha}")
        if dirty:
            raise RuntimeError("Formal campaign requires a clean worktree at launch")
    return {
        "branch": branch,
        "parent_commit_sha": "412e69221a30fa3779e57e22fa42073697bdf4f8",
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
        raise RuntimeError(f"Missing validation-selected epoch in {metrics_path}")
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
        "model=mvcge_mag_v2",
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
    print(f"[MvCGE-MAG V2 {mode}] {dataset} seed={seed} {variant}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
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


def _write_smoke_readme() -> None:
    path = RESEARCH_ROOT / "README.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# MvCGE-MAG V2 Anchored Expert Screen\n\n"
            "Validation-only single-block MAG adaptation screen. The smoke uses Movies, seed 42, four fixed variants, and one epoch. The formal campaign covers Movies, Grocery, and ele-fashion with seeds 42–44.\n\n"
            "The one-epoch smoke showed some zero-load expert slots in MoE modality routes. Exact Top-2 routing and the active-only per-modality load-balance formula were verified; this early routing concentration is recorded for formal checkpoint diagnostics and did not trigger a design change.\n\n"
            "See [REPORT.md](REPORT.md) and `data/` for compact tracked records. Raw outputs and checkpoints remain under the ignored `outputs/` tree.\n",
            encoding="utf-8",
        )


def run_smoke(device: str) -> None:
    prov = provenance(smoke=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for variant in VARIANTS:
        try:
            row = run_one(
                "Movies", 42, variant, device=device, mode="smoke", epochs=1, resume=False
            )
            from scripts.analyze_mvcge_mag_v2_screen import audit_checkpoint

            row["audit"] = audit_checkpoint(
                row, root=ROOT, output_root=OUTPUT_ROOT, device_name=device
            )
            rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "dataset": "Movies",
                    "seed": 42,
                    "variant": variant,
                    "error": repr(exc),
                    "oom": "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower(),
                    "nonfinite": any(x in str(exc).lower() for x in ("non-finite", "nan", "inf")),
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
                "failures": failures,
                "rows": rows,
                "provenance": prov,
                "zero_load_experts": [
                    {
                        "variant": row["variant"],
                        "modality": modality,
                        "expert_indices": [
                            i for i, value in enumerate(values["selection_share"]) if value == 0.0
                        ],
                    }
                    for row in rows if row["variant"] != "M0_anchor"
                    for modality, values in row["audit"]["modalities"].items()
                    if any(value == 0.0 for value in values["selection_share"])
                ],
                "routing_review": {
                    "top2_exact_active_nodes_verified": all(
                        row["audit"]["top2_exact_active"] for row in rows if row["variant"] != "M0_anchor"
                    ),
                    "active_only_balance_formula_verified": all(
                        row["audit"]["load_balance_formula_consistent"] for row in rows
                    ),
                    "interpretation": "Some MoE expert profiles had zero Top-2 load in one-epoch smoke. Exact active-node Top-2 weights and the specified per-modality load-balance formula were verified; this is recorded as an early routing concentration observation, not changed by design or used to alter the frozen model.",
                },
                "audit_passed": len(rows) == 4
                and not failures
                and all(row["audit"]["finite"] and not row["audit"]["test_metrics_present"] for row in rows),
            },
        )
    _write_smoke_readme()
    summary = read_json(DATA_ROOT / "smoke_summary.json")
    checkpoint_count = sum(Path(ROOT / row["checkpoint_path"]).is_file() for row in rows)
    summary["checkpoint_count"] = checkpoint_count
    summary["test_metrics_absent"] = all(
        not any(key.lower().startswith("test") for key in row["metrics"])
        for row in rows
    )
    summary["audit_passed"] = bool(
        len(rows) == 4
        and checkpoint_count == 4
        and not failures
        and summary["test_metrics_absent"]
        and all(row["audit"]["finite"] for row in rows)
    )
    write_json(DATA_ROOT / "smoke_summary.json", summary)
    write_json(
        DATA_ROOT / "environment.json",
        {
            "python": sys.version,
            "platform": sys.platform,
            "torch": __import__("torch").__version__,
            "cuda_available": __import__("torch").cuda.is_available(),
            "cuda_device_count": __import__("torch").cuda.device_count(),
            "device": device,
            "protocol": "unified_full_graph_nc_v1",
        },
    )
    print(f"[MvCGE-MAG V2 smoke] {len(rows)}/4 complete; failures={len(failures)}", flush=True)
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
                    for old in failures:
                        if (old.get("dataset"), int(old.get("seed", -1)), old.get("variant")) == key:
                            old["resolved"] = True
                            old["resolved_at_utc"] = now()
                except Exception as exc:
                    failure = {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "error": repr(exc),
                        "oom": "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower(),
                        "nonfinite": any(x in str(exc).lower() for x in ("non-finite", "nan", "inf")),
                        "resolved": False,
                        "at_utc": now(),
                    }
                    failures.append(failure)
                    failures_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(failures_path, failures)
                    print(f"[MvCGE-MAG V2 failure] {failure}", flush=True)
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
                failures_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(failures_path, failures)
    unresolved = [item for item in failures if not item.get("resolved", False)]
    print(
        f"[MvCGE-MAG V2 campaign] completed={len(completed_keys)}/36; unresolved_failures={len(unresolved)}",
        flush=True,
    )
    if len(completed_keys) != 36 or unresolved:
        raise SystemExit(1)
    subprocess.run(
        [sys.executable, "scripts/analyze_mvcge_mag_v2_screen.py", "--device", device],
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
