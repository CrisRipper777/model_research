#!/usr/bin/env python3
"""Run the validation-only CSE-MAG V1 four-model screening campaign.

This launcher deliberately uses the repository's unified NC task runner instead
of defining a private training loop. Development uses Train + Validation only;
Test evaluation is disabled for the full screening campaign.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRANCH = "exp/cse_mag_v1_local_global_screen"
BASE_SHA = "579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    ("B0_independent", "independent"),
    ("B1_shared_static", "shared_static"),
    ("B2_mvcge_style", "mvcge_style"),
    ("B3_v1", "v1"),
)
DEFAULT_OUT = ROOT / "outputs" / "cse_mag_v1_local_global_screen"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def check_provenance(allow_dirty: bool) -> dict[str, object]:
    branch = git_value("branch", "--show-current")
    head = git_value("rev-parse", "HEAD")
    dirty = git_value("status", "--porcelain")
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH!r}, found {branch!r}")
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Working tree is dirty. Commit/stash changes or rerun with --allow-dirty."
        )
    return {
        "branch": branch,
        "head": head,
        "base_sha": BASE_SHA,
        "worktree_clean": not bool(dirty),
        "started_at_utc": now(),
    }


def run_one(
    dataset: str,
    seed: int,
    label: str,
    variant: str,
    *,
    device: str,
    out_root: Path,
    epochs: int | None,
    resume: bool,
) -> dict[str, object]:
    run_dir = out_root / "runs" / dataset / f"seed_{seed}" / label
    metrics_path = run_dir / "run_metrics.json"
    ckpt_path = run_dir / "best.pt"
    hydra_dir = run_dir / "hydra"

    if resume and metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("runs"):
            metrics = payload["runs"][0]["metrics"]
            metadata = payload["runs"][0].get("metadata", {})
            return {
                "status": "reused",
                "dataset": dataset,
                "seed": seed,
                "label": label,
                "variant": variant,
                "metrics": metrics,
                "metadata": metadata,
                "metrics_path": str(metrics_path.relative_to(ROOT)),
            }

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=cse_mag_v1",
        f"model.variant={variant}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        "task.evaluate_test=false",
        f"task.run_metrics_path={metrics_path}",
        f"task.save_ckpt_path={ckpt_path}",
        f"hydra.run.dir={hydra_dir}",
    ]
    if epochs is not None:
        command.append(f"task.epochs={int(epochs)}")
        if int(epochs) <= 2:
            command.extend(
                [
                    "task.early_stop_min_epoch=1",
                    "task.patience=2",
                    "task.early_stop_min_delta=0.0",
                ]
            )

    print(
        f"[CSE-V1] {dataset} seed={seed} {label}/{variant}",
        flush=True,
    )
    subprocess.run(command, cwd=ROOT, check=True)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"Expected one run in {metrics_path}")
    run_payload = payload["runs"][0]
    metrics = dict(run_payload["metrics"])
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError("Screening run unexpectedly contains Test metrics")
    return {
        "status": "completed",
        "dataset": dataset,
        "seed": seed,
        "label": label,
        "variant": variant,
        "metrics": metrics,
        "metadata": run_payload.get("metadata", {}),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_cell: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        for label, variant in VARIANTS:
            cell = [
                row
                for row in rows
                if row["dataset"] == dataset and row["label"] == label
            ]
            if not cell:
                continue
            acc = [float(row["metrics"]["val_acc"]) for row in cell]
            f1 = [float(row["metrics"]["val_macro_f1"]) for row in cell]
            params = [
                int(row["metadata"].get("model_parameters", 0))
                + int(row["metadata"].get("classifier_parameters", 0))
                for row in cell
            ]
            by_cell[f"{dataset}/{label}"] = {
                "dataset": dataset,
                "label": label,
                "variant": variant,
                "n": len(cell),
                "val_acc_mean": _mean(acc),
                "val_acc_std": _pstdev(acc),
                "val_macro_f1_mean": _mean(f1),
                "val_macro_f1_std": _pstdev(f1),
                "total_trainable_parameters_mean": _mean([float(x) for x in params]),
            }

    paired: list[dict[str, object]] = []
    target_rows = {
        (str(row["dataset"]), int(row["seed"])): row
        for row in rows
        if row["label"] == "B3_v1"
    }
    for baseline_label, _ in VARIANTS[:-1]:
        deltas_acc: list[float] = []
        deltas_f1: list[float] = []
        dataset_means: dict[str, dict[str, float]] = {}
        for dataset in DATASETS:
            dataset_acc: list[float] = []
            dataset_f1: list[float] = []
            for seed in SEEDS:
                target = target_rows.get((dataset, seed))
                baseline = next(
                    (
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and int(row["seed"]) == seed
                        and row["label"] == baseline_label
                    ),
                    None,
                )
                if target is None or baseline is None:
                    continue
                da = float(target["metrics"]["val_acc"]) - float(
                    baseline["metrics"]["val_acc"]
                )
                df = float(target["metrics"]["val_macro_f1"]) - float(
                    baseline["metrics"]["val_macro_f1"]
                )
                dataset_acc.append(da)
                dataset_f1.append(df)
                deltas_acc.append(da)
                deltas_f1.append(df)
            if dataset_acc:
                dataset_means[dataset] = {
                    "delta_val_acc": _mean(dataset_acc),
                    "delta_val_macro_f1": _mean(dataset_f1),
                }
        paired.append(
            {
                "comparison": f"B3_v1-minus-{baseline_label}",
                "paired_runs": len(deltas_acc),
                "delta_val_acc_mean": _mean(deltas_acc),
                "delta_val_macro_f1_mean": _mean(deltas_f1),
                "positive_acc_pairs": sum(x > 0 for x in deltas_acc),
                "positive_f1_pairs": sum(x > 0 for x in deltas_f1),
                "dataset_means": dataset_means,
            }
        )

    return {
        "cells": list(by_cell.values()),
        "paired_v1_comparisons": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--campaign", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    provenance = check_provenance(args.allow_dirty)
    out_root = Path(args.out_dir).expanduser().resolve()
    datasets = ("Movies",) if args.smoke else DATASETS
    seeds = (42,) if args.smoke else SEEDS
    epochs = 1 if args.smoke and args.epochs is None else args.epochs

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            for label, variant in VARIANTS:
                try:
                    rows.append(
                        run_one(
                            dataset,
                            seed,
                            label,
                            variant,
                            device=args.device,
                            out_root=out_root,
                            epochs=epochs,
                            resume=args.resume,
                        )
                    )
                except Exception as exc:
                    failure = {
                        "dataset": dataset,
                        "seed": seed,
                        "label": label,
                        "variant": variant,
                        "error": repr(exc),
                    }
                    failures.append(failure)
                    write_json(
                        out_root
                        / "failures"
                        / dataset
                        / f"seed_{seed}_{label}.json",
                        failure,
                    )
                    print(f"[CSE-V1 failed] {failure}", flush=True)

    manifest = {
        **provenance,
        "finished_at_utc": now(),
        "mode": "smoke" if args.smoke else "campaign",
        "datasets": list(datasets),
        "seeds": list(seeds),
        "variants": [
            {"label": label, "variant": variant} for label, variant in VARIANTS
        ],
        "expected_runs": len(datasets) * len(seeds) * len(VARIANTS),
        "completed_or_reused_runs": len(rows),
        "failures": failures,
        "device": args.device,
        "epochs_override": epochs,
        "task_protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False,
        "selection": "best_validation_accuracy",
        "summary": summarize(rows),
    }
    write_json(out_root / "campaign_manifest.json", manifest)
    write_json(out_root / "run_rows.json", rows)

    print(
        f"[CSE-V1 campaign] completed={len(rows)}/{manifest['expected_runs']} "
        f"failed={len(failures)} output={out_root}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
