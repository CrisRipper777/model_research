#!/usr/bin/env python3
"""Run the frozen SPGPR B0–B1 preflight and validation-only NC campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "spgpr_b01_filter_decomposition_screen"
DATA = RESEARCH / "data"
BASE_SHA = "e8470fdd529f4e71100c32f25d75a400095fc6de"
BRANCH = "exp/spgpr_b01_filter_decomposition_screen"
DATASETS = ("Movies", "Grocery", "ele-fashion")
REPEAT_DATASETS = ("Movies", "Grocery")
MODES = {
    "U": "uniform",
    "P": "positive_shared",
    "S": "signed_shared",
    "I": "signed_independent",
    "SP": "signed_shared_private",
}
SEEDS = (42, 43, 44)
TMP_ROOT = Path("/tmp/spgpr_b01_filter_decomposition_screen")
SPLIT_ROOT = Path("/hdd1/DataInHere/YHF/data/MAGB_split")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_provenance() -> None:
    branch = git_value("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"runner must stay on {BRANCH}, found {branch}")
    if git_value("rev-parse", "HEAD") != BASE_SHA:
        # Runs are allowed after the experiment code has been committed on this branch.
        if git_value("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
            raise RuntimeError(f"HEAD is not descended from required base {BASE_SHA}")


def fixed_split_override(dataset: str) -> str | None:
    if dataset in {"Movies", "Grocery"}:
        return str(SPLIT_ROOT / f"{dataset}_nc_seed42_train0.6_val0.2.pt")
    # ele-fashion uses its dataset-provided fixed split.pt in configs/dataset/ele-fashion.yaml.
    return None


def command_for(
    *,
    dataset: str,
    mode: str,
    seed: int,
    checkpoint: Path,
    metrics_path: Path | None,
    hydra_dir: Path,
    device: str,
    epochs: int | None = None,
    max_train_batches: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc" if dataset != "sports-copurchase" else "task=lp",
        "model=spgpr_mag_v0",
        f"model.filter_mode={MODES[mode]}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={hydra_dir}",
    ]
    if dataset != "sports-copurchase":
        command.append("task.development_no_test=true")
        split = fixed_split_override(dataset)
        if split:
            command.append(f"dataset.nc_split_path={split}")
    if metrics_path is not None:
        command.append(f"task.run_metrics_path={metrics_path}")
    if epochs is not None:
        command.append(f"task.epochs={epochs}")
    if max_train_batches is not None:
        command.append(f"task.max_train_batches={max_train_batches}")
    return command


def launch(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def read_single_run(metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"development_no_test missing from {metrics_path}")
    if any(key.startswith("test_") for key in payload["runs"][0]["metrics"]):
        raise RuntimeError(f"test metrics found in {metrics_path}")
    if len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"expected one run in {metrics_path}")
    item = payload["runs"][0]
    return {
        "seed": int(item["seed"]),
        "metrics": item["metrics"],
        "metadata": item["metadata"],
    }


def summarize_repeatability(rows: list[dict]) -> list[dict]:
    summary = []
    for dataset in REPEAT_DATASETS:
        for metric, source in (
            ("val_accuracy", "val_acc"),
            ("val_macro_f1", "val_macro_f1"),
            ("val_ce", "val_ce"),
            ("best_epoch", "best_epoch"),
        ):
            values = [
                float(row[metric]) for row in rows if row["dataset"] == dataset
            ]
            if len(values) != 3:
                continue
            summary.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "population_sd": statistics.pstdev(values),
                    "max_min_range": max(values) - min(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "source_metric": source,
                }
            )
    return summary


def run_tests_and_repeatability(device: str) -> int:
    validate_provenance()
    RESEARCH.mkdir(parents=True, exist_ok=True)
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = RESEARCH / "smoke_status.json"
    status = {
        "base_sha": BASE_SHA,
        "branch": git_value("branch", "--show-current"),
        "started_at": now(),
        "device": device,
        "test_evaluation": False,
        "fixed_splits": {
            "Movies": fixed_split_override("Movies"),
            "Grocery": fixed_split_override("Grocery"),
            "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
        },
        "tests": {"status": "running", "command": [sys.executable, "-m", "pytest", "-q"]},
        "repeatability_runs": [],
        "nc_runs": [],
        "lp_runs": [],
    }
    write_json(status_path, status)

    test_log = TMP_ROOT / "logs" / "pytest_all.log"
    test_code = launch([sys.executable, "-m", "pytest", "-q"], test_log)
    status["tests"].update(
        {
            "status": "passed" if test_code == 0 else "failed",
            "return_code": test_code,
            "log_path": str(test_log),
            "finished_at": now(),
        }
    )
    write_json(status_path, status)
    if test_code:
        status["status"] = "failed"
        write_json(status_path, status)
        return 1

    repeat_rows: list[dict] = []
    for dataset in REPEAT_DATASETS:
        for repeat_id in range(1, 4):
            stem = f"{dataset}_S_seed42_repeat{repeat_id}"
            metrics = DATA / "repeatability_runs" / f"{stem}.json"
            checkpoint = TMP_ROOT / "checkpoints" / "repeatability" / f"{stem}.pt"
            hydra = TMP_ROOT / "hydra" / stem
            log = TMP_ROOT / "logs" / f"{stem}.log"
            command = command_for(
                dataset=dataset,
                mode="S",
                seed=42,
                checkpoint=checkpoint,
                metrics_path=metrics,
                hydra_dir=hydra,
                device=device,
            )
            code = launch(command, log)
            item = {
                "dataset": dataset,
                "variant": "S",
                "seed": 42,
                "repeat_id": repeat_id,
                "return_code": code,
                "metrics_path": str(metrics),
                "checkpoint_path": str(checkpoint),
                "log_path": str(log),
                "command": command,
                "finished_at": now(),
            }
            if code == 0 and metrics.is_file() and checkpoint.is_file():
                result = read_single_run(metrics)
                item["best_epoch"] = result["metadata"]["best_epoch"]
                item["metrics"] = result["metrics"]
                repeat_rows.append(
                    {
                        "dataset": dataset,
                        "variant": "S",
                        "seed": 42,
                        "repeat_id": repeat_id,
                        "val_accuracy": float(result["metrics"]["val_acc"]),
                        "val_macro_f1": float(result["metrics"]["val_macro_f1"]),
                        "val_ce": float(result["metrics"]["val_ce"]),
                        "best_epoch": int(result["metadata"]["best_epoch"]),
                        "variation_type": "same-seed execution variation",
                        "metrics_path": str(metrics),
                    }
                )
            status["repeatability_runs"].append(item)
            write_csv(DATA / "repeatability_by_run.csv", repeat_rows)
            write_csv(
                DATA / "repeatability_summary.csv", summarize_repeatability(repeat_rows)
            )
            write_json(status_path, status)
            if code or not metrics.is_file() or not checkpoint.is_file():
                status["status"] = "failed during repeatability"
                write_json(status_path, status)
                return 1

    if len(repeat_rows) != 6:
        raise RuntimeError(f"expected six repeatability runs, found {len(repeat_rows)}")
    status["repeatability_status"] = "passed"

    for mode in MODES:
        stem = f"Movies_{mode}_seed42_smoke"
        metrics = DATA / "smoke_nc" / f"{mode}.json"
        checkpoint = TMP_ROOT / "checkpoints" / "smoke" / f"{stem}.pt"
        hydra = TMP_ROOT / "hydra" / stem
        log = TMP_ROOT / "logs" / f"{stem}.log"
        command = command_for(
            dataset="Movies",
            mode=mode,
            seed=42,
            checkpoint=checkpoint,
            metrics_path=metrics,
            hydra_dir=hydra,
            device=device,
            epochs=2,
        )
        code = launch(command, log)
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        item = {
            "dataset": "Movies",
            "variant": mode,
            "seed": 42,
            "epochs": 2,
            "return_code": code,
            "trained": code == 0 and "Epoch 00002" in log_text,
            "validated": "Val Acc" in log_text,
            "checkpoint_saved": "Saved checkpoint:" in log_text and checkpoint.is_file(),
            "metrics_path": str(metrics),
            "checkpoint_path": str(checkpoint),
            "log_path": str(log),
            "command": command,
            "finished_at": now(),
        }
        status["nc_runs"].append(item)
        write_json(status_path, status)
        if code or not item["trained"] or not item["validated"] or not item["checkpoint_saved"]:
            status["status"] = "failed during NC smoke"
            write_json(status_path, status)
            return 1

    for mode in ("S", "SP"):
        stem = f"sports_{mode}_seed42_lp_smoke"
        checkpoint = TMP_ROOT / "checkpoints" / "smoke" / f"{stem}.pt"
        hydra = TMP_ROOT / "hydra" / stem
        log = TMP_ROOT / "logs" / f"{stem}.log"
        command = command_for(
            dataset="sports-copurchase",
            mode=mode,
            seed=42,
            checkpoint=checkpoint,
            metrics_path=None,
            hydra_dir=hydra,
            device=device,
            epochs=2,
            max_train_batches=2,
        )
        code = launch(command, log)
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        fanout_match = re.search(r"Train neighbor sampling fanouts: (\[[^\]]+\])", log_text)
        removed = [int(value) for value in re.findall(r"Positive Message Edges Removed (\d+)", log_text)]
        observed = {
            "link_neighbor_loader": "Loader: LinkNeighborLoader" in log_text,
            "sampler_fanouts": json.loads(fanout_match.group(1)) if fanout_match else None,
            "positive_message_edges_removed_by_epoch": removed,
            "positive_supervision_removal_observed": bool(removed) and all(value > 0 for value in removed),
            "forward_backward_completed": code == 0 and "Train Loss" in log_text,
            "validation_inference_completed": "Eval node embeddings:" in log_text,
            "checkpoint_saved": "Saved checkpoint:" in log_text and checkpoint.is_file(),
        }
        item = {
            "dataset": "sports-copurchase",
            "variant": mode,
            "seed": 42,
            "epochs": 2,
            "max_train_batches": 2,
            "return_code": code,
            "observed": observed,
            "checkpoint_path": str(checkpoint),
            "log_path": str(log),
            "command": command,
            "finished_at": now(),
        }
        status["lp_runs"].append(item)
        write_json(status_path, status)
        if code or not all(
            [
                observed["link_neighbor_loader"],
                observed["sampler_fanouts"] == [5, 5, 5],
                observed["positive_supervision_removal_observed"],
                observed["forward_backward_completed"],
                observed["validation_inference_completed"],
                observed["checkpoint_saved"],
            ]
        ):
            status["status"] = "failed during LP smoke"
            write_json(status_path, status)
            return 1

    status["status"] = "passed"
    status["finished_at"] = now()
    write_json(status_path, status)
    return 0


def run_formal(device: str, force: bool) -> int:
    validate_provenance()
    smoke_path = RESEARCH / "smoke_status.json"
    if not smoke_path.is_file():
        raise RuntimeError("run repository tests, repeatability, and both smokes first")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or smoke.get("repeatability_status") != "passed":
        raise RuntimeError("tests, six repeatability runs, NC smoke, and LP smoke must pass first")

    manifest_path = RESEARCH / "run_manifest.json"
    manifest = {
        "experiment": "SPGPR B0–B1 Shared/Private GPR Filter Decomposition Screen",
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "head_at_start": git_value("rev-parse", "HEAD"),
        "datasets": list(DATASETS),
        "fixed_dataset_splits": smoke["fixed_splits"],
        "variants": MODES,
        "model_training_seeds": list(SEEDS),
        "num_runs_per_process": 1,
        "expected_runs": 45,
        "paired_seed_design": True,
        "split_count_per_dataset": 1,
        "evaluate_test": False,
        "development_no_test": True,
        "optimizer_and_hyperparameters": "unchanged project NC protocol",
        "started_at": now(),
        "status": "running",
        "runs": [],
    }
    completed_prior: dict[tuple[str, str, int], dict] = {}
    if manifest_path.is_file() and not force:
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("base_sha") != BASE_SHA:
            raise RuntimeError("existing formal manifest uses a different base SHA")
        for item in old_manifest.get("runs", []):
            if item.get("status") == "complete":
                completed_prior[(item["dataset"], item["variant"], int(item["seed"]))] = item

    for dataset in DATASETS:
        for mode in MODES:
            for seed in SEEDS:
                key = (dataset, mode, seed)
                old = completed_prior.get(key)
                if old and Path(old["checkpoint_path"]).is_file() and Path(old["run_metrics_path"]).is_file():
                    manifest["runs"].append(old)
                    continue
                stem = f"{dataset}_{mode}_seed{seed}"
                metrics = DATA / "runs" / dataset / f"{mode}_seed{seed}.json"
                checkpoint = TMP_ROOT / "checkpoints" / "formal" / f"{stem}.pt"
                hydra = TMP_ROOT / "hydra" / f"formal_{stem}"
                log = TMP_ROOT / "logs" / f"formal_{stem}.log"
                command = command_for(
                    dataset=dataset,
                    mode=mode,
                    seed=seed,
                    checkpoint=checkpoint,
                    metrics_path=metrics,
                    hydra_dir=hydra,
                    device=device,
                )
                item = {
                    "dataset": dataset,
                    "variant": mode,
                    "filter_mode": MODES[mode],
                    "seed": seed,
                    "split_path": fixed_split_override(dataset)
                    or smoke["fixed_splits"][dataset],
                    "evaluate_test": False,
                    "development_no_test": True,
                    "status": "running",
                    "command": command,
                    "run_metrics_path": str(metrics),
                    "checkpoint_path": str(checkpoint),
                    "log_path": str(log),
                    "started_at": now(),
                }
                manifest["runs"].append(item)
                write_json(manifest_path, manifest)
                code = launch(command, log)
                if code == 0 and metrics.is_file() and checkpoint.is_file():
                    run = read_single_run(metrics)
                    item.update(
                        {
                            "status": "complete",
                            "return_code": code,
                            "best_epoch": int(run["metadata"]["best_epoch"]),
                            "metrics": run["metrics"],
                            "finished_at": now(),
                        }
                    )
                else:
                    item.update(
                        {
                            "status": "failed",
                            "return_code": code,
                            "finished_at": now(),
                        }
                    )
                    manifest["status"] = f"failed at {dataset}/{mode}/seed{seed}"
                    manifest["finished_at"] = now()
                    write_json(manifest_path, manifest)
                    return 1
                write_json(manifest_path, manifest)

    expected_keys = {
        (dataset, mode, seed)
        for dataset in DATASETS
        for mode in MODES
        for seed in SEEDS
    }
    actual = {
        (item["dataset"], item["variant"], int(item["seed"]))
        for item in manifest["runs"]
        if item.get("status") == "complete"
    }
    if actual != expected_keys or len(manifest["runs"]) != 45:
        raise RuntimeError(f"formal campaign incomplete: {len(actual)}/45")
    manifest["status"] = "complete"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="tests, repeatability, and both execution smokes")
    mode.add_argument("--formal", action="store_true", help="run the 45 fixed-split validation-only NC jobs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="rerun completed formal configurations")
    args = parser.parse_args()
    if args.preflight:
        return run_tests_and_repeatability(args.device)
    if args.formal:
        return run_formal(args.device, args.force)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
