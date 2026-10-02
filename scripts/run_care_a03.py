#!/usr/bin/env python3
"""Run the frozen CARE-MAG A0.3 smokes and validation-only NC campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "care_a03_adaptation_placement_audit"
DATASETS = ("Movies", "Grocery", "ele-fashion")
FACTORS = {
    "G0": ("global", "off"),
    "GS": ("global", "static"),
    "GC": ("global", "context"),
    "N0": ("node", "off"),
    "NS": ("node", "static"),
    "NC": ("node", "context"),
}
SEEDS = (42, 43, 44)
BASE_SHA = "a44a556517112f15a8550f81c4b916906756d77f"
BRANCH = "exp/care_a03_adaptation_placement_audit"
TMP_ROOT = Path("/tmp/care_a03_adaptation_placement_audit")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def validate_provenance() -> None:
    branch = git_value("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"runner must stay on {BRANCH}, found {branch}")
    if git_value("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError(f"HEAD is not descended from required base {BASE_SHA}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_for(
    *,
    dataset: str,
    task: str,
    short_name: str,
    seed: int,
    num_runs: int,
    metrics_path: Path | None,
    checkpoint_path: Path,
    hydra_dir: Path,
    epochs: int | None = None,
    max_train_batches: int | None = None,
) -> list[str]:
    trajectory, adapter = FACTORS[short_name]
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        f"task={task}",
        "model=care_mag_v1",
        f"model.trajectory_mode={trajectory}",
        f"model.adapter_mode={adapter}",
        f"seed={seed}",
        f"num_runs={num_runs}",
        "device=cuda:0",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint_path}",
        f"hydra.run.dir={hydra_dir}",
    ]
    if task == "nc":
        command.append("task.development_no_test=true")
    if metrics_path is not None:
        command.append(f"task.run_metrics_path={metrics_path}")
    if epochs is not None:
        command.append(f"task.epochs={epochs}")
    if max_train_batches is not None:
        command.append(f"task.max_train_batches={max_train_batches}")
    return command


def launch(command: list[str], log_path: Path, timeout: int | None = None) -> int:
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
            timeout=timeout,
            check=False,
        )
    return int(result.returncode)


def tests_and_smokes() -> int:
    validate_provenance()
    (RESEARCH / "data" / "runs").mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = RESEARCH / "smoke_status.json"
    status = {
        "base_sha": BASE_SHA,
        "branch": git_value("branch", "--show-current"),
        "started_at": now(),
        "device": "cuda:0",
        "test_evaluation": False,
        "tests": {"status": "running", "command": [sys.executable, "-m", "pytest", "-q"]},
        "nc_runs": [],
        "lp_runs": [],
    }
    write_json(status_path, status)

    tests_log = TMP_ROOT / "logs" / "pytest_all.log"
    test_code = launch([sys.executable, "-m", "pytest", "-q"], tests_log)
    status["tests"].update(
        {
            "status": "passed" if test_code == 0 else "failed",
            "return_code": test_code,
            "log_path": str(tests_log),
            "finished_at": now(),
        }
    )
    write_json(status_path, status)
    if test_code:
        status["status"] = "failed"
        status["finished_at"] = now()
        write_json(status_path, status)
        return 1

    failed = False
    for short_name in FACTORS:
        metrics = RESEARCH / "data" / "smoke_nc" / f"{short_name}.json"
        checkpoint = TMP_ROOT / "smoke" / "nc" / f"Movies_{short_name}.pt"
        hydra_dir = TMP_ROOT / "smoke" / "hydra" / f"nc_Movies_{short_name}"
        command = command_for(
            dataset="Movies",
            task="nc",
            short_name=short_name,
            seed=42,
            num_runs=1,
            metrics_path=metrics,
            checkpoint_path=checkpoint,
            hydra_dir=hydra_dir,
            epochs=2,
        )
        log = TMP_ROOT / "logs" / f"smoke_nc_Movies_{short_name}.log"
        code = launch(command, log)
        log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        item = {
            "dataset": "Movies",
            "variant": short_name,
            "trajectory_mode": FACTORS[short_name][0],
            "adapter_mode": FACTORS[short_name][1],
            "seed": 42,
            "return_code": code,
            "trained": code == 0 and "Epoch 00002" in log_text,
            "validated": "Val Acc" in log_text,
            "checkpoint_saved": "Saved checkpoint:" in log_text and checkpoint.is_file(),
            "run_metrics_path": str(metrics),
            "checkpoint_path": str(checkpoint),
            "log_path": str(log),
            "command": command,
            "finished_at": now(),
        }
        status["nc_runs"].append(item)
        write_json(status_path, status)
        failed |= bool(code) or not item["trained"] or not item["validated"] or not item["checkpoint_saved"]
        if failed:
            break

    if not failed:
        for short_name in ("GC", "NC"):
            checkpoint = TMP_ROOT / "smoke" / "lp" / f"sports_{short_name}.pt"
            hydra_dir = TMP_ROOT / "smoke" / "hydra" / f"lp_sports_{short_name}"
            command = command_for(
                dataset="sports-copurchase",
                task="lp",
                short_name=short_name,
                seed=42,
                num_runs=1,
                metrics_path=None,
                checkpoint_path=checkpoint,
                hydra_dir=hydra_dir,
                epochs=2,
                max_train_batches=2,
            )
            log = TMP_ROOT / "logs" / f"smoke_lp_sports_{short_name}.log"
            code = launch(command, log)
            log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            match = re.search(r"Train neighbor sampling fanouts: (\[[^\]]+\])", log_text)
            removed = [int(value) for value in re.findall(r"Positive Message Edges Removed (\d+)", log_text)]
            item = {
                "dataset": "sports-copurchase",
                "variant": short_name,
                "trajectory_mode": FACTORS[short_name][0],
                "adapter_mode": FACTORS[short_name][1],
                "seed": 42,
                "epochs": 2,
                "max_train_batches": 2,
                "return_code": code,
                "observed": {
                    "link_neighbor_loader": "Loader: LinkNeighborLoader" in log_text,
                    "sampler_fanouts": json.loads(match.group(1)) if match else None,
                    "positive_message_edges_removed_by_epoch": removed,
                    "positive_supervision_removal_observed": bool(removed) and all(v > 0 for v in removed),
                    "forward_backward_completed": code == 0 and "Train Loss" in log_text,
                    "validation_inference_completed": "Eval node embeddings:" in log_text,
                    "checkpoint_saved": "Saved checkpoint:" in log_text and checkpoint.is_file(),
                },
                "checkpoint_path": str(checkpoint),
                "log_path": str(log),
                "command": command,
                "finished_at": now(),
            }
            status["lp_runs"].append(item)
            write_json(status_path, status)
            observed = item["observed"]
            failed |= bool(code) or not all(
                [
                    observed["link_neighbor_loader"],
                    observed["sampler_fanouts"] == [5, 5, 5],
                    observed["positive_supervision_removal_observed"],
                    observed["forward_backward_completed"],
                    observed["validation_inference_completed"],
                    observed["checkpoint_saved"],
                ]
            )
            if failed:
                break

    status["status"] = "failed" if failed else "passed"
    status["finished_at"] = now()
    write_json(status_path, status)
    return 1 if failed else 0


def expected_checkpoint_paths(base: Path, num_runs: int) -> list[Path]:
    if num_runs == 1:
        return [base]
    return [base.with_name(f"{base.stem}_run{i}{base.suffix}") for i in range(1, num_runs + 1)]


def run_formal(*, force: bool) -> int:
    validate_provenance()
    status_path = RESEARCH / "smoke_status.json"
    if not status_path.is_file():
        raise RuntimeError("run all tests and NC/LP smokes before the formal campaign")
    smoke = json.loads(status_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or smoke.get("tests", {}).get("status") != "passed":
        raise RuntimeError("the full tests and all required smokes must pass before formal NC")

    manifest_path = RESEARCH / "run_manifest.json"
    metrics_root = RESEARCH / "data" / "runs"
    metrics_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "head_at_start": git_value("rev-parse", "HEAD"),
        "seed": 42,
        "model_seeds": list(SEEDS),
        "num_runs_per_configuration": 3,
        "datasets": list(DATASETS),
        "variants": {
            name: {"trajectory_mode": modes[0], "adapter_mode": modes[1]}
            for name, modes in FACTORS.items()
        },
        "epochs": 300,
        "task": "full-graph NC",
        "evaluate_test": False,
        "development_no_test": True,
        "started_at": now(),
        "runs": [],
    }
    old: dict[tuple[str, str], dict] = {}
    if manifest_path.is_file() and not force:
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("base_sha") != BASE_SHA or prior.get("datasets") != list(DATASETS):
                raise RuntimeError("existing run manifest does not match this A0.3 campaign")
            old = {
                (item["dataset"], item["variant"]): item
                for item in prior.get("runs", [])
                if item.get("status") == "complete"
            }
        except json.JSONDecodeError:
            old = {}

    for dataset in DATASETS:
        for short_name in FACTORS:
            key = (dataset, short_name)
            metrics = metrics_root / dataset / f"{short_name}.json"
            checkpoint = TMP_ROOT / "formal" / "checkpoints" / dataset / f"{short_name}.pt"
            checkpoints = expected_checkpoint_paths(checkpoint, 3)
            if (
                key in old
                and metrics.is_file()
                and all(path.is_file() for path in checkpoints)
            ):
                item = dict(old[key])
                item["resumed"] = True
                manifest["runs"].append(item)
                write_json(manifest_path, manifest)
                continue

            hydra_dir = TMP_ROOT / "formal" / "hydra" / dataset / short_name
            command = command_for(
                dataset=dataset,
                task="nc",
                short_name=short_name,
                seed=42,
                num_runs=3,
                metrics_path=metrics,
                checkpoint_path=checkpoint,
                hydra_dir=hydra_dir,
            )
            log = TMP_ROOT / "formal" / "logs" / f"{dataset}_{short_name}.log"
            item = {
                "dataset": dataset,
                "variant": short_name,
                "trajectory_mode": FACTORS[short_name][0],
                "adapter_mode": FACTORS[short_name][1],
                "status": "running",
                "run_seeds": list(SEEDS),
                "evaluate_test": False,
                "development_no_test": True,
                "run_metrics_path": str(metrics),
                "checkpoint_path": str(checkpoint),
                "checkpoint_paths": [str(path) for path in checkpoints],
                "log_path": str(log),
                "command": command,
                "started_at": now(),
            }
            manifest["runs"].append(item)
            write_json(manifest_path, manifest)
            code = launch(command, log)
            item.update(
                {
                    "status": "complete" if code == 0 else "failed",
                    "return_code": code,
                    "checkpoints_saved": all(path.is_file() for path in checkpoints),
                    "finished_at": now(),
                }
            )
            write_json(manifest_path, manifest)
            if code or not item["checkpoints_saved"]:
                manifest["status"] = "failed; stopped at first failed configuration"
                manifest["finished_at"] = now()
                write_json(manifest_path, manifest)
                return 1

    manifest["status"] = "complete"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-only", action="store_true", help="run all tests and required GPU smokes")
    mode.add_argument("--formal", action="store_true", help="run the frozen 54-run validation-only NC campaign")
    parser.add_argument("--force", action="store_true", help="rerun completed formal configurations")
    args = parser.parse_args()
    if args.smoke_only:
        return tests_and_smokes()
    if args.formal:
        return run_formal(force=args.force)
    parser.error("choose --smoke-only or --formal")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
