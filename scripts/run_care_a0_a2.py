#!/usr/bin/env python3
"""Run the frozen CARE-MAG A0–A2 smoke checks or NC campaign."""

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
RESEARCH = ROOT / "research" / "care_a0_a2_context_adapter_screen"
VARIANTS = ("prior_only", "structural_base", "static_adapter", "context_adapter")
DATASETS = ("Movies", "Grocery", "ele-fashion")
TMP_ROOT = Path("/tmp/care_a0_a2_context_adapter_screen")


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


def command_for(
    *,
    dataset: str,
    task: str,
    variant: str,
    seed: int,
    num_runs: int,
    metrics_path: Path | None,
    checkpoint_path: Path,
    hydra_dir: Path,
    epochs: int | None = None,
    max_train_batches: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        f"task={task}",
        "model=care_mag_v0",
        f"model.variant={variant}",
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


def launch(command: list[str], log_path: Path, timeout: int | None = None) -> tuple[int, str]:
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
    return int(result.returncode), str(log_path)


def run_smokes() -> int:
    status_path = RESEARCH / "smoke_status.json"
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    status = {
        "base_sha": git_value("rev-parse", "refs/remotes/origin/main"),
        "started_at": now(),
        "device": "cuda:0",
        "test_evaluation": False,
        "nc_runs": [],
        "lp_run": None,
    }
    failed = False
    for variant in VARIANTS:
        metrics = RESEARCH / "data" / "smoke_nc" / f"{variant}.json"
        checkpoint = TMP_ROOT / "smoke" / "nc" / f"Movies_{variant}.pt"
        hydra_dir = TMP_ROOT / "smoke" / "hydra" / f"nc_Movies_{variant}"
        command = command_for(
            dataset="Movies",
            task="nc",
            variant=variant,
            seed=42,
            num_runs=1,
            metrics_path=metrics,
            checkpoint_path=checkpoint,
            hydra_dir=hydra_dir,
            epochs=2,
        )
        log = TMP_ROOT / "logs" / f"smoke_nc_Movies_{variant}.log"
        code, log_name = launch(command, log)
        result = {
            "dataset": "Movies",
            "variant": variant,
            "seed": 42,
            "return_code": code,
            "run_metrics_path": str(metrics),
            "checkpoint_path": str(checkpoint),
            "log_path": log_name,
            "command": command,
            "finished_at": now(),
        }
        status["nc_runs"].append(result)
        write_json(status_path, status)
        if code:
            failed = True

    if not failed:
        command = command_for(
            dataset="sports-copurchase",
            task="lp",
            variant="context_adapter",
            seed=42,
            num_runs=1,
            metrics_path=None,
            checkpoint_path=TMP_ROOT / "smoke" / "lp" / "sports_context.pt",
            hydra_dir=TMP_ROOT / "smoke" / "hydra" / "lp_sports_context",
            epochs=2,
            max_train_batches=2,
        )
        log = TMP_ROOT / "logs" / "smoke_lp_sports_context.log"
        code, log_name = launch(command, log)
        log_text = log.read_text(encoding="utf-8", errors="replace")
        fanout_match = re.search(
            r"Train neighbor sampling fanouts: (\[[^\]]+\])", log_text
        )
        removed_edges = [
            int(value)
            for value in re.findall(
                r"Positive Message Edges Removed (\d+)", log_text
            )
        ]
        status["lp_run"] = {
            "dataset": "sports-copurchase",
            "variant": "context_adapter",
            "seed": 42,
            "epochs": 2,
            "max_train_batches": 2,
            "return_code": code,
            "observed": {
                "link_neighbor_loader": "Loader: LinkNeighborLoader" in log_text,
                "sampler_fanouts": json.loads(fanout_match.group(1)) if fanout_match else None,
                "positive_message_edges_removed_by_epoch": removed_edges,
                "validation_inference_completed": "Eval node embeddings:" in log_text,
                "checkpoint_saved": "Saved checkpoint:" in log_text,
            },
            "checkpoint_path": str(TMP_ROOT / "smoke" / "lp" / "sports_context.pt"),
            "log_path": log_name,
            "command": command,
            "finished_at": now(),
        }
        failed |= bool(code)
    status["finished_at"] = now()
    status["status"] = "failed" if failed else "passed"
    write_json(status_path, status)
    return 1 if failed else 0


def run_formal(*, force: bool) -> int:
    manifest_path = RESEARCH / "run_manifest.json"
    metrics_root = RESEARCH / "data" / "runs"
    metrics_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_sha": git_value("rev-parse", "refs/remotes/origin/main"),
        "branch": git_value("branch", "--show-current"),
        "seed": 42,
        "model_seeds": [42, 43, 44],
        "num_runs_per_configuration": 3,
        "datasets": list(DATASETS),
        "variants": list(VARIANTS),
        "epochs": 300,
        "task": "full-graph NC",
        "evaluate_test": False,
        "development_no_test": True,
        "started_at": now(),
        "runs": [],
    }
    old = {}
    if manifest_path.exists() and not force:
        try:
            old = {
                (item["dataset"], item["variant"]): item
                for item in json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
                if item.get("status") == "complete"
            }
        except (KeyError, json.JSONDecodeError):
            old = {}

    for dataset in DATASETS:
        for variant in VARIANTS:
            key = (dataset, variant)
            metrics = metrics_root / dataset / f"{variant}.json"
            checkpoint = TMP_ROOT / "formal" / "checkpoints" / dataset / f"{variant}.pt"
            hydra_dir = TMP_ROOT / "formal" / "hydra" / dataset / variant
            if key in old and metrics.is_file() and checkpoint.with_name(
                f"{checkpoint.stem}_run3{checkpoint.suffix}"
            ).is_file():
                item = dict(old[key])
                item["resumed"] = True
                manifest["runs"].append(item)
                write_json(manifest_path, manifest)
                continue

            command = command_for(
                dataset=dataset,
                task="nc",
                variant=variant,
                seed=42,
                num_runs=3,
                metrics_path=metrics,
                checkpoint_path=checkpoint,
                hydra_dir=hydra_dir,
            )
            log = TMP_ROOT / "formal" / "logs" / f"{dataset}_{variant}.log"
            item = {
                "dataset": dataset,
                "variant": variant,
                "status": "running",
                "run_seeds": [42, 43, 44],
                "evaluate_test": False,
                "development_no_test": True,
                "run_metrics_path": str(metrics),
                "checkpoint_path": str(checkpoint),
                "log_path": str(log),
                "command": command,
                "started_at": now(),
            }
            manifest["runs"].append(item)
            write_json(manifest_path, manifest)
            code, log_name = launch(command, log)
            item.update(
                {
                    "status": "complete" if code == 0 else "failed",
                    "return_code": code,
                    "log_path": log_name,
                    "finished_at": now(),
                }
            )
            write_json(manifest_path, manifest)
            if code:
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
    mode.add_argument("--smoke-only", action="store_true", help="run GPU NC and sampled-LP smoke checks")
    mode.add_argument("--formal", action="store_true", help="run the frozen 36-run NC campaign (default)")
    parser.add_argument("--force", action="store_true", help="rerun completed configurations")
    args = parser.parse_args()
    if args.smoke_only:
        return run_smokes()
    return run_formal(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
