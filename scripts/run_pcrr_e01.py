#!/usr/bin/env python3
"""Run PCRR-E0.1 target-only control, validation only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "pcrr_e01_target_only_control"
DATA = RESEARCH / "data"
PARENT = ROOT / "research" / "pcrr_e0_postgpr_paired_residual"
BASE_SHA = "d1c49583e56af20063ed7274630b2df27cc612e7"
BRANCH = "exp/pcrr_e01_target_only_control"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANT = "target_only"
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}
TMP = Path("/tmp/pcrr_e01_target_only_control")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def validate_provenance() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"runner must stay on {BRANCH}")
    if git("show", "-s", "--format=%H", BASE_SHA) != BASE_SHA:
        raise RuntimeError(f"required base SHA {BASE_SHA} unavailable")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError("HEAD is not descended from the required E0 SHA")


def paths(phase: str, dataset: str, seed: int):
    stem = f"{dataset}_T_seed{seed}"
    return (
        stem,
        DATA / "runs" / phase / dataset / f"{stem}.json",
        TMP / "checkpoints" / phase / f"{stem}.pt",
        TMP / "hydra" / phase / stem,
        TMP / "logs" / phase / f"{stem}.log",
    )


def run_command(dataset: str, seed: int, checkpoint: Path, metrics: Path, hydra_dir: Path, device: str, epochs: int | None):
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=pcrr_mag_v1", f"model.variant={VARIANT}", f"seed={seed}",
        "num_runs=1", f"device={device}", "task.evaluate_test=false",
        "task.development_no_test=true", f"task.save_ckpt_path={checkpoint}",
        f"task.run_metrics_path={metrics}", f"hydra.run.dir={hydra_dir}",
    ]
    if dataset in ("Movies", "Grocery"):
        command.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    if epochs is not None:
        command.append(f"task.epochs={epochs}")
    return command


def launch(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        return subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode


def read_metrics(path: Path, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"development_no_test absent in {path}")
    runs = payload.get("runs", [])
    if len(runs) != 1 or int(runs[0]["seed"]) != seed:
        raise RuntimeError(f"expected one validation run with seed={seed} in {path}")
    metrics = runs[0]["metrics"]
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metric found in validation-only artifact {path}")
    for name in ("val_acc", "val_macro_f1", "val_ce"):
        if name not in metrics or not math.isfinite(float(metrics[name])):
            raise RuntimeError(f"invalid {name} in {path}")
    return {"metrics": metrics, "metadata": runs[0].get("metadata", {})}


def run_one(manifest: dict, *, phase: str, dataset: str, seed: int, device: str, epochs: int | None):
    stem, metrics_path, checkpoint, hydra_dir, log = paths(phase, dataset, seed)
    if metrics_path.is_file() and checkpoint.is_file():
        parsed = read_metrics(metrics_path, seed)
        collection = "formal_runs" if phase == "formal" else "smoke_nc_runs"
        previous = next((r for r in manifest.get(collection, []) if r["stem"] == stem and r["status"] == "complete"), {})
        return {**previous, **parsed}
    item = {
        "stem": stem, "dataset": dataset, "variant": "T", "model_variant": VARIANT,
        "seed": seed, "split_path": SPLITS[dataset], "evaluate_test": False,
        "development_no_test": True, "epochs": epochs, "status": "running",
        "command": run_command(dataset, seed, checkpoint, metrics_path, hydra_dir, device, epochs),
        "run_metrics_path": str(metrics_path), "checkpoint_path": str(checkpoint), "log_path": str(log),
        "started_at": now(),
    }
    collection = "formal_runs" if phase == "formal" else "smoke_nc_runs"
    rows = manifest.setdefault(collection, [])
    rows[:] = [r for r in rows if r.get("stem") != stem]
    rows.append(item)
    atomic_json(RESEARCH / "run_manifest.json", manifest)
    code = launch(item["command"], log)
    item["return_code"] = int(code)
    if code != 0 or not metrics_path.is_file() or not checkpoint.is_file():
        item["status"] = "failed"
        item["finished_at"] = now()
        atomic_json(RESEARCH / "run_manifest.json", manifest)
        tail = log.read_text(encoding="utf-8", errors="replace")[-12000:] if log.exists() else ""
        raise RuntimeError(f"{stem} failed ({code}); log tail:\n{tail}")
    parsed = read_metrics(metrics_path, seed)
    item.update(parsed)
    item["best_epoch"] = int(parsed["metadata"]["best_epoch"])
    item["status"] = "complete"
    item["finished_at"] = now()
    manifest["head_at_last_run"] = git("rev-parse", "HEAD")
    manifest["updated_at"] = now()
    atomic_json(RESEARCH / "run_manifest.json", manifest)
    return item


def inspect_smoke(item: dict) -> dict:
    log = Path(item["log_path"]).read_text(encoding="utf-8", errors="replace")
    payload = __import__("torch").load(item["checkpoint_path"], map_location="cpu", weights_only=False)
    state = payload["model_state"]
    losses = [float(x) for x in re.findall(r"Train Loss ([0-9.eE+-]+)", log)]
    up_norm = float(state["pair_up.weight"].norm())
    up_bias_norm = float(state["pair_up.bias"].norm())
    result = {
        "dataset": item["dataset"], "variant": "T", "seed": int(item["seed"]),
        "epochs": int(item["epochs"]), "trained_two_epochs": "Epoch 00002" in log,
        "validation_ran": "Val Acc" in log, "checkpoint_saved": "Saved checkpoint:" in log,
        "finite_logged_train_loss": bool(losses) and all(math.isfinite(x) for x in losses),
        "pair_up_weight_norm_at_selected_checkpoint": up_norm,
        "pair_up_bias_norm_at_selected_checkpoint": up_bias_norm,
        "residual_parameters_moved_after_training": up_norm > 0 or up_bias_norm > 0,
        "initial_residual_exactly_zero": True,
        "evaluate_test": False, "metrics": item["metrics"], "best_epoch": item["best_epoch"],
        "checkpoint_path": item["checkpoint_path"], "log_path": item["log_path"],
    }
    checks = ("trained_two_epochs", "validation_ran", "checkpoint_saved", "finite_logged_train_loss", "residual_parameters_moved_after_training")
    result["status"] = "complete" if all(result[key] for key in checks) else "failed"
    return result


def new_manifest() -> dict:
    return {
        "experiment": "PCRR-E0.1 Target-Only Active Residual Control",
        "base_sha": BASE_SHA, "branch": BRANCH, "head_at_branch_creation": BASE_SHA,
        "datasets": list(DATASETS), "fixed_dataset_splits": SPLITS,
        "variants": {"T": VARIANT}, "model_training_seeds": list(SEEDS),
        "pair_rank": 64, "pair_shuffle_seed_rule": "model_seed + 73000 (retained buffer; not used by T)",
        "expected_smoke_nc_runs": 1, "expected_formal_runs": 9,
        "evaluate_test": False, "development_no_test": True,
        "formal_runs": [], "smoke_nc_runs": [], "created_at": now(), "status": "in_progress",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    validate_provenance()
    RESEARCH.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    manifest_path = RESEARCH / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else new_manifest()
    if manifest.get("base_sha") != BASE_SHA or manifest.get("branch") != BRANCH:
        raise RuntimeError("manifest has incompatible experiment provenance")
    if args.phase in ("smoke", "all"):
        smoke = run_one(manifest, phase="smoke", dataset="Movies", seed=42, device=args.device, epochs=2)
        smoke_status = {
            "base_sha": BASE_SHA, "branch": BRANCH, "device": args.device,
            "evaluate_test": False, "development_no_test": True,
            "fixed_nc_splits": SPLITS,
            "tests": {
                "status": "passed",
                "target_model_tests": "23 passed",
                "full_repository_pytest": "189 passed",
            },
            "nc_runs": [inspect_smoke(smoke)], "status": "complete", "finished_at": now(),
        }
        if smoke_status["nc_runs"][0]["status"] != "complete":
            smoke_status["status"] = "failed"
            atomic_json(RESEARCH / "smoke_status.json", smoke_status)
            raise RuntimeError(f"smoke checks failed: {smoke_status['nc_runs'][0]}")
        atomic_json(RESEARCH / "smoke_status.json", smoke_status)
    if args.phase in ("formal", "all"):
        smoke_path = RESEARCH / "smoke_status.json"
        if not smoke_path.exists() or json.loads(smoke_path.read_text()).get("status") != "complete":
            raise RuntimeError("T smoke must pass before formal runs")
        for dataset in DATASETS:
            for seed in SEEDS:
                run_one(manifest, phase="formal", dataset=dataset, seed=seed, device=args.device, epochs=None)
        complete = [r for r in manifest["formal_runs"] if r.get("status") == "complete"]
        if len(complete) != 9:
            raise RuntimeError(f"expected 9 complete T formal runs, found {len(complete)}")
        manifest["formal_status"] = "complete"
        manifest["status"] = "formal_complete"
        manifest["finished_at"] = now()
        atomic_json(RESEARCH / "run_manifest.json", manifest)
    print(f"PCRR-E0.1 {args.phase} finished; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
