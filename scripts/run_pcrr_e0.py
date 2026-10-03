#!/usr/bin/env python3
"""Run the predeclared PCRR-E0 validation-only campaign and smoke checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "pcrr_e0_postgpr_paired_residual"
DATA = RESEARCH / "data"
BASE_SHA = "3ef56df39a2d554aaac9a97ae9f6d4943d345f13"
BRANCH = "exp/pcrr_e0_postgpr_paired_residual"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = {"B": "base", "P": "paired", "S": "shuffled"}
SEEDS = (42, 43, 44)
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}
TMP = Path("/tmp/pcrr_e0_postgpr_paired_residual")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_provenance() -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != BRANCH:
        raise RuntimeError(f"runner must stay on {BRANCH}; current branch is {branch}")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError(f"HEAD is not descended from required base {BASE_SHA}")
    if git("show", "-s", "--format=%H", BASE_SHA) != BASE_SHA:
        raise RuntimeError(f"required base SHA {BASE_SHA} is unavailable")
    if not head:
        raise RuntimeError("unable to establish current HEAD")


def _run_paths(category: str, dataset: str, variant: str, seed: int):
    stem = f"{dataset}_{variant}_seed{seed}"
    metrics = DATA / "runs" / category / dataset / f"{stem}.json"
    checkpoint = TMP / "checkpoints" / category / f"{stem}.pt"
    hydra_dir = TMP / "hydra" / category / stem
    log = TMP / "logs" / category / f"{stem}.log"
    return stem, metrics, checkpoint, hydra_dir, log


def _command(
    *,
    dataset: str,
    variant: str,
    seed: int,
    checkpoint: Path,
    metrics: Path | None,
    hydra_dir: Path,
    device: str,
    epochs: int | None = None,
    max_train_batches: int | None = None,
) -> list[str]:
    is_lp = dataset == "sports-copurchase"
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=lp" if is_lp else "task=nc",
        "model=pcrr_mag_v0",
        f"model.variant={VARIANTS[variant]}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={hydra_dir}",
    ]
    if not is_lp:
        command.append("task.development_no_test=true")
        if dataset in ("Movies", "Grocery"):
            command.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    if metrics is not None and not is_lp:
        command.append(f"task.run_metrics_path={metrics}")
    if epochs is not None:
        command.append(f"task.epochs={epochs}")
    if max_train_batches is not None:
        command.append(f"task.max_train_batches={max_train_batches}")
    if is_lp:
        command.append("task.num_neighbors=[5,5,5]")
    return command


def _launch(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def _read_nc_metrics(path: Path, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"development_no_test missing from {path}")
    runs = payload.get("runs", [])
    if len(runs) != 1 or int(runs[0]["seed"]) != seed:
        raise RuntimeError(f"expected one NC run for seed {seed} in {path}")
    metrics = runs[0]["metrics"]
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metrics found in validation-only file {path}")
    for key in ("val_acc", "val_macro_f1", "val_ce"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"missing/non-finite {key} in {path}")
    return {"metrics": metrics, "metadata": runs[0].get("metadata", {})}


def _load_manifest() -> dict:
    path = RESEARCH / "run_manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("base_sha") != BASE_SHA or manifest.get("branch") != BRANCH:
            raise RuntimeError("existing PCRR-E0 manifest has incompatible provenance")
        return manifest
    return {
        "experiment": "PCRR-E0 Post-GPR Paired Cross-Modal Residual Refinement Screen",
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "head_at_branch_creation": BASE_SHA,
        "head_at_last_run": git("rev-parse", "HEAD"),
        "datasets": list(DATASETS),
        "fixed_dataset_splits": SPLITS,
        "variants": VARIANTS,
        "model_training_seeds": list(SEEDS),
        "pair_rank": 64,
        "pair_shuffle_seed_rule": "model_seed + 73000",
        "expected_smoke_nc_runs": 3,
        "expected_smoke_lp_runs": 2,
        "expected_formal_runs": 27,
        "evaluate_test": False,
        "development_no_test": True,
        "formal_runs": [],
        "smoke_nc_runs": [],
        "smoke_lp_runs": [],
        "created_at": now(),
        "status": "in_progress",
    }


def _record(manifest: dict, collection: str, item: dict) -> None:
    records = manifest.setdefault(collection, [])
    key_fields = ("dataset", "variant", "seed")
    key = tuple(item.get(field) for field in key_fields)
    existing = next(
        (i for i, old in enumerate(records)
         if tuple(old.get(field) for field in key_fields) == key),
        None,
    )
    if existing is None:
        records.append(item)
    else:
        records[existing] = item
    manifest["head_at_last_run"] = git("rev-parse", "HEAD")
    manifest["updated_at"] = now()
    atomic_json(RESEARCH / "run_manifest.json", manifest)


def _one_nc_run(
    manifest: dict,
    *,
    category: str,
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    epochs: int | None = None,
    max_train_batches: int | None = None,
) -> dict:
    stem, metrics_path, checkpoint, hydra_dir, log_path = _run_paths(
        category, dataset, variant, seed
    )
    collection = "formal_runs" if category == "formal" else "smoke_nc_runs"
    old = next(
        (item for item in manifest.get(collection, [])
         if item.get("stem") == stem and item.get("status") == "complete"),
        None,
    )
    if old and metrics_path.is_file() and checkpoint.is_file():
        parsed = _read_nc_metrics(metrics_path, seed)
        return {**old, **parsed}

    command = _command(
        dataset=dataset,
        variant=variant,
        seed=seed,
        checkpoint=checkpoint,
        metrics=metrics_path,
        hydra_dir=hydra_dir,
        device=device,
        epochs=epochs,
        max_train_batches=max_train_batches,
    )
    item = {
        "stem": stem,
        "dataset": dataset,
        "variant": variant,
        "model_variant": VARIANTS[variant],
        "seed": seed,
        "split_path": SPLITS[dataset],
        "evaluate_test": False,
        "development_no_test": True,
        "status": "running",
        "command": command,
        "run_metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint),
        "log_path": str(log_path),
        "started_at": now(),
    }
    if epochs is not None:
        item["epochs"] = epochs
    if max_train_batches is not None:
        item["max_train_batches"] = max_train_batches
    _record(manifest, collection, item)
    return_code = _launch(command, log_path)
    item["return_code"] = return_code
    if return_code != 0 or not metrics_path.is_file() or not checkpoint.is_file():
        item["status"] = "failed"
        item["finished_at"] = now()
        _record(manifest, collection, item)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-10000:]
        raise RuntimeError(f"{stem} failed (exit {return_code}); log tail:\n{tail}")
    parsed = _read_nc_metrics(metrics_path, seed)
    item["status"] = "complete"
    item["metrics"] = parsed["metrics"]
    item["metadata"] = parsed["metadata"]
    item["best_epoch"] = int(parsed["metadata"]["best_epoch"])
    item["finished_at"] = now()
    _record(manifest, collection, item)
    return {**item, **parsed}


def _checkpoint_model_state(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model_state"]


def _check_nc_smoke_item(item: dict) -> dict:
    import torch

    log_path = Path(item["log_path"])
    text = log_path.read_text(encoding="utf-8", errors="replace")
    checkpoint = Path(item["checkpoint_path"])
    state = _checkpoint_model_state(checkpoint)
    train_losses = [float(value) for value in re.findall(r"Train Loss ([0-9.eE+-]+)", text)]
    pnorm = float(state["pair_up.weight"].norm())
    pbias = float(state["pair_up.bias"].norm())
    variant = item["variant"]
    pair_parameters_moved = pnorm > 0 or pbias > 0
    observed = {
        "dataset": item["dataset"],
        "variant": variant,
        "seed": int(item["seed"]),
        "epochs": int(item["epochs"]),
        "trained_two_epochs": "Epoch 00002" in text,
        "validation_ran": "Val Acc" in text,
        "checkpoint_saved": checkpoint.is_file() and "Saved checkpoint:" in text,
        "finite_logged_train_loss": bool(train_losses) and all(math.isfinite(v) for v in train_losses),
        "pair_up_weight_norm_at_selected_checkpoint": pnorm,
        "pair_up_bias_norm_at_selected_checkpoint": pbias,
        "residual_parameters_moved_after_training": pair_parameters_moved,
        "residual_parameter_expectation_met": pair_parameters_moved if variant in ("P", "S") else not pair_parameters_moved,
        "initial_residual_exactly_zero": variant in ("P", "S"),
        "evaluate_test": False,
        "metrics": item["metrics"],
        "best_epoch": item["best_epoch"],
        "checkpoint_path": str(checkpoint),
        "log_path": str(log_path),
    }
    required = (
        "trained_two_epochs", "validation_ran", "checkpoint_saved",
        "finite_logged_train_loss", "residual_parameter_expectation_met",
    )
    observed["status"] = "complete" if all(observed[key] for key in required) else "failed"
    return observed


def _run_lp_smoke(manifest: dict, device: str, smoke_status: dict) -> None:
    collection = manifest.setdefault("smoke_lp_runs", [])
    for variant in ("B", "P"):
        stem = f"sports-copurchase_{variant}_seed42"
        checkpoint = TMP / "checkpoints" / "smoke" / f"{stem}.pt"
        hydra_dir = TMP / "hydra" / "smoke" / stem
        log_path = TMP / "logs" / "smoke" / f"{stem}.log"
        old = next((row for row in collection if row.get("stem") == stem and row.get("status") == "complete"), None)
        if old and checkpoint.is_file() and log_path.is_file():
            continue
        command = _command(
            dataset="sports-copurchase", variant=variant, seed=42,
            checkpoint=checkpoint, metrics=None, hydra_dir=hydra_dir,
            device=device, epochs=2, max_train_batches=2,
        )
        item = {
            "stem": stem, "dataset": "sports-copurchase", "variant": variant,
            "seed": 42, "epochs": 2, "max_train_batches": 2,
            "num_neighbors": [5, 5, 5],
            "positive_supervision_edge_removal": "repository global_eid protocol",
            "evaluate_test": False, "checkpoint_path": str(checkpoint),
            "log_path": str(log_path), "command": command, "status": "running",
        }
        _record(manifest, "smoke_lp_runs", item)
        code = _launch(command, log_path)
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        removed = [int(value) for value in re.findall(r"Positive Message Edges Removed (\d+)", text)]
        losses = [float(value) for value in re.findall(r"Train Loss ([0-9.eE+-]+)", text)]
        item.update({
            "return_code": code,
            "link_neighbor_loader": "Loader: LinkNeighborLoader" in text,
            "sampler_fanouts_observed": "Train neighbor sampling fanouts: [5, 5, 5]" in text,
            "positive_message_edges_removed_per_logged_batch": removed,
            "sampled_batch_train_loss_finite": bool(losses) and all(math.isfinite(v) for v in losses),
            "forward_backward_and_validation": code == 0 and "Val MRR" in text,
            "checkpoint_saved": checkpoint.is_file() and "Saved checkpoint:" in text,
            "test_evaluated": False,
            "finished_at": now(),
        })
        item["status"] = "complete" if (
            code == 0 and item["link_neighbor_loader"] and item["sampler_fanouts_observed"]
            and len(removed) >= 2 and item["sampled_batch_train_loss_finite"]
            and item["forward_backward_and_validation"] and item["checkpoint_saved"]
        ) else "failed"
        _record(manifest, "smoke_lp_runs", item)
        if item["status"] != "complete":
            raise RuntimeError(f"LP smoke failed for {variant}; inspect {log_path}")
    smoke_status["lp_runs"] = collection
    smoke_status["lp_protocol_checks"] = {
        "loader": "LinkNeighborLoader",
        "fanout": [5, 5, 5],
        "positive_supervision_edge_removal": "global_eid positive-message-edge masking observed per logged batch",
        "validation_inference": "Val MRR appears after training batches",
        "evaluate_test": False,
    }


def run_smoke(manifest: dict, device: str, smoke_status: dict) -> None:
    smoke_status.update({
        "status": "running", "base_sha": BASE_SHA, "branch": BRANCH,
        "device": device, "evaluate_test": False, "development_no_test": True,
        "fixed_nc_splits": SPLITS, "started_at": smoke_status.get("started_at", now()),
        "nc_runs": smoke_status.get("nc_runs", []),
    })
    atomic_json(RESEARCH / "smoke_status.json", smoke_status)
    for variant in ("B", "P", "S"):
        item = _one_nc_run(
            manifest, category="smoke", dataset="Movies", variant=variant,
            seed=42, device=device, epochs=2,
        )
        smoke_item = _check_nc_smoke_item(item)
        smoke_status["nc_runs"] = [
            row for row in smoke_status["nc_runs"] if row.get("variant") != variant
        ] + [smoke_item]
        atomic_json(RESEARCH / "smoke_status.json", smoke_status)
        if smoke_item["status"] != "complete":
            raise RuntimeError(f"NC smoke checks failed for {variant}: {smoke_item}")
    _run_lp_smoke(manifest, device, smoke_status)
    smoke_status["status"] = "complete"
    smoke_status["finished_at"] = now()
    atomic_json(RESEARCH / "smoke_status.json", smoke_status)


def run_formal(manifest: dict, device: str) -> None:
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                _one_nc_run(
                    manifest, category="formal", dataset=dataset, variant=variant,
                    seed=seed, device=device,
                )
    completed = [row for row in manifest["formal_runs"] if row.get("status") == "complete"]
    if len(completed) != 27:
        raise RuntimeError(f"expected 27 completed validation runs, found {len(completed)}")
    manifest["formal_status"] = "complete"
    manifest["status"] = "formal_complete"
    manifest["finished_at"] = now()
    atomic_json(RESEARCH / "run_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    validate_provenance()
    RESEARCH.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    status_path = RESEARCH / "smoke_status.json"
    smoke_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {
        "base_sha": BASE_SHA, "branch": BRANCH, "tests": {"status": "pending"}
    }
    if args.phase in {"smoke", "all"}:
        run_smoke(manifest, args.device, smoke_status)
    if args.phase in {"formal", "all"}:
        if smoke_status.get("status") != "complete":
            raise RuntimeError("smoke phase must complete before formal runs")
        run_formal(manifest, args.device)
    print(f"PCRR-E0 phase {args.phase} finished; manifest: {RESEARCH / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
