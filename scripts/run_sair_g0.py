#!/usr/bin/env python3
"""Run SAIR-G0's NC smoke, LP protocol smoke, and validation-only NC grid."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "sair_g0_anchor_response_readout"
DATA = RESEARCH / "data"
BASE_SHA = "997b89b8b0e59ca2da9654d07cc184cc1ce14169"
BRANCH = "exp/sair_g0_anchor_response_readout"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = {"B": "base", "C": "generic", "D": "decoupled"}
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}
TMP = Path("/tmp/sair_g0_anchor_response_readout")
SOURCE_PATHS = (
    "src/models/sair_mag_v0.py", "configs/model/sair_mag_v0.yaml",
    "scripts/run_sair_g0.py", "scripts/analyze_sair_g0.py",
    "tests/test_sair_mag_v0.py", "research/sair_g0_anchor_response_readout/design_audit.md",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def validate_source(manifest: dict) -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"runner must stay on {BRANCH}")
    if git("show", "-s", "--format=%H", BASE_SHA) != BASE_SHA:
        raise RuntimeError(f"required parent SHA {BASE_SHA} unavailable")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError("current HEAD is not descended from the required base")
    source_sha = manifest.get("source_commit_sha")
    if not source_sha or source_sha == "PENDING_SOURCE_COMMIT":
        raise RuntimeError("source_commit_sha must be recorded in a commit before any run")
    if git("merge-base", source_sha, "HEAD") != source_sha:
        raise RuntimeError("HEAD does not contain the declared source commit")
    for path in SOURCE_PATHS:
        subprocess.run(["git", "diff", "--quiet", source_sha, "--", path], cwd=ROOT, check=True)


def _paths(phase: str, dataset: str, variant: str, seed: int):
    stem = f"{dataset}_{variant}_seed{seed}"
    return (
        stem, DATA / "runs" / phase / dataset / f"{stem}.json",
        TMP / "checkpoints" / phase / f"{stem}.pt",
        TMP / "hydra" / phase / stem, TMP / "logs" / phase / f"{stem}.log",
    )


def _nc_command(dataset, variant, seed, checkpoint, metrics, hydra_dir, device, epochs):
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=sair_mag_v0", f"model.variant={VARIANTS[variant]}", f"seed={seed}",
        "num_runs=1", f"device={device}", "task.evaluate_test=false",
        "task.development_no_test=true", f"task.save_ckpt_path={checkpoint}",
        f"task.run_metrics_path={metrics}", f"hydra.run.dir={hydra_dir}",
    ]
    if dataset in ("Movies", "Grocery"):
        command.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    if epochs is not None:
        command.append(f"task.epochs={epochs}")
    return command


def _launch(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(result.returncode)


def _read_metrics(path: Path, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"development_no_test missing in {path}")
    runs = payload.get("runs", [])
    if len(runs) != 1 or int(runs[0]["seed"]) != seed:
        raise RuntimeError(f"expected one run for seed {seed} in {path}")
    metrics = runs[0]["metrics"]
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metric found in validation-only artifact {path}")
    for key in ("val_acc", "val_macro_f1", "val_ce"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"invalid {key} in {path}")
    return {"metrics": metrics, "metadata": runs[0].get("metadata", {})}


def _run_nc_one(manifest, *, phase, dataset, variant, seed, device, epochs):
    stem, metrics_path, checkpoint, hydra_dir, log_path = _paths(phase, dataset, variant, seed)
    collection = "formal_runs" if phase == "formal" else "smoke_nc_runs"
    old = next((row for row in manifest.get(collection, []) if row.get("stem") == stem and row.get("status") == "complete"), None)
    if old and metrics_path.is_file() and checkpoint.is_file():
        return {**old, **_read_metrics(metrics_path, seed)}
    command = _nc_command(dataset, variant, seed, checkpoint, metrics_path, hydra_dir, device, epochs)
    item = {
        "stem": stem, "dataset": dataset, "variant": variant,
        "model_variant": VARIANTS[variant], "seed": seed,
        "split_path": SPLITS[dataset], "evaluate_test": False,
        "development_no_test": True, "epochs": epochs,
        "source_commit_sha": manifest["source_commit_sha"], "status": "running",
        "command": command, "run_metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint), "log_path": str(log_path), "started_at": now(),
    }
    rows = manifest.setdefault(collection, [])
    rows[:] = [row for row in rows if row.get("stem") != stem]
    rows.append(item)
    atomic_json(RESEARCH / "run_manifest.json", manifest)
    code = _launch(command, log_path)
    item["return_code"] = code
    if code != 0 or not metrics_path.is_file() or not checkpoint.is_file():
        item.update(status="failed", finished_at=now())
        atomic_json(RESEARCH / "run_manifest.json", manifest)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:] if log_path.exists() else ""
        raise RuntimeError(f"{stem} failed ({code}); log tail:\n{tail}")
    parsed = _read_metrics(metrics_path, seed)
    item.update(parsed)
    item["best_epoch"] = int(parsed["metadata"]["best_epoch"])
    item.update(status="complete", finished_at=now())
    atomic_json(RESEARCH / "run_manifest.json", manifest)
    return item


def _inspect_nc_smoke(item):
    import torch
    log = Path(item["log_path"]).read_text(encoding="utf-8", errors="replace")
    payload = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
    state = payload["model_state"]
    losses = [float(v) for v in re.findall(r"Train Loss ([0-9.eE+-]+)", log)]
    pair_norm = float(state["pair_up.weight"].norm())
    expected_update = item["variant"] in ("C", "D")
    result = {
        "dataset": item["dataset"], "variant": item["variant"], "seed": item["seed"],
        "epochs": item["epochs"], "trained_two_epochs": "Epoch 00002" in log,
        "validation_ran": "Val Acc" in log,
        "checkpoint_saved": "Saved checkpoint:" in log,
        "finite_logged_train_loss": bool(losses) and all(math.isfinite(v) for v in losses),
        "pair_up_weight_norm_at_selected_checkpoint": pair_norm,
        "pair_up_bias_norm_at_selected_checkpoint": float(state["pair_up.bias"].norm()),
        "base_pair_up_remained_zero": item["variant"] != "B" or (pair_norm == 0.0 and float(state["pair_up.bias"].norm()) == 0.0),
        "adapter_updated_for_C_D": not expected_update or pair_norm > 0.0,
        "evaluate_test": False, "metrics": item["metrics"], "best_epoch": item["best_epoch"],
        "checkpoint_path": item["checkpoint_path"], "log_path": item["log_path"],
    }
    checks = ("trained_two_epochs", "validation_ran", "checkpoint_saved", "finite_logged_train_loss", "base_pair_up_remained_zero", "adapter_updated_for_C_D")
    result["status"] = "complete" if all(result[key] for key in checks) else "failed"
    return result


def _lp_command(variant, checkpoint, hydra_dir, device):
    return [
        sys.executable, "-m", "src.main", "dataset=sports-copurchase", "task=lp",
        "model=sair_mag_v0", f"model.variant={VARIANTS[variant]}", "seed=42", "num_runs=1",
        f"device={device}", "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={hydra_dir}", "task.epochs=2", "task.max_train_batches=2",
        "task.num_neighbors=[5,5,5]",
    ]


def _run_lp_smoke(manifest, device):
    import torch
    output = []
    rows = manifest.setdefault("smoke_lp_runs", [])
    for variant in ("B", "D"):
        stem = f"sports-copurchase_{variant}_seed42"
        checkpoint = TMP / "checkpoints" / "lp_smoke" / f"{stem}.pt"
        hydra_dir = TMP / "hydra" / "lp_smoke" / stem
        log_path = TMP / "logs" / "lp_smoke" / f"{stem}.log"
        command = _lp_command(variant, checkpoint, hydra_dir, device)
        item = {
            "stem": stem, "dataset": "sports-copurchase", "variant": variant,
            "model_variant": VARIANTS[variant], "seed": 42, "epochs": 2,
            "max_train_batches": 2, "num_neighbors": [5, 5, 5],
            "positive_supervision_edge_removal": "repository global_eid protocol",
            "evaluate_test": False, "source_commit_sha": manifest["source_commit_sha"],
            "checkpoint_path": str(checkpoint), "log_path": str(log_path),
            "command": command, "status": "running", "started_at": now(),
        }
        rows[:] = [row for row in rows if row.get("stem") != stem]
        rows.append(item)
        atomic_json(RESEARCH / "run_manifest.json", manifest)
        code = _launch(command, log_path)
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        removed = [int(v) for v in re.findall(r"Positive Message Edges Removed (\d+)", log)]
        losses = [float(v) for v in re.findall(r"Train Loss ([0-9.eE+-]+)", log)]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.is_file() else {}
        state = payload.get("model_state", {})
        item.update({
            "return_code": code,
            "link_neighbor_loader": "Loader: LinkNeighborLoader" in log,
            "sampler_fanouts_observed": "Train neighbor sampling fanouts: [5, 5, 5]" in log,
            "positive_message_edges_removed_per_epoch": removed,
            "positive_edge_removal_observed": any(v > 0 for v in removed),
            "finite_sampled_batch_train_loss": bool(losses) and all(math.isfinite(v) for v in losses),
            "forward_backward_validation": code == 0 and "Val MRR" in log,
            "checkpoint_saved": checkpoint.is_file() and "Saved checkpoint:" in log,
            "pair_up_weight_norm": float(state["pair_up.weight"].norm()) if "pair_up.weight" in state else None,
            "d_response_readout_selected": variant != "D" or VARIANTS[variant] == "decoupled",
            "test_evaluated": False, "finished_at": now(),
        })
        required = ("link_neighbor_loader", "sampler_fanouts_observed", "positive_edge_removal_observed", "finite_sampled_batch_train_loss", "forward_backward_validation", "checkpoint_saved", "d_response_readout_selected")
        ok = code == 0 and len(removed) >= 2 and all(item[k] for k in required)
        item["status"] = "complete" if ok else "failed"
        atomic_json(RESEARCH / "run_manifest.json", manifest)
        if not ok:
            raise RuntimeError(f"LP smoke failed for {variant}; inspect {log_path}")
        output.append(item)
    return output


def _run_smoke(manifest, device):
    smoke_path = RESEARCH / "smoke_status.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke.update({"source_commit_sha": manifest["source_commit_sha"], "device": device, "status": "running", "started_at": now()})
    atomic_json(smoke_path, smoke)
    nc = []
    for variant in ("B", "C", "D"):
        item = _run_nc_one(manifest, phase="smoke", dataset="Movies", variant=variant, seed=42, device=device, epochs=2)
        checked = _inspect_nc_smoke(item)
        nc.append(checked)
        smoke["nc_runs"] = [r for r in smoke.get("nc_runs", []) if r.get("variant") != variant] + [checked]
        atomic_json(smoke_path, smoke)
        if checked["status"] != "complete":
            raise RuntimeError(f"NC smoke checks failed for {variant}: {checked}")
    lp = _run_lp_smoke(manifest, device)
    smoke.update({
        "nc_runs": nc, "lp_runs": lp,
        "initial_function_equality": "verified by test_sair_mag_v0.py for eval and RNG-reset train mode",
        "lp_protocol_checks": {"loader": "LinkNeighborLoader", "fanout": [5, 5, 5], "positive_edge_removal": "global_eid positive-message-edge masking", "validation_inference": "Val MRR", "test_evaluated": False},
        "status": "complete", "finished_at": now(),
    })
    atomic_json(smoke_path, smoke)
    manifest.update(smoke_status="complete", status="smoke_complete")
    atomic_json(RESEARCH / "run_manifest.json", manifest)


def _run_formal(manifest, device):
    smoke = json.loads((RESEARCH / "smoke_status.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "complete":
        raise RuntimeError("both NC and LP smoke must pass before formal runs")
    for dataset in DATASETS:
        for variant in ("B", "C", "D"):
            for seed in SEEDS:
                _run_nc_one(manifest, phase="formal", dataset=dataset, variant=variant, seed=seed, device=device, epochs=None)
    complete = [r for r in manifest["formal_runs"] if r.get("status") == "complete"]
    if len(complete) != 27:
        raise RuntimeError(f"expected 27 formal runs, found {len(complete)}")
    manifest.update(formal_status="complete", status="formal_complete", finished_at=now())
    atomic_json(RESEARCH / "run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest_path = RESEARCH / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source(manifest)
    if args.phase in ("smoke", "all"):
        _run_smoke(manifest, args.device)
    if args.phase in ("formal", "all"):
        _run_formal(manifest, args.device)
    print(f"SAIR-G0 {args.phase} finished: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
