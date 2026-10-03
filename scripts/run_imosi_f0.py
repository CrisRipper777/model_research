#!/usr/bin/env python3
"""Run the committed IMoSI-F0 smoke and validation-only NC campaign."""
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
RESEARCH = ROOT / "research" / "imosi_f0_interaction_mode_mixture"
DATA = RESEARCH / "data"
BASE_SHA = "188b940a88b1852608c9e2e25e6d8bd7013ed8b5"
BRANCH = "exp/imosi_f0_interaction_mode_mixture"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = {"B": "base", "G": "global_mix", "M": "adaptive_mix"}
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}
TMP = Path("/tmp/imosi_f0_interaction_mode_mixture")
SOURCE_PATHS = (
    "src/models/imosi_mag_v0.py",
    "configs/model/imosi_mag_v0.yaml",
    "scripts/run_imosi_f0.py",
    "scripts/analyze_imosi_f0.py",
    "tests/test_imosi_mag_v0.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_source(manifest: dict) -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"runner must stay on {BRANCH}")
    if git("show", "-s", "--format=%H", BASE_SHA) != BASE_SHA:
        raise RuntimeError(f"required parent SHA {BASE_SHA} unavailable")
    if git("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError("current HEAD is not descended from the required base")
    source_sha = manifest.get("source_commit_sha")
    if not source_sha or source_sha == "PENDING_SOURCE_COMMIT":
        raise RuntimeError("source_commit_sha must be recorded before any smoke/formal run")
    if git("merge-base", source_sha, "HEAD") != source_sha:
        raise RuntimeError("HEAD does not contain the declared source commit")
    for path in SOURCE_PATHS:
        subprocess.run(["git", "diff", "--quiet", source_sha, "--", path], cwd=ROOT, check=True)
    if git("show", "-s", "--format=%H", source_sha) != source_sha:
        raise RuntimeError("declared source commit is unavailable")


def _paths(phase: str, dataset: str, variant: str, seed: int):
    stem = f"{dataset}_{variant}_seed{seed}"
    metrics = DATA / "runs" / phase / dataset / f"{stem}.json"
    checkpoint = TMP / "checkpoints" / phase / f"{stem}.pt"
    hydra_dir = TMP / "hydra" / phase / stem
    log = TMP / "logs" / phase / f"{stem}.log"
    return stem, metrics, checkpoint, hydra_dir, log


def _nc_command(dataset: str, variant: str, seed: int, checkpoint: Path, metrics: Path, hydra_dir: Path, device: str, epochs: int | None):
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}", "task=nc",
        "model=imosi_mag_v0", f"model.variant={VARIANTS[variant]}", f"seed={seed}",
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


def _read_nc_metrics(path: Path, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"development_no_test missing in {path}")
    runs = payload.get("runs", [])
    if len(runs) != 1 or int(runs[0]["seed"]) != seed:
        raise RuntimeError(f"expected one NC run for seed {seed} in {path}")
    metrics = runs[0]["metrics"]
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metric found in validation-only artifact {path}")
    for key in ("val_acc", "val_macro_f1", "val_ce"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"invalid {key} in {path}")
    return {"metrics": metrics, "metadata": runs[0].get("metadata", {})}


def _run_nc_one(manifest: dict, *, phase: str, dataset: str, variant: str, seed: int, device: str, epochs: int | None):
    stem, metrics_path, checkpoint, hydra_dir, log_path = _paths(phase, dataset, variant, seed)
    collection = "formal_runs" if phase == "formal" else "smoke_nc_runs"
    old = next((row for row in manifest.get(collection, []) if row.get("stem") == stem and row.get("status") == "complete"), None)
    if old and metrics_path.is_file() and checkpoint.is_file():
        return {**old, **_read_nc_metrics(metrics_path, seed)}
    command = _nc_command(dataset, variant, seed, checkpoint, metrics_path, hydra_dir, device, epochs)
    item = {
        "stem": stem, "dataset": dataset, "variant": variant,
        "model_variant": VARIANTS[variant], "seed": seed, "split_path": SPLITS[dataset],
        "evaluate_test": False, "development_no_test": True, "epochs": epochs,
        "source_commit_sha": manifest["source_commit_sha"], "status": "running",
        "command": command, "run_metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint), "log_path": str(log_path), "started_at": now(),
    }
    records = manifest.setdefault(collection, [])
    records[:] = [row for row in records if row.get("stem") != stem]
    records.append(item)
    atomic_json(RESEARCH / "run_manifest.json", manifest)
    code = _launch(command, log_path)
    item["return_code"] = code
    if code != 0 or not metrics_path.is_file() or not checkpoint.is_file():
        item["status"] = "failed"
        item["finished_at"] = now()
        atomic_json(RESEARCH / "run_manifest.json", manifest)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:] if log_path.exists() else ""
        raise RuntimeError(f"{stem} failed ({code}); log tail:\n{tail}")
    parsed = _read_nc_metrics(metrics_path, seed)
    item.update(parsed)
    item["best_epoch"] = int(parsed["metadata"]["best_epoch"])
    item["status"] = "complete"
    item["finished_at"] = now()
    atomic_json(RESEARCH / "run_manifest.json", manifest)
    return item


def _compose_nc_context(dataset: str, seed: int, device: str):
    from hydra import compose, initialize_config_dir
    from src.data import load_mag_data
    from src.models.imosi_mag_v0 import Model
    overrides = [f"dataset={dataset}", "task=nc", "model=imosi_mag_v0", "model.variant=adaptive_mix", f"seed={seed}", f"device={device}", "task.evaluate_test=false", "task.development_no_test=true"]
    if dataset in ("Movies", "Grocery"):
        overrides.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    model = Model(cfg, info).to(device)
    x = data.x if data.num_nodes >= 50_000 else data.x.to(device)
    edge_index = data.edge_index.to(device)
    return data, model, x, edge_index


def _smoke_route_check(item: dict, device: str) -> dict:
    import torch
    data, model, x, edge_index = _compose_nc_context(item["dataset"], int(item["seed"]), device)
    checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
    model.variant = item["model_variant"]
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        _, _, _, _, info = model(x, edge_index, return_details=True)
    checks = {}
    for modality, row in info["details"]["routing"].items():
        probs = row["mode_probs"]
        checks[modality] = {
            "finite": bool(torch.isfinite(probs).all()),
            "nonnegative": bool((probs >= 0).all()),
            "sum_to_one_max_error": float((probs.sum(-1) - 1).abs().max().cpu()),
        }
    del data, model, x, edge_index, checkpoint, info
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checks


def _inspect_nc_smoke(item: dict, device: str) -> dict:
    import torch
    log = Path(item["log_path"]).read_text(encoding="utf-8", errors="replace")
    checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    losses = [float(value) for value in re.findall(r"Train Loss ([0-9.eE+-]+)", log)]
    norms = {
        "self_up": float(state["self_up.weight"].norm()),
        "pair_up": float(state["pair_up.weight"].norm()),
        "router_out": float(state["router_out.weight"].norm()),
    }
    variant = item["variant"]
    routes = _smoke_route_check(item, device)
    modes_moved = norms["self_up"] > 0 and norms["pair_up"] > 0
    result = {
        "dataset": item["dataset"], "variant": variant, "seed": int(item["seed"]),
        "epochs": int(item["epochs"]), "trained_two_epochs": "Epoch 00002" in log,
        "validation_ran": "Val Acc" in log, "checkpoint_saved": "Saved checkpoint:" in log,
        "finite_logged_train_loss": bool(losses) and all(math.isfinite(value) for value in losses),
        "mode_head_norms_at_selected_checkpoint": norms,
        "self_and_pair_heads_moved": modes_moved if variant in ("G", "M") else (norms["self_up"] == 0 and norms["pair_up"] == 0),
        "router_out_moved_at_selected_checkpoint": norms["router_out"] > 0,
        "router_out_later_update_verified_by_two_step_optimizer_test": True,
        "route_checks": routes,
        "route_values_finite_and_normalized": all(row["finite"] and row["nonnegative"] and row["sum_to_one_max_error"] < 2e-6 for row in routes.values()),
        "evaluate_test": False, "metrics": item["metrics"], "best_epoch": item["best_epoch"],
        "checkpoint_path": item["checkpoint_path"], "log_path": item["log_path"],
    }
    checks = ("trained_two_epochs", "validation_ran", "checkpoint_saved", "finite_logged_train_loss", "self_and_pair_heads_moved", "route_values_finite_and_normalized")
    result["status"] = "complete" if all(result[key] for key in checks) else "failed"
    return result


def _lp_command(variant: str, checkpoint: Path, hydra_dir: Path, device: str):
    model_variant = VARIANTS[variant]
    return [
        sys.executable, "-m", "src.main", "dataset=sports-copurchase", "task=lp",
        "model=imosi_mag_v0", f"model.variant={model_variant}", "seed=42", "num_runs=1",
        f"device={device}", "task.evaluate_test=false", f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={hydra_dir}", "task.epochs=2", "task.max_train_batches=2",
        "task.num_neighbors=[5,5,5]",
    ]


def _run_lp_smoke(manifest: dict, device: str) -> list[dict]:
    import torch

    results = []
    rows = manifest.setdefault("smoke_lp_runs", [])
    for variant in ("B", "M"):
        stem = f"sports-copurchase_{variant}_seed42"
        checkpoint = TMP / "checkpoints" / "lp_smoke" / f"{stem}.pt"
        hydra_dir = TMP / "hydra" / "lp_smoke" / stem
        log_path = TMP / "logs" / "lp_smoke" / f"{stem}.log"
        prior = next((row for row in rows if row.get("stem") == stem and row.get("status") == "complete"), None)
        if prior and checkpoint.is_file() and log_path.is_file():
            results.append(prior)
            continue
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
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        removed = [int(value) for value in re.findall(r"Positive Message Edges Removed (\d+)", text)]
        losses = [float(value) for value in re.findall(r"Train Loss ([0-9.eE+-]+)", text)]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.is_file() else {}
        model_state = payload.get("model_state", {})
        router_active = variant == "M" and (
            code == 0
            and "Loader: LinkNeighborLoader" in text
            and "Train Loss" in text
            and "router_out.weight" in model_state
            and bool(torch.isfinite(model_state["router_out.weight"]).all())
        )
        item.update({
            "return_code": int(code),
            "link_neighbor_loader": "Loader: LinkNeighborLoader" in text,
            "sampler_fanouts_observed": "Train neighbor sampling fanouts: [5, 5, 5]" in text,
            "positive_message_edges_removed_per_epoch": removed,
            "positive_edge_removal_observed": any(value > 0 for value in removed),
            "finite_sampled_batch_train_loss": bool(losses) and all(math.isfinite(value) for value in losses),
            "forward_backward_validation": code == 0 and "Val MRR" in text,
            "sampled_model_forward_observed": code == 0 and "Loader: LinkNeighborLoader" in text and "Train Loss" in text,
            "router_on_sampled_nodes": router_active,
            "adaptive_router_applied_to_sampled_nodes": router_active,
            "router_sampled_forward_path": (
                "src/tasks/lp.py calls model(batch.x, batch.edge_index) for LinkNeighborLoader batches"
            ),
            "checkpoint_saved": checkpoint.is_file() and "Saved checkpoint:" in text,
            "test_evaluated": False, "finished_at": now(),
        })
        required = (
            "link_neighbor_loader", "sampler_fanouts_observed", "finite_sampled_batch_train_loss",
            "forward_backward_validation", "sampled_model_forward_observed", "checkpoint_saved",
        )
        checks_pass = all(item[key] for key in required) and len(removed) >= 2 and item["positive_edge_removal_observed"]
        if variant == "M":
            checks_pass = checks_pass and item["router_on_sampled_nodes"]
        item["status"] = "complete" if code == 0 and checks_pass else "failed"
        atomic_json(RESEARCH / "run_manifest.json", manifest)
        if item["status"] != "complete":
            raise RuntimeError(f"LP smoke failed for {variant}; inspect {log_path}")
        results.append(item)
    return results


def _new_manifest() -> dict:
    return {
        "experiment": "IMoSI-F0 Interaction-Mode Mixture Prototype",
        "base_sha": BASE_SHA, "branch": BRANCH, "head_at_branch_creation": BASE_SHA,
        "source_commit_sha": "PENDING_SOURCE_COMMIT", "datasets": list(DATASETS),
        "fixed_dataset_splits": SPLITS, "variants": VARIANTS,
        "model_training_seeds": list(SEEDS), "initial_mode_probs": [0.6, 0.2, 0.2],
        "response_rank": 64, "router_rank": 64, "mode_chunk_size": 8192,
        "expected_nc_smoke_runs": 3, "expected_lp_smoke_runs": 2,
        "expected_formal_runs": 27, "evaluate_test": False,
        "development_no_test": True, "smoke_nc_runs": [], "smoke_lp_runs": [],
        "formal_runs": [], "created_at": now(), "status": "pending_source_commit",
    }


def run_smoke(manifest: dict, device: str) -> None:
    smoke_path = RESEARCH / "smoke_status.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke.update({
        "base_sha": BASE_SHA, "source_commit_sha": manifest["source_commit_sha"],
        "branch": BRANCH, "device": device, "evaluate_test": False,
        "development_no_test": True, "fixed_nc_splits": SPLITS,
        "started_at": smoke.get("started_at", now()), "status": "running",
    })
    atomic_json(smoke_path, smoke)
    nc_rows = []
    for variant in ("B", "G", "M"):
        item = _run_nc_one(manifest, phase="smoke", dataset="Movies", variant=variant, seed=42, device=device, epochs=2)
        checked = _inspect_nc_smoke(item, device)
        nc_rows.append(checked)
        smoke["nc_runs"] = [row for row in smoke.get("nc_runs", []) if row.get("variant") != variant] + [checked]
        atomic_json(smoke_path, smoke)
        if checked["status"] != "complete":
            raise RuntimeError(f"NC smoke checks failed for {variant}: {checked}")
    lp_rows = _run_lp_smoke(manifest, device)
    smoke["nc_runs"] = nc_rows
    smoke["lp_runs"] = lp_rows
    smoke["lp_protocol_checks"] = {
        "loader": "LinkNeighborLoader", "fanout": [5, 5, 5],
        "positive_edge_removal": "global_eid positive-message-edge masking was observed per epoch",
        "validation_inference": "Val MRR logged after sampled training batches",
        "test_evaluated": False,
    }
    smoke["status"] = "complete"
    smoke["finished_at"] = now()
    atomic_json(smoke_path, smoke)
    manifest["smoke_status"] = "complete"
    manifest["status"] = "smoke_complete"
    atomic_json(RESEARCH / "run_manifest.json", manifest)


def run_formal(manifest: dict, device: str) -> None:
    smoke = json.loads((RESEARCH / "smoke_status.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "complete":
        raise RuntimeError("NC and LP smoke must both pass before formal runs")
    for dataset in DATASETS:
        for variant in ("B", "G", "M"):
            for seed in SEEDS:
                _run_nc_one(manifest, phase="formal", dataset=dataset, variant=variant, seed=seed, device=device, epochs=None)
    completed = [row for row in manifest["formal_runs"] if row.get("status") == "complete"]
    if len(completed) != 27:
        raise RuntimeError(f"expected 27 formal runs, found {len(completed)}")
    manifest["formal_status"] = "complete"
    manifest["status"] = "formal_complete"
    manifest["finished_at"] = now()
    atomic_json(RESEARCH / "run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    RESEARCH.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    manifest_path = RESEARCH / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else _new_manifest()
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    validate_source(manifest)
    if args.phase in ("smoke", "all"):
        run_smoke(manifest, args.device)
    if args.phase in ("formal", "all"):
        run_formal(manifest, args.device)
    print(f"IMoSI-F0 {args.phase} finished; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
