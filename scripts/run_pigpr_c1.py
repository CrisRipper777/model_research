#!/usr/bin/env python3
"""Run the frozen PIGPR-C1 validation-only attribution campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "pigpr_c1_direct_gpr_attribution"
DATA = RESEARCH / "data"
BASE_SHA = "66017c7a83af127d842e58238cd88cc2e0aa08ff"
BRANCH = "exp/pigpr_c1_direct_gpr_attribution"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = {
    "RU": "raw_uniform_protected",
    "RUD": "raw_uniform_direct",
    "R0UD": "raw_uniform0_direct",
    "RFD": "raw_fixed_prior_direct",
    "RGD": "raw_gpr_direct",
    "AGD": "anchored_gpr_direct",
}
SEEDS = (42, 43, 44)
TMP_ROOT = Path("/tmp/pigpr_c1_direct_gpr_attribution")
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
    if git_value("merge-base", BASE_SHA, "HEAD") != BASE_SHA:
        raise RuntimeError(f"HEAD is not descended from required base {BASE_SHA}")


def split_path(dataset: str) -> str:
    if dataset in {"Movies", "Grocery"}:
        return str(SPLIT_ROOT / f"{dataset}_nc_seed42_train0.6_val0.2.pt")
    return "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt"


def command_for(
    *, dataset: str, variant: str, seed: int, checkpoint: Path,
    metrics_path: Path | None, hydra_dir: Path, device: str,
    epochs: int | None = None, max_train_batches: int | None = None,
) -> list[str]:
    is_lp = dataset == "sports-copurchase"
    command = [
        sys.executable, "-m", "src.main", f"dataset={dataset}",
        "task=lp" if is_lp else "task=nc", "model=pigpr_mag_v1",
        f"model.variant={VARIANTS[variant]}", f"seed={seed}", "num_runs=1",
        f"device={device}", "task.evaluate_test=false",
        f"task.save_ckpt_path={checkpoint}", f"hydra.run.dir={hydra_dir}",
    ]
    if not is_lp:
        command.append("task.development_no_test=true")
        if dataset in {"Movies", "Grocery"}:
            command.append(f"dataset.nc_split_path={split_path(dataset)}")
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
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return int(result.returncode)


def read_single_run(metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"development_no_test missing from {metrics_path}")
    if len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"expected one run in {metrics_path}")
    item = payload["runs"][0]
    metrics = item["metrics"]
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metrics present in {metrics_path}")
    required = {"val_acc", "val_macro_f1", "val_ce"}
    if not required.issubset(metrics):
        raise RuntimeError(f"validation metrics missing from {metrics_path}")
    return {"seed": int(item["seed"]), "metrics": metrics, "metadata": item["metadata"]}


def summarize_repeats(rows: list[dict]) -> list[dict]:
    import statistics
    output = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        if len(subset) != 3:
            continue
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
            values = [float(row[metric]) for row in subset]
            output.append({
                "dataset": dataset, "variant": "AGD", "metric": metric, "n": 3,
                "mean": statistics.mean(values), "population_sd": statistics.pstdev(values),
                "max_min_range": max(values) - min(values), "minimum": min(values),
                "maximum": max(values), "variation_type": "same-seed execution variation",
            })
    return output


def run_preflight(device: str) -> int:
    validate_provenance()
    RESEARCH.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = RESEARCH / "smoke_status.json"
    old_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    old_repeat_items = {
        (item.get("dataset"), int(item.get("repeat_id", -1))): item
        for item in old_status.get("repeatability_runs", [])
    }
    status = {
        "base_sha": BASE_SHA, "branch": git_value("branch", "--show-current"),
        "started_at": now(), "device": device, "evaluate_test": False,
        "development_no_test": True,
        "fixed_splits": {dataset: split_path(dataset) for dataset in DATASETS},
        "ru_regression_gate": {"status": "running"}, "tests": {"status": "pending"},
        "repeatability_runs": [], "nc_runs": [], "lp_runs": [],
    }
    write_json(status_path, status)

    gate_log = TMP_ROOT / "logs" / "pytest_ru_regression.log"
    gate_code = launch(
        [sys.executable, "-m", "pytest", "-q", "tests/test_pigpr_mag_v1.py::test_ru_regresses_to_pigpr_v0_ru"],
        gate_log,
    )
    gate_text = gate_log.read_text(encoding="utf-8", errors="replace")
    status["ru_regression_gate"] = {
        "status": "passed" if gate_code == 0 else "failed", "return_code": gate_code,
        "log_path": str(gate_log), "evidence": gate_text[-2000:], "finished_at": now(),
    }
    write_json(status_path, status)
    if gate_code:
        status["status"] = "failed at C0 RU regression gate; formal campaign prohibited"
        write_json(status_path, status)
        return 1

    test_log = TMP_ROOT / "logs" / "pytest_all.log"
    test_code = launch([sys.executable, "-m", "pytest", "-q"], test_log)
    status["tests"] = {
        "status": "passed" if test_code == 0 else "failed", "return_code": test_code,
        "command": [sys.executable, "-m", "pytest", "-q"],
        "log_path": str(test_log), "finished_at": now(),
    }
    write_json(status_path, status)
    if test_code:
        status["status"] = "failed at repository test suite"
        write_json(status_path, status)
        return 1

    repeat_rows: list[dict] = []
    for dataset in DATASETS:
        for repeat_id in range(1, 4):
            stem = f"{dataset}_AGD_seed42_repeat{repeat_id}"
            metrics = DATA / "repeatability_runs" / f"{stem}.json"
            checkpoint = TMP_ROOT / "checkpoints" / "repeatability" / f"{stem}.pt"
            hydra, log = TMP_ROOT / "hydra" / stem, TMP_ROOT / "logs" / f"{stem}.log"
            old_item = old_repeat_items.get((dataset, repeat_id))
            if metrics.is_file() and checkpoint.is_file():
                run = read_single_run(metrics)
                item = old_item or {
                    "dataset": dataset, "variant": "AGD", "seed": 42,
                    "repeat_id": repeat_id, "metrics_path": str(metrics),
                    "checkpoint_path": str(checkpoint), "log_path": str(log),
                }
                item.update({"return_code": 0, "reused_completed_run": True,
                             "best_epoch": int(run["metadata"]["best_epoch"]),
                             "metrics": run["metrics"]})
                status["repeatability_runs"].append(item)
                repeat_rows.append({
                    "dataset": dataset, "variant": "AGD", "seed": 42, "repeat_id": repeat_id,
                    "val_accuracy": float(run["metrics"]["val_acc"]),
                    "val_macro_f1": float(run["metrics"]["val_macro_f1"]),
                    "val_ce": float(run["metrics"]["val_ce"]),
                    "best_epoch": int(run["metadata"]["best_epoch"]),
                    "variation_type": "same-seed execution variation", "metrics_path": str(metrics),
                })
                write_csv(DATA / "repeatability_by_run.csv", repeat_rows)
                write_csv(DATA / "repeatability_summary.csv", summarize_repeats(repeat_rows))
                write_json(status_path, status)
                continue
            command = command_for(dataset=dataset, variant="AGD", seed=42, checkpoint=checkpoint,
                                  metrics_path=metrics, hydra_dir=hydra, device=device)
            code = launch(command, log)
            item = {"dataset": dataset, "variant": "AGD", "seed": 42, "repeat_id": repeat_id,
                    "return_code": code, "metrics_path": str(metrics), "checkpoint_path": str(checkpoint),
                    "log_path": str(log), "command": command, "finished_at": now()}
            if code == 0 and metrics.is_file() and checkpoint.is_file():
                run = read_single_run(metrics)
                item["best_epoch"] = int(run["metadata"]["best_epoch"])
                item["metrics"] = run["metrics"]
                repeat_rows.append({
                    "dataset": dataset, "variant": "AGD", "seed": 42, "repeat_id": repeat_id,
                    "val_accuracy": float(run["metrics"]["val_acc"]),
                    "val_macro_f1": float(run["metrics"]["val_macro_f1"]),
                    "val_ce": float(run["metrics"]["val_ce"]),
                    "best_epoch": int(run["metadata"]["best_epoch"]),
                    "variation_type": "same-seed execution variation", "metrics_path": str(metrics),
                })
            status["repeatability_runs"].append(item)
            write_csv(DATA / "repeatability_by_run.csv", repeat_rows)
            write_csv(DATA / "repeatability_summary.csv", summarize_repeats(repeat_rows))
            write_json(status_path, status)
            if code or not metrics.is_file() or not checkpoint.is_file():
                status["status"] = f"failed during repeatability at {stem}"
                write_json(status_path, status)
                return 1
    if len(repeat_rows) != 9:
        raise RuntimeError(f"expected nine AGD repeats, found {len(repeat_rows)}")
    status["repeatability_status"] = "passed"

    for variant in VARIANTS:
        stem = f"Movies_{variant}_seed42_smoke"
        metrics = DATA / "smoke_nc" / f"{variant}.json"
        checkpoint = TMP_ROOT / "checkpoints" / "smoke" / f"{stem}.pt"
        hydra, log = TMP_ROOT / "hydra" / stem, TMP_ROOT / "logs" / f"{stem}.log"
        command = command_for(dataset="Movies", variant=variant, seed=42, checkpoint=checkpoint,
                              metrics_path=metrics, hydra_dir=hydra, device=device, epochs=2)
        code = launch(command, log)
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        item = {
            "dataset": "Movies", "variant": variant, "seed": 42, "epochs": 2,
            "return_code": code, "trained": code == 0 and "Epoch 00002" in text,
            "validated": "Val Acc" in text,
            "checkpoint_saved": "Saved checkpoint:" in text and checkpoint.is_file(),
            "metrics_path": str(metrics), "checkpoint_path": str(checkpoint),
            "log_path": str(log), "command": command, "finished_at": now(),
        }
        status["nc_runs"].append(item)
        write_json(status_path, status)
        if code or not item["trained"] or not item["validated"] or not item["checkpoint_saved"]:
            status["status"] = f"failed during NC smoke at {variant}"
            write_json(status_path, status)
            return 1

    for variant in ("RGD", "AGD"):
        stem = f"sports_{variant}_seed42_lp_smoke"
        checkpoint = TMP_ROOT / "checkpoints" / "smoke" / f"{stem}.pt"
        hydra, log = TMP_ROOT / "hydra" / stem, TMP_ROOT / "logs" / f"{stem}.log"
        command = command_for(dataset="sports-copurchase", variant=variant, seed=42,
                              checkpoint=checkpoint, metrics_path=None, hydra_dir=hydra,
                              device=device, epochs=2, max_train_batches=2)
        code = launch(command, log)
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        fanout_match = re.search(r"Train neighbor sampling fanouts: (\[[^\]]+\])", text)
        removed = [int(v) for v in re.findall(r"Positive Message Edges Removed (\d+)", text)]
        observed = {
            "link_neighbor_loader": "Loader: LinkNeighborLoader" in text,
            "sampler_fanouts": json.loads(fanout_match.group(1)) if fanout_match else None,
            "positive_message_edges_removed_by_epoch": removed,
            "positive_supervision_removal_observed": bool(removed) and all(v > 0 for v in removed),
            "forward_backward_completed": code == 0 and "Train Loss" in text,
            "validation_inference_completed": "Eval node embeddings:" in text,
            "checkpoint_saved": "Saved checkpoint:" in text and checkpoint.is_file(),
            "test_evaluation_disabled": "task.evaluate_test=false" in " ".join(command),
        }
        item = {"dataset": "sports-copurchase", "variant": variant, "seed": 42,
                "epochs": 2, "max_train_batches": 2, "return_code": code,
                "observed": observed, "checkpoint_path": str(checkpoint),
                "log_path": str(log), "command": command, "finished_at": now()}
        status["lp_runs"].append(item)
        write_json(status_path, status)
        if code or not all((observed["link_neighbor_loader"], observed["sampler_fanouts"] == [5, 5, 5],
                            observed["positive_supervision_removal_observed"],
                            observed["forward_backward_completed"],
                            observed["validation_inference_completed"],
                            observed["checkpoint_saved"])):
            status["status"] = f"failed during LP smoke at {variant}"
            write_json(status_path, status)
            return 1

    status["status"] = "passed"
    status["finished_at"] = now()
    write_json(status_path, status)
    return 0


def run_formal(device: str, force: bool) -> int:
    validate_provenance()
    status_path = RESEARCH / "smoke_status.json"
    if not status_path.is_file():
        raise RuntimeError("run tests, repeatability and smoke checks first")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "passed" or status.get("repeatability_status") != "passed":
        raise RuntimeError("tests, AGD repeats, NC smoke, and LP smoke must pass first")

    manifest_path = RESEARCH / "run_manifest.json"
    existing: dict[tuple[str, str, int], dict] = {}
    if manifest_path.is_file() and not force:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("base_sha") != BASE_SHA:
            raise RuntimeError("existing campaign manifest has a different base SHA")
        for run in old.get("runs", []):
            key = (run.get("dataset"), run.get("variant"), int(run.get("seed", -1)))
            if run.get("status") == "complete" and Path(run["checkpoint_path"]).is_file() and Path(run["run_metrics_path"]).is_file():
                existing[key] = run

    manifest = {
        "experiment": "PIGPR-C1 Direct GPR Attribution Audit", "base_sha": BASE_SHA,
        "branch": BRANCH, "head_at_start": git_value("rev-parse", "HEAD"),
        "datasets": list(DATASETS), "fixed_dataset_splits": status["fixed_splits"],
        "variants": VARIANTS, "model_training_seeds": list(SEEDS),
        "num_runs_per_process": 1, "expected_runs": 54, "paired_seed_design": True,
        "split_count_per_dataset": 1, "evaluate_test": False, "development_no_test": True,
        "optimizer_and_hyperparameters": "unchanged project full-graph NC protocol",
        "started_at": now(), "status": "running", "runs": [],
    }
    write_json(manifest_path, manifest)
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                key = (dataset, variant, seed)
                if key in existing:
                    manifest["runs"].append(existing[key])
                    write_json(manifest_path, manifest)
                    continue
                stem = f"{dataset}_{variant}_seed{seed}"
                metrics = DATA / "runs" / dataset / f"{variant}_seed{seed}.json"
                checkpoint = TMP_ROOT / "checkpoints" / "formal" / f"{stem}.pt"
                hydra, log = TMP_ROOT / "hydra" / f"formal_{stem}", TMP_ROOT / "logs" / f"formal_{stem}.log"
                command = command_for(dataset=dataset, variant=variant, seed=seed,
                                      checkpoint=checkpoint, metrics_path=metrics,
                                      hydra_dir=hydra, device=device)
                item = {
                    "dataset": dataset, "variant": variant, "model_variant": VARIANTS[variant],
                    "seed": seed, "split_path": split_path(dataset), "evaluate_test": False,
                    "development_no_test": True, "status": "running", "command": command,
                    "run_metrics_path": str(metrics), "checkpoint_path": str(checkpoint),
                    "log_path": str(log), "started_at": now(),
                }
                manifest["runs"].append(item)
                write_json(manifest_path, manifest)
                code = launch(command, log)
                if code == 0 and metrics.is_file() and checkpoint.is_file():
                    run = read_single_run(metrics)
                    item.update({"status": "complete", "return_code": code,
                                 "best_epoch": int(run["metadata"]["best_epoch"]),
                                 "metrics": run["metrics"], "finished_at": now()})
                else:
                    item.update({"status": "failed", "return_code": code, "finished_at": now()})
                    manifest["status"] = f"failed at {dataset}/{variant}/seed{seed}"
                    manifest["finished_at"] = now()
                    write_json(manifest_path, manifest)
                    return 1
                write_json(manifest_path, manifest)
                completed = sum(r.get("status") == "complete" for r in manifest["runs"])
                print(f"COMPLETE {completed}/54 {dataset}/{variant}/seed{seed} best_epoch={item['best_epoch']}", flush=True)

    completed = [r for r in manifest["runs"] if r.get("status") == "complete"]
    expected = {(d, v, s) for d in DATASETS for v in VARIANTS for s in SEEDS}
    keys = {(r["dataset"], r["variant"], int(r["seed"])) for r in completed}
    if len(completed) != 54 or keys != expected:
        raise RuntimeError(f"expected 54 unique runs, found {len(completed)}")
    for run in completed:
        if run.get("evaluate_test") is not False or run.get("development_no_test") is not True:
            raise RuntimeError("test guard metadata failed")
        if any(k.startswith("test_") for k in run["metrics"]):
            raise RuntimeError("test metrics present in formal run records")
    manifest["status"] = "complete"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)

    command = [sys.executable, str(ROOT / "scripts" / "analyze_pigpr_c1.py"), "--device", device]
    analyzer_log = TMP_ROOT / "logs" / "analyze.log"
    code = launch(command, analyzer_log)
    manifest["analysis_status"] = "passed" if code == 0 else "failed"
    manifest["analysis_log"] = str(analyzer_log)
    write_json(manifest_path, manifest)
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "formal", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="rerun all formal cells")
    args = parser.parse_args()
    if args.phase in {"preflight", "all"}:
        code = run_preflight(args.device)
        if code or args.phase == "preflight":
            return code
    return run_formal(args.device, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
