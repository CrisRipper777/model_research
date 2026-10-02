#!/usr/bin/env python3
"""Run the frozen PIGPR-C0 validation-only NC and smoke protocol."""

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
RESEARCH = ROOT / "research" / "pigpr_c0_prior_anchored_backbone_audit"
DATA = RESEARCH / "data"
BASE_SHA = "2fca8b31dc85edb8c28f9aa4066318b1e8557ad3"
BRANCH = "exp/pigpr_c0_prior_anchored_backbone_audit"
DATASETS = ("Movies", "Grocery", "ele-fashion")
REPEAT_DATASETS = ("Movies", "Grocery")
VARIANTS = {
    "PO": "prior_only",
    "RU": "raw_uniform_protected",
    "AU": "anchored_uniform_protected",
    "AP": "anchored_prior_protected",
    "AGP": "anchored_gpr_protected",
    "AGD": "anchored_gpr_direct",
}
SEEDS = (42, 43, 44)
TMP_ROOT = Path("/tmp/pigpr_c0_prior_anchored_backbone_audit")
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


def fixed_split_override(dataset: str) -> str | None:
    if dataset in {"Movies", "Grocery"}:
        return str(SPLIT_ROOT / f"{dataset}_nc_seed42_train0.6_val0.2.pt")
    return None


def command_for(
    *,
    dataset: str,
    variant: str,
    seed: int,
    checkpoint: Path,
    metrics_path: Path | None,
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
        "model=pigpr_mag_v0",
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


def repeatability_summary(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in REPEAT_DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        if len(subset) != 3:
            continue
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
            values = [float(row[metric]) for row in subset]
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "population_sd": statistics.pstdev(values),
                    "max_min_range": max(values) - min(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "variation_type": "same-seed execution variation",
                }
            )
    return output


def run_preflight(device: str) -> int:
    validate_provenance()
    RESEARCH.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = RESEARCH / "smoke_status.json"
    status = {
        "base_sha": BASE_SHA,
        "branch": git_value("branch", "--show-current"),
        "started_at": now(),
        "device": device,
        "evaluate_test": False,
        "development_no_test": True,
        "fixed_splits": {
            "Movies": fixed_split_override("Movies"),
            "Grocery": fixed_split_override("Grocery"),
            "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
        },
        "ru_regression_gate": {"status": "running"},
        "tests": {"status": "pending"},
        "repeatability_runs": [],
        "nc_runs": [],
        "lp_runs": [],
    }
    write_json(status_path, status)

    gate_log = TMP_ROOT / "logs" / "pytest_ru_regression.log"
    gate_code = launch(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_pigpr_mag_v0.py::test_ru_matches_previous_spgpr_uniform_forward_with_mapped_weights",
        ],
        gate_log,
    )
    gate_text = gate_log.read_text(encoding="utf-8", errors="replace")
    status["ru_regression_gate"] = {
        "status": "passed" if gate_code == 0 else "failed",
        "return_code": gate_code,
        "log_path": str(gate_log),
        "evidence": gate_text[-2000:],
        "finished_at": now(),
    }
    write_json(status_path, status)
    if gate_code:
        status["status"] = "failed at RU regression gate; formal campaign prohibited"
        write_json(status_path, status)
        return 1

    test_log = TMP_ROOT / "logs" / "pytest_all.log"
    test_code = launch([sys.executable, "-m", "pytest", "-q"], test_log)
    status["tests"] = {
        "status": "passed" if test_code == 0 else "failed",
        "return_code": test_code,
        "command": [sys.executable, "-m", "pytest", "-q"],
        "log_path": str(test_log),
        "finished_at": now(),
    }
    write_json(status_path, status)
    if test_code:
        status["status"] = "failed at repository test suite"
        write_json(status_path, status)
        return 1

    repeat_rows: list[dict] = []
    for dataset in REPEAT_DATASETS:
        for repeat_id in range(1, 4):
            stem = f"{dataset}_RU_seed42_repeat{repeat_id}"
            metrics = DATA / "repeatability_runs" / f"{stem}.json"
            checkpoint = TMP_ROOT / "checkpoints" / "repeatability" / f"{stem}.pt"
            hydra = TMP_ROOT / "hydra" / stem
            log = TMP_ROOT / "logs" / f"{stem}.log"
            command = command_for(
                dataset=dataset,
                variant="RU",
                seed=42,
                checkpoint=checkpoint,
                metrics_path=metrics,
                hydra_dir=hydra,
                device=device,
            )
            code = launch(command, log)
            item = {
                "dataset": dataset,
                "variant": "RU",
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
                run = read_single_run(metrics)
                item["best_epoch"] = int(run["metadata"]["best_epoch"])
                item["metrics"] = run["metrics"]
                repeat_rows.append(
                    {
                        "dataset": dataset,
                        "variant": "RU",
                        "seed": 42,
                        "repeat_id": repeat_id,
                        "val_accuracy": float(run["metrics"]["val_acc"]),
                        "val_macro_f1": float(run["metrics"]["val_macro_f1"]),
                        "val_ce": float(run["metrics"]["val_ce"]),
                        "best_epoch": int(run["metadata"]["best_epoch"]),
                        "variation_type": "same-seed execution variation",
                        "metrics_path": str(metrics),
                    }
                )
            status["repeatability_runs"].append(item)
            write_csv(DATA / "repeatability_by_run.csv", repeat_rows)
            write_csv(DATA / "repeatability_summary.csv", repeatability_summary(repeat_rows))
            write_json(status_path, status)
            if code or not metrics.is_file() or not checkpoint.is_file():
                status["status"] = f"failed during repeatability at {stem}"
                write_json(status_path, status)
                return 1
    if len(repeat_rows) != 6:
        raise RuntimeError(f"expected six repeatability runs, found {len(repeat_rows)}")
    status["repeatability_status"] = "passed"

    for variant in VARIANTS:
        stem = f"Movies_{variant}_seed42_smoke"
        metrics = DATA / "smoke_nc" / f"{variant}.json"
        checkpoint = TMP_ROOT / "checkpoints" / "smoke" / f"{stem}.pt"
        hydra = TMP_ROOT / "hydra" / stem
        log = TMP_ROOT / "logs" / f"{stem}.log"
        command = command_for(
            dataset="Movies",
            variant=variant,
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
            "variant": variant,
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
            status["status"] = f"failed during NC smoke at {variant}"
            write_json(status_path, status)
            return 1

    for variant in ("AGP", "AGD"):
        stem = f"sports_{variant}_seed42_lp_smoke"
        checkpoint = TMP_ROOT / "checkpoints" / "smoke" / f"{stem}.pt"
        hydra = TMP_ROOT / "hydra" / stem
        log = TMP_ROOT / "logs" / f"{stem}.log"
        command = command_for(
            dataset="sports-copurchase",
            variant=variant,
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
            "test_evaluation_disabled": "task.evaluate_test=false" in " ".join(command),
        }
        item = {
            "dataset": "sports-copurchase",
            "variant": variant,
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
        raise RuntimeError("run tests, repeatability, and smoke checks first")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "passed" or status.get("repeatability_status") != "passed":
        raise RuntimeError("tests, RU regression gate, repeatability, NC smoke, and LP smoke must pass first")

    manifest_path = RESEARCH / "run_manifest.json"
    existing: dict[tuple[str, str, int], dict] = {}
    if manifest_path.is_file() and not force:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("base_sha") != BASE_SHA:
            raise RuntimeError("existing campaign manifest has a different base SHA")
        for run in old.get("runs", []):
            if run.get("status") == "complete":
                existing[(run["dataset"], run["variant"], int(run["seed"]))] = run

    manifest = {
        "experiment": "PIGPR-C0 Prior-Anchored GPR Backbone Audit",
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "head_at_start": git_value("rev-parse", "HEAD"),
        "datasets": list(DATASETS),
        "fixed_dataset_splits": status["fixed_splits"],
        "variants": VARIANTS,
        "model_training_seeds": list(SEEDS),
        "num_runs_per_process": 1,
        "expected_runs": 54,
        "paired_seed_design": True,
        "split_count_per_dataset": 1,
        "evaluate_test": False,
        "development_no_test": True,
        "optimizer_and_hyperparameters": "unchanged project full-graph NC protocol",
        "started_at": now(),
        "status": "running",
        "runs": [],
    }
    write_json(manifest_path, manifest)

    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                key = (dataset, variant, seed)
                old = existing.get(key)
                if old and Path(old["checkpoint_path"]).is_file() and Path(old["run_metrics_path"]).is_file():
                    manifest["runs"].append(old)
                    write_json(manifest_path, manifest)
                    continue
                stem = f"{dataset}_{variant}_seed{seed}"
                metrics = DATA / "runs" / dataset / f"{variant}_seed{seed}.json"
                checkpoint = TMP_ROOT / "checkpoints" / "formal" / f"{stem}.pt"
                hydra = TMP_ROOT / "hydra" / f"formal_{stem}"
                log = TMP_ROOT / "logs" / f"formal_{stem}.log"
                command = command_for(
                    dataset=dataset,
                    variant=variant,
                    seed=seed,
                    checkpoint=checkpoint,
                    metrics_path=metrics,
                    hydra_dir=hydra,
                    device=device,
                )
                item = {
                    "dataset": dataset,
                    "variant": variant,
                    "model_variant": VARIANTS[variant],
                    "seed": seed,
                    "split_path": fixed_split_override(dataset) or status["fixed_splits"][dataset],
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
                    item.update({"status": "failed", "return_code": code, "finished_at": now()})
                    manifest["status"] = f"failed at {dataset}/{variant}/seed{seed}"
                    manifest["finished_at"] = now()
                    write_json(manifest_path, manifest)
                    return 1
                write_json(manifest_path, manifest)
                print(
                    f"COMPLETE {len([r for r in manifest['runs'] if r.get('status') == 'complete'])}/54 "
                    f"{dataset}/{variant}/seed{seed} best_epoch={item['best_epoch']}",
                    flush=True,
                )

    completed = [run for run in manifest["runs"] if run.get("status") == "complete"]
    keys = {(run["dataset"], run["variant"], int(run["seed"])) for run in completed}
    expected = {
        (dataset, variant, seed)
        for dataset in DATASETS
        for variant in VARIANTS
        for seed in SEEDS
    }
    if len(completed) != 54 or keys != expected:
        raise RuntimeError(f"expected exactly 54 unique runs, found {len(completed)}")
    for run in completed:
        if run.get("evaluate_test") is not False or run.get("development_no_test") is not True:
            raise RuntimeError(f"test guard failed for {run['dataset']}/{run['variant']}")
        if any(key.startswith("test_") for key in run["metrics"]):
            raise RuntimeError(f"test metrics found for {run['dataset']}/{run['variant']}")
    manifest["status"] = "complete"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)

    analyzer_command = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_pigpr_c0.py"),
        "--device",
        device,
    ]
    diagnostic_files = (
        DATA / "propagation_diagnostics.csv",
        DATA / "semantic_drift.csv",
        DATA / "filter_diagnostics.csv",
        DATA / "intervention_by_run.csv",
    )
    if all(path.is_file() for path in diagnostic_files):
        analyzer_command.append("--reuse-diagnostics")
    analyzer_log = TMP_ROOT / "logs" / "analyze.log"
    analyzer_code = launch(analyzer_command, analyzer_log)
    if analyzer_code:
        manifest["analysis_status"] = "failed"
        manifest["analysis_log"] = str(analyzer_log)
        write_json(manifest_path, manifest)
        return analyzer_code
    manifest["analysis_status"] = "passed"
    manifest["analysis_log"] = str(analyzer_log)
    write_json(manifest_path, manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "formal", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="discard resume records and rerun all formal cells")
    args = parser.parse_args()
    if args.phase in {"preflight", "all"}:
        code = run_preflight(args.device)
        if code or args.phase == "preflight":
            return code
    return run_formal(args.device, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
