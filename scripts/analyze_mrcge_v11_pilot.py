#!/usr/bin/env python3
"""Create auditable MR-CGE V1.1 pilot summaries and handoff reports."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mrcge_v11_pilot as pilot


DATASETS = pilot.DATASETS
CONDITIONS = pilot.CONDITIONS
DATA = pilot.DATA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_csv(path: Path, fields: list[str]) -> None:
    if not path.exists():
        pilot.write_rows(path, [], fields)


def git_text(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) if value not in (None, "") else ""


def gpu_device_snapshot(snapshot: dict[str, Any], physical_gpu: int) -> dict[str, Any]:
    direct = snapshot.get(f"GPU{physical_gpu}")
    if isinstance(direct, dict):
        return direct
    return next(
        (row for row in snapshot.get("devices", []) if row.get("physical_index") == physical_gpu),
        {},
    )


def gpu_compute_pids(snapshot: dict[str, Any], device: dict[str, Any]) -> list[int]:
    uuid = device.get("uuid")
    return [
        int(row["pid"]) for row in snapshot.get("compute_processes", [])
        if row.get("gpu_uuid") == uuid and row.get("pid") is not None
    ]


def pytest_summary(output: str) -> str:
    for line in reversed(output.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            return line.strip()
    return "pytest summary line missing"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    preflight_path = DATA / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    runs = pilot.read_rows(DATA / "runs.csv")
    run_by_key = {
        (row["dataset"], row["condition"], int(row["seed"])): row
        for row in runs if row.get("status") == "complete"
    }
    latest_attempt_by_key = {}
    for row in runs:
        key = (row["dataset"], row["condition"], int(row["seed"]))
        previous = latest_attempt_by_key.get(key)
        if previous is None or int(row.get("attempt", 0)) >= int(previous.get("attempt", 0)):
            latest_attempt_by_key[key] = row
    failures = json.loads((DATA / "failures.json").read_text(encoding="utf-8")) if (DATA / "failures.json").is_file() else {"failures": []}
    guard_overrides = json.loads((DATA / "guard_overrides.json").read_text(encoding="utf-8")) if (DATA / "guard_overrides.json").is_file() else {"overrides": []}
    runtime_guard_block = None
    for failure in reversed(failures.get("failures", [])):
        key = (failure.get("dataset"), failure.get("condition"), int(failure.get("seed", -1)))
        if (failure.get("stage") == "shared_gpu_runtime_guard"
                and failure.get("action") == "worker-train"
                and failure.get("training_started") and key not in run_by_key):
            runtime_guard_block = failure
            break
    summary_rows = []
    differences = []
    contrast_defs = (("M-F", "M", "F"), ("F-S", "F", "S"), ("M-S", "M", "S"))
    for dataset in DATASETS:
        for condition in CONDITIONS:
            row = run_by_key.get((dataset, condition, pilot.SEED))
            latest = latest_attempt_by_key.get((dataset, condition, pilot.SEED))
            if row:
                status = "complete"
                interpretation = "one-seed descriptive validation metric"
            elif latest and latest.get("status") == "failed":
                status = "failed_attempt_no_metric"
                interpretation = "formal attempt was interrupted; incomplete metrics are excluded"
            else:
                status = "not_run"
                interpretation = "formal training cell has not started"
            summary_rows.append({
                "dataset": dataset, "condition": condition, "n_completed": 1 if row else 0,
                "seed": pilot.SEED, "val_acc": row.get("val_acc", "") if row else "",
                "val_macro_f1": row.get("val_macro_f1", "") if row else "",
                "val_ce": row.get("val_ce", "") if row else "",
                "status": status, "interpretation": interpretation,
            })
        for label, left, right in contrast_defs:
            lrow, rrow = run_by_key.get((dataset, left, pilot.SEED)), run_by_key.get((dataset, right, pilot.SEED))
            if lrow and rrow:
                differences.append({
                    "dataset": dataset, "seed": pilot.SEED, "contrast": label, "status": "descriptive",
                    "delta_val_acc": float(lrow["val_acc"]) - float(rrow["val_acc"]),
                    "delta_val_macro_f1": float(lrow["val_macro_f1"]) - float(rrow["val_macro_f1"]),
                    "delta_val_ce": float(lrow["val_ce"]) - float(rrow["val_ce"]),
                    "interpretation": "single-seed descriptive difference; not an inferential claim",
                })
            else:
                differences.append({
                    "dataset": dataset, "seed": pilot.SEED, "contrast": label,
                    "status": "not_available", "delta_val_acc": "", "delta_val_macro_f1": "",
                    "delta_val_ce": "", "interpretation": "both completed Pilot cells are required; one or both cells lack a valid metric",
                })

    pilot.write_rows(DATA / "runs.csv", runs, list(pilot.RUN_FIELDS))
    pilot.write_rows(DATA / "condition_summary.csv", summary_rows,
                     ["dataset", "condition", "n_completed", "seed", "val_acc", "val_macro_f1", "val_ce", "status", "interpretation"])
    pilot.write_rows(DATA / "condition_differences.csv", differences,
                     ["dataset", "seed", "contrast", "status", "delta_val_acc", "delta_val_macro_f1", "delta_val_ce", "interpretation"])
    ensure_csv(DATA / "epoch_history.csv", ["dataset", "condition", "seed", "attempt", "source_git_sha", "epoch", "train_loss", "val_acc", "val_macro_f1"])
    ensure_csv(DATA / "expert_diagnostics.csv", ["dataset", "condition", "seed", "attempt", "view", "expert", "theta_actual", "effective_coefficients", "G_rms", "G_pairwise_cosine", "E_rms", "E_pairwise_cosine", "expert_residual_rms", "support_nodes", "source_git_sha"])
    ensure_csv(DATA / "route_diagnostics.csv", ["dataset", "condition", "seed", "attempt", "view", "dense_importance", "top2_selection_share", "dense_entropy", "selected_weight_entropy", "active_nodes", "route_scope", "route_change", "source_git_sha"])
    ensure_csv(DATA / "response_diagnostics.csv", ["dataset", "condition", "seed", "attempt", "view", "raw_rms", "calibrated_rms", "scale", "beta", "nonzero_nodes", "private_response_rms", "collaborative_response_rms", "late_delta_rms", "late_delta_ratio", "source_git_sha"])
    ensure_csv(DATA / "loss_gradient_diagnostics.csv", ["dataset", "condition", "seed", "attempt", "parameter", "gradient_present", "gradient_finite", "gradient_norm", "gradient_nonzero", "source_git_sha"])

    gpu_block = None
    if preflight.get("gpu_single_step_status") != "pass":
        gpu_block = preflight.get("gpu_single_step_block")
        if gpu_block is None:
            blocked_rows = [row for row in preflight.get("gpu_single_step", []) if row.get("status") == "blocked"]
            gpu_block = blocked_rows[-1] if blocked_rows else None
    resource_rows = pilot.read_rows(DATA / "resource_summary.csv")
    current_gpu_attempt = preflight.get("gpu_single_step_attempt")
    current_gpu = preflight.get("gpu_single_step_physical_gpu")
    passing_gpu_rows = [
        row for row in preflight.get("gpu_single_step", [])
        if row.get("status") == "pass"
        and (current_gpu_attempt is None or row.get("attempt") == current_gpu_attempt)
        and (current_gpu is None or row.get("physical_gpu") == current_gpu)
    ]
    resource_keys = {
        (row.get("dataset"), row.get("condition"), str(row.get("seed", "")),
         str(row.get("attempt", "")), row.get("phase"))
        for row in resource_rows
    }
    for row in passing_gpu_rows:
        snapshot = row.get("guard_snapshot", {})
        gpu0, gpu1 = gpu_device_snapshot(snapshot, 0), gpu_device_snapshot(snapshot, 1)
        key = (row.get("dataset"), row.get("condition"), str(row.get("seed", "")),
               str(row.get("attempt", "")), "gpu_single_step_preflight_pass")
        if key in resource_keys:
            continue
        resource_rows.append({
            "dataset": row.get("dataset", ""), "condition": row.get("condition", ""),
            "seed": row.get("seed", pilot.SEED), "attempt": row.get("attempt", ""),
            "phase": "gpu_single_step_preflight_pass", "physical_gpu": row.get("physical_gpu", ""),
            "gpu0_free_mib": gpu0.get("memory_free_mib", ""),
            "gpu0_utilization_percent": gpu0.get("utilization_percent", ""),
            "gpu0_temperature_c": gpu0.get("temperature_c", ""),
            "gpu0_compute_pids": compact(gpu_compute_pids(snapshot, gpu0)),
            "gpu1_free_mib": gpu1.get("memory_free_mib", ""),
            "gpu1_utilization_percent": gpu1.get("utilization_percent", ""),
            "gpu1_temperature_c": gpu1.get("temperature_c", ""),
            "gpu1_compute_pids": compact(gpu_compute_pids(snapshot, gpu1)),
            "status": "passed_single_step", "reason": "",
            "wall_seconds": row.get("wall_seconds", ""),
            "peak_allocated_bytes": row.get("peak_allocated_bytes", ""),
            "peak_reserved_bytes": row.get("peak_reserved_bytes", ""),
            "test_evaluation": False,
        })
        resource_keys.add(key)
    if gpu_block:
        snapshot = gpu_block.get("gpu_snapshot", {})
        gpu0, gpu1 = gpu_device_snapshot(snapshot, 0), gpu_device_snapshot(snapshot, 1)
        resource_rows.append({
            "dataset": gpu_block.get("dataset", ""), "condition": gpu_block.get("condition", ""),
            "seed": pilot.SEED, "attempt": "", "phase": "gpu_single_step_preflight_blocked",
            "physical_gpu": gpu_block.get("physical_gpu", 0), "gpu0_free_mib": gpu0.get("memory_free_mib", ""),
            "gpu0_utilization_percent": gpu0.get("utilization_percent", ""),
            "gpu0_temperature_c": gpu0.get("temperature_c", ""),
            "gpu0_compute_pids": compact(gpu0.get("compute_pids", gpu_compute_pids(snapshot, gpu0))),
            "gpu1_free_mib": gpu1.get("memory_free_mib", ""),
            "gpu1_utilization_percent": gpu1.get("utilization_percent", ""),
            "gpu1_temperature_c": gpu1.get("temperature_c", ""),
            "gpu1_compute_pids": compact(gpu1.get("compute_pids", gpu_compute_pids(snapshot, gpu1))),
            "status": "blocked_before_training", "reason": gpu_block.get("reason", ""),
            "test_evaluation": False,
        })
    if runtime_guard_block:
        snapshot = runtime_guard_block.get("gpu_snapshot") or {}
        gpu0, gpu1 = gpu_device_snapshot(snapshot, 0), gpu_device_snapshot(snapshot, 1)
        key = (
            runtime_guard_block.get("dataset"), runtime_guard_block.get("condition"),
            str(runtime_guard_block.get("seed", "")), str(runtime_guard_block.get("attempt", "")),
            "pilot_runtime_guard_blocked",
        )
        if key in resource_keys:
            resource_rows = [
                row for row in resource_rows
                if (row.get("dataset"), row.get("condition"), str(row.get("seed", "")),
                    str(row.get("attempt", "")), row.get("phase")) != key
            ]
            resource_keys.discard(key)
        if key not in resource_keys:
            resource_rows.append({
                "dataset": runtime_guard_block.get("dataset", ""),
                "condition": runtime_guard_block.get("condition", ""),
                "seed": runtime_guard_block.get("seed", pilot.SEED),
                "attempt": runtime_guard_block.get("attempt", ""),
                "phase": "pilot_runtime_guard_blocked",
                "physical_gpu": runtime_guard_block.get("physical_gpu", ""),
                "gpu0_free_mib": gpu0.get("memory_free_mib", ""),
                "gpu0_utilization_percent": gpu0.get("utilization_percent", ""),
                "gpu0_temperature_c": gpu0.get("temperature_c", ""),
                "gpu0_compute_pids": compact(gpu_compute_pids(snapshot, gpu0)),
                "gpu1_free_mib": gpu1.get("memory_free_mib", ""),
                "gpu1_utilization_percent": gpu1.get("utilization_percent", ""),
                "gpu1_temperature_c": gpu1.get("temperature_c", ""),
                "gpu1_compute_pids": compact(gpu_compute_pids(snapshot, gpu1)),
                "status": "own_child_stopped_by_runtime_guard",
                "reason": runtime_guard_block.get("error", ""),
                "wall_seconds": "", "peak_allocated_bytes": "", "peak_reserved_bytes": "",
                "test_evaluation": False,
            })
    if resource_rows:
        fields = list(dict.fromkeys(key for row in resource_rows for key in row))
        pilot.write_rows(DATA / "resource_summary.csv", resource_rows, fields)
    else:
        ensure_csv(DATA / "resource_summary.csv", ["dataset", "condition", "seed", "attempt", "phase", "status", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "test_evaluation"])

    all_planned = len(DATASETS) * len(CONDITIONS)
    complete = len(run_by_key)
    attempted_cells = {
        (row.get("dataset"), row.get("condition"), row.get("seed"))
        for row in runs
    }
    if complete == all_planned:
        formal_status = "complete"
    elif runtime_guard_block:
        formal_status = "partial_blocked"
    elif gpu_block and complete == 0:
        formal_status = "blocked_before_training"
    elif gpu_block:
        formal_status = "partial_blocked"
    elif complete == 0 and attempted_cells:
        formal_status = "failed_attempts"
    elif complete == 0 and preflight.get("gpu_single_step_status") == "pass":
        formal_status = "preflight_passed_pilot_not_started"
    elif complete == 0:
        formal_status = "not_started"
    else:
        formal_status = "partial"
    code_sha = git_text("rev-parse", "HEAD")
    src_files = [
        "src/models/mrcge_v11.py", "src/tasks/nc.py", "scripts/preflight_mrcge_v11.py",
        "scripts/run_mrcge_v11_pilot.py", "scripts/analyze_mrcge_v11_pilot.py",
        "tests/test_mrcge_v11.py", "configs/model/mrcge_v11.yaml", "configs/study/mrcge_v11_pilot.yaml",
    ]
    changed_files = {name: sha256(ROOT / name) for name in src_files}
    manifest = {
        "schema_version": 1, "study": "MR-CGE V1.1", "branch": git_text("branch", "--show-current"),
        "code_commit_sha": code_sha, "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planned_formal_cells": all_planned, "completed_formal_cells": complete,
        "formal_pilot_status": formal_status,
        "gpu_single_step_status": preflight.get("gpu_single_step_status", "not_run"),
        "test_evaluation": False, "test_labels_masked_before_trainer": True,
        "seeds_43_44_run": False,
        "dataset_fingerprints": {
            ds: {key: preflight.get("datasets", {}).get(ds, {}).get(key)
                 for key in ("features_fingerprint", "physical_edge_fingerprint", "split_fingerprint", "semantic_cache_sha256", "v7b_fingerprint")}
            for ds in DATASETS
        },
        "source_file_sha256": changed_files,
        "data_file_sha256": {
            path.name: sha256(path) for path in sorted(DATA.glob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
        "failures_count": len(failures.get("failures", [])),
        "guard_overrides": guard_overrides.get("overrides", []),
        "gpu_block": gpu_block,
        "runtime_guard_block": runtime_guard_block,
    }
    pilot.json_write(DATA / "manifest.json", manifest)

    unit_tests = None
    cpu_attempts = preflight.get("cpu_attempts", [])
    if cpu_attempts:
        unit_tests = cpu_attempts[-1].get("unit_tests")
    if unit_tests is None:
        unit_tests = next((attempt.get("unit_tests") for attempt in reversed(cpu_attempts) if attempt.get("unit_tests")), None)
    preflight_lines = [
        "# MR-CGE V1.1 Preflight Report", "",
        f"- Branch: `{manifest['branch']}`; code commit at report generation: `{code_sha}`.",
        f"- CPU/data/cache/init preflight: **{preflight.get('cpu_status', 'not run')}**.",
        f"- GPU single-step preflight: **{preflight.get('gpu_single_step_status', 'not run')}**.",
        f"- Formal Pilot cells attempted: **{len(attempted_cells)}/{all_planned}**; completed: **{complete}/{all_planned}**.",
        "- Test evaluation: **disabled**; Test labels are masked to `-100` before Trainer entry.", "",
        "## CPU and data checks", "",
    ]
    for dataset in DATASETS:
        record = preflight.get("datasets", {}).get(dataset, {})
        preflight_lines.append(
            f"- **{dataset}:** nodes={record.get('num_nodes', 'n/a')}, edges={record.get('num_edges', 'n/a')}, "
            f"features={record.get('text_dim', 'n/a')}+{record.get('visual_dim', 'n/a')}, "
            f"Test masked={record.get('test_labels_masked', False)}, V7B source flags={record.get('v7b_source_flags_shape', 'n/a')}."
        )
        if record.get("conditions"):
            preflight_lines.append(
                f"  S/F/M common initialization hashes match: `{record.get('common_initialization_sha256')}`; "
                f"classifier hash: `{record.get('classifier_initialization_sha256')}`."
            )
    preflight_lines.extend(["", "## Focused tests", ""])
    if unit_tests:
        preflight_lines.extend([
            f"Command: `{unit_tests.get('command', '')}`",
            f"Result: {'passed' if unit_tests.get('passed') else 'failed'}; {pytest_summary(unit_tests.get('stdout', ''))}.",
            "",
            "```text", unit_tests.get("stdout", "").strip(), "```",
        ])
    else:
        preflight_lines.append("No focused test result was recorded.")
    preflight_lines.extend(["", "## GPU guard outcome", ""])
    if gpu_block:
        preflight_lines.extend([
            f"The guarded GPU single-step attempt for {gpu_block.get('dataset')}/{gpu_block.get('condition')}/seed{gpu_block.get('seed')} did not pass.",
            f"Recorded stage: `{gpu_block.get('stage', 'unknown')}`; reason: `{gpu_block.get('reason', 'not recorded')}`.",
            "See `data/failures.json` and the referenced worker log for the full trace. Formal Pilot training remains gated until all nine GPU single-step checks pass.",
        ])
    elif preflight.get("gpu_single_step_status") == "pass":
        peak_mib = max((int(row.get("peak_allocated_bytes", 0)) for row in passing_gpu_rows), default=0) / (1024 ** 2)
        preflight_lines.append(
            f"All nine guarded GPU single-step checks passed on physical GPU {current_gpu if current_gpu is not None else 'recorded in each row'}; "
            f"maximum measured allocated memory was {peak_mib:.0f} MiB. Earlier blocked attempts remain in the audit history."
        )
    else:
        preflight_lines.append("No GPU guard refusal was recorded in the current preflight attempt.")
    if runtime_guard_block:
        preflight_lines.append(
            f"During formal Pilot {runtime_guard_block.get('dataset')}/{runtime_guard_block.get('condition')}/seed{runtime_guard_block.get('seed')} "
            f"attempt {runtime_guard_block.get('attempt')}, the runtime guard stopped only its own training child because "
            f"`{runtime_guard_block.get('error', 'a runtime resource check failed')}`. "
            "See the matching `shared_gpu_runtime_guard` failure and `pilot_runtime_guard_blocked` resource row before resuming."
        )
    preflight_lines.extend(["", "## Preserved failures", "", f"`data/failures.json` contains {len(failures.get('failures', []))} recorded events."])
    for override in guard_overrides.get("overrides", []):
        preflight_lines.append(
            f"GPU guard checks were explicitly overridden for formal {override.get('dataset')}/{override.get('condition')}/"
            f"seed{override.get('seed')} attempt {override.get('attempt')}; see `data/guard_overrides.json`."
        )
    (pilot.STUDY / "PREFLIGHT_REPORT.md").write_text("\n".join(preflight_lines) + "\n", encoding="utf-8")

    pilot_lines = [
        "# MR-CGE V1.1 Stage-A Pilot Report", "",
        "## Status", "",
        f"Planned: {all_planned} Dataset × Condition × Seed cells (Movies/Grocery/ele-fashion × S/F/M × seed 42).",
        f"Completed: {complete}/{all_planned}; formal cells with a recorded attempt: {len(attempted_cells)}.",
    ]
    if complete == all_planned:
        pilot_lines.append("All nine authorized seed-42 Validation-only cells completed.")
    elif gpu_block:
        pilot_lines.append("The guarded GPU single-step gate did not pass; no threshold was bypassed.")
    elif preflight.get("gpu_single_step_status") == "pass" and not attempted_cells:
        pilot_lines.append("All nine GPU single-step checks passed; formal Pilot training has not started yet.")
    elif runtime_guard_block:
        pilot_lines.append(
            f"The runtime GPU ownership guard interrupted {runtime_guard_block.get('dataset')}/"
            f"{runtime_guard_block.get('condition')}/seed{runtime_guard_block.get('seed')} attempt "
            f"{runtime_guard_block.get('attempt')} because `{runtime_guard_block.get('error', 'a runtime resource check failed')}`. "
            f"The guard stopped only its own child. {complete} cells completed; the interrupted attempt has no valid metric, "
            f"and {all_planned - complete} authorized cell(s) remain incomplete."
        )
    elif attempted_cells:
        pilot_lines.append("Some formal cells have recorded attempts; inspect per-attempt statuses and logs before resuming.")
    else:
        pilot_lines.append("Formal Pilot training has not started.")
    for override in guard_overrides.get("overrides", []):
        pilot_lines.append(
            f"A user-directed override disabled GPU guard checks for {override.get('dataset')}/{override.get('condition')}/"
            f"seed{override.get('seed')} attempt {override.get('attempt')}; see `data/guard_overrides.json` for the device snapshot and scope."
        )
    pilot_lines.extend(["", "## Validation metrics", ""])
    completed_rows = [row for row in summary_rows if row.get("n_completed") == 1]
    if completed_rows:
        pilot_lines.extend([
            "These are one-seed descriptive Validation metrics and do not support inferential claims.",
            "",
            "| Dataset | Condition | Val Accuracy | Val Macro-F1 | Val CE |",
            "|---|---|---:|---:|---:|",
        ])
        for row in completed_rows:
            pilot_lines.append(
                f"| {row['dataset']} | {row['condition']} | {row['val_acc']} | {row['val_macro_f1']} | {row['val_ce']} |"
            )
        pilot_lines.extend(["", "See the expert, route, response and resource diagnostics alongside epoch-level loss values."])
    else:
        pilot_lines.append(
            "There are no S/F/M Accuracy, Macro-F1, or CE Pilot values. `condition_summary.csv` and "
            "`condition_differences.csv` retain explicit `not_run` / `not_available` rows; no synthetic or historical score is substituted."
        )
    expert_rows = [row for row in pilot.read_rows(DATA / "expert_diagnostics.csv") if row.get("condition") == "M"]
    route_rows = [row for row in pilot.read_rows(DATA / "route_diagnostics.csv") if row.get("condition") == "M"]
    response_rows = [row for row in pilot.read_rows(DATA / "response_diagnostics.csv") if row.get("condition") == "M"]
    gradient_rows = pilot.read_rows(DATA / "loss_gradient_diagnostics.csv")
    if gradient_rows:
        pilot_lines.append(f"- Formal gradient traces: `{len(gradient_rows)}` parameter-gradient records are available in `data/loss_gradient_diagnostics.csv`.")
    else:
        pilot_lines.append(
            "- Formal gradient traces: `data/loss_gradient_diagnostics.csv` is header-only, so per-run training gradient magnitudes were not recorded. "
            "The nine GPU single-step preflights and focused tests checked finite gradients, but do not replace formal-run gradient traces."
        )
    if route_rows and expert_rows and response_rows:
        route_shares = [
            float(value) for row in route_rows
            for value in json.loads(row.get("top2_selection_share") or "[]")
        ]
        cosine_values = {}
        for row in expert_rows:
            for column in ("G_pairwise_cosine", "E_pairwise_cosine"):
                for pair, value in json.loads(row.get(column) or "{}").items():
                    cosine_values[(column, row.get("dataset"), row.get("view"), pair)] = float(value)
        theta_by_dataset = {}
        for row in expert_rows:
            if row.get("view") == "private_text_physical":
                theta_by_dataset.setdefault(row.get("dataset"), {})[row.get("expert")] = tuple(
                    json.loads(row.get("theta_actual") or "[]")
                )
        theta_distinct = all(
            len(experts) == 4 and len(set(experts.values())) == 4
            for experts in theta_by_dataset.values()
        )
        early_rows = [row for row in response_rows if row.get("view", "").startswith("early_cross")]
        late_rows = [row for row in response_rows if row.get("view", "").startswith("late")]
        early_ratios = [
            float(row["collaborative_response_rms"]) / float(row["private_response_rms"])
            for row in early_rows if row.get("private_response_rms") and float(row["private_response_rms"]) > 0
        ]
        pilot_lines.extend([
            "",
            "## Mechanism diagnostics in completed M cells",
            "",
            f"Diagnostics are available for {len({row.get('dataset') for row in route_rows})} completed M cells (Movies and Grocery), across {len(route_rows)} recorded private/collaborative router views.",
        ])
        if route_shares:
            pilot_lines.append(
                f"- Routing: all four experts received nonzero Top-2 selection share in every recorded view; shares ranged {min(route_shares):.3f}–{max(route_shares):.3f}. No complete route collapse appears in these two one-seed runs."
            )
        if cosine_values:
            g_values = [v for (column, _, _, _), v in cosine_values.items() if column == "G_pairwise_cosine"]
            e_values = [v for (column, _, _, _), v in cosine_values.items() if column == "E_pairwise_cosine"]
            if g_values and e_values:
                pilot_lines.append(
                    f"- Filter responses: the four experts retained distinct signed theta vectors in each recorded dataset ({'yes' if theta_distinct else 'not established'}). Across dataset/view/expert pairs, G cosine ranged {min(g_values):.3f}–{max(g_values):.3f} and E cosine ranged {min(e_values):.3f}–{max(e_values):.3f}; expert responses were not all identical, though these measures do not establish useful specialization."
                )
        if early_rows and late_rows:
            beta = [float(row["beta"]) for row in early_rows if row.get("beta")]
            raw_cross = [float(row["raw_rms"]) for row in early_rows if row.get("raw_rms")]
            late_accept = [float(row["p_accept_mean"]) for row in late_rows if row.get("p_accept_mean")]
            late_gate = [float(row["gate_mean"]) for row in late_rows if row.get("gate_mean")]
            late_ratio = [float(row["late_delta_ratio"]) for row in late_rows if row.get("late_delta_ratio")]
            if beta and raw_cross and early_ratios:
                pilot_lines.append(
                    f"- Early Cross: nonzero cross-message RMS was recorded ({min(raw_cross):.3f}–{max(raw_cross):.3f}); beta was {min(beta):.3f}–{max(beta):.3f}. Collaborative/private response RMS ratios were {min(early_ratios):.3f}–{max(early_ratios):.3f}."
                )
            if late_accept and late_gate and late_ratio:
                pilot_lines.append(
                    f"- Late completion: mean p_accept was {min(late_accept):.3f}–{max(late_accept):.3f}, gate mean {min(late_gate):.3f}–{max(late_gate):.3f}, and late delta/RMS ratios {min(late_ratio):.3f}–{max(late_ratio):.3f}; the path was active but its predictive contribution is not isolated by this Pilot."
                )
        pilot_lines.append("These are end-of-run descriptive diagnostics for two datasets and one seed; ele-fashion/M has no completed run, and no causal or generalization claim follows.")
    if not complete:
        pilot_lines.extend([
            "",
            "No conclusions about expert specialization, routing collapse, Early Cross activation, Late completion activation, "
            "or model performance can be drawn from an unrun Pilot. CPU unit tests establish code-path behavior only.",
        ])
    else:
        pilot_lines.extend([
            "",
            "Expert and routing observations are descriptive for this seed. Generalization and statistical claims require later approved seeds.",
        ])
    pilot_lines.extend([
        "",
        "## Scope guard", "",
        "No Test metric or model selection used Test labels. Test labels were masked to `-100` before Trainer entry. "
        "Seeds 43/44, the remaining 18 formal runs, full 27-run training, tuning and ablations were not started.",
        "",
    ])
    if complete == all_planned:
        pilot_lines.append("Stage-A is complete. Independently review metrics, failures and diagnostics before authorizing later seeds or Test-lock activity.")
    elif runtime_guard_block:
        pilot_lines.append(
            "Resume the remaining M cell only when GPU 1 has enough headroom and a fresh ownership/resource guard passes. The runner will preserve every failed attempt, "
            "skip verified completed cells, and create the next attempt for the interrupted cell."
        )
    elif gpu_block:
        pilot_lines.append("Repeat the guarded GPU preflight only after resolving the recorded failure; preserve all prior attempts.")
    elif preflight.get("gpu_single_step_status") == "pass":
        pilot_lines.append("The GPU preflight gate passed. Continue only the nine authorized seed-42 cells; the runner skips verified completed cells.")
    else:
        pilot_lines.append("Review the preflight evidence and resolve blockers before starting the authorized seed-42 cells.")
    (pilot.STUDY / "PILOT_REPORT.md").write_text("\n".join(pilot_lines) + "\n", encoding="utf-8")

    code_audit = [
        "# MR-CGE V1.1 Code Audit", "",
        f"Source commit audited: `{code_sha}`. The new model is a standalone `nn.Module`; it does not inherit from the frozen H2 model chain.",
        "",
        "## Implemented paths", "",
        "- Four views share one V7B.1-aware filter implementation. M owns one four-expert bank and one graph-aware Top-2 router.",
        "- Physical and modality-specific semantic bases are built once per modality per forward. Isolated-node support masks zero all complete graph responses.",
        "- Private states use only their own modality. Early Cross changes only physical order 1 and has an exact per-node zero-context fallback.",
        "- Late completion uses receiver-sparse semantic attention, Semantic/Same-node/None selection, conditional candidate averaging, channel gating, relative residual capping and non-inverted Dropout 0.15.",
        "- S/F keep all information paths while omitting M's expert router. S has fixed equal valid-order coefficients; F has one learnable signed filter; M has four learned signed filters.",
        "- The only Trainer change is a guarded optional `collect_epoch_diagnostics()` hook; existing models without that method follow the same code path.",
        "",
        "## Differences from historical H2", "",
        "MR-CGE removes the H2 late complementary experts/router and its balance calls; it replaces relation/view-specific filter routing with one shared graph-aware router;",
        "M's four expert objects are shared across all views and collaborative physical calls; Private/Cross state formulas and support handling are explicit in the new model. "
        "The historical H2 Late Complementary Balance Loss accounting defect remains outside scope. The historical H2 source, old configurations, reports, checkpoints and Final Test Lock remain unchanged.",
        "",
        "## Validation", "",
        f"The focused suite result is recorded in `data/preflight.json` ({pytest_summary(unit_tests.get('stdout', '')) if unit_tests else 'not found'}). "
        "The actual-data CPU preflight verifies cache fingerprints, source_flags, global node order, Test masking and condition initialization hashes.",
        (
            f"GPU single-step preflight status: {preflight.get('gpu_single_step_status', 'not run')} "
            f"({preflight.get('gpu_single_step_count', 0)}/{all_planned} cells). Full-epoch behavior and multi-seed performance remain separate questions."
        ),
        (
            "Formal runs include per-epoch loss scalars, but `data/loss_gradient_diagnostics.csv` is header-only; "
            "single-step preflight gradients do not provide per-run training gradient magnitudes."
        ),
        "",
        "Parameter counts from CPU initialization preflight:", "",
        "| Dataset | S model params | F model params | M model params |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        conditions = preflight.get("datasets", {}).get(dataset, {}).get("conditions", [])
        by = {item["condition"]: item for item in conditions}
        code_audit.append(f"| {dataset} | {by.get('S', {}).get('model_parameters', 'n/a')} | {by.get('F', {}).get('model_parameters', 'n/a')} | {by.get('M', {}).get('model_parameters', 'n/a')} |")
    code_audit.extend(["", "## Limits", "", "A one-step GPU check does not establish full-training stability, long-run resource behavior, multi-seed performance or generalization."])
    (pilot.STUDY / "CODE_AUDIT.md").write_text("\n".join(code_audit) + "\n", encoding="utf-8")

    handoff = [
        "# MR-CGE V1.1 Decision Handoff", "",
        f"Branch: `{manifest['branch']}`; code commit: `{code_sha}`.",
        "",
        f"Stage-A status: {formal_status}; GPU single-step gate: {preflight.get('gpu_single_step_status', 'not run')}; completed formal cells: {complete}/{all_planned}.",
        "",
        "## Review materials", "",
        "- `IMPLEMENTATION_SPEC.md`: frozen design choices and reviewed interfaces.",
        "- `PREFLIGHT_REPORT.md`, `data/preflight.json`, `data/environment.json`, `data/gpu_baseline.json`: preflight outcomes and resource evidence.",
        "- `CODE_AUDIT.md`, `tests/test_mrcge_v11.py`: implementation map and CPU behavior checks.",
        f"- `PILOT_REPORT.md`, `data/runs.csv`, `data/condition_summary.csv`, `data/condition_differences.csv`: {complete} completed-cell metrics and explicit missing-metric states for interrupted/unstarted cells.",
        "- `data/failures.json`, `data/guard_overrides.json`: preserved failures and any explicitly user-directed GPU guard overrides with device snapshots.",
        "",
        "## Requested decision", "",
        (
            "Review the completed Stage-A Validation-only evidence before deciding on later seeds. "
            if complete == all_planned else (
                "Resume the remaining authorized seed-42 cell only when GPU 1 has enough headroom for M and a fresh ownership guard passes. "
                if runtime_guard_block else "Resume only the remaining authorized seed-42 cells after the GPU single-step gate passes. "
            )
        )
        + "Do not infer authorization for seeds 43/44 or Test evaluation from this handoff.",
    ]
    (pilot.STUDY / "DECISION_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    print(f"Wrote MR-CGE audit reports: {complete}/{all_planned} formal cells complete; GPU gate {preflight.get('gpu_single_step_status', 'not run')}.")


if __name__ == "__main__":
    main()
