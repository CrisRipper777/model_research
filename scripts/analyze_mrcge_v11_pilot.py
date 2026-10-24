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
    summary_rows = []
    differences = []
    contrast_defs = (("M-F", "M", "F"), ("F-S", "F", "S"), ("M-S", "M", "S"))
    for dataset in DATASETS:
        for condition in CONDITIONS:
            row = run_by_key.get((dataset, condition, pilot.SEED))
            summary_rows.append({
                "dataset": dataset, "condition": condition, "n_completed": 1 if row else 0,
                "seed": pilot.SEED, "val_acc": row.get("val_acc", "") if row else "",
                "val_macro_f1": row.get("val_macro_f1", "") if row else "",
                "val_ce": row.get("val_ce", "") if row else "",
                "status": row.get("status", "not_run") if row else "not_run_gpu_preflight_blocked",
                "interpretation": "one-seed descriptive validation metric" if row else "no Pilot metric; formal training did not start",
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
                    "delta_val_ce": "", "interpretation": "both Pilot cells are required; no formal training started",
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
    if preflight.get("gpu_single_step_status") == "pass":
        resource_rows = [row for row in resource_rows if row.get("phase") != "gpu_single_step_preflight_blocked"]
    if not resource_rows and gpu_block:
        snapshot = gpu_block.get("gpu_snapshot", {})
        gpu0, gpu1 = snapshot.get("GPU0", {}), snapshot.get("GPU1", {})
        resource_rows.append({
            "dataset": gpu_block.get("dataset", ""), "condition": gpu_block.get("condition", ""),
            "seed": pilot.SEED, "attempt": "", "phase": "gpu_single_step_preflight_blocked",
            "physical_gpu": 0, "gpu0_free_mib": gpu0.get("memory_free_mib", ""),
            "gpu0_utilization_percent": gpu0.get("utilization_percent", ""),
            "gpu0_temperature_c": gpu0.get("temperature_c", ""),
            "gpu0_compute_pids": compact(gpu0.get("compute_pids")),
            "gpu1_free_mib": gpu1.get("memory_free_mib", ""),
            "gpu1_utilization_percent": gpu1.get("utilization_percent", ""),
            "gpu1_temperature_c": gpu1.get("temperature_c", ""),
            "gpu1_compute_pids": compact(gpu1.get("compute_pids")),
            "status": "blocked_before_training", "reason": gpu_block.get("reason", ""),
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
    failures = json.loads((DATA / "failures.json").read_text(encoding="utf-8")) if (DATA / "failures.json").is_file() else {"failures": []}
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
        "gpu_block": gpu_block,
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
        f"- Formal Pilot cells started: **{complete}/{all_planned}**.",
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
    else:
        preflight_lines.append("No GPU guard refusal was recorded.")
    preflight_lines.extend(["", "## Preserved failures", "", f"`data/failures.json` contains {len(failures.get('failures', []))} recorded events."])
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
    elif attempted_cells:
        pilot_lines.append("Some formal cells have recorded attempts; inspect per-attempt statuses and logs before resuming.")
    else:
        pilot_lines.append("Formal Pilot training has not started.")
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
        pilot_lines.extend(["", "Use the expert, route, response, loss-gradient and resource diagnostics to inspect model behavior."])
    else:
        pilot_lines.append(
            "There are no S/F/M Accuracy, Macro-F1, or CE Pilot values. `condition_summary.csv` and "
            "`condition_differences.csv` retain explicit `not_run` / `not_available` rows; no synthetic or historical score is substituted."
        )
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
        "- `PILOT_REPORT.md`, `data/runs.csv`, `data/condition_summary.csv`, `data/condition_differences.csv`: explicit record that no Pilot metrics exist.",
        "- `data/failures.json`: original fingerprint-field preflight error and GPU guard refusal.",
        "",
        "## Requested decision", "",
        (
            "Review the completed Stage-A Validation-only evidence before deciding on later seeds. "
            if complete == all_planned else "Resume only the remaining authorized seed-42 cells after the GPU single-step gate passes. "
        )
        + "Do not infer authorization for seeds 43/44 or Test evaluation from this handoff.",
    ]
    (pilot.STUDY / "DECISION_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    print(f"Wrote MR-CGE audit reports: {complete}/{all_planned} formal cells complete; GPU gate {preflight.get('gpu_single_step_status', 'not run')}.")


if __name__ == "__main__":
    main()
