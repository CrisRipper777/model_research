#!/usr/bin/env python3
"""Repair known RM-CGE V2 derived CSV headers and verify rows against run JSON."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v2 import (
    ARTIFACT_CSV_FIELDS,
    DATA_DIR,
    ROOT,
    sha256_file,
    write_json,
)


def _decode_cell(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return float(value)
        except ValueError:
            return value


def _matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-9)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(_matches(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(_matches(actual[k], expected[k]) for k in expected)
    return actual == expected


def write_table(name: str, rows: list[dict[str, Any]]) -> None:
    fields = ARTIFACT_CSV_FIELDS[name]
    with (DATA_DIR / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def audit_sources() -> dict[str, Any]:
    with (DATA_DIR / "runs.csv").open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    if len(runs) != 27:
        raise RuntimeError(f"expected 27 run rows, found {len(runs)}")
    keys = [(row["dataset"], row["variant"], int(row["seed"])) for row in runs]
    if len(set(keys)) != 27:
        raise RuntimeError("run ledger contains duplicate or missing Dataset × Variant × Seed keys")
    if any(row["status"] != "completed_and_verified" for row in runs):
        raise RuntimeError("run ledger contains a non-verified run")
    if any(row["test_metrics_present"].lower() != "false" for row in runs):
        raise RuntimeError("run ledger indicates Test metrics")

    source_diagnostics: dict[tuple[str, str, int], dict[str, Any]] = {}
    expected_history_rows = 0
    finite_values_checked = 0
    for row in runs:
        key = (row["dataset"], row["variant"], int(row["seed"]))
        path = ROOT / row["diagnostics_path"]
        diag = json.loads(path.read_text(encoding="utf-8"))
        source_diagnostics[key] = diag
        if diag.get("test_metrics_present") is not False:
            raise RuntimeError(f"Test metrics were present in {key}")
        if not diag.get("reloaded_checkpoint_matches_validation_accuracy"):
            raise RuntimeError(f"Best-Accuracy checkpoint reload failed for {key}")
        if not diag.get("best_f1_checkpoint_reloads"):
            raise RuntimeError(f"Best-F1 checkpoint reload failed for {key}")
        history = diag.get("history", [])
        expected_history_rows += len(history)
        for record in history:
            for metric in ("train_loss", "train_ce", "train_ce_eval", "train_accuracy_eval", "train_macro_f1_eval", "val_ce", "val_acc", "val_macro_f1"):
                value = float(record[metric])
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite {metric} in {key}, epoch {record.get('epoch')}")
                finite_values_checked += 1

    def read_rows(name: str) -> list[dict[str, str]]:
        with (DATA_DIR / name).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ARTIFACT_CSV_FIELDS[name]:
                raise RuntimeError(f"wrong header after repair in {name}: {reader.fieldnames}")
            return list(reader)

    histories = read_rows("epoch_history.csv")
    if len(histories) != expected_history_rows:
        raise RuntimeError(f"epoch history count mismatch: {len(histories)} != {expected_history_rows}")
    for name, expected in (
        ("edge_weight_diagnostics.csv", 54),
        ("message_scale_diagnostics.csv", 54),
        ("expert_router_diagnostics.csv", 54),
        ("frozen_interventions.csv", 108),
        ("resource_profile.csv", 27),
    ):
        rows = read_rows(name)
        if len(rows) != expected:
            raise RuntimeError(f"row count mismatch in {name}: {len(rows)} != {expected}")

    mechanism_rows = read_rows("message_scale_diagnostics.csv")
    compared = 0
    for row in mechanism_rows:
        key = (row["dataset"], row["variant"], int(row["seed"]))
        modality = row["modality"]
        values = source_diagnostics[key]["mechanism_details"]["modalities"][modality]
        for column in ("u1_rms", "delta_rms", "delta_hat_rms", "cross_rms", "cross_hat_rms", "knn_rms", "knn_hat_rms", "router_entropy", "router_correction_rms", "weighted_graph_correction_rms", "expert_output_rms"):
            actual = _decode_cell(row[column])
            expected = values.get(column)
            if not _matches(actual, expected):
                raise RuntimeError(f"{column} does not match source JSON for {key}/{modality}")
            compared += 1
        for column in ("expert_coefficients", "expert_output_rms_by_expert", "expert_output_pairwise_cosine"):
            actual = _decode_cell(row[column])
            expected = values.get(column)
            if not _matches(actual, expected):
                raise RuntimeError(f"{column} does not match source JSON for {key}/{modality}")
            compared += 1

    for row in read_rows("edge_weight_diagnostics.csv"):
        key = (row["dataset"], row["variant"], int(row["seed"]))
        source = source_diagnostics[key]["mechanism_details"]["edge_weights"]
        values = source[row["modality"]]
        for column in ("mean", "std", "p10", "p50", "p90", "mean_abs_from_one", "fraction_abs_gt_0_1", "fraction_abs_gt_0_2", "mean_abs_reverse_difference"):
            if not _matches(_decode_cell(row[column]), values.get(column)):
                raise RuntimeError(f"{column} edge diagnostic does not match source JSON for {key}/{row['modality']}")
            compared += 1
        for column in ("text_visual_correlation", "text_visual_correlation_defined", "text_visual_mean_abs_difference"):
            if not _matches(_decode_cell(row[column]), source.get(column)):
                raise RuntimeError(f"{column} edge diagnostic does not match source JSON for {key}")
            compared += 1

    router_rows = read_rows("expert_router_diagnostics.csv")
    for row in router_rows:
        key = (row["dataset"], row["variant"], int(row["seed"]))
        values = source_diagnostics[key]["mechanism_details"]["modalities"][row["modality"]]
        for column in ARTIFACT_CSV_FIELDS["expert_router_diagnostics.csv"][4:]:
            if not _matches(_decode_cell(row[column]), values.get(column)):
                raise RuntimeError(f"{column} router diagnostic does not match source JSON for {key}/{row['modality']}")
            compared += 1

    csv_artifacts = {}
    for path in sorted(DATA_DIR.glob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.reader(handle))
        if not raw_rows:
            raise RuntimeError(f"empty CSV artifact: {path.name}")
        widths = sorted({len(row) for row in raw_rows})
        if widths != [len(raw_rows[0])]:
            raise RuntimeError(f"CSV row width mismatch in {path.name}: {widths}")
        csv_artifacts[path.name] = {
            "rows_including_header": len(raw_rows),
            "columns": len(raw_rows[0]),
            "sha256": sha256_file(path),
        }

    return {
        "status": "passed",
        "verified_runs": len(runs),
        "unique_run_keys": len(set(keys)),
        "test_metrics_present": False,
        "finite_epoch_metric_values_checked": finite_values_checked,
        "epoch_history_rows": len(histories),
        "message_scale_source_fields_compared": compared,
        "checkpoints_reloaded": {"best_accuracy": True, "best_macro_f1": True},
        "csv_artifacts": csv_artifacts,
    }


def main() -> None:
    with (DATA_DIR / "runs.csv").open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    histories: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for run in runs:
        run["error"] = run.get("error") or ""
        diag = json.loads((ROOT / run["diagnostics_path"]).read_text(encoding="utf-8"))
        for epoch in diag["history"]:
            histories.append({"dataset": run["dataset"], "variant": run["variant"], "seed": int(run["seed"]), **epoch})
        intervention_rows.extend(diag["interventions"])
        details = diag["mechanism_details"]
        edge = details.get("edge_weights", {})
        for modality in ("text", "visual"):
            if modality in edge:
                edge_rows.append({
                    "dataset": run["dataset"], "variant": run["variant"], "seed": int(run["seed"]), "modality": modality,
                    **edge[modality],
                    "text_visual_correlation": edge.get("text_visual_correlation"),
                    "text_visual_correlation_defined": edge.get("text_visual_correlation_defined"),
                    "text_visual_mean_abs_difference": edge.get("text_visual_mean_abs_difference"),
                })
        for modality, values in details.get("modalities", {}).items():
            item = {"dataset": run["dataset"], "variant": run["variant"], "seed": int(run["seed"]), "modality": modality, **values}
            scale_rows.append(item)
            router_rows.append({key: item.get(key) for key in ARTIFACT_CSV_FIELDS["expert_router_diagnostics.csv"]})
        gradient_rows.append({
            "dataset": run["dataset"], "variant": run["variant"], "seed": int(run["seed"]),
            "gradients": diag.get("gradient_diagnostics", {}),
            "parameter_update_norms": diag["parameter_update_norms"],
        })
        run_json = json.loads((ROOT / run["run_metrics_path"]).read_text(encoding="utf-8"))
        metadata = run_json["runs"][0]["metadata"]
        resource_rows.append({
            "dataset": run["dataset"], "variant": run["variant"], "seed": int(run["seed"]),
            "wall_seconds": float(run["wall_seconds"]),
            "epochs_completed": metadata.get("epochs_completed"),
            "mean_epoch_wall_seconds": metadata.get("mean_epoch_wall_seconds"),
            "peak_allocated_bytes": metadata.get("cuda_peak_allocated_bytes"),
            "peak_reserved_bytes": metadata.get("cuda_peak_reserved_bytes"),
            "cuda_device_name": metadata.get("cuda_device_name"),
        })
    write_table("runs.csv", runs)
    write_table("epoch_history.csv", histories)
    write_table("edge_weight_diagnostics.csv", edge_rows)
    write_table("message_scale_diagnostics.csv", scale_rows)
    write_table("expert_router_diagnostics.csv", router_rows)
    write_table("frozen_interventions.csv", intervention_rows)
    write_table("resource_profile.csv", resource_rows)
    write_json(DATA_DIR / "gradient_audit.json", {"runs": gradient_rows})
    audit = audit_sources()
    report = {
        "repair": "derived CSV/JSON evidence regenerated from immutable per-run checkpoint diagnostics and run metrics; training artifacts were not changed",
        "source_audit": audit,
    }
    write_json(DATA_DIR / "artifact_integrity.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
