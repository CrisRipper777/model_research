#!/usr/bin/env python3
"""Validate and summarize the independent MR-CGE V1.2 confirmation campaign."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/mrcge_v12_confirmation"
DATA = STUDY / "data"
OLD_DATA = ROOT / "research/mrcge_v12/data"
BASE_SHA = "712bf477085b8e9022a746330e3ae48e12fa0841"
DATASETS = ("Movies", "Grocery", "Toys", "ele-fashion")
M_CONDITIONS = ("M_repaired", "M_anchor")
ALL_CONDITIONS = ("S_repaired", "F_sinit", *M_CONDITIONS)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnum(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def checkpoint_audit(row: dict[str, str], path_field: str, hash_field: str,
                     metric_fields: dict[str, str]) -> tuple[bool, str, dict[str, Any]]:
    path = Path(row.get(path_field, ""))
    if not path.is_file():
        return False, f"missing checkpoint: {path}", {}
    actual_sha = sha256_file(path)
    if actual_sha != row.get(hash_field):
        return False, f"checkpoint SHA256 mismatch: {path}", {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    provenance = payload.get("run_metadata", {}).get("study_provenance", {})
    metrics = payload.get("metrics", {})
    for field, metric_name in metric_fields.items():
        expected, actual = fnum(row.get(field)), fnum(metrics.get(metric_name))
        if expected is not None and actual is None:
            return False, f"checkpoint lacks metric {metric_name}", payload
        if expected is not None and actual is not None and abs(expected - actual) > 4e-6:
            return False, f"{field} differs from reloaded checkpoint metric {metric_name}", payload
    if provenance.get("test_evaluation") is not False or provenance.get("test_label_values_indexed") is not False:
        return False, "checkpoint provenance does not certify Test-off", payload
    if row.get("source_git_sha") and provenance.get("source_git_sha") != row["source_git_sha"]:
        return False, "checkpoint source SHA differs from run row", payload
    if row.get("model_source_sha256") and provenance.get("model_source_sha256") != row["model_source_sha256"]:
        return False, "checkpoint model source SHA differs from run row", payload
    provenance_fields = {
        "config_sha256": "config_sha256", "dataset_fingerprint": "dataset_fingerprint",
        "physical_edge_fingerprint": "physical_edge_fingerprint", "split_fingerprint": "split_fingerprint",
        "split_file_sha256": "split_file_sha256", "feature_fingerprint": "feature_fingerprint",
        "semantic_cache_sha256": "semantic_cache_sha256", "candidate_fingerprint": "candidate_fingerprint",
        "v7b_fingerprint": "v7b_fingerprint",
    }
    for row_field, provenance_field in provenance_fields.items():
        expected, recorded = row.get(row_field), provenance.get(provenance_field)
        if expected and recorded != expected:
            return False, f"checkpoint {provenance_field} differs from run row", payload
    expected_trainer = row.get("trainer_source_sha256")
    if expected_trainer and provenance.get("task_source_sha256") != expected_trainer:
        return False, "checkpoint Trainer source SHA differs from run row", payload
    return True, "verified", payload


def validate_run(row: dict[str, str], campaign: str) -> dict[str, Any]:
    result: dict[str, Any] = {**row, "source_campaign": campaign,
                              "checkpoint_accuracy_verified": False,
                              "checkpoint_best_f1_verified": False,
                              "checkpoint_verification_note": ""}
    if row.get("status") != "complete":
        return result
    if row.get("test_evaluation", "").lower() != "false":
        result["checkpoint_verification_note"] = "run row does not certify Test-off"
        return result
    if row.get("validation_rng_unchanged", "").lower() != "true":
        result["checkpoint_verification_note"] = "run row does not certify unchanged Validation RNG"
        return result
    primary_map = {
        "val_acc": "val_acc", "val_macro_f1": "val_macro_f1", "val_ce": "val_ce",
        "train_acc": "train_accuracy_eval", "train_ce": "train_ce_eval",
    }
    ok_acc, note_acc, acc_payload = checkpoint_audit(
        row, "checkpoint_path", "checkpoint_sha256", primary_map
    )
    secondary_map = {
        "best_f1_val_acc": "val_acc", "best_f1_val_macro_f1": "val_macro_f1",
        "best_f1_val_ce": "val_ce",
    }
    ok_f1, note_f1, f1_payload = checkpoint_audit(
        row, "best_f1_checkpoint_path", "best_f1_checkpoint_sha256", secondary_map
    )
    result["checkpoint_accuracy_verified"] = ok_acc
    result["checkpoint_best_f1_verified"] = ok_f1
    result["checkpoint_verification_note"] = "; ".join(note for note in (note_acc, note_f1) if note != "verified")
    for prefix, payload in (("accuracy", acc_payload), ("best_f1", f1_payload)):
        state = payload.get("model_state", {}) if isinstance(payload, dict) else {}
        provenance = payload.get("run_metadata", {}).get("study_provenance", {}) if isinstance(payload, dict) else {}
        if provenance.get("task_source_sha256"):
            result.setdefault("trainer_source_sha256", provenance["task_source_sha256"])
        if "rho_raw" in state and "gamma_raw" in state:
            result[f"{prefix}_rho_json"] = json.dumps(torch.sigmoid(state["rho_raw"]).tolist())
            result[f"{prefix}_gamma_json"] = json.dumps(torch.sigmoid(state["gamma_raw"]).tolist())
    if campaign.startswith("confirmation") and not (ok_acc and ok_f1):
        result["eligible_complete"] = False
    else:
        # Historical CSV evidence remains usable if an old checkpoint later becomes unavailable.
        result["eligible_complete"] = True
    return result


def choose_selected_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old_rows = [row for row in read_csv(OLD_DATA / "runs.csv")
                if row.get("status") == "complete" and row.get("seed") == "42"
                and row.get("condition") in M_CONDITIONS]
    new_rows = [row for row in read_csv(DATA / "runs.csv") if row.get("condition") in ALL_CONDITIONS]
    validated = [validate_run(row, "V1.2_seed42_history") for row in old_rows]
    validated.extend(validate_run(row, "confirmation_new_run") for row in new_rows)
    desired = set()
    for dataset in DATASETS:
        for condition in M_CONDITIONS:
            desired.update((dataset, condition, seed) for seed in (42, 43, 44))
    desired.update({("ele-fashion", "S_repaired", 42), ("ele-fashion", "F_sinit", 42)})
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        if row.get("eligible_complete") is not True:
            continue
        try:
            key = (row["dataset"], row["condition"], int(row["seed"]))
        except (KeyError, ValueError):
            continue
        if key in desired:
            grouped[key].append(row)
    selected = []
    duplicate_audit = []
    for key in sorted(desired):
        candidates = grouped.get(key, [])
        if not candidates:
            continue
        candidates.sort(key=lambda row: int(row.get("attempt", 0)), reverse=True)
        chosen = candidates[0]
        chosen["selected_for_combined"] = True
        selected.append(chosen)
        if len(candidates) > 1:
            duplicate_audit.append({
                "dataset": key[0], "condition": key[1], "seed": key[2],
                "selected_attempt": chosen.get("attempt"),
                "other_complete_attempts": json.dumps([row.get("attempt") for row in candidates[1:]]),
            })
    return selected, duplicate_audit


def dataset_summaries(selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = {(row["dataset"], row["condition"], int(row["seed"])): row for row in selected}
    summaries = []
    differences = []
    paired_summaries = []
    for dataset in DATASETS:
        for condition in M_CONDITIONS:
            rows = [index[(dataset, condition, seed)] for seed in (42, 43, 44)
                    if (dataset, condition, seed) in index]
            vals = {
                "val_acc_pct": [100 * float(row["val_acc"]) for row in rows],
                "val_macro_f1_pct": [100 * float(row["val_macro_f1"]) for row in rows],
                "val_ce": [float(row["val_ce"]) for row in rows],
                "train_minus_val_acc_pp": [100 * (float(row["train_acc"]) - float(row["val_acc"])) for row in rows],
                "best_acc_epoch": [float(row["best_acc_epoch"]) for row in rows if fnum(row.get("best_acc_epoch")) is not None],
                "wall_seconds": [float(row["wall_seconds"]) for row in rows if fnum(row.get("wall_seconds")) is not None],
                "peak_allocated_gib": [int(row["peak_allocated_bytes"]) / (1024 ** 3) for row in rows if fnum(row.get("peak_allocated_bytes")) is not None],
                "peak_reserved_gib": [int(row["peak_reserved_bytes"]) / (1024 ** 3) for row in rows if fnum(row.get("peak_reserved_bytes")) is not None],
            }
            record: dict[str, Any] = {"dataset": dataset, "condition": condition, "n_seeds": len(rows),
                                     "seeds": json.dumps([int(row["seed"]) for row in rows])}
            for name, values in vals.items():
                record[f"{name}_mean"] = statistics.mean(values) if values else ""
                record[f"{name}_sample_sd"] = statistics.stdev(values) if len(values) >= 2 else ""
            summaries.append(record)
        pairs = []
        for seed in (42, 43, 44):
            repaired = index.get((dataset, "M_repaired", seed))
            anchor = index.get((dataset, "M_anchor", seed))
            if repaired is None or anchor is None:
                continue
            diff = {
                "dataset": dataset, "seed": seed,
                "repaired_attempt": repaired.get("attempt"), "anchor_attempt": anchor.get("attempt"),
                "delta_val_acc_pp_anchor_minus_repaired": 100 * (float(anchor["val_acc"]) - float(repaired["val_acc"])),
                "delta_val_macro_f1_pp_anchor_minus_repaired": 100 * (float(anchor["val_macro_f1"]) - float(repaired["val_macro_f1"])),
                "delta_val_ce_anchor_minus_repaired": float(anchor["val_ce"]) - float(repaired["val_ce"]),
                "delta_train_minus_val_acc_pp_anchor_minus_repaired": 100 * (
                    (float(anchor["train_acc"]) - float(anchor["val_acc"])) -
                    (float(repaired["train_acc"]) - float(repaired["val_acc"]))
                ),
                "delta_best_epoch_anchor_minus_repaired": int(anchor["best_acc_epoch"]) - int(repaired["best_acc_epoch"]),
                "delta_wall_seconds_anchor_minus_repaired": float(anchor["wall_seconds"]) - float(repaired["wall_seconds"]),
            }
            differences.append(diff)
            pairs.append(diff)
        for metric in ("delta_val_acc_pp_anchor_minus_repaired", "delta_val_macro_f1_pp_anchor_minus_repaired",
                       "delta_val_ce_anchor_minus_repaired"):
            values = [float(row[metric]) for row in pairs]
            paired_summaries.append({
                "dataset": dataset, "metric": metric, "n_pairs": len(values),
                "mean_delta": statistics.mean(values) if values else "",
                "sample_sd_delta": statistics.stdev(values) if len(values) >= 2 else "",
                "minimum_delta": min(values) if values else "",
                "maximum_delta": max(values) if values else "",
                "positive_pairs": sum(value > 0 for value in values),
                "negative_pairs": sum(value < 0 for value in values),
                "zero_pairs": sum(value == 0 for value in values),
            })
    return summaries, differences, paired_summaries


def vector_mean(values: list[list[float]]) -> str:
    if not values:
        return ""
    width = min(map(len, values))
    return json.dumps([statistics.mean(row[i] for row in values) for i in range(width)])


def parse_json_vector(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, (list, tuple)):
        try:
            return [float(item) for item in parsed]
        except (TypeError, ValueError):
            return None
    return None


def expert_summaries(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = {(row["dataset"], row["condition"], str(row["seed"]), str(row["attempt"])) for row in selected
              if row.get("condition") in M_CONDITIONS}
    filters = [row for row in read_csv(DATA / "filter_diagnostics.csv") + read_csv(OLD_DATA / "filter_diagnostics.csv")
               if (row.get("dataset"), row.get("condition"), row.get("seed"), row.get("attempt")) in chosen]
    routes = [row for row in read_csv(DATA / "route_diagnostics.csv") + read_csv(OLD_DATA / "route_diagnostics.csv")
              if (row.get("dataset"), row.get("condition"), row.get("seed"), row.get("attempt")) in chosen]
    responses = [row for row in read_csv(DATA / "response_diagnostics.csv") + read_csv(OLD_DATA / "response_diagnostics.csv")
                 if (row.get("dataset"), row.get("condition"), row.get("seed"), row.get("attempt")) in chosen]
    keys = sorted({(row.get("dataset", ""), row.get("condition", ""), row.get("view", ""))
                   for row in (*filters, *routes, *responses)})
    result = []
    for dataset, condition, view in keys:
        frows = [row for row in filters if (row.get("dataset"), row.get("condition"), row.get("view")) == (dataset, condition, view)]
        rrows = [row for row in routes if (row.get("dataset"), row.get("condition"), row.get("view")) == (dataset, condition, view)]
        xrows = [row for row in responses if (row.get("dataset"), row.get("condition"), row.get("view")) == (dataset, condition, view)]
        selection = [parse_json_vector(row.get("top2_selection_share")) for row in rrows]
        selection = [value for value in selection if value]
        dense = [parse_json_vector(row.get("dense_importance")) for row in rrows]
        dense = [value for value in dense if value]
        theta_by_expert, coeff_by_expert = [], []
        for expert in range(4):
            expert_rows = [row for row in frows if int(row.get("expert", -1)) == expert]
            theta_values = [parse_json_vector(row.get("theta_actual")) for row in expert_rows]
            coeff_values = [parse_json_vector(row.get("effective_coefficients")) for row in expert_rows]
            theta_values = [value for value in theta_values if value]
            coeff_values = [value for value in coeff_values if value]
            theta_by_expert.append(parse_json_vector(vector_mean(theta_values)) if theta_values else None)
            coeff_by_expert.append(parse_json_vector(vector_mean(coeff_values)) if coeff_values else None)
        g_rms, e_rms = defaultdict(list), defaultdict(list)
        for row in frows:
            expert = int(row["expert"])
            if fnum(row.get("G_rms")) is not None:
                g_rms[expert].append(float(row["G_rms"]))
            if fnum(row.get("E_rms")) is not None:
                e_rms[expert].append(float(row["E_rms"]))
        numeric_means = {}
        for field in ("physical_response_rms", "semantic_response_rms", "beta", "raw_rms",
                      "calibrated_rms", "p_accept_mean", "late_delta_ratio", "rho_text", "rho_visual",
                      "gamma_text", "gamma_visual"):
            values = [float(row[field]) for row in xrows if fnum(row.get(field)) is not None]
            numeric_means[f"mean_{field}"] = statistics.mean(values) if values else ""
        selected_runs = [row for row in selected if row.get("dataset") == dataset
                         and row.get("condition") == condition and row.get("condition") in M_CONDITIONS]
        rho_vectors = [parse_json_vector(row.get("accuracy_rho_json")) for row in selected_runs]
        gamma_vectors = [parse_json_vector(row.get("accuracy_gamma_json")) for row in selected_runs]
        rho_vectors.extend(parse_json_vector(row.get("rho_json")) for row in xrows)
        gamma_vectors.extend(parse_json_vector(row.get("gamma_json")) for row in xrows)
        rho_vectors = [value for value in rho_vectors if value]
        gamma_vectors = [value for value in gamma_vectors if value]
        result.append({
            "dataset": dataset, "condition": condition, "view": view,
            "n_seed_view_diagnostics": len({row.get("seed") for row in (*frows, *rrows, *xrows)}),
            "theta_by_expert_mean_json": json.dumps(theta_by_expert),
            "effective_coefficients_mean_json": json.dumps(coeff_by_expert),
            "top2_selection_share_mean_json": vector_mean(selection),
            "dense_importance_mean_json": vector_mean(dense),
            "minimum_top2_share": min((min(value) for value in selection), default=""),
            "dead_expert_observed": any(any(value == 0 for value in row) for row in selection),
            "G_rms_mean_by_expert_json": json.dumps([statistics.mean(g_rms[i]) if g_rms[i] else None for i in sorted(set(g_rms) | set(e_rms))]),
            "E_rms_mean_by_expert_json": json.dumps([statistics.mean(e_rms[i]) if e_rms[i] else None for i in sorted(set(g_rms) | set(e_rms))]),
            "rho_mean_json": vector_mean(rho_vectors), "gamma_mean_json": vector_mean(gamma_vectors),
            **numeric_means,
        })
    return result


def paired_route_differences(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = {(row["dataset"], row["condition"], str(row["seed"]), str(row["attempt"])) for row in selected
              if row.get("condition") in M_CONDITIONS}
    rows = [row for row in read_csv(DATA / "route_diagnostics.csv") + read_csv(OLD_DATA / "route_diagnostics.csv")
            if (row.get("dataset"), row.get("condition"), row.get("seed"), row.get("attempt")) in chosen]
    paired: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("route_scope") != "private":
            continue
        paired[(row.get("dataset", ""), int(row.get("seed", -1)), row.get("view", ""))][row.get("condition", "")] = row
    differences = []
    for dataset in DATASETS:
        views = sorted({view for ds, _, view in paired if ds == dataset})
        for view in views:
            for expert in range(4):
                seed_deltas = []
                for seed in (42, 43, 44):
                    pair = paired.get((dataset, seed, view), {})
                    repaired = parse_json_vector(pair.get("M_repaired", {}).get("top2_selection_share"))
                    anchor = parse_json_vector(pair.get("M_anchor", {}).get("top2_selection_share"))
                    if repaired and anchor and expert < len(repaired) and expert < len(anchor):
                        seed_deltas.append((seed, anchor[expert] - repaired[expert]))
                vals = [value for _, value in seed_deltas]
                same_direction = len(vals) == 3 and (all(value > 0 for value in vals) or all(value < 0 for value in vals))
                differences.append({
                    "dataset": dataset, "view": view, "expert": expert,
                    "n_paired_seeds": len(vals), "seed_deltas_json": json.dumps(seed_deltas),
                    "mean_anchor_minus_repaired_top2_share": statistics.mean(vals) if vals else "",
                    "minimum_delta": min(vals) if vals else "", "maximum_delta": max(vals) if vals else "",
                    "positive_seeds": sum(value > 0 for value in vals),
                    "negative_seeds": sum(value < 0 for value in vals),
                    "same_direction_all_three": same_direction,
                })
    return differences


def response_comparison(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = {(row["dataset"], row["condition"], str(row["seed"]), str(row["attempt"])): row for row in selected}
    rows = read_csv(DATA / "response_diagnostics.csv") + read_csv(OLD_DATA / "response_diagnostics.csv")
    result = []
    for row in rows:
        key = (row.get("dataset"), row.get("condition"), row.get("seed"), row.get("attempt"))
        if key not in chosen:
            continue
        selected_run = chosen[key]
        view = row.get("view", "")
        modality = view.rsplit("_", 1)[-1] if view else ""
        record = {**row, "modality": modality,
                  "accuracy_checkpoint_rho_json": selected_run.get("accuracy_rho_json", ""),
                  "accuracy_checkpoint_gamma_json": selected_run.get("accuracy_gamma_json", ""),
                  "test_evaluation": False}
        physical = fnum(row.get("physical_response_rms"))
        semantic = fnum(row.get("semantic_response_rms"))
        record["physical_to_semantic_response_rms_ratio"] = physical / semantic if physical is not None and semantic else ""
        result.append(record)
    return result


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value in (None, ""):
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_reports(selected: list[dict[str, Any]], summaries: list[dict[str, Any]],
                  differences: list[dict[str, Any]], paired_summaries: list[dict[str, Any]],
                  manifest: dict[str, Any]) -> None:
    preflight_path = DATA / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    gpu_rows = preflight.get("gpu_single_step", [])
    gpu_counts = {status: sum(row.get("status") == status for row in gpu_rows)
                  for status in ("pass", "blocked_resource", "failed")}
    cpu_attempts = preflight.get("cpu_preflight_attempts", [])
    cpu_attempt_text = "\n".join(
        f"- Attempt {row.get('attempt')}: **{row.get('status')}** — {row.get('reason', 'completed')}"
        for row in cpu_attempts
    ) or "- CPU preflight attempt history is not separately versioned in the original record."
    new_rows = [row for row in selected if row.get("source_campaign") == "confirmation_new_run"]
    checkpoint_ok = sum(row.get("checkpoint_accuracy_verified") is True and
                        row.get("checkpoint_best_f1_verified") is True for row in new_rows)
    status_counts = {
        "complete": manifest.get("completed_new_cells", 0),
        "failed": manifest.get("failed_new_cells", 0),
        "blocked_resource": manifest.get("blocked_new_cells", 0),
        "unstarted": manifest.get("unstarted_new_cells", 18),
    }
    split_hashes = sorted({row.get("split_fingerprint", "") for row in preflight.get("datasets", [])})
    freeze_sha = preflight.get("source_git_sha", "not yet frozen")
    src_hashes = preflight.get("source_file_sha256s", {})

    summary_lookup = {(row["dataset"], row["condition"]): row for row in summaries}
    paired_lookup = {(row["dataset"], row["metric"]): row for row in paired_summaries}
    seed_table = []
    ce_table = []
    for dataset in DATASETS:
        row = []
        ce_row = []
        for condition in M_CONDITIONS:
            summary = summary_lookup.get((dataset, condition), {})
            acc = fnum(summary.get("val_acc_pct_mean"))
            f1 = fnum(summary.get("val_macro_f1_pct_mean"))
            acc_sd = fnum(summary.get("val_acc_pct_sample_sd"))
            f1_sd = fnum(summary.get("val_macro_f1_pct_sample_sd"))
            acc_sd_text = f"{acc_sd:.3f}" if acc_sd is not None else "n/a"
            f1_sd_text = f"{f1_sd:.3f}" if f1_sd is not None else "n/a"
            row.append(f"{acc:.3f} ± {acc_sd_text} / {f1:.3f} ± {f1_sd_text}"
                       if acc is not None and f1 is not None else "n/a")
            ce_mean = fnum(summary.get("val_ce_mean"))
            ce_sd = fnum(summary.get("val_ce_sample_sd"))
            ce_sd_text = f"{ce_sd:.4f}" if ce_sd is not None else "n/a"
            ce_row.append(f"{ce_mean:.4f} ± {ce_sd_text}" if ce_mean is not None else "n/a")
        seed_table.append([dataset, *row])
        ce_table.append([dataset, *ce_row])

    paired_table = []
    for dataset in DATASETS:
        acc = paired_lookup.get((dataset, "delta_val_acc_pp_anchor_minus_repaired"), {})
        f1 = paired_lookup.get((dataset, "delta_val_macro_f1_pp_anchor_minus_repaired"), {})
        ce = paired_lookup.get((dataset, "delta_val_ce_anchor_minus_repaired"), {})
        acc_range = (f"{float(acc['minimum_delta']):.3f} … {float(acc['maximum_delta']):.3f}"
                     if fnum(acc.get("minimum_delta")) is not None else "")
        f1_range = (f"{float(f1['minimum_delta']):.3f} … {float(f1['maximum_delta']):.3f}"
                    if fnum(f1.get("minimum_delta")) is not None else "")
        paired_table.append([
            dataset, acc.get("mean_delta", ""),
            acc_range,
            f"+{acc.get('positive_pairs', 0)} / −{acc.get('negative_pairs', 0)}",
            f1.get("mean_delta", ""),
            f1_range,
            f"+{f1.get('positive_pairs', 0)} / −{f1.get('negative_pairs', 0)}",
            ce.get("mean_delta", ""),
        ])
    equal_weighted = {}
    for metric in ("delta_val_acc_pp_anchor_minus_repaired", "delta_val_macro_f1_pp_anchor_minus_repaired"):
        values = [fnum(paired_lookup.get((dataset, metric), {}).get("mean_delta")) for dataset in DATASETS]
        values = [value for value in values if value is not None]
        equal_weighted[metric] = statistics.mean(values) if values else None

    fashion = [row for row in selected if row.get("dataset") == "ele-fashion" and str(row.get("seed")) == "42"]
    fashion.sort(key=lambda row: ALL_CONDITIONS.index(row["condition"]))
    fashion_table = [[row.get("condition"),
                      100 * float(row["val_acc"]) if fnum(row.get("val_acc")) is not None else "",
                      100 * float(row["val_macro_f1"]) if fnum(row.get("val_macro_f1")) is not None else "",
                      row.get("val_ce", ""), row.get("source_campaign", ""), row.get("checkpoint_accuracy_verified", "")]
                     for row in fashion]
    historical = [
        ["Movies", "H2_hybrid_plus", 56.779, 49.780, 56.569, 48.330],
        ["Grocery", "H2_hybrid_plus", 83.856, 75.972, 83.953, 76.899],
        ["Toys", "H2_physical_3hop (historical 4-hop: 80.325 / 77.472)", 80.269, 77.052, 80.317, 77.287],
        ["ele-fashion", "H2_hybrid_plus", 87.719, 74.653, 88.033, 76.171],
    ]
    diagnostic_rows = read_csv(DATA / "expert_view_summary.csv")
    dead_count = sum(str(row.get("dead_expert_observed", "")).lower() == "true" for row in diagnostic_rows)
    diag_views = len(diagnostic_rows)
    route_diffs = read_csv(DATA / "expert_route_paired_differences.csv")
    route_full_pairs = [row for row in route_diffs if int(row.get("n_paired_seeds", 0)) == 3]
    route_consistent = sum(str(row.get("same_direction_all_three", "")).lower() == "true" for row in route_full_pairs)
    main = f"""# MR-CGE V1.2 Multi-seed Confirmation

Status: **{manifest.get('completed_new_cells', 0)}/18 new formal cells complete**. The seed-42 V1.2 rows remain read-only historical evidence. New training uses the frozen V1.2 model, trainer, configuration, data and seed-42 split; no V1.3 changes were made.

## Cell accounting

{md_table(['Completed cells', 'Failed cells', 'Resource-blocked cells', 'Unstarted cells', 'GPU preflights pass / blocked / failed', 'New checkpoint pairs reverified'], [[status_counts['complete'], status_counts['failed'], status_counts['blocked_resource'], status_counts['unstarted'], f"{gpu_counts['pass']} / {gpu_counts['blocked_resource']} / {gpu_counts['failed']}", checkpoint_ok]])}

This report uses Best Validation Accuracy checkpoints for its primary Accuracy/Macro-F1/CE values. Best-F1 checkpoint values remain in their separate columns and are not mixed into the primary table. Test evaluation is disabled throughout.

## Three-seed M summary

Values are mean ± sample SD across available seeds 42, 43 and 44, shown as Accuracy % / Macro-F1 %. Missing seeds are not imputed.

{md_table(['Dataset', 'M_repaired Acc / F1', 'M_anchor Acc / F1'], seed_table)}

Mean Validation CE ± sample SD:

{md_table(['Dataset', 'M_repaired CE', 'M_anchor CE'], ce_table)}

## Paired M_anchor − M_repaired differences

Accuracy and Macro-F1 deltas are percentage points. The `positive pairs` count reports seeds with a positive anchor-minus-repaired delta. These are descriptive paired differences; three seeds do not establish statistical significance.

{md_table(['Dataset', 'Acc mean Δ pp', 'Acc range', 'Acc + / − seeds', 'F1 mean Δ pp', 'F1 range', 'F1 + / − seeds', 'CE mean Δ'], paired_table)}

Equal-weighted mean of available dataset-level paired deltas: Accuracy **{equal_weighted['delta_val_acc_pp_anchor_minus_repaired']:.3f} pp**; Macro-F1 **{equal_weighted['delta_val_macro_f1_pp_anchor_minus_repaired']:.3f} pp**. This is descriptive and does not determine the model choice.

## ele-fashion seed-42 S/F/M comparison

{md_table(['Condition', 'Val Accuracy %', 'Val Macro-F1 %', 'Val CE', 'Source', 'Accuracy checkpoint SHA verified'], fashion_table)}

## Historical H2 / DiP context

{md_table(['Dataset', 'H2 reference', 'H2 Acc %', 'H2 F1 %', 'DiP Acc %', 'DiP F1 %'], historical)}

These are cross-campaign reference means, not matched controls. H2/DiP runs came from other campaigns; some baseline implementations were adapted to the shared protocol. They do not support SOTA or significance claims. Source records: `research/hybrid_cge_v11/PILOT_REPORT.md`, `research/hybrid_cge_paper_evidence_v1/BASELINE_REPORT.md`, and its `data/condition_summary.csv`.

## Expert and view diagnostics

The descriptive summary contains {diag_views} dataset/condition/view rows. Rows with a zero Top-2 share in at least one selected checkpoint/view: {dead_count}. `expert_route_paired_differences.csv` reports same-seed Top-2 share changes; {route_consistent} of {len(route_full_pairs)} fully paired dataset/view/expert combinations keep the same direction across all three seeds. `response_comparison.csv` and `expert_view_summary.csv` retain expert theta, effective coefficients, physical/semantic response RMS, rho/gamma, Early Physical Cross, and Late selector/residual summaries where recorded. These are descriptive diagnostics and do not identify causal contributions.

## Protocol and limits

- Fixed baseline commit: `{BASE_SHA}`. Confirmation branch: `exp/mrcge_v12_multiseed_confirmation`.
- Seeds 43/44 use the existing seed-42 splits; observed split fingerprints: `{', '.join(split_hashes) or 'pending'}`.
- The paired initialization audit compares actual state tensors: M conditions differ only at `filter_bank.theta[0]`; Fashion S/F states and initial logits agree within the recorded tolerance.
- Checkpoint SHA, reload metrics, model/trainer/config/data provenance, Validation RNG, and Test-off are recorded per completed cell. Historical checkpoint availability is reported separately.
- No expert collaboration causal claim, statistical significance claim, Test metric, or V1.3 experiment is made.

## Raw evidence

`data/new_runs.csv`, `data/combined_selected_runs.csv`, `data/epoch_history.csv`, `data/paired_differences.csv`, `data/dataset_summary.csv`, `data/ele_fashion_comparison.csv`, `data/expert_view_summary.csv`, `data/expert_route_paired_differences.csv`, `data/response_comparison.csv`, `data/resources.csv`, `data/preflight.json`, `data/failures.json`, and `data/manifest.json`.
"""
    (STUDY / "MULTISEED_REPORT.md").write_text(main, encoding="utf-8")

    freeze = f"""# Design Freeze

- Base: `{BASE_SHA}` (`exp/mrcge_v12_anchor_performance_screen`).
- Confirmation branch: `exp/mrcge_v12_multiseed_confirmation`.
- Training source SHA at CPU preflight: `{freeze_sha}`.
- Model SHA256: `{src_hashes.get('model', '')}`.
- Trainer SHA256 (`src/tasks/nc.py`): `{src_hashes.get('trainer', '')}`.
- Inference SHA256: `{src_hashes.get('inference', '')}`.
- Model/task configuration and dataset loader/split hashes are recorded in `data/preflight.json`.
- The frozen model, trainer, inference, model/task config, data loader, graph construction, semantic cache and split implementation are unchanged from the base commit.
- Formal matrix: 16 M cells (four datasets × two M conditions × seeds 43/44), plus ele-fashion S/F seed 42.
- No Response Strength, architectural edits, tuning, Test evaluation, LP, or seed-42 M retraining.
"""
    (STUDY / "DESIGN_FREEZE.md").write_text(freeze, encoding="utf-8")

    implementation = f"""# Implementation Audit

The independent runner is `scripts/run_mrcge_v12_confirmation.py`; the analyzer is `scripts/analyze_mrcge_v12_confirmation.py`. New writes use `outputs/mrcge_v12_confirmation/` and `research/mrcge_v12_confirmation/`. The seed-42 V1.2 output and research folders are read-only inputs.

The runner accepts dataset, condition, and seed filters, reserves unique attempt directories, verifies checkpoints before skipping completed cells, and records failures/guard blocks without overwriting attempts. All formal and preflight GPU work uses physical GPU 1 under the copied V1.2 ownership and 2-GiB headroom guard.

The CPU preflight compares actual M state tensors and Fashion S/F initial logits. The GPU single-step preflight checks finite losses/gradients, initialization-seeded forward/backward behavior, RNG, split/cache fingerprints, Test-off, and resource peaks. Each formal worker reloads both selected checkpoints and reproduces Validation metrics before recording completion.

"""
    (STUDY / "IMPLEMENTATION_AUDIT.md").write_text(implementation, encoding="utf-8")

    gpu_summary = "\n".join(
        f"- `{row.get('dataset')}/{row.get('condition')}/seed{row.get('seed')}`: **{row.get('status')}**; {row.get('reason', 'single-step preflight record present')}"
        for row in gpu_rows
    ) or "- No GPU single-step attempt has been recorded yet."
    preflight_report = f"""# Preflight Report

- CPU/data status: **{preflight.get('cpu_status', 'not run')}** across {len(preflight.get('datasets', []))} dataset×seed groups.
- Initialization audits: {sum(row.get('status') == 'pass' for row in preflight.get('initialization_audits', []))}/{len(preflight.get('initialization_audits', []))} passed.
- CPU attempt history:
{cpu_attempt_text}
- GPU preflight status counts: {gpu_counts}.
- Test evaluation: `{preflight.get('test_evaluation', False)}`.
- Source hashes at CPU preflight: `{json.dumps(src_hashes, sort_keys=True)}`.

## GPU single-step records

{gpu_summary}

The GPU baseline snapshot is in `data/gpu_baseline.json`; per-phase snapshots and required free-memory thresholds are in `data/resources.csv`. A resource block leaves the associated formal cell incomplete and does not authorize use of another GPU or affect other processes.
"""
    (STUDY / "PREFLIGHT_REPORT.md").write_text(preflight_report, encoding="utf-8")

    handoff = f"""# Decision Handoff

The V1.2 multi-seed confirmation screen has {manifest.get('completed_new_cells', 0)} of 18 new cells complete, {manifest.get('failed_attempt_rows', 0)} failed attempt rows, and {manifest.get('blocked_resource_rows', 0)} resource-blocked rows. Review `MULTISEED_REPORT.md` and the paired/raw data before making any model choice.

This handoff does not select M_anchor or M_repaired. It does not authorize or start V1.3, Node–View Reliability Strength, new seeds, tuning, or another ablation. The next phase waits for independent review.
"""
    (STUDY / "DECISION_HANDOFF.md").write_text(handoff, encoding="utf-8")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    selected, duplicate_audit = choose_selected_runs()
    summaries, differences, paired_summaries = dataset_summaries(selected)
    write_csv(DATA / "combined_selected_runs.csv", selected)
    write_csv(DATA / "dataset_summary.csv", summaries)
    write_csv(DATA / "paired_differences.csv", differences)
    write_csv(DATA / "paired_summary.csv", paired_summaries)

    fashion_rows = [row for row in selected if row.get("dataset") == "ele-fashion" and row.get("seed") == "42"]
    write_csv(DATA / "ele_fashion_comparison.csv", fashion_rows)
    write_csv(DATA / "expert_view_summary.csv", expert_summaries(selected))
    write_csv(DATA / "expert_route_paired_differences.csv", paired_route_differences(selected))
    write_csv(DATA / "response_comparison.csv", response_comparison(selected))
    write_csv(DATA / "duplicate_complete_attempt_audit.csv", duplicate_audit)

    baselines = [
        {"dataset": "Movies", "h2_reference": "H2_hybrid_plus", "h2_accuracy_pct": 56.779,
         "h2_macro_f1_pct": 49.780, "dip_accuracy_pct": 56.569, "dip_macro_f1_pct": 48.330},
        {"dataset": "Grocery", "h2_reference": "H2_hybrid_plus", "h2_accuracy_pct": 83.856,
         "h2_macro_f1_pct": 75.972, "dip_accuracy_pct": 83.953, "dip_macro_f1_pct": 76.899},
        {"dataset": "Toys", "h2_reference": "H2_physical_3hop", "h2_accuracy_pct": 80.269,
         "h2_macro_f1_pct": 77.052, "dip_accuracy_pct": 80.317, "dip_macro_f1_pct": 77.287,
         "h2_historical_4hop_accuracy_pct": 80.325, "h2_historical_4hop_macro_f1_pct": 77.472},
        {"dataset": "ele-fashion", "h2_reference": "H2_hybrid_plus", "h2_accuracy_pct": 87.719,
         "h2_macro_f1_pct": 74.653, "dip_accuracy_pct": 88.033, "dip_macro_f1_pct": 76.171},
    ]
    write_csv(DATA / "historical_baselines.csv", baselines)

    run_rows = read_csv(DATA / "runs.csv")
    preflight = json.loads((DATA / "preflight.json").read_text(encoding="utf-8")) if (DATA / "preflight.json").is_file() else {}
    events = []
    for row in preflight.get("gpu_single_step", []):
        events.append({"event_id": f"gpu_{row.get('attempt')}_{row.get('dataset')}_{row.get('condition')}_{row.get('seed')}",
                       "stage": row.get("stage", "gpu_single_step"), **row})
    for row in run_rows:
        events.append({"event_id": f"train_{row.get('dataset')}_{row.get('condition')}_{row.get('seed')}_{row.get('attempt')}",
                       "stage": "formal_training", **row})
    write_csv(DATA / "new_runs.csv", events)

    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    expected = {(dataset, condition, seed) for dataset in DATASETS for condition in M_CONDITIONS
                for seed in (42, 43, 44)}
    expected |= {("ele-fashion", condition, 42) for condition in ("S_repaired", "F_sinit")}
    new_expected = {(dataset, condition, seed) for dataset in DATASETS for condition in M_CONDITIONS
                    for seed in (43, 44)}
    new_expected |= {("ele-fashion", condition, 42) for condition in ("S_repaired", "F_sinit")}
    present = {(row["dataset"], row["condition"], int(row["seed"])) for row in selected}
    all_attempts: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        all_attempts[(row.get("dataset", ""), row.get("condition", ""), int(row.get("seed", -1)))].append(row)
    complete_new = {(row["dataset"], row["condition"], int(row["seed"])) for row in selected
                    if row.get("source_campaign") == "confirmation_new_run"
                    and row.get("checkpoint_accuracy_verified") is True
                    and row.get("checkpoint_best_f1_verified") is True}
    failed_keys = {(row.get("dataset", ""), row.get("condition", ""), int(row.get("seed", -1)))
                   for row in run_rows if row.get("status") == "failed"}
    blocked_keys = {(row.get("dataset", ""), row.get("condition", ""), int(row.get("seed", -1)))
                    for row in run_rows if row.get("status") == "blocked_resource"}
    latest_preflight = {}
    for row in preflight.get("gpu_single_step", []):
        key = (row.get("dataset", ""), row.get("condition", ""), int(row.get("seed", -1)))
        if key in new_expected and key not in complete_new and key not in failed_keys and key not in blocked_keys:
            if key not in latest_preflight or int(row.get("attempt", 0)) >= int(latest_preflight[key].get("attempt", 0)):
                latest_preflight[key] = row
    failed_keys |= {key for key, row in latest_preflight.items() if row.get("status") == "failed"}
    blocked_keys |= {key for key, row in latest_preflight.items() if row.get("status") == "blocked_resource"}
    failed_keys -= complete_new
    blocked_keys -= complete_new | failed_keys
    unstarted_keys = new_expected - complete_new - failed_keys - blocked_keys
    manifest.update({
        "base_sha": BASE_SHA, "combined_selected_rows": len(selected),
        "expected_combined_rows": len(expected), "missing_qualified_cells": [
            {"dataset": d, "condition": c, "seed": s} for d, c, s in sorted(expected - present)
        ],
        "completed_new_cells": len(complete_new),
        "failed_new_cells": len(failed_keys), "blocked_new_cells": len(blocked_keys),
        "unstarted_new_cells": len(unstarted_keys),
        "completed_cells": len(complete_new), "failed_cells": len(failed_keys),
        "blocked_resource_cells": len(blocked_keys), "unstarted_cells": len(unstarted_keys),
        "failed_attempt_rows": sum(row.get("status") == "failed" for row in run_rows),
        "blocked_resource_rows": sum(row.get("status") == "blocked_resource" for row in run_rows),
        "unstarted_requested_cells": len(unstarted_keys),
        "gpu_preflight_cell_records": len(preflight.get("gpu_single_step", [])),
        "test_evaluation": False,
        "history_checkpoints_reverified": sum(row.get("source_campaign") == "V1.2_seed42_history"
                                               and row.get("checkpoint_accuracy_verified") is True
                                               and row.get("checkpoint_best_f1_verified") is True for row in selected),
        "last_analysis_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    })
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_reports(selected, summaries, differences, paired_summaries, manifest)
    print(json.dumps({"selected_rows": len(selected), "expected_rows": len(expected),
                      "new_run_rows": len(run_rows), "missing_cells": len(expected - present),
                      "manifest": str(DATA / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
