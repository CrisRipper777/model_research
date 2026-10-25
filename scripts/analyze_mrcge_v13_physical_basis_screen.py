#!/usr/bin/env python3
"""Build fixed-split and NoP0 descriptive comparisons from immutable run rows."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mrcge_v13_physical_basis_screen import (
    BASE_SHA, DATA, EXPECTED_MOVIES_SPLIT, HYBRID_PAPER, NO_P0_CONDITION,
    NO_P0_MODEL, V13_MODEL,
    read_json, read_rows, sha256_file, verify_protected_history_file_hashes, write_rows,
)

METRICS = ("val_acc", "val_macro_f1", "val_ce")
RUN_FIELDS = ("model", "condition", "dataset", "seed", "attempt", "status")


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        raise ValueError(f"missing numeric field {key} in row {row}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {key} in row {row}")
    return result


def latest_complete(rows: list[dict[str, str]], key: tuple[str, str, str, int]) -> dict[str, str]:
    model, condition, dataset, seed = key
    matches = [row for row in rows if row.get("model") == model and row.get("condition") == condition
               and row.get("dataset") == dataset and int(row.get("seed", -1)) == seed
               and row.get("status") == "complete"]
    if not matches:
        raise RuntimeError(f"no successful run row for {key}")
    selected = max(matches, key=lambda row: int(row.get("attempt", 0) or 0))
    if selected.get("test_evaluation", "False").lower() not in ("false", "0", ""):
        raise RuntimeError(f"Test evaluation is not disabled for {key}")
    if selected.get("checkpoint_reload_verified", "False").lower() not in ("true", "1"):
        raise RuntimeError(f"checkpoint reload was not verified for {key}")
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                               ("best_f1_checkpoint_path", "best_f1_checkpoint_sha256")):
        path = Path(selected.get(path_key, ""))
        if not path.is_file() or sha256_file(path) != selected.get(hash_key):
            raise RuntimeError(f"checkpoint record failed local SHA256 audit for {key}: {path}")
    return selected


def v13_history(dataset: str, seed: int) -> dict[str, str]:
    rows = read_rows(ROOT / "research/mrcge_v13/data/runs.csv")
    matches = [row for row in rows if row.get("dataset") == dataset and row.get("variant") == "joint_view"
               and int(row.get("seed", -1)) == seed and row.get("status") == "complete"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen V1.3-A history row for {dataset}/seed{seed}")
    return matches[0]


def hybrid_history_seed42() -> dict[str, str]:
    rows = read_rows(ROOT / "research/hybrid_cge_paper_evidence_v1/data/runs.csv")
    source_sha = sha256_file(ROOT / "src/models/hybrid_cge_paper_v1.py")
    matches = [row for row in rows if row.get("dataset") == "Movies" and row.get("model") == HYBRID_PAPER
               and row.get("paper_condition") == "H2_physical_3hop" and row.get("seed") == "42"
               and row.get("status") == "complete" and row.get("model_source_sha256") == source_sha]
    if len(matches) != 1:
        raise RuntimeError("corrected Hybrid-CGE H2 seed42 reference is missing or ambiguous")
    return matches[0]


def assert_common_fixed_split(rows: list[dict[str, str]]) -> None:
    fingerprints = {row.get("split_fingerprint") for row in rows}
    if len(fingerprints) != 1 or "" in fingerprints:
        raise RuntimeError(f"fixed-split comparison contains split mismatch: {sorted(fingerprints)}")
    if next(iter(fingerprints)) != EXPECTED_MOVIES_SPLIT:
        raise RuntimeError("Movies fixed-split comparison differs from the authorized seed42 split")


def make_fixed_split_comparison(runs: list[dict[str, str]]) -> list[dict[str, object]]:
    hybrid42 = hybrid_history_seed42()
    result: list[dict[str, object]] = []
    for seed in (42, 43, 44):
        if seed == 42:
            h = hybrid42
            h_record = {
                "model": HYBRID_PAPER, "condition": "H2_physical_3hop", "dataset": "Movies", "seed": seed,
                "attempt": h["attempt"], "status": "historical_reference",
                "source_git_sha": h.get("source_git_sha"), "split_fingerprint": h.get("split_fingerprint"),
                "val_acc": h.get("val_acc"), "val_macro_f1": h.get("val_macro_f1"), "val_ce": h.get("val_ce"),
                "best_f1_val_acc": h.get("best_f1_val_acc"), "best_f1_val_macro_f1": h.get("best_f1_val_macro_f1"),
                "best_f1_val_ce": h.get("best_f1_val_ce"), "train_acc": h.get("train_acc"),
                "train_macro_f1": h.get("train_macro_f1"), "train_ce": h.get("train_ce"),
                "train_val_acc_gap": h.get("train_val_acc_gap"), "best_acc_epoch": h.get("best_accuracy_epoch"),
                "best_f1_epoch": h.get("best_f1_epoch"), "model_parameters": h.get("model_parameters"),
                "classifier_parameters": h.get("classifier_parameters"), "total_parameters": h.get("parameter_count"),
                "wall_seconds": h.get("wall_seconds"), "peak_allocated_bytes": h.get("peak_allocated_bytes"),
                "peak_reserved_bytes": h.get("peak_reserved_bytes"), "source": "corrected_historical_attempt",
            }
        else:
            run = latest_complete(runs, (HYBRID_PAPER, "H2_physical_3hop", "Movies", seed))
            h_record = {**run, "source": "new_fixed_split_attempt"}
        v = v13_history("Movies", seed) if seed in (42, 43) else latest_complete(
            runs, (V13_MODEL, "joint_view", "Movies", seed))
        if seed in (42, 43):
            v_record = {
                "model": V13_MODEL, "condition": "joint_view", "dataset": "Movies", "seed": seed,
                "attempt": v.get("attempt"), "status": "historical_reference",
                "source_git_sha": v.get("source_git_sha"), "split_fingerprint": v.get("split_fingerprint"),
                "val_acc": v.get("val_acc"), "val_macro_f1": v.get("val_macro_f1"), "val_ce": v.get("val_ce"),
                "best_f1_val_acc": v.get("best_f1_val_acc"), "best_f1_val_macro_f1": v.get("best_f1_val_macro_f1"),
                "best_f1_val_ce": v.get("best_f1_val_ce"), "train_acc": v.get("train_acc"),
                "train_macro_f1": v.get("train_macro_f1"), "train_ce": v.get("train_ce"),
                "best_acc_epoch": v.get("best_acc_epoch"), "best_f1_epoch": v.get("best_f1_epoch"),
                "model_parameters": v.get("model_parameters"), "classifier_parameters": v.get("classifier_parameters"),
                "total_parameters": v.get("total_parameters"), "wall_seconds": v.get("wall_seconds"),
                "peak_allocated_bytes": v.get("peak_allocated_bytes"), "peak_reserved_bytes": v.get("peak_reserved_bytes"),
                "source": "frozen_v13_a_history",
            }
        else:
            v_record = {**v, "source": "new_fixed_split_attempt"}
        result.extend((h_record, v_record))
    assert_common_fixed_split(result)
    by_seed = {(row["model"], int(row["seed"])): row for row in result}
    for row in result:
        seed = int(row["seed"])
        peer = by_seed[(V13_MODEL if row["model"] == HYBRID_PAPER else HYBRID_PAPER, seed)]
        for metric in METRICS:
            row[f"delta_vs_{peer['model']}_{metric}"] = number(row, metric) - number(peer, metric)
    return result


def summary_rows(fixed_rows: list[dict[str, object]], paired: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    groups = [("Movies", HYBRID_PAPER, "H2_physical_3hop", (42, 43, 44)),
              ("Movies", V13_MODEL, "joint_view", (42, 43, 44))]
    for dataset, model, condition, seeds in groups:
        selected = [row for row in fixed_rows if row.get("dataset") == dataset and row.get("model") == model
                    and row.get("condition") == condition and int(row.get("seed", -1)) in seeds]
        if len(selected) != len(seeds):
            raise RuntimeError(f"summary group incomplete: {dataset}/{model}/{condition}")
        record: dict[str, object] = {"dataset": dataset, "model": model, "condition": condition,
                                     "n": len(selected), "seeds": ",".join(map(str, seeds))}
        for metric in METRICS:
            values = [number(row, metric) for row in selected]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_population_sd"] = statistics.pstdev(values)
        for metric in ("train_acc", "train_ce", "wall_seconds", "peak_reserved_bytes", "total_parameters"):
            values = [number(row, metric) for row in selected if row.get(metric) not in (None, "")]
            record[f"{metric}_mean"] = statistics.mean(values) if values else ""
        rows.append(record)
    fixed_deltas = []
    for seed in (42, 43, 44):
        h = next(row for row in fixed_rows if row.get("model") == HYBRID_PAPER and int(row.get("seed", -1)) == seed)
        v = next(row for row in fixed_rows if row.get("model") == V13_MODEL and int(row.get("seed", -1)) == seed)
        fixed_deltas.append({metric: number(h, metric) - number(v, metric) for metric in METRICS})
    fixed_delta_record: dict[str, object] = {"dataset": "Movies", "model": "paired_difference",
        "condition": "Hybrid-CGE minus V1.3-A", "n": 3, "seeds": "42,43,44"}
    for metric in METRICS:
        values = [row[metric] for row in fixed_deltas]
        fixed_delta_record[f"delta_{metric}_mean"] = statistics.mean(values)
        fixed_delta_record[f"delta_{metric}_population_sd"] = statistics.pstdev(values)
    rows.append(fixed_delta_record)
    for dataset in ("Movies", "Grocery"):
        delta_rows = [row for row in paired if row.get("dataset") == dataset]
        if len(delta_rows) != 2:
            raise RuntimeError(f"expected two paired NoP0 differences for {dataset}")
        for which, prefix, model, condition in (("baseline", "baseline_", V13_MODEL, "joint_view"),
                                                 ("no_p0", "no_p0_", NO_P0_MODEL, NO_P0_CONDITION)):
            record = {"dataset": dataset, "model": model, "condition": condition,
                      "n": 2, "seeds": "42,43"}
            for metric in METRICS:
                values = [number(row, f"{prefix}{metric}") for row in delta_rows]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_population_sd"] = statistics.pstdev(values)
            for metric in ("train_acc", "train_ce", "wall_seconds", "peak_reserved_bytes", "total_parameters"):
                values = [number(row, f"{prefix}{metric}") for row in delta_rows
                          if row.get(f"{prefix}{metric}") not in (None, "")]
                record[f"{metric}_mean"] = statistics.mean(values) if values else ""
            rows.append(record)
        record = {"dataset": dataset, "model": "paired_difference", "condition": "NoP0 minus V1.3-A",
                  "n": 2, "seeds": "42,43"}
        for metric in METRICS:
            values = [number(row, f"delta_{metric}") for row in delta_rows]
            record[f"delta_{metric}_mean"] = statistics.mean(values)
            record[f"delta_{metric}_population_sd"] = statistics.pstdev(values)
        rows.append(record)
    return rows


def analyze() -> dict[str, object]:
    if not (DATA / "preflight.json").is_file():
        raise RuntimeError("preflight record is missing")
    preflight = read_json(DATA / "preflight.json", {})
    protected = preflight.get("protected_historical_research_sha256s")
    if not protected:
        raise RuntimeError("CPU preflight lacks the protected-history hash baseline")
    verify_protected_history_file_hashes(protected)
    runs = read_rows(DATA / "runs.csv")
    fixed = make_fixed_split_comparison(runs)
    paired = []
    no_p0_runs = {}
    for dataset in ("Movies", "Grocery"):
        for seed in (42, 43):
            model_row = latest_complete(runs, (NO_P0_MODEL, NO_P0_CONDITION, dataset, seed))
            baseline = v13_history(dataset, seed)
            if model_row.get("split_fingerprint") != baseline.get("split_fingerprint"):
                raise RuntimeError(f"NoP0 and V1.3-A split mismatch for {dataset}/seed{seed}")
            no_p0_runs[(dataset, seed)] = model_row
            record: dict[str, object] = {"dataset": dataset, "seed": seed,
                "model": NO_P0_MODEL, "condition": NO_P0_CONDITION,
                "baseline_model": V13_MODEL, "baseline_condition": "joint_view",
                "split_fingerprint": model_row["split_fingerprint"],
                "no_p0_attempt": model_row["attempt"], "baseline_attempt": baseline.get("attempt", "historical"),
                "train_val_acc_gap_no_p0": model_row.get("train_val_acc_gap", ""),
                "train_val_ce_gap_no_p0": model_row.get("train_val_ce_gap", ""),
                "no_p0_train_acc": model_row.get("train_acc", ""),
                "no_p0_train_ce": model_row.get("train_ce", ""),
                "no_p0_wall_seconds": model_row.get("wall_seconds", ""),
                "no_p0_peak_reserved_bytes": model_row.get("peak_reserved_bytes", ""),
                "no_p0_total_parameters": model_row.get("total_parameters", ""),
                "baseline_train_acc": baseline.get("train_acc", ""),
                "baseline_train_ce": baseline.get("train_ce", ""),
                "baseline_train_val_acc_gap": (float(baseline["train_acc"]) - float(baseline["val_acc"])
                                                if baseline.get("train_acc") and baseline.get("val_acc") else ""),
                "baseline_train_val_ce_gap": (float(baseline["train_ce"]) - float(baseline["val_ce"])
                                               if baseline.get("train_ce") and baseline.get("val_ce") else ""),
                "baseline_wall_seconds": baseline.get("wall_seconds", ""),
                "baseline_peak_reserved_bytes": baseline.get("peak_reserved_bytes", ""),
                "baseline_total_parameters": baseline.get("total_parameters", ""),
                "test_evaluation": False}
            for metric in METRICS:
                record[f"baseline_{metric}"] = number(baseline, metric)
                record[f"no_p0_{metric}"] = number(model_row, metric)
                record[f"delta_{metric}"] = number(model_row, metric) - number(baseline, metric)
            paired.append(record)
    summary = summary_rows(fixed, paired)
    write_rows(DATA / "fixed_split_hybrid_comparison.csv", fixed,
               list(dict.fromkeys(key for row in fixed for key in row)))
    write_rows(DATA / "paired_differences.csv", paired,
               list(dict.fromkeys(key for row in paired for key in row)))
    write_rows(DATA / "dataset_summary.csv", summary,
               list(dict.fromkeys(key for row in summary for key in row)))
    result = {"status": "complete", "analyzed_utc": __import__("datetime").datetime.now().astimezone().isoformat(),
        "base_sha": BASE_SHA, "source_git_sha": preflight.get("source_git_sha"),
        "fixed_split_comparison_rows": fixed, "paired_differences": paired,
        "dataset_summary": summary, "test_evaluation": False,
        "protected_history_files_verified": len(protected),
        "interpretation_limits": [
            "Fixed split and same training seed improve comparability but do not pair parameter initialization across different model parameterizations.",
            "NoP0 changes the effective multi-order filter and its coupled Early Physical Cross response; the comparison does not isolate a causal zero-order effect.",
            "Means and population standard deviations are descriptive only; no significance or winner declaration is made.",
        ]}
    (DATA / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    output = analyze()
    print(json.dumps({"status": output["status"], "fixed_rows": len(output["fixed_split_comparison_rows"]),
                      "paired_rows": len(output["paired_differences"])}, ensure_ascii=False))
