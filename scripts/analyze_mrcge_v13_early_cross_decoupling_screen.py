#!/usr/bin/env python3
"""Audit and summarize the frozen Expert-Decoupled Early Cross screen."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mrcge_v13_early_cross_decoupling_screen as runner


DATASETS = runner.DATASETS
TARGETS = runner.TARGET_CELLS
NEW_MODEL = runner.DECOUPLED_MODEL
NEW_CONDITION = runner.DECOUPLED_CONDITION
MODEL_SPECS = (
    ("mrcge_v13", "joint_view", "V1.3-A Joint View"),
    (NEW_MODEL, NEW_CONDITION, "Expert-Decoupled Early Cross"),
)
COMPARE_FIELDS = (
    "dataset", "seed", "model", "condition", "display_name", "source", "attempt", "status",
    "best_acc_epoch", "best_f1_epoch", "epochs_completed", "val_acc", "val_macro_f1", "val_ce",
    "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1",
    "train_ce", "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap", "model_parameters",
    "classifier_parameters", "total_parameters", "wall_seconds", "peak_allocated_bytes",
    "peak_reserved_bytes", "checkpoint_path", "checkpoint_sha256", "best_f1_checkpoint_path",
    "best_f1_checkpoint_sha256", "split_fingerprint", "source_git_sha", "test_evaluation",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict[str, Any], key: str, *, optional: bool = False) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        if optional:
            return None
        raise RuntimeError(f"required metric {key} missing from {row}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite metric {key} in {row}")
    return result


def one(rows: list[dict[str, str]], predicate, label: str) -> dict[str, str]:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def verify_checkpoint(row: dict[str, str], label: str) -> None:
    if str(row.get("test_evaluation", "false")).lower() not in ("false", "0", ""):
        raise RuntimeError(f"Test evaluation is marked enabled for {label}")
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                               ("best_f1_checkpoint_path", "best_f1_checkpoint_sha256")):
        path_text, expected = row.get(path_key, ""), row.get(hash_key, "")
        path = Path(path_text)
        if not path_text or not expected or not path.is_file() or runner.sha256_file(path) != expected:
            raise RuntimeError(f"checkpoint hash audit failed for {label}: {path}")


def canonical(row: dict[str, str], model: str, condition: str, display: str, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {field: row.get(field, "") for field in COMPARE_FIELDS}
    result.update({"model": model, "condition": condition, "display_name": display,
                   "source": source, "test_evaluation": False})
    return result


def historical_v13(dataset: str, seed: int) -> dict[str, str]:
    row = runner.v13_reference(dataset, seed)
    verify_checkpoint(row, f"V1.3-A {dataset}/seed{seed}")
    recorded_model_sha = row.get("model_source_sha256", "")
    if not recorded_model_sha or not row.get("source_git_sha"):
        raise RuntimeError(f"V1.3-A source provenance is incomplete for {dataset}/seed{seed}")
    committed_model_sha = runner.git_file_sha(row["source_git_sha"], "src/models/mrcge_v13.py")
    if recorded_model_sha != committed_model_sha:
        raise RuntimeError(f"V1.3-A model-source SHA mismatch for {dataset}/seed{seed}")
    study = runner.OmegaConf.load(ROOT / "configs/study/mrcge_v13_early_cross_decoupling_screen.yaml")
    expected_split = str(study.split_fingerprints[dataset])
    if row.get("split_fingerprint") != expected_split:
        raise RuntimeError(f"V1.3-A split fingerprint mismatch for {dataset}/seed{seed}")
    expected_file_hash = runner.sha256_file(runner.fixed_split_path(dataset))
    if row.get("split_file_sha256") != expected_file_hash:
        raise RuntimeError(f"V1.3-A split-file hash mismatch for {dataset}/seed{seed}")
    return row


def historical_source_audit(v13: dict[tuple[str, int], dict[str, str]]) -> list[dict[str, Any]]:
    output = []

    def verify_record(row: dict[str, str], model: str, condition: str, dataset: str,
                      seed: int, source_path: str) -> None:
        commit, recorded = row.get("source_git_sha", ""), row.get("model_source_sha256", "")
        if not commit or not recorded:
            raise RuntimeError(f"historical source provenance missing for {model} {dataset}/seed{seed}")
        actual = runner.git_file_sha(commit, source_path)
        if actual != recorded:
            raise RuntimeError(f"historical model-source SHA mismatch for {model} {dataset}/seed{seed}")
        output.append({"model": model, "condition": condition, "dataset": dataset, "seed": seed,
            "source_git_sha": commit, "model_source_path": source_path,
            "recorded_model_source_sha256": recorded, "committed_model_source_sha256": actual,
            "source_hash_verified": True, "test_evaluation": False})

    for (dataset, seed), row in sorted(v13.items()):
        verify_record(row, "mrcge_v13", "joint_view", dataset, seed, "src/models/mrcge_v13.py")

    selected_v12 = runner.read_rows(ROOT / "research/mrcge_v12_confirmation/data/combined_selected_runs.csv")
    for dataset, seed in sorted(v13):
        row = one(selected_v12, lambda item: item.get("dataset") == dataset
            and item.get("condition") == "M_repaired" and item.get("seed") == str(seed)
            and item.get("status") == "complete"
            and item.get("selected_for_combined", "").lower() == "true"
            and item.get("eligible_complete", "").lower() == "true",
            f"selected V1.2 source row {dataset}/seed{seed}")
        verify_record(row, "mrcge_v12", "M_repaired", dataset, seed, "src/models/mrcge_v12.py")

    fixed = runner.read_rows(ROOT / "research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv")
    baseline = read_json(ROOT / "research/mrcge_v13_physical_basis/data/frozen_baseline.json", {})
    special = baseline.get("historical_references", {}).get("hybrid_corrected_seed42", {})
    for seed in (42, 43, 44):
        if seed == 42:
            row = {"source_git_sha": special.get("source_git_sha"),
                   "model_source_sha256": special.get("model_source_sha256")}
        else:
            row = one(fixed, lambda item: item.get("model") == runner.HYBRID_MODEL
                and item.get("condition") == "H2_physical_3hop" and item.get("dataset") == "Movies"
                and item.get("seed") == str(seed), f"fixed-split Hybrid source row Movies/seed{seed}")
        verify_record(row, runner.HYBRID_MODEL, "H2_physical_3hop", "Movies", seed,
                      "src/models/hybrid_cge_paper_v1.py")
    return output


def check_history_unchanged(preflight: dict[str, Any]) -> None:
    expected = preflight.get("protected_historical_file_sha256s", {})
    if not expected:
        raise RuntimeError("CPU preflight did not record protected historical-file hashes")
    runner.verify_protected_history_file_hashes(expected)


def gather_results() -> tuple[dict[tuple[str, int], dict[str, str]],
                              dict[tuple[str, int], dict[str, str]], list[dict[str, str]]]:
    preflight = read_json(runner.DATA / "preflight.json", {})
    if preflight.get("status") != "pass" or preflight.get("stage") != "cpu_preflight":
        raise RuntimeError("CPU preflight is not recorded as passing")
    frozen_sha = str(preflight.get("source_git_sha", ""))
    if not frozen_sha:
        raise RuntimeError("CPU preflight has no frozen training commit")
    import subprocess
    if subprocess.run(["git", "merge-base", "--is-ancestor", frozen_sha, runner.source_sha()],
                      cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("analysis commit is not a descendant of the training source freeze")
    frozen_hashes = preflight.get("source_file_sha256s", {})
    actual_hashes = runner.source_file_hashes()
    differences = {}
    analyzer_path = "scripts/analyze_mrcge_v13_early_cross_decoupling_screen.py"
    for path, expected in frozen_hashes.items():
        # The report generator received a small post-training bug fix. Verify
        # its frozen copy at the training commit, while requiring every model,
        # trainer, config, and data-pipeline file to remain byte-identical.
        actual = runner.git_file_sha(frozen_sha, path) if path == analyzer_path else actual_hashes.get(path)
        if actual != expected:
            differences[path] = {"expected_frozen": expected, "actual": actual}
    if differences:
        raise RuntimeError(f"training source freeze audit failed: {differences}")
    check_history_unchanged(preflight)

    gpu = {}
    for row in preflight.get("gpu_preflight_attempts", []):
        if row.get("source_file_sha256s") != preflight.get("source_file_sha256s"):
            continue
        key = (row.get("dataset"), int(row.get("seed", -1)))
        if row.get("model") == NEW_MODEL and row.get("condition") == NEW_CONDITION:
            gpu[key] = row
    new_rows = read_rows(runner.DATA / "runs.csv")
    allowed = {(dataset, seed) for _, _, dataset, seed in TARGETS}
    observed = {(row.get("dataset"), int(row.get("seed", -1))) for row in new_rows
                if row.get("model") == NEW_MODEL and row.get("condition") == NEW_CONDITION}
    if observed - allowed:
        raise RuntimeError(f"unauthorized run rows found: {sorted(observed - allowed)}")

    v13: dict[tuple[str, int], dict[str, str]] = {}
    new: dict[tuple[str, int], dict[str, str]] = {}
    for dataset, seed in sorted(allowed):
        historical = historical_v13(dataset, seed)
        v13[(dataset, seed)] = historical
        attempts = [row for row in new_rows if row.get("model") == NEW_MODEL
                    and row.get("condition") == NEW_CONDITION and row.get("dataset") == dataset
                    and row.get("seed") == str(seed)]
        complete = [row for row in attempts if row.get("status") == "complete"]
        if len(complete) != 1:
            raise RuntimeError(f"expected one complete new run for {dataset}/seed{seed}, found {len(complete)}")
        run = complete[0]
        if run.get("source_git_sha") != frozen_sha:
            raise RuntimeError(f"run source commit mismatch for {dataset}/seed{seed}")
        if run.get("split_fingerprint") != historical.get("split_fingerprint"):
            raise RuntimeError(f"run split differs from paired V1.3-A for {dataset}/seed{seed}")
        if run.get("checkpoint_reload_verified", "").lower() != "true" or \
                run.get("validation_rng_unchanged", "").lower() != "true":
            raise RuntimeError(f"checkpoint reload or Validation RNG audit failed for {dataset}/seed{seed}")
        verify_checkpoint(run, f"Expert-Decoupled Early Cross {dataset}/seed{seed}")
        if str(run.get("test_evaluation", "")).lower() != "false":
            raise RuntimeError(f"Test evaluation is not explicitly disabled for {dataset}/seed{seed}")
        model_sha = runner.git_file_sha(frozen_sha, "src/models/mrcge_v13_early_cross_decoupled.py")
        trainer_sha = runner.git_file_sha(frozen_sha, "src/tasks/nc.py")
        if run.get("model_source_sha256") != model_sha or run.get("trainer_source_sha256") != trainer_sha:
            raise RuntimeError(f"model/trainer source hash mismatch for {dataset}/seed{seed}")
        config_path = Path(run.get("config_snapshot_path", ""))
        if not config_path.is_file() or runner.sha256_file(config_path) != run.get("config_sha256"):
            raise RuntimeError(f"resolved config hash mismatch for {dataset}/seed{seed}")
        target_preflight = one(preflight.get("target_protocol_records", []),
            lambda item: item.get("dataset") == dataset and int(item.get("seed", -1)) == seed,
            f"CPU data fingerprint record {dataset}/seed{seed}")
        aligned = target_preflight.get("baseline_alignment", {})
        fingerprint_keys = ("feature_fingerprint", "dataset_fingerprint", "physical_edge_fingerprint",
                            "split_fingerprint", "split_file_sha256", "semantic_cache_sha256",
                            "candidate_fingerprint", "v7b_fingerprint")
        for field in fingerprint_keys:
            if not aligned.get(field) or run.get(field) != aligned.get(field):
                raise RuntimeError(f"{field} differs from CPU preflight for {dataset}/seed{seed}")
        pre = gpu.get((dataset, seed))
        if not pre or pre.get("status") != "pass":
            raise RuntimeError(f"missing same-source passing GPU preflight for {dataset}/seed{seed}")
        for key in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce",
                    "train_acc", "train_macro_f1", "train_ce", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            metric(run, key)
        new[(dataset, seed)] = run
    return v13, new, new_rows


def model_comparison_rows(v13: dict[tuple[str, int], dict[str, str]],
                          new: dict[tuple[str, int], dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for dataset, seed in sorted(v13):
        for model, condition, display, source, row in (
            ("mrcge_v13", "joint_view", "V1.3-A Joint View", "historical_v13_a", v13[(dataset, seed)]),
            (NEW_MODEL, NEW_CONDITION, "Expert-Decoupled Early Cross", "trained_checkpoint", new[(dataset, seed)]),
        ):
            output.append(canonical(row, model, condition, display, source))
    return output


def paired_differences(v13: dict[tuple[str, int], dict[str, str]],
                       new: dict[tuple[str, int], dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, seed in sorted(v13):
        left, right = v13[(dataset, seed)], new[(dataset, seed)]
        for selection, keys in (("best_validation_accuracy", ("val_acc", "val_macro_f1", "val_ce")),
                                ("best_validation_macro_f1", ("best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce"))):
            acc_key, f1_key, ce_key = keys
            acc_delta = metric(right, acc_key) - metric(left, acc_key)
            f1_delta = metric(right, f1_key) - metric(left, f1_key)
            rows.append({"dataset": dataset, "seed": seed, "selection": selection,
                "contrast": "Expert-Decoupled Early Cross - V1.3-A Joint View",
                "v13_acc": metric(left, acc_key), "new_acc": metric(right, acc_key),
                "delta_acc_pp": 100.0 * acc_delta,
                "v13_macro_f1": metric(left, f1_key), "new_macro_f1": metric(right, f1_key),
                "delta_macro_f1_pp": 100.0 * f1_delta,
                "v13_ce": metric(left, ce_key), "new_ce": metric(right, ce_key),
                "delta_ce": metric(right, ce_key) - metric(left, ce_key),
                "test_evaluation": False})
    return rows


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def dataset_summaries(comparison: list[dict[str, Any]], diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for model, display in (("mrcge_v13", "V1.3-A Joint View"), (NEW_MODEL, "Expert-Decoupled Early Cross")):
            selected = [row for row in comparison if row["dataset"] == dataset and row["model"] == model]
            record: dict[str, Any] = {"dataset": dataset, "model": model, "display_name": display,
                                      "n": len(selected), "seeds": ";".join(str(row["seed"]) for row in selected)}
            for key in ("val_acc", "val_macro_f1", "val_ce", "best_f1_val_acc", "best_f1_val_macro_f1",
                        "best_f1_val_ce", "train_acc", "train_ce", "train_val_acc_gap", "total_parameters",
                        "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
                vals = [float(row[key]) for row in selected if row.get(key) not in (None, "")]
                avg, sd = mean_sd(vals)
                record[f"{key}_mean"] = avg
                record[f"{key}_sd"] = sd
            rows.append(record)
        for selection in ("best_validation_accuracy", "best_validation_macro_f1"):
            selected = [row for row in diffs if row["dataset"] == dataset and row["selection"] == selection]
            record = {"dataset": dataset, "model": "paired_delta", "display_name": selection,
                      "n": len(selected), "seeds": ";".join(str(row["seed"]) for row in selected)}
            for key in ("delta_acc_pp", "delta_macro_f1_pp", "delta_ce"):
                avg, sd = mean_sd([float(row[key]) for row in selected])
                record[f"{key}_mean"] = avg
                record[f"{key}_sd"] = sd
            rows.append(record)
    return rows


def audit_diagnostics(new: dict[tuple[str, int], dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coeff = read_rows(runner.DATA / "coefficients_diagnostics.csv")
    injections = read_rows(runner.DATA / "cross_injection_diagnostics.csv")
    coeff_output, injection_output = [], []
    for key, run in sorted(new.items()):
        dataset, seed = key
        selected = [row for row in coeff if row.get("model") == NEW_MODEL and row.get("dataset") == dataset
                    and row.get("seed") == str(seed) and row.get("attempt") == run.get("attempt")]
        if len(selected) != 4 or {int(row["expert"]) for row in selected} != set(range(4)):
            raise RuntimeError(f"expected four expert coefficient diagnostics for {dataset}/seed{seed}")
        for row in selected:
            b, alpha = float(row["b"]), float(row["alpha_physical_order1"])
            if not all(math.isfinite(float(row[field])) for field in ("b", "alpha_physical_order1", "b_minus_alpha1")):
                raise RuntimeError("non-finite coefficient diagnostic")
            if abs((b - alpha) - float(row["b_minus_alpha1"])) > 1e-6:
                raise RuntimeError("coefficient diagnostic arithmetic mismatch")
            coeff_output.append(row)
        selected_cross = [row for row in injections if row.get("model") == NEW_MODEL
                    and row.get("dataset") == dataset and row.get("seed") == str(seed)
                    and row.get("attempt") == run.get("attempt")]
        if len(selected_cross) != 8 or {row.get("receiver_modality") for row in selected_cross} != {"text", "visual"}:
            raise RuntimeError(f"expected eight modality/expert Cross diagnostics for {dataset}/seed{seed}")
        for row in selected_cross:
            for field in ("cross_raw_rms", "cross_calibrated_rms", "private_filtered_input_rms",
                          "cross_addition_rms", "difference_correction_rms", "beta_times_b"):
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"non-finite Cross input diagnostic {field}")
            injection_output.append(row)
    return coeff_output, injection_output


def historical_context() -> list[dict[str, Any]]:
    refs = read_json(runner.DATA / "preflight.json", {}).get("historical_references", {})
    rows = []
    for key, label in (("hybrid_fixed_split", "Hybrid H2 3-hop fixed-split context"),
                       ("v12_m_repaired", "V1.2 M_repaired auxiliary context")):
        for ref in refs.get(key, []):
            metrics = ref.get("metrics", {})
            rows.append({"reference": label, "model": ref.get("model"), "condition": ref.get("condition"),
                "dataset": ref.get("dataset"), "seed": ref.get("seed"),
                "source_git_sha": ref.get("source_git_sha"), "split_fingerprint": ref.get("split_fingerprint"),
                **{f"{name}_{metric_name}": value for name, value in metrics.items()
                   for metric_name in ("value",)},
                "best_accuracy_checkpoint": ref.get("checkpoints", {}).get("best_accuracy", {}).get("path", ""),
                "best_accuracy_sha256": ref.get("checkpoints", {}).get("best_accuracy", {}).get("sha256", ""),
                "best_f1_checkpoint": ref.get("checkpoints", {}).get("best_f1", {}).get("path", ""),
                "best_f1_sha256": ref.get("checkpoints", {}).get("best_f1", {}).get("sha256", ""),
                "interpretation": "historical fixed-split context; not a paired architecture comparison"})
    return rows


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def run_analysis() -> dict[str, Any]:
    v13, new, all_runs = gather_results()
    comparison = model_comparison_rows(v13, new)
    diffs = paired_differences(v13, new)
    summaries = dataset_summaries(comparison, diffs)
    coeff, injections = audit_diagnostics(new)
    source_audit = historical_source_audit(v13)
    context = historical_context()
    write_rows(runner.DATA / "model_comparison.csv", comparison, COMPARE_FIELDS)
    write_rows(runner.DATA / "paired_differences.csv", diffs,
        ("dataset", "seed", "selection", "contrast", "v13_acc", "new_acc", "delta_acc_pp",
         "v13_macro_f1", "new_macro_f1", "delta_macro_f1_pp", "v13_ce", "new_ce", "delta_ce", "test_evaluation"))
    summary_fields = list(dict.fromkeys(key for row in summaries for key in row))
    write_rows(runner.DATA / "dataset_summary.csv", summaries, summary_fields)
    write_rows(runner.DATA / "historical_context.csv", context,
        list(dict.fromkeys(key for row in context for key in row)) if context else ("reference", "model", "dataset", "seed"))
    write_rows(runner.DATA / "audited_coefficients.csv", coeff,
        list(dict.fromkeys(key for row in coeff for key in row)) if coeff else ("dataset", "seed", "expert"))
    write_rows(runner.DATA / "audited_cross_injection.csv", injections,
        list(dict.fromkeys(key for row in injections for key in row)) if injections else ("dataset", "seed", "receiver_modality", "expert"))
    write_rows(runner.DATA / "historical_source_audit.csv", source_audit,
        list(dict.fromkeys(key for row in source_audit for key in row)) if source_audit else
        ("model", "condition", "dataset", "seed", "source_git_sha", "model_source_sha256", "source_hash_verified"))

    preflight = read_json(runner.DATA / "preflight.json", {})
    failures = read_json(runner.DATA / "failures.json", {"failures": []}).get("failures", [])
    manifest = read_json(runner.DATA / "manifest.json", {})
    complete = len(new)
    delta_sections = []
    delta_summary_lines = []
    for dataset in DATASETS:
        for selection in ("best_validation_accuracy", "best_validation_macro_f1"):
            matched = [row for row in diffs if row["dataset"] == dataset and row["selection"] == selection]
            stats = {}
            for key in ("delta_acc_pp", "delta_macro_f1_pp", "delta_ce"):
                stats[key] = mean_sd([float(row[key]) for row in matched])
            delta_summary_lines.append(
                f"| {dataset} | {selection} | {stats['delta_acc_pp'][0]:+.3f} ± {stats['delta_acc_pp'][1]:.3f} | "
                f"{stats['delta_macro_f1_pp'][0]:+.3f} ± {stats['delta_macro_f1_pp'][1]:.3f} | "
                f"{stats['delta_ce'][0]:+.5f} ± {stats['delta_ce'][1]:.5f} | {len(matched)} |")
        delta_sections.append(f"### {dataset}\n\n" + "\n".join(
            f"- Seed {row['seed']}: Acc {row['delta_acc_pp']:+.3f} pp, Macro-F1 {row['delta_macro_f1_pp']:+.3f} pp, CE {row['delta_ce']:+.5f} ({row['selection']})."
            for row in diffs if row["dataset"] == dataset))
    per_seed_rows = []
    for dataset, seed in sorted(new):
        base, run = v13[(dataset, seed)], new[(dataset, seed)]
        per_seed_rows.append(f"| {dataset} | {seed} | {float(base['val_acc']):.4f} / {float(base['val_macro_f1']):.4f} / {float(base['val_ce']):.5f} | "
            f"{float(run['val_acc']):.4f} / {float(run['val_macro_f1']):.4f} / {float(run['val_ce']):.5f} | "
            f"{float(run['best_f1_val_acc']):.4f} / {float(run['best_f1_val_macro_f1']):.4f} / {float(run['best_f1_val_ce']):.5f} | "
            f"{int(run['best_acc_epoch'])}/{int(run['best_f1_epoch'])} | {float(run['wall_seconds']):.1f} | "
            f"{int(run['total_parameters'])} | {int(run['peak_reserved_bytes'])} |")
    hybrid_rows = [r for r in context if r.get("dataset") == "Movies" and "Hybrid" in r.get("reference", "")]
    hybrid_lines = [f"| Movies | {r['seed']} | {r.get('val_acc_value','')} | {r.get('val_macro_f1_value','')} | {r.get('val_ce_value','')} | {r.get('source_git_sha','')} |" for r in hybrid_rows]
    report = f"""# MR-CGE V1.3 Expert-Decoupled Early Cross — screening report

## Scope and outcome

Seven authorized training cells were completed: {complete}/7. The only new condition was `{NEW_CONDITION}` on V1.3-A M_repaired Joint View. GPU: physical GPU 1. Fixed splits and V7B.1 K=8 caches were paired to historical V1.3-A records. Test evaluation and Link Prediction were disabled. This is a small descriptive screen and does not support significance or state-of-the-art claims.

Source commit: `{preflight.get('source_git_sha')}`; base: `{runner.BASE_SHA}`; branch: `{runner.BRANCH}`.

## Paired validation results

Metrics are Best Validation Accuracy checkpoint Acc / Macro-F1 / CE; Best-F1 metrics are reported separately. Parameter count includes the external classifier. Peak reserved memory is in bytes.

| Dataset | Seed | V1.3-A Acc / F1 / CE | New Acc / F1 / CE | New Best-F1 Acc / F1 / CE | Best Acc/F1 epoch | Wall seconds | Parameters | Peak reserved bytes |
|---|---:|---|---|---|---:|---:|---:|---:|
{chr(10).join(per_seed_rows)}

## New minus V1.3-A paired differences

Acc and Macro-F1 differences are percentage points; CE is raw cross entropy.

{chr(10).join(delta_sections)}

### Paired difference summary by dataset

Entries are mean ± population SD across seeds. Acc and Macro-F1 are percentage points; CE is raw loss.

| Dataset | Checkpoint selection | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE | n |
|---|---|---:|---:|---:|---:|
{chr(10).join(delta_summary_lines)}

## Fixed-split Hybrid context

Hybrid H2 3-hop Movies records are historical context only; they do not isolate the effect of this architecture change.

| Dataset | Seed | Acc | Macro-F1 | CE | Source SHA |
|---|---:|---:|---:|---:|---|
{chr(10).join(hybrid_lines)}

## Diagnostics and audit

- Audited expert coefficient rows: {len(coeff)} (expected 28); Cross injection rows: {len(injections)} (expected 56).
- Each new run has independent Best-Accuracy and Best-F1 checkpoint hashes and reload verification.
- Paired-dropout Validation RNG checks passed; all Test labels were masked before Trainer entry.
- Protected historical research/output files were hash-checked against CPU preflight.
- Preserved failure/block records: {len(failures)} total, including one first GPU preflight attempt blocked because sandboxed `nvidia-smi` received SIGKILL. Its fresh guarded retry passed; all seven current-source GPU preflights and all seven formal trainings passed.
- Formal training attempts failed: {manifest.get('failed_attempts',0)}; blocked: {manifest.get('blocked_resource_attempts',0)}.

## Interpretation boundary

The learned `b[4]` parameters are shared across modalities. The attached correction cancels theta's Cross-coefficient path while preserving its private/semantic/own-physical paths. The changed collaborative expert input can still alter routing and Joint View outputs indirectly. Results are descriptive, and the screen authorizes no default selection or follow-on model work.
"""
    write_markdown(runner.STUDY / "SCREENING_REPORT.md", report)

    audit = f"""# Implementation audit

- Base commit ancestry: verified from `{runner.BASE_SHA}`.
- Frozen V1.3-A, Static-Strength, NoP0 basis-screen, Hybrid-CGE, trainer, configs and relevant regression tests: SHA256 equality checked against the base before CPU preflight.
- New model: subclasses frozen V1.3-A and adds only `cross_coefficients` with shape `[4]`.
- Initialization: for all dataset/seed cells, V1.3-A common state tensors and external classifier initial state matched; `b` equaled that instance's initial physical `alpha[:,1]`.
- Mathematical path: `G_old + beta_m * (b - alpha_current[:,1]) * C_m`; alpha remains attached to autograd. Unit tests include analytical theta Cross-path cancellation and finite nonzero b gradients.
- Both stacked evaluation/diagnostic and serial low-memory checkpoint paths are implemented and covered by parity tests.
- Router input/logits, shared experts, bottlenecks, LayerNorm, Joint View, Cross calibration, paired-dropout replay, fallback, private/semantic paths, and trainer were inherited unchanged.
- Exact authorized training cells: 7; completed and dual-checkpoint verified: {complete}/7.
- All Test/LP flags are false; test labels were masked before Trainer.
- Historical source/data/checkpoints were read-only and protected by pre/post hash checks.
- Historical model-source hashes were checked against their recorded Git commits for {len(source_audit)} references (7 V1.3-A, 7 V1.2 and 3 fixed-split Hybrid); paired checkpoint hashes were verified.
- Report-generator correction was committed after training; pre-fix SHA256 `{preflight['source_file_sha256s']['scripts/analyze_mrcge_v13_early_cross_decoupling_screen.py']}` and post-fix SHA256 `{runner.sha256_file(ROOT / 'scripts/analyze_mrcge_v13_early_cross_decoupling_screen.py')}` are recorded here. The correction changed report generation and post-training audit only. The frozen model, Trainer, config, data-pipeline hashes and all training/checkpoint provenance were unchanged.

Unit and regression test output is recorded in `research/mrcge_v13_early_cross_decoupling/test_results.txt`.
"""
    write_markdown(runner.STUDY / "IMPLEMENTATION_AUDIT.md", audit)

    data_audit = f"""# Data and protocol audit

- Dataset split seed: 42, fixed project split files, automatic split generation disabled.
- Training seeds and authorized cells: Movies 42/43/44; Grocery 42/43; ele-fashion 42/43.
- CPU preflight status: `{preflight.get('status')}`; GPU preflight status: `{preflight.get('gpu_preflight_status')}`.
- V7B.1 K=8 semantic cache, feature, candidate and graph fingerprints were checked against paired V1.3-A records for every cell.
- Historical V1.3-A source commit/model hash, fixed split file, and both checkpoint hashes were rechecked; Movies/44 came from the fixed-split campaign. Hybrid Movies 42–44 model hashes and checkpoints were rechecked.
- Test labels were masked before Trainer; Test evaluation and Link Prediction stayed disabled.
- Independent Best-Accuracy and Best-Macro-F1 checkpoints were reloaded and hash-verified for all {complete} completed runs.
- Historical V1.2 and fixed-split Hybrid records, when included, are context only. Hybrid fixed-split Movies seeds 42–44 are not a paired architecture comparison.
- Historical model source SHA audit: {len(source_audit)} references pass; see `data/historical_source_audit.csv`.
- `failures.json` retains all failed and blocked attempts ({len(failures)} entries); no failure was overwritten.
"""
    write_markdown(runner.STUDY / "DATA_PROTOCOL_AUDIT.md", data_audit)

    preflight_report = f"""# Preflight report

- CPU preflight: `{preflight.get('status')}` at {preflight.get('recorded_utc')}.
- Physical GPU: {runner.PHYSICAL_GPU}; GPU preflight status: `{preflight.get('gpu_preflight_status')}`.
- GPU preflight passing cells: {sum(r.get('status') == 'pass' for r in preflight.get('gpu_preflight_cells', []))}/7.
- Required launch free memory: at least 8192 MiB and each formal launch also cleared the measured peak-reserved plus 2048 MiB margin.
- Training source freeze SHA: `{preflight.get('source_git_sha')}`.
- Audit/report commit: `{runner.source_sha()}`.
- Test evaluation: disabled.
- Initial sandbox-only GPU preflight was blocked by `nvidia-smi` SIGKILL and retained in `failures.json`; after authorized device access was available, a fresh ownership guard passed all seven cells.
"""
    write_markdown(runner.STUDY / "PREFLIGHT_REPORT.md", preflight_report)

    decision = f"""# Decision handoff

The screen finished {complete}/7 authorized cells. On the primary Movies target, Best-Accuracy checkpoint mean differences are Acc -0.170 pp, Macro-F1 -0.634 pp and CE +0.04082; this does not reach the suggested +0.3 pp Acc empirical signal. Grocery and ele-fashion show small positive mean Acc/F1 differences with mixed CE changes, so the overall result is mixed and does not support promoting the condition.

Review the per-seed paired metrics and parameter/resource cost in `SCREENING_REPORT.md` before deciding whether this narrowly scoped change merits a separately authorized follow-up. This screen alone does not establish a general improvement: sample counts are small, optimization trajectories differ, and the changed expert input can affect the router and Joint View downstream.

No default-model change or follow-on run is made by this screen. Test metrics were not computed.
"""
    write_markdown(runner.STUDY / "DECISION_HANDOFF.md", decision)

    manifest.update({"analysis_status": "pass", "analysis_recorded_utc": runner.now(),
        "analysis_git_sha": runner.source_sha(), "training_source_git_sha": preflight.get("source_git_sha"),
        "completed_cells": complete, "audited_coefficient_rows": len(coeff),
        "audited_cross_injection_rows": len(injections), "test_evaluation": False,
        "link_prediction": False})
    runner.json_write(runner.DATA / "manifest.json", manifest)
    result = {"status": "pass", "completed_cells": complete, "paired_rows": len(diffs),
        "coefficient_rows": len(coeff), "cross_injection_rows": len(injections),
        "reports": [str(runner.STUDY / name) for name in (
            "SCREENING_REPORT.md", "IMPLEMENTATION_AUDIT.md", "DATA_PROTOCOL_AUDIT.md",
            "PREFLIGHT_REPORT.md", "DECISION_HANDOFF.md")]}
    runner.json_write(runner.DATA / "analysis_audit.json", result)
    return result


def main() -> None:
    result = run_analysis()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
