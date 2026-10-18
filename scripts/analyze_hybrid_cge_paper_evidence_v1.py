#!/usr/bin/env python3
"""Generate validation-only tables for Hybrid-CGE paper evidence V1.

This analyzer reads recorded Train/Validation metrics and historical H2 rows.
It never reads Test labels or computes Test metrics.  Checkpoint-level class and
cross-modal diagnostics are collected by the separate explicit diagnostic mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STUDY = ROOT / "research/hybrid_cge_paper_evidence_v1"
DATA = STUDY / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion", "Toys", "Reddit-S")
SEEDS = (42, 43, 44)
ABLATIONS = ("A1_no_early_physical_cross", "A2_no_late_completion", "A3_no_moe", "A4_no_same_node")
HOP = "H2_physical_3hop"
FASHION = ("F1_fashion_dropout_aux", "F2_fashion_class_balanced_4hop", HOP,
           "F4_fashion_class_balanced_3hop")
NEW_BASELINES = ("gcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
REUSED_BASELINES = ("sage", "mmgcn", "rolemag")
BASELINES = (*NEW_BASELINES, *REUSED_BASELINES)
METRICS = ("val_acc", "val_macro_f1", "val_ce")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def avg(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    return mean_sd([float(row[key]) for row in rows if row.get(key) not in (None, "")])


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def load_all_complete_rows() -> list[dict[str, Any]]:
    rows = read_csv(DATA / "runs.csv")
    complete = [row for row in rows if row.get("status") == "complete"]
    if any(str(row.get("test_evaluation", "")).lower() != "false" for row in complete):
        raise RuntimeError("a complete current row does not explicitly confirm Test evaluation is disabled")
    for row in complete:
        for key in (*METRICS, "train_acc", "train_macro_f1", "train_ce_eval"):
            if row.get(key) not in (None, "") and not math.isfinite(float(row[key])):
                raise RuntimeError(f"non-finite {key} in {row.get('dataset')}/{row.get('condition')}/{row.get('seed')}")
    return complete


def _run_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (row["dataset"], row["model"], row["condition"], int(row["seed"]))


def load_current_rows() -> list[dict[str, Any]]:
    """Select the latest completed attempt per formal Dataset × Model × Seed.

    Earlier attempts stay in runs.csv and are listed in superseded_runs.csv;
    they are never silently overwritten or pooled with corrected runs.
    """
    by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in load_all_complete_rows():
        key = _run_key(row)
        try:
            attempt = int(row.get("attempt", 0))
        except (TypeError, ValueError):
            attempt = 0
        previous = by_key.get(key)
        try:
            previous_attempt = int(previous.get("attempt", 0)) if previous else -1
        except (TypeError, ValueError):
            previous_attempt = 0
        if previous is None or attempt >= previous_attempt:
            by_key[key] = row
    return list(by_key.values())


def superseded_attempt_rows() -> list[dict[str, Any]]:
    rows = load_all_complete_rows()
    current = {_run_key(row): row for row in load_current_rows()}
    affected = {
        "A1_no_early_physical_cross", HOP,
        FASHION[0], FASHION[1], FASHION[3],
    }
    output = []
    for row in rows:
        key = _run_key(row)
        selected = current.get(key)
        if selected is None or row is selected or row.get("attempt") == selected.get("attempt"):
            continue
        try:
            if int(row.get("attempt", 0)) >= int(selected.get("attempt", 0)):
                continue
        except (TypeError, ValueError):
            continue
        if row.get("condition") in affected and selected.get("source_git_sha") != row.get("source_git_sha"):
            output.append({
                **row,
                "superseded_by_attempt": selected.get("attempt", ""),
                "superseded_by_source_git_sha": selected.get("source_git_sha", ""),
                "superseded_reason": "SameNode path was inadvertently disabled by paper condition renaming; corrected wrapper rerun",
            })
    return output


def require_complete_matrix(rows: list[dict[str, Any]]) -> dict[str, int]:
    formal = [row for row in rows if not row.get("reused_from_branch")]
    counts = {
        "ablations": sum(row["condition"] in ABLATIONS for row in formal),
        "hop": sum(row["condition"] == HOP for row in formal),
        "new_baselines": sum(row["model"] in NEW_BASELINES for row in formal),
        "fashion": sum(row["condition"] in {FASHION[0], FASHION[1], FASHION[3]} for row in formal),
        "reused_baselines": sum(row["model"] in REUSED_BASELINES and bool(row.get("reused_from_branch")) for row in rows),
    }
    expected = {"ablations": 60, "hop": 15, "new_baselines": 90, "fashion": 9, "reused_baselines": 45}
    if counts != expected:
        raise RuntimeError(f"experiment matrix incomplete or misclassified: observed={counts}, expected={expected}")
    for stage in ("ablations", "hop", "baselines", "fashion"):
        gate = read_json(DATA / f"{stage}_stage_gate.json", {})
        if gate.get("status") != "pass" or gate.get("test_evaluation") is not False:
            raise RuntimeError(f"{stage} stage gate is missing or failed")
    return counts


def historical_h2_rows() -> dict[tuple[str, int], dict[str, Any]]:
    sources = (
        (ROOT / "research/hybrid_cge_v11/data/runs.csv", "H2_hybrid_plus", set(DATASETS[:3]), "research/hybrid_cge_v11"),
        (ROOT / "research/hybrid_cge_validation_expansion_v1/data/runs.csv", "H2-Frozen", set(DATASETS[3:]), "research/hybrid_cge_validation_expansion_v1"),
    )
    result = {}
    for path, condition, datasets, lineage in sources:
        for row in read_csv(path):
            if row.get("dataset") not in datasets or row.get("condition") != condition or row.get("status") != "complete":
                continue
            record: dict[str, Any] = dict(row)
            record.update({"model": "hybrid_cge_v11", "condition": "H2_historical_4hop",
                           "seed": int(row["seed"]), "lineage": lineage, "historical_reference": True})
            result[(row["dataset"], int(row["seed"]))] = record
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if set(result) != expected:
        raise RuntimeError(f"historical H2 reference is incomplete: {len(set(result) & expected)}/{len(expected)}")
    return result


def paired_differences(rows: list[dict[str, Any]], historical: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {(row["dataset"], row["condition"], int(row["seed"])): row for row in rows}
    ablation, hop, fashion = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            base = historical[(dataset, seed)]
            for condition in ABLATIONS:
                current = indexed[(dataset, condition, seed)]
                ablation.append({"dataset": dataset, "condition": condition, "seed": seed,
                                 "reference_condition": base["condition"], "reference_lineage": base["lineage"],
                                 **{f"delta_{metric}": float(current[metric]) - float(base[metric]) for metric in METRICS},
                                 "current_checkpoint_sha256": current["best_accuracy_checkpoint_sha256"],
                                 "reference_checkpoint_sha256": base.get("best_accuracy_checkpoint_sha256", "")})
            current = indexed[(dataset, HOP, seed)]
            hop.append({"dataset": dataset, "seed": seed, "comparison": "3hop_minus_historical_4hop",
                        "reference_lineage": base["lineage"],
                        **{f"delta_{metric}": float(current[metric]) - float(base[metric]) for metric in METRICS},
                        "current_checkpoint_sha256": current["best_accuracy_checkpoint_sha256"],
                        "reference_checkpoint_sha256": base.get("best_accuracy_checkpoint_sha256", "")})
    fashion_base = {(dataset, seed): historical[(dataset, seed)] for dataset in ("ele-fashion",) for seed in SEEDS}
    fashion_conditions = {
        "F0_historical_4hop": None,
        "F1_fashion_dropout_aux": FASHION[0],
        "F2_fashion_class_balanced_4hop": FASHION[1],
        "F3_fashion_original_3hop": HOP,
        "F4_fashion_class_balanced_3hop": FASHION[3],
    }
    for seed in SEEDS:
        ref = fashion_base[("ele-fashion", seed)]
        for label, condition in fashion_conditions.items():
            current = ref if condition is None else indexed[("ele-fashion", condition, seed)]
            fashion.append({"dataset": "ele-fashion", "candidate": label, "condition": current["condition"],
                            "seed": seed, "reference": "F0_historical_4hop", "reference_lineage": ref["lineage"],
                            **{f"delta_{metric}": float(current[metric]) - float(ref[metric]) for metric in METRICS},
                            "checkpoint_sha256": current.get("best_accuracy_checkpoint_sha256", ""),
                            "reference_checkpoint_sha256": ref.get("best_accuracy_checkpoint_sha256", "")})
    return ablation, hop, fashion


def summary_for(rows: list[dict[str, Any]], keys: tuple[str, ...], metrics: tuple[str, ...] = METRICS) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    output = []
    for key, selected in sorted(buckets.items()):
        out: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        out["n"] = len(selected)
        for metric in metrics:
            mu, sd = avg(selected, metric)
            out[f"{metric}_mean"] = mu
            out[f"{metric}_sample_sd"] = sd
        output.append(out)
    return output


def checkpoint_selection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("acc", "macro_f1", "ce")
    result = []
    for row in rows:
        out = {key: row.get(key, "") for key in ("dataset", "model", "condition", "seed", "best_accuracy_epoch", "best_f1_epoch",
                "best_accuracy_checkpoint_sha256", "best_f1_checkpoint_sha256")}
        for split in ("val", "train"):
            for metric in metrics:
                source = "train_ce_eval" if split == "train" and metric == "ce" else f"{split}_{metric}"
                out[f"best_accuracy_{split}_{metric}"] = row.get(source, "")
        for split in ("val", "train"):
            for metric in metrics:
                source = f"best_f1_{split}_{metric}"
                if split == "train" and metric == "ce":
                    source = "best_f1_train_ce"
                out[f"best_f1_{split}_{metric}"] = row.get(source, "")
        for target, a, b in (("delta_val_acc_best_f1_minus_acc", "best_f1_val_acc", "val_acc"),
                             ("delta_val_macro_f1_best_f1_minus_acc", "best_f1_val_macro_f1", "val_macro_f1"),
                             ("delta_val_ce_best_f1_minus_acc", "best_f1_val_ce", "val_ce")):
            if row.get(a) not in (None, "") and row.get(b) not in (None, ""):
                out[target] = float(row[a]) - float(row[b])
            else:
                out[target] = ""
        result.append(out)
    return result


def table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate_reports() -> None:
    rows = load_current_rows()
    superseded = superseded_attempt_rows()
    write_csv(DATA / "superseded_runs.csv", superseded)
    counts = require_complete_matrix(rows)
    historical = historical_h2_rows()
    ablation_pairs, hop_pairs, fashion_pairs = paired_differences(rows, historical)
    write_csv(DATA / "ablation_paired_differences.csv", ablation_pairs)
    write_csv(DATA / "hop_depth_paired_differences.csv", hop_pairs)
    write_csv(DATA / "fashion_paired_differences.csv", fashion_pairs)
    checkpoint_rows = checkpoint_selection_rows(rows)
    write_csv(DATA / "checkpoint_selection_summary.csv", checkpoint_rows)
    checkpoint_groups = summary_for(checkpoint_rows, ("model", "condition"),
                                    ("delta_val_acc_best_f1_minus_acc", "delta_val_macro_f1_best_f1_minus_acc",
                                     "delta_val_ce_best_f1_minus_acc"))
    # The aggregate makes the checkpoint tradeoff readable by model family;
    # exact dataset/seed values remain available in the row-level file.
    grouped_original: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_original.setdefault((row["model"], row["condition"]), []).append(row)
    for out in checkpoint_groups:
        original = grouped_original[(out["model"], out["condition"])]
        for name, key in (("train_val_acc_gap", "train_val_acc_gap"),
                          ("train_val_macro_f1_gap", "train_val_macro_f1_gap"),
                          ("train_val_ce_gap", "train_val_ce_gap")):
            mu, sd = avg(original, key)
            out[f"{name}_mean"] = mu
            out[f"{name}_sample_sd"] = sd
    write_csv(DATA / "checkpoint_selection_aggregate.csv", checkpoint_groups)
    current = rows
    all_rows = [*rows, *historical.values()]
    write_csv(DATA / "condition_summary.csv", summary_for(all_rows, ("dataset", "condition"),
              (*METRICS, "best_f1_val_acc", "best_f1_val_macro_f1", "best_f1_val_ce", "train_acc", "train_macro_f1", "train_ce_eval")))
    baseline_rows = [r for r in all_rows if r.get("model") in BASELINES]
    write_csv(DATA / "baseline_summary.csv", summary_for(baseline_rows, ("dataset", "model", "condition"),
              (*METRICS, "best_f1_val_macro_f1", "train_val_acc_gap", "train_val_macro_f1_gap", "train_val_ce_gap")))

    historical_labels = {"H2_hybrid_plus": "earlier H2_hybrid_plus campaign", "H2-Frozen": "earlier H2-Frozen campaign"}
    ablation_table = []
    for d in DATASETS:
        for c in ABLATIONS:
            subset = [r for r in current if r["dataset"] == d and r["condition"] == c]
            pair = [r for r in ablation_pairs if r["dataset"] == d and r["condition"] == c]
            ac, asc = avg(subset, "val_acc"); f1, f1s = avg(subset, "val_macro_f1")
            da, das = avg(pair, "delta_val_acc"); df, dfs = avg(pair, "delta_val_macro_f1")
            ablation_table.append((d, c, f"{len(subset)}/3", f"{pct(ac)} ± {pct(asc)}", f"{pct(f1)} ± {pct(f1s)}",
                                   f"{pct(da)} ± {pct(das)}", f"{pct(df)} ± {pct(dfs)}"))
    freeze_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    original_freeze_sha = "0db724e5eec704a1f3db92329bfbbac056df7202"
    ablation = ["# Ablation report", "", "## Implementation checks", "",
                f"The historical `src/models/hybrid_cge_v11.py` and checkpoints remain unchanged. A post-freeze source audit found that the paper wrapper replaced the parent's `H2_hybrid_plus` condition with paper condition IDs, while the inherited late path used that exact string to enable SameNode. This silently disabled SameNode in A1, 3-hop, and Fashion F1/F2/F4; the original 39 attempts are retained in `data/runs.csv` and listed in `data/superseded_runs.csv`. The wrapper now explicitly enables the existing SameNode helper for every late-completion arm except A4, and the corrected 39 attempts were appended without replacing their predecessors. Corrected implementation/training source is `{freeze_sha}`; the first freeze `{original_freeze_sha}` is retained as the source for unaffected runs. Ten focused Hybrid-CGE paper tests and all 581 repository tests passed after the correction. All selected 60 ablation cells pass their current stage gate; initial classifier and shared-model hashes match within Dataset × Seed, and Train/Validation metrics, checkpoint hashes/reloads and Test-off flags pass. Raw attempts and selected rows are traceable in `data/runs.csv` and `data/superseded_runs.csv`.", "",
                "The comparison reference is the seed-aligned historical 4-hop H2 row. It is from a prior campaign and commit, so the paired deltas are descriptive, not same-batch or matched-initialization effects. The historical Pilot remains unchanged.", "",
                table("Validation metrics and paired delta versus historical 4-hop H2", ("Dataset", "Ablation", "n", "Val Acc", "Val Macro-F1", "Δ Acc", "Δ Macro-F1"), ablation_table), "",
                "The four mechanisms are true framework-level path removals (see `ABLATION_DESIGN.md`): corrected A1 bypasses early physical cross computation while retaining SameNode; A2 skips late completion; A3 removes expert/router parameter trees and their balance loss while retaining SameNode; A4 masks and skips the same-node path while retaining 0.15 late-cross dropout. Negative or near-zero changes are retained as observed, not interpreted causally from only three seeds.", "",
                "Training was Validation-only. Test labels were masked before the Trainer and Test metrics were not evaluated.", ""]
    (STUDY / "ABLATION_REPORT.md").write_text("\n".join(ablation), encoding="utf-8")

    base_table = []
    for d in DATASETS:
        for m in BASELINES:
            subset = [r for r in baseline_rows if r["dataset"] == d and r["model"] == m]
            ac, asc = avg(subset, "val_acc"); f1, f1s = avg(subset, "val_macro_f1"); ce, ces = avg(subset, "val_ce")
            bf, _ = avg(subset, "best_f1_val_macro_f1")
            base_table.append((d, m, f"{len(subset)}/3", f"{pct(ac)} ± {pct(asc)}", f"{pct(f1)} ± {pct(f1s)}",
                               f"{num(ce)} ± {num(ces)}", pct(bf)))
    baseline = ["# Same-protocol baseline comparison", "",
                "All nine requested families are represented: six new training adaptations and 45 audited reused SAGE/MMGCN/RoleMAG rows. Every value is from the common Validation-only full-graph node-classification protocol. The project adaptations and their departures from the original method objectives are documented in `BASELINE_AUDIT.md`; no unrelated published benchmark score is mixed into this table.", "",
                table("Validation results (mean ± sample SD over seeds)", ("Dataset", "Model", "n", "Val Accuracy", "Val Macro-F1", "Val CE", "Best-F1 Macro-F1"), base_table), "",
                "Best-Accuracy and Best-F1 checkpoints are separately recorded. See `data/checkpoint_selection_summary.csv` for per-run checkpoint differences and Train/Validation gaps. All new baseline rows have finite metrics and intact checkpoint hashes; Test evaluation is false.", ""]
    (STUDY / "BASELINE_REPORT.md").write_text("\n".join(baseline), encoding="utf-8")

    checkpoint_report = ["# Validation checkpoint selection differences", "",
                         "The primary checkpoint is selected by Validation Accuracy. The secondary checkpoint is selected by Validation Macro-F1 using the same frozen stopping trajectory. For each run, `Best-F1 minus Best-Accuracy` is shown for Validation Accuracy, Macro-F1 and CE; Train/Validation gaps at the Best-Accuracy checkpoint are included to expose overfit differences. Exact rows and checkpoint digests are in `data/checkpoint_selection_summary.csv`.", "",
                         "| Model | Condition | n | Δ Val Acc (F1−Acc) | Δ Val Macro-F1 (F1−Acc) | Δ Val CE (F1−Acc) | Train−Val Acc gap | Train−Val F1 gap | Train−Val CE gap |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in checkpoint_groups:
        checkpoint_report.append("| " + " | ".join((str(row["model"]), str(row["condition"]), str(row["n"]),
            f"{pct(row.get('delta_val_acc_best_f1_minus_acc_mean'))} ± {pct(row.get('delta_val_acc_best_f1_minus_acc_sample_sd'))}",
            f"{pct(row.get('delta_val_macro_f1_best_f1_minus_acc_mean'))} ± {pct(row.get('delta_val_macro_f1_best_f1_minus_acc_sample_sd'))}",
            f"{num(row.get('delta_val_ce_best_f1_minus_acc_mean'))} ± {num(row.get('delta_val_ce_best_f1_minus_acc_sample_sd'))}",
            f"{pct(row.get('train_val_acc_gap_mean'))} ± {pct(row.get('train_val_acc_gap_sample_sd'))}",
            f"{pct(row.get('train_val_macro_f1_gap_mean'))} ± {pct(row.get('train_val_macro_f1_gap_sample_sd'))}",
            f"{num(row.get('train_val_ce_gap_mean'))} ± {num(row.get('train_val_ce_gap_sample_sd'))}")) + " |")
    checkpoint_report += ["", "A positive Δ Macro-F1 means choosing the F1 checkpoint improves Macro-F1 but may trade away Accuracy; CE is recalculated unweighted on Train/Validation. Historical/reused rows are identified in the source CSV and should not be treated as fresh training cells. All values exclude Test.", ""]
    (STUDY / "CHECKPOINT_SELECTION_REPORT.md").write_text("\n".join(checkpoint_report), encoding="utf-8")

    hop_table = []
    for d in DATASETS:
        three = [r for r in current if r["dataset"] == d and r["condition"] == HOP]
        four = [historical[(d, s)] for s in SEEDS]
        delta = [r for r in hop_pairs if r["dataset"] == d]
        a3, sd3 = avg(three, "val_acc"); f3, fsd3 = avg(three, "val_macro_f1")
        a4, sd4 = avg(four, "val_acc"); f4, fsd4 = avg(four, "val_macro_f1")
        da, das = avg(delta, "delta_val_acc"); df, dfs = avg(delta, "delta_val_macro_f1")
        hop_table.append((d, f"{pct(a3)} ± {pct(sd3)}", f"{pct(f3)} ± {pct(fsd3)}", f"{pct(a4)} ± {pct(sd4)}",
                          f"{pct(f4)} ± {pct(fsd4)}", f"{pct(da)} ± {pct(das)}", f"{pct(df)} ± {pct(dfs)}"))
    hop_report = ["# Physical propagation depth: three-hop versus four-hop", "",
                  "All 15 corrected 3-hop runs completed. The 3-hop implementation retains four experts and Top-2, constructs only P·H, P²·H and P³·H, and preserves the SameNode path. First attempts from the initial freeze are preserved and superseded in `data/superseded_runs.csv`. Existing 4-hop checkpoint compatibility was tested; historical checkpoints and source records were not changed.", "",
                  "| Dataset | 3-hop Val Acc | 3-hop Val F1 | Historical 4-hop Val Acc | Historical 4-hop Val F1 | Paired Δ Acc | Paired Δ F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    hop_report.extend("| " + " | ".join(row) + " |" for row in hop_table)
    hop_report += ["", "Seed-aligned paired differences and both checkpoint hashes are in `data/hop_depth_paired_differences.csv`. The 4-hop rows are from earlier campaigns (Movies/Grocery/ele-fashion: `H2_hybrid_plus`; Toys/Reddit-S: `H2-Frozen`), so this is a cross-campaign reference comparison, not a fresh matched control. Three seeds provide descriptive evidence only. A global depth choice is made in `MODEL_SELECTION.md` using the declared Accuracy-primary, Macro-F1-secondary rule; Test was not evaluated.", ""]
    (STUDY / "HOP_DEPTH_REPORT.md").write_text("\n".join(hop_report), encoding="utf-8")

    fashion_names = ("F0_historical_4hop", "F1_fashion_dropout_aux", "F2_fashion_class_balanced_4hop", "F3_fashion_original_3hop", "F4_fashion_class_balanced_3hop")
    fashion_table = []
    for name in fashion_names:
        selected = [r for r in fashion_pairs if r["candidate"] == name]
        source_rows = [historical[("ele-fashion", int(r2["seed"]))] if name == "F0_historical_4hop" else
                       next(r for r in current if r["dataset"] == "ele-fashion" and r["condition"] == r2["condition"] and int(r["seed"]) == int(r2["seed"])) for r2 in selected]
        ac, asc = avg(source_rows, "val_acc"); f1, f1s = avg(source_rows, "val_macro_f1"); ce, ces = avg(source_rows, "val_ce")
        bf, _ = avg(source_rows, "best_f1_val_macro_f1")
        da, _ = avg(selected, "delta_val_acc"); df, _ = avg(selected, "delta_val_macro_f1")
        fashion_table.append((name, f"{len(source_rows)}/3", f"{pct(ac)} ± {pct(asc)}", f"{pct(f1)} ± {pct(f1s)}",
                              f"{num(ce)} ± {num(ces)}", pct(bf), f"{pct(da)}", f"{pct(df)}"))
    fashion_report = ["# ele-fashion finite optimization study", "",
                      "Fashion candidates were selected using Validation only. F0 is the historical 4-hop H2 reference; F1/F2/F4 have three new seeds; F3 aliases the 3-hop H2 cells and was not duplicated. Class-balanced weights are derived from Train counts only, normalized inverse-square-root, clipped to [0.5, 2.0], and applied only to main CE.", "",
                      "| Candidate | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 | Δ Acc vs F0 | Δ F1 vs F0 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fashion_report.extend("| " + " | ".join(row) + " |" for row in fashion_table)
    fashion_report += ["", "F1 jointly changes main dropout to 0.30 and private auxiliary CE to 0.1; their separate effects are not identifiable. Weight decay remains fixed at 1e-4 for every candidate, so this study estimates no weight-decay effect. F2 isolates class-balanced main CE at 4-hop, while F4 combines class balancing with 3-hop. F3 is the same corrected 3-hop model and data as the hop-depth matrix.", "",
                       "Per-seed checkpoint differences are in `data/fashion_paired_differences.csv`. Train/Validation class-wise metrics (Classes 1 and 11 plus low-frequency Train classes) are reported in this section and stored in `data/fashion_classwise_metrics.csv` and `data/fashion_classwise_summary.csv`. The earlier Fashion attempts with the disabled SameNode path remain preserved and superseded. All conclusions are dataset-specific and Validation-only; Test was not evaluated.", ""]
    (STUDY / "FASHION_TUNING_REPORT.md").write_text("\n".join(fashion_report), encoding="utf-8")

    formal_training_source_shas = sorted({str(row.get("source_git_sha", "")) for row in rows
                                          if not row.get("reused_from_branch")})
    manifest = {
        "study": "Hybrid-CGE paper evidence V1", "formal_new_runs_expected": 174,
        "formal_new_runs_complete": sum(1 for row in current if not row.get("reused_from_branch")),
        "raw_complete_attempt_rows": len(load_all_complete_rows()),
        "superseded_implementation_attempts": len(superseded),
        "matrix_counts": counts, "reused_baseline_rows": counts["reused_baselines"],
        "test_evaluation": False, "test_labels_used_for_selection": False,
        "historical_4hop_reference_is_cross_campaign": True,
        "original_implementation_freeze_sha": original_freeze_sha,
        "corrected_implementation_freeze_sha": freeze_sha,
        "formal_training_source_shas": formal_training_source_shas,
    }
    (DATA / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"REPORTS COMPLETE: {manifest['formal_new_runs_complete']}/174 formal runs; Test not evaluated")


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten scalar/list diagnostics while omitting booleans and tensors."""
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            flattened.update(_flatten_numeric(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            flattened[prefix] = number
    return flattened


def generate_checkpoint_diagnostics(physical_gpu: int) -> None:
    """Re-evaluate locked checkpoints on Train/Val only and collect details.

    The all-node forward is transductive and uses features/edges only.  Metric
    indexing is restricted to Train and Validation; Test labels are masked to
    -100 before model construction and are never indexed by this routine.
    """
    import torch
    from scripts import run_hybrid_cge_paper_evidence_v1 as runner
    from src.data import load_mag_data

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={physical_gpu} to isolate checkpoint diagnostics")
    runner.gpu_guard._assert_physical_gpu_idle(physical_gpu)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("checkpoint diagnostics require one visible CUDA device")
    device = torch.device("cuda:0")
    current = load_current_rows()
    require_complete_matrix(current)
    historical = historical_h2_rows()
    by_key = {(row["dataset"], row["condition"], int(row["seed"])): row for row in current}

    jobs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            ref = historical[(dataset, seed)]
            jobs.append({"dataset": dataset, "seed": seed, "condition": "H2_full", "candidate": "H2_4hop",
                         "checkpoint": ref["best_accuracy_checkpoint"],
                         "checkpoint_sha256": ref["best_accuracy_checkpoint_sha256"], "historical": True})
            hop = by_key[(dataset, HOP, seed)]
            jobs.append({"dataset": dataset, "seed": seed, "condition": HOP, "candidate": "H2_3hop",
                         "checkpoint": hop["best_accuracy_checkpoint"],
                         "checkpoint_sha256": hop["best_accuracy_checkpoint_sha256"], "historical": False})
    fashion_conditions = {
        "F1": "F1_fashion_dropout_aux",
        "F2": "F2_fashion_class_balanced_4hop",
        "F4": "F4_fashion_class_balanced_3hop",
    }
    for seed in SEEDS:
        for candidate, condition in fashion_conditions.items():
            row = by_key[("ele-fashion", condition, seed)]
            jobs.append({"dataset": "ele-fashion", "seed": seed, "condition": condition, "candidate": candidate,
                         "checkpoint": row["best_accuracy_checkpoint"],
                         "checkpoint_sha256": row["best_accuracy_checkpoint_sha256"], "historical": False})

    class_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    weight_audits: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    cross_detail_records: list[dict[str, Any]] = []
    for job in jobs:
        checkpoint = Path(job["checkpoint"])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != job["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch before diagnostic: {checkpoint}")
        candidate = str(job["candidate"])
        cache = runner.current_cache(job["dataset"])
        cfg = runner.build_cfg(job["dataset"], runner.PAPER_MODEL, int(job["seed"]), str(device),
                               str(job["condition"]), cache)
        raw = load_mag_data(cfg, "nc", int(job["seed"]))
        safe = runner.mask_test_labels(raw)
        if bool((safe.y[safe.test_idx] != -100).any()):
            raise RuntimeError("Test-label mask audit failed before diagnostic inference")
        runner.set_seed(int(job["seed"]))
        model = runner.get_model(cfg, safe, cache, device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        head = torch.nn.Linear(model.out_dim, int(safe.num_classes)).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        head.load_state_dict(payload["head_state"], strict=True)
        model.eval(); head.eval()
        cpu_state = torch.random.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state(device).clone()
        with torch.no_grad():
            embedding, _, _, _, info = model(safe.x.to(device), safe.edge_index.to(device), return_details=True)
            logits = head(embedding).detach().cpu().float()
        rng_unchanged = torch.equal(cpu_state, torch.random.get_rng_state()) and torch.equal(cuda_state, torch.cuda.get_rng_state(device))
        if not rng_unchanged:
            raise RuntimeError(f"Train/Validation diagnostic changed RNG state: {job['dataset']}/{candidate}/seed{job['seed']}")

        # Index only the two permitted supervision/evaluation splits below.
        train_idx = safe.train_idx.detach().cpu().long()
        val_idx = safe.val_idx.detach().cpu().long()
        y = safe.y.detach().cpu().long()
        train_y = y[train_idx]
        train_counts = torch.bincount(train_y, minlength=int(safe.num_classes)).tolist()
        if job["dataset"] == "ele-fashion" and candidate in {"H2_4hop", "H2_3hop", "F1", "F2", "F4"}:
            source_row = None
            if candidate in {"F1", "F2", "F4", "H2_3hop"}:
                source_row = by_key[("ele-fashion", str(job["condition"]), int(job["seed"]))]
            weights = None
            if source_row is not None and source_row.get("main_ce_class_weights"):
                parsed_weights = json.loads(source_row["main_ce_class_weights"])
                if parsed_weights is not None:
                    weights = parsed_weights
                if weights is not None and len(weights) != int(safe.num_classes):
                    raise RuntimeError("recorded class-weight vector does not match Fashion class count")
            for class_id, count in enumerate(train_counts):
                weight = float(weights[class_id]) if weights is not None else None
                support_rows.append({"candidate": candidate, "condition": job["condition"], "seed": job["seed"],
                                     "class_id": class_id, "train_support": int(count),
                                     "main_ce_class_weight": weight if weight is not None else "",
                                     "class_balanced_main_ce": weights is not None,
                                     "test_evaluation": False})
            if weights is not None:
                unsupported = [class_id for class_id, count in enumerate(train_counts) if count == 0]
                zero_weight = [class_id for class_id, weight in enumerate(weights) if float(weight) == 0.0]
                observed_weights = [float(weights[class_id]) for class_id, count in enumerate(train_counts) if count > 0]
                if set(unsupported) != set(zero_weight):
                    raise RuntimeError(f"zero class weights do not exactly match classes absent from Train: {zero_weight}/{unsupported}")
                if any(not 0.5 <= weight <= 2.0 for weight in observed_weights):
                    raise RuntimeError("observed Train class weights fall outside [0.5, 2.0]")
                weight_audits.append({"candidate": candidate, "condition": job["condition"], "seed": job["seed"],
                                      "train_support_by_class": [int(value) for value in train_counts],
                                      "class_weights": [float(value) for value in weights],
                                      "observed_class_weight_bounds": [min(observed_weights), max(observed_weights)],
                                      "zero_weight_class_ids": zero_weight,
                                      "zero_support_class_ids": unsupported,
                                      "zero_support_weight_has_no_train_target": True,
                                      "weights_use_validation_or_test_counts": False,
                                      "test_evaluation": False})
        requested = [class_id for class_id in (1, 11) if 0 <= class_id < int(safe.num_classes)]
        low_frequency = sorted((count, class_id) for class_id, count in enumerate(train_counts)
                               if count > 0 and class_id not in requested)[:3]
        class_ids = [(f"class_{class_id}", class_id) for class_id in requested]
        class_ids += [(f"low_train_support_{rank}", class_id) for rank, (_, class_id) in enumerate(low_frequency, start=1)]
        prediction = logits.argmax(-1)
        for split_name, indices in (("train", train_idx), ("validation", val_idx)):
            labels = y[indices].numpy()
            preds = prediction[indices].numpy()
            if (labels < 0).any():
                raise RuntimeError(f"masked labels reached {split_name} diagnostic")
            for class_name, class_id in class_ids:
                positives = labels == class_id
                predicted = preds == class_id
                tp = int((positives & predicted).sum())
                support = int(positives.sum())
                predicted_count = int(predicted.sum())
                precision = tp / predicted_count if predicted_count else 0.0
                recall = tp / support if support else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
                class_rows.append({"dataset": job["dataset"], "candidate": candidate, "condition": job["condition"],
                                   "seed": job["seed"], "split": split_name, "class_name": class_name,
                                   "class_id": class_id, "train_support": int(train_counts[class_id]),
                                   "evaluation_support": support, "predicted_count": predicted_count,
                                   "precision": precision, "recall": recall, "f1": f1,
                                   "checkpoint_sha256": digest, "test_evaluation": False})

        details = info.get("details", {})
        flattened = _flatten_numeric(details)
        if job["condition"] in ("H2_full", HOP):
            for metric, value in flattened.items():
                cross_rows.append({"dataset": job["dataset"], "condition": job["condition"],
                                   "candidate": candidate, "seed": job["seed"], "metric": metric,
                                   "value": value, "checkpoint_sha256": digest, "test_evaluation": False})
            cross_detail_records.append({"dataset": job["dataset"], "condition": job["condition"],
                                         "candidate": candidate, "seed": job["seed"],
                                         "checkpoint_sha256": digest, "details": details,
                                         "test_evaluation": False, "test_labels_indexed": False})
        del model, head, raw, safe, logits, embedding
        torch.cuda.empty_cache()

    # This file is specifically the Fashion optimization audit. Other
    # datasets' forward-path diagnostics are recorded separately above.
    class_rows = [row for row in class_rows if row["dataset"] == "ele-fashion"]
    write_csv(DATA / "fashion_classwise_metrics.csv", class_rows)
    write_csv(DATA / "fashion_train_class_support.csv", support_rows)
    with (DATA / "fashion_class_weight_audit.json").open("w", encoding="utf-8") as stream:
        json.dump({"test_evaluation": False, "class_counts_use_train_idx_only": True,
                   "audits": weight_audits}, stream, indent=2, sort_keys=True, allow_nan=False)
    write_csv(DATA / "cross_modal_diagnostics.csv", cross_rows)
    with (DATA / "cross_modal_diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump({"test_evaluation": False, "test_labels_indexed": False,
                   "evaluated_splits": ["train", "validation"], "runs": cross_detail_records},
                  stream, indent=2, sort_keys=True, allow_nan=False)
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in cross_rows:
        grouped.setdefault((row["dataset"], row["condition"], row["metric"]), []).append(float(row["value"]))
    summary = []
    for (dataset, condition, metric), values in sorted(grouped.items()):
        mu, sd = mean_sd(values)
        summary.append({"dataset": dataset, "condition": condition, "metric": metric, "n": len(values),
                        "mean": mu, "sample_sd": sd, "test_evaluation": False})
    write_csv(DATA / "cross_modal_diagnostics_summary.csv", summary)
    class_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in class_rows:
        class_groups.setdefault((row["candidate"], row["split"], row["class_name"]), []).append(row)
    class_summary = []
    for (candidate, split, class_name), records in sorted(class_groups.items()):
        class_ids = sorted({int(record["class_id"]) for record in records})
        values: dict[str, Any] = {"candidate": candidate, "split": split, "class_name": class_name,
                                  "class_ids": ",".join(map(str, class_ids)), "n_seeds": len(records)}
        for metric in ("train_support", "evaluation_support", "precision", "recall", "f1"):
            mu, sd = mean_sd([float(record[metric]) for record in records])
            values[f"{metric}_mean"] = mu
            values[f"{metric}_sample_sd"] = sd
        class_summary.append(values)
    write_csv(DATA / "fashion_classwise_summary.csv", class_summary)

    # Concise, directly readable diagnostics alongside the machine-readable
    # scalar and full per-run details.
    wanted_cross = (
        "early_physical_cross.text.calibrated_rms",
        "early_physical_cross.visual.calibrated_rms",
        "modalities.text.same_semantic_weighted_rms",
        "modalities.visual.same_semantic_weighted_rms",
        "modalities.text.late_completion.semantic_source_weight",
        "modalities.visual.late_completion.semantic_source_weight",
        "modalities.text.late_completion.same_node_source_weight",
        "modalities.visual.late_completion.same_node_source_weight",
        "modalities.text.late_completion.residual_to_collaborative_rms",
        "modalities.visual.late_completion.residual_to_collaborative_rms",
    )
    cross_index = {(row["dataset"], row["condition"], row["metric"]): row for row in summary}
    cross_table = []
    for dataset in DATASETS:
        for condition in ("H2_full", HOP):
            cells = []
            for metric in wanted_cross:
                result = cross_index.get((dataset, condition, metric))
                cells.append("—" if result is None else f"{float(result['mean']):.4f} ± {float(result['sample_sd']):.4f}")
            cross_table.append((dataset, condition, *cells))
    diagnostic_report = [
        "# Physical and semantic cross-modal diagnostics", "",
        "Best-Validation-Accuracy checkpoints from historical H2 4-hop and current 3-hop runs were reloaded with the frozen compatible wrapper. The model was set to evaluation mode and returned its forward diagnostics. Train and Validation metrics index only their own rows; Test labels were masked before model construction and never indexed. All reported numbers are descriptive forward activations/routing summaries, not causal effects.", "",
        "For `late_completion`, source weights average the node-wise selector probabilities. Semantic and same-node candidates can both be available, while `none` remains a third option. `same_semantic_weighted_rms` is the within-modality semantic graph contribution; early calibrated cross RMS is the other modality's physical-neighbor message after scale calibration. Residual ratios summarize the late contribution relative to its collaborative state.", "",
        "| Dataset | Depth | Early cross RMS (target text) | Early cross RMS (target visual) | Text same-semantic RMS | Visual same-semantic RMS | Text semantic source weight | Visual semantic source weight | Text same-node source weight | Visual same-node source weight | Text late residual/state RMS | Visual late residual/state RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    diagnostic_report.extend("| " + " | ".join(row) + " |" for row in cross_table)
    diagnostic_report += ["", "Full scalar paths and per-seed checkpoint details are in `data/cross_modal_diagnostics.csv`, `data/cross_modal_diagnostics_summary.csv` and `data/cross_modal_diagnostics.json`. Seed-to-seed variation and dataset shifts should be read directly; activation scale alone does not establish utility or mechanism importance.", ""]
    (STUDY / "PHYSICAL_SEMANTIC_DIAGNOSTICS.md").write_text("\n".join(diagnostic_report), encoding="utf-8")

    fashion_report_path = STUDY / "FASHION_TUNING_REPORT.md"
    content = fashion_report_path.read_text(encoding="utf-8") if fashion_report_path.is_file() else "# ele-fashion finite optimization study\n"
    if "## Train/Validation class-wise diagnostics" not in content:
        class_table = []
        candidate_order = ("H2_4hop", "F1", "F2", "H2_3hop", "F4")
        for candidate in candidate_order:
            for class_name in ("class_1", "class_11", "low_train_support_1", "low_train_support_2", "low_train_support_3"):
                val = next((row for row in class_summary if row["candidate"] == candidate and row["split"] == "validation" and row["class_name"] == class_name), None)
                train = next((row for row in class_summary if row["candidate"] == candidate and row["split"] == "train" and row["class_name"] == class_name), None)
                if val:
                    class_table.append((candidate, class_name, str(val["class_ids"]), pct(val["recall_mean"]), pct(val["f1_mean"]),
                                        pct(train["f1_mean"] if train else None), str(val["n_seeds"]) + "/3"))
        section = ["", "## Train/Validation class-wise diagnostics", "",
                   "Class-wise precision/recall/F1 is from Best-Accuracy checkpoints. Classes 1 and 11 are explicit; the three low-frequency classes are selected separately per seed from Train support counts, excluding Classes 1 and 11. Test labels are never indexed. Per-seed class IDs, counts and metrics are available in `data/fashion_classwise_metrics.csv`; aggregate mean and sample SD are in `data/fashion_classwise_summary.csv`.", "",
                   "| Candidate | Class | Class ID* | Val Recall | Val F1 | Train F1 | n |", "|---|---|---:|---:|---:|---:|---:|"]
        section.extend("| " + " | ".join(row) + " |" for row in class_table)
        section += ["", "*For low-frequency ranks, the selected class ID can differ by seed; inspect the per-seed CSV for the exact Train support rule and class identity.", ""]
        fashion_report_path.write_text(content + "\n".join(section), encoding="utf-8")
    if "## Class-weight support audit" not in fashion_report_path.read_text(encoding="utf-8"):
        weight_lines = ["", "## Class-weight support audit", "",
                        "Class weights are computed from Train labels only. This Fashion split has a class with zero Train support; the current opt-in implementation assigns zero to such an unobserved output class, while every observed class weight is normalized inverse-square-root frequency clipped to [0.5, 2.0]. Because the absent class never appears as a Train target, its zero entry is not selected by the Train CE loss. This is explicitly reported as the zero-support convention rather than claiming every vector entry is in [0.5, 2.0]. Exact seed-wise class supports and weights are in `data/fashion_train_class_support.csv` and `data/fashion_class_weight_audit.json`.", ""]
        with fashion_report_path.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(weight_lines))

    print(f"CHECKPOINT DIAGNOSTICS COMPLETE: {len(jobs)} checkpoints; {len(class_rows)} Train/Validation class rows; Test not evaluated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", action="store_true", help="generate result tables (requires all stage gates)")
    parser.add_argument("--diagnose-checkpoints", action="store_true", help="Train/Validation-only checkpoint and cross-modal diagnostics")
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    if args.reports == args.diagnose_checkpoints:
        parser.error("select exactly one of --reports or --diagnose-checkpoints")
    if args.reports:
        generate_reports()
    else:
        generate_checkpoint_diagnostics(args.physical_gpu)


if __name__ == "__main__":
    main()
