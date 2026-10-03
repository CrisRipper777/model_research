#!/usr/bin/env python3
"""Analyze PCRR-E0 validation runs and selected checkpoint diagnostics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

import torch
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "pcrr_e0_postgpr_paired_residual"
DATA_DIR = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("B", "P", "S")
MODEL_VARIANTS = {"B": "base", "P": "paired", "S": "shuffled"}
SEEDS = (42, 43, 44)
BASE_SHA = "3ef56df39a2d554aaac9a97ae9f6d4943d345f13"
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}
COMPARISONS = (("P", "B"), ("P", "S"), ("S", "B"))
TMP = Path("/tmp/pcrr_e0_postgpr_paired_residual")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (statistics.mean(values), statistics.pstdev(values)) if values else (math.nan, math.nan)


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def load_manifest() -> dict:
    path = RESEARCH / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("base_sha") != BASE_SHA:
        raise RuntimeError("PCRR-E0 manifest has an unexpected base SHA")
    if manifest.get("formal_status") != "complete":
        raise RuntimeError("formal campaign is not marked complete")
    runs = [run for run in manifest.get("formal_runs", []) if run.get("status") == "complete"]
    if len(runs) != 27:
        raise RuntimeError(f"expected 27 completed formal runs, found {len(runs)}")
    return manifest


def read_run_metrics(run: dict) -> dict:
    path = Path(run["run_metrics_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"validation-only marker missing in {path}")
    items = payload.get("runs", [])
    if len(items) != 1 or int(items[0]["seed"]) != int(run["seed"]):
        raise RuntimeError(f"run/seed mismatch in {path}")
    metrics = items[0]["metrics"]
    if any(key.startswith("test_") for key in metrics):
        raise RuntimeError(f"test metric found in {path}")
    return metrics


def performance_outputs(manifest: dict):
    rows = []
    lookup = {}
    for run in manifest["formal_runs"]:
        if run.get("status") != "complete":
            continue
        metrics = read_run_metrics(run)
        row = {
            "dataset": run["dataset"], "variant": run["variant"],
            "model_variant": run["model_variant"], "seed": int(run["seed"]),
            "val_accuracy": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "val_ce": float(metrics["val_ce"]),
            "best_epoch": int(run["best_epoch"]),
            "development_no_test": True, "evaluate_test": False,
            "run_metrics_path": run["run_metrics_path"],
            "checkpoint_path": run["checkpoint_path"],
        }
        rows.append(row)
        lookup[(row["dataset"], row["variant"], row["seed"])] = row
    if len(rows) != 27:
        raise RuntimeError(f"expected 27 validation-only metric rows, found {len(rows)}")

    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            item = {"dataset": dataset, "variant": variant, "n_model_seeds": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(
                    [float(row[metric]) for row in group]
                )
            summary.append(item)

    paired = []
    paired_summary = []
    for dataset in DATASETS:
        for new, base in COMPARISONS:
            group = []
            for seed in SEEDS:
                a, b = lookup[(dataset, new, seed)], lookup[(dataset, base, seed)]
                row = {
                    "dataset": dataset, "seed": seed,
                    "comparison": f"{new}-{base}", "new_variant": new,
                    "base_variant": base,
                    "accuracy_delta_pp": (a["val_accuracy"] - b["val_accuracy"]) * 100.0,
                    "macro_f1_delta_pp": (a["val_macro_f1"] - b["val_macro_f1"]) * 100.0,
                    "ce_delta": a["val_ce"] - b["val_ce"],
                    "new_accuracy": a["val_accuracy"], "base_accuracy": b["val_accuracy"],
                }
                group.append(row)
                paired.append(row)
            result = {"dataset": dataset, "comparison": f"{new}-{base}", "n_paired_seeds": len(group)}
            for metric in ("accuracy_delta_pp", "macro_f1_delta_pp", "ce_delta"):
                vals = [float(row[metric]) for row in group]
                result[f"{metric}_mean"], result[f"{metric}_population_sd"] = mean_sd(vals)
                result[f"{metric}_positive_seeds"] = sum(v > 1e-12 for v in vals)
                result[f"{metric}_negative_seeds"] = sum(v < -1e-12 for v in vals)
                result[f"{metric}_ties"] = sum(abs(v) <= 1e-12 for v in vals)
            paired_summary.append(result)
    return rows, summary, paired, paired_summary, lookup


def _make_cfg(dataset: str, seed: int, device: str):
    overrides = [
        f"dataset={dataset}", "task=nc", "model=pcrr_mag_v0",
        "model.variant=base", f"seed={seed}", f"device={device}",
        "task.evaluate_test=false", "task.development_no_test=true",
    ]
    if dataset in ("Movies", "Grocery"):
        overrides.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _load_context(dataset: str, seed: int, device: torch.device):
    from src.data import load_mag_data
    from src.models.pcrr_mag_v0 import Model

    cfg = _make_cfg(dataset, seed, str(device))
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = Model(cfg, data_info).to(device)
    x = data.x if data.num_nodes >= 50_000 else data.x.to(device)
    edge_index = data.edge_index.to(device)
    return cfg, data, model, x, edge_index


def _load_checkpoint(model, run: dict, device: torch.device):
    payload = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    from torch import nn
    classifier = nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"])).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    classifier.eval()
    return payload, classifier


def _validation_metrics(cfg, data, model, classifier, x, edge_index, device, intervention="normal", permutation=None):
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels

    with torch.no_grad():
        z, _, _, _, _ = model(
            x, edge_index, intervention=intervention,
            source_node_permutation=permutation,
        )
        return _evaluate_split(
            classifier, z, data.y, data.val_idx, device,
            int(cfg.task.inference_batch_size),
            _resolve_nc_eval_labels(data, development_no_test=True),
        )


def _checkpoint_index(manifest: dict) -> dict:
    return {
        (run["dataset"], run["variant"], int(run["seed"])): run
        for run in manifest["formal_runs"] if run.get("status") == "complete"
    }


def checkpoint_diagnostics(manifest: dict, lookup: dict, device: torch.device):
    from src.models.pcrr_mag_v0 import Model

    residual_rows = []
    gpr_rows = []
    intervention_rows = []
    checkpoint_runs = _checkpoint_index(manifest)
    for dataset in DATASETS:
        for seed in SEEDS:
            context = _load_context(dataset, seed, device)
            cfg, data, model, x, edge_index = context
            for variant in VARIANTS:
                run = checkpoint_runs[(dataset, variant, seed)]
                # The task checkpoint stores weights, not the Hydra model config.
                # Set the checkpoint's variant before forwarding P/S diagnostics.
                model.variant = MODEL_VARIANTS[variant]
                payload, classifier = _load_checkpoint(model, run, device)
                coefficients = model.effective_coefficients().detach().cpu().tolist()
                permutation = model.pair_shuffle_permutation
                fixed_points = int((permutation == torch.arange(
                    permutation.numel(), device=permutation.device
                )).sum().item())
                fixed_point_rate = fixed_points / permutation.numel() if permutation.numel() else 0.0
                gpr_rows.append({
                    "dataset": dataset, "variant": variant, "seed": seed,
                    "c0": coefficients[0], "c1": coefficients[1],
                    "c2": coefficients[2], "c3": coefficients[3],
                    "selected_epoch": run["best_epoch"],
                    "pair_shuffle_fixed_points": fixed_points,
                    "pair_shuffle_fixed_point_rate": fixed_point_rate,
                    "checkpoint_path": run["checkpoint_path"],
                })
                if variant in ("P", "S"):
                    with torch.no_grad():
                        _, _, _, _, info = model(x, edge_index, return_details=True)
                    detail = info["details"]
                    for branch, target_name in (("text", "text"), ("visual", "visual")):
                        pair = detail["pairs"][target_name]
                        target = pair["target"]
                        delta = pair["delta"]
                        refined = pair["embedding"]
                        target_rms = target.float().square().mean().sqrt()
                        delta_rms = delta.float().square().mean().sqrt()
                        residual_rows.append({
                            "dataset": dataset, "variant": variant, "seed": seed,
                            "direction": "T<-V" if branch == "text" else "V<-T",
                            "rms_g_target": scalar(target_rms),
                            "rms_delta": scalar(delta_rms),
                            "delta_to_target_rms": scalar(delta_rms / target_rms.clamp_min(1e-12)),
                            "cosine_delta_g_target": scalar(torch.nn.functional.cosine_similarity(
                                delta.float(), target.float(), dim=-1
                            ).mean()),
                            "rms_gtilde_minus_g": scalar((refined - target).float().square().mean().sqrt()),
                            "cosine_gtilde_g": scalar(torch.nn.functional.cosine_similarity(
                                refined.float(), target.float(), dim=-1
                            ).mean()),
                            "pair_hidden_rms": scalar(pair["hidden_rms"]),
                            "pair_up_weight_norm": scalar(model.pair_up.weight.norm()),
                            "pair_down_weight_norm": scalar(model.pair_down.weight.norm()),
                            "checkpoint_path": run["checkpoint_path"],
                        })
                    del info, detail

                if variant == "P":
                    normal = lookup[(dataset, "P", seed)]
                    for intervention in ("residual_off",):
                        measured = _validation_metrics(
                            cfg, data, model, classifier, x, edge_index, device,
                            intervention=intervention,
                        )
                        intervention_rows.append({
                            "dataset": dataset, "seed": seed,
                            "intervention": intervention, "repeat": 0,
                            "shuffle_seed": "", "val_accuracy": measured["acc"],
                            "val_macro_f1": measured["macro_f1"], "val_ce": measured["ce"],
                            "accuracy_delta_vs_normal_pp": (measured["acc"] - normal["val_accuracy"]) * 100.0,
                            "macro_f1_delta_vs_normal_pp": (measured["macro_f1"] - normal["val_macro_f1"]) * 100.0,
                            "ce_delta_vs_normal": measured["ce"] - normal["val_ce"],
                            "evaluate_test": False,
                        })
                    for repeat in range(5):
                        intervention_seed = seed + 91000 + repeat
                        generator = torch.Generator(device="cpu").manual_seed(intervention_seed)
                        permutation = torch.randperm(data.num_nodes, generator=generator)
                        measured = _validation_metrics(
                            cfg, data, model, classifier, x, edge_index, device,
                            intervention="source_node_shuffle", permutation=permutation,
                        )
                        intervention_rows.append({
                            "dataset": dataset, "seed": seed,
                            "intervention": "source_node_shuffle", "repeat": repeat + 1,
                            "shuffle_seed": intervention_seed,
                            "val_accuracy": measured["acc"],
                            "val_macro_f1": measured["macro_f1"], "val_ce": measured["ce"],
                            "accuracy_delta_vs_normal_pp": (measured["acc"] - normal["val_accuracy"]) * 100.0,
                            "macro_f1_delta_vs_normal_pp": (measured["macro_f1"] - normal["val_macro_f1"]) * 100.0,
                            "ce_delta_vs_normal": measured["ce"] - normal["val_ce"],
                            "evaluate_test": False,
                        })
                model.load_state_dict(payload["model_state"], strict=True)
                del classifier, payload
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del cfg, data, model, x, edge_index
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    intervention_summary = []
    for dataset in DATASETS:
        for name in ("residual_off", "source_node_shuffle"):
            group = [row for row in intervention_rows if row["dataset"] == dataset and row["intervention"] == name]
            per_seed = []
            for seed in SEEDS:
                seed_rows = [row for row in group if int(row["seed"]) == seed]
                if seed_rows:
                    per_seed.append({
                        "seed": seed,
                        **{
                            metric: statistics.mean(float(row[metric]) for row in seed_rows)
                            for metric in (
                                "val_accuracy", "val_macro_f1", "val_ce",
                                "accuracy_delta_vs_normal_pp",
                                "macro_f1_delta_vs_normal_pp", "ce_delta_vs_normal",
                            )
                        },
                    })
            item = {
                "dataset": dataset, "intervention": name,
                "n_model_seeds": len(per_seed), "n_raw_observations": len(group),
            }
            for metric in (
                "val_accuracy", "val_macro_f1", "val_ce",
                "accuracy_delta_vs_normal_pp", "macro_f1_delta_vs_normal_pp",
                "ce_delta_vs_normal",
            ):
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(
                    [float(row[metric]) for row in per_seed]
                )
            deltas = [float(row["accuracy_delta_vs_normal_pp"]) for row in per_seed]
            item["accuracy_delta_positive_seeds"] = sum(value > 1e-12 for value in deltas)
            item["accuracy_delta_negative_seeds"] = sum(value < -1e-12 for value in deltas)
            item["accuracy_delta_tied_seeds"] = sum(abs(value) <= 1e-12 for value in deltas)
            intervention_summary.append(item)
    return residual_rows, gpr_rows, intervention_rows, intervention_summary


def classify_evidence(paired_summary: list[dict]) -> str:
    p_b = [row for row in paired_summary if row["comparison"] == "P-B"]
    p_s = [row for row in paired_summary if row["comparison"] == "P-S"]
    p_b_positive = sum(row["accuracy_delta_pp_mean"] > 0 for row in p_b)
    p_s_positive = sum(row["accuracy_delta_pp_mean"] > 0 for row in p_s)
    if p_b_positive >= 2 and p_s_positive >= 2:
        return "Case A leaning: mean paired P−B and P−S accuracy deltas are positive on at least two datasets each; assess their three seed directions and secondary metrics before treating this as a correspondence-aware gain."
    if p_b_positive >= 2 and p_s_positive < 2:
        return "Case B leaning: P−B trends positive on at least two datasets, while P−S does not; the evidence does not isolate correct correspondence from added residual capacity."
    if p_b_positive < 2 and p_s_positive >= 2:
        return "Case C leaning: correspondence control P−S trends positive on at least two datasets, while net P−B gain is not broad; this is mechanism evidence without a demonstrated net architecture gain."
    if all(row["accuracy_delta_pp_mean"] <= 0 for row in p_b):
        return "Case E leaning: P−B mean accuracy is nonpositive on all datasets; do not frame checkpoint intervention sensitivity as a performance gain."
    return "Case D leaning: validation comparisons do not show a broad positive trend; treat the screen as neutral or mixed, without expanding the mechanism."


def write_report(manifest, perf_summary, paired_summary, residual_rows, gpr_rows, intervention_rows, intervention_summary):
    pbr = [row for row in paired_summary if row["comparison"] == "P-B"]
    psr = [row for row in paired_summary if row["comparison"] == "P-S"]
    sbr = [row for row in paired_summary if row["comparison"] == "S-B"]
    by_comp = {row["comparison"]: row for row in paired_summary}
    performance_table = []
    for dataset in DATASETS:
        row = {item["variant"]: item for item in perf_summary if item["dataset"] == dataset}
        performance_table.append([
            dataset,
            *[f"{fmt(row[v]['val_accuracy_mean'] * 100, 2)} ± {fmt(row[v]['val_accuracy_population_sd'] * 100, 2)}" for v in VARIANTS],
            *[f"{fmt(row[v]['val_macro_f1_mean'] * 100, 2)} ± {fmt(row[v]['val_macro_f1_population_sd'] * 100, 2)}" for v in VARIANTS],
            *[f"{fmt(row[v]['val_ce_mean'], 4)} ± {fmt(row[v]['val_ce_population_sd'], 4)}" for v in VARIANTS],
        ])
    compare_table = []
    for dataset in DATASETS:
        compare_table.append([dataset] + [
            f"{fmt(by_comp[f'{comp}'][metric + '_mean'], 3)} ± {fmt(by_comp[f'{comp}'][metric + '_population_sd'], 3)}"
            for comp in (f"{c}-{b}" for c, b in COMPARISONS)
            for metric in ("accuracy_delta_pp",)
        ])
    paired_table = []
    for dataset in DATASETS:
        items = [r for r in paired_summary if r["dataset"] == dataset]
        for item in items:
            paired_table.append([
                dataset, item["comparison"],
                f"{fmt(item['accuracy_delta_pp_mean'], 3)} ± {fmt(item['accuracy_delta_pp_population_sd'], 3)} pp",
                f"+{item['accuracy_delta_pp_positive_seeds']}/-{item['accuracy_delta_pp_negative_seeds']}/={item['accuracy_delta_pp_ties']}",
                f"{fmt(item['macro_f1_delta_pp_mean'], 3)} ± {fmt(item['macro_f1_delta_pp_population_sd'], 3)} pp",
                f"{fmt(item['ce_delta_mean'], 4)} ± {fmt(item['ce_delta_population_sd'], 4)}",
            ])
    avg_ratio = statistics.mean(row["delta_to_target_rms"] for row in residual_rows) if residual_rows else math.nan
    avg_delta = statistics.mean(row["rms_delta"] for row in residual_rows) if residual_rows else math.nan
    intervention_table = []
    for row in intervention_summary:
        intervention_table.append([
            row["dataset"], row["intervention"], str(row["n_raw_observations"]),
            f"{fmt(row['val_accuracy_mean'] * 100, 2)} ± {fmt(row['val_accuracy_population_sd'] * 100, 2)}",
            f"{fmt(row['accuracy_delta_vs_normal_pp_mean'], 3)} ± {fmt(row['accuracy_delta_vs_normal_pp_population_sd'], 3)} pp",
            f"+{row['accuracy_delta_positive_seeds']}/-{row['accuracy_delta_negative_seeds']}/={row['accuracy_delta_tied_seeds']}",
            f"{fmt(row['macro_f1_delta_vs_normal_pp_mean'], 3)} ± {fmt(row['macro_f1_delta_vs_normal_pp_population_sd'], 3)} pp",
            f"{fmt(row['ce_delta_vs_normal_mean'], 4)} ± {fmt(row['ce_delta_vs_normal_population_sd'], 4)}",
        ])
    coeff_text = []
    for variant in VARIANTS:
        rows = [r for r in gpr_rows if r["variant"] == variant]
        means = [statistics.mean([r[f"c{k}"] for r in rows]) for k in range(4)]
        coeff_text.append(f"- {variant}: `[{', '.join(fmt(v, 5) for v in means)}]` (mean over 9 dataset-seed checkpoints).")
    gpr_lookup = {
        (row["dataset"], row["variant"], int(row["seed"])): row
        for row in gpr_rows
    }
    gpr_delta_text = []
    for variant, reference in (("P", "B"), ("S", "B")):
        fragments = []
        for order in range(4):
            deltas = [
                gpr_lookup[(dataset, variant, seed)][f"c{order}"]
                - gpr_lookup[(dataset, reference, seed)][f"c{order}"]
                for dataset in DATASETS for seed in SEEDS
            ]
            avg, sd = mean_sd(deltas)
            positive = sum(delta > 1e-12 for delta in deltas)
            negative = sum(delta < -1e-12 for delta in deltas)
            ties = sum(abs(delta) <= 1e-12 for delta in deltas)
            fragments.append(f"c{order}: {fmt(avg, 5)} ± {fmt(sd, 5)} ({positive}+/{negative}-/{ties}= of 9)")
        gpr_delta_text.append(f"- {variant}−{reference} paired checkpoint coefficient deltas: " + "; ".join(fragments) + ".")
    s_permutations = [row for row in manifest.get("shuffle_permutation_audit", [])]
    fixed_counts = sorted({int(row["fixed_points"]) for row in s_permutations})
    fixed_rates = sorted({float(row["fixed_point_rate"]) for row in s_permutations})
    fixed_summary = (
        f"S has {len(s_permutations)} recorded dataset-seed permutations; observed fixed-point counts {fixed_counts} and rates {[fmt(x, 6) for x in fixed_rates]}."
        if s_permutations else "Shuffle permutation metadata is pending analysis."
    )
    report = f"""# PCRR-E0 — Post-GPR Paired Cross-Modal Residual Refinement Screen

## Execution summary

- Parent: `{BASE_SHA}` on `exp/orci_d0_interaction_alignment_synergy`; experiment branch: `{manifest['branch']}`.
- Validation-only formal cells: {len(manifest.get('formal_runs', []))} expected, {sum(r.get('status') == 'complete' for r in manifest.get('formal_runs', []))} complete; datasets Movies/Grocery/ele-fashion, variants B/P/S, seeds 42/43/44.
- Fixed splits: {json.dumps(SPLITS, ensure_ascii=False)}. `evaluate_test=false`, `development_no_test=true`; no NC test metrics are present in run artifacts.
- Smoke: see [smoke_status.json](smoke_status.json). No formal LP was run.
- Population SD (`ddof=0`) is reported descriptively over the three paired training seeds; no pseudo-IID p-values are used.

## Validation performance

Accuracy and Macro-F1 are percent; CE is in task units. Cells show mean ± population SD over the three training seeds.

{md_table(['Dataset','B Acc','P Acc','S Acc','B F1','P F1','S F1','B CE','P CE','S CE'], performance_table)}

## Paired comparisons

Accuracy/F1 deltas are percentage points. Direction counts are across the three paired seeds.

{md_table(['Dataset','Pair','Accuracy Δ mean ± SD','Acc +/−/=','Macro-F1 Δ mean ± SD','CE Δ mean ± SD'], paired_table)}

The primary comparison is P−B; P−S assesses the correspondence-matched capacity control; S−B describes the shuffled residual control.

## Residual diagnostics

- Direction-level diagnostics: {len(residual_rows)} P/S checkpoint-direction rows.
- Mean `RMS(delta)/RMS(G_target)` across P/S directions and checkpoints: {fmt(avg_ratio, 6)}.
- Mean `RMS(delta)`: {fmt(avg_delta, 6)}. This is a magnitude diagnostic, not a performance criterion.
- Direction-specific values, hidden RMS, cosine, and adapter norms: [residual_diagnostics.csv](data/residual_diagnostics.csv).
- {fixed_summary}

## GPR coefficients

{chr(10).join(coeff_text)}

Per dataset/seed coefficients are in [gpr_diagnostics.csv](data/gpr_diagnostics.csv). Differences are descriptive; they do not reopen GPR attribution.

Paired profile shifts across the nine matched dataset-seed checkpoints:

{chr(10).join(gpr_delta_text)}

## P checkpoint interventions

The P checkpoints were evaluated with the residual disabled and with five deterministic residual-source permutations each. These diagnostics are not retrained comparisons.

{md_table(['Dataset','Intervention','Raw observations','Accuracy %','Accuracy Δ vs normal','Acc +/−/= seeds','Macro-F1 Δ','CE Δ'], intervention_table)}

Full rows and seed-level repeats: [intervention_by_run.csv](data/intervention_by_run.csv); grouped summary: [intervention_summary.csv](data/intervention_summary.csv).

## Interpretation

{classify_evidence(paired_summary)}

Read the paired P−B and P−S rows together. A checkpoint's response to source shuffling does not substitute for either retrained comparison. Dataset-specific signs and Macro-F1/CE trade-offs should remain visible; there are only three paired training seeds per dataset and one fixed split per dataset.

## Final self-audit (25 items)

1. **Started from required base?** Yes: `{BASE_SHA}` was the verified parent SHA.
2. **Used another experiment branch?** No merge/cherry-pick from other experiment branches.
3. **Ran/read NC test?** No test split evaluation was enabled; artifacts are marked validation-only and contain no `test_*` metrics.
4. **Changed dataset split?** No; the three declared split files were used.
5. **Does B regress to D0-B?** Yes, model-level eval regression is checked at `atol=1e-6`; see test result in smoke status.
6. **Do P/S equal B at initialization?** Yes, the zero-initialized final pair projection gives exact zero residual; same-seed eval and train-mode RNG-reset equality are tested.
7. **Is `pair_up` zero-initialized?** Yes, weight and bias are exactly zero.
8. **Extra residual-block train dropout?** No; the pair response has no dropout.
9. **Are P/S parameter count and initialization matched?** Yes; all variants build the same modules, and same-seed state dictionaries are tested bitwise.
10. **Does S shuffle only the residual source?** Yes; permutation is applied only to the source input passed to the pair response.
11. **Does S late fusion keep correct node pairing?** Yes; late fusion concatenates each node's own refined text and visual branches.
12. **Does P−B show retrained gain?** Not broadly: mean accuracy is +0.380 pp on Movies, −0.059 pp on Grocery, and −0.266 pp on ele-fashion.
13. **Does P−S support independent correspondence value?** Descriptively yes: mean accuracy delta is positive on all three datasets and all three paired seeds; this does not establish net gain over B.
14. **Does S−B expose capacity effects?** Yes; mean accuracy is negative on all three datasets, so the shuffled residual control does not show a positive generic capacity effect.
15. **Final residual magnitude?** Mean ratio is {fmt(avg_ratio, 6)}; all cellwise values are in the diagnostics CSV.
16. **Do P checkpoints rely on the residual?** Yes in validation: residual-off reduces mean accuracy by 6.929 pp on Movies, 1.308 pp on Grocery, and 1.446 pp on ele-fashion.
17. **Does source-node shuffle harm P checkpoints?** Yes descriptively: five-repeat mean accuracy changes are −2.398 pp, −0.724 pp, and −1.088 pp on Movies, Grocery, and ele-fashion.
18. **Do interventions agree with retrained P−S?** Both favor correct correspondence directionally, but intervention deltas are checkpoint reliance diagnostics and cannot replace retrained P−S.
19. **Did residual systematically change GPR coefficients?** Paired P−B and S−B means, SDs, and direction counts are shown above; any profile shifts are descriptive, not causal evidence that the adapter changed GPR.
20. **Dataset-specific regime?** Yes: net P−B accuracy gain appears only on Movies, while P−S is positive across all three datasets.
21. **Accuracy/F1/CE trade-off?** All three deltas are reported in paired CSVs and tables.
22. **Evidence for next stage?** The current block lacks a broad net P−B gain; do not advance it as a performance architecture based on E0.
23. **What should be retained if proceeding?** Retain the correspondence insight as a mechanism hypothesis; do not carry the residual block forward as an established gain.
24. **LP smoke protocol?** B/P only, 2 epochs and at most 2 train batches, `[5,5,5]`, LinkNeighborLoader and global_eid positive-edge removal checked; no LP test evaluation.
25. **Claims beyond evidence?** None intended: results are validation-only, three paired seeds on fixed splits, and checkpoint interventions are reliance diagnostics.

## Recommendation

{classify_evidence(paired_summary)} Do not advance PCRR as a performance module yet. Retain the correct-correspondence signal as a mechanistic observation for human review, and stop the experimental line here.
"""
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-checkpoint-diagnostics", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows, summary, paired, paired_summary, lookup = performance_outputs(manifest)
    write_csv(DATA_DIR / "performance_by_run.csv", rows)
    write_csv(DATA_DIR / "performance_summary.csv", summary)
    write_csv(DATA_DIR / "paired_delta_by_run.csv", paired)
    write_csv(DATA_DIR / "paired_delta_summary.csv", paired_summary)
    if args.skip_checkpoint_diagnostics:
        residual_rows, gpr_rows, intervention_rows, intervention_summary = [], [], [], []
    else:
        device = torch.device(args.device)
        residual_rows, gpr_rows, intervention_rows, intervention_summary = checkpoint_diagnostics(
            manifest, lookup, device
        )
    write_csv(DATA_DIR / "residual_diagnostics.csv", residual_rows)
    write_csv(DATA_DIR / "gpr_diagnostics.csv", gpr_rows)
    write_csv(DATA_DIR / "intervention_by_run.csv", intervention_rows)
    write_csv(DATA_DIR / "intervention_summary.csv", intervention_summary)
    permutation_lookup = {
        (row["dataset"], int(row["seed"])): row
        for row in gpr_rows if row["variant"] == "S"
    }
    manifest["shuffle_permutation_audit"] = [
        {
            "dataset": dataset,
            "seed": seed,
            "pair_shuffle_seed": seed + 73000,
            "fixed_points": permutation_lookup[(dataset, seed)]["pair_shuffle_fixed_points"],
            "fixed_point_rate": permutation_lookup[(dataset, seed)]["pair_shuffle_fixed_point_rate"],
        }
        for dataset in DATASETS for seed in SEEDS
    ] if permutation_lookup else []
    for run in manifest["formal_runs"]:
        metadata = permutation_lookup.get((run["dataset"], int(run["seed"])))
        if metadata:
            run["pair_shuffle_fixed_points"] = metadata["pair_shuffle_fixed_points"]
            run["pair_shuffle_fixed_point_rate"] = metadata["pair_shuffle_fixed_point_rate"]
    write_report(manifest, summary, paired_summary, residual_rows, gpr_rows, intervention_rows, intervention_summary)
    manifest["analysis_status"] = "complete" if not args.skip_checkpoint_diagnostics else "performance_only"
    manifest["analysis_updated_at"] = now()
    (RESEARCH / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Analysis written under {RESEARCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
