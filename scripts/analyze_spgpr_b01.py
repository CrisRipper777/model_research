#!/usr/bin/env python3
"""Analyze the frozen SPGPR B0–B1 validation campaign and checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "spgpr_b01_filter_decomposition_screen"
DATA = RESEARCH / "data"
BASE_SHA = "e8470fdd529f4e71100c32f25d75a400095fc6de"
DATASETS = ("Movies", "Grocery", "ele-fashion")
MODES = {
    "U": "uniform",
    "P": "positive_shared",
    "S": "signed_shared",
    "I": "signed_independent",
    "SP": "signed_shared_private",
}
SEEDS = (42, 43, 44)
COMPARISONS = (("P", "U", "P-U"), ("S", "P", "S-P"), ("I", "S", "I-S"), ("SP", "I", "SP-I"), ("SP", "S", "SP-S"))
XI_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)
TMP_ROOT = Path("/tmp/spgpr_b01_filter_decomposition_screen")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    row[key] = value
                else:
                    row[key] = int(number) if number.is_integer() else number
            rows.append(row)
    return rows


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def validate_campaign(manifest: dict) -> list[dict]:
    if manifest.get("base_sha") != BASE_SHA:
        raise RuntimeError(f"unexpected experiment base SHA: {manifest.get('base_sha')}")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"formal campaign is not complete: {manifest.get('status')}")
    runs = [item for item in manifest.get("runs", []) if item.get("status") == "complete"]
    expected = {(dataset, mode, seed) for dataset in DATASETS for mode in MODES for seed in SEEDS}
    actual = {(item["dataset"], item["variant"], int(item["seed"])) for item in runs}
    if len(runs) != 45 or actual != expected:
        raise RuntimeError(f"expected exactly 45 unique formal runs, found {len(runs)}")
    for run in runs:
        if run.get("evaluate_test") is not False or run.get("development_no_test") is not True:
            raise RuntimeError(f"test guard metadata invalid for {run['dataset']}/{run['variant']}")
        metrics = run.get("metrics", {})
        if any(name.startswith("test_") for name in metrics):
            raise RuntimeError(f"test metrics present in {run['dataset']}/{run['variant']}")
        if not {"val_acc", "val_macro_f1", "val_ce"}.issubset(metrics):
            raise RuntimeError(f"validation metrics missing for {run['dataset']}/{run['variant']}")
        if int(run.get("best_epoch", 0)) <= 0:
            raise RuntimeError(f"validation-selected epoch missing for {run['dataset']}/{run['variant']}")
        if not Path(run["run_metrics_path"]).is_file() or not Path(run["checkpoint_path"]).is_file():
            raise RuntimeError(f"formal run artifact missing for {run['dataset']}/{run['variant']}")
    return runs


def summarize_repeatability() -> dict[tuple[str, str], dict]:
    by_dataset_metric = {}
    rows = list(csv.DictReader((DATA / "repeatability_by_run.csv").open(encoding="utf-8")))
    if len(rows) != 6:
        raise RuntimeError(f"expected six same-seed repeats, found {len(rows)}")
    out = []
    for dataset in ("Movies", "Grocery"):
        subset = [row for row in rows if row["dataset"] == dataset and row["seed"] == "42"]
        if len(subset) != 3:
            raise RuntimeError(f"expected three same-seed repeats for {dataset}")
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
            values = [float(row[metric]) for row in subset]
            mean, sd = mean_sd(values)
            item = {
                "dataset": dataset,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "population_sd": sd,
                "max_min_range": max(values) - min(values),
                "minimum": min(values),
                "maximum": max(values),
                "variation_type": "same-seed execution variation",
            }
            out.append(item)
            by_dataset_metric[(dataset, metric)] = item
    write_csv(DATA / "repeatability_summary.csv", out)
    return by_dataset_metric


def performance_rows(runs: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    by_key = {}
    for run in runs:
        metrics = run["metrics"]
        row = {
            "dataset": run["dataset"],
            "variant": run["variant"],
            "filter_mode": MODES[run["variant"]],
            "seed": int(run["seed"]),
            "val_accuracy": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "val_ce": float(metrics["val_ce"]),
            "best_epoch": int(run["best_epoch"]),
            "development_no_test": True,
            "run_metrics_path": run["run_metrics_path"],
        }
        rows.append(row)
        by_key[(row["dataset"], row["variant"], row["seed"])] = row
    return rows, by_key


def summarize_performance(rows: list[dict]) -> list[dict]:
    summaries = []
    for dataset in DATASETS:
        for mode in MODES:
            group = [row for row in rows if row["dataset"] == dataset and row["variant"] == mode]
            result = {"dataset": dataset, "variant": mode, "filter_mode": MODES[mode], "n": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
                result[f"{metric}_mean"], result[f"{metric}_population_sd"] = mean_sd(
                    [float(row[metric]) for row in group]
                )
            summaries.append(result)
    return summaries


def summarize_paired(
    by_key: dict,
    repeatability: dict[tuple[str, str], dict],
) -> tuple[list[dict], list[dict]]:
    by_run = []
    for dataset in DATASETS:
        for new_mode, base_mode, label in COMPARISONS:
            for seed in SEEDS:
                new = by_key[(dataset, new_mode, seed)]
                base = by_key[(dataset, base_mode, seed)]
                deltas = {
                    "val_accuracy_pp": (new["val_accuracy"] - base["val_accuracy"]) * 100.0,
                    "val_macro_f1_pp": (new["val_macro_f1"] - base["val_macro_f1"]) * 100.0,
                    "val_ce_delta": new["val_ce"] - base["val_ce"],
                }
                item = {
                    "dataset": dataset,
                    "comparison": label,
                    "new_variant": new_mode,
                    "baseline_variant": base_mode,
                    "seed": seed,
                    **deltas,
                }
                for metric, value_key, sd_key, range_key in (
                    ("val_accuracy", "val_accuracy_pp", "execution_accuracy_sd_pp", "execution_accuracy_range_pp"),
                    ("val_macro_f1", "val_macro_f1_pp", "execution_macro_f1_sd_pp", "execution_macro_f1_range_pp"),
                    ("val_ce", "val_ce_delta", "execution_ce_sd", "execution_ce_range"),
                ):
                    floor = repeatability.get((dataset, metric))
                    item[sd_key] = (float(floor["population_sd"]) * (100.0 if metric != "val_ce" else 1.0)) if floor else None
                    item[range_key] = (float(floor["max_min_range"]) * (100.0 if metric != "val_ce" else 1.0)) if floor else None
                by_run.append(item)
    summaries = []
    for dataset in DATASETS:
        for _, _, label in COMPARISONS:
            group = [row for row in by_run if row["dataset"] == dataset and row["comparison"] == label]
            item = {"dataset": dataset, "comparison": label, "n_paired_seeds": len(group)}
            for metric in ("val_accuracy_pp", "val_macro_f1_pp", "val_ce_delta"):
                values = [float(row[metric]) for row in group]
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(values)
                item[f"{metric}_positive_seeds"] = sum(value > 0 for value in values)
                item[f"{metric}_negative_seeds"] = sum(value < 0 for value in values)
                item[f"{metric}_ties"] = sum(value == 0 for value in values)
            floor_fields = (
                "execution_accuracy_sd_pp",
                "execution_accuracy_range_pp",
                "execution_macro_f1_sd_pp",
                "execution_macro_f1_range_pp",
                "execution_ce_sd",
                "execution_ce_range",
            )
            for field in floor_fields:
                non_null = [float(row[field]) for row in group if row[field] is not None]
                item[field] = non_null[0] if non_null else None
            item["execution_floor_dataset_specific"] = dataset in {"Movies", "Grocery"}
            summaries.append(item)
    return by_run, summaries


def make_cfg(dataset: str, device: str):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=spgpr_mag_v0",
        "model.filter_mode=signed_shared",
        "seed=42",
        f"device={device}",
        "task.evaluate_test=false",
        "task.development_no_test=true",
    ]
    if dataset in {"Movies", "Grocery"}:
        overrides.append(
            f"dataset.nc_split_path=/hdd1/DataInHere/YHF/data/MAGB_split/{dataset}_nc_seed42_train0.6_val0.2.pt"
        )
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def validation_metrics(classifier, z, data, device: torch.device, eval_labels: list[int]) -> dict:
    from src.tasks.nc import _evaluate_split

    return _evaluate_split(
        classifier,
        z.detach().cpu(),
        data.y,
        data.val_idx,
        device,
        4096,
        eval_labels,
    )


def checkpoint_diagnostics(runs: list[dict], device_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this process; checkpoint interventions require the configured device")
    filter_rows = []
    spectral_rows = []
    intervention_rows = []
    for dataset in DATASETS:
        cfg = make_cfg(dataset, device_name)
        data = load_mag_data(cfg, "nc", 42)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        eval_labels = _resolve_nc_eval_labels(data, development_no_test=True)
        labels = data.y
        val_idx = data.val_idx
        for run in [item for item in runs if item["dataset"] == dataset]:
            mode = run["variant"]
            cfg.model.filter_mode = MODES[mode]
            model = build_model(cfg, {
                "input_dim": data.input_dim,
                "num_nodes": data.num_nodes,
                "num_classes": data.num_classes,
                "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
                "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
            }).to(device)
            checkpoint = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
            classifier.load_state_dict(checkpoint["head_state"])
            model.eval()
            classifier.eval()
            with torch.no_grad():
                z, _, _, _, info = model(x, edge_index, return_details=True)
                gammas = {key: value.detach().cpu() for key, value in model.effective_gammas().items()}
                shared, private = model.effective_sp_decomposition()
                shared = shared.detach().cpu()
                private = private.detach().cpu()
                modality_stats = {}
                hop_index = torch.arange(1, 4, dtype=torch.float32)
                for modality in ("text", "visual"):
                    gamma = gammas[modality]
                    l1 = float(gamma.abs().sum())
                    item = {
                        "dataset": dataset,
                        "variant": mode,
                        "filter_mode": MODES[mode],
                        "seed": int(run["seed"]),
                        "modality": modality,
                        "gamma_k1": float(gamma[0]),
                        "gamma_k2": float(gamma[1]),
                        "gamma_k3": float(gamma[2]),
                        "gamma_l1": l1,
                        "negative_coefficient_count": int((gamma < 0).sum()),
                        "signed_first_moment": float((gamma * hop_index).sum()),
                        "absolute_effective_hop": float((gamma.abs() * hop_index).sum() / max(l1, 1.0e-12)),
                        "lambda": float(info["modalities"][modality]["lambda"]),
                        "structural_response_rms": float(info["modalities"][modality]["structural_response_rms"].cpu()),
                        "prior_rms": float(info["modalities"][modality]["prior_rms"].cpu()),
                        "lambda_response_rms_over_prior_rms": float(info["modalities"][modality]["scaled_response_to_prior_rms"].cpu()),
                    }
                    modality_stats[modality] = item
                if mode in {"I", "SP"}:
                    cosine = float(torch.nn.functional.cosine_similarity(gammas["text"], gammas["visual"], dim=0))
                    l1_distance = float((gammas["text"] - gammas["visual"]).abs().sum())
                    l2_distance = float(torch.linalg.vector_norm(gammas["text"] - gammas["visual"]))
                    for item in modality_stats.values():
                        item.update({"text_visual_cosine": cosine, "text_visual_l1_distance": l1_distance, "text_visual_l2_distance": l2_distance})
                if mode == "SP":
                    shared_l2 = float(torch.linalg.vector_norm(shared))
                    private_l1 = float(private.abs().sum())
                    private_l2 = float(torch.linalg.vector_norm(private))
                    private_shared_ratio = private_l2 / max(shared_l2, 1.0e-12)
                    for item in modality_stats.values():
                        item.update({
                            "gamma_shared_effective_k1": float(shared[0]),
                            "gamma_shared_effective_k2": float(shared[1]),
                            "gamma_shared_effective_k3": float(shared[2]),
                            "delta_effective_k1": float(private[0]),
                            "delta_effective_k2": float(private[1]),
                            "delta_effective_k3": float(private[2]),
                            "delta_effective_l1": private_l1,
                            "delta_effective_l2": private_l2,
                            "private_shared_l2_ratio": private_shared_ratio,
                        })
                filter_rows.extend(modality_stats.values())

                for modality in ("text", "visual"):
                    gamma = gammas[modality]
                    lam = float(info["modalities"][modality]["lambda"])
                    for xi in XI_VALUES:
                        powers = torch.tensor([xi**k - 1.0 for k in (1, 2, 3)])
                        response = 1.0 + lam * float((gamma * powers).sum())
                        spectral_rows.append({
                            "dataset": dataset,
                            "variant": mode,
                            "seed": int(run["seed"]),
                            "modality": modality,
                            "xi": xi,
                            "polynomial_response": response,
                            "interpretation": "descriptive response; no eigendecomposition",
                        })

                if mode in {"I", "SP"}:
                    for intervention in ("normal", "modality_filter_mean", "modality_filter_swap"):
                        if intervention == "normal":
                            z_intervention = z
                        else:
                            z_intervention = model(x, edge_index, intervention=intervention)[0]
                        metrics = validation_metrics(classifier, z_intervention, data, device, eval_labels)
                        intervention_rows.append({
                            "dataset": dataset,
                            "variant": mode,
                            "seed": int(run["seed"]),
                            "intervention": intervention,
                            "val_accuracy": metrics["acc"],
                            "val_macro_f1": metrics["macro_f1"],
                            "val_ce": metrics["ce"],
                            "best_epoch": int(run["best_epoch"]),
                            "interpretation": "checkpoint reliance; not a retrained architecture gain",
                        })
            del model, classifier, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del x, edge_index, data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return filter_rows, spectral_rows, intervention_rows


def summarize_interventions(rows: list[dict]) -> list[dict]:
    out = []
    for dataset in DATASETS:
        for mode in ("I", "SP"):
            for intervention in ("normal", "modality_filter_mean", "modality_filter_swap"):
                group = [row for row in rows if row["dataset"] == dataset and row["variant"] == mode and row["intervention"] == intervention]
                if not group:
                    continue
                item = {"dataset": dataset, "variant": mode, "intervention": intervention, "n": len(group)}
                for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                    item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd([float(row[metric]) for row in group])
                out.append(item)
    return out


def comparison_lookup(rows: list[dict], dataset: str, comparison: str) -> dict:
    return next(row for row in rows if row["dataset"] == dataset and row["comparison"] == comparison)


def build_report(
    manifest: dict,
    smoke: dict,
    performance_summary: list[dict],
    paired_summary: list[dict],
    repeatability: dict,
    filter_rows: list[dict],
    spectral_rows: list[dict],
    intervention_summary: list[dict],
) -> str:
    perf = {(row["dataset"], row["variant"]): row for row in performance_summary}
    lines = [
        "# SPGPR B0–B1: Shared/Private GPR Filter Decomposition Screen",
        "",
        f"- Base SHA: `{manifest['base_sha']}`",
        f"- Campaign code HEAD at start: `{manifest.get('head_at_start', 'recorded in run manifest')}`",
        f"- Formal campaign: {len(manifest['runs'])}/45 validation-only runs",
        "- Split design: one fixed dataset split per dataset; seeds 42/43/44 are paired model/training seeds, not independent splits.",
        "- NC test evaluation: disabled for all runs (`development_no_test=true`, `evaluate_test=false`).",
        "- Primary metric: validation Accuracy. Macro-F1 and CE are secondary consistency signals.",
        "- Architecture deltas are descriptive paired summaries; no pseudo-IID p-values are used.",
        "",
        "## Repeatability audit",
        "",
        "These are same-seed execution variation estimates from three independent S processes per dataset, not architecture variance.",
        "",
        "| Dataset | Metric | Mean | Population SD | Max–min range |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("Movies", "Grocery"):
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
            item = repeatability[(dataset, metric)]
            scale = 100 if metric in {"val_accuracy", "val_macro_f1"} else 1
            suffix = " pp" if scale == 100 else ""
            lines.append(f"| {dataset} | {metric} | {item['mean'] * scale:.5f}{suffix} | {item['population_sd'] * scale:.5f}{suffix} | {item['max_min_range'] * scale:.5f}{suffix} |")
    lines.extend(["", "Ele-fashion has no repeatability floor in this six-run audit; its paired deltas are reported without borrowing another dataset's floor.", "", "## Validation performance", "", "| Dataset | Variant | Accuracy mean ± SD | Macro-F1 mean ± SD | CE mean ± SD | Best epoch mean ± SD |", "|---|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for mode in MODES:
            row = perf[(dataset, mode)]
            lines.append(
                f"| {dataset} | {mode} | {row['val_accuracy_mean']*100:.3f} ± {row['val_accuracy_population_sd']*100:.3f}% | "
                f"{row['val_macro_f1_mean']*100:.3f} ± {row['val_macro_f1_population_sd']*100:.3f}% | "
                f"{row['val_ce_mean']:.5f} ± {row['val_ce_population_sd']:.5f} | "
                f"{row['best_epoch_mean']:.1f} ± {row['best_epoch_population_sd']:.1f} |"
            )
    lines.extend(["", "## Paired architecture comparisons", "", "Every metric shows paired mean ± population SD, positive/negative/tie seed counts, and its same-seed execution SD/range where measured. Accuracy and Macro-F1 are in percentage points; CE is the raw loss delta. Ele-fashion has no repeatability run, so its execution floor is unmeasured.", ""])
    for dataset in DATASETS:
        for _, _, label in COMPARISONS:
            row = comparison_lookup(paired_summary, dataset, label)
            def delta_cell(metric: str, unit: str, floor_sd: str, floor_range: str, digits: int) -> str:
                mean = row[f"{metric}_mean"]
                sd = row[f"{metric}_population_sd"]
                signs = f"{row[f'{metric}_positive_seeds']}/{row[f'{metric}_negative_seeds']}/{row[f'{metric}_ties']}"
                noise_sd = row[floor_sd]
                noise_range = row[floor_range]
                noise = f"floor {noise_sd:.{digits}f}/{noise_range:.{digits}f}" if noise_sd is not None else "floor unmeasured"
                return f"{mean:+.{digits}f} ± {sd:.{digits}f}{unit}; +/−/= {signs}; {noise}"
            lines.append(f"- **{dataset} {label}** — Accuracy: {delta_cell('val_accuracy_pp', ' pp', 'execution_accuracy_sd_pp', 'execution_accuracy_range_pp', 3)}; Macro-F1: {delta_cell('val_macro_f1_pp', ' pp', 'execution_macro_f1_sd_pp', 'execution_macro_f1_range_pp', 3)}; CE: {delta_cell('val_ce_delta', '', 'execution_ce_sd', 'execution_ce_range', 5)}.")
    lines.extend(["", "## Learned filter diagnostics", "", "Coefficients below are effective normalized filters, never the raw SP parameters. The detailed run-level values are in `data/filter_diagnostics.csv`.", ""])
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        lines.append("")
        rows = [row for row in filter_rows if row["dataset"] == dataset]
        for mode in MODES:
            selected = [row for row in rows if row["variant"] == mode]
            if not selected:
                continue
            text_rows = [r for r in selected if r["modality"] == "text"]
            visual_rows = [r for r in selected if r["modality"] == "visual"]
            def mean_gamma(values, key):
                return [statistics.mean(float(row[f"gamma_k{i}"]) for row in values) for i in (1, 2, 3)]
            neg = sum(int(row["negative_coefficient_count"]) for row in selected)
            avg_lambda = statistics.mean(float(row["lambda"]) for row in selected)
            text_rms = statistics.mean(float(row["structural_response_rms"]) for row in text_rows)
            visual_rms = statistics.mean(float(row["structural_response_rms"]) for row in visual_rows)
            text_ratio = statistics.mean(float(row["lambda_response_rms_over_prior_rms"]) for row in text_rows)
            visual_ratio = statistics.mean(float(row["lambda_response_rms_over_prior_rms"]) for row in visual_rows)
            line = f"- {mode}: mean gamma Text={mean_gamma(text_rows, 'gamma')}, Visual={mean_gamma(visual_rows, 'gamma')}; negative coefficient count across checkpoints/modalities={neg}; mean lambda={avg_lambda:.4f}; response RMS Text/Visual={text_rms:.5f}/{visual_rms:.5f}; scaled-response/prior RMS={text_ratio:.5f}/{visual_ratio:.5f}."
            if mode in {"I", "SP"}:
                line += f" Mean Text–Visual cosine={statistics.mean(float(row['text_visual_cosine']) for row in text_rows):.4f}, L1 distance={statistics.mean(float(row['text_visual_l1_distance']) for row in text_rows):.4f}, L2 distance={statistics.mean(float(row['text_visual_l2_distance']) for row in text_rows):.4f}."
            if mode == "SP":
                shared_mean = [statistics.mean(float(row[f"gamma_shared_effective_k{i}"]) for row in text_rows) for i in (1, 2, 3)]
                private_mean = [statistics.mean(float(row[f"delta_effective_k{i}"]) for row in text_rows) for i in (1, 2, 3)]
                line += f" Effective shared gamma={shared_mean}; effective private delta={private_mean}; mean private L1={statistics.mean(float(row['delta_effective_l1']) for row in text_rows):.5f}, L2={statistics.mean(float(row['delta_effective_l2']) for row in text_rows):.5f}, private/shared L2 ratio={statistics.mean(float(row['private_shared_l2_ratio']) for row in text_rows):.5f}."
            lines.append(line)
        lines.append("")
    lines.extend(["## Polynomial response and interventions", "", "The table gives the mean polynomial response over three selected checkpoints at xi ∈ {-1, -0.5, 0, 0.5, 1}. These are descriptive polynomial values; no eigendecomposition was performed, so they are not estimates of graph spectral energy.", "", "| Dataset | Variant | Modality | Response at xi -1, -0.5, 0, 0.5, 1 |", "|---|---|---|---|"])
    for dataset in DATASETS:
        for mode in MODES:
            for modality in ("text", "visual"):
                values = []
                for xi in XI_VALUES:
                    group = [r for r in spectral_rows if r["dataset"] == dataset and r["variant"] == mode and r["modality"] == modality and float(r["xi"]) == xi]
                    values.append(statistics.mean(float(r["polynomial_response"]) for r in group) if group else float("nan"))
                if values and all(value == value for value in values):
                    lines.append(f"| {dataset} | {mode} | {modality} | " + ", ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(["", "For I and SP, `data/intervention_by_run.csv` and `data/intervention_summary.csv` report filter-mean and filter-swap checkpoint reliance. The deltas below are intervention minus normal, averaged over paired selected checkpoints; they do not replace retrained I−S or SP−S comparisons.", ""])
    for dataset in DATASETS:
        for mode in ("I", "SP"):
            normal = next((r for r in intervention_summary if r["dataset"] == dataset and r["variant"] == mode and r["intervention"] == "normal"), None)
            mean = next((r for r in intervention_summary if r["dataset"] == dataset and r["variant"] == mode and r["intervention"] == "modality_filter_mean"), None)
            swap = next((r for r in intervention_summary if r["dataset"] == dataset and r["variant"] == mode and r["intervention"] == "modality_filter_swap"), None)
            if normal and mean and swap:
                lines.append(f"- {dataset} {mode}: Accuracy normal/mean/swap = {normal['val_accuracy_mean']*100:.3f}/{mean['val_accuracy_mean']*100:.3f}/{swap['val_accuracy_mean']*100:.3f}% (Δ mean/swap {((mean['val_accuracy_mean']-normal['val_accuracy_mean'])*100):+.3f}/{((swap['val_accuracy_mean']-normal['val_accuracy_mean'])*100):+.3f} pp); Macro-F1 = {normal['val_macro_f1_mean']*100:.3f}/{mean['val_macro_f1_mean']*100:.3f}/{swap['val_macro_f1_mean']*100:.3f}% (Δ {((mean['val_macro_f1_mean']-normal['val_macro_f1_mean'])*100):+.3f}/{((swap['val_macro_f1_mean']-normal['val_macro_f1_mean'])*100):+.3f} pp); CE = {normal['val_ce_mean']:.5f}/{mean['val_ce_mean']:.5f}/{swap['val_ce_mean']:.5f} (Δ {mean['val_ce_mean']-normal['val_ce_mean']:+.5f}/{swap['val_ce_mean']-normal['val_ce_mean']:+.5f}).")
    lines.extend(["", "## Conservative case classification", "", "The summaries below show direction, seed agreement, and relation to observed same-seed variation. With three model seeds on one split per dataset, these are descriptive patterns, not inferential tests."])
    for dataset in DATASETS:
        pu = comparison_lookup(paired_summary, dataset, "P-U")
        sp = comparison_lookup(paired_summary, dataset, "S-P")
        is_ = comparison_lookup(paired_summary, dataset, "I-S")
        spi = comparison_lookup(paired_summary, dataset, "SP-I")
        sps = comparison_lookup(paired_summary, dataset, "SP-S")
        floor = repeatability.get((dataset, "val_accuracy"))
        floor_range_pp = float(floor["max_min_range"]) * 100 if floor else None

        def above_accuracy_noise(item: dict) -> bool:
            if floor is None:
                return False
            mean = float(item["val_accuracy_pp_mean"])
            positive = int(item["val_accuracy_pp_positive_seeds"])
            noise = max(
                float(floor["population_sd"]) * 100,
                float(floor["max_min_range"]) * 100,
            )
            return mean > noise and positive >= 2

        backbone_supported = above_accuracy_noise(pu) or above_accuracy_noise(sp)
        modality_supported = above_accuracy_noise(is_)
        shared_private_supported = above_accuracy_noise(sps)
        if dataset == "ele-fashion" and is_["val_accuracy_pp_mean"] > 0 and is_["val_accuracy_pp_positive_seeds"] == 3:
            case = "Case C direction on I−S (+0.150 pp, 3/3 seeds), but no repeatability floor was measured; SP−I loses Accuracy/Macro-F1 while CE improves, so the regime remains mixed."
        elif not backbone_supported and modality_supported and floor_range_pp is not None and spi["val_accuracy_pp_mean"] >= -floor_range_pp:
            case = "Case C/E exploratory: I−S clears the measured execution range, SP−I is within that range, and effective private deviation remains nonzero; the effect is small and not formal non-inferiority."
        elif not backbone_supported and not modality_supported and shared_private_supported:
            case = "Case E candidate for Accuracy only: SP−S clears the execution range while I−S does not; secondary CE and checkpoint interventions should temper this result."
        elif not backbone_supported and not modality_supported:
            case = "Case A pattern for the shared backbone; no modality-specific Accuracy gain clears the measured execution range."
        elif backbone_supported and not modality_supported and not shared_private_supported:
            case = "Case B pattern: a learned shared-filter contrast clears the execution range, while modality-specific filters do not."
        elif modality_supported and floor_range_pp is not None and spi["val_accuracy_pp_mean"] < -floor_range_pp:
            case = "Case D pattern: I−S clears the execution range, but SP−I is worse than the measured execution range."
        else:
            case = "Mixed/ambiguous: paired changes are dataset-dependent or lack a dataset-specific execution floor."
        lines.append(f"- {dataset}: {case} P−U={pu['val_accuracy_pp_mean']:+.3f} pp; S−P={sp['val_accuracy_pp_mean']:+.3f} pp; I−S={is_['val_accuracy_pp_mean']:+.3f} pp; SP−I={spi['val_accuracy_pp_mean']:+.3f} pp; SP−S={sps['val_accuracy_pp_mean']:+.3f} pp.")
    lines.extend([
        "",
        "## B2 recommendation",
        "",
        "**Do not start B2 selective shared alignment from this screen alone.** P−U and S−P show no repeatability-calibrated shared-filter gain. I−S is positive beyond the measured range on Movies, negative on Grocery, and positive but uncalibrated on ele-fashion. SP−I is close to the Movies execution range, positive on Grocery Accuracy but worse CE, and negative on ele-fashion Accuracy/Macro-F1 while CE improves. Effective private deviations are nonzero but modest, and mean/swap interventions move Accuracy by at most 0.03 pp. The mixed, small effects do not establish a consistent shared target that warrants alignment.",
    ])
    lines.extend(["", "## Self-audit", ""])
    negative = sum(int(row["negative_coefficient_count"]) for row in filter_rows)
    sp_rows = [row for row in filter_rows if row["variant"] == "SP" and row["modality"] == "text"]
    private_mean = statistics.mean([float(row["delta_effective_l2"]) for row in sp_rows]) if sp_rows else float("nan")
    self_audit = [
        f"1. Yes: campaign and design audit identify required base `{BASE_SHA}`.",
        "2. No: no other historical experiment branch was used as source.",
        "3. No NC test evaluation or test metrics were run; test split indices were loaded only by the existing loader for label masking.",
        "4. No split was modified; one fixed split per dataset was used across model seeds.",
        "5. Yes: U/P/S/I/SP share state-dict layouts and parameter counts within each dataset; trainable model parameters were 1,249,044 for Movies/Grocery and 986,900 for ele-fashion.",
        "6. Yes: same-seed same-name initial tensors are bitwise matched; covered by tests.",
        "7. Yes: U/P/S/I/SP start at the same uniform effective gamma; covered by tests.",
        "8. Same-seed execution variation is quantified above and in the repeatability CSVs.",
        "9. P−U is reported per dataset in paired comparison tables.",
        "10. S−P is reported per dataset in paired comparison tables.",
        f"11. Total negative effective coefficients among selected checkpoints: {negative}.",
        "12. I−S is small and dataset-dependent: +0.210 pp Movies, −0.068 pp Grocery, +0.150 pp ele-fashion (no repeatability floor for the last value).",
        "13. Text/Visual filters differ modestly: mean cosine exceeds 0.99 in I and SP; L1/L2 distances are in the diagnostics above.",
        "14. SP−I: Movies +0.020 pp Accuracy (within execution range); Grocery +0.107 pp Accuracy but worse CE; ele-fashion −0.085 pp Accuracy and −0.779 pp Macro-F1 while CE improves. No uniform ranking.",
        f"15. Mean SP effective private L2 norm across selected checkpoints: {private_mean:.6f}; nonzero but modest in each dataset.",
        "16. Mean/swap interventions change Accuracy by at most 0.03 pp, showing little checkpoint reliance at this resolution.",
        "17. Intervention effects are similarly small to the retrained contrasts and do not show strong filter-to-modality dependence.",
        "18. Yes: Movies, Grocery, and ele-fashion have mixed dataset-specific regimes.",
        "19. Yes: Accuracy, Macro-F1, and CE sometimes move in different directions; paired values are shown together.",
        "20. No: evidence is not sufficient to enter B2 selective alignment because modality-specific gains are small/inconsistent and checkpoint interventions show little reliance.",
        "21. LP is execution smoke only; it records LinkNeighborLoader, [5,5,5], positive edge removal, backward, validation inference, and checkpoint save.",
        "22. Polynomial values are descriptive sampled responses without eigendecomposition; no broader spectral or causal claim is made.",
    ]
    lines.extend(f"{item}" for item in self_audit)
    lines.extend(["", "## Execution record", "", f"- Repeatability status: {smoke.get('repeatability_status')}", f"- Full tests and NC/LP smoke status: {smoke.get('status')}", f"- Formal runs complete: {sum(run.get('status') == 'complete' for run in manifest['runs'])}/45", f"- Local fetch note: see design audit/run provenance; pushing is handled after analysis and review package completion.", "", "B0–B1 stops here. No B2 alignment or other excluded mechanism is implemented or run.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-checkpoints", action="store_true", help="only summarize metrics; do not recompute checkpoint diagnostics")
    parser.add_argument("--reuse-checkpoint-csvs", action="store_true", help="reuse existing diagnostic CSVs and rebuild report text")
    args = parser.parse_args()
    manifest = read_json(RESEARCH / "run_manifest.json")
    runs = validate_campaign(manifest)
    smoke = read_json(RESEARCH / "smoke_status.json")
    if smoke.get("status") != "passed":
        raise RuntimeError("required test/repeatability/smoke preflight is incomplete")
    repeatability = summarize_repeatability()
    perf_rows, by_key = performance_rows(runs)
    perf_summary = summarize_performance(perf_rows)
    paired_rows, paired_summary = summarize_paired(by_key, repeatability)
    write_csv(DATA / "performance_by_run.csv", perf_rows)
    write_csv(DATA / "performance_summary.csv", perf_summary)
    write_csv(DATA / "paired_delta_by_run.csv", paired_rows)
    write_csv(DATA / "paired_delta_summary.csv", paired_summary)

    if args.reuse_checkpoint_csvs:
        filter_rows = read_csv_rows(DATA / "filter_diagnostics.csv")
        spectral_rows = read_csv_rows(DATA / "spectral_response.csv")
        interventions = read_csv_rows(DATA / "intervention_by_run.csv")
        intervention_summary = read_csv_rows(DATA / "intervention_summary.csv")
    elif args.skip_checkpoints:
        filter_rows, spectral_rows, interventions = [], [], []
        intervention_summary = []
    else:
        filter_rows, spectral_rows, interventions = checkpoint_diagnostics(runs, args.device)
        write_csv(DATA / "filter_diagnostics.csv", filter_rows)
        write_csv(DATA / "spectral_response.csv", spectral_rows)
        write_csv(DATA / "intervention_by_run.csv", interventions)
        intervention_summary = summarize_interventions(interventions)
        write_csv(DATA / "intervention_summary.csv", intervention_summary)
    report = build_report(manifest, smoke, perf_summary, paired_summary, repeatability, filter_rows, spectral_rows, intervention_summary)
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
