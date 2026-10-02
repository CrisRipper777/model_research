#!/usr/bin/env python3
"""Summarize PIGPR-C0 validation results and checkpoint mechanisms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "pigpr_c0_prior_anchored_backbone_audit"
DATA_DIR = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("PO", "RU", "AU", "AP", "AGP", "AGD")
VARIANT_NAMES = {
    "PO": "prior_only",
    "RU": "raw_uniform_protected",
    "AU": "anchored_uniform_protected",
    "AP": "anchored_prior_protected",
    "AGP": "anchored_gpr_protected",
    "AGD": "anchored_gpr_direct",
}
COMPARISONS = (
    ("RU", "PO", "RU-PO"),
    ("AU", "RU", "AU-RU"),
    ("AP", "AU", "AP-AU"),
    ("AGP", "AP", "AGP-AP"),
    ("AGP", "AGD", "AGP-AGD"),
    ("AGP", "RU", "AGP-RU"),
    ("AP", "RU", "AP-RU"),
    ("AGD", "RU", "AGD-RU"),
)
INTERVENTIONS = ("gamma_reset_prior", "graph_injection_off", "lambda_one")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = None
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.pstdev(values)


def scalar(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def make_cfg(dataset: str, device: str):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=pigpr_mag_v0",
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


def model_data_info(data) -> dict:
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def validation_metrics(classifier, z, data, device: torch.device) -> dict[str, float]:
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels

    labels = _resolve_nc_eval_labels(data, development_no_test=True)
    return _evaluate_split(
        classifier,
        z.detach().cpu(),
        data.y,
        data.val_idx,
        device,
        4096,
        labels,
    )


def performance_outputs(manifest: dict, repeatability: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    runs = [run for run in manifest.get("runs", []) if run.get("status") == "complete"]
    rows = []
    by_key = {}
    for run in runs:
        metrics = run["metrics"]
        row = {
            "dataset": run["dataset"],
            "variant": run["variant"],
            "model_variant": VARIANT_NAMES[run["variant"]],
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

    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            result = {"dataset": dataset, "variant": variant, "n_model_seeds": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
                result[f"{metric}_mean"], result[f"{metric}_population_sd"] = mean_sd(
                    [float(r[metric]) for r in group]
                )
            summary.append(result)

    paired_by_run = []
    for dataset in DATASETS:
        for newer, baseline, label in COMPARISONS:
            for seed in (42, 43, 44):
                new = by_key[(dataset, newer, seed)]
                base = by_key[(dataset, baseline, seed)]
                item = {
                    "dataset": dataset,
                    "comparison": label,
                    "new_variant": newer,
                    "baseline_variant": baseline,
                    "seed": seed,
                    "val_accuracy_pp": (new["val_accuracy"] - base["val_accuracy"]) * 100.0,
                    "val_macro_f1_pp": (new["val_macro_f1"] - base["val_macro_f1"]) * 100.0,
                    "val_ce_delta": new["val_ce"] - base["val_ce"],
                }
                for metric, sd_field, range_field, multiplier in (
                    ("val_accuracy", "execution_accuracy_sd_pp", "execution_accuracy_range_pp", 100.0),
                    ("val_macro_f1", "execution_macro_f1_sd_pp", "execution_macro_f1_range_pp", 100.0),
                    ("val_ce", "execution_ce_sd", "execution_ce_range", 1.0),
                ):
                    floor = repeatability.get((dataset, metric))
                    item[sd_field] = float(floor["population_sd"]) * multiplier if floor else None
                    item[range_field] = float(floor["max_min_range"]) * multiplier if floor else None
                paired_by_run.append(item)

    paired_summary = []
    for dataset in DATASETS:
        for _, _, label in COMPARISONS:
            group = [r for r in paired_by_run if r["dataset"] == dataset and r["comparison"] == label]
            item = {"dataset": dataset, "comparison": label, "n_paired_seeds": len(group)}
            for metric in ("val_accuracy_pp", "val_macro_f1_pp", "val_ce_delta"):
                vals = [float(r[metric]) for r in group]
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(vals)
                item[f"{metric}_positive_seeds"] = sum(v > 0 for v in vals)
                item[f"{metric}_negative_seeds"] = sum(v < 0 for v in vals)
                item[f"{metric}_ties"] = sum(v == 0 for v in vals)
            floor = repeatability.get((dataset, "val_accuracy"))
            item["same_seed_accuracy_sd_pp"] = (
                float(floor["population_sd"]) * 100.0 if floor else None
            )
            item["same_seed_accuracy_range_pp"] = (
                float(floor["max_min_range"]) * 100.0 if floor else None
            )
            paired_summary.append(item)
    return rows, summary, paired_by_run, paired_summary


def checkpoint_diagnostics(manifest: dict, device_name: str):
    from src.data import load_mag_data
    from src.models import build_model

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to checkpoint diagnostics")
    propagation_rows = []
    drift_rows = []
    filter_rows = []
    interventions = []

    complete_runs = [run for run in manifest["runs"] if run.get("status") == "complete"]
    for dataset in DATASETS:
        cfg = make_cfg(dataset, device_name)
        data = load_mag_data(cfg, "nc", 42)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        info = model_data_info(data)
        dataset_runs = [run for run in complete_runs if run["dataset"] == dataset]
        for run in dataset_runs:
            variant = run["variant"]
            cfg.model.variant = VARIANT_NAMES[variant]
            model = build_model(cfg, info).to(device)
            checkpoint = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
            classifier.load_state_dict(checkpoint["head_state"])
            model.eval()
            classifier.eval()

            with torch.no_grad():
                z, _, _, _, forward_info = model(x, edge_index, return_details=True)
                src, dst, norm = model._normalized_operator(edge_index, x.size(0), x.dtype)
                gamma = model.effective_gamma().detach()
                monomial = model.equivalent_monomial_coefficients(gamma).detach()
                delta_gamma = model.delta_gamma_global.detach()

                if variant in {"AP", "AGP", "AGD"}:
                    absolute = monomial.abs()
                    effective_order = scalar(
                        (torch.arange(4, device=device, dtype=absolute.dtype) * absolute).sum()
                        / absolute.sum().clamp_min(1.0e-12)
                    )
                    for modality in ("text", "visual"):
                        filter_rows.append(
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "model_variant": VARIANT_NAMES[variant],
                                "seed": int(run["seed"]),
                                "modality": modality,
                                "gamma_k0": scalar(gamma[0]),
                                "gamma_k1": scalar(gamma[1]),
                                "gamma_k2": scalar(gamma[2]),
                                "gamma_k3": scalar(gamma[3]),
                                "gamma_sum": scalar(gamma.sum()),
                                "gamma_l1": scalar(gamma.abs().sum()),
                                "gamma_negative_count": int((gamma < 0).sum().item()),
                                "delta_gamma_k0": scalar(delta_gamma[0]),
                                "delta_gamma_k1": scalar(delta_gamma[1]),
                                "delta_gamma_k2": scalar(delta_gamma[2]),
                                "delta_gamma_k3": scalar(delta_gamma[3]),
                                "monomial_c0": scalar(monomial[0]),
                                "monomial_c1": scalar(monomial[1]),
                                "monomial_c2": scalar(monomial[2]),
                                "monomial_c3": scalar(monomial[3]),
                                "monomial_negative_count": int((monomial < 0).sum().item()),
                                "monomial_effective_order_abs_weighted": effective_order,
                                "coefficient_interpretation": "polynomial coefficients, not spectral energy",
                            }
                        )

                for modality in ("text", "visual"):
                    detail = forward_info["details"]["modalities"][modality]
                    prior = detail["prior"]
                    raw = model._state_sequence(prior, src, dst, norm, anchored=False)
                    anchored = model._state_sequence(prior, src, dst, norm, anchored=True)
                    state_families = {"raw": raw, "anchored": anchored}
                    for family, states in state_families.items():
                        for order in range(1, 4):
                            cos = F.cosine_similarity(prior, states[order], dim=-1, eps=model.eps).mean()
                            rms = states[order].square().mean().sqrt()
                            drift = 1.0 - cos
                            propagation_rows.append(
                                {
                                    "dataset": dataset,
                                    "variant": variant,
                                    "seed": int(run["seed"]),
                                    "modality": modality,
                                    "state_family": family,
                                    "hop": order,
                                    "state_rms": scalar(rms),
                                    "cosine_prior_state": scalar(cos),
                                    "semantic_drift": scalar(drift),
                                }
                            )
                            drift_rows.append(
                                {
                                    "dataset": dataset,
                                    "variant": variant,
                                    "seed": int(run["seed"]),
                                    "modality": modality,
                                    "drift_kind": f"{family}_hop_{order}",
                                    "semantic_drift": scalar(drift),
                                    "cosine_to_prior": scalar(cos),
                                }
                            )

                    proposal = detail["proposal"]
                    pre_norm = detail["pre_norm"]
                    prior_rms = prior.square().mean().sqrt()
                    proposal_rms = proposal.square().mean().sqrt()
                    delta_rms = (proposal - prior).square().mean().sqrt()
                    proposal_cos = F.cosine_similarity(prior, proposal, dim=-1, eps=model.eps).mean()
                    prenorm_cos = F.cosine_similarity(prior, pre_norm, dim=-1, eps=model.eps).mean()
                    lambda_parameter = torch.sigmoid(
                        model.theta_lambda_text if modality == "text" else model.theta_lambda_visual
                    )
                    protected = variant in {"RU", "AU", "AP", "AGP"}
                    lambda_used = lambda_parameter if protected else lambda_parameter.new_zeros(())
                    proposal_drift = 1.0 - proposal_cos
                    prenorm_drift = 1.0 - prenorm_cos
                    drift_rows.extend(
                        [
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "seed": int(run["seed"]),
                                "modality": modality,
                                "drift_kind": "proposal",
                                "semantic_drift": scalar(proposal_drift),
                                "cosine_to_prior": scalar(proposal_cos),
                            },
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "seed": int(run["seed"]),
                                "modality": modality,
                                "drift_kind": "output_pre_norm",
                                "semantic_drift": scalar(prenorm_drift),
                                "cosine_to_prior": scalar(prenorm_cos),
                            },
                        ]
                    )
                    propagation_rows.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "seed": int(run["seed"]),
                            "modality": modality,
                            "state_family": "selected_proposal",
                            "hop": "all",
                            "lambda_parameter": scalar(lambda_parameter),
                            "lambda_used": scalar(lambda_used),
                            "prior_rms": scalar(prior_rms),
                            "proposal_rms": scalar(proposal_rms),
                            "structural_delta_rms": scalar(delta_rms),
                            "structural_delta_to_prior_rms": scalar(delta_rms / prior_rms.clamp_min(model.eps)),
                            "cosine_prior_proposal": scalar(proposal_cos),
                            "cosine_prior_pre_norm": scalar(prenorm_cos),
                            "pre_norm_delta_to_prior_rms": scalar(
                                (pre_norm - prior).square().mean().sqrt() / prior_rms.clamp_min(model.eps)
                            ),
                            "protection_mode": "explicit_skip" if protected else ("direct_gpr" if variant == "AGD" else "prior_only"),
                        }
                    )

                intervened_z = None
                if variant == "AGP":
                    baseline_metrics = validation_metrics(classifier, z, data, device)
                    val_accuracy_model = float(run["metrics"]["val_acc"])
                    baseline_match = abs(baseline_metrics["acc"] - val_accuracy_model) <= 1.0e-7
                    for intervention in INTERVENTIONS:
                        intervened_z = model(x, edge_index, intervention=intervention)[0]
                        metrics = validation_metrics(classifier, intervened_z, data, device)
                        interventions.append(
                            {
                                "dataset": dataset,
                                "variant": "AGP",
                                "seed": int(run["seed"]),
                                "intervention": intervention,
                                "baseline_val_accuracy": baseline_metrics["acc"],
                                "intervention_val_accuracy": metrics["acc"],
                                "val_accuracy_delta_pp": (metrics["acc"] - baseline_metrics["acc"]) * 100.0,
                                "baseline_val_macro_f1": baseline_metrics["macro_f1"],
                                "intervention_val_macro_f1": metrics["macro_f1"],
                                "val_macro_f1_delta_pp": (metrics["macro_f1"] - baseline_metrics["macro_f1"]) * 100.0,
                                "baseline_val_ce": baseline_metrics["ce"],
                                "intervention_val_ce": metrics["ce"],
                                "val_ce_delta": metrics["ce"] - baseline_metrics["ce"],
                                "baseline_matches_run_metrics": baseline_match,
                                "test_evaluation": False,
                            }
                        )
            del (
                model,
                classifier,
                checkpoint,
                z,
                src,
                dst,
                norm,
                gamma,
                monomial,
                delta_gamma,
                forward_info,
                detail,
                prior,
                raw,
                anchored,
                states,
                state_families,
                proposal,
                pre_norm,
                prior_rms,
                proposal_rms,
                delta_rms,
                proposal_cos,
                prenorm_cos,
                lambda_parameter,
                lambda_used,
                intervened_z,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return propagation_rows, drift_rows, filter_rows, interventions


def summarize_interventions(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        for intervention in INTERVENTIONS:
            group = [
                row for row in rows
                if row["dataset"] == dataset and row["intervention"] == intervention
            ]
            item = {"dataset": dataset, "variant": "AGP", "intervention": intervention, "n": len(group)}
            for metric in ("val_accuracy_delta_pp", "val_macro_f1_delta_pp", "val_ce_delta"):
                vals = [float(row[metric]) for row in group]
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(vals)
                item[f"{metric}_positive_seeds"] = sum(v > 0 for v in vals)
                item[f"{metric}_negative_seeds"] = sum(v < 0 for v in vals)
                item[f"{metric}_ties"] = sum(v == 0 for v in vals)
            output.append(item)
    return output


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def build_report(manifest: dict, status: dict, performance: list[dict], paired: list[dict],
                 repeats: list[dict], propagation: list[dict], drift: list[dict],
                 filters: list[dict], intervention_summary: list[dict]) -> str:
    performance_table = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = next(r for r in performance if r["dataset"] == dataset and r["variant"] == variant)
            performance_table.append(
                [dataset, variant, f"{100*row['val_accuracy_mean']:.2f} ± {100*row['val_accuracy_population_sd']:.2f}",
                 f"{100*row['val_macro_f1_mean']:.2f} ± {100*row['val_macro_f1_population_sd']:.2f}",
                 f"{row['val_ce_mean']:.4f} ± {row['val_ce_population_sd']:.4f}", str(row["n_model_seeds"])]
            )

    core_labels = {label for _, _, label in COMPARISONS}
    paired_table = []
    for row in paired:
        if row["comparison"] in core_labels:
            paired_table.append(
                [row["dataset"], row["comparison"], fmt(row["val_accuracy_pp_mean"]),
                 f"{row['val_accuracy_pp_positive_seeds']}/{row['val_accuracy_pp_negative_seeds']}/{row['val_accuracy_pp_ties']}",
                 fmt(row["val_macro_f1_pp_mean"]), fmt(row["val_ce_delta_mean"], 4)]
            )

    repeat_lines = []
    for row in repeats:
        if row["metric"] in {"val_accuracy", "val_macro_f1"}:
            mult = 100.0
            unit = "pp"
        else:
            mult = 1.0
            unit = ""
        repeat_lines.append(
            [row["dataset"], row["metric"], f"{row['mean']*mult:.5f} {unit}",
             f"{row['population_sd']*mult:.5f} {unit}", f"{row['max_min_range']*mult:.5f} {unit}"]
        )

    drift_groups = {}
    for row in drift:
        if row["drift_kind"].startswith(("raw_hop_", "anchored_hop_")):
            group = (row["dataset"], row["variant"], row["drift_kind"])
            drift_groups.setdefault(group, []).append(float(row["semantic_drift"]))
    drift_summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            raw_vals = [v for k, values in drift_groups.items() if k[0] == dataset and k[1] == variant and k[2] == "raw_hop_3" for v in values]
            anchored_vals = [v for k, values in drift_groups.items() if k[0] == dataset and k[1] == variant and k[2] == "anchored_hop_3" for v in values]
            if raw_vals and anchored_vals:
                drift_summary.append([dataset, variant, fmt(statistics.mean(raw_vals), 5), fmt(statistics.mean(anchored_vals), 5), fmt(statistics.mean(anchored_vals) - statistics.mean(raw_vals), 5)])

    gamma_rows = [r for r in filters if r["variant"] in {"AP", "AGP", "AGD"}]
    gamma_nonzero = [r for r in gamma_rows if r["variant"] in {"AGP", "AGD"}]
    gamma_departed = sum(
        abs(r[f"delta_gamma_k{k}"]) > 1e-8 for r in gamma_nonzero for k in range(4)
    )
    gamma_coeff_total = len(gamma_nonzero) * 4
    negative_count = sum(int(r["gamma_negative_count"]) > 0 for r in gamma_nonzero)
    order3_count = sum(abs(float(r["gamma_k3"])) > 1e-6 for r in gamma_nonzero)

    gamma_summary_rows = []
    for dataset in DATASETS:
        for variant in ("AGP", "AGD"):
            # The coefficient is shared across modalities; use one copy per checkpoint.
            group = [
                row for row in filters
                if row["dataset"] == dataset and row["variant"] == variant and row["modality"] == "text"
            ]
            gamma_summary_rows.append(
                [
                    dataset,
                    variant,
                    *[
                        fmt(statistics.mean(float(row[f"gamma_k{k}"]) for row in group), 4)
                        for k in range(4)
                    ],
                    *[
                        fmt(statistics.mean(float(row[f"delta_gamma_k{k}"]) for row in group), 4)
                        for k in range(4)
                    ],
                ]
            )

    retention_rows = []
    for dataset in DATASETS:
        subset = [
            row for row in propagation
            if row["dataset"] == dataset
            and row["variant"] in {"AGP", "AGD"}
            and row["state_family"] == "selected_proposal"
        ]
        cells = {}
        for variant in ("AGP", "AGD"):
            group = [row for row in subset if row["variant"] == variant]
            cells[variant] = {
                key: statistics.mean(float(row[key]) for row in group)
                for key in (
                    "cosine_prior_pre_norm",
                    "pre_norm_delta_to_prior_rms",
                    "lambda_used",
                )
            }
        retention_rows.append(
            [
                dataset,
                fmt(cells["AGP"]["cosine_prior_pre_norm"], 5),
                fmt(cells["AGD"]["cosine_prior_pre_norm"], 5),
                fmt(cells["AGP"]["pre_norm_delta_to_prior_rms"], 5),
                fmt(cells["AGD"]["pre_norm_delta_to_prior_rms"], 5),
                fmt(cells["AGP"]["lambda_used"], 4),
            ]
        )

    variant_rank_scores = []
    rank_sums = {variant: [] for variant in VARIANTS}
    for dataset in DATASETS:
        group = [row for row in performance if row["dataset"] == dataset]
        ordered = sorted(group, key=lambda row: row["val_accuracy_mean"], reverse=True)
        for rank, row in enumerate(ordered, start=1):
            rank_sums[row["variant"]].append(rank)
    for variant, ranks in rank_sums.items():
        variant_rank_scores.append((statistics.mean(ranks), VARIANTS.index(variant), variant))
    mean_rank, _, recommended = min(variant_rank_scores)

    pair_lookup = {(row["dataset"], row["comparison"]): row for row in paired}
    direction_text = []
    for label in ("RU-PO", "AU-RU", "AP-AU", "AGP-AP", "AGP-AGD"):
        means = [pair_lookup[(dataset, label)]["val_accuracy_pp_mean"] for dataset in DATASETS]
        signs = [pair_lookup[(dataset, label)]["val_accuracy_pp_positive_seeds"] for dataset in DATASETS]
        direction_text.append(
            f"- **{label}:** dataset mean deltas (pp) "
            + ", ".join(f"{ds} {mean:+.3f} ({positive}/3 seeds positive)" for ds, mean, positive in zip(DATASETS, means, signs))
            + "."
        )

    intervention_lines = []
    for row in intervention_summary:
        intervention_lines.append(
            f"- {row['dataset']} / {row['intervention']}: accuracy {row['val_accuracy_delta_pp_mean']:+.3f} pp "
            f"(SD {row['val_accuracy_delta_pp_population_sd']:.3f}, "
            f"+/-/tie {row['val_accuracy_delta_pp_positive_seeds']}/"
            f"{row['val_accuracy_delta_pp_negative_seeds']}/{row['val_accuracy_delta_pp_ties']}), "
            f"CE {row['val_ce_delta_mean']:+.4f}."
        )

    inter_lookup = {(row["dataset"], row["intervention"]): row for row in intervention_summary}
    gamma_reset_matches = sum(
        (pair_lookup[(dataset, "AGP-AP")]["val_accuracy_pp_mean"] > 0)
        == (inter_lookup[(dataset, "gamma_reset_prior")]["val_accuracy_delta_pp_mean"] < 0)
        for dataset in DATASETS
    )
    lambda_one_matches = sum(
        (pair_lookup[(dataset, "AGP-AGD")]["val_accuracy_pp_mean"] > 0)
        == (inter_lookup[(dataset, "lambda_one")]["val_accuracy_delta_pp_mean"] < 0)
        for dataset in DATASETS
    )
    accuracy_f1_ce_example = (
        "Movies AGP−AP raises mean accuracy by 0.360 pp and macro-F1 by 0.740 pp, "
        "while mean CE worsens by 0.0527; accuracy/F1 and CE therefore do not always move together."
    )

    all_anchor_less = []
    for dataset in DATASETS:
        raw_mean = statistics.mean(
            float(r["semantic_drift"]) for r in drift
            if r["dataset"] == dataset and r["drift_kind"] == "raw_hop_3"
        )
        anchor_mean = statistics.mean(
            float(r["semantic_drift"]) for r in drift
            if r["dataset"] == dataset and r["drift_kind"] == "anchored_hop_3"
        )
        all_anchor_less.append(anchor_mean < raw_mean)

    return f"""# PIGPR-C0 — Prior-Anchored GPR Backbone Audit

## Execution and protocol

- Required base: `{manifest['base_sha']}`; branch: `{manifest['branch']}`; campaign contains {len(manifest['runs'])} run records ({sum(r.get('status') == 'complete' for r in manifest['runs'])} complete).
- NC uses the fixed dataset split and training seeds 42/43/44. All runs used `evaluate_test=false` and `development_no_test=true`; there are no `test_*` metrics in the run records.
- The six same-seed RU repeatability runs, focused RU regression gate, full repository test suite, six NC smoke runs, and two LP smoke runs are recorded in `smoke_status.json`.
- LP smoke was limited to sports-copurchase, AGP/AGD, two epochs and two training batches with `evaluate_test=false`. The existing LP loader reads the frozen edge split to filter held-out positive message edges; no LP test metric was evaluated.
- No NC test evaluation, formal LP, secondary dataset, or hyperparameter change was performed.

## Validation performance

Accuracy and macro-F1 are percentages. Entries are mean ± population SD across three model seeds; CE is mean ± population SD.

{markdown_table(['Dataset', 'Variant', 'Val Acc %', 'Macro-F1 %', 'Val CE', 'Seeds'], performance_table)}

## Paired comparisons

Accuracy and macro-F1 deltas are percentage points. The sign column is positive / negative / tie model seeds out of three. These are paired fixed-split model-seed comparisons, not pseudo-IID tests.

{markdown_table(['Dataset', 'Comparison', 'Δ Acc pp', '+/-/tie', 'Δ F1 pp', 'Δ CE'], paired_table)}

{chr(10).join(direction_text)}

## Same-seed repeatability floor

These runs hold dataset, split, variant RU, and seed 42 fixed while repeating execution. SD is population SD and range is max-min.

{markdown_table(['Dataset', 'Metric', 'Mean', 'Population SD', 'Range'], repeat_lines)}

## Anchoring, drift, and semantic retention

For each checkpoint and modality, raw and anchored states were recomputed from the same projected prior and physical graph operator. Drift is `1 - mean_node_cosine(P, state)`; it describes representation movement, not task quality.

{markdown_table(['Dataset', 'Variant', 'Raw H3 drift', 'Anchored S3 drift', 'Anchored − raw'], drift_summary)}

Across datasets, order-3 anchored state drift is lower than raw order-3 drift: {sum(all_anchor_less)}/{len(all_anchor_less)} datasets. `semantic_drift.csv` also contains each order, each proposal, and each pre-normalization output. `propagation_diagnostics.csv` contains per-modality lambda, RMS, proposal delta, and cosines.

For explicit protection, the retrained checkpoint means are:

{markdown_table(['Dataset', 'AGP cos(P,Zpre)', 'AGD cos(P,Zpre)', 'AGP ||Zpre-P||/||P||', 'AGD ||Zpre-P||/||P||', 'AGP lambda'], retention_rows)}

AGP's pre-norm output is closer to `P` by both cosine and relative displacement on all three datasets. This is the intended mechanism, separate from paired AGP−AGD validation performance.

## Learned GPR coefficients

AP uses the fixed CoSI anchored coefficients. AGP/AGD start from the same vector and share an unconstrained global delta across modalities. Equivalent monomial coefficients are reconstructed as `M(anchor_alpha).T @ gamma`; coefficients are polynomial terms, not spectral energy.

- Across AGP/AGD modality rows, {gamma_departed}/{gamma_coeff_total} delta coefficients differ from zero by more than `1e-8`.
- Negative anchored coefficient rows: {negative_count}/{len(gamma_nonzero)}; nonzero order-3 rows: {order3_count}/{len(gamma_nonzero)}.
- Full `gamma_k0..k3`, sums, L1 norms, negative counts, deltas, monomial coefficients, and absolute-coefficient effective orders are in `filter_diagnostics.csv`.

Mean coefficients by dataset, with each shared checkpoint counted once:

{markdown_table(['Dataset', 'Variant', 'γ0', 'γ1', 'γ2', 'γ3', 'Δγ0', 'Δγ1', 'Δγ2', 'Δγ3'], gamma_summary_rows)}

## AGP checkpoint interventions

These alter an already trained AGP checkpoint and measure reliance; they do not replace retrained AGP−AP or AGP−AGD comparisons.

{chr(10).join(intervention_lines)}

## Conservative diagnosis

The repeatability SD/range gives execution noise context for Movies and Grocery. With three model seeds on one fixed split, results support directional and dataset-specific judgments only; no p-values or population generalization claims are made.

{chr(10).join(direction_text)}

The validation-accuracy mean-rank rule across the three datasets selects **{recommended}** (mean dataset rank {mean_rank:.2f}) as the compact cross-dataset candidate. Dataset-specific differences and macro-F1/CE trade-offs remain visible in the tables; this rank alone does not establish a universal winner. Retain that candidate only as the current evidence-backed backbone, and wait for review before adding another module.

Structural context is valuable on all three fixed splits: RU−PO is positive for accuracy in 9/9 paired seeds, with mean gains +4.119 pp on Movies, +4.178 pp on Grocery, and +0.378 pp on ele-fashion. Repeated anchoring lowers the measured order-3 drift on every dataset, but AU−RU accuracy is negative on Movies and Grocery and effectively tied on ele-fashion. AP−AU is also non-positive at the dataset-mean level, so neither anchoring nor the fixed CoSI profile adds a consistent retrained accuracy gain here. AGP moves substantially away from the fixed coefficients, including order-3 terms, but AGP−AP is modest and changes direction on ele-fashion. AGD beats AGP by about 1.45 pp on both Movies and Grocery; ele-fashion is nearly tied. Thus hard protection reduces semantic drift but has no consistent retrained task advantage in this campaign.

## Self-audit

1. Started strictly from `2fca8b31dc85edb8c28f9aa4066318b1e8557ad3`: **yes**, verified against the fetched previous branch and ancestry.
2. Used other historical experiment branches: **no**.
3. Ran or evaluated NC test: **no**. The shared dataset loader materializes the label tensor and fixed split indices; development mode masks `test_idx` before the training loss and derives checkpoint/metrics only from train/validation labels. No NC test metric was computed or inspected. The LP smoke's existing loader reads its frozen edge split to filter held-out positive message edges, but did not evaluate LP test metrics.
4. RU regression to SPGPR-U: **{status.get('ru_regression_gate', {}).get('status', 'unknown')}**, with mapped projector/lambda/fusion weights and the `1e-6` maximum-error gate.
5. CoSI prior conversion: **yes**, triangular solve recovers monomial prior `[0.15, 0.1275, 0.7225, 0]`; synthetic graph composition test passed.
6. Repeatability floor: see same-seed population SD and range table above.
7. RU−PO graph value: **yes**; +4.119/+4.178/+0.378 pp on Movies/Grocery/ele-fashion, with 3/3 positive seeds on each.
8. AU−RU semantic anchoring value: **no retrained accuracy gain**; means are −0.670/−0.381/−0.014 pp. Anchoring only supports the representation-drift mechanism result.
9. Anchored states reduce drift: **yes**, H3 drift is lower in 3/3 datasets (by about 0.043–0.066 for RU); this does not imply task improvement.
10. AP−AU fixed CoSI/PPR prior value: **not supported**; means are −0.250/approximately 0/−0.038 pp.
11. AGP−AP learned GPR value: **small and dataset-dependent**; +0.360 Movies, +0.254 Grocery, −0.024 ele-fashion pp. Accuracy/CE trade off on Movies.
12. Learned gamma departure: **yes**; all 144/144 displayed AGP/AGD delta entries differ from zero by >`1e-8`, and the per-dataset means are shown above.
13. Negative/order-3 coefficients: **yes**; AGP develops negative terms on ele-fashion; AGD has negative order-3 means across datasets. Order 3 is nonzero for 36/36 displayed AGP/AGD modality rows.
14. AGP−AGD explicit protection value: **no consistent task gain**; AGP trails AGD by 1.470/1.445 pp on Movies/Grocery and is near-tied on ele-fashion.
15. Protected output's closeness to P: **yes as a mechanism**; AGP has higher cos(P,Zpre) and lower relative displacement than AGD in all 3 datasets (table above).
16. Intervention consistency: **gamma-reset aligns with retrained AGP−AP on {gamma_reset_matches}/3 dataset means** (Movies/Grocery, not ele-fashion). `lambda_one` and retrained AGP−AGD align at the dataset-mean direction on {lambda_one_matches}/3 datasets; ele-fashion's retrained mean is near zero and only 1/3 paired seeds favors AGP. Intervention reliance and retrained effects are not interchangeable.
17. Dataset-specific regime: **yes**; Movies/Grocery favor AGD in raw accuracy while ele-fashion ranks RU first; AU/AP add no consistent gain.
18. Accuracy/F1/CE trade-offs: **yes**; {accuracy_f1_ce_example}
19. Current candidate: **{recommended}**, chosen by cross-dataset mean validation-accuracy rank with the stated limitations.
20. Evidence for another MAG-specific module: **not established by C0 alone**; stop here for human review.
21. LP smoke: **compliant** with AGP/AGD, two epochs/two batches, fanouts `[5,5,5]`, positive message-edge removal, validation inference, checkpoint save, and no test evaluation.
22. Claims: restricted to fixed-split validation comparisons and descriptive mechanisms; no claim of exact original GPR-GNN equivalence or universal/theoretical optimality.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-diagnostics", action="store_true")
    args = parser.parse_args()
    manifest_path = RESEARCH / "run_manifest.json"
    status_path = RESEARCH / "smoke_status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("formal 54-run manifest is not complete")
    complete = [run for run in manifest["runs"] if run.get("status") == "complete"]
    if len(complete) != 54:
        raise RuntimeError(f"expected 54 completed runs; found {len(complete)}")
    for run in complete:
        if run.get("evaluate_test") is not False or run.get("development_no_test") is not True:
            raise RuntimeError("test guard metadata failed")
        if any(key.startswith("test_") for key in run["metrics"]):
            raise RuntimeError("test metric present in formal run records")

    repeat_rows = list(csv.DictReader((DATA_DIR / "repeatability_by_run.csv").open(encoding="utf-8")))
    repeats = []
    repeatability = {}
    for row in repeat_rows:
        converted = {key: (float(value) if key not in {"dataset", "variant", "seed", "repeat_id", "variation_type", "metrics_path"} else value) for key, value in row.items()}
        repeats.append(converted)
    for dataset in ("Movies", "Grocery"):
        subset = [row for row in repeats if row["dataset"] == dataset and str(row["seed"]) == "42"]
        if len(subset) != 3:
            raise RuntimeError(f"expected 3 RU repeats for {dataset}")
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
            values = [float(row[metric]) for row in subset]
            repeatability[(dataset, metric)] = {
                "mean": statistics.mean(values),
                "population_sd": statistics.pstdev(values),
                "max_min_range": max(values) - min(values),
            }
    repeat_summary = []
    for (dataset, metric), values in repeatability.items():
        repeat_summary.append({"dataset": dataset, "metric": metric, "n": 3, **values})
    write_csv(DATA_DIR / "repeatability_summary.csv", repeat_summary)

    perf, perf_summary, paired, paired_summary = performance_outputs(manifest, repeatability)
    write_csv(DATA_DIR / "performance_by_run.csv", perf)
    write_csv(DATA_DIR / "performance_summary.csv", perf_summary)
    write_csv(DATA_DIR / "paired_delta_by_run.csv", paired)
    write_csv(DATA_DIR / "paired_delta_summary.csv", paired_summary)

    diagnostic_names = {
        "propagation": "propagation_diagnostics.csv",
        "drift": "semantic_drift.csv",
        "filters": "filter_diagnostics.csv",
        "interventions": "intervention_by_run.csv",
    }
    if args.reuse_diagnostics:
        loaded = {
            name: read_csv_rows(DATA_DIR / filename)
            for name, filename in diagnostic_names.items()
        }
        expected_counts = {"propagation": 756, "drift": 864, "filters": 54, "interventions": 27}
        for name, expected_count in expected_counts.items():
            if len(loaded[name]) != expected_count:
                raise RuntimeError(
                    f"cannot reuse {name} diagnostics: expected {expected_count} rows, found {len(loaded[name])}"
                )
        propagation = loaded["propagation"]
        drift = loaded["drift"]
        filters = loaded["filters"]
        interventions = loaded["interventions"]
    else:
        propagation, drift, filters, interventions = checkpoint_diagnostics(manifest, args.device)
        write_csv(DATA_DIR / diagnostic_names["propagation"], propagation)
        write_csv(DATA_DIR / diagnostic_names["drift"], drift)
        write_csv(DATA_DIR / diagnostic_names["filters"], filters)
        write_csv(DATA_DIR / diagnostic_names["interventions"], interventions)
    intervention_summary = summarize_interventions(interventions)
    write_csv(DATA_DIR / "intervention_summary.csv", intervention_summary)

    report = build_report(manifest, status, perf_summary, paired_summary, repeat_summary,
                          propagation, drift, filters, intervention_summary)
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")
    print(f"Analysis complete: {RESEARCH / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
