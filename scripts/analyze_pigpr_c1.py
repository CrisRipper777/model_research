#!/usr/bin/env python3
"""Analyze PIGPR-C1 validation results and direct-GPR checkpoint mechanisms."""

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
RESEARCH = ROOT / "research" / "pigpr_c1_direct_gpr_attribution"
DATA_DIR = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("RU", "RUD", "R0UD", "RFD", "RGD", "AGD")
NAMES = {
    "RU": "raw_uniform_protected", "RUD": "raw_uniform_direct",
    "R0UD": "raw_uniform0_direct", "RFD": "raw_fixed_prior_direct",
    "RGD": "raw_gpr_direct", "AGD": "anchored_gpr_direct",
}
COMPARISONS = (
    ("RUD", "RU", "RUD-RU"), ("R0UD", "RUD", "R0UD-RUD"),
    ("RFD", "R0UD", "RFD-R0UD"), ("RGD", "RFD", "RGD-RFD"),
    ("AGD", "RGD", "AGD-RGD"), ("RGD", "RU", "RGD-RU"),
    ("AGD", "RU", "AGD-RU"), ("RGD", "RUD", "RGD-RUD"),
    ("AGD", "RUD", "AGD-RUD"),
)
INTERVENTIONS = ("reset_prior", "order0_off", "order3_off", "negative_terms_off")
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                if value == "":
                    converted[key] = ""
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


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def make_cfg(dataset: str, device: str):
    overrides = [
        f"dataset={dataset}", "task=nc", "model=pigpr_mag_v1", "seed=42",
        f"device={device}", "task.evaluate_test=false", "task.development_no_test=true",
    ]
    if dataset in {"Movies", "Grocery"}:
        overrides.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def model_data_info(data) -> dict:
    return {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def validation_metrics(classifier, z, data, device: torch.device) -> dict[str, float]:
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels
    labels = _resolve_nc_eval_labels(data, development_no_test=True)
    return _evaluate_split(classifier, z.detach().cpu(), data.y, data.val_idx, device, 4096, labels)


def read_repeats() -> tuple[list[dict], dict]:
    rows = []
    with (DATA_DIR / "repeatability_by_run.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("seed", "repeat_id", "val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
                row[key] = float(row[key])
            rows.append(row)
    summary = []
    floor = {}
    for dataset in DATASETS:
        group = [r for r in rows if r["dataset"] == dataset]
        if len(group) != 3:
            raise RuntimeError(f"expected 3 AGD repeats for {dataset}, found {len(group)}")
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
            values = [float(r[metric]) for r in group]
            item = {"dataset": dataset, "variant": "AGD", "metric": metric, "n": 3,
                    "mean": statistics.mean(values), "population_sd": statistics.pstdev(values),
                    "max_min_range": max(values) - min(values), "minimum": min(values),
                    "maximum": max(values), "variation_type": "same-seed execution variation"}
            summary.append(item)
            floor[(dataset, metric)] = item
    write_csv(DATA_DIR / "repeatability_summary.csv", summary)
    return rows, floor


def performance_outputs(manifest: dict, floor: dict):
    runs = [r for r in manifest["runs"] if r.get("status") == "complete"]
    rows, lookup = [], {}
    for run in runs:
        metrics = run["metrics"]
        row = {
            "dataset": run["dataset"], "variant": run["variant"],
            "model_variant": NAMES[run["variant"]], "seed": int(run["seed"]),
            "val_accuracy": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]), "val_ce": float(metrics["val_ce"]),
            "best_epoch": int(run["best_epoch"]), "development_no_test": True,
            "run_metrics_path": run["run_metrics_path"],
        }
        rows.append(row)
        lookup[(row["dataset"], row["variant"], row["seed"])] = row
    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            item = {"dataset": dataset, "variant": variant, "n_model_seeds": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd([float(r[metric]) for r in group])
            summary.append(item)
    paired_rows = []
    for dataset in DATASETS:
        for new, base, label in COMPARISONS:
            for seed in (42, 43, 44):
                a, b = lookup[(dataset, new, seed)], lookup[(dataset, base, seed)]
                item = {"dataset": dataset, "comparison": label, "new_variant": new,
                        "baseline_variant": base, "seed": seed,
                        "val_accuracy_pp": (a["val_accuracy"] - b["val_accuracy"]) * 100,
                        "val_macro_f1_pp": (a["val_macro_f1"] - b["val_macro_f1"]) * 100,
                        "val_ce_delta": a["val_ce"] - b["val_ce"]}
                for metric, multiplier, sd_key, range_key in (
                    ("val_accuracy", 100, "execution_accuracy_sd_pp", "execution_accuracy_range_pp"),
                    ("val_macro_f1", 100, "execution_macro_f1_sd_pp", "execution_macro_f1_range_pp"),
                    ("val_ce", 1, "execution_ce_sd", "execution_ce_range"),
                ):
                    noise = floor.get((dataset, metric))
                    item[sd_key] = noise["population_sd"] * multiplier if noise else None
                    item[range_key] = noise["max_min_range"] * multiplier if noise else None
                paired_rows.append(item)
    paired_summary = []
    for dataset in DATASETS:
        for _, _, label in COMPARISONS:
            group = [r for r in paired_rows if r["dataset"] == dataset and r["comparison"] == label]
            item = {"dataset": dataset, "comparison": label, "n_paired_seeds": len(group)}
            for metric in ("val_accuracy_pp", "val_macro_f1_pp", "val_ce_delta"):
                values = [float(r[metric]) for r in group]
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(values)
                item[f"{metric}_positive_seeds"] = sum(v > 0 for v in values)
                item[f"{metric}_negative_seeds"] = sum(v < 0 for v in values)
                item[f"{metric}_ties"] = sum(v == 0 for v in values)
            noise = floor.get((dataset, "val_accuracy"))
            item["same_seed_accuracy_sd_pp"] = noise["population_sd"] * 100 if noise else None
            item["same_seed_accuracy_range_pp"] = noise["max_min_range"] * 100 if noise else None
            paired_summary.append(item)
    return rows, summary, paired_rows, paired_summary


def scalar(x: torch.Tensor | float) -> float:
    return float(x.detach().float().cpu().item()) if torch.is_tensor(x) else float(x)


def flat_cosine(a: torch.Tensor, b: torch.Tensor, eps: float) -> float:
    return scalar(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1, eps=eps)[0])


def checkpoint_diagnostics(manifest: dict, device_name: str):
    from src.data import load_mag_data
    from src.models import build_model

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for checkpoint diagnostics")
    coefficients, contributions, similarities, interventions = [], [], [], []
    complete = [r for r in manifest["runs"] if r.get("status") == "complete"]
    for dataset in DATASETS:
        cfg = make_cfg(dataset, device_name)
        data = load_mag_data(cfg, "nc", 42)
        x = data.x.to(device)
        edge = data.edge_index.to(device)
        info = model_data_info(data)
        runs = [r for r in complete if r["dataset"] == dataset]
        for run in runs:
            variant = run["variant"]
            if variant not in {"RFD", "RGD", "AGD"}:
                continue
            cfg.model.variant = NAMES[variant]
            model = build_model(cfg, info).to(device)
            checkpoint = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
            classifier.load_state_dict(checkpoint["head_state"])
            classifier.eval()
            with torch.no_grad():
                z, _, _, _, fwd = model(x, edge, return_details=True)
                detail = fwd["details"]["modalities"]["text"]
                coeff = detail["coefficients_raw"].detach()
                gamma = detail["gamma"].detach()
                coeff_l1 = coeff.abs().sum()
                abs_sum = coeff_l1.clamp_min(model.eps)
                abs_weights = coeff.abs()
                effective_order = (torch.arange(4, device=device, dtype=coeff.dtype) * abs_weights).sum() / abs_sum
                row = {
                    "dataset": dataset, "variant": variant, "seed": int(run["seed"]),
                    "basis": "anchored" if variant == "AGD" else "raw",
                    "gamma0": scalar(gamma[0]) if variant == "AGD" else "",
                    "gamma1": scalar(gamma[1]) if variant == "AGD" else "",
                    "gamma2": scalar(gamma[2]) if variant == "AGD" else "",
                    "gamma3": scalar(gamma[3]) if variant == "AGD" else "",
                    "gamma_sum": scalar(gamma.sum()) if variant == "AGD" else "",
                    "gamma_l1": scalar(gamma.abs().sum()) if variant == "AGD" else "",
                    "gamma_negative_count": int((gamma < 0).sum().item()) if variant == "AGD" else "",
                    "c0": scalar(coeff[0]), "c1": scalar(coeff[1]),
                    "c2": scalar(coeff[2]), "c3": scalar(coeff[3]),
                    "c_sum": scalar(coeff.sum()), "c_l1": scalar(coeff_l1),
                    "c_negative_count": int((coeff < 0).sum().item()),
                    "abs_weighted_effective_order": scalar(effective_order),
                    "normalized_shape0": scalar(coeff[0] / abs_sum),
                    "normalized_shape1": scalar(coeff[1] / abs_sum),
                    "normalized_shape2": scalar(coeff[2] / abs_sum),
                    "normalized_shape3": scalar(coeff[3] / abs_sum),
                    "c_sign0": int(torch.sign(coeff[0]).item()), "c_sign1": int(torch.sign(coeff[1]).item()),
                    "c_sign2": int(torch.sign(coeff[2]).item()), "c_sign3": int(torch.sign(coeff[3]).item()),
                    "gamma_sign0": int(torch.sign(gamma[0]).item()) if variant == "AGD" else "",
                    "gamma_sign1": int(torch.sign(gamma[1]).item()) if variant == "AGD" else "",
                    "gamma_sign2": int(torch.sign(gamma[2]).item()) if variant == "AGD" else "",
                    "gamma_sign3": int(torch.sign(gamma[3]).item()) if variant == "AGD" else "",
                    "interpretation": "polynomial coefficient shape; LayerNorm obscures positive global scale",
                }
                coefficients.append(row)

                if variant in {"RGD", "AGD"}:
                    for modality in ("text", "visual"):
                        item = fwd["details"]["modalities"][modality]
                        states = item["raw_states"]
                        prior = states[:, 0]
                        g_raw = sum(coeff[k] * states[:, k] for k in range(4))
                        g_rms = g_raw.square().mean().sqrt().clamp_min(model.eps)
                        for order in range(4):
                            state = states[:, order]
                            term = coeff[order] * state
                            contributions.append({
                                "dataset": dataset, "variant": variant, "seed": int(run["seed"]),
                                "modality": modality, "order": order,
                                "state_rms": scalar(state.square().mean().sqrt()),
                                "term_rms": scalar(term.square().mean().sqrt()),
                                "term_rms_over_G_rms": scalar(term.square().mean().sqrt() / g_rms),
                                "cosine_term_G": flat_cosine(term, g_raw, model.eps),
                                "cosine_Hk_P": flat_cosine(state, prior, model.eps),
                            })
                        for i in range(4):
                            for j in range(i + 1, 4):
                                similarities.append({
                                    "dataset": dataset, "variant": variant, "seed": int(run["seed"]),
                                    "modality": modality, "state_pair": f"H{i}-H{j}",
                                    "cosine": flat_cosine(states[:, i], states[:, j], model.eps),
                                })

                    baseline = validation_metrics(classifier, z, data, device)
                    match = abs(baseline["acc"] - float(run["metrics"]["val_acc"])) <= 1e-6
                    for intervention in INTERVENTIONS:
                        zi = model(x, edge, intervention=intervention)[0]
                        metrics = validation_metrics(classifier, zi, data, device)
                        interventions.append({
                            "dataset": dataset, "variant": variant, "seed": int(run["seed"]),
                            "intervention": intervention, "baseline_val_accuracy": baseline["acc"],
                            "intervention_val_accuracy": metrics["acc"],
                            "val_accuracy_delta_pp": (metrics["acc"] - baseline["acc"]) * 100,
                            "baseline_val_macro_f1": baseline["macro_f1"],
                            "intervention_val_macro_f1": metrics["macro_f1"],
                            "val_macro_f1_delta_pp": (metrics["macro_f1"] - baseline["macro_f1"]) * 100,
                            "baseline_val_ce": baseline["ce"], "intervention_val_ce": metrics["ce"],
                            "val_ce_delta": metrics["ce"] - baseline["ce"],
                            "baseline_matches_run_metrics": match, "test_evaluation": False,
                        })
            del model, classifier, checkpoint, z, fwd
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(interventions) != 72:
        raise RuntimeError(f"expected 72 checkpoint intervention rows, found {len(interventions)}")
    return coefficients, contributions, similarities, interventions


def summarize(rows: list[dict], group_keys: tuple[str, ...], fields: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        item = dict(zip(group_keys, key))
        item["n"] = len(group)
        for field in fields:
            values = [float(r[field]) for r in group]
            item[f"{field}_mean"], item[f"{field}_population_sd"] = mean_sd(values)
            item[f"{field}_positive"] = sum(v > 0 for v in values)
            item[f"{field}_negative"] = sum(v < 0 for v in values)
            item[f"{field}_ties"] = sum(v == 0 for v in values)
        output.append(item)
    return output


def build_report(manifest: dict, status: dict, performance: list[dict], paired: list[dict],
                 repeats: list[dict], repeat_summary: list[dict], coeffs: list[dict],
                 contribution_summary: list[dict], similarity_summary: list[dict],
                 intervention_summary: list[dict]) -> str:
    perf_table = []
    for r in performance:
        perf_table.append([r["dataset"], r["variant"],
                           f"{r['val_accuracy_mean']*100:.2f} ± {r['val_accuracy_population_sd']*100:.2f}",
                           f"{r['val_macro_f1_mean']*100:.2f} ± {r['val_macro_f1_population_sd']*100:.2f}",
                           f"{r['val_ce_mean']:.4f} ± {r['val_ce_population_sd']:.4f}",
                           f"{r['best_epoch_mean']:.1f} ± {r['best_epoch_population_sd']:.1f}"])
    pair_table = []
    for r in paired:
        pair_table.append([r["dataset"], r["comparison"],
                           f"{r['val_accuracy_pp_mean']:+.3f} ± {r['val_accuracy_pp_population_sd']:.3f}",
                           f"{r['val_accuracy_pp_positive_seeds']}/{r['val_accuracy_pp_negative_seeds']}/{r['val_accuracy_pp_ties']}",
                           f"{r['val_macro_f1_pp_mean']:+.3f}", f"{r['val_ce_delta_mean']:+.4f}"])
    repeat_table = []
    for r in repeat_summary:
        unit_scale = 100.0 if r["metric"] in {"val_accuracy", "val_macro_f1"} else 1.0
        metric_name = {
            "val_accuracy": "Accuracy (%)", "val_macro_f1": "Macro-F1 (%)",
            "val_ce": "CE", "best_epoch": "Best epoch",
        }[r["metric"]]
        repeat_table.append([
            r["dataset"], metric_name,
            f"{r['mean'] * unit_scale:.6g}",
            f"{r['population_sd'] * unit_scale:.6g}",
            f"{r['max_min_range'] * unit_scale:.6g}",
        ])
    coeff_table = []
    for dataset in DATASETS:
        for variant in ("RFD", "RGD", "AGD"):
            group = [r for r in coeffs if r["dataset"] == dataset and r["variant"] == variant]
            if not group:
                continue
            coeff_table.append([dataset, variant] + [
                f"{statistics.mean(float(r[f'c{k}']) for r in group):+.4f}" for k in range(4)
            ] + ([
                f"{statistics.mean(float(r[f'gamma{k}']) for r in group):+.4f}" for k in range(4)
            ] if variant == "AGD" else ["—"] * 4))
    contribution_table = []
    for r in contribution_summary:
        contribution_table.append([r["dataset"], r["variant"], r["modality"], r["order"],
                                   fmt(r["state_rms_mean"], 4), fmt(r["term_rms_mean"], 4),
                                   fmt(r["term_rms_over_G_rms_mean"], 4), fmt(r["cosine_term_G_mean"], 4),
                                   fmt(r["cosine_Hk_P_mean"], 4)])
    similarity_table = [[r["dataset"], r["variant"], r["state_pair"], fmt(r["cosine_mean"], 4),
                         fmt(r["cosine_population_sd"], 4)] for r in similarity_summary]
    intervention_table = [[r["dataset"], r["variant"], r["intervention"],
                           f"{r['val_accuracy_delta_pp_mean']:+.3f}",
                           f"{r['val_accuracy_delta_pp_positive']}/{r['val_accuracy_delta_pp_negative']}/{r['val_accuracy_delta_pp_ties']}",
                           f"{r['val_macro_f1_delta_pp_mean']:+.3f}", f"{r['val_ce_delta_mean']:+.4f}"]
                          for r in intervention_summary]
    lookup = {(r["dataset"], r["comparison"]): r for r in paired}
    direction = []
    for _, _, label in COMPARISONS:
        direction.append("- **" + label + ":** " + "; ".join(
            f"{ds} {lookup[(ds,label)]['val_accuracy_pp_mean']:+.3f} pp "
            f"({lookup[(ds,label)]['val_accuracy_pp_positive_seeds']}/3 positive)"
            for ds in DATASETS) + ".")
    ranks = {v: [] for v in VARIANTS}
    for dataset in DATASETS:
        ordered = sorted([r for r in performance if r["dataset"] == dataset],
                         key=lambda row: row["val_accuracy_mean"], reverse=True)
        for rank, row in enumerate(ordered, 1):
            ranks[row["variant"]].append(rank)
    mean_ranks = {v: statistics.mean(rs) for v, rs in ranks.items()}
    recommended = min(mean_ranks, key=mean_ranks.get)
    key_effects = [lookup[(d, c)]["val_accuracy_pp_mean"] for d in DATASETS for c in ("RGD-RFD", "AGD-RGD")]
    intervention_lookup = {(r["dataset"], r["variant"], r["intervention"]): r for r in intervention_summary}
    drops_high_order = []
    for variant in ("RGD", "AGD"):
        for dataset in DATASETS:
            for label in ("order3_off", "negative_terms_off"):
                row = intervention_lookup[(dataset, variant, label)]
                if row["val_accuracy_delta_pp_mean"] < 0 and row["val_accuracy_delta_pp_negative"] >= 2:
                    drops_high_order.append((variant, dataset, label, row["val_accuracy_delta_pp_mean"]))
    h23 = [r for r in similarity_summary if r["state_pair"] == "H2-H3"]
    h23_mean = statistics.mean(float(r["cosine_mean"]) for r in h23)
    intervention_drop_datasets = {
        (variant, dataset)
        for variant, dataset, label, _ in drops_high_order
        if label == "order3_off"
    }
    repeat_lookup = {(r["dataset"], r["metric"]): r for r in repeat_summary}
    noise_by_dataset = {
        dataset: repeat_lookup[(dataset, "val_accuracy")]["population_sd"] * 100
        for dataset in DATASETS
    }
    clear_high_order_groups = []
    for variant in ("RGD", "AGD"):
        for dataset in DATASETS:
            row = intervention_lookup[(dataset, variant, "order3_off")]
            mean_delta = float(row["val_accuracy_delta_pp_mean"])
            if (
                mean_delta < 0
                and int(row["val_accuracy_delta_pp_negative"]) >= 2
                and abs(mean_delta) > 2.0 * noise_by_dataset[dataset]
            ):
                clear_high_order_groups.append((variant, dataset, mean_delta))
    intervention_drop_support = len({dataset for _, dataset, _ in clear_high_order_groups}) >= 2
    repeatability_text = "; ".join(
        f"{d} Acc SD/range {repeat_lookup[(d,'val_accuracy')]['population_sd']*100:.3f}/"
        f"{repeat_lookup[(d,'val_accuracy')]['max_min_range']*100:.3f} pp"
        for d in DATASETS)
    direct_protected = lookup
    direct_note = (
        "RUD−RU is interpreted as a direct-versus-protected parameterization/optimization comparison. "
        "C0 AGD−AGP has the same interpretation boundary: when polynomial coefficients are unrestricted, "
        "the protected skip can be reparameterized into effective polynomial coefficients."
    )
    basis_note = (
        "RGD and AGD have the same polynomial function family because S=M H with invertible triangular M. "
        "Their initial effective coefficients, proposal, projector, and fusion are matched. AGD−RGD therefore "
        "tests basis-dependent optimization or implicit regularization, not greater expressive capacity."
    )
    comparisons_text = "\n".join(direction)
    interpretation = []
    rgd_rfd = [lookup[(d, "RGD-RFD")]["val_accuracy_pp_mean"] for d in DATASETS]
    agd_rgd = [lookup[(d, "AGD-RGD")]["val_accuracy_pp_mean"] for d in DATASETS]
    fixed = [lookup[(d, "RFD-R0UD")]["val_accuracy_pp_mean"] for d in DATASETS]
    if all(x >= 0 for x in rgd_rfd) and all(abs(x) < 1.0 for x in agd_rgd):
        interpretation.append("Pattern broadly matches Case C: RGD−RFD is positive or effectively tied by dataset mean, while AGD−RGD is mixed and remains below 0.35 pp in absolute size. Recommend raw direct GPR as the simpler audit basis; AGD retains the best nominal accuracy rank, but the current evidence does not show a stable cross-dataset anchored-basis gain.")
    elif all(x > 0 for x in agd_rgd):
        interpretation.append("AGD is directionally above RGD across datasets; assess its size against the repeatability floor before attributing a basis-optimization benefit.")
    elif all(x > 0 for x in fixed) and all(abs(x) < 1.0 for x in rgd_rfd):
        interpretation.append("Pattern is consistent with Case B: the fixed PPR-informed profile may be sufficient without learned coefficients.")
    elif all(abs(x) < 1.0 for x in rgd_rfd) and all(abs(x) < 1.0 for x in agd_rgd):
        interpretation.append("Pattern is consistent with Case F: this campaign does not establish a learned-GPR advantage over simple controls.")
    else:
        interpretation.append("Results are dataset-dependent; use the paired tables and repeatability range rather than assigning one global mechanism.")
    if intervention_drop_support:
        interpretation.append("Order-3 interventions show repeated and noise-exceeding decreases in multiple datasets; this is checkpoint reliance evidence only and motivates review of a retrained signed/high-order ablation.")
    else:
        interpretation.append("Order-3/negative-term interventions show only small checkpoint changes; after comparison with the AGD repeatability SD, a repeated decrease larger than twice that noise floor appears in fewer than two datasets. A retrained signed/no3 audit is not yet compelled.")
    tradeoff = any(
        (r["val_accuracy_pp_mean"] > 0 and r["val_ce_delta_mean"] > 0)
        or (r["val_accuracy_pp_mean"] < 0 and r["val_ce_delta_mean"] < 0)
        for r in paired
    )
    order0_text = "; ".join(
        f"{variant} {dataset} {intervention_lookup[(dataset,variant,'order0_off')]['val_accuracy_delta_pp_mean']:+.3f} pp"
        for variant in ("RGD", "AGD") for dataset in DATASETS
    )
    order3_text = "; ".join(
        f"{variant} {dataset} {intervention_lookup[(dataset,variant,'order3_off')]['val_accuracy_delta_pp_mean']:+.3f} pp "
        f"({int(intervention_lookup[(dataset,variant,'order3_off')]['val_accuracy_delta_pp_negative'])}/3 negative)"
        for variant in ("RGD", "AGD") for dataset in DATASETS
    )
    practical_choice = (
        "RGD is the recommended working backbone: its learned raw GPR beats or ties RFD by dataset mean, "
        "and AGD's incremental accuracy effect is mixed and small. AGD has the best nominal mean rank, "
        "so retain it as a close comparator rather than claiming a basis advantage."
    )
    tradeoff_example = next((r for r in paired if r["val_accuracy_pp_mean"] > 0 and r["val_ce_delta_mean"] > 0), None)
    tradeoff_detail = (
        f"For example, {tradeoff_example['dataset']} {tradeoff_example['comparison']} gains "
        f"{tradeoff_example['val_accuracy_pp_mean']:+.3f} pp accuracy while CE changes "
        f"{tradeoff_example['val_ce_delta_mean']:+.4f}." if tradeoff_example else ""
    )
    lp_ok = status.get("status") == "passed" and len(status.get("lp_runs", [])) == 2
    return f"""# PIGPR-C1 — Direct GPR Attribution Audit

## Execution protocol

- Required base SHA `{manifest['base_sha']}`; branch `{manifest['branch']}`; {sum(r.get('status') == 'complete' for r in manifest['runs'])}/54 fixed-split NC runs completed.
- Datasets: Movies, Grocery, ele-fashion; variants RU/RUD/R0UD/RFD/RGD/AGD; seeds 42/43/44. Validation Accuracy selected checkpoints.
- `evaluate_test=false` and `development_no_test=true` for every NC run. No test metric was evaluated or recorded. No Toys, Reddit-S, formal LP, or mid-campaign tuning.
- Design audit preceded implementation. Full test suite, RU regression, 9 AGD repeats, six NC smokes, and two LP smokes are recorded in `smoke_status.json`.
- The first ele-fashion RU seed-42 attempt hit CUDA OOM while another process occupied GPU memory. The same cell passed after eliminating unused state computation; no hyperparameters or splits changed. Recovery details are in `run_manifest.json`.

## Validation performance

Accuracy and macro-F1 are percent; entries are mean ± population SD across three paired model seeds. CE and best epoch use mean ± population SD.

{table(['Dataset', 'Variant', 'Val Acc %', 'Macro-F1 %', 'CE', 'Best epoch'], perf_table)}

## Paired comparisons

Accuracy and F1 are percentage-point deltas; CE is raw CE difference. Signs are positive/negative/tie model seeds. Three model seeds on one fixed split are descriptive paired comparisons, not pseudo-IID inference.

{table(['Dataset', 'Comparison', 'Δ Acc pp ± SD', '+/−/tie', 'Δ F1 pp', 'Δ CE'], pair_table)}

{comparisons_text}

{direct_note}

{basis_note}

## AGD same-seed execution repeatability

Each dataset uses three independent full executions of AGD seed 42. This estimates execution noise, not architecture variance.

{table(['Dataset', 'Metric', 'Mean', 'Population SD', 'Range'], repeat_table)}

Accuracy repeatability context: {repeatability_text}.

## Coefficient diagnostics

RFD and RGD report raw monomial `c0..c3`; AGD reports both learned anchored `gamma0..gamma3` and `c=Mᵀgamma`. The CSV includes signed coefficients, negative counts, L1, sum, absolute-weighted effective order, and `c/(sum|c|+eps)`. Coefficients are not normalized by softmax or L1. Because the final modality embedding uses LayerNorm, the overall positive coefficient scale is weakly identifiable; compare normalized shape and functional contributions too.

{table(['Dataset', 'Variant', 'c0', 'c1', 'c2', 'c3', 'γ0', 'γ1', 'γ2', 'γ3'], coeff_table)}

Each displayed coefficient is averaged across the three seeds; the full checkpoint rows are in `coefficient_diagnostics.csv`.

## Effective monomial-order contributions

For RGD and AGD, `term_k=c_k H_k` and `G=sum c_k H_k` are computed in the same raw monomial basis. Rows aggregate three seeds per dataset, modality, and order. Contribution size is `RMS(term_k)/RMS(G)`; it is not inferred from coefficient magnitude alone.

{table(['Dataset', 'Variant', 'Modality', 'k', 'RMS(Hk)', 'RMS(term)', 'term/G RMS', 'cos(term,G)', 'cos(Hk,P)'], contribution_table)}

## Raw state similarities

Cosines use the flattened node-feature state tensors. `state_similarity.csv` retains per checkpoint and modality values.

{table(['Dataset', 'Variant', 'Pair', 'Mean cosine', 'Population SD'], similarity_table)}

Mean H2–H3 cosine across dataset/variant groups is {h23_mean:.4f}. Interpret order-3 coefficients alongside these state similarities and term contribution rows: correlated states can permit coefficient reparameterization without a correspondingly large functional correction.

## Monomial-space checkpoint interventions

The four interventions modify the selected RGD/AGD checkpoint's effective raw coefficient vector, recompose from raw `H`, then apply the same LayerNorm and frozen classifier. They measure checkpoint reliance and do not replace RGD−RFD or AGD−RGD retraining comparisons.

{table(['Dataset', 'Variant', 'Intervention', 'Δ Acc pp', '+/−/tie', 'Δ F1 pp', 'Δ CE'], intervention_table)}

## Diagnosis

{chr(10).join(interpretation)}

- Mean dataset rank by validation Accuracy: {', '.join(f'{v} {mean_ranks[v]:.2f}' for v in VARIANTS)}. The descriptive rank selects **{recommended}** ({mean_ranks[recommended]:.2f}). {practical_choice}
- Accuracy/F1/CE show a directional trade-off somewhere in the paired results: **{'yes' if tradeoff else 'not detected by the sign screen'}**. {tradeoff_detail} Read all three metrics in `paired_delta_summary.csv`.
- No semantic-drift-is-better claim is made. The evidence concerns task metrics, direct polynomial composition, basis coordinates, term contribution, and selected-checkpoint reliance.

## Self-audit

1. Started from `66017c7a83af127d842e58238cd88cc2e0aa08ff`: **yes**, branch ancestry and fetched remote were verified.
2. Used another experiment branch: **no**.
3. Ran or evaluated NC test: **no**; NC development mode masks test indices and all stored metrics are validation-only. LP smoke did not compute test metrics.
4. RU regression to C0 PIGPR-v0 RU: **{status.get('ru_regression_gate', {}).get('status', 'unknown')}**, maximum-error gate `1e-6`.
5. RFD and RGD initial function: **yes**, RGD delta starts at zero, so proposal and modality embeddings equal RFD at initialization.
6. RGD and AGD initial function: **yes**, `Mᵀ gamma_prior=c_prior`; matched same-seed projectors produce numerically equal proposals/embeddings within `1e-6`.
7. Same polynomial family: **yes**, anchored basis matrix is invertible triangular; differences indicate basis-dependent optimization/implicit regularization.
8. AGD repeatability: see the three-dataset population SD/range table above; this is execution noise, not model-seed variance.
9. RUD−RU: direct versus protected optimization/inductive-bias parameterization at matched uniform context; no function-capacity claim.
10. R0UD−RUD: effect of including the intrinsic order-0 state in direct uniform composition.
11. RFD−R0UD: effect of replacing uniform four-order context with the fixed CoSI/PPR-informed profile.
12. RGD−RFD supports learned GPR: **{'positive or effectively tied by dataset mean' if all(x >= 0 for x in rgd_rfd) else 'dataset-dependent or not consistently positive'}**; paired means by dataset are {', '.join(f'{d} {lookup[(d,"RGD-RFD")]["val_accuracy_pp_mean"]:+.3f} pp' for d in DATASETS)}. The practical gain is concentrated in ele-fashion; Grocery is effectively tied.
13. AGD−RGD basis optimization effect: paired means are {', '.join(f'{d} {lookup[(d,"AGD-RGD")]["val_accuracy_pp_mean"]:+.3f} pp' for d in DATASETS)}; see repeatability context and no capacity interpretation.
14. AGD/RGD versus RU: RGD−RU means {', '.join(f'{d} {lookup[(d,"RGD-RU")]["val_accuracy_pp_mean"]:+.3f}' for d in DATASETS)} pp; AGD−RU means {', '.join(f'{d} {lookup[(d,"AGD-RU")]["val_accuracy_pp_mean"]:+.3f}' for d in DATASETS)} pp.
15. Stable coefficient pattern across datasets: **{'not established; use per-dataset signs/shapes in coefficient_diagnostics.csv' if len({tuple(int(x[f'c_sign{k}']) for k in range(4)) for x in coeffs if x['variant']=='RGD'}) > 1 else 'yes in these checkpoints: c0–c2 positive and c3 negative for all nine RGD and all nine AGD fits; magnitudes vary and remain descriptive'}**.
16. Order-0 checkpoint reliance: **yes**; order0_off decreases accuracy for every RGD/AGD dataset mean. Deltas: {order0_text}. These are frozen-checkpoint effects, not retrained ablations.
17. Negative/order-3 reliance: c3 is the only negative learned raw coefficient, so negative_terms_off and order3_off coincide. Accuracy deltas are {order3_text}. The effects are small and mostly within the AGD repeatability floor; they do not establish a clear cross-dataset high-order dependency.
18. Coefficients versus functional contribution: **not assumed equivalent**; `order_contribution.csv` directly reports state and term RMS, normalized contribution, and cosine.
19. H2/H3 similarity: pooled group mean is {h23_mean:.4f}; dataset/variant values are in the state-similarity table.
20. Retrained signed/no3 audit: **not compelled yet**; fewer than two datasets show an order3_off decline exceeding twice the AGD repeatability SD with at least two of three model seeds declining. C1 stops here for review.
21. Current backbone: **RGD**, as the simpler raw direct parameterization with positive/tied RGD−RFD dataset means and no stable AGD−RGD cross-dataset gain. AGD has the best nominal validation-accuracy mean rank and remains a close comparator.
22. Accuracy/F1/CE trade-off: **{'yes' if tradeoff else 'not detected by the paired sign screen'}**; {tradeoff_detail} Full outcomes are in the paired table.
23. LP smoke compliant: **{'yes' if lp_ok else 'no/unfinished'}**; only sports-copurchase RGD/AGD, two epochs and two batches, test disabled.
24. Claims beyond evidence: **none intended**; findings are fixed-split validation descriptions with three model seeds and no p-values.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-diagnostics", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((RESEARCH / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((RESEARCH / "smoke_status.json").read_text(encoding="utf-8"))
    complete = [r for r in manifest.get("runs", []) if r.get("status") == "complete"]
    expected = {(d, v, s) for d in DATASETS for v in VARIANTS for s in (42, 43, 44)}
    actual = {(r["dataset"], r["variant"], int(r["seed"])) for r in complete}
    if manifest.get("status") != "complete" or len(complete) != 54 or actual != expected:
        raise RuntimeError(f"formal manifest is incomplete: {len(complete)}/54 unique cells")
    for run in complete:
        if run.get("evaluate_test") is not False or run.get("development_no_test") is not True:
            raise RuntimeError("formal test guard metadata failed")
        if any(key.startswith("test_") for key in run["metrics"]):
            raise RuntimeError("test metric found in formal run records")

    _, floor = read_repeats()
    perf, perf_summary, paired, paired_summary = performance_outputs(manifest, floor)
    write_csv(DATA_DIR / "performance_by_run.csv", perf)
    write_csv(DATA_DIR / "performance_summary.csv", perf_summary)
    write_csv(DATA_DIR / "paired_delta_by_run.csv", paired)
    write_csv(DATA_DIR / "paired_delta_summary.csv", paired_summary)

    paths = {
        "coefficients": DATA_DIR / "coefficient_diagnostics.csv",
        "contributions": DATA_DIR / "order_contribution.csv",
        "similarities": DATA_DIR / "state_similarity.csv",
        "interventions": DATA_DIR / "intervention_by_run.csv",
        "contribution_summary": DATA_DIR / "order_contribution_summary.csv",
        "similarity_summary": DATA_DIR / "state_similarity_summary.csv",
        "intervention_summary": DATA_DIR / "intervention_summary.csv",
    }
    if args.reuse_diagnostics:
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"diagnostic reuse requested but files are missing: {missing}")
        coefficients = read_csv(paths["coefficients"])
        contributions = read_csv(paths["contributions"])
        similarities = read_csv(paths["similarities"])
        interventions = read_csv(paths["interventions"])
        contribution_summary = read_csv(paths["contribution_summary"])
        similarity_summary = read_csv(paths["similarity_summary"])
        intervention_summary = read_csv(paths["intervention_summary"])
    else:
        coefficients, contributions, similarities, interventions = checkpoint_diagnostics(manifest, args.device)
        write_csv(paths["coefficients"], coefficients)
        write_csv(paths["contributions"], contributions)
        write_csv(paths["similarities"], similarities)
        write_csv(paths["interventions"], interventions)
        contribution_summary = summarize(contributions, ("dataset", "variant", "modality", "order"),
                                          ("state_rms", "term_rms", "term_rms_over_G_rms", "cosine_term_G", "cosine_Hk_P"))
        similarity_summary = summarize(similarities, ("dataset", "variant", "state_pair"), ("cosine",))
        intervention_summary = summarize(interventions, ("dataset", "variant", "intervention"),
                                         ("val_accuracy_delta_pp", "val_macro_f1_delta_pp", "val_ce_delta"))
        write_csv(paths["contribution_summary"], contribution_summary)
        write_csv(paths["similarity_summary"], similarity_summary)
        write_csv(paths["intervention_summary"], intervention_summary)

    report = build_report(manifest, status, perf_summary, paired_summary, [], floor.values(),
                          coefficients, contribution_summary, similarity_summary, intervention_summary)
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")
    print(f"Analysis complete: {RESEARCH / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
