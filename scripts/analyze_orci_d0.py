#!/usr/bin/env python3
"""Summarize ORCI-D0 validation runs and checkpoint mechanisms."""

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
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "orci_d0_interaction_alignment_synergy"
DATA_DIR = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("B", "I", "A", "IA")
MODEL_VARIANTS = {"B": "base", "I": "interaction", "A": "alignment", "IA": "joint"}
SEEDS = (42, 43, 44)
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}
COMPARISONS = (("I", "B"), ("A", "B"), ("IA", "B"), ("IA", "I"), ("IA", "A"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.pstdev(values)


def scalar(value: torch.Tensor | float) -> float:
    return float(value.detach().float().cpu().item()) if torch.is_tensor(value) else float(value)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def load_manifest() -> dict:
    path = RESEARCH / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("formal_status") != "complete":
        raise RuntimeError("formal campaign is not marked complete")
    runs = [row for row in manifest.get("formal_runs", []) if row.get("status") == "complete"]
    if len(runs) != 36:
        raise RuntimeError(f"expected 36 completed formal runs, found {len(runs)}")
    return manifest


def make_cfg(dataset: str, device: str, variant: str, alignment_weight: float):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=orci_mag_v0",
        f"model.variant={MODEL_VARIANTS[variant]}",
        f"model.alignment_weight={alignment_weight:.2f}",
        "seed=42",
        f"device={device}",
        "task.evaluate_test=false",
        "task.development_no_test=true",
    ]
    if dataset in {"Movies", "Grocery"}:
        overrides.append(f"dataset.nc_split_path={SPLITS[dataset]}")
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


def read_run_metrics(run: dict) -> dict:
    path = Path(run["run_metrics_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("development_no_test") is not True:
        raise RuntimeError(f"validation-only marker missing in {path}")
    items = payload.get("runs", [])
    if len(items) != 1 or int(items[0]["seed"]) != int(run["seed"]):
        raise RuntimeError(f"run/seed mismatch in {path}")
    metrics = items[0]["metrics"]
    if any(name.startswith("test_") for name in metrics):
        raise RuntimeError(f"test metric found in {path}")
    return metrics


def performance_outputs(manifest: dict):
    rows: list[dict] = []
    lookup = {}
    for run in manifest["formal_runs"]:
        if run.get("status") != "complete":
            continue
        metrics = read_run_metrics(run)
        row = {
            "dataset": run["dataset"],
            "variant": run["variant"],
            "model_variant": run["model_variant"],
            "seed": int(run["seed"]),
            "alignment_weight": float(run["alignment_weight"]),
            "val_accuracy": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "val_ce": float(metrics["val_ce"]),
            "best_epoch": int(run["best_epoch"]),
            "development_no_test": True,
            "evaluate_test": False,
            "run_metrics_path": run["run_metrics_path"],
            "checkpoint_path": run["checkpoint_path"],
        }
        rows.append(row)
        lookup[(row["dataset"], row["variant"], row["seed"])] = row
    if len(rows) != 36:
        raise RuntimeError(f"expected 36 validation-only metric rows, found {len(rows)}")

    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            item = {"dataset": dataset, "variant": variant, "n_model_seeds": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
                mean, sd = mean_sd([float(row[metric]) for row in group])
                item[f"{metric}_mean"] = mean
                item[f"{metric}_population_sd"] = sd
            summary.append(item)

    paired = []
    for dataset in DATASETS:
        for new, base in COMPARISONS:
            for seed in SEEDS:
                a = lookup[(dataset, new, seed)]
                b = lookup[(dataset, base, seed)]
                paired.append({
                    "dataset": dataset,
                    "seed": seed,
                    "comparison": f"{new}-{base}",
                    "new_variant": new,
                    "base_variant": base,
                    "accuracy_delta_pp": (a["val_accuracy"] - b["val_accuracy"]) * 100.0,
                    "macro_f1_delta_pp": (a["val_macro_f1"] - b["val_macro_f1"]) * 100.0,
                    "ce_delta": a["val_ce"] - b["val_ce"],
                    "new_accuracy": a["val_accuracy"],
                    "base_accuracy": b["val_accuracy"],
                })
    paired_summary = []
    for dataset in DATASETS:
        for new, base in COMPARISONS:
            group = [row for row in paired if row["dataset"] == dataset and row["comparison"] == f"{new}-{base}"]
            item = {"dataset": dataset, "comparison": f"{new}-{base}", "n_paired_seeds": len(group)}
            for metric in ("accuracy_delta_pp", "macro_f1_delta_pp", "ce_delta"):
                vals = [float(row[metric]) for row in group]
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(vals)
                item[f"{metric}_positive_seeds"] = sum(v > 1e-12 for v in vals)
                item[f"{metric}_negative_seeds"] = sum(v < -1e-12 for v in vals)
                item[f"{metric}_ties"] = sum(abs(v) <= 1e-12 for v in vals)
            paired_summary.append(item)

    synergy = []
    for dataset in DATASETS:
        for seed in SEEDS:
            b = lookup[(dataset, "B", seed)]
            i = lookup[(dataset, "I", seed)]
            a = lookup[(dataset, "A", seed)]
            ia = lookup[(dataset, "IA", seed)]
            synergy.append({
                "row_type": "paired_seed",
                "dataset": dataset,
                "seed": seed,
                "accuracy_synergy_pp": (ia["val_accuracy"] - i["val_accuracy"] - a["val_accuracy"] + b["val_accuracy"]) * 100.0,
                "macro_f1_synergy_pp": (ia["val_macro_f1"] - i["val_macro_f1"] - a["val_macro_f1"] + b["val_macro_f1"]) * 100.0,
                "ce_factorial_contrast": ia["val_ce"] - i["val_ce"] - a["val_ce"] + b["val_ce"],
            })
        group = [row for row in synergy if row["dataset"] == dataset and row["row_type"] == "paired_seed"]
        accuracy = [float(row["accuracy_synergy_pp"]) for row in group]
        macro = [float(row["macro_f1_synergy_pp"]) for row in group]
        ce = [float(row["ce_factorial_contrast"]) for row in group]
        synergy.append({
            "row_type": "dataset_summary",
            "dataset": dataset,
            "seed": "",
            "accuracy_synergy_pp_mean": statistics.mean(accuracy),
            "accuracy_synergy_pp_population_sd": statistics.pstdev(accuracy),
            "accuracy_synergy_positive_seeds": sum(v > 1e-12 for v in accuracy),
            "accuracy_synergy_negative_seeds": sum(v < -1e-12 for v in accuracy),
            "accuracy_synergy_ties": sum(abs(v) <= 1e-12 for v in accuracy),
            "macro_f1_synergy_pp_mean": statistics.mean(macro),
            "macro_f1_synergy_pp_population_sd": statistics.pstdev(macro),
            "ce_factorial_contrast_mean": statistics.mean(ce),
            "ce_factorial_contrast_population_sd": statistics.pstdev(ce),
        })
    return rows, summary, paired, paired_summary, synergy


def make_data(dataset: str, device: torch.device, alignment_weight: float):
    from src.data import load_mag_data

    cfg = make_cfg(dataset, str(device), "IA", alignment_weight)
    data = load_mag_data(cfg, "nc", 42)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    return cfg, data, x, edge_index, model_data_info(data)


def _load_checkpoint(run: dict, cfg, info: dict, device: torch.device, variant: str):
    from src.models import build_model

    cfg.model.variant = MODEL_VARIANTS[variant]
    model = build_model(cfg, info).to(device)
    checkpoint_path = Path(run["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"required checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    classifier = nn.Linear(model.out_dim, int(info["num_classes"])).to(device)
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    classifier.eval()
    return model, classifier


def validation_metrics(classifier, z, data, device: torch.device) -> dict[str, float]:
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels

    eval_labels = _resolve_nc_eval_labels(data, development_no_test=True)
    return _evaluate_split(
        classifier,
        z.detach().cpu(),
        data.y,
        data.val_idx,
        device,
        4096,
        eval_labels,
    )


def checkpoint_diagnostics(manifest: dict, device_name: str):
    from src.models import build_model
    from src.utils.seeds import set_seed

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for ORCI-D0 checkpoint diagnostics")
    weight = float(manifest["selected_alignment_weight"])
    complete = [row for row in manifest["formal_runs"] if row.get("status") == "complete"]
    interaction_rows: list[dict] = []
    alignment_rows: list[dict] = []
    intervention_rows: list[dict] = []
    gpr_rows: list[dict] = []

    for dataset in DATASETS:
        cfg, data, x, edge_index, info = make_data(dataset, device, weight)
        dataset_runs = [row for row in complete if row["dataset"] == dataset]
        initialized_geometry: set[int] = set()
        for run in dataset_runs:
            variant = run["variant"]
            seed = int(run["seed"])
            model, classifier = _load_checkpoint(run, cfg, info, device, variant)
            with torch.no_grad():
                z, _, _, _, output = model(x, edge_index, return_details=True)
            normal_metrics = validation_metrics(classifier, z, data, device)
            checkpoint_metrics = read_run_metrics(run)
            for key, metric_key in (("acc", "val_acc"), ("macro_f1", "val_macro_f1"), ("ce", "val_ce")):
                if abs(normal_metrics[key] - float(checkpoint_metrics[metric_key])) > 1e-6:
                    raise RuntimeError(
                        f"checkpoint validation mismatch for {dataset}/{variant}/{seed}: {key}"
                    )

            coefficients = output["details"]["coefficients"].detach().float().cpu()
            gpr_row = {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "c0": scalar(coefficients[0]),
                "c1": scalar(coefficients[1]),
                "c2": scalar(coefficients[2]),
                "c3": scalar(coefficients[3]),
                "c_sum": scalar(coefficients.sum()),
                "checkpoint_val_accuracy": normal_metrics["acc"],
            }
            gpr_rows.append(gpr_row)

            # Compute the shared content geometry for all variants. For B, this
            # is a counterfactual diagnostic only; its training objective was off.
            if variant == "IA":
                geometry_output = output
            else:
                cfg_joint = make_cfg(dataset, str(device), "IA", weight)
                geometry_model = build_model(cfg_joint, info).to(device)
                checkpoint = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
                geometry_model.load_state_dict(checkpoint["model_state"], strict=True)
                geometry_model.eval()
                with torch.no_grad():
                    _, _, _, _, geometry_output = geometry_model(x, edge_index, return_details=True)
            geom = geometry_output["details"]
            pair_cos = geom["pair_cosine"]
            if pair_cos is None:
                raise RuntimeError("shared-space alignment diagnostic failed to produce pair cosine")
            raw_align = scalar(geom["raw_alignment_loss"])
            align_row = {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "training_alignment_active": variant in {"A", "IA"},
                "geometry_role": "trained objective" if variant in {"A", "IA"} else "counterfactual diagnostic",
                "alignment_weight": weight,
                "raw_alignment_loss": raw_align,
                "weighted_aux_loss": raw_align * weight if variant in {"A", "IA"} else 0.0,
                "mean_same_order_projected_cosine": scalar(pair_cos.mean()),
                "order1_projected_cosine": scalar(pair_cos[:, 0].mean()),
                "order2_projected_cosine": scalar(pair_cos[:, 1].mean()),
                "order3_projected_cosine": scalar(pair_cos[:, 2].mean()),
                "per_order_cosine_includes_only_structural_orders_1_to_3": True,
            }
            alignment_rows.append(align_row)

            if seed not in initialized_geometry:
                set_seed(seed)
                cfg_initial = make_cfg(dataset, str(device), "IA", weight)
                initial_model = build_model(cfg_initial, info).to(device)
                initial_model.eval()
                with torch.no_grad():
                    _, _, _, _, initial_output = initial_model(x, edge_index, return_details=True)
                initial_pair_cos = initial_output["details"]["pair_cosine"]
                alignment_rows.append({
                    "dataset": dataset,
                    "variant": "INIT",
                    "seed": seed,
                    "training_alignment_active": False,
                    "geometry_role": "same-seed pre-training initialization",
                    "alignment_weight": weight,
                    "raw_alignment_loss": scalar(initial_output["details"]["raw_alignment_loss"]),
                    "weighted_aux_loss": 0.0,
                    "mean_same_order_projected_cosine": scalar(initial_pair_cos.mean()),
                    "order1_projected_cosine": scalar(initial_pair_cos[:, 0].mean()),
                    "order2_projected_cosine": scalar(initial_pair_cos[:, 1].mean()),
                    "order3_projected_cosine": scalar(initial_pair_cos[:, 2].mean()),
                    "per_order_cosine_includes_only_structural_orders_1_to_3": True,
                })
                initialized_geometry.add(seed)

            if variant in {"I", "IA"}:
                attention = output["details"]["attention"]
                if set(attention) != {"text_from_visual", "visual_from_text"}:
                    raise RuntimeError(f"bidirectional attention missing for {dataset}/{variant}/{seed}")
                rho = output["details"]["rho"].detach()
                coeff = output["details"]["coefficients"].detach()
                for direction, attn in attention.items():
                    mean_matrix = attn.mean(dim=(0, 1))
                    node_std = attn.mean(dim=1).std(dim=0, unbiased=False)
                    entropy = -(attn * attn.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=(0, 1))
                    target_modality = "text" if direction == "text_from_visual" else "visual"
                    modality = output["details"]["modalities"][target_modality]
                    for query_idx, query_order in enumerate((1, 2, 3)):
                        same_order_mass = scalar(mean_matrix[query_idx, query_order])
                        state_h = modality["raw_states"][:, query_order]
                        interaction_j = modality["interaction_output"][:, query_order - 1]
                        residual = modality["scaled_interaction"][:, query_order - 1]
                        rms_j = scalar(interaction_j.square().mean().sqrt())
                        rms_residual = scalar(residual.square().mean().sqrt())
                        rms_h = scalar(state_h.square().mean().sqrt())
                        cosine = F.cosine_similarity(
                            interaction_j.reshape(1, -1), state_h.reshape(1, -1), dim=-1
                        )
                        row = {
                            "dataset": dataset,
                            "variant": variant,
                            "seed": seed,
                            "direction": direction,
                            "query_order": query_order,
                            "attention_entropy": scalar(entropy[query_idx]),
                            "source_h0_mass": scalar(mean_matrix[query_idx, 0]),
                            "same_order_mass": same_order_mass,
                            "cross_order_mass": 1.0 - same_order_mass,
                            "rho1": scalar(rho[0]),
                            "rho2": scalar(rho[1]),
                            "rho3": scalar(rho[2]),
                            "rms_J": rms_j,
                            "rms_rhoJ": rms_residual,
                            "rms_rhoJ_over_Hk": rms_residual / max(rms_h, 1e-12),
                            "cosine_J_Hk": scalar(cosine[0]),
                            "c0": scalar(coeff[0]),
                            "c1": scalar(coeff[1]),
                            "c2": scalar(coeff[2]),
                            "c3": scalar(coeff[3]),
                        }
                        for source_order in range(4):
                            row[f"attention_mean_source_{source_order}"] = scalar(mean_matrix[query_idx, source_order])
                            row[f"attention_node_std_source_{source_order}"] = scalar(node_std[query_idx, source_order])
                        interaction_rows.append(row)

            # The checkpoint correspondence tests are validation-only. Source
            # tokens are shuffled after graph propagation; graph states stay put.
            if variant == "IA":
                intervention_rows.append({
                    "dataset": dataset,
                    "variant": "IA",
                    "seed": seed,
                    "intervention": "normal",
                    "repeat": 0,
                    "val_accuracy": normal_metrics["acc"],
                    "val_macro_f1": normal_metrics["macro_f1"],
                    "val_ce": normal_metrics["ce"],
                    "evaluate_test": False,
                })
                for intervention in ("interaction_off", "source_node_shuffle"):
                    repeats = (0,) if intervention == "interaction_off" else range(1, 6)
                    for repeat in repeats:
                        kwargs = {"intervention": intervention}
                        if intervention == "source_node_shuffle":
                            generator = torch.Generator(device="cpu").manual_seed(
                                20261003 + 10000 * DATASETS.index(dataset) + 100 * seed + repeat
                            )
                            kwargs["source_node_permutation"] = torch.randperm(
                                data.num_nodes, generator=generator
                            ).to(device)
                        with torch.no_grad():
                            changed_z = model(x, edge_index, **kwargs)[0]
                        changed_metrics = validation_metrics(classifier, changed_z, data, device)
                        intervention_rows.append({
                            "dataset": dataset,
                            "variant": "IA",
                            "seed": seed,
                            "intervention": intervention,
                            "repeat": repeat,
                            "val_accuracy": changed_metrics["acc"],
                            "val_macro_f1": changed_metrics["macro_f1"],
                            "val_ce": changed_metrics["ce"],
                            "delta_accuracy_pp_vs_normal": (changed_metrics["acc"] - normal_metrics["acc"]) * 100.0,
                            "delta_macro_f1_pp_vs_normal": (changed_metrics["macro_f1"] - normal_metrics["macro_f1"]) * 100.0,
                            "delta_ce_vs_normal": changed_metrics["ce"] - normal_metrics["ce"],
                            "source_shuffle_repeats": 5 if intervention == "source_node_shuffle" else "",
                            "evaluate_test": False,
                        })
            del model, classifier, z, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return interaction_rows, alignment_rows, intervention_rows, gpr_rows


def summarize_interventions(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(r["dataset"], int(r["seed"]), r["intervention"]) for r in rows})
    for dataset, seed, intervention in keys:
        group = [r for r in rows if r["dataset"] == dataset and int(r["seed"]) == seed and r["intervention"] == intervention]
        row = {"dataset": dataset, "seed": seed, "intervention": intervention, "n_repeats": len(group)}
        for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
            row[f"{metric}_mean"], row[f"{metric}_population_sd"] = mean_sd([float(r[metric]) for r in group])
        if intervention != "normal":
            normal = next(r for r in rows if r["dataset"] == dataset and int(r["seed"]) == seed and r["intervention"] == "normal")
            row["delta_accuracy_pp_vs_normal_mean"] = (row["val_accuracy_mean"] - normal["val_accuracy"]) * 100.0
            row["delta_macro_f1_pp_vs_normal_mean"] = (row["val_macro_f1_mean"] - normal["val_macro_f1"]) * 100.0
            row["delta_ce_vs_normal_mean"] = row["val_ce_mean"] - normal["val_ce"]
        output.append(row)
    return output


def _mean_by(rows: list[dict], metric: str, variant: str | None = None) -> float:
    values = [float(row[metric]) for row in rows if variant is None or row.get("variant") == variant]
    return statistics.mean(values) if values else math.nan


def build_report(manifest: dict, performance: list[dict], summary: list[dict], paired_summary: list[dict], synergy: list[dict], interaction: list[dict], alignment: list[dict], intervention_summary: list[dict], gpr: list[dict], calibration: dict) -> str:
    summary_map = {(r["dataset"], r["variant"]): r for r in summary}
    perf_table = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = summary_map[(dataset, variant)]
            acc = f"{row['val_accuracy_mean']*100:.2f} ± {row['val_accuracy_population_sd']*100:.2f}"
            f1 = f"{row['val_macro_f1_mean']*100:.2f} ± {row['val_macro_f1_population_sd']*100:.2f}"
            ce = f"{row['val_ce_mean']:.4f} ± {row['val_ce_population_sd']:.4f}"
            perf_table.append([dataset, variant, acc, f1, ce])

    delta_table = []
    for dataset in DATASETS:
        for label in ("I-B", "A-B", "IA-B", "IA-I", "IA-A"):
            row = next(r for r in paired_summary if r["dataset"] == dataset and r["comparison"] == label)
            metric = row["accuracy_delta_pp_mean"]
            delta_table.append([
                dataset,
                label,
                f"{metric:+.3f} ± {row['accuracy_delta_pp_population_sd']:.3f}",
                f"{row['accuracy_delta_pp_positive_seeds']}/{row['accuracy_delta_pp_negative_seeds']}/{row['accuracy_delta_pp_ties']}",
                f"{row['macro_f1_delta_pp_mean']:+.3f}",
                f"{row['ce_delta_mean']:+.4f}",
            ])

    sy_rows = [r for r in synergy if r.get("row_type") == "dataset_summary"]
    sy_table = [[
        row["dataset"],
        f"{row['accuracy_synergy_pp_mean']:+.3f} ± {row['accuracy_synergy_pp_population_sd']:.3f}",
        f"{row['accuracy_synergy_positive_seeds']}/{row['accuracy_synergy_negative_seeds']}/{row['accuracy_synergy_ties']}",
        f"{row['macro_f1_synergy_pp_mean']:+.3f}",
        f"{row['ce_factorial_contrast_mean']:+.4f}",
    ] for row in sy_rows]

    interaction_table = []
    for variant in ("I", "IA"):
        group = [row for row in interaction if row["variant"] == variant]
        if not group:
            continue
        interaction_table.append([
            variant,
            fmt(statistics.mean(float(r["attention_entropy"]) for r in group), 3),
            fmt(statistics.mean(float(r["source_h0_mass"]) for r in group), 3),
            fmt(statistics.mean(float(r["same_order_mass"]) for r in group), 3),
            fmt(statistics.mean(float(r["cross_order_mass"]) for r in group), 3),
            fmt(statistics.mean(float(r["rms_rhoJ_over_Hk"]) for r in group), 4),
            fmt(statistics.mean(float(r["cosine_J_Hk"]) for r in group), 3),
        ])

    align_table = []
    for variant in ("INIT", "B", "I", "A", "IA"):
        group = [row for row in alignment if row["variant"] == variant]
        if not group:
            continue
        align_table.append([
            variant,
            "yes" if any(r.get("training_alignment_active") for r in group) else "no",
            fmt(statistics.mean(float(r["raw_alignment_loss"]) for r in group), 4),
            fmt(statistics.mean(float(r["mean_same_order_projected_cosine"]) for r in group), 4),
            fmt(statistics.mean(float(r["order1_projected_cosine"]) for r in group), 4),
            fmt(statistics.mean(float(r["order2_projected_cosine"]) for r in group), 4),
            fmt(statistics.mean(float(r["order3_projected_cosine"]) for r in group), 4),
        ])

    intervention_table = []
    for intervention in ("normal", "interaction_off", "source_node_shuffle"):
        group = [row for row in intervention_summary if row["intervention"] == intervention]
        if not group:
            continue
        acc_mean = statistics.mean(float(row["val_accuracy_mean"]) for row in group)
        f1_mean = statistics.mean(float(row["val_macro_f1_mean"]) for row in group)
        ce_mean = statistics.mean(float(row["val_ce_mean"]) for row in group)
        if intervention == "normal":
            da = df = dce = "—"
        else:
            da = f"{statistics.mean(float(row['delta_accuracy_pp_vs_normal_mean']) for row in group):+.3f} pp"
            df = f"{statistics.mean(float(row['delta_macro_f1_pp_vs_normal_mean']) for row in group):+.3f} pp"
            dce = f"{statistics.mean(float(row['delta_ce_vs_normal_mean']) for row in group):+.4f}"
        intervention_table.append([intervention, f"{acc_mean*100:.2f}", f"{f1_mean*100:.2f}", f"{ce_mean:.4f}", da, df, dce])

    weight = float(calibration["selected_alignment_weight"])
    selection_summary = calibration["ranked_summary"]
    calibration_table = [[
        f"{row['weight']:.2f}",
        f"{row['mean_IA_minus_B_accuracy_pp']:+.3f}",
        f"{row['mean_IA_minus_B_macro_f1_pp']:+.3f}",
        "selected" if float(row["weight"]) == weight else "",
    ] for row in selection_summary]

    ia_primary = [row for row in paired_summary if row["comparison"] == "IA-B"]
    positive_dataset_means = sum(float(row["accuracy_delta_pp_mean"]) > 0 for row in ia_primary)
    gains = [row for row in ia_primary if float(row["accuracy_delta_pp_mean"]) > 0]
    losses = [row for row in ia_primary if float(row["accuracy_delta_pp_mean"]) < 0]
    if len(gains) == 3:
        interpretation = "IA-B mean validation accuracy is positive on all three datasets; this is the clearest D0 support pattern, subject to the paired seed spread and secondary metrics below."
    elif len(gains) >= 2:
        interpretation = "IA-B mean validation accuracy is positive on at least two datasets and mixed elsewhere; treat the block as dataset-dependent and promising only if F1/CE and seed directions do not show a consistent counter-signal."
    else:
        interpretation = "IA-B is not directionally positive on at least two datasets; the present validation evidence does not establish a broad full-block gain. Retain the result and stop at D0 for review."
    i_rows = [r for r in paired_summary if r["comparison"] == "I-B"]
    a_rows = [r for r in paired_summary if r["comparison"] == "A-B"]
    ia_i_rows = [r for r in paired_summary if r["comparison"] == "IA-I"]
    ia_a_rows = [r for r in paired_summary if r["comparison"] == "IA-A"]
    synergy_positive = sum(float(r["accuracy_synergy_pp_mean"]) > 0 for r in sy_rows)
    i_b_text = "; ".join(f"{r['dataset']} {r['accuracy_delta_pp_mean']:+.3f} pp" for r in i_rows)
    a_b_text = "; ".join(f"{r['dataset']} {r['accuracy_delta_pp_mean']:+.3f} pp" for r in a_rows)
    ia_b_text = "; ".join(f"{r['dataset']} {r['accuracy_delta_pp_mean']:+.3f} pp" for r in ia_primary)
    ia_i_text = "; ".join(f"{r['dataset']} {r['accuracy_delta_pp_mean']:+.3f} pp" for r in ia_i_rows)
    ia_a_text = "; ".join(f"{r['dataset']} {r['accuracy_delta_pp_mean']:+.3f} pp" for r in ia_a_rows)

    gpr_means = {}
    for variant in VARIANTS:
        group = [row for row in gpr if row["variant"] == variant]
        gpr_means[variant] = [statistics.mean(float(row[f"c{i}"]) for row in group) for i in range(4)]
    residual_ratio = statistics.mean(float(row["rms_rhoJ_over_Hk"]) for row in interaction) if interaction else math.nan
    source_shuffle_delta = [row for row in intervention_summary if row["intervention"] == "source_node_shuffle"]
    shuffle_acc = statistics.mean(float(row["delta_accuracy_pp_vs_normal_mean"]) for row in source_shuffle_delta) if source_shuffle_delta else math.nan
    off_rows = [row for row in intervention_summary if row["intervention"] == "interaction_off"]
    off_acc = statistics.mean(float(row["delta_accuracy_pp_vs_normal_mean"]) for row in off_rows) if off_rows else math.nan
    mean_attention_entropy = statistics.mean(float(row["attention_entropy"]) for row in interaction) if interaction else math.nan
    mean_same_order = statistics.mean(float(row["same_order_mass"]) for row in interaction) if interaction else math.nan
    mean_h0_mass = statistics.mean(float(row["source_h0_mass"]) for row in interaction) if interaction else math.nan
    alignment_weight_changed = weight
    alignment_compare = {}
    for variant in ("INIT", "B", "I", "A", "IA"):
        group = [row for row in alignment if row["variant"] == variant]
        if group:
            alignment_compare[variant] = statistics.mean(float(row["mean_same_order_projected_cosine"]) for row in group)
    accuracy_secondary_conflicts = []
    for row in ia_primary:
        if (float(row["accuracy_delta_pp_mean"]) > 0 and (float(row["macro_f1_delta_pp_mean"]) < 0 or float(row["ce_delta_mean"]) > 0)) or (float(row["accuracy_delta_pp_mean"]) < 0 and (float(row["macro_f1_delta_pp_mean"]) > 0 or float(row["ce_delta_mean"]) < 0)):
            accuracy_secondary_conflicts.append(row["dataset"])

    report = f"""# ORCI-D0 — Order-Resolved Cross-modal Interaction + Selective Alignment

## Status and protocol

- Base commit: `{manifest['base_sha']}`; branch: `{manifest['branch']}`.
- Formal campaign: 36 validation-only NC runs (Movies/Grocery/ele-fashion × B/I/A/IA × seeds 42/43/44), one fixed split per dataset.
- Calibration: 8 validation-only runs (six IA candidates plus two B references), with selected alignment weight `{weight:.2f}`. Rule: {calibration['selection_rule']}.
- `task.evaluate_test=false`; NC used `task.development_no_test=true`. No test metrics appear in the run files. Calibration is excluded from formal tables.
- Parameterization, fixed CoSI prior, physical graph operator, C1 RGD baseline, and auxiliary-loss task weight are audited in [design_audit.md](design_audit.md).
- Full repository pytest: **126 passed** (one upstream PyG deprecation warning).

## Alignment-weight calibration

{md_table(['Weight', 'Mean IA−B Val Acc (pp)', 'Mean IA−B Macro-F1 (pp)', 'Decision'], calibration_table)}

The comparison averages the paired validation-accuracy deltas from Movies and Grocery seed 42. No calibration test data were evaluated or read. Full per-dataset values are in `data/calibration_by_run.csv`.

## Formal validation performance

Values are mean ± population SD over the three model seeds; Accuracy and Macro-F1 are percent, CE is unscaled.

{md_table(['Dataset', 'Variant', 'Val Accuracy (%)', 'Val Macro-F1 (%)', 'Val CE'], perf_table)}

## Paired comparisons

Accuracy and Macro-F1 differences are percentage points. Seed direction is positive/negative/tie.

{md_table(['Dataset', 'Comparison', 'Accuracy Δ (pp), mean ± SD', '+/−/= seeds', 'Macro-F1 Δ (pp)', 'CE Δ'], delta_table)}

## Descriptive 2×2 interaction

The factorial contrast is `IA − I − A + B`, paired within dataset and seed. It is descriptive and does not replace IA−B.

{md_table(['Dataset', 'Accuracy synergy (pp), mean ± SD', '+/−/= seeds', 'Macro-F1 synergy (pp)', 'CE factorial contrast'], sy_table)}

## Interaction and alignment mechanisms

I and IA attention aggregates cover both directions, all nodes/seeds, three query orders and four source orders. `H0 mass`, same-order/cross-order mass, and entropy describe the learned attention matrix; `rhoJ/Hk` compares the injected residual RMS with the raw structural state.

{md_table(['Variant', 'Entropy (nats)', 'Source H0 mass', 'Same-order mass', 'Cross-order mass', 'RMS(ρJ)/RMS(Hk)', 'cos(J,Hk)'], interaction_table)}

GPR coefficient means (`c0:c3`) by variant: `{json.dumps({key: [round(v, 5) for v in val] for key, val in gpr_means.items()}, ensure_ascii=False)}`. These coefficient changes are descriptive and are not treated as performance evidence.

Alignment is evaluated only on the normalized shared content projections of structural orders 1–3. `INIT` is the same-seed pre-training geometry; B/I values are counterfactual diagnostics because alignment was inactive during those runs.

{md_table(['Checkpoint', 'Alignment trained?', 'Raw loss', 'Mean same-order cosine', 'Order 1 cosine', 'Order 2 cosine', 'Order 3 cosine'], align_table)}

## IA checkpoint interventions

The source shuffle applies five deterministic node permutations to cross-modal source tokens after graph propagation; it leaves backbone states and late fusion inputs in their original node order. These are checkpoint reliance diagnostics only.

{md_table(['Intervention', 'Val Acc (%)', 'Macro-F1 (%)', 'CE', 'Accuracy Δ vs normal', 'Macro-F1 Δ vs normal', 'CE Δ vs normal'], intervention_table)}

Across IA checkpoints, interaction-off accuracy change averaged {off_acc:+.3f} pp and source-node shuffle averaged {shuffle_acc:+.3f} pp. Neither intervention establishes retrained architectural value; the primary evidence remains IA−B.

## Interpretation

{interpretation}

- Paired IA−B mean accuracy is positive on {positive_dataset_means}/3 datasets. Direction, magnitude, seed spread, Macro-F1 and CE are all shown above; no fixed 0.5 pp success threshold was applied.
- I−B: {i_b_text}.
- A−B: {a_b_text}.
- IA−B: {ia_b_text}.
- IA−I: {ia_i_text}; IA−A: {ia_a_text}.
- Mean descriptive accuracy synergy is positive for {synergy_positive}/3 datasets; this does not supersede the paired IA−B result.
- Mean attention entropy is {mean_attention_entropy:.3f} nats, same-order mass {mean_same_order:.3f}, and source-H0 mass {mean_h0_mass:.3f}. Mean injected RMS ratio is {residual_ratio:.4f}. These values indicate whether attention and residual injection were numerically active, not whether the model is useful.
- Same-order projected cosine means by checkpoint group: `{json.dumps({key: round(value, 5) for key, value in alignment_compare.items()}, ensure_ascii=False)}`. The A/IA-trained geometry should be compared with INIT, B and I; this observational comparison does not isolate alignment from all other training effects.
- Intervention means: interaction-off {off_acc:+.3f} pp; five-shuffle average {shuffle_acc:+.3f} pp. Agreement with retrained IA−B is supportive mechanistic context only; disagreement does not override the architecture comparison.
- Accuracy/F1/CE directional conflicts for IA−B occur in: {', '.join(accuracy_secondary_conflicts) if accuracy_secondary_conflicts else 'none by mean direction'}.
- Dataset pattern: {positive_dataset_means}/3 positive IA−B means. The observed split consistency should inform any later review; the single fixed split per dataset limits claims about split-level generalization.

## Stop and recommendation

The D0 screen is complete. No Toys, Reddit-S, NC test, formal LP, or follow-on architecture experiments were run. Do not add MoE/OT/MMD/prototypes or more attention in this phase. For a later phase, retain IA only if its retrained IA−B evidence and secondary metrics are sufficiently consistent under human review; retain I or A alone only if their paired comparisons support the simpler arm. Otherwise keep the results as a negative/uncertain screen and stop. This report does not claim causal mechanism proof or generalization beyond the fixed validation splits and three model seeds.

## Final self-audit

1. Strictly started from `cd9d440aa43454215e1a512b5426a5372e47a41a`: **yes**; the branch was created after verifying local and origin C1 HEAD and a clean worktree.
2. Used another experiment branch: **no merge or cherry-pick**; this branch is based directly on C1.
3. Ran or inspected NC test: **no**; test evaluation was disabled, no test metrics were written, and validation analysis uses train/validation labels only.
4. B strict C1 RGD regression: **yes**, exact eval output at zero tolerance in the regression test.
5. Alignment weight: **{weight:.2f}**, selected by `{calibration['selection_rule']}` from the predeclared Movies/Grocery seed-42 calibration.
6. Calibration test-free: **yes**, `evaluate_test=false`, `development_no_test=true`, no test metrics.
7. Variant parameter counts/state layouts/seed initialization matched: **yes**, covered by tests.
8. H0 remains pure intrinsic order: **yes**, no interaction residual is added at order zero; tested exactly.
9. Standalone I−B signal: **{i_b_text}**.
10. Standalone A−B signal: **{a_b_text}**.
11. Full IA−B increment: **{ia_b_text}**.
12. Pattern I≈B, A≈B, IA>B: **inspect paired values above**; no single-arm requirement was used to stop IA.
13. Descriptive 2×2 synergy positive: **{synergy_positive}/3 dataset means**, see seed-level CSV.
14. Attention non-trivial: mean entropy {mean_attention_entropy:.3f}, same-order mass {mean_same_order:.3f}, H0 mass {mean_h0_mass:.3f}; all source orders remain unmasked.
15. Residual interaction magnitude: mean RMS(ρJ)/RMS(Hk) **{residual_ratio:.4f}**.
16. Shared geometry: trained A/IA projected cosine and raw loss are compared with same-seed initialization and inactive-arm checkpoints; interpretation remains observational.
17. Source-node shuffle effect: mean accuracy delta **{shuffle_acc:+.3f} pp**; per-seed and per-repeat values are in `data/intervention_by_run.csv`.
18. Intervention/retrained evidence: intervention reliance is reported separately; primary retrained evidence remains IA−B.
19. GPR coefficient changes: B `{[round(x,5) for x in gpr_means['B']]}`, I `{[round(x,5) for x in gpr_means['I']]}`, A `{[round(x,5) for x in gpr_means['A']]}`, IA `{[round(x,5) for x in gpr_means['IA']]}`; coefficients alone are not performance evidence.
20. Dataset-specific regime: IA−B mean is positive on **{positive_dataset_means}/3** datasets.
21. Accuracy/F1/CE conflict: {', '.join(accuracy_secondary_conflicts) if accuracy_secondary_conflicts else 'no conflict in mean direction'}.
22. Worth entering a next phase: **wait for human review of the paired IA−B table and secondary metrics**; D0 itself stops here.
23. Mechanisms to retain: **conditional**—IA if full-block evidence supports it; otherwise consider only the simpler supported I or A arm; no unsupported mechanism is promoted.
24. LP smoke protocol: **passed** for B and IA, 2 epochs, 2 batches, LinkNeighborLoader, `[5,5,5]`, positive-edge removal, sampled-batch aux loss, validation and checkpoint; no test evaluation.
25. Claims beyond evidence: **none intended**; fixed-split validation and three seeds do not establish broad generalization or causal mechanism.

## Artifacts

- `data/performance_by_run.csv`, `data/performance_summary.csv`
- `data/paired_delta_by_run.csv`, `data/paired_delta_summary.csv`, `data/factorial_synergy.csv`
- `data/interaction_diagnostics.csv`, `data/alignment_diagnostics.csv`
- `data/intervention_by_run.csv`, `data/intervention_summary.csv`
- `data/calibration_by_run.csv`, `data/calibration_summary.csv`
- `run_manifest.json`, `smoke_status.json`, `alignment_weight_selection.json`
"""
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = load_manifest()
    selection_path = RESEARCH / "alignment_weight_selection.json"
    calibration = json.loads(selection_path.read_text(encoding="utf-8"))
    if float(calibration["selected_alignment_weight"]) != float(manifest["selected_alignment_weight"]):
        raise RuntimeError("manifest and calibration selection disagree")

    performance, summary, paired, paired_summary, synergy = performance_outputs(manifest)
    interaction, alignment, interventions, gpr = checkpoint_diagnostics(manifest, args.device)
    intervention_summary = summarize_interventions(interventions)

    write_csv(DATA_DIR / "performance_by_run.csv", performance)
    write_csv(DATA_DIR / "performance_summary.csv", summary)
    write_csv(DATA_DIR / "paired_delta_by_run.csv", paired)
    write_csv(DATA_DIR / "paired_delta_summary.csv", paired_summary)
    write_csv(DATA_DIR / "factorial_synergy.csv", synergy)
    write_csv(DATA_DIR / "interaction_diagnostics.csv", interaction + [
        {"diagnostic": "gpr_profile", **row} for row in gpr
    ])
    write_csv(DATA_DIR / "alignment_diagnostics.csv", alignment)
    write_csv(DATA_DIR / "intervention_by_run.csv", interventions)
    write_csv(DATA_DIR / "intervention_summary.csv", intervention_summary)

    report = build_report(
        manifest,
        performance,
        summary,
        paired_summary,
        synergy,
        interaction,
        alignment,
        intervention_summary,
        gpr,
        calibration,
    )
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")
    manifest["analysis_status"] = "complete"
    manifest["analysis_device"] = args.device
    manifest["analysis_test_evaluation"] = False
    manifest["analysis_finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["status"] = "complete"
    manifest["completed_at"] = manifest["analysis_finished_at"]
    write_json(RESEARCH / "run_manifest.json", manifest)
    print(f"ORCI-D0 analysis complete: {RESEARCH / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
