#!/usr/bin/env python3
"""Validation-only analysis for the CARE-MAG A0.3 placement audit."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "care_a03_adaptation_placement_audit"
BASE_SHA = "a44a556517112f15a8550f81c4b916906756d77f"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
FACTORS = {
    "G0": ("global", "off"),
    "GS": ("global", "static"),
    "GC": ("global", "context"),
    "N0": ("node", "off"),
    "NS": ("node", "static"),
    "NC": ("node", "context"),
}
DIRECT_COMPARISONS = (
    ("A_trajectory", "N0 - G0", "N0", "G0"),
    ("A_trajectory", "NS - GS", "NS", "GS"),
    ("A_trajectory", "NC - GC", "NC", "GC"),
    ("B_generic_adapter", "GS - G0", "GS", "G0"),
    ("B_generic_adapter", "NS - N0", "NS", "N0"),
    ("C_context_beyond_static", "GC - GS", "GC", "GS"),
    ("C_context_beyond_static", "NC - NS", "NC", "NS"),
    ("D_direct_context", "GC - G0", "GC", "G0"),
    ("D_direct_context", "NC - N0", "NC", "N0"),
)
ALL_COMPARISONS = DIRECT_COMPARISONS + (
    ("E_factorial_interaction", "interaction_context=(NC-NS)-(GC-GS)", None, None),
    ("E_factorial_interaction", "interaction_adapter=(NC-N0)-(GC-G0)", None, None),
)
METRIC_FIELDS = (
    ("val_accuracy", "val_accuracy_fraction"),
    ("val_macro_f1", "val_macro_f1_fraction"),
    ("val_ce", "val_ce"),
)
SELECTED_DIAGNOSTICS = ("GC", "N0", "NS", "NC")
TRAJECTORY_VARIANTS = ("N0", "NS", "NC")
ADAPTER_VARIANTS = ("GC", "NC")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def load_performance(manifest: dict) -> tuple[list[dict], dict]:
    if manifest.get("base_sha") != BASE_SHA:
        raise RuntimeError(f"unexpected source SHA in manifest: {manifest.get('base_sha')}")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"formal manifest is not complete: {manifest.get('status')}")
    expected = len(DATASETS) * len(FACTORS)
    completed = [item for item in manifest.get("runs", []) if item.get("status") == "complete"]
    if len(completed) != expected:
        raise RuntimeError(f"expected {expected} completed configurations, found {len(completed)}")

    performance: list[dict] = []
    by_run: dict[tuple[str, str, int], dict] = {}
    seen = set()
    for run in completed:
        key = (run["dataset"], run["variant"])
        if key in seen or run["dataset"] not in DATASETS or run["variant"] not in FACTORS:
            raise RuntimeError(f"unexpected or duplicate formal run: {key}")
        seen.add(key)
        if run.get("evaluate_test") is not False or run.get("development_no_test") is not True:
            raise RuntimeError(f"test guard metadata is invalid for {key}")
        payload = read_json(Path(run["run_metrics_path"]))
        if payload.get("development_no_test") is not True:
            raise RuntimeError(f"development_no_test missing in {run['run_metrics_path']}")
        if payload.get("run_seeds") != list(SEEDS):
            raise RuntimeError(f"unexpected paired seeds in {run['run_metrics_path']}")
        if len(payload.get("runs", [])) != len(SEEDS):
            raise RuntimeError(f"expected three seeds in {run['run_metrics_path']}")
        for item in payload["runs"]:
            seed = int(item["seed"])
            if seed not in SEEDS:
                raise RuntimeError(f"unexpected seed {seed}")
            metrics = item["metrics"]
            test_keys = [name for name in metrics if name.startswith("test_")]
            if test_keys:
                raise RuntimeError(f"test metrics found in development output: {test_keys}")
            needed = {"val_acc", "val_macro_f1", "val_ce"}
            if not needed.issubset(metrics):
                raise RuntimeError(f"validation metrics missing for {key}, seed={seed}")
            metadata = item.get("metadata", {})
            if metadata.get("development_no_test") is not True:
                raise RuntimeError(f"per-run no-test metadata missing for {key}, seed={seed}")
            if metadata.get("best_epoch") is None:
                raise RuntimeError(f"validation-selected checkpoint epoch missing for {key}, seed={seed}")
            row = {
                "dataset": key[0],
                "variant": key[1],
                "trajectory_mode": FACTORS[key[1]][0],
                "adapter_mode": FACTORS[key[1]][1],
                "seed": seed,
                "val_accuracy": float(metrics["val_acc"]),
                "val_macro_f1": float(metrics["val_macro_f1"]),
                "val_ce": float(metrics["val_ce"]),
                "best_epoch": int(metadata["best_epoch"]),
                "model_parameters": int(metadata["model_parameters"]),
                "development_no_test": True,
                "run_metrics_path": run["run_metrics_path"],
            }
            performance.append(row)
            by_run[(key[0], key[1], seed)] = row

    if len(performance) != 54 or len(by_run) != 54:
        raise RuntimeError(f"expected 54 paired run rows, found {len(performance)}")
    for dataset in DATASETS:
        parameter_counts = {
            int(row["model_parameters"])
            for row in performance
            if row["dataset"] == dataset
        }
        if len(parameter_counts) != 1:
            raise RuntimeError(
                f"factorial cells have different parameter totals within {dataset}: "
                f"{sorted(parameter_counts)}"
            )
    expected_keys = {(d, v) for d in DATASETS for v in FACTORS}
    if seen != expected_keys:
        raise RuntimeError("formal manifest does not contain every dataset × factorial cell")
    return performance, by_run


def summarize_performance(performance: list[dict]) -> list[dict]:
    result = []
    for dataset in DATASETS:
        for variant in FACTORS:
            rows = [r for r in performance if r["dataset"] == dataset and r["variant"] == variant]
            entry = {
                "dataset": dataset,
                "variant": variant,
                "trajectory_mode": FACTORS[variant][0],
                "adapter_mode": FACTORS[variant][1],
                "n": len(rows),
            }
            for field in ("val_accuracy", "val_macro_f1", "val_ce", "best_epoch"):
                entry[f"{field}_mean"], entry[f"{field}_population_sd"] = mean_sd(
                    [float(r[field]) for r in rows]
                )
            result.append(entry)
    return result


def metric_delta(new: dict, old: dict) -> dict:
    return {
        "val_accuracy_pp": (new["val_accuracy"] - old["val_accuracy"]) * 100.0,
        "val_macro_f1_pp": (new["val_macro_f1"] - old["val_macro_f1"]) * 100.0,
        "val_ce_delta": new["val_ce"] - old["val_ce"],
    }


def paired_deltas(by_run: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    for dataset in DATASETS:
        for group, name, newer, baseline in DIRECT_COMPARISONS:
            for seed in SEEDS:
                delta = metric_delta(
                    by_run[(dataset, newer, seed)], by_run[(dataset, baseline, seed)]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "comparison_group": group,
                        "comparison": name,
                        **delta,
                    }
                )
        for seed in SEEDS:
            cells = {name: by_run[(dataset, name, seed)] for name in FACTORS}
            context_delta = metric_delta(cells["NC"], cells["NS"])
            context_global_delta = metric_delta(cells["GC"], cells["GS"])
            adapter_delta = metric_delta(cells["NC"], cells["N0"])
            adapter_global_delta = metric_delta(cells["GC"], cells["G0"])
            interactions = {
                key: context_delta[key] - context_global_delta[key]
                for key in context_delta
            }
            adapter_interactions = {
                key: adapter_delta[key] - adapter_global_delta[key]
                for key in adapter_delta
            }
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "comparison_group": "E_factorial_interaction",
                    "comparison": "interaction_context=(NC-NS)-(GC-GS)",
                    **interactions,
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "comparison_group": "E_factorial_interaction",
                    "comparison": "interaction_adapter=(NC-N0)-(GC-G0)",
                    **adapter_interactions,
                }
            )

    summary: list[dict] = []
    for dataset in DATASETS:
        for _, name, _, _ in ALL_COMPARISONS:
            group = [r for r in rows if r["dataset"] == dataset and r["comparison"] == name]
            if len(group) != len(SEEDS):
                raise RuntimeError(f"unpaired or missing contrast: {dataset} {name}")
            entry = {
                "dataset": dataset,
                "comparison_group": group[0]["comparison_group"],
                "comparison": name,
                "n_paired": len(group),
            }
            for metric in ("val_accuracy_pp", "val_macro_f1_pp", "val_ce_delta"):
                values = [float(row[metric]) for row in group]
                entry[f"{metric}_mean"], entry[f"{metric}_population_sd"] = mean_sd(values)
                entry[f"{metric}_positive_count"] = sum(v > 1e-12 for v in values)
                entry[f"{metric}_negative_count"] = sum(v < -1e-12 for v in values)
                entry[f"{metric}_tie_count"] = sum(abs(v) <= 1e-12 for v in values)
            summary.append(entry)

    factorial_rows = []
    for name in (
        "interaction_context=(NC-NS)-(GC-GS)",
        "interaction_adapter=(NC-N0)-(GC-G0)",
    ):
        for dataset in (*DATASETS, "ALL_DATASETS"):
            group = [
                row
                for row in rows
                if row["comparison"] == name
                and (dataset == "ALL_DATASETS" or row["dataset"] == dataset)
            ]
            entry = {
                "dataset": dataset,
                "interaction": name,
                "n_paired": len(group),
            }
            for metric in ("val_accuracy_pp", "val_macro_f1_pp", "val_ce_delta"):
                values = [float(row[metric]) for row in group]
                entry[f"{metric}_mean"], entry[f"{metric}_population_sd"] = mean_sd(values)
                entry[f"{metric}_positive_count"] = sum(v > 1e-12 for v in values)
                entry[f"{metric}_negative_count"] = sum(v < -1e-12 for v in values)
                entry[f"{metric}_tie_count"] = sum(abs(v) <= 1e-12 for v in values)
            factorial_rows.append(entry)
    return rows, summary, factorial_rows


def hydra_config(dataset: str, trajectory: str, adapter: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=care_mag_v1",
                "seed=42",
                "num_runs=1",
                "task.evaluate_test=false",
                "task.development_no_test=true",
            ],
        )
    cfg.model.trajectory_mode = trajectory
    cfg.model.adapter_mode = adapter
    return cfg


def evaluate_embedding(head, z: torch.Tensor, data, eval_labels, device) -> dict:
    from src.tasks.nc import _evaluate_split

    result = _evaluate_split(
        head,
        z.detach().cpu(),
        data.y,
        data.val_idx,
        device,
        batch_size=4096,
        eval_labels=eval_labels,
    )
    return {
        "val_accuracy": float(result["acc"]),
        "val_macro_f1": float(result["macro_f1"]),
        "val_ce": float(result["ce"]),
    }


def metric_delta_from_normal(metrics: dict, normal: dict) -> dict:
    return {
        "delta_val_accuracy_pp_vs_normal": (metrics["val_accuracy"] - normal["val_accuracy"]) * 100.0,
        "delta_val_macro_f1_pp_vs_normal": (metrics["val_macro_f1"] - normal["val_macro_f1"]) * 100.0,
        "delta_val_ce_vs_normal": metrics["val_ce"] - normal["val_ce"],
    }


def mechanism_record(dataset: str, variant: str, seed: int, modality: str, info: dict, details: dict, checkpoint_path: Path, normal_metrics: dict, formal_metrics: dict, normal_match: dict) -> dict:
    base = info[modality]
    coeff = details["coefficients"][modality]
    record = {
        "dataset": dataset,
        "variant": variant,
        "trajectory_mode": FACTORS[variant][0],
        "adapter_mode": FACTORS[variant][1],
        "seed": seed,
        "modality": modality,
        "lambda": float(base["lambda"].item()),
        "gamma": float(base["gamma"].item()),
        "effective_hop_mean": float(base["effective_hop_mean"].item()),
        "effective_hop_std": float(base["effective_hop_std"].item()),
        "alpha_entropy_mean": float(base["alpha_entropy_mean"].item()),
        "alpha_entropy_std": float(base["alpha_entropy_std"].item()),
        "dominant_hop_k1_frequency": float(base["dominant_hop_frequency"][0].item()),
        "dominant_hop_k2_frequency": float(base["dominant_hop_frequency"][1].item()),
        "dominant_hop_k3_frequency": float(base["dominant_hop_frequency"][2].item()),
        "response_rms": float(base["response_rms"].item()),
        "delta_rms": float(base["delta_rms"].item()),
        "delta_response_rms_ratio": float(base["delta_response_rms_ratio"].item()),
        "gamma_delta_response_rms_ratio": float(base["gamma_delta_response_rms_ratio"].item()),
        "lambda_gamma_delta_response_rms_ratio": float(base["lambda"].item()) * float(base["gamma_delta_response_rms_ratio"].item()),
        "coefficient_node_std": float(base["coefficient_node_std"].item()),
        "coefficient_mean_by_rank": json.dumps(base["coefficient_mean"].cpu().tolist(), separators=(",", ":")),
        "coefficient_std_by_rank": json.dumps(base["coefficient_std"].cpu().tolist(), separators=(",", ":")),
        "delta_response_cosine_mean": float(base["delta_response_cosine_mean"].item()),
        "delta_response_cosine_std": float(base["delta_response_cosine_std"].item()),
        "normal_val_accuracy": normal_metrics["val_accuracy"],
        "normal_val_macro_f1": normal_metrics["val_macro_f1"],
        "normal_val_ce": normal_metrics["val_ce"],
        "formal_val_accuracy": formal_metrics["val_accuracy"],
        "formal_val_macro_f1": formal_metrics["val_macro_f1"],
        "formal_val_ce": formal_metrics["val_ce"],
        "normal_reproduces_formal_accuracy": normal_match["val_accuracy"],
        "normal_reproduces_formal_macro_f1": normal_match["val_macro_f1"],
        "normal_reproduces_formal_ce": normal_match["val_ce"],
        "checkpoint_path": str(checkpoint_path),
    }
    for hop in (1, 2, 3):
        record[f"alpha_k{hop}_mean"] = float(base["alpha_mean"][hop - 1].item())
        record[f"alpha_k{hop}_node_std"] = float(base["alpha_std"][hop - 1].item())
    return record


def run_mechanism_analysis(manifest: dict, by_run: dict, device: torch.device) -> tuple[list[dict], list[dict], list[dict]]:
    from src.data import load_mag_data
    from src.models.care_mag_v1 import Model
    from src.tasks.nc import _resolve_nc_eval_labels

    torch.set_num_threads(8)
    diagnostics: list[dict] = []
    trajectory_rows: list[dict] = []
    adapter_rows: list[dict] = []
    run_lookup = {(item["dataset"], item["variant"]): item for item in manifest["runs"]}

    for dataset_index, dataset in enumerate(DATASETS):
        data_cfg = hydra_config(dataset, "node", "context")
        data = load_mag_data(data_cfg, "nc", 42)
        eval_labels = _resolve_nc_eval_labels(data, development_no_test=True)
        data_info = {
            "input_dim": data.input_dim,
            "num_nodes": data.num_nodes,
            "num_classes": data.num_classes,
            "text_dim": int(data.x_t.shape[1]),
            "visual_dim": int(data.x_i.shape[1]),
        }
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        for variant in SELECTED_DIAGNOSTICS:
            run_entry = run_lookup[(dataset, variant)]
            checkpoint_base = Path(run_entry["checkpoint_path"])
            trajectory_mode, adapter_mode = FACTORS[variant]
            cfg = hydra_config(dataset, trajectory_mode, adapter_mode)
            for run_id, seed in enumerate(SEEDS, start=1):
                checkpoint_path = checkpoint_base.with_name(
                    f"{checkpoint_base.stem}_run{run_id}{checkpoint_base.suffix}"
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if int(checkpoint.get("seed", -1)) != seed:
                    raise RuntimeError(
                        f"checkpoint seed mismatch at {checkpoint_path}: expected {seed}, "
                        f"got {checkpoint.get('seed')}"
                    )
                model = Model(cfg, data_info).to(device)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.eval()
                head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                head.load_state_dict(checkpoint["head_state"], strict=True)
                head.eval()

                with torch.no_grad():
                    z, _, _, _, info = model(x, edge_index, return_details=True)
                    normal = evaluate_embedding(head, z, data, eval_labels, device)
                    expected = by_run[(dataset, variant, seed)]
                    normal_match = {
                        "val_accuracy": abs(normal["val_accuracy"] - expected["val_accuracy"]) <= 1e-10,
                        "val_macro_f1": abs(normal["val_macro_f1"] - expected["val_macro_f1"]) <= 1e-10,
                        "val_ce": abs(normal["val_ce"] - expected["val_ce"]) <= 1e-6,
                    }
                    details = info["details"]
                    for modality in ("text", "visual"):
                        diagnostics.append(
                            mechanism_record(
                                dataset,
                                variant,
                                seed,
                                modality,
                                info,
                                details,
                                checkpoint_path,
                                normal,
                                expected,
                                normal_match,
                            )
                        )

                # The normal detailed forward retains per-node hop stacks;
                # release them before the repeated intervention passes.
                del z, info, details
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                trajectory_outputs: dict[tuple[str, int], dict] = {}
                if variant in TRAJECTORY_VARIANTS:
                    trajectory_outputs[("normal", 0)] = normal
                    for mode in ("traj_global_mean",):
                        with torch.no_grad():
                            z = model(x, edge_index, intervention=mode)[0]
                            trajectory_outputs[(mode, 0)] = evaluate_embedding(
                                head, z, data, eval_labels, device
                            )
                    for repeat in range(1, 6):
                        shuffle_seed = dataset_index * 1000000 + seed * 100 + repeat
                        with torch.no_grad():
                            z = model(
                                x,
                                edge_index,
                                intervention="traj_node_shuffle",
                                shuffle_seed=shuffle_seed,
                            )[0]
                            trajectory_outputs[("traj_node_shuffle", repeat)] = evaluate_embedding(
                                head, z, data, eval_labels, device
                            )
                    for (mode, repeat), metrics in trajectory_outputs.items():
                        trajectory_rows.append(
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "seed": seed,
                                "intervention": mode,
                                "repeat": repeat,
                                "shuffle_seed": (
                                    dataset_index * 1000000 + seed * 100 + repeat
                                    if mode == "traj_node_shuffle"
                                    else ""
                                ),
                                **metrics,
                                **metric_delta_from_normal(metrics, normal),
                                "checkpoint_path": str(checkpoint_path),
                            }
                        )

                if variant in ADAPTER_VARIANTS:
                    adapter_outputs: dict[tuple[str, int], dict] = {("normal", 0): normal}
                    for mode in ("adapter_off", "coeff_global_mean"):
                        with torch.no_grad():
                            z = model(x, edge_index, intervention=mode)[0]
                            adapter_outputs[(mode, 0)] = evaluate_embedding(
                                head, z, data, eval_labels, device
                            )
                    for repeat in range(1, 6):
                        shuffle_seed = dataset_index * 2000000 + seed * 100 + repeat
                        with torch.no_grad():
                            z = model(
                                x,
                                edge_index,
                                intervention="coeff_node_shuffle",
                                shuffle_seed=shuffle_seed,
                            )[0]
                            adapter_outputs[("coeff_node_shuffle", repeat)] = evaluate_embedding(
                                head, z, data, eval_labels, device
                            )
                    for (mode, repeat), metrics in adapter_outputs.items():
                        adapter_rows.append(
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "seed": seed,
                                "intervention": mode,
                                "repeat": repeat,
                                "shuffle_seed": (
                                    dataset_index * 2000000 + seed * 100 + repeat
                                    if mode == "coeff_node_shuffle"
                                    else ""
                                ),
                                **metrics,
                                **metric_delta_from_normal(metrics, normal),
                                "checkpoint_path": str(checkpoint_path),
                            }
                        )

                del z, model, head, checkpoint
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del data, x, edge_index
    return diagnostics, trajectory_rows, adapter_rows


def summarize_interventions(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(r["dataset"], r["variant"], r["intervention"]) for r in rows})
    for dataset, variant, intervention in keys:
        group = [
            row for row in rows
            if (row["dataset"], row["variant"], row["intervention"])
            == (dataset, variant, intervention)
        ]
        entry = {
            "dataset": dataset,
            "variant": variant,
            "trajectory_mode": FACTORS[variant][0],
            "adapter_mode": FACTORS[variant][1],
            "intervention": intervention,
            "n": len(group),
        }
        for metric in (
            "val_accuracy",
            "val_macro_f1",
            "val_ce",
            "delta_val_accuracy_pp_vs_normal",
            "delta_val_macro_f1_pp_vs_normal",
            "delta_val_ce_vs_normal",
        ):
            values = [float(row[metric]) for row in group]
            entry[f"{metric}_mean"], entry[f"{metric}_population_sd"] = mean_sd(values)
        result.append(entry)
    return result


def comparison_status(rows: list[dict], comparison: str, dataset: str, margin_pp: float = 0.5) -> str:
    row = next(r for r in rows if r["dataset"] == dataset and r["comparison"] == comparison)
    mean = float(row["val_accuracy_pp_mean"])
    pos = int(row["val_accuracy_pp_positive_count"])
    neg = int(row["val_accuracy_pp_negative_count"])
    if mean >= margin_pp and pos >= 2:
        return "positive"
    if mean <= -margin_pp and neg >= 2:
        return "negative"
    if abs(mean) < margin_pp:
        return "approximately_zero"
    return "mixed_or_uncertain"


def interpret_results(paired_summary: list[dict], factorial_rows: list[dict]) -> tuple[str, list[str], dict]:
    statuses = {
        (dataset, row["comparison"]): comparison_status(paired_summary, row["comparison"], dataset)
        for dataset in DATASETS
        for row in paired_summary
        if row["dataset"] == dataset
    }
    trajectory_pos = sum(statuses[(d, "N0 - G0")] == "positive" for d in DATASETS)
    trajectory_zero = sum(statuses[(d, "N0 - G0")] == "approximately_zero" for d in DATASETS)
    cg_pos = sum(statuses[(d, "GC - GS")] == "positive" for d in DATASETS)
    cn_pos = sum(statuses[(d, "NC - NS")] == "positive" for d in DATASETS)
    cg_zero = sum(statuses[(d, "GC - GS")] == "approximately_zero" for d in DATASETS)
    cn_zero = sum(statuses[(d, "NC - NS")] == "approximately_zero" for d in DATASETS)
    generic_adapter_pos = sum(
        statuses[(d, comparison)] == "positive"
        for d in DATASETS
        for comparison in ("GS - G0", "NS - N0")
    )
    context_interaction = next(
        row for row in factorial_rows
        if row["dataset"] == "ALL_DATASETS"
        and row["interaction"] == "interaction_context=(NC-NS)-(GC-GS)"
    )
    adapter_interaction = next(
        row for row in factorial_rows
        if row["dataset"] == "ALL_DATASETS"
        and row["interaction"] == "interaction_adapter=(NC-N0)-(GC-G0)"
    )
    interaction_neutral_or_positive = (
        context_interaction["val_accuracy_pp_mean"] >= -0.5
        and adapter_interaction["val_accuracy_pp_mean"] >= -0.5
    )

    if trajectory_pos >= 2 and cg_zero >= 2 and cn_zero >= 2:
        case = "Case 1"
        label = (
            "Node-conditioned trajectory is the leading adaptation placement in this screen; "
            "post-trajectory context correction has no clear incremental validation-accuracy value."
        )
    elif trajectory_pos >= 2 and cg_pos >= 2 and cn_zero >= 2:
        case = "Case 2"
        label = (
            "The results are consistent with functional redundancy: context conditioning helps "
            "with global trajectories but adds little after node-conditioned trajectory readout."
        )
    elif cg_pos >= 2 and cn_pos >= 2 and interaction_neutral_or_positive:
        case = "Case 3"
        label = (
            "Trajectory and feature adaptation may be complementary; context adapters improve "
            "over static adapters in both trajectory modes, without a clearly negative pooled interaction."
        )
    elif trajectory_zero >= 2 and cg_pos >= 2:
        case = "Case 4"
        label = (
            "Feature-level context conditioning appears more valuable than recipient-specific "
            "trajectory preference under this screen."
        )
    elif trajectory_zero >= 2 and cg_pos == 0 and cn_pos == 0 and generic_adapter_pos == 0:
        case = "Case 5"
        label = (
            "Neither trajectory adaptation nor context-over-static correction shows stable "
            "incremental value; the current evidence does not support moving to MoE."
        )
    else:
        case = "Mixed result"
        label = (
            "The primary validation-accuracy effects vary across datasets or fail the prespecified "
            "descriptive consistency rule; retain the per-dataset evidence without a uniform claim."
        )

    metric_tradeoffs = []
    for row in paired_summary:
        if row["dataset"] == "ALL_DATASETS":
            continue
        acc = float(row["val_accuracy_pp_mean"])
        f1 = float(row["val_macro_f1_pp_mean"])
        ce = float(row["val_ce_delta_mean"])
        secondary_opposition = (acc > 0 and (f1 < 0 or ce > 0)) or (
            acc < 0 and (f1 > 0 or ce < 0)
        )
        if secondary_opposition:
            metric_tradeoffs.append(
                f"{row['dataset']} {row['comparison']} (Acc {acc:+.3f} pp, "
                f"Macro-F1 {f1:+.3f} pp, CE {ce:+.5f})"
            )
    dataset_dependent = any(
        len({statuses[(d, comp)] for d in DATASETS}) > 1
        for comp in ("N0 - G0", "GC - GS", "NC - NS")
    )
    counts = {
        "case": case,
        "trajectory_positive_datasets": trajectory_pos,
        "trajectory_approximately_zero_datasets": trajectory_zero,
        "global_context_positive_datasets": cg_pos,
        "node_context_positive_datasets": cn_pos,
        "global_context_approximately_zero_datasets": cg_zero,
        "node_context_approximately_zero_datasets": cn_zero,
        "dataset_dependent": dataset_dependent,
    }
    notes = [
        f"Node-trajectory value N0−G0: {trajectory_pos}/3 datasets meet the primary positive rule; {trajectory_zero}/3 are within ±0.5 accuracy points.",
        f"Context beyond static: GC−GS is positive in {cg_pos}/3 datasets; NC−NS is positive in {cn_pos}/3.",
        f"Pooled descriptive context interaction is {context_interaction['val_accuracy_pp_mean']:+.3f} accuracy points; pooled adapter interaction is {adapter_interaction['val_accuracy_pp_mean']:+.3f} points. These are paired descriptive summaries, not hypothesis tests.",
        f"Classification: {case}. {label}",
    ]
    return case, notes, counts


def format_performance_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Dataset | Cell | Val Accuracy (%) | Macro-F1 (%) | CE | Best epoch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} ({row['trajectory_mode']}/{row['adapter_mode']}) | "
            f"{100*row['val_accuracy_mean']:.2f} ± {100*row['val_accuracy_population_sd']:.2f} | "
            f"{100*row['val_macro_f1_mean']:.2f} ± {100*row['val_macro_f1_population_sd']:.2f} | "
            f"{row['val_ce_mean']:.4f} ± {row['val_ce_population_sd']:.4f} | "
            f"{row['best_epoch_mean']:.1f} ± {row['best_epoch_population_sd']:.1f} |"
        )
    return lines


def format_comparison_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Dataset | Comparison | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | Acc paired signs (+/−/=) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    ordered_names = [name for _, name, _, _ in ALL_COMPARISONS]
    for dataset in DATASETS:
        for name in ordered_names:
            row = next(r for r in rows if r["dataset"] == dataset and r["comparison"] == name)
            lines.append(
                f"| {dataset} | {name} | "
                f"{row['val_accuracy_pp_mean']:+.3f} ± {row['val_accuracy_pp_population_sd']:.3f} | "
                f"{row['val_macro_f1_pp_mean']:+.3f} ± {row['val_macro_f1_pp_population_sd']:.3f} | "
                f"{row['val_ce_delta_mean']:+.5f} ± {row['val_ce_delta_population_sd']:.5f} | "
                f"{row['val_accuracy_pp_positive_count']}/{row['val_accuracy_pp_negative_count']}/{row['val_accuracy_pp_tie_count']} |"
            )
    return lines


def format_intervention_table(rows: list[dict], metric_prefix: str = "") -> list[str]:
    lines = [
        "| Dataset | Cell | Intervention | Accuracy (%) | Macro-F1 (%) | CE | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for row in rows:
            if row["dataset"] != dataset:
                continue
            if metric_prefix and not row["variant"] in ("GC", "NC"):
                continue
            lines.append(
                f"| {dataset} | {row['variant']} | {row['intervention']} (n={row['n']}) | "
                f"{100*row['val_accuracy_mean']:.2f} ± {100*row['val_accuracy_population_sd']:.2f} | "
                f"{100*row['val_macro_f1_mean']:.2f} ± {100*row['val_macro_f1_population_sd']:.2f} | "
                f"{row['val_ce_mean']:.4f} ± {row['val_ce_population_sd']:.4f} | "
                f"{row['delta_val_accuracy_pp_vs_normal_mean']:+.3f} | "
                f"{row['delta_val_macro_f1_pp_vs_normal_mean']:+.3f} | "
                f"{row['delta_val_ce_vs_normal_mean']:+.5f} |"
            )
    return lines


def write_design_audit(manifest: dict, smoke: dict) -> None:
    text = f"""# CARE-MAG A0.3 design and protocol audit

## Provenance

- Required parent SHA: `{BASE_SHA}`.
- Experiment branch: `exp/care_a03_adaptation_placement_audit`.
- The runner verified that the branch is descended from the required SHA before tests, smokes, and the campaign.
- The work used the authorized A0.2 source branch only; no other historical experiment branch was inspected or used.

## Question and design

The 2×3 factorial separates recipient-specific hop utilization from residual feature correction. Trajectory factors are `global` and `node`; adapter factors are `off`, `static`, and `context`. G0/GS/GC and N0/NS/NC share the v0 module construction, parameter layout, propagation, initialization and fusion. No trainable module was added. Global hop scores reuse each modality's v0 `Wp`, `Wr`, hop embeddings, and score vector on graph means, then broadcast the resulting hop probabilities to all nodes.

The node × adapter cells are regression mapped to the v0 structural, static and context variants with strict state loading and numerical equality checks.

## Frozen training protocol

- Datasets and split: Movies, Grocery and ele-fashion; model/training seeds 42, 43, 44; each dataset's configured NC split stays fixed.
- Task: full-graph NC, validation Accuracy checkpoint selection, AdamW, and the repository's unchanged learning rate, weight decay, stopping, clipping and evaluation configuration.
- `task.evaluate_test=false` and `task.development_no_test=true` for every formal run. The formal analyzer rejects any `test_*` metric key.
- 18 dataset × architecture configurations × 3 paired seeds = 54 runs. The launcher stops on the first failed configuration and can resume completed configurations.
- No NC test, formal LP campaign, hyperparameter adjustment or A3/MoE experiment is part of this stage.

## Required gates

- Full repository tests: {smoke.get('tests', {}).get('status', 'not recorded')} (exit {smoke.get('tests', {}).get('return_code', 'n/a')}).
- Movies, seed 42: six architecture cells, two epochs each; each must train, validate and save a checkpoint.
- LP is execution smoke only: sports-copurchase, GC and NC, two epochs and at most two sampled training batches, with test evaluation disabled. It must report `LinkNeighborLoader`, fanouts `[5,5,5]`, positive supervision edge removal, forward/backward, validation inference and a saved checkpoint.
- Formal NC campaign status: {manifest.get('status', 'not yet run')}.

## Analysis plan

Validation Accuracy is primary because it selects checkpoints. Paired effects are calculated within dataset and seed. Accuracy and Macro-F1 deltas are expressed in percentage points; CE deltas remain on the original scale. Means and population SDs are descriptive; no p-values are reported. Macro-F1 and CE are secondary consistency signals, and conflicts are labeled as trade-offs.

Direct retraining comparisons estimate architecture value. Trajectory interventions measure reliance of an already-selected checkpoint on node-to-hop correspondence. Adapter interventions analogously probe coefficient correspondence at GC/NC selected checkpoints. These evidence types remain separate. The descriptive “clear positive” rule is a mean Accuracy gain of at least 0.5 points with at least two of three paired seeds positive; values within ±0.5 points are called approximately zero. This rule is for readable conservative interpretation, not a significance threshold.

## No tuning / stop boundary

Model widths, graph processing, optimizer and all training settings remain frozen. Results are not used to modify the model or continue the campaign with different settings. After this audit, stop for human review; do not start A3 MoE, dynamic trust, LP campaign or NC test.
"""
    (RESEARCH / "design_audit.md").write_text(text, encoding="utf-8")


def make_report(manifest: dict, performance_summary: list[dict], paired_summary: list[dict], factorial_rows: list[dict], diagnostics: list[dict], trajectory_summary: list[dict], adapter_summary: list[dict], smoke: dict, interpretation: tuple[str, list[str], dict]) -> str:
    case, notes, counts = interpretation
    lines = [
        "# CARE-MAG A0.3 — Adaptation Placement Audit",
        "",
        f"- Base SHA: `{manifest['base_sha']}`",
        f"- Final campaign status: `{manifest['status']}`",
        "- Design: 2×3 factorial; full-graph NC; best checkpoints selected on validation Accuracy.",
        "- Test evaluation: disabled; analyzer verified validation-only metrics and rejected any `test_*` keys.",
        "- Reported seed summaries are paired descriptive results (mean ± population SD); no p-values.",
        "",
        "## Validation performance",
        "",
        "Accuracy and Macro-F1 are shown as percentages; CE remains on its original scale.",
        "",
        *format_performance_table(performance_summary),
        "",
        "## Paired retraining contrasts",
        "",
        "Each delta is first cell minus second cell, paired by dataset and seed. Accuracy and Macro-F1 are percentage-point differences; CE is unscaled. These rows estimate architecture value from separately retrained models.",
        "",
        *format_comparison_table(paired_summary),
        "",
        "## Factorial interactions",
        "",
        "The interaction rows are constructed separately for each dataset and seed, then summarized. Pooled `ALL_DATASETS` is a descriptive aggregation across the nine paired dataset-seed cells, not an inferential sample.",
        "",
        "| Dataset | Interaction | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | n paired |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in factorial_rows:
        lines.append(
            f"| {row['dataset']} | {row['interaction']} | "
            f"{row['val_accuracy_pp_mean']:+.3f} ± {row['val_accuracy_pp_population_sd']:.3f} | "
            f"{row['val_macro_f1_pp_mean']:+.3f} ± {row['val_macro_f1_pp_population_sd']:.3f} | "
            f"{row['val_ce_delta_mean']:+.5f} ± {row['val_ce_delta_population_sd']:.5f} | {row['n_paired']} |"
        )
    lines += [
        "",
        "## Selected-checkpoint trajectory interventions",
        "",
        "`traj_global_mean` replaces node-specific alpha rows by their mean; `traj_node_shuffle` permutes alpha rows with five deterministic seeds per checkpoint. The resulting deltas measure checkpoint reliance on recipient-to-hop correspondence. They are not retrained architecture effects.",
        "",
        *format_intervention_table(trajectory_summary),
        "",
        "## Selected-checkpoint adapter interventions",
        "",
        "For GC and NC checkpoints, `adapter_off` removes the residual correction; `coeff_global_mean` removes node variation in coefficients; `coeff_node_shuffle` permutes coefficient rows in five deterministic repeats. These are checkpoint reliance tests, separate from GC−GS and NC−NS retraining contrasts.",
        "",
        *format_intervention_table(adapter_summary, metric_prefix="adapter"),
        "",
        "## Mechanism diagnostics",
        "",
        "Each selected checkpoint is reported by modality. Alpha means and node standard deviations, effective hop, entropy, dominant-hop frequency, response RMS, correction/response RMS ratios, coefficient node variation, lambda, gamma and cosine are recorded in the CSV. The CSV also records whether a normal re-evaluation reproduces the saved formal validation metrics. No cross-modality rank disagreement is calculated because the modality adapter bases are not aligned.",
        "",
        "| Dataset | Cell | Modality | λ | γ | α k1/k2/k3 means | α node SD k1/k2/k3 | Effective hop | Entropy | Dominant hop frequency | R RMS | Δ/R | γΔ/R | Node coefficient SD | cos(Δ,R) |",
        "|---|---|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in diagnostics:
        grouped.setdefault((row["dataset"], row["variant"], row["modality"]), []).append(row)
    for dataset in DATASETS:
        for variant in SELECTED_DIAGNOSTICS:
            for modality in ("text", "visual"):
                group = grouped[(dataset, variant, modality)]
                def avg(field):
                    return statistics.mean(float(item[field]) for item in group)
                alpha_means = "/".join(f"{avg(f'alpha_k{i}_mean'):.3f}" for i in (1, 2, 3))
                alpha_stds = "/".join(f"{avg(f'alpha_k{i}_node_std'):.3f}" for i in (1, 2, 3))
                dominant = "/".join(f"{avg(f'dominant_hop_k{i}_frequency'):.2f}" for i in (1, 2, 3))
                lines.append(
                    f"| {dataset} | {variant} | {modality} | {avg('lambda'):.3f} | {avg('gamma'):.3f} | "
                    f"{alpha_means} | {alpha_stds} | {avg('effective_hop_mean'):.3f} ± {avg('effective_hop_std'):.3f} | "
                    f"{avg('alpha_entropy_mean'):.3f} ± {avg('alpha_entropy_std'):.3f} | {dominant} | "
                    f"{avg('response_rms'):.4f} | {avg('delta_response_rms_ratio'):.4f} | "
                    f"{avg('gamma_delta_response_rms_ratio'):.4f} | {avg('coefficient_node_std'):.5f} | "
                    f"{avg('delta_response_cosine_mean'):.4f} |"
                )
    lines += [
        "",
        "## Conservative interpretation",
        "",
        f"**{case}:** {notes[-1].split('. ', 1)[-1]}",
        "",
        *[f"- {note}" for note in notes[:-1]],
        f"- Primary-accuracy consistency rule: mean delta at least ±0.5 points and at least two of three paired seeds in that direction. Current summary: {counts}.",
    ]
    if case == "Case 5":
        lines.append(
            "- This screen leaves diffusion, prior retention and global hop utilization shared across cells, so it cannot attribute the strong baseline to any one of them. The result is compatible with those shared components carrying most of the performance, but does not establish that explanation. If a later stage is approved after review, dynamic structural trust or the backbone are reasonable next audit targets; neither is run here."
        )
    if any(
        (row["val_accuracy_pp_mean"] > 0 and (row["val_macro_f1_pp_mean"] < 0 or row["val_ce_delta_mean"] > 0))
        or (row["val_accuracy_pp_mean"] < 0 and (row["val_macro_f1_pp_mean"] > 0 or row["val_ce_delta_mean"] < 0))
        for row in paired_summary
        if row["dataset"] != "ALL_DATASETS"
    ):
        lines.append("- **Metric trade-off:** at least one paired contrast has Accuracy pointing one way while Macro-F1 or CE points the other way. Accuracy remains primary; the disagreement is reported rather than treated as support.")
    else:
        lines.append("- Macro-F1 and CE do not show a direction conflict with Accuracy at the summarized comparison level; they remain secondary signals.")

    lp_results = smoke.get("lp_runs", [])
    lp_pass = len(lp_results) == 2 and all(
        item.get("return_code") == 0
        and item.get("observed", {}).get("sampler_fanouts") == [5, 5, 5]
        and item.get("observed", {}).get("link_neighbor_loader") is True
        and item.get("observed", {}).get("positive_supervision_removal_observed") is True
        and item.get("observed", {}).get("validation_inference_completed") is True
        for item in lp_results
    )
    trajectory_lookup = {(r["dataset"], r["variant"], r["intervention"]): r for r in trajectory_summary}
    global_alpha_std_max = max(
        float(row[f"alpha_k{hop}_node_std"])
        for row in diagnostics
        if row["trajectory_mode"] == "global"
        for hop in (1, 2, 3)
    )
    global_alpha_invariant = global_alpha_std_max <= 1e-8
    normal_replay_matches = all(
        bool(row["normal_reproduces_formal_accuracy"])
        and bool(row["normal_reproduces_formal_macro_f1"])
        and bool(row["normal_reproduces_formal_ce"])
        for row in diagnostics
    )
    shuffle_trajectory = sum(
        trajectory_lookup[(d, v, "traj_node_shuffle")]["delta_val_accuracy_pp_vs_normal_mean"] < 0
        for d in DATASETS for v in TRAJECTORY_VARIANTS
    )
    self_audit = [
        f"1. **是否严格从 `{BASE_SHA}` 开始？** 是；fetch 后核验远端指定分支和本地 HEAD 一致，再从该 SHA 建分支。",
        "2. **是否访问过任何其他历史实验分支？** 没有检查、checkout、merge 或使用其他实验分支；只按协议执行了 `git fetch origin`，之后只核验指定 A0.2 分支 SHA 并读取该来源提交/current worktree。",
        "3. **是否运行/读取 test metrics？** 否；正式配置关闭 test，分析器拒绝 `test_*` 键，只评估 validation。",
        "4. **六 variants 参数量是否一致？** 是；测试逐格验证参数数量、state_dict 键和布局完全一致。",
        "5. **同 seed 初始化是否一致？** 是；测试逐张量验证六格同 seed bitwise identical。",
        "6. **N0/NS/NC 是否严格回归到 v0 对应 variants？** 是； strict-load 后 eval 输出逐元素相等。",
        f"7. **Global trajectory 是否真的 node-invariant？** {'是' if global_alpha_invariant else '否'}；selected global checkpoint 的最大 alpha node SD={global_alpha_std_max:.3g}（容差 1e-8）。",
        "8. **Node trajectory 是否真的 node-varying？** 是；测试验证 alpha 存在 recipient 间变化，formal alpha node SD 保存在 diagnostics。",
        f"9. **N0−G0 是否建立 node-specific trajectory 的 retrained value？** {counts['trajectory_positive_datasets']}/3 数据集满足预设 Accuracy 正向规则；详见 paired contrasts。",
        f"10. **trajectory global-mean/shuffle 是否显示 checkpoint correspondence reliance？** 五次 shuffle 后 Accuracy 平均下降的 node trajectory 单元为 {shuffle_trajectory}/9；详细 paired checkpoint deltas 在 intervention CSV。",
        f"11. **intervention 与 retrained comparison 是否一致？** 独立重算 normal validation 对正式 checkpoint 指标{'逐项复现' if normal_replay_matches else '存在差异，见 mechanism_diagnostics 的复算标志'}；两类结果仍作为不同证据列出，不把 checkpoint reliance 当作架构增益。",
        "12. **GS−G0 / NS−N0 是否显示 generic adapter capacity？** 两个配对对照均按 dataset/seed 汇总；见 B 类 comparison。",
        "13. **GC−GS / NC−NS 是否显示 context beyond static？** 分别按 dataset/seed 汇总并用 Accuracy 优先解释；见 C 类 comparison。",
        f"14. **factorial interaction 是正、负还是接近零？** context: pooled mean {next(r for r in factorial_rows if r['dataset']=='ALL_DATASETS' and r['interaction'].startswith('interaction_context'))['val_accuracy_pp_mean']:+.3f} pp; adapter: {next(r for r in factorial_rows if r['dataset']=='ALL_DATASETS' and r['interaction'].startswith('interaction_adapter'))['val_accuracy_pp_mean']:+.3f} pp。它们是描述性汇总，需结合各数据集行判断。",
        f"15. **是否出现 trajectory/adapter redundancy？** {case} 中的 context×trajectory paired pattern {'符合' if case == 'Case 2' else '未明确符合'}功能冗余模式。",
        f"16. **是否存在 dataset-specific regime？** {'是；关键 comparisons 的 Accuracy 分类随 dataset 变化。' if counts['dataset_dependent'] else '未按当前描述性规则检出关键 effect 分类随数据集变化。'}",
        f"17. **是否仍有明显 Accuracy vs Macro-F1/CE trade-off？** {'有；见报告中标记的对照。' if any('Metric trade-off' in line for line in lines) else '未检出摘要方向冲突。'}",
        f"18. **是否有充分证据进入 A3 MoE？** 否；本阶段仅形成供人工审查的 placement 证据，不自动进入 MoE。",
        f"19. **LP smoke 是否仍满足 sampled protocol？** {'是' if lp_pass else '否/未完整通过'}；只包含 sports-copurchase 的 GC/NC 两个短 sampled smoke，不含性能 campaign。",
        "20. **有没有超出证据的结论？** 结论仅针对三个固定 split、三次 paired model/training seeds 和 validation；不宣称 test 泛化或 LP 质量。",
    ]
    lines += ["", "## Self-audit", "", *self_audit, ""]
    lines += [
        "## Stop recommendation",
        "",
        "本阶段已完成后停止并等待人工审查。不要自行继续 A3 MoE、dynamic trust、cross-modal router cue、auxiliary supervision、expert diversity loss、OT、正式 LP 或 NC test。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = read_json(RESEARCH / "run_manifest.json")
    performance, by_run = load_performance(manifest)
    performance_summary = summarize_performance(performance)
    paired, paired_summary, factorial_rows = paired_deltas(by_run)

    data_dir = RESEARCH / "data"
    write_csv(data_dir / "performance_by_run.csv", performance)
    write_csv(data_dir / "performance_summary.csv", performance_summary)
    write_csv(data_dir / "paired_delta_by_run.csv", paired)
    write_csv(data_dir / "paired_delta_summary.csv", paired_summary)
    write_csv(data_dir / "factorial_effects.csv", factorial_rows)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the analyzer process")
    smoke = read_json(RESEARCH / "smoke_status.json")
    if smoke.get("status") != "passed":
        raise RuntimeError("smoke_status.json is not passed")
    diagnostics, trajectory_rows, adapter_rows = run_mechanism_analysis(
        manifest, by_run, device
    )
    trajectory_summary = summarize_interventions(trajectory_rows)
    adapter_summary = summarize_interventions(adapter_rows)
    write_csv(data_dir / "mechanism_diagnostics.csv", diagnostics)
    write_csv(data_dir / "trajectory_intervention_by_run.csv", trajectory_rows)
    write_csv(data_dir / "trajectory_intervention_summary.csv", trajectory_summary)
    write_csv(data_dir / "adapter_intervention_by_run.csv", adapter_rows)
    write_csv(data_dir / "adapter_intervention_summary.csv", adapter_summary)

    interpretation = interpret_results(paired_summary, factorial_rows)
    write_design_audit(manifest, smoke)
    report = make_report(
        manifest,
        performance_summary,
        paired_summary,
        factorial_rows,
        diagnostics,
        trajectory_summary,
        adapter_summary,
        smoke,
        interpretation,
    )
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")
    print(f"A0.3 analysis complete: {RESEARCH / 'report.md'}")
    print(interpretation[0], interpretation[1][-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
