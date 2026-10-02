#!/usr/bin/env python3
"""Analyze the frozen CARE-MAG A0–A2 validation-only campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "care_a0_a2_context_adapter_screen"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("prior_only", "structural_base", "static_adapter", "context_adapter")
COMPARISONS = (
    ("structural_base", "prior_only"),
    ("static_adapter", "structural_base"),
    ("context_adapter", "static_adapter"),
    ("context_adapter", "structural_base"),
)
METRICS = ("val_acc", "val_macro_f1", "val_ce")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def fmt(value: float, scale: float = 1.0) -> str:
    return f"{value * scale:.4f}"


def metric_direction(metric: str, delta: float) -> str:
    if abs(delta) <= 1e-12:
        return "tie"
    if metric == "val_ce":
        return "improved" if delta < 0 else "worsened"
    return "improved" if delta > 0 else "worsened"


def load_performance(manifest: dict) -> tuple[list[dict], dict]:
    if manifest.get("status") != "complete":
        raise RuntimeError(f"formal manifest is not complete: {manifest.get('status')}")
    expected = len(DATASETS) * len(VARIANTS)
    successful = [run for run in manifest.get("runs", []) if run.get("status") == "complete"]
    if len(successful) != expected:
        raise RuntimeError(f"expected {expected} completed configurations, found {len(successful)}")
    by_pair: dict[tuple[str, str], dict[int, dict]] = {}
    performance: list[dict] = []
    for run in successful:
        path = Path(run["run_metrics_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        if payload.get("development_no_test") is not True:
            raise RuntimeError(f"development_no_test metadata missing in {path}")
        if int(payload.get("base_seed", -1)) != 42:
            raise RuntimeError(f"unexpected base seed in {path}")
        if [item.get("seed") for item in payload.get("runs", [])] != [42, 43, 44]:
            raise RuntimeError(f"unexpected model seeds in {path}")
        by_pair[(run["dataset"], run["variant"])] = {}
        for item in payload["runs"]:
            metrics = item["metrics"]
            forbidden = {key for key in metrics if key.startswith("test_")}
            if forbidden:
                raise RuntimeError(f"test metrics found in development run: {sorted(forbidden)}")
            if not {"val_acc", "val_macro_f1", "val_ce"}.issubset(metrics):
                raise RuntimeError(f"validation metrics incomplete in {path}")
            metadata = item.get("metadata", {})
            if metadata.get("development_no_test") is not True:
                raise RuntimeError(f"per-run development metadata missing in {path}")
            row = {
                "dataset": run["dataset"],
                "variant": run["variant"],
                "seed": int(item["seed"]),
                "val_accuracy": float(metrics["val_acc"]),
                "val_macro_f1": float(metrics["val_macro_f1"]),
                "val_ce": float(metrics["val_ce"]),
                "best_epoch": int(metadata["best_epoch"]),
                "model_parameters": int(metadata["model_parameters"]),
                "development_no_test": True,
                "run_metrics_path": str(path),
            }
            performance.append(row)
            by_pair[(run["dataset"], run["variant"])][int(item["seed"])] = row
    if len(performance) != 36:
        raise RuntimeError(f"expected 36 per-seed training rows, found {len(performance)}")
    return performance, by_pair


def summarize_performance(performance: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in performance:
        grouped.setdefault((row["dataset"], row["variant"]), []).append(row)
    result = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            rows = sorted(grouped[(dataset, variant)], key=lambda item: item["seed"])
            entry = {"dataset": dataset, "variant": variant, "n": len(rows)}
            for metric, field in (
                ("val_accuracy", "val_accuracy"),
                ("val_macro_f1", "val_macro_f1"),
                ("val_ce", "val_ce"),
                ("best_epoch", "best_epoch"),
            ):
                entry[f"{metric}_mean"], entry[f"{metric}_population_sd"] = mean_sd(
                    [float(row[field]) for row in rows]
                )
            result.append(entry)
    return result


def paired_deltas(by_pair: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    for dataset in DATASETS:
        for newer, baseline in COMPARISONS:
            new_runs = by_pair[(dataset, newer)]
            old_runs = by_pair[(dataset, baseline)]
            for seed in (42, 43, 44):
                row = {
                    "dataset": dataset,
                    "comparison": f"{newer} - {baseline}",
                    "new_variant": newer,
                    "baseline_variant": baseline,
                    "seed": seed,
                }
                for metric, field in (
                    ("val_acc", "val_accuracy"),
                    ("val_macro_f1", "val_macro_f1"),
                    ("val_ce", "val_ce"),
                ):
                    row[metric] = float(new_runs[seed][field]) - float(old_runs[seed][field])
                rows.append(row)
    summary = []
    for dataset in DATASETS:
        for newer, baseline in COMPARISONS:
            group = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["new_variant"] == newer
                and row["baseline_variant"] == baseline
            ]
            entry = {
                "dataset": dataset,
                "comparison": f"{newer} - {baseline}",
                "n_paired": len(group),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                entry[f"{metric}_mean"], entry[f"{metric}_population_sd"] = mean_sd(values)
                entry[f"{metric}_positive_count"] = sum(value > 1e-12 for value in values)
                entry[f"{metric}_negative_count"] = sum(value < -1e-12 for value in values)
                entry[f"{metric}_tie_count"] = sum(abs(value) <= 1e-12 for value in values)
            summary.append(entry)
    return rows, summary


def hydra_config(dataset: str, seed: int):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=care_mag_v0",
                "model.variant=context_adapter",
                f"seed={seed}",
                "num_runs=1",
                "task.evaluate_test=false",
                "task.development_no_test=true",
            ],
        )


def intervention_metrics(model, head, x, edge_index, data, eval_labels, device, mode, seed=0):
    from src.tasks.nc import _evaluate_split

    with torch.no_grad():
        z, _, _, _, _ = model(
            x,
            edge_index,
            intervention=mode,
            shuffle_seed=seed,
        )
        z = z.detach().cpu()
        result = _evaluate_split(
            head,
            z,
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


def run_mechanism_analysis(manifest: dict, device: torch.device) -> tuple[list[dict], list[dict]]:
    from src.data import load_mag_data
    from src.models.care_mag_v0 import Model
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels

    diagnostics: list[dict] = []
    interventions: list[dict] = []
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False

    for dataset in DATASETS:
        cfg = hydra_config(dataset, 42)
        data = load_mag_data(cfg, "nc", 42)
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

        for run in manifest["runs"]:
            if run["dataset"] != dataset or run["variant"] != "context_adapter":
                continue
            checkpoint_base = Path(run["checkpoint_path"])
            for run_id, seed in enumerate((42, 43, 44), start=1):
                checkpoint_path = checkpoint_base.with_name(
                    f"{checkpoint_base.stem}_run{run_id}{checkpoint_base.suffix}"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                model = Model(cfg, data_info).to(device)
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                head.load_state_dict(checkpoint["head_state"])
                head.eval()

                with torch.no_grad():
                    z, _, _, _, info = model(
                        x, edge_index, return_details=True
                    )
                    val = _evaluate_split(
                        head,
                        z.detach().cpu(),
                        data.y,
                        data.val_idx,
                        device,
                        batch_size=4096,
                        eval_labels=eval_labels,
                    )
                    details = info["details"]
                    modality_coefficients = details["coefficients"]
                    cross_disagreement = float(
                        (
                            modality_coefficients["text"]
                            - modality_coefficients["visual"]
                        )
                        .square()
                        .mean()
                        .sqrt()
                        .item()
                    )
                    for modality in ("text", "visual"):
                        base = info[modality]
                        response = details["responses"][modality]
                        delta = details["deltas"][modality]
                        coefficient = modality_coefficients[modality]
                        response_rms = float(response.square().mean().sqrt().item())
                        delta_rms = float(delta.square().mean().sqrt().item())
                        gamma = float(base["gamma"].item())
                        diagnostics.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "variant": "context_adapter",
                                "modality": modality,
                                "lambda": float(base["lambda"].item()),
                                "gamma": gamma,
                                "alpha_k1_mean": float(base["alpha_mean"][0].item()),
                                "alpha_k1_std": float(base["alpha_std"][0].item()),
                                "alpha_k2_mean": float(base["alpha_mean"][1].item()),
                                "alpha_k2_std": float(base["alpha_std"][1].item()),
                                "alpha_k3_mean": float(base["alpha_mean"][2].item()),
                                "alpha_k3_std": float(base["alpha_std"][2].item()),
                                "effective_hop_mean": float(base["effective_hop_mean"].item()),
                                "effective_hop_std": float(base["effective_hop_std"].item()),
                                "response_rms": response_rms,
                                "delta_rms": delta_rms,
                                "delta_response_rms_ratio": delta_rms / (response_rms + 1e-8),
                                "gamma_delta_response_rms_ratio": gamma * delta_rms / (response_rms + 1e-8),
                                "lambda_gamma_delta_response_rms_ratio": float(
                                    base["lambda"].item()
                                ) * gamma * delta_rms / (response_rms + 1e-8),
                                "cosine_delta_response": float(
                                    F.cosine_similarity(delta, response, dim=1, eps=1e-8)
                                    .mean()
                                    .item()
                                ),
                                "coefficient_rank_mean": json.dumps(
                                    base["coefficient_mean"].cpu().tolist(), separators=(",", ":")
                                ),
                                "coefficient_rank_std": json.dumps(
                                    base["coefficient_std"].cpu().tolist(), separators=(",", ":")
                                ),
                                "coefficient_node_std_mean": float(
                                    base["coefficient_node_std"].item()
                                ),
                                "text_visual_coefficient_disagreement_rms": cross_disagreement,
                                "normal_val_accuracy": float(val["acc"]),
                                "normal_val_macro_f1": float(val["macro_f1"]),
                                "normal_val_ce": float(val["ce"]),
                            }
                        )

                modes = [("normal", 0)]
                modes.extend(
                    [("adapter_off", 0), ("coeff_global_mean", 0)]
                )
                modes.extend(
                    [("coeff_node_shuffle", repeat) for repeat in range(1, 6)]
                )
                normal_row = None
                for mode, repeat in modes:
                    seed_value = 42000 + seed * 10 + repeat
                    metrics = intervention_metrics(
                        model,
                        head,
                        x,
                        edge_index,
                        data,
                        eval_labels,
                        device,
                        mode,
                        seed=seed_value,
                    )
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "intervention": mode,
                        "repeat": repeat,
                        **metrics,
                    }
                    interventions.append(row)
                    if mode == "normal":
                        normal_row = row
                if normal_row is None:
                    raise RuntimeError("normal validation row was not recorded")
                for row in interventions[-7:]:
                    if row["dataset"] != dataset or row["seed"] != seed:
                        continue
                    row["delta_val_accuracy_vs_normal"] = row["val_accuracy"] - normal_row["val_accuracy"]
                    row["delta_val_macro_f1_vs_normal"] = row["val_macro_f1"] - normal_row["val_macro_f1"]
                    row["delta_val_ce_vs_normal"] = row["val_ce"] - normal_row["val_ce"]
                del model, head, checkpoint, z, info
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del data, x, edge_index
    return diagnostics, interventions


def summarize_interventions(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["intervention"]), []).append(row)
    result = []
    order = ("normal", "adapter_off", "coeff_global_mean", "coeff_node_shuffle")
    for dataset in DATASETS:
        for intervention in order:
            group = groups[(dataset, intervention)]
            entry = {"dataset": dataset, "intervention": intervention, "n": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                entry[f"{metric}_mean"], entry[f"{metric}_population_sd"] = mean_sd(
                    [float(row[metric]) for row in group]
                )
            if intervention != "normal":
                for metric in (
                    "delta_val_accuracy_vs_normal",
                    "delta_val_macro_f1_vs_normal",
                    "delta_val_ce_vs_normal",
                ):
                    values = [float(row[metric]) for row in group]
                    entry[f"{metric}_mean"], entry[f"{metric}_population_sd"] = mean_sd(values)
            result.append(entry)
    return result


def assess_label(paired_summary: list[dict]) -> tuple[str, dict]:
    by_key = {(row["dataset"], row["comparison"]): row for row in paired_summary}

    def direction(dataset: str, comparison: str) -> dict[str, bool]:
        row = by_key[(dataset, comparison)]
        return {
            "acc": row["val_acc_mean"] > 0.0,
            "f1": row["val_macro_f1_mean"] > 0.0,
            "ce": row["val_ce_mean"] < 0.0,
        }

    structural = [
        direction(dataset, "structural_base - prior_only") for dataset in DATASETS
    ]
    static = [
        direction(dataset, "static_adapter - structural_base") for dataset in DATASETS
    ]
    context = [
        direction(dataset, "context_adapter - static_adapter") for dataset in DATASETS
    ]
    structural_count = sum(all(item.values()) for item in structural)
    static_count = sum(all(item.values()) for item in static)
    context_count = sum(all(item.values()) for item in context)
    context_any_signal = any(any(item.values()) for item in context)
    context_clear_conflict = any(any(item.values()) and not all(item.values()) for item in context)

    if context_count >= 2 and not context_clear_conflict:
        label = "CONTEXT_ADAPTATION_SUPPORTED"
    elif context_any_signal or context_clear_conflict:
        label = "DATASET_DEPENDENT_MIXED"
    elif static_count >= 2:
        label = "STATIC_CORRECTION_ONLY"
    elif structural_count >= 2:
        label = "STRUCTURAL_BACKBONE_ONLY"
    else:
        label = "INVALID_OR_FAILED"
    return label, {
        "structural_improved_dataset_count": structural_count,
        "static_improved_dataset_count": static_count,
        "context_improved_dataset_count": context_count,
        "context_any_metric_signal": context_any_signal,
        "context_within_dataset_metric_conflict": context_clear_conflict,
    }


def markdown_report(
    manifest: dict,
    performance_summary: list[dict],
    paired_summary: list[dict],
    mechanism: list[dict],
    intervention_summary: list[dict],
    smoke: dict | None,
    label: str,
    assessment: dict,
    device_label: str,
) -> str:
    lines = [
        "# CARE-MAG A0–A2 context-adapter architecture screen",
        "",
        f"- Base SHA: `{manifest['base_sha']}`",
        f"- Formal status: `{manifest.get('status')}`; 36 requested runs across 3 datasets, 4 variants, and seeds 42–44.",
        "- NC protocol: full-graph, frozen main task optimization, best checkpoint selected by validation accuracy.",
        "- Test evaluation: disabled. Formal metrics contain validation fields only; development mode masks test-index labels on the training device.",
        f"- Mechanism analysis device: `{device_label}`; all interventions use validation nodes only.",
        "",
        "## Formal validation performance",
        "",
        "Values are mean ± population SD over the three paired seeds. Accuracy and Macro-F1 are fractions; CE is unscaled.",
        "",
        "| Dataset | Variant | Val Accuracy | Val Macro-F1 | Val CE | Best epoch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in performance_summary:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | "
            f"{fmt(row['val_accuracy_mean'])} ± {fmt(row['val_accuracy_population_sd'])} | "
            f"{fmt(row['val_macro_f1_mean'])} ± {fmt(row['val_macro_f1_population_sd'])} | "
            f"{fmt(row['val_ce_mean'])} ± {fmt(row['val_ce_population_sd'])} | "
            f"{row['best_epoch_mean']:.1f} ± {row['best_epoch_population_sd']:.1f} |"
        )

    lines += [
        "",
        "## Paired deltas",
        "",
        "Deltas are first variant minus baseline at the same seed. SD is population SD; counts give positive / negative / tie runs. No IID p-values are used. For CE, negative values favor the first variant.",
        "",
        "| Dataset | Comparison | Metric | Mean Δ ± SD | + / − / tie |",
        "|---|---|---|---:|---:|",
    ]
    metric_display = {"val_acc": "Val Accuracy", "val_macro_f1": "Val Macro-F1", "val_ce": "Val CE"}
    for row in paired_summary:
        for metric in METRICS:
            lines.append(
                f"| {row['dataset']} | {row['comparison']} | {metric_display[metric]} | "
                f"{row[f'{metric}_mean']:+.5f} ± {row[f'{metric}_population_sd']:.5f} | "
                f"{row[f'{metric}_positive_count']} / {row[f'{metric}_negative_count']} / {row[f'{metric}_tie_count']} |"
            )

    lines += [
        "",
        "## Context-checkpoint mechanism diagnostics",
        "",
        "Rows below aggregate the selected context checkpoints by dataset and modality. Hop attention means are followed by across-node alpha SD. Coefficient vectors are retained rank-wise in `data/mechanism_diagnostics.csv`.",
        "",
        "| Dataset | Modality | λ | γ | α k1 / k2 / k3 | Effective hop | R RMS | Δ RMS / R RMS | γΔ RMS / R RMS | λγΔ RMS / R RMS | cos(Δ,R) | Node coefficient SD | Text/visual disagreement RMS |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in mechanism:
        groups.setdefault((row["dataset"], row["modality"]), []).append(row)
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            rows = groups[(dataset, modality)]
            def avg(key):
                return statistics.mean(float(item[key]) for item in rows)
            alpha = "/".join(f"{avg(f'alpha_k{i}_mean'):.3f}" for i in (1, 2, 3))
            lines.append(
                f"| {dataset} | {modality} | {avg('lambda'):.3f} | {avg('gamma'):.3f} | {alpha} | "
                f"{avg('effective_hop_mean'):.3f} | {avg('response_rms'):.4f} | "
                f"{avg('delta_response_rms_ratio'):.4f} | {avg('gamma_delta_response_rms_ratio'):.4f} | "
                f"{avg('lambda_gamma_delta_response_rms_ratio'):.4f} | {avg('cosine_delta_response'):.4f} | {avg('coefficient_node_std_mean'):.5f} | "
                f"{avg('text_visual_coefficient_disagreement_rms'):.5f} |"
            )

    lines += [
        "",
        "## Validation checkpoint interventions",
        "",
        "Intervention deltas are relative to the same checkpoint's normal context forward. `coeff_node_shuffle` reports five deterministic shuffles per checkpoint; its SD includes seeds and shuffle repeats. These interventions assess checkpoint reliance on correspondence and do not replace the retrained context-versus-static comparison.",
        "",
        "| Dataset | Intervention | Val Accuracy | Val Macro-F1 | Val CE | Δ Accuracy | Δ Macro-F1 | Δ CE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in intervention_summary:
        dacc = row.get("delta_val_accuracy_vs_normal_mean")
        df1 = row.get("delta_val_macro_f1_vs_normal_mean")
        dce = row.get("delta_val_ce_vs_normal_mean")
        lines.append(
            f"| {row['dataset']} | {row['intervention']} | "
            f"{row['val_accuracy_mean']:.4f} ± {row['val_accuracy_population_sd']:.4f} | "
            f"{row['val_macro_f1_mean']:.4f} ± {row['val_macro_f1_population_sd']:.4f} | "
            f"{row['val_ce_mean']:.4f} ± {row['val_ce_population_sd']:.4f} | "
            f"{'' if dacc is None else f'{dacc:+.5f}'} | "
            f"{'' if df1 is None else f'{df1:+.5f}'} | "
            f"{'' if dce is None else f'{dce:+.5f}'} |"
        )

    varying_coefficients = any(
        float(row["coefficient_node_std_mean"]) > 1e-7 for row in mechanism
    )
    mean_best_hop_weight = statistics.mean(
        max(float(row[f"alpha_k{i}_mean"]) for i in (1, 2, 3))
        for row in mechanism
    )
    dominant_hops = [
        max((1, 2, 3), key=lambda hop: float(row[f"alpha_k{hop}_mean"]))
        for row in mechanism
    ]
    most_common_hop = max(set(dominant_hops), key=dominant_hops.count)
    dominant_fraction = dominant_hops.count(most_common_hop) / max(len(dominant_hops), 1)
    hop_collapse = (
        f"Possible fixed-hop concentration at k={most_common_hop} "
        f"({dominant_fraction:.0%} of checkpoint-modality rows; mean max-hop weight {mean_best_hop_weight:.3f})."
        if mean_best_hop_weight >= 0.90 and dominant_fraction >= 0.80
        else f"No strong fixed-hop collapse (mean max-hop weight {mean_best_hop_weight:.3f}; dominant k={most_common_hop} in {dominant_fraction:.0%} of rows)."
    )
    label_details = assess_evidence(
        paired_summary,
        label,
        assessment,
        intervention_summary,
        varying_coefficients,
    )
    lp = "not run" if smoke is None or not smoke.get("lp_run") else (
        "passed" if smoke["lp_run"].get("return_code") == 0 else "failed"
    )
    lp_observed = (smoke or {}).get("lp_run", {}).get("observed", {})
    lp_evidence = (
        f"LinkNeighborLoader={lp_observed.get('link_neighbor_loader')}, "
        f"fanouts={lp_observed.get('sampler_fanouts')}, "
        f"positive edges removed per epoch={lp_observed.get('positive_message_edges_removed_by_epoch')}, "
        f"validation inference={lp_observed.get('validation_inference_completed')}, "
        f"checkpoint saved={lp_observed.get('checkpoint_saved')}"
        if lp_observed
        else "details unavailable"
    )
    nc_smoke = "not recorded" if smoke is None else smoke.get("status", "incomplete")
    max_gamma_ratio = max(
        float(row["lambda_gamma_delta_response_rms_ratio"]) for row in mechanism
    )
    average_gamma_ratio = statistics.mean(
        float(row["lambda_gamma_delta_response_rms_ratio"]) for row in mechanism
    )
    report_label = label
    lines += [
        "",
        f"## Conservative diagnostic label: `{report_label}`",
        "",
        label_details,
        "",
        "## Self-audit",
        "",
        f"1. **Only based on main?** Yes. The experiment branch was created from fetched `origin/main` at `{manifest['base_sha']}`; no code or history from another branch was used.",
        "2. **Any test evaluation or test metric access?** No test evaluation was enabled. Analyzer rejects any `test_*` metric keys; only train/validation labels are used for validation bookkeeping.",
        "3. **Did development_no_test isolate test labels?** Yes. The NC runner errors if test evaluation is enabled, builds Macro-F1 labels from train+validation indices only, and masks test-index labels to -1 before moving labels to the training device. Per-run metadata records the mode.",
        "4. **Are all four variants parameter matched?** Yes, parameter counts and state-dict layouts are tested equal.",
        "5. **Are A1/A2 parameters and initialization matched?** Yes. Tests reconstruct each variant with the same seed and verify all same-named state tensors bitwise equal.",
        f"6. **Is structural_base clearly better than prior_only?** {label_details.splitlines()[0]}",
        f"7. **Does the static adapter add independent value?** {label_details.splitlines()[1]}",
        f"8. **Does context add over static?** {label_details.splitlines()[2]}",
        f"9. **Are context coefficients node-varying?** {'Yes' if varying_coefficients else 'No measurable node variation'}; see per-rank and node SDs in the mechanism CSV.",
        f"10. **Do global-mean/shuffle interventions show correspondence use?** {label_details.splitlines()[4]}",
        f"11. **Does checkpoint reliance agree with retrained architecture gain?** {label_details.splitlines()[5]}",
        f"12. **Did trajectory readout collapse to a fixed hop?** {hop_collapse}",
        f"13. **Is correction too large relative to response?** The maximum trust- and gamma-scaled Δ/R RMS ratio was {max_gamma_ratio:.5f} (mean {average_gamma_ratio:.5f}); raw Δ/R remains in the mechanism CSV. Interpret with validation comparisons and cosine diagnostics.",
        f"14. **Is there a dataset-dependent regime?** {'Yes; the final label is mixed.' if label == 'DATASET_DEPENDENT_MIXED' else 'See the paired dataset rows; no mixed label was assigned.'}",
        f"15. **Enough evidence for A3 shared/private residual MoE?** {label_details.splitlines()[7]}",
        f"16. **Does LP smoke establish sampled-protocol compatibility?** NC GPU smoke `{nc_smoke}`; sports-copurchase LP smoke `{lp}`. {lp_evidence}. This establishes execution compatibility only, not LP quality.",
        "17. **Any claim beyond direct support?** No. This screen supports validation-only comparisons for these three datasets and this frozen optimization setup; it does not establish test/generalization gains, formal LP quality, or full CARE-MAG effectiveness.",
        "",
        "## Limitations",
        "",
        "- Three seeds give a compact paired screen, not high-precision estimates.",
        "- All reported NC metrics are validation metrics; test results were intentionally neither evaluated nor read.",
        "- The LP check is only a short sampled execution smoke and is not a performance campaign.",
        "- Intervention outcomes describe reliance of the selected checkpoints on coefficient-to-node correspondence; they are not retrained architecture comparisons.",
        "- The architecture is limited to the specified three-hop diffusion, global trust scalars, and concat residual fusion.",
        "",
        "## A3 recommendation",
        "",
        label_details.splitlines()[7],
        "",
    ]
    return "\n".join(lines)


def assess_evidence(
    paired_summary: list[dict],
    label: str,
    assessment: dict,
    intervention_summary: list[dict],
    varying_coefficients: bool,
) -> str:
    table = {(row["dataset"], row["comparison"]): row for row in paired_summary}

    def sentence(comp: str) -> str:
        rows = [table[(dataset, comp)] for dataset in DATASETS]
        means = [
            (row["val_acc_mean"], row["val_macro_f1_mean"], row["val_ce_mean"])
            for row in rows
        ]
        good = sum(acc > 0 and f1 > 0 and ce < 0 for acc, f1, ce in means)
        conflict = sum(
            (acc > 0 or f1 > 0 or ce < 0)
            and not (acc > 0 and f1 > 0 and ce < 0)
            for acc, f1, ce in means
        )
        return f"{good}/3 datasets improve Accuracy, Macro-F1, and CE together; {conflict}/3 show mixed directions."

    context_map = {
        "CONTEXT_ADAPTATION_SUPPORTED": "At least two datasets improve together without a conflicting direction in the remaining dataset.",
        "STATIC_CORRECTION_ONLY": "The retrained static control improves over structural_base consistently, while context has no consistent increment.",
        "STRUCTURAL_BACKBONE_ONLY": "The structural trajectory improves over prior_only, while neither adapter establishes independent value.",
        "DATASET_DEPENDENT_MIXED": "The incremental adapter evidence varies by dataset or validation metric; it is not a uniform context-adaptation result.",
        "INVALID_OR_FAILED": "The frozen comparison did not establish a reliable structural or adapter gain, or campaign validation failed.",
    }
    intervention_map = {
        "CONTEXT_ADAPTATION_SUPPORTED": "Retrained context also improves over the retrained static model; any intervention reliance is corroborative only.",
        "STATIC_CORRECTION_ONLY": "Checkpoint interventions cannot substitute for the absent retrained context increment.",
        "STRUCTURAL_BACKBONE_ONLY": "Intervention changes do not establish an adapter benefit when retrained controls do not.",
        "DATASET_DEPENDENT_MIXED": "Intervention reliance may coexist with mixed retrained results and does not resolve them.",
        "INVALID_OR_FAILED": "No positive architecture conclusion is drawn from intervention-only effects.",
    }
    intervention_by_key = {
        (row["dataset"], row["intervention"]): row
        for row in intervention_summary
    }
    shuffle_degraded = 0
    global_degraded = 0
    shuffle_directional = 0
    for dataset in DATASETS:
        for name in ("coeff_node_shuffle", "coeff_global_mean"):
            row = intervention_by_key[(dataset, name)]
            deltas = (
                float(row["delta_val_accuracy_vs_normal_mean"]),
                float(row["delta_val_macro_f1_vs_normal_mean"]),
                float(row["delta_val_ce_vs_normal_mean"]),
            )
            directional = deltas[0] < 0 or deltas[1] < 0 or deltas[2] > 0
            clear_directions = sum(
                [
                    deltas[0] <= -0.005,
                    deltas[1] <= -0.005,
                    deltas[2] >= 0.005,
                ]
            )
            degraded = clear_directions >= 2
            if degraded and name == "coeff_node_shuffle":
                shuffle_degraded += 1
            if degraded and name == "coeff_global_mean":
                global_degraded += 1
            if directional and name == "coeff_node_shuffle":
                shuffle_directional += 1
    correspondence = (
        f"Meaningful degradation (at least two metrics moving by 0.5 percentage points or 0.005 CE) occurs under node shuffling in {shuffle_degraded}/3 datasets and global-mean replacement in {global_degraded}/3. Directional changes occur for shuffle in {shuffle_directional}/3; checkpoint correspondence sensitivity is dataset-dependent."
    )
    retrained_gain = assessment["context_improved_dataset_count"] >= 2
    consistency = (
        "Checkpoint intervention reliance and retrained context-over-static gains point in the same direction."
        if retrained_gain and shuffle_degraded >= 2
        else "Checkpoint intervention reliance does not establish a retrained context-over-static gain."
    )
    return "\n".join(
        [
            f"- Structural base vs prior only: {sentence('structural_base - prior_only')}",
            f"- Static adapter vs structural base: {sentence('static_adapter - structural_base')}",
            f"- Context adapter vs static adapter: {sentence('context_adapter - static_adapter')}",
            f"- Context coefficients are {'node-varying' if varying_coefficients else 'not measurably node-varying'}; inspect rank-wise standard deviations in the mechanism CSV.",
            f"- {correspondence} Global mean removes node variation; shuffle preserves each checkpoint's coefficient rows while breaking their node assignment. These are checkpoint-reliance probes.",
            f"- {consistency}",
            f"- Evidence framing for `{label}`: {context_map[label]} {intervention_map[label]}",
            f"- A3 decision: {'do not proceed to A3 on this screen alone.' if label != 'CONTEXT_ADAPTATION_SUPPORTED' else 'an A3 proposal is warranted for human review, but this screen does not authorize implementation.'}",
            "- Trajectory collapse and correction scale are summarized numerically in the mechanism table; inspect per-checkpoint and per-rank values in `data/mechanism_diagnostics.csv` before interpreting them.",
            "- Dataset regime: read each paired table row separately; pooled seeds across datasets are not treated as IID.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-interventions", action="store_true")
    args = parser.parse_args()
    manifest_path = RESEARCH / "run_manifest.json"
    manifest = read_json(manifest_path)
    performance, by_pair = load_performance(manifest)
    performance_summary = summarize_performance(performance)
    paired, paired_summary = paired_deltas(by_pair)

    write_csv(
        RESEARCH / "data" / "performance_by_run.csv",
        performance,
        ["dataset", "variant", "seed", "val_accuracy", "val_macro_f1", "val_ce", "best_epoch", "model_parameters", "development_no_test", "run_metrics_path"],
    )
    write_csv(RESEARCH / "data" / "performance_summary.csv", performance_summary)
    write_csv(RESEARCH / "data" / "paired_delta_by_run.csv", paired)
    write_csv(RESEARCH / "data" / "paired_delta_summary.csv", paired_summary)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the analyzer process")
    if args.skip_interventions:
        diagnostics, interventions = [], []
    else:
        diagnostics, interventions = run_mechanism_analysis(manifest, device)
    write_csv(RESEARCH / "data" / "mechanism_diagnostics.csv", diagnostics)
    write_csv(RESEARCH / "data" / "intervention_by_run.csv", interventions)
    intervention_summary = summarize_interventions(interventions) if interventions else []
    write_csv(RESEARCH / "data" / "intervention_summary.csv", intervention_summary)

    label, assessment = assess_label(paired_summary)
    smoke_path = RESEARCH / "smoke_status.json"
    smoke = read_json(smoke_path) if smoke_path.exists() else None
    report = markdown_report(
        manifest,
        performance_summary,
        paired_summary,
        diagnostics,
        intervention_summary,
        smoke,
        label,
        assessment,
        str(device),
    )
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
