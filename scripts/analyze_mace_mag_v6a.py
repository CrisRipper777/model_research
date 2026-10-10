#!/usr/bin/env python3
"""Summarize the frozen V6A validation-only campaign and selected mechanisms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESEARCH = ROOT / "research" / "mace_mag_v6a_expert_collaboration"
DATA_ROOT = RESEARCH / "data"
OUTPUT_ROOT = ROOT / "outputs" / "mace_mag_v6a_expert_collaboration"
DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("a0_static", "a1_node", "a2_cross", "a3_intra")
COMPARISONS = (
    ("a1_node-a0_static", "a1_node", "a0_static"),
    ("a2_cross-a1_node", "a2_cross", "a1_node"),
    ("a2_cross-a3_intra", "a2_cross", "a3_intra"),
    ("a3_intra-a1_node", "a3_intra", "a1_node"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def population_sd(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _load_model(variant: str, data_info: dict[str, Any], checkpoint: dict[str, Any]):
    import torch
    from omegaconf import OmegaConf

    from src.models.mace_mag_v6a import Model

    model_cfg = OmegaConf.load(ROOT / "configs" / "model" / "mace_mag_v6a.yaml")
    model_cfg.variant = variant
    model = Model(OmegaConf.create({"model": model_cfg}), data_info)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _selected_route_diagnostics(
    model, modality: str, detail: dict[str, Any], collaboration: dict[str, Any], active
) -> dict[str, Any]:
    import torch

    route = detail["routes"][modality]
    indices = route["top_indices"][active]
    if indices.numel():
        expert_counts = torch.bincount(indices.reshape(-1), minlength=4).float()
        expert_distribution = (expert_counts / expert_counts.sum()).cpu().tolist()
        pairs = indices.sort(dim=-1).values
        pair_ids = pairs[:, 0] * 4 + pairs[:, 1]
        pair_counts = torch.bincount(pair_ids, minlength=16).float()
        pair_probability = pair_counts[pair_counts > 0] / pair_counts.sum()
        pair_entropy = float(
            (-(pair_probability * pair_probability.log()).sum()).cpu()
        )
        pair_used = int((pair_counts > 0).sum().item())
        pair_distribution = {
            f"{i // 4}-{i % 4}": float(pair_counts[i] / pair_counts.sum())
            for i in range(16)
            if pair_counts[i] > 0
        }
    else:
        expert_distribution = [0.0] * 4
        pair_entropy = 0.0
        pair_used = 0
        pair_distribution = {}
    target_route = route
    mixture = torch.einsum(
        "nm,nmd->nd", target_route["route_weights"], detail["expert_values"][modality]
    )
    modality_index = 0 if modality == "text" else 1
    tau = float(torch.sigmoid(model.structural_strength_raw[modality_index]).detach().cpu())
    if modality in collaboration:
        item = collaboration[modality]
        attention = item["attention"].detach()
        target_ids = item["target_indices"].detach()
        source_ids = item["source_indices"].detach()
        if active.any():
            active_attention = attention[active]
            null_mass = float(active_attention[:, :, -1].mean().cpu())
            source_mass = float(active_attention[:, :, :-1].sum(-1).mean().cpu())
            matching = target_ids.unsqueeze(-1) == source_ids.unsqueeze(1)
            same_id_mass = float(
                (active_attention[:, :, :-1] * matching[active].float())
                .sum(-1)
                .mean()
                .cpu()
            )
        else:
            null_mass = source_mass = same_id_mass = 0.0
        correction = item["correction"].detach()
        lam = float(
            torch.sigmoid(model.collaboration_strength_raw[modality_index])
            .detach()
            .cpu()
        )
        if active.any():
            structural = tau * mixture[active]
            actual_collaboration = lam * correction[active]
            structure_rms = float(structural.float().square().mean().sqrt().cpu())
            collaboration_rms = float(
                actual_collaboration.float().square().mean().sqrt().cpu()
            )
            rms_ratio = collaboration_rms / max(structure_rms, 1.0e-12)
            correction_rms = float(correction[active].float().square().mean().sqrt().cpu())
        else:
            rms_ratio = correction_rms = 0.0
        source_modality = item["source_modality"]
        attention_enabled = True
    else:
        null_mass = source_mass = same_id_mass = None
        lam = 0.0
        rms_ratio = correction_rms = 0.0
        source_modality = None
        attention_enabled = False
    return {
        "expert_use_distribution_json": json.dumps(expert_distribution),
        "selected_top2_pair_distribution_json": json.dumps(pair_distribution),
        "selected_top2_unique_pair_count": pair_used,
        "selected_top2_pair_entropy_nats": pair_entropy,
        "attention_enabled": attention_enabled,
        "attention_source_modality": source_modality,
        "mean_source_attention_mass": source_mass,
        "mean_null_attention_mass": null_mass,
        "mean_same_expert_id_attention_mass": same_id_mass,
        "correction_rms_unscaled": correction_rms,
        "scaled_collaboration_to_structure_rms_ratio": rms_ratio,
        "structural_strength": tau,
        "collaboration_strength": lam,
    }


def _audit_checkpoint_rows(rows: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    import torch

    from scripts.run_mace_mag_v6a import load_preflight_dataset

    records: list[dict[str, Any]] = []
    device_obj = torch.device(device)
    for dataset in DATASETS:
        x_cpu, edge_cpu, data_info = load_preflight_dataset(dataset)
        x, edge_index = x_cpu.to(device_obj), edge_cpu.to(device_obj)
        data_rows = [row for row in rows if row["dataset"] == dataset]
        for row in data_rows:
            checkpoint = torch.load(
                ROOT / row["checkpoint_path"], map_location="cpu", weights_only=False
            )
            model = _load_model(row["variant"], data_info, checkpoint).to(device_obj).eval()
            with torch.no_grad():
                z, _, _, aux_loss, info = model(x, edge_index, return_details=True)
            if not torch.isfinite(z).all() or not torch.isfinite(aux_loss):
                raise FloatingPointError(
                    f"non-finite selected checkpoint output: {dataset} {row['seed']} {row['variant']}"
                )
            active = info["details"]["active"]
            gradient_info = row["metadata"].get("gradient_diagnostics") or {}
            key_gradients = [
                value.get("gradient_nonzero", False)
                for key, value in gradient_info.items()
                if ".key.weight" in key
            ] if row["variant"] in {"a2_cross", "a3_intra"} else []
            value_gradients = [
                value.get("gradient_nonzero", False)
                for key, value in gradient_info.items()
                if ".value.weight" in key
            ] if row["variant"] in {"a2_cross", "a3_intra"} else []
            for modality in ("text", "visual"):
                diag = _selected_route_diagnostics(
                    model,
                    modality,
                    info["details"],
                    info["details"]["collaboration"],
                    active,
                )
                records.append(
                    {
                        "dataset": dataset,
                        "seed": int(row["seed"]),
                        "variant": row["variant"],
                        "target_modality": modality,
                        "active_nodes": int(active.sum().item()),
                        "gradient_key_nonzero": (
                            all(key_gradients) if key_gradients else None
                        ),
                        "gradient_value_nonzero": (
                            all(value_gradients) if value_gradients else None
                        ),
                        "gradient_diagnostics_json": json.dumps(gradient_info),
                        **diag,
                    }
                )
            del z, aux_loss, info, model, checkpoint
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
        del x, edge_index, x_cpu, edge_cpu
        print(f"[V6A diagnostics] audited {dataset}: {len(data_rows)} checkpoints", flush=True)
    return records


def _summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    index = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in rows}
    for dataset in DATASETS:
        for variant in VARIANTS:
            cells = [index[(dataset, seed, variant)] for seed in (42, 43, 44)]
            acc = [float(row["metrics"]["val_acc"]) for row in cells]
            f1 = [float(row["metrics"]["val_macro_f1"]) for row in cells]
            metadata = [row["metadata"] for row in cells]
            model_params = [int(item["model_parameters"]) for item in metadata]
            head_params = [int(item["classifier_parameters"]) for item in metadata]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "n_seeds": len(cells),
                    "val_accuracy_mean": mean(acc),
                    "val_accuracy_population_sd": population_sd(acc),
                    "val_macro_f1_mean": mean(f1),
                    "val_macro_f1_population_sd": population_sd(f1),
                    "best_epoch_mean": mean([float(m["best_epoch"]) for m in metadata]),
                    "model_trainable_parameters": model_params[0],
                    "classifier_trainable_parameters": head_params[0],
                    "total_trainable_parameters": model_params[0] + head_params[0],
                    "mean_training_wall_seconds": mean(
                        [float(m["training_wall_seconds"]) for m in metadata]
                    ),
                    "mean_train_step_seconds": mean(
                        [float(m["mean_train_step_seconds"]) for m in metadata]
                    ),
                    "peak_cuda_allocated_bytes": max(
                        int(m["cuda_peak_allocated_bytes"] or 0) for m in metadata
                    ),
                    "peak_cuda_reserved_bytes": max(
                        int(m["cuda_peak_reserved_bytes"] or 0) for m in metadata
                    ),
                }
            )

    paired_rows: list[dict[str, Any]] = []
    for comparison, left, right in COMPARISONS:
        paired_values = []
        for dataset in DATASETS:
            local = []
            for seed in (42, 43, 44):
                left_row = index[(dataset, seed, left)]
                right_row = index[(dataset, seed, right)]
                d_acc = 100.0 * (
                    float(left_row["metrics"]["val_acc"])
                    - float(right_row["metrics"]["val_acc"])
                )
                d_f1 = 100.0 * (
                    float(left_row["metrics"]["val_macro_f1"])
                    - float(right_row["metrics"]["val_macro_f1"])
                )
                record = {
                    "row_type": "paired_run",
                    "comparison": comparison,
                    "dataset": dataset,
                    "seed": seed,
                    "delta_val_accuracy_pp": d_acc,
                    "delta_val_macro_f1_pp": d_f1,
                    "accuracy_direction": "positive" if d_acc > 1e-12 else "negative" if d_acc < -1e-12 else "tie",
                    "macro_f1_direction": "positive" if d_f1 > 1e-12 else "negative" if d_f1 < -1e-12 else "tie",
                }
                paired_rows.append(record)
                local.append(record)
                paired_values.append(record)
            for name, subset in ((dataset, local),):
                acc_delta = [r["delta_val_accuracy_pp"] for r in subset]
                f1_delta = [r["delta_val_macro_f1_pp"] for r in subset]
                paired_rows.append(
                    {
                        "row_type": "summary",
                        "comparison": comparison,
                        "dataset": name,
                        "seed": "ALL",
                        "delta_val_accuracy_pp": mean(acc_delta),
                        "delta_val_macro_f1_pp": mean(f1_delta),
                        "accuracy_positive_pairs": sum(v > 1e-12 for v in acc_delta),
                        "accuracy_negative_pairs": sum(v < -1e-12 for v in acc_delta),
                        "macro_f1_positive_pairs": sum(v > 1e-12 for v in f1_delta),
                        "macro_f1_negative_pairs": sum(v < -1e-12 for v in f1_delta),
                    }
                )
        acc_delta = [r["delta_val_accuracy_pp"] for r in paired_values]
        f1_delta = [r["delta_val_macro_f1_pp"] for r in paired_values]
        paired_rows.append(
            {
                "row_type": "summary",
                "comparison": comparison,
                "dataset": "ALL",
                "seed": "ALL",
                "delta_val_accuracy_pp": mean(acc_delta),
                "delta_val_macro_f1_pp": mean(f1_delta),
                "accuracy_positive_pairs": sum(v > 1e-12 for v in acc_delta),
                "accuracy_negative_pairs": sum(v < -1e-12 for v in acc_delta),
                "macro_f1_positive_pairs": sum(v > 1e-12 for v in f1_delta),
                "macro_f1_negative_pairs": sum(v < -1e-12 for v in f1_delta),
            }
        )
    return summary_rows, paired_rows


def _diagnostic_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANTS:
        for modality in ("text", "visual"):
            subset = [
                r
                for r in rows
                if r["variant"] == variant and r["target_modality"] == modality
            ]
            use = [json.loads(r["expert_use_distribution_json"]) for r in subset]
            mean_use = [mean([item[e] for item in use]) for e in range(4)]
            null = [r["mean_null_attention_mass"] for r in subset if r["mean_null_attention_mass"] is not None]
            source = [r["mean_source_attention_mass"] for r in subset if r["mean_source_attention_mass"] is not None]
            same = [r["mean_same_expert_id_attention_mass"] for r in subset if r["mean_same_expert_id_attention_mass"] is not None]
            ratios = [r["scaled_collaboration_to_structure_rms_ratio"] for r in subset]
            lambdas = [r["collaboration_strength"] for r in subset]
            taus = [r["structural_strength"] for r in subset]
            pair_diversity = [r["selected_top2_unique_pair_count"] for r in subset]
            key_grad = [r["gradient_key_nonzero"] for r in subset if r["gradient_key_nonzero"] is not None]
            output.append(
                {
                    "variant": variant,
                    "target_modality": modality,
                    "mean_expert_use_distribution_json": json.dumps(mean_use),
                    "mean_unique_top2_combinations": mean(pair_diversity),
                    "mean_source_attention_mass": mean(source) if source else None,
                    "mean_null_attention_mass": mean(null) if null else None,
                    "mean_same_expert_id_attention_mass": mean(same) if same else None,
                    "mean_scaled_collaboration_to_structure_rms_ratio": mean(ratios),
                    "mean_structural_strength": mean(taus),
                    "mean_collaboration_strength": mean(lambdas),
                    "runs_with_nonzero_key_gradient": sum(bool(v) for v in key_grad),
                    "runs_with_key_gradient_observed": len(key_grad),
                }
            )
    return output


def _report(
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    diagnostics_summary: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    freeze_sha: str,
) -> str:
    env = read_json(DATA_ROOT / "environment.json")
    pre = read_json(DATA_ROOT / "preflight_summary.json")
    smoke = read_json(DATA_ROOT / "smoke_summary.json")
    tests = read_json(DATA_ROOT / "test_summary.json")
    manifest = read_json(DATA_ROOT / "campaign_manifest.json")
    summary_index = {(r["dataset"], r["variant"]): r for r in summary_rows}
    paired_index = {
        (r["comparison"], r["dataset"]): r
        for r in paired_rows
        if r["row_type"] == "summary"
    }

    lines = [
        "# MACE-MAG V6A: shared structural expert collaboration",
        "",
        "## Protocol and reproducibility",
        "",
        f"- Branch: `{manifest['provenance']['branch']}`; parent: `{manifest['provenance']['parent_commit_sha']}`; freeze SHA: `{freeze_sha}`.",
        f"- Campaign: {len(run_rows)}/36 validation-only full-graph NC runs; protocol `{manifest['protocol']}`; datasets Movies/Grocery/ele-fashion; seeds 42/43/44; variants A0–A3.",
        "- Every formal command explicitly set `task.evaluate_test=false`. Checkpoint selection used Validation Accuracy. Macro-F1 is Validation-only. No Test metric was computed, read, or saved.",
        "- No formal LP, hyperparameter search, sampling, split change, or post-freeze architecture change was performed.",
        f"- Environment: Python {env['python'].split('|')[0].strip()}, PyTorch {env['torch']}, CUDA toolkit {env['torch_cuda']}, PyG {env['torch_geometric']}; device {env['device_name'] if 'device_name' in env else env['selected_device']['device_name']}.",
        f"- Dataset preflight and A2 full-graph forward/backward passed on all three datasets; smoke passed {smoke['completed_runs']}/4. Preflight used no labels/splits and only synthetic class targets for backward validation.",
        f"- Full repository pytest: {tests['full_suite_result']}.",
        "",
        "## Formal validation performance",
        "",
        "Means and population standard deviations are over the three model seeds; accuracy and Macro-F1 are shown in percent. Runtime and GPU memory refer to each selected formal training process.",
        "",
        "| Dataset | Variant | Val Accuracy (%) | Val Macro-F1 (%) | Best epoch mean | Trainable params (model + head) | Mean training time (s) | Peak allocated (GiB) | Peak reserved (GiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            r = summary_index[(dataset, variant)]
            lines.append(
                f"| {dataset} | {variant} | {100*r['val_accuracy_mean']:.2f} ± {100*r['val_accuracy_population_sd']:.2f} | {100*r['val_macro_f1_mean']:.2f} ± {100*r['val_macro_f1_population_sd']:.2f} | {r['best_epoch_mean']:.1f} | {r['model_trainable_parameters']:,} + {r['classifier_trainable_parameters']:,} | {r['mean_training_wall_seconds']:.1f} | {r['peak_cuda_allocated_bytes']/1024**3:.2f} | {r['peak_cuda_reserved_bytes']/1024**3:.2f} |"
            )
    lines.extend(
        [
            "",
            "A0/A1 train the same shared trunk and classifier parameter count. A2/A3 each add 131,218 trainable attention parameters plus two modality collaboration-strength scalars (131,220 extra model parameters total); A2 and A3 have identical trainable parameter counts and initial attention tensors.",
            "",
            "## Paired variant comparisons",
            "",
            "Deltas are percentage points paired by dataset and seed. Positive/negative counts are descriptive only; no significance test was performed.",
            "",
            "| Comparison | Dataset | Accuracy Δ (pp) | Accuracy +/− pairs | Macro-F1 Δ (pp) | Macro-F1 +/− pairs |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for comp, _, _ in COMPARISONS:
        for dataset in (*DATASETS, "ALL"):
            r = paired_index[(comp, dataset)]
            lines.append(
                f"| {comp} | {dataset} | {r['delta_val_accuracy_pp']:+.3f} | {r['accuracy_positive_pairs']}/{3 if dataset != 'ALL' else 9} / {r['accuracy_negative_pairs']}/{3 if dataset != 'ALL' else 9} | {r['delta_val_macro_f1_pp']:+.3f} | {r['macro_f1_positive_pairs']}/{3 if dataset != 'ALL' else 9} / {r['macro_f1_negative_pairs']}/{3 if dataset != 'ALL' else 9} |"
            )
    lines.extend(
        [
            "",
            "Seed-level paired values are preserved in `data/paired_comparisons.csv`.",
            "",
            "## Collaboration diagnostics",
            "",
            "Selected-checkpoint diagnostics describe use patterns, not causal task utility. Expert use is the mean proportion of selected Top-2 slots; combination count is the mean number of distinct unordered Top-2 pairs within a run.",
            "",
            "| Variant | Target modality | Expert-use distribution [0..3] | Mean distinct Top-2 pairs | Source attention | Null attention | Same-ID attention | Scaled correction / structure RMS | Mean τ | Mean λ | Nonzero key-grad runs |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in diagnostics_summary:
        display = lambda v: "n/a" if v is None else f"{v:.3f}"
        uses = ", ".join(f"{float(v):.3f}" for v in json.loads(r["mean_expert_use_distribution_json"]))
        total = r["runs_with_key_gradient_observed"]
        lines.append(
            f"| {r['variant']} | {r['target_modality']} | [{uses}] | {r['mean_unique_top2_combinations']:.2f} | {display(r['mean_source_attention_mass'])} | {display(r['mean_null_attention_mass'])} | {display(r['mean_same_expert_id_attention_mass'])} | {r['mean_scaled_collaboration_to_structure_rms_ratio']:.4f} | {r['mean_structural_strength']:.3f} | {r['mean_collaboration_strength']:.3f} | {r['runs_with_nonzero_key_gradient']}/{total or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "A0/A1 have no active attention module; their attention measures are `n/a`. V6A smoke/preflight and first-epoch training telemetry record nonzero key/value gradients for the collaboration arms. Attention weights are not interpreted as effect utility.",
            "",
            "## Historical performance context",
            "",
            "These are historical validation results from different branch/configuration runs, not strict paired comparisons or retrained baselines.",
            "",
            "| Dataset | Historical model | Val Accuracy (%) | Val Macro-F1 (%) | Provenance note |",
            "|---|---|---:|---:|---|",
        ]
    )
    historical = {
        "Movies": {
            "V4A R0_raw": ((56.029, 0.385), (47.997, 1.637)),
            "PIGPR-C1 RGD": ((56.389, 0.170), (49.698, 0.281)),
        },
        "Grocery": {
            "V4A R0_raw": ((83.094, 0.325), (75.766, 1.387)),
            "PIGPR-C1 RGD": ((83.094, 0.193), (77.342, 0.228)),
        },
        "ele-fashion": {
            "V4A R0_raw": ((87.413, 0.128), (74.778, 1.560)),
            "PIGPR-C1 RGD": ((87.324, 0.042), (74.536, 0.252)),
        },
    }
    for dataset in DATASETS:
        for method, (acc, f1) in historical[dataset].items():
            source = (
                "V4A branch `exp/mvcge_mag_v4a_raw_anchored_residual_screen`; R0 Raw trajectory"
                if method.startswith("V4A")
                else "PIGPR-C1 commit `cd9d440`; RGD direct GPR"
            )
            lines.append(
                f"| {dataset} | {method} | {acc[0]:.3f} ± {acc[1]:.3f} | {f1[0]:.3f} ± {f1[1]:.3f} | {source} |"
            )
    lines.extend(
        [
            "",
            "The historical protocols/configurations differ from this frozen V6A campaign, so this table gives context only. V6A does not rerun Raw GPR or PIGPR-C1.",
            "",
            "## Resources and failures",
            "",
            f"- GPU: {env['selected_device']['device_name']} on `{env['device']}`, {env['selected_device']['total_memory_bytes']/1024**3:.1f} GiB total. At launch GPU0 had an unrelated VLLM process using about 23.1 GiB; GPU1 was selected and no other process was terminated.",
            "- Preflight peak: Movies/Grocery/ele-fashion allocated about "
            + "/".join(
                f"{r['forward_backward']['cuda_peak_allocated_bytes']/1024**3:.2f} GiB"
                for r in pre["datasets"]
            )
            + ". Formal peak per run is in `data/summary.csv` and `data/resource_profile.json`.",
            "- Full-graph execution used the standard unchunked propagation and no activation checkpointing or CPU feature staging. OOMs: none. No variant-specific budget or dimension changes were made.",
            "- Outputs and selected checkpoints remain under ignored `outputs/mace_mag_v6a_expert_collaboration/`; raw datasets and large model files are not committed.",
            "",
            "## Conclusion",
            "",
        ]
    )
    pair_a2_a1 = paired_index[("a2_cross-a1_node", "ALL")]
    pair_a2_a3 = paired_index[("a2_cross-a3_intra", "ALL")]
    a2_acc_ds = [paired_index[("a2_cross-a1_node", d)]["delta_val_accuracy_pp"] for d in DATASETS]
    a2_f1_ds = [paired_index[("a2_cross-a1_node", d)]["delta_val_macro_f1_pp"] for d in DATASETS]
    a2a3_acc_ds = [paired_index[("a2_cross-a3_intra", d)]["delta_val_accuracy_pp"] for d in DATASETS]
    lines.append(
        f"- A2−A1 averaged {pair_a2_a1['delta_val_accuracy_pp']:+.3f} pp Accuracy and {pair_a2_a1['delta_val_macro_f1_pp']:+.3f} pp Macro-F1 over nine paired runs; dataset Accuracy directions were {sum(v > 0 for v in a2_acc_ds)}/3 positive and dataset Macro-F1 directions {sum(v > 0 for v in a2_f1_ds)}/3 positive. This is the direct validation signal for adding cross-modal collaboration."
    )
    lines.append(
        f"- A2−A3 averaged {pair_a2_a3['delta_val_accuracy_pp']:+.3f} pp Accuracy and {pair_a2_a3['delta_val_macro_f1_pp']:+.3f} pp Macro-F1; dataset Accuracy directions were {sum(v > 0 for v in a2a3_acc_ds)}/3 positive. This compares cross-modal sourcing with a matched-capacity intra-modal attention control."
    )
    all_a2 = [summary_index[(d, "a2_cross")] for d in DATASETS]
    all_hist_comp = []
    for d in DATASETS:
        hist = historical[d]["PIGPR-C1 RGD"][0][0]
        all_hist_comp.append(
            100 * float(summary_index[(d, "a2_cross")]["val_accuracy_mean"]) - hist
        )
    lines.append(
        f"- A2 has the best Validation Accuracy on {sum(max(summary_index[(d,v)]['val_accuracy_mean'] for v in VARIANTS) == summary_index[(d,'a2_cross')]['val_accuracy_mean'] for d in DATASETS)}/3 datasets among the four V6A variants. Against historical PIGPR-C1 RGD, A2 Accuracy mean differences are {', '.join(f'{v:+.3f} pp' for v in all_hist_comp)} (context only, not strict paired evidence)."
    )
    if pair_a2_a1["delta_val_accuracy_pp"] > 0 and pair_a2_a1["delta_val_macro_f1_pp"] > 0 and pair_a2_a3["delta_val_accuracy_pp"] > 0:
        decision = "The descriptive validation evidence supports retaining expert-level cross-modal collaboration for a separately approved next study; it does not establish statistical significance or causal utility."
    elif pair_a2_a1["delta_val_accuracy_pp"] > 0 and pair_a2_a3["delta_val_accuracy_pp"] <= 0:
        decision = "A2 improves on A1 on average but does not outperform its matched-capacity A3 control on average; the evidence does not isolate cross-modal sourcing as the source of the gain. Treat the route as unresolved before any follow-on."
    else:
        decision = "The current validation evidence does not establish a positive average gain from expert-level cross-modal collaboration over A1. Keep the negative or mixed result and require human review before changing the collaboration level or launching another experiment."
    lines.append(f"- Decision: {decision}")
    lines.extend(
        [
            "",
            "These conclusions use Validation metrics only, three model seeds on fixed splits, and descriptive paired deltas. They are not significance tests or claims about Test generalization.",
            "",
            "## Artifacts",
            "",
            "- `DESIGN.md`, `README.md`, `data/environment.json`, `data/test_summary.json`, `data/preflight_summary.json`, `data/smoke_summary.json`",
            "- `data/campaign_manifest.json`, `data/run_rows.json`, `data/summary.csv`, `data/paired_comparisons.csv`",
            "- `data/resource_profile.json`, `data/collaboration_diagnostics.csv`",
            f"- Formal logs/checkpoints: `outputs/mace_mag_v6a_expert_collaboration/formal/`",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(device: str) -> None:
    run_rows = read_json(DATA_ROOT / "run_rows.json")
    if len(run_rows) != 36:
        raise RuntimeError(f"expected 36 formal rows, found {len(run_rows)}")
    if any(
        any(str(key).lower().startswith("test") for key in row["metrics"])
        for row in run_rows
    ):
        raise RuntimeError("Test metric key found in formal run rows")
    summary_rows, paired_rows = _summaries(run_rows)
    diag_rows = _audit_checkpoint_rows(run_rows, device)
    diagnostics_summary = _diagnostic_summary(diag_rows)
    write_csv(DATA_ROOT / "summary.csv", summary_rows)
    write_csv(DATA_ROOT / "paired_comparisons.csv", paired_rows)
    write_csv(DATA_ROOT / "collaboration_diagnostics.csv", diag_rows)

    env = read_json(DATA_ROOT / "environment.json")
    resource = {
        "device": env["device"],
        "device_name": env["selected_device"]["device_name"],
        "total_memory_bytes": env["selected_device"]["total_memory_bytes"],
        "initial_free_memory_bytes": env["selected_device"]["free_memory_bytes"],
        "initial_nvidia_smi": env["nvidia_smi_snapshot"],
        "other_processes_terminated": False,
        "memory_optimizations": {
            "edge_chunking": False,
            "activation_checkpointing": False,
            "cpu_feature_staging": False,
            "explanation": "Normal full-graph path fit all preflights and formal runs; no memory optimization was enabled.",
        },
        "preflight": read_json(DATA_ROOT / "preflight_summary.json")["datasets"],
        "formal_run_resources": [
            {
                "dataset": r["dataset"],
                "seed": r["seed"],
                "variant": r["variant"],
                "training_wall_seconds": r["metadata"].get("training_wall_seconds"),
                "mean_epoch_wall_seconds": r["metadata"].get("mean_epoch_wall_seconds"),
                "mean_train_step_seconds": r["metadata"].get("mean_train_step_seconds"),
                "cuda_peak_allocated_bytes": r["metadata"].get("cuda_peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": r["metadata"].get("cuda_peak_reserved_bytes"),
            }
            for r in run_rows
        ],
        "formal_oom_runs": [
            f
            for f in read_json(OUTPUT_ROOT / "formal" / "failures.json")
            if f.get("oom")
        ]
        if (OUTPUT_ROOT / "formal" / "failures.json").is_file()
        else [],
        "formal_failed_runs": read_json(OUTPUT_ROOT / "formal" / "failures.json")
        if (OUTPUT_ROOT / "formal" / "failures.json").is_file()
        else [],
    }
    write_json(DATA_ROOT / "resource_profile.json", resource)
    manifest = read_json(DATA_ROOT / "campaign_manifest.json")
    report_text = _report(
        summary_rows,
        paired_rows,
        diagnostics_summary,
        run_rows,
        manifest["provenance"]["freeze_commit_sha"],
    )
    (RESEARCH / "REPORT.md").write_text(report_text, encoding="utf-8")
    print("[V6A analysis] summaries and selected-checkpoint diagnostics saved", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    analyze(args.device)


if __name__ == "__main__":
    main()
