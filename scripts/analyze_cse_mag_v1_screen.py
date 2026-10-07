#!/usr/bin/env python3
"""Build tracked research artifacts from the CSE-MAG validation screen."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BRANCH = "exp/cse_mag_v1_local_global_screen"
FROZEN_SHA = "dc0e50c621f60ce802e3e27493ff6b8f45be911d"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    ("B0_independent", "independent"),
    ("B1_shared_static", "shared_static"),
    ("B2_mvcge_style", "mvcge_style"),
    ("B3_v1", "v1"),
)
DEFAULT_OUTPUT = ROOT / "outputs" / "cse_mag_v1_local_global_screen"
DEFAULT_RESEARCH = ROOT / "research" / "cse_mag_v1_local_global_screen"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_payload(device: str) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        index = torch.device(device).index or 0
        gpu = {
            "device": device,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        }
    nvidia_smi = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        nvidia_smi = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branch": BRANCH,
        "frozen_commit_sha": FROZEN_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_environment": "yhf_env",
        "packages": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "torch_geometric": package_version("torch-geometric"),
            "dgl": package_version("dgl"),
            "numpy": package_version("numpy"),
            "scikit_learn": package_version("scikit-learn"),
            "hydra_core": package_version("hydra-core"),
            "omegaconf": package_version("omegaconf"),
        },
        "selected_gpu": gpu,
        "nvidia_smi_gpu_rows": nvidia_smi,
        "screening": {
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "variants": [label for label, _ in VARIANTS],
            "protocol": "unified_full_graph_nc_v1",
            "task_evaluate_test": False,
            "selection": "best_validation_accuracy",
        },
    }


def parameter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in DATASETS:
        for label, variant in VARIANTS:
            cell = [
                row
                for row in rows
                if row["dataset"] == dataset and row["label"] == label
            ]
            if not cell:
                continue
            metrics = [row["metrics"] for row in cell]
            metadata = [row["metadata"] for row in cell]
            model_params = {int(item["model_parameters"]) for item in metadata}
            classifier_params = {int(item["classifier_parameters"]) for item in metadata}
            if len(model_params) != 1 or len(classifier_params) != 1:
                raise ValueError(f"Parameter counts varied within {dataset}/{label}")
            model_trainable = model_params.pop()
            classifier = classifier_params.pop()
            result.append(
                {
                    "dataset": dataset,
                    "variant_label": label,
                    "variant": variant,
                    "num_runs": len(cell),
                    "val_acc_mean": mean([float(m["val_acc"]) for m in metrics]),
                    "val_acc_std": pstdev([float(m["val_acc"]) for m in metrics]),
                    "val_macro_f1_mean": mean([float(m["val_macro_f1"]) for m in metrics]),
                    "val_macro_f1_std": pstdev([float(m["val_macro_f1"]) for m in metrics]),
                    "mean_best_epoch": mean([float(m["best_epoch"]) for m in metadata]),
                    "model_trainable_params": model_trainable,
                    "classifier_params": classifier,
                    "total_trainable_params": model_trainable + classifier,
                }
            )
    return result


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "variant_label",
        "variant",
        "num_runs",
        "val_acc_mean",
        "val_acc_std",
        "val_macro_f1_mean",
        "val_macro_f1_std",
        "mean_best_epoch",
        "model_trainable_params",
        "classifier_params",
        "total_trainable_params",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["dataset"], int(row["seed"]), row["label"]): row for row in rows
    }
    output = []
    for baseline, _ in VARIANTS[:-1]:
        pairs: list[tuple[float, float]] = []
        per_dataset: dict[str, list[tuple[float, float]]] = {}
        for dataset in DATASETS:
            deltas = []
            for seed in SEEDS:
                v1 = indexed.get((dataset, seed, "B3_v1"))
                control = indexed.get((dataset, seed, baseline))
                if v1 is None or control is None:
                    continue
                delta = (
                    float(v1["metrics"]["val_acc"])
                    - float(control["metrics"]["val_acc"]),
                    float(v1["metrics"]["val_macro_f1"])
                    - float(control["metrics"]["val_macro_f1"]),
                )
                pairs.append(delta)
                deltas.append(delta)
            per_dataset[dataset] = deltas

        row: dict[str, Any] = {
            "comparison": f"B3_v1-minus-{baseline}",
            "paired_runs": len(pairs),
            "delta_val_acc_mean": mean([x[0] for x in pairs]),
            "delta_val_macro_f1_mean": mean([x[1] for x in pairs]),
            "positive_acc_pairs": sum(x[0] > 0 for x in pairs),
            "positive_acc_pairs_over_9": f"{sum(x[0] > 0 for x in pairs)}/9",
            "positive_f1_pairs": sum(x[1] > 0 for x in pairs),
            "positive_f1_pairs_over_9": f"{sum(x[1] > 0 for x in pairs)}/9",
        }
        for dataset in DATASETS:
            deltas = per_dataset[dataset]
            stem = dataset.replace("-", "_")
            row[f"{stem}_delta_val_acc_mean"] = mean([x[0] for x in deltas])
            row[f"{stem}_delta_val_macro_f1_mean"] = mean([x[1] for x in deltas])
            row[f"{stem}_positive_acc_seeds"] = sum(x[0] > 0 for x in deltas)
            row[f"{stem}_positive_acc_seeds_over_3"] = f"{sum(x[0] > 0 for x in deltas)}/3"
            row[f"{stem}_positive_f1_seeds"] = sum(x[1] > 0 for x in deltas)
            row[f"{stem}_positive_f1_seeds_over_3"] = f"{sum(x[1] > 0 for x in deltas)}/3"
        output.append(row)
    return output


def write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["comparison"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_b2_checkpoint(output_dir: Path) -> dict[str, Any]:
    """Check actual Top-K coverage and gradient flow on the Movies B2 checkpoint."""
    from src.data import load_mag_data
    from src.models.cse_mag_v1 import Model

    run_dir = output_dir / "runs" / "Movies" / "seed_42" / "B2_mvcge_style"
    cfg_path = run_dir / "hydra" / ".hydra" / "config.yaml"
    checkpoint_path = run_dir / "best.pt"
    cfg = OmegaConf.load(cfg_path)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    data = load_mag_data(cfg, "nc", 42)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    parameter_counts = {}
    for label, variant in VARIANTS:
        variant_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        variant_cfg.model.variant = variant
        variant_model = Model(variant_cfg, data_info)
        parameter_counts[label] = {
            "model_total_parameters_including_frozen_modules": sum(
                parameter.numel() for parameter in variant_model.parameters()
            ),
            "model_trainable_parameters": sum(
                parameter.numel()
                for parameter in variant_model.parameters()
                if parameter.requires_grad
            ),
            "classifier_parameters": 257 * int(data.num_classes),
        }
    model = Model(cfg, data_info)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    torch.manual_seed(42)
    model.train()
    model.zero_grad(set_to_none=True)
    z, _, _, aux_loss, info = model(data.x, data.edge_index, return_details=True)
    finite = bool(torch.isfinite(z).all() and torch.isfinite(aux_loss))
    (z.square().mean() + aux_loss).backward()
    num_experts = int(model.naive_num_experts)
    selected = torch.zeros(num_experts, dtype=torch.long)
    for modality in ("text", "visual"):
        indices = info["details"][modality]["top_indices"].detach().cpu().reshape(-1)
        selected += torch.bincount(indices, minlength=num_experts)
    grad_norms = []
    for expert in model.naive_experts:
        grads = [p.grad for p in expert.parameters() if p.grad is not None]
        grad_norms.append(
            float(torch.sqrt(sum(g.detach().square().sum() for g in grads)).item())
            if grads
            else 0.0
        )
    return {
        "evidence_source": "formal Movies seed 42 B2 best checkpoint, full graph, train-mode forward/backward",
        "test_metrics_computed": False,
        "finite_embedding_and_aux_loss": finite,
        "expert_count": num_experts,
        "top_k": int(model.naive_top_k),
        "selected_assignments_by_expert_across_both_modalities": selected.tolist(),
        "expert_gradient_l2_norms": grad_norms,
        "all_experts_selected": bool((selected > 0).all()),
        "all_experts_have_nonzero_gradient": bool(all(norm > 0 for norm in grad_norms)),
        "balance_loss_finite": bool(torch.isfinite(aux_loss)),
        "parameter_counts_by_variant": parameter_counts,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def report_markdown(
    manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    env: dict[str, Any],
    b2_audit: dict[str, Any],
) -> str:
    by_cell = {(row["dataset"], row["variant_label"]): row for row in summary}
    overall_acc = {
        label: mean(
            [float(row["metrics"]["val_acc"]) for row in manifest["run_rows"] if row["label"] == label]
        )
        for label, _ in VARIANTS
    }
    dataset_winners = {}
    for dataset in DATASETS:
        candidates = [by_cell[(dataset, label)] for label, _ in VARIANTS if (dataset, label) in by_cell]
        max_acc = max((float(row["val_acc_mean"]) for row in candidates), default=float("nan"))
        dataset_winners[dataset] = [row["variant_label"] for row in candidates if float(row["val_acc_mean"]) == max_acc]

    lines = [
        "# CSE-MAG V1 Local/Global Screening Report",
        "",
        "## 1. Provenance and protocol",
        "",
        f"- Branch: `{BRANCH}`",
        f"- Frozen commit SHA: `{FROZEN_SHA}`",
        f"- Environment: Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}",
        f"- GPU: {env.get('selected_gpu', {}).get('name', 'unavailable')} (`{env.get('selected_gpu', {}).get('device', 'n/a')}`)",
        f"- Datasets: {', '.join(DATASETS)}; seeds: {', '.join(map(str, SEEDS))}",
        "- Protocol: `unified_full_graph_nc_v1`; checkpoint selected by best Validation Accuracy.",
        "- Test evaluation: **false** for the smoke and all campaign runs.",
        f"- Completed runs: {manifest.get('completed_or_reused_runs', 0)} / {manifest.get('expected_runs', 36)}; failures: {len(manifest.get('failures', []))}.",
        "",
        "## 2. Code and smoke audit",
        "",
        "- Targeted tests: `4 passed`; full `tests/`: `4 passed` (one upstream PyG deprecation warning).",
        "- Test invocation note: the `pytest` shell entry point initially resolved to the host Python 3.8 install; both requested test runs passed via `conda run -n yhf_env python -m pytest`.",
        "- Smoke: Movies, seed 42, four variants, one epoch each; all four completed, saved checkpoints and validation-only `run_metrics.json`, with no OOM/NaN/Inf.",
        "- Smoke ran with the launcher's `--allow-dirty` guard for a test-only fairness assertion; model and configuration files were unchanged. That test change was committed as the frozen checkpoint before the formal campaign.",
        "- Initialization fairness: same-seed Text/Visual projectors, all fusion weights, and downstream classifier are tensor-equal across variants; tested.",
        "- V1 prior initialization: Local/Global mixture `(0.5, 0.5)` and structural gate `sigmoid(-2) = 0.1192029`; tested.",
        "- Isolated-node prior preservation: Local and Global displacements are zero and V1 output equals its intrinsic prior; tested.",
        f"- B2 Top-K audit on the post-campaign Movies/seed 42 best checkpoint: selected {b2_audit['top_k']} of {b2_audit['expert_count']} experts per modality/node; all experts selected = `{b2_audit['all_experts_selected']}`, all experts had nonzero gradient = `{b2_audit['all_experts_have_nonzero_gradient']}`.",
        "- Graph operator removes self-loops; dataset configuration also has `add_self_loops: false`. No implementation or scientific-model changes were needed; the only code change was the initialization fairness test extension.",
        "- B2 is an MvCGE-style direct-transfer control, not an exact MvCGE reproduction.",
        "",
        "## 3. Main validation results",
        "",
        "| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Model trainable params | Classifier params |",
        "|---|---|---:|---:|---:|---:|",
    ]
    failures = manifest.get("failures", [])
    if failures:
        failure_lines = ["- Campaign failures:"]
        for failure in failures:
            failure_lines.append(
                f"  - {failure.get('dataset')} seed {failure.get('seed')} "
                f"{failure.get('label')}: `{failure.get('error')}`"
            )
    else:
        failure_lines = ["- Campaign failures: none."]
    section_three = lines.index("## 3. Main validation results")
    lines[section_three:section_three] = failure_lines + [""]
    for dataset in DATASETS:
        for label, _ in VARIANTS:
            row = by_cell.get((dataset, label))
            if row is None:
                continue
            lines.append(
                f"| {dataset} | {label} | {pct(row['val_acc_mean'])} ± {pct(row['val_acc_std'])} | "
                f"{pct(row['val_macro_f1_mean'])} ± {pct(row['val_macro_f1_std'])} | "
                f"{row['model_trainable_params']:,} | {row['classifier_params']:,} |"
            )

    lines.extend(["", "## 4. Paired V1 comparisons", ""])
    lines.extend(
        [
            "| Comparison | Overall Δ Val Accuracy | Overall Δ Val Macro-F1 | Positive accuracy pairs | Positive F1 pairs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    paired_index = {row["comparison"]: row for row in paired}
    for baseline, _ in VARIANTS[:-1]:
        row = paired_index[f"B3_v1-minus-{baseline}"]
        lines.append(
            f"| B3_v1 − {baseline} | {row['delta_val_acc_mean']:+.5f} | "
            f"{row['delta_val_macro_f1_mean']:+.5f} | {row['positive_acc_pairs']}/9 | {row['positive_f1_pairs']}/9 |"
        )
    lines.extend(
        [
            "",
            "Dataset-level paired means and positive seeds:",
            "",
            "| Comparison | Dataset | Δ Val Accuracy mean | Δ Val Macro-F1 mean | Positive accuracy seeds | Positive F1 seeds |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for baseline, _ in VARIANTS[:-1]:
        row = paired_index[f"B3_v1-minus-{baseline}"]
        for dataset in DATASETS:
            stem = dataset.replace("-", "_")
            lines.append(
                f"| B3_v1 − {baseline} | {dataset} | {row[f'{stem}_delta_val_acc_mean']:+.5f} | "
                f"{row[f'{stem}_delta_val_macro_f1_mean']:+.5f} | "
                f"{row[f'{stem}_positive_acc_seeds']}/3 | {row[f'{stem}_positive_f1_seeds']}/3 |"
            )

    lines.extend(["", "## 5. Observations", ""])
    for dataset, winners in dataset_winners.items():
        lines.append(f"- Highest mean Validation Accuracy on {dataset}: {', '.join(winners)}.")
    for baseline, _ in VARIANTS[:-1]:
        wins = overall_acc["B3_v1"] > overall_acc[baseline]
        lines.append(
            f"- Across all paired runs, B3_v1 overall mean Validation Accuracy versus {baseline}: "
            f"{overall_acc['B3_v1'] - overall_acc[baseline]:+.5f}; it is higher = `{wins}`."
        )
    for baseline, _ in VARIANTS[:-1]:
        paired_row = paired_index[f"B3_v1-minus-{baseline}"]
        lines.append(
            f"- B3_v1 overall mean Validation Macro-F1 delta versus {baseline}: "
            f"{paired_row['delta_val_macro_f1_mean']:+.5f}."
        )
    movies_b2 = paired_index["B3_v1-minus-B2_mvcge_style"]
    lines.append(
        f"- On Movies, B3_v1's mean Validation Accuracy delta versus B2 is "
        f"{movies_b2['Movies_delta_val_acc_mean']:+.5f}, while its mean Macro-F1 delta is "
        f"{movies_b2['Movies_delta_val_macro_f1_mean']:+.5f}."
    )
    counts = {
        label: int(next(row["model_trainable_params"] for row in summary if row["variant_label"] == label))
        for label, _ in VARIANTS
        if any(row["variant_label"] == label for row in summary)
    }
    lines.append(
        "- Model trainable parameter counts differ by variant: "
        + ", ".join(f"{label} {count:,}" for label, count in counts.items())
        + ". Classifier size is 5,140 parameters for every variant."
    )
    lines.append("- These are descriptive validation results from a coarse screen; no significance tests were run.")

    lines.extend(
        [
            "",
            "## 6. Decision map",
            "",
            "Conditions A–D use the overall mean Validation Accuracy across the nine paired runs. For E, ‘clearly best’ means uniquely highest dataset-level mean Validation Accuracy on all three datasets.",
            "",
        ]
    )
    b3 = overall_acc["B3_v1"]
    checks = [
        (b3 > overall_acc["B1_shared_static"] and b3 >= overall_acc["B0_independent"], "A", "Shared capability plus conditional utilization is worth continuing."),
        (b3 > overall_acc["B1_shared_static"] and overall_acc["B0_independent"] > b3, "B", "Adaptation may help, while full expert sharing may be too strong; consider a shared core with lightweight modality adaptation next."),
        (overall_acc["B1_shared_static"] >= b3, "C", "Node-specific routing has not shown value; do not further complicate the router yet."),
        (overall_acc["B2_mvcge_style"] >= b3, "D", "The Local/Global functional expert redesign has not exceeded generic MvCGE-style sharing; inspect response space before changing the router."),
        (all("B0_independent" in dataset_winners[d] and len(dataset_winners[d]) == 1 for d in DATASETS), "E", "B0 is the unique highest-accuracy variant on all three dataset means; revisit the fully shared structural-expert assumption."),
    ]
    triggered = [f"- **{key}.** {text}" for condition, key, text in checks if condition]
    lines.extend(triggered or ["- None of A–E triggers under the stated Validation Accuracy comparisons; retain the coarse-screen boundary and inspect the reported metric tradeoffs."])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--skip-b2-route-audit", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    research_dir = args.research_dir.resolve()
    manifest = read_json(output_dir / "campaign_manifest.json")
    run_rows = read_json(output_dir / "run_rows.json")
    if manifest.get("mode") != "campaign":
        raise RuntimeError("output campaign_manifest.json is not from --campaign")
    if manifest.get("branch") != BRANCH or manifest.get("base_sha") != "579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e":
        raise RuntimeError("campaign provenance does not match the frozen branch/base")
    if manifest.get("evaluate_test") is not False:
        raise RuntimeError("campaign manifest does not confirm evaluate_test=false")
    for row in run_rows:
        if any(key.startswith("test_") for key in row.get("metrics", {})):
            raise RuntimeError("test metrics found in validation campaign rows")
        for value in row.get("metrics", {}).values():
            if not math.isfinite(float(value)):
                raise RuntimeError("non-finite campaign metric found")
    if not args.skip_b2_route_audit:
        b2_audit = audit_b2_checkpoint(output_dir)
    else:
        b2_audit = {"skipped": True}

    data_dir = research_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_json(data_dir / "campaign_manifest.json", manifest)
    write_json(data_dir / "run_rows.json", run_rows)
    env = environment_payload(args.device)
    env["formal_campaign_completed_runs"] = manifest.get("completed_or_reused_runs", 0)
    env["formal_campaign_expected_runs"] = manifest.get("expected_runs", 36)
    write_json(data_dir / "environment.json", env)

    summary = parameter_rows(run_rows)
    paired = paired_rows(run_rows)
    write_summary_csv(data_dir / "summary.csv", summary)
    write_paired_csv(data_dir / "paired_comparisons.csv", paired)

    smoke_path = data_dir / "smoke_summary.json"
    smoke = read_json(smoke_path) if smoke_path.exists() else {}
    smoke["campaign_b2_routing_audit"] = b2_audit
    if "parameter_counts_by_variant" in b2_audit:
        smoke["verified_parameter_counts"] = b2_audit["parameter_counts_by_variant"]
    write_json(smoke_path, smoke)
    report = report_markdown(
        {**manifest, "run_rows": run_rows}, summary, paired, env, b2_audit
    )
    (research_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"Wrote research artifacts: {research_dir} "
        f"({manifest.get('completed_or_reused_runs', 0)}/{manifest.get('expected_runs', 36)} runs)"
    )


if __name__ == "__main__":
    main()
