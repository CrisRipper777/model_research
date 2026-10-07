#!/usr/bin/env python3
"""Analyze SOSB-MAG V1.5 campaigns and legacy CSE-MAG checkpoints.

Diagnostics read features, physical graph edges, checkpoint weights and
validation-selected run metadata. They do not consult label fields, index Test
labels, or compute any Test metric.
"""

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

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
LEGACY_VARIANTS = (("B1_shared_static", "shared_static"), ("B3_v1", "v1"))
DEFAULT_OLD_OUTPUT = ROOT / "outputs" / "cse_mag_v1_local_global_screen"
DEFAULT_RESEARCH = ROOT / "research" / "sosb_mag_v15_basis_screen"
DEFAULT_CAMPAIGN_OUTPUT = ROOT / "outputs" / "sosb_mag_v15_basis_screen" / "campaign"
NEW_VARIANTS = (
    ("A0_legacy_lg", "A0_legacy_lg"),
    ("A1_rawpoly_shared", "A1_rawpoly_shared"),
    ("A2_sosb_shared", "A2_sosb_shared"),
    ("A3_sosb_modality", "A3_sosb_modality"),
)
PAIRINGS = (
    ("A1_rawpoly_shared", "A0_legacy_lg"),
    ("A2_sosb_shared", "A1_rawpoly_shared"),
    ("A2_sosb_shared", "A0_legacy_lg"),
    ("A3_sosb_modality", "A2_sosb_shared"),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _resolve(path_like: str | Path) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    candidate = ROOT / path
    return candidate.resolve()


def _compose_config(dataset: str, seed: int):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=cse_mag_v1",
                f"seed={seed}",
                "task.evaluate_test=false",
            ],
        )


def load_features_and_edges(dataset: str, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the graph and modality features without loading labels/splits."""
    cfg = _compose_config(dataset, seed)
    ds = cfg.dataset
    if str(ds.source).lower() == "magb":
        import dgl

        import_graphs, _ = dgl.load_graphs(str(_resolve(ds.graph_path)))
        if not import_graphs:
            raise ValueError(f"No graph found for {dataset}")
        graph = import_graphs[0]
        # Deliberately access only edges and node count; graph labels are unused.
        src, dst = graph.edges()
        edge_index = torch.stack([src.long(), dst.long()], dim=0)
        num_nodes = int(graph.num_nodes())
        text = torch.from_numpy(
            np.load(_resolve(ds.text_feat_path), allow_pickle=False).astype(np.float32, copy=False)
        )
        visual = torch.from_numpy(
            np.load(_resolve(ds.image_feat_path), allow_pickle=False).astype(np.float32, copy=False)
        )
        x = torch.cat([text, visual], dim=1).contiguous()
    elif str(ds.source).lower() == "mmgraph":
        x = torch.load(_resolve(ds.joint_feat_path), map_location="cpu", weights_only=False)
        x = torch.as_tensor(x, dtype=torch.float32).contiguous()
        edge_raw = torch.load(_resolve(ds.edge_path), map_location="cpu", weights_only=False)
        edge_index = torch.as_tensor(edge_raw, dtype=torch.long)
        if edge_index.dim() != 2:
            raise ValueError(f"Invalid edge tensor for {dataset}: {tuple(edge_index.shape)}")
        if edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t()
        num_nodes = int(x.size(0))
    else:
        raise ValueError(f"Unsupported source for label-free diagnostics: {ds.source}")

    edge_index, _ = torch_geometric_remove_self_loops(edge_index)
    edge_index = torch_geometric_to_undirected(edge_index, num_nodes)
    if x.size(0) != num_nodes:
        raise ValueError(f"Feature/node mismatch for {dataset}: {x.size(0)} != {num_nodes}")
    return x, edge_index.contiguous()


def torch_geometric_remove_self_loops(edge_index: torch.Tensor) -> tuple[torch.Tensor, None]:
    from torch_geometric.utils import remove_self_loops

    return remove_self_loops(edge_index)


def torch_geometric_to_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    from torch_geometric.utils import to_undirected

    return to_undirected(edge_index, num_nodes=num_nodes)


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().reshape(-1).float()
    if values.numel() == 0:
        return {key: float("nan") for key in ("mean", "std", "p10", "p50", "p90")}
    qs = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p10": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p90": float(qs[2].item()),
    }


def _cosine_metrics(a: torch.Tensor, b: torch.Tensor, active: torch.Tensor) -> dict[str, float]:
    aa = a[active].float()
    bb = b[active].float()
    node_cos = F.cosine_similarity(aa, bb, dim=-1, eps=1.0e-8)
    flat_cos = F.cosine_similarity(aa.reshape(-1), bb.reshape(-1), dim=0, eps=1.0e-8)
    return {
        "mean_node_cosine": float(node_cos.mean().item()),
        "flattened_cosine": float(flat_cos.item()),
        "rms_a": float(torch.sqrt(torch.mean(aa.square())).item()),
        "rms_b": float(torch.sqrt(torch.mean(bb.square())).item()),
        "node_cosine_std": float(node_cos.std(unbiased=False).item()),
        "node_cosine_p50": float(torch.quantile(node_cos, 0.5).item()),
        "node_cosine_p90": float(torch.quantile(node_cos, 0.9).item()),
    }


def _js_per_node(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    midpoint = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - midpoint.log())).sum(dim=-1) + 0.5 * (
        q * (q.log() - midpoint.log())
    ).sum(dim=-1)


@torch.no_grad()
def diagnose_one_legacy_checkpoint(
    dataset: str,
    seed: int,
    label: str,
    variant: str,
    output_root: Path,
    device: torch.device,
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    from src.models.cse_mag_v1 import Model

    checkpoint_path = output_root / "runs" / dataset / f"seed_{seed}" / label / "best.pt"
    hydra_config_path = (
        output_root / "runs" / dataset / f"seed_{seed}" / label / "hydra" / ".hydra" / "config.yaml"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_info = checkpoint["data_info"]
    cfg = OmegaConf.load(hydra_config_path)
    cfg.model.variant = variant
    model = Model(cfg, data_info)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    if cached_data is None:
        cached_data = load_features_and_edges(dataset, seed)
    x_cpu, edge_cpu = cached_data
    x = x_cpu.to(device)
    edge_index = edge_cpu.to(device)
    z, _, _, _, info = model(x, edge_index, return_details=True)
    if not torch.isfinite(z).all():
        raise FloatingPointError(f"Non-finite embeddings in {dataset}/{seed}/{label}")

    num_nodes = int(x.size(0))
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))
    active = degree > 0
    detail = info["details"]
    modalities: dict[str, Any] = {}
    for modality in ("text", "visual"):
        item = detail[modality]
        modalities[modality] = {
            "routing_local_probability": _stats(item["routing"][:, 0]),
            "routing_global_probability": _stats(item["routing"][:, 1]),
            "structural_gate": _stats(item["gate"]),
            "raw_response_cosine": _cosine_metrics(
                item["local_delta"], item["global_delta"], active
            ),
            "expert_response_cosine": _cosine_metrics(
                item["local_response"], item["global_response"], active
            ),
        }

    text_routing = detail["text"]["routing"]
    visual_routing = detail["visual"]["routing"]
    js = _js_per_node(text_routing, visual_routing)
    if label == "B1_shared_static":
        first = detail["text"]
        learned = {
            "local_weight": float(first["routing"][0, 0].item()),
            "global_weight": float(first["routing"][0, 1].item()),
            "structural_gate": float(first["gate"][0, 0].item()),
        }
    else:
        learned = {
            "text_local_weight": modalities["text"]["routing_local_probability"],
            "visual_local_weight": modalities["visual"]["routing_local_probability"],
            "text_structural_gate": modalities["text"]["structural_gate"],
            "visual_structural_gate": modalities["visual"]["structural_gate"],
            "per_node_text_visual_routing_js": _stats(js),
        }

    result = {
        "dataset": dataset,
        "seed": seed,
        "variant_label": label,
        "variant": variant,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "num_nodes": num_nodes,
        "active_nodes": int(active.sum().item()),
        "isolated_nodes": int((~active).sum().item()),
        "learned_composition": learned,
        "modalities": modalities,
        "label_access": {
            "label_fields_consulted": False,
            "split_indices_loaded": False,
            "test_labels_accessed": False,
            "test_metrics_computed": False,
        },
    }
    del model, x, edge_index, z, info
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_legacy_diagnostics(output_root: Path, research_dir: Path, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    expected = [
        (dataset, seed, label, variant)
        for dataset in DATASETS
        for seed in SEEDS
        for label, variant in LEGACY_VARIANTS
    ]
    missing = []
    available = []
    for dataset, seed, label, variant in expected:
        checkpoint = output_root / "runs" / dataset / f"seed_{seed}" / label / "best.pt"
        if checkpoint.is_file():
            available.append((dataset, seed, label, variant))
        else:
            missing.append(str(checkpoint))

    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "available" if len(available) == len(expected) else ("partial" if available else "unavailable"),
        "expected_checkpoints": len(expected),
        "available_checkpoints": len(available),
        "missing_checkpoints": missing,
        "device": device_name,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "variants": [label for label, _ in LEGACY_VARIANTS],
        "label_policy": "Only feature arrays, edge indices, checkpoint weights, and model metadata were consulted. No label fields were read/indexed and no split indices or Test metrics were used. MAGB graph deserialization may materialize node fields internally, but this script accesses only graph edges and node count.",
        "diagnostics": [],
    }
    if not available:
        write_json(research_dir / "data" / "legacy_v1_diagnostics.json", payload)
        write_legacy_markdown(research_dir / "LEGACY_DIAGNOSTIC.md", payload)
        return payload

    data_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for dataset, seed, label, variant in available:
        if dataset not in data_cache:
            data_cache[dataset] = load_features_and_edges(dataset, seed)
        result = diagnose_one_legacy_checkpoint(
            dataset,
            seed,
            label,
            variant,
            output_root,
            device,
            data_cache[dataset],
        )
        payload["diagnostics"].append(result)

    write_json(research_dir / "data" / "legacy_v1_diagnostics.json", payload)
    write_legacy_markdown(research_dir / "LEGACY_DIAGNOSTIC.md", payload)
    return payload


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_legacy_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Legacy CSE-MAG V1 Diagnostics",
        "",
        f"- Status: **{payload['status']}** ({payload['available_checkpoints']}/{payload['expected_checkpoints']} checkpoints).",
        f"- Device: `{payload['device']}`.",
        f"- Label policy: {payload['label_policy']}",
        "- This is descriptive checkpoint analysis only: no shuffle, intervention, retraining, or causal claim.",
        "",
    ]
    if payload["missing_checkpoints"]:
        lines.append("Missing checkpoints:")
        lines.extend(f"- `{path}`" for path in payload["missing_checkpoints"])
        lines.append("")
    if not payload["diagnostics"]:
        lines.append("No checkpoint diagnostics were computed.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "| Dataset | Seed | Variant | wL | wG | Gate | Text local p mean | Visual local p mean | Text gate mean | Visual gate mean | routing JS mean |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["diagnostics"]:
        learned = item["learned_composition"]
        if item["variant_label"] == "B1_shared_static":
            wl = _fmt(learned["local_weight"])
            wg = _fmt(learned["global_weight"])
            gate = _fmt(learned["structural_gate"])
            js = "—"
        else:
            wl = wg = gate = "—"
            js = _fmt(learned["per_node_text_visual_routing_js"]["mean"])
        text = item["modalities"]["text"]
        visual = item["modalities"]["visual"]
        lines.append(
            f"| {item['dataset']} | {item['seed']} | {item['variant_label']} | {wl} | {wg} | {gate} | "
            f"{_fmt(text['routing_local_probability']['mean'])} | "
            f"{_fmt(visual['routing_local_probability']['mean'])} | "
            f"{_fmt(text['structural_gate']['mean'])} | {_fmt(visual['structural_gate']['mean'])} | {js} |"
        )

    lines.extend(
        [
            "",
            "## Response-space cosine and scale",
            "",
            "All response comparisons use active nodes only; cosine distributions are per node, and flattened cosine/RMS use the same active-node subset.",
            "",
            "| Dataset | Seed | Variant | Modality | cos(DL,DG) node mean | cos(DL,DG) flattened | RMS(DL) | RMS(DG) | cos(RL,RG) node mean | cos(RL,RG) flattened | RMS(RL) | RMS(RG) |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["diagnostics"]:
        for modality, values in item["modalities"].items():
            raw = values["raw_response_cosine"]
            expert = values["expert_response_cosine"]
            lines.append(
                f"| {item['dataset']} | {item['seed']} | {item['variant_label']} | {modality} | "
                f"{_fmt(raw['mean_node_cosine'])} | {_fmt(raw['flattened_cosine'])} | {_fmt(raw['rms_a'])} | {_fmt(raw['rms_b'])} | "
                f"{_fmt(expert['mean_node_cosine'])} | {_fmt(expert['flattened_cosine'])} | {_fmt(expert['rms_a'])} | {_fmt(expert['rms_b'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _package_version(name: str) -> str | None:
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
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        smi = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "conda_environment": "yhf_env",
        "packages": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "torch_geometric": _package_version("torch-geometric"),
            "dgl": _package_version("dgl"),
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "hydra_core": _package_version("hydra-core"),
            "omegaconf": _package_version("omegaconf"),
        },
        "selected_gpu": gpu,
        "nvidia_smi_gpu_rows": smi,
        "screening": {
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "variants": [label for label, _ in NEW_VARIANTS],
            "protocol": "unified_full_graph_nc_v1",
            "task_evaluate_test": False,
            "selection": "best_validation_accuracy",
        },
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in DATASETS:
        for label, variant in NEW_VARIANTS:
            cell = [row for row in rows if row["dataset"] == dataset and row["label"] == label]
            if not cell:
                continue
            metadata = [row["metadata"] for row in cell]
            model_counts = {int(item["model_parameters"]) for item in metadata}
            classifier_counts = {int(item["classifier_parameters"]) for item in metadata}
            if len(model_counts) != 1 or len(classifier_counts) != 1:
                raise ValueError(f"Trainable parameter counts varied within {dataset}/{label}")
            model_params = next(iter(model_counts))
            classifier_params = next(iter(classifier_counts))
            result.append(
                {
                    "dataset": dataset,
                    "variant_label": label,
                    "variant": variant,
                    "num_runs": len(cell),
                    "val_acc_mean": _mean([float(row["metrics"]["val_acc"]) for row in cell]),
                    "val_acc_std": _pstdev([float(row["metrics"]["val_acc"]) for row in cell]),
                    "val_macro_f1_mean": _mean([float(row["metrics"]["val_macro_f1"]) for row in cell]),
                    "val_macro_f1_std": _pstdev([float(row["metrics"]["val_macro_f1"]) for row in cell]),
                    "mean_best_epoch": _mean([float(item["best_epoch"]) for item in metadata]),
                    "model_trainable_params": model_params,
                    "classifier_params": classifier_params,
                    "total_trainable_params": model_params + classifier_params,
                }
            )
    return result


def build_paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["dataset"], int(row["seed"]), row["label"]): row for row in rows}
    output = []
    for target_label, baseline_label in PAIRINGS:
        overall: list[tuple[float, float]] = []
        by_dataset: dict[str, list[tuple[float, float]]] = {}
        for dataset in DATASETS:
            deltas = []
            for seed in SEEDS:
                target = indexed.get((dataset, seed, target_label))
                baseline = indexed.get((dataset, seed, baseline_label))
                if target is None or baseline is None:
                    continue
                delta = (
                    float(target["metrics"]["val_acc"]) - float(baseline["metrics"]["val_acc"]),
                    float(target["metrics"]["val_macro_f1"]) - float(baseline["metrics"]["val_macro_f1"]),
                )
                overall.append(delta)
                deltas.append(delta)
            by_dataset[dataset] = deltas
        row: dict[str, Any] = {
            "comparison": f"{target_label}-minus-{baseline_label}",
            "paired_runs": len(overall),
            "delta_val_acc_mean": _mean([item[0] for item in overall]),
            "delta_val_macro_f1_mean": _mean([item[1] for item in overall]),
            "positive_acc_pairs": sum(item[0] > 0 for item in overall),
            "positive_acc_pairs_over_9": f"{sum(item[0] > 0 for item in overall)}/9",
            "positive_f1_pairs": sum(item[1] > 0 for item in overall),
            "positive_f1_pairs_over_9": f"{sum(item[1] > 0 for item in overall)}/9",
        }
        for dataset in DATASETS:
            deltas = by_dataset[dataset]
            stem = dataset.replace("-", "_")
            row[f"{stem}_delta_val_acc_mean"] = _mean([item[0] for item in deltas])
            row[f"{stem}_delta_val_macro_f1_mean"] = _mean([item[1] for item in deltas])
            row[f"{stem}_positive_acc_seeds"] = sum(item[0] > 0 for item in deltas)
            row[f"{stem}_positive_acc_seeds_over_3"] = f"{sum(item[0] > 0 for item in deltas)}/3"
            row[f"{stem}_positive_f1_seeds"] = sum(item[1] > 0 for item in deltas)
            row[f"{stem}_positive_f1_seeds_over_3"] = f"{sum(item[1] > 0 for item in deltas)}/3"
        output.append(row)
    return output


def _condition_number(gram: torch.Tensor) -> tuple[float, float]:
    symmetric = 0.5 * (gram + gram.t())
    eigenvalues = torch.linalg.eigvalsh(symmetric.double())
    jitter = 1.0e-6 * max(1.0, abs(float(torch.trace(symmetric).item())) / symmetric.size(0))
    return float(((eigenvalues.max() + jitter) / (eigenvalues.min() + jitter)).item()), jitter


@torch.no_grad()
def analyze_selected_checkpoint(
    dataset: str,
    seed: int,
    label: str,
    variant: str,
    output_root: Path,
    device: torch.device,
    cached_data: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from src.models.sosb_mag_v15 import Model

    run_dir = output_root / "runs" / dataset / f"seed_{seed}" / label
    checkpoint_path = run_dir / "best.pt"
    cfg_path = run_dir / "hydra" / ".hydra" / "config.yaml"
    if not checkpoint_path.is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"Missing selected checkpoint/config for {dataset}/{seed}/{label}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if any(key.lower().startswith("test") for key in checkpoint.get("metrics", {})):
        raise RuntimeError(f"Test metrics found in selected checkpoint: {checkpoint_path}")
    cfg = OmegaConf.load(cfg_path)
    cfg.model.variant = variant
    model = Model(cfg, checkpoint["data_info"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    if cached_data is None:
        cached_data = load_features_and_edges(dataset, seed)
    x_cpu, edge_cpu = cached_data
    z, _, _, _, info = model(x_cpu.to(device), edge_cpu.to(device), return_details=True)
    if not torch.isfinite(z).all():
        raise FloatingPointError(f"Non-finite embedding in {dataset}/{seed}/{label}")

    basis_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    active = info["active_nodes"]
    for modality in ("text", "visual"):
        item = info["details"][modality]
        basis = item["basis"].float()
        if not torch.isfinite(basis).all():
            raise FloatingPointError(f"Non-finite selected basis in {dataset}/{seed}/{label}/{modality}")
        active_basis = basis[:, active]
        if active_basis.size(1) == 0:
            raise ValueError(f"No active nodes in {dataset}/{seed}")
        gram = torch.einsum("knd,lnd->kl", active_basis, active_basis) / (
            active_basis.size(1) * active_basis.size(2)
        )
        condition, jitter = _condition_number(gram)
        diagonal = torch.diag(gram)
        offdiag = gram - torch.diag(diagonal)
        breakdown = item["breakdown"].float().mean(dim=-1)
        prior = item["prior"][active].float()
        response = item["response"][active].float()
        scaled = item["scaled_correction"][active].float()
        prior_rms = float(torch.sqrt(prior.square().mean()).item())
        response_rms = float(torch.sqrt(response.square().mean()).item())
        scaled_rms = float(torch.sqrt(scaled.square().mean()).item())
        gate = float(item["gate"].item())
        beta = item["beta"].detach().cpu().tolist()
        row = {
            "dataset": dataset,
            "seed": seed,
            "variant_label": label,
            "variant": variant,
            "modality": modality,
            "active_nodes": int(active.sum().item()),
            "gram_diag_mean": float(diagonal.mean().item()),
            "gram_diag_min": float(diagonal.min().item()),
            "gram_diag_max": float(diagonal.max().item()),
            "gram_offdiag_abs_mean": float(offdiag.abs().sum().item() / max(offdiag.numel() - len(diagonal), 1)),
            "gram_offdiag_abs_max": float(offdiag.abs().max().item()),
            "gram_condition_number": condition,
            "gram_condition_jitter": jitter,
            "basis_breakdown_fraction_order1": float(breakdown[0].item()),
            "basis_breakdown_fraction_order2": float(breakdown[1].item()),
            "basis_breakdown_fraction_order3": float(breakdown[2].item()),
            "basis_breakdown_fraction_order4": float(breakdown[3].item()),
            "effective_gate": gate,
            "prior_rms": prior_rms,
            "structural_response_rms": response_rms,
            "scaled_structural_rms": scaled_rms,
            "scaled_structural_to_prior_rms_ratio": scaled_rms / max(prior_rms, 1.0e-12),
            "breakdown_definition": "fraction of hidden channels whose unregularized residual RMS is below 1e-5",
        }
        basis_rows.append(row)
        if variant == "A3_sosb_modality":
            profile = modality
        else:
            profile = "shared"
        for order, value in enumerate(beta, start=1):
            if variant != "A3_sosb_modality" and modality != "text":
                continue
            coefficient_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant_label": label,
                    "coefficient_profile": profile,
                    "order": order,
                    "effective_beta": float(value),
                    "effective_gate": gate,
                }
            )
    del model, z, info, x_cpu, edge_cpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return basis_rows, coefficient_rows


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_report(
    research_dir: Path,
    campaign: dict[str, Any],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    freeze_commit: str | None,
) -> None:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in summary:
        by_variant.setdefault(str(row["variant_label"]), []).append(row)
    cross_dataset = {
        label: {
            "acc": _mean([float(row["val_acc_mean"]) for row in values]),
            "f1": _mean([float(row["val_macro_f1_mean"]) for row in values]),
        }
        for label, values in by_variant.items()
    }
    pair_by_name = {row["comparison"]: row for row in paired}
    raw = [row for row in basis_rows if row["variant_label"] == "A1_rawpoly_shared"]
    sosb = [row for row in basis_rows if row["variant_label"] in {"A2_sosb_shared", "A3_sosb_modality"}]
    raw_off = _mean([float(row["gram_offdiag_abs_mean"]) for row in raw])
    sosb_off = _mean([float(row["gram_offdiag_abs_mean"]) for row in sosb])
    raw_cond = _mean([float(row["gram_condition_number"]) for row in raw])
    sosb_cond = _mean([float(row["gram_condition_number"]) for row in sosb])
    lines = [
        "# SOSB-MAG V1.5 Structural Basis Screen",
        "",
        "## Protocol and scope",
        "",
        f"- Branch: `exp/sosb_mag_v15_basis_screen`; freeze commit: `{freeze_commit or 'not recorded'}`.",
        f"- Formal runs: {len(campaign.get('run_rows', []))}/{len(DATASETS) * len(SEEDS) * len(NEW_VARIANTS)}; unresolved failures: {len(campaign.get('unresolved_failures', []))} (failure attempts recorded: {len(campaign.get('failures', []))}).",
        "- Validation Accuracy selects the checkpoint. `task.evaluate_test=false`; no Test metrics were computed and no Test labels were indexed.",
        "- This screen compares structural response coordinates with a protected intrinsic residual; it does not test node routing or expert specialization.",
        "- SOSB is an **OptBasis-inspired signal-conditioned orthogonal Krylov basis**, not an exact OptBasisGNN reproduction.",
        "- Breakdown uses unregularized residual RMS for the `<1e-5` decision because adding `eps=1e-8` first would floor RMS at `1e-4`; normalization uses `sqrt(mean(square)+eps)`. This makes the configured breakdown threshold operative.",
        "- Gram condition number is `cond(G + jitter*I)`, where `jitter=1e-6*max(1, abs(trace(G))/K)`; RawPoly and SOSB use the same Gram definition.",
        "- No HPO, Test evaluation, significance testing, or causal intervention was performed.",
        "",
        "## Validation results",
        "",
        "Values below average the three dataset-level seed means equally. They are descriptive summaries, not pooled node metrics.",
        "",
        "| Variant | Validation Accuracy | Validation Macro-F1 |",
        "|---|---:|---:|",
    ]
    for label, _ in NEW_VARIANTS:
        stats = cross_dataset.get(label)
        if stats:
            lines.append(f"| {label} | {_fmt(stats['acc'])} | {_fmt(stats['f1'])} |")
    lines.extend(["", "## Paired comparisons", "", "Positive counts use paired dataset-seed runs, with no inferential test.", ""])
    lines.extend(
        [
            "| Comparison | Paired Δ Accuracy | Paired Δ Macro-F1 | Positive Acc. pairs | Positive F1 pairs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target, baseline in PAIRINGS:
        name = f"{target}-minus-{baseline}"
        row = pair_by_name.get(name)
        if row:
            lines.append(
                f"| {name} | {_fmt(float(row['delta_val_acc_mean']))} | {_fmt(float(row['delta_val_macro_f1_mean']))} | "
                f"{row['positive_acc_pairs_over_9']} | {row['positive_f1_pairs_over_9']} |"
            )
    lines.extend(
        [
            "",
            "## Conditioning and breakdown",
            "",
            f"Across selected A1/A2/A3 modality-checkpoints, mean absolute off-diagonal Gram entry was {raw_off:.5g} for RawPoly and {sosb_off:.5g} for SOSB.",
            f"Mean jittered Gram condition number was {raw_cond:.5g} for RawPoly and {sosb_cond:.5g} for SOSB. These describe coordinate conditioning; they do not establish a performance cause.",
            "SOSB breakdown fractions are recorded per dataset, seed, modality, and order in `data/basis_diagnostics.csv`.",
            "",
            "## Interpretation rules and observed evidence",
            "",
            "1. If A1 improves over A0 while A2 is close to A1, the richer K=4 response space has task evidence, while orthogonal coordinates have no separate task advantage in this screen.",
            "2. If A2 improves over A1 and SOSB has healthier Gram conditioning, signal-conditioned orthogonal coordinates have both task and conditioning evidence. This does not show orthogonality caused the task difference.",
            "3. If A3 improves over A2, modality-level structural coefficient/gate preferences remain useful after signal-conditioned bases.",
            "4. If A3 is close to A2, shared coefficients remain viable; the screen does not prove modality structural differences are absent.",
            "5. If A1/A2/A3 are consistently weaker than A0, the current Local/Global inductive bias remains stronger in these runs; this is not a reason to add an SOSB-MoE in this stage.",
            "",
            "Observed paired deltas above provide the evidence for these conditions. Results are limited to three datasets and three seeds per dataset.",
            "",
            "## Scientific boundaries",
            "",
            "This experiment evaluates coordinate conditioning, redundancy, and modality-conditioned structural response. It does not show that orthogonalization expands the polynomial function space, that SOSB is task-optimal, that node routing is unnecessary, or how many experts a later model should use. No M=4 expert bank or router was implemented. Stop this stage here.",
            "",
            "## Reproducibility artifacts",
            "",
            "- Run-level results: `data/run_rows.json`; aggregate manifest: `data/campaign_manifest.json`.",
            "- Validation summaries and paired comparisons: `data/summary.csv`, `data/paired_comparisons.csv`.",
            "- Selected-checkpoint basis/profile records: `data/basis_diagnostics.csv`, `data/coefficient_profiles.csv`.",
            "- Legacy V1 readout: `LEGACY_DIAGNOSTIC.md` and `data/legacy_v1_diagnostics.json`.",
        ]
    )
    (research_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(research_dir: Path, campaign: dict[str, Any], freeze_commit: str | None) -> None:
    lines = [
        "# SOSB-MAG V1.5 Structural Basis Screen",
        "",
        "This directory contains the validation-only four-variant screen requested for the SOSB-MAG V1.5 structural response basis.",
        "",
        "## Variants",
        "",
        "- `A0_legacy_lg`: protected intrinsic path plus shared bias-free Local/Global response experts, static two-way mixture, and sigmoid gate.",
        "- `A1_rawpoly_shared`: four per-order active-RMS normalized raw polynomial responses and shared L2-normalized coefficients/gate.",
        "- `A2_sosb_shared`: four signal-conditioned, channel-wise orthogonal Krylov responses and shared coefficients/gate.",
        "- `A3_sosb_modality`: same SOSB basis with separate Text and Visual coefficient vectors/gates.",
        "",
        "All variants use the same modality projectors, hidden dimension 256, dropout 0.2, physical no-self-loop graph, protected intrinsic residual, and late fusion. Variant-specific modules are instantiated in a fixed order and frozen when inactive. No router, MoE, cross-modal attention, private expert, topology learning, auxiliary diversity loss, or GPR baseline is included.",
        "",
        "## Protocol",
        "",
        "Datasets: Movies, Grocery, ele-fashion. Seeds: 42, 43, 44. The full campaign comprises 36 runs under `unified_full_graph_nc_v1`; the selected checkpoint maximizes Validation Accuracy. `task.evaluate_test=false` for smoke and campaign. Test metrics and Test label indexing are prohibited.",
        "",
        "## Numerical definitions",
        "",
        "Active nodes have nonzero physical degree after self-loop removal. Active RMS and channel inner products only use active nodes; isolated structural coordinates are exact zeros. SOSB uses two-pass channel-wise modified Gram-Schmidt. Breakdown compares unregularized residual RMS against 1e-5, then uses `sqrt(mean(square)+1e-8)` for normalization. The distinction is necessary because adding eps before comparison floors the RMS at 1e-4.",
        "",
        "The condition number is computed as `cond(G + jitter*I)` with `jitter=1e-6*max(1, abs(trace(G))/K)`. Paired comparisons are descriptive, with no significance tests.",
        "",
        "## Run state",
        "",
        f"- Freeze commit: `{freeze_commit or 'not recorded'}`.",
        f"- Campaign completed: {len(campaign.get('run_rows', []))}/36; failures: {len(campaign.get('failures', []))}.",
        "- Smoke and formal run outputs/checkpoints remain under ignored `outputs/`; only compact JSON/CSV/Markdown research records are tracked here.",
        "",
        "See `LEGACY_DIAGNOSTIC.md` for the prior V1 checkpoint analysis and `REPORT.md` for results and interpretation.",
    ]
    (research_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign_analysis(
    output_root: Path,
    research_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    data_dir = research_dir / "data"
    campaign_path = data_dir / "campaign_manifest.json"
    rows_path = data_dir / "run_rows.json"
    if not campaign_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError("Formal campaign manifest/run_rows not found; run the campaign first")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if campaign.get("task_evaluate_test") is not False:
        raise RuntimeError("Campaign manifest does not certify task.evaluate_test=false")
    if any(any(key.lower().startswith("test") for key in row.get("metrics", {})) for row in rows):
        raise RuntimeError("Test metric key found in campaign run rows")

    summary_rows = build_summary_rows(rows)
    summary_fields = [
        "dataset", "variant_label", "variant", "num_runs", "val_acc_mean", "val_acc_std",
        "val_macro_f1_mean", "val_macro_f1_std", "mean_best_epoch", "model_trainable_params",
        "classifier_params", "total_trainable_params",
    ]
    write_csv(data_dir / "summary.csv", summary_fields, summary_rows)
    paired_rows = build_paired_rows(rows)
    paired_fields = list(paired_rows[0]) if paired_rows else ["comparison"]
    write_csv(data_dir / "paired_comparisons.csv", paired_fields, paired_rows)

    device = torch.device(device_name)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    basis_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        feature_cache[dataset] = load_features_and_edges(dataset, 42)
    for dataset in DATASETS:
        for seed in SEEDS:
            for label, variant in NEW_VARIANTS[1:]:
                b_rows, c_rows = analyze_selected_checkpoint(
                    dataset,
                    seed,
                    label,
                    variant,
                    output_root,
                    device,
                    feature_cache[dataset],
                )
                basis_rows.extend(b_rows)
                coefficient_rows.extend(c_rows)

    basis_fields = [
        "dataset", "seed", "variant_label", "variant", "modality", "active_nodes",
        "gram_diag_mean", "gram_diag_min", "gram_diag_max", "gram_offdiag_abs_mean",
        "gram_offdiag_abs_max", "gram_condition_number", "gram_condition_jitter",
        "basis_breakdown_fraction_order1", "basis_breakdown_fraction_order2",
        "basis_breakdown_fraction_order3", "basis_breakdown_fraction_order4",
        "effective_gate", "prior_rms", "structural_response_rms", "scaled_structural_rms",
        "scaled_structural_to_prior_rms_ratio", "breakdown_definition",
    ]
    write_csv(data_dir / "basis_diagnostics.csv", basis_fields, basis_rows)
    coefficient_fields = [
        "dataset", "seed", "variant_label", "coefficient_profile", "order", "effective_beta", "effective_gate"
    ]
    write_csv(data_dir / "coefficient_profiles.csv", coefficient_fields, coefficient_rows)
    environment = environment_payload(device_name)
    write_json(data_dir / "environment.json", environment)

    freeze_value = campaign.get("freeze_commit_sha")
    build_report(research_dir, campaign, summary_rows, paired_rows, basis_rows, freeze_value)
    write_readme(research_dir, campaign, freeze_value)
    campaign["summary_rows"] = summary_rows
    campaign["paired_comparisons"] = paired_rows
    campaign["basis_diagnostics_rows"] = len(basis_rows)
    campaign["coefficient_profile_rows"] = len(coefficient_rows)
    campaign["environment_artifact"] = "data/environment.json"
    write_json(campaign_path, campaign)
    return {
        "runs": len(rows),
        "summary_rows": len(summary_rows),
        "paired_comparisons": len(paired_rows),
        "basis_rows": len(basis_rows),
        "coefficient_rows": len(coefficient_rows),
        "report": str(research_dir / "REPORT.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--campaign-only", action="store_true")
    parser.add_argument("--legacy-output-dir", type=Path, default=DEFAULT_OLD_OUTPUT)
    parser.add_argument("--campaign-output-dir", type=Path, default=DEFAULT_CAMPAIGN_OUTPUT)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    research_dir = args.research_dir.resolve()
    if not args.campaign_only:
        payload = run_legacy_diagnostics(
            args.legacy_output_dir.resolve(), research_dir, args.device
        )
        print(
            f"Legacy diagnostics: {payload['available_checkpoints']}/"
            f"{payload['expected_checkpoints']} checkpoints; status={payload['status']}"
        )
    if not args.legacy_only:
        result = run_campaign_analysis(
            args.campaign_output_dir.resolve(), research_dir, args.device
        )
        print(f"Campaign analysis: {json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
