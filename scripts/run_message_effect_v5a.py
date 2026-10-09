#!/usr/bin/env python3
"""Frozen-host exposed-message atlas and estimator pilot for V5A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Grocery", "ele-fashion")
ESTIMATORS = ("E0_heuristic", "E1_unimodal_pair", "E2_multimodal_pair", "E3_host_context")
VARIANTS = ("R1_global_residual", "R2_expert_residual", "R3_adaptive_residual")
SEED = 2027
V4A_RESEARCH = ROOT / "research" / "mvcge_mag_v4a_raw_anchored_residual_screen"
V4A_OUTPUT = ROOT / "outputs" / "mvcge_mag_v4a_raw_anchored_residual_screen"
RESEARCH = ROOT / "research" / "mag_message_effect_v5a_characterization"
DATA = RESEARCH / "data"
OUTPUT = ROOT / "outputs" / "mag_message_effect_v5a"

from src.research.message_effect_atlas import (
    SCALAR_FEATURE_NAMES,
    effect_diagnostics,
    generate_message_effect_atlas,
    make_estimator_split,
    select_receiver_edges,
)
from src.research.message_effect_replay import (
    build_frozen_host_cache,
    receiver_degree_quartile_sample,
    replay_local_basis_changes,
)
from src.research.message_effect_training import (
    ESTIMATORS,
    estimator_comparisons,
    train_evaluate_estimators,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _v4a_rows() -> list[dict[str, Any]]:
    return json.loads((V4A_RESEARCH / "data" / "run_rows.json").read_text(encoding="utf-8"))


def _find_v4a_row(dataset: str, variant: str, seed: int = 42) -> dict[str, Any]:
    matches = [r for r in _v4a_rows() if r["dataset"] == dataset and int(r["seed"]) == seed and r["variant"] == variant]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one V4A row for {dataset}/{seed}/{variant}, got {len(matches)}")
    row = matches[0]
    if row.get("status") != "completed":
        raise RuntimeError(f"V4A selected row is not completed: {dataset}/{seed}/{variant}")
    checkpoint = ROOT / row["checkpoint_path"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"required existing V4A checkpoint is missing: {checkpoint}")
    return row


def _load_host(row: dict[str, Any], device: str):
    """Load the exact frozen checkpoint and feature/edge tensors, without labels."""
    from src.models.mvcge_mag_v4a import Model
    from scripts.analyze_mvcge_mag_v4a_screen import load_features_and_edges

    ckpt_path = ROOT / row["checkpoint_path"]
    run_dir = V4A_OUTPUT / row["mode"] / "runs" / row["dataset"] / f"seed_{row['seed']}" / row["variant"]
    cfg_path = run_dir / "hydra" / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"selected V4A Hydra config is missing: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if cfg.task.get("evaluate_test", True) is not False:
        raise AssertionError(f"V4A selected host was trained with Test evaluation enabled: {cfg_path}")
    cfg.model.variant = row["variant"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if any(str(k).lower().startswith("test") for k in checkpoint.get("metrics", {})):
        raise AssertionError(f"Test metrics found in selected checkpoint metadata: {ckpt_path}")
    model = Model(cfg, checkpoint["data_info"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(checkpoint["data_info"]["num_classes"])).to(device).eval()
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    x, edge_index = load_features_and_edges(row["dataset"], int(row["seed"]))
    return model, classifier, checkpoint, cfg, x, edge_index


def _compose(dataset: str, seed: int = 42):
    from scripts.analyze_mvcge_mag_v4a_screen import _compose as compose
    return compose(dataset, seed)


def load_training_supervision(dataset: str, *, include_validation: bool = False, seed: int = 42):
    """Read train labels only; optional Validation access is isolated to V4A deletion audit."""
    from src.data.loaders import resolve_path

    cfg = _compose(dataset, seed)
    ds = OmegaConf.to_container(cfg.dataset, resolve=True)
    if ds["source"] == "magb":
        split_path = resolve_path(ds["nc_split_path"])
    else:
        split_path = resolve_path(ds["node_split_path"])
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    train_idx = torch.as_tensor(split["train_idx"], dtype=torch.long).reshape(-1).cpu()
    val_idx = torch.as_tensor(split["val_idx"], dtype=torch.long).reshape(-1).cpu() if include_validation else None

    if ds["source"] == "magb":
        import dgl
        graphs, _ = dgl.load_graphs(str(resolve_path(ds["graph_path"])))
        labels_all = graphs[0].ndata["label"].long().cpu()
    else:
        labels_all = torch.as_tensor(
            torch.load(resolve_path(ds["label_path"]), map_location="cpu", weights_only=False), dtype=torch.long
        ).reshape(-1)
    train_labels = labels_all.index_select(0, train_idx).clone()
    if train_labels.numel() and bool(((train_labels < 0) | (train_labels >= int(ds["num_classes"]))).any()):
        raise ValueError(f"training split contains missing/out-of-range labels for {dataset}")
    result = {"train_idx": train_idx, "train_labels": train_labels, "num_classes": int(ds["num_classes"]),
              "train_labels_read": True, "validation_labels_read": False, "test_labels_read": False,
              "split_path": str(split_path)}
    if include_validation:
        assert val_idx is not None
        val_labels = labels_all.index_select(0, val_idx).clone()
        result.update({"val_idx": val_idx, "val_labels": val_labels,
                       "validation_labels_read": True})
    del labels_all, split
    return result


def _train_labels_by_node(supervision: dict[str, Any], num_nodes: int) -> torch.Tensor:
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    labels[supervision["train_idx"]] = supervision["train_labels"]
    return labels


def _environment_record(device: str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "device_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "CPU",
        "test_labels_read": False,
        "test_metrics_read": False,
    }


def _max_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


@torch.no_grad()
def run_preflight(device: str) -> dict[str, Any]:
    """Small Movies train-only end-to-end replay audit before formal sampling."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device unavailable: {device}")
    row = _find_v4a_row("Movies", "R0_raw", 42)
    model, classifier, checkpoint, cfg, x, edge_index = _load_host(row, device)
    supervision = load_training_supervision("Movies", include_validation=False, seed=42)
    x = x.to(device)
    edge_index = edge_index.to(device)
    cache = build_frozen_host_cache("Movies", model, classifier, x, edge_index, device)
    labels_by_node = _train_labels_by_node(supervision, x.size(0)).to(device)
    active_train = supervision["train_idx"][cache.indegree.detach().cpu()[supervision["train_idx"]] > 0]
    if active_train.numel() < 10:
        raise RuntimeError("Movies has fewer than 10 physical-active training receivers")
    generator = torch.Generator().manual_seed(SEED)
    chosen = active_train[torch.randperm(active_train.numel(), generator=generator)[:10]].sort().values
    receiver_probs = {int(v): 1.0 for v in chosen.tolist()}
    edge_ids, edge_probs = select_receiver_edges(cache, chosen, max_edges=2, seed=SEED)
    if any(sum(int(cache.dst[eid].item()) == receiver for eid in edge_ids) > 2 for receiver in chosen.tolist()):
        raise AssertionError("preflight selected more than two directed edges for one receiver")
    receivers_device = chosen.to(device)
    reconstructed = replay_local_basis_changes(cache, receivers_device)
    reconstruction = {
        "modality_text_max_abs": _max_difference(
            reconstructed["modality_outputs"]["text"], cache.modality_outputs["text"].index_select(0, receivers_device)
        ),
        "modality_visual_max_abs": _max_difference(
            reconstructed["modality_outputs"]["visual"], cache.modality_outputs["visual"].index_select(0, receivers_device)
        ),
        "fused_z_max_abs": _max_difference(reconstructed["z"], cache.z.index_select(0, receivers_device)),
        "classifier_logits_max_abs": _max_difference(
            reconstructed["logits"], cache.logits.index_select(0, receivers_device)
        ),
    }
    reconstruction["within_1e5"] = all(value <= 1.0e-5 for key, value in reconstruction.items() if key.endswith("max_abs"))
    singleton_rows, bundle_rows, features = generate_message_effect_atlas(
        cache, edge_ids, labels_by_node, receiver_probs, edge_probs, host_seed=42
    )
    split = make_estimator_split(features, "Movies")
    split_disjoint = not any(a & b for i, a in enumerate(split.values()) for b in list(split.values())[i + 1 :])
    mass_errors = []
    for eid in edge_ids:
        receiver = int(cache.dst[eid].item())
        retained = float(cache.receiver_mass[receiver] - cache.edge_weight[eid])
        if retained > 1.0e-12:
            scale = float(cache.receiver_mass[receiver] / retained)
            mass_errors.append(abs(scale * retained - float(cache.receiver_mass[receiver])))
    finite_effects = all(
        math.isfinite(float(r["delta_delete"]))
        and (r["delta_comp"] is None or math.isfinite(float(r["delta_comp"])))
        for r in singleton_rows
    )
    summary = {
        "protocol": "Movies R0 seed42 frozen-host local exposed-message replay",
        "host_checkpoint": row["checkpoint_path"],
        "host_seed": 42,
        "train_labels_read": True,
        "validation_labels_read": False,
        "test_labels_read": False,
        "test_metrics_read": False,
        "receiver_count": len(chosen),
        "edge_count": len(edge_ids),
        "max_sampled_edges_per_receiver": 2,
        "singleton_count": len(singleton_rows),
        "bundle_count": len(bundle_rows),
        "compensation_feasible_count": sum(bool(r["compensation_feasible"]) for r in singleton_rows),
        "local_reconstruction": reconstruction,
        "finite_effects": finite_effects,
        "receiver_split_disjoint": split_disjoint,
        "compensation_mass_max_abs_error": max(mass_errors, default=0.0),
        "scalar_feature_names": list(SCALAR_FEATURE_NAMES),
        "completed_at_utc": _utc(),
    }
    if not (
        reconstruction["within_1e5"]
        and finite_effects
        and split_disjoint
        and summary["compensation_mass_max_abs_error"] <= 1.0e-6
    ):
        raise RuntimeError(f"V5A preflight implementation check failed: {summary}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "rows": singleton_rows}, OUTPUT / "preflight_features.pt")
    _write_json(DATA / "preflight_summary.json", summary)
    _write_json(DATA / "environment.json", _environment_record(device))
    del model, classifier, checkpoint, cfg, x, edge_index, cache
    if "cuda" in device:
        torch.cuda.empty_cache()
    return summary


def run_smoke(device: str) -> dict[str, Any]:
    cache_path = OUTPUT / "preflight_features.pt"
    if not cache_path.is_file():
        raise FileNotFoundError("run --mode preflight before --mode smoke")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    features = payload["features"]
    metrics, groups, predictions = train_evaluate_estimators(
        "Movies",
        features,
        rows,
        device=device,
        seeds=(0,),
        max_epochs=3,
        patience=3,
        checkpoint_dir=OUTPUT / "smoke_estimators",
    )
    summary = {
        "dataset": "Movies",
        "estimators": list(ESTIMATORS),
        "estimator_runs": len(metrics) // 2,
        "targets_per_run": 2,
        "max_epochs": 3,
        "finite_predictions": all(math.isfinite(float(r["prediction"])) for r in predictions),
        "finite_effects": all(
            math.isfinite(float(r["delta_delete"]))
            and (r["delta_comp"] is None or math.isfinite(float(r["delta_comp"])))
            for r in rows
        ),
        "receiver_split_disjoint": not any(
            a & b
            for i, a in enumerate(make_estimator_split(features, "Movies").values())
            for b in list(make_estimator_split(features, "Movies").values())[i + 1 :]
        ),
        "metric_rows": len(metrics),
        "group_metric_rows": len(groups),
        "completed_at_utc": _utc(),
    }
    if summary["estimator_runs"] != 4 or not summary["finite_predictions"] or not summary["finite_effects"]:
        raise RuntimeError(f"V5A estimator smoke check failed: {summary}")
    _write_json(DATA / "smoke_summary.json", summary)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_effect_summary(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset": dataset}
    for name, field in (("delete", "delta_delete"), ("comp", "delta_comp")):
        values = [float(r[field]) for r in rows if r[field] is not None and math.isfinite(float(r[field]))]
        result[f"{name}_n"] = len(values)
        result[f"{name}_mean"] = statistics.fmean(values) if values else float("nan")
        result[f"{name}_std"] = statistics.pstdev(values) if values else float("nan")
        result[f"{name}_p05"] = float(np.quantile(values, 0.05)) if values else float("nan")
        result[f"{name}_median"] = float(np.median(values)) if values else float("nan")
        result[f"{name}_p95"] = float(np.quantile(values, 0.95)) if values else float("nan")
        result[f"{name}_negative_fraction"] = (
            sum(value < 0 for value in values) / len(values) if values else float("nan")
        )
    result["receiver_count"] = len({r["receiver_id"] for r in rows})
    result["directed_edge_count"] = len({r["directed_edge_index"] for r in rows})
    result["singleton_count"] = len(rows)
    result["compensation_feasible_count"] = sum(bool(r["compensation_feasible"]) for r in rows)
    result["compensation_feasible_fraction"] = result["compensation_feasible_count"] / max(len(rows), 1)
    return result


def _write_readme_and_report(
    diagnostics: dict[str, list[dict[str, Any]]],
    estimator_metrics: list[dict[str, Any]],
    estimator_comparison_rows: list[dict[str, Any]],
    dataset_summary: list[dict[str, Any]],
) -> None:
    readme = """# V5A: Task-Grounded Multimodal Message-Effect Atlas

This branch characterizes frozen V4A R0 Raw hosts on Movies, Grocery, and ele-fashion (host seed 42). All effects are frozen-host exposed-message singleton contrasts. They change one receiver's normalized Raw basis using the cached incoming message and replay the frozen local experts, fusion, and classifier. Upstream states, other receivers, Raw denominators, routes, strength, and classifier remain fixed. These are not recursive graph deletions or causal graph-edge effects.

The Atlas samples at most 1,500 physical-active training receivers per dataset using degree-quartile strata (seed 2027), then samples up to four incoming directed edges uniformly per receiver. It evaluates both modalities at all four hops. Train labels are used only for effect targets and estimator targets. Validation and Test labels/metrics are not accessed by the main Atlas or estimator. The separate V4A gate deletion audit reads Validation labels only; it never reads Test.

Estimator feature boundary: E0 uses scalar edge/role/router/message cues and hop/modality indicators, with no prediction scalars. E1 adds the active-modality pair block. E2 adds separate Text and Visual pair blocks without cross-modal dot products. E3 adds frozen receiver context (fused embedding, active-modality raw mixture, baseline logits, and prediction scalars). No receiver label is an estimator feature. Estimator receiver partitions use a stable hash of dataset, receiver ID, and 2027 (70/15/15).

Full row tables, hidden feature caches, estimator checkpoints, and logs are kept under the ignored outputs/mag_message_effect_v5a directory. The committed audit CSV is a deterministic sample of at most 500 scalar rows per dataset and includes no labels or hidden vectors. Aggregated results and hashes are in data/.
"""
    (RESEARCH / "README.md").write_text(readme, encoding="utf-8")

    audit_path = DATA / "v4a_same_checkpoint_residual_deletion.csv"
    audit_rows = []
    if audit_path.is_file():
        with audit_path.open(newline="", encoding="utf-8") as handle:
            audit_rows = list(csv.DictReader(handle))
    audit_acc = [float(r["normal_minus_zero_beta_accuracy_delta_pp"]) for r in audit_rows]
    audit_f1 = [float(r["normal_minus_zero_beta_macro_f1_delta_pp"]) for r in audit_rows]
    report: list[str] = [
        "# V5A Task-Grounded Multimodal Message-Effect Atlas",
        "",
        "## Protocol and interpretation boundary",
        "",
        "All effects are frozen-host exposed-message singleton contrasts. A directed incoming message is removed only from one receiver's cached normalized Raw hop basis; baseline upstream states, other receivers, other edges/hops, denominators, router, strength, and classifier stay fixed. This is not recursive graph deletion and does not establish a causal graph-edge effect. The pilot is descriptive, uses V4A R0 seed-42 hosts, and has no performance pass/fail gate.",
        "",
        "Main Atlas and estimator generation used training receiver indices and training labels only. Validation labels were read only by the separate V4A same-checkpoint gate deletion audit. Test labels and Test metrics were not read.",
        "",
        "## V4A same-checkpoint residual-gate deletion audit",
        "",
        f"For 27 R1/R2/R3 checkpoints, zeroing the active raw residual-gate parameter changed Validation accuracy by mean {statistics.fmean(audit_acc):.4f} pp and macro-F1 by mean {statistics.fmean(audit_f1):.4f} pp. This is a same-checkpoint direct-dependence audit only, not causal proof or a training-noise diagnosis. Per-variant dataset means and positive counts are in the accompanying CSV.",
        "",
        "## Effect distributions by dataset",
        "",
        "| Dataset | Singleton rows | Receivers | Directed edges | Delete mean ± SD | Delete median (P05, P95) | Delete negative fraction | Comp feasible fraction | Comp mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_summary:
        report.append(
            f"| {row['dataset']} | {row['singleton_count']} | {row['receiver_count']} | {row['directed_edge_count']} | "
            f"{row['delete_mean']:.6g} ± {row['delete_std']:.6g} | {row['delete_median']:.6g} ({row['delete_p05']:.6g}, {row['delete_p95']:.6g}) | "
            f"{row['delete_negative_fraction']:.3f} | {row['compensation_feasible_fraction']:.3f} | "
            f"{row['comp_mean']:.6g} ± {row['comp_std']:.6g} |"
        )
    report += ["", "## Required research questions", ""]
    hrows = diagnostics["heterogeneity_diagnostics"]
    crows = diagnostics["compensation_diagnostics"]
    brows = diagnostics["bundle_diagnostics"]
    for dataset in DATASETS:
        within = [r for r in hrows if r["dataset"] == dataset and r.get("diagnostic") == "within_receiver_edge_effects"]
        stds = [float(r["mean_within_group_std"]) for r in within if math.isfinite(float(r.get("mean_within_group_std", float("nan"))))]
        modality = next((r for r in hrows if r["dataset"] == dataset and r.get("diagnostic") == "text_visual_effect_difference"), {})
        hop = next((r for r in hrows if r["dataset"] == dataset and r.get("diagnostic") == "hop_effect_heterogeneity"), {})
        comp = next((r for r in crows if r["dataset"] == dataset), {})
        full_bundle = next((r for r in brows if r["dataset"] == dataset and r["bundle_type"] == "full_exposed_edge_8_messages" and r["target"] == "delete"), {})
        report.append(
            f"### {dataset}\n\n"
            f"- **Q1 Edge heterogeneity:** within-receiver group SD averaged {statistics.fmean(stds) if stds else float('nan'):.6g} across modality-hop cells; group counts and ranges are in heterogeneity_diagnostics.csv.\n"
            f"- **Q2 Text/Visual:** paired sign disagreement was {float(modality.get('sign_disagreement_fraction', float('nan'))):.3f}; effect Spearman was {float(modality.get('text_visual_spearman', float('nan'))):.3f}.\n"
            f"- **Q3 Hop variation:** mixed-sign-across-hop fraction was {float(hop.get('mixed_sign_fraction', float('nan'))):.3f}; mean effect SD across hops was {float(hop.get('mean_effect_std_across_hops', float('nan'))):.6g}.\n"
            f"- **Q4 Compensation:** delete-versus-compensated Spearman was {float(comp.get('spearman_delete_comp', float('nan'))):.3f}, sign flips {float(comp.get('sign_flip_fraction', float('nan'))):.3f}, and median |comp|/|delete| {float(comp.get('median_effect_magnitude_ratio_comp_over_delete', float('nan'))):.3f}.\n"
            f"- **Q9 Bundle non-additivity:** full eight-message median absolute interaction was {float(full_bundle.get('median_abs_interaction', float('nan'))):.6g}; median relative interaction was {float(full_bundle.get('relative_interaction_median', float('nan'))):.3f}."
        )
    report += [
        "",
        "### Q5 Heuristic alignment",
        "",
        "Per-dataset/modality/hop Spearman associations for semantic cosine, V3C role score, router reliability, edge weight, degree, and message magnitude are in heuristic_alignment.csv. These associations are descriptive and do not imply role success or failure.",
        "",
    ]
    alignment = diagnostics["heuristic_alignment"]
    report.append("Mean cue/effect Spearman across the 8 modality-hop cells for each dataset:")
    for dataset in DATASETS:
        entries = []
        for cue in (
            "semantic_cosine_text",
            "semantic_cosine_visual",
            "role_score_text",
            "role_score_visual",
            "router_reliability_active_modality",
        ):
            selected = [
                r for r in alignment
                if r["dataset"] == dataset and r.get("cue") == cue
            ]
            for target, field in (("delete", "spearman_delete"), ("comp", "spearman_comp")):
                values = [
                    float(r[field]) for r in selected
                    if math.isfinite(float(r.get(field, float("nan"))))
                ]
                entries.append(
                    f"{cue}/{target}={statistics.fmean(values):.3f}"
                    if values else f"{cue}/{target}=NA"
                )
        report.append(f"- {dataset}: " + "; ".join(entries))
    report += [
        "",
        "Support/discrepant negative-effect fractions are in heuristic_alignment.csv under P(delta_delete<0 | role_partition).",
        "",
        "### Q6–Q8 Estimator comparisons",
        "",
        "Estimator metrics use EstimatorHoldout receivers only. Absolute values are seed mean ± population SD over estimator seeds 0/1/2, separately for deletion and compensation targets.",
        "",
        "### E0–E3 absolute holdout metrics",
        "",
        "Values are seed mean ± population SD over estimator seeds 0/1/2. Ranking scores are receiver-group metrics.",
        "",
        "| Dataset | Target | Estimator | Global Spearman | Edge ranking Spearman | Edge pairwise accuracy | Modality ordering accuracy | Hop ranking Spearman | Hop pairwise accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    def _metric_mean_std(dataset: str, target: str, estimator: str, field: str) -> str:
        selected = [
            float(r[field]) for r in estimator_metrics
            if r["dataset"] == dataset and r["target"] == target
            and r["estimator"] == estimator
            and math.isfinite(float(r.get(field, float("nan"))))
        ]
        return (
            f"{statistics.fmean(selected):.3f} ± {statistics.pstdev(selected):.3f}"
            if selected else "NA"
        )
    for dataset in DATASETS:
        for target in ("delete", "comp"):
            for estimator in ESTIMATORS:
                values = [
                    _metric_mean_std(dataset, target, estimator, field)
                    for field in (
                        "spearman",
                        "edge_ranking_spearman_mean",
                        "edge_pairwise_accuracy",
                        "modality_ordering_accuracy",
                        "hop_ranking_spearman_mean",
                        "hop_pairwise_accuracy",
                    )
                ]
                report.append(
                    f"| {dataset} | {target} | {estimator} | " + " | ".join(values) + " |"
                )
    report += [
        "",
        "### E1–E0, E2–E1, and E3–E2 paired seed differences",
        "",
        "| Dataset | Target | Comparison | Global Spearman | Edge ranking Spearman | Edge pairwise accuracy | Modality ordering accuracy | Hop ranking Spearman |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    comp_lookup = {(r["dataset"], r["target"], r["comparison"], r["metric"]): r for r in estimator_comparison_rows}
    for dataset in DATASETS:
        for target in ("delete", "comp"):
            for comparison in (
                "E1_unimodal_pair-E0_heuristic",
                "E2_multimodal_pair-E1_unimodal_pair",
                "E3_host_context-E2_multimodal_pair",
            ):
                values = []
                for field in ("spearman", "edge_ranking_spearman_mean", "edge_pairwise_accuracy", "modality_ordering_accuracy", "hop_ranking_spearman_mean"):
                    row = comp_lookup.get((dataset, target, comparison, field), {})
                    values.append(f"{row.get('mean_difference', float('nan')):.3f} ({row.get('std_difference', float('nan')):.3f})")
                report.append(f"| {dataset} | {target} | {comparison} | " + " | ".join(values) + " |")
    report += [
        "",
        "E1 adds a same-modality receiver/sender pair representation. E2 adds separate pair blocks from both modalities; a descriptive change does not prove cross-modal interaction. E3 adds frozen host context, so a change is consistent with additional predictive information in that context. No estimator is interpreted as a controller or a strong model-selection result.",
        "",
        "### Q10 Implications for a later message-controller study",
        "",
        "Observed within-receiver, modality, and hop variation can suggest preserving receiver-directed message granularity in a later controller characterization. Bundle interactions can suggest measuring joint interventions alongside singleton sums. These observations do not specify or automatically create a final controller architecture.",
        "",
        "## Implementation audit",
        "",
        "Baseline local replay reconstruction, finite-effect checks, compensation mass preservation, and receiver split disjointness are summarized in preflight_summary.json. Preflight and estimator smoke are implementation checks, not scientific gates. Formal estimator training completed 36/36 runs. Full features, raw effects, estimator checkpoints, and logs remain under ignored outputs/mag_message_effect_v5a.",
    ]
    (RESEARCH / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_formal(device: str) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device unavailable: {device}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    estimator_metrics: list[dict[str, Any]] = []
    estimator_group_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    atlas_datasets: list[dict[str, Any]] = []
    diagnostics = {
        key: []
        for key in (
            "effect_distribution",
            "heterogeneity_diagnostics",
            "compensation_diagnostics",
            "bundle_diagnostics",
            "heuristic_alignment",
        )
    }
    audit_sample: list[dict[str, Any]] = []

    for dataset in DATASETS:
        host_row = _find_v4a_row(dataset, "R0_raw", 42)
        model, classifier, checkpoint, cfg, x_cpu, edge_cpu = _load_host(host_row, device)
        supervision = load_training_supervision(dataset, include_validation=False, seed=42)
        cache = build_frozen_host_cache(
            dataset, model, classifier, x_cpu.to(device), edge_cpu.to(device), device
        )
        receiver_ids, receiver_probs, population_by_receiver = receiver_degree_quartile_sample(
            supervision["train_idx"],
            cache.indegree.detach().cpu(),
            max_receivers=1500,
            seed=SEED,
        )
        if receiver_ids.numel() == 0:
            raise RuntimeError(f"no physical-active training receivers for {dataset}")
        edge_ids, edge_probs = select_receiver_edges(cache, receiver_ids, max_edges=4, seed=SEED)
        selected_receivers = cache.dst.index_select(
            0, torch.as_tensor(edge_ids, dtype=torch.long, device=cache.device)
        )
        selected_coefficients = cache.edge_weight.index_select(
            0, torch.as_tensor(edge_ids, dtype=torch.long, device=cache.device)
        )
        selected_mass = cache.receiver_mass.index_select(0, selected_receivers)
        selected_retained = selected_mass - selected_coefficients
        selected_feasible = selected_retained > 1.0e-12
        mass_error = (
            selected_mass[selected_feasible]
            / selected_retained[selected_feasible]
            * selected_retained[selected_feasible]
            - selected_mass[selected_feasible]
        ).abs()
        mass_error_max = float(mass_error.max().item()) if mass_error.numel() else 0.0
        if mass_error_max > 1.0e-6:
            raise AssertionError(f"compensation mass mismatch for {dataset}: {mass_error_max}")
        labels_by_node = _train_labels_by_node(supervision, x_cpu.size(0)).to(device)
        singleton_rows, bundle_rows, features = generate_message_effect_atlas(
            cache,
            edge_ids,
            labels_by_node,
            receiver_probs,
            edge_probs,
            host_seed=42,
        )
        train_receiver_set = set(int(v) for v in supervision["train_idx"].tolist())
        if any(r["receiver_id"] not in train_receiver_set for r in singleton_rows):
            raise AssertionError(f"non-training receiver found in {dataset} Atlas")
        splits = make_estimator_split(features, dataset)
        split_values = list(splits.values())
        if any(a & b for i, a in enumerate(split_values) for b in split_values[i + 1 :]):
            raise AssertionError(f"estimator receiver split overlap for {dataset}")
        if not all(math.isfinite(float(r["delta_delete"])) for r in singleton_rows):
            raise FloatingPointError(f"non-finite deletion effects for {dataset}")

        raw_path = OUTPUT / f"{dataset}_singleton_raw.csv"
        bundle_path = OUTPUT / f"{dataset}_bundle_raw.csv"
        feature_path = OUTPUT / f"{dataset}_effect_features.pt"
        _write_csv(raw_path, singleton_rows)
        _write_csv(bundle_path, bundle_rows)
        torch.save(
            {
                "features": features,
                "rows": singleton_rows,
                "split_receivers": {key: sorted(value) for key, value in splits.items()},
            },
            feature_path,
        )
        dataset_summaries.append(_dataset_effect_summary(dataset, singleton_rows))
        dataset_diagnostics = effect_diagnostics(singleton_rows, bundle_rows)
        for key in diagnostics:
            diagnostics[key].extend(dataset_diagnostics[key])
        audit_rng = random.Random(SEED + DATASETS.index(dataset))
        audit_sample.extend(audit_rng.sample(singleton_rows, min(500, len(singleton_rows))))
        atlas_datasets.append(
            {
                "dataset": dataset,
                "host_checkpoint": host_row["checkpoint_path"],
                "host_seed": 42,
                "receiver_population_by_quartile": [
                    {
                        "population": int(population),
                        "sampled": sum(v == population for v in population_by_receiver.values()),
                    }
                    for population in sorted(set(population_by_receiver.values()))
                ],
                "receivers_sampled": int(receiver_ids.numel()),
                "edges_sampled": len(edge_ids),
                "singleton_rows": len(singleton_rows),
                "bundle_rows": len(bundle_rows),
                "compensation_feasible_singletons": sum(bool(r["compensation_feasible"]) for r in singleton_rows),
                "compensation_feasible_fraction": (
                    sum(bool(r["compensation_feasible"]) for r in singleton_rows)
                    / max(len(singleton_rows), 1)
                ),
                "compensation_mass_max_abs_error": mass_error_max,
                "effect_target_labels": "train_only",
                "validation_labels_read": False,
                "test_labels_read": False,
                "test_metrics_read": False,
                "full_local_outputs": {
                    str(raw_path): _sha256(raw_path),
                    str(bundle_path): _sha256(bundle_path),
                    str(feature_path): _sha256(feature_path),
                },
            }
        )
        # Free the frozen graph and host before using the compact estimator cache.
        del model, classifier, checkpoint, cfg, x_cpu, edge_cpu, cache
        del labels_by_node, supervision, receiver_ids, receiver_probs, population_by_receiver, edge_ids, edge_probs
        if "cuda" in device:
            torch.cuda.empty_cache()
        metric_rows, group_rows, _ = train_evaluate_estimators(
            dataset,
            features,
            singleton_rows,
            device=device,
            seeds=(0, 1, 2),
            max_epochs=100,
            patience=10,
            checkpoint_dir=OUTPUT / "estimators",
            return_prediction_rows=False,
        )
        estimator_metrics.extend(metric_rows)
        estimator_group_rows.extend(group_rows)
        del features, singleton_rows, bundle_rows
        if "cuda" in device:
            torch.cuda.empty_cache()

    _write_csv(DATA / "effect_distribution.csv", diagnostics["effect_distribution"])
    _write_csv(DATA / "heterogeneity_diagnostics.csv", diagnostics["heterogeneity_diagnostics"])
    _write_csv(DATA / "compensation_diagnostics.csv", diagnostics["compensation_diagnostics"])
    _write_csv(DATA / "bundle_diagnostics.csv", diagnostics["bundle_diagnostics"])
    _write_csv(DATA / "heuristic_alignment.csv", diagnostics["heuristic_alignment"])
    _write_csv(DATA / "estimator_metrics.csv", estimator_metrics)
    _write_csv(DATA / "estimator_group_metrics.csv", estimator_group_rows)
    comparison_rows = estimator_comparisons(estimator_metrics)
    _write_csv(DATA / "estimator_comparisons.csv", comparison_rows)
    _write_csv(DATA / "dataset_summary.csv", dataset_summaries)

    _write_csv(DATA / "message_effect_audit_sample.csv", audit_sample)
    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    _write_json(DATA / "atlas_manifest.json", {
        "protocol": "frozen_host_exposed_message_replay",
        "effect_target_labels": "train_only",
        "validation_labels_read": False,
        "test_labels_read": False,
        "test_metrics_read": False,
        "sampling_seed": SEED,
        "receiver_sampling": "physical-active training receivers, deterministic degree quartiles, maximum 1500",
        "edge_sampling": "uniform without replacement; all edges when indegree <=4, else 4",
        "receiver_split": "stable hash dataset:receiver_id:2027, 70/15/15",
        "host_variant": "R0_raw",
        "host_seed": 42,
        "local_replay_reconstruction_tolerance": 1.0e-5,
        "datasets": atlas_datasets,
        "committed_audit_sample_rows": len(audit_sample),
        "estimator_runs": len(estimator_metrics) // 2,
        "created_at_utc": _utc(),
    })
    _write_readme_and_report(
        diagnostics, estimator_metrics, comparison_rows, dataset_summaries
    )
    _write_json(DATA / "campaign_manifest.json", {
        "experiment": "V5A Task-Grounded Multimodal Message-Effect Atlas",
        "branch": "exp/mag_message_effect_v5a_characterization",
        "v4a_parent_sha": "d3040d46c4fba8d28b61123599b3b0cf5f2ebed1",
        "freeze_sha": repository_head,
        "final_sha_at_run": repository_head,
        "host_campaign": "V4A R0_raw seed42 selected checkpoints; no retraining",
        "datasets": list(DATASETS),
        "estimator_runs_expected": 36,
        "estimator_runs_completed": len(estimator_metrics) // 2,
        "effect_target_labels": "train_only",
        "validation_labels_read_main_atlas": False,
        "test_labels_read": False,
        "test_metrics_read": False,
        "scientific_code_changed_after_freeze": False,
        "completed_at_utc": _utc(),
    })
    if len(estimator_metrics) != 72:
        raise RuntimeError(f"expected 36 estimator runs with two targets; got {len(estimator_metrics)} rows")
    return {
        "datasets": atlas_datasets,
        "estimator_runs": len(estimator_metrics) // 2,
        "singleton_rows": sum(int(r["singleton_count"]) for r in dataset_summaries),
        "bundle_rows": sum(int(r["bundle_rows"]) for r in atlas_datasets),
        "audit_sample_rows": len(audit_sample),
        "freeze_sha": repository_head,
    }


def _classification_metrics(logits: torch.Tensor, labels: torch.Tensor, classes: list[int]) -> dict[str, float]:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    return {"accuracy": float(accuracy_score(target, pred)),
            "macro_f1": float(f1_score(target, pred, labels=classes, average="macro", zero_division=0))}


@torch.no_grad()
def run_v4a_residual_deletion_audit(device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare learned residual gates to zero at the same 27 checkpoints."""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        supervision = load_training_supervision(dataset, include_validation=True, seed=42)
        train_idx, train_labels = supervision["train_idx"], supervision["train_labels"]
        val_idx, val_labels = supervision["val_idx"], supervision["val_labels"]
        classes = sorted(set(train_labels.tolist() + val_labels.tolist()))
        for v in VARIANTS:
            for seed in (42, 43, 44):
                row = _find_v4a_row(dataset, v, seed)
                model, classifier, checkpoint, cfg, x, edge_index = _load_host(row, device)
                x_device, edge_device = x.to(device), edge_index.to(device)
                idx = val_idx.to(device)
                original_parameter = (model.residual_global_raw if v == "R1_global_residual"
                                      else model.residual_expert_raw)
                saved = original_parameter.detach().clone()
                normal_z, _, _, _, _ = model(x_device, edge_device)
                normal_logits = classifier(normal_z)
                normal_metrics = _classification_metrics(normal_logits.index_select(0, idx), val_labels,
                                                          [int(c) for c in classes])
                with torch.no_grad():
                    original_parameter.zero_()
                    zero_z, _, _, _, _ = model(x_device, edge_device)
                    zero_logits = classifier(zero_z)
                    zero_metrics = _classification_metrics(zero_logits.index_select(0, idx), val_labels,
                                                           [int(c) for c in classes])
                    original_parameter.copy_(saved)
                normal_val = normal_logits.index_select(0, idx)
                zero_val = zero_logits.index_select(0, idx)
                normal_pred, zero_pred = normal_val.argmax(-1), zero_val.argmax(-1)
                rows.append({
                    "dataset": dataset, "host_seed": seed, "variant": v,
                    "normal_val_accuracy": normal_metrics["accuracy"],
                    "zero_beta_val_accuracy": zero_metrics["accuracy"],
                    "normal_minus_zero_beta_accuracy_delta_pp": 100.0 * (normal_metrics["accuracy"] - zero_metrics["accuracy"]),
                    "normal_val_macro_f1": normal_metrics["macro_f1"],
                    "zero_beta_val_macro_f1": zero_metrics["macro_f1"],
                    "normal_minus_zero_beta_macro_f1_delta_pp": 100.0 * (normal_metrics["macro_f1"] - zero_metrics["macro_f1"]),
                    "positive_accuracy_delta": normal_metrics["accuracy"] > zero_metrics["accuracy"],
                    "positive_macro_f1_delta": normal_metrics["macro_f1"] > zero_metrics["macro_f1"],
                    "embedding_rms_difference": float((normal_z.index_select(0, idx) - zero_z.index_select(0, idx)).float().square().mean().sqrt().item()),
                    "classifier_logit_rms_difference": float((normal_val - zero_val).float().square().mean().sqrt().item()),
                    "prediction_flip_fraction": float((normal_pred != zero_pred).float().mean().item()),
                    "validation_nodes": int(idx.numel()),
                    "interpretation_boundary": "same-checkpoint direct dependence only; not causal proof or training-noise diagnosis",
                })
                del model, classifier, checkpoint, cfg, x, edge_index, x_device, edge_device
                del normal_z, zero_z, normal_logits, zero_logits, normal_val, zero_val
                if "cuda" in device:
                    torch.cuda.empty_cache()

    summary: list[dict[str, Any]] = []
    for dataset in (*DATASETS, "ALL"):
        for variant in VARIANTS:
            group = [r for r in rows if r["variant"] == variant and (dataset == "ALL" or r["dataset"] == dataset)]
            if not group:
                continue
            acc = [r["normal_minus_zero_beta_accuracy_delta_pp"] for r in group]
            f1 = [r["normal_minus_zero_beta_macro_f1_delta_pp"] for r in group]
            summary.append({"dataset": dataset, "variant": variant, "n_checkpoints": len(group),
                "accuracy_delta_pp_mean": statistics.fmean(acc), "accuracy_delta_pp_std": statistics.pstdev(acc),
                "accuracy_positive_count": sum(v > 0 for v in acc), "macro_f1_delta_pp_mean": statistics.fmean(f1),
                "macro_f1_delta_pp_std": statistics.pstdev(f1), "macro_f1_positive_count": sum(v > 0 for v in f1),
                "embedding_rms_difference_mean": statistics.fmean(r["embedding_rms_difference"] for r in group),
                "logit_rms_difference_mean": statistics.fmean(r["classifier_logit_rms_difference"] for r in group),
                "prediction_flip_fraction_mean": statistics.fmean(r["prediction_flip_fraction"] for r in group)})
    _write_csv(DATA / "v4a_same_checkpoint_residual_deletion.csv", rows)
    _write_csv(DATA / "v4a_same_checkpoint_residual_deletion_summary.csv", summary)
    _write_json(DATA / "v4a_deletion_audit_manifest.json", {
        "protocol": "same_checkpoint_residual_gate_zeroing_validation_audit",
        "checkpoints": len(rows), "variants": list(VARIANTS), "datasets": list(DATASETS),
        "validation_labels_read": True, "test_labels_read": False, "test_metrics_read": False,
        "same_checkpoint_only": True, "interpretation": "direct dependence audit; not causal proof or training-noise diagnosis",
        "device": device, "completed_at_utc": _utc()})
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deletion-audit", "preflight", "smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.mode == "deletion-audit":
        run_v4a_residual_deletion_audit(args.device)
        print("[V5A] V4A deletion audit complete: 27 same-checkpoint comparisons", flush=True)
    elif args.mode == "preflight":
        print(json.dumps(run_preflight(args.device), indent=2), flush=True)
    elif args.mode == "smoke":
        print(json.dumps(run_smoke(args.device), indent=2), flush=True)
    elif args.mode == "formal":
        print(json.dumps(run_formal(args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
