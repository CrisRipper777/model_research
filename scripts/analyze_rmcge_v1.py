#!/usr/bin/env python3
"""Validation-only frozen-checkpoint diagnostics for RM-CGE V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Grocery", "ele-fashion")


def _metric_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: list[int],
) -> dict[str, float]:
    prediction = logits.argmax(dim=-1).cpu()
    targets = targets.detach().cpu()
    return {
        "val_acc": float((prediction == targets).float().mean().item()),
        "val_macro_f1": float(
            f1_score(
                targets.numpy(),
                prediction.numpy(),
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _flatten_layer_diagnostics(details: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in details["layers"]:
        for modality, modality_detail in layer["modalities"].items():
            for relation, relation_detail in modality_detail["relations"].items():
                rows.append(
                    {
                        "layer": int(layer["layer"]),
                        "modality": modality,
                        "relation": relation,
                        **relation_detail,
                        "relation_marginal": modality_detail[
                            "mean_joint_relation_marginal"
                        ],
                        "expert_use": modality_detail["mean_joint_expert_use"],
                        "route_entropy_nats": modality_detail[
                            "joint_route_entropy_nats"
                        ],
                        "delta_rms": modality_detail["delta_rms"],
                        "relative_update_rms": modality_detail[
                            "relative_update_rms"
                        ],
                    }
                )
    return rows


def analyze_checkpoint(
    *,
    dataset: str,
    checkpoint_path: str | Path,
    cache_entry: dict[str, Any],
    device: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run diagnostics on one selected checkpoint without indexing Test labels."""
    from scripts.run_rmcge_v1 import _git, build_config, now_utc, write_json
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels
    from src.utils.summary import count_parameters

    if dataset not in DATASETS:
        raise ValueError(f"unsupported RM-CGE dataset {dataset!r}")
    cfg = build_config(dataset, device=device, seed=42)
    cfg.model.semantic_cache_path = cache_entry.get(
        "cache_path_abs", str(ROOT / cache_entry["cache_path"])
    )
    cfg.model.semantic_feature_fingerprint = cache_entry["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache_entry["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache_entry["v7b_fingerprint"]
    data = load_mag_data(cfg, "nc", 42)
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset}: NC loader did not return both modalities")
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }
    model = build_model(cfg, data_info)
    model.load_semantic_cache(
        cfg.model.semantic_cache_path,
        expected_dataset=dataset,
        expected_feature_fingerprint=cache_entry["feature_fingerprint"],
        expected_candidate_fingerprint=cache_entry["candidate_fingerprint"],
        expected_v7b_fingerprint=cache_entry["v7b_fingerprint"],
    )
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = ROOT / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("selection") != "best_val_accuracy":
        raise ValueError("diagnostics require a Validation Accuracy-selected checkpoint")
    if set(checkpoint.get("metrics", {})) != {"val_acc", "val_macro_f1"}:
        raise ValueError("checkpoint contains non-Validation metrics")
    if bool(cfg.task.evaluate_test):
        raise RuntimeError("RM-CGE analysis requires evaluate_test=false")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes))
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    model = model.to(device).eval()
    classifier = classifier.to(device).eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    val_idx = data.val_idx.to(device)
    val_targets = data.y[data.val_idx].detach().cpu()
    metric_labels = _resolve_nc_eval_labels(data, include_test=False)

    with torch.no_grad():
        z, _, _, aux_loss, model_info = model(
            x, edge_index, return_details=True
        )
        base_logits = classifier(z)
        base_val_logits = base_logits[val_idx].detach().cpu()
        baseline = _metric_summary(base_val_logits, val_targets, metric_labels)
        if not torch.isfinite(z).all() or not torch.isfinite(aux_loss):
            raise FloatingPointError("checkpoint produced non-finite embeddings or aux loss")

        interventions = {
            "cross_modal_off": {"disable_cross_modal": True},
            "semantic_relation_off": {"disable_relations": ["semantic"]},
            "physical_relation_off": {"disable_relations": ["physical"]},
            "graph_updates_off": {
                "disable_relations": ["physical", "semantic"]
            },
            "intrinsic_skip_off": {"disable_intrinsic_skip": True},
        }
        intervention_rows: dict[str, Any] = {}
        base_prediction = base_val_logits.argmax(dim=-1)
        base_logit_rms = float(base_val_logits.float().square().mean().sqrt())
        base_embedding_rms = float(z.float().square().mean().sqrt())
        for name, intervention in interventions.items():
            changed_z = model(x, edge_index, intervention=intervention)[0]
            changed_logits = classifier(changed_z)[val_idx].detach().cpu()
            changed_metric = _metric_summary(
                changed_logits, val_targets, metric_labels
            )
            changed_prediction = changed_logits.argmax(dim=-1)
            intervention_rows[name] = {
                **changed_metric,
                "val_acc_delta_percentage_points": (
                    changed_metric["val_acc"] - baseline["val_acc"]
                )
                * 100.0,
                "val_macro_f1_delta_percentage_points": (
                    changed_metric["val_macro_f1"] - baseline["val_macro_f1"]
                )
                * 100.0,
                "prediction_flip_fraction": float(
                    (changed_prediction != base_prediction).float().mean()
                ),
                "val_logit_relative_delta_rms": float(
                    (changed_logits - base_val_logits).float().square().mean().sqrt()
                )
                / max(base_logit_rms, 1.0e-12),
                "embedding_relative_delta_rms": float(
                    (changed_z - z).float().square().mean().sqrt()
                )
                / max(base_embedding_rms, 1.0e-12),
            }
            del changed_z, changed_logits

    details = model_info["details"]
    relation_rows = _flatten_layer_diagnostics(details)
    cache = model.semantic_metadata or {}
    flags = torch.load(
        cfg.model.semantic_cache_path, map_location="cpu", weights_only=False
    )["source_flags"]
    source_count_histogram = {
        source: {
            str(value): int((flags[:, column] == value).sum())
            for value in range(3)
        }
        for column, source in ((0, "text"), (1, "visual"))
    }
    physical_degree = model.physical_degree.detach().cpu()
    support_summary = {
        "physical": {
            "directed_edges": int(model.physical_operator._nnz()),
            "valid_nodes": int((physical_degree > 0).sum()),
            "valid_node_fraction": float((physical_degree > 0).float().mean()),
        },
        "text_semantic": {
            "directed_edges": int(model.semantic_pt._nnz()),
            "valid_nodes": int((model.semantic_degree_text > 0).sum()),
            "valid_node_fraction": float(
                (model.semantic_degree_text > 0).float().mean()
            ),
        },
        "visual_semantic": {
            "directed_edges": int(model.semantic_pv._nnz()),
            "valid_nodes": int((model.semantic_degree_visual > 0).sum()),
            "valid_node_fraction": float(
                (model.semantic_degree_visual > 0).float().mean()
            ),
        },
    }
    result = {
        "recorded_at_utc": now_utc(),
        "dataset": dataset,
        "seed": int(checkpoint["seed"]),
        "checkpoint_path": str(path.relative_to(ROOT))
        if path.is_relative_to(ROOT)
        else str(path),
        "commit_sha": _git("rev-parse", "HEAD"),
        "checkpoint_selection": checkpoint["selection"],
        "best_epoch": int(checkpoint["epoch"]),
        "validation_metrics": baseline,
        "checkpoint_metrics_match_recomputed_validation": all(
            abs(float(checkpoint["metrics"][key]) - baseline[key]) < 1.0e-7
            for key in ("val_acc", "val_macro_f1")
        ),
        "parameter_count": count_parameters(model)
        + count_parameters(classifier),
        "cache_path": str(cfg.model.semantic_cache_path),
        "cache_fingerprints": {
            "feature": cache.get("feature_fingerprint"),
            "candidate": cache.get("candidate_fingerprint"),
            "v7b": cache.get("v7b_fingerprint"),
        },
        "source_flags_count_0_1_2": source_count_histogram,
        "supports": support_summary,
        "layers": details["layers"],
        "relation_diagnostics": relation_rows,
        "fusion": details["fusion"],
        "frozen_interventions": intervention_rows,
        "aux_loss": float(aux_loss.detach().cpu()),
        "evaluate_test": False,
        "test_labels_indexed": False,
        "test_metrics_computed": False,
        "interpretation": (
            "Intervention deltas are frozen-checkpoint Validation diagnostics; "
            "they are not retrained ablations or estimates of causal generalization."
        ),
    }
    if output_path is not None:
        out = Path(output_path)
        if not out.is_absolute():
            out = ROOT / out
        write_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-entry", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scripts.run_rmcge_v1 import _choose_device

    device = _choose_device(args.device)
    entry = json.loads(args.cache_entry.read_text(encoding="utf-8"))
    result = analyze_checkpoint(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        cache_entry=entry,
        device=device,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
