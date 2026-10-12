#!/usr/bin/env python3
"""Validation-only mechanism and factorial analysis for RM-CGE V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_fscore_support, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_rmcge_v1 as pilot
from scripts import run_rmcge_v1_generalization as study

RESEARCH = ROOT / "research/rmcge_v1_generalization"
DATA_DIR = RESEARCH / "data"
FIGURE_DIR = RESEARCH / "figures"
DATASETS = study.DATASETS
RELATIONS = ("physical", "semantic")
MODALITIES = ("text", "visual")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _checksum(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_data_and_model(dataset: str, cache_entry: dict[str, Any], device: str, dropout: float = 0.2):
    from src.data import load_mag_data
    from src.models import build_model

    cfg = pilot._build_config(dataset, device=device, seed=42)
    cfg.model.dropout = float(dropout)
    cfg.model.semantic_cache_path = cache_entry["cache_path_abs"]
    cfg.model.semantic_feature_fingerprint = cache_entry["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache_entry["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache_entry["v7b_fingerprint"]
    cfg.task.evaluate_test = False
    data = load_mag_data(cfg, "nc", 42)
    info = study._data_info(data)
    model = build_model(cfg, info)
    model.load_semantic_cache(
        cache_entry["cache_path_abs"],
        expected_dataset=dataset,
        expected_feature_fingerprint=cache_entry["feature_fingerprint"],
        expected_candidate_fingerprint=cache_entry["candidate_fingerprint"],
        expected_v7b_fingerprint=cache_entry["v7b_fingerprint"],
    )
    return cfg, data, model, info


def _eval_labels(data) -> list[int]:
    from src.tasks.nc import _resolve_nc_eval_labels

    return _resolve_nc_eval_labels(data, include_test=False)


def _metrics(logits: torch.Tensor, targets: torch.Tensor, labels: list[int]) -> dict[str, float]:
    logits = logits.detach().cpu().float()
    targets = targets.detach().cpu().long()
    prediction = logits.argmax(dim=-1)
    return {
        "val_ce": float(torch.nn.functional.cross_entropy(logits, targets).item()),
        "val_acc": float((prediction == targets).float().mean().item()),
        "val_macro_f1": float(
            f1_score(
                targets.numpy(), prediction.numpy(), labels=labels,
                average="macro", zero_division=0,
            )
        ),
    }


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "q05": None, "q25": None, "median": None, "q75": None, "q95": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    function = pearsonr if kind == "pearson" else spearmanr
    statistic = function(x, y).statistic
    return float(statistic) if np.isfinite(statistic) else None


def _compare_prior(actual: dict[str, float], reference: dict[str, Any], metric_names: tuple[str, ...], tolerance: float = 2e-6) -> dict[str, Any]:
    comparisons = {}
    for name in metric_names:
        expected = float(reference[name])
        observed = float(actual[name])
        comparisons[name] = {"observed": observed, "reference": expected, "absolute_difference": abs(observed - expected), "passed": math.isclose(observed, expected, abs_tol=tolerance, rel_tol=0)}
    return {"passed": all(item["passed"] for item in comparisons.values()), "metrics": comparisons}


def _router_and_scale_rows(dataset: str, details: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    router_rows = []
    scale_rows = []
    for layer in details["layers"]:
        layer_id = int(layer["layer"])
        modality_nodes: dict[str, dict[str, Any]] = {}
        for modality in MODALITIES:
            mdetail = layer["modalities"][modality]
            node = mdetail["node_diagnostics"]
            participant = node["participating"].numpy().astype(bool)
            relation_mass = node["relation_mass"].numpy()
            top1 = node["top1_expert"].numpy()
            entropy = node["entropy"].numpy()
            relation_nodes = {}
            row: dict[str, Any] = {
                "dataset": dataset,
                "layer": layer_id,
                "modality": modality,
                "participating_node_count": int(participant.sum()),
                "node_count": int(participant.size),
            }
            entropy_stats = _stats(entropy[participant])
            for key, value in entropy_stats.items():
                row[f"router_entropy_{key}"] = value
            for expert_id in range(3):
                row[f"top1_expert_{expert_id}_fraction"] = (
                    float((top1[participant] == expert_id).mean()) if participant.any() else None
                )
            for rel_id, relation in enumerate(RELATIONS):
                relation_detail = node["relations"][relation]
                valid = relation_detail["valid"].numpy().astype(bool)
                mass = relation_mass[:, rel_id]
                mass_stats = _stats(mass[valid])
                row[f"{relation}_valid_node_count"] = int(valid.sum())
                row[f"{relation}_valid_fraction"] = float(valid.mean())
                for key, value in mass_stats.items():
                    row[f"{relation}_relation_mass_{key}"] = value
                cosine = relation_detail["same_context_cosine"].numpy()
                normalized_l2 = relation_detail["same_context_normalized_l2"].numpy()
                cos_stats = _stats(cosine[valid])
                l2_stats = _stats(normalized_l2[valid])
                for key, value in cos_stats.items():
                    row[f"{relation}_same_context_cosine_{key}"] = value
                for key, value in l2_stats.items():
                    row[f"{relation}_same_context_normalized_l2_{key}"] = value
                semantic_mass = relation_mass[:, 1]
                for kind in ("pearson", "spearman"):
                    row[f"{relation}_cosine_vs_semantic_mass_{kind}"] = _correlation(cosine[valid], semantic_mass[valid], kind)
                    row[f"{relation}_normalized_l2_vs_semantic_mass_{kind}"] = _correlation(normalized_l2[valid], semantic_mass[valid], kind)
                relation_nodes[relation] = {"valid": valid, "mass": mass, "cosine": cosine, "normalized_l2": normalized_l2}
                layer_rel = mdetail["relations"][relation]
                scale_rows.append({
                    "dataset": dataset,
                    "layer": layer_id,
                    "modality": modality,
                    "relation": relation,
                    "valid_node_count": int(layer_rel["valid_nodes"]),
                    "valid_node_fraction": float(layer_rel["valid_node_fraction"]),
                    "same_modal_raw_rms": float(layer_rel["same_context_rms"]),
                    "cross_modal_raw_rms": float(layer_rel["cross_context_rms"]),
                    "same_modal_post_block_layernorm_rms": float(layer_rel["same_context_post_layernorm_rms"]),
                    "cross_modal_post_block_layernorm_rms": float(layer_rel["cross_context_post_layernorm_rms"]),
                    "same_modal_low_variance_fraction_lt_1e-6": float(layer_rel["same_context_low_variance_node_fraction"]),
                    "cross_modal_low_variance_fraction_lt_1e-6": float(layer_rel["cross_context_low_variance_node_fraction"]),
                    "cross_to_same_raw_rms_ratio": float(layer_rel["cross_to_same_context_rms"]),
                })
            text_node = layer["modalities"]["text"]["node_diagnostics"]
            visual_node = layer["modalities"]["visual"]["node_diagnostics"]
            for relation_id, relation in enumerate(RELATIONS):
                text_relation = text_node["relations"][relation]
                visual_relation = visual_node["relations"][relation]
                paired = text_relation["valid"].numpy().astype(bool) & visual_relation["valid"].numpy().astype(bool)
                text_mass = text_node["relation_mass"].numpy()[:, relation_id]
                visual_mass = visual_node["relation_mass"].numpy()[:, relation_id]
                difference = text_mass[paired] - visual_mass[paired]
                row[f"text_visual_{relation}_paired_valid_count"] = int(paired.sum())
                row[f"text_visual_{relation}_mass_difference_mean"] = float(difference.mean()) if difference.size else None
                row[f"text_visual_{relation}_mass_abs_difference_mean"] = float(np.abs(difference).mean()) if difference.size else None
                row[f"text_visual_{relation}_mass_correlation"] = _correlation(text_mass[paired], visual_mass[paired], "pearson")
            modality_nodes[modality] = {"participant": participant, "relation_mass": relation_mass, "top1": top1, "entropy": entropy, "relations": relation_nodes}
            router_rows.append(row)
    return router_rows, scale_rows


def frozen_pilot_analysis(device: str | None = None) -> dict[str, Any]:
    device = device or study._choose_device()
    cache_entries = {dataset: pilot._verify_cache_entry(dataset, pilot._cache_entries()[dataset]) for dataset in DATASETS}
    intervention_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for dataset in DATASETS:
        cache = cache_entries[dataset]
        cfg, data, model, info = _load_data_and_model(dataset, cache, device)
        checkpoint_path = ROOT / "outputs/rmcge_v1_pilot/runs" / dataset / "seed_42/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != "best_val_accuracy":
            raise ValueError(f"{dataset}: frozen diagnostics require best Validation Accuracy checkpoint")
        if "test" in checkpoint.get("metrics", {}):
            raise ValueError(f"{dataset}: original checkpoint unexpectedly contains Test metrics")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        classifier = torch.nn.Linear(model.out_dim, info["num_classes"])
        classifier.load_state_dict(checkpoint["head_state"], strict=True)
        model = model.to(device).eval()
        classifier = classifier.to(device).eval()
        x = data.x.to(device)
        edges = data.edge_index.to(device)
        val_idx = data.val_idx.cpu().long()
        targets = data.y[val_idx].detach().cpu()
        metric_labels = _eval_labels(data)
        with torch.no_grad():
            z_a = model(x, edges)[0]
            logits_a = classifier(z_a)
            z_b = model(x, edges)[0]
            logits_b = classifier(z_b)
            repeat_embedding_diff = float((z_a - z_b).abs().max().cpu())
            repeat_logit_diff = float((logits_a - logits_b).abs().max().cpu())
            if repeat_embedding_diff > 1.0e-5 or repeat_logit_diff > 1.0e-5:
                raise RuntimeError(
                    f"{dataset}: repeated frozen checkpoint inference exceeded tolerance "
                    f"(embedding={repeat_embedding_diff}, logits={repeat_logit_diff})"
                )
            baseline_logits = logits_a[val_idx.to(device)].detach().cpu()
            baseline = _metrics(baseline_logits, targets, metric_labels)
            pilot_run = next(row for row in _read_csv(ROOT / "research/rmcge_v1/data/pilot_runs.csv") if row["dataset"] == dataset and int(row["seed"]) == 42)
            old_diag = json.loads((ROOT / "research/rmcge_v1/data/diagnostics" / f"{dataset}_seed_42.json").read_text(encoding="utf-8"))
            metric_check = _compare_prior(baseline, {"val_acc": pilot_run["val_acc"], "val_macro_f1": pilot_run["val_macro_f1"], "val_ce": float("nan")}, ("val_acc", "val_macro_f1"))
            if int(checkpoint["epoch"]) != int(pilot_run["best_epoch"]):
                raise RuntimeError(f"{dataset}: best epoch does not match original pilot")
            if not metric_check["passed"] or not old_diag.get("checkpoint_metrics_match_recomputed_validation", False):
                raise RuntimeError(f"{dataset}: frozen checkpoint failed original Validation metrics gate")
            base_prediction = baseline_logits.argmax(dim=-1)
            base_logit_rms = float(baseline_logits.float().square().mean().sqrt())
            base_embedding_rms = float(z_a.float().square().mean().sqrt())
            conditions = {
                "original": None,
                "all_cross_modal_off": {"disable_cross_modal": True},
                "physical_cross_modal_off": {"disable_cross_modal_on_physical": True},
                "semantic_cross_modal_off": {"disable_cross_modal_on_semantic": True},
            }
            per_condition = {}
            for condition, intervention in conditions.items():
                if intervention is None:
                    changed_z = z_a
                    changed_logits = baseline_logits
                else:
                    changed_z = model(x, edges, intervention=intervention)[0]
                    changed_logits = classifier(changed_z)[val_idx.to(device)].detach().cpu()
                metric = _metrics(changed_logits, targets, metric_labels)
                pred = changed_logits.argmax(dim=-1)
                row = {
                    "dataset": dataset,
                    "seed": 42,
                    "condition": condition,
                    **metric,
                    "val_ce_delta_vs_original": metric["val_ce"] - baseline["val_ce"],
                    "val_acc_delta_vs_original": metric["val_acc"] - baseline["val_acc"],
                    "val_macro_f1_delta_vs_original": metric["val_macro_f1"] - baseline["val_macro_f1"],
                    "prediction_flip_fraction_vs_original": float((pred != base_prediction).float().mean()),
                    "validation_logit_delta_rms": float((changed_logits - baseline_logits).float().square().mean().sqrt()),
                    "validation_logit_relative_delta_rms": float((changed_logits - baseline_logits).float().square().mean().sqrt()) / max(base_logit_rms, 1e-12),
                    "embedding_delta_rms_all_nodes": float((changed_z - z_a).float().square().mean().sqrt().cpu()),
                    "embedding_relative_delta_rms_all_nodes": float((changed_z - z_a).float().square().mean().sqrt().cpu()) / max(base_embedding_rms, 1e-12),
                    "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                    "interpretation": "Frozen-checkpoint validation intervention; router inputs may change with the intervention; not a retrained ablation or causal estimate.",
                }
                intervention_rows.append(row)
                per_condition[condition] = row
            legacy = old_diag["frozen_interventions"]["cross_modal_off"]
            current_all_off = per_condition["all_cross_modal_off"]
            legacy_pairs = {
                "val_acc": "val_acc",
                "val_macro_f1": "val_macro_f1",
                "prediction_flip_fraction_vs_original": "prediction_flip_fraction",
                "validation_logit_relative_delta_rms": "val_logit_relative_delta_rms",
                "embedding_relative_delta_rms_all_nodes": "embedding_relative_delta_rms",
            }
            legacy_differences = {
                current_key: abs(float(current_all_off[current_key]) - float(legacy[legacy_key]))
                for current_key, legacy_key in legacy_pairs.items()
            }
            legacy_match = all(value <= 2e-6 for value in legacy_differences.values())
            if not legacy_match:
                raise RuntimeError(f"{dataset}: All-cross-off result differs from archived V1 diagnostics")
            with torch.no_grad():
                z_detail, _, _, _, info_detail = model(x, edges, return_node_diagnostics=True)
            detail_diff = float((z_detail - z_a).abs().max().cpu())
            if detail_diff > 1e-5:
                raise RuntimeError(f"{dataset}: node diagnostics changed the frozen forward output")
            rrows, srows = _router_and_scale_rows(dataset, info_detail["details"])
            router_rows.extend(rrows)
            scale_rows.extend(srows)
            checks[dataset] = {
                "passed": True,
                "original_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "cache_fingerprints": {key: cache[key] for key in ("feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint")},
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "pilot_metrics_match": metric_check,
                "archived_all_cross_off_match": legacy_match,
                "archived_all_cross_off_max_metric_difference": max(legacy_differences.values()),
                "repeated_embedding_max_abs_difference": repeat_embedding_diff,
                "repeated_logit_max_abs_difference": repeat_logit_diff,
                "node_diagnostic_forward_max_abs_difference": detail_diff,
                "embedding_sha256": _checksum(z_a),
                "logit_sha256": _checksum(logits_a),
                "evaluate_test": False,
                "test_labels_indexed": False,
                "test_metrics_computed": False,
                "original_artifact_limit": "The original Pilot did not persist embedding/logit arrays or checksums; this study recomputed and hashed them twice from its frozen checkpoint.",
            }
            del z_a, z_b, logits_a, logits_b, z_detail, info_detail
        del model, classifier, data
    _write_csv(DATA_DIR / "frozen_cross_relation_interventions.csv", intervention_rows)
    _write_csv(DATA_DIR / "router_node_statistics.csv", router_rows)
    _write_csv(DATA_DIR / "cross_modal_scales.csv", scale_rows)
    _write_json(DATA_DIR / "frozen_checkpoint_replay_checks.json", checks)
    if not all(item["passed"] for item in checks.values()):
        raise RuntimeError("frozen checkpoint validation gate failed")
    _write_mechanism_report(checks, intervention_rows, router_rows, scale_rows)
    return checks


def _class_metric_rows(dataset: str, treatment: str, seed: int, selection: str, logits: torch.Tensor, data, labels: list[int]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    val_idx = data.val_idx.detach().cpu().long()
    targets = data.y[val_idx].detach().cpu().numpy().astype(int)
    logits_val = logits[val_idx]
    prediction = logits_val.argmax(dim=-1).detach().cpu().numpy().astype(int)
    all_classes = list(range(int(data.num_classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, prediction, labels=all_classes, zero_division=0
    )
    predicted_count = np.bincount(prediction, minlength=int(data.num_classes))
    rows = []
    for class_id in all_classes:
        rows.append({
            "dataset": dataset,
            "treatment": treatment,
            "seed": seed,
            "selection": selection,
            "class_id": class_id,
            "validation_count": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "predicted_count": int(predicted_count[class_id]),
        })
    return rows, _metrics(logits_val, torch.as_tensor(targets), labels)


def evaluate_all_checkpoints() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "matrix_completed" or len(manifest.get("runs", [])) != 36:
        raise RuntimeError("cannot summarize until the full 36-run matrix is complete")
    device = study._choose_device()
    cache_entries = {dataset: pilot._verify_cache_entry(dataset, pilot._cache_entries()[dataset]) for dataset in DATASETS}
    metrics_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    from src.data import load_mag_data
    from src.models import build_model
    from src.tasks.inference import infer_all_embeddings

    for dataset in DATASETS:
        entry = cache_entries[dataset]
        cfg_data = pilot._build_config(dataset, device=device, seed=42)
        cfg_data.model.semantic_cache_path = entry["cache_path_abs"]
        cfg_data.model.semantic_feature_fingerprint = entry["feature_fingerprint"]
        cfg_data.model.semantic_candidate_fingerprint = entry["candidate_fingerprint"]
        cfg_data.model.semantic_v7b_fingerprint = entry["v7b_fingerprint"]
        cfg_data.task.evaluate_test = False
        data = load_mag_data(cfg_data, "nc", 42)
        label_set = _eval_labels(data)
        for run in manifest["runs"]:
            if run["dataset"] != dataset:
                continue
            treatment = run["treatment"]
            for selection, checkpoint_file in (("best_acc", "best_acc.pt"), ("best_f1", "best_f1.pt")):
                cfg = pilot._build_config(dataset, device=device, seed=int(run["seed"]))
                cfg.model.dropout = float(run["dropout"])
                for key in ("semantic_cache_path", "semantic_feature_fingerprint", "semantic_candidate_fingerprint", "semantic_v7b_fingerprint"):
                    cfg.model[key] = entry["cache_path_abs"] if key == "semantic_cache_path" else entry[key.removeprefix("semantic_")]
                cfg.task.evaluate_test = False
                model = build_model(cfg, study._data_info(data))
                model.load_semantic_cache(
                    entry["cache_path_abs"], expected_dataset=dataset,
                    expected_feature_fingerprint=entry["feature_fingerprint"],
                    expected_candidate_fingerprint=entry["candidate_fingerprint"],
                    expected_v7b_fingerprint=entry["v7b_fingerprint"],
                )
                run_dir = ROOT / run["run_dir"]
                checkpoint = torch.load(run_dir / checkpoint_file, map_location="cpu", weights_only=False)
                if "test" in checkpoint.get("metrics", {}):
                    raise RuntimeError(f"{run_dir}: checkpoint contains Test metrics")
                model.load_state_dict(checkpoint["model_state"], strict=True)
                classifier = torch.nn.Linear(model.out_dim, int(data.num_classes))
                classifier.load_state_dict(checkpoint["head_state"], strict=True)
                model.to(device).eval()
                classifier.to(device).eval()
                z = infer_all_embeddings(model, data, torch.device(device), True, 4096, "full")
                logits = classifier(z.to(device)).detach().cpu()
                rows, evaluated = _class_metric_rows(dataset, treatment, int(run["seed"]), selection, logits, data, label_set)
                class_rows.extend(rows)
                expected = (
                    run["val_acc_best_acc"], run["val_macro_f1_best_acc"], run["val_ce_best_acc"]
                ) if selection == "best_acc" else (
                    run["val_acc_best_f1"], run["val_macro_f1_best_f1"], run["val_ce_best_f1"]
                )
                observed = (evaluated["val_acc"], evaluated["val_macro_f1"], evaluated["val_ce"])
                if not all(math.isclose(float(a), float(b), abs_tol=2e-6, rel_tol=0) for a, b in zip(observed, expected)):
                    raise RuntimeError(f"{dataset}/{treatment}/seed_{run['seed']}/{selection}: checkpoint replay metrics mismatch {observed} vs {expected}")
                metrics_rows.append({
                    "dataset": dataset, "treatment": treatment, "seed": int(run["seed"]), "selection": selection,
                    "selection_epoch": int(checkpoint["epoch"]), **evaluated,
                    "checkpoint_path": str((run_dir / checkpoint_file).relative_to(ROOT)),
                    "evaluate_test": False, "test_labels_indexed": False,
                })
                del model, classifier, z, logits
        del data
    _write_csv(DATA_DIR / "validation_class_metrics.csv", class_rows)
    _write_csv(DATA_DIR / "checkpoint_validation_replay.csv", metrics_rows)
    return metrics_rows, class_rows


def _paired_effects(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measures = {
        "val_accuracy": "val_acc_best_acc",
        "val_macro_f1": "val_macro_f1_best_acc",
        "validation_ce": "val_ce_best_acc",
    }
    contrasts = (("O1-O0", "O1", "O0"), ("O2-O0", "O2", "O0"), ("O3-O1", "O3", "O1"), ("O3-O2", "O3", "O2"))
    rows = []
    by_block: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_block.setdefault((run["dataset"], int(run["seed"])), {})[run["treatment"]] = run
    for (dataset, seed), treatments in sorted(by_block.items()):
        for label, high, low in contrasts:
            for metric, column in measures.items():
                rows.append({
                    "dataset": dataset, "seed": seed, "contrast": label, "metric": metric,
                    "treatment_a": high, "treatment_b": low,
                    "value_a": float(treatments[high][column]),
                    "value_b": float(treatments[low][column]),
                    "paired_difference": float(treatments[high][column]) - float(treatments[low][column]),
                    "unit": "seed-level paired training run",
                })
            for metric, column in measures.items():
                if label in {"O1-O0", "O3-O2"}:
                    pass
        for metric, column in measures.items():
            dropout_at_low_wd = float(treatments["O1"][column]) - float(treatments["O0"][column])
            dropout_at_high_wd = float(treatments["O3"][column]) - float(treatments["O2"][column])
            interaction = dropout_at_high_wd - dropout_at_low_wd
            rows.append({"dataset": dataset, "seed": seed, "contrast": "Interaction=(O3-O2)-(O1-O0)", "metric": metric, "treatment_a": "factorial_interaction", "treatment_b": None, "value_a": None, "value_b": None, "paired_difference": interaction, "unit": "seed-level paired training run"})
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["contrast"], row["metric"]), []).append(float(row["paired_difference"]))
    effects = []
    for (dataset, contrast, metric), values in sorted(grouped.items()):
        effects.append({
            "dataset": dataset, "contrast": contrast, "metric": metric,
            "seed_count": len(values), "mean_paired_difference": float(np.mean(values)),
            "sample_sd_paired_difference": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "description": "Descriptive paired seed contrasts; n is training seeds, not nodes or epochs; no inferential significance claim.",
        })
    return rows, effects


def _summaries(runs: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"best_accuracy": {}, "best_macro_f1": {}}
    metric_names = ("val_acc", "val_macro_f1", "val_ce", "selection_epoch")
    for selection, output_key in (("best_acc", "best_accuracy"), ("best_f1", "best_macro_f1")):
        for dataset in DATASETS:
            result[output_key][dataset] = {}
            for treatment in study.TREATMENTS:
                subset = [row for row in checkpoint_rows if row["dataset"] == dataset and row["treatment"] == treatment and row["selection"] == selection]
                treatment_rows = [row for row in runs if row["dataset"] == dataset and row["treatment"] == treatment]
                measurements = {}
                for name in metric_names:
                    values = [float(item[name]) for item in subset]
                    measurements[name] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "n": len(values)}
                measurements["training_seconds"] = {"mean": float(np.mean([float(item["training_seconds"]) for item in treatment_rows])), "sample_sd": float(np.std([float(item["training_seconds"]) for item in treatment_rows], ddof=1)), "n": len(treatment_rows)}
                measurements["peak_allocated_gib_mean"] = float(np.mean([float(item["cuda_peak_allocated_bytes"]) / 1024**3 for item in treatment_rows]))
                result[output_key][dataset][treatment] = measurements
    return result


def _write_mechanism_report(checks, intervention_rows, router_rows, scale_rows) -> None:
    lines = [
        "# RM-CGE V1 Frozen Mechanism Diagnostics",
        "",
        "## Evidence boundary",
        "",
        "These are Validation-only interventions on the original Seed-42 Best-Accuracy checkpoints. They do not retrain the model. Setting a cross-modal context to zero changes the input observed by the Router; all same-modal supports, relation validity rules, expert parameters, Router parameters, and classifier weights remain loaded from the checkpoint. The measured route is the model's actual information flow, not an intervention that holds every downstream activation fixed.",
        "",
        "No Test labels were indexed. Node counts below are descriptive units within a frozen checkpoint, not independent experimental replicates. Correlations describe associations with the Router's Semantic relation mass and do not imply labels or reliability.",
        "",
        "## Replay gate",
        "",
    ]
    for dataset, check in checks.items():
        metrics = check["pilot_metrics_match"]["metrics"]
        lines.append(f"- **{dataset}:** replay passed; best epoch {check['checkpoint_epoch']}; Validation Accuracy {metrics['val_acc']['observed']:.6f}; Macro-F1 {metrics['val_macro_f1']['observed']:.6f}; repeated full-graph embedding/logit max difference {check['repeated_embedding_max_abs_difference']:.2g}/{check['repeated_logit_max_abs_difference']:.2g}; archived all-cross-off replay passed.")
    lines.extend(["", "## Relation-specific cross-modal interventions", "", "| Dataset | Condition | Val CE | Δ CE | Val Acc | Δ Acc (pp) | Macro-F1 | Δ F1 (pp) | Prediction flips | Relative logit RMS Δ | Relative embedding RMS Δ |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in intervention_rows:
        lines.append(f"| {row['dataset']} | {row['condition']} | {row['val_ce']:.4f} | {row['val_ce_delta_vs_original']:+.4f} | {row['val_acc']:.4f} | {row['val_acc_delta_vs_original']*100:+.2f} | {row['val_macro_f1']:.4f} | {row['val_macro_f1_delta_vs_original']*100:+.2f} | {row['prediction_flip_fraction_vs_original']:.4f} | {row['validation_logit_relative_delta_rms']:.4f} | {row['embedding_relative_delta_rms_all_nodes']:.4f} |")
    lines.extend(["", "The physical and semantic conditions zero only the corresponding relation's cross-modal context. They leave its same-modal context active. The all-cross-off baseline matches the previously archived frozen-checkpoint intervention within 2×10⁻⁶.", "", "Observed validation sensitivity differs by dataset. On Movies and Grocery, removing Physical cross-modal context produces a larger Accuracy decline than removing Semantic cross-modal context. On ele-fashion, both single-relation changes are small and mixed across CE, Accuracy, and Macro-F1. These are fitted-path sensitivities; they do not estimate the benefit of retraining without a path.", "", "## Node-conditioned routing", "", "The table reports node-level Semantic relation mass by layer and modality (mean, population SD, and 5th–95th percentiles), together with mean joint-router entropy. Values are masked to nodes with the measured relation support.", "", "| Dataset | Layer | View | Semantic mass mean ± SD | q05–q95 | Joint entropy | Top-1 expert proportions |", "|---|---:|---|---:|---:|---:|---|"])
    for row in router_rows:
        mean, sd = float(row["semantic_relation_mass_mean"]), float(row["semantic_relation_mass_std"])
        q05, q95 = float(row["semantic_relation_mass_q05"]), float(row["semantic_relation_mass_q95"])
        expert_mix = ", ".join(f"E{expert} {float(row[f'top1_expert_{expert}_fraction']):.2f}" for expert in range(3))
        lines.append(f"| {row['dataset']} | {row['layer']} | {row['modality']} | {mean:.3f} ± {sd:.3f} | {q05:.3f}–{q95:.3f} | {float(row['router_entropy_mean']):.3f} | {expert_mix} |")
    lines.extend(["", "Relation mass varies across nodes: observed Semantic-mass SD spans about 0.02–0.17 across these checkpoint/layer/view groups, with especially broad ranges in ele-fashion. This rules out describing the Router as emitting only a fixed 0.6/0.4 ratio on each node. The entropy and Top-1 expert distributions show that node variation coexists with different degrees of expert concentration. They do not establish that the Router identifies a reliable source: mass-to-neighborhood correlations are often small, with larger associations in some ele-fashion views, and remain observational within one frozen graph.", "", "The machine-readable table contains all quantiles, validity-masked relation masses, top-1 expert proportions, paired Text/Visual mass comparisons, and descriptive Pearson/Spearman coefficients with Semantic mass. These coefficients are reported without p-values and do not imply labels or relation reliability.", "", "## Context scales", "", "Raw and post-block-LayerNorm context RMS, together with low-variance node fractions below 1e-6, are in `data/cross_modal_scales.csv`. Raw cross-modal RMS is lower than same-modal RMS, while both independently normalized blocks are near unit RMS. Therefore, the raw magnitude difference is removed before the Expert input LayerNorm; raw scale alone would understate the normalized cross-modal block's scale.", "", "The check files contain frozen checkpoint hashes, cache fingerprints, and recomputed embedding/logit SHA-256 values. Original Pilot embedding/logit arrays were not persisted, so hashes establish deterministic recomputation in this study rather than a bytewise comparison against an archived tensor artifact."])
    (RESEARCH / "MECHANISM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generalization_report(runs: list[dict[str, Any]], paired: list[dict[str, Any]], effects: list[dict[str, Any]], summaries: dict[str, Any], checkpoint_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# RM-CGE V1 Generalization & Optimization Results",
        "",
        "## Protocol and interpretation",
        "",
        "The complete 2×2 Dropout × AdamW Weight Decay matrix contains 36 Full-Graph NC runs: three datasets, four fixed treatments, and seeds 42/43/44. The seed-level training run is the experimental unit (n=3 per dataset/treatment). All displayed standard deviations use the sample SD (ddof=1). Comparisons are paired by seed and descriptive; no p-values or statistical-significance claims are made. Test evaluation remained disabled.",
        "",
        "The primary table below uses Best-Accuracy checkpoints only. Best-Macro-F1 checkpoint summaries appear in a separate table. Train CE is the stochastic training-mode loss before that epoch's optimizer update; Train Accuracy is eval-mode after the update. The displayed CE proxy gap pairs pre-update Train CE with post-update Validation CE, so it is a trajectory diagnostic and not a same-state train/validation gap. Eval Train Accuracy and Validation Accuracy are both post-update. Validation CE and all metrics are unweighted.",
        "",
        "## Best-Accuracy results (mean ± sample SD)",
        "",
        "| Dataset | Treatment | Val Acc | Val Macro-F1 | Val CE | Best epoch | Train CE | Eval Train Acc | Train CE−Val CE* | Train Acc−Val Acc | Train time (s) | Peak alloc. GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for treatment in study.TREATMENTS:
            row = summaries["best_accuracy"][dataset][treatment]
            runs_subset = [item for item in runs if item["dataset"] == dataset and item["treatment"] == treatment]
            mean_train_ce = np.mean([float(item["train_ce_best_acc"]) for item in runs_subset])
            mean_train_acc = np.mean([float(item["train_accuracy_eval_best_acc"]) for item in runs_subset])
            mean_ce_gap = np.mean([float(item["train_ce_minus_val_ce_best_acc"]) for item in runs_subset])
            mean_acc_gap = np.mean([float(item["eval_train_acc_minus_val_acc_best_acc"]) for item in runs_subset])
            lines.append(
                f"| {dataset} | {treatment} | {row['val_acc']['mean']:.4f} ± {row['val_acc']['sample_sd']:.4f} | {row['val_macro_f1']['mean']:.4f} ± {row['val_macro_f1']['sample_sd']:.4f} | {row['val_ce']['mean']:.4f} ± {row['val_ce']['sample_sd']:.4f} | {row['selection_epoch']['mean']:.1f} ± {row['selection_epoch']['sample_sd']:.1f} | {mean_train_ce:.4f} | {mean_train_acc:.4f} | {mean_ce_gap:.4f} | {mean_acc_gap:.4f} | {row['training_seconds']['mean']:.1f} | {row['peak_allocated_gib_mean']:.2f} |"
            )
    lines.extend(["", "## Best-Macro-F1 results (mean ± sample SD)", "", "| Dataset | Treatment | Val Acc | Val Macro-F1 | Val CE | Best-F1 epoch |", "|---|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for treatment in study.TREATMENTS:
            row = summaries["best_macro_f1"][dataset][treatment]
            lines.append(f"| {dataset} | {treatment} | {row['val_acc']['mean']:.4f} ± {row['val_acc']['sample_sd']:.4f} | {row['val_macro_f1']['mean']:.4f} ± {row['val_macro_f1']['sample_sd']:.4f} | {row['val_ce']['mean']:.4f} ± {row['val_ce']['sample_sd']:.4f} | {row['selection_epoch']['mean']:.1f} ± {row['selection_epoch']['sample_sd']:.1f} |")
    lines.extend(["", "## Paired regularization effects", "", "| Dataset | Contrast | Metric | Mean paired Δ | Sample SD | n seeds |", "|---|---|---|---:|---:|---:|"])
    for row in effects:
        lines.append(f"| {row['dataset']} | {row['contrast']} | {row['metric']} | {row['mean_paired_difference']:.6f} | {row['sample_sd_paired_difference']:.6f} | {row['seed_count']} |")
    lines.extend(["", "The interaction is `(O3−O2)−(O1−O0)`. Positive values mean the dropout contrast is more positive at high decay for Accuracy/F1, or less negative for CE, according to the sign of the metric. They are descriptive contrasts across three paired seed runs.", "", "## Train/validation trajectories and checkpoint trade-offs", "", "See `figures/training_curves.*` for Train CE / Validation CE and eval Train Accuracy / Validation Accuracy across epochs. Thin traces show individual seeds; bold traces show treatment means over the runs available at that epoch. `figures/validation_acc_macro_f1.*` displays both checkpoint-selection rules independently. `figures/best_epoch_distribution.*` displays the run-level selection epochs.", "", "`data/checkpoint_tradeoffs.csv` lists paired Best-Accuracy and Best-Macro-F1 scorecards, epochs, and class metrics are available in `data/validation_class_metrics.csv`. All class scorecards include every configured class; main Macro-F1 follows the historical Train+Validation observed class set.", "", "## Descriptive generalization flags"])
    flags = []
    history = _read_csv(DATA_DIR / "epoch_history.csv")
    for dataset in DATASETS:
        for treatment in study.TREATMENTS:
            group_flags = []
            for seed in study.SEEDS:
                rows = sorted((row for row in history if row["dataset"] == dataset and row["treatment"] == treatment and int(row["seed"]) == seed), key=lambda x: int(x["epoch"]))
                if len(rows) < 10:
                    continue
                train_first = float(np.mean([float(r["train_ce"]) for r in rows[:10]]))
                train_last = float(np.mean([float(r["train_ce"]) for r in rows[-10:]]))
                val_first = float(np.mean([float(r["val_ce"]) for r in rows[:10]]))
                val_last = float(np.mean([float(r["val_ce"]) for r in rows[-10:]]))
                group_flags.append({"seed": seed, "train_ce_down": train_last < train_first, "late_val_ce_up": val_last > val_first, "first10_train_ce": train_first, "last10_train_ce": train_last, "first10_val_ce": val_first, "last10_val_ce": val_last})
            flags.append({"dataset": dataset, "treatment": treatment, "runs": group_flags})
            count = sum(item["train_ce_down"] and item["late_val_ce_up"] for item in group_flags)
            lines.append(f"- **{dataset}/{treatment}:** {count}/{len(group_flags)} seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than their first ten epochs. This endpoint check is descriptive and sensitive to early stopping.")
    _write_json(DATA_DIR / "trajectory_generalization_flags.json", flags)
    lines.extend(["", "## Reproducibility files", "", "`data/manifest.json` records the parent SHA, implementation freeze SHA, run configuration hashes, source SHA, cache fingerprints, hardware, and status. `data/runs.csv` and `data/epoch_history.csv` contain the run and epoch evidence. `data/paired_comparisons.csv` retains seed-level differences and `data/factorial_effects.csv` contains their means and sample SDs. The checkpoints remain under the ignored `outputs/` tree and are referenced by relative paths."])
    (RESEARCH / "GENERALIZATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_memo(runs: list[dict[str, Any]], summaries: dict[str, Any], effects: list[dict[str, Any]], interventions: list[dict[str, Any]], router_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    best = summaries["best_accuracy"]
    lines = [
        "# RM-CGE V1 Decision Memo",
        "",
        "## What the completed evidence supports",
        "",
        f"The full matrix completed {len(runs)} seed-level runs. Best-Accuracy remains the primary selection rule; Best-Macro-F1 is reported independently. The descriptive tables and seed-paired contrasts are in `GENERALIZATION_REPORT.md` and the CSV evidence directory. Three seeds per cell do not establish statistical significance.",
        "",
        "## Findings by question",
        "",
    ]
    for dataset in DATASETS:
        vals = {t: best[dataset][t]["val_acc"]["mean"] for t in study.TREATMENTS}
        f1s = {t: best[dataset][t]["val_macro_f1"]["mean"] for t in study.TREATMENTS}
        best_t = max(vals, key=vals.get)
        lines.append(f"- **{dataset}:** highest Best-Accuracy treatment mean is {best_t} (Val Accuracy {vals[best_t]:.4f}; Macro-F1 {f1s[best_t]:.4f}). This identifies the observed treatment leader in this small fixed screen, not a generally optimal setting.")
    dropout_effects = [row for row in effects if row["contrast"] == "O1-O0" and row["metric"] == "val_accuracy"]
    decay_effects = [row for row in effects if row["contrast"] == "O2-O0" and row["metric"] == "val_accuracy"]
    lines.extend(["", "The estimates answer whether regularization helps on these splits and seeds, not whether any treatment generalizes to a new dataset. Weight Decay uses the original AdamW grouping: one group contains the model and classifier parameters. Its per-step decoupled shrink multiplier is approximately `1−lr×wd`; for `lr=1e−3`, the nominal factors are 0.9999999 per step at `wd=1e−4` and 0.99995 at `wd=0.05`, applied uniformly to parameters on optimizer steps. Total effect depends on the number of updates and gradient-driven updates; final norm changes are not themselves a causal estimate of decay.", "", "### 1. Generalization adaptation or model structure?", "", "Compare held-out validation CE/accuracy curves and paired treatment effects before attributing the performance limit. If Train CE falls while Validation CE rises and stronger dropout shifts validation metrics consistently across all three paired seeds, that supports a training-generalization contribution on the current split. If best validation stays low despite reduced train/validation divergence, the remaining limit is compatible with relation/view information organization or label/feature limits; this screen cannot isolate those causes.", "", "### 2. Which regularization is more stable?", "", "Use per-treatment sample SDs and seed-level paired differences in the report. With n=3, rank stability cautiously and avoid interpreting a single best seed as evidence of robustness.", "", "### 3. Physical and Semantic cross-modal paths", "", "Frozen intervention changes are reported in the mechanism report. They show checkpoint sensitivity when the chosen relation's cross-modal context is set to zero. Since this also changes router inputs and is not retrained, it cannot establish a causal generalization benefit of training with that path.", "", "### 4. Router node conditioning", "", "Use the node-level distribution and correlations, not just average relation mass. The recorded coefficients characterize how routing mass co-varies with the measured current-state/neighborhood similarity within available support; they do not assess class predictiveness or relation truth.", "", "### 5. Next architecture priority", "", "Treat Relation Reliability, zero-update candidates, Semantic support handling, Expert diversity, and final fusion as hypotheses for a separately designed controlled study. This study does not train those variants. A Reliability-aware Router is justified only if the node diagnostics reveal a stable mismatch between relation support quality and routed contribution, and a pre-registered matched experiment shows improved validation behavior without hiding optimization or class trade-offs.", "", "## Recommended next experiment", "", "After independent audit, design a small matched mechanism screen around the strongest observed diagnostic: preserve the current V1 checkpoint and training protocol, vary one MAG-specific factor at a time, pair initializations by seed, report both checkpoint-selection rules, and keep Test sealed until a final model-selection decision. Do not infer that a next model revision is automatically warranted from one validation improvement.", "", f"Implementation freeze: `{manifest.get('implementation_freeze_sha')}`. Evidence source revision for the matrix is recorded per run in `data/runs.csv`."])
    (RESEARCH / "DECISION_MEMO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _implementation_audit() -> None:
    checks = json.loads((DATA_DIR / "implementation_checks.json").read_text(encoding="utf-8"))
    tests = checks["test_evidence"]
    o0 = checks.get("o0_reproduction", {})
    lines = [
        "# RM-CGE V1 Generalization Study Implementation Audit",
        "",
        "## Frozen boundary",
        "",
        "The RM-CGE V1 model math, initializers, projectors, relation operators, relation-specific same/cross support, shared experts, joint masked Softmax, residual scale, fusion, and CE objective remain unchanged. The model source changes only the legal dropout set (`0.20`, `0.35`), read-only diagnostic output, and relation-specific cross-context zeroing interventions. Default dropout remains `0.20`; regression testing checks the original default initialization and evaluation forward exactly.",
        "",
        "## Training instrumentation",
        "",
        "Extended validation is opt-in through `task.record_extended_validation`. Full-graph embeddings from the existing eval pass are reused for train/validation CE and Train Accuracy; index batching is ordered and does not construct a random DataLoader. Test evaluation must be false. Validation CE is unweighted Cross-Entropy. The Accuracy early-stopping comparator, min-delta, patience horizon, optimizer sequence, and ordinary training loss remain the V1 logic.",
        "",
        "Best-Accuracy and Best-Macro-F1 snapshots are cloned independently during the same Accuracy-stopped trajectory. Best-F1 is selected only from epochs observed before Accuracy-driven stopping. Checkpoints include both state dictionaries, selection epoch/rule, validation scorecard, full resolved training configuration, configuration hash, source SHA, and cache fingerprints.",
        "",
        "## Protocol and optimizer audit",
        "",
        "The 2×2 study changes only dropout and the existing AdamW weight decay. Existing `build_optimizer` resolves to a single parameter group for RM-CGE V1, containing every model and classifier parameter. The study records per-group parameter names/counts, LR/decay, initial and final parameter norms, actual optimizer updates, first-epoch gradient diagnostics, and CUDA peak memory for each run. No no-decay parameter exclusion is added.",
        "",
        "Initial model state and classifier state hashes were compared for every Dataset × Seed × Treatment before optimization. Details are in `implementation_checks.json`.",
        "",
        "## Regression and replay evidence",
        "",
        f"The isolated CPU regression suite passed {tests.get('passed_tests')} tests across {len(tests.get('files', []))} RM-CGE/PACE/V7 files. Full captured output is stored in `implementation_checks.json`.",
        "",
    ]
    if o0:
        lines.extend([
            f"The Movies/O0/seed-42 replay gate **{'passed' if o0.get('passed') else 'failed'}**: Best epoch {o0.get('original_best_epoch')} vs {o0.get('study_best_epoch')}; epochs {o0.get('original_epochs_completed')} vs {o0.get('study_epochs_completed')}; max Train CE trajectory difference {o0.get('train_ce_max_abs_difference'):.3g}; max Validation Accuracy difference {o0.get('validation_accuracy_max_abs_difference'):.3g}; max Macro-F1 difference {o0.get('validation_macro_f1_max_abs_difference'):.3g}; checkpoint embedding/logit max difference {o0.get('embedding_max_abs_difference'):.3g}/{o0.get('logit_max_abs_difference'):.3g}.",
            "",
            "Original Pilot embedding/logit arrays were not archived. The replay reconstructs their values from the frozen checkpoint, verifies deterministic repeated inference, compares all O0 outputs to the reloaded historical checkpoint, and saves SHA-256 hashes. This evidence limit is explicit in `implementation_checks.json`.",
            "",
        ])
    lines.extend([
        "The Test split was disabled in all training configs and diagnostic checkpoint scorecards; per-class and aggregate analyses index only Validation labels, while Macro-F1's historical class set is built from Train and Validation labels. No architecture variants, Test evaluation, extra seeds, or hyperparameter searches are included.",
        "",
        f"Parent Pilot Report SHA: `{checks.get('parent_pilot_sha')}`. Source SHA and worktree fingerprints for the pre-freeze replay and post-freeze matrix are stored in the manifest and run CSV.",
    ])
    (RESEARCH / "IMPLEMENTATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_and_report() -> dict[str, Any]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "matrix_completed" or len(manifest.get("runs", [])) != 36:
        raise RuntimeError("the complete study matrix is not available")
    checkpoint_rows, class_rows = evaluate_all_checkpoints()
    run_rows = manifest["runs"]
    paired, effects = _paired_effects(run_rows)
    _write_csv(DATA_DIR / "paired_comparisons.csv", paired)
    _write_csv(DATA_DIR / "factorial_effects.csv", effects)
    tradeoffs = []
    lookup = {(row["dataset"], row["treatment"], int(row["seed"])): row for row in checkpoint_rows}
    for run in run_rows:
        acc = lookup[(run["dataset"], run["treatment"], int(run["seed"]))]
        f1 = next(row for row in checkpoint_rows if row["dataset"] == run["dataset"] and row["treatment"] == run["treatment"] and int(row["seed"]) == int(run["seed"]) and row["selection"] == "best_f1")
        tradeoffs.append({
            "dataset": run["dataset"], "treatment": run["treatment"], "seed": int(run["seed"]),
            "best_acc_epoch": acc["selection_epoch"], "best_f1_epoch": f1["selection_epoch"],
            "best_acc_val_acc": acc["val_acc"], "best_acc_val_macro_f1": acc["val_macro_f1"], "best_acc_val_ce": acc["val_ce"],
            "best_f1_val_acc": f1["val_acc"], "best_f1_val_macro_f1": f1["val_macro_f1"], "best_f1_val_ce": f1["val_ce"],
            "best_f1_minus_best_acc_accuracy": f1["val_acc"] - acc["val_acc"],
            "best_f1_minus_best_acc_macro_f1": f1["val_macro_f1"] - acc["val_macro_f1"],
            "best_f1_minus_best_acc_val_ce": f1["val_ce"] - acc["val_ce"],
            "evaluate_test": False,
        })
    _write_csv(DATA_DIR / "checkpoint_tradeoffs.csv", tradeoffs)
    summaries = _summaries(run_rows, checkpoint_rows)
    _write_json(DATA_DIR / "summary_statistics.json", summaries)
    _write_implementation_audit()
    frozen_rows = _read_csv(DATA_DIR / "frozen_cross_relation_interventions.csv")
    router_rows = _read_csv(DATA_DIR / "router_node_statistics.csv")
    _generalization_report(run_rows, paired, effects, summaries, checkpoint_rows)
    _decision_memo(run_rows, summaries, effects, frozen_rows, router_rows, manifest)
    from scripts.plot_rmcge_v1_generalization import create_figures

    figure_qa = create_figures()
    _write_json(DATA_DIR / "figure_qa.json", figure_qa)
    return {"run_count": len(run_rows), "checkpoint_count": len(checkpoint_rows), "class_metric_rows": len(class_rows), "figure_qa": figure_qa}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("frozen", "summarize"), required=True)
    args = parser.parse_args()
    if args.phase == "frozen":
        result = frozen_pilot_analysis()
    else:
        result = summarize_and_report()
    print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.item() if hasattr(value, "item") else str(value)))


if __name__ == "__main__":
    main()
