from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m0_adaptive_propagation import (  # frozen data/protocol helpers
    _metric_from_embeddings,
    load_m0_data,
    restrict_labels_to_train_val,
)
from src.models.adaptive_prop_m0 import (
    Model,
    common_parameter_names,
    global_mean_control,
    neighborhood_shuffle_indices,
    parameter_counts,
    target_mean_control,
    tie_modality_controls,
)
from src.tasks.common import build_optimizer, scheduler_step
from src.utils.seeds import set_seed

DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
NEW_VARIANTS = ("extent_wide", "single_basis_static", "single_basis_target")
OUT_DIR = PROJECT_ROOT / "outputs" / "m01_adaptive_correction_attribution"
M0_OUT_DIR = PROJECT_ROOT / "outputs" / "m0_adaptive_propagation_screen"
M0_PERFORMANCE = PROJECT_ROOT / "research" / "m0_adaptive_propagation_screen" / "data" / "performance_by_run.csv"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def make_config(dataset: str, seed: int, variant: str, device: str, epochs: int | None = None):
    overrides = [
        f"dataset={dataset}", "task=nc", "model=adaptive_prop_m01",
        f"model.variant={variant}", f"seed={seed}", "num_runs=1",
        f"device={device}", "task.evaluate_test=false",
    ]
    if epochs is not None:
        overrides.append(f"task.epochs={int(epochs)}")
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _model_data_info(data) -> dict[str, int]:
    return {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def _state_hash(state, names: tuple[str, ...]) -> str:
    import hashlib
    dig = hashlib.sha256()
    for name in names:
        dig.update(name.encode("utf-8"))
        dig.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return dig.hexdigest()


def audit_initialization(cfg, data_info: dict[str, int], seed: int) -> dict[str, Any]:
    variants = ("extent", "extent_wide", "single_basis", "single_basis_static", "single_basis_target")
    models = {}
    hashes = {}
    counts = {}
    states = {}
    for variant in variants:
        set_seed(seed)
        local_cfg = copy.deepcopy(cfg)
        local_cfg.model.variant = variant
        model = Model(local_cfg, data_info)
        names = common_parameter_names(model)
        state = model.state_dict()
        models[variant] = model
        hashes[variant] = _state_hash(state, names)
        states[variant] = {name: state[name].detach().cpu().clone() for name in names}
        counts[variant] = parameter_counts(model)
    common_pairs = {}
    for i, left in enumerate(variants):
        for right in variants[i + 1:]:
            names_equal = common_parameter_names(models[left]) == common_parameter_names(models[right])
            common_pairs[f"{left}__{right}"] = names_equal and all(
                torch.equal(states[left][name], states[right][name]) for name in states[left]
            )
    b, static, target = (models[k] for k in ("single_basis", "single_basis_static", "single_basis_target"))
    head_equal = all(torch.equal(a.weight, b_.weight) and torch.equal(a.bias, b_.bias)
                     for a, b_ in zip(b.correction_heads, target.correction_heads))
    basis_equal = all(torch.equal(getattr(b, key), getattr(other, key))
                      for other in (static, target) for key in ("basis_u", "basis_v"))
    total_match = abs(counts["extent_wide"]["model_total"] - counts["single_basis"]["model_total"]) <= 32
    target_match = counts["single_basis_target"]["model_total"] == counts["single_basis"]["model_total"]
    static_delta = counts["single_basis"]["model_total"] - counts["single_basis_static"]["model_total"]
    audit = {
        "dataset": str(cfg.dataset.name), "seed": int(seed), "common_hashes": hashes,
        "common_pairwise_equal": common_pairs, "common_all_equal": all(common_pairs.values()),
        "b_target_correction_heads_equal": head_equal,
        "b_static_target_basis_equal_to_b": basis_equal,
        "a_wide_b_model_param_delta": counts["extent_wide"]["model_total"] - counts["single_basis"]["model_total"],
        "a_wide_b_within_32": total_match, "b_target_b_param_delta": counts["single_basis_target"]["model_total"] - counts["single_basis"]["model_total"],
        "b_target_b_params_equal": target_match, "b_b_static_param_delta": static_delta,
        "b_static_difference_is_128": static_delta == 128,
        "parameter_counts": counts,
    }
    for model in models.values():
        del model
    return audit


def _quantiles(values) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {k: float("nan") for k in ("mean", "std", "q10", "q25", "median", "q75", "q90")}
    q10, q25, med, q75, q90 = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"mean": float(arr.mean()), "std": float(arr.std()), "q10": float(q10),
            "q25": float(q25), "median": float(med), "q75": float(q75), "q90": float(q90)}


def _evaluate(model, classifier, x, edges, val_idx, val_y, num_classes, overrides=None):
    model.eval()
    classifier.eval()
    with torch.no_grad():
        z, _, _, _, _ = model(x, edges, control_overrides=overrides)
        return _metric_from_embeddings(classifier, z, val_idx, val_y, num_classes)


def _control_diagnostics(aux: dict[str, Any], val_idx: torch.Tensor, variant: str) -> dict[str, Any]:
    controls = aux["controls"]
    output: dict[str, Any] = {"g": [], "c": [], "correction_ratio": [], "mean_abs_g_tv": float("nan"),
                              "mean_abs_c_tv": float("nan"), "within_node_g_std": [], "within_node_c_std": []}
    for key in ("g", "c", "correction_ratio"):
        for m in range(2):
            v = controls[key][m].reshape(-1)
            output[key].append(_quantiles(v))
    for key in ("g", "c"):
        left, right = controls[key]
        output[f"mean_abs_{key}_tv"] = float((left - right).abs().mean()) if left.numel() else float("nan")
    dst = aux["edge_index_nonself"][1].long()
    degree = aux["degree"].cpu().long()
    val_set = {int(i) for i in val_idx.detach().cpu().tolist()}
    eligible = {i for i in val_set if int(degree[i]) >= 5}
    for key in ("g", "c"):
        for m in range(2):
            if not controls[key][m].numel():
                output[f"within_node_{key}_std"].append({"eligible_targets": 0, **_quantiles([]), "median_range_q90_q10": float("nan"), "range_q25": float("nan"), "range_q75": float("nan")})
                continue
            grouped: dict[int, list[float]] = defaultdict(list)
            for edge_id, target in enumerate(dst.tolist()):
                if target in eligible:
                    grouped[target].append(float(controls[key][m][edge_id, 0]))
            stds = [float(np.std(v, ddof=0)) for v in grouped.values() if len(v) >= 2]
            ranges = [float(np.quantile(v, .9) - np.quantile(v, .1)) for v in grouped.values() if len(v) >= 2]
            summary = _quantiles(stds)
            summary["eligible_targets"] = len(stds)
            summary["median_range_q90_q10"] = float(np.median(ranges)) if ranges else float("nan")
            summary["range_q25"] = float(np.quantile(ranges, .25)) if ranges else float("nan")
            summary["range_q75"] = float(np.quantile(ranges, .75)) if ranges else float("nan")
            output[f"within_node_{key}_std"].append(summary)
    if variant == "single_basis_static":
        output["learned_c_static"] = [float(controls["c"][m][0, 0]) for m in range(2)]
    if variant == "single_basis_target":
        output["b_target_within_target_max_std"] = max(
            (v["std"] for v in output["within_node_c_std"] if np.isfinite(v["std"])), default=0.0
        )
    return output


def _make_controls_dict(controls):
    return {name: pair for name, pair in controls.items() if pair[0].numel()}


def _interventions(model, classifier, x, edges, val_idx, val_y, num_classes, aux, variant: str):
    controls = aux["controls"]
    active = _make_controls_dict(controls)
    dst = aux["edge_index_nonself"][1].long()
    base = _evaluate(model, classifier, x, edges, val_idx, val_y, num_classes)
    rows = []

    def add(name, overrides, repeat=None):
        metrics = _evaluate(model, classifier, x, edges, val_idx, val_y, num_classes, overrides)
        rows.append({"intervention": name, "repeat_seed": repeat, **metrics,
                     "delta_accuracy": metrics["val_accuracy"] - base["val_accuracy"],
                     "delta_macro_f1": metrics["val_macro_f1"] - base["val_macro_f1"],
                     "delta_ce": metrics["val_ce"] - base["val_ce"]})

    if "g" in active:
        add("extent_off", {**active, "g": tuple(torch.ones_like(v) for v in active["g"])})
    if "c" in active:
        add("function_off", {**active, "c": tuple(torch.zeros_like(v) for v in active["c"])})
    if variant in {"extent_wide", "single_basis_static", "single_basis_target"} and "g" in active:
        for seed in range(1001, 1006):
            changed = []
            for m in range(2):
                perm = neighborhood_shuffle_indices(dst, seed + m * 10000)
                changed.append(active["g"][m][perm])
            add("g_shuffle_only", {**active, "g": tuple(changed)}, seed)
    if variant in {"single_basis_static", "single_basis_target"}:
        for seed in range(1001, 1006):
            changed = []
            for m in range(2):
                perm = neighborhood_shuffle_indices(dst, seed + m * 10000)
                changed.append(active["c"][m][perm])
            add("c_shuffle_only", {**active, "c": tuple(changed)}, seed)
    if "c" in active:
        add("c_target_mean", {**active, "c": tuple(target_mean_control(active["c"][m], dst, x.size(0)) for m in range(2))})
        add("c_global_mean", {**active, "c": tuple(global_mean_control(active["c"][m]) for m in range(2))})
    if active:
        add("modality_tied", tie_modality_controls(active))
    return {"normal": base, "rows": rows}


def _grad_audit(model: Model) -> dict[str, dict[str, float]]:
    groups = {
        "gate": ("gate_heads.",), "correction_head": ("correction_heads.",),
        "static_c": ("static_c_logits",), "basis": ("basis_u", "basis_v"),
        "relation_encoder": ("rel_proj_t.", "rel_proj_v.", "phi_pair.", "phi_rel.", "phi_mod.", "modality_embeddings"),
    }
    output = {}
    for group, prefixes in groups.items():
        grads = [p.grad.detach() for name, p in model.named_parameters() if name.startswith(prefixes) and p.grad is not None]
        output[group] = {"finite": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
                         "norm": float(torch.sqrt(sum(g.float().pow(2).sum() for g in grads)).item()) if grads else 0.0,
                         "parameter_tensors_with_grad": len(grads)}
    return output


def _assert_audit(audit):
    for key in ("common_all_equal", "b_target_correction_heads_equal", "b_static_target_basis_equal_to_b",
                "a_wide_b_within_32", "b_target_b_params_equal", "b_static_difference_is_128"):
        if not audit[key]:
            raise AssertionError(f"M0.1 initialization/parameter audit failed at {key}: {audit}")


def run_one(dataset: str, seed: int, variant: str, device_name: str, out_dir: Path = OUT_DIR,
            max_epochs: int | None = None, save_checkpoint: bool = True) -> dict[str, Any]:
    cfg = make_config(dataset, seed, variant, device_name, max_epochs)
    data = load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    data_info = _model_data_info(data)
    audit = audit_initialization(cfg, data_info, seed)
    _assert_audit(audit)
    device = torch.device(device_name)
    set_seed(seed)
    model = Model(cfg, data_info).to(device)
    torch.manual_seed(seed + 1907)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1907)
    classifier = nn.Linear(model.out_dim, data.num_classes).to(device)
    set_seed(seed)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    x, edges = data.x.to(device), data.edge_index.to(device)
    train_idx, val_idx = splits["train"].to(device), splits["validation"].to(device)
    labels = data.y.to(device)
    train_y, val_y = labels[train_idx], labels[val_idx]
    if bool((train_y < 0).any()) or bool((val_y < 0).any()) or data.test_idx is not None:
        raise AssertionError("M0.1 exposed missing train/val labels or test indices")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_acc, best_epoch, patience_left = -1.0, 0, int(cfg.task.patience)
    best_model = best_head = None
    best_metrics = None
    first_grads = {}
    start = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, int(cfg.task.epochs) + 1):
        epochs_run = epoch
        model.train(); classifier.train(); optimizer.zero_grad(set_to_none=True)
        z, _, _, aux_loss, _ = model(x, edges)
        logits = classifier(z[train_idx])
        loss = nn.functional.cross_entropy(logits, train_y) + float(cfg.task.loss.aux_weight) * aux_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at {dataset}/{seed}/{variant}/{epoch}")
        loss.backward()
        if epoch == 1:
            first_grads = _grad_audit(model)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()),
                                      max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True)
        optimizer.step()
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        del z, logits, loss
        current = _evaluate(model, classifier, x, edges, val_idx, val_y, data.num_classes)
        if current["val_accuracy"] > best_acc + float(cfg.task.early_stop_min_delta):
            best_acc, best_epoch, best_metrics = current["val_accuracy"], epoch, current
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            patience_left = int(cfg.task.patience)
        elif epoch >= int(cfg.task.early_stop_min_epoch):
            patience_left -= 1
            if patience_left <= 0:
                break
    train_time = time.perf_counter() - start
    if best_model is None or best_head is None:
        raise RuntimeError("training ended without a validation-selected checkpoint")
    model.load_state_dict(best_model); classifier.load_state_dict(best_head)
    model.eval()
    with torch.no_grad():
        z, _, _, _, aux = model(x, edges, return_diagnostics=True)
        metrics = _metric_from_embeddings(classifier, z, val_idx, val_y, data.num_classes)
    diagnostics = _control_diagnostics(aux, val_idx, variant)
    intervention_start = time.perf_counter()
    interventions = _interventions(model, classifier, x, edges, val_idx, val_y, data.num_classes, aux, variant)
    intervention_time = time.perf_counter() - intervention_start
    counts = parameter_counts(model)
    classifier_params = sum(p.numel() for p in classifier.parameters())
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    record = {
        "status": "completed", "dataset": dataset, "seed": int(seed), "variant": variant,
        "best_epoch": best_epoch, "epochs_run": epochs_run, **metrics,
        "parameter_counts": counts, "model_params": counts["model_total"],
        "classifier_params": classifier_params, "trainable_params": counts["model_total"] + classifier_params,
        "peak_gpu_memory_bytes": peak_memory, "training_time_sec": train_time,
        "intervention_time_sec": intervention_time, "gradient_audit_epoch1": first_grads,
        "control_diagnostics": diagnostics, "interventions": interventions,
        "initialization_audit": audit, "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False, "test_split_field_attached_to_data": False,
        "test_labels_exposed_to_runner": False, "data_info": data_info,
    }
    run_json = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    write_json(run_json, record)
    if save_checkpoint:
        checkpoint = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_model, "head_state": best_head, "data_info": data_info,
                    "dataset": dataset, "seed": int(seed), "variant": variant,
                    "best_epoch": best_epoch, "best_val_metrics": metrics,
                    "protocol": "unified_full_graph_nc_v1", "evaluate_test": False}, checkpoint)
        record["checkpoint"] = str(checkpoint.relative_to(PROJECT_ROOT))
        write_json(run_json, record)
    print(f"[done] {dataset}/{seed}/{variant} epoch={best_epoch} acc={metrics['val_accuracy']:.4f} "
          f"f1={metrics['val_macro_f1']:.4f} ce={metrics['val_ce']:.4f} train={train_time:.1f}s "
          f"peak={peak_memory/1024**3:.2f}GiB", flush=True)
    del z, aux, data, x, edges, model, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _load_b_checkpoint(dataset: str, seed: int, device_name: str, out_dir: Path = OUT_DIR):
    cfg = make_config(dataset, seed, "single_basis", device_name)
    data = load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    info = _model_data_info(data)
    ckpt_path = M0_OUT_DIR / "checkpoints" / dataset / f"seed_{seed}_single_basis.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"required committed M0-B checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("variant") != "single_basis" or int(ckpt.get("seed", -1)) != seed or ckpt.get("dataset") != dataset:
        raise AssertionError(f"checkpoint metadata mismatch: {ckpt_path}")
    model = Model(cfg, info).to(device_name)
    model.load_state_dict(ckpt["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, data.num_classes).to(device_name)
    classifier.load_state_dict(ckpt["head_state"], strict=True)
    data_info = ckpt.get("data_info", {})
    if data_info != info:
        raise AssertionError(f"checkpoint data_info mismatch {data_info} != {info}")
    device = torch.device(device_name)
    x, edges = data.x.to(device), data.edge_index.to(device)
    val_idx = splits["validation"].to(device)
    labels = data.y.to(device)
    val_y = labels[val_idx]
    if data.test_idx is not None:
        raise AssertionError("test split attached while loading old B checkpoint")
    model.eval(); classifier.eval()
    with torch.no_grad():
        z, _, _, _, aux = model(x, edges, return_diagnostics=True)
        normal = _metric_from_embeddings(classifier, z, val_idx, val_y, data.num_classes)
    # Confirm the same checkpoint/classifier/val protocol as the committed M0 record.
    m0_record = M0_OUT_DIR / "runs" / dataset / f"seed_{seed}" / "single_basis.json"
    if not m0_record.is_file():
        raise FileNotFoundError(f"M0 run record missing: {m0_record}")
    old = json.loads(m0_record.read_text(encoding="utf-8"))
    for key in ("val_accuracy", "val_macro_f1", "val_ce"):
        if abs(float(normal[key]) - float(old[key])) > 1e-6:
            raise AssertionError(f"loaded M0 B does not reproduce {key}: {normal[key]} vs {old[key]}")
    return cfg, data, model, classifier, x, edges, val_idx, val_y, aux, normal, ckpt_path, splits


def run_b_attributions(dataset: str, seed: int, device_name: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    cfg, data, model, classifier, x, edges, val_idx, val_y, aux, normal, ckpt_path, splits = _load_b_checkpoint(dataset, seed, device_name, out_dir)
    controls = _make_controls_dict(aux["controls"])
    dst = aux["edge_index_nonself"][1].long()
    nclasses = int(data.num_classes)
    rows = []

    def add(name, overrides, repeat=None):
        metrics = _evaluate(model, classifier, x, edges, val_idx, val_y, nclasses, overrides)
        rows.append({"intervention": name, "repeat_seed": repeat, **metrics,
                     "delta_accuracy": metrics["val_accuracy"] - normal["val_accuracy"],
                     "delta_macro_f1": metrics["val_macro_f1"] - normal["val_macro_f1"],
                     "delta_ce": metrics["val_ce"] - normal["val_ce"]})

    add("function_off", {**controls, "c": tuple(torch.zeros_like(v) for v in controls["c"])})
    add("c_target_mean", {**controls, "c": tuple(target_mean_control(v, dst, x.size(0)) for v in controls["c"])})
    add("c_global_mean", {**controls, "c": tuple(global_mean_control(v) for v in controls["c"])})
    for seed_rep in range(1001, 1006):
        perm_pair = tuple(neighborhood_shuffle_indices(dst, seed_rep + m * 10000) for m in range(2))
        g_new = tuple(controls["g"][m][perm_pair[m]] for m in range(2))
        c_new = tuple(controls["c"][m][perm_pair[m]] for m in range(2))
        add("g_shuffle_only", {**controls, "g": g_new}, seed_rep)
        add("c_shuffle_only", {**controls, "c": c_new}, seed_rep)
        add("edge_control_shuffle", {**controls, "g": g_new, "c": c_new}, seed_rep)
    # Record the magnitude of within-target c variation for interpreting the mean intervention.
    c_target = tuple(target_mean_control(v, dst, x.size(0)) for v in controls["c"])
    target_identity_max = max(float((c_target[m] - controls["c"][m]).abs().max()) for m in range(2))
    diagnostics = _control_diagnostics(aux, val_idx, "single_basis")
    degree = aux["degree"].cpu().long()
    eligible = [int(i) for i in splits["validation"].tolist() if int(degree[int(i)]) >= 5]
    node_c_stats = []
    variance_parts = []
    for modality in range(2):
        c = aux["controls"]["c"][modality].reshape(-1).numpy().astype(np.float64)
        by_target: dict[int, list[float]] = defaultdict(list)
        for edge_i, target in enumerate(dst.detach().cpu().tolist()):
            by_target[int(target)].append(float(c[edge_i]))
        target_stds = [float(np.std(by_target[target], ddof=0)) for target in eligible if target in by_target]
        target_ranges = [float(np.quantile(by_target[target], .9) - np.quantile(by_target[target], .1)) for target in eligible if target in by_target]
        node_c_stats.append({"modality": "text" if modality == 0 else "visual", "eligible_target_count": len(target_stds),
                             **_quantiles(target_stds), "within_target_range_q90_q10_median": float(np.median(target_ranges)) if target_ranges else float("nan"),
                             "within_target_range_q25": float(np.quantile(target_ranges, .25)) if target_ranges else float("nan"),
                             "within_target_range_q75": float(np.quantile(target_ranges, .75)) if target_ranges else float("nan")})
        all_targets = list(by_target)
        grand = float(c.mean()) if c.size else float("nan")
        ss_total = float(((c - grand) ** 2).sum()) if c.size else 0.0
        ss_between = sum(len(vals) * (float(np.mean(vals)) - grand) ** 2 for vals in by_target.values())
        ss_within = sum(sum((v - float(np.mean(vals))) ** 2 for v in vals) for vals in by_target.values())
        variance_parts.append({"modality": "text" if modality == 0 else "visual", "edge_count": len(c),
                               "target_count": len(all_targets), "total_variance_population": ss_total / max(len(c), 1),
                               "between_fraction": ss_between / ss_total if ss_total else 0.0,
                               "within_fraction": ss_within / ss_total if ss_total else 0.0,
                               "ss_total": ss_total, "ss_between": ss_between, "ss_within": ss_within})
    record = {"status": "completed", "dataset": dataset, "seed": seed, "source_checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
              "normal": normal, "rows": rows, "control_diagnostics": diagnostics,
              "within_node_c_variation": node_c_stats, "c_variance_decomposition": variance_parts,
              "c_target_mean_max_abs_control_change": target_identity_max,
              "protocol": "same M0-B checkpoint/classifier/H0, validation only, no retraining, no test"}
    write_json(out_dir / "b_attributions" / dataset / f"seed_{seed}.json", record)
    print(f"[B attribution] {dataset}/{seed} loaded committed checkpoint; within-target std="
          f"{[round(r['median'], 6) for r in node_c_stats]}", flush=True)
    del data, model, classifier, x, edges, aux
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def refresh_checkpoint_diagnostics(dataset: str, seed: int, variant: str, device_name: str, out_dir: Path = OUT_DIR):
    """Recompute extra edge/target control summaries from a selected checkpoint; no optimization."""
    if variant not in NEW_VARIANTS:
        raise ValueError(f"diagnostic refresh only accepts M0.1 variants, got {variant}")
    cfg = make_config(dataset, seed, variant, device_name)
    data = load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    info = _model_data_info(data)
    checkpoint = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
    record_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    if not checkpoint.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"cannot refresh missing selected checkpoint or run record: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Model(cfg, info).to(device_name)
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    device = torch.device(device_name)
    with torch.no_grad():
        _, _, _, _, aux = model(data.x.to(device), data.edge_index.to(device), return_diagnostics=True)
    controls = aux["controls"]
    cd = json.loads(record_path.read_text(encoding="utf-8"))
    diagnostics = cd["control_diagnostics"]
    diagnostics["g_fraction_lt_0_1"] = [float((g < .1).float().mean()) if g.numel() else float("nan") for g in controls["g"]]
    diagnostics["g_fraction_gt_0_9"] = [float((g > .9).float().mean()) if g.numel() else float("nan") for g in controls["g"]]
    if variant == "single_basis_target":
        dst = aux["edge_index_nonself"][1].long()
        edge_count = int(dst.numel())
        first = torch.full((data.num_nodes,), edge_count, dtype=torch.long)
        if edge_count:
            edge_ids = torch.arange(edge_count, dtype=torch.long)
            first.scatter_reduce_(0, dst, edge_ids, reduce="amin", include_self=True)
            selected = first[first < edge_count]
        else:
            selected = torch.empty(0, dtype=torch.long)
        diagnostics["target_level_c"] = [_quantiles(controls["c"][m][selected, 0]) for m in range(2)]
        diagnostics["target_level_c_target_count"] = int(selected.numel())
        diagnostics["edge_expanded_c"] = diagnostics["c"]
        max_deviation = 0.0
        for m in range(2):
            target_by_node = controls["c"][m].new_zeros((data.num_nodes, 1))
            if selected.numel():
                target_by_node[dst[selected]] = controls["c"][m][selected]
                max_deviation = max(max_deviation, float((controls["c"][m] - target_by_node[dst]).abs().max()))
        diagnostics["b_target_within_target_max_abs_deviation"] = max_deviation
        if diagnostics["b_target_within_target_max_abs_deviation"] > 1e-7:
            raise AssertionError("B-target c is not constant within a target")
    cd["control_diagnostics"] = diagnostics
    write_json(record_path, cd)
    print(f"[diagnostics refreshed, no training] {dataset}/{seed}/{variant}", flush=True)
    del data, model, aux
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_diagnostic_refresh(args):
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                refresh_checkpoint_diagnostics(dataset, seed, variant, args.device, args.output_dir)


def run_smoke(device_name: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    dataset, seed = "Movies", 42
    cfg = make_config(dataset, seed, "extent_wide", device_name, epochs=2)
    data = load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    info = _model_data_info(data)
    audit = audit_initialization(cfg, info, seed)
    _assert_audit(audit)
    device = torch.device(device_name)
    x, edges = data.x.to(device), data.edge_index.to(device)
    train_idx, val_idx = splits["train"].to(device), splits["validation"].to(device)
    labels = data.y.to(device)
    summaries = []
    for variant in NEW_VARIANTS:
        cfg.model.variant = variant
        set_seed(seed)
        model = Model(cfg, info).to(device)
        torch.manual_seed(seed + 1907)
        if device.type == "cuda": torch.cuda.manual_seed_all(seed + 1907)
        classifier = nn.Linear(model.out_dim, data.num_classes).to(device)
        model.train(); classifier.train()
        z = model(x, edges)[0]
        loss = nn.functional.cross_entropy(classifier(z[train_idx]), labels[train_idx])
        loss.backward()
        gradients = _grad_audit(model)
        required = ["gate", "relation_encoder"]
        if variant == "single_basis_static": required += ["static_c", "basis"]
        if variant == "single_basis_target": required += ["correction_head", "basis"]
        for group in required:
            if not gradients[group]["finite"] or gradients[group]["norm"] <= 0:
                raise AssertionError(f"missing finite nonzero {group} gradient in {variant}: {gradients[group]}")
        if variant == "extent_wide" and (model.basis_u is not None or model.correction_heads is not None):
            raise AssertionError("A-wide must remain extent-only")
        model.eval()
        with torch.no_grad():
            z, _, _, _, aux = model(x, edges, return_diagnostics=True)
        if not bool(torch.isfinite(z).all()):
            raise AssertionError(f"non-finite smoke embedding in {variant}")
        if variant == "single_basis_target":
            for m in range(2):
                c, dst = aux["controls"]["c"][m].reshape(-1), aux["edge_index_nonself"][1]
                for node in torch.unique(dst):
                    values = c[dst == node]
                    if values.numel() > 1 and float(values.std(unbiased=False)) > 1e-7:
                        raise AssertionError("B-target c differs within a target")
        with torch.no_grad():
            intervention = _interventions(model, classifier, x, edges, val_idx, labels[val_idx], data.num_classes, aux, variant)
        if variant == "single_basis_target":
            target_mean_row = next(r for r in intervention["rows"] if r["intervention"] == "c_target_mean")
            if any(abs(target_mean_row[k] - intervention["normal"][k]) > 1e-6
                   for k in ("val_accuracy", "val_macro_f1", "val_ce")):
                raise AssertionError("B-target c-target-mean must be a prediction-level no-op")
        summaries.append({"variant": variant, "gradients": gradients, "params": parameter_counts(model),
                          "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                          "intervention_names": [r["intervention"] for r in intervention["rows"]]})
        del model, classifier, z, aux
        if device.type == "cuda": torch.cuda.empty_cache()
    # Existing M0-B checkpoint is loaded strictly and receives all requested attribution operations.
    b_record = run_b_attributions(dataset, seed, device_name, out_dir)
    if b_record["c_target_mean_max_abs_control_change"] <= 1e-7:
        raise AssertionError("the trained original B smoke graph unexpectedly had zero c granularity")
    payload = {"status": "smoke_passed", "dataset": dataset, "seed": seed, "variants": summaries,
               "initialization_and_parameter_audit": audit,
               "loaded_b_checkpoint": b_record["source_checkpoint"],
               "b_interventions": [r["intervention"] for r in b_record["rows"]],
               "b_checkpoint_reproduced_normal": b_record["normal"],
               "test_idx_loaded": False, "test_labels_exposed": False, "test_evaluation": False,
               "device": device_name, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    write_json(out_dir / "smoke" / "Movies_seed_42" / "smoke_result.json", payload)
    print(f"[M0.1 smoke passed] Movies/42 variants={len(summaries)} device={device_name}", flush=True)
    return payload


def run_campaign(args):
    smoke_file = args.output_dir / "smoke" / "Movies_seed_42" / "smoke_result.json"
    if not smoke_file.is_file() or json.loads(smoke_file.read_text(encoding="utf-8")).get("status") != "smoke_passed":
        raise RuntimeError("M0.1 Movies/42 smoke must pass before formal campaign")
    results = []
    total = len(args.datasets) * len(args.seeds) * len(args.variants)
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                runfile = args.output_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if runfile.is_file():
                    previous = json.loads(runfile.read_text(encoding="utf-8"))
                    if previous.get("status") == "completed":
                        results.append(previous)
                        print(f"[skip-complete] {dataset}/{seed}/{variant}", flush=True)
                        continue
                print(f"[run {len(results)+1}/{total}] {dataset}/{seed}/{variant} on {args.device}", flush=True)
                result = run_one(dataset, seed, variant, args.device, args.output_dir, args.max_epochs)
                results.append(result)
                write_json(args.output_dir / "campaign_progress.json", {
                    "requested_runs": total, "completed_unique_runs": len({(r["dataset"], r["seed"], r["variant"]) for r in results}),
                    "datasets": list(args.datasets), "seeds": list(args.seeds), "variants": list(args.variants),
                    "device": args.device, "evaluate_test": False,
                })
    # Reuse all nine already-trained B checkpoints for attribution, never retraining B.
    b_runs = [run_b_attributions(d, s, args.device, args.output_dir) for d in args.datasets for s in args.seeds]
    write_json(args.output_dir / "campaign_result.json", {
        "status": "completed" if len(results) == total and len(b_runs) == len(args.datasets)*len(args.seeds) else "partial",
        "completed_new_runs": len(results), "requested_new_runs": total,
        "completed_b_checkpoint_attributions": len(b_runs), "datasets": list(args.datasets),
        "seeds": list(args.seeds), "variants": list(args.variants), "device": args.device,
        "evaluate_test": False, "link_prediction": False,
    })
    print(f"[M0.1 campaign finished] new={len(results)}/{total}; reused_B={len(b_runs)}/9", flush=True)


def main():
    parser = argparse.ArgumentParser(description="M0.1 adaptive correction attribution closure")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--refresh-diagnostics", action="store_true", help="Recompute checkpoint control summaries without training")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=NEW_VARIANTS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=NEW_VARIANTS, default=list(NEW_VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"[m0.1] device={args.device} gpu={torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu'} "
          f"torch={torch.__version__} pyg={__import__('torch_geometric').__version__}", flush=True)
    if args.smoke:
        run_smoke(args.device, args.output_dir)
    elif args.refresh_diagnostics:
        run_diagnostic_refresh(args)
    elif args.campaign:
        run_campaign(args)
    elif args.dataset and args.seed is not None and args.variant:
        run_one(args.dataset, args.seed, args.variant, args.device, args.output_dir, args.max_epochs)
    else:
        parser.error("choose --smoke, --campaign, or one --dataset/--seed/--variant run")


if __name__ == "__main__":
    main()
