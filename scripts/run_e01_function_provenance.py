from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m0_adaptive_propagation import (  # frozen data/protocol helpers
    _metric_from_embeddings,
    load_m0_data,
    restrict_labels_to_train_val,
)
from src.models.adaptive_prop_m0 import (
    Model as M0Model,
    fixed_degree_mean,
    incoming_degree,
    neighborhood_shuffle_indices,
    remove_self_messages,
)
from src.models.provenance_executor_e01 import (
    E01_VARIANTS,
    KEEP_VARIANTS,
    Model,
    provenance_interventions,
)
from src.models.structured_executor_e0 import (
    FUNCTION_NAMES,
    Model as E0Model,
    _rms,
    pi_global_mean,
    pi_target_mean,
)
from src.tasks.common import build_optimizer, scheduler_step
from src.utils.seeds import set_seed


DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
OUT_DIR = PROJECT_ROOT / "outputs" / "e01_function_provenance_preservation"
E0_RESEARCH = PROJECT_ROOT / "research" / "e0_structured_relation_function_executor"
E0_PERFORMANCE = E0_RESEARCH / "data" / "performance_by_run.csv"
M0_PERFORMANCE = PROJECT_ROOT / "research" / "m0_adaptive_propagation_screen" / "data" / "performance_by_run.csv"
_INIT_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def make_config(dataset: str, seed: int, variant: str, device: str,
                epochs: int | None = None, e0_model: bool = False):
    overrides = [
        f"dataset={dataset}", "task=nc",
        f"model={'structured_executor_e0' if e0_model else 'provenance_executor_e01'}",
        f"model.variant={'edge_mix' if e0_model else variant}",
        f"seed={seed}", "num_runs=1", f"device={device}", "task.evaluate_test=false",
    ]
    if epochs is not None:
        overrides.append(f"task.epochs={int(epochs)}")
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def model_data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _module_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def audit_initialization(dataset: str, seed: int, data_info: dict[str, int], cfg) -> dict[str, Any]:
    cache_key = (dataset, int(seed))
    if cache_key in _INIT_CACHE:
        return _INIT_CACHE[cache_key]

    e01_states: dict[str, dict[str, torch.Tensor]] = {}
    e01_hashes: dict[str, str] = {}
    parameter_counts: dict[str, dict[str, int]] = {}
    classifier_states: dict[str, dict[str, torch.Tensor]] = {}
    classifier_hashes: dict[str, str] = {}
    for variant in E01_VARIANTS:
        set_seed(seed)
        local = copy.deepcopy(cfg)
        local.model.variant = variant
        model = Model(local, data_info)
        e01_states[variant] = _module_snapshot(model)
        e01_hashes[variant] = tensor_hash(e01_states[variant])
        model_count = sum(int(value.numel()) for value in model.parameters())
        trainable_count = sum(int(value.numel()) for value in model.parameters() if value.requires_grad)
        torch.manual_seed(seed + 1907)
        classifier = nn.Linear(model.out_dim, data_info["num_classes"])
        classifier_states[variant] = _module_snapshot(classifier)
        classifier_hashes[variant] = tensor_hash(classifier_states[variant])
        parameter_counts[variant] = {
            "model_params": model_count,
            "trainable_model_params": trainable_count,
            "classifier_params": sum(int(value.numel()) for value in classifier.parameters()),
            "total_trainable_params": trainable_count + sum(
                int(value.numel()) for value in classifier.parameters() if value.requires_grad
            ),
        }
        del model, classifier

    pairwise = {}
    for i, left in enumerate(E01_VARIANTS):
        for right in E01_VARIANTS[i + 1:]:
            pairwise[f"{left}__{right}"] = (
                e01_states[left].keys() == e01_states[right].keys()
                and all(torch.equal(e01_states[left][name], e01_states[right][name])
                        for name in e01_states[left])
                and all(torch.equal(classifier_states[left][name], classifier_states[right][name])
                        for name in classifier_states[left])
            )
    exact_parameters = len({row["total_trainable_params"] for row in parameter_counts.values()}) == 1
    if not all(pairwise.values()) or not exact_parameters:
        raise AssertionError(f"E0.1 variants do not share exact initialization/capacity: {pairwise}; {parameter_counts}")

    e0_cfg = make_config(dataset, seed, "edge_mix", "cpu", e0_model=True)
    set_seed(seed)
    e0 = E0Model(e0_cfg, data_info)
    e0_state = _module_snapshot(e0)
    reference = e01_states["keep_edge"]
    common_names = sorted(e0_state)
    missing = sorted(set(common_names) - set(reference))
    mismatched = [name for name in common_names if name in reference and not torch.equal(e0_state[name], reference[name])]
    if missing or mismatched:
        raise AssertionError(f"E0 common initialization differs: missing={missing}, mismatched={mismatched[:10]}")

    audit = {
        "dataset": dataset,
        "seed": int(seed),
        "variant_hashes": e01_hashes,
        "classifier_hashes": classifier_hashes,
        "pairwise_model_and_classifier_bitwise_equal": pairwise,
        "all_variants_bitwise_equal": all(pairwise.values()),
        "exact_trainable_parameter_match": exact_parameters,
        "parameter_counts": parameter_counts,
        "e0_common_init": {
            "e0_model_variant": "edge_mix",
            "common_state_entries": len(common_names),
            "bitwise_equal": True,
            "mismatched_entries": 0,
        },
    }
    _INIT_CACHE[cache_key] = audit
    return audit


def _gradient_audit(model: nn.Module) -> dict[str, dict[str, Any]]:
    groups = {
        "relation_encoder": ("rel_proj_t.", "rel_proj_v.", "phi_pair.", "phi_rel.",
                             "phi_mod.", "modality_embeddings"),
        "router": ("router.",),
        "smooth_expert": ("smooth_transforms.",),
        "relational_expert": ("relational_source.", "relational_mlps."),
        "cross_expert": ("cross_modal_source.", "cross_modal_mlps."),
        "composer_first_layer": ("provenance_composers.0.0.", "provenance_composers.1.0."),
        "composer_final_layer": ("provenance_composers.0.2.", "provenance_composers.1.2."),
        "fusion": ("fusion.",),
    }
    result = {}
    for name, prefixes in groups.items():
        grads = [parameter.grad.detach() for key, parameter in model.named_parameters()
                 if key.startswith(prefixes) and parameter.grad is not None]
        result[name] = {
            "finite": bool(grads) and all(bool(torch.isfinite(grad).all()) for grad in grads),
            "norm": float(torch.sqrt(sum(grad.float().square().sum() for grad in grads)).item()) if grads else 0.0,
            "parameter_tensors_with_grad": len(grads),
        }
    return result


def _metric(model, classifier, x, edge_index, val_idx, val_y, class_count,
            control_overrides=None, composer_input_mode="normal", composer_off=False):
    model.eval()
    classifier.eval()
    with torch.no_grad():
        z, _, _, _, _ = model(
            x, edge_index, control_overrides=control_overrides,
            composer_input_mode=composer_input_mode, composer_off=composer_off,
        )
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("non-finite E0.1 validation embeddings")
        logits = classifier(z[val_idx])
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite E0.1 validation logits")
        return _metric_from_embeddings(classifier, z, val_idx, val_y, class_count)


def _distribution(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: float("nan") for key in ("mean", "std", "q10", "q25", "median", "q75", "q90")}
    q10, q25, median, q75, q90 = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)),
            "q10": float(q10), "q25": float(q25), "median": float(median),
            "q75": float(q75), "q90": float(q90)}


def _context_diagnostics(aux: dict[str, Any], val_idx: torch.Tensor, eps: float,
                         dataset: str, seed: int, variant: str):
    channel_norm_rows, distinct_rows, mass_rows, composer_rows = [], [], [], []
    names = ("smooth", "relational", "cross_modal")
    for modality_id, modality in enumerate(("text", "visual")):
        channels = aux["channel_contexts"][modality_id]
        c_mix = aux["c_mix"][modality_id]
        comp = aux["composer_output"][modality_id]
        selected = [value[val_idx] for value in channels]
        mix_selected, comp_selected = c_mix[val_idx], comp[val_idx]
        rms_by_channel = [((value.square().mean(-1) + eps).sqrt()) for value in selected]
        mix_rms = (mix_selected.square().mean(-1) + eps).sqrt()
        for name, value, rms_value in zip(names, selected, rms_by_channel):
            stats = _distribution(rms_value)
            channel_norm_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                      "modality": modality, "channel": name,
                                      "n_validation_nodes": int(val_idx.numel()), **stats})
        stats = _distribution(mix_rms)
        channel_norm_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                  "modality": modality, "channel": "mix",
                                  "n_validation_nodes": int(val_idx.numel()), **stats})

        pair_defs = ((0, 1, "smooth_relational"), (0, 2, "smooth_cross_modal"),
                     (1, 2, "relational_cross_modal"))
        for left_id, right_id, pair_name in pair_defs:
            left, right = selected[left_id], selected[right_id]
            left_norm, right_norm = torch.linalg.vector_norm(left, dim=-1), torch.linalg.vector_norm(right, dim=-1)
            valid = (left_norm > eps) & (right_norm > eps)
            if bool(valid.any()):
                cosine = torch.nn.functional.cosine_similarity(left[valid], right[valid], dim=-1, eps=eps)
            else:
                cosine = torch.empty(0, dtype=left.dtype)
            stats = _distribution(cosine)
            distinct_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                                  "modality": modality, "pair": pair_name,
                                  "valid_node_count": int(valid.sum()), **stats,
                                  "iqr": stats["q75"] - stats["q25"]})

        total_mass = sum(rms_by_channel)
        for name, rms_value in zip(names, rms_by_channel):
            mass = rms_value / (total_mass + eps)
            stats = _distribution(mass)
            mass_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                              "modality": modality, "channel": name, **stats})

        comp_rms = (comp_selected.square().mean(-1) + eps).sqrt()
        ratio = comp_rms / (mix_rms + eps)
        nonzero = (torch.linalg.vector_norm(comp_selected, dim=-1) > eps) & (
            torch.linalg.vector_norm(mix_selected, dim=-1) > eps
        )
        cosine = torch.nn.functional.cosine_similarity(
            comp_selected[nonzero], mix_selected[nonzero], dim=-1, eps=eps
        ) if bool(nonzero.any()) else torch.empty(0, dtype=comp_selected.dtype)
        ratio_stats, cosine_stats = _distribution(ratio), _distribution(cosine)
        composer_rows.append({"dataset": dataset, "seed": seed, "variant": variant,
                              "modality": modality, "valid_cosine_node_count": int(nonzero.sum()),
                              "ratio_mean": ratio_stats["mean"], "ratio_std": ratio_stats["std"],
                              "ratio_median": ratio_stats["median"],
                              "ratio_q25": ratio_stats["q25"], "ratio_q75": ratio_stats["q75"],
                              "ratio_iqr": ratio_stats["q75"] - ratio_stats["q25"],
                              "ratio_q90": ratio_stats["q90"],
                              "cosine_mean": cosine_stats["mean"], "cosine_std": cosine_stats["std"],
                              "cosine_median": cosine_stats["median"],
                              "cosine_q10": cosine_stats["q10"], "cosine_q90": cosine_stats["q90"]})
    return channel_norm_rows, distinct_rows, mass_rows, composer_rows


def _scale_diagnostics(aux: dict[str, Any], val_idx: torch.Tensor,
                       dataset: str, seed: int, variant: str):
    """Summarize frozen E0 calibration behavior on validation-target edges."""
    edge_dst = aux["edge_index_nonself"][1].detach().cpu().long()
    validation_nodes = val_idx.detach().cpu().long()
    target_mask = torch.isin(edge_dst, validation_nodes)
    scale_fields = aux.get("expert_scales")
    if not scale_fields:
        raise KeyError("model diagnostics omitted frozen E0 expert calibration scales")
    rows = []
    for modality_id, modality in enumerate(("text", "visual")):
        for name, per_modality in scale_fields.items():
            values = per_modality[modality_id].detach().cpu().reshape(-1)
            if values.numel() != edge_dst.numel():
                raise AssertionError(
                    f"{name}/{modality} has {values.numel()} values for {edge_dst.numel()} non-self edges"
                )
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError(f"non-finite E0 calibration diagnostic: {name}/{modality}")
            selected = values[target_mask]
            if selected.numel() == 0:
                raise AssertionError(f"no validation-target edges available for {dataset}/{modality}")
            array = selected.numpy().astype(np.float64, copy=False)
            q10, q25, median, q75, q90, q99 = np.quantile(
                array, [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
            )
            rows.append({
                "dataset": dataset, "seed": int(seed), "variant": variant,
                "modality": modality, "diagnostic": name,
                "n_validation_target_edges": int(selected.numel()),
                "mean": float(array.mean()), "std": float(array.std(ddof=0)),
                "min": float(array.min()), "q10": float(q10), "q25": float(q25),
                "median": float(median), "q75": float(q75), "q90": float(q90),
                "q99": float(q99), "max": float(array.max()),
                "all_finite": True, "clamped": False,
            })
    return rows


def audit_calibration(args):
    """Post-hoc forward-only audit of all selected checkpoints; no retraining or test evaluation."""
    records_dir = args.output_dir / "runs"
    run_paths = sorted(records_dir.glob("*/*/*.json"))
    expected = {(d, s, v) for d in DATASETS for s in SEEDS for v in E01_VARIANTS}
    keyed = {}
    for path in run_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["dataset"], int(record["seed"]), record["variant"])
        if record.get("status") == "completed":
            if key in keyed:
                raise AssertionError(f"duplicate completed formal record during calibration audit: {key}")
            keyed[key] = (path, record)
    if set(keyed) != expected:
        raise AssertionError(f"calibration audit requires exactly 36 formal runs; found {len(keyed)}")

    device = torch.device(args.device)
    audit_rows = []
    peak_bytes = 0
    for index, key in enumerate(sorted(expected), start=1):
        dataset, seed, variant = key
        run_path, record = keyed[key]
        checkpoint_path = args.output_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing selected checkpoint for {key}: {checkpoint_path}")
        cfg = make_config(dataset, seed, variant, args.device, e0_model=False)
        data = load_m0_data(cfg, seed)
        split = restrict_labels_to_train_val(data)
        val_idx = split["validation"]
        data_info = model_data_info(data)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = Model(cfg, data_info).to(device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
        scale_rows = _scale_diagnostics(aux, val_idx, dataset, seed, variant)
        if len(scale_rows) != 14:
            raise AssertionError(f"expected 14 modality×scale rows for {key}, found {len(scale_rows)}")
        record["expert_scale_rows"] = scale_rows
        record["calibration_audit"] = {
            "status": "passed", "scope": "validation-target non-self edges",
            "all_finite": True, "clamping_applied": False,
            "checkpoint_reloaded_without_training": True,
            "test_evaluation": False, "link_prediction": False,
        }
        write_json(run_path, record)
        audit_rows.extend(scale_rows)
        if device.type == "cuda":
            peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated(device)))
        del model, data, x, edge_index, checkpoint, aux
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[E0.1 calibration audit] {index}/36 {dataset}/{seed}/{variant} finite; unclamped", flush=True)

    result = {
        "status": "complete", "completed_runs": 36,
        "rows": len(audit_rows), "expected_rows": 36 * 2 * 7,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if device.type == "cuda" else "cpu",
        "scope": "validation-target non-self edges; selected checkpoints; forward only",
        "all_finite": all(row["all_finite"] for row in audit_rows),
        "clamping_applied": False, "retraining": False,
        "test_evaluation": False, "link_prediction": False,
        "peak_allocated_gpu_memory_bytes": peak_bytes,
    }
    if len(audit_rows) != result["expected_rows"] or not result["all_finite"]:
        raise AssertionError(f"incomplete or non-finite calibration audit: {result}")
    write_json(args.output_dir / "calibration_audit_result.json", result)
    print(f"[E0.1 calibration audit complete] {result['rows']} rows; all finite; no clamping", flush=True)
    return result


def _intervention_rows(model, classifier, x, edge_index, val_idx, val_y, class_count,
                       base_metrics: dict[str, float], base_aux: dict[str, Any],
                       dataset: str, seed: int):
    if model.e01_variant != "keep_edge":
        raise ValueError("E0.1 checkpoint interventions are defined for KeepEdge only")
    pi = base_aux["controls"]["pi"]
    nonself_dst = base_aux["edge_index_nonself"][1].long()
    rows = []

    def add(name, *, pi_override=None, composer_mode="normal", composer_off=False, repeat=None):
        overrides = None if pi_override is None else {"pi": pi_override}
        metrics = _metric(model, classifier, x, edge_index, val_idx, val_y, class_count,
                          control_overrides=overrides, composer_input_mode=composer_mode,
                          composer_off=composer_off)
        row = {"dataset": dataset, "seed": seed, "intervention": name, "repeat_seed": repeat,
               **metrics,
               "delta_accuracy_pp": 100.0 * (metrics["val_accuracy"] - base_metrics["val_accuracy"]),
               "delta_macro_f1_pp": 100.0 * (metrics["val_macro_f1"] - base_metrics["val_macro_f1"]),
               "delta_ce": metrics["val_ce"] - base_metrics["val_ce"]}
        if name == "provenance_permutation":
            with torch.no_grad():
                _, _, _, _, aux = model(
                    x, edge_index, composer_input_mode=composer_mode, return_diagnostics=True
                )
            row["max_abs_cmix_change"] = max(
                float((aux["c_mix"][m] - base_aux["c_mix"][m]).abs().max()) for m in range(2)
            )
        rows.append(row)

    add("provenance_collapse", composer_mode="collapse")
    for order in ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        add("provenance_permutation", composer_mode="permute:" + ",".join(map(str, order)),
            repeat="".join(map(str, order)))
    add("composer_off", composer_off=True)
    for name, repeat, intervention_pi in provenance_interventions(pi, nonself_dst, int(x.size(0))):
        add(name, pi_override=intervention_pi, repeat=repeat)
    return rows


def _run_e0_regressions(dataset: str, seed: int, cfg, data, data_info, device):
    e0_cfg = make_config(dataset, seed, "edge_mix", str(device), e0_model=True)
    set_seed(seed)
    e0 = E0Model(e0_cfg, data_info).to(device).eval()
    set_seed(seed)
    e01_cfg = copy.deepcopy(cfg)
    e01_cfg.model.variant = "keep_edge"
    e01 = Model(e01_cfg, data_info).to(device).eval()
    regression = {"h0": True, "p": True, "q": True, "r": True, "u": True,
                  "loo_context": True, "edge_support": True, "degree": True,
                  "smooth_expert": True, "relational_expert": True,
                  "cross_expert": True, "router": True}
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        s0 = e0.relation_state(x, edge_index)
        s1 = e01.relation_state(x, edge_index)
        for key in ("h0", "p", "contexts"):
            regression["loo_context" if key == "contexts" else key] = all(
                torch.allclose(a, b, rtol=1e-6, atol=1e-7) for a, b in zip(s0[key], s1[key])
            )
        # Reuse the same computed LOO context for q/r/u. Independent CUDA
        # index_add reductions can differ in their last few bits even when the
        # contexts themselves satisfy their explicit tolerance check.
        q0, qv0, r0, u0 = e0._functional_states(
            s0["p"], s0["contexts"], s0["deg_z"], s0["src"], s0["dst"]
        )
        q1, qv1, r1, u1 = e01._functional_states(
            s0["p"], s0["contexts"], s0["deg_z"], s0["src"], s0["dst"]
        )
        regression["q"] = all(torch.equal(a, b) for a, b in zip((q0, qv0), (q1, qv1)))
        regression["r"] = torch.equal(r0, r1)
        regression["u"] = all(torch.equal(a, b) for a, b in zip(u0, u1))
        regression["edge_support"] = torch.equal(s0["src"], s1["src"]) and torch.equal(s0["dst"], s1["dst"])
        regression["degree"] = torch.equal(s0["degree"], s1["degree"])
        e0_out = e0._expert_outputs(s0["h0"], s0["src"], s0["dst"])
        e01_out = e01._expert_outputs(s1["h0"], s1["src"], s1["dst"])
        for index, key in ((0, "smooth_expert"), (1, "relational_expert"), (2, "cross_expert")):
            regression[key] = all(torch.equal(a, b) for a, b in zip(e0_out[index], e01_out[index]))
        pi0, pi1 = [], []
        for modality in range(2):
            pi0.append(torch.softmax(e0.router(u0[modality]), dim=-1))
            pi1.append(torch.softmax(e01.router(u1[modality]), dim=-1))
        regression["router"] = all(torch.equal(a, b) for a, b in zip(pi0, pi1))
    if not all(regression.values()):
        raise AssertionError(f"E0/E0.1 module regression failed: {regression}")
    del e0, e01
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return regression


def run_one(dataset: str, seed: int, variant: str, device_name: str,
            out_dir: Path = OUT_DIR, epochs: int | None = None,
            save_checkpoint: bool = True, run_interventions: bool = True) -> dict[str, Any]:
    cfg = make_config(dataset, seed, variant, device_name, epochs)
    data = load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    data_info = model_data_info(data)
    initialization = audit_initialization(dataset, seed, data_info, cfg)
    if not initialization["all_variants_bitwise_equal"] or not initialization["exact_trainable_parameter_match"]:
        raise AssertionError("E0.1 common initialization/parameter audit failed")

    device = torch.device(device_name)
    set_seed(seed)
    model = Model(cfg, data_info).to(device)
    torch.manual_seed(seed + 1907)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1907)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    set_seed(seed)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    train_idx, val_idx = splits["train"].to(device), splits["validation"].to(device)
    labels = data.y.to(device)
    train_y, val_y = labels[train_idx], labels[val_idx]
    if bool((train_y < 0).any()) or bool((val_y < 0).any()) or data.test_idx is not None:
        raise AssertionError("E0.1 train/validation protocol exposed invalid labels or test indices")
    if bool(cfg.task.evaluate_test):
        raise AssertionError("E0.1 requires task.evaluate_test=false")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_acc, best_epoch = -1.0, 0
    patience_left = int(cfg.task.patience)
    best_model = best_classifier = None
    best_metrics = None
    first_gradients = {}
    started = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, int(cfg.task.epochs) + 1):
        epochs_run = epoch
        model.train(); classifier.train(); optimizer.zero_grad(set_to_none=True)
        z, _, _, aux_loss, _ = model(x, edge_index)
        logits = classifier(z[train_idx])
        loss = nn.functional.cross_entropy(logits, train_y) + float(cfg.task.loss.aux_weight) * aux_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at {dataset}/{seed}/{variant}/epoch{epoch}")
        loss.backward()
        if epoch == 1:
            first_gradients = _gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()),
                                       max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True)
        optimizer.step()
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        del z, logits, loss

        current = _metric(model, classifier, x, edge_index, val_idx, val_y, int(data.num_classes))
        if current["val_accuracy"] > best_acc + float(cfg.task.early_stop_min_delta):
            best_acc, best_epoch = current["val_accuracy"], epoch
            best_metrics = current
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_classifier = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            patience_left = int(cfg.task.patience)
        elif epoch >= int(cfg.task.early_stop_min_epoch):
            patience_left -= 1
            if patience_left <= 0:
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    if best_model is None or best_classifier is None:
        raise RuntimeError("E0.1 training ended without a validation-selected checkpoint")
    for name, audit in first_gradients.items():
        if not audit["finite"] or audit["norm"] <= 0:
            raise AssertionError(f"E0.1 first-epoch gradients invalid for {name}: {audit}")

    model.load_state_dict(best_model, strict=True)
    classifier.load_state_dict(best_classifier, strict=True)
    model.eval(); classifier.eval()
    with torch.no_grad():
        z, _, _, _, base_aux = model(x, edge_index, return_diagnostics=True)
        selected_metrics = _metric_from_embeddings(classifier, z, val_idx, val_y, data.num_classes)
        if not torch.isfinite(z).all():
            raise FloatingPointError("non-finite selected E0.1 checkpoint embeddings")
    identity = {
        "channel_sum_max_abs": max(
            float((base_aux["c_mix"][m] - sum(base_aux["channel_contexts"][m])).abs().max())
            for m in range(2)
        ),
        "static_global_pi_max_abs": None,
        "target_mean_pi_max_abs": None,
        "premix_collapsed_input_max_abs": None,
    }
    if identity["channel_sum_max_abs"] > 1e-6:
        raise AssertionError(f"Cmix != Cs+Cr+Cx: {identity['channel_sum_max_abs']}")
    pi = base_aux["controls"]["pi"]
    dst = base_aux["edge_index_nonself"][1].long()
    if variant == "keep_static":
        identity["static_global_pi_max_abs"] = max(float((pi_global_mean(pi[m]) - pi[m]).abs().max()) for m in range(2))
        if identity["static_global_pi_max_abs"] > 1e-6:
            raise AssertionError("KeepStatic global routing identity failed")
    if variant == "keep_target":
        identity["target_mean_pi_max_abs"] = max(
            float((pi_target_mean(pi[m], dst, data.num_nodes) - pi[m]).abs().max()) for m in range(2)
        )
        # A target's edge rows are identical by construction. CUDA index_add
        # averaging can still leave a few ppm of reduction-order error on
        # large-degree nodes; the CPU identity unit test remains at 1e-6.
        if identity["target_mean_pi_max_abs"] > 1e-5:
            raise AssertionError("KeepTarget target-mean routing identity failed")
    if variant == "premix_edge_control":
        with torch.no_grad():
            collapsed_z = model(x, edge_index, composer_input_mode="collapse")[0]
        identity["premix_collapsed_input_max_abs"] = float((collapsed_z - z).abs().max())
        if identity["premix_collapsed_input_max_abs"] > 2e-6:
            raise AssertionError("PremixEdgeControl should already have collapsed provenance")

    context_rows = _context_diagnostics(base_aux, val_idx.cpu(), model.eps, dataset, seed, variant)
    intervention_rows = []
    intervention_seconds = 0.0
    if variant == "keep_edge" and run_interventions:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_started = time.perf_counter()
        intervention_rows = _intervention_rows(
            model, classifier, x, edge_index, val_idx, val_y, int(data.num_classes),
            selected_metrics, base_aux, dataset, seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_seconds = time.perf_counter() - intervention_started

    model_params = sum(int(parameter.numel()) for parameter in model.parameters())
    classifier_params = sum(int(parameter.numel()) for parameter in classifier.parameters())
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    record = {
        "status": "completed", "dataset": dataset, "seed": int(seed), "variant": variant,
        "best_epoch": best_epoch, "epochs_run": epochs_run, **selected_metrics,
        "model_params": model_params, "classifier_params": classifier_params,
        "total_trainable_params": model_params + classifier_params,
        "peak_gpu_memory_bytes": peak_bytes, "training_time_sec": train_seconds,
        "intervention_time_sec": intervention_seconds,
        "gradient_audit_epoch1": first_gradients,
        "initialization_audit": initialization,
        "identity_checks": identity,
        "channel_norm_rows": context_rows[0], "channel_distinctness_rows": context_rows[1],
        "channel_mass_rows": context_rows[2], "composer_diagnostic_rows": context_rows[3],
        "intervention_rows": intervention_rows,
        "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        "test_indices_attached": False, "test_labels_exposed": False,
        "link_prediction": False, "data_info": data_info,
    }
    run_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    write_json(run_path, record)
    if save_checkpoint:
        checkpoint = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_model, "head_state": best_classifier,
                    "data_info": data_info, "dataset": dataset, "seed": int(seed),
                    "variant": variant, "best_epoch": best_epoch,
                    "best_val_metrics": selected_metrics,
                    "protocol": "unified_full_graph_nc_v1", "evaluate_test": False}, checkpoint)
        record["checkpoint"] = str(checkpoint.relative_to(PROJECT_ROOT))
        write_json(run_path, record)

    print(f"[E0.1 done] {dataset}/{seed}/{variant} epoch={best_epoch} "
          f"acc={selected_metrics['val_accuracy']:.4f} f1={selected_metrics['val_macro_f1']:.4f} "
          f"ce={selected_metrics['val_ce']:.4f} train={train_seconds:.1f}s "
          f"interventions={intervention_seconds:.1f}s peak={peak_bytes/1024**3:.2f}GiB", flush=True)
    del z, base_aux, data, x, edge_index, model, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _smoke_intervention_identities(record: dict[str, Any], out_dir: Path):
    data_info = record["data_info"]
    dataset, seed = record["dataset"], int(record["seed"])
    cfg = make_config(dataset, seed, "keep_edge", "cpu", e0_model=False)
    run_path = out_dir / "runs" / dataset / f"seed_{seed}" / "keep_edge.json"
    checkpoint_path = out_dir / "checkpoints" / dataset / f"seed_{seed}_keep_edge.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("smoke KeepEdge checkpoint needed for intervention audit is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Model(cfg, data_info).eval()
    model.load_state_dict(checkpoint["model_state"])
    data = load_m0_data(cfg, seed)
    split = restrict_labels_to_train_val(data)
    val_idx = split["validation"]
    with torch.no_grad():
        _, _, _, _, aux = model(data.x, data.edge_index, return_diagnostics=True)
        pi = aux["controls"]["pi"]
        dst = aux["edge_index_nonself"][1]
        shuffle_checks = {}
        for modality in range(2):
            order = neighborhood_shuffle_indices(dst, 1001 + modality * 10000)
            for node in torch.unique(dst):
                assert sorted(map(tuple, pi[modality][dst == node].tolist())) == sorted(
                    map(tuple, pi[modality][order][dst == node].tolist())
                )
            shuffle_checks[str(modality)] = True
        route_items = provenance_interventions(pi, dst, data.num_nodes)
        collapsed = model(data.x, data.edge_index, composer_input_mode="collapse", return_diagnostics=True)
        composer_off = model(data.x, data.edge_index, composer_off=True, return_diagnostics=True)
        permutations = [item for item in route_items if item[0] == "pi_shuffle_within_target"]
        smooth = next(item for item in route_items if item[0] == "smooth_only")[2]
        smooth_result = model(data.x, data.edge_index, control_overrides={"pi": smooth}, return_diagnostics=True)
    assert float((collapsed[4]["c_mix"][0] - aux["c_mix"][0]).abs().max()) == 0.0
    assert float((composer_off[4]["delta"][0] - aux["c_mix"][0]).abs().max()) == 0.0
    assert len(permutations) == 5
    for modality in range(2):
        cs, cr, cx = smooth_result[4]["channel_contexts"][modality]
        assert float(cr.abs().max()) == 0.0 and float(cx.abs().max()) == 0.0
        assert torch.equal(smooth_result[4]["composer_input"][modality], torch.cat((cs, cr, cx), -1))
    return {
        "shuffle_multiset_preserved_by_target": shuffle_checks,
        "provenance_permutation_count": len(permutations),
        "collapse_cmix_unchanged": True,
        "composer_off_delta_equals_cmix": True,
        "smooth_only_channels_and_input_correct": True,
        "test_access": False,
    }


def run_smoke(device_name: str, out_dir: Path = OUT_DIR):
    dataset, seed = "Movies", 42
    # Force the actual NC protocol to expose only train/validation labels.
    cfg = make_config(dataset, seed, "keep_edge", device_name, epochs=2)
    data = load_m0_data(cfg, seed)
    smoke_splits = restrict_labels_to_train_val(data)
    data_info = model_data_info(data)
    init = audit_initialization(dataset, seed, data_info, cfg)
    device = torch.device(device_name)
    regression = _run_e0_regressions(dataset, seed, cfg, data, data_info, device)
    # Record the small-final-layer initialization check before any optimizer step.
    set_seed(seed)
    initial_model = Model(copy.deepcopy(cfg), data_info).to(device).eval()
    with torch.no_grad():
        _, _, _, _, initial_aux = initial_model(
            data.x.to(device), data.edge_index.to(device), return_diagnostics=True
        )
    initial_composer = []
    for modality_id, modality in enumerate(("text", "visual")):
        c_mix = initial_aux["c_mix"][modality_id][smoke_splits["validation"]]
        comp = initial_aux["composer_output"][modality_id][smoke_splits["validation"]]
        ratio = (comp.square().mean(-1) + initial_model.eps).sqrt() / (
            (c_mix.square().mean(-1) + initial_model.eps).sqrt() + initial_model.eps
        )
        stats = _distribution(ratio)
        initial_composer.append({"modality": modality, "median": stats["median"], "q90": stats["q90"]})
    if any(row["median"] >= 1.0 for row in initial_composer):
        raise AssertionError(f"initial composer median correction exceeds base context: {initial_composer}")
    del initial_model, initial_aux
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = []
    smoke_out_dir = out_dir / "smoke_runs"
    for variant in E01_VARIANTS:
        record = run_one(dataset, seed, variant, device_name, smoke_out_dir,
                         epochs=2, save_checkpoint=(variant == "keep_edge"),
                         run_interventions=(variant == "keep_edge"))
        if not all(item["finite"] and item["norm"] > 0 for item in record["gradient_audit_epoch1"].values()):
            raise AssertionError(f"E0.1 smoke gradients invalid for {variant}")
        results.append({"dataset": dataset, "seed": seed,
                        "variant": variant, "best_epoch": record["best_epoch"],
                        "metrics": {key: record[key] for key in ("val_accuracy", "val_macro_f1", "val_ce")},
                        "trainable_params": record["total_trainable_params"],
                        "peak_gpu_memory_bytes": record["peak_gpu_memory_bytes"],
                        "gradient_audit_epoch1": record["gradient_audit_epoch1"],
                        "identity_checks": record["identity_checks"],
                        "composer_init_ratio_median_q90": [
                            {key: row[key] for key in ("modality", "ratio_median", "ratio_q90")}
                            for row in record["composer_diagnostic_rows"]
                        ]})
    e01_state = {
        "status": "smoke_passed", "dataset": dataset, "seed": seed, "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "e0_common_regression": regression,
        "initialization_and_parameter_audit": init,
        "initial_composer_base_ratio": initial_composer,
        "variants": results,
        "intervention_identity_audit": _smoke_intervention_identities(
            results[-1] | {"data_info": data_info}, smoke_out_dir
        ),
        "edge_chunk_size": int(cfg.model.edge_chunk_size),
        "evaluate_test": False, "test_indices_attached": False,
        "test_labels_exposed": False, "link_prediction": False,
    }
    write_json(out_dir / "smoke" / "Movies_seed_42" / "smoke_result.json", e01_state)
    print(f"[E0.1 smoke passed] Movies/42 variants=4 device={device_name}", flush=True)
    del data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return e01_state


def run_campaign(args):
    smoke_path = args.output_dir / "smoke" / "Movies_seed_42" / "smoke_result.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "smoke_passed":
        raise RuntimeError("E0.1 Movies/42 four-variant smoke must pass before formal campaign")
    total = len(args.datasets) * len(args.seeds) * len(args.variants)
    completed = []
    failure_path = args.output_dir / "campaign_failures.json"
    failure_events = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else []
    failed_keys = {(row["dataset"], int(row["seed"]), row["variant"]) for row in failure_events}
    retried_keys: set[tuple[str, int, str]] = set()
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                run_key = (dataset, int(seed), variant)
                run_path = args.output_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if run_path.is_file():
                    previous = json.loads(run_path.read_text(encoding="utf-8"))
                    if previous.get("status") == "completed" and previous.get("evaluate_test") is False:
                        completed.append(previous)
                        if run_key in failed_keys:
                            retried_keys.add(run_key)
                        print(f"[skip-complete] {dataset}/{seed}/{variant}", flush=True)
                        continue
                if run_key in failed_keys:
                    retried_keys.add(run_key)
                print(f"[run {len(completed)+1}/{total}] {dataset}/{seed}/{variant} on {args.device}", flush=True)
                try:
                    record = run_one(dataset, seed, variant, args.device, args.output_dir,
                                     run_interventions=(variant == "keep_edge"))
                except Exception as error:
                    failure_events.append({"dataset": dataset, "seed": int(seed), "variant": variant,
                                           "error_type": type(error).__name__, "error": str(error),
                                           "formal_run_record_written": False})
                    write_json(failure_path, failure_events)
                    raise
                completed.append(record)
                write_json(args.output_dir / "campaign_progress.json", {
                    "requested_runs": total,
                    "completed_unique_runs": len({(r["dataset"], r["seed"], r["variant"]) for r in completed}),
                    "datasets": list(args.datasets), "seeds": list(args.seeds),
                    "variants": list(args.variants), "device": args.device,
                    "evaluate_test": False,
                    "failed_attempts_recorded": len(failure_events),
                    "formal_retries": len(retried_keys),
                })
    unique = {(r["dataset"], int(r["seed"]), r["variant"]) for r in completed}
    if len(completed) != total or len(unique) != total:
        raise AssertionError(f"E0.1 campaign has duplicate/missing run rows: {len(completed)}/{total}, unique={len(unique)}")
    result = {
        "status": "completed", "requested_new_runs": total,
        "completed_new_runs": len(completed), "failed_formal_runs": len(failure_events),
        "formal_reruns": len(retried_keys),
        "datasets": list(args.datasets), "seeds": list(args.seeds),
        "variants": list(args.variants), "device": args.device,
        "sum_training_time_seconds": sum(float(r["training_time_sec"]) for r in completed),
        "sum_keep_edge_intervention_time_seconds": sum(float(r["intervention_time_sec"])
                                                          for r in completed if r["variant"] == "keep_edge"),
        "max_peak_gpu_memory_bytes": max(int(r["peak_gpu_memory_bytes"]) for r in completed),
        "evaluate_test": False, "link_prediction": False,
    }
    write_json(args.output_dir / "campaign_result.json", result)
    print(f"[E0.1 campaign finished] {len(completed)}/{total}; "
          f"training={result['sum_training_time_seconds']:.1f}s; "
          f"interventions={result['sum_keep_edge_intervention_time_seconds']:.1f}s; "
          f"peak={result['max_peak_gpu_memory_bytes']/1024**3:.2f}GiB", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E0.1 Function-Provenance Preservation Screen")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--audit-calibration", action="store_true",
                        help="forward-only scale audit of all completed formal checkpoints")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=E01_VARIANTS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=E01_VARIANTS, default=list(E01_VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    gpu = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"
    print(f"[E0.1] device={args.device} gpu={gpu} torch={torch.__version__} "
          f"pyg={__import__('torch_geometric').__version__}", flush=True)
    if args.smoke:
        run_smoke(args.device, args.output_dir)
    elif args.campaign:
        run_campaign(args)
    elif args.audit_calibration:
        audit_calibration(args)
    elif args.dataset and args.seed is not None and args.variant:
        run_one(args.dataset, args.seed, args.variant, args.device, args.output_dir, args.epochs,
                run_interventions=(args.variant == "keep_edge"))
    else:
        parser.error("choose --smoke, --campaign, or one --dataset/--seed/--variant run")


if __name__ == "__main__":
    main()
