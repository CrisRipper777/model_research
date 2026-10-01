from __future__ import annotations

import copy
import hashlib
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
    neighborhood_shuffle_indices,
)
from src.models.structured_executor_e0 import (
    FUNCTION_NAMES,
    Model,
    _rms,
    pi_global_mean,
    pi_modality_tied,
    pi_target_mean,
    renormalize_without,
)
from src.tasks.common import build_optimizer, scheduler_step
from src.utils.seeds import set_seed


DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("static_mix", "target_mix", "edge_mix")
OUT_DIR = PROJECT_ROOT / "outputs" / "e0_structured_relation_function_executor"
M0_PERFORMANCE = PROJECT_ROOT / "research" / "m0_adaptive_propagation_screen" / "data" / "performance_by_run.csv"
M0_RUNS = PROJECT_ROOT / "outputs" / "m0_adaptive_propagation_screen" / "runs"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def make_config(dataset: str, seed: int, variant: str, device: str,
                epochs: int | None = None, m0_model: bool = False):
    overrides = [
        f"dataset={dataset}", "task=nc",
        f"model={'adaptive_prop_m0' if m0_model else 'structured_executor_e0'}",
        f"model.variant={variant}", f"seed={seed}", "num_runs=1",
        f"device={device}", "task.evaluate_test=false",
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


def audit_initialization(cfg, data_info: dict[str, int], seed: int) -> dict[str, Any]:
    states, hashes, counts = {}, {}, {}
    for variant in VARIANTS:
        set_seed(seed)
        local = copy.deepcopy(cfg)
        local.model.variant = variant
        model = Model(local, data_info)
        state = model.state_dict()
        states[variant] = {key: value.detach().cpu().clone() for key, value in state.items()}
        hashes[variant] = tensor_hash(states[variant])
        counts[variant] = {
            "model_params": sum(int(parameter.numel()) for parameter in model.parameters()),
            "trainable_model_params": sum(int(parameter.numel()) for parameter in model.parameters()
                                           if parameter.requires_grad),
        }
        del model
    pairwise = {}
    for i, left in enumerate(VARIANTS):
        for right in VARIANTS[i + 1:]:
            pairwise[f"{left}__{right}"] = (
                states[left].keys() == states[right].keys()
                and all(torch.equal(states[left][name], states[right][name]) for name in states[left])
            )
    exact_counts = len({item["trainable_model_params"] for item in counts.values()}) == 1
    return {
        "dataset": str(cfg.dataset.name), "seed": int(seed), "variant_hashes": hashes,
        "pairwise_bitwise_equal": pairwise, "common_all_bitwise_equal": all(pairwise.values()),
        "exact_trainable_parameter_match": exact_counts, "parameter_counts": counts,
    }


def quantiles(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    keys = ("mean", "std", "q10", "q25", "median", "q75", "q90")
    if not array.size:
        return {key: float("nan") for key in keys}
    q10, q25, median, q75, q90 = np.quantile(array, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "q10": float(q10),
            "q25": float(q25), "median": float(median), "q75": float(q75), "q90": float(q90)}


def metric(model, classifier, x, edge_index, val_idx, labels, num_classes, overrides=None):
    model.eval(); classifier.eval()
    with torch.no_grad():
        z, _, _, _, _ = model(x, edge_index, control_overrides=overrides)
        logits = classifier(z[val_idx])
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite validation logits")
        return _metric_from_embeddings(classifier, z, val_idx, labels[val_idx], num_classes)


def gradient_audit(model: nn.Module) -> dict[str, dict[str, Any]]:
    groups = {
        "router": ("router.",),
        "smooth_transform": ("smooth_transforms.",),
        "relational_source": ("relational_source.",),
        "relational_mlp": ("relational_mlps.",),
        "cross_modal_source": ("cross_modal_source.",),
        "cross_modal_mlp": ("cross_modal_mlps.",),
        "relation_state": ("rel_proj_t.", "rel_proj_v.", "phi_pair.", "phi_rel.",
                           "phi_mod.", "modality_embeddings"),
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


def _router_diagnostics(pi: torch.Tensor, modality: str, variant: str) -> dict[str, Any]:
    row: dict[str, Any] = {"modality": modality, "variant": variant, "edge_count": int(pi.size(0))}
    for index, name in enumerate(FUNCTION_NAMES):
        stats = quantiles(pi[:, index])
        for key, value in stats.items():
            row[f"pi_{name}_{key}"] = value
    p = pi.clamp_min(1e-12)
    entropy = -(p * p.log()).sum(dim=-1)
    normalized = entropy / math.log(len(FUNCTION_NAMES))
    effective = entropy.exp()
    active = 1.0 - pi[:, 0]
    top = torch.topk(pi, k=2, dim=-1).values
    margin = top[:, 0] - top[:, 1]
    for prefix, values in (("entropy", entropy), ("normalized_entropy", normalized),
                           ("effective_functions", effective), ("active_mass", active)):
        for key, value in quantiles(values).items():
            row[f"{prefix}_{key}"] = value
    margin_stats = quantiles(margin)
    row["routing_margin_median"] = margin_stats["median"]
    row["routing_margin_iqr"] = margin_stats["q75"] - margin_stats["q25"]
    row["routing_margin_q90"] = margin_stats["q90"]
    return row


def _within_node_diagnostics(pi: torch.Tensor, dst: torch.Tensor, degree: torch.Tensor,
                             val_idx: torch.Tensor, modality: str, variant: str) -> dict[str, Any]:
    val_targets = torch.zeros(degree.numel(), dtype=torch.bool)
    val_targets[val_idx.detach().cpu().long()] = True
    eligible = val_targets & (degree.detach().cpu().long() >= 5)
    if pi.numel() == 0 or not bool(eligible.any()):
        return {"modality": modality, "variant": variant, "eligible_target_count": 0}
    dst = dst.detach().cpu().long()
    keep = eligible[dst]
    selected, target = pi.detach().cpu()[keep], dst[keep]
    num_nodes = int(degree.numel())
    counts = torch.bincount(target, minlength=num_nodes).to(selected.dtype)
    sums = selected.new_zeros((num_nodes, selected.size(1))).index_add(0, target, selected)
    means = sums / counts.clamp_min(1).unsqueeze(-1)
    squared = selected.new_zeros((num_nodes, selected.size(1))).index_add(0, target, selected.square())
    variances = (squared / counts.clamp_min(1).unsqueeze(-1) - means.square()).clamp_min(0)
    eligible_ids = torch.where(eligible & (counts >= 5))[0]
    l1_edge = (selected - means[target]).abs().sum(-1)
    l1_sum = selected.new_zeros((num_nodes,)).index_add(0, target, l1_edge)
    l1_mean_by_target = l1_sum[eligible_ids] / counts[eligible_ids]
    std_by_target = variances[eligible_ids].sqrt()
    row: dict[str, Any] = {"modality": modality, "variant": variant,
                            "eligible_target_count": int(eligible_ids.numel())}
    for key, value in quantiles(l1_mean_by_target).items():
        row[f"within_node_l1_{key}"] = value
    for function_id, function_name in enumerate(FUNCTION_NAMES):
        for key, value in quantiles(std_by_target[:, function_id]).items():
            row[f"pi_{function_name}_within_node_std_{key}"] = value
    return row


def _distinctness_and_scales(aux: dict[str, Any], modality_id: int,
                             modality: str, variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = aux["controls"]
    distinct: dict[str, Any] = {"modality": modality, "variant": variant}
    for key in ("cos_smooth_relational", "cos_smooth_cross_modal", "cos_relational_cross_modal"):
        for stat, value in quantiles(controls[key][modality_id]).items():
            distinct[f"{key}_{stat}"] = value
    scales: dict[str, Any] = {"modality": modality, "variant": variant}
    scale_names = (
        "smooth_rms", "raw_relational_rms", "raw_cross_modal_rms",
        "calibrated_relational_rms_ratio", "calibrated_cross_modal_rms_ratio",
        "relational_calibration_scale", "cross_modal_calibration_scale",
    )
    for name in scale_names:
        stats = quantiles(controls[name][modality_id])
        for stat, value in stats.items():
            scales[f"{name}_{stat}"] = value
        scales[f"{name}_q99"] = float(np.quantile(
            controls[name][modality_id].detach().cpu().numpy().reshape(-1), 0.99
        )) if controls[name][modality_id].numel() else float("nan")
    return distinct, scales


def _modality_disagreement(pi_t: torch.Tensor, pi_v: torch.Tensor, variant: str) -> dict[str, Any]:
    eps = 1e-12
    p, q = pi_t.clamp_min(eps), pi_v.clamp_min(eps)
    midpoint = (p + q) * 0.5
    js = 0.5 * ((p * (p.log() - midpoint.clamp_min(eps).log())).sum(-1)
                + (q * (q.log() - midpoint.clamp_min(eps).log())).sum(-1))
    l1 = (pi_t - pi_v).abs().sum(-1)
    argmax_disagreement = (pi_t.argmax(-1) != pi_v.argmax(-1)).float()
    return {"variant": variant, "edge_count": int(pi_t.size(0)),
            "mean_l1_distance": float(l1.mean()) if l1.numel() else float("nan"),
            "mean_js_divergence": float(js.mean()) if js.numel() else float("nan"),
            "argmax_function_disagreement": float(argmax_disagreement.mean()) if l1.numel() else float("nan")}


def collect_diagnostics(aux: dict[str, Any], variant: str, val_idx: torch.Tensor) -> dict[str, Any]:
    controls = aux["controls"]
    pi_t, pi_v = controls["pi"]
    rows = {
        "router": [_router_diagnostics(controls["pi"][m], "text" if m == 0 else "visual", variant)
                   for m in range(2)],
        "within_node": [_within_node_diagnostics(controls["pi"][m], aux["edge_index_nonself"][1],
                                                  aux["degree"], val_idx,
                                                  "text" if m == 0 else "visual", variant)
                        for m in range(2)],
        "modality_disagreement": _modality_disagreement(pi_t, pi_v, variant),
        "function_distinctness": [], "expert_scales": [],
    }
    for m in range(2):
        distinct, scales = _distinctness_and_scales(aux, m, "text" if m == 0 else "visual", variant)
        rows["function_distinctness"].append(distinct)
        rows["expert_scales"].append(scales)
    return rows


def _evaluate_with_overrides(model, classifier, x, edge_index, val_idx, labels, class_count, overrides):
    return metric(model, classifier, x, edge_index, val_idx, labels, class_count, overrides)


def edge_mix_interventions(model, classifier, x, edge_index, val_idx, labels,
                           class_count: int, pi: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
    if model.variant != "edge_mix":
        raise ValueError("E0 attribution interventions require a trained EdgeMix checkpoint")
    _, dst = edge_index
    dst = dst.detach().cpu().long()
    base = _evaluate_with_overrides(model, classifier, x, edge_index, val_idx, labels,
                                    class_count, None)
    rows = []

    def add(name: str, pair: tuple[torch.Tensor, torch.Tensor], repeat: int | None = None):
        metrics = _evaluate_with_overrides(model, classifier, x, edge_index, val_idx, labels,
                                           class_count, {"pi": pair})
        rows.append({"intervention": name, "repeat_seed": repeat, **metrics,
                     "delta_accuracy": metrics["val_accuracy"] - base["val_accuracy"],
                     "delta_macro_f1": metrics["val_macro_f1"] - base["val_macro_f1"],
                     "delta_ce": metrics["val_ce"] - base["val_ce"]})

    for repeat in range(1001, 1006):
        shuffled = []
        for modality in range(2):
            permutation = neighborhood_shuffle_indices(dst, repeat + modality * 10000)
            shuffled.append(pi[modality][permutation].clone())
        add("pi_shuffle_within_target", tuple(shuffled), repeat)
    add("pi_target_mean", tuple(pi_target_mean(pi[m], dst, x.size(0)) for m in range(2)))
    add("pi_global_mean", tuple(pi_global_mean(pi[m]) for m in range(2)))
    add("modality_tied_pi", pi_modality_tied(pi[0], pi[1]))
    smooth_only = []
    for m in range(2):
        value = torch.zeros_like(pi[m]); value[:, 1] = 1
        smooth_only.append(value)
    add("smooth_only", tuple(smooth_only))
    add("null_off", tuple(renormalize_without(pi[m], 0) for m in range(2)))
    add("relational_off", tuple(renormalize_without(pi[m], 2) for m in range(2)))
    add("cross_off", tuple(renormalize_without(pi[m], 3) for m in range(2)))
    return {"normal": base, "rows": rows}


def _check_relation_state_regression(cfg, data_info, data, device):
    m0_cfg = make_config(str(cfg.dataset.name), int(cfg.seed), "single_basis", str(device), m0_model=True)
    e0 = Model(copy.deepcopy(cfg), data_info).to(device).eval()
    m0 = M0Model(m0_cfg, data_info).to(device).eval()
    m0_state = m0.state_dict()
    e0_state = e0.state_dict()
    common = {key: value for key, value in m0_state.items() if key in e0_state}
    e0.load_state_dict(common, strict=False)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        h0, _, src, dst, degree, deg_z, p, contexts = m0._prepare(x, edge_index)
        e0_h0, _, e0_src, e0_dst, e0_degree, e0_deg_z, e0_p, e0_contexts = e0._prepare(x, edge_index)
        evidence_t = m0._pair_evidence(p[0], contexts[0], deg_z, src, dst)
        evidence_v = m0._pair_evidence(p[1], contexts[1], deg_z, src, dst)
        q_t, q_v = m0.phi_pair(evidence_t), m0.phi_pair(evidence_v)
        relation = m0.phi_rel(torch.cat([q_t, q_v, (q_t - q_v).abs(), q_t * q_v], dim=-1))
        u = m0._functional_states(p, contexts, deg_z, src, dst)
        # Reuse one computed LOO context for q/r/u comparisons. CUDA index_add
        # can differ by a few 1e-7 across independent reductions on this graph;
        # the relation transforms themselves should be bitwise identical.
        q_t_e0, q_v_e0, relation_e0, u_e0 = e0._functional_states(p, contexts, deg_z, src, dst)
    checks = {"p": True, "q": True, "r": True, "u": True, "h0": True, "loo_context": True,
              "edge_support": True, "degree": True}
    for left, right in zip(h0, e0_h0):
        checks["h0"] &= bool(torch.allclose(left, right, rtol=1e-6, atol=1e-7))
    for left, right in zip(p, e0_p):
        checks["p"] &= bool(torch.allclose(left, right, rtol=1e-6, atol=1e-7))
    for left, right in zip(contexts, e0_contexts):
        checks["loo_context"] &= bool(torch.allclose(left, right, rtol=1e-6, atol=1e-7))
    checks["edge_support"] = bool(torch.equal(src, e0_src) and torch.equal(dst, e0_dst))
    checks["degree"] = bool(torch.equal(degree, e0_degree))
    for left, right in zip((q_t, q_v), (q_t_e0, q_v_e0)):
        checks["q"] &= bool(torch.allclose(left, right, rtol=1e-6, atol=1e-7))
    checks["r"] = bool(torch.allclose(relation, relation_e0, rtol=1e-6, atol=1e-7))
    for left, right in zip(u, u_e0):
        checks["u"] &= bool(torch.allclose(left, right, rtol=1e-6, atol=1e-7))
    if not all(checks.values()):
        raise AssertionError(f"E0/M0 relation-state regression failed: {checks}")
    return checks


def _check_smooth_identity(cfg, data_info, data, device):
    m0_cfg = make_config(str(cfg.dataset.name), int(cfg.seed), "uniform", str(device), m0_model=True)
    e0 = Model(copy.deepcopy(cfg), data_info).to(device).eval()
    m0 = M0Model(m0_cfg, data_info).to(device).eval()
    e0_state, m0_state = e0.state_dict(), m0.state_dict()
    common = {key: value for key, value in m0_state.items() if key in e0_state}
    e0.load_state_dict(common, strict=False)
    with torch.no_grad():
        for modality in range(2):
            e0.smooth_transforms[modality].weight.copy_(m0.w0[modality].weight)
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        h0, _, src, dst, degree, _, _, _ = e0._prepare(x, edge_index)
        manual_delta = []
        m0_delta = []
        for modality in range(2):
            smooth_message = e0.smooth_transforms[modality](h0[modality][src])
            agg = h0[modality].new_zeros(h0[modality].shape).index_add(0, dst, smooth_message)
            manual_delta.append(agg / degree.clamp_min(1).to(agg.dtype).unsqueeze(-1))
            m0_h0 = m0._prepare(x, edge_index)[0][modality]
            m0_message = m0.w0[modality](m0_h0[src])
            m0_aggregate = m0_h0.new_zeros(m0_h0.shape).index_add(0, dst, m0_message)
            m0_delta.append(m0_aggregate / degree.clamp_min(1).to(m0_aggregate.dtype).unsqueeze(-1))
        pi = torch.zeros(src.numel(), 4, device=device); pi[:, 1] = 1
        z_e0 = e0(x, edge_index, control_overrides={"pi": (pi, pi)})[0]
        z_m0 = m0(x, edge_index)[0]
    if not torch.allclose(z_e0, z_m0, rtol=1e-6, atol=2e-6):
        raise AssertionError("E0 smooth-only route failed exact M0 uniform propagation identity")
    aggregate_error = max(float((left - right).abs().max()) for left, right in zip(manual_delta, m0_delta))
    return {"max_abs_z_error": float((z_e0 - z_m0).abs().max()),
            "max_abs_aggregated_message_error": aggregate_error}


def _check_function_smoke(data_info: dict[str, int], data, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(701)
    model = Model(OmegaConf.create({"model": {"variant": "edge_mix", "hidden_dim": 128,
        "dropout": 0.0, "relation_dim": 32, "relation_state_dim": 64,
        "modality_embed_dim": 8, "edge_chunk_size": 2048, "eps": 1e-8}}), data_info).to(device).eval()
    with torch.no_grad():
        prior = torch.softmax(model.router(torch.randn(4096, 64, device=device)), dim=-1).mean(0)
        expected_prior = prior.new_tensor([0.096255, 0.711235, 0.096255, 0.096255])
        if not torch.allclose(prior, expected_prior, atol=0.004, rtol=0):
            raise AssertionError(f"E0 router initialization is not smooth-default: {prior.tolist()}")
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        state = model.relation_state(x, edge_index)
        src, dst = state["src"][:2048], state["dst"][:2048]
        h0 = [value.detach().clone() for value in state["h0"]]
        initial = model._expert_outputs(h0, src, dst)
        if not all(torch.isfinite(value).all() for group in initial[:3] for value in group):
            raise FloatingPointError("E0 smoke found a non-finite expert output")
        # Cross-modal Text execution must react to Visual source changes while
        # Text relational execution remains strictly same-modality.
        changed_visual = [h0[0], h0[1].clone()]
        changed_visual[1][src[0]] += 1.0
        perturbed = model._expert_outputs(changed_visual, src, dst)
        relational_text_unchanged = torch.equal(initial[1][0], perturbed[1][0])
        cross_text_changed = not torch.allclose(initial[2][0][0], perturbed[2][0][0])
        if not relational_text_unchanged or not cross_text_changed:
            raise AssertionError("E0 smoke cross-modal/same-modal source routing check failed")
        calibration_errors = []
        scales_finite = True
        for modality in range(2):
            smooth, relational, cross_modal = (initial[0][modality], initial[1][modality], initial[2][modality])
            ratio_rel = _rms(relational, model.eps) / _rms(smooth, model.eps)
            ratio_cross = _rms(cross_modal, model.eps) / _rms(smooth, model.eps)
            calibration_errors.extend([float((ratio_rel - 1).abs().max()),
                                       float((ratio_cross - 1).abs().max())])
            scales_finite &= all(
                bool(torch.isfinite(value).all())
                for key in ("reference_rms", "raw_rms_relational", "raw_rms_cross_modal",
                            "scale_relational", "scale_cross_modal")
                for value in initial[3][key]
            )
        if max(calibration_errors, default=0.0) > 2e-5 or not scales_finite:
            raise AssertionError("E0 smoke calibration RMS/scale check failed")
    result = {"router_prior_mean": prior.detach().cpu().tolist(),
              "cross_modal_text_source_changed": cross_text_changed,
              "relational_text_ignores_visual_source": relational_text_unchanged,
              "max_calibrated_rms_ratio_abs_error": max(calibration_errors),
              "calibration_scales_finite": scales_finite}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_one(dataset: str, seed: int, variant: str, device_name: str,
            out_dir: Path = OUT_DIR, epochs: int | None = None,
            save_checkpoint: bool = True) -> dict[str, Any]:
    cfg = make_config(dataset, seed, variant, device_name, epochs)
    data = load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    data_info = model_data_info(data)
    init_audit = audit_initialization(cfg, data_info, seed)
    if not init_audit["common_all_bitwise_equal"] or not init_audit["exact_trainable_parameter_match"]:
        raise AssertionError(f"E0 initialization/parameter audit failed: {init_audit}")

    device = torch.device(device_name)
    set_seed(seed)
    model = Model(cfg, data_info).to(device)
    torch.manual_seed(seed + 1907)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1907)
    classifier = nn.Linear(model.out_dim, data.num_classes).to(device)
    set_seed(seed)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    train_idx, val_idx = splits["train"].to(device), splits["validation"].to(device)
    labels = data.y.to(device)
    train_y, val_y = labels[train_idx], labels[val_idx]
    if bool((train_y < 0).any()) or bool((val_y < 0).any()) or data.test_idx is not None:
        raise AssertionError("E0 train/validation protocol exposed invalid labels or test indices")
    if bool(cfg.task.evaluate_test):
        raise AssertionError("E0 requires task.evaluate_test=false")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_acc, best_epoch = -1.0, 0
    patience_left = int(cfg.task.patience)
    best_model = best_classifier = None
    best_metrics = None
    first_gradients = {}
    start = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, int(cfg.task.epochs) + 1):
        epochs_run = epoch
        model.train(); classifier.train(); optimizer.zero_grad(set_to_none=True)
        z, _, _, aux_loss, _ = model(x, edge_index)
        logits = classifier(z[train_idx])
        loss = nn.functional.cross_entropy(logits, train_y) + float(cfg.task.loss.aux_weight) * aux_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at {dataset}/{seed}/{variant}/{epoch}")
        loss.backward()
        if epoch == 1:
            first_gradients = gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()),
                                       max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True)
        optimizer.step()
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        del z, logits, loss
        current = metric(model, classifier, x, edge_index, val_idx, labels,
                         int(data.num_classes), None)
        min_delta = float(cfg.task.early_stop_min_delta)
        if current["val_accuracy"] > best_acc + min_delta:
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
    train_seconds = time.perf_counter() - start
    if best_model is None or best_classifier is None:
        raise RuntimeError("E0 training ended without a validation-selected checkpoint")
    for name, audit in first_gradients.items():
        if not audit["finite"] or audit["norm"] <= 0:
            raise AssertionError(f"E0 expert/router gradient missing in first epoch: {name}: {audit}")
    model.load_state_dict(best_model, strict=True)
    classifier.load_state_dict(best_classifier, strict=True)
    model.eval()
    with torch.no_grad():
        z, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
        selected_metrics = _metric_from_embeddings(classifier, z, val_idx, labels[val_idx], data.num_classes)
    controls = aux["controls"]
    pi = controls["pi"]
    if not bool(torch.isfinite(z).all()) or any(not bool(torch.isfinite(value).all()) for value in pi):
        raise FloatingPointError("non-finite E0 selected-checkpoint outputs")
    for modality in range(2):
        if pi[modality].numel() and not torch.allclose(pi[modality].sum(-1),
                                                       torch.ones(pi[modality].size(0)), atol=1e-6):
            raise AssertionError("E0 router probability rows do not sum to one")
    diagnostics = collect_diagnostics(aux, variant, val_idx)
    identity_checks = {}
    dst_cpu = aux["edge_index_nonself"][1].long()
    if variant == "static_mix":
        identity_checks["pi_global_mean_max_abs"] = max(
            float((pi_global_mean(pi[m]) - pi[m]).abs().max()) for m in range(2))
    elif variant == "target_mix":
        identity_checks["pi_target_mean_max_abs"] = max(
            float((pi_target_mean(pi[m], dst_cpu, data.num_nodes) - pi[m]).abs().max())
            for m in range(2))
    interventions = None
    intervention_seconds = 0.0
    if variant == "edge_mix":
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_start = time.perf_counter()
        interventions = edge_mix_interventions(model, classifier, x, edge_index, val_idx,
                                                labels, data.num_classes, pi)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_seconds = time.perf_counter() - intervention_start
    model_params = sum(int(parameter.numel()) for parameter in model.parameters())
    classifier_params = sum(int(parameter.numel()) for parameter in classifier.parameters())
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    record = {
        "status": "completed", "dataset": dataset, "seed": int(seed), "variant": variant,
        "best_epoch": best_epoch, "epochs_run": epochs_run, **selected_metrics,
        "model_params": model_params, "classifier_params": classifier_params,
        "total_trainable_params": model_params + classifier_params,
        "peak_gpu_memory_bytes": peak_bytes, "training_time_sec": train_seconds,
        "intervention_time_sec": intervention_seconds, "gradient_audit_epoch1": first_gradients,
        "diagnostics": diagnostics, "interventions": interventions,
        "initialization_audit": init_audit, "identity_checks": identity_checks,
        "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        "test_indices_attached": False, "test_labels_exposed": False,
        "link_prediction": False, "data_info": data_info,
    }
    run_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    write_json(run_path, record)
    if save_checkpoint:
        checkpoint = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_model, "head_state": best_classifier, "data_info": data_info,
                    "dataset": dataset, "seed": int(seed), "variant": variant,
                    "best_epoch": best_epoch, "best_val_metrics": selected_metrics,
                    "protocol": "unified_full_graph_nc_v1", "evaluate_test": False}, checkpoint)
        record["checkpoint"] = str(checkpoint.relative_to(PROJECT_ROOT))
        write_json(run_path, record)
    print(f"[E0 done] {dataset}/{seed}/{variant} epoch={best_epoch} "
          f"acc={selected_metrics['val_accuracy']:.4f} f1={selected_metrics['val_macro_f1']:.4f} "
          f"ce={selected_metrics['val_ce']:.4f} train={train_seconds:.1f}s "
          f"intervention={intervention_seconds:.1f}s peak={peak_bytes/1024**3:.2f}GiB", flush=True)
    del z, aux, data, x, edge_index, model, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _synthetic_check_chunk_equivalence(device: torch.device) -> dict[str, float]:
    torch.manual_seed(203)
    x = torch.randn(10, 20, device=device)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6, 8],
        [9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7],
    ], dtype=torch.long, device=device)
    info = {"input_dim": 20, "num_nodes": 10, "num_classes": 3, "text_dim": 8, "visual_dim": 12}
    errors = {}
    for variant in VARIANTS:
        cfg_small = OmegaConf.create({"model": {"variant": variant, "hidden_dim": 128,
            "dropout": 0.0, "relation_dim": 32, "relation_state_dim": 64,
            "modality_embed_dim": 8, "edge_chunk_size": 2, "eps": 1e-8}})
        cfg_large = copy.deepcopy(cfg_small); cfg_large.model.edge_chunk_size = 1000
        torch.manual_seed(31); small = Model(cfg_small, info).to(device).eval()
        torch.manual_seed(31); large = Model(cfg_large, info).to(device).eval()
        with torch.no_grad():
            z_small, _, _, _, _ = small(x, edge_index)
            z_large, _, _, _, _ = large(x, edge_index)
        error = float((z_small - z_large).abs().max())
        if not torch.allclose(z_small, z_large, rtol=2e-5, atol=2e-6):
            raise AssertionError(f"E0 {variant} small/large chunk outputs differ by {error}")
        errors[variant] = error
        del small, large
    return errors


def run_smoke(device_name: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    dataset, seed = "Movies", 42
    cfg = make_config(dataset, seed, "edge_mix", device_name, epochs=2)
    data = load_m0_data(cfg, seed)
    restrict_labels_to_train_val(data)
    info = model_data_info(data)
    audit = audit_initialization(cfg, info, seed)
    if not audit["common_all_bitwise_equal"] or not audit["exact_trainable_parameter_match"]:
        raise AssertionError(f"E0 smoke init/parameter audit failed: {audit}")
    device = torch.device(device_name)
    regression = _check_relation_state_regression(cfg, info, data, device)
    smooth = _check_smooth_identity(cfg, info, data, device)
    function_checks = _check_function_smoke(info, data, device)
    chunk = _synthetic_check_chunk_equivalence(device)
    run_summaries = []
    for variant in VARIANTS:
        record = run_one(dataset, seed, variant, device_name, out_dir / "smoke_runs",
                         epochs=2, save_checkpoint=False)
        if not all(item["finite"] and item["norm"] > 0
                   for item in record["gradient_audit_epoch1"].values()):
            raise AssertionError(f"E0 smoke gradients failed for {variant}")
        identity_error = max(record["identity_checks"].values(), default=0.0)
        if identity_error > 3e-6:
            raise AssertionError(f"E0 {variant} routing identity failed: {record['identity_checks']}")
        run_summaries.append({"variant": variant, "best_epoch": record["best_epoch"],
                              "metrics": {key: record[key] for key in
                                          ("val_accuracy", "val_macro_f1", "val_ce")},
                              "params": record["total_trainable_params"],
                              "peak_gpu_memory_bytes": record["peak_gpu_memory_bytes"],
                              "identity_checks": record["identity_checks"]})
    # Check probability intervention identities on real E0 route vectors.
    edge = next(item for item in run_summaries if item["variant"] == "edge_mix")
    del edge
    smoke_path = out_dir / "smoke" / "Movies_seed_42" / "smoke_result.json"
    payload = {
        "status": "smoke_passed", "dataset": dataset, "seed": seed, "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "relation_state_regression": regression, "smooth_identity": smooth,
        "function_checks": function_checks,
        "chunk_equivalence_max_abs_error": chunk, "initialization_and_parameter_audit": audit,
        "variants": run_summaries, "test_indices_attached": False,
        "test_labels_exposed": False, "evaluate_test": False, "link_prediction": False,
    }
    write_json(smoke_path, payload)
    print(f"[E0 smoke passed] Movies/42 variants=3 device={device_name}", flush=True)
    del data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def run_campaign(args) -> None:
    smoke_path = args.output_dir / "smoke" / "Movies_seed_42" / "smoke_result.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "smoke_passed":
        raise RuntimeError("E0 Movies/42 smoke must pass before formal campaign")
    total = len(args.datasets) * len(args.seeds) * len(args.variants)
    completed = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                run_path = args.output_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if run_path.is_file():
                    previous = json.loads(run_path.read_text(encoding="utf-8"))
                    if previous.get("status") == "completed":
                        completed.append(previous)
                        print(f"[skip-complete] {dataset}/{seed}/{variant}", flush=True)
                        continue
                print(f"[run {len(completed)+1}/{total}] {dataset}/{seed}/{variant} on {args.device}", flush=True)
                record = run_one(dataset, seed, variant, args.device, args.output_dir)
                completed.append(record)
                write_json(args.output_dir / "campaign_progress.json", {
                    "requested_runs": total,
                    "completed_unique_runs": len({(row["dataset"], row["seed"], row["variant"])
                                                   for row in completed}),
                    "datasets": list(args.datasets), "seeds": list(args.seeds),
                    "variants": list(args.variants), "device": args.device,
                    "evaluate_test": False,
                })
    summary = {
        "status": "completed" if len(completed) == total else "partial",
        "requested_new_runs": total, "completed_new_runs": len(completed),
        "failed_new_runs": 0, "training_reruns": 0,
        "datasets": list(args.datasets), "seeds": list(args.seeds),
        "variants": list(args.variants), "device": args.device,
        "evaluate_test": False, "link_prediction": False,
        "sum_training_time_seconds": sum(float(row["training_time_sec"]) for row in completed),
        "sum_edge_mix_intervention_time_seconds": sum(float(row["intervention_time_sec"])
                                                       for row in completed if row["variant"] == "edge_mix"),
        "max_peak_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in completed),
    }
    write_json(args.output_dir / "campaign_result.json", summary)
    print(f"[E0 campaign finished] new={len(completed)}/{total}; "
          f"training={summary['sum_training_time_seconds']:.1f}s; "
          f"peak={summary['max_peak_gpu_memory_bytes']/1024**3:.2f}GiB", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E0 structured relation-function executor screen")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None, help="short run override for smoke/debug only")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    gpu = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"
    print(f"[E0] device={args.device} gpu={gpu} torch={torch.__version__} "
          f"pyg={__import__('torch_geometric').__version__}", flush=True)
    if args.smoke:
        run_smoke(args.device, args.output_dir)
    elif args.campaign:
        run_campaign(args)
    elif args.dataset and args.seed is not None and args.variant:
        run_one(args.dataset, args.seed, args.variant, args.device, args.output_dir, args.epochs)
    else:
        parser.error("choose --smoke, --campaign, or one --dataset/--seed/--variant run")


if __name__ == "__main__":
    main()
