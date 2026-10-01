from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m0_adaptive_propagation import load_m0_data, restrict_labels_to_train_val
from src.models.adaptive_prop_m0 import Model as M0Model, fixed_degree_mean
from src.models.adaptive_prop_n1 import (
    MASTER_DIM,
    VARIANTS,
    Model,
    common_parameter_names,
    copy_m0_uniform_common_weights,
    parameter_counts,
)
from src.tasks.common import build_optimizer, scheduler_step
from src.utils.seeds import set_seed

DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS_ORDERED = ("smooth_only", "static_strength", "same_state_strength", "cross_state_strength")
SOURCE_BRANCH = "exp/n0_recipient_state_function_context"
SOURCE_SHA = "98081ab965b1bb832218659862a0f3e3ba288dea"
OUT_DIR = PROJECT_ROOT / "outputs" / "n1_recipient_function_strength_mixer"
RESEARCH_DIR = PROJECT_ROOT / "research" / "n1_recipient_function_strength_mixer"
SHUFFLE_SEEDS = (3101, 3102, 3103, 3104, 3105)
STAT_KEYS = ("mean", "std", "median", "q10", "q25", "q75", "q90", "q99")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_config(dataset: str, seed: int, variant: str, device: str, epochs: int | None = None):
    overrides = [
        f"dataset={dataset}", "task=nc", "model=adaptive_prop_n1",
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


def build_classifier(out_dim: int, num_classes: int, seed: int,
                     device: torch.device | str = "cpu") -> nn.Linear:
    """Initialize the NC classifier independently from added model modules."""
    torch.manual_seed(int(seed) + 1907)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + 1907)
    return nn.Linear(int(out_dim), int(num_classes)).to(resolved_device)


def audit_initialization(dataset: str, seed: int, data_info: dict[str, int], cfg) -> dict[str, Any]:
    states, classifiers, hashes, classifier_hashes, counts = {}, {}, {}, {}, {}
    for variant in VARIANTS_ORDERED:
        set_seed(seed)
        local = copy.deepcopy(cfg)
        local.model.variant = variant
        model = Model(local, data_info)
        states[variant] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        hashes[variant] = tensor_hash(states[variant])
        count = parameter_counts(model)
        classifier = build_classifier(model.out_dim, data_info["num_classes"], seed)
        classifiers[variant] = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
        classifier_hashes[variant] = tensor_hash(classifiers[variant])
        counts[variant] = {
            **count,
            "classifier_params": sum(int(value.numel()) for value in classifier.parameters()),
            "total_trainable_params": count["model_total"] + sum(
                int(value.numel()) for value in classifier.parameters() if value.requires_grad
            ),
        }
        del model, classifier
    pairwise = {}
    for i, left in enumerate(VARIANTS_ORDERED):
        for right in VARIANTS_ORDERED[i + 1:]:
            pairwise[f"{left}__{right}"] = (
                states[left].keys() == states[right].keys()
                and all(torch.equal(states[left][name], states[right][name]) for name in states[left])
                and classifiers[left].keys() == classifiers[right].keys()
                and all(torch.equal(classifiers[left][name], classifiers[right][name])
                        for name in classifiers[left])
            )
    exact_params = len({row["total_trainable_params"] for row in counts.values()}) == 1
    exact_classifier = len({row["classifier_params"] for row in counts.values()}) == 1
    audit = {
        "dataset": dataset, "seed": int(seed), "variant_hashes": hashes,
        "classifier_hashes": classifier_hashes, "pairwise_bitwise_equal": pairwise,
        "all_bitwise_equal": all(pairwise.values()), "exact_parameter_match": exact_params,
        "exact_classifier_parameter_match": exact_classifier, "parameter_counts": counts,
    }
    if not audit["all_bitwise_equal"] or not exact_params or not exact_classifier:
        raise AssertionError(f"N1 variants failed initialization fairness: {audit}")
    return audit


def initialization_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, counts in audit["parameter_counts"].items():
        rows.append({
            "dataset": audit["dataset"], "seed": audit["seed"], "variant": variant,
            "model_hash": audit["variant_hashes"][variant],
            "classifier_hash": audit["classifier_hashes"][variant],
            "model_params": counts["model_total"], "classifier_params": counts["classifier_params"],
            "total_params": counts["total_trainable_params"],
            "all_variants_bitwise_equal": audit["all_bitwise_equal"],
            "exact_parameter_match": audit["exact_parameter_match"],
        })
    return rows


def quantiles(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: float("nan") for key in STAT_KEYS}
    q10, q25, median, q75, q90, q99 = np.quantile(array, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=0)),
        "median": float(median), "q10": float(q10), "q25": float(q25),
        "q75": float(q75), "q90": float(q90), "q99": float(q99),
    }


def safe_spearman(left: torch.Tensor | np.ndarray, right: torch.Tensor | np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.ptp(a[mask]) == 0 or np.ptp(b[mask]) == 0:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def metric_from_embeddings(classifier, z, val_idx, val_y, num_classes: int) -> dict[str, float]:
    logits = classifier(z[val_idx])
    ce = nn.functional.cross_entropy(logits, val_y)
    pred = logits.argmax(-1)
    return {
        "val_accuracy": float((pred == val_y).float().mean().item()),
        "val_macro_f1": float(f1_score(
            val_y.detach().cpu().numpy(), pred.detach().cpu().numpy(),
            labels=list(range(num_classes)), average="macro", zero_division=0,
        )),
        "val_ce": float(ce.item()),
    }


@torch.no_grad()
def evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes: int,
             beta_overrides=None) -> dict[str, float]:
    model.eval(); classifier.eval()
    z = model(x, edge_index, beta_overrides=beta_overrides)[0]
    return metric_from_embeddings(classifier, z, val_idx, val_y, num_classes)


def gradient_audit(model: nn.Module) -> dict[str, dict[str, Any]]:
    prefixes = {
        "projectors": ("proj_t.", "proj_v."), "smooth": ("w_s.",),
        "absdiff": ("w_d.",), "product": ("w_p.",),
        "mixers": ("strength_mixers.",), "fusion": ("fusion.", "residual_norms."),
    }
    output = {}
    for group, names in prefixes.items():
        params = [p for name, p in model.named_parameters() if name.startswith(names)]
        grads = [p.grad.detach() for p in params if p.grad is not None]
        output[group] = {
            "parameter_tensors": len(params), "gradient_tensors": len(grads),
            "finite": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
            "norm": float(torch.sqrt(sum(g.float().square().sum() for g in grads)).item()) if grads else 0.0,
        }
    return output


def validate_smoke_gradients(variant: str, audit: dict[str, Any]) -> None:
    always_active = ("projectors", "smooth", "fusion")
    for group in always_active:
        row = audit[group]
        if not row["finite"] or row["norm"] <= 0:
            raise AssertionError(f"{variant}: expected active finite {group} gradients, got {row}")
    for group in ("absdiff", "product"):
        row = audit[group]
        if variant == "smooth_only":
            if row["gradient_tensors"] != 0:
                raise AssertionError(f"smooth_only should leave {group} without gradients: {row}")
        elif not row["finite"] or row["norm"] <= 0:
            raise AssertionError(f"{variant}: expected active finite {group} gradients, got {row}")
    mixer = audit["mixers"]
    if variant == "smooth_only":
        if mixer["gradient_tensors"] != 0:
            raise AssertionError("smooth_only mixer should not contribute to the update")
    elif not mixer["finite"] or mixer["gradient_tensors"] != mixer["parameter_tensors"]:
        raise AssertionError(f"{variant}: mixer gradients missing or non-finite: {mixer}")


def smooth_m0_regression(dataset: str, seed: int, cfg_n1, data, device: torch.device) -> dict[str, Any]:
    cfg_m0 = make_config(dataset, seed, "uniform", str(device))
    info = model_data_info(data)
    torch.manual_seed(seed)
    m0 = M0Model(cfg_m0, info).to(device)
    local = copy.deepcopy(cfg_n1)
    local.model.variant = "smooth_only"
    torch.manual_seed(seed)
    n1 = Model(local, info).to(device)
    copy_m0_uniform_common_weights(m0, n1)
    m0.eval(); n1.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        m0_z = m0(x, edge_index)[0]
        n1_z, _, _, _, aux = n1(x, edge_index, return_diagnostics=True)
        m0_h0, _, src, dst, degree, *_ = m0._prepare(x, edge_index)
        m0_contexts = []
        m0_tilde = []
        for modality in range(2):
            aggregate = m0_h0[modality].new_zeros(m0_h0[modality].shape)
            for begin in range(0, src.numel(), m0.edge_chunk_size):
                end = min(begin + m0.edge_chunk_size, src.numel())
                aggregate = aggregate.index_add(
                    0, dst[begin:end], m0.w0[modality](m0_h0[modality][src[begin:end]])
                )
            context = fixed_degree_mean(aggregate, degree)
            m0_contexts.append(context)
            m0_tilde.append(m0.residual_norms[modality](m0_h0[modality] + context))
    errors = {"z_max_abs": float((m0_z - n1_z).abs().max())}
    for modality, name in enumerate(("text", "visual")):
        errors[f"h0_{name}_max_abs"] = float((m0_h0[modality] - aux["h0"][modality]).abs().max())
        errors[f"C_S_{name}_max_abs"] = float((m0_contexts[modality] - aux["contexts"][modality][0]).abs().max())
        errors[f"h_tilde_{name}_max_abs"] = float((m0_tilde[modality] - aux["h_tilde"][modality]).abs().max())
    # CUDA index_add uses nondeterministic atomic accumulation. Common states,
    # formulas and reduction order match; the measured sub-2e-6 discrepancy is
    # covered by the prior M0 CUDA regression budget of 1e-5 absolute error.
    passed = torch.allclose(m0_z, n1_z, rtol=1e-6, atol=1e-5) and max(errors.values()) <= 1e-5
    if not passed:
        raise AssertionError(f"N1 SmoothOnly does not regress to M0 UNI: {errors}")
    return {"dataset": dataset, "seed": int(seed), "passed": True,
            "rtol": 1e-6, "atol": 1e-5, "tolerance_basis": "CUDA index_add atomic accumulation",
            "max_abs_errors": errors,
            "master_dim": aux["master_dim"]}


def _distribution_row(dataset: str, seed: int, variant: str, modality: str,
                      group: str, values: torch.Tensor, scope: str) -> dict[str, Any]:
    return {"dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
            "group": group, "scope": scope, **quantiles(values)}


def collect_diagnostics(dataset: str, seed: int, variant: str, aux: dict[str, Any],
                        val_idx: torch.Tensor) -> dict[str, list[dict[str, Any]]]:
    val_cpu = val_idx.detach().cpu().long()
    degree = aux["degree"].detach().cpu()
    nonisolated = degree > 0
    val_mask = torch.zeros_like(degree, dtype=torch.bool)
    val_mask[val_cpu] = True
    contexts = tuple(tuple(value.detach().cpu() for value in pair) for pair in aux["contexts"])
    betas = tuple(value.detach().cpu() for value in aux["betas"])
    strength, variation, contribution, rms_rows, distinct = [], [], [], [], []
    beta_rms = []
    for modality, modality_name in enumerate(("text", "visual")):
        for scope, mask in (("nonisolated_all", nonisolated), ("validation_targets", val_mask)):
            selected = mask
            for column, function in enumerate(("D", "P")):
                values = betas[modality][selected, column]
                strength.append(_distribution_row(dataset, seed, variant, modality_name,
                                                  f"beta_{function}", values, scope))
        val_values = betas[modality][val_mask]
        for column, function in enumerate(("D", "P")):
            values = val_values[:, column]
            q = quantiles(values)
            mean = q["mean"]
            variation.append({
                "dataset": dataset, "seed": seed, "variant": variant,
                "modality": modality_name, "function": function,
                "validation_std_i": q["std"], "validation_iqr": q["q75"] - q["q25"],
                "validation_cv": q["std"] / abs(mean) if abs(mean) > 1e-12 else float("nan"),
                "validation_mean": mean, "n_validation_targets": int(values.numel()),
            })
        channels = contexts[modality]
        rms = [channel.square().mean(dim=-1).sqrt() for channel in channels]
        for function, value in zip(("S", "D", "P"), rms):
            rms_rows.append(_distribution_row(dataset, seed, variant, modality_name,
                                              f"RMS_C_{function}", value[val_mask], "validation_targets"))
        for channel, function in ((1, "D"), (2, "P")):
            ratio = rms[channel] / (rms[0] + 1e-12)
            rms_rows.append(_distribution_row(dataset, seed, variant, modality_name,
                                              f"RMS_ratio_{function}_over_S", ratio[val_mask],
                                              "validation_targets"))
        eps = 1e-12
        for beta_column, function, channel in ((0, "D", 1), (1, "P", 2)):
            ratio = betas[modality][:, beta_column] * rms[channel] / (rms[0] + eps)
            contribution.append(_distribution_row(dataset, seed, variant, modality_name,
                                                  f"effective_A_{function}", ratio[val_mask],
                                                  "validation_targets"))
        pairs = ((0, 1, "S_vs_D"), (0, 2, "S_vs_P"), (1, 2, "D_vs_P"))
        for i, j, pair_name in pairs:
            cosine = nn.functional.cosine_similarity(channels[i], channels[j], dim=-1, eps=1e-12)
            distinct.append(_distribution_row(dataset, seed, variant, modality_name,
                                              pair_name, cosine[val_mask], "validation_targets"))
        beta_rms.append(rms)
    disagreement = []
    if variant in {"same_state_strength", "cross_state_strength"}:
        val_t, val_v = betas[0][val_mask], betas[1][val_mask]
        for column, function in enumerate(("D", "P")):
            disagreement.append({
                "dataset": dataset, "seed": seed, "variant": variant,
                "measure": f"{function}_strength_TV_spearman",
                "value": safe_spearman(val_t[:, column], val_v[:, column]),
                "n_validation_targets": int(val_t.size(0)),
            })
        l1 = (val_t - val_v).abs().sum(-1)
        cosine = nn.functional.cosine_similarity(val_t, val_v, dim=-1, eps=1e-12)
        disagreement.extend([
            {"dataset": dataset, "seed": seed, "variant": variant, "measure": "recipient_strength_L1", "value": float(l1.mean())},
            {"dataset": dataset, "seed": seed, "variant": variant, "measure": "recipient_strength_cosine", "value": float(cosine.mean())},
        ])
    return {
        "strength": strength, "variation": variation, "contribution": contribution,
        "channel_rms": rms_rows, "distinctness": distinct,
        "disagreement": disagreement,
    }


def _intervention_row(dataset, seed, name, metrics, base, repeat_seed=None):
    return {
        "dataset": dataset, "seed": int(seed), "variant": "cross_state_strength",
        "intervention": name, "repeat_seed": repeat_seed,
        "val_accuracy": metrics["val_accuracy"], "val_macro_f1": metrics["val_macro_f1"],
        "val_ce": metrics["val_ce"],
        "delta_accuracy": metrics["val_accuracy"] - base["val_accuracy"],
        "delta_macro_f1": metrics["val_macro_f1"] - base["val_macro_f1"],
        "delta_ce": metrics["val_ce"] - base["val_ce"],
    }


def beta_off(betas, columns: tuple[int, ...]):
    modified = [value.clone() for value in betas]
    for value in modified:
        for column in columns:
            value[:, column] = 0
    return tuple(modified)


def shuffle_validation_beta_tuples(betas, val_idx: torch.Tensor, seed: int):
    """Independently shuffle each modality's paired [D,P] recipient tuple."""
    modified = [value.clone() for value in betas]
    indices = val_idx.detach().cpu().long()
    for modality in range(2):
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + modality * 10000)
        permutation = torch.randperm(indices.numel(), generator=generator)
        tuples = betas[modality][indices].clone()
        modified[modality][indices] = tuples[permutation].to(modified[modality].device)
    return tuple(modified)


def validation_global_mean_beta(betas, val_idx: torch.Tensor):
    modified = [value.clone() for value in betas]
    indices = val_idx.detach().cpu().long()
    for modality in range(2):
        mean = betas[modality][indices].mean(0, keepdim=True)
        modified[modality][indices] = mean.expand(indices.numel(), -1)
    return tuple(modified)


def tie_validation_beta_modalities(betas, val_idx: torch.Tensor):
    modified = [value.clone() for value in betas]
    indices = val_idx.detach().cpu().long()
    tied = (betas[0][indices] + betas[1][indices]) * 0.5
    modified[0][indices] = tied
    modified[1][indices] = tied
    return tuple(modified)


def run_interventions(model, classifier, x, edge_index, val_idx, val_y, num_classes,
                      dataset: str, seed: int, base: dict[str, float], betas):
    rows = []
    val_cpu = val_idx.detach().cpu().long()
    original = [value.detach().clone() for value in betas]
    for label, columns in (("D_off", (0,)), ("P_off", (1,)), ("DP_off", (0, 1))):
        modified = beta_off(original, columns)
        metrics = evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes, modified)
        rows.append(_intervention_row(dataset, seed, label, metrics, base))

    for repeat_seed in SHUFFLE_SEEDS:
        modified = shuffle_validation_beta_tuples(original, val_cpu, repeat_seed)
        metrics = evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes, modified)
        rows.append(_intervention_row(dataset, seed, "node_beta_shuffle", metrics, base, repeat_seed))

    modified = validation_global_mean_beta(original, val_cpu)
    metrics = evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes, modified)
    rows.append(_intervention_row(dataset, seed, "validation_global_mean", metrics, base))

    modified = tie_validation_beta_modalities(original, val_cpu)
    metrics = evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes, modified)
    rows.append(_intervention_row(dataset, seed, "modality_tied", metrics, base))
    return rows


def run_one(dataset: str, seed: int, variant: str, device_name: str,
            out_dir: Path = OUT_DIR, epochs: int | None = None,
            save_checkpoint: bool = True, run_intervention_controls: bool = True,
            preloaded_data=None, init_audit=None, smoke: bool = False) -> dict[str, Any]:
    cfg = make_config(dataset, seed, variant, device_name, epochs)
    data = preloaded_data if preloaded_data is not None else load_m0_data(cfg, seed)
    splits = restrict_labels_to_train_val(data)
    data_info = model_data_info(data)
    init_audit = init_audit or audit_initialization(dataset, seed, data_info, cfg)
    if not init_audit["all_bitwise_equal"] or not init_audit["exact_parameter_match"]:
        raise AssertionError("N1 fairness audit failed")
    if data.test_idx is not None or bool(cfg.task.evaluate_test):
        raise AssertionError("N1 must not attach or evaluate test indices")

    device = torch.device(device_name)
    set_seed(seed)
    model = Model(cfg, data_info).to(device)
    model_init_hash = tensor_hash(model.state_dict())
    classifier = build_classifier(model.out_dim, int(data.num_classes), seed, device)
    classifier_init_hash = tensor_hash(classifier.state_dict())
    if model_init_hash != init_audit["variant_hashes"][variant]:
        raise AssertionError(f"{dataset}/{seed}/{variant}: runtime model init differs from audit")
    if classifier_init_hash != init_audit["classifier_hashes"][variant]:
        raise AssertionError(f"{dataset}/{seed}/{variant}: runtime classifier init differs from audit")
    set_seed(seed)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    train_idx, val_idx = splits["train"].to(device), splits["validation"].to(device)
    labels = data.y.to(device)
    train_y, val_y = labels[train_idx], labels[val_idx]
    if bool((train_y < 0).any()) or bool((val_y < 0).any()):
        raise AssertionError("N1 train/validation labels contain missing labels")
    if bool(cfg.task.evaluate_test) or data.test_idx is not None:
        raise AssertionError("N1 test boundary violated")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_acc, best_epoch = -1.0, 0
    patience_left = int(cfg.task.patience)
    best_model = best_classifier = best_metrics = None
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
            raise FloatingPointError(f"non-finite N1 loss: {dataset}/{seed}/{variant}/epoch{epoch}")
        loss.backward()
        if epoch == 1:
            first_gradients = gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()),
            max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        del z, logits, loss
        current = evaluate(model, classifier, x, edge_index, val_idx, val_y, int(data.num_classes))
        if current["val_accuracy"] > best_acc + float(cfg.task.early_stop_min_delta):
            best_acc, best_epoch, best_metrics = current["val_accuracy"], epoch, current
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_classifier = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            patience_left = int(cfg.task.patience)
        elif epoch >= int(cfg.task.early_stop_min_epoch):
            patience_left -= 1
            if patience_left <= 0:
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    if best_model is None or best_classifier is None:
        raise RuntimeError("N1 training ended without a validation-selected checkpoint")
    validate_smoke_gradients(variant, first_gradients) if smoke else None
    model.load_state_dict(best_model, strict=True)
    classifier.load_state_dict(best_classifier, strict=True)
    model.eval(); classifier.eval()
    with torch.no_grad():
        z, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
        selected_metrics = metric_from_embeddings(classifier, z, val_idx, val_y, int(data.num_classes))
        if not torch.isfinite(z).all():
            raise FloatingPointError("N1 checkpoint embeddings are non-finite")
        for beta in aux["betas"]:
            if not torch.isfinite(beta).all() or bool((beta < 0).any()):
                raise FloatingPointError("N1 strengths must be finite and nonnegative")
    diagnostics = collect_diagnostics(dataset, seed, variant, aux, val_idx)
    intervention_rows = []
    intervention_seconds = 0.0
    if variant == "cross_state_strength" and run_intervention_controls:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_start = time.perf_counter()
        intervention_rows = run_interventions(
            model, classifier, x, edge_index, val_idx, val_y, int(data.num_classes),
            dataset, seed, selected_metrics, aux["betas"],
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        intervention_seconds = time.perf_counter() - intervention_start
    base_identity = evaluate(model, classifier, x, edge_index, val_idx, val_y,
                             int(data.num_classes), tuple(value.clone() for value in aux["betas"]))
    identity_error = max(abs(base_identity[key] - selected_metrics[key]) for key in selected_metrics)
    # CUDA scatter accumulation can vary at sub-micro precision across repeated
    # inferences; the override is the identical beta tensor.
    if identity_error > 1e-5:
        raise AssertionError(f"beta override identity failed: {identity_error}")
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    row = {
        "status": "completed", "dataset": dataset, "seed": int(seed), "variant": variant,
        "smoke": bool(smoke),
        "best_epoch": best_epoch, "epochs_run": epochs_run, **selected_metrics,
        "model_params": sum(int(p.numel()) for p in model.parameters()),
        "classifier_params": sum(int(p.numel()) for p in classifier.parameters()),
        "total_params": sum(int(p.numel()) for p in model.parameters()) + sum(int(p.numel()) for p in classifier.parameters()),
        "peak_gpu_memory_bytes": peak_bytes, "training_time_sec": training_seconds,
        "intervention_time_sec": intervention_seconds, "gradient_audit_epoch1": first_gradients,
        "diagnostics": diagnostics, "interventions": intervention_rows,
        "initialization_audit": init_audit, "control_override_identity_max_abs_metric": identity_error,
        "control_override_identity_tolerance": 1e-5,
        "runtime_model_init_hash": model_init_hash,
        "runtime_classifier_init_hash": classifier_init_hash,
        "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        "test_idx_attached": False, "test_labels_exposed": False, "link_prediction": False,
        "validation_label_indices_only": True, "data_info": data_info,
    }
    run_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    write_json(run_path, row)
    if save_checkpoint:
        ckpt = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": best_model, "head_state": best_classifier,
            "data_info": data_info, "dataset": dataset, "seed": int(seed),
            "variant": variant, "best_epoch": best_epoch,
            "best_val_metrics": selected_metrics, "protocol": "unified_full_graph_nc_v1",
            "evaluate_test": False,
        }, ckpt)
        row["checkpoint"] = str(ckpt.relative_to(PROJECT_ROOT))
        write_json(run_path, row)
    print(
        f"[N1 done] {dataset}/{seed}/{variant} epoch={best_epoch} "
        f"acc={selected_metrics['val_accuracy']:.4f} f1={selected_metrics['val_macro_f1']:.4f} "
        f"ce={selected_metrics['val_ce']:.4f} train={training_seconds:.1f}s "
        f"interventions={intervention_seconds:.1f}s peak={peak_bytes / 1024**3:.2f}GiB",
        flush=True,
    )
    del z, aux, data, x, edge_index, model, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def run_smoke(device_name: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    dataset, seed = "Movies", 42
    cfg0 = make_config(dataset, seed, "smooth_only", device_name, epochs=1)
    data = load_m0_data(cfg0, seed)
    restrict_labels_to_train_val(data)
    info = model_data_info(data)
    audit = audit_initialization(dataset, seed, info, cfg0)
    device = torch.device(device_name)
    regression = smooth_m0_regression(dataset, seed, cfg0, data, device)
    # Independent brute-force message check and mask checks are also in the fast test suite.
    smoke_rows = []
    for variant in VARIANTS_ORDERED:
        row = run_one(dataset, seed, variant, device_name, out_dir=out_dir / "smoke_artifacts", epochs=1,
                      save_checkpoint=True, run_intervention_controls=(variant == "cross_state_strength"),
                      preloaded_data=data, init_audit=audit, smoke=True)
        if variant == "static_strength":
            for item in row["diagnostics"]["variation"]:
                if item["validation_std_i"] > 1e-7:
                    raise AssertionError(f"StaticStrength beta must be constant across nodes: {item}")
        if variant in {"same_state_strength", "cross_state_strength"}:
            if not any(item["validation_std_i"] > 0 for item in row["diagnostics"]["variation"]):
                raise AssertionError(f"{variant} did not produce node-varying strengths in smoke")
        smoke_rows.append({"variant": variant, "status": row["status"],
                           "gradient_audit_epoch1": row["gradient_audit_epoch1"],
                           "checkpoint_identity": row["control_override_identity_max_abs_metric"]})
    result = {
        "dataset": dataset, "seed": seed, "device": device_name,
        "initialization_fairness": audit, "smooth_m0_uniform_regression": regression,
        "four_variant_one_epoch_training_smoke": smoke_rows,
        "status": "passed", "evaluate_test": False,
    }
    write_json(out_dir / "smoke" / "movies42.json", result)
    return result


def check_provenance() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    remote_ref = subprocess.check_output(
        ["git", "rev-parse", f"refs/remotes/origin/{SOURCE_BRANCH}"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if head != SOURCE_SHA or remote_ref != SOURCE_SHA or branch != "exp/n1_recipient_function_strength_mixer":
        raise RuntimeError(
            f"N1 provenance mismatch: HEAD={head}, origin/{SOURCE_BRANCH}={remote_ref}, branch={branch}"
        )
    return {"source_branch": SOURCE_BRANCH, "source_sha": SOURCE_SHA,
            "remote_source_sha": remote_ref, "experiment_branch": branch, "head_at_start": head}


def run_campaign(args) -> None:
    provenance = check_provenance()
    if args.smoke:
        run_smoke(args.device, Path(args.out_dir))
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_records, errors, audits, regression = [], [], [], None
    total_start = time.perf_counter()
    if args.regression:
        cfg = make_config("Movies", 42, "smooth_only", args.device)
        data = load_m0_data(cfg, 42)
        restrict_labels_to_train_val(data)
        regression = smooth_m0_regression("Movies", 42, cfg, data, torch.device(args.device))
        write_json(out_dir / "regression" / "m0_uniform_movies42.json", regression)
        del data
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg = make_config(dataset, seed, "smooth_only", args.device, args.epochs)
            data = load_m0_data(cfg, seed)
            restrict_labels_to_train_val(data)
            info = model_data_info(data)
            audit = audit_initialization(dataset, seed, info, cfg)
            audits.extend(initialization_rows(audit))
            for variant in VARIANTS_ORDERED:
                run_path = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if run_path.exists():
                    try:
                        existing = json.loads(run_path.read_text(encoding="utf-8"))
                        if existing.get("status") == "completed" and not existing.get("smoke", False):
                            run_records.append(existing)
                            continue
                        if not args.retry_failed:
                            errors.append({"dataset": dataset, "seed": seed, "variant": variant,
                                           "status": "existing_noncompleted_record_not_retried"})
                            continue
                    except Exception:
                        if not args.retry_failed:
                            errors.append({"dataset": dataset, "seed": seed, "variant": variant,
                                           "status": "existing_record_unreadable_not_retried"})
                            continue
                try:
                    result = run_one(
                        dataset, seed, variant, args.device, out_dir=out_dir,
                        epochs=args.epochs, save_checkpoint=True,
                        run_intervention_controls=True, preloaded_data=data,
                        init_audit=audit, smoke=False,
                    )
                    run_records.append(result)
                except Exception as exc:
                    entry = {"dataset": dataset, "seed": seed, "variant": variant,
                             "status": "failed", "error": repr(exc),
                             "traceback": traceback.format_exc(), "retry_attempt": bool(args.retry_failed)}
                    errors.append(entry)
                    write_json(out_dir / "failures" / dataset / f"seed_{seed}_{variant}.json", entry)
                    print(f"[N1 failed] {dataset}/{seed}/{variant}: {exc!r}", flush=True)
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elapsed = time.perf_counter() - total_start
    manifest = {
        **provenance, "datasets": list(DATASETS), "seeds": list(SEEDS),
        "variants": list(VARIANTS_ORDERED), "expected_runs": 36,
        "completed_runs": len([row for row in run_records if row.get("status") == "completed"]),
        "failures": errors, "runtime_seconds": elapsed, "device": args.device,
        "epochs_override": args.epochs, "protocol": "unified_full_graph_nc_v1",
        "evaluate_test": False, "test_labels_exposed": False, "link_prediction": False,
        "smooth_m0_uniform_regression": regression,
        "training_config": OmegaConf.to_container(make_config("Movies", 42, "smooth_only", args.device).task,
                                                    resolve=True),
    }
    write_json(out_dir / "campaign_manifest.json", manifest)
    write_json(out_dir / "failures.json", errors)
    write_csv(RESEARCH_DIR / "data" / "parameter_init_audit.csv", audits)
    if len(run_records) >= 1:
        from scripts.analyze_n1_recipient_function_strength_mixer import analyze_campaign
        analyze_campaign(out_dir, RESEARCH_DIR, manifest)
    print(f"[N1 campaign] completed={manifest['completed_runs']}/36 failed={len(errors)} "
          f"elapsed={elapsed:.1f}s device={args.device}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Explicitly retry prior failed/unreadable run records")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    if not args.smoke and not args.campaign:
        parser.error("choose --smoke or --campaign")
    run_campaign(args)


if __name__ == "__main__":
    main()
