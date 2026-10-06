from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.analysis.r3mag_context_utils import (
    ResidualAdapter,
    ResponsePredictor,
    apply_bounded_residual,
    apply_signed_permutation,
    build_relation_context,
    coherent_control_donors,
    direction_geometry,
    gram_preservation_errors,
    make_signed_permutation,
    matched_context_shuffle,
)
from src.analysis.r3mag_h1_response_audit import (
    CONTROL_SEEDS,
    DATA_SEED,
    PARTITION_SEED,
    _degree_and_prediction_buckets,
    _json_safe,
    _operator,
    _train_host,
    _write_json,
    assert_disjoint_splits,
    evaluate_joint_action_utilities,
    evaluate_single_action_utilities,
    load_fixed_data,
    matched_control_donors,
    stratified_internal_partition,
    tensor_sha256,
)
from src.analysis.r3mag_response_core import (
    GlobalDualBranchResponseHost,
    R3MAGHostConfig,
    action_names,
    make_action_candidates,
    normalized_response_directions,
)
from src.utils.seeds import set_seed


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports/r3mag_design_freeze/h11_h2"
OUTPUT_DIR = ROOT / "outputs/r3mag_design_freeze/h11_h2"
DATASETS = ("Movies", "Grocery", "ele-fashion")
HOST_SEEDS = (42, 43, 44)
CONTEXT_DIM = 25
PREDICTOR_SPLIT_SEED = 20261007
EPSILONS = (0.1, 0.2)


def _count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _index_hashes(data, partitions: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        "OriginalTrain": tensor_sha256(data.train_idx),
        "HostTrain": tensor_sha256(partitions["host_train"]),
        "ResponseTrain": tensor_sha256(partitions["response_train"]),
        "Audit": tensor_sha256(partitions["audit"]),
        "Val": tensor_sha256(data.val_idx),
        "Test": tensor_sha256(data.test_idx),
    }


def _initial_run_qa() -> dict[str, Any]:
    """Create the run QA payload before checkpoint validation writes into it."""
    return {
        "internal_splits_disjoint": True,
        "split_hashes_match_h1": True,
        "test_labels_read": False,
        "host_frozen": False,
        "relation_context_finite": False,
        "relation_context_shape_pass": False,
        "audit_target_used_for_training": False,
    }


def _verify_previous_split(dataset: str, host_seed: int, hashes: dict[str, str]) -> dict[str, Any]:
    path = REPORT_DIR.parent / "h1" / "per_run" / f"{dataset}_seed{host_seed}.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    expected = old["split"]["index_sha256"]
    mismatches = {name: (hashes[name], expected[name]) for name in hashes if hashes[name] != expected[name]}
    if mismatches:
        raise AssertionError(f"{dataset}/seed{host_seed}: split hashes differ from H1: {mismatches}")
    if old["split"].get("test_labels_read") is not False:
        raise AssertionError(f"Previous H1 report does not certify untouched test labels: {path}")
    return {"path": str(path), "matches_previous_h1": True}


def _load_host(data, dataset: str, host_seed: int, operator: torch.Tensor, partitions,
               device: torch.device, *, allow_train_missing: bool = True):
    checkpoint = OUTPUT_DIR.parent / "h1" / dataset / f"seed{host_seed}" / "host_best_val.pt"
    reused = checkpoint.exists()
    if not reused:
        if not allow_train_missing:
            raise FileNotFoundError(checkpoint)
        checkpoint = OUTPUT_DIR.parent / "h1" / dataset / f"seed{host_seed}" / "host_best_val.pt"
        _train_host(data, operator, partitions, host_seed, device, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = R3MAGHostConfig(**saved.get("config", {}))
    host = GlobalDualBranchResponseHost(
        int(data.x_t.size(1)), int(data.x_i.size(1)), int(data.num_classes), config
    ).to(device)
    host.load_state_dict(saved["model_state"])
    host.eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    with torch.no_grad():
        logits, states_text, states_visual, global_text, global_visual = host(
            data.x_t.to(device), data.x_i.to(device), operator
        )
    return host, logits, states_text, states_visual, global_text, global_visual, {
        "path": str(checkpoint), "reused": bool(reused), "rerun": not bool(reused),
        "best_epoch": int(saved.get("best_epoch", -1)),
    }


def _stratified_response_split(
    response_idx: torch.Tensor, response_labels: torch.Tensor, seed: int = PREDICTOR_SPLIT_SEED
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ids = torch.as_tensor(response_idx, dtype=torch.long).cpu()
    labels = torch.as_tensor(response_labels, dtype=torch.long).cpu()
    rng = np.random.default_rng(int(seed))
    train: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    singleton_classes: list[int] = []
    allocations: dict[str, dict[str, int]] = {}
    for class_id in sorted(int(x) for x in torch.unique(labels).tolist()):
        class_ids = ids[labels == class_id].numpy()
        shuffled = rng.permutation(class_ids)
        if len(shuffled) < 2:
            n_valid = 0
            singleton_classes.append(class_id)
        else:
            n_valid = min(len(shuffled) - 1, max(1, int(round(0.2 * len(shuffled)))))
        valid.append(shuffled[:n_valid])
        train.append(shuffled[n_valid:])
        allocations[str(class_id)] = {"response_train": int(len(shuffled)), "predictor_train": int(len(shuffled) - n_valid), "predictor_val": int(n_valid)}
    train_idx = torch.as_tensor(np.concatenate(train) if train else [], dtype=torch.long).sort().values
    valid_idx = torch.as_tensor(np.concatenate(valid) if valid else [], dtype=torch.long).sort().values
    if set(train_idx.tolist()) & set(valid_idx.tolist()) or set(torch.cat([train_idx, valid_idx]).tolist()) != set(ids.tolist()):
        raise AssertionError("predictor train/val split must be disjoint and cover ResponseTrain")
    return train_idx, valid_idx, {
        "seed": int(seed), "method": "class-stratified NumPy permutation; rounded 20% validation with train/val nonempty when class size >=2",
        "singleton_classes_without_validation": singleton_classes, "allocations": allocations,
        "train_sha256": tensor_sha256(train_idx), "val_sha256": tensor_sha256(valid_idx),
    }


def _receiver_features(logits: torch.Tensor, states_text: list[torch.Tensor], states_visual: list[torch.Tensor],
                       global_text: torch.Tensor, global_visual: torch.Tensor) -> torch.Tensor:
    probability = logits.softmax(dim=-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1, keepdim=True)
    top = probability.topk(k=min(2, probability.size(-1)), dim=-1).values
    margin = (top[:, 0] - top[:, 1]).unsqueeze(-1) if top.size(-1) > 1 else top[:, :1]
    return torch.cat([
        states_text[0], global_text, states_visual[0], global_visual, logits, entropy, margin
    ], dim=-1).detach().float()


def _compute_targets(host, states_text, states_visual, global_text, global_visual,
                     logits, labels: torch.Tensor, indices: torch.Tensor, device: torch.device):
    ids = indices.to(device)
    y = labels[indices.cpu()].long().to(device)
    gt, _ = normalized_response_directions(states_text, global_text)
    gv, _ = normalized_response_directions(states_visual, global_visual)
    gt, gv = gt[ids], gv[ids]
    cg_t, cg_v = global_text[ids], global_visual[ids]
    base_logits = logits[ids]
    ct = make_action_candidates(cg_t, gt, 0.1)
    cv = make_action_candidates(cg_v, gv, 0.1)
    ut, uv, _, _ = evaluate_single_action_utilities(host, ct, cv, cg_t, cg_v, y, base_logits)
    slopes_t = (ut[:, 1:5] - ut[:, 5:9]) / 0.2
    slopes_v = (uv[:, 1:5] - uv[:, 5:9]) / 0.2
    target = torch.cat([slopes_t, slopes_v], dim=-1)
    action_utilities = torch.stack([ut[:, 1:9], uv[:, 1:9]], dim=1)
    return target, action_utilities


def _predictor_train(x_train, y_train, x_val, y_val, *, seed: int, max_epochs: int,
                     patience: int, device: torch.device) -> tuple[ResponsePredictor, dict[str, Any]]:
    set_seed(seed)
    model = ResponsePredictor(x_train.size(-1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_loss, best_epoch, best_state, left = float("inf"), 0, None, int(patience)
    epochs_ran = 0
    for epoch in range(1, int(max_epochs) + 1):
        epochs_ran = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(x_train), y_train)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite response predictor training Huber")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.smooth_l1_loss(model(x_val), y_val).item())
        if val_loss < best_loss - 1e-9:
            best_loss, best_epoch, left = val_loss, epoch, int(patience)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                break
    if best_state is None:
        raise RuntimeError("response predictor did not produce a best state")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_ran": epochs_ran, "best_val_huber": best_loss,
                   "parameter_count": _count_parameters(model)}


def _prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, action_u: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spearman: list[float] = []
    sign_ba: list[float] = []
    skipped_spearman: list[int] = []
    skipped_sign: list[int] = []
    for dim in range(y_true.shape[1]):
        if np.ptp(y_true[:, dim]) > 1e-12 and np.ptp(y_pred[:, dim]) > 1e-12:
            value = spearmanr(y_true[:, dim], y_pred[:, dim]).statistic
            if np.isfinite(value):
                spearman.append(float(value))
            else:
                skipped_spearman.append(dim)
        else:
            skipped_spearman.append(dim)
        nonzero = np.abs(y_true[:, dim]) > 1e-12
        truth_sign = y_true[nonzero, dim] > 0
        if np.unique(truth_sign).size == 2:
            sign_ba.append(float(balanced_accuracy_score(truth_sign, y_pred[nonzero, dim] > 0)))
        else:
            skipped_sign.append(dim)
    selected_by_modality = []
    regret_by_modality = []
    chosen_indices = []
    for modality, offset in enumerate((0, 4)):
        predicted_slopes = y_pred[:, offset : offset + 4]
        order = np.abs(predicted_slopes).argmax(axis=1)
        action = order + np.where(predicted_slopes[np.arange(len(order)), order] > 0, 0, 4)
        chosen_indices.append(action)
        chosen = action_u[:, modality, action]
        regret = action_u[:, modality].max(axis=1) - chosen
        selected_by_modality.append(chosen)
        regret_by_modality.append(regret)
    selected = np.stack(selected_by_modality, axis=1)
    regret = np.stack(regret_by_modality, axis=1)
    return {
        "mean_per_dimension_spearman": float(np.mean(spearman)) if spearman else None,
        "spearman_dimensions_included": len(spearman), "spearman_dimensions_skipped": skipped_spearman,
        "mean_per_dimension_sign_balanced_accuracy": float(np.mean(sign_ba)) if sign_ba else None,
        "sign_dimensions_included": len(sign_ba), "sign_dimensions_skipped_single_class": skipped_sign,
        "selected_action_utility": {"text": float(selected[:, 0].mean()), "visual": float(selected[:, 1].mean()), "mean": float(selected.mean())},
        "regret": {"text": float(regret[:, 0].mean()), "visual": float(regret[:, 1].mean()), "mean": float(regret.mean())},
    }, {"selected_utility": selected, "regret": regret, "chosen_action_index": np.stack(chosen_indices, axis=1)}


def _run_h2a(host, receiver, context, disagreement, states_text, states_visual, global_text, global_visual,
             logits, labels, response_idx, audit_idx, degree_bucket, predicted_class,
             *, seed: int, max_epochs: int, patience: int, device: torch.device):
    response_labels = labels[response_idx]
    train_ids, val_ids, split_meta = _stratified_response_split(response_idx, response_labels)
    target_response, _ = _compute_targets(host, states_text, states_visual, global_text, global_visual,
                                          logits, labels, response_idx, device)
    # Membership split is fixed before fitting; standardization sees predictor-train targets only.
    pos = {int(node): i for i, node in enumerate(response_idx.tolist())}
    train_rows = torch.tensor([pos[int(i)] for i in train_ids.tolist()], device=device)
    val_rows = torch.tensor([pos[int(i)] for i in val_ids.tolist()], device=device)
    target_train_raw = target_response[train_rows]
    target_mean = target_train_raw.mean(0)
    target_std = target_train_raw.std(0, unbiased=False).clamp_min(1e-6)
    target_response_scaled = (target_response - target_mean) / target_std
    context_resp = context[response_idx.to(device)]
    context_audit = context[audit_idx.to(device)]
    resp_donors, resp_shuffle_counts = matched_context_shuffle(response_idx, degree_bucket, predicted_class, seed + 3101)
    audit_donors, audit_shuffle_counts = matched_context_shuffle(audit_idx, degree_bucket, predicted_class, seed + 6203)
    shuffled_resp = context[resp_donors.to(device)]
    shuffled_audit = context[audit_donors.to(device)]
    receiver_resp, receiver_audit = receiver[response_idx.to(device)], receiver[audit_idx.to(device)]
    zeros_resp, zeros_audit = torch.zeros_like(context_resp), torch.zeros_like(context_audit)
    feature_resp = {
        "P1_ReceiverOnly": torch.cat([receiver_resp, zeros_resp], dim=-1),
        "P2_ReceiverPlusRealContext": torch.cat([receiver_resp, context_resp], dim=-1),
        "P3_ReceiverPlusShuffledContext": torch.cat([receiver_resp, shuffled_resp], dim=-1),
    }
    feature_audit = {
        "P1_ReceiverOnly": torch.cat([receiver_audit, zeros_audit], dim=-1),
        "P2_ReceiverPlusRealContext": torch.cat([receiver_audit, context_audit], dim=-1),
        "P3_ReceiverPlusShuffledContext": torch.cat([receiver_audit, shuffled_audit], dim=-1),
        "P2_eval_shuffle": torch.cat([receiver_audit, shuffled_audit], dim=-1),
    }
    results: dict[str, Any] = {}
    prediction_cache: dict[str, dict[str, np.ndarray]] = {}
    model_cache: dict[str, ResponsePredictor] = {}
    training_cache: dict[str, dict[str, Any]] = {}
    for name in ("P1_ReceiverOnly", "P2_ReceiverPlusRealContext", "P3_ReceiverPlusShuffledContext"):
        model, training = _predictor_train(
            feature_resp[name][train_rows], target_response_scaled[train_rows],
            feature_resp[name][val_rows], target_response_scaled[val_rows], seed=seed,
            max_epochs=max_epochs, patience=patience, device=device,
        )
        training_cache[name] = training
        model_cache[name] = model
    # Audit targets and labels are first indexed after every predictor variant is fitted.
    target_audit, audit_actions = _compute_targets(host, states_text, states_visual, global_text, global_visual,
                                                  logits, labels, audit_idx, device)
    for name, model in model_cache.items():
        with torch.no_grad():
            pred = model(feature_audit[name]) * target_std + target_mean
        metrics, arrays = _prediction_metrics(target_audit.detach().cpu().numpy(), pred.detach().cpu().numpy(),
                                               audit_actions.detach().cpu().numpy())
        results[name] = {"training": training_cache[name], "metrics": metrics}
        prediction_cache[name] = arrays
    # Reuse the exact fitted P2 weights; only the Audit context correspondence changes.
    p2_model = model_cache["P2_ReceiverPlusRealContext"]
    with torch.no_grad():
        p2_eval_pred = p2_model(feature_audit["P2_eval_shuffle"]) * target_std + target_mean
    p2_eval_metrics, p2_eval_arrays = _prediction_metrics(
        target_audit.detach().cpu().numpy(), p2_eval_pred.detach().cpu().numpy(), audit_actions.detach().cpu().numpy()
    )
    results["P2_eval_shuffle"] = {"training": {"same_model_as": "P2_ReceiverPlusRealContext"}, "metrics": p2_eval_metrics}
    prediction_cache["P2_eval_shuffle"] = p2_eval_arrays
    # One exploratory disagreement split: bottom/top quartile of H0 cross-modal edge-cue difference.
    d = disagreement[audit_idx.to(device)].detach().cpu().numpy()
    order = np.argsort(d, kind="stable")
    q = max(1, len(order) // 4)
    subgroup = {"low_disagreement": order[:q], "high_disagreement": order[-q:]}
    subgroup_metrics: dict[str, Any] = {}
    for group, rows in subgroup.items():
        delta_selected = prediction_cache["P2_ReceiverPlusRealContext"]["selected_utility"][rows] - prediction_cache["P1_ReceiverOnly"]["selected_utility"][rows]
        delta_regret = prediction_cache["P2_ReceiverPlusRealContext"]["regret"][rows] - prediction_cache["P1_ReceiverOnly"]["regret"][rows]
        subgroup_metrics[group] = {
            "node_count": int(len(rows)),
            "mean_disagreement": float(d[rows].mean()),
            "P2_minus_P1_selected_action_utility": {"text": float(delta_selected[:, 0].mean()), "visual": float(delta_selected[:, 1].mean()), "mean": float(delta_selected.mean())},
            "P2_minus_P1_regret": {"text": float(delta_regret[:, 0].mean()), "visual": float(delta_regret[:, 1].mean()), "mean": float(delta_regret.mean())},
        }
    results["disagreement_subgroups"] = subgroup_metrics
    results["parameter_counts"] = {name: results[name]["training"]["parameter_count"] for name in (
        "P1_ReceiverOnly", "P2_ReceiverPlusRealContext", "P3_ReceiverPlusShuffledContext")}
    if len(set(results["parameter_counts"].values())) != 1:
        raise AssertionError("P1/P2/P3 must have equal parameter counts")
    return results, {
        "predictor_split": split_meta,
        "response_target_stats_train_only": {"mean": target_mean.detach().cpu().tolist(), "std": target_std.detach().cpu().tolist()},
        "response_target_generated_after_frozen_host": True,
        "audit_targets_used_for_training": False,
        "audit_target_count": int(len(audit_idx)),
        "response_train_target_count": int(len(response_idx)),
        "response_targets_finite": bool(torch.isfinite(target_response).all()),
        "audit_targets_finite_for_evaluation": bool(torch.isfinite(target_audit).all()),
        "audit_target_used_for_training": False,
        "predictor_training_used_response_train_only": True,
        "response_context_shuffle": resp_shuffle_counts,
        "audit_context_shuffle": audit_shuffle_counts,
        "context_shuffle_same_split_only": True,
    }


def _adapter_fit(host, receiver, context, states_text, states_visual, global_text, global_visual,
                labels, response_idx, train_ids, val_ids, train_rows, val_rows, response_context,
                *, name: str, seed: int, max_epochs: int, patience: int, device: torch.device):
    feature = torch.cat([receiver[response_idx.to(device)], response_context], dim=-1)
    x_train, x_val = feature[train_rows], feature[val_rows]
    y_train = labels[train_ids].long().to(device)
    y_val = labels[val_ids].long().to(device)
    set_seed(seed)
    adapter = ResidualAdapter(feature.size(-1), response_dim=global_text.size(-1)).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=5e-4)
    before = {k: v.detach().cpu().clone() for k, v in host.state_dict().items()}
    best_ce, best_epoch, best_state, left = float("inf"), 0, None, int(patience)
    epochs_ran = 0
    for epoch in range(1, int(max_epochs) + 1):
        epochs_ran = epoch
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        raw, gate = adapter(x_train)
        ct, cv, _, _ = apply_bounded_residual(
            global_text[train_ids.to(device)], global_visual[train_ids.to(device)], raw, gate
        )
        train_logits = host.classify_responses(ct, cv)
        loss = F.cross_entropy(train_logits, y_train)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: nonfinite adapter CE")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        adapter.eval()
        with torch.no_grad():
            raw_v, gate_v = adapter(x_val)
            ctv, cvv, _, _ = apply_bounded_residual(
                global_text[val_ids.to(device)], global_visual[val_ids.to(device)], raw_v, gate_v
            )
            val_ce = float(F.cross_entropy(host.classify_responses(ctv, cvv), y_val).item())
        if val_ce < best_ce - 1e-9:
            best_ce, best_epoch, left = val_ce, epoch, int(patience)
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            left -= 1
            if left <= 0:
                break
    if best_state is None:
        raise RuntimeError(f"{name}: adapter did not produce a best state")
    adapter.load_state_dict(best_state)
    adapter.eval()
    after = host.state_dict()
    host_state_unchanged = all(torch.equal(before[k], after[k].detach().cpu()) for k in before)
    no_host_grad = all(parameter.grad is None for parameter in host.parameters())
    if not host_state_unchanged or not no_host_grad:
        raise AssertionError("frozen host state or gradients changed during adapter fit")
    return adapter, {"best_epoch": best_epoch, "epochs_ran": epochs_ran, "best_val_ce": best_ce,
                     "parameter_count": _count_parameters(adapter), "host_state_unchanged": host_state_unchanged,
                     "host_gradients_absent": no_host_grad}


def _adapter_eval(host, adapter, receiver, contexts, states_text, states_visual, global_text, global_visual,
                  audit_idx, labels, baseline_logits, device):
    ids = audit_idx.to(device)
    feature = torch.cat([receiver[ids], contexts], dim=-1)
    with torch.no_grad():
        raw, gate_logits = adapter(feature)
        corrected_t, corrected_v, gates, ratio = apply_bounded_residual(
            global_text[ids], global_visual[ids], raw, gate_logits
        )
        logits = host.classify_responses(corrected_t, corrected_v)
        finite = bool(torch.isfinite(raw).all() and torch.isfinite(gates).all() and torch.isfinite(ratio).all() and torch.isfinite(logits).all())
    y = labels[audit_idx.cpu()].long().to(device)
    baseline_ce = F.cross_entropy(baseline_logits[ids], y, reduction="none")
    ce = F.cross_entropy(logits, y, reduction="none")
    predictions = logits.argmax(-1)
    metrics = {
        "ce": float(ce.mean().item()),
        "accuracy": float((predictions == y).float().mean().item()),
        "macro_f1": float(f1_score(y.detach().cpu().numpy(), predictions.detach().cpu().numpy(),
                                   labels=list(range(logits.size(-1))), average="macro", zero_division=0)),
        "delta_ce_mean_global_minus_adapter": float((baseline_ce - ce).mean().item()),
        "positive_improvement_fraction": float(((baseline_ce - ce) > 0).float().mean().item()),
        "harmful_fraction": float(((baseline_ce - ce) < 0).float().mean().item()),
        "gate_mean": {"text": float(gates[:, 0].mean().item()), "visual": float(gates[:, 1].mean().item())},
        "gate_median": {"text": float(gates[:, 0].median().item()), "visual": float(gates[:, 1].median().item())},
        "residual_norm_ratio_mean": {"text": float(ratio[:, 0].mean().item()), "visual": float(ratio[:, 1].mean().item())},
        "residual_outputs_finite": finite,
        "audit_count": int(len(audit_idx)),
    }
    return metrics, logits.detach(), {"ce_by_node": ce.detach().cpu(), "delta_ce_by_node": (baseline_ce - ce).detach().cpu()}


def _run_h2b(host, receiver, context, states_text, states_visual, global_text, global_visual,
             logits, labels, response_idx, audit_idx, train_ids, val_ids, split_meta,
             degree_bucket, predicted_class, *, seed: int, max_epochs: int, patience: int, device: torch.device):
    position = {int(node): i for i, node in enumerate(response_idx.tolist())}
    train_rows = torch.tensor([position[int(node)] for node in train_ids.tolist()], dtype=torch.long, device=device)
    val_rows = torch.tensor([position[int(node)] for node in val_ids.tolist()], dtype=torch.long, device=device)
    context_resp = context[response_idx.to(device)]
    response_donors, response_shuffle = matched_context_shuffle(response_idx, degree_bucket, predicted_class, seed + 3101)
    audit_donors, audit_shuffle = matched_context_shuffle(audit_idx, degree_bucket, predicted_class, seed + 6203)
    shuffled_resp = context[response_donors.to(device)]
    shuffled_audit = context[audit_donors.to(device)]
    variants = {
        "B1_ReceiverResidual": torch.zeros_like(context_resp),
        "B2_ContextResidual": context_resp,
        "B3_ShuffledContextResidual": shuffled_resp,
    }
    audit_contexts = {
        "B1_ReceiverResidual": torch.zeros_like(context[audit_idx.to(device)]),
        "B2_ContextResidual": context[audit_idx.to(device)],
        "B3_ShuffledContextResidual": shuffled_audit,
    }
    y_response = labels[response_idx].long()
    train_split_ids, val_split_ids = train_ids, val_ids
    results: dict[str, Any] = {}
    adapters: dict[str, ResidualAdapter] = {}
    host_snapshot = {k: v.detach().cpu().clone() for k, v in host.state_dict().items()}
    for name in ("B1_ReceiverResidual", "B2_ContextResidual", "B3_ShuffledContextResidual"):
        adapter, training = _adapter_fit(host, receiver, context, states_text, states_visual,
            global_text, global_visual, labels, response_idx, train_split_ids, val_split_ids,
            train_rows, val_rows, variants[name], name=name, seed=seed, max_epochs=max_epochs,
            patience=patience, device=device)
        results[name] = {"training": training}
        adapters[name] = adapter
        if name == "B2_ContextResidual":
            b2_adapter = adapter
    # Audit labels are first evaluated after every adapter variant is fitted.
    for name, adapter in adapters.items():
        metrics, _, _ = _adapter_eval(host, adapter, receiver, audit_contexts[name], states_text,
            states_visual, global_text, global_visual, audit_idx, labels, logits, device)
        results[name]["metrics"] = metrics
    b2_eval_metrics, _, _ = _adapter_eval(host, b2_adapter, receiver, shuffled_audit, states_text,
        states_visual, global_text, global_visual, audit_idx, labels, logits, device)
    results["B2_eval_shuffle"] = {"training": {"same_model_as": "B2_ContextResidual"}, "metrics": b2_eval_metrics}
    base_y = labels[audit_idx.cpu()].long().to(device)
    base_logits = logits[audit_idx.to(device)]
    base_ce = F.cross_entropy(base_logits, base_y)
    base_pred = base_logits.argmax(-1)
    results["B0_GlobalFrozen"] = {"metrics": {
        "ce": float(base_ce.item()), "accuracy": float((base_pred == base_y).float().mean().item()),
        "macro_f1": float(f1_score(base_y.cpu().numpy(), base_pred.cpu().numpy(), labels=list(range(base_logits.size(-1))), average="macro", zero_division=0)),
        "audit_count": int(len(audit_idx)),
    }}
    for name in ("B1_ReceiverResidual", "B2_ContextResidual", "B3_ShuffledContextResidual", "B2_eval_shuffle"):
        metrics = results[name]["metrics"]
        metrics["ce_delta_vs_B0"] = metrics["ce"] - results["B0_GlobalFrozen"]["metrics"]["ce"]
        metrics["accuracy_delta_vs_B0"] = metrics["accuracy"] - results["B0_GlobalFrozen"]["metrics"]["accuracy"]
        metrics["macro_f1_delta_vs_B0"] = metrics["macro_f1"] - results["B0_GlobalFrozen"]["metrics"]["macro_f1"]
    pairs = (("B1_minus_B0", "B1_ReceiverResidual", "B0_GlobalFrozen"),
             ("B2_minus_B0", "B2_ContextResidual", "B0_GlobalFrozen"),
             ("B2_minus_B1", "B2_ContextResidual", "B1_ReceiverResidual"),
             ("B2_minus_B3", "B2_ContextResidual", "B3_ShuffledContextResidual"),
             ("B2_minus_B2_eval_shuffle", "B2_ContextResidual", "B2_eval_shuffle"))
    results["pairwise_comparisons"] = {}
    for label, left, right in pairs:
        lm, rm = results[left]["metrics"], results[right]["metrics"]
        results["pairwise_comparisons"][label] = {
            "ce_left_minus_right": lm["ce"] - rm["ce"],
            "accuracy_left_minus_right": lm["accuracy"] - rm["accuracy"],
            "macro_f1_left_minus_right": lm["macro_f1"] - rm["macro_f1"],
        }
    param_counts = {name: results[name]["training"]["parameter_count"] for name in variants}
    if len(set(param_counts.values())) != 1:
        raise AssertionError("B1/B2/B3 parameter counts differ")
    host_unchanged = all(torch.equal(host_snapshot[k], host.state_dict()[k].detach().cpu()) for k in host_snapshot)
    if not host_unchanged:
        raise AssertionError("host changed across residual screen")
    results["parameter_counts"] = param_counts
    return results, {
        "response_train_split": split_meta, "response_context_shuffle": response_shuffle,
        "audit_context_shuffle": audit_shuffle, "shuffle_split_boundary_pass": True,
        "audit_labels_used_for_training": False, "host_state_unchanged_after_all_adapters": host_unchanged,
        "adapter_trainable_parameters_only": True,
    }


def _action_bank_metrics(host, directions_t, directions_v, global_t, global_v,
                        audit_idx, labels, baseline_logits, epsilon, device):
    ids = audit_idx.to(device)
    g_t, g_v = global_t[ids], global_v[ids]
    d_t, d_v = directions_t.to(device), directions_v.to(device)
    cand_t = make_action_candidates(g_t, d_t, epsilon)
    cand_v = make_action_candidates(g_v, d_v, epsilon)
    ut, uv, _, _ = evaluate_single_action_utilities(host, cand_t, cand_v, g_t, g_v, labels, baseline_logits)
    joint = evaluate_joint_action_utilities(host, cand_t, cand_v, g_t, g_v, labels, baseline_logits)
    def single_summary(value: torch.Tensor):
        means = value.mean(0)
        return {"node_oracle_mean": float(value.max(1).values.mean().item()),
                "best_global_action_mean": float(means.max().item()),
                "node_specific_headroom": float((value.max(1).values.mean() - means.max()).item())}
    separate = joint.flatten(1).max(1).values
    shared = torch.diagonal(joint, dim1=1, dim2=2).max(1).values
    return {
        "text": single_summary(ut), "visual": single_summary(uv),
        "joint": {"separate_pair_oracle_mean": float(separate.mean().item()),
                  "shared_pair_oracle_mean": float(shared.mean().item()),
                  "separate_minus_shared_headroom": float((separate - shared).mean().item())},
        "action_count": int(cand_t.size(1)),
    }, (cand_t, cand_v)


def _run_h11(host, states_text, states_visual, global_text, global_visual, logits,
             audit_idx, audit_labels, edge_index, num_nodes, *, repeats: int, device: torch.device):
    own_t_all, zero_t = normalized_response_directions(states_text, global_text)
    own_v_all, zero_v = normalized_response_directions(states_visual, global_visual)
    own_t, own_v = own_t_all[audit_idx.to(device)], own_v_all[audit_idx.to(device)]
    raw_text = torch.stack(states_text, dim=1) - global_text[:, None, :]
    raw_visual = torch.stack(states_visual, dim=1) - global_visual[:, None, :]

    def borrow(raw_directions: torch.Tensor, global_response: torch.Tensor, donors: torch.Tensor) -> torch.Tensor:
        target_norm = global_response[audit_idx.to(device)].norm(dim=-1, keepdim=True).unsqueeze(1)
        selected = torch.stack([raw_directions[donors[:, k].to(device), k] for k in range(4)], dim=1)
        direction_norm = selected.norm(dim=-1, keepdim=True)
        matched = selected / direction_norm.clamp_min(1e-12) * target_norm
        return torch.where(direction_norm > 1e-12, matched, torch.zeros_like(matched))

    degree, degree_bucket, predicted = _degree_and_prediction_buckets(edge_index, num_nodes, logits.argmax(-1))
    geometries: dict[str, list[dict[str, Any]]] = {k: [] for k in ("OwnStructural", "CoherentDonor", "IndependentDonor", "GeometryScrambled")}
    utility_by_epsilon: dict[str, Any] = {}
    qa = {"coherent_one_donor_for_all_branches_orders": True, "coherent_self_donor_count": 0,
          "signed_permutation_valid": True, "signed_permutation_gram_abs_max": 0.0,
          "signed_permutation_gram_rel_max": 0.0, "normmatch_max_relative_error": 0.0,
          "action_cardinality_pass": True, "epsilon_zero_max_abs_error": 0.0,
          "matching_counts_coherent": [], "matching_counts_independent": [],
          "near_zero_direction_count_text": int(zero_t.sum().item()),
          "near_zero_direction_count_visual": int(zero_v.sum().item())}
    control_seed_list = CONTROL_SEEDS[:int(repeats)]
    bank_values: dict[str, dict[float, list[dict[str, Any]]]] = {
        name: {eps: [] for eps in EPSILONS} for name in geometries
    }
    audit_global_t, audit_global_v = global_text[audit_idx.to(device)], global_visual[audit_idx.to(device)]
    # Unique direction geometry is independent of epsilon; compute per repeat for sampled controls.
    for repeat, control_seed in enumerate(control_seed_list):
        coherent, coherent_counts = coherent_control_donors(audit_idx.cpu(), degree_bucket, predicted, num_nodes, control_seed)
        independent, independent_counts = matched_control_donors(audit_idx.cpu(), degree_bucket, predicted, num_nodes, control_seed)
        qa["matching_counts_coherent"].append(coherent_counts)
        qa["matching_counts_independent"].append(independent_counts)
        qa["coherent_self_donor_count"] += int(coherent_counts.get("self_donor_count", 0))
        coherent_full = coherent[:, None, None].expand(-1, 2, 4)
        if not torch.equal(coherent_full[:, 0, :], coherent_full[:, 1, :]):
            raise AssertionError("coherent donor identity changed across branches or orders")
        if int((coherent_full[:, 0, 0] == audit_idx.cpu()).sum().item()) != 0:
            raise AssertionError("coherent donor selected its own target")
        coherent_orders = coherent[:, None].expand(-1, 4)
        coherent_t = borrow(raw_text, global_text, coherent_orders)
        coherent_v = borrow(raw_visual, global_visual, coherent_orders)
        independent_t = borrow(raw_text, global_text, independent[:, 0])
        independent_v = borrow(raw_visual, global_visual, independent[:, 1])
        permutation, signs = make_signed_permutation(own_t.size(-1), int(control_seed) + 7919)
        qa["signed_permutation_valid"] &= bool(
            torch.equal(torch.sort(permutation).values, torch.arange(own_t.size(-1)))
            and set(signs.tolist()).issubset({-1.0, 1.0})
        )
        scrambled_t = apply_signed_permutation(own_t, permutation, signs)
        scrambled_v = apply_signed_permutation(own_v, permutation, signs)
        for name, dt, dv in (("OwnStructural", own_t, own_v), ("CoherentDonor", coherent_t, coherent_v),
                             ("IndependentDonor", independent_t, independent_v), ("GeometryScrambled", scrambled_t, scrambled_v)):
            if name != "OwnStructural" or repeat == 0:
                geometries[name].append({"text": direction_geometry(dt), "visual": direction_geometry(dv)})
        sampled = torch.arange(min(64, audit_idx.numel()), device=device)
        gram_t = gram_preservation_errors(own_t[sampled], scrambled_t[sampled])
        gram_v = gram_preservation_errors(own_v[sampled], scrambled_v[sampled])
        gram_abs = max(gram_t["absolute_max"], gram_v["absolute_max"])
        gram_rel = max(gram_t["scale_normalized_relative_max"], gram_v["scale_normalized_relative_max"])
        gram_elementwise_rel = max(gram_t["elementwise_relative_max_near_zero_sensitive"],
                                   gram_v["elementwise_relative_max_near_zero_sensitive"])
        qa["signed_permutation_gram_abs_max"] = max(qa["signed_permutation_gram_abs_max"], gram_abs)
        qa["signed_permutation_gram_rel_max"] = max(qa["signed_permutation_gram_rel_max"], gram_rel)
        qa["signed_permutation_gram_elementwise_rel_max_near_zero_sensitive"] = max(
            qa.get("signed_permutation_gram_elementwise_rel_max_near_zero_sensitive", 0.0), gram_elementwise_rel
        )
        for epsilon in EPSILONS:
            bank_directions = {
                "OwnStructural": (own_t, own_v), "CoherentDonor": (coherent_t, coherent_v),
                "IndependentDonor": (independent_t, independent_v), "GeometryScrambled": (scrambled_t, scrambled_v),
            }
            for name, (dt, dv) in bank_directions.items():
                if name == "OwnStructural" and repeat > 0:
                    continue
                summary, (ct, cv) = _action_bank_metrics(host, dt, dv, global_text, global_visual,
                    audit_idx, audit_labels, logits[audit_idx.to(device)], epsilon, device)
                bank_values[name][epsilon].append(summary)
                ref_t_norm = audit_global_t.norm(dim=-1, keepdim=True)
                ref_v_norm = audit_global_v.norm(dim=-1, keepdim=True)
                for candidate, reference in ((ct, ref_t_norm), (cv, ref_v_norm)):
                    err = (candidate[:, 1:].norm(dim=-1) - reference).abs() / reference.clamp_min(1e-12)
                    qa["normmatch_max_relative_error"] = max(qa["normmatch_max_relative_error"], float(err.max().item()))
                    qa["action_cardinality_pass"] &= candidate.size(1) == 9
                    zero = make_action_candidates(candidate[:, 0], dt if candidate is ct else dv, 0.0)
                    target = candidate[:, 0:1].expand_as(zero)
                    qa["epsilon_zero_max_abs_error"] = max(qa["epsilon_zero_max_abs_error"], float((zero - target).abs().max().item()))
    def average_geometry(items):
        result = {}
        for mod in ("text", "visual"):
            result[mod] = {}
            for field in ("mean_pairwise_cosine", "mean_absolute_pairwise_cosine", "mean_effective_rank", "median_effective_rank"):
                result[mod][field] = float(np.mean([item[mod][field] for item in items]))
            result[mod]["mean_gram_eigenvalues_descending"] = np.mean(
                [item[mod]["mean_gram_eigenvalues_descending"] for item in items], axis=0
            ).tolist()
            result[mod]["node_count"] = int(items[0][mod]["node_count"])
        return result
    for name in geometries:
        geometries[name] = average_geometry(geometries[name])
    for epsilon in EPSILONS:
        utility_by_epsilon[str(epsilon)] = {}
        for name in bank_values:
            bank_runs = bank_values[name][epsilon]
            utility_by_epsilon[str(epsilon)][name] = {
                metric: {field: float(np.mean([run[metric][field] for run in bank_runs])) for field in bank_runs[0][metric]}
                for metric in ("text", "visual", "joint")
            }
            utility_by_epsilon[str(epsilon)][name]["repeat_sd"] = {
                metric: {field: float(np.std([run[metric][field] for run in bank_runs], ddof=1)) if len(bank_runs) > 1 else 0.0
                         for field in bank_runs[0][metric]} for metric in ("text", "visual", "joint")
            }
    qa["geometry_scramble_matches_own_effective_rank"] = max(
        abs(geometries["GeometryScrambled"][mod]["mean_effective_rank"] - geometries["OwnStructural"][mod]["mean_effective_rank"])
        for mod in ("text", "visual")
    ) < 1e-4
    qa["signed_permutation_gram_relative_error_definition"] = "per sampled node: max absolute Gram-entry error / max absolute original Gram entry; elementwise relative max retained as near-zero-sensitive diagnostic"
    qa["signed_permutation_gram_pass"] = qa["signed_permutation_gram_abs_max"] < 1e-4 and qa["signed_permutation_gram_rel_max"] < 1e-5
    qa["normmatch_pass"] = qa["normmatch_max_relative_error"] < 1e-5
    qa["epsilon_zero_pass"] = qa["epsilon_zero_max_abs_error"] < 1e-6
    qa["self_donor_pass"] = qa["coherent_self_donor_count"] == 0
    return {"geometry": geometries, "utility_by_epsilon": utility_by_epsilon,
            "repeat_count": int(repeats), "control_seeds": control_seed_list, "qa": qa}


def _run_one(dataset: str, host_seed: int, *, phases: set[str], repeats: int,
             predictor_epochs: int, adapter_epochs: int, patience: int,
             device: torch.device, output_root: Path) -> dict[str, Any]:
    data, split_source, _ = load_fixed_data(dataset)
    partitions, partition_meta = stratified_internal_partition(data.train_idx, data.y, seed=PARTITION_SEED)
    assert_disjoint_splits({"HostTrain": partitions["host_train"], "ResponseTrain": partitions["response_train"],
        "Audit": partitions["audit"], "Val": data.val_idx, "Test": data.test_idx})
    hashes = _index_hashes(data, partitions)
    split_reference = _verify_previous_split(dataset, host_seed, hashes)
    device_operator = _operator(data.edge_index, int(data.num_nodes), device)
    host, logits, states_text, states_visual, global_text, global_visual, checkpoint = _load_host(
        data, dataset, host_seed, device_operator, partitions, device
    )
    # Keep the label tensor unmaterialized here; below, only HostTrain, ResponseTrain,
    # Audit, and original-validation index slices are read. Test labels are never sliced.
    labels = data.y
    response_idx, audit_idx = partitions["response_train"], partitions["audit"]
    receiver = _receiver_features(logits, states_text, states_visual, global_text, global_visual)
    context, disagreement, context_meta = build_relation_context(
        data.edge_index.to(device), states_text[0], states_visual[0], global_text, global_visual
    )
    _, degree_bucket, predicted_class = _degree_and_prediction_buckets(data.edge_index, int(data.num_nodes), logits.argmax(-1))
    run: dict[str, Any] = {
        "dataset": dataset, "host_seed": int(host_seed), "data_seed": DATA_SEED,
        "partition_seed": PARTITION_SEED, "split_source": str(split_source),
        "split_hashes": hashes, "split_counts": {"OriginalTrain": int(data.train_idx.numel()),
            "HostTrain": int(partitions["host_train"].numel()), "ResponseTrain": int(response_idx.numel()),
            "Audit": int(audit_idx.numel()), "Val": int(data.val_idx.numel()), "Test": int(data.test_idx.numel())},
        "split_matches_previous_h1": split_reference,
        "partition_metadata": {"method": partition_meta["method"], "small_class_fallback_classes": partition_meta["small_class_fallback_classes"]},
        "checkpoint": checkpoint,
        "protocol_deviations": [],
        "protocol_notes": [
            "P1 and B1 are receiver-only with a zero-filled context block so context and receiver variants share the same input width and parameter count.",
            "Predictor validation allocation is rounded within each ResponseTrain class; singleton classes stay in predictor-train and are listed in split metadata.",
            "Gram QA reports a scale-normalized max relative error; elementwise near-zero-sensitive relative error is retained separately.",
        ],
        "test_labels_read": False, "test_metrics_computed": False,
        "qa": _initial_run_qa(),
        "context": context_meta,
        "receiver_feature_dim": int(receiver.size(-1)), "context_dim": int(context.size(-1)),
    }
    h1_report_path = REPORT_DIR.parent / "h1" / "per_run" / f"{dataset}_seed{host_seed}.json"
    h1_old = json.loads(h1_report_path.read_text(encoding="utf-8"))
    # Confirm the reused frozen checkpoint reproduces the earlier validation selection.
    with torch.no_grad():
        val_ids = data.val_idx.to(device)
        val_labels = labels[data.val_idx].long().to(device)
        val_logits = logits[val_ids]
        restored_val_acc = float((val_logits.argmax(-1) == val_labels).float().mean().item())
        restored_val_ce = float(F.cross_entropy(val_logits, val_labels).item())
    old_host = h1_old["host_training"]
    val_match = abs(restored_val_acc - float(old_host["restored_checkpoint_val_accuracy"])) < 1e-7 and abs(restored_val_ce - float(old_host["restored_checkpoint_val_ce"])) < 1e-6
    run["reused_checkpoint_validation_match"] = {"pass": bool(val_match), "accuracy": restored_val_acc, "ce": restored_val_ce,
        "previous_accuracy": old_host["restored_checkpoint_val_accuracy"], "previous_ce": old_host["restored_checkpoint_val_ce"]}
    run["qa"]["reused_checkpoint_validation_match"] = bool(val_match)
    if not val_match:
        raise AssertionError("reused checkpoint validation metrics differ from H1 report")
    run["qa"].update({"internal_splits_disjoint": True, "split_hashes_match_h1": True,
        "test_labels_read": False, "host_frozen": all(not p.requires_grad for p in host.parameters()),
        "relation_context_finite": bool(torch.isfinite(context).all()),
        "relation_context_shape_pass": context.shape == (int(data.num_nodes), CONTEXT_DIM),
        "audit_target_used_for_training": False})

    if "h11" in phases:
        audit_labels = labels[audit_idx].long().to(device)
        run["h11"] = _run_h11(host, states_text, states_visual, global_text, global_visual, logits,
            audit_idx, audit_labels, data.edge_index, int(data.num_nodes), repeats=repeats, device=device)
        reference_errors = []
        h11_independent = run["h11"]["utility_by_epsilon"]
        for epsilon in EPSILONS:
            new_metrics = h11_independent[str(epsilon)]["IndependentDonor"]
            previous = h1_old["epsilon_results"][str(epsilon)]["shuffled_control"]
            reference_errors.extend([
                abs(new_metrics["text"]["node_oracle_mean"] - previous["text_node_oracle_mean"]),
                abs(new_metrics["text"]["node_specific_headroom"] - previous["text_headroom_mean"]),
                abs(new_metrics["visual"]["node_oracle_mean"] - previous["visual_node_oracle_mean"]),
                abs(new_metrics["visual"]["node_specific_headroom"] - previous["visual_headroom_mean"]),
                abs(new_metrics["joint"]["separate_minus_shared_headroom"] - previous["modality_headroom_mean"]),
            ])
        reference_error = max(reference_errors)
        run["h11"]["qa"]["previous_independent_reference_max_abs_error"] = float(reference_error)
        run["h11"]["qa"]["previous_independent_reference_pass"] = bool(reference_error < 1e-6)
        if reference_error >= 1e-6:
            raise AssertionError("recomputed independent donor bank differs from the previous H1 reference")
    if "h2a" in phases or "h2b" in phases:
        train_ids, val_ids, split_meta = _stratified_response_split(
            response_idx, labels[response_idx], PREDICTOR_SPLIT_SEED
        )
    if "h2a" in phases:
        probe, probe_qa = _run_h2a(host, receiver, context, disagreement, states_text, states_visual,
            global_text, global_visual, logits, labels, response_idx, audit_idx, degree_bucket,
            predicted_class, seed=host_seed, max_epochs=predictor_epochs, patience=patience, device=device)
        run["h2_probe"] = probe
        run["h2_probe_qa"] = probe_qa
        run["qa"].update({key: probe_qa[key] for key in (
            "response_targets_finite", "audit_targets_finite_for_evaluation", "audit_target_used_for_training")})
    if "h2b" in phases:
        residual, residual_qa = _run_h2b(host, receiver, context, states_text, states_visual,
            global_text, global_visual, logits, labels, response_idx, audit_idx, train_ids, val_ids,
            split_meta, degree_bucket, predicted_class, seed=host_seed, max_epochs=adapter_epochs,
            patience=patience, device=device)
        run["h2_residual"] = residual
        run["h2_residual_qa"] = residual_qa
    output_path = output_root / f"{dataset}_seed{host_seed}.json"
    _write_json(output_path, run)
    run["compact_json_path"] = str(output_path)
    return run


def _aggregate_rows(runs: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        dataset, seed = run["dataset"], run["host_seed"]
        if phase == "h11":
            result = run["h11"]
            for epsilon, banks in result["utility_by_epsilon"].items():
                for bank, metrics in banks.items():
                    if bank == "repeat_sd":
                        continue
                    row = {"dataset": dataset, "host_seed": seed, "epsilon": epsilon, "bank": bank}
                    for metric, values in metrics.items():
                        for key, value in values.items():
                            row[f"{metric}_{key}"] = value
                    row.update({f"geometry_{mod}_{key}": value for mod in ("text", "visual")
                        for key, value in result["geometry"][bank][mod].items() if key != "mean_gram_eigenvalues_descending"})
                    rows.append(row)
        elif phase == "h2a":
            for variant in ("P1_ReceiverOnly", "P2_ReceiverPlusRealContext", "P3_ReceiverPlusShuffledContext", "P2_eval_shuffle"):
                metrics = run["h2_probe"][variant]["metrics"]
                rows.append({"dataset": dataset, "host_seed": seed, "variant": variant,
                    "mean_per_dimension_spearman": metrics["mean_per_dimension_spearman"],
                    "mean_per_dimension_sign_balanced_accuracy": metrics["mean_per_dimension_sign_balanced_accuracy"],
                    "selected_utility_mean": metrics["selected_action_utility"]["mean"],
                    "regret_mean": metrics["regret"]["mean"],
                    "parameter_count": run["h2_probe"]["parameter_counts"]["P1_ReceiverOnly"]})
        elif phase == "h2b":
            for variant in ("B0_GlobalFrozen", "B1_ReceiverResidual", "B2_ContextResidual", "B3_ShuffledContextResidual", "B2_eval_shuffle"):
                metrics = run["h2_residual"][variant]["metrics"]
                rows.append({"dataset": dataset, "host_seed": seed, "variant": variant,
                    "ce": metrics["ce"], "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "ce_delta_vs_B0": metrics.get("ce_delta_vs_B0", 0.0),
                    "positive_improvement_fraction": metrics.get("positive_improvement_fraction"),
                    "harmful_fraction": metrics.get("harmful_fraction"),
                    "parameter_count": run["h2_residual"].get("parameter_counts", {}).get(variant)})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(runs: list[dict[str, Any]], phases: set[str], qa: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_run_dir = REPORT_DIR / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        _write_json(per_run_dir / f"{run['dataset']}_seed{run['host_seed']}.json", run)

    def fmt(value: Any, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    if "h11" in phases:
        rows = _aggregate_rows(runs, "h11")
        _write_csv(REPORT_DIR / "h11_metrics.csv", rows)
        lines = ["# R³-MAG H1.1 action-bank control screen", "", "Descriptive analysis only; no H1 pass/fail gate is applied. Test labels and metrics were not read.", "",
            "## Findings", "", "The report compares own structural directions, coherent single-donor trajectories, the previous independent-donor null, and shared signed-permutation geometry scrambles. All banks use eight unique directions per node across the two modalities, nine actions per modality including NOOP, NormMatch, and full joint frozen-host evaluation.", "",
            "Geometry fields report the mean across Audit targets separately for Text and Visual. Utility fields are nats of CE reduction. For each registered epsilon, every dataset/host-seed appears in h11_metrics.csv.", "",
            "## Direction geometry", "", "| Dataset | Seed | Bank | Text effective rank | Visual effective rank | Text mean |cos| | Visual mean |cos| |", "|---|---:|---|---:|---:|---:|---:|"]
        for row in rows:
            if row["epsilon"] != str(EPSILONS[0]):
                continue
            lines.append(f"| {row['dataset']} | {row['host_seed']} | {row['bank']} | {row['geometry_text_mean_effective_rank']:.4f} | {row['geometry_visual_mean_effective_rank']:.4f} | {row['geometry_text_mean_absolute_pairwise_cosine']:.4f} | {row['geometry_visual_mean_absolute_pairwise_cosine']:.4f} |")
        lines += ["", "## Utility comparison", "", "| Dataset | Seed | ε | Bank | Text oracle | Text headroom | Visual oracle | Visual headroom | Joint separate−shared |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['dataset']} | {row['host_seed']} | {row['epsilon']} | {row['bank']} | {row['text_node_oracle_mean']:.5f} | {row['text_node_specific_headroom']:.5f} | {row['visual_node_oracle_mean']:.5f} | {row['visual_node_specific_headroom']:.5f} | {row['joint_separate_minus_shared_headroom']:.5f} |")
        geometry_deltas = []
        for run in runs:
            for modality in ("text", "visual"):
                own = run["h11"]["geometry"]["OwnStructural"][modality]
                independent = run["h11"]["geometry"]["IndependentDonor"][modality]
                geometry_deltas.append((
                    independent["mean_effective_rank"] - own["mean_effective_rank"],
                    independent["mean_absolute_pairwise_cosine"] - own["mean_absolute_pairwise_cosine"],
                ))
        mean_rank_delta = float(np.mean([x[0] for x in geometry_deltas]))
        mean_abs_cos_delta = float(np.mean([x[1] for x in geometry_deltas]))
        metric_map = {(r["dataset"], r["host_seed"], r["epsilon"], r["bank"]): r for r in rows}
        def average_bank_difference(left: str, right: str, field: str) -> float:
            vals = [metric_map[(run["dataset"], run["host_seed"], str(eps), left)][field]
                    - metric_map[(run["dataset"], run["host_seed"], str(eps), right)][field]
                    for run in runs for eps in EPSILONS]
            return float(np.mean(vals))
        text_coherent = average_bank_difference("OwnStructural", "CoherentDonor", "text_node_specific_headroom")
        visual_coherent = average_bank_difference("OwnStructural", "CoherentDonor", "visual_node_specific_headroom")
        joint_coherent = average_bank_difference("OwnStructural", "CoherentDonor", "joint_separate_minus_shared_headroom")
        text_scramble = average_bank_difference("OwnStructural", "GeometryScrambled", "text_node_oracle_mean")
        visual_scramble = average_bank_difference("OwnStructural", "GeometryScrambled", "visual_node_oracle_mean")
        joint_scramble = average_bank_difference("OwnStructural", "GeometryScrambled", "joint_separate_pair_oracle_mean")
        lines += ["", "## Four requested interpretation points", "",
            f"A. Independent minus own effective rank is {mean_rank_delta:+.3f} on average; mean absolute pairwise cosine changes by {mean_abs_cos_delta:+.3f}. Positive rank and negative |cos| shifts indicate a more diverse independent-donor bank.",
            f"B. Own minus coherent-donor headroom averages {text_coherent:+.4f} Text, {visual_coherent:+.4f} Visual, and {joint_coherent:+.4f} joint separate-vs-shared utility.",
            f"C. Own minus geometry-scrambled oracle utility averages {text_scramble:+.4f} Text, {visual_scramble:+.4f} Visual, and {joint_scramble:+.4f} joint pair utility; compare with preserved Gram/effective-rank QA.",
            "D. Higher independent-donor rank/lower correlation together with a narrowed own-vs-coherent gap would make action-bank diversity a plausible contributor to the previous negative structural-minus-independent excess. These controls are descriptive and cannot assign a causal share.", ""]
        lines.insert(-1, f"Observed here, independent-donor rank is much higher and |cos| lower, but coherent donors also beat own directions on average despite nearly matching own-bank rank/correlation. Geometry scrambling preserves own Gram geometry while lowering average single-modality oracle utility. This makes action-bank diversity a plausible part of the independent-null advantage, but the coherent result shows diversity alone does not account for the full gap; donor alignment and host-coordinate sensitivity remain possible contributors.")
        (REPORT_DIR / "h11_report.md").write_text("\n".join(lines), encoding="utf-8")
    if "h2a" in phases:
        rows = _aggregate_rows(runs, "h2a")
        _write_csv(REPORT_DIR / "h2_probe_metrics.csv", rows)
        rowmap = {(row["dataset"], row["host_seed"], row["variant"]): row for row in rows}
        lines = ["# R³-MAG H2-A relation-context response probe", "", "Primary metrics are mean per-dimension Spearman, sign balanced accuracy, predicted-action utility, and within-modality regret. All targets come from ε=0.1 frozen-host own-structural ± actions. P1 is receiver-only with an all-zero padded context block; P1/P2/P3 therefore share exactly the same 2-layer MLP and parameter count. P3 context is shuffled inside ResponseTrain; Audit receives a separate Audit-only shuffle. Audit labels are evaluation-only.", "", "## Audit metrics", "", "| Dataset | Seed | Variant | Mean Spearman | Sign BA | Selected utility | Regret | Parameters |", "|---|---:|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['dataset']} | {row['host_seed']} | {row['variant']} | {fmt(row['mean_per_dimension_spearman'])} | {fmt(row['mean_per_dimension_sign_balanced_accuracy'])} | {row['selected_utility_mean']:.5f} | {row['regret_mean']:.5f} | {row['parameter_count']} |")
        lines += ["", "## Context correspondence contrasts", "", "| Dataset | Seed | P2−P1 utility | P2−P1 regret | P2−P3 Spearman | P2−P3 utility | P2−eval-shuffle utility |", "|---|---:|---:|---:|---:|---:|---:|"]
        for run in runs:
            key = (run["dataset"], run["host_seed"])
            p1 = rowmap[(*key, "P1_ReceiverOnly")]
            p2 = rowmap[(*key, "P2_ReceiverPlusRealContext")]
            p3 = rowmap[(*key, "P3_ReceiverPlusShuffledContext")]
            p2s = rowmap[(*key, "P2_eval_shuffle")]
            lines.append(f"| {key[0]} | {key[1]} | {p2['selected_utility_mean']-p1['selected_utility_mean']:+.5f} | {p2['regret_mean']-p1['regret_mean']:+.5f} | {fmt(p2['mean_per_dimension_spearman']-p3['mean_per_dimension_spearman'])} | {p2['selected_utility_mean']-p3['selected_utility_mean']:+.5f} | {p2['selected_utility_mean']-p2s['selected_utility_mean']:+.5f} |")
        mean_by_variant = {variant: {field: float(np.mean([r[field] for r in rows if r["variant"] == variant]))
            for field in ("mean_per_dimension_spearman", "mean_per_dimension_sign_balanced_accuracy", "selected_utility_mean", "regret_mean")}
            for variant in ("P1_ReceiverOnly", "P2_ReceiverPlusRealContext", "P3_ReceiverPlusShuffledContext", "P2_eval_shuffle")}
        p2m, p1m, p3m, p2sm = (mean_by_variant[k] for k in ("P2_ReceiverPlusRealContext", "P1_ReceiverOnly", "P3_ReceiverPlusShuffledContext", "P2_eval_shuffle"))
        lines += ["", f"Across the nine runs, P2−P1 is {p2m['selected_utility_mean']-p1m['selected_utility_mean']:+.5f} selected utility and {p2m['regret_mean']-p1m['regret_mean']:+.5f} regret; P2−P3 is {p2m['mean_per_dimension_spearman']-p3m['mean_per_dimension_spearman']:+.4f} Spearman and {p2m['selected_utility_mean']-p3m['selected_utility_mean']:+.5f} utility. P2−P2-eval-shuffle is {p2m['selected_utility_mean']-p2sm['selected_utility_mean']:+.5f} utility. These aggregate differences are small; inspect per-run rows for sign stability.", ""]
        lines += ["Under the requested qualitative reading, this probe is H2 Weak for the tested predictor: P2 is close to P1/P3, the predictive rank metrics do not consistently favor real context, and shuffling P2 at Audit barely changes utility. This does not test more expressive relation models or establish that context carries no signal.", ""]
        lines += ["", "## High/low disagreement subgroup", "", "Groups are the bottom/top Audit quartiles of mean neighbor |H0 text cosine − H0 visual cosine|. Deltas are P2−P1.", "", "| Dataset | Seed | Group | Selected utility delta | Regret delta | Nodes |", "|---|---:|---|---:|---:|---:|"]
        for run in runs:
            for group, values in run["h2_probe"]["disagreement_subgroups"].items():
                lines.append(f"| {run['dataset']} | {run['host_seed']} | {group} | {values['P2_minus_P1_selected_action_utility']['mean']:+.5f} | {values['P2_minus_P1_regret']['mean']:+.5f} | {values['node_count']} |")
        lines += ["", "This subgroup is exploratory. No R² or discrete best-hop accuracy is a primary outcome.", ""]
        (REPORT_DIR / "h2_probe_report.md").write_text("\n".join(lines), encoding="utf-8")
    if "h2b" in phases:
        rows = _aggregate_rows(runs, "h2b")
        _write_csv(REPORT_DIR / "h2_residual_metrics.csv", rows)
        lines = ["# R³-MAG H2-B minimal functional residual screen", "", "The host is fully frozen. The adapter predicts bounded Text/Visual residuals and scalar gates, then uses the original modality LayerNorm, fusion, and classifier. B1 uses receiver features plus a zero context block; B2 uses real context; B3 uses a split-restricted shuffled context. Their architectures and parameter counts match. B2-eval-shuffle evaluates the trained B2 on independently shuffled Audit context.", "", "## Audit metrics", "", "| Dataset | Seed | Variant | CE | Accuracy | Macro-F1 | CE delta vs B0 | Positive improvement | Harmful | Parameters |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            pos = "" if row["positive_improvement_fraction"] is None else fmt(row["positive_improvement_fraction"])
            harmful = "" if row["harmful_fraction"] is None else fmt(row["harmful_fraction"])
            lines.append(f"| {row['dataset']} | {row['host_seed']} | {row['variant']} | {row['ce']:.5f} | {row['accuracy']:.5f} | {row['macro_f1']:.5f} | {row['ce_delta_vs_B0']:.5f} | {pos} | {harmful} | {row['parameter_count']} |")
        lines += ["", "## Pairwise contrasts", "", "Negative CE favors the left variant; positive Accuracy/Macro-F1 favors the left variant.", "", "| Dataset | Seed | Contrast | Δ CE | Δ Accuracy | Δ Macro-F1 |", "|---|---:|---|---:|---:|---:|"]
        for run in runs:
            for contrast, values in run["h2_residual"]["pairwise_comparisons"].items():
                lines.append(f"| {run['dataset']} | {run['host_seed']} | {contrast} | {values['ce_left_minus_right']:+.5f} | {values['accuracy_left_minus_right']:+.5f} | {values['macro_f1_left_minus_right']:+.5f} |")
        pair_names = ("B2_minus_B1", "B2_minus_B3", "B2_minus_B2_eval_shuffle")
        means = {name: {metric: float(np.mean([run["h2_residual"]["pairwise_comparisons"][name][metric] for run in runs]))
                        for metric in ("ce_left_minus_right", "accuracy_left_minus_right", "macro_f1_left_minus_right")}
                 for name in pair_names}
        b2_rows = [row for row in rows if row["variant"] == "B2_ContextResidual"]
        b2_positive = float(np.mean([float(row["positive_improvement_fraction"]) for row in b2_rows]))
        b2_harmful = float(np.mean([float(row["harmful_fraction"]) for row in b2_rows]))
        lines += ["", f"Across the nine runs, B2−B1 averages ΔCE {means['B2_minus_B1']['ce_left_minus_right']:+.5f}, ΔAccuracy {means['B2_minus_B1']['accuracy_left_minus_right']:+.5f}, and ΔMacro-F1 {means['B2_minus_B1']['macro_f1_left_minus_right']:+.5f}. B2−B3 averages ΔCE {means['B2_minus_B3']['ce_left_minus_right']:+.5f}; B2−eval-shuffle averages ΔCE {means['B2_minus_B2_eval_shuffle']['ce_left_minus_right']:+.5f}. B2's per-node ΔCE is positive for {b2_positive:.1%} and harmful for {b2_harmful:.1%} of Audit nodes on average. Read alongside the per-run variation; the small differences do not establish a reliable context effect.", "",
            "Taken together with H2-A, the residual screen also fits H2 Weak for this minimal adapter: B2 is very close to receiver-only and shuffled controls, with no clear real-context correspondence advantage. Adapters can reduce mean CE relative to B0 while helping fewer than half of nodes, so that global reduction alone is not evidence that relation context supplied it.", "",
            "Per-modality gate mean/median, residual norm ratio, and per-node ΔCE positive/harmful fractions are retained in per-run JSON. These are mechanism diagnostics and no automatic significance gate is applied.", ""]
        (REPORT_DIR / "h2_residual_report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(REPORT_DIR / "qa_report.json", qa)
    readme = """# R³-MAG H1.1 and H2 preflight screen

This branch adds a compact follow-up to the H1 frozen-host audit. It does not implement H3 or the final R³-MAG model.

## Fixed protocol

- Movies and Grocery use dataset seed 42; ele-fashion uses its official split. Internal HostTrain/ResponseTrain/Audit partition seed is 20261006. Frozen host seeds are 42, 43, and 44. Every internal and external split index hash is checked against the prior H1 per-run record before evaluation.
- Existing `outputs/r3mag_design_freeze/h1/<dataset>/seed<seed>/host_best_val.pt` checkpoints are reused. A missing checkpoint is retrained only with the previous H1 host function and settings. Per-run JSON records path and reuse status.
- Test labels and metrics are untouched. Test indices are hashed as metadata only. Original validation labels are read solely to verify the reused checkpoint's prior validation metrics. ResponseTrain labels are used only for response targets, predictor split stratification, and predictor/adapter fit; Audit labels are indexed after the host is frozen and only for evaluation.
- H1.1 runs 20 deterministic control seeds and ε=0.1/0.2. Coherent donor uses one node donor for both modalities and all orders. Independent donor retains the previous H1 sampler. Geometry scramble uses one shared signed coordinate permutation across modalities and targets per repeat.
- H2-A uses ε=0.1 own-structural slope targets, training-only per-dimension standardization, class-stratified ResponseTrain 80/20 split seed 20261007, Huber loss, and a two-layer 128-unit MLP. P1/P2/P3 are capacity matched by zero-padding context for P1. P3 and P2-eval-shuffle use deterministic context derangements confined to ResponseTrain and Audit separately.
- H2-B trains only a 128-unit adapter with AdamW (1e-3, weight decay 5e-4), validation CE early stopping, up to 500 epochs and patience 50. B1/B2/B3 have equal architecture and parameter count. B0 uses the frozen host directly.

## Protocol deviations and implementation notes

No deviations were made to the registered datasets/splits, H1 host architecture/settings, host seeds, H1.1 epsilons/repeat count, or H2 optimizer/loss/epoch caps. The only implementation convention is zero-padding the context block for P1/B1 to keep parameter counts equal while supplying those variants no relational information. ResponseTrain's class-stratified 20% validation count is rounded per class; singleton classes remain in predictor-train and are listed in each run's split metadata. Matching fallbacks, singleton shuffle maps, and any checkpoint retraining are recorded per run.

## QA and outputs

See `qa_report.json` for split/checkpoint, donor, signed-permutation, Gram, NormMatch, action-count, split-boundary, context-finiteness, equal-capacity, audit-label-boundary, and frozen-host checks. Detailed compact run records are in `per_run/`. Large intermediate outputs are ignored under `outputs/r3mag_design_freeze/h11_h2/`.

Smoke mode uses Movies/seed42, two H1.1 controls, and short H2 optimization. Smoke results are implementation QA only and do not enter tracked formal reports. Formal reports list all negative results and protocol deviations without a mechanical all-seeds pass/fail gate.

Run from the repository root in `yhf_env`:

```bash
PYTHONPATH=. python -m pytest -q tests/test_r3mag_response_core.py tests/test_r3mag_h11_h2.py
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen --smoke --phase h11
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen --smoke --phase h2a
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen --smoke --phase h2b
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen
```
"""
    (REPORT_DIR / "README.md").write_text(readme, encoding="utf-8")


def _write_qa(runs: list[dict[str, Any]], phases: set[str], smoke: bool) -> dict[str, Any]:
    per_run = {}
    all_pass = True
    expected_false = {"test_labels_read", "test_metrics_computed", "audit_target_used_for_training",
                      "audit_targets_used_for_training", "audit_labels_used_for_training"}

    def collect_flags(value: Any, key: str = "") -> list[bool]:
        if isinstance(value, bool):
            return [not value if key in expected_false else value]
        if isinstance(value, dict):
            return [flag for child_key, child in value.items() for flag in collect_flags(child, str(child_key))]
        if isinstance(value, list):
            return [flag for child in value for flag in collect_flags(child, key)]
        return []

    for run in runs:
        qa = dict(run["qa"])
        if "h11" in run:
            qa["h11"] = run["h11"]["qa"]
        if "h2_probe_qa" in run:
            qa["h2a"] = run["h2_probe_qa"]
            qa["p1_p2_p3_equal_parameters"] = len(set(run["h2_probe"]["parameter_counts"].values())) == 1
        if "h2_residual_qa" in run:
            qa["h2b"] = run["h2_residual_qa"]
            qa["b1_b2_b3_equal_parameters"] = len(set(run["h2_residual"]["parameter_counts"].values())) == 1
        per_run[f"{run['dataset']}_seed{run['host_seed']}"] = qa
        flags = collect_flags(qa)
        all_pass &= all(flags)
    return {"smoke_only": bool(smoke), "phases": sorted(phases), "all_assertions_pass": bool(all_pass),
            "run_count": len(runs), "per_run": per_run,
            "test_labels_read": False, "test_metrics_computed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="R3-MAG H1.1 and H2 frozen-host context screen")
    parser.add_argument("--smoke", action="store_true", help="Movies/seed42 only with short optimization and two H1.1 repeats")
    parser.add_argument("--phase", choices=("h11", "h2a", "h2b", "all"), default="all")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--predictor-epochs", type=int, default=None)
    parser.add_argument("--adapter-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    smoke = bool(args.smoke)
    datasets = ["Movies"] if smoke else args.datasets or list(DATASETS)
    seeds = [42] if smoke else args.seeds or list(HOST_SEEDS)
    repeats = args.repeats or (2 if smoke else 20)
    predictor_epochs = args.predictor_epochs or (3 if smoke else 500)
    adapter_epochs = args.adapter_epochs or (3 if smoke else 500)
    patience = args.patience or (3 if smoke else 50)
    if repeats < 1 or (not smoke and repeats < 20):
        raise ValueError("formal H1.1 requires at least 20 repeats")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    phases = {"h11", "h2a", "h2b"} if args.phase == "all" else {args.phase}
    output_root = OUTPUT_DIR / ("smoke" if smoke else "per_run")
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for dataset in datasets:
        for seed in seeds:
            print(f"[R3MAG H11/H2] dataset={dataset} seed={seed} phases={','.join(sorted(phases))} device={device}", flush=True)
            run = _run_one(dataset, seed, phases=phases, repeats=repeats, predictor_epochs=predictor_epochs,
                           adapter_epochs=adapter_epochs, patience=patience, device=device, output_root=output_root)
            runs.append(run)
            print(f"[R3MAG H11/H2] complete dataset={dataset} seed={seed}", flush=True)
    qa = _write_qa(runs, phases, smoke)
    if smoke:
        _write_json(output_root / "qa_report.json", qa)
        print(f"Smoke completed: {output_root}; QA pass={qa['all_assertions_pass']}", flush=True)
    elif datasets == list(DATASETS) and seeds == list(HOST_SEEDS):
        _write_reports(runs, phases, qa)
        print(f"Formal reports written to {REPORT_DIR}; QA pass={qa['all_assertions_pass']}", flush=True)
    else:
        _write_json(output_root / "qa_report.json", qa)
        print(f"Subset run saved to {output_root}; compact tracked aggregate not rewritten", flush=True)


if __name__ == "__main__":
    main()
