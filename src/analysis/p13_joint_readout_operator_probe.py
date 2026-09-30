from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.analysis.p0p1_propagation_probe import (
    DATASETS,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCH,
    PATIENCE,
    WEIGHT_DECAY,
    ProbeData,
    load_probe_data,
    read_train_val_indices,
)
from src.analysis.p11_p12_operator_rescue import (
    MODALITIES,
    OPERATORS,
    load_frozen_h0,
    operator_context,
    operator_message,
    selected_validation_edges,
)


BLOCKS = ("H_T", "H_V", "S_T", "D_T", "P_T", "S_V", "D_V", "P_V")
OPERATOR_BLOCK = {"smooth": "S", "absdiff": "D", "product": "P"}
EDGE_BATCH_SIZE = 32768


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def joint_feature_slices(hidden_dim: int = HIDDEN_DIM) -> dict[str, slice]:
    """Return the fixed P1.3 block layout in the required concatenation order."""
    return {name: slice(i * hidden_dim, (i + 1) * hidden_dim) for i, name in enumerate(BLOCKS)}


def joint_feature_dim(hidden_dim: int = HIDDEN_DIM) -> int:
    return len(BLOCKS) * hidden_dim


def new_joint_head(input_dim: int, num_classes: int, device: torch.device | str) -> nn.Linear:
    """Create the only trainable prediction module allowed in P1.3."""
    if input_dim != joint_feature_dim():
        raise ValueError(f"joint head expects {joint_feature_dim()} input features, got {input_dim}")
    return nn.Linear(input_dim, num_classes, device=device)


def build_joint_features(
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Build one feature matrix using exactly the P1.2 parameter-free operators."""
    if h_t.shape != h_v.shape:
        raise ValueError(f"text/visual H0 shapes differ: {tuple(h_t.shape)} vs {tuple(h_v.shape)}")
    n, hidden_dim = h_t.shape
    if hidden_dim != HIDDEN_DIM:
        raise ValueError(f"P1.3 expects the fixed hidden dimension {HIDDEN_DIM}, got {hidden_dim}")
    contexts: dict[str, torch.Tensor] = {}
    degree_ref: torch.Tensor | None = None
    for modality, h in (("text", h_t), ("visual", h_v)):
        for operator in OPERATORS:
            context, degree = operator_context(h, edge_index, n, operator)
            if degree_ref is None:
                degree_ref = degree
            elif not torch.equal(degree, degree_ref):
                raise AssertionError("operator contexts did not use the same physical graph degrees")
            contexts[f"{OPERATOR_BLOCK[operator]}_{'T' if modality == 'text' else 'V'}"] = context
    assert degree_ref is not None
    features = torch.cat(
        [h_t, h_v, contexts["S_T"], contexts["D_T"], contexts["P_T"],
         contexts["S_V"], contexts["D_V"], contexts["P_V"]],
        dim=-1,
    ).detach()
    expected = (n, joint_feature_dim(hidden_dim))
    if tuple(features.shape) != expected:
        raise AssertionError(f"joint feature shape {tuple(features.shape)} != {expected}")
    if features.requires_grad:
        raise AssertionError("joint H0/context features must be frozen")
    return features, contexts, degree_ref


def context_block_name(operator: str, modality: str) -> str:
    if operator not in OPERATORS or modality not in MODALITIES:
        raise ValueError(f"unknown operator/modality: {operator}/{modality}")
    return f"{OPERATOR_BLOCK[operator]}_{'T' if modality == 'text' else 'V'}"


def modality_sign_disagreement(text_utility: Sequence[float], visual_utility: Sequence[float]) -> dict[str, float]:
    """Return exact three-way sign disagreement and P1.2-compatible positive/nonpositive disagreement."""
    text = np.asarray(text_utility, dtype=float)
    visual = np.asarray(visual_utility, dtype=float)
    if text.shape != visual.shape or text.size == 0:
        raise ValueError("text and visual utilities must be nonempty aligned arrays")
    return {
        "three_way_sign_disagreement": float(np.mean(np.sign(text) != np.sign(visual))),
        "positive_vs_nonpositive_sign_disagreement": float(np.mean((text > 0) != (visual > 0))),
        "text_exact_zero_fraction": float(np.mean(text == 0)),
        "visual_exact_zero_fraction": float(np.mean(visual == 0)),
    }


def preferred_channel_from_utilities(utilities: dict[str, float]) -> str:
    """Choose a positive-best tested operator; none means all are nonpositive."""
    values = [(float(utilities[op]), op) for op in OPERATORS]
    best_value, best_operator = max(values, key=lambda item: item[0])
    return best_operator if best_value > 0.0 else "none"


def summarize_within_node_preferences(
    targets: Sequence[int],
    degrees: Sequence[int],
    preferences: Sequence[str],
    degree_min: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Summarize positive-best operator diversity among eligible target nodes."""
    if not (len(targets) == len(degrees) == len(preferences)):
        raise ValueError("targets, degrees, and preferences must have equal lengths")
    grouped: dict[int, dict[str, Any]] = {}
    for target, degree, preference in zip(targets, degrees, preferences):
        target, degree = int(target), int(degree)
        if degree < degree_min:
            continue
        row = grouped.setdefault(target, {"target": target, "dst_degree": degree, "positive_preferred_edge_count": 0, "positive_functions": set()})
        if row["dst_degree"] != degree:
            raise AssertionError(f"target {target} has inconsistent original degree")
        if preference != "none":
            row["positive_preferred_edge_count"] += 1
            row["positive_functions"].add(str(preference))
    per_target: list[dict[str, Any]] = []
    for target in sorted(grouped):
        row = grouped[target]
        functions = sorted(row["positive_functions"], key=OPERATORS.index)
        per_target.append({
            "target": target,
            "dst_degree": row["dst_degree"],
            "positive_preferred_edge_count": row["positive_preferred_edge_count"],
            "distinct_positive_functions": len(functions),
            "positive_functions": ";".join(functions),
            "multi_function_neighborhood": len(functions) >= 2,
            "no_positive_preference": len(functions) == 0,
        })
    n = len(per_target)
    summary: dict[str, float | int] = {
        "eligible_target_count": n,
        "multi_function_neighborhood_ratio": float(np.mean([x["multi_function_neighborhood"] for x in per_target])) if n else float("nan"),
        "median_distinct_positive_functions_per_target": float(np.median([x["distinct_positive_functions"] for x in per_target])) if n else float("nan"),
        "no_positive_preference_target_fraction": float(np.mean([x["no_positive_preference"] for x in per_target])) if n else float("nan"),
        "median_positive_preferred_edge_count": float(np.median([x["positive_preferred_edge_count"] for x in per_target])) if n else float("nan"),
    }
    return per_target, summary


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    prediction = logits.argmax(-1)
    return {
        "val_acc": float((prediction == labels).float().mean().item()),
        "val_macro_f1": float(f1_score(
            labels.detach().cpu().numpy(), prediction.detach().cpu().numpy(),
            labels=list(range(num_classes)), average="macro", zero_division=0,
        )),
        "val_ce": float(F.cross_entropy(logits, labels).item()),
    }


def fit_joint_head(
    features: torch.Tensor,
    data: ProbeData,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_epoch: int = MIN_EPOCH,
) -> tuple[nn.Linear, dict[str, Any]]:
    """Train exactly one Linear head over the fixed eight-block feature matrix."""
    set_seed(seed)
    if features.shape != (data.num_nodes, joint_feature_dim()):
        raise ValueError(f"unexpected joint feature shape: {tuple(features.shape)}")
    head = new_joint_head(features.size(1), data.num_classes, features.device)
    trainable_linears = [m for m in [head] if isinstance(m, nn.Linear) and any(p.requires_grad for p in m.parameters())]
    if len(trainable_linears) != 1:
        raise AssertionError("training must expose exactly one trainable Linear classifier")
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    labels = data.labels.to(features.device)
    train_idx, val_idx = data.train_idx.to(features.device), data.val_idx.to(features.device)
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    x = features.detach()
    best_acc, best_epoch, wait = -1.0, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x[train_idx]), train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(x[val_idx])
            current = metrics_from_logits(val_logits, val_labels, data.num_classes)
        if current["val_acc"] > best_acc:
            best_acc = current["val_acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            wait = 0
        elif epoch >= min_epoch:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("joint linear-head training produced no validation-selected state")
    head.load_state_dict(best_state)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    with torch.no_grad():
        final_metrics = metrics_from_logits(head(x[val_idx]), val_labels, data.num_classes)
    return head, {
        "dataset": data.name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "feature_dim": int(features.size(1)),
        "trainable_linear_head_count": 1,
        **final_metrics,
    }


def _context_removal_logits_fast(
    head: nn.Linear,
    full_logits_by_node: torch.Tensor,
    h: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    operator: str,
    modality: str,
) -> torch.Tensor:
    hidden_dim = h.size(1)
    block_slice = joint_feature_slices(hidden_dim)[context_block_name(operator, modality)]
    message = operator_message(operator, h[src], h[dst]).double()
    weight_slice = head.weight[:, block_slice].double()
    logit_effect = (message / degree[dst].double().unsqueeze(-1)) @ weight_slice.T
    return full_logits_by_node[dst] - logit_effect


def _context_removal_logits_brute(
    head: nn.Linear,
    features: torch.Tensor,
    h: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    operator: str,
    modality: str,
) -> torch.Tensor:
    hidden_dim = h.size(1)
    block_slice = joint_feature_slices(hidden_dim)[context_block_name(operator, modality)]
    rows: list[torch.Tensor] = []
    for source, target in zip(src.tolist(), dst.tolist()):
        changed = features[target].double().clone()
        message = operator_message(operator, h[source], h[target]).double()
        changed[block_slice] -= message / degree[target].double()
        rows.append(F.linear(changed.unsqueeze(0), head.weight.double(), head.bias.double()).squeeze(0))
    return torch.stack(rows) if rows else features.new_empty((0, head.out_features), dtype=torch.float64)


@torch.no_grad()
def validate_joint_fast_vs_brute(
    head: nn.Linear,
    features: torch.Tensor,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    sample_size: int = 24,
) -> dict[str, float]:
    """Compare exact linear-slice removal with explicit feature edits for all 3×2 blocks."""
    src, dst = src[:sample_size], dst[:sample_size]
    full_logits = F.linear(features.double(), head.weight.double(), head.bias.double())
    max_errors: dict[str, float] = {}
    for operator in OPERATORS:
        for modality, h in (("text", h_t), ("visual", h_v)):
            fast = _context_removal_logits_fast(head, full_logits, h, src, dst, degree, operator, modality)
            brute = _context_removal_logits_brute(head, features, h, src, dst, degree, operator, modality)
            error = float((fast - brute).abs().max().item()) if fast.numel() else 0.0
            if not torch.allclose(fast, brute, rtol=1e-7, atol=1e-8):
                raise AssertionError(f"{operator}/{modality}: fast vs brute max_abs={error}")
            max_errors[f"{operator}_{modality}_logit_max_abs"] = error
    return max_errors


def assert_edge_alignment(
    current: pd.DataFrame,
    p0_edge_path: Path,
    p12_edge_path: Path,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    """Require P1.3 row order, count, keys and original degree to match P0 and P1.2."""
    current_keys = current[["src", "dst", "dst_degree"]].astype("int64").reset_index(drop=True)
    checks: dict[str, Any] = {}
    for label, path in (("p0", p0_edge_path), ("p12", p12_edge_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} validation edge table missing: {path}")
        prior = pd.read_csv(path, usecols=["src", "dst", "dst_degree"])
        prior_keys = prior[["src", "dst", "dst_degree"]].astype("int64").reset_index(drop=True)
        if len(prior_keys) != len(current_keys):
            raise AssertionError(f"{dataset}/{seed}: P1.3 vs {label} edge row count differs ({len(current_keys)} vs {len(prior_keys)})")
        if not prior_keys.equals(current_keys):
            same_set = set(map(tuple, prior_keys.to_numpy())) == set(map(tuple, current_keys.to_numpy()))
            raise AssertionError(f"{dataset}/{seed}: P1.3 vs {label} edge order differs; same triple set={same_set}")
        checks[f"{label}_row_count"] = len(prior_keys)
        checks[f"{label}_ordered_src_dst_degree_match"] = True
        if "dataset" in prior.columns and set(prior.dataset.astype(str)) != {dataset}:
            raise AssertionError(f"{label} edge table dataset metadata mismatch")
        if "seed" in prior.columns and set(prior.seed.astype(int)) != {int(seed)}:
            raise AssertionError(f"{label} edge table seed metadata mismatch")
    checks.update({"dataset": dataset, "seed": int(seed), "current_row_count": len(current_keys), "edge_set_match": True, "edge_order_match": True})
    return checks


def _channel_usage(
    head: nn.Linear,
    features: torch.Tensor,
    val_idx: torch.Tensor,
) -> list[dict[str, Any]]:
    slices = joint_feature_slices()
    rows = []
    with torch.no_grad():
        for block in BLOCKS:
            z = features[val_idx, slices[block]].double()
            weight = head.weight[:, slices[block]].double()
            contribution = F.linear(z, weight, bias=None)
            norms = torch.linalg.vector_norm(contribution, ord=2, dim=-1)
            rows.append({
                "block": block,
                "weight_frobenius_norm": float(torch.linalg.vector_norm(weight).item()),
                "validation_feature_rms": float(torch.sqrt(torch.mean(z.square())).item()),
                "mean_logit_contribution_l2": float(norms.mean().item()),
                "median_logit_contribution_l2": float(norms.median().item()),
                "validation_node_count": int(len(val_idx)),
            })
    return rows


def _whole_channel_ablation(
    head: nn.Linear,
    features: torch.Tensor,
    data: ProbeData,
) -> list[dict[str, Any]]:
    val_idx = data.val_idx.to(features.device)
    labels = data.labels.to(features.device)[val_idx]
    slices = joint_feature_slices()
    with torch.no_grad():
        full_logits = head(features[val_idx])
        full = metrics_from_logits(full_logits, labels, data.num_classes)
        rows = []
        for block in BLOCKS[2:]:
            ablated = features[val_idx].clone()
            ablated[:, slices[block]] = 0
            metrics = metrics_from_logits(head(ablated), labels, data.num_classes)
            rows.append({
                "block": block,
                "full_val_ce": full["val_ce"],
                "ablated_val_ce": metrics["val_ce"],
                "delta_val_ce": metrics["val_ce"] - full["val_ce"],
                "full_val_acc": full["val_acc"],
                "ablated_val_acc": metrics["val_acc"],
                "delta_val_acc": metrics["val_acc"] - full["val_acc"],
                "full_val_macro_f1": full["val_macro_f1"],
                "ablated_val_macro_f1": metrics["val_macro_f1"],
                "delta_val_macro_f1": metrics["val_macro_f1"] - full["val_macro_f1"],
                "refit": False,
            })
    return rows


@torch.no_grad()
def build_joint_edge_table(
    data: ProbeData,
    features: torch.Tensor,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    head: nn.Linear,
    device: torch.device,
    batch_size: int = EDGE_BATCH_SIZE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute one shared full prediction/CE and six single-block edge removals."""
    src_all, dst_all, degree = selected_validation_edges(data, device)
    labels = data.labels.to(device)
    full_logits_by_node = F.linear(features.double(), head.weight.double(), head.bias.double())
    logits_full = full_logits_by_node[dst_all]
    y_all = labels[dst_all]
    ce_full = F.cross_entropy(logits_full, y_all, reduction="none")
    table = pd.DataFrame({
        "dataset": data.name,
        "seed": int(getattr(data, "seed", -1)),
        "src": src_all.cpu().numpy(),
        "dst": dst_all.cpu().numpy(),
        "dst_degree": degree[dst_all].cpu().numpy(),
        "dst_label": y_all.cpu().numpy(),
        "ce_full": ce_full.cpu().numpy(),
    })
    h_by_modality = {"text": h_t, "visual": h_v}
    raw: dict[str, list[np.ndarray]] = {f"u_raw_{op}_{mod}": [] for op in OPERATORS for mod in MODALITIES}
    relative: dict[str, list[np.ndarray]] = {f"u_relative_{op}_{mod}": [] for op in OPERATORS for mod in MODALITIES}
    ce_removed: dict[str, list[np.ndarray]] = {f"ce_removed_{op}_{mod}": [] for op in OPERATORS for mod in MODALITIES}
    for start in range(0, src_all.numel(), batch_size):
        sl = slice(start, start + batch_size)
        src, dst = src_all[sl], dst_all[sl]
        y, full_ce = labels[dst], ce_full[sl]
        logits_full_batch = full_logits_by_node[dst]
        for operator in OPERATORS:
            for modality in MODALITIES:
                removed_logits = _context_removal_logits_fast(
                    head, full_logits_by_node, h_by_modality[modality], src, dst, degree, operator, modality,
                )
                removed_ce = F.cross_entropy(removed_logits, y, reduction="none")
                utility = removed_ce - full_ce
                key = f"{operator}_{modality}"
                raw[f"u_raw_{key}"].append(utility.cpu().numpy())
                relative[f"u_relative_{key}"].append((utility / full_ce.clamp_min(1e-12)).cpu().numpy())
                ce_removed[f"ce_removed_{key}"].append(removed_ce.cpu().numpy())
    for fields in (raw, relative, ce_removed):
        for name, chunks in fields.items():
            table[name] = np.concatenate(chunks) if chunks else np.empty((0,), dtype=float)
    return table, {"full_ce_unique_column": True, "row_count": len(table)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _metric_rows(head: nn.Linear, features: torch.Tensor, data: ProbeData) -> dict[str, float]:
    val_idx = data.val_idx.to(features.device)
    labels = data.labels.to(features.device)[val_idx]
    with torch.no_grad():
        return metrics_from_logits(head(features[val_idx]), labels, data.num_classes)


def run_one(
    dataset: str,
    seed: int,
    output_root: Path,
    data_root: str,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    smoke: bool = False,
) -> dict[str, Any]:
    """Fit one common joint linear head and evaluate aligned edge counterfactuals."""
    data = load_probe_data(dataset, seed, data_root)
    if hasattr(data, "test_idx"):
        raise AssertionError("P1.3 data object unexpectedly exposes test indices")
    data.seed = int(seed)
    p0_root = Path("outputs/p0p1_propagation_heterogeneity")
    p12_root = Path("outputs/p11_p12_operator_rescue")
    p0_run = p0_root / "runs" / dataset / f"seed_{seed}"
    semantic_ckpt = p0_root / "checkpoints" / "runs" / dataset / f"seed_{seed}_semantic_only.pt"
    semantic_model, h_t, h_v, h0_record = load_frozen_h0(
        data, semantic_ckpt, device, p0_run / "probe_performance.csv",
    )
    if not all(not p.requires_grad and p.grad is None for p in semantic_model.parameters()):
        raise AssertionError("P0 semantic checkpoint is not fully frozen")
    frozen_before = {k: v.detach().cpu().clone() for k, v in semantic_model.state_dict().items()}
    h0_before = (h_t.detach().cpu().clone(), h_v.detach().cpu().clone())
    edge_index = data.edge_index.to(device)
    if edge_index.numel() and torch.any(edge_index[0] == edge_index[1]):
        raise AssertionError("physical graph contains self-loops")
    src, dst, degree = selected_validation_edges(data, device)
    if torch.any(degree[dst] < 1):
        raise AssertionError("selected edge has zero original incoming degree")
    features, contexts, context_degree = build_joint_features(h_t, h_v, edge_index)
    if not torch.equal(degree, context_degree):
        raise AssertionError("P1.3 and context calculations disagree on original graph degree")
    if features.size(1) != 8 * HIDDEN_DIM:
        raise AssertionError("P1.3 joint feature dimension is not 8*hidden_dim")

    p0_edges = p0_run / "validation_message_evidence.csv.gz"
    p12_edges = p12_root / "runs" / dataset / f"seed_{seed}" / "operator_edge_utility.csv.gz"
    alignment = assert_edge_alignment(
        pd.DataFrame({"src": src.cpu().numpy(), "dst": dst.cpu().numpy(), "dst_degree": degree[dst].cpu().numpy()}),
        p0_edges, p12_edges, dataset, seed,
    )

    used_epochs = min(max_epochs, 3) if smoke else max_epochs
    used_patience = min(PATIENCE, max(1, used_epochs // 2)) if smoke else PATIENCE
    used_min_epoch = min(MIN_EPOCH, used_epochs) if smoke else MIN_EPOCH
    head, performance = fit_joint_head(
        features, data, seed,
        max_epochs=used_epochs,
        patience=used_patience,
        min_epoch=used_min_epoch,
    )
    if any(parameter.requires_grad or parameter.grad is not None for parameter in semantic_model.parameters()):
        raise AssertionError("P0 encoder became trainable during joint-head fitting")
    for name, value in semantic_model.state_dict().items():
        if not torch.equal(value.detach().cpu(), frozen_before[name]):
            raise AssertionError(f"frozen P0 semantic parameter changed: {name}")
    if not torch.equal(h_t.detach().cpu(), h0_before[0]) or not torch.equal(h_v.detach().cpu(), h0_before[1]):
        raise AssertionError("frozen semantic H0 changed during P1.3")
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise AssertionError("trained joint head was not frozen before counterfactual evaluation")

    table, full_ce_check = build_joint_edge_table(data, features, h_t, h_v, head, device)
    table["seed"] = int(seed)
    alignment_full = assert_edge_alignment(table, p0_edges, p12_edges, dataset, seed)
    fast_errors = validate_joint_fast_vs_brute(head, features, h_t, h_v, src, dst, degree)
    checkpoint_dir = output_root / "checkpoints" / ("smoke" if smoke else "runs") / dataset
    run_dir = output_root / ("smoke" if smoke else "runs") / dataset / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"seed_{seed}_joint_linear_head.pt"
    torch.save({
        "dataset": dataset,
        "seed": int(seed),
        "h0_checkpoint": str(semantic_ckpt),
        "feature_order": list(BLOCKS),
        "feature_dim": int(features.size(1)),
        "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "performance": performance,
        "test_evaluation": False,
    }, checkpoint_path)
    if not smoke:
        table.to_csv(run_dir / "joint_operator_edge_utility.csv.gz", index=False, compression="gzip")
        pd.DataFrame([performance]).to_csv(run_dir / "joint_probe_performance.csv", index=False)
    usage = _channel_usage(head, features, data.val_idx.to(device))
    ablation = _whole_channel_ablation(head, features, data)
    run_record = {
        "dataset": dataset,
        "seed": int(seed),
        "status": "smoke_passed" if smoke else "completed",
        "nodes": data.num_nodes,
        "directed_graph_edges": int(edge_index.size(1)),
        "validation_target_messages": int(len(table)),
        "feature_order": list(BLOCKS),
        "feature_dim": int(features.size(1)),
        "trainable_model": "one Linear(8*hidden_dim, num_classes)",
        "training": {"optimizer": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                     "max_epochs": used_epochs, "patience": used_patience, "min_epoch": used_min_epoch,
                     "gradient_clip": 1.0, "selection": "best validation accuracy"},
        "semantic_h0": h0_record,
        "frozen_encoder_unchanged": True,
        "common_h0_identical_across_contexts": True,
        "no_self_loops": True,
        "fixed_original_degree_denominator": True,
        "edge_alignment": alignment_full,
        "fast_vs_brute_logit_max_abs": fast_errors,
        "full_ce_check": full_ce_check,
        "head_performance": performance,
        "channel_usage": usage,
        "whole_channel_ablation": ablation,
        "validation_labels_used_for_checkpoint_selection_and_counterfactual_utility": True,
        "test_evaluation": False,
        "link_prediction": False,
        "checkpoint": str(checkpoint_path),
    }
    write_json(run_dir / "run_result.json", run_record)
    if smoke:
        write_json(output_root / "smoke" / dataset / f"seed_{seed}" / "smoke_result.json", run_record)
    del data, semantic_model, features, contexts, h_t, h_v, head, table
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_record


def run_correctness_smoke(output_root: Path, data_root: str, device: torch.device) -> dict[str, Any]:
    """Run Movies/42 with a three-epoch head after explicit toy correctness checks."""
    train, val = read_train_val_indices({
        "train_idx": torch.tensor([0]), "val_idx": torch.tensor([1]), "test_idx": object(),
    })
    if train.tolist() != [0] or val.tolist() != [1]:
        raise AssertionError("split accessor used unexpected fields")
    toy = torch.tensor([[2.0], [6.0], [10.0]])
    toy_edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    expected = {"smooth": 3.0, "absdiff": 2.0, "product": 30.0}
    for operator, value in expected.items():
        context, degree = operator_context(toy, toy_edges, 3, operator)
        removed = context[2] - operator_message(operator, toy[0], toy[2]) / degree[2]
        if degree[2].item() != 2 or not torch.allclose(removed, torch.tensor([value])):
            raise AssertionError(f"{operator}: original-degree denominator check failed")
    if preferred_channel_from_utilities({"smooth": 0.0, "absdiff": -0.1, "product": 0.0}) != "none":
        raise AssertionError("preferred-channel none rule failed")
    if preferred_channel_from_utilities({"smooth": -0.1, "absdiff": 0.2, "product": 0.2}) != "absdiff":
        raise AssertionError("preferred-channel argmax tie order is not deterministic")
    target_rows, diversity = summarize_within_node_preferences(
        [1, 1, 1, 2, 2], [5, 5, 5, 4, 4], ["smooth", "absdiff", "none", "smooth", "product"],
    )
    if len(target_rows) != 1 or target_rows[0]["distinct_positive_functions"] != 2 or diversity["multi_function_neighborhood_ratio"] != 1.0:
        raise AssertionError("within-node functional diversity calculation failed")
    run = run_one("Movies", 42, output_root, data_root, device, max_epochs=3, smoke=True)
    run["smoke_checks"] = {
        "split_accessor_no_test_access": "passed",
        "fixed_original_degree_denominator": expected,
        "preferred_channel_none_logic": "passed",
        "within_node_diversity": "passed",
        "edge_alignment_p0_p12": "passed",
        "all_six_fast_vs_brute": "passed",
        "joint_feature_dim": run["feature_dim"],
        "frozen_h0": run["frozen_encoder_unchanged"],
    }
    write_json(output_root / "smoke" / "Movies" / "seed_42" / "smoke_result.json", run)
    return run


__all__ = [
    "DATASETS", "OPERATORS", "MODALITIES", "BLOCKS", "joint_feature_slices", "joint_feature_dim", "new_joint_head",
    "build_joint_features", "context_block_name", "modality_sign_disagreement", "preferred_channel_from_utilities",
    "summarize_within_node_preferences", "fit_joint_head", "validate_joint_fast_vs_brute",
    "assert_edge_alignment", "build_joint_edge_table", "run_one", "run_correctness_smoke", "write_json",
]
