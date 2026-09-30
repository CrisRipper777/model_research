from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

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
    NeutralOneHopProbe,
    ProbeData,
    load_probe_data,
    read_train_val_indices,
)


OPERATORS = ("smooth", "absdiff", "product")
MODALITIES = ("text", "visual")
EDGE_BATCH_SIZE = 65536


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def operator_message(operator: str, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
    """Apply one of the three fixed, parameter-free edge transforms."""
    if operator == "smooth":
        return h_src
    if operator == "absdiff":
        return torch.abs(h_src - h_dst)
    if operator == "product":
        return h_src * h_dst
    raise ValueError(f"unknown fixed operator: {operator}")


def operator_context(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    operator: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform incoming mean of transformed messages; graph must have no self loops."""
    src, dst = edge_index
    if src.numel() and torch.any(src == dst):
        raise ValueError("operator_context requires self-loops to be removed")
    degree = torch.bincount(dst, minlength=num_nodes)
    context_sum = torch.zeros_like(h)
    if src.numel():
        messages = operator_message(operator, h[src], h[dst])
        context_sum.index_add_(0, dst, messages)
    context = context_sum / degree.clamp_min(1).to(h.dtype).unsqueeze(-1)
    return context, degree


def operator_features(
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    edge_index: torch.Tensor,
    operator: str,
) -> torch.Tensor:
    c_t, _ = operator_context(h_t, edge_index, h_t.size(0), operator)
    c_v, _ = operator_context(h_v, edge_index, h_v.size(0), operator)
    return torch.cat([h_t, h_v, c_t, c_v], dim=-1)


def selected_validation_edges(data: ProbeData, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the common non-self directed edge set whose destination is validation."""
    edge_index = data.edge_index.to(device)
    src, dst = edge_index
    if src.numel() and torch.any(src == dst):
        raise AssertionError("physical graph contains self loops")
    validation = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    validation[data.val_idx.to(device)] = True
    keep = validation[dst]
    src, dst = src[keep], dst[keep]
    degree = torch.bincount(edge_index[1], minlength=data.num_nodes)
    if src.numel() == 0:
        raise ValueError(f"{data.name}: no incoming directed edges to validation nodes")
    if torch.any(degree[dst] < 1):
        raise AssertionError("a selected directed edge has zero original degree")
    return src, dst, degree


def load_frozen_h0(
    data: ProbeData,
    checkpoint_path: Path,
    device: torch.device,
    reference_performance_path: Path | None = None,
) -> tuple[NeutralOneHopProbe, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load only the semantic-only P0 checkpoint and freeze its full network."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"semantic-only checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bool(checkpoint.get("contextual", True)):
        raise ValueError(f"expected a semantic-only checkpoint: {checkpoint_path}")
    model = NeutralOneHopProbe(
        data.x_t.size(1), data.x_v.size(1), data.num_classes,
        int(checkpoint.get("hidden_dim", HIDDEN_DIM)), contextual=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    x_t, x_v = data.x_t.to(device), data.x_v.to(device)
    with torch.no_grad():
        h_t = model.norm_t(model.proj_t(x_t)).detach()
        h_v = model.norm_v(model.proj_v(x_v)).detach()
        z_sem = torch.cat([h_t, h_v], dim=-1)
        logits = model.head(z_sem)
        val_idx = data.val_idx.to(device)
        val_labels = data.labels.to(device)[val_idx]
        val_logits = logits[val_idx]
        metrics = {
            "val_acc": float((val_logits.argmax(-1) == val_labels).float().mean().item()),
            "val_macro_f1": float(f1_score(
                val_labels.cpu().numpy(), val_logits.argmax(-1).cpu().numpy(),
                labels=list(range(data.num_classes)), average="macro", zero_division=0,
            )),
            "val_ce": float(F.cross_entropy(val_logits, val_labels).item()),
        }
    recorded = checkpoint.get("metrics", {})
    checkpoint_diffs = {
        key: float(metrics[key] - recorded[key])
        for key in metrics if key in recorded
    }
    reference_diffs: dict[str, float] = {}
    if reference_performance_path is not None and reference_performance_path.is_file():
        performance = pd.read_csv(reference_performance_path)
        match = performance[
            (performance["probe"] == "semantic_only")
            & (performance["dataset"] == data.name)
            & (performance["seed"].astype(int) == int(checkpoint["seed"]))
        ]
        if len(match) != 1:
            raise ValueError(f"expected one committed semantic-only metric row in {reference_performance_path}")
        row = match.iloc[0]
        ref_columns = {"val_acc": "val_acc", "val_macro_f1": "val_macro_f1", "val_ce": "val_ce"}
        reference_diffs = {key: float(metrics[key] - float(row[col])) for key, col in ref_columns.items()}
    max_abs_diff = max((abs(v) for v in (*checkpoint_diffs.values(), *reference_diffs.values())), default=0.0)
    if max_abs_diff > 2e-5:
        raise AssertionError(
            f"semantic checkpoint reproduction differs from committed P0 metrics: max_abs_diff={max_abs_diff}"
        )
    details = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_seed": int(checkpoint["seed"]),
        "checkpoint_best_epoch": int(checkpoint.get("best_epoch", -1)),
        "reproduced_metrics": metrics,
        "checkpoint_metric_differences": checkpoint_diffs,
        "p0_csv_metric_differences": reference_diffs,
        "max_abs_metric_difference": max_abs_diff,
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
    }
    return model, h_t, h_v, details


def _metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    prediction = logits.argmax(-1)
    return {
        "val_acc": float((prediction == labels).float().mean().item()),
        "val_macro_f1": float(f1_score(
            labels.detach().cpu().numpy(), prediction.detach().cpu().numpy(),
            labels=list(range(num_classes)), average="macro", zero_division=0,
        )),
        "val_ce": float(F.cross_entropy(logits, labels).item()),
    }


def fit_operator_head(
    features: torch.Tensor,
    data: ProbeData,
    device: torch.device,
    seed: int,
    operator: str,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_epoch: int = MIN_EPOCH,
) -> tuple[nn.Linear, dict[str, Any]]:
    """Train only a new linear head on a frozen operator feature matrix."""
    set_seed(seed)
    head = nn.Linear(features.size(1), data.num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    labels = data.labels.to(device)
    train_idx, val_idx = data.train_idx.to(device), data.val_idx.to(device)
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    x = features.detach()
    if x.requires_grad:
        raise AssertionError("operator features must be frozen before classifier training")
    best_acc, best_epoch, wait = -1.0, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(x)
        loss = F.cross_entropy(logits[train_idx], train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        head.eval()
        with torch.no_grad():
            current = _metrics(head(x[val_idx]), val_labels, data.num_classes)
        if current["val_acc"] > best_acc:
            best_acc = current["val_acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_metrics = current
            wait = 0
        elif epoch >= min_epoch:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{operator}: head training produced no validation-selected state")
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        final_metrics = _metrics(head(x[val_idx]), val_labels, data.num_classes)
    details = {
        "operator": operator,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        **final_metrics,
    }
    return head, details


def _operator_removal_logits_fast(
    head: nn.Linear,
    full_logits_by_node: torch.Tensor,
    h_src: torch.Tensor,
    h_dst: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    modality: str,
    operator: str,
    hidden_dim: int,
) -> torch.Tensor:
    message = operator_message(operator, h_src[src], h_dst[dst]).double()
    start = 2 * hidden_dim if modality == "text" else 3 * hidden_dim
    weight = head.weight[:, start : start + hidden_dim].double()
    effect = (message / degree[dst].double().unsqueeze(-1)) @ weight.T
    return full_logits_by_node[dst] - effect


def _operator_removal_logits_brute(
    head: nn.Linear,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    c_t: torch.Tensor,
    c_v: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    modality: str,
    operator: str,
    hidden_dim: int,
) -> torch.Tensor:
    h_src = h_t if modality == "text" else h_v
    h_dst = h_t if modality == "text" else h_v
    context = c_t if modality == "text" else c_v
    start = 2 * hidden_dim if modality == "text" else 3 * hidden_dim
    rows: list[torch.Tensor] = []
    for source, target in zip(src.tolist(), dst.tolist()):
        z = torch.cat([h_t[target].double(), h_v[target].double(), c_t[target].double(), c_v[target].double()])
        message = operator_message(operator, h_src[source], h_dst[target]).double()
        z[start : start + hidden_dim] -= message / degree[target].double()
        rows.append(F.linear(z.unsqueeze(0), head.weight.double(), head.bias.double()).squeeze(0))
    return torch.stack(rows) if rows else h_t.new_empty((0, head.out_features), dtype=torch.float64)


@torch.no_grad()
def validate_operator_fast_vs_brute(
    head: nn.Linear,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    c_t: torch.Tensor,
    c_v: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    operator: str,
) -> dict[str, float]:
    hidden_dim = h_t.size(1)
    features = torch.cat([h_t, h_v, c_t, c_v], dim=-1).double()
    full_logits = F.linear(features, head.weight.double(), head.bias.double())
    errors: dict[str, float] = {}
    for modality, h_src, h_dst in (("text", h_t, h_t), ("visual", h_v, h_v)):
        fast = _operator_removal_logits_fast(
            head, full_logits, h_src, h_dst, src, dst, degree, modality, operator, hidden_dim,
        )
        brute = _operator_removal_logits_brute(
            head, h_t, h_v, c_t, c_v, src, dst, degree, modality, operator, hidden_dim,
        )
        max_error = float((fast - brute).abs().max().item())
        if not torch.allclose(fast, brute, rtol=1e-7, atol=1e-8):
            raise AssertionError(f"{operator}/{modality}: fast vs brute logits error {max_error}")
        errors[f"{modality}_logit_max_abs"] = max_error
    return errors


@torch.no_grad()
def build_operator_edge_table(
    data: ProbeData,
    h_t: torch.Tensor,
    h_v: torch.Tensor,
    heads: dict[str, nn.Linear],
    device: torch.device,
    p0_edge_path: Path | None = None,
    batch_size: int = EDGE_BATCH_SIZE,
    correctness_check: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build one wide aligned row per physical directed message."""
    src_all, dst_all, degree = selected_validation_edges(data, device)
    labels = data.labels.to(device)
    h_by_modality = {"text": h_t, "visual": h_v}
    table = pd.DataFrame({
        "dataset": data.name,
        "seed": int(getattr(data, "seed", -1)),
        "src": src_all.cpu().numpy(),
        "dst": dst_all.cpu().numpy(),
        "dst_degree": degree[dst_all].cpu().numpy(),
        "dst_label": labels[dst_all].cpu().numpy(),
    })
    # The source seed is attached by the caller to avoid adding a field to ProbeData.
    correctness_errors: dict[str, float] = {}
    for operator in OPERATORS:
        head = heads[operator]
        c_t, _ = operator_context(h_t, data.edge_index.to(device), data.num_nodes, operator)
        c_v, _ = operator_context(h_v, data.edge_index.to(device), data.num_nodes, operator)
        z = torch.cat([h_t, h_v, c_t, c_v], dim=-1)
        logits_by_node = F.linear(z.double(), head.weight.double(), head.bias.double())
        full_by_edge = logits_by_node[dst_all]
        full_ce_parts: list[np.ndarray] = []
        utility_parts: dict[str, list[np.ndarray]] = {
            f"u_raw_{operator}_{modality}": [] for modality in MODALITIES
        }
        relative_parts: dict[str, list[np.ndarray]] = {
            f"u_relative_{operator}_{modality}": [] for modality in MODALITIES
        }
        removed_ce_parts: dict[str, list[np.ndarray]] = {
            f"ce_removed_{operator}_{modality}": [] for modality in MODALITIES
        }
        ce_full_parts: list[np.ndarray] = []
        for start in range(0, src_all.numel(), batch_size):
            sl = slice(start, start + batch_size)
            src, dst = src_all[sl], dst_all[sl]
            y = labels[dst]
            logits_full = full_by_edge[sl]
            ce_full = F.cross_entropy(logits_full, y, reduction="none")
            ce_full_parts.append(ce_full.cpu().numpy())
            for modality in MODALITIES:
                h = h_by_modality[modality]
                c = c_t if modality == "text" else c_v
                removed_logits = _operator_removal_logits_fast(
                    head, logits_by_node, h, h, src, dst, degree,
                    modality, operator, h_t.size(1),
                )
                ce_removed = F.cross_entropy(removed_logits, y, reduction="none")
                raw = ce_removed - ce_full
                key = f"{operator}_{modality}"
                utility_parts[f"u_raw_{key}"].append(raw.cpu().numpy())
                relative_parts[f"u_relative_{key}"].append((raw / ce_full.clamp_min(1e-12)).cpu().numpy())
                removed_ce_parts[f"ce_removed_{key}"].append(ce_removed.cpu().numpy())
        ce_full_all = np.concatenate(ce_full_parts)
        table[f"ce_full_{operator}"] = ce_full_all
        for key, chunks in utility_parts.items():
            table[key] = np.concatenate(chunks)
        for key, chunks in relative_parts.items():
            table[key] = np.concatenate(chunks)
        for key, chunks in removed_ce_parts.items():
            table[key] = np.concatenate(chunks)
        if correctness_check:
            n = min(24, src_all.numel())
            errors = validate_operator_fast_vs_brute(
                head, h_t, h_v, c_t, c_v, src_all[:n], dst_all[:n], degree, operator,
            )
            correctness_errors.update({f"{operator}_{k}": v for k, v in errors.items()})

    if p0_edge_path is not None:
        prior = pd.read_csv(p0_edge_path, usecols=["src", "dst"])
        current_keys = table[["src", "dst"]].astype("int64")
        prior_keys = prior[["src", "dst"]].astype("int64")
        if len(prior_keys) != len(current_keys) or not prior_keys.equals(current_keys.reset_index(drop=True)):
            raise AssertionError("P1.2 physical validation edge set/order differs from the P0 evidence")
    return table, correctness_errors


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


def run_one(
    dataset: str,
    seed: int,
    output_root: Path,
    data_root: str,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run three frozen-H0 operator heads for one dataset and seed."""
    data = load_probe_data(dataset, seed, data_root)
    if hasattr(data, "test_idx"):
        raise AssertionError("P1.2 data object must not expose test indices")
    data.seed = seed
    p0_root = Path("outputs/p0p1_propagation_heterogeneity")
    p0_run = p0_root / "runs" / dataset / f"seed_{seed}"
    semantic_ckpt = p0_root / "checkpoints" / "runs" / dataset / f"seed_{seed}_semantic_only.pt"
    semantic_model, h_t, h_v, h0_record = load_frozen_h0(
        data, semantic_ckpt, device, p0_run / "probe_performance.csv",
    )
    if not all(not p.requires_grad and p.grad is None for p in semantic_model.parameters()):
        raise AssertionError("P0 semantic checkpoint parameters are not fully frozen")
    frozen_before = {k: v.detach().cpu().clone() for k, v in semantic_model.state_dict().items()}
    edge_index = data.edge_index.to(device)
    src, dst, degree = selected_validation_edges(data, device)
    heads: dict[str, nn.Linear] = {}
    features: dict[str, torch.Tensor] = {}
    perf_rows: list[dict[str, Any]] = []
    checkpoint_dir = output_root / "checkpoints" / ("smoke" if smoke else "runs") / dataset
    for operator in OPERATORS:
        features[operator] = operator_features(h_t, h_v, edge_index, operator).detach()
        if features[operator].shape != (data.num_nodes, 4 * HIDDEN_DIM):
            raise AssertionError(f"{operator}: unexpected feature shape {tuple(features[operator].shape)}")
        if not torch.equal(features[operator][:, : 2 * HIDDEN_DIM], torch.cat([h_t, h_v], dim=-1)):
            raise AssertionError(f"{operator}: common frozen H0 differs")
        head, metrics = fit_operator_head(
            features[operator], data, device, seed, operator,
            max_epochs=min(max_epochs, 3) if smoke else max_epochs,
            patience=min(PATIENCE, max(1, min(max_epochs, 3) // 2)) if smoke else PATIENCE,
            min_epoch=min(MIN_EPOCH, min(max_epochs, 3)) if smoke else MIN_EPOCH,
        )
        heads[operator] = head
        perf_rows.append({"dataset": dataset, "seed": seed, **metrics})
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "dataset": dataset, "seed": seed, "operator": operator,
            "h0_checkpoint": str(semantic_ckpt), "best_epoch": metrics["best_epoch"],
            "metrics": metrics, "head_state": head.state_dict(),
            "feature_dim": int(features[operator].size(1)),
            "test_evaluation": False,
        }, checkpoint_dir / f"seed_{seed}_{operator}_head.pt")

    for key, value in semantic_model.state_dict().items():
        if not torch.equal(value.detach().cpu(), frozen_before[key]):
            raise AssertionError(f"frozen semantic parameter changed: {key}")
    if any(parameter.grad is not None for parameter in semantic_model.parameters()):
        raise AssertionError("gradient appeared on a frozen semantic encoder parameter")
    if any(not torch.equal(features[a][:, : 2 * HIDDEN_DIM], features[b][:, : 2 * HIDDEN_DIM])
           for a in OPERATORS for b in OPERATORS):
        raise AssertionError("operator probes do not share identical H0")
    p0_edges = p0_run / "validation_message_evidence.csv.gz"
    edge_table, correctness = build_operator_edge_table(
        data, h_t, h_v, heads, device, p0_edges, correctness_check=True,
    )
    edge_table["seed"] = seed
    run_root = output_root / ("smoke" if smoke else "runs") / dataset / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    if not smoke:
        edge_table.to_csv(run_root / "operator_edge_utility.csv.gz", index=False, compression="gzip")
        pd.DataFrame(perf_rows).to_csv(run_root / "operator_probe_performance.csv", index=False)
    summary = {
        "dataset": dataset,
        "seed": seed,
        "status": "smoke_passed" if smoke else "completed",
        "nodes": data.num_nodes,
        "directed_graph_edges": int(data.edge_index.size(1)),
        "validation_target_messages": int(len(edge_table)),
        "operators": list(OPERATORS),
        "modalities": list(MODALITIES),
        "semantic_h0": h0_record,
        "head_performance": perf_rows,
        "common_h0_identical": True,
        "frozen_encoder_unchanged": True,
        "no_self_loops": bool(not torch.any(data.edge_index[0] == data.edge_index[1])),
        "same_physical_validation_edge_set": True,
        "fixed_original_degree_denominator": True,
        "fast_vs_brute_logit_max_abs": correctness,
        "test_evaluation": False,
        "link_prediction": False,
    }
    write_json(run_root / "run_result.json", summary)
    del data, h_t, h_v, heads, features, edge_table, semantic_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_correctness_smoke(output_root: Path, data_root: str, device: torch.device) -> dict[str, Any]:
    """Run the requested Movies/42 smoke with short operator-head fits."""
    from src.analysis.p0p1_propagation_probe import read_train_val_indices as guarded_reader

    train, val = guarded_reader({
        "train_idx": torch.tensor([0]), "val_idx": torch.tensor([1]), "test_idx": object(),
    })
    if train.tolist() != [0] or val.tolist() != [1]:
        raise AssertionError("train/validation split guard failed")
    toy_h = torch.tensor([[2.0], [6.0], [10.0]])
    toy_edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    expected_removed = {"smooth": 3.0, "absdiff": 2.0, "product": 30.0}
    for operator, expected in expected_removed.items():
        toy_context, toy_degree = operator_context(toy_h, toy_edges, 3, operator)
        message = operator_message(operator, toy_h[0], toy_h[2])
        removed = toy_context[2] - message / toy_degree[2]
        if toy_degree[2].item() != 2 or not torch.allclose(removed, torch.tensor([expected])):
            raise AssertionError(f"{operator}: fixed original-degree denominator check failed")
    summary = run_one("Movies", 42, output_root, data_root, device, max_epochs=3, smoke=True)
    summary["operator_formula_and_fixed_denominator_check"] = expected_removed
    summary["split_accessor_test_guard"] = "passed; only train_idx and val_idx requested"
    write_json(output_root / "smoke" / "Movies" / "seed_42" / "smoke_result.json", summary)
    return summary


def summarize_probe_performance(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["dataset", "seed", "operator"]).reset_index(drop=True)


__all__ = [
    "DATASETS", "OPERATORS", "MODALITIES", "operator_message", "operator_context",
    "operator_features", "selected_validation_edges", "load_frozen_h0", "fit_operator_head",
    "build_operator_edge_table", "validate_operator_fast_vs_brute", "run_one",
    "run_correctness_smoke", "read_train_val_indices", "summarize_probe_performance",
]


def reconstruct_raw_delta(utility_relative: torch.Tensor | np.ndarray, ce_full: torch.Tensor | np.ndarray):
    """Recover signed raw ΔCE from P0 relative utility and its CE denominator."""
    return utility_relative * ce_full
