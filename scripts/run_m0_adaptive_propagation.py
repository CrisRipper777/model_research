from __future__ import annotations

import argparse
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
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_utils import ensure_edge_index, preprocess_edge_index
from src.data.loaders import (
    _load_dgl_graph,
    _load_numpy_feature,
    _load_torch_tensor,
    _split_modalities_from_joint,
    resolve_path,
)
from src.data.types import MAGData
from src.models.adaptive_prop_m0 import (
    Model,
    common_parameter_names,
    neighborhood_shuffle_indices,
    parameter_counts,
    tie_modality_controls,
)
from src.tasks.common import build_optimizer, scheduler_step
from src.utils.seeds import set_seed

DATASETS = ("Movies", "Grocery", "ele-fashion")
VARIANTS = ("semantic", "uniform", "extent", "single_basis", "multi_basis")
SEEDS = (42, 43, 44)
OUT_DIR = PROJECT_ROOT / "outputs" / "m0_adaptive_propagation_screen"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temp.replace(path)


def make_config(dataset: str, seed: int, variant: str, device: str, epochs: int | None = None):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=adaptive_prop_m0",
        f"model.variant={variant}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        "task.evaluate_test=false",
    ]
    if epochs is not None:
        overrides.append(f"task.epochs={int(epochs)}")
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def load_m0_data(cfg, seed: int) -> MAGData:
    """Load the frozen NC graph while requesting only train/validation split fields."""
    ds = cfg.dataset
    source = str(ds.source).lower()
    if source == "magb":
        raw_edge, labels, num_nodes = _load_dgl_graph(ds.graph_path)
        x_t = _load_numpy_feature(ds.text_feat_path, ds.get("feature_dtype", "float32"))
        x_v = _load_numpy_feature(ds.image_feat_path, ds.get("feature_dtype", "float32"))
        if x_t.size(0) != num_nodes or x_v.size(0) != num_nodes:
            raise ValueError(f"{ds.name}: modality feature rows do not match graph nodes")
        x = torch.cat([x_t, x_v], dim=-1).contiguous()
        split_path = resolve_path(ds.nc_split_path)
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Frozen NC split is missing: {split_path}. M0 will not generate a new split."
            )
        split = torch.load(split_path, map_location="cpu", weights_only=False, mmap=True)
        train_idx = torch.as_tensor(split["train_idx"], dtype=torch.long).contiguous()
        val_idx = torch.as_tensor(split["val_idx"], dtype=torch.long).contiguous()
        edge_index = preprocess_edge_index(
            raw_edge,
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=bool(ds.get("add_self_loops", False)),
        )
        return MAGData(
            name=str(ds.name), source=source, task="nc", x=x, x_t=x_t, x_i=x_v,
            edge_index=edge_index, y=labels, train_idx=train_idx, val_idx=val_idx,
            test_idx=None, num_nodes=num_nodes, num_classes=int(ds.num_classes),
            info={"nc_split_path": str(split_path), "test_split_field_read": False},
        )
    if source == "mmgraph":
        x = _load_torch_tensor(ds.joint_feat_path, ds.get("feature_dtype", "float32"))
        num_nodes = int(x.size(0))
        x_i, x_t = _split_modalities_from_joint(
            x,
            int(ds.text_dim) if "text_dim" in ds else None,
            int(ds.visual_dim) if "visual_dim" in ds else None,
        )
        if x_t is None or x_i is None:
            raise ValueError(f"{ds.name}: M0 requires Text and Visual feature dimensions")
        raw_edges = torch.load(resolve_path(ds.edge_path), map_location="cpu", weights_only=False)
        labels = torch.as_tensor(
            torch.load(resolve_path(ds.label_path), map_location="cpu", weights_only=False),
            dtype=torch.long,
        )
        split = torch.load(resolve_path(ds.node_split_path), map_location="cpu", weights_only=False, mmap=True)
        train_idx = torch.as_tensor(split["train_idx"], dtype=torch.long).contiguous()
        val_idx = torch.as_tensor(split["val_idx"], dtype=torch.long).contiguous()
        edge_index = preprocess_edge_index(
            ensure_edge_index(raw_edges),
            num_nodes,
            make_undirected=bool(ds.get("make_undirected", True)),
            with_self_loops=bool(ds.get("add_self_loops", False)),
        )
        return MAGData(
            name=str(ds.name), source=source, task="nc", x=x, x_i=x_i, x_t=x_t,
            edge_index=edge_index, y=labels, train_idx=train_idx, val_idx=val_idx,
            test_idx=None, num_nodes=num_nodes, num_classes=int(ds.num_classes),
            info={"node_split_path": str(resolve_path(ds.node_split_path)), "test_split_field_read": False},
        )
    raise ValueError(f"Unknown data source: {source}")


def restrict_labels_to_train_val(data: MAGData) -> dict[str, torch.Tensor]:
    if data.y is None or data.train_idx is None or data.val_idx is None:
        raise ValueError("M0 NC data must contain labels plus train/validation indices")
    # Copy only explicitly allowed labels. The test key is never requested from
    # the split mapping and the returned MAGData has test_idx=None.
    allowed = torch.full_like(data.y, -100)
    allowed[data.train_idx] = data.y[data.train_idx]
    allowed[data.val_idx] = data.y[data.val_idx]
    data.y = allowed
    data.test_idx = None
    if data.y.numel() != data.num_nodes:
        raise ValueError("label vector/node count mismatch")
    return {"train": data.train_idx, "validation": data.val_idx}


def _state_hash(state: dict[str, torch.Tensor], names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_common_initialization(cfg, data_info: dict[str, int], seed: int) -> dict[str, Any]:
    states, counts, hashes = {}, {}, {}
    variants = ("extent", "single_basis", "multi_basis")
    for variant in variants:
        set_seed(seed)
        local_cfg = copy.deepcopy(cfg)
        local_cfg.model.variant = variant
        model = Model(local_cfg, data_info)
        names = common_parameter_names(model)
        state = model.state_dict()
        states[variant] = {name: state[name].detach().cpu().clone() for name in names}
        hashes[variant] = _state_hash(state, names)
        counts[variant] = parameter_counts(model)
        del model
    comparisons = {}
    for left, right in (("extent", "single_basis"), ("extent", "multi_basis"), ("single_basis", "multi_basis")):
        comparisons[f"{left}__{right}"] = all(
            torch.equal(states[left][name], states[right][name]) for name in states[left]
        ) and states[left].keys() == states[right].keys()
    basis_match = counts["single_basis"]["correction_basis"] == counts["multi_basis"]["correction_basis"]
    return {
        "dataset": str(cfg.dataset.name), "seed": int(seed), "common_hashes": hashes,
        "common_pairwise_equal": comparisons, "common_all_equal": all(comparisons.values()),
        "basis_params_single": counts["single_basis"]["correction_basis"],
        "basis_params_multi": counts["multi_basis"]["correction_basis"],
        "basis_params_equal": basis_match,
        "parameter_counts": counts,
    }


def _quantiles(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {key: float("nan") for key in ("mean", "std", "q10", "q25", "median", "q75", "q90")}
    q10, q25, med, q75, q90 = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "q10": float(q10),
        "q25": float(q25), "median": float(med), "q75": float(q75), "q90": float(q90),
    }


def _control_stats(aux_info: dict[str, Any], val_idx: torch.Tensor) -> dict[str, Any]:
    controls = aux_info["controls"]
    stats: dict[str, Any] = {"g": [], "c": [], "correction_ratio": [], "pi": []}
    for m in range(2):
        g = controls["g"][m].reshape(-1)
        if g.numel():
            g_stats = _quantiles(g)
            g_stats["fraction_lt_0_1"] = float((g < 0.1).float().mean())
            g_stats["fraction_gt_0_9"] = float((g > 0.9).float().mean())
        else:
            g_stats = {}
        stats["g"].append(g_stats)
        c = controls["c"][m].reshape(-1)
        if c.numel():
            c_stats = _quantiles(c)
            c_stats["fraction_lt_0_1"] = float((c < 0.1).float().mean())
            c_stats["fraction_gt_0_9"] = float((c > 0.9).float().mean())
        else:
            c_stats = {}
        stats["c"].append(c_stats)
        ratio = controls["correction_ratio"][m].reshape(-1)
        stats["correction_ratio"].append(_quantiles(ratio))
        pi = controls["pi"][m]
        if pi.numel():
            entropy = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(dim=-1)
            normalized = entropy / math.log(pi.size(-1))
            stats["pi"].append({
                "basis_mean": pi.mean(dim=0).tolist(),
                "basis_std": pi.std(dim=0, unbiased=False).tolist(),
                "normalized_entropy": _quantiles(normalized),
                "effective_bases": _quantiles(entropy.exp()),
            })
        else:
            stats["pi"].append({})

    for key in ("g", "c"):
        left, right = controls[key]
        stats[f"mean_abs_{key}_tv"] = float((left - right).abs().mean()) if left.numel() else float("nan")
    if controls["pi"][0].numel():
        stats["mean_l1_pi_tv"] = float((controls["pi"][0] - controls["pi"][1]).abs().sum(dim=-1).mean())
    else:
        stats["mean_l1_pi_tv"] = float("nan")

    edge_index = aux_info["edge_index_nonself"]
    dst = edge_index[1]
    degree = aux_info["degree"].cpu()
    val_set = set(int(v) for v in val_idx.tolist())
    eligible = {i for i in val_set if int(degree[i]) >= 5}
    within = []
    for m in range(2):
        if controls["g"][m].numel() == 0:
            within.append({"eligible_targets": 0, "median_std": float("nan"),
                           "q25_std": float("nan"), "q75_std": float("nan")})
            continue
        node_values: dict[int, list[float]] = defaultdict(list)
        for edge_id, target in enumerate(dst.tolist()):
            if target in eligible:
                node_values[target].append(float(controls["g"][m][edge_id, 0]))
        node_std = [float(np.std(vals, ddof=0)) for vals in node_values.values() if len(vals) > 1]
        within.append({
            "eligible_targets": len(node_std),
            "median_std": float(np.median(node_std)) if node_std else float("nan"),
            "q25_std": float(np.quantile(node_std, 0.25)) if node_std else float("nan"),
            "q75_std": float(np.quantile(node_std, 0.75)) if node_std else float("nan"),
        })
    stats["within_node_g_std"] = within
    return stats


def _metric_from_embeddings(classifier, z, val_idx, val_y, class_count: int) -> dict[str, float]:
    logits = classifier(z[val_idx])
    loss = nn.functional.cross_entropy(logits, val_y)
    pred = logits.argmax(dim=-1)
    return {
        "val_accuracy": float((pred == val_y).float().mean().item()),
        "val_macro_f1": float(f1_score(
            val_y.detach().cpu().numpy(), pred.detach().cpu().numpy(),
            labels=list(range(class_count)), average="macro", zero_division=0,
        )),
        "val_ce": float(loss.item()),
    }


@torch.no_grad()
def _evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes, overrides=None):
    model.eval()
    classifier.eval()
    z, _, _, _, _ = model(x, edge_index, control_overrides=overrides)
    return _metric_from_embeddings(classifier, z, val_idx, val_y, num_classes)


def _control_interventions(model, classifier, x, edge_index, val_idx, val_y, num_classes, controls, dst):
    output: list[dict[str, Any]] = []
    device, dtype = x.device, x.dtype
    base = _evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes)

    def add(name: str, overrides: dict[str, tuple[torch.Tensor, torch.Tensor]], repeat: int | None = None):
        metrics = _evaluate(model, classifier, x, edge_index, val_idx, val_y, num_classes, overrides)
        row = {"intervention": name, "repeat_seed": repeat, **metrics}
        row["delta_ce"] = metrics["val_ce"] - base["val_ce"]
        row["delta_accuracy"] = metrics["val_accuracy"] - base["val_accuracy"]
        row["delta_macro_f1"] = metrics["val_macro_f1"] - base["val_macro_f1"]
        output.append(row)

    active = {name: controls[name] for name in ("g", "c", "pi") if controls[name][0].numel()}
    if "g" in active:
        all_one = tuple(torch.ones_like(value) for value in active["g"])
        add("extent_off", {**active, "g": all_one})
    if model.variant in {"single_basis", "multi_basis"}:
        zeros = tuple(torch.zeros_like(value) for value in active["c"])
        add("function_off", {**active, "c": zeros})
    if model.variant in {"extent", "single_basis", "multi_basis"}:
        for shuffle_seed in (1001, 1002, 1003, 1004, 1005):
            shuffled = {}
            for name, pair in active.items():
                out_pair = []
                for modality in range(2):
                    permutation = neighborhood_shuffle_indices(dst, shuffle_seed + modality * 10000)
                    out_pair.append(pair[modality][permutation])
                shuffled[name] = tuple(out_pair)
            add("edge_control_shuffle", shuffled, shuffle_seed)
        add("modality_tied", tie_modality_controls(active))
    if model.variant == "multi_basis":
        uniform_pi = tuple(torch.full_like(value, 1.0 / value.size(-1)) for value in active["pi"])
        add("basis_uniform", {**active, "pi": uniform_pi})
    return {"normal": base, "rows": output}


def _grad_audit(model: Model) -> dict[str, dict[str, float]]:
    prefixes = {
        "gate": ("gate_heads.",),
        "correction": ("correction_heads.",),
        "basis_head": ("basis_heads.",),
        "basis": ("basis_u", "basis_v"),
    }
    result = {}
    for group, group_prefixes in prefixes.items():
        grads = [p.grad.detach() for name, p in model.named_parameters() if name.startswith(group_prefixes) and p.grad is not None]
        result[group] = {
            "finite": all(bool(torch.isfinite(grad).all()) for grad in grads),
            "norm": float(torch.sqrt(sum(grad.float().pow(2).sum() for grad in grads)).item()) if grads else 0.0,
            "parameter_tensors_with_grad": len(grads),
        }
    return result


def run_one(dataset: str, seed: int, variant: str, device_name: str, out_dir: Path = OUT_DIR,
            max_epochs: int | None = None, save_checkpoint: bool = True) -> dict[str, Any]:
    cfg = make_config(dataset, seed, variant, device_name, epochs=max_epochs)
    data = load_m0_data(cfg, seed)
    split_indices = restrict_labels_to_train_val(data)
    data_info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }
    audit = audit_common_initialization(cfg, data_info, seed)
    if not audit["common_all_equal"] or not audit["basis_params_equal"]:
        raise AssertionError(f"A/B/C initialization/parameter audit failed: {audit}")

    device = torch.device(device_name)
    set_seed(seed)
    model = Model(cfg, data_info).to(device)
    # Reuse the same classifier seed and dropout stream in every variant.
    torch.manual_seed(seed + 1907)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1907)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    set_seed(seed)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_idx = split_indices["train"].to(device)
    val_idx = split_indices["validation"].to(device)
    labels = data.y.to(device)
    train_y, val_y = labels[train_idx], labels[val_idx]
    if bool((train_y < 0).any()) or bool((val_y < 0).any()):
        raise ValueError("M0 train/validation split contains a missing label")
    if data.test_idx is not None:
        raise AssertionError("M0 data loader unexpectedly exposed test indices")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_acc = -1.0
    best_epoch = 0
    patience_left = int(cfg.task.patience)
    best_model = best_head = None
    first_gradients: dict[str, Any] = {}
    training_started = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, int(cfg.task.epochs) + 1):
        epochs_run = epoch
        model.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        z, _, _, aux_loss, _ = model(x, edge_index)
        logits = classifier(z[train_idx])
        loss = nn.functional.cross_entropy(logits, train_y) + float(cfg.task.loss.aux_weight) * aux_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at {dataset}/{seed}/{variant}/epoch{epoch}")
        loss.backward()
        if epoch == 1:
            first_gradients = _grad_audit(model)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()),
            max_norm=float(cfg.task.grad_clip), error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        del z, logits, loss

        current = _evaluate(model, classifier, x, edge_index, val_idx, val_y, int(data.num_classes))
        if current["val_accuracy"] > best_acc + float(cfg.task.early_stop_min_delta):
            best_acc = current["val_accuracy"]
            best_epoch = epoch
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            best_metrics = current
            patience_left = int(cfg.task.patience)
        elif epoch >= int(cfg.task.early_stop_min_epoch):
            patience_left -= 1
            if patience_left <= 0:
                break
    training_time = time.perf_counter() - training_started
    if best_model is None or best_head is None:
        raise RuntimeError("training ended without a validation-selected checkpoint")
    model.load_state_dict(best_model)
    classifier.load_state_dict(best_head)

    model.eval()
    with torch.no_grad():
        z, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
        normal_metrics = _metric_from_embeddings(classifier, z, val_idx, val_y, int(data.num_classes))
    diagnostics = _control_stats(aux, val_idx.detach().cpu())
    dst = aux["edge_index_nonself"][1]
    interventions_started = time.perf_counter()
    interventions = _control_interventions(
        model, classifier, x, edge_index, val_idx, val_y, int(data.num_classes),
        aux["controls"], dst,
    )
    intervention_time = time.perf_counter() - interventions_started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    counts = parameter_counts(model)
    classifier_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    record = {
        "status": "completed", "dataset": dataset, "seed": int(seed), "variant": variant,
        "best_epoch": int(best_epoch), "epochs_run": int(epochs_run),
        **normal_metrics,
        "model_params": counts["model_total"], "classifier_params": classifier_params,
        "trainable_params": counts["model_total"] + classifier_params,
        "parameter_counts": counts, "peak_gpu_memory_bytes": peak_memory,
        "training_time_sec": training_time, "intervention_time_sec": intervention_time,
        "gradient_audit_epoch1": first_gradients, "control_diagnostics": diagnostics,
        "interventions": interventions, "initialization_audit": audit,
        "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        "test_split_field_attached_to_data": False, "test_labels_exposed_to_runner": False,
        "data_info": data_info,
    }
    run_json = out_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
    write_json(run_json, record)
    if save_checkpoint:
        checkpoint = out_dir / "checkpoints" / dataset / f"seed_{seed}_{variant}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": best_model, "head_state": best_head, "data_info": data_info,
            "dataset": dataset, "seed": int(seed), "variant": variant,
            "best_epoch": int(best_epoch), "best_val_metrics": normal_metrics,
            "protocol": "unified_full_graph_nc_v1", "evaluate_test": False,
        }, checkpoint)
        record["checkpoint"] = str(checkpoint.relative_to(PROJECT_ROOT))
        write_json(run_json, record)
    print(
        f"[done] {dataset} seed={seed} {variant}: epoch={best_epoch} "
        f"val_acc={normal_metrics['val_accuracy']:.4f} val_f1={normal_metrics['val_macro_f1']:.4f} "
        f"val_ce={normal_metrics['val_ce']:.4f} train_s={training_time:.1f} "
        f"peak_gb={peak_memory / (1024**3):.2f}",
        flush=True,
    )
    del z, aux, data, x, edge_index, model, classifier, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def run_smoke(device: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    dataset, seed = "Movies", 42
    cfg = make_config(dataset, seed, "semantic", device, epochs=2)
    data = load_m0_data(cfg, seed)
    split_indices = restrict_labels_to_train_val(data)
    data_info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }
    audit = audit_common_initialization(cfg, data_info, seed)
    if not audit["common_all_equal"] or not audit["basis_params_equal"]:
        raise AssertionError(f"common initialization audit failed: {audit}")
    device_obj = torch.device(device)
    x, edges = data.x.to(device_obj), data.edge_index.to(device_obj)
    train_idx = split_indices["train"].to(device_obj)
    val_idx = split_indices["validation"].to(device_obj)
    labels = data.y.to(device_obj)
    train_y = labels[train_idx]
    results = []
    initial_ratios = {}
    for variant in VARIANTS:
        cfg.model.variant = variant
        set_seed(seed)
        model = Model(cfg, data_info).to(device_obj)
        torch.manual_seed(seed + 1907)
        if device_obj.type == "cuda":
            torch.cuda.manual_seed_all(seed + 1907)
        classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device_obj)
        set_seed(seed)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=float(cfg.task.lr), weight_decay=float(cfg.task.weight_decay),
        )
        if device_obj.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_obj)
        model.eval()
        with torch.no_grad():
            z0, _, _, _, diag0 = model(x, edges, return_diagnostics=True)
        if z0.shape != (data.num_nodes, 128) or not bool(torch.isfinite(z0).all()):
            raise AssertionError(f"invalid forward result for {variant}")
        if int(diag0["num_nonself_messages"]) != int((edges[0] != edges[1]).sum()):
            raise AssertionError("physical support changed beyond self-loop removal")
        if not torch.equal(diag0["edge_index_nonself"], edges[:, edges[0] != edges[1]].cpu()):
            raise AssertionError("directed non-self physical edge order/support changed")
        controls = diag0["controls"]
        if variant in {"extent", "single_basis", "multi_basis"}:
            for gate in controls["g"]:
                if gate.numel() and (float(gate.min()) < 0 or float(gate.max()) > 1):
                    raise AssertionError("gate outside [0,1]")
                if abs(float(gate.mean()) - float(torch.sigmoid(torch.tensor(2.0))) ) > 0.08:
                    raise AssertionError("initial gate mean is not near sigmoid(2)")
            if variant in {"single_basis", "multi_basis"}:
                for correction in controls["c"]:
                    if correction.numel() and (float(correction.min()) < 0 or float(correction.max()) > 1):
                        raise AssertionError("correction strength outside [0,1]")
                    if abs(float(correction.mean()) - float(torch.sigmoid(torch.tensor(-2.0)))) > 0.08:
                        raise AssertionError("initial correction mean is not near sigmoid(-2)")
                ratios = torch.cat(controls["correction_ratio"], dim=0)
                if ratios.numel() and not float(ratios.mean()) < 1.0:
                    raise AssertionError("initial correction dominates the default message")
                initial_ratios[variant] = {
                    "mean": float(ratios.mean()) if ratios.numel() else 0.0,
                    "median": float(ratios.median()) if ratios.numel() else 0.0,
                }
            if variant == "multi_basis":
                for pi in controls["pi"]:
                    if pi.numel() and not torch.allclose(pi.sum(dim=-1), torch.ones(pi.size(0)), atol=1e-6):
                        raise AssertionError("basis probabilities do not sum to one")
                    if pi.numel() and float((pi.mean(dim=0) - 0.25).abs().max()) > 0.02:
                        raise AssertionError("initial basis routing is not near uniform")

        model.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        z, _, _, aux_loss, _ = model(x, edges)
        loss = nn.functional.cross_entropy(classifier(z[train_idx]), train_y) + aux_loss
        if not bool(torch.isfinite(loss)):
            raise AssertionError("non-finite smoke loss")
        loss.backward()
        grad = _grad_audit(model)
        for group in ("gate", "correction", "basis_head", "basis"):
            required = (variant == "extent" and group == "gate") or (
                variant == "single_basis" and group in {"gate", "correction", "basis"}
            ) or (variant == "multi_basis" and group in {"gate", "correction", "basis_head", "basis"})
            if required and (not grad[group]["finite"] or grad[group]["norm"] <= 0):
                raise AssertionError(f"missing finite nonzero {group} gradient for {variant}: {grad[group]}")
        optimizer.step()
        model.eval()
        classifier.eval()
        with torch.no_grad():
            forward, _, _, _, _ = model(x, edges)
            inferred = model.inference(data.x, data.edge_index, device=device_obj)
        max_diff = float((forward.detach().cpu() - inferred).abs().max())
        if max_diff > 1e-5:
            raise AssertionError(f"inference/forward mismatch ({max_diff}) for {variant}")
        result = {
            "variant": variant, "forward_shape": list(forward.shape),
            "gradients": grad, "inference_forward_max_abs": max_diff,
            "parameter_counts": parameter_counts(model),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device_obj)) if device_obj.type == "cuda" else 0,
        }
        results.append(result)
        del model, classifier, optimizer, z0, diag0, z, forward, inferred, loss
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    basis_match = parameter_counts(Model(OmegaConf.create({"model": {"variant": "single_basis", "hidden_dim": 128, "dropout": 0.2}}), data_info))["correction_basis"] == parameter_counts(
        Model(OmegaConf.create({"model": {"variant": "multi_basis", "hidden_dim": 128, "dropout": 0.2}}), data_info)
    )["correction_basis"]
    denom_check = float(
        (torch.tensor([[2.0], [0.0]], device=device_obj).sum(dim=0) / torch.tensor([2.0], device=device_obj)).item()
    )
    if denom_check != 1.0:
        raise AssertionError("fixed-degree mean check failed")
    payload = {
        "status": "smoke_passed", "dataset": dataset, "seed": seed,
        "variants": results, "common_initialization_audit": audit,
        "initial_correction_default_ratio": initial_ratios,
        "bc_basis_parameter_match": basis_match,
        "fixed_degree_mean_check": denom_check,
        "modality_slices": {"text_dim": data_info["text_dim"], "visual_dim": data_info["visual_dim"], "order": "Text then Visual"},
        "test_idx_loaded": False, "test_labels_exposed": False,
        "test_evaluation": False,
    }
    write_json(out_dir / "smoke" / "Movies_seed_42" / "smoke_result.json", payload)
    print(f"[smoke-passed] Movies/42 variants={len(results)}", flush=True)
    del data, x, edges, labels
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def run_campaign(args) -> None:
    smoke_path = args.output_dir / "smoke" / "Movies_seed_42" / "smoke_result.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "smoke_passed":
        raise RuntimeError("Movies/42 five-variant smoke must pass before formal training")
    results = []
    total = len(args.datasets) * len(args.seeds) * len(args.variants)
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in args.variants:
                path = args.output_dir / "runs" / dataset / f"seed_{seed}" / f"{variant}.json"
                if path.is_file():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if old.get("status") == "completed":
                        print(f"[skip-complete] {dataset} seed={seed} {variant}", flush=True)
                        results.append(old)
                        continue
                completed = len(results)
                print(f"[run {completed + 1}/{total}] {dataset} seed={seed} {variant} on {args.device}", flush=True)
                try:
                    record = run_one(dataset, seed, variant, args.device, args.output_dir, args.max_epochs)
                except Exception as exc:
                    failed_path = args.output_dir / "failed_runs.json"
                    failed = json.loads(failed_path.read_text(encoding="utf-8")) if failed_path.exists() else []
                    failed.append({"dataset": dataset, "seed": seed, "variant": variant,
                                   "error": repr(exc), "timestamp_unix": time.time()})
                    write_json(failed_path, failed)
                    raise
                results.append(record)
                write_json(args.output_dir / "campaign_progress.json", {
                    "requested_runs": total,
                    "completed_unique_runs": len({(r['dataset'], r['seed'], r['variant']) for r in results}),
                    "datasets": list(args.datasets), "seeds": list(args.seeds),
                    "variants": list(args.variants), "device": args.device,
                    "evaluate_test": False,
                })
    completed = len({(r["dataset"], r["seed"], r["variant"]) for r in results if r.get("status") == "completed"})
    write_json(args.output_dir / "campaign_result.json", {
        "status": "completed" if completed == total else "partial",
        "completed_runs": completed, "requested_runs": total,
        "datasets": list(args.datasets), "seeds": list(args.seeds),
        "variants": list(args.variants), "device": args.device,
        "evaluate_test": False, "link_prediction": False,
    })
    print(f"[campaign-finished] {completed}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="M0 adaptive propagation mechanism screen")
    parser.add_argument("--smoke", action="store_true", help="Movies/42 five-variant correctness smoke")
    parser.add_argument("--campaign", action="store_true", help="Run the requested dataset × seed × variant matrix")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(
        f"[m0] device={args.device} gpu={torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu'} "
        f"torch={torch.__version__} pyg={__import__('torch_geometric').__version__}",
        flush=True,
    )
    if args.smoke:
        run_smoke(args.device, args.output_dir)
    elif args.campaign:
        run_campaign(args)
    elif args.dataset and args.seed is not None and args.variant:
        run_one(args.dataset, args.seed, args.variant, args.device, args.output_dir, args.max_epochs)
    else:
        parser.error("choose --smoke, --campaign, or one --dataset/--seed/--variant run")


if __name__ == "__main__":
    main()
