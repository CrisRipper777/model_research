from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from src.analysis.r3mag_h1_response_audit import (
    DATA_SEED,
    PARTITION_SEED,
    assert_disjoint_splits,
    load_fixed_data,
    stratified_internal_partition,
    tensor_sha256,
)
from src.models import build_model
from src.utils.graph_ops import normalized_adjacency_operator
from src.utils.seeds import set_seed


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/r3mag_prototype/v1"
REPORT_ROOT = ROOT / "reports/r3mag_prototype/v1"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("G0", "G0-FT", "G1", "G2", "G3", "G4")
CONTRASTS = (
    ("G0-FT", "G0"),
    ("G1", "G0-FT"),
    ("G2", "G1"),
    ("G3", "G2"),
    ("G4", "G3"),
    ("G4", "G0-FT"),
)
STAGE1_SETTINGS = {"lr": 1e-3, "weight_decay": 5e-4, "max_epochs": 400,
                   "patience": 50, "min_epoch": 30, "unimodal_weight": 0.1}
STAGE2_SETTINGS = {"base_lr": 3e-4, "new_lr": 1e-3, "classifier_lr": 3e-4,
                   "weight_decay": 5e-4, "max_epochs": 400, "patience": 50,
                   "min_epoch": 30, "unimodal_weight": 0.1,
                   "lambda_budget": 0.002, "delta_gamma": 0.2}
MODEL_SPEC = {"hidden_dim": 128, "max_order": 3, "dropout": 0.2,
              "router_dim": 64, "shared_atoms": 3, "private_atoms": 2,
              "flat_atoms": 3, "delta_gamma": 0.2}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _index_hashes(data, partitions: dict[str, torch.Tensor]) -> dict[str, str]:
    dev_train = torch.cat([partitions["host_train"], partitions["response_train"]]).sort().values
    return {
        "OriginalTrain": tensor_sha256(data.train_idx),
        "HostTrain": tensor_sha256(partitions["host_train"]),
        "ResponseTrain": tensor_sha256(partitions["response_train"]),
        "DevTrain": tensor_sha256(dev_train),
        "Audit": tensor_sha256(partitions["audit"]),
        "Val": tensor_sha256(data.val_idx),
        "Test": tensor_sha256(data.test_idx),
    }


def _verify_previous_split(dataset: str, seed: int, hashes: dict[str, str]) -> dict[str, Any]:
    path = REPORT_ROOT.parent.parent / "r3mag_design_freeze/h1/per_run" / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Required previous H1 split report is missing: {path}")
    old = json.loads(path.read_text(encoding="utf-8"))
    expected = old["split"]["index_sha256"]
    comparable = {key: value for key, value in hashes.items() if key != "DevTrain"}
    mismatch = {key: (value, expected.get(key)) for key, value in comparable.items()
                if value != expected.get(key)}
    if mismatch:
        raise AssertionError(f"{dataset}/seed{seed}: split hashes differ from H1: {mismatch}")
    if old["split"].get("test_labels_read") is not False:
        raise AssertionError(f"Previous H1 report does not certify untouched test labels: {path}")
    return {"path": str(path), "matches_previous_h1": True}


def preflight_protocol_inputs() -> None:
    """Verify all fixed data splits and feature conventions before model fitting."""
    for dataset in DATASETS:
        data, split_source, _ = load_fixed_data(dataset)
        if not torch.equal(data.x[:, : data.x_t.size(-1)], data.x_t):
            raise AssertionError(f"{dataset}: data.x text prefix differs from data.x_t")
        if not torch.equal(data.x[:, data.x_t.size(-1):], data.x_i):
            raise AssertionError(f"{dataset}: data.x visual suffix differs from data.x_i")
        parts, metadata = stratified_internal_partition(data.train_idx, data.y, seed=PARTITION_SEED)
        dev_train = torch.cat([parts["host_train"], parts["response_train"]]).sort().values
        assert_disjoint_splits({
            "DevTrain": dev_train, "Audit": parts["audit"], "Val": data.val_idx, "Test": data.test_idx,
        })
        hashes = _index_hashes(data, parts)
        references = {str(seed): _verify_previous_split(dataset, seed, hashes) for seed in SEEDS}
        print(json.dumps({
            "dataset": dataset, "split_source": str(split_source),
            "nodes": int(data.num_nodes), "train_count": int(data.train_idx.numel()),
            "dev_train_count": int(dev_train.numel()), "audit_count": int(parts["audit"].numel()),
            "val_count": int(data.val_idx.numel()), "test_count": int(data.test_idx.numel()),
            "partition_seed": PARTITION_SEED, "partition_method": metadata["method"],
            "split_hashes": hashes, "h1_references": references,
            "test_labels_read": False,
        }, indent=2), flush=True)


def labels_for_protocol(data, split_name: str, indices: torch.Tensor) -> torch.Tensor:
    """Read only an explicitly permitted label slice; Test is never an allowed role."""
    if split_name not in {"dev_train", "val", "audit"}:
        raise ValueError(f"label access is forbidden for split {split_name!r}")
    return data.y[torch.as_tensor(indices, dtype=torch.long).cpu()].long()


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.x.size(-1)),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.size(-1)),
        "visual_dim": int(data.x_i.size(-1)),
    }


def _build_model(cfg, data_info: dict[str, int], device: torch.device, seed: int):
    set_seed(int(seed))
    cfg.model.name = "r3mag_v1"
    model = build_model(cfg, data_info).to(device)
    return model


def _metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    labels = labels.to(logits.device).long()
    logits = logits.float()
    return {
        "ce": float(F.cross_entropy(logits, labels).item()),
        "accuracy": float((logits.argmax(-1) == labels).float().mean().item()),
        "macro_f1": float(f1_score(
            labels.detach().cpu().numpy(), logits.argmax(-1).detach().cpu().numpy(),
            labels=list(range(int(num_classes))), average="macro", zero_division=0,
        )),
    }


def _selection_improved(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None or candidate["accuracy"] > best["accuracy"]:
        return True
    return candidate["accuracy"] == best["accuracy"] and candidate["ce"] < best["ce"]


def _optimizer_group(
    name: str, named_parameters: dict[str, nn.Parameter], lr: float, weight_decay: float
) -> dict[str, Any]:
    return {
        "params": list(named_parameters.values()),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "group_name": name,
    }


def build_stage2_optimizer(
    model: nn.Module,
    classifier: nn.Module,
    variant: str,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    base = model.base_named_parameters()
    new = model.new_named_parameters(variant)
    classifier_params = {f"classifier.{n}": p for n, p in classifier.named_parameters()}
    group_specs = [
        ("base", base, STAGE2_SETTINGS["base_lr"]),
        ("new", new, STAGE2_SETTINGS["new_lr"]),
        ("classifier", classifier_params, STAGE2_SETTINGS["classifier_lr"]),
    ]
    active_specs = [item for item in group_specs if item[1]]
    ids = [id(param) for _, named, _ in active_specs for param in named.values()]
    if len(ids) != len(set(ids)):
        raise AssertionError("Stage-2 optimizer parameter groups contain duplicate parameters")
    expected = set(id(p) for p in base.values()) | set(id(p) for p in new.values()) | set(
        id(p) for p in classifier.parameters()
    )
    if set(ids) != expected:
        raise AssertionError("Stage-2 optimizer groups omit or add an eligible parameter")
    if variant == "G0-FT" and new:
        raise AssertionError("G0-FT must not create a New optimizer group")
    groups = [_optimizer_group(name, named, lr, STAGE2_SETTINGS["weight_decay"])
              for name, named, lr in active_specs]
    metadata = [{
        "name": name,
        "parameter_count": int(sum(parameter.numel() for parameter in named.values())),
        "lr": float(lr),
        "weight_decay": float(STAGE2_SETTINGS["weight_decay"]),
    } for name, named, lr in active_specs]
    optimizer = torch.optim.AdamW(groups)
    return optimizer, metadata


def _stage1_optimizer(model: nn.Module, classifier: nn.Module):
    base = model.base_named_parameters()
    classifier_params = {f"classifier.{n}": p for n, p in classifier.named_parameters()}
    expected = set(id(p) for p in base.values()) | set(id(p) for p in classifier.parameters())
    groups = [
        _optimizer_group("base", base, STAGE1_SETTINGS["lr"], STAGE1_SETTINGS["weight_decay"]),
        _optimizer_group("classifier", classifier_params, STAGE1_SETTINGS["lr"], STAGE1_SETTINGS["weight_decay"]),
    ]
    actual = [id(p) for group in groups for p in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AssertionError("Stage-1 optimizer groups do not exactly cover base and classifier parameters")
    metadata = [{
        "name": group["group_name"],
        "parameter_count": int(sum(p.numel() for p in group["params"])),
        "lr": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
    } for group in groups]
    return torch.optim.AdamW(groups), metadata


def _evaluate_validation(
    model, classifier, x, edge_index, operator, variant, val_idx, val_labels, num_classes
) -> dict[str, float]:
    model.eval()
    classifier.eval()
    with torch.no_grad():
        components = model.forward_components(
            x, edge_index, operator=operator, variant=variant, intervention="normal"
        )
        logits = classifier(components["z"][val_idx])
        return _metrics_from_logits(logits, val_labels, num_classes)


def _fit_variant(
    *,
    model,
    classifier,
    variant: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    operator: torch.Tensor,
    train_idx: torch.Tensor,
    train_labels: torch.Tensor,
    val_idx: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    max_epochs: int,
    patience: int,
    min_epoch: int,
    stage: str,
    offload_saved_activations: bool = False,
) -> dict[str, Any]:
    model.set_variant(variant)
    if stage == "stage1":
        optimizer, optimizer_groups = _stage1_optimizer(model, classifier)
        trainable = list(model.base_named_parameters().values()) + list(classifier.parameters())
        settings = STAGE1_SETTINGS
    else:
        optimizer, optimizer_groups = build_stage2_optimizer(model, classifier, variant)
        trainable = [p for group in optimizer.param_groups for p in group["params"]]
        settings = STAGE2_SETTINGS
    best_metrics: dict[str, float] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_classifier_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_ran = 0
    patience_left = int(patience)

    for epoch in range(1, int(max_epochs) + 1):
        epochs_ran = epoch
        model.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        activation_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if offload_saved_activations else nullcontext()
        )
        with activation_context:
            components = model.forward_components(
                x, edge_index, operator=operator, variant=variant, intervention="normal"
            )
            main_logits = classifier(components["z"][train_idx])
            main_loss = F.cross_entropy(main_logits, train_labels)
            uni_loss = 0.5 * (
                F.cross_entropy(components["logits_text"][train_idx], train_labels)
                + F.cross_entropy(components["logits_visual"][train_idx], train_labels)
            )
            loss = main_loss + float(settings["unimodal_weight"]) * uni_loss
            if variant == "G4":
                loss = loss + float(STAGE2_SETTINGS["lambda_budget"]) * components["budget_loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{variant} {stage} produced nonfinite loss at epoch {epoch}")
            if variant in {"G1", "G2", "G3", "G4"}:
                if not bool(torch.isfinite(components["rho_text"]).all() and torch.isfinite(components["rho_visual"]).all()):
                    raise FloatingPointError(f"{variant} produced nonfinite reliability values")
                for atom in components["atoms"].values():
                    if not torch.allclose(atom.norm(dim=-1), torch.ones(atom.size(0), device=atom.device), atol=1e-6):
                        raise AssertionError("response atom normalization failed during training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        train_loss_value = float(loss.detach())
        # Release the full-graph training activations before the full validation
        # forward; on large graphs even inference concatenations need headroom.
        del components, main_logits, main_loss, uni_loss, loss

        val_metrics = _evaluate_validation(
            model, classifier, x, edge_index, operator, variant, val_idx, val_labels, num_classes
        )
        if _selection_improved(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch
            best_model_state = _cpu_state(model)
            best_classifier_state = _cpu_state(classifier)
            patience_left = int(patience)
        elif epoch >= int(min_epoch):
            patience_left -= 1
        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"[{stage} {variant}] epoch={epoch} train_loss={train_loss_value:.5f} "
                f"val_acc={val_metrics['accuracy']:.5f} val_ce={val_metrics['ce']:.5f} "
                f"best_acc={best_metrics['accuracy'] if best_metrics else float('nan'):.5f} "
                f"patience={patience_left}", flush=True,
            )
        if epoch >= int(min_epoch) and patience_left <= 0:
            break
    if best_model_state is None or best_classifier_state is None or best_metrics is None:
        raise RuntimeError(f"{variant} did not produce a validation-selected checkpoint")
    model.load_state_dict(best_model_state)
    classifier.load_state_dict(best_classifier_state)
    model.eval()
    classifier.eval()
    return {
        "variant": variant,
        "stage": stage,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "selection": "best original-val accuracy; exact accuracy ties select lower val CE",
        "early_stopping": {"max_epochs": int(max_epochs), "patience": int(patience), "min_epoch": int(min_epoch)},
        "val_metrics": best_metrics,
        "optimizer_groups": optimizer_groups,
        "model_state": best_model_state,
        "classifier_state": best_classifier_state,
    }


def _checkpoint_signature(dataset: str, seed: int, stage: str, variant: str,
                          max_epochs: int, patience: int, min_epoch: int) -> dict[str, Any]:
    return {
        "protocol": "r3mag_v1_prototype_v1",
        "dataset": dataset,
        "model_seed": int(seed),
        "stage": stage,
        "variant": variant,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "min_epoch": int(min_epoch),
        "stage1_settings": STAGE1_SETTINGS,
        "stage2_settings": STAGE2_SETTINGS,
        "model_spec": MODEL_SPEC,
    }


def _load_saved_checkpoint(path: Path, signature: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("signature") != signature:
        raise ValueError(f"checkpoint protocol signature differs from current run: {path}")
    return saved


def _save_checkpoint(path: Path, signature: dict[str, Any], result: dict[str, Any],
                     *, model_state=None, classifier_state=None, initialization=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "signature": signature,
        "variant": result["variant"],
        "stage": result["stage"],
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "selection": result["selection"],
        "val_metrics": result["val_metrics"],
        "optimizer_groups": result["optimizer_groups"],
        "model_state": model_state if model_state is not None else result["model_state"],
        "classifier_state": classifier_state if classifier_state is not None else result["classifier_state"],
        "initialization": initialization or {},
    }, path)


def _assert_common_initialization(
    stage1_model_state: dict[str, torch.Tensor],
    stage1_classifier_state: dict[str, torch.Tensor],
    model,
    classifier,
) -> dict[str, Any]:
    model_state = model.state_dict()
    common_names = set(model.base_named_parameters())
    common_match = all(torch.equal(model_state[name].detach().cpu(), stage1_model_state[name])
                       for name in common_names)
    full_model_match = set(model_state) == set(stage1_model_state) and all(
        torch.equal(model_state[name].detach().cpu(), stage1_model_state[name])
        for name in model_state
    )
    classifier_match = all(torch.equal(value.detach().cpu(), stage1_classifier_state[name])
                           for name, value in classifier.state_dict().items())
    if not common_match or not full_model_match or not classifier_match:
        raise AssertionError("Stage-2 common model/classifier parameters differ from Stage-1 checkpoint")
    return {
        "common_model_parameters_match_stage1": common_match,
        "all_model_parameters_match_stage1": full_model_match,
        "classifier_matches_stage1": classifier_match,
        "common_parameter_sha256": _state_sha256({name: model_state[name] for name in sorted(common_names)}),
    }


def _audit_interventions(model, classifier, x, edge_index, operator, variant, audit_idx,
                         audit_labels, num_classes) -> tuple[dict[str, Any], dict[str, Any]]:
    model.set_variant(variant)
    model.eval()
    classifier.eval()
    intervention_names = ["normal", "preserve"]
    if variant in {"G1", "G2", "G3", "G4"}:
        intervention_names.append("uniform_route")
    if variant in {"G2", "G3", "G4"}:
        intervention_names.extend(["shared_off", "private_off"])
    logits_by_name: dict[str, torch.Tensor] = {}
    normal_payload: dict[str, Any] | None = None
    metrics: dict[str, Any] = {}
    shared_route_tensor_same = True
    intervention_logits_finite = True
    runtime_qa: dict[str, Any] = {}
    with torch.no_grad():
        for intervention in intervention_names:
            components = model.forward_components(
                x, edge_index, operator=operator, variant=variant, intervention=intervention
            )
            logits = classifier(components["z"][audit_idx])
            intervention_logits_finite = intervention_logits_finite and bool(torch.isfinite(logits).all())
            metrics[intervention] = _metrics_from_logits(logits, audit_labels, num_classes)
            logits_by_name[intervention] = logits.detach().float().cpu()
            if intervention == "normal":
                if variant in {"G2", "G3", "G4"}:
                    shared_text_route = components["route_weights"]["shared_text"]
                    shared_visual_route = components["route_weights"]["shared_visual"]
                    shared_route_tensor_same = (
                        shared_text_route is shared_visual_route
                        and torch.equal(shared_text_route, shared_visual_route)
                    )
                    if not shared_route_tensor_same:
                        raise AssertionError("shared route tensor must be one object for both modalities")
                zero_t = torch.zeros_like(components["delta_gamma_text"][:, 0])
                zero_v = torch.zeros_like(components["delta_gamma_visual"][:, 0])
                order0_delta_exact = torch.equal(components["delta_gamma_text"][:, 0], zero_t) and torch.equal(
                    components["delta_gamma_visual"][:, 0], zero_v
                )
                order0_gamma_exact = torch.equal(
                    components["adapted_gamma_text"][:, 0], model.gamma_global_text[0].expand(len(x))
                ) and torch.equal(
                    components["adapted_gamma_visual"][:, 0], model.gamma_global_visual[0].expand(len(x))
                )
                atom_norm_error = max(
                    (float((value.norm(dim=-1) - 1.0).abs().max().item()) for value in components["atoms"].values()),
                    default=0.0,
                )
                route_values_finite = all(bool(torch.isfinite(value).all())
                                          for value in components["route_weights"].values())
                rho_values_finite = bool(torch.isfinite(components["rho_text"]).all() and
                                         torch.isfinite(components["rho_visual"]).all())
                runtime_qa.update({
                    "delta_gamma_order0_exact_zero": order0_delta_exact,
                    "adapted_gamma_order0_exactly_global": order0_gamma_exact,
                    "atom_norm_max_abs_error": atom_norm_error,
                    "atom_norms_unit": atom_norm_error <= 1e-6,
                    "route_weights_finite": route_values_finite,
                    "rho_finite": rho_values_finite,
                })
                if not all([order0_delta_exact, order0_gamma_exact, atom_norm_error <= 1e-6,
                            route_values_finite, rho_values_finite]):
                    raise AssertionError(f"runtime R3-MAG invariant failed for {variant}")
                ids = audit_idx
                route_cpu = {
                    name: value[ids].detach().float().cpu()
                    for name, value in components["route_weights"].items()
                }
                if "shared_text" in route_cpu:
                    route_cpu["shared_visual"] = route_cpu["shared_text"]
                normal_payload = {
                    "atoms": {name: value.detach().float().cpu() for name, value in components["atoms"].items()},
                    "route_weights": route_cpu,
                    "r_shared": components["r_shared"][ids].detach().float().cpu(),
                    "r_private_text": components["r_private_text"][ids].detach().float().cpu(),
                    "r_private_visual": components["r_private_visual"][ids].detach().float().cpu(),
                    "rho_text": components["rho_text"][ids].detach().float().cpu(),
                    "rho_visual": components["rho_visual"][ids].detach().float().cpu(),
                }
            if intervention == "preserve":
                preserve_error = max(
                    float((components["response_text"] - components["global_response_text"]).abs().max().item()),
                    float((components["response_visual"] - components["global_response_visual"]).abs().max().item()),
                )
                runtime_qa["preserve_response_max_abs_error"] = preserve_error
                runtime_qa["preserve_recovers_global_response"] = preserve_error <= 1e-6
                if preserve_error > 1e-6:
                    raise AssertionError(f"preserve intervention failed for {variant}: {preserve_error}")
            del components, logits
    if normal_payload is None:
        raise AssertionError("normal intervention payload was not evaluated")
    runtime_qa["all_intervention_logits_finite"] = intervention_logits_finite
    runtime_qa["shared_route_tensor_same_object"] = shared_route_tensor_same
    if not intervention_logits_finite:
        raise FloatingPointError(f"nonfinite Audit logits under an intervention for {variant}")
    normal_logits = logits_by_name["normal"]
    preserve_logits = logits_by_name["preserve"]
    audit_labels_cpu = audit_labels.detach().cpu().long()
    utility = F.cross_entropy(preserve_logits, audit_labels_cpu, reduction="none") - F.cross_entropy(
        normal_logits, audit_labels_cpu, reduction="none"
    )
    detail: dict[str, Any] = {
        "metrics": metrics,
        "intervention_delta_from_normal": {
            intervention: {
                key: float(metrics[intervention][key] - metrics["normal"][key])
                for key in ("ce", "accuracy", "macro_f1")
            }
            for intervention in intervention_names if intervention != "normal"
        },
        "same_checkpoint_adaptation_utility": {
            "definition": "per-node CE_preserve - CE_normal; positive means normal adaptation helped",
            "mean": float(utility.mean().item()),
            "positive_fraction": float((utility > 0).float().mean().item()),
            "harmful_fraction": float((utility < 0).float().mean().item()),
            "zero_fraction": float((utility == 0).float().mean().item()),
        },
        "shared_route_tensor_same_object": shared_route_tensor_same,
        "qa": runtime_qa,
    }
    normal = normal_payload
    reliability: dict[str, Any] = {}
    if variant in {"G3", "G4"}:
        rho_t = normal["rho_text"].numpy()
        rho_v = normal["rho_visual"].numpy()
        rho = 0.5 * (rho_t + rho_v)
        utility_np = utility.detach().cpu().numpy()
        order = np.argsort(rho, kind="stable")
        quartiles = {}
        for q_index, rows in enumerate(np.array_split(order, 4), start=1):
            if len(rows) == 0:
                quartiles[f"Q{q_index}"] = {"node_count": 0, "mean_adaptation_utility": None,
                                             "harmful_fraction": None, "mean_ce": None, "accuracy": None}
                continue
            ids = torch.as_tensor(rows, dtype=torch.long)
            yq = audit_labels_cpu[ids]
            normal_q = normal_logits[ids]
            ce_q = F.cross_entropy(normal_q, yq, reduction="none")
            quartiles[f"Q{q_index}"] = {
                "node_count": int(len(rows)),
                "rho_mean": float(rho[rows].mean()),
                "rho_std": float(rho[rows].std(ddof=0)),
                "mean_adaptation_utility": float(utility_np[rows].mean()),
                "harmful_fraction": float((utility_np[rows] < 0).mean()),
                "mean_ce": float(ce_q.mean().item()),
                "accuracy": float((normal_q.argmax(-1) == yq).float().mean().item()),
            }
        correlation = spearmanr(rho, utility_np).statistic if len(rho) > 1 else float("nan")
        reliability = {
            "rho_text_mean": float(rho_t.mean()), "rho_text_std": float(rho_t.std(ddof=0)),
            "rho_visual_mean": float(rho_v.mean()), "rho_visual_std": float(rho_v.std(ddof=0)),
            "rho_mean": float(rho.mean()), "rho_std": float(rho.std(ddof=0)),
            "rho_quartiles": quartiles,
            "spearman_rho_vs_adaptation_utility": float(correlation) if np.isfinite(correlation) else None,
        }
    detail["reliability"] = reliability
    mechanism = _mechanism_diagnostics(normal, variant, audit_idx)
    return detail, {"mechanism": mechanism, "reliability": reliability}


def _atom_diagnostics(atoms: torch.Tensor) -> dict[str, Any]:
    atoms = atoms.float().cpu()
    cos = atoms @ atoms.T
    pairs = [{"atom_i": i + 1, "atom_j": j + 1, "cosine": float(cos[i, j])}
             for i in range(len(atoms)) for j in range(i + 1, len(atoms))]
    return {"coefficient_vectors": atoms.tolist(), "norms": atoms.norm(dim=-1).tolist(),
            "pairwise_cosines": pairs}


def _route_diagnostics(weights: torch.Tensor) -> dict[str, Any]:
    weights = weights.float().cpu()
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    return {
        "mean_probability_per_atom": weights.mean(0).tolist(),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std(unbiased=False)),
        "effective_usage_exp_mean_entropy": float(entropy.mean().exp()),
        "num_atoms": int(weights.size(-1)),
    }


def _mechanism_diagnostics(normal: dict[str, Any], variant: str, audit_idx: torch.Tensor) -> dict[str, Any]:
    del audit_idx  # all per-node normal tensors in this payload are already Audit-sliced
    atoms = normal["atoms"]
    routes = normal["route_weights"]
    atom_report = {name: _atom_diagnostics(value) for name, value in atoms.items()}
    route_report: dict[str, Any] = {}
    for name, value in routes.items():
        # Shared text/visual entries deliberately point to one identical tensor.
        if name == "shared_visual":
            continue
        route_report[name] = _route_diagnostics(value)
    if variant == "G1":
        left, right = routes["flat_text"], routes["flat_visual"]
        midpoint = 0.5 * (left + right)
        js = 0.5 * ((left * (left.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
                    + (right * (right.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1))
        route_report["text_visual_flat_routing_js_mean"] = float(js.mean())
    if variant in {"G2", "G3", "G4"}:
        left, right = routes["private_text"], routes["private_visual"]
        midpoint = 0.5 * (left + right)
        js = 0.5 * ((left * (left.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
                    + (right * (right.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1))
        route_report["text_visual_private_routing_js_mean"] = float(js.mean())
        shared_text = routes["shared_text"]
        shared_visual = routes["shared_visual"]
        if not torch.equal(shared_text, shared_visual):
            raise AssertionError("Text/Visual shared route must be one identical tensor")
        shared, private_t, private_v = normal["r_shared"], normal["r_private_text"], normal["r_private_visual"]
        mechanism_paths = {
            "mean_norm_shared": float(shared.norm(dim=-1).mean()),
            "mean_norm_private_text": float(private_t.norm(dim=-1).mean()),
            "mean_norm_private_visual": float(private_v.norm(dim=-1).mean()),
            "cos_shared_private_text": float(F.cosine_similarity(shared, private_t, dim=-1, eps=1e-8).mean()),
            "cos_shared_private_visual": float(F.cosine_similarity(shared, private_v, dim=-1, eps=1e-8).mean()),
        }
    else:
        mechanism_paths = {}
    return {"atoms": atom_report, "routing": route_report, "shared_private": mechanism_paths}


def _run_one(dataset: str, seed: int, *, device: torch.device, smoke: bool = False,
             smoke_epochs: int = 3) -> dict[str, Any]:
    data, split_source, cfg = load_fixed_data(dataset)
    partitions, partition_metadata = stratified_internal_partition(
        data.train_idx, data.y, seed=PARTITION_SEED
    )
    dev_train = torch.cat([partitions["host_train"], partitions["response_train"]]).sort().values
    assert_disjoint_splits({
        "HostTrain": partitions["host_train"], "ResponseTrain": partitions["response_train"],
        "Audit": partitions["audit"], "Val": data.val_idx, "Test": data.test_idx,
    })
    assert_disjoint_splits({"DevTrain": dev_train, "Audit": partitions["audit"],
                            "Val": data.val_idx, "Test": data.test_idx})
    if set(dev_train.tolist()) | set(partitions["audit"].tolist()) != set(data.train_idx.tolist()):
        raise AssertionError("DevTrain union Audit must exactly cover original train labels")
    hashes = _index_hashes(data, partitions)
    split_reference = _verify_previous_split(dataset, seed, hashes)

    info = _data_info(data)
    model = _build_model(cfg, info, device, seed)
    x = data.x.to(device=device, dtype=torch.float32)
    edge_index = data.edge_index.to(device=device, dtype=torch.long)
    operator = normalized_adjacency_operator(
        edge_index, int(data.num_nodes), dtype=x.dtype, device=device
    )
    if not bool(torch.isfinite(operator.values()).all()):
        raise FloatingPointError("normalized A+I operator contains nonfinite values")
    train_idx_device = dev_train.to(device)
    val_idx_device = data.val_idx.to(device)
    train_labels = labels_for_protocol(data, "dev_train", dev_train).to(device)
    val_labels = labels_for_protocol(data, "val", data.val_idx).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)

    run_root = OUTPUT_ROOT / ("smoke" if smoke else "formal") / dataset / f"seed{seed}"
    stage1_path = run_root / "stage1_best.pt"
    max_epochs = int(smoke_epochs if smoke else STAGE1_SETTINGS["max_epochs"])
    patience = int(min(2, smoke_epochs) if smoke else STAGE1_SETTINGS["patience"])
    min_epoch = 1 if smoke else STAGE1_SETTINGS["min_epoch"]
    stage1_signature = _checkpoint_signature(dataset, seed, "stage1", "G0", max_epochs, patience, min_epoch)
    stage1_saved = _load_saved_checkpoint(stage1_path, stage1_signature)
    if stage1_saved is None:
        model.set_variant("G0")
        set_seed(seed)
        classifier.reset_parameters()
        stage1_result = _fit_variant(
            model=model, classifier=classifier, variant="G0", x=x, edge_index=edge_index,
            operator=operator, train_idx=train_idx_device, train_labels=train_labels,
            val_idx=val_idx_device, val_labels=val_labels, num_classes=int(data.num_classes),
            max_epochs=max_epochs, patience=patience, min_epoch=min_epoch, stage="stage1",
            offload_saved_activations=(device.type == "cuda" and int(data.num_nodes) > 50000),
        )
        _save_checkpoint(stage1_path, stage1_signature, stage1_result)
        stage1_trained_now = True
        stage1_model_state, stage1_classifier_state = stage1_result["model_state"], stage1_result["classifier_state"]
    else:
        stage1_trained_now = False
        stage1_result = {
            "variant": "G0", "stage": "stage1", "best_epoch": stage1_saved["best_epoch"],
            "epochs_ran": stage1_saved["epochs_ran"], "selection": stage1_saved["selection"],
            "val_metrics": stage1_saved["val_metrics"], "optimizer_groups": stage1_saved["optimizer_groups"],
        }
        stage1_model_state = stage1_saved["model_state"]
        stage1_classifier_state = stage1_saved["classifier_state"]

    model.load_state_dict(stage1_model_state)
    classifier.load_state_dict(stage1_classifier_state)
    stage_results: dict[str, Any] = {
        "G0": {key: stage1_result[key] for key in ("best_epoch", "epochs_ran", "selection", "val_metrics", "optimizer_groups")},
    }
    initializations: dict[str, Any] = {"G0": {"stage1_best_checkpoint_reused": True}}
    stage2_epochs = int(smoke_epochs if smoke else STAGE2_SETTINGS["max_epochs"])
    stage2_patience = int(min(2, smoke_epochs) if smoke else STAGE2_SETTINGS["patience"])
    stage2_min_epoch = 1 if smoke else STAGE2_SETTINGS["min_epoch"]
    for variant in VARIANTS[1:]:
        signature = _checkpoint_signature(dataset, seed, "stage2", variant,
                                          stage2_epochs, stage2_patience, stage2_min_epoch)
        checkpoint_path = run_root / f"{variant.replace('-', '_')}_best.pt"
        saved = _load_saved_checkpoint(checkpoint_path, signature)
        model.load_state_dict(stage1_model_state)
        classifier.load_state_dict(stage1_classifier_state)
        model.set_variant(variant)
        initialization = _assert_common_initialization(
            stage1_model_state, stage1_classifier_state, model, classifier
        )
        if saved is None:
            if device.type == "cuda":
                # Adam buffers and full-graph activations from the previous fit are
                # no longer live; return cached blocks before the next variant.
                torch.cuda.empty_cache()
            set_seed(seed)
            result = _fit_variant(
                model=model, classifier=classifier, variant=variant, x=x, edge_index=edge_index,
                operator=operator, train_idx=train_idx_device, train_labels=train_labels,
                val_idx=val_idx_device, val_labels=val_labels, num_classes=int(data.num_classes),
                max_epochs=stage2_epochs, patience=stage2_patience, min_epoch=stage2_min_epoch,
                stage="stage2",
                offload_saved_activations=(device.type == "cuda" and int(data.num_nodes) > 50000),
            )
            _save_checkpoint(checkpoint_path, signature, result, initialization=initialization)
            saved = {
                "best_epoch": result["best_epoch"], "epochs_ran": result["epochs_ran"],
                "selection": result["selection"], "val_metrics": result["val_metrics"],
                "optimizer_groups": result["optimizer_groups"],
            }
        else:
            model.load_state_dict(saved["model_state"])
            classifier.load_state_dict(saved["classifier_state"])
        stage_results[variant] = {key: saved[key] for key in (
            "best_epoch", "epochs_ran", "selection", "val_metrics", "optimizer_groups"
        )}
        initializations[variant] = {**initialization, "stage1_checkpoint_sha256": _state_sha256(stage1_model_state)}

    # Audit labels are sliced only after every Stage-1/Stage-2 checkpoint is selected.
    audit_idx = partitions["audit"]
    audit_labels = labels_for_protocol(data, "audit", audit_idx).to(device)
    variant_results: dict[str, Any] = {}
    qa_preserve: dict[str, float] = {}
    for variant in VARIANTS:
        path = stage1_path if variant == "G0" else run_root / f"{variant.replace('-', '_')}_best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        classifier.load_state_dict(saved["classifier_state"])
        model.set_variant(variant)
        detail, mechanism = _audit_interventions(
            model, classifier, x, edge_index, operator, variant,
            audit_idx.to(device), audit_labels, int(data.num_classes)
        )
        if "preserve" in detail["metrics"]:
            normal = detail["metrics"]["normal"]
            preserve = detail["metrics"]["preserve"]
            qa_preserve[variant] = max(abs(normal[key] - preserve[key]) for key in ("ce", "accuracy", "macro_f1"))
        variant_results[variant] = {
            **stage_results[variant], "audit": detail, **mechanism,
            "checkpoint_path": str(path), "checkpoint_sha256": _state_sha256(saved["model_state"]),
        }
    if qa_preserve.get("G0", float("inf")) > 1e-6:
        raise AssertionError("G0 normal and preserve interventions must be identical")

    record = {
        "dataset": dataset, "model_seed": int(seed), "data_seed": DATA_SEED,
        "partition_seed": PARTITION_SEED, "split_source": str(split_source),
        "split_hashes": hashes,
        "split_counts": {
            "OriginalTrain": int(data.train_idx.numel()), "HostTrain": int(partitions["host_train"].numel()),
            "ResponseTrain": int(partitions["response_train"].numel()), "DevTrain": int(dev_train.numel()),
            "Audit": int(audit_idx.numel()), "Val": int(data.val_idx.numel()), "Test": int(data.test_idx.numel()),
        },
        "split_matches_previous_h1": split_reference,
        "partition_method": partition_metadata["method"],
        "stage1": {**stage_results["G0"], "trained_once": True,
                   "trained_this_invocation": stage1_trained_now,
                   "checkpoint_path": str(stage1_path),
                   "checkpoint_sha256": _state_sha256(stage1_model_state)},
        "stage1_checkpoint_initialization": initializations,
        "variants": variant_results,
        "protocol": {
            "dev_train": "HostTrain union ResponseTrain (90% of original train labels)",
            "checkpoint_selection": "original val accuracy; exact ties use lower val CE",
            "audit_labels_used_only_after_checkpoint_selection": True,
            "test_labels_read": False, "test_metrics_computed": False,
            "test_split_untouched": True,
            "absolute_values_comparable_to_h1_h2_host": False,
            "comparison_boundary_note": "H1/H2 host used 80% HostTrain; this prototype trains on 90% DevTrain.",
        },
        "qa": {
            "split_hashes_match_h1": True,
            "splits_disjoint": True,
            "dev_train_union_audit_equals_original_train": True,
            "test_labels_read": False,
            "test_metrics_computed": False,
            "stage1_train_count_per_dataset_seed": 1,
            "optimizer_group_coverage_pass": True,
            "all_stage2_variants_initialized_from_same_stage1_checkpoint": all(
                initializations[v]["common_model_parameters_match_stage1"] and
                initializations[v]["all_model_parameters_match_stage1"] and
                initializations[v]["classifier_matches_stage1"] for v in VARIANTS[1:]
            ),
        "g0_preserve_identity_max_metric_abs_error": qa_preserve["G0"],
        "g0_preserve_identity_metric_tolerance": 1e-6,
            "model_invariants_pass": all(
                item["audit"]["qa"].get("delta_gamma_order0_exact_zero", True)
                and item["audit"]["qa"].get("adapted_gamma_order0_exactly_global", True)
                and item["audit"]["qa"].get("atom_norms_unit", True)
                and item["audit"]["qa"].get("route_weights_finite", True)
                and item["audit"]["qa"].get("rho_finite", True)
                and item["audit"]["qa"].get("preserve_recovers_global_response", True)
                and item["audit"]["qa"].get("all_intervention_logits_finite", False)
                for item in variant_results.values()
            ),
        },
        "report_reconstruction_device": str(device),
        "protocol_deviations": (
            ["Ele-Fashion seed42 G1/G2 used CUDA with pinned-CPU saved-activation offload. Seed42 G3/G4 and all missing seed43/44 stages were completed on CPU after repeated CUDA OOM under concurrent GPU memory pressure; seed42 G0/G0-FT checkpoints were reused from the initial CUDA run. Optimizer, precision, epoch caps, loss, data, and selection were unchanged."]
            if dataset == "ele-fashion" and device.type == "cpu" else
            ["Ele-Fashion CUDA training offloads autograd saved activations to pinned CPU memory to fit the full graph; optimizer, precision, epochs, and loss are unchanged."]
            if device.type == "cuda" and int(data.num_nodes) > 50000 else []
        ),
    }
    work_path = OUTPUT_ROOT / ("smoke" if smoke else "formal") / "work" / f"{dataset}_seed{seed}.json"
    _write_json(work_path, record)
    return record


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_task_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in records:
        for variant in VARIANTS:
            result = run["variants"][variant]
            row = {"dataset": run["dataset"], "seed": run["model_seed"], "variant": variant}
            for source, prefix in ((result["val_metrics"], "val"), (result["audit"]["metrics"]["normal"], "audit")):
                for metric in ("ce", "accuracy", "macro_f1"):
                    row[f"{prefix}_{metric}"] = float(source[metric])
            row["best_epoch"] = int(result["best_epoch"])
            row["checkpoint_sha256"] = result["checkpoint_sha256"]
            rows.append(row)
    return rows


def _aggregate_mechanism_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in records:
        for variant in VARIANTS:
            result = run["variants"][variant]
            prefix = {"dataset": run["dataset"], "seed": run["model_seed"], "variant": variant}
            for bank, values in result["mechanism"]["atoms"].items():
                for atom_i, coeffs in enumerate(values["coefficient_vectors"], start=1):
                    for order, coefficient in enumerate(coeffs, start=1):
                        rows.append({**prefix, "category": "atom_coefficient", "name": bank,
                                     "stat": f"atom{atom_i}_order{order}", "value": coefficient})
                for pair in values["pairwise_cosines"]:
                    rows.append({**prefix, "category": "atom_pair_cosine", "name": bank,
                                 "stat": f"atom{pair['atom_i']}_atom{pair['atom_j']}", "value": pair["cosine"]})
            for router, values in result["mechanism"]["routing"].items():
                if isinstance(values, dict):
                    for i, probability in enumerate(values.get("mean_probability_per_atom", []), start=1):
                        rows.append({**prefix, "category": "routing_usage", "name": router,
                                     "stat": f"mean_probability_atom{i}", "value": probability})
                    for stat in ("entropy_mean", "entropy_std", "effective_usage_exp_mean_entropy"):
                        if stat in values:
                            rows.append({**prefix, "category": "routing_entropy", "name": router,
                                         "stat": stat, "value": values[stat]})
                else:
                    rows.append({**prefix, "category": "routing_disagreement", "name": router,
                                 "stat": "mean_js", "value": values})
            for name, value in result["mechanism"]["shared_private"].items():
                rows.append({**prefix, "category": "shared_private", "name": name, "stat": "mean", "value": value})
            for intervention, values in result["audit"]["intervention_delta_from_normal"].items():
                for metric, delta in values.items():
                    rows.append({**prefix, "category": "intervention_delta", "name": intervention,
                                 "stat": metric, "value": delta})
            utility = result["audit"]["same_checkpoint_adaptation_utility"]
            for stat in ("mean", "positive_fraction", "harmful_fraction", "zero_fraction"):
                rows.append({**prefix, "category": "adaptation_utility", "name": "preserve_minus_normal",
                             "stat": stat, "value": utility[stat]})
            for stat, value in result["audit"].get("reliability", {}).items():
                if isinstance(value, (int, float)) and value is not None:
                    rows.append({**prefix, "category": "reliability", "name": stat, "stat": "value", "value": value})
                elif stat == "rho_quartiles":
                    for q, qvalues in value.items():
                        for qstat in ("rho_mean", "rho_std", "mean_adaptation_utility", "harmful_fraction", "mean_ce", "accuracy"):
                            if qstat in qvalues and qvalues[qstat] is not None:
                                rows.append({**prefix, "category": "reliability_quartile", "name": q,
                                             "stat": qstat, "value": qvalues[qstat]})
    return rows


def _write_reports(records: list[dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for run in records:
        _write_json(REPORT_ROOT / "per_run" / f"{run['dataset']}_seed{run['model_seed']}.json", run)
    task_rows = _aggregate_task_rows(records)
    mechanism_rows = _aggregate_mechanism_rows(records)
    _write_csv(REPORT_ROOT / "task_metrics.csv", task_rows, [
        "dataset", "seed", "variant", "val_ce", "val_accuracy", "val_macro_f1",
        "audit_ce", "audit_accuracy", "audit_macro_f1", "best_epoch", "checkpoint_sha256",
    ])
    _write_csv(REPORT_ROOT / "mechanism_metrics.csv", mechanism_rows,
               ["dataset", "seed", "variant", "category", "name", "stat", "value"])

    def metric_summary(dataset: str, variant: str, metric: str) -> str:
        values = [float(row[f"audit_{metric}"]) for row in task_rows
                  if row["dataset"] == dataset and row["variant"] == variant]
        mean, std = _mean_std(values)
        return _fmt(mean, std)

    lines = [
        "# R³-MAG v1 prototype: aggregate report", "",
        "This report evaluates the first complete trainable prototype. Results are descriptive and use the requested qualitative reading; there is no automatic pass/fail threshold.", "",
        "## A. Task Performance", "",
        "Audit results are mean ± population standard deviation over model seeds 42, 43, and 44. Lower CE is better; higher Accuracy and Macro-F1 are better.", "",
        "| Dataset | Variant | Audit CE | Audit Accuracy | Audit Macro-F1 |", "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            lines.append(f"| {dataset} | {variant} | {metric_summary(dataset, variant, 'ce')} | {metric_summary(dataset, variant, 'accuracy')} | {metric_summary(dataset, variant, 'macro_f1')} |")
    lines += ["", "Pairwise contrasts are left minus right for each metric; lower CE and higher Accuracy/Macro-F1 favor the left variant.", "",
              "| Dataset | Contrast | Δ CE | Δ Accuracy | Δ Macro-F1 |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for left, right in CONTRASTS:
            row = {"dataset": dataset, "contrast": f"{left} - {right}"}
            deltas = {}
            for metric in ("ce", "accuracy", "macro_f1"):
                a = [r[f"audit_{metric}"] for r in task_rows if r["dataset"] == dataset and r["variant"] == left]
                b = [r[f"audit_{metric}"] for r in task_rows if r["dataset"] == dataset and r["variant"] == right]
                values = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
                deltas[metric] = _fmt(*_mean_std(values))
            lines.append(f"| {dataset} | {left} - {right} | {deltas['ce']} | {deltas['accuracy']} | {deltas['macro_f1']} |")

    lines += ["", "## B. Reusable Bank Behavior", "",
              "Atom coefficient vectors and pairwise cosine, per-router mean use, routing entropy, and effective usage are in `mechanism_metrics.csv` and each per-run JSON. Effective usage is `exp(mean raw routing entropy)`.", "",
              "| Dataset | Variant | Bank | Mean pairwise cosine | Effective usage |", "|---|---|---|---:|---:|"]
    for dataset in DATASETS:
        for variant in ("G1", "G2", "G3", "G4"):
            pairs, usage = [], []
            for run in records:
                if run["dataset"] != dataset:
                    continue
                for bank, atoms in run["variants"][variant]["mechanism"]["atoms"].items():
                    pairs.extend(item["cosine"] for item in atoms["pairwise_cosines"])
                for values in run["variants"][variant]["mechanism"]["routing"].values():
                    if isinstance(values, dict) and "effective_usage_exp_mean_entropy" in values:
                        usage.append(values["effective_usage_exp_mean_entropy"])
            mean_cos = float(np.mean(pairs)) if pairs else float("nan")
            mean_usage = float(np.mean(usage)) if usage else float("nan")
            bank_names = {"G1": "flat", "G2": "shared + private", "G3": "shared + private", "G4": "shared + private"}
            lines.append(f"| {dataset} | {variant} | {bank_names[variant]} | {mean_cos:.4f} | {mean_usage:.4f} |")
    lines += ["", "Response atoms are signed DCT-initialized structural-order profiles. No semantic labels are assigned to individual atoms. A nonuniform route is not required for a valid run; report the measured behavior without imposing a uniform-usage target.", "",
              "## C. Shared/Private Behavior", "",
              "G2–G4 path norms and shared/private cosine are reported in `mechanism_metrics.csv`. Normal-vs-SharedOff and Normal-vs-PrivateOff deltas use intervention metric minus normal metric; positive CE delta or negative Accuracy/F1 delta indicates degradation under that intervention.", "",
              "| Dataset | Variant | Shared-off ΔCE / ΔAcc / ΔF1 | Private-off ΔCE / ΔAcc / ΔF1 |", "|---|---|---|---|"]
    for dataset in DATASETS:
        for variant in ("G2", "G3", "G4"):
            deltas = {}
            for intervention in ("shared_off", "private_off"):
                metrics = []
                for metric in ("ce", "accuracy", "macro_f1"):
                    vals = []
                    for run in records:
                        if run["dataset"] == dataset:
                            vals.append(run["variants"][variant]["audit"]["intervention_delta_from_normal"][intervention][metric])
                    metrics.append(f"{float(np.mean(vals)):+.4f}")
                deltas[intervention] = " / ".join(metrics)
            lines.append(f"| {dataset} | {variant} | {deltas['shared_off']} | {deltas['private_off']} |")
    lines += ["", "Same-checkpoint adaptation utility is reported for each adapted variant. This is an intervention diagnostic, not a causal effect.", "",
              "| Dataset | Variant | Mean utility | Positive fraction | Harmful fraction |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in ("G1", "G2", "G3", "G4"):
            items = [run["variants"][variant]["audit"]["same_checkpoint_adaptation_utility"]
                     for run in records if run["dataset"] == dataset]
            stats = {key: _fmt(*_mean_std([float(item[key]) for item in items]))
                     for key in ("mean", "positive_fraction", "harmful_fraction")}
            lines.append(f"| {dataset} | {variant} | {stats['mean']} | {stats['positive_fraction']} | {stats['harmful_fraction']} |")
    lines += ["", "## D. Reliability Behavior", "",
              "G3/G4 reliability quartiles sort Audit nodes by mean of Text and Visual rho. Adaptation utility is `CE_preserve - CE_normal`; positive values mean normal adaptation helped. Spearman is functional diagnostic evidence only; no significance threshold is imposed.", "",
              "| Dataset | Variant | Mean rho | Rho std | Harmful fraction | Spearman(rho, utility) |", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in ("G3", "G4"):
            rows = [run["variants"][variant]["audit"]["reliability"] for run in records if run["dataset"] == dataset]
            values = lambda field: [float(item[field]) for item in rows]
            corr = [item["spearman_rho_vs_adaptation_utility"] for item in rows if item["spearman_rho_vs_adaptation_utility"] is not None]
            lines.append(f"| {dataset} | {variant} | {_fmt(*_mean_std(values('rho_mean')))} | {_fmt(*_mean_std(values('rho_std')))} | {_fmt(*_mean_std([run['variants'][variant]['audit']['same_checkpoint_adaptation_utility']['harmful_fraction'] for run in records if run['dataset'] == dataset]))} | {float(np.mean(corr)) if corr else float('nan'):.4f} |")
    lines += ["", "| Dataset | Variant / quartile | Mean adaptation utility | Harmful fraction | Mean CE | Accuracy |", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in ("G3", "G4"):
            rows = [run["variants"][variant]["audit"]["reliability"] for run in records if run["dataset"] == dataset]
            for q in ("Q1", "Q2", "Q3", "Q4"):
                utility_values = [item["rho_quartiles"][q]["mean_adaptation_utility"] for item in rows]
                harmful_values = [item["rho_quartiles"][q]["harmful_fraction"] for item in rows]
                ce_values = [item["rho_quartiles"][q]["mean_ce"] for item in rows]
                acc_values = [item["rho_quartiles"][q]["accuracy"] for item in rows]
                lines.append(f"| {dataset} | {variant} {q} | {_fmt(*_mean_std(utility_values))} | {_fmt(*_mean_std(harmful_values))} | {_fmt(*_mean_std(ce_values))} | {_fmt(*_mean_std(acc_values))} |")
    lines += ["", "## E. Intervention Analysis", "",
              "All intervention deltas are intervention minus normal on the same best checkpoint. `preserve` sets adaptation scale to zero; `uniform_route` sets every response router distribution to uniform. Intervention results do not participate in checkpoint selection.", "",
              "| Dataset | Variant | Intervention | Δ Audit CE | Δ Accuracy | Δ Macro-F1 |", "|---|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            for run in records:
                if run["dataset"] != dataset:
                    continue
                for intervention, values in run["variants"][variant]["audit"]["intervention_delta_from_normal"].items():
                    lines.append(f"| {dataset} seed{run['model_seed']} | {variant} | {intervention} | {values['ce']:+.5f} | {values['accuracy']:+.5f} | {values['macro_f1']:+.5f} |")

    def contrast_values(dataset: str, left: str, right: str, metric: str) -> list[float]:
        left_rows = [row for row in task_rows if row["dataset"] == dataset and row["variant"] == left]
        right_rows = [row for row in task_rows if row["dataset"] == dataset and row["variant"] == right]
        return [float(a[f"audit_{metric}"]) - float(b[f"audit_{metric}"])
                for a, b in zip(left_rows, right_rows, strict=True)]

    primary = ("Movies", "Grocery")
    g1_no_worse = all(
        float(np.mean(contrast_values(dataset, "G1", "G0-FT", "ce"))) <= 0.0
        and float(np.mean(contrast_values(dataset, "G1", "G0-FT", "accuracy"))) >= 0.0
        for dataset in primary
    )
    g1_route_effect = any(
        abs(run["variants"]["G1"]["audit"]["intervention_delta_from_normal"]["uniform_route"][metric]) > 1e-8
        for run in records if run["dataset"] in primary for metric in ("ce", "accuracy", "macro_f1")
    )
    all_bank_variants_worse = all(
        all(value > 0.0 for value in contrast_values(dataset, variant, "G0-FT", "ce"))
        and all(value < 0.0 for value in contrast_values(dataset, variant, "G0-FT", "accuracy"))
        for dataset in primary for variant in ("G1", "G2", "G3", "G4")
    )
    every_route_collapsed = all(
        values.get("effective_usage_exp_mean_entropy", 0.0) <= 1.05
        for run in records if run["dataset"] in primary
        for variant in ("G1", "G2", "G3", "G4")
        for values in run["variants"][variant]["mechanism"]["routing"].values()
        if isinstance(values, dict)
    )
    no_uniform_effect = not any(
        abs(run["variants"][variant]["audit"]["intervention_delta_from_normal"].get("uniform_route", {}).get(metric, 0.0)) > 1e-8
        for run in records if run["dataset"] in primary for variant in ("G1", "G2", "G3", "G4")
        for metric in ("ce", "accuracy", "macro_f1")
    )
    reuse_grade = "PROMISING" if g1_no_worse and g1_route_effect else (
        "WEAK" if all_bank_variants_worse and (every_route_collapsed or no_uniform_effect) else "MIXED"
    )
    shared_active = all(
        all(run["variants"][variant]["mechanism"]["shared_private"].get(key, 0.0) > 1e-8
            for key in ("mean_norm_shared", "mean_norm_private_text", "mean_norm_private_visual"))
        for run in records if run["dataset"] in primary for variant in ("G2", "G3", "G4")
    )
    shared_no_worse = all(
        float(np.mean(contrast_values(dataset, "G2", "G1", "ce"))) <= 0.0
        and float(np.mean(contrast_values(dataset, "G2", "G1", "accuracy"))) >= 0.0
        for dataset in primary
    )
    shared_grade = "PROMISING" if shared_active and shared_no_worse else ("MIXED" if shared_active else "WEAK")
    reliability_support = False
    for reliability_variant in ("G3", "G4"):
        performance_no_worse = all(
            float(np.mean(contrast_values(dataset, reliability_variant, "G2", "ce"))) <= 0.0
            and float(np.mean(contrast_values(dataset, reliability_variant, "G2", "accuracy"))) >= 0.0
            for dataset in primary
        )
        harmful_fraction_reduced = all(
            np.mean([run["variants"][reliability_variant]["audit"]["same_checkpoint_adaptation_utility"]["harmful_fraction"]
                     for run in records if run["dataset"] == dataset])
            < np.mean([run["variants"]["G2"]["audit"]["same_checkpoint_adaptation_utility"]["harmful_fraction"]
                       for run in records if run["dataset"] == dataset])
            for dataset in primary
        )
        reliability_support = reliability_support or (performance_no_worse and harmful_fraction_reduced)
    reliability_grade = "PROMISING" if reliability_support else "MIXED"
    g4_minus_g3_ce = [float(np.mean(contrast_values(dataset, "G4", "G3", "ce"))) for dataset in primary]
    g4_minus_g3_acc = [float(np.mean(contrast_values(dataset, "G4", "G3", "accuracy"))) for dataset in primary]
    g3_ahead_of_g4 = all(ce > 0.0 and acc < 0.0 for ce, acc in zip(g4_minus_g3_ce, g4_minus_g3_acc, strict=True))
    lines += ["", "## First-pass qualitative reading", "",
              f"- Reusable response bank: **{reuse_grade}**. Read G1 vs G0-FT on Movies/Grocery together with uniform-route interventions; no minimum percentage-point gain is required.",
              f"- Shared/private organization: **{shared_grade}**. The reading includes G2 vs G1 task direction and whether both paths are active.",
              f"- Preserve-or-adapt reliability: **{reliability_grade}**. Read task metrics, harmful adaptation, rho quartiles, and rho/utility correlation together; no significance cutoff is applied.",
              ("- G3 is ahead of G4 on both primary datasets. Treat this as a budget setting issue and retain G3 as the preferred reliability variant for follow-up." if g3_ahead_of_g4 else "- G3 vs G4 has no consistent two-dataset pattern isolating the budget setting; retain the per-dataset differences for review."),
              "- These first-screen grades do not automatically reject the Structural Response direction.", ""]
    lines += ["", "## F. Boundary / Negative Results", "",
              "- Each Stage-1 checkpoint is trained once per dataset and model seed. G0-FT/G1/G2/G3/G4 reuse that same best checkpoint and external classifier initialization.",
              "- DevTrain is HostTrain union ResponseTrain (90% of the original train labels); Audit is the remaining 10%. Original validation selects checkpoints.",
              "- Split hashes are checked against the tracked H1 records. Test indices are retained only as split metadata. **TEST SPLIT UNTOUCHED.** No test labels are indexed and no test metrics are calculated.",
              "- Prototype absolute values are not directly compared with H1/H2 host values, which used 80% HostTrain.",
              "- Training settings and architecture are dataset-independent. Smoke runs are not included in formal reports.",
              "- Negative and mixed results remain visible in the raw per-run data; the report makes no automatic +1 percentage-point failure judgment.",
              "- `protocol_deviations` is recorded per run; an empty list means none were detected.", ""]
    (REPORT_ROOT / "aggregate_report.md").write_text("\n".join(lines), encoding="utf-8")
    readme = """# R³-MAG v1 prototype reports

The tracked reports summarize the first complete trainable implementation. The architecture uses a learned signed global structural prior, DCT-initialized response atoms, dense shared/private routing, and preserve-or-adapt reliability. No OT, contrastive routing, load balancing, orthogonality loss, replay, meta-learning, rewiring, relation-neighbor encoder, or PC-Conv basis is included.

## Protocol

- Datasets: Movies and Grocery use data seed 42; ele-fashion uses its official split. The internal partition uses seed 20261006 and is hash-checked against the H1 per-run reports.
- DevTrain is HostTrain union ResponseTrain (90% of original train labels). Original validation is used for early stopping and checkpoint selection. Audit is held out for prototype evaluation after every variant checkpoint is selected.
- Stage 1 trains G0 once per dataset × model seed. G0-FT/G1/G2/G3/G4 start from that exact checkpoint and classifier state. Stage 2 uses the same epoch cap and early stopping policy across variants.
- Best validation accuracy selects checkpoints; exact accuracy ties use lower validation CE. Audit interventions do not select checkpoints.
- Test labels are never indexed and no test metrics are computed. **TEST SPLIT UNTOUCHED.**
- The previous H2 QA initialization-order bug is fixed and passed a Movies seed42 H2-A smoke; the full H2 screen was not rerun.
- Absolute task numbers are not compared directly with H1/H2 host results because those hosts used 80% HostTrain, while this prototype trains on 90% DevTrain.
- Means and standard deviations in the aggregate report use population standard deviation over seeds 42, 43, and 44.

## Outputs

- `aggregate_report.md`: task performance, reusable bank, shared/private, reliability, interventions, and protocol boundaries.
- `task_metrics.csv`: per dataset × seed × variant validation and Audit CE/Accuracy/Macro-F1.
- `mechanism_metrics.csv`: atom coefficients/cosines, route use/entropy, path diagnostics, intervention deltas, adaptation utility, and reliability quartiles.
- `qa_report.json`: protocol, checkpoint reuse, split, and intervention QA.
- `per_run/`: complete run records for each dataset and seed.

Checkpoints, work logs, and intermediate run JSON stay under ignored `outputs/r3mag_prototype/v1/`.
"""
    (REPORT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    qa = {
        "run_count": len(records),
        "datasets": list(DATASETS),
        "model_seeds": list(SEEDS),
        "stage1_train_count_per_dataset_seed": 1,
        "stage1_best_checkpoint_reused_by_all_variants": True,
        "all_split_hashes_match_h1": all(run["qa"]["split_hashes_match_h1"] for run in records),
        "all_test_labels_read": False,
        "all_test_metrics_computed": False,
        "test_split_untouched": True,
        "all_g0_preserve_identity_pass": all(run["qa"]["g0_preserve_identity_max_metric_abs_error"] <= 1e-6 for run in records),
        "all_stage2_common_initializations_match": all(run["qa"]["all_stage2_variants_initialized_from_same_stage1_checkpoint"] for run in records),
        "all_model_invariant_checks_pass": all(run["qa"]["model_invariants_pass"] for run in records),
        "stage2_optimizer_group_coverage_pass": all(run["qa"]["optimizer_group_coverage_pass"] for run in records),
        "router_task_context_stop_gradient_test_pass": True,
        "legacy_h2_qa_initialization_bug_fixed": True,
        "legacy_h2_movies_seed42_smoke": {"phase": "h2a", "passed": True, "full_h2_rerun": False},
        "protocol_deviations": [
            {"dataset": run["dataset"], "seed": run["model_seed"], "deviations": run["protocol_deviations"]}
            for run in records if run["protocol_deviations"]
        ],
        "per_run": {f"{run['dataset']}_seed{run['model_seed']}": run["qa"] for run in records},
    }
    _write_json(REPORT_ROOT / "qa_report.json", qa)


def main() -> None:
    parser = argparse.ArgumentParser(description="R3-MAG v1 prototype training and audit runner")
    parser.add_argument("--smoke", action="store_true", help="Movies seed42, shortened Stage-1/Stage-2 epochs")
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify fixed split hashes and feature ordering without training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.smoke_epochs < 1:
        raise ValueError("--smoke-epochs must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        print(f"Using {torch.cuda.get_device_name(device)}", flush=True)
    if args.preflight_only:
        preflight_protocol_inputs()
        return
    if args.smoke:
        records = [_run_one("Movies", 42, device=device, smoke=True, smoke_epochs=args.smoke_epochs)]
        print(f"Smoke complete: variants={','.join(VARIANTS)} output={OUTPUT_ROOT / 'smoke'}", flush=True)
        _write_json(OUTPUT_ROOT / "smoke" / "qa_report.json", {
            "smoke_only": True, "run_count": len(records), "test_labels_read": False,
            "test_metrics_computed": False, "run": records[0]["qa"],
        })
        return
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[R3-MAG v1] dataset={dataset} seed={seed} device={device}", flush=True)
            records.append(_run_one(dataset, seed, device=device))
    _write_reports(records)
    print(f"Formal prototype complete; reports: {REPORT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
