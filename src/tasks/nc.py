from __future__ import annotations

import json
import logging
import hashlib
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader as TorchDataLoader
from omegaconf import OmegaConf

from src.data import MAGData
from src.models import build_model
from src.tasks.common import (
    build_optimizer,
    clone_state_dict,
    load_state_dict_cpu,
    scheduler_step,
)
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode
from src.utils.metrics import format_pct
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _resolve_training_mode(cfg, model=None) -> str:
    """Resolve the frozen NC path independently of model capability flags."""
    mode = str(cfg.task.get("training_mode", "full_graph")).strip().lower()
    if mode != "full_graph":
        raise ValueError(
            "unified_full_graph_nc_v1 requires task.training_mode='full_graph'; "
            f"got {mode!r}"
        )
    # MLP has no graph computation; it uses feature-only node minibatches.
    # Graph models always use one full graph forward, even if a model preset
    # advertises full_graph_training=False.
    return "full_graph" if _uses_graph_encoder(cfg) else "feature_only"


def _resolve_nc_eval_labels(data: MAGData, *, include_test: bool = True) -> list[int]:
    """Use one stable Macro-F1 label set across the supervised task splits."""
    if data.y is None:
        raise ValueError("NC data must contain labels")
    if data.num_classes is None:
        raise ValueError("NC data must define num_classes")
    split_indices = [idx for idx in (data.train_idx, data.val_idx) if idx is not None]
    if include_test and data.test_idx is not None:
        split_indices.append(data.test_idx)
    if not split_indices:
        raise ValueError("NC data must contain at least one supervised split")
    all_indices = torch.cat([idx.reshape(-1) for idx in split_indices]).to(data.y.device)
    observed = data.y[all_indices].detach().cpu()
    valid = observed[(observed >= 0) & (observed < int(data.num_classes))]
    labels = sorted({int(value) for value in valid.tolist()})
    if not labels:
        raise ValueError("NC supervised splits contain no valid class labels")
    return labels


@torch.no_grad()
def _evaluate_split(
    classifier: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
    eval_labels: list[int] | tuple[int, ...],
) -> dict[str, float]:
    classifier.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_idx in TorchDataLoader(idx.cpu(), batch_size=batch_size, shuffle=False):
        logits = classifier(z[batch_idx].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
        targets.append(labels[batch_idx].cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": float(
            f1_score(
                target.numpy(),
                pred.numpy(),
                labels=list(eval_labels),
                average="macro",
                zero_division=0,
            )
        ),
    }


@torch.no_grad()
def _evaluate_indices_from_embeddings(
    classifier: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Compute CE/accuracy on cached CPU embeddings without touching CUDA RNG."""
    del device
    classifier.eval()
    indices = idx.detach().cpu().long()
    labels_cpu = labels.detach().cpu()
    weight = classifier.weight.detach().cpu()
    bias = classifier.bias.detach().cpu() if classifier.bias is not None else None
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch_indices in indices.split(int(batch_size)):
        logits = nn.functional.linear(z[batch_indices], weight, bias)
        targets = labels_cpu[batch_indices]
        loss_sum += float(
            nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        )
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total += int(targets.numel())
    if total == 0:
        raise ValueError("cannot evaluate an empty node index")
    return {"cross_entropy": loss_sum / total, "acc": correct / total}


def _parameter_l2_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().double().square().sum().cpu())
    return total**0.5


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _extended_parameter_audit(model, classifier, optimizer) -> dict[str, object]:
    parameter_names = {
        id(parameter): f"model.{name}" for name, parameter in model.named_parameters()
    }
    parameter_names.update(
        {
            id(parameter): f"classifier.{name}"
            for name, parameter in classifier.named_parameters()
        }
    )
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        groups.append(
            {
                "group_index": index,
                "parameter_tensors": len(parameters),
                "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_names": [
                    parameter_names.get(id(parameter), f"unnamed_parameter_{position}")
                    for position, parameter in enumerate(parameters)
                ],
            }
        )
    return {
        "model_l2": _parameter_l2_norm(model.parameters()),
        "classifier_l2": _parameter_l2_norm(classifier.parameters()),
        "combined_l2": _parameter_l2_norm(
            list(model.parameters()) + list(classifier.parameters())
        ),
        "model_state_sha256": _module_state_sha256(model),
        "classifier_state_sha256": _module_state_sha256(classifier),
        "optimizer_groups": groups,
    }


def _checkpoint_path_for_run(path_like: str | Path, cfg, run_id: int) -> Path:
    path = Path(str(path_like))
    if int(cfg.num_runs) > 1:
        path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_single_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    run_id: int,
    seed: int,
    eval_labels: list[int],
) -> dict[str, float]:
    inference_mode = resolve_inference_mode(cfg)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info)
    if bool(getattr(model, "collect_v7_diagnostics", False)) and resolve_inference_mode(cfg) != "full":
        raise ValueError("PSCE-MAG V7A requires task.inference_mode='full'")
    if bool(getattr(model, "requires_global_semantic_candidates", False)):
        cache_path = cfg.model.get("semantic_cache_path")
        if not cache_path:
            raise ValueError("PSCE-MAG NC requires model.semantic_cache_path")
        cache_kwargs = {
            "expected_dataset": str(cfg.dataset.name),
            "expected_feature_fingerprint": cfg.model.get("semantic_feature_fingerprint"),
            "expected_candidate_fingerprint": cfg.model.get("semantic_candidate_fingerprint"),
        }
        if hasattr(model, "semantic_v7b_fingerprint"):
            cache_kwargs["expected_v7b_fingerprint"] = cfg.model.get("semantic_v7b_fingerprint")
        model.load_semantic_cache(str(cache_path), **cache_kwargs)
    model = model.to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = build_optimizer(
        list(model.parameters()) + list(classifier.parameters()), cfg, model=model
    )
    criterion = nn.CrossEntropyLoss()
    training_mode = _resolve_training_mode(cfg, model)
    uses_graph = _uses_graph_encoder(cfg)
    full_graph_training = training_mode == "full_graph"

    # Graph encoders use the entire graph every epoch. Only MLP receives
    # feature-only minibatches.
    x_all = data.x.to(device)
    y_all = data.y.to(device)
    edge_index_all = data.edge_index.to(device) if uses_graph else None
    train_idx_all = data.train_idx.to(device)
    train_loader = None
    if not uses_graph:
        train_loader = TorchDataLoader(
            data.train_idx.cpu(),
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )

    logger.info(
        "[Run %d/%d] seed=%d | model params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(classifier),
    )
    logger.info("Protocol: %s", str(cfg.task.protocol_version))
    logger.info("Training mode: %s", training_mode)
    logger.info("Loader: %s", "FullGraph" if uses_graph else "NodeDataLoader")
    logger.info("Inference mode: %s", inference_mode)
    logger.info(
        "Model parameters=%d | Classifier parameters=%d | optimizer=%s | groups=%s | hidden_dim=%s | num_layers=%s",
        count_parameters(model),
        count_parameters(classifier),
        type(optimizer).__name__,
        [
            {"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])}
            for group in optimizer.param_groups
        ],
        cfg.model.get("hidden_dim", None),
        cfg.model.get("num_layers", None),
    )

    best_val = -1.0
    best_metrics: dict[str, float] = {}
    best_model_state = None
    best_head_state = None
    best_epoch: int | None = None
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    min_delta = float(cfg.task.get("early_stop_min_delta", 0.0))
    grad_clip = float(cfg.task.get("grad_clip", 1.0))
    aux_weight = float(cfg.task.loss.aux_weight)
    inference_batch_size = int(cfg.task.inference_batch_size)
    eval_every = int(cfg.task.eval_every)
    if eval_every < 1:
        raise ValueError("task.eval_every must be >= 1")

    record_run_telemetry = bool(getattr(model, "record_run_telemetry", False))
    record_epoch_history = bool(getattr(model, "record_epoch_history", False))
    record_extended_validation = bool(
        cfg.task.get("record_extended_validation", False)
    )
    if record_extended_validation and not full_graph_training:
        raise ValueError("extended validation is currently scoped to full-graph NC")
    if record_extended_validation and bool(cfg.task.get("evaluate_test", True)):
        raise ValueError("extended-validation study requires evaluate_test=false")
    if record_extended_validation and not cfg.task.get("best_f1_ckpt_path"):
        raise ValueError("extended-validation study requires task.best_f1_ckpt_path")
    if record_extended_validation and not cfg.task.get("save_ckpt_path"):
        raise ValueError("extended-validation study requires task.save_ckpt_path")
    epoch_history: list[dict[str, float | int]] = []
    epoch_wall_seconds: list[float] = []
    train_step_seconds: list[float] = []
    optimizer_update_count = 0
    best_f1 = -1.0
    best_f1_epoch: int | None = None
    best_f1_metrics: dict[str, float] | None = None
    best_f1_model_state = None
    best_f1_head_state = None
    best_acc_extended_metrics: dict[str, float] | None = None
    best_f1_extended_metrics: dict[str, float] | None = None
    initial_parameter_audit = (
        _extended_parameter_audit(model, classifier, optimizer)
        if record_extended_validation
        else None
    )
    if record_run_telemetry and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()

    for epoch in range(1, int(cfg.task.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        classifier.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        if full_graph_training:
            if hasattr(model, "_batch_n_id"):
                model._batch_n_id = None
            z, _, _, aux_loss, _ = model(x_all, edge_index_all)
            logits = classifier(z[train_idx_all])
            labels = y_all[train_idx_all]
            loss = nn.functional.cross_entropy(logits, labels) + aux_weight * aux_loss
            loss.backward()
            if epoch == 1 and hasattr(model, "capture_gradient_diagnostics"):
                model.capture_gradient_diagnostics()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer_update_count += 1
            train_loss = float(loss.detach().item())
            del z
        else:
            total_loss = 0.0
            total_examples = 0
            for batch_idx in train_loader:
                batch_idx = batch_idx.to(device)
                labels = y_all[batch_idx]
                z, _, _, aux_loss, _ = model(x_all[batch_idx], None)
                logits = classifier(z)
                loss = nn.functional.cross_entropy(logits, labels) + aux_weight * aux_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if epoch == 1 and hasattr(model, "capture_gradient_diagnostics"):
                    model.capture_gradient_diagnostics()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()),
                    max_norm=grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer_update_count += 1
                total_loss += float(loss.detach().item()) * int(labels.numel())
                total_examples += int(labels.numel())
            train_loss = total_loss / max(total_examples, 1)

        train_step_seconds.append(time.perf_counter() - epoch_started)

        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        if epoch % eval_every != 0:
            epoch_wall_seconds.append(time.perf_counter() - epoch_started)
            logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
            continue

        z = infer_all_embeddings(
            model, data, device, uses_graph, inference_batch_size, inference_mode
        )
        val_metrics = _evaluate_split(
            classifier,
            z,
            data.y,
            data.val_idx,
            device,
            inference_batch_size,
            eval_labels,
        )
        extended_epoch_metrics: dict[str, float] | None = None
        if record_extended_validation:
            # Reuse the existing eval embedding tensor. These linear-head-only
            # evaluations are deterministic and do not consume model/dropout RNG.
            train_eval = _evaluate_indices_from_embeddings(
                classifier,
                z,
                data.y,
                data.train_idx,
                device,
                inference_batch_size,
            )
            val_eval = _evaluate_indices_from_embeddings(
                classifier,
                z,
                data.y,
                data.val_idx,
                device,
                inference_batch_size,
            )
            extended_epoch_metrics = {
                "train_ce": train_loss,
                "train_accuracy_eval": train_eval["acc"],
                "val_ce": val_eval["cross_entropy"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        logger.info(
            "Epoch %05d | Train Loss %.4f | Val Acc %.2f | Val Macro-F1 %.2f",
            epoch,
            train_loss,
            format_pct(val_metrics["acc"]),
            format_pct(val_metrics["macro_f1"]),
        )
        epoch_wall_seconds.append(time.perf_counter() - epoch_started)
        if record_epoch_history:
            history_row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "epoch_wall_seconds": epoch_wall_seconds[-1],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if extended_epoch_metrics is not None:
                history_row.update(extended_epoch_metrics)
            epoch_history.append(history_row)

        improved = val_metrics["acc"] > best_val + min_delta
        stop_early = False
        if improved:
            best_val = val_metrics["acc"]
            best_epoch = epoch
            best_metrics = {
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            if extended_epoch_metrics is not None:
                best_acc_extended_metrics = dict(extended_epoch_metrics)
            best_model_state = clone_state_dict(model)
            best_head_state = clone_state_dict(classifier)
            patience_left = patience_total
        elif epoch >= min_epoch:
            patience_left -= 1
            logger.info(
                "Patience %d/%d | Best Val Acc %.2f",
                patience_total - patience_left,
                patience_total,
                format_pct(best_val),
            )
            stop_early = patience_left <= 0
        if extended_epoch_metrics is not None and val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_f1_epoch = epoch
            best_f1_metrics = {
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            best_f1_extended_metrics = dict(extended_epoch_metrics)
            best_f1_model_state = clone_state_dict(model)
            best_f1_head_state = clone_state_dict(classifier)
        del z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stop_early:
            logger.info("Early stopping at epoch %03d", epoch)
            break

    if best_model_state is None or best_head_state is None:
        raise RuntimeError("NC training finished without a validation-selected checkpoint")
    if record_extended_validation and (
        best_f1_model_state is None or best_f1_head_state is None
    ):
        raise RuntimeError("extended-validation study finished without a Best-F1 checkpoint")
    final_parameter_audit = (
        _extended_parameter_audit(model, classifier, optimizer)
        if record_extended_validation
        else None
    )
    load_state_dict_cpu(model, best_model_state)
    load_state_dict_cpu(classifier, best_head_state)

    semantic_checkpoint_diagnostics = None
    if bool(getattr(model, "collect_v7_diagnostics", False)) and model.variant != "a0_raw":
        model.eval()
        with torch.no_grad():
            z_semantic, _, _, _, semantic_info = model(
                x_all, edge_index_all, return_details=True
            )
            z_semantic_cpu = z_semantic.detach().cpu()
            semantic_checkpoint_diagnostics = semantic_info.get("details", {})
            del z_semantic, semantic_info

            model.set_semantic_enabled(False)
            z_physical, _, _, _, _ = model(x_all, edge_index_all)
            z_physical_cpu = z_physical.detach().cpu()
            closed_metrics = _evaluate_split(
                classifier,
                z_physical_cpu,
                data.y,
                data.val_idx,
                device,
                inference_batch_size,
                eval_labels,
            )
            semantic_checkpoint_diagnostics["closed_path"] = {
                "output_delta_rms": float(
                    (z_semantic_cpu - z_physical_cpu).float().square().mean().sqrt()
                ),
                "output_rms": float(z_semantic_cpu.float().square().mean().sqrt()),
                "val_acc": closed_metrics["acc"],
                "val_macro_f1": closed_metrics["macro_f1"],
                "val_acc_delta_from_open": closed_metrics["acc"] - best_metrics["val_acc"],
                "val_macro_f1_delta_from_open": (
                    closed_metrics["macro_f1"] - best_metrics["val_macro_f1"]
                ),
            }
            model.set_semantic_enabled(True)
            del z_semantic_cpu, z_physical, z_physical_cpu
        model.train()

    if bool(cfg.task.get("evaluate_test", True)):
        z = infer_all_embeddings(
            model, data, device, uses_graph, inference_batch_size, inference_mode
        )
        test_metrics = _evaluate_split(
            classifier,
            z,
            data.y,
            data.test_idx,
            device,
            inference_batch_size,
            eval_labels,
        )
        best_metrics["test_acc"] = test_metrics["acc"]
        best_metrics["test_macro_f1"] = test_metrics["macro_f1"]
        del z

    run_metadata = {
        "best_epoch": int(best_epoch),
        "model_parameters": count_parameters(model),
        "classifier_parameters": count_parameters(classifier),
        "optimizer": type(optimizer).__name__,
        "optimizer_groups": [
            {"lr": float(group["lr"]), "weight_decay": float(group["weight_decay"])}
            for group in optimizer.param_groups
        ],
        "hidden_dim": cfg.model.get("hidden_dim", None),
        "num_layers": cfg.model.get("num_layers", None),
        "other_depth_fields": {
            name: cfg.model.get(name)
            for name in ("d_model", "q_dim", "mp_hops")
            if cfg.model.get(name) is not None
        },
    }
    if record_run_telemetry:
        run_metadata.update(
            {
                "training_wall_seconds": float(time.perf_counter() - training_started),
                "mean_epoch_wall_seconds": float(
                    sum(epoch_wall_seconds) / max(len(epoch_wall_seconds), 1)
                ),
                "mean_train_step_seconds": float(
                    sum(train_step_seconds) / max(len(train_step_seconds), 1)
                ),
                "epochs_completed": len(epoch_wall_seconds),
                "gradient_diagnostics": getattr(model, "_gradient_diagnostics", None),
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
                "cuda_peak_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(device))
                    if device.type == "cuda"
                    else None
                ),
                "cuda_device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "CPU"
                ),
            }
        )
    if record_epoch_history:
        run_metadata["epoch_history"] = epoch_history
    if record_extended_validation:
        study_provenance = cfg.task.get("extended_checkpoint_metadata", {})
        if OmegaConf.is_config(study_provenance):
            study_provenance = OmegaConf.to_container(study_provenance, resolve=True)
        run_metadata.update(
            {
                "extended_validation": True,
                "train_ce_mode": "training-mode stochastic forward, pre-optimizer-step loss",
                "train_accuracy_mode": "eval-mode no-grad prediction after epoch update",
                "validation_ce_mode": "eval-mode no-grad unweighted cross entropy",
                "accuracy_stopping_horizon_epochs": len(epoch_wall_seconds),
                "optimizer_update_count": optimizer_update_count,
                "initial_parameter_audit": initial_parameter_audit,
                "final_parameter_audit": final_parameter_audit,
                "dual_checkpoint_selection": {
                    "best_accuracy": {
                        "epoch": int(best_epoch),
                        "metrics": best_acc_extended_metrics,
                        "rule": "Validation Accuracy > previous best + early_stop_min_delta",
                    },
                    "best_macro_f1": {
                        "epoch": int(best_f1_epoch),
                        "metrics": best_f1_extended_metrics,
                        "rule": "highest Validation Macro-F1 through the Accuracy stopping horizon; earliest epoch wins ties",
                    },
                },
                "study_provenance": study_provenance,
            }
        )
    if semantic_checkpoint_diagnostics is not None:
        run_metadata["semantic_checkpoint_diagnostics"] = semantic_checkpoint_diagnostics
    if getattr(model, "semantic_candidate_fingerprint", None) is not None:
        run_metadata["semantic_candidate_fingerprint"] = model.semantic_candidate_fingerprint
        run_metadata["semantic_feature_fingerprint"] = model.semantic_feature_fingerprint
    if getattr(model, "semantic_v7b_fingerprint", None) is not None:
        run_metadata["semantic_v7b_fingerprint"] = model.semantic_v7b_fingerprint
    save_ckpt_path = cfg.task.get("save_ckpt_path")
    if save_ckpt_path:
        path = _checkpoint_path_for_run(save_ckpt_path, cfg, run_id)
        accuracy_checkpoint = {
                "task": "nc",
                "protocol_version": str(cfg.task.protocol_version),
                "seed": seed,
                "selection": "best_val_accuracy",
                "epoch": best_epoch,
                "metrics": (
                    dict(best_acc_extended_metrics)
                    if record_extended_validation
                    else dict(best_metrics)
                ),
                "run_metadata": run_metadata,
                "model_state": (
                    best_model_state if record_extended_validation else clone_state_dict(model)
                ),
                "head_state": (
                    best_head_state if record_extended_validation else clone_state_dict(classifier)
                ),
                "data_info": data_info,
            }
        if record_extended_validation:
            accuracy_checkpoint["selection_scope"] = (
                "best Accuracy checkpoint observed before Accuracy-driven early stopping"
            )
            torch.save(accuracy_checkpoint, path)
            f1_path = _checkpoint_path_for_run(
                cfg.task.get("best_f1_ckpt_path"), cfg, run_id
            )
            f1_checkpoint = {
                **accuracy_checkpoint,
                "selection": "best_val_macro_f1_within_accuracy_stopping_horizon",
                "epoch": int(best_f1_epoch),
                "metrics": dict(best_f1_extended_metrics),
                "model_state": best_f1_model_state,
                "head_state": best_f1_head_state,
                "selection_scope": (
                    "highest Macro-F1 among epochs observed before Accuracy-driven early stopping; earliest epoch wins ties"
                ),
            }
            torch.save(f1_checkpoint, f1_path)
            logger.info("Saved Best-Accuracy checkpoint: %s | epoch=%s", path, best_epoch)
            logger.info("Saved Best-F1 checkpoint: %s | epoch=%s", f1_path, best_f1_epoch)
        else:
            torch.save(
                {
                    "task": "nc",
                    "protocol_version": str(cfg.task.protocol_version),
                    "seed": seed,
                    "selection": "best_val_accuracy",
                    "epoch": best_epoch,
                    "metrics": dict(best_metrics),
                    "run_metadata": run_metadata,
                    "model_state": clone_state_dict(model),
                    "head_state": clone_state_dict(classifier),
                    "data_info": data_info,
                },
                path,
            )
        logger.info("Saved checkpoint: %s | best_epoch=%s", path, best_epoch)

    logger.info(
        "[Run %d] Best Val Acc %.2f | Val Macro-F1 %.2f | epoch=%d",
        run_id + 1,
        format_pct(best_metrics["val_acc"]),
        format_pct(best_metrics["val_macro_f1"]),
        int(best_epoch),
    )
    if "test_acc" in best_metrics:
        logger.info(
            "[Run %d] Test Acc %.2f | Test Macro-F1 %.2f",
            run_id + 1,
            format_pct(best_metrics["test_acc"]),
            format_pct(best_metrics["test_macro_f1"]),
        )
    return {**best_metrics, "_run_metadata": run_metadata}


def run_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, tuple[float, float]]:
    if data.y is None or data.train_idx is None or data.val_idx is None or data.test_idx is None:
        raise ValueError("NC data must contain y/train_idx/val_idx/test_idx")
    _resolve_training_mode(cfg)
    eval_labels = _resolve_nc_eval_labels(
        data, include_test=bool(cfg.task.get("evaluate_test", True))
    )

    run_results = []
    for run_id in range(int(cfg.num_runs)):
        seed = int(cfg.seed) + run_id
        set_seed(seed)
        run_results.append(
            _run_single_nc(cfg, data, device, logger, run_id, seed, eval_labels)
        )

    run_metrics_path = cfg.task.get("run_metrics_path")
    if run_metrics_path:
        path = Path(str(run_metrics_path)).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol_version": str(cfg.task.get("protocol_version", "")),
                    "base_seed": int(cfg.seed),
                    "run_seeds": [int(cfg.seed) + run_id for run_id in range(int(cfg.num_runs))],
                    "aggregation": "mean ± population std (ddof=0)",
                    "runs": [
                        {
                            "run_id": run_id,
                            "seed": int(cfg.seed) + run_id,
                            "metrics": {
                                key: float(value)
                                for key, value in item.items()
                                if not key.startswith("_")
                            },
                            "metadata": item.get("_run_metadata", {}),
                        }
                        for run_id, item in enumerate(run_results)
                    ],
                },
                handle,
                indent=2,
            )
    output: dict[str, tuple[float, float]] = {}
    keys = ["val_acc", "val_macro_f1"]
    if bool(cfg.task.get("evaluate_test", True)):
        keys.extend(["test_acc", "test_macro_f1"])
    names = {
        "val_acc": "Val Accuracy",
        "val_macro_f1": "Val Macro-F1",
        "test_acc": "Test Accuracy",
        "test_macro_f1": "Test Macro-F1",
    }
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    for key in keys:
        mean, std = mean_std([item[key] for item in run_results])
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", names[key], format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output
