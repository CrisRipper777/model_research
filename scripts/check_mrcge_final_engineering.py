#!/usr/bin/env python3
"""Lightweight synthetic checks for the frozen MR-CGE Final implementation."""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mrcge_final_benchmark_utils import (  # noqa: E402
    build_config, load_prepared_manifest,
)
from src.models.mrcge_final import Model as FinalModel  # noqa: E402
from src.models.mrcge_final_sampled import Model as SampledModel  # noqa: E402
from src.models.mrcge_v13 import Model as ReferenceModel  # noqa: E402
from src.models.predictor import LinkPredictor  # noqa: E402
from src.models.rmcge_v2 import build_modality_knn_operators  # noqa: E402
from src.data import EdgeSplit, MAGData  # noqa: E402
from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b  # noqa: E402
from src.tasks.lp import _build_message_edge_lookup, _exclude_positive_label_edges_by_global_eid  # noqa: E402
from src.tasks.lp import run_lp  # noqa: E402


def _cfg(num_classes: int | None = 3):
    model = OmegaConf.load(ROOT / "configs/model/mrcge_final.yaml")
    model.name = "mrcge_final"
    return OmegaConf.create({"model": model}), {
        "input_dim": 8, "num_nodes": 8, "num_classes": num_classes,
        "text_dim": 4, "visual_dim": 4,
    }


def _attach_synthetic_semantics(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = torch.tensor([[0, 2, 1, 2, 3, 4, 5, 6, 2, 4, 0, 4],
                          [1, 3, 4, 5, 6, 7, 7, 7, 4, 5, 6, 7]], dtype=torch.long)
    # Canonical undirected candidate edge IDs in the same sorted order expected by V7B.
    canonical = torch.stack((torch.minimum(pairs[0], pairs[1]), torch.maximum(pairs[0], pairs[1])))
    codes = canonical[0] * model.num_nodes + canonical[1]
    unique_codes = torch.unique(codes, sorted=True)
    unique_edges = torch.stack((unique_codes // model.num_nodes, unique_codes % model.num_nodes))
    flags = torch.zeros((unique_codes.numel(), 2), dtype=torch.uint8)
    for index, (source, target) in enumerate(unique_edges.t().tolist()):
        flags[index, 0] = 1 if index % 2 == 0 or index % 4 == 0 else 0
        flags[index, 1] = 1 if index % 2 == 1 or index % 4 == 0 else 0
    operators = build_modality_knn_operators(unique_edges, flags, model.num_nodes)
    model.semantic_source_flags_v7b = flags
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index", operators[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight", operators[modality]["weight"])
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_feature_fingerprint = "b" * 64
    model.semantic_v7b_fingerprint = "c" * 64
    return unique_edges, flags, operators["text"]["edge_index"]


def check_full_model_parity_and_gradient() -> None:
    cfg, info = _cfg()
    torch.manual_seed(314159)
    reference = ReferenceModel(cfg, info)
    torch.manual_seed(314159)
    final = FinalModel(cfg, info)
    reference_state = reference.state_dict()
    final_state = final.state_dict()
    if reference_state.keys() != final_state.keys():
        raise AssertionError("Full Final state_dict key set differs from V1.3-A reference")
    for key in reference_state:
        if not torch.equal(reference_state[key], final_state[key]):
            raise AssertionError(f"Full Final initialization differs from V1.3-A reference at {key}")

    _attach_synthetic_semantics(reference)
    _attach_synthetic_semantics(final)
    x = torch.arange(64, dtype=torch.float32).view(8, 8) / 31.0
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7],
    ], dtype=torch.long)
    reference.eval()
    final.eval()
    with torch.no_grad():
        ref_out = reference(x, edge_index)
        final_out = final(x, edge_index)
    for label, left, right in (
        ("embedding", ref_out[0], final_out[0]),
        ("auxiliary loss", ref_out[3], final_out[3]),
        ("Text logits", ref_out[4]["unimodal_logits"]["text"], final_out[4]["unimodal_logits"]["text"]),
        ("Visual logits", ref_out[4]["unimodal_logits"]["visual"], final_out[4]["unimodal_logits"]["visual"]),
    ):
        if not torch.equal(left, right):
            raise AssertionError(f"Full Final differs from V1.3-A reference in {label}")

    # Check the normal Full model has finite gradients and its checkpoint reloads.
    final.train()
    final.zero_grad(set_to_none=True)
    output = final(x, edge_index)
    loss = output[0].square().mean() + output[3]
    loss.backward()
    grads = [parameter.grad for parameter in final.parameters() if parameter.grad is not None]
    if not grads or not all(torch.isfinite(grad).all() for grad in grads):
        raise AssertionError("Full Final synthetic backward pass has absent/non-finite gradients")
    buffer = io.BytesIO()
    torch.save(final.state_dict(), buffer)
    buffer.seek(0)
    restored = FinalModel(cfg, info)
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    for key, value in final.state_dict().items():
        if not torch.equal(value, restored.state_dict()[key]):
            raise AssertionError(f"Full Final checkpoint roundtrip differs at {key}")


def check_sampled_adapter_mapping_mask_and_update() -> None:
    cfg, info = _cfg(num_classes=None)
    cfg.model.name = "mrcge_final_sampled"
    cfg.model.semantic_context_physical_fanout = [1, 1, 1]
    torch.manual_seed(271828)
    model = SampledModel(cfg, info)
    if not model.supports_link_prediction or model.requires_global_semantic_candidates:
        raise AssertionError("sampled adapter capability flags do not enable the supported LP path")
    unique_edges, flags, _ = _attach_synthetic_semantics(model)
    model.bind_lp_graph(info_x := torch.randn(8, 8), torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7],
    ], dtype=torch.long))
    train_edges = model._lp_global_edge_index
    node_ids = torch.arange(8, dtype=torch.long)
    batch = Data(
        x=info_x.clone(),
        edge_index=train_edges.clone(),
        e_id=torch.arange(train_edges.size(1), dtype=torch.long),
        n_id=node_ids.clone(),
        edge_label_index=torch.tensor([[2, 2], [3, 6]], dtype=torch.long),
        edge_label=torch.tensor([1.0, 0.0]),
        num_nodes=8,
    )
    positive_eids = _build_message_edge_lookup(train_edges, 8)
    _exclude_positive_label_edges_by_global_eid(batch, positive_eids)
    model.prepare_sampled_lp_batch(batch, sample_seed=1234)
    if not torch.equal(batch.n_id[:8], node_ids):
        raise AssertionError("adapter changed the original LinkNeighborLoader node order")
    if not torch.equal(batch.edge_label_index, torch.tensor([[2, 2], [3, 6]])):
        raise AssertionError("adapter changed edge_label_index endpoint-local IDs")
    if batch.n_id.numel() != batch.x.size(0):
        raise AssertionError("appended local features do not align with global node IDs")
    if model._sampled_semantic_edge_flags is None:
        raise AssertionError("adapter dropped local V7B source flags")
    global_to_flag = {
        (int(source), int(target)): tuple(int(v) for v in flag)
        for (source, target), flag in zip(unique_edges.t().tolist(), flags.tolist())
    }
    for modality in ("text", "visual"):
        original_edges = getattr(model, f"semantic_{modality}_edge_index")
        original_weights = getattr(model, f"semantic_{modality}_weight")
        directed_weight_map = {
            (int(source), int(target)): float(weight)
            for (source, target), weight in zip(original_edges.t().tolist(), original_weights.tolist())
        }
        local_edges, local_weights = model._sampled_semantic_edges[modality]
        local_flags = model._sampled_semantic_edge_flags[modality]
        if local_weights.numel() != local_edges.size(1) or local_flags.shape != (local_edges.size(1), 2):
            raise AssertionError("local semantic weights/source flags lost edge alignment")
        for edge_id, (local_source, local_target) in enumerate(local_edges.t().tolist()):
            global_edge = (int(batch.n_id[local_source]), int(batch.n_id[local_target]))
            canonical = tuple(sorted(global_edge))
            if global_edge not in directed_weight_map or not torch.isclose(
                local_weights[edge_id], torch.tensor(directed_weight_map[global_edge]), atol=0.0, rtol=0.0
            ):
                raise AssertionError("adapter changed a cached directed semantic edge or its weight")
            if tuple(int(v) for v in local_flags[edge_id].tolist()) != global_to_flag[canonical]:
                raise AssertionError("global-to-local semantic mapping changed V7B source flags")

    # The supervised positive and its reverse may not re-enter through context expansion.
    for source, target in batch.edge_index.t().tolist():
        global_source, global_target = int(batch.n_id[source]), int(batch.n_id[target])
        if (global_source, global_target) in {(2, 3), (3, 2)}:
            raise AssertionError("global_eid positive-edge mask was bypassed by semantic context")

    predictor = LinkPredictor(128, 32, 2, 0.0)
    projection = torch.nn.Linear(model.out_dim, 128)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection.parameters()) + list(predictor.parameters()), lr=1e-3)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux_loss, _ = model(batch.x, batch.edge_index)
    z = projection(z)
    src, dst = batch.edge_label_index
    logits = predictor.score_pairs(z[src], z[dst])
    loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label) + aux_loss
    loss.backward()
    trainable_grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not trainable_grads or not all(torch.isfinite(grad).all() for grad in trainable_grads):
        raise AssertionError("sampled LP synthetic batch produced absent/non-finite model gradients")
    before = model.rho_raw.detach().clone()
    optimizer.step()
    if torch.equal(before, model.rho_raw.detach()):
        raise AssertionError("sampled LP synthetic optimizer step did not update the encoder")
    model.clear_sampled_lp_context()
    model.eval()
    with torch.no_grad():
        full_output = model(info_x, train_edges)
    if full_output[0].shape != (8, model.out_dim) or model._sampled_semantic_edges is not None:
        raise AssertionError("adapter did not return to the full-graph inference path")


def check_original_lp_trainer_one_synthetic_batch() -> None:
    """Exercise run_lp/LinkNeighborLoader on tiny generated data, with Test disabled."""
    node_count = 12
    x_text = torch.randn(node_count, 4)
    x_visual = torch.randn(node_count, 4)
    features = torch.cat((x_text, x_visual), dim=1)
    train_pos = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long)
    edge_index = torch.cat((train_pos, train_pos.flip(0)), dim=1)
    with tempfile.TemporaryDirectory(prefix="mrcge-final-synthetic-lp-") as tmp_name:
        temp = Path(tmp_name)
        cache_path, cache_meta = build_semantic_candidate_cache_v7b(
            dataset="SyntheticLP",
            text_features=x_text,
            visual_features=x_visual,
            physical_edge_index=edge_index,
            cache_dir=temp / "cache",
            top_k=8,
            seed=7,
            thread_count=1,
        )
        model_cfg = OmegaConf.load(ROOT / "configs/model/mrcge_final.yaml")
        model_cfg.name = "mrcge_final_sampled"
        model_cfg.semantic_cache_path = str(cache_path)
        model_cfg.semantic_feature_fingerprint = cache_meta["feature_fingerprint"]
        model_cfg.semantic_candidate_fingerprint = cache_meta["candidate_fingerprint"]
        model_cfg.semantic_v7b_fingerprint = cache_meta["v7b_fingerprint"]
        model_cfg.semantic_context_physical_fanout = [1, 1, 1]
        task_cfg = OmegaConf.load(ROOT / "configs/task/mrcge_final_lp_benchmark.yaml")
        task_cfg.epochs = 1
        task_cfg.patience = 2
        task_cfg.early_stop_min_epoch = 1
        task_cfg.batch_size = 2
        task_cfg.num_neighbors = [2, 2, 2]
        task_cfg.max_train_batches = 1
        task_cfg.eval_every = 1
        task_cfg.inference_batch_size = 16
        task_cfg.eval_edge_batch_size = 2
        task_cfg.loader_num_workers = 0
        task_cfg.evaluate_test = False
        task_cfg.save_ckpt_path = str(temp / "synthetic_best_validation.pt")
        cfg = OmegaConf.create({
            "seed": 42, "num_runs": 1,
            "dataset": {"name": "SyntheticLP", "make_undirected": True},
            "model": model_cfg, "task": task_cfg,
        })
        split = EdgeSplit(
            train={"source_node": train_pos[0].clone(), "target_node": train_pos[1].clone()},
            valid={
                "source_node": torch.tensor([7, 8]),
                "target_node": torch.tensor([8, 9]),
                "target_node_neg": torch.tensor([[10, 11], [0, 1]], dtype=torch.long),
            },
            test={
                "source_node": torch.tensor([9, 10]),
                "target_node": torch.tensor([10, 11]),
                "target_node_neg": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            },
            metadata={"undirected": True},
        )
        data = MAGData(
            name="SyntheticLP", source="mmgraph", task="lp", x=features,
            x_t=x_text, x_i=x_visual, edge_index=edge_index, num_nodes=node_count,
            num_classes=None, edge_split=split,
        )
        logger = logging.getLogger("mrcge_final_synthetic_lp")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        result = run_lp(cfg, data, torch.device("cpu"), logger)
        if "val_mrr" not in result or "test_mrr" in result:
            raise AssertionError("synthetic LP Trainer did not stay Validation-only")
        checkpoint = torch.load(task_cfg.save_ckpt_path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection") != "best_val_mrr" or checkpoint["epoch"] != 1:
            raise AssertionError("synthetic LP Trainer did not save a Best-Val-MRR checkpoint")

        fresh_cfg = OmegaConf.create({"model": model_cfg})
        restored = SampledModel(fresh_cfg, {
            "input_dim": 8, "num_nodes": node_count, "num_classes": None,
            "text_dim": 4, "visual_dim": 4,
        })
        restored.load_semantic_cache(
            str(cache_path), expected_dataset="SyntheticLP",
            expected_feature_fingerprint=cache_meta["feature_fingerprint"],
            expected_candidate_fingerprint=cache_meta["candidate_fingerprint"],
            expected_v7b_fingerprint=cache_meta["v7b_fingerprint"],
        )
        restored.load_state_dict(checkpoint["model_state"])
        if not torch.isfinite(torch.tensor(result["val_mrr"][0])):
            raise AssertionError("synthetic LP Trainer returned a non-finite Validation result")


def check_fixed_nc_split_paths() -> None:
    manifest = load_prepared_manifest()
    for dataset in ("Movies", "Grocery", "Toys", "Reddit-S"):
        one = build_config("nc", dataset, 42, "cpu", manifest["datasets"][dataset])
        two = build_config("nc", dataset, 44, "cpu", manifest["datasets"][dataset])
        if one.dataset.nc_split_path != two.dataset.nc_split_path:
            raise AssertionError(f"{dataset}: NC split path follows the training seed")
        if bool(one.dataset.auto_generate_splits) or bool(two.dataset.auto_generate_splits):
            raise AssertionError(f"{dataset}: automatic NC split generation remains enabled")
    official = build_config("nc", "ele-fashion", 44, "cpu", manifest["datasets"]["ele-fashion"])
    if official.dataset.node_split_path != manifest["datasets"]["ele-fashion"]["split"]["path"]:
        raise AssertionError("ele-fashion no longer points to its official split.pt")


def main() -> None:
    check_full_model_parity_and_gradient()
    print("PASS: frozen Full state/init/output parity, finite backward, checkpoint reload")
    check_sampled_adapter_mapping_mask_and_update()
    print("PASS: LP global/local mappings, V7B flags, positive-edge mask, synthetic update, full-graph inference path")
    check_original_lp_trainer_one_synthetic_batch()
    print("PASS: original run_lp + LinkNeighborLoader one-batch update, Validation checkpoint, Test disabled")
    check_fixed_nc_split_paths()
    print("PASS: seed42 fixed MAGB NC split paths and official ele-fashion split")


if __name__ == "__main__":
    main()
