from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b
from src.data.types import MAGData
from src.models import build_model
from src.models.pace_mag_v1 import (
    build_physical_operator,
    build_semantic_operators,
    row_quality_aggregate,
    validate_v7b_cache_graph,
)


@pytest.fixture(scope="module")
def toy_graph(tmp_path_factory):
    torch.manual_seed(701)
    rng = np.random.default_rng(701)
    text = rng.normal(size=(14, 7)).astype(np.float32)
    visual = rng.normal(size=(14, 5)).astype(np.float32)
    physical = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 7, 6, 8, 7]],
        dtype=torch.long,
    )
    cache_path, metadata = build_semantic_candidate_cache_v7b(
        dataset="Toy",
        text_features=text,
        visual_features=visual,
        physical_edge_index=physical,
        cache_dir=tmp_path_factory.mktemp("pace_cache"),
    )
    return {
        "text": torch.from_numpy(text),
        "visual": torch.from_numpy(visual),
        "x": torch.from_numpy(np.concatenate((text, visual), axis=1)),
        "physical": physical,
        "cache_path": str(cache_path),
        "metadata": metadata,
    }


def make_cfg(variant: str = "e2_full", *, epochs: int = 1, expert_chunk_size: int = 32768, dropout: float = 0.05):
    return OmegaConf.create(
        {
            "dataset": {"name": "Toy"},
            "model": {
                "name": "pace_mag_v1",
                "variant": variant,
                "hidden_dim": 16,
                "dropout": dropout,
                "num_experts": 3,
                "expert_bottleneck": 8,
                "expert_chunk_size": expert_chunk_size,
                "activation_checkpointing": True,
                "router_hidden_dim": 9,
                "relation_embedding_dim": 4,
                "modality_embedding_dim": 3,
                "cross_modal_bottleneck": 6,
                "rho_init": 0.1,
                "rho_max": 0.25,
                "beta_phy_init": 0.5,
                "beta_sem_init": 0.3,
                "beta_max": 1.0,
                "gamma": 0.2,
                "semantic_edge_weight": 0.8,
                "eps": 1.0e-8,
            },
            "task": {
                "name": "nc",
                "protocol_version": "unified_full_graph_nc_v1",
                "optimizer": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": epochs,
                "patience": 30,
                "early_stop_min_epoch": 30,
                "early_stop_min_delta": 0.0,
                "eval_every": 1,
                "inference_mode": "full",
                "inference_batch_size": 128,
                "grad_clip": 1.0,
                "loss": {"aux_weight": 1.0},
                "evaluate_test": False,
                "save_ckpt_path": None,
                "run_metrics_path": None,
                "training_mode": "full_graph",
            },
            "seed": 42,
            "num_runs": 1,
        }
    )


def make_model(variant: str, toy_graph, *, device="cpu"):
    info = {
        "input_dim": 12,
        "num_nodes": toy_graph["x"].size(0),
        "num_classes": 3,
        "text_dim": toy_graph["text"].size(1),
        "visual_dim": toy_graph["visual"].size(1),
    }
    model = build_model(make_cfg(variant), info)
    model.load_semantic_cache(
        toy_graph["cache_path"],
        expected_dataset="Toy",
        expected_feature_fingerprint=toy_graph["metadata"]["feature_fingerprint"],
        expected_candidate_fingerprint=toy_graph["metadata"]["candidate_fingerprint"],
        expected_v7b_fingerprint=toy_graph["metadata"]["v7b_fingerprint"],
    )
    return model.to(device)


def test_sparse_pt_pv_reconstruct_pu_and_respect_raw_proposal_counts():
    unique = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 2], [1, 2]], dtype=torch.uint8)
    views = build_semantic_operators(unique, flags, 5)
    assert torch.allclose(views["pt"].to_dense() + views["pv"].to_dense(), views["pu"].to_dense(), atol=1e-7)
    assert views["text_share"].tolist() == pytest.approx([1.0, 0.0, 0.5])
    assert views["visual_share"].tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert torch.equal(views["pt"].to_dense(), views["pt"].to_dense().T)
    assert torch.equal(views["pv"].to_dense(), views["pv"].to_dense().T)
    assert torch.allclose(views["q_text"] + views["q_visual"], views["q_union"], atol=1e-7)


@pytest.mark.parametrize(
    "flags",
    [torch.tensor([[0, 0]], dtype=torch.uint8), torch.tensor([[3, 0]], dtype=torch.uint8)],
)
def test_invalid_source_counts_are_rejected(flags):
    with pytest.raises(ValueError):
        build_semantic_operators(torch.tensor([[0], [1]]), flags, 2)


def test_v7b_bidirected_order_flag_alignment_and_duplicates_are_checked():
    unique = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    directed = torch.cat((unique, unique.flip(0)), dim=1)
    flags = torch.tensor([[1, 0], [1, 2]], dtype=torch.uint8)
    validate_v7b_cache_graph(directed, unique, flags, 3)
    with pytest.raises(ValueError, match="directed edges"):
        validate_v7b_cache_graph(directed[:, [0, 2, 1, 3]], unique, flags, 3)
    duplicate_unique = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="canonical|sorted"):
        validate_v7b_cache_graph(torch.empty((2, 0), dtype=torch.long), duplicate_unique, flags, 2)


def test_physical_graph_drops_self_loops_and_duplicate_edges():
    edges = torch.tensor([[0, 1, 1, 0, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]])
    p0, q = build_physical_operator(edges, 4)
    dense = p0.to_dense()
    assert torch.allclose(dense, dense.T)
    assert dense[0, 0] == 0 and dense[2, 2] == 0
    assert torch.allclose(dense[0, 1], torch.tensor(1.0 / 2**0.5))
    assert q[3] == 0


def test_empty_rows_have_no_aggregation_even_when_downstream_can_have_bias():
    p0, q = build_physical_operator(torch.empty((2, 0), dtype=torch.long), 4)
    message = torch.randn(4, 6)
    aggregate = row_quality_aggregate(p0, message, q, 1e-8)
    assert torch.equal(aggregate, torch.zeros_like(aggregate))


def test_cross_modal_rho_zero_exactly_returns_target_message():
    target = torch.randn(7, 4)
    other = torch.randn(7, 4)
    transformed = torch.tanh(other)
    from src.models.pace_mag_v1 import Model

    message = Model.compose_cross_modal_message(target, transformed, target.new_zeros(()))
    assert torch.equal(message, target)


@pytest.mark.parametrize("variant", ["e0_source_agnostic", "e1_no_cross_modal", "e2_full", "e3_single_expert_control", "e4_static_routing"])
def test_all_variants_forward_finite_and_router_is_a_probability_distribution(variant, toy_graph):
    model = make_model(variant, toy_graph).eval()
    with torch.no_grad():
        z, _, _, aux, info = model(toy_graph["x"], toy_graph["physical"], return_details=True)
    assert z.shape == (toy_graph["x"].size(0), 16)
    assert torch.isfinite(z).all() and torch.isfinite(aux)
    assert aux.item() == 0.0
    assert info["details"]["final_embedding"]["finite"]
    if variant != "e3_single_expert_control":
        for relation in ("physical", "text", "visual"):
            for modality in ("text", "visual"):
                row = info["details"]["router"][relation][modality]
                assert sum(row["mean_probability"]) == pytest.approx(1.0, abs=1e-6)
                assert all(np.isfinite(value) and value >= 0 for value in row["mean_probability"])
                assert np.isfinite(row["mean_entropy"])
                assert row["valid_node_probability_sum_min"] == pytest.approx(1.0, abs=1e-6)
                assert row["valid_node_probability_sum_max"] == pytest.approx(1.0, abs=1e-6)
    assert info["details"]["effective_node_fraction"]["physical"] < 1.0


def test_expert_bank_is_shared_across_all_relation_modality_calls(toy_graph):
    model = make_model("e2_full", toy_graph)
    assert len(model.experts) == 3
    assert len({id(expert) for expert in model.experts}) == 3
    calls = [0, 0, 0]
    hooks = [expert.register_forward_hook(lambda _m, _i, _o, k=k: calls.__setitem__(k, calls[k] + 1)) for k, expert in enumerate(model.experts)]
    model.eval()
    with torch.no_grad():
        model(toy_graph["x"], toy_graph["physical"])
    for hook in hooks:
        hook.remove()
    assert calls == [6, 6, 6]


def test_e0_replaces_real_source_split_with_two_half_union_views(toy_graph):
    model = make_model("e0_source_agnostic", toy_graph)
    assert torch.allclose(model.semantic_pt.to_dense(), 0.5 * model.semantic_pu.to_dense(), atol=1e-7)
    assert torch.allclose(model.semantic_pv.to_dense(), 0.5 * model.semantic_pu.to_dense(), atol=1e-7)
    assert torch.allclose(model.semantic_q_text, 0.5 * model.semantic_q_union, atol=1e-7)
    assert torch.allclose(model.semantic_q_visual, 0.5 * model.semantic_q_union, atol=1e-7)


def test_e3_single_expert_parameter_count_matches_dynamic_processing_budget(toy_graph):
    data_info = {"input_dim": 12, "num_nodes": 14, "num_classes": 3, "text_dim": 7, "visual_dim": 5}
    full = build_model(make_cfg("e2_full"), data_info)
    single = build_model(make_cfg("e3_single_expert_control"), data_info)
    full_dynamic = sum(p.numel() for p in full.experts.parameters()) + sum(p.numel() for p in full.router.parameters()) + sum(p.numel() for p in full.relation_embedding.parameters()) + sum(p.numel() for p in full.modality_embedding.parameters())
    single_dynamic = sum(p.numel() for p in single.single_expert.parameters())
    assert abs(full_dynamic - single_dynamic) <= max(100, int(0.01 * full_dynamic))
    assert len(single.experts) == 0 and single.router is None


def _group_norms(model):
    groups = {"projectors": 0.0, "cross_modal": 0.0, "experts": 0.0, "router": 0.0, "fusion": 0.0}
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        group = "fusion" if root in {"fusion", "fusion_norm", "skip"} else root
        if group in groups and parameter.grad is not None:
            groups[group] += float(parameter.grad.detach().float().norm())
    return groups


def test_full_has_finite_nonzero_gradients_and_adamw_updates_core_modules(toy_graph):
    model = make_model("e2_full", toy_graph).train()
    classifier = torch.nn.Linear(model.out_dim, 3)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=1e-4)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    labels = torch.arange(toy_graph["x"].size(0)) % 3
    z, _, _, aux, _ = model(toy_graph["x"], toy_graph["physical"])
    loss = torch.nn.functional.cross_entropy(classifier(z), labels) + aux
    loss.backward()
    norms = _group_norms(model)
    assert all(np.isfinite(value) and value > 0 for value in norms.values())
    model.capture_gradient_diagnostics()
    assert all(row["gradient_finite"] for row in model._gradient_diagnostics.values())
    optimizer.step()
    updated = {
        group: any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if name.startswith(group) or (group == "fusion" and name.startswith(("fusion_norm.", "skip.")))
        )
        for group in ("projectors", "cross_modal", "experts", "router", "fusion")
    }
    assert all(updated.values())


def test_e1_cross_modal_parameters_have_no_task_gradient_or_adamw_update(toy_graph):
    model = make_model("e1_no_cross_modal", toy_graph).train()
    classifier = torch.nn.Linear(model.out_dim, 3)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=0.5)
    disabled = [(name, parameter, parameter.detach().clone()) for name, parameter in model.named_parameters() if name.startswith(("cross_modal.", "rho_raw"))]
    z, _, _, _, _ = model(toy_graph["x"], toy_graph["physical"])
    torch.nn.functional.cross_entropy(classifier(z), torch.arange(z.size(0)) % 3).backward()
    assert all(parameter.grad is None for _, parameter, _ in disabled)
    optimizer.step()
    assert all(torch.equal(parameter, before) for _, parameter, before in disabled)


def test_checkpoint_and_cache_reload_preserve_inference(toy_graph, tmp_path):
    first = make_model("e2_full", toy_graph).eval()
    with torch.no_grad():
        expected = first(toy_graph["x"], toy_graph["physical"])[0]
    path = tmp_path / "pace.pt"
    torch.save(first.state_dict(), path)
    second = make_model("e2_full", toy_graph).eval()
    second.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        observed = second(toy_graph["x"], toy_graph["physical"])[0]
    assert torch.allclose(expected, observed, atol=1e-7, rtol=1e-7)


def test_chunked_expert_activation_checkpointing_matches_full_batch(toy_graph):
    info = {"input_dim": 12, "num_nodes": 14, "num_classes": 3, "text_dim": 7, "visual_dim": 5}
    full = build_model(make_cfg("e2_full", expert_chunk_size=64, dropout=0.0), info)
    full.load_semantic_cache(toy_graph["cache_path"], expected_dataset="Toy")
    chunked = build_model(make_cfg("e2_full", expert_chunk_size=4, dropout=0.0), info)
    chunked.load_semantic_cache(toy_graph["cache_path"], expected_dataset="Toy")
    chunked.load_state_dict(full.state_dict())
    full.train()
    chunked.train()
    z_full = full(toy_graph["x"], toy_graph["physical"])[0]
    z_chunked = chunked(toy_graph["x"], toy_graph["physical"])[0]
    assert torch.allclose(z_full, z_chunked, atol=1e-6, rtol=1e-6)
    z_full.square().mean().backward()
    z_chunked.square().mean().backward()
    for (name_a, param_a), (name_b, param_b) in zip(full.named_parameters(), chunked.named_parameters()):
        assert name_a == name_b
        assert param_a.grad is None and param_b.grad is None or torch.allclose(param_a.grad, param_b.grad, atol=2e-5, rtol=2e-5)


def test_nc_evaluate_test_false_uses_validation_only_and_saves_no_test_metrics(toy_graph, tmp_path, monkeypatch):
    import src.tasks.nc as nc_task

    cfg = make_cfg("e2_full", epochs=1)
    cache_meta = toy_graph["metadata"]
    cfg.model.semantic_cache_path = toy_graph["cache_path"]
    cfg.model.semantic_feature_fingerprint = cache_meta["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache_meta["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache_meta["v7b_fingerprint"]
    cfg.task.save_ckpt_path = str(tmp_path / "selected.pt")
    cfg.task.run_metrics_path = str(tmp_path / "run_metrics.json")
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 99, 99, 99, 99])
    data = MAGData(
        name="Toy", source="test", task="nc", x=toy_graph["x"],
        x_t=toy_graph["text"], x_i=toy_graph["visual"],
        edge_index=toy_graph["physical"], num_nodes=14, num_classes=3, y=y,
        train_idx=torch.arange(0, 8), val_idx=torch.tensor([8, 9]), test_idx=torch.arange(10, 14),
    )
    seen = []
    original = nc_task._evaluate_split

    def recording_evaluate(classifier, z, labels, idx, device, batch_size, eval_labels):
        seen.append((idx.detach().cpu().tolist(), list(eval_labels)))
        return original(classifier, z, labels, idx, device, batch_size, eval_labels)

    monkeypatch.setattr(nc_task, "_evaluate_split", recording_evaluate)
    logger = logging.getLogger("pace_mag_v1.test_no_test_metrics")
    result = nc_task.run_nc(cfg, data, torch.device("cpu"), logger)
    payload = torch.load(tmp_path / "selected.pt", map_location="cpu", weights_only=False)
    assert set(payload["metrics"]) == {"val_acc", "val_macro_f1"}
    assert "test_acc" not in payload["metrics"] and "test_macro_f1" not in payload["metrics"]
    assert "test_acc" not in result and "test_macro_f1" not in result
    assert len(seen) == 1 and seen[0][0] == [8, 9]
    assert 99 not in seen[0][1]
    assert cfg.task.evaluate_test is False


def test_feature_fingerprint_and_cache_node_count_mismatches_are_rejected(toy_graph):
    info = {"input_dim": 12, "num_nodes": 14, "num_classes": 3, "text_dim": 7, "visual_dim": 5}
    wrong_nodes = build_model(make_cfg(), {**info, "num_nodes": 15})
    with pytest.raises(ValueError, match="num_nodes"):
        wrong_nodes.load_semantic_cache(toy_graph["cache_path"], expected_dataset="Toy")
    model = build_model(make_cfg(), info)
    with pytest.raises(ValueError, match="feature fingerprint"):
        model.load_semantic_cache(toy_graph["cache_path"], expected_dataset="Toy", expected_feature_fingerprint="bad")
    model = build_model(make_cfg(), info)
    with pytest.raises(ValueError, match="candidate graph fingerprint"):
        model.load_semantic_cache(toy_graph["cache_path"], expected_dataset="Toy", expected_candidate_fingerprint="bad")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_graph_cuda_forward_backward(toy_graph):
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    model = make_model("e2_full", toy_graph, device=device).train()
    classifier = torch.nn.Linear(model.out_dim, 3, device=device)
    x = toy_graph["x"].to(device)
    edge_index = toy_graph["physical"].to(device)
    z, _, _, aux, _ = model(x, edge_index)
    loss = torch.nn.functional.cross_entropy(classifier(z), torch.arange(z.size(0), device=device) % 3) + aux
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
