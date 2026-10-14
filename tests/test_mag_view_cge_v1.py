from __future__ import annotations

import io

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mag_view_cge_v1 import Model, build_validated_view_operators
from src.models.rmcge_v22 import Model as LegacyModel


def _cfg(model_name: str, condition: str):
    base = OmegaConf.load("configs/model/mag_view_cge_v1.yaml")
    base.name = model_name
    if model_name == "rmcge_v22":
        base = OmegaConf.load("configs/model/rmcge_v22.yaml")
        base.condition = "M0"
    else:
        base.condition = condition
    return OmegaConf.create({"model": base, "seed": 42})


def _operators(model):
    unique = torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.uint8)
    ops = build_validated_view_operators(unique, flags, 5)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index_v2", ops[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight_v2", ops[modality]["weight"])
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_v7b_fingerprint = "b" * 64
    return ops


def _data_info():
    return {"input_dim": 8, "text_dim": 4, "visual_dim": 4, "num_nodes": 5, "num_classes": 3}


def test_source_operator_provenance_direction_and_row_normalization():
    unique = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.uint8)
    ops = build_validated_view_operators(unique, flags, 5)
    for name in ("text", "visual"):
        edge_index, weight = ops[name]["edge_index"], ops[name]["weight"]
        row_sum = torch.zeros(5).index_add_(0, edge_index[1], weight)
        assert torch.allclose(row_sum[row_sum > 0], torch.ones_like(row_sum[row_sum > 0]))
        assert torch.unique(edge_index[0] * 5 + edge_index[1]).numel() == edge_index.size(1)
    assert ops["text"]["edge_index"].size(1) == 4  # (0,1) and (1,2), symmetrized once
    assert ops["visual"]["edge_index"].size(1) == 4  # (0,2) and (1,2), symmetrized once
    assert torch.equal(row_sum[4:], torch.zeros(1))
    counted = build_validated_view_operators(
        torch.tensor([[0], [1]]), torch.tensor([[2, 0]], dtype=torch.uint8), 3
    )
    assert counted["text"]["edge_index"].size(1) == 2
    with pytest.raises(ValueError, match="duplicate canonical"):
        build_validated_view_operators(
            torch.tensor([[0, 0], [1, 1]]), torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8), 3
        )
    with pytest.raises(ValueError, match="out-of-range"):
        build_validated_view_operators(
            torch.tensor([[0], [4]]), torch.tensor([[1, 0]], dtype=torch.uint8), 3
        )


def test_cross_source_target_uses_opposite_modality_support():
    model = Model(_cfg("mag_view_cge_v1", "A2_cross_source"), _data_info())
    ops = _operators(model)
    values = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    text_result, text_edges, text_source = model._cross_source_context("text", values)
    visual_result, visual_edges, visual_source = model._cross_source_context("visual", values)
    assert text_source == "visual" and visual_source == "text"
    assert torch.equal(text_edges, ops["visual"]["edge_index"])
    assert torch.equal(visual_edges, ops["text"]["edge_index"])
    assert torch.equal(text_result, torch.zeros_like(text_result).index_add_(0, text_edges[1], values[text_edges[0]] * ops["visual"]["weight"][:, None]))
    assert torch.equal(visual_result, torch.zeros_like(visual_result).index_add_(0, visual_edges[1], values[visual_edges[0]] * ops["text"]["weight"][:, None]))


def test_a0_exactly_delegates_and_initialization_is_matched():
    torch.manual_seed(42)
    legacy = LegacyModel(_cfg("rmcge_v22", "A0_legacy"), _data_info()).eval()
    _operators(legacy)
    legacy_head = torch.nn.Linear(legacy.out_dim, 3)

    torch.manual_seed(42)
    candidate = Model(_cfg("mag_view_cge_v1", "A0_legacy"), _data_info()).eval()
    _operators(candidate)
    candidate_head = torch.nn.Linear(candidate.out_dim, 3)
    for name, value in legacy.named_parameters():
        assert torch.equal(value, dict(candidate.named_parameters())[name]), name
    assert torch.equal(legacy_head.weight, candidate_head.weight)
    assert torch.equal(legacy_head.bias, candidate_head.bias)

    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    with torch.no_grad():
        old = legacy(x, edge_index)[0]
        new = candidate(x, edge_index)[0]
    assert torch.equal(old, new)
    assert float((old - new).abs().max()) == 0.0


def test_shared_expert_objects_and_semantic_gradients():
    model = Model(_cfg("mag_view_cge_v1", "A1_separate_intra"), _data_info()).train()
    _operators(model)
    ids = [id(module) for module in model.experts]
    assert len(ids) == len(set(ids)) == 4
    assert sum(1 for name, module in model.named_modules() if name in {"experts.0", "experts.1", "experts.2", "experts.3"}) == 4

    prior = torch.randn(5, 256)
    context = torch.randn(5, 256)
    scale = torch.ones(256)
    edges = model.semantic_text_edge_index_v2
    contribution, _ = model._semantic_view(
        target_modality="text", modality_index=0, view_index=0, prior=prior,
        raw_context=context, scale=scale, edge_index=edges, enabled=True,
        collect_details=False,
    )
    weight = torch.randn_like(contribution)
    (contribution * weight).sum().backward()
    assert model.gamma_s_raw.grad is not None and torch.isfinite(model.gamma_s_raw.grad).all()
    assert model.gamma_s_raw.grad[0].abs() > 0
    assert model.view_router[-1].weight.grad is not None
    assert torch.isfinite(model.view_router[-1].weight.grad).all()
    assert model.view_router[-1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.experts.parameters())


def test_empty_views_are_zero_and_semantic_isolate_remains_active():
    model = Model(_cfg("mag_view_cge_v1", "A2_cross_source"), _data_info()).train()
    _operators(model)
    prior = torch.randn(5, 256)
    context = torch.randn(5, 256)
    scale = torch.ones(256)
    empty = torch.empty((2, 0), dtype=torch.long)
    result, _ = model._semantic_view(
        target_modality="text", modality_index=0, view_index=0, prior=prior,
        raw_context=torch.zeros_like(context), scale=scale, edge_index=empty,
        enabled=True, collect_details=False,
    )
    assert torch.equal(result, torch.zeros_like(result))

    # Node 4 has no physical edge in this graph but receives the semantic edge 3->4.
    physical_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    semantic_edges = torch.tensor([[3], [4]], dtype=torch.long)
    result, diag = model._semantic_view(
        target_modality="text", modality_index=0, view_index=0, prior=prior,
        raw_context=context, scale=scale, edge_index=semantic_edges,
        enabled=True, collect_details=True,
    )
    assert diag["active_nodes"] == 1
    assert result[4].abs().sum() > 0
    assert torch.equal(result[:4], torch.zeros_like(result[:4]))
    assert not bool((physical_edges[1] == 4).any())

    # Exercise the complete NC forward: node 4 is physically isolated but gets
    # a Text-KNN incoming edge and therefore a nonzero semantic-view update.
    full = Model(_cfg("mag_view_cge_v1", "A1_separate_intra"), _data_info()).eval()
    unique = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=torch.uint8)
    ops = build_validated_view_operators(unique, flags, 5)
    for modality in ("text", "visual"):
        setattr(full, f"semantic_{modality}_edge_index_v2", ops[modality]["edge_index"])
        setattr(full, f"semantic_{modality}_weight_v2", ops[modality]["weight"])
    full.semantic_candidate_fingerprint = "c" * 64
    full.semantic_v7b_fingerprint = "d" * 64
    x = torch.randn(5, 8)
    with torch.no_grad():
        semantic_on = full(x, physical_edges)[0]
        full.set_view_interventions(same_off=True)
        semantic_off = full(x, physical_edges)[0]
    assert (semantic_on[4] - semantic_off[4]).abs().sum() > 0


def test_a1_a2_common_initialization_and_cross_projection_gradients():
    hashes = []
    for condition in ("A1_separate_intra", "A2_cross_source"):
        torch.manual_seed(123)
        model = Model(_cfg("mag_view_cge_v1", condition), _data_info()).train()
        _operators(model)
        head = torch.nn.Linear(256, 3)
        hashes.append((
            {name: value.detach().clone() for name, value in model.named_parameters()},
            head.weight.detach().clone(), head.bias.detach().clone(),
        ))
    assert hashes[0][0].keys() == hashes[1][0].keys()
    assert all(torch.equal(hashes[0][0][name], hashes[1][0][name]) for name in hashes[0][0])
    assert torch.equal(hashes[0][1], hashes[1][1]) and torch.equal(hashes[0][2], hashes[1][2])

    model = Model(_cfg("mag_view_cge_v1", "A2_cross_source"), _data_info()).train()
    _operators(model)
    sender = torch.randn(5, 256)
    transformed = model.cross_projections["visual_to_text"](sender)
    context, edges, source = model._cross_source_context("text", transformed)
    contribution, _ = model._semantic_view(
        target_modality="text", modality_index=0, view_index=1,
        prior=torch.randn_like(sender), raw_context=context, scale=torch.ones(256),
        edge_index=edges, enabled=True, collect_details=False,
    )
    (contribution * torch.randn_like(contribution)).sum().backward()
    assert source == "visual"
    grad = model.cross_projections["visual_to_text"][0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert model.gamma_c_raw.grad is not None and model.gamma_c_raw.grad[0].abs() > 0


def test_checkpoint_state_reload_reproduces_view_output():
    torch.manual_seed(9)
    model = Model(_cfg("mag_view_cge_v1", "A2_cross_source"), _data_info()).eval()
    _operators(model)
    x = torch.randn(5, 8)
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    with torch.no_grad():
        expected = model(x, edge)[0]
    assert expected.shape == (5, 256)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    clone = Model(_cfg("mag_view_cge_v1", "A2_cross_source"), _data_info()).eval()
    _operators(clone)
    clone.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        actual = clone(x, edge)[0]
    assert torch.equal(expected, actual)


def test_a2_calls_one_shared_expert_bank_for_each_view():
    model = Model(_cfg("mag_view_cge_v1", "A2_cross_source"), _data_info()).eval()
    _operators(model)
    calls = [0, 0, 0, 0]
    handles = [expert.register_forward_hook(lambda _m, _i, _o, idx=idx: calls.__setitem__(idx, calls[idx] + 1)) for idx, expert in enumerate(model.experts)]
    x = torch.randn(5, 8)
    edge = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    with torch.no_grad():
        output = model(x, edge)[0]
    for handle in handles:
        handle.remove()
    assert output.shape == (5, 256)
    assert calls == [6, 6, 6, 6]  # physical + S + C for each target modality
