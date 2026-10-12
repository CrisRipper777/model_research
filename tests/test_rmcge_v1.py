from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.data import MAGData
from src.models.rmcge_v1 import (
    Model,
    RelationExpertLayer,
    build_physical_operator,
    build_semantic_operators,
    joint_masked_softmax,
)
from src.tasks.nc import _resolve_nc_eval_labels


def _layer(*, chunk_size: int = 8) -> RelationExpertLayer:
    return RelationExpertLayer(
        128,
        num_experts=3,
        expert_bottleneck=64,
        router_hidden_dim=64,
        relation_embedding_dim=8,
        modality_embedding_dim=8,
        cross_modal_bottleneck=64,
        dropout=0.2,
        router_correction_init_std=0.01,
        physical_prior=0.6,
        semantic_prior=0.4,
        expert_chunk_size=chunk_size,
    )


def _cfg(*, num_nodes: int, text_dim: int = 4, visual_dim: int = 3):
    return OmegaConf.create(
        {
            "model": {
                "hidden_dim": 128,
                "dropout": 0.2,
                "num_layers": 2,
                "num_experts": 3,
                "expert_bottleneck": 64,
                "expert_chunk_size": 4,
                "activation_checkpointing": True,
                "router_hidden_dim": 64,
                "relation_embedding_dim": 8,
                "modality_embedding_dim": 8,
                "cross_modal_bottleneck": 64,
                "router_correction_init_std": 0.01,
                "physical_prior": 0.6,
                "semantic_prior": 0.4,
                "residual_scale": 0.5,
            }
        }
    )


def _model(num_nodes: int = 8, text_dim: int = 4, visual_dim: int = 3) -> Model:
    return Model(
        _cfg(num_nodes=num_nodes, text_dim=text_dim, visual_dim=visual_dim),
        {
            "num_nodes": num_nodes,
            "text_dim": text_dim,
            "visual_dim": visual_dim,
            "input_dim": text_dim + visual_dim,
            "num_classes": 3,
        },
    )


def _attach_semantic_views(
    model: Model,
    unique_edges: torch.Tensor | None = None,
    flags: torch.Tensor | None = None,
) -> None:
    if unique_edges is None:
        unique_edges = torch.tensor(
            [[0, 0, 1, 2, 3, 4, 5], [1, 2, 2, 3, 4, 5, 6]],
            dtype=torch.long,
        )
    if flags is None:
        flags = torch.tensor(
            [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, 1], [1, 0]],
            dtype=torch.uint8,
        )
    views = build_semantic_operators(unique_edges, flags, model.num_nodes)
    model.semantic_pt = views["p_text"]
    model.semantic_pv = views["p_visual"]
    model.semantic_degree_text = views["degree_text"]
    model.semantic_degree_visual = views["degree_visual"]


def _edge_value(operator: torch.Tensor, source: int, target: int) -> float:
    indices = operator.indices()
    positions = torch.nonzero(
        (indices[0] == source) & (indices[1] == target), as_tuple=False
    ).flatten()
    return float(operator.values()[positions[0]]) if positions.numel() else 0.0


def test_physical_operator_deduplicates_removes_loops_and_row_normalizes():
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 0, 2, 3], [0, 1, 0, 2, 1, 1, 2, 3]],
        dtype=torch.long,
    )
    operator, degree = build_physical_operator(edge_index, 5)
    assert operator.layout == torch.sparse_coo
    assert operator._nnz() == 4
    assert degree.tolist() == [1, 2, 1, 0, 0]
    assert torch.allclose(
        torch.sparse.sum(operator, dim=1).to_dense(),
        torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    assert _edge_value(operator, 0, 1) == 1.0
    assert _edge_value(operator, 1, 0) == 0.5
    assert _edge_value(operator, 1, 2) == 0.5
    assert not bool((operator.indices()[0] == operator.indices()[1]).any())


def test_semantic_flags_use_presence_and_both_edges_are_full_in_each_view():
    unique = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    # Counts 2 remain a presence bit. Edge (1,2) is present in both views.
    flags = torch.tensor([[0, 2], [2, 0], [1, 1]], dtype=torch.uint8)
    views = build_semantic_operators(unique, flags, num_nodes=4)
    pt, pv = views["p_text"], views["p_visual"]
    assert pt.layout == pv.layout == torch.sparse_coo
    assert pt._nnz() == pv._nnz() == 4
    assert views["degree_text"].tolist() == [1, 1, 2, 0]
    assert views["degree_visual"].tolist() == [1, 2, 1, 0]
    assert torch.allclose(
        torch.sparse.sum(pt, dim=1).to_dense(),
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )
    assert torch.allclose(
        torch.sparse.sum(pv, dim=1).to_dense(),
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )
    # The shared edge is retained in both relation supports, then normalized
    # independently according to each graph's row degree.
    assert _edge_value(pt, 1, 2) == 1.0
    assert _edge_value(pv, 1, 2) == 0.5
    assert bool((pt.values() > 0).all()) and bool((pv.values() > 0).all())


def test_isolated_rows_propagate_exact_zero_without_dense_adjacency():
    empty = torch.empty((2, 0), dtype=torch.long)
    operator, degree = build_physical_operator(empty, 3)
    states = torch.randn(3, 128)
    message = torch.sparse.mm(operator, states)
    assert operator.layout == torch.sparse_coo and operator._nnz() == 0
    assert degree.tolist() == [0, 0, 0]
    assert torch.equal(message, torch.zeros_like(message))


def test_joint_softmax_masks_invalid_relations_and_handles_all_invalid_rows():
    logits = torch.randn(3, 2, 3)
    valid = torch.tensor([[True, True], [True, False], [False, False]])
    probabilities = joint_masked_softmax(logits, valid)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(
        probabilities.sum(dim=(1, 2)), torch.tensor([1.0, 1.0, 0.0])
    )
    assert torch.equal(probabilities[1, 1], torch.zeros(3))
    assert torch.equal(probabilities[2], torch.zeros(2, 3))
    assert torch.allclose(probabilities[1, 0].sum(), torch.tensor(1.0))


def test_router_has_nearby_trainable_prior_and_node_conditioned_scores():
    layer = _layer()
    n = 128
    h = layer.block_norm(torch.randn(n, 128))
    u = layer.block_norm(torch.randn(n, 128))
    c = layer.block_norm(torch.randn(n, 128))
    count = torch.full((n,), 8, dtype=torch.long)
    logits = torch.stack(
        [
            layer.router_logits(
                h, u, c, count, relation_id=relation, modality_id=0
            )
            for relation in range(2)
        ],
        dim=1,
    )
    probabilities = joint_masked_softmax(
        logits, torch.ones((n, 2), dtype=torch.bool)
    )
    physical_mass = float(probabilities[:, 0].sum(dim=-1).mean())
    assert abs(physical_mass - 0.6) < 0.04
    assert isinstance(layer.router_log_prior, torch.nn.Parameter)
    assert torch.count_nonzero(layer.router[-1].weight) > 0
    assert float(probabilities[:, 0].std(dim=0).max()) > 1.0e-6
    assert layer.router_logits(
        h, u, c, count, relation_id=0, modality_id=0
    ).std(dim=0).mean() > 0


class _RecordingIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, value):
        self.inputs.append(value.detach().clone())
        return value


def test_cross_modal_context_uses_the_target_relations_support():
    model = _model(num_nodes=4)
    layer = model.encoder_layers[0]
    phy, degree_phy = build_physical_operator(
        torch.tensor([[0, 1, 2], [1, 2, 3]]), 4
    )
    unique = torch.tensor([[0, 0, 2], [1, 3, 3]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.uint8)
    views = build_semantic_operators(unique, flags, 4)
    vt, tv = _RecordingIdentity(), _RecordingIdentity()
    layer.cross_modal["visual_to_text"] = vt
    layer.cross_modal["text_to_visual"] = tv
    h_text, h_visual = torch.randn(4, 128), torch.randn(4, 128)
    model._execute_layer(
        0,
        h_text,
        h_visual,
        {"physical": phy, "text": views["p_text"], "visual": views["p_visual"]},
        {
            "physical": degree_phy,
            "text": views["degree_text"],
            "visual": views["degree_visual"],
        },
        intervention={},
        collect_details=False,
    )
    normalized_text = layer.state_norms["text"](h_text)
    normalized_visual = layer.state_norms["visual"](h_visual)
    assert len(vt.inputs) == len(tv.inputs) == 2
    assert torch.allclose(
        vt.inputs[0], torch.sparse.mm(phy, normalized_visual)
    )
    assert torch.allclose(
        vt.inputs[1], torch.sparse.mm(views["p_text"], normalized_visual)
    )
    assert torch.allclose(
        tv.inputs[0], torch.sparse.mm(phy, normalized_text)
    )
    assert torch.allclose(
        tv.inputs[1], torch.sparse.mm(views["p_visual"], normalized_text)
    )
    assert not torch.allclose(
        vt.inputs[1], torch.sparse.mm(views["p_visual"], normalized_visual)
    )
    assert not torch.allclose(
        tv.inputs[1], torch.sparse.mm(views["p_text"], normalized_text)
    )


class _BiasedCross(torch.nn.Module):
    def forward(self, value):
        return torch.ones_like(value)


def test_all_invalid_contexts_cannot_create_cross_or_expert_bias_messages():
    model = _model(num_nodes=3).eval()
    layer = model.encoder_layers[0]
    for direction in ("visual_to_text", "text_to_visual"):
        layer.cross_modal[direction] = _BiasedCross()
    empty = torch.empty((2, 0), dtype=torch.long)
    p0, d0 = build_physical_operator(empty, 3)
    views = build_semantic_operators(
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 2), dtype=torch.uint8),
        3,
    )
    h_text, h_visual = torch.randn(3, 128), torch.randn(3, 128)
    text_next, visual_next, details = model._execute_layer(
        0,
        h_text,
        h_visual,
        {"physical": p0, "text": views["p_text"], "visual": views["p_visual"]},
        {
            "physical": d0,
            "text": views["degree_text"],
            "visual": views["degree_visual"],
        },
        intervention={},
        collect_details=True,
    )
    assert torch.equal(text_next, h_text)
    assert torch.equal(visual_next, h_visual)
    assert details is not None
    for modality in ("text", "visual"):
        assert details["modalities"][modality]["participating_nodes"] == 0
        assert details["modalities"][modality]["mean_joint_expert_use"] == [
            0.0,
            0.0,
            0.0,
        ]
        for relation in ("physical", "semantic"):
            row = details["modalities"][modality]["relations"][relation]
            assert row["cross_context_rms"] == 0.0
            assert row["valid_nodes"] == 0


def test_layer_two_receives_the_updated_states_from_layer_one():
    torch.manual_seed(9)
    model = _model(num_nodes=8).eval()
    _attach_semantic_views(model)
    x = torch.randn(8, 7)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    calls = []
    original = model._execute_layer

    def wrapped(index, h_text, h_visual, *args, **kwargs):
        output = original(index, h_text, h_visual, *args, **kwargs)
        calls.append(
            (
                index,
                h_text.detach().clone(),
                h_visual.detach().clone(),
                output[0].detach().clone(),
                output[1].detach().clone(),
            )
        )
        return output

    model._execute_layer = wrapped
    with torch.no_grad():
        z, *_ = model(x, edge_index)
    assert z.shape == (8, 128)
    assert len(calls) == 2
    assert torch.equal(calls[1][1], calls[0][3])
    assert torch.equal(calls[1][2], calls[0][4])


def test_expert_and_router_sharing_is_layer_local_and_parameters_are_distinct():
    model = _model()
    assert model.encoder_layers[0].experts[0] is model.encoder_layers[0].experts[0]
    assert model.encoder_layers[0].router is model.encoder_layers[0].router
    assert (
        model.encoder_layers[0].experts[0]
        is not model.encoder_layers[1].experts[0]
    )
    assert (
        model.encoder_layers[0].router
        is not model.encoder_layers[1].router
    )
    assert (
        model.encoder_layers[0].experts[0].first.weight
        is not model.encoder_layers[1].experts[0].first.weight
    )


def test_one_expert_bank_and_router_are_called_for_all_four_contexts():
    model = _model(num_nodes=8).eval()
    _attach_semantic_views(model)
    layer = model.encoder_layers[0]
    x = torch.randn(8, 7)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    physical, degree_physical = build_physical_operator(edge_index, 8)
    calls = {"router": 0, "experts": [0, 0, 0]}
    hooks = [
        layer.router.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__(
                "router", calls["router"] + 1
            )
        )
    ]
    for expert_index, expert in enumerate(layer.experts):
        hooks.append(
            expert.register_forward_hook(
                lambda _module, _inputs, _output, index=expert_index: calls[
                    "experts"
                ].__setitem__(index, calls["experts"][index] + 1)
            )
        )
    try:
        with torch.no_grad():
            model._execute_layer(
                0,
                x[:, :4].new_zeros((8, 128)),
                x[:, :4].new_ones((8, 128)),
                {
                    "physical": physical,
                    "text": model.semantic_pt,
                    "visual": model.semantic_pv,
                },
                {
                    "physical": degree_physical,
                    "text": model.semantic_degree_text,
                    "visual": model.semantic_degree_visual,
                },
                intervention={},
                collect_details=False,
            )
    finally:
        for hook in hooks:
            hook.remove()
    # There are four contexts per layer, evaluated in two configured chunks.
    assert calls["router"] == 4
    assert calls["experts"] == [8, 8, 8]
    assert len(layer.experts) == 3


def test_full_forward_backward_adamw_updates_checkpoint_reload_and_zero_aux():
    torch.manual_seed(42)
    model = _model()
    _attach_semantic_views(model)
    x = torch.randn(8, 7)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 0, 2], [1, 2, 3, 4, 5, 6, 7, 3, 6]],
        dtype=torch.long,
    )
    labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    train_idx = torch.tensor([0, 1, 2, 3, 4])
    classifier = torch.nn.Linear(128, 3)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1.0e-3,
        weight_decay=1.0e-4,
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux_loss, _ = model(x, edge_index)
    assert aux_loss.ndim == 0 and aux_loss.item() == 0.0
    loss = torch.nn.functional.cross_entropy(
        classifier(z[train_idx]), labels
    ) + aux_loss
    loss.backward()
    gradient_audit = model.capture_gradient_diagnostics()
    required = {"projectors", "cross_modal", "experts", "router", "fusion"}
    assert required.issubset(gradient_audit)
    for group in required:
        row = gradient_audit[group]
        assert row["gradient_finite"]
        assert row["gradient_nonzero"] > 0
        assert row["gradient_l2_norm"] > 0
        assert row["gradient_present"] == row["parameter_count"]
        assert row["all_parameter_tensors_have_finite_nonzero_gradients"]
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(classifier.parameters()),
        1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    for group in required:
        changed = [
            not torch.equal(before[name], value.detach())
            for name, value in model.named_parameters()
            if (
                name.startswith("projectors.")
                if group == "projectors"
                else ".cross_modal." in name
                if group == "cross_modal"
                else ".experts." in name
                if group == "experts"
                else any(
                    key in name
                    for key in (
                        "router.",
                        "relation_embedding.",
                        "modality_embedding.",
                        "router_log_prior",
                    )
                )
                if group == "router"
                else name.startswith(("fusion.", "skip.", "fusion_norm."))
            )
        ]
        assert changed and any(changed), group

    model.eval()
    classifier.eval()
    with torch.no_grad():
        expected = model(x, edge_index)[0]
    restored = _model()
    _attach_semantic_views(restored)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    with torch.no_grad():
        observed = restored(x, edge_index)[0]
    assert torch.allclose(expected, observed, atol=1.0e-6, rtol=1.0e-6)


def test_eval_label_set_excludes_test_classes_when_test_is_disabled():
    data = MAGData(
        name="toy",
        source="test",
        task="nc",
        x=torch.zeros(3, 1),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        y=torch.tensor([0, 0, 2]),
        train_idx=torch.tensor([0]),
        val_idx=torch.tensor([1]),
        test_idx=torch.tensor([2]),
        num_classes=3,
    )
    assert _resolve_nc_eval_labels(data, include_test=False) == [0]
    assert _resolve_nc_eval_labels(data, include_test=True) == [0, 2]
