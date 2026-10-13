from __future__ import annotations

import io

import torch
from omegaconf import OmegaConf

import src.models.rmcge_v2 as rmcge_v2_module
from src.models.psce_mag_v7a import Model as V7AModel, chunked_weighted_spmm
from src.models.rmcge_v2 import Model, build_modality_knn_operators
from src.utils.seeds import set_seed


def _cfg(variant: str):
    return OmegaConf.create(
        {
            "model": {
                "name": "rmcge_v2",
                "variant": variant,
                "hidden_dim": 256,
                "physical_hops": 4,
                "semantic_hops": 2,
                "num_experts": 4,
                "top_k": 2,
                "expert_bottleneck": 64,
                "expert_feature_scale": 0.1,
                "router_dim": 64,
                "router_hidden_dim": 128,
                "modality_embed_dim": 16,
                "text_knn_k": 8,
                "visual_knn_k": 8,
                "relation_dim": 32,
                "relation_hidden_dim": 64,
                "semantic_edge_weight_init": 0.8,
                "semantic_strength_init": 0.2,
                "dropout": 0.2,
                "balance_weight": 0.01,
                "eps": 1e-8,
                "relation_chunk_size": 32,
                "propagation_chunk_size": 16,
                "edge_chunk_size": 3,
                "node_chunk_size": 2,
                "scale_floor": 1e-3,
            }
        }
    )


def _info(num_nodes: int = 6):
    return {
        "input_dim": 4,
        "text_dim": 2,
        "visual_dim": 2,
        "num_nodes": num_nodes,
        "num_classes": 3,
    }


def _graph():
    # Reciprocal edges, a physical isolate, and an input self-loop.
    return torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 0, 2, 1, 3, 2, 4, 3, 4]],
        dtype=torch.long,
    )


def _attach_small_knn(model: Model):
    unique = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 2], [1, 1], [0, 1]], dtype=torch.uint8)
    supports = build_modality_knn_operators(unique, flags, model.num_nodes)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index_v2", supports[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight_v2", supports[modality]["weight"])
    model.semantic_source_flags_v7b = flags
    model.semantic_candidate_fingerprint = "synthetic-cache"
    model.semantic_v7b_fingerprint = "synthetic-v7b"


def _assert_finite_gradients(model: Model, names: tuple[str, ...]):
    params = dict(model.named_parameters())
    for name in names:
        grad = params[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.norm() > 0, name


def test_v0_matches_v7a_a0_raw_state_and_forward():
    set_seed(42)
    candidate = Model(_cfg("V0"), _info())
    set_seed(42)
    reference_cfg = _cfg("V0")
    reference_cfg.model.variant = "a0_raw"
    reference = V7AModel(reference_cfg, _info())
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    for key in candidate.state_dict():
        assert torch.equal(candidate.state_dict()[key], reference.state_dict()[key]), key
    x = torch.randn(6, 4)
    edge = _graph()
    candidate.eval()
    reference.eval()
    with torch.no_grad():
        actual = candidate(x, edge, return_details=True)
        expected = reference(x, edge, return_details=True)
    for actual_tensor, expected_tensor in zip(actual[:4], expected[:4]):
        if actual_tensor is not None:
            assert torch.equal(actual_tensor, expected_tensor)
    assert torch.equal(actual[4]["active_nodes"], expected[4]["active_nodes"])
    assert torch.equal(actual[4]["balance_loss"], expected[4]["balance_loss"])


def test_v1_v2_parameters_and_classifier_are_paired_and_unit_edges_match():
    set_seed(43)
    v1 = Model(_cfg("V1"), _info())
    head_v1 = torch.nn.Linear(256, 3)
    set_seed(43)
    v2 = Model(_cfg("V2"), _info())
    head_v2 = torch.nn.Linear(256, 3)
    assert v1.state_dict().keys() == v2.state_dict().keys()
    for key in v1.state_dict():
        assert torch.equal(v1.state_dict()[key], v2.state_dict()[key]), key
    assert torch.equal(head_v1.weight, head_v2.weight)
    assert torch.equal(head_v1.bias, head_v2.bias)
    _attach_small_knn(v1)
    _attach_small_knn(v2)
    v2.set_interventions(force_unit_edges=True)
    v1.eval()
    v2.eval()
    x, edge = torch.randn(6, 4), _graph()
    with torch.no_grad():
        one = v1(x, edge)
        two = v2(x, edge, return_details=True)
    for actual, expected in zip(one[:4], two[:4]):
        if actual is not None:
            assert torch.equal(actual, expected)
    text_details = two[4]["details"]["modalities"]["text"]
    assert len(text_details["expert_output_rms_by_expert"]) == 4
    assert len(text_details["expert_output_pairwise_cosine"]) == 6


def test_v7b_both_edges_are_present_in_both_normalized_knn_views():
    unique = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    flags = torch.tensor([[0, 2], [2, 0], [1, 1]], dtype=torch.uint8)
    views = build_modality_knn_operators(unique, flags, 4)
    text_edges = views["text"]["edge_index"]
    visual_edges = views["visual"]["edge_index"]
    assert text_edges.shape[1] == 4
    assert visual_edges.shape[1] == 4
    both = (text_edges[0] == 1) & (text_edges[1] == 2)
    both_visual = (visual_edges[0] == 1) & (visual_edges[1] == 2)
    assert views["text"]["weight"][both].item() == 0.5
    # The edge remains present in the Visual graph and is normalized against
    # that graph's own target degree (one here); the Text degree differs.
    assert views["visual"]["weight"][both_visual].item() == 1.0
    assert not bool((text_edges[0] == 3).any())
    assert not bool((visual_edges[0] == 3).any())


def test_empty_knn_rows_and_physical_isolates_receive_zero_context(monkeypatch):
    model = Model(_cfg("V1"), _info()).eval()
    _attach_small_knn(model)
    model.set_interventions(knn_off=True, node_router_off=True)
    x, edge = torch.randn(6, 4), _graph()
    original = rmcge_v2_module.chunked_weighted_spmm
    calls = []

    def recording_spmm(state, edge_index, weight, chunk_size):
        output = original(state, edge_index, weight, chunk_size)
        calls.append((state.detach().clone(), edge_index.detach().clone(), weight.detach().clone(), output.detach().clone()))
        return output

    monkeypatch.setattr(rmcge_v2_module, "chunked_weighted_spmm", recording_spmm)
    with torch.no_grad():
        model(x, edge)
    # With V1 and KNN disabled, the two calls are Text<-Visual then Visual<-Text.
    text_prior = model.projectors["text"](x[:, :2])
    visual_prior = model.projectors["visual"](x[:, 2:])
    src, dst, norm, active, _ = model._normalized_operator(edge, x.size(0), text_prior.dtype)
    directed = torch.stack((src, dst))
    expected_t_input = model.cross_projections["visual_to_text"](visual_prior)
    expected_v_input = model.cross_projections["text_to_visual"](text_prior)
    assert len(calls) == 2
    assert torch.equal(calls[0][0], expected_t_input)
    assert torch.equal(calls[1][0], expected_v_input)
    assert torch.equal(calls[0][3], original(expected_t_input, directed, norm, 16))
    assert torch.equal(calls[1][3], original(expected_v_input, directed, norm, 16))
    assert torch.equal(calls[0][3][5], torch.zeros(256))
    assert torch.equal(calls[1][3][5], torch.zeros(256))


def test_empty_sparse_message_is_exact_zero_and_backpropagates_safely():
    x = torch.randn(3, 5, requires_grad=True)
    edges = torch.empty((2, 0), dtype=torch.long)
    weights = torch.empty((0,), requires_grad=True)
    result = chunked_weighted_spmm(x, edges, weights, chunk_size=2)
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert weights.grad is not None and weights.grad.numel() == 0


def test_edge_scorer_uses_ordered_receiver_and_sender_features():
    model = Model(_cfg("V2"), _info()).eval()
    # First hidden unit reads q_text[receiver, 0] - q_text[sender, 0].
    with torch.no_grad():
        for parameter in model.edge_scorer.parameters():
            parameter.zero_()
        model.edge_scorer[0].weight[0, 0] = 1.0
        model.edge_scorer[0].weight[0, 32] = -1.0
        model.edge_scorer[2].weight[0, 0] = 1.0
    text = torch.tensor([[1.0] + [0.0] * 255, [3.0] + [0.0] * 255])
    visual = torch.zeros_like(text)
    q_text = text[:, :32]
    q_visual = visual[:, :32]
    degree = torch.ones(2)
    forward = model._score_edges(
        text=text, visual=visual, q_text=q_text, q_visual=q_visual,
        src=torch.tensor([0]), dst=torch.tensor([1]), degree=degree,
        modality_index=0,
    )
    reverse = model._score_edges(
        text=text, visual=visual, q_text=q_text, q_visual=q_visual,
        src=torch.tensor([1]), dst=torch.tensor([0]), degree=degree,
        modality_index=0,
    )
    assert not torch.equal(forward, reverse)


def test_chunked_weighted_spmm_matches_reference_forward_and_backward():
    torch.manual_seed(4)
    x = torch.randn(5, 3, requires_grad=True)
    weight = torch.randn(8, requires_grad=True)
    edge = torch.tensor(
        [[0, 1, 2, 3, 1, 4, 0, 2], [1, 1, 0, 2, 3, 0, 4, 4]], dtype=torch.long
    )
    probe = torch.randn(5, 3)
    output = chunked_weighted_spmm(x, edge, weight, chunk_size=3)
    loss = (output * probe).sum()
    grad_x, grad_weight = torch.autograd.grad(loss, (x, weight))
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    expected = torch.zeros_like(x_ref).index_add(
        0, edge[1], x_ref[edge[0]] * weight_ref.unsqueeze(-1)
    )
    ref_loss = (expected * probe).sum()
    ref_grad_x, ref_grad_weight = torch.autograd.grad(ref_loss, (x_ref, weight_ref))
    assert torch.allclose(output, expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grad_x, ref_grad_x, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grad_weight, ref_grad_weight, atol=1e-7, rtol=1e-6)


def test_v2_first_step_gradient_and_adamw_updates():
    set_seed(44)
    model = Model(_cfg("V2"), _info()).train()
    _attach_small_knn(model)
    x, edge = torch.randn(6, 4), _graph()
    head = torch.nn.Linear(256, 3)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    z, _, _, aux, _ = model(x, edge)
    loss = torch.nn.functional.cross_entropy(head(z[:4]), torch.tensor([0, 1, 2, 0])) + aux
    loss.backward()
    diagnostics = model.capture_gradient_diagnostics()
    assert diagnostics["edge_scorer.2.weight"]["gradient_nonzero"]
    assert diagnostics["edge_scorer.2.bias"]["gradient_nonzero"]
    _assert_finite_gradients(
        model,
        (
            "cross_projections.visual_to_text.0.weight",
            "cross_projections.text_to_visual.0.weight",
            "cross_coefficients",
            "node_router.3.weight",
            "experts.0.linear1.weight",
            "projectors.text.linear1.weight",
            "fusion_linear1.weight",
        ),
    )
    assert model.named_parameters() is not None
    optimizer.step()
    updated = dict(model.named_parameters())
    for name in (
        "edge_scorer.2.weight",
        "cross_projections.visual_to_text.0.weight",
        "cross_projections.text_to_visual.0.weight",
        "cross_projections.visual_to_text.2.weight",
        "cross_projections.text_to_visual.2.weight",
        "node_router.3.weight",
        "experts.0.linear1.weight",
    ):
        assert not torch.equal(before[name], updated[name]), name
    optimizer.zero_grad(set_to_none=True)
    z2, _, _, aux2, _ = model(x, edge)
    loss2 = torch.nn.functional.cross_entropy(head(z2[:4]), torch.tensor([0, 1, 2, 0])) + aux2
    loss2.backward()
    second_step = model.capture_gradient_diagnostics()
    assert second_step["edge_scorer.0.weight"]["gradient_nonzero"]
    assert second_step["delta_coefficients"]["gradient_nonzero"]
    assert second_step["edge_scorer.0.weight"]["gradient_finite"]
    optimizer.step()


def test_checkpoint_round_trip_reproduces_logits():
    set_seed(45)
    source = Model(_cfg("V2"), _info()).eval()
    _attach_small_knn(source)
    x, edge = torch.randn(6, 4), _graph()
    with torch.no_grad():
        expected = source(x, edge)[0]
    handle = io.BytesIO()
    torch.save(source.state_dict(), handle)
    handle.seek(0)
    restored = Model(_cfg("V2"), _info()).eval()
    _attach_small_knn(restored)
    restored.load_state_dict(torch.load(handle, weights_only=True))
    with torch.no_grad():
        actual = restored(x, edge)[0]
    assert torch.equal(actual, expected)


def test_disabled_added_paths_restore_parent_forward():
    for variant in ("V1", "V2"):
        set_seed(46)
        candidate = Model(_cfg(variant), _info()).eval()
        set_seed(46)
        reference_cfg = _cfg("V0")
        reference_cfg.model.variant = "a0_raw"
        reference = V7AModel(reference_cfg, _info()).eval()
        _attach_small_knn(candidate)
        candidate.set_interventions(
            force_unit_edges=True, cross_off=True, knn_off=True, node_router_off=True
        )
        x, edge = torch.randn(6, 4), _graph()
        with torch.no_grad():
            actual = candidate(x, edge)
            expected = reference(x, edge)
        for actual_tensor, expected_tensor in zip(actual[:4], expected[:4]):
            if actual_tensor is not None:
                assert torch.equal(actual_tensor, expected_tensor)
