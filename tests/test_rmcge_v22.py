from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from src.models.psce_mag_v7a import chunked_weighted_spmm
from src.models.rmcge_v2 import Model as V2Model, build_modality_knn_operators
from src.models.rmcge_v21 import Model as V21Model
from src.models.rmcge_v22 import ChunkedPairGateAggregate, Model
from src.utils.seeds import set_seed


def _cfg(condition: str):
    return OmegaConf.create({"model": {
        "name": "rmcge_v22", "condition": condition, "variant": "C0",
        "hidden_dim": 256, "physical_hops": 4, "semantic_hops": 2,
        "num_experts": 4, "top_k": 2, "expert_bottleneck": 64,
        "expert_feature_scale": 0.1, "router_dim": 64, "router_hidden_dim": 128,
        "modality_embed_dim": 16, "text_knn_k": 8, "visual_knn_k": 8,
        "relation_dim": 32, "relation_hidden_dim": 64,
        "semantic_edge_weight_init": 0.8, "semantic_strength_init": 0.2,
        "dropout": 0.2, "balance_weight": 0.01, "eps": 1e-8,
        "relation_chunk_size": 16, "propagation_chunk_size": 3,
        "edge_chunk_size": 3, "purpose_edge_chunk_size": 3,
        "content_edge_chunk_size": 2, "content_rank": 32,
        "content_gate_hidden": 64, "modality_embedding_dim": 8,
        "node_chunk_size": 2, "scale_floor": 1e-3,
    }})


def _info(num_nodes: int = 8):
    return {"input_dim": 8, "text_dim": 4, "visual_dim": 4, "num_nodes": num_nodes, "num_classes": 3}


def _graph():
    return torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)


def _attach_cache(model):
    unique = torch.tensor([[0, 0, 1, 2, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 2], [1, 1], [0, 1], [1, 1]], dtype=torch.uint8)
    supports = build_modality_knn_operators(unique, flags, model.num_nodes)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index_v2", supports[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight_v2", supports[modality]["weight"])
    model.semantic_candidate_fingerprint = "synthetic-cache"
    model.semantic_v7b_fingerprint = "synthetic-v7b"


def _gate_model(condition="M3"):
    set_seed(117)
    model = Model(_cfg(condition), _info())
    # Use an intentionally nonzero gate for the expressivity tests.
    with torch.no_grad():
        model.content_gate[-1].weight.normal_(0, 0.1)
        model.content_gate[-1].bias.normal_(0, 0.1)
    return model


def test_m0_matches_frozen_v21_c0_forward_and_classifier_initialization():
    set_seed(120)
    candidate = Model(_cfg("M0"), _info()).eval()
    head = torch.nn.Linear(256, 3)
    set_seed(120)
    reference = V21Model(_cfg("C0"), _info()).eval()
    reference_head = torch.nn.Linear(256, 3)
    _attach_cache(candidate)
    _attach_cache(reference)
    common = set(reference.state_dict())
    for key in common:
        assert torch.equal(candidate.state_dict()[key], reference.state_dict()[key]), key
    assert all(torch.equal(head.state_dict()[k], reference_head.state_dict()[k]) for k in head.state_dict())
    x, edge = torch.randn(8, 8), _graph()
    with torch.no_grad():
        actual = candidate(x, edge, return_details=True)
        expected = reference(x, edge, return_details=True)
    for left, right in zip(actual[:4], expected[:4]):
        if left is not None:
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert actual[4]["variant"] == "M0"


def test_all_arms_have_identical_content_and_shared_public_initialization():
    models, heads = {}, {}
    for condition in ("M0", "M1", "M2", "M3", "M4"):
        set_seed(121)
        models[condition] = Model(_cfg(condition), _info())
        heads[condition] = torch.nn.Linear(256, 3)
    reference_params = dict(models["M0"].named_parameters())
    for condition, candidate in models.items():
        params = dict(candidate.named_parameters())
        for name in reference_params:
            assert torch.equal(reference_params[name], params[name]), (condition, name)
        assert all(torch.equal(heads["M0"].state_dict()[k], heads[condition].state_dict()[k]) for k in heads["M0"].state_dict())
    content_keys = [key for key in models["M0"].state_dict() if key.startswith("content_")]
    assert content_keys
    for condition in ("M1", "M2", "M3", "M4"):
        assert all(torch.equal(models["M0"].state_dict()[key], models[condition].state_dict()[key]) for key in content_keys)
    assert torch.count_nonzero(models["M3"].content_gate[-1].weight) == 0
    assert torch.count_nonzero(models["M3"].content_gate[-1].bias) == 0
    assert torch.count_nonzero(models["M3"].content_output.weight) > 0
    assert torch.equal(models["M3"].content_eta, torch.full((4,), 0.25))


def test_content_off_and_intra_off_restore_c0_logits_for_every_arm():
    models = {}
    for condition in ("M0", "M1", "M2", "M3", "M4"):
        set_seed(122)
        models[condition] = Model(_cfg(condition), _info()).eval()
    x, edge = torch.randn(8, 8), _graph()
    for model in models.values():
        _attach_cache(model)
        model.set_interventions(content_off=True, intra_selection_off=True)
    with torch.no_grad():
        reference = models["M0"](x, edge)[0]
        for condition in ("M1", "M2", "M3", "M4"):
            torch.testing.assert_close(models[condition](x, edge)[0], reference, atol=2e-6, rtol=2e-6)


def test_m1_and_m2_gate_inputs_only_depend_on_declared_endpoint():
    model = _gate_model()
    q = torch.randn(5, 32)
    k = torch.randn(5, 32)
    modality = model.content_modality_embedding.weight[0]
    sender_a = model._edge_gates(q, k, "sender", modality)
    sender_b = model._edge_gates(torch.randn_like(q) * 100, k, "sender", modality)
    receiver_a = model._edge_gates(q, k, "receiver", modality)
    receiver_b = model._edge_gates(q, torch.randn_like(k) * 100, "receiver", modality)
    torch.testing.assert_close(sender_a, sender_b, atol=0, rtol=0)
    torch.testing.assert_close(receiver_a, receiver_b, atol=0, rtol=0)


def test_m1_sender_precomputation_and_m2_post_aggregation_are_equivalent():
    model = _gate_model()
    src = torch.tensor([0, 1, 2, 0, 2, 3])
    dst = torch.tensor([3, 3, 3, 4, 4, 4])
    edge_index = torch.stack((src, dst))
    weight = torch.tensor([0.7, 0.6, 0.5, 0.8, 0.4, 0.9])
    q, k, v = torch.randn(5, 32), torch.randn(5, 32), torch.randn(5, 32)
    modality = model.content_modality_embedding.weight[1]
    sender_gate = model._edge_gates(torch.zeros_like(k), k, "sender", modality)
    precomputed = chunked_weighted_spmm(sender_gate * v, edge_index, weight, 2)
    sender_reference = torch.zeros_like(precomputed)
    for index in range(src.numel()):
        sender_reference[dst[index]] += weight[index] * sender_gate[src[index]] * v[src[index]]
    torch.testing.assert_close(precomputed, sender_reference, atol=1e-6, rtol=1e-6)

    receiver_gate = model._edge_gates(q, torch.zeros_like(q), "receiver", modality)
    aggregate_then_gate = receiver_gate * chunked_weighted_spmm(v, edge_index, weight, 2)
    edge_gate = receiver_gate[dst]
    gate_then_aggregate = torch.zeros_like(aggregate_then_gate)
    gate_then_aggregate.index_add_(0, dst, weight[:, None] * edge_gate * v[src])
    torch.testing.assert_close(aggregate_then_gate, gate_then_aggregate, atol=1e-6, rtol=1e-6)


def test_pair_gate_changes_channels_by_receiver_beyond_scalar_edge_reweighting():
    model = _gate_model()
    # Make gate channel 0 = f(q0*k0) and channel 1 = -f(q0*k0).
    with torch.no_grad():
        model.content_gate[0].weight.zero_()
        model.content_gate[0].bias.zero_()
        model.content_gate[0].weight[0, 64] = 1.0
        model.content_gate[2].weight.zero_()
        model.content_gate[2].bias.zero_()
        model.content_gate[2].weight[0, 0] = 1.0
        model.content_gate[2].weight[1, 0] = -1.0
    q = torch.zeros(2, 32)
    q[:, 0] = torch.tensor([1.0, -1.0])
    k = torch.zeros(2, 32)
    k[:, 0] = 1.0
    v = torch.zeros(2, 32)
    v[:, 0:2] = 1.0
    src, dst = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])
    physical = torch.full((4,), 0.5)
    modality = model.content_modality_embedding.weight[0]
    output = ChunkedPairGateAggregate.apply(
        q, k, v, src, dst, physical, modality,
        model.content_gate[0].weight, model.content_gate[0].bias,
        model.content_gate[2].weight, model.content_gate[2].bias, 1,
    )
    assert not torch.allclose(output[0], output[1])
    assert not torch.allclose(output[0, 0], output[0, 1])
    # Any scalar weighting of this shared value vector has equal first channels.
    assert not torch.allclose(output[0], torch.full_like(output[0], output[0, 0].item()))


def test_pair_gate_custom_backward_matches_dense_reference():
    torch.manual_seed(123)
    n, rank = 5, 32
    q, k, v = (torch.randn(n, rank, requires_grad=True) for _ in range(3))
    src = torch.tensor([0, 1, 2, 0, 2, 3, 4])
    dst = torch.tensor([1, 1, 1, 3, 3, 4, 4])
    weight = torch.rand(src.numel())
    modality = torch.randn(8, requires_grad=True)
    w1 = torch.randn(64, 136, requires_grad=True) * 0.03
    b1 = torch.randn(64, requires_grad=True) * 0.02
    w2 = torch.randn(32, 64, requires_grad=True) * 0.04
    b2 = torch.randn(32, requires_grad=True) * 0.01
    args = (q, k, v, src, dst, weight, modality, w1, b1, w2, b2)
    actual = ChunkedPairGateAggregate.apply(*args, 2)
    q_e, k_e = q[dst], k[src]
    features = torch.cat((q_e, k_e, q_e * k_e, (q_e - k_e).abs(), modality.expand(src.numel(), -1)), -1)
    gate = F_gelu_linear(features, w1, b1, w2, b2)
    expected = torch.zeros(n, rank)
    expected.index_add_(0, dst, gate * v[src] * weight[:, None])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    probe = torch.randn_like(actual)
    variables = (q, k, v, modality, w1, b1, w2, b2)
    grad_actual = torch.autograd.grad(actual, variables, probe, retain_graph=True)
    grad_expected = torch.autograd.grad(expected, variables, probe)
    for actual_grad, expected_grad in zip(grad_actual, grad_expected):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)


def F_gelu_linear(features, w1, b1, w2, b2):
    return torch.tanh(torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(features, w1, b1)), w2, b2))


def test_real_loss_reaches_zero_initialized_gate_output_layer():
    model = Model(_cfg("M3"), _info()).train()
    _attach_cache(model)
    x, edge = torch.randn(8, 8), _graph()
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    logits = torch.nn.Linear(256, 3)(model(x, edge)[0])
    torch.nn.functional.cross_entropy(logits[:6], labels[:6]).backward()
    gradient = model.content_gate[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0


def test_zero_degree_content_is_exactly_zero_and_degree_one_can_be_nonzero():
    model = _gate_model("M3").eval()
    prior, cross = torch.randn(4, 256), torch.randn(4, 256)
    src, dst = torch.tensor([0]), torch.tensor([1])
    physical = torch.tensor([0.75])
    active = torch.tensor([False, True, False, False])
    scale = torch.ones(256)
    with torch.no_grad():
        normalized, _ = model._content_residual(prior, cross, src, dst, physical, active, scale, 0)
    assert torch.count_nonzero(normalized[0]) == 0
    assert torch.count_nonzero(normalized[2:]) == 0
    assert normalized[1].norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke requires an authorized visible GPU")
def test_cuda_pair_gate_backward_smoke():
    device = torch.device("cuda:0")
    q = torch.randn(6, 32, device=device, requires_grad=True)
    k = torch.randn(6, 32, device=device, requires_grad=True)
    v = torch.randn(6, 32, device=device, requires_grad=True)
    src = torch.tensor([0, 1, 2, 0, 2, 3, 4, 5], device=device)
    dst = torch.tensor([1, 1, 1, 3, 3, 4, 4, 4], device=device)
    physical = torch.rand(src.numel(), device=device)
    modality = torch.randn(8, device=device, requires_grad=True)
    w1 = torch.nn.Parameter(torch.randn(64, 136, device=device) * 0.02)
    b1 = torch.nn.Parameter(torch.zeros(64, device=device))
    w2 = torch.nn.Parameter(torch.randn(32, 64, device=device) * 0.02)
    b2 = torch.nn.Parameter(torch.zeros(32, device=device))
    out = ChunkedPairGateAggregate.apply(q, k, v, src, dst, physical, modality, w1, b1, w2, b2, 3)
    out.square().sum().backward()
    torch.cuda.synchronize(device)
    for tensor in (q.grad, k.grad, v.grad, modality.grad, w1.grad, b1.grad, w2.grad, b2.grad):
        assert tensor is not None and torch.isfinite(tensor).all() and tensor.norm() > 0
