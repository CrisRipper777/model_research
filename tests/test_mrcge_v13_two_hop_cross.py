from __future__ import annotations

import io

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from src.models.mrcge_v13 import Model as FrozenV13Model
from src.models.mrcge_v13_two_hop_cross import Model as TwoHopModel
from src.models.psce_mag_v7a import chunked_weighted_spmm


DATA_INFO = {"input_dim": 8, "text_dim": 4, "visual_dim": 4, "num_nodes": 12, "num_classes": 3}
X = torch.arange(96, dtype=torch.float32).reshape(12, 8) / 31.0
PHYSICAL = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]], dtype=torch.long,
)
SEMANTIC = torch.tensor([[0, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10, 0]], dtype=torch.long)
LABELS = torch.tensor([0, 1, 2, 1, 0, 2, 0, 1, 2, 1, 0, 2])


def attach_semantic(model):
    for modality, edges in (("text", SEMANTIC), ("visual", SEMANTIC.flip(0))):
        setattr(model, f"semantic_{modality}_edge_index", edges.clone())
        setattr(model, f"semantic_{modality}_weight", torch.ones(edges.size(1)))
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_feature_fingerprint = "b" * 64
    model.semantic_v7b_fingerprint = "c" * 64


def make_model(variant="raw", seed=42):
    torch.manual_seed(seed)
    cfg = OmegaConf.load("configs/model/mrcge_v13_two_hop_cross.yaml")
    cfg.two_hop_variant = variant
    model = TwoHopModel(OmegaConf.create({"model": cfg}), DATA_INFO)
    attach_semantic(model)
    return model


def make_frozen(seed=42):
    torch.manual_seed(seed)
    cfg = OmegaConf.load("configs/model/mrcge_v13.yaml")
    cfg.condition = "M_repaired"
    cfg.v13_variant = "joint_view"
    cfg.sender_private_ratio = 0.0
    model = FrozenV13Model(OmegaConf.create({"model": cfg}), DATA_INFO)
    attach_semantic(model)
    return model


@pytest.mark.parametrize("variant", ["raw", "complementary"])
def test_zero_lambda_preserves_initialization_eval_outputs_logits_and_train_gradients(variant):
    frozen = make_frozen(seed=123)
    ref_head = nn.Linear(frozen.out_dim, DATA_INFO["num_classes"])
    torch.manual_seed(123)
    candidate = make_model(variant, seed=123)
    new_head = nn.Linear(candidate.out_dim, DATA_INFO["num_classes"])
    state = frozen.state_dict()
    candidate_state = candidate.state_dict()
    assert set(candidate_state) == set(state) | {"two_hop_raw"}
    assert all(torch.equal(value, candidate_state[name]) for name, value in state.items())
    assert torch.equal(candidate.two_hop_raw, torch.zeros(2))
    assert torch.equal(ref_head.weight, new_head.weight)
    assert torch.equal(ref_head.bias, new_head.bias)

    frozen.eval(); candidate.eval()
    with torch.no_grad():
        z0, _, _, aux0, info0 = frozen(X, PHYSICAL, return_details=True)
        z1, _, _, aux1, info1 = candidate(X, PHYSICAL, return_details=True)
    torch.testing.assert_close(z0, z1, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(aux0, aux1, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(ref_head(z0), new_head(z1), atol=2e-6, rtol=2e-5)
    for modality in ("text", "visual"):
        route0 = info0["details"]["collaborative_routes"][modality]
        route1 = info1["details"]["collaborative_routes"][modality]
        for key in ("dense_importance", "top2_selection_share"):
            torch.testing.assert_close(torch.tensor(route0[key]), torch.tensor(route1[key]), atol=2e-6, rtol=2e-5)
        for key in ("dense_entropy", "selected_weight_entropy"):
            assert abs(route0[key] - route1[key]) <= 2e-6
    for modality in ("text", "visual"):
        assert torch.equal(info0["unimodal_logits"][modality], info1["unimodal_logits"][modality])

    frozen.train(); candidate.train(); ref_head.train(); new_head.train()
    torch.manual_seed(20261027)
    z0, _, _, aux0, _ = frozen(X, PHYSICAL)
    (F.cross_entropy(ref_head(z0), LABELS) + aux0).backward()
    torch.manual_seed(20261027)
    z1, _, _, aux1, _ = candidate(X, PHYSICAL)
    (F.cross_entropy(new_head(z1), LABELS) + aux1).backward()
    torch.testing.assert_close(z0, z1, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(aux0, aux1, atol=2e-7, rtol=2e-6)
    ref_params, new_params = dict(frozen.named_parameters()), dict(candidate.named_parameters())
    assert set(new_params) == set(ref_params) | {"two_hop_raw"}
    for name, param in ref_params.items():
        new_grad = new_params[name].grad
        assert (param.grad is None) == (new_grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(new_grad, param.grad, atol=2e-7, rtol=2e-6, msg=name)


def test_raw_and_complementary_share_all_initial_state_and_classifier_rng():
    raw = make_model("raw", seed=211)
    raw_head = nn.Linear(raw.out_dim, DATA_INFO["num_classes"])
    torch.manual_seed(211)
    complementary = make_model("complementary", seed=211)
    comp_head = nn.Linear(complementary.out_dim, DATA_INFO["num_classes"])
    assert raw.state_dict().keys() == complementary.state_dict().keys()
    assert all(torch.equal(v, complementary.state_dict()[k]) for k, v in raw.state_dict().items())
    assert torch.equal(raw_head.weight, comp_head.weight)
    assert torch.equal(raw_head.bias, comp_head.bias)


def test_projection_is_supported_global_and_stop_gradient():
    model = make_model("complementary", seed=301).eval()
    x1 = torch.randn(12, 256, requires_grad=True)
    edges, weights, support, _ = model._physical_operator(PHYSICAL, 12, x1.dtype)
    sender = torch.randn(12, 256, requires_grad=True)
    projection = model.cross_projections["visual_to_text"]
    z, metrics, tensors = model._two_hop_candidate(
        "text", x1, sender, projection, edges, weights, support, collect=True,
    )
    assert torch.isfinite(z).all()
    assert metrics["projection_coefficient"] is not None
    orthogonal = model._masked_inner(tensors["candidate"], tensors["same_node_reference"], support)
    scale = model._masked_inner(tensors["candidate"], tensors["candidate"], support).sqrt()
    ref = model._masked_inner(tensors["same_node_reference"], tensors["same_node_reference"], support).sqrt()
    assert abs(float(orthogonal)) <= 2e-5 * max(float(scale * ref), 1.0)
    grad_sender = torch.autograd.grad(tensors["candidate"].sum(), sender, allow_unused=True)[0]
    assert grad_sender is None
    assert not tensors["same_node_reference"].requires_grad
    assert not tensors["projection_coefficient"].requires_grad
    assert metrics["raw_x2_same_node_cosine_status"] == "ok"
    assert metrics["complementary_residual_same_node_cosine"] is not None


def test_projection_and_candidate_calibration_zero_degenerate_cap_and_empty():
    model = make_model("raw", seed=331)
    support = torch.ones(4, dtype=torch.bool)
    reference = torch.ones(4, 3)
    near_zero, info = model._calibrate_candidate(reference * 1.0e-4, reference, support)
    assert torch.count_nonzero(near_zero) == 0
    assert info["near_zero_degenerate"] == 1
    capped, info = model._calibrate_candidate(reference * 0.1, reference, support)
    assert info["scale_capped"] == 1
    assert float(info["scale"]) == 4.0
    torch.testing.assert_close(model._rms(capped, support), torch.tensor(0.4))
    empty = torch.zeros(4, dtype=torch.bool)
    zero, info = model._calibrate_candidate(reference, reference, empty)
    assert torch.isfinite(zero).all() and torch.count_nonzero(zero) == 0
    assert info["near_zero_degenerate"] == 1 and info["support_nodes"] == 0
    assert not info["scale"].requires_grad


def test_raw_second_hop_reuses_frozen_directed_operator_and_masks_isolated_nodes():
    model = make_model("raw", seed=351)
    directed_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edges, weights, support, _ = model._physical_operator(directed_edges, 4, torch.float32)
    assert support.tolist() == [False, True, True, False]
    x1 = torch.arange(32, dtype=torch.float32).reshape(4, 8).requires_grad_()
    sender = torch.randn(4, 256, requires_grad=True)
    cross_projection = model.cross_projections["visual_to_text"]
    candidate, metrics, tensors = model._two_hop_candidate(
        "text", x1, sender, cross_projection, edges, weights, support, collect=False,
    )
    expected_x2 = chunked_weighted_spmm(x1, edges, weights, model.physical_chunk_size)
    expected_x2 = expected_x2 * support.to(expected_x2.dtype).unsqueeze(-1)
    torch.testing.assert_close(tensors["x2"], expected_x2)
    assert torch.count_nonzero(tensors["x2"][[0, 3]]) == 0
    assert not metrics["scale"].requires_grad
    assert torch.isfinite(candidate).all()


def test_zero_early_cross_produces_zero_calibrated_raw_and_complementary_candidates():
    model_raw = make_model("raw", seed=361)
    model_comp = make_model("complementary", seed=361)
    edges, weights, support, _ = model_raw._physical_operator(PHYSICAL, 12, torch.float32)
    x1 = torch.zeros(12, 256, requires_grad=True)
    sender = torch.randn(12, 256, requires_grad=True)
    for model in (model_raw, model_comp):
        candidate, metrics, tensors = model._two_hop_candidate(
            "text", x1, sender, model.cross_projections["visual_to_text"],
            edges, weights, support, collect=True,
        )
        assert torch.count_nonzero(tensors["x2"]) == 0
        assert torch.count_nonzero(candidate) == 0
        assert metrics["near_zero_degenerate"] == 1
        assert metrics["scale"] == 0.0


@pytest.mark.parametrize("variant", ["raw", "complementary"])
def test_two_hop_parameter_has_finite_nonzero_gradient_and_is_optimizer_managed(variant):
    model = make_model(variant, seed=401)
    head = nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    assert any(parameter is model.two_hop_raw for group in optimizer.param_groups for parameter in group["params"])
    model.train()
    z, _, _, aux, _ = model(X, PHYSICAL)
    weights = torch.linspace(0.15, 1.75, z.numel()).reshape_as(z)
    (z.mul(weights).sum() + aux).backward()
    grad = model.two_hop_raw.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) == 2


@pytest.mark.parametrize("variant", ["raw", "complementary"])
def test_lambda_changes_expert_input_but_not_router_input(variant):
    model = make_model(variant, seed=501).eval()
    with torch.no_grad():
        _, _, _, _, before = model(X, PHYSICAL, return_details=True)
        model.two_hop_raw.copy_(torch.tensor([0.6, -0.5]))
        _, _, _, _, after = model(X, PHYSICAL, return_details=True)
    assert before["details"]["collaborative_routes"] == after["details"]["collaborative_routes"]
    assert before["details"]["routes"] == after["details"]["routes"]
    for modality in ("text", "visual"):
        old = before["details"]["filters"][f"collaborative_{modality}_physical"]["G_expert_rms"]
        new = after["details"]["filters"][f"collaborative_{modality}_physical"]["G_expert_rms"]
        assert old != new
        injection = after["details"]["two_hop_cross"][modality]["expert_injection"]
        assert any(value > 0 for value in injection["expert_addition_rms"])
        assert len(injection["expert_addition_over_old_input_rms"]) == 4


@pytest.mark.parametrize("variant", ["raw", "complementary"])
def test_low_memory_checkpoint_and_collect_path_match_outputs_and_gradients(variant):
    stacked = make_model(variant, seed=601)
    low = make_model(variant, seed=601)
    low.load_state_dict(stacked.state_dict(), strict=True)
    with torch.no_grad():
        stacked.two_hop_raw.copy_(torch.tensor([0.12, -0.09]))
        low.two_hop_raw.copy_(stacked.two_hop_raw)
    for model in (stacked, low):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        model.late_cross_dropout = 0.0
        model.train()
    head_a, head_b = nn.Linear(256, 3), nn.Linear(256, 3)
    head_b.load_state_dict(head_a.state_dict())
    z_a, _, _, aux_a, _ = stacked(X, PHYSICAL, return_details=True)
    (F.cross_entropy(head_a(z_a), LABELS) + aux_a).backward()
    z_b, _, _, aux_b, _ = low(X, PHYSICAL)
    (F.cross_entropy(head_b(z_b), LABELS) + aux_b).backward()
    torch.testing.assert_close(z_a, z_b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(aux_a, aux_b, atol=2e-6, rtol=2e-5)
    for name, parameter in stacked.named_parameters():
        other = dict(low.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(other.grad, parameter.grad, atol=5e-5, rtol=5e-4, msg=name)


@pytest.mark.parametrize("variant", ["raw", "complementary"])
def test_strict_reload_empty_graph_is_finite_and_eval_does_not_advance_rng(variant):
    model = make_model(variant, seed=701).eval()
    with torch.no_grad():
        model.two_hop_raw.copy_(torch.tensor([0.3, -0.2]))
        expected = model(X, PHYSICAL, return_details=True)[0]
    state = model.state_dict()
    restored = make_model(variant, seed=702).eval()
    restored.load_state_dict(state, strict=True)
    cpu_rng = torch.random.get_rng_state().clone()
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        actual = restored(X, PHYSICAL, return_details=True)[0]
        empty = restored(X, empty_edges, return_details=True)
    assert torch.equal(expected, actual)
    assert torch.isfinite(empty[0]).all() and torch.isfinite(empty[3])
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert empty[4]["details"]["physical_support_nodes"] == 0
    for modality in ("text", "visual"):
        info = empty[4]["details"]["two_hop_cross"][modality]
        assert info["near_zero_degenerate"] == 1
        assert info["calibrated_rms"] == 0.0

