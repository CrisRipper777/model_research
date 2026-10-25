from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from types import SimpleNamespace

from src.models import mrcge_v11, mrcge_v12
from src.tasks.nc import _compose_full_graph_loss


def cfg(condition: str, *, physical_hops: int = 3, semantic_hops: int = 1):
    return OmegaConf.create({"model": {
        "condition": condition, "hidden_dim": 256, "physical_hops": physical_hops,
        "semantic_hops": semantic_hops, "num_experts": 4, "top_k": 2,
        "expert_bottleneck": 64, "expert_eta": 0.1, "dropout": 0.2,
        "balance_weight": 0.01, "rho_init": 0.25, "gamma_init": 0.08,
        "beta_init": 0.25, "same_node_enabled": True, "late_cross_dropout": 0.15,
        "semantic_knn_k": 8, "eps": 1.0e-8, "physical_chunk_size": 64,
        "semantic_chunk_size": 64, "cross_edge_chunk_size": 64,
        "attention_dim": 32, "router_hidden_dim": 256, "router_embed_dim": 8,
        "channel_gate_hidden": 128, "fusion_dropout": 0.2,
    }})


def make_model(condition: str, *, legacy: bool = False, hops=(3, 1), seed: int = 123):
    torch.manual_seed(seed)
    module = mrcge_v11 if legacy else mrcge_v12
    model = module.Model(cfg("M" if legacy else condition, physical_hops=hops[0], semantic_hops=hops[1]), {
        "input_dim": 12, "text_dim": 5, "visual_dim": 7,
        "num_nodes": 12, "num_classes": 3,
    })
    # Lightweight, label-free V7B-shaped operators for CPU unit tests.
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]], dtype=torch.long)
    weights = torch.ones(edges.size(1), dtype=torch.float32)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index", edges.clone())
        setattr(model, f"semantic_{modality}_weight", weights.clone())
    model.semantic_candidate_fingerprint = "1" * 64
    model.semantic_feature_fingerprint = "2" * 64
    model.semantic_v7b_fingerprint = "3" * 64
    model.semantic_candidate_fingerprint_state.copy_(torch.tensor(list(bytes.fromhex("1" * 64)), dtype=torch.uint8))
    model.semantic_feature_fingerprint_state.copy_(torch.tensor(list(bytes.fromhex("2" * 64)), dtype=torch.uint8))
    return model


def common_state(model):
    excluded = ("single_filter.theta", "filter_bank.theta")
    return {k: v for k, v in model.state_dict().items() if k not in excluded}


def m_filter_reference(model, basis, support, current, cross, modality_id=0, relation_id=0):
    outputs, _, _, _ = model.filter_bank(basis, support, len(basis) - 1)
    first = basis[1] if len(basis) > 1 else basis[0]
    logits = model.router(current, first, cross, modality_id, relation_id)
    dense = torch.softmax(logits.float(), dim=-1)
    top_logits, top_ids = torch.topk(logits, model.top_k, dim=-1)
    top_weights = torch.softmax(top_logits.float(), dim=-1).to(outputs.dtype)
    route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights.float()).to(outputs.dtype)
    selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
    if bool(support.any()):
        importance = dense[support].mean(dim=0)
        share = selected[support].float().mean(dim=0) / model.top_k
        balance = model.num_experts * (importance * share).sum()
    else:
        balance = dense.sum() * 0.0
    response = (outputs * route_weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    return response, balance


def test_hops_and_configuration_validation():
    model = make_model("M_repaired")
    assert model.physical_hops == 3 and model.semantic_hops == 1
    with pytest.raises(ValueError, match="physical_hops=3"):
        make_model("M_repaired", hops=(4, 2))
    assert len(model._basis(torch.randn(12, 256), torch.tensor([[0, 1], [1, 2]]),
                            torch.ones(2), torch.ones(12, dtype=torch.bool),
                            model.physical_hops, 64)) == 4


def test_condition_initializations_and_external_classifier_rng():
    s = make_model("S_repaired", seed=42)
    s_head = torch.nn.Linear(256, 3)
    f = make_model("F_sinit", seed=42)
    f_head = torch.nn.Linear(256, 3)
    assert torch.equal(s.single_filter.theta, torch.ones(4))
    assert torch.equal(f.single_filter.theta, torch.ones(4))
    assert f.single_filter.theta.requires_grad and not s.single_filter.theta.requires_grad
    assert sum(p.numel() for p in f.parameters()) - sum(p.numel() for p in s.parameters()) == 4
    assert common_state(s).keys() == common_state(f).keys()
    assert all(torch.equal(common_state(s)[k], common_state(f)[k]) for k in common_state(s))
    assert all(torch.equal(s_head.state_dict()[k], f_head.state_dict()[k]) for k in s_head.state_dict())

    repaired = make_model("M_repaired", seed=42)
    anchor = make_model("M_anchor", seed=42)
    expected = torch.tensor([0.40, 0.30, 0.20, 0.10])
    assert torch.equal(anchor.filter_bank.theta[0], expected)
    assert torch.equal(repaired.filter_bank.theta[1:], anchor.filter_bank.theta[1:])
    assert repaired.filter_bank.theta.requires_grad and anchor.filter_bank.theta.requires_grad
    mismatches = [k for k, v in repaired.state_dict().items()
                  if not torch.equal(v, anchor.state_dict()[k])]
    assert mismatches == ["filter_bank.theta"]


def test_single_shared_filter_and_router_for_all_graph_views():
    model = make_model("M_repaired")
    assert len(model.filter_bank.experts) == 4
    assert len(model.router.net) == 4
    assert len({id(module) for module in model.filter_bank.experts}) == 4
    assert "filter_bank" in dict(model.named_children())
    assert not any(name.startswith("private_") or name.startswith("collaborative_")
                   for name, _ in model.named_parameters())


def test_signed_theta_support_and_truncated_normalization():
    model = make_model("M_repaired")
    with torch.no_grad():
        model.filter_bank.theta[0].copy_(torch.tensor([0.4, -0.3, 0.2, -0.1]))
    coeff = model.filter_bank.coefficients(1, torch.float32)
    assert torch.allclose(coeff[0, :2], torch.tensor([0.4 / 0.7, -0.3 / 0.7]))
    assert torch.equal(coeff[:, 2:], torch.zeros(4, 2))
    single_s = make_model("S_repaired").single_filter
    assert torch.equal(single_s.coefficients(3, torch.float32), torch.full((4,), 0.25))
    assert torch.equal(single_s.coefficients(1, torch.float32), torch.tensor([0.5, 0.5, 0.0, 0.0]))
    basis = [torch.randn(12, 256) for _ in range(4)]
    support = torch.tensor([True] * 9 + [False] * 3)
    model.eval()
    response, _, _, _ = model._filter(basis, support, basis[0], 0, 0, torch.zeros_like(basis[0]), collect=False)
    assert torch.equal(response[~support], torch.zeros_like(response[~support]))


def test_paired_rng_replays_dropout_and_restores_external_stream():
    torch.manual_seed(7)
    state = mrcge_v12.capture_rng_state(torch.device("cpu"))
    first = F.dropout(torch.ones(512), p=0.35, training=True)
    after_first = torch.random.get_rng_state().clone()
    with mrcge_v12.replay_rng_state(state, torch.device("cpu")):
        second = F.dropout(torch.ones(512), p=0.35, training=True)
    assert torch.equal(first, second)
    assert torch.equal(after_first, torch.random.get_rng_state())
    third = F.dropout(torch.ones(512), p=0.35, training=True)
    assert not torch.equal(first, third)


def test_test_label_masking_and_private_ce_are_train_only():
    data = SimpleNamespace(
        y=torch.tensor([0, 1, 2, 0, 1, 2, 1, 2]),
        train_idx=torch.tensor([0, 1, 2]), val_idx=torch.tensor([3, 4]),
        test_idx=torch.tensor([5, 6, 7]),
    )
    from scripts.run_mrcge_v12_screen import mask_test_labels
    safe = mask_test_labels(data)
    assert torch.equal(safe.y[safe.test_idx], torch.full((3,), -100))
    assert torch.equal(safe.y[safe.train_idx], data.y[data.train_idx])
    assert torch.equal(safe.y[safe.val_idx], data.y[data.val_idx])
    info = {"unimodal_logits": {"text": torch.randn(8, 3, requires_grad=True),
                                "visual": torch.randn(8, 3, requires_grad=True)}}
    main_logits = torch.randn(8, 3, requires_grad=True)
    main_ce = F.cross_entropy(main_logits[data.train_idx], safe.y[data.train_idx])
    total, private_ce = _compose_full_graph_loss(
        model=SimpleNamespace(supports_unimodal_aux_loss=True), info=info,
        main_ce=main_ce, aux_loss=main_ce.new_zeros(()), aux_weight=1.0,
        unimodal_weight=0.3, labels=safe.y, train_idx=data.train_idx,
    )
    expected = 0.5 * sum(F.cross_entropy(info["unimodal_logits"][m][data.train_idx], safe.y[data.train_idx])
                         for m in ("text", "visual"))
    assert torch.allclose(private_ce, expected)
    assert torch.allclose(total, main_ce + 0.3 * expected)


@pytest.mark.parametrize("condition", ["F_sinit", "M_repaired"])
def test_private_collaborative_zero_cross_pairs_dropout(condition):
    model = make_model(condition).train()
    basis = [torch.randn(12, 256) for _ in range(4 if condition == "M_repaired" else 2)]
    support = torch.tensor([True] * 10 + [False] * 2)
    current = torch.randn(12, 256)
    zero = torch.zeros_like(current)
    start = mrcge_v12.capture_rng_state(torch.device("cpu"))
    private, _, _, _ = model._filter(basis, support, current, 0, 0, zero, collect=False)
    after_private = torch.random.get_rng_state().clone()
    with mrcge_v12.replay_rng_state(start, torch.device("cpu")):
        collaborative, _, _, _ = model._filter(basis, support, current, 0, 0, zero, collect=False)
    assert torch.equal(private, collaborative)
    assert torch.equal(after_private, torch.random.get_rng_state())
    private_state = torch.randn(12, 256)
    paired = private_state + 0.25 * (collaborative - private)
    assert torch.equal(paired, private_state)


def test_m_checkpoint_matches_old_filter_outputs_loss_gradients_and_rng():
    old = make_model("M_repaired", legacy=True).train()
    new = make_model("M_repaired").train()
    new.load_state_dict(old.state_dict(), strict=True)
    basis0 = [torch.randn(32, 256) for _ in range(4)]
    support = torch.tensor([True] * 28 + [False] * 4)
    current = torch.randn(32, 256)
    cross = torch.randn(32, 256) * 0.01

    old_basis = [v.clone().requires_grad_() for v in basis0]
    new_basis = [v.clone().requires_grad_() for v in basis0]
    torch.manual_seed(999)
    old_response, old_balance = m_filter_reference(old, old_basis, support, current, cross)
    old_loss = old_response.square().mean() + 0.03 * old_balance
    old_loss.backward()
    old_rng_after_forward_backward = torch.random.get_rng_state().clone()

    torch.manual_seed(999)
    new_response, _, new_balance, _ = new._filter(new_basis, support, current, 0, 0, cross, collect=False)
    new_loss = new_response.square().mean() + 0.03 * new_balance
    new_rng_after_forward = torch.random.get_rng_state().clone()
    new_loss.backward()
    new_rng_after_backward = torch.random.get_rng_state().clone()

    assert torch.allclose(new_response, old_response, atol=1.0e-5, rtol=1.0e-4)
    assert torch.allclose(new_loss, old_loss, atol=1.0e-6, rtol=1.0e-5)
    assert torch.equal(new_rng_after_forward, torch.random.get_rng_state())
    assert torch.equal(new_rng_after_forward, new_rng_after_backward)
    assert torch.isfinite(new_loss)
    for name, parameter in old.named_parameters():
        other = dict(new.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None, name
        else:
            assert other.grad is not None, name
            assert torch.allclose(other.grad, parameter.grad, atol=2.0e-5, rtol=2.0e-4), name
    assert torch.equal(old_rng_after_forward_backward, new_rng_after_backward)


def test_small_cross_has_no_independent_dropout_noise_floor():
    model = make_model("M_repaired").train()
    basis = [torch.randn(24, 256) for _ in range(4)]
    support = torch.ones(24, dtype=torch.bool)
    current = torch.randn(24, 256)
    zero = torch.zeros_like(current)
    perturb = torch.full_like(current, 1.0e-7)
    start = torch.random.get_rng_state().clone()
    base, _, _, _ = model._filter(basis, support, current, 0, 0, zero, collect=False)
    torch.random.set_rng_state(start)
    slightly_crossed = list(basis)
    slightly_crossed[1] = slightly_crossed[1] + perturb
    changed, _, _, _ = model._filter(slightly_crossed, support, current, 0, 0, perturb, collect=False)
    assert float((changed - base).abs().max()) < 1.0e-3


def test_full_v11_v12_eval_function_matches_on_identical_small_graph():
    old = make_model("M_repaired", legacy=True, seed=27).eval()
    new = make_model("M_repaired", seed=81).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(12, 12)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]])
    with torch.no_grad():
        out_old = old(x, edges, return_details=False)[0]
        out_new = new(x, edges, return_details=False)[0]
    assert torch.allclose(out_old, out_new, atol=1.0e-5, rtol=1.0e-4)


@pytest.mark.parametrize("condition", list(mrcge_v12.CONDITIONS))
def test_small_full_graph_forward_backward_save_reload_and_validation_rng(condition):
    model = make_model(condition, seed=81)
    x = torch.randn(12, 12)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]])
    head = torch.nn.Linear(256, 3)
    model.train()
    z, _, _, aux, info = model(x, edges, return_details=False)
    labels = torch.tensor([0, 1, 2] * 4)
    loss = F.cross_entropy(head(z[:9]), labels[:9]) + aux + 0.3 * sum(
        F.cross_entropy(info["unimodal_logits"][m][:9], labels[:9]) for m in ("text", "visual")
    ) / 2
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)

    state = copy.deepcopy(model.state_dict())
    clone = make_model(condition, seed=13)
    clone.load_state_dict(state, strict=True)
    model.eval(); clone.eval()
    cpu_rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        out_a = model(x, edges, return_details=True)[0]
        out_b = clone(x, edges, return_details=True)[0]
    assert torch.equal(out_a, out_b)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())


def test_m_anchor_and_repaired_state_dict_compatibility_with_v11():
    old = make_model("M_repaired", legacy=True, seed=19)
    new = make_model("M_repaired", seed=19)
    result = new.load_state_dict(old.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert old.state_dict().keys() == new.state_dict().keys()
    old_trainable = {k for k, p in old.named_parameters()}
    new_trainable = {k for k, p in new.named_parameters()}
    assert old_trainable == new_trainable


def test_late_none_selector_and_fusion_output_contract():
    model = make_model("M_repaired").eval()
    model.same_node_enabled = False
    n = 12
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    model.semantic_visual_edge_index = empty_edges
    model.semantic_visual_weight = torch.empty(0)
    u = torch.randn(n, 256)
    out, detail = model._late("text", 0, "visual", u, u, u, collect=True)
    assert torch.equal(out, u)
    assert detail["p_accept_mean"] == 0.0
    assert detail["selector_share"] == [0.0, 0.0, 1.0]
    assert model.out_dim == 256
