from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mvcge_mag_v22 import Model as V22Model
from src.models.mvcge_mag_v22b import Model


VARIANTS = ("D0_static", "D1_free_node", "D2_struct_free", "D3_struct_residual")
OLD_VARIANTS = {"D0_static": "R0_modality_static", "D1_free_node": "R1_free_node"}
DATA_INFO = {"input_dim": 11, "text_dim": 6, "visual_dim": 5, "num_nodes": 6, "num_classes": 3}


def _cfg(variant: str, *, dropout: float = 0.0, trajectory_order: int = 4, name: str = "mvcge_mag_v22b"):
    return OmegaConf.create(
        {
            "model": {
                "name": name,
                "variant": variant,
                "hidden_dim": 256,
                "dropout": dropout,
                "trajectory_order": trajectory_order,
                "num_experts": 4,
                "top_k": 2,
                "expert_bottleneck": 64,
                "expert_feature_scale": 0.1,
                "router_dim": 64,
                "router_hidden_dim": 128,
                "modality_embed_dim": 16,
                "direct_strength_init": -1.15,
                "evidence_scalar_dim": 11,
                "expert_key_embed_dim": 16,
                "compatibility_dim": 64,
                "node_residual_init": 0.10,
                "compatibility_residual_init": 0.10,
                "balance_weight": 0.01,
                "eps": 1.0e-8,
            }
        }
    )


def _path_graph(*, loops: bool = False):
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    if loops:
        edges = torch.cat([edges, torch.tensor([[0, 2, 5], [0, 2, 5]])], dim=1)
    return edges


def _assert_equal(a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 1.0e-7):
    assert torch.allclose(a, b, atol=atol, rtol=0.0), (
        name,
        float((a - b).abs().max()) if a.numel() else 0.0,
    )


def _same_init_pair(v22_variant: str, v22b_variant: str, seed: int = 913):
    torch.manual_seed(seed)
    old = V22Model(_cfg(v22_variant, name="mvcge_mag_v22"), DATA_INFO).eval()
    torch.manual_seed(seed)
    new = Model(_cfg(v22b_variant), DATA_INFO).eval()
    old_state, new_state = old.state_dict(), new.state_dict()
    for key, value in old_state.items():
        assert key in new_state, key
        assert torch.equal(value, new_state[key]), key
    return old, new


def test_all_four_variants_forward_backward_finite():
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Model(_cfg(variant), DATA_INFO).train()
        z, _, _, aux, info = model(x, _path_graph(), return_details=True)
        assert z.shape == (6, 256)
        assert torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        for modality in ("text", "visual"):
            item = info["details"][modality]
            for key in ("prior", "basis", "alpha", "expert_outputs", "mixture", "scaled_mixture", "selection_logits", "route_weights", "strength"):
                assert torch.isfinite(item[key]).all(), (variant, modality, key)
        (z.square().mean() + aux).backward()
        active_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert active_grads and all(torch.isfinite(g).all() for g in active_grads)


def test_all_variants_have_exact_same_seed_shared_state_and_classifier_rng():
    snapshots = {}
    for variant in VARIANTS:
        torch.manual_seed(123)
        model = Model(_cfg(variant), DATA_INFO)
        classifier = torch.nn.Linear(model.out_dim, 3)
        snapshots[variant] = (
            {k: v.detach().clone() for k, v in model.state_dict().items()},
            {k: v.detach().clone() for k, v in classifier.state_dict().items()},
        )
    reference = snapshots[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        assert reference[0].keys() == snapshots[variant][0].keys()
        for key, value in reference[0].items():
            assert torch.equal(value, snapshots[variant][0][key]), (variant, key)
        for key, value in reference[1].items():
            assert torch.equal(value, snapshots[variant][1][key]), (variant, key)


@pytest.mark.parametrize("variant", VARIANTS)
def test_downstream_classifier_rng_matches_v22(variant):
    torch.manual_seed(887)
    old = V22Model(_cfg("R0_modality_static", name="mvcge_mag_v22"), DATA_INFO)
    old_classifier = torch.nn.Linear(old.out_dim, 3)
    torch.manual_seed(887)
    new = Model(_cfg(variant), DATA_INFO)
    new_classifier = torch.nn.Linear(new.out_dim, 3)
    for key, value in old_classifier.state_dict().items():
        assert torch.equal(value, new_classifier.state_dict()[key]), (variant, key)


@pytest.mark.parametrize("variant", ("D0_static", "D1_free_node"))
def test_initial_full_forward_exactly_regresses_to_v22(variant):
    old_variant = OLD_VARIANTS[variant]
    old, new = _same_init_pair(old_variant, variant)
    torch.manual_seed(71)
    x, graph = torch.randn(6, 11), _path_graph()
    with torch.no_grad():
        z0, _, _, aux0, info0 = old(x, graph, return_details=True)
        z1, _, _, aux1, info1 = new(x, graph, return_details=True)
    _assert_equal(z0, z1, "fused z")
    _assert_equal(aux0, aux1, "aux_loss")
    for modality in ("text", "visual"):
        a, b = info0["details"][modality], info1["details"][modality]
        for key in ("prior", "basis", "alpha", "expert_outputs", "selection_logits", "route_weights", "strength", "output", "mixture", "scaled_mixture"):
            _assert_equal(a[key], b[key], f"{modality}.{key}")


def test_direct_strength_initial_values_match_old_head_bias():
    model = Model(_cfg("D0_static"), DATA_INFO)
    expected = torch.sigmoid(torch.tensor(-1.15))
    assert torch.equal(model.direct_strength_raw, torch.full((2,), -1.15))
    assert torch.equal(model.strength_head.weight, torch.zeros_like(model.strength_head.weight))
    assert torch.equal(model.strength_head.bias, torch.full_like(model.strength_head.bias, -1.15))
    assert torch.equal(torch.sigmoid(model.direct_strength_raw), expected.expand(2))


@pytest.mark.parametrize("variant", VARIANTS)
def test_strength_is_node_static_for_every_variant(variant):
    model = Model(_cfg(variant, dropout=0.2), DATA_INFO).eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        strength = info["router"][modality]["strength"]
        assert torch.equal(strength, strength[:1].expand_as(strength))
        assert float((strength - strength[:1]).abs().max()) == 0.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_strength_is_independent_of_input_and_graph(variant):
    model = Model(_cfg(variant), DATA_INFO).eval()
    first = model(torch.randn(6, 11), _path_graph(), return_details=True)[4]["router"]
    second = model(torch.randn(6, 11) * 7.0, torch.tensor([[0, 1], [1, 0]]), return_details=True)[4]["router"]
    for modality in ("text", "visual"):
        assert torch.equal(first[modality]["strength"], second[modality]["strength"])
        assert torch.equal(first[modality]["strength_raw"], second[modality]["strength_raw"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_strength_gradient_is_isolated_from_all_router_modules(variant):
    model = Model(_cfg(variant), DATA_INFO)
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    strength_sum = info["router"]["text"]["strength"].sum() + info["router"]["visual"]["strength"].sum()
    strength_sum.backward()
    assert model.direct_strength_raw.grad is not None
    assert torch.isfinite(model.direct_strength_raw.grad).all()
    assert torch.count_nonzero(model.direct_strength_raw.grad)
    for module in (model.context_encoder, model.selection_head, model.strength_head):
        assert all(parameter.grad is None for parameter in module.parameters())


def test_autograd_grad_strength_has_no_router_dependency():
    model = Model(_cfg("D3_struct_residual"), DATA_INFO)
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    strength_sum = info["router"]["text"]["strength"].sum()
    router_parameters = tuple(model.context_encoder.parameters()) + tuple(model.selection_head.parameters())
    gradients = torch.autograd.grad(
        strength_sum,
        (model.direct_strength_raw, *router_parameters),
        allow_unused=True,
    )
    direct_gradient, *router_gradients = gradients
    assert direct_gradient is not None and torch.isfinite(direct_gradient).all()
    assert torch.count_nonzero(direct_gradient)
    assert all(g is None or not torch.count_nonzero(g) for g in router_gradients)


@pytest.mark.parametrize("variant", VARIANTS)
def test_task_output_can_train_direct_strength(variant):
    torch.manual_seed(68)
    model = Model(_cfg(variant), DATA_INFO)
    z = model(torch.randn(6, 11), _path_graph())[0]
    probe = torch.linspace(-1.0, 1.0, z.numel()).reshape_as(z)
    (z * probe).sum().backward()
    assert model.direct_strength_raw.grad is not None
    assert torch.isfinite(model.direct_strength_raw.grad).all()
    assert torch.count_nonzero(model.direct_strength_raw.grad)


@pytest.mark.parametrize("variant", VARIANTS)
def test_legacy_strength_and_compatibility_modules_are_frozen_and_unused(variant):
    model = Model(_cfg(variant), DATA_INFO)
    assert all(not p.requires_grad for p in model.strength_head.parameters())
    assert model.direct_strength_raw.requires_grad
    compat_params = [model.expert_key_embedding, *model.expert_query_proj.parameters(), *model.expert_key_proj.parameters(), model.compatibility_residual_raw]
    assert all(not p.requires_grad for p in compat_params)
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert "compatibility" not in route
        assert "compatibility_logits" not in route


def test_variant_specific_trainable_parameter_activity():
    for variant in VARIANTS:
        model = Model(_cfg(variant), DATA_INFO)
        evidence_trainable = variant in {"D2_struct_free", "D3_struct_residual"}
        residual_trainable = variant == "D3_struct_residual"
        assert all(p.requires_grad == evidence_trainable for p in model.evidence_scalar_projector.parameters())
        assert all(p.requires_grad == evidence_trainable for p in model.evidence_norm.parameters())
        assert model.node_residual_raw.requires_grad == residual_trainable
        assert model.direct_strength_raw.requires_grad
        assert all(not p.requires_grad for p in model.strength_head.parameters())


def test_d0_static_logits_and_routes_are_node_constant():
    model = Model(_cfg("D0_static", dropout=0.2), DATA_INFO).train()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        for key in ("selection_logits", "route_weights", "strength"):
            assert torch.equal(route[key], route[key][:1].expand_as(route[key]))
        assert route["router_input"].shape == (1, 272)


def test_d1_selection_path_matches_v22_r1():
    old, new = _same_init_pair("R1_free_node", "D1_free_node")
    x, graph = torch.randn(6, 11), _path_graph()
    with torch.no_grad():
        old_info = old(x, graph, return_details=True)[4]
        new_info = new(x, graph, return_details=True)[4]
    for modality in ("text", "visual"):
        _assert_equal(new_info["router"][modality]["selection_logits"], old_info["router"][modality]["selection_logits"], modality)


def test_d2_uses_structure_free_logits_without_eta_blending():
    model = Model(_cfg("D2_struct_free"), DATA_INFO).eval()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert torch.equal(route["selection_logits"], route["structure_free_logits"])
        assert float(route["eta"]) == 0.0


def test_d3_uses_exact_static_centered_residual_logits_and_eta_init():
    model = Model(_cfg("D3_struct_residual"), DATA_INFO).eval()
    assert torch.allclose(torch.sigmoid(model.node_residual_raw), torch.full((2,), 0.10), atol=1e-7, rtol=0)
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for i, modality in enumerate(("text", "visual")):
        route = info["router"][modality]
        expected = route["static_logits"] + torch.sigmoid(model.node_residual_raw[i]) * (route["structure_free_logits"] - route["static_logits"])
        _assert_equal(route["selection_logits"], expected, modality)
        assert torch.allclose(route["eta"], torch.tensor(0.10), atol=1e-7, rtol=0)


def test_d2_and_d3_share_exact_evidence_and_structure_free_logits_at_initialization():
    x, graph = torch.randn(6, 11), _path_graph()
    results = {}
    for variant in ("D2_struct_free", "D3_struct_residual"):
        torch.manual_seed(19)
        model = Model(_cfg(variant), DATA_INFO).eval()
        results[variant] = model(x, graph, return_details=True)[4]
    for modality in ("text", "visual"):
        a = results["D2_struct_free"]["details"][modality]
        b = results["D3_struct_residual"]["details"][modality]
        for key in ("reliability", "mu", "sigma", "degree_norm", "hop_cos", "hop_disp", "scalar_evidence", "L_rel", "EVID"):
            assert torch.equal(a[key], b[key]), (modality, key)
        assert torch.equal(results["D2_struct_free"]["router"][modality]["structure_free_logits"], results["D3_struct_residual"]["router"][modality]["structure_free_logits"])


def test_structural_evidence_exactly_matches_v22_definition():
    torch.manual_seed(101)
    old = V22Model(_cfg("R2_structure_grounded", name="mvcge_mag_v22"), DATA_INFO).eval()
    torch.manual_seed(101)
    new = Model(_cfg("D2_struct_free"), DATA_INFO).eval()
    x, graph = torch.randn(6, 11), _path_graph()
    with torch.no_grad():
        old_info = old(x, graph, return_details=True)[4]
        new_info = new(x, graph, return_details=True)[4]
    for modality in ("text", "visual"):
        for key in ("reliability", "mu", "sigma", "degree_norm", "hop_cos", "hop_disp", "scalar_evidence", "L_rel", "EVID"):
            _assert_equal(old_info["details"][modality][key], new_info["details"][modality][key], f"{modality}.{key}")


def test_top2_has_exactly_two_nonzero_experts_and_balance_is_unchanged():
    model = Model(_cfg("D2_struct_free"), DATA_INFO).eval()
    _, _, _, aux, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    active, balances = info["active_nodes"], []
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert torch.equal((route["route_weights"] > 0).sum(-1), torch.full((6,), 2))
        dense = route["dense_probs"][active].mean(0)
        share = route["selected_mask"][active].float().mean(0) / 2
        balance = 4 * (dense * share).sum()
        _assert_equal(route["importance"], dense, modality + ".importance")
        _assert_equal(route["selection_share"], share, modality + ".share")
        _assert_equal(route["balance"], balance, modality + ".balance")
        balances.append(balance)
    _assert_equal(aux, 0.01 * 0.5 * (balances[0] + balances[1]), "aux")


def test_isolated_nodes_have_zero_structural_correction_and_output_equals_prior():
    model = Model(_cfg("D2_struct_free"), DATA_INFO).eval()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        item = info["details"][modality]
        assert torch.count_nonzero(item["basis"][:, 5]) == 0
        assert torch.count_nonzero(item["expert_outputs"][:, 5]) == 0
        assert torch.count_nonzero(item["scaled_mixture"][5]) == 0
        assert torch.equal(item["output"][5], item["prior"][5])
        for key in ("mu", "sigma", "degree_norm", "hop_cos", "hop_disp", "EVID"):
            assert torch.count_nonzero(item[key][5]) == 0, (modality, key)


def test_physical_operator_removes_loops_adds_none_and_active_is_degree_positive():
    src, dst, _, active, loops = Model._normalized_operator(_path_graph(loops=True), 6, torch.float32)
    assert loops == 3
    assert not torch.any(src == dst)
    assert torch.equal(active, torch.tensor([True, True, True, True, True, False]))


def test_trajectory_order_is_configured_and_fixed_to_four():
    model = Model(_cfg("D2_struct_free", trajectory_order=4), DATA_INFO)
    assert model.trajectory_order == 4
    with pytest.raises(ValueError, match="trajectory_order=4"):
        Model(_cfg("D2_struct_free", trajectory_order=3), DATA_INFO)
    result = model(torch.randn(6, 11), _path_graph(), return_details=True)[4]
    assert result["details"]["text"]["basis"].shape[0] == 4
