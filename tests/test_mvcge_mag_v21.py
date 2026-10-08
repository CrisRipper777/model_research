from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mvcge_mag_v2 import Model as V2Model
from src.models.mvcge_mag_v21 import Model


VARIANTS = (
    "U0_modality_static",
    "U1_node_selection",
    "U2_node_selection_strength",
    "U3_collaborative",
)


def _cfg(variant: str, *, dropout: float = 0.0, trajectory_order: int = 4):
    return OmegaConf.create(
        {
            "model": {
                "name": "mvcge_mag_v21",
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
                "strength_init": -1.15,
                "balance_weight": 0.01,
                "eps": 1.0e-8,
            }
        }
    )


def _v2_cfg(variant: str):
    cfg = _cfg("U2_node_selection_strength", dropout=0.0)
    cfg.model.name = "mvcge_mag_v2"
    cfg.model.variant = variant
    cfg.model.num_layers = 4
    cfg.model.base_gate_init = -2.0
    cfg.model.direct_strength_init = -1.15
    cfg.model.residual_strength_init = -2.0
    cfg.model.residual_max = 0.25
    return cfg


def _data_info():
    return {"input_dim": 11, "text_dim": 6, "visual_dim": 5, "num_nodes": 6, "num_classes": 3}


def _path_graph(*, loops: bool = False):
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    if loops:
        loops_tensor = torch.tensor([[0, 2, 5], [0, 2, 5]], dtype=torch.long)
        edges = torch.cat([edges, loops_tensor], dim=1)
    return edges


def test_all_variants_forward_backward_and_diagnostics_are_finite():
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Model(_cfg(variant), _data_info()).train()
        z, _, _, aux, info = model(x, _path_graph(), return_details=True)
        assert z.shape == (6, 256)
        assert torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        assert torch.isfinite(info["balance_loss"])
        for modality in ("text", "visual"):
            details = info["details"][modality]
            for key in (
                "prior", "basis", "alpha", "expert_inputs", "expert_outputs", "mixture",
                "scaled_mixture", "selection_logits", "dense_probs", "route_weights",
                "strength", "importance", "selection_share",
            ):
                assert torch.isfinite(details[key]).all(), (variant, modality, key)
        (z.square().mean() + aux).backward()
        grads = [
            p.grad for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_same_seed_initialization_matches_all_shared_modules_and_classifier():
    snapshots = {}
    for variant in VARIANTS:
        torch.manual_seed(123)
        model = Model(_cfg(variant), _data_info())
        classifier = torch.nn.Linear(model.out_dim, 3)
        snapshots[variant] = {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "classifier": {
                key: value.detach().clone() for key, value in classifier.state_dict().items()
            },
        }
    reference = snapshots[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        assert reference["model"].keys() == snapshots[variant]["model"].keys()
        for key, value in reference["model"].items():
            assert torch.equal(value, snapshots[variant]["model"][key]), (variant, key)
        for key, value in reference["classifier"].items():
            assert torch.equal(value, snapshots[variant]["classifier"][key])


def test_u2_regresses_to_v2_m1_direct_moe_at_same_seed():
    seed = 913
    torch.manual_seed(seed)
    old = V2Model(_v2_cfg("M1_direct_moe"), _data_info()).eval()
    torch.manual_seed(seed)
    new = Model(_cfg("U2_node_selection_strength"), _data_info()).eval()
    assert old.state_dict().keys() >= new.state_dict().keys()
    for key, value in new.state_dict().items():
        assert torch.equal(value, old.state_dict()[key]), key

    torch.manual_seed(71)
    x = torch.randn(6, 11)
    graph = _path_graph()
    with torch.no_grad():
        z_old, _, _, aux_old, info_old = old(x, graph, return_details=True)
        z_new, _, _, aux_new, info_new = new(x, graph, return_details=True)
    assert torch.allclose(z_old, z_new, atol=1e-7, rtol=0.0)
    assert torch.allclose(aux_old, aux_new, atol=1e-7, rtol=0.0)
    for modality in ("text", "visual"):
        before = info_old["details"][modality]
        after = info_new["details"][modality]
        for key in (
            "prior", "basis", "alpha", "expert_outputs", "selection_logits",
            "route_weights", "strength", "output", "mixture", "scaled_mixture",
        ):
            assert torch.allclose(before[key], after[key], atol=1e-7, rtol=0.0), (
                modality,
                key,
                (before[key] - after[key]).abs().max().item(),
            )


def test_isolated_nodes_have_zero_basis_experts_and_correction():
    torch.manual_seed(31)
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, _path_graph(), return_details=True)
        for modality in ("text", "visual"):
            details = info["details"][modality]
            assert torch.count_nonzero(details["basis"][:, 5]) == 0
            assert torch.count_nonzero(details["expert_inputs"][:, 5]) == 0
            assert torch.count_nonzero(details["expert_outputs"][:, 5]) == 0
            assert torch.count_nonzero(details["mixture"][5]) == 0
            assert torch.count_nonzero(details["scaled_mixture"][5]) == 0
            assert torch.equal(details["output"][5], details["prior"][5])


def test_physical_operator_removes_input_loops_and_adds_none():
    src, dst, _, active, loops = Model._normalized_operator(
        _path_graph(loops=True), 6, torch.float32
    )
    assert loops == 3
    assert not torch.any(src == dst)
    assert not torch.any(src == dst)
    assert not active[5]


def test_hadamard_profiles_are_initialized_as_requested():
    model = Model(_cfg("U2_node_selection_strength"), _data_info())
    expected = 0.5 * torch.tensor(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        dtype=torch.float32,
    )
    assert torch.equal(model.alpha_raw.detach(), expected)
    effective = model._effective_alpha(model.alpha_raw, model.eps)
    assert torch.allclose(effective, expected, atol=1e-7, rtol=0.0)


def test_top2_has_exactly_two_nonzero_weights():
    model = Model(_cfg("U3_collaborative"), _data_info()).eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert torch.equal(route["selected_mask"].sum(-1), torch.full((6,), 2))
        assert torch.equal((route["route_weights"] > 0).sum(-1), torch.full((6,), 2))
        assert torch.allclose(route["route_weights"].sum(-1), torch.ones(6), atol=1e-7)


def test_u0_static_context_is_constant_even_with_encoder_dropout():
    torch.manual_seed(97)
    model = Model(_cfg("U0_modality_static", dropout=0.2), _data_info()).train()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert route["router_input"].shape == (1, 272)
        assert torch.equal(route["selection_logits"], route["selection_logits"][:1].expand(6, -1))
        assert torch.equal(route["route_weights"], route["route_weights"][:1].expand(6, -1))
        assert torch.equal(route["strength"], route["strength"][:1].expand(6))
        assert torch.count_nonzero(route["router_blocks"]["cross"]) == 0


def test_u1_routes_by_node_but_uses_one_static_strength_row():
    torch.manual_seed(17)
    model = Model(_cfg("U1_node_selection", dropout=0.2), _data_info()).train()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert route["router_input"].shape == (6, 272)
        assert route["selection_logits"].std(dim=0).max() > 0
        assert torch.equal(route["strength"], route["strength"][:1].expand(6))
        assert torch.count_nonzero(route["router_blocks"]["cross"]) == 0


def test_u2_can_vary_selection_and_strength_by_node():
    torch.manual_seed(27)
    model = Model(_cfg("U2_node_selection_strength", dropout=0.2), _data_info()).train()
    with torch.no_grad():
        model.strength_head.weight.normal_(mean=0.0, std=0.1)
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert route["selection_logits"].std(dim=0).max() > 0
        assert route["strength"].std() > 0


def test_cross_context_blocks_match_variant_definitions():
    torch.manual_seed(73)
    x = torch.randn(6, 11)
    outputs = {}
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, _path_graph(), return_details=True)
        outputs[variant] = (model, info)
    model, info = outputs["U3_collaborative"]
    expected = 0.5 * (
        info["details"]["text"]["router_blocks"]["ego"]
        + info["details"]["visual"]["router_blocks"]["ego"]
    )
    for modality in ("text", "visual"):
        assert torch.allclose(
            info["details"][modality]["router_blocks"]["cross"], expected, atol=1e-7
        )
    for variant in ("U0_modality_static", "U1_node_selection"):
        for modality in ("text", "visual"):
            cross = outputs[variant][1]["details"][modality]["router_blocks"]["cross"]
            assert torch.count_nonzero(cross) == 0


def test_load_balance_matches_v2_active_only_formula_for_all_variants():
    torch.manual_seed(101)
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, aux, info = model(x, _path_graph(), return_details=True)
        per_modality = []
        for modality in ("text", "visual"):
            route = info["router"][modality]
            active = info["active_nodes"]
            dense = route["dense_probs"][active].mean(dim=0)
            share = route["selected_mask"][active].float().mean(dim=0) / 2
            expected = model.num_experts * (dense * share).sum()
            assert torch.allclose(route["importance"], dense, atol=1e-7)
            assert torch.allclose(route["selection_share"], share, atol=1e-7)
            assert torch.allclose(share.sum(), torch.tensor(1.0), atol=1e-7)
            per_modality.append(expected)
        expected_aux = 0.01 * 0.5 * sum(per_modality)
        assert torch.allclose(aux, expected_aux, atol=1e-7)


def test_all_variants_have_same_initial_strength_sigmoid_minus_1_15():
    expected = 1.0 / (1.0 + math.exp(1.15))
    torch.manual_seed(55)
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, _path_graph(), return_details=True)
        assert torch.count_nonzero(model.strength_head.weight) == 0
        assert math.isclose(float(model.strength_head.bias.item()), -1.15, abs_tol=1e-7)
        for modality in ("text", "visual"):
            strength = info["router"][modality]["strength"]
            assert torch.allclose(strength, torch.full_like(strength, expected), atol=1e-7)


def test_context_encoder_call_count_and_static_row_shapes():
    for variant, expected_calls in (
        ("U0_modality_static", 2),
        ("U1_node_selection", 4),
        ("U2_node_selection_strength", 2),
        ("U3_collaborative", 2),
    ):
        model = Model(_cfg(variant, dropout=0.2), _data_info()).train()
        shapes = []
        hook = model.context_encoder.register_forward_pre_hook(
            lambda _module, args: shapes.append(tuple(args[0].shape))
        )
        model(torch.randn(6, 11), _path_graph())
        hook.remove()
        assert len(shapes) == expected_calls, (variant, shapes)
        if variant == "U0_modality_static":
            assert shapes == [(1, 272), (1, 272)]
        if variant == "U1_node_selection":
            assert shapes == [(6, 272), (1, 272), (6, 272), (1, 272)]


def test_trajectory_order_is_read_and_fixed_to_four():
    model = Model(_cfg("U2_node_selection_strength", trajectory_order=4), _data_info())
    assert model.trajectory_order == 4
    with pytest.raises(ValueError, match="trajectory_order=4"):
        Model(_cfg("U2_node_selection_strength", trajectory_order=3), _data_info())
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    assert info["details"]["text"]["basis"].shape[0] == 4

