from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mvcge_mag_v21 import Model as V21Model
from src.models.mvcge_mag_v22 import Model


VARIANTS = (
    "R0_modality_static",
    "R1_free_node",
    "R2_structure_grounded",
    "R3_expert_compatibility",
)
OLD_VARIANT = {
    "R0_modality_static": "U0_modality_static",
    "R1_free_node": "U1_node_selection",
}


def _cfg(variant: str, *, dropout: float = 0.0, trajectory_order: int = 4):
    return OmegaConf.create(
        {
            "model": {
                "name": "mvcge_mag_v22",
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


def _v21_cfg(variant: str):
    cfg = _cfg("R0_modality_static")
    cfg.model.name = "mvcge_mag_v21"
    cfg.model.variant = variant
    del cfg.model.evidence_scalar_dim
    del cfg.model.expert_key_embed_dim
    del cfg.model.compatibility_dim
    del cfg.model.node_residual_init
    del cfg.model.compatibility_residual_init
    return cfg


def _data_info():
    return {"input_dim": 11, "text_dim": 6, "visual_dim": 5, "num_nodes": 6, "num_classes": 3}


def _path_graph(*, loops: bool = False):
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    if loops:
        edges = torch.cat(
            [edges, torch.tensor([[0, 2, 5], [0, 2, 5]], dtype=torch.long)], dim=1
        )
    return edges


def _assert_same(left: torch.Tensor, right: torch.Tensor, name: str, atol=1.0e-7):
    assert torch.allclose(left, right, atol=atol, rtol=0.0), (
        name,
        (left - right).abs().max().item() if left.numel() else 0.0,
    )


def test_all_variants_forward_backward_and_diagnostics_are_finite():
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Model(_cfg(variant), _data_info()).train()
        z, _, _, aux, info = model(x, _path_graph(), return_details=True)
        assert z.shape == (6, 256)
        assert torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        for modality in ("text", "visual"):
            details = info["details"][modality]
            for key in (
                "prior", "basis", "alpha", "expert_inputs", "expert_outputs", "mixture",
                "scaled_mixture", "selection_logits", "dense_probs", "route_weights",
                "strength", "importance", "selection_share", "EVID", "scalar_evidence",
            ):
                assert torch.isfinite(details[key]).all(), (variant, modality, key)
        (z.square().mean() + aux).backward()
        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_same_seed_shared_and_new_modules_match_across_variants():
    snapshots = {}
    for variant in VARIANTS:
        torch.manual_seed(123)
        model = Model(_cfg(variant), _data_info())
        classifier = torch.nn.Linear(model.out_dim, 3)
        snapshots[variant] = {
            "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
            "classifier": {key: value.detach().clone() for key, value in classifier.state_dict().items()},
        }
    reference = snapshots[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        assert reference["model"].keys() == snapshots[variant]["model"].keys()
        for key, value in reference["model"].items():
            assert torch.equal(value, snapshots[variant]["model"][key]), (variant, key)
        for key, value in reference["classifier"].items():
            assert torch.equal(value, snapshots[variant]["classifier"][key])


@pytest.mark.parametrize("variant", ["R0_modality_static", "R1_free_node"])
def test_old_classifier_initialization_is_rng_identical(variant):
    torch.manual_seed(887)
    old = V21Model(_v21_cfg(OLD_VARIANT[variant]), _data_info())
    old_classifier = torch.nn.Linear(old.out_dim, 3)
    torch.manual_seed(887)
    new = Model(_cfg(variant), _data_info())
    new_classifier = torch.nn.Linear(new.out_dim, 3)
    for key, value in old_classifier.state_dict().items():
        assert torch.equal(value, new_classifier.state_dict()[key]), key


@pytest.mark.parametrize("variant", ["R0_modality_static", "R1_free_node"])
def test_v22_regresses_to_v21_u0_u1_full_forward(variant):
    seed = 913
    old_variant = OLD_VARIANT[variant]
    torch.manual_seed(seed)
    old = V21Model(_v21_cfg(old_variant), _data_info()).eval()
    torch.manual_seed(seed)
    new = Model(_cfg(variant), _data_info()).eval()
    for key, value in new.state_dict().items():
        if key in old.state_dict():
            assert torch.equal(value, old.state_dict()[key]), key

    torch.manual_seed(71)
    x = torch.randn(6, 11)
    graph = _path_graph()
    with torch.no_grad():
        z_old, _, _, aux_old, info_old = old(x, graph, return_details=True)
        z_new, _, _, aux_new, info_new = new(x, graph, return_details=True)
    _assert_same(z_old, z_new, "fused z")
    _assert_same(aux_old, aux_new, "aux_loss")
    for modality in ("text", "visual"):
        before = info_old["details"][modality]
        after = info_new["details"][modality]
        for key in (
            "prior", "basis", "alpha", "expert_outputs", "selection_logits",
            "route_weights", "strength", "output", "mixture", "scaled_mixture",
        ):
            _assert_same(before[key], after[key], f"{modality}.{key}")


def test_isolated_nodes_keep_zero_trajectory_experts_and_evidence():
    torch.manual_seed(31)
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, _path_graph(), return_details=True)
        for modality in ("text", "visual"):
            item = info["details"][modality]
            for key in (
                "basis", "expert_inputs", "expert_outputs", "mixture", "scaled_mixture",
            ):
                assert torch.count_nonzero(item[key][..., 5, :] if item[key].dim() == 3 else item[key][5]) == 0
            assert torch.equal(item["output"][5], item["prior"][5])
            for key in (
                "neighbor_count", "mu", "sigma", "degree_norm", "hop_cos", "hop_disp",
                "scalar_evidence", "L_rel", "EVID",
            ):
                assert torch.count_nonzero(item[key][5]) == 0, (variant, modality, key)


def test_physical_operator_removes_input_loops_and_adds_none():
    src, dst, _, active, loops = Model._normalized_operator(
        _path_graph(loops=True), 6, torch.float32
    )
    assert loops == 3
    assert not torch.any(src == dst)
    assert not active[5]


def test_edge_reliability_is_bounded_and_identical_endpoints_are_one():
    u = torch.tensor([[3.0, 4.0], [3.0, 4.0], [0.0, 1.0]])
    src = torch.tensor([0, 0, 1])
    dst = torch.tensor([1, 2, 2])
    rel = Model._edge_reliability(u, src, dst, 1.0e-8)
    assert torch.all((rel >= 0) & (rel <= 1))
    assert float(rel[0]) == 1.0


def test_scalar_evidence_is_exactly_eleven_features_and_structural_slot_only():
    torch.manual_seed(17)
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, _path_graph(), return_details=True)
        for modality in ("text", "visual"):
            item = info["details"][modality]
            route = info["router"][modality]
            assert item["scalar_evidence"].shape == (6, 11)
            assert route["router_input"].shape[-1] == 272
            assert torch.count_nonzero(route["router_blocks"]["cross"]) == 0
            fourth = route["router_blocks"]["fourth"]
            if variant in ("R0_modality_static", "R1_free_node"):
                assert torch.count_nonzero(fourth) == 0
                assert torch.count_nonzero(item["reliability"]) == 0
            else:
                assert torch.equal(fourth, item["EVID"])
                assert torch.equal(route["router_input"][:, 192:256], item["EVID"])


def test_all_variants_keep_modality_static_strength_in_train_and_eval():
    torch.manual_seed(55)
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant, dropout=0.2), _data_info())
        with torch.no_grad():
            model.strength_head.weight.normal_(mean=0.0, std=0.1)
        for training in (True, False):
            model.train(training)
            with torch.no_grad():
                _, _, _, _, info = model(x, _path_graph(), return_details=True)
            for modality in ("text", "visual"):
                strength = info["router"][modality]["strength"]
                assert torch.equal(strength, strength[:1].expand_as(strength))


def test_r0_routes_are_modality_static():
    torch.manual_seed(97)
    model = Model(_cfg("R0_modality_static", dropout=0.2), _data_info()).train()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert route["router_input"].shape == (1, 272)
        for key in ("selection_logits", "route_weights", "strength"):
            assert torch.equal(route[key], route[key][:1].expand_as(route[key]))
        assert torch.count_nonzero(route["router_blocks"]["fourth"]) == 0


def test_r2_eta_initialization_and_logit_blend():
    torch.manual_seed(51)
    model = Model(_cfg("R2_structure_grounded"), _data_info()).eval()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    expected_eta = 0.1
    for modality_index, modality in enumerate(("text", "visual")):
        route = info["router"][modality]
        assert math.isclose(float(torch.sigmoid(model.node_residual_raw[modality_index])), expected_eta, abs_tol=1e-7)
        expected = route["static_logits"] + route["eta"] * (
            route["structure_free_logits"] - route["static_logits"]
        )
        _assert_same(route["r2_logits"], expected, f"{modality}.r2_blend")
        _assert_same(route["selection_logits"], expected, f"{modality}.final")


@pytest.mark.parametrize("raw,expected", [(-float("inf"), 0.0), (float("inf"), 1.0)])
def test_eta_zero_and_one_edge_cases(raw, expected):
    model = Model(_cfg("R2_structure_grounded"), _data_info()).eval()
    with torch.no_grad():
        model.node_residual_raw.fill_(raw)
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert torch.equal(route["eta"], torch.tensor(expected))
        target = route["static_logits"] if expected == 0 else route["structure_free_logits"]
        _assert_same(route["selection_logits"], target, modality)


def test_r3_compatibility_is_centered_and_added_with_kappa():
    model = Model(_cfg("R3_expert_compatibility"), _data_info()).eval()
    _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for i, modality in enumerate(("text", "visual")):
        route = info["router"][modality]
        compatibility = route["compatibility"]
        assert compatibility.shape == (6, 4)
        assert torch.allclose(compatibility.mean(dim=-1), torch.zeros(6), atol=1e-7)
        assert math.isclose(float(route["kappa"]), 0.1, abs_tol=1e-7)
        _assert_same(
            route["selection_logits"],
            route["r2_logits"] + route["kappa"] * compatibility,
            modality,
        )


def test_compatibility_only_gradient_does_not_reach_alpha():
    model = Model(_cfg("R3_expert_compatibility"), _data_info())
    h_struct = torch.randn(6, 128, requires_grad=True)
    alpha = model._effective_alpha(model.alpha_raw, model.eps)
    compatibility = model._compatibility(h_struct, alpha)
    grad = torch.autograd.grad(compatibility.square().sum(), model.alpha_raw, allow_unused=True)[0]
    assert grad is None


def test_top2_and_active_only_final_logit_balance_formula():
    model = Model(_cfg("R3_expert_compatibility"), _data_info()).eval()
    with torch.no_grad():
        _, _, _, aux, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    balances = []
    for modality in ("text", "visual"):
        route = info["router"][modality]
        active = info["active_nodes"]
        assert torch.equal(route["selected_mask"].sum(-1), torch.full((6,), 2))
        assert torch.equal((route["route_weights"] > 0).sum(-1), torch.full((6,), 2))
        assert torch.allclose(route["route_weights"].sum(-1), torch.ones(6), atol=1e-7)
        dense = route["dense_probs"][active].mean(dim=0)
        share = route["selected_mask"][active].float().mean(dim=0) / 2
        expected = model.num_experts * (dense * share).sum()
        _assert_same(route["importance"], dense, f"{modality}.importance")
        _assert_same(route["selection_share"], share, f"{modality}.share")
        _assert_same(route["balance"], expected, f"{modality}.balance")
        balances.append(expected)
    _assert_same(aux, 0.01 * 0.5 * (balances[0] + balances[1]), "aux")


def test_new_module_freezing_follows_variant_activity():
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info())
        evidence_trainable = variant in {"R2_structure_grounded", "R3_expert_compatibility"}
        compatibility_trainable = variant == "R3_expert_compatibility"
        assert all(p.requires_grad == evidence_trainable for p in model.evidence_scalar_projector.parameters())
        assert all(p.requires_grad == evidence_trainable for p in model.evidence_norm.parameters())
        assert model.node_residual_raw.requires_grad == evidence_trainable
        assert model.compatibility_residual_raw.requires_grad == compatibility_trainable
        assert all(p.requires_grad == compatibility_trainable for p in model.expert_query_proj.parameters())


def test_trajectory_order_is_config_driven_and_fixed_to_four():
    model = Model(_cfg("R2_structure_grounded", trajectory_order=4), _data_info())
    assert model.trajectory_order == 4
    with pytest.raises(ValueError, match="trajectory_order=4"):
        Model(_cfg("R2_structure_grounded", trajectory_order=3), _data_info())
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    assert info["details"]["text"]["basis"].shape[0] == 4
