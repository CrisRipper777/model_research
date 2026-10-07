from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from src.models.mvcge_mag_v2 import Model


VARIANTS = (
    "M0_anchor",
    "M1_direct_moe",
    "M2_anchored_moe",
    "M3_collaborative_moe",
)


def _cfg(variant: str):
    return OmegaConf.create(
        {
            "model": {
                "name": "mvcge_mag_v2",
                "variant": variant,
                "hidden_dim": 256,
                "num_layers": 4,
                "dropout": 0.0,
                "trajectory_order": 4,
                "num_experts": 4,
                "top_k": 2,
                "expert_bottleneck": 64,
                "expert_feature_scale": 0.1,
                "router_dim": 64,
                "router_hidden_dim": 128,
                "modality_embed_dim": 16,
                "base_gate_init": -2.0,
                "direct_strength_init": -1.15,
                "residual_strength_init": -2.0,
                "residual_max": 0.25,
                "balance_weight": 0.01,
                "eps": 1.0e-8,
            }
        }
    )


def _data_info():
    return {"input_dim": 11, "text_dim": 6, "visual_dim": 5, "num_nodes": 6, "num_classes": 3}


def _path_graph():
    return torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )


def test_all_variants_forward_backward_and_diagnostics_are_finite():
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Model(_cfg(variant), _data_info())
        model.train()
        z, _, _, aux, info = model(x, _path_graph(), return_details=True)
        assert z.shape == (6, 256)
        assert torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        assert torch.isfinite(info["balance_loss"])
        for modality in ("text", "visual"):
            details = info["details"][modality]
            for key in (
                "prior", "basis", "beta", "base", "base_scaled", "alpha",
                "expert_inputs", "expert_outputs", "mixture", "scaled_mixture",
                "selection_logits", "dense_probs", "route_weights", "strength",
            ):
                assert torch.isfinite(details[key]).all(), (variant, modality, key)
        (z.square().mean() + aux).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_matched_initialization_is_variant_invariant_except_strength_bias():
    snapshots = {}
    for variant in VARIANTS:
        torch.manual_seed(123)
        model = Model(_cfg(variant), _data_info())
        classifier = torch.nn.Linear(model.out_dim, 3)
        snapshots[variant] = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        snapshots[variant]["classifier"] = {
            key: value.detach().clone() for key, value in classifier.state_dict().items()
        }
    reference = snapshots["M2_anchored_moe"]
    for variant in VARIANTS:
        for key, value in reference.items():
            if key == "strength_head.bias" or key == "classifier":
                continue
            assert torch.equal(value, snapshots[variant][key]), (variant, key)
        for key, value in reference["classifier"].items():
            assert torch.equal(value, snapshots[variant]["classifier"][key])
    assert math.isclose(snapshots["M1_direct_moe"]["strength_head.bias"].item(), -1.15, abs_tol=1e-7)
    assert math.isclose(snapshots["M2_anchored_moe"]["strength_head.bias"].item(), -2.0, abs_tol=1e-7)
    assert math.isclose(snapshots["M3_collaborative_moe"]["strength_head.bias"].item(), -2.0, abs_tol=1e-7)


def test_isolates_preserve_prior_and_have_zero_structural_experts():
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
            assert torch.equal(details["output"][5], details["prior"][5])


def test_raw_polynomial_trajectory_matches_fixed_order_four_definition():
    torch.manual_seed(19)
    model = Model(_cfg("M3_collaborative_moe"), _data_info()).eval()
    x = torch.randn(6, 11)
    with torch.no_grad():
        _, _, _, _, info = model(x, _path_graph(), return_details=True)
    active = info["active_nodes"]
    for modality in ("text", "visual"):
        item = info["details"][modality]
        states = []
        state = item["prior"]
        src, dst, norm, _, _ = model._normalized_operator(_path_graph(), 6, item["prior"].dtype)
        for _ in range(4):
            state = model._propagate(state, src, dst, norm)
            state = state * active.to(state.dtype).unsqueeze(-1)
            states.append(state)
        for order, state in enumerate(states):
            expected = model._active_rms_norm(state, active)
            assert torch.allclose(item["basis"][order], expected, atol=1e-7, rtol=0.0)


def test_anchor_and_hadamard_profile_initialization():
    model = Model(_cfg("M2_anchored_moe"), _data_info()).eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    expected_beta = torch.full((4,), 0.5)
    expected_alpha = torch.tensor(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        dtype=torch.float32,
    ) * 0.5
    for modality in ("text", "visual"):
        assert torch.allclose(info["details"][modality]["beta"], expected_beta, atol=1e-7, rtol=0)
        assert torch.allclose(info["details"][modality]["alpha"], expected_alpha, atol=1e-7, rtol=0)
    assert torch.allclose(model.alpha_raw @ model.alpha_raw.T, torch.eye(4), atol=1e-7, rtol=0)


def test_top_two_is_exact_and_selected_weights_sum_to_one():
    model = Model(_cfg("M3_collaborative_moe"), _data_info()).eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        route = info["router"][modality]
        assert torch.equal(route["selected_mask"].sum(dim=-1), torch.full((6,), 2))
        assert torch.equal((route["route_weights"] > 0).sum(dim=-1), torch.full((6,), 2))
        assert torch.allclose(route["route_weights"].sum(dim=-1), torch.ones(6), atol=1e-7)


def test_cross_view_context_is_zero_for_m1_m2_and_average_for_m3():
    x = torch.randn(6, 11)
    cross_contexts = {}
    collaborative_model = None
    collaborative_details = None
    for variant in ("M1_direct_moe", "M2_anchored_moe", "M3_collaborative_moe"):
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, _path_graph(), return_details=True)
        for modality in ("text", "visual"):
            cross_contexts[(variant, modality)] = info["router"][modality]["router_input"][:, 192:256]
        if variant == "M3_collaborative_moe":
            collaborative_model = model
            collaborative_details = info["details"]
    for variant in ("M1_direct_moe", "M2_anchored_moe"):
        for modality in ("text", "visual"):
            assert torch.count_nonzero(cross_contexts[(variant, modality)]) == 0
    expected = 0.5 * (
        collaborative_model.router_norms["text"](
            collaborative_model.router_projectors["text"](
                collaborative_details["text"]["prior"]
            )
        )
        + collaborative_model.router_norms["visual"](
            collaborative_model.router_projectors["visual"](
                collaborative_details["visual"]["prior"]
            )
        )
    )
    assert torch.allclose(cross_contexts[("M3_collaborative_moe", "text")], expected, atol=1e-6)


def test_load_balance_uses_active_nodes_only_and_m0_has_no_auxiliary_loss():
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, aux, info = model(x, _path_graph(), return_details=True)
        if variant == "M0_anchor":
            assert aux.item() == 0.0
        else:
            for modality in ("text", "visual"):
                route = info["router"][modality]
                active = info["active_nodes"]
                share = route["selected_mask"][active].float().mean(dim=0) / 2
                assert torch.allclose(route["selection_share"], share, atol=1e-7)
                assert torch.allclose(route["selection_share"].sum(), torch.tensor(1.0), atol=1e-7)
            expected = 0.01 * 0.5 * sum(
                model.num_experts
                * (info["router"][m]["importance"] * info["router"][m]["selection_share"]).sum()
                for m in ("text", "visual")
            )
            assert torch.allclose(aux, expected, atol=1e-7)


def test_strength_initialization_and_independent_router_heads():
    expected = {
        "M0_anchor": 1 / (1 + math.exp(2)),
        "M1_direct_moe": 1 / (1 + math.exp(1.15)),
        "M2_anchored_moe": 0.25 / (1 + math.exp(2)),
        "M3_collaborative_moe": 0.25 / (1 + math.exp(2)),
    }
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
        if variant == "M0_anchor":
            assert math.isclose(float(torch.sigmoid(model.gamma_base)), expected[variant], abs_tol=1e-7)
        else:
            gate = info["details"]["text"]["strength"]
            if variant in {"M2_anchored_moe", "M3_collaborative_moe"}:
                gate = 0.25 * gate
            assert torch.allclose(gate, torch.full_like(gate, expected[variant]), atol=1e-7)
        assert model.selection_head is not model.strength_head
        assert model.selection_head.weight.data_ptr() != model.strength_head.weight.data_ptr()


def test_self_loops_are_removed_without_adding_any():
    x = torch.randn(3, 11)
    loops = torch.tensor([[0, 1, 2], [0, 1, 2]])
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, loops, return_details=True)
        assert info["input_self_loops_removed"] == 3
        assert not info["active_nodes"].any()
        for modality in ("text", "visual"):
            item = info["details"][modality]
            assert torch.count_nonzero(item["basis"]) == 0
            assert torch.equal(item["output"], item["prior"])
