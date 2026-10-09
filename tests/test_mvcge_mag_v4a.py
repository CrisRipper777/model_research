from __future__ import annotations

import copy
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mvcge_mag_v3c import Model as V3CModel
from src.models.mvcge_mag_v4a import Model
from src.tasks.common import build_optimizer


VARIANTS = (
    "R0_raw",
    "R1_global_residual",
    "R2_expert_residual",
    "R3_adaptive_residual",
)
DATA_INFO = {
    "input_dim": 6,
    "text_dim": 3,
    "visual_dim": 3,
    "num_nodes": 6,
    "num_classes": 3,
}


def _cfg(variant: str, *, dropout: float = 0.0):
    return OmegaConf.create(
        {
            "model": {
                "name": "mvcge_mag_v4a",
                "variant": variant,
                "hidden_dim": 256,
                "dropout": dropout,
                "trajectory_order": 4,
                "num_experts": 4,
                "top_k": 2,
                "expert_bottleneck": 64,
                "expert_feature_scale": 0.1,
                "router_dim": 64,
                "router_hidden_dim": 128,
                "modality_embed_dim": 16,
                "strength_init": -1.15,
                "residual_beta_max": 0.25,
                "context_mix_init": 0.10,
                "role_mix_init": 0.10,
                "semantic_layernorm_eps": 1.0e-5,
                "evidence_scalar_dim": 11,
                "expert_key_embed_dim": 16,
                "compatibility_dim": 64,
                "node_residual_init": 0.10,
                "compatibility_residual_init": 0.10,
                "balance_weight": 0.01,
                "eps": 1.0e-8,
                "weight_decay": 0.05,
                "lr": 0.001,
            },
            "task": {"optimizer": "adamw", "lr": 0.001, "weight_decay": 0.05},
        }
    )


def _v3c_cfg():
    cfg = _cfg("R0_raw")
    cfg.model.name = "mvcge_mag_v3c"
    cfg.model.variant = "F0_raw"
    del cfg.model.residual_beta_max
    del cfg.model.weight_decay
    del cfg.model.lr
    del cfg.task
    return cfg


def _graph(*, loops: bool = False):
    edges = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 2, 1, 3],
            [1, 0, 2, 1, 3, 2, 4, 3, 2, 0, 3, 1],
        ],
        dtype=torch.long,
    )
    if loops:
        edges = torch.cat(
            [edges, torch.tensor([[0, 2, 5], [0, 2, 5]], dtype=torch.long)], dim=1
        )
    return edges


def _features(seed: int = 17):
    return torch.randn(6, 6, generator=torch.Generator().manual_seed(seed))


def _close(a: torch.Tensor, b: torch.Tensor, *, atol: float = 1.0e-7, rtol: float = 0.0):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), float((a - b).abs().max())


def _model(variant="R0_raw", *, seed=913, dropout=0.0):
    torch.manual_seed(seed)
    return Model(_cfg(variant, dropout=dropout), DATA_INFO)


def _v3c_model(seed=913):
    torch.manual_seed(seed)
    return V3CModel(_v3c_cfg(), DATA_INFO)


def test_01_all_variants_forward_backward_and_diagnostics_are_finite():
    x, graph = _features(), _graph(loops=True)
    for variant in VARIANTS:
        model = _model(variant, dropout=0.1).train()
        z, _, _, aux, info = model(x, graph, return_details=True)
        assert z.shape == (6, 256) and torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        for modality in ("text", "visual"):
            item = info["details"][modality]
            for key in (
                "prior", "raw_states", "raw_trajectory", "support_messages",
                "discrepant_messages", "support_trajectory", "discrepant_trajectory",
                "role_contrast_basis", "raw_profiles", "raw_expert_outputs",
                "residual_safe", "expert_outputs", "selection_logits", "route_weights",
                "strength", "support_norm_weight", "discrepant_norm_weight",
            ):
                assert torch.isfinite(item[key]).all(), (variant, modality, key)
        (z.square().mean() + aux).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_02_same_seed_common_initialization_and_classifier_rng_are_exact():
    snapshots = {}
    for variant in VARIANTS:
        model = _model(variant)
        classifier = torch.nn.Linear(model.out_dim, 3)
        snapshots[variant] = (copy.deepcopy(model.state_dict()), classifier.state_dict())
    base_state, base_head = snapshots[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        state, head = snapshots[variant]
        for key, value in base_state.items():
            assert key in state and torch.equal(value, state[key]), (variant, key)
        for key, value in base_head.items():
            assert torch.equal(value, head[key]), (variant, key)


def test_03_r0_state_and_forward_regress_exactly_to_v3c_f0():
    torch.manual_seed(913)
    old = V3CModel(_v3c_cfg(), DATA_INFO).eval()
    torch.manual_seed(913)
    new = Model(_cfg("R0_raw"), DATA_INFO).eval()
    for key, value in old.state_dict().items():
        assert key in new.state_dict() and torch.equal(value, new.state_dict()[key]), key
    x, graph = _features(71), _graph(loops=True)
    with torch.no_grad():
        before = old(x, graph, return_details=True)
        after = new(x, graph, return_details=True)
    assert torch.equal(before[0], after[0]) and torch.equal(before[3], after[3])
    for modality in ("text", "visual"):
        old_item = before[4]["details"][modality]
        new_item = after[4]["details"][modality]
        for key in ("prior", "raw_trajectory", "alpha", "raw_profiles", "expert_outputs", "route_weights", "strength", "output"):
            _close(old_item[key], new_item[key])


def test_04_zero_initialized_r1_r2_r3_full_outputs_equal_r0():
    x, graph = _features(71), _graph(loops=True)
    base = _model("R0_raw").eval()
    with torch.no_grad():
        expected = base(x, graph)
    for variant in VARIANTS[1:]:
        model = _model(variant).eval()
        with torch.no_grad():
            actual = model(x, graph)
        assert torch.equal(expected[0], actual[0]), variant
        assert torch.equal(expected[3], actual[3]), variant


@pytest.mark.parametrize("variant", VARIANTS)
def test_05_role_partition_exactly_matches_v3c_and_adds_no_edges(variant):
    model = _model(variant)
    v3 = V3CModel(_v3c_cfg(), DATA_INFO)
    x, graph = _features(), _graph(loops=True)
    src0, dst0, raw0, active0, loops0 = model._normalized_operator(graph, 6, x.dtype)
    src1, dst1, raw1, active1, loops1 = v3._normalized_operator(graph, 6, x.dtype)
    assert loops0 == loops1 == 3
    assert torch.equal(src0, src1) and torch.equal(dst0, dst1)
    assert torch.equal(raw0, raw1) and torch.equal(active0, active1)
    for modality in ("text", "visual"):
        features = model._split_modalities(x)[modality]
        old_role = v3._role_partition(features, src0, dst0, raw0)
        new_role = model._role_partition(features, src0, dst0, raw0)
        assert torch.equal(old_role["supportive_mask"], new_role["supportive_mask"])
        assert torch.equal(old_role["discrepant_mask"], new_role["discrepant_mask"])
        _close(new_role["support_norm_weight"] + new_role["discrepant_norm_weight"], raw0)


def test_06_r0_removes_input_self_loops_without_adding_any():
    src, dst, norm, active, loops = Model._normalized_operator(_graph(loops=True), 6, torch.float32)
    assert loops == 3 and not torch.any(src == dst) and norm.numel() == src.numel()
    assert torch.equal(active, torch.tensor([True, True, True, True, True, False]))


def test_07_r0_raw_trajectory_exactly_matches_v3c_f0():
    model, v3 = _model("R0_raw").eval(), _v3c_model().eval()
    with torch.no_grad():
        new = model(_features(71), _graph(), return_details=True)[4]
        old = v3(_features(71), _graph(), return_details=True)[4]
    for modality in ("text", "visual"):
        _close(new["details"][modality]["raw_trajectory"], old["details"][modality]["raw_trajectory"])


def test_08_last_hop_support_and_discrepant_messages_use_previous_raw_state():
    model = _model("R2_expert_residual").eval()
    x, graph = _features(51), _graph()
    with torch.no_grad():
        info = model(x, graph, return_details=True)[4]
    src, dst, raw_norm, active, _ = model._normalized_operator(graph, 6, x.dtype)
    for modality in ("text", "visual"):
        item = info["details"][modality]
        for hop in range(4):
            prev = item["prior"] if hop == 0 else item["raw_states"][hop - 1]
            role = info["role_channels"][modality]
            expected_s = model._propagate(prev, src, dst, role["support_norm_weight"])
            expected_d = model._propagate(prev, src, dst, role["discrepant_norm_weight"])
            expected_s *= active.to(prev.dtype).unsqueeze(-1)
            expected_d *= active.to(prev.dtype).unsqueeze(-1)
            _close(item["support_messages"][hop], expected_s, atol=1e-7)
            _close(item["discrepant_messages"][hop], expected_d, atol=1e-6)


def test_09_hop1_message_partition_conserves_raw_state():
    model = _model("R1_global_residual").eval()
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        _close(item["support_messages"][0] + item["discrepant_messages"][0], item["raw_states"][0], atol=1e-6, rtol=1e-6)


def test_10_hop2_messages_both_use_raw_hop1_not_role_recursion():
    model = _model("R1_global_residual").eval()
    x, graph = _features(), _graph()
    with torch.no_grad():
        info = model(x, graph, return_details=True)[4]
    src, dst, _, active, _ = model._normalized_operator(graph, 6, x.dtype)
    for modality, item in info["details"].items():
        role = info["role_channels"][modality]
        prev = item["raw_states"][0]
        sup = model._propagate(prev, src, dst, role["support_norm_weight"]) * active.unsqueeze(-1)
        disc = model._propagate(prev, src, dst, role["discrepant_norm_weight"]) * active.unsqueeze(-1)
        _close(item["support_messages"][1], sup)
        _close(item["discrepant_messages"][1], disc, atol=1e-6)


def test_11_hop2_decomposition_conserves_raw_state():
    model = _model("R3_adaptive_residual").eval()
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        _close(item["support_messages"][1] + item["discrepant_messages"][1], item["raw_states"][1], atol=1e-6, rtol=1e-6)


def test_12_shared_denominator_is_exactly_raw_hop_channelwise_rms():
    model = _model("R2_expert_residual").eval()
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    active = torch.tensor([True, True, True, True, True, False])
    for item in details.values():
        expected = torch.stack([
            torch.sqrt(state[active].square().mean(0) + model.eps)
            for state in item["raw_states"]
        ])
        _close(item["raw_denominators"], expected, atol=0.0)


def test_13_normalized_support_and_discrepant_sum_to_raw():
    model = _model("R2_expert_residual").eval()
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        _close(item["support_trajectory"] + item["discrepant_trajectory"], item["raw_trajectory"], atol=1e-6, rtol=1e-6)


def test_14_role_channels_use_physical_active_mask_and_shared_raw_norm():
    model = _model("R2_expert_residual").eval()
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        assert torch.count_nonzero(item["support_trajectory"][:, 5]) == 0
        assert torch.count_nonzero(item["discrepant_trajectory"][:, 5]) == 0
        for hop in range(4):
            denom = item["raw_denominators"][hop]
            _close(item["support_trajectory"][hop, :5], item["support_messages"][hop, :5] / denom)
            _close(item["discrepant_trajectory"][hop, :5], item["discrepant_messages"][hop, :5] / denom, atol=1e-6)


def test_15_role_contrast_is_support_minus_discrepant():
    model = _model("R3_adaptive_residual").eval()
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        _close(item["role_contrast_basis"], item["support_trajectory"] - item["discrepant_trajectory"], atol=0.0)


def test_16_raw_profile_exactly_matches_v3c_f0():
    model, v3 = _model("R0_raw").eval(), _v3c_model().eval()
    x, graph = _features(82), _graph()
    with torch.no_grad():
        a = model(x, graph, return_details=True)[4]["details"]
        b = v3(x, graph, return_details=True)[4]["details"]
    for modality in ("text", "visual"):
        _close(a[modality]["raw_profiles"], b[modality]["raw_profiles"])


def test_17_raw_expert_output_exactly_matches_v3c_f0():
    model, v3 = _model("R0_raw").eval(), _v3c_model().eval()
    x, graph = _features(82), _graph()
    with torch.no_grad():
        a = model(x, graph, return_details=True)[4]["details"]
        b = v3(x, graph, return_details=True)[4]["details"]
    for modality in ("text", "visual"):
        _close(a[modality]["raw_expert_outputs"], b[modality]["expert_outputs"])


def test_18_role_residual_never_enters_raw_expert_mlp_input():
    model = _model("R3_adaptive_residual").eval()
    seen = []
    hooks = [expert.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone())) for expert in model.experts]
    try:
        with torch.no_grad():
            info = model(_features(51), _graph(), return_details=True)[4]
    finally:
        for hook in hooks:
            hook.remove()
    assert len(seen) == 2 * model.num_experts
    for mod_i, modality in enumerate(("text", "visual")):
        expected = info["details"][modality]["raw_profiles"]
        for expert_id in range(model.num_experts):
            _close(seen[mod_i * model.num_experts + expert_id], expected[expert_id])
        assert not torch.equal(info["details"][modality]["expert_inputs"], info["details"][modality]["final_expert_outputs"])


def test_19_shared_profile_residual_stops_gradient_to_alpha_but_not_contrast():
    alpha = torch.randn(4, 4, requires_grad=True)
    contrast = torch.randn(4, 6, 8, requires_grad=True)
    residual = Model._shared_residual_profile(alpha, contrast)
    residual.square().sum().backward()
    assert alpha.grad is None
    assert contrast.grad is not None and torch.isfinite(contrast.grad).all()


def test_20_r1_and_r2_shared_profile_matches_detached_current_alpha():
    model = _model("R2_expert_residual").eval()
    with torch.no_grad():
        item = model(_features(), _graph(), return_details=True)[4]["details"]["text"]
    expected = Model._shared_residual_profile(item["alpha"], item["role_contrast_basis"])
    _close(item["residual_shared_raw"], expected)


def test_21_r3_profile_uses_independent_gamma_not_alpha():
    model = _model("R3_adaptive_residual")
    with torch.no_grad():
        model.residual_gamma_raw[0] = torch.tensor([0.2, -0.1, 0.3, -0.4])
        item = model(_features(), _graph(), return_details=True)[4]["details"]["text"]
    expected_gamma = model._normalized_gamma(model.residual_gamma_raw, model.eps)
    expected = Model._adaptive_residual_profile(expected_gamma, item["role_contrast_basis"])
    _close(item["residual_adaptive_raw"], expected)
    assert not torch.allclose(item["residual_adaptive_raw"], item["residual_shared_raw"])


def test_22_gamma_initialization_is_exact_hadamard_and_matches_alpha_profile():
    model = _model("R3_adaptive_residual")
    expected = 0.5 * torch.tensor(
        [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]],
        dtype=torch.float32,
    )
    assert torch.equal(model.residual_gamma_raw, expected)
    _close(model._normalized_gamma(model.residual_gamma_raw, model.eps), model._effective_alpha(model.alpha_raw, model.eps), atol=0.0)


def test_23_gamma_is_four_by_four_and_shared_across_modalities():
    model = _model("R3_adaptive_residual").eval()
    assert tuple(model.residual_gamma_raw.shape) == (4, 4)
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    _close(details["text"]["gamma"], details["visual"]["gamma"], atol=0.0)


def test_24_global_gate_initializes_zero_and_has_quarter_derivative():
    model = _model("R1_global_residual")
    assert torch.equal(model.residual_global_raw, torch.zeros(1))
    beta = model.residual_beta_max * torch.tanh(model.residual_global_raw)
    grad = torch.autograd.grad(beta.sum(), model.residual_global_raw)[0]
    _close(beta, torch.zeros_like(beta), atol=0.0)
    _close(grad, torch.full_like(grad, 0.25), atol=1e-7)


def test_25_expert_gate_initializes_zero():
    for variant in ("R2_expert_residual", "R3_adaptive_residual"):
        model = _model(variant)
        assert torch.equal(model.residual_expert_raw, torch.zeros(4))
        beta = model.residual_beta_max * torch.tanh(model.residual_expert_raw)
        assert torch.equal(beta, torch.zeros_like(beta))


def test_26_beta_is_bounded_by_quarter():
    model = _model("R2_expert_residual")
    with torch.no_grad():
        model.residual_expert_raw.copy_(torch.tensor([-100.0, -2.0, 2.0, 100.0]))
    _, beta = model._residual_betas()
    assert torch.all(beta >= -0.25) and torch.all(beta <= 0.25)


def test_27_r1_beta_is_global_across_experts_and_modalities():
    model = _model("R1_global_residual").eval()
    with torch.no_grad():
        model.residual_global_raw.fill_(0.7)
        info = model(_features(), _graph(), return_details=True)[4]
    expected = 0.25 * torch.tanh(model.residual_global_raw[0])
    for modality in ("text", "visual"):
        _close(info["details"][modality]["expert_beta"], expected.expand(4))


def test_28_r2_r3_expert_beta_is_shared_across_modalities():
    for variant in ("R2_expert_residual", "R3_adaptive_residual"):
        model = _model(variant).eval()
        with torch.no_grad():
            model.residual_expert_raw.copy_(torch.tensor([-0.6, -0.2, 0.3, 0.8]))
            info = model(_features(), _graph(), return_details=True)[4]
        _close(info["details"]["text"]["expert_beta"], info["details"]["visual"]["expert_beta"], atol=0.0)


def test_29_safety_scale_never_exceeds_one():
    model = _model("R3_adaptive_residual")
    raw = torch.randn(4, 6, 8)
    q = torch.randn(4, 6, 8) * 3.0
    safe, scale, _, _ = model._safety_calibrate(raw, q, torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool))
    assert torch.all(scale <= 1.0) and torch.all(scale >= 0.0)
    assert torch.isfinite(safe).all()


def test_30_safety_downscales_large_residual_to_raw_profile_rms_bound():
    model = _model("R3_adaptive_residual")
    active = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    raw = torch.randn(4, 6, 8)
    q = 5.0 * raw
    safe, scale, raw_rms, _ = model._safety_calibrate(raw, q, active)
    safe_rms = torch.sqrt(safe[:, active].square().mean(dim=(1, 2)) + model.eps)
    assert torch.all(scale < 1.0)
    assert torch.all(safe_rms <= raw_rms + 1.0e-6)


def test_31_safety_never_upscales_a_weak_residual():
    model = _model("R3_adaptive_residual")
    active = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    raw = torch.randn(4, 6, 8)
    q = 0.1 * raw
    safe, scale, _, _ = model._safety_calibrate(raw, q, active)
    assert torch.equal(scale, torch.ones_like(scale))
    _close(safe, q, atol=0.0)


def test_32_safety_statistics_are_detached_but_residual_still_gets_task_gradient():
    model = _model("R3_adaptive_residual")
    active = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    raw = torch.randn(4, 6, 8, requires_grad=True)
    q = torch.randn(4, 6, 8, requires_grad=True)
    safe, scale, _, _ = model._safety_calibrate(raw, q, active)
    assert not scale.requires_grad
    safe.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_33_optimizer_exempts_only_global_and_expert_gates_not_gamma():
    model = _model("R3_adaptive_residual")
    head = torch.nn.Linear(model.out_dim, 3)
    optimizer = build_optimizer(list(model.parameters()) + list(head.parameters()), _cfg("R3_adaptive_residual"), model=model)
    assert model.no_weight_decay_parameter_names >= {
        "context_mix_raw", "role_mix_raw", "residual_global_raw", "residual_expert_raw"
    }
    global_group = next(g for g in optimizer.param_groups if id(model.residual_global_raw) in {id(p) for p in g["params"]})
    expert_group = next(g for g in optimizer.param_groups if id(model.residual_expert_raw) in {id(p) for p in g["params"]})
    gamma_group = next(g for g in optimizer.param_groups if id(model.residual_gamma_raw) in {id(p) for p in g["params"]})
    assert global_group["weight_decay"] == expert_group["weight_decay"] == 0.0
    assert gamma_group["weight_decay"] == pytest.approx(0.05)


def test_34_r0_freezes_all_v4a_added_parameters():
    model = _model("R0_raw")
    assert not model.residual_global_raw.requires_grad
    assert not model.residual_expert_raw.requires_grad
    assert not model.residual_gamma_raw.requires_grad


def test_35_r1_only_global_gate_is_trainable_among_added_parameters():
    model = _model("R1_global_residual")
    assert model.residual_global_raw.requires_grad
    assert not model.residual_expert_raw.requires_grad and not model.residual_gamma_raw.requires_grad


def test_36_r2_only_expert_gate_is_trainable_among_added_parameters():
    model = _model("R2_expert_residual")
    assert not model.residual_global_raw.requires_grad
    assert model.residual_expert_raw.requires_grad and not model.residual_gamma_raw.requires_grad


def test_37_r3_expert_gate_and_gamma_are_trainable():
    model = _model("R3_adaptive_residual")
    assert not model.residual_global_raw.requires_grad
    assert model.residual_expert_raw.requires_grad and model.residual_gamma_raw.requires_grad


def test_38_gamma_gradient_is_zero_when_beta_is_zero():
    model = _model("R3_adaptive_residual").eval()
    z, _, _, aux, _ = model(_features(91), _graph())
    (z.square().mean() + aux).backward()
    assert model.residual_gamma_raw.grad is not None
    assert torch.equal(model.residual_gamma_raw.grad, torch.zeros_like(model.residual_gamma_raw.grad))


def test_39_nonzero_beta_gives_finite_nonzero_gamma_gradient():
    model = _model("R3_adaptive_residual").eval()
    with torch.no_grad():
        model.residual_expert_raw.copy_(torch.tensor([-0.6, -0.2, 0.3, 0.8]))
    z, _, _, aux, _ = model(_features(91), _graph())
    (z.square().mean() + aux).backward()
    grad = model.residual_gamma_raw.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.linalg.vector_norm(grad) > 0


def test_40_every_variant_keeps_static_routes_and_strength():
    x, graph = _features(), _graph()
    for variant in VARIANTS:
        model = _model(variant).eval()
        with torch.no_grad():
            info = model(x, graph, return_details=True)[4]
        for modality in ("text", "visual"):
            route = info["details"][modality]
            assert torch.equal(route["selection_logits"], route["selection_logits"][:1].expand_as(route["selection_logits"]))
            assert torch.equal(route["strength"], route["strength"][:1].expand_as(route["strength"]))


def test_41_every_variant_has_exactly_two_nonzero_top2_weights():
    for variant in VARIANTS:
        model = _model(variant).eval()
        with torch.no_grad():
            info = model(_features(), _graph(), return_details=True)[4]
        for item in info["details"].values():
            assert torch.all((item["route_weights"] != 0).sum(dim=-1) == 2)


def test_42_load_balance_formula_is_unchanged():
    for variant in VARIANTS:
        model = _model(variant).eval()
        with torch.no_grad():
            z, _, _, aux, info = model(_features(), _graph(), return_details=True)
        balances = []
        for item in info["details"].values():
            active = info["active_nodes"]
            importance = item["dense_probs"][active].mean(dim=0)
            share = item["selected_mask"][active].float().mean(dim=0) / model.top_k
            balance = model.num_experts * (importance * share).sum()
            _close(item["balance"], balance)
            balances.append(balance)
        expected_aux = model.balance_weight * 0.5 * sum(balances)
        _close(aux, expected_aux)


def test_43_trajectory_order_is_fixed_to_four_hops():
    for variant in VARIANTS:
        model = _model(variant)
        assert model.trajectory_order == 4
        with torch.no_grad():
            info = model(_features(), _graph(), return_details=True)[4]
        for item in info["details"].values():
            assert item["raw_trajectory"].shape[0] == 4
            assert item["role_contrast_basis"].shape[0] == 4


def test_44_shared_and_adaptive_untrained_profiles_are_equal():
    model = _model("R3_adaptive_residual").eval()
    with torch.no_grad():
        details = model(_features(41), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        _close(item["residual_shared_raw"], item["residual_adaptive_raw"], atol=0.0)


def test_45_safety_bound_holds_for_model_residual_profiles():
    model = _model("R3_adaptive_residual").eval()
    with torch.no_grad():
        details = model(_features(51), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        assert torch.all(item["safety_scale"] <= 1.0)
        assert torch.all(item["residual_safe"].square().mean(dim=(1, 2)).sqrt() <= item["raw_profile_rms"] + 1e-6)


def test_46_inactive_nodes_are_zero_in_role_contrast_and_safe_residual():
    model = _model("R3_adaptive_residual").eval()
    with torch.no_grad():
        details = model(_features(), _graph(), return_details=True)[4]["details"]
    for item in details.values():
        assert torch.count_nonzero(item["role_contrast_basis"][:, 5]) == 0
        assert torch.count_nonzero(item["residual_safe"][:, 5]) == 0


def test_47_gamma_and_beta_controls_are_finite_for_extreme_raw_values():
    model = _model("R3_adaptive_residual")
    with torch.no_grad():
        model.residual_gamma_raw.fill_(1.0e20)
        model.residual_expert_raw.fill_(1.0e20)
    gamma = model._normalized_gamma(model.residual_gamma_raw, model.eps)
    _, beta = model._residual_betas()
    assert torch.isfinite(gamma).all() and torch.isfinite(beta).all()


def test_48_r1_r2_r3_do_not_change_alpha_or_role_assignment():
    x, graph = _features(91), _graph()
    base = _model("R0_raw").eval()
    with torch.no_grad():
        base_info = base(x, graph, return_details=True)[4]
    for variant in VARIANTS[1:]:
        model = _model(variant).eval()
        with torch.no_grad():
            info = model(x, graph, return_details=True)[4]
        assert torch.equal(model.alpha_raw, base.alpha_raw)
        for modality in ("text", "visual"):
            assert torch.equal(
                info["role_channels"][modality]["supportive_mask"],
                base_info["role_channels"][modality]["supportive_mask"],
            )


def test_49_training_forward_without_diagnostics_recomputes_role_hops_and_backpropagates():
    model = _model("R3_adaptive_residual").train()
    z, _, _, aux, _ = model(_features(92), _graph())
    (z.square().mean() + aux).backward()
    assert torch.isfinite(z).all() and torch.isfinite(aux)
    parameters = dict(model.named_parameters())
    for name in ("projectors.text.linear1.weight", "residual_expert_raw", "residual_gamma_raw"):
        gradient = parameters[name].grad
        if gradient is not None:
            assert torch.isfinite(gradient).all(), name
