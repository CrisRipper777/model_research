from __future__ import annotations

import copy
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mvcge_mag_v22 import Model as V22Model
from src.models.mvcge_mag_v3a import Model as V3AModel
from src.models.mvcge_mag_v3c import Model
from src.tasks.common import build_optimizer


VARIANTS = (
    "F0_raw",
    "F1_support",
    "F2_role_dual_smooth",
    "F3_role_functional",
)
DATA_INFO = {
    "input_dim": 6,
    "text_dim": 3,
    "visual_dim": 3,
    "num_nodes": 6,
    "num_classes": 3,
}


def _cfg(variant: str, *, name: str = "mvcge_mag_v3c", dropout: float = 0.0):
    return OmegaConf.create(
        {
            "model": {
                "name": name,
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


def _v3a_cfg():
    cfg = _cfg("C0_raw", name="mvcge_mag_v3a")
    del cfg.model.role_mix_init
    del cfg.model.semantic_layernorm_eps
    return cfg


def _v22_cfg():
    cfg = _v3a_cfg()
    cfg.model.name = "mvcge_mag_v22"
    cfg.model.variant = "R0_modality_static"
    del cfg.model.context_mix_init
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
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(6, 6, generator=generator)


def _close(a: torch.Tensor, b: torch.Tensor, *, tol: float = 1.0e-7):
    assert torch.allclose(a, b, atol=tol, rtol=0.0), float((a - b).abs().max())


def test_all_variants_forward_backward_and_role_diagnostics_are_finite():
    x, graph = _features(), _graph(loops=True)
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Model(_cfg(variant, dropout=0.2), DATA_INFO).train()
        z, _, _, aux, info = model(x, graph, return_details=True)
        assert z.shape == (6, 256)
        assert torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        for modality in ("text", "visual"):
            item = info["details"][modality]
            for key in (
                "prior", "raw_trajectory", "support_trajectory",
                "discrepant_smooth_trajectory", "discrepant_signed_trajectory",
                "raw_profiles", "support_profiles", "discrepant_smooth_profiles",
                "discrepant_signed_profiles", "expert_inputs", "expert_outputs",
                "selection_logits", "route_weights", "strength", "role_score",
                "support_norm_weight", "discrepant_norm_weight",
            ):
                assert torch.isfinite(item[key]).all(), (variant, modality, key)
        (z.square().mean() + aux).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_all_variants_have_exact_common_initialization_and_classifier_rng():
    snapshots = {}
    for variant in VARIANTS:
        torch.manual_seed(913)
        model = Model(_cfg(variant), DATA_INFO)
        classifier = torch.nn.Linear(model.out_dim, 3)
        snapshots[variant] = (copy.deepcopy(model.state_dict()), classifier.state_dict())
    first = snapshots[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        assert first[0].keys() == snapshots[variant][0].keys()
        for key, value in first[0].items():
            assert torch.equal(value, snapshots[variant][0][key]), (variant, key)
        for key, value in first[1].items():
            assert torch.equal(value, snapshots[variant][1][key]), (variant, key)


@pytest.mark.parametrize("variant", VARIANTS)
def test_classifier_rng_matches_v22_r0(variant):
    torch.manual_seed(887)
    old = V22Model(_v22_cfg(), DATA_INFO)
    old_classifier = torch.nn.Linear(old.out_dim, 3)
    torch.manual_seed(887)
    new = Model(_cfg(variant), DATA_INFO)
    new_classifier = torch.nn.Linear(new.out_dim, 3)
    for key, value in old_classifier.state_dict().items():
        assert torch.equal(value, new_classifier.state_dict()[key]), (variant, key)


def test_f0_exact_full_forward_and_raw_trajectory_regression_to_v3a_c0():
    seed = 913
    torch.manual_seed(seed)
    old = V3AModel(_v3a_cfg(), DATA_INFO).eval()
    torch.manual_seed(seed)
    new = Model(_cfg("F0_raw"), DATA_INFO).eval()
    old_state, new_state = old.state_dict(), new.state_dict()
    for key, value in old_state.items():
        assert key in new_state and torch.equal(value, new_state[key]), key

    x, graph = _features(71), _graph(loops=True)
    with torch.no_grad():
        z0, _, _, aux0, info0 = old(x, graph, return_details=True)
        z1, _, _, aux1, info1 = new(x, graph, return_details=True)
    _close(z0, z1)
    _close(aux0, aux1)
    assert info1["variant"] == "F0_raw"
    for modality in ("text", "visual"):
        before, after = info0["details"][modality], info1["details"][modality]
        for key in (
            "prior", "raw_trajectory", "alpha_raw", "raw_profiles", "expert_outputs",
            "selection_logits", "route_weights", "strength", "output", "mixture",
            "scaled_mixture",
        ):
            old_key = "alpha" if key == "alpha_raw" else key
            _close(before[old_key], after[key])
    assert info1["input_self_loops_removed"] == 3


def test_f0_v22_shared_state_and_classifier_initialization_are_exact():
    torch.manual_seed(221)
    old = V22Model(_v22_cfg(), DATA_INFO)
    old_head = torch.nn.Linear(old.out_dim, 3)
    torch.manual_seed(221)
    new = Model(_cfg("F0_raw"), DATA_INFO)
    new_head = torch.nn.Linear(new.out_dim, 3)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    for key, value in old_head.state_dict().items():
        assert torch.equal(value, new_head.state_dict()[key]), key


def test_physical_operator_removes_input_self_loops_and_adds_none():
    src, dst, norm, active, loops = Model._normalized_operator(
        _graph(loops=True), 6, torch.float32
    )
    assert loops == 3
    assert not torch.any(src == dst)
    assert torch.equal(active, torch.tensor([True, True, True, True, True, False]))
    assert norm.numel() == src.numel()


def test_layernorm_semantic_transform_is_parameter_free_and_matches_formula():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x = _features().requires_grad_()
    actual = model._semantic_features(x[:, :3])
    transformed = torch.nn.functional.layer_norm(
        x[:, :3].detach(), (3,), weight=None, bias=None, eps=1.0e-5
    )
    expected = transformed / torch.linalg.vector_norm(
        transformed, dim=-1, keepdim=True
    ).clamp_min(1.0e-5)
    _close(actual, expected)
    assert not actual.requires_grad
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "raw",
    [torch.zeros(6, 3), torch.ones(6, 3), torch.full((6, 3), 1.0e-12)],
)
def test_zero_or_constant_semantic_features_are_finite_zero(raw):
    result = Model._semantic_features(raw)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0


def test_semantic_edge_cosines_are_clamped_to_unit_interval():
    semantic = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    cosine = Model._edge_cosine(semantic, torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]))
    assert torch.all((cosine >= -1.0) & (cosine <= 1.0))
    _close(cosine, torch.tensor([1.0, 0.0, -1.0]))


def test_local_mu_uses_incoming_physical_neighbor_cosine_mean_and_isolated_zero():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x, graph = _features(), _graph()
    inputs = model._split_modalities(x)
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    role = model._role_partition(inputs["text"], src, dst, raw)
    expected_count = torch.zeros(6).index_add(0, dst, torch.ones_like(role["semantic_cosine"]))
    expected_sum = torch.zeros(6).index_add(0, dst, role["semantic_cosine"])
    expected_mu = expected_sum / expected_count.clamp_min(1.0)
    _close(role["incoming_count"], expected_count)
    _close(role["similarity_sum"], expected_sum)
    _close(role["local_mean"], expected_mu)
    assert role["local_mean"][5] == 0


def test_role_score_exact_local_relative_formula():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x, graph = _features(22), _graph()
    inputs = model._split_modalities(x)
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    role = model._role_partition(inputs["visual"], src, dst, raw)
    expected = role["semantic_cosine"] - 0.5 * (
        role["local_mean"][src] + role["local_mean"][dst]
    )
    _close(role["role_score"], expected)


def test_support_and_discrepant_masks_are_mutually_exclusive_exhaustive_and_ties_support():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x, graph = _features(29), _graph()
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    role = model._role_partition(model._split_modalities(x)["text"], src, dst, raw)
    assert not torch.any(role["supportive_mask"] & role["discrepant_mask"])
    assert torch.all(role["supportive_mask"] | role["discrepant_mask"])
    assert torch.all(role["supportive_mask"] == (role["role_score"] >= 0.0))
    tie = role["role_score"] == 0
    assert torch.all(role["supportive_mask"][tie])


def test_undirected_reverse_edges_receive_identical_roles():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x, graph = _features(37), _graph()
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    role = model._role_partition(model._split_modalities(x)["text"], src, dst, raw)
    edge_to_role = {
        (int(s), int(d)): bool(mask)
        for s, d, mask in zip(src.tolist(), dst.tolist(), role["supportive_mask"].tolist())
    }
    for (s, d), mask in edge_to_role.items():
        assert edge_to_role[(d, s)] == mask


def test_role_partition_is_independent_of_projectors_router_and_classifier():
    model = Model(_cfg("F0_raw"), DATA_INFO).eval()
    x, graph = _features(41), _graph()
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    first = model._role_partition(model._split_modalities(x)["text"], src, dst, raw)
    with torch.no_grad():
        for p in model.projectors.parameters():
            p.add_(1.0)
        for p in model.router_projectors.parameters():
            p.mul_(0.0)
        for p in model.selection_head.parameters():
            p.add_(3.0)
        classifier = torch.nn.Linear(model.out_dim, 3)
        for p in classifier.parameters():
            p.mul_(0.0)
    second = model._role_partition(model._split_modalities(x)["text"], src, dst, raw)
    assert torch.equal(first["role_score"], second["role_score"])
    assert torch.equal(first["supportive_mask"], second["supportive_mask"])


def test_text_and_visual_roles_depend_only_on_their_own_raw_modality_features():
    model = Model(_cfg("F0_raw"), DATA_INFO).eval()
    x, graph = _features(53), _graph()
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    first = model._split_modalities(x)
    text_first = model._role_partition(first["text"], src, dst, raw)
    visual_first = model._role_partition(first["visual"], src, dst, raw)
    changed_text = x.clone()
    changed_text[:, :3] = torch.roll(changed_text[:, :3], 1, 0) * 2.0
    split = model._split_modalities(changed_text)
    text_second = model._role_partition(split["text"], src, dst, raw)
    visual_second = model._role_partition(split["visual"], src, dst, raw)
    assert not torch.equal(text_first["role_score"], text_second["role_score"])
    assert torch.equal(visual_first["role_score"], visual_second["role_score"])
    changed_visual = x.clone()
    changed_visual[:, 3:] = torch.roll(changed_visual[:, 3:], 2, 0) * 3.0
    split = model._split_modalities(changed_visual)
    text_third = model._role_partition(split["text"], src, dst, raw)
    visual_third = model._role_partition(split["visual"], src, dst, raw)
    assert torch.equal(text_first["role_score"], text_third["role_score"])
    assert not torch.equal(visual_first["role_score"], visual_third["role_score"])


def test_role_computation_is_detached_and_per_modality():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x = _features().requires_grad_()
    src, dst, raw, _, _ = model._normalized_operator(_graph(), 6, torch.float32)
    roles = {
        m: model._role_partition(v, src, dst, raw)
        for m, v in model._split_modalities(x).items()
    }
    assert all(not roles[m]["role_score"].requires_grad for m in ("text", "visual"))
    assert not torch.equal(roles["text"]["role_score"], roles["visual"]["role_score"])


def test_role_weights_partition_raw_operator_without_channel_renormalization():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x, graph = _features(61), _graph()
    inputs = model._split_modalities(x)
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    role = model._role_partition(inputs["text"], src, dst, raw)
    _close(role["support_norm_weight"], raw * role["supportive_mask"].float())
    _close(role["discrepant_norm_weight"], raw * role["discrepant_mask"].float())
    _close(role["support_norm_weight"] + role["discrepant_norm_weight"], raw)
    assert float(role["partition_max_abs_error"]) <= 1.0e-7


def test_support_and_discrepant_active_masks_are_incoming_count_gt_zero():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    x, graph = _features(67), _graph()
    src, dst, raw, _, _ = model._normalized_operator(graph, 6, torch.float32)
    role = model._role_partition(model._split_modalities(x)["visual"], src, dst, raw)
    assert torch.equal(role["support_active"], role["support_incoming_count"] > 0)
    assert torch.equal(role["discrepant_active"], role["discrepant_incoming_count"] > 0)


@pytest.mark.parametrize("channel", ["support", "discrepant"])
def test_channel_without_incoming_edges_has_zero_states_and_trajectory(channel):
    model = Model(_cfg("F0_raw"), DATA_INFO)
    prior = _features(73)
    src, dst = torch.tensor([0, 1]), torch.tensor([1, 0])
    role = {
        "support_norm_weight": torch.zeros(2),
        "discrepant_norm_weight": torch.zeros(2),
        "support_active": torch.zeros(6, dtype=torch.bool),
        "discrepant_active": torch.zeros(6, dtype=torch.bool),
    }
    result = model._role_trajectories(prior, src, dst, role)
    state_key = "support_states" if channel == "support" else "discrepant_states"
    trajectory_key = (
        "support_trajectory"
        if channel == "support"
        else "discrepant_smooth_trajectory"
    )
    assert torch.count_nonzero(result[state_key]) == 0
    assert torch.count_nonzero(result[trajectory_key]) == 0
    if channel == "discrepant":
        assert torch.count_nonzero(result["discrepant_signed_trajectory"]) == 0


def test_role_trajectory_recurrences_and_separate_channel_active_rms_norm():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    prior = _features(79)
    src, dst, raw, _, _ = model._normalized_operator(_graph(), 6, torch.float32)
    role = model._role_partition(model._split_modalities(_features(83))["text"], src, dst, raw)
    result = model._role_trajectories(prior, src, dst, role)

    state = prior
    support_states, disc_states, signed = [], [], []
    previous = prior
    for hop in range(4):
        state = model._propagate(state, src, dst, role["support_norm_weight"])
        state = state * role["support_active"].float().unsqueeze(-1)
        support_states.append(state)
        expected_disc = model._propagate(
            previous, src, dst, role["discrepant_norm_weight"]
        )
        expected_disc = expected_disc * role["discrepant_active"].float().unsqueeze(-1)
        disc_states.append(expected_disc)
        signed.append(
            (previous - expected_disc) * role["discrepant_active"].float().unsqueeze(-1)
        )
        previous = expected_disc
    _close(result["support_states"], torch.stack(support_states))
    _close(result["discrepant_states"], torch.stack(disc_states))
    _close(result["discrepant_signed_states"], torch.stack(signed))
    _close(
        result["support_trajectory"],
        torch.stack([model._active_rms_norm(v, role["support_active"]) for v in support_states]),
    )
    _close(
        result["discrepant_smooth_trajectory"],
        torch.stack([model._active_rms_norm(v, role["discrepant_active"]) for v in disc_states]),
    )
    _close(
        result["discrepant_signed_trajectory"],
        torch.stack([model._active_rms_norm(v, role["discrepant_active"]) for v in signed]),
    )


def test_discrepant_signed_first_and_second_order_differences():
    model = Model(_cfg("F0_raw"), DATA_INFO)
    prior = _features(89)
    src, dst, raw, _, _ = model._normalized_operator(_graph(), 6, torch.float32)
    role = model._role_partition(model._split_modalities(_features(97))["visual"], src, dst, raw)
    result = model._role_trajectories(prior, src, dst, role)
    p1 = result["discrepant_states"][0]
    p2 = result["discrepant_states"][1]
    active = role["discrepant_active"].float().unsqueeze(-1)
    d1 = (prior - p1) * active
    d2 = (p1 - p2) * active
    _close(result["discrepant_signed_states"][0], d1)
    _close(result["discrepant_signed_states"][1], d2)


def test_profiles_use_one_shared_alpha_bank_and_exact_variant_formulas():
    x, graph = _features(101), _graph()
    for variant in VARIANTS:
        model = Model(_cfg(variant), DATA_INFO).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, graph, return_details=True)
        for modality in ("text", "visual"):
            item = info["details"][modality]
            alpha = item["alpha"]
            _close(item["raw_profiles"], torch.einsum("mk,knd->mnd", alpha, item["raw_trajectory"]))
            _close(item["support_profiles"], torch.einsum("mk,knd->mnd", alpha, item["support_trajectory"]))
            _close(item["discrepant_smooth_profiles"], torch.einsum("mk,knd->mnd", alpha, item["discrepant_smooth_trajectory"]))
            _close(item["discrepant_signed_profiles"], torch.einsum("mk,knd->mnd", alpha, item["discrepant_signed_trajectory"]))
            lam = item["role_mix"].view(4, 1, 1)
            if variant == "F0_raw":
                _close(item["expert_inputs"], item["raw_profiles"])
            elif variant == "F1_support":
                _close(item["expert_inputs"], item["support_profiles"])
            elif variant == "F2_role_dual_smooth":
                _close(item["expert_inputs"], (1.0 - lam) * item["support_profiles"] + lam * item["discrepant_smooth_profiles"])
            else:
                _close(item["expert_inputs"], (1.0 - lam) * item["support_profiles"] + lam * item["discrepant_signed_profiles"])
            assert "alpha_support" not in item and "alpha_discrepant" not in item
    assert not any("alpha_support" in name or "alpha_discrepant" in name for name in Model(_cfg("F0_raw"), DATA_INFO).state_dict())


def test_role_mix_is_exact_point_one_shared_vector_and_variant_trainability():
    for variant in VARIANTS:
        model = Model(_cfg(variant), DATA_INFO)
        assert model.role_mix_raw.shape == (4,)
        assert torch.equal(torch.sigmoid(model.role_mix_raw), torch.full((4,), 0.1))
        assert model.role_mix_raw.requires_grad == (variant in {"F2_role_dual_smooth", "F3_role_functional"})
        assert not model.context_mix_raw.requires_grad
        assert not model.alpha_effective_raw.requires_grad
        assert not hasattr(model, "role_mix_text_raw")
        assert not hasattr(model, "role_mix_visual_raw")


def test_role_mix_has_no_weight_decay_and_parameter_count_pairs_match():
    counts = {}
    for variant in VARIANTS:
        model = Model(_cfg(variant), DATA_INFO)
        counts[variant] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        classifier = torch.nn.Linear(model.out_dim, 3)
        optimizer = build_optimizer(
            list(model.parameters()) + list(classifier.parameters()), _cfg(variant), model=model
        )
        role_id = id(model.role_mix_raw)
        group = next(g for g in optimizer.param_groups if any(id(p) == role_id for p in g["params"]))
        assert group["weight_decay"] == 0.0
        if variant in {"F0_raw", "F1_support"}:
            assert not model.role_mix_raw.requires_grad
    assert counts["F0_raw"] == counts["F1_support"]
    assert counts["F2_role_dual_smooth"] == counts["F3_role_functional"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_role_mix_perturbation_affects_only_trainable_dual_variants(variant):
    model = Model(_cfg(variant), DATA_INFO).eval()
    with torch.no_grad():
        for p in model.selection_head.parameters():
            p.zero_()
        model.selection_head.bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        before = model(_features(107), _graph())[0].clone()
        model.role_mix_raw.add_(0.6)
        after = model(_features(107), _graph())[0]
    if variant in {"F0_raw", "F1_support"}:
        assert torch.equal(before, after)
    else:
        assert not torch.equal(before, after)


def test_all_variants_static_route_strength_top2_and_active_load_balance_match_v3a():
    x, graph = _features(109), _graph()
    for variant in VARIANTS:
        model = Model(_cfg(variant), DATA_INFO).eval()
        _, _, _, aux, info = model(x, graph, return_details=True)
        for modality in ("text", "visual"):
            route = info["router"][modality]
            assert torch.equal(route["selection_logits"], route["selection_logits"][:1].expand_as(route["selection_logits"]))
            assert torch.equal(route["strength"], route["strength"][:1].expand_as(route["strength"]))
            assert torch.equal((route["route_weights"] > 0).sum(-1), torch.full((6,), 2))
            assert torch.allclose(route["route_weights"].sum(-1), torch.ones(6), atol=1.0e-7, rtol=0.0)
            active = info["active_nodes"]
            importance = route["dense_probs"][active].mean(dim=0)
            share = route["selected_mask"][active].float().mean(dim=0) / 2
            expected_balance = 4.0 * (importance * share).sum()
            _close(route["importance"], importance)
            _close(route["selection_share"], share)
            _close(route["balance"], expected_balance)
        _close(aux, 0.01 * 0.5 * (info["router"]["text"]["balance"] + info["router"]["visual"]["balance"]))


def test_trajectory_order_is_fixed_to_four():
    cfg = _cfg("F3_role_functional")
    cfg.model.trajectory_order = 3
    with pytest.raises(ValueError, match="trajectory_order"):
        Model(cfg, DATA_INFO)


def test_role_partition_tensors_have_no_gradient_path_to_input():
    model = Model(_cfg("F3_role_functional"), DATA_INFO)
    x = _features(113).requires_grad_()
    src, dst, raw, _, _ = model._normalized_operator(_graph(), 6, torch.float32)
    role = model._role_partition(model._split_modalities(x)["text"], src, dst, raw)
    assert all(not role[key].requires_grad for key in ("semantic_features", "semantic_cosine", "local_mean", "role_score", "support_norm_weight", "discrepant_norm_weight"))
