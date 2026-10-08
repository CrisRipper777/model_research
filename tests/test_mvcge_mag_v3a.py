from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.mvcge_mag_v22 import Model as V22Model
from src.models.mvcge_mag_v3a import Model
from src.tasks.common import build_optimizer


VARIANTS = (
    "C0_raw",
    "C1_effective",
    "C2_dual_shared_profile",
    "C3_dual_context_profile",
)
DATA_INFO = {
    "input_dim": 6,
    "text_dim": 3,
    "visual_dim": 3,
    "num_nodes": 6,
    "num_classes": 3,
}


def _cfg(variant: str, *, dropout: float = 0.0, trajectory_order: int = 4, name: str = "mvcge_mag_v3a"):
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
                "strength_init": -1.15,
                "context_mix_init": 0.10,
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


def _v22_cfg():
    cfg = _cfg("R0_modality_static", name="mvcge_mag_v22")
    del cfg.model.context_mix_init
    del cfg.model.weight_decay
    del cfg.model.lr
    del cfg.task
    return cfg


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


def _features(seed: int = 17):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(6, 6, generator=generator)


def _assert_same(left: torch.Tensor, right: torch.Tensor, name: str, atol: float = 1.0e-7):
    assert torch.allclose(left, right, atol=atol, rtol=0.0), (
        name,
        float((left - right).abs().max()) if left.numel() else 0.0,
    )


def _common_state(v22: V22Model, v3a: Model):
    old, new = v22.state_dict(), v3a.state_dict()
    for key, value in old.items():
        assert key in new, key
        assert torch.equal(value, new[key]), key


def test_all_variants_forward_backward_and_diagnostics_are_finite():
    x = _features()
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Model(_cfg(variant, dropout=0.2), DATA_INFO).train()
        z, _, _, aux, info = model(x, _path_graph(), return_details=True)
        assert z.shape == (6, 256)
        assert torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        for modality in ("text", "visual"):
            item = info["details"][modality]
            for key in (
                "prior", "raw_trajectory", "effective_trajectory", "raw_profiles",
                "effective_shared_profiles", "effective_context_profiles", "mixed_profiles",
                "expert_outputs", "selection_logits", "route_weights", "strength",
                "semantic_similarity", "semantic_weight", "effective_degree",
            ):
                assert torch.isfinite(item[key]).all(), (variant, modality, key)
        (z.square().mean() + aux).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_same_seed_common_initialization_and_classifier_rng_match_across_variants():
    snapshots = {}
    for variant in VARIANTS:
        torch.manual_seed(913)
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
def test_downstream_classifier_rng_matches_v22_r0(variant):
    torch.manual_seed(887)
    old = V22Model(_v22_cfg(), DATA_INFO)
    old_classifier = torch.nn.Linear(old.out_dim, 3)
    torch.manual_seed(887)
    new = Model(_cfg(variant), DATA_INFO)
    new_classifier = torch.nn.Linear(new.out_dim, 3)
    for key, value in old_classifier.state_dict().items():
        assert torch.equal(value, new_classifier.state_dict()[key]), (variant, key)


def test_c0_exact_full_forward_regression_to_v22_r0():
    seed = 913
    torch.manual_seed(seed)
    old = V22Model(_v22_cfg(), DATA_INFO).eval()
    torch.manual_seed(seed)
    new = Model(_cfg("C0_raw"), DATA_INFO).eval()
    _common_state(old, new)
    x = _features(71)
    graph = _path_graph(loops=True)
    with torch.no_grad():
        z0, _, _, aux0, info0 = old(x, graph, return_details=True)
        z1, _, _, aux1, info1 = new(x, graph, return_details=True)
    _assert_same(z0, z1, "fused z")
    _assert_same(aux0, aux1, "aux loss")
    for modality in ("text", "visual"):
        before = info0["details"][modality]
        after = info1["details"][modality]
        for old_key, new_key in (
            ("prior", "prior"),
            ("basis", "raw_trajectory"),
            ("alpha", "alpha_raw"),
            ("expert_inputs", "raw_profiles"),
            ("expert_outputs", "expert_outputs"),
            ("selection_logits", "selection_logits"),
            ("route_weights", "route_weights"),
            ("strength", "strength"),
            ("output", "output"),
            ("mixture", "mixture"),
            ("scaled_mixture", "scaled_mixture"),
        ):
            _assert_same(before[old_key], after[new_key], f"{modality}.{old_key}")
    assert info1["input_self_loops_removed"] == 3


def test_physical_operator_removes_input_self_loops_and_adds_none():
    src, dst, _, active, loops = Model._normalized_operator(
        _path_graph(loops=True), 6, torch.float32
    )
    assert loops == 3
    assert not torch.any(src == dst)
    assert torch.equal(active, torch.tensor([True, True, True, True, True, False]))


def test_isolated_node_has_zero_raw_effective_trajectory_and_correction():
    model = Model(_cfg("C2_dual_shared_profile"), DATA_INFO).eval()
    with torch.no_grad():
        _, _, _, _, info = model(_features(), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        item = info["details"][modality]
        assert torch.count_nonzero(item["raw_trajectory"][:, 5]) == 0
        assert torch.count_nonzero(item["effective_trajectory"][:, 5]) == 0
        assert torch.count_nonzero(item["expert_outputs"][:, 5]) == 0
        assert torch.count_nonzero(item["scaled_mixture"][5]) == 0
        assert torch.equal(item["output"][5], item["prior"][5])


@pytest.mark.parametrize(
    ("right", "expected_cosine", "expected_weight"),
    [
        ([2.0, -1.0, 0.5], 1.0, 1.0),
        ([1.0, 2.0, 0.0], 0.0, 0.5),
        ([-2.0, 1.0, -0.5], -1.0, 0.0),
        ([0.0, 0.0, 0.0], 0.0, 0.5),
    ],
)
def test_semantic_edge_scores_for_identical_orthogonal_opposite_and_zero_vectors(
    right, expected_cosine, expected_weight
):
    raw = torch.tensor([[2.0, -1.0, 0.5], right], dtype=torch.float32)
    normalized = Model._safe_normalize_raw_features(raw, 1.0e-8)
    src, dst = torch.tensor([0]), torch.tensor([1])
    cosine = Model._semantic_edge_similarity(normalized, src, dst)
    weight = (0.5 * (1.0 + cosine)).clamp(0.0, 1.0)
    assert float(cosine[0]) == pytest.approx(expected_cosine, abs=1.0e-7)
    assert float(weight[0]) == pytest.approx(expected_weight, abs=1.0e-7)
    assert torch.isfinite(weight).all() and torch.all((weight >= 0) & (weight <= 1))


def test_effective_weighted_degree_formula_and_subunit_degree_regression():
    model = Model(_cfg("C0_raw"), DATA_INFO)
    raw = torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
    src, dst = torch.tensor([0, 1]), torch.tensor([1, 0])
    op = model._effective_operator(raw, src, dst)
    assert torch.allclose(op["semantic_weight"], torch.tensor([0.8, 0.8]), atol=1e-7)
    assert torch.allclose(op["effective_degree"], torch.tensor([0.8, 0.8]), atol=1e-7)
    assert torch.allclose(op["norm_weight"], torch.ones(2), atol=1e-7)
    assert torch.isfinite(op["norm_weight"]).all()


def test_effective_degree_is_incoming_weight_sum_and_normalized_edges_are_finite():
    model = Model(_cfg("C0_raw"), DATA_INFO)
    raw = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([2, 2, 1])
    op = model._effective_operator(raw, src, dst)
    expected = torch.zeros(3).index_add(0, dst, op["semantic_weight"])
    assert torch.equal(op["effective_degree"], expected)
    assert torch.isfinite(op["norm_weight"]).all()


def test_effective_operator_is_detached_from_training_parameters_and_projectors():
    model = Model(_cfg("C2_dual_shared_profile"), DATA_INFO).eval()
    x = _features().requires_grad_()
    graph = _path_graph()
    before = model(x, graph, return_details=True)[4]["details"]["text"]["effective_operator_weight"].clone()
    with torch.no_grad():
        for parameter in model.projectors.parameters():
            parameter.add_(3.0)
        for parameter in model.router_projectors.parameters():
            parameter.mul_(0.0)
    after = model(x, graph, return_details=True)[4]["details"]["text"]["effective_operator_weight"]
    assert torch.equal(before, after)
    assert not after.requires_grad


def test_text_and_visual_effective_operators_use_only_their_own_raw_features():
    model = Model(_cfg("C1_effective"), DATA_INFO).eval()
    x = _features(19)
    graph = _path_graph()
    first = model(x, graph, return_details=True)[4]["details"]
    changed_text = x.clone()
    changed_text[:, :3] = torch.roll(changed_text[:, :3], shifts=1, dims=0) * 2.0
    second = model(changed_text, graph, return_details=True)[4]["details"]
    assert not torch.equal(first["text"]["effective_operator_weight"], second["text"]["effective_operator_weight"])
    assert torch.equal(first["visual"]["effective_operator_weight"], second["visual"]["effective_operator_weight"])

    changed_visual = x.clone()
    changed_visual[:, 3:] = torch.roll(changed_visual[:, 3:], shifts=2, dims=0) * 3.0
    third = model(changed_visual, graph, return_details=True)[4]["details"]
    assert torch.equal(first["text"]["effective_operator_weight"], third["text"]["effective_operator_weight"])
    assert not torch.equal(first["visual"]["effective_operator_weight"], third["visual"]["effective_operator_weight"])


def test_c0_profile_is_raw_and_c1_profile_is_effective_shared_alpha():
    x, graph = _features(), _path_graph()
    for variant, expected_key in (("C0_raw", "raw_profiles"), ("C1_effective", "effective_shared_profiles")):
        model = Model(_cfg(variant), DATA_INFO).eval()
        item = model(x, graph, return_details=True)[4]["details"]["text"]
        _assert_same(item["expert_inputs"], item[expected_key], variant)


def test_c2_uses_shared_alpha_and_exact_convex_context_interpolation():
    model = Model(_cfg("C2_dual_shared_profile"), DATA_INFO).eval()
    with torch.no_grad():
        _, _, _, _, info = model(_features(), _path_graph(), return_details=True)
    for modality in ("text", "visual"):
        item = info["details"][modality]
        lam = item["context_mix"].view(4, 1, 1)
        _assert_same(item["alpha"], item["alpha_effective"], f"{modality} shared alpha")
        _assert_same(
            item["mixed_profiles"],
            (1.0 - lam) * item["raw_profiles"] + lam * item["effective_shared_profiles"],
            f"{modality} convex mix",
        )


def test_c3_uses_raw_and_effective_alpha_banks_separately():
    model = Model(_cfg("C3_dual_context_profile"), DATA_INFO).eval()
    info = model(_features(), _path_graph(), return_details=True)[4]
    for modality in ("text", "visual"):
        item = info["details"][modality]
        raw_expected = torch.einsum("mk,knd->mnd", item["alpha_raw"], item["raw_trajectory"])
        eff_expected = torch.einsum("mk,knd->mnd", item["alpha_effective"], item["effective_trajectory"])
        lam = item["context_mix"].view(4, 1, 1)
        _assert_same(item["raw_profiles"], raw_expected, f"{modality} raw alpha")
        _assert_same(item["effective_context_profiles"], eff_expected, f"{modality} effective alpha")
        _assert_same(item["mixed_profiles"], (1-lam)*raw_expected + lam*eff_expected, f"{modality} C3 mix")


def test_context_mix_initialization_is_shared_across_modalities_and_exactly_point_one():
    for variant in ("C2_dual_shared_profile", "C3_dual_context_profile"):
        model = Model(_cfg(variant), DATA_INFO)
        lam = torch.sigmoid(model.context_mix_raw)
        assert model.context_mix_raw.shape == (4,)
        assert torch.equal(lam, torch.full((4,), 0.1))
        assert model.context_mix_raw is not getattr(model, "context_mix_text_raw", None)
        assert model.context_mix_raw is not getattr(model, "context_mix_visual_raw", None)


def test_c2_and_c3_initial_full_function_are_equal():
    torch.manual_seed(509)
    c2 = Model(_cfg("C2_dual_shared_profile"), DATA_INFO).eval()
    torch.manual_seed(509)
    c3 = Model(_cfg("C3_dual_context_profile"), DATA_INFO).eval()
    x, graph = _features(28), _path_graph(loops=True)
    with torch.no_grad():
        z2, _, _, aux2, info2 = c2(x, graph, return_details=True)
        z3, _, _, aux3, info3 = c3(x, graph, return_details=True)
    _assert_same(z2, z3, "C2/C3 fused")
    _assert_same(aux2, aux3, "C2/C3 aux")
    for modality in ("text", "visual"):
        a, b = info2["details"][modality], info3["details"][modality]
        for key in (
            "raw_trajectory", "effective_trajectory", "raw_profiles",
            "effective_shared_profiles", "effective_context_profiles", "mixed_profiles",
            "expert_outputs", "selection_logits", "route_weights", "strength", "output",
        ):
            _assert_same(a[key], b[key], f"{modality}.{key}")


def test_perturbing_effective_alpha_changes_c3_but_not_c2_output():
    x, graph = _features(47), _path_graph()
    c2 = Model(_cfg("C2_dual_shared_profile"), DATA_INFO).eval()
    c3 = Model(_cfg("C3_dual_context_profile"), DATA_INFO).eval()
    with torch.no_grad():
        for model in (c2, c3):
            model.selection_head.bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        before2 = c2(x, graph)[0].clone()
        before3 = c3(x, graph)[0].clone()
        c2.alpha_effective_raw[0, 0].add_(0.4)
        c3.alpha_effective_raw[0, 0].add_(0.4)
        after2 = c2(x, graph)[0]
        after3 = c3(x, graph)[0]
    assert torch.equal(before2, after2)
    assert not torch.equal(before3, after3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_context_mix_perturbation_only_changes_c2_and_c3(variant):
    model = Model(_cfg(variant), DATA_INFO).eval()
    x, graph = _features(59), _path_graph()
    with torch.no_grad():
        before = model(x, graph)[0].clone()
        model.context_mix_raw.add_(0.7)
        after = model(x, graph)[0]
    if variant in {"C0_raw", "C1_effective"}:
        assert torch.equal(before, after)
    else:
        assert not torch.equal(before, after)


def test_variant_trainability_and_legacy_router_modules_are_frozen():
    for variant in VARIANTS:
        model = Model(_cfg(variant), DATA_INFO)
        assert model.alpha_effective_raw.requires_grad == (variant == "C3_dual_context_profile")
        assert model.context_mix_raw.requires_grad == (variant in {"C2_dual_shared_profile", "C3_dual_context_profile"})
        for module in (model.evidence_scalar_projector, model.evidence_norm, model.expert_query_proj, model.expert_key_proj):
            assert all(not parameter.requires_grad for parameter in module.parameters())
        for parameter in (model.expert_key_embedding, model.node_residual_raw, model.compatibility_residual_raw):
            assert not parameter.requires_grad


def test_context_mix_raw_has_no_weight_decay_optimizer_group():
    model = Model(_cfg("C2_dual_shared_profile"), DATA_INFO)
    cfg = _cfg("C2_dual_shared_profile")
    classifier = torch.nn.Linear(model.out_dim, 3)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    context_id = id(model.context_mix_raw)
    context_groups = [group for group in optimizer.param_groups if any(id(p) == context_id for p in group["params"])]
    assert len(context_groups) == 1
    assert context_groups[0]["weight_decay"] == 0.0
    other_group = next(group for group in optimizer.param_groups if id(model.alpha_raw) in {id(p) for p in group["params"]})
    assert other_group["weight_decay"] == 0.05


def test_all_variants_have_static_same_modality_logits_strength_and_top2():
    x, graph = _features(), _path_graph()
    for variant in VARIANTS:
        model = Model(_cfg(variant), DATA_INFO).eval()
        info = model(x, graph, return_details=True)[4]
        for modality in ("text", "visual"):
            route = info["router"][modality]
            assert torch.equal(route["selection_logits"], route["selection_logits"][:1].expand_as(route["selection_logits"]))
            assert torch.equal(route["strength"], route["strength"][:1].expand_as(route["strength"]))
            assert torch.equal((route["route_weights"] > 0).sum(-1), torch.full((6,), 2))
            assert torch.allclose(route["route_weights"].sum(-1), torch.ones(6), atol=1e-7)


def test_active_only_v22_load_balance_formula_is_unchanged():
    model = Model(_cfg("C3_dual_context_profile"), DATA_INFO).eval()
    _, _, _, aux, info = model(_features(), _path_graph(), return_details=True)
    active, balances = info["active_nodes"], []
    for modality in ("text", "visual"):
        route = info["router"][modality]
        importance = route["dense_probs"][active].mean(dim=0)
        share = route["selected_mask"][active].float().mean(dim=0) / 2
        expected = 4 * (importance * share).sum()
        _assert_same(route["importance"], importance, modality + " importance")
        _assert_same(route["selection_share"], share, modality + " share")
        _assert_same(route["balance"], expected, modality + " balance")
        balances.append(expected)
    _assert_same(aux, 0.01 * 0.5 * (balances[0] + balances[1]), "aux")


def test_raw_and_effective_trajectories_are_independently_active_rms_normalized():
    model = Model(_cfg("C2_dual_shared_profile"), DATA_INFO).eval()
    info = model(_features(), _path_graph(), return_details=True)[4]
    active = info["active_nodes"]
    for modality in ("text", "visual"):
        item = info["details"][modality]
        for key in ("raw_trajectory", "effective_trajectory"):
            trajectory = item[key]
            for hop in range(4):
                rms_sq = trajectory[hop, active].square().mean(dim=0)
                assert torch.allclose(rms_sq, torch.ones_like(rms_sq), atol=1e-5, rtol=0.0)
                assert torch.count_nonzero(trajectory[hop, ~active]) == 0


def test_trajectory_order_is_fixed_to_four():
    model = Model(_cfg("C0_raw", trajectory_order=4), DATA_INFO)
    assert model.trajectory_order == 4
    with pytest.raises(ValueError, match="trajectory_order=4"):
        Model(_cfg("C0_raw", trajectory_order=3), DATA_INFO)


def test_effective_operator_has_no_gradient_path_to_raw_features():
    model = Model(_cfg("C1_effective"), DATA_INFO)
    raw = torch.randn(6, 3, requires_grad=True)
    op = model._effective_operator(raw, torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert all(not value.requires_grad for value in op.values())
