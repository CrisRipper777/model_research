from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts.run_r0_mature_response_expert_prototype import (
    SOURCE_SHA,
    audit_initialization,
    build_classifier,
    make_config,
    model_data_info,
    smooth_compatibility_regression,
)
from src.models.mature_response_r0 import (
    NUM_EXPERTS,
    PROMPT_DIM,
    TOP_K,
    VARIANTS,
    Model,
    incoming_degree,
    pairwise_prompt_cosines,
    parameter_counts,
    prompt_orthogonality_loss,
    remove_self_messages,
    self_anchored_low_high,
)
from src.models.adaptive_prop_m0 import Model as M0Model
from src.models.adaptive_prop_n1 import Model as N1Model
from src.utils.seeds import set_seed


@pytest.fixture
def info():
    return {"input_dim": 12, "text_dim": 5, "visual_dim": 7, "num_nodes": 9, "num_classes": 3}


@pytest.fixture
def graph():
    x = torch.randn(9, 12)
    # Includes physical self loops, node 8 is isolated, directions are asymmetric.
    edges = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 1, 4, 7],
        [0, 1, 1, 2, 2, 3, 4, 4, 7, 4],
    ])
    return x, edges


def cfg(variant: str, edge_chunk_size: int = 3):
    return OmegaConf.create({"model": {
        "variant": variant, "hidden_dim": 128, "dropout": 0.2,
        "edge_chunk_size": edge_chunk_size, "orth_weight": 1e-3,
    }})


def test_source_sha_and_variant_contract():
    assert SOURCE_SHA == "2fbe1f14cda0d33d72f8c8468ef9068af28f6217"
    assert VARIANTS == ("smooth_base", "bank_generic", "bank_crossmoe_generic", "bank_crossmoe_protected")
    assert NUM_EXPERTS == 3 and TOP_K == 2 and PROMPT_DIM == 32


def test_text_visual_split_and_self_edge_removal(graph, info):
    x, edge_index = graph
    model = Model(cfg("smooth_base"), info)
    text, visual = model.split_modalities(x)
    assert text.shape == (9, 5) and visual.shape == (9, 7)
    filtered, src, dst = remove_self_messages(edge_index)
    assert not bool((src == dst).any())
    assert filtered.shape[1] == edge_index.shape[1] - 2
    degree = incoming_degree(dst, x.size(0))
    assert int(degree.sum()) == src.numel()
    assert degree[8].item() == 0


def test_low_high_exact_formula_and_degree_zero(graph):
    x, edge_index = graph
    intrinsic = x[:, :4]
    _, src, dst = remove_self_messages(edge_index)
    degree = incoming_degree(dst, x.size(0))
    low, high, neighbor_mean = self_anchored_low_high(intrinsic, src, dst, degree, 2)
    brute_sum = intrinsic.new_zeros(intrinsic.shape)
    for source, target in zip(src.tolist(), dst.tolist()):
        brute_sum[target] += intrinsic[source]
    expected_low = (intrinsic + brute_sum) / (degree.to(intrinsic.dtype).unsqueeze(-1) + 1)
    assert torch.allclose(low, expected_low, atol=1e-7, rtol=1e-6)
    assert torch.allclose(high, intrinsic - low, atol=1e-7, rtol=1e-6)
    assert torch.allclose(low + high, intrinsic, atol=2e-7, rtol=1e-6)
    assert torch.equal(low[8], intrinsic[8])
    assert torch.equal(high[8], torch.zeros_like(high[8]))
    assert torch.equal(neighbor_mean[8], torch.zeros_like(neighbor_mean[8]))


def test_chunked_and_unchunked_response_equivalence(graph):
    x, edge_index = graph
    intrinsic = torch.randn(x.size(0), 128)
    _, src, dst = remove_self_messages(edge_index)
    degree = incoming_degree(dst, x.size(0))
    a = self_anchored_low_high(intrinsic, src, dst, degree, 1)
    b = self_anchored_low_high(intrinsic, src, dst, degree, 1000)
    for left, right in zip(a, b):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)


def test_transforms_are_modality_independent_and_finite(info, graph):
    model = Model(cfg("bank_generic"), info)
    x, edge_index = graph
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    assert model.low_transforms[0] is not model.low_transforms[1]
    assert model.high_transforms[0] is not model.high_transforms[1]
    for value in (*aux["low"], *aux["high"]):
        assert bool(torch.isfinite(value).all())
        assert value.shape == (9, 128)


def test_moe_shapes_topk_weights_prompts_and_nonzero_router_init(info, graph):
    model = Model(cfg("bank_crossmoe_generic"), info)
    x, edge_index = graph
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    assert len(model.cross_experts) == 2
    for modality in range(2):
        assert model.cross_prompts[modality].shape == (3, 32)
        assert model.routers[modality][0].in_features == 512
        assert model.routers[modality][-1].out_features == 3
        assert model.cross_experts[modality][0][0].in_features == 416
        assert model.cross_experts[modality][0][-1].out_features == 128
        info_row = aux["moe"][modality]
        assert info_row["expert_outputs"].shape == (9, 3, 128)
        assert info_row["top_indices"].shape == (9, 2)
        assert info_row["top_weights"].shape == (9, 2)
        assert torch.allclose(info_row["top_weights"].sum(-1), torch.ones(9), atol=1e-7)
        assert bool(torch.isfinite(info_row["router_logits"]).all())
        assert model.routers[modality][-1].weight.std() > 0
        assert not torch.equal(model.routers[modality][-1].weight, torch.zeros_like(model.routers[modality][-1].weight))
        assert bool(torch.isfinite(aux["cross"][modality]).all())


def test_prompt_orthogonality_matches_ordered_offdiagonal_formula(info):
    model = Model(cfg("bank_crossmoe_generic"), info)
    prompts = model.cross_prompts[0]
    normalized = F.normalize(prompts, dim=-1)
    expected = torch.exp(normalized @ normalized.T)
    mask = ~torch.eye(3, dtype=torch.bool)
    expected = expected[mask].mean()
    assert torch.allclose(prompt_orthogonality_loss(prompts), expected, atol=1e-7)
    assert pairwise_prompt_cosines(prompts).numel() == 6


def test_orthogonality_loss_only_active_for_crossmoe(info, graph):
    x, edge_index = graph
    for variant in ("smooth_base", "bank_generic"):
        loss = Model(cfg(variant), info)(x, edge_index)[3]
        assert loss.item() == 0.0
    for variant in ("bank_crossmoe_generic", "bank_crossmoe_protected"):
        loss = Model(cfg(variant), info)(x, edge_index)[3]
        assert torch.isfinite(loss) and loss.item() > 0


def test_generic_and_protected_composer_contract(info, graph):
    x, edge_index = graph
    generic = Model(cfg("bank_crossmoe_generic"), info).eval()
    z, _, structural, _, aux = generic(x, edge_index, return_diagnostics=True)
    assert generic.generic_composers[0][0].in_features == 512
    assert generic.generic_composers[0][0].out_features == 103
    assert generic.generic_composers[0][-1].in_features == 103
    assert aux["attention"] == [None, None]
    assert bool(aux["cross_active"])
    assert torch.count_nonzero(aux["composer_inputs"][0][:, -128:]) > 0
    for modality in range(2):
        expected_u = generic.response_residual_norms[modality](
            aux["intrinsic"][modality] + aux["composition_response"][modality]
        )
        expected_z = generic.final_residual_norms[modality](
            expected_u + generic.post_ffns[modality](expected_u)
        )
        assert torch.allclose(structural[modality], expected_z, atol=1e-6, rtol=1e-6)
    expected_final = generic.fusion(torch.cat([aux["intrinsic"][0], aux["intrinsic"][1], *structural], -1))
    assert torch.allclose(z, expected_final, atol=1e-6, rtol=1e-6)

    protected = Model(cfg("bank_crossmoe_protected"), info)
    _, _, _, _, paux = protected(x, edge_index, return_diagnostics=True)
    assert protected.protected_attention[0].embed_dim == 128
    assert protected.protected_attention[0].num_heads == 4
    for modality in range(2):
        assert paux["protected_queries"][modality].shape == (9, 1, 128)
        assert torch.equal(paux["protected_queries"][modality].squeeze(1), paux["intrinsic"][modality])
        expected_tokens = torch.stack([
            paux["low"][modality], paux["high"][modality], paux["cross"][modality]
        ], dim=1)
        assert torch.equal(paux["protected_tokens"][modality], expected_tokens)
        assert paux["attention"][modality].shape == (9, 4, 3)
        assert bool(torch.isfinite(paux["attention"][modality]).all())


def test_bank_generic_disables_cross_response_and_protected_path(info, graph):
    x, edge_index = graph
    model = Model(cfg("bank_generic"), info)
    _, _, _, orth, aux = model(x, edge_index, return_diagnostics=True)
    assert not aux["cross_active"] and orth.item() == 0
    assert torch.count_nonzero(aux["cross"][0]) == 0
    assert torch.count_nonzero(aux["composer_inputs"][0][:, -128:]) == 0
    assert aux["attention"] == [None, None]


def test_smooth_base_uses_compatibility_path_and_all_variants_have_exact_capacity(info, graph):
    x, edge_index = graph
    model = Model(cfg("smooth_base"), info)
    z, _, structural, orth, aux = model(x, edge_index, return_diagnostics=True)
    assert orth.item() == 0 and aux["attention"] == [None, None]
    for modality in range(2):
        expected = model.smooth_residual_norms[modality](
            aux["intrinsic"][modality] + aux["smooth_delta"][modality]
        )
        assert torch.allclose(structural[modality], expected, atol=1e-7)
    counts = parameter_counts(model)
    assert counts["generic_first_stage_both_modalities"] == 132302
    assert counts["protected_mha_both_modalities"] == 132096
    assert counts["generic_active_composer_both_modalities"] - counts["protected_active_composer_both_modalities"] == 206

    states = []
    for variant in VARIANTS:
        set_seed(42)
        candidate = Model(cfg(variant), info)
        states.append({key: value.detach().clone() for key, value in candidate.state_dict().items()})
        assert parameter_counts(candidate)["model_trainable"] == counts["model_trainable"]
    for state in states[1:]:
        assert all(torch.equal(states[0][key], state[key]) for key in states[0])


def test_classifier_initialization_is_seed_isolated(info):
    left = build_classifier(128, 3, 44)
    _ = torch.randn(1000)
    right = build_classifier(128, 3, 44)
    assert all(torch.equal(left.state_dict()[key], right.state_dict()[key]) for key in left.state_dict())


def test_runtime_gradient_paths_for_generic_and_protected(info, graph):
    x, edge_index = graph
    for variant, active_prefixes in (
        ("bank_crossmoe_generic", ("cross_prompts.", "cross_experts.", "routers.", "generic_composers.")),
        ("bank_crossmoe_protected", ("cross_prompts.", "cross_experts.", "routers.", "protected_attention.")),
    ):
        model = Model(cfg(variant), info)
        z, _, _, orth, _ = model(x, edge_index)
        loss = z.square().mean() + model.orth_weight * orth
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith(active_prefixes) and parameter.grad is not None:
                assert bool(torch.isfinite(parameter.grad).all())
        assert model.fusion[0].weight.grad is not None
        assert bool(torch.isfinite(model.fusion[0].weight.grad).all())
    generic = Model(cfg("bank_crossmoe_generic"), info)
    generic(x, edge_index)[0].sum().backward()
    assert all(p.grad is None for p in generic.protected_attention.parameters())
    protected = Model(cfg("bank_crossmoe_protected"), info)
    protected(x, edge_index)[0].sum().backward()
    assert all(p.grad is None for p in protected.generic_composers.parameters())


def test_response_overrides_identity_and_off_controls(info, graph):
    x, edge_index = graph
    model = Model(cfg("bank_crossmoe_protected"), info).eval()
    z, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    identity = model(
        x, edge_index,
        response_overrides={"L": tuple(v.clone() for v in aux["low"]),
                            "H": tuple(v.clone() for v in aux["high"]),
                            "X": tuple(v.clone() for v in aux["cross"])},
        routing_override=tuple((m["top_indices"].clone(), m["top_weights"].clone()) for m in aux["moe"]),
    )[0]
    assert torch.allclose(z, identity, atol=1e-7, rtol=1e-7)
    for key, values in (("L", aux["low"]), ("H", aux["high"]), ("X", aux["cross"])):
        masked = tuple(torch.zeros_like(v) for v in values)
        intervened = model(x, edge_index, response_overrides={key: masked})[0]
        assert bool(torch.isfinite(intervened).all())


def test_router_tuple_shuffle_preserves_complete_rows_and_is_node_local(info, graph):
    x, edge_index = graph
    model = Model(cfg("bank_crossmoe_protected"), info).eval()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    val = torch.tensor([2, 5, 7])
    perm = torch.tensor([1, 2, 0])
    shuffled = []
    for item in aux["moe"]:
        indices, weights = item["top_indices"].clone(), item["top_weights"].clone()
        old = torch.cat([indices[val], weights[val]], dim=-1).clone()
        indices[val] = indices[val[perm]]
        weights[val] = weights[val[perm]]
        new = torch.cat([indices[val], weights[val]], dim=-1)
        assert sorted(map(tuple, old.tolist())) == sorted(map(tuple, new.tolist()))
        shuffled.append((indices, weights))
    base = model(x, edge_index)[0]
    moved = model(x, edge_index, routing_override=tuple(shuffled))[0]
    keep = torch.tensor([0, 1, 3, 4, 6, 8])
    assert torch.equal(base[keep], moved[keep])


def test_validation_only_response_override_does_not_change_other_nodes(info, graph):
    x, edge_index = graph
    model = Model(cfg("bank_crossmoe_protected"), info).eval()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    val = torch.tensor([1, 3, 6])
    high = [value.detach().clone() for value in aux["high"]]
    for value in high:
        value[val] = 0
    base = model(x, edge_index)[0]
    changed = model(x, edge_index, response_overrides={"H": tuple(high)})[0]
    other = torch.tensor([0, 2, 4, 5, 7, 8])
    assert torch.equal(base[other], changed[other])


def test_smooth_m0_n1_forward_compatibility_regression(info, graph):
    x, edge_index = graph
    data = type("Data", (), {})()
    data.x = x; data.edge_index = edge_index; data.x_t = x[:, :5]; data.x_i = x[:, 5:]
    data.input_dim = 12; data.num_nodes = 9; data.num_classes = 3
    cfg_r0 = make_config("Movies", 42, "smooth_base", "cpu", epochs=1)
    result = smooth_compatibility_regression("Movies", 42, cfg_r0, data, torch.device("cpu"))
    assert result["status"] == "passed"
    assert result["max_abs_error"] <= 1e-5


def test_initialization_audit_and_active_capacity(info):
    cfg_r0 = make_config("Movies", 42, "smooth_base", "cpu", epochs=1)
    audit = audit_initialization("Movies", 42, info, cfg_r0)
    assert audit["all_bitwise_equal"]
    assert audit["exact_model_parameter_match"]
    assert audit["exact_classifier_parameter_match"]
    assert audit["exact_total_parameter_match"]
    counts = audit["parameter_counts"]["smooth_base"]
    assert counts["generic_first_stage_both_modalities"] == 132302
    assert counts["protected_mha_both_modalities"] == 132096


def test_topk_router_weight_override_is_valid_and_finite(info, graph):
    x, edge_index = graph
    model = Model(cfg("bank_crossmoe_generic"), info).eval()
    _, _, _, _, aux = model(x, edge_index, return_diagnostics=True)
    for item in aux["moe"]:
        assert torch.allclose(item["top_weights"].sum(-1), torch.ones(x.size(0)), atol=1e-7)
        assert int(item["top_indices"].max()) < NUM_EXPERTS
        assert int(item["top_indices"].min()) >= 0


def test_all_output_and_diagnostic_tensors_are_finite(info, graph):
    x, edge_index = graph
    for variant in VARIANTS:
        model = Model(cfg(variant), info).eval()
        z, intrinsic, structural, orth, aux = model(x, edge_index, return_diagnostics=True)
        assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(orth))
        for values in (intrinsic, structural, aux["low_raw"], aux["high_raw"], aux["low"], aux["high"], aux["cross"]):
            assert all(bool(torch.isfinite(value).all()) for value in values)


def test_configuration_fixes_orth_weight_and_full_graph_protocol():
    cfg_r0 = make_config("Movies", 42, "smooth_base", "cpu", epochs=1)
    assert not cfg_r0.task.evaluate_test
    assert cfg_r0.task.training_mode == "full_graph"
    assert cfg_r0.task.scheduler in (None, "null")
    assert cfg_r0.model.orth_weight == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        Model(OmegaConf.create({"model": {"variant": "smooth_base", "orth_weight": 1e-2}}),
              {"input_dim": 12, "text_dim": 5, "visual_dim": 7})
