from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

import src.models.imosi_mag_v0 as imosi_module
from src.models.imosi_mag_v0 import Model as IMoSIModel
from src.models.pcrr_mag_v1 import Model as PCRRModel
from src.tasks.common import resolve_num_neighbors
from src.tasks.nc import _should_evaluate_test


NODES = 13
DATA_INFO = {"input_dim": 7, "text_dim": 4, "visual_dim": 3, "num_nodes": NODES}
MODE_PROBS = (0.6, 0.2, 0.2)


def cfg(variant: str, name: str = "imosi_mag_v0", seed: int = 42):
    return OmegaConf.create(
        {
            "seed": seed,
            "model": {
                "name": name,
                "variant": variant,
                "hidden_dim": 256,
                "num_layers": 3,
                "dropout": 0.2,
                "diffusion_add_self_loops": True,
                "eps": 1e-8,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "response_rank": 64,
                "router_rank": 64,
                "initial_mode_probs": list(MODE_PROBS),
                "pair_rank": 64,
            },
        }
    )


def graph(n: int = NODES):
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def features(n: int = NODES):
    return torch.randn((n, 7), generator=torch.Generator().manual_seed(771))


def make_imosi(variant="base", seed=42):
    torch.manual_seed(seed)
    return IMoSIModel(cfg(variant, seed=seed), DATA_INFO)


def make_pcrr(seed=42):
    torch.manual_seed(seed)
    pcrr_cfg = cfg("base", name="pcrr_mag_v1", seed=seed)
    pcrr_cfg.model.pair_rank = 64
    return PCRRModel(pcrr_cfg, DATA_INFO)


def test_modality_split_respects_declared_text_then_visual_widths():
    model = make_imosi()
    split = model._split_modalities(features())
    assert torch.equal(split["text"], features()[:, :4])
    assert torch.equal(split["visual"], features()[:, 4:])


def test_rgd_propagation_and_gpr_match_pcrr_v1():
    imosi = make_imosi("base").eval()
    pcrr = make_pcrr().eval()
    imosi_state = imosi.state_dict()
    pcrr_state = pcrr.state_dict()
    pcrr.load_state_dict({key: imosi_state[key] for key in pcrr_state}, strict=True)
    _, _, _, _, imosi_info = imosi(features(), graph(), return_details=True)
    _, _, _, _, pcrr_info = pcrr(features(), graph(), return_details=True)
    a = imosi_info["details"]["modalities"]
    b = pcrr_info["details"]["modalities"]
    for modality in ("text", "visual"):
        assert torch.equal(a[modality]["prior"], b[modality]["prior"])
        assert torch.equal(a[modality]["raw_states"], b[modality]["raw_states"])
        assert torch.equal(a[modality]["proposal"], b[modality]["proposal"])
        assert torch.equal(a[modality]["embedding"], b[modality]["embedding"])


def test_base_eval_output_regresses_to_pcrr_v1_base():
    imosi = make_imosi("base", 187).eval()
    pcrr = make_pcrr(187).eval()
    imosi_state = imosi.state_dict()
    pcrr.load_state_dict({key: imosi_state[key] for key in pcrr.state_dict()}, strict=True)
    expected = pcrr(features(), graph())[0]
    actual = imosi(features(), graph())[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_shared_rgdfusionpair_tensors_match_pcrr_v1_bitwise():
    imosi = make_imosi("base", 83)
    pcrr = make_pcrr(83)
    old, new = pcrr.state_dict(), imosi.state_dict()
    shared = set(old) & set(new)
    for key in shared:
        assert torch.equal(old[key], new[key]), key
    assert {key for key in old if key.startswith("projectors.")} <= shared
    assert {key for key in old if key.startswith("fusion_")} <= shared
    assert {"pair_norm.weight", "pair_down.weight", "pair_up.weight"} <= shared


def test_f0_initialization_restores_global_cpu_rng_and_downstream_head_rng():
    torch.manual_seed(2301)
    pcrr = PCRRModel(cfg("base", name="pcrr_mag_v1", seed=42), DATA_INFO)
    pcrr_rng = torch.random.get_rng_state()
    pcrr_head = torch.nn.Linear(256, 7)
    pcrr_head_state = {k: v.detach().clone() for k, v in pcrr_head.state_dict().items()}

    torch.manual_seed(2301)
    imosi = IMoSIModel(cfg("base", seed=42), DATA_INFO)
    imosi_rng = torch.random.get_rng_state()
    imosi_head = torch.nn.Linear(256, 7)
    assert torch.equal(pcrr_rng, imosi_rng)
    for key, value in pcrr_head_state.items():
        assert torch.equal(value, imosi_head.state_dict()[key]), key


def test_pair_features_have_target_source_absdiff_product_order():
    target = torch.tensor([[1.0, -2.0]])
    source = torch.tensor([[4.0, 3.0]])
    expected = torch.tensor([[1.0, -2.0, 4.0, 3.0, 3.0, 5.0, 4.0, -6.0]])
    assert torch.equal(IMoSIModel._pair_features(target, source), expected)


def test_self_mode_uses_target_as_source_and_paired_mode_uses_other_modality():
    model = make_imosi("global_mix").eval()
    target = torch.randn(5, 256)
    prior = torch.randn(5, 256)
    other = torch.randn(5, 256)
    seen = []
    hook = model.pair_norm.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    model._interaction_chunk(target, prior, other, 0, adaptive_router=False)
    hook.remove()
    assert len(seen) == 2
    assert torch.equal(seen[0], IMoSIModel._pair_features(target, target))
    assert torch.equal(seen[1], IMoSIModel._pair_features(target, other))


def test_pair_encoder_is_shared_but_self_and_pair_heads_are_distinct():
    model = make_imosi("global_mix")
    assert model.pair_norm is not None and model.pair_down is not None
    assert model.self_up is not model.pair_up
    assert [name for name, _ in model.named_modules() if name in {"pair_norm", "pair_down", "self_up", "pair_up"}] == [
        "pair_norm", "pair_down", "pair_up", "self_up"
    ]


def test_self_pair_and_router_outputs_are_exactly_zero_initialized():
    model = make_imosi("adaptive_mix")
    for layer in (model.self_up, model.pair_up, model.router_out):
        assert torch.count_nonzero(layer.weight) == 0
        assert torch.count_nonzero(layer.bias) == 0


def test_modality_specific_base_logits_start_at_declared_probabilities():
    model = make_imosi("global_mix")
    expected = torch.tensor(MODE_PROBS).expand(2, -1)
    torch.testing.assert_close(torch.softmax(model.mode_base_logits, dim=-1), expected, rtol=0, atol=1e-7)


def test_adaptive_initial_node_logits_equal_global_logits_and_routes():
    global_model = make_imosi("global_mix").eval()
    adaptive_model = make_imosi("adaptive_mix").eval()
    adaptive_model.load_state_dict(global_model.state_dict(), strict=True)
    global_details = global_model(features(), graph(), return_details=True)[4]["details"]["routing"]
    adaptive_details = adaptive_model(features(), graph(), return_details=True)[4]["details"]["routing"]
    for modality in ("text", "visual"):
        assert torch.count_nonzero(adaptive_details[modality]["node_delta_logits"]) == 0
        assert torch.equal(global_details[modality]["mode_probs"], adaptive_details[modality]["mode_probs"])


def test_initial_b_g_m_eval_outputs_are_elementwise_identical_and_match_rgd():
    models = [make_imosi(variant).eval() for variant in ("base", "global_mix", "adaptive_mix")]
    outputs = [model(features(), graph())[0] for model in models]
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])
    torch.testing.assert_close(outputs[0], make_pcrr().eval()(features(), graph())[0], rtol=0, atol=1e-6)


def test_initial_b_g_m_train_outputs_match_after_rng_reset():
    models = [make_imosi(variant).train() for variant in ("base", "global_mix", "adaptive_mix")]
    rng_state = torch.random.get_rng_state()
    outputs = []
    for model in models:
        torch.random.set_rng_state(rng_state)
        outputs.append(model(features(), graph())[0])
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_first_optimizer_step_gives_self_and_pair_heads_finite_nonzero_gradients():
    model = make_imosi("adaptive_mix").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    probe = torch.randn((NODES, 256), generator=torch.Generator().manual_seed(19))
    optimizer.zero_grad(set_to_none=True)
    (model(features(), graph())[0] * probe).sum().backward()
    for layer in (model.self_up, model.pair_up):
        grad = layer.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    optimizer.step()


def test_router_out_gets_finite_nonzero_gradient_after_expert_residuals_move():
    model = make_imosi("adaptive_mix").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    probe = torch.randn((NODES, 256), generator=torch.Generator().manual_seed(29))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model(features(), graph())[0] * probe).sum().backward()
        optimizer.step()
    grad = model.router_out.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_global_mix_routes_are_constant_within_each_modality():
    model = make_imosi("global_mix").eval()
    routes = model(features(), graph(), return_details=True)[4]["details"]["routing"]
    for modality in ("text", "visual"):
        expected = routes[modality]["mode_base_probs"].expand(NODES, -1)
        assert torch.equal(routes[modality]["mode_probs"], expected)


def test_nonzero_adaptive_router_can_create_node_specific_routes():
    model = make_imosi("adaptive_mix").eval()
    with torch.no_grad():
        model.router_out.weight.normal_(0.0, 0.05)
        model.router_out.bias.copy_(torch.tensor([0.2, -0.1, 0.0]))
    routes = model(features(), graph(), return_details=True)[4]["details"]["routing"]
    for modality in ("text", "visual"):
        assert routes[modality]["mode_probs"].std(dim=0).max() > 0
        assert routes[modality]["node_delta_logits"].std(dim=0).max() > 0


def test_mode_probabilities_are_finite_nonnegative_and_sum_to_one():
    model = make_imosi("adaptive_mix").eval()
    routes = model(features(), graph(), return_details=True)[4]["details"]["routing"]
    for modality in ("text", "visual"):
        probs = routes[modality]["mode_probs"]
        assert torch.isfinite(probs).all()
        assert torch.all(probs >= 0)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(NODES), rtol=0, atol=2e-7)


def test_mode_mixture_equals_weighted_sum_of_preserve_self_and_pair_experts():
    model = make_imosi("adaptive_mix").eval()
    with torch.no_grad():
        model.self_up.weight.normal_(0, 0.01)
        model.pair_up.weight.normal_(0, 0.01)
        model.router_out.weight.normal_(0, 0.01)
    target = torch.randn(9, 256)
    prior = torch.randn(9, 256)
    other = torch.randn(9, 256)
    refined, probs, _, delta_self, delta_pair = model._interaction_chunk(
        target, prior, other, 1, adaptive_router=True
    )
    reference = (
        probs[:, 0:1] * target
        + probs[:, 1:2] * (target + delta_self)
        + probs[:, 2:3] * (target + delta_pair)
    )
    torch.testing.assert_close(refined, reference, rtol=1e-6, atol=1e-6)


def test_no_post_refinement_layer_norm_is_added():
    model = make_imosi("adaptive_mix")
    names = dict(model.named_modules())
    assert "refinement_norm" not in names
    assert "post_refinement_norm" not in names
    assert isinstance(model.router_norm, torch.nn.LayerNorm)
    assert isinstance(model.pair_norm, torch.nn.LayerNorm)


def test_b_g_m_parameter_count_and_state_dict_layout_match():
    models = [make_imosi(name) for name in ("base", "global_mix", "adaptive_mix")]
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert len(set(counts)) == 1
    keys = [list(model.state_dict()) for model in models]
    shapes = [[tuple(t.shape) for t in model.state_dict().values()] for model in models]
    assert all(keys[0] == values for values in keys[1:])
    assert all(shapes[0] == values for values in shapes[1:])


def test_same_seed_b_g_m_named_tensors_are_bitwise_identical():
    models = [make_imosi(name, 55) for name in ("base", "global_mix", "adaptive_mix")]
    states = [model.state_dict() for model in models]
    for key in states[0]:
        assert all(torch.equal(states[0][key], state[key]) for state in states[1:]), key


def test_chunked_mode_refinement_matches_single_full_chunk(monkeypatch):
    model = make_imosi("adaptive_mix").eval()
    with torch.no_grad():
        model.self_up.weight.normal_(0, 0.01)
        model.pair_up.weight.normal_(0, 0.01)
        model.router_out.weight.normal_(0, 0.01)
    monkeypatch.setattr(imosi_module, "_MODE_CHUNK_SIZE", 3)
    chunked = model(features(), graph())[0]
    monkeypatch.setattr(imosi_module, "_MODE_CHUNK_SIZE", NODES + 1)
    full = model(features(), graph())[0]
    torch.testing.assert_close(chunked, full, rtol=0, atol=1e-6)


def test_aux_loss_is_exactly_zero():
    for variant in ("base", "global_mix", "adaptive_mix"):
        aux_loss = make_imosi(variant).eval()(features(), graph())[3]
        assert aux_loss.shape == torch.Size([]) and aux_loss.item() == 0.0


def test_gradients_are_finite():
    model = make_imosi("adaptive_mix").train()
    model(features(), graph())[0].square().mean().backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_inference_matches_eval_forward():
    model = make_imosi("adaptive_mix").eval()
    expected = model(features(), graph())[0].detach().cpu()
    actual = model.inference(features(), graph(), device=torch.device("cpu"))
    assert torch.equal(actual, expected)


def test_model_exposes_three_hop_lp_sampler_depth():
    model = make_imosi()
    assert model.requires_full_lp_sampler_depth is True
    task_cfg = OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    assert resolve_num_neighbors(task_cfg) == [5, 5, 5]


def test_existing_development_no_test_gate_blocks_test_evaluation():
    cfg_value = OmegaConf.create({"task": {"development_no_test": True, "evaluate_test": False}})
    assert _should_evaluate_test(cfg_value) is False
    cfg_value.task.evaluate_test = True
    with pytest.raises(ValueError, match="requires task.evaluate_test=false"):
        _should_evaluate_test(cfg_value)


def test_globalize_router_keeps_global_mode_logits_for_adaptive_variant():
    model = make_imosi("adaptive_mix").eval()
    with torch.no_grad():
        model.self_up.weight.normal_(0, 0.01)
        model.pair_up.weight.normal_(0, 0.01)
        model.router_out.weight.normal_(0, 0.1)
    normal = model(features(), graph(), return_details=True)[4]["details"]["routing"]
    globalized = model(features(), graph(), intervention="globalize_router", return_details=True)[4]["details"]["routing"]
    for modality in ("text", "visual"):
        assert torch.equal(globalized[modality]["mode_probs"], globalized[modality]["mode_base_probs"].expand(NODES, -1))
        assert normal[modality]["mode_probs"].std(dim=0).max() > 0


def test_router_feature_vector_has_declared_six_block_order():
    model = make_imosi("adaptive_mix")
    target, prior, other = (torch.randn(4, 256) for _ in range(3))
    features_value = model._router_features(target, prior, other)
    expected = torch.cat(
        [target, prior, target - prior, other, (target - other).abs(), target * other], dim=-1
    )
    assert features_value.shape == (4, 1536)
    assert torch.equal(features_value, expected)


def test_router_layer_dimensions_match_1536_to_64_to_3_specification():
    model = make_imosi("adaptive_mix")
    assert tuple(model.router_norm.normalized_shape) == (1536,)
    assert tuple(model.router_down.weight.shape) == (64, 1536)
    assert tuple(model.router_out.weight.shape) == (3, 64)


def test_base_variant_is_preserve_only_in_routing_details():
    model = make_imosi("base").eval()
    output, _, _, _, info = model(features(), graph(), return_details=True)
    for modality in ("text", "visual"):
        probs = info["details"]["routing"][modality]["mode_probs"]
        assert torch.equal(probs[:, 0], torch.ones(NODES))
        assert torch.count_nonzero(probs[:, 1:]) == 0
    pcrr = make_pcrr().eval()
    torch.testing.assert_close(output, pcrr(features(), graph())[0], rtol=0, atol=1e-6)


def test_globalized_adaptive_checkpoint_matches_global_mix_forward():
    adaptive = make_imosi("adaptive_mix").eval()
    global_mix = make_imosi("global_mix").eval()
    global_mix.load_state_dict(adaptive.state_dict(), strict=True)
    with torch.no_grad():
        adaptive.router_out.weight.normal_(0, 0.08)
        global_mix.load_state_dict(adaptive.state_dict(), strict=True)
    actual = adaptive(features(), graph(), intervention="globalize_router")[0]
    expected = global_mix(features(), graph())[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_router_never_receives_more_than_one_mode_chunk(monkeypatch):
    model = make_imosi("adaptive_mix").eval()
    monkeypatch.setattr(imosi_module, "_MODE_CHUNK_SIZE", 4)
    observed = []
    hook = model.router_norm.register_forward_pre_hook(
        lambda _module, args: observed.append(tuple(args[0].shape))
    )
    model(features(), graph())
    hook.remove()
    assert observed and max(shape[0] for shape in observed) <= 4
    assert all(shape[1] == 1536 for shape in observed)


def test_effective_expert_contribution_stats_match_chunk_tensors():
    model = make_imosi("adaptive_mix").eval()
    with torch.no_grad():
        model.self_up.weight.normal_(0, 0.02)
        model.pair_up.weight.normal_(0, 0.02)
        model.router_out.weight.normal_(0, 0.02)
    target, prior, other = (torch.randn(7, 256) for _ in range(3))
    _, probs, node_delta, delta_self, delta_pair = model._interaction_chunk(
        target, prior, other, 0, adaptive_router=True
    )
    stats = model._new_mode_stats(target)
    model._accumulate_mode_stats(
        stats, target, delta_self, delta_pair, probs, node_delta,
        torch.softmax(model.mode_base_logits[0], dim=-1),
    )
    result = model._finish_mode_stats(stats)
    effective_self = probs[:, 1:2] * delta_self
    effective_pair = probs[:, 2:3] * delta_pair
    torch.testing.assert_close(result["rms_effective_self"], effective_self.square().mean().sqrt())
    torch.testing.assert_close(result["rms_effective_pair"], effective_pair.square().mean().sqrt())
    torch.testing.assert_close(result["effective_self_to_target_rms"], effective_self.square().mean().sqrt() / target.square().mean().sqrt())
