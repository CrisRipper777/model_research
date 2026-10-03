from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

import src.models.pcrr_mag_v0 as pcrr_module
from src.models.orci_mag_v0 import Model as OrciModel
from src.models.pcrr_mag_v0 import Model as PCRRModel
from src.tasks.common import resolve_num_neighbors


NODES = 9
DATA_INFO = {"input_dim": 7, "text_dim": 4, "visual_dim": 3, "num_nodes": NODES}


def cfg(variant: str, seed: int = 42):
    return OmegaConf.create(
        {
            "seed": seed,
            "model": {
                "name": "pcrr_mag_v0",
                "variant": variant,
                "hidden_dim": 256,
                "num_layers": 3,
                "dropout": 0.2,
                "diffusion_add_self_loops": True,
                "eps": 1.0e-8,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "pair_rank": 64,
            },
        }
    )


def graph(n: int = NODES):
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def features(n: int = NODES):
    return torch.randn((n, 7), generator=torch.Generator().manual_seed(71))


def make_model(variant: str = "base", seed: int = 42):
    torch.manual_seed(seed)
    return PCRRModel(cfg(variant, seed), DATA_INFO)


def test_split_uses_declared_text_then_visual_widths():
    model = make_model()
    split = model._split_modalities(features())
    assert torch.equal(split["text"], features()[:, :4])
    assert torch.equal(split["visual"], features()[:, 4:])


def test_raw_states_and_gpr_composition_follow_three_hop_equations():
    model = make_model().eval()
    x = features()
    z, _, _, _, info = model(x, graph(), return_details=True)
    details = info["details"]["modalities"]
    src, dst, norm = model._normalized_operator(graph(), NODES, torch.float32)
    for modality in ("text", "visual"):
        states = details[modality]["raw_states"]
        assert torch.equal(states[:, 0], details[modality]["prior"])
        for order in range(1, 4):
            assert torch.allclose(
                states[:, order],
                model._propagate(states[:, order - 1], src, dst, norm),
                atol=0,
                rtol=0,
            )
        proposal = sum(model.effective_coefficients()[k] * states[:, k] for k in range(4))
        expected = torch.nn.functional.layer_norm(proposal, (256,), eps=1.0e-8)
        assert torch.equal(details[modality]["proposal"], proposal)
        assert torch.equal(details[modality]["embedding"], expected)
    assert z.shape == (NODES, 256)


def test_base_eval_output_regresses_to_orci_d0_base():
    torch.manual_seed(891)
    orci_cfg = cfg("base")
    orci_cfg.model.name = "orci_mag_v0"
    orci_cfg.model.variant = "base"
    d0 = OrciModel(orci_cfg, DATA_INFO).eval()
    torch.manual_seed(891)
    base = make_model("base", seed=891).eval()
    expected = d0(features(), graph())[0]
    actual = base(features(), graph())[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_pair_feature_order_is_target_source_absdiff_product():
    target = torch.tensor([[1.0, -2.0]])
    source = torch.tensor([[4.0, 3.0]])
    expected = torch.tensor([[1.0, -2.0, 4.0, 3.0, 3.0, 5.0, 4.0, -6.0]])
    assert torch.equal(PCRRModel._pair_features(target, source), expected)


def test_pair_response_modules_are_shared_by_both_directions():
    model = make_model("paired")
    seen = {"norm": 0, "down": 0, "up": 0}
    hooks = [
        model.pair_norm.register_forward_hook(lambda *_: seen.__setitem__("norm", seen["norm"] + 1)),
        model.pair_down.register_forward_hook(lambda *_: seen.__setitem__("down", seen["down"] + 1)),
        model.pair_up.register_forward_hook(lambda *_: seen.__setitem__("up", seen["up"] + 1)),
    ]
    target, source = torch.randn(5, 256), torch.randn(5, 256)
    model._pair_components(target, source)
    model._pair_components(source, target)
    for hook in hooks:
        hook.remove()
    assert seen == {"norm": 2, "down": 2, "up": 2}
    names = [name for name, _ in model.named_modules() if name.startswith("pair_")]
    assert names == ["pair_norm", "pair_down", "pair_up"]


def test_swapping_target_and_source_changes_directional_pair_input():
    t = torch.randn(3, 256)
    v = torch.randn(3, 256)
    assert not torch.equal(PCRRModel._pair_features(t, v), PCRRModel._pair_features(v, t))


def test_pair_down_shape_is_1024_to_64():
    model = make_model()
    assert tuple(model.pair_down.weight.shape) == (64, 1024)
    assert tuple(model.pair_down.bias.shape) == (64,)


def test_pair_up_shape_is_64_to_256():
    model = make_model()
    assert tuple(model.pair_up.weight.shape) == (256, 64)
    assert tuple(model.pair_up.bias.shape) == (256,)


def test_pair_up_weight_and_bias_are_exactly_zero_at_initialization():
    model = make_model()
    assert torch.count_nonzero(model.pair_up.weight) == 0
    assert torch.count_nonzero(model.pair_up.bias) == 0


def test_paired_initial_residual_is_exactly_zero():
    _, _, _, _, info = make_model("paired").eval()(features(), graph(), return_details=True)
    for details in info["details"]["pairs"].values():
        assert torch.count_nonzero(details["delta"]) == 0


def test_shuffled_initial_residual_is_exactly_zero():
    _, _, _, _, info = make_model("shuffled").eval()(features(), graph(), return_details=True)
    for details in info["details"]["pairs"].values():
        assert torch.count_nonzero(details["delta"]) == 0


def test_same_seed_b_p_s_initial_outputs_are_elementwise_identical():
    outputs = []
    for variant in ("base", "paired", "shuffled"):
        outputs.append(make_model(variant).eval()(features(), graph())[0])
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_same_rng_state_gives_identical_train_mode_initial_forward():
    models = [make_model(variant).train() for variant in ("base", "paired", "shuffled")]
    outputs = []
    state = torch.random.get_rng_state()
    for model in models:
        torch.random.set_rng_state(state)
        outputs.append(model(features(), graph())[0])
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_optimizer_step_gives_pair_up_finite_nonzero_gradient():
    model = make_model("paired").train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(features(), graph())[0].square().mean().backward()
    grad = model.pair_up.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    optimizer.step()


def test_pair_down_receives_gradient_after_pair_up_moves_from_zero():
    model = make_model("paired").train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(features(), graph())[0].square().mean().backward()
        optimizer.step()
    grad = model.pair_down.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


def test_p_source_correspondence_is_same_node():
    _, _, _, _, info = make_model("paired").eval()(features(), graph(), return_details=True)
    details = info["details"]
    assert torch.equal(details["pairs"]["text"]["source"], details["modalities"]["visual"]["embedding"])
    assert torch.equal(details["pairs"]["visual"]["source"], details["modalities"]["text"]["embedding"])


def test_s_uses_a_fixed_deterministic_derangement():
    first = make_model("shuffled", seed=44).pair_shuffle_permutation
    second = make_model("shuffled", seed=44).pair_shuffle_permutation
    assert torch.equal(first, second)
    assert torch.equal(first.sort().values, torch.arange(NODES))
    assert torch.count_nonzero(first == torch.arange(NODES)) == 0


def test_s_reuses_identical_permutation_on_repeated_forwards():
    model = make_model("shuffled").eval()
    x, edge = features(), graph()
    _, _, _, _, first = model(x, edge, return_details=True)
    _, _, _, _, second = model(x, edge, return_details=True)
    assert torch.equal(first["details"]["used_source_permutation"], second["details"]["used_source_permutation"])


def test_source_shuffle_does_not_change_backbone_embeddings():
    model = make_model("paired").eval()
    x, edge = features(), graph()
    _, _, _, _, normal = model(x, edge, return_details=True)
    permutation = torch.roll(torch.arange(NODES), 1)
    _, _, _, _, shuffled = model(
        x, edge, intervention="source_node_shuffle", source_node_permutation=permutation,
        return_details=True,
    )
    for modality in ("text", "visual"):
        assert torch.equal(
            normal["details"]["modalities"][modality]["embedding"],
            shuffled["details"]["modalities"][modality]["embedding"],
        )


def test_source_shuffle_changes_only_pair_source_and_not_fusion_alignment():
    model = make_model("paired").eval()
    with torch.no_grad():
        model.pair_up.weight.normal_(0, 0.01)
    x, edge = features(), graph()
    permutation = torch.roll(torch.arange(NODES), 1)
    _, _, _, _, info = model(
        x, edge, intervention="source_node_shuffle", source_node_permutation=permutation,
        return_details=True,
    )
    details = info["details"]
    g_text = details["modalities"]["text"]["embedding"]
    g_visual = details["modalities"]["visual"]["embedding"]
    assert torch.equal(details["pairs"]["text"]["source"], g_visual.index_select(0, permutation))
    assert torch.equal(details["pairs"]["visual"]["source"], g_text.index_select(0, permutation))
    assert torch.equal(details["late_fusion_input"], torch.cat([
        details["pairs"]["text"]["embedding"], details["pairs"]["visual"]["embedding"]
    ], dim=-1))


def test_chunked_pair_response_matches_full_batch_eval():
    model = make_model("paired").eval()
    with torch.no_grad():
        model.pair_up.weight.normal_(0, 0.01)
    original = pcrr_module._PAIR_CHUNK_SIZE
    try:
        pcrr_module._PAIR_CHUNK_SIZE = NODES + 1
        full = model(features(), graph())[0]
        pcrr_module._PAIR_CHUNK_SIZE = 2
        chunked = model(features(), graph())[0]
    finally:
        pcrr_module._PAIR_CHUNK_SIZE = original
    torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-6)


def test_b_p_s_parameter_counts_are_identical():
    counts = [sum(parameter.numel() for parameter in make_model(v).parameters()) for v in ("base", "paired", "shuffled")]
    assert counts[0] == counts[1] == counts[2]


def test_b_p_s_state_dict_layouts_are_identical():
    models = [make_model(v) for v in ("base", "paired", "shuffled")]
    keys = [tuple(model.state_dict().keys()) for model in models]
    shapes = [tuple(value.shape for value in model.state_dict().values()) for model in models]
    assert keys[0] == keys[1] == keys[2]
    assert shapes[0] == shapes[1] == shapes[2]


def test_same_seed_b_p_s_named_initial_parameters_are_bitwise_identical():
    models = [make_model(v, 47) for v in ("base", "paired", "shuffled")]
    states = [model.state_dict() for model in models]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key
        assert torch.equal(states[0][key], states[2][key]), key


def test_aux_loss_is_exactly_zero_scalar():
    for variant in ("base", "paired", "shuffled"):
        aux_loss = make_model(variant).eval()(features(), graph())[3]
        assert aux_loss.shape == torch.Size([])
        assert aux_loss.item() == 0.0


def test_gradients_are_finite():
    model = make_model("paired").train()
    model(features(), graph())[0].square().mean().backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_inference_matches_eval_forward():
    model = make_model("paired").eval()
    expected = model(features(), graph())[0].detach().cpu()
    actual = model.inference(features(), graph(), device=torch.device("cpu"))
    assert torch.equal(actual, expected)


def test_lp_sampler_depth_is_three_hops_with_five_neighbors():
    model = make_model()
    cfg_value = OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    assert model.requires_full_lp_sampler_depth is True
    assert resolve_num_neighbors(cfg_value) == [5, 5, 5]


def test_residual_off_makes_pair_deltas_zero_after_training():
    model = make_model("paired").eval()
    with torch.no_grad():
        model.pair_up.weight.normal_(0, 0.01)
    _, _, _, _, info = model(features(), graph(), intervention="residual_off", return_details=True)
    assert all(torch.count_nonzero(pair["delta"]) == 0 for pair in info["details"]["pairs"].values())


def test_private_derangement_generator_does_not_change_global_rng_state():
    torch.manual_seed(1301)
    before = torch.random.get_rng_state()
    first = PCRRModel._make_derangement(101, 73042)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert torch.equal(first, PCRRModel._make_derangement(101, 73042))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA staging regression requires a GPU")
def test_cpu_feature_staging_checkpoint_preserves_dropout_rng_and_gradients():
    device = torch.device("cuda:0")
    model = make_model("paired").to(device).train()
    x = torch.randn((9001, 4), generator=torch.Generator().manual_seed(195))
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)

    model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)
    checkpointed = model._project_modality(x, "text", checkpoint_activations=True)
    checkpointed.square().mean().backward()
    checkpointed_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.projectors["text"].named_parameters()
    }

    model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)
    direct = model._project_modality(x, "text", checkpoint_activations=False)
    direct.square().mean().backward()
    direct_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.projectors["text"].named_parameters()
    }
    torch.testing.assert_close(checkpointed, direct, rtol=0, atol=0)
    for name in checkpointed_grads:
        torch.testing.assert_close(checkpointed_grads[name], direct_grads[name], rtol=1e-6, atol=1e-7)
