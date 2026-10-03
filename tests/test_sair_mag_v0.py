from __future__ import annotations

import torch
from omegaconf import OmegaConf

import src.models.sair_mag_v0 as sair_module
from src.models.pcrr_mag_v1 import Model as PCRRV1
from src.models.sair_mag_v0 import Model as SAIR


NODES = 11
DATA_INFO = {"input_dim": 7, "text_dim": 4, "visual_dim": 3, "num_nodes": NODES}


def cfg(variant="base", name="sair_mag_v0", seed=42):
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "name": name, "variant": variant, "hidden_dim": 256,
            "num_layers": 3, "dropout": 0.2, "diffusion_add_self_loops": True,
            "eps": 1.0e-8, "global_prior_restart": 0.15,
            "global_prior_order": 2, "pair_rank": 64,
        },
    })


def model(variant="base", seed=42):
    torch.manual_seed(seed)
    return SAIR(cfg(variant, seed=seed), DATA_INFO)


def graph(n=NODES):
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def features(seed=71, n=NODES):
    return torch.randn((n, 7), generator=torch.Generator().manual_seed(seed))


def _rms(x):
    return x.float().square().mean().sqrt()


def test_modality_split_has_exact_text_visual_widths():
    split = model()._split_modalities(features())
    assert split["text"].shape == (NODES, 4)
    assert split["visual"].shape == (NODES, 3)
    assert torch.equal(split["text"], features()[:, :4])
    assert torch.equal(split["visual"], features()[:, 4:])


def test_base_proposal_and_embedding_regress_to_pcrr_v1():
    torch.manual_seed(19)
    old = PCRRV1(cfg("base", "pcrr_mag_v1"), DATA_INFO).eval()
    torch.manual_seed(19)
    new = SAIR(cfg("base"), DATA_INFO).eval()
    assert all(torch.equal(old.state_dict()[k], new.state_dict()[k]) for k in old.state_dict())
    old_info = old(features(), graph(), return_details=True)[4]["details"]["modalities"]
    new_info = new(features(), graph(), return_details=True)[4]["details"]["modalities"]
    for modality in ("text", "visual"):
        torch.testing.assert_close(new_info[modality]["proposal"], old_info[modality]["proposal"], rtol=0, atol=1e-6)
        torch.testing.assert_close(new_info[modality]["embedding"], old_info[modality]["embedding"], rtol=0, atol=1e-6)


def test_base_eval_output_regresses_to_pcrr_v1():
    torch.manual_seed(29)
    old = PCRRV1(cfg("base", "pcrr_mag_v1"), DATA_INFO).eval()
    torch.manual_seed(29)
    new = SAIR(cfg("base"), DATA_INFO).eval()
    torch.testing.assert_close(new(features(), graph())[0], old(features(), graph())[0], rtol=0, atol=1e-6)


def test_prior_is_the_projector_output():
    net = model().eval()
    split = net._split_modalities(features())
    _, _, _, _, info = net(features(), graph(), return_details=True)
    for modality in ("text", "visual"):
        expected = net.projectors[modality](split[modality])
        torch.testing.assert_close(info["details"]["modalities"][modality]["prior"], expected, rtol=0, atol=0)


def test_response_is_proposal_minus_c0_prior():
    net = model("decoupled").eval()
    _, _, _, _, info = net(features(), graph(), return_details=True)
    details = info["details"]
    for row in details["modalities"].values():
        expected = row["proposal"] - details["coefficients"][0] * row["prior"]
        torch.testing.assert_close(row["structural_response"], expected, rtol=0, atol=0)


def test_small_graph_response_equals_explicit_positive_order_sum():
    net = model("decoupled").eval()
    _, _, _, _, info = net(features(), graph(), return_details=True)
    coeff = info["details"]["coefficients"]
    for row in info["details"]["modalities"].values():
        states = row["raw_states"]
        explicit = sum(coeff[k] * states[:, k] for k in range(1, 4))
        torch.testing.assert_close(row["structural_response"], explicit, rtol=0, atol=1e-6)


def test_streaming_gpr_response_equals_explicit_positive_order_sum():
    torch.manual_seed(7)
    net = model().eval()
    x = torch.randn(17, 256)
    edges = graph(17)
    src, dst, norm = net._normalized_operator(edges, 17, x.dtype)
    coeff = torch.tensor([0.4, 0.2, 0.3, 0.1])
    proposal = sair_module._StreamingRawGPR.apply(x, coeff, src, dst, norm)
    states = net._raw_states(x, src, dst, norm)
    explicit = sum(coeff[k] * states[k] for k in range(1, 4))
    torch.testing.assert_close(proposal - coeff[0] * x, explicit, rtol=0, atol=1e-6)


def test_d_pair_feature_layout_is_anchor_response():
    net = model("decoupled")
    p, r = torch.randn(5, 256), torch.randn(5, 256)
    expected = torch.cat([p, r, (p-r).abs(), p*r], dim=-1)
    assert torch.equal(net._pair_features(p, r), expected)


def test_c_pair_feature_layout_is_g_g_zero_square():
    net = model("generic")
    g = torch.randn(5, 256)
    expected = torch.cat([g, g, torch.zeros_like(g), g.square()], dim=-1)
    assert torch.equal(net._pair_features(g, g), expected)


def test_c_adapter_receives_only_g():
    net = model("generic").eval()
    _, _, _, _, info = net(features(), graph(), return_details=True)
    for modality, row in info["details"]["pairs"].items():
        g = info["details"]["modalities"][modality]["embedding"]
        assert row["target"] is g and row["source"] is g


def test_d_text_adapter_does_not_read_visual_features():
    net = model("decoupled").eval()
    x = features()
    first = net(x, graph(), return_details=True)[4]["details"]["pairs"]["text"]["delta"]
    changed = x.clone()
    changed[:, 4:] += 100 * torch.randn_like(changed[:, 4:])
    second = net(changed, graph(), return_details=True)[4]["details"]["pairs"]["text"]["delta"]
    assert torch.equal(first, second)


def test_text_and_visual_use_the_same_pair_modules():
    net = model("decoupled").eval()
    counts = {"norm": 0, "down": 0, "up": 0}
    hooks = [
        net.pair_norm.register_forward_hook(lambda *_: counts.__setitem__("norm", counts["norm"] + 1)),
        net.pair_down.register_forward_hook(lambda *_: counts.__setitem__("down", counts["down"] + 1)),
        net.pair_up.register_forward_hook(lambda *_: counts.__setitem__("up", counts["up"] + 1)),
    ]
    net(features(), graph())
    for hook in hooks:
        hook.remove()
    assert counts == {"norm": 2, "down": 2, "up": 2}
    assert [name for name, _ in net.named_modules() if name.startswith("pair_")] == ["pair_norm", "pair_down", "pair_up"]


def test_pair_down_and_up_dimensions_match_protocol():
    net = model()
    assert tuple(net.pair_down.weight.shape) == (64, 1024)
    assert tuple(net.pair_up.weight.shape) == (256, 64)


def test_pair_up_is_exact_zero_initialized():
    for variant in ("base", "generic", "decoupled"):
        net = model(variant)
        assert torch.count_nonzero(net.pair_up.weight) == 0
        assert torch.count_nonzero(net.pair_up.bias) == 0


def test_initial_eval_outputs_are_bitwise_identical():
    outputs = [model(v).eval()(features(), graph())[0] for v in ("base", "generic", "decoupled")]
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])


def test_initial_train_outputs_match_after_rng_reset():
    nets = [model(v).train() for v in ("base", "generic", "decoupled")]
    state = torch.random.get_rng_state()
    outputs = []
    for net in nets:
        torch.random.set_rng_state(state)
        outputs.append(net(features(), graph())[0])
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])


def test_variant_parameter_counts_are_identical():
    counts = [sum(p.numel() for p in model(v).parameters()) for v in ("base", "generic", "decoupled")]
    assert len(set(counts)) == 1


def test_variant_state_dict_layouts_are_identical():
    nets = [model(v) for v in ("base", "generic", "decoupled")]
    keys = [list(net.state_dict()) for net in nets]
    shapes = [[tuple(t.shape) for t in net.state_dict().values()] for net in nets]
    assert keys[0] == keys[1] == keys[2]
    assert shapes[0] == shapes[1] == shapes[2]


def test_same_seed_named_tensors_are_bitwise_identical():
    nets = [model(v, seed=81) for v in ("base", "generic", "decoupled")]
    states = [dict(net.named_parameters()) for net in nets]
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name]), name
        assert torch.equal(states[0][name], states[2][name]), name


def test_pcrr_base_and_sair_base_construction_preserve_global_cpu_rng_state():
    torch.manual_seed(105)
    PCRRV1(cfg("base", "pcrr_mag_v1"), DATA_INFO)
    expected = torch.random.get_rng_state()
    torch.manual_seed(105)
    SAIR(cfg("base"), DATA_INFO)
    actual = torch.random.get_rng_state()
    assert torch.equal(actual, expected)


def test_generic_pair_up_gets_finite_nonzero_first_step_gradient():
    net = model("generic").train()
    net(features(), graph())[0].square().mean().backward()
    grad = net.pair_up.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_decoupled_pair_up_gets_finite_nonzero_first_step_gradient():
    net = model("decoupled").train()
    net(features(), graph())[0].square().mean().backward()
    grad = net.pair_up.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_pair_down_gets_gradient_after_pair_up_moves():
    net = model("decoupled").train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        net(features(), graph())[0].square().mean().backward()
        optimizer.step()
    grad = net.pair_down.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_d_readout_gradient_reaches_projectors_and_nonzero_order_coefficients():
    net = model("decoupled").train()
    with torch.no_grad():
        net.pair_up.weight.normal_(0, 0.02)
    _, _, _, _, info = net(features(), graph(), return_details=True)
    delta = torch.cat([row["delta"] for row in info["details"]["pairs"].values()])
    delta.square().mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in net.projectors.parameters())
    grad = net.delta_c_raw.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad[1:]) > 0


def test_chunked_generic_matches_full_pair_reference(monkeypatch):
    net = model("generic").eval()
    monkeypatch.setattr(sair_module, "_PAIR_CHUNK_SIZE", 3)
    chunked = net(features(), graph())[0]
    monkeypatch.setattr(sair_module, "_PAIR_CHUNK_SIZE", NODES + 1)
    full = net(features(), graph())[0]
    torch.testing.assert_close(chunked, full, rtol=0, atol=1e-6)


def test_chunked_decoupled_matches_full_pair_reference(monkeypatch):
    net = model("decoupled").eval()
    with torch.no_grad():
        net.pair_up.weight.normal_(0, 0.02)
    monkeypatch.setattr(sair_module, "_PAIR_CHUNK_SIZE", 3)
    chunked = net(features(), graph())[0]
    monkeypatch.setattr(sair_module, "_PAIR_CHUNK_SIZE", NODES + 1)
    full = net(features(), graph())[0]
    torch.testing.assert_close(chunked, full, rtol=0, atol=1e-6)


def test_pair_features_are_built_only_for_configured_node_chunks(monkeypatch):
    net = model("decoupled").eval()
    monkeypatch.setattr(sair_module, "_PAIR_CHUNK_SIZE", 3)
    observed = []
    hook = net.pair_norm.register_forward_pre_hook(lambda _m, args: observed.append(args[0].shape[0]))
    net(features(), graph())
    hook.remove()
    assert observed and max(observed) <= 3


def test_aux_loss_is_exactly_zero():
    aux = model("decoupled").eval()(features(), graph())[3]
    assert aux.shape == torch.Size([]) and aux.item() == 0.0


def test_all_variant_gradients_are_finite():
    for variant in ("base", "generic", "decoupled"):
        net = model(variant).train()
        net(features(), graph())[0].square().mean().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads), variant


def test_inference_matches_eval_forward():
    net = model("decoupled").eval()
    expected = net(features(), graph())[0].detach().cpu()
    actual = net.inference(features(), graph(), device=torch.device("cpu"))
    assert torch.equal(actual, expected)


def test_model_requires_full_lp_sampler_depth_and_cpu_staging():
    net = model()
    assert net.requires_full_lp_sampler_depth is True
    assert net.supports_cpu_feature_staging is True


def test_lp_sampler_resolution_is_three_hops_of_five():
    from src.tasks.lp import _resolve_lp_num_neighbors
    task_cfg = OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    assert _resolve_lp_num_neighbors(task_cfg, model()) == [5, 5, 5]


def test_development_no_test_gate_is_preserved():
    from src.tasks.nc import _should_evaluate_test
    task_cfg = OmegaConf.create({"task": {"development_no_test": True, "evaluate_test": False}})
    assert _should_evaluate_test(task_cfg) is False


def test_large_graph_streaming_forward_uses_response_subtraction_without_raw_bank(monkeypatch):
    # Exercise the streaming branch at a small synthetic size without allocating a
    # 50k x 256 test fixture; the branch threshold is a memory policy constant.
    monkeypatch.setattr(sair_module, "_ACTIVATION_CHECKPOINT_NODE_THRESHOLD", 5)
    net = model("decoupled").eval()
    x = features(n=7)
    _, _, _, _, info = net(x, graph(7), return_details=True)
    for row in info["details"]["modalities"].values():
        assert "raw_states" not in row
        torch.testing.assert_close(
            row["structural_response"],
            row["proposal"] - info["details"]["coefficients"][0] * row["prior"],
            rtol=0, atol=0,
        )
