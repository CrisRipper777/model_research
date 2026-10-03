from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

import src.models.pcrr_mag_v0 as v0_module
import src.models.pcrr_mag_v1 as v1_module
from src.models.pcrr_mag_v0 import Model as PCRRV0
from src.models.pcrr_mag_v1 import Model as PCRRV1


NODES = 11
DATA_INFO = {"input_dim": 7, "text_dim": 4, "visual_dim": 3, "num_nodes": NODES}


def cfg(variant: str, name: str, seed: int = 42):
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
                "eps": 1.0e-8,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "pair_rank": 64,
            },
        }
    )


def make_model(variant: str, *, version: int = 1, seed: int = 42):
    torch.manual_seed(seed)
    cls = PCRRV1 if version == 1 else PCRRV0
    name = f"pcrr_mag_v{version}"
    return cls(cfg(variant, name, seed), DATA_INFO)


def graph(n: int = NODES):
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def features(seed: int = 71, n: int = NODES):
    return torch.randn((n, 7), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("old_variant,new_variant", [("base", "base"), ("paired", "paired"), ("shuffled", "shuffled")])
def test_v1_bps_eval_regresses_to_v0_after_loading_identical_trained_state(old_variant, new_variant):
    old = make_model(old_variant, version=0, seed=137)
    # A nonzero response makes the P/S source semantics part of the regression gate.
    with torch.no_grad():
        old.pair_up.weight.normal_(0, 0.015)
        old.pair_up.bias.normal_(0, 0.01)
    new = make_model(new_variant, version=1, seed=901)
    new.load_state_dict(old.state_dict(), strict=True)
    old.eval()
    new.eval()
    for intervention, permutation in (
        ("normal", None),
        ("residual_off", None),
        ("source_node_shuffle", torch.roll(torch.arange(NODES), 3)),
    ):
        expected = old(features(), graph(), intervention=intervention, source_node_permutation=permutation)[0]
        actual = new(features(), graph(), intervention=intervention, source_node_permutation=permutation)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_target_only_source_is_exactly_target_for_both_modalities():
    model = make_model("target_only").eval()
    _, _, _, _, info = model(features(), graph(), return_details=True)
    for direction, details in info["details"]["pairs"].items():
        assert details["source"] is details["target"]
        assert torch.equal(details["source"], details["target"])
        expected = torch.cat(
            [details["target"], details["target"], torch.zeros_like(details["target"]), details["target"].square()],
            dim=-1,
        )
        assert torch.equal(model._pair_features(details["target"], details["source"]), expected), direction


def test_target_only_pair_branch_does_not_depend_on_other_modality_source():
    model = make_model("target_only").eval()
    x = features()
    first = model(x, graph(), return_details=True)[4]["details"]["pairs"]["text"]["delta"]
    changed_visual = x.clone()
    changed_visual[:, 4:] += torch.randn_like(changed_visual[:, 4:]) * 10
    second = model(changed_visual, graph(), return_details=True)[4]["details"]["pairs"]["text"]["delta"]
    assert torch.equal(first, second)


def test_p_still_uses_same_node_other_modality_source():
    model = make_model("paired").eval()
    _, _, _, _, info = model(features(), graph(), return_details=True)
    details = info["details"]
    assert details["pairs"]["text"]["source"] is details["modalities"]["visual"]["embedding"]
    assert details["pairs"]["visual"]["source"] is details["modalities"]["text"]["embedding"]


def test_s_only_permutates_residual_source_and_keeps_late_fusion_node_aligned():
    model = make_model("shuffled").eval()
    _, _, _, _, info = model(features(), graph(), return_details=True)
    details = info["details"]
    permutation = details["used_source_permutation"]
    assert permutation is model.pair_shuffle_permutation
    assert torch.equal(details["pairs"]["text"]["source"], details["modalities"]["visual"]["embedding"][permutation])
    assert torch.equal(details["pairs"]["visual"]["source"], details["modalities"]["text"]["embedding"][permutation])
    assert torch.equal(details["late_fusion_input"][:, :256], details["pairs"]["text"]["embedding"])
    assert torch.equal(details["late_fusion_input"][:, 256:], details["pairs"]["visual"]["embedding"])


def test_pair_response_network_is_shared_between_directions():
    model = make_model("target_only").eval()
    counts = {"norm": 0, "down": 0, "up": 0}
    hooks = [
        model.pair_norm.register_forward_hook(lambda *_: counts.__setitem__("norm", counts["norm"] + 1)),
        model.pair_down.register_forward_hook(lambda *_: counts.__setitem__("down", counts["down"] + 1)),
        model.pair_up.register_forward_hook(lambda *_: counts.__setitem__("up", counts["up"] + 1)),
    ]
    model(features(), graph())
    for hook in hooks:
        hook.remove()
    assert counts == {"norm": 2, "down": 2, "up": 2}
    assert [name for name, _ in model.named_modules() if name.startswith("pair_")] == ["pair_norm", "pair_down", "pair_up"]


def test_b_p_s_t_parameter_count_and_state_dict_layout_match():
    models = [make_model(name) for name in ("base", "paired", "shuffled", "target_only")]
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert len(set(counts)) == 1
    keys = [list(model.state_dict()) for model in models]
    shapes = [[tuple(t.shape) for t in model.state_dict().values()] for model in models]
    assert keys.count(keys[0]) == len(keys)
    assert shapes.count(shapes[0]) == len(shapes)


def test_same_seed_b_p_s_t_named_tensors_are_bitwise_identical():
    models = [make_model(name, seed=47) for name in ("base", "paired", "shuffled", "target_only")]
    states = [model.state_dict() for model in models]
    for key in states[0]:
        assert all(torch.equal(states[0][key], state[key]) for state in states[1:]), key


def test_pair_up_is_exactly_zero_initialized_in_all_variants():
    for variant in ("base", "paired", "shuffled", "target_only"):
        model = make_model(variant)
        assert torch.count_nonzero(model.pair_up.weight) == 0
        assert torch.count_nonzero(model.pair_up.bias) == 0


def test_b_p_s_t_initial_eval_outputs_are_elementwise_identical():
    outputs = [make_model(v).eval()(features(), graph())[0] for v in ("base", "paired", "shuffled", "target_only")]
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_b_p_s_t_train_outputs_match_after_rng_reset():
    models = [make_model(v).train() for v in ("base", "paired", "shuffled", "target_only")]
    state = torch.random.get_rng_state()
    outputs = []
    for model in models:
        torch.random.set_rng_state(state)
        outputs.append(model(features(), graph())[0])
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_target_only_residual_starts_exactly_zero():
    _, _, _, _, info = make_model("target_only").eval()(features(), graph(), return_details=True)
    assert all(torch.count_nonzero(row["delta"]) == 0 for row in info["details"]["pairs"].values())


def test_target_only_pair_up_gets_finite_nonzero_first_step_gradient():
    model = make_model("target_only").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    model(features(), graph())[0].square().mean().backward()
    grad = model.pair_up.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    optimizer.step()


def test_target_only_pair_down_gets_finite_gradient_after_pair_up_moves():
    model = make_model("target_only").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(features(), graph())[0].square().mean().backward()
        optimizer.step()
    grad = model.pair_down.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0


def test_chunked_target_only_residual_matches_full_residual(monkeypatch):
    model = make_model("target_only").eval()
    monkeypatch.setattr(v1_module, "_PAIR_CHUNK_SIZE", 3)
    chunked = model(features(), graph())[0]
    monkeypatch.setattr(v1_module, "_PAIR_CHUNK_SIZE", NODES + 1)
    full = model(features(), graph())[0]
    torch.testing.assert_close(chunked, full, rtol=0, atol=1e-6)


def test_target_only_aux_loss_is_exact_zero():
    aux_loss = make_model("target_only").eval()(features(), graph())[3]
    assert aux_loss.shape == torch.Size([]) and aux_loss.item() == 0.0


def test_target_only_gradients_are_finite():
    model = make_model("target_only").train()
    model(features(), graph())[0].square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_target_only_inference_matches_eval_forward():
    model = make_model("target_only").eval()
    expected = model(features(), graph())[0].detach().cpu()
    actual = model.inference(features(), graph(), device=torch.device("cpu"))
    assert torch.equal(actual, expected)


def test_target_only_residual_off_disables_both_self_refinement_branches():
    model = make_model("target_only").eval()
    with torch.no_grad():
        model.pair_up.weight.normal_(0, 0.01)
    _, _, _, _, info = model(features(), graph(), intervention="residual_off", return_details=True)
    assert all(torch.count_nonzero(row["delta"]) == 0 for row in info["details"]["pairs"].values())


def test_v1_legacy_bps_feature_and_derangement_helpers_match_v0():
    target = torch.randn(4, 256)
    source = torch.randn(4, 256)
    assert torch.equal(v1_module.Model._pair_features(target, source), v0_module.Model._pair_features(target, source))
    for n in (0, 1, 17):
        assert torch.equal(v1_module.Model._make_derangement(n, 99), v0_module.Model._make_derangement(n, 99))


def test_target_only_pair_source_ignores_supplied_source_permutation():
    model = make_model("target_only").eval()
    permutation = torch.roll(torch.arange(NODES), 1)
    _, _, _, _, info = model(features(), graph(), intervention="source_node_shuffle", source_node_permutation=permutation, return_details=True)
    pairs = info["details"]["pairs"]
    for details in pairs.values():
        assert details["source"] is details["target"]
