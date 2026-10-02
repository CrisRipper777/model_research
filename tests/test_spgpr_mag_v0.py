from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.spgpr_mag_v0 import Model
from src.tasks.lp import _resolve_lp_num_neighbors
from src.utils.summary import count_parameters


MODES = (
    "uniform",
    "positive_shared",
    "signed_shared",
    "signed_independent",
    "signed_shared_private",
)


def make_cfg(mode: str = "signed_shared"):
    return OmegaConf.create(
        {
            "model": {
                "filter_mode": mode,
                "hidden_dim": 8,
                "num_layers": 3,
                "dropout": 0.2,
                "diffusion_add_self_loops": True,
                "eps": 1.0e-8,
                "structural_scale_init": 0.5,
            }
        }
    )


def make_model(mode: str = "signed_shared", seed: int = 123):
    torch.manual_seed(seed)
    return Model(
        make_cfg(mode),
        {"input_dim": 7, "text_dim": 3, "visual_dim": 4},
    )


def test_text_visual_split_uses_declared_order_and_widths():
    model = make_model()
    x = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    split = model._split_modalities(x)
    assert torch.equal(split["text"], x[:, :3])
    assert torch.equal(split["visual"], x[:, 3:])


def test_forward_is_finite_with_expected_shape_and_zero_aux_loss():
    model = make_model().eval()
    x = torch.randn(5, 7)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    z, first, second, aux_loss, _ = model(x, edge_index)
    assert z.shape == (5, 8)
    assert torch.isfinite(z).all()
    assert first is None and second is None
    assert aux_loss.item() == 0.0


def test_symmetric_normalized_propagation_and_three_hops():
    model = make_model()
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    src, dst, norm = model._normalized_operator(edge_index, 3, torch.float32)
    actual = model._propagate(torch.tensor([[1.0], [2.0], [4.0]]), src, dst, norm)
    # add_remaining_self_loops gives degrees [2, 3, 2]; Ahat is symmetric.
    expected = torch.tensor(
        [
            [0.5 + 2.0 / (6.0**0.5)],
            [2.0 / 3.0 + 5.0 / (6.0**0.5)],
            [2.0 + 2.0 / (6.0**0.5)],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, atol=1e-6)
    model.eval()
    _, _, _, _, info = model(torch.randn(3, 7), edge_index, return_details=True)
    assert model.num_layers == 3
    assert info["details"]["responses"]["text"].shape == (3, 3, 8)
    assert info["details"]["responses"]["visual"].shape == (3, 3, 8)


def test_uniform_and_positive_filter_constraints():
    uniform = make_model("uniform").effective_gammas()
    expected = torch.tensor([1.0 / 3.0] * 3)
    assert torch.equal(uniform["text"], expected)
    assert torch.equal(uniform["visual"], expected)
    positive = make_model("positive_shared").effective_gammas()
    for gamma in positive.values():
        assert torch.all(gamma > 0)
        assert torch.allclose(gamma.sum(), torch.tensor(1.0))


def test_signed_modes_have_unit_l1_norm_and_finite_signednorm():
    for mode in ("signed_shared", "signed_independent", "signed_shared_private"):
        gammas = make_model(mode).effective_gammas()
        for gamma in gammas.values():
            assert torch.isfinite(gamma).all()
            assert torch.allclose(gamma.abs().sum(), torch.tensor(1.0))
    model = make_model("signed_shared")
    with torch.no_grad():
        model.theta_signed_shared.copy_(torch.tensor([1.0, -0.25, 2.0]))
    assert torch.isfinite(model.effective_gammas()["text"]).all()
    assert (model.effective_gammas()["text"] < 0).any()


def test_all_modes_start_with_identical_uniform_effective_filters():
    expected = torch.full((3,), 1.0 / 3.0)
    for mode in MODES:
        for gamma in make_model(mode).effective_gammas().values():
            assert torch.equal(gamma, expected)


def test_shared_private_zero_difference_and_manual_filter_divergence():
    model = make_model("signed_shared_private")
    gamma = model.effective_gammas()
    assert torch.equal(gamma["text"], gamma["visual"])
    with torch.no_grad():
        model.theta_sp_diff.copy_(torch.tensor([0.5, -0.25, 0.0]))
    gamma = model.effective_gammas()
    assert not torch.allclose(gamma["text"], gamma["visual"])
    shared, private = model.effective_sp_decomposition()
    assert torch.allclose(shared, (gamma["text"] + gamma["visual"]) / 2)
    assert torch.allclose(private, (gamma["text"] - gamma["visual"]) / 2)


def test_independent_filters_can_change_independently_and_shared_shapes_match():
    independent = make_model("signed_independent")
    with torch.no_grad():
        independent.theta_ind_text.copy_(torch.tensor([1.0, -0.5, 1.0]))
    gamma = independent.effective_gammas()
    assert not torch.allclose(gamma["text"], gamma["visual"])
    shared = make_model("signed_shared").effective_gammas()
    assert shared["text"].shape == shared["visual"].shape == (3,)
    assert torch.equal(shared["text"], shared["visual"])


def test_modes_have_identical_parameter_count_state_layout_and_seeded_values():
    models = {mode: make_model(mode, seed=777) for mode in MODES}
    reference = models[MODES[0]]
    ref_state = reference.state_dict()
    for model in models.values():
        state = model.state_dict()
        assert list(state) == list(ref_state)
        assert {key: value.shape for key, value in state.items()} == {
            key: value.shape for key, value in ref_state.items()
        }
        assert count_parameters(model) == count_parameters(reference)
        for key, value in state.items():
            assert torch.equal(value, ref_state[key]), key


def test_lambdas_are_global_scalars_initialized_to_half():
    model = make_model()
    assert model.theta_lambda_text.shape == torch.Size([])
    assert model.theta_lambda_visual.shape == torch.Size([])
    assert torch.sigmoid(model.theta_lambda_text).item() == 0.5
    assert torch.sigmoid(model.theta_lambda_visual).item() == 0.5


def test_inference_matches_eval_forward_and_gradients_are_finite():
    model = make_model("signed_independent")
    x = torch.randn(6, 7)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    model.eval()
    expected = model(x, edge_index)[0].detach()
    actual = model.inference(x, edge_index, device=torch.device("cpu"))
    assert torch.allclose(actual, expected.cpu())
    model.train()
    model.zero_grad(set_to_none=True)
    z, _, _, aux_loss, _ = model(x, edge_index)
    (z.square().mean() + aux_loss).backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_sampler_depth_uses_three_hops_and_configured_fanouts():
    model = make_model()
    cfg = OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    assert model.requires_full_lp_sampler_depth is True
    assert _resolve_lp_num_neighbors(cfg, model) == [5, 5, 5]
