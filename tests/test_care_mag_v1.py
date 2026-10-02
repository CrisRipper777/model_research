from __future__ import annotations

import itertools

import pytest
import torch
from omegaconf import OmegaConf

from src.models.care_mag_v0 import Model as ModelV0
from src.models.care_mag_v1 import Model
from src.tasks.lp import _resolve_lp_num_neighbors


DATA_INFO = {
    "input_dim": 7,
    "text_dim": 4,
    "visual_dim": 3,
    "num_nodes": 9,
    "num_classes": 3,
}
MODES = tuple(itertools.product(("global", "node"), ("off", "static", "context")))


def _cfg(trajectory="node", adapter="context"):
    return OmegaConf.create(
        {
            "model": {
                "name": "care_mag_v1",
                "trajectory_mode": trajectory,
                "adapter_mode": adapter,
                "hidden_dim": 16,
                "num_layers": 3,
                "dropout": 0.2,
                "norm": "layernorm",
                "state_dim": 8,
                "adapter_rank": 4,
                "diffusion_add_self_loops": True,
                "trust_init": 0.5,
                "adapter_scale_init": 0.1,
                "adapter_coeff_radius": 0.5,
                "eps": 1e-8,
            },
            "task": {"num_neighbors": [5, 5, 5]},
        }
    )


def _v0_cfg(variant):
    cfg = _cfg()
    cfg.model.name = "care_mag_v0"
    cfg.model.variant = variant
    del cfg.model.trajectory_mode
    del cfg.model.adapter_mode
    return cfg


def _graph():
    return torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 0, 2, 5],
            [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 5, 8, 8, 2],
        ],
        dtype=torch.long,
    )


def _x():
    torch.manual_seed(31)
    return torch.randn(9, DATA_INFO["input_dim"])


def _make(trajectory="node", adapter="context", seed=17):
    torch.manual_seed(seed)
    return Model(_cfg(trajectory, adapter), DATA_INFO)


def test_text_visual_split_and_finite_forward_contract():
    model = _make()
    model.eval()
    x = _x()
    z, first, second, aux_loss, info = model(x, _graph(), return_details=True)
    details = info["details"]
    torch.testing.assert_close(
        details["priors"]["text"], model.projectors["text"](x[:, :4])
    )
    torch.testing.assert_close(
        details["priors"]["visual"], model.projectors["visual"](x[:, 4:])
    )
    assert z.shape == (9, 16)
    assert first is None and second is None
    assert aux_loss.item() == 0.0
    assert torch.isfinite(z).all()


@pytest.mark.parametrize("trajectory,adapter", MODES)
def test_alpha_rows_sum_to_one_and_diagnostics_are_finite(trajectory, adapter):
    model = _make(trajectory, adapter).eval()
    _, _, _, _, info = model(_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        alpha = info["details"]["alphas"][modality]
        torch.testing.assert_close(
            alpha.sum(dim=1), torch.ones(alpha.size(0)), rtol=0.0, atol=1e-6
        )
        assert torch.isfinite(alpha).all()
        for key in (
            "effective_hop_mean",
            "effective_hop_std",
            "alpha_entropy_mean",
            "alpha_entropy_std",
            "response_rms",
            "delta_rms",
            "gamma_delta_rms",
            "coefficient_node_std",
            "delta_response_cosine_mean",
        ):
            assert torch.isfinite(info[modality][key])


def test_global_trajectory_is_node_invariant_per_modality():
    model = _make("global", "off").eval()
    _, _, _, _, info = model(_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        alpha = info["details"]["alphas"][modality]
        assert torch.equal(alpha, alpha[:1].expand_as(alpha))
        assert torch.count_nonzero(info[modality]["alpha_std"]) == 0


def test_node_trajectory_can_have_recipient_specific_alpha():
    model = _make("node", "off").eval()
    _, _, _, _, info = model(_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        alpha = info["details"]["alphas"][modality]
        assert float(alpha.std(dim=0, unbiased=False).max()) > 1e-8


def test_six_factorial_cells_have_identical_parameter_count_keys_and_seed_init():
    models = {
        (trajectory, adapter): _make(trajectory, adapter, seed=123)
        for trajectory, adapter in MODES
    }
    first = models[MODES[0]].state_dict()
    first_count = sum(parameter.numel() for parameter in models[MODES[0]].parameters())
    for mode, model in models.items():
        state = model.state_dict()
        assert list(state) == list(first), mode
        assert sum(parameter.numel() for parameter in model.parameters()) == first_count
        for name, value in state.items():
            assert torch.equal(value, first[name]), f"{mode}: {name} differs"


@pytest.mark.parametrize(
    "adapter,legacy_variant",
    (("off", "structural_base"), ("static", "static_adapter"), ("context", "context_adapter")),
)
def test_node_factorial_cells_strict_load_and_match_v0_eval(adapter, legacy_variant):
    torch.manual_seed(987)
    old = ModelV0(_v0_cfg(legacy_variant), DATA_INFO).eval()
    new = _make("node", adapter, seed=111).eval()
    result = new.load_state_dict(old.state_dict(), strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    x, edge_index = _x(), _graph()
    old_z = old(x, edge_index)[0]
    new_z = new(x, edge_index)[0]
    assert torch.equal(new_z, old_z)


def test_static_coefficients_are_node_invariant_and_context_can_vary():
    static = _make("node", "static").eval()
    context = _make("node", "context").eval()
    static_info = static(_x(), _graph(), return_details=True)[4]
    context_info = context(_x(), _graph(), return_details=True)[4]
    for modality in ("text", "visual"):
        static_coeff = static_info["details"]["coefficients"][modality]
        context_coeff = context_info["details"]["coefficients"][modality]
        assert torch.equal(static_coeff, static_coeff[:1].expand_as(static_coeff))
        assert float(context_coeff.std(dim=0, unbiased=False).max()) > 1e-8


def test_off_adapter_correction_is_exactly_zero():
    model = _make("node", "off").eval()
    _, _, _, _, info = model(_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        assert torch.count_nonzero(info["details"]["deltas"][modality]) == 0


@pytest.mark.parametrize("trajectory", ("global", "node"))
def test_global_and_node_modes_support_finite_backward(trajectory):
    model = _make(trajectory, "context").train()
    z = model(_x(), _graph())[0]
    z.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_inference_matches_eval_forward():
    model = _make("global", "static").eval()
    x, edge_index = _x(), _graph()
    expected = model(x, edge_index)[0].detach().cpu()
    actual = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=3)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_lp_three_hop_sampler_depth_is_compatible():
    model = _make()
    assert model.requires_full_lp_sampler_depth is True
    assert _resolve_lp_num_neighbors(_cfg(), model) == [5, 5, 5]


def test_default_model_width_is_256():
    cfg = _cfg()
    del cfg.model.hidden_dim
    del cfg.model.state_dim
    del cfg.model.adapter_rank
    model = Model(cfg, DATA_INFO)
    assert model.out_dim == 256


def test_trajectory_global_mean_replaces_rows_with_their_mean_and_recomputes_response():
    model = _make("node", "off").eval()
    x, edge_index = _x(), _graph()
    normal = model(x, edge_index, return_details=True)[4]["details"]
    intervened = model(
        x, edge_index, intervention="traj_global_mean", return_details=True
    )[4]["details"]
    for modality in ("text", "visual"):
        alpha = normal["alphas"][modality]
        changed = intervened["alphas"][modality]
        expected = alpha.mean(dim=0, keepdim=True).expand_as(alpha)
        assert torch.equal(changed, expected)
        stack = normal["response_stacks"][modality]
        expected_response = (expected.unsqueeze(-1) * stack).sum(dim=1)
        assert torch.equal(intervened["responses"][modality], expected_response)


def test_trajectory_shuffle_preserves_alpha_multiset_and_is_deterministic():
    model = _make("node", "off").eval()
    x, edge_index = _x(), _graph()
    normal = model(x, edge_index, return_details=True)[4]["details"]["alphas"]
    first = model(
        x, edge_index, intervention="traj_node_shuffle", shuffle_seed=87,
        return_details=True,
    )[4]["details"]["alphas"]
    second = model(
        x, edge_index, intervention="traj_node_shuffle", shuffle_seed=87,
        return_details=True,
    )[4]["details"]["alphas"]
    for modality in ("text", "visual"):
        assert torch.equal(first[modality], second[modality])
        assert sorted(tuple(row.tolist()) for row in first[modality]) == sorted(
            tuple(row.tolist()) for row in normal[modality]
        )
        assert not torch.equal(first[modality], normal[modality])
        torch.testing.assert_close(
            first[modality].sum(dim=1), torch.ones(first[modality].size(0)),
            rtol=0.0, atol=1e-6,
        )


def test_trajectory_interventions_are_noops_for_global_mode():
    model = _make("global", "context").eval()
    x, edge_index = _x(), _graph()
    normal = model(x, edge_index)[0]
    for intervention in ("traj_global_mean", "traj_node_shuffle"):
        changed = model(x, edge_index, intervention=intervention, shuffle_seed=87)[0]
        assert torch.equal(changed, normal)
