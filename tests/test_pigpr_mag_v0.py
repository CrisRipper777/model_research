from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.pigpr_mag_v0 import Model as PIGPR
from src.models.spgpr_mag_v0 import Model as SPGPR
from src.tasks.lp import _resolve_lp_num_neighbors
from src.utils.summary import count_parameters


VARIANTS = (
    "prior_only",
    "raw_uniform_protected",
    "anchored_uniform_protected",
    "anchored_prior_protected",
    "anchored_gpr_protected",
    "anchored_gpr_direct",
)
DATA_INFO = {"input_dim": 7, "text_dim": 3, "visual_dim": 4}


def make_cfg(variant: str = "anchored_gpr_protected"):
    return OmegaConf.create(
        {
            "model": {
                "variant": variant,
                "hidden_dim": 8,
                "num_layers": 3,
                "dropout": 0.2,
                "diffusion_add_self_loops": True,
                "eps": 1.0e-8,
                "anchor_alpha": 0.1,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "structural_scale_init": 0.5,
            }
        }
    )


def make_model(variant: str = "anchored_gpr_protected", seed: int = 123):
    torch.manual_seed(seed)
    return PIGPR(make_cfg(variant), DATA_INFO)


def graph(num_nodes: int = 6):
    src, dst = [], []
    for node in range(num_nodes):
        other = (node + 1) % num_nodes
        src.extend((node, other))
        dst.extend((other, node))
    return torch.tensor([src, dst], dtype=torch.long)


def test_text_visual_split_uses_declared_order_and_widths():
    model = make_model()
    x = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    split = model._split_modalities(x)
    assert torch.equal(split["text"], x[:, :3])
    assert torch.equal(split["visual"], x[:, 3:])


def test_forward_finite_shape_three_hops_and_zero_auxiliary_loss():
    model = make_model().eval()
    x = torch.randn(6, 7)
    z, first, second, aux_loss, _ = model(x, graph())
    assert model.num_layers == 3
    assert model.out_dim == 8
    assert z.shape == (6, 8) and torch.isfinite(z).all()
    assert first is None and second is None
    assert aux_loss.item() == 0.0


def test_raw_propagation_and_anchored_recurrence_are_exact():
    model = make_model().eval()
    prior = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    src, dst, norm = model._normalized_operator(graph(), 6, prior.dtype)
    raw, anchored = model._raw_and_anchored_states(prior, src, dst, norm)
    assert len(raw) == len(anchored) == 4
    assert torch.equal(anchored[0], prior)
    assert torch.equal(raw[0], prior)
    for order in range(1, 4):
        expected_raw = model._propagate(raw[order - 1], src, dst, norm)
        expected_anchor = 0.9 * model._propagate(anchored[order - 1], src, dst, norm) + 0.1 * prior
        assert torch.allclose(raw[order], expected_raw, atol=0, rtol=0)
        assert torch.allclose(anchored[order], expected_anchor, atol=0, rtol=0)
    assert not torch.equal(anchored[2], 0.9 * model._propagate(anchored[1], src, dst, norm))


def test_global_prior_and_anchored_coefficients_match_cosi_formula():
    model = make_model()
    expected_monomial = torch.tensor([0.15, 0.1275, 0.7225, 0.0])
    expected_anchored = torch.tensor(
        [0.0555555556, 0.0524691358, 0.8919753086, 0.0]
    )
    assert torch.allclose(model.monomial_prior, expected_monomial, atol=1e-7)
    assert torch.allclose(model.gamma_prior, expected_anchored, atol=1e-7)
    matrix = model._anchored_basis_matrix(
        4, model.anchor_alpha, dtype=model.gamma_prior.dtype, device=model.gamma_prior.device
    )
    assert torch.allclose(matrix.T @ model.gamma_prior, model.monomial_prior, atol=1e-7)


def test_anchored_and_monomial_compositions_match_on_synthetic_graph():
    model = make_model()
    prior = torch.randn(6, 5)
    src, dst, norm = model._normalized_operator(graph(), 6, prior.dtype)
    raw, anchored = model._raw_and_anchored_states(prior, src, dst, norm)
    monomial = sum(model.monomial_prior[k] * raw[k] for k in range(4))
    anchored_mix = sum(model.gamma_prior[k] * anchored[k] for k in range(4))
    assert torch.allclose(anchored_mix, monomial, atol=1e-6, rtol=1e-6)


def test_ru_matches_previous_spgpr_uniform_forward_with_mapped_weights():
    torch.manual_seed(1001)
    pigpr = PIGPR(make_cfg("raw_uniform_protected"), DATA_INFO).eval()
    spgpr_cfg = OmegaConf.create(
        {
            "model": {
                "filter_mode": "uniform",
                "hidden_dim": 8,
                "num_layers": 3,
                "dropout": 0.2,
                "diffusion_add_self_loops": True,
                "eps": 1.0e-8,
                "structural_scale_init": 0.5,
            }
        }
    )
    previous = SPGPR(spgpr_cfg, DATA_INFO).eval()
    pig_state = pigpr.state_dict()
    previous_state = previous.state_dict()
    common = {
        key: value
        for key, value in previous_state.items()
        if key in pig_state and pig_state[key].shape == value.shape
    }
    expected_common = {
        key for key in previous_state if key.startswith(("projectors.", "fusion_", "theta_lambda_"))
    }
    assert set(common) == expected_common
    pig_state.update(common)
    pigpr.load_state_dict(pig_state)
    x = torch.randn(11, 7)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9]],
        dtype=torch.long,
    )
    actual = pigpr(x, edge_index)[0]
    expected = previous(x, edge_index)[0]
    max_abs_error = (actual - expected).abs().max().item()
    assert max_abs_error <= 1e-6, max_abs_error


def test_po_is_independent_of_graph_content():
    model = make_model("prior_only").eval()
    x = torch.randn(6, 7)
    first = graph()
    second = torch.tensor([[0, 2, 3], [5, 1, 4]])
    assert torch.equal(model(x, first)[0], model(x, second)[0])


def test_anchored_uniform_changes_representation_from_raw_uniform():
    raw = make_model("raw_uniform_protected", seed=5).eval()
    anchored = make_model("anchored_uniform_protected", seed=5).eval()
    x = torch.randn(6, 7)
    raw_z = raw(x, graph())[0]
    anchored_z = anchored(x, graph())[0]
    assert not torch.allclose(raw_z, anchored_z, atol=1e-7, rtol=1e-7)


def test_ap_fixed_prior_is_not_a_learned_delta():
    model = make_model("anchored_prior_protected").train()
    assert not model.gamma_prior.requires_grad
    assert torch.equal(model.effective_gamma(), model.gamma_prior)
    z = model(torch.randn(6, 7), graph())[0]
    z.sum().backward()
    assert model.delta_gamma_global.grad is None


def test_agp_and_agd_start_at_prior_and_can_learn_negative_and_order_three():
    for variant in ("anchored_gpr_protected", "anchored_gpr_direct"):
        model = make_model(variant).train()
        assert torch.equal(model.delta_gamma_global, torch.zeros(4))
        assert torch.equal(model.effective_gamma(), model.gamma_prior)
        output = model(torch.randn(6, 7), graph())[0]
        probe = torch.randn_like(output)
        (output * probe).sum().backward()
        grad = model.delta_gamma_global.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad[3].abs().item() > 0.0
        with torch.no_grad():
            model.delta_gamma_global[0] = -0.2
            model.delta_gamma_global[3] = 0.15
        gamma = model.effective_gamma()
        assert gamma[0] < 0
        assert gamma[3] > 0


def test_all_variants_have_identical_parameter_layout_count_and_seeded_state():
    models = {variant: make_model(variant, seed=777) for variant in VARIANTS}
    reference = models[VARIANTS[0]]
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
    model = make_model("anchored_gpr_protected")
    x = torch.randn(6, 7)
    edge_index = graph()
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
    cfg = OmegaConf.create(
        {"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}}
    )
    assert model.requires_full_lp_sampler_depth is True
    assert _resolve_lp_num_neighbors(cfg, model) == [5, 5, 5]
