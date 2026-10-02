from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.pigpr_mag_v0 import Model as PIGPRv0
from src.models.pigpr_mag_v1 import Model as PIGPRv1
from src.tasks.lp import _resolve_lp_num_neighbors
from src.utils.summary import count_parameters


VARIANTS = (
    "raw_uniform_protected",
    "raw_uniform_direct",
    "raw_uniform0_direct",
    "raw_fixed_prior_direct",
    "raw_gpr_direct",
    "anchored_gpr_direct",
)
DATA_INFO = {"input_dim": 7, "text_dim": 3, "visual_dim": 4}


def make_cfg(variant: str = "raw_gpr_direct"):
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


def make_model(variant: str = "raw_gpr_direct", seed: int = 123):
    torch.manual_seed(seed)
    return PIGPRv1(make_cfg(variant), DATA_INFO)


def graph(num_nodes: int = 6):
    src, dst = [], []
    for node in range(num_nodes):
        other = (node + 1) % num_nodes
        src.extend((node, other))
        dst.extend((other, node))
    return torch.tensor([src, dst], dtype=torch.long)


def test_split_uses_declared_modalities_and_widths():
    model = make_model()
    x = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    split = model._split_modalities(x)
    assert torch.equal(split["text"], x[:, :3])
    assert torch.equal(split["visual"], x[:, 3:])


def test_raw_states_follow_three_hop_recurrence():
    model = make_model().eval()
    prior = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    src, dst, norm = model._normalized_operator(graph(), 6, prior.dtype)
    raw, _ = model._raw_and_anchored_states(prior, src, dst, norm)
    assert torch.equal(raw[0], prior)
    for order in range(1, 4):
        assert torch.equal(raw[order], model._propagate(raw[order - 1], src, dst, norm))


def test_anchored_states_follow_restart_recurrence():
    model = make_model().eval()
    prior = torch.randn(6, 5)
    src, dst, norm = model._normalized_operator(graph(), 6, prior.dtype)
    _, anchored = model._raw_and_anchored_states(prior, src, dst, norm)
    assert torch.equal(anchored[0], prior)
    for order in range(1, 4):
        expected = 0.9 * model._propagate(anchored[order - 1], src, dst, norm) + 0.1 * prior
        assert torch.equal(anchored[order], expected)


def test_anchored_basis_matrix_matches_recurrence_coefficients():
    model = make_model()
    expected = torch.tensor(
        [[1.0, 0, 0, 0], [0.1, 0.9, 0, 0], [0.1, 0.09, 0.81, 0], [0.1, 0.09, 0.081, 0.729]]
    )
    assert torch.allclose(model.anchored_basis_matrix, expected, atol=1e-7)


def test_monomial_conversion_is_transpose_basis_map():
    model = make_model()
    gamma = torch.tensor([0.3, -0.2, 0.7, 0.1])
    expected = model.anchored_basis_matrix.T @ gamma
    assert torch.equal(model.equivalent_monomial_coefficients(gamma), expected)


def test_coSI_prior_is_formula_derived_and_matches_anchor_prior():
    model = make_model()
    expected_c = torch.tensor([0.15, 0.1275, 0.7225, 0.0])
    expected_gamma = torch.tensor([0.0555555556, 0.0524691358, 0.8919753086, 0.0])
    assert torch.allclose(model.monomial_prior, expected_c, atol=1e-7)
    assert torch.allclose(model.gamma_prior, expected_gamma, atol=1e-7)
    assert torch.allclose(model.anchored_basis_matrix.T @ model.gamma_prior, model.monomial_prior, atol=1e-7)


def test_raw_and_anchored_polynomials_are_equivalent_on_synthetic_graph():
    model = make_model().eval()
    prior = torch.randn(6, 5)
    src, dst, norm = model._normalized_operator(graph(), 6, prior.dtype)
    raw, anchored = model._raw_and_anchored_states(prior, src, dst, norm)
    gamma = torch.tensor([0.2, -0.1, 0.4, 0.3])
    coefficients = model.equivalent_monomial_coefficients(gamma)
    raw_mix = sum(coefficients[k] * raw[k] for k in range(4))
    anchored_mix = sum(gamma[k] * anchored[k] for k in range(4))
    assert torch.allclose(raw_mix, anchored_mix, atol=1e-6, rtol=1e-6)


def test_ru_regresses_to_pigpr_v0_ru():
    torch.manual_seed(1001)
    current = PIGPRv1(make_cfg("raw_uniform_protected"), DATA_INFO).eval()
    previous_cfg = make_cfg("raw_uniform_protected")
    previous = PIGPRv0(previous_cfg, DATA_INFO).eval()
    current_state = current.state_dict()
    previous_state = previous.state_dict()
    common = {
        key: value
        for key, value in previous_state.items()
        if key in current_state
        and current_state[key].shape == value.shape
        and key.startswith(("projectors.", "fusion_", "theta_lambda_"))
    }
    expected_keys = {
        key
        for key in previous_state
        if key.startswith(("projectors.", "fusion_", "theta_lambda_"))
    }
    assert set(common) == expected_keys
    current_state.update(common)
    current.load_state_dict(current_state)
    x = torch.randn(11, 7)
    edge = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9]],
        dtype=torch.long,
    )
    actual = current(x, edge)[0]
    expected = previous(x, edge)[0]
    assert (actual - expected).abs().max().item() <= 1e-6


def test_rud_proposal_matches_ru_before_output_composition():
    ru = make_model("raw_uniform_protected", 10).eval()
    rud = make_model("raw_uniform_direct", 10).eval()
    x, edge = torch.randn(6, 7), graph()
    ru_info = ru(x, edge, return_details=True)[4]
    rud_info = rud(x, edge, return_details=True)[4]
    for modality in ("text", "visual"):
        assert torch.equal(
            ru_info["details"]["modalities"][modality]["proposal"],
            rud_info["details"]["modalities"][modality]["proposal"],
        )


def test_r0ud_uses_intrinsic_order_zero_exactly():
    model = make_model("raw_uniform0_direct").eval()
    info = model(torch.randn(6, 7), graph(), return_details=True)[4]
    detail = info["details"]["modalities"]["text"]
    expected = sum(0.25 * detail["raw_states"][:, k] for k in range(4))
    assert torch.equal(detail["proposal"], expected)
    assert torch.equal(detail["coefficients_raw"], torch.full((4,), 0.25))


def test_rfd_coefficients_are_exact_fixed_prior_and_not_trainable_in_forward():
    model = make_model("raw_fixed_prior_direct").train()
    assert torch.equal(model.effective_coefficients(), model.monomial_prior)
    x, edge = torch.randn(6, 7), graph()
    before = model.effective_coefficients().clone()
    model(x, edge)[0].sum().backward()
    assert torch.equal(model.effective_coefficients(), before)
    assert model.delta_c_raw.grad is None


def test_rgd_starts_with_zero_delta_and_prior_coefficients():
    model = make_model("raw_gpr_direct")
    assert torch.equal(model.delta_c_raw, torch.zeros(4))
    assert torch.equal(model.effective_coefficients(), model.monomial_prior)


def test_agd_starts_with_prior_gamma():
    model = make_model("anchored_gpr_direct")
    assert torch.equal(model.delta_gamma_anchor, torch.zeros(4))
    assert torch.equal(model.effective_gamma(), model.gamma_prior)


def test_rfd_and_rgd_initial_proposals_and_modality_embeddings_match():
    rfd, rgd = make_model("raw_fixed_prior_direct", 32).eval(), make_model("raw_gpr_direct", 32).eval()
    x, edge = torch.randn(6, 7), graph()
    a, b = rfd(x, edge, return_details=True)[4], rgd(x, edge, return_details=True)[4]
    for modality in ("text", "visual"):
        da = a["details"]["modalities"][modality]
        db = b["details"]["modalities"][modality]
        assert torch.equal(da["proposal"], db["proposal"])
        assert torch.equal(da["embedding"], db["embedding"])


def test_rgd_and_agd_initial_proposals_and_modality_embeddings_match():
    rgd, agd = make_model("raw_gpr_direct", 45).eval(), make_model("anchored_gpr_direct", 45).eval()
    x, edge = torch.randn(6, 7), graph()
    a, b = rgd(x, edge, return_details=True)[4], agd(x, edge, return_details=True)[4]
    for modality in ("text", "visual"):
        da = a["details"]["modalities"][modality]
        db = b["details"]["modalities"][modality]
        assert torch.allclose(da["proposal"], db["proposal"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(da["embedding"], db["embedding"], atol=1e-6, rtol=1e-6)


def test_learned_raw_coefficients_can_be_negative_and_order_three_is_trainable():
    model = make_model("raw_gpr_direct").train()
    z = model(torch.randn(6, 7), graph())[0]
    (z * torch.randn_like(z)).sum().backward()
    assert model.delta_c_raw.grad is not None
    assert torch.isfinite(model.delta_c_raw.grad).all()
    assert model.delta_c_raw.grad[3].abs() > 0
    with torch.no_grad():
        model.delta_c_raw[0] = -0.5
    assert model.effective_coefficients()[0] < 0


def test_learned_anchor_coefficients_can_be_negative_and_order_three_is_trainable():
    model = make_model("anchored_gpr_direct").train()
    z = model(torch.randn(6, 7), graph())[0]
    (z * torch.randn_like(z)).sum().backward()
    assert model.delta_gamma_anchor.grad is not None
    assert torch.isfinite(model.delta_gamma_anchor.grad).all()
    assert model.delta_gamma_anchor.grad[3].abs() > 0
    with torch.no_grad():
        model.delta_gamma_anchor[0] = -0.5
    assert model.effective_gamma()[0] < 0


def test_all_variants_have_equal_parameter_count_and_state_layout():
    models = {variant: make_model(variant, 777) for variant in VARIANTS}
    reference = models[VARIANTS[0]]
    expected = reference.state_dict()
    for model in models.values():
        state = model.state_dict()
        assert list(state) == list(expected)
        assert {key: value.shape for key, value in state.items()} == {
            key: value.shape for key, value in expected.items()
        }
        assert count_parameters(model) == count_parameters(reference)


def test_same_seed_initialization_is_bitwise_identical_for_all_variants():
    models = [make_model(variant, 2026) for variant in VARIANTS]
    reference = models[0].state_dict()
    for model in models[1:]:
        for key, value in model.state_dict().items():
            assert torch.equal(value, reference[key]), key


def test_direct_models_keep_lambda_parameters_and_initialize_at_half():
    for variant in VARIANTS:
        model = make_model(variant)
        assert model.theta_lambda_text.shape == torch.Size([])
        assert model.theta_lambda_visual.shape == torch.Size([])
        assert torch.sigmoid(model.theta_lambda_text).item() == 0.5
        assert torch.sigmoid(model.theta_lambda_visual).item() == 0.5


def test_inference_matches_eval_forward():
    model = make_model("anchored_gpr_direct").eval()
    x, edge = torch.randn(6, 7), graph()
    expected = model(x, edge)[0].detach()
    actual = model.inference(x, edge, device=torch.device("cpu"))
    assert torch.allclose(actual, expected.cpu())


def test_gradient_values_are_finite():
    model = make_model("raw_gpr_direct").train()
    z, _, _, aux_loss, _ = model(torch.randn(6, 7), graph())
    (z.square().mean() + aux_loss).backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_lp_sampler_depth_resolves_to_three_hops():
    model = make_model()
    cfg = OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    assert model.requires_full_lp_sampler_depth is True
    assert _resolve_lp_num_neighbors(cfg, model) == [5, 5, 5]


def test_auxiliary_loss_is_zero_and_output_interface_is_preserved():
    model = make_model().eval()
    z, first, second, aux_loss, _ = model(torch.randn(6, 7), graph())
    assert model.out_dim == 8
    assert z.shape == (6, 8)
    assert first is None and second is None
    assert aux_loss.item() == 0.0


def test_checkpoint_interventions_recompose_in_raw_basis():
    model = make_model("anchored_gpr_direct").eval()
    x, edge = torch.randn(6, 7), graph()
    normal = model(x, edge, return_details=True)[4]["details"]["modalities"]["text"]
    # The analyzer uses the regular no-details path for lower memory use.
    assert torch.isfinite(model(x, edge, intervention="reset_prior")[0]).all()
    reset = model(x, edge, intervention="reset_prior", return_details=True)[4]["details"]["modalities"]["text"]
    assert torch.allclose(reset["proposal"], sum(model.monomial_prior[k] * normal["raw_states"][:, k] for k in range(4)), atol=1e-6)


def test_all_gpr_interventions_work_without_details():
    x, edge = torch.randn(6, 7), graph()
    for variant in ("raw_gpr_direct", "anchored_gpr_direct"):
        model = make_model(variant).eval()
        for intervention in ("reset_prior", "order0_off", "order3_off", "negative_terms_off"):
            assert torch.isfinite(model(x, edge, intervention=intervention)[0]).all()
