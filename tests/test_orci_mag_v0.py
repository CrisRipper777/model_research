from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

import src.models.orci_mag_v0 as orci_module
from src.models.orci_mag_v0 import Model as OrciModel
from src.models.pigpr_mag_v1 import Model as C1Model
from src.tasks.common import resolve_num_neighbors


def _cfg(variant: str, alignment_weight: float = 0.05):
    return OmegaConf.create(
        {
            "model": {
                "name": "orci_mag_v0",
                "variant": variant,
                "hidden_dim": 256,
                "num_layers": 3,
                "dropout": 0.2,
                "diffusion_add_self_loops": True,
                "eps": 1.0e-8,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "interaction_dim": 64,
                "num_heads": 4,
                "head_dim": 16,
                "interaction_dropout": 0.1,
                "interaction_scale_init": 0.1,
                "relative_order_bias_init": 0.25,
                "alignment_weight": alignment_weight,
            }
        }
    )


def _data_info():
    return {"input_dim": 7, "text_dim": 4, "visual_dim": 3}


def _graph(n: int = 8):
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return edge_index


def _features(n: int = 8):
    generator = torch.Generator().manual_seed(31415)
    return torch.randn((n, 7), generator=generator)


def _model(variant: str, alignment_weight: float = 0.05):
    return OrciModel(_cfg(variant, alignment_weight), _data_info())


def _eval(model, variant: str | None = None):
    model.eval()
    return model(_features(), _graph(), return_details=True)


def test_modality_split_uses_configured_text_then_visual_widths():
    model = _model("base")
    split = model._split_modalities(_features())
    assert split["text"].shape == (8, 4)
    assert split["visual"].shape == (8, 3)
    torch.testing.assert_close(torch.cat([split["text"], split["visual"]], 1), _features())


def test_base_is_numerical_regression_to_c1_raw_gpr_direct():
    torch.manual_seed(892)
    c1 = C1Model(
        OmegaConf.create(
            {
                "model": {
                    "variant": "raw_gpr_direct",
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "dropout": 0.2,
                    "eps": 1.0e-8,
                    "anchor_alpha": 0.1,
                    "global_prior_restart": 0.15,
                    "global_prior_order": 2,
                    "diffusion_add_self_loops": True,
                }
            }
        ),
        _data_info(),
    )
    torch.manual_seed(892)
    base = _model("base")
    for name in (
        "projectors.text.linear1.weight",
        "projectors.visual.linear2.weight",
        "delta_c_raw",
        "fusion_linear1.weight",
        "fusion_linear2.weight",
        "fusion_skip.weight",
    ):
        torch.testing.assert_close(
            base.state_dict()[name], c1.state_dict()[name], rtol=0, atol=0
        )
    c1.eval()
    base.eval()
    z_c1 = c1(_features(), _graph())[0]
    z_base = base(_features(), _graph())[0]
    torch.testing.assert_close(z_base, z_c1, rtol=0, atol=0)


def test_h0_is_never_updated_by_interaction():
    model = _model("joint")
    _, _, _, _, info = _eval(model)
    details = info["details"]["modalities"]
    for modality in ("text", "visual"):
        torch.testing.assert_close(
            details[modality]["updated_states"][:, 0],
            details[modality]["raw_states"][:, 0],
            rtol=0,
            atol=0,
        )


def test_interaction_token_r0_equals_h0():
    model = _model("joint")
    _, _, _, _, info = _eval(model)
    for modality in ("text", "visual"):
        detail = info["details"]["modalities"][modality]
        torch.testing.assert_close(
            detail["interaction_tokens"][:, 0], detail["raw_states"][:, 0]
        )


def test_interaction_tokens_r1_to_r3_are_hk_minus_h0():
    model = _model("joint")
    _, _, _, _, info = _eval(model)
    for modality in ("text", "visual"):
        detail = info["details"]["modalities"][modality]
        expected = detail["raw_states"][:, 1:] - detail["raw_states"][:, :1]
        torch.testing.assert_close(detail["interaction_tokens"][:, 1:], expected)


def test_attention_query_key_value_shapes_follow_order_specification():
    model = _model("interaction")
    _, _, _, _, info = _eval(model)
    assert info["details"]["attention_shapes"] == {
        "query": (8, 4, 3, 16),
        "key": (8, 4, 4, 16),
        "value": (8, 4, 4, 16),
    }


def test_attention_tensor_shape_is_node_head_query_source():
    model = _model("interaction")
    _, _, _, _, info = _eval(model)
    for weights in info["details"]["attention"].values():
        assert weights.shape == (8, 4, 3, 4)


def test_attention_rows_sum_to_one_over_all_source_orders():
    model = _model("interaction")
    _, _, _, _, info = _eval(model)
    for weights in info["details"]["attention"].values():
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones((8, 4, 3)), rtol=1e-6, atol=1e-6
        )


def test_bidirectional_interaction_outputs_and_attention_are_finite():
    model = _model("interaction")
    _, _, _, _, info = _eval(model)
    for value in info["details"]["attention"].values():
        assert torch.isfinite(value).all()
    for modality in ("text", "visual"):
        assert torch.isfinite(info["details"]["modalities"][modality]["interaction_output"]).all()


def test_relative_order_bias_indexes_source_minus_query_delta():
    model = _model("interaction")
    expected = torch.tensor([[2, 3, 4, 5], [1, 2, 3, 4], [0, 1, 2, 3]])
    torch.testing.assert_close(model.relative_order_index.cpu(), expected)


def test_initial_relative_bias_favors_same_order():
    model = _model("interaction")
    bias = model.relative_order_bias.detach()
    torch.testing.assert_close(bias[:, 3], torch.zeros(4))
    assert torch.all(bias[:, 3] > bias[:, 2])
    assert torch.all(bias[:, 3] > bias[:, 4])


def test_cross_order_attention_is_not_masked():
    model = _model("interaction")
    _, _, _, _, info = _eval(model)
    for weights in info["details"]["attention"].values():
        assert torch.all(weights > 0)


def test_global_order_scales_initialize_to_point_one():
    model = _model("interaction")
    torch.testing.assert_close(torch.sigmoid(model.theta_rho), torch.full((3,), 0.1))


def test_interaction_changes_higher_orders_and_preserves_raw_backbone():
    torch.manual_seed(88)
    base = _model("base")
    torch.manual_seed(88)
    interaction = _model("interaction")
    interaction.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    interaction.eval()
    _, _, _, _, base_info = base(_features(), _graph(), return_details=True)
    _, _, _, _, int_info = interaction(_features(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        base_states = base_info["details"]["modalities"][modality]["raw_states"]
        changed = int_info["details"]["modalities"][modality]["updated_states"]
        assert torch.equal(changed[:, 0], base_states[:, 0])
        assert not torch.allclose(changed[:, 1:], base_states[:, 1:])


def test_alignment_only_main_forward_matches_base_under_same_weights():
    torch.manual_seed(101)
    base = _model("base")
    alignment = _model("alignment")
    alignment.load_state_dict(base.state_dict())
    base.eval()
    alignment.eval()
    z_base = base(_features(), _graph())[0]
    z_alignment = alignment(_features(), _graph())[0]
    torch.testing.assert_close(z_alignment, z_base, rtol=0, atol=0)


def test_joint_and_interaction_main_forward_match_in_eval_mode():
    torch.manual_seed(102)
    interaction = _model("interaction")
    joint = _model("joint")
    joint.load_state_dict(interaction.state_dict())
    interaction.eval()
    joint.eval()
    z_i = interaction(_features(), _graph())[0]
    z_ia = joint(_features(), _graph())[0]
    torch.testing.assert_close(z_ia, z_i, rtol=0, atol=0)


def test_base_aux_loss_is_exact_zero():
    assert _eval(_model("base"))[3].item() == 0.0


def test_interaction_aux_loss_is_exact_zero():
    assert _eval(_model("interaction"))[3].item() == 0.0


@pytest.mark.parametrize("variant", ["alignment", "joint"])
def test_alignment_variants_return_finite_nonnegative_aux_loss(variant):
    aux_loss = _eval(_model(variant))[3]
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    assert aux_loss >= 0


def test_alignment_uses_only_three_structural_orders():
    _, _, _, aux_loss, info = _eval(_model("alignment", 0.05))
    details = info["details"]
    cosine = details["pair_cosine"]
    assert cosine.shape == (8, 3)
    text = torch.nn.functional.normalize(details["content_tokens"]["text"][:, 1:], dim=-1)
    visual = torch.nn.functional.normalize(details["content_tokens"]["visual"][:, 1:], dim=-1)
    expected = torch.nn.functional.cosine_similarity(text, visual, dim=-1)
    torch.testing.assert_close(cosine, expected)
    torch.testing.assert_close(aux_loss, 0.05 * (1 - expected).mean())


def test_content_projection_is_one_shared_query_key_alignment_module():
    model = _model("joint")
    named = dict(model.named_parameters())
    assert "content_proj.weight" in named
    assert sum(name.endswith("content_proj.weight") for name in named) == 1
    assert model.content_proj is model.content_proj


def test_variant_parameter_counts_are_identical():
    counts = {
        sum(parameter.numel() for parameter in _model(variant).parameters())
        for variant in ("base", "interaction", "alignment", "joint")
    }
    assert len(counts) == 1


def test_variant_state_dict_layouts_are_identical():
    layouts = [tuple(_model(variant).state_dict()) for variant in ("base", "interaction", "alignment", "joint")]
    assert all(layout == layouts[0] for layout in layouts[1:])


def test_same_seed_variant_parameters_are_bitwise_identical():
    models = []
    for variant in ("base", "interaction", "alignment", "joint"):
        torch.manual_seed(2048)
        models.append(_model(variant))
    reference = models[0].state_dict()
    for model in models[1:]:
        for key, value in model.state_dict().items():
            assert torch.equal(value, reference[key]), key


def test_all_active_joint_forward_gradients_are_finite():
    model = _model("joint")
    model.train()
    z, _, _, aux_loss, _ = model(_features(), _graph())
    (z.square().mean() + aux_loss).backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("variant", ["alignment", "joint"])
def test_alignment_loss_gradients_reach_content_projection_and_projectors(variant):
    model = _model(variant)
    model.zero_grad(set_to_none=True)
    _, _, _, aux_loss, _ = model(_features(), _graph())
    aux_loss.backward()
    assert model.content_proj.weight.grad is not None
    assert torch.isfinite(model.content_proj.weight.grad).all()
    assert model.projectors["text"].linear1.weight.grad is not None
    assert model.projectors["visual"].linear1.weight.grad is not None


def test_inference_matches_eval_forward():
    model = _model("joint")
    model.eval()
    expected = model(_features(), _graph())[0].detach().cpu()
    actual = model.inference(_features(), _graph(), device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a working CUDA device")
def test_cpu_staged_large_features_match_gpu_resident_features():
    device = torch.device("cuda")
    model = _model("base").to(device).eval()
    features = torch.randn((8197, 7), generator=torch.Generator().manual_seed(71))
    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    staged = model(features, edge_index)[0]
    resident = model(features.to(device), edge_index)[0]
    torch.testing.assert_close(staged, resident, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a working CUDA device")
def test_cpu_staged_projector_checkpoint_preserves_cuda_dropout_rng_and_gradients():
    device = torch.device("cuda")
    direct = _model("base").to(device).train()
    checkpointed = _model("base").to(device).train()
    checkpointed.load_state_dict(direct.state_dict())
    features = torch.randn((8195, 4), generator=torch.Generator().manual_seed(72))

    torch.manual_seed(812)
    expected = direct._project_modality(features, "text", False)
    expected.square().mean().backward()
    expected_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in direct.projectors["text"].named_parameters()
    }

    torch.manual_seed(812)
    actual = checkpointed._project_modality(features, "text", True)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for name, parameter in checkpointed.projectors["text"].named_parameters():
        torch.testing.assert_close(parameter.grad, expected_grads[name], rtol=2e-5, atol=2e-6)


def test_full_lp_sampler_depth_resolves_to_five_five_five():
    model = _model("joint")
    cfg = OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    assert model.requires_full_lp_sampler_depth is True
    assert resolve_num_neighbors(cfg) == [5, 5, 5]


def test_model_exposes_expected_output_dimension_and_gpr_prior():
    model = _model("base")
    assert model.out_dim == 256
    torch.testing.assert_close(model.c_prior, torch.tensor([0.15, 0.1275, 0.7225, 0.0]))
    torch.testing.assert_close(model.delta_c_raw, torch.zeros(4))


def test_large_graph_chunked_propagation_matches_single_index_add_and_gradients():
    num_nodes, hidden_dim, num_edges = 2048, 16, 200_001
    generator = torch.Generator().manual_seed(42)
    src = torch.randint(num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(num_nodes, (num_edges,), generator=generator)
    norm = torch.rand(num_edges, generator=generator)
    base = torch.randn((num_nodes, hidden_dim), generator=generator)

    single_state = base.clone().requires_grad_()
    single_messages = single_state[src] * norm.unsqueeze(-1)
    single = torch.zeros_like(single_state).index_add(0, dst, single_messages)
    single.square().mean().backward()

    chunked_state = base.clone().requires_grad_()
    chunked = OrciModel._propagate(chunked_state, src, dst, norm)
    chunked.square().mean().backward()
    torch.testing.assert_close(chunked, single, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(chunked_state.grad, single_state.grad, rtol=1e-5, atol=1e-7)
    assert torch.isfinite(chunked_state.grad).all()


def test_large_graph_activation_checkpoint_preserves_dropout_outputs_and_gradients(monkeypatch):
    monkeypatch.setattr(orci_module, "_ACTIVATION_CHECKPOINT_NODE_THRESHOLD", 1)
    reference = _model("base")
    checkpointed = _model("base")
    checkpointed.load_state_dict(reference.state_dict())
    reference._enable_activation_checkpointing = False
    reference.train()
    checkpointed.train()

    torch.manual_seed(7001)
    z_reference = reference(_features(), _graph())[0]
    z_reference.square().mean().backward()
    reference_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }

    checkpointed.zero_grad(set_to_none=True)
    torch.manual_seed(7001)
    z_checkpointed = checkpointed(_features(), _graph())[0]
    z_checkpointed.square().mean().backward()
    torch.testing.assert_close(z_checkpointed, z_reference, rtol=1e-5, atol=1e-6)
    for name, parameter in checkpointed.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=1e-5, atol=1e-7)


def test_streaming_gpr_matches_explicit_mixture_and_analytic_gradients():
    num_nodes, hidden_dim, num_edges = 2048, 16, 200_001
    generator = torch.Generator().manual_seed(91)
    src = torch.randint(num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(num_nodes, (num_edges,), generator=generator)
    norm = torch.rand(num_edges, generator=generator)
    base_prior = torch.randn((num_nodes, hidden_dim), generator=generator)
    base_coefficients = torch.randn(4, generator=generator)

    from src.models.orci_mag_v0 import _StreamingRawGPR

    prior_ref = base_prior.clone().requires_grad_()
    coeff_ref = base_coefficients.clone().requires_grad_()
    states = [prior_ref]
    for _ in range(3):
        states.append(OrciModel._propagate(states[-1], src, dst, norm))
    expected = sum(coeff_ref[k] * states[k] for k in range(4))
    expected.square().mean().backward()

    prior_stream = base_prior.clone().requires_grad_()
    coeff_stream = base_coefficients.clone().requires_grad_()
    actual = _StreamingRawGPR.apply(prior_stream, coeff_stream, src, dst, norm)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(prior_stream.grad, prior_ref.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(coeff_stream.grad, coeff_ref.grad, rtol=1e-5, atol=1e-6)


def test_large_alignment_loss_recomputation_matches_full_loss_and_gradients(monkeypatch):
    monkeypatch.setattr(orci_module, "_ACTIVATION_CHECKPOINT_NODE_THRESHOLD", 1)
    reference = _model("alignment", 0.05)
    checkpointed = _model("alignment", 0.05)
    checkpointed.load_state_dict(reference.state_dict())
    reference._enable_activation_checkpointing = False
    reference.train()
    checkpointed.train()

    torch.manual_seed(512)
    z_ref, _, _, aux_ref, _ = reference(_features(), _graph())
    (z_ref.square().mean() + aux_ref).backward()
    ref_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }

    checkpointed.zero_grad(set_to_none=True)
    torch.manual_seed(512)
    z_ckpt, _, _, aux_ckpt, _ = checkpointed(_features(), _graph())
    (z_ckpt.square().mean() + aux_ckpt).backward()
    torch.testing.assert_close(z_ckpt, z_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(aux_ckpt, aux_ref, rtol=1e-5, atol=1e-7)
    for name, parameter in checkpointed.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, ref_grads[name], rtol=1e-4, atol=1e-6)


def test_large_interaction_correction_matches_explicit_detail_forward(monkeypatch):
    monkeypatch.setattr(orci_module, "_ACTIVATION_CHECKPOINT_NODE_THRESHOLD", 1)
    model = _model("joint")
    model.eval()
    explicit = model(_features(), _graph(), return_details=True)[0]
    efficient = model(_features(), _graph())[0]
    torch.testing.assert_close(efficient, explicit, rtol=1e-5, atol=1e-6)


def test_large_interaction_checkpoint_preserves_dropout_loss_and_gradients(monkeypatch):
    monkeypatch.setattr(orci_module, "_ACTIVATION_CHECKPOINT_NODE_THRESHOLD", 1)
    reference = _model("joint", 0.05)
    checkpointed = _model("joint", 0.05)
    checkpointed.load_state_dict(reference.state_dict())
    reference._enable_activation_checkpointing = False
    reference.train()
    checkpointed.train()

    torch.manual_seed(9501)
    z_ref, _, _, aux_ref, _ = reference(_features(), _graph())
    (z_ref.square().mean() + aux_ref).backward()
    ref_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }

    checkpointed.zero_grad(set_to_none=True)
    torch.manual_seed(9501)
    z_ckpt, _, _, aux_ckpt, _ = checkpointed(_features(), _graph())
    (z_ckpt.square().mean() + aux_ckpt).backward()
    torch.testing.assert_close(z_ckpt, z_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(aux_ckpt, aux_ref, rtol=1e-5, atol=1e-7)
    for name, parameter in checkpointed.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, ref_grads[name], rtol=1e-4, atol=1e-6)


def test_source_node_shuffle_changes_only_cross_modal_sources():
    model = _model("interaction")
    model.eval()
    permutation = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4])
    normal = model(_features(), _graph())[0]
    shuffled = model(
        _features(),
        _graph(),
        intervention="source_node_shuffle",
        source_node_permutation=permutation,
    )[0]
    assert not torch.allclose(normal, shuffled)


def test_interaction_off_sets_all_residual_interactions_to_zero():
    model = _model("joint")
    model.eval()
    _, _, _, _, info = model(_features(), _graph(), intervention="interaction_off", return_details=True)
    for modality in ("text", "visual"):
        detail = info["details"]["modalities"][modality]
        assert torch.count_nonzero(detail["scaled_interaction"]) == 0
        torch.testing.assert_close(
            detail["updated_states"], detail["raw_states"], rtol=0, atol=0
        )


def test_large_interaction_off_uses_streamed_gpr_and_matches_base(monkeypatch):
    monkeypatch.setattr(orci_module, "_ACTIVATION_CHECKPOINT_NODE_THRESHOLD", 1)
    base = _model("base").eval()
    joint = _model("joint").eval()
    joint.load_state_dict(base.state_dict(), strict=False)

    expected = base(_features(), _graph())[0]
    actual = joint(_features(), _graph(), intervention="interaction_off")[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
