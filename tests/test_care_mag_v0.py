from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.data.types import MAGData
from src.models.care_mag_v0 import Model
from src.tasks.lp import _resolve_lp_num_neighbors
from src.tasks.nc import (
    _evaluate_split,
    _resolve_nc_eval_labels,
    _should_evaluate_test,
    _training_labels,
)


DATA_INFO = {
    "input_dim": 7,
    "text_dim": 4,
    "visual_dim": 3,
    "num_nodes": 9,
    "num_classes": 3,
}


def _cfg(variant: str = "context_adapter"):
    return OmegaConf.create(
        {
            "model": {
                "name": "care_mag_v0",
                "variant": variant,
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


def _graph():
    return torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 0, 2, 5],
            [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 5, 8, 8, 2],
        ],
        dtype=torch.long,
    )


def _make_model(variant: str = "context_adapter", seed: int = 17):
    torch.manual_seed(seed)
    return Model(_cfg(variant), DATA_INFO)


def _synthetic_x():
    torch.manual_seed(31)
    return torch.randn(9, DATA_INFO["input_dim"])


def test_text_visual_split_uses_data_info_widths():
    model = _make_model()
    model.eval()
    x = _synthetic_x()
    _, _, _, _, info = model(x, _graph(), return_details=True)
    details = info["details"]
    expected_text = model.projectors["text"](x[:, : DATA_INFO["text_dim"]])
    expected_visual = model.projectors["visual"](
        x[:, DATA_INFO["text_dim"] :]
    )
    torch.testing.assert_close(details["priors"]["text"], expected_text)
    torch.testing.assert_close(details["priors"]["visual"], expected_visual)


def test_forward_shapes_are_finite_and_auxiliary_loss_is_zero():
    model = _make_model()
    z, first, second, aux_loss, _ = model(_synthetic_x(), _graph())
    assert z.shape == (9, model.out_dim)
    assert first is None and second is None
    assert aux_loss.shape == () and aux_loss.item() == 0.0
    assert torch.isfinite(z).all()


def test_trajectory_alpha_sums_to_one_and_effective_hop_is_valid():
    model = _make_model()
    model.eval()
    _, _, _, _, info = model(_synthetic_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        alpha = info["details"]["alphas"][modality]
        torch.testing.assert_close(
            alpha.sum(dim=1), torch.ones(alpha.size(0)), rtol=0.0, atol=1e-6
        )
        effective_hop = (alpha * alpha.new_tensor([1.0, 2.0, 3.0])).sum(dim=1)
        assert torch.all(effective_hop >= 1.0)
        assert torch.all(effective_hop <= 3.0)
        assert 1.0 <= float(info[modality]["effective_hop_mean"]) <= 3.0


def test_prior_only_is_independent_of_physical_edges_in_eval_mode():
    model = _make_model("prior_only")
    model.eval()
    x = _synthetic_x()
    z_a = model(x, _graph())[0]
    z_b = model(x, torch.empty((2, 0), dtype=torch.long))[0]
    torch.testing.assert_close(z_a, z_b, rtol=0.0, atol=0.0)


def test_structural_base_correction_is_exactly_zero():
    model = _make_model("structural_base")
    model.eval()
    _, _, _, _, info = model(_synthetic_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        assert torch.count_nonzero(info["details"]["deltas"][modality]) == 0


def test_static_adapter_coefficients_are_node_invariant():
    model = _make_model("static_adapter")
    model.eval()
    _, _, _, _, info = model(_synthetic_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        coefficient = info["details"]["coefficients"][modality]
        assert torch.equal(coefficient, coefficient[:1].expand_as(coefficient))


def test_context_adapter_can_produce_node_varying_coefficients():
    model = _make_model("context_adapter")
    model.eval()
    _, _, _, _, info = model(_synthetic_x(), _graph(), return_details=True)
    for modality in ("text", "visual"):
        coefficient = info["details"]["coefficients"][modality]
        assert float(coefficient.std(dim=0, unbiased=False).max()) > 1e-8
        assert float(coefficient.min()) >= 0.5
        assert float(coefficient.max()) <= 1.5


def test_all_variants_have_equal_parameter_counts():
    counts = {
        variant: sum(parameter.numel() for parameter in _make_model(variant).parameters())
        for variant in ("prior_only", "structural_base", "static_adapter", "context_adapter")
    }
    assert len(set(counts.values())) == 1


def test_all_variants_have_bitwise_identical_seed_matched_initialization():
    models = {
        variant: _make_model(variant, seed=123)
        for variant in ("prior_only", "structural_base", "static_adapter", "context_adapter")
    }
    reference = models["prior_only"].state_dict()
    for variant, model in models.items():
        assert list(model.state_dict()) == list(reference)
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference[name]), f"{variant}: {name} differs"


def test_context_global_mean_coefficient_intervention_is_deterministic():
    model = _make_model("context_adapter")
    model.eval()
    x, edge_index = _synthetic_x(), _graph()
    first = model(x, edge_index, intervention="coeff_global_mean", return_details=True)
    second = model(x, edge_index, intervention="coeff_global_mean", return_details=True)
    torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
    for modality in ("text", "visual"):
        coefficient = first[4]["details"]["coefficients"][modality]
        assert torch.equal(coefficient, coefficient[:1].expand_as(coefficient))


def test_inference_matches_eval_forward():
    model = _make_model()
    model.eval()
    x, edge_index = _synthetic_x(), _graph()
    expected = model(x, edge_index)[0].detach().cpu()
    actual = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=3)
    torch.testing.assert_close(actual, expected)


def test_gradients_are_finite():
    model = _make_model()
    model.train()
    z = model(_synthetic_x(), _graph())[0]
    z.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_structural_base_matches_context_adapter_with_adapter_off():
    structural = _make_model("structural_base", seed=987)
    context = _make_model("context_adapter", seed=987)
    structural.eval()
    context.eval()
    x, edge_index = _synthetic_x(), _graph()
    expected = structural(x, edge_index)[0]
    actual = context(x, edge_index, intervention="adapter_off")[0]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_three_layer_encoder_requires_matching_three_hop_lp_sampler():
    model = _make_model()
    assert model.num_layers == 3
    assert model.requires_full_lp_sampler_depth is True
    assert _resolve_lp_num_neighbors(_cfg(), model) == [5, 5, 5]


def _nc_cfg(*, development_no_test=True, evaluate_test=False):
    return OmegaConf.create(
        {"task": {"development_no_test": development_no_test, "evaluate_test": evaluate_test}}
    )


def test_development_no_test_blocks_test_evaluation():
    assert _should_evaluate_test(_nc_cfg()) is False
    with pytest.raises(ValueError, match="requires task.evaluate_test=false"):
        _should_evaluate_test(_nc_cfg(evaluate_test=True))


def test_development_label_set_does_not_read_test_split_labels():
    data = MAGData(
        name="synthetic",
        source="synthetic",
        task="nc",
        x=torch.zeros(3, 2),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        y=torch.tensor([0, 1, 2]),
        train_idx=torch.tensor([0]),
        val_idx=torch.tensor([1]),
        test_idx=torch.tensor([999]),
        num_classes=3,
    )
    assert _resolve_nc_eval_labels(data, development_no_test=True) == [0, 1]


def test_development_training_label_tensor_masks_test_positions():
    data = MAGData(
        name="synthetic",
        source="synthetic",
        task="nc",
        x=torch.zeros(4, 2),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        y=torch.tensor([0, 1, 2, 1]),
        train_idx=torch.tensor([0]),
        val_idx=torch.tensor([1]),
        test_idx=torch.tensor([2, 3]),
        num_classes=3,
    )
    labels = _training_labels(data, torch.device("cpu"), development_no_test=True)
    assert labels.tolist() == [0, 1, -1, -1]


def test_validation_evaluator_reports_cross_entropy():
    classifier = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    z = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 2])
    result = _evaluate_split(
        classifier,
        z,
        labels,
        torch.tensor([0, 1, 2]),
        torch.device("cpu"),
        batch_size=2,
        eval_labels=[0, 1, 2],
    )
    expected = torch.nn.functional.cross_entropy(classifier(z), labels).item()
    assert result["ce"] == pytest.approx(expected)
