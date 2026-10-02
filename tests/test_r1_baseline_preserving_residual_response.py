from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts.run_r1_baseline_preserving_residual_response import (
    SOURCE_SHA,
    audit_initialization,
    initial_identity_audit,
    response_branch_regression,
    smooth_compatibility_regression,
)
from src.models.mature_response_r0 import Model as R0Model, remove_self_messages, self_anchored_low_high
from src.models.residual_response_r1 import Model, VARIANTS, parameter_counts
from src.utils.seeds import set_seed


@pytest.fixture
def info():
    return {"input_dim": 12, "text_dim": 5, "visual_dim": 7,
            "num_nodes": 12, "num_classes": 3}


@pytest.fixture
def data():
    set_seed(10)
    x = torch.randn(12, 12)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 6],
        [0, 1, 1, 2, 2, 3, 4, 4, 7, 8, 9, 4, 9, 6],
    ])
    return SimpleNamespace(
        x=x, x_t=x[:, :5], x_i=x[:, 5:], edge_index=edge_index,
        input_dim=12, num_nodes=12, num_classes=3, y=torch.tensor(
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        ),
        train_idx=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        val_idx=torch.tensor([8, 9, 10, 11]), test_idx=None,
        info={"test_split_field_read": False},
    )


def cfg(variant="smooth_base"):
    return OmegaConf.create({"model": {
        "name": "residual_response_r1", "variant": variant, "hidden_dim": 128,
        "dropout": 0.2, "edge_chunk_size": 3, "orth_weight": 1e-3,
    }})


def test_source_and_fixed_variants():
    assert SOURCE_SHA == "d60db509e15e830a09a691665dcbe72435137477"
    assert VARIANTS == ("smooth_base", "residual_generic", "residual_protected")


def test_physical_self_edge_removal_indegree_and_low_high_formula(data):
    edge, src, dst = remove_self_messages(data.edge_index)
    assert not bool((src == dst).any())
    degree = torch.zeros(data.num_nodes, dtype=torch.long)
    degree.index_add_(0, dst, torch.ones_like(dst))
    assert int(degree.sum()) == src.numel()
    intrinsic = data.x[:, :4]
    low, high, mean = self_anchored_low_high(intrinsic, src, dst, degree, 2)
    total = torch.zeros_like(intrinsic)
    total.index_add_(0, dst, intrinsic[src])
    expected_low = (intrinsic + total) / (degree.to(intrinsic.dtype).unsqueeze(-1) + 1)
    assert torch.allclose(low, expected_low, atol=1e-7, rtol=1e-6)
    assert torch.allclose(high, intrinsic - expected_low, atol=1e-7, rtol=1e-6)
    assert torch.allclose(mean, total / degree.clamp_min(1).float().unsqueeze(-1))
    assert edge.size(1) == src.numel()


def test_smooth_bank_transforms_are_decoupled_and_bias_free(info):
    model = Model(cfg(), info)
    for modality in range(2):
        transforms = (
            model.smooth_transforms[modality],
            model.bank_low_transforms[modality],
            model.bank_high_transforms[modality],
        )
        assert all(tuple(layer.weight.shape) == (128, 128) for layer in transforms)
        assert all(layer.bias is None for layer in transforms)
        assert len({layer.weight.data_ptr() for layer in transforms}) == 3
    assert model.correction_transforms[0].bias is None
    assert model.correction_transforms[1].bias is None
    assert all(torch.count_nonzero(layer.weight) == 0 for layer in model.correction_transforms)


def test_initialization_capacity_classifier_and_exact_function_identity(info, data):
    audit = audit_initialization("Movies", 42, info, cfg())
    assert audit["all_bitwise_equal"]
    assert audit["exact_model_parameter_match"]
    assert audit["exact_classifier_parameter_match"]
    assert audit["exact_total_parameter_match"]
    assert len({row["model_trainable"] for row in audit["parameter_counts"].values()}) == 1
    result = initial_identity_audit("Movies", 42, cfg(), info, data, torch.device("cpu"))
    assert result["status"] == "passed"
    assert result["exact_zero_error"]
    assert result["max_abs_error"] == 0.0


def test_smooth_path_maps_to_m0_n1_and_r0(info, data):
    result = smooth_compatibility_regression(
        "Movies", 42, cfg(), info, data, torch.device("cpu")
    )
    assert result["status"] == "passed"
    assert result["max_abs_error"] <= 1e-5


def test_r0_low_high_crossmoe_generic_and_protected_regression(info, data):
    result = response_branch_regression(
        "Movies", 42, cfg(), info, data, torch.device("cpu")
    )
    assert result["status"] == "passed"
    assert result["max_abs_error"] <= 1e-6


def test_correction_off_shuffle_tuple_and_node_locality(info, data):
    model = Model(cfg("residual_protected"), info).eval()
    with torch.no_grad():
        for layer in model.correction_transforms:
            torch.nn.init.normal_(layer.weight, std=0.01)
    base_z, _, base_state, _, aux = model(data.x, data.edge_index, return_diagnostics=True)
    off_z, _, off_state, _, _ = model(data.x, data.edge_index, correction_off=True)
    assert torch.equal(off_state[0], aux["smooth_state"][0])
    assert torch.equal(off_state[1], aux["smooth_state"][1])
    val_idx = data.val_idx
    corrections = [value.detach().clone() for value in aux["correction"]]
    shuffled = [value.clone() for value in corrections]
    for modality in range(2):
        permutation = torch.tensor([1, 0, 3, 2])
        shuffled[modality][val_idx] = corrections[modality][val_idx[permutation]]
        assert sorted(map(tuple, shuffled[modality][val_idx].tolist())) == sorted(
            map(tuple, corrections[modality][val_idx].tolist())
        )
    moved_z, _, moved_state, _, _ = model(
        data.x, data.edge_index, correction_override=tuple(shuffled)
    )
    other = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    assert torch.equal(base_state[0][other], moved_state[0][other])
    assert torch.equal(base_state[1][other], moved_state[1][other])
    assert torch.equal(base_z[other], moved_z[other])


def test_no_test_or_lp_is_attached_to_synthetic_nc_data(data):
    assert data.test_idx is None
    assert data.info["test_split_field_read"] is False
    assert not hasattr(data, "edge_label_index")
