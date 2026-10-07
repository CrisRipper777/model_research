from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.models.sosb_mag_v15 import Model
from src.tasks.nc import _resolve_nc_eval_labels


VARIANTS = (
    "A0_legacy_lg",
    "A1_rawpoly_shared",
    "A2_sosb_shared",
    "A3_sosb_modality",
)


def _cfg(variant: str):
    return OmegaConf.create(
        {
            "model": {
                "name": "sosb_mag_v15",
                "variant": variant,
                "hidden_dim": 256,
                "num_layers": 4,
                "dropout": 0.0,
                "expert_bottleneck": 64,
                "basis_order": 4,
                "eps": 1.0e-8,
                "basis_breakdown_eps": 1.0e-5,
                "gate_init": -2.0,
            }
        }
    )


def _data_info():
    return {"input_dim": 11, "text_dim": 6, "visual_dim": 5, "num_nodes": 6, "num_classes": 3}


def _path_graph():
    # Five active nodes in a path and one isolated node.
    src = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4])
    dst = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3])
    return torch.stack([src, dst], dim=0)


def test_all_variants_forward_backward_and_diagnostics_are_finite():
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Model(_cfg(variant), _data_info())
        model.train()
        z, _, _, aux, info = model(x, _path_graph(), return_details=True)
        assert z.shape == (6, 256)
        assert torch.isfinite(z).all() and torch.isfinite(aux)
        assert info["variant"] == variant
        for modality in ("text", "visual"):
            details = info["details"][modality]
            assert torch.isfinite(details["prior"]).all()
            assert torch.isfinite(details["basis"]).all()
            assert torch.isfinite(details["beta"]).all()
            assert torch.isfinite(details["gate"]).all()
        (z.square().mean() + aux).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_common_initialization_is_variant_invariant():
    snapshots = {}
    for variant in VARIANTS:
        torch.manual_seed(123)
        model = Model(_cfg(variant), _data_info())
        classifier = torch.nn.Linear(model.out_dim, _data_info()["num_classes"])
        common_prefixes = (
            "projectors.",
            "fusion_linear1.",
            "fusion_linear2.",
            "fusion_skip.",
            "fusion_norm.",
        )
        snapshots[variant] = {
            "common": {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
                if name.startswith(common_prefixes)
            },
            "classifier": {k: v.detach().clone() for k, v in classifier.state_dict().items()},
        }
    reference = snapshots[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        current = snapshots[variant]
        assert current["common"].keys() == reference["common"].keys()
        for name, value in reference["common"].items():
            assert torch.equal(value, current["common"][name]), (variant, name)
        for name, value in reference["classifier"].items():
            assert torch.equal(value, current["classifier"][name]), (variant, name)


def test_isolated_nodes_remain_exactly_on_intrinsic_path():
    x = torch.randn(6, 11)
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, _path_graph(), return_details=True)
        for modality in ("text", "visual"):
            details = info["details"][modality]
            assert torch.count_nonzero(details["basis"][:, 5]) == 0
            assert torch.equal(details["output"][5], details["prior"][5])


def _gram(basis: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    # basis is [order, nodes, channels]; average channelwise inner products.
    selected = basis[:, active]
    return torch.einsum("knd,lnd->kl", selected, selected) / selected.size(1) / selected.size(2)


def test_raw_polynomial_basis_is_not_forced_orthogonal():
    torch.manual_seed(18)
    model = Model(_cfg("A1_rawpoly_shared"), _data_info()).eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    active = info["active_nodes"]
    gram = _gram(info["details"]["text"]["basis"], active)
    offdiag = gram - torch.diag(torch.diag(gram))
    assert offdiag.abs().max() > 1.0e-3


def test_sosb_basis_is_orthogonal_on_stable_toy_channels():
    torch.manual_seed(37)
    model = Model(_cfg("A2_sosb_shared"), _data_info()).eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
    details = info["details"]["text"]
    basis = details["basis"]
    breakdown = details["breakdown"]
    # Exclude channels near the regularization scale: eps=1e-8 intentionally
    # prevents exact unit RMS for those near-breakdown residuals.
    stable_channels = (~breakdown.any(dim=0)) & (
        details["residual_rms"].min(dim=0).values > 1.0e-2
    )
    assert stable_channels.any()
    stable = basis[:, :, stable_channels]
    per_channel_gram = torch.einsum(
        "knd,lnd->kld", stable[:, info["active_nodes"]], stable[:, info["active_nodes"]]
    ) / info["active_nodes"].sum()
    eye = torch.eye(4, dtype=per_channel_gram.dtype).unsqueeze(-1)
    assert torch.max(torch.abs(per_channel_gram - eye)) < 1.0e-4
    aggregate = _gram(basis, info["active_nodes"])
    assert torch.allclose(torch.diag(aggregate), torch.ones(4), atol=1.0e-4, rtol=0.0)


def test_breakdown_channels_are_zero_and_finite():
    # On one edge the channelwise Krylov span has dimension at most two.
    x = torch.randn(3, 11)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    model = Model(_cfg("A2_sosb_shared"), _data_info()).eval()
    with torch.no_grad():
        _, _, _, _, info = model(x, edge_index, return_details=True)
    for modality in ("text", "visual"):
        details = info["details"][modality]
        assert details["breakdown"].any()
        assert torch.isfinite(details["basis"]).all()
        for order in range(4):
            broken = details["breakdown"][order]
            if broken.any():
                assert torch.count_nonzero(details["basis"][order, :, broken]) == 0


def test_beta_and_gate_initialization():
    expected_beta = torch.full((4,), 0.5)
    expected_gate = 1.0 / (1.0 + math.exp(2.0))
    for variant in VARIANTS[1:]:
        model = Model(_cfg(variant), _data_info())
        model.eval()
        with torch.no_grad():
            _, _, _, _, info = model(torch.randn(6, 11), _path_graph(), return_details=True)
        for modality in ("text", "visual"):
            details = info["details"][modality]
            assert torch.allclose(details["beta"], expected_beta, atol=1.0e-7, rtol=0.0)
            assert torch.allclose(
                details["gate"], torch.tensor(expected_gate), atol=1.0e-7, rtol=0.0
            )
    legacy = Model(_cfg("A0_legacy_lg"), _data_info())
    assert math.isclose(float(torch.sigmoid(legacy.legacy_gamma)), expected_gate, abs_tol=1e-7)


def test_self_loops_are_removed_and_never_added():
    x = torch.randn(3, 11)
    only_loops = torch.tensor([[0, 1, 2], [0, 1, 2]])
    for variant in VARIANTS:
        model = Model(_cfg(variant), _data_info()).eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, only_loops, return_details=True)
        assert info["input_self_loops_removed"] == 3
        assert not info["active_nodes"].any()
        for modality in ("text", "visual"):
            details = info["details"][modality]
            assert torch.count_nonzero(details["basis"]) == 0
            assert torch.equal(details["output"], details["prior"])


def test_validation_only_macro_f1_label_set_does_not_index_test_labels():
    data = SimpleNamespace(
        y=torch.tensor([0, 1, 2, 3]),
        num_classes=4,
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([1, 2]),
        test_idx=torch.tensor([3]),
    )
    assert _resolve_nc_eval_labels(data, include_test=False) == [0, 1, 2]
    assert _resolve_nc_eval_labels(data, include_test=True) == [0, 1, 2, 3]
