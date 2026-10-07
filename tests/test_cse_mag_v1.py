from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from src.models.cse_mag_v1 import Model


def _cfg(variant: str):
    return OmegaConf.create(
        {
            "model": {
                "name": "cse_mag_v1",
                "variant": variant,
                "hidden_dim": 16,
                "num_layers": 3,
                "dropout": 0.0,
                "expert_bottleneck": 8,
                "router_hidden_dim": 12,
                "modality_embed_dim": 4,
                "global_order": 3,
                "global_t": 1.0,
                "gate_init_bias": -2.0,
                "naive_num_experts": 4,
                "naive_top_k": 2,
                "mvcge_balance_weight": 0.01,
                "eps": 1.0e-8,
            }
        }
    )


def _data_info():
    return {
        "input_dim": 11,
        "text_dim": 6,
        "visual_dim": 5,
        "num_nodes": 6,
        "num_classes": 3,
    }


def _toy_graph():
    # Undirected path 0-1-2-3-4; node 5 is isolated.
    src = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4])
    dst = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3])
    return torch.stack([src, dst], dim=0)


def test_all_screen_variants_forward_and_backward():
    x = torch.randn(6, 11)
    edge_index = _toy_graph()
    for variant in ("independent", "shared_static", "mvcge_style", "v1"):
        torch.manual_seed(42)
        model = Model(_cfg(variant), _data_info())
        model.train()
        z, _, _, aux_loss, info = model(x, edge_index)
        assert z.shape == (6, 16)
        assert torch.isfinite(z).all()
        assert torch.isfinite(aux_loss)
        assert info["variant"] == variant
        loss = z.square().mean() + aux_loss
        loss.backward()
        active_grads = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert active_grads
        assert all(torch.isfinite(grad).all() for grad in active_grads)


def test_common_initialization_is_variant_invariant():
    snapshots = {}
    for variant in ("independent", "shared_static", "mvcge_style", "v1"):
        torch.manual_seed(123)
        model = Model(_cfg(variant), _data_info())
        snapshots[variant] = {
            "projector": model.projectors["text"].linear1.weight.detach().clone(),
            "fusion": model.fusion_skip.weight.detach().clone(),
        }

    reference = snapshots["v1"]
    for variant, state in snapshots.items():
        assert torch.equal(state["projector"], reference["projector"]), variant
        assert torch.equal(state["fusion"], reference["fusion"]), variant


def test_v1_initial_composer_is_balanced_and_prior_first():
    torch.manual_seed(7)
    model = Model(_cfg("v1"), _data_info())
    model.eval()
    x = torch.randn(6, 11)
    with torch.no_grad():
        _, _, _, _, info = model(x, _toy_graph(), return_details=True)

    expected_gate = 1.0 / (1.0 + math.exp(2.0))
    for modality in ("text", "visual"):
        routing = info["details"][modality]["routing"]
        gate = info["details"][modality]["gate"]
        assert torch.allclose(
            routing, torch.full_like(routing, 0.5), atol=1.0e-7, rtol=0.0
        )
        assert torch.allclose(
            gate,
            torch.full_like(gate, expected_gate),
            atol=1.0e-7,
            rtol=0.0,
        )


def test_isolated_node_structural_displacements_are_finite():
    torch.manual_seed(9)
    model = Model(_cfg("v1"), _data_info())
    model.eval()
    x = torch.randn(6, 11)
    with torch.no_grad():
        _, _, _, _, info = model(x, _toy_graph(), return_details=True)
    for modality in ("text", "visual"):
        details = info["details"][modality]
        assert torch.allclose(
            details["local_delta"][5],
            torch.zeros_like(details["local_delta"][5]),
            atol=0.0,
            rtol=0.0,
        )
        assert torch.allclose(
            details["global_delta"][5],
            torch.zeros_like(details["global_delta"][5]),
            atol=0.0,
            rtol=0.0,
        )
        assert torch.allclose(
            details["output"][5],
            details["prior"][5],
            atol=1.0e-7,
            rtol=0.0,
        )
