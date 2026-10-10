from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data import MAGData
from src.models import build_model
from src.models.mace_mag_v6a import ExpertCollaboration, Model
from src.models.mvcge_mag_v22b import Model as V22BModel
from src.tasks.nc import _resolve_nc_eval_labels


VARIANTS = ("a0_static", "a1_node", "a2_cross", "a3_intra")
DATA_INFO = {"input_dim": 12, "text_dim": 5, "visual_dim": 7, "num_classes": 3}


def _cfg(variant: str, model_name: str = "mace_mag_v6a"):
    return OmegaConf.create(
        {
            "model": {
                "name": model_name,
                "variant": variant,
                "hidden_dim": 256,
                "dropout": 0.2,
                "trajectory_order": 4,
                "num_experts": 4,
                "top_k": 2,
                "expert_bottleneck": 64,
                "expert_feature_scale": 0.1,
                "router_dim": 64,
                "router_hidden_dim": 128,
                "modality_embed_dim": 16,
                "attention_dim": 64,
                "attention_heads": 1,
                "structural_strength_init": 0.25,
                "collaboration_strength_init": 0.15,
                "balance_weight": 0.01,
                "eps": 1.0e-8,
                "direct_strength_init": -1.15,
            }
        }
    )


def _graph(device="cpu"):
    x = torch.randn(8, DATA_INFO["input_dim"], device=device)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 6, 6], [1, 0, 2, 1, 4, 3, 6, 5, 6]],
        dtype=torch.long,
        device=device,
    )
    return x, edge_index


def _state_close(left, right, *, atol=0.0, rtol=0.0):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=atol, rtol=rtol)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_cpu_forward_backward_finite_and_nc_interface(variant):
    torch.manual_seed(101)
    model = Model(_cfg(variant), DATA_INFO)
    x, edge_index = _graph()
    z, aux1, aux2, aux_loss, info = model(x, edge_index)
    assert z.shape == (x.size(0), 256)
    assert aux1 is None and aux2 is None
    assert aux_loss.ndim == 0 and aux_loss.requires_grad
    assert model.out_dim == 256
    assert info["variant"] == variant
    head = torch.nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    labels = torch.randint(DATA_INFO["num_classes"], (x.size(0),))
    loss = torch.nn.functional.cross_entropy(head(z), labels) + aux_loss
    loss.backward()
    assert torch.isfinite(z).all()
    assert torch.isfinite(aux_loss)
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_seed_matched_variants_keep_trunk_attention_and_classifier_initialization():
    states = {}
    heads = {}
    models = {}
    for variant in VARIANTS:
        torch.manual_seed(2026)
        model = Model(_cfg(variant), DATA_INFO)
        head = torch.nn.Linear(model.out_dim, DATA_INFO["num_classes"])
        states[variant] = model.state_dict()
        heads[variant] = head.state_dict()
        models[variant] = model
    for variant in VARIANTS[1:]:
        _state_close(states[VARIANTS[0]], states[variant])
        _state_close(heads[VARIANTS[0]], heads[variant])
    _state_close(
        {
            k: v
            for k, v in states["a2_cross"].items()
            if k.startswith("collaboration.")
        },
        {
            k: v
            for k, v in states["a3_intra"].items()
            if k.startswith("collaboration.")
        },
    )
    assert sum(p.numel() for p in models["a2_cross"].collaboration.parameters() if p.requires_grad) > 0
    assert sum(p.numel() for p in models["a3_intra"].collaboration.parameters() if p.requires_grad) > 0
    assert all(not p.requires_grad for p in models["a0_static"].collaboration.parameters())
    assert all(not p.requires_grad for p in models["a1_node"].collaboration.parameters())
    assert sum(p.numel() for p in models["a2_cross"].parameters() if p.requires_grad) == sum(
        p.numel() for p in models["a3_intra"].parameters() if p.requires_grad
    )
    assert sum(p.numel() for p in models["a0_static"].parameters() if p.requires_grad) == sum(
        p.numel() for p in models["a1_node"].parameters() if p.requires_grad
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_top2_has_two_unique_experts_and_normalized_weights(variant):
    torch.manual_seed(7)
    model = Model(_cfg(variant), DATA_INFO).eval()
    x, edge_index = _graph()
    _, _, _, _, info = model(x, edge_index, return_details=True)
    for modality in ("text", "visual"):
        route = info["details"]["routes"][modality]
        assert route["top_indices"].shape == (x.size(0), 2)
        assert (route["top_indices"][:, 0] != route["top_indices"][:, 1]).all()
        torch.testing.assert_close(route["top_weights"].sum(-1), torch.ones(x.size(0)))
        assert (route["route_weights"] > 0).sum(-1).eq(2).all()


def test_a0_static_and_a1_node_routing_contexts():
    torch.manual_seed(9)
    x, edge_index = _graph()
    static_model = Model(_cfg("a0_static"), DATA_INFO).eval()
    node_model = Model(_cfg("a1_node"), DATA_INFO).eval()
    static_info = static_model(x, edge_index, return_details=True)[-1]
    node_info = node_model(x, edge_index, return_details=True)[-1]
    for modality in ("text", "visual"):
        static_route = static_info["details"]["routes"][modality]
        node_route = node_info["details"]["routes"][modality]
        assert torch.equal(
            static_route["selection_logits"],
            static_route["selection_logits"][:1].expand_as(
                static_route["selection_logits"]
            ),
        )
        assert node_route["selection_logits"].std(dim=0).max() > 0


def test_a2_correction_has_real_cross_modal_gradient_dependency():
    torch.manual_seed(13)
    model = Model(_cfg("a2_cross"), DATA_INFO).eval()
    x, edge_index = _graph()
    _, _, _, _, info = model(x, edge_index, return_details=True)
    details = info["details"]
    text_correction = details["collaboration"]["text"]["correction"]
    visual_tokens = details["expert_values"]["visual"]
    grad = torch.autograd.grad(
        text_correction.square().sum(), visual_tokens, retain_graph=True
    )[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    visual_correction = details["collaboration"]["visual"]["correction"]
    text_tokens = details["expert_values"]["text"]
    reverse_grad = torch.autograd.grad(visual_correction.square().sum(), text_tokens)[0]
    assert torch.isfinite(reverse_grad).all()
    assert reverse_grad.abs().sum() > 0


def test_a3_local_corrections_do_not_depend_on_the_other_modality_tokens():
    torch.manual_seed(17)
    model = Model(_cfg("a3_intra"), DATA_INFO).eval()
    x, edge_index = _graph()
    _, _, _, _, info = model(x, edge_index, return_details=True)
    details = info["details"]
    text_corr = details["collaboration"]["text"]["correction"]
    visual_corr = details["collaboration"]["visual"]["correction"]
    assert details["collaboration"]["text"]["source_modality"] == "text"
    assert details["collaboration"]["visual"]["source_modality"] == "visual"
    assert torch.autograd.grad(
        text_corr.square().sum(),
        details["expert_values"]["visual"],
        allow_unused=True,
        retain_graph=True,
    )[0] is None
    assert torch.autograd.grad(
        visual_corr.square().sum(), details["expert_values"]["text"], allow_unused=True
    )[0] is None


def test_explicitly_zero_collaboration_matches_a1_output_path():
    torch.manual_seed(29)
    a1 = Model(_cfg("a1_node"), DATA_INFO).eval()
    torch.manual_seed(29)
    a2 = Model(_cfg("a2_cross"), DATA_INFO).eval()
    x, edge_index = _graph()
    z1, _, _, aux1, _ = a1(x, edge_index)
    z2, _, _, aux2, _ = a2(x, edge_index, collaboration_scale=0.0)
    torch.testing.assert_close(z1, z2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(aux1, aux2, atol=0.0, rtol=0.0)


def test_null_source_is_zero_and_attention_is_finite():
    torch.manual_seed(31)
    module = ExpertCollaboration(256, 64, 4, 2, 0.0).eval()
    target = torch.zeros(5, 4, 256)
    source = torch.zeros(5, 4, 256)
    ids = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1], [0, 1]])
    weights = torch.full((5, 2), 0.5)
    routes = {
        "text": {"top_indices": ids, "top_weights": weights},
        "visual": {"top_indices": ids.flip(-1), "top_weights": weights},
    }
    output = module(
        {"text": target, "visual": source}, routes, torch.ones(5, dtype=torch.bool), variant="a2_cross"
    )
    for modality in ("text", "visual"):
        item = output[modality]
        assert torch.isfinite(item["attention"]).all()
        torch.testing.assert_close(
            item["correction"], torch.zeros_like(item["correction"]), atol=0.0, rtol=0.0
        )
        assert (item["attention"][:, :, -1] > 0).all()
        torch.testing.assert_close(item["attention"].sum(-1), torch.ones(5, 2))


def test_isolated_nodes_have_no_structural_or_collaborative_correction():
    torch.manual_seed(37)
    model = Model(_cfg("a2_cross"), DATA_INFO).eval()
    x = torch.randn(5, DATA_INFO["input_dim"])
    edge_index = torch.tensor([[0, 1, 3, 3], [1, 0, 3, 3]], dtype=torch.long)
    z, _, _, aux_loss, info = model(x, edge_index, return_details=True)
    active = info["details"]["active"]
    assert active.tolist() == [True, True, False, False, False]
    assert torch.isfinite(z).all() and torch.isfinite(aux_loss)
    for modality in ("text", "visual"):
        correction = info["details"]["collaboration"][modality]["correction"]
        assert torch.equal(correction[~active], torch.zeros_like(correction[~active]))
        basis = info["details"]["modalities"][modality]["basis"]
        assert torch.equal(basis[:, ~active], torch.zeros_like(basis[:, ~active]))


def test_empty_graph_keeps_intrinsic_path_and_finite_router_attention():
    torch.manual_seed(41)
    model = Model(_cfg("a2_cross"), DATA_INFO).eval()
    x = torch.randn(4, DATA_INFO["input_dim"])
    edges = torch.empty((2, 0), dtype=torch.long)
    z, _, _, aux_loss, info = model(x, edges, return_details=True)
    assert torch.isfinite(z).all() and torch.isfinite(aux_loss)
    assert aux_loss.requires_grad
    aux_loss.backward()
    assert not info["active_nodes"].any()
    for modality in ("text", "visual"):
        correction = info["details"]["collaboration"][modality]["correction"]
        assert torch.equal(correction, torch.zeros_like(correction))
        attention = info["details"]["collaboration"][modality]["attention"]
        assert torch.isfinite(attention).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_factory_inference_shape_and_checkpoint_roundtrip(variant):
    torch.manual_seed(43)
    cfg = _cfg(variant)
    model = build_model(cfg, DATA_INFO).eval()
    x, edge_index = _graph()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = build_model(cfg, DATA_INFO).eval()
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        expected = model(x, edge_index)[0]
        inferred = restored.inference(x, edge_index, device=torch.device("cpu"))
        assert inferred.shape == (x.size(0), 256)
        torch.testing.assert_close(inferred, expected.detach(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("variant,old_variant", [("a0_static", "D0_static"), ("a1_node", "D1_free_node")])
def test_small_graph_same_weight_regression_against_v22b(variant, old_variant):
    torch.manual_seed(53)
    old_cfg = _cfg(old_variant, model_name="mvcge_mag_v22b")
    old_model = V22BModel(old_cfg, DATA_INFO).eval()
    new_model = Model(_cfg(variant), DATA_INFO).eval()
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    shared = {
        key: value
        for key, value in old_state.items()
        if key in new_state and new_state[key].shape == value.shape
    }
    required_prefixes = (
        "projectors.",
        "alpha_raw",
        "experts.",
        "router_projectors.",
        "router_norms.",
        "modality_embedding.",
        "context_encoder.",
        "selection_head.",
        "fusion_linear1.",
        "fusion_linear2.",
        "fusion_skip.",
        "fusion_norm.",
    )
    assert all(any(key.startswith(prefix) for key in shared) for prefix in required_prefixes)
    new_model.load_state_dict(shared, strict=False)
    with torch.no_grad():
        new_model.structural_strength_raw.copy_(old_model.direct_strength_raw)
    x, edge_index = _graph()
    old_z, _, _, old_aux, old_info = old_model(x, edge_index, return_details=True)
    new_z, _, _, new_aux, new_info = new_model(x, edge_index, return_details=True)
    torch.testing.assert_close(new_z, old_z, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(new_aux, old_aux, atol=2e-6, rtol=2e-6)
    for modality in ("text", "visual"):
        old_route = old_info["details"][modality]
        new_route = new_info["details"]["routes"][modality]
        torch.testing.assert_close(
            new_route["selection_logits"], old_route["selection_logits"], atol=2e-6, rtol=2e-6
        )
        assert torch.equal(new_route["top_indices"], old_route["top_indices"])


def test_nc_label_set_omits_test_split_when_test_evaluation_is_disabled():
    data = MAGData(
        name="toy",
        source="toy",
        task="nc",
        x=torch.zeros(5, 2),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=5,
        y=torch.tensor([0, 1, 0, 2, 2]),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2]),
        test_idx=torch.tensor([3, 4]),
        num_classes=3,
    )
    assert _resolve_nc_eval_labels(data, include_test=False) == [0, 1]
    assert _resolve_nc_eval_labels(data, include_test=True) == [0, 1, 2]
