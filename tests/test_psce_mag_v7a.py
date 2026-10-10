from __future__ import annotations

import inspect
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.semantic_candidates import (
    build_semantic_candidate_cache,
    load_semantic_candidate_cache,
)
from src.data.types import MAGData
from src.models.mvcge_mag_v4a import Model as V4AModel
from src.models.psce_mag_v7a import Model, chunked_weighted_spmm
from src.tasks import lp as lp_task
from src.tasks.nc import _resolve_nc_eval_labels


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("a0_raw", "a1_static_dual", "a2_conditional_dual")


def _config(variant: str):
    model_cfg = OmegaConf.load(ROOT / "configs/model/psce_mag_v7a.yaml")
    model_cfg.variant = variant
    return OmegaConf.create({"model": model_cfg})


def _physical_edges(num_nodes: int = 18) -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    for node in range(num_nodes - 3):
        pairs.extend(((node, node + 1), (node + 1, node)))
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


@pytest.fixture()
def toy_graph(tmp_path):
    rng = np.random.default_rng(184)
    num_nodes = 18
    text = rng.normal(size=(num_nodes, 11)).astype(np.float32)
    visual = rng.normal(size=(num_nodes, 7)).astype(np.float32)
    edge_index = _physical_edges(num_nodes)
    cache_path, metadata = build_semantic_candidate_cache(
        dataset="toy-mag",
        text_features=text,
        visual_features=visual,
        physical_edge_index=edge_index,
        cache_dir=tmp_path / "cache",
        thread_count=1,
    )
    x = torch.from_numpy(np.concatenate([text, visual], axis=1))
    data_info = {
        "input_dim": 18,
        "num_nodes": num_nodes,
        "num_classes": 3,
        "text_dim": 11,
        "visual_dim": 7,
    }
    return SimpleNamespace(
        x=x,
        edge_index=edge_index,
        text=text,
        visual=visual,
        cache_path=cache_path,
        metadata=metadata,
        data_info=data_info,
    )


def _new_model(variant: str, toy_graph, seed: int = 11) -> Model:
    torch.manual_seed(seed)
    model = Model(_config(variant), toy_graph.data_info)
    if variant != "a0_raw":
        model.load_semantic_cache(
            str(toy_graph.cache_path),
            expected_dataset="toy-mag",
            expected_feature_fingerprint=toy_graph.metadata["feature_fingerprint"],
            expected_candidate_fingerprint=toy_graph.metadata["candidate_fingerprint"],
        )
    return model


def test_all_variants_forward_backward_and_lightweight_info(toy_graph):
    for variant in VARIANTS:
        model = _new_model(variant, toy_graph).train()
        z, aux1, aux2, aux_loss, info = model(toy_graph.x, toy_graph.edge_index)
        assert z.shape == (toy_graph.x.size(0), 256)
        assert aux1 is None and aux2 is None
        assert aux_loss.ndim == 0 and torch.isfinite(aux_loss)
        assert torch.isfinite(z).all()
        assert "details" not in info
        (z.square().mean() + aux_loss).backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if variant != "a0_raw":
            assert model.relation_scorer[-1].weight.grad is not None


def test_modality_slicing_is_text_then_visual(toy_graph):
    model = _new_model("a0_raw", toy_graph)
    parts = model._split_modalities(toy_graph.x)
    assert torch.equal(parts["text"], toy_graph.x[:, :11])
    assert torch.equal(parts["visual"], toy_graph.x[:, 11:])


def test_a0_matches_v4a_r0_raw_common_checkpoint_path(toy_graph):
    v4_cfg = OmegaConf.load(ROOT / "configs/model/mvcge_mag_v4a.yaml")
    v4_cfg.variant = "R0_raw"
    reference = V4AModel(OmegaConf.create({"model": v4_cfg}), toy_graph.data_info)
    candidate = _new_model("a0_raw", toy_graph)
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    common = {
        key: value
        for key, value in reference_state.items()
        if key in candidate_state and value.shape == candidate_state[key].shape
    }
    assert common
    missing, unexpected = candidate.load_state_dict(common, strict=False)
    assert not unexpected
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        ref = reference(toy_graph.x, toy_graph.edge_index)
        got = candidate(toy_graph.x, toy_graph.edge_index)
    torch.testing.assert_close(got[0], ref[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(got[3], ref[3], atol=2e-6, rtol=2e-5)
    assert "semantic_fingerprint_state" in missing


def test_raw_and_classifier_initialization_parity(toy_graph):
    def build_pair(model_cls, cfg):
        torch.manual_seed(1234)
        model = model_cls(cfg, toy_graph.data_info)
        head = torch.nn.Linear(model.out_dim, 3)
        return model, head

    v4_cfg = OmegaConf.load(ROOT / "configs/model/mvcge_mag_v4a.yaml")
    v4_cfg.variant = "R0_raw"
    old, old_head = build_pair(
        V4AModel, OmegaConf.create({"model": v4_cfg})
    )
    new, new_head = build_pair(Model, _config("a0_raw"))
    old_state, new_state = old.state_dict(), new.state_dict()
    common = set(old_state) & set(new_state)
    assert common
    assert all(torch.equal(old_state[key], new_state[key]) for key in common)
    assert all(torch.equal(old_head.state_dict()[key], new_head.state_dict()[key]) for key in old_head.state_dict())


def test_a1_a2_semantic_parameter_count_and_initialization_match(toy_graph):
    a1 = _new_model("a1_static_dual", toy_graph, seed=77)
    a2 = _new_model("a2_conditional_dual", toy_graph, seed=77)
    assert sum(p.numel() for p in a1.parameters()) == sum(p.numel() for p in a2.parameters())
    a1_state, a2_state = a1.state_dict(), a2.state_dict()
    semantic_keys = [key for key in a1_state if key.startswith(("relation_", "semantic_"))]
    assert semantic_keys
    assert all(torch.equal(a1_state[key], a2_state[key]) for key in semantic_keys)


def test_candidate_cache_has_no_self_edges_and_is_symmetric(toy_graph):
    cache = load_semantic_candidate_cache(toy_graph.cache_path)
    edge = cache["edge_index"]
    unique = cache["unique_edge_index"]
    assert not torch.any(edge[0] == edge[1])
    assert torch.all(unique[0] < unique[1])
    assert torch.equal(edge[:, unique.size(1):], unique.flip(0))
    assert torch.equal(edge[:, :unique.size(1)], unique)


def test_candidate_union_is_deduplicated_and_degree_is_fixed(toy_graph):
    cache = load_semantic_candidate_cache(toy_graph.cache_path)
    edges = cache["edge_index"]
    degree = cache["degree"]
    assert degree.sum().item() == edges.size(1)
    assert torch.equal(torch.bincount(edges[0], minlength=degree.numel()), degree)
    codes = cache["unique_edge_index"][0] * toy_graph.x.size(0) + cache["unique_edge_index"][1]
    assert torch.all(codes[1:] > codes[:-1])


def test_candidate_build_is_deterministic_and_fingerprint_bound(toy_graph, tmp_path):
    second_path, second_meta = build_semantic_candidate_cache(
        dataset="toy-mag",
        text_features=toy_graph.text,
        visual_features=toy_graph.visual,
        physical_edge_index=toy_graph.edge_index,
        cache_dir=tmp_path / "second",
        thread_count=1,
    )
    assert second_meta["candidate_fingerprint"] == toy_graph.metadata["candidate_fingerprint"]
    assert torch.equal(
        load_semantic_candidate_cache(second_path)["edge_index"],
        load_semantic_candidate_cache(toy_graph.cache_path)["edge_index"],
    )
    with pytest.raises(ValueError, match="feature fingerprint"):
        load_semantic_candidate_cache(
            toy_graph.cache_path,
            expected_feature_fingerprint="0" * 64,
        )


def test_candidate_builder_interface_has_no_label_or_split_argument():
    names = set(inspect.signature(build_semantic_candidate_cache).parameters)
    assert not names.intersection({"labels", "y", "train_idx", "val_idx", "test_idx", "split"})


def test_candidate_metadata_records_retrieval_and_no_label_fields(toy_graph):
    metadata = toy_graph.metadata
    assert metadata["index_provenance"]["faiss_version"]
    assert metadata["index_provenance"]["text"]["k"] == 8
    assert metadata["index_provenance"]["visual"]["k"] == 8
    assert metadata["construction_seconds"] >= 0
    assert "label" not in str(metadata).lower()
    assert "split" not in str(metadata).lower()


def test_relation_score_and_both_modality_projections_receive_nc_gradient(toy_graph):
    model = _new_model("a2_conditional_dual", toy_graph).train()
    classifier = torch.nn.Linear(256, 3)
    z, _, _, aux_loss, _ = model(toy_graph.x, toy_graph.edge_index)
    labels = torch.arange(z.size(0)) % 3
    (torch.nn.functional.cross_entropy(classifier(z), labels) + aux_loss).backward()
    for parameter in (
        model.relation_projections["text"].weight,
        model.relation_projections["visual"].weight,
        model.relation_scorer[0].weight,
        model.relation_scorer[2].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm().item() > 0


def test_semantic_edge_weights_remain_differentiable(toy_graph):
    model = _new_model("a1_static_dual", toy_graph).train()
    z, _, _, _, _ = model(toy_graph.x, toy_graph.edge_index)
    z.square().mean().backward()
    assert model.relation_scorer[-1].bias.grad is not None
    assert model.relation_scorer[-1].bias.grad.abs().item() > 0


def test_fixed_degree_normalization_does_not_cancel_weight_scale(toy_graph):
    model = _new_model("a1_static_dual", toy_graph).eval()
    h0 = torch.randn_like(toy_graph.x[:, :256])
    edges = model.semantic_unique_edge_index
    low = model._semantic_propagate(h0, torch.full((edges.size(1),), -4.0))
    high = model._semantic_propagate(h0, torch.full((edges.size(1),), 4.0))
    assert high.norm() > low.norm() * 5


def test_relation_features_give_symmetric_undirected_scores(toy_graph):
    model = _new_model("a1_static_dual", toy_graph).eval()
    priors = {
        modality: model.projectors[modality](chunk)
        for modality, chunk in (
            ("text", toy_graph.x[:, :11]),
            ("visual", toy_graph.x[:, 11:]),
        )
    }
    projected = {
        modality: model.relation_projections[modality](prior)
        for modality, prior in priors.items()
    }
    unique = model.semantic_unique_edge_index
    forward = model._score_semantic_edges(projected, unique)
    reverse = model._score_semantic_edges(projected, unique.flip(0))
    torch.testing.assert_close(forward, reverse, atol=0, rtol=0)


def test_semantic_coefficients_have_four_distinct_unit_responses(toy_graph):
    model = _new_model("a1_static_dual", toy_graph)
    coefficients = model._effective_alpha(model.semantic_coefficients_raw, model.eps)
    expected = torch.tensor(
        [[1, 0], [0, 1], [2**-0.5, 2**-0.5], [2**-0.5, -(2**-0.5)]],
        dtype=coefficients.dtype,
    )
    torch.testing.assert_close(coefficients, expected, atol=2e-6, rtol=2e-6)
    assert torch.allclose(coefficients.norm(dim=-1), torch.ones(4), atol=2e-6)


def test_semantic_first_and_mixed_second_hop_match_sparse_formula(toy_graph):
    model = _new_model("a1_static_dual", toy_graph).eval()
    state = torch.randn(toy_graph.x.size(0), 5)
    scores = torch.linspace(-1, 1, model.semantic_unique_edge_index.size(1))
    actual_s1 = model._semantic_propagate(state, scores)
    cache = load_semantic_candidate_cache(toy_graph.cache_path)
    u, v = cache["unique_edge_index"]
    weight = scores.sigmoid() / (
        cache["degree"][u].float() * cache["degree"][v].float()
    ).clamp_min(1).sqrt()
    adjacency = torch.zeros((state.size(0), state.size(0)))
    adjacency[v, u] = weight
    adjacency[u, v] = weight
    expected_s1 = adjacency @ state
    torch.testing.assert_close(actual_s1, expected_s1, atol=1e-6, rtol=1e-6)
    src, dst, norm, _, _ = model._normalized_operator(
        toy_graph.edge_index, state.size(0), state.dtype
    )
    actual_s2 = model._propagate(actual_s1, src, dst, norm)
    physical = torch.zeros((state.size(0), state.size(0)))
    physical[dst, src] = norm
    torch.testing.assert_close(actual_s2, physical @ expected_s1, atol=1e-6, rtol=1e-6)


def test_shared_expert_bank_processes_physical_and_semantic_profiles(toy_graph):
    model = _new_model("a1_static_dual", toy_graph).train()
    calls = {idx: 0 for idx in range(4)}
    handles = []
    for idx, expert in enumerate(model.experts):
        handles.append(expert.register_forward_hook(lambda _m, _i, _o, idx=idx: calls.__setitem__(idx, calls[idx] + 1)))
    z, _, _, _, _ = model(toy_graph.x, toy_graph.edge_index)
    assert calls == {idx: 4 for idx in range(4)}
    z.square().mean().backward()
    for handle in handles:
        handle.remove()
    assert not hasattr(model, "semantic_experts")
    assert any(
        expert.linear1.weight.grad is not None and expert.linear1.weight.grad.norm() > 0
        for expert in model.experts
    )


def test_static_and_node_conditional_semantic_router_inputs(toy_graph):
    n = toy_graph.x.size(0)
    captured: dict[str, torch.Tensor] = {}
    for variant in ("a1_static_dual", "a2_conditional_dual"):
        model = _new_model(variant, toy_graph).eval()
        seen = []
        handle = model.semantic_context_encoder[0].register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].detach().clone())
        )
        prior = torch.randn(n, 256)
        physical = torch.randn(n, 256)
        semantic = torch.randn(n, 256)
        raw = torch.randn(n, 256)
        active = torch.ones(n, dtype=torch.bool)
        route = model._semantic_route(0, prior, physical, semantic, raw, active)
        handle.remove()
        captured[variant] = seen[0]
        if variant == "a1_static_dual":
            assert captured[variant].shape == (1, 3 * 256 + model.modality_embed_dim + 1)
            assert route["encoded"].size(0) == 1
            assert torch.equal(route["strength"], route["strength"][:1].expand(n))
        else:
            assert captured[variant].shape == (n, 3 * 256 + model.modality_embed_dim + 1)
            assert route["encoded"].size(0) == n


def test_disabling_semantic_path_recovers_a0_with_shared_weights(toy_graph):
    a0 = _new_model("a0_raw", toy_graph, seed=9).eval()
    a2 = _new_model("a2_conditional_dual", toy_graph, seed=9).eval()
    common_weights = {
        key: value
        for key, value in a0.state_dict().items()
        if key not in {"semantic_fingerprint_state", "semantic_feature_fingerprint_state"}
    }
    a2.load_state_dict(common_weights, strict=False)
    a2.set_semantic_enabled(False)
    with torch.no_grad():
        expected = a0(toy_graph.x, toy_graph.edge_index)
        actual = a2(toy_graph.x, toy_graph.edge_index)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)


def test_semantic_active_nodes_survive_physical_isolates_and_zero_state_is_finite(toy_graph):
    model = _new_model("a2_conditional_dual", toy_graph).eval()
    active1, active2 = model._semantic_active_masks(
        *model._normalized_operator(toy_graph.edge_index, toy_graph.x.size(0), torch.float32)[:2],
        torch.zeros_like(toy_graph.x[:, :256]),
        model._normalized_operator(toy_graph.edge_index, toy_graph.x.size(0), torch.float32)[3],
    )
    physical_isolates = ~model._normalized_operator(
        toy_graph.edge_index, toy_graph.x.size(0), torch.float32
    )[3]
    assert torch.all(active1[physical_isolates])
    assert not torch.any(active2[physical_isolates])
    zero = torch.zeros((toy_graph.x.size(0), 256), requires_grad=True)
    raw = torch.ones_like(zero)
    calibrated, scale, _, _ = model._calibrate_semantic(zero, raw, active1)
    assert torch.isfinite(calibrated).all() and torch.isfinite(scale)
    assert calibrated.count_nonzero().item() == 0
    calibrated.sum().backward()
    assert zero.grad is not None and torch.isfinite(zero.grad).all()
    output = model(toy_graph.x, toy_graph.edge_index)[0]
    assert torch.isfinite(output).all()


def test_training_and_inference_keep_same_candidate_fingerprint(toy_graph):
    model = _new_model("a2_conditional_dual", toy_graph)
    expected = toy_graph.metadata["candidate_fingerprint"]
    train_info = model(toy_graph.x, toy_graph.edge_index)[4]
    model.eval()
    with torch.no_grad():
        eval_info = model(toy_graph.x, toy_graph.edge_index)[4]
    assert train_info["semantic_candidate_fingerprint"] == expected
    assert eval_info["semantic_candidate_fingerprint"] == expected
    assert model.state_dict()["semantic_fingerprint_state"].numel() == 32
    assert model.state_dict()["semantic_feature_fingerprint_state"].numel() == 32


def test_checkpoint_restore_rejects_different_candidate_graph(toy_graph, tmp_path):
    source = _new_model("a2_conditional_dual", toy_graph)
    state = source.state_dict()
    other_text = toy_graph.text.copy()
    other_text[0, 0] += 5.0
    path, _ = build_semantic_candidate_cache(
        dataset="toy-mag",
        text_features=other_text,
        visual_features=toy_graph.visual,
        physical_edge_index=toy_graph.edge_index,
        cache_dir=tmp_path / "other-cache",
        thread_count=1,
    )
    target = _new_model("a2_conditional_dual", toy_graph)
    target.load_semantic_cache(str(path), expected_dataset="toy-mag")
    with pytest.raises(RuntimeError, match="fingerprint"):
        target.load_state_dict(state)


def test_checkpoint_restore_rejects_changed_features_with_same_knn_graph(toy_graph, tmp_path):
    source = _new_model("a2_conditional_dual", toy_graph)
    state = source.state_dict()
    path, metadata = build_semantic_candidate_cache(
        dataset="toy-mag",
        text_features=toy_graph.text * 3.0,
        visual_features=toy_graph.visual * 2.0,
        physical_edge_index=toy_graph.edge_index,
        cache_dir=tmp_path / "rescaled-cache",
        thread_count=1,
    )
    assert metadata["candidate_fingerprint"] == toy_graph.metadata["candidate_fingerprint"]
    assert metadata["feature_fingerprint"] != toy_graph.metadata["feature_fingerprint"]
    target = _new_model("a2_conditional_dual", toy_graph)
    target.load_semantic_cache(str(path), expected_dataset="toy-mag")
    with pytest.raises(RuntimeError, match="input features fingerprint"):
        target.load_state_dict(state)


def test_chunked_weighted_spmm_matches_dense_forward_and_backward():
    torch.manual_seed(6)
    edge = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 2, 0]], dtype=torch.long)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(edge.size(1), dtype=torch.float64, requires_grad=True)
    actual = chunked_weighted_spmm(x, edge, weight, chunk_size=2)
    grad_actual = torch.autograd.grad(actual.square().sum(), (x, weight))
    dense_x = x.detach().clone().requires_grad_()
    dense_weight = weight.detach().clone().requires_grad_()
    adjacency = torch.zeros((3, 3), dtype=torch.float64)
    adjacency.index_put_((edge[1], edge[0]), dense_weight, accumulate=True)
    expected = adjacency @ dense_x
    grad_expected = torch.autograd.grad(expected.square().sum(), (dense_x, dense_weight))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(grad_actual[0], grad_expected[0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(grad_actual[1], grad_expected[1], atol=1e-12, rtol=1e-12)


def test_forward_outputs_and_masks_are_finite_and_correct(toy_graph):
    model = _new_model("a2_conditional_dual", toy_graph).eval()
    with torch.no_grad():
        z, _, _, aux_loss, info = model(toy_graph.x, toy_graph.edge_index)
    physical_active = torch.zeros(toy_graph.x.size(0), dtype=torch.bool)
    physical_active[toy_graph.edge_index.reshape(-1).unique()] = True
    assert torch.equal(info["active_nodes"], physical_active)
    assert torch.isfinite(z).all() and torch.isfinite(aux_loss)
    assert not any("test" in key.lower() for key in info)


def test_validation_only_label_universe_excludes_test_only_classes():
    data = SimpleNamespace(
        y=torch.tensor([0, 1, 1, 2, 3]),
        num_classes=4,
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2, 3]),
        test_idx=torch.tensor([4]),
    )
    assert _resolve_nc_eval_labels(data, include_test=False) == [0, 1, 2]
    assert _resolve_nc_eval_labels(data, include_test=True) == [0, 1, 2, 3]


def test_v7a_lp_path_fails_before_local_id_sampling(monkeypatch, toy_graph):
    class UnsupportedModel:
        requires_global_semantic_candidates = True

    monkeypatch.setattr(lp_task, "build_model", lambda _cfg, _info: UnsupportedModel())
    cfg = OmegaConf.create(
        {
            "model": {"name": "psce_mag_v7a"},
            "task": {"inference_mode": "full"},
        }
    )
    data = MAGData(
        name="toy-mag",
        source="magb",
        task="lp",
        x=toy_graph.x,
        x_t=toy_graph.x[:, :11],
        x_i=toy_graph.x[:, 11:],
        edge_index=toy_graph.edge_index,
        num_nodes=toy_graph.x.size(0),
        num_classes=None,
    )
    with pytest.raises(NotImplementedError, match="full-graph NC only"):
        lp_task._run_single_lp(cfg, data, torch.device("cpu"), logging.getLogger("test"), 0, 42)
