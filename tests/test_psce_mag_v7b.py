from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.semantic_candidates import _faiss_knn, load_semantic_candidate_cache, mean_center_l2_normalize
from src.data.semantic_candidates_v7b import (
    build_semantic_candidate_cache_v7b,
    degree_preserving_double_edge_swap,
    load_semantic_candidate_cache_v7b,
    write_random_candidate_cache_v7b,
)
from src.models.psce_mag_v7a import Model as V7AModel, chunked_weighted_spmm
from src.models.psce_mag_v7b import Model as V7BModel, VARIANTS


@pytest.fixture()
def graph_cache(tmp_path):
    rng = np.random.default_rng(71)
    n = 32
    text = rng.normal(size=(n, 12)).astype(np.float32)
    visual = rng.normal(size=(n, 9)).astype(np.float32)
    nodes = torch.arange(n)
    phys = torch.stack(
        [torch.cat([nodes, nodes]), torch.cat([(nodes + 1) % n, (nodes - 1) % n])]
    ).long()
    path, metadata = build_semantic_candidate_cache_v7b(
        dataset="toy-v7b",
        text_features=text,
        visual_features=visual,
        physical_edge_index=phys,
        cache_dir=tmp_path,
    )
    return {
        "path": path,
        "metadata": metadata,
        "text": torch.from_numpy(text),
        "visual": torch.from_numpy(visual),
        "physical": phys,
        "tmp": tmp_path,
        "n": n,
    }


def _config(variant: str, model_name: str = "psce_mag_v7b"):
    cfg = OmegaConf.create(
        {
            "model": {
                "name": model_name,
                "variant": variant,
                "hidden_dim": 256,
                "physical_hops": 4,
                "semantic_hops": 2,
                "num_experts": 4,
                "top_k": 2,
                "expert_bottleneck": 64,
                "expert_feature_scale": 0.1,
                "router_dim": 64,
                "router_hidden_dim": 128,
                "modality_embed_dim": 16,
                "text_knn_k": 8,
                "visual_knn_k": 8,
                "relation_dim": 32,
                "relation_hidden_dim": 64,
                "semantic_edge_weight_init": 0.8,
                "semantic_strength_init": 0.2,
                "dropout": 0.0,
                "balance_weight": 0.01,
                "eps": 1.0e-8,
                "relation_chunk_size": 11,
                "propagation_chunk_size": 13,
            }
        }
    )
    return cfg


def _model(variant, cache, *, seed=17):
    torch.manual_seed(seed)
    model = V7BModel(
        _config(variant),
        {"input_dim": 21, "text_dim": 12, "visual_dim": 9, "num_nodes": cache["n"]},
    )
    model.load_semantic_cache(
        str(cache["path"]),
        expected_dataset="toy-v7b",
        expected_feature_fingerprint=cache["metadata"]["feature_fingerprint"],
        expected_candidate_fingerprint=cache["metadata"]["candidate_fingerprint"],
        expected_v7b_fingerprint=cache["metadata"]["v7b_fingerprint"],
    )
    return model


def _features(cache):
    return torch.cat((cache["text"], cache["visual"]), dim=1)


def test_candidate_cache_provenance_counts_and_fingerprints(graph_cache):
    loaded = load_semantic_candidate_cache_v7b(
        graph_cache["path"],
        expected_dataset="toy-v7b",
        expected_feature_fingerprint=graph_cache["metadata"]["feature_fingerprint"],
        expected_candidate_fingerprint=graph_cache["metadata"]["candidate_fingerprint"],
        expected_v7b_fingerprint=graph_cache["metadata"]["v7b_fingerprint"],
    )
    flags = loaded["source_flags"]
    assert flags.shape == (loaded["unique_edge_index"].size(1), 2)
    assert flags.dtype == torch.uint8
    assert set(flags.unique().tolist()) <= {0, 1, 2}
    assert bool((flags.sum(1) > 0).all())
    assert torch.equal(loaded["edge_index"][:, : flags.size(0)], loaded["unique_edge_index"])
    assert torch.equal(loaded["edge_index"][:, flags.size(0) :], loaded["unique_edge_index"].flip(0))
    assert graph_cache["metadata"]["cache_contains_labels"] is False
    assert graph_cache["metadata"]["candidate_phase_label_access"] is False


def test_source_counts_are_recomputed_from_raw_directed_queries(graph_cache):
    cache = load_semantic_candidate_cache_v7b(graph_cache["path"])
    flags = cache["source_flags"].numpy()
    n = graph_cache["n"]
    expected = np.zeros_like(flags)
    edge_codes = cache["unique_edge_index"][0].numpy() * n + cache["unique_edge_index"][1].numpy()
    for modality, column in (("text", 0), ("visual", 1)):
        normalized, _ = mean_center_l2_normalize(graph_cache[modality].numpy())
        ids, _, _ = _faiss_knn(normalized, k=8, seed=2026, thread_count=8)
        raw = np.sort(
            np.stack([np.repeat(np.arange(n), 8), ids.reshape(-1)], axis=1), axis=1
        )
        codes, counts = np.unique(raw[:, 0] * n + raw[:, 1], return_counts=True)
        expected[np.searchsorted(edge_codes, codes), column] = np.minimum(counts, 2)
    assert np.array_equal(flags, expected)


def test_v7b_cache_remains_readable_by_v7a_and_matches_legacy_fingerprint(graph_cache):
    old_contract = load_semantic_candidate_cache(
        graph_cache["path"], expected_dataset="toy-v7b"
    )
    new = load_semantic_candidate_cache_v7b(graph_cache["path"])
    assert graph_cache["metadata"]["candidate_fingerprint"] == graph_cache["metadata"]["v7a_candidate_fingerprint"]
    assert torch.equal(old_contract["unique_edge_index"], new["unique_edge_index"])
    assert torch.equal(old_contract["degree"], new["degree"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_cpu_forward_backward_finite(variant, graph_cache):
    model = _model(variant, graph_cache)
    model.train()
    z, _, _, aux, info = model(_features(graph_cache), graph_cache["physical"])
    loss = z.square().mean() + aux
    loss.backward()
    assert z.shape == (graph_cache["n"], 256)
    assert torch.isfinite(z).all() and torch.isfinite(aux)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert info["variant"] == variant


def test_b0_matches_frozen_v7a_with_identical_weights(graph_cache):
    info = {"input_dim": 21, "text_dim": 12, "visual_dim": 9, "num_nodes": graph_cache["n"]}
    torch.manual_seed(113)
    v7a = V7AModel(_config("a2_conditional_dual", "psce_mag_v7a"), info)
    torch.manual_seed(113)
    b0 = V7BModel(_config("b0_v7a_original"), info)
    a_state, b_state = v7a.state_dict(), b0.state_dict()
    assert a_state.keys() == b_state.keys()
    assert all(torch.equal(a_state[key], b_state[key]) for key in a_state)
    v7a.load_semantic_cache(str(graph_cache["path"]), expected_dataset="toy-v7b")
    b0.load_semantic_cache(str(graph_cache["path"]), expected_dataset="toy-v7b")
    v7a.eval(); b0.eval()
    with torch.no_grad():
        a = v7a(_features(graph_cache), graph_cache["physical"], return_details=True)
        b = b0(_features(graph_cache), graph_cache["physical"], return_details=True)
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[3], b[3], atol=0, rtol=0)
    assert a[4]["details"]["edge_weight"] == b[4]["details"]["edge_weight"]


@pytest.mark.parametrize("variant", ("b1_fixed_knn", "b2_random_fixed"))
def test_fixed_variants_use_exact_constant_weight_and_no_scorer_grad(variant, graph_cache):
    model = _model(variant, graph_cache)
    model.eval()
    with torch.no_grad():
        _, _, _, _, info = model(_features(graph_cache), graph_cache["physical"], return_details=True)
    assert info["details"]["edge_weight"]["mean"] == pytest.approx(0.8, abs=1e-7)
    assert all(not p.requires_grad for p in model.relation_scorer.parameters())


def test_b2_random_graph_exactly_preserves_degrees_and_changes_topology(graph_cache):
    base = load_semantic_candidate_cache_v7b(graph_cache["path"])
    first, summary_a = degree_preserving_double_edge_swap(
        base["unique_edge_index"], graph_cache["n"], seed=2026, swaps_per_edge=1
    )
    second, summary_b = degree_preserving_double_edge_swap(
        base["unique_edge_index"], graph_cache["n"], seed=2027, swaps_per_edge=1
    )
    assert first.size(1) == second.size(1) == base["unique_edge_index"].size(1)
    assert torch.equal(torch.bincount(first.reshape(-1), minlength=graph_cache["n"]), base["degree"])
    assert torch.equal(torch.bincount(second.reshape(-1), minlength=graph_cache["n"]), base["degree"])
    assert torch.unique(first[0] * graph_cache["n"] + first[1]).numel() == first.size(1)
    assert not torch.equal(first, second)
    assert summary_a["degree_sequence_exact"] and summary_b["degree_sequence_exact"]


def test_random_cache_loader_and_seed_mapping(graph_cache):
    out = graph_cache["tmp"] / "random.pt"
    rec = write_random_candidate_cache_v7b(graph_cache["path"], out, seed=2026, swaps_per_edge=1)
    loaded = load_semantic_candidate_cache_v7b(out, expected_candidate_fingerprint=rec["candidate_fingerprint"])
    assert loaded["metadata"]["random_graph"]["seed"] == 2026
    assert not bool(loaded["source_flags"].any())


def test_relative_weights_are_symmetric_bounded_shift_invariant_and_differentiable(graph_cache):
    model = _model("b4_relative_modality", graph_cache)
    scores = torch.linspace(-1.3, 1.7, model.semantic_unique_edge_index.size(1), requires_grad=True)
    weights, directed = model._relative_edge_weight(scores)
    shifted, _ = model._relative_edge_weight(scores + 4.25)
    torch.testing.assert_close(weights, shifted, atol=1e-6, rtol=1e-6)
    assert float(weights.min()) > 0.2 and float(weights.max()) < 1.4
    assert torch.isfinite(weights).all() and torch.isfinite(directed).all()
    e = weights.numel()
    assert torch.equal(directed[:e], directed[e:])
    weights.square().mean().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_b3_b4_scorer_shape_and_initialization_match(graph_cache):
    shared = _model("b3_relative_shared", graph_cache, seed=123)
    modality = _model("b4_relative_modality", graph_cache, seed=123)
    assert shared.relation_scorer[0].weight.shape == (64, 130)
    assert shared.relation_scorer[2].weight.shape == (2, 64)
    assert shared.relation_scorer[2].bias is None
    assert all(torch.equal(shared.state_dict()[key], modality.state_dict()[key]) for key in shared.state_dict())
    assert torch.equal(shared.relation_scorer[2].weight[0], shared.relation_scorer[2].weight[1])


def test_b0_b1_b2_common_parameters_and_classifier_initialization_match(graph_cache):
    variants = ["b0_v7a_original", "b1_fixed_knn", "b2_random_fixed"]
    models = [_model(variant, graph_cache, seed=66) for variant in variants]
    reference = models[0].state_dict()
    common = [key for key in reference if not key.startswith("semantic_fingerprint_state")]
    for model in models[1:]:
        state = model.state_dict()
        for key in common:
            assert torch.equal(reference[key], state[key]), key
    classifiers = []
    for variant in variants:
        model = _model(variant, graph_cache, seed=66)
        classifiers.append(torch.nn.Linear(model.out_dim, 7))
    assert all(torch.equal(classifiers[0].weight, x.weight) and torch.equal(classifiers[0].bias, x.bias) for x in classifiers[1:])


def test_b3_uses_same_semantic_weights_and_b4_can_diverge(graph_cache):
    b3 = _model("b3_relative_shared", graph_cache)
    b4 = _model("b4_relative_modality", graph_cache)
    b3.eval(); b4.eval()
    with torch.no_grad():
        out3 = b3(_features(graph_cache), graph_cache["physical"], return_details=True)
        out4 = b4(_features(graph_cache), graph_cache["physical"], return_details=True)
        assert out3[4]["details"]["edge_weights_by_modality"]["text"]["mean"] == pytest.approx(
            out3[4]["details"]["edge_weights_by_modality"]["visual"]["mean"], abs=1e-9
        )
        b4.relation_scorer[2].weight[1].add_(0.03)
        out_diverged = b4(_features(graph_cache), graph_cache["physical"], return_details=True)
    assert out4[4]["details"]["edge_weights_by_modality"]["text_visual_mean_absolute_difference"] < 1e-7
    assert out_diverged[4]["details"]["edge_weights_by_modality"]["text_visual_mean_absolute_difference"] > 0
    assert not torch.equal(out4[0], out_diverged[0])


def test_b4_task_loss_reaches_both_relation_heads_projections_and_experts(graph_cache):
    model = _model("b4_relative_modality", graph_cache)
    z, _, _, aux, _ = model(_features(graph_cache), graph_cache["physical"])
    (z.square().mean() + aux).backward()
    diagnostics = model.capture_gradient_diagnostics()
    for key in (
        "relation_scorer.output_text",
        "relation_scorer.output_visual",
        "relation_projections.text.weight",
        "relation_projections.visual.weight",
        "experts.0.linear1.weight",
    ):
        assert diagnostics[key]["gradient_finite"], key
        assert diagnostics[key]["gradient_nonzero"], key


def test_shared_experts_receive_physical_and_semantic_path_gradients(graph_cache):
    model = _model("b4_relative_modality", graph_cache).eval()
    expert_outputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(4)}
    handles = []
    for expert_id, expert in enumerate(model.experts):
        def capture(_module, _inputs, output, expert_id=expert_id):
            output.retain_grad()
            expert_outputs[expert_id].append(output)
        handles.append(expert.register_forward_hook(capture))
    classifier = torch.nn.Linear(256, 5)
    z, _, _, aux, _ = model(_features(graph_cache), graph_cache["physical"])
    labels = torch.arange(graph_cache["n"]) % 5
    (torch.nn.functional.cross_entropy(classifier(z), labels) + aux).backward()
    for handle in handles:
        handle.remove()
    physical_gradient = 0.0
    semantic_gradient = 0.0
    for calls in expert_outputs.values():
        assert len(calls) == 4  # text physical, text semantic, visual physical, visual semantic
        physical_gradient += sum(float(call.grad.norm()) for call in (calls[0], calls[2]) if call.grad is not None)
        semantic_gradient += sum(float(call.grad.norm()) for call in (calls[1], calls[3]) if call.grad is not None)
    assert physical_gradient > 0
    assert semantic_gradient > 0


def test_chunked_sparse_propagation_matches_dense_reference_and_gradients():
    edge = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    weight = torch.tensor([0.2, 0.2, 0.3, 0.3, 0.5, 0.5], requires_grad=True)
    x = torch.randn(3, 5, requires_grad=True)
    actual = chunked_weighted_spmm(x, edge, weight, chunk_size=2)
    matrix = torch.zeros((3, 3))
    matrix.index_put_((edge[1], edge[0]), weight, accumulate=True)
    expected = matrix @ x
    torch.testing.assert_close(actual, expected)
    grad_a = torch.autograd.grad(actual.square().sum(), (x, weight), retain_graph=True)
    grad_e = torch.autograd.grad(expected.square().sum(), (x, weight))
    for left, right in zip(grad_a, grad_e):
        torch.testing.assert_close(left, right)


def test_raw_physical_path_matches_v7a_for_b0(graph_cache):
    v7a = V7AModel(_config("a2_conditional_dual", "psce_mag_v7a"), {"input_dim": 21, "text_dim": 12, "visual_dim": 9, "num_nodes": graph_cache["n"]})
    b0 = V7BModel(_config("b0_v7a_original"), {"input_dim": 21, "text_dim": 12, "visual_dim": 9, "num_nodes": graph_cache["n"]})
    b0.load_state_dict(v7a.state_dict())
    v7a.load_semantic_cache(str(graph_cache["path"]))
    b0.load_semantic_cache(str(graph_cache["path"]))
    v7a.set_semantic_enabled(False); b0.set_semantic_enabled(False)
    v7a.eval(); b0.eval()
    with torch.no_grad():
        a = v7a(_features(graph_cache), graph_cache["physical"])
        b = b0(_features(graph_cache), graph_cache["physical"])
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[3], b[3], atol=0, rtol=0)


def test_lp_global_id_cache_path_is_explicitly_unsupported(graph_cache):
    model = _model("b4_relative_modality", graph_cache)
    assert model.supports_link_prediction is False
    with pytest.raises(RuntimeError, match="global-ID semantic cache"):
        model(_features(graph_cache)[:-1], graph_cache["physical"])


def test_relative_output_heads_have_no_global_bias(graph_cache):
    model = _model("b4_relative_modality", graph_cache)
    assert model.relation_scorer[-1].bias is None
    assert model.relation_scorer[0].in_features == 130
