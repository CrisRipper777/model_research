from __future__ import annotations

import hashlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.semantic_candidates_v7b import degree_preserving_double_edge_swap
from src.models.mag_information_probe_v1 import (
    Model,
    SLOT_NAMES,
    build_physical_operator,
    build_row_mean_operator,
    build_semantic_operators_from_flags,
)


def _model(condition: str, *, dropout: float = 0.0) -> Model:
    return Model(
        OmegaConf.create({"hidden_dim": 8, "dropout": dropout}),
        {"text_dim": 5, "visual_dim": 7},
        treatment=condition,
    )


def _small_graphs():
    physical = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    # (0,1) is in both sources; (1,2) is Text-only; (2,3) is Visual-only.
    unique = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    flags = torch.tensor([[1, 1], [1, 0], [0, 2]], dtype=torch.uint8)
    return physical, unique, flags


def _state_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def test_row_mean_operators_are_sparse_and_isolates_stay_zero():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 2], [1, 0, 2, 1, 2, 2]])
    operator, degree = build_physical_operator(edge_index, 4)
    assert operator.layout == torch.sparse_coo
    assert operator.shape == (4, 4)
    assert torch.equal(degree, torch.tensor([1, 2, 1, 0]))
    dense = operator.to_dense()
    assert torch.equal(dense.diag(), torch.zeros(4))
    assert torch.allclose(dense.sum(1), torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(dense[3], torch.zeros(4))


def test_semantic_views_are_independently_normalized_and_shared_edges_remain_in_both():
    _, unique, flags = _small_graphs()
    p_text, p_visual, deg_text, deg_visual = build_semantic_operators_from_flags(
        unique, flags, 4
    )
    text = p_text.to_dense()
    visual = p_visual.to_dense()
    assert torch.equal(deg_text, torch.tensor([1, 2, 1, 0]))
    assert torch.equal(deg_visual, torch.tensor([1, 1, 1, 1]))
    assert text[0, 1] > 0 and visual[0, 1] > 0
    assert text[1, 2] > 0 and visual[1, 2] == 0
    assert visual[2, 3] > 0 and text[2, 3] == 0
    assert torch.allclose(text.sum(1), torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(visual.sum(1), torch.ones(4))


def test_treatments_share_parameter_shapes_and_initialization():
    hashes = []
    shapes = []
    for condition in ("I0", "I1", "I2", "I3", "I4"):
        torch.manual_seed(42)
        model = _model(condition)
        head = torch.nn.Linear(model.out_dim, 4)
        hashes.append((_state_sha(model), _state_sha(head)))
        shapes.append([(name, tuple(value.shape)) for name, value in model.state_dict().items()])
    assert len(set(hashes)) == 1
    assert all(shape == shapes[0] for shape in shapes[1:])


def test_zeroed_context_slots_produce_no_context_update_and_i0_matches_intrinsic():
    torch.manual_seed(7)
    physical, unique, flags = _small_graphs()
    p_phy, _ = build_physical_operator(physical, 4)
    p_text, p_visual, _, _ = build_semantic_operators_from_flags(unique, flags, 4)
    x_text = torch.randn(4, 5)
    x_visual = torch.randn(4, 7)
    model = _model("I0").eval()
    fused, detail = model(
        x_text, x_visual, p_phy, p_text, p_visual, return_details=True
    )
    assert torch.isfinite(fused).all()
    assert detail["active_slots"] == []
    assert torch.count_nonzero(detail["slots_text"]) == 0
    assert torch.count_nonzero(detail["slots_visual"]) == 0
    assert torch.count_nonzero(detail["context_delta_text"]) == 0
    assert torch.count_nonzero(detail["context_delta_visual"]) == 0
    for processor in model.context_processors.values():
        assert processor.linear1.bias is None
        assert processor.linear2.bias is None


def test_context_mask_changes_only_expected_slot_gradients():
    torch.manual_seed(19)
    physical, unique, flags = _small_graphs()
    p_phy, _ = build_physical_operator(physical, 4)
    p_text, p_visual, _, _ = build_semantic_operators_from_flags(unique, flags, 4)
    x_text = torch.randn(4, 5)
    x_visual = torch.randn(4, 7)
    model = _model("I1").eval()
    classifier = torch.nn.Linear(model.out_dim, 3)
    logits = classifier(model(x_text, x_visual, p_phy, p_text, p_visual))
    logits.square().mean().backward()
    grad = model.context_processors["text"].linear1.weight.grad
    assert grad is not None
    blocks = grad.view(8, len(SLOT_NAMES), 8)
    norms = blocks.norm(dim=(0, 2))
    assert bool((norms[:2] > 0).all())
    assert torch.equal(norms[2:], torch.zeros_like(norms[2:]))


def test_degree_preserving_null_keeps_exact_degree_and_simple_graph():
    edges = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 3, 2, 4, 3, 5, 4, 5]],
        dtype=torch.long,
    )
    random_edges, audit = degree_preserving_double_edge_swap(
        edges, 6, seed=122, swaps_per_edge=4
    )
    before = torch.bincount(edges.flatten(), minlength=6)
    after = torch.bincount(random_edges.flatten(), minlength=6)
    assert audit["degree_sequence_exact"] is True
    assert audit["edge_count_before"] == audit["edge_count_after"] == edges.size(1)
    assert torch.equal(before, after)
    assert bool((random_edges[0] < random_edges[1]).all())
    assert torch.unique(random_edges[0] * 6 + random_edges[1]).numel() == edges.size(1)


def test_model_backward_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(23)
    physical, unique, flags = _small_graphs()
    p_phy, _ = build_physical_operator(physical, 4)
    p_text, p_visual, _, _ = build_semantic_operators_from_flags(unique, flags, 4)
    model = _model("I3").eval()
    head = torch.nn.Linear(model.out_dim, 3)
    x_text = torch.randn(4, 5)
    x_visual = torch.randn(4, 7)
    original = head(model(x_text, x_visual, p_phy, p_text, p_visual))
    assert torch.isfinite(original).all()
    original.square().mean().backward()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "head": head.state_dict()}, path)
    reloaded_model = _model("I3").eval()
    reloaded_head = torch.nn.Linear(reloaded_model.out_dim, 3)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    reloaded_model.load_state_dict(payload["model"])
    reloaded_head.load_state_dict(payload["head"])
    replay = reloaded_head(
        reloaded_model(x_text, x_visual, p_phy, p_text, p_visual)
    )
    assert torch.equal(original.detach(), replay.detach())


def test_train_val_label_view_does_not_expose_test_targets():
    from scripts.run_mag_information_utility_v1 import make_train_val_label_view

    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    train = torch.tensor([0, 1])
    valid = torch.tensor([2, 3])
    test = torch.tensor([4, 5])
    visible = make_train_val_label_view(labels, train, valid)
    changed = labels.clone()
    changed[test] = torch.tensor([99, 99])
    changed_visible = make_train_val_label_view(changed, train, valid)
    assert torch.equal(visible, changed_visible)
    assert torch.equal(visible[4:], torch.full((2,), -100))


def test_evaluation_preserves_torch_rng_state():
    from scripts.run_mag_information_utility_v1 import evaluate_logits

    logits = torch.tensor([[3.0, 0.0], [0.1, 0.2], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 1])
    before = torch.random.get_rng_state().clone()
    metrics = evaluate_logits(logits, labels, torch.arange(3), num_classes=2)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert set(metrics) == {"cross_entropy", "accuracy", "macro_f1"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_run_provenance_reads_the_graph_bundle_cache_field(tmp_path, monkeypatch):
    import scripts.run_mag_information_utility_v1 as runner

    split_path = tmp_path / "split.pt"
    cache_path = tmp_path / "candidates.pt"
    split_path.write_bytes(b"split")
    cache_path.write_bytes(b"cache")
    monkeypatch.setattr(runner, "nc_split_path", lambda cfg: split_path)
    monkeypatch.setattr(runner, "_candidate_cache_path", lambda dataset: cache_path)
    monkeypatch.setattr(runner, "file_sha256", lambda path: f"sha:{Path(path).name}")
    metadata = {
        "feature_fingerprint": "feature-fp",
        "candidate_fingerprint": "candidate-fp",
        "v7b_fingerprint": "v7b-fp",
    }
    bundle = {
        "cache": {"metadata": metadata},
        "random_audit": {
            "source_feature_fingerprint": "feature-fp",
            "source_candidate_fingerprint": "candidate-fp",
            "source_v7b_fingerprint": "v7b-fp",
            "physical_edge_fingerprint": "physical-fp",
        },
    }
    data = SimpleNamespace(
        num_nodes=4,
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x_t=np.ones((4, 3), dtype=np.float32),
        x_i=np.ones((4, 2), dtype=np.float32),
    )
    cfg = OmegaConf.create({"dataset": {"name": "Movies"}, "seed": 42})
    provenance = runner._run_provenance("Movies", 42, cfg, bundle, data)
    assert provenance["semantic_cache_path"] == str(cache_path)
    assert provenance["semantic_cache_sha256"] == "sha:candidates.pt"
    assert provenance["feature_fingerprint"] == "feature-fp"
    assert provenance["physical_edge_fingerprint"] == "physical-fp"


def test_checkpoint_node_ce_replay_uses_narrow_measured_roundoff_tolerance():
    from scripts.run_mag_information_utility_v1 import _assert_node_ce_replay

    reference = np.asarray([0.25, 1.75, 2.5], dtype=np.float32)
    replay = reference + np.asarray([1.9e-6, -2.1e-6, 0.0], dtype=np.float32)
    observed = _assert_node_ce_replay(replay, reference)
    assert 2.0e-6 < observed <= 3.0e-6
    with pytest.raises(AssertionError, match="exceeded the measured roundoff envelope"):
        _assert_node_ce_replay(reference + 3.1e-6, reference)
