from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.analysis.pace_mag_v1_audit import audit_forward, tensor_sha256, validation_metrics
from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b
from src.models import build_model
from src.models.pace_mag_v1 import build_semantic_operators


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    rng = np.random.default_rng(103)
    text = rng.normal(size=(14, 7)).astype(np.float32)
    visual = rng.normal(size=(14, 5)).astype(np.float32)
    physical = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 7, 6, 8, 7]],
        dtype=torch.long,
    )
    cache_path, metadata = build_semantic_candidate_cache_v7b(
        dataset="AuditToy", text_features=text, visual_features=visual,
        physical_edge_index=physical, cache_dir=tmp_path_factory.mktemp("pace_audit_cache"),
    )
    return {
        "text": torch.from_numpy(text),
        "visual": torch.from_numpy(visual),
        "x": torch.from_numpy(np.concatenate((text, visual), axis=1)),
        "physical": physical,
        "cache_path": str(cache_path),
        "metadata": metadata,
    }


def make_model(graph, *, chunk_size=32768):
    cfg = OmegaConf.create({
        "dataset": {"name": "AuditToy"},
        "model": {
            "name": "pace_mag_v1", "variant": "e2_full", "hidden_dim": 16, "dropout": 0.0,
            "num_experts": 3, "expert_bottleneck": 8, "expert_chunk_size": chunk_size,
            "activation_checkpointing": True, "router_hidden_dim": 9,
            "relation_embedding_dim": 4, "modality_embedding_dim": 3,
            "cross_modal_bottleneck": 6, "rho_init": 0.1, "rho_max": 0.25,
            "beta_phy_init": 0.5, "beta_sem_init": 0.3, "beta_max": 1.0,
            "gamma": 0.2, "semantic_edge_weight": 0.8, "eps": 1e-8,
        },
        "task": {"name": "nc", "evaluate_test": False},
        "seed": 42,
    })
    info = {"input_dim": 12, "num_nodes": 14, "num_classes": 3, "text_dim": 7, "visual_dim": 5}
    model = build_model(cfg, info)
    meta = graph["metadata"]
    model.load_semantic_cache(
        graph["cache_path"], expected_dataset="AuditToy",
        expected_feature_fingerprint=meta["feature_fingerprint"],
        expected_candidate_fingerprint=meta["candidate_fingerprint"],
        expected_v7b_fingerprint=meta["v7b_fingerprint"],
    )
    return model.eval()


def test_audit_forward_reproduces_original_embedding_and_logits(graph):
    model = make_model(graph)
    classifier = torch.nn.Linear(16, 3).eval()
    with torch.inference_mode():
        original = model(graph["x"], graph["physical"])[0]
        original_logits = classifier(original)
        trace = audit_forward(model, graph["x"], graph["physical"], classifier, capture=True)
    assert torch.allclose(original, trace.embedding, atol=1e-6, rtol=1e-6)
    assert torch.allclose(original_logits, trace.logits, atol=1e-6, rtol=1e-6)
    assert len(trace.routes) == 6 and len(trace.aggregates) == 6
    assert len(trace.expert_statistics) == 6


def test_checkpoint_reload_with_matching_semantic_cache_preserves_audit_output(graph, tmp_path):
    first = make_model(graph)
    classifier = torch.nn.Linear(16, 3).eval()
    checkpoint = tmp_path / "toy_e2.pt"
    torch.save({"model_state": first.state_dict(), "head_state": classifier.state_dict()}, checkpoint)
    second = make_model(graph)
    restored_classifier = torch.nn.Linear(16, 3).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    second.load_state_dict(payload["model_state"], strict=True)
    restored_classifier.load_state_dict(payload["head_state"], strict=True)
    with torch.inference_mode():
        expected = audit_forward(first, graph["x"], graph["physical"], classifier)
        observed = audit_forward(second, graph["x"], graph["physical"], restored_classifier)
    assert torch.allclose(expected.embedding, observed.embedding, atol=1e-7, rtol=1e-7)
    assert torch.allclose(expected.logits, observed.logits, atol=1e-7, rtol=1e-7)


def test_source_graph_statistics_verify_sparse_reconstruction_and_weight_scale(graph):
    from scripts.audit_pace_mag_v1 import _context_rows, _source_graph_diagnostics
    from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b

    model = make_model(graph)
    model._audit_raw_features = {"text": graph["text"], "visual": graph["visual"]}
    cache = load_semantic_candidate_cache_v7b(graph["cache_path"], expected_dataset="AuditToy")
    with torch.inference_mode():
        model(graph["x"], graph["physical"])
        rows = _source_graph_diagnostics("AuditToy", model, cache)
        trace_on = audit_forward(model, graph["x"], graph["physical"], capture=True)
        trace_off = audit_forward(model, graph["x"], graph["physical"], cross_modal=False, capture=True)
        context_rows, cross_rows = _context_rows(
            "AuditToy", trace_on, trace_off, model.semantic_q_text, model.semantic_q_visual,
        )
    values = {row["metric"]: row.get("value") for row in rows if row["family"] == "linear_reconstruction" and row["modality"] == "text"}
    assert values["max_abs_q_weighted_context_minus_PU_H"] < 1e-5
    assert values["max_abs_reconstructed_row_mean_minus_PU_row_mean"] < 1e-5
    scale = next(row["value"] for row in rows if row["metric"] == "max_abs_text_context_0_8_vs_0_9")
    assert scale < 1e-5
    source_rms = [row for row in context_rows if row["family"] == "source_context_rms"]
    assert len(source_rms) == 8
    assert all(np.isfinite(row["value"]) for row in source_rms)
    assert len(cross_rows) == 4


def test_cross_modal_off_is_frozen_and_original_forward_restores(graph):
    model = make_model(graph)
    classifier = torch.nn.Linear(16, 3).eval()
    before = tensor_sha256((model, classifier))
    with torch.inference_mode():
        baseline = audit_forward(model, graph["x"], graph["physical"], classifier, capture=True)
        off = audit_forward(model, graph["x"], graph["physical"], classifier, cross_modal=False)
        restored = model(graph["x"], graph["physical"])[0]
    assert torch.isfinite(off.embedding).all()
    assert torch.allclose(restored, baseline.embedding, atol=1e-6, rtol=1e-6)
    assert not torch.equal(off.embedding, baseline.embedding)
    assert tensor_sha256((model, classifier)) == before
    assert not model.training and not classifier.training


def test_router_interventions_keep_invalid_rows_zero_and_permutation_is_seeded(graph):
    model = make_model(graph)
    # Replace only this test instance's sparse cache with one valid edge. Nodes
    # 2..N-1 therefore have no semantic source context.
    views = build_semantic_operators(
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[1, 0]], dtype=torch.uint8),
        model.num_nodes,
    )
    model.semantic_pu = views["pu"]
    model.semantic_pt = views["pt"]
    model.semantic_pv = views["pv"]
    model.semantic_q_union = views["q_union"]
    model.semantic_q_text = views["q_text"]
    model.semantic_q_visual = views["q_visual"]
    with torch.inference_mode():
        original = audit_forward(model, graph["x"], graph["physical"], router_mode="original", capture=True)
        uniform = audit_forward(model, graph["x"], graph["physical"], router_mode="uniform", capture=True)
        mean = audit_forward(model, graph["x"], graph["physical"], router_mode="view_mean", capture=True)
        perm_a = audit_forward(model, graph["x"], graph["physical"], router_mode="node_permuted", router_seed=51, capture=True)
        perm_b = audit_forward(model, graph["x"], graph["physical"], router_mode="node_permuted", router_seed=51, capture=True)
    for relation in ("text", "visual"):
        route = original.routes[(relation, "text")]
        q = model.semantic_q_text if relation == "text" else model.semantic_q_visual
        valid = q > 0
        assert torch.equal(route[~valid], torch.zeros_like(route[~valid]))
        assert torch.equal(uniform.routes[(relation, "text")][~valid], torch.zeros_like(route[~valid]))
        assert torch.equal(mean.routes[(relation, "text")][~valid], torch.zeros_like(route[~valid]))
        assert torch.equal(perm_a.routes[(relation, "text")][~valid], torch.zeros_like(route[~valid]))
        assert torch.equal(perm_a.routes[(relation, "text")], perm_b.routes[(relation, "text")])
        if bool(valid.any()):
            assert torch.allclose(uniform.routes[(relation, "text")][valid].sum(dim=-1), torch.ones(int(valid.sum())))
            mean_rows = mean.routes[(relation, "text")][valid]
            assert torch.allclose(mean_rows, mean_rows[:1].expand_as(mean_rows))
            assert torch.allclose(torch.sort(route[valid], dim=0).values, torch.sort(perm_a.routes[(relation, "text")][valid], dim=0).values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"physical_response": False},
        {"semantic_response": False},
        {"intrinsic_skip": False},
    ],
)
def test_physical_semantic_and_intrinsic_skip_gates_are_independent_and_restore(graph, kwargs):
    model = make_model(graph)
    classifier = torch.nn.Linear(16, 3).eval()
    with torch.inference_mode():
        baseline = audit_forward(model, graph["x"], graph["physical"], classifier, capture=True)
        intervened = audit_forward(model, graph["x"], graph["physical"], classifier, **kwargs)
        restored = audit_forward(model, graph["x"], graph["physical"], classifier)
    assert torch.isfinite(intervened.embedding).all()
    assert torch.isfinite(intervened.logits).all()
    assert torch.allclose(restored.embedding, baseline.embedding, atol=1e-6, rtol=1e-6)
    assert not torch.equal(intervened.embedding, baseline.embedding)


def test_validation_metrics_derive_class_set_from_train_and_val_only():
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    labels = torch.tensor([0, 1, 2, 99, 99])
    train_idx = torch.tensor([0, 1])
    val_idx = torch.tensor([2])
    metrics, class_rows = validation_metrics(logits, labels, train_idx, val_idx, num_classes=3)
    assert metrics["eval_label_ids_train_val_only"] == [0, 1, 2]
    assert metrics["val_acc"] == pytest.approx(1.0)
    assert metrics["val_macro_f1"] == pytest.approx(1 / 3)
    assert len(class_rows) == 3
    assert 99 not in metrics["eval_label_ids_train_val_only"]


def test_intervention_row_reports_deltas_against_baseline_metrics():
    from types import SimpleNamespace
    from scripts.audit_pace_mag_v1 import _intervention_row

    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]])
    trace = SimpleNamespace(embedding=torch.ones((4, 3)), logits=logits)
    metric = {
        "val_ce": 0.048587, "val_acc": 1.0, "val_macro_f1": 1.0,
        "_labels": torch.tensor([0, 1, 0, 1]),
        "_train_idx": torch.tensor([0, 1]), "_val_idx": torch.tensor([2, 3]), "_num_classes": 2,
    }
    row = _intervention_row("AuditToy", "R0_original", trace, trace, metric, trace, mode="original", seed=17)
    assert row["delta_val_ce"] == pytest.approx(0.0, abs=1e-6)
    assert row["delta_val_acc"] == 0.0
    assert row["delta_val_macro_f1"] == 0.0


def test_trained_parameter_audit_summarizes_bounded_scalars_and_router_weights(graph):
    from scripts.audit_pace_mag_v1 import _trained_parameter_audit

    model = make_model(graph)
    result = _trained_parameter_audit("AuditToy", model)
    rho = result["bounded_scalars"]["rho"]
    assert len(rho["effective"]) == 2 and all(len(row) == 2 for row in rho["effective"])
    assert np.allclose(
        rho["effective_derivative_wrt_raw"],
        np.asarray(rho["effective"]) * (1 - np.asarray(rho["effective"]) / rho["maximum"]),
    )
    router = result["router"]
    assert router["trainable_parameter_count"] == sum(p.numel() for p in model.router.parameters())
    assert router["conditioning_embedding_parameter_count"] > 0
    assert router["trained_parameter_tensors"]
    assert all(len(row["sha256"]) == 64 and np.isfinite(row["rms"]) for row in router["trained_parameter_tensors"])


def test_audit_executes_without_optimizer_and_cache_mismatch_is_rejected(graph, monkeypatch):
    model = make_model(graph)
    classifier = torch.nn.Linear(16, 3).eval()

    def forbidden_step(*args, **kwargs):
        raise AssertionError("an audit must not run an optimizer step")

    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden_step)
    with torch.inference_mode():
        result = audit_forward(model, graph["x"], graph["physical"], classifier)
    assert torch.isfinite(result.embedding).all()

    cfg = OmegaConf.create({"dataset": {"name": "AuditToy"}, "model": {"name": "pace_mag_v1", "variant": "e2_full", "hidden_dim": 16, "num_experts": 3, "expert_bottleneck": 8, "router_hidden_dim": 9, "relation_embedding_dim": 4, "modality_embedding_dim": 3, "cross_modal_bottleneck": 6}})
    wrong = build_model(cfg, {"input_dim": 12, "num_nodes": 14, "num_classes": 3, "text_dim": 7, "visual_dim": 5})
    with pytest.raises(ValueError, match="candidate graph fingerprint"):
        wrong.load_semantic_cache(graph["cache_path"], expected_dataset="AuditToy", expected_candidate_fingerprint="wrong")
