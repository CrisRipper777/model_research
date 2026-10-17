from __future__ import annotations

import inspect
import io

import pytest
import torch
from omegaconf import OmegaConf

import src.models.hybrid_cge_v11 as hybrid_module
from src.models.hybrid_cge_v11 import CONDITIONS, Model, receiver_sparse_attention


def _cfg(condition: str):
    model_cfg = OmegaConf.load("configs/model/hybrid_cge_v11.yaml")
    model_cfg.condition = condition
    model_cfg.late_cross_dropout = 0.0 if condition == CONDITIONS[0] else 0.15
    return OmegaConf.create({"model": model_cfg, "seed": 42})


def _operators(model):
    unique = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 1]], dtype=torch.uint8)
    from src.models.rmcge_v2 import build_modality_knn_operators

    operators = build_modality_knn_operators(unique, flags, 5)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index_v2", operators[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight_v2", operators[modality]["weight"])
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_v7b_fingerprint = "b" * 64
    return operators


def _model(condition: str, seed: int = 42):
    torch.manual_seed(seed)
    model = Model(
        _cfg(condition),
        {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 5, "num_classes": 3},
    )
    _operators(model)
    return model


def _physical_edges():
    return torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )


def _cached_text_physical_cross(model, x):
    inputs = model._split_modalities(x)
    priors = {m: model.projectors[m](inputs[m]) for m in ("text", "visual")}
    src, dst, physical, active, _ = model._normalized_operator(_physical_edges(), x.size(0), x.dtype)
    states, scale = model._physical_states(priors["text"], src, dst, physical, active)
    raw = model._physical_cross_message(
        priors["visual"], model.cross_projections["visual_to_text"], src, dst,
        physical, active, model.propagation_chunk_size,
    )
    hat = raw / scale * active.to(raw.dtype).unsqueeze(-1)
    return priors["text"], states, scale, raw, hat, (src, dst, physical, active)


def test_h1_h2_have_exact_common_initialization_and_one_shared_expert_bank():
    torch.manual_seed(42)
    h1 = Model(_cfg(CONDITIONS[0]), {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 5, "num_classes": 3})
    head1 = torch.nn.Linear(256, 3)
    torch.manual_seed(42)
    h2 = Model(_cfg(CONDITIONS[1]), {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 5, "num_classes": 3})
    head2 = torch.nn.Linear(256, 3)
    assert h1.state_dict().keys() == h2.state_dict().keys()
    for key, value in h1.state_dict().items():
        assert torch.equal(value, h2.state_dict()[key]), key
    assert torch.equal(head1.weight, head2.weight)
    assert torch.equal(head1.bias, head2.bias)
    assert len(h1.experts) == len({id(expert) for expert in h1.experts}) == 4
    assert len(h2.experts) == len({id(expert) for expert in h2.experts}) == 4
    assert not any("role_film" in key for key in h1.state_dict())
    assert not any("role_film" in key for key in h2.state_dict())
    assert h1.cross_dropout_p == 0.0 and h2.cross_dropout_p == 0.15
    assert torch.allclose(h1.physical_cross_coefficients, torch.full((4,), 0.25))
    assert torch.allclose(h1.tau, torch.full((2,), 0.25), atol=1e-7)


def test_private_state_ignores_cross_message_and_collaborative_uses_it(monkeypatch):
    model = _model(CONDITIONS[0], 7).eval()
    x = torch.randn(5, 7)
    prior, states, scale, raw, cross_hat, (src, dst, physical, active) = _cached_text_physical_cross(model, x)
    observed_cross = []
    original_correction = model._node_correction

    def spy(h, u1, delta, cross, modality_index):
        observed_cross.append(cross.detach().clone())
        return original_correction(h, u1, delta, cross, modality_index)

    monkeypatch.setattr(model, "_node_correction", spy)
    private_a, collab_a, _, diag_a = model._private_and_collaborative_state(
        "text", 0, prior, states, scale, raw, cross_hat, active, True
    )
    private_b, collab_b, _, _ = model._private_and_collaborative_state(
        "text", 0, prior, states, scale, torch.zeros_like(raw), torch.zeros_like(cross_hat), active, False
    )
    assert torch.equal(private_a, private_b)
    assert torch.allclose(collab_b, private_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(collab_a, private_a, atol=1e-7, rtol=1e-7)
    assert torch.equal(observed_cross[0], torch.zeros_like(cross_hat))
    assert torch.equal(observed_cross[1], cross_hat)
    assert diag_a["physical_cross_calibrated_rms"] > 0

    # Each expert sees its private profile, same-source profile, and then the
    # corresponding early-cross-conditioned profile through the same object.
    calls: dict[int, list[torch.Tensor]] = {id(expert): [] for expert in model.experts}
    handles = []
    for expert in model.experts:
        handles.append(
            expert.register_forward_pre_hook(
                lambda module, args, store=calls: store[id(module)].append(args[0].detach().clone())
            )
        )
    model._private_and_collaborative_state("text", 0, prior, states, scale, raw, cross_hat, active, False)
    for handle in handles:
        handle.remove()
    for expert_id, expert in enumerate(model.experts):
        values = calls[id(expert)]
        assert len(values) == 3
        assert torch.allclose(
            values[2] - values[0], model.physical_cross_coefficients[expert_id] * cross_hat,
            atol=1e-6, rtol=1e-5,
        )


def test_receiver_sparse_attention_is_edge_only_normalized_and_differentiable():
    torch.manual_seed(9)
    query = torch.randn(5, 4, requires_grad=True)
    key = torch.randn(5, 4, requires_grad=True)
    value = torch.randn(5, 6, requires_grad=True)
    edges = torch.tensor([[0, 1, 3, 4, 0, 2], [2, 2, 1, 1, 4, 4]])
    output, attention = receiver_sparse_attention(query, key, value, edges, chunk_size=2)
    sums = torch.zeros(5).index_add_(0, edges[1], attention.detach())
    receivers = torch.unique(edges[1])
    assert torch.allclose(sums[receivers], torch.ones_like(sums[receivers]), atol=1e-7)
    assert torch.equal(output[[0, 3]], torch.zeros_like(output[[0, 3]]))
    (output.square().mean() + attention.square().mean()).backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.norm() > 0


def test_late_attention_uses_opposite_source_knn_and_same_node_is_h2_only(monkeypatch):
    model = _model(CONDITIONS[0], 3).eval()
    expected = model.semantic_visual_edge_index_v2.clone()
    wrong = model.semantic_text_edge_index_v2.clone()
    assert not torch.equal(expected, wrong)
    original = hybrid_module.receiver_sparse_attention
    observed = {}

    def spy(query, key, value, edge_index, chunk_size=8192):
        observed["edges"] = edge_index.clone()
        return original(query, key, value, edge_index, chunk_size)

    monkeypatch.setattr(hybrid_module, "receiver_sparse_attention", spy)
    u_collab = torch.randn(5, 256)
    u_private = torch.randn(5, 256)
    h_source = torch.randn(5, 256)
    _, _, h1_weights, h1_diag, _ = model._late_completion(
        "text", 0, "visual", h_source, u_private, u_collab, True
    )
    assert torch.equal(observed["edges"], expected)
    assert torch.equal(h1_weights[:, 1], torch.zeros(5))
    assert torch.allclose(h1_weights[:, 0] + h1_weights[:, 2], torch.ones(5))
    assert h1_diag["same_node_support_nodes"] == 0
    assert "physical_cross" not in inspect.signature(model._late_completion).parameters

    h2 = _model(CONDITIONS[1], 3).eval()
    _, _, h2_weights, h2_diag, _ = h2._late_completion(
        "text", 0, "visual", h_source, u_private, u_collab, True
    )
    assert torch.allclose(h2_weights[:, :].sum(dim=1), torch.ones(5))
    assert h2_diag["same_node_support_nodes"] == 5
    assert torch.allclose(h2_weights[0], torch.tensor([0.4, 0.4, 0.2]), atol=1e-7)
    assert torch.allclose(h1_weights[0], torch.tensor([2.0 / 3.0, 0.0, 1.0 / 3.0]), atol=1e-7)


def test_empty_late_candidates_give_exact_zero_delta_and_relative_control_never_amplifies():
    model = _model(CONDITIONS[0], 11).eval()
    empty = torch.empty((2, 0), dtype=torch.long)
    model.semantic_visual_edge_index_v2 = empty
    model.semantic_visual_weight_v2 = torch.empty((0,))
    base = torch.randn(5, 256)
    source = torch.randn_like(base)
    final, delta, weights, _, _ = model._late_completion(
        "text", 0, "visual", source, source, base, False
    )
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(final, base)
    assert torch.equal(weights, torch.tensor([[0.0, 0.0, 1.0]]).expand_as(weights))

    small = torch.full((2, 4), 0.01)
    reference = torch.full((2, 4), 2.0)
    unchanged, scale = model._relative_control(small, reference, torch.tensor(0.25), 1e-8)
    assert torch.equal(scale, torch.ones_like(scale))
    assert torch.equal(unchanged, small)
    large = torch.full((2, 4), 10.0)
    clipped, scale = model._relative_control(large, torch.ones_like(large), torch.tensor(0.25), 1e-8)
    assert torch.all(scale < 1.0)
    assert torch.all(clipped.norm(dim=-1) <= 0.25 * reference.new_full((2,), 2.0).norm(dim=-1) + 1e-6)


def test_h2_dropout_is_nodewise_noninverted_training_only_and_late_only():
    h1 = _model(CONDITIONS[0], 2)
    h2 = _model(CONDITIONS[1], 2)
    residual = torch.ones(512, 256)
    h1.train()
    h1_out, h1_keep = h1._apply_late_cross_dropout(residual)
    assert torch.equal(h1_out, residual) and h1_keep.all()
    h2.train()
    torch.manual_seed(4)
    dropped, keep = h2._apply_late_cross_dropout(residual)
    assert 0.10 < float((~keep).float().mean()) < 0.20
    assert torch.equal(dropped[keep[:, 0]], residual[keep[:, 0]])
    assert torch.equal(dropped[~keep[:, 0]], torch.zeros_like(dropped[~keep[:, 0]]))
    assert set(torch.unique(dropped).tolist()) <= {0.0, 1.0}
    h2.eval()
    eval_out, eval_keep = h2._apply_late_cross_dropout(residual)
    assert torch.equal(eval_out, residual) and eval_keep.all()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_full_graph_loss_gradients_update_eval_rng_and_checkpoint_roundtrip(condition):
    model = _model(condition, 19).train()
    x = torch.randn(5, 7)
    z, _, _, balance_aux, info = model(x, _physical_edges())
    assert z.shape == (5, 256)
    assert set(info["unimodal_logits"]) == {"text", "visual"}
    assert info["role_film_enabled"] is False
    assert torch.isfinite(z).all() and torch.isfinite(balance_aux)
    labels = torch.tensor([0, 1, 2, 1, 0])
    train_idx = torch.tensor([0, 1, 2])
    from src.tasks.nc import _compose_full_graph_loss

    loss, private_ce = _compose_full_graph_loss(
        model=model,
        info=info,
        main_ce=torch.nn.functional.cross_entropy(torch.nn.Linear(256, 3)(z[train_idx]), labels[train_idx]),
        aux_loss=balance_aux,
        aux_weight=1.0,
        unimodal_weight=0.3,
        labels=torch.tensor([0, 1, 2, -100, -100]),
        train_idx=train_idx,
    )
    assert private_ce is not None and torch.isfinite(private_ce) and torch.isfinite(loss)
    loss.backward()
    gradients = model.capture_gradient_diagnostics()
    for key, row in gradients.items():
        assert row["gradient_finite"], key
    for key in (
        "physical_cross_coefficients",
        "cross_projections.visual_to_text.0.weight",
        "cross_projections.text_to_visual.0.weight",
        "query_projections.text.weight",
        "key_projections.visual.weight",
        "source_selector.2.weight",
        "channel_gate.2.weight",
        "comp_router.3.weight",
        "unimodal_heads.text.weight",
    ):
        assert gradients[key]["gradient_present"] and gradients[key]["gradient_nonzero"], key
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    model.eval()
    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        eval_z = model(x, _physical_edges())[0]
    assert torch.equal(rng_before, torch.random.get_rng_state())
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = _model(condition, 19).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        restored_z = restored(x, _physical_edges())[0]
    assert torch.equal(eval_z, restored_z)


def test_global_cache_index_contract_and_missing_physical_edges_are_explicit():
    model = _model(CONDITIONS[0])
    with pytest.raises(ValueError, match="global physical edge_index"):
        model(torch.randn(5, 7), None)
    with pytest.raises(RuntimeError, match="global IDs"):
        model(torch.randn(4, 7), _physical_edges())
