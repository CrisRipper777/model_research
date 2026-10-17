from __future__ import annotations

import io
import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.models import cg_cge_v1
from src.models.cg_cge_v1 import CONDITIONS, Model, receiver_sparse_attention


def _cfg(condition: str):
    model_cfg = OmegaConf.load("configs/model/cg_cge_v1.yaml")
    model_cfg.condition = condition
    return OmegaConf.create({"model": model_cfg, "seed": 42})


def _operators(model):
    # Unique canonical pairs, with deliberately different source supports.
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
    model = Model(_cfg(condition), {
        "input_dim": 7, "text_dim": 3, "visual_dim": 4,
        "num_nodes": 5, "num_classes": 3,
    })
    _operators(model)
    return model


def _physical_edges():
    return torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )


def test_receiver_sparse_softmax_and_query_key_value_gradients():
    torch.manual_seed(7)
    query = torch.randn(5, 4, requires_grad=True)
    key = torch.randn(5, 4, requires_grad=True)
    value = torch.randn(5, 6, requires_grad=True)
    edges = torch.tensor([[0, 1, 3, 4, 0, 2], [2, 2, 1, 1, 4, 4]])
    result, attention = receiver_sparse_attention(query, key, value, edges, chunk_size=2)
    row_sum = torch.zeros(5).index_add_(0, edges[1], attention.detach())
    supported = torch.unique(edges[1])
    assert torch.allclose(row_sum[supported], torch.ones_like(row_sum[supported]), atol=1e-7)
    assert torch.equal(result[[0, 3]], torch.zeros_like(result[[0, 3]]))
    loss = (result * torch.randn_like(result)).sum()
    loss.backward()
    for parameter in (query, key, value):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_conditions_share_initial_tree_head_and_expert_bank():
    b1 = _model(CONDITIONS[0], 42)
    b2 = _model(CONDITIONS[1], 42)
    assert b1.state_dict().keys() == b2.state_dict().keys()
    for key, value in b1.state_dict().items():
        assert torch.equal(value, b2.state_dict()[key]), key
    head1 = torch.nn.Linear(256, 3)
    torch.manual_seed(42)
    b1 = _model(CONDITIONS[0], 42)
    head1 = torch.nn.Linear(256, 3)
    torch.manual_seed(42)
    b2 = _model(CONDITIONS[1], 42)
    head2 = torch.nn.Linear(256, 3)
    assert torch.equal(head1.weight, head2.weight)
    assert torch.equal(head1.bias, head2.bias)
    assert b1.experts is not b2.experts
    assert len(b1.experts) == len(b2.experts) == 4
    assert not b1.role_film_scale.requires_grad
    assert b2.role_film_scale.requires_grad
    x = torch.randn(5, 256)
    assert torch.equal(b2._expert_input(0, 0, 0, x), x)


def test_forward_auxiliary_heads_gradients_checkpoint_and_eval_rng():
    model = _model("B2_conditioned_experts", 11).train()
    x = torch.randn(5, 7)
    z, _, _, aux, info = model(x, _physical_edges())
    assert z.shape == (5, 256)
    assert set(info["unimodal_logits"]) == {"text", "visual"}
    assert torch.isfinite(z).all() and torch.isfinite(aux)
    loss = z.square().mean() + aux + sum(v.square().mean() for v in info["unimodal_logits"].values())
    loss.backward()
    diagnostics = model.capture_gradient_diagnostics()
    assert model._gradient_diagnostics == diagnostics
    assert diagnostics["view_router.2.weight"]["gradient_nonzero"]
    for name in ("query_projections.text.weight", "key_projections.visual.weight",
                 "cross_projections.visual_to_text.0.weight", "unimodal_heads.text.weight",
                 "projectors.text.linear1.weight", "role_film_scale", "role_film_shift"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
    before = model.role_film_scale.detach().clone()
    before_shift = model.role_film_shift.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert not torch.equal(before, model.role_film_scale)
    assert not torch.equal(before_shift, model.role_film_shift)

    model.eval()
    cpu_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        eval_z, _, _, _, eval_info = model(x, _physical_edges())
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    assert "unimodal_logits" not in eval_info
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = _model("B2_conditioned_experts", 11).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        restored_z = restored(x, _physical_edges())[0]
    assert torch.allclose(eval_z, restored_z, atol=0, rtol=0)


def test_empty_and_single_source_candidate_masks_are_exact():
    model = _model("B1_complementary", 3).eval()
    u = torch.randn(5, 256)
    h = torch.randn_like(u)
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    model.semantic_text_edge_index_v2 = empty_edges
    model.semantic_visual_edge_index_v2 = empty_edges
    no_candidate, delta, weights, _, _ = model._cross_completion(
        "text", 0, "visual", u, h, h,
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(0), torch.zeros(5, dtype=torch.bool),
    )
    assert torch.equal(no_candidate, u)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(weights, torch.tensor([[0.0, 0.0, 1.0]]).expand_as(weights))

    src = torch.tensor([0, 1])
    dst = torch.tensor([1, 0])
    active = torch.tensor([True, True, False, False, False])
    physical_only, delta, weights, _, _ = model._cross_completion(
        "text", 0, "visual", u, h, h, src, dst, torch.ones(2), active,
    )
    assert torch.isfinite(physical_only).all()
    assert torch.equal(weights[:, 1], torch.zeros(5))
    assert torch.equal(weights[2:, 0], torch.zeros(3))
    assert torch.equal(weights[2:, 1], torch.zeros(3))
    assert torch.equal(weights[2:, 2], torch.ones(3))
    assert torch.equal(delta[2:], torch.zeros_like(delta[2:]))


def test_text_completion_uses_visual_support_and_initial_selector_probabilities(monkeypatch):
    model = _model("B1_complementary", 13).eval()
    expected_visual_support = model.semantic_visual_edge_index_v2.clone()
    target_support = model.semantic_text_edge_index_v2.clone()
    assert not torch.equal(expected_visual_support, target_support)
    original = cg_cge_v1.receiver_sparse_attention
    observed = {}

    def spy(query, key, value, edge_index, chunk_size=8192):
        observed["edges"] = edge_index.clone()
        return original(query, key, value, edge_index, chunk_size)

    monkeypatch.setattr(cg_cge_v1, "receiver_sparse_attention", spy)
    u_target, u_source = torch.randn(5, 256), torch.randn(5, 256)
    src, dst = torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4])
    active = torch.tensor([False, True, True, True, True])
    _, _, source_weights, _, _ = model._cross_completion(
        "text", 0, "visual", u_target, u_source, u_source,
        src, dst, torch.ones(4), active, True,
    )
    assert torch.equal(observed["edges"], expected_visual_support)
    assert not torch.equal(observed["edges"], target_support)
    assert torch.allclose(source_weights[1], torch.tensor([0.45, 0.35, 0.20]), atol=1e-7)


def test_auxiliary_ce_is_opt_in_exact_and_train_indices_only():
    from src.tasks.nc import _compose_full_graph_loss

    class SupportsAux:
        supports_unimodal_aux_loss = True

    class Legacy:
        supports_unimodal_aux_loss = False

    main = torch.tensor(1.3, requires_grad=True)
    balance = torch.tensor(0.4, requires_grad=True)
    old_loss = main + 1.0 * balance
    actual_old, omitted = _compose_full_graph_loss(
        model=Legacy(), info={}, main_ce=main, aux_loss=balance,
        aux_weight=1.0, unimodal_weight=0.0,
        labels=torch.tensor([0, 1, -100]), train_idx=torch.tensor([0, 1]),
    )
    assert omitted is None
    assert torch.equal(actual_old, old_loss)

    logits = {
        "text": torch.tensor([[3.0, 0.0], [0.0, 2.0], [100.0, -100.0]], requires_grad=True),
        "visual": torch.tensor([[0.0, 1.0], [2.0, 0.0], [-100.0, 100.0]], requires_grad=True),
    }
    labels = torch.tensor([0, 1, -100])
    train_idx = torch.tensor([0, 1])
    total, unimodal = _compose_full_graph_loss(
        model=SupportsAux(), info={"unimodal_logits": logits},
        main_ce=main, aux_loss=balance, aux_weight=1.0,
        unimodal_weight=0.3, labels=labels, train_idx=train_idx,
    )
    expected = 0.5 * (
        torch.nn.functional.cross_entropy(logits["text"][train_idx], labels[train_idx])
        + torch.nn.functional.cross_entropy(logits["visual"][train_idx], labels[train_idx])
    )
    assert torch.allclose(unimodal, expected)
    assert torch.allclose(total, main + 0.3 * expected + balance)
    total.backward()
    for value in logits.values():
        assert value.grad is not None and value.grad[:2].norm() > 0
        assert torch.equal(value.grad[2], torch.zeros_like(value.grad[2]))
    before = logits["text"].detach().clone()
    torch.optim.SGD([logits["text"]], lr=0.1).step()
    assert not torch.equal(before, logits["text"])
    with pytest.raises(ValueError, match="declares supports_unimodal_aux_loss"):
        _compose_full_graph_loss(
            model=Legacy(), info={"unimodal_logits": logits}, main_ce=main,
            aux_loss=balance, aux_weight=1.0, unimodal_weight=0.3,
            labels=labels, train_idx=train_idx,
        )


def test_nc_trainer_updates_and_checkpoints_opt_in_unimodal_heads(tmp_path, monkeypatch):
    from src.tasks import nc as nc_task

    class TinyAuxModel(torch.nn.Module):
        supports_unimodal_aux_loss = True
        record_run_telemetry = True
        record_epoch_history = True
        requires_global_semantic_candidates = False
        out_dim = 4

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.unimodal_heads = torch.nn.ModuleDict({
                "text": torch.nn.Linear(4, 3),
                "visual": torch.nn.Linear(4, 3),
            })

        def forward(self, x, edge_index, *, return_details=False):
            del edge_index, return_details
            z = self.encoder(x)
            info = {}
            if self.training:
                info["unimodal_logits"] = {name: head(z) for name, head in self.unimodal_heads.items()}
            aux = z.new_zeros(())
            return z, None, None, aux, info

    torch.manual_seed(19)
    initial_model = TinyAuxModel()
    initial_heads = {key: value.detach().clone() for key, value in initial_model.unimodal_heads.state_dict().items()}
    monkeypatch.setattr(nc_task, "build_model", lambda cfg, data_info: TinyAuxModel())
    data = SimpleNamespace(
        x=torch.randn(7, 4), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        y=torch.tensor([0, 1, 2, 0, 1, 2, -100]),
        train_idx=torch.tensor([0, 1, 2, 3]), val_idx=torch.tensor([4, 5]), test_idx=torch.tensor([6]),
        num_nodes=7, num_classes=3, input_dim=4, x_t=None, x_i=None,
    )
    cfg = OmegaConf.create({
        "model": {"name": "cg_cge_v1", "hidden_dim": 4},
        "dataset": {"name": "tiny"}, "seed": 19, "num_runs": 1, "device": "cpu",
        "task": {
            "protocol_version": "unified_full_graph_nc_v1", "training_mode": "full_graph",
            "optimizer": "adamw", "lr": 1e-2, "weight_decay": 0.0, "epochs": 1,
            "patience": 2, "early_stop_min_epoch": 1, "early_stop_min_delta": 0.0,
            "grad_clip": 1.0, "batch_size": 4, "inference_mode": "full",
            "inference_batch_size": 8, "eval_every": 1, "scheduler": None,
            "evaluate_test": False, "record_extended_validation": True,
            "record_train_ce_separately": True,
            "save_ckpt_path": str(tmp_path / "best_accuracy.pt"),
            "best_f1_ckpt_path": str(tmp_path / "best_f1.pt"),
            "run_metrics_path": str(tmp_path / "metrics.json"),
            "extended_checkpoint_metadata": {"test_evaluation": False},
            "loss": {"aux_weight": 1.0, "unimodal_weight": 0.3},
        },
    })
    nc_task.run_nc(cfg, data, torch.device("cpu"), logging.getLogger("cg-cge-test"))
    payload = torch.load(tmp_path / "best_accuracy.pt", map_location="cpu", weights_only=False)
    for name, initial in initial_heads.items():
        updated = payload["model_state"][f"unimodal_heads.{name}"]
        assert not torch.equal(initial, updated)
    metrics = __import__("json").loads((tmp_path / "metrics.json").read_text())
    run = metrics["runs"][0]
    assert run["metadata"]["unimodal_auxiliary_weight"] == 0.3
    assert "train_unimodal_ce" in run["metadata"]["epoch_history"][0]
    assert not any(key.startswith("test_") for key in run["metrics"])


def test_custom_sparse_attention_matches_reference_forward_and_backward():
    torch.manual_seed(29)
    q = torch.randn(6, 5, requires_grad=True)
    k = torch.randn(6, 5, requires_grad=True)
    v = torch.randn(6, 4, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 4, 3, 5, 0], [2, 2, 2, 3, 3, 3, 5]])
    coeff = torch.randn(6, 4)
    attn_coeff = torch.randn(edges.size(1))

    actual, actual_attention = receiver_sparse_attention(q, k, v, edges, chunk_size=2)
    actual_loss = (actual * coeff).sum() + 0.07 * (actual_attention * attn_coeff).sum()
    actual_grads = torch.autograd.grad(actual_loss, (q, k, v))

    q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v))
    src, dst = edges
    logits = (q_ref[dst] * k_ref[src]).sum(-1) / (q_ref.size(-1) ** 0.5)
    ref_attention = torch.zeros_like(logits)
    ref_output = torch.zeros_like(v_ref)
    for receiver in torch.unique(dst):
        selected = dst == receiver
        ref_attention[selected] = torch.softmax(logits[selected], dim=0)
    ref_output.index_add_(0, dst, v_ref[src] * ref_attention[:, None])
    ref_loss = (ref_output * coeff).sum() + 0.07 * (ref_attention * attn_coeff).sum()
    reference_grads = torch.autograd.grad(ref_loss, (q_ref, k_ref, v_ref))
    assert torch.allclose(actual, ref_output, atol=1e-7, rtol=1e-7)
    assert torch.allclose(actual_attention, ref_attention, atol=1e-7, rtol=1e-7)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.allclose(actual_grad, reference_grad, atol=2e-6, rtol=2e-6)
