from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from src.models.mrcge_v13 import Model as NodeRouterModel
from src.models.mrcge_v13_view_global_router import Model as ViewGlobalRouterModel


DATA_INFO = {"input_dim": 8, "text_dim": 4, "visual_dim": 4, "num_nodes": 12, "num_classes": 3}
X = torch.arange(96, dtype=torch.float32).reshape(12, 8) / 31.0
PHYSICAL = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]])
SEMANTIC = torch.tensor([[0, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10, 0]])
LABELS = torch.tensor([0, 1, 2, 1, 0, 2, 0, 1, 2, 1, 0, 2])


def make_model(global_router: bool, seed: int = 42):
    torch.manual_seed(seed)
    cfg = OmegaConf.load(
        "configs/model/mrcge_v13_view_global_router.yaml" if global_router
        else "configs/model/mrcge_v13.yaml"
    )
    cfg.v13_variant = "joint_view"
    cfg.sender_private_ratio = 0.0
    model_cfg = OmegaConf.create({"model": cfg})
    cls = ViewGlobalRouterModel if global_router else NodeRouterModel
    model = cls(model_cfg, DATA_INFO)
    for modality, edges in (("text", SEMANTIC), ("visual", SEMANTIC.flip(0))):
        setattr(model, f"semantic_{modality}_edge_index", edges.clone())
        setattr(model, f"semantic_{modality}_weight", torch.ones(edges.size(1)))
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_feature_fingerprint = "b" * 64
    model.semantic_v7b_fingerprint = "c" * 64
    return model


def parameter_count(model):
    return sum(value.numel() for value in model.parameters())


def assert_global_summary(route):
    share = torch.tensor(route["top2_selection_share"])
    assert torch.isclose(share.sum(), torch.tensor(1.0))
    assert torch.equal((share > 1e-6), share == 0.5)
    assert int((share > 1e-6).sum()) == 2


def test_init_is_identical_and_no_state_or_classifier_parameters_change():
    reference = make_model(False, seed=123)
    reference_head = nn.Linear(reference.out_dim, DATA_INFO["num_classes"])
    candidate = make_model(True, seed=123)
    candidate_head = nn.Linear(candidate.out_dim, DATA_INFO["num_classes"])
    ref_state, new_state = reference.state_dict(), candidate.state_dict()
    assert set(ref_state) == set(new_state)
    assert all(torch.equal(value, new_state[name]) for name, value in ref_state.items())
    assert parameter_count(reference) == parameter_count(candidate)
    assert torch.equal(reference_head.weight, candidate_head.weight)
    assert torch.equal(reference_head.bias, candidate_head.bias)


def test_eval_routes_are_view_global_and_distinct_calls_remain_independent():
    model = make_model(True, seed=11).eval()
    basis = [torch.randn(12, 256) for _ in range(4)]
    current = torch.randn(12, 256)
    cross = torch.randn(12, 256)
    support = torch.tensor([True, True, True, False, True, False, True, True, False, True, False, True])
    _, route_a, balance_a, _ = model._filter(basis, support, current, 0, 0, cross, collect=True)
    assert torch.isfinite(balance_a)
    active = support
    assert torch.equal(route_a["selected"][active], route_a["selected"][active][:1].expand(int(active.sum()), -1))
    assert torch.equal(route_a["weights"][active], route_a["weights"][active][:1].expand(int(active.sum()), -1))

    # Each call consumes its own logits/support mean; no route is cached across views.
    logits_a = torch.tensor([[4.0, 2.0, 0.0, -1.0], [2.0, 4.0, 0.0, -1.0]])
    logits_b = torch.tensor([[-1.0, 0.0, 4.0, 2.0], [-1.0, 0.0, 2.0, 4.0]])
    effective_a = model._view_global_logits(logits_a, torch.ones(2, dtype=torch.bool))
    effective_b = model._view_global_logits(logits_b, torch.ones(2, dtype=torch.bool))
    assert torch.topk(effective_a[0], 2).indices.tolist() != torch.topk(effective_b[0], 2).indices.tolist()

    # Eval/collect traverses all six full-model calls and exposes standard route telemetry.
    with torch.no_grad():
        _, _, _, _, info = model(X, PHYSICAL, return_details=True)
    details = info["details"]
    names = ("private_text_physical", "private_text_semantic", "private_visual_physical",
             "private_visual_semantic")
    summaries = [details["routes"][name] for name in names]
    summaries += [details["collaborative_routes"][name] for name in ("text", "visual")]
    for summary in summaries:
        assert_global_summary(summary)
        assert len(summary["dense_importance"]) == 4
        assert torch.isfinite(torch.tensor(summary["dense_importance"])).all()


def test_low_memory_training_checkpoint_backward_and_empty_support_are_finite():
    model = make_model(True, seed=29).train()
    classifier = nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = 0.0
    router_calls = []
    handle = model.router.net[3].register_forward_hook(lambda *args: router_calls.append(1))
    output, _, _, aux_loss, _ = model(X, PHYSICAL, return_details=False)
    train_snapshots = list(model._last_route_snapshot.values())
    assert len(train_snapshots) == 6
    for snapshot in train_snapshots:
        summary = model._route_summary(snapshot["route"], snapshot["support"])
        assert_global_summary(summary)
    loss = F.cross_entropy(classifier(output), LABELS) + aux_loss
    loss.backward()
    handle.remove()
    assert torch.isfinite(loss)
    assert len(router_calls) >= 12  # six calls plus non-reentrant checkpoint recomputation
    router_grad = model.router.net[3].weight.grad
    assert router_grad is not None and torch.isfinite(router_grad).all() and float(router_grad.norm()) > 0
    expert_grads = [p.grad for p in model.filter_bank.experts.parameters() if p.grad is not None]
    assert expert_grads and all(torch.isfinite(grad).all() for grad in expert_grads)
    assert any(float(grad.norm()) > 0 for grad in expert_grads)

    empty = torch.empty((2, 0), dtype=torch.long)
    model.zero_grad(set_to_none=True)
    model.semantic_text_edge_index = empty
    model.semantic_text_weight = torch.empty(0)
    model.semantic_visual_edge_index = empty
    model.semantic_visual_weight = torch.empty(0)
    model.eval()
    with torch.no_grad():
        empty_output, _, _, empty_aux, _ = model(X, empty, return_details=True)
    assert torch.isfinite(empty_output).all() and torch.isfinite(empty_aux)
