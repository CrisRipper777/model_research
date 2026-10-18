from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.run_hybrid_cge_paper_evidence_v1 import (
    NO_MOE_PARAMETER_PREFIXES,
    module_hash,
)
from src.models import build_model
from src.models.rmcge_v2 import build_modality_knn_operators
from src.tasks.nc import _training_class_weights


def _make_model(condition: str, seed: int = 17):
    cfg = OmegaConf.create({"model": OmegaConf.load("configs/model/hybrid_cge_paper_v1.yaml"), "seed": seed})
    cfg.model.paper_condition = condition
    if condition in {"H2_physical_3hop", "F4_fashion_class_balanced_3hop"}:
        cfg.model.physical_hops = 3
    torch.manual_seed(seed)
    model = build_model(
        cfg,
        {"input_dim": 7, "text_dim": 3, "visual_dim": 4, "num_nodes": 6, "num_classes": 3},
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    unique = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 1], [1, 0], [1, 1]], dtype=torch.uint8)
    operators = build_modality_knn_operators(unique, flags, 6)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index_v2", operators[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight_v2", operators[modality]["weight"])
    model.semantic_candidate_fingerprint = "a" * 64
    model.semantic_v7b_fingerprint = "b" * 64
    return model, edge_index


def _train_loss(model, edge_index):
    x = torch.randn(6, 7)
    model.train()
    z, _, _, aux, info = model(x, edge_index)
    head = nn.Linear(256, 3)
    train_idx = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 1, 2, 1, 0, -100])
    loss = nn.functional.cross_entropy(head(z[train_idx]), labels[train_idx]) + aux
    if getattr(model, "supports_unimodal_aux_loss", False):
        from src.tasks.nc import _compose_full_graph_loss

        loss, _ = _compose_full_graph_loss(
            model=model, info=info, main_ce=loss - aux, aux_loss=aux, aux_weight=1.0,
            unimodal_weight=0.3, labels=labels, train_idx=train_idx,
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z).all()
    return z, info


def test_ablation_and_hop_models_keep_matching_initial_public_parameters():
    conditions = (
        "A1_no_early_physical_cross", "A2_no_late_completion", "A3_no_moe",
        "A4_no_same_node", "H2_physical_3hop",
    )
    models = []
    heads = []
    for condition in conditions:
        model, _ = _make_model(condition, 55)
        torch.manual_seed(55)
        head = nn.Linear(256, 3)
        models.append(model)
        heads.append(head)
    for head in heads[1:]:
        assert torch.equal(head.weight, heads[0].weight)
        assert torch.equal(head.bias, heads[0].bias)
    reference = dict(models[0].state_dict())
    for model in models[1:]:
        for name, value in model.state_dict().items():
            if name not in reference:
                continue
            if name == "alpha_raw" and value.shape != reference[name].shape:
                torch.testing.assert_close(value, reference[name][:, :3], atol=0, rtol=0)
            else:
                assert torch.equal(value, reference[name]), name
    assert models[-1].alpha_raw.shape == (4, 3)
    assert models[-1].physical_hops == 3
    paired_hashes = {
        module_hash(model, excluded_prefixes=NO_MOE_PARAMETER_PREFIXES)
        for model in models[:4]
    }
    assert len(paired_hashes) == 1


def test_a1_skips_early_physical_cross_and_duplicate_collaborative_route(monkeypatch):
    model, edges = _make_model("A1_no_early_physical_cross")
    calls = {"physical_cross": 0, "node_correction": 0, "expert": [0, 0, 0, 0]}
    original_correction = model._node_correction

    def cross_spy(*args, **kwargs):
        calls["physical_cross"] += 1
        raise AssertionError("early physical cross operator was called")

    def correction_spy(*args, **kwargs):
        calls["node_correction"] += 1
        return original_correction(*args, **kwargs)

    monkeypatch.setattr(model, "_physical_cross_message", cross_spy)
    monkeypatch.setattr(model, "_node_correction", correction_spy)
    handles = [expert.register_forward_hook(
        lambda module, inputs, output, i=i: calls["expert"].__setitem__(i, calls["expert"][i] + 1)
    ) for i, expert in enumerate(model.experts)]
    model.train()
    z, _, _, aux, info = model(torch.randn(6, 7), edges)
    for handle in handles:
        handle.remove()
    # Across both modalities, each expert is called by two private physical,
    # two same-source, and two late-completion paths. The additional two calls
    # in full H2 would be the duplicated collaborative physical expert pass.
    assert calls == {"physical_cross": 0, "node_correction": 2, "expert": [6, 6, 6, 6]}
    assert info["early_physical_cross_executed"] is False
    loss = z[:4].square().mean() + aux
    loss.backward()
    assert model.cross_projections["visual_to_text"][0].weight.grad is not None
    assert model.cross_projections["visual_to_text"][0].weight.grad.norm() > 0


def test_a2_skips_every_late_completion_module(monkeypatch):
    model, edges = _make_model("A2_no_late_completion")
    monkeypatch.setattr(model, "_late_completion", lambda *a, **k: (_ for _ in ()).throw(AssertionError("late path called")))
    z, info = _train_loss(model, edges)
    assert info["late_completion_executed"] is False
    for module in (model.query_projections, model.key_projections, model.source_selector,
                   model.channel_gate, model.interaction, model.comp_router):
        for parameter in module.parameters():
            assert parameter.grad is None
    assert z.shape == (6, 256)


def test_a3_removes_all_moe_parameters_and_uses_zero_balance_loss(monkeypatch):
    model, edges = _make_model("A3_no_moe")
    for name in ("experts", "selection_head", "strength_head", "node_router", "view_router", "comp_router"):
        assert not hasattr(model, name)
    names = {name for name, _ in model.named_parameters()}
    assert not any(token in name for name in names for token in ("expert", "router", "selection_head"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt_names = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert len(opt_names) == sum(len(group["params"]) for group in optimizer.param_groups)
    monkeypatch.setattr(model, "_expert_mixture", lambda *a, **k: (_ for _ in ()).throw(AssertionError("expert called")))
    z, _, _, aux_loss, info = model(torch.randn(6, 7), edges)
    loss = z[:4].square().mean() + aux_loss
    loss.backward()
    assert torch.isfinite(z).all()
    assert aux_loss.item() == 0.0
    assert info["moe_executed"] is False
    assert info["moe_router_parameters_present"] is False
    assert model.cross_projections["visual_to_text"][0].weight.grad is not None


def test_a4_skips_same_node_projection_and_retains_late_dropout():
    model, edges = _make_model("A4_no_same_node")
    calls = {"visual_to_text": 0, "text_to_visual": 0}
    handles = []
    for name in calls:
        handles.append(model.cross_projections[name].register_forward_pre_hook(
            lambda module, args, key=name: calls.__setitem__(key, calls[key] + 1)
        ))
    model.eval()
    with torch.no_grad():
        _, _, _, _, info = model(torch.randn(6, 7), edges, return_details=True)
    for handle in handles:
        handle.remove()
    assert calls == {"visual_to_text": 2, "text_to_visual": 2}
    assert model.cross_dropout_p == 0.15
    for modality in ("text", "visual"):
        late = info["details"]["modalities"][modality]["late_completion"]
        assert late["same_node_support_nodes"] == 0
        assert late["same_node_source_weight"] == 0.0


def test_three_hop_forward_never_runs_fourth_physical_propagation(monkeypatch):
    model, edges = _make_model("H2_physical_3hop")
    calls = 0
    original = model._propagate

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_propagate", spy)
    model.eval()
    with torch.no_grad():
        model(torch.randn(6, 7), edges)
    assert calls == 2 * 3
    assert model.alpha_raw.shape == (4, 3)
    assert len(model.experts) == 4 and model.top_k == 2


def test_main_class_weights_are_train_only_normalized_and_clipped():
    labels = torch.tensor([0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 2])
    train_idx = torch.tensor([0, 1, 2, 4, 5])
    weights_a = _training_class_weights(labels, train_idx, 3, enabled=True)
    labels[10] = 1  # outside train_idx: this cannot influence the weights
    weights_b = _training_class_weights(labels, train_idx, 3, enabled=True)
    torch.testing.assert_close(weights_a, weights_b)
    assert float(weights_a.min()) >= 0.5
    assert float(weights_a.max()) <= 2.0
    assert _training_class_weights(labels, train_idx, 3, enabled=False) is None


def test_private_auxiliary_ce_remains_unweighted_when_main_weights_are_used():
    from src.tasks.nc import _compose_full_graph_loss

    class SupportsPrivateAux:
        supports_unimodal_aux_loss = True

    labels = torch.tensor([0, 1, 1, 2, -100])
    train_idx = torch.tensor([0, 1, 2, 3])
    text_logits = torch.tensor([[2., 0., 0.], [0., 2., 0.], [0., 2., 0.], [0., 0., 2.], [0., 0., 0.]])
    visual_logits = text_logits.flip(1)
    info = {"unimodal_logits": {"text": text_logits, "visual": visual_logits}}
    main_weights = _training_class_weights(
        labels, train_idx, 3, enabled=True
    )
    main_logits = torch.zeros(4, 3)
    weighted_main_ce = nn.functional.cross_entropy(
        main_logits, labels[train_idx], weight=main_weights
    )
    aux_loss = torch.tensor(0.0)
    combined, observed_aux = _compose_full_graph_loss(
        model=SupportsPrivateAux(), info=info, main_ce=weighted_main_ce,
        aux_loss=aux_loss, aux_weight=1.0, unimodal_weight=0.5,
        labels=labels, train_idx=train_idx,
    )
    expected_aux = 0.5 * (
        nn.functional.cross_entropy(text_logits[train_idx], labels[train_idx])
        + nn.functional.cross_entropy(visual_logits[train_idx], labels[train_idx])
    )
    torch.testing.assert_close(observed_aux, expected_aux)
    torch.testing.assert_close(combined, weighted_main_ce + 0.5 * expected_aux)
