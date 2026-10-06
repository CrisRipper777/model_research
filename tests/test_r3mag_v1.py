from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.analysis.r3mag_h1_response_audit import assert_disjoint_splits, stratified_internal_partition
from src.experiments.r3mag_v1_prototype import (
    _assert_common_initialization,
    _fit_variant,
    build_stage2_optimizer,
    labels_for_protocol,
)
from src.models import build_model
from src.models.r3mag_v1 import Model, ReLUDropout, ResponseAtomBank, clip_unit
from src.utils.graph_ops import normalized_adjacency_operator
from src.utils.seeds import set_seed


def _toy_inputs():
    torch.manual_seed(17)
    x_t = torch.randn(8, 5)
    x_i = torch.randn(8, 3)
    x = torch.cat([x_t, x_i], dim=-1)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5, 6], [1, 0, 2, 1, 3, 2, 5, 4, 7]], dtype=torch.long
    )
    data = SimpleNamespace(x=x, x_t=x_t, x_i=x_i, edge_index=edge_index, num_nodes=8, num_classes=4)
    cfg = OmegaConf.create({"model": {"name": "r3mag_v1"}})
    info = {"input_dim": 8, "num_nodes": 8, "num_classes": 4, "text_dim": 5, "visual_dim": 3}
    return data, cfg, info


def _model():
    _, cfg, info = _toy_inputs()
    return build_model(cfg, info)


def test_factory_feature_split_matches_data_text_visual_convention():
    data, cfg, info = _toy_inputs()
    model = build_model(cfg, info)
    text, visual = model._split_features(data.x)
    assert torch.equal(text, data.x_t)
    assert torch.equal(visual, data.x_i)
    assert torch.equal(torch.cat([text, visual], dim=-1), data.x)


def test_normalized_operator_finite_and_states_have_expected_shapes():
    data, _, _ = _toy_inputs()
    operator = normalized_adjacency_operator(data.edge_index, data.num_nodes)
    assert operator.is_sparse
    assert torch.isfinite(operator.values()).all()
    model = _model()
    model.eval()
    states = model.propagate_states(torch.randn(data.num_nodes, 128), operator, max_order=3)
    assert len(states) == 4
    assert all(state.shape == (data.num_nodes, 128) for state in states)


def test_response_atoms_are_unit_norm_and_dct_initialized_orthogonal():
    bank = ResponseAtomBank(3, 3)
    atoms = bank()
    assert torch.allclose(atoms.norm(dim=-1), torch.ones(3), atol=1e-7)
    assert torch.allclose(atoms @ atoms.T, torch.eye(3), atol=1e-6)
    assert not torch.equal(atoms[0], atoms[1])


@pytest.mark.parametrize("variant", ["G1", "G2", "G3", "G4"])
def test_adaptation_never_changes_order_zero_and_preserve_recovers_global(variant):
    data, _, _ = _toy_inputs()
    model = _model().eval()
    operator = normalized_adjacency_operator(data.edge_index, data.num_nodes)
    normal = model.forward_components(data.x, data.edge_index, operator=operator, variant=variant)
    preserve = model.forward_components(
        data.x, data.edge_index, operator=operator, variant=variant, intervention="preserve"
    )
    assert torch.equal(normal["delta_gamma_text"][:, 0], torch.zeros(data.num_nodes))
    assert torch.equal(normal["delta_gamma_visual"][:, 0], torch.zeros(data.num_nodes))
    assert torch.equal(normal["adapted_gamma_text"][:, 0], model.gamma_global_text[0].expand(data.num_nodes))
    assert torch.equal(normal["adapted_gamma_visual"][:, 0], model.gamma_global_visual[0].expand(data.num_nodes))
    assert torch.allclose(preserve["response_text"], preserve["global_response_text"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(preserve["response_visual"], preserve["global_response_visual"], atol=1e-6, rtol=1e-6)


def test_g0_normal_equals_preserve_exactly():
    data, _, _ = _toy_inputs()
    model = _model().eval()
    operator = normalized_adjacency_operator(data.edge_index, data.num_nodes)
    normal = model.forward_components(data.x, data.edge_index, operator=operator, variant="G0")
    preserve = model.forward_components(
        data.x, data.edge_index, operator=operator, variant="G0", intervention="preserve"
    )
    assert torch.equal(normal["z"], preserve["z"])
    assert torch.equal(normal["response_text"], preserve["response_text"])
    assert torch.equal(normal["response_visual"], preserve["response_visual"])


def test_shared_route_is_the_same_tensor_for_text_and_visual():
    data, _, _ = _toy_inputs()
    model = _model().eval()
    operator = normalized_adjacency_operator(data.edge_index, data.num_nodes)
    components = model.forward_components(data.x, data.edge_index, operator=operator, variant="G2")
    text_route = components["route_weights"]["shared_text"]
    visual_route = components["route_weights"]["shared_visual"]
    assert text_route is visual_route
    assert torch.equal(text_route, visual_route)


def test_detached_task_context_cannot_send_router_gradient_to_unimodal_logits():
    model = _model()
    logits_t = torch.randn(6, 4, requires_grad=True)
    logits_v = torch.randn(6, 4, requires_grad=True)
    context = model.task_context_from_logits(logits_t, logits_v)
    u = torch.randn(6, 64)
    router_input = torch.cat([
        u, context["p_text"], context["entropy_text"], context["margin_text"], context["js"]
    ], dim=-1)
    route = model.flat_router_text(router_input)
    grad_t, grad_v = torch.autograd.grad(route.square().sum(), (logits_t, logits_v), allow_unused=True)
    assert grad_t is None and grad_v is None
    assert all(not value.requires_grad for value in context.values())
    assert torch.isfinite(context["js"]).all()


def test_clip_unit_bounds_vectors_and_does_not_amplify_small_vectors():
    values = torch.tensor([[0.2, -0.3, 0.1], [6.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    clipped = clip_unit(values)
    assert torch.all(clipped.norm(dim=-1) <= 1.0 + 1e-7)
    assert torch.equal(clipped[0], values[0])
    assert torch.equal(clipped[2], values[2])
    assert torch.allclose(clipped[1].norm(), torch.tensor(1.0))


def test_fused_relu_dropout_matches_standard_forward_and_gradient():
    base_a = torch.randn(16, 8, requires_grad=True)
    base_b = base_a.detach().clone().requires_grad_(True)
    torch.manual_seed(91)
    out_a = torch.nn.functional.dropout(torch.relu(base_a), p=0.2, training=True, inplace=False)
    loss_a = out_a.square().sum()
    loss_a.backward()
    torch.manual_seed(91)
    out_b = ReLUDropout(0.2).train()(base_b)
    loss_b = out_b.square().sum()
    loss_b.backward()
    assert torch.equal(out_a, out_b)
    assert torch.allclose(base_a.grad, base_b.grad, atol=1e-7, rtol=1e-7)


def test_normal_uniform_preserve_and_shared_private_interventions_are_finite():
    data, _, _ = _toy_inputs()
    model = _model().eval()
    operator = normalized_adjacency_operator(data.edge_index, data.num_nodes)
    cases = [(variant, name) for variant in ("G1", "G2", "G3", "G4")
             for name in ("normal", "uniform_route", "preserve")]
    cases += [(variant, name) for variant in ("G2", "G3", "G4")
              for name in ("shared_off", "private_off")]
    for variant, intervention in cases:
        result = model.forward_components(
            data.x, data.edge_index, operator=operator, variant=variant, intervention=intervention
        )
        assert torch.isfinite(result["z"]).all()
        assert torch.isfinite(result["rho_text"]).all()
        assert torch.isfinite(result["rho_visual"]).all()


def test_stage2_variants_load_identical_stage1_model_and_classifier_state():
    data, _, info = _toy_inputs()
    stage1 = Model(None, info)
    classifier_stage1 = torch.nn.Linear(stage1.out_dim, data.num_classes)
    model_state = {key: value.detach().clone() for key, value in stage1.state_dict().items()}
    classifier_state = {key: value.detach().clone() for key, value in classifier_stage1.state_dict().items()}
    for variant in ("G0-FT", "G1", "G2", "G3", "G4"):
        model = Model(None, info)
        classifier = torch.nn.Linear(model.out_dim, data.num_classes)
        model.load_state_dict(model_state)
        classifier.load_state_dict(classifier_state)
        qa = _assert_common_initialization(model_state, classifier_state, model, classifier)
        assert qa["common_model_parameters_match_stage1"]
        assert qa["all_model_parameters_match_stage1"]
        assert qa["classifier_matches_stage1"]


def test_devtrain_audit_val_test_indices_are_disjoint_and_test_label_access_is_rejected():
    labels = torch.tensor([0, 1] * 50)
    train = torch.arange(60)
    parts, _ = stratified_internal_partition(train, labels, seed=20261006)
    dev_train = torch.cat([parts["host_train"], parts["response_train"]]).sort().values
    val, test = torch.arange(60, 80), torch.arange(80, 100)
    assert_disjoint_splits({"DevTrain": dev_train, "Audit": parts["audit"], "Val": val, "Test": test})
    assert set(dev_train.tolist()) | set(parts["audit"].tolist()) == set(train.tolist())
    fake_data = SimpleNamespace(y=labels)
    assert torch.equal(labels_for_protocol(fake_data, "dev_train", dev_train), labels[dev_train])
    with pytest.raises(ValueError, match="forbidden"):
        labels_for_protocol(fake_data, "test", test)


def test_stage2_optimizer_groups_have_no_omissions_or_duplicates_and_g0ft_has_no_new_group():
    data, _, info = _toy_inputs()
    for variant in ("G0-FT", "G1", "G2", "G3", "G4"):
        model = Model(None, info)
        classifier = torch.nn.Linear(model.out_dim, data.num_classes)
        optimizer, metadata = build_stage2_optimizer(model, classifier, variant)
        groups = model.optimizer_parameter_groups(variant)
        expected_ids = {id(p) for p in groups["base"].values()} | {id(p) for p in groups["new"].values()} | {
            id(p) for p in classifier.parameters()
        }
        actual_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(actual_ids) == len(set(actual_ids))
        assert set(actual_ids) == expected_ids
        assert "new" not in {item["name"] for item in metadata} if variant == "G0-FT" else "new" in {item["name"] for item in metadata}


def test_same_seed_stage1_smoke_is_deterministic_on_toy_graph():
    data, _, info = _toy_inputs()
    operator = normalized_adjacency_operator(data.edge_index, data.num_nodes)
    train_idx = torch.tensor([0, 1, 2, 3, 4, 5])
    train_labels = torch.tensor([0, 1, 2, 3, 0, 1])
    val_idx = torch.tensor([6, 7])
    val_labels = torch.tensor([2, 3])
    results = []
    for _ in range(2):
        set_seed(77)
        model = Model(None, info)
        classifier = torch.nn.Linear(model.out_dim, data.num_classes)
        result = _fit_variant(
            model=model, classifier=classifier, variant="G0", x=data.x, edge_index=data.edge_index,
            operator=operator, train_idx=train_idx, train_labels=train_labels,
            val_idx=val_idx, val_labels=val_labels, num_classes=data.num_classes,
            max_epochs=2, patience=2, min_epoch=1, stage="stage1",
        )
        results.append(result)
    assert results[0]["best_epoch"] == results[1]["best_epoch"]
    assert results[0]["val_metrics"] == results[1]["val_metrics"]
    assert all(torch.equal(results[0]["model_state"][key], results[1]["model_state"][key])
               for key in results[0]["model_state"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA activation-offload QA requires a GPU")
def test_cuda_training_with_saved_activation_offload():
    data, _, info = _toy_inputs()
    device = torch.device("cuda:0")
    model = Model(None, info).to(device)
    classifier = torch.nn.Linear(model.out_dim, data.num_classes).to(device)
    edge_index = data.edge_index.to(device)
    operator = normalized_adjacency_operator(edge_index, data.num_nodes, device=device)
    result = _fit_variant(
        model=model, classifier=classifier, variant="G0", x=data.x.to(device),
        edge_index=edge_index, operator=operator,
        train_idx=torch.tensor([0, 1, 2, 3, 4, 5], device=device),
        train_labels=torch.tensor([0, 1, 2, 3, 0, 1], device=device),
        val_idx=torch.tensor([6, 7], device=device),
        val_labels=torch.tensor([2, 3], device=device), num_classes=data.num_classes,
        max_epochs=1, patience=1, min_epoch=1, stage="stage1",
        offload_saved_activations=True,
    )
    assert torch.isfinite(torch.tensor(result["val_metrics"]["ce"]))


def test_graph_model_forward_matches_repo_five_item_api():
    data, _, _ = _toy_inputs()
    model = _model().eval()
    output = model(data.x, data.edge_index)
    assert len(output) == 5
    assert output[0].shape == (data.num_nodes, 128)
    assert output[3].ndim == 0
    assert isinstance(output[4], dict)
