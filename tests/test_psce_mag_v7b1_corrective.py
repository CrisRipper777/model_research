from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b
from src.models.psce_mag_v7a import Model as V7AModel
from src.models.psce_mag_v7b import Model as V7BModel
from src.tasks.common import build_optimizer, scheduler_step


ROOT = Path(__file__).resolve().parents[1]
RELATION_PREFIXES = ("relation_projections.", "relation_scorer.")
ATOL = 2e-6
RTOL = 2e-5


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture()
def mini_graph(tmp_path):
    rng = np.random.default_rng(7001)
    n = 24
    text = rng.normal(size=(n, 12)).astype(np.float32)
    visual = rng.normal(size=(n, 9)).astype(np.float32)
    nodes = torch.arange(n)
    physical = torch.stack(
        (torch.cat((nodes, nodes)), torch.cat(((nodes + 1) % n, (nodes - 1) % n)))
    ).long()
    cache_path, metadata = build_semantic_candidate_cache_v7b(
        dataset="v7b1-parity-toy",
        text_features=text,
        visual_features=visual,
        physical_edge_index=physical,
        cache_dir=tmp_path,
        thread_count=1,
    )
    x = torch.from_numpy(np.concatenate((text, visual), axis=1))
    labels = (torch.arange(n) * 7 % 4).long()
    return {
        "x": x,
        "physical": physical,
        "labels": labels,
        "train_idx": torch.arange(18),
        "cache_path": cache_path,
        "metadata": metadata,
        "data_info": {
            "input_dim": 21,
            "text_dim": 12,
            "visual_dim": 9,
            "num_nodes": n,
            "num_classes": 4,
        },
    }


def _model_config(model_name: str, variant: str):
    model = OmegaConf.load(ROOT / f"configs/model/{model_name}.yaml")
    model.variant = variant
    # Keep CPU parity tests representative of the frozen NC dropout setting.
    model.dropout = 0.2
    model.relation_chunk_size = 19
    model.propagation_chunk_size = 23
    return OmegaConf.create({"model": model})


def _build_pair(graph, seed: int = 77):
    torch.manual_seed(seed)
    a2 = V7AModel(_model_config("psce_mag_v7a", "a2_conditional_dual"), graph["data_info"])
    head_a2 = torch.nn.Linear(a2.out_dim, graph["data_info"]["num_classes"])
    torch.manual_seed(seed)
    b0 = V7BModel(_model_config("psce_mag_v7b", "b0_v7a_original"), graph["data_info"])
    head_b0 = torch.nn.Linear(b0.out_dim, graph["data_info"]["num_classes"])
    for model in (a2, b0):
        model.load_semantic_cache(
            str(graph["cache_path"]),
            expected_dataset="v7b1-parity-toy",
            expected_feature_fingerprint=graph["metadata"]["feature_fingerprint"],
            expected_candidate_fingerprint=graph["metadata"]["candidate_fingerprint"],
            **(
                {"expected_v7b_fingerprint": graph["metadata"]["v7b_fingerprint"]}
                if isinstance(model, V7BModel)
                else {}
            ),
        )
    return a2, head_a2, b0, head_b0


def _task_cfg():
    return OmegaConf.create(
        {
            "model": {"lr": 1.0e-3, "weight_decay": 1.0e-4},
            "task": {
                "optimizer": "adamw",
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
                "grad_clip": 1.0,
                "scheduler": None,
                "scheduler_warmup_epochs": 10,
                "scheduler_min_lr": 1.0e-5,
            },
        }
    )


def _loss(model, head, graph):
    z, _, _, aux_loss, _ = model(graph["x"], graph["physical"])
    logits = head(z.index_select(0, graph["train_idx"]))
    ce = torch.nn.functional.cross_entropy(logits, graph["labels"][graph["train_idx"]])
    return z, logits, ce, aux_loss, ce + aux_loss


def _assert_module_grads_match(a2, b0, head_a2, head_b0):
    a2_params, b0_params = dict(a2.named_parameters()), dict(b0.named_parameters())
    assert a2_params.keys() == b0_params.keys()
    for name in a2_params:
        left, right = a2_params[name].grad, b0_params[name].grad
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL, msg=name)
    for name in head_a2.state_dict():
        torch.testing.assert_close(
            head_a2.weight.grad if name == "weight" else head_a2.bias.grad,
            head_b0.weight.grad if name == "weight" else head_b0.bias.grad,
            atol=ATOL,
            rtol=RTOL,
            msg=f"classifier.{name}",
        )


def _max_parameter_difference(a2, b0, head_a2=None, head_b0=None):
    differences = [
        float((dict(a2.named_parameters())[name] - dict(b0.named_parameters())[name]).abs().max())
        for name in dict(a2.named_parameters())
    ]
    if head_a2 is not None:
        differences.extend(
            float((left - right).abs().max())
            for left, right in zip(head_a2.parameters(), head_b0.parameters())
        )
    return max(differences, default=0.0)


def test_b0_parameter_permissions_match_v7a_and_parameter_delta_is_exact(mini_graph):
    a2, _, b0, _ = _build_pair(mini_graph)
    a2_params, b0_params = dict(a2.named_parameters()), dict(b0.named_parameters())
    assert a2_params.keys() == b0_params.keys()
    for name in a2_params:
        assert a2_params[name].shape == b0_params[name].shape, name
        assert a2_params[name].requires_grad == b0_params[name].requires_grad, name
    relation_names = [name for name in b0_params if name.startswith(RELATION_PREFIXES)]
    assert relation_names
    assert all(b0_params[name].requires_grad for name in relation_names)
    assert all(a2_params[name].requires_grad for name in relation_names)

    old_b0_count = sum(parameter.numel() for parameter in b0.parameters() if parameter.requires_grad)
    for module in (b0.relation_projections, b0.relation_scorer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    simulated_old_count = sum(
        parameter.numel() for parameter in b0.parameters() if parameter.requires_grad
    )
    restored_count = sum(parameter.numel() for parameter in a2.parameters() if parameter.requires_grad)
    relation_parameter_count = sum(
        parameter.numel()
        for name, parameter in a2.named_parameters()
        if name.startswith(RELATION_PREFIXES)
    )
    assert old_b0_count - simulated_old_count == relation_parameter_count == 24_705
    assert restored_count - simulated_old_count == 24_705


@pytest.mark.parametrize("variant", ("b1_fixed_knn", "b2_random_fixed"))
def test_fixed_weight_variants_stay_frozen_and_relative_heads_stay_trainable(variant, mini_graph):
    cfg = _model_config("psce_mag_v7b", variant)
    model = V7BModel(cfg, mini_graph["data_info"])
    assert all(not p.requires_grad for p in model.relation_projections.parameters())
    assert all(not p.requires_grad for p in model.relation_scorer.parameters())
    for relative in ("b3_relative_shared", "b4_relative_modality"):
        relative_model = V7BModel(
            _model_config("psce_mag_v7b", relative), mini_graph["data_info"]
        )
        assert all(p.requires_grad for p in relative_model.relation_projections.parameters())
        assert all(p.requires_grad for p in relative_model.relation_scorer.parameters())


def test_same_seed_initialization_includes_classifier_and_common_parameters(mini_graph):
    a2, head_a2, b0, head_b0 = _build_pair(mini_graph, seed=918)
    a2_state, b0_state = a2.state_dict(), b0.state_dict()
    assert a2_state.keys() == b0_state.keys()
    for name in a2_state:
        assert torch.equal(a2_state[name], b0_state[name]), name
    for name in head_a2.state_dict():
        assert torch.equal(head_a2.state_dict()[name], head_b0.state_dict()[name]), name


def test_training_forward_and_gradients_match_with_nonzero_relation_gradients(mini_graph):
    a2, head_a2, b0, head_b0 = _build_pair(mini_graph, seed=44)
    a2.train(); head_a2.train(); b0.train(); head_b0.train()
    rng_state = torch.random.get_rng_state()
    old = _loss(a2, head_a2, mini_graph)
    torch.random.set_rng_state(rng_state)
    new = _loss(b0, head_b0, mini_graph)
    for label, left, right in zip(
        ("embedding", "logits", "CE", "auxiliary", "total"), old, new
    ):
        torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL, msg=label)
    old[-1].backward()
    torch.random.set_rng_state(rng_state)
    # Re-run the B0 forward from the same dropout stream before backward.
    b0.zero_grad(set_to_none=True); head_b0.zero_grad(set_to_none=True)
    new = _loss(b0, head_b0, mini_graph)
    new[-1].backward()
    _assert_module_grads_match(a2, b0, head_a2, head_b0)

    for name in (
        "relation_projections.text.weight",
        "relation_projections.visual.weight",
        "relation_scorer.0.weight",
        "relation_scorer.0.bias",
        "relation_scorer.2.weight",
        "relation_scorer.2.bias",
        "experts.0.linear1.weight",
        "alpha_raw",
        "semantic_coefficients_raw",
        "semantic_selection_head.weight",
    ):
        gradient = dict(b0.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient).item() > 0, name
    assert head_b0.weight.grad is not None and torch.count_nonzero(head_b0.weight.grad) > 0


def test_adamw_clipping_and_multistep_trajectory_match(mini_graph):
    a2, head_a2, b0, head_b0 = _build_pair(mini_graph, seed=72)
    cfg = _task_cfg()
    params_a2 = list(a2.parameters()) + list(head_a2.parameters())
    params_b0 = list(b0.parameters()) + list(head_b0.parameters())
    optimizer_a2 = build_optimizer(params_a2, cfg, model=a2)
    optimizer_b0 = build_optimizer(params_b0, cfg, model=b0)
    assert len(optimizer_a2.param_groups) == len(optimizer_b0.param_groups)
    for left, right in zip(optimizer_a2.param_groups, optimizer_b0.param_groups):
        assert {k: v for k, v in left.items() if k != "params"} == {
            k: v for k, v in right.items() if k != "params"
        }
    names_a2 = {id(parameter): name for name, parameter in a2.named_parameters()}
    names_b0 = {id(parameter): name for name, parameter in b0.named_parameters()}
    names_a2.update({id(parameter): f"classifier.{name}" for name, parameter in head_a2.named_parameters()})
    names_b0.update({id(parameter): f"classifier.{name}" for name, parameter in head_b0.named_parameters()})
    groups_a2 = [[names_a2[id(parameter)] for parameter in group["params"]] for group in optimizer_a2.param_groups]
    groups_b0 = [[names_b0[id(parameter)] for parameter in group["params"]] for group in optimizer_b0.param_groups]
    assert groups_a2 == groups_b0

    relation_before = {
        name: parameter.detach().clone()
        for name, parameter in b0.named_parameters()
        if name.startswith(RELATION_PREFIXES)
    }
    losses_a2, losses_b0 = [], []
    for step in range(1, 4):
        a2.train(); head_a2.train(); b0.train(); head_b0.train()
        optimizer_a2.zero_grad(set_to_none=True)
        optimizer_b0.zero_grad(set_to_none=True)
        state = torch.Generator().manual_seed(700 + step).get_state()
        torch.random.set_rng_state(state)
        out_a2 = _loss(a2, head_a2, mini_graph)
        torch.random.set_rng_state(state)
        out_b0 = _loss(b0, head_b0, mini_graph)
        torch.testing.assert_close(out_a2[1], out_b0[1], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(out_a2[-1], out_b0[-1], atol=ATOL, rtol=RTOL)
        losses_a2.append(float(out_a2[-1].detach()))
        losses_b0.append(float(out_b0[-1].detach()))
        out_a2[-1].backward()
        out_b0[-1].backward()
        _assert_module_grads_match(a2, b0, head_a2, head_b0)
        torch.nn.utils.clip_grad_norm_(params_a2, 1.0, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(params_b0, 1.0, error_if_nonfinite=True)
        scheduler_step(cfg, optimizer_a2, step, 3)
        scheduler_step(cfg, optimizer_b0, step, 3)
        optimizer_a2.step()
        optimizer_b0.step()
        assert _max_parameter_difference(a2, b0, head_a2, head_b0) <= 1e-7

        named_a2 = dict(a2.named_parameters())
        named_b0 = dict(b0.named_parameters())
        for name in named_a2:
            state_a = optimizer_a2.state.get(named_a2[name], {})
            state_b = optimizer_b0.state.get(named_b0[name], {})
            assert state_a.keys() == state_b.keys(), name
            for key in state_a:
                left, right = state_a[key], state_b[key]
                if torch.is_tensor(left):
                    torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL, msg=f"{name}.{key}")
                else:
                    assert left == right, f"{name}.{key}"

    assert losses_a2 == pytest.approx(losses_b0, abs=ATOL, rel=RTOL)
    named_b0 = dict(b0.named_parameters())
    for name, before in relation_before.items():
        assert torch.count_nonzero(named_b0[name].detach() - before).item() > 0, name
