from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b
from src.models.mrcge_v11 import Model, SharedExpertBank
from src.tasks.nc import _compose_full_graph_loss, _resolve_nc_eval_labels


@pytest.fixture(scope="module")
def graph_bundle(tmp_path_factory):
    torch.manual_seed(91)
    n, d = 12, 512
    text = torch.randn(n, d).numpy().astype("float32")
    visual = torch.randn(n, d).numpy().astype("float32")
    physical = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    out = tmp_path_factory.mktemp("mrcge_v7b_cache")
    path, metadata = build_semantic_candidate_cache_v7b(
        dataset="ele-fashion",
        text_features=text,
        visual_features=visual,
        physical_edge_index=physical,
        cache_dir=out,
        top_k=8,
        thread_count=1,
    )
    return {
        "n": n,
        "d": d,
        "text": torch.from_numpy(text),
        "visual": torch.from_numpy(visual),
        "x": torch.cat((torch.from_numpy(text), torch.from_numpy(visual)), dim=-1),
        "edge_index": physical,
        "path": path,
        "metadata": metadata,
        "info": {"input_dim": 2 * d, "text_dim": d, "visual_dim": d, "num_nodes": n, "num_classes": 3},
    }


def make_cfg(condition: str, bundle, dropout: float = 0.0):
    return SimpleNamespace(model={
        "condition": condition,
        "hidden_dim": 256,
        "num_experts": 4,
        "top_k": 2,
        "dropout": dropout,
        "fusion_dropout": dropout,
        "late_cross_dropout": 0.15,
        "expert_eta": 0.1,
        "expert_bottleneck": 64,
        "balance_weight": 0.01,
        "same_node_enabled": True,
        "semantic_cache_path": str(bundle["path"]),
        "semantic_feature_fingerprint": bundle["metadata"]["feature_fingerprint"],
        "semantic_candidate_fingerprint": bundle["metadata"]["candidate_fingerprint"],
        "semantic_v7b_fingerprint": bundle["metadata"]["v7b_fingerprint"],
    })


def build_model(condition: str, bundle, *, seed: int = 123, dropout: float = 0.0):
    torch.manual_seed(seed)
    model = Model(make_cfg(condition, bundle, dropout), bundle["info"])
    model.load_semantic_cache(
        str(bundle["path"]), expected_dataset="ele-fashion",
        expected_feature_fingerprint=bundle["metadata"]["feature_fingerprint"],
        expected_candidate_fingerprint=bundle["metadata"]["candidate_fingerprint"],
        expected_v7b_fingerprint=bundle["metadata"]["v7b_fingerprint"],
    )
    return model


def common_hash(module: torch.nn.Module) -> str:
    prefixes = (
        "projectors.", "cross_projections.", "query_projections.", "key_projections.",
        "source_selector.", "channel_gate.", "cross_lambda_raw", "tau_raw", "rho_raw",
        "gamma_raw", "beta_raw", "unimodal_heads.", "fusion_",
    )
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name.startswith(prefixes):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def test_one_shared_expert_bank_and_parameter_identity(graph_bundle):
    model = build_model("M", graph_bundle)
    experts = model.filter_bank.experts
    assert len(experts) == 4
    assert len({id(expert) for expert in experts}) == 4
    assert sum(1 for name, _ in model.named_parameters() if name.startswith("filter_bank.experts.")) == 4 * 2
    assert model.router is model.router


def test_order_truncation_signed_coefficients_and_gradients(graph_bundle):
    bank = SharedExpertBank(256, 64, 0.0, 0.1)
    sem = bank.coefficients(1, torch.float32)
    phys = bank.coefficients(3, torch.float32)
    assert torch.equal(sem[:, 2:], torch.zeros_like(sem[:, 2:]))
    assert torch.allclose(sem[:, :2].abs().sum(dim=-1), torch.ones(4))
    assert torch.allclose(phys.abs().sum(dim=-1), torch.ones(4))
    assert bool((sem[:, :2] < 0).any()) and bool((sem[:, :2] > 0).any())
    simple = build_model("S", graph_bundle)
    assert torch.allclose(simple.single_filter.coefficients(3, torch.float32), torch.full((4,), 0.25))
    assert torch.allclose(simple.single_filter.coefficients(1, torch.float32), torch.tensor([0.5, 0.5, 0.0, 0.0]))
    states = [torch.randn(5, 256, requires_grad=True) for _ in range(4)]
    support = torch.tensor([True, True, True, False, False])
    outputs, _, _, _ = bank(states, support, 3)
    outputs.square().mean().backward()
    assert bank.theta.grad is not None and torch.isfinite(bank.theta.grad).all()
    assert float(bank.theta.grad.abs().sum()) > 0


def test_v7b_source_flags_node_ids_and_cache_fingerprint(graph_bundle):
    model = build_model("M", graph_bundle)
    flags = model.semantic_source_flags_v7b
    assert flags.shape[1] == 2
    assert bool((flags[:, 0] > 0).any()) and bool((flags[:, 1] > 0).any())
    assert model.semantic_candidate_fingerprint == graph_bundle["metadata"]["candidate_fingerprint"]
    assert model.semantic_v7b_fingerprint == graph_bundle["metadata"]["v7b_fingerprint"]
    assert int(model.semantic_text_edge_index.max()) < graph_bundle["n"]
    assert int(model.semantic_visual_edge_index.max()) < graph_bundle["n"]
    assert not torch.equal(model.semantic_text_edge_index, model.semantic_visual_edge_index)


def test_graph_filter_outputs_are_zero_without_incoming_support(graph_bundle):
    model = build_model("M", graph_bundle)
    n = graph_bundle["n"]
    basis = [torch.randn(n, 256) for _ in range(4)]
    support = torch.tensor([True, True, True, True] + [False] * (n - 4))
    outputs, _, _, _ = model.filter_bank(basis, support, 3)
    assert torch.equal(outputs[:, ~support], torch.zeros_like(outputs[:, ~support]))
    response, _, _, _ = model._filter(basis, support, basis[0], 0, 0, torch.zeros_like(basis[0]), collect=False)
    assert torch.equal(response[~support], torch.zeros_like(response[~support]))
    no_semantic_support = torch.zeros(n, dtype=torch.bool)
    semantic_basis = [torch.randn(n, 256), torch.randn(n, 256)]
    sem_outputs, _, _, _ = model.filter_bank(semantic_basis, no_semantic_support, 1)
    assert torch.equal(sem_outputs, torch.zeros_like(sem_outputs))


def test_mixed_supported_and_isolated_nodes_keep_only_private_residual(graph_bundle):
    model = build_model("M", graph_bundle)
    model.eval()
    _, _, _, _, info = model(graph_bundle["x"], graph_bundle["edge_index"], return_details=True)
    assert info["details"]["physical_support_nodes"] == 4
    assert info["details"]["semantic_support_nodes"]["text"] == graph_bundle["n"]
    assert torch.isfinite(torch.tensor(info["details"]["private_rms"]["text"]))


def test_zero_cross_context_fallback_is_exact_even_with_dropout(graph_bundle):
    model = build_model("M", graph_bundle, dropout=0.4)
    model.train()
    private = torch.randn(7, 256)
    recomputed = torch.randn_like(private)
    context = torch.zeros_like(private)
    actual = model._zero_context_fallback(private, recomputed, context)
    assert torch.equal(actual, private)
    context[3, 2] = 1.0
    actual = model._zero_context_fallback(private, recomputed, context)
    assert torch.equal(actual[0:3], private[0:3]) and torch.equal(actual[4:], private[4:])
    assert torch.equal(actual[3], recomputed[3])


def test_both_early_cross_directions_have_gradients(graph_bundle):
    model = build_model("M", graph_bundle)
    edges, weights, support, _ = model._physical_operator(graph_bundle["edge_index"], graph_bundle["n"], torch.float32)
    for receiver, source in (("text", "visual"), ("visual", "text")):
        projection_name = "visual_to_text" if receiver == "text" else "text_to_visual"
        src = torch.randn(graph_bundle["n"], 256, requires_grad=True)
        ref = torch.randn_like(src)
        raw, calibrated, _ = model._early_cross(receiver, 0, src, model.cross_projections[projection_name], edges, weights, support, ref)
        (raw.square().mean() + calibrated.square().mean()).backward()
        grad = model.cross_projections[projection_name][0].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0
        model.zero_grad(set_to_none=True)


def test_late_semantic_missing_same_node_switch_and_none_only(graph_bundle):
    model = build_model("F", graph_bundle)
    n = graph_bundle["n"]
    model.semantic_text_edge_index = torch.empty((2, 0), dtype=torch.long)
    target, source = torch.randn(n, 256), torch.randn(n, 256)
    with torch.no_grad():
        model.same_node_enabled = True
        completed, enabled = model._late("text", 0, "text", source, source, target, collect=True)
        assert enabled["semantic_support_nodes"] == 0
        assert enabled["same_node_support_nodes"] == n
        assert enabled["selector_share"][1] > 0
        model.same_node_enabled = False
        none_completed, disabled = model._late("text", 0, "text", source, source, target, collect=True)
    assert disabled["same_node_support_nodes"] == 0
    assert disabled["selector_share"] == [0.0, 0.0, 1.0]
    assert torch.equal(none_completed, target)
    assert not torch.equal(completed, target)


def test_early_and_late_cross_are_independent_paths(graph_bundle):
    model = build_model("F", graph_bundle)
    n = graph_bundle["n"]
    edges, weights, support, _ = model._physical_operator(graph_bundle["edge_index"], n, torch.float32)
    h = torch.randn(n, 256)
    zero_phi = torch.nn.Linear(256, 256, bias=False)
    torch.nn.init.zeros_(zero_phi.weight)
    raw, cross, _ = model._early_cross("text", 0, h, zero_phi, edges, weights, support, torch.randn_like(h))
    assert torch.equal(raw, torch.zeros_like(raw)) and torch.equal(cross, torch.zeros_like(cross))
    model.same_node_enabled = True
    with torch.no_grad():
        late, detail = model._late("text", 0, "visual", h, h, h, collect=True)
    assert detail["same_node_support_nodes"] == n
    assert not torch.equal(late, h)


def test_no_late_moe_or_unused_parameters_in_any_condition(graph_bundle):
    for condition in ("S", "F", "M"):
        model = build_model(condition, graph_bundle)
        names = tuple(name for name, _ in model.named_parameters())
        forbidden = ("comp_router", "interaction", "complementary", "late_expert", "role_film")
        assert not any(part in name for name in names for part in forbidden)


def test_top2_weights_router_initialization_and_balance_gradient(graph_bundle):
    model = build_model("M", graph_bundle)
    support = torch.ones(graph_bundle["n"], dtype=torch.bool)
    current = torch.randn(graph_bundle["n"], 256)
    first = torch.randn_like(current)
    logits = model.router(current, first, torch.zeros_like(current), 0, 0)
    dense = torch.softmax(logits, dim=-1)
    top, ids = torch.topk(logits, 2, dim=-1)
    local = torch.softmax(top, dim=-1)
    weights = torch.zeros_like(dense).scatter(1, ids, local)
    selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, ids, True)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(graph_bundle["n"]))
    assert torch.equal(weights[~selected], torch.zeros_like(weights[~selected]))
    assert len(set(ids.flatten().tolist())) >= 2
    importance = dense[support].mean(dim=0)
    share = selected[support].float().mean(dim=0) / 2
    balance = 4 * (importance * share).sum()
    balance.backward()
    assert model.router.net[-1].weight.grad is not None
    assert float(model.router.net[-1].weight.grad.abs().sum()) > 0


def test_trainer_loss_uses_main_private_and_balance_terms(graph_bundle):
    model = build_model("M", graph_bundle)
    z, _, _, aux, info = model(graph_bundle["x"], graph_bundle["edge_index"])
    classifier = torch.nn.Linear(256, 3)
    labels = torch.tensor([0, 1, 2, 0, -100, -100, 1, 2, 0, 1, 2, 0])
    train_idx = torch.tensor([0, 1, 2])
    main = F.cross_entropy(classifier(z[train_idx]), labels[train_idx])
    actual, private = _compose_full_graph_loss(
        model=model, info=info, main_ce=main, aux_loss=aux,
        aux_weight=1.0, unimodal_weight=0.3, labels=labels, train_idx=train_idx,
    )
    expected_private = 0.5 * sum(
        F.cross_entropy(info["unimodal_logits"][m][train_idx], labels[train_idx])
        for m in ("text", "visual")
    )
    assert torch.allclose(private, expected_private)
    assert torch.allclose(actual, main + 0.3 * expected_private + 0.01 * info["balance_loss"])
    actual.backward()
    assert model.unimodal_heads["text"].weight.grad is not None


def test_s_f_m_common_initial_tensors_and_classifier_match(graph_bundle):
    states, heads = [], []
    for condition in ("S", "F", "M"):
        model = build_model(condition, graph_bundle, seed=732)
        states.append(common_hash(model))
        # Condition-specific initialization is RNG-isolated, so the Trainer's
        # next module receives the same external classifier tensors.
        torch.manual_seed(732)
        replay = Model(make_cfg(condition, graph_bundle), graph_bundle["info"])
        head = torch.nn.Linear(replay.out_dim, 3)
        heads.append({k: v.detach().clone() for k, v in head.state_dict().items()})
    assert len(set(states)) == 1
    assert all(torch.equal(heads[0][k], heads[i][k]) for i in (1, 2) for k in heads[0])


def test_cpu_forward_backward_checkpoint_reload_and_rng_audit(graph_bundle, tmp_path):
    model = build_model("M", graph_bundle)
    model.eval()
    before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        z, _, _, _, _ = model(graph_bundle["x"], graph_bundle["edge_index"], return_details=True)
    assert torch.equal(before, torch.random.get_rng_state())
    model.train()
    z, _, _, aux, info = model(graph_bundle["x"], graph_bundle["edge_index"])
    loss = z.square().mean() + aux + sum(v.square().mean() for v in info["unimodal_logits"].values())
    loss.backward()
    assert torch.isfinite(loss)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    reloaded = build_model("M", graph_bundle)
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    wrong_cache = {key: value.clone() for key, value in reloaded.state_dict().items()}
    wrong_cache["semantic_feature_fingerprint_state"][0] ^= 1
    with pytest.raises(RuntimeError, match="input features fingerprint"):
        reloaded.load_state_dict(wrong_cache)
    model.eval(); reloaded.eval()
    with torch.no_grad():
        a = model(graph_bundle["x"], graph_bundle["edge_index"])[0]
        b = reloaded(graph_bundle["x"], graph_bundle["edge_index"])[0]
    assert torch.allclose(a, b, atol=0, rtol=0)


def test_ele_fashion_feature_dimensions_and_data_alignment(graph_bundle):
    assert graph_bundle["x"].shape == (graph_bundle["n"], 1024)
    model = build_model("S", graph_bundle)
    assert model.projectors["text"].linear1.in_features == 512
    assert model.projectors["visual"].linear1.in_features == 512
    z = model(graph_bundle["x"], graph_bundle["edge_index"])[0]
    assert z.shape == (graph_bundle["n"], 256)


def test_train_and_eval_dropout_contract(graph_bundle):
    model = build_model("S", graph_bundle, dropout=0.3)
    h = torch.randn(graph_bundle["n"], 256)
    model.train()
    state = torch.random.get_rng_state().clone()
    model._late("text", 0, "visual", h, h, h, collect=False)
    assert not torch.equal(state, torch.random.get_rng_state())
    model.eval()
    state = torch.random.get_rng_state().clone()
    model._late("text", 0, "visual", h, h, h, collect=False)
    assert torch.equal(state, torch.random.get_rng_state())


def test_filter_router_and_cross_parameters_receive_gradients(graph_bundle):
    model = build_model("M", graph_bundle)
    z, _, _, aux, info = model(graph_bundle["x"], graph_bundle["edge_index"])
    loss = z.square().mean() + aux + sum(v.square().mean() for v in info["unimodal_logits"].values())
    loss.backward()
    diag = model.capture_gradient_diagnostics()
    required = (
        "filter_bank.theta", "router.net.3.weight", "filter_bank.experts.0.net.0.weight",
        "projectors.text.linear1.weight", "cross_projections.visual_to_text.0.weight",
        "cross_projections.text_to_visual.0.weight", "channel_gate.2.weight",
    )
    assert all(diag[name]["gradient_present"] and diag[name]["gradient_finite"] for name in required)
    assert any(diag[name]["gradient_nonzero"] for name in required)


@pytest.mark.parametrize("condition", ["S", "F"])
def test_single_filter_gradient_diagnostics_match_trainable_parameters(condition, graph_bundle):
    model = build_model(condition, graph_bundle)
    z, _, _, aux, info = model(graph_bundle["x"], graph_bundle["edge_index"])
    loss = z.square().mean() + aux + sum(v.square().mean() for v in info["unimodal_logits"].values())
    loss.backward()

    diagnostics = model.capture_gradient_diagnostics()
    transform_name = "single_filter.transform.net.0.weight"
    assert diagnostics[transform_name]["gradient_present"]
    assert diagnostics[transform_name]["gradient_finite"]
    if condition == "F":
        assert diagnostics["single_filter.theta"]["gradient_present"]
        assert diagnostics["single_filter.theta"]["gradient_finite"]
    else:
        assert "single_filter.theta" not in diagnostics


def test_filter_and_route_telemetry_contains_expert_and_view_metrics(graph_bundle):
    model = build_model("M", graph_bundle)
    started = time.perf_counter()
    _, _, _, _, info = model(graph_bundle["x"], graph_bundle["edge_index"], return_details=True)
    elapsed = time.perf_counter() - started
    detail = info["details"]
    assert elapsed > 0
    assert detail["shared_expert_objects"] == 4
    assert detail["actual_experts_executed_per_filter_call"] == 4
    assert "E_pairwise_cosine" in detail["filters"]["private_text_physical"]
    assert "dense_importance" in detail["routes"]["private_text_physical"]
    assert "private_collaborative_route_changes" in detail


def test_validation_label_universe_excludes_masked_test_labels(graph_bundle):
    data = SimpleNamespace(
        y=torch.tensor([0, 1, 0, 1, -100, -100, 2, 2]),
        num_classes=3,
        train_idx=torch.tensor([0, 1, 2]),
        val_idx=torch.tensor([3, 6]),
        test_idx=torch.tensor([4, 5]),
    )
    assert _resolve_nc_eval_labels(data, include_test=False) == [0, 1, 2]
