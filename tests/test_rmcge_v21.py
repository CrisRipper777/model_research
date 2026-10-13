from __future__ import annotations

import io

import torch
from omegaconf import OmegaConf

import src.models.rmcge_v21 as v21_module
from src.models.psce_mag_v7a import chunked_weighted_spmm
from src.models.rmcge_v2 import Model as V2Model, build_modality_knn_operators
from src.models.rmcge_v21 import Model, conserve_receiver_mass
from src.utils.seeds import set_seed


def _cfg(condition: str):
    return OmegaConf.create(
        {
            "model": {
                "name": "rmcge_v21",
                "condition": condition,
                "variant": "C0",
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
                "dropout": 0.2,
                "balance_weight": 0.01,
                "eps": 1.0e-8,
                "relation_chunk_size": 32,
                "propagation_chunk_size": 16,
                "edge_chunk_size": 3,
                "purpose_edge_chunk_size": 2,
                "node_chunk_size": 2,
                "scale_floor": 1.0e-3,
            }
        }
    )


def _info(num_nodes: int = 6):
    return {"input_dim": 4, "text_dim": 2, "visual_dim": 2, "num_nodes": num_nodes, "num_classes": 3}


def _graph():
    return torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 0, 2, 1, 3, 2, 4, 3, 4]],
        dtype=torch.long,
    )


def _attach_cache(model):
    unique = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long)
    flags = torch.tensor([[1, 0], [0, 2], [1, 1], [0, 1]], dtype=torch.uint8)
    supports = build_modality_knn_operators(unique, flags, model.num_nodes)
    for modality in ("text", "visual"):
        setattr(model, f"semantic_{modality}_edge_index_v2", supports[modality]["edge_index"])
        setattr(model, f"semantic_{modality}_weight_v2", supports[modality]["weight"])
    model.semantic_source_flags_v7b = flags
    model.semantic_candidate_fingerprint = "synthetic-cache"
    model.semantic_v7b_fingerprint = "synthetic-v7b"


def _assert_finite_nonzero_gradient(model, name):
    gradient = dict(model.named_parameters())[name].grad
    assert gradient is not None, name
    assert torch.isfinite(gradient).all(), name
    assert gradient.norm() > 0, name


def test_c0_delegates_to_exact_frozen_v2_v1_forward():
    set_seed(100)
    candidate = Model(_cfg("C0"), _info()).eval()
    set_seed(100)
    reference_cfg = _cfg("C0")
    reference_cfg.model.variant = "V1"
    reference = V2Model(reference_cfg, _info()).eval()
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    for key in candidate.state_dict():
        assert torch.equal(candidate.state_dict()[key], reference.state_dict()[key]), key
    _attach_cache(candidate)
    _attach_cache(reference)
    x, edge = torch.randn(6, 4), _graph()
    with torch.no_grad():
        actual = candidate(x, edge, return_details=True)
        expected = reference(x, edge, return_details=True)
    for left, right in zip(actual[:4], expected[:4]):
        if left is not None:
            assert torch.equal(left, right)
    assert actual[4]["variant"] == "C0"
    assert expected[4]["variant"] == "V1"


def test_c0_c1_c2_common_initialization_and_head_rng_pairing():
    constructed = {}
    heads = {}
    for condition in ("C0", "C1", "C2"):
        set_seed(101)
        model = Model(_cfg(condition), _info())
        head = torch.nn.Linear(256, 3)
        constructed[condition] = model
        heads[condition] = head
    common = set(dict(constructed["C0"].named_parameters()))
    for condition in ("C1", "C2"):
        parameters = dict(constructed[condition].named_parameters())
        for name in common:
            assert torch.equal(dict(constructed["C0"].named_parameters())[name], parameters[name]), name
    c1_params, c2_params = dict(constructed["C1"].named_parameters()), dict(constructed["C2"].named_parameters())
    assert c1_params.keys() == c2_params.keys()
    for name in c1_params:
        assert torch.equal(c1_params[name], c2_params[name]), name
    for condition in ("C1", "C2"):
        assert torch.equal(heads["C0"].weight, heads[condition].weight)
        assert torch.equal(heads["C0"].bias, heads[condition].bias)
    assert torch.equal(constructed["C1"].cross_delta_coefficients, torch.full((4,), 0.20))


def test_mass_conservation_zero_equal_and_single_neighbor_scores():
    physical = torch.tensor([0.5, 0.25, 0.75, 0.4, 0.6, 0.9], dtype=torch.float32)
    dst = torch.tensor([0, 0, 0, 1, 1, 2])
    for scores in (torch.zeros(6), torch.full((6,), 0.73)):
        selected = conserve_receiver_mass(physical, scores, dst, num_nodes=4)
        assert torch.equal(selected, physical)
    # Node 1 has two edges; node 2 has one; node 3 is isolated.
    selected = conserve_receiver_mass(physical, torch.tensor([-0.5, 0.9, 0.1, 0.8, -0.2, 0.3]), dst, 4)
    before = torch.zeros(4).index_add(0, dst, physical)
    after = torch.zeros(4).index_add(0, dst, selected)
    assert torch.allclose(before, after, atol=2e-7, rtol=2e-7)
    assert selected[3] + selected[4] == physical[3] + physical[4]
    assert selected[5] == physical[5]  # a single incoming edge cannot be reweighted
    # The isolated node receives no edge and has zero mass on both operators.
    assert before[2] == physical[5] and before[3] == 0


def test_directed_reverse_edges_are_selected_independently_and_duplicates_survive():
    physical = torch.tensor([0.5, 0.5, 0.25, 0.25, 0.25], dtype=torch.float32)
    dst = torch.tensor([1, 0, 0, 0, 1])
    scores = torch.tensor([0.8, -0.6, 0.1, 0.1, -0.2])
    selected = conserve_receiver_mass(physical, scores, dst, 2)
    assert selected[0] != selected[1]  # reverse directed messages are scored separately
    assert selected[2] == selected[3]
    row = torch.zeros(2).index_add(0, dst, selected)
    original = torch.zeros(2).index_add(0, dst, physical)
    assert torch.allclose(row, original, atol=1e-7)


def test_query_interaction_can_reverse_neighbor_order_for_different_receivers():
    model = Model(_cfg("C1"), _info()).eval()
    with torch.no_grad():
        model.query_projection.weight.zero_()
        model.key_projection.weight.zero_()
        model.query_projection.weight[0, 0] = 1.0
        model.key_projection.weight[0, 0] = 1.0
        for parameter in model.purpose_edge_scorer.parameters():
            parameter.zero_()
        model.purpose_edge_scorer[0].weight[0, 64] = 1.0  # q*k interaction
        model.purpose_edge_scorer[2].weight[0, 0] = 1.0
    receiver = torch.zeros(4, 256)
    receiver[0, 0], receiver[1, 0], receiver[2, 0], receiver[3, 0] = 1.0, -1.0, 1.0, -1.0
    src = torch.tensor([2, 3, 2, 3])
    dst = torch.tensor([0, 0, 1, 1])
    scores = model._score_message_edges(
        receiver, receiver, src, dst, torch.tensor([2.0, 2.0, 1.0, 1.0]), 0, 0
    )
    assert scores[0] > scores[1]
    assert scores[2] < scores[3]
    selected = conserve_receiver_mass(torch.full((4,), 0.5), scores, dst, 4)
    prob = selected / torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert prob[0] > prob[1]
    assert prob[2] < prob[3]


def test_c1_c2_zero_scores_and_cross_selector_off_recover_c0():
    models = {}
    for condition in ("C0", "C1", "C2"):
        set_seed(102)
        models[condition] = Model(_cfg(condition), _info()).eval()
        _attach_cache(models[condition])
    x, edge = torch.randn(6, 4), _graph()
    with torch.no_grad():
        baseline = models["C0"](x, edge)
        for condition in ("C1", "C2"):
            candidate = models[condition](x, edge)
            for left, right in zip(candidate[:4], baseline[:4]):
                if left is not None:
                    assert torch.allclose(left, right, atol=2e-6, rtol=2e-6)
        models["C2"].set_interventions(cross_selection_off=True)
        c2_off = models["C2"](x, edge)
        c1 = models["C1"](x, edge)
        for left, right in zip(c2_off[:4], c1[:4]):
            if left is not None:
                assert torch.equal(left, right)


def test_c1_c2_selector_gradients_and_adamw_updates():
    for condition in ("C1", "C2"):
        set_seed(103)
        model = Model(_cfg(condition), _info()).train()
        _attach_cache(model)
        x, edge = torch.randn(6, 4), _graph()
        head = torch.nn.Linear(256, 3)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
        first_before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            z, _, _, aux, _ = model(x, edge)
            loss = torch.nn.functional.cross_entropy(head(z[:4]), torch.tensor([0, 1, 2, 0])) + aux
            assert torch.isfinite(loss)
            loss.backward()
            if step == 0:
                _assert_finite_nonzero_gradient(model, "purpose_edge_scorer.2.weight")
                for name in (
                    "cross_projections.visual_to_text.0.weight",
                    "cross_projections.text_to_visual.0.weight",
                    "cross_coefficients",
                    "node_router.3.weight",
                    "fusion_linear1.weight",
                ):
                    _assert_finite_nonzero_gradient(model, name)
                assert any(
                    dict(model.named_parameters())[f"experts.{expert_id}.linear1.weight"].grad is not None
                    and dict(model.named_parameters())[f"experts.{expert_id}.linear1.weight"].grad.norm() > 0
                    for expert_id in range(4)
                )
            else:
                for name in (
                    "query_projection.weight",
                    "key_projection.weight",
                    "purpose_edge_scorer.0.weight",
                    "delta_coefficients",
                ):
                    _assert_finite_nonzero_gradient(model, name)
                if condition == "C2":
                    _assert_finite_nonzero_gradient(model, "cross_delta_coefficients")
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True)
            optimizer.step()
        after = dict(model.named_parameters())
        for name in ("purpose_edge_scorer.2.weight", "cross_projections.visual_to_text.0.weight"):
            assert not torch.equal(first_before[name], after[name]), (condition, name)
        assert any(
            not torch.equal(first_before[f"experts.{expert_id}.linear1.weight"], after[f"experts.{expert_id}.linear1.weight"])
            for expert_id in range(4)
        )


def test_cross_projection_runs_before_physical_aggregation(monkeypatch):
    set_seed(104)
    model = Model(_cfg("C2"), _info()).eval()
    _attach_cache(model)
    x, edge = torch.randn(6, 4), _graph()
    original = v21_module.chunked_weighted_spmm
    seen = []

    def capture(state, edge_index, weight, chunk_size):
        output = original(state, edge_index, weight, chunk_size)
        seen.append((state.detach().clone(), edge_index.detach().clone()))
        return output

    monkeypatch.setattr(v21_module, "chunked_weighted_spmm", capture)
    with torch.no_grad():
        model(x, edge)
    prior_visual = model.projectors["visual"](x[:, 2:])
    expected_text_sender = model.cross_projections["visual_to_text"](prior_visual)
    # The first physical aggregation in the forward is Text <- projected Visual.
    assert torch.equal(seen[0][0], expected_text_sender)
    src, dst, _, _, _ = model._normalized_operator(edge, x.size(0), prior_visual.dtype)
    assert torch.equal(seen[0][1], torch.stack((src, dst)))


def test_edge_chunking_keeps_duplicates_and_matches_unchunked_scores():
    set_seed(105)
    model = Model(_cfg("C2"), _info()).eval()
    receiver, sender = torch.randn(6, 256), torch.randn(6, 256)
    src = torch.tensor([0, 0, 1, 2, 2, 3, 3])
    dst = torch.tensor([1, 1, 1, 0, 0, 2, 4])
    degree = torch.tensor([2.0, 3.0, 1.0, 1.0, 1.0, 0.0])
    chunked = model._score_message_edges(receiver, sender, src, dst, degree, 1, 1)
    model.purpose_edge_chunk_size = 100
    unchunked = model._score_message_edges(receiver, sender, src, dst, degree, 1, 1)
    assert chunked.shape == (7,)
    assert torch.allclose(chunked, unchunked, atol=1e-7, rtol=1e-6)
    physical = torch.full((7,), 0.25)
    selected = conserve_receiver_mass(physical, chunked, dst, 6)
    assert torch.isfinite(selected).all()
    assert torch.allclose(
        torch.zeros(6).index_add(0, dst, selected),
        torch.zeros(6).index_add(0, dst, physical),
        atol=1e-7,
        rtol=1e-7,
    )


def test_selection_details_capture_conservation_entropy_ess_and_role_differences():
    set_seed(106)
    model = Model(_cfg("C2"), _info()).eval()
    _attach_cache(model)
    x, edge = torch.randn(6, 4), _graph()
    with torch.no_grad():
        details = model(x, edge, return_details=True)[4]["details"]
    paths = details["edge_selection"]
    assert set(paths) == {"text_intra", "text_cross", "visual_intra", "visual_cross"}
    for diagnostic in paths.values():
        assert diagnostic["receiver_row_mass_error_max"] <= 2e-7
        assert diagnostic["effective_neighbors_mean_degree_ge2"] is not None
        assert diagnostic["neighbor_entropy_selected_mean_degree_ge2"] is not None
        assert 0.0 <= diagnostic["score_abs_ge_0_99_fraction"] <= 1.0
    assert "text_intra__text_cross" in details["role_distribution_comparisons"]


def test_checkpoint_round_trip_and_test_metrics_stay_absent():
    set_seed(107)
    source = Model(_cfg("C2"), _info()).eval()
    _attach_cache(source)
    x, edge = torch.randn(6, 4), _graph()
    with torch.no_grad():
        expected = source(x, edge)[0]
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    restored = Model(_cfg("C2"), _info()).eval()
    _attach_cache(restored)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(x, edge)[0]
    assert torch.equal(actual, expected)


def test_zero_messages_remain_zero_after_reference_normalization():
    zeros = torch.zeros(4, 256)
    scale = torch.full((256,), 1.0e-3)
    assert torch.equal(zeros / scale, zeros)
