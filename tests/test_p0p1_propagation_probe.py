from __future__ import annotations

import pandas as pd
import torch

from src.analysis.p0p1_propagation_probe import (
    NeutralOneHopProbe,
    deterministic_edge_sample,
    margin_utility,
    neighbor_mean,
    read_train_val_indices,
    relative_ce_utility,
    slice_text_visual,
    validate_fast_utility,
)


def test_joint_feature_slicing_preserves_text_then_visual_order():
    joined = torch.tensor([[1.0, 2.0, 10.0, 20.0, 30.0]])
    text, visual = slice_text_visual(joined, text_dim=2, visual_dim=3)
    assert torch.equal(text, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(visual, torch.tensor([[10.0, 20.0, 30.0]]))


def test_incoming_context_excludes_self_loops_and_uses_source_to_destination():
    h = torch.tensor([[2.0], [7.0], [99.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    context, degree = neighbor_mean(h, edge_index, num_nodes=3)
    assert degree.tolist() == [0, 1, 1]
    assert context.squeeze(-1).tolist() == [0.0, 2.0, 7.0]
def test_neighbor_mean_rejects_unremoved_self_loop():
    h = torch.ones((2, 1))
    edge_index = torch.tensor([[0, 1], [0, 1]])
    try:
        neighbor_mean(h, edge_index, num_nodes=2)
    except ValueError as error:
        assert "self-loops" in str(error)
    else:
        raise AssertionError("self-loop context was accepted")


def test_single_message_removal_keeps_original_denominator():
    h = torch.tensor([[2.0], [6.0], [0.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    context, degree = neighbor_mean(h, edge_index, num_nodes=3)
    removed = context[2] - h[0] / degree[2]
    assert torch.equal(degree[2], torch.tensor(2))
    assert torch.allclose(context[2], torch.tensor([4.0]))
    assert torch.allclose(removed, torch.tensor([3.0]))


def test_vectorized_message_utility_matches_brute_force():
    torch.manual_seed(7)
    n, hidden, classes = 5, 8, 3
    network = NeutralOneHopProbe(4, 6, classes, hidden, contextual=True).eval()
    h_t = torch.randn(n, hidden)
    h_v = torch.randn(n, hidden)
    edge_index = torch.tensor([[0, 1, 0, 2, 4], [2, 2, 1, 3, 3]])
    c_t, degree = neighbor_mean(h_t, edge_index, n)
    c_v, _ = neighbor_mean(h_v, edge_index, n)
    labels = torch.tensor([1, 2, 0, 1, 1])
    errors = validate_fast_utility(
        network,
        h_t,
        h_v,
        c_t,
        c_v,
        labels,
        edge_index[0, :3],
        edge_index[1, :3],
        degree,
    )
    assert errors["text_logit_max_abs"] < 1e-6
    assert errors["visual_logit_max_abs"] < 1e-6
    assert errors["text_utility_max_abs"] < 1e-8
    assert errors["visual_utility_max_abs"] < 1e-8


def test_signed_ce_utility_and_margin_direction():
    full = torch.tensor([1.0, 1.0])
    removed = torch.tensor([2.0, 0.5])
    utility = relative_ce_utility(full, removed)
    assert utility[0] > 0
    assert utility[1] < 0
    logits_full = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    logits_removed = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    margins = margin_utility(logits_full, logits_removed, torch.tensor([0, 0]))
    assert margins[0] > 0
    assert margins[1] < 0


def test_split_reader_never_requests_test_indices():
    class GuardedPayload(dict):
        def __getitem__(self, key):
            if key == "test_idx":
                raise AssertionError("test_idx was accessed")
            return super().__getitem__(key)

    train, val = read_train_val_indices(
        GuardedPayload(
            train_idx=torch.tensor([0, 1]),
            val_idx=torch.tensor([2]),
            test_idx=object(),
        )
    )
    assert train.tolist() == [0, 1]
    assert val.tolist() == [2]


def test_edge_sampling_is_deterministic():
    table = pd.DataFrame({"src": range(100), "dst": range(100, 200)})
    a = deterministic_edge_sample(table, max_rows=17, seed=123)
    b = deterministic_edge_sample(table, max_rows=17, seed=123)
    assert a.equals(b)
    assert len(a) == 17
