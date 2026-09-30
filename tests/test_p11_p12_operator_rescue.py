from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.analysis.p0p1_propagation_probe import NeutralOneHopProbe, ProbeData
from src.analysis.p11_p12_operator_rescue import (
    OPERATORS,
    load_frozen_h0,
    operator_context,
    operator_features,
    operator_message,
    read_train_val_indices,
    reconstruct_raw_delta,
    selected_validation_edges,
    validate_operator_fast_vs_brute,
)


def _probe_data() -> ProbeData:
    return ProbeData(
        name="toy", source="toy", x_t=torch.randn(4, 3), x_v=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1, 2, 0], [2, 2, 3, 3]]),
        labels=torch.tensor([0, 1, 2, 1]), train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2, 3]), num_nodes=4, num_classes=3, paths={},
    )


def test_fixed_operator_message_formulas():
    src = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    dst = torch.tensor([[2.0, 1.0], [-1.0, 5.0]])
    assert torch.equal(operator_message("smooth", src, dst), src)
    assert torch.equal(operator_message("absdiff", src, dst), torch.abs(src - dst))
    assert torch.equal(operator_message("product", src, dst), src * dst)


def test_operator_context_rejects_self_loop_and_removal_keeps_original_degree():
    h = torch.tensor([[2.0], [6.0], [0.0]])
    edges = torch.tensor([[0, 1], [2, 2]])
    context, degree = operator_context(h, edges, 3, "smooth")
    assert degree[2].item() == 2
    assert context[2].item() == 4.0
    removed = context[2] - operator_message("smooth", h[0], h[2]) / degree[2]
    assert removed.item() == 3.0
    try:
        operator_context(h, torch.tensor([[0], [0]]), 3, "smooth")
    except ValueError as error:
        assert "self-loops" in str(error)
    else:
        raise AssertionError("self-loop was accepted")


def test_frozen_semantic_checkpoint_and_common_h0(tmp_path: Path):
    data = _probe_data()
    model = NeutralOneHopProbe(3, 2, 3, 8, contextual=False)
    checkpoint = tmp_path / "semantic.pt"
    torch.save({"contextual": False, "seed": 42, "hidden_dim": 8,
                "model_state": model.state_dict(), "metrics": {}}, checkpoint)
    frozen_model, h_t, h_v, _ = load_frozen_h0(data, checkpoint, torch.device("cpu"))
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in frozen_model.parameters())
    assert not hasattr(data, "test_idx")
    features = [operator_features(h_t, h_v, data.edge_index, operator) for operator in OPERATORS]
    common_h0 = torch.cat([h_t, h_v], dim=-1)
    assert all(torch.equal(feature[:, :16], common_h0) for feature in features)
    assert torch.equal(features[0][:, :16], features[1][:, :16])
    assert torch.equal(features[1][:, :16], features[2][:, :16])


def test_same_physical_validation_edge_set_for_all_operators():
    data = _probe_data()
    src, dst, degree = selected_validation_edges(data, torch.device("cpu"))
    assert list(zip(src.tolist(), dst.tolist())) == [(0, 2), (1, 2), (2, 3), (0, 3)]
    assert degree[dst].tolist() == [2, 2, 2, 2]
    # All three operations consume the same src/dst tensors; no per-operator filtering exists.
    sets = [{(int(s), int(d)) for s, d in zip(src.tolist(), dst.tolist())} for _ in OPERATORS]
    assert sets[0] == sets[1] == sets[2]


def test_vectorized_removed_logits_equal_brute_force_for_all_operators():
    torch.manual_seed(7)
    n, hidden, classes = 5, 8, 3
    h_t, h_v = torch.randn(n, hidden), torch.randn(n, hidden)
    edges = torch.tensor([[0, 1, 0, 2, 4], [2, 2, 1, 3, 3]])
    src, dst = edges[:, :4]
    degree = torch.bincount(edges[1], minlength=n)
    for operator in OPERATORS:
        c_t, _ = operator_context(h_t, edges, n, operator)
        c_v, _ = operator_context(h_v, edges, n, operator)
        head = nn.Linear(hidden * 4, classes).eval()
        errors = validate_operator_fast_vs_brute(head, h_t, h_v, c_t, c_v, src, dst, degree, operator)
        assert max(errors.values()) < 1e-8


def test_raw_delta_reconstruction_from_p0_relative_utility():
    relative = np.array([-0.2, 0.3, 0.0], dtype=np.float64)
    ce_full = np.array([0.5, 2.0, 1.0], dtype=np.float64)
    raw = reconstruct_raw_delta(relative, ce_full)
    assert np.array_equal(raw, np.array([-0.1, 0.6, 0.0], dtype=np.float64))


def test_split_reader_never_accesses_test_indices():
    class Guarded(dict):
        def __getitem__(self, key):
            if key in {"test_idx", "test_index", "test"}:
                raise AssertionError("test split access is forbidden")
            return super().__getitem__(key)

    train, val = read_train_val_indices(Guarded({
        "train_idx": torch.tensor([0, 1]), "val_idx": torch.tensor([2]), "test_idx": object(),
    }))
    assert train.tolist() == [0, 1]
    assert val.tolist() == [2]
