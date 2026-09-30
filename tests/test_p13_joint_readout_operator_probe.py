from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.analysis.p0p1_propagation_probe import NeutralOneHopProbe, ProbeData, read_train_val_indices
from src.analysis import p11_p12_operator_rescue as p12
from src.analysis.p13_joint_readout_operator_probe import (
    BLOCKS,
    assert_edge_alignment,
    build_joint_features,
    joint_feature_dim,
    joint_feature_slices,
    modality_sign_disagreement,
    new_joint_head,
    preferred_channel_from_utilities,
    summarize_within_node_preferences,
    validate_joint_fast_vs_brute,
)


def _probe_data() -> ProbeData:
    return ProbeData(
        name="toy", source="toy", x_t=torch.randn(5, 3), x_v=torch.randn(5, 2),
        edge_index=torch.tensor([[0, 1, 0, 2], [3, 3, 4, 4]]),
        labels=torch.tensor([0, 1, 0, 1, 2]), train_idx=torch.tensor([0, 1, 2]),
        val_idx=torch.tensor([3, 4]), num_nodes=5, num_classes=3, paths={},
    )


def test_joint_layout_order_and_feature_dimension():
    data = _probe_data()
    h_t, h_v = torch.randn(data.num_nodes, 128), torch.randn(data.num_nodes, 128)
    features, contexts, _ = build_joint_features(h_t, h_v, data.edge_index)
    slices = joint_feature_slices()
    assert tuple(BLOCKS) == ("H_T", "H_V", "S_T", "D_T", "P_T", "S_V", "D_V", "P_V")
    assert features.shape == (data.num_nodes, 8 * 128)
    assert joint_feature_dim() == 1024
    expected = {"H_T": h_t, "H_V": h_v, **contexts}
    for block in BLOCKS:
        assert torch.equal(features[:, slices[block]], expected[block])
    assert not features.requires_grad


def test_only_one_trainable_linear_head_is_created():
    head = new_joint_head(1024, 4, torch.device("cpu"))
    trainable = [module for module in head.modules() if isinstance(module, nn.Linear) and any(p.requires_grad for p in module.parameters())]
    assert isinstance(head, nn.Linear)
    assert len(trainable) == 1
    assert not list(head.children())


def test_semantic_h0_is_frozen_and_context_formulas_are_the_p12_functions(tmp_path: Path):
    data = _probe_data()
    model = NeutralOneHopProbe(3, 2, 3, 8, contextual=False)
    checkpoint = tmp_path / "semantic.pt"
    torch.save({"contextual": False, "seed": 42, "hidden_dim": 8,
                "model_state": model.state_dict(), "metrics": {}}, checkpoint)
    frozen, h_t, h_v, _ = p12.load_frozen_h0(data, checkpoint, torch.device("cpu"))
    assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
    assert p12.operator_message is __import__("src.analysis.p13_joint_readout_operator_probe", fromlist=["operator_message"]).operator_message
    assert p12.operator_context is __import__("src.analysis.p13_joint_readout_operator_probe", fromlist=["operator_context"]).operator_context
    src = torch.tensor([[1.0, -2.0]])
    dst = torch.tensor([[2.0, 1.0]])
    assert torch.equal(p12.operator_message("smooth", src, dst), src)
    assert torch.equal(p12.operator_message("absdiff", src, dst), torch.abs(src-dst))
    assert torch.equal(p12.operator_message("product", src, dst), src*dst)
    assert h_t.shape == h_v.shape == (data.num_nodes, 8)


def test_same_physical_edge_set_and_order_matches_p0_and_p12(tmp_path: Path):
    current = pd.DataFrame({"src": [0, 1, 0], "dst": [3, 3, 4], "dst_degree": [2, 2, 2]})
    p0 = tmp_path / "p0.csv"
    p12_path = tmp_path / "p12.csv"
    rows = current.assign(dataset="toy", seed=42)
    rows.to_csv(p0, index=False)
    rows.to_csv(p12_path, index=False)
    result = assert_edge_alignment(current, p0, p12_path, "toy", 42)
    assert result["edge_set_match"] and result["edge_order_match"]
    rows.iloc[::-1].to_csv(p12_path, index=False)
    with pytest.raises(AssertionError, match="edge order differs"):
        assert_edge_alignment(current, p0, p12_path, "toy", 42)


def test_fixed_original_degree_denominator_is_retained_after_removal():
    h = torch.tensor([[2.0], [6.0], [10.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    context, degree = p12.operator_context(h, edge_index, 3, "smooth")
    removed = context[2] - p12.operator_message("smooth", h[0], h[2]) / degree[2]
    assert degree[2].item() == 2
    assert removed.item() == 3.0


def test_fast_removal_matches_brute_force_for_all_six_channels():
    torch.manual_seed(7)
    data = _probe_data()
    h_t, h_v = torch.randn(data.num_nodes, 128), torch.randn(data.num_nodes, 128)
    features, _, degree = build_joint_features(h_t, h_v, data.edge_index)
    head = new_joint_head(1024, data.num_classes, torch.device("cpu")).eval()
    src, dst = data.edge_index
    errors = validate_joint_fast_vs_brute(head, features, h_t, h_v, src, dst, degree)
    assert len(errors) == 6
    assert max(errors.values()) < 1e-8


def test_preferred_channel_none_and_positive_argmax_logic():
    assert preferred_channel_from_utilities({"smooth": 0.0, "absdiff": -0.1, "product": 0.0}) == "none"
    assert preferred_channel_from_utilities({"smooth": -0.1, "absdiff": 0.2, "product": 0.2}) == "absdiff"
    assert preferred_channel_from_utilities({"smooth": 0.3, "absdiff": 0.2, "product": 0.1}) == "smooth"


def test_sign_disagreement_reports_zero_aware_and_p12_compatible_conventions():
    result = modality_sign_disagreement([0.0, 1.0, -1.0], [1.0, -1.0, -1.0])
    assert result["three_way_sign_disagreement"] == 2 / 3
    assert result["positive_vs_nonpositive_sign_disagreement"] == 2 / 3
    assert result["text_exact_zero_fraction"] == 1 / 3


def test_within_node_diversity_excludes_none_and_low_degree_targets():
    targets, summary = summarize_within_node_preferences(
        [1, 1, 1, 2, 2], [5, 5, 5, 4, 4], ["smooth", "absdiff", "none", "smooth", "product"], degree_min=5,
    )
    assert len(targets) == 1
    assert targets[0]["target"] == 1
    assert targets[0]["positive_preferred_edge_count"] == 2
    assert targets[0]["distinct_positive_functions"] == 2
    assert targets[0]["multi_function_neighborhood"] is True
    assert summary["multi_function_neighborhood_ratio"] == 1.0
    no_positive, empty_summary = summarize_within_node_preferences([4, 4], [5, 5], ["none", "none"])
    assert no_positive[0]["distinct_positive_functions"] == 0
    assert no_positive[0]["no_positive_preference"] is True
    assert empty_summary["no_positive_preference_target_fraction"] == 1.0


def test_probe_data_and_split_reader_never_expose_test_indices():
    data = _probe_data()
    assert not hasattr(data, "test_idx")

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
