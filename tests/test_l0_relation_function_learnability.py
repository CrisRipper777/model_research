from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.l0_relation_function_learnability import (
    DATASETS,
    EVIDENCE_VARIANTS,
    KEY_COLUMNS,
    MASTER_DIM,
    SEEDS,
    EvidenceMLP,
    M0StateDirectProbe,
    audit_utility_formula,
    balanced_group_folds,
    build_master_features,
    count_parameters,
    feature_stats,
    high_margin_metrics,
    inner_group_split,
    m0_pair_evidence,
    metric_bundle,
    shuffle_targets_within_destination,
    standardize_targets,
    target_balanced_weights,
    target_reliability,
    load_utility_table,
)
from src.models.adaptive_prop_m0 import graph_standardized_log_degree, leave_one_out_context


ROOT = Path(__file__).resolve().parents[1]
P13 = ROOT / "outputs/p13_joint_readout_operator_probe"


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
def test_full_p13_table_and_head_checkpoint_exist(dataset, seed):
    table = P13 / "runs" / dataset / f"seed_{seed}" / "joint_operator_edge_utility.csv.gz"
    checkpoint = P13 / "checkpoints/runs" / dataset / f"seed_{seed}_joint_linear_head.pt"
    assert table.is_file() and checkpoint.is_file()


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
def test_targets_formula_finite_and_feature_source_columns(dataset, seed):
    path = P13 / "runs" / dataset / f"seed_{seed}" / "joint_operator_edge_utility.csv.gz"
    frame, y = load_utility_table(path, dataset, seed)
    audit = audit_utility_formula(frame, y)
    assert y.shape == (len(frame), 6)
    assert audit["finite_targets"]
    assert audit["max_formula_abs_error"] == 0
    assert set(KEY_COLUMNS).issubset(frame.columns)
    assert "dst_label" not in frame.columns
    assert "ce_full" not in frame.columns
    assert not any("ce_removed" in name or "u_relative" in name for name in frame.columns)


def _small_layout():
    torch.manual_seed(3)
    ht, hv = torch.randn(4, 128), torch.randn(4, 128)
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
    src, dst = edge_index
    degree = torch.tensor([0, 0, 0, 3])
    return build_master_features(ht, hv, src, dst, degree, edge_index), ht, hv, edge_index, degree


def test_master_dimension_and_no_node_ids_labels_or_utility_inputs():
    layout, *_ = _small_layout()
    assert layout.master.shape == (3, MASTER_DIM)
    assert layout.master.shape[1] == 1542
    assert len(layout.names) == 1542
    assert not any(x in " ".join(layout.names).lower() for x in ("src", "dst", "label", "utility", "ce"))
    assert torch.isfinite(layout.master).all()


def test_full_neighbor_mean_matches_brute_force():
    layout, ht, hv, edge_index, degree = _small_layout()
    src = edge_index[0]
    expected_t = ht[src].mean(0)
    expected_v = hv[src].mean(0)
    block = layout.master[:, layout.slices["full_neighbor"]]
    assert torch.allclose(block[0, :128], expected_t)
    assert torch.allclose(block[0, 128:], expected_v)


def test_loo_neighbor_mean_matches_brute_force():
    layout, ht, hv, edge_index, degree = _small_layout()
    src = edge_index[0]
    block = layout.master[:, layout.slices["loo_neighbor"]]
    for row, source in enumerate(src.tolist()):
        expected_t = torch.stack([ht[x] for x in src.tolist() if x != source]).mean(0)
        expected_v = torch.stack([hv[x] for x in src.tolist() if x != source]).mean(0)
        assert torch.allclose(block[row, :128], expected_t, atol=2e-7, rtol=1e-6)
        assert torch.allclose(block[row, 128:], expected_v, atol=2e-7, rtol=1e-6)


def test_loo_degree_one_is_zero_and_matches_m0_scalar_context():
    ht = torch.randn(2, 128)
    edge_index = torch.tensor([[0], [1]])
    layout = build_master_features(ht, ht.clone(), edge_index[0], edge_index[1],
                                   torch.tensor([0, 1]), edge_index)
    loo = layout.master[0, layout.slices["loo_neighbor"]]
    assert torch.equal(loo, torch.zeros_like(loo))
    projected = torch.randn(2, 32)
    context = leave_one_out_context(projected, edge_index[0], edge_index[1], torch.tensor([0, 1]))
    assert torch.equal(context, torch.zeros_like(context))


def test_degree_scalars_use_graph_standardized_log_degree():
    layout, *_ = _small_layout()
    degree = torch.tensor([0, 0, 0, 3])
    src, dst = torch.tensor([0, 1, 2]), torch.tensor([3, 3, 3])
    z = graph_standardized_log_degree(degree)
    scalars = layout.master[:, layout.slices["scalars"]]
    assert torch.allclose(scalars[:, 4], z[dst])
    assert torch.allclose(scalars[:, 5], z[src])


def test_feature_masks_encode_the_five_fixed_evidence_sets():
    layout, *_ = _small_layout()
    masks = layout.masks
    assert list(masks) == list(EVIDENCE_VARIANTS)
    assert int(masks["SIM_ONLY"].sum()) == 2
    assert int(masks["TARGET_ONLY"].sum()) == 513
    assert int(masks["ENDPOINT"].sum()) == 1028
    assert int(masks["ENDPOINT_LOCAL"].sum()) == MASTER_DIM
    assert int(masks["ENDPOINT_LOCAL_SHUFFLED_TARGET"].sum()) == MASTER_DIM
    assert not masks["ENDPOINT"][layout.slices["full_neighbor"]].any()
    assert not masks["ENDPOINT"][layout.slices["loo_neighbor"]].any()


def test_evidence_parameter_count_and_initialization_are_exactly_equal():
    states = []
    counts = []
    for _ in EVIDENCE_VARIANTS:
        torch.manual_seed(9102)
        model = EvidenceMLP()
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        counts.append(count_parameters(model))
    assert counts == [198278] * len(EVIDENCE_VARIANTS)
    assert all(torch.equal(states[0][k], state[k]) for state in states[1:] for k in states[0])


def test_target_group_outer_folds_are_disjoint_and_reused_by_dst():
    dst = np.repeat(np.arange(60), np.arange(1, 61))
    assignment = balanced_group_folds(dst, 3, seed=123)
    assert assignment.dst.is_unique
    assert assignment.fold.nunique() == 3
    mapped = pd.Series(assignment.fold.to_numpy(), index=assignment.dst).loc[dst].to_numpy()
    for target in np.unique(dst):
        assert np.unique(mapped[dst == target]).size == 1
    assert mapped.min() == 0 and mapped.max() == 2


def test_inner_group_split_keeps_targets_disjoint():
    dst = np.repeat(np.arange(60), 3)
    outer = np.arange(len(dst))
    train, val = inner_group_split(dst, outer, seed=82)
    assert not set(dst[train]).intersection(dst[val])
    assert len(np.unique(dst[val])) in (5, 6, 7)


def test_feature_standardization_can_use_train_rows_only():
    x = torch.randn(12, 7)
    rows = np.arange(8)
    mean1, std1 = feature_stats(x, rows)
    x[8:] += 10000
    mean2, std2 = feature_stats(x, rows)
    assert torch.equal(mean1, mean2) and torch.equal(std1, std2)


def test_target_normalization_can_use_train_rows_only():
    y = np.random.default_rng(1).normal(size=(12, 6)).astype(np.float32)
    rows = np.arange(8)
    mean1, std1 = standardize_targets(y, rows)
    y[8:] += 10000
    mean2, std2 = standardize_targets(y, rows)
    assert np.array_equal(mean1, mean2) and np.array_equal(std1, std2)


def test_target_balanced_weights_inverse_degree_and_mean_one():
    degrees = np.array([1, 1, 4, 4, 4], dtype=np.int64)
    weights = target_balanced_weights(degrees)
    assert np.isclose(weights.mean(), 1.0)
    assert np.allclose(weights / weights[0], np.array([1, 1, .25, .25, .25]))


def test_six_vector_shuffle_preserves_each_target_tuple_multiset():
    dst = np.repeat(np.arange(8), 5)
    y = np.arange(8 * 5 * 6, dtype=np.float32).reshape(40, 6)
    shuffled, audit = shuffle_targets_within_destination(y, dst, np.arange(40), seed=7)
    assert audit["multiset_preserved"]
    for target in np.unique(dst):
        ix = dst == target
        before = y[ix][np.lexsort(y[ix].T[::-1])]
        after = shuffled[ix][np.lexsort(shuffled[ix].T[::-1])]
        assert np.array_equal(before, after)


def test_shuffle_changes_correspondence_when_target_has_multiple_edges():
    dst = np.repeat(np.arange(4), 4)
    y = np.arange(4 * 4 * 6, dtype=np.float32).reshape(16, 6)
    shuffled, audit = shuffle_targets_within_destination(y, dst, np.arange(16), seed=13)
    assert audit["eligible_groups"] == 4
    assert audit["changed_groups"] == 4
    assert not np.array_equal(shuffled, y)


def test_direct_state_dimensions_pair_formula_and_frozen_h0():
    torch.manual_seed(17)
    ht, hv = torch.randn(4, 128), torch.randn(4, 128)
    src = torch.tensor([0, 1, 2, 0])
    dst = torch.tensor([3, 3, 3, 2])
    degree = torch.bincount(dst, minlength=4)
    deg_z = graph_standardized_log_degree(degree)
    model = M0StateDirectProbe()
    pos = torch.arange(len(src))
    out, state = model.forward_from_edge_positions(ht, hv, pos, src, dst, degree, deg_z)
    assert out.shape == (4, 6)
    assert state["q_t"].shape == (4, 32) and state["q_v"].shape == (4, 32)
    assert state["r"].shape == (4, 64)
    assert state["u_t"].shape == (4, 64) and state["u_v"].shape == (4, 64)
    assert not ht.requires_grad and not hv.requires_grad
    target, source = torch.randn(4, 32), torch.randn(4, 32)
    loo = torch.randn(4, 1)
    z_i, z_j = torch.randn(4), torch.randn(4)
    actual = m0_pair_evidence(target, source, loo, z_i, z_j)
    cosine = torch.nn.functional.cosine_similarity(target, source, dim=-1).unsqueeze(-1)
    expected = torch.cat([target, source, (target-source).abs(), target*source,
                          cosine, loo, z_i[:, None], z_j[:, None]], dim=-1)
    assert torch.equal(actual, expected)
    assert actual.shape == (4, 132)


def test_direct_state_loo_context_matches_m0_bruteforce():
    torch.manual_seed(22)
    p = torch.randn(5, 32)
    src = torch.tensor([0, 1, 2, 3, 0])
    dst = torch.tensor([4, 4, 4, 4, 2])
    degree = torch.bincount(dst, minlength=5)
    exact = leave_one_out_context(p, src, dst, degree)
    expected = []
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbors = [int(other) for other, target in zip(src.tolist(), dst.tolist())
                     if target == d and other != s]
        if neighbors:
            loo_mean = p[neighbors].mean(0)
            expected.append(torch.nn.functional.cosine_similarity(p[s:s+1], loo_mean[None], dim=-1)[0])
        else:
            expected.append(torch.tensor(0.0))
    assert torch.allclose(exact[:, 0], torch.stack(expected), atol=1e-7)


def test_sign_metrics_and_single_class_auc_are_na():
    y = np.array([1., 2., 3., 4.])
    pred = np.array([4., 3., 2., 1.])
    dst = np.array([0, 0, 1, 1])
    degrees = np.array([2, 2, 2, 2])
    metrics = metric_bundle(y, pred, dst, degrees, "text", "smooth_utility")
    assert np.isnan(metrics["sign_auroc"])
    assert metrics["sign_prevalence"] == 1
    assert np.isfinite(metrics["within_target_residual_spearman"])


def test_high_margin_threshold_is_test_subset_quantile_only():
    y = np.array([-8., -4., -1., 0., 1., 2., 5., 9.])
    pred = y.copy()
    result = high_margin_metrics(y, pred)
    assert result["high_margin_threshold_abs"] == np.quantile(np.abs(y), .75)
    assert result["high_margin_n"] == int((np.abs(y) >= np.quantile(np.abs(y), .75)).sum())
    assert np.isclose(result["high_margin_spearman"], 1.0)


def test_outer_train_statistics_are_independent_of_test_rows():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    train = np.array([0, 1, 2, 3, 4, 5])
    mean_a, std_a = feature_stats(x, train)
    x[6:] = 1e9
    mean_b, std_b = feature_stats(x, train)
    assert torch.equal(mean_a, mean_b) and torch.equal(std_a, std_b)


def test_same_destination_edge_shuffle_keeps_six_target_tuples():
    dst = np.array([10, 10, 10, 11, 11, 12])
    y = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3],
                  [4, 5, 6, 7, 8, 9], [2, 1, 0, 4, 3, 2],
                  [8, 7, 6, 5, 4, 3], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    out, audit = shuffle_targets_within_destination(y, dst, np.arange(len(dst)), seed=9917)
    assert audit["multiset_preserved"]
    assert np.array_equal(out[dst == 12], y[dst == 12])
    for node in (10, 11):
        ix = dst == node
        assert {tuple(row) for row in out[ix]} == {tuple(row) for row in y[ix]}


def test_probe_input_dimension_has_no_appended_endpoint_ids():
    layout, *_ = _small_layout()
    assert layout.master.shape[1] == 1542
    assert all(len(name.split("_")) == 2 and name.startswith("feature_") for name in layout.names)


def test_cross_seed_reliability_uses_function_deltas_not_alternative_raw_utility():
    tables = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            offset = seed / 10
            tables[(dataset, seed)] = pd.DataFrame({
                "src": [0, 1, 2, 3], "dst": [4, 4, 5, 5],
                "u_raw_smooth_text": np.array([0.1, 0.2, 0.3, 0.4]) + offset,
                "u_raw_absdiff_text": np.array([0.2, 0.2, 0.5, 0.3]) + offset,
                "u_raw_product_text": np.array([0.0, 0.7, 0.1, 0.8]) + offset,
                "u_raw_smooth_visual": np.array([0.2, 0.4, 0.2, 0.1]) + offset,
                "u_raw_absdiff_visual": np.array([0.3, 0.2, 0.5, 0.1]) + offset,
                "u_raw_product_visual": np.array([0.6, 0.1, 0.3, 0.4]) + offset,
            })
    result = target_reliability(tables)
    row = result[(result.dataset == "Movies") & (result.modality == "text") &
                 (result.target == "Delta_D") & (result.seed_a == 42) &
                 (result.seed_b == 43)].iloc[0]
    assert row.spearman == pytest.approx(1.0)
    assert row.centered_spearman == pytest.approx(1.0)
    assert row.overlap_fraction_smaller == 1.0
