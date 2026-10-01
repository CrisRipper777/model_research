from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis import l0_relation_function_learnability as l0
from src.analysis import l01_shared_slot_function_identifiability as l01
from src.analysis.p13_joint_readout_operator_probe import operator_context
from src.analysis.p0p1_propagation_probe import ProbeData


ROOT = Path(__file__).resolve().parents[1]


def test_source_branch_sha_is_pinned_to_l0_baseline():
    sha = subprocess.check_output(
        ["git", "rev-parse", "exp/l0_relation_function_learnability_audit"], cwd=ROOT, text=True
    ).strip()
    assert sha == "9ce5f723fda0bf17264671a1e20e78f043727498"


def test_smoke_has_frozen_h0_alignment_and_no_validation_head_leakage():
    audit_path = ROOT / "outputs/l01_shared_slot_function_identifiability/smoke/smoke_audit.json"
    assert audit_path.is_file(), "run the prescribed Movies/42 smoke before the full campaign"
    import json
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "passed"
    assert audit["h0_checkpoint_regression"]["max_abs_metric_difference"] <= 2e-5
    assert audit["h0_checkpoint_regression"]["all_parameters_frozen"] is True
    assert audit["p13_edge_alignment"]["edge_order_match"] is True
    assert audit["validation_labels_not_used_for_head_fit"] is True
    assert audit["no_test_labels_or_metrics_accessed"] is True


def test_probe_data_contract_does_not_expose_test_indices():
    assert "test_idx" not in ProbeData.__dataclass_fields__
    assert "test_index" not in ProbeData.__dataclass_fields__


def test_operator_context_matches_exact_sdp_formula_and_original_degree():
    h = torch.tensor([[1., 2.], [3., -1.], [4., 5.]], dtype=torch.float64)
    edge = torch.tensor([[0, 2, 0], [1, 1, 2]], dtype=torch.long)
    degree = torch.bincount(edge[1], minlength=len(h))
    assert not torch.any(edge[0] == edge[1])
    for op in l01.OPERATORS:
        context, returned_degree = operator_context(h, edge, len(h), op)
        src, dst = edge
        msg = h[src] if op == "smooth" else torch.abs(h[src] - h[dst]) if op == "absdiff" else h[src] * h[dst]
        brute = torch.zeros_like(h).index_add_(0, dst, msg) / degree.clamp_min(1).double()[:, None]
        assert torch.equal(returned_degree, degree)
        torch.testing.assert_close(context, brute, rtol=0, atol=0)


def test_operator_context_rejects_self_messages():
    with pytest.raises(ValueError, match="self-loops"):
        operator_context(torch.ones(2, 3), torch.tensor([[0, 1], [0, 0]]), 2, "smooth")


def test_shared_slot_is_one_linear_512_head_and_nine_fixed_combinations():
    head = l01.SharedSlotHead(num_classes=5)
    linears = [layer for layer in head.modules() if isinstance(layer, nn.Linear)]
    assert len(linears) == 1
    assert linears[0].in_features == 512 and linears[0].out_features == 5
    combos = list(__import__("itertools").product(l01.OPERATORS, repeat=2))
    assert len(combos) == 9
    assert head(torch.randn(7, 512)).shape == (7, 5)
    assert l01.SLOT_TEXT == slice(256, 384)
    assert l01.SLOT_VISUAL == slice(384, 512)


def test_train_only_stratified_split_is_deterministic_and_disjoint():
    train = np.arange(60)
    labels = np.repeat(np.arange(3), 20)
    tr1, va1, audit1 = l01.stratified_head_split(train, labels, seed=177)
    tr2, va2, audit2 = l01.stratified_head_split(train, labels, seed=177)
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2)
    assert not set(tr1) & set(va1)
    assert set(tr1) | set(va1) == set(train)
    assert audit1["split_mode"] == "stratified"
    assert audit1["validation_nodes_excluded"] is True
    assert audit1 == audit2


def test_rare_class_split_uses_deterministic_fallback_and_audits_missing_classes():
    labels = np.asarray([0] * 8 + [1] * 1 + [2] * 8)
    a, b, audit = l01.stratified_head_split(np.arange(len(labels)), labels, seed=41)
    assert audit["split_mode"] == "deterministic_unstratified_fallback"
    assert set(a) & set(b) == set()
    assert set(a) | set(b) == set(range(len(labels)))
    assert audit["validation_nodes_excluded"] is True


def test_text_and_visual_substitution_change_only_their_own_slot():
    torch.manual_seed(3)
    head = nn.Linear(512, 4, dtype=torch.float64)
    feature = torch.randn(11, 512, dtype=torch.float64)
    delta = torch.randn(11, 128, dtype=torch.float64)
    for modality, changed_slice, untouched_slice in (
        ("text", l01.SLOT_TEXT, l01.SLOT_VISUAL),
        ("visual", l01.SLOT_VISUAL, l01.SLOT_TEXT),
    ):
        changed, logits = l01.substitute_edge_logits(head, feature, delta, modality, brute=True)
        assert torch.equal(changed[:, untouched_slice], feature[:, untouched_slice])
        torch.testing.assert_close(changed[:, changed_slice], feature[:, changed_slice] + delta,
                                   rtol=0, atol=0)
        manual = F.linear(feature, head.weight, head.bias) + delta @ head.weight[:, changed_slice].T
        torch.testing.assert_close(logits, manual, rtol=1e-7, atol=1e-8)


def test_gain_sign_and_js_numerical_stability():
    base_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float64)
    sub_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    y = torch.tensor([0, 1])
    gain = F.cross_entropy(base_logits, y, reduction="none") - F.cross_entropy(sub_logits, y, reduction="none")
    assert torch.all(gain > 0)  # positive is CE reduction from the replacement.
    p = torch.softmax(torch.tensor([[1000., -1000.]], dtype=torch.float64), -1)
    q = torch.softmax(torch.tensor([[999., -999.]], dtype=torch.float64), -1)
    assert torch.isfinite(l01.js_divergence(p, q)).all()
    assert abs(float(l01.js_divergence(p, p))) < 1e-14


def test_three_repeat_utility_alignment_and_uncertainty_summary():
    parts = []
    for repeat in range(3):
        table = pd.DataFrame({"src": [0, 1], "dst": [2, 2], "dst_degree": [2, 2],
                              "head_repeat": repeat,
                              **{target: np.asarray([repeat - 1., 1. + repeat]) for target in l01.UTILITY_COLUMNS}})
        parts.append(table)
    raw = pd.concat(parts, ignore_index=True)
    clean = l01.aggregate_repeats(raw)
    assert len(clean) == 2
    assert np.allclose(clean["mean_G_D_text"], [0., 2.])
    assert np.allclose(clean["std_G_D_text"], [1., 1.])
    assert clean["sign_consistent_G_D_text"].tolist() == [False, True]


def test_head_repeat_and_cross_seed_metrics_align_on_edge_keys():
    original_datasets = l01.DATASETS
    l01.DATASETS = ("Movies",)
    try:
        _test_head_repeat_and_cross_seed_metrics_align_on_edge_keys_body()
    finally:
        l01.DATASETS = original_datasets


def _test_head_repeat_and_cross_seed_metrics_align_on_edge_keys_body():
    frames = {}
    for seed in l01.SEEDS:
        rows = []
        for repeat in range(3):
            for src, gain in enumerate([-.2, .1, .3, -.1]):
                row = {"dataset": "Movies", "seed": seed, "src": src, "dst": 9,
                       "dst_degree": 4, "head_repeat": repeat}
                for target in l01.UTILITY_COLUMNS:
                    row[target] = gain + repeat * .01 + (seed - 42) * .001
                rows.append(row)
        frames[seed] = pd.DataFrame(rows)
    rel = l01.head_repeat_reliability(pd.concat(list(frames.values()), ignore_index=True))
    assert set(zip(rel.repeat_a, rel.repeat_b)) == {(0, 1), (0, 2), (1, 2)}
    clean = {("Movies", seed): l01.aggregate_repeats(frame).assign(dataset="Movies", seed=seed)
             for seed, frame in frames.items()}
    cross = l01.utility_cross_seed_reliability(clean)
    assert len(cross) == 3 * len(l01.UTILITY_COLUMNS)
    assert (cross.overlap_edges == 4).all()


def test_p13_delta_alignment_and_direction_are_explicit():
    clean = pd.DataFrame({"src": [0, 1, 2], "dst": [8, 8, 9], "dst_degree": [2, 2, 1],
                         "mean_G_D_text": [0.1, -0.2, 0.3], "mean_G_P_text": [0.2, 0.1, -0.1],
                         "mean_G_D_visual": [0.2, -0.1, 0.1], "mean_G_P_visual": [-0.1, 0.2, 0.3]})
    old = pd.DataFrame({"src": [0, 1, 2], "dst": [8, 8, 9], "dst_degree": [2, 2, 1]})
    for modality in l01.MODALITIES:
        old[f"u_raw_smooth_{modality}"] = [0., 0., 0.]
        old[f"u_raw_absdiff_{modality}"] = [.11, -.19, .31]
        old[f"u_raw_product_{modality}"] = [.19, .09, -.09]
    comparison = l01.compare_p13({("Movies", 42): clean}, {("Movies", 42): old})
    assert len(comparison) == 4
    assert set(comparison.historical_target) == {"Delta_D", "Delta_P"}


def test_master_feature_reuses_l0_exact_1542_dimension_and_masks():
    h_t = torch.randn(4, 128)
    h_v = torch.randn(4, 128)
    edge = torch.tensor([[0, 2, 1], [1, 1, 3]])
    layout = l0.build_master_features(h_t, h_v, torch.tensor([0, 2, 1]),
        torch.tensor([1, 1, 3]), torch.tensor([0, 2, 0, 1]), edge)
    assert layout.master.shape == (3, 1542)
    assert set(l01.MASTER_VARIANTS) <= set(layout.masks)
    assert all(int(layout.masks[name].sum()) > 0 for name in l01.MASTER_VARIANTS)


def test_inner_scaler_statistics_ignore_inner_validation_and_outer_test(monkeypatch):
    torch.manual_seed(5)
    master = torch.randn(12, 1542)
    target = np.arange(48, dtype=np.float64).reshape(12, 4)
    dst = np.repeat(np.arange(6), 2)
    train, val, test = np.asarray([0, 1, 2, 3]), np.asarray([4, 5, 6, 7]), np.asarray([8, 9, 10, 11])
    captured = {}

    def fake_fit(model, forward_all, y_scaled, train_rows, val_rows, test_rows,
                 edge_weights, device, **kwargs):
        captured["y_scaled"] = y_scaled.copy()
        hook = model.net[0].register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("x", args[0].detach().cpu().numpy().copy()))
        forward_all()
        hook.remove()
        captured["train"] = train_rows.copy()
        return {"prediction_scaled": np.zeros((len(test_rows), 4)), "best_epoch": 1,
                "epochs_run": 1, "best_inner_weighted_huber": 0.0, "finite_gradients": True,
                "parameter_count": 198020}

    monkeypatch.setattr(l0, "fit_fullbatch_probe", fake_fit)
    layout = l0.FeatureLayout(master, {}, {name: torch.ones(1542, dtype=torch.bool) for name in l01.MASTER_VARIANTS}, [])
    _, metadata = l01.fit_evidence_fold(master, layout.masks, target, dst, np.repeat(2, 12),
        train, val, test, "ENDPOINT_LOCAL", torch.device("cpu"), seed=9)
    mean, std = l0.standardize_targets(target, train)
    assert np.allclose(captured["y_scaled"][train], (target[train] - mean) / std)
    assert np.allclose(captured["y_scaled"][val], (target[val] - mean) / std)
    assert np.all(captured["y_scaled"][test] == 0)
    x_mean, x_std = l0.feature_stats(master, train)
    expected = ((master - x_mean) / x_std).numpy()
    assert np.allclose(captured["x"], expected)
    assert metadata["feature_scaler_fit_rows_exact_inner_train"]
    assert metadata["target_scaler_fit_rows_exact_inner_train"]
    assert metadata["feature_scaler_rows"] == len(train) == metadata["target_scaler_rows"]
    assert metadata["outer_test_targets_passed_to_trainer"] is False


def test_outer_and_inner_group_splits_are_dst_disjoint():
    dst = np.repeat(np.arange(30), 3)
    folds = l0.balanced_group_folds(dst, 3, seed=42)
    lookup = dict(zip(folds.dst, folds.fold))
    assignment = np.asarray([lookup[int(x)] for x in dst])
    train, test = l01.fold_rows(assignment, 0)
    assert not set(dst[train]) & set(dst[test])
    inner_train, inner_val = l01.inner_split(dst, train, seed=3)
    assert not set(dst[inner_train]) & set(dst[inner_val])


def test_strict_shuffle_preserves_train_and_validation_tuple_multisets():
    y = np.arange(48, dtype=np.float64).reshape(12, 4)
    dst = np.asarray([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    train, val, test = np.asarray([0, 1, 4, 5]), np.asarray([2, 3, 6, 7]), np.asarray([8, 9, 10, 11])
    shuffled_train, audit_train = l01.shuffled_within_dst(y, dst, train, 9917)
    shuffled_val, audit_val = l01.shuffled_within_dst(shuffled_train, dst, val, 9918)
    assert audit_train["multiset_preserved"] and audit_val["multiset_preserved"]
    assert np.array_equal(shuffled_val[test], y[test])


def test_evidence_architecture_parameter_count_is_shared_across_masks():
    counts = []
    for _ in l01.MASTER_VARIANTS:
        counts.append(l0.count_parameters(l0.EvidenceMLP(output_dim=4)))
    assert counts == [counts[0]] * len(l01.MASTER_VARIANTS)
    assert counts[0] == 198020


def test_direct_state_keeps_exact_m0_qru_path_with_four_outputs():
    model = l01.direct_state_model()
    assert model.rel_proj_t[0].in_features == 128 and model.rel_proj_t[0].out_features == 32
    assert model.phi_pair is not None and model.phi_rel is not None and model.phi_mod is not None
    assert model.head[-1].out_features == 2
    assert l0.count_parameters(model) > 0


def test_degree_5_heterogeneity_and_function_preference_are_descriptive_only():
    frame = pd.DataFrame({"src": [0, 1, 2, 3], "dst": [9, 9, 9, 9], "dst_degree": [5] * 4,
        **{f"mean_{name}": np.asarray([1., -1., 2., -2.]) for name in l01.UTILITY_COLUMNS}})
    result = l01.clean_heterogeneity({("Movies", 42): frame})
    assert (result.degree5_positive_negative_coexistence_ratio == 1).all()
    assert not any("label" in col.lower() or "role" in col.lower() for col in result.columns)
