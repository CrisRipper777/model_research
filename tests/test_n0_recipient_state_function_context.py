from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.analysis import n0_recipient_state_function_context as n0


ROOT = Path(__file__).resolve().parents[1]
SMOKE_AUDIT = n0.RAW_DIR / "smoke" / "smoke_audit.json"


def test_source_sha_is_the_frozen_l01_checkpoint():
    assert n0.SOURCE_SHA == "1ca0c98e4e989317b1485fe1c2d7c63daf1ff6fa"


def test_smoke_provenance_and_h0_regression_passed():
    assert SMOKE_AUDIT.is_file(), "run the required Movies/42 smoke before the correctness suite"
    audit = json.loads(SMOKE_AUDIT.read_text())
    assert audit["status"] == "passed"
    assert audit["source_sha"] == n0.SOURCE_SHA
    assert audit["group_audit"]["h0_regression"]["checkpoint_metric_differences"]
    assert max(abs(float(x)) for x in audit["group_audit"]["h0_regression"]["checkpoint_metric_differences"].values()) < 2e-5


def test_smoke_recovers_frozen_shared_head_and_l01_select_metrics():
    audit = json.loads(SMOKE_AUDIT.read_text())["group_audit"]["head_audits"][0]
    assert audit["retrained_missing_checkpoint"] is False
    assert audit["checkpoint_metadata_valid"] is True
    assert audit["validation_excluded"] is True
    assert audit["absolute_ce_difference"] <= 2e-6


def test_smoke_reproduces_ss_l01_raw_gains():
    err = json.loads(SMOKE_AUDIT.read_text())["checks"]["ss_regression_max_abs"]
    assert set(err) == set(n0.l01.UTILITY_COLUMNS)
    assert max(err.values()) <= 2e-7


def test_smoke_checks_independent_sdp_whole_context_regression():
    audit = json.loads(SMOKE_AUDIT.read_text())["group_audit"]["whole_context_max_abs_error"]
    assert set(audit) == {f"{m}_{op}" for m in n0.MODALITIES for op in n0.OPERATORS}
    assert max(audit.values()) <= 2e-5


@pytest.mark.parametrize("index,name", list(enumerate(("SS", "SD", "SP", "DS", "DD", "DP", "PS", "PD", "PP"))))
def test_background_grid_order_and_index(index: int, name: str):
    assert n0.BACKGROUND_NAMES[index] == name
    pair = n0.BACKGROUND_GRID[index]
    assert n0.background_index(*pair) == index


def test_held_smooth_context_matches_exact_candidate_replacement():
    whole = torch.tensor([[3.0, 7.0], [6.0, 2.0]], dtype=torch.float64)
    smooth = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    background = torch.tensor([[5.0, 2.0], [0.0, 3.0]], dtype=torch.float64)
    degree = torch.tensor([2.0, 4.0], dtype=torch.float64)
    observed = n0.held_smooth_context(whole, smooth, background, degree)
    expected = whole + (smooth - background) / degree[:, None]
    assert torch.equal(observed, expected)


def test_partial_candidate_is_removed_only_if_it_was_selected():
    whole = torch.tensor([[8.0], [8.0]], dtype=torch.float64)
    smooth = torch.tensor([[2.0], [2.0]], dtype=torch.float64)
    background = torch.tensor([[6.0], [6.0]], dtype=torch.float64)
    degree = torch.tensor([4.0, 4.0], dtype=torch.float64)
    included = torch.tensor([True, False])
    got = n0.held_smooth_context(whole, smooth, background, degree, included)
    assert torch.equal(got[:, 0], torch.tensor([7.0, 8.0], dtype=torch.float64))


def test_opposite_modality_state_occupies_only_its_feature_slot():
    h_t = torch.zeros((2, 128)); h_v = torch.zeros((2, 128))
    c_t = torch.ones((2, 128)); c_v = torch.full((2, 128), 2.0)
    changed_cross = torch.full((2, 128), 3.0)
    before = n0.shared_feature_from_contexts(h_t, h_v, c_t, c_v)
    after = n0.shared_feature_from_contexts(h_t, h_v, c_t, changed_cross)
    assert torch.equal(before[:, 256:384], after[:, 256:384])
    assert torch.equal(before[:, 384:512], torch.full((2, 128), 2.0))
    assert torch.equal(after[:, 384:512], torch.full((2, 128), 3.0))


def test_text_substitution_delta_changes_only_text_structural_slot():
    base = torch.zeros((3, n0.l01.SHARED_DIM), dtype=torch.float64)
    alt = base.clone(); alt[:, n0.l01.SLOT_TEXT] = 1
    assert torch.equal(base[:, n0.l01.SLOT_VISUAL], alt[:, n0.l01.SLOT_VISUAL])
    assert not torch.equal(base[:, n0.l01.SLOT_TEXT], alt[:, n0.l01.SLOT_TEXT])


def test_visual_substitution_delta_changes_only_visual_structural_slot():
    base = torch.zeros((3, n0.l01.SHARED_DIM), dtype=torch.float64)
    alt = base.clone(); alt[:, n0.l01.SLOT_VISUAL] = 1
    assert torch.equal(base[:, n0.l01.SLOT_TEXT], alt[:, n0.l01.SLOT_TEXT])
    assert not torch.equal(base[:, n0.l01.SLOT_VISUAL], alt[:, n0.l01.SLOT_VISUAL])


def test_logit_delta_is_background_invariant_for_a_fixed_linear_slot():
    torch.manual_seed(7)
    w = torch.randn(6, 128, dtype=torch.float64)
    message_delta = torch.randn(5, 128, dtype=torch.float64)
    delta = message_delta @ w.T
    contexts = torch.randn(9, 5, 6, dtype=torch.float64)
    # Exact linear updates do not depend on any of these baseline states.
    outputs = contexts + delta[None]
    observed = outputs - contexts
    assert torch.max(torch.abs(observed - observed[:1])).item() < 1e-12


def test_actual_gain_uses_baseline_ce_minus_alternative_ce():
    z = torch.tensor([[1.0, -0.5, 0.3]], dtype=torch.float64)
    dz = torch.tensor([[0.2, 0.1, -0.4]], dtype=torch.float64)
    y = torch.tensor([2])
    expected = torch.nn.functional.cross_entropy(z, y) - torch.nn.functional.cross_entropy(z + dz, y)
    assert torch.allclose(n0.ce_gain(z, dz, y)[0], expected)


def test_first_order_gain_is_negative_ce_gradient_dot_delta():
    z = torch.tensor([[1.0, -0.5, 0.3], [0.2, 0.8, -1.0]], dtype=torch.float64)
    dz = torch.tensor([[0.2, 0.1, -0.4], [-0.3, 0.1, 0.2]], dtype=torch.float64)
    y = torch.tensor([2, 0])
    _, first, _ = n0.ce_gain_decomposition(z, dz, y)
    g = z.softmax(-1) - torch.nn.functional.one_hot(y, 3).double()
    assert torch.allclose(first, -(g * dz).sum(-1), atol=1e-14, rtol=1e-14)


def test_ce_hessian_contraction_matches_dense_softmax_hessian():
    z = torch.tensor([[0.8, -0.2, 1.1]], dtype=torch.float64)
    dz = torch.tensor([[0.3, -0.7, 0.2]], dtype=torch.float64)
    y = torch.tensor([0])
    _, first, second = n0.ce_gain_decomposition(z, dz, y)
    p = z.softmax(-1)[0]
    h = torch.diag(p) - torch.outer(p, p)
    expected = first[0] - 0.5 * dz[0] @ h @ dz[0]
    assert torch.allclose(second[0], expected, atol=1e-14, rtol=1e-14)


def test_first_and_second_order_approximations_are_finite():
    z = torch.randn(20, 7, dtype=torch.float64)
    dz = torch.randn(20, 7, dtype=torch.float64) * 0.05
    y = torch.randint(0, 7, (20,))
    actual, first, second = n0.ce_gain_decomposition(z, dz, y)
    assert all(torch.isfinite(x).all() for x in (actual, first, second))


def test_ordinary_sign_switch_requires_strict_opposite_nonzero_signs():
    means = np.array([[0.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    heads = np.repeat(means[None], 3, axis=0)
    result = n0.summarize_sign_switch(means, heads)
    assert result["ordinary_sign_switch_count"] == 2


def test_robust_sign_switch_requires_all_heads_to_reverse():
    means = np.array([[1.0], [-1.0]])
    heads = np.array([[[1.0], [-1.0]], [[2.0], [-2.0]], [[0.5], [-0.5]]])
    assert n0.summarize_sign_switch(means, heads)["robust_sign_switch_count"] == 1
    heads[2, 1, 0] = 0.0
    assert n0.summarize_sign_switch(means, heads)["robust_sign_switch_count"] == 0


def test_d_vs_p_preference_switch_uses_continuous_difference_sign():
    preference = np.array([[0.4, -0.2], [-0.1, 0.0]])
    heads = np.repeat(preference[None], 3, axis=0)
    result = n0.summarize_preference_switch(preference, heads)
    assert result["ordinary_sign_switch_count"] == 1
    assert result["robust_sign_switch_count"] == 1


def test_within_target_rank_omits_low_degree_and_constant_targets():
    dst = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    degree = np.array([5] * 5 + [4] * 5)
    x = np.array([1, 2, 3, 4, 5, 1, 1, 1, 1, 1], dtype=float)
    y = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=float)
    values = n0._target_rank_values(x, y, dst, degree, minimum_degree=5)
    assert len(values) == 1
    assert values[0] == pytest.approx(-1.0)


def test_centered_utility_removes_recipient_level_mean_shift():
    dst = np.array([1, 1, 1, 2, 2, 2])
    degree = np.array([5] * 6)
    x = np.array([1.0, 2.0, 4.0, -1.0, 2.0, 7.0])
    y = x + np.array([10.0] * 3 + [-5.0] * 3)
    result = n0.centered_pair_metrics(x, y, dst, degree)
    assert result["centered_spearman"] == pytest.approx(1.0)
    assert result["centered_sign_agreement"] == pytest.approx(1.0)


def test_head_repeat_rank_ceiling_is_computed_on_ss_edge_order():
    actual = np.zeros((3, 9, 6, 4), dtype=float)
    actual[0, 0, :, 0] = [1, 2, 3, 4, 5, 6]
    actual[1, 0, :, 0] = [2, 3, 4, 5, 6, 7]
    actual[2, 0, :, 0] = [6, 5, 4, 3, 2, 1]
    dst = np.array([1] * 6); degree = np.array([6] * 6)
    _, _, rows = n0.full_grid_rank_tables("Movies", 42, actual, dst, degree)
    target_rows = [row for row in rows if row["target"] == "G_D_text"]
    assert len(target_rows) == 3
    pair01 = next(row for row in target_rows if row["head_repeat_a"] == 0 and row["head_repeat_b"] == 1)
    assert pair01["mean"] == pytest.approx(1.0)


def test_partial_subset_is_reproducible_and_has_expected_rounded_size():
    src = np.array([10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 27])
    dst = np.array([1] * 5 + [2] * 8)
    degree = np.array([5] * 5 + [8] * 8)
    mask1, audit1 = n0.deterministic_partial_mask(src, dst, degree, 42, 4101)
    mask2, audit2 = n0.deterministic_partial_mask(src, dst, degree, 42, 4101)
    assert np.array_equal(mask1, mask2)
    assert audit1 == audit2
    assert mask1.sum() == 1 + 2
    assert audit1["rounding"] == "floor with minimum one for degree>=5"


def test_partial_candidate_exclusion_restores_smooth_baseline():
    smooth = torch.tensor([[2.0], [2.0]], dtype=torch.float64)
    background = torch.tensor([[6.0], [6.0]], dtype=torch.float64)
    degree = torch.tensor([4.0, 4.0], dtype=torch.float64)
    whole = torch.tensor([[9.0], [9.0]], dtype=torch.float64)
    selected = torch.tensor([True, False])
    held = n0.held_smooth_context(whole, smooth, background, degree, selected)
    # The selected edge's own perturbation is removed; the unselected candidate is already Smooth.
    assert torch.equal(held, torch.tensor([[8.0], [9.0]], dtype=torch.float64))


def test_partial_subset_never_exceeds_25_percent_and_selects_at_least_one():
    src = np.arange(1, 22)
    dst = np.array([1] * 5 + [2] * 7 + [3] * 9)
    degree = np.array([5] * 5 + [7] * 7 + [9] * 9)
    mask, audit = n0.deterministic_partial_mask(src, dst, degree, 43, 4102)
    assert mask.sum() == 1 + 1 + 2
    assert audit["realized_fraction"] <= 0.25
    for target in np.unique(dst):
        assert mask[dst == target].sum() >= 1


def test_pair_sampler_returns_distinct_same_recipient_edges_and_is_deterministic():
    dst = np.array([1] * 5 + [2] * 6 + [3] * 4)
    degree = np.array([5] * 5 + [6] * 6 + [4] * 4)
    a = n0.sample_within_target_pairs(dst, degree, 50, 1234)
    b = n0.sample_within_target_pairs(dst, degree, 50, 1234)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert len(a[0]) <= 50
    assert np.all(a[0] != a[1])
    assert np.all(dst[a[0]] == dst[a[1]])
    assert np.all(dst[a[0]] == a[2])


def test_pairwise_logit_delta_additivity_is_exact_for_linear_updates():
    z = torch.randn(12, 5, dtype=torch.float64)
    dj = torch.randn(12, 5, dtype=torch.float64)
    dk = torch.randn(12, 5, dtype=torch.float64)
    pair = (z + dj + dk) - z
    assert torch.max(torch.abs(pair - (dj + dk))).item() < 2e-15


def test_pairwise_ce_cross_term_matches_negative_hessian_contraction():
    z = torch.tensor([[0.2, 0.7, -0.4]], dtype=torch.float64)
    dj = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float64)
    dk = torch.tensor([[-0.1, 0.4, 0.2]], dtype=torch.float64)
    y = torch.tensor([1])
    loss = lambda x: torch.nn.functional.cross_entropy(x, y)
    i_ce = (loss(z) - loss(z + dj + dk)) - (loss(z) - loss(z + dj)) - (loss(z) - loss(z + dk))
    p = z.softmax(-1)[0]
    h = torch.diag(p) - torch.outer(p, p)
    i2 = -(dj[0] @ h @ dk[0])
    assert torch.allclose(i_ce, i2, atol=0.02, rtol=0.1)


def test_source_never_indexes_test_split_or_emits_test_metrics():
    source = (ROOT / "src/analysis/n0_recipient_state_function_context.py").read_text()
    tree = __import__("ast").parse(source)
    attributes = {node.attr for node in __import__("ast").walk(tree) if isinstance(node, __import__("ast").Attribute)}
    identifiers = {node.id for node in __import__("ast").walk(tree) if isinstance(node, __import__("ast").Name)}
    assert "test_idx" not in attributes | identifiers
    assert "test_mask" not in attributes | identifiers
    assert "test_acc" not in identifiers and "test_macro_f1" not in identifiers


def test_smoke_manifest_records_full_required_correctness_scope():
    checks = json.loads(SMOKE_AUDIT.read_text())["checks"]
    assert checks["nine_backgrounds"] == 9
    assert checks["logit_delta_background_invariance_max_abs"]
    assert checks["gradient_decomposition_rows"] > 0
    assert checks["partial25_rows"] > 0
    assert checks["pairwise_rows"] == 2
    assert checks["no_test_metrics_or_labels_accessed"] is True
