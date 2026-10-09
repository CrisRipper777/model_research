from __future__ import annotations

from collections import defaultdict

import pytest
import torch
from torch import nn

from src.research.message_effect_estimator import MessageEffectEstimator
from src.research.message_effect_replay import (
    FrozenHostCache,
    estimator_receiver_split,
    receiver_degree_quartile_sample,
    replay_local_basis_changes,
    sample_receiver_edges,
)
from src.research.message_effect_v5a1 import (
    TAUS,
    matched_effect_from_losses,
    matched_local_baseline,
    matched_old_effect_diagnostics,
    old_effect_from_losses,
    oracle_group_rows,
    selection_group_rows,
    sensitivity_diagnostics,
)
from src.research.message_effect_v5a1_estimator import (
    SingleTargetMessageEffectEstimator,
    make_edge_rank_pairs,
    pair_weighted_mean,
    ranknet_pair_loss,
    receiver_equal_mean,
    receiver_equal_row_weights,
    train_quantiles,
)
from src.research.message_effect_v5a1_training import ranker_metrics, ranker_sensitivity
from src.research.message_effect_v5a1_training import _save_checkpoint
from scripts import run_message_effect_v5a1 as v5a1_runner
from scripts.run_message_effect_v5a1 import _effect_strata, _pair_gap_strata, _selection_for_ranker


class TinyReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.expert_feature_scale = 0.0
        self.fusion_skip = nn.Linear(4, 2)
        self.fusion_linear1 = nn.Linear(4, 2)
        self.fusion_linear2 = nn.Linear(2, 2)
        self.fusion_dropout = nn.Identity()
        self.fusion_norm = nn.LayerNorm(2)


def tiny_cache():
    torch.manual_seed(19)
    n, h, k, classes = 4, 2, 4, 3
    model = TinyReplayModel().eval()
    classifier = nn.Linear(h, classes).eval()
    prior = {m: torch.randn(n, h) for m in ("text", "visual")}
    basis = {m: torch.randn(k, n, h) for m in ("text", "visual")}
    route = {m: torch.softmax(torch.randn(n, k), dim=-1) for m in ("text", "visual")}
    strength = {m: torch.full((n,), 0.25) for m in ("text", "visual")}
    # The replay implementation only consumes the cache fields below plus the dataclass placeholders.
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 1, 2, 3]])
    cache = FrozenHostCache(
        dataset="tiny", model=model, classifier=classifier,
        src=edge_index[0], dst=edge_index[1], edge_weight=torch.tensor([.5, .5, 1., 1.]),
        active=torch.ones(n, dtype=torch.bool), indegree=torch.tensor([0, 2, 1, 1]),
        prior=prior, raw_states={m: torch.randn(k, n, h) for m in prior}, raw_basis=basis,
        denominators={m: torch.ones(k, h) for m in prior}, alpha=torch.eye(k), route_weights=route,
        strength=strength, modality_outputs={}, raw_mixture={}, z=torch.empty((n, h)),
        logits=torch.empty((n, classes)), probabilities=torch.empty((n, classes)),
        semantic_cosine={m: torch.zeros(edge_index.size(1)) for m in prior},
        role_score={m: torch.zeros(edge_index.size(1)) for m in prior},
        support_flag={m: torch.ones(edge_index.size(1), dtype=torch.bool) for m in prior},
        router_cosine={m: torch.zeros(edge_index.size(1)) for m in prior},
        router_reliability={m: torch.ones(edge_index.size(1)) for m in prior},
        receiver_mass=torch.tensor([0., 1., 1., 1.]), device=torch.device("cpu"),
    )
    base = replay_local_basis_changes(cache, torch.arange(n), changes=[])
    cache.modality_outputs = base["modality_outputs"]
    cache.z = base["z"]
    cache.logits = base["logits"]
    cache.probabilities = cache.logits.softmax(-1)
    return cache


def mini_rows():
    rows = []
    vals = {
        (1, "text", 1): [-.2, .1], (1, "visual", 1): [.3, -.1],
        (1, "text", 2): [-.3, .2], (1, "visual", 2): [.2, -.2],
        (1, "text", 3): [-.1, .3], (1, "visual", 3): [.1, -.3],
        (1, "text", 4): [-.4, .4], (1, "visual", 4): [.4, -.4],
    }
    for (_, modality, hop), effects in vals.items():
        for edge, effect in enumerate(effects, start=10):
            rows.append({"receiver_id": 1, "directed_edge_index": edge, "modality": modality,
                         "hop": hop, "delta_delete_matched": effect, "delta_comp_matched": effect / 2})
    return rows


def test_same_host_and_sample_selection_is_deterministic():
    indegree = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4])
    train = torch.arange(9)
    first = receiver_degree_quartile_sample(train, indegree, max_receivers=6, seed=2027)
    second = receiver_degree_quartile_sample(train, indegree, max_receivers=6, seed=2027)
    assert torch.equal(first[0], second[0]) and first[1] == second[1]
    dst = torch.tensor([1, 1, 1, 1, 1, 2, 2])
    assert sample_receiver_edges("D", 1, dst, seed=2027)[0].tolist() == sample_receiver_edges("D", 1, dst, seed=2027)[0].tolist()


def test_matched_baseline_calls_zero_change_replay(monkeypatch):
    cache = type("Cache", (), {"device": torch.device("cpu")})()
    seen = {}
    def fake_replay(c, receivers, changes=()):
        seen["changes"] = changes
        logits = torch.tensor([[2., -1.], [0., 1.]])
        return {"logits": logits, "z": logits + 1}
    monkeypatch.setattr("src.research.message_effect_v5a1.replay_local_basis_changes", fake_replay)
    result = matched_local_baseline(cache, torch.tensor([2, 3]), torch.tensor([0, 1]))
    assert seen["changes"] == []
    assert torch.isfinite(result["loss"]).all()


def test_matched_baseline_local_logits_reproduce_cached_host():
    cache = tiny_cache()
    receivers = torch.arange(4)
    replay = replay_local_basis_changes(cache, receivers, changes=[])
    base = matched_local_baseline(cache, receivers, torch.tensor([0, 1, 2, 0]))
    torch.testing.assert_close(base["logits"], cache.logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(replay["z"], cache.z, atol=1e-6, rtol=1e-6)


def test_matched_delta_is_modified_loss_minus_local_loss():
    logits = torch.tensor([[1.2, -.2, .1], [.1, .4, -.5]])
    labels = torch.tensor([0, 2])
    baseline = torch.tensor([.8, 1.4])
    got = matched_effect_from_losses(logits, labels, baseline)
    torch.testing.assert_close(got, torch.nn.functional.cross_entropy(logits, labels, reduction="none") - baseline)


def test_old_delta_remains_available():
    logits = torch.tensor([[.2, -.3], [1., 0.]])
    labels = torch.tensor([0, 1])
    old = old_effect_from_losses(logits, labels, torch.tensor([.6, .9]))
    assert torch.isfinite(old).all()


def test_matched_minus_old_equals_full_loss_minus_local_loss():
    logits = torch.tensor([[.2, -.3], [1., 0.]])
    labels = torch.tensor([0, 1])
    local, full = torch.tensor([.6, .9]), torch.tensor([.7, .8])
    matched = matched_effect_from_losses(logits, labels, local)
    old = old_effect_from_losses(logits, labels, full)
    torch.testing.assert_close(matched - old, full - local)
    assert matched_old_effect_diagnostics(matched.tolist(), old.tolist())["n"] == 2


def test_delete_matched_effect_is_finite():
    cache = tiny_cache()
    receiver, label = torch.tensor([1]), torch.tensor([0])
    base = matched_local_baseline(cache, receiver, label)
    replay = replay_local_basis_changes(cache, receiver, [{"modality": "text", "hop": 1, "delta": torch.ones(1, 2) * .1}])
    assert torch.isfinite(matched_effect_from_losses(replay["logits"], label, base["loss"])).all()


def test_comp_matched_effect_finite_when_mass_retained():
    cache = tiny_cache()
    receiver, label = torch.tensor([1]), torch.tensor([0])
    base = matched_local_baseline(cache, receiver, label)
    replay = replay_local_basis_changes(cache, receiver, [{"modality": "visual", "hop": 2, "delta": torch.ones(1, 2) * -.02}])
    assert torch.isfinite(matched_effect_from_losses(replay["logits"], label, base["loss"])).all()


def test_matched_bundle_subtracts_same_local_baseline():
    cache = tiny_cache()
    receiver, label = torch.tensor([1]), torch.tensor([0])
    base = matched_local_baseline(cache, receiver, label)
    changes = [{"modality": "text", "hop": 1, "delta": torch.ones(1, 2) * .05},
               {"modality": "visual", "hop": 1, "delta": torch.ones(1, 2) * -.02}]
    bundle = replay_local_basis_changes(cache, receiver, changes)
    effect = matched_effect_from_losses(bundle["logits"], label, base["loss"])
    assert effect.shape == (1,) and torch.isfinite(effect).all()


def test_sensitivity_thresholds_are_fixed_and_ordered():
    assert TAUS == (0., 1e-6, 1e-5, 1e-4, 1e-3)


def test_text_visual_robust_sign_eligibility_respects_tau():
    rows = [{"receiver_id": 1, "directed_edge_index": 2, "hop": 1, "modality": "text", "delta_delete_matched": 1e-7},
            {"receiver_id": 1, "directed_edge_index": 2, "hop": 1, "modality": "visual", "delta_delete_matched": -2e-4}]
    result = sensitivity_diagnostics(rows, "delete")
    tau0 = next(r for r in result if r["diagnostic"] == "text_visual_sign" and r["tau"] == 0)
    tau1 = next(r for r in result if r["diagnostic"] == "text_visual_sign" and r["tau"] == 1e-6)
    assert tau0["eligible_pairs"] == 1 and tau0["sign_disagreement_fraction"] == 1
    assert tau1["eligible_pairs"] == 0


def test_hop_strong_mixed_sign_uses_both_threshold_sides():
    rows = [{"receiver_id": 1, "directed_edge_index": 2, "modality": "text", "hop": k,
             "delta_delete_matched": v} for k, v in enumerate([-.2, -.01, .03, .4], 1)]
    result = sensitivity_diagnostics(rows, "delete")
    tau = next(r for r in result if r["diagnostic"] == "hop_robust_sign" and r["tau"] == 1e-3)
    assert tau["strong_mixed_sign_groups"] == 1
    assert tau["strong_mixed_sign_fraction"] == 1


def test_edge_pair_gap_threshold_counts_only_eligible_pairs():
    rows = [{"receiver_id": 1, "directed_edge_index": e, "modality": "text", "hop": 1,
             "delta_delete_matched": v} for e, v in ((1, 0.), (2, 1e-4), (3, 2e-3))]
    result = sensitivity_diagnostics(rows, "delete")
    tau = next(r for r in result if r["diagnostic"] == "edge_pair_gap" and r["tau"] == 1e-3)
    assert tau["candidate_pairs"] == 3 and tau["eligible_pairs"] == 2


def test_train_derived_magnitude_quantiles_ignore_holdout_by_construction():
    train = torch.tensor([1., 2., 3., 4.])
    cut = train_quantiles(train)
    altered_holdout = torch.tensor([1e9, -1e9])
    assert cut == train_quantiles(train)
    assert cut == [1.75, 2.5, 3.25]
    assert altered_holdout.numel() and cut == train_quantiles(train)


def test_receiver_split_matches_v5a_stable_hash_split():
    ids = torch.tensor([2, 4, 9, 12, 20])
    assert estimator_receiver_split("Movies", ids, seed=2027) == estimator_receiver_split("Movies", ids, seed=2027)
    assert set.union(*estimator_receiver_split("Movies", ids).values()) == set(ids.tolist())


@pytest.mark.parametrize("variant", ["E0_heuristic", "E1_unimodal_pair", "E2_multimodal_pair", "E3_host_context"])
def test_single_target_estimator_outputs_one_column(variant):
    model = SingleTargetMessageEffectEstimator(variant, 18, host_dim=8 if variant == "E3_host_context" else None)
    kwargs = {}
    if variant == "E1_unimodal_pair": kwargs["pair_active"] = torch.randn(3, 1024)
    if variant in {"E2_multimodal_pair", "E3_host_context"}:
        kwargs.update(pair_text=torch.randn(3, 1024), pair_visual=torch.randn(3, 1024))
    if variant == "E3_host_context": kwargs["host_context"] = torch.randn(3, 8)
    assert model(torch.randn(3, 18), **kwargs).shape == (3, 1)


def test_delete_and_comp_single_target_models_do_not_share_parameters():
    delete = SingleTargetMessageEffectEstimator("E0_heuristic", 18)
    comp = SingleTargetMessageEffectEstimator("E0_heuristic", 18)
    assert not any(a.data_ptr() == b.data_ptr() for a, b in zip(delete.parameters(), comp.parameters()))


def test_matched_multi_target_reuses_v5a_two_output_architecture():
    old = MessageEffectEstimator("E2_multimodal_pair", 18)
    matched = MessageEffectEstimator("E2_multimodal_pair", 18)
    assert old.predictor[-1].out_features == matched.predictor[-1].out_features == 2
    assert matched.pair_text_encoder is not None and matched.pair_visual_encoder is not None


def test_pointwise_train_weights_equalize_receiver_total_mass():
    receiver = torch.tensor([1, 1, 1, 2, 3, 3])
    weights = receiver_equal_row_weights(receiver)
    mass = defaultdict(float)
    for r, w in zip(receiver.tolist(), weights.tolist()): mass[r] += w
    assert max(mass.values()) - min(mass.values()) < 1e-6


def test_pointwise_validation_mean_weights_receivers_equally():
    values = torch.tensor([1., 3., 7., 9., 11.])
    receivers = torch.tensor([1, 1, 2, 2, 2])
    assert float(receiver_equal_mean(values, receivers)) == pytest.approx((2. + 9.) / 2)


def test_rank_pairs_are_limited_to_same_receiver_modality_hop():
    rows = mini_rows()
    target = torch.tensor([r["delta_delete_matched"] for r in rows])
    pairs = make_edge_rank_pairs(rows, target)
    for a, b in zip(pairs["left"].tolist(), pairs["right"].tolist()):
        assert (rows[a]["receiver_id"], rows[a]["modality"], rows[a]["hop"]) == (rows[b]["receiver_id"], rows[b]["modality"], rows[b]["hop"])


def test_four_edges_generate_at_most_six_unordered_pairs():
    rows = [dict(receiver_id=1, modality="text", hop=1, directed_edge_index=e) for e in range(4)]
    pairs = make_edge_rank_pairs(rows, torch.tensor([0., 1., 2., 3.]))
    assert pairs["left"].numel() == 6


def test_pair_direction_for_delta_a_b():
    delta_a, delta_b = -1., 2.
    order = torch.tensor([1. if delta_b - delta_a > 0 else -1.])
    score_a, score_b = torch.tensor([-2.]), torch.tensor([3.])
    assert order.item() == 1 and score_a.item() < score_b.item()


def test_ranknet_loss_direction_is_correct():
    order = torch.tensor([1.])
    good = ranknet_pair_loss(torch.tensor([-2.]), torch.tensor([2.]), order)
    bad = ranknet_pair_loss(torch.tensor([2.]), torch.tensor([-2.]), order)
    assert good.item() < bad.item()


def test_exact_target_ties_are_skipped_in_rank_pairs():
    rows = [dict(receiver_id=1, modality="text", hop=1, directed_edge_index=e) for e in range(3)]
    pairs = make_edge_rank_pairs(rows, torch.tensor([1., 1., 2.]))
    assert pairs["left"].numel() == 2


def test_each_edge_group_pair_weights_sum_to_one():
    rows = [dict(receiver_id=1, modality="text", hop=1, directed_edge_index=e) for e in range(4)]
    pairs = make_edge_rank_pairs(rows, torch.tensor([0., 1., 2., 3.]))
    assert pairs["weight"].sum().item() == pytest.approx(1.)


def test_rank_validation_pairs_can_be_restricted_to_validation_receivers():
    rows = [dict(receiver_id=r, modality="text", hop=1, directed_edge_index=e) for r in (1, 2) for e in range(2)]
    target = torch.arange(4, dtype=torch.float32)
    val = make_edge_rank_pairs(rows, target, {2})
    assert set(rows[int(i)]["receiver_id"] for i in val["left"]) == {2}


def test_holdout_receivers_are_absent_from_validation_pair_assembly():
    rows = [dict(receiver_id=r, modality="text", hop=1, directed_edge_index=e) for r in (1, 2, 3) for e in range(2)]
    target = torch.arange(6, dtype=torch.float32)
    val = make_edge_rank_pairs(rows, target, {2})
    assert 3 not in {rows[int(i)]["receiver_id"] for i in val["left"].tolist() + val["right"].tolist()}


def test_ranker_metrics_omit_cross_group_calibration_fields():
    rows = mini_rows()
    truth = [r["delta_delete_matched"] for r in rows]
    metrics, _ = ranker_metrics(rows, truth, truth, dataset="D", target="delete", model="Ranker", estimator="E0", seed=0)
    assert not ({"spearman", "mae", "modality_ordering_accuracy", "hop_pairwise_accuracy"} & set(metrics))


def test_random_expected_effect_equals_group_mean():
    rows = [dict(receiver_id=1, modality="text", hop=1, directed_edge_index=i,
                 delta_delete_matched=v) for i, v in enumerate([-.3, .1, .8])]
    _, summaries = oracle_group_rows(rows, "delete")
    edge = next(r for r in summaries if r["granularity"] == "edge")
    assert edge["random_expected_mean"] == pytest.approx((-.3 + .1 + .8) / 3)


def test_oracle_effect_equals_group_minimum():
    rows = [dict(receiver_id=1, modality="text", hop=1, directed_edge_index=i,
                 delta_delete_matched=v) for i, v in enumerate([-.3, .1, .8])]
    _, summaries = oracle_group_rows(rows, "delete")
    edge = next(r for r in summaries if r["granularity"] == "edge")
    assert edge["oracle_effect_mean"] == pytest.approx(-.3)


def test_model_gain_vs_random_formula_is_correct():
    rows = [{"receiver_id": 1, "modality": "text", "hop": 1, "directed_edge_index": i,
             "delta_delete_matched": v} for i, v in enumerate([-.2, .1])]
    outputs, _ = selection_group_rows(rows, [.0, 1.], "delete", model_name="P", dataset="D", estimator="E0", seed=0)
    assert outputs[0]["model_gain_vs_random"] == pytest.approx((-.2 + .1) / 2 - (-.2))


def test_oracle_regret_formula_is_correct():
    rows = [{"receiver_id": 1, "modality": "text", "hop": 1, "directed_edge_index": i,
             "delta_delete_matched": v} for i, v in enumerate([-.2, .1])]
    outputs, _ = selection_group_rows(rows, [1., 0.], "delete", model_name="P", dataset="D", estimator="E0", seed=0)
    assert outputs[0]["oracle_regret"] == pytest.approx(.1 - (-.2))


def test_top1_chance_equals_inverse_group_size():
    rows = [{"receiver_id": 1, "modality": "text", "hop": 1, "directed_edge_index": i,
             "delta_delete_matched": float(i)} for i in range(4)]
    result, _ = selection_group_rows(rows, [0., 1., 2., 3.], "delete", model_name="P", dataset="D", estimator="E0", seed=0)
    assert result[0]["top1_chance"] == .25


def test_top1_excess_is_raw_agreement_minus_chance():
    rows = [{"receiver_id": 1, "modality": "text", "hop": 1, "directed_edge_index": i,
             "delta_delete_matched": float(i)} for i in range(2)]
    result, _ = selection_group_rows(rows, [0., 1.], "delete", model_name="P", dataset="D", estimator="E0", seed=0)
    assert result[0]["top1_excess"] == pytest.approx(result[0]["top1_agreement"] - .5)


def test_pairwise_chance_excess_is_accuracy_minus_half():
    rows = mini_rows()
    truth = [r["delta_delete_matched"] for r in rows]
    sensitivity = ranker_sensitivity(rows, truth, truth, [0.], dataset="D", target="delete", model="Ranker", estimator="E0", seed=0)[0]
    assert sensitivity["pairwise_excess_over_0.5"] == pytest.approx(sensitivity["pairwise_accuracy"] - .5)


def test_no_validation_or_test_label_fields_enter_target_functions():
    # The matched target API accepts only explicitly supplied receiver labels and baseline/replay logits.
    cache = tiny_cache()
    result = matched_local_baseline(cache, torch.tensor([1]), torch.tensor([0]))
    assert result["loss"].numel() == 1
    assert not any(k in result for k in ("val_labels", "test_labels", "test_metrics"))


def test_forward_backward_are_finite_for_single_target_and_ranknet():
    model = SingleTargetMessageEffectEstimator("E0_heuristic", 18)
    pred = model(torch.randn(5, 18))
    loss = pred.square().mean() + ranknet_pair_loss(pred[:2, 0], pred[2:4, 0], torch.tensor([1., -1.])).mean()
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_writer_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "nested" / "checkpoints" / "model.pt"
    _save_checkpoint(str(path), {"value": torch.tensor([1.])})
    assert torch.load(path, weights_only=False)["value"].item() == 1.


def test_train_derived_absolute_magnitude_strata_emit_holdout_metrics():
    rows = []
    targets = []
    for receiver, values in ((1, [-.4, -.2, .2, .4]), (2, [-.3, -.1, .1, .3])):
        for modality in ("text", "visual"):
            for hop, value in enumerate(values, 1):
                rows.append({"receiver_id": receiver, "modality": modality, "hop": hop,
                             "directed_edge_index": 10 + hop})
                targets.append(value)
    features = {"target_delete": torch.tensor(targets)}
    pred = {"delete": targets}
    result = _effect_strata(rows, features, pred, {1}, {2}, "tiny", "S", "E0", 0, "delete")
    assert len(result) == 4 and {r["stratum_name"] for r in result} == {"Q0-Q25", "Q25-Q50", "Q50-Q75", "Q75-Q100"}
    assert all(r["holdout_rows"] >= 0 for r in result)


def test_train_derived_edge_gap_strata_emit_holdout_pair_metrics():
    rows = [
        {"receiver_id": receiver, "modality": "text", "hop": 1, "directed_edge_index": edge}
        for receiver in (1, 2) for edge in (10, 11, 12, 13)
    ]
    features = {"target_delete": torch.tensor([0., .1, .2, .3, 0., .2, .4, .6])}
    pred = features["target_delete"].tolist()
    result = _pair_gap_strata(rows, features, pred, {1}, {2}, "tiny", "S", "E0", 0, "delete")
    assert len(result) == 4 and all(r["stratum_type"] == "edge_pair_effect_gap" for r in result)


def test_ranker_selection_accepts_target_metadata_without_duplicate_argument():
    rows = [{"receiver_id": 1, "modality": "text", "hop": 1, "directed_edge_index": edge,
             "delta_delete_matched": effect} for edge, effect in ((10, -.1), (11, .2))]
    result, summary = _selection_for_ranker(rows, [-1., 1.], "delete", dataset="tiny", target="delete",
                                            estimator="E0", seed=0)
    assert result[0]["model"] == "Ranker" and summary["groups_n"] == 1


def test_dataset_summary_computes_old_matched_rank_correlation():
    rows = [
        {"delta_delete_old": -.2, "delta_delete_matched": -.1, "delta_comp_old": .1, "delta_comp_matched": .2},
        {"delta_delete_old": .1, "delta_delete_matched": .2, "delta_comp_old": -.2, "delta_comp_matched": -.1},
    ]
    points = [{"dataset": "tiny", "target": target, "model": model, "edge_pairwise_accuracy": .55}
              for target in ("delete", "comp") for model in ("M", "S")]
    ranks = [{"dataset": "tiny", "target": target, "edge_pairwise_accuracy": .52,
              "edge_top1_excess_over_chance": .01} for target in ("delete", "comp")]
    selections = [{"dataset": "tiny", "target": target, "model": "Ranker",
                   "model_gain_vs_random": .001, "oracle_regret": .002} for target in ("delete", "comp")]
    oracles = [{"dataset": "tiny", "target": target, "granularity": granularity,
                "oracle_gain_vs_random_mean": .003} for target in ("delete", "comp")
               for granularity in ("edge", "modality", "hop")]
    result = v5a1_runner._dataset_summary("tiny", rows, points, ranks, selections, oracles)
    assert len(result) == 2 and result[0]["old_matched_spearman"] == pytest.approx(1.)


def test_report_writer_formats_csv_loaded_metrics(tmp_path, monkeypatch):
    research = tmp_path / "research"
    data = research / "data"
    data.mkdir(parents=True)
    monkeypatch.setattr(v5a1_runner, "RESEARCH", research)
    monkeypatch.setattr(v5a1_runner, "DATA_DIR", data)
    v5a1_runner._write_csv(data / "dataset_summary.csv", [{
        "dataset": "Movies", "target": "delete", "old_matched_spearman": .8,
        "pointwise_M_edge_pairwise_mean": .52, "pointwise_S_edge_pairwise_mean": .53,
        "ranker_edge_pairwise_mean": .51, "ranker_top1_excess_mean": .01,
        "ranker_model_gain_vs_random_mean": .001, "ranker_oracle_regret_mean": .002,
        "edge_oracle_gain_mean": .003, "modality_oracle_gain_mean": .004,
        "hop_oracle_gain_mean": .005,
    }])
    v5a1_runner._write_csv(data / "matched_effect_diagnostics.csv", [{
        "dataset": "Movies", "target": "delete", "diagnostic": "matched_vs_old_effect",
        "old_matched_spearman": .8, "matched_minus_old_median_abs": .0001,
        "matched_minus_old_p95_abs": .001, "old_matched_sign_disagreement_fraction": .1,
    }])
    v5a1_runner._write_csv(data / "matched_effect_sensitivity.csv", [{
        "dataset": "Movies", "target": "delete", "diagnostic": "text_visual_sign", "tau": 0.,
        "eligible_fraction": 1., "sign_disagreement_fraction": .4,
    }, {"dataset": "Movies", "target": "delete", "diagnostic": "hop_robust_sign", "tau": 0.,
        "strong_mixed_sign_fraction": .5}])
    v5a1_runner._write_csv(data / "oracle_leverage.csv", [{
        "dataset": "Movies", "target": "delete", "granularity": "edge", "groups_n": 10,
        "oracle_gain_vs_random_mean": .003, "oracle_gain_vs_random_median": .002,
        "oracle_gain_vs_random_p90": .01, "fraction_gain_gt_0.0001": .8,
    }])
    v5a1_runner._finalize_report()
    report = (research / "REPORT.md").read_text()
    assert "Movies" in report and "Q1. Matched baseline effect" in report
