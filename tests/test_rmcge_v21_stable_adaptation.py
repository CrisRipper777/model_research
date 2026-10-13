from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from src.models.rmcge_v21 import conserve_receiver_mass
from src.research.rmcge_v21_stable_adaptation import (
    apply_joint_trainability,
    apply_stage_trainability,
    compare_logits,
    finite_metrics,
    is_selector_name,
    is_stage1_selector_name,
    make_joint_optimizer,
    make_stage1_optimizer,
    metric_bundle,
    module_sha256,
    physical_degree_groups,
    select_main_candidates,
    selector_state_sha256,
)
from scripts.analyze_rmcge_v21_stable_adaptation import _append_warm_history
from scripts.run_rmcge_v21_stable_adaptation import _zero_test_attestation
from src.tasks.nc import _resolve_nc_eval_labels


class TinySelectorModel(nn.Module):
    def __init__(self, *, zero_score: bool = False):
        super().__init__()
        self.common = nn.Linear(3, 2)
        self.query_projection = nn.Linear(3, 2, bias=False)
        self.key_projection = nn.Linear(3, 2, bias=False)
        self.edge_target_modality_embedding = nn.Embedding(2, 2)
        self.edge_purpose_embedding = nn.Embedding(2, 2)
        self.purpose_edge_scorer = nn.Sequential(nn.Linear(2, 3), nn.GELU(), nn.Linear(3, 1))
        self.cross_delta_coefficients = nn.Parameter(torch.ones(1))
        if zero_score:
            nn.init.zeros_(self.purpose_edge_scorer[-1].weight)
            nn.init.zeros_(self.purpose_edge_scorer[-1].bias)

    def forward(self, x):
        query = self.query_projection(x)
        key = self.key_projection(x)
        score = self.purpose_edge_scorer(query + key)
        return self.common(x) + 0.1 * score * self.cross_delta_coefficients


def _candidate(stage, epoch, accuracy, f1, ce=1.0):
    return {
        "stage": stage,
        "epoch": epoch,
        "metrics": {"validation": {"accuracy": accuracy, "macro_f1": f1, "cross_entropy": ce}},
        "model_state": {},
        "head_state": {},
    }


def test_selector_parameter_names_are_scoped():
    assert is_selector_name("query_projection.weight")
    assert is_selector_name("purpose_edge_scorer.2.weight")
    assert is_selector_name("cross_delta_coefficients")
    assert not is_selector_name("delta_coefficients")
    assert not is_selector_name("experts.0.linear1.weight")


def test_stage1_trainable_names_include_c2_delta_only_for_c2():
    assert is_stage1_selector_name("query_projection.weight", "C1")
    assert not is_stage1_selector_name("cross_delta_coefficients", "C1")
    assert is_stage1_selector_name("cross_delta_coefficients", "C2")


def test_stage1_freezes_public_model_and_classifier():
    model, head = TinySelectorModel(), nn.Linear(2, 2)
    names = apply_stage_trainability(model, head, "C1")
    assert set(names) == {name for name, _ in model.named_parameters() if is_stage1_selector_name(name, "C1")}
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if not is_stage1_selector_name(name, "C1"))
    assert all(not parameter.requires_grad for parameter in head.parameters())


def test_stage1_eval_mode_still_backpropagates_to_scorer():
    model, head = TinySelectorModel(zero_score=True), nn.Linear(2, 2)
    apply_stage_trainability(model, head, "C1")
    model.eval()
    head.eval()
    x = torch.tensor([[1.0, 0.2, -0.5], [-0.3, 0.7, 0.4]])
    logits = head(model(x))
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    assert model.training is False
    assert loss.requires_grad
    loss.backward()
    assert model.purpose_edge_scorer[-1].weight.grad is not None
    assert model.purpose_edge_scorer[-1].weight.grad.norm() > 0


def test_stage1_optimizer_has_only_selectors_and_zero_state():
    model = TinySelectorModel()
    optimizer = make_stage1_optimizer(model, "C1", learning_rate=1.0e-3, weight_decay=1.0e-4)
    ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for name, parameter in model.named_parameters() if is_stage1_selector_name(name, "C1")}
    assert ids == expected
    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1.0e-4)


def test_stage1_w1_does_not_train_cross_delta_but_w2_does():
    c1, c2 = TinySelectorModel(), TinySelectorModel()
    o1 = make_stage1_optimizer(c1, "C1", learning_rate=1e-3, weight_decay=1e-4)
    o2 = make_stage1_optimizer(c2, "C2", learning_rate=1e-3, weight_decay=1e-4)
    p1 = {id(parameter) for group in o1.param_groups for parameter in group["params"]}
    p2 = {id(parameter) for group in o2.param_groups for parameter in group["params"]}
    assert id(c1.cross_delta_coefficients) not in p1
    assert id(c2.cross_delta_coefficients) in p2


def test_stage1_updates_selector_without_changing_public_state():
    model, head = TinySelectorModel(zero_score=True), nn.Linear(2, 2)
    apply_stage_trainability(model, head, "C1")
    common_before = module_sha256(model, exclude_selectors=True)
    scorer_before = model.purpose_edge_scorer[-1].weight.detach().clone()
    optimizer = make_stage1_optimizer(model, "C1", learning_rate=1e-2, weight_decay=0.0)
    x = torch.tensor([[1.0, 0.2, -0.5], [-0.3, 0.7, 0.4]])
    nn.functional.cross_entropy(head(model(x)), torch.tensor([0, 1])).backward()
    optimizer.step()
    assert module_sha256(model, exclude_selectors=True) == common_before
    assert not torch.equal(scorer_before, model.purpose_edge_scorer[-1].weight)


def test_joint_unfreezes_all_parameter_groups_and_updates_common_model():
    model, head = TinySelectorModel(), nn.Linear(2, 2)
    apply_stage_trainability(model, head, "C2")
    apply_joint_trainability(model, head)
    before = model.common.weight.detach().clone()
    optimizer = make_joint_optimizer(model, head, common_lr=1e-3, classifier_lr=1e-3, selector_lr=2e-3, weight_decay=0.0)
    x = torch.randn(6, 3)
    nn.functional.cross_entropy(head(model(x)), torch.tensor([0, 1, 0, 1, 1, 0])).backward()
    optimizer.step()
    assert not torch.equal(before, model.common.weight)
    assert not optimizer.state == {}


def test_joint_optimizer_has_matched_learning_rates_and_fresh_state():
    model, head = TinySelectorModel(), nn.Linear(2, 2)
    optimizer = make_joint_optimizer(model, head, common_lr=1e-4, classifier_lr=1e-4, selector_lr=5e-4, weight_decay=1e-4)
    assert not optimizer.state
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"common_model", "classifier", "selector"}
    assert groups["common_model"]["lr"] == pytest.approx(1e-4)
    assert groups["classifier"]["lr"] == pytest.approx(1e-4)
    assert groups["selector"]["lr"] == pytest.approx(5e-4)
    assert all(group["weight_decay"] == pytest.approx(1e-4) for group in groups.values())


def test_w0_joint_optimizer_has_no_selector_group():
    model, head = nn.Linear(3, 2), nn.Linear(2, 2)
    optimizer = make_joint_optimizer(model, head, common_lr=1e-4, classifier_lr=1e-4, selector_lr=5e-4, weight_decay=1e-4)
    assert {group["group_name"] for group in optimizer.param_groups} == {"common_model", "classifier"}


def test_metric_bundle_uses_requested_macro_class_set_and_zero_division():
    logits = torch.tensor([[8.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    target = torch.tensor([0, 1, 1])
    all_classes = metric_bundle(logits, target, [0, 1, 2])
    observed_classes = metric_bundle(logits, target, [0, 1])
    assert all_classes["accuracy"] == pytest.approx(2 / 3)
    assert all_classes["macro_f1"] == pytest.approx(observed_classes["macro_f1"] * 2 / 3)
    assert torch.isfinite(torch.tensor(all_classes["cross_entropy"]))


def test_metric_bundle_reports_ce_from_logits():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    result = metric_bundle(logits, labels, [0, 1])
    assert result["accuracy"] == 1.0
    assert result["cross_entropy"] == pytest.approx(float(nn.functional.cross_entropy(logits, labels)))


def test_metric_bundle_rejects_nonfinite_logits():
    with pytest.raises(FloatingPointError):
        metric_bundle(torch.tensor([[float("nan"), 1.0]]), torch.tensor([0]), [0, 1])


def test_finite_metrics_checks_nested_records():
    assert finite_metrics({"metric": 0.2, "nested": [0, 1]})
    assert not finite_metrics({"metric": float("inf")})


def test_t0_candidate_is_retained_when_joint_does_not_improve():
    t0 = _candidate("t0", 0, 0.8, 0.4)
    joint = _candidate("joint", 1, 0.80005, 0.6)
    acc, f1 = select_main_candidates([t0, joint], min_delta=1e-4)
    assert acc is t0
    assert f1 is joint


def test_best_accuracy_and_best_f1_are_independent():
    t0 = _candidate("t0", 0, 0.80, 0.40)
    by_acc = _candidate("joint", 3, 0.83, 0.42)
    by_f1 = _candidate("joint", 5, 0.81, 0.51)
    acc, f1 = select_main_candidates([t0, by_acc, by_f1])
    assert acc is by_acc
    assert f1 is by_f1


def test_stage1_checkpoint_is_rejected_as_main_candidate():
    with pytest.raises(ValueError, match="only joint epochs"):
        select_main_candidates([_candidate("t0", 0, 0.5, 0.5), _candidate("selector", 12, 0.9, 0.9)])


def test_initial_logits_comparison_respects_tolerance_and_shape():
    reference = torch.ones(3, 2)
    close = reference + 1e-6
    assert compare_logits(reference, close)["allclose"]
    with pytest.raises(ValueError, match="shapes"):
        compare_logits(reference, torch.ones(2, 2))


def test_state_fingerprint_changes_when_tensor_changes():
    model = nn.Linear(2, 2)
    before = module_sha256(model)
    with torch.no_grad():
        model.weight[0, 0] += 0.01
    assert module_sha256(model) != before


def test_public_state_hash_excludes_added_selector_parameters():
    model = TinySelectorModel()
    before = module_sha256(model, exclude_selectors=True)
    with torch.no_grad():
        model.query_projection.weight.add_(1.0)
    assert module_sha256(model, exclude_selectors=True) == before
    assert selector_state_sha256(model)


def test_degree_groups_are_disjoint_and_partition_physical_degree():
    degree = torch.tensor([0, 1, 2, 3, 4, 7, 8, 12])
    groups = physical_degree_groups(degree)
    fine = torch.stack([groups[key] for key in ("degree_0", "degree_1", "degree_2_3", "degree_4_7", "degree_ge8")])
    assert torch.equal(fine.sum(dim=0), torch.ones_like(degree))
    assert torch.equal(groups["degree_lt2"], groups["degree_0"] | groups["degree_1"])
    assert torch.equal(groups["degree_ge2"], degree >= 2)


def test_official_macro_f1_label_resolution_excludes_test_only_class():
    class Data:
        y = torch.tensor([0, 1, 2, 9])
        train_idx = torch.tensor([0, 1])
        val_idx = torch.tensor([2])
        test_idx = torch.tensor([3])
        num_classes = 10

    assert _resolve_nc_eval_labels(Data(), include_test=False) == [0, 1, 2]
    assert _resolve_nc_eval_labels(Data(), include_test=True) == [0, 1, 2, 9]


def test_test_attestation_flags_are_allowed_but_test_metric_fields_are_rejected():
    record = {"status": "completed"}
    _zero_test_attestation(record)
    assert record["test_evaluation"] is False
    assert record["test_label_values_indexed"] is False

    with pytest.raises(RuntimeError, match="unexpected Test metric fields"):
        _zero_test_attestation({"test_accuracy": 0.99})


def test_warm_history_analysis_uses_attempt_path_from_run_registry(tmp_path, monkeypatch):
    result_path = tmp_path / "attempt" / "result.json"
    result_path.parent.mkdir()
    result_path.write_text(json.dumps({
        "best_accuracy_epoch": 2,
        "best_f1_epoch": 3,
        "history": [{"stage": "joint", "epoch": 2}, {"stage": "joint", "epoch": 3}],
    }))
    import scripts.analyze_rmcge_v21_stable_adaptation as analyzer

    monkeypatch.setattr(analyzer, "ROOT", tmp_path)
    run_index = {("Movies", 42, condition): {"result_path": "attempt/result.json"} for condition in ("W0", "W1", "W2")}
    rows = []
    _append_warm_history("Movies", 42, rows, run_index)
    assert len(rows) == 6
    assert sum(row["is_best_accuracy_epoch"] for row in rows) == 3
    assert sum(row["is_best_f1_epoch"] for row in rows) == 3


def test_receiver_mass_conservation_and_zero_score_identity():
    physical = torch.tensor([0.2, 0.3, 0.4, 0.6])
    dst = torch.tensor([0, 0, 1, 1])
    scores = torch.zeros_like(physical)
    selected = conserve_receiver_mass(physical, scores, dst, num_nodes=3)
    assert torch.allclose(selected, physical, atol=1e-7, rtol=1e-7)
    before = torch.zeros(3).index_add_(0, dst, physical)
    after = torch.zeros(3).index_add_(0, dst, selected)
    assert torch.allclose(before, after, atol=1e-7, rtol=1e-7)


def test_receiver_mass_is_preserved_after_nonzero_selection():
    physical = torch.tensor([0.1, 0.3, 0.6, 0.2, 0.8])
    dst = torch.tensor([0, 0, 0, 1, 1])
    scores = torch.tensor([-0.8, 0.3, 0.9, -0.2, 0.6])
    selected = conserve_receiver_mass(physical, scores, dst, num_nodes=2)
    assert torch.allclose(torch.zeros(2).index_add_(0, dst, physical), torch.zeros(2).index_add_(0, dst, selected), atol=1e-6)


def test_t0_initialization_audit_records_actual_triplet_equivalence():
    path = "research/rmcge_v21_stable_adaptation/data/initialization_audit.json"
    try:
        with open(path, encoding="utf-8") as stream:
            audit = json.load(stream)
    except FileNotFoundError:
        pytest.skip("actual GPU t0 audit is produced by the gated preflight")
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        row = audit["pairs"][f"{dataset}/seed_42"]
        assert row["status"] == "passed"
        assert row["public_initial_state_exact"] is True
        assert row["classifier_initial_state_exact"] is True
        assert row["new_w1_w2_parameter_state_exact"] is True
        assert row["t0_logits"]["W1_vs_W0"]["allclose"] is True
        assert row["t0_logits"]["W2_vs_W0"]["allclose"] is True
        assert row["test_evaluation"] is False
