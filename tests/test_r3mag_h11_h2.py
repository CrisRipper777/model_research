import inspect

import torch
import torch.nn.functional as F

from src.analysis.r3mag_context_utils import (
    ResidualAdapter,
    ResponsePredictor,
    apply_bounded_residual,
    apply_signed_permutation,
    build_relation_context,
    coherent_control_donors,
    direction_geometry,
    gram_preservation_errors,
    make_signed_permutation,
    matched_context_shuffle,
)
from src.analysis.r3mag_h11_h2_context_screen import _stratified_response_split
from src.analysis.r3mag_h1_response_audit import _degree_and_prediction_buckets


def test_coherent_donor_is_one_nonself_node_shared_by_all_branches_and_orders():
    n = 48
    audit = torch.arange(12)
    degree_bucket = torch.tensor([i // 12 for i in range(n)])
    predicted = torch.tensor([i % 3 for i in range(n)])
    first, counts = coherent_control_donors(audit, degree_bucket, predicted, n, seed=310)
    second, _ = coherent_control_donors(audit, degree_bucket, predicted, n, seed=310)
    expanded = first[:, None, None].expand(-1, 2, 4)
    assert torch.equal(first, second)
    assert expanded.shape == (len(audit), 2, 4)
    assert torch.equal(expanded[:, 0, :], expanded[:, 1, :])
    assert not bool((first == audit).any())
    assert counts["self_donor_count"] == 0
    assert sum(counts.get(tier, 0) for tier in ("joint", "degree", "global")) == len(audit)


def test_coherent_donor_falls_back_from_joint_to_degree_when_needed():
    degree_bucket = torch.tensor([0, 0, 0, 1, 1, 1])
    predicted = torch.arange(6)
    targets = torch.tensor([0, 1, 2, 3])
    donors, counts = coherent_control_donors(targets, degree_bucket, predicted, 6, seed=19)
    assert counts.get("joint", 0) == 0
    assert counts["degree"] == len(targets)
    assert not bool((donors == targets).any())
    assert all(int(degree_bucket[d]) == int(degree_bucket[t]) for t, d in zip(targets, donors, strict=True))


def test_signed_permutation_is_orthogonal_and_preserves_all_gram_matrices():
    torch.manual_seed(2)
    directions = torch.randn(13, 4, 32)
    permutation, signs = make_signed_permutation(32, seed=1009)
    scrambled = apply_signed_permutation(directions, permutation, signs)
    assert torch.equal(torch.sort(permutation).values, torch.arange(32))
    assert set(signs.tolist()) == {-1.0, 1.0}
    assert torch.allclose(directions @ directions.transpose(1, 2), scrambled @ scrambled.transpose(1, 2), atol=2e-5, rtol=1e-6)
    own_geometry = direction_geometry(directions)
    scrambled_geometry = direction_geometry(scrambled)
    assert abs(own_geometry["mean_effective_rank"] - scrambled_geometry["mean_effective_rank"]) < 1e-5
    assert abs(own_geometry["mean_pairwise_cosine"] - scrambled_geometry["mean_pairwise_cosine"]) < 1e-5
    gram_error = gram_preservation_errors(directions, scrambled)
    assert gram_error["absolute_max"] < 2e-5
    assert gram_error["scale_normalized_relative_max"] < 1e-5


def test_relation_context_is_label_free_finite_and_zero_for_isolated_nodes():
    assert "labels" not in inspect.signature(build_relation_context).parameters
    torch.manual_seed(5)
    n, dim = 5, 7
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 3], [1, 0, 2, 1, 3, 3]])
    reps = [torch.randn(n, dim) for _ in range(4)]
    context, disagreement, meta = build_relation_context(edge_index, *reps)
    assert context.shape == (n, 25)
    assert disagreement.shape == (n,)
    assert torch.isfinite(context).all() and torch.isfinite(disagreement).all()
    assert torch.equal(context[3], torch.zeros(25))
    assert torch.equal(context[4], torch.zeros(25))
    assert meta["isolated_node_count"] == 2


def test_context_shuffle_is_deterministic_split_local_and_avoids_self_map():
    n = 30
    split = torch.tensor([2, 5, 8, 11, 14, 17, 20, 23])
    degree_bucket = torch.zeros(n, dtype=torch.long)
    predicted = torch.arange(n)
    a, counts = matched_context_shuffle(split, degree_bucket, predicted, seed=17)
    b, _ = matched_context_shuffle(split, degree_bucket, predicted, seed=17)
    assert torch.equal(a, b)
    assert set(a.tolist()).issubset(set(split.tolist()))
    assert not bool((a == split).any())
    assert counts["self_map_count"] == 0
    assert counts["split_size"] == len(split)


def test_context_shuffle_records_singleton_self_map_when_split_has_one_node():
    split = torch.tensor([7])
    buckets = torch.zeros(12, dtype=torch.long)
    preds = torch.zeros(12, dtype=torch.long)
    donors, counts = matched_context_shuffle(split, buckets, preds, seed=3)
    assert torch.equal(donors, split)
    assert counts["singleton_self_map"] == 1
    assert counts["self_map_count"] == 1


def test_response_train_predictor_split_is_fixed_disjoint_and_stratified():
    response = torch.arange(60)
    labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
    train, val, metadata = _stratified_response_split(response, labels, seed=20261007)
    train2, val2, metadata2 = _stratified_response_split(response, labels, seed=20261007)
    assert torch.equal(train, train2) and torch.equal(val, val2)
    assert metadata == metadata2
    assert not (set(train.tolist()) & set(val.tolist()))
    assert set(torch.cat([train, val]).tolist()) == set(response.tolist())
    for cls in (0, 1, 2):
        assert int((labels[train] == cls).sum()) == 16
        assert int((labels[val] == cls).sum()) == 4


def test_predictor_and_adapter_capacity_and_bounded_residual_frozen_host_gradients():
    torch.manual_seed(8)
    n, h, classes = 10, 8, 3
    receiver_dim, context_dim = 23, 25
    p1 = ResponsePredictor(receiver_dim + context_dim)
    p2 = ResponsePredictor(receiver_dim + context_dim)
    assert sum(p.numel() for p in p1.parameters()) == sum(p.numel() for p in p2.parameters())
    b1 = ResidualAdapter(receiver_dim + context_dim, response_dim=h)
    b2 = ResidualAdapter(receiver_dim + context_dim, response_dim=h)
    assert sum(p.numel() for p in b1.parameters()) == sum(p.numel() for p in b2.parameters())

    host = torch.nn.Sequential(torch.nn.LayerNorm(h), torch.nn.Linear(h, classes))
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    state_before = {k: v.detach().clone() for k, v in host.state_dict().items()}
    context_input = torch.randn(n, receiver_dim + context_dim)
    global_t, global_v = torch.randn(n, h), torch.randn(n, h)
    adapter = ResidualAdapter(receiver_dim + context_dim, response_dim=h)
    raw, gate_logits = adapter(context_input)
    corrected_t, corrected_v, gates, ratio = apply_bounded_residual(global_t, global_v, raw, gate_logits)
    logits = host(corrected_t) + host(corrected_v)
    loss = F.cross_entropy(logits, torch.arange(n) % classes)
    loss.backward()
    assert torch.isfinite(corrected_t).all() and torch.isfinite(corrected_v).all()
    assert torch.isfinite(gates).all() and torch.isfinite(ratio).all()
    assert float(ratio.max()) <= 0.2 + 1e-6
    assert all(parameter.grad is None for parameter in host.parameters())
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert all(torch.equal(state_before[k], host.state_dict()[k]) for k in state_before)


def test_zero_residual_is_finite_and_leaves_global_prior_unchanged():
    prior_t, prior_v = torch.randn(4, 8), torch.randn(4, 8)
    raw = torch.zeros(4, 2, 8)
    gate = torch.zeros(4, 2)
    corrected_t, corrected_v, gates, ratio = apply_bounded_residual(prior_t, prior_v, raw, gate)
    assert torch.isfinite(corrected_t).all() and torch.isfinite(corrected_v).all()
    assert torch.isfinite(gates).all() and torch.isfinite(ratio).all()
    assert torch.equal(corrected_t, prior_t) and torch.equal(corrected_v, prior_v)
    assert torch.equal(ratio, torch.zeros_like(ratio))
