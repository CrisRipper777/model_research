import torch
import torch.nn.functional as F

from src.analysis.r3mag_h1_response_audit import (
    _borrowed_directions,
    assert_disjoint_splits,
    evaluate_joint_action_utilities,
    evaluate_single_action_utilities,
    matched_control_donors,
    stratified_internal_partition,
)
from src.analysis.r3mag_response_core import (
    GlobalDualBranchResponseHost,
    R3MAGHostConfig,
    action_names,
    make_action_candidates,
    normalized_response_directions,
)


def _small_operator(n=12):
    edges = []
    for i in range(n):
        edges.extend([(i, i), (i, (i + 1) % n), ((i + 1) % n, i)])
    indices = torch.tensor(edges, dtype=torch.long).T
    values = torch.full((indices.size(1),), 1.0 / 3.0)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def _host_fixture():
    torch.manual_seed(44)
    n = 12
    model = GlobalDualBranchResponseHost(
        5, 7, 3, R3MAGHostConfig(hidden_dim=8, max_order=3, dropout=0.2)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x_t = torch.randn(n, 5)
    x_v = torch.randn(n, 7)
    operator = _small_operator(n)
    with torch.no_grad():
        logits, states_t, states_v, global_t, global_v = model(x_t, x_v, operator)
    labels = torch.arange(n) % 3
    return model, states_t, states_v, global_t, global_v, logits, labels


def test_action_count_normmatch_and_epsilon_zero_identity():
    model, states_t, states_v, global_t, global_v, _, _ = _host_fixture()
    dirs_t, _ = normalized_response_directions(states_t, global_t)
    dirs_v, _ = normalized_response_directions(states_v, global_v)
    cand_t = make_action_candidates(global_t, dirs_t, 0.2)
    cand_v = make_action_candidates(global_v, dirs_v, 0.2)
    assert len(action_names(model.config.max_order)) == 9
    assert cand_t.shape == (12, 9, 8)
    for candidates, reference in ((cand_t, global_t), (cand_v, global_v)):
        relative = (candidates[:, 1:].norm(dim=-1) - reference.norm(dim=-1, keepdim=True)).abs()
        relative /= reference.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        assert float(relative.max()) < 1e-5
    zero_t = make_action_candidates(global_t, dirs_t, 0.0)
    zero_v = make_action_candidates(global_v, dirs_v, 0.0)
    assert torch.equal(zero_t, global_t[:, None].expand_as(zero_t))
    assert torch.equal(zero_v, global_v[:, None].expand_as(zero_v))


def test_noop_logits_and_eval_are_deterministic():
    model, _, _, global_t, global_v, logits, _ = _host_fixture()
    first = model.classify_responses(global_t, global_v)
    second = model.classify_responses(global_t, global_v)
    assert float((first - logits).abs().max()) < 1e-6
    assert torch.equal(first, second)


def test_joint_separate_oracle_dominates_shared_and_uses_real_joint_forward():
    model, states_t, states_v, global_t, global_v, baseline_logits, labels = _host_fixture()
    dirs_t, _ = normalized_response_directions(states_t, global_t)
    dirs_v, _ = normalized_response_directions(states_v, global_v)
    cand_t = make_action_candidates(global_t, dirs_t, 0.2)
    cand_v = make_action_candidates(global_v, dirs_v, 0.2)
    joint = evaluate_joint_action_utilities(
        model, cand_t, cand_v, global_t, global_v, labels, baseline_logits, node_batch_size=3
    )
    shared = torch.diagonal(joint, dim1=1, dim2=2).max(dim=1).values
    separate = joint.flatten(1).max(dim=1).values
    assert bool((separate >= shared - 1e-7).all())
    # Explicitly verify a mixed pair against direct frozen forward evaluation.
    node, text_action, visual_action = 4, 1, 8
    direct_logits = model.classify_responses(cand_t[node, text_action], cand_v[node, visual_action])
    base_ce = F.cross_entropy(baseline_logits[node : node + 1], labels[node : node + 1])
    direct_u = base_ce - F.cross_entropy(direct_logits.reshape(1, -1), labels[node : node + 1])
    assert abs(float(direct_u - joint[node, text_action, visual_action])) < 1e-6


def test_stratified_partition_is_deterministic_disjoint_and_covers_train():
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    train_idx = torch.arange(30)
    a, meta_a = stratified_internal_partition(train_idx, labels, seed=20261006)
    b, meta_b = stratified_internal_partition(train_idx, labels, seed=20261006)
    assert meta_a == meta_b
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert_disjoint_splits(a)
    assert set(torch.cat(list(a.values())).tolist()) == set(train_idx.tolist())
    assert sum(x.numel() for x in a.values()) == len(train_idx)


def test_matched_control_donors_break_self_mapping_and_are_deterministic():
    n = 32
    audit = torch.arange(8)
    degree_bucket = torch.tensor([i // 8 for i in range(n)])
    predicted = torch.tensor([i % 2 for i in range(n)])
    first, counts = matched_control_donors(audit, degree_bucket, predicted, n, seed=20261006)
    second, _ = matched_control_donors(audit, degree_bucket, predicted, n, seed=20261006)
    assert torch.equal(first, second)
    assert first.shape == (len(audit), 2, 4)
    assert not bool((first == audit[:, None, None]).any())
    assert counts["joint"] == len(audit) * 2 * 4


def test_borrowed_raw_direction_is_rescaled_to_target_global_norm():
    _, states_t, _, global_t, _, _, _ = _host_fixture()
    audit = torch.tensor([0, 4, 9])
    donors = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3], [8, 7, 6, 5]])
    borrowed, _ = _borrowed_directions(global_t, states_t, audit, donors)
    target_norm = global_t[audit].norm(dim=-1)
    relative = (borrowed.norm(dim=-1) - target_norm[:, None]).abs() / target_norm[:, None].clamp_min(1e-12)
    assert float(relative.max()) < 1e-6
    raw_from_first_donor = states_t[0][donors[0, 0]] - global_t[donors[0, 0]]
    expected = raw_from_first_donor / raw_from_first_donor.norm().clamp_min(1e-12) * target_norm[0]
    assert torch.allclose(borrowed[0, 0], expected, atol=1e-7, rtol=1e-6)


def test_vectorized_single_action_utilities_match_bruteforce():
    model, states_t, states_v, global_t, global_v, baseline_logits, labels = _host_fixture()
    dirs_t, _ = normalized_response_directions(states_t, global_t)
    dirs_v, _ = normalized_response_directions(states_v, global_v)
    cand_t = make_action_candidates(global_t, dirs_t, 0.1)
    cand_v = make_action_candidates(global_v, dirs_v, 0.1)
    ut, uv, logits_t, logits_v = evaluate_single_action_utilities(
        model, cand_t, cand_v, global_t, global_v, labels, baseline_logits
    )
    assert logits_t.shape == (12, 9, 3)
    assert logits_v.shape == (12, 9, 3)
    for node in (0, 3, 7, 11):
        for action in (0, 1, 4, 8):
            ce_base = F.cross_entropy(baseline_logits[node : node + 1], labels[node : node + 1])
            direct_t = model.classify_responses(cand_t[node, action], global_v[node])
            direct_v = model.classify_responses(global_t[node], cand_v[node, action])
            direct_ut = ce_base - F.cross_entropy(direct_t.reshape(1, -1), labels[node : node + 1])
            direct_uv = ce_base - F.cross_entropy(direct_v.reshape(1, -1), labels[node : node + 1])
            assert abs(float(direct_ut - ut[node, action])) < 1e-6
            assert abs(float(direct_uv - uv[node, action])) < 1e-6
