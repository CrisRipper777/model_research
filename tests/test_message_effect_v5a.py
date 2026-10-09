from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from scripts import run_message_effect_v5a as runner
from src.research.message_effect_atlas import (
    SCALAR_FEATURE_NAMES,
    generate_message_effect_atlas,
    make_estimator_split,
    select_receiver_edges,
)
from src.research.message_effect_estimator import MessageEffectEstimator, parameter_free_pair
from src.research.message_effect_replay import (
    build_frozen_host_cache,
    bundle_interaction,
    compensated_basis_delta,
    estimator_receiver_split,
    loss_delta,
    receiver_degree_quartile_sample,
    replay_bundle,
    replay_local_basis_changes,
    replay_singleton,
    sample_receiver_edges,
    singleton_basis_delta,
)
from src.research.message_effect_training import _target_stats


class ToyRawHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 4
        self.input_dim = 6
        self.text_dim = 3
        self.visual_dim = 3
        self.trajectory_order = 4
        self.num_experts = 4
        self.expert_feature_scale = 0.1
        self.eps = 1.0e-8
        self.projectors = nn.ModuleDict(
            {"text": nn.Linear(3, 4), "visual": nn.Linear(3, 4)}
        )
        self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.router_projectors = nn.ModuleDict(
            {"text": nn.Linear(4, 4), "visual": nn.Linear(4, 4)}
        )
        self.router_norms = nn.ModuleDict(
            {"text": nn.LayerNorm(4), "visual": nn.LayerNorm(4)}
        )
        self.alpha_raw = nn.Parameter(torch.randn(4, 4))
        self.fusion_skip = nn.Linear(8, 4)
        self.fusion_linear1 = nn.Linear(8, 4)
        self.fusion_dropout = nn.Dropout(0.1)
        self.fusion_linear2 = nn.Linear(4, 4)
        self.fusion_norm = nn.LayerNorm(4)

    @staticmethod
    def _effective_alpha(raw: torch.Tensor, eps: float) -> torch.Tensor:
        return raw / (raw.norm(dim=-1, keepdim=True) + eps)

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"text": x[:, :3], "visual": x[:, 3:]}

    @staticmethod
    def _normalized_operator(edge_index, num_nodes: int, dtype: torch.dtype):
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        src, dst = edge_index
        degree = torch.bincount(dst, minlength=num_nodes).to(dtype)
        norm = degree.clamp_min(1).rsqrt()[src] * degree.clamp_min(1).rsqrt()[dst]
        return src, dst, norm, degree > 0, 0

    @staticmethod
    def _propagate(state, src, dst, weight):
        return torch.zeros_like(state).index_add(0, dst, state[src] * weight[:, None])

    def _active_rms_norm(self, state, active):
        result = torch.zeros_like(state)
        denominator = torch.sqrt(state[active].square().mean(0) + self.eps)
        result[active] = state[active] / denominator
        return result

    def forward(self, x, edge_index, return_details=False):
        inputs = self._split_modalities(x)
        src, dst, edge_weight, active, _ = self._normalized_operator(
            edge_index, x.size(0), x.dtype
        )
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        outputs, details, router, roles = {}, {}, {}, {}
        for modality in ("text", "visual"):
            prior = self.projectors[modality](inputs[modality])
            states = []
            state = prior
            denoms = []
            for _ in range(4):
                state = self._propagate(state, src, dst, edge_weight)
                state = state * active[:, None]
                states.append(state)
                denoms.append(torch.sqrt(state[active].square().mean(0) + self.eps))
            raw_states = torch.stack(states)
            basis = torch.stack(
                [self._active_rms_norm(value, active) for value in states]
            )
            profiles = torch.einsum("mk,knd->mnd", alpha, basis)
            experts = torch.stack(
                [profiles[k] + self.expert_feature_scale * self.experts[k](profiles[k])
                 for k in range(4)],
                dim=0,
            )
            route_weights = torch.softmax(
                torch.arange(1, 5, dtype=x.dtype, device=x.device), dim=0
            ).expand(x.size(0), -1)
            strength = x.new_full((x.size(0),), 0.4)
            mixture = (route_weights[:, :, None] * experts.permute(1, 0, 2)).sum(1)
            output = prior + strength[:, None] * mixture
            outputs[modality] = output
            router_state = self.router_norms[modality](
                self.router_projectors[modality](prior)
            )
            cosine = torch.nn.functional.cosine_similarity(
                router_state[src], router_state[dst], dim=-1
            )
            router[modality] = {
                "route_weights": route_weights,
                "strength": strength,
            }
            roles[modality] = {
                "semantic_cosine": cosine,
                "role_score": cosine * 0.5,
                "supportive_mask": cosine >= 0,
            }
            details[modality] = {
                "prior": prior,
                "raw_states": raw_states,
                "raw_trajectory": basis,
                "raw_denominators": torch.stack(denoms),
                "raw_mixture": mixture,
                "output": output,
            }
        fused = torch.cat((outputs["text"], outputs["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(torch.nn.functional.gelu(self.fusion_linear1(fused)))
            )
        )
        info = {
            "details": details,
            "router": router,
            "role_channels": roles,
        }
        return z, None, None, x.new_zeros(()), info


@pytest.fixture
def toy_cache():
    torch.manual_seed(11)
    host = ToyRawHost().eval()
    classifier = nn.Linear(4, 3).eval()
    x = torch.randn(5, 6)
    edge_index = torch.tensor(
        [[0, 1, 2, 1, 3, 4, 0, 4, 2, 3], [1, 0, 1, 2, 4, 3, 4, 0, 3, 2]]
    )
    cache = build_frozen_host_cache("toy", host, classifier, x, edge_index, "cpu")
    return cache, x, edge_index


def _edge_id(cache, sender: int, receiver: int) -> int:
    found = torch.nonzero(
        (cache.src == sender) & (cache.dst == receiver), as_tuple=False
    ).reshape(-1)
    assert found.numel() == 1
    return int(found.item())


def test_host_cache_tensors_are_finite(toy_cache):
    cache, _, _ = toy_cache
    for mapping in (
        cache.prior, cache.raw_states, cache.raw_basis, cache.denominators,
        cache.route_weights, cache.strength, cache.modality_outputs,
    ):
        for tensor in mapping.values():
            assert torch.isfinite(tensor).all()
    assert torch.isfinite(cache.z).all()
    assert torch.isfinite(cache.logits).all()


def test_local_reconstruction_matches_full_modality_outputs(toy_cache):
    cache, _, _ = toy_cache
    receivers = torch.arange(5)
    replay = replay_local_basis_changes(cache, receivers)
    for modality in ("text", "visual"):
        torch.testing.assert_close(
            replay["modality_outputs"][modality],
            cache.modality_outputs[modality],
            atol=1e-6,
            rtol=1e-6,
        )


def test_local_reconstruction_matches_full_fused_embedding(toy_cache):
    cache, _, _ = toy_cache
    replay = replay_local_basis_changes(cache, torch.arange(5))
    torch.testing.assert_close(replay["z"], cache.z, atol=1e-6, rtol=1e-6)


def test_local_reconstruction_matches_full_classifier_logits(toy_cache):
    cache, _, _ = toy_cache
    replay = replay_local_basis_changes(cache, torch.arange(5))
    torch.testing.assert_close(replay["logits"], cache.logits, atol=1e-6, rtol=1e-6)


def test_singleton_modifies_only_selected_receiver_modality_hop(toy_cache):
    cache, _, _ = toy_cache
    receiver = torch.tensor([1])
    delta = torch.ones((1, 4)) * 0.1
    replay = replay_singleton(cache, receiver, "text", 2, delta)
    for modality in ("text", "visual"):
        for hop in range(4):
            expected = cache.raw_basis[modality][hop, 1]
            actual = replay["basis"][modality][0, hop]
            if modality == "text" and hop == 1:
                torch.testing.assert_close(actual, expected + delta[0])
            else:
                torch.testing.assert_close(actual, expected)


def test_scalar_and_batched_replay_agree(toy_cache):
    cache, _, _ = toy_cache
    receivers = torch.tensor([0, 1, 2])
    deltas = torch.randn(3, 4) * 0.01
    batched = replay_singleton(cache, receivers, "visual", 3, deltas)
    for i, receiver in enumerate(receivers.tolist()):
        scalar = replay_singleton(cache, torch.tensor([receiver]), "visual", 3, deltas[i:i + 1])
        torch.testing.assert_close(scalar["logits"][0], batched["logits"][i])


def test_hop1_message_is_weight_times_sender_prior(toy_cache):
    cache, _, _ = toy_cache
    eid = _edge_id(cache, 0, 1)
    q = cache.edge_weight[eid] * cache.prior["text"][0]
    expected = cache.edge_weight[eid] * cache.prior["text"][0]
    torch.testing.assert_close(q, expected)


def test_hop2_message_is_weight_times_sender_t1(toy_cache):
    cache, _, _ = toy_cache
    eid = _edge_id(cache, 0, 1)
    q = cache.edge_weight[eid] * cache.raw_states["text"][0, 0]
    expected = cache.edge_weight[eid] * cache.raw_states["text"][0, 0]
    torch.testing.assert_close(q, expected)


def test_deletion_uses_frozen_baseline_denominator(toy_cache):
    cache, _, _ = toy_cache
    eid = _edge_id(cache, 0, 1)
    message = cache.edge_weight[eid] * cache.prior["text"][0]
    delta = singleton_basis_delta(message, cache.denominators["text"][0])
    assert torch.isfinite(delta).all()
    torch.testing.assert_close(delta, -message / cache.denominators["text"][0])


def test_degree_one_compensation_is_infeasible():
    delta, scale, retained_mass, feasible = compensated_basis_delta(
        torch.tensor([2.0, 4.0]), torch.tensor([2.0, 4.0]),
        torch.tensor([1.0, 1.0]), receiver_mass=0.5, edge_weight=0.5,
    )
    assert not feasible
    assert delta is scale is retained_mass is None


def test_compensated_retained_coefficient_mass_is_preserved():
    delta, scale, retained_mass, feasible = compensated_basis_delta(
        torch.tensor([4.0, 8.0]), torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]), receiver_mass=1.0, edge_weight=0.25,
    )
    assert feasible
    assert scale == pytest.approx(4 / 3)
    assert retained_mass == pytest.approx(1.0)
    assert torch.isfinite(delta).all()


def test_compensated_aggregate_formula():
    raw = torch.tensor([4.0, 8.0])
    message = torch.tensor([1.0, 2.0])
    denominator = torch.tensor([2.0, 4.0])
    delta, scale, _, feasible = compensated_basis_delta(
        raw, message, denominator, receiver_mass=1.0, edge_weight=0.25
    )
    assert feasible and scale is not None
    expected = (scale * (raw - message) - raw) / denominator
    torch.testing.assert_close(delta, expected)


def test_same_hop_text_visual_bundle_changes_both_modalities(toy_cache):
    cache, _, _ = toy_cache
    delta = torch.ones((1, 4)) * 0.01
    replay = replay_bundle(
        cache,
        torch.tensor([1]),
        [
            {"modality": "text", "hop": 2, "delta": delta},
            {"modality": "visual", "hop": 2, "delta": -delta},
        ],
    )
    torch.testing.assert_close(replay["basis"]["text"][0, 1], cache.raw_basis["text"][1, 1] + delta[0])
    torch.testing.assert_close(replay["basis"]["visual"][0, 1], cache.raw_basis["visual"][1, 1] - delta[0])


def test_modality_all_hop_bundle_changes_four_hops(toy_cache):
    cache, _, _ = toy_cache
    changes = [
        {"modality": "text", "hop": hop, "delta": torch.ones((1, 4)) * hop * 0.001}
        for hop in range(1, 5)
    ]
    replay = replay_bundle(cache, torch.tensor([1]), changes)
    for hop in range(4):
        assert not torch.equal(replay["basis"]["text"][0, hop], cache.raw_basis["text"][hop, 1])
    assert torch.equal(replay["basis"]["visual"], cache.raw_basis["visual"][:, 1:2].transpose(0, 1))


def test_full_bundle_has_eight_basis_contributions(toy_cache):
    cache, _, _ = toy_cache
    changes = [
        {"modality": modality, "hop": hop, "delta": torch.ones((1, 4)) * 0.001}
        for modality in ("text", "visual") for hop in range(1, 5)
    ]
    replay = replay_bundle(cache, torch.tensor([1]), changes)
    assert sum(
        not torch.equal(replay["basis"][modality][0, hop], cache.raw_basis[modality][hop, 1])
        for modality in ("text", "visual") for hop in range(4)
    ) == 8


def test_reverse_directed_edge_is_not_synchronized(toy_cache):
    cache, _, _ = toy_cache
    receivers = torch.tensor([0, 1])
    delta = torch.stack((torch.zeros(4), torch.ones(4) * 0.1))
    replay = replay_singleton(cache, receivers, "text", 1, delta)
    # The reverse edge 1->0 is a separate receiver message, so node 0's basis stays baseline.
    torch.testing.assert_close(replay["basis"]["text"][0, 0], cache.raw_basis["text"][0, 0])
    torch.testing.assert_close(replay["basis"]["text"][1, 0], cache.raw_basis["text"][0, 1] + delta[1])


def test_receiver_stratified_sampling_is_deterministic():
    train = torch.arange(40)
    indegree = torch.arange(40) % 9
    a, pa, _ = receiver_degree_quartile_sample(train, indegree, max_receivers=12, seed=2027)
    b, pb, _ = receiver_degree_quartile_sample(train, indegree, max_receivers=12, seed=2027)
    assert torch.equal(a, b)
    assert pa == pb
    assert len(a) == 12


def test_degree_at_most_four_takes_all_incoming_edges(toy_cache):
    cache, _, _ = toy_cache
    eid, probability = sample_receiver_edges("toy", 1, cache.dst, max_edges=4, seed=2027)
    expected = torch.nonzero(cache.dst.cpu() == 1, as_tuple=False).reshape(-1)
    assert torch.equal(eid, expected)
    assert probability == 1.0


def test_degree_above_four_samples_exactly_four_deterministically():
    dst = torch.tensor([7] * 9 + [8, 9])
    first, p1 = sample_receiver_edges("unit", 7, dst, max_edges=4, seed=2027)
    second, p2 = sample_receiver_edges("unit", 7, dst, max_edges=4, seed=2027)
    assert first.numel() == 4
    assert torch.equal(first, second)
    assert p1 == p2 == pytest.approx(4 / 9)


def test_estimator_receiver_splits_are_disjoint():
    splits = estimator_receiver_split("toy", torch.arange(200), seed=2027)
    values = list(splits.values())
    assert all(not (a & b) for i, a in enumerate(values) for b in values[i + 1:])
    assert sum(map(len, values)) == 200


def test_atlas_uses_only_train_receiver_indices_and_no_labels_in_rows(toy_cache):
    cache, _, _ = toy_cache
    train_receivers = {1, 2, 3, 4}
    edge_ids = [
        i for i, receiver in enumerate(cache.dst.tolist()) if receiver in train_receivers
    ]
    labels = torch.full((5,), -1, dtype=torch.long)
    labels[torch.tensor(sorted(train_receivers))] = torch.tensor([0, 1, 2, 0])
    receiver_probs = {receiver: 1.0 for receiver in train_receivers}
    edge_probs = {receiver: 1.0 for receiver in train_receivers}
    rows, _, _ = generate_message_effect_atlas(
        cache, edge_ids, labels, receiver_probs, edge_probs
    )
    assert {r["receiver_id"] for r in rows} <= train_receivers
    assert all("label" not in r and "target_label" not in r for r in rows)


def test_atlas_supervision_loader_never_requests_validation_or_test_split(monkeypatch):
    class Split(dict):
        def __getitem__(self, key):
            if key in {"val_idx", "test_idx"}:
                raise AssertionError(f"forbidden split access: {key}")
            return super().__getitem__(key)

    cfg = runner.OmegaConf.create(
        {"dataset": {"source": "node", "node_split_path": "split.pt", "label_path": "labels.pt", "num_classes": 3}}
    )
    monkeypatch.setattr(runner, "_compose", lambda dataset, seed: cfg)
    monkeypatch.setattr("src.data.loaders.resolve_path", lambda value: value)
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, **kwargs: (
            Split(train_idx=torch.tensor([0, 2, 4]), val_idx=torch.tensor([1]), test_idx=torch.tensor([3]))
            if path == "split.pt"
            else torch.tensor([0, 2, 1, 2, 0])
        ),
    )
    result = runner.load_training_supervision("toy", include_validation=False)
    assert result["train_idx"].tolist() == [0, 2, 4]
    assert result["train_labels"].tolist() == [0, 1, 0]
    assert result["validation_labels_read"] is False
    assert result["test_labels_read"] is False


def test_singleton_effect_sign_is_modified_loss_minus_baseline():
    logits = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    labels = torch.tensor([0, 1])
    baseline = torch.tensor([0.3, 0.4])
    expected = torch.nn.functional.cross_entropy(logits, labels, reduction="none") - baseline
    torch.testing.assert_close(loss_delta(logits, labels, baseline), expected)


def test_e0_rejects_hidden_pair_and_host_vectors():
    model = MessageEffectEstimator("E0_heuristic", scalar_dim=len(SCALAR_FEATURE_NAMES))
    scalar = torch.randn(2, len(SCALAR_FEATURE_NAMES))
    assert model(scalar).shape == (2, 2)
    with pytest.raises(ValueError):
        model(scalar, pair_active=torch.randn(2, 1024))
    assert not any("pair" in name or "host" in name for name, _ in model.named_parameters())


def test_e1_has_only_active_modality_pair_block():
    model = MessageEffectEstimator("E1_unimodal_pair", scalar_dim=len(SCALAR_FEATURE_NAMES))
    out = model(torch.randn(2, len(SCALAR_FEATURE_NAMES)), pair_active=torch.randn(2, 1024))
    assert out.shape == (2, 2)
    assert model.pair_active_encoder is not None
    assert model.pair_text_encoder is None and model.pair_visual_encoder is None
    with pytest.raises(ValueError):
        model(torch.randn(2, len(SCALAR_FEATURE_NAMES)), pair_text=torch.randn(2, 1024), pair_visual=torch.randn(2, 1024))


def test_e2_has_separate_text_and_visual_pair_blocks():
    model = MessageEffectEstimator("E2_multimodal_pair", scalar_dim=len(SCALAR_FEATURE_NAMES))
    out = model(
        torch.randn(2, len(SCALAR_FEATURE_NAMES)),
        pair_text=torch.randn(2, 1024),
        pair_visual=torch.randn(2, 1024),
    )
    assert out.shape == (2, 2)
    assert model.pair_text_encoder is not model.pair_visual_encoder
    assert model.host_encoder is None


def test_e3_receives_frozen_host_context():
    model = MessageEffectEstimator(
        "E3_host_context", scalar_dim=len(SCALAR_FEATURE_NAMES), host_dim=523
    )
    out = model(
        torch.randn(2, len(SCALAR_FEATURE_NAMES)),
        pair_text=torch.randn(2, 1024),
        pair_visual=torch.randn(2, 1024),
        host_context=torch.randn(2, 523),
    )
    assert out.shape == (2, 2)
    assert model.host_encoder is not None


def test_estimator_features_exclude_receiver_labels():
    assert not any("label" in name.lower() for name in SCALAR_FEATURE_NAMES)
    model = MessageEffectEstimator("E0_heuristic", scalar_dim=len(SCALAR_FEATURE_NAMES))
    assert not any("label" in name.lower() for name, _ in model.named_parameters())


def test_compensation_target_mask_stats_ignore_infeasible_rows():
    target = torch.tensor([1.0, float("nan"), 3.0, float("nan")])
    mean, std = _target_stats(target, torch.arange(4))
    assert mean.item() == pytest.approx(2.0)
    assert std.item() == pytest.approx(1.0)


def test_predictor_forward_and_backward_are_finite():
    model = MessageEffectEstimator("E1_unimodal_pair", scalar_dim=17)
    scalar = torch.randn(5, 17, requires_grad=True)
    pair = torch.randn(5, 1024, requires_grad=True)
    output = model(scalar, pair_active=pair)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(scalar.grad).all()
    assert torch.isfinite(pair.grad).all()


def test_bundle_interaction_formula():
    interaction, relative = bundle_interaction(2.5, [1.0, -0.5])
    assert interaction == pytest.approx(2.0)
    assert relative == pytest.approx(2.0 / 1.5)


def test_pair_encoder_feature_order_and_shape():
    receiver = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sender = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    pair = parameter_free_pair(receiver, sender)
    assert pair.shape == (1, 16)
    assert torch.isfinite(pair).all()


def test_receiver_edge_sampler_returns_probability_min_one_four_over_degree():
    dst = torch.tensor([3] * 8)
    _, probability = sample_receiver_edges("toy", 3, dst, max_edges=4)
    assert probability == pytest.approx(0.5)


def test_formal_split_fixed_across_estimator_seeds():
    features = {
        "receiver_id": torch.arange(100).repeat_interleave(2),
    }
    split_a = make_estimator_split(features, "toy")
    split_b = make_estimator_split(features, "toy")
    assert split_a == split_b


def test_cache_has_no_dense_adjacency_or_edge_dictionary(toy_cache):
    cache, _, _ = toy_cache
    assert not hasattr(cache, "adjacency")
    assert not hasattr(cache, "edge_to_receiver")


def test_non_selected_receivers_are_unchanged_in_singleton_replay(toy_cache):
    cache, _, _ = toy_cache
    delta = torch.ones((1, 4)) * 0.02
    replay = replay_singleton(cache, torch.tensor([1]), "visual", 4, delta)
    assert torch.equal(replay["basis"]["visual"][0, 0], cache.raw_basis["visual"][0, 1])
    assert torch.equal(replay["basis"]["visual"][0, 1], cache.raw_basis["visual"][1, 1])
    assert torch.equal(replay["basis"]["text"][0], cache.raw_basis["text"][:, 1])
    assert torch.equal(replay["basis"]["visual"][0, 2], cache.raw_basis["visual"][2, 1])


def test_atlas_features_have_no_receiver_label_column(toy_cache):
    cache, _, _ = toy_cache
    train_receivers = {0, 1, 2, 3, 4}
    labels = torch.tensor([0, 1, 2, 0, 1])
    edge_ids = torch.arange(cache.src.numel())
    probs = {i: 1.0 for i in train_receivers}
    rows, _, features = generate_message_effect_atlas(cache, edge_ids, labels, probs, probs)
    assert features["scalar"].shape[1] == len(SCALAR_FEATURE_NAMES)
    assert "label" not in features
    assert all("label" not in row for row in rows)
