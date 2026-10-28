from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from scripts.mrcge_v13_nc5_lp2_utils import (
    assert_clip_feature_order,
    build_config,
    loaded_lp_split_path,
    mask_test_labels,
)
from scripts.prepare_mrcge_v13_nc5_lp2 import _validate_nc_split
from scripts import run_mrcge_v13_nc5_lp2 as nc5_runner
from src.data import EdgeSplit, MAGData
from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b, load_semantic_candidate_cache_v7b
from src.models import LinkPredictor, build_model
from src.tasks.lp import _build_forbidden_edge_keys, _sample_filtered_negative_targets
from src.tasks.mrcge_v13_full_graph_lp import (
    PROTOCOL_VERSION,
    audit_loaded_message_graph,
    backward_fold_with_leaf,
    build_fold_message_graphs,
    deterministic_train_folds,
    LPSplitConflictError,
    train_only_physical_graph,
    validate_lp_split,
)


def _edge_dict(edges: list[tuple[int, int]], *, negatives: torch.Tensor | None = None):
    result = {
        "source_node": torch.tensor([edge[0] for edge in edges], dtype=torch.long),
        "target_node": torch.tensor([edge[1] for edge in edges], dtype=torch.long),
    }
    if negatives is not None:
        result["target_node_neg"] = negatives
    return result


def _synthetic_split() -> EdgeSplit:
    return EdgeSplit(
        train=_edge_dict([(0, 1), (1, 0), (2, 3), (2, 3), (4, 5), (5, 4)]),
        valid=_edge_dict([(0, 2)], negatives=torch.tensor([[5, 6]])),
        test=_edge_dict([(1, 3)], negatives=torch.tensor([[6, 7]])),
        metadata={"undirected": True},
    )


def test_nc_fixed_seed42_path_does_not_follow_training_seed():
    cfg42 = build_config("Toys", "nc", 42)
    cfg44 = build_config("Toys", "nc", 44)
    assert cfg42.dataset.nc_split_path == cfg44.dataset.nc_split_path
    assert "seed42" in cfg42.dataset.nc_split_path
    assert cfg42.dataset.auto_generate_splits is False
    assert cfg44.task.evaluate_test is False
    assert cfg44.model.name == "mrcge_v13"
    assert cfg44.model.condition == "M_repaired"
    assert cfg44.model.v13_variant == "joint_view"
    assert float(cfg44.model.sender_private_ratio) == 0.0


def test_nc_test_label_mask_keeps_train_and_validation():
    data = MAGData(
        name="synthetic", source="magb", task="nc", x=torch.zeros(8, 3), edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.arange(8), train_idx=torch.tensor([0, 1, 2]), val_idx=torch.tensor([3, 4]),
        test_idx=torch.tensor([5, 6, 7]), num_nodes=8, num_classes=8,
    )
    safe = mask_test_labels(data)
    assert safe.y.tolist() == [0, 1, 2, 3, 4, -100, -100, -100]
    assert data.y.tolist() == list(range(8))


def test_nc_split_validator_checks_disjointness_coverage_and_hash(tmp_path):
    path = tmp_path / "seed42.pt"
    torch.save({
        "train_idx": torch.tensor([0, 1]), "val_idx": torch.tensor([2]),
        "test_idx": torch.tensor([3]), "seed": 42,
    }, path)
    result = _validate_nc_split(path, 4, split_seed=42)
    assert result["covers_all_nodes"] is True
    assert result["partition_overlap_counts"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    assert len(result["sha256"]) == 64
    torch.save({
        "train_idx": torch.tensor([0, 1]), "val_idx": torch.tensor([1]),
        "test_idx": torch.tensor([2, 3]), "seed": 42,
    }, path)
    with pytest.raises(ValueError, match="partitions overlap"):
        _validate_nc_split(path, 4, split_seed=42)


def test_mmgraph_clip_order_is_text_then_visual():
    x = torch.arange(2 * 1024).reshape(2, 1024)
    assert_clip_feature_order(x, x[:, :512], x[:, 512:])
    with pytest.raises(ValueError, match="order mismatch"):
        assert_clip_feature_order(x, x[:, 512:], x[:, :512])


def test_lp_split_path_metadata_supports_mmgraph_loader_key():
    assert loaded_lp_split_path({"edge_split_path": "/data/lp-edge-split.pt"}) == "/data/lp-edge-split.pt"
    assert loaded_lp_split_path({"lp_split_path": "/data/lp.pt"}) == "/data/lp.pt"
    assert loaded_lp_split_path({}) is None


def test_gpu_baseline_uses_exported_v13_guard_api(monkeypatch):
    snapshot = {
        "captured_utc": "synthetic", "devices": [
            {"physical_index": 0, "uuid": "GPU-0"},
            {"physical_index": 1, "uuid": "GPU-1"},
        ],
        "compute_processes": [
            {"gpu_uuid": "GPU-0", "pid": 101},
            {"gpu_uuid": "GPU-1", "pid": 202},
        ],
    }
    monkeypatch.setattr(nc5_runner.guard_impl.v13, "smi_snapshot", lambda: snapshot)
    baseline, pids = nc5_runner._capture_gpu_baseline(1)
    assert pids == {202}
    assert baseline["authorized_baseline_compute_pids"] == [202]
    assert baseline["guard_bypassed"] is False


def test_cuda_memory_stats_reset_initializes_context_first(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("init"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append(("reset", device)))
    device = torch.device("cuda:0")
    nc5_runner._reset_cuda_peak_memory_stats(device)
    assert calls == ["init", ("reset", device)]


def test_lp_train_only_graph_and_three_fold_bidirectional_holdout():
    split = _synthetic_split()
    audit = validate_lp_split(split, 8)
    assert audit["positive_counts_raw"]["train"] == 6
    assert audit["positive_counts_undirected_unique"]["train"] == 3
    assert audit["positive_partition_overlap_counts"] == {"train_valid": 0, "train_test": 0, "valid_test": 0}
    assert audit["test_negative_candidates_read"] is False

    train = torch.stack((split.train["source_node"], split.train["target_node"]))
    graph = train_only_physical_graph(split, 8)
    graph_keys = set(map(tuple, graph.t().tolist()))
    assert graph_keys == {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}
    fake_data = MAGData(name="synthetic", source="mmgraph", task="lp", x=torch.zeros(8, 2),
                        edge_index=graph, edge_split=split, num_nodes=8)
    assert audit_loaded_message_graph(fake_data, graph, 8)["unique_train_only_directed_edges"] == 6

    folds = deterministic_train_folds(train, num_nodes=8, fold_seed=42)
    assert [fold.size(1) for fold in folds] == [1, 1, 1]
    graphs = build_fold_message_graphs(train, folds, num_nodes=8)
    seen = set()
    for fold, fold_graph in zip(folds, graphs):
        fold_keys = {tuple(edge) for edge in fold.t().tolist()}
        assert not (fold_keys & seen)
        seen |= fold_keys
        message_keys = set(map(tuple, fold_graph.t().tolist()))
        for left, right in fold_keys:
            assert (left, right) not in message_keys
            assert (right, left) not in message_keys
    assert seen == {(0, 1), (2, 3), (4, 5)}
    repeated = deterministic_train_folds(train, num_nodes=8, fold_seed=42)
    assert all(torch.equal(left, right) for left, right in zip(repeated, folds))


def test_lp_split_overlap_and_invalid_ids_stop():
    split = _synthetic_split()
    split.valid = _edge_dict([(1, 0)], negatives=torch.tensor([[5]]))
    with pytest.raises(ValueError, match="partitions overlap"):
        validate_lp_split(split, 8)
    bad = _synthetic_split()
    bad.train = _edge_dict([(0, 99)])
    with pytest.raises(ValueError, match="out-of-range"):
        train_only_physical_graph(bad, 8)


def test_lp_split_conflict_carries_auditable_self_loop_and_overlap_counts():
    split = _synthetic_split()
    split.train = _edge_dict([(0, 0), (0, 1)])
    split.valid = _edge_dict([(1, 0)], negatives=torch.tensor([[5]]))
    with pytest.raises(LPSplitConflictError) as caught:
        validate_lp_split(split, 8)
    assert caught.value.audit["self_loop_positive_counts"]["train"] == 1
    assert caught.value.audit["positive_partition_overlap_counts"]["train_valid"] == 1
    assert caught.value.audit["test_negative_candidates_read"] is False


def test_train_negative_sampling_filters_known_positive_edges():
    split = _synthetic_split()
    known = _build_forbidden_edge_keys(split, num_nodes=8, undirected=True)
    source = torch.tensor([0, 2, 4], dtype=torch.long)
    negatives = _sample_filtered_negative_targets(
        source, num_nodes=8, num_neg=1, forbidden_keys=known,
        generator=torch.Generator().manual_seed(4201),
    )
    for src, dst in zip(source.tolist(), negatives[:, 0].tolist()):
        assert src != dst
        assert src * 8 + dst not in known.tolist()
        assert dst * 8 + src not in known.tolist()


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 5)
        self.aux_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, x):
        z = torch.tanh(self.linear(x))
        return z, z.square().mean() * self.aux_scale


def test_detach_leaf_batch_gradient_matches_direct_fold_bce():
    torch.manual_seed(901)
    encoder_direct = _TinyEncoder()
    projection_direct = nn.Linear(5, 4)
    decoder_direct = LinkPredictor(4, 7, 3, 0.0)
    encoder_leaf = copy.deepcopy(encoder_direct)
    projection_leaf = copy.deepcopy(projection_direct)
    decoder_leaf = copy.deepcopy(decoder_direct)
    x = torch.randn(6, 3)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 2], [1, 2, 3, 4, 5, 0, 4, 5]])
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    aux_weight = 0.7

    z_direct, aux_direct = encoder_direct(x)
    projected_direct = projection_direct(z_direct)
    logits = decoder_direct.score_pairs(projected_direct[edges[0]], projected_direct[edges[1]])
    objective = nn.functional.binary_cross_entropy_with_logits(logits, labels) + aux_weight * aux_direct
    objective.backward()
    direct_modules = [encoder_direct, projection_direct, decoder_direct]
    direct_gradients = [[p.grad.detach().clone() for p in module.parameters()] for module in direct_modules]

    z_leaf_forward, aux_leaf = encoder_leaf(x)
    optimizer = torch.optim.SGD(
        [*encoder_leaf.parameters(), *projection_leaf.parameters(), *decoder_leaf.parameters()], lr=0.0,
    )
    stats = backward_fold_with_leaf(
        model=encoder_leaf, projection=projection_leaf, decoder=decoder_leaf,
        z=z_leaf_forward, aux_loss=aux_leaf, supervision_edges=edges, labels=labels,
        optimizer=optimizer, batch_size=3, aux_weight=aux_weight, grad_clip=1e9,
        shuffle_generator=torch.Generator().manual_seed(123),
    )
    assert math_isclose(stats["bce"], float(nn.functional.binary_cross_entropy_with_logits(logits.detach(), labels)))
    leaf_modules = [encoder_leaf, projection_leaf, decoder_leaf]
    for direct_module, leaf_module, expected in zip(direct_modules, leaf_modules, direct_gradients):
        for p_direct, p_leaf, g_direct in zip(direct_module.parameters(), leaf_module.parameters(), expected):
            assert p_leaf.grad is not None
            torch.testing.assert_close(p_leaf.grad, g_direct, atol=2e-6, rtol=2e-5)


def math_isclose(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def test_lp_protocol_fixed_and_test_disabled():
    cfg = build_config("sports-copurchase", "lp", 44)
    assert cfg.task.protocol_version == PROTOCOL_VERSION
    assert cfg.task.evaluate_test is False
    assert cfg.task.training_mode == "full_graph_three_fold_edge_holdout"
    assert cfg.task.optimizer == "adam"
    assert cfg.task.epochs == 150 and cfg.task.eval_every == 2
    assert cfg.task.decoder.proj_dim == 128
    assert cfg.task.supervision_edge_batch_size == 2048


def test_v13_synthetic_forward_backward_cache_and_strict_reload(tmp_path):
    torch.manual_seed(19)
    dataset = "Toys"
    num_nodes = 12
    text = torch.randn(num_nodes, 4)
    visual = torch.randn(num_nodes, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    cache_path, metadata = build_semantic_candidate_cache_v7b(
        dataset=dataset, text_features=text, visual_features=visual, physical_edge_index=edge_index,
        cache_dir=tmp_path, top_k=8, thread_count=1,
    )
    loaded = load_semantic_candidate_cache_v7b(
        cache_path, expected_dataset=dataset,
        expected_feature_fingerprint=metadata["feature_fingerprint"],
        expected_candidate_fingerprint=metadata["candidate_fingerprint"],
        expected_v7b_fingerprint=metadata["v7b_fingerprint"],
    )
    assert loaded["metadata"]["candidate_phase_label_access"] is False
    assert loaded["metadata"]["cache_contains_labels"] is False
    cfg = build_config(dataset, "nc", 42)
    cfg.model.hidden_dim = 256
    cfg.model.semantic_cache_path = str(cache_path)
    cfg.model.semantic_feature_fingerprint = metadata["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = metadata["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = metadata["v7b_fingerprint"]
    info = {"input_dim": 9, "num_nodes": num_nodes, "num_classes": 3, "text_dim": 4, "visual_dim": 5}
    model = build_model(cfg, info)
    model.load_semantic_cache(
        str(cache_path), expected_dataset=dataset,
        expected_feature_fingerprint=metadata["feature_fingerprint"],
        expected_candidate_fingerprint=metadata["candidate_fingerprint"],
        expected_v7b_fingerprint=metadata["v7b_fingerprint"],
    )
    x = torch.cat((text, visual), dim=1)
    model.train()
    z, _, _, auxiliary, _ = model(x, edge_index)
    (z.square().mean() + auxiliary).backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    state = copy.deepcopy(model.state_dict())
    model2 = build_model(cfg, info)
    model2.load_semantic_cache(
        str(cache_path), expected_dataset=dataset,
        expected_feature_fingerprint=metadata["feature_fingerprint"],
        expected_candidate_fingerprint=metadata["candidate_fingerprint"],
        expected_v7b_fingerprint=metadata["v7b_fingerprint"],
    )
    model2.load_state_dict(state, strict=True)
    head = nn.Linear(model2.out_dim, 3)
    head2 = nn.Linear(model2.out_dim, 3)
    head2.load_state_dict(copy.deepcopy(head.state_dict()), strict=True)
    model.eval(); model2.eval()
    with torch.no_grad():
        z1 = model(x, edge_index)[0]
        z2 = model2(x, edge_index)[0]
    torch.testing.assert_close(z1, z2)
