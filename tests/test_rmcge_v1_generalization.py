from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data import MAGData
from src.models.rmcge_v1 import Model, build_semantic_operators
from src.tasks.nc import _evaluate_indices_from_embeddings, run_nc


def _model_cfg(dropout: float = 0.2):
    return OmegaConf.create(
        {
            "model": {
                "hidden_dim": 128,
                "dropout": dropout,
                "num_layers": 2,
                "num_experts": 3,
                "expert_bottleneck": 64,
                "expert_chunk_size": 16,
                "activation_checkpointing": True,
                "router_hidden_dim": 64,
                "relation_embedding_dim": 8,
                "modality_embedding_dim": 8,
                "cross_modal_bottleneck": 64,
                "router_correction_init_std": 0.01,
                "physical_prior": 0.6,
                "semantic_prior": 0.4,
                "residual_scale": 0.5,
            }
        }
    )


def _new_model(seed: int, dropout: float = 0.2, num_nodes: int = 8) -> Model:
    torch.manual_seed(seed)
    return Model(
        _model_cfg(dropout),
        {
            "num_nodes": num_nodes,
            "text_dim": 4,
            "visual_dim": 3,
            "input_dim": 7,
            "num_classes": 3,
        },
    )


def _attach_views(model: Model) -> torch.Tensor:
    unique = torch.tensor(
        [[0, 0, 1, 2, 3, 4], [1, 2, 2, 3, 4, 5]], dtype=torch.long
    )
    flags = torch.tensor(
        [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, 1]], dtype=torch.uint8
    )
    views = build_semantic_operators(unique, flags, model.num_nodes)
    model.semantic_pt = views["p_text"]
    model.semantic_pv = views["p_visual"]
    model.semantic_degree_text = views["degree_text"]
    model.semantic_degree_visual = views["degree_visual"]
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    return edge_index


def test_dropout_020_initialization_and_eval_forward_match_and_035_builds():
    default_model = _new_model(seed=123, dropout=0.2).eval()
    explicit_model = _new_model(seed=123, dropout=0.20).eval()
    default_edges = _attach_views(default_model)
    explicit_edges = _attach_views(explicit_model)
    assert default_model.state_dict().keys() == explicit_model.state_dict().keys()
    for name, value in default_model.state_dict().items():
        assert torch.equal(value, explicit_model.state_dict()[name]), name
    x = torch.randn(8, 7)
    with torch.no_grad():
        default_z = default_model(x, default_edges)[0]
        explicit_z = explicit_model(x, explicit_edges)[0]
    assert torch.equal(default_z, explicit_z)

    high_dropout = _new_model(seed=123, dropout=0.35)
    _attach_views(high_dropout)
    assert high_dropout.dropout == pytest.approx(0.35)
    for name, value in default_model.state_dict().items():
        assert torch.equal(value, high_dropout.state_dict()[name]), name
    with pytest.raises(ValueError, match="dropout"):
        _new_model(seed=123, dropout=0.25)


def test_relation_specific_cross_modal_interventions_are_local_and_stateless():
    model = _new_model(seed=81).eval()
    edge_index = _attach_views(model)
    x = torch.randn(8, 7)
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        original = model(x, edge_index, return_node_diagnostics=True)
        physical_off = model(
            x,
            edge_index,
            return_node_diagnostics=True,
            intervention={"disable_cross_modal_on_physical": True},
        )
        semantic_off = model(
            x,
            edge_index,
            return_node_diagnostics=True,
            intervention={"disable_cross_modal_on_semantic": True},
        )
    assert not torch.allclose(original[0], physical_off[0])
    assert not torch.allclose(original[0], semantic_off[0])
    for layer_index in range(2):
        for modality in ("text", "visual"):
            baseline_rows = original[4]["details"]["layers"][layer_index]["modalities"][modality]["relations"]
            physical_rows = physical_off[4]["details"]["layers"][layer_index]["modalities"][modality]["relations"]
            semantic_rows = semantic_off[4]["details"]["layers"][layer_index]["modalities"][modality]["relations"]
            assert physical_rows["physical"]["cross_context_rms"] == 0.0
            assert semantic_rows["semantic"]["cross_context_rms"] == 0.0
            for relation in ("physical", "semantic"):
                    if layer_index == 0:
                        assert baseline_rows[relation]["same_context_rms"] == pytest.approx(
                            physical_rows[relation]["same_context_rms"], abs=1e-7
                        )
                        assert baseline_rows[relation]["same_context_rms"] == pytest.approx(
                            semantic_rows[relation]["same_context_rms"], abs=1e-7
                        )
                    else:
                        # Layer 2 re-propagates the changed layer-1 state, while
                        # its same-modal support remains active and nonzero.
                        assert physical_rows[relation]["same_context_rms"] > 0.0
                        assert semantic_rows[relation]["same_context_rms"] > 0.0
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key]), key
    node = original[4]["details"]["layers"][0]["modalities"]["text"]["node_diagnostics"]
    assert node["relation_mass"].shape == (8, 2)
    assert node["top1_expert"].shape == (8,)
    assert node["top1_expert"][7].item() == -1
    assert node["entropy"][7].item() == 0.0


def test_router_summary_uses_relation_specific_valid_node_masks():
    from scripts.analyze_rmcge_v1_generalization import _router_and_scale_rows

    model = _new_model(seed=52).eval()
    edges = _attach_views(model)
    x = torch.randn(8, 7)
    with torch.no_grad():
        _, _, _, _, info = model(x, edges, return_node_diagnostics=True)
    router_rows, scale_rows = _router_and_scale_rows("toy", info["details"])
    assert len(router_rows) == 4
    assert len(scale_rows) == 8
    text_layer1 = next(
        row for row in router_rows
        if row["layer"] == 1 and row["modality"] == "text"
    )
    assert text_layer1["physical_valid_node_count"] == 6
    assert text_layer1["semantic_valid_node_count"] == 6
    assert text_layer1["semantic_relation_mass_n"] == 6
    assert sum(text_layer1[f"top1_expert_{index}_fraction"] for index in range(3)) == pytest.approx(1.0)


def test_extended_eval_metric_helper_is_correct_and_rng_invariant():
    classifier = torch.nn.Linear(3, 2).eval()
    z = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    labels = torch.tensor([0, 1, 0, 1])
    idx = torch.tensor([0, 2, 3])
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    observed = _evaluate_indices_from_embeddings(
        classifier, z, labels, idx, torch.device("cpu"), batch_size=2
    )
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all())
        )
    with torch.no_grad():
        logits = classifier(z[idx])
        expected_ce = float(F.cross_entropy(logits, labels[idx]))
        expected_acc = float((logits.argmax(dim=-1) == labels[idx]).float().mean())
    assert observed["cross_entropy"] == pytest.approx(expected_ce, abs=1e-7)
    assert observed["acc"] == pytest.approx(expected_acc, abs=1e-7)


class _TinyEncoder(torch.nn.Module):
    out_dim = 3
    record_run_telemetry = True
    record_epoch_history = True

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x, edge_index):
        del edge_index
        return self.linear(x), None, None, x.new_zeros(()), {}


def _tiny_training_cfg(tmp_path, *, extended: bool):
    run_name = "extended" if extended else "ordinary"
    run_dir = tmp_path / run_name
    task = {
        "protocol_version": "unified_full_graph_nc_v1",
        "training_mode": "full_graph",
        "optimizer": "adamw",
        "lr": 0.01,
        "weight_decay": 0.0001,
        "epochs": 5,
        "patience": 2,
        "early_stop_min_epoch": 1,
        "early_stop_min_delta": 0.0001,
        "grad_clip": 1.0,
        "eval_every": 1,
        "inference_mode": "full",
        "inference_batch_size": 3,
        "scheduler": None,
        "evaluate_test": False,
        "loss": {"aux_weight": 1.0},
        "save_ckpt_path": str(run_dir / "best_acc.pt"),
        "best_f1_ckpt_path": str(run_dir / "best_f1.pt"),
        "run_metrics_path": str(run_dir / "run_metrics.json"),
        "record_extended_validation": extended,
        "extended_checkpoint_metadata": {
            "config_sha256": "unit-test-config-hash",
            "git_sha": "unit-test-git-sha",
            "resolved_config": {"model": {"dropout": 0.2}},
        },
    }
    return OmegaConf.create(
        {
            "model": {"name": "tiny_encoder"},
            "task": task,
            "num_runs": 1,
            "seed": 7,
        }
    )


def test_extended_validation_preserves_trajectory_and_saves_both_checkpoints(
    tmp_path, monkeypatch
):
    import src.tasks.nc as nc

    monkeypatch.setattr(nc, "build_model", lambda cfg, data_info: _TinyEncoder())
    data = MAGData(
        name="toy",
        source="generalization-test",
        task="nc",
        x=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.0], [0.0, 0.5], [-1.0, 0.5]]
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        num_nodes=6,
        y=torch.tensor([0, 1, 0, 1, 0, 2]),
        train_idx=torch.tensor([0, 1, 2]),
        val_idx=torch.tensor([3, 4]),
        test_idx=torch.tensor([5]),
        num_classes=3,
    )
    logger = logging.getLogger("rmcge-generalization-test")
    ordinary = _tiny_training_cfg(tmp_path, extended=False)
    extended = _tiny_training_cfg(tmp_path, extended=True)
    run_nc(ordinary, data, torch.device("cpu"), logger)
    run_nc(extended, data, torch.device("cpu"), logger)

    ordinary_run = json.loads(
        Path(ordinary.task.run_metrics_path).read_text()
    )["runs"][0]
    extended_run = json.loads(
        Path(extended.task.run_metrics_path).read_text()
    )["runs"][0]
    old_history = ordinary_run["metadata"]["epoch_history"]
    new_history = extended_run["metadata"]["epoch_history"]
    assert len(old_history) == len(new_history)
    assert ordinary_run["metadata"]["best_epoch"] == extended_run["metadata"]["best_epoch"]
    assert ordinary_run["metadata"]["epochs_completed"] == extended_run["metadata"]["epochs_completed"]
    extended_initial = extended_run["metadata"]["initial_parameter_audit"]
    assert len(extended_initial["model_state_sha256"]) == 64
    assert len(extended_initial["classifier_state_sha256"]) == 64
    for old, new in zip(old_history, new_history):
        for key in ("train_loss", "val_acc", "val_macro_f1"):
            assert old[key] == new[key]
        assert {"train_ce", "train_accuracy_eval", "val_ce"} <= new.keys()

    accuracy_path = tmp_path / "extended/best_acc.pt"
    f1_path = tmp_path / "extended/best_f1.pt"
    accuracy = torch.load(accuracy_path, map_location="cpu", weights_only=False)
    best_f1 = torch.load(f1_path, map_location="cpu", weights_only=False)
    assert accuracy["selection"] == "best_val_accuracy"
    assert best_f1["selection"] == "best_val_macro_f1_within_accuracy_stopping_horizon"
    assert accuracy["run_metadata"]["study_provenance"]["git_sha"] == "unit-test-git-sha"
    assert accuracy["epoch"] == extended_run["metadata"]["best_epoch"]
    assert best_f1["epoch"] == extended_run["metadata"]["dual_checkpoint_selection"]["best_macro_f1"]["epoch"]
    assert accuracy["metrics"]["val_ce"] >= 0
    assert best_f1["metrics"]["val_ce"] >= 0
    assert extended_run["metrics"].keys() == {"val_acc", "val_macro_f1"}
    assert all("test" not in key for key in extended_run["metrics"])

    restored = _TinyEncoder()
    restored.load_state_dict(accuracy["model_state"])
    classifier = torch.nn.Linear(3, 3)
    classifier.load_state_dict(accuracy["head_state"])
    restored.eval()
    classifier.eval()
    with torch.no_grad():
        z = restored(data.x, data.edge_index)[0]
        val_logits = classifier(z[data.val_idx])
        val_ce = F.cross_entropy(val_logits, data.y[data.val_idx])
        val_acc = (val_logits.argmax(-1) == data.y[data.val_idx]).float().mean()
    assert float(val_ce) == pytest.approx(accuracy["metrics"]["val_ce"], abs=1e-7)
    assert float(val_acc) == pytest.approx(accuracy["metrics"]["val_acc"], abs=1e-7)

    restored_f1 = _TinyEncoder()
    restored_f1.load_state_dict(best_f1["model_state"])
    classifier_f1 = torch.nn.Linear(3, 3)
    classifier_f1.load_state_dict(best_f1["head_state"])
    restored_f1.eval()
    classifier_f1.eval()
    with torch.no_grad():
        z_f1 = restored_f1(data.x, data.edge_index)[0]
        f1_logits = classifier_f1(z_f1[data.val_idx])
        f1_ce = F.cross_entropy(f1_logits, data.y[data.val_idx])
        f1_acc = (f1_logits.argmax(-1) == data.y[data.val_idx]).float().mean()
    assert float(f1_ce) == pytest.approx(best_f1["metrics"]["val_ce"], abs=1e-7)
    assert float(f1_acc) == pytest.approx(best_f1["metrics"]["val_acc"], abs=1e-7)
