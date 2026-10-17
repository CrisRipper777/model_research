from __future__ import annotations

from omegaconf import OmegaConf

from scripts.run_hybrid_cge_validation_expansion_v1 import (
    CONDITIONS,
    CONFIG_CONDITIONS,
    build_cfg,
    historical_dataset_provenance_checks,
    model_dropout_audit,
)
from src.models import build_model


def _hybrid_model(condition: str):
    dummy_cache = {
        "cache_path_abs": "/tmp/not-loaded.pt",
        "feature_fingerprint": "f" * 64,
        "candidate_fingerprint": "c" * 64,
        "v7b_fingerprint": "b" * 64,
    }
    cfg = build_cfg("Movies", "hybrid_cge_v11", 42, "cpu", condition, dummy_cache)
    data_info = {
        "input_dim": 1280,
        "num_nodes": 8,
        "num_classes": 20,
        "text_dim": 512,
        "visual_dim": 768,
    }
    return cfg, build_model(cfg, data_info)


def test_study_config_keeps_hybrid_structure_and_only_changes_selected_training_values():
    config = OmegaConf.load("configs/study/hybrid_cge_validation_expansion_v1.yaml")
    assert config.conditions["H2-Frozen"].dropout == 0.20
    assert config.conditions["H2-Frozen"].weight_decay == 1e-4
    assert config.conditions["H2-R1"].dropout == 0.35
    assert config.conditions["H2-R1"].weight_decay == 1e-4
    assert config.conditions["H2-R2"].dropout == 0.30
    assert config.conditions["H2-R2"].weight_decay == 1e-3
    assert config.training.private_auxiliary_ce["H2-Frozen"] == 0.30
    assert config.training.private_auxiliary_ce["H2-R1"] == 0.30
    assert config.training.private_auxiliary_ce["H2-R2"] == 0.10
    assert config.conditions["H2-R1"].late_cross_dropout == 0.15
    assert config.conditions["H2-R2"].late_cross_dropout == 0.15
    assert CONDITIONS == ("H2-R1", "H2-R2")
    assert CONFIG_CONDITIONS == ("H2-Frozen", "H2-R1", "H2-R2")


def test_actual_hybrid_dropout_modules_follow_each_model_dropout_value():
    _, r1 = _hybrid_model("H2-R1")
    _, r2 = _hybrid_model("H2-R2")
    audit1 = model_dropout_audit(r1, "hybrid_cge_v11", 0.35)
    audit2 = model_dropout_audit(r2, "hybrid_cge_v11", 0.30)
    assert audit1["nn_dropout_modules"]
    assert audit2["nn_dropout_modules"]
    assert set(audit1["nn_dropout_modules"]) == set(audit2["nn_dropout_modules"])
    assert set(audit1["nn_dropout_modules"].values()) == {0.35}
    assert set(audit2["nn_dropout_modules"].values()) == {0.30}
    assert audit1["custom_late_cross_dropout"] == 0.15
    assert audit2["custom_late_cross_dropout"] == 0.15
    assert r1.condition == r2.condition == "H2_hybrid_plus"
    assert r1.num_experts == r2.num_experts == 4
    assert r1.hidden_dim == r2.hidden_dim == 256
    assert r1.physical_hops == r2.physical_hops == 4
    assert r1.semantic_hops == r2.semantic_hops == 2


def test_research_training_configuration_disables_test_and_keeps_common_budget():
    dummy_cache = {
        "cache_path_abs": "/tmp/not-loaded.pt",
        "feature_fingerprint": "f" * 64,
        "candidate_fingerprint": "c" * 64,
        "v7b_fingerprint": "b" * 64,
    }
    for condition in CONFIG_CONDITIONS:
        cfg = build_cfg("Movies", "hybrid_cge_v11", 42, "cpu", condition, dummy_cache)
        assert cfg.task.evaluate_test is False
        assert cfg.task.epochs == 300
        assert cfg.task.patience == 30
        assert cfg.task.early_stop_min_epoch == 30
        assert cfg.task.grad_clip == 1.0
        assert cfg.task.lr == 1e-3
        assert cfg.model.late_cross_dropout == 0.15


def test_current_optimization_data_and_cache_match_frozen_h2_provenance():
    checks = historical_dataset_provenance_checks()
    assert set(checks) == {"Movies", "Grocery", "ele-fashion"}
    assert all(all(dataset_checks.values()) for dataset_checks in checks.values())
