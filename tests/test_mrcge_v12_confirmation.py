from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_mrcge_v12_confirmation as confirmation


def test_confirmation_matrix_is_exactly_18_new_cells() -> None:
    cells = confirmation.confirmation_cells()
    assert len(cells) == 18
    assert all(seed in (43, 44) for dataset, condition, seed in cells if condition.startswith("M_"))
    assert {cell for cell in cells if cell[0] == "ele-fashion" and cell[1] in ("S_repaired", "F_sinit")} == {
        ("ele-fashion", "S_repaired", 42), ("ele-fashion", "F_sinit", 42),
    }
    assert not any(seed == 42 and condition.startswith("M_") for _, condition, seed in cells)


def test_confirmation_filters_select_independent_cells() -> None:
    assert confirmation.confirmation_cells(("Movies",), ("M_anchor",), (43,)) == [
        ("Movies", "M_anchor", 43),
    ]
    assert set(confirmation.confirmation_cells(("ele-fashion",), seeds=(42,))) == {
        ("ele-fashion", "S_repaired", 42), ("ele-fashion", "F_sinit", 42),
    }


def test_new_seed_cfg_reuses_frozen_seed42_split_without_generation() -> None:
    cache = {
        "cache_path_abs": "/tmp/unused_cache.pt",
        "feature_fingerprint": "feature", "candidate_fingerprint": "candidate",
        "v7b_fingerprint": "v7b",
    }
    cfg = confirmation.build_cfg("Movies", "M_repaired", 43, "cpu", cache)
    assert cfg.seed == 43
    assert Path(cfg.dataset.nc_split_path).name == "Movies_nc_seed42_train0.6_val0.2.pt"
    assert cfg.dataset.auto_generate_splits is False
    assert cfg.task.evaluate_test is False


def test_new_runner_cannot_write_to_v12_screen_output() -> None:
    assert confirmation.DATA == confirmation.ROOT / "research/mrcge_v12_confirmation/data"
    assert confirmation.OUTPUT == confirmation.ROOT / "outputs/mrcge_v12_confirmation"
    assert "mrcge_v12_screen" not in str(confirmation.OUTPUT)


def test_only_authorized_physical_gpu_can_be_selected() -> None:
    with pytest.raises(RuntimeError, match="GPU 1 only"):
        confirmation.run_confirmation_gpu_preflight(physical_gpu=0)
