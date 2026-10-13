from __future__ import annotations

import csv

import pytest

import scripts.analyze_rmcge_v21 as analysis
import scripts.run_rmcge_v21 as runner


def _synthetic_rows():
    rows = []
    for dataset_id, dataset in enumerate(runner.DATASETS):
        for condition_id, condition in enumerate(runner.CONDITIONS):
            for seed_id, seed in enumerate(runner.SEEDS):
                base = 0.5 + 0.01 * dataset_id + 0.001 * seed_id
                rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "seed": str(seed),
                    "status": "completed_and_verified",
                    "test_metrics_present": "False",
                    "val_acc": str(base + 0.01 * condition_id),
                    "val_macro_f1": str(base - 0.1 + 0.02 * condition_id),
                    "val_ce": str(1.0 - 0.1 * condition_id + 0.01 * seed_id),
                    "best_f1_val_acc": str(base - 0.01 * condition_id),
                    "best_f1_val_macro_f1": str(base - 0.1 + 0.03 * condition_id),
                    "best_f1_val_ce": str(1.1 - 0.1 * condition_id + 0.01 * seed_id),
                })
    return rows


def test_append_csv_upserts_by_run_key_without_schema_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    path = tmp_path / "runs.csv"
    fields = runner.CSV_FIELDS["runs.csv"]
    runner.append_csv(path, [{"dataset": "Movies", "condition": "C1", "seed": 42, "status": "failed"}], fields)
    runner.append_csv(path, [{"dataset": "Movies", "condition": "C1", "seed": 42, "status": "completed_and_verified"}], fields)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "completed_and_verified"
    assert list(rows[0]) == fields


def test_csv_array_and_object_cells_are_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    path = tmp_path / "message_scale_diagnostics.csv"
    runner.append_csv(path, [{
        "dataset": "Movies", "condition": "C2", "seed": 42, "modality": "text",
        "expert_coefficients": {"intra": [0.2, 0.3]},
        "expert_output_rms_by_expert": [0.4, 0.5],
    }], runner.CSV_FIELDS[path.name])
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    import json

    assert json.loads(row["expert_coefficients"]) == {"intra": [0.2, 0.3]}
    assert json.loads(row["expert_output_rms_by_expert"]) == [0.4, 0.5]


def test_read_csv_rejects_wrong_schema_and_duplicate_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "DATA_DIR", tmp_path)
    path = tmp_path / "runs.csv"
    path.write_text("dataset,condition\nMovies,C0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema differs"):
        analysis._read_csv("runs.csv")
    path.write_text(",".join(runner.CSV_FIELDS["runs.csv"]) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runner.CSV_FIELDS["runs.csv"])
        writer.writerow({"dataset": "Movies", "condition": "C0", "seed": 42})
        writer.writerow({"dataset": "Movies", "condition": "C0", "seed": 42})
    with pytest.raises(RuntimeError, match="duplicate primary key"):
        analysis._read_csv("runs.csv", ("dataset", "condition", "seed"))


def test_paired_contrasts_preserve_seed_differences_and_sample_sd(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analysis, "append_csv", runner.append_csv)
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    result = analysis.derive_paired_results(_synthetic_rows())
    summary = result["paired_summary"]["Movies"]["C1-C0"]["val_acc"]
    assert summary["seed_deltas"] == {"42": pytest.approx(0.01), "43": pytest.approx(0.01), "44": pytest.approx(0.01)}
    assert summary["mean_delta"] == pytest.approx(0.01)
    assert summary["sample_sd"] == pytest.approx(0.0, abs=1e-12)
    assert len(result["paired_rows"]) == 81
    assert len(result["summary_rows"]) == 54


def test_result_table_keeps_accuracy_and_macro_f1_rows_distinct():
    rows = []
    for dataset in runner.DATASETS:
        for condition_id, condition in enumerate(runner.CONDITIONS):
            rows.append({"dataset": dataset, "condition": condition, "metric": "val_acc", "mean": 0.7 + condition_id * 0.01, "sample_sd": 0.02})
    table = analysis._format_results_table(rows, "val_acc")
    assert table[0] == "| Dataset | C0 | C1 | C2 |"
    assert "70.00 ± 2.00" in table[2]
    assert "72.00 ± 2.00" in table[2]


def test_frozen_configuration_matches_requested_matrix():
    from omegaconf import OmegaConf

    model = OmegaConf.load(runner.ROOT / "configs/model/rmcge_v21.yaml")
    study = OmegaConf.load(runner.ROOT / "configs/study/rmcge_v21_pilot.yaml")
    assert model.name == "rmcge_v21"
    assert study.conditions == ["C0", "C1", "C2"]
    assert study.datasets == ["Movies", "Grocery", "ele-fashion"]
    assert study.seeds == [42, 43, 44]
    assert study.training.evaluate_test is False
