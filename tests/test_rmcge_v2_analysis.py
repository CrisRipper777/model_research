import pytest

from scripts.analyze_rmcge_v2 import primary_checkpoint_metric
from scripts.run_rmcge_v2 import append_csv, append_mechanism_rows, ensure_csv_header


def test_primary_checkpoint_metric_reads_validation_ce_from_selected_history():
    record = {
        "best_epoch": 2,
        "val_accuracy": 0.81,
        "history": [
            {"epoch": 1, "val_ce": 0.9},
            {"epoch": 2, "val_ce": 0.7},
        ],
    }

    assert primary_checkpoint_metric(record, "val_ce") == pytest.approx(0.7)
    assert primary_checkpoint_metric(record, "val_accuracy") == pytest.approx(0.81)


def test_primary_checkpoint_metric_fails_when_selected_validation_ce_is_missing():
    record = {"best_epoch": 2, "history": [{"epoch": 1, "val_ce": 0.9}]}

    with pytest.raises(ValueError, match="selected epoch 2 has no Validation CE"):
        primary_checkpoint_metric(record, "val_ce")


def test_header_only_artifact_can_be_migrated_to_current_schema(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("old_a,old_b\n", encoding="utf-8")

    ensure_csv_header(path, ["dataset", "metric", "error"])

    assert path.read_text(encoding="utf-8") == "dataset,metric,error\n"
    append_csv(path, [{"dataset": "Movies", "metric": 0.5}], ["dataset", "metric", "error"])
    assert path.read_text(encoding="utf-8").splitlines()[-1] == "Movies,0.5,"


def test_populated_artifact_schema_mismatch_is_not_silently_rewritten(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("old_a,old_b\nkeep,these\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to change CSV schema with existing data"):
        ensure_csv_header(path, ["new_a", "new_b"])
    with pytest.raises(RuntimeError, match="CSV schema mismatch"):
        append_csv(path, [{"new_a": "value", "new_b": "value"}], ["new_a", "new_b"])
    assert path.read_text(encoding="utf-8") == "old_a,old_b\nkeep,these\n"


def test_v0_mechanism_serialization_preserves_checkpoint_source_values(tmp_path, monkeypatch):
    import csv
    import json

    import scripts.run_rmcge_v2 as runner

    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    values = {
        "expert_selection_share": [0.1, 0.2, 0.3, 0.4],
        "base_expert_selection_share": [0.4, 0.3, 0.2, 0.1],
        "expert_output_rms": 0.75,
        "expert_output_rms_by_expert": [0.7, 0.8, 0.9, 1.0],
        "expert_output_pairwise_cosine": {"0_1": 0.25},
        "weighted_graph_correction_rms": 0.125,
    }
    append_mechanism_rows({
        "dataset": "Movies",
        "variant": "V0",
        "seed": 42,
        "mechanism_details": {"edge_weights": {}, "modalities": {"text": values}},
    })

    with (tmp_path / "message_scale_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["expert_output_rms"]) == pytest.approx(0.75)
    assert float(row["weighted_graph_correction_rms"]) == pytest.approx(0.125)
    assert json.loads(row["expert_selection_share"]) == values["expert_selection_share"]

    with (tmp_path / "expert_router_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        router_row = next(csv.DictReader(handle))
    assert json.loads(router_row["expert_output_pairwise_cosine"]) == values["expert_output_pairwise_cosine"]
