#!/usr/bin/env python3
"""Label-free diagnostics for a selected PACE-MAG Full validation checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "research/pace_mag_v1/data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def analyze_checkpoint(
    *,
    dataset: str,
    checkpoint_path: str | Path,
    cache_entry: dict[str, Any],
    device: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    from src.models import build_model
    from scripts.run_pace_mag_v1 import _build_config, _candidate_inputs, _git, now_utc, write_json

    text, visual, physical_edge_index, _sources, data_info = _candidate_inputs(dataset)
    text_tensor = torch.from_numpy(np.ascontiguousarray(text, dtype=np.float32))
    visual_tensor = torch.from_numpy(np.ascontiguousarray(visual, dtype=np.float32))
    x = torch.cat((text_tensor, visual_tensor), dim=1)
    cfg = _build_config(dataset, "e2_full", device=device)
    cfg.model.semantic_cache_path = cache_entry.get("cache_path_abs", str(ROOT / cache_entry["cache_path"]))
    cfg.model.semantic_feature_fingerprint = cache_entry["feature_fingerprint"]
    cfg.model.semantic_candidate_fingerprint = cache_entry["candidate_fingerprint"]
    cfg.model.semantic_v7b_fingerprint = cache_entry["v7b_fingerprint"]
    info = {
        "input_dim": int(x.size(1)),
        "num_nodes": int(data_info["num_nodes"]),
        "num_classes": 0,
        "text_dim": int(text_tensor.size(1)),
        "visual_dim": int(visual_tensor.size(1)),
    }
    model = build_model(cfg, info)
    model.load_semantic_cache(
        str(cfg.model.semantic_cache_path),
        expected_dataset=dataset,
        expected_feature_fingerprint=cache_entry["feature_fingerprint"],
        expected_candidate_fingerprint=cache_entry["candidate_fingerprint"],
        expected_v7b_fingerprint=cache_entry["v7b_fingerprint"],
    )
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = ROOT / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("selection") != "best_val_accuracy":
        raise ValueError("diagnostics require the Validation Accuracy-selected checkpoint")
    if set(checkpoint.get("metrics", {})) != {"val_acc", "val_macro_f1"}:
        raise ValueError("checkpoint contains unexpected metrics; expected Validation-only metrics")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    x = x.to(device)
    edge_index = physical_edge_index.to(device)
    with torch.no_grad():
        z, _, _, aux_loss, info = model(x, edge_index, return_details=True)
    details = info["details"]
    if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(aux_loss)):
        raise FloatingPointError("selected validation checkpoint produced non-finite embeddings or aux loss")
    result = {
        "recorded_at_utc": now_utc(),
        "dataset": dataset,
        "variant": "e2_full",
        "seed": int(checkpoint["seed"]),
        "checkpoint_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "checkpoint_selection": checkpoint["selection"],
        "validation_metrics_only": dict(checkpoint["metrics"]),
        "commit_sha": _git("rev-parse", "HEAD"),
        "semantic_cache_path": cache_entry.get("cache_path", str(cfg.model.semantic_cache_path)),
        "semantic_feature_fingerprint": cache_entry["feature_fingerprint"],
        "semantic_candidate_fingerprint": cache_entry["candidate_fingerprint"],
        "semantic_v7b_fingerprint": cache_entry["v7b_fingerprint"],
        "num_nodes": int(data_info["num_nodes"]),
        "physical_edges": int(data_info["num_edges_physical"]),
        "effective_node_fraction": details["effective_node_fraction"],
        "input_message_rms": details["message_rms"],
        "aggregate_message_rms": details["aggregate_rms"],
        "expert_output_rms": details["expert_output_rms"],
        "expert_pairwise_cosine": details["expert_pairwise_cosine"],
        "router": details["router"],
        "physical_response_rms": details["response_rms"]["physical"],
        "semantic_response_rms": details["response_rms"]["semantic"],
        "beta_phy": details["beta_phy"],
        "beta_sem": details["beta_sem"],
        "rho": details["rho"],
        "gamma_fixed": details["gamma_fixed"],
        "final_embedding": details["final_embedding"],
        "aux_loss": float(aux_loss.cpu()),
        "diagnostic_label_access": False,
        "test_labels_loaded": False,
        "test_metrics_computed": False,
        "notes": [
            "Diagnostics use the selected Validation checkpoint and only Text/Visual features plus physical/candidate graph topology.",
            "No NC labels or split files are loaded by this analysis path.",
            "PT + PV = PU is an operator-level reconstruction; row-quality aggregation and expert nonlinearities do not make the complete model linearly reconstructible.",
        ],
    }
    if output_path is not None:
        out = Path(output_path)
        if not out.is_absolute():
            out = ROOT / out
        write_json(out, result)
    return result


def summarize_pilot() -> list[dict[str, Any]]:
    source = DATA_DIR / "pilot_runs.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Pilot run table is missing: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["variant"]), []).append(row)
    summary = []
    for (dataset, variant), group in sorted(groups.items()):
        completed = [row for row in group if row["status"] == "completed"]
        failed = [row for row in group if row["status"] == "failed"]

        def mean(key: str) -> float | str:
            values = [float(row[key]) for row in completed if row.get(key) not in (None, "")]
            return sum(values) / len(values) if values else ""

        peak_allocated = [int(row["cuda_peak_allocated_bytes"]) for row in completed if row.get("cuda_peak_allocated_bytes")]
        peak_reserved = [int(row["cuda_peak_reserved_bytes"]) for row in completed if row.get("cuda_peak_reserved_bytes")]
        e3_widths = [int(row["e3_single_expert_bottleneck"]) for row in completed if row.get("e3_single_expert_bottleneck")]
        e3_deltas = [int(row["e3_parameter_difference"]) for row in completed if row.get("e3_parameter_difference")]
        summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "runs_expected": len(group),
                "runs_completed": len(completed),
                "runs_failed": len(failed),
                "val_acc": mean("val_acc"),
                "val_macro_f1": mean("val_macro_f1"),
                "best_epoch": mean("best_epoch"),
                "training_seconds": mean("training_seconds"),
                "max_peak_allocated_bytes": max(peak_allocated) if peak_allocated else "",
                "max_peak_reserved_bytes": max(peak_reserved) if peak_reserved else "",
                "e3_single_expert_bottleneck": e3_widths[0] if e3_widths else "",
                "e3_signed_parameter_delta": e3_deltas[0] if e3_deltas else "",
                "interpretation": "single-seed validation observation; no significance claim",
            }
        )
    target = DATA_DIR / "pilot_summary.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["dataset", "variant"])
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-pilot", action="store_true")
    parser.add_argument("--dataset", choices=("Movies", "Grocery", "ele-fashion"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache-entry-json", help="JSON file containing the validated cache manifest row")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.summarize_pilot:
        print(json.dumps(summarize_pilot(), indent=2))
        return
    if not args.dataset or not args.checkpoint or not args.cache_entry_json:
        parser.error("checkpoint diagnostics require --dataset, --checkpoint, and --cache-entry-json")
    entry_path = Path(args.cache_entry_json)
    if not entry_path.is_absolute():
        entry_path = ROOT / entry_path
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    result = analyze_checkpoint(
        dataset=args.dataset,
        checkpoint_path=args.checkpoint,
        cache_entry=entry,
        device=args.device,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
