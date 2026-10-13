#!/usr/bin/env python3
"""Capture frozen-checkpoint Physical Strength scalars without training or labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rmcge_v22 import (  # noqa: E402
    DATASETS,
    DATA_DIR,
    CONDITIONS,
    SEEDS,
    build_config,
    load_model_checkpoint,
    mask_test_labels,
    read_rows,
    state_sha256,
    write_rows,
)
from scripts.run_rmcge_v21 import verify_cache_entries  # noqa: E402
from src.data import load_mag_data  # noqa: E402


FIELDS = [
    "dataset", "condition", "seed", "modality", "active_node_count",
    "physical_strength_mean", "physical_strength_std", "physical_strength_min",
    "physical_strength_max", "checkpoint_sha256", "eval_cpu_rng_unchanged",
    "model_parameters_unchanged", "test_evaluation",
]


def collect(dataset: str, condition: str, seed: int, cache: dict) -> list[dict]:
    cfg = build_config(dataset, condition, seed, "cpu", cache)
    raw = load_mag_data(cfg, "nc", seed)
    safe = mask_test_labels(raw)
    # Use the registered Best Accuracy checkpoint, including the preserved-successful
    # attempt path for Fashion/M0/seed 42.
    row = next(
        row for row in read_rows(DATA_DIR / "runs.csv")
        if row["dataset"] == dataset and row["condition"] == condition and int(row["seed"]) == seed
    )
    checkpoint_path = Path(row["best_accuracy_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    _payload, model, _head, _initial = load_model_checkpoint(cfg, safe, cache, checkpoint_path, torch.device("cpu"))
    model.eval()
    model_state_before = state_sha256(model)
    captured: list[dict] = []
    original = model._physical_route

    def capture_route(modality_index, prior, active):
        route = original(modality_index, prior, active)
        values = route["strength"].detach().float()[active]
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError(f"non-finite/empty Physical Strength for {dataset}/{condition}/seed_{seed}")
        captured.append({
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "modality": ("text", "visual")[modality_index],
            "active_node_count": int(values.numel()),
            "physical_strength_mean": float(values.mean()),
            "physical_strength_std": float(values.std(unbiased=False)),
            "physical_strength_min": float(values.min()),
            "physical_strength_max": float(values.max()),
            "checkpoint_sha256": row["best_accuracy_checkpoint_sha256"],
            "test_evaluation": False,
        })
        return route

    # Instance-level replacement wraps the inherited route helper but leaves the
    # model source, parameters, and output computation unchanged.
    model._physical_route = capture_route
    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        logits, _aux, _selection_weights, _selection_scores, _details = model(
            safe.x, safe.edge_index, return_details=True
        )
    if not torch.isfinite(logits).all():
        raise ValueError(f"non-finite frozen forward for {dataset}/{condition}/seed_{seed}")
    rng_unchanged = torch.equal(rng_before, torch.random.get_rng_state())
    model_unchanged = model_state_before == state_sha256(model)
    if not rng_unchanged or not model_unchanged or len(captured) != 2:
        raise RuntimeError(f"Physical Strength capture/RNG check failed for {dataset}/{condition}/seed_{seed}")
    for result in captured:
        result["eval_cpu_rng_unchanged"] = rng_unchanged
        result["model_parameters_unchanged"] = model_unchanged
    del model, raw, safe, logits
    return captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="optional smoke limit; 0 runs the full frozen matrix")
    args = parser.parse_args()
    cache = verify_cache_entries()
    run_rows = read_rows(DATA_DIR / "runs.csv")
    expected = {(dataset, condition, seed) for dataset in DATASETS for condition in CONDITIONS for seed in SEEDS}
    actual = {(row["dataset"], row["condition"], int(row["seed"])) for row in run_rows if row["status"] == "complete"}
    if actual != expected:
        raise ValueError(f"expected 45 completed frozen runs, got missing={sorted(expected-actual)}")
    keys = [(dataset, condition, seed) for seed in SEEDS for dataset in DATASETS for condition in CONDITIONS]
    if args.limit:
        keys = keys[:args.limit]
    output = []
    for index, (dataset, condition, seed) in enumerate(keys, 1):
        output.extend(collect(dataset, condition, seed, cache[dataset]))
        print(f"PHYSICAL_STRENGTH {index}/{len(keys)} {dataset}/{condition}/seed_{seed}", flush=True)
    if args.limit:
        print(f"smoke_complete_rows={len(output)}")
        return
    if len(output) != 90:
        raise ValueError(f"expected 90 modality records, got {len(output)}")
    path = DATA_DIR / "physical_strength_diagnostics.csv"
    write_rows(path, output, FIELDS)
    print(f"wrote {path} rows={len(output)}", flush=True)


if __name__ == "__main__":
    main()
