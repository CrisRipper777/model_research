from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.p0p1_propagation_probe import (  # noqa: E402
    DATASETS,
    HIDDEN_DIM,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    LEARNING_RATE,
    ProbeData,
    build_edge_table,
    deterministic_edge_sample,
    group_intervention_rows,
    load_probe_data,
    neighbor_mean,
    relative_ce_utility,
    set_all_seeds,
    slice_text_visual,
    train_probe,
)


def _correctness_smoke(data: ProbeData, seed: int, device: torch.device) -> dict[str, float]:
    """Fast assertions for the protocol and counterfactual arithmetic."""
    if hasattr(data, "test_idx"):
        raise AssertionError("ProbeData must not expose test_idx")
    if data.edge_index.numel() and (data.edge_index[0] == data.edge_index[1]).any():
        raise AssertionError("self-loops reached the probe")
    if torch.isin(data.train_idx, data.val_idx).any():
        raise AssertionError("train and validation indices overlap")
    if data.labels[torch.cat([data.train_idx, data.val_idx])].min() < 0:
        raise AssertionError("visible train/validation labels must be valid class IDs")

    # Text and Visual boundaries follow the configured joint [Text, Visual] order.
    joined = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    text, visual = slice_text_visual(joined, 2, 3)
    assert torch.equal(text, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(visual, torch.tensor([[3.0, 4.0, 5.0]]))

    # Directed incoming mean excludes self messages and uses src -> dst.
    toy_h = torch.tensor([[1.0], [3.0], [100.0]])
    toy_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    toy_context, toy_degree = neighbor_mean(toy_h, toy_edges, 3)
    assert torch.equal(toy_degree, torch.tensor([0, 1, 1]))
    assert torch.equal(toy_context.squeeze(-1), torch.tensor([0.0, 1.0, 3.0]))
    assert torch.equal(toy_context[2], toy_h[1])  # own value 100 is never mixed in

    # Removing one message retains the original degree in the divisor.
    two_message_h = torch.tensor([[2.0], [6.0], [0.0]])
    two_message_edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    full_context, full_degree = neighbor_mean(two_message_h, two_message_edges, 3)
    removed_context = full_context[2] - two_message_h[0] / full_degree[2]
    assert torch.allclose(full_context[2], torch.tensor([4.0]))
    assert torch.allclose(removed_context, torch.tensor([3.0]))

    # The signed utility definition matches its stated helpful/harmful direction.
    assert relative_ce_utility(torch.tensor([1.0]), torch.tensor([2.0])).item() > 0
    assert relative_ce_utility(torch.tensor([1.0]), torch.tensor([0.5])).item() < 0

    # Fast edge-logit changes equal a brute-force fixed-denominator recomputation.
    set_all_seeds(seed)
    n, hdim, classes = 4, 8, 3
    from src.analysis.p0p1_propagation_probe import NeutralOneHopProbe

    network = NeutralOneHopProbe(5, 6, classes, hdim, contextual=True).to(device).eval()
    h_t = torch.randn(n, hdim, device=device)
    h_v = torch.randn(n, hdim, device=device)
    edges = torch.tensor([[0, 1, 0, 2], [2, 2, 1, 3]], dtype=torch.long, device=device)
    c_t, degree = neighbor_mean(h_t, edges, n)
    c_v, _ = neighbor_mean(h_v, edges, n)
    picked_src, picked_dst = edges[:, :2]
    from src.analysis.p0p1_propagation_probe import validate_fast_utility

    errors = validate_fast_utility(
        network, h_t, h_v, c_t, c_v, torch.tensor([0, 1, 2, 0], device=device),
        picked_src, picked_dst, degree,
    )
    # A controlled split mapping fails if the accessor ever asks for test_idx.
    from src.analysis.p0p1_propagation_probe import read_train_val_indices

    train, val = read_train_val_indices(
        {"train_idx": torch.tensor([0]), "val_idx": torch.tensor([1]), "test_idx": object()}
    )
    assert train.tolist() == [0] and val.tolist() == [1]
    # Sampling is deterministic for fixed table order and seed.
    sample_input = pd.DataFrame({"value": np.arange(1000)})
    sample_a = deterministic_edge_sample(sample_input, 37, seed)
    sample_b = deterministic_edge_sample(sample_input, 37, seed)
    assert sample_a.equals(sample_b)
    return errors


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_one(
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    smoke: bool = False,
) -> None:
    data = load_probe_data(dataset, seed, args.data_root)
    print(
        f"[load] {dataset} seed={seed} nodes={data.num_nodes} directed_edges={data.edge_index.size(1)} "
        f"train={data.train_idx.numel()} val={data.val_idx.numel()} "
        f"features=({data.x_t.size(1)},{data.x_v.size(1)})",
        flush=True,
    )
    correctness = _correctness_smoke(data, seed, device)
    print(f"[correctness] {dataset} seed={seed} fast_vs_brute_max_abs={correctness}", flush=True)

    run_root = args.output_dir / ("smoke" if smoke else "runs") / dataset / f"seed_{seed}"
    checkpoint_root = args.output_dir / "checkpoints" / ("smoke" if smoke else "runs") / dataset
    epochs = min(args.max_epochs, 3) if smoke else args.max_epochs
    perf_rows = []
    fits = {}
    for contextual, mode in ((False, "semantic_only"), (True, "uniform_1hop")):
        checkpoint = checkpoint_root / f"seed_{seed}_{mode}.pt"
        fit = train_probe(
            data=data,
            device=device,
            seed=seed,
            contextual=contextual,
            hidden_dim=args.hidden_dim,
            max_epochs=epochs,
            patience=min(args.patience, max(1, epochs // 2)) if smoke else args.patience,
            min_epoch=min(args.min_epoch, epochs) if smoke else args.min_epoch,
            checkpoint_path=checkpoint,
        )
        fit.network.seed = seed
        fits[mode] = fit
        perf_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "probe": mode,
                **fit.metrics,
                "checkpoint": str(checkpoint),
            }
        )
        print(f"[fit] {dataset} seed={seed} probe={mode} metrics={fit.metrics}", flush=True)

    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perf_rows).to_csv(run_root / "probe_performance.csv", index=False)
    if smoke:
        # Smoke output is isolated from campaign outputs and is not consumed by analysis.
        table = build_edge_table(data, fits["uniform_1hop"], device, correctness_check=True)
        table.head(min(128, len(table))).to_csv(run_root / "edge_smoke.csv", index=False)
        _write_json(
            run_root / "smoke_result.json",
            {
                "dataset": dataset,
                "seed": seed,
                "status": "passed",
                "correctness_max_abs_error": correctness,
                "test_evaluation": False,
                "visible_indices": ["train_idx", "val_idx"],
                "probe_performance": perf_rows,
            },
        )
        return

    edge_table = build_edge_table(data, fits["uniform_1hop"], device, correctness_check=True)
    raw_path = run_root / "validation_message_evidence.csv.gz"
    edge_table.to_csv(raw_path, index=False, compression="gzip")
    interventions = group_intervention_rows(
        data, fits["uniform_1hop"], edge_table, seed, device, repeats=10
    )
    pd.DataFrame(interventions).to_csv(run_root / "group_interventions.csv", index=False)
    _write_json(
        run_root / "run_result.json",
        {
            "dataset": dataset,
            "seed": seed,
            "status": "completed",
            "nodes": data.num_nodes,
            "directed_edges": int(data.edge_index.size(1)),
            "train_nodes": int(data.train_idx.numel()),
            "validation_nodes": int(data.val_idx.numel()),
            "validation_target_messages": len(edge_table),
            "correctness_max_abs_error": correctness,
            "probe_performance": perf_rows,
            "edge_evidence_path": str(raw_path),
            "group_intervention_repeats": 10,
            "test_evaluation": False,
        },
    )
    print(
        f"[saved] {dataset} seed={seed} messages={len(edge_table)} "
        f"edges={raw_path} interventions={len(interventions)}",
        flush=True,
    )
    del edge_table, data, fits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neutral P0/P1 MAG propagation probes")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-root", default="/hdd1/DataInHere/YHF/data")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p0p1_propagation_heterogeneity"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--min-epoch", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="Movies/seed42 correctness and short fit only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.datasets, args.seeds = ["Movies"], [42]
    if args.hidden_dim != HIDDEN_DIM and not args.smoke:
        print(f"[config] hidden_dim={args.hidden_dim}", flush=True)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    print(
        f"[campaign] datasets={args.datasets} seeds={args.seeds} device={device} "
        f"hidden={args.hidden_dim} epochs={min(args.max_epochs, 3) if args.smoke else args.max_epochs} "
        f"lr={LEARNING_RATE} weight_decay={WEIGHT_DECAY}",
        flush=True,
    )
    for dataset in args.datasets:
        for seed in args.seeds:
            run_one(dataset, seed, args, device, smoke=args.smoke)


if __name__ == "__main__":
    main()
