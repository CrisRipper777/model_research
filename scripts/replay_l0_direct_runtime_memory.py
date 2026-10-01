from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.l0_relation_function_learnability import (
    SEEDS, balanced_group_folds, edge_positions_for_table, inner_group_split,
    load_h0_and_master, load_utility_table, run_direct_state_fit,
)
from src.analysis.p0p1_propagation_probe import DATASETS
from src.models.adaptive_prop_m0 import graph_standardized_log_degree

TABLE_ROOT = ROOT / "outputs/p13_joint_readout_operator_probe/runs"
DATA_ROOT = ROOT.parent / "data"
DATA_DIR = ROOT / "research/l0_relation_function_learnability_audit/data"
OUTPUT = DATA_DIR / "direct_state_fit_runtime_memory_audit.csv"
METRICS = ["spearman", "pearson", "r2", "sign_auroc", "sign_balanced_accuracy",
           "sign_prevalence", "within_target_residual_spearman",
           "per_target_eligible_count", "per_target_valid_count", "per_target_rank_mean",
           "per_target_rank_median", "per_target_rank_q25", "per_target_rank_q75"]
METRIC_REPLAY_TOLERANCE = 1e-2
LOSS_REPLAY_TOLERANCE = 1e-5


def main() -> None:
    device = torch.device("cuda:1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested GPU fit instrumentation")
    torch.cuda.set_device(device)
    historical = pd.read_csv(DATA_DIR / "direct_state_probe_by_fold.csv")
    records = []

    for dataset in DATASETS:
        for seed in SEEDS:
            path = TABLE_ROOT / dataset / f"seed_{seed}" / "joint_operator_edge_utility.csv.gz"
            frame, targets = load_utility_table(path, dataset, seed)
            data, h_t, h_v, _, h0_record, positions, graph_src, graph_dst, degree = load_h0_and_master(
                dataset, seed, frame, str(DATA_ROOT), device
            )
            degree_z = graph_standardized_log_degree(torch.as_tensor(degree, dtype=torch.long)).numpy()
            assignment = balanced_group_folds(frame.dst.to_numpy(dtype=np.int64), 3, seed=seed + 1709)
            fold_map = assignment.set_index("dst").fold
            dst = frame.dst.to_numpy(dtype=np.int64)
            fold_ids = frame.dst.map(fold_map).to_numpy(dtype=np.int64)
            degree_by_edge = frame.dst_degree.to_numpy(dtype=np.int64)

            for fold in range(3):
                outer_train = np.flatnonzero(fold_ids != fold)
                test_rows = np.flatnonzero(fold_ids == fold)
                train_rows, val_rows = inner_group_split(dst, outer_train, seed=seed + 9100 + fold)
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline_bytes = torch.cuda.memory_allocated(device)
                start = time.perf_counter()
                rows, _, metadata = run_direct_state_fit(
                    h_t, h_v, positions, graph_src, graph_dst, degree, degree_z,
                    targets, dst, degree_by_edge, train_rows, val_rows, test_rows,
                    dataset, seed, fold, device, max_epochs=150, save_path=None,
                )
                torch.cuda.synchronize(device)
                fit_seconds = time.perf_counter() - start
                peak_bytes = torch.cuda.max_memory_allocated(device)

                prior = historical[(historical.dataset == dataset) &
                                   (historical.seed == seed) &
                                   (historical.fold == fold)]
                current = pd.DataFrame(rows)
                joined = prior.merge(current, on=["dataset", "seed", "fold", "model",
                                                  "modality", "target_type"],
                                     suffixes=("_original", "_replay"), validate="one_to_one")
                if len(joined) != 6:
                    raise AssertionError(f"{dataset}/{seed}/{fold}: replay rows failed to align")
                differences = []
                for metric in METRICS:
                    a = joined[f"{metric}_original"].to_numpy(dtype=float)
                    b = joined[f"{metric}_replay"].to_numpy(dtype=float)
                    finite = np.isfinite(a) & np.isfinite(b)
                    if finite.any():
                        differences.extend(np.abs(a[finite] - b[finite]).tolist())
                    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                        raise AssertionError(f"{dataset}/{seed}/{fold}: {metric} finite mask changed")
                max_metric_diff = max(differences, default=0.0)
                original_best = torch.load(
                    ROOT / "outputs/l0_relation_function_learnability_audit/checkpoints"
                    / dataset / f"seed_{seed}_fold_{fold}_M0StateDirect.pt",
                    map_location="cpu", weights_only=False,
                )["metadata"]
                best_epoch_match = int(metadata["best_epoch"]) == int(original_best["best_epoch"])
                best_loss_diff = abs(float(metadata["best_inner_weighted_huber"])
                                      - float(original_best["best_inner_weighted_huber"]))
                if not best_epoch_match or best_loss_diff > LOSS_REPLAY_TOLERANCE:
                    raise AssertionError(
                        f"{dataset}/{seed}/{fold}: replay training metadata mismatch "
                        f"metric={max_metric_diff} epoch_match={best_epoch_match} loss={best_loss_diff}"
                    )
                records.append({
                    "dataset": dataset, "seed": seed, "fold": fold,
                    "fit_call_wall_seconds": fit_seconds,
                    "cuda_baseline_allocated_bytes": int(baseline_bytes),
                    "cuda_peak_allocated_bytes": int(peak_bytes),
                    "cuda_peak_incremental_bytes": int(max(0, peak_bytes - baseline_bytes)),
                    "best_epoch": int(metadata["best_epoch"]),
                    "best_inner_weighted_huber": float(metadata["best_inner_weighted_huber"]),
                    "metrics_within_replay_tolerance": bool(max_metric_diff <= METRIC_REPLAY_TOLERANCE),
                    "metric_replay_tolerance": METRIC_REPLAY_TOLERANCE,
                    "max_metric_abs_difference": float(max_metric_diff),
                    "best_epoch_matches_original": best_epoch_match,
                    "best_inner_loss_abs_difference": float(best_loss_diff),
                    "p13_h0_regression_max_abs_metric_diff": float(
                        h0_record["max_abs_metric_difference"]
                    ),
                })
                print(f"[replay] {dataset}/{seed}/fold{fold} {fit_seconds:.2f}s "
                      f"peak={peak_bytes / 1024**3:.3f}GiB delta={max_metric_diff:.2e}",
                      flush=True)
            del data, h_t, h_v
            torch.cuda.empty_cache()

    audit = pd.DataFrame(records).sort_values(["dataset", "seed", "fold"])
    if len(audit) != 27 or not audit.best_epoch_matches_original.all():
        raise AssertionError("DirectState instrumentation replay incomplete")
    audit.to_csv(OUTPUT, index=False)
    summary = {
        "status": "passed",
        "purpose": "instrumented deterministic replay; original campaign metrics/checkpoints unchanged",
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "replay_fits": len(audit),
        "total_fit_call_wall_seconds": float(audit.fit_call_wall_seconds.sum()),
        "max_cuda_peak_allocated_bytes": int(audit.cuda_peak_allocated_bytes.max()),
        "max_cuda_peak_allocated_gib": float(audit.cuda_peak_allocated_bytes.max() / 1024**3),
        "max_metric_abs_difference": float(audit.max_metric_abs_difference.max()),
        "all_best_epochs_and_validation_losses_reproduced": bool(
            audit.best_epoch_matches_original.all()
            and (audit.best_inner_loss_abs_difference <= LOSS_REPLAY_TOLERANCE).all()
        ),
        "all_fold_metrics_within_tolerance": bool(
            (audit.max_metric_abs_difference <= METRIC_REPLAY_TOLERANCE).all()
        ),
        "metric_replay_tolerance": METRIC_REPLAY_TOLERANCE,
        "csv": str(OUTPUT.relative_to(ROOT)),
    }
    (DATA_DIR / "direct_state_fit_runtime_memory_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
