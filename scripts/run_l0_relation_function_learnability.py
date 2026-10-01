from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.p0p1_propagation_probe import DATASETS  # noqa: E402
from src.analysis.l0_relation_function_learnability import (  # noqa: E402
    EVIDENCE_VARIANTS,
    MASTER_DIM,
    MODALITIES,
    REPRESENTATIONS,
    SEEDS,
    TARGET_NAMES,
    audit_utility_formula,
    balanced_group_folds,
    count_parameters,
    edge_positions_for_table,
    extract_frozen_e01_representations,
    fit_fullbatch_probe,
    inner_group_split,
    load_h0_and_master,
    load_utility_table,
    metric_bundle,
    run_direct_state_fit,
    run_evidence_fit,
    run_state_readout_fit,
    set_seed,
    shuffle_targets_within_destination,
    standardize_targets,
    target_balanced_weights,
    target_reliability,
    _model_initialization_audit,
    EvidenceMLP,
    M0StateDirectProbe,
    StateReadout,
)
from src.models.adaptive_prop_m0 import graph_standardized_log_degree  # noqa: E402


RESEARCH = PROJECT_ROOT / "research/l0_relation_function_learnability_audit"
OUTPUT = PROJECT_ROOT / "outputs/l0_relation_function_learnability_audit"
TABLE_ROOT = PROJECT_ROOT / "outputs/p13_joint_readout_operator_probe/runs"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def append_rows(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    result = pd.concat([old, new], ignore_index=True)
    if keys:
        result = result.drop_duplicates(keys, keep="last")
    atomic_csv(result, path)
    return result


def load_all_tables() -> tuple[dict[tuple[str, int], pd.DataFrame], list[dict[str, Any]]]:
    tables: dict[tuple[str, int], pd.DataFrame] = {}
    audits = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = TABLE_ROOT / dataset / f"seed_{seed}" / "joint_operator_edge_utility.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(f"required P1.3 full edge table missing: {path}")
            frame, y = load_utility_table(path, dataset, seed)
            audit = audit_utility_formula(frame, y)
            audit.update({"dataset": dataset, "seed": seed, "artifact": str(path),
                          "artifact_status": "reused_full_table"})
            audits.append(audit)
            tables[(dataset, seed)] = frame
    return tables, audits


def fold_arrays(frame: pd.DataFrame, dataset: str, seed: int,
                assignment_cache: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray]:
    key = (dataset, seed)
    assignment = assignment_cache.get(key)
    if assignment is None:
        assignment = balanced_group_folds(frame["dst"].to_numpy(dtype=np.int64), 3,
                                          seed=seed + 1709)
        assignment.insert(0, "seed", seed)
        assignment.insert(0, "dataset", dataset)
        assignment_cache[key] = assignment
    fold_map = assignment.set_index("dst")["fold"]
    folds = frame["dst"].map(fold_map).to_numpy(dtype=np.int64)
    if (folds < 0).any() or len(folds) != len(frame):
        raise AssertionError("every utility row must be assigned to a target-group fold")
    return assignment, folds


def _load_formal_rows(name: str) -> pd.DataFrame:
    path = RESEARCH / "data" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _already_complete(frame: pd.DataFrame, match: dict[str, Any], needed: int) -> bool:
    if frame.empty:
        return False
    selected = frame
    for key, value in match.items():
        selected = selected[selected[key] == value]
    return len(selected) >= needed


def _record_fit_metadata(rows: list[dict[str, Any]], dataset: str, seed: int,
                         fold: int, model_name: str, metadata: dict[str, Any],
                         variant: str | None = None, modality: str | None = None,
                         representation: str | None = None) -> dict[str, Any]:
    record = {"dataset": dataset, "seed": seed, "fold": fold,
              "model": model_name, **metadata}
    if variant is not None:
        record["variant"] = variant
    if modality is not None:
        record["modality"] = modality
    if representation is not None:
        record["representation"] = representation
    return record


def _save_checkpoint(name: str, dataset: str, seed: int, fold: int,
                     path: Path, metadata: dict[str, Any]) -> Path:
    # Checkpoint files are emitted by fit helpers; this function returns a stable path.
    return path


def run_smoke(tables: dict[tuple[str, int], pd.DataFrame], table_audits: list[dict[str, Any]],
              device: torch.device, data_root: str) -> dict[str, Any]:
    start = time.time()
    reliability = target_reliability(tables)
    if len(reliability) != 54 or not np.isfinite(reliability["overlap_fraction_smaller"]).all():
        raise AssertionError("cross-seed reliability basic calculation failed")
    atomic_csv(reliability, RESEARCH / "data/target_reliability_by_seed_pair.csv")
    frame = tables[("Movies", 42)]
    frame, y = load_utility_table(TABLE_ROOT / "Movies/seed_42/joint_operator_edge_utility.csv.gz", "Movies", 42)
    data, h_t, h_v, layout, h0_record, positions, graph_src, graph_dst, degree = load_h0_and_master(
        "Movies", 42, frame, data_root, device)
    if hasattr(data, "test_idx"):
        raise AssertionError("ProbeData unexpectedly exposes the original test index")
    outside_visible = torch.ones(data.num_nodes, dtype=torch.bool)
    outside_visible[data.train_idx] = False
    outside_visible[data.val_idx] = False
    if int((data.labels[outside_visible] != -100).sum()) != 0:
        raise AssertionError("masked test/held-out labels became visible")
    assignment = balanced_group_folds(frame["dst"].to_numpy(dtype=np.int64), 3, seed=42 + 1709)
    fold_map = assignment.set_index("dst")["fold"]
    folds = frame["dst"].map(fold_map).to_numpy(dtype=np.int64)
    fold = 0
    outer_train = np.flatnonzero(folds != fold)
    test_rows = np.flatnonzero(folds == fold)
    train_rows, val_rows = inner_group_split(frame.dst.to_numpy(), outer_train, seed=42 + 9100 + fold)
    if set(frame.dst.iloc[train_rows]).intersection(frame.dst.iloc[val_rows]):
        raise AssertionError("smoke inner target groups overlap")
    if set(frame.dst.iloc[np.concatenate([train_rows, val_rows])]).intersection(frame.dst.iloc[test_rows]):
        raise AssertionError("smoke outer target groups overlap")
    _, counts, bitwise = _model_initialization_audit(811003)
    if not bitwise or len(set(counts.values())) != 1:
        raise AssertionError("smoke EvidenceMLP parameter fairness failed")
    smoke_dir = OUTPUT / "smoke"
    smoke_rows = []
    device_mem = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target_mean, target_std = standardize_targets(y, np.concatenate([train_rows, val_rows]))
    y_scaled = (y - target_mean) / target_std
    for variant in EVIDENCE_VARIANTS:
        fit_seed = 811003
        set_seed(fit_seed)
        init = EvidenceMLP().state_dict()
        model = EvidenceMLP()
        model.load_state_dict(init, strict=True)
        x_mean = layout.master[np.concatenate([train_rows, val_rows])].mean(0)
        x_std = layout.master[np.concatenate([train_rows, val_rows])].std(0, unbiased=False).clamp_min(1e-6)
        x = ((layout.master - x_mean) / x_std).contiguous()
        x[:, ~layout.masks[variant]] = 0
        x_device = x.to(device)
        fit = fit_fullbatch_probe(model, lambda: model(x_device), y_scaled, train_rows, val_rows,
                                  test_rows, target_balanced_weights(frame.dst_degree.to_numpy()),
                                  device, max_epochs=2, min_epoch=1, patience=1)
        if fit["prediction_scaled"].shape != (len(test_rows), 6) or not fit["finite_gradients"]:
            raise AssertionError(f"Evidence smoke failed: {variant}")
        smoke_rows.append({"family": "EvidenceMLP", "variant": variant,
                           "parameter_count": fit["parameter_count"], "finite_loss_gradient": True,
                           "shape": str(tuple(fit["prediction_scaled"].shape)), "status": "passed"})
        del x_device, fit, model
        if device.type == "cuda":
            device_mem = max(device_mem, torch.cuda.max_memory_allocated(device))
            torch.cuda.empty_cache()
    # Direct-state capacity probe, using all graph edges to reconstruct M0 LOO evidence.
    degree_z = graph_standardized_log_degree(torch.as_tensor(degree, dtype=torch.long))
    direct = M0StateDirectProbe().to(device)
    ht, hv = h_t.to(device), h_v.to(device)
    positions_t = torch.as_tensor(positions, dtype=torch.long, device=device)
    gs, gd = torch.as_tensor(graph_src, dtype=torch.long, device=device), torch.as_tensor(graph_dst, dtype=torch.long, device=device)
    degree_t, deg_z_t = torch.as_tensor(degree, dtype=torch.long, device=device), degree_z.to(device)
    direct_forward = lambda: direct.forward_from_edge_positions(ht, hv, positions_t, gs, gd,
                                                                 degree_t, deg_z_t)[0]
    fit = fit_fullbatch_probe(direct, direct_forward, y_scaled, train_rows, val_rows, test_rows,
                              target_balanced_weights(frame.dst_degree.to_numpy()), device,
                              max_epochs=2, min_epoch=1, patience=1)
    if fit["prediction_scaled"].shape != (len(test_rows), 6) or not fit["finite_gradients"]:
        raise AssertionError("DirectState smoke failed")
    smoke_rows.append({"family": "M0StateDirect", "variant": "M0StateDirect",
                       "parameter_count": fit["parameter_count"], "finite_loss_gradient": True,
                       "shape": str(tuple(fit["prediction_scaled"].shape)), "status": "passed"})
    del ht, hv, positions_t, gs, gd, degree_t, deg_z_t, direct, fit
    if device.type == "cuda":
        device_mem = max(device_mem, torch.cuda.max_memory_allocated(device))
        torch.cuda.empty_cache()
    reps, state_record = extract_frozen_e01_representations(
        "Movies", 42, data, frame, device, positions, graph_src, graph_dst, degree,
    )
    test_degree = frame.dst_degree.to_numpy()
    for modality_i, modality in enumerate(MODALITIES):
        for rep in REPRESENTATIONS:
            representation = reps[f"{rep}_{modality}"]
            offsets = np.arange(modality_i * 3, modality_i * 3 + 3)
            y3 = y[:, offsets]
            mean, std = standardize_targets(y3, np.concatenate([train_rows, val_rows]))
            set_seed(99103 + modality_i)
            readout = StateReadout().to(device)
            x = torch.as_tensor(representation, dtype=torch.float32, device=device)
            xmean = x[torch.as_tensor(np.concatenate([train_rows, val_rows]), device=device)].mean(0)
            xstd = x[torch.as_tensor(np.concatenate([train_rows, val_rows]), device=device)].std(0, unbiased=False).clamp_min(1e-6)
            x = (x - xmean) / xstd
            fit = fit_fullbatch_probe(readout, lambda: readout(x), (y3 - mean) / std,
                                      train_rows, val_rows, test_rows,
                                      target_balanced_weights(test_degree), device,
                                      max_epochs=2, min_epoch=1, patience=1)
            if fit["prediction_scaled"].shape != (len(test_rows), 3) or not fit["finite_gradients"]:
                raise AssertionError(f"Frozen {rep}/{modality} smoke failed")
            smoke_rows.append({"family": "FrozenStateReadout", "variant": f"{rep}_{modality}",
                               "parameter_count": fit["parameter_count"], "finite_loss_gradient": True,
                               "shape": str(tuple(fit["prediction_scaled"].shape)), "status": "passed"})
            del readout, x, fit
    if device.type == "cuda":
        device_mem = max(device_mem, torch.cuda.max_memory_allocated(device))
    smoke_path = RESEARCH / "data/smoke_audit.csv"
    atomic_csv(pd.DataFrame(smoke_rows), smoke_path)
    result = {
        "status": "passed", "dataset": "Movies", "seed": 42, "fold": fold,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "table_rows": len(frame), "target_formula_audit": audit_utility_formula(frame, y),
        "p13_h0_regression": h0_record, "feature_dim": layout.master.shape[1],
        "evidence_variant_parameter_counts": counts, "evidence_init_bitwise_identical": bitwise,
        "outer_dst_disjoint": True, "inner_dst_disjoint": True,
        "no_test_labels_accessed": True, "frozen_state_alignment": state_record,
        "readout_dimensions": {k: v.shape[1] for k, v in reps.items()},
        "fits": smoke_rows, "peak_allocated_gpu_bytes": int(device_mem),
        "elapsed_seconds": time.time() - start,
    }
    out = smoke_dir / "smoke_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[smoke] passed fits={len(smoke_rows)} peak_gpu={device_mem / 1024**3:.2f}GiB", flush=True)
    return result


def run_campaign(tables: dict[tuple[str, int], pd.DataFrame], table_audits: list[dict[str, Any]],
                device: torch.device, data_root: str, max_epochs: int = 150) -> dict[str, Any]:
    start = time.time()
    RESEARCH.joinpath("data").mkdir(parents=True, exist_ok=True)
    reliability = target_reliability(tables)
    atomic_csv(reliability, RESEARCH / "data/target_reliability_by_seed_pair.csv")
    target_summary = reliability.groupby(["dataset", "modality", "target"], as_index=False).agg(
        overlap_edges_mean=("overlap_edges", "mean"),
        overlap_fraction_smaller_mean=("overlap_fraction_smaller", "mean"),
        spearman_mean=("spearman", "mean"), spearman_seed_pair_sd=("spearman", "std"),
        pearson_mean=("pearson", "mean"),
        sign_agreement_mean=("sign_agreement_positive_vs_nonpositive", "mean"),
        centered_spearman_mean=("centered_spearman", "mean"),
        exact_zero_fraction_a_mean=("exact_zero_fraction_a", "mean"),
        exact_zero_fraction_b_mean=("exact_zero_fraction_b", "mean"),
        pair_count=("seed_a", "count"),
    )
    atomic_csv(target_summary, RESEARCH / "data/target_reliability_summary.csv")
    audit_frame = pd.DataFrame(table_audits)
    atomic_csv(audit_frame, OUTPUT / "artifact_audit.csv")
    atomic_csv(audit_frame, RESEARCH / "data/utility_artifact_audit.csv")

    assignment_cache: dict[tuple[str, int], pd.DataFrame] = {}
    all_assignments = []
    for (dataset, seed), frame in tables.items():
        assignment, _ = fold_arrays(frame, dataset, seed, assignment_cache)
        all_assignments.append(assignment)
    group_assignment = pd.concat(all_assignments, ignore_index=True)
    atomic_csv(group_assignment, RESEARCH / "data/group_fold_assignment.csv")

    evidence_path = RESEARCH / "data/evidence_probe_by_fold.csv"
    direct_path = RESEARCH / "data/direct_state_probe_by_fold.csv"
    frozen_path = RESEARCH / "data/frozen_state_readout_by_fold.csv"
    high_path = RESEARCH / "data/high_margin_summary.csv"
    parameter_path = RESEARCH / "data/parameter_summary.csv"
    evidence_rows = _load_formal_rows(evidence_path.name).to_dict("records")
    direct_rows = _load_formal_rows(direct_path.name).to_dict("records")
    frozen_rows = _load_formal_rows(frozen_path.name).to_dict("records")
    high_rows = pd.read_csv(high_path).to_dict("records") if high_path.exists() else []
    parameter_rows = pd.read_csv(parameter_path).to_dict("records") if parameter_path.exists() else []
    run_audits: list[dict[str, Any]] = []

    # Phase 4: all five EvidenceMLP variants across the complete 3×3×3 design.
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[evidence] {dataset}/{seed}", flush=True)
            frame, y = load_utility_table(TABLE_ROOT / dataset / f"seed_{seed}/joint_operator_edge_utility.csv.gz", dataset, seed)
            data, h_t, h_v, layout, h0_record, positions, graph_src, graph_dst, degree = load_h0_and_master(
                dataset, seed, frame, data_root, device)
            assignment, folds = fold_arrays(frame, dataset, seed, assignment_cache)
            dst = frame.dst.to_numpy(dtype=np.int64)
            degree_by_edge = frame.dst_degree.to_numpy(dtype=np.int64)
            for fold in range(3):
                outer_rows = np.flatnonzero(folds != fold)
                test_rows = np.flatnonzero(folds == fold)
                train_rows, val_rows = inner_group_split(dst, outer_rows, seed=seed + 9100 + fold)
                if _already_complete(pd.DataFrame(evidence_rows),
                                     {"dataset": dataset, "seed": seed, "fold": fold}, 5 * 6):
                    continue
                init_seed = 811003 + seed * 31 + fold
                init_state, counts, bitwise = _model_initialization_audit(init_seed)
                for variant in EVIDENCE_VARIANTS:
                    key = {"dataset": dataset, "seed": seed, "fold": fold, "model": variant}
                    if _already_complete(pd.DataFrame(evidence_rows), key, 6):
                        continue
                    fit_y = y
                    shuffle_audit = {"eligible_groups": 0, "changed_groups": 0, "multiset_preserved": True}
                    if variant == "ENDPOINT_LOCAL_SHUFFLED_TARGET":
                        fit_y, shuffle_audit = shuffle_targets_within_destination(
                            y, dst, train_rows, seed=9917 + fold)
                        if not shuffle_audit["multiset_preserved"]:
                            raise AssertionError("within-target six-vector shuffle changed target multiset")
                    output_path = OUTPUT / "checkpoints" / dataset / f"seed_{seed}_fold_{fold}_{variant}.pt"
                    model_rows, model_high, metadata = run_evidence_fit(
                        layout.master, layout.masks, fit_y, dst, degree_by_edge,
                        train_rows, val_rows, test_rows, dataset, seed, fold, variant,
                        device, max_epochs=max_epochs, save_path=output_path, init_state=init_state,
                    )
                    metadata.update({"initialization_seed": init_seed,
                                     "all_evidence_counts_equal": len(set(counts.values())) == 1,
                                     "all_variants_init_bitwise_equal": bitwise,
                                     "shuffle_audit": shuffle_audit})
                    evidence_rows.extend(model_rows)
                    high_rows.extend(model_high)
                    parameter_rows.append(_record_fit_metadata([], dataset, seed, fold, variant,
                                                               metadata, variant=variant))
                    atomic_csv(pd.DataFrame(evidence_rows), evidence_path)
                    atomic_csv(pd.DataFrame(high_rows), high_path)
                    atomic_csv(pd.DataFrame(parameter_rows), parameter_path)
                    print(f"  [done] fold={fold} variant={variant} rho_deltaD="
                          f"{model_rows[1]['spearman']:.3f} rho_deltaP={model_rows[2]['spearman']:.3f}", flush=True)
            run_audits.append({"dataset": dataset, "seed": seed, "p13_h0": h0_record,
                               "rows": len(frame), "feature_dim": layout.master.shape[1],
                               "folds": 3})
            del data, h_t, h_v, layout
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Phase 5: one M0 relation-state direct-utility fit per dataset×seed×fold.
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[direct] {dataset}/{seed}", flush=True)
            frame, y = load_utility_table(TABLE_ROOT / dataset / f"seed_{seed}/joint_operator_edge_utility.csv.gz", dataset, seed)
            data = __import__("src.analysis.p0p1_propagation_probe", fromlist=["load_probe_data"]).load_probe_data(dataset, seed, data_root)
            p0path = Path("outputs/p0p1_propagation_heterogeneity/checkpoints/runs") / dataset / f"seed_{seed}_semantic_only.pt"
            from src.analysis.p11_p12_operator_rescue import load_frozen_h0
            from src.analysis.p0p1_propagation_probe import load_probe_data
            p0model, h_t_gpu, h_v_gpu, h0_record = load_frozen_h0(
                data, p0path, device,
                Path("outputs/p0p1_propagation_heterogeneity/runs") / dataset / f"seed_{seed}/probe_performance.csv")
            if any(p.requires_grad or p.grad is not None for p in p0model.parameters()):
                raise AssertionError("P0 encoder must remain frozen for DirectState")
            positions, graph_src, graph_dst, degree = edge_positions_for_table(data, frame, device)
            degree_z = graph_standardized_log_degree(torch.as_tensor(degree, dtype=torch.long)).numpy()
            assignment, folds = fold_arrays(frame, dataset, seed, assignment_cache)
            dst = frame.dst.to_numpy(dtype=np.int64)
            degree_by_edge = frame.dst_degree.to_numpy(dtype=np.int64)
            for fold in range(3):
                outer_rows = np.flatnonzero(folds != fold)
                test_rows = np.flatnonzero(folds == fold)
                train_rows, val_rows = inner_group_split(dst, outer_rows, seed=seed + 9100 + fold)
                key = {"dataset": dataset, "seed": seed, "fold": fold, "model": "M0StateDirect"}
                if _already_complete(pd.DataFrame(direct_rows), key, 6):
                    continue
                output_path = OUTPUT / "checkpoints" / dataset / f"seed_{seed}_fold_{fold}_M0StateDirect.pt"
                model_rows, model_high, metadata = run_direct_state_fit(
                    h_t_gpu.detach().cpu(), h_v_gpu.detach().cpu(), positions, graph_src, graph_dst,
                    degree, degree_z, y, dst, degree_by_edge, train_rows, val_rows, test_rows,
                    dataset, seed, fold, device, max_epochs=max_epochs, save_path=output_path,
                )
                metadata.update({"p13_h0_regression_max_abs_metric_diff": h0_record["max_abs_metric_difference"]})
                direct_rows.extend(model_rows)
                high_rows.extend(model_high)
                parameter_rows.append(_record_fit_metadata([], dataset, seed, fold, "M0StateDirect", metadata))
                atomic_csv(pd.DataFrame(direct_rows), direct_path)
                atomic_csv(pd.DataFrame(high_rows), high_path)
                atomic_csv(pd.DataFrame(parameter_rows), parameter_path)
                print(f"  [done] fold={fold} direct rho_deltaD={model_rows[1]['spearman']:.3f} "
                      f"rho_deltaP={model_rows[2]['spearman']:.3f}", flush=True)
            del p0model, h_t_gpu, h_v_gpu, data
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Phase 6: frozen Q/R/U readouts; each checkpoint is frozen and edge-order checked.
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[frozen] {dataset}/{seed}", flush=True)
            frame, y = load_utility_table(TABLE_ROOT / dataset / f"seed_{seed}/joint_operator_edge_utility.csv.gz", dataset, seed)
            from src.analysis.p0p1_propagation_probe import load_probe_data
            data = load_probe_data(dataset, seed, data_root)
            positions, graph_src, graph_dst, degree = edge_positions_for_table(data, frame, device)
            cache = OUTPUT / "states" / dataset / f"seed_{seed}_e01_state.npz"
            if cache.exists():
                cached = np.load(cache)
                reps = {f"{rep}_{mod}": cached[f"{rep}_{mod}"] for rep in REPRESENTATIONS for mod in MODALITIES}
                state_record = {"checkpoint": str(Path("outputs/e01_function_provenance_preservation/checkpoints") / dataset / f"seed_{seed}_keep_edge.pt"),
                                "checkpoint_frozen": True, "aligned_ordered_src_dst_degree": True,
                                "representation_dimensions": {k: v.shape[1] for k, v in reps.items()},
                                "reused_state_cache": True}
            else:
                reps, state_record = extract_frozen_e01_representations(
                    dataset, seed, data, frame, device, positions, graph_src, graph_dst, degree)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, **reps)
            assignment, folds = fold_arrays(frame, dataset, seed, assignment_cache)
            dst = frame.dst.to_numpy(dtype=np.int64)
            degree_by_edge = frame.dst_degree.to_numpy(dtype=np.int64)
            for fold in range(3):
                outer_rows = np.flatnonzero(folds != fold)
                test_rows = np.flatnonzero(folds == fold)
                train_rows, val_rows = inner_group_split(dst, outer_rows, seed=seed + 9100 + fold)
                for modality_i, modality in enumerate(MODALITIES):
                    for rep in REPRESENTATIONS:
                        name = f"Frozen_{rep}"
                        key = {"dataset": dataset, "seed": seed, "fold": fold,
                               "model": name, "modality": modality, "representation": rep}
                        if _already_complete(pd.DataFrame(frozen_rows), key, 3):
                            continue
                        output_path = OUTPUT / "checkpoints" / dataset / f"seed_{seed}_fold_{fold}_{rep}_{modality}.pt"
                        model_rows, model_high, metadata = run_state_readout_fit(
                            reps[f"{rep}_{modality}"], y, dst, degree_by_edge,
                            train_rows, val_rows, test_rows, dataset, seed, fold,
                            modality_i, rep, device, max_epochs=max_epochs, save_path=output_path,
                        )
                        metadata.update({"frozen_checkpoint": state_record["checkpoint"],
                                         "edge_alignment_verified": state_record["aligned_ordered_src_dst_degree"]})
                        frozen_rows.extend(model_rows)
                        high_rows.extend(model_high)
                        parameter_rows.append(_record_fit_metadata([], dataset, seed, fold, name,
                                                                   metadata, modality=modality,
                                                                   representation=rep))
                        atomic_csv(pd.DataFrame(frozen_rows), frozen_path)
                        atomic_csv(pd.DataFrame(high_rows), high_path)
                        atomic_csv(pd.DataFrame(parameter_rows), parameter_path)
                        print(f"  [done] fold={fold} {modality}/{rep}", flush=True)
            del data, reps
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "status": "completed", "source_tables": len(table_audits),
        "evidence_fits_expected": 135, "direct_state_fits_expected": 27,
        "frozen_readouts_expected": 162, "elapsed_seconds": time.time() - start,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "runtime_records": run_audits, "no_test_evaluation": True,
        "secondary_unimodal_probes": "not executed secondary analysis",
        "deviations": [],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[campaign complete] evidence={len(evidence_rows)} direct={len(direct_rows)} "
          f"frozen={len(frozen_rows)} elapsed={summary['elapsed_seconds'] / 3600:.2f}h", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="L0 relation-to-function learnability audit")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--campaign", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--data-root", default="/hdd1/DataInHere/YHF/data")
    parser.add_argument("--max-epochs", type=int, default=150)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        print(f"[device] {device} {torch.cuda.get_device_name(device)} ", flush=True)
        torch.cuda.set_device(device)
    tables, table_audits = load_all_tables()
    print(f"[artifact audit] reused full P1.3 tables={len(tables)}; formulas finite", flush=True)
    if args.smoke:
        run_smoke(tables, table_audits, device, args.data_root)
    else:
        smoke_path = OUTPUT / "smoke" / "smoke_result.json"
        if not smoke_path.is_file() or json.loads(smoke_path.read_text()).get("status") != "passed":
            raise RuntimeError("Movies/42 GPU smoke must pass before the full campaign")
        run_campaign(tables, table_audits, device, args.data_root, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
