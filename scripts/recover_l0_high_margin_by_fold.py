from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.l0_relation_function_learnability import (  # noqa: E402
    DATASETS, EVIDENCE_VARIANTS, MODALITIES, REPRESENTATIONS, SEEDS,
    EvidenceMLP, M0StateDirectProbe, StateReadout, edge_positions_for_table,
    feature_stats, high_margin_metrics, inner_group_split, load_h0_and_master,
    load_utility_table, balanced_group_folds, standardize_targets,
)
from src.analysis.p0p1_propagation_probe import load_probe_data  # noqa: E402
from src.models.adaptive_prop_m0 import graph_standardized_log_degree  # noqa: E402


TABLES = ROOT / "outputs/p13_joint_readout_operator_probe/runs"
OUT = ROOT / "outputs/l0_relation_function_learnability_audit/checkpoints"
RESEARCH = ROOT / "research/l0_relation_function_learnability_audit/data"


def load_fold(frame: pd.DataFrame, dataset: str, seed: int):
    assignment = balanced_group_folds(frame.dst.to_numpy(dtype=np.int64), 3, seed=seed + 1709)
    mapping = assignment.set_index("dst").fold
    folds = frame.dst.map(mapping).to_numpy(dtype=np.int64)
    dst = frame.dst.to_numpy(dtype=np.int64)
    for fold in range(3):
        outer = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        train, val = inner_group_split(dst, outer, seed=seed + 9100 + fold)
        yield fold, outer, train, val, test


def recover(device: torch.device) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[recover] {dataset}/{seed}", flush=True)
            frame, y = load_utility_table(TABLES / dataset / f"seed_{seed}/joint_operator_edge_utility.csv.gz", dataset, seed)
            data, h_t, h_v, layout, _, positions, graph_src, graph_dst, degree = load_h0_and_master(
                dataset, seed, frame, "/hdd1/DataInHere/YHF/data", device)
            dst = frame.dst.to_numpy(dtype=np.int64)
            degree_edge = frame.dst_degree.to_numpy(dtype=np.int64)
            npz_path = ROOT / "outputs/l0_relation_function_learnability_audit/states" / dataset / f"seed_{seed}_e01_state.npz"
            state_reps = np.load(npz_path)
            folds = list(load_fold(frame, dataset, seed))
            for fold, outer, train, val, test in folds:
                y_mean, y_std = standardize_targets(y, outer)
                for variant in EVIDENCE_VARIANTS:
                    ckpt_path = OUT / dataset / f"seed_{seed}_fold_{fold}_{variant}.pt"
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    mean, std = feature_stats(layout.master, outer)
                    x = ((layout.master - mean) / std).contiguous()
                    x[:, ~layout.masks[variant]] = 0
                    model = EvidenceMLP().to(device).eval()
                    model.load_state_dict(ckpt["state_dict"], strict=True)
                    with torch.no_grad():
                        pred = model(x.to(device))[torch.as_tensor(test, device=device)].cpu().numpy() * y_std + y_mean
                    for idx in (1, 2, 4, 5):
                        target_i = idx % 3
                        result = high_margin_metrics(y[test, idx], pred[:, idx])
                        rows.append({"dataset": dataset, "seed": seed, "fold": fold, "model": variant,
                                     "modality": MODALITIES[idx // 3],
                                     "target_type": "delta_absdiff" if target_i == 1 else "delta_product",
                                     **result})
                    del model, ckpt

                # DirectState: one frozen H0 graph reconstruction and one output pass per outer fold.
                data_cpu = data
                from src.analysis.p11_p12_operator_rescue import load_frozen_h0
                h0_model, ht, hv, _ = load_frozen_h0(
                    data_cpu,
                    ROOT / "outputs/p0p1_propagation_heterogeneity/checkpoints/runs" / dataset / f"seed_{seed}_semantic_only.pt",
                    device,
                    ROOT / "outputs/p0p1_propagation_heterogeneity/runs" / dataset / f"seed_{seed}/probe_performance.csv")
                degree_z = graph_standardized_log_degree(torch.as_tensor(degree, dtype=torch.long)).to(device)
                direct_path = OUT / dataset / f"seed_{seed}_fold_{fold}_M0StateDirect.pt"
                direct_ckpt = torch.load(direct_path, map_location="cpu", weights_only=False)
                direct = M0StateDirectProbe().to(device).eval()
                direct.load_state_dict(direct_ckpt["state_dict"], strict=True)
                pos = torch.as_tensor(positions, dtype=torch.long, device=device)
                gs = torch.as_tensor(graph_src, dtype=torch.long, device=device)
                gd = torch.as_tensor(graph_dst, dtype=torch.long, device=device)
                deg = torch.as_tensor(degree, dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = direct.forward_from_edge_positions(ht, hv, pos, gs, gd, deg, degree_z)[0]
                    pred = pred[torch.as_tensor(test, device=device)].cpu().numpy() * y_std + y_mean
                for idx in (1, 2, 4, 5):
                    result = high_margin_metrics(y[test, idx], pred[:, idx])
                    rows.append({"dataset": dataset, "seed": seed, "fold": fold, "model": "M0StateDirect",
                                 "modality": MODALITIES[idx // 3],
                                 "target_type": "delta_absdiff" if idx % 3 == 1 else "delta_product",
                                 **result})
                del direct, direct_ckpt, h0_model, ht, hv, pos, gs, gd, deg

                for modality_i, modality in enumerate(MODALITIES):
                    cols = np.arange(modality_i * 3, modality_i * 3 + 3)
                    y_mean3, y_std3 = standardize_targets(y[:, cols], outer)
                    for representation in REPRESENTATIONS:
                        x = torch.as_tensor(state_reps[f"{representation}_{modality}"], dtype=torch.float32)
                        xmean, xstd = feature_stats(x, outer)
                        x = ((x - xmean) / xstd).contiguous().to(device)
                        state_path = OUT / dataset / f"seed_{seed}_fold_{fold}_{representation}_{modality}.pt"
                        state_ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
                        readout = StateReadout().to(device).eval()
                        readout.load_state_dict(state_ckpt["state_dict"], strict=True)
                        with torch.no_grad():
                            pred3 = readout(x)[torch.as_tensor(test, device=device)].cpu().numpy() * y_std3 + y_mean3
                        for output_i in (1, 2):
                            target_idx = modality_i * 3 + output_i
                            result = high_margin_metrics(y[test, target_idx], pred3[:, output_i])
                            rows.append({"dataset": dataset, "seed": seed, "fold": fold,
                                         "model": f"Frozen_{representation}",
                                         "modality": modality, "representation": representation,
                                         "target_type": "delta_absdiff" if output_i == 1 else "delta_product",
                                         **result})
                        del readout, state_ckpt, x
            del data, layout, h_t, h_v, state_reps
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output = pd.DataFrame(rows)
    if len(output) != 3 * 3 * 3 * (5 * 4 + 4 + 2 * 3 * 2):
        raise AssertionError(f"expected 972 high-margin rows, got {len(output)}")
    output.to_csv(RESEARCH / "high_margin_by_fold.csv", index=False)
    print(f"[recovered] high-margin by-fold rows={len(output)}", flush=True)
    return output


if __name__ == "__main__":
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    recover(dev)
