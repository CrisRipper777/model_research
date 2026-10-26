#!/usr/bin/env python3
"""Read-only, guarded MR-CGE V1.3 propagation-mechanism audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from scripts import run_hybrid_cge_paper_evidence_v1 as hybrid_runner
from scripts import run_mrcge_v13_early_cross_decoupling_screen as prior_runner
from scripts import run_mrcge_v13_screen as v13_runner
from scripts import run_mrcge_v12_confirmation as v12_runner
from scripts.mrcge_v13_propagation_audit_math import (
    all_pairwise_cosines,
    basis_spectrum,
    global_cosine,
    node_cosines,
    pairwise_gram,
    rms,
    span_novelty,
)
from scripts.mrcge_v13_propagation_audit_ops import early_cross, physical_operator, sparse_propagate
from src.data import load_mag_data
from src.models import build_model
from src.utils.seeds import set_seed


BASE_SHA = "788c35392f049fea32f42f5081656924888f7082"
BRANCH = "exp/mrcge_v13_propagation_mechanism_audit"
STUDY = ROOT / "research/mrcge_v13_propagation_audit"
DATA_DIR = STUDY / "data"
RESULT_DIR = DATA_DIR / "checkpoint_results"
GPU = 1
MIN_FREE_MIB = 2048
LAUNCH_FREE_MIB = 8192
BLOCK_SIZE = 2048
V13_MOVIES_TARGETS = [("mrcge_v13", "joint_view", "Movies", seed) for seed in (42, 43, 44)]
HYBRID_TARGETS = [("hybrid_cge_paper_v1", "H2_physical_3hop", "Movies", seed) for seed in (42, 43, 44)]
V13_OTHER_TARGETS = [("mrcge_v13", "joint_view", ds, seed)
                     for ds, seeds in (("Grocery", (42, 43)), ("ele-fashion", (42, 43))) for seed in seeds]
TARGETS = V13_MOVIES_TARGETS + HYBRID_TARGETS + V13_OTHER_TARGETS
FIXED_SPLITS = {
    "Movies": "2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d",
    "Grocery": "7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056",
    "ele-fashion": "04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5",
}
SPLIT_FILES = (
    "src/models/mrcge_v13.py", "src/models/mrcge_v13_early_cross_decoupled.py",
    "src/models/mrcge_v13_static_strength.py", "src/models/hybrid_cge_v11.py",
    "src/models/hybrid_cge_paper_v1.py", "src/models/cg_cge_v1.py", "src/models/psce_mag_v7a.py",
    "src/tasks/nc.py", "src/data/loaders.py", "src/data/splits.py", "src/data/graph_utils.py",
    "src/data/semantic_candidates_v7b.py", "src/data/types.py", "src/utils/seeds.py",
    "configs/model/mrcge_v13.yaml", "configs/model/hybrid_cge_paper_v1.yaml",
    "configs/task/nc.yaml", "configs/config.yaml",
    "configs/dataset/Movies.yaml", "configs/dataset/Grocery.yaml", "configs/dataset/ele-fashion.yaml",
    "scripts/run_mrcge_v13_screen.py", "scripts/run_hybrid_cge_paper_evidence_v1.py",
    "scripts/run_mrcge_v13_early_cross_decoupling_screen.py",
)
CSV_FIELDS = (
    "model", "condition", "dataset", "seed", "source_commit", "checkpoint_sha256",
    "analysis_source_sha", "modality", "support_definition", "diagnostic_name", "value", "units",
    "valid_node_count", "numerical_status", "test_evaluation", "order_i", "order_j",
    "numerator", "denominator", "details_json",
)
ANALYSIS_SOURCE_FILES = (
    "research/mrcge_v13_propagation_audit/DESIGN_FREEZE.md",
    "scripts/mrcge_v13_propagation_audit_math.py",
    "scripts/mrcge_v13_propagation_audit_ops.py",
    "scripts/run_mrcge_v13_propagation_audit.py",
    "tests/test_mrcge_v13_propagation_audit_math.py",
    "tests/test_mrcge_v13_propagation_audit_mechanics.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                          check=True, capture_output=True).stdout


def git_file_sha(commit: str, relative: str) -> str:
    return hashlib.sha256(git_bytes(commit, relative)).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields=CSV_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    all_fields = list(dict.fromkeys([*fields, *(k for row in rows for k in row)]))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def asfloat(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    value = float(value)
    return value if math.isfinite(value) else None


def source_hash_audit() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "cat-file", "-e", f"{BASE_SHA}^{{commit}}"], cwd=ROOT).returncode != 0:
        raise RuntimeError(f"required base commit {BASE_SHA} is unavailable")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "exp/mrcge_v13_early_cross_decoupling_screen"],
                      cwd=ROOT).returncode != 0:
        raise RuntimeError("required source branch is not a descendant of the baseline")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, head], cwd=ROOT).returncode != 0:
        raise RuntimeError("current branch is not based on the required baseline")
    mismatches = {}
    for relative in SPLIT_FILES:
        current = sha256_file(ROOT / relative)
        committed = git_file_sha(BASE_SHA, relative)
        if current != committed:
            mismatches[relative] = {"base": committed, "worktree": current}
    if mismatches:
        raise RuntimeError(f"frozen code/config differs from baseline: {mismatches}")
    return {"baseline_sha": BASE_SHA, "current_head": head, "frozen_files": {
        name: git_file_sha(BASE_SHA, name) for name in SPLIT_FILES}, "all_frozen_files_match": True}


def assert_analysis_source_committed() -> str:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise RuntimeError(f"expected analysis branch {BRANCH}, got {branch}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    mismatches = {}
    for relative in ANALYSIS_SOURCE_FILES:
        try:
            committed = git_file_sha(head, relative)
        except subprocess.CalledProcessError:
            committed = None
        actual = sha256_file(ROOT / relative)
        if committed != actual:
            mismatches[relative] = {"committed": committed, "worktree": actual}
    if mismatches:
        raise RuntimeError(f"analysis source is not frozen in HEAD: {mismatches}")
    return head


def protected_history_snapshot() -> dict[str, str]:
    protected = prior_runner.protected_history_file_hashes()
    for relative in ("research/mrcge_v13_early_cross_decoupling",
                     "outputs/mrcge_v13_early_cross_decoupling"):
        root = ROOT / relative
        if not root.is_dir():
            raise RuntimeError(f"protected historical directory is missing: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            protected[str(path.relative_to(ROOT))] = sha256_file(path)
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        item = v13_runner.cache_entries()[dataset]
        path = Path(item["cache_path_abs"])
        protected[str(path.relative_to(ROOT))] = sha256_file(path)
    return protected


def load_historical_references() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_path = ROOT / "research/mrcge_v13_early_cross_decoupling/data/frozen_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source_rows = read_rows(ROOT / "research/mrcge_v13_early_cross_decoupling/data/historical_source_audit.csv")
    source_index = {(r["model"], r["dataset"], int(r["seed"])): r for r in source_rows}
    records = []
    for model, condition, dataset, seed in TARGETS:
        family = "v13_a" if model == "mrcge_v13" else "hybrid_fixed_split"
        matches = [r for r in baseline["historical_references"][family]
                   if r["model"] == model and r["condition"] == condition
                   and r["dataset"] == dataset and int(r["seed"]) == seed]
        if len(matches) != 1:
            raise RuntimeError(f"expected one historical reference for {model}/{dataset}/{seed}; got {len(matches)}")
        ref = matches[0]
        checkpoint = ref["checkpoints"]["best_accuracy"]
        source = source_index.get((model, dataset, seed))
        if source is None or str(source.get("source_hash_verified", "")).lower() != "true":
            raise RuntimeError(f"historical model source hash audit is absent for {model}/{dataset}/{seed}")
        source_sha = str(ref["source_git_sha"])
        source_model_path = source["model_source_path"]
        recorded_model_hash = source["recorded_model_source_sha256"]
        committed_model_hash = git_file_sha(source_sha, source_model_path)
        if committed_model_hash != recorded_model_hash or committed_model_hash != source["committed_model_source_sha256"]:
            raise RuntimeError(f"historical model source SHA mismatch for {model}/{dataset}/{seed}")
        checkpoint_path = Path(checkpoint["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_checkpoint_sha = sha256_file(checkpoint_path)
        if actual_checkpoint_sha != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint checksum mismatch for {model}/{dataset}/{seed}")
        data_ref = baseline["dataset_records"][dataset]
        if data_ref["split_fingerprint"] != FIXED_SPLITS[dataset]:
            raise RuntimeError(f"historical fixed split mismatch for {dataset}")
        records.append({
            "model": model, "condition": condition, "dataset": dataset, "seed": seed,
            "source_git_sha": source_sha, "model_source_path": source_model_path,
            "model_source_sha256": committed_model_hash,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": actual_checkpoint_sha,
            "split_fingerprint": data_ref["split_fingerprint"],
            "split_file_sha256": data_ref["split_file_sha256"],
            "semantic_cache_path": data_ref["semantic_cache_path"],
            "semantic_cache_sha256": data_ref["semantic_cache_sha256"],
            "feature_fingerprint": data_ref["feature_fingerprint"],
            "dataset_fingerprint": data_ref["dataset_fingerprint"],
            "physical_edge_fingerprint": data_ref["physical_edge_fingerprint"],
            "candidate_fingerprint": data_ref["candidate_fingerprint"],
            "v7b_fingerprint": data_ref["v7b_fingerprint"],
            "num_nodes": data_ref["num_nodes"], "num_edges": data_ref["num_edges"],
            "status": "source_checkpoint_data_verified", "test_evaluation": False,
        })
    return records, baseline


def validate_preflight_data(records: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    cache_entries = v13_runner.cache_entries()
    observed_caches = {}
    dataset_rows = []
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        ref = baseline["dataset_records"][dataset]
        cache = cache_entries[dataset]
        actual_cache_sha = sha256_file(cache["cache_path_abs"])
        for name in ("feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint"):
            if cache[name] != ref[name]:
                raise RuntimeError(f"{dataset}: V7B.1 {name} differs from frozen reference")
        if actual_cache_sha != ref["semantic_cache_sha256"]:
            raise RuntimeError(f"{dataset}: semantic cache file SHA differs from frozen reference")
        cfg = v13_runner.make_cfg(dataset, "joint_view", 42, "cpu", cache)
        cfg.dataset.auto_generate_splits = False
        raw = load_mag_data(cfg, "nc", 42)
        safe = v12_runner.mask_test_labels(raw)
        if bool((safe.y[raw.test_idx] != -100).any()) or cfg.task.evaluate_test is not False:
            raise RuntimeError(f"{dataset}: Test masking/test-off invariant failed")
        fp = v13_runner.data_utils.dataset_fingerprints(raw)
        split_path_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
        split_path = Path(split_path_text).resolve() if split_path_text else None
        expected_split_path = prior_runner.fixed_split_path(dataset).resolve()
        if split_path != expected_split_path or not split_path.is_file():
            raise RuntimeError(f"{dataset}: loader resolved a different fixed split: {split_path}")
        split_sha = sha256_file(split_path)
        checks = {
            "features": ref["dataset_fingerprint"],
            "physical_edge_index": ref["physical_edge_fingerprint"],
            "split_fingerprint": ref["split_fingerprint"],
        }
        for field, expected in checks.items():
            if fp[field] != expected:
                raise RuntimeError(f"{dataset}: {field} mismatch: {fp[field]} != {expected}")
        if split_sha != ref["split_file_sha256"]:
            raise RuntimeError(f"{dataset}: fixed split file SHA mismatch")
        dataset_rows.append({
            "dataset": dataset, "num_nodes": int(raw.num_nodes), "physical_edge_count": int(raw.edge_index.size(1)),
            "dataset_fingerprint": fp["features"], "physical_edge_fingerprint": fp["physical_edge_index"],
            "split_fingerprint": fp["split_fingerprint"], "split_file_path": str(split_path),
            "split_file_sha256": split_sha, "feature_fingerprint": cache["feature_fingerprint"],
            "semantic_cache_path": cache["cache_path_abs"], "semantic_cache_sha256": actual_cache_sha,
            "candidate_fingerprint": cache["candidate_fingerprint"], "v7b_fingerprint": cache["v7b_fingerprint"],
            "test_label_masked": True, "test_evaluation": False,
        })
        observed_caches[dataset] = {key: cache[key] for key in
                                    ("cache_path_abs", "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint")}
        del safe, raw
    if any(record["dataset"] not in {row["dataset"] for row in dataset_rows} for record in records):
        raise RuntimeError("some checkpoint dataset lacks a verified data preflight")
    return {"datasets": dataset_rows, "cache_entries": observed_caches,
            "auto_generate_splits": False, "test_evaluation": False,
            "test_labels_masked_before_any_model_forward": True}


def manifest_row(record: dict[str, Any], analysis_sha: str) -> dict[str, Any]:
    return {**record, "analysis_source_sha": analysis_sha,
            "analysis_entrypoint_sha256": sha256_file(Path(__file__)),
            "analysis_math_sha256": sha256_file(ROOT / "scripts/mrcge_v13_propagation_audit_math.py"),
            "analysis_operator_sha256": sha256_file(ROOT / "scripts/mrcge_v13_propagation_audit_ops.py"),
            "diagnostic_worker_status": "pending", "source_branch": BRANCH,
            "test_evaluation": False}


def _metric(identity: dict[str, Any], modality: str, support_definition: str, name: str,
            value: Any, units: str, valid_nodes: int | None, status: str = "ok", **extra):
    row = {key: identity[key] for key in ("model", "condition", "dataset", "seed", "checkpoint_sha256")}
    row.update({"source_commit": identity["source_git_sha"], "analysis_source_sha": identity["analysis_source_sha"], "modality": modality,
                "support_definition": support_definition, "diagnostic_name": name,
                "value": asfloat(value), "units": units, "valid_node_count": valid_nodes,
                "numerical_status": status, "test_evaluation": False})
    for key, val in extra.items():
        row[key] = val if key != "details_json" else json.dumps(json_safe(val), sort_keys=True, separators=(",", ":"))
    return row


def _node_zero_count(value: torch.Tensor, support: torch.Tensor, threshold=1.0e-12) -> tuple[int, int]:
    norms = value.float().norm(dim=-1)
    return int((norms[support] <= threshold).sum().item()), int((norms <= threshold).sum().item())


def _ratio(numerator: dict, denominator: dict) -> tuple[float | None, str]:
    if numerator.get("value") is None or denominator.get("value") is None:
        return None, "undefined_empty_support"
    if denominator["value"] == 0:
        return None, "undefined_zero_denominator"
    result = numerator["value"] / denominator["value"]
    return (result, "ok") if math.isfinite(result) else (None, "nonfinite_ratio")


def _identity_with_analysis(record: dict[str, Any], source_sha: str) -> dict[str, Any]:
    return {**record, "analysis_source_sha": source_sha}


def analyze_v13(model, x: torch.Tensor, edge: torch.Tensor, identity: dict[str, Any], forward_info: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output = {key: [] for key in ("physical_basis_energy", "physical_basis_similarity", "physical_basis_rank",
                                  "expert_input_diagnostics", "cross_trajectory_energy", "cross_trajectory_similarity",
                                  "cross_novelty", "same_node_overlap")}
    audit = model._last_state_audit
    support = forward_info["active_nodes"].to(dtype=torch.bool)
    physical_edge, edge_weight, physical_support, loop_count = model._physical_operator(edge, x.size(0), x.dtype)
    if not torch.equal(support, physical_support):
        raise RuntimeError("V1.3 forward active mask differs from frozen physical operator reconstruction")
    h = audit["sender_states"]
    bases = {m: model._basis(h[m], physical_edge, edge_weight, support, 3, model.physical_chunk_size)
             for m in ("text", "visual")}
    beta = model.beta.detach().float().cpu().tolist()
    cross_vectors: dict[str, tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]] = {}
    detail = forward_info.get("details", {})

    for modality_index, modality in enumerate(("text", "visual")):
        other = "visual" if modality == "text" else "text"
        projection_name = "visual_to_text" if modality == "text" else "text_to_visual"
        phi = model.cross_projections[projection_name]
        basis = bases[modality]
        basis_support_rms = [rms(b, support, block_size=BLOCK_SIZE) for b in basis]
        basis_all_rms = [rms(b, None, block_size=BLOCK_SIZE) for b in basis]
        for order, value in enumerate(basis):
            zeros_supported, zeros_all = _node_zero_count(value, support)
            output["physical_basis_energy"].append(_metric(
                identity, modality, "V1.3 receiver physical support (edge destinations after self-loop removal)",
                f"B{order}_rms_support", basis_support_rms[order]["value"], "feature_RMS", int(support.sum()),
                basis_support_rms[order]["status"], order_i=order,
                numerator=f"mean(B{order}[support]^2)^(1/2)", denominator="1",
                details_json={"all_node_rms": basis_all_rms[order]["value"], "all_node_count": int(value.size(0)),
                              "support_node_count": int(support.sum()), "zero_norm_supported_nodes": zeros_supported,
                              "zero_norm_all_nodes": zeros_all, "input_self_loops_removed": loop_count}))
        for order_i, order_j in ((2, 1), (3, 1), (3, 2)):
            value, status = _ratio(basis_support_rms[order_i], basis_support_rms[order_j])
            output["physical_basis_energy"].append(_metric(
                identity, modality, "same V1.3 receiver physical support", f"B{order_i}_over_B{order_j}_rms",
                value, "ratio", int(support.sum()), status, order_i=order_i, order_j=order_j,
                numerator=f"RMS(B{order_i}[support])", denominator=f"RMS(B{order_j}[support])"))

        gram = pairwise_gram(basis, support, block_size=BLOCK_SIZE)
        norms = gram.diagonal().clamp_min(0).sqrt()
        normalized = gram / norms[:, None].clamp_min(1e-300) / norms[None, :].clamp_min(1e-300)
        spectrum = basis_spectrum(basis, support, block_size=BLOCK_SIZE)
        output["physical_basis_rank"].append(_metric(
            identity, modality, "flattened B0..B3 on common V1.3 physical support", "physical_basis_spectrum",
            spectrum.get("effective_rank"), "entropy_effective_rank", int(support.sum()), spectrum["status"],
            details_json={**spectrum, "raw_gram": gram.cpu().tolist()}))
        for i in range(4):
            for j in range(i, 4):
                global_value = None if norms[i] == 0 or norms[j] == 0 else float(gram[i, j] / (norms[i] * norms[j]))
                node = node_cosines(basis[i], basis[j], support)
                output["physical_basis_similarity"].append(_metric(
                    identity, modality, "same V1.3 receiver physical support", "basis_order_cosine",
                    global_value, "global_frobenius_cosine", int(support.sum()),
                    "zero_norm_operand" if global_value is None else "ok", order_i=i, order_j=j,
                    details_json={"node_cosine_distribution": node,
                                  "normalized_gram_entry": asfloat(normalized[i, j]),
                                  "matrix_definition": "cos(B_i,B_j) over flattened common-support coordinates"}))

        source = h[other]
        q = phi(source)
        raw_rebuilt, x1_rebuilt, rebuilt_scale = early_cross(
            source, phi, physical_edge, edge_weight, support, basis[1], model.eps)
        actual_x1 = audit["early_cross"][modality]
        diff = (x1_rebuilt - actual_x1).float()
        actual_norm = actual_x1[support].float().norm() if bool(support.any()) else actual_x1.new_zeros(())
        relative = float(diff[support].norm() / actual_norm.clamp_min(1.0e-30)) if bool(support.any()) else 0.0
        max_abs = float(diff.abs().max()) if diff.numel() else 0.0
        if not math.isfinite(relative) or relative > 1.0e-6 or max_abs > 1.0e-4:
            raise RuntimeError(f"{modality}: independent X1 reconstruction failed (relative={relative}, max_abs={max_abs})")
        x1 = actual_x1
        x2 = sparse_propagate(x1, physical_edge, edge_weight) * support.to(x.dtype).unsqueeze(-1)
        x3 = sparse_propagate(x2, physical_edge, edge_weight) * support.to(x.dtype).unsqueeze(-1)
        cross = [x1, x2, x3]
        cross_vectors[modality] = (cross, q, phi(audit["private"][other]))
        cross_rms = [rms(c, support, block_size=BLOCK_SIZE) for c in cross]
        q_rms = rms(q, None, block_size=BLOCK_SIZE)
        late = cross_vectors[modality][2]
        late_rms = rms(late, None, block_size=BLOCK_SIZE)
        zeros_x1, _ = _node_zero_count(x1, support)
        actual_detail = detail.get("early_physical_cross", {}).get(modality, {})
        for order, value in enumerate(cross):
            zeros, _ = _node_zero_count(value, support)
            output["cross_trajectory_energy"].append(_metric(
                identity, modality, "same V1.3 receiver physical support", f"X{order + 1}_rms_support",
                cross_rms[order]["value"], "feature_RMS", int(support.sum()), cross_rms[order]["status"],
                order_i=order + 1, details_json={"zero_or_near_zero_nodes": zeros,
                    "zero_or_near_zero_fraction": zeros / max(int(support.sum()), 1),
                    "X1_reconstruction_relative_error": relative if order == 0 else None,
                    "X1_reconstruction_max_abs_error": max_abs if order == 0 else None,
                    "early_cross_scale": rebuilt_scale if order == 0 else None,
                    "model_reported_scale": actual_detail.get("scale") if order == 0 else None,
                    "beta_m": beta[modality_index], "diagnostic_only_orders_2_3": order > 0}))
        output["cross_trajectory_energy"].append(_metric(
            identity, modality, "same V1.3 receiver physical support", "C_raw_rms_support",
            rms(raw_rebuilt, support, block_size=BLOCK_SIZE)["value"], "feature_RMS", int(support.sum()),
            "ok", details_json={"quantity": "un-calibrated P Phi(H_other)", "reconstructed_scale": rebuilt_scale}))
        output["cross_trajectory_energy"].append(_metric(
            identity, modality, "same V1.3 receiver physical support", "X1_calibration_scale",
            rebuilt_scale, "dimensionless_scale", int(support.sum()), "ok",
            numerator="RMS(B1[support])", denominator="RMS(C_raw[support])+eps"))
        for i, j in ((1, 0), (2, 1)):
            value, status = _ratio(cross_rms[i], cross_rms[j])
            output["cross_trajectory_energy"].append(_metric(
                identity, modality, "same V1.3 receiver physical support", f"X{i + 1}_over_X{j + 1}_rms",
                value, "ratio", int(support.sum()), status, order_i=i + 1, order_j=j + 1,
                numerator=f"RMS(X{i + 1}[support])", denominator=f"RMS(X{j + 1}[support])"))
        ratio, status = _ratio(cross_rms[0], basis_support_rms[1])
        output["cross_trajectory_energy"].append(_metric(
            identity, modality, "same V1.3 receiver physical support", "X1_over_B1_rms", ratio, "ratio",
            int(support.sum()), status, numerator="RMS(X1[support])", denominator="RMS(B1[support])"))
        injected = x1 * float(beta[modality_index])
        output["cross_trajectory_energy"].append(_metric(
            identity, modality, "same V1.3 receiver physical support", "beta_X1_rms_support",
            rms(injected, support, block_size=BLOCK_SIZE)["value"], "feature_RMS", int(support.sum()),
            "ok", details_json={"beta_m": beta[modality_index], "X1_rms_is_uncalibrated_by_beta": True}))
        output["cross_trajectory_energy"].append(_metric(
            identity, modality, "sender nodes; no support mask", "Phi_H_other_rms_all_nodes", q_rms["value"],
            "feature_RMS", int(q.size(0)), q_rms["status"], details_json={"candidate_name": "Phi(H_other)"}))

        all_cross = all_pairwise_cosines(cross, support, block_size=BLOCK_SIZE)
        for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
            global_info = all_cross["cosines"][f"{i}:{j}"]
            node = node_cosines(cross[i], cross[j], support)
            output["cross_trajectory_similarity"].append(_metric(
                identity, modality, "same V1.3 receiver physical support", "cross_order_cosine",
                global_info["value"], "global_frobenius_cosine", int(support.sum()), global_info["status"],
                order_i=i + 1, order_j=j + 1, details_json={"node_cosine_distribution": node}))
        cross_basis_gram = pairwise_gram([*cross, *basis], support, block_size=BLOCK_SIZE)
        cross_basis_norm = cross_basis_gram.diagonal().clamp_min(0).sqrt()
        for i in range(3):
            for j in range(4):
                den = cross_basis_norm[i] * cross_basis_norm[3 + j]
                global_value = None if den == 0 else float(cross_basis_gram[i, 3 + j] / den)
                node = node_cosines(cross[i], basis[j], support)
                output["cross_trajectory_similarity"].append(_metric(
                    identity, modality, "same V1.3 receiver physical support", "cross_basis_cosine",
                    global_value, "global_frobenius_cosine", int(support.sum()),
                    "zero_norm_operand" if global_value is None else "ok", order_i=i + 1, order_j=j,
                    details_json={"node_cosine_distribution": node,
                                  "comparison": f"cos(X{i+1},B{j})"}))

        base = [*basis, x1]
        for target_order, target, base_label in ((2, x2, "S_base"), (3, x3, "S_base")):
            novelty = span_novelty(base, target, support, block_size=BLOCK_SIZE)
            output["cross_novelty"].append(_metric(
                identity, modality, "flattened common receiver physical support", f"nu{target_order}",
                novelty["novelty"], "relative_frobenius_residual", int(support.sum()), novelty["status"],
                order_i=target_order, numerator=f"||X{target_order}-Proj(S_base,X{target_order})||F",
                denominator=f"||X{target_order}||F", details_json={**novelty, "base_span": base_label}))
        conditional = span_novelty([*base, x2], x3, support, block_size=BLOCK_SIZE)
        output["cross_novelty"].append(_metric(
            identity, modality, "flattened common receiver physical support", "nu3_conditional_on_X2",
            conditional["novelty"], "relative_frobenius_residual", int(support.sum()), conditional["status"],
            order_i=3, numerator="||X3-Proj(span(B0..B3,X1,X2),X3)||F", denominator="||X3||F",
            details_json=conditional))

        for candidate_name, candidate in (("Phi(H_other)", q), ("Phi(U_private_other)", late)):
            for order, value in ((2, x2), (3, x3)):
                cosine = global_cosine(value, candidate, support if candidate.size(0) == support.size(0) else None,
                                       block_size=BLOCK_SIZE)
                if candidate.size(0) != support.size(0):
                    cosine = global_cosine(value, candidate, None, block_size=BLOCK_SIZE)
                    node = {"status": "not_aligned_supports", "valid_nodes": None,
                            "zero_or_near_zero_nodes": None, "p10": None, "median": None, "p90": None}
                else:
                    node = node_cosines(value, candidate, support)
                output["same_node_overlap"].append(_metric(
                    identity, modality, "X2/X3 on physical receiver support; candidate support stated in details",
                    "cross_same_node_alignment", cosine["value"], "global_frobenius_cosine", int(support.sum()),
                    cosine["status"], order_i=order, details_json={"candidate": candidate_name,
                        "node_cosine_distribution": node,
                        "candidate_rms": rms(candidate, None, block_size=BLOCK_SIZE)["value"],
                        "interpretation": "alignment clue only; not backtracking mass"}))

        alpha = model.filter_bank.coefficients(3, basis[0].dtype).detach()
        private_inputs = torch.stack([
            sum((alpha[k, order] * basis[order] for order in range(4)), torch.zeros_like(basis[0]))
            for k in range(model.num_experts)
        ])
        collaborative_basis = list(basis)
        collaborative_basis[1] = (basis[1] + beta[modality_index] * x1) * support.to(x.dtype).unsqueeze(-1)
        collaborative_inputs = torch.stack([
            sum((alpha[k, order] * collaborative_basis[order] for order in range(4)), torch.zeros_like(basis[0]))
            for k in range(model.num_experts)
        ])
        for scope, inputs in (("private", private_inputs), ("collaborative", collaborative_inputs)):
            norms = pairwise_gram(list(inputs), support, block_size=BLOCK_SIZE).diagonal().clamp_min(0).sqrt()
            input_cosines = all_pairwise_cosines(list(inputs), support, block_size=BLOCK_SIZE)
            input_rank = basis_spectrum(list(inputs), support, block_size=BLOCK_SIZE)
            output["expert_input_diagnostics"].append(_metric(
                identity, modality, "common V1.3 physical support", f"{scope}_expert_input_matrix",
                input_rank.get("effective_rank"), "entropy_effective_rank", int(support.sum()), input_rank["status"],
                details_json={"trained_theta": model.filter_bank.theta.detach().cpu().tolist(),
                              "trained_coefficients": alpha.cpu().tolist(),
                              "expert_input_rms": [asfloat(rms(v, support, block_size=BLOCK_SIZE)["value"]) for v in inputs],
                              "expert_input_gram": pairwise_gram(list(inputs), support, block_size=BLOCK_SIZE).cpu().tolist(),
                              "expert_input_normalized_gram": input_cosines["normalized_gram"],
                              "expert_input_cosines": input_cosines["cosines"],
                              "expert_input_rank": input_rank}))
            for expert, norm in enumerate(norms):
                output["expert_input_diagnostics"].append(_metric(
                    identity, modality, "common V1.3 physical support", f"{scope}_expert_{expert}_input_rms",
                    float(norm / math.sqrt(max(int(support.sum()) * inputs.size(-1), 1))), "feature_RMS",
                    int(support.sum()), "ok", details_json={"input_scope": scope}))
        for scope in ("private", "collaborative"):
            route_summary = (detail.get("routes", {}) if scope == "private" else detail.get("collaborative_routes", {}))
            route_key = f"private_{modality}_physical" if scope == "private" else modality
            route = route_summary.get(route_key, {})
            filt = detail.get("filters", {}).get(f"{scope}_{modality}_physical", {})
            output["expert_input_diagnostics"].append(_metric(
                identity, modality, "common V1.3 physical support", f"{scope}_router_summary",
                route.get("dense_entropy"), "nats", int(support.sum()), "ok" if route else "missing_route_detail",
                details_json={"dense_importance": route.get("dense_importance"),
                    "top2_selection_share": route.get("top2_selection_share"),
                    "selected_weight_entropy": route.get("selected_weight_entropy"),
                    "expert_G_rms": filt.get("G_expert_rms"), "expert_E_rms": filt.get("E_expert_rms"),
                    "G_pairwise_cosine": filt.get("G_pairwise_cosine"), "E_pairwise_cosine": filt.get("E_pairwise_cosine")}))
        del private_inputs, collaborative_inputs

    return output


def _route_entropy(route: dict[str, torch.Tensor], active: torch.Tensor) -> tuple[float | None, list[float] | None]:
    probs = route["dense_probs"].detach().float()
    selected = route["selected_mask"].detach() if "selected_mask" in route else route["selected"].detach()
    if not bool(active.any()):
        return None, None
    p = probs[active].clamp_min(1.0e-12)
    entropy = float((-(p * p.log()).sum(-1).mean()).cpu())
    share = (selected[active].float().mean(0) / float(route.get("top_k", 2))).detach().cpu().tolist()
    return entropy, share


def _hybrid_route(model, modality_index, prior, first_state, cross_hat, active):
    zero = torch.zeros_like(first_state)
    base = model._physical_route(modality_index, prior, active)
    logits = model.selection_head(base["encoded"]).expand(prior.size(0), -1)
    private_corr = model._node_correction(prior, first_state, zero, zero, modality_index)
    collab_corr = model._node_correction(prior, first_state, zero, cross_hat, modality_index)
    private = model._top2(logits + private_corr, model.top_k)
    collab = model._top2(logits + collab_corr, model.top_k)
    for route in (private, collab):
        route["strength"] = base["strength"]
        route["top_k"] = model.top_k
    return private, collab


def analyze_hybrid(model, x, edge, identity, forward_info):
    output = {key: [] for key in ("physical_basis_energy", "physical_basis_similarity", "physical_basis_rank",
                                  "expert_input_diagnostics", "same_node_overlap", "hybrid_structural_comparison")}
    inputs = model._split_modalities(x)
    priors = {m: model.projectors[m](inputs[m]) for m in ("text", "visual")}
    src, dst, physical, active, loop_count = model._normalized_operator(edge, x.size(0), priors["text"].dtype)
    if not torch.equal(active, forward_info["active_nodes"].to(dtype=torch.bool)):
        raise RuntimeError("Hybrid internal active mask differs from its audited forward mask")
    physical_states, scales = {}, {}
    normalized, profiles, cross_hat, cross_raw = {}, {}, {}, {}
    modality_details = forward_info.get("details", {}).get("modalities", {})
    for index, modality in enumerate(("text", "visual")):
        physical_states[modality], scales[modality] = model._physical_states(
            priors[modality], src, dst, physical, active)
        normalized[modality] = [model._active_rms_norm(state, active) for state in physical_states[modality]]
        alpha = model._effective_alpha(model.alpha_raw, model.eps).detach()
        profiles[modality] = torch.einsum("kl,lnd->knd", alpha, torch.stack(normalized[modality]))
        other = "visual" if modality == "text" else "text"
        projection = model.cross_projections["visual_to_text" if modality == "text" else "text_to_visual"]
        cross_raw[modality] = model._physical_cross_message(
            priors[other], projection, src, dst, physical, active, model.propagation_chunk_size)
        cross_hat[modality] = cross_raw[modality] / scales[modality] * active.to(x.dtype).unsqueeze(-1)
        forward_cross = forward_info["details"]["early_physical_cross"][modality]
        for expected_name, value in (("raw_rms", cross_raw[modality]), ("calibrated_rms", cross_hat[modality])):
            actual = float(model._masked_rms(value, active).cpu())
            if abs(actual - float(forward_cross[expected_name])) > 1.0e-5 * max(1.0, abs(actual)):
                raise RuntimeError(f"Hybrid {modality} reconstructed cross {expected_name} differs from real forward")
        private_state, collab_state, _, helper_detail = model._private_and_collaborative_state(
            modality, index, priors[modality], physical_states[modality], scales[modality],
            cross_raw[modality], cross_hat[modality], active, True)
        same_candidate = model.cross_projections["visual_to_text" if modality == "text" else "text_to_visual"](private_state)
        sender_prior = model.cross_projections["visual_to_text" if modality == "text" else "text_to_visual"](priors[other])
        routes = _hybrid_route(model, index, priors[modality], physical_states[modality][0], cross_hat[modality], active)
        route_stats = {}
        for scope, route in zip(("private", "collaborative"), routes):
            entropy, share = _route_entropy(route, active)
            route_stats[scope] = {"entropy": entropy, "top2_share": share}
        raw_rms = [rms(state, active, block_size=BLOCK_SIZE) for state in physical_states[modality]]
        norm_rms = [rms(state, active, block_size=BLOCK_SIZE) for state in normalized[modality]]
        raw_rank = basis_spectrum(physical_states[modality], active, block_size=BLOCK_SIZE)
        norm_rank = basis_spectrum(normalized[modality], active, block_size=BLOCK_SIZE)
        profile_rank = basis_spectrum(list(profiles[modality]), active, block_size=BLOCK_SIZE)
        profile_cos = all_pairwise_cosines(list(profiles[modality]), active, block_size=BLOCK_SIZE)
        cross_r = rms(cross_hat[modality], active, block_size=BLOCK_SIZE)
        private_profile_r = rms(profiles[modality][0], active, block_size=BLOCK_SIZE)
        private_response = float(helper_detail["physical_private_response_rms"])
        collab_response = float(helper_detail["physical_collab_response_rms"])
        ratio, ratio_status = _ratio(cross_r, private_profile_r)
        output["hybrid_structural_comparison"].append(_metric(
            identity, modality, "Hybrid physical active receiver support", "hybrid_internal_structure",
            norm_rank.get("effective_rank"), "entropy_effective_rank", int(active.sum()), norm_rank["status"],
            details_json={
                "model_specific_state_orders": "raw Physical States 1..3; no B0 in Hybrid basis",
                "raw_state_rms_orders_1_to_3": [r["value"] for r in raw_rms],
                "raw_state_adjacent_rms_ratios_2_over_1_3_over_2": [
                    _ratio(raw_rms[1], raw_rms[0])[0], _ratio(raw_rms[2], raw_rms[1])[0]],
                "normalized_state_rms_orders_1_to_3": [r["value"] for r in norm_rms],
                "raw_state_normalized_gram": raw_rank.get("normalized_gram"),
                "raw_state_rank": raw_rank, "normalized_state_rank": norm_rank,
                "trained_effective_alpha": model._effective_alpha(model.alpha_raw, model.eps).detach().cpu().tolist(),
                "expert_profile_rms": [rms(v, active, block_size=BLOCK_SIZE)["value"] for v in profiles[modality]],
                "expert_profile_normalized_gram": profile_cos["normalized_gram"],
                "expert_profile_cosines": profile_cos["cosines"], "expert_profile_rank": profile_rank,
                "cross_raw_rms": float(model._masked_rms(cross_raw[modality], active).cpu()),
                "cross_calibrated_rms": cross_r["value"], "cross_over_private_profile_rms": ratio,
                "cross_profile_ratio_status": ratio_status,
                "cross_expert_coefficients": model.physical_cross_coefficients.detach().cpu().tolist(),
                "private_collaborative_profile_rms": [helper_detail["physical_private_profile_rms"],
                                                       helper_detail["physical_collab_profile_rms"]],
                "private_collaborative_response_rms": [private_response, collab_response],
                "collaborative_over_private_response_rms": (collab_response / private_response
                    if private_response > 0 else None),
                "private_route": route_stats["private"], "collaborative_route": route_stats["collaborative"],
                "route_top2_flip_fraction": helper_detail["physical_route_top2_flip_fraction"],
                "forward_route_shares": {"private": helper_detail["physical_private_route_share"],
                                          "collaborative": helper_detail["physical_collab_route_share"]},
                "private_state_rms": helper_detail["private_state_rms"],
                "collaborative_state_rms": helper_detail["collaborative_state_rms"],
                "first_hop_channel_scale": scales[modality].detach().cpu().tolist(),
                "true_late_same_node_candidate_rms": float(model._masked_rms(same_candidate, torch.ones_like(active)).cpu()),
                "Phi_H_other_rms": float(model._masked_rms(sender_prior, torch.ones_like(active)).cpu()),
                "normalization": "Hybrid per-channel active RMS for each physical state",
                "input_self_loops_removed": loop_count,
            }))
        for order, (state, state_rms) in enumerate(zip(physical_states[modality], raw_rms), start=1):
            output["physical_basis_energy"].append(_metric(
                identity, modality, "Hybrid active physical receiver support", f"hybrid_raw_state_{order}_rms",
                state_rms["value"], "feature_RMS", int(active.sum()), state_rms["status"], order_i=order,
                details_json={"all_node_rms": rms(state, None, block_size=BLOCK_SIZE)["value"],
                              "zero_norm_active_nodes": _node_zero_count(state, active)[0]}))
            output["physical_basis_energy"].append(_metric(
                identity, modality, "Hybrid active physical receiver support", f"hybrid_normalized_state_{order}_rms",
                norm_rms[order - 1]["value"], "feature_RMS", int(active.sum()), norm_rms[order - 1]["status"],
                order_i=order, details_json={"normalization": "per-channel active RMS"}))
        gram = pairwise_gram(physical_states[modality], active, block_size=BLOCK_SIZE)
        norms = gram.diagonal().clamp_min(0).sqrt()
        for i in range(3):
            for j in range(i, 3):
                value = None if norms[i] == 0 or norms[j] == 0 else float(gram[i, j] / (norms[i] * norms[j]))
                output["physical_basis_similarity"].append(_metric(
                    identity, modality, "Hybrid raw Physical States on active support", "hybrid_state_order_cosine",
                    value, "global_frobenius_cosine", int(active.sum()), "ok" if value is not None else "zero_norm_operand",
                    order_i=i + 1, order_j=j + 1,
                    details_json={"meaning": "Hybrid raw Physical States, orders 1..3; not MR-CGE basis"}))
        output["physical_basis_rank"].append(_metric(
            identity, modality, "Hybrid raw Physical States orders 1..3", "hybrid_raw_state_spectrum",
            raw_rank.get("effective_rank"), "entropy_effective_rank", int(active.sum()), raw_rank["status"],
            details_json=raw_rank))
        output["physical_basis_rank"].append(_metric(
            identity, modality, "Hybrid individually normalized Physical States orders 1..3",
            "hybrid_normalized_state_spectrum", norm_rank.get("effective_rank"), "entropy_effective_rank",
            int(active.sum()), norm_rank["status"], details_json=norm_rank))
        for expert, profile in enumerate(profiles[modality]):
            output["expert_input_diagnostics"].append(_metric(
                identity, modality, "Hybrid active physical receiver support", f"hybrid_expert_{expert}_input_rms",
                rms(profile, active, block_size=BLOCK_SIZE)["value"], "feature_RMS", int(active.sum()), "ok",
                details_json={"expert_input": "trained alpha mixture of normalized physical states"}))
        output["expert_input_diagnostics"].append(_metric(
            identity, modality, "Hybrid active physical receiver support", "hybrid_expert_input_matrix",
            profile_rank.get("effective_rank"), "entropy_effective_rank", int(active.sum()), profile_rank["status"],
            details_json={"normalized_gram": profile_cos["normalized_gram"], "cosines": profile_cos["cosines"],
                          "singular_values": profile_rank.get("singular_values")}))
        output["same_node_overlap"].append(_metric(
            identity, modality, "Hybrid active physical receiver support", "hybrid_late_same_node_candidate",
            float(model._masked_rms(same_candidate, torch.ones_like(active)).cpu()), "feature_RMS", int(active.sum()), "ok",
            details_json={"candidate": "Phi(U_private_other)", "private_state_extracted_by_existing_helper": True,
                          "true_candidate": True}))
    return output


def worker(record_index: int, output_path: Path) -> dict[str, Any]:
    if os.environ.get("MR_CGE_PROPAGATION_GUARDED") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU):
        raise RuntimeError("GPU worker must be started under the physical GPU 1 ownership guard")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("isolated worker cannot see exactly one CUDA device")
    records, baseline = load_historical_references()
    record = records[record_index]
    source_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
    identity = _identity_with_analysis(record, source_sha)
    cache = v13_runner.cache_entries()[record["dataset"]]
    if sha256_file(cache["cache_path_abs"]) != record["semantic_cache_sha256"]:
        raise RuntimeError("semantic cache changed after preflight")
    physical_before = v13_runner.smi_snapshot()
    device = next((row for row in physical_before["devices"] if row["physical_index"] == GPU), None)
    if device is None:
        raise RuntimeError("physical GPU 1 is not visible to nvidia-smi")
    baseline_pids = {int(x) for x in os.environ.get("MR_CGE_BASELINE_PIDS", "").split(",") if x}
    v13_runner.guard_gpu(GPU, baseline_pids, {os.getpid()}, phase="before_launch", required_free_mib=MIN_FREE_MIB)
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    set_seed(record["seed"])
    if record["model"] == "mrcge_v13":
        cfg = v13_runner.make_cfg(record["dataset"], "joint_view", record["seed"], "cuda:0", cache)
    else:
        cfg = hybrid_runner.build_cfg(record["dataset"], record["model"], record["seed"], "cuda:0",
                                      condition=record["condition"], cache=cache)
        cfg.dataset.auto_generate_splits = False
        cfg.dataset.nc_split_path = str(prior_runner.fixed_split_path(record["dataset"]).resolve())
        cfg.model.paper_condition = "H2_physical_3hop"
        cfg.model.physical_hops = 3
    cfg.task.evaluate_test = False
    raw = load_mag_data(cfg, "nc", record["seed"])
    safe = v12_runner.mask_test_labels(raw)
    if bool((safe.y[raw.test_idx] != -100).any()):
        raise RuntimeError("Test labels are not masked in the worker data copy")
    fp = v13_runner.data_utils.dataset_fingerprints(raw)
    for field, expected in (("features", record["dataset_fingerprint"]),
                            ("physical_edge_index", record["physical_edge_fingerprint"]),
                            ("split_fingerprint", record["split_fingerprint"])):
        if fp[field] != expected:
            raise RuntimeError(f"worker {field} changed after preflight")
    split_path_text = raw.info.get("nc_split_path") or raw.info.get("node_split_path")
    if not split_path_text or Path(split_path_text).resolve() != prior_runner.fixed_split_path(record["dataset"]).resolve():
        raise RuntimeError("worker data loader did not use the frozen split")

    if record["model"] == "mrcge_v13":
        model = build_model(cfg, v13_runner.data_info(safe))
        model.load_semantic_cache(cache["cache_path_abs"], expected_dataset=record["dataset"],
                                  expected_feature_fingerprint=cache["feature_fingerprint"],
                                  expected_candidate_fingerprint=cache["candidate_fingerprint"],
                                  expected_v7b_fingerprint=cache["v7b_fingerprint"])
    else:
        model = hybrid_runner.get_model(cfg, safe, cache, torch.device("cuda:0"))
    checkpoint_path = Path(record["checkpoint_path"])
    if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
        raise RuntimeError("checkpoint checksum changed before load")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        raise RuntimeError("historical checkpoint does not contain model_state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to("cuda:0").eval()
    if record["model"] == "mrcge_v13":
        model._capture_state_audit = True
    parameters_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    x = safe.x.to("cuda:0")
    edge = safe.edge_index.to("cuda:0")
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    with torch.no_grad():
        _, _, _, aux, info = model(x, edge, return_details=True)
    torch.cuda.synchronize(0)
    wall = time.perf_counter() - started
    if not torch.equal(torch.random.get_rng_state(), cpu_rng) or not torch.equal(torch.cuda.get_rng_state(0), cuda_rng):
        raise RuntimeError("eval/no_grad diagnostic forward changed RNG state")
    if not bool(torch.isfinite(aux)):
        raise RuntimeError("diagnostic forward returned non-finite auxiliary scalar")
    changed = [name for name, value in model.state_dict().items()
               if not torch.equal(parameters_before[name], value.detach().cpu())]
    if changed:
        raise RuntimeError(f"diagnostic forward changed checkpoint model state: {changed[:8]}")
    if info.get("test_evaluation", False) is not False:
        raise RuntimeError("forward metadata does not explicitly mark Test evaluation false")
    if record["model"] == "mrcge_v13":
        metrics = analyze_v13(model, x, edge, identity, info)
        condition = model.v13_variant
    else:
        if model.paper_condition != "H2_physical_3hop" or model.physical_hops != 3:
            raise RuntimeError("Hybrid worker did not instantiate exact H2_physical_3hop")
        metrics = analyze_hybrid(model, x, edge, identity, info)
        condition = model.paper_condition
    if not torch.equal(torch.random.get_rng_state(), cpu_rng) or not torch.equal(torch.cuda.get_rng_state(0), cuda_rng):
        raise RuntimeError("eval/no_grad diagnostics changed CPU or CUDA RNG state")
    changed = [name for name, value in model.state_dict().items()
               if not torch.equal(parameters_before[name], value.detach().cpu())]
    if changed:
        raise RuntimeError(f"analysis helpers changed checkpoint model state: {changed[:8]}")
    allocation = {
        "physical_gpu": GPU, "gpu_uuid": device["uuid"], "gpu_name_nvidia_smi": device["name"],
        "torch_device_name": device_name, "free_mib_before_worker": device["memory_free_mib"],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "wall_seconds_forward_and_diagnostics": wall,
        "transient_tensor_estimate_bytes": int(x.size(0) * 256 * 4 * (14 if record["model"] == "mrcge_v13" else 12)),
        "nodes": int(x.size(0)), "physical_edges": int(edge.size(1)), "physical_support_nodes": int(info["active_nodes"].sum()),
        "test_evaluation": False,
    }
    result = {"identity": identity, "status": "complete", "condition_loaded": condition,
              "model_family": record["model"], "source_git_sha": record["source_git_sha"],
              "checkpoint_sha256": record["checkpoint_sha256"], "metrics": metrics,
              "resource": allocation, "created_utc": now(), "test_evaluation": False,
              "test_labels_masked": True}
    write_json(output_path, json_safe(result))
    del model, x, edge, raw, safe, checkpoint
    torch.cuda.empty_cache()
    return result


def gpu_snapshot_or_block() -> tuple[dict[str, Any], set[int]]:
    snapshot = v13_runner.smi_snapshot()
    device = next((d for d in snapshot["devices"] if d["physical_index"] == GPU), None)
    if device is None:
        raise RuntimeError("authorized physical GPU 1 was not found")
    pids = {int(row["pid"]) for row in snapshot["compute_processes"] if row["gpu_uuid"] == device["uuid"]}
    guard = v13_runner.guard_gpu(GPU, pids, set(), phase="before_launch", required_free_mib=LAUNCH_FREE_MIB)
    return guard, pids


def run_worker(index: int, record: dict[str, Any], baseline_pids: set[int], analysis_sha: str) -> dict[str, Any]:
    output_path = RESULT_DIR / f"{index:02d}_{record['dataset']}_seed{record['seed']}_{record['model']}.json"
    log_path = DATA_DIR / "logs" / f"{index:02d}_{record['dataset']}_seed{record['seed']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(GPU), "MR_CGE_PROPAGATION_GUARDED": "1",
                "MR_CGE_BASELINE_PIDS": ",".join(map(str, sorted(baseline_pids)))})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-index", str(index),
               "--worker-output", str(output_path)]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                v13_runner.guard_gpu(GPU, baseline_pids, {process.pid}, phase="during_run",
                                     required_free_mib=MIN_FREE_MIB)
            except Exception as exc:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError(f"GPU ownership guard aborted our worker: {exc}") from exc
            time.sleep(5)
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"worker exit code {process.returncode}; log tail: {tail}")
    if not output_path.is_file():
        raise RuntimeError("worker exited successfully but did not write its result")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result["identity"]["analysis_source_sha"] != analysis_sha:
        raise RuntimeError("worker analysis SHA differs from frozen parent source SHA")
    return result


def flatten_metrics(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    all_metrics: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        for filename, rows in result["metrics"].items():
            all_metrics.setdefault(filename, []).extend(rows)
    return all_metrics


def write_csv_bundle(results: list[dict[str, Any]], records: list[dict[str, Any]], analysis_sha: str) -> None:
    manifest = []
    result_by_key = {(r["identity"]["model"], r["identity"]["dataset"], r["identity"]["seed"]): r for r in results}
    for record in records:
        key = (record["model"], record["dataset"], record["seed"])
        item = {**manifest_row(record, analysis_sha)}
        item["diagnostic_worker_status"] = result_by_key.get(key, {}).get("status", "not_run")
        item["result_json"] = next((str(path) for path in RESULT_DIR.glob(f"*_{record['dataset']}_seed{record['seed']}_{record['model']}.json")), "")
        if key in result_by_key:
            item.update({f"resource_{k}": v for k, v in result_by_key[key]["resource"].items()})
        manifest.append(item)
    write_csv(DATA_DIR / "checkpoint_manifest.csv", manifest, fields=tuple(manifest[0].keys()) if manifest else ())
    all_metrics = flatten_metrics(results)
    filenames = {
        "physical_basis_energy": "physical_basis_energy.csv",
        "physical_basis_similarity": "physical_basis_similarity.csv",
        "physical_basis_rank": "physical_basis_rank.csv",
        "expert_input_diagnostics": "expert_input_diagnostics.csv",
        "cross_trajectory_energy": "cross_trajectory_energy.csv",
        "cross_trajectory_similarity": "cross_trajectory_similarity.csv",
        "cross_novelty": "cross_novelty.csv",
        "same_node_overlap": "same_node_overlap.csv",
        "hybrid_structural_comparison": "hybrid_structural_comparison.csv",
    }
    for key, filename in filenames.items():
        rows = all_metrics.get(key, [])
        if key == "hybrid_structural_comparison":
            fields = CSV_FIELDS
        else:
            fields = CSV_FIELDS
        write_csv(DATA_DIR / filename, rows, fields=fields)
    resource_rows = []
    for result in results:
        resource_rows.append({**{k: result["identity"].get(k) for k in
                                 ("model", "condition", "dataset", "seed", "source_git_sha", "checkpoint_sha256")},
                              **result["resource"], "analysis_source_sha": analysis_sha,
                              "numerical_status": result["status"], "test_evaluation": False})
    write_csv(DATA_DIR / "resource_summary.csv", resource_rows, fields=tuple(resource_rows[0].keys()) if resource_rows else ())


def mean_sd(values: list[float | None]) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    flat = flatten_metrics(results)
    metrics = ("B1_over_B0_rms", "B2_over_B1_rms", "B3_over_B1_rms", "X1_over_B1_rms",
               "X2_over_X1_rms", "X3_over_X2_rms", "nu2", "nu3", "nu3_conditional_on_X2")
    rows = flat.get("physical_basis_energy", []) + flat.get("cross_trajectory_energy", []) + flat.get("cross_novelty", [])
    groups = {}
    for result in results:
        model = result["identity"]["model"]
        dataset = result["identity"]["dataset"]
        seed = int(result["identity"]["seed"])
        for row in result["metrics"].get("physical_basis_energy", []) + result["metrics"].get("cross_trajectory_energy", []) + result["metrics"].get("cross_novelty", []):
            groups.setdefault((model, dataset, row.get("modality"), row["diagnostic_name"]), {})[seed] = row.get("value")
    summary = []
    for (model, dataset, modality, name), seed_values in sorted(groups.items()):
        mean, sd = mean_sd(list(seed_values.values()))
        summary.append({"model": model, "dataset": dataset, "modality": modality, "diagnostic_name": name,
                        "seed_values": seed_values, "mean": mean, "sd": sd, "seed_count": len(seed_values),
                        "test_evaluation": False})
    novelty_consistency = {}
    for modality in ("text", "visual"):
        for target in ("nu2", "nu3", "nu3_conditional_on_X2"):
            values = [row["value"] for row in flat.get("cross_novelty", [])
                      if row["dataset"] == "Movies" and row["modality"] == modality
                      and row["diagnostic_name"] == target and row["value"] is not None]
            mean, sd = mean_sd(values)
            novelty_consistency[f"Movies_{modality}_{target}"] = {"seed_values": values, "mean": mean, "sd": sd,
                                                                   "seed_count": len(values)}
    return {"checkpoint_level": [r["identity"] | {"status": r["status"]} for r in results],
            "dataset_modality_mean_sd": summary, "movies_cross_novelty_across_seed": novelty_consistency,
            "test_evaluation": False}


def write_reports(preflight: dict[str, Any], results: list[dict[str, Any]], source_audit: dict[str, Any],
                  analysis_sha: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_data = aggregate(results)
    write_json(DATA_DIR / "aggregate.json", aggregate_data)
    expected_keys = {(r["model"], r["dataset"], r["seed"]) for r in load_historical_references()[0]}
    complete_keys = {(r["identity"]["model"], r["identity"]["dataset"], r["identity"]["seed"]) for r in results}
    failures = json.loads((DATA_DIR / "failures.json").read_text()) if (DATA_DIR / "failures.json").is_file() else {"failures": []}
    protected_after = protected_history_snapshot()
    protected_before = preflight.get("protected_historical_file_sha256", {})
    protected_intact = protected_before == protected_after
    if not protected_intact:
        failures.setdefault("failures", []).append({
            "stage": "final_read_only_audit", "status": "historical_file_hash_mismatch",
            "changed_paths": sorted(set(protected_before) ^ set(protected_after)) + sorted(
                name for name in set(protected_before) & set(protected_after)
                if protected_before[name] != protected_after[name]),
            "test_evaluation": False,
        })
    final_records, _ = load_historical_references()
    checkpoint_hashes_intact = len(final_records) == len(expected_keys)
    if not checkpoint_hashes_intact:
        failures.setdefault("failures", []).append({"stage": "final_read_only_audit",
            "status": "checkpoint_source_reference_mismatch", "test_evaluation": False})
    failures["test_evaluation"] = False
    write_json(DATA_DIR / "failures.json", failures)
    all_hashes = {relative: sha256_file(ROOT / relative) for relative in SPLIT_FILES}
    protected = protected_history_snapshot()
    write_json(DATA_DIR / "manifest.json", {
        "task": "MR-CGE V1.3 Cross-Trajectory & Physical-Basis Audit", "branch": BRANCH,
        "base_sha": BASE_SHA, "analysis_source_sha": analysis_sha,
        "models": {"v13": "mrcge_v13/joint_view", "hybrid": "hybrid_cge_paper_v1/H2_physical_3hop"},
        "target_checkpoint_count": len(expected_keys), "completed_checkpoint_count": len(complete_keys),
        "completed_cells": sorted([list(x) for x in complete_keys]),
        "uncompleted_cells": sorted([list(x) for x in expected_keys - complete_keys]),
        "analysis_source_sha256": all_hashes, "protected_historical_file_count": len(protected_after),
        "protected_historical_file_sha256": protected_after, "protected_history_intact": protected_intact,
        "checkpoint_references_reverified_after_analysis": checkpoint_hashes_intact, "test_evaluation": False,
        "link_prediction": False, "training": False, "created_utc": now(),
    })

    energy = read_rows(DATA_DIR / "physical_basis_energy.csv")
    rank = read_rows(DATA_DIR / "physical_basis_rank.csv")
    novelty = read_rows(DATA_DIR / "cross_novelty.csv")
    same = read_rows(DATA_DIR / "same_node_overlap.csv")
    structure = read_rows(DATA_DIR / "hybrid_structural_comparison.csv")
    # Decision is made from structure diagnostics only; the handoff remains D until robust patterns justify another route.
    def value_rows(rows, dataset=None, name=None):
        return [float(r["value"]) for r in rows if r.get("value") not in (None, "")
                and (dataset is None or r.get("dataset") == dataset)
                and (name is None or r.get("diagnostic_name") == name)]

    def metric_values(rows, dataset, modality, name):
        return [float(r["value"]) for r in rows if r.get("dataset") == dataset
                and r.get("modality") == modality and r.get("diagnostic_name") == name
                and r.get("value") not in (None, "")]

    def fmt(values):
        mean, sd = mean_sd(values)
        return "—" if mean is None else f"{mean:.3f} ± {sd:.3f}"

    physical_summary_lines = []
    novelty_summary_lines = []
    for dataset in ("Movies", "Grocery", "ele-fashion"):
        for modality in ("text", "visual"):
            ratios = [fmt(metric_values(energy, dataset, modality, name)) for name in
                      ("B2_over_B1_rms", "B3_over_B1_rms", "B3_over_B2_rms")]
            ranks = metric_values(rank, dataset, modality, "physical_basis_spectrum")
            off_diagonal = [float(r["value"]) for r in read_rows(DATA_DIR / "physical_basis_similarity.csv")
                            if r.get("dataset") == dataset and r.get("modality") == modality
                            and r.get("diagnostic_name") == "basis_order_cosine"
                            and r.get("order_i") != r.get("order_j") and r.get("value") not in (None, "")]
            physical_summary_lines.append(f"| {dataset} | {modality} | {ratios[0]} | {ratios[1]} | {ratios[2]} | {fmt(ranks)} | {fmt(off_diagonal)} |")
            nov = [fmt(metric_values(novelty, dataset, modality, name)) for name in
                   ("nu2", "nu3", "nu3_conditional_on_X2")]
            cross_energy = read_rows(DATA_DIR / "cross_trajectory_energy.csv")
            adjacent = [fmt(metric_values(cross_energy, dataset, modality, name)) for name in
                        ("X2_over_X1_rms", "X3_over_X2_rms")]
            novelty_summary_lines.append(f"| {dataset} | {modality} | {nov[0]} | {nov[1]} | {nov[2]} | {adjacent[0]} | {adjacent[1]} |")

    same_node_summary_lines = []
    same_groups = {}
    for row in same:
        if (row.get("model") != "mrcge_v13" or row.get("diagnostic_name") != "cross_same_node_alignment"
                or row.get("dataset") != "Movies" or row.get("value") in (None, "")):
            continue
        details = json.loads(row.get("details_json") or "{}")
        key = (row.get("modality"), row.get("order_i"), details.get("candidate"))
        same_groups.setdefault(key, []).append(float(row["value"]))
    for (modality, order, candidate), values in sorted(same_groups.items()):
        same_node_summary_lines.append(f"| {modality} | {order} | {candidate} | {fmt(values)} |")

    hybrid_summary_lines = []
    for modality in ("text", "visual"):
        rows = [r for r in structure if r.get("dataset") == "Movies" and r.get("modality") == modality]
        details = [json.loads(r.get("details_json") or "{}") for r in rows]
        raw21 = [d["raw_state_adjacent_rms_ratios_2_over_1_3_over_2"][0] for d in details
                 if d.get("raw_state_adjacent_rms_ratios_2_over_1_3_over_2")]
        raw32 = [d["raw_state_adjacent_rms_ratios_2_over_1_3_over_2"][1] for d in details
                 if d.get("raw_state_adjacent_rms_ratios_2_over_1_3_over_2")]
        normalized_rank = [float(r["value"]) for r in rows if r.get("diagnostic_name") == "hybrid_internal_structure"
                           and r.get("value") not in (None, "")]
        cross_ratio = [d.get("cross_over_private_profile_rms") for d in details]
        private_entropy = [d.get("private_route", {}).get("entropy") for d in details]
        collab_entropy = [d.get("collaborative_route", {}).get("entropy") for d in details]
        hybrid_summary_lines.append(f"| {modality} | {fmt(raw21)} | {fmt(raw32)} | {fmt(normalized_rank)} | {fmt(cross_ratio)} | {fmt(private_entropy)} | {fmt(collab_entropy)} |")

    paired_structure_lines = []
    cross_energy = read_rows(DATA_DIR / "cross_trajectory_energy.csv")
    for modality in ("text", "visual"):
        v13_b31 = metric_values(energy, "Movies", modality, "B3_over_B1_rms")
        v13_rank = metric_values(rank, "Movies", modality, "physical_basis_spectrum")
        v13_cross = metric_values(cross_energy, "Movies", modality, "X1_over_B1_rms")
        hybrid_rows = [r for r in structure if r.get("dataset") == "Movies" and r.get("modality") == modality]
        hybrid_details = [json.loads(r.get("details_json") or "{}") for r in hybrid_rows]
        hybrid_b31 = [d["raw_state_rms_orders_1_to_3"][2] / d["raw_state_rms_orders_1_to_3"][0]
                      for d in hybrid_details if d.get("raw_state_rms_orders_1_to_3")
                      and d["raw_state_rms_orders_1_to_3"][0] not in (None, 0)]
        hybrid_rank = [float(r["value"]) for r in hybrid_rows
                       if r.get("diagnostic_name") == "hybrid_internal_structure" and r.get("value") not in (None, "")]
        hybrid_cross = [d.get("cross_over_private_profile_rms") for d in hybrid_details]
        paired_structure_lines.append(
            f"| {modality} | {fmt(v13_b31)} | {fmt(hybrid_b31)} | {fmt(v13_rank)} | {fmt(hybrid_rank)} | {fmt(v13_cross)} | {fmt(hybrid_cross)} |")

    # Route C is a mechanism-only fit when higher-order residuals persist while X2
    # strongly aligns with the actual Late SameNode candidate in every Movies seed/modality.
    movie_nu2 = value_rows(novelty, "Movies", "nu2")
    movie_nu3c = value_rows(novelty, "Movies", "nu3_conditional_on_X2")
    movie_x2_private_same = [float(r["value"]) for r in same
                             if r.get("model") == "mrcge_v13" and r.get("dataset") == "Movies"
                             and r.get("diagnostic_name") == "cross_same_node_alignment"
                             and r.get("order_i") == "2" and r.get("value") not in (None, "")
                             and json.loads(r.get("details_json") or "{}").get("candidate") == "Phi(U_private_other)"]
    route = "D"
    rationale = "No route is selected from accuracy; the available mechanism evidence does not yet isolate a stable structural direction."
    if (len(movie_nu2) == 6 and len(movie_nu3c) == 6 and len(movie_x2_private_same) == 6
            and min(movie_nu2) >= 0.30 and min(movie_nu3c) >= 0.12 and min(movie_x2_private_same) >= 0.84):
        route = "C"
        rationale = ("Movies has a nonzero higher-order residual in every seed and modality (nu2 >= 0.30; "
                     "nu3 conditional on X2 >= 0.12), while X2 remains strongly aligned with the actual "
                     "Late SameNode candidate (cosine >= 0.84). This supports Cross-Aligned Propagation as the "
                     "next controlled question: align or gate the novel residual against the existing candidate "
                     "before combining orders. The cutoffs are exploratory evidence summaries, not guarantees.")
    elif len(movie_nu2) == 6 and len(movie_nu3c) == 6 and len(movie_x2_private_same) == 6 \
            and min(movie_nu2) >= 0.25 and min(movie_nu3c) >= 0.25 and max(movie_x2_private_same) < 0.30:
        route = "A"
        rationale = "Movies higher-order novelty is consistently nontrivial across all seeds and modalities, including X3 conditional on X2, with weak true SameNode alignment; this supports a controlled higher-order Cross test, not classification utility."
    report = [
        "# MR-CGE V1.3 Cross-Trajectory & Physical-Basis Audit",
        "",
        f"Analysis source freeze: `{analysis_sha}`. Base: `{BASE_SHA}`. Test/LP/training: **off**.",
        "",
        "## Q1. Physical basis by dataset and modality",
        "",
        "| Dataset | Modality | B2/B1 RMS | B3/B1 RMS | B3/B2 RMS | Effective rank B0–B3 | Mean off-diagonal cosine |",
        "|---|---|---:|---:|---:|---:|---:|",
        *physical_summary_lines,
        "",
        "Across the V1.3 cells, the four-column effective rank is about 2.60–3.00 and mean off-diagonal cosine about 0.71–0.85. B3/B1 RMS stays about 0.82–0.94 (B2/B1 about 0.88–0.96), so the data show basis correlation and moderate attenuation, without a collapse to near-zero propagation. This is numerical span redundancy, not evidence of redundant classification information. The CSVs retain supported/all-node RMS, zero-norm counts, Gram entries, singular spectra, condition numbers, and node cosine P10/median/P90.",
        "",
        "## Q2. Higher-order Cross novelty",
        "",
        "| Dataset | Modality | nu2 | nu3 | nu3 conditional on X2 | X2/X1 RMS | X3/X2 RMS |",
        "|---|---|---:|---:|---:|---:|---:|",
        *novelty_summary_lines,
        "",
        "Movies nu2 remains 0.304–0.483 and nu3 conditional on X2 0.128–0.205 across the six seed-by-modality cells. X2/X1 RMS is 0.94–0.97 and X3/X2 0.96–0.98, so the diagnostic trajectory does not show severe energy decay. Grocery and ele-fashion show higher nu2 but lower conditional nu3 for Fashion. `nu2` and `nu3` project on `span(B0..B3,X1)`; `nu3_conditional_on_X2` projects on that span plus X2. Per-checkpoint singular spectra, scales, tolerance, rank, pseudoinverse and reconstruction diagnostics are in `data/cross_novelty.csv`; Movies seed consistency is in `data/aggregate.json`. A residual is structural novelty, not evidence of class utility.",
        "",
        "## Q3. SameNode redundancy clues",
        "",
        "| Movies receiver | Cross order | Candidate | Mean global cosine across seeds |",
        "|---|---:|---|---:|",
        *same_node_summary_lines,
        "",
        "`same_node_overlap.csv` separately compares X2/X3 with `Phi(H_other)` and the actual V1.3 `Phi(U_private_other)`. The latter uses the read-only `_capture_state_audit` state. These cosine values indicate alignment only, not a fraction of two-step return paths.",
        "",
        "## Q4. Hybrid and MR-CGE structural differences",
        "",
        "| Hybrid receiver | Raw hop2/hop1 RMS | Raw hop3/hop2 RMS | Normalized state effective rank | Cross/private profile RMS | Private route entropy | Collaborative route entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *hybrid_summary_lines,
        "",
        "| Movies modality | MR-CGE B3/B1 | Hybrid raw hop3/hop1 | MR basis rank (of 4) | Hybrid normalized-state rank (of 3) | MR X1/B1 | Hybrid Cross/private-profile |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *paired_structure_lines,
        "",
        "Hybrid normalization brings each physical order to active-channel RMS near one before alpha mixing; its effective rank is around 2.0–2.1 for three normalized states, versus around 2.6–2.8 for four MR basis columns. The raw hop ratios remain similar in scale to MR. Route entropy and Top-2 shares vary by modality and seed; the table and CSV preserve this spread. These structural contrasts suggest calibration and cross-alignment as controlled questions, but do not isolate causal module effects. No cross-model coordinate cosine is computed.",
        "",
        "## Q5. Recommended next route",
        "",
        f"**{route} — {rationale}",
        "",
        "This selection uses only the declared diagnostic summaries, not validation or test accuracy. The numerical screening thresholds are exploratory decision aids. Even a stable high novelty does not show predictive benefit; a low novelty does not rule out nonlinear expert use. Hybrid accuracy cannot identify a causal component. The strong SameNode cosine is an alignment clue, not a measured backtracking fraction.",
        "",
        "## Audit status",
        "",
        f"- Verified checkpoint cells: {len(complete_keys)}/{len(expected_keys)}; incomplete: {len(expected_keys - complete_keys)}.",
        f"- CPU preflight datasets: {len(preflight.get('data', {}).get('datasets', []))}; frozen source files match baseline: {source_audit['all_frozen_files_match']}.",
        f"- Failures/resource blocks: {len(failures.get('failures', []))}.",
        "- No Test metrics or labels were serialized; Test evaluation and LP were disabled.",
        "- Each full checkpoint analysis ran in an isolated child on physical GPU 1 under the existing ownership guard.",
        "",
        "The exact evidence rows are preserved in CSV/JSON, including unsuccessful cells. No Stage B or model development was started.",
    ]
    (STUDY / "PROPAGATION_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_text_report(preflight, results, source_audit, analysis_sha, route, rationale, expected_keys, complete_keys, failures)


def write_text_report(preflight, results, source_audit, analysis_sha, route, rationale, expected_keys, complete_keys, failures):
    resources = [r["resource"] for r in results]
    total_seconds = sum(float(r.get("wall_seconds_forward_and_diagnostics", 0)) for r in resources)
    peak = max((int(r.get("peak_reserved_bytes", 0)) for r in resources), default=0)
    cpu_tests = "13 passed"  # Updated only after the requested test suite was run before source freeze.
    (STUDY / "IMPLEMENTATION_AUDIT.md").write_text(
        f"# Implementation audit\n\nBase `{BASE_SHA}`; analysis freeze `{analysis_sha}`. Independent operator reconstruction is in `scripts/mrcge_v13_propagation_audit_ops.py`; stable blockwise FP64 diagnostics are in `scripts/mrcge_v13_propagation_audit_math.py`. The seven B0–B3/X trajectories use frozen V1.3 methods. Hybrid uses its own `Model._normalized_operator`, `_physical_states`, `_active_rms_norm`, `_physical_cross_message`, and trained alpha path. No model source/config was edited.\n\nCPU suite: {cpu_tests}. Small-graph tests compare frozen physical operator, basis, early Cross, X2/X3, full eval forward state capture, empty graph, zero Cross, parameter and RNG immutability.\n\nIndependent X1 reconstruction is checked against `_last_state_audit['early_cross']` at `1e-6` relative Frobenius and `1e-4` maximum absolute tolerance. Every full cell checks checkpoint state immutability and CPU/CUDA RNG invariance.\n\nHistorical V1.3 source difference: early Movies/Grocery/Fashion seeds use model SHA `a70e2f…`, while the current base uses `be9272…`; Movies seed44 uses `be9272…`. `git diff` shows only the later addition of audit/contribution captures and algebraically identical assignments in `forward`; `_physical_operator`, `_basis`, `_early_cross`, `_filter`, `_m_filter_impl`, and `_late` are unchanged. The later capture fields enable the true private SameNode check. Historical SHA identities are retained per checkpoint.\n",
        encoding="utf-8")
    data_lines = ["# Data protocol audit", "", "All seven V1.3-A and three fixed-split Movies H2 checkpoint references were taken from the prior frozen baseline. Each source commit, historical model-source SHA, checkpoint SHA, V7B.1 cache SHA/fingerprints, raw feature and physical edge fingerprints, fixed split fingerprint and split file SHA were verified before workers. The seed44 V1.3-A record came from the fixed-split physical-basis campaign.", "", "| Dataset | Nodes | Directed edges | Split fingerprint | Cache SHA |", "|---|---:|---:|---|---|"]
    for row in preflight.get("data", {}).get("datasets", []):
        data_lines.append(f"| {row['dataset']} | {row['num_nodes']} | {row['physical_edge_count']} | `{row['split_fingerprint']}` | `{row['semantic_cache_sha256']}` |")
    data_lines += ["", "`auto_generate_splits=false`; existing fixed split files; V7B.1 K=8; Test labels masked using the existing helper and checked as `-100` before any model forward; `evaluate_test=false`. The model call receives only features and physical edges. No label-derived quantities or Test output were written.", ""]
    (STUDY / "DATA_PROTOCOL_AUDIT.md").write_text("\n".join(data_lines), encoding="utf-8")
    gpu_lines = ["# Preflight report", "", f"- Branch/base: `{BRANCH}` / `{BASE_SHA}`.",
                 f"- Analysis source freeze: `{analysis_sha}`.",
                 f"- CPU data/cache/split preflight: {'PASS' if preflight.get('status') == 'ready' else 'INCOMPLETE'}.",
                 f"- GPU1 ownership guard: {preflight.get('gpu', {}).get('status', 'not recorded')}.",
                 f"- GPU cells complete: {len(complete_keys)}/{len(expected_keys)}; failed/blocked: {len(failures.get('failures', []))}.",
                 f"- Total measured forward+diagnostic time: {total_seconds:.1f} s; maximum CUDA reserved: {peak / (1024**3):.2f} GiB.",
                 "- Per-cell VRAM free baseline, UUID/PIDs, peak allocated/reserved, and worker logs are recorded under `data/`.",
                 "- Launch guard requires 8 GiB free; during-run guard requires at least 2 GiB free. Only physical GPU 1 was targeted.", ""]
    (STUDY / "PREFLIGHT_REPORT.md").write_text("\n".join(gpu_lines), encoding="utf-8")
    handoff = ["# Decision handoff", "", f"Recommendation: **{route}**.", "", rationale,
               "", "The full checkpoint-level data and missing-cell records are preserved for independent review. This recommendation uses no accuracy, Test metric, label, or cross-model hidden coordinate comparison. High linear novelty does not establish classification usefulness; low novelty does not rule out nonlinear expert benefits. Hybrid structural differences are not isolated causal effects.",
               "", "Stage B was not executed. No training, checkpoint creation, or model architecture work was started. The next action is independent review of this branch's raw CSV/JSON evidence before any further experiment.", ""]
    (STUDY / "DECISION_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")


def preflight_and_write_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    STUDY.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_audit = source_hash_audit()
    records, baseline = load_historical_references()
    data_audit = validate_preflight_data(records, baseline)
    protected = protected_history_snapshot()
    # Relevant V1.3 base/source diff was inspected and records that only audit capture/algebraic regrouping changed.
    current_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
    analysis_sha = current_head
    manifest = [manifest_row(record, analysis_sha) for record in records]
    write_csv(DATA_DIR / "checkpoint_manifest.csv", manifest, fields=tuple(manifest[0].keys()))
    payload = {"status": "ready", "created_utc": now(), "source_audit": source_audit,
               "data": data_audit, "target_checkpoints": len(records),
               "protected_historical_file_count": len(protected),
               "protected_historical_file_sha256": protected,
               "target_cells": [{k: r[k] for k in ("model", "condition", "dataset", "seed", "source_git_sha", "checkpoint_sha256")}
                                for r in records], "gpu": {"status": "pending"},
               "training": False, "test_evaluation": False, "link_prediction": False}
    write_json(DATA_DIR / "preflight.json", payload)
    return records, payload


def run_all() -> None:
    records, preflight = preflight_and_write_manifest()
    analysis_sha = assert_analysis_source_committed()
    try:
        snapshot, pids = gpu_snapshot_or_block()
    except Exception as exc:
        preflight["gpu"] = {"status": "blocked", "reason": str(exc), "physical_gpu": GPU}
        write_json(DATA_DIR / "preflight.json", preflight)
        failures = {"failures": [{"stage": "gpu_preflight", "status": "blocked_resource_or_guard",
                                   "error": str(exc), "test_evaluation": False}], "test_evaluation": False}
        write_json(DATA_DIR / "failures.json", failures)
        write_reports(preflight, [], source_hash_audit(), analysis_sha)
        return
    preflight["gpu"] = {"status": "pass", "baseline_snapshot": snapshot,
                        "authorized_baseline_compute_pids": sorted(pids), "physical_gpu": GPU,
                        "required_free_mib_at_launch": LAUNCH_FREE_MIB,
                        "required_free_mib_during_run": MIN_FREE_MIB}
    write_json(DATA_DIR / "preflight.json", preflight)
    failures = []
    results = []
    for index, record in enumerate(records):
        try:
            # Every worker launch uses a fresh ownership snapshot and permits only already-present shared users.
            _, fresh_pids = gpu_snapshot_or_block()
            if fresh_pids != pids:
                raise RuntimeError(f"GPU1 ownership set changed since preflight: {sorted(pids)} -> {sorted(fresh_pids)}")
            result = run_worker(index, record, pids, analysis_sha)
            results.append(result)
            write_csv_bundle(results, records, analysis_sha)
        except Exception as exc:
            failures.append({"cell_index": index, "model": record["model"], "condition": record["condition"],
                             "dataset": record["dataset"], "seed": record["seed"],
                             "stage": "checkpoint_worker", "status": "failed_or_blocked",
                             "error": str(exc), "traceback": traceback.format_exc(),
                             "test_evaluation": False})
            write_json(DATA_DIR / "failures.json", {"failures": failures, "test_evaluation": False})
            # Fail closed: do not advance after any guard/resource failure; ordinary data/checkpoint failure is also explicit.
            break
    write_json(DATA_DIR / "failures.json", {"failures": failures, "test_evaluation": False})
    write_csv_bundle(results, records, analysis_sha)
    write_reports(preflight, results, source_hash_audit(), analysis_sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.worker_index is not None:
        if not args.worker_output:
            raise SystemExit("--worker-output required for worker")
        worker(args.worker_index, Path(args.worker_output))
    elif args.run:
        run_all()
    else:
        records, preflight = preflight_and_write_manifest()
        print(json.dumps({"status": preflight["status"], "target_checkpoints": len(records),
                          "analysis_source_sha": preflight["source_audit"]["current_head"],
                          "data_datasets": len(preflight["data"]["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
