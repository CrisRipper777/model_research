#!/usr/bin/env python3
"""Run the validation-only SOSB-MAG V1.5 structural basis screen."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BRANCH = "exp/sosb_mag_v15_basis_screen"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = (
    ("A0_legacy_lg", "A0_legacy_lg"),
    ("A1_rawpoly_shared", "A1_rawpoly_shared"),
    ("A2_sosb_shared", "A2_sosb_shared"),
    ("A3_sosb_modality", "A3_sosb_modality"),
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sosb_mag_v15_basis_screen"
DEFAULT_RESEARCH = ROOT / "research" / "sosb_mag_v15_basis_screen"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_sha() -> str | None:
    value = git_value(
        "log",
        "--format=%H",
        "-1",
        "--grep=^Freeze SOSB-MAG V1.5 basis screening implementation$",
    )
    return value or None


def check_provenance(*, smoke: bool, allow_dirty: bool) -> dict[str, Any]:
    branch = git_value("branch", "--show-current")
    head = git_value("rev-parse", "HEAD")
    dirty = git_value("status", "--porcelain")
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH!r}, found {branch!r}")
    if dirty and not allow_dirty:
        raise RuntimeError("Working tree is dirty; use --allow-dirty only for the pre-freeze smoke.")
    frozen = freeze_sha()
    if not smoke:
        if frozen is None:
            raise RuntimeError("Freeze commit not found; finish smoke and commit the frozen implementation first.")
        if head != frozen:
            raise RuntimeError(f"Formal campaign requires HEAD to equal freeze commit {frozen}; found {head}.")
        if dirty:
            raise RuntimeError("Formal campaign requires a clean worktree after the freeze commit.")
    return {
        "branch": branch,
        "head": head,
        "freeze_commit_sha": frozen,
        "worktree_clean": not bool(dirty),
        "started_at_utc": now(),
    }


def validate_metrics(metrics_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing run metrics/checkpoint: {metrics_path} / {checkpoint_path}")
    payload = read_json(metrics_path)
    if len(payload.get("runs", [])) != 1:
        raise RuntimeError(f"Expected exactly one run in {metrics_path}")
    run = payload["runs"][0]
    metrics = dict(run["metrics"])
    if any(key.lower().startswith("test") for key in metrics):
        raise RuntimeError(f"Test metrics present despite evaluate_test=false: {metrics_path}")
    for key in ("val_acc", "val_macro_f1"):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise FloatingPointError(f"Missing or non-finite {key} in {metrics_path}")
    if "best_epoch" not in run.get("metadata", {}):
        raise RuntimeError(f"Missing validation-selected best epoch in {metrics_path}")
    return {"metrics": metrics, "metadata": dict(run.get("metadata", {}))}


def train_loss_from_log(log_path: Path) -> float:
    if not log_path.is_file():
        raise FileNotFoundError(str(log_path))
    values = re.findall(r"Train Loss ([+-]?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)", log_path.read_text(encoding="utf-8", errors="replace"))
    if not values:
        raise RuntimeError(f"Could not find a training loss in {log_path}")
    result = float(values[-1])
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite training loss in {log_path}")
    return result


def _gram_diagnostics(basis: torch.Tensor, active: torch.Tensor, stable_channels: torch.Tensor | None = None) -> dict[str, float]:
    selected = basis[:, active]
    if stable_channels is not None and bool(stable_channels.any()):
        selected = selected[:, :, stable_channels]
    count = int(selected.size(1) * selected.size(2))
    if count == 0:
        raise ValueError("Cannot compute a Gram matrix without active node/channel entries")
    gram = torch.einsum("knd,lnd->kl", selected.float(), selected.float()) / count
    symmetric = 0.5 * (gram + gram.t())
    eig = torch.linalg.eigvalsh(symmetric)
    jitter = 1.0e-6 * max(1.0, abs(float(torch.trace(symmetric).item())) / symmetric.size(0))
    condition = float(((eig.max() + jitter) / (eig.min() + jitter)).item())
    diagonal = torch.diag(gram)
    off = gram - torch.diag(diagonal)
    return {
        "gram_diag_min": float(diagonal.min().item()),
        "gram_diag_max": float(diagonal.max().item()),
        "gram_diag_mean": float(diagonal.mean().item()),
        "gram_offdiag_abs_mean": float(off.abs().sum().item() / max(off.numel() - len(diagonal), 1)),
        "gram_offdiag_abs_max": float(off.abs().max().item()),
        "gram_condition_number": condition,
        "gram_condition_jitter": jitter,
    }


def audit_checkpoint(
    dataset: str,
    seed: int,
    label: str,
    variant: str,
    output_root: Path,
    device_name: str,
) -> dict[str, Any]:
    from scripts.analyze_sosb_mag_v15_basis_screen import load_features_and_edges
    from src.models.sosb_mag_v15 import Model

    run_dir = output_root / "runs" / dataset / f"seed_{seed}" / label
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(run_dir / "hydra" / ".hydra" / "config.yaml")
    cfg.model.variant = variant
    model = Model(cfg, checkpoint["data_info"])
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(device_name)
    model = model.to(device).eval()
    x_cpu, edge_cpu = load_features_and_edges(dataset, seed)
    with torch.no_grad():
        z, _, _, _, info = model(x_cpu.to(device), edge_cpu.to(device), return_details=True)
    if not torch.isfinite(z).all():
        raise FloatingPointError(f"Non-finite embedding in {dataset}/{seed}/{label}")
    result: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "variant_label": label,
        "embedding_finite": True,
        "parameter_count_model": int(checkpoint["run_metadata"]["model_parameters"]),
        "parameter_count_classifier": int(checkpoint["run_metadata"]["classifier_parameters"]),
        "modalities": {},
        "test_metrics_present": False,
    }
    for modality in ("text", "visual"):
        item = info["details"][modality]
        for key in ("basis", "beta", "gate", "prior", "response", "scaled_correction"):
            if not torch.isfinite(item[key]).all():
                raise FloatingPointError(f"Non-finite {key}: {dataset}/{seed}/{label}/{modality}")
        active = info["active_nodes"]
        basis = item["basis"]
        is_sosb = variant in {"A2_sosb_shared", "A3_sosb_modality"}
        stable_channels = ~item["breakdown"].any(dim=0) if is_sosb else None
        gram = _gram_diagnostics(basis, active, stable_channels)
        active_prior = item["prior"][active]
        active_response = item["response"][active]
        active_scaled = item["scaled_correction"][active]
        prior_rms = float(torch.sqrt(active_prior.float().square().mean()).item())
        response_rms = float(torch.sqrt(active_response.float().square().mean()).item())
        scaled_rms = float(torch.sqrt(active_scaled.float().square().mean()).item())
        breakdown = item["breakdown"].float().mean(dim=-1).detach().cpu().tolist()
        result["modalities"][modality] = {
            **gram,
            "basis_finite": True,
            "beta_finite": True,
            "gate_finite": True,
            "response_finite": True,
            "effective_beta": item["beta"].detach().cpu().tolist(),
            "gate": float(item["gate"].item()),
            "prior_rms": prior_rms,
            "structural_response_rms": response_rms,
            "scaled_structural_rms": scaled_rms,
            "scaled_structural_to_prior_rms_ratio": scaled_rms / max(prior_rms, 1.0e-12),
            "breakdown_fraction_by_order": breakdown,
            "stable_channel_count": int(stable_channels.sum().item()) if stable_channels is not None else None,
        }
    del model, x_cpu, edge_cpu, z, info
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_one(
    dataset: str,
    seed: int,
    label: str,
    variant: str,
    *,
    device: str,
    output_root: Path,
    mode: str,
    epochs: int | None,
    resume: bool,
) -> dict[str, Any]:
    run_dir = output_root / mode / "runs" / dataset / f"seed_{seed}" / label
    metrics_path = run_dir / "run_metrics.json"
    checkpoint_path = run_dir / "best.pt"
    if resume and metrics_path.is_file() and checkpoint_path.is_file():
        checked = validate_metrics(metrics_path, checkpoint_path)
        return {
            "status": "reused",
            "dataset": dataset,
            "seed": seed,
            "label": label,
            "variant": variant,
            **checked,
            "train_loss": train_loss_from_log(run_dir / "hydra" / "main.log"),
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=sosb_mag_v15",
        f"model.variant={variant}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        "task.evaluate_test=false",
        f"task.run_metrics_path={metrics_path}",
        f"task.save_ckpt_path={checkpoint_path}",
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    if epochs is not None:
        command.append(f"task.epochs={int(epochs)}")
        if int(epochs) <= 2:
            command.extend(
                [
                    "task.early_stop_min_epoch=1",
                    "task.patience=2",
                    "task.early_stop_min_delta=0.0",
                ]
            )
    print(f"[SOSB-V1.5 {mode}] {dataset} seed={seed} {label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    checked = validate_metrics(metrics_path, checkpoint_path)
    return {
        "status": "completed",
        "dataset": dataset,
        "seed": seed,
        "label": label,
        "variant": variant,
        **checked,
        "train_loss": train_loss_from_log(run_dir / "hydra" / "main.log"),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }


def write_campaign_state(
    data_dir: Path,
    provenance: dict[str, Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    mode: str,
    device: str,
    epochs: int | None,
    expected: int,
) -> None:
    write_json(data_dir / "run_rows.json", rows)
    write_json(
        data_dir / ("smoke_summary.json" if mode == "smoke" else "campaign_manifest.json"),
        {
            **provenance,
            "updated_at_utc": now(),
            "mode": mode,
            "datasets": ["Movies"] if mode == "smoke" else list(DATASETS),
            "seeds": [42] if mode == "smoke" else list(SEEDS),
            "variants": [label for label, _ in VARIANTS],
            "expected_runs": expected,
            "completed_or_reused_runs": len(rows),
            "failures": failures,
            "unresolved_failures": [item for item in failures if not item.get("resolved", False)],
            "device": device,
            "epochs_override": epochs,
            "task_protocol": "unified_full_graph_nc_v1",
            "task_evaluate_test": False,
            "selection": "best_validation_accuracy",
            "test_evaluation_attempted": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--campaign", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    is_smoke = bool(args.smoke)
    mode = "smoke" if is_smoke else "campaign"
    provenance = check_provenance(smoke=is_smoke, allow_dirty=args.allow_dirty)
    output_root = args.out_dir.expanduser().resolve()
    datasets = ("Movies",) if is_smoke else DATASETS
    seeds = (42,) if is_smoke else SEEDS
    epochs = (1 if args.epochs is None else args.epochs) if is_smoke else args.epochs
    expected = len(datasets) * len(seeds) * len(VARIANTS)
    research_data = DEFAULT_RESEARCH / "data"
    campaign_state_dir = output_root / "campaign" / "state"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not is_smoke and args.resume:
        previous_manifest = campaign_state_dir / "campaign_manifest.json"
        if previous_manifest.is_file():
            failures = list(read_json(previous_manifest).get("failures", []))

    for dataset in datasets:
        for seed in seeds:
            for label, variant in VARIANTS:
                try:
                    row = run_one(
                        dataset,
                        seed,
                        label,
                        variant,
                        device=args.device,
                        output_root=output_root,
                        mode=mode,
                        epochs=epochs,
                        resume=args.resume,
                    )
                    rows.append(row)
                    if is_smoke:
                        audit = audit_checkpoint(
                            dataset,
                            seed,
                            label,
                            variant,
                            output_root / "smoke",
                            args.device,
                        )
                        row["audit"] = audit
                        audit["train_loss"] = row["train_loss"]
                        audit["metrics"] = row["metrics"]
                        audit["metadata"] = row["metadata"]
                        audit["checkpoint_path"] = row["checkpoint_path"]
                    else:
                        for previous_failure in failures:
                            same_run = (
                                previous_failure.get("dataset") == dataset
                                and int(previous_failure.get("seed", -1)) == seed
                                and previous_failure.get("label") == label
                            )
                            if same_run and not previous_failure.get("resolved", False):
                                previous_failure["resolved"] = True
                                previous_failure["resolved_at_utc"] = now()
                    write_campaign_state(
                        research_data if is_smoke else campaign_state_dir,
                        provenance,
                        rows,
                        failures,
                        mode=mode,
                        device=args.device,
                        epochs=epochs,
                        expected=expected,
                    )
                except Exception as exc:
                    failure = {
                        "dataset": dataset,
                        "seed": seed,
                        "label": label,
                        "variant": variant,
                        "error": repr(exc),
                        "oom": "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower(),
                        "nonfinite": any(token in str(exc).lower() for token in ("non-finite", "nan", "inf")),
                    }
                    failures.append(failure)
                    write_json(
                        output_root / mode / "failures" / dataset / f"seed_{seed}_{label}.json",
                        failure,
                    )
                    write_campaign_state(
                        research_data if is_smoke else campaign_state_dir,
                        provenance,
                        rows,
                        failures,
                        mode=mode,
                        device=args.device,
                        epochs=epochs,
                        expected=expected,
                    )
                    print(f"[SOSB-V1.5 failed] {failure}", flush=True)

    if is_smoke and not failures and len(rows) == 4:
        smoke = read_json(research_data / "smoke_summary.json")
        smoke["audit_passed"] = all(
            row.get("audit", {}).get("embedding_finite")
            and row.get("audit", {}).get("test_metrics_present") is False
            and all(
                m.get("basis_finite")
                and m.get("beta_finite")
                and m.get("gate_finite")
                and m.get("response_finite")
                for m in row["audit"]["modalities"].values()
            )
            for row in rows
        )
        smoke["checkpoint_count"] = sum(Path(ROOT / row["checkpoint_path"]).is_file() for row in rows)
        smoke["gram_audits"] = [
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "variant": row["label"],
                "modalities": {
                    modality: {
                        key: values[key]
                        for key in (
                            "gram_diag_mean",
                            "gram_diag_min",
                            "gram_diag_max",
                            "gram_offdiag_abs_max",
                            "gram_condition_number",
                        )
                    }
                    for modality, values in row["audit"]["modalities"].items()
                },
            }
            for row in rows
        ]
        smoke["breakdown_statistics"] = [
            {
                "variant": row["label"],
                "modality": modality,
                "by_order": values["breakdown_fraction_by_order"],
            }
            for row in rows
            for modality, values in row["audit"]["modalities"].items()
        ]
        smoke["parameter_counts"] = [
            {
                "variant": row["label"],
                "model_trainable_params": row["audit"]["parameter_count_model"],
                "classifier_params": row["audit"]["parameter_count_classifier"],
            }
            for row in rows
        ]
        smoke["test_metrics_absent"] = all(not any(k.lower().startswith("test") for k in row["metrics"]) for row in rows)
        write_json(research_data / "smoke_summary.json", smoke)
        if not smoke["audit_passed"] or smoke["checkpoint_count"] != 4 or not smoke["test_metrics_absent"]:
            failures.append({"error": "Smoke audit checks failed", "oom": False, "nonfinite": False})

    if not is_smoke and len(rows) == expected and not any(
        not item.get("resolved", False) for item in failures
    ):
        write_campaign_state(
            research_data,
            provenance,
            rows,
            failures,
            mode=mode,
            device=args.device,
            epochs=epochs,
            expected=expected,
        )
    print(
        f"[SOSB-V1.5 {mode}] completed={len(rows)}/{expected} failed={len(failures)}; "
        f"outputs={output_root / mode}",
        flush=True,
    )
    if any(not item.get("resolved", False) for item in failures) or len(rows) != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
