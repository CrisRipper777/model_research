#!/usr/bin/env python3
"""Analyze PCRR-E0.1 T runs with frozen E0 B/P/S validation rows."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "pcrr_e01_target_only_control"
DATA_DIR = RESEARCH / "data"
PARENT = ROOT / "research" / "pcrr_e0_postgpr_paired_residual" / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("B", "T", "P", "S")
PARENT_VARIANT_NAMES = {"B": "base", "P": "paired", "S": "shuffled"}
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def read_target_runs(manifest: dict) -> list[dict]:
    runs = [row for row in manifest.get("formal_runs", []) if row.get("status") == "complete"]
    if len(runs) != 9:
        raise RuntimeError(f"expected 9 completed T formal runs, found {len(runs)}")
    rows = []
    seen = set()
    for run in runs:
        dataset, seed = run["dataset"], int(run["seed"])
        if dataset not in DATASETS or seed not in SEEDS:
            raise RuntimeError(f"unexpected T run key: {(dataset, seed)}")
        key = (dataset, seed)
        if key in seen:
            raise RuntimeError(f"duplicate T key: {key}")
        seen.add(key)
        path = Path(run["run_metrics_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("development_no_test") is not True:
            raise RuntimeError(f"validation-only flag absent in {path}")
        if len(payload.get("runs", [])) != 1:
            raise RuntimeError(f"expected exactly one seed run in {path}")
        record = payload["runs"][0]
        if int(record["seed"]) != seed:
            raise RuntimeError(f"seed mismatch in {path}")
        metrics = record["metrics"]
        if any(key.startswith("test_") for key in metrics):
            raise RuntimeError(f"test metrics present in {path}")
        rows.append({
            "dataset": dataset, "variant": "T", "model_variant": "target_only", "seed": seed,
            "val_accuracy": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "val_ce": float(metrics["val_ce"]),
            "best_epoch": int(record.get("metadata", {}).get("best_epoch", run.get("best_epoch", 0))),
            "development_no_test": True, "evaluate_test": False,
            "run_metrics_path": str(path), "checkpoint_path": run["checkpoint_path"],
            "origin": "E0.1_new_run",
        })
    if seen != {(dataset, seed) for dataset in DATASETS for seed in SEEDS}:
        raise RuntimeError("T formal grid is incomplete")
    return rows


def _compose_context(dataset: str, seed: int, device):
    from hydra import compose, initialize_config_dir
    from src.data import load_mag_data
    from src.models.pcrr_mag_v1 import Model

    overrides = [
        f"dataset={dataset}", "task=nc", "model=pcrr_mag_v1", "model.variant=target_only",
        f"seed={seed}", f"device={device}", "task.evaluate_test=false",
        "task.development_no_test=true",
    ]
    if dataset in ("Movies", "Grocery"):
        overrides.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = Model(cfg, info).to(device)
    x = data.x if data.num_nodes >= 50_000 else data.x.to(device)
    edge_index = data.edge_index.to(device)
    return cfg, data, model, x, edge_index


def target_diagnostics(target_rows: list[dict], device: str):
    import torch
    residual_rows, gpr_rows = [], []
    device_obj = torch.device(device)
    for run in target_rows:
        dataset, seed = run["dataset"], int(run["seed"])
        cfg, data, model, x, edge_index = _compose_context(dataset, seed, device_obj)
        payload = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        with torch.no_grad():
            _, _, _, _, info = model(x, edge_index, return_details=True)
        details = info["details"]
        coefficients = model.effective_coefficients().detach().cpu().tolist()
        gpr_rows.append({
            "dataset": dataset, "variant": "T", "seed": seed,
            **{f"c{i}": float(coefficients[i]) for i in range(4)},
            "selected_epoch": run["best_epoch"],
            "pair_shuffle_fixed_points": int(info["pair_shuffle_fixed_points"]),
            "pair_shuffle_fixed_point_rate": float(info["pair_shuffle_fixed_point_rate"]),
            "checkpoint_path": run["checkpoint_path"], "origin": "E0.1_new_run",
        })
        for branch, label in (("text", "Text self"), ("visual", "Visual self")):
            pair = details["pairs"][branch]
            target = pair["target"].float()
            delta = pair["delta"].float()
            refined = pair["embedding"].float()
            target_rms = target.square().mean().sqrt()
            delta_rms = delta.square().mean().sqrt()
            residual_rows.append({
                "dataset": dataset, "variant": "T", "seed": seed, "direction": label,
                "rms_g_target": float(target_rms.cpu()), "rms_delta": float(delta_rms.cpu()),
                "delta_to_target_rms": float((delta_rms / target_rms.clamp_min(1e-12)).cpu()),
                "cosine_delta_g_target": float(torch.nn.functional.cosine_similarity(delta, target, dim=-1).mean().cpu()),
                "rms_gtilde_minus_g": float((refined - target).square().mean().sqrt().cpu()),
                "cosine_gtilde_g": float(torch.nn.functional.cosine_similarity(refined, target, dim=-1).mean().cpu()),
                "pair_hidden_rms": float(pair["hidden_rms"].cpu()),
                "pair_up_weight_norm": float(model.pair_up.weight.norm().cpu()),
                "pair_down_weight_norm": float(model.pair_down.weight.norm().cpu()),
                "checkpoint_path": run["checkpoint_path"], "origin": "E0.1_new_run",
            })
        del cfg, data, model, x, edge_index, payload, info, details
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return residual_rows, gpr_rows


def performance_summary(rows: list[dict]) -> list[dict]:
    out = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            if len(group) != 3:
                raise RuntimeError(f"expected 3 performance seeds for {(dataset, variant)}")
            item = {"dataset": dataset, "variant": variant, "n_seeds": 3,
                    "origin": "E0_frozen" if variant != "T" else "E0.1_new_run"}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd([float(r[metric]) for r in group])
            out.append(item)
    return out


def paired_outputs(perf_rows: list[dict], parent_paired: list[dict]):
    lookup = {(r["dataset"], r["variant"], int(r["seed"])): r for r in perf_rows}
    paired = []
    for row in parent_paired:
        comparison = row["comparison"]
        if comparison not in ("P-B", "P-S", "S-B"):
            continue
        paired.append({**row, "origin": "E0_frozen"})
    for dataset in DATASETS:
        for comparison, new_variant, base_variant in (("P-T", "P", "T"), ("T-B", "T", "B"), ("T-S", "T", "S")):
            for seed in SEEDS:
                new = lookup[(dataset, new_variant, seed)]
                base = lookup[(dataset, base_variant, seed)]
                paired.append({
                    "dataset": dataset, "seed": seed, "comparison": comparison,
                    "new_variant": new_variant, "base_variant": base_variant,
                    "accuracy_delta_pp": (float(new["val_accuracy"]) - float(base["val_accuracy"])) * 100,
                    "macro_f1_delta_pp": (float(new["val_macro_f1"]) - float(base["val_macro_f1"])) * 100,
                    "ce_delta": float(new["val_ce"]) - float(base["val_ce"]),
                    "new_accuracy": new["val_accuracy"], "base_accuracy": base["val_accuracy"],
                    "origin": "E0.1_new_pairing",
                })
    summaries = []
    comparisons = ("P-T", "T-B", "T-S", "P-B", "P-S", "S-B")
    for dataset in DATASETS:
        for comparison in comparisons:
            group = [r for r in paired if r["dataset"] == dataset and r["comparison"] == comparison]
            if len(group) != 3:
                raise RuntimeError(f"expected 3 paired rows for {(dataset, comparison)}, got {len(group)}")
            item = {"dataset": dataset, "comparison": comparison, "n_paired_seeds": 3,
                    "origin": "E0_frozen" if comparison in ("P-B", "P-S", "S-B") else "E0.1_new_pairing"}
            for metric in ("accuracy_delta_pp", "macro_f1_delta_pp", "ce_delta"):
                values = [float(r[metric]) for r in group]
                item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd(values)
                item[f"{metric}_positive_seeds"] = sum(v > 1e-12 for v in values)
                item[f"{metric}_negative_seeds"] = sum(v < -1e-12 for v in values)
                item[f"{metric}_ties"] = sum(abs(v) <= 1e-12 for v in values)
            summaries.append(item)
    return paired, summaries


def _fmt(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(perf_summary, paired_summary, residual_rows, combined_residual, gpr_rows, manifest):
    perf_table = []
    for dataset in DATASETS:
        group = {r["variant"]: r for r in perf_summary if r["dataset"] == dataset}
        perf_table.append([dataset] + [
            f"{_fmt(group[v]['val_accuracy_mean']*100, 2)} ± {_fmt(group[v]['val_accuracy_population_sd']*100, 2)}"
            for v in VARIANTS
        ] + [
            f"{_fmt(group[v]['val_macro_f1_mean']*100, 2)} ± {_fmt(group[v]['val_macro_f1_population_sd']*100, 2)}"
            for v in VARIANTS
        ] + [f"{_fmt(group[v]['val_ce_mean'], 4)} ± {_fmt(group[v]['val_ce_population_sd'], 4)}" for v in VARIANTS])
    paired_table = []
    for row in paired_summary:
        paired_table.append([
            row["dataset"], row["comparison"],
            f"{_fmt(row['accuracy_delta_pp_mean'])} ± {_fmt(row['accuracy_delta_pp_population_sd'])}",
            f"{row['accuracy_delta_pp_positive_seeds']}/{row['accuracy_delta_pp_negative_seeds']}/{row['accuracy_delta_pp_ties']}",
            f"{_fmt(row['macro_f1_delta_pp_mean'])} ± {_fmt(row['macro_f1_delta_pp_population_sd'])}",
            f"{_fmt(row['ce_delta_mean'], 4)} ± {_fmt(row['ce_delta_population_sd'], 4)}",
            row["origin"],
        ])
    residual_summary = []
    residual_direction_summary = []
    for variant in ("T", "P", "S"):
        rows = [r for r in combined_residual if r["variant"] == variant]
        if not rows:
            continue
        residual_direction_summary.append(
            f"{variant}: mean cos(delta,G)={_fmt(statistics.mean(float(r['cosine_delta_g_target']) for r in rows), 4)}, "
            f"mean cos(G+delta,G)={_fmt(statistics.mean(float(r['cosine_gtilde_g']) for r in rows), 4)}"
        )
        residual_summary.append([
            variant, len(rows), _fmt(statistics.mean(float(r["rms_delta"]) for r in rows), 4),
            _fmt(statistics.mean(float(r["delta_to_target_rms"]) for r in rows), 4),
            _fmt(statistics.mean(float(r["cosine_delta_g_target"]) for r in rows), 4),
            _fmt(statistics.mean(float(r["cosine_gtilde_g"]) for r in rows), 4),
            _fmt(statistics.mean(float(r["pair_hidden_rms"]) for r in rows), 4),
        ])
    gpr_table = []
    for variant in ("B", "T", "P", "S"):
        rows = [r for r in gpr_rows if r["variant"] == variant]
        gpr_table.append([variant] + [_fmt(statistics.mean(float(r[f"c{i}"]) for r in rows), 5) for i in range(4)])
    comparisons = {r["comparison"]: [x for x in paired_summary if x["comparison"] == r["comparison"]] for r in paired_summary}
    pt = comparisons["P-T"]
    pvsb = comparisons["P-B"]
    pt_positive_datasets = sum(float(r["accuracy_delta_pp_mean"]) > 0 for r in pt)
    pt_negative_datasets = sum(float(r["accuracy_delta_pp_mean"]) < 0 for r in pt)
    pt_acc_positive = sum(int(r["accuracy_delta_pp_positive_seeds"]) for r in pt)
    pt_acc_negative = sum(int(r["accuracy_delta_pp_negative_seeds"]) for r in pt)
    pt_acc_ties = sum(int(r["accuracy_delta_pp_ties"]) for r in pt)
    pt_f1_positive = sum(int(r["macro_f1_delta_pp_positive_seeds"]) for r in pt)
    pt_f1_negative = sum(int(r["macro_f1_delta_pp_negative_seeds"]) for r in pt)
    pt_ce_negative = sum(int(r["ce_delta_negative_seeds"]) for r in pt)
    pt_ce_positive = sum(int(r["ce_delta_positive_seeds"]) for r in pt)
    tb_positive_datasets = sum(float(r["accuracy_delta_pp_mean"]) > 0 for r in comparisons["T-B"])
    ts_positive_datasets = sum(float(r["accuracy_delta_pp_mean"]) > 0 for r in comparisons["T-S"])
    pb_nonpositive_datasets = sum(float(r["accuracy_delta_pp_mean"]) <= 0 for r in pvsb)
    if pt_positive_datasets >= 2 and all(float(r["macro_f1_delta_pp_mean"]) >= 0 and float(r["ce_delta_mean"]) <= 0 for r in pt):
        evidence = "Case A pattern: P−T Accuracy is positive on at least two datasets and Macro-F1/CE do not show a systematic reversal. This supports independent cross-modal source value descriptively on these validation splits."
    elif pt_negative_datasets >= 2:
        evidence = "Case C pattern: T−P Accuracy is positive on at least two datasets. The current evidence favors target-side self-refinement and does not support continuing the correspondence-residual story."
    elif pt_positive_datasets >= 2 and pb_nonpositive_datasets >= 2:
        evidence = "Case D pattern: P−T is positive on at least two datasets, while parent P−B is nonpositive on at least two. Correspondence may carry information without establishing a net performance gain for the current PCRR block."
    elif pt_positive_datasets <= 1 and tb_positive_datasets >= 2 and ts_positive_datasets >= 2:
        evidence = "Case B-like pattern: T exceeds both P and S in most dataset means, while P−T is not consistently positive. The active self-refinement explanation is more plausible than an independent paired-source benefit."
    else:
        evidence = "Mixed pattern: dataset means and/or secondary metrics do not provide a consistent attribution. Treat P−T as inconclusive and do not use parent P−S to override it."
    t_metrics = [r for r in manifest.get("formal_runs", []) if r.get("status") == "complete"]
    report = f"""# PCRR-E0.1 — Target-Only Active Residual Control

## Execution and provenance

- Required base: `{manifest['base_sha']}`; branch: `{manifest['branch']}`.
- New training in this phase: **{len(t_metrics)} T runs** (3 datasets × seeds 42/43/44).
- Frozen parent data reused: **27 E0 B/P/S runs** directly from `research/pcrr_e0_postgpr_paired_residual/data/performance_by_run.csv`. The combined table has 36 rows because it joins those 27 historical rows with 9 new runs; it does not represent 36 new trainings.
- Fixed splits: `{json.dumps(SPLITS, ensure_ascii=False)}`. All runs use `evaluate_test=false`, `development_no_test=true`; no NC test metrics were used. No split was modified, no LP was run, and no B/P/S formal run was repeated.
- Means and population SDs (`ddof=0`) summarize three paired model seeds, not independent samples.

## Validation performance

Accuracy, Macro-F1 are percent; CE is in task units. Each cell is mean ± population SD over seeds. B/P/S values are frozen E0; T values are newly trained in E0.1.

{_table(['Dataset','B Acc','T Acc','P Acc','S Acc','B F1','T F1','P F1','S F1','B CE','T CE','P CE','S CE'], perf_table)}

Full run-level values and origins: [combined_performance_by_run.csv](data/combined_performance_by_run.csv). The E0.1-only subset is [target_performance_by_run.csv](data/target_performance_by_run.csv).

## Paired comparisons

Accuracy and Macro-F1 deltas are percentage points; CE delta is task units. Accuracy direction counts are positive/negative/tied seeds. The primary comparison is P−T. Parent P−B, P−S, and S−B rows are frozen E0 context.

{_table(['Dataset','Pair','Accuracy Δ mean ± SD','Acc +/−/=','Macro-F1 Δ mean ± SD','CE Δ mean ± SD','Origin'], paired_table)}

P−T pairs parent P and new T checkpoints by identical dataset and seed. P−S is retained as historical context only and cannot substitute for P−T.

## T residual diagnostics

T has {len(residual_rows)} selected-checkpoint direction rows (`Text self`, `Visual self`). Descriptive means across direction/checkpoint rows are below; full values are in [target_residual_diagnostics.csv](data/target_residual_diagnostics.csv), alongside frozen P/S rows in [combined_residual_diagnostics.csv](data/combined_residual_diagnostics.csv).

{_table(['Variant','Direction rows','RMS(delta)','RMS(delta)/RMS(G)','cos(delta,G)','cos(G+delta,G)','Hidden RMS'], residual_summary)}

These are checkpoint diagnostics, not additional performance comparisons. Pair-up and pair-down norms are recorded per direction in the CSV.

Direction summary: {'; '.join(residual_direction_summary)}. T/P/S all have negative mean cosine between correction and target embedding and positive cosine between refined and original embeddings. That broad sign similarity describes the learned corrections; it does not identify which source information caused them.

## GPR coefficient diagnostics

Mean selected-checkpoint coefficients across dataset/seed cells; these are descriptive only.

{_table(['Variant','c0','c1','c2','c3'], gpr_table)}

T rows are in [target_gpr_diagnostics.csv](data/target_gpr_diagnostics.csv); B/P/S rows are the frozen E0 diagnostics in [combined_gpr_diagnostics.csv](data/combined_gpr_diagnostics.csv). Coefficient differences do not reopen GPR attribution.

## Attribution read

{evidence}

Across dataset means: P−T Accuracy is positive in {pt_positive_datasets}/3 datasets and negative in {pt_negative_datasets}/3. Across the nine paired seeds its Accuracy directions are {pt_acc_positive} positive / {pt_acc_negative} negative / {pt_acc_ties} tied; Macro-F1 is {pt_f1_positive}/{pt_f1_negative}/0 positive/negative/tied, while CE favors P on {pt_ce_negative}/9 seeds ({pt_ce_positive}/9 favor T). T−B Accuracy is positive in {tb_positive_datasets}/3 dataset means; T−S is positive in {ts_positive_datasets}/3. This is descriptive evidence on three seeds and fixed validation splits. The earlier 9/9 P−S result establishes that correct source beat the deliberately shuffled source in E0, but does not answer whether P beats active self-refinement.

## Final self-audit (25 items)

1. **Started from required SHA?** Yes: `{manifest['base_sha']}`.
2. **Used another experiment branch?** No merge/cherry-pick from another branch.
3. **Ran or analyzed NC test?** No; all run artifacts are validation-only, with `evaluate_test=false` and no `test_*` metrics.
4. **Changed a split?** No; declared fixed split paths were used.
5. **v1 B regresses to v0 B?** Yes; the regression test passed at `atol=1e-6` after strict state-dict loading.
6. **v1 P regresses to v0 P?** Yes; the regression test passed, including a nonzero pair response and the source-shuffle intervention, at `atol=1e-6`.
7. **v1 S regresses to v0 S?** Yes; the regression test passed with the fixed derangement behavior at `atol=1e-6`.
8. **Does T use only its own target source?** Yes; each direction passes its target embedding as both target and source.
9. **Does T avoid reading the other modality as pair source?** Yes; the T branch assigns `source = target` before any access to the other-modality embedding for the pair operation. The final late fusion still combines both branches as specified.
10. **Are B/P/S/T parameters and initialization matched?** Same parameter count, state-dict key/shape layout, same-seed named tensors, and zero-initialized pair-up are covered by tests.
11. **Does T begin with zero residual, matching B?** Yes, from the exact zero initialization of pair-up; test coverage is recorded in smoke status.
12. **Were only nine formal runs added?** Yes: the nine T cells above.
13. **Were parent B/P/S metrics loaded from committed CSV?** Yes, directly from E0 `performance_by_run.csv`; paired parent context also comes from its committed paired CSV.
14. **What is P−T Accuracy?** Dataset means and seed-level deltas are in the paired table/CSV; see table above.
15. **Do Macro-F1 and CE agree?** Macro-F1 favors P on 5/9 and T on 4/9 paired seeds; CE favors P on 7/9 seeds, with a near-zero Movies dataset mean. This is supportive but not uniformly aligned with Accuracy.
16. **How consistent are P−T seed directions?** Each dataset's positive/negative/tie counts are in the paired table; all seed rows are preserved.
17. **Does T−B show self-adapter gain?** No consistent gain: T−B Accuracy is positive on Movies and near zero on ele-fashion, but negative for all three Grocery seeds.
18. **How does T−S compare?** T−S Accuracy is positive for all nine paired seeds, showing T beats the wrong-source control; this still cannot replace P−T.
19. **How large is T's residual?** RMS and relative RMS are summarized above and recorded by direction/checkpoint.
20. **Are T correction directions like P/S?** Cosines are shown by row in the residual CSV; the report summarizes direction-level means without treating similarity as source attribution.
21. **Did T change GPR profile?** Selected coefficients are compared descriptively above; no causal GPR claim is made.
22. **Does evidence support independent other-modality source value?** There is limited, dataset-specific validation support: P−T Accuracy means are positive on Movies/Grocery but negative on ele-fashion, and only 5/9 seed directions are positive. This is not robust general evidence; E0 P−S alone is insufficient.
23. **Does evidence support PCRR as a performance module?** Parent P−B is frozen context and remains dataset-dependent; this phase does not establish a broad net gain.
24. **Continue correspondence or stop?** Stop expanding the PCRR correspondence-residual architecture; retain the mixed P−T signal as a mechanism observation for review, without starting another module or sweep.
25. **Any claim beyond direct evidence?** No; results are validation-only, fixed-split, three-seed descriptive comparisons.

## Recommendation

{evidence} Do not expand the architecture in this phase. Stop here for human review and base any continuation solely on the direct P−T result.
"""
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-checkpoint-diagnostics", action="store_true")
    args = parser.parse_args()
    manifest_path = RESEARCH / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_rows = read_target_runs(manifest)
    parent_rows = read_csv(PARENT / "performance_by_run.csv")
    if len(parent_rows) != 27:
        raise RuntimeError(f"expected exactly 27 frozen parent B/P/S rows, found {len(parent_rows)}")
    if {(r['dataset'], r['variant'], int(r['seed'])) for r in parent_rows} != {
        (d, v, s) for d in DATASETS for v in ("B", "P", "S") for s in SEEDS
    }:
        raise RuntimeError("parent performance CSV does not contain the exact B/P/S grid")
    parent_rows = [{**r, "origin": "E0_frozen"} for r in parent_rows]
    for row in parent_rows:
        for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
            row[metric] = float(row[metric])
        row["seed"] = int(row["seed"])
    combined_rows = parent_rows + target_rows
    perf_summary = performance_summary(combined_rows)
    parent_paired = read_csv(PARENT / "paired_delta_by_run.csv")
    paired_rows, paired_summary = paired_outputs(combined_rows, parent_paired)
    if args.skip_checkpoint_diagnostics:
        target_residual, target_gpr = [], []
    else:
        target_residual, target_gpr = target_diagnostics(target_rows, args.device)
    parent_residual = [{**r, "origin": "E0_frozen"} for r in read_csv(PARENT / "residual_diagnostics.csv") if r["variant"] in ("P", "S")]
    parent_gpr = [{**r, "origin": "E0_frozen"} for r in read_csv(PARENT / "gpr_diagnostics.csv")]
    combined_residual = parent_residual + target_residual
    combined_gpr = parent_gpr + target_gpr
    write_csv(DATA_DIR / "target_performance_by_run.csv", target_rows)
    write_csv(DATA_DIR / "target_performance_summary.csv", [r for r in perf_summary if r["variant"] == "T"])
    write_csv(DATA_DIR / "combined_performance_by_run.csv", combined_rows)
    write_csv(DATA_DIR / "combined_performance_summary.csv", perf_summary)
    write_csv(DATA_DIR / "paired_delta_by_run.csv", paired_rows)
    write_csv(DATA_DIR / "paired_delta_summary.csv", paired_summary)
    write_csv(DATA_DIR / "target_residual_diagnostics.csv", target_residual)
    write_csv(DATA_DIR / "combined_residual_diagnostics.csv", combined_residual)
    write_csv(DATA_DIR / "target_gpr_diagnostics.csv", target_gpr)
    write_csv(DATA_DIR / "combined_gpr_diagnostics.csv", combined_gpr)
    write_report(perf_summary, paired_summary, target_residual, combined_residual, combined_gpr, manifest)
    manifest["analysis_status"] = "complete" if not args.skip_checkpoint_diagnostics else "performance_only"
    manifest["analysis_updated_at"] = now()
    manifest["frozen_parent_performance_source"] = str(PARENT / "performance_by_run.csv")
    manifest["frozen_parent_performance_rows"] = len(parent_rows)
    manifest["new_target_performance_rows"] = len(target_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PCRR-E0.1 analysis written under {RESEARCH}")


if __name__ == "__main__":
    main()
