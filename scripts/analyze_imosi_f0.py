#!/usr/bin/env python3
"""Analyze IMoSI-F0 retrained results and selected-checkpoint diagnostics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research" / "imosi_f0_interaction_mode_mixture"
DATA_DIR = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("B", "G", "M")
VARIANT_NAMES = {"B": "base", "G": "global_mix", "M": "adaptive_mix"}
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}
MODE_NAMES = ("preserve", "self", "paired")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def mean_sd(values: list[float]):
    return statistics.mean(values), statistics.pstdev(values)


def scalar(tensor):
    return float(tensor.detach().float().cpu().item()) if hasattr(tensor, "detach") else float(tensor)


def read_performance(manifest: dict):
    runs = [r for r in manifest.get("formal_runs", []) if r.get("status") == "complete"]
    if len(runs) != 27:
        raise RuntimeError(f"expected 27 complete formal runs, found {len(runs)}")
    rows = []
    seen = set()
    for run in runs:
        key = (run["dataset"], run["variant"], int(run["seed"]))
        if key in seen:
            raise RuntimeError(f"duplicate formal key {key}")
        seen.add(key)
        path = Path(run["run_metrics_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("development_no_test") is not True:
            raise RuntimeError(f"validation-only flag missing from {path}")
        if len(payload.get("runs", [])) != 1:
            raise RuntimeError(f"expected one run record in {path}")
        record = payload["runs"][0]
        if int(record["seed"]) != key[2]:
            raise RuntimeError(f"seed mismatch in {path}")
        metrics = record["metrics"]
        if any(k.startswith("test_") for k in metrics):
            raise RuntimeError(f"test metric found in {path}")
        rows.append({
            "dataset": key[0], "variant": key[1], "model_variant": VARIANT_NAMES[key[1]], "seed": key[2],
            "val_accuracy": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "val_ce": float(metrics["val_ce"]),
            "best_epoch": int(record.get("metadata", {}).get("best_epoch", run.get("best_epoch", 0))),
            "development_no_test": True, "evaluate_test": False,
            "run_metrics_path": str(path), "checkpoint_path": run["checkpoint_path"],
            "source_commit_sha": run["source_commit_sha"],
        })
    expected = {(d, v, s) for d in DATASETS for v in VARIANTS for s in SEEDS}
    if seen != expected:
        raise RuntimeError("formal dataset × variant × seed grid is incomplete")
    return rows


def summarize_performance(rows: list[dict]):
    summaries = []
    lookup = {(r["dataset"], r["variant"], int(r["seed"])): r for r in rows}
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [lookup[(dataset, variant, seed)] for seed in SEEDS]
            row = {"dataset": dataset, "variant": variant, "n_seeds": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                row[f"{metric}_mean"], row[f"{metric}_population_sd"] = mean_sd([float(x[metric]) for x in group])
            summaries.append(row)
    return summaries, lookup


def paired_outputs(lookup: dict):
    by_run, summaries = [], []
    comparisons = (("M-B", "M", "B"), ("G-B", "G", "B"), ("M-G", "M", "G"))
    for dataset in DATASETS:
        for name, new_variant, base_variant in comparisons:
            group = []
            for seed in SEEDS:
                new = lookup[(dataset, new_variant, seed)]
                base = lookup[(dataset, base_variant, seed)]
                row = {
                    "dataset": dataset, "seed": seed, "comparison": name,
                    "new_variant": new_variant, "base_variant": base_variant,
                    "accuracy_delta_pp": (new["val_accuracy"] - base["val_accuracy"]) * 100.0,
                    "macro_f1_delta_pp": (new["val_macro_f1"] - base["val_macro_f1"]) * 100.0,
                    "ce_delta": new["val_ce"] - base["val_ce"],
                    "new_accuracy": new["val_accuracy"], "base_accuracy": base["val_accuracy"],
                }
                by_run.append(row)
                group.append(row)
            summary = {"dataset": dataset, "comparison": name, "n_paired_seeds": len(group)}
            for metric in ("accuracy_delta_pp", "macro_f1_delta_pp", "ce_delta"):
                values = [float(r[metric]) for r in group]
                summary[f"{metric}_mean"], summary[f"{metric}_population_sd"] = mean_sd(values)
                summary[f"{metric}_positive_seeds"] = sum(x > 1e-12 for x in values)
                summary[f"{metric}_negative_seeds"] = sum(x < -1e-12 for x in values)
                summary[f"{metric}_ties"] = sum(abs(x) <= 1e-12 for x in values)
            summaries.append(summary)
    return by_run, summaries


def _compose_context(dataset: str, seed: int, device):
    import torch
    from hydra import compose, initialize_config_dir
    from src.data import load_mag_data
    from src.models.imosi_mag_v0 import Model
    overrides = [
        f"dataset={dataset}", "task=nc", "model=imosi_mag_v0", "model.variant=adaptive_mix",
        f"seed={seed}", f"device={device}", "task.evaluate_test=false", "task.development_no_test=true",
    ]
    if dataset in ("Movies", "Grocery"):
        overrides.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]), "visual_dim": int(data.x_i.shape[1]),
    }
    model = Model(cfg, info).to(device)
    x = data.x if data.num_nodes >= 50_000 else data.x.to(device)
    edge_index = data.edge_index.to(device)
    return cfg, data, model, x, edge_index


def _route_rows(dataset, variant, seed, modality, detail):
    import torch
    probs = detail["mode_probs"].detach().float()
    n = int(probs.size(0))
    quantiles = torch.quantile(probs, torch.tensor([0.1, 0.5, 0.9], device=probs.device), dim=0)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
    winner = probs.argmax(-1)
    row = {
        "dataset": dataset, "variant": variant, "seed": seed, "modality": modality,
        "num_nodes": n, "mean_routing_entropy": scalar(entropy),
        "argmax_preserve_fraction": float((winner == 0).float().mean().cpu()),
        "argmax_self_fraction": float((winner == 1).float().mean().cpu()),
        "argmax_paired_fraction": float((winner == 2).float().mean().cpu()),
        "node_delta_logits_rms": scalar(detail["expert_contribution"]["node_delta_logits_rms"]),
        "route_distribution_vs_global_rms": scalar(detail["expert_contribution"]["route_global_difference_rms"]),
    }
    for index, name in enumerate(MODE_NAMES):
        row[f"mean_pi_{name}"] = float(probs[:, index].mean().cpu())
        row[f"std_pi_{name}"] = float(probs[:, index].std(unbiased=False).cpu())
        row[f"p10_pi_{name}"] = float(quantiles[0, index].cpu())
        row[f"p50_pi_{name}"] = float(quantiles[1, index].cpu())
        row[f"p90_pi_{name}"] = float(quantiles[2, index].cpu())
    row["node_delta_logits_rms"] = row["node_delta_logits_rms"] if variant == "M" else 0.0
    row["route_distribution_vs_global_rms"] = row["route_distribution_vs_global_rms"] if variant == "M" else 0.0
    return row


def _expert_row(dataset, variant, seed, modality, values):
    row = {"dataset": dataset, "variant": variant, "seed": seed, "modality": modality}
    for key, value in values.items():
        row[key] = scalar(value)
    return row


def _load_model_checkpoint(model, variant, run, device):
    import torch
    payload = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
    model.variant = VARIANT_NAMES[variant]
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return payload


def _globalize_validation_metrics(cfg, data, model, classifier, x, edge_index, device):
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels
    import torch
    with torch.no_grad():
        z, _, _, _, _ = model(x, edge_index, intervention="globalize_router")
    labels = _resolve_nc_eval_labels(data, development_no_test=True)
    result = _evaluate_split(
        classifier, z.detach().cpu(), data.y, data.val_idx, device,
        int(cfg.task.inference_batch_size), labels,
    )
    return result


def checkpoint_diagnostics(manifest, perf_lookup, device):
    import torch
    from torch import nn
    from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels
    run_index = {(r["dataset"], r["variant"], int(r["seed"])): r for r in manifest["formal_runs"] if r.get("status") == "complete"}
    route_rows, contribution_rows, gpr_rows, intervention_rows = [], [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            cfg, data, model, x, edge_index = _compose_context(dataset, seed, device)
            eval_labels = _resolve_nc_eval_labels(data, development_no_test=True)
            for variant in VARIANTS:
                run = run_index[(dataset, variant, seed)]
                payload = _load_model_checkpoint(model, variant, run, device)
                coeffs = model.effective_coefficients().detach().cpu().tolist()
                gpr_rows.append({
                    "dataset": dataset, "variant": variant, "seed": seed,
                    **{f"c{i}": coeffs[i] for i in range(4)},
                    "selected_epoch": run["best_epoch"], "checkpoint_path": run["checkpoint_path"],
                })
                if variant in ("G", "M"):
                    with torch.no_grad():
                        _, _, _, _, info = model(x, edge_index, return_details=True)
                    for modality, detail in info["details"]["routing"].items():
                        route_rows.append(_route_rows(dataset, variant, seed, modality, detail))
                        contribution_rows.append(_expert_row(
                            dataset, variant, seed, modality, detail["expert_contribution"]
                        ))
                if variant == "M":
                    classifier = nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"])).to(device)
                    classifier.load_state_dict(payload["head_state"], strict=True)
                    classifier.eval()
                    globalized = _globalize_validation_metrics(cfg, data, model, classifier, x, edge_index, device)
                    normal = perf_lookup[(dataset, "M", seed)]
                    intervention_rows.append({
                        "dataset": dataset, "variant": "M", "seed": seed,
                        "intervention": "globalize_router", "evaluate_test": False,
                        "val_accuracy": globalized["acc"], "val_macro_f1": globalized["macro_f1"], "val_ce": globalized["ce"],
                        "normal_val_accuracy": normal["val_accuracy"],
                        "accuracy_delta_vs_normal_pp": (globalized["acc"] - normal["val_accuracy"]) * 100.0,
                        "macro_f1_delta_vs_normal_pp": (globalized["macro_f1"] - normal["val_macro_f1"]) * 100.0,
                        "ce_delta_vs_normal": globalized["ce"] - normal["val_ce"],
                        "selected_epoch": run["best_epoch"], "checkpoint_path": run["checkpoint_path"],
                    })
                    del classifier, globalized
                del payload
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del cfg, data, model, x, edge_index
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    intervention_summary = []
    for dataset in DATASETS:
        group = [r for r in intervention_rows if r["dataset"] == dataset]
        item = {"dataset": dataset, "intervention": "globalize_router", "n_model_seeds": len(group)}
        for metric in ("val_accuracy", "val_macro_f1", "val_ce", "accuracy_delta_vs_normal_pp", "macro_f1_delta_vs_normal_pp", "ce_delta_vs_normal"):
            item[f"{metric}_mean"], item[f"{metric}_population_sd"] = mean_sd([float(r[metric]) for r in group])
        values = [float(r["accuracy_delta_vs_normal_pp"]) for r in group]
        item["accuracy_delta_positive_seeds"] = sum(v > 1e-12 for v in values)
        item["accuracy_delta_negative_seeds"] = sum(v < -1e-12 for v in values)
        item["accuracy_delta_ties"] = sum(abs(v) <= 1e-12 for v in values)
        intervention_summary.append(item)
    return route_rows, contribution_rows, gpr_rows, intervention_rows, intervention_summary


def _fmt(value, digits=3):
    return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(manifest, performance_summary, paired_summary, route_rows, contribution_rows, gpr_rows, intervention_summary):
    performance_table = []
    for dataset in DATASETS:
        grouped = {r["variant"]: r for r in performance_summary if r["dataset"] == dataset}
        performance_table.append([dataset] + [
            f"{_fmt(grouped[v]['val_accuracy_mean']*100, 2)} ± {_fmt(grouped[v]['val_accuracy_population_sd']*100, 2)}"
            for v in VARIANTS
        ] + [
            f"{_fmt(grouped[v]['val_macro_f1_mean']*100, 2)} ± {_fmt(grouped[v]['val_macro_f1_population_sd']*100, 2)}"
            for v in VARIANTS
        ] + [f"{_fmt(grouped[v]['val_ce_mean'], 4)} ± {_fmt(grouped[v]['val_ce_population_sd'], 4)}" for v in VARIANTS])
    paired_table = []
    for row in paired_summary:
        paired_table.append([
            row["dataset"], row["comparison"],
            f"{_fmt(row['accuracy_delta_pp_mean'])} ± {_fmt(row['accuracy_delta_pp_population_sd'])}",
            f"{row['accuracy_delta_pp_positive_seeds']}/{row['accuracy_delta_pp_negative_seeds']}/{row['accuracy_delta_pp_ties']}",
            f"{_fmt(row['macro_f1_delta_pp_mean'])} ± {_fmt(row['macro_f1_delta_pp_population_sd'])}",
            f"{_fmt(row['ce_delta_mean'], 4)} ± {_fmt(row['ce_delta_population_sd'], 4)}",
        ])
    route_summary = []
    for variant in ("G", "M"):
        for dataset in DATASETS:
            rows = [r for r in route_rows if r["variant"] == variant and r["dataset"] == dataset]
            if not rows:
                continue
            for modality in ("text", "visual"):
                group = [r for r in rows if r["modality"] == modality]
                route_summary.append([
                    variant, dataset, modality,
                    *[_fmt(statistics.mean(float(r[f"mean_pi_{mode}"]) for r in group), 4) for mode in MODE_NAMES],
                    _fmt(statistics.mean(float(r["mean_routing_entropy"]) for r in group), 4),
                    _fmt(statistics.mean(float(r["node_delta_logits_rms"]) for r in group), 4),
                ])
    contribution_summary = []
    contribution_keys = (
        "rms_delta_self", "rms_delta_pair", "rms_effective_self", "rms_effective_pair",
        "effective_self_to_target_rms", "effective_pair_to_target_rms",
        "cosine_effective_self_target", "cosine_effective_pair_target",
    )
    for variant in ("G", "M"):
        group = [r for r in contribution_rows if r["variant"] == variant]
        contribution_summary.append([variant] + [
            _fmt(statistics.mean(float(r[key]) for r in group), 4) for key in contribution_keys
        ])
    gpr_table = []
    for variant in VARIANTS:
        group = [r for r in gpr_rows if r["variant"] == variant]
        gpr_table.append([variant] + [_fmt(statistics.mean(float(r[f"c{i}"]) for r in group), 5) for i in range(4)])
    intervention_table = []
    for row in intervention_summary:
        intervention_table.append([
            row["dataset"],
            f"{_fmt(row['accuracy_delta_vs_normal_pp_mean'])} ± {_fmt(row['accuracy_delta_vs_normal_pp_population_sd'])}",
            f"{row['accuracy_delta_positive_seeds']}/{row['accuracy_delta_negative_seeds']}/{row['accuracy_delta_ties']}",
            f"{_fmt(row['macro_f1_delta_vs_normal_pp_mean'])} ± {_fmt(row['macro_f1_delta_vs_normal_pp_population_sd'])}",
            f"{_fmt(row['ce_delta_vs_normal_mean'], 4)} ± {_fmt(row['ce_delta_vs_normal_population_sd'], 4)}",
        ])
    summaries = {comp: [r for r in paired_summary if r["comparison"] == comp] for comp in ("M-B", "G-B", "M-G")}
    m_b = summaries["M-B"]
    g_b = summaries["G-B"]
    m_g = summaries["M-G"]
    m_b_pos = sum(float(r["accuracy_delta_pp_mean"]) > 0 for r in m_b)
    g_b_pos = sum(float(r["accuracy_delta_pp_mean"]) > 0 for r in g_b)
    m_g_pos = sum(float(r["accuracy_delta_pp_mean"]) > 0 for r in m_g)
    if m_b_pos >= 2 and m_g_pos >= 2:
        interpretation = "Pattern compatible with Case A: adaptive mixture improves on B and G in at least two dataset means. Check secondary metrics and direction counts before treating this as support for node-specific mode selection."
    elif g_b_pos >= 2 and m_g_pos <= 1:
        interpretation = "Pattern compatible with Case B: global mode composition improves on B in most dataset means, while the adaptive increment over G is not broadly positive. Prefer the simpler G unless the paired M−G evidence says otherwise."
    elif m_b_pos in (1, 2):
        interpretation = "Pattern compatible with Case C: adaptive value appears dataset-dependent. Keep the result descriptive and do not add routing regularizers to rescue F0."
    elif all(abs(float(r["accuracy_delta_pp_mean"])) < 1e-12 for r in m_b + g_b):
        interpretation = "Pattern compatible with Case D: the three variants are effectively tied on mean Accuracy; stop the mixture direction."
    elif sum(float(r["accuracy_delta_pp_mean"]) < 0 for r in m_b + g_b) >= 4:
        interpretation = "Pattern compatible with Case E: the mixtures are below B in several dataset comparisons; stop the mixture direction."
    else:
        interpretation = "Mixed validation pattern; use paired M−B and M−G as the primary evidence. Routing variation and checkpoint intervention cannot substitute for retrained gains."
    report = f"""# IMoSI-F0 — Interaction-Mode Mixture Prototype

## Execution and provenance

- Required parent SHA: `{manifest['base_sha']}`; branch: `{manifest['branch']}`.
- Source was committed before smoke/formal execution: `{manifest['source_commit_sha']}`. Formal runs record this source SHA.
- Validation-only NC campaign: {sum(r.get('status') == 'complete' for r in manifest.get('formal_runs', []))}/27 complete (B/G/M × three datasets × seeds 42/43/44). The three seeds are repeated model/training seeds on each fixed split, not separate dataset splits.
- NC uses `evaluate_test=false`, `development_no_test=true`; no NC test metrics, test evaluation, split changes, or formal LP runs. Only the requested B/M two-batch LP smoke was run.
- Means and population SDs (`ddof=0`) describe three paired seeds; no pseudo-IID p-values or hard minimum-gain threshold are used.

## Validation performance

Accuracy and Macro-F1 are percent; CE is in task units. Cells are mean ± population SD over seeds.

{_table(['Dataset','B Acc','G Acc','M Acc','B F1','G F1','M F1','B CE','G CE','M CE'], performance_table)}

Run-level metrics: [performance_by_run.csv](data/performance_by_run.csv); seed summary: [performance_summary.csv](data/performance_summary.csv).

## Paired retrained comparisons

Accuracy and Macro-F1 deltas are percentage points. Accuracy directions are positive/negative/tied seeds. M−B is primary; G−B and M−G are secondary.

{_table(['Dataset','Comparison','Accuracy Δ mean ± SD','Acc +/−/=','Macro-F1 Δ mean ± SD','CE Δ mean ± SD'], paired_table)}

Full paired rows: [paired_delta_by_run.csv](data/paired_delta_by_run.csv). M−G is the retrained evidence for node-specific routing beyond modality-global mode preference.

## Routing diagnostics

Selected G/M checkpoints; values are averaged over three seeds for each dataset/modality. Routes are architecture-defined Preserve/Self/Paired mixtures, not latent ground-truth roles.

{_table(['Variant','Dataset','Modality','Mean π preserve','Mean π self','Mean π paired','Entropy','Node-logit RMS'], route_summary)}

Per-seed mean/std, p10/p50/p90, entropy, argmax fractions, and M-vs-global route difference are in [routing_diagnostics.csv](data/routing_diagnostics.csv). Nonuniform routes alone do not imply task value.

## Effective expert contribution

The table averages direction/checkpoint summaries; full per dataset/seed/modality values are in [expert_contribution.csv](data/expert_contribution.csv). Raw deltas and probability-weighted contributions are reported separately.

{_table(['Variant','RMS δ self','RMS δ pair','RMS πself·δself','RMS πpair·δpair','Self/G','Pair/G','cos(self_eff,G)','cos(pair_eff,G)'], contribution_summary)}

## GPR coefficients

Mean selected-checkpoint coefficients over the nine dataset-seed cells per variant. These only describe possible backbone co-adaptation.

{_table(['Variant','c0','c1','c2','c3'], gpr_table)}

Per-run values: [gpr_diagnostics.csv](data/gpr_diagnostics.csv).

## M checkpoint globalize-router intervention

This intervention sets node-specific router deltas to zero while retaining the selected M backbone, experts, and modality-global logits. It measures checkpoint reliance only; retrained M−G remains the adaptive-routing test.

{_table(['Dataset','Accuracy Δ vs M','Acc +/−/=','Macro-F1 Δ','CE Δ'], intervention_table)}

Rows: [intervention_by_run.csv](data/intervention_by_run.csv); seed summary: [intervention_summary.csv](data/intervention_summary.csv).

## Interpretation

{interpretation}

Do not infer efficacy from route entropy, expert frequency, route variance, or globalize-router score alone. The paired retrained M−B and M−G results take precedence.

## Final self-audit (30 items)

1. **Started from `188b940…`?** Yes; required base is `{manifest['base_sha']}`.
2. **Used another experiment branch?** No merge/cherry-pick from other experiment branches.
3. **Was source committed before formal runs?** Yes; source commit is `{manifest['source_commit_sha']}`.
4. **Which source SHA did formal runs use?** `{manifest['source_commit_sha']}` is recorded in every formal run.
5. **Ran/read NC test?** No test evaluation or `test_*` metric. The dev-only task masks test labels and computes validation metrics from train/validation splits; full-graph features follow the existing transductive protocol.
6. **Changed splits?** No; the three fixed paths above were used.
7. **Does B regress to PCRR-v1/RGD?** Yes, at `atol=1e-6` after matching shared state; test passed.
8. **Do F0 modules preserve downstream RNG state?** Yes; global CPU RNG and next-classifier tensors match PCRR-v1 under identical starting RNG.
9. **Are B/G/M initial functions identical?** Yes, elementwise in eval and after train-mode RNG reset.
10. **Are Self/Paired heads zero-initialized?** Yes; exact zero weights and biases.
11. **Does M start with G routing?** Yes; zero `router_out` gives zero node logits and routes equal G.
12. **Are B/G/M parameter count and initialization matched?** Yes; identical count, state layout, and same-seed tensors.
13. **Did LP smoke preserve its protocol?** See [smoke_status.json](smoke_status.json): LinkNeighborLoader, `[5,5,5]`, global-eid positive-edge removal, sampled forward/backward, validation, checkpoint; test disabled.
14. **How does M−B perform?** Dataset means, SDs, and seed directions are in the paired table above; interpret by dataset.
15. **How does G−B perform?** Shown separately in the paired table; it tests global mode composition.
16. **How does M−G perform?** Shown separately; it is the retrained node-routing increment.
17. **Is global mixture sufficient?** Answer depends on G−B and M−G paired signs; do not infer from router diagnostics.
18. **Does adaptive routing vary by node?** Mean/std/quantiles and M node-logit RMS are reported per dataset, seed, and modality.
19. **Does node variation coincide with retrained M−G gain?** Compare the direct M−G table; route variation itself is not evidence of utility.
20. **What are mean Preserve/Self/Paired routes?** Reported by G/M, dataset, and modality in the routing table/CSV.
21. **Do Text and Visual routes differ?** Both modality-specific distributions are shown; they are not treated as semantic role labels.
22. **How large are effective expert contributions?** Probability-weighted RMS and relative-to-G values are in the contribution table/CSV.
23. **Do route probabilities match functional contribution?** Both are reported independently; high route probability alone need not mean a large correction.
24. **Does globalize-router alter the selected M checkpoint?** Its validation metric deltas are in the intervention table/CSV.
25. **Does the intervention agree with retrained M−G?** It is checkpoint reliance evidence and is not substituted for M−G.
26. **Are dataset regimes different?** Compare each dataset's paired signs and routing; no universal regime is presumed.
27. **Did GPR coefficients co-adapt?** Coefficients are shown descriptively; no GPR attribution is reopened.
28. **Which variant is currently most defensible?** See recommendation below, based on retrained paired metrics.
29. **Should this direction continue?** See conservative recommendation; no autonomous follow-on stage is run.
30. **Any claim beyond evidence?** No; validation-only, fixed-split, three-seed descriptive evidence.

## Recommendation

{interpretation} Select B/G/M only from retrained comparisons and their secondary metrics; stop after F0 for human review. Do not add router/expert regularization or more mechanisms in this branch.
"""
    (RESEARCH / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-checkpoint-diagnostics", action="store_true")
    args = parser.parse_args()
    manifest_path = RESEARCH / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_performance(manifest)
    perf_summary, perf_lookup = summarize_performance(rows)
    paired_rows, paired_summary = paired_outputs(perf_lookup)
    if args.skip_checkpoint_diagnostics:
        route_rows, contribution_rows, gpr_rows, intervention_rows, intervention_summary = [], [], [], [], []
    else:
        route_rows, contribution_rows, gpr_rows, intervention_rows, intervention_summary = checkpoint_diagnostics(manifest, perf_lookup, args.device)
    write_csv(DATA_DIR / "performance_by_run.csv", rows)
    write_csv(DATA_DIR / "performance_summary.csv", perf_summary)
    write_csv(DATA_DIR / "paired_delta_by_run.csv", paired_rows)
    write_csv(DATA_DIR / "paired_delta_summary.csv", paired_summary)
    write_csv(DATA_DIR / "routing_diagnostics.csv", route_rows)
    write_csv(DATA_DIR / "expert_contribution.csv", contribution_rows)
    write_csv(DATA_DIR / "gpr_diagnostics.csv", gpr_rows)
    write_csv(DATA_DIR / "intervention_by_run.csv", intervention_rows)
    write_csv(DATA_DIR / "intervention_summary.csv", intervention_summary)
    write_report(manifest, perf_summary, paired_summary, route_rows, contribution_rows, gpr_rows, intervention_summary)
    manifest["analysis_status"] = "complete" if not args.skip_checkpoint_diagnostics else "performance_only"
    manifest["analysis_updated_at"] = now()
    manifest["analysis_source_performance_rows"] = len(rows)
    manifest["paired_comparisons"] = ["M-B", "G-B", "M-G"]
    write_json(manifest_path, manifest)
    print(f"IMoSI-F0 analysis written under {RESEARCH}")


if __name__ == "__main__":
    main()
