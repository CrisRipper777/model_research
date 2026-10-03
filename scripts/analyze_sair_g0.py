#!/usr/bin/env python3
"""Summarize SAIR-G0 validation runs and selected-checkpoint geometry."""
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
RESEARCH = ROOT / "research" / "sair_g0_anchor_response_readout"
DATA_DIR = RESEARCH / "data"
DATASETS = ("Movies", "Grocery", "ele-fashion")
SEEDS = (42, 43, 44)
VARIANTS = ("B", "C", "D")
MODEL_VARIANTS = {"B": "base", "C": "generic", "D": "decoupled"}
SPLITS = {
    "Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt",
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path):
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


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def mean_sd(values):
    return statistics.mean(values), statistics.pstdev(values)


def scalar(value):
    return float(value.detach().float().cpu().item()) if hasattr(value, "detach") else float(value)


def _rms(tensor):
    import torch
    value = tensor.detach().float()
    return float(value.square().mean().sqrt().cpu())


def _cos(left, right):
    import torch
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).cpu())


def read_performance(manifest):
    runs = [r for r in manifest.get("formal_runs", []) if r.get("status") == "complete"]
    if len(runs) != 27:
        raise RuntimeError(f"expected 27 complete formal runs, found {len(runs)}")
    rows, seen = [], set()
    for run in runs:
        key = (run["dataset"], run["variant"], int(run["seed"]))
        if key in seen:
            raise RuntimeError(f"duplicate formal key {key}")
        seen.add(key)
        path = Path(run["run_metrics_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("development_no_test") is not True:
            raise RuntimeError(f"validation-only flag missing from {path}")
        records = payload.get("runs", [])
        if len(records) != 1 or int(records[0]["seed"]) != key[2]:
            raise RuntimeError(f"unexpected run record in {path}")
        metrics = records[0]["metrics"]
        if any(name.startswith("test_") for name in metrics):
            raise RuntimeError(f"test metric present in {path}")
        rows.append({
            "dataset": key[0], "variant": key[1], "model_variant": MODEL_VARIANTS[key[1]], "seed": key[2],
            "val_accuracy": float(metrics["val_acc"]), "val_macro_f1": float(metrics["val_macro_f1"]),
            "val_ce": float(metrics["val_ce"]),
            "best_epoch": int(records[0].get("metadata", {}).get("best_epoch", run.get("best_epoch", 0))),
            "development_no_test": True, "evaluate_test": False,
            "run_metrics_path": str(path), "checkpoint_path": run["checkpoint_path"],
            "source_commit_sha": run["source_commit_sha"],
        })
    expected = {(d, v, s) for d in DATASETS for v in VARIANTS for s in SEEDS}
    if seen != expected:
        raise RuntimeError("formal dataset × variant × seed grid is incomplete")
    return rows


def summarize_performance(rows):
    summaries, lookup = [], {(r["dataset"], r["variant"], int(r["seed"])): r for r in rows}
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [lookup[(dataset, variant, seed)] for seed in SEEDS]
            row = {"dataset": dataset, "variant": variant, "n_seeds": len(group)}
            for metric in ("val_accuracy", "val_macro_f1", "val_ce"):
                row[f"{metric}_mean"], row[f"{metric}_population_sd"] = mean_sd([float(x[metric]) for x in group])
            summaries.append(row)
    return summaries, lookup


def paired_deltas(lookup):
    rows, summaries = [], []
    comparisons = (("D-B", "D", "B"), ("D-C", "D", "C"), ("C-B", "C", "B"))
    for dataset in DATASETS:
        for name, new_variant, base_variant in comparisons:
            group = []
            for seed in SEEDS:
                new, base = lookup[(dataset, new_variant, seed)], lookup[(dataset, base_variant, seed)]
                row = {
                    "dataset": dataset, "seed": seed, "comparison": name,
                    "new_variant": new_variant, "base_variant": base_variant,
                    "accuracy_delta_pp": 100 * (new["val_accuracy"] - base["val_accuracy"]),
                    "macro_f1_delta_pp": 100 * (new["val_macro_f1"] - base["val_macro_f1"]),
                    "ce_delta": new["val_ce"] - base["val_ce"],
                }
                rows.append(row)
                group.append(row)
            summary = {"dataset": dataset, "comparison": name, "n_paired_seeds": len(group)}
            for metric in ("accuracy_delta_pp", "macro_f1_delta_pp", "ce_delta"):
                values = [float(r[metric]) for r in group]
                summary[f"{metric}_mean"], summary[f"{metric}_population_sd"] = mean_sd(values)
                summary[f"{metric}_positive_seeds"] = sum(v > 1e-12 for v in values)
                summary[f"{metric}_negative_seeds"] = sum(v < -1e-12 for v in values)
                summary[f"{metric}_ties"] = sum(abs(v) <= 1e-12 for v in values)
            summaries.append(summary)
    for name, new_variant, base_variant in comparisons:
        group = [r for r in rows if r["comparison"] == name]
        summary = {"dataset": "ALL", "comparison": name, "n_paired_seeds": len(group)}
        for metric in ("accuracy_delta_pp", "macro_f1_delta_pp", "ce_delta"):
            values = [float(r[metric]) for r in group]
            summary[f"{metric}_mean"], summary[f"{metric}_population_sd"] = mean_sd(values)
            summary[f"{metric}_positive_seeds"] = sum(v > 1e-12 for v in values)
            summary[f"{metric}_negative_seeds"] = sum(v < -1e-12 for v in values)
            summary[f"{metric}_ties"] = sum(abs(v) <= 1e-12 for v in values)
        summaries.append(summary)
    return rows, summaries


def _compose_context(dataset, seed, device):
    from hydra import compose, initialize_config_dir
    from src.data import load_mag_data
    from src.models.sair_mag_v0 import Model
    overrides = [
        f"dataset={dataset}", "task=nc", "model=sair_mag_v0", "model.variant=decoupled",
        f"seed={seed}", f"device={device}", "task.evaluate_test=false", "task.development_no_test=true",
    ]
    if dataset in ("Movies", "Grocery"):
        overrides.append(f"dataset.nc_split_path={SPLITS[dataset]}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)
    data = load_mag_data(cfg, "nc", seed)
    info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes, "text_dim": int(data.x_t.shape[1]),
            "visual_dim": int(data.x_i.shape[1])}
    net = Model(cfg, info).to(device)
    x = data.x if data.num_nodes >= 50_000 else data.x.to(device)
    return cfg, data, net, x, data.edge_index.to(device)


def _load_checkpoint(net, variant, run, device):
    import torch
    payload = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=False)
    net.variant = MODEL_VARIANTS[variant]
    net.load_state_dict(payload["model_state"], strict=True)
    net.to(device).eval()
    return payload


def checkpoint_diagnostics(manifest, device):
    import torch
    index = {(r["dataset"], r["variant"], int(r["seed"])): r for r in manifest["formal_runs"] if r.get("status") == "complete"}
    geometry, adapters, coefficients = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            _, data, net, x, edge = _compose_context(dataset, seed, device)
            for variant in VARIANTS:
                run = index[(dataset, variant, seed)]
                payload = _load_checkpoint(net, variant, run, device)
                with torch.no_grad():
                    _, _, _, _, info = net(x, edge, return_details=True)
                details = info["details"]
                coeff = details["coefficients"].detach().float().cpu().tolist()
                coefficients.append({
                    "dataset": dataset, "variant": variant, "seed": seed,
                    **{f"c{i}": coeff[i] for i in range(4)},
                    "selected_epoch": run["best_epoch"], "checkpoint_path": run["checkpoint_path"],
                })
                pair_up_norm = scalar(net.pair_up.weight.norm())
                pair_down_norm = scalar(net.pair_down.weight.norm())
                for modality in ("text", "visual"):
                    mod = details["modalities"][modality]
                    p, r, g = mod["prior"], mod["structural_response"], mod["embedding"]
                    rp, c0p = _rms(p), _rms(coeff[0] * p)
                    rr = _rms(r)
                    geometry.append({
                        "dataset": dataset, "variant": variant, "seed": seed, "modality": modality,
                        "rms_p": rp, "rms_r": rr, "rms_r_over_rms_p": rr / max(rp, 1e-30),
                        "cosine_p_r": _cos(p, r), "cosine_p_g": _cos(p, g), "cosine_r_g": _cos(r, g),
                        "rms_c0p": c0p, "rms_r_over_rms_c0p": rr / max(c0p, 1e-30),
                        "selected_epoch": run["best_epoch"],
                    })
                    if variant in ("C", "D"):
                        pair = details["pairs"][modality]
                        delta, hidden = pair["delta"], pair["hidden"]
                        # Large-graph diagnostics avoid retaining the full pair hidden tensor.
                        hidden_rms = scalar(pair["hidden_rms"])
                        row = {
                            "dataset": dataset, "variant": variant, "seed": seed, "modality": modality,
                            "rms_delta": _rms(delta), "rms_delta_over_rms_g": _rms(delta) / max(_rms(g), 1e-30),
                            "cosine_delta_g": _cos(delta, g), "pair_hidden_rms": hidden_rms,
                            "pair_up_weight_norm": pair_up_norm, "pair_down_weight_norm": pair_down_norm,
                            "selected_epoch": run["best_epoch"],
                        }
                        if variant == "D":
                            row["cosine_delta_p"] = _cos(delta, p)
                            row["cosine_delta_r"] = _cos(delta, r)
                        else:
                            row["cosine_delta_p"] = ""
                            row["cosine_delta_r"] = ""
                        adapters.append(row)
                del payload, info
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del data, net, x, edge
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return geometry, adapters, coefficients


def _fmt(value, digits=3):
    return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _summarize_diagnostics(rows, keys, value_keys):
    output = []
    for group_key in keys:
        group = [row for row in rows if all(row[k] == v for k, v in group_key.items())]
        if not group:
            continue
        row = dict(group_key)
        for metric in value_keys:
            values = [float(x[metric]) for x in group]
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(values)
        output.append(row)
    return output


def write_report(manifest, perf_summary, paired_summary, geometry, adapters, gpr, smoke):
    perf_rows = []
    for dataset in DATASETS:
        group = {r["variant"]: r for r in perf_summary if r["dataset"] == dataset}
        perf_rows.append([dataset] + [
            f"{_fmt(group[v]['val_accuracy_mean']*100, 2)} ± {_fmt(group[v]['val_accuracy_population_sd']*100, 2)}" for v in VARIANTS
        ] + [
            f"{_fmt(group[v]['val_macro_f1_mean']*100, 2)} ± {_fmt(group[v]['val_macro_f1_population_sd']*100, 2)}" for v in VARIANTS
        ] + [f"{_fmt(group[v]['val_ce_mean'], 4)} ± {_fmt(group[v]['val_ce_population_sd'], 4)}" for v in VARIANTS])
    paired_rows = []
    for row in paired_summary:
        paired_rows.append([
            row["dataset"], row["comparison"], row["n_paired_seeds"],
            f"{_fmt(row['accuracy_delta_pp_mean'])} ± {_fmt(row['accuracy_delta_pp_population_sd'])}",
            f"{row['accuracy_delta_pp_positive_seeds']}/{row['accuracy_delta_pp_negative_seeds']}/{row['accuracy_delta_pp_ties']}",
            f"{_fmt(row['macro_f1_delta_pp_mean'])} ± {_fmt(row['macro_f1_delta_pp_population_sd'])}",
            f"{_fmt(row['ce_delta_mean'], 4)} ± {_fmt(row['ce_delta_population_sd'], 4)}",
        ])
    geo_groups = [{"variant": v, "modality": m} for v in VARIANTS for m in ("text", "visual")]
    geo_summary = _summarize_diagnostics(
        geometry, geo_groups,
        ["rms_r_over_rms_p", "cosine_p_r", "cosine_p_g", "cosine_r_g", "rms_r_over_rms_c0p"],
    )
    geometry_rows = [[r["variant"], r["modality"]] + [
        f"{_fmt(r[f'{metric}_mean'])} ± {_fmt(r[f'{metric}_sd'])}"
        for metric in ("rms_r_over_rms_p", "cosine_p_r", "cosine_p_g", "cosine_r_g", "rms_r_over_rms_c0p")
    ] for r in geo_summary]
    adapter_groups = [{"variant": v, "modality": m} for v in ("C", "D") for m in ("text", "visual")]
    adapter_summary = _summarize_diagnostics(
        adapters, adapter_groups,
        ["rms_delta_over_rms_g", "cosine_delta_g", "pair_hidden_rms", "pair_up_weight_norm", "pair_down_weight_norm"],
    )
    adapter_rows = [[r["variant"], r["modality"]] + [
        f"{_fmt(r[f'{metric}_mean'])} ± {_fmt(r[f'{metric}_sd'])}"
        for metric in ("rms_delta_over_rms_g", "cosine_delta_g", "pair_hidden_rms", "pair_up_weight_norm", "pair_down_weight_norm")
    ] for r in adapter_summary]
    coeff_summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [r for r in gpr if r["dataset"] == dataset and r["variant"] == variant]
            coeff_summary.append([dataset, variant] + [
                f"{statistics.mean(float(r[f'c{i}']) for r in group):.5f} ± {statistics.pstdev(float(r[f'c{i}']) for r in group):.5f}"
                for i in range(4)
            ])
    smoke_rows = []
    for row in smoke.get("nc_runs", []):
        metrics = row.get("metrics", {})
        smoke_rows.append([row["variant"], row.get("status"), _fmt(metrics.get("val_acc", float("nan"))*100, 2), _fmt(metrics.get("val_macro_f1", float("nan"))*100, 2), _fmt(metrics.get("val_ce", float("nan")), 4), row.get("pair_up_weight_norm_at_selected_checkpoint")])
    lp_rows = [[r.get("variant"), r.get("status"), r.get("link_neighbor_loader"), r.get("sampler_fanouts_observed"), r.get("positive_edge_removal_observed"), r.get("forward_backward_validation"), r.get("checkpoint_saved")] for r in smoke.get("lp_runs", [])]

    primary = [r for r in paired_summary if r["dataset"] == "ALL"]
    lookup = {r["comparison"]: r for r in primary}
    db, dc, cb = lookup["D-B"], lookup["D-C"], lookup["C-B"]
    d_supported_datasets = sum(
        next(r for r in paired_summary if r["dataset"] == d and r["comparison"] == "D-B")["accuracy_delta_pp_mean"] > 0
        and next(r for r in paired_summary if r["dataset"] == d and r["comparison"] == "D-C")["accuracy_delta_pp_mean"] > 0
        for d in DATASETS
    )
    if d_supported_datasets >= 2:
        diagnosis = "D has positive mean Accuracy deltas against both B and C in at least two datasets; inspect F1/CE direction and seed spread before treating this as support. The design screen is suggestive only with three model seeds."
        recommendation = "D is a candidate for human review, not an established replacement; keep B as the conservative reference until metric conflicts and per-seed patterns are reviewed."
        g1 = "Do not start G1 automatically. First review whether D's paired Accuracy and secondary-metric pattern is consistently favorable across the datasets."
    elif cb["accuracy_delta_pp_mean"] > 0 and abs(dc["accuracy_delta_pp_mean"]) <= abs(db["accuracy_delta_pp_mean"]):
        diagnosis = "The generic adapter is at least as directionally promising as provenance exposure on pooled means; this does not show that P/R provenance adds value beyond generic capacity."
        recommendation = "Prefer B pending human review; C may remain a simple readout-capacity control if its validation pattern is consistently favorable."
        g1 = "No. The current retrained controls do not establish value specific to anchor-response provenance, so frequency decomposition is not justified by this screen."
    else:
        diagnosis = "The pooled paired means do not show a stable D advantage over both controls. Any dataset-specific gains remain exploratory and should not be generalized."
        recommendation = "B, the unchanged RGD readout, is the conservative recommendation unless a clear dataset-specific use case is identified."
        g1 = "No. Stop new core-module development pending human review; these results do not motivate G1 frequency decomposition."

    lines = [
        "# SAIR-G0 — Semantic-Anchor / Structural-Response Decoupled Readout Screen",
        "",
        f"Generated: {now()}",
        "",
        "## Experiment summary",
        "",
        f"- Base SHA: `{manifest['base_sha']}`; source SHA: `{manifest['source_commit_sha']}`; branch: `{manifest['branch']}`.",
        f"- Validation-only campaign: {len([r for r in manifest.get('formal_runs', []) if r.get('status') == 'complete'])}/27 completed; seeds 42/43/44 are model/training seeds on fixed splits.",
        "- No NC test metrics were requested or accepted by the runner. No split changes, Toys, Reddit-S, or formal LP runs were made.",
        "- Readout variants: B = RGD base; C = generic adapter on G; D = shared adapter on raw P/R response provenance.",
        "",
        "## Smoke",
        "",
        _table(["NC variant", "status", "Val Acc %", "Macro-F1 %", "CE", "selected pair_up norm"], smoke_rows),
        "",
        "Movies seed 42 used two epochs for each NC smoke run. B retained exact zero `pair_up`; C and D updated the initially zero adapter. Initial B/C/D eval equality and RNG-reset train equality are covered by the committed tests.",
        "",
        _table(["LP variant", "status", "LinkNeighborLoader", "[5,5,5]", "positive edge removal", "forward/backward + validation", "checkpoint"], lp_rows),
        "",
        "LP smoke was limited to sports-copurchase B/D, two epochs, two train batches, fanouts [5,5,5], with test evaluation disabled. It is a protocol check only.",
        "",
        "## Validation performance",
        "",
        "Values are mean ± population SD over three training seeds. Accuracy and Macro-F1 are percentages; CE is cross-entropy.",
        "",
        _table(["Dataset", "B Acc", "C Acc", "D Acc", "B Macro-F1", "C Macro-F1", "D Macro-F1", "B CE", "C CE", "D CE"], perf_rows),
        "",
        "## Paired comparisons",
        "",
        "Accuracy deltas are percentage points. Seed counts are positive/negative/tie. Comparisons are paired by dataset and seed. No pseudo-IID p-values or fixed performance threshold are used.",
        "",
        _table(["Dataset", "Comparison", "Pairs", "Accuracy Δ pp mean ± SD", "+/−/=", "Macro-F1 Δ pp mean ± SD", "CE Δ mean ± SD"], paired_rows),
        "",
        "## Anchor-response geometry",
        "",
        "Cosines are computed over flattened checkpoint tensors, summarized across dataset × seed observations by modality/variant. They describe geometry and do not imply beneficial or harmful information.",
        "",
        _table(["Variant", "Modality", "RMS(R)/RMS(P)", "cos(P,R)", "cos(P,G)", "cos(R,G)", "RMS(R)/RMS(c0P)"], geometry_rows),
        "",
        "The per-run values, including RMS(P), RMS(R), and RMS(c0P), are in `data/anchor_response_diagnostics.csv`.",
        "",
        "## Adapter and GPR diagnostics",
        "",
        _table(["Variant", "Modality", "RMS(δ)/RMS(G)", "cos(δ,G)", "pair hidden RMS", "pair_up norm", "pair_down norm"], adapter_rows),
        "",
        "For D, per-run `cos(δ,P)` and `cos(δ,R)` are in `data/adapter_diagnostics.csv`; corresponding values for C are intentionally blank. Adapter activity is descriptive, not performance evidence.",
        "",
        _table(["Dataset", "Variant", "c0", "c1", "c2", "c3"], coeff_summary),
        "",
        "## Conservative diagnosis and recommendation",
        "",
        diagnosis,
        "",
        recommendation,
        "",
        g1,
        "",
        "`R = Q - c0P` is reported only as a structure-induced response. The experiment does not show that it is pure smoothing, low-frequency, or heterophilous information. Accuracy/F1/CE conflicts and per-seed variation should be reviewed directly in the CSVs.",
        "",
        "## Self-audit (30 items)",
        "",
        f"1. Started strictly from `997b89b8b0e59ca2da9654d07cc184cc1ce14169`: **yes**.\n2. Used another experiment branch: **no**; branch ancestry is the required F0 SHA.\n3. Source committed before smoke/formal: **yes**, source SHA `{manifest['source_commit_sha']}` was fixed in the manifest before running.\n4. Formal source SHA: `{manifest['source_commit_sha']}`.\n5. NC test read/run: **no**; runner requires validation-only flags and rejects test metrics.\n6. Dataset split modified: **no**.\n7. B regression to PCRR-v1/RGD: **yes**, output and proposal/embedding tests pass at atol 1e-6.\n8. Module construction preserves PCRR base RNG compatibility: **yes**, CPU RNG state and shared state tensors are bitwise matched by tests.\n9. B/C/D parameter counts, state layouts, and same-seed initialization matched: **yes**, tested.\n10. Initial B/C/D functions identical: **yes**, eval and RNG-reset train tests.\n11. R equals Q−c0P and explicit k≥1 sum: **yes**, small and StreamingRawGPR synthetic tests pass at atol 1e-6.\n12. D uses raw R without separate normalization: **yes**.\n13. C adapter input is only G: **yes**, its exact pair layout is [G,G,0,G*G].\n14. D is modality-local: **yes**, text adapter correction is invariant to changes in visual input.\n15. D−B Accuracy/F1/CE: see `D-B` rows in the paired table; pooled Accuracy `{_fmt(db['accuracy_delta_pp_mean'])} ± {_fmt(db['accuracy_delta_pp_population_sd'])}` pp, Macro-F1 `{_fmt(db['macro_f1_delta_pp_mean'])} ± {_fmt(db['macro_f1_delta_pp_population_sd'])}` pp, CE `{_fmt(db['ce_delta_mean'], 4)} ± {_fmt(db['ce_delta_population_sd'], 4)}`.\n16. D−C: pooled Accuracy `{_fmt(dc['accuracy_delta_pp_mean'])} ± {_fmt(dc['accuracy_delta_pp_population_sd'])}` pp, Macro-F1 `{_fmt(dc['macro_f1_delta_pp_mean'])} ± {_fmt(dc['macro_f1_delta_pp_population_sd'])}` pp, CE `{_fmt(dc['ce_delta_mean'], 4)} ± {_fmt(dc['ce_delta_population_sd'], 4)}`; dataset rows above.\n17. C−B: pooled Accuracy `{_fmt(cb['accuracy_delta_pp_mean'])} ± {_fmt(cb['accuracy_delta_pp_population_sd'])}` pp, Macro-F1 `{_fmt(cb['macro_f1_delta_pp_mean'])} ± {_fmt(cb['macro_f1_delta_pp_population_sd'])}` pp, CE `{_fmt(cb['ce_delta_mean'], 4)} ± {_fmt(cb['ce_delta_population_sd'], 4)}`.\n18. Evidence provenance adds value beyond generic adapter: {('not established; D was not consistently better than both controls' if d_supported_datasets < 2 else 'directionally suggestive, but only three seeds and secondary metrics require human review')}.\n19. RMS(R)/RMS(P): see geometry table and per-run CSV; these are dataset/modality/checkpoint-specific measurements.\n20. Is cos(P,R) dataset/modality dependent: see per-run geometry CSV; the pooled rows above must not be mistaken for a universal constant.\n21. D correction magnitude: `RMS(δ)/RMS(G)` in adapter table, by modality and across seed/dataset checkpoints.\n22. Is δ closer to P, R, or G: D `cos(δ,P)`, `cos(δ,R)`, `cos(δ,G)` are recorded per run in the adapter CSV; compare by dataset/modality without treating cosine as utility.\n23. GPR co-adaptation: compare c0:c3 across B/C/D in the table; the table is descriptive and does not reopen attribution.\n24. Dataset-specific regime: inspect the paired per-dataset rows; pooled averages can conceal opposite directions.\n25. Accuracy/F1/CE conflict: all three metrics are reported together per paired seed and dataset; no metric is silently privileged after seeing results.\n26. Current recommendation: {recommendation}\n27. Worth entering G1 frequency decomposition: {g1}\n28. If not, stop new core modules: yes, await human review before further core-module work.\n29. LP smoke protocol correctness: see LP table; loader, fanout, edge removal, forward/backward, validation and checkpoint checks were required for completion.\n30. Claims beyond direct evidence: none intended; R is not labeled as heterophilous or frequency-pure, and diagnostics are not treated as performance evidence.",
        "",
    ]
    (RESEARCH / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads((RESEARCH / "run_manifest.json").read_text(encoding="utf-8"))
    performance = read_performance(manifest)
    perf_summary, lookup = summarize_performance(performance)
    paired, paired_summary = paired_deltas(lookup)
    geometry, adapters, gpr = checkpoint_diagnostics(manifest, args.device)
    write_csv(DATA_DIR / "performance_by_run.csv", performance)
    write_csv(DATA_DIR / "performance_summary.csv", perf_summary)
    write_csv(DATA_DIR / "paired_delta_by_run.csv", paired)
    write_csv(DATA_DIR / "paired_delta_summary.csv", paired_summary)
    write_csv(DATA_DIR / "anchor_response_diagnostics.csv", geometry)
    write_csv(DATA_DIR / "adapter_diagnostics.csv", adapters)
    write_csv(DATA_DIR / "gpr_diagnostics.csv", gpr)
    smoke = json.loads((RESEARCH / "smoke_status.json").read_text(encoding="utf-8"))
    write_report(manifest, perf_summary, paired_summary, geometry, adapters, gpr, smoke)
    manifest["analysis_status"] = "complete"
    manifest["analysis_finished_at"] = now()
    write_json(RESEARCH / "run_manifest.json", manifest)
    print(f"SAIR-G0 analysis written under {RESEARCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
