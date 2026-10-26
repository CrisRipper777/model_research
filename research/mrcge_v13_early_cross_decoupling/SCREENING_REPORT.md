# MR-CGE V1.3 Expert-Decoupled Early Cross — screening report

## Scope and outcome

Seven authorized training cells were completed: 7/7. The only new condition was `expert_decoupled_early_cross` on V1.3-A M_repaired Joint View. GPU: physical GPU 1. Fixed splits and V7B.1 K=8 caches were paired to historical V1.3-A records. Test evaluation and Link Prediction were disabled. This is a small descriptive screen and does not support significance or state-of-the-art claims.

Source commit: `4ea0e0aeebdc39cc9329db47ef9c047485ee8916`; base: `021959d82ec74a71955c97521867dc0d3f4a6f01`; branch: `exp/mrcge_v13_early_cross_decoupling_screen`.

## Paired validation results

Metrics are Best Validation Accuracy checkpoint Acc / Macro-F1 / CE; Best-F1 metrics are reported separately. Parameter count includes the external classifier. Peak reserved memory is in bytes.

| Dataset | Seed | V1.3-A Acc / F1 / CE | New Acc / F1 / CE | New Best-F1 Acc / F1 / CE | Best Acc/F1 epoch | Wall seconds | Parameters | Peak reserved bytes |
|---|---:|---|---|---|---:|---:|---:|---:|
| Grocery | 42 | 0.8351 / 0.7662 / 0.63923 | 0.8354 / 0.7695 / 0.67304 | 0.8343 / 0.7711 / 0.66746 | 85/80 | 99.7 | 2067939 | 3034578944 |
| Grocery | 43 | 0.8369 / 0.7694 / 0.64558 | 0.8375 / 0.7753 / 0.64823 | 0.8369 / 0.7796 / 0.64810 | 77/79 | 92.9 | 2067939 | 3034578944 |
| Movies | 42 | 0.5597 / 0.4820 / 1.37037 | 0.5564 / 0.4738 / 1.44108 | 0.5480 / 0.5014 / 1.41612 | 90/78 | 106.4 | 2067939 | 3019898880 |
| Movies | 43 | 0.5522 / 0.4917 / 1.37982 | 0.5531 / 0.4776 / 1.41279 | 0.5432 / 0.4956 / 1.43293 | 80/81 | 97.9 | 2067939 | 3019898880 |
| Movies | 44 | 0.5585 / 0.4803 / 1.35421 | 0.5558 / 0.4836 / 1.37298 | 0.5465 / 0.4929 / 1.36312 | 67/56 | 131.0 | 2067939 | 3019898880 |
| ele-fashion | 42 | 0.8710 / 0.7328 / 0.40106 | 0.8738 / 0.7398 / 0.42229 | 0.8550 / 0.7498 / 0.43927 | 127/157 | 515.0 | 1799627 | 15286140928 |
| ele-fashion | 43 | 0.8750 / 0.7379 / 0.40386 | 0.8737 / 0.7484 / 0.43080 | 0.8654 / 0.7615 / 0.40979 | 148/113 | 569.8 | 1799627 | 15286140928 |

## New minus V1.3-A paired differences

Acc and Macro-F1 differences are percentage points; CE is raw cross entropy.

### Movies

- Seed 42: Acc -0.330 pp, Macro-F1 -0.816 pp, CE +0.07071 (best_validation_accuracy).
- Seed 42: Acc -0.420 pp, Macro-F1 +0.849 pp, CE -0.02099 (best_validation_macro_f1).
- Seed 43: Acc +0.090 pp, Macro-F1 -1.412 pp, CE +0.03297 (best_validation_accuracy).
- Seed 43: Acc -0.300 pp, Macro-F1 -0.821 pp, CE +0.04286 (best_validation_macro_f1).
- Seed 44: Acc -0.270 pp, Macro-F1 +0.325 pp, CE +0.01878 (best_validation_accuracy).
- Seed 44: Acc -0.330 pp, Macro-F1 -0.330 pp, CE -0.05904 (best_validation_macro_f1).
### Grocery

- Seed 42: Acc +0.029 pp, Macro-F1 +0.327 pp, CE +0.03380 (best_validation_accuracy).
- Seed 42: Acc +0.117 pp, Macro-F1 +0.229 pp, CE +0.03052 (best_validation_macro_f1).
- Seed 43: Acc +0.059 pp, Macro-F1 +0.592 pp, CE +0.00264 (best_validation_accuracy).
- Seed 43: Acc +0.205 pp, Macro-F1 +0.233 pp, CE -0.02523 (best_validation_macro_f1).
### ele-fashion

- Seed 42: Acc +0.276 pp, Macro-F1 +0.692 pp, CE +0.02122 (best_validation_accuracy).
- Seed 42: Acc -1.473 pp, Macro-F1 +0.175 pp, CE +0.01803 (best_validation_macro_f1).
- Seed 43: Acc -0.133 pp, Macro-F1 +1.052 pp, CE +0.02694 (best_validation_accuracy).
- Seed 43: Acc +1.094 pp, Macro-F1 +0.606 pp, CE -0.01812 (best_validation_macro_f1).

### Paired difference summary by dataset

Entries are mean ± population SD across seeds. Acc and Macro-F1 are percentage points; CE is raw loss.

| Dataset | Checkpoint selection | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE | n |
|---|---|---:|---:|---:|---:|
| Movies | best_validation_accuracy | -0.170 ± 0.185 | -0.634 ± 0.720 | +0.04082 ± 0.02192 | 3 |
| Movies | best_validation_macro_f1 | -0.350 ± 0.051 | -0.101 ± 0.701 | -0.01239 ± 0.04204 | 3 |
| Grocery | best_validation_accuracy | +0.044 ± 0.015 | +0.460 ± 0.133 | +0.01822 ± 0.01558 | 2 |
| Grocery | best_validation_macro_f1 | +0.161 ± 0.044 | +0.231 ± 0.002 | +0.00265 ± 0.02788 | 2 |
| ele-fashion | best_validation_accuracy | +0.072 ± 0.205 | +0.872 ± 0.180 | +0.02408 ± 0.00286 | 2 |
| ele-fashion | best_validation_macro_f1 | -0.189 ± 1.284 | +0.390 ± 0.216 | -0.00005 ± 0.01807 | 2 |

## Fixed-split Hybrid context

Hybrid H2 3-hop Movies records are historical context only; they do not isolate the effect of this architecture change.

| Dataset | Seed | Acc | Macro-F1 | CE | Source SHA |
|---|---:|---:|---:|---:|---|
| Movies | 42 | 0.5686862468719482 | 0.5025715503097979 | 1.403075098991394 | 5e052dbc26237ddb534e0f779d91951f26079748 |
| Movies | 43 | 0.5704858899116516 | 0.4943332711998701 | 1.3445202112197876 | cb641c4947c13d2305cfe83110cfdf9e35d42517 |
| Movies | 44 | 0.5692861676216125 | 0.4884497412889724 | 1.3362191915512085 | cb641c4947c13d2305cfe83110cfdf9e35d42517 |

## Diagnostics and audit

- Audited expert coefficient rows: 28 (expected 28); Cross injection rows: 56 (expected 56).
- Each new run has independent Best-Accuracy and Best-F1 checkpoint hashes and reload verification.
- Paired-dropout Validation RNG checks passed; all Test labels were masked before Trainer entry.
- Protected historical research/output files were hash-checked against CPU preflight.
- Preserved failure/block records: 1 total, including one first GPU preflight attempt blocked because sandboxed `nvidia-smi` received SIGKILL. Its fresh guarded retry passed; all seven current-source GPU preflights and all seven formal trainings passed.
- Formal training attempts failed: 0; blocked: 0.

## Interpretation boundary

The learned `b[4]` parameters are shared across modalities. The attached correction cancels theta's Cross-coefficient path while preserving its private/semantic/own-physical paths. The changed collaborative expert input can still alter routing and Joint View outputs indirectly. Results are descriptive, and the screen authorizes no default selection or follow-on model work.
