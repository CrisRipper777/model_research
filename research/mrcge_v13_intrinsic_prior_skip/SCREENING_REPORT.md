# Screening Report — Intrinsic-Prior Fusion Controlled Screen

Base `184deaaf12b21c9ee01d3328aef15d86bcc50346`; frozen training source `c7f9209c60f18059d330ac23d7b3ab1b13af2e6e`; branch `exp/mrcge_v13_intrinsic_prior_skip_screen`.

Only the registered `intrinsic_prior_skip` model was trained. Best-Accuracy metrics are paired to the same-seed V1.3-A Joint View reference; Best-F1 checkpoint results remain a separate comparison. These validation results are exploratory and are not statistical significance evidence.

## Paired Validation results

| Dataset | Seed | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | Best-F1 Δ Accuracy (pp) | Best-F1 Δ Macro-F1 (pp) | Best-F1 Δ CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | -0.2999 | -0.3127 | -0.001894 | -0.2999 | 0.5898 | -0.004749 |
| Movies | 43 | 0.2699 | -0.4914 | -0.005324 | -2.3395 | -0.2646 | 0.123614 |
| Movies | 44 | -0.4499 | 0.7783 | 0.029215 | -0.8998 | 0.9640 | 0.029975 |
| Grocery | 42 | -0.0586 | -0.2974 | 0.015262 | -0.3221 | 0.2010 | 0.050912 |
| Grocery | 43 | 0.0000 | 0.7654 | -0.002053 | -0.2928 | 0.0994 | 0.004837 |

## Per-dataset descriptive summary

| Dataset | n | Δ Accuracy mean ± population SD (pp) | Δ Macro-F1 mean ± population SD (pp) | Δ CE mean ± population SD | Train CE mean | Validation CE mean | Train−Validation CE gap mean | Lambda mean | Prior RMS mean | Wall time mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 3 | -0.159969 ± 0.310098 | -0.008610 ± 0.561175 | 0.007332 ± 0.015537 | 0.759304 | 1.375466 | -0.616161 | -0.021049 | 0.017907 | 124.366504 |
| Grocery | 2 | -0.029284 ± 0.029284 | 0.233999 ± 0.531397 | 0.006604 ± 0.008658 | 0.209493 | 0.649013 | -0.439519 | -0.026333 | 0.022485 | 128.365960 |

## Initialization, gradients, and run audits

The frozen subclass adds exactly one zero-initialized scalar. Public parent-model state and external NC classifier initialization matched the base for all seven preflight dataset/seed pairs. Dedicated CPU tests confirmed evaluation parity for embeddings, logits, losses, router and completion outputs; training-loss/shared-gradient and forward-RNG parity; finite nonzero scalar gradient and optimizer inclusion; invariant intermediate states; and checkpoint reload/edge-case finiteness. Dedicated tests: 5 passed. Existing V1.1, V1.2, V1.3 and Two-Hop regressions: 73 passed.

All five Stage 1 cells completed once from the frozen source SHA. Both Best-Accuracy and Best-F1 checkpoints were reloaded and SHA256-recorded for each cell. Scalar-gradient, optimizer, validation-RNG, exact data/cache fingerprints, and protected historical-file audits passed. The protected historical set contained 2096 files and remained unchanged. Test evaluation and Link Prediction were off.

## Mechanism and generalization

Final lambda was negative in all five runs and close to zero: Movies −0.01728, −0.02197, −0.02390; Grocery −0.02790, −0.02476. Mean lambda was −0.021049 for Movies and −0.026333 for Grocery. This indicates a small suppression of the original intrinsic contribution, not an enhancement. Mean `RMS(lambda W_s H0)` was 0.017907 (Movies) and 0.022485 (Grocery), equal to 1.71% and 2.34% of the original skip's weighted RMS. Per-seed cosine, RMS scales, lambda trajectories, and gaps are in `data/prior_skip_diagnostics.csv`, `data/epoch_history.csv`, and `data/runs.csv`.

Train/Validation comparison uses the same-seed historical V1.3-A Joint View Best-Accuracy records. Historical training metrics were joined to the exact reference rows; Movies seed 44 comes from the verified Physical-Basis fixed-split row. Per-seed values are in `data/generalization_comparison.csv`, and the source records are in `data/historical_reference_audit.csv`.

| Dataset | Historical Train CE | Candidate Train CE | Δ Train CE | Historical Validation CE | Candidate Validation CE | Δ Validation CE | Historical Train−Validation CE | Candidate Train−Validation CE | Δ CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.789336 | 0.759304 | −0.030032 | 1.368133 | 1.375466 | +0.007332 | −0.578797 | −0.616161 | −0.037364 |
| Grocery | 0.216964 | 0.209493 | −0.007471 | 0.642408 | 0.649013 | +0.006604 | −0.425444 | −0.439519 | −0.014075 |

The negative Train−Validation CE gaps became more negative on average, indicating a wider CE generalization gap: by 0.0374 for Movies and 0.0141 for Grocery. Mean Train−Validation Accuracy gap also widened, from 17.05% to 18.45% on Movies (+1.40 pp) and from 10.62% to 10.76% on Grocery (+0.14 pp). These summaries are descriptive; the small seed counts do not establish statistical significance.

## Stage 1 pre-registered decision

Status: **no_go**; promoted: **False**; Stage 2 authorized: **False**.

| Gate | Result |
|---|---|
| five_of_five_complete | PASS |
| all_numeric_and_checkpoint_audits_pass | PASS |
| historical_references_verified | PASS |
| no_duplicate_or_retry_attempts | PASS |
| frozen_data_cache_fingerprints_match | PASS |
| protected_historical_files_unchanged | PASS |
| Movies_gate | FAIL |
| Grocery_gate | PASS |

Movies requires mean Δ Accuracy ≥ +0.20 pp, at least two positive seeds, mean Δ Macro-F1 ≥ −0.10 pp, and mean Δ CE ≤ +0.020. Grocery requires mean Δ Accuracy ≥ −0.10 pp, mean Δ Macro-F1 ≥ −0.30 pp, and mean Δ CE ≤ +0.020. All five valid cells, reloads, data fingerprints, scalar-gradient/optimizer evidence, Test-off and protected-history audits are required.

## Final recommendation

**NO-GO** — Stage 1 did not pass all pre-registered gates; ele-fashion was not run.

No Test metrics were evaluated. No Link Prediction task was run. No statistical significance claim is made.
