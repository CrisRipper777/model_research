# PCRR-E0.1 — Target-Only Active Residual Control

## Execution and provenance

- Required base: `d1c49583e56af20063ed7274630b2df27cc612e7`; branch: `exp/pcrr_e01_target_only_control`; exact experiment source snapshot: `5b8f9edbef04146a655be49c1a42663dbfe463f2`. Runs were completed before committing, then the executed source files were committed unchanged at this source snapshot.
- New training in this phase: **9 T runs** (3 datasets × seeds 42/43/44).
- Frozen parent data reused: **27 E0 B/P/S runs** directly from `research/pcrr_e0_postgpr_paired_residual/data/performance_by_run.csv`. The combined table has 36 rows because it joins those 27 historical rows with 9 new runs; it does not represent 36 new trainings.
- Fixed splits: `{"Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt", "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt", "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt"}`. All runs use `evaluate_test=false`, `development_no_test=true`; no NC test metrics were used. No split was modified, no LP was run, and no B/P/S formal run was repeated.
- Means and population SDs (`ddof=0`) summarize three paired model seeds, not independent samples.

## Validation performance

Accuracy, Macro-F1 are percent; CE is in task units. Each cell is mean ± population SD over seeds. B/P/S values are frozen E0; T values are newly trained in E0.1.

| Dataset | B Acc | T Acc | P Acc | S Acc | B F1 | T F1 | P F1 | S F1 | B CE | T CE | P CE | S CE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Movies | 56.53 ± 0.19 | 56.70 ± 0.07 | 56.91 ± 0.06 | 56.14 ± 0.06 | 48.65 ± 0.98 | 48.39 ± 1.35 | 50.06 ± 1.58 | 48.93 ± 0.65 | 1.3751 ± 0.0274 | 1.3681 ± 0.0175 | 1.3680 ± 0.0203 | 1.3525 ± 0.0120 |
| Grocery | 83.04 ± 0.23 | 82.92 ± 0.15 | 82.98 ± 0.22 | 82.44 ± 0.06 | 76.78 ± 0.44 | 76.12 ± 1.09 | 76.62 ± 0.29 | 75.87 ± 0.39 | 0.6573 ± 0.0270 | 0.6556 ± 0.0092 | 0.6505 ± 0.0095 | 0.6789 ± 0.0193 |
| ele-fashion | 87.52 ± 0.16 | 87.54 ± 0.09 | 87.25 ± 0.07 | 86.84 ± 0.10 | 74.28 ± 0.17 | 75.33 ± 0.57 | 73.42 ± 0.53 | 72.19 ± 0.26 | 0.4253 ± 0.0165 | 0.4229 ± 0.0081 | 0.4062 ± 0.0050 | 0.4235 ± 0.0096 |

Full run-level values and origins: [combined_performance_by_run.csv](data/combined_performance_by_run.csv). The E0.1-only subset is [target_performance_by_run.csv](data/target_performance_by_run.csv).

## Paired comparisons

Accuracy and Macro-F1 deltas are percentage points; CE delta is task units. Accuracy direction counts are positive/negative/tied seeds. The primary comparison is P−T. Parent P−B, P−S, and S−B rows are frozen E0 context.

| Dataset | Pair | Accuracy Δ mean ± SD | Acc +/−/= | Macro-F1 Δ mean ± SD | CE Δ mean ± SD | Origin |
|---|---|---|---|---|---|---|
| Movies | P-T | 0.210 ± 0.107 | 3/0/0 | 1.667 ± 0.454 | -0.0001 ± 0.0329 | E0.1_new_pairing |
| Movies | T-B | 0.170 ± 0.135 | 2/0/1 | -0.262 ± 0.497 | -0.0070 ± 0.0301 | E0.1_new_pairing |
| Movies | T-S | 0.560 ± 0.116 | 3/0/0 | -0.542 ± 1.308 | 0.0156 ± 0.0260 | E0.1_new_pairing |
| Movies | P-B | 0.380 ± 0.177 | 3/0/0 | 1.405 ± 0.602 | -0.0071 ± 0.0468 | E0_frozen |
| Movies | P-S | 0.770 ± 0.014 | 3/0/0 | 1.124 ± 1.352 | 0.0155 ± 0.0277 | E0_frozen |
| Movies | S-B | -0.390 ± 0.191 | 0/3/0 | 0.280 ± 0.817 | -0.0226 ± 0.0200 | E0_frozen |
| Grocery | P-T | 0.059 ± 0.063 | 2/0/1 | 0.499 ± 0.832 | -0.0052 ± 0.0180 | E0.1_new_pairing |
| Grocery | T-B | -0.117 ± 0.086 | 0/3/0 | -0.661 ± 0.663 | -0.0016 ± 0.0196 | E0.1_new_pairing |
| Grocery | T-S | 0.478 ± 0.181 | 3/0/0 | 0.249 ± 0.983 | -0.0233 ± 0.0169 | E0.1_new_pairing |
| Grocery | P-B | -0.059 ± 0.041 | 0/2/1 | -0.162 ± 0.275 | -0.0068 ± 0.0327 | E0_frozen |
| Grocery | P-S | 0.537 ± 0.242 | 3/0/0 | 0.748 ± 0.481 | -0.0284 ± 0.0211 | E0_frozen |
| Grocery | S-B | -0.595 ± 0.267 | 0/3/0 | -0.910 ± 0.365 | 0.0216 ± 0.0147 | E0_frozen |
| ele-fashion | P-T | -0.286 ± 0.125 | 0/3/0 | -1.915 ± 1.094 | -0.0167 ± 0.0059 | E0.1_new_pairing |
| ele-fashion | T-B | 0.020 ± 0.239 | 2/1/0 | 1.053 ± 0.435 | -0.0023 ± 0.0208 | E0.1_new_pairing |
| ele-fashion | T-S | 0.696 ± 0.068 | 3/0/0 | 3.146 ± 0.319 | -0.0005 ± 0.0176 | E0.1_new_pairing |
| ele-fashion | P-B | -0.266 ± 0.196 | 1/2/0 | -0.862 ± 0.686 | -0.0191 ± 0.0212 | E0_frozen |
| ele-fashion | P-S | 0.409 ± 0.160 | 3/0/0 | 1.231 ± 0.788 | -0.0173 ± 0.0135 | E0_frozen |
| ele-fashion | S-B | -0.675 ± 0.205 | 0/3/0 | -2.094 ± 0.117 | -0.0018 ± 0.0158 | E0_frozen |

P−T pairs parent P and new T checkpoints by identical dataset and seed. P−S is retained as historical context only and cannot substitute for P−T.

## T residual diagnostics

T has 18 selected-checkpoint direction rows (`Text self`, `Visual self`). Descriptive means across direction/checkpoint rows are below; full values are in [target_residual_diagnostics.csv](data/target_residual_diagnostics.csv), alongside frozen P/S rows in [combined_residual_diagnostics.csv](data/combined_residual_diagnostics.csv).

| Variant | Direction rows | RMS(delta) | RMS(delta)/RMS(G) | cos(delta,G) | cos(G+delta,G) | Hidden RMS |
|---|---|---|---|---|---|---|
| T | 18 | 0.3048 | 0.3048 | -0.3928 | 0.9474 | 1.3754 |
| P | 18 | 0.2492 | 0.2492 | -0.3190 | 0.9672 | 1.2164 |
| S | 18 | 0.2310 | 0.2310 | -0.3266 | 0.9704 | 1.2454 |

These are checkpoint diagnostics, not additional performance comparisons. Pair-up and pair-down norms are recorded per direction in the CSV.

Direction summary: T: mean cos(delta,G)=-0.3928, mean cos(G+delta,G)=0.9474; P: mean cos(delta,G)=-0.3190, mean cos(G+delta,G)=0.9672; S: mean cos(delta,G)=-0.3266, mean cos(G+delta,G)=0.9704. T/P/S all have negative mean cosine between correction and target embedding and positive cosine between refined and original embeddings. That broad sign similarity describes the learned corrections; it does not identify which source information caused them.

## GPR coefficient diagnostics

Mean selected-checkpoint coefficients across dataset/seed cells; these are descriptive only.

| Variant | c0 | c1 | c2 | c3 |
|---|---|---|---|---|
| B | 0.24068 | 0.05211 | 0.61539 | -0.10190 |
| T | 0.22802 | 0.06645 | 0.63559 | -0.08356 |
| P | 0.22984 | 0.06681 | 0.63429 | -0.08513 |
| S | 0.22086 | 0.07845 | 0.64662 | -0.07339 |

T rows are in [target_gpr_diagnostics.csv](data/target_gpr_diagnostics.csv); B/P/S rows are the frozen E0 diagnostics in [combined_gpr_diagnostics.csv](data/combined_gpr_diagnostics.csv). Coefficient differences do not reopen GPR attribution.

## Attribution read

Case D pattern: P−T is positive on at least two datasets, while parent P−B is nonpositive on at least two. Correspondence may carry information without establishing a net performance gain for the current PCRR block.

Across dataset means: P−T Accuracy is positive in 2/3 datasets and negative in 1/3. Across the nine paired seeds its Accuracy directions are 5 positive / 3 negative / 1 tied; Macro-F1 is 5/4/0 positive/negative/tied, while CE favors P on 7/9 seeds (2/9 favor T). T−B Accuracy is positive in 2/3 dataset means; T−S is positive in 3/3. This is descriptive evidence on three seeds and fixed validation splits. The earlier 9/9 P−S result establishes that correct source beat the deliberately shuffled source in E0, but does not answer whether P beats active self-refinement.

## Final self-audit (25 items)

1. **Started from required SHA?** Yes: `d1c49583e56af20063ed7274630b2df27cc612e7`.
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

Case D pattern: P−T is positive on at least two datasets, while parent P−B is nonpositive on at least two. Correspondence may carry information without establishing a net performance gain for the current PCRR block. Do not expand the architecture in this phase. Stop here for human review and base any continuation solely on the direct P−T result.
