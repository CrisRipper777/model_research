# PCRR-E0 — Post-GPR Paired Cross-Modal Residual Refinement Screen

## Execution summary

- Parent: `3ef56df39a2d554aaac9a97ae9f6d4943d345f13` on `exp/orci_d0_interaction_alignment_synergy`; experiment branch: `exp/pcrr_e0_postgpr_paired_residual`.
- Validation-only formal cells: 27 expected, 27 complete; datasets Movies/Grocery/ele-fashion, variants B/P/S, seeds 42/43/44.
- Fixed splits: {"Movies": "/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt", "Grocery": "/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt", "ele-fashion": "/hdd1/DataInHere/YHF/data/ele-fashion/split.pt"}. `evaluate_test=false`, `development_no_test=true`; no NC test metrics are present in run artifacts.
- Smoke: see [smoke_status.json](smoke_status.json). No formal LP was run.
- Population SD (`ddof=0`) is reported descriptively over the three paired training seeds; no pseudo-IID p-values are used.

## Validation performance

Accuracy and Macro-F1 are percent; CE is in task units. Cells show mean ± population SD over the three training seeds.

| Dataset | B Acc | P Acc | S Acc | B F1 | P F1 | S F1 | B CE | P CE | S CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Movies | 56.53 ± 0.19 | 56.91 ± 0.06 | 56.14 ± 0.06 | 48.65 ± 0.98 | 50.06 ± 1.58 | 48.93 ± 0.65 | 1.3751 ± 0.0274 | 1.3680 ± 0.0203 | 1.3525 ± 0.0120 |
| Grocery | 83.04 ± 0.23 | 82.98 ± 0.22 | 82.44 ± 0.06 | 76.78 ± 0.44 | 76.62 ± 0.29 | 75.87 ± 0.39 | 0.6573 ± 0.0270 | 0.6505 ± 0.0095 | 0.6789 ± 0.0193 |
| ele-fashion | 87.52 ± 0.16 | 87.25 ± 0.07 | 86.84 ± 0.10 | 74.28 ± 0.17 | 73.42 ± 0.53 | 72.19 ± 0.26 | 0.4253 ± 0.0165 | 0.4062 ± 0.0050 | 0.4235 ± 0.0096 |

## Paired comparisons

Accuracy/F1 deltas are percentage points. Direction counts are across the three paired seeds.

| Dataset | Pair | Accuracy Δ mean ± SD | Acc +/−/= | Macro-F1 Δ mean ± SD | CE Δ mean ± SD |
| --- | --- | --- | --- | --- | --- |
| Movies | P-B | 0.380 ± 0.177 pp | +3/-0/=0 | 1.405 ± 0.602 pp | -0.0071 ± 0.0468 |
| Movies | P-S | 0.770 ± 0.014 pp | +3/-0/=0 | 1.124 ± 1.352 pp | 0.0155 ± 0.0277 |
| Movies | S-B | -0.390 ± 0.191 pp | +0/-3/=0 | 0.280 ± 0.817 pp | -0.0226 ± 0.0200 |
| Grocery | P-B | -0.059 ± 0.041 pp | +0/-2/=1 | -0.162 ± 0.275 pp | -0.0068 ± 0.0327 |
| Grocery | P-S | 0.537 ± 0.242 pp | +3/-0/=0 | 0.748 ± 0.481 pp | -0.0284 ± 0.0211 |
| Grocery | S-B | -0.595 ± 0.267 pp | +0/-3/=0 | -0.910 ± 0.365 pp | 0.0216 ± 0.0147 |
| ele-fashion | P-B | -0.266 ± 0.196 pp | +1/-2/=0 | -0.862 ± 0.686 pp | -0.0191 ± 0.0212 |
| ele-fashion | P-S | 0.409 ± 0.160 pp | +3/-0/=0 | 1.231 ± 0.788 pp | -0.0173 ± 0.0135 |
| ele-fashion | S-B | -0.675 ± 0.205 pp | +0/-3/=0 | -2.094 ± 0.117 pp | -0.0018 ± 0.0158 |

The primary comparison is P−B; P−S assesses the correspondence-matched capacity control; S−B describes the shuffled residual control.

## Residual diagnostics

- Direction-level diagnostics: 36 P/S checkpoint-direction rows.
- Mean `RMS(delta)/RMS(G_target)` across P/S directions and checkpoints: 0.240095.
- Mean `RMS(delta)`: 0.240095. This is a magnitude diagnostic, not a performance criterion.
- Direction-specific values, hidden RMS, cosine, and adapter norms: [residual_diagnostics.csv](data/residual_diagnostics.csv).
- S has 9 recorded dataset-seed permutations; observed fixed-point counts [0] and rates ['0.000000'].

## GPR coefficients

- B: `[0.24068, 0.05211, 0.61539, -0.10190]` (mean over 9 dataset-seed checkpoints).
- P: `[0.22984, 0.06681, 0.63429, -0.08513]` (mean over 9 dataset-seed checkpoints).
- S: `[0.22086, 0.07845, 0.64662, -0.07339]` (mean over 9 dataset-seed checkpoints).

Per dataset/seed coefficients are in [gpr_diagnostics.csv](data/gpr_diagnostics.csv). Differences are descriptive; they do not reopen GPR attribution.

Paired profile shifts across the nine matched dataset-seed checkpoints:

- P−B paired checkpoint coefficient deltas: c0: -0.01083 ± 0.01731 (2+/7-/0= of 9); c1: 0.01471 ± 0.02301 (7+/2-/0= of 9); c2: 0.01890 ± 0.02966 (7+/2-/0= of 9); c3: 0.01678 ± 0.02586 (7+/2-/0= of 9).
- S−B paired checkpoint coefficient deltas: c0: -0.01982 ± 0.01497 (1+/8-/0= of 9); c1: 0.02634 ± 0.02259 (8+/1-/0= of 9); c2: 0.03123 ± 0.02766 (8+/1-/0= of 9); c3: 0.02852 ± 0.02366 (8+/1-/0= of 9).

## P checkpoint interventions

The P checkpoints were evaluated with the residual disabled and with five deterministic residual-source permutations each. These diagnostics are not retrained comparisons.

| Dataset | Intervention | Raw observations | Accuracy % | Accuracy Δ vs normal | Acc +/−/= seeds | Macro-F1 Δ | CE Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Movies | residual_off | 3 | 49.98 ± 1.56 | -6.929 ± 1.528 pp | +0/-3/=0 | -25.137 ± 2.381 pp | 0.2542 ± 0.0843 |
| Movies | source_node_shuffle | 15 | 54.51 ± 0.36 | -2.398 ± 0.304 pp | +0/-3/=0 | -3.940 ± 0.819 pp | 0.0936 ± 0.0276 |
| Grocery | residual_off | 3 | 81.67 ± 0.05 | -1.308 ± 0.183 pp | +0/-3/=0 | -2.175 ± 0.487 pp | 0.0475 ± 0.0068 |
| Grocery | source_node_shuffle | 15 | 82.25 ± 0.27 | -0.724 ± 0.060 pp | +0/-3/=0 | -1.039 ± 0.274 pp | 0.0266 ± 0.0120 |
| ele-fashion | residual_off | 3 | 85.81 ± 0.02 | -1.446 ± 0.084 pp | +0/-3/=0 | -4.847 ± 0.505 pp | 0.1152 ± 0.0119 |
| ele-fashion | source_node_shuffle | 15 | 86.16 ± 0.10 | -1.088 ± 0.145 pp | +0/-3/=0 | -3.321 ± 0.844 pp | 0.0728 ± 0.0179 |

Full rows and seed-level repeats: [intervention_by_run.csv](data/intervention_by_run.csv); grouped summary: [intervention_summary.csv](data/intervention_summary.csv).

## Interpretation

Case C leaning: correspondence control P−S trends positive on at least two datasets, while net P−B gain is not broad; this is mechanism evidence without a demonstrated net architecture gain.

Read the paired P−B and P−S rows together. A checkpoint's response to source shuffling does not substitute for either retrained comparison. Dataset-specific signs and Macro-F1/CE trade-offs should remain visible; there are only three paired training seeds per dataset and one fixed split per dataset.

## Final self-audit (25 items)

1. **Started from required base?** Yes: `3ef56df39a2d554aaac9a97ae9f6d4943d345f13` was the verified parent SHA.
2. **Used another experiment branch?** No merge/cherry-pick from other experiment branches.
3. **Ran/read NC test?** No test split evaluation was enabled; artifacts are marked validation-only and contain no `test_*` metrics.
4. **Changed dataset split?** No; the three declared split files were used.
5. **Does B regress to D0-B?** Yes, model-level eval regression is checked at `atol=1e-6`; see test result in smoke status.
6. **Do P/S equal B at initialization?** Yes, the zero-initialized final pair projection gives exact zero residual; same-seed eval and train-mode RNG-reset equality are tested.
7. **Is `pair_up` zero-initialized?** Yes, weight and bias are exactly zero.
8. **Extra residual-block train dropout?** No; the pair response has no dropout.
9. **Are P/S parameter count and initialization matched?** Yes; all variants build the same modules, and same-seed state dictionaries are tested bitwise.
10. **Does S shuffle only the residual source?** Yes; permutation is applied only to the source input passed to the pair response.
11. **Does S late fusion keep correct node pairing?** Yes; late fusion concatenates each node's own refined text and visual branches.
12. **Does P−B show retrained gain?** Not broadly: mean accuracy is +0.380 pp on Movies, −0.059 pp on Grocery, and −0.266 pp on ele-fashion.
13. **Does P−S support independent correspondence value?** Descriptively yes: mean accuracy delta is positive on all three datasets and all three paired seeds; this does not establish net gain over B.
14. **Does S−B expose capacity effects?** Yes; mean accuracy is negative on all three datasets, so the shuffled residual control does not show a positive generic capacity effect.
15. **Final residual magnitude?** Mean ratio is 0.240095; all cellwise values are in the diagnostics CSV.
16. **Do P checkpoints rely on the residual?** Yes in validation: residual-off reduces mean accuracy by 6.929 pp on Movies, 1.308 pp on Grocery, and 1.446 pp on ele-fashion.
17. **Does source-node shuffle harm P checkpoints?** Yes descriptively: five-repeat mean accuracy changes are −2.398 pp, −0.724 pp, and −1.088 pp on Movies, Grocery, and ele-fashion.
18. **Do interventions agree with retrained P−S?** Both favor correct correspondence directionally, but intervention deltas are checkpoint reliance diagnostics and cannot replace retrained P−S.
19. **Did residual systematically change GPR coefficients?** Paired P−B and S−B means, SDs, and direction counts are shown above; any profile shifts are descriptive, not causal evidence that the adapter changed GPR.
20. **Dataset-specific regime?** Yes: net P−B accuracy gain appears only on Movies, while P−S is positive across all three datasets.
21. **Accuracy/F1/CE trade-off?** All three deltas are reported in paired CSVs and tables.
22. **Evidence for next stage?** The current block lacks a broad net P−B gain; do not advance it as a performance architecture based on E0.
23. **What should be retained if proceeding?** Retain the correspondence insight as a mechanism hypothesis; do not carry the residual block forward as an established gain.
24. **LP smoke protocol?** B/P only, 2 epochs and at most 2 train batches, `[5,5,5]`, LinkNeighborLoader and global_eid positive-edge removal checked; no LP test evaluation.
25. **Claims beyond evidence?** None intended: results are validation-only, three paired seeds on fixed splits, and checkpoint interventions are reliance diagnostics.

## Recommendation

Case C leaning: correspondence control P−S trends positive on at least two datasets, while net P−B gain is not broad; this is mechanism evidence without a demonstrated net architecture gain. Do not advance PCRR as a performance module yet. Retain the correct-correspondence signal as a mechanistic observation for human review, and stop the experimental line here.
