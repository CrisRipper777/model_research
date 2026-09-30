# P1.1 + P1.2: robustness and fixed-operator rescue

This study audits the P0/P1 validation edge evidence on raw ΔCE, then tests whether the same frozen semantic message can change utility sign under three fixed relation operations. It covers Movies, Grocery, and ele-fashion, seeds 42–44, node classification only, and validation targets only.

## Protocol

- P1.1 reconstructs `U_raw = U_relative × CE_full = CE_removed − CE_full` from the complete P0 edge tables. It reports H1/H2/H3 using raw ΔCE, compares relative utility, contrasts frozen raw cosine with P0 task-projected cosine, and repeats H1 for destination degree ≥5.
- P1.2 loads each P0 semantic-only checkpoint and freezes both modality projections and LayerNorms. The same `H0 = [h_text || h_visual]`, graph, and validation-directed physical edge set feed all three operators.
- `smooth` aggregates `h_j`; `absdiff` aggregates `abs(h_j − h_i)`; `product` aggregates `h_i * h_j`. All use uniform means. Only a new `Linear(4 × 128, classes)` head is trained with the P0 AdamW and early-stopping settings.
- Per-message raw utility is `CE_removed − CE_full`. Removal subtracts `O(i,j)/d_i` with the original degree unchanged. Relative utility is also saved. Head accuracy, Macro-F1, and CE are sanity metrics, not the primary conclusion.
- There is no test split evaluation, LP, router, MoE, or learned operator transform.

## Reproduction

Activate `yhf_env`, then run from the repository root:

```bash
python -m pytest -q tests/test_p11_p12_operator_rescue.py
python scripts/run_p11_p12_operator_rescue.py --smoke --device cuda:0
python scripts/run_p11_p12_operator_rescue.py --datasets Movies Grocery ele-fashion --seeds 42 43 44 --device cuda:0
python scripts/analyze_p11_p12_operator_rescue.py --p0-output-dir outputs/p0p1_propagation_heterogeneity --output-dir outputs/p11_p12_operator_rescue --research-dir research/p11_p12_operator_rescue --datasets Movies Grocery ele-fashion --seeds 42 43 44
```

The Movies/42 smoke gates the full run and checks the frozen encoder, common H0, operator message formulas, self-loop exclusion, fixed denominator, common physical edge set, fast/brute logits, and the validation-only split accessor.

## Artifacts

- `report.md`: combined P1.1/P1.2 results and next-step recommendation.
- `run_manifest.json`: source provenance, environment, commands, checks, and output inventory.
- `p11/`: raw ΔCE robustness tables and report.
- `p12/`: probe metrics, operator utility/rescue summaries, transition matrices, modality disagreement, and a deterministic edge sample.
- `figures/`: the four requested PNG figures plus editable PDF/SVG exports. `figures/qa/` preserves alignment, text-size, collision, and source-preflight results; the TIFF export is in the ignored `outputs/p11_p12_operator_rescue/figure_exports/` directory.
- `outputs/p11_p12_operator_rescue/`: ignored full edge tables and linear-head checkpoints. Full per-edge results are aligned by dataset, seed, source, destination, and modality across operators.

The fixed primitives provide evidence about propagation-function heterogeneity only for the three operations tested. The report keeps that scope explicit.
