# MAG Information Utility V1 — Decision Memo

This memo reports descriptive evidence from the frozen 3-dataset × 5-treatment × 3-seed experiment. It does not claim significance or generalize beyond this probe and these datasets.

## Evidence by question

- **Physical propagation (I1-I0)**: Accuracy paired mean deltas are Movies: +1.58 pp, Grocery: +3.33 pp, ele-fashion: +0.22 pp; positive seed differences: 9/9. Macro-F1 and CE paired summaries are available in `data/paired_information_gains.csv`. Treat a mixed sign pattern as dataset- or seed-dependent evidence, not a universal effect.
- **Cross-modal physical processing (I2-I1)**: Accuracy paired mean deltas are Movies: +0.30 pp, Grocery: +0.54 pp, ele-fashion: +0.09 pp; positive seed differences: 8/9. Macro-F1 and CE paired summaries are available in `data/paired_information_gains.csv`. Treat a mixed sign pattern as dataset- or seed-dependent evidence, not a universal effect.
- **Opening real semantic channels (I3-I2)**: Accuracy paired mean deltas are Movies: -0.05 pp, Grocery: -0.06 pp, ele-fashion: +0.17 pp; positive seed differences: 5/9. Macro-F1 and CE paired summaries are available in `data/paired_information_gains.csv`. Treat a mixed sign pattern as dataset- or seed-dependent evidence, not a universal effect.
- **Real versus degree-matched random topology (I3-I4)**: Accuracy paired mean deltas are Movies: +0.27 pp, Grocery: +0.08 pp, ele-fashion: +0.23 pp; positive seed differences: 8/9. Macro-F1 and CE paired summaries are available in `data/paired_information_gains.csv`. Treat a mixed sign pattern as dataset- or seed-dependent evidence, not a universal effect.

## Simple-probe capacity calibration

- **Movies**: I1−R0 Accuracy = -1.86 ± 0.72 pp; I2−R0 = -1.56 ± 0.83 pp. R0 has a different, generally larger model, so a gap limits how far this probe's relation findings should be generalized.
- **Grocery**: I1−R0 Accuracy = -1.77 ± 0.57 pp; I2−R0 = -1.23 ± 0.73 pp. R0 has a different, generally larger model, so a gap limits how far this probe's relation findings should be generalized.
- **ele-fashion**: I1−R0 Accuracy = -0.06 ± 0.13 pp; I2−R0 = 0.03 ± 0.21 pp. R0 has a different, generally larger model, so a gap limits how far this probe's relation findings should be generalized.

## What this supports for future MAG graph experts

Use the observed contrasts to decide which information paths deserve a capacity-matched follow-up on a stronger physical-graph backbone. A stable physical gain would support experts that process physical neighborhoods by modality; a repeatable I2 gain would motivate testing explicit cross-modal neighborhood processing; a repeatable I3−I4 gain would justify retaining semantic topology as a candidate relation. None of these probe results alone establishes a general architectural rule.

## Capability calibration and caveats

Compare I0–I4 with historical V7B.1 R0 Raw using `data/r0_reference_comparison.csv`. R0 has different parameter count and architecture, so a gap bounds how strongly this simple probe can support claims about semantic utility. Random graphs preserve the exact degree sequence and edge count, but their connected components and physical overlap differ from the real KNN views. Only one random topology per dataset and source was evaluated.

No Test labels were used. Validation subgroup analyses are descriptive and do not create filtering rules. Small groups are marked in `data/node_group_metrics.csv` and `data/conditional_error_analysis.csv`.
