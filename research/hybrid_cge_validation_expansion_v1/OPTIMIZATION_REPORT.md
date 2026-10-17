# Hybrid-CGE configuration optimization

H2-Frozen is the historical H2 result and was not rerun. New R1/R2 runs use this branch's frozen runner, fixed data/cache/splits and paired seeds. The table is Validation-only.

| Condition | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 | Train−Val Accuracy gap | Train−Val F1 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2-Frozen | 9/9 | 76.12% | 66.80% | 0.8001 | 68.02% | 8.49% | 13.90% |
| H2-R1 | 9/9 | 75.94% | 67.03% | 0.8077 | 67.99% | 9.19% | 14.58% |
| H2-R2 | 9/9 | 75.91% | 66.98% | 0.8152 | 67.62% | 9.41% | 15.10% |

**Selected global configuration:** `H2-Frozen`. Mean accuracy determines eligibility; if no candidate exceeds H2-Frozen, H2-Frozen is retained. The paired seed-by-dataset deltas are in `data/optimization_paired_differences.csv`.

R1 changes only model dropout (.20 → .35) while keeping AdamW weight decay and private auxiliary CE at the H2 values. R2 changes dropout, weight decay and private auxiliary CE together; its individual contributions and their interaction cannot be identified from this bundled comparison. Three seeds support descriptive paired comparisons, not strong inferential claims.

## Paired deltas against historical H2-Frozen

Across the nine matched Dataset × Seed cells, mean paired deltas (candidate minus H2-Frozen) were:

| Condition | Best-Accuracy Val Acc | Same-checkpoint Macro-F1 | Val CE | Best-F1 Macro-F1 | Train−Val Acc gap | Train−Val Macro-F1 gap | Train−Val CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2-R1 | −0.181 pp | +0.226 pp | +0.0077 | −0.028 pp | +0.705 pp | +0.681 pp | −0.0262 |
| H2-R2 | −0.206 pp | +0.182 pp | +0.0151 | −0.403 pp | +0.917 pp | +1.204 pp | −0.0325 |

The registered primary metric therefore retains H2-Frozen. R1 has a small average same-checkpoint Macro-F1 gain, but does not improve Accuracy and slightly widens the mean accuracy and Macro-F1 gaps. R2 does not improve the primary or secondary aggregate outcomes and widens both gaps. The negative Train−Val CE-gap deltas mean the Train CE advantage increased on average. R1’s comparison changes dropout alone, so it is evidence about that full training configuration; R2 simultaneously changes dropout, weight decay and auxiliary-loss weight, so these data cannot assign a separate effect to weight decay, auxiliary loss or their interaction. Per-cell deltas, including CE-gap changes and checkpoint hashes, are retained in `data/optimization_paired_differences.csv`.

Historical Pilot scores are reference evidence, not exact acceptance thresholds. This study compares internal validation outcomes and does not inspect Test labels or metrics.
