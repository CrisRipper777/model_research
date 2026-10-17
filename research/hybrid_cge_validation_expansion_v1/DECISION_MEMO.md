# Decision memo

## Registered selection

Selected global Hybrid-CGE configuration: H2-Frozen. The registered primary metric was mean Validation Accuracy across Movies, Grocery and ele-fashion, with historical H2-Frozen retained unless a new condition exceeded it.

- H2-Frozen: 76.12% mean Accuracy, 66.80% same-checkpoint Macro-F1, 0.8001 mean Validation CE.
- H2-R1: 75.94% mean Accuracy, 67.03% same-checkpoint Macro-F1, 0.8077 mean Validation CE.
- H2-R2: 75.91% mean Accuracy, 66.98% same-checkpoint Macro-F1, 0.8152 mean Validation CE.

H2-Frozen wins the primary Accuracy comparison and has the lowest CE. R1 has a small +0.23 percentage-point same-checkpoint Macro-F1 mean difference, but its Best-F1 checkpoint mean is effectively tied with H2-Frozen (67.99% vs 68.02%); these descriptive three-seed results do not establish an F1 improvement. R2 does not lead either metric. Both new configurations have wider mean Train−Validation Accuracy and Macro-F1 gaps. R1 changes dropout only; R2 bundles dropout, weight decay and auxiliary CE changes, so R2 cannot identify individual hyperparameter effects. See OPTIMIZATION_REPORT.md and data/optimization_paired_differences.csv.

## Cross-dataset and baseline comparison

The global H2-Frozen setting transfers to both first-use datasets without tuning: 80.33% Validation Accuracy / 77.47% Macro-F1 on Toys, and 96.60% / 93.02% on Reddit-S. There is no collapse relative to the four project baselines. These are Validation results only.

| Dataset | Hybrid-CGE Accuracy | Best baseline Accuracy | Margin | Hybrid-CGE Macro-F1 | Best baseline Macro-F1 | Margin |
|---|---:|---|---:|---:|---|---:|
| Movies | 56.78% | MMGCN 55.69% | +1.09 pp | 49.78% | GraphSAGE 48.46% | +1.32 pp |
| Grocery | 83.86% | GraphSAGE 83.32% | +0.54 pp | 75.97% | GraphSAGE 76.23% | −0.26 pp |
| ele-fashion | 87.72% | MLP 88.10% | −0.38 pp | 74.65% | MLP 77.40% | −2.75 pp |
| Toys | 80.33% | MMGCN 78.86% | +1.47 pp | 77.47% | GraphSAGE 76.08% | +1.39 pp |
| Reddit-S | 96.60% | MMGCN 96.08% | +0.52 pp | 93.02% | MMGCN 92.18% | +0.84 pp |

Values are three-seed means; per-seed values and sample SDs are in BASELINE_REPORT.md and data/runs.csv. Differences are descriptive, not significance tests. The baseline adapters and their limits are documented in BASELINE_REPORT.md; these results are not exact reproductions of all official benchmark implementations.

## Answers to the study questions

- H2-Frozen remains best on the registered primary Accuracy aggregate. R1 is marginally higher on same-checkpoint Macro-F1 mean, but not on Best-F1 checkpoint mean; R2 is behind H2-Frozen on both mean Accuracy and Macro-F1.
- H2-Frozen is the selected shared training configuration for the main model and both new datasets.
- Validation results support useful transfer to Toys and Reddit-S across three seeds. No serious dataset-specific collapse was observed.
- Accuracy/Macro-F1 trade-offs are favorable on Movies, Toys and Reddit-S, close on Grocery, and unfavorable on ele-fashion. In particular, MLP exceeds Hybrid-CGE on ele-fashion by 0.38 pp Accuracy and 2.75 pp Macro-F1; report that limitation rather than hiding it behind aggregate scores.
- The current evidence supports freezing the Hybrid-CGE body and preparing paper-stage evaluation, subject to independent review and the documented adaptation caveats. It does not support an SOTA claim.
- If further work is approved, first inspect Validation class-wise errors and training/feature alignment on ele-fashion, alongside the already recorded Train−Validation gaps. These results alone do not show whether optimization, features or cross-modal structure caused the gap; they do not justify adding a new architecture component.

## Completion and provenance

Completed runs: 84/84 (18 optimization + 6 generalization + 60 baselines). All 84 rows report finite Train/Validation metrics, Test evaluation disabled and Validation RNG unchanged. Best-Accuracy checkpoint files and hashes were verified for all 84 rows; all 69 available Best-F1 checkpoint files and hashes were verified. No Test labels or metrics were used for selection or reporting.

Two preparation failures are preserved in data/failures.json; both occurred before training and were resolved. No formal training cell failed. The runner, source/config/data/cache fingerprints, raw rows and checkpoint hashes are in data/runs.csv and adjacent data artifacts. Checkpoint binaries, original features and graph/cache files remain outside Git.
