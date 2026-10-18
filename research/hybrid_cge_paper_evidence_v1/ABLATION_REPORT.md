# Ablation report

## Implementation checks

The frozen H2 implementation is unchanged. Mechanism-level call-path tests pass for A1–A4. Real-data Forward/Backward preflights and the 60 formal run matrix are in progress; this report will be filled from `data/runs.csv` after the stage gate passes.

## Validation results

Results will report Validation Accuracy and Macro-F1 per Dataset × Seed, paired changes against the matched historical H2-Frozen reference, sample mean/SD, and exact completed/failed cell counts. All training uses Test labels masked to `-100`; Test is not evaluated.

## Interpretation

No mechanism claim is made until the paired matrix is complete. A negative or inconsistent ablation result will be retained as observed.
