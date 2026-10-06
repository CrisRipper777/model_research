# R³-MAG v1 prototype reports

The tracked reports summarize the first complete trainable implementation. The architecture uses a learned signed global structural prior, DCT-initialized response atoms, dense shared/private routing, and preserve-or-adapt reliability. No OT, contrastive routing, load balancing, orthogonality loss, replay, meta-learning, rewiring, relation-neighbor encoder, or PC-Conv basis is included.

## Protocol

- Datasets: Movies and Grocery use data seed 42; ele-fashion uses its official split. The internal partition uses seed 20261006 and is hash-checked against the H1 per-run reports.
- DevTrain is HostTrain union ResponseTrain (90% of original train labels). Original validation is used for early stopping and checkpoint selection. Audit is held out for prototype evaluation after every variant checkpoint is selected.
- Stage 1 trains G0 once per dataset × model seed. G0-FT/G1/G2/G3/G4 start from that exact checkpoint and classifier state. Stage 2 uses the same epoch cap and early stopping policy across variants.
- Best validation accuracy selects checkpoints; exact accuracy ties use lower validation CE. Audit interventions do not select checkpoints.
- Test labels are never indexed and no test metrics are computed. **TEST SPLIT UNTOUCHED.**
- The previous H2 QA initialization-order bug is fixed and passed a Movies seed42 H2-A smoke; the full H2 screen was not rerun.
- Absolute task numbers are not compared directly with H1/H2 host results because those hosts used 80% HostTrain, while this prototype trains on 90% DevTrain.
- Means and standard deviations in the aggregate report use population standard deviation over seeds 42, 43, and 44.

## Outputs

- `aggregate_report.md`: task performance, reusable bank, shared/private, reliability, interventions, and protocol boundaries.
- `task_metrics.csv`: per dataset × seed × variant validation and Audit CE/Accuracy/Macro-F1.
- `mechanism_metrics.csv`: atom coefficients/cosines, route use/entropy, path diagnostics, intervention deltas, adaptation utility, and reliability quartiles.
- `qa_report.json`: protocol, checkpoint reuse, split, and intervention QA.
- `per_run/`: complete run records for each dataset and seed.

Checkpoints, work logs, and intermediate run JSON stay under ignored `outputs/r3mag_prototype/v1/`.
