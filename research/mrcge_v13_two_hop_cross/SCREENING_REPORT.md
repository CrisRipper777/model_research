# MR-CGE V1.3 Two-Hop Cross Controlled Screening

## Outcome

No model GPU preflight passed, so registered Stage 1 training did not start. Stage 2 was not authorized and the Validation promotion gates were not evaluated. This report records an execution block, not a model-performance result.

## Historical Hybrid-CGE H2 context

These fixed-split Validation rows are historical context only; their checkpoints were not loaded or re-evaluated.

| Movies seed | Accuracy | Macro-F1 | CE |
|---:|---:|---:|---:|
| 42 | 0.568686 | 0.502572 | 1.403075 |
| 43 | 0.570486 | 0.494333 | 1.344520 |
| 44 | 0.569286 | 0.488450 | 1.336219 |

## Execution record

- CPU protocol and initialization checks passed; see [PREFLIGHT_REPORT.md](PREFLIGHT_REPORT.md).
- Six GPU 1 baseline attempts failed in the existing ownership guard because `nvidia-smi` could not reliably communicate with the NVIDIA driver. A separate pre-launch guard once read GPU 1 at 24,246 MiB free with no compute PID, but the runner could not reproduce a valid baseline and no GPU model worker launched.
- Training runs: 0/6 Stage 1; 0/4 conditional Stage 2.
- Delta metrics and promotion decision: not evaluated.
- Test evaluation and link prediction: disabled and not run.

No statistical or generalization claim can be made from this blocked screen.
