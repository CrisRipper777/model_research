# MR-CGE V1.2 screening report

Status: **halted during Stage 3 by the shared-GPU ownership guard**. This is a partial screening report, not a 12-cell comparison. Training source freeze: `9cfe6e491d2ea2a58de01101af5032199c37cf14`. Raw rows, full epoch history, failures, and resource records are in `data/`.

## Completed cells

Both completed Movies cells used seed 42, attempt 1, source SHA `9cfe6e491d2ea2a58de01101af5032199c37cf14`, shared initial common-state and classifier hashes, Best Accuracy and Best Macro-F1 checkpoints, and Validation RNG audits. Their checkpoint hashes matched the recorded files. Test evaluation was false.

| Dataset / condition | Best Accuracy epoch | Best Validation Accuracy | Best Macro-F1 at Accuracy checkpoint | Best F1 epoch | Best F1 checkpoint Macro-F1 | Allocated / reserved peak |
|---|---:|---:|---:|---:|---:|---:|
| Movies / S_repaired | 86 | 53.3293% | 42.8662% | 74 | 45.5657% | 1.69 / 2.29 GiB |
| Movies / F_sinit | 72 | 53.6293% | 43.1021% | 79 | 45.5607% | 2.01 / 2.57 GiB |

On this one dataset and seed, F_sinit minus S_repaired was **+0.30 percentage points** in Best Accuracy and **+0.24 points** in Macro-F1 at the Best Accuracy checkpoint. Best-F1 checkpoint scores were nearly equal (45.57% vs 45.56%). These are descriptive pilot values only and do not support a winner selection.

## Guard stop and remaining cells

Movies/M_repaired attempt 1 was created, but the live guard observed new external compute PID `4030361` on GPU 1 at `2026-10-25T05:57:53Z`. It aborted only this study's child. The attempt log contains no training epoch or checkpoint. The runner recorded the attempt as failed and the cell as `blocked_resource`. The stage was then stopped as required by the ownership rule.

The remaining nine core cells and two optional ele-fashion M cells are recorded as `blocked_resource`, attempt 0, with no training started. They are not completed results. No Fashion training was attempted.

## Findings and limits

- Movies/S and Movies/F both produced finite training histories, reloaded Validation checkpoints, matching checkpoint hashes, and unchanged Validation RNG.
- The V1.1/V1.2 Fashion M preflight peak improvement is recorded in `PREFLIGHT_REPORT.md`; no formal Fashion metric exists.
- M_repaired and M_anchor have no formal score in this partial screen. No conclusion about their accuracy, anchor effect, routing use during training, or closeness to H2/DiP is supported.
- The historical Movies V1.1 values are contextual only. This incomplete single-seed slice is not a matched four-condition study.
- The guard event occurred despite ample free memory. The reason to stop was the new process, not an OOM or low-memory event.

## Test and history boundary

No Test label or metric was read. Test remained disabled and masked. Existing V1.1 files and results were not modified. No other user's GPU process was signalled or stopped.
