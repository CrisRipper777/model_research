# MR-CGE V1.3 Decision Handoff

## Review state

The V1.3 Validation screen is complete and frozen for independent review. Work
branch: `exp/mrcge_v13_view_cross_screen`. Required starting commit:
`a7930442e85ff9450fded54d5869915b434c7431`. The implementation freeze commit
is `be5ec6b45d4ee7125d402b2751baa2169e546e50`; the corrected gradient-audit
fix is `169cddd72aaaba00a5ce3fd25657981dd40c5e27`; the formal preflight/run
source commit recorded in each successful row is
`78284c82308b117e98f6c2ef59bf5acbe13369bf`. Later commit
`1216376dd234679313c896c61ef3480848b584c8` adds post-training diagnostics and
strict CPU reload auditing only. The final repository HEAD is the commit that
contains this report and the completed records; consult `git rev-parse HEAD`
after checkout.

## Authorized screen and outcome

- 12/12 authorized Dataset × Variant × Seed cells completed.
- Formal ledger also preserves 2 GPU-guard-aborted Attempts and 7 blocked
  starts. Each later succeeded; there are no incomplete unique cells.
- Corrected-source GPU preflight passed 12/12.
- No Test labels were evaluated, no LP work was run, and no A+B combination,
  Toys, seed 44, or follow-on version was started.
- V1.2 model, Trainer, config, and confirmation runner hashes match the frozen
  starting state.
- All 24 formal Accuracy/F1 checkpoints passed hash, metadata, and Validation
  reload checks in the run path. A posthoc CPU audit strictly reloaded all 12
  Best-Accuracy checkpoints and reproduced Validation Acc/F1/CE without
  changing RNG.

## Main observations for reviewer

- V1.3-A `joint_view` has positive two-seed mean paired Acc/F1 changes on
  Movies (+0.555/+1.593 pp) and Grocery (+0.820/+0.430 pp), but ele-fashion is
  −0.102/−0.310 pp.
- V1.3-B `sender_private` has small mean Accuracy changes (Movies −0.075 pp,
  Grocery +0.059 pp, ele-fashion +0.036 pp), with Macro-F1 changes
  +0.725/−0.226/+0.285 pp respectively.
- The selected A view strengths moved away from 1, including low Grocery
  seed-43 Semantic scales. The diagnostics confirm the mechanism operated;
  they do not show causal benefit or true graph reliability.
- Early Cross calibrated/neighbor RMS is 1 by the existing calibration
  definition. Raw Cross and Late residual summaries are descriptive and do
  not determine whether the incoming signal is useful or redundant.
- This screen does not nominate a winner. Historical V1.2 rows are from a
  distinct campaign, and there are only two seeds per dataset here.

Detailed per-seed and mean values, Best-F1 results, compute, guard history,
and caveats are in `SCREENING_REPORT.md` and `PREFLIGHT_REPORT.md`.

## Key files

- Implementation freeze: `DESIGN_FREEZE.md`
- Source and correctness audit: `IMPLEMENTATION_AUDIT.md`
- GPU/data/resource history: `PREFLIGHT_REPORT.md`
- Results and interpretation: `SCREENING_REPORT.md`
- Formal runner: `../../scripts/run_mrcge_v13_screen.py`
- Analyzer: `../../scripts/analyze_mrcge_v13_screen.py`
- Model: `../../src/models/mrcge_v13.py`
- Tests: `../../tests/test_mrcge_v13.py`
- Immutable attempt rows: `data/runs.csv`
- Epoch history: `data/epoch_history.csv`
- Per-seed paired comparison: `data/paired_differences.csv`
- Dataset descriptive means: `data/dataset_summary.csv`
- Resource snapshots: `data/resource_summary.csv`, `data/resource_by_cell.csv`
- View measures: `data/view_diagnostics.csv`,
  `data/weighted_contribution_diagnostics.csv`
- Preflight and frozen fingerprints: `data/preflight.json`,
  `data/frozen_baseline.json`
- Failures, checkpoint audit, and counts: `data/failures.json`,
  `data/checkpoint_audit.json`, `data/manifest.json`

Large model checkpoints, worker logs, and GPU snapshots remain on the server
under `outputs/mrcge_v13_screen/`; checkpoints are not part of the Git data
commit.

## Independent review request

Please review the implementation, source-to-run provenance, saved result
records, and the descriptive conclusions before selecting any direction.
Specifically inspect the A collaborative update and paired-dropout zero-Cross
path, B gradient flow through the source Private state, V1.2 source hashes,
the trainer's dual checkpoint selection, the Best-F1 macro-F1 class universe,
GPU attempt accounting, and the seed-paired Validation table. Do not treat
these two-seed screen means as a significance result or causal decomposition.

No additional experiment or model change is queued by this handoff.
