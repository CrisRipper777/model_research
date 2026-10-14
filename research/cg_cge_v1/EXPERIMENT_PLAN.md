# CG-CGE V1 Experiment Plan

## Conditions and run order

| Condition | Change | Seeds |
|---|---|---|
| B0 `A2_cross_source` | Frozen external View-CGE V1 reference; do not retrain | 42, 43, 44 |
| B1 `B1_complementary` | Complementary completion, no active Role-FiLM | 42, 43, 44 |
| B2 `B2_conditioned_experts` | B1 with Semantic/Complementary Role-FiLM | 42, 43, 44 |

Formal work is 18 new runs. Complete the six seed-42 cells first and inspect finite histories, gradients, checkpoint reloads, label masking, paired initialization, memory and timing. If these integrity checks pass, complete seeds 43 and 44. A low Validation score alone is never a stop condition. Keep any failed attempt and retry it into a new attempt directory; never replace its record.

Within each run, fixed settings are `unified_full_graph_nc_v1`, official split, cached V7B.1 K=8 operators, AdamW lr `1e-3`, weight decay `1e-4`, dropout `0.2`, max 300 epochs, patience 30, minimum epoch 30, gradient clip 1.0, main-head best Validation Accuracy and separate best Validation Macro-F1. Test labels are replaced with `-100` before entering the trainer, and `evaluate_test=false`.

## Pairing and B0 gate

Before training, `preflight_cg_cge_v1.py --cpu-only` verifies each Dataset × Seed split and compares it to the frozen View-CGE manifest. The comparison includes official split fingerprint, feature fingerprint, physical graph fingerprint, exact semantic cache SHA256, candidate and V7B.1 fingerprints, plus `unified_full_graph_nc_v1`. B0 eligibility is stored separately for each Dataset × Seed; a dataset's three-seed paired comparison is reported only if all three cells match.

For every Dataset × Seed, B1 and B2 must have identical model-state and external-head initialization hashes. The same common parameter tree is used; B1 Role-FiLM has zero values and is frozen/bypassed, while B2 begins as exact identity and can update.

## Preflight gates

1. Focused CPU unit tests cover operator direction, per-receiver softmax, finite nonzero Q/K/V gradients, empty and single-source behavior, gate initialization, condition initialization, auxiliary CE and checkpoint reload.
2. Full repository pytest passes with CUDA hidden.
3. CPU data/init preflight passes for all datasets and seeds.
4. GPU owner/idle guard confirms the requested physical GPU has no other compute process or unattributed memory before real-data forward/backward checks and before each formal run. No other process is stopped or shared.
5. Full-graph one-step B1/B2 GPU audit checks finite loss/gradients, nonzero required mechanism gradients, evaluation RNG preservation, elapsed step time and allocated/reserved peaks for all three datasets.
6. Only after code review and the preflight gates, record the Implementation Freeze Commit. Formal runner source SHA is recorded separately after the freeze.

## Evidence written by the runner

`data/runs.csv` contains per-cell metrics, parameter counts, resource values, checkpoint paths/hashes, split/cache fingerprints and Test/RNG flags. `epoch_history.csv` contains main CE, unimodal CE, existing balance auxiliary loss, train/validation Accuracy/F1/CE and timing. Separate files preserve classwise metrics, confusion matrices, mechanism diagnostics, resource summaries, first-step gradient audits, checkpoint integrity, failures, and the seed-42 gate. Checkpoints and raw data remain in ignored server-local `outputs/cg_cge_v1/`.

## Analysis

Report Best-Accuracy and Best-F1 Validation Accuracy, Macro-F1 and CE, plus Train/Validation gaps. Report per-seed B1−B0, B2−B0 and B2−B1 only when B0 is strictly paired; summarize mean, sample SD and positive/negative/zero seed counts. Inspect cross-candidate raw/calibrated RMS, source P/S/None weights, gate saturation, residual-to-unimodal ratio, attention entropy, expert selection and Role-FiLM updates. None of these mechanism signals is itself evidence of task improvement. No Test metric is permitted in outputs or narrative.
