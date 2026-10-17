# Hybrid-CGE V1.1 Experiment Plan

## Frozen matrix

| Condition | Description | Seeds |
|---|---|---|
| H1 `H1_hybrid` | Private state + Early Physical Cross collaboration + Late Semantic/None completion; no Late Cross Dropout | 42, 43, 44 |
| H2 `H2_hybrid_plus` | H1 plus Late Same-node candidate and training-only per-node Late Cross Dropout 0.15 | 42, 43, 44 |

Datasets: Movies, Grocery, ele-fashion. Total: 18 new complete runs. Historical A2 and B1 are not retrained. Training settings: full-graph NC, official split, frozen V7B.1 cache K=8, AdamW lr 1e-3, weight decay 1e-4, dropout 0.2 (in addition to H2's explicitly scoped late residual dropout), up to 300 epochs, patience 30, minimum epoch 30, gradient clipping 1.0, best-Validation-Accuracy and best-Validation-Macro-F1 checkpoints. Private auxiliary CE weight is 0.3. Test evaluation is disabled and Test labels are replaced by -100 before trainer entry.

## Pre-training checks

1. Create this branch from exact parent `a7040189ee1237c80ebe2203ca3d204be4c43462`; record branch and source revisions. Preserve every old artifact.
2. Commit the written mathematical design before implementation. Complete code review, focused tests and full repository pytest before the Implementation Freeze Commit.
3. Run CPU data/init preflight for all nine Dataset×Seed cells. Compare official split, raw and projected features, physical graph, exact semantic cache SHA256, candidate and V7B.1 fingerprints against both frozen reference studies. Store per-reference per-cell eligibility; only aggregate comparisons whose full three-seed blocks pass.
4. Build H1/H2 from the same Dataset×Seed initialization and require exact initial model and classifier hashes. Verify masked Test labels and untouched Train/Validation labels.
5. On an idle, isolated physical GPU, run real-data forward/backward/update preflight for every dataset and condition. Require finite loss/gradients/parameters, nonzero gradients for active mechanism paths, RNG preservation during evaluation, one optimizer step, checkpoint save/reload and resource recording. Do not use or disrupt another GPU process.
6. Record the Implementation Freeze Commit only after code and tests pass. The formal runner records its own source commit SHA separately.

## Formal execution gate

Run the six seed-42 cells first. Inspect: finite epoch histories and validation metrics; H1/H2 initialization parity; no cache/split mismatch; expected physical cross message reaches both expert profiles and `_node_correction`; Private auxiliary heads and the active semantic attention path receive gradients; H2 Same-node path and dropout behave as specified; checkpoint hashes/reloads; no Test metric or label access; no OOM or other resource interference. A low Accuracy alone is not a stop reason. If all checks pass, run all twelve seed-43/44 cells without changing frozen code/configuration.

Keep failed and retried attempts in separate immutable attempt directories and append records to `data/failures.json`. Never overwrite a failed log/checkpoint or use a failed attempt as an accepted formal result.

## Performance optimization boundary

After the default 18-run matrix and report are complete, decide whether an extra training-strategy screen has a clear motivation. It is optional. If run, select at most two complete configurations with defensible hypotheses from dropout 0.2/0.35, weight decay 1e-4/1e-3, private auxiliary CE 0.3/0.1 and learning rate 1e-3/5e-4. Screen on all three datasets at seed 42. Only expand a promising configuration to seeds 43/44. Store it under a new condition and preserve default H1/H2 rows unchanged. No new mechanism is allowed in this screen.

## Evidence and analysis

For every accepted run retain Validation Best-Accuracy/Best-F1 metrics, Train/Validation gaps, epoch history, classwise and confusion evidence, model/head initialization hashes, data/cache fingerprints, checkpoint path/SHA256/bytes, resource profile, first-step gradient audit, mechanism diagnostics, source SHA and environment. Checkpoint and raw graph/feature/cache files stay in ignored server-local output; commit compact CSV/JSON analysis evidence, not large checkpoints or raw data.

Report paired same-seed H1/H2 differences, and historical H1/H2 versus B1/A2 differences only for strict matched fingerprint cells. Focus on whether Movies B1's improvement remains while ele-fashion A2's advantage recovers. Include Accuracy, Macro-F1 and CE; both checkpoint rules; per-seed values; gaps; time/memory/parameters; Early Physical Cross RMS and router corrections; Semantic/SameNode/None weights; Late residual magnitudes; relative-cap frequency and scales; expert use; Private auxiliary CE; and Fashion minority classes. Separate measured performance, confirmed code behavior, mechanistic hypotheses and unresolved explanations. Test results and Test-derived selection are forbidden.

## Stop condition

Complete the report, audit and decision memo, self-review the code/data/claims, commit compact evidence and push `exp/hybrid_cge_v11`. Stop after delivery. Do not start a successor model.
