# Implementation Audit

- Branch: `exp/mag_view_collaborative_experts_v1`.
- Implementation Freeze Commit: `f981a5f0164b1ed73c379724139e75c84abb0c98`.
- Formal-run source commit: `608dbe406e698ce5dea7facd1cdcdb927b6943a7`; the evidence/report commit is the final delivery commit on this branch.
- Evidence/run count: 27/27; epoch-history rows 3715; failed/retried attempts recorded: 3.
- Final test results: 542 passed, 0 failed, 0 skipped, 21 warnings in 58.89 s; focused View-CGE: 14 passed.
- Checkpoints independently checked for file size and SHA256: 54/54.
- A0 real-data forward compatibility maximum absolute error: 1.90734863e-06; per-dataset values are in `data/preflight.json`.
- Seed42 initialization and split gate: `True`.
- Physical GPU: 1; device `NVIDIA GeForce RTX 3090`; maximum formal allocated memory 11.12 GiB.
- Test evaluation: disabled. The runner replaced Test labels with `-100` before calling NC; no test metric columns are accepted by the analyzer.
- Split and cache fingerprints: `data/manifest.json`; initial model/head hashes and split match are checked per Dataset × Seed.
- Run histories, classwise scores, confusion matrices, degree groups, transitions, view/router diagnostics and frozen shutoff sensitivity are retained as CSV/JSON under `data/`.

## Engineering findings

The preflight first found that V7B.1 source provenance values are proposal counts, not Boolean flags; validation was corrected to preserve the existing `>0` operator semantics. The first full-graph backward then found an extra expert axis in the physical mixture, causing a 159 GiB illegal indexing allocation. The mixture output shape was fixed, asserted in tests, and the A1/A2 full-graph one-step preflight passed on all three datasets. No formal training began before these fixes.

## Provenance and safety checks

- A0 delegates to the frozen V2.2 M0 forward. The new common parameter tree and classifier initialization were compared against it on real data.
- A1/A2 use V7B.1 Text/Visual source supports and global node IDs. The cross-source direction tests confirm Text receives Visual-KNN and Visual receives Text-KNN messages.
- Semantic view masks are applied after expert response; physically isolated nodes retain valid semantic messages.
- Full graph only; link prediction is explicitly unsupported. No labels were used for graph/cache creation.
- Current evidence does not establish causal utility for individual routed experts; frozen closure is checkpoint sensitivity only.

## Preserved execution failures

Movies/A0/seed42 attempt 1 stopped before training because checkpoint metadata was absent. Attempt 2 trained 103 epochs and wrote checkpoints, then failed in post-training diagnostics on a node mask with an expert-leading tensor axis; the defect was fixed, both failed attempt directories were retained, and attempt 3 completed the formal cell. After seed43 completed, the GPU-idle guard stopped before seed44 when an independent LLH worker (PID 27585, 816 MiB on physical GPU 1) appeared. No process was interrupted; after it exited, seed43 was verified/skipped and seed44 completed. The three records remain in `data/failures.json`.

## Evidence categories and self-review

- **Code facts:** A0 delegates the V2.2 M0 forward; A1/A2 make Physical, same-source Semantic and (A2) opposite-source-support Cross views explicit. All views share one four-expert bank and one S/C router. Direction, expert identity, initialization isolation, gradients and empty-view masking are tested.
- **Measured results:** the frozen preflight, paired hashes/fingerprints, 27 Test-disabled runs, 54 hashed checkpoints, classwise/degree/transition data and view/router diagnostics are stored under `data/`.
- **Unproven mechanism claims:** nonzero messages, route selection or prediction sensitivity do not prove causal utility for an expert or view; frozen view closure is not a retrained ablation. Three seeds are descriptive only.
- **Self-review:** the contribution is the relation-source organization and opposite-source support path, not shared experts alone (already present in V7A). Semantic messages are not duplicated in Physical input; the legacy Physical pathway remains and its weighted contribution is larger in the aggregate diagnostics. A1/A0 and A2/A1 are whole-condition contrasts, not isolated component effects. Test labels/metrics were not used; LP support is not claimed; no next model version is prescribed.
