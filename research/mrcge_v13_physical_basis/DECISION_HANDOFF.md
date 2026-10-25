# Decision Handoff — MR-CGE V1.3 Physical Basis Screen

## Review checkpoint

The authorized campaign is complete on branch `exp/mrcge_v13_physical_basis_screen`. It began from `4864db09356fb3a426c03dc1e2c097af5342aebb`. Implementation and run provenance are frozen at source commit `cb641c4947c13d2305cfe83110cfdf9e35d42517`; the final branch HEAD is recorded in the completion message and Git history. All seven authorized new cells completed: **7 successful, 0 failed, 0 resource-blocked, 0 unstarted**.

The implementation, data protocol, and preflight records are in `DESIGN_FREEZE.md`, `IMPLEMENTATION_AUDIT.md`, `DATA_PROTOCOL_AUDIT.md`, and `PREFLIGHT_REPORT.md`. Metrics and mechanism interpretation are in `SCREENING_REPORT.md`; the machine-readable evidence is under `data/`. Checkpoints and full logs remain on the server under `outputs/mrcge_v13_physical_basis/` and are not committed.

## Findings for independent review

- On the same frozen Movies split, Hybrid-CGE H2 3-hop exceeded V1.3-A's three-seed mean by **1.27 percentage points Accuracy** and **1.04 points Macro-F1**; mean CE was **0.00686 lower**. This comparison is descriptive: each model family has distinct parameterization and initialization order.
- Physical NoP0 did not improve the two-seed mean on either dataset. Movies changed by **−0.42 pp Accuracy**, **−1.21 pp Macro-F1**, and **+0.0173 CE**. Grocery changed by **−0.47 pp Accuracy**, **−0.06 pp Macro-F1**, and **+0.0392 CE**.
- The coefficient audit confirms that Physical order zero is masked exactly for both Private and Collaborative experts, Semantic orders zero and one remain active, and the shared `theta` receives gradients. The model still propagates from the original basis and retains the intrinsic residual.
- Early Cross calibrated response and Private/Collaborative Physical response changed with the NoP0 renormalization. This screen does not isolate the zero-order contribution from that coupled Cross change.
- There is no evidence here for a final model selection. Two NoP0 seeds are descriptive, and no significance test was authorized.

## Verification and safety audit

- Focused implementation tests: **11 passed**. Required regression set after training: **65 passed**, with 21 deprecation warnings.
- Original-switch initialization/evaluation/training equivalence, classifier initialization, NoP0 coefficient masking, unchanged Semantic orders, shared-parameter gradients, checkpointed training vs evaluation path, strict checkpoint reload, support/dropout behavior, and corrected Hybrid SameNode support were covered.
- The seven GPU-1 launch guards and runtime monitors passed. No new foreign GPU compute PID was observed, GPU 0 was unused, Test evaluation was disabled, Test labels were masked before Trainer entry, and Validation RNG was unchanged.
- The final analysis verified all 115 protected historical research files unchanged, all successful run/checkpoint hashes, and the fixed split fingerprints.

## Review boundary

This branch is ready for independent review. No additional training, Test evaluation, link prediction, Static-Strength, extra seed, physical-order search, Cross redesign, or later model version is included. Please review this branch and its evidence before deciding whether any new study should be authorized.
