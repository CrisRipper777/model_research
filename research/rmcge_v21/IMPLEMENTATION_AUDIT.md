# RM-CGE V2.1 Implementation Audit

Implementation Freeze preflight ran on branch `exp/rmcge_v21_message_purpose_selection` from V2 evidence `ba1fbbd100a0b218ec66853e29ea0cfee8cfa5b4`. Full CPU regression: **484 passed, 1 skipped**. Targeted V2.1 coverage includes graph mass conservation, zero/equal scores, isolates/single-neighbor rows, directed and duplicate edges, chunking, receiver-conditioned rank reversal, C0 exact delegation, paired initialization, zero-effect equivalence, Cross Projection direction, task gradients/AdamW updates, and checkpoint reload.

GPU smoke used NVIDIA GeForce RTX 3090. Peak allocated memory across the smoke was 12.22 GiB. GPU runtime details and initial hashes are in `data/implementation_checks.json`.

C0 parameter initialization and Forward delegate to RM-CGE V2/V1. Initialization audit records per Dataset × Seed C0/C1/C2 model and classifier SHA-256 values; common parameters and the full C1/C2 added state match exactly. Added modules are created inside a forked RNG scope so they do not shift classifier initialization.

The physical operator remains the inherited V7A/R0 operator. C1/C2 use edge-chunk scoring, FP32 differentiable receiver scatter and chunked SpMM; they do not form a dense adjacency or an E×hidden message matrix. The edge scorer's shared output bias is expected to have zero task gradient under receiver-wise normalization's shift invariance.

Test indices are required by the shared task data object, but this runner disables Test evaluation and never indexes Test labels. Initialization fingerprints intentionally include only features/topology and Train/Validation indices and labels. No Test metrics appear in run records or checkpoints.

The first report-generation pass exposed a formatting bug: template fragments were written literally into Markdown. The analyzer's report construction was corrected after training; the correction changes no model, run record, checkpoint or metric. This audit and the artifact manifest were regenerated from the corrected analyzer.

Formal artifacts: 27 verified runs, 3633 epoch rows, 108 edge-path rows, 135 frozen-intervention rows. Preserved failure records: 0.
