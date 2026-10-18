# Decision memo

## Evidence status

- The formal matrix is complete: 174/174 new Dataset × Model × Seed cells, plus 45 reused baseline cells with their parent-branch fingerprints audited. The raw `runs.csv` retains 258 complete attempts: 219 original rows and 39 corrected reruns. All 39 superseded rows and checkpoints remain preserved and are listed in `data/superseded_runs.csv`.
- Training sources are intentionally split: unaffected cells use initial freeze `0db724e5eec704a1f3db92329bfbbac056df7202`; corrected A1, 3-hop and Fashion F1/F2/F4 cells use `5e052dbc26237ddb534e0f779d91951f26079748`. The correction freeze fixed SameNode dispatch. Ten focused tests and 581 repository tests passed after correction; 13/13 correction preflight cells and all formal stage gates passed.
- A CPU-side constructor audit found exact equality for all common non-`alpha_raw` parameters and the classifier across the 15 H2 4-hop/3-hop Dataset × Seed pairs. Only the expected physical coefficient matrix shape differs. Ablation shared initialization was independently checked by the stage gate.
- Corrected checkpoint diagnostics reloaded 39 checkpoints. SameNode support and nonzero selector contribution are present in H2/3-hop/Fashion conditions; A4 remains the deliberate SameNode-off condition. Train/Validation diagnostics preserve RNG state and mask Test labels. See `IMPLEMENTATION_CORRECTION_AUDIT.md`, `PHYSICAL_SEMANTIC_DIAGNOSTICS.md`, and `data/final_evidence_audit.json`.
- The final-model lock is proposed and pending independent review. The Test runner remains default-deny. No Test labels or metrics were used, and no Test run was performed.

## Main findings

- **Ablations:** All four are framework-level path removals, not output multiplication by zero. A1 removes early physical-cross computation and retains the late path; A2 skips late completion; A3 removes shared MoE/router parameters and balance loss; A4 disables SameNode projection while retaining late-cross Dropout 0.15. Results are mixed and mostly small. Three seeds and historical 4-hop controls do not support strong causal claims; all paired values remain in `ABLATION_REPORT.md`.
- **Depth:** 3-hop is higher in Accuracy and Macro-F1 on four of five datasets, with Toys favoring 4-hop. Equal-dataset mean paired changes are only +0.02 pp Accuracy and +0.33 pp Macro-F1. Since the 4-hop comparator is historical and not matched to a new campaign, select 3-hop as a provisional working global configuration, not as a proven generalization or causal improvement.
- **Baselines:** The same-protocol results show Hybrid-CGE is competitive, not universally superior. DiP is stronger on Grocery Accuracy and Fashion Accuracy/Macro-F1. The nine-family table and adaptation caveats are in `BASELINE_REPORT.md` and `BASELINE_AUDIT.md`.
- **Fashion:** Accuracy-primary selection retains F3 (3-hop H2). F4 produces the highest mean Macro-F1, trading away 0.35 pp Accuracy relative to historical F0. F1 changes Dropout and auxiliary CE together; Weight Decay is fixed, so neither individual effects nor WD interactions are identified. A zero-Train-support class is assigned zero weight and receives no Train target.
- **Checkpoint selection:** Best-F1 checkpoints generally trade Accuracy for Macro-F1 relative to Best-Accuracy checkpoints. The aggregate and per-cell CE/Accuracy/F1 differences and Train–Validation gaps are in `CHECKPOINT_SELECTION_REPORT.md`.
- **Physical/Semantic diagnostics:** Corrected forward summaries show dataset- and modality-dependent contributions. SameNode weights are nonzero after the wrapper fix; these activation and routing summaries are descriptive and do not measure causal utility. Initial disabled-SameNode diagnostics are preserved separately.

## Implementation correction and replay boundary

The first paper wrapper passed renamed condition IDs into a frozen parent that enabled SameNode only for the literal `H2_hybrid_plus` condition. Thus the first A1, 3-hop and Fashion F1/F2/F4 attempts incorrectly had SameNode disabled. This was an implementation bug established by source inspection and all-zero initial support diagnostics, not an accelerator attribution. The wrapper now explicitly invokes the controlled late-completion path and disables SameNode only for A4. The 39 affected cells were rerun from the corrected freeze; first attempts remain unchanged. See `IMPLEMENTATION_CORRECTION_AUDIT.md`.

## Claim boundary and recommendation

The implementation and protocol checks support that the tested paths were present/removed as specified, that data/cache/split fingerprints stayed fixed for matched reruns, that common initialization and checkpoint integrity were audited, and that all reported scores are Train/Validation-only. The experiments support descriptive comparisons across the registered conditions. They do **not** establish significance, mechanism causality, state-of-the-art superiority, or broad generalization beyond these five datasets and three seeds. Historical 4-hop comparisons are cross-campaign. The Fashion Dropout/Auxiliary-CE bundle cannot be disentangled; Weight Decay was not varied.

Recommend `H2_physical_3hop` as the provisional global Validation choice and as the Accuracy-primary Fashion choice, retaining 4-hop and F4 as comparators. Keep `FINAL_MODEL_LOCK.json` at `pending_review`. This handoff contains no authorization to evaluate Test; wait for independent review and separate user authorization.
