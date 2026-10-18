# Hybrid-CGE Paper Evidence V1 Protocol

## Goal and scope

This branch produces validation-only evidence for four mechanism ablations, a controlled 3-hop versus 4-hop comparison, nine same-protocol baselines, and four finite Fashion configurations. It adds an isolated final Test runner with a default-deny policy but does not execute Test. The frozen parent implementation and all pre-existing data/checkpoints remain unchanged.

The complete H2 reference is Hidden=256, four physical hops, four shared experts, Top-2, V7B.1 semantic K=8, Dropout=0.2, late-cross Dropout=0.15, AdamW LR=1e-3, weight decay=1e-4, private auxiliary CE=0.3, and Best Validation Accuracy checkpoint selection. Seeds are 42, 43, 44; datasets are Movies, Grocery, ele-fashion, Toys, and Reddit-S.

## Inputs and protocol

Every newly trained model uses the repository's existing features, global node order, physical graph, Train/Validation/Test split, and per-dataset V7B.1 semantic cache. All semantic caches are feature-only and use independent Text and Visual K=8 searches. Split, feature, graph, and cache fingerprints are stored for each run. Graph models use full-graph transductive supervised node classification. Test labels are masked to `-100` before the trainer; Test evaluation is disabled throughout.

Training uses the existing unified full-graph NC budget: AdamW, LR 1e-3, weight decay 1e-4 except where a candidate explicitly changes it (none in this study), at most 300 epochs, patience 30, minimum epoch 30, minimum accuracy improvement 1e-4, gradient clip 1.0, evaluation every epoch, Validation Accuracy stopping/checkpoint selection, and a secondary Best Validation Macro-F1 checkpoint. Private Text/Visual auxiliary CE is Train-only and unweighted. Fashion class weights are computed only from Train label counts using normalized `1/sqrt(n_c)` and clipped to [0.5, 2.0]; only the main CE receives them.

## Formal run matrix

- Ablations: four mechanisms × five datasets × three seeds = 60.
- H2 physical 3-hop: five datasets × three seeds = 15 new runs. The 4-hop H2 comparison uses the existing Validation-only H2 records: `H2_hybrid_plus` for Movies/Grocery/ele-fashion and `H2-Frozen` for Toys/Reddit-S, each with seeds 42/43/44. These were produced in earlier campaigns, so the seed-paired deltas are descriptive comparisons across run lineages, not same-batch or matched-initialization estimates. Their source records, configurations, and checkpoint provenance must remain separately identified in the reports.
- New baselines: GCN, MGAT, DiP, DGF, DMGC, LGMRec × five datasets × three seeds = 90.
- Reused baseline evidence, if its full audit passes: SAGE, MMGCN, RoleMAG × five datasets × three seeds = 45. MLP and MvCGE-MAG are excluded.
- Fashion candidates: F1, F2 and F4 × three seeds = 9. F3 aliases the matching three-hop result and is not trained twice. F0 is the historical H2-Frozen reference.

Formal new training total is 174, with 45 separately marked reused baseline rows. No fresh full H2 4-hop training cells are added to this count. A new cell is complete only after finite metrics, masked-Test verification, checkpoint hash/reload validation, and Validation RNG audit.

## Matching and checkpoints

Within Dataset × Seed, the four new ablations are initialized from the same common model/head initialization before any structural intervention. The 3-hop condition retains the original four-expert bank and Top-2 router; it changes only propagation depth and the alpha basis width. H2 4-hop checkpoints remain load-compatible because the historical model class and checkpoint files are untouched. Per-run records include full and shared-parameter initial model SHA, classifier SHA, source/config SHA, input/split/cache fingerprints, checkpoint SHA, training history and Test-off status.

## Selection and Test boundary

Validation Accuracy is primary and Validation Macro-F1, CE, classwise metrics and Train–Validation gaps are secondary. New ablations are paired within the same Dataset × Seed and have verified matching shared initialization. The 3-hop scores are seed-aligned with historical 4-hop H2 records but have distinct source commits and run campaigns; reports must show this lineage and must not present them as same-batch replicated controls. Fashion candidates are dataset-specific and cannot establish a global model change. No published scores from mismatched protocols enter the same-protocol tables.

The final Test runner first requires an explicit `--final-test` flag and a lock with approved status; it checks the locked model/config/source/seed/checkpoint digest and loads only that checkpoint without training. The delivered lock is pending independent review, so this branch cannot execute a final Test during this stage.
