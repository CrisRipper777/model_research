# L0.1 — Independent Shared-Slot Function Identifiability Audit

This validation-only audit tests whether replacing a single incoming Smooth message with AbsDiff or Product changes predictions under one shared downstream structural slot. It reuses the frozen P0/P1.3 semantic embeddings and fixed directed physical graph. It does not train a router, sample latent functions, or evaluate the test split.

## Protocol

- Source: `exp/l0_relation_function_learnability_audit` at `9ce5f723fda0bf17264671a1e20e78f043727498`.
- Datasets/seeds: Movies, Grocery and ele-fashion; seeds 42, 43 and 44.
- Shared head: one `Linear(512, num_classes)` for all nine Text/Visual S/D/P context pairs. Its only labels are original `train_idx`, split 80/20 into deterministic stratified `head_train` and `head_select`. NC validation labels are used only to form substitution gains.
- Graph/context: directed `j -> i`, no self messages, P1.3 Smooth/AbsDiff/Product formulas, whole-neighborhood sum divided by the original physical incoming degree.
- Clean targets: three independently fitted shared heads produce repeat-level gains; their edge-aligned mean is the four-output target `[G_D^T, G_P^T, G_D^V, G_P^V]`. Repeat dispersion and sign agreement remain available in the data tables.
- Learnability: L0's 1542D master evidence and masks, three dst-disjoint folds, dst-disjoint inner validation, inner-train-only feature/target scalers, strict within-destination tuple shuffle in both inner train and inner validation, fixed-alpha Ridge sanity, utility-supervised M0 q/r/u state, and frozen E0.1 Q/R/U readouts.
- Test labels/metrics, link prediction, new function labels, stochastic routers and new model mechanisms are out of scope.

## Run

Activate `yhf_env` from the repository root:

```bash
conda run --no-capture-output -n yhf_env python scripts/run_l01_shared_slot_function_identifiability.py --device cuda:1 --smoke-only
conda run --no-capture-output -n yhf_env python scripts/run_l01_shared_slot_function_identifiability.py --device cuda:1 --skip-smoke
```

The formal command assumes the Movies/42 smoke has passed. It writes committed, compact summaries under `data/`, while per-edge utilities, split records, head checkpoints and raw fit tables go under the gitignored `outputs/l01_shared_slot_function_identifiability/` directory.

## Interpretation boundary

JS divergence and logit shifts are empirical output-separation measurements. They do not establish theorem-level injectivity or recovery of a latent graph. A positive gain is a task-specific counterfactual under the fitted shared head, not a ground-truth edge role. Head-repeat variability is model-fitting uncertainty; it does not imply an aleatoric stochastic function process.

The final evidence and decision label are in [report.md](report.md). Protocol, provenance, split, H0 and fit-count audits are in [run_manifest.json](run_manifest.json).

## Final result

Outcome: **`DATASET_DEPENDENT_MIXED`**. Movies and Grocery provide little clean edge-level predictability beyond weak target context; ele-fashion is more predictable, but endpoint evidence is at least as useful as endpoint-plus-local evidence and strict shuffled correspondence retains signal. This does not justify an edge-wise router or a stochastic function distribution. Six figures passed panel alignment, PDF text, and collision QA; see the report and manifest for the full audit.
