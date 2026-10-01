# N0 — Recipient-State-Conditioned Functional Utility Audit

N0 audits how validation counterfactual CE utility changes when the recipient's Text and Visual incoming-neighborhood backgrounds change. It reuses the frozen P0/P1.3 H0 states and all 27 L0.1 shared-slot linear heads. The only edge transformations are Smooth, AbsDiff, and Product.

## Reproduction

Run from the repository root in `yhf_env`:

```bash
python scripts/run_n0_recipient_state_function_context.py --device cuda:1
python scripts/analyze_n0_recipient_state_function_context.py
python scripts/plot_n0_recipient_state_function_context.py
python -m pytest -q tests/test_n0_recipient_state_function_context.py tests/test_l01_shared_slot_function_identifiability.py
```

The campaign command first runs the required Movies/seed 42/head repeat 0 smoke. It stops if that smoke fails. `--smoke-only` runs only the smoke. `--formal-only` requires an existing passed smoke audit. Missing L0.1 shared-head checkpoints alone may be retrained with the exact L0.1 trainer and split protocol; present checkpoints are reused and checked against historical head-select CE. SS gain regression uses `rtol=1e-7, atol=2e-7`; the small absolute allowance covers observed floating-point drift when frozen H0 outputs are recomputed.

## Frozen protocol

- Source branch/SHA: `exp/l01_shared_slot_function_identifiability` / `1ca0c98e4e989317b1485fe1c2d7c63daf1ff6fa`.
- Datasets/seeds: Movies, Grocery, ele-fashion × 42, 43, 44.
- Candidate edge is held at Smooth in its own modality background; the other modality uses its complete neighborhood background.
- Original physical indegree is fixed and self messages are absent.
- Full recipient background grid: SS, SD, SP, DS, DD, DP, PS, PD, PP.
- Partial audit changes a deterministic hash-selected `floor(0.25 × degree)` incoming subset, with a minimum of one selected edge for degree ≥5. The candidate edge is excluded from its own perturbation when selected. The opposite modality stays Smooth.
- Eligibility for rank and partial audits is validation targets with original indegree ≥5.
- Pairwise CE-curvature sanity samples at most 500 distinct same-recipient validation edge pairs per head and modality under SS with AbsDiff replacements.

## Interpretation boundary

For fixed edge, alternative, modality, and head, the logit delta is constant across recipient backgrounds by linearity and fixed degree. Conditional gain variation therefore measures recipient-state-conditioned task-loss geometry. Pairwise CE nonadditivity is compared with the softmax Hessian cross term and is not treated as message semantic interaction. N0 trains no model, uses no test labels or metrics, and does not establish ground-truth edge roles or validate any proposed multiset architecture.

## Outputs

- `data/`: per-run tables and three-seed dataset summaries.
- `figures/`: six figures exported as 600 dpi PNG and TIFF plus editable SVG and PDF.
- `report.md`: answers to the 20 audit questions and self-audit A–J.
- `figure_qa.md`: panel claims, uncertainty definitions, and final figure QA results.
- `run_manifest.json`: provenance, checkpoint recovery, correctness, run counts, runtime, and memory records.
- `outputs/n0_recipient_state_function_context/` is gitignored and contains edge-level raw tables, smoke/formal audit JSON, and figure QA files.
