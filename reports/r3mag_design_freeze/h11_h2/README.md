# R³-MAG H1.1 and H2 preflight screen

This branch adds a compact follow-up to the H1 frozen-host audit. It does not implement H3 or the final R³-MAG model.

## Fixed protocol

- Movies and Grocery use dataset seed 42; ele-fashion uses its official split. Internal HostTrain/ResponseTrain/Audit partition seed is 20261006. Frozen host seeds are 42, 43, and 44. Every internal and external split index hash is checked against the prior H1 per-run record before evaluation.
- Existing `outputs/r3mag_design_freeze/h1/<dataset>/seed<seed>/host_best_val.pt` checkpoints are reused. A missing checkpoint is retrained only with the previous H1 host function and settings. Per-run JSON records path and reuse status.
- Test labels and metrics are untouched. Test indices are hashed as metadata only. Original validation labels are read solely to verify the reused checkpoint's prior validation metrics. ResponseTrain labels are used only for response targets, predictor split stratification, and predictor/adapter fit; Audit labels are indexed after the host is frozen and only for evaluation.
- H1.1 runs 20 deterministic control seeds and ε=0.1/0.2. Coherent donor uses one node donor for both modalities and all orders. Independent donor retains the previous H1 sampler. Geometry scramble uses one shared signed coordinate permutation across modalities and targets per repeat.
- H2-A uses ε=0.1 own-structural slope targets, training-only per-dimension standardization, class-stratified ResponseTrain 80/20 split seed 20261007, Huber loss, and a two-layer 128-unit MLP. P1/P2/P3 are capacity matched by zero-padding context for P1. P3 and P2-eval-shuffle use deterministic context derangements confined to ResponseTrain and Audit separately.
- H2-B trains only a 128-unit adapter with AdamW (1e-3, weight decay 5e-4), validation CE early stopping, up to 500 epochs and patience 50. B1/B2/B3 have equal architecture and parameter count. B0 uses the frozen host directly.

## Protocol deviations and implementation notes

No deviations were made to the registered datasets/splits, H1 host architecture/settings, host seeds, H1.1 epsilons/repeat count, or H2 optimizer/loss/epoch caps. The only implementation convention is zero-padding the context block for P1/B1 to keep parameter counts equal while supplying those variants no relational information. ResponseTrain's class-stratified 20% validation count is rounded per class; singleton classes remain in predictor-train and are listed in each run's split metadata. Matching fallbacks, singleton shuffle maps, and any checkpoint retraining are recorded per run.

## QA and outputs

See `qa_report.json` for split/checkpoint, donor, signed-permutation, Gram, NormMatch, action-count, split-boundary, context-finiteness, equal-capacity, audit-label-boundary, and frozen-host checks. Detailed compact run records are in `per_run/`. Large intermediate outputs are ignored under `outputs/r3mag_design_freeze/h11_h2/`.

Smoke mode uses Movies/seed42, two H1.1 controls, and short H2 optimization. Smoke results are implementation QA only and do not enter tracked formal reports. Formal reports list all negative results and protocol deviations without a mechanical all-seeds pass/fail gate.

Run from the repository root in `yhf_env`:

```bash
PYTHONPATH=. python -m pytest -q tests/test_r3mag_response_core.py tests/test_r3mag_h11_h2.py
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen --smoke --phase h11
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen --smoke --phase h2a
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen --smoke --phase h2b
PYTHONPATH=. python -m src.analysis.r3mag_h11_h2_context_screen
```
