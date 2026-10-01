# N1 — Recipient-Conditioned Function-Strength Mixing Screen

Validation-only MAG node-classification mechanism screen on Movies, Grocery and ele-fashion; seeds 42–44.

Run from the repository root with `conda activate yhf_env`, then `python scripts/run_n1_recipient_function_strength_mixer.py --smoke --device cuda:1` and `python scripts/run_n1_recipient_function_strength_mixer.py --campaign --regression --device cuda:1`.

The campaign uses full-graph training, train labels only, validation-accuracy checkpoint selection and the frozen unified NC optimizer/scheduler/early-stop settings. `task.evaluate_test=false`; no test indices or link-prediction task are used.

Machine-readable tables are in `data/`; figures are in `figures/`; the N1 implementation and checks live in `src/models/adaptive_prop_n1.py`, `scripts/` and `tests/`. Large run records and checkpoints are gitignored under `outputs/n1_recipient_function_strength_mixer/`.
