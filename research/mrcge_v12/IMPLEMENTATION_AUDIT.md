# MR-CGE V1.2 implementation audit

Date: 2026-10-25. Source files are listed in the repository root under `src/models/mrcge_v12.py`, `configs/model/mrcge_v12.yaml`, `configs/study/mrcge_v12_screen.yaml`, `scripts/preflight_mrcge_v12.py`, `scripts/run_mrcge_v12_screen.py`, and `tests/test_mrcge_v12.py`.

## Changes and evidence

- Physical highest propagation order is configured and validated as 3; semantic highest order is 1. The filters still contain four and two basis terms respectively.
- During training, each modality's collaborative physical-filter call replays the private call's CPU/CUDA RNG state inside a forked RNG context. The fork restores the surrounding stream, so checkpoint recomputation preserves the paired masks without consuming extra random draws in later modules.
- M's complete shared expert/router segment is checkpointed with `use_reentrant=False` and RNG preservation enabled. The training path avoids full diagnostic tensors unless explicitly requested. All four experts still execute; Top-2 controls the mixture, not whether expert kernels run.
- Eval retains the original V1.1 expert stack/einsum/router-reduction calculation. This avoids a segmented accumulation rounding drift found during earlier regression attempts and preserves V1.1 checkpoint behavior.
- S/F initialization audit confirms equal common model state, classifier state, effective filter coefficients, and initial eval logits; only F theta is learnable. M_repaired/M_anchor share all state except the specified first theta row.
- No registered parameter was found to be newly unused in the forward path. The external NC trainer interface and Train-only private CE are retained.

## Verification

Command: `/home/m3/miniconda3/envs/yhf_env/bin/python -m pytest -q tests/test_mrcge_v12.py`
Result: **17 passed** in 112.83 seconds (6 dependency deprecation/future warnings).

The unit suite covers shared four-view parameters, hop validation, expert masks and signed normalization, initialization invariants, paired dropout and zero/small cross behavior, router/Top-2/balance behavior, private/collaborative gradients, checkpoint output/loss/gradient/RNG equivalence, late routing and fusion, Train-only loss/Test masking, cache fingerprints, save/reload and Validation RNG, variable feature dimensions, and CPU execution.

The protected GPU preflight also ran an old-V1.1 versus checkpointed-V1.2 M-filter CUDA audit for each M dataset/condition pair. Across all six pairs (Movies, Grocery, Toys, ele-fashion × M_repaired/M_anchor), the 192-node synthetic response, loss, router/filter gradients, and RNG checks passed. Maximum absolute gradient error was `5.96e-8`; response tolerance was `atol=1e-5, rtol=1e-4`, loss tolerance `atol=1e-6, rtol=1e-5`, and gradient tolerance `atol=2e-5, rtol=2e-4`. Checkpoint backward restored the forward RNG state. The synthetic allocated-memory increment fell from about 22.3 MiB in V1.1 to about 5.6 MiB in V1.2; allocator-reserved peaks were similar and CUDA wall times are retained per run in `data/preflight.json` as microbenchmark observations, not an end-to-end speed claim.

## Historical V1.1 checkpoint regression

The actual Best Accuracy M checkpoints for Movies, Grocery, and Toys loaded strictly with matching state keys. Validation metrics matched exactly, validation RNG was unchanged, Test labels remained masked, and no Test metric was computed. All embedding/logit elements were within FP32 `atol=1e-5, rtol=1e-4`; maximum embedding/logit absolute errors were respectively:

| Dataset | Embedding max abs error | Validation logit max abs error | Validation Accuracy / Macro-F1 / CE |
|---|---:|---:|---|
| Movies | `1.67e-6` | `2.86e-6` | 0.552789 / 0.483802 / 1.405776 |
| Grocery | `1.67e-6` | `3.81e-6` | 0.826940 / 0.756757 / 0.669436 |
| Toys | `1.43e-6` | `3.34e-6` | 0.795361 / 0.762703 / 0.730716 |

The measured differences were at or below repeated V1.1 CUDA scatter variation. Full checkpoint paths, SHA-256, metrics, element counts, and repeat-variation values are preserved in `data/v11_checkpoint_regression.json`.

Known limit: this establishes numerical agreement for the tested checkpoints and controlled M-filter path; it does not establish an end-to-end training-speed improvement or sparse four-expert execution.
