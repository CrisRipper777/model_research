# Optimization and comparison review

## Trained coefficients and Router parameters

The bounded scalars below are read from the frozen E2 checkpoints. The derivative is with respect to the learned raw scalar under `value = maximum * sigmoid(raw)`.

| Dataset | Effective `rho` range across 4 source→target cells | `d rho / d rho_raw` range | `beta_phy` | `beta_sem` | Fixed `gamma` |
|---|---:|---:|---:|---:|---:|
| Movies | 0.0997–0.1048 | 0.0600–0.0609 | 0.5102 | 0.2979 | 0.2 |
| Grocery | 0.1011–0.1045 | 0.0602–0.0608 | 0.5159 | 0.3051 | 0.2 |
| ele-fashion | 0.1028–0.1079 | 0.0605–0.0613 | 0.5258 | 0.3206 | 0.2 |

The four-by-four source/target values and raw parameters are recorded in `data/optimization_evidence.json`. `rho` stays close to its 0.1 initialization and far from its 0.25 upper bound; its effective derivative also stays close to the initial 0.06. The physical and semantic bounded scales remain interior to `[0, 1]`. This does not support a saturation diagnosis.

Each checkpoint contains 34,115 trainable Router parameters plus 40 relation/modality embedding parameters. The machine evidence gives per-tensor shape, mean, standard deviation, extrema, L2 norm, RMS, and SHA-256 for the trained Router and conditioning embeddings. It does not duplicate their weight arrays; the checkpoint hashes bind the audit to the exact saved values. Route probabilities and nodewise distributions are separately reported in `data/router_diagnostics.csv`.

## Gradient and update evidence

The stored E2 gradient summaries are from epoch 1, before global-norm clipping. Cross-modal MLP and `rho_raw` gradients were present and finite in all three datasets:

| Group | Parameter count | Gradient L2 range | RMS-from-L2 range |
|---|---:|---:|---:|
| Cross-modal MLP | 33,152 | 7.12e-4–1.21e-3 | 3.91e-6–6.64e-6 |
| `rho_raw` | 4 | 3.89e-6–1.14e-5 | 1.94e-6–5.69e-6 |
| Router | 34,115 | 5.54e-2–1.13e-1 | 3.00e-4–6.10e-4 |
| Experts | 123,456 | 0.372–0.530 | 0.00106–0.00151 |

An existing real-graph preflight audit also recorded finite, nonzero gradients for all four `rho_raw` values and all 33,152 cross-modal parameters, followed by changes to all of those parameters in its single AdamW preflight step. The largest absolute update was about `1e-3`. That preflight is evidence that gradients and an update path existed for that one checked step; it is not an epoch trajectory and is not a measurement performed by this frozen audit.

Raw gradient L2 values are not an importance ranking. Parameter groups differ greatly in element count and scale, and an AdamW update depends on its moment state, weight decay, clipping, and learning rate. RMS-from-L2 is included only as a size-adjusted description. The small cross-modal and `rho_raw` gradients, together with a small output-layer initialization and multiplicative `rho`, motivate checking cross-modal scale during future training, but do not prove an optimization defect.

## Frozen cross-modal scale

The measured `RMS(rho*Phi)/RMS(H)` ratio is about `1.8–3.7%` on Movies, `2.4–3.1%` on Grocery, and `8.3–10.4%` on ele-fashion. Aggregated cross-on/off differences are not substantially attenuated relative to the sent-message change, yet turning the branch off changes Validation accuracy by at most `0.03` percentage points and flips at most `0.27%` of Validation predictions. This is evidence of a small and prediction-redundant cross-modal contribution in these fitted checkpoints. It is not proof that another initialization or a retrained branch cannot help.

The one-seed retrained Pilot comparison points in the same cautious direction: E2's Validation Accuracy relative to E1 (cross-modal disabled during training) is +0.15 pp on Movies, -0.15 pp on Grocery, and -0.02 pp on ele-fashion. Its Macro-F1 differences are -0.68, -1.39, and -0.31 pp, respectively. These small, mixed differences do not establish a stable benefit or harm.

## Training-curve evidence

The saved logs contain train loss and Validation Accuracy/Macro-F1, but not train Accuracy, train Macro-F1, Validation CE, per-epoch gradients, optimizer state, or intermediate weights.

| Dataset | Logged epochs | Best-Accuracy checkpoint epoch | First train loss | Train loss at best epoch | Minimum / final train loss | Best Validation Acc / F1 (%) | Last Validation Acc / F1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 118 | 88 | 2.9469 | 0.4636 | 0.1306 / 0.1306 | 54.23 / 45.27 | 52.04 / 44.21 |
| Grocery | 121 | 91 | 3.1943 | 0.1522 | 0.0449 / 0.0463 | 81.79 / 73.51 | 81.64 / 73.86 |
| ele-fashion | 165 | 135 | 2.6204 | 0.2558 | 0.1997 / 0.2096 | 87.64 / 74.72 | 87.53 / 75.60 |

Training loss kept falling after the best-Accuracy checkpoint. Movies also shows a visible Validation decline by the last logged epoch; Grocery and ele-fashion finish close in Accuracy, while Macro-F1 rises after the Accuracy-selected epoch. This supports retaining the saved best-Validation checkpoint and shows a metric tradeoff, but the missing training metrics and intermediate states prevent a full overfitting, underfitting, or stopping-rule diagnosis.

## E2/E3 initialization and cost control

Fresh E2 and E3 models constructed with the same seed were compared without loading weights or training. All 25 common pre-expert parameter tensors were bitwise equal for each dataset. Common scalar and LayerNorm tensors were also equal. The four common Fusion/Skip tensors (98,560 parameters total) were different, consistent with their later initialization occurring after the architectures consume different random draws. The exact tensor-by-tensor comparison is in `data/initialization_audit.json`.

| Quantity | E2 Full | E3 single-expert control | E3 − E2 |
|---|---:|---:|---:|
| Expert + Router/conditioning parameters | 157,611 | 157,814 | +203 (+0.13%) |
| Approximate expert/Router linear MACs per node across six calls | 941,568 | 944,640 | +3,072 (+0.33%) |

These are close parameter and linear-MAC controls, not strict capacity or optimization equivalence. E2 uses three routed experts plus a Router; E3 uses one wider expert. Their computation and gradient allocation differ, and the differing Fusion/Skip initialization affects 98,560 common parameters. The Pilot's one-seed result therefore cannot isolate routing from all initialization and architectural effects.

The original Pilot CSV corrects an overgeneralized sentence in the old `PILOT_REPORT.md`: E3 has the highest Validation Accuracy among the variants on Movies (0.553389) and Grocery (0.821083), while E4 is highest on ele-fashion (0.879002; E3 is 0.878081). This is Accuracy only; Macro-F1 rankings differ. The historical report and CSV were not modified.

## What the evidence can establish

The stored evidence rules out missing or non-finite first-epoch cross-modal/`rho` gradients in the recorded checks, and the selected checkpoints do not have a saturated `rho` parameter. It also shows that the trained cross-modal path has a small scale on two of three datasets and little frozen prediction impact. It cannot reconstruct optimization dynamics between epoch 1 and the selected checkpoint or distinguish a weakly initialized branch from one learned to be redundant. That requires new training records and replicated Validation comparisons; no such training was started in this task.
