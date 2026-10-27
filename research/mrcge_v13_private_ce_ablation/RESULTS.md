# Private Auxiliary CE Ablation Results

Branch `exp/mrcge_v13_private_ce_ablation`; base `a3bf807f6bb9c1330a5276acbee70836bb140086`; training source SHA `82c76e5d30caca257ea5cd49b2b23e2601f6e579`.

Only `task.loss.unimodal_weight` changed (0.3 to 0.0). The model is the original `mrcge_v13.Model`, M_repaired / Joint View. Results are Validation-only and exploratory.

## Per-seed paired Validation metrics

| Dataset | Seed | New Acc | New Macro-F1 | New CE | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 55.909% | 47.357% | 1.41669 | -0.060 | -0.841 | +0.04632 |
| Movies | 43 | 55.489% | 45.336% | 1.40678 | +0.270 | -3.835 | +0.02696 |
| Movies | 44 | 55.189% | 46.925% | 1.39125 | -0.660 | -1.110 | +0.03704 |
| Grocery | 42 | 83.075% | 76.499% | 0.68543 | -0.439 | -0.123 | +0.04619 |
| Grocery | 43 | 83.631% | 77.297% | 0.65489 | -0.059 | +0.362 | +0.00931 |

## Dataset summary

| Dataset | n | New Acc mean ± SD | New F1 mean ± SD | New CE mean ± SD | Δ Acc mean ± SD (pp) | Δ F1 mean ± SD (pp) | Δ CE mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 3 | 55.529% ± 0.295 | 46.539% ± 0.869 | 1.40491 ± 0.01047 | -0.150 ± 0.385 | -1.929 ± 1.353 | +0.03677 ± 0.00790 |
| Grocery | 2 | 83.353% ± 0.278 | 76.898% ± 0.399 | 0.67016 ± 0.01527 | -0.249 ± 0.190 | +0.120 ± 0.242 | +0.02775 ± 0.01844 |

## Generalization

Train−Validation gap is defined as Train minus Validation. Negative CE gap magnitude is the train/validation CE separation.

| Dataset | Historical Train CE | New Train CE | Δ Train CE | Historical Val CE | New Val CE | Δ Val CE | Historical Acc gap | New Acc gap | Δ Acc gap | Historical CE gap | New CE gap | Δ CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.78934 | 0.72598 | -0.06335 | 1.36813 | 1.40491 | +0.03677 | 17.05 pp | 19.34 pp | +2.29 pp | -0.57880 | -0.67893 | -0.10013 |
| Grocery | 0.21696 | 0.18999 | -0.02697 | 0.64241 | 0.67016 | +0.02775 | 10.62 pp | 11.84 pp | +1.22 pp | -0.42544 | -0.48017 | -0.05472 |

## Protocol and checkpoint audit

Five cells ran serially with separate local checkpoint directories. Both Best-Accuracy and Best-F1 checkpoints were reloaded and compared to Trainer metadata; all ten reload audits passed. Same-seed model and external classifier initialization hashes matched the historical V1.3-A records for all five cells. The CPU smoke used the original `src.models.mrcge_v13.Model`; the zero-weight loss was finite, Projector gradients were finite/nonzero, and Private heads/logits remained present without auxiliary-head gradients.

All runs used physical GPU 1 (RTX 3090, UUID recorded per row). Mean wall time was 42.7 s for Movies and 44.1 s for Grocery; peak allocated/reserved memory was 2042/2850 MiB and 2091/2794 MiB, respectively. Test labels were masked before Trainer entry; `evaluate_test=false` and Link Prediction was not run. Fixed seed-42 splits and the V7B.1 K=8 cache were reused. See `data/manifest.json` for source/config/data hashes and `data/runs.csv` for checkpoint hashes, training times, GPU snapshots, and per-run gaps. Best-F1 metrics and SHA256 values are recorded separately in `data/runs.csv`.

## Conclusion

**KEEP_0.3**. Removing Private CE lowered mean Validation Accuracy on both datasets (−0.150 pp Movies; −0.249 pp Grocery) and raised Validation CE in all five paired seeds. Movies Macro-F1 also fell in all three seeds (mean −1.929 pp); Grocery Macro-F1 increased slightly on average (+0.120 pp), with opposite signs across its two seeds. Train CE fell, while Validation CE and Train−Validation gaps worsened on both datasets. This supports retaining 0.3 for now. The seed counts are small and the Accuracy differences are modest, so this is a descriptive ablation result, not a significance claim. No additional unimodal weights or other ablations are part of this study.
