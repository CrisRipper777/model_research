# Historical V2.1 Classwise and Prediction Diagnostics

## Scope and method

This is a validation-only reanalysis of the 27 historical V2.1 runs. It reloads each run's Best Accuracy and Best Macro-F1 checkpoint (54 reloads total), validates its dataset/condition/seed, source selection rule, V7B feature/candidate/cache fingerprints, and finiteness, then compares recomputed Validation Accuracy, Macro-F1, and CE with checkpoint metadata (absolute tolerance `2e-5`). Every comparison passed. The project Macro-F1 class set is retained: labels observed on Train or Validation, `zero_division=0`. Test labels are never indexed. Per-node predictions/CE exist only in process memory and are not exported.

Machine-readable outputs are `data/checkpoint_metrics.csv`, `data/classwise_metrics.csv`, `data/confusion_matrices.csv`, `data/prediction_transitions.csv`, `data/degree_group_metrics.csv`, `data/coldstart_epoch_comparison.csv`, `data/source_checkpoints.json`, and `data/phase_a_manifest.json`.

## Historical global results

Values are mean ± sample SD across seeds. `Best Accuracy` and `Best F1` use separate historical checkpoints; the latter is selected inside the Accuracy-driven stopping horizon.

### Best Accuracy checkpoint

| Dataset | Condition | Validation Accuracy | Macro-F1 | CE |
|---|---|---:|---:|---:|
| Movies | C0 | 55.87 ± 0.40% | 47.20 ± 1.21% | 1.4172 ± 0.0459 |
| Movies | C1 | 56.08 ± 0.39% | 47.84 ± 0.26% | 1.4255 ± 0.0717 |
| Movies | C2 | 55.81 ± 0.24% | 49.15 ± 0.41% | 1.3979 ± 0.0485 |
| Grocery | C0 | 83.35 ± 0.38% | 75.83 ± 1.63% | 0.6721 ± 0.0560 |
| Grocery | C1 | 83.60 ± 0.44% | 76.45 ± 0.52% | 0.6662 ± 0.0442 |
| Grocery | C2 | 83.38 ± 0.45% | 76.46 ± 1.24% | 0.6601 ± 0.0127 |
| ele-fashion | C0 | 87.55 ± 0.25% | 75.34 ± 0.44% | 0.4363 ± 0.0329 |
| ele-fashion | C1 | 87.43 ± 0.14% | 73.54 ± 0.51% | 0.4149 ± 0.0077 |
| ele-fashion | C2 | 87.32 ± 0.06% | 73.61 ± 0.46% | 0.4129 ± 0.0111 |

### Best Macro-F1 checkpoint

| Dataset | Condition | Validation Accuracy | Macro-F1 | CE |
|---|---|---:|---:|---:|
| Movies | C0 | 55.33 ± 0.67% | 49.89 ± 0.10% | 1.4225 ± 0.0341 |
| Movies | C1 | 54.64 ± 0.26% | 50.20 ± 0.07% | 1.4519 ± 0.0512 |
| Movies | C2 | 55.51 ± 0.54% | 49.89 ± 0.41% | 1.3928 ± 0.0447 |
| Grocery | C0 | 83.06 ± 0.42% | 76.79 ± 1.11% | 0.6864 ± 0.0682 |
| Grocery | C1 | 82.93 ± 0.46% | 76.76 ± 0.73% | 0.6878 ± 0.0279 |
| Grocery | C2 | 83.03 ± 0.41% | 76.59 ± 1.21% | 0.6704 ± 0.0296 |
| ele-fashion | C0 | 86.52 ± 0.20% | 76.38 ± 0.17% | 0.4509 ± 0.0314 |
| ele-fashion | C1 | 86.83 ± 0.30% | 75.61 ± 0.31% | 0.4060 ± 0.0032 |
| ele-fashion | C2 | 86.77 ± 0.27% | 75.45 ± 0.19% | 0.4092 ± 0.0002 |

## ele-fashion class pattern

The Validation class set has 11 observed labels (IDs 0–4 and 6–11); class 5 is absent from Train and Validation. The smallest observed Validation support is 45, so none of the included classes has fewer than 10 examples. Macro-F1 still gives each observed class equal weight, so the lower-support classes can have high variance and materially affect the macro average.

With the Best Accuracy checkpoint, C1−C0 mean F1 is most negative for class 11 (−0.101), class 1 (−0.045), class 7 (−0.016), class 3 (−0.016), and class 4 (−0.011). For classes 11 and 1, this coincides with a substantial recall drop (−0.087 and −0.061) despite higher precision (+0.181 and +0.080): the selectors make fewer predictions for these classes (about −62 and −108 predictions per seed, respectively). The C2−C0 pattern remains concentrated in classes 11 and 1 (F1 −0.106 and −0.041). With Best F1 checkpoints, losses are led by classes 1 (−0.047), 7 (−0.038), and 11 (−0.029); class 1 and 11 recall again falls. Thus, the classwise pattern persists across checkpoint criteria, but its rank and magnitude vary.

These are descriptive Validation comparisons. They do not establish why predictions shift or whether the selector directly acts on every affected node.

## Same-node changes and CE

For ele-fashion Best Accuracy, C1 vs C0 flips 3.32% of Validation predictions on average (414 wrong-to-right versus 450 right-to-wrong across seeds); C2 vs C0 flips 3.53% (426 versus 494). Per-node CE improves for about 32.5% of nodes in C1 vs C0 and 27.2% in C2 vs C0; average CE decreases overall, consistent with the lower global CE, while hard predictions and macro-F1 worsen. Best-F1 checkpoints show larger prediction changes: C1 vs C0 flips 5.75%, with 799 wrong-to-right and 708 right-to-wrong changes; CE improves for about 28.1% of nodes. The aggregate can therefore improve probabilistic confidence while reducing minority-class recall.

## Physical-degree subgroups

In ele-fashion Best Accuracy, C1−C0 Macro-F1 is −1.41 pp for degree 1 and −2.44 pp for degree ≥2; corresponding Validation Accuracy changes are −0.09 pp and −0.15 pp. Degree 2–3 has the largest negative subgroup Accuracy/F1 change (−0.50/−3.74 pp), while degree 4–7 and ≥8 are slightly positive on average. This is not a degree-1-only failure. The degree-1 predictions still flip 3.74% of the time, but these nodes cannot perform relative selection over multiple physical neighbors: their changes are indirect consequences of shared parameters and higher-order propagation. Best-F1 subgroup accuracy increases slightly while F1 decreases for both degree <2 and ≥2, reinforcing the importance of checkpoint criterion and class balance.

The degree-zero group contains one Validation node and is not interpretable; it is retained separately rather than merged. Exact counts and per-seed results are in the CSV.

## Train/Validation trajectory

On ele-fashion, C1/C2 Best Accuracy occurs at mean epochs 124/118 versus 176 for C0. At those checkpoints, Train Macro-F1 is 0.803/0.802 versus 0.871 for C0, while Validation Macro-F1 is 0.735/0.736 versus 0.753. C1/C2 also have lower Validation CE (0.415/0.413 versus 0.436). At Best-F1 epochs, Validation Macro-F1 narrows the difference (0.756/0.755 versus 0.764) but does not eliminate it. This jointly shows reduced observed training fit and lower validation F1 with lower validation CE and earlier selected epochs. It is consistent with changed class preference and/or optimization behavior, but does not identify a unique cause.

## Limits

Degree groups are observational partitions, not interventions. In particular, degree-1 metric changes do not mean those receivers selected among neighbors. Checkpoint selection itself uses Validation labels as specified by the historical protocol. This diagnosis is not a new training ablation and does not use Test results.
