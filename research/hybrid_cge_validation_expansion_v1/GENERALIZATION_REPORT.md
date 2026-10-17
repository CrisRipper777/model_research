# Hybrid-CGE generalization to Toys and Reddit-S

The single globally selected condition is H2-Frozen from the Movies/Grocery/ele-fashion Validation study. Toys and Reddit-S did not participate in configuration selection. All available results use Validation only.

## Aggregate results

| Dataset | n | Best-Accuracy Val Acc | Same-checkpoint Macro-F1 | Val CE | Best-F1 Macro-F1 | Train−Val Acc gap |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 3/3 | 80.33% | 77.47% | 0.6724 | 77.77% | 6.36% |
| Reddit-S | 3/3 | 96.60% | 93.02% | 0.1476 | 93.05% | 3.14% |

## Toys

20,695 nodes and 113,402 directed preprocessed physical edges. Configured text and visual dimensions are 768 each. Feature fingerprint: cf8514296e0f409894a053415e3120be04b0acb482b897b49df763a6344021a4. Physical edge fingerprint: 85d171974a0d28b8434963d0b8a421c84a2b6742d9b777c43a06211435d0db63. V7B cache SHA256: 03afd66b0b4062e09efe8678277b49dee92863837f57c76d0cb6b2a834053b09. Cache feature/candidate/V7B fingerprints: 969ac25017a2a60384eedf9aedef431e08b0cf029e538d1fcbb8aa81f74c4375 / 1235db05c59fbba08c0e29aff1a996c4e966568d4ce944ebef3ab22271ce6e90 / b151bba6e2d46714e2a26e5da936f1a50b60ab29048fef1a319c95387fca76f7.

| Seed | Best-Accuracy epoch | Val Acc | Same-checkpoint Macro-F1 | Val CE | Best-F1 epoch | Best-F1 Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 56 | 80.24% | 77.35% | 0.6800 | 63 | 77.65% |
| 43 | 58 | 80.07% | 77.45% | 0.6723 | 69 | 77.53% |
| 44 | 44 | 80.67% | 77.62% | 0.6650 | 55 | 78.14% |

## Reddit-S

15,894 nodes and 283,080 directed preprocessed physical edges. Configured text and visual dimensions are 768 each. Feature fingerprint: 1bb58da5d2dac5321112be2d06f0db0af70439c7aee4ac774a37c8e55ffa16c8. Physical edge fingerprint: 67c31a841618caefdb2b7df15a4eb1e1a853e66dc7ab12c9f2e7d22c217eba1d. V7B cache SHA256: 7ba9f372e9f906a3551f1511f58ffac7d41ee4dc131cd1deb932d701cb56b4b5. Cache feature/candidate/V7B fingerprints: f0a945c36bd2548a8131a22e707cbf1dd299243b53cae9e77efd4041c6537dcd / 908f636ae1add0cc99e4a08b55df70b3abb3c6e519e0a5d0d147f2d0f587d587 / 8bc381bc82598ebd736009582c1547207b56b9c193c58e31b3d9ab649a5bd79b.

| Seed | Best-Accuracy epoch | Val Acc | Same-checkpoint Macro-F1 | Val CE | Best-F1 epoch | Best-F1 Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 49 | 96.41% | 93.39% | 0.1391 | 52 | 93.47% |
| 43 | 53 | 96.89% | 93.22% | 0.1247 | 53 | 93.22% |
| 44 | 76 | 96.51% | 92.45% | 0.1789 | 76 | 92.45% |

Per-seed project split hashes, configured feature paths and file hashes, and source attribution are in data/data_audit.json. Per-run data/cache/checkpoint fingerprints are in data/runs.csv and data/checkpoint_selection_summary.csv. Test evaluation stayed disabled. No dataset-specific tuning or architecture change was applied.
