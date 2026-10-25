# MR-CGE V1.3 Physical Basis Refinement — Screening Report

## Scope and protocol

This report covers only the seven authorized Validation training cells. The Movies seed42 split fingerprint is `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d`; Grocery uses its frozen seed42 split fingerprint `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056`. The Hybrid/V1.3 comparison pools corrected Hybrid seed42 attempt 02 with new fixed-split seeds43/44, and V1.3-A seeds42/43 with new seed44. Historical Hybrid seeds43/44 on different splits are excluded.

Best Validation Accuracy is the primary checkpoint. Best Macro-F1 is recorded separately. Test labels were masked before Trainer entry and Test evaluation stayed off. Values below are descriptive; accuracy and Macro-F1 are percentages, CE is the recorded cross-entropy.

## Fixed-split Hybrid-CGE vs V1.3-A on Movies

| Model | Seed | Val Acc | Val Macro-F1 | Val CE | Train Acc | Best Acc / Best F1 epoch | Parameters | Seconds | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid-CGE H2 3-hop | 42 | 56.87% | 50.26% | 1.4031 | 78.73% | 102 / 100 | 2,273,270 | 45.1 | 4.29 |
| Hybrid-CGE H2 3-hop | 43 | 57.05% | 49.43% | 1.3445 | 74.01% | 84 / 67 | 2,273,270 | 54.3 | 4.29 |
| Hybrid-CGE H2 3-hop | 44 | 56.93% | 48.84% | 1.3362 | 69.52% | 74 / 91 | 2,273,270 | 50.2 | 4.29 |
| MR-CGE V1.3-A | 42 | 55.97% | 48.20% | 1.3704 | 72.76% | 74 / 90 | 2,067,935 | 45.9 | 2.63 |
| MR-CGE V1.3-A | 43 | 55.22% | 49.17% | 1.3798 | 73.77% | 73 / 66 | 2,067,935 | 45.5 | 2.63 |
| MR-CGE V1.3-A | 44 | 55.85% | 48.03% | 1.3542 | 70.66% | 66 / 83 | 2,067,935 | 53.3 | 2.75 |

Best-F1 checkpoint metrics are independent of the Best-Accuracy rows above:

| Model | Seed | Best-F1 checkpoint Val Acc | Val Macro-F1 | Val CE |
|---|---:|---:|---:|---:|
| Hybrid-CGE H2 3-hop | 42 | 55.76% | 51.98% | 1.4110 |
| Hybrid-CGE H2 3-hop | 43 | 54.47% | 51.76% | 1.3746 |
| Hybrid-CGE H2 3-hop | 44 | 55.43% | 51.64% | 1.3865 |
| MR-CGE V1.3-A | 42 | 55.22% | 49.29% | 1.4371 |
| MR-CGE V1.3-A | 43 | 54.62% | 50.38% | 1.3901 |
| MR-CGE V1.3-A | 44 | 54.98% | 49.62% | 1.4222 |

| Best-Accuracy metric | Hybrid mean ± population SD | V1.3-A mean ± population SD | Hybrid − V1.3-A, paired by seed (mean ± population SD) |
|---|---:|---:|---:|
| Validation Accuracy | 56.9486% ± 0.0748 pp | 55.6789% ± 0.3289 pp | +1.2697 pp ± 0.4027 pp |
| Validation Macro-F1 | 49.5118% ± 0.5792 pp | 48.4678% ± 0.5018 pp | +1.0440 pp ± 0.7521 pp |
| Validation CE | 1.36127 ± 0.02975 | 1.36813 ± 0.01057 | −0.00686 ± 0.02885 |

Per-seed Hybrid minus V1.3-A deltas (percentage points for Acc/F1):

| Seed | Δ Acc | Δ Macro-F1 | Δ CE |
|---:|---:|---:|---:|
| 42 | +0.90 pp | +2.06 pp | +0.03270 |
| 43 | +1.83 pp | +0.26 pp | −0.03530 |
| 44 | +1.08 pp | +0.81 pp | −0.01799 |

After split alignment, Hybrid-CGE has a descriptive mean advantage of 1.27 pp in Accuracy and 1.04 pp in Macro-F1; mean CE is lower by 0.00686. The three seeds show variation, especially in F1 and CE. Same split, training seeds and task improve comparability, but the two model families have different parameterizations and initialization computation order. This is not a parameter-initialization-matched causal comparison and does not establish statistical significance.

## Physical NoP0 vs V1.3-A

The primary comparison uses Best-Accuracy checkpoints. Deltas are NoP0 minus the frozen V1.3-A result on the same dataset and seed. Acc/F1 deltas are percentage points.

| Dataset | Seed | V1.3-A Acc / F1 / CE | NoP0 Acc / F1 / CE | Δ Acc | Δ F1 | Δ CE |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 55.97% / 48.20% / 1.3704 | 55.49% / 47.54% / 1.4120 | −0.48 pp | −0.66 pp | +0.04159 |
| Movies | 43 | 55.22% / 49.17% / 1.3798 | 54.86% / 47.41% / 1.3727 | −0.36 pp | −1.76 pp | −0.00708 |
| Movies mean (n=2) | — | 55.59% / 48.68% / 1.3751 | 55.17% / 47.48% / 1.3924 | **−0.42 pp** | **−1.21 pp** | **+0.01726** |
| Grocery | 42 | 83.51% / 76.62% / 0.6392 | 83.10% / 76.37% / 0.7042 | −0.41 pp | −0.25 pp | +0.06497 |
| Grocery | 43 | 83.69% / 76.94% / 0.6456 | 83.16% / 77.06% / 0.6591 | −0.53 pp | +0.13 pp | +0.01347 |
| Grocery mean (n=2) | — | 83.60% / 76.78% / 0.6424 | 83.13% / 76.72% / 0.6816 | **−0.47 pp** | **−0.06 pp** | **+0.03922** |

The population SD of the two seed-paired deltas is Movies: Acc 0.060 pp, Macro-F1 0.549 pp, CE 0.02434; Grocery: Acc 0.059 pp, Macro-F1 0.188 pp, CE 0.02575.

Train/Validation gaps and resource measurements are shown below. Gaps are Train minus Validation; CE gaps are negative when Validation CE is higher. Historical V1.3-A rows did not retain Train Macro-F1, so that gap is shown only for NoP0.

| Dataset | Seed | Model | Train Acc / Macro-F1 / CE | Acc / Macro-F1 / CE gap | Parameters | Seconds | Peak reserved GiB |
|---|---:|---|---:|---:|---:|---:|---:|
| Movies | 42 | V1.3-A | 72.76% / n/a / 0.7879 | +16.79 pp / n/a / −0.5824 | 2,067,935 | 45.9 | 2.63 |
| Movies | 42 | NoP0 | 75.24% / 76.13% / 0.7247 | +19.75 pp / +28.59 pp / −0.6873 | 2,067,935 | 57.0 | 2.75 |
| Movies | 43 | V1.3-A | 74.77% / n/a / 0.7402 | +19.55 pp / n/a / −0.6396 | 2,067,935 | 45.5 | 2.63 |
| Movies | 43 | NoP0 | 70.75% / 70.17% / 0.8476 | +15.89 pp / +22.76 pp / −0.5251 | 2,067,935 | 47.3 | 2.75 |
| Grocery | 42 | V1.3-A | 94.30% / n/a / 0.2149 | +10.79 pp / n/a / −0.4243 | 2,067,935 | 46.0 | 2.59 |
| Grocery | 42 | NoP0 | 97.75% / 97.16% / 0.1010 | +14.64 pp / +20.79 pp / −0.6032 | 2,067,935 | 73.0 | 2.69 |
| Grocery | 43 | V1.3-A | 94.14% / n/a / 0.2190 | +10.45 pp / n/a / −0.4266 | 2,067,935 | 46.1 | 2.59 |
| Grocery | 43 | NoP0 | 93.06% / 90.14% / 0.2465 | +9.90 pp / +13.08 pp / −0.4126 | 2,067,935 | 54.8 | 2.69 |

NoP0 preserves the parameter count. Compared with the baseline, new runs used 0.12 GiB more peak reserved memory on Movies and 0.11 GiB more on Grocery. Per-run duration varied; the two-seed means were Movies 52.2 s vs 45.7 s and Grocery 63.9 s vs 46.0 s.

The Best-F1 checkpoint metrics are also retained separately:

| Dataset | Seed | Model | Best-F1 epoch | Val Acc | Val Macro-F1 | Val CE |
|---|---:|---|---:|---:|---:|---:|
| Movies | 42 | V1.3-A | 90 | 55.22% | 49.29% | 1.4371 |
| Movies | 42 | NoP0 | 55 | 55.10% | 48.49% | 1.3688 |
| Movies | 43 | V1.3-A | 66 | 54.62% | 50.38% | 1.3901 |
| Movies | 43 | NoP0 | 53 | 54.23% | 48.52% | 1.3843 |
| Grocery | 42 | V1.3-A | 73 | 83.31% | 76.88% | 0.6369 |
| Grocery | 42 | NoP0 | 131 | 83.07% | 77.68% | 0.7822 |
| Grocery | 43 | V1.3-A | 87 | 83.48% | 77.73% | 0.6733 |
| Grocery | 43 | NoP0 | 87 | 83.02% | 77.86% | 0.6842 |

NoP0 did not improve either dataset's two-seed mean on the primary Best-Accuracy rule. Movies Accuracy fell 0.42 pp and Macro-F1 fell 1.21 pp; Grocery Accuracy fell 0.47 pp, Macro-F1 was nearly unchanged (−0.06 pp), and CE increased by 0.0392. The Best-F1 checkpoint rows give a similar picture: Movies Macro-F1 is lower in both seeds; Grocery is mixed by seed and does not provide a consistent gain.

## Effective coefficients and response diagnostics

The tables use Best-Accuracy checkpoint diagnostics. Effective coefficient matrices below are rounded to three decimals; each row lists experts 0–3 in order. Baseline Physical rows show orders 0–3; NoP0 rows show orders 1–3 because order 0 is exactly zero. Semantic rows show orders 0–1; columns for orders 2–3 remain zero. The baseline rows were recalculated from the strict frozen V1.3-A Best-Accuracy checkpoints, whose path and SHA256 are recorded alongside raw `theta` in `data/coefficients_diagnostics.csv`.

| Dataset | Seed | Condition | Effective Physical α, experts 0–3 | Effective Semantic α, experts 0–3 |
|---|---:|---|---|---|
| Movies | 42 | V1.3-A | `[.540,-.225,.166,-.068]; [-.248,.524,-.091,.137]; [.200,.146,-.408,.246]; [-.107,-.164,.334,.394]` | `[.706,-.294]; [-.321,.679]; [.578,.422]; [-.395,-.605]` |
| Movies | 42 | NoP0 | `[0,-.463,.377,-.160]; [0,.702,-.124,.174]; [0,.183,-.519,.298]; [0,-.170,.381,.449]` | `[.700,-.300]; [-.314,.686]; [.596,.404]; [-.371,-.629]` |
| Movies | 43 | V1.3-A | `[.524,-.239,.158,-.078]; [-.247,.516,-.085,.152]; [.201,.153,-.397,.249]; [-.109,-.155,.338,.399]` | `[.687,-.313]; [-.324,.676]; [.569,.431]; [-.413,-.587]` |
| Movies | 43 | NoP0 | `[0,-.478,.354,-.168]; [0,.681,-.137,.182]; [0,.177,-.528,.296]; [0,-.179,.368,.453]` | `[.694,-.306]; [-.322,.678]; [.603,.397]; [-.390,-.610]` |
| Grocery | 42 | V1.3-A | `[.549,-.215,.173,-.063]; [-.237,.514,-.072,.176]; [.198,.141,-.413,.248]; [-.121,-.148,.326,.404]` | `[.719,-.281]; [-.316,.684]; [.584,.416]; [-.450,-.550]` |
| Grocery | 42 | NoP0 | `[0,-.501,.319,-.181]; [0,.652,-.112,.235]; [0,.200,-.474,.326]; [0,-.192,.308,.500]` | `[.683,-.317]; [-.369,.631]; [.559,.441]; [-.395,-.605]` |
| Grocery | 43 | V1.3-A | `[.556,-.210,.174,-.060]; [-.248,.503,-.083,.167]; [.193,.140,-.423,.243]; [-.112,-.157,.331,.400]` | `[.725,-.275]; [-.330,.670]; [.580,.420]; [-.418,-.582]` |
| Grocery | 43 | NoP0 | `[0,-.484,.342,-.174]; [0,.651,-.109,.240]; [0,.179,-.515,.305]; [0,-.178,.321,.500]` | `[.703,-.297]; [-.380,.620]; [.578,.422]; [-.483,-.517]` |

`sP` and `sS` are the mean Joint View strengths for the Private branch. Weighted Physical RMS values are reported as Text/Visual pairs. `Cross calibrated` is the Early Physical Cross calibrated RMS; `cross response P→C` gives Private then Collaborative response RMS for NoP0. Late is the Late residual ratio. All pairs are Text / Visual.

| Dataset | Seed | Condition | Private sP | Private sS | Weighted Physical RMS, private | Weighted Physical RMS, collaborative | Cross calibrated RMS | Cross response P→C RMS | Late residual ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | V1.3-A | 1.526 / 1.825 | 1.204 / 1.418 | .350 / .432 | .342 / .428 | .784 / .710 | n/a | .078 / .034 |
| Movies | 42 | NoP0 | 1.807 / 1.965 | 1.458 / 1.744 | .412 / .486 | .318 / .498 | .769 / .699 | .892→.741 / .948→.972 | .139 / .052 |
| Movies | 43 | V1.3-A | 1.848 / 1.946 | .350 / .192 | .429 / .462 | .435 / .467 | .792 / .707 | n/a | .037 / .043 |
| Movies | 43 | NoP0 | 1.684 / 1.927 | .760 / .571 | .386 / .464 | .262 / .476 | .778 / .719 | .894→.666 / .931→.956 | .133 / .055 |
| Grocery | 42 | V1.3-A | 1.764 / 1.812 | 1.771 / 1.815 | .421 / .444 | .413 / .450 | .682 / .640 | n/a | .078 / .076 |
| Grocery | 42 | NoP0 | 1.956 / 1.922 | 1.969 / 1.938 | .494 / .489 | .477 / .486 | .677 / .624 | .971→.939 / .968→.967 | .136 / .121 |
| Grocery | 43 | V1.3-A | 1.753 / 1.856 | 1.750 / 1.850 | .424 / .453 | .421 / .452 | .682 / .639 | n/a | .092 / .077 |
| Grocery | 43 | NoP0 | 1.920 / 1.883 | 1.917 / 1.881 | .479 / .478 | .476 / .482 | .685 / .625 | .958→.954 / .965→.977 | .067 / .066 |

The NoP0 coefficient audit confirms exact Physical `α[0]=0` for all four experts in both Private and Collaborative paths, while Semantic `α[0:2]` remain active and `theta` remains trainable. The Early Cross calibrated RMS and the Private-to-Collaborative response changed along with the renormalized first-order Physical coefficient. This is an expected coupled consequence of NoP0, not an independently varied Cross intervention. These diagnostics describe the trained models and do not establish that a Router, coefficient or Gate change caused a metric difference.

## Interpretation boundary

The fixed-split Hybrid comparison supports a descriptive remaining performance gap after aligning split and training seed; it does not resolve statistical significance or a final model selection. The NoP0 screen does not support an average improvement on Movies and does not show a consistent Grocery benefit. Since removing Physical order zero also renormalizes orders 1–3 and changes the Early Cross response, this is a joint mechanism screen rather than an isolated estimate of the zero-order term's causal contribution. Only two NoP0 seeds were run, as authorized.

Machine-readable result tables and checkpoint hashes are in `data/fixed_split_hybrid_comparison.csv`, `data/paired_differences.csv`, `data/dataset_summary.csv`, `data/runs.csv`, `data/coefficients_diagnostics.csv`, and `data/view_diagnostics.csv`.
