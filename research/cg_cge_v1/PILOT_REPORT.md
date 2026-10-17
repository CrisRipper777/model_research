# CG-CGE V1 Pilot Report

Validation-only report. Test evaluation was disabled; Test labels were masked before training.
All B1/B2 results use the frozen new implementation and internal paired baseline. B0 is included only where the full protocol fingerprint gate passed.
Implementation Freeze Commit: `aac21ef7d4f8d4e25b2dd61cf1ed5fd6b8a67556`; formal source commit: `c3d32ae3b32e331cb4390b05823dbd36a50d8162`.

## Best-Accuracy checkpoint: Validation metrics

| Dataset | Condition | Accuracy % | Macro-F1 % | CE | Train–Val Acc gap pp | Train–Val F1 gap pp | Train–Val CE gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | B1_complementary | 56.519 ± 0.360 | 49.611 ± 1.226 | 1.375 ± 0.045 | 17.379 ± 5.374 | 23.069 ± 7.871 | -0.622 ± 0.180 |
| Movies | B2_conditioned_experts | 56.469 ± 0.396 | 49.581 ± 1.641 | 1.374 ± 0.035 | 17.562 ± 3.433 | 23.372 ± 5.017 | -0.623 ± 0.124 |
| Movies | A2_cross_source | 55.959 ± 0.241 | 47.953 ± 0.534 | 1.392 ± 0.053 | 17.113 ± 7.169 | 24.010 ± 9.072 | -0.606 ± 0.249 |
| Grocery | B1_complementary | 83.631 ± 0.442 | 76.416 ± 0.923 | 0.648 ± 0.054 | 9.366 ± 3.542 | 13.872 ± 5.539 | -0.405 ± 0.148 |
| Grocery | B2_conditioned_experts | 83.660 ± 0.487 | 76.158 ± 0.836 | 0.632 ± 0.030 | 8.875 ± 2.010 | 13.619 ± 3.319 | -0.375 ± 0.074 |
| Grocery | A2_cross_source | 83.514 ± 0.456 | 76.276 ± 0.942 | 0.666 ± 0.055 | 10.411 ± 3.148 | 15.298 ± 4.914 | -0.451 ± 0.145 |
| ele-fashion | B1_complementary | 87.576 ± 0.189 | 74.316 ± 0.277 | 0.414 ± 0.005 | 2.565 ± 0.248 | 8.493 ± 0.629 | -0.135 ± 0.005 |
| ele-fashion | B2_conditioned_experts | 87.433 ± 0.222 | 74.053 ± 0.341 | 0.399 ± 0.005 | 2.067 ± 0.496 | 6.873 ± 1.158 | -0.099 ± 0.016 |
| ele-fashion | A2_cross_source | 87.665 ± 0.099 | 75.124 ± 0.939 | 0.430 ± 0.024 | 3.154 ± 0.914 | 9.529 ± 1.352 | -0.174 ± 0.034 |

## Best-F1 checkpoint

| Dataset | Condition | Accuracy % | Macro-F1 % | CE |
|---|---|---:|---:|---:|
| Movies | B1_complementary | 55.659 ± 0.862 | 50.913 ± 0.563 | 1.394 ± 0.070 |
| Movies | B2_conditioned_experts | 54.219 ± 1.310 | 50.823 ± 1.031 | 1.470 ± 0.061 |
| Movies | A2_cross_source | 54.759 ± 0.618 | 49.571 ± 0.742 | 1.440 ± 0.037 |
| Grocery | B1_complementary | 83.153 ± 0.597 | 76.825 ± 0.845 | 0.669 ± 0.064 |
| Grocery | B2_conditioned_experts | 83.153 ± 0.208 | 76.781 ± 0.815 | 0.661 ± 0.035 |
| Grocery | A2_cross_source | 83.289 ± 0.476 | 77.168 ± 1.306 | 0.695 ± 0.072 |
| ele-fashion | B1_complementary | 86.530 ± 0.610 | 75.910 ± 0.365 | 0.422 ± 0.017 |
| ele-fashion | B2_conditioned_experts | 86.823 ± 0.355 | 75.284 ± 0.368 | 0.415 ± 0.012 |
| ele-fashion | A2_cross_source | 86.918 ± 0.384 | 76.481 ± 0.250 | 0.435 ± 0.016 |

## Paired differences

Differences are candidate minus reference; accuracy and Macro-F1 are percentage points, CE is raw. Mean/sample SD and positive/negative/zero seed counts are descriptive for n=3.

| Dataset | Contrast | Δ Accuracy pp (mean ± SD; +/−/0) | Δ Macro-F1 pp (mean ± SD; +/−/0) | Δ CE (mean ± SD; +/−/0) |
|---|---|---:|---:|---:|
| Movies | B1−B0 | 0.560 ± 0.125; 3/0/0 | 1.657 ± 1.489; 3/0/0 | -0.017 ± 0.097; 2/1/0 |
| Movies | B2−B0 | 0.510 ± 0.159; 3/0/0 | 1.628 ± 2.099; 2/1/0 | -0.019 ± 0.088; 2/1/0 |
| Movies | B2−B1 | -0.050 ± 0.092; 1/2/0 | -0.030 ± 0.954; 2/1/0 | -0.002 ± 0.010; 1/2/0 |
| Grocery | B1−B0 | 0.117 ± 0.183; 2/1/0 | 0.140 ± 0.911; 1/2/0 | -0.018 ± 0.016; 0/3/0 |
| Grocery | B2−B0 | 0.146 ± 0.077; 3/0/0 | -0.118 ± 0.775; 1/2/0 | -0.033 ± 0.030; 0/3/0 |
| Grocery | B2−B1 | 0.029 ± 0.128; 2/1/0 | -0.259 ± 0.141; 0/3/0 | -0.015 ± 0.025; 1/2/0 |
| ele-fashion | B1−B0 | -0.089 ± 0.287; 1/1/1 | -0.808 ± 0.747; 0/3/0 | -0.017 ± 0.023; 1/2/0 |
| ele-fashion | B2−B0 | -0.232 ± 0.304; 1/2/0 | -1.071 ± 0.887; 0/3/0 | -0.031 ± 0.019; 0/3/0 |
| ele-fashion | B2−B1 | -0.143 ± 0.118; 0/3/0 | -0.263 ± 0.177; 0/3/0 | -0.014 ± 0.005; 0/3/0 |

## Individual formal runs

| Dataset | Condition | Seed | Val Acc % | Val Macro-F1 % | Val CE | Best Acc epoch | Best F1 epoch | Wall sec | Peak allocated GiB | Peak reserved GiB | Film scale/shift update norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | B1_complementary | 42 | 56.539 | 49.439 | 1.4107 | 98 | 73 | 46.3 | 3.66 | 4.22 | 0/0 |
| Movies | B1_complementary | 43 | 56.149 | 48.479 | 1.3900 | 82 | 100 | 39.3 | 3.66 | 4.22 | 0/0 |
| Movies | B1_complementary | 44 | 56.869 | 50.913 | 1.3252 | 64 | 71 | 34.2 | 3.66 | 4.22 | 0/0 |
| Movies | B2_conditioned_experts | 42 | 56.389 | 48.308 | 1.3988 | 99 | 100 | 47.9 | 3.66 | 4.21 | 0.755/0.593 |
| Movies | B2_conditioned_experts | 43 | 56.119 | 49.001 | 1.3892 | 77 | 102 | 38.9 | 3.66 | 4.21 | 0.631/0.497 |
| Movies | B2_conditioned_experts | 44 | 56.899 | 51.433 | 1.3333 | 74 | 85 | 41.2 | 3.66 | 4.21 | 0.645/0.523 |
| Grocery | B1_complementary | 42 | 83.572 | 77.059 | 0.6346 | 78 | 94 | 37.8 | 3.57 | 4.09 | 0/0 |
| Grocery | B1_complementary | 43 | 84.100 | 76.832 | 0.6015 | 58 | 57 | 31.7 | 3.57 | 4.09 | 0/0 |
| Grocery | B1_complementary | 44 | 83.221 | 75.358 | 0.7071 | 101 | 119 | 52.1 | 3.57 | 4.09 | 0/0 |
| Grocery | B2_conditioned_experts | 42 | 83.455 | 76.638 | 0.6304 | 75 | 89 | 39.2 | 3.57 | 4.09 | 0.69/0.549 |
| Grocery | B2_conditioned_experts | 43 | 84.217 | 76.643 | 0.6034 | 65 | 88 | 35.9 | 3.57 | 4.09 | 0.677/0.564 |
| Grocery | B2_conditioned_experts | 44 | 83.309 | 75.192 | 0.6637 | 82 | 97 | 42.3 | 3.57 | 4.09 | 0.776/0.598 |
| ele-fashion | B1_complementary | 42 | 87.737 | 74.457 | 0.4158 | 145 | 154 | 247.0 | 16.72 | 18.27 | 0/0 |
| ele-fashion | B1_complementary | 43 | 87.624 | 73.997 | 0.4170 | 142 | 169 | 247.3 | 16.72 | 18.27 | 0/0 |
| ele-fashion | B1_complementary | 44 | 87.368 | 74.493 | 0.4081 | 129 | 144 | 241.0 | 16.72 | 18.27 | 0/0 |
| ele-fashion | B2_conditioned_experts | 42 | 87.685 | 74.385 | 0.3959 | 130 | 138 | 243.5 | 16.72 | 18.31 | 1.5/0.881 |
| ele-fashion | B2_conditioned_experts | 43 | 87.348 | 73.704 | 0.4055 | 101 | 123 | 205.3 | 16.72 | 18.31 | 1.14/0.75 |
| ele-fashion | B2_conditioned_experts | 44 | 87.266 | 74.070 | 0.3962 | 114 | 140 | 225.2 | 16.72 | 18.31 | 1.42/0.885 |

## Resources and execution environment

| Dataset | Condition | Wall seconds mean ± SD | Mean epoch seconds | Peak allocated GiB | Peak reserved GiB | Trainable model params | Optimizer params |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | B1_complementary | 39.960 ± 6.058 | 0.286 | 3.66 | 4.22 | 2268128 | 2273268 |
| Movies | B2_conditioned_experts | 42.698 ± 4.649 | 0.302 | 3.66 | 4.21 | 2276320 | 2281460 |
| Grocery | B1_complementary | 40.544 ± 10.467 | 0.289 | 3.57 | 4.09 | 2268128 | 2273268 |
| Grocery | B2_conditioned_experts | 39.143 ± 3.234 | 0.297 | 3.57 | 4.09 | 2276320 | 2281460 |
| ele-fashion | B1_complementary | 245.111 ± 3.539 | 1.264 | 16.72 | 18.27 | 2001872 | 2004956 |
| ele-fashion | B2_conditioned_experts | 224.685 ± 19.124 | 1.356 | 16.72 | 18.31 | 2010064 | 2013148 |

Formal runs used physical GPU0: NVIDIA GeForce RTX 3090, driver 535.309.01, 24 GiB. Peak accepted-run reservation was 18.31 GiB. GPU1 was not used; the preflight snapshot showed 13787 MiB and 98% utilization there.
Runtime: Python 3.12.13; PyTorch 2.4.0+cu121; CUDA build 12.1; cuDNN 90100; NumPy 2.4.3. `matmul_tf32=false`, `cudnn_tf32=true`, deterministic algorithms `false`, `CUBLAS_WORKSPACE_CONFIG=None`.
All 36 checkpoint files passed stored SHA256 and byte-count verification. The peak reserved memory remained below the 24 GiB device capacity; no formal run recorded OOM.

## Optimization and mechanism diagnostics

- B1 and B2 use the same dropout (0.2), AdamW learning rate (1e-3), and weight decay (1e-4). This whole-model comparison does not identify dropout or weight-decay main effects or their interaction; a factorial study would be needed.
- Per-epoch main CE, unimodal CE, balance auxiliary loss, Train/Validation Accuracy/F1/CE, and timing are in `data/epoch_history.csv`.
- Mechanism activation does not establish causal utility. The three-seed paired comparisons are descriptive and do not prove population-level significance.
- Test labels and metrics are excluded.

### Auxiliary loss and checkpoint selection

| Dataset | Condition | First-epoch mean unimodal CE | Best-Accuracy-epoch mean unimodal CE | Best-F1 minus Best-Accuracy Δ Acc pp | Δ Macro-F1 pp | Δ CE |
|---|---|---:|---:|---:|---:|---:|
| Movies | B1_complementary | 3.3037 ± 0.1953 | 1.2423 ± 0.0477 | -0.8598 ± 0.5284 | 1.3023 ± 1.1934 | 0.0183 ± 0.0570 |
| Movies | B2_conditioned_experts | 3.3037 ± 0.1953 | 1.2384 ± 0.0310 | -2.2496 ± 1.0975 | 1.2422 ± 1.1484 | 0.0960 ± 0.0451 |
| Grocery | B1_complementary | 3.2801 ± 0.0118 | 0.5639 ± 0.0632 | -0.4783 ± 0.1613 | 0.4085 ± 0.0782 | 0.0214 ± 0.0148 |
| Grocery | B2_conditioned_experts | 3.2801 ± 0.0118 | 0.5741 ± 0.0270 | -0.5076 ± 0.3265 | 0.6228 ± 0.1727 | 0.0284 ± 0.0046 |
| ele-fashion | B1_complementary | 2.6861 ± 0.1103 | 0.4433 ± 0.0009 | -1.0467 ± 0.7425 | 1.5946 ± 0.1704 | 0.0080 ± 0.0142 |
| ele-fashion | B2_conditioned_experts | 2.6861 ± 0.1103 | 0.4507 ± 0.0079 | -0.6103 ± 0.2122 | 1.2310 ± 0.2663 | 0.0157 ± 0.0140 |

### Physical / Semantic / Complementary path diagnostics

Means across the three Best-Accuracy checkpoints; cross source weights include unsupported rows as exact zero.

| Dataset | Condition | Modality | Physical profile RMS | Same-source response RMS | P/S/None weight | Raw P/S RMS | Calibrated P/S RMS | Gate mean / saturated | Cross/U RMS | Attention entropy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | B1_complementary | text | 1.001 | 1.232 | 0.985/0.011/0.004 | 0.5261/0.5967 | 0.5261/0.5261 | 0.782/0.367 | 0.08216 | 0.9275 |
| Movies | B1_complementary | visual | 1 | 1.196 | 0.642/0.168/0.190 | 0.4974/0.6079 | 0.4974/0.4974 | 0.757/0.324 | 0.05693 | 1.05 |
| Movies | B2_conditioned_experts | text | 1.001 | 1.23 | 0.986/0.010/0.003 | 0.5336/0.6079 | 0.5336/0.5336 | 0.785/0.380 | 0.0843 | 0.9195 |
| Movies | B2_conditioned_experts | visual | 1.001 | 1.192 | 0.658/0.160/0.182 | 0.4911/0.5986 | 0.4911/0.4911 | 0.755/0.319 | 0.05512 | 1.003 |
| Grocery | B1_complementary | text | 0.998 | 1.211 | 0.818/0.180/0.002 | 0.7605/0.8344 | 0.7605/0.7605 | 0.769/0.487 | 0.07957 | 1.703 |
| Grocery | B1_complementary | visual | 0.9979 | 1.366 | 0.406/0.581/0.013 | 0.9254/1.009 | 0.9254/0.9254 | 0.754/0.510 | 0.08132 | 1.211 |
| Grocery | B2_conditioned_experts | text | 0.9985 | 1.192 | 0.840/0.155/0.005 | 0.6929/0.7518 | 0.6929/0.6929 | 0.764/0.491 | 0.07162 | 1.832 |
| Grocery | B2_conditioned_experts | visual | 0.9984 | 1.32 | 0.398/0.585/0.018 | 0.8263/0.9028 | 0.8263/0.8263 | 0.749/0.505 | 0.07128 | 1.166 |
| ele-fashion | B1_complementary | text | 1.027 | 2.298 | 0.697/0.272/0.030 | 1.421/1.562 | 1.421/1.421 | 0.823/0.566 | 0.2107 | 1.169 |
| ele-fashion | B1_complementary | visual | 1.027 | 1.536 | 0.148/0.839/0.013 | 1.387/1.547 | 1.387/1.387 | 0.857/0.714 | 0.2426 | 1.291 |
| ele-fashion | B2_conditioned_experts | text | 1.023 | 2.192 | 0.705/0.258/0.037 | 1.336/1.465 | 1.336/1.336 | 0.810/0.522 | 0.1908 | 1.363 |
| ele-fashion | B2_conditioned_experts | visual | 1.022 | 1.492 | 0.126/0.862/0.012 | 1.291/1.442 | 1.291/1.291 | 0.843/0.700 | 0.2156 | 1.45 |

### Shared-expert selection by call path

Mean Top-2 selection share across Best-Accuracy checkpoints, modalities, datasets and seeds. Shares are descriptive and do not prove expert specialization.
| Condition | Call path | Expert 0 | Expert 1 | Expert 2 | Expert 3 |
|---|---|---:|---:|---:|---:|
| B1_complementary | Physical | 0.377 | 0.252 | 0.148 | 0.223 |
| B1_complementary | Same-source Semantic | 0.223 | 0.280 | 0.232 | 0.265 |
| B1_complementary | Complementary | 0.269 | 0.261 | 0.230 | 0.240 |
| B2_conditioned_experts | Physical | 0.379 | 0.223 | 0.163 | 0.235 |
| B2_conditioned_experts | Same-source Semantic | 0.231 | 0.259 | 0.250 | 0.260 |
| B2_conditioned_experts | Complementary | 0.263 | 0.274 | 0.239 | 0.224 |

### Role-FiLM updates

B1 is an identity bypass with frozen FiLM parameters. B2 parameter change norms are measured from the zero initialization at the Best-Accuracy checkpoint.

| Dataset | B2 scale update norm mean ± SD | B2 shift update norm mean ± SD |
|---|---:|---:|
| Movies | 0.677 ± 0.068 | 0.538 ± 0.050 |
| Grocery | 0.714 ± 0.054 | 0.571 ± 0.025 |
| ele-fashion | 1.355 ± 0.190 | 0.839 ± 0.077 |

## Independent implementation and claim audit

| Review question | Evidence and judgement |
|---|---|
| Is the unimodal representation independent of cross-modal completion? | Code path builds each `U_m` from that modality's projector, physical propagation and same-source KNN before either direction calls completion; the cross candidate is applied once afterward. This is supported by the forward tests and source review. |
| Is Physical Cross duplicated in the unimodal backbone? | No. The reused trunk's old physical cross input is zeroed for the independent stage; `P V_(o→m)` is calculated only in the new completion branch. |
| Does Semantic Attention use a receiver query? | Yes. Receiver Q and opposite-source K enter edge-only receiver-wise sparse softmax; all 18 runs recorded finite nonzero Q/K and sender-Phi gradients. |
| Is No-completion active or constant? | Its per-node weight varied across Best-Accuracy checkpoints (0.000–0.424); it is not a constant option. It is near zero in several cells, while Movies visual uses a material reject fraction, so rejection is available but not equally used across modalities/domains. |
| Are the layered gates suppressing Cross to zero? | The measured cross-residual/unimodal RMS ratio ranged 0.020–0.254. Cross remains secondary to the backbone, but is not numerically zero; this scale does not establish task utility. |
| Are core experts shared across Physical, Semantic and Complementary calls? | The code invokes the same four `self.experts` objects; unit tests confirm one shared bank. The path-level usage table shows different selection distributions, but that is suggestive activity rather than causal specialization evidence. |
| Does Role-FiLM adapt and preserve the B1 identity control? | B1 FiLM is frozen/bypassed; B2 starts at exact identity, receives nonzero finite gradients in all runs and has nonzero scale/shift updates. B2 has no stable Validation advantage over B1 in this matrix. |
| Does auxiliary CE use only permitted labels? | NC computes both auxiliary CEs from `train_idx`; Test labels are `-100` at trainer entry. Unit tests check values, gradients and optimizer/checkpoint updates. All 18 run records show Test evaluation false and no Test metrics. |
| Does the model improve over B0? | Paired results are mixed and dataset-dependent: small gains on Movies/Grocery are accompanied by lower Macro-F1 on ele-fashion. See all per-seed differences above; n=3 is descriptive. |
| Is a simpler condition preferable for continued work? | B2 does not provide a consistent gain over B1 and adds 8,192 trainable FiLM parameters. B1 is the more defensible reference if this line is revisited; this report does not authorize a successor model. |

## Failures and unresolved issues

Preserved records: 3 runner failures and 2 GPU preflight attempts/refusals. The October 14 GPU refusal was due to active foreign GPU processes; no process was stopped. On October 17, the ordinary command sandbox lacked NVIDIA device nodes, while the read-only host probe showed GPU0 idle and the full preflight then passed. Two pre-training runner failures were caused by resetting CUDA memory statistics before lazy context initialization; the fix explicitly initializes CUDA. A third attempt trained and saved checkpoints but failed archival because CG-CGE wrote gradient diagnostics to the wrong telemetry attribute; the mismatch was fixed and Movies/B1/seed42 was rerun in a new attempt. These records remain in `data/failures.json`, `data/preflight_attempts.json` and local run folders.
The failed diagnostics attempt and accepted retry had the same data/split/cache and identical initial model/head hashes, but different Validation trajectories (failed attempt Best-Accuracy 56.5087% / Macro-F1 49.6891%; accepted retry 56.5387% / 49.4394%). The cause is unresolved. This study used dropout and GPU sparse/index operations with deterministic algorithms disabled; it did not isolate which source accounts for the trajectory difference, so it is not attributed solely to CUDA.
Across the 18 accepted runs, losses and gradients were finite and no OOM occurred. Remaining optimization concerns are substantial Train–Validation Accuracy/F1 gaps on Movies and Grocery, three-seed uncertainty, and checkpoint-selection sensitivity. Exact full-trajectory repeatability was not demonstrated.
No Test evaluation was performed.