# MR-CGE V1.3 Two-Hop Cross Controlled Screening

Recorded: 2026-10-26T13:25:07+00:00
Source commit: `cf0859663241ce9e66a6a380152f0b5c2d759bed`
Base: `988419283ef8f49c392de954aa739afc95c4c8cc`

## Protocol

Validation-only full-graph NC with fixed seed-42 splits, V7B.1 cache, Test labels masked before Trainer, and Test/LP disabled. Runs used physical GPU 1 under the user-authorized shared-GPU mode: PID ownership, utilization and temperature checks were bypassed, while CUDA device pinning and runtime memory checks remained active. No unrelated process was stopped or modified.

The H2 results below are historical Validation context only and were not loaded or re-evaluated.

| Movies seed | H2 Accuracy | H2 Macro-F1 | H2 CE |
|---:|---:|---:|---:|
| 42 | 0.568686 | 0.502572 | 1.403075 |
| 43 | 0.570486 | 0.494333 | 1.344520 |
| 44 | 0.569286 | 0.488450 | 1.336219 |

## Execution outcome

The final GPU preflight campaign passed 6/6 cells. Stage 1 completed all six Movies runs with verified checkpoint reloads. Earlier blocked preflight attempts are retained in the data records. Both registered arms failed the promotion gates, so Stage 2 was not authorized. Test evaluation and link prediction remained disabled.

## Run results

| Dataset | Arm | Seed | Status | Val Acc | Val Macro-F1 | Val CE | Train–Val Acc gap | Best epoch |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Movies | raw | 42 | complete | 0.559088 | 0.494100 | 1.398714 | 0.187188 | 75 |
| Movies | raw | 43 | complete | 0.552490 | 0.489280 | 1.348170 | 0.133505 | 57 |
| Movies | raw | 44 | complete | 0.554589 | 0.484669 | 1.367447 | 0.165295 | 67 |
| Movies | complementary | 42 | complete | 0.559988 | 0.472976 | 1.404665 | 0.203983 | 84 |
| Movies | complementary | 43 | complete | 0.550990 | 0.480209 | 1.357526 | 0.154199 | 63 |
| Movies | complementary | 44 | complete | 0.554889 | 0.484564 | 1.383626 | 0.197985 | 78 |

## Best-Accuracy checkpoint mechanism diagnostics

| Dataset | Arm | Seed | lambda (text, visual) | Calibrated candidate RMS (text, visual) | Scale (text, visual) | Expert addition RMS (text) | Expert addition RMS (visual) |
|---|---|---:|---|---|---|---|---|
| Movies | raw | 42 | [0.04076547920703888, 0.006029680371284485] | [0.7860543131828308, 0.7043498754501343] | [1.047560453414917, 1.0438294410705566] | [0.001855898299254477, 0.004342096857726574, 0.001220682286657393, 0.001378325978294015] | [0.0002378354693064466, 0.000556444691028446, 0.00015643183724023402, 0.00017663405742496252] |
| Movies | raw | 43 | [0.04447418078780174, -0.014163925312459469] | [0.8114066123962402, 0.7194098830223083] | [1.0277931690216064, 1.052444338798523] | [0.0022200453095138073, 0.004789009224623442, 0.0014113520737737417, 0.001430204720236361] | [0.0005994811072014272, 0.0012931813253089786, 0.0003811088972724974, 0.00038619968108832836] |
| Movies | raw | 44 | [0.052944064140319824, -0.0044458224438130856] | [0.7894988656044006, 0.7145522832870483] | [1.0391734838485718, 1.0569090843200684] | [0.0024876282550394535, 0.00564626045525074, 0.001580434269271791, 0.0017502401024103165] | [0.00018098141299560666, 0.00041078004869632423, 0.00011498069943627343, 0.0001273345114896074] |
| Movies | complementary | 42 | [-0.024535853415727615, -0.011635588482022285] | [0.7734474539756775, 0.7044773697853088] | [2.4585819244384766, 2.2410717010498047] | [0.0011131643550470471, 0.0025880387984216213, 0.0007273100782185793, 0.0008342111250385642] | [0.00046376269892789423, 0.0010782197350636125, 0.00030300940852612257, 0.00034754618536680937] |
| Movies | complementary | 43 | [0.048369742929935455, -0.005373748484998941] | [0.8042952418327332, 0.7149339914321899] | [2.840270757675171, 2.0480692386627197] | [0.00240258383564651, 0.005193020682781935, 0.0015321546234190464, 0.0015428548213094473] | [0.00022672380146104842, 0.0004900480853393674, 0.00014458433724939823, 0.00014559408009517938] |
| Movies | complementary | 44 | [0.027406038716435432, -0.02184758149087429] | [0.78035569190979, 0.7093476057052612] | [2.4265029430389404, 2.0545241832733154] | [0.0012878766283392906, 0.0029167355969548225, 0.0008119738195091486, 0.0009113572305068374] | [0.0008930285694077611, 0.0020224980544298887, 0.0005630319938063622, 0.0006319457315839827] |

## Stage 1 paired Validation screen

| Arm | Mean ΔAcc (pp) | SD | Positive seeds | Mean ΔF1 (pp) | Mean ΔCE | Gate |
|---|---:|---:|---:|---:|---:|---|
| raw | -0.1400 | 0.1805 | 1/3 | 0.4671 | 0.00331 | FAIL |
| complementary | -0.1500 | 0.1606 | 1/3 | -0.5429 | 0.01381 | FAIL |

Selected arm: **none** (no_arm_passed).

## Interpretation

This is an exploratory Validation screen, not a significance test or evidence of Test generalization. Stage 2 runs only the single Stage 1 selected arm on Grocery and ele-fashion seeds 42/43. No selection means no Stage 2 runs.
