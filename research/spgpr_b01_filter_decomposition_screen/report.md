# SPGPR B0–B1: Shared/Private GPR Filter Decomposition Screen

- Base SHA: `e8470fdd529f4e71100c32f25d75a400095fc6de`
- Campaign code HEAD at start: `c0e9fb459a4d4077115d0dc4e499bddac5af7d83`
- Formal campaign: 45/45 validation-only runs
- Split design: one fixed dataset split per dataset; seeds 42/43/44 are paired model/training seeds, not independent splits.
- NC test evaluation: disabled for all runs (`development_no_test=true`, `evaluate_test=false`).
- Primary metric: validation Accuracy. Macro-F1 and CE are secondary consistency signals.
- Architecture deltas are descriptive paired summaries; no pseudo-IID p-values are used.

## Repeatability audit

These are same-seed execution variation estimates from three independent S processes per dataset, not architecture variance.

| Dataset | Metric | Mean | Population SD | Max–min range |
|---|---:|---:|---:|---:|
| Movies | val_accuracy | 55.34893 pp | 0.01414 pp | 0.02999 pp |
| Movies | val_macro_f1 | 47.28776 pp | 0.77445 pp | 1.86489 pp |
| Movies | val_ce | 1.38535 | 0.00783 | 0.01744 |
| Movies | best_epoch | 77.33333 | 3.29983 | 7.00000 |
| Grocery | val_accuracy | 81.83504 pp | 0.01380 pp | 0.02928 pp |
| Grocery | val_macro_f1 | 75.19174 pp | 0.00651 pp | 0.01380 pp |
| Grocery | val_ce | 0.67668 | 0.00006 | 0.00013 |
| Grocery | best_epoch | 72.00000 | 0.00000 | 0.00000 |

Ele-fashion has no repeatability floor in this six-run audit; its paired deltas are reported without borrowing another dataset's floor.

## Validation performance

| Dataset | Variant | Accuracy mean ± SD | Macro-F1 mean ± SD | CE mean ± SD | Best epoch mean ± SD |
|---|---|---:|---:|---:|---:|
| Movies | U | 55.489 ± 0.234% | 46.999 ± 0.348% | 1.41565 ± 0.02258 | 84.0 ± 14.1 |
| Movies | P | 55.309 ± 0.200% | 47.630 ± 1.011% | 1.44097 ± 0.01295 | 93.7 ± 5.0 |
| Movies | S | 55.259 ± 0.159% | 47.010 ± 0.863% | 1.44225 ± 0.05053 | 92.0 ± 17.8 |
| Movies | I | 55.469 ± 0.228% | 47.260 ± 0.827% | 1.42839 ± 0.02851 | 92.3 ± 12.7 |
| Movies | SP | 55.489 ± 0.256% | 47.217 ± 1.000% | 1.42753 ± 0.03889 | 90.3 ± 19.7 |
| Grocery | U | 81.718 ± 0.060% | 75.365 ± 0.215% | 0.70299 ± 0.01952 | 83.3 ± 14.1 |
| Grocery | P | 81.737 ± 0.091% | 75.290 ± 0.355% | 0.69585 ± 0.01375 | 73.7 ± 5.4 |
| Grocery | S | 81.747 ± 0.100% | 75.339 ± 0.317% | 0.69580 ± 0.01380 | 73.7 ± 5.4 |
| Grocery | I | 81.679 ± 0.055% | 75.382 ± 0.257% | 0.68373 ± 0.01016 | 70.3 ± 7.0 |
| Grocery | SP | 81.786 ± 0.083% | 75.453 ± 0.546% | 0.70571 ± 0.01249 | 82.3 ± 10.9 |
| ele-fashion | U | 87.447 ± 0.070% | 74.002 ± 0.354% | 0.41141 ± 0.00865 | 142.7 ± 9.4 |
| ele-fashion | P | 87.372 ± 0.013% | 74.231 ± 0.667% | 0.40247 ± 0.00410 | 142.7 ± 8.3 |
| ele-fashion | S | 87.317 ± 0.139% | 74.150 ± 1.082% | 0.40840 ± 0.00324 | 136.3 ± 23.5 |
| ele-fashion | I | 87.467 ± 0.039% | 74.802 ± 0.425% | 0.41730 ± 0.00825 | 178.3 ± 9.0 |
| ele-fashion | SP | 87.382 ± 0.063% | 74.022 ± 0.347% | 0.40646 ± 0.00308 | 140.0 ± 10.6 |

## Paired architecture comparisons

Every metric shows paired mean ± population SD, positive/negative/tie seed counts, and its same-seed execution SD/range where measured. Accuracy and Macro-F1 are in percentage points; CE is the raw loss delta. Ele-fashion has no repeatability run, so its execution floor is unmeasured.

- **Movies P-U** — Accuracy: -0.180 ± 0.401 pp; +/−/= 1/2/0; floor 0.014/0.030; Macro-F1: +0.631 ± 0.665 pp; +/−/= 2/1/0; floor 0.774/1.865; CE: +0.02532 ± 0.02431; +/−/= 2/1/0; floor 0.00783/0.01744.
- **Movies S-P** — Accuracy: -0.050 ± 0.357 pp; +/−/= 1/1/1; floor 0.014/0.030; Macro-F1: -0.619 ± 1.663 pp; +/−/= 2/1/0; floor 0.774/1.865; CE: +0.00128 ± 0.04850; +/−/= 1/2/0; floor 0.00783/0.01744.
- **Movies I-S** — Accuracy: +0.210 ± 0.130 pp; +/−/= 3/0/0; floor 0.014/0.030; Macro-F1: +0.249 ± 1.330 pp; +/−/= 1/2/0; floor 0.774/1.865; CE: -0.01387 ± 0.02223; +/−/= 2/1/0; floor 0.00783/0.01744.
- **Movies SP-I** — Accuracy: +0.020 ± 0.057 pp; +/−/= 2/1/0; floor 0.014/0.030; Macro-F1: -0.042 ± 1.187 pp; +/−/= 2/1/0; floor 0.774/1.865; CE: -0.00085 ± 0.01134; +/−/= 2/1/0; floor 0.00783/0.01744.
- **Movies SP-S** — Accuracy: +0.230 ± 0.123 pp; +/−/= 3/0/0; floor 0.014/0.030; Macro-F1: +0.207 ± 1.862 pp; +/−/= 2/1/0; floor 0.774/1.865; CE: -0.01472 ± 0.01512; +/−/= 1/2/0; floor 0.00783/0.01744.
- **Grocery P-U** — Accuracy: +0.020 ± 0.120 pp; +/−/= 1/1/1; floor 0.014/0.029; Macro-F1: -0.076 ± 0.270 pp; +/−/= 2/1/0; floor 0.007/0.014; CE: -0.00713 ± 0.01015; +/−/= 1/2/0; floor 0.00006/0.00013.
- **Grocery S-P** — Accuracy: +0.010 ± 0.014 pp; +/−/= 1/0/2; floor 0.014/0.029; Macro-F1: +0.050 ± 0.038 pp; +/−/= 2/0/1; floor 0.007/0.014; CE: -0.00005 ± 0.00005; +/−/= 1/2/0; floor 0.00006/0.00013.
- **Grocery I-S** — Accuracy: -0.068 ± 0.073 pp; +/−/= 1/2/0; floor 0.014/0.029; Macro-F1: +0.043 ± 0.573 pp; +/−/= 2/1/0; floor 0.007/0.014; CE: -0.01208 ± 0.01217; +/−/= 0/3/0; floor 0.00006/0.00013.
- **Grocery SP-I** — Accuracy: +0.107 ± 0.073 pp; +/−/= 3/0/0; floor 0.014/0.029; Macro-F1: +0.070 ± 0.766 pp; +/−/= 1/2/0; floor 0.007/0.014; CE: +0.02199 ± 0.01762; +/−/= 3/0/0; floor 0.00006/0.00013.
- **Grocery SP-S** — Accuracy: +0.039 ± 0.028 pp; +/−/= 2/0/1; floor 0.014/0.029; Macro-F1: +0.113 ± 0.319 pp; +/−/= 2/1/0; floor 0.007/0.014; CE: +0.00991 ± 0.00970; +/−/= 2/1/0; floor 0.00006/0.00013.
- **ele-fashion P-U** — Accuracy: -0.075 ± 0.078 pp; +/−/= 0/3/0; floor unmeasured; Macro-F1: +0.229 ± 0.957 pp; +/−/= 1/2/0; floor unmeasured; CE: -0.00894 ± 0.00562; +/−/= 0/3/0; floor unmeasured.
- **ele-fashion S-P** — Accuracy: -0.055 ± 0.150 pp; +/−/= 2/1/0; floor unmeasured; Macro-F1: -0.081 ± 1.749 pp; +/−/= 2/1/0; floor unmeasured; CE: +0.00593 ± 0.00164; +/−/= 3/0/0; floor unmeasured.
- **ele-fashion I-S** — Accuracy: +0.150 ± 0.119 pp; +/−/= 3/0/0; floor unmeasured; Macro-F1: +0.652 ± 1.499 pp; +/−/= 2/1/0; floor unmeasured; CE: +0.00890 ± 0.00817; +/−/= 2/1/0; floor unmeasured.
- **ele-fashion SP-I** — Accuracy: -0.085 ± 0.086 pp; +/−/= 1/2/0; floor unmeasured; Macro-F1: -0.779 ± 0.250 pp; +/−/= 0/3/0; floor unmeasured; CE: -0.01083 ± 0.00938; +/−/= 1/2/0; floor unmeasured.
- **ele-fashion SP-S** — Accuracy: +0.065 ± 0.200 pp; +/−/= 1/2/0; floor unmeasured; Macro-F1: -0.128 ± 1.408 pp; +/−/= 1/2/0; floor unmeasured; CE: -0.00194 ± 0.00135; +/−/= 0/3/0; floor unmeasured.

## Learned filter diagnostics

Coefficients below are effective normalized filters, never the raw SP parameters. The detailed run-level values are in `data/filter_diagnostics.csv`.

### Movies

- U: mean gamma Text=[0.3333333432674408, 0.3333333432674408, 0.3333333432674408], Visual=[0.3333333432674408, 0.3333333432674408, 0.3333333432674408]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4957; response RMS Text/Visual=0.50439/0.67312; scaled-response/prior RMS=0.24194/0.34479.
- P: mean gamma Text=[0.3456609745820363, 0.34620513518651325, 0.3081338902314504], Visual=[0.3456609745820363, 0.34620513518651325, 0.3081338902314504]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4951; response RMS Text/Visual=0.51239/0.67828; scaled-response/prior RMS=0.24497/0.34785.
- S: mean gamma Text=[0.3457450369993846, 0.34611453612645465, 0.3081403970718384], Visual=[0.3457450369993846, 0.34611453612645465, 0.3081403970718384]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4951; response RMS Text/Visual=0.50730/0.67620; scaled-response/prior RMS=0.24256/0.34670.
- I: mean gamma Text=[0.339419682820638, 0.3492554525534312, 0.3113248348236084], Visual=[0.34820178151130676, 0.3412514229615529, 0.31054683526357013]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4954; response RMS Text/Visual=0.50772/0.67724; scaled-response/prior RMS=0.24304/0.34726. Mean Text–Visual cosine=0.9998, L1 distance=0.0176, L2 distance=0.0119.
- SP: mean gamma Text=[0.32690898577372235, 0.3496949076652527, 0.32339611649513245], Visual=[0.3657001554965973, 0.34035058816274005, 0.29394922653834027]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4952; response RMS Text/Visual=0.50398/0.67092; scaled-response/prior RMS=0.24129/0.34359. Mean Text–Visual cosine=0.9963, L1 distance=0.0776, L2 distance=0.0498. Effective shared gamma=[0.3463045557339986, 0.3450227479139964, 0.3086726665496826]; effective private delta=[-0.01939558486143748, 0.004672159751256307, 0.014723444978396097]; mean private L1=0.03879, L2=0.02488, private/shared L2 ratio=0.04303.

### Grocery

- U: mean gamma Text=[0.3333333432674408, 0.3333333432674408, 0.3333333432674408], Visual=[0.3333333432674408, 0.3333333432674408, 0.3333333432674408]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.5090; response RMS Text/Visual=0.59461/0.62941; scaled-response/prior RMS=0.30205/0.32093.
- P: mean gamma Text=[0.3179582158724467, 0.3439701199531555, 0.33807166417439777], Visual=[0.3179582158724467, 0.3439701199531555, 0.33807166417439777]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.5086; response RMS Text/Visual=0.58800/0.62022; scaled-response/prior RMS=0.29844/0.31594.
- S: mean gamma Text=[0.3179610272248586, 0.3438320557276408, 0.3382069369157155], Visual=[0.3179610272248586, 0.3438320557276408, 0.3382069369157155]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.5086; response RMS Text/Visual=0.58800/0.62023; scaled-response/prior RMS=0.29844/0.31595.
- I: mean gamma Text=[0.32855790853500366, 0.34003181258837384, 0.3314102788766225], Visual=[0.32057616114616394, 0.3419308662414551, 0.33749298254648846]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.5080; response RMS Text/Visual=0.58401/0.61500; scaled-response/prior RMS=0.29594/0.31303. Mean Text–Visual cosine=0.9998, L1 distance=0.0160, L2 distance=0.0102.
- SP: mean gamma Text=[0.327492892742157, 0.3378499050935109, 0.334657222032547], Visual=[0.3065165678660075, 0.35144445300102234, 0.34203897913297016]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.5093; response RMS Text/Visual=0.59486/0.63012; scaled-response/prior RMS=0.30242/0.32140. Mean Text–Visual cosine=0.9990, L1 distance=0.0420, L2 distance=0.0261. Effective shared gamma=[0.3170047402381897, 0.3446471691131592, 0.3383481005827586]; effective private delta=[0.010488162438074747, -0.006797273953755696, -0.0036908785502115884]; mean private L1=0.02098, L2=0.01303, private/shared L2 ratio=0.02256.

### ele-fashion

- U: mean gamma Text=[0.3333333432674408, 0.3333333432674408, 0.3333333432674408], Visual=[0.3333333432674408, 0.3333333432674408, 0.3333333432674408]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4730; response RMS Text/Visual=0.55735/0.55218; scaled-response/prior RMS=0.26459/0.26055.
- P: mean gamma Text=[0.3126554588476817, 0.36332961916923523, 0.32401490211486816], Visual=[0.3126554588476817, 0.36332961916923523, 0.32401490211486816]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4728; response RMS Text/Visual=0.55559/0.54701; scaled-response/prior RMS=0.26366/0.25798.
- S: mean gamma Text=[0.3137235641479492, 0.36055561900138855, 0.3257207969824473], Visual=[0.3137235641479492, 0.36055561900138855, 0.3257207969824473]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4743; response RMS Text/Visual=0.55299/0.54520; scaled-response/prior RMS=0.26305/0.25801.
- I: mean gamma Text=[0.30836079518000287, 0.36818405985832214, 0.32345513502756756], Visual=[0.3402087291081746, 0.33786021669705707, 0.32193108399709064]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4661; response RMS Text/Visual=0.56350/0.57018; scaled-response/prior RMS=0.26436/0.26466. Mean Text–Visual cosine=0.9971, L1 distance=0.0637, L2 distance=0.0440.
- SP: mean gamma Text=[0.2889402707417806, 0.38951653242111206, 0.32154316703478497], Visual=[0.3368981679280599, 0.33329800764719647, 0.32980382442474365]; negative coefficient count across checkpoints/modalities=0; mean lambda=0.4737; response RMS Text/Visual=0.55257/0.54974; scaled-response/prior RMS=0.26269/0.25978. Mean Text–Visual cosine=0.9918, L1 distance=0.1124, L2 distance=0.0744. Effective shared gamma=[0.3129192292690277, 0.3614072601000468, 0.32567350069681805]; effective private delta=[-0.02397894859313965, 0.028109262386957806, -0.00413032869497935]; mean private L1=0.05622, L2=0.03719, private/shared L2 ratio=0.06429.

## Polynomial response and interventions

The table gives the mean polynomial response over three selected checkpoints at xi ∈ {-1, -0.5, 0, 0.5, 1}. These are descriptive polynomial values; no eigendecomposition was performed, so they are not estimates of graph spectral energy.

| Dataset | Variant | Modality | Response at xi -1, -0.5, 0, 0.5, 1 |
|---|---|---|---|
| Movies | U | text | 0.3591, 0.4593, 0.5193, 0.6595, 1.0000 |
| Movies | U | visual | 0.3191, 0.4255, 0.4893, 0.6382, 1.0000 |
| Movies | P | text | 0.3736, 0.4612, 0.5210, 0.6637, 1.0000 |
| Movies | P | visual | 0.3315, 0.4250, 0.4888, 0.6411, 1.0000 |
| Movies | S | text | 0.3735, 0.4611, 0.5209, 0.6637, 1.0000 |
| Movies | S | visual | 0.3316, 0.4250, 0.4889, 0.6411, 1.0000 |
| Movies | I | text | 0.3758, 0.4622, 0.5204, 0.6623, 1.0000 |
| Movies | I | visual | 0.3265, 0.4236, 0.4888, 0.6413, 1.0000 |
| Movies | SP | text | 0.3760, 0.4643, 0.5202, 0.6600, 1.0000 |
| Movies | SP | visual | 0.3264, 0.4207, 0.4894, 0.6450, 1.0000 |
| Grocery | U | text | 0.3221, 0.4280, 0.4916, 0.6399, 1.0000 |
| Grocery | U | visual | 0.3206, 0.4268, 0.4905, 0.6391, 1.0000 |
| Grocery | P | text | 0.3335, 0.4335, 0.4920, 0.6379, 1.0000 |
| Grocery | P | visual | 0.3319, 0.4321, 0.4908, 0.6371, 1.0000 |
| Grocery | S | text | 0.3333, 0.4334, 0.4920, 0.6379, 1.0000 |
| Grocery | S | visual | 0.3318, 0.4321, 0.4908, 0.6371, 1.0000 |
| Grocery | I | text | 0.3305, 0.4316, 0.4928, 0.6402, 1.0000 |
| Grocery | I | visual | 0.3303, 0.4317, 0.4912, 0.6377, 1.0000 |
| Grocery | SP | text | 0.3262, 0.4296, 0.4912, 0.6388, 1.0000 |
| Grocery | SP | visual | 0.3389, 0.4352, 0.4903, 0.6350, 1.0000 |
| ele-fashion | U | text | 0.3668, 0.4658, 0.5251, 0.6636, 1.0000 |
| ele-fashion | U | visual | 0.3719, 0.4700, 0.5289, 0.6663, 1.0000 |
| ele-fashion | P | text | 0.3956, 0.4750, 0.5253, 0.6619, 1.0000 |
| ele-fashion | P | visual | 0.4004, 0.4792, 0.5291, 0.6646, 1.0000 |
| ele-fashion | S | text | 0.3913, 0.4730, 0.5241, 0.6610, 1.0000 |
| ele-fashion | S | visual | 0.3955, 0.4766, 0.5274, 0.6633, 1.0000 |
| ele-fashion | I | text | 0.4071, 0.4827, 0.5308, 0.6653, 1.0000 |
| ele-fashion | I | visual | 0.3868, 0.4786, 0.5369, 0.6734, 1.0000 |
| ele-fashion | SP | text | 0.4194, 0.4830, 0.5245, 0.6586, 1.0000 |
| ele-fashion | SP | visual | 0.3709, 0.4686, 0.5282, 0.6665, 1.0000 |

For I and SP, `data/intervention_by_run.csv` and `data/intervention_summary.csv` report filter-mean and filter-swap checkpoint reliance. The deltas below are intervention minus normal, averaged over paired selected checkpoints; they do not replace retrained I−S or SP−S comparisons.

- Movies I: Accuracy normal/mean/swap = 55.469/55.469/55.469% (Δ mean/swap +0.000/+0.000 pp); Macro-F1 = 47.260/47.257/47.227% (Δ -0.003/-0.033 pp); CE = 1.42839/1.42839/1.42839 (Δ +0.00000/+0.00001).
- Movies SP: Accuracy normal/mean/swap = 55.489/55.479/55.459% (Δ mean/swap -0.010/-0.030 pp); Macro-F1 = 47.217/47.269/47.135% (Δ +0.052/-0.082 pp); CE = 1.42753/1.42750/1.42747 (Δ -0.00003/-0.00006).
- Grocery I: Accuracy normal/mean/swap = 81.679/81.679/81.689% (Δ mean/swap +0.000/+0.010 pp); Macro-F1 = 75.382/75.382/75.388% (Δ +0.000/+0.005 pp); CE = 0.68373/0.68375/0.68377 (Δ +0.00002/+0.00004).
- Grocery SP: Accuracy normal/mean/swap = 81.786/81.796/81.796% (Δ mean/swap +0.010/+0.010 pp); Macro-F1 = 75.453/75.458/75.458% (Δ +0.005/+0.005 pp); CE = 0.70571/0.70576/0.70580 (Δ +0.00004/+0.00009).
- ele-fashion I: Accuracy normal/mean/swap = 87.467/87.457/87.450% (Δ mean/swap -0.010/-0.017 pp); Macro-F1 = 74.802/74.803/74.819% (Δ +0.002/+0.017 pp); CE = 0.41730/0.41707/0.41685 (Δ -0.00023/-0.00045).
- ele-fashion SP: Accuracy normal/mean/swap = 87.382/87.392/87.385% (Δ mean/swap +0.010/+0.003 pp); Macro-F1 = 74.022/74.057/74.072% (Δ +0.035/+0.050 pp); CE = 0.40646/0.40624/0.40603 (Δ -0.00023/-0.00044).

## Conservative case classification

The summaries below show direction, seed agreement, and relation to observed same-seed variation. With three model seeds on one split per dataset, these are descriptive patterns, not inferential tests.
- Movies: Case C/E exploratory: I−S clears the measured execution range, SP−I is within that range, and effective private deviation remains nonzero; the effect is small and not formal non-inferiority. P−U=-0.180 pp; S−P=-0.050 pp; I−S=+0.210 pp; SP−I=+0.020 pp; SP−S=+0.230 pp.
- Grocery: Case E candidate for Accuracy only: SP−S clears the execution range while I−S does not; secondary CE and checkpoint interventions should temper this result. P−U=+0.020 pp; S−P=+0.010 pp; I−S=-0.068 pp; SP−I=+0.107 pp; SP−S=+0.039 pp.
- ele-fashion: Case C direction on I−S (+0.150 pp, 3/3 seeds), but no repeatability floor was measured; SP−I loses Accuracy/Macro-F1 while CE improves, so the regime remains mixed. P−U=-0.075 pp; S−P=-0.055 pp; I−S=+0.150 pp; SP−I=-0.085 pp; SP−S=+0.065 pp.

## B2 recommendation

**Do not start B2 selective shared alignment from this screen alone.** P−U and S−P show no repeatability-calibrated shared-filter gain. I−S is positive beyond the measured range on Movies, negative on Grocery, and positive but uncalibrated on ele-fashion. SP−I is close to the Movies execution range, positive on Grocery Accuracy but worse CE, and negative on ele-fashion Accuracy/Macro-F1 while CE improves. Effective private deviations are nonzero but modest, and mean/swap interventions move Accuracy by at most 0.03 pp. The mixed, small effects do not establish a consistent shared target that warrants alignment.

## Self-audit

1. Yes: campaign and design audit identify required base `e8470fdd529f4e71100c32f25d75a400095fc6de`.
2. No: no other historical experiment branch was used as source.
3. No NC test evaluation or test metrics were run; test split indices were loaded only by the existing loader for label masking.
4. No split was modified; one fixed split per dataset was used across model seeds.
5. Yes: U/P/S/I/SP share state-dict layouts and parameter counts within each dataset; trainable model parameters were 1,249,044 for Movies/Grocery and 986,900 for ele-fashion.
6. Yes: same-seed same-name initial tensors are bitwise matched; covered by tests.
7. Yes: U/P/S/I/SP start at the same uniform effective gamma; covered by tests.
8. Same-seed execution variation is quantified above and in the repeatability CSVs.
9. P−U is reported per dataset in paired comparison tables.
10. S−P is reported per dataset in paired comparison tables.
11. Total negative effective coefficients among selected checkpoints: 0.
12. I−S is small and dataset-dependent: +0.210 pp Movies, −0.068 pp Grocery, +0.150 pp ele-fashion (no repeatability floor for the last value).
13. Text/Visual filters differ modestly: mean cosine exceeds 0.99 in I and SP; L1/L2 distances are in the diagnostics above.
14. SP−I: Movies +0.020 pp Accuracy (within execution range); Grocery +0.107 pp Accuracy but worse CE; ele-fashion −0.085 pp Accuracy and −0.779 pp Macro-F1 while CE improves. No uniform ranking.
15. Mean SP effective private L2 norm across selected checkpoints: 0.025032; nonzero but modest in each dataset.
16. Mean/swap interventions change Accuracy by at most 0.03 pp, showing little checkpoint reliance at this resolution.
17. Intervention effects are similarly small to the retrained contrasts and do not show strong filter-to-modality dependence.
18. Yes: Movies, Grocery, and ele-fashion have mixed dataset-specific regimes.
19. Yes: Accuracy, Macro-F1, and CE sometimes move in different directions; paired values are shown together.
20. No: evidence is not sufficient to enter B2 selective alignment because modality-specific gains are small/inconsistent and checkpoint interventions show little reliance.
21. LP is execution smoke only; it records LinkNeighborLoader, [5,5,5], positive edge removal, backward, validation inference, and checkpoint save.
22. Polynomial values are descriptive sampled responses without eigendecomposition; no broader spectral or causal claim is made.

## Execution record

- Repeatability status: passed
- Full tests and NC/LP smoke status: passed
- Formal runs complete: 45/45
- Local fetch note: see design audit/run provenance; pushing is handled after analysis and review package completion.

B0–B1 stops here. No B2 alignment or other excluded mechanism is implemented or run.
