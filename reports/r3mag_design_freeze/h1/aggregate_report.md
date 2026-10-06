# R³-MAG H1 design-freeze preflight aggregate report

**TEST SPLIT UNTOUCHED.** No test labels were read and no test metric was computed.

## Protocol deviations and interpretation

- Test class histograms are withheld because computing one would require reading test labels. Counts and index hashes are reported.
- Numerical QA deviation: non-NOOP float32 logits/CE vectorization and repeat checks use a fixed 1e-5 tolerance after formal Movies CUDA comparisons showed batch-shape differences up to 4.8e-6; NOOP identity remains held to 1e-6. Actual maximum errors are recorded per run.
- HostTrain/ResponseTrain/Audit partitions use fixed seed 20261006, stratified within the fixed original train split. Original-train labels are read for this required stratification and to derive partition histograms. ResponseTrain labels do not enter host fitting or response utility. The Audit label slice used as CE targets is first indexed after best-validation checkpoint restoration and freeze.
- Bootstrap intervals resample Audit targets only. They do not quantify causal or independent-graph uncertainty.
- Frozen-host interventions are local diagnostics, not causal effects; argmax actions are not latent roles.

Evidence tiers use primary datasets Movies and Grocery. For tiering only, the predeclared practical CE-excess threshold is 0.01 nats/node. STRONG requires every primary dataset × host seed × epsilon headroom excess estimate to meet that threshold with within-run 95% node-bootstrap intervals above zero. MIXED / MARGINAL means at least 70% of those estimates are positive; statistically detectable estimates below the practical threshold remain practically marginal. Otherwise WEAK / FAIL. This rule is applied unchanged to H1-Node (both modalities) and H1-Modality; raw values remain primary.

**H1-Node evidence: WEAK / FAIL**
**H1-Modality evidence: WEAK / FAIL**

## H1 result reading

Structural node-oracle utility is positive in absolute terms, but text and visual node-specific headroom excess are below the matched shuffled controls for every primary seed × epsilon, with all within-run 95% intervals below zero. **NO evidence for structure-specific node adaptation.**
Separate Text/Visual actions have positive absolute headroom, but the matched shuffled controls produce greater separate-vs-shared headroom for every primary seed × epsilon, with all within-run 95% intervals below zero. **NO evidence for modality-specific structural response.**

These are frozen-host local response diagnostics, not causal effects. ele-fashion shows the same negative structural excess direction here and remains a stress dataset.

## Mean structural vs shuffled utilities

Cells show structural / shuffled / structural-minus-shuffled means averaged over all three host seeds and both epsilons (CE utility, nats/node). Paired per-run bootstrap intervals remain in the table above and CSV.

| Dataset | Text node oracle | Text node headroom | Visual node oracle | Visual node headroom | Modality separate-vs-shared headroom |
|---|---:|---:|---:|---:|---:|
| Movies | 0.0965 / 0.1617 / -0.0652 | 0.0881 / 0.1361 / -0.0480 | 0.1673 / 0.2630 / -0.0957 | 0.1515 / 0.2397 / -0.0882 | 0.0465 / 0.0678 / -0.0213 |
| Grocery | 0.0672 / 0.1038 / -0.0365 | 0.0651 / 0.0928 / -0.0276 | 0.0874 / 0.1332 / -0.0459 | 0.0872 / 0.1247 / -0.0375 | 0.0180 / 0.0289 / -0.0109 |
| ele-fashion | 0.0484 / 0.0880 / -0.0396 | 0.0458 / 0.0724 / -0.0266 | 0.0363 / 0.0598 / -0.0235 | 0.0307 / 0.0488 / -0.0180 | 0.0093 / 0.0170 / -0.0077 |

## Host and split protocol

Host: dual projector branches (Linear→LayerNorm→ReLU→Dropout), shared symmetric normalized A+I physical graph, K=3 response basis, global trainable signed gamma initialized by gamma_k=0.2×0.8^k, modality LayerNorm, one-layer fusion MLP and classifier. Hyperparameters are fixed across datasets: hidden=128, dropout=0.2, AdamW lr=1e-3, weight decay=5e-4, max epochs=1000, patience=100. Checkpoint selection follows the NC runner's min_epoch=30 and min_delta=1e-4, using original-val accuracy only.

Response action set has nine actions per modality (NOOP, ± for each k=0…3), with epsilon 0.1 and 0.2 and L2 NormMatch. Each shuffled-control repeat borrows real directions within degree-quantile × frozen-predicted-class buckets, falling back to degree and then global buckets. Each repeat executes all joint action pairs through frozen fusion and classifier.

## Split counts and index hashes

| Dataset | Split source | Data seed | HostTrain | ResponseTrain | Audit | Val | Test | Partition seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Grocery | `/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt` | 42 | 8198 | 1024 | 1022 | 3415 | 3415 | 20261006 |
| Movies | `/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt` | 42 | 8002 | 1003 | 998 | 3334 | 3335 | 20261006 |
| ele-fashion | `/hdd1/DataInHere/YHF/data/ele-fashion/split.pt` | official split | 46927 | 5868 | 5864 | 9777 | 29330 | 20261006 |

Index SHA256 values and class histograms are in each per-run summary JSON. Test histogram is withheld by design.

## Primary H1 results

Means are over three host seeds; each row preserves epsilon. Utility is CE reduction in nats/node. Positive excess means structure-specific gain over the matched shuffled control. Parenthesized endpoints show the envelope of per-seed 95% target-bootstrap intervals (minimum lower to maximum upper), not a pooled confidence interval; per-seed intervals are in the CSV and JSON.

| Dataset | ε | Text oracle excess (run-CI envelope) | Text headroom excess (run-CI envelope) | Visual oracle excess (run-CI envelope) | Visual headroom excess (run-CI envelope) | Modality headroom structural | shuffled | excess (run-CI envelope) | seed signs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.1 | -0.04560 (-0.05616, -0.03742) | -0.03285 (-0.03895, -0.02513) | -0.07026 (-0.08767, -0.05681) | -0.05988 (-0.07252, -0.04690) | 0.03177 | 0.04871 | -0.01694 (-0.02339, -0.01160) | 0/3 |
| Movies | 0.2 | -0.08482 (-0.10543, -0.06973) | -0.06306 (-0.07508, -0.04893) | -0.12109 (-0.15355, -0.09657) | -0.11647 (-0.14278, -0.09066) | 0.06120 | 0.08693 | -0.02573 (-0.03768, -0.01627) | 0/3 |
| Grocery | 0.1 | -0.02639 (-0.03378, -0.01885) | -0.01757 (-0.02454, -0.01182) | -0.03389 (-0.04048, -0.02768) | -0.02338 (-0.03574, -0.01225) | 0.01223 | 0.02162 | -0.00939 (-0.01407, -0.00529) | 0/3 |
| Grocery | 0.2 | -0.04671 (-0.05985, -0.03325) | -0.03771 (-0.04918, -0.02583) | -0.05783 (-0.07104, -0.04570) | -0.05161 (-0.06734, -0.03042) | 0.02374 | 0.03621 | -0.01247 (-0.02010, -0.00533) | 0/3 |
| ele-fashion | 0.1 | -0.02807 (-0.03227, -0.02324) | -0.01719 (-0.02047, -0.01371) | -0.01644 (-0.01889, -0.01348) | -0.01257 (-0.01434, -0.01036) | 0.00648 | 0.01274 | -0.00626 (-0.00789, -0.00490) | 0/3 |
| ele-fashion | 0.2 | -0.05123 (-0.05892, -0.04226) | -0.03591 (-0.04126, -0.02898) | -0.03060 (-0.03507, -0.02492) | -0.02350 (-0.02711, -0.01960) | 0.01217 | 0.02127 | -0.00910 (-0.01196, -0.00653) | 0/3 |

## Across-seed and epsilon stability

- Movies ε=0.1 text_headroom_excess: 0/3 host-seed estimates are positive.
- Movies ε=0.2 text_headroom_excess: 0/3 host-seed estimates are positive.
- Movies text_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- Movies ε=0.1 visual_headroom_excess: 0/3 host-seed estimates are positive.
- Movies ε=0.2 visual_headroom_excess: 0/3 host-seed estimates are positive.
- Movies visual_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- Movies ε=0.1 modality_headroom_excess: 0/3 host-seed estimates are positive.
- Movies ε=0.2 modality_headroom_excess: 0/3 host-seed estimates are positive.
- Movies modality_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- Grocery ε=0.1 text_headroom_excess: 0/3 host-seed estimates are positive.
- Grocery ε=0.2 text_headroom_excess: 0/3 host-seed estimates are positive.
- Grocery text_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- Grocery ε=0.1 visual_headroom_excess: 0/3 host-seed estimates are positive.
- Grocery ε=0.2 visual_headroom_excess: 0/3 host-seed estimates are positive.
- Grocery visual_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- Grocery ε=0.1 modality_headroom_excess: 0/3 host-seed estimates are positive.
- Grocery ε=0.2 modality_headroom_excess: 0/3 host-seed estimates are positive.
- Grocery modality_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- ele-fashion ε=0.1 text_headroom_excess: 0/3 host-seed estimates are positive.
- ele-fashion ε=0.2 text_headroom_excess: 0/3 host-seed estimates are positive.
- ele-fashion text_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- ele-fashion ε=0.1 visual_headroom_excess: 0/3 host-seed estimates are positive.
- ele-fashion ε=0.2 visual_headroom_excess: 0/3 host-seed estimates are positive.
- ele-fashion visual_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.
- ele-fashion ε=0.1 modality_headroom_excess: 0/3 host-seed estimates are positive.
- ele-fashion ε=0.2 modality_headroom_excess: 0/3 host-seed estimates are positive.
- ele-fashion modality_headroom_excess: 3/3 host seeds agree in sign across ε=0.1 and 0.2; 0/6 seed×epsilon estimates are positive.

## Per-run records

Each dataset/seed JSON contains training metrics, learned signed gamma, basis diagnostics, paired bootstrap intervals, all 20 shuffled repeat summaries, fallback counts, and correctness QA. Each per-node CSV is a deterministic sample (up to 256 Audit nodes per epsilon) with all structural single-modality action utilities and shuffled/joint summaries.
