# N0 — Recipient-State-Conditioned Functional Utility Audit

**Final label: `TARGET_LEVEL_STATE_MODULATION`**

## Scope and interpretation

N0 reuses the frozen P0/P1.3 H0 representations and the 27 L0.1 linear shared-slot heads. It measures counterfactual validation CE gains over nine recipient neighborhood backgrounds and deterministic same-modality partial-25% states. No model or utility predictor was trained.

For every fixed edge, operator, modality, and head, the replacement logit delta is background-invariant because the head is linear and the physical degree is fixed. Any change in conditional gain is recipient-state-conditioned task-loss geometry. It is not representation-level or message-message semantic interaction.

All dataset summaries report the mean and population SD over the three seeds. They are descriptive; no significance test is used.

## Dataset-specific primary summary

| Dataset | Full-grid context SD | Full-grid context range | Context/head ratio | Ordinary sign switch | Robust sign switch | SS-vs-background target rank ρ | Partial-25 context range | Same-modal mean | Cross-modal mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.00227 ± 0.00010 | 0.00666 ± 0.00030 | 0.435 ± 0.008 | 0.087 ± 0.001 | 0.006 ± 0.001 | 0.979 ± 0.000 | 0.00091 ± 0.00005 | 0.00208 ± 0.00011 | 0.00214 ± 0.00012 |
| Grocery | 0.00440 ± 0.00025 | 0.01286 ± 0.00074 | 1.357 ± 0.073 | 0.086 ± 0.003 | 0.008 ± 0.001 | 0.969 ± 0.000 | 0.00108 ± 0.00007 | 0.00283 ± 0.00016 | 0.00480 ± 0.00026 |
| ele-fashion | 0.00099 ± 0.00002 | 0.00270 ± 0.00004 | 0.568 ± 0.018 | 0.042 ± 0.001 | 0.001 ± 0.000 | 0.987 ± 0.000 | 0.00014 ± 0.00000 | 0.00046 ± 0.00000 | 0.00119 ± 0.00003 |

The table reports run-level means over four modality/operator targets and the nine backgrounds; the rank column averages valid target-level correlations over non-SS backgrounds. Each displayed ± is population SD across seeds.

## Rank ceiling and partial-state comparison

| Dataset | SS vs full-grid rank ρ | Centered full-grid ρ | SS head-repeat ceiling ρ | Partial25 centered ρ |
|---|---:|---:|---:|---:|
| Movies | 0.979 ± 0.000 | 0.988 ± 0.000 | 0.706 ± 0.011 | 0.999 ± 0.000 |
| Grocery | 0.969 ± 0.000 | 0.967 ± 0.001 | 0.768 ± 0.008 | 0.994 ± 0.000 |
| ele-fashion | 0.987 ± 0.000 | 0.992 ± 0.000 | 0.693 ± 0.008 | 0.998 ± 0.000 |

## Answers to the 20 research questions

Q1_SS identity. Yes. All 27 head repeats passed edge-aligned L0.1 raw-gain regression at rtol=1e-7, atol=2e-7; per-run maximum absolute errors are recorded in the manifest.
Q2 logit-delta invariance. Yes. Maximum background-difference errors by run are at most 0.
Q3 conditional CE gain. Movies: full-grid mean gain range=0.00666 ± 0.00030; recipient-state-conditioned task utility, not message interaction; Grocery: full-grid mean gain range=0.01286 ± 0.00074; recipient-state-conditioned task utility, not message interaction; ele-fashion: full-grid mean gain range=0.00270 ± 0.00004; recipient-state-conditioned task utility, not message interaction
Q4 context vs head uncertainty. Movies: context/head ratio=0.435 ± 0.008; Grocery: context/head ratio=1.357 ± 0.073; ele-fashion: context/head ratio=0.568 ± 0.018. Context variation exceeds head-repeat variation on average in Grocery; ratios are descriptive and have no cutoff.
Q5 ordinary sign switching. Movies: ordinary full-grid switch=0.087 ± 0.001; Grocery: ordinary full-grid switch=0.086 ± 0.003; ele-fashion: ordinary full-grid switch=0.042 ± 0.001. See sign_switch_summary.csv for each target and seed.
Q6 robust sign switching. Movies: all-head robust full-grid switch=0.006 ± 0.001; Grocery: all-head robust full-grid switch=0.008 ± 0.001; ele-fashion: all-head robust full-grid switch=0.001 ± 0.000. Robust switching is much rarer than ordinary switching.
Q7 D-vs-P preference switching. Movies: full grid ordinary/robust=0.091 ± 0.006/0.005 ± 0.001; partial25+SS=0.018 ± 0.002/0.000 ± 0.000; Grocery: full grid ordinary/robust=0.070 ± 0.003/0.004 ± 0.001; partial25+SS=0.014 ± 0.000/0.000 ± 0.000; ele-fashion: full grid ordinary/robust=0.036 ± 0.002/0.001 ± 0.000; partial25+SS=0.007 ± 0.000/0.000 ± 0.000
Q8 within-target ranking. Movies: SS-vs-background mean ρ=0.979 ± 0.000; SS head-repeat ceiling ρ=0.706 ± 0.011; Grocery: SS-vs-background mean ρ=0.969 ± 0.000; SS head-repeat ceiling ρ=0.768 ± 0.008; ele-fashion: SS-vs-background mean ρ=0.987 ± 0.000; SS head-repeat ceiling ρ=0.693 ± 0.008. Targets are validation nodes with original indegree ≥5; only nonconstant rank pairs count.
Q9 centered ranking. Movies: centered full-grid ρ=0.988 ± 0.000; partial25 centered ρ=0.999 ± 0.000; Grocery: centered full-grid ρ=0.967 ± 0.001; partial25 centered ρ=0.994 ± 0.000; ele-fashion: centered full-grid ρ=0.992 ± 0.000; partial25 centered ρ=0.998 ± 0.000. Centered sign agreement is also reported in the table.
Q10 same-modality effect. Movies: mean |ΔG|=0.00208 ± 0.00011, robust switch=0.001 ± 0.000, rank=0.985 ± 0.001; Grocery: mean |ΔG|=0.00283 ± 0.00016, robust switch=0.001 ± 0.000, rank=0.977 ± 0.000; ele-fashion: mean |ΔG|=0.00046 ± 0.00000, robust switch=0.000 ± 0.000, rank=0.990 ± 0.000
Q11 cross-modal effect. Movies: mean |ΔG|=0.00214 ± 0.00012, robust switch=0.000 ± 0.000, rank=0.984 ± 0.000; Grocery: mean |ΔG|=0.00480 ± 0.00026, robust switch=0.001 ± 0.000, rank=0.977 ± 0.000; ele-fashion: mean |ΔG|=0.00119 ± 0.00003, robust switch=0.000 ± 0.000, rank=0.990 ± 0.000
Q12 dataset differences. The dataset-specific three-seed summaries below are primary; no pooled-only conclusion is used.
Q13 partial neighborhood. Movies: mean |ΔG|=0.00039 ± 0.00002, ordinary=0.016 ± 0.000, robust=0.000 ± 0.000, context/head=0.097 ± 0.002; Grocery: mean |ΔG|=0.00045 ± 0.00003, ordinary=0.017 ± 0.000, robust=0.000 ± 0.000, context/head=0.285 ± 0.014; ele-fashion: mean |ΔG|=0.00006 ± 0.00000, ordinary=0.009 ± 0.001, robust=0.000 ± 0.000, context/head=0.132 ± 0.004. The partial state changes a floor-rounded 25% subset with three hash-determined subsets.
Q14 extreme-only sensitivity. Movies: full/partial mean gain-range ratio=7.4 ± 0.1×; Grocery: full/partial mean gain-range ratio=11.9 ± 0.2×; ele-fashion: full/partial mean gain-range ratio=19.3 ± 0.2×. Partial effects remain measurable, but are much smaller than full-grid changes; partial robust sign switches are near zero. Evidence is strongest under broad backgrounds, not exclusively absent under partial perturbation.
Q15 CE geometry. Movies: second-order Spearman=1.000 ± 0.000, Pearson=1.000 ± 0.000, MAE=0.00017 ± 0.00001, R²=0.999 ± 0.000; Grocery: second-order Spearman=0.999 ± 0.000, Pearson=0.998 ± 0.000, MAE=0.00058 ± 0.00004, R²=0.996 ± 0.000; ele-fashion: second-order Spearman=1.000 ± 0.000, Pearson=1.000 ± 0.000, MAE=0.00010 ± 0.00000, R²=1.000 ± 0.000. Relative-error median and q90 are in gradient_decomposition.csv; second order closely tracks actual gains.
Q16 pairwise nonadditivity. Movies: max logit-additivity error=8.88e-16, corr(I_CE,I2)=0.998 ± 0.000, MAE=4.26e-06 ± 9.7e-07, median |I_CE|=9.42e-06 ± 1.0e-06; Grocery: max logit-additivity error=8.88e-16, corr(I_CE,I2)=0.996 ± 0.001, MAE=7.87e-06 ± 1.8e-06, median |I_CE|=2.00e-06 ± 5.7e-07; ele-fashion: max logit-additivity error=8.88e-16, corr(I_CE,I2)=0.999 ± 0.000, MAE=3.10e-07 ± 1.1e-07, median |I_CE|=2.21e-07 ± 1.2e-08
Q17 overall evidence. TARGET_LEVEL_STATE_MODULATION; all datasets show background-dependent gain magnitudes but highly stable within-target and centered edge ordering, including partial25. Grocery has the largest contextuality relative to head-repeat variation.
Q18 abandon context-free edge function label. PARTIALLY. A fixed edge utility magnitude is not context-free, but the within-target ordering signal is stable across states. N0 does not establish a ground-truth latent function label; L0.1 local-evidence predictability remains weak on Movies/Grocery and modest on ele-fashion.
Q19 recipient-conditioned multiset model. NO for a recipient-conditioned multiset interaction model: rank reordering and robust preference/sign switching are too limited to support that design. A lower-complexity target-conditioned function-strength hypothesis is worth examining. No architecture is implemented here.
Q20 next phase. Design a narrow follow-up for target-conditioned function strength. Compare state-conditioned gain magnitude against a context-free edge score while preserving the observed stable within-target ordering; define validation-only selection and an untouched final evaluation split before any model implementation.

## Pairwise CE-curvature sanity

The pairwise logit-delta audit confirms linear additivity. The CE interaction term `I_CE = G_jk − G_j − G_k` is compared with the local second-order cross term `I₂ = −δ_jᵀHδ_k`; these quantities describe loss curvature only and are not semantic interaction evidence.

## Self-audit

- **A CE nonadditivity called semantic interaction:** No; pairwise term is explicitly labeled CE loss curvature.
- **B logit delta additivity checked:** Yes; formal max errors recorded and constrained to floating-point tolerance.
- **C task-loss contextuality overstated as representation interaction:** No; gains vary through recipient baseline state; edge logit deltas are invariant.
- **D partial25 considered:** Yes; D25/P25 across three deterministic subsets are included.
- **E near-zero jitter treated as robust:** No; robust requires all three heads strictly positive in one state and strictly negative in another.
- **F head uncertainty confused with context:** No; per-edge context/head ratio is reported descriptively.
- **G ranks omitted:** No; within-target and centered ranks plus SS head-repeat ceiling are reported.
- **H same/cross modality distinguished:** Yes; separate intervention tables are reported.
- **I multiset correctness presumed:** No; N0 only motivates a hypothesis and makes no architecture claim.
- **J test access:** No test targets or labels/metrics were used; only validation recipient labels enter CE.

## Artifacts

Machine-readable tables are in `data/`; the full edge-level gain grids and partial states are gitignored under `outputs/n0_recipient_state_function_context/raw/`. See `run_manifest.json` for source provenance, checkpoints, per-run regression audits, runtime, and GPU memory.

## Limitations

All gains are task-specific counterfactual quantities on validation targets and depend on the frozen L0.1 head. The full S/D/P backgrounds are strong interventions; partial-25% results are the milder audit. Common-edge cross-seed matching is not used as a primary analysis because overlap is limited. These results do not establish ground-truth edge roles, statistical significance, or the correctness of a recipient-conditioned multiset model.
