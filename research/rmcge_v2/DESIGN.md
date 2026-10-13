# RM-CGE V2 — Design Review

**Status:** reviewed before model implementation or V2 training.
**Baseline:** `8dec795e4b05b208b73e57e3d87ebfc215a1a75a` on
`exp/mag_information_utility_v1`.
**New branch:** `exp/rmcge_v2_edge_conditioned_experts`.
**Question:** can receiver-modality-conditioned messages on the fixed physical
graph improve the shared V7A/R0 expert backbone, with fixed cross-modal and
KNN context as the V1 control?

This is a three-condition, validation-only comparison on Movies, Grocery,
and ele-fashion for seeds 42, 43, and 44. It does not add expert banks,
routers beyond the specified node correction, learned graph structure, or
auxiliary objectives.

## 1. V0 is the existing strong physical reference

V0 will instantiate the existing `src.models.psce_mag_v7a.Model` body with
`variant: a0_raw`, through a thin subclass whose V0 forward delegates directly
to the parent implementation. It will not reimplement a similar GCN. This
retains the existing modality projectors, four physical hops, active-node RMS
normalization, normalized four-expert `alpha_raw` profiles, four shared
bottleneck experts, modality-level global Top-2 router, physical Strength,
fusion, and `0.01` physical router balance term.

The parent physical operator removes input self-loops, uses the full-graph
loader's undirected edge list and its existing edge multiplicities, computes
destination degree, and assigns each directed message `j → i` the coefficient
`d_i^-1/2 d_j^-1/2`. Isolated nodes are inactive and every propagated state is
zeroed there before Active-RMS Norm. V0's module/state names, initialization,
forward outputs, classifier initialization, and auxiliary loss will be
compared against a directly constructed V7A `a0_raw` reference.

## 2. V1 and V2 have the same public parameter structure

V1 and V2 instantiate the same parent trunk and the same added modules in the
same order. Added-module initialization runs inside `torch.random.fork_rng`,
so it does not advance the random stream used by the linear NC classifier.
For every Dataset × Seed, V1 and V2 must have identical hashes for all public
trunk parameters and the classifier, plus identical names, shapes, and initial
values for all model parameters. V0's trunk and classifier hashes are also
paired with those common initial states.

The extra modules are two bias-free directional Cross Projections
`256→64→256`, four shared per-expert `a/b/c` coefficients, an eight-dimensional
target-modality embedding and one shared Edge Scorer, and an eight-dimensional
modality embedding plus a 64-hidden-unit Node Router correction. V1 and V2
therefore have equal parameter counts and tensor shapes. V1 leaves the Edge
Scorer inactive and has no nonzero `Delta` path; this is a difference in
effective computation, recorded explicitly. V0 has no V2-specific parameters.

The V1−V0 comparison measures the whole specified context-plus-node-routing
bundle. The V2−V1 comparison isolates enabling the learned physical edge
weights within that bundle, while still changing which added parameters get
task gradients. Neither is interpreted as a perfect capacity-matched causal
intervention.

## 3. Directional physical edge scoring

For each directed physical message `j → i`, the model computes

```text
q_T = W_edge_T H_T       q_V = W_edge_V H_V
r_ij^m = [q_T[i], q_T[j], q_V[i], q_V[j],
          cos(H_T[i],H_T[j]), cos(H_V[i],H_V[j]),
          log1p(deg_i), log1p(deg_j), e_m]
logit_ij^m = SharedEdgeMLP(r_ij^m)
w_ij^m = 2 sigmoid(logit_ij^m)
```

`src=j` and `dst=i` follow the repository's source-to-target convention.
Both directions are scored as separate ordered pairs; a reverse weight is
never copied from the forward edge. The Edge Scorer is one shared MLP for both
receiving modalities, conditioned only by the target modality embedding. Its
last Linear weight and bias start at zero, so both target views start with
`w=1`.

The existing normalized coefficient is multiplied by the score:
`Ptilde_m[j→i] = P[j→i] * w_ij^m`. No degree is recomputed and weights are not
renormalized. V0/V1 use unit physical weights. V2 scoring, weighted sparse
aggregation, and custom SpMM backward operate in bounded edge chunks; no dense
`N×N` adjacency or edge-by-node matrix is formed. The implementation may
evaluate `Delta = SpMM(H, P * (w−1))` and set `M=U1+Delta`; this is algebraically
`Ptilde@H−U1`, has the same edge-score gradient, and keeps `Delta` exactly
zero when `w` is one.

## 4. Physical change, cross-modal context, and KNN context

For each target modality, the four parent physical states remain
`U_l=P^l@H`, `l=1…4`, and each receives the parent Active-RMS Norm. For each
expert `k`,

```text
B_m,k = sum_l alpha_k,l * ActiveRMS(U_m,l)
U1_m   = P @ H_m
M_m    = Ptilde_m @ H_m
Delta_m = M_m − U1_m
```

V0 disables all additions. V1 defines `M=U1`, making `Delta` exactly zero.
V2 starts at unit weights and therefore also starts with zero (or roundoff-level)
`Delta`. The denominator is never derived from `Delta` itself.

The two bias-free directional projections run **before** physical
aggregation:

```text
C_T = Ptilde_T @ Phi_V_to_T(H_V)
C_V = Ptilde_V @ Phi_T_to_V(H_T)
```

The cross projection is `Linear(256,64,bias=False) → GELU →
Linear(64,256,bias=False)`. Thus a physical isolate receives exactly zero
cross-modal context. It cannot consume another modality's KNN neighbors as a
physical message.

The V7B.1 K=8 cache is loaded and fingerprint-validated. Canonical cache edges
with `source_flags[:,0] > 0` build `P_T`; those with
`source_flags[:,1] > 0` build `P_V`. The supports are separately symmetrized,
deduplicated, retained with unit unweighted edge mass, and row normalized
independently. A Both edge is present in both supports. Each semantic path is
one fixed, same-modality KNN mean:
`S_T=P_T@H_T`, `S_V=P_V@H_V`; empty rows stay zero. No KNN search, edge score,
source router, or semantic cross branch is added.

Each modality's detached reference scale is the per-feature RMS of `U1` over
active physical nodes, clamped below by the frozen `scale_floor=1e-3`:

```text
scale_m   = stopgrad(clamp_min(RMS_active(U1_m), 1e-3))
Delta_hat = Delta / scale_m
C_hat     = C / scale_m
S_hat     = S / scale_m
Q_m,k     = B_m,k + a_k*Delta_hat_m + b_k*C_hat_m + c_k*S_hat_m
```

All three coefficient vectors are shared between target modalities and start
at `a=.20`, `b=.25`, `c=.08`. Since the denominator comes from the unchanged
physical reference, `Delta=0` implies `Delta_hat=0`; small changes cannot be
turned into a unit-RMS signal by self-normalization. Floor-hit dimensions,
raw and scaled RMS, coefficient-weighted RMS, expert-profile RMS, and frozen
downstream effects will be reported.

## 5. One parent expert bank and a node-level selection correction

Each `Q_m,k` goes through the corresponding parent shared expert:

```text
E_m,k = Q_m,k + 0.1 * ExpertMLP_k(Q_m,k)
```

These are the same four expert objects for Text and Visual. There is no
second physical/cross/semantic bank. The parent's modality-level global router
produces the base logits and the original physical Strength. For V1/V2, a
zero-output-initialized node correction reads
`[H_m, U1_m, Delta_hat_m, C_hat_m, modality_id]`, maps through a 64-unit MLP,
and adds four logits to the base logits. The final four logits feed the
existing Top-2 selection and selected-weight softmax. No new Strength or
source gate is present. V0 uses the unmodified global route.

V1/V2's balance term is computed from the actual corrected routes and only
active physical nodes. It reuses the parent definition: mean dense probability
times normalized Top-2 selection share, summed over experts and scaled by the
number of experts. Per-modality terms are averaged; the model returns
`0.01 * balance`. The NC task's `aux_weight=1` therefore trains exactly
`Train CE + 0.01 * existing-R0-balance`. V0 follows the parent calculation
directly.

The zero Edge Scorer output layer must have a finite, nonzero first-step
gradient. Its preceding scorer layers may have zero initial gradient because
the final layer is zero; after the first update they must be eligible to learn.
The `a_k` gradient may initially be zero because `Delta=0`; it must be checked
again after nonzero edge changes can form. Tests and run audits distinguish a
zero expected initial context from a broken parameter path. Cross Projections,
`b/c`, Node Router output, experts, base projectors/router, and fusion are
audited for finite gradients and actual checkpoint updates.

## 6. Equivalence and finite-graph review checklist

| Question | Frozen answer and required evidence |
|---|---|
| Why one shared bank? | Preserve R0's exact four expert objects across target modalities; new information changes each expert input, not the number of banks. Check one object/state per expert and gradients from both views. |
| Is V0 R0? | V0 delegates to the existing V7A `a0_raw` forward; compare state keys/initial values, projector and four propagated states, expert profiles, route/Strength, fusion, auxiliary loss, classifier logits on the same inputs. |
| Do V1/V2 differ only as intended? | Same module names/shapes/init, config and classifier. Force V2 weights to one and compare every returned tensor with V1 in eval mode. |
| Are edge endpoints ordered? | Build features explicitly from `dst=i` receiver and `src=j` sender. Score reverse directed messages independently and test that asymmetric synthetic endpoint features can yield unequal reverse scores. |
| Is old normalization retained? | Reuse V7A's normalized `src,dst,norm,active` output exactly; multiply only `norm` by `w`. Compare `w=1` to V1. |
| Does Delta isolate edge conditioning? | Compute `M−U1`; use the same reference scale from active `U1`; test zero at unit weights and measure the normalized and coefficient-weighted changes. |
| Is Cross projection pre-aggregation? | Apply the bias-free modality map to sender states before edge SpMM; test direct values on a hand-built directed graph. |
| Are isolates/empty KNN rows zero? | Sparse aggregation has no incoming terms and outputs zero; verify after every context path and after scaling. |
| Can scale normalization amplify noise? | Clamp reference scales at `1e-3`; never divide by a message's own RMS; record how many feature dimensions hit the floor. |
| Does zero scoring learn? | Zero final scorer layer gives unit weights; test final-layer gradient/update while `Delta` starts zero. Test later-step edge-score and `a` updates separately. |
| Does node routing really select experts? | Add zero-initialized node logits before Top-2, calculate balance on corrected Top-2 assignment, preserve the old Strength, and measure route flips/mass. |
| What capacity mismatch remains? | V1 and V2 match all total shapes; V1's scorer is inactive and V0 lacks added context/router parameters. Report trainable count and effective activity, not just total count. |

## 7. Training, diagnostics, and evidence boundary

The runner will use the existing `unified_full_graph_nc_v1` `run_nc` path with
AdamW, LR `1e-3`, weight decay `1e-4`, dropout `.20`, 300-epoch maximum,
patience 30, minimum epoch 30, min delta `1e-4`, grad clip 1, and evaluation
every epoch. `evaluate_test=false`; checkpoint selection is Validation
Accuracy, with Best-Macro-F1 retained only as a diagnostic. The existing
extended NC scorecard reuses eval embeddings for Train/Validation CE and
accuracy and stores epoch history without an extra stochastic model forward.

The matrix is paired within Dataset × Seed. Before Freeze, engineering checks
will include the full repository regression suite, exact V0/R0 tests, V1/V2
unit-edge equivalence, sparse-message tests, first-step gradients, one actual
AdamW update, checkpoint reload, and a complete ele-fashion forward/backward
update on RTX 3090. Freeze will include design, implementation, tests, configs,
and audits. Seed 42's nine runs must pass integrity gates before seeds 43/44.

Post-training diagnostics will summarize directional edge weights by modality,
reverse-edge asymmetry and target-degree bins; all raw/scaled context and
expert contribution scales; route probability/Top-2 use and logit corrections;
parameter updates; four frozen one-path-off interventions; and runtime,
memory, validation scorecards, and paired seed differences. Frozen
interventions measure sensitivity of the fitted checkpoint, not retrained
ablations or edge-level causality. Three seeds are descriptive, not a
significance test. V2 versus V0 is internally paired against the strong
physical baseline; historical V7B.1 R0 values are background only.

No Test evaluation, Test-label indexing, HPO, additional variants, or new
training objective is part of this study.
