# PIGPR-C0 design audit

## Provenance and frozen protocol

- Required base: `2fca8b31dc85edb8c28f9aa4066318b1e8557ad3` on `exp/spgpr_b01_filter_decomposition_screen`.
- The local branch and fetched `origin/exp/spgpr_b01_filter_decomposition_screen` both resolved to that SHA, with a clean worktree. This branch was created directly from the required SHA. No other experiment branch was merged or cherry-picked.
- NC uses the existing fixed dataset splits, `development_no_test=true`, and `evaluate_test=false`; no NC test metrics are requested or read. The campaign uses training seeds 42, 43, and 44 on the fixed split. No formal LP or mid-campaign tuning is in scope.
- The frozen comparison changes only the PIGPR variant's information flow. Projectors, fusion, optimizer, training protocol, and data splits are held constant.

## A. CoSI anchored states

`src/models/cosi_mag_final.py` constructs each modality's states from its projected intrinsic representation `H0` using the same normalized physical propagation operator for that modality:

```text
S0 = H0
Sk = (1 - alpha) A_m S(k-1) + alpha H0
```

The configured default is `alpha = 0.1`, `max_order = 3`. Each recurrence injects the original `H0` again, rather than merely retaining the injection from the previous order.

## B. CoSI PPR prior and basis conversion

The configured restart is `r = 0.15`, prior order is 2, and maximum order is 3. CoSI's `_make_global_prior()` sets the restart mass for orders below the target order, all remaining mass at the target order, and zero above it. Its monomial coefficients are therefore:

```text
c = [r, r(1-r), (1-r)^2, 0]
  = [0.15, 0.1275, 0.7225, 0]
```

To verify the anchored-state coefficients, construct the coefficient matrix `M` whose row `j` contains the monomial coefficients of anchored state `Sj`. For `alpha = 0.1`, `M[j,j] = (1-alpha)^j` and `M[j,q] = alpha(1-alpha)^q` for `q < j`; `M[0,0] = 1`. Composition `sum_j gamma[j] Sj` has monomial coefficients `M.T @ gamma`, so CoSI solves `M.T @ gamma = c`. Recomputing this with the CoSI helper's triangular solve gives:

```text
gamma = [0.0555555556, 0.0524691358, 0.8919753086, 0]
M.T @ gamma = [0.15, 0.1275, 0.7225, 0]
```

The values above were calculated from the stated restart/order/alpha formula, not inserted as constants. PIGPR implements its own prior and basis-conversion helpers and tests both these values and the polynomial equivalence on a synthetic graph.

## C. CoSI composition and the protected-path distinction

CoSI combines a global coefficient vector with modality and node adjustments:

```text
eta_i^m = gamma_global + delta_gamma_m + node_delta_i^m
z_m = sum_k eta_i,k^m S_i,k^m
```

Because `S0 = H0` contributes to this sum, CoSI retains an intrinsic semantic contribution. Its final composition does not guarantee a separate, explicit `H0` skip path: the effective order-0 coefficient can be changed by the learned adjustments. PIGPR tests that direct composition (`AGD`) separately from explicit protected injection (`AGP`).

## D. Existing SPGPR-v0 protected injection

`src/models/spgpr_mag_v0.py` projects text and visual features independently, forms raw states by repeatedly applying the same symmetric normalized physical graph operator, and computes:

```text
C_m = sum_(k=1..3) gamma_m,k (H_m,k - P_m)
Z_m = LayerNorm(P_m + lambda_m C_m)
```

For `filter_mode=uniform`, `gamma_k = 1/3` and `lambda_m = sigmoid(theta_lambda_m)`. The fusion network then applies the shared concatenation, residual MLP, and LayerNorm. PIGPR's `RU` must match this old uniform SPGPR computation within `1e-6` after copying the projector, lambda, and fusion weights.

## E. What PIGPR carries forward

PIGPR absorbs only:

- CoSI's anchored state recurrence and explicit order-0 state;
- its PPR-informed global prior and unconstrained shared global GPR coefficient learning;
- SPGPR's independent modality projectors, physical graph operator, protected structural residual, scalar modality lambdas, and fusion path.

PIGPR excludes learned relation weights, semantic edge calibration, local relation context, cross-order attention, node-specific hop preference, modality-private filters, shared/private decomposition, OT, and MoE. No depth, split, optimizer, or training-hyperparameter search is allowed in this stage.

## Frozen variant semantics

All six variants construct the same parameter and buffer layout from the same initialization order. Only state selection, coefficient use, and the protected/direct output equation differ.

| Code | State/profile | Modality representation before fusion |
|---|---|---|
| PO | no graph state | `LN(P)` |
| RU | raw `H1:H3`, uniform | `LN(P + lambda (mean(H1:H3) - P))` |
| AU | anchored `S1:S3`, uniform | `LN(P + lambda (mean(S1:S3) - P))` |
| AP | anchored `S0:S3`, fixed converted CoSI prior | `LN(P + lambda (G - P))` |
| AGP | anchored `S0:S3`, `gamma_prior + delta_gamma_global` | `LN(P + lambda (G - P))` |
| AGD | same states/profile as AGP | `LN(G)` |

The direct variant still includes intrinsic semantics because `S0=P` and every anchored state includes repeated `P` injection. It lacks an explicit guaranteed prior injection path. Coefficient diagnostics are polynomial coefficients, not spectral energy.
