# MAG View-Collaborative Experts V1

## Question and relation to prior work

This study tests whether arranging existing physical, same-modality KNN, and
cross-source KNN messages as separately routed views improves full-graph MAG
node classification. Shared experts across physical and semantic views already
appeared in PSCE-MAG V7A, so shared use itself is not claimed as new. RM-CGE
V2.2 already has modality-source KNN operators and receiver-conditioned edge
content gates; V1 does not add or retune edge weights. It changes where those
fixed-source messages enter the shared expert bank and avoids injecting one KNN
message both into a physical profile and a separate semantic view.

## Conditions

- `A0_legacy`: calls the frozen RM-CGE V2.2 M0 forward directly.
- `A1_separate_intra`: physical profiles retain four physical hops and the
  existing directional physical cross projection. Text-KNN and Visual-KNN
  messages become their own same-source semantic views.
- `A2_cross_source`: adds the sender-modality support view. A Text target uses
  `S_visual Phi_(visual->text)(H_visual)`; a Visual target uses
  `S_text Phi_(text->visual)(H_text)`.

All conditions instantiate the same V2.2 common model tree and the same new
view-router/gamma parameters. New-module construction is isolated with
`torch.random.fork_rng`, preserving the later NC classifier initialization.

## Equations

Let `H_m` be the existing projected modality state. Physical hop states are
`U_P^(m,l)=P^l H_m`, for `l=1..4`, normalized by the inherited active-node
RMS. `C_P^m=P Phi_(o->m)(H_o)`. The physical expert input is

`B_(m,k)^P = sum_l alpha_(k,l) Norm(U_P^(m,l)) + b_k Norm(C_P^m)`,

where `b_k` is the inherited V2.2 cross coefficient. In A1/A2 the old KNN
profile term and the V2.2 content-gate term are absent from this profile.
The physical path uses the inherited V2.2 base route, node correction,
Top-2 selection, and node-level strength, and retains the `0.1 F_k` residual.

The same-source view is `U_S^m=S_m H_m`. The cross-source view is
`U_C^m=S_o Phi_(o->m)(H_o)`. Both are divided by a detached per-channel RMS
of the corresponding target modality's first physical hop, clamped by the
existing scale floor. A view mask is computed from its own incoming support.
Its response is `R_r^m=Uhat_r^m + sum_k pi_(m,r,k) F_k(Uhat_r^m)`, using the
same `self.experts[k]` objects and one shared 64-wide view router for S/C.
The router sees target state, calibrated view state, target-modality embedding,
and view-type embedding, then uses Top-2. Its final layer starts at zero.

`Z_m=H_m + rho_m^P R_P^m + gamma_m^S M_S^m R_S^m + gamma_m^C M_C^m R_C^m`.
Positive modality-specific gamma values use softplus, initialized at 0.08 and
0.05. A1's C term is exactly disabled. Empty views are masked after their
expert response and before fusion. The only auxiliary loss is the single
inherited physical-router balance term; semantic route usage is diagnostic.

## Frozen data/training boundary

The study uses the existing V7B.1 K=8 cache and V2 operator construction,
`unified_full_graph_nc_v1`, official per-seed splits, full-graph transductive
NC, AdamW `1e-3`, weight decay `1e-4`, dropout `0.2`, 300 epochs, patience 30,
minimum epoch 30, gradient clipping 1.0, and Validation Accuracy selection.
Best Validation Macro-F1 is retained independently. Test metrics are disabled;
Test labels are replaced with `-100` before training and validation. Link
prediction is explicitly unsupported because caches use global node IDs.

Source code, implementation checks, the experiment plan, and preflight evidence
are frozen before any formal run. Claims are based only on paired internal A0,
A1, and A2 validation results over the same Dataset × Seed.
