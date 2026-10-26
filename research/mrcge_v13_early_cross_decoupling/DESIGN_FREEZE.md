# MR-CGE V1.3-A Expert-Decoupled Early Cross — design freeze

## Base and scope

- Repository: `/hdd1/DataInHere/YHF/model_research`
- Base: `021959d82ec74a71955c97521867dc0d3f4a6f01` on `exp/mrcge_v13_static_strength_screen`
- New branch: `exp/mrcge_v13_early_cross_decoupling_screen`
- One new condition: V1.3-A `M_repaired` + `joint_view` + expert-decoupled Early Cross
- Authorized new training cells: Movies seeds 42/43/44; Grocery 42/43; ele-fashion 42/43
- Physical GPU: 1 only. No Test evaluation, Link Prediction, additional seed, or reference retraining.

## Frozen mechanisms confirmed in source

The V1.3-A expert bank has four shared experts, one shared `theta[4,4]`, shared Bottlenecks and LayerNorms, and one shared Graph-aware Top-2 Router. Physical bases are orders 0–3; Semantic bases are orders 0–1. Private and Collaborative Physical calls use the same bank. Collaborative order 1 is `B_m,1 + beta_m*C_m`; `_early_cross` retains the original projection, physical weighted aggregation, RMS calibration, support mask, and zero-message behavior. The Router still receives the original `current`, Collaborative `first_neighbor`, their difference, modality/relation embeddings, and `C_m`. The inherited Joint View, paired RNG replay for Private/Collaborative dropout, and `_zero_context_fallback` remain active.

V1.3-A has both a stacked `SharedExpertBank.forward` route used for ordinary Evaluation/diagnostics and a serial `_m_filter_impl` route used under the non-reentrant low-memory training checkpoint. The new correction is applied to each route's per-expert filtered tensor before its Bottleneck and LayerNorm; it does not change the Router input or logits.

## Sole mathematical change

For `alpha[k,l] = theta[k,l] / (sum_j |theta[k,j]| + eps)`, Private Physical remains `G_private[m,k] = sum_l alpha[k,l] B_m,l`. Semantic continues to use the existing order-0/1 coefficients. The calibrated Cross context remains `C_m`, and `beta_m = sigmoid(beta_raw[m])` is unchanged.

The old Collaborative expert input is `G_collab_old[m,k] = G_private[m,k] + beta_m*alpha[k,1]*C_m`. Add exactly four trainable shared-over-modality values `cross_coefficients[k]`, initialized by deterministic copy from this model's own just-initialized `alpha_initial[k,1]`. Apply only:

`Delta_G[m,k] = beta_m * (cross_coefficients[k] - alpha_current[k,1]) * C_m`

`G_collab_new[m,k] = G_collab_old[m,k] + Delta_G[m,k]`

`alpha_current` stays attached to autograd in the forward correction so the original theta-to-Cross coefficient gradient is canceled. Only the initial copy may detach. No extra random initialization is performed. When initialized (or when `b` is explicitly copied from current alpha), the correction is zero and the frozen V1.3-A forward is preserved. When `C_m=0`, the correction is exactly zero and the inherited fallback returns the Private response.

## Implementation and acceptance checks

The new model subclasses the frozen V1.3-A model, adds only `cross_coefficients[4]`, and locally adapts only the Physical filter's stacked Evaluation/diagnostic path and serial low-memory checkpoint path. The Collaborative basis, Router definition/input, theta, alpha normalization, experts, Joint View, Cross source/calibration, Late completion, fusion, and Trainer remain frozen.

Before GPU work, tests will check: complete public state and external classifier initialization equivalence; exact initial Eval/Train outputs and paired dropout replay; all four new coefficients' finite nonzero task gradients; analytic cancellation of only the theta-to-Cross coefficient path while Private/Semantic/own-Physical theta gradients remain; the two filter execution paths; zero Cross and empty support; both modality directions; b independence from theta and Forward-time resetting; and equivalence when b is recopied from a perturbed current alpha. Historical inputs, checkpoints, and study records are read-only and hash-audited.

## Data, safety, and interpretation boundary

All datasets use the frozen split fingerprints and verified V7B.1 K=8 semantic caches, `auto_generate_splits=false`, and training seeds independent of split seed 42. Test labels are masked before Trainer entry. The fixed training recipe is the V1.3-A full-graph NC recipe (AdamW, LR 0.001, WD 0.0001, 300 max epochs, patience 30, minimum epoch 30, delta 0.0001, gradient clip 1.0, dropout 0.2, Late dropout 0.15, unimodal CE 0.3, balance loss 0.01). Best Validation Accuracy and Best Validation Macro-F1 checkpoints are independently saved, strictly reloaded, and audited.

Every authorized cell receives its own physical GPU 1 one-step preflight. The existing ownership guard must pass a fresh baseline, at least 8 GiB launch free memory and at least 2 GiB above measured peak reserved, and continuous foreign-PID/resource monitoring. Only this task's child may be stopped. Any block/failure/retry is retained.

The results are a small-sample descriptive screen against historical fixed-split V1.3-A and Hybrid references. Different optimization trajectories and the joint effect on routing/Joint View prevent strict causal attribution. No significance or SOTA claims, default-model selection, or follow-on model work is authorized by this freeze.
