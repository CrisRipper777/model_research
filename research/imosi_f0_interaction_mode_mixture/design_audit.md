# IMoSI-F0 design audit — Interaction-Mode Mixture Prototype

## Provenance and scope

- Required parent branch: `exp/pcrr_e01_target_only_control`
- Required parent SHA: `188b940a88b1852608c9e2e25e6d8bd7013ed8b5`
- This branch was created directly from that clean SHA; no other experiment branch is merged or cherry-picked.
- This is an interaction-mode mixture screen, not another PCRR extension or an attention/alignment/OT experiment.
- Fixed NC splits remain unchanged. NC test evaluation and formal LP are out of scope; the requested LP smoke is protocol-only.

## Frozen representation backbone

The raw direct GPR / RGD core is frozen from PCRR-MAG v1: the same modality projectors, normalized physical graph operator, raw 0-to-3-hop GPR, `c_prior + delta_c_raw`, modality LayerNorm, streaming GPR, chunked projection, CPU feature staging, and node-aligned late fusion. No topology, edge weights, propagation operator, anchoring, relation calibration, modality-specific GPR, or propagation depth is changed.

For each modality, the interaction operates on the representation `G_m` produced by that fixed backbone. Preserve, Self, and Paired are representation-level interaction modes; they are not edge roles:

- Preserve: `G_m`.
- Self: `G_m + r_self(G_m, G_m)`.
- Paired: `G_m + r_pair(G_m, G_other)` with same-node other-modality source.

The pair representation remains `[target, source, abs(target-source), target*source]` with width 1024. Self and Paired share `pair_norm` and `pair_down`, but use separate zero-initialized `self_up` and `pair_up` response heads.

## Mode selection and safe start

The routing feature is `[G_target, P_target, G_target-P_target, G_other, abs(G_target-G_other), G_target*G_other]` (1536 dimensions). It contains no labels, degree bins, classifier outputs, or validation information. The router is shared across modalities. Each modality has its own learnable `[preserve,self,paired]` base logits, initialized from `[0.6,0.2,0.2]`; the node-specific router output starts exactly zero.

All F0-only modules are initialized in a CPU RNG fork seeded with `model_seed + 91000`, then the original global CPU RNG state is restored. The shared RGD/fusion/pair modules retain PCRR-v1 creation order so their tensors and the downstream NC classifier RNG remain matched.

All variants instantiate the same state layout:

- B / Base: Preserve only; bypasses the mode experts and must regress to PCRR-v1 Base.
- G / Global mix: joint training of both experts with one learned mode mixture per modality shared across nodes.
- M / Adaptive mix: joint end-to-end training of backbone, experts, and node router; logits add the router's node delta to the modality's base logits.

The final representation is computed as `G + pi_self*delta_self + pi_pair*delta_pair`, with no post-refinement LayerNorm. Zero-initialized expert heads make B/G/M exactly equal to RGD at initialization; zero router output makes initial M routes equal G routes.

## Evidence and interpretation boundaries

E0/E0.1 provide empirical motivation for preserving Self and Paired as candidate modes, but whether node-level routing is learnable or useful remains unknown. The complete retrained M−B comparison is the primary evidence. G−B tests whether global mode composition suffices; M−G tests the incremental value of node-specific routing. No individual expert is required to beat B alone.

Routing heterogeneity, entropy, expert-use frequencies, or checkpoint `globalize_router` changes are descriptive/reliance evidence only. Retrained M−G remains the evidence for adaptive routing. No auxiliary routing/expert regularizer or performance threshold is introduced. Interpretation uses paired seeds and population SD, without pseudo-IID p-values or latent-role claims.

## Memory and task protocol

Mode refinement is streamed by modality and node chunks of 8192. The implementation computes each chunk's Self correction, Paired correction, route, and mixture before moving on; it does not materialize an `[N,1536]` router input or keep four full-node correction tensors resident. Large-graph training uses activation checkpointing around chunk refinement. The NC task remains CE-only and unchanged; LP is only the requested two-batch protocol smoke for B/M, with `[5,5,5]` neighbors, positive-edge removal, and validation inference.

## Planned gates

1. Add implementation/tests and run the full repository pytest suite.
2. Verify clean worktree and commit source before any smoke or training.
3. Record the source commit in `run_manifest.json`; only then run B/G/M NC smoke, B/M LP smoke, and 27 validation-only formal NC runs.
4. Analyze selected checkpoints for routing distributions, effective expert contributions, GPR coefficients, and M `globalize_router` checkpoint intervention.
5. Commit results separately, push the experiment branch, and stop for human review.
