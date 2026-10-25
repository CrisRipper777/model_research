# Implementation audit

Static-Strength subclasses the frozen V1.3-A M_repaired backbone and mirrors its Forward locally so modality_id is explicit at both Private and Collaborative scale calls. The only new trainable tensor is `static_logits[2,2]`; it starts at zero and maps through `2*sigmoid`. The inherited Joint View MLP is constructed inside its isolated RNG scope to preserve the common backbone and external classifier initialization, then removed from the parameter tree. Private uses `sP*rho*R_P + sS*gamma*R_S`; Collaborative adds `rho*sP*(R_P_collab-R_P_priv)` and reuses the same modality-level scales. No module hooks, global mutable mode, or dataset-name dispatch is used.

The static tests cover factory/config gating, exact shared initialization and external head initialization, initial Eval/Train equivalence, paired-dropout replay, common gradients, nonzero Static task gradients, modality/relation separation, node constancy, identical Private/Collaborative scales, both Early Cross directions, no-support behavior, low-memory checkpointing, strict checkpoint reload, and validation RNG preservation. Historical V1.3-A, NoP0, Hybrid-CGE source and result files are hash-checked and read-only.

Training source SHA: `7704c4edd43cddf4f7418a732bf5a6805e10d6d9`. Static model parameters: four beyond the V1.3-A common model, with Joint View removed.

After the seven runs completed, the report generator needed explicit CSV field lists at four write sites. That analysis-only correction did not change the model, training runner, configs or any training artifacts; the frozen training SHA remains the source recorded in every run row.
