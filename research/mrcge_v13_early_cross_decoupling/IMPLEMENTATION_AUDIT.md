# Implementation audit

- Base commit ancestry: verified from `021959d82ec74a71955c97521867dc0d3f4a6f01`.
- Frozen V1.3-A, Static-Strength, NoP0 basis-screen, Hybrid-CGE, trainer, configs and relevant regression tests: SHA256 equality checked against the base before CPU preflight.
- New model: subclasses frozen V1.3-A and adds only `cross_coefficients` with shape `[4]`.
- Initialization: for all dataset/seed cells, V1.3-A common state tensors and external classifier initial state matched; `b` equaled that instance's initial physical `alpha[:,1]`.
- Mathematical path: `G_old + beta_m * (b - alpha_current[:,1]) * C_m`; alpha remains attached to autograd. Unit tests include analytical theta Cross-path cancellation and finite nonzero b gradients.
- Both stacked evaluation/diagnostic and serial low-memory checkpoint paths are implemented and covered by parity tests.
- Router input/logits, shared experts, bottlenecks, LayerNorm, Joint View, Cross calibration, paired-dropout replay, fallback, private/semantic paths, and trainer were inherited unchanged.
- Exact authorized training cells: 7; completed and dual-checkpoint verified: 7/7.
- All Test/LP flags are false; test labels were masked before Trainer.
- Historical source/data/checkpoints were read-only and protected by pre/post hash checks.
- Historical model-source hashes were checked against their recorded Git commits for 17 references (7 V1.3-A, 7 V1.2 and 3 fixed-split Hybrid); paired checkpoint hashes were verified.
- Report-generator correction was committed after training; pre-fix SHA256 `ab0bfa1e80af7b2cdda466aecd24b2ecad8dd6c218dc0ce881f0fc006b007788` and post-fix SHA256 `fcd941e1881f20e67f88bcfd260d13b64ad7e9aa22ded944953d7793515da180` are recorded here. The correction changed report generation and post-training audit only. The frozen model, Trainer, config, data-pipeline hashes and all training/checkpoint provenance were unchanged.

Unit and regression test output is recorded in `research/mrcge_v13_early_cross_decoupling/test_results.txt`.
