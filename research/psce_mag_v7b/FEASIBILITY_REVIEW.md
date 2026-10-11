# PSCE-MAG V7B feasibility and code review

Review date: 2026-10-11
Branch: `exp/v7b_provenance_relative_semantic_edges`
Baseline: `bcf0d8328642dc6747f8f94f42e56b1717555d64`

## Decision

The implementation can reuse the V7A full-graph NC task, physical path, shared experts, conditional semantic router, balance loss, candidate-cache attachment point, and sparse weighted SpMM. B0/B1/B2 will inherit the V7A model and retain its original forward path. B3/B4 require a V7B forward path because Text and Visual need distinct first-hop semantic weights in B4; that path will retain the V7A order `S1 = P_sem @ H0`, then `S2 = P_phy @ S1`, and reuse its calibration, experts, router, fusion, and balance methods.

The cache builder can call the existing V7A centering and FAISS routines, while preserving raw directed query proposals long enough to count each source. The new cache will keep the V7A graph fingerprint as a separately verifiable identity and add a V7B provenance fingerprint. Candidate graph equality will be checked against the committed V7A summary and local frozen cache before formal training.

## Existing interfaces reviewed

- `src/models/psce_mag_v7a.py`: independent encoders, physical normalization/4-hop Raw path, active RMS, Hadamard coefficients, shared expert transforms, physical and semantic routers, balance loss, final fusion, cache fingerprint checks, `ChunkedWeightedSpMM`, and selected-checkpoint diagnostics.
- `src/data/semantic_candidates.py`: deterministic sorted canonical edges, symmetric directed layout aligned to unique edges, fixed degree, exact FAISS for the two smaller graphs and seeded IVF for ele-fashion, and tensor/metadata fingerprint validation.
- `src/tasks/nc.py` and `src/tasks/inference.py`: full-graph `unified_full_graph_nc_v1`, Validation Accuracy checkpointing, optional test evaluation, cache attachment before device transfer, and full inference. The V7B launch config will explicitly set `task.evaluate_test=false`.
- `src/tasks/lp.py`: sampled LP node IDs are local; the V7A model already rejects application of its global-ID semantic cache. V7B will retain that rejection.
- `scripts/run_psce_mag_v7a.py`: source-only dataset input loading avoids labels/splits for graph construction; its campaign provenance, metrics validation, checkpoint validation, and resume patterns are suitable to adapt.
- `tests/test_psce_mag_v7a.py`: covers existing physical-path and cache invariants and remains unchanged.

## Frozen scope and planned deltas

Keep V7A A2's public parameters, physical graph and Raw propagation, experts, physical routing, semantic coefficients, conditional semantic router/strength, one-sided semantic RMS calibration, fusion, balance objective, and NC protocol. Add only candidate source counts, a two-output relation scorer, relative local edge weights, three fixed/control variants, provenance caches, and per-modality first-hop semantic operators for B4. B0 will be a direct V7A A2 forward; B1/B2 will use constant 0.8 edge weights, with B2 changing only the semantic candidate topology through degree-preserving double-edge swaps.

## Numerical and implementation risks

1. **FAISS ties or cache mismatch.** The builder must preserve the frozen V7A FAISS configuration and stable tie ordering. Graph tensors, not just edge count or degree, will be compared byte-for-byte to each V7A cache.
2. **Source-count orientation.** Each canonical edge's Text/Visual source count must come from both raw query directions before graph symmetrization. The count is capped at two and copied identically to the reverse propagation edge.
3. **Relative-weight gradients and translation invariance.** Endpoint means use differentiable `index_add`; scores are not detached. Tests will check a global score shift leaves weights unchanged while scorer and both output-head gradients remain finite.
4. **Fixed-weight controls.** B1/B2 will bypass the scorer and use a literal 0.8 vector, so optimizer parameter enumeration must exclude their unused relation modules.
5. **Random rewiring.** Double-edge swaps must preserve all degrees exactly and reject loops/duplicates. The fashion graph is large; swap throughput and topology change will be measured before generating all seed graphs. Failure to reach the predeclared swap target will stop the formal campaign rather than silently use a different null graph.
6. **B4 memory.** Two E-length weight vectors are needed, but edge indices and degree are shared. Scoring and SpMM remain chunked; no dense pairwise matrix or edge-by-expert messages are introduced. Preflight will run forward, backward, optimizer update, and complete validation inference on all three graphs.
7. **Head identifiability.** B3 averages the same two scorer outputs. Its common score is learnable, while head-difference directions are unidentifiable under that average; this will be stated in the report.

## Baseline regression strategy

- On a CPU toy graph with one state dict, compare `psce_mag_v7b` B0 and the frozen `psce_mag_v7a` A2 for embeddings, auxiliary loss, and semantic details.
- Compare B0/B1/B2 common public module tensors and classifier initialization under each seed. Compare all shared B3/B4 tensors plus scorer shape and exact initial values.
- On the Movies full graph, load a V7A A2 checkpoint and candidate cache into both model implementations and compare full-graph outputs and auxiliary losses within `atol=2e-6, rtol=2e-5` before any formal run.
- Verify exact base candidate edge and degree equality for all three datasets against V7A's locally cached tensors and frozen fingerprints.

## GPU availability

The host exposes two RTX 3090 devices. At review time GPU 1 was idle with 24 GiB free; GPU 0 was occupied by an unrelated process with about 23 GiB allocated. Only GPU 1 will be used. No process will be terminated.
