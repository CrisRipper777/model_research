# PSCE-MAG V7A feasibility and implementation review

Review date: 2026-10-10
Branch: `exp/v7a_physical_semantic_collaborative_experts`
Baseline: `af848e6c241994d34827fab8d37611159d167965`

## Decision

The feature-only candidate graph builder and full-graph NC model are implemented. FAISS is installed and its exact/IVF APIs have been exercised. Candidate caches were built for all three requested datasets without reading label or split fields. The historical Movies V4A R0 checkpoint maps into the new A0 Raw path with zero observed output and auxiliary-loss error on the full graph.

The CPU correctness suite passes (27 V7A tests and 356 repository tests). Three one-epoch Movies CPU integration smokes exercised the Hydra training path, validation selection, checkpoint serialization, and cache-fingerprint validation. A host-level run then passed the GPU acceptance gate on GPU 1 (RTX 3090): A2 completed one train/backward/optimizer step and full-graph validation on all three datasets, and A0/A1/A2 Movies GPU smokes completed. The sandbox itself has no NVIDIA device nodes; the host GPU was accessed through the approved host execution path. No process was stopped.

## Reusable code and new components

- The V4A `R0_raw` path in V3C/V3A/V2.1 supplied the common projectors, loop-free symmetric physical normalization, four Raw hops, active-node RMS, Hadamard coefficients, bias-free expert transforms, static Top-2 router, strength head, and late fusion block. New A0 matches the historical checkpoint path on Movies.
- `src/tasks/nc.py` retains the existing full-graph `unified_full_graph_nc_v1` training loop and Validation Accuracy checkpoint selection. V7A attaches its candidate cache before device transfer and records selected-checkpoint semantic diagnostics using validation only.
- `src/data/semantic_candidates.py` performs label-free mean-centering, L2 normalization, K=8 retrieval, Text/Visual union, canonical deduplication, symmetrization, fixed-degree calculation, integrity checks, and cache fingerprinting.
- `src/models/psce_mag_v7a.py` implements A0/A1/A2, a differentiable multimodal relation scorer, chunked edge scoring and weighted sparse aggregation, checkpointed shared expert transforms, one-sided semantic RMS calibration, and cache/feature fingerprint checks.
- `src/tasks/lp.py` rejects V7A before LP sampled-subgraph processing because the candidate IDs are global and LP node IDs are local.

No V6A Attention, deletion-effect estimate, OT/InfoNCE loss, independent relation graph, private expert pool, or NC sampling path was added.

## Candidate graph results

K=8 is fixed for each modality. The FAISS index, seed, thread count, recall sample, cache fingerprints, degree/similarity quantiles, and timing are recorded in `data/candidate_graph_summary.json`.

| Dataset | Nodes | Undirected candidate edges | Nonphysical fraction | Text/Visual overlap Jaccard | Feature-only sampled Recall@8 (Text / Visual) |
|---|---:|---:|---:|---:|---:|
| Movies | 16,672 | 205,881 | 0.9121 | 0.0340 | exact retrieval |
| Grocery | 17,074 | 205,557 | 0.9378 | 0.0399 | exact retrieval |
| ele-fashion | 97,766 | 1,102,200 | 0.9615 | 0.0811 | 0.9883 / 0.9883 |

The approximate recall audit used 256 feature queries with seed 2026. These topology statistics do not establish task value.

## Dependency and memory plan

`faiss-cpu==1.15.1` is installed in `yhf_env`; the package install left the existing PyTorch and NumPy versions unchanged. Movies and Grocery use exact `IndexFlatIP`; ele-fashion uses seeded `IndexIVFFlat`, with settings captured in the cache metadata. No dense `N × N` similarity matrix is built.

For ele-fashion, the symmetric candidate graph has about 2.20 million directed entries. The model scores canonical pairs in chunks, recomputes score chunks in backward, and uses a chunked custom weighted SpMM whose backward recomputes edge gathers instead of saving edge-by-hidden messages. Shared expert FFNs are activation-checkpointed during training. On the 24 GiB GPU preflight, peak allocated memory was 3.30 GiB (Movies), 3.15 GiB (Grocery), and 13.79 GiB (ele-fashion); peak reserved memory was 3.51, 3.51, and 14.71 GiB, respectively. All three completed without OOM. Exact resource and timing records are in `data/preflight_summary.json`.

## Numerical and code checks completed

- Mapped 48 shared V4A R0 weights from the historical Movies seed-42 checkpoint. Across all 16,672 nodes, max absolute and relative `z` errors were both 0; auxiliary-loss error was 0. Tolerance: `atol=2e-6`, `rtol=2e-5`. Full details are in `data/raw_regression_summary.json`.
- Seeded A0 common-trunk and NC classifier initialization matches V4A R0. A1/A2 have identical parameter counts and initial semantic tensors.
- CPU tests check all three variants, feature slicing, candidate rules/determinism/version binding, non-self symmetric deduplication, relation and projection gradients, fixed-degree weight scaling, shared experts, coefficient rows, semantic propagation formula, physical-isolate masks, semantic-off degeneration, cache restore mismatch for both graph and feature fingerprints, dense-reference SpMM forward/backward, validation-only label universe, and LP rejection.
- The focused V7A tests passed 27/27; the full repository suite passed 356/356. The test artifact records the implementation fingerprint.

## GPU preflight and smoke gate

The A2 preflight passed on `cuda:1` for Movies, Grocery, and ele-fashion. Each row used the NC train labels for one finite loss/backward/optimizer step and then ran full-graph validation inference. Test metrics were disabled and absent. All recorded router, shared expert, semantic coefficient, relation projection/scorer, and semantic router gradients were finite and nonzero. Peak allocated/reserved memory and relation-scoring, semantic-propagation, optimizer-step, and validation timings are recorded per dataset in `data/preflight_summary.json`.

One-epoch GPU Movies smokes for A0, A1, and A2 also passed checkpoint and validation-only audits. The smoke audit confirms finite validation metrics, absent Test metrics, and selected-checkpoint semantic diagnostics for A1/A2. These one-epoch values are software-path checks, not formal comparative evidence. Detailed records are in `data/smoke_summary.json`.

GPU 1 was shared with other activity before and during these checks; no process was stopped. The sandbox did not expose `/dev/nvidia*`, so GPU work used the approved host execution path. The machine's GPU state is external to this branch.
