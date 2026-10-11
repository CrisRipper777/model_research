# PACE-MAG V1 implementation audit

## Baseline and branch

- Repository: `CrisRipper777/model_research`
- Verified base branch: `exp/v7b1_corrective_replication`
- Verified base SHA: `aa62807c113fde17db058d05d558d2635bccc90b`
- New branch: `exp/pace_mag_v1_relation_view_experts`
- The workspace was clean before branch creation. The fetched remote base still matched the requested SHA. No historical branch or commit was modified.
- Implementation Freeze SHA: pending the final implementation commit.

## Environment

Captured by `research/pace_mag_v1/data/environment.json`:

- Python 3.12.13
- PyTorch 2.4.0+cu121; CUDA build 12.1
- PyG 2.7.0; DGL 2.4.0+cu121; FAISS 1.15.1
- Two NVIDIA GeForce RTX 3090 GPUs
- Selected execution device: `cuda:1`; initial free memory was about 23.4 GiB.
- GPU 0 was already running another VLLM process at high utilization and about 23 GiB memory use. It was not interrupted or selected.
- The `nvidia-smi` executable transiently failed its first probe, while PyTorch successfully enumerated both GPUs. The later environment snapshot succeeded and recorded both GPUs and the active processes.

## Implementation files

Added for the method branch:

- `src/models/pace_mag_v1.py`: all five variants, sparse physical/semantic operators, cache validation, model, optional diagnostics, and gradient summaries.
- `configs/model/pace_mag_v1.yaml`: frozen common architecture and execution settings.
- `scripts/run_pace_mag_v1.py`: cache audit, real-graph preflight, validation-only campaign runner, per-run configuration/resource/provenance records.
- `scripts/analyze_pace_mag_v1.py`: label-free selected-checkpoint diagnostics.
- `tests/test_pace_mag_v1.py`: operator, source flag, cache, model variant, gradient/update, checkpoint, validation-only, chunking, CPU, and CUDA checks.
- `research/pace_mag_v1/DESIGN.md`: method and variant specification.

Existing training framework, data, splits, semantic caches, and historical outputs were not modified. No original graph, labels, splits, cached candidate graph, or historical result was regenerated. Reused semantic cache files stay under the existing ignored `outputs/psce_mag_v7b/semantic_cache/` paths. Pilot checkpoints/logs go under ignored `outputs/pace_mag_v1_pilot/`.

## Cache and source graph audit

The model loader validates V7B schema/fingerprints, dataset, feature identity, candidate identity, V7B source fingerprint, node count, edge range, unique canonical edge order, source-flag shape/dtype/range, and the exact symmetric directed-edge order associated with the unique-edge flag rows. On first forward it also hashes the actual input Text and Visual arrays and compares both hashes to the cache identity.

The run script independently recomputes the feature fingerprint from each configured feature file before accepting the cache. Its label-free audit is in `data/cache_audit.json`:

| Dataset | Nodes | Unique KNN union edges | Directed edges | Cache |
| --- | ---: | ---: | ---: | --- |
| Movies | 16,672 | 205,881 | 411,762 | Existing V7B cache |
| Grocery | 17,074 | 205,557 | 411,114 | Existing V7B cache |
| ele-fashion | 97,766 | 1,102,200 | 2,204,400 | Existing V7B cache |

All three cache feature, candidate, and V7B fingerprints matched. All cache payloads are label-free and include valid raw source counts in `{0,1,2}`.

## Correctness and regression checks

Commands run in `yhf_env`:

```bash
python -m py_compile src/models/pace_mag_v1.py scripts/run_pace_mag_v1.py scripts/analyze_pace_mag_v1.py tests/test_pace_mag_v1.py
python -m pytest -q tests/test_pace_mag_v1.py
python -m pytest -q tests/test_psce_mag_v7a.py tests/test_psce_mag_v7b.py tests/test_psce_mag_v7b1_corrective.py
```

Results:

- PACE-MAG tests: 22 passed. Coverage includes `PT+PV=PU`, source-only/Both allocations and 0/1/2 source counts, symmetric edge alignment, repeated physical edges and self-loops, empty-row suppression, exact rho-zero degeneration, E0–E4 forward passes, shared expert object reuse, router probability checks, approximate E3 parameter matching, finite nonzero E2 gradients and actual AdamW updates, E1 missing cross-modal gradients/updates, cache/feature/node-count fingerprint rejection, checkpoint/cache reload inference, `evaluate_test=false`, chunked activation-checkpoint forward/gradient agreement, and CUDA forward/backward.
- Existing V7A/V7B/V7B.1 tests: 56 passed.
- Both suites emitted upstream PyG/checkpoint deprecation warnings only; no test failures remain.

## Actual graph preflight

`python scripts/run_pace_mag_v1.py --phase preflight --dataset Movies --device cuda:1` completed twice while finalizing the implementation. The final recorded attempt is `outputs/pace_mag_v1_pilot/preflight/Movies_attempt_02/preflight_checkpoint.pt`; structured results are in `data/implementation_checks.json` and `data/gradient_audit.json`.

The final attempt completed one full Movies E2 forward, train-split Cross-Entropy backward, gradient clipping, and AdamW step, then saved/reloaded the model and verified inference. Loss and all checked gradients were finite. Projectors, cross-modal MLPs, experts, router, and fusion each had finite nonzero gradients and changed parameters. Reloaded embeddings differed by at most `8.35e-7`. `aux_loss` was zero. Test metrics were disabled and not computed. Peak CUDA allocated memory was 1,572,432,384 bytes (about 1.46 GiB); peak reserved was 1,799,356,416 bytes (about 1.68 GiB). No OOM or training exception occurred.

The added chunk size is 32,768 nodes with non-reentrant activation checkpointing for expert MLP chunks. A small-graph test confirms the chunked and whole-batch expert paths agree in forward values and gradients with dropout disabled. This controls activation memory on ele-fashion without changing operators or learned equations.

## Gradient checks

The real Movies E2 preflight found nonzero gradients and AdamW updates in the Projectors, both Cross-Modal bottlenecks, all three Experts, Router, Fusion, `rho`, and both `beta` scalars. The unit audit separately confirms that E1's cross-modal parameters and `rho` receive `grad=None` and do not change under AdamW weight decay. These checks establish implementation wiring and optimization, not mechanism effectiveness.

## Implementation differences and known limits

- No structural deviation from the requested equations was found.
- `P_T+P_V=P_U` is verified at the sparse-operator level. The row-quality aggregation is an explicit extra normalization, and the downstream expert/router operations are nonlinear, so whole-model linear reconstruction is not claimed.
- E3 matches the full Expert+Router+Embedding parameter budget with the closest integer single-MLP width. Exact equality is not promised; per-run width and signed parameter delta are recorded in the Pilot CSV.
- E0 keeps both learnable semantic route slots, as requested; it does not represent every possible source-agnostic model.
- The model deliberately rejects Link Prediction and any graph node count different from its global-ID semantic cache.
- Pilot and Test metrics do not yet exist at this implementation-freeze stage. Implementation success is not evidence of an accuracy benefit.
