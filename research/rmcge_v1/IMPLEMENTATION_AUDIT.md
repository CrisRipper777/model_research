# RM-CGE V1 implementation audit

**Audited source revision:** `e32eaeb676b3851cffc0067c549f42a748569e12` (implementation working tree immediately before the freeze commit)
**Configuration:** `configs/model/rmcge_v1.yaml`
**Scope:** full-graph node classification, largest-graph CUDA preflight, cache integrity, PACE/V7 regression, finite gradients and updates, and checkpoint reload. This is an engineering preflight, not a training result.

## Summary

The implementation follows the frozen design in `DESIGN.md`. RM-CGE lives in its own model/config/runner/analyzer/test paths. The only shared training-runner change is opt-in epoch history collection, enabled by RM-CGE and inactive for existing models. The targeted PACE-MAG and V7 tests pass. The ele-fashion full-graph forward, backward, and AdamW step are finite; all audited core groups receive nonzero gradients and change under the optimizer. The validation-only checkpoint round trip is numerically consistent within `1e-6` absolute and relative tolerance. `evaluate_test=false`; no Test metric or Test label index was accessed by the preflight.

Machine-readable evidence: `data/implementation_checks.json`, `data/gradient_audit.json`, `data/cache_audit.json`, and `data/manifest.json`. The preflight checkpoint and resolved runtime configuration are under the ignored `outputs/rmcge_v1_pilot/preflight/` directory and are not committed.

## Cache and graph operators

The runner validated each existing V7B cache against the repository manifest and the actual Text/Visual feature identities. It checks dataset/node dimensions, candidate and V7B fingerprints, canonical candidate edges, and the two-column source flags. No FAISS job was run and no cache was modified. All three cache audits report `candidate_phase_label_access=false` and `cache_contains_labels=false`.

| Dataset | Nodes | Physical undirected edges | Text support edges | Visual support edges | Both-source edges |
|---|---:|---:|---:|---:|---:|
| Movies | 16,672 | 160,802 | 110,714 | 102,161 | 6,994 |
| Grocery | 17,074 | 142,262 | 109,198 | 104,566 | 8,207 |
| ele-fashion | 97,766 | 399,172 | 582,723 | 608,849 | 89,372 |

The implementation constructs sparse COO operators. Physical edges are made loop-free, undirected, and duplicate-free. Semantic edges are selected independently by `source_flags[:, modality] > 0`; values 1 and 2 both mean presence, and a Both edge appears at full unit weight in each support. Each relation is independently row-normalized, with isolated rows remaining zero. Synthetic tests check deduplication, row sums, flags 0/1/2, Both support, and empty rows. No dense node-by-node adjacency is built.

## Information flow, sharing, and routing

The tests check that Text semantic cross-modal context consumes the Text support applied to Visual state (`P_T H_V`), and Visual semantic cross-modal context consumes the Visual support applied to Text state (`P_V H_T`). Physical cross-modal context uses the common physical support. The direction MLP follows aggregation, and its output is masked for invalid rows so the MLP bias cannot inject a message. Instrumented tests verify that the second layer receives the first layer's updated states and recomputes propagation.

Within each layer there is one expert bank and one router shared by both modalities and both relation roles. Tests verify shared object identity and distinct parameters between layers. The router is conditioned on node state and contexts as well as learned relation/modality embeddings and neighborhood degree. Its single softmax spans all six relation-expert candidates. Tests check the approximate 0.6/0.4 initialization marginal, normalized joint mass, invalid relation masking, a sole valid relation, and exact zero for all-invalid rows.

The four expert blocks use the same non-affine feature LayerNorm rule before concatenation. Experts are `4d → 64 → d`; residual updates use the fixed factor 0.5. Fusion includes both updated modalities, absolute difference, elementwise interaction, and a learned skip from both initial projected states. The auxiliary loss is an exact scalar zero.

## Training and checkpoint engineering

The preflight used the full ele-fashion graph (97,766 nodes), hidden dimension 128, two layers, three experts, bottleneck 64, dropout 0.2, and the frozen configuration. It completed a full-graph forward, train-index-only Cross-Entropy backward, gradient clipping/AdamW parameter update, and checkpoint save/reload. Preflight measurements:

- Model parameters: 761,170; NC classifier parameters: 1,548; total: 762,718.
- Peak CUDA allocated: 5,854,918,656 bytes (about 5.45 GiB); peak reserved: 6,463,422,464 bytes (about 6.02 GiB).
- All floating outputs, core gradients, and audited parameters were finite.
- Projector, cross-modal, expert, router, graph-normalization, and fusion groups had finite nonzero gradients. Every parameter tensor in these groups changed after the AdamW step.
- Reloaded checkpoint maximum absolute output difference: `1.1920928955078125e-06`; `allclose(atol=1e-6, rtol=1e-6)` passed.
- Auxiliary loss exactly zero; Test evaluation disabled; no Test metrics and no Test labels indexed.

The runtime detected a free RTX 3090 (physical GPU 1), isolated it by setting `CUDA_VISIBLE_DEVICES=1` before CUDA initialization, and used it as logical `cuda:0`. GPU 0 was already occupied by another process and was not stopped or selected. The final successful run saw one visible GPU. Full environment and `nvidia-smi` snapshot are in `data/implementation_checks.json`.

## Regression tests

The final preflight ran:

```text
python -m pytest -q \
  tests/test_rmcge_v1.py \
  tests/test_pace_mag_v1.py \
  tests/test_pace_mag_v1_audit.py \
  tests/test_psce_mag_v7a.py \
  tests/test_psce_mag_v7b.py \
  tests/test_psce_mag_v7b1_corrective.py
```

Result: **101 passed, 1 skipped**, exit code 0. The 15 warnings are upstream PyG distributed deprecation and PyTorch checkpoint CPU-autocast deprecation warnings. Existing PACE/V7 source files and historical audit/results are unchanged in the implementation diff.

## Startup attempts and final evidence

Attempt notes are machine-readable in `data/preflight_attempts.json`. An earlier preflight initialized a small CUDA context on GPU 0 because the existing shared seed helper called `torch.cuda.manual_seed_all`; it did not stop or interfere with the other process, but this prompted process-level GPU visibility isolation. Two starts of the isolated runner then exposed a `mem_get_info` call that required an explicit logical device. The runner was corrected to query the selected logical device, and the final isolated preflight passed. The final evidence supersedes those startup attempts. No model dimensions or mathematical choices were changed to make the preflight pass.

## Freeze boundary and remaining limits

The exact source/config/test snapshot represented by the implementation freeze is fingerprinted in the machine-readable checks. The three formal seed-42 trainings, per-epoch Validation history, selected-checkpoint mechanism diagnostics, and frozen branch-off Validation interventions are intentionally pending and belong in `PILOT_REPORT.md`. There is no evidence here about generalization, multi-seed stability, or superiority over prior methods. No Test evaluation is authorized in this run.
