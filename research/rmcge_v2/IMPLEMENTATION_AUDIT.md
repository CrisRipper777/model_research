# RM-CGE V2 Implementation Audit

## Frozen mathematical contract

The design review is recorded in `DESIGN.md`. V0 delegates directly to
V7A `a0_raw`. V1/V2 retain that physical operator and shared expert objects;
their added paths are initialized in a forked CPU RNG scope. V1/V2 share the
same parameter tree. Physical edge weights multiply the original R0 normalized
coefficients without recomputing degree or renormalizing. The change path is
computed as `SpMM(H, P*(w-1))`; cross-modal projections run before physical
aggregation. Text and Visual KNN supports are independently formed from V7B.1
presence flags, retaining Both edges in both views. New contexts use detached
per-feature active-node RMS of the raw first physical hop with a `1e-3` floor.
The single parent expert bank receives the combined profiles, and the
zero-initialized node correction is added before the existing Top-2 rule.
Balance is recomputed from the corrected routes over active physical nodes.

## Engineering evidence

Preflight ran with `conda run -n yhf_env python scripts/run_rmcge_v2.py
preflight --gpu 1`. The recorded full regression command was
`python -m pytest -q`: **461 passed, 1 skipped**. The separate V2 targeted suite had 10
passed, covering V0 state/output identity, V1/V2 parameter and classifier
pairing, unit-edge equivalence, disabled-path recovery, directed receiver and
sender features, provenance supports, isolated/empty rows, custom SpMM
forward/backward, first-step gradients, AdamW updates, and checkpoint reload.

V0 is a subclass delegation to the V7A `a0_raw` forward; the direct state and
forward tests compare outputs exactly on the same CPU inputs. V1/V2 initial
parameter names, shapes, and values match exactly; their classifier states
also match. The cache-backed initialization audit passed this check for all
nine Dataset × Seed pairs. Each audit row records the V0/V1/V2 initial model
and head hashes. In the unit-edge test, V2 with forced unit weights returns
the exact same four output tuple values as V1. The test also checks that
disabling all new paths restores V7A `a0_raw` output.

The cache audit matched the existing V7B.1 metadata and current feature
sources: Movies has 16,672 nodes / 160,802 physical directed edges / 205,881
canonical KNN union edges; Grocery has 17,074 / 142,262 / 205,557; ele-fashion
has 97,766 / 399,172 / 1,102,200. Feature, candidate, and V7B fingerprints
are in `data/implementation_checks.json`. Source flags are validated by the
V7B loader; modality supports use flag presence and each directed support is
row-normalized separately.

The environment was Python 3.12.13, NumPy 2.4.3, PyTorch 2.4.0+cu121, CUDA
build 12.1, cuDNN 90100, PyG 2.7.0, DGL 2.4.0+cu121, and FAISS 1.15.1. Driver
535.309.01 exposed two RTX 3090s. Physical GPU 0 had 1,394 MiB free and was
left untouched; physical GPU 1 had 24,246 MiB free and was isolated for the
smoke step. TF32 matmul was false, cuDNN TF32 true, deterministic algorithms
false, and `CUBLAS_WORKSPACE_CONFIG` unset. The run code did not change these
settings. `nvidia-smi` process and device snapshots are retained in the JSON.

The ele-fashion V2 two-step forward/backward/AdamW smoke completed on the 24 GB
RTX 3090 with finite losses (2.5558 then 2.2296) and gradients. Peak allocated
memory was 12,100,887,552 bytes (about 11.27 GiB); peak reserved memory was
12,561,940,480 bytes. Both layers of both Cross Projection directions, Node
Router, experts, projectors, base selection/strength, and fusion received
finite gradients; audited parameters updated. The Edge Scorer output layer
received a finite nonzero first-step gradient. At step one, Delta coefficients
and the zero-output Edge Scorer hidden layer had zero task gradients, as
expected while weights start at one and the scorer's output layer starts at
zero. At step two, both the scorer hidden layer and Delta coefficients had
finite nonzero gradients. This confirms the initially dormant paths become
trainable after the first edge update.

Training seeds use the repository's Python/NumPy/CPU/CUDA `set_seed` helper.
Added module construction is isolated with `torch.random.fork_rng` so the
classifier remains paired. Dropout is 0.2. Expert, edge-score chunk, and
node-router chunk activation checkpointing uses PyTorch's non-reentrant API
with default RNG preservation. Training-time extended Train/Validation
metrics reuse cached eval embeddings; they do not add stochastic model
forwards. CUDA `index_add`/scatter accumulation can have backend-dependent
floating-point order, but the evidence does not attribute all possible run
variation to CUDA.

`evaluate_test=false`; no Test metric or Test-indexed label read is requested
by the V2 runner. The full machine-readable preflight record and unchanged
historical artifacts are preserved in this research branch.

## Known comparison boundary

V1/V2 have matching parameter names, shapes, and initialization, but the V1
edge scorer is inactive and has no task gradients. V0 has no added context or
node-router parameters. V1−V0 measures their complete bundle; V2−V1 measures
the edge path while activating scorer parameters. Neither contrast is a pure
capacity-matched causal effect. Three seeds support descriptive paired
comparisons, not significance claims.
