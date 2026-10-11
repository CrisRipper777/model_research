# PSCE-MAG V7B design

## Questions and controlled variants

V7B asks whether feature KNN relations carry useful structure beyond a degree-matched random relation, whether learned weights improve over fixed weights, whether relative score centering improves relation learning, and whether Text and Visual benefit from modality-specific weights over a shared topology.

| Arm | Topology | Relation weight | Interpretation |
|---|---|---|---|
| B0 `b0_v7a_original` | V7A Text/Visual KNN union | V7A shared sigmoid scorer | Retrained V7A A2 control |
| B1 `b1_fixed_knn` | Same true KNN union | Fixed `a=0.8` | Learning-vs-fixed control |
| B2 `b2_random_fixed` | Degree-preserving rewired union | Fixed `a=0.8` | True-vs-random relation control |
| B3 `b3_relative_shared` | Same true KNN union | Shared relative score from two-head scorer | Removes global score translation freedom |
| B4 `b4_relative_modality` | Same true KNN union | Separate relative Text/Visual scores | Modality-specific edge reliability |

No outcomes are assumed. Candidate topology and edge weights are diagnostics rather than causal explanations.

## Candidate provenance

Text and Visual features are independently mean-centered and L2-normalized, then queried with the frozen V7A FAISS rule at K=8. Their directed query proposals are counted before union/symmetrization. For each canonical pair, source flags record Text and Visual proposal counts in `{0,1,2}`; flags are shared by the two propagation directions. The V7A candidate fingerprint is computed from the unchanged edge-index, unique-edge, and degree tensors. V7B adds feature, source, FAISS, physical-graph, and construction provenance in a separate complete fingerprint.

B2 uses canonical undirected double-edge swaps, three successful swaps per starting edge. Rejected proposals include loops, duplicate edges, and duplicate endpoints. Each seed-specific null topology retains every node's exact degree and the edge count; the Physical graph remains unchanged.

## Frozen model body and relation scorer

B0 directly inherits the V7A A2 model and forward path. B1/B2 keep that path but return a literal 0.8 edge-weight vector without using or training the scorer. B3/B4 reuse V7A encoders, physical propagation, shared experts, physical and semantic routers, semantic strength, balance loss, fusion, active-RMS handling, checkpointing, and chunked sparse propagation.

The relation scorer projects each modality's intrinsic state to 32 dimensions. For every canonical edge it concatenates `[u_i ⊙ u_j, |u_i-u_j|]` for Text and Visual plus `p_text/2` and `p_visual/2`, giving 130 inputs. A shared `130→64→2` MLP emits Text/Visual scores; the final projection has no bias and its two small-random rows start equal.

For each score channel, endpoint means are computed by summing the undirected score into both endpoints and dividing by the fixed candidate degree. The local centered score is `delta_ij=s_ij−(mu_i+mu_j)/2`. The scalar reliability is `a_ij=0.8(1+0.75 tanh(delta_ij))`, then normalized by the fixed `sqrt(d_i d_j)`. It is symmetric, differentiable, bounded in `(0.2,1.4)`, and invariant to a global score translation. It does not conserve each node's total semantic message mass.

B3 averages the two outputs before this centering, so both modal views receive one shared edge operator. The output-head difference direction has no identifiable effect in B3. B4 centers each output separately and shares one candidate topology and edge index.

Semantic order remains `S1=P_sem@H0`, followed by `S2=P_phy@S1`. Existing one-sided semantic RMS calibration, conditional Router, shared experts, expert residual scale, and final fusion are unchanged.

## Training and evaluation

All variants use `unified_full_graph_nc_v1` on Movies, Grocery, and ele-fashion with seeds 42/43/44, AdamW, learning rate `0.001`, weight decay `0.0001`, a 300-epoch cap, patience 30, the frozen NC splits, Validation Accuracy checkpoint selection, and Validation Macro-F1 as a secondary measure. `task.evaluate_test=false` is explicit. Link prediction is unsupported because semantic-cache IDs are global while LP sampled-subgraph IDs are local.

The campaign has 45 formal runs, all tied to one freeze commit, candidate/source fingerprints, unique dataset-seed-variant keys, checkpoints and resumable per-run records. GPU work targets `cuda:1`; no other process is stopped.
