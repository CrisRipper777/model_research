# Unlabeled graph and neighborhood-attribute audit

Physical and semantic structure was audited from node features, graph topology, and V7B cache provenance. KNN construction and random rewiring did not load labels. The separate homophily table uses Train labels only for descriptive counts; it does not influence graph construction or scoring.

## Graph structure

| Dataset | Relation | Nodes | Undirected edges | Mean degree | Median degree | Isolated % | Components | Largest-component % | Physical edge overlap % | Text/Visual overlap % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | physical | 16672 | 80401 | 9.645 | 5.0 | 0.00 | 692 | 88.66 | 100.00 | 0.00 |
| Movies | text_knn | 16672 | 110714 | 13.281 | 11.0 | 0.00 | 1 | 100.00 | 6.15 | 6.32 |
| Movies | visual_knn | 16672 | 102161 | 12.255 | 10.0 | 0.00 | 2 | 99.92 | 14.51 | 6.85 |
| Grocery | physical | 17074 | 71131 | 8.332 | 4.0 | 0.00 | 395 | 94.18 | 100.00 | 0.00 |
| Grocery | text_knn | 17074 | 109198 | 12.791 | 11.0 | 0.00 | 2 | 99.71 | 6.14 | 7.52 |
| Grocery | visual_knn | 17074 | 104566 | 12.249 | 11.0 | 0.00 | 2 | 99.90 | 8.49 | 7.85 |
| ele-fashion | physical | 97766 | 199586 | 4.083 | 2.0 | 0.00 | 3279 | 88.69 | 100.00 | 0.00 |
| ele-fashion | text_knn | 97766 | 582723 | 11.921 | 10.0 | 0.00 | 86 | 98.00 | 5.78 | 15.34 |
| ele-fashion | visual_knn | 97766 | 608849 | 12.455 | 11.0 | 0.00 | 56 | 99.20 | 3.05 | 14.68 |

## KNN source proposals

Source flags are reported over the deduplicated undirected candidate union before modality-specific propagation. The two view counts include `Both` edges, so an edge supported by both modalities remains in both KNN views.

| Dataset | Text-only | Visual-only | Both | Candidate union | Text proposals (%) | Visual proposals (%) | Both (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grocery | 100991 | 96359 | 8207 | 205557 | 53.12 | 50.87 | 3.99 |
| Movies | 103720 | 95167 | 6994 | 205881 | 53.78 | 49.62 | 3.40 |
| ele-fashion | 493351 | 519477 | 89372 | 1102200 | 52.87 | 55.24 | 8.11 |

## Degree-matched random semantic controls

Each view is rewired separately and keeps the exact degree sequence, active node set, and edge count. The following structural quantities were recorded to expose properties that degree matching does not control.

| Dataset | View | Edges | Exact degree match | Real components | Random components | Source/null Jaccard | Real physical overlap | Random physical overlap |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Grocery | text | 109198 | True | 2 | 1 | 0.00174 | 6707 | 75 |
| Grocery | visual | 104566 | True | 2 | 1 | 0.00181 | 8880 | 77 |
| Movies | text | 110714 | True | 1 | 1 | 0.00186 | 6811 | 95 |
| Movies | visual | 102161 | True | 2 | 1 | 0.00181 | 14823 | 105 |
| ele-fashion | text | 582723 | True | 86 | 1 | 0.00133 | 33701 | 101 |
| ele-fashion | visual | 608849 | True | 56 | 1 | 0.00137 | 18595 | 81 |

## Neighbor feature similarity and dispersion

`graph_structure_stats.csv` reports Text and Visual edge-cosine distributions for each relation, node-to-neighbor-mean cosine distributions, per-dimension neighborhood variance, and edge cosine means for nodes in the low/high physical-degree quartiles. Similarities use the same mean-centered, row-L2-normalized feature preparation as the V7B KNN cache. These describe feature geometry, not predictive utility.

## Train-only label relation statistics

`train_only_label_relation_stats.csv` records Train–Train edge count, endpoint coverage, observed unordered class transitions, edge homophily, and the Train class-marginal random-agreement reference. Validation and Test labels are not used. Homophily is descriptive structure-label alignment and is not a prediction-gain estimate.

## KNN provenance and random semantic controls

Random Text and Visual views are rewired independently by valid degree-preserving double-edge swaps; they preserve edge count and the exact per-node degree sequence, have no loops or duplicate edges, and are fixed across all training seeds. The older V7B union random graph is not used as either modality-specific null.

A single random topology per dataset and source is one null sample, not a distribution over random graphs. Degree matching does not guarantee matching component count, clustering, or overlap with the physical graph.
