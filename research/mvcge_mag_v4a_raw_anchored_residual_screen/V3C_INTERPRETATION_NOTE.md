# V3C interpretation carried into V4A

V3C demonstrated substantial functional action-space novelty from role-based graph actions. Its performance results also established that the two edge roles cannot be interpreted independently of their integration into the backbone:

1. The Support-only arm was materially below Raw.
2. Adding ordinary-smoothed Discrepant information to Support-only recovered performance: `F2 − F1` was **+0.360 percentage points Accuracy** and **+0.452 points Macro-F1**, with Accuracy positive in 7/9 pairs, Macro-F1 positive in 6/9 pairs, and positive mean Accuracy delta on all three datasets. This supports the view that Discrepant edges contain task information.
3. The signed `H − P_- H` treatment was below ordinary smoothing: `F3 − F2` was **−0.226 points Accuracy** and **−0.471 points Macro-F1**, with Accuracy positive in 0/9 pairs. The current signed treatment is unsupported by these results.
4. The role-decomposed replacement backbone was below Raw. That outcome does not by itself establish that the fixed role assignment is not task-aligned.

V3C replaced the Raw trajectory with role-homogeneous trajectories, omitted mixed-role multi-hop paths such as `P+P-` and `P-P+`, independently RMS-normalized role channels, and used a sigmoid mixture near 0.1. Each is an integration confound for the role signal. V4A therefore preserves the complete Raw path and introduces fixed role contrast only as a zero-initialized bounded residual. It does not change role assignment. The next decision should be based on this integration-controlled screen before considering a learned role estimator.

The role contrast in V4A is only the relative Supportive-minus-Discrepant contribution. It is not called a heterophily or high-pass basis.
