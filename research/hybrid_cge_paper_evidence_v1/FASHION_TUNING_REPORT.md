# ele-fashion finite optimization study

F0 is the historical H2-Frozen four-hop reference. F1 (4-hop, Dropout 0.30, private auxiliary CE 0.1), F2 (4-hop, Train-only class-balanced main CE), and F4 (3-hop, Train-only class-balanced main CE) each have three planned seeds. F3 aliases the three-hop original-configuration cells and is not retrained.

For class-balanced candidates, each class weight is derived only from `train_idx` counts as normalized inverse square-root frequency and clipped to `[0.5,2.0]`. Only the main Train CE uses weights. Private auxiliary Text/Visual CE remains unweighted; all Train/Validation reporting CE is ordinary unweighted CE.

Results, checkpoint metric differences, minority-class metrics (including Classes 1 and 11), and Accuracy/Macro-F1 tradeoffs will be filled from completed Validation-only runs.
