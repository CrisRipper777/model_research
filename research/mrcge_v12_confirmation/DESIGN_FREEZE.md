# Design Freeze

- Base: `712bf477085b8e9022a746330e3ae48e12fa0841` (`exp/mrcge_v12_anchor_performance_screen`).
- Confirmation branch: `exp/mrcge_v12_multiseed_confirmation`.
- Training source SHA at CPU preflight: `74214858914065898920f37ecc8cf592e8ebd5ef`.
- Model SHA256: `d9a7632f725a7c742dc083fe7737bc5c7384e9fa3bc91ffa64ead01e4a5d4ca6`.
- Trainer SHA256 (`src/tasks/nc.py`): `362350f07bb2d076b0f4f791872fbf0105ba695681bc15b3f2a09353bc6aacfd`.
- Inference SHA256: `f6e0df653a03a6ab5be98016fdc495fcff7704f42a73bdd546c36f7fc63be0aa`.
- Model/task configuration and dataset loader/split hashes are recorded in `data/preflight.json`.
- The frozen model, trainer, inference, model/task config, data loader, graph construction, semantic cache and split implementation are unchanged from the base commit.
- Formal matrix: 16 M cells (four datasets × two M conditions × seeds 43/44), plus ele-fashion S/F seed 42.
- No Response Strength, architectural edits, tuning, Test evaluation, LP, or seed-42 M retraining.
