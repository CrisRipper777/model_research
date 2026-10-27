"""NC-only Early-Cross / Late-Completion ablations for MR-CGE V1.3-A.

The formal encoder remains in :mod:`mrcge_v13`; this subclass changes only the
two stage methods named by ``cfg.model.ablation_kind`` and adds no parameters.
"""

from __future__ import annotations

import torch

from src.models.mrcge_v13 import Model as BaseModel


ABLATION_KINDS = ("none", "no_early_cross", "no_late_completion")


class Model(BaseModel):
    supports_link_prediction = False

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)
        self.ablation_kind = str(cfg.model.get("ablation_kind", "none"))
        if self.ablation_kind not in ABLATION_KINDS:
            raise ValueError(f"ablation_kind must be one of {ABLATION_KINDS}")

    def _early_cross(self, receiver, receiver_id, source_state, projection,
                     edge_index, edge_weight, support, reference):
        if self.ablation_kind != "no_early_cross":
            return super()._early_cross(
                receiver, receiver_id, source_state, projection,
                edge_index, edge_weight, support, reference,
            )
        zero = torch.zeros_like(reference)
        return zero, zero, reference.new_zeros(())

    def _late(self, target, target_id, source, h_source, u_private_source,
              u_collaborative_target, *, collect):
        if self.ablation_kind != "no_late_completion":
            return super()._late(
                target, target_id, source, h_source, u_private_source,
                u_collaborative_target, collect=collect,
            )
        zero = self._diagnostic_scalar(u_collaborative_target.new_zeros(()), collect)
        return u_collaborative_target, {
            "late_delta_rms": zero,
            "late_delta_ratio": zero,
        }
