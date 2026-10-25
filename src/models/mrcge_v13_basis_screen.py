"""MR-CGE V1.3-A with an explicit Physical-basis NoP0 screening switch.

The parent V1.3 model remains frozen. This wrapper replaces only its shared
expert-bank shell while reusing the exact initialized Parameter/module objects.
The new bank applies a relation-specific coefficient mask in both its normal
stack path and the parent's low-memory checkpointed path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.mrcge_v13 import Model as FrozenV13Model
from src.models.mrcge_v13 import SharedExpertBank


SCREEN_VARIANTS = ("joint_view_original", "joint_view_physical_no_p0")


class RelationAwareExpertBank(SharedExpertBank):
    """V1.3 bank shell with an explicit relation-aware effective coefficient API."""

    def __init__(
        self,
        source: SharedExpertBank,
        *,
        physical_no_p0: bool,
        physical_hops: int,
        semantic_hops: int,
        eps: float,
    ) -> None:
        # Re-register the existing parameters and modules directly. This adds
        # no random initialization, preserves state_dict keys/order, and keeps
        # the theta Parameter shared by Physical and Semantic computations.
        nn.Module.__init__(self)
        self.eta = source.eta
        self.theta = source.theta
        self.experts = source.experts
        self.norms = source.norms
        self.physical_no_p0 = bool(physical_no_p0)
        self.physical_hops = int(physical_hops)
        self.semantic_hops = int(semantic_hops)
        self.eps = float(eps)

    def effective_coefficients(self, relation: str, dtype: torch.dtype) -> torch.Tensor:
        """Return signed L1-normalized coefficients for one named relation."""
        if relation == "physical":
            max_order = self.physical_hops
        elif relation == "semantic":
            max_order = self.semantic_hops
        else:
            raise ValueError(f"unsupported graph relation {relation!r}")
        order = torch.arange(4, device=self.theta.device)
        mask = order <= max_order
        if relation == "physical" and self.physical_no_p0:
            mask = mask & (order > 0)
        raw = self.theta * mask.to(self.theta.dtype)[None, :]
        return (raw / (raw.abs().sum(dim=-1, keepdim=True) + self.eps)).to(dtype)

    def coefficients(self, max_order: int, dtype: torch.dtype) -> torch.Tensor:
        """Adapt the frozen bank API's order argument to its explicit relation."""
        matches = []
        if int(max_order) == self.physical_hops:
            matches.append("physical")
        if int(max_order) == self.semantic_hops:
            matches.append("semantic")
        if len(matches) != 1:
            raise ValueError(
                f"max_order={max_order} does not identify exactly one frozen V1.3 relation"
            )
        return self.effective_coefficients(matches[0], dtype)


class Model(FrozenV13Model):
    """Frozen V1.3-A plus an opt-in, Physical-only zero-order exclusion."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        mc = cfg.model
        basis_screen_variant = str(mc.get("basis_screen_variant", "joint_view_original"))
        physical_no_p0 = bool(mc.get("physical_no_p0", False))
        if basis_screen_variant not in SCREEN_VARIANTS:
            raise ValueError(f"basis_screen_variant must be one of {SCREEN_VARIANTS}")
        expected_no_p0 = basis_screen_variant == "joint_view_physical_no_p0"
        if physical_no_p0 != expected_no_p0:
            raise ValueError("basis_screen_variant and explicit physical_no_p0 flag disagree")
        super().__init__(cfg, data_info)
        self.basis_screen_variant = basis_screen_variant
        self.physical_no_p0 = physical_no_p0
        if self.v13_variant != "joint_view":
            raise ValueError("the basis screen is defined only on the V1.3-A joint_view condition")
        if not self.condition.startswith("M_"):
            raise ValueError("the basis screen requires the frozen M_repaired expert bank")
        source = self.filter_bank
        self.filter_bank = RelationAwareExpertBank(
            source,
            physical_no_p0=self.physical_no_p0,
            physical_hops=self.physical_hops,
            semantic_hops=self.semantic_hops,
            eps=self.eps,
        )

    def physical_alpha(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.filter_bank.effective_coefficients(
            "physical", dtype or self.filter_bank.theta.dtype
        )

    def semantic_alpha(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.filter_bank.effective_coefficients(
            "semantic", dtype or self.filter_bank.theta.dtype
        )
