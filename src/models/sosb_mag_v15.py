from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops


_MODALITIES = ("text", "visual")
_VARIANTS = {
    "A0_legacy_lg",
    "A1_rawpoly_shared",
    "A2_sosb_shared",
    "A3_sosb_modality",
}


class _IntrinsicProjector(nn.Module):
    """The modality intrinsic projector used by the matched CSE-MAG screen."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.skip(x) + self.linear2(self.dropout(F.gelu(self.linear1(x))))
        )


class _ResponseExpert(nn.Module):
    """Bias-free Local/Global response transform for the legacy control."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.linear2 = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class Model(nn.Module):
    """SOSB-MAG V1.5 matched structural basis screen.

    SOSB here means an OptBasis-inspired signal-conditioned orthogonal Krylov
    basis. It is not an exact OptBasisGNN reproduction. All four variants keep
    the same protected modality priors and late residual fusion.
    """

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("sosb_mag_v15 requires positive text and visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError("sosb_mag_v15 requires exact [text, visual] feature layout")

        self.variant = str(model_cfg.get("variant", "A2_sosb_shared"))
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.expert_bottleneck = int(model_cfg.get("expert_bottleneck", 64))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.basis_order = int(model_cfg.get("basis_order", 4))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.basis_breakdown_eps = float(model_cfg.get("basis_breakdown_eps", 1.0e-5))
        self.gate_init = float(model_cfg.get("gate_init", -2.0))
        if self.hidden_dim != 256:
            raise ValueError("V1.5 screen fixes hidden_dim=256")
        if self.basis_order != 4:
            raise ValueError("V1.5 screen fixes basis_order=4")
        if self.eps <= 0 or self.basis_breakdown_eps <= 0:
            raise ValueError("eps and basis_breakdown_eps must be positive")

        self.num_layers = self.basis_order
        self.out_dim = self.hidden_dim

        # Instantiate all modules/parameters in a variant-invariant order.
        # Therefore a same-seed classifier created after this model also has
        # identical initialization across variants.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
                "visual": _IntrinsicProjector(self.visual_dim, self.hidden_dim, self.dropout_p),
            }
        )
        self.legacy_experts = nn.ModuleDict(
            {
                "local": _ResponseExpert(
                    self.hidden_dim, self.expert_bottleneck, self.dropout_p
                ),
                "global": _ResponseExpert(
                    self.hidden_dim, self.expert_bottleneck, self.dropout_p
                ),
            }
        )
        self.legacy_theta = nn.Parameter(torch.zeros(2))
        self.legacy_gamma = nn.Parameter(torch.tensor(self.gate_init, dtype=torch.float32))
        self.beta_raw_shared = nn.Parameter(torch.full((4,), 0.5))
        self.gamma_shared = nn.Parameter(torch.tensor(self.gate_init, dtype=torch.float32))
        self.beta_raw_text = nn.Parameter(torch.full((4,), 0.5))
        self.beta_raw_visual = nn.Parameter(torch.full((4,), 0.5))
        self.gamma_text = nn.Parameter(torch.tensor(self.gate_init, dtype=torch.float32))
        self.gamma_visual = nn.Parameter(torch.tensor(self.gate_init, dtype=torch.float32))
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        self._freeze_inactive_variant_parameters()

    @staticmethod
    def _set_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _freeze_inactive_variant_parameters(self) -> None:
        self._set_trainable(self.projectors, True)
        for module in (
            self.fusion_linear1,
            self.fusion_linear2,
            self.fusion_skip,
            self.fusion_norm,
        ):
            self._set_trainable(module, True)
        legacy = self.variant == "A0_legacy_lg"
        raw_or_shared_sosb = self.variant in {"A1_rawpoly_shared", "A2_sosb_shared"}
        modality_sosb = self.variant == "A3_sosb_modality"
        self._set_trainable(self.legacy_experts, legacy)
        self.legacy_theta.requires_grad_(legacy)
        self.legacy_gamma.requires_grad_(legacy)
        self.beta_raw_shared.requires_grad_(raw_or_shared_sosb)
        self.gamma_shared.requires_grad_(raw_or_shared_sosb)
        self.beta_raw_text.requires_grad_(modality_sosb)
        self.beta_raw_visual.requires_grad_(modality_sosb)
        self.gamma_text.requires_grad_(modality_sosb)
        self.gamma_visual.requires_grad_(modality_sosb)

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [nodes, {self.input_dim}], got {tuple(x.shape)}")
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    def _normalized_operator(
        self, edge_index: torch.Tensor | None, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if edge_index is None:
            raise ValueError("sosb_mag_v15 requires edge_index")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        edge_index = edge_index.to(dtype=torch.long)
        input_loops = int((edge_index[0] == edge_index[1]).sum().item())
        edge_index, _ = remove_self_loops(edge_index)
        src, dst = edge_index
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=dtype, device=edge_index.device))
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        norm = inv_sqrt[src] * inv_sqrt[dst]
        return src, dst, norm, degree > 0, input_loops

    @staticmethod
    def _propagate(
        state: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, norm: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(state).index_add(0, dst, state[src] * norm.unsqueeze(-1))

    def _active_rms_norm(self, value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(value)
        if bool(active.any()):
            active_values = value[active]
            rms = torch.sqrt(active_values.square().mean(dim=0) + self.eps)
            out[active] = active_values / rms
        return out

    def _raw_polynomial_basis(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = []
        state = prior
        for _ in range(self.basis_order):
            state = self._propagate(state, src, dst, norm)
            state = state * active.to(state.dtype).unsqueeze(-1)
            states.append(state)
        raw = torch.stack([self._active_rms_norm(item, active) for item in states], dim=0)
        breakdown = torch.zeros(
            (self.basis_order, self.hidden_dim), dtype=torch.bool, device=prior.device
        )
        return raw, breakdown

    def _sosb_basis(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v0 = self._active_rms_norm(prior, active)
        basis = [v0]
        breakdowns = []
        residual_rms_rows = []
        for _order in range(1, self.basis_order + 1):
            residual = self._propagate(basis[-1], src, dst, norm)
            residual = residual * active.to(residual.dtype).unsqueeze(-1)
            # Channel-wise modified Gram-Schmidt over active nodes, with one
            # re-orthogonalization pass for float32 stability.
            if bool(active.any()):
                for _pass in range(2):
                    for previous in basis:
                        coeff = (residual[active] * previous[active]).mean(dim=0)
                        residual = residual - previous * coeff.unsqueeze(0)
            residual = residual * active.to(residual.dtype).unsqueeze(-1)
            # The raw RMS is used for the breakdown decision. Adding eps first
            # would floor the RMS at 1e-4, above the configured 1e-5 threshold.
            raw_rms = (
                torch.sqrt(residual[active].square().mean(dim=0))
                if bool(active.any())
                else torch.zeros(self.hidden_dim, dtype=prior.dtype, device=prior.device)
            )
            breakdown = raw_rms < self.basis_breakdown_eps
            residual_rms_rows.append(raw_rms)
            regularized_rms = (
                torch.sqrt(residual[active].square().mean(dim=0) + self.eps)
                if bool(active.any())
                else torch.ones_like(raw_rms)
            )
            value = residual / regularized_rms.unsqueeze(0)
            value[:, breakdown] = 0.0
            value = value * active.to(value.dtype).unsqueeze(-1)
            basis.append(value)
            breakdowns.append(breakdown)
        stacked = torch.stack(basis, dim=0)
        breakdown = torch.stack(breakdowns, dim=0)
        residual_rms = torch.stack(residual_rms_rows, dim=0)
        # V0 is needed for orthogonalization but excluded from the response.
        return stacked[1:], breakdown, stacked, residual_rms

    def _effective_beta(self, raw: torch.Tensor) -> torch.Tensor:
        return raw / (raw.norm(p=2) + self.eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, norm, active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )
        outputs: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, torch.Tensor]] = {}
        info: dict[str, Any] = {
            "variant": self.variant,
            "active_nodes": active.detach(),
            "input_self_loops_removed": input_self_loops,
            "modalities": {},
        }

        for modality in _MODALITIES:
            prior = priors[modality]
            if self.variant == "A0_legacy_lg":
                local_state = self._propagate(prior, src, dst, norm)
                local_delta = (local_state - prior) * active.to(prior.dtype).unsqueeze(-1)
                global_delta = torch.zeros_like(prior)
                state = prior
                for order in range(1, 4):
                    state = self._propagate(state, src, dst, norm)
                    global_delta = global_delta + ((-1.0) ** order) / math.factorial(order) * state
                local_response = self.legacy_experts["local"](local_delta)
                global_response = self.legacy_experts["global"](global_delta)
                beta = F.softmax(self.legacy_theta, dim=0)
                gate = torch.sigmoid(self.legacy_gamma)
                correction = beta[0] * local_response + beta[1] * global_response
                basis = torch.stack([local_delta, global_delta], dim=0)
                breakdown = torch.zeros((2, self.hidden_dim), dtype=torch.bool, device=x.device)
                response = correction
                scaled = gate * correction
                output = prior + scaled
                details[modality] = {
                    "prior": prior,
                    "local_delta": local_delta,
                    "global_delta": global_delta,
                    "local_response": local_response,
                    "global_response": global_response,
                    "basis": basis,
                    "breakdown": breakdown,
                    "beta": beta,
                    "gate": gate.reshape(()),
                    "response": response,
                    "scaled_correction": scaled,
                    "output": output,
                }
            else:
                if self.variant == "A1_rawpoly_shared":
                    basis, breakdown = self._raw_polynomial_basis(
                        prior, src, dst, norm, active
                    )
                else:
                    basis, breakdown, _basis_with_v0, residual_rms = self._sosb_basis(
                        prior, src, dst, norm, active
                    )
                if self.variant == "A3_sosb_modality":
                    beta_raw = self.beta_raw_text if modality == "text" else self.beta_raw_visual
                    gamma = self.gamma_text if modality == "text" else self.gamma_visual
                else:
                    beta_raw = self.beta_raw_shared
                    gamma = self.gamma_shared
                beta = self._effective_beta(beta_raw)
                gate = torch.sigmoid(gamma)
                response = torch.einsum("k,knd->nd", beta, basis)
                scaled = gate * response
                output = prior + scaled
                details[modality] = {
                    "prior": prior,
                    "basis": basis,
                    "breakdown": breakdown,
                    "beta": beta,
                    "gate": gate.reshape(()),
                    "response": response,
                    "scaled_correction": scaled,
                    "output": output,
                }
                if self.variant.startswith("A2") or self.variant.startswith("A3"):
                    # V0 is retained only as a diagnostic for the full SOSB Gram.
                    details[modality]["basis_with_v0"] = _basis_with_v0
                    details[modality]["residual_rms"] = residual_rms

            outputs[modality] = details[modality]["output"]
            info["modalities"][modality] = {
                "gate": details[modality]["gate"].detach(),
                "beta": details[modality]["beta"].detach(),
                "breakdown_fraction_by_order": details[modality]["breakdown"].float().mean(dim=-1).detach(),
            }

        fused = torch.cat([outputs["text"], outputs["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
        )
        if return_details:
            info["details"] = details
        return z, None, None, z.new_zeros(()), info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), None if edge_index is None else edge_index.to(device))
        return z.detach().cpu()
