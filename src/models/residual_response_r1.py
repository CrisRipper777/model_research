from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from src.models.mature_response_r0 import (
    Model as R0Model,
    NUM_EXPERTS,
    PROMPT_DIM,
    TOP_K,
    incoming_degree,
    remove_self_messages,
    self_anchored_low_high,
)


VARIANTS = ("smooth_base", "residual_generic", "residual_protected")


class Model(R0Model):
    """R0 response prototype added as a zero-initialized residual to Smooth."""

    def __init__(self, cfg: Any, data_info: dict[str, int]):
        variant = str(cfg.model.get("variant", "smooth_base")).strip().lower()
        if variant not in VARIANTS:
            raise ValueError(f"R1 model.variant must be one of {VARIANTS}, got {variant!r}")

        # Let R0 construct every response module in its frozen order. This keeps
        # its projector, response-bank, CrossMoE, and composer initializations.
        r0_cfg = copy.deepcopy(cfg)
        r0_cfg.model.variant = {
            "smooth_base": "smooth_base",
            "residual_generic": "bank_crossmoe_generic",
            "residual_protected": "bank_crossmoe_protected",
        }[variant]
        super().__init__(r0_cfg, data_info)
        self.variant = variant
        self.orth_weight = float(cfg.model.get("orth_weight", 1e-3))
        if self.orth_weight != 1e-3:
            raise ValueError("R1 fixes orth_weight=1e-3")

        # R0's low_transforms are now explicitly the independent bank W_L.
        # W_S is initialized after all R0 modules so their initialization stays
        # exactly aligned with the R0 prototype.
        self.smooth_transforms = nn.ModuleList(
            [nn.Linear(128, 128, bias=False) for _ in range(2)]
        )
        for layer in self.smooth_transforms:
            nn.init.xavier_uniform_(layer.weight)
        self.correction_transforms = nn.ModuleList(
            [nn.Linear(128, 128, bias=False) for _ in range(2)]
        )
        for layer in self.correction_transforms:
            nn.init.zeros_(layer.weight)
        self._r1_topology_cache: dict[Any, Any] = {}

    def _smooth_neighbor_mean(
        self, intrinsic: torch.Tensor, edge_index: torch.Tensor,
        src: torch.Tensor, dst: torch.Tensor, num_nodes: int,
    ) -> torch.Tensor:
        """Stable segmented reduction for reproducible zero-correction identity."""
        key = (edge_index.device.type, edge_index.device.index, edge_index.data_ptr(),
               int(edge_index.size(1)), int(edge_index._version))
        cached = self._r1_topology_cache.get(key)
        if cached is None:
            order = torch.argsort(dst, stable=True)
            sorted_src = src[order]
            sorted_dst = dst[order]
            unique_dst, lengths = torch.unique_consecutive(sorted_dst, return_counts=True)
            counts = [int(value) for value in lengths.detach().cpu().tolist()]
            chunks = []
            segment_begin, edge_begin, edge_count = 0, 0, 0
            for segment, count in enumerate(counts):
                if edge_count and edge_count + count > self.edge_chunk_size:
                    chunks.append((segment_begin, segment, edge_begin, edge_begin + edge_count))
                    segment_begin, edge_begin, edge_count = segment, edge_begin + edge_count, 0
                edge_count += count
            if edge_count:
                chunks.append((segment_begin, len(counts), edge_begin, edge_begin + edge_count))
            cached = (sorted_src, unique_dst, lengths, tuple(chunks))
            self._r1_topology_cache = {key: cached}

        sorted_src, unique_dst, lengths, chunks = cached
        result = intrinsic.new_zeros((num_nodes, intrinsic.size(-1)))
        for segment_begin, segment_end, edge_begin, edge_end in chunks:
            partial = torch.segment_reduce(
                intrinsic[sorted_src[edge_begin:edge_end]],
                reduce="sum",
                lengths=lengths[segment_begin:segment_end],
            )
            result.index_copy_(0, unique_dst[segment_begin:segment_end], partial)
        degree = torch.zeros(num_nodes, dtype=torch.long, device=dst.device)
        if dst.numel():
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.long))
        return result / degree.clamp_min(1).to(intrinsic.dtype).unsqueeze(-1)

    @property
    def bank_low_transforms(self) -> nn.ModuleList:
        """R0 low transforms, retained as the response bank's W_L."""
        return self.low_transforms

    @property
    def bank_high_transforms(self) -> nn.ModuleList:
        """R0 high transforms, retained as the response bank's W_H."""
        return self.high_transforms

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        response_overrides: dict[str, Any] | None = None,
        routing_override: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        correction_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        correction_off: bool = False,
    ):
        if edge_index is None:
            raise ValueError("R1 requires the original physical edge_index")
        x_t, x_v = self.split_modalities(x)
        intrinsic = [self.proj_t(x_t), self.proj_v(x_v)]
        _, src, dst = remove_self_messages(edge_index)
        degree = incoming_degree(dst, int(x.size(0)))

        raw_low, raw_high, neighbor_mean = [], [], []
        low, high, smooth_delta, smooth_state = [], [], [], []
        for modality in range(2):
            lraw, hraw, nmean = self_anchored_low_high(
                intrinsic[modality], src, dst, degree, self.edge_chunk_size
            )
            raw_low.append(lraw)
            raw_high.append(hraw)
            neighbor_mean.append(nmean)
            low.append(self.low_response_norms[modality](
                self.bank_low_transforms[modality](lraw)
            ))
            high.append(self.high_response_norms[modality](
                self.bank_high_transforms[modality](hraw)
            ))
            # A stable sum avoids CUDA atomic-order noise between the baseline
            # and zero-initialized residual variants. It computes the same
            # physical incoming-neighbor mean as M0/N1/R0.
            smooth_mean = self._smooth_neighbor_mean(
                intrinsic[modality], edge_index, src, dst, int(x.size(0))
            )
            neighbor_mean[-1] = smooth_mean
            delta = self.smooth_transforms[modality](smooth_mean)
            smooth_delta.append(delta)
            smooth_state.append(self.smooth_residual_norms[modality](
                intrinsic[modality] + delta
            ))

        override = response_overrides or {}
        if "L" in override:
            low = [override["L"][m].to(device=low[m].device, dtype=low[m].dtype)
                   for m in range(2)]
        if "H" in override:
            high = [override["H"][m].to(device=high[m].device, dtype=high[m].dtype)
                    for m in range(2)]

        orth_loss = x.new_zeros(())
        cross = [intrinsic[0].new_zeros(intrinsic[0].shape),
                 intrinsic[1].new_zeros(intrinsic[1].shape)]
        moe_info: list[dict[str, torch.Tensor]] = []
        attention: list[torch.Tensor | None] = [None, None]
        composer_inputs: list[torch.Tensor | None] = [None, None]
        composition_response: list[torch.Tensor | None] = [None, None]
        pre_ffn: list[torch.Tensor | None] = [None, None]
        branch_structural: list[torch.Tensor | None] = [None, None]
        residual: list[torch.Tensor | None] = [None, None]
        correction = [torch.zeros_like(value) for value in smooth_state]
        structural = list(smooth_state)
        cross_active = self.variant != "smooth_base"

        if cross_active:
            # Keep response-branch dropout from advancing the global stream.
            # At W_C=0, the fusion dropout therefore receives the same mask as
            # Smooth, including while the models are in training mode.
            devices = [x.device] if x.device.type == "cuda" else []
            rng_context = torch.random.fork_rng(devices=devices, enabled=True)
            with rng_context:
                cross, moe_info = self._cross_response(
                    intrinsic, low, high, routing_override, return_diagnostics
                )
                orth_loss = sum(self._prompt_orthogonality(p) for p in self.cross_prompts)
                if "X" in override:
                    cross = [override["X"][m].to(device=cross[m].device,
                                                  dtype=cross[m].dtype)
                             for m in range(2)]

                for modality in range(2):
                    if self.variant == "residual_protected":
                        query = intrinsic[modality].unsqueeze(1)
                        tokens = torch.stack(
                            [low[modality], high[modality], cross[modality]], dim=1
                        )
                        response, weights = self.protected_attention[modality](
                            query, tokens, tokens, need_weights=True,
                            average_attn_weights=False,
                        )
                        response = response.squeeze(1)
                        attention[modality] = weights.squeeze(2)
                        composer_inputs[modality] = torch.cat(
                            [intrinsic[modality], low[modality], high[modality],
                             cross[modality]], dim=-1
                        )
                    else:
                        composer_input = torch.cat(
                            [intrinsic[modality], low[modality], high[modality],
                             cross[modality]], dim=-1
                        )
                        response = self.generic_composers[modality](composer_input)
                        composer_inputs[modality] = composer_input
                    u = self.response_residual_norms[modality](
                        intrinsic[modality] + response
                    )
                    branch = self.final_residual_norms[modality](
                        u + self.post_ffns[modality](u)
                    )
                    composition_response[modality] = response
                    pre_ffn[modality] = u
                    branch_structural[modality] = branch
                    residual[modality] = branch - intrinsic[modality]
                    correction[modality] = self.correction_transforms[modality](
                        residual[modality]
                    )
                    structural[modality] = smooth_state[modality] + correction[modality]

        if correction_off:
            correction = [torch.zeros_like(value) for value in smooth_state]
            structural = list(smooth_state)
        elif correction_override is not None:
            correction = [correction_override[m].to(
                device=smooth_state[m].device, dtype=smooth_state[m].dtype
            ) for m in range(2)]
            structural = [smooth_state[m] + correction[m] for m in range(2)]

        z = self.fusion(torch.cat(
            [intrinsic[0], intrinsic[1], structural[0], structural[1]], dim=-1
        ))
        if not return_diagnostics:
            return z, intrinsic, structural, orth_loss, {}

        aux: dict[str, Any] = {
            "intrinsic": intrinsic,
            "neighbor_mean": neighbor_mean,
            "low_raw": raw_low,
            "high_raw": raw_high,
            "low": low,
            "high": high,
            "cross": cross,
            "smooth_delta": smooth_delta,
            "smooth_state": smooth_state,
            "structural": structural,
            "branch_structural": branch_structural,
            "residual": residual,
            "correction": correction,
            "composition_response": composition_response,
            "composer_inputs": composer_inputs,
            "pre_ffn": pre_ffn,
            "degree": degree,
            "src": src,
            "dst": dst,
            "orth_loss": orth_loss,
            "cross_active": cross_active,
            "moe": moe_info,
            "attention": attention,
        }
        return z, intrinsic, structural, orth_loss, aux

    @staticmethod
    def _prompt_orthogonality(prompts: torch.Tensor) -> torch.Tensor:
        from src.models.mature_response_r0 import prompt_orthogonality_loss
        return prompt_orthogonality_loss(prompts)


def parameter_counts(model: Model) -> dict[str, int]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters()
                    if parameter.requires_grad)
    generic_first = sum(int(parameter.numel()) for module in model.generic_composers
                        for parameter in module.parameters())
    protected_first = sum(int(parameter.numel()) for module in model.protected_attention
                          for parameter in module.parameters())
    correction = sum(int(parameter.numel()) for module in model.correction_transforms
                     for parameter in module.parameters())
    smooth = sum(int(parameter.numel()) for module in model.smooth_transforms
                 for parameter in module.parameters())
    return {
        "model_total": total,
        "model_trainable": trainable,
        "smooth_transforms": smooth,
        "bank_low_transforms": sum(p.numel() for module in model.bank_low_transforms
                                    for p in module.parameters()),
        "bank_high_transforms": sum(p.numel() for module in model.bank_high_transforms
                                     for p in module.parameters()),
        "correction_transforms": correction,
        "generic_first_stage_both_modalities": generic_first,
        "protected_mha_both_modalities": protected_first,
    }
