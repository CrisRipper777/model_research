from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from src.models.adaptive_prop_m0 import fixed_degree_mean
from src.models.structured_executor_e0 import (
    FUNCTION_NAMES,
    Model as E0Model,
    _rms,
    pi_global_mean,
    pi_modality_tied,
    pi_target_mean,
)


E01_VARIANTS = (
    "premix_edge_control",
    "keep_static",
    "keep_target",
    "keep_edge",
)
ROUTE_VARIANT = {
    "premix_edge_control": "edge_mix",
    "keep_static": "static_mix",
    "keep_target": "target_mix",
    "keep_edge": "edge_mix",
}
KEEP_VARIANTS = {"keep_static", "keep_target", "keep_edge"}


class Model(E0Model):
    """E0 function bank with matched-capacity node-level provenance composers."""

    def __init__(self, cfg: Any, data_info: dict[str, int]):
        experiment_variant = str(cfg.model.get("variant", "keep_edge")).strip().lower()
        if experiment_variant not in E01_VARIANTS:
            raise ValueError(f"model.variant must be one of {E01_VARIANTS}, got {experiment_variant!r}")

        # Construct every frozen E0 module in the exact E0 order, then append
        # the only new capacity. This preserves the common-module RNG stream.
        base_cfg = copy.deepcopy(cfg)
        base_cfg.model.variant = ROUTE_VARIANT[experiment_variant]
        super().__init__(base_cfg, data_info)
        self.e01_variant = experiment_variant

        self.provenance_composers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(2)
        ])
        for composer in self.provenance_composers:
            nn.init.xavier_uniform_(composer[0].weight)
            nn.init.zeros_(composer[0].bias)
            nn.init.normal_(composer[2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(composer[2].bias)

    @staticmethod
    def _composer_input(
        variant: str,
        contexts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        c_mix: torch.Tensor,
        mode: str = "normal",
    ) -> torch.Tensor:
        c_s, c_r, c_x = contexts
        if mode == "collapse" or variant == "premix_edge_control":
            c_mean = c_mix / 3.0
            return torch.cat((c_mean, c_mean, c_mean), dim=-1)
        if mode == "normal":
            return torch.cat((c_s, c_r, c_x), dim=-1)
        if mode.startswith("permute:"):
            order = tuple(int(value) for value in mode.split(":", 1)[1].split(","))
            if sorted(order) != [0, 1, 2]:
                raise ValueError(f"invalid provenance permutation {order}")
            channels = (c_s, c_r, c_x)
            return torch.cat(tuple(channels[index] for index in order), dim=-1)
        raise ValueError(f"unknown composer input mode: {mode}")

    def aggregate_contexts(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        control_overrides: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_diagnostics: bool = False,
        composer_input_mode: str = "normal",
        composer_off: bool = False,
    ):
        if edge_index is None:
            raise ValueError("provenance_executor_e01 requires physical edge_index")
        h0, nonself, src, dst, degree, deg_z, p, relation_contexts = self._prepare(x, edge_index)
        num_nodes, num_edges = int(x.size(0)), int(src.numel())
        pooled = None if self.variant == "edge_mix" else self._route_pool(
            p, relation_contexts, deg_z, src, dst, degree, num_nodes)
        sums = [h0[0].new_zeros((num_nodes, self.hidden_dim)) for _ in range(6)]
        diagnostic_pi = [[], []]
        scale_keys = (
            "smooth_rms", "raw_relational_rms", "raw_cross_modal_rms",
            "relational_calibration_scale", "cross_modal_calibration_scale",
            "calibrated_relational_rms_ratio", "calibrated_cross_modal_rms_ratio",
        )
        diagnostic_scales = {key: [[], []] for key in scale_keys}

        for begin in range(0, num_edges, self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, num_edges)
            cs, cd = src[begin:end], dst[begin:end]
            _, _, _, states = self._functional_states(
                p,
                [relation_contexts[0][begin:end], relation_contexts[1][begin:end]],
                deg_z,
                cs,
                cd,
            )
            route_inputs = []
            for modality in range(2):
                if self.variant == "static_mix":
                    route_inputs.append(pooled[modality].expand(end - begin, -1))
                elif self.variant == "target_mix":
                    route_inputs.append(pooled[modality][cd])
                else:
                    route_inputs.append(states[modality])
            pi = [torch.softmax(self.router(value), dim=-1) for value in route_inputs]
            pi = [self._overridden(pi[m], control_overrides, "pi", m, begin, end)
                  for m in range(2)]
            smooth, relational, cross_modal, scales = self._expert_outputs(h0, cs, cd)
            for modality in range(2):
                channel_messages = (
                    pi[modality][:, 1:2] * smooth[modality],
                    pi[modality][:, 2:3] * relational[modality],
                    pi[modality][:, 3:4] * cross_modal[modality],
                )
                base_index = modality * 3
                for channel_index, message in enumerate(channel_messages):
                    sums[base_index + channel_index] = sums[base_index + channel_index].index_add(
                        0, cd, message
                    )
                if return_diagnostics:
                    diagnostic_pi[modality].append(pi[modality].detach().float().cpu())
                    scale_values = {
                        "smooth_rms": scales["reference_rms"][modality],
                        "raw_relational_rms": scales["raw_rms_relational"][modality],
                        "raw_cross_modal_rms": scales["raw_rms_cross_modal"][modality],
                        "relational_calibration_scale": scales["scale_relational"][modality],
                        "cross_modal_calibration_scale": scales["scale_cross_modal"][modality],
                        "calibrated_relational_rms_ratio": _rms(relational[modality], self.eps)
                        / _rms(smooth[modality], self.eps),
                        "calibrated_cross_modal_rms_ratio": _rms(cross_modal[modality], self.eps)
                        / _rms(smooth[modality], self.eps),
                    }
                    for name, value in scale_values.items():
                        diagnostic_scales[name][modality].append(value.detach().float().cpu())

        contexts_by_modality = []
        composer_inputs = []
        composer_outputs = []
        deltas = []
        h_tilde = []
        for modality in range(2):
            base_index = modality * 3
            channels = tuple(fixed_degree_mean(sums[base_index + i], degree) for i in range(3))
            c_mix = channels[0] + channels[1] + channels[2]
            composer_input = self._composer_input(
                self.e01_variant, channels, c_mix, mode=composer_input_mode
            )
            correction = self.provenance_composers[modality](composer_input)
            if composer_off:
                correction = torch.zeros_like(c_mix)
            contexts_by_modality.append((channels, c_mix))
            composer_inputs.append(composer_input)
            composer_outputs.append(correction)
            deltas.append(c_mix + correction)
            h_tilde.append(self.residual_norms[modality](h0[modality] + deltas[-1]))

        fused = self.fusion(torch.cat([h0[0], h0[1], h_tilde[0], h_tilde[1]], dim=-1))
        aux: dict[str, Any] = {
            "num_nonself_messages": num_edges,
            "degree": degree.detach(),
        }
        if return_diagnostics:
            aux["controls"] = {"pi": tuple(
                torch.cat(diagnostic_pi[m], dim=0) if diagnostic_pi[m]
                else torch.empty((0, len(FUNCTION_NAMES)))
                for m in range(2)
            )}
            aux["expert_scales"] = {
                key: tuple(torch.cat(diagnostic_scales[key][m], dim=0)
                           if diagnostic_scales[key][m] else torch.empty((0, 1))
                           for m in range(2))
                for key in scale_keys
            }
            aux["edge_index_nonself"] = nonself.detach().cpu()
            aux["channel_contexts"] = tuple(
                tuple(value.detach().float().cpu() for value in contexts_by_modality[m][0])
                for m in range(2)
            )
            aux["c_mix"] = tuple(value.detach().float().cpu() for _, value in contexts_by_modality)
            aux["composer_input"] = tuple(value.detach().float().cpu() for value in composer_inputs)
            aux["composer_output"] = tuple(value.detach().float().cpu() for value in composer_outputs)
            aux["delta"] = tuple(value.detach().float().cpu() for value in deltas)
        return fused, None, None, fused.new_zeros(()), aux

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        control_overrides: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_diagnostics: bool = False,
        composer_input_mode: str = "normal",
        composer_off: bool = False,
    ):
        return self.aggregate_contexts(
            x,
            edge_index,
            control_overrides=control_overrides,
            return_diagnostics=return_diagnostics,
            composer_input_mode=composer_input_mode,
            composer_off=composer_off,
        )

    @torch.no_grad()
    def inference(self, x: torch.Tensor, edge_index: torch.Tensor,
                  device: torch.device | str | None = None, batch_size: int = 65536) -> torch.Tensor:
        del batch_size
        self.eval()
        device = torch.device(device) if device is not None else next(self.parameters()).device
        z, _, _, _, _ = self(x.to(device), edge_index.to(device))
        return z.detach().cpu()


def provenance_interventions(pi: tuple[torch.Tensor, torch.Tensor], dst: torch.Tensor, num_nodes: int):
    """Return E0-compatible routing transforms plus fixed intervention seeds."""
    from src.models.adaptive_prop_m0 import neighborhood_shuffle_indices

    dst = dst.detach().cpu().long()
    rows: list[tuple[str, int | None, tuple[torch.Tensor, torch.Tensor]]] = []
    for repeat in range(1001, 1006):
        moved = tuple(
            pi[modality][neighborhood_shuffle_indices(dst, repeat + modality * 10000)].clone()
            for modality in range(2)
        )
        rows.append(("pi_shuffle_within_target", repeat, moved))
    rows.append(("pi_target_mean", None, tuple(pi_target_mean(pi[m], dst, num_nodes) for m in range(2))))
    rows.append(("pi_global_mean", None, tuple(pi_global_mean(pi[m]) for m in range(2))))
    rows.append(("modality_tied_pi", None, pi_modality_tied(pi[0], pi[1])))
    smooth = []
    for modality in range(2):
        value = torch.zeros_like(pi[modality])
        value[:, 1] = 1
        smooth.append(value)
    rows.append(("smooth_only", None, tuple(smooth)))
    return rows


def relative_context_mass(channels: tuple[torch.Tensor, torch.Tensor, torch.Tensor], eps: float):
    rms_values = [(_rms(value, eps).squeeze(-1)) for value in channels]
    total = sum(rms_values)
    return tuple(value / (total + eps) for value in rms_values)
