from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.graph_ops import normalized_adjacency_operator


VARIANTS = ("G0", "G0-FT", "G1", "G2", "G3", "G4")
INTERVENTIONS = ("normal", "preserve", "uniform_route", "shared_off", "private_off")


class _ReLUDropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
        output = F.relu(inputs)
        if training and probability > 0:
            output = F.dropout(output, p=probability, training=True, inplace=True)
        ctx.probability = probability
        ctx.training = training
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (output,) = ctx.saved_tensors
        grad_input = grad_output * output.gt(0).to(grad_output.dtype)
        if ctx.training and ctx.probability > 0:
            grad_input = grad_input / (1.0 - ctx.probability)
        return grad_input, None, None


class ReLUDropout(nn.Module):
    """ReLU followed by dropout, with an activation-saving equivalent backward."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _ReLUDropoutFunction.apply(inputs, self.probability, self.training)


def clip_unit(value: torch.Tensor) -> torch.Tensor:
    """Clip each final-dimension vector to the unit ball without amplifying it."""
    norm = value.norm(dim=-1, keepdim=True)
    return value / norm.clamp_min(1.0)


def dct_orthogonal_templates(count: int, width: int, *, dtype=torch.float32) -> torch.Tensor:
    """Return the first ``count`` orthonormal DCT-II rows in ``width`` dimensions."""
    if count < 1 or width < count:
        raise ValueError(f"require 1 <= count <= width, got {count}, {width}")
    position = torch.arange(width, dtype=dtype)
    rows = []
    for order in range(count):
        scale = math.sqrt(1.0 / width) if order == 0 else math.sqrt(2.0 / width)
        rows.append(scale * torch.cos(math.pi * (position + 0.5) * order / width))
    return torch.stack(rows)


class ResponseAtomBank(nn.Module):
    """Reusable signed structural-order correction profiles, never latent vectors."""

    def __init__(self, num_atoms: int, num_structural_orders: int = 3) -> None:
        super().__init__()
        initial = dct_orthogonal_templates(num_atoms, num_structural_orders)
        self.theta = nn.Parameter(initial)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.theta, p=2, dim=-1, eps=1e-12)


class Router(nn.Module):
    def __init__(self, input_dim: int, num_routes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            ReLUDropout(dropout),
            nn.Identity(),  # keep the established final Linear state-dict index
            nn.Linear(64, num_routes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).softmax(dim=-1)


class Model(nn.Module):
    """R³-MAG v1 global prior with reusable routed structural responses."""

    def __init__(self, cfg, data_info: dict[str, Any]) -> None:
        super().__init__()
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        self.num_classes = int(data_info["num_classes"])
        if self.text_dim < 1 or self.visual_dim < 1 or self.num_classes < 1:
            raise ValueError("R3-MAG requires positive text_dim, visual_dim, and num_classes")
        self.hidden_dim = 128
        self.max_order = 3
        self.router_dim = 64
        self.out_dim = self.hidden_dim

        self.text_projector = self._projector(self.text_dim)
        self.visual_projector = self._projector(self.visual_dim)
        gamma = torch.tensor([0.2 * (0.8**k) for k in range(4)], dtype=torch.float32)
        self.gamma_global_text = nn.Parameter(gamma.clone())
        self.gamma_global_visual = nn.Parameter(gamma.clone())
        self.text_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            ReLUDropout(0.2),
        )
        self.unimodal_head_text = nn.Linear(self.hidden_dim, self.num_classes)
        self.unimodal_head_visual = nn.Linear(self.hidden_dim, self.num_classes)

        self.router_state_text = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.router_dim),
            nn.LayerNorm(self.router_dim),
            nn.ReLU(),
        )
        self.router_state_visual = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.router_dim),
            nn.LayerNorm(self.router_dim),
            nn.ReLU(),
        )
        task_dim = self.num_classes + 3  # detached probabilities, entropy, margin, JS
        self.flat_bank = ResponseAtomBank(3, 3)
        self.flat_router_text = Router(self.router_dim + task_dim, 3)
        self.flat_router_visual = Router(self.router_dim + task_dim, 3)

        self.shared_bank = ResponseAtomBank(3, 3)
        self.private_bank_text = ResponseAtomBank(2, 3)
        self.private_bank_visual = ResponseAtomBank(2, 3)
        self.shared_router = Router(4 * self.router_dim + 2 * self.num_classes + 1, 3)
        self.private_router_text = Router(self.router_dim + task_dim, 2)
        self.private_router_visual = Router(self.router_dim + task_dim, 2)

        self.rho_global_text = nn.Parameter(torch.tensor(-2.0))
        self.rho_global_visual = nn.Parameter(torch.tensor(-2.0))
        reliability_dim = self.router_dim + 6
        self.reliability_gate = nn.Sequential(
            nn.Linear(reliability_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(self.reliability_gate[-1].bias, -2.0)

        self.variant = "G0"

    @staticmethod
    def _projector(input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            ReLUDropout(0.2),
        )

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.long:
            x = x.float()
        expected = self.text_dim + self.visual_dim
        if x.ndim != 2 or x.size(-1) != expected:
            raise ValueError(
                f"expected concatenated [text, visual] features of width {expected}, got {tuple(x.shape)}"
            )
        return x[:, : self.text_dim], x[:, self.text_dim : expected]

    @staticmethod
    def propagate_states(h0: torch.Tensor, operator: torch.Tensor, max_order: int = 3):
        states = [h0]
        for _ in range(max_order):
            states.append(torch.sparse.mm(operator, states[-1]))
        return states

    @staticmethod
    def _combine(states: list[torch.Tensor], gamma: torch.Tensor) -> torch.Tensor:
        return sum(weight * state for weight, state in zip(gamma, states, strict=True))

    @staticmethod
    def task_context_from_logits(
        logits_text: torch.Tensor, logits_visual: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Build detached, label-free task statistics for routing."""
        p_text = logits_text.detach().softmax(dim=-1)
        p_visual = logits_visual.detach().softmax(dim=-1)
        eps = 1e-12
        entropy_text = -(p_text * p_text.clamp_min(eps).log()).sum(-1, keepdim=True)
        entropy_visual = -(p_visual * p_visual.clamp_min(eps).log()).sum(-1, keepdim=True)
        margin_text = Model._top1_margin(p_text)
        margin_visual = Model._top1_margin(p_visual)
        midpoint = (0.5 * (p_text + p_visual)).clamp_min(eps)
        js = 0.5 * (
            (p_text * (p_text.clamp_min(eps).log() - midpoint.log())).sum(-1, keepdim=True)
            + (p_visual * (p_visual.clamp_min(eps).log() - midpoint.log())).sum(-1, keepdim=True)
        )
        return {
            "p_text": p_text.detach(),
            "p_visual": p_visual.detach(),
            "entropy_text": entropy_text.detach(),
            "entropy_visual": entropy_visual.detach(),
            "margin_text": margin_text.detach(),
            "margin_visual": margin_visual.detach(),
            "js": js.clamp_min(0.0).detach(),
        }

    @staticmethod
    def _top1_margin(probability: torch.Tensor) -> torch.Tensor:
        top = probability.topk(k=min(2, probability.size(-1)), dim=-1).values
        if top.size(-1) == 1:
            return top[:, :1]
        return (top[:, :1] - top[:, 1:2])

    @staticmethod
    def _route_entropy(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        normalized = entropy / math.log(max(weights.size(-1), 2))
        return entropy.squeeze(-1), normalized.clamp(0.0, 1.0)

    @staticmethod
    def _compose_atoms(weights: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        return weights @ atoms

    @staticmethod
    def _prepend_zero(structural_delta: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [structural_delta.new_zeros((structural_delta.size(0), 1)), structural_delta], dim=-1
        )

    def set_variant(self, variant: str) -> None:
        variant = str(variant)
        if variant not in VARIANTS:
            raise ValueError(f"unknown R3-MAG variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant

    def base_named_parameters(self) -> dict[str, nn.Parameter]:
        prefixes = (
            "text_projector.", "visual_projector.", "text_norm.", "visual_norm.", "fusion.",
            "unimodal_head_text.", "unimodal_head_visual.",
        )
        names = {
            name: parameter
            for name, parameter in self.named_parameters()
            if name.startswith(prefixes) or name in {"gamma_global_text", "gamma_global_visual"}
        }
        return names

    def new_named_parameters(self, variant: str | None = None) -> dict[str, nn.Parameter]:
        base = self.base_named_parameters()
        candidates = {name: p for name, p in self.named_parameters() if name not in base}
        if variant is None:
            return candidates
        if variant == "G1":
            prefixes = ("router_state_", "flat_bank.", "flat_router_", "rho_global_")
        elif variant == "G2":
            prefixes = (
                "router_state_", "shared_bank.", "private_bank_", "shared_router.",
                "private_router_", "rho_global_",
            )
        elif variant in {"G3", "G4"}:
            prefixes = (
                "router_state_", "shared_bank.", "private_bank_", "shared_router.",
                "private_router_", "reliability_gate.",
            )
        elif variant in {"G0", "G0-FT"}:
            return {}
        else:
            raise ValueError(f"unknown R3-MAG variant {variant!r}")
        return {name: p for name, p in candidates.items() if name.startswith(prefixes)}

    def optimizer_parameter_groups(self, variant: str | None = None) -> dict[str, dict[str, nn.Parameter]]:
        return {
            "base": self.base_named_parameters(),
            "new": self.new_named_parameters(variant),
        }

    def forward_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        operator: torch.Tensor | None = None,
        variant: str | None = None,
        intervention: str = "normal",
    ) -> dict[str, Any]:
        chosen_variant = self.variant if variant is None else str(variant)
        if chosen_variant not in VARIANTS:
            raise ValueError(f"unknown R3-MAG variant {chosen_variant!r}")
        if intervention not in INTERVENTIONS:
            raise ValueError(f"unknown intervention {intervention!r}")
        if chosen_variant in {"G0", "G0-FT", "G1"} and intervention in {"shared_off", "private_off"}:
            raise ValueError(f"{intervention} is only defined for G2/G3/G4")
        x_text, x_visual = self._split_features(x)
        if operator is None:
            operator = normalized_adjacency_operator(
                edge_index, x.size(0), dtype=x.dtype, device=x.device
            )
        h0_text = self.text_projector(x_text)
        h0_visual = self.visual_projector(x_visual)
        states_text = self.propagate_states(h0_text, operator, self.max_order)
        states_visual = self.propagate_states(h0_visual, operator, self.max_order)
        global_response_text = self._combine(states_text, self.gamma_global_text)
        global_response_visual = self._combine(states_visual, self.gamma_global_visual)
        z_global_text = self.text_norm(global_response_text)
        z_global_visual = self.visual_norm(global_response_visual)
        logits_text = self.unimodal_head_text(z_global_text)
        logits_visual = self.unimodal_head_visual(z_global_visual)
        task = self.task_context_from_logits(logits_text, logits_visual)
        uses_router = chosen_variant in {"G1", "G2", "G3", "G4"}
        u_text = (
            self.router_state_text(torch.cat([h0_text, global_response_text], dim=-1))
            if uses_router else None
        )
        u_visual = (
            self.router_state_visual(torch.cat([h0_visual, global_response_visual], dim=-1))
            if uses_router else None
        )
        count = int(x.size(0))
        zero = x.new_zeros((count,))
        aux: dict[str, Any] = {
            "states_text": states_text,
            "states_visual": states_visual,
            "global_response_text": global_response_text,
            "global_response_visual": global_response_visual,
            "z_global_text": z_global_text,
            "z_global_visual": z_global_visual,
            "logits_text": logits_text,
            "logits_visual": logits_visual,
            "p_text": task["p_text"],
            "p_visual": task["p_visual"],
            "entropy_text": task["entropy_text"].squeeze(-1),
            "entropy_visual": task["entropy_visual"].squeeze(-1),
            "margin_text": task["margin_text"].squeeze(-1),
            "margin_visual": task["margin_visual"].squeeze(-1),
            "js": task["js"].squeeze(-1),
            "u_text": u_text,
            "u_visual": u_visual,
            "variant": chosen_variant,
            "intervention": intervention,
            "route_weights": {},
            "atoms": {},
            "delta_gamma_text": x.new_zeros((count, 4)),
            "delta_gamma_visual": x.new_zeros((count, 4)),
            "rho_text": zero,
            "rho_visual": zero,
            "r_shared": x.new_zeros((count, 3)),
            "r_private_text": x.new_zeros((count, 3)),
            "r_private_visual": x.new_zeros((count, 3)),
            "budget_loss": x.new_zeros(()),
        }

        response_text = global_response_text
        response_visual = global_response_visual
        if chosen_variant in {"G1", "G2", "G3", "G4"}:
            r_shared = x.new_zeros((count, 3))
            r_private_text = x.new_zeros((count, 3))
            r_private_visual = x.new_zeros((count, 3))
            route_weights: dict[str, torch.Tensor] = {}
            atoms: dict[str, torch.Tensor] = {}
            if chosen_variant == "G1":
                atoms_flat = self.flat_bank()
                flat_input_text = torch.cat(
                    [u_text, task["p_text"], task["entropy_text"], task["margin_text"], task["js"]], dim=-1
                )
                flat_input_visual = torch.cat(
                    [u_visual, task["p_visual"], task["entropy_visual"], task["margin_visual"], task["js"]], dim=-1
                )
                w_text = self.flat_router_text(flat_input_text)
                w_visual = self.flat_router_visual(flat_input_visual)
                if intervention == "uniform_route":
                    w_text = torch.full_like(w_text, 1.0 / w_text.size(-1))
                    w_visual = torch.full_like(w_visual, 1.0 / w_visual.size(-1))
                r_private_text = self._compose_atoms(w_text, atoms_flat)
                r_private_visual = self._compose_atoms(w_visual, atoms_flat)
                route_weights.update({"flat_text": w_text, "flat_visual": w_visual})
                atoms["flat"] = atoms_flat
                d_text = clip_unit(r_private_text)
                d_visual = clip_unit(r_private_visual)
                rho_text = self.rho_global_text.sigmoid().expand(count)
                rho_visual = self.rho_global_visual.sigmoid().expand(count)
                if intervention == "preserve":
                    rho_text, rho_visual = torch.zeros_like(rho_text), torch.zeros_like(rho_visual)
                delta_gamma_text = self._prepend_zero(d_text) * rho_text[:, None] * 0.2
                delta_gamma_visual = self._prepend_zero(d_visual) * rho_visual[:, None] * 0.2
            else:
                atoms_shared = self.shared_bank()
                atoms_private_text = self.private_bank_text()
                atoms_private_visual = self.private_bank_visual()
                shared_input = torch.cat(
                    [u_text, u_visual, (u_text - u_visual).abs(), u_text * u_visual,
                     task["p_text"], task["p_visual"], task["js"]], dim=-1
                )
                w_shared = self.shared_router(shared_input)
                private_input_text = torch.cat(
                    [u_text, task["p_text"], task["entropy_text"], task["margin_text"], task["js"]], dim=-1
                )
                private_input_visual = torch.cat(
                    [u_visual, task["p_visual"], task["entropy_visual"], task["margin_visual"], task["js"]], dim=-1
                )
                w_private_text = self.private_router_text(private_input_text)
                w_private_visual = self.private_router_visual(private_input_visual)
                if intervention == "uniform_route":
                    w_shared = torch.full_like(w_shared, 1.0 / w_shared.size(-1))
                    w_private_text = torch.full_like(w_private_text, 1.0 / w_private_text.size(-1))
                    w_private_visual = torch.full_like(w_private_visual, 1.0 / w_private_visual.size(-1))
                # The very same shared route tensor composes both modalities.
                r_shared = self._compose_atoms(w_shared, atoms_shared)
                r_private_text = self._compose_atoms(w_private_text, atoms_private_text)
                r_private_visual = self._compose_atoms(w_private_visual, atoms_private_visual)
                if intervention == "shared_off":
                    r_shared = torch.zeros_like(r_shared)
                if intervention == "private_off":
                    r_private_text = torch.zeros_like(r_private_text)
                    r_private_visual = torch.zeros_like(r_private_visual)
                route_weights.update({
                    "shared_text": w_shared,
                    "shared_visual": w_shared,
                    "private_text": w_private_text,
                    "private_visual": w_private_visual,
                })
                atoms.update({
                    "shared": atoms_shared,
                    "private_text": atoms_private_text,
                    "private_visual": atoms_private_visual,
                })
                d_text = clip_unit(r_shared + r_private_text)
                d_visual = clip_unit(r_shared + r_private_visual)
                private_entropy_text, private_entropy_text_norm = self._route_entropy(w_private_text)
                private_entropy_visual, private_entropy_visual_norm = self._route_entropy(w_private_visual)
                shared_entropy, shared_entropy_norm = self._route_entropy(w_shared)
                disagreement_text = 1.0 - F.cosine_similarity(r_shared, r_private_text, dim=-1, eps=1e-8)
                disagreement_visual = 1.0 - F.cosine_similarity(r_shared, r_private_visual, dim=-1, eps=1e-8)
                if chosen_variant in {"G3", "G4"}:
                    reliability_text = torch.cat([
                        u_text, shared_entropy_norm, private_entropy_text_norm,
                        disagreement_text[:, None], task["entropy_text"], task["margin_text"], task["js"],
                    ], dim=-1)
                    reliability_visual = torch.cat([
                        u_visual, shared_entropy_norm, private_entropy_visual_norm,
                        disagreement_visual[:, None], task["entropy_visual"], task["margin_visual"], task["js"],
                    ], dim=-1)
                    rho_text = self.reliability_gate(reliability_text).sigmoid().squeeze(-1)
                    rho_visual = self.reliability_gate(reliability_visual).sigmoid().squeeze(-1)
                else:
                    rho_text = self.rho_global_text.sigmoid().expand(count)
                    rho_visual = self.rho_global_visual.sigmoid().expand(count)
                if intervention == "preserve":
                    rho_text, rho_visual = torch.zeros_like(rho_text), torch.zeros_like(rho_visual)
                delta_gamma_text = self._prepend_zero(d_text) * rho_text[:, None] * 0.2
                delta_gamma_visual = self._prepend_zero(d_visual) * rho_visual[:, None] * 0.2
                if chosen_variant == "G4":
                    budget_loss = 0.5 * (rho_text.mean() + rho_visual.mean())
                else:
                    budget_loss = zero.new_zeros(())
                aux.update({
                    "shared_route_entropy": shared_entropy,
                    "shared_route_entropy_normalized": shared_entropy_norm.squeeze(-1),
                    "private_route_entropy_text": private_entropy_text,
                    "private_route_entropy_visual": private_entropy_visual,
                    "correction_disagreement_text": disagreement_text,
                    "correction_disagreement_visual": disagreement_visual,
                    "reliability_features_text": reliability_text if chosen_variant in {"G3", "G4"} else None,
                    "reliability_features_visual": reliability_visual if chosen_variant in {"G3", "G4"} else None,
                    "shared_route_entropy_normalized_private_text": private_entropy_text_norm.squeeze(-1),
                    "shared_route_entropy_normalized_private_visual": private_entropy_visual_norm.squeeze(-1),
                    "budget_loss": budget_loss,
                })
            gamma_text = self.gamma_global_text[None, :] + delta_gamma_text
            gamma_visual = self.gamma_global_visual[None, :] + delta_gamma_visual
            response_text = sum(gamma_text[:, k : k + 1] * states_text[k] for k in range(4))
            response_visual = sum(gamma_visual[:, k : k + 1] * states_visual[k] for k in range(4))
            aux.update({
                "delta_gamma_text": delta_gamma_text,
                "delta_gamma_visual": delta_gamma_visual,
                "adapted_gamma_text": gamma_text,
                "adapted_gamma_visual": gamma_visual,
                "rho_text": rho_text,
                "rho_visual": rho_visual,
                "r_shared": r_shared,
                "r_private_text": r_private_text,
                "r_private_visual": r_private_visual,
                "route_weights": route_weights,
                "atoms": atoms,
            })
        else:
            gamma_text = self.gamma_global_text.expand(count, -1)
            gamma_visual = self.gamma_global_visual.expand(count, -1)
            aux.update({
                "adapted_gamma_text": gamma_text,
                "adapted_gamma_visual": gamma_visual,
            })

        z_text = self.text_norm(response_text)
        z_visual = self.visual_norm(response_visual)
        z = self.fusion(torch.cat([z_text, z_visual], dim=-1))
        aux.update({"z": z, "response_text": response_text, "response_visual": response_visual})
        return aux

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        device: torch.device | str,
        batch_size: int,
    ) -> torch.Tensor:
        """Return exact full-graph embeddings on CPU for the repo inference API."""
        del batch_size  # all three propagation orders depend on the full graph
        x_device = x.to(device)
        edge_device = edge_index.to(device) if edge_index is not None else None
        z = self.forward(x_device, edge_device)[0]
        return z.detach().cpu()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        variant: str | None = None,
        intervention: str = "normal",
    ):
        components = self.forward_components(x, edge_index, variant=variant, intervention=intervention)
        return components["z"], None, None, components["z"].new_zeros(()), components


__all__ = ["Model", "ResponseAtomBank", "clip_unit", "dct_orthogonal_templates"]
