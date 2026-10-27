"""MR-CGE V1.3-A with a View-Global Top-2 expert router."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.mrcge_v13 import Model as V13Model


class Model(V13Model):
    """Keep V1.3-A state and experts; share one Top-2 choice per filter view."""

    @staticmethod
    def _view_global_logits(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Average node logits over this call's active receiver set, then broadcast."""
        active = support.to(dtype=torch.bool)
        if bool(active.any()):
            global_logits = logits[active].float().mean(dim=0, keepdim=True).to(logits.dtype)
        else:
            global_logits = logits.new_zeros((1, logits.size(-1)))
        return global_logits.expand_as(logits)

    def _filter(
        self,
        basis: list[torch.Tensor],
        support: torch.Tensor,
        current: torch.Tensor,
        relation_id: int,
        modality_id: int,
        cross_context: torch.Tensor,
        *,
        collect: bool,
    ):
        max_order = len(basis) - 1
        if not self.condition.startswith("M_"):
            return super()._filter(
                basis, support, current, relation_id, modality_id, cross_context, collect=collect
            )

        # Eval/collect uses the full expert stack, as in V1.3-A. Globalize the
        # exact same node logits before dense probabilities and Top-2 selection.
        if collect or not (self.training and torch.is_grad_enabled()):
            outputs, filtered, coefficients, residual = self.filter_bank(basis, support, max_order)
            first = basis[1] if len(basis) > 1 else basis[0]
            node_logits = self.router(current, first, cross_context, modality_id, relation_id)
            logits = self._view_global_logits(node_logits, support)
            dense = torch.softmax(logits.float(), dim=-1)
            top_logits, top_ids = torch.topk(logits, self.top_k, dim=-1)
            top_weights = torch.softmax(top_logits.float(), dim=-1).to(outputs.dtype)
            route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights.float()).to(outputs.dtype)
            selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
            active = support
            if bool(active.any()):
                importance = dense[active].mean(dim=0)
                selected_share = selected[active].float().mean(dim=0) / float(self.top_k)
                balance = self.num_experts * (importance * selected_share).sum()
            else:
                importance = dense.sum(dim=0) * 0.0
                selected_share = importance
                balance = dense.sum() * 0.0
            response = (outputs * route_weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
            route = {
                "logits": logits, "dense": dense, "weights": route_weights,
                "selected": selected, "importance": importance, "share": selected_share,
                "balance": balance,
            }
            detail = self._filter_summary({
                "G": filtered, "E": outputs, "residual": residual,
                "coefficients": coefficients.detach().cpu().tolist(),
            }, support) if collect else {}
            return response, route, balance, detail

        def compute(*inputs):
            return self._m_filter_impl(
                list(inputs[:max_order + 1]), inputs[max_order + 1], inputs[max_order + 2],
                inputs[max_order + 3], modality_id, relation_id, collect=collect,
            )

        args = (*basis, support, current, cross_context)
        if self.training and torch.is_grad_enabled() and not collect:
            response, balance, importance, selected_share, entropy, selected_entropy = checkpoint(
                lambda *values: compute(*values)[:6], *args,
                use_reentrant=False, preserve_rng_state=True,
            )
            route = {
                "importance": importance, "share": selected_share,
                "dense_entropy": entropy, "selected_entropy": selected_entropy,
                "active_nodes": support.sum().detach(),
            }
            return response, route, balance, {}
        response, balance, importance, selected_share, entropy, selected_entropy, extra = compute(*args)
        route = extra["route"]
        detail = extra["detail"]
        return response, route, balance, detail

    def _m_filter_impl(
        self, basis: list[torch.Tensor], support: torch.Tensor, current: torch.Tensor,
        cross_context: torch.Tensor, modality_id: int, relation_id: int, *, collect: bool,
    ):
        """V1.3-A low-memory expert path with the same view-global routing rule."""
        max_order = len(basis) - 1
        coefficients = self.filter_bank.coefficients(max_order, basis[0].dtype)
        first = basis[1] if len(basis) > 1 else basis[0]
        n = current.size(0)
        mod = self.router.modality_embedding.weight[modality_id].expand(n, -1)
        rel = self.router.relation_embedding.weight[relation_id].expand(n, -1)
        router_parts = (current, first, first - current, mod, rel, cross_context)
        first_layer = self.router.net[0]
        offset = 0
        pre_activation = None
        for part in router_parts:
            width = part.size(-1)
            contribution = F.linear(part, first_layer.weight[:, offset:offset + width], bias=None)
            pre_activation = contribution if pre_activation is None else pre_activation + contribution
            offset += width
        if offset != first_layer.in_features:
            raise RuntimeError("router input component widths do not match its frozen linear layer")
        if first_layer.bias is not None:
            pre_activation = pre_activation + first_layer.bias
        router_hidden = F.gelu(pre_activation)
        mask = support.to(basis[0].dtype).unsqueeze(-1)
        outputs = []
        filtered_values = []
        residuals = []
        for expert_id, expert in enumerate(self.filter_bank.experts):
            filtered = None
            for order in range(max_order + 1):
                term = coefficients[expert_id, order] * basis[order]
                filtered = term if filtered is None else filtered + term
            residual = self.eta * expert(filtered)
            value = self.filter_bank.norms[expert_id](filtered + residual) * mask
            outputs.append(value)
            if collect:
                filtered_values.append(filtered)
                residuals.append(residual * mask)

        node_logits = self.router.net[3](self.router.net[2](router_hidden))
        logits = self._view_global_logits(node_logits, support)
        dense = torch.softmax(logits.float(), dim=-1)
        top_logits, top_ids = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits.float(), dim=-1)
        route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights)
        selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
        response = None
        for expert_id, value in enumerate(outputs):
            weighted = value * route_weights[:, expert_id:expert_id + 1].to(value.dtype)
            response = weighted if response is None else response + weighted

        active = support
        if bool(active.any()):
            importance = dense[active].mean(dim=0)
            selected_share = selected[active].float().mean(dim=0) / float(self.top_k)
            balance = self.num_experts * (importance * selected_share).sum()
            dense_active = dense[active].clamp_min(1.0e-12)
            selected_active = route_weights[active].clamp_min(1.0e-12)
            entropy = -(dense_active * dense_active.log()).sum(dim=-1).mean()
            selected_entropy = -(selected_active * selected_active.log()).sum(dim=-1).mean()
        else:
            importance = dense.sum(dim=0) * 0.0
            selected_share = importance.detach()
            balance = dense.sum() * 0.0
            entropy = dense.sum() * 0.0
            selected_entropy = dense.sum() * 0.0

        route_summary = {
            "importance": importance.detach(), "share": selected_share.detach(),
            "dense_entropy": entropy.detach(), "selected_entropy": selected_entropy.detach(),
            "active_nodes": active.sum().detach(),
        }
        if collect:
            route = {
                "logits": logits, "dense": dense, "weights": route_weights,
                "selected": selected, **route_summary, "balance": balance,
            }
            detail = self._filter_summary({
                "G": torch.stack(filtered_values), "E": torch.stack(outputs),
                "residual": torch.stack(residuals),
                "coefficients": coefficients.detach().cpu().tolist(),
            }, support)
        else:
            route = route_summary
            detail = {}
        return response, balance, route_summary["importance"], route_summary["share"], \
            route_summary["dense_entropy"], route_summary["selected_entropy"], \
            {"route": route, "detail": detail}
