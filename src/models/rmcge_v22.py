"""RM-CGE V2.2: receiver-conditioned cross-modal message content experts.

The V2.1 C1 public model is the common initialization tree. M0--M3 mask its
Intra selector; M4 enables the unchanged V2.1 C1 path. Content modules are
initialized in an isolated RNG scope so they do not shift shared model or
classifier initialization.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from src.models.psce_mag_v7a import chunked_weighted_spmm
from src.models.rmcge_v21 import Model as RMCGEV21Model, _MODALITIES


CONDITIONS = ("M0", "M1", "M2", "M3", "M4")
CONTENT_MODES = {"M0": "off", "M1": "sender", "M2": "receiver", "M3": "pair", "M4": "pair"}


class ChunkedPairGateAggregate(Function):
    """Pair-gated low-rank scatter that recomputes one edge chunk in backward.

    Only node tensors and edge indices are saved. Neither the full E x 256
    messages nor a full E x 32 gate tensor survives the forward pass.
    """

    @staticmethod
    def forward(ctx, q, k, v, src, dst, physical, modality, w1, b1, w2, b2, chunk_size):
        out = q.new_zeros((q.size(0), v.size(-1)))
        chunk_size = int(chunk_size)
        for start in range(0, src.numel(), chunk_size):
            end = min(start + chunk_size, src.numel())
            s, d = src[start:end], dst[start:end]
            q_e, k_e = q.index_select(0, d), k.index_select(0, s)
            features = torch.cat((q_e, k_e, q_e * k_e, (q_e - k_e).abs(), modality.expand(s.numel(), -1)), dim=-1)
            hidden = F.gelu(F.linear(features, w1, b1))
            gate = torch.tanh(F.linear(hidden, w2, b2))
            messages = gate * v.index_select(0, s) * physical[start:end, None]
            out.index_add_(0, d, messages)
        ctx.save_for_backward(q, k, v, src, dst, physical, modality, w1, b1, w2, b2)
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, src, dst, physical, modality, w1, b1, w2, b2 = ctx.saved_tensors
        needs = ctx.needs_input_grad
        grad_q = torch.zeros_like(q) if needs[0] else None
        grad_k = torch.zeros_like(k) if needs[1] else None
        grad_v = torch.zeros_like(v) if needs[2] else None
        grad_modality = torch.zeros_like(modality) if needs[6] else None
        grad_w1 = torch.zeros_like(w1) if needs[7] else None
        grad_b1 = torch.zeros_like(b1) if needs[8] else None
        grad_w2 = torch.zeros_like(w2) if needs[9] else None
        grad_b2 = torch.zeros_like(b2) if needs[10] else None

        for start in range(0, src.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, src.numel())
            s, d = src[start:end], dst[start:end]
            req = []
            slots = []

            def local(value, required: bool, slot: str):
                if not required:
                    return value.index_select(0, d if slot == "q" else s).detach() if slot in {"q", "k", "v"} else value.detach()
                if slot in {"q", "k", "v"}:
                    value = value.index_select(0, d if slot == "q" else s)
                value = value.detach().requires_grad_(True)
                req.append(value)
                slots.append(slot)
                return value

            with torch.enable_grad():
                q_e = local(q, needs[0], "q")
                k_e = local(k, needs[1], "k")
                v_e = local(v, needs[2], "v")
                mod_e = local(modality, needs[6], "mod")
                w1_e = local(w1, needs[7], "w1")
                b1_e = local(b1, needs[8], "b1")
                w2_e = local(w2, needs[9], "w2")
                b2_e = local(b2, needs[10], "b2")
                features = torch.cat((q_e, k_e, q_e * k_e, (q_e - k_e).abs(), mod_e.expand(s.numel(), -1)), dim=-1)
                hidden = F.gelu(F.linear(features, w1_e, b1_e))
                gate = torch.tanh(F.linear(hidden, w2_e, b2_e))
                messages = gate * v_e * physical[start:end, None]
                grad_values = torch.autograd.grad(
                    messages,
                    req,
                    grad_outputs=grad_out.index_select(0, d),
                    allow_unused=True,
                ) if req else ()

            for slot, value in zip(slots, grad_values):
                if value is None:
                    continue
                if slot == "q":
                    grad_q.index_add_(0, d, value)
                elif slot == "k":
                    grad_k.index_add_(0, s, value)
                elif slot == "v":
                    grad_v.index_add_(0, s, value)
                elif slot == "mod":
                    grad_modality.add_(value)
                elif slot == "w1":
                    grad_w1.add_(value)
                elif slot == "b1":
                    grad_b1.add_(value)
                elif slot == "w2":
                    grad_w2.add_(value)
                elif slot == "b2":
                    grad_b2.add_(value)

        return grad_q, grad_k, grad_v, None, None, None, grad_modality, grad_w1, grad_b1, grad_w2, grad_b2, None


class Model(RMCGEV21Model):
    """Five V2.2 comparison arms sharing the V2.1 collaborative backbone."""

    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    supports_link_prediction = False
    record_run_telemetry = True
    record_epoch_history = True
    collect_v7_diagnostics = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        requested = str(cfg.model.get("condition", cfg.model.get("variant", "M0"))).upper()
        if requested not in CONDITIONS:
            raise ValueError(f"rmcge_v22 condition must be one of {CONDITIONS}")
        base_cfg = copy.deepcopy(cfg)
        base_cfg.model.condition = "C1"
        super().__init__(base_cfg, data_info)
        # Keep the V2.1 C1 implementation identity for its internal helpers.
        self.condition = "C1"
        self.study_condition = requested
        self.content_mode = CONTENT_MODES[requested]
        self.content_rank = int(cfg.model.get("content_rank", 32))
        self.content_gate_hidden = int(cfg.model.get("content_gate_hidden", 64))
        self.modality_embedding_dim = int(cfg.model.get("modality_embedding_dim", 8))
        self.content_edge_chunk_size = int(cfg.model.get("content_edge_chunk_size", self.purpose_edge_chunk_size))
        if self.content_rank != 32 or self.content_gate_hidden != 64 or self.modality_embedding_dim != 8:
            raise ValueError("V2.2 frozen dimensions are rank=32, gate_hidden=64, modality_embedding_dim=8")
        if self.content_edge_chunk_size < 1:
            raise ValueError("content_edge_chunk_size must be positive")
        # These modules are present in every arm. Forking protects all inherited
        # public parameters and the subsequently constructed NC classifier.
        with torch.random.fork_rng(devices=[]):
            self.content_query = nn.Linear(self.hidden_dim, self.content_rank, bias=False)
            self.content_key = nn.Linear(self.hidden_dim, self.content_rank, bias=False)
            self.content_value = nn.Linear(self.hidden_dim, self.content_rank, bias=False)
            self.content_output = nn.Linear(self.content_rank, self.hidden_dim, bias=False)
            self.content_modality_embedding = nn.Embedding(2, self.modality_embedding_dim)
            self.content_gate = nn.Sequential(
                nn.Linear(4 * self.content_rank + self.modality_embedding_dim, self.content_gate_hidden),
                nn.GELU(),
                nn.Linear(self.content_gate_hidden, self.content_rank),
            )
            nn.init.zeros_(self.content_gate[-1].weight)
            nn.init.zeros_(self.content_gate[-1].bias)
            self.content_eta = nn.Parameter(torch.full((self.num_experts,), 0.25))
        self._v22_intervention = {"content_off": requested == "M0", "receiver_query_permutation": False}
        self._capture_content_nodes = False
        self._last_content_by_modality: dict[str, torch.Tensor] = {}
        # M0--M3 have no active Intra selector. M4 uses the original C1 path.
        self._v21_intervention["intra_selection_off"] = requested != "M4"
        self._initial_content_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.named_parameters()
            if name.startswith("content_")
        }

    def set_interventions(
        self,
        *,
        content_off: bool = False,
        intra_selection_off: bool | None = None,
        cross_off: bool = False,
        knn_off: bool = False,
        node_router_off: bool = False,
        receiver_query_permutation: bool = False,
        force_unit_edges: bool = False,
        cross_selection_off: bool = False,
    ) -> None:
        if intra_selection_off is None:
            intra_selection_off = self.study_condition != "M4"
        super().set_interventions(
            force_unit_edges=force_unit_edges,
            intra_selection_off=intra_selection_off,
            cross_selection_off=cross_selection_off,
            cross_off=cross_off,
            knn_off=knn_off,
            node_router_off=node_router_off,
        )
        self._v22_intervention = {
            "content_off": bool(content_off) or self.study_condition == "M0",
            "receiver_query_permutation": bool(receiver_query_permutation),
        }

    def _gate_features(self, q: torch.Tensor, k: torch.Tensor, mode: str, modality: torch.Tensor) -> torch.Tensor:
        if mode == "sender":
            q = torch.zeros_like(q)
        elif mode == "receiver":
            k = torch.zeros_like(k)
        elif mode != "pair":
            raise ValueError(f"unknown content mode: {mode}")
        return torch.cat((q, k, q * k, (q - k).abs(), modality.expand(q.size(0), -1)), dim=-1)

    def _edge_gates(self, q: torch.Tensor, k: torch.Tensor, mode: str, modality: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.content_gate(self._gate_features(q, k, mode, modality)))

    def _content_residual(
        self,
        prior: torch.Tensor,
        cross_message: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        physical: torch.Tensor,
        active: torch.Tensor,
        scale: torch.Tensor,
        modality_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._v22_intervention["content_off"] or self.content_mode == "off":
            zero = torch.zeros_like(prior)
            return zero, {"lowrank_rms": 0.0, "residual_rms": 0.0, "residual_cross_cosine": 0.0}
        q = self.content_query(prior)
        k = self.content_key(cross_message)
        v = self.content_value(cross_message)
        if self._v22_intervention["receiver_query_permutation"]:
            # Deterministic label-free permutation; does not consume the train RNG.
            permutation = torch.roll(torch.arange(q.size(0), device=q.device), shifts=1)
            q = q.index_select(0, permutation)
        modality = self.content_modality_embedding.weight[modality_index]
        edge_index = torch.stack((src, dst), dim=0)
        if self.content_mode == "sender":
            q_zero = torch.zeros_like(k)
            gate = self._edge_gates(q_zero, k, "sender", modality)
            lowrank = chunked_weighted_spmm(gate * v, edge_index, physical, self.propagation_chunk_size)
        elif self.content_mode == "receiver":
            # Receiver-only FiLM commutes with linear neighbor aggregation.
            aggregate = chunked_weighted_spmm(v, edge_index, physical, self.propagation_chunk_size)
            zeros = torch.zeros_like(q)
            gate = self._edge_gates(q, zeros, "receiver", modality)
            lowrank = gate * aggregate
        else:
            lowrank = ChunkedPairGateAggregate.apply(
                q, k, v, src, dst, physical, modality,
                self.content_gate[0].weight, self.content_gate[0].bias,
                self.content_gate[2].weight, self.content_gate[2].bias,
                self.content_edge_chunk_size,
            )
            gate = None
        residual = self.content_output(lowrank)
        residual = residual * active.to(residual.dtype).unsqueeze(-1)
        # V2/V2.1 use a detached, per-channel first physical-hop RMS reference.
        reference = scale.detach().float()
        normalized = residual / reference
        normalized = normalized * active.to(normalized.dtype).unsqueeze(-1)
        if self._capture_content_nodes:
            self._last_content_by_modality[_MODALITIES[modality_index]] = normalized.detach()
        diagnostics = {
            "lowrank_rms": self._scalar_rms(lowrank, active),
            "residual_rms": self._scalar_rms(residual, active),
            "normalized_residual_rms": self._scalar_rms(normalized, active),
            "residual_cross_cosine": float(self._cosine_mean(residual, cross_message, active).detach().cpu()),
            "eta_weighted_rms_by_expert": [
                self._scalar_rms(self.content_eta[index] * normalized, active)
                for index in range(self.num_experts)
            ],
            "gate_mode": self.content_mode,
            "gate_mean": float(gate.detach().float().mean().cpu()) if gate is not None else None,
            "gate_std": float(gate.detach().float().std(unbiased=False).cpu()) if gate is not None else None,
        }
        return normalized, diagnostics

    @torch.no_grad()
    def content_gate_diagnostics(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, Any]:
        """Stream edge gates into sufficient statistics without retaining E x rank."""
        if self.content_mode == "off":
            return {modality: {"status": "disabled"} for modality in _MODALITIES}
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, _, _, _ = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        output: dict[str, Any] = {}
        for modality_index, modality_name in enumerate(_MODALITIES):
            other = "visual" if modality_name == "text" else "text"
            cross = self.cross_projections[
                "visual_to_text" if modality_name == "text" else "text_to_visual"
            ](priors[other])
            q, k = self.content_query(priors[modality_name]), self.content_key(cross)
            modality = self.content_modality_embedding.weight[modality_index]
            total = torch.zeros(self.content_rank, device=x.device)
            square = torch.zeros_like(total)
            product = torch.zeros((self.content_rank, self.content_rank), device=x.device)
            count = 0
            saturated = 0
            query_delta_sum = 0.0
            for start in range(0, src.numel(), self.content_edge_chunk_size):
                end = min(start + self.content_edge_chunk_size, src.numel())
                s, d = src[start:end], dst[start:end]
                if self.content_mode == "sender":
                    q_edge, k_edge = torch.zeros((s.numel(), self.content_rank), device=x.device), k[s]
                elif self.content_mode == "receiver":
                    q_edge, k_edge = q[d], torch.zeros((s.numel(), self.content_rank), device=x.device)
                else:
                    q_edge, k_edge = q[d], k[s]
                gates = self._edge_gates(q_edge, k_edge, self.content_mode, modality)
                total += gates.float().sum(0)
                square += gates.float().square().sum(0)
                product += gates.float().T @ gates.float()
                count += int(gates.size(0))
                saturated += int((gates.float().abs() >= 0.99).sum().item())
                if self.content_mode == "pair":
                    q_changed = torch.roll(q, shifts=1, dims=0)[d]
                    changed = self._edge_gates(q_changed, k[s], "pair", modality)
                    query_delta_sum += float((gates.float() - changed.float()).square().sum().cpu())
            safe_count = max(count, 1)
            mean = total / safe_count
            variance = (square / safe_count - mean.square()).clamp_min(0)
            covariance = product / safe_count - mean[:, None] * mean[None, :]
            std = variance.sqrt()
            correlation = covariance / (std[:, None] * std[None, :]).clamp_min(1.0e-12)
            output[modality_name] = {
                "edge_count": count,
                "channel_mean": mean.cpu().tolist(),
                "channel_std": std.cpu().tolist(),
                "channel_correlation": correlation.cpu().tolist(),
                "abs_ge_0_99_fraction": saturated / max(count * self.content_rank, 1),
                "receiver_query_roll_gate_rms": (query_delta_sum / max(count * self.content_rank, 1)) ** 0.5,
            }
        return output

    def _forward_v22_added(self, x: torch.Tensor, edge_index: torch.Tensor, *, return_details: bool):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, physical, active, loops = self._normalized_operator(edge_index, x.size(0), priors["text"].dtype)
        degree_float = torch.zeros(x.size(0), dtype=torch.float32, device=x.device)
        degree_float.index_add_(0, dst, torch.ones(dst.numel(), dtype=torch.float32, device=x.device))
        degree = degree_float.long()
        edge_index_directed = torch.stack((src, dst), dim=0)
        alpha = self._effective_alpha(self.alpha_raw, self.eps)
        outputs: dict[str, torch.Tensor] = {}
        balances: list[torch.Tensor] = []
        profile_diagnostics: dict[str, dict[str, Any]] = {}
        selection_weights: dict[str, torch.Tensor] = {}
        selection_scores: dict[str, torch.Tensor] = {}

        for modality_index, modality in enumerate(_MODALITIES):
            other = "visual" if modality == "text" else "text"
            prior = priors[modality]
            state = prior
            raw_states = []
            for _ in range(self.physical_hops):
                state = self._propagate(state, src, dst, physical)
                state = state * active.to(state.dtype).unsqueeze(-1)
                raw_states.append(state)
            raw_u1 = raw_states[0]
            normalized_states = torch.stack([self._active_rms_norm(value, active) for value in raw_states], dim=0)
            if bool(active.any()):
                scale = torch.sqrt(raw_u1[active].float().square().mean(dim=0) + self.eps).to(prior.dtype)
            else:
                scale = prior.new_zeros((self.hidden_dim,))
            scale = scale.detach().clamp_min(self.scale_floor)

            x_cross = self.cross_projections[
                "visual_to_text" if modality == "text" else "text_to_visual"
            ](priors[other])
            cross_base = chunked_weighted_spmm(x_cross, edge_index_directed, physical, self.propagation_chunk_size)
            if self._v21_intervention["cross_off"]:
                cross_base = torch.zeros_like(prior)
            semantic_edges = getattr(self, f"semantic_{modality}_edge_index_v2")
            semantic_weight = getattr(self, f"semantic_{modality}_weight_v2").to(device=prior.device, dtype=prior.dtype)
            if semantic_edges.device != prior.device:
                semantic_edges = semantic_edges.to(prior.device)
            semantic = (
                torch.zeros_like(prior)
                if self._v21_intervention["knn_off"]
                else chunked_weighted_spmm(prior, semantic_edges, semantic_weight, self.propagation_chunk_size)
            )

            zero_scores = physical.new_zeros(src.numel())
            if self._v21_intervention["intra_selection_off"]:
                intra_scores, intra_selected = zero_scores, physical
            else:
                intra_scores = self._score_message_edges(prior, prior, src, dst, degree_float, modality_index, 0)
                from src.models.rmcge_v21 import conserve_receiver_mass
                intra_selected = conserve_receiver_mass(physical, intra_scores, dst, x.size(0))
            intra_delta = chunked_weighted_spmm(
                prior, edge_index_directed, intra_selected - physical, self.propagation_chunk_size
            )
            intra_hat = intra_delta / scale
            cross_base_hat = cross_base / scale
            semantic_hat = semantic / scale
            content_hat, content_diagnostics = self._content_residual(
                prior, x_cross, src, dst, physical, active, scale, modality_index
            )
            physical_profiles = torch.einsum("kl,lnd->knd", alpha, normalized_states)
            q_profiles = (
                physical_profiles
                + self.delta_coefficients[:, None, None] * intra_hat[None, :, :]
                + self.cross_coefficients[:, None, None] * cross_base_hat[None, :, :]
                + self.knn_coefficients[:, None, None] * semantic_hat[None, :, :]
                + self.content_eta[:, None, None] * content_hat[None, :, :]
            )
            expert_outputs = []
            for expert_id, expert in enumerate(self.experts):
                profile = q_profiles[expert_id]
                transformed = checkpoint(expert, profile, use_reentrant=False) if self.training and torch.is_grad_enabled() else expert(profile)
                expert_outputs.append(profile + self.expert_feature_scale * transformed)
            expert_values = torch.stack(expert_outputs, dim=0)

            base_route = self._physical_route(modality_index, prior, active)
            if self._v21_intervention["node_router_off"]:
                route = base_route
                correction_logits = prior.new_zeros((prior.size(0), self.num_experts))
            else:
                correction_logits = self._node_correction(prior, raw_u1, intra_hat, cross_base_hat, modality_index)
                base_logits = self.selection_head(base_route["encoded"]).expand(prior.size(0), -1)
                route = self._top2(base_logits + correction_logits, self.top_k)
                route["strength"] = base_route["strength"]
                route["router_state"] = base_route["router_state"]
                route["static_input"] = base_route["static_input"]
                route["encoded"] = base_route["encoded"]
                importance, selection_share, balance = self._route_balance(route, active)
                route.update({"active": active, "importance": importance, "selection_share": selection_share, "balance": balance})
            if self._v21_intervention["node_router_off"]:
                route["active"] = active
            balances.append(route["balance"])
            mixture = torch.einsum("nk,knd->nd", route["route_weights"], expert_values)
            outputs[modality] = prior + route["strength"].unsqueeze(-1) * mixture
            selection_weights[f"{modality}_intra"] = intra_selected
            selection_scores[f"{modality}_intra"] = intra_scores
            if return_details:
                profile_diagnostics[modality] = {
                    "u1_rms": self._scalar_rms(raw_u1, active),
                    "cross_rms": self._scalar_rms(cross_base, active),
                    "cross_hat_rms": self._scalar_rms(cross_base_hat, active),
                    "knn_rms": self._scalar_rms(semantic, active),
                    "knn_hat_rms": self._scalar_rms(semantic_hat, active),
                    "intra_delta_rms": self._scalar_rms(intra_delta, active),
                    "content": content_diagnostics,
                    "physical_profile_rms_by_expert": [self._scalar_rms(physical_profiles[k], active) for k in range(self.num_experts)],
                    "combined_profile_rms_by_expert": [self._scalar_rms(q_profiles[k], active) for k in range(self.num_experts)],
                    "expert_output_rms_by_expert": [self._scalar_rms(expert_values[k], active) for k in range(self.num_experts)],
                    "intrinsic_rms": self._scalar_rms(prior, active),
                    "weighted_graph_correction_rms": self._scalar_rms(outputs[modality] - prior, active),
                    "router_entropy": self._route_entropy(route["dense_probs"], active),
                    "expert_selection_share": route["selection_share"].detach().cpu().tolist(),
                    "expert_coefficients": {
                        "cross_base": self.cross_coefficients.detach().cpu().tolist(),
                        "knn": self.knn_coefficients.detach().cpu().tolist(),
                        "intra": self.delta_coefficients.detach().cpu().tolist(),
                        "content_eta": self.content_eta.detach().cpu().tolist(),
                    },
                }
            del raw_states, normalized_states

        balance = 0.5 * (balances[0] + balances[1])
        aux_loss = self.balance_weight * balance
        fused_input = torch.cat((outputs["text"], outputs["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused_input)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused_input))))
        )
        info: dict[str, Any] = {
            "variant": self.study_condition,
            "condition": self.study_condition,
            "active_nodes": active.detach(),
            "input_self_loops_removed": loops,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "interventions": {**self._v21_intervention, **self._v22_intervention},
        }
        if return_details:
            info["details"] = {
                "modalities": profile_diagnostics,
                "fusion_input_rms": self._scalar_rms(fused_input),
                "fused_output_rms": self._scalar_rms(z),
                "gradient_diagnostics": self._last_gradient_diagnostics,
            }
        return z, None, None, aux_loss, info

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if edge_index is None:
            raise ValueError("RM-CGE V2.2 requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("RM-CGE V2.2 is full-graph NC only")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("RM-CGE V2.2 requires a validated V7B.1 semantic cache")
        if self.study_condition == "M0":
            result = super().forward(x, edge_index, return_details=return_details)
            result[4]["variant"] = "M0"
            result[4]["condition"] = "M0"
            result[4]["interventions"] = {**self._v21_intervention, **self._v22_intervention}
            return result
        return self._forward_v22_added(x, edge_index, return_details=return_details)

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        diagnostics = super().capture_gradient_diagnostics()
        parameters = dict(self.named_parameters())
        for name in (
            "content_query.weight", "content_key.weight", "content_value.weight",
            "content_output.weight", "content_modality_embedding.weight",
            "content_gate.0.weight", "content_gate.0.bias",
            "content_gate.2.weight", "content_gate.2.bias", "content_eta",
        ):
            parameter = parameters[name]
            gradient = parameter.grad
            norm = float(gradient.detach().float().norm().cpu()) if gradient is not None else 0.0
            diagnostics[name] = {
                "gradient_present": gradient is not None,
                "gradient_finite": bool(gradient is None or torch.isfinite(gradient).all()),
                "gradient_norm": norm,
                "gradient_nonzero": bool(norm > 0.0),
            }
        self._last_gradient_diagnostics = diagnostics
        return diagnostics


__all__ = ["Model", "CONDITIONS", "CONTENT_MODES", "ChunkedPairGateAggregate"]
