"""RM-CGE V2: edge-conditioned messages for the shared V7A expert bank."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b
from src.models.psce_mag_v7a import Model as V7AModel, _MODALITIES, chunked_weighted_spmm


VARIANTS = ("V0", "V1", "V2")


def build_modality_knn_operators(
    unique_edge_index: torch.Tensor,
    source_flags: torch.Tensor,
    num_nodes: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build separately row-normalized symmetric KNN supports from V7B flags."""
    if unique_edge_index.ndim != 2 or unique_edge_index.shape[0] != 2:
        raise ValueError("unique_edge_index must have shape [2,E]")
    if source_flags.shape != (unique_edge_index.size(1), 2):
        raise ValueError("source_flags must have shape [E,2]")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for column, modality in enumerate(_MODALITIES):
        pairs = unique_edge_index[:, source_flags[:, column] > 0]
        if pairs.numel():
            directed = torch.cat((pairs, pairs.flip(0)), dim=1).contiguous()
            degree = torch.zeros(num_nodes, dtype=torch.float32, device=directed.device)
            degree.index_add_(0, directed[1], torch.ones(directed.size(1), device=directed.device))
            weight = degree[directed[1]].clamp_min(1).reciprocal()
        else:
            directed = unique_edge_index.new_empty((2, 0))
            weight = torch.empty(0, dtype=torch.float32, device=unique_edge_index.device)
        result[modality] = {"edge_index": directed, "weight": weight}
    return result


class Model(V7AModel):
    """V0 delegates exactly to V7A/R0; V1/V2 add context and node routing."""

    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    supports_link_prediction = False
    record_run_telemetry = True
    record_epoch_history = True
    collect_v7_diagnostics = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        requested = str(cfg.model.get("variant", "V0")).upper()
        if requested not in VARIANTS:
            raise ValueError(f"rmcge_v2 variant must be one of {VARIANTS}")
        base_cfg = copy.deepcopy(cfg)
        base_cfg.model.variant = "a0_raw"
        super().__init__(base_cfg, data_info)
        self.study_variant = requested
        # Keep the parent identity so every V0 forward and parent helper follows
        # the exact a0_raw path.
        self.variant = "a0_raw"
        self.requires_global_semantic_candidates = True
        self._intervention = {
            "force_unit_edges": False,
            "cross_off": False,
            "knn_off": False,
            "node_router_off": False,
        }
        self._collect_diagnostic_tensors = False
        self._last_gradient_diagnostics: dict[str, Any] | None = None
        self.semantic_v7b_fingerprint: str | None = None
        self.register_buffer(
            "semantic_source_flags_v7b",
            torch.empty((0, 2), dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "semantic_text_edge_index_v2",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "semantic_text_weight_v2",
            torch.empty((0,), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "semantic_visual_edge_index_v2",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "semantic_visual_weight_v2",
            torch.empty((0,), dtype=torch.float32),
            persistent=False,
        )
        self.scale_floor = float(cfg.model.get("scale_floor", 1.0e-3))
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 16384))
        self.node_chunk_size = int(cfg.model.get("node_chunk_size", 8192))
        if self.scale_floor <= 0 or self.edge_chunk_size < 1 or self.node_chunk_size < 1:
            raise ValueError("scale floor and V2 chunk sizes must be positive")

        if requested != "V0":
            # Added parameters must not perturb the subsequent NC classifier RNG.
            with torch.random.fork_rng(devices=[]):
                self.cross_projections = nn.ModuleDict(
                    {
                        "visual_to_text": nn.Sequential(
                            nn.Linear(self.hidden_dim, 64, bias=False),
                            nn.GELU(),
                            nn.Linear(64, self.hidden_dim, bias=False),
                        ),
                        "text_to_visual": nn.Sequential(
                            nn.Linear(self.hidden_dim, 64, bias=False),
                            nn.GELU(),
                            nn.Linear(64, self.hidden_dim, bias=False),
                        ),
                    }
                )
                self.delta_coefficients = nn.Parameter(torch.full((self.num_experts,), 0.20))
                self.cross_coefficients = nn.Parameter(torch.full((self.num_experts,), 0.25))
                self.knn_coefficients = nn.Parameter(torch.full((self.num_experts,), 0.08))
                self.edge_projections = nn.ModuleDict(
                    {
                        "text": nn.Linear(self.hidden_dim, 32, bias=False),
                        "visual": nn.Linear(self.hidden_dim, 32, bias=False),
                    }
                )
                self.edge_modality_embedding = nn.Embedding(2, 8)
                self.edge_scorer = nn.Sequential(
                    nn.Linear(4 * 32 + 2 + 2 + 8, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                )
                nn.init.zeros_(self.edge_scorer[-1].weight)
                nn.init.zeros_(self.edge_scorer[-1].bias)
                self.node_modality_embedding = nn.Embedding(2, 8)
                self.node_router = nn.Sequential(
                    nn.Linear(4 * self.hidden_dim + 8, 64),
                    nn.GELU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(64, self.num_experts),
                )
                nn.init.zeros_(self.node_router[-1].weight)
                nn.init.zeros_(self.node_router[-1].bias)

    def load_semantic_cache(
        self,
        path: str,
        *,
        expected_dataset: str | None = None,
        expected_feature_fingerprint: str | None = None,
        expected_candidate_fingerprint: str | None = None,
        expected_v7b_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        metadata = super().load_semantic_cache(
            path,
            expected_dataset=expected_dataset,
            expected_feature_fingerprint=expected_feature_fingerprint,
            expected_candidate_fingerprint=expected_candidate_fingerprint,
        )
        cache = load_semantic_candidate_cache_v7b(
            path,
            expected_dataset=expected_dataset,
            expected_feature_fingerprint=expected_feature_fingerprint,
            expected_candidate_fingerprint=expected_candidate_fingerprint,
            expected_v7b_fingerprint=expected_v7b_fingerprint,
        )
        flags = cache["source_flags"].to(dtype=torch.uint8, device="cpu")
        unique = cache["unique_edge_index"].to(dtype=torch.long, device="cpu")
        if flags.ndim != 2 or flags.shape != (unique.size(1), 2):
            raise ValueError("V7B source flags must align with canonical candidate edges")
        self.semantic_source_flags_v7b = flags
        operators = build_modality_knn_operators(unique, flags, self.num_nodes)
        for modality in _MODALITIES:
            setattr(self, f"semantic_{modality}_edge_index_v2", operators[modality]["edge_index"])
            setattr(self, f"semantic_{modality}_weight_v2", operators[modality]["weight"])
        self.semantic_v7b_fingerprint = str(metadata["v7b_fingerprint"])
        return metadata

    def set_interventions(
        self,
        *,
        force_unit_edges: bool = False,
        cross_off: bool = False,
        knn_off: bool = False,
        node_router_off: bool = False,
    ) -> None:
        self._intervention = {
            "force_unit_edges": bool(force_unit_edges),
            "cross_off": bool(cross_off),
            "knn_off": bool(knn_off),
            "node_router_off": bool(node_router_off),
        }

    def _score_edges(
        self,
        *,
        text: torch.Tensor,
        visual: torch.Tensor,
        q_text: torch.Tensor,
        q_visual: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        degree: torch.Tensor,
        modality_index: int,
    ) -> torch.Tensor:
        if src.numel() == 0:
            return text.new_empty((0,))
        modality = self.edge_modality_embedding.weight[modality_index]
        log_degree = torch.log1p(degree)
        text_unit = F.normalize(text, p=2, dim=-1, eps=1.0e-8)
        visual_unit = F.normalize(visual, p=2, dim=-1, eps=1.0e-8)
        scores: list[torch.Tensor] = []
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            src_chunk, dst_chunk = src[start:end], dst[start:end]

            def score_chunk(qt, qv, ht, hv, tu, vu, src_ids, dst_ids, logdeg, mod):
                qt_i, qt_j = qt[dst_ids], qt[src_ids]
                qv_i, qv_j = qv[dst_ids], qv[src_ids]
                cos_t = (tu[dst_ids] * tu[src_ids]).sum(dim=-1, keepdim=True)
                cos_v = (vu[dst_ids] * vu[src_ids]).sum(dim=-1, keepdim=True)
                features = torch.cat(
                    (
                        qt_i,
                        qt_j,
                        qv_i,
                        qv_j,
                        cos_t,
                        cos_v,
                        logdeg[dst_ids, None],
                        logdeg[src_ids, None],
                        mod.expand(src_ids.numel(), -1),
                    ),
                    dim=-1,
                )
                return self.edge_scorer(features).squeeze(-1)

            if self.training and torch.is_grad_enabled():
                score = checkpoint(
                    score_chunk,
                    q_text,
                    q_visual,
                    text,
                    visual,
                    text_unit,
                    visual_unit,
                    src_chunk,
                    dst_chunk,
                    log_degree,
                    modality,
                    use_reentrant=False,
                )
            else:
                score = score_chunk(
                    q_text,
                    q_visual,
                    text,
                    visual,
                    text_unit,
                    visual_unit,
                    src_chunk,
                    dst_chunk,
                    log_degree,
                    modality,
                )
            scores.append(score)
        return torch.cat(scores, dim=0)

    def _node_correction(
        self,
        h: torch.Tensor,
        u1: torch.Tensor,
        delta: torch.Tensor,
        cross: torch.Tensor,
        modality_index: int,
    ) -> torch.Tensor:
        modality = self.node_modality_embedding.weight[modality_index]
        pieces: list[torch.Tensor] = []
        for start in range(0, h.size(0), self.node_chunk_size):
            end = min(start + self.node_chunk_size, h.size(0))

            def route_chunk(h_all, u_all, d_all, c_all, mod, lo=start, hi=end):
                values = torch.cat(
                    (h_all[lo:hi], u_all[lo:hi], d_all[lo:hi], c_all[lo:hi]),
                    dim=-1,
                )
                values = torch.cat((values, mod.expand(hi - lo, -1)), dim=-1)
                return self.node_router(values)

            if self.training and torch.is_grad_enabled():
                piece = checkpoint(
                    route_chunk, h, u1, delta, cross, modality, use_reentrant=False
                )
            else:
                piece = route_chunk(h, u1, delta, cross, modality)
            pieces.append(piece)
        return torch.cat(pieces, dim=0)

    @staticmethod
    def _scalar_rms(value: torch.Tensor, active: torch.Tensor | None = None) -> float:
        if active is not None:
            value = value[:, active, :] if value.ndim == 3 else value[active]
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().square().mean().sqrt().cpu())

    @staticmethod
    def _route_entropy(probs: torch.Tensor, active: torch.Tensor) -> float:
        if not bool(active.any()):
            return 0.0
        p = probs[active].float().clamp_min(1.0e-12)
        return float((-(p * p.log()).sum(dim=-1).mean()).detach().cpu())

    @staticmethod
    def _expert_pairwise_cosines(values: torch.Tensor, active: torch.Tensor) -> dict[str, float]:
        if not bool(active.any()):
            return {}
        flattened = values[:, active, :].detach().float().flatten(start_dim=1)
        normalized = F.normalize(flattened, dim=-1, eps=1.0e-12)
        result = {}
        for left in range(values.size(0)):
            for right in range(left + 1, values.size(0)):
                result[f"{left}_{right}"] = float((normalized[left] * normalized[right]).sum().cpu())
        return result

    def _edge_diagnostics(
        self, weights: dict[str, torch.Tensor], src: torch.Tensor, dst: torch.Tensor, degree: torch.Tensor
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        detached = {key: value.detach().float() for key, value in weights.items()}
        for modality, value in detached.items():
            if value.numel():
                quantiles = torch.quantile(value, value.new_tensor([0.1, 0.5, 0.9]))
                bins: dict[str, Any] = {}
                target_degree = degree[dst]
                for name, mask in (
                    ("0", target_degree == 0),
                    ("1_2", (target_degree >= 1) & (target_degree <= 2)),
                    ("3_5", (target_degree >= 3) & (target_degree <= 5)),
                    ("6_10", (target_degree >= 6) & (target_degree <= 10)),
                    ("11_20", (target_degree >= 11) & (target_degree <= 20)),
                    ("gt20", target_degree > 20),
                ):
                    if bool(mask.any()):
                        bins[name] = float(value[mask].mean().cpu())
                result[modality] = {
                    "mean": float(value.mean().cpu()),
                    "std": float(value.std(unbiased=False).cpu()),
                    "p10": float(quantiles[0].cpu()),
                    "p50": float(quantiles[1].cpu()),
                    "p90": float(quantiles[2].cpu()),
                    "mean_abs_from_one": float((value - 1).abs().mean().cpu()),
                    "fraction_abs_gt_0_1": float(((value - 1).abs() > 0.1).float().mean().cpu()),
                    "fraction_abs_gt_0_2": float(((value - 1).abs() > 0.2).float().mean().cpu()),
                    "mean_by_target_degree_bin": bins,
                }
        if detached["text"].numel() and detached["visual"].numel():
            a, b = detached["text"], detached["visual"]
            denominator = a.std(unbiased=False) * b.std(unbiased=False)
            result["text_visual_correlation"] = (
                float((((a - a.mean()) * (b - b.mean())).mean() / denominator).cpu())
                if a.numel() > 1 and float(denominator.cpu()) > 1.0e-12
                else 0.0
            )
            result["text_visual_correlation_defined"] = bool(
                float(denominator.cpu()) > 1.0e-12
            )
            result["text_visual_mean_abs_difference"] = float((a - b).abs().mean().cpu())
            # Reverse directions are looked up by ordered endpoint key. If an
            # input topology contains multiplicities, the recorded first reverse
            # match remains descriptive of the retained edge list.
            n = int(degree.numel())
            key = src * n + dst
            if key.numel():
                order = torch.argsort(key)
                sorted_key = key[order]
                reverse_key = dst * n + src
                pos = torch.searchsorted(sorted_key, reverse_key).clamp_max(sorted_key.numel() - 1)
                valid = sorted_key[pos] == reverse_key
                if bool(valid.any()):
                    reverse_idx = order[pos[valid]]
                    for modality in ("text", "visual"):
                        result.setdefault(modality, {})["mean_abs_reverse_difference"] = float(
                            (detached[modality][valid] - detached[modality][reverse_idx]).abs().mean().cpu()
                        )
        return result

    def _forward_added(self, x: torch.Tensor, edge_index: torch.Tensor, *, return_details: bool):
        inputs = self._split_modalities(x)
        priors = {m: self.projectors[m](inputs[m]) for m in _MODALITIES}
        src, dst, physical_norm, active, loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )
        degree = torch.zeros(x.size(0), device=x.device, dtype=priors["text"].dtype)
        degree.index_add_(0, dst, torch.ones(dst.numel(), device=x.device, dtype=degree.dtype))
        edge_index_directed = torch.stack((src, dst), dim=0)

        alpha = self._effective_alpha(self.alpha_raw, self.eps)

        edge_weights: dict[str, torch.Tensor] = {}
        if self.study_variant == "V2" and not self._intervention["force_unit_edges"]:
            q_text = self.edge_projections["text"](priors["text"])
            q_visual = self.edge_projections["visual"](priors["visual"])
            for modality_index, modality in enumerate(_MODALITIES):
                logits = self._score_edges(
                    text=priors["text"],
                    visual=priors["visual"],
                    q_text=q_text,
                    q_visual=q_visual,
                    src=src,
                    dst=dst,
                    degree=degree,
                    modality_index=modality_index,
                )
                edge_weights[modality] = 2.0 * torch.sigmoid(logits)
        else:
            edge_weights = {m: physical_norm.new_ones(src.numel()) for m in _MODALITIES}

        outputs: dict[str, torch.Tensor] = {}
        routes: dict[str, dict[str, torch.Tensor]] = {}
        balances: list[torch.Tensor] = []
        profile_rms: dict[str, dict[str, Any]] = {}
        for modality_index, modality in enumerate(_MODALITIES):
            other = "visual" if modality == "text" else "text"
            prior = priors[modality]
            p_norm = physical_norm * edge_weights[modality]
            if self.study_variant == "V0":
                # V0 never reaches this method. Keep the guard explicit.
                raise AssertionError("V0 must delegate to the frozen parent forward")
            state = prior
            raw_values: list[torch.Tensor] = []
            for _ in range(self.physical_hops):
                state = self._propagate(state, src, dst, physical_norm)
                state = state * active.to(state.dtype).unsqueeze(-1)
                raw_values.append(state)
            raw_u1 = raw_values[0]
            normalized_states = torch.stack(
                [self._active_rms_norm(value, active) for value in raw_values], dim=0
            )
            if bool(active.any()):
                scale = torch.sqrt(raw_u1[active].float().square().mean(dim=0) + self.eps).to(prior.dtype)
            else:
                scale = prior.new_zeros((self.hidden_dim,))
            scale = scale.detach().clamp_min(self.scale_floor)

            if self.study_variant == "V2" and not self._intervention["force_unit_edges"]:
                delta = chunked_weighted_spmm(
                    prior,
                    edge_index_directed,
                    physical_norm * (edge_weights[modality] - 1.0),
                    self.propagation_chunk_size,
                )
            else:
                delta = torch.zeros_like(prior)
            transformed_cross = self.cross_projections[
                "visual_to_text" if modality == "text" else "text_to_visual"
            ](priors[other])
            if self._intervention["cross_off"]:
                cross = torch.zeros_like(prior)
            else:
                cross = chunked_weighted_spmm(
                    transformed_cross,
                    edge_index_directed,
                    p_norm,
                    self.propagation_chunk_size,
                )
            if self._intervention["knn_off"]:
                semantic = torch.zeros_like(prior)
            else:
                semantic_edges = getattr(self, f"semantic_{modality}_edge_index_v2")
                semantic_weights = getattr(self, f"semantic_{modality}_weight_v2").to(
                    device=prior.device, dtype=prior.dtype
                )
                if semantic_edges.device != prior.device:
                    semantic_edges = semantic_edges.to(prior.device)
                semantic = chunked_weighted_spmm(
                    prior, semantic_edges, semantic_weights, self.propagation_chunk_size
                )
            delta_hat, cross_hat, semantic_hat = delta / scale, cross / scale, semantic / scale
            physical_profiles = torch.einsum("kl,lnd->knd", alpha, normalized_states)
            q_profiles = (
                physical_profiles
                + self.delta_coefficients[:, None, None] * delta_hat[None, :, :]
                + self.cross_coefficients[:, None, None] * cross_hat[None, :, :]
                + self.knn_coefficients[:, None, None] * semantic_hat[None, :, :]
            )
            expert_outputs: list[torch.Tensor] = []
            for expert_id, expert in enumerate(self.experts):
                profile = q_profiles[expert_id]
                transformed = (
                    checkpoint(expert, profile, use_reentrant=False)
                    if self.training and torch.is_grad_enabled()
                    else expert(profile)
                )
                expert_outputs.append(profile + self.expert_feature_scale * transformed)
            expert_values = torch.stack(expert_outputs, dim=0)

            base_route = self._physical_route(modality_index, prior, active)
            if self._intervention["node_router_off"]:
                route = base_route
                correction_logits = prior.new_zeros((prior.size(0), self.num_experts))
            else:
                correction_logits = self._node_correction(
                    prior, raw_u1, delta_hat, cross_hat, modality_index
                )
                base_logits = self.selection_head(base_route["encoded"]).expand(
                    prior.size(0), -1
                )
                route = self._top2(base_logits + correction_logits, self.top_k)
                route["strength"] = base_route["strength"]
                route["router_state"] = base_route["router_state"]
                route["static_input"] = base_route["static_input"]
                route["encoded"] = base_route["encoded"]
                importance, selection_share, balance = self._route_balance(route, active)
                route["active"] = active
                route["importance"] = importance
                route["selection_share"] = selection_share
                route["balance"] = balance
            if self._intervention["node_router_off"]:
                route["active"] = active
            routes[modality] = route
            balances.append(route["balance"])
            mixture = torch.einsum("nk,knd->nd", route["route_weights"], expert_values)
            physical_correction = route["strength"].unsqueeze(-1) * mixture
            outputs[modality] = prior + physical_correction
            if return_details:
                profile_rms[modality] = {
                    "u1_rms": self._scalar_rms(raw_u1, active),
                    "delta_rms": self._scalar_rms(delta, active),
                    "delta_hat_rms": self._scalar_rms(delta_hat, active),
                    "delta_weighted_rms_by_expert": [
                        self._scalar_rms(self.delta_coefficients[k] * delta_hat, active)
                        for k in range(self.num_experts)
                    ],
                    "cross_rms": self._scalar_rms(cross, active),
                    "cross_hat_rms": self._scalar_rms(cross_hat, active),
                    "cross_weighted_rms_by_expert": [
                        self._scalar_rms(self.cross_coefficients[k] * cross_hat, active)
                        for k in range(self.num_experts)
                    ],
                    "knn_rms": self._scalar_rms(semantic, active),
                    "knn_hat_rms": self._scalar_rms(semantic_hat, active),
                    "knn_weighted_rms_by_expert": [
                        self._scalar_rms(self.knn_coefficients[k] * semantic_hat, active)
                        for k in range(self.num_experts)
                    ],
                    "floor_hit_dimensions": int((scale <= self.scale_floor).sum().item()),
                    "physical_profile_rms_by_expert": [
                        self._scalar_rms(physical_profiles[k], active)
                        for k in range(self.num_experts)
                    ],
                    "combined_profile_rms_by_expert": [
                        self._scalar_rms(q_profiles[k], active)
                        for k in range(self.num_experts)
                    ],
                    "expert_output_rms": self._scalar_rms(expert_values, active),
                    "expert_output_rms_by_expert": [
                        self._scalar_rms(expert_values[k], active)
                        for k in range(self.num_experts)
                    ],
                    "expert_output_pairwise_cosine": self._expert_pairwise_cosines(
                        expert_values, active
                    ),
                    "weighted_graph_correction_rms": self._scalar_rms(physical_correction, active),
                    "intrinsic_rms": self._scalar_rms(prior, active),
                    "router_correction_rms": self._scalar_rms(correction_logits, active),
                    "router_entropy": self._route_entropy(route["dense_probs"], active),
                    "dense_probability_mean": route["dense_probs"][active].mean(dim=0).detach().cpu().tolist()
                    if bool(active.any()) else [0.0] * self.num_experts,
                    "expert_selection_share": route["selection_share"].detach().cpu().tolist(),
                    "base_expert_selection_share": base_route["selection_share"].detach().cpu().tolist(),
                    "router_top2_flip_fraction": float(
                        (route["top_indices"][active] != base_route["top_indices"][active])
                        .any(dim=-1).float().mean().cpu()
                    ) if bool(active.any()) else 0.0,
                    "expert_coefficients": {
                        "delta": self.delta_coefficients.detach().cpu().tolist(),
                        "cross": self.cross_coefficients.detach().cpu().tolist(),
                        "knn": self.knn_coefficients.detach().cpu().tolist(),
                    },
                }
            del raw_values, normalized_states

        balance = 0.5 * (balances[0] + balances[1])
        aux_loss = self.balance_weight * balance
        fused_input = torch.cat((outputs["text"], outputs["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused_input)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused_input)))
            )
        )
        info: dict[str, Any] = {
            "variant": self.study_variant,
            "active_nodes": active.detach(),
            "input_self_loops_removed": loops,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "interventions": dict(self._intervention),
        }
        if return_details:
            info["details"] = {
                "modalities": profile_rms,
                "edge_weights": self._edge_diagnostics(edge_weights, src, dst, degree),
                "fusion_input_rms": self._scalar_rms(fused_input),
                "fusion_input_mean": float(fused_input.detach().float().mean().cpu()),
                "fusion_input_std": float(fused_input.detach().float().std(unbiased=False).cpu()),
                "fused_output_rms": self._scalar_rms(z),
                "fused_output_mean": float(z.detach().float().mean().cpu()),
                "fused_output_std": float(z.detach().float().std(unbiased=False).cpu()),
                "gradient_diagnostics": self._last_gradient_diagnostics,
            }
        return z, None, None, aux_loss, info

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if self.study_variant == "V0":
            return super().forward(x, edge_index, return_details=return_details)
        if edge_index is None:
            raise ValueError("RM-CGE V2 requires the global physical edge_index")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("RM-CGE V2 is full-graph NC only")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("RM-CGE V1/V2 require a validated V7B.1 semantic cache")
        return self._forward_added(x, edge_index, return_details=return_details)

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        if self.study_variant == "V0":
            return super().capture_gradient_diagnostics()
        selected = (
            "alpha_raw",
            "delta_coefficients",
            "cross_coefficients",
            "knn_coefficients",
            "edge_scorer.2.weight",
            "edge_scorer.2.bias",
            "edge_scorer.0.weight",
            "cross_projections.visual_to_text.0.weight",
            "cross_projections.visual_to_text.2.weight",
            "cross_projections.text_to_visual.0.weight",
            "cross_projections.text_to_visual.2.weight",
            "node_router.3.weight",
            "node_router.3.bias",
            "experts.0.linear1.weight",
            "selection_head.weight",
            "strength_head.bias",
            "fusion_linear1.weight",
            "projectors.text.linear1.weight",
            "projectors.visual.linear1.weight",
        )
        params = dict(self.named_parameters())
        result: dict[str, Any] = {}
        for name in selected:
            grad = params[name].grad
            norm = float(grad.detach().float().norm().cpu()) if grad is not None else 0.0
            result[name] = {
                "gradient_present": grad is not None,
                "gradient_finite": bool(grad is None or torch.isfinite(grad).all()),
                "gradient_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        self._last_gradient_diagnostics = result
        self._gradient_diagnostics = result
        return result


__all__ = ["Model", "VARIANTS", "build_modality_knn_operators"]
