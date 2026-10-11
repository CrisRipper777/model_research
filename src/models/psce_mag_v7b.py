"""V7B provenance-aware relative semantic relation variants."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b
from src.models.psce_mag_v7a import Model as V7AModel, _MODALITIES, chunked_weighted_spmm

VARIANTS = (
    "b0_v7a_original",
    "b1_fixed_knn",
    "b2_random_fixed",
    "b3_relative_shared",
    "b4_relative_modality",
)


class Model(V7AModel):
    """Reuse V7A A2's trunk and override only relation weighting where needed."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        requested = str(cfg.model.get("variant", "b4_relative_modality"))
        if requested not in VARIANTS:
            raise ValueError(f"V7B variant must be one of {VARIANTS}")
        self.v7b_variant = requested
        # The parent is instantiated as the frozen A2 body for every V7B arm.
        cfg.model.variant = "a2_conditional_dual"
        super().__init__(cfg, data_info)
        self.variant = "a2_conditional_dual"
        self.v7b_relative = requested in {"b3_relative_shared", "b4_relative_modality"}
        self.register_buffer(
            "semantic_source_flags", torch.empty((0, 2), dtype=torch.uint8), persistent=False
        )
        self.semantic_v7b_fingerprint: str | None = None
        self._collect_fixed_edge_details = False
        self._last_fixed_edge_scores: torch.Tensor | None = None
        if self.v7b_relative:
            # Isolate these additions from the RNG stream used by the common
            # V7A trunk and the NC classifier, preserving paired init.
            with torch.random.fork_rng(devices=[]):
                self.relation_scorer = nn.Sequential(
                    nn.Linear(4 * self.relation_dim + 2, self.relation_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.relation_hidden_dim, 2, bias=False),
                )
                nn.init.normal_(self.relation_scorer[-1].weight, mean=0.0, std=0.005)
                with torch.no_grad():
                    self.relation_scorer[-1].weight[1].copy_(self.relation_scorer[-1].weight[0])
        else:
            # Fixed-weight controls do not optimize or use relation parameters.
            for module in (self.relation_projections, self.relation_scorer):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

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
        self.semantic_source_flags = cache["source_flags"]
        self.semantic_v7b_fingerprint = str(metadata["v7b_fingerprint"])
        return metadata

    def _score_semantic_edges(
        self,
        projected: dict[str, torch.Tensor],
        unique_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.v7b_variant == "b0_v7a_original":
            scores = super()._score_semantic_edges(projected, unique_edge_index)
            self._last_fixed_edge_scores = scores.detach() if self._collect_fixed_edge_details else None
            return scores
        if not self.v7b_relative:
            # Parent forward applies sigmoid, yielding the exact constant 0.8.
            scores = projected["text"].new_full(
                (unique_edge_index.size(1),), math.log(0.8 / 0.2)
            )
            self._last_fixed_edge_scores = scores.detach() if self._collect_fixed_edge_details else None
            return scores
        if self.semantic_source_flags.size(0) != unique_edge_index.size(1):
            raise RuntimeError("V7B source flags do not align with candidate edges")
        if unique_edge_index.numel() == 0:
            return projected["text"].new_empty((0, 2))
        pieces: list[torch.Tensor] = []
        for start in range(0, unique_edge_index.size(1), self.relation_chunk_size):
            end = min(start + self.relation_chunk_size, unique_edge_index.size(1))
            edge_chunk = unique_edge_index[:, start:end]
            flags = self.semantic_source_flags[start:end].to(
                device=projected["text"].device, dtype=projected["text"].dtype
            ) * 0.5

            def score_chunk(text_projection, visual_projection, edges, source_flags):
                left, right = edges
                text_left = text_projection.index_select(0, left)
                text_right = text_projection.index_select(0, right)
                visual_left = visual_projection.index_select(0, left)
                visual_right = visual_projection.index_select(0, right)
                relation_input = torch.cat(
                    [
                        text_left * text_right,
                        (text_left - text_right).abs(),
                        visual_left * visual_right,
                        (visual_left - visual_right).abs(),
                        source_flags,
                    ],
                    dim=-1,
                )
                return self.relation_scorer(relation_input)

            if self.training and torch.is_grad_enabled():
                values = checkpoint(
                    score_chunk,
                    projected["text"],
                    projected["visual"],
                    edge_chunk,
                    flags,
                    use_reentrant=False,
                )
            else:
                values = score_chunk(
                    projected["text"], projected["visual"], edge_chunk, flags
                )
            pieces.append(values)
        return torch.cat(pieces, dim=0)

    def _relative_edge_weight(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unique = self.semantic_unique_edge_index
        if unique.size(1) != scores.numel():
            raise RuntimeError("relative relation scores do not align with candidate graph")
        if unique.size(1) == 0:
            empty = scores.new_empty((0,))
            return empty, empty
        source, target = unique
        degree = self.semantic_candidate_degree.to(device=scores.device, dtype=scores.dtype)
        endpoint_sum = scores.new_zeros((degree.numel(),))
        endpoint_sum.index_add_(0, source, scores)
        endpoint_sum.index_add_(0, target, scores)
        endpoint_mean = endpoint_sum / degree.clamp_min(1.0)
        delta = scores - 0.5 * (endpoint_mean[source] + endpoint_mean[target])
        weight = 0.8 * (1.0 + 0.75 * torch.tanh(delta))
        normalized = weight * (degree[source] * degree[target]).clamp_min(1.0).rsqrt()
        directed = torch.cat((normalized, normalized), dim=0)
        return weight, directed

    def _details_summary(
        self,
        *,
        semantic_weight: torch.Tensor | None,
        semantic_states: dict[str, dict[str, torch.Tensor]],
        routes: dict[str, dict[str, dict[str, torch.Tensor]]],
        physical_corrections: dict[str, torch.Tensor],
        semantic_corrections: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        result = super()._details_summary(
            semantic_weight=semantic_weight,
            semantic_states=semantic_states,
            routes=routes,
            physical_corrections=physical_corrections,
            semantic_corrections=semantic_corrections,
        )
        for modality, states in semantic_states.items():
            scales = [float(states[f"scale{k}"].detach().cpu()) for k in (1, 2)]
            result["modalities"][modality]["calibration_scale_mean"] = scales
            result["modalities"][modality]["calibration_scale_below_one"] = [
                value < 0.999999 for value in scales
            ]
        return result

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if self.v7b_variant not in {"b3_relative_shared", "b4_relative_modality"}:
            self._collect_fixed_edge_details = bool(return_details)
            self._last_fixed_edge_scores = None
            output = super().forward(x, edge_index, return_details=return_details)
            output[4]["variant"] = self.v7b_variant
            if return_details and output[4].get("semantic_enabled"):
                if self._last_fixed_edge_scores is None:
                    raise RuntimeError("fixed-path edge diagnostics are missing the selected edge scores")
                weights = torch.sigmoid(self._last_fixed_edge_scores)
                flags = self.semantic_source_flags.to(device=weights.device)
                groups = {
                    "text_only": (flags[:, 0] > 0) & (flags[:, 1] == 0),
                    "visual_only": (flags[:, 0] == 0) & (flags[:, 1] > 0),
                    "both": (flags[:, 0] > 0) & (flags[:, 1] > 0),
                }
                group_summary = {
                    name: {"count": int(mask.sum().item()), "weight": self._quantile_summary(weights[mask])}
                    for name, mask in groups.items()
                    if bool(mask.any())
                }
                if self.v7b_variant == "b2_random_fixed":
                    group_summary = {"randomized": {"count": int(weights.numel()), "weight": self._quantile_summary(weights)}}
                edge_summary = self._quantile_summary(weights)
                if self.v7b_variant == "b0_v7a_original":
                    lower_threshold, upper_threshold = 0.005, 0.995
                else:
                    lower_threshold, upper_threshold = 0.205, 1.395
                if self.v7b_variant in {"b1_fixed_knn", "b2_random_fixed"}:
                    lower_fraction = upper_fraction = 0.0
                else:
                    lower_fraction = float((weights <= lower_threshold).float().mean().cpu()) if weights.numel() else 0.0
                    upper_fraction = float((weights >= upper_threshold).float().mean().cpu()) if weights.numel() else 0.0
                output[4]["details"]["edge_weights_by_modality"] = {
                    "text": edge_summary,
                    "visual": edge_summary,
                    "text_visual_correlation": 1.0,
                    "text_visual_mean_absolute_difference": 0.0,
                    "text_lower_saturation_fraction": lower_fraction,
                    "text_upper_saturation_fraction": upper_fraction,
                    "visual_lower_saturation_fraction": lower_fraction,
                    "visual_upper_saturation_fraction": upper_fraction,
                    "source_groups": group_summary,
                }
                if not self.training:
                    with torch.no_grad():
                        priors = {
                            modality: self.projectors[modality](features)
                            for modality, features in self._split_modalities(x).items()
                        }
                    active = output[4]["active_nodes"]
                    for modality in _MODALITIES:
                        correction_rms = output[4]["details"]["modalities"][modality]["semantic_contribution_rms"]
                        intrinsic_rms = float(self._rms(priors[modality], active).cpu())
                        output[4]["details"]["modalities"][modality]["semantic_contribution_to_intrinsic_rms"] = (
                            float(correction_rms / max(intrinsic_rms, 1.0e-12))
                        )
            self._collect_fixed_edge_details = False
            self._last_fixed_edge_scores = None
            return output
        return self._relative_forward(x, edge_index, return_details=return_details)

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        diagnostics = super().capture_gradient_diagnostics()
        if not self.v7b_relative:
            return diagnostics
        gradient = self.relation_scorer[-1].weight.grad
        for head, name in enumerate(("text", "visual")):
            row = None if gradient is None else gradient[head]
            norm = float(row.detach().float().norm().cpu()) if row is not None else 0.0
            diagnostics[f"relation_scorer.output_{name}"] = {
                "gradient_present": row is not None,
                "gradient_finite": bool(row is None or torch.isfinite(row).all()),
                "gradient_norm": norm,
                "gradient_nonzero": bool(norm > 0.0),
            }
        self._gradient_diagnostics = diagnostics
        return diagnostics

    def _relative_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        return_details: bool = False,
    ):
        if x.size(0) != self.num_nodes and self.semantic_candidate_fingerprint is not None:
            raise RuntimeError(
                "PSCE-MAG V7A is full-graph NC only: the global-ID semantic cache cannot be used with a sampled LP subgraph"
            )
        inputs = self._split_modalities(x)
        priors = {modality: self.projectors[modality](inputs[modality]) for modality in _MODALITIES}
        source, target, physical_norm, physical_active, input_self_loops = self._normalized_operator(
            edge_index, x.size(0), priors["text"].dtype
        )

        physical_bases: dict[str, torch.Tensor] = {}
        for modality in _MODALITIES:
            states: list[torch.Tensor] = []
            state = priors[modality]
            for _ in range(self.physical_hops):
                state = self._propagate(state, source, target, physical_norm)
                state = state * physical_active.to(state.dtype).unsqueeze(-1)
                states.append(state)
            physical_bases[modality] = torch.stack(
                [self._active_rms_norm(item, physical_active) for item in states], dim=0
            )
            del states, state

        alpha_physical = self._effective_alpha(self.alpha_raw, self.eps)
        outputs: dict[str, torch.Tensor] = {}
        physical_routes: dict[str, dict[str, torch.Tensor]] = {}
        physical_corrections: dict[str, torch.Tensor] = {}
        physical_balances: dict[str, torch.Tensor] = {}
        semantic_routes: dict[str, dict[str, torch.Tensor]] = {}
        semantic_corrections: dict[str, torch.Tensor] = {}
        semantic_states: dict[str, dict[str, torch.Tensor]] = {}
        semantic_weights: dict[str, torch.Tensor] | None = None

        semantic_active_run = self.variant != "a0_raw" and self._semantic_enabled
        edge_scores: torch.Tensor | None = None
        directed_weights: dict[str, torch.Tensor] | None = None
        projected_relations: dict[str, torch.Tensor] = {}
        if self.profile_forward and x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        relation_started = time.perf_counter()
        if semantic_active_run:
            if self.semantic_candidate_fingerprint is None:
                raise RuntimeError("V7A A1/A2 requires a validated semantic candidate cache")
            if self.semantic_candidate_degree.numel() != x.size(0):
                raise RuntimeError(
                    "global semantic candidate cache cannot be applied to a sampled LP subgraph"
                )
            projected_relations = {
                modality: self.relation_projections[modality](priors[modality])
                for modality in _MODALITIES
            }
            edge_scores = self._score_semantic_edges(
                projected_relations, self.semantic_unique_edge_index
            )
            if self.v7b_variant == "b3_relative_shared":
                shared_scores = edge_scores.mean(dim=-1)
                score_by_modality = {modality: shared_scores for modality in _MODALITIES}
            else:
                score_by_modality = {
                    "text": edge_scores[:, 0],
                    "visual": edge_scores[:, 1],
                }
            semantic_weights = {}
            directed_weights = {}
            if self.v7b_variant == "b3_relative_shared":
                shared_weight, shared_directed = self._relative_edge_weight(shared_scores)
                semantic_weights = {modality: shared_weight for modality in _MODALITIES}
                directed_weights = {modality: shared_directed for modality in _MODALITIES}
            else:
                for modality in _MODALITIES:
                    weight, directed = self._relative_edge_weight(score_by_modality[modality])
                    semantic_weights[modality] = weight
                    directed_weights[modality] = directed
            if self.profile_forward and x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            if self.profile_forward:
                self._last_timing["relation_scoring_seconds"] = (
                    time.perf_counter() - relation_started
                )
        elif self.profile_forward:
            self._last_timing["relation_scoring_seconds"] = 0.0
        if self.profile_forward:
            self._last_timing["semantic_propagation_seconds"] = 0.0

        for modality_index, modality in enumerate(_MODALITIES):
            prior = priors[modality]
            _, physical_expert_values = self._expert_profiles(
                alpha_physical,
                physical_bases[modality],
                self.experts,
                self.expert_feature_scale,
            )
            route = self._physical_route(modality_index, prior, physical_active)
            route["active"] = physical_active
            physical_routes[modality] = route
            physical_mixture = torch.einsum(
                "nm,mnd->nd", route["route_weights"], physical_expert_values
            )
            physical_correction = route["strength"].unsqueeze(-1) * physical_mixture
            physical_corrections[modality] = physical_correction
            physical_balances[modality] = route["balance"]
            outputs[modality] = prior + physical_correction

            if semantic_active_run:
                assert edge_scores is not None
                assert semantic_weights is not None and directed_weights is not None
                semantic_weight = semantic_weights[modality]
                modality_directed_weights = directed_weights[modality]
                semantic_started = time.perf_counter()
                if self.profile_forward and prior.device.type == "cuda":
                    torch.cuda.synchronize(prior.device)
                sem_state1_raw = chunked_weighted_spmm(
                    prior,
                    self.semantic_candidate_edge_index,
                    modality_directed_weights,
                    self.propagation_chunk_size,
                )
                sem_active1, sem_active2 = self._semantic_active_masks(
                    source, target, sem_state1_raw, physical_active
                )
                sem_state1_raw = sem_state1_raw * sem_active1.to(prior.dtype).unsqueeze(-1)
                sem_state1, sem_scale1, sem_rms1, raw_rms1 = self._calibrate_semantic(
                    sem_state1_raw, physical_bases[modality][0], sem_active1
                )
                sem_state2_raw = self._propagate(sem_state1, source, target, physical_norm)
                sem_state2_raw = sem_state2_raw * sem_active2.to(prior.dtype).unsqueeze(-1)
                sem_state2, sem_scale2, sem_rms2, raw_rms2 = self._calibrate_semantic(
                    sem_state2_raw, physical_bases[modality][1], sem_active2
                )
                if self.profile_forward and prior.device.type == "cuda":
                    torch.cuda.synchronize(prior.device)
                if self.profile_forward:
                    self._last_timing["semantic_propagation_seconds"] += (
                        time.perf_counter() - semantic_started
                    )
                sem_states_for_profiles = torch.stack([sem_state1, sem_state2], dim=0)
                alpha_semantic = self._effective_alpha(
                    self.semantic_coefficients_raw, self.eps
                )
                _, sem_expert_values = self._expert_profiles(
                    alpha_semantic,
                    sem_states_for_profiles,
                    self.experts,
                    self.expert_feature_scale,
                )
                sem_route = self._semantic_route(
                    modality_index,
                    prior,
                    physical_bases[modality][0],
                    sem_state1,
                    physical_bases[modality][0],
                    sem_active1,
                )
                semantic_routes[modality] = sem_route
                sem_mixture = torch.einsum(
                    "nm,mnd->nd", sem_route["route_weights"], sem_expert_values
                )
                sem_correction = sem_route["strength"].unsqueeze(-1) * sem_mixture
                semantic_corrections[modality] = sem_correction
                outputs[modality] = outputs[modality] + sem_correction
                semantic_states[modality] = {
                    "state1": sem_state1,
                    "state2": sem_state2,
                    "raw1": physical_bases[modality][0],
                    "raw2": physical_bases[modality][1],
                    "active1": sem_active1,
                    "active2": sem_active2,
                    "scale1": sem_scale1,
                    "scale2": sem_scale2,
                    "rms1": sem_rms1,
                    "rms2": sem_rms2,
                    "raw_rms1": raw_rms1,
                    "raw_rms2": raw_rms2,
                }

        physical_balance = 0.5 * (physical_balances["text"] + physical_balances["visual"])
        if semantic_active_run:
            semantic_balance = 0.5 * (
                semantic_routes["text"]["balance"] + semantic_routes["visual"]["balance"]
            )
            balance_loss = 0.5 * (physical_balance + semantic_balance)
        else:
            semantic_balance = physical_balance.new_zeros(())
            balance_loss = physical_balance
        aux_loss = self.balance_weight * balance_loss
        fused = torch.cat([outputs["text"], outputs["visual"]], dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )
        info: dict[str, Any] = {
            "variant": self.variant,
            "active_nodes": physical_active.detach(),
            "input_self_loops_removed": input_self_loops,
            "balance_loss": balance_loss.detach(),
            "physical_balance_loss": physical_balance.detach(),
            "semantic_balance_loss": semantic_balance.detach(),
            "semantic_enabled": semantic_active_run,
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "variant": self.v7b_variant,
            "modalities": {
                modality: {
                    "physical_strength_mean": physical_routes[modality]["strength"].detach().mean(),
                    **(
                        {"semantic_strength_mean": semantic_routes[modality]["strength"].detach().mean()}
                        if semantic_active_run
                        else {}
                    ),
                }
                for modality in _MODALITIES
            },
        }
        if return_details:
            averaged_weight = (
                None
                if semantic_weights is None
                else 0.5 * (semantic_weights["text"] + semantic_weights["visual"])
            )
            details = self._details_summary(
                semantic_weight=averaged_weight,
                semantic_states=semantic_states,
                routes={"physical": physical_routes, "semantic": semantic_routes},
                physical_corrections=physical_corrections,
                semantic_corrections=semantic_corrections,
            )
            if semantic_weights is not None:
                text_weight = semantic_weights["text"]
                visual_weight = semantic_weights["visual"]
                if text_weight.numel() > 1 and visual_weight.numel() > 1:
                    corr = torch.corrcoef(torch.stack([text_weight.detach().float(), visual_weight.detach().float()]))[0, 1]
                    correlation = float(corr.cpu()) if torch.isfinite(corr) else None
                else:
                    correlation = None
                edge_diag = {
                    modality: self._quantile_summary(semantic_weights[modality])
                    for modality in _MODALITIES
                }
                edge_diag.update({
                    "text_visual_correlation": correlation,
                    "text_visual_mean_absolute_difference": float((text_weight - visual_weight).abs().mean().detach().cpu()) if text_weight.numel() else 0.0,
                    "text_lower_saturation_fraction": float((text_weight <= 0.205).float().mean().detach().cpu()) if text_weight.numel() else 0.0,
                    "text_upper_saturation_fraction": float((text_weight >= 1.395).float().mean().detach().cpu()) if text_weight.numel() else 0.0,
                    "visual_lower_saturation_fraction": float((visual_weight <= 0.205).float().mean().detach().cpu()) if visual_weight.numel() else 0.0,
                    "visual_upper_saturation_fraction": float((visual_weight >= 1.395).float().mean().detach().cpu()) if visual_weight.numel() else 0.0,
                })
                flags = self.semantic_source_flags.to(device=text_weight.device)
                categories = {
                    "text_only": (flags[:, 0] > 0) & (flags[:, 1] == 0),
                    "visual_only": (flags[:, 0] == 0) & (flags[:, 1] > 0),
                    "both": (flags[:, 0] > 0) & (flags[:, 1] > 0),
                }
                edge_diag["source_groups"] = {
                    group: {
                        "count": int(mask.sum().item()),
                        "text": self._quantile_summary(text_weight[mask]),
                        "visual": self._quantile_summary(visual_weight[mask]),
                    }
                    for group, mask in categories.items()
                }
                details["edge_weights_by_modality"] = edge_diag
                details["semantic_v7b_fingerprint"] = self.semantic_v7b_fingerprint
                for modality in _MODALITIES:
                    states = semantic_states[modality]
                    details["modalities"][modality]["semantic_contribution_to_intrinsic_rms"] = float(
                        details["modalities"][modality]["semantic_contribution_rms"]
                        / self._rms(priors[modality], physical_active).clamp_min(1.0e-12).cpu()
                    )
                    details["modalities"][modality]["calibration_scale_mean"] = [
                        float(states[f"scale{k}"].detach().cpu()) for k in (1, 2)
                    ]
                    details["modalities"][modality]["calibration_scale_below_one"] = [
                        bool(float(states[f"scale{k}"].detach().cpu()) < 0.999999) for k in (1, 2)
                    ]
            info["details"] = details
        return z, None, None, aux_loss, info
