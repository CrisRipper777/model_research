from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import add_remaining_self_loops


_MODALITIES = ("text", "visual")
_VARIANTS = {"base", "global_mix", "adaptive_mix"}
_INTERVENTIONS = {"normal", "globalize_router"}
_PROPAGATION_CHUNK_THRESHOLD = 200_000
_PROPAGATION_CHUNK_SIZE = 32_768
_ACTIVATION_CHECKPOINT_NODE_THRESHOLD = 50_000
_FEATURE_PROJECTION_CHUNK_SIZE = 8_192
_PAIR_CHUNK_SIZE = 8_192
_MODE_CHUNK_SIZE = 8_192


class _IntrinsicProjector(nn.Module):
    """C1 RGD modality projector, kept numerically and structurally identical."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(
            self.linear2(self.dropout(F.gelu(self.linear1(x)))) + self.skip(x)
        )


class _StreamingRawGPR(torch.autograd.Function):
    """Memory-bounded raw GPR mixture with the exact analytic backward."""

    @staticmethod
    def _propagate(state, src, dst, norm):
        output = torch.zeros_like(state)
        if src.numel() <= _PROPAGATION_CHUNK_THRESHOLD:
            output.index_add_(0, dst, state[src] * norm.unsqueeze(-1))
            return output
        for start in range(0, src.numel(), _PROPAGATION_CHUNK_SIZE):
            stop = min(start + _PROPAGATION_CHUNK_SIZE, src.numel())
            output.index_add_(
                0,
                dst[start:stop],
                state[src[start:stop]] * norm[start:stop].unsqueeze(-1),
            )
        return output

    @staticmethod
    def _propagate_transpose(state, src, dst, norm):
        output = torch.zeros_like(state)
        if src.numel() <= _PROPAGATION_CHUNK_THRESHOLD:
            output.index_add_(0, src, state[dst] * norm.unsqueeze(-1))
            return output
        for start in range(0, src.numel(), _PROPAGATION_CHUNK_SIZE):
            stop = min(start + _PROPAGATION_CHUNK_SIZE, src.numel())
            output.index_add_(
                0,
                src[start:stop],
                state[dst[start:stop]] * norm[start:stop].unsqueeze(-1),
            )
        return output

    @staticmethod
    def forward(ctx, prior, coefficients, src, dst, norm):
        ctx.save_for_backward(prior, coefficients, src, dst, norm)
        state = prior
        output = prior * coefficients[0]
        for order in range(1, coefficients.numel()):
            state = _StreamingRawGPR._propagate(state, src, dst, norm)
            output.add_(state, alpha=float(coefficients[order].detach().item()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        prior, coefficients, src, dst, norm = ctx.saved_tensors
        coefficient_grad = torch.empty_like(coefficients)
        state = prior
        for order in range(coefficients.numel()):
            coefficient_grad[order] = (grad_output * state).sum()
            if order + 1 < coefficients.numel():
                state = _StreamingRawGPR._propagate(state, src, dst, norm)

        grad_state = coefficients[-1] * grad_output
        for order in range(coefficients.numel() - 1, 0, -1):
            grad_state = (
                coefficients[order - 1] * grad_output
                + _StreamingRawGPR._propagate_transpose(grad_state, src, dst, norm)
            )
        return grad_state, coefficient_grad, None, None, None


class Model(nn.Module):
    """Raw-GPR RGD with Preserve/Self/Paired interaction-mode mixtures."""

    requires_full_lp_sampler_depth = True
    supports_cpu_feature_staging = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("imosi_mag_v0 requires positive text and visual dimensions")
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "imosi_mag_v0 requires exact [text, visual] widths: "
                f"input_dim={self.input_dim}, text_dim={self.text_dim}, "
                f"visual_dim={self.visual_dim}"
            )

        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.num_layers = int(model_cfg.get("num_layers", 3))
        self.dropout_p = float(model_cfg.get("dropout", 0.2))
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.diffusion_add_self_loops = bool(
            model_cfg.get("diffusion_add_self_loops", True)
        )
        self.variant = str(model_cfg.get("variant", "base")).lower()
        self.response_rank = int(model_cfg.get("response_rank", 64))
        self.router_rank = int(model_cfg.get("router_rank", 64))
        initial_mode_probs = tuple(
            float(value) for value in model_cfg.get("initial_mode_probs", [0.6, 0.2, 0.2])
        )
        self.model_seed = int(cfg.get("seed", 42))
        self._enable_activation_checkpointing = True
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
        if self.hidden_dim != 256 or self.num_layers != 3:
            raise ValueError("IMoSI-F0 fixes hidden_dim=256 and num_layers=3")
        if self.response_rank != 64 or self.router_rank != 64:
            raise ValueError("IMoSI-F0 fixes response_rank=64 and router_rank=64")
        if len(initial_mode_probs) != 3 or any(value <= 0 for value in initial_mode_probs):
            raise ValueError("initial_mode_probs must contain three positive probabilities")
        if abs(sum(initial_mode_probs) - 1.0) > 1e-8:
            raise ValueError("initial_mode_probs must sum to one")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        self.out_dim = self.hidden_dim

        # Match C1/D0 module creation order through the entire RGD/fusion core.
        self.projectors = nn.ModuleDict(
            {
                "text": _IntrinsicProjector(
                    self.text_dim, self.hidden_dim, self.dropout_p
                ),
                "visual": _IntrinsicProjector(
                    self.visual_dim, self.hidden_dim, self.dropout_p
                ),
            }
        )
        self.delta_c_raw = nn.Parameter(torch.zeros(self.num_layers + 1))
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(self.dropout_p)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        self.global_prior_restart = float(model_cfg.get("global_prior_restart", 0.15))
        self.global_prior_order = int(model_cfg.get("global_prior_order", 2))
        c_prior = self._make_global_prior()
        self.register_buffer("c_prior", c_prior.clone(), persistent=True)

        # Every variant has exactly the same pair module names and initialization.
        self.pair_norm = nn.LayerNorm(4 * self.hidden_dim)
        self.pair_down = nn.Linear(4 * self.hidden_dim, self.response_rank)
        self.pair_up = nn.Linear(self.response_rank, self.hidden_dim)
        nn.init.zeros_(self.pair_up.weight)
        nn.init.zeros_(self.pair_up.bias)
        permutation = self._make_derangement(
            int(data_info.get("num_nodes", 0)), self.model_seed + 73000
        )
        self.register_buffer("pair_shuffle_permutation", permutation, persistent=True)

        # F0-only parameters use a private CPU RNG stream. Restoring the caller's
        # RNG state preserves the downstream NC classifier initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.model_seed + 91000)
            self.self_up = nn.Linear(self.response_rank, self.hidden_dim)
            nn.init.zeros_(self.self_up.weight)
            nn.init.zeros_(self.self_up.bias)
            mode_probs = torch.tensor(initial_mode_probs, dtype=torch.float32)
            self.mode_base_logits = nn.Parameter(mode_probs.log().repeat(2, 1))
            self.router_norm = nn.LayerNorm(6 * self.hidden_dim)
            self.router_down = nn.Linear(6 * self.hidden_dim, self.router_rank)
            self.router_out = nn.Linear(self.router_rank, 3)
            nn.init.zeros_(self.router_out.weight)
            nn.init.zeros_(self.router_out.bias)

    def _make_global_prior(self) -> torch.Tensor:
        prior = torch.zeros(self.num_layers + 1, dtype=torch.float32)
        restart = self.global_prior_restart
        target = self.global_prior_order
        if self.num_layers >= target:
            for order in range(target):
                prior[order] = restart * (1.0 - restart) ** order
            prior[target] = (1.0 - restart) ** target
        else:
            for order in range(self.num_layers):
                prior[order] = restart * (1.0 - restart) ** order
            prior[self.num_layers] = (1.0 - restart) ** self.num_layers
        return prior

    @staticmethod
    def _make_derangement(num_nodes: int, seed: int) -> torch.Tensor:
        if num_nodes < 0:
            raise ValueError("num_nodes must be nonnegative")
        if num_nodes <= 1:
            return torch.arange(num_nodes, dtype=torch.long)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        order = torch.randperm(num_nodes, generator=generator)
        permutation = torch.empty(num_nodes, dtype=torch.long)
        permutation[order] = order.roll(-1)
        return permutation

    def effective_coefficients(self) -> torch.Tensor:
        return self.c_prior + self.delta_c_raw

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"imosi_mag_v0 expected x shape [nodes, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )
        if x.dtype == torch.long:
            x = x.float()
        return {
            "text": x[:, : self.text_dim],
            "visual": x[:, self.text_dim : self.text_dim + self.visual_dim],
        }

    def _project_modality(
        self,
        modality_input: torch.Tensor,
        modality: str,
        checkpoint_activations: bool,
    ) -> torch.Tensor:
        projector = self.projectors[modality]
        parameter_device = next(projector.parameters()).device
        if modality_input.device == parameter_device:
            if checkpoint_activations:
                return checkpoint(projector, modality_input, use_reentrant=False)
            return projector(modality_input)

        output_chunks = []
        device_anchor = torch.empty(0, device=parameter_device)
        for start in range(0, modality_input.size(0), _FEATURE_PROJECTION_CHUNK_SIZE):
            chunk = modality_input[start : start + _FEATURE_PROJECTION_CHUNK_SIZE]

            def project_chunk(value: torch.Tensor, _device_anchor: torch.Tensor):
                return projector(value.to(parameter_device))

            if checkpoint_activations:
                projected = checkpoint(
                    project_chunk, chunk, device_anchor, use_reentrant=False
                )
            else:
                projected = project_chunk(chunk, device_anchor)
            output_chunks.append(projected)
        return torch.cat(output_chunks, dim=0)

    def _normalized_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("imosi_mag_v0 requires edge_index")
        edge_index = edge_index.to(dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.diffusion_add_self_loops:
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        src, dst = edge_index
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, dst, torch.ones(dst.numel(), dtype=dtype, device=dst.device))
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        return src, dst, inv_sqrt[src] * inv_sqrt[dst]

    @staticmethod
    def _propagate(
        state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        if src.numel() <= _PROPAGATION_CHUNK_THRESHOLD:
            messages = state[src] * norm.unsqueeze(-1)
            return torch.zeros_like(state).index_add(0, dst, messages)
        output = torch.zeros_like(state)
        for start in range(0, src.numel(), _PROPAGATION_CHUNK_SIZE):
            stop = min(start + _PROPAGATION_CHUNK_SIZE, src.numel())
            messages = state[src[start:stop]] * norm[start:stop].unsqueeze(-1)
            output.index_add_(0, dst[start:stop], messages)
        return output

    def _raw_states(
        self,
        prior: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        norm: torch.Tensor,
    ) -> list[torch.Tensor]:
        states = [prior]
        state = prior
        for _ in range(self.num_layers):
            state = self._propagate(state, src, dst, norm)
            states.append(state)
        return states

    def _fusion_residual(self, fused: torch.Tensor) -> torch.Tensor:
        return self.fusion_linear2(
            self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
        )

    @staticmethod
    def _pair_features(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [target, source, torch.abs(target - source), target * source], dim=-1
        )

    def _pair_components(
        self, target: torch.Tensor, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._pair_features(target, source)
        hidden = F.gelu(self.pair_down(self.pair_norm(features)))
        return hidden, self.pair_up(hidden)

    def _pair_response(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        *,
        checkpoint_activations: bool,
        return_hidden: bool,
        collect_hidden: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        residual_chunks = []
        hidden_chunks = []
        hidden_square_sum = target.new_zeros(())
        hidden_count = 0
        for start in range(0, target.size(0), _PAIR_CHUNK_SIZE):
            target_chunk = target[start : start + _PAIR_CHUNK_SIZE]
            source_chunk = source[start : start + _PAIR_CHUNK_SIZE]
            if return_hidden:
                hidden, residual = self._pair_components(target_chunk, source_chunk)
                hidden_square_sum = hidden_square_sum + hidden.square().sum()
                hidden_count += hidden.numel()
                if collect_hidden:
                    hidden_chunks.append(hidden)
            elif checkpoint_activations:
                residual = checkpoint(
                    lambda t, s: self._pair_components(t, s)[1],
                    target_chunk,
                    source_chunk,
                    use_reentrant=False,
                )
            else:
                _, residual = self._pair_components(target_chunk, source_chunk)
            residual_chunks.append(residual)
        residual = torch.cat(residual_chunks, dim=0)
        hidden = torch.cat(hidden_chunks, dim=0) if collect_hidden else None
        hidden_rms = (
            (hidden_square_sum / max(hidden_count, 1)).sqrt()
            if return_hidden
            else target.new_zeros(())
        )
        return residual, hidden, hidden_rms

    def _validate_permutation(
        self, permutation: torch.Tensor | None, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        if permutation is None:
            raise ValueError("source_node_shuffle requires source_node_permutation")
        permutation = permutation.to(device=device, dtype=torch.long)
        expected = torch.arange(num_nodes, device=device)
        if permutation.shape != expected.shape or not torch.equal(
            permutation.sort().values, expected
        ):
            raise ValueError("source_node_permutation must be a node permutation")
        return permutation

    def _router_features(
        self,
        target: torch.Tensor,
        prior: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                target,
                prior,
                target - prior,
                other,
                torch.abs(target - other),
                target * other,
            ],
            dim=-1,
        )

    def _interaction_chunk(
        self,
        target: torch.Tensor,
        prior: torch.Tensor,
        other: torch.Tensor,
        modality_index: int,
        *,
        adaptive_router: bool,
    ):
        self_features = self._pair_features(target, target)
        self_hidden = F.gelu(self.pair_down(self.pair_norm(self_features)))
        delta_self = self.self_up(self_hidden)
        del self_features, self_hidden

        pair_features = self._pair_features(target, other)
        pair_hidden = F.gelu(self.pair_down(self.pair_norm(pair_features)))
        delta_pair = self.pair_up(pair_hidden)
        del pair_features, pair_hidden

        if adaptive_router:
            router_features = self._router_features(target, prior, other)
            node_delta_logits = self.router_out(
                F.gelu(self.router_down(self.router_norm(router_features)))
            )
            del router_features
        else:
            node_delta_logits = target.new_zeros((target.size(0), 3))
        base_logits = self.mode_base_logits[modality_index]
        logits = base_logits.unsqueeze(0) + node_delta_logits
        mode_probs = torch.softmax(logits, dim=-1)
        refined = (
            target
            + mode_probs[:, 1:2] * delta_self
            + mode_probs[:, 2:3] * delta_pair
        )
        return refined, mode_probs, node_delta_logits, delta_self, delta_pair

    @staticmethod
    def _new_mode_stats(reference: torch.Tensor) -> dict[str, Any]:
        return {
            "target_sq": reference.new_zeros(()),
            "self_sq": reference.new_zeros(()),
            "pair_sq": reference.new_zeros(()),
            "self_effective_sq": reference.new_zeros(()),
            "pair_effective_sq": reference.new_zeros(()),
            "self_cosine_sum": reference.new_zeros(()),
            "pair_cosine_sum": reference.new_zeros(()),
            "node_delta_sq": reference.new_zeros(()),
            "route_global_diff_sq": reference.new_zeros(()),
            "numel": 0,
            "num_nodes": 0,
        }

    def _accumulate_mode_stats(
        self,
        stats: dict[str, Any],
        target: torch.Tensor,
        delta_self: torch.Tensor,
        delta_pair: torch.Tensor,
        mode_probs: torch.Tensor,
        node_delta_logits: torch.Tensor,
        global_probs: torch.Tensor,
    ) -> None:
        # Diagnostics are detached summaries; no full-graph expert tensors are kept.
        target_f = target.detach().float()
        self_f = delta_self.detach().float()
        pair_f = delta_pair.detach().float()
        probs_f = mode_probs.detach().float()
        self_effective = probs_f[:, 1:2] * self_f
        pair_effective = probs_f[:, 2:3] * pair_f
        stats["target_sq"] += target_f.square().sum()
        stats["self_sq"] += self_f.square().sum()
        stats["pair_sq"] += pair_f.square().sum()
        stats["self_effective_sq"] += self_effective.square().sum()
        stats["pair_effective_sq"] += pair_effective.square().sum()
        stats["self_cosine_sum"] += F.cosine_similarity(
            self_effective, target_f, dim=-1
        ).sum()
        stats["pair_cosine_sum"] += F.cosine_similarity(
            pair_effective, target_f, dim=-1
        ).sum()
        stats["node_delta_sq"] += node_delta_logits.detach().float().square().sum()
        stats["route_global_diff_sq"] += (
            probs_f - global_probs.detach().float().unsqueeze(0)
        ).square().sum()
        stats["numel"] += target.numel()
        stats["num_nodes"] += target.size(0)

    @staticmethod
    def _finish_mode_stats(stats: dict[str, Any]) -> dict[str, torch.Tensor]:
        count = max(int(stats["numel"]), 1)
        nodes = max(int(stats["num_nodes"]), 1)
        target_rms = (stats["target_sq"] / count).sqrt()
        delta_self_rms = (stats["self_sq"] / count).sqrt()
        delta_pair_rms = (stats["pair_sq"] / count).sqrt()
        effective_self_rms = (stats["self_effective_sq"] / count).sqrt()
        effective_pair_rms = (stats["pair_effective_sq"] / count).sqrt()
        return {
            "rms_target": target_rms,
            "rms_delta_self": delta_self_rms,
            "rms_delta_pair": delta_pair_rms,
            "rms_effective_self": effective_self_rms,
            "rms_effective_pair": effective_pair_rms,
            "effective_self_to_target_rms": effective_self_rms / target_rms.clamp_min(1e-12),
            "effective_pair_to_target_rms": effective_pair_rms / target_rms.clamp_min(1e-12),
            "cosine_effective_self_target": stats["self_cosine_sum"] / nodes,
            "cosine_effective_pair_target": stats["pair_cosine_sum"] / nodes,
            "node_delta_logits_rms": (stats["node_delta_sq"] / (3 * nodes)).sqrt(),
            "route_global_difference_rms": (stats["route_global_diff_sq"] / (3 * nodes)).sqrt(),
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        intervention: str = "normal",
        return_details: bool = False,
    ):
        intervention = str(intervention).lower()
        if intervention not in _INTERVENTIONS:
            raise ValueError(f"unsupported IMoSI-F0 intervention: {intervention}")
        if x.dim() != 2:
            raise ValueError("imosi_mag_v0 expects x with shape [nodes, features]")
        checkpoint_activations = (
            self._enable_activation_checkpointing
            and self.training
            and x.size(0) >= _ACTIVATION_CHECKPOINT_NODE_THRESHOLD
        )
        inputs = self._split_modalities(x)
        priors = {
            modality: self._project_modality(
                inputs[modality], modality, checkpoint_activations
            )
            for modality in _MODALITIES
        }
        src, dst, norm = self._normalized_operator(
            edge_index, x.size(0), inputs["text"].dtype
        )
        coefficients = self.effective_coefficients()
        streaming_gpr = x.size(0) >= _ACTIVATION_CHECKPOINT_NODE_THRESHOLD

        composed: dict[str, torch.Tensor] = {}
        modality_details: dict[str, dict[str, torch.Tensor]] = {}
        for modality in _MODALITIES:
            if streaming_gpr:
                proposal = _StreamingRawGPR.apply(
                    priors[modality], coefficients, src, dst, norm
                )
                embedding = F.layer_norm(proposal, (self.hidden_dim,), eps=self.eps)
                if return_details:
                    modality_details[modality] = {
                        "prior": priors[modality],
                        "proposal": proposal,
                        "embedding": embedding,
                    }
            else:
                states = (
                    checkpoint(
                        self._raw_states,
                        priors[modality],
                        src,
                        dst,
                        norm,
                        use_reentrant=False,
                    )
                    if checkpoint_activations
                    else self._raw_states(priors[modality], src, dst, norm)
                )
                proposal = sum(
                    coefficients[order] * states[order]
                    for order in range(self.num_layers + 1)
                )
                embedding = F.layer_norm(proposal, (self.hidden_dim,), eps=self.eps)
                if return_details:
                    modality_details[modality] = {
                        "prior": priors[modality],
                        "raw_states": torch.stack(states, dim=1),
                        "proposal": proposal,
                        "embedding": embedding,
                    }
            composed[modality] = embedding

        refined: dict[str, torch.Tensor] = {}
        routing_details: dict[str, dict[str, Any]] = {}
        modality_index = {"text": 0, "visual": 1}
        mode_base_probs = torch.softmax(self.mode_base_logits, dim=-1)
        adaptive_router = (
            self.variant == "adaptive_mix" and intervention == "normal"
        )
        for target_name, other_name in (("text", "visual"), ("visual", "text")):
            target = composed[target_name]
            if self.variant == "base":
                refined[target_name] = target
                if return_details:
                    probs = target.new_zeros((target.size(0), 3))
                    probs[:, 0] = 1.0
                    routing_details[target_name] = {
                        "mode_probs": probs,
                        "node_delta_logits": target.new_zeros((target.size(0), 3)),
                        "mode_base_probs": mode_base_probs[modality_index[target_name]],
                        "expert_contribution": self._finish_mode_stats(self._new_mode_stats(target)),
                    }
                if return_details:
                    modality_details[target_name]["refined_embedding"] = target
                    modality_details[target_name]["mode_probs"] = routing_details[target_name]["mode_probs"]
                continue

            other = composed[other_name]
            prior = priors[target_name]
            global_probs = mode_base_probs[modality_index[target_name]]
            refined_chunks = []
            route_chunks = []
            node_delta_chunks = []
            stats = self._new_mode_stats(target)
            for chunk_start in range(0, target.size(0), _MODE_CHUNK_SIZE):
                chunk_stop = min(chunk_start + _MODE_CHUNK_SIZE, target.size(0))
                target_chunk = target[chunk_start:chunk_stop]
                prior_chunk = prior[chunk_start:chunk_stop]
                other_chunk = other[chunk_start:chunk_stop]
                use_checkpoint = checkpoint_activations and not return_details
                if use_checkpoint:
                    refined_chunk = checkpoint(
                        lambda t, p, o, mi=modality_index[target_name]: self._interaction_chunk(
                            t, p, o, mi, adaptive_router=adaptive_router
                        )[0],
                        target_chunk,
                        prior_chunk,
                        other_chunk,
                        use_reentrant=False,
                    )
                    refined_chunks.append(refined_chunk)
                else:
                    result = self._interaction_chunk(
                        target_chunk,
                        prior_chunk,
                        other_chunk,
                        modality_index[target_name],
                        adaptive_router=adaptive_router,
                    )
                    refined_chunk, probs, node_delta, delta_self, delta_pair = result
                    refined_chunks.append(refined_chunk)
                    if return_details:
                        route_chunks.append(probs)
                        node_delta_chunks.append(node_delta)
                        self._accumulate_mode_stats(
                            stats, target_chunk, delta_self, delta_pair, probs,
                            node_delta, global_probs,
                        )
            refined[target_name] = torch.cat(refined_chunks, dim=0)
            if return_details:
                routes = torch.cat(route_chunks, dim=0)
                node_deltas = torch.cat(node_delta_chunks, dim=0)
                routing_details[target_name] = {
                    "mode_probs": routes,
                    "node_delta_logits": node_deltas,
                    "mode_base_probs": global_probs,
                    "expert_contribution": self._finish_mode_stats(stats),
                }
                modality_details[target_name]["refined_embedding"] = refined[target_name]
                modality_details[target_name]["mode_probs"] = routes
                modality_details[target_name]["node_delta_logits"] = node_deltas

        # Late fusion remains the unchanged node-aligned text/visual fusion.
        fused = torch.cat([refined["text"], refined["visual"]], dim=-1)
        residual_fusion = (
            checkpoint(self._fusion_residual, fused, use_reentrant=False)
            if checkpoint_activations
            else self._fusion_residual(fused)
        )
        z = self.fusion_norm(self.fusion_skip(fused) + residual_fusion)
        aux_loss = priors["text"].new_zeros(())
        permutation_fixed_points = int(
            (self.pair_shuffle_permutation == torch.arange(
                self.pair_shuffle_permutation.numel(),
                device=self.pair_shuffle_permutation.device,
            )).sum().item()
        ) if self.pair_shuffle_permutation.numel() else 0
        info: dict[str, Any] = {
            "variant": self.variant,
            "intervention": intervention,
            "coefficients": coefficients.detach(),
            "mode_base_probs": mode_base_probs.detach(),
            "pair_shuffle_fixed_points": permutation_fixed_points,
        }
        if return_details:
            info["details"] = {
                "modalities": modality_details,
                "routing": routing_details,
                "coefficients": coefficients,
                "late_fusion_input": fused,
                "self_up_weight_norm": self.self_up.weight.norm(),
                "pair_up_weight_norm": self.pair_up.weight.norm(),
                "pair_down_weight_norm": self.pair_down.weight.norm(),
            }
        return z, None, None, aux_loss, info

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
        z, _, _, _, _ = self.forward(
            x
            if self.supports_cpu_feature_staging
            and x.size(0) >= _ACTIVATION_CHECKPOINT_NODE_THRESHOLD
            else x.to(device),
            None if edge_index is None else edge_index.to(device),
        )
        return z.detach().cpu()
