"""MR-CGE V1.3 view collaboration and sender-private cross variants."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import remove_self_loops

from src.data.semantic_candidates_v7b import load_semantic_candidate_cache_v7b
from src.models.cg_cge_v1 import receiver_sparse_attention
from src.models.psce_mag_v7a import (
    _MODALITIES,
    _IntrinsicProjector,
    chunked_weighted_spmm,
)
from src.models.rmcge_v2 import build_modality_knn_operators


CONDITIONS = ("S_repaired", "F_sinit", "M_repaired", "M_anchor")


def capture_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Capture CPU and selected CUDA RNG state for paired filter dropout."""
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu_state, cuda_state


@contextmanager
def replay_rng_state(state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device):
    """Replay a prior call's RNG stream locally, then restore the caller's stream."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.random.set_rng_state(state[0])
        if device.type == "cuda" and state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield


def _run_bottleneck(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    if module.training and torch.is_grad_enabled():
        return checkpoint(module, value, use_reentrant=False, preserve_rng_state=True)
    return module(value)


class Bottleneck(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedExpertBank(nn.Module):
    """One expert object bank used for every private/collaborative graph view."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float, eta: float):
        super().__init__()
        self.eta = float(eta)
        self.theta = nn.Parameter(
            torch.tensor(
                [
                    [0.62, -0.27, 0.18, -0.09],
                    [-0.31, 0.57, -0.11, 0.19],
                    [0.22, 0.16, -0.53, 0.29],
                    [-0.18, -0.23, 0.37, 0.46],
                ],
                dtype=torch.float32,
            )
        )
        self.experts = nn.ModuleList(
            [Bottleneck(hidden_dim, bottleneck_dim, dropout) for _ in range(4)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])

    def coefficients(self, max_order: int, dtype: torch.dtype) -> torch.Tensor:
        order_mask = torch.arange(4, device=self.theta.device) <= int(max_order)
        raw = self.theta * order_mask.to(self.theta.dtype)[None, :]
        return (raw / (raw.abs().sum(dim=-1, keepdim=True) + 1.0e-8)).to(dtype)

    def forward(
        self,
        basis: list[torch.Tensor],
        support: torch.Tensor,
        max_order: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficients = self.coefficients(max_order, basis[0].dtype)
        stacked = torch.stack(basis[: max_order + 1], dim=0)
        masked_coefficients = coefficients[:, : max_order + 1]
        filtered = torch.einsum("kl,lnd->knd", masked_coefficients, stacked)
        outputs = []
        residuals = []
        mask = support.to(filtered.dtype).unsqueeze(-1)
        for index, expert in enumerate(self.experts):
            residual = self.eta * _run_bottleneck(expert, filtered[index])
            value = self.norms[index](filtered[index] + residual)
            outputs.append(value * mask)
            residuals.append(residual * mask)
        return torch.stack(outputs), filtered, coefficients, torch.stack(residuals)


class SingleFilter(nn.Module):
    """S/F common single-filter path; S is fixed and F learns from equal weights."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float, eta: float, learnable: bool):
        super().__init__()
        initial = torch.ones(4, dtype=torch.float32)
        if learnable:
            self.theta = nn.Parameter(initial)
        else:
            self.register_buffer("theta", initial, persistent=True)
        self.transform = Bottleneck(hidden_dim, bottleneck_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.eta = float(eta)
        self.learnable = bool(learnable)

    def coefficients(self, max_order: int, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.arange(4, device=self.theta.device) <= int(max_order)
        raw = self.theta * mask.to(self.theta.dtype)
        if self.learnable:
            raw = raw / (raw.abs().sum() + 1.0e-8)
        else:
            raw = raw / float(max_order + 1)
        return raw.to(dtype)

    def forward(self, basis: list[torch.Tensor], support: torch.Tensor, max_order: int):
        coefficients = self.coefficients(max_order, basis[0].dtype)
        value = torch.einsum("l,lnd->nd", coefficients[: max_order + 1], torch.stack(basis[: max_order + 1]))
        residual = self.eta * _run_bottleneck(self.transform, value)
        output = self.norm(value + residual)
        output = output * support.to(output.dtype).unsqueeze(-1)
        return output, value, coefficients, residual


class GraphAwareRouter(nn.Module):
    def __init__(self, hidden_dim: int, embed_dim: int, hidden_router_dim: int, dropout: float):
        super().__init__()
        self.modality_embedding = nn.Embedding(2, embed_dim)
        self.relation_embedding = nn.Embedding(2, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2 * embed_dim + hidden_dim, hidden_router_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_router_dim, 4),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.012)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-0.009, -0.003, 0.004, 0.008]))

    def forward(
        self,
        current: torch.Tensor,
        first_neighbor: torch.Tensor,
        cross_context: torch.Tensor,
        modality_id: int,
        relation_id: int,
    ) -> torch.Tensor:
        n = current.size(0)
        mod = self.modality_embedding.weight[modality_id].expand(n, -1)
        rel = self.relation_embedding.weight[relation_id].expand(n, -1)
        value = torch.cat(
            (current, first_neighbor, first_neighbor - current, mod, rel, cross_context), dim=-1
        )
        return self.net(value)


class JointViewContribution(nn.Module):
    """Shared Physical/Semantic contribution scales with a chunked wide input."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.input = nn.Linear(5 * hidden_dim, bottleneck_dim)
        self.output = nn.Linear(bottleneck_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        h: torch.Tensor,
        d_physical: torch.Tensor,
        d_semantic: torch.Tensor,
    ) -> torch.Tensor:
        """Return [s_P, s_S] without allocating a [N, 5H] concatenation."""
        parts = (h, d_physical, d_semantic, (d_physical - d_semantic).abs(), d_physical * d_semantic)
        pre_activation = None
        offset = 0
        for part in parts:
            width = part.size(-1)
            contribution = F.linear(part, self.input.weight[:, offset:offset + width], bias=None)
            pre_activation = contribution if pre_activation is None else pre_activation + contribution
            offset += width
        if offset != self.input.in_features:
            raise ValueError(f"joint-view input has width {offset}; expected {self.input.in_features}")
        if self.input.bias is not None:
            pre_activation = pre_activation + self.input.bias
        strength_logits = self.output(F.gelu(pre_activation))
        return 2.0 * torch.sigmoid(strength_logits)


class Model(nn.Module):
    """M_repaired backbone with one explicitly selected V1.3 mechanism."""

    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    supports_link_prediction = False
    supports_unimodal_aux_loss = True
    record_run_telemetry = True
    record_epoch_history = True
    collect_v7_diagnostics = False

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        mc = cfg.model
        self.condition = str(mc.get("condition", "M_repaired"))
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if self.condition != "M_repaired":
            raise ValueError("MR-CGE V1.3 freezes M_repaired as its only backbone condition")
        self.v13_variant = str(mc.get("v13_variant", "base"))
        if self.v13_variant not in ("base", "joint_view", "sender_private"):
            raise ValueError("v13_variant must be base, joint_view, or sender_private")
        self.sender_private_ratio = float(mc.get("sender_private_ratio", 0.0))
        self.variant = self.v13_variant
        if not 0.0 <= self.sender_private_ratio <= 1.0:
            raise ValueError("sender_private_ratio must lie in [0,1]")
        if self.v13_variant != "sender_private" and self.sender_private_ratio != 0.0:
            raise ValueError("sender_private_ratio is active only for the sender_private variant")
        if self.v13_variant == "sender_private" and self.sender_private_ratio not in (0.0, 0.3):
            raise ValueError("formal sender_private training fixes sender_private_ratio=0.3; 0 is compatibility-only")
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if min(self.text_dim, self.visual_dim) <= 0 or self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError("mrcge_v11 requires a concatenated Text/Visual feature matrix")
        self.num_nodes = int(data_info.get("num_nodes", 0))
        self.num_classes = int(data_info["num_classes"])
        self.hidden_dim = int(mc.get("hidden_dim", 256))
        self.num_experts = int(mc.get("num_experts", 4))
        self.top_k = int(mc.get("top_k", 2))
        self.dropout_p = float(mc.get("dropout", 0.2))
        self.eps = float(mc.get("eps", 1.0e-8))
        self.physical_chunk_size = int(mc.get("physical_chunk_size", 32768))
        self.semantic_chunk_size = int(mc.get("semantic_chunk_size", 32768))
        self.cross_edge_chunk_size = int(mc.get("cross_edge_chunk_size", 8192))
        self.attention_dim = int(mc.get("attention_dim", 32))
        self.late_cross_dropout = float(mc.get("late_cross_dropout", 0.15))
        self.same_node_enabled = bool(mc.get("same_node_enabled", True))
        self.eta = float(mc.get("expert_eta", 0.1))
        self.balance_weight = float(mc.get("balance_weight", 0.01))
        self.physical_hops = int(mc.get("physical_hops", 3))
        self.semantic_hops = int(mc.get("semantic_hops", 1))
        if (self.physical_hops, self.semantic_hops) != (3, 1):
            raise ValueError("MR-CGE V1.2 freezes physical_hops=3 and semantic_hops=1")
        if (self.hidden_dim, self.num_experts, self.top_k) != (256, 4, 2):
            raise ValueError("MR-CGE V1.2 freezes hidden_dim=256, four experts and Top-2")

        # Construct all common modules in the same order for S/F/M. Their initial
        # tensors can be checked by exact state_dict hashes under a fixed seed.
        self.projectors = nn.ModuleDict({
            "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
            "visual": _IntrinsicProjector(self.visual_dim, self.hidden_dim, self.dropout_p),
        })
        self.cross_projections = nn.ModuleDict({
            "visual_to_text": nn.Sequential(
                nn.Linear(self.hidden_dim, 64, bias=False), nn.GELU(), nn.Linear(64, self.hidden_dim, bias=False)
            ),
            "text_to_visual": nn.Sequential(
                nn.Linear(self.hidden_dim, 64, bias=False), nn.GELU(), nn.Linear(64, self.hidden_dim, bias=False)
            ),
        })
        self.query_projections = nn.ModuleDict({m: nn.Linear(self.hidden_dim, self.attention_dim, bias=False) for m in _MODALITIES})
        self.key_projections = nn.ModuleDict({m: nn.Linear(self.hidden_dim, self.attention_dim, bias=False) for m in _MODALITIES})
        self.source_selector = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + 3, 128), nn.GELU(), nn.Linear(128, 3)
        )
        nn.init.zeros_(self.source_selector[-1].weight)
        with torch.no_grad():
            self.source_selector[-1].bias.copy_(torch.log(torch.tensor([0.45, 0.35, 0.20])))
        self.channel_gate = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, int(mc.get("channel_gate_hidden", 128))),
            nn.GELU(), nn.Linear(int(mc.get("channel_gate_hidden", 128)), self.hidden_dim)
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.constant_(self.channel_gate[-1].bias, math.log(0.73 / 0.27))
        self.cross_lambda_raw = nn.Parameter(torch.zeros(2))
        self.tau_raw = nn.Parameter(torch.full((2,), self._logit((0.25 - 0.1) / 0.7)))
        self.rho_raw = nn.Parameter(torch.full((2,), self._logit(float(mc.get("rho_init", 0.25)))))
        self.gamma_raw = nn.Parameter(torch.full((2,), self._logit(float(mc.get("gamma_init", 0.08)))))
        self.beta_raw = nn.Parameter(torch.full((2,), self._logit(float(mc.get("beta_init", 0.25)))))
        self.unimodal_heads = nn.ModuleDict({m: nn.Linear(self.hidden_dim, self.num_classes) for m in _MODALITIES})
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(float(mc.get("fusion_dropout", 0.2)))
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # Exactly one filter implementation is installed for the selected arm.
        # Restore the RNG state after condition-specific initialization so the
        # external NC classifier starts from the same tensors in S/F/M.
        with torch.random.fork_rng(devices=[]):
            if self.condition.startswith("M_"):
                self.filter_bank = SharedExpertBank(self.hidden_dim, int(mc.get("expert_bottleneck", 64)), self.dropout_p, self.eta)
                if self.condition == "M_anchor":
                    with torch.no_grad():
                        self.filter_bank.theta[0].copy_(torch.tensor([0.40, 0.30, 0.20, 0.10]))
                self.router = GraphAwareRouter(
                    self.hidden_dim, int(mc.get("router_embed_dim", 8)),
                    int(mc.get("router_hidden_dim", 256)), self.dropout_p
                )
            else:
                self.single_filter = SingleFilter(
                    self.hidden_dim, int(mc.get("expert_bottleneck", 64)), self.dropout_p,
                    self.eta, learnable=self.condition == "F_sinit"
                )

        # Keep V1.2 shared modules and the external NC classifier initialization
        # unchanged: the new network's random first layer is isolated from the
        # caller's RNG stream, and its zero output layer starts at unit scales.
        if self.v13_variant == "joint_view":
            with torch.random.fork_rng(devices=[]):
                self.joint_view = JointViewContribution(self.hidden_dim, 32)

        self.register_buffer("semantic_source_flags_v7b", torch.empty((0, 2), dtype=torch.uint8), persistent=False)
        self.register_buffer("semantic_text_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("semantic_text_weight", torch.empty((0,), dtype=torch.float32), persistent=False)
        self.register_buffer("semantic_visual_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("semantic_visual_weight", torch.empty((0,), dtype=torch.float32), persistent=False)
        self.register_buffer("semantic_candidate_fingerprint_state", torch.zeros((32,), dtype=torch.uint8))
        self.register_buffer("semantic_feature_fingerprint_state", torch.zeros((32,), dtype=torch.uint8))
        self.semantic_candidate_fingerprint: str | None = None
        self.semantic_feature_fingerprint: str | None = None
        self.semantic_v7b_fingerprint: str | None = None
        self.semantic_cache_metadata: dict[str, Any] | None = None
        self._last_diagnostics: dict[str, Any] = {}
        self._last_gradient_diagnostics: dict[str, Any] = {}
        self._last_route_snapshot: dict[str, Any] = {}
        self._last_train_v13_diagnostics: dict[str, float | int] = {}
        self._last_val_v13_diagnostics: dict[str, float | int] = {}
        self._capture_state_audit = False
        self._last_state_audit: dict[str, Any] = {}
        self._current_epoch = 0
        self._last_balance_raw_tensor: torch.Tensor | None = None
        self._last_train_balance_raw_tensor: torch.Tensor | None = None
        self.out_dim = self.hidden_dim

    @staticmethod
    def _strength_stats(values: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        # Unsupported nodes carry no graph message; report their scale neutrally.
        neutral = torch.ones_like(values)
        effective = torch.where(support, values, neutral).detach().float()
        if effective.numel() == 0:
            one = values.new_ones((), dtype=torch.float32)
            zero = values.new_zeros((), dtype=torch.float32)
            return {"mean": one, "std": zero, "p10": one, "p50": one, "p90": one,
                    "deviation_fraction": zero}
        quantiles = torch.quantile(effective, effective.new_tensor([0.1, 0.5, 0.9]))
        return {
            "mean": effective.mean(),
            "std": effective.std(unbiased=False),
            "p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2],
            "deviation_fraction": ((effective - 1.0).abs() > 1.0e-6).float().mean(),
        }

    @staticmethod
    def _prefix_diagnostics(prefix: str, values: dict[str, Any]) -> dict[str, float | int]:
        return {
            f"{prefix}_{key}": float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }

    @staticmethod
    def _diagnostic_scalar(value: torch.Tensor, collect: bool) -> float | torch.Tensor:
        value = value.detach()
        return float(value.cpu()) if collect else value

    @staticmethod
    def _logit(value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("initial bounded scale must be between zero and one")
        return math.log(value / (1.0 - value))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.rho_raw)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_raw)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)

    @property
    def cross_lambda(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.cross_lambda_raw)

    @property
    def tau(self) -> torch.Tensor:
        return 0.1 + 0.7 * torch.sigmoid(self.tau_raw)

    def load_semantic_cache(
        self,
        path: str,
        *,
        expected_dataset: str | None = None,
        expected_feature_fingerprint: str | None = None,
        expected_candidate_fingerprint: str | None = None,
        expected_v7b_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        cache = load_semantic_candidate_cache_v7b(
            path,
            expected_dataset=expected_dataset,
            expected_feature_fingerprint=expected_feature_fingerprint,
            expected_candidate_fingerprint=expected_candidate_fingerprint,
            expected_v7b_fingerprint=expected_v7b_fingerprint,
        )
        metadata = cache["metadata"]
        if self.num_nodes and self.num_nodes != int(metadata["num_nodes"]):
            raise ValueError("semantic cache node count differs from the model graph")
        if int(metadata["text_dim"]) != self.text_dim or int(metadata["visual_dim"]) != self.visual_dim:
            raise ValueError("semantic cache feature dimensions differ from model input")
        flags = cache["source_flags"].to(dtype=torch.uint8, device="cpu")
        unique = cache["unique_edge_index"].to(dtype=torch.long, device="cpu")
        if flags.shape != (unique.size(1), 2):
            raise ValueError("V7B source_flags do not align with unique edge IDs")
        operators = build_modality_knn_operators(unique, flags, self.num_nodes)
        self.semantic_source_flags_v7b = flags
        for modality in _MODALITIES:
            setattr(self, f"semantic_{modality}_edge_index", operators[modality]["edge_index"])
            setattr(self, f"semantic_{modality}_weight", operators[modality]["weight"])
        self.semantic_candidate_fingerprint = str(metadata["candidate_fingerprint"])
        self.semantic_feature_fingerprint = str(metadata["feature_fingerprint"])
        self.semantic_v7b_fingerprint = str(metadata["v7b_fingerprint"])
        self.semantic_cache_metadata = metadata
        self.semantic_candidate_fingerprint_state.copy_(torch.tensor(list(bytes.fromhex(self.semantic_candidate_fingerprint)), dtype=torch.uint8))
        self.semantic_feature_fingerprint_state.copy_(torch.tensor(list(bytes.fromhex(self.semantic_feature_fingerprint)), dtype=torch.uint8))
        if self.num_nodes == 0:
            self.num_nodes = int(metadata["num_nodes"])
        return metadata

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        for name, expected_buffer, description in (
            ("semantic_candidate_fingerprint_state", self.semantic_candidate_fingerprint_state, "candidate graph"),
            ("semantic_feature_fingerprint_state", self.semantic_feature_fingerprint_state, "input features"),
        ):
            saved = state_dict.get(name)
            if saved is not None:
                if not torch.equal(saved.detach().cpu(), expected_buffer.detach().cpu()):
                    raise RuntimeError(f"checkpoint semantic {description} fingerprint differs from the attached V7B.1 cache")
            elif self.semantic_candidate_fingerprint is not None and strict:
                raise RuntimeError(f"checkpoint is missing its semantic {description} fingerprint")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _physical_operator(edge_index: torch.Tensor, n: int, dtype: torch.dtype):
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("physical edge_index must have shape [2,E]")
        loops = int((edge_index[0] == edge_index[1]).sum().item())
        edge_index, _ = remove_self_loops(edge_index.long())
        src, dst = edge_index
        degree = torch.zeros(n, dtype=dtype, device=edge_index.device)
        if dst.numel():
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=dtype))
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        weight = inv_sqrt[src] * inv_sqrt[dst]
        return edge_index, weight, degree > 0, loops

    @staticmethod
    def _support(edge_index: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        if edge_index.numel():
            mask[edge_index[1]] = True
        return mask

    def _basis(
        self,
        initial: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        support: torch.Tensor,
        max_order: int,
        chunk_size: int,
    ) -> list[torch.Tensor]:
        basis = [F.layer_norm(initial, (self.hidden_dim,))]
        for _ in range(max_order):
            propagated = chunked_weighted_spmm(basis[-1], edge_index, edge_weight, chunk_size)
            basis.append(propagated * support.to(propagated.dtype).unsqueeze(-1))
        return basis

    @staticmethod
    def _rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            value = value[mask]
        if value.numel() == 0:
            return value.new_zeros((), dtype=torch.float32)
        return value.detach().float().square().mean().sqrt()

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
            output, filtered, coefficients, residual = self.single_filter(basis, support, max_order)
            detail = self._filter_summary({
                "G": filtered.unsqueeze(0), "E": output.unsqueeze(0),
                "residual": residual.unsqueeze(0),
                "coefficients": coefficients.detach().cpu().tolist(),
            }, support) if collect else {}
            return output, None, torch.zeros((), device=output.device), {
                **detail,
            }

        # Preserve V1.1's exact eval reduction path for historical checkpoint
        # regression. The low-memory arithmetic below is training-only (or a
        # normal training forward without diagnostics).
        if collect or not (self.training and torch.is_grad_enabled()):
            outputs, filtered, coefficients, residual = self.filter_bank(basis, support, max_order)
            first = basis[1] if len(basis) > 1 else basis[0]
            logits = self.router(current, first, cross_context, modality_id, relation_id)
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
        """Compute one shared expert/router call without retaining a four-expert stack."""
        max_order = len(basis) - 1
        coefficients = self.filter_bank.coefficients(max_order, basis[0].dtype)
        first = basis[1] if len(basis) > 1 else basis[0]
        # Prepare the deterministic router input projection before expert
        # Dropout. The router Dropout is applied after all four experts, which
        # preserves V1.1's train-time RNG consumption order.
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
            # The enclosing non-reentrant checkpoint recomputes this complete
            # expert/router boundary, so avoid nesting one checkpoint per MLP.
            residual = self.eta * expert(filtered)
            value = self.filter_bank.norms[expert_id](filtered + residual) * mask
            outputs.append(value)
            if collect:
                filtered_values.append(filtered)
                residuals.append(residual * mask)

        logits = self.router.net[3](self.router.net[2](router_hidden))
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

    @staticmethod
    def _route_summary(route: dict[str, torch.Tensor] | None, support: torch.Tensor) -> dict[str, Any] | None:
        if route is None:
            return None
        if "dense" not in route:
            return {
                "dense_importance": route["importance"].detach().float().cpu().tolist(),
                "top2_selection_share": route["share"].detach().float().cpu().tolist(),
                "dense_entropy": float(route["dense_entropy"].detach().cpu()),
                "selected_weight_entropy": float(route["selected_entropy"].detach().cpu()),
                "active_nodes": int(route["active_nodes"].detach().cpu()),
            }
        mask = support
        dense = route["dense"].detach().float()
        selected = route["selected"].detach()
        if bool(mask.any()):
            importance = dense[mask].mean(dim=0)
            selection_share = selected[mask].float().mean(dim=0) / 2.0
            entropy = -(dense[mask].clamp_min(1.0e-12) * dense[mask].clamp_min(1.0e-12).log()).sum(dim=-1).mean()
            selected_entropy = -(route["weights"].detach().float()[mask].clamp_min(1.0e-12)
                                 * route["weights"].detach().float()[mask].clamp_min(1.0e-12).log()).sum(dim=-1).mean()
        else:
            importance = dense.new_zeros((4,))
            selection_share = dense.new_zeros((4,))
            entropy = dense.new_zeros(())
            selected_entropy = dense.new_zeros(())
        return {
            "dense_importance": importance.cpu().tolist(),
            "top2_selection_share": selection_share.cpu().tolist(),
            "dense_entropy": float(entropy.cpu()),
            "selected_weight_entropy": float(selected_entropy.cpu()),
            "active_nodes": int(mask.sum().detach().cpu()),
        }

    @classmethod
    def _compact_route_state(cls, route: dict[str, torch.Tensor], support: torch.Tensor) -> dict[str, torch.Tensor]:
        if "dense" not in route:
            return {key: route[key].detach() for key in
                    ("importance", "share", "dense_entropy", "selected_entropy", "active_nodes")}
        summary = cls._route_summary(route, support)
        device = route["dense"].device
        return {
            "importance": torch.tensor(summary["dense_importance"], dtype=torch.float32, device=device),
            "share": torch.tensor(summary["top2_selection_share"], dtype=torch.float32, device=device),
            "dense_entropy": torch.tensor(summary["dense_entropy"], dtype=torch.float32, device=device),
            "selected_entropy": torch.tensor(summary["selected_weight_entropy"], dtype=torch.float32, device=device),
            "active_nodes": torch.tensor(summary["active_nodes"], dtype=torch.long, device=device),
        }

    @classmethod
    def _filter_summary(cls, detail: dict[str, Any], support: torch.Tensor) -> dict[str, Any]:
        g, e, residual = detail["G"], detail["E"], detail["residual"]
        mask = support
        chunk_size = 8192

        def per_expert_rms(values: torch.Tensor) -> list[float]:
            values = values.detach()
            result = []
            for value in values:
                total = torch.zeros((), dtype=torch.float32, device=value.device)
                count = 0
                for start in range(0, value.size(0), chunk_size):
                    stop = min(start + chunk_size, value.size(0))
                    part = value[start:stop]
                    if mask is not None:
                        part = part[mask[start:stop]]
                    if part.numel():
                        total = total + part.float().square().sum()
                        count += part.numel()
                result.append(float((total / max(count, 1)).sqrt().cpu()))
            return result

        def pairwise(values: torch.Tensor) -> dict[str, float]:
            if values.size(0) < 2 or not bool(mask.any()):
                return {}
            values = values.detach()
            result = {}
            for left in range(values.size(0)):
                for right in range(left + 1, values.size(0)):
                    dot = torch.zeros((), dtype=torch.float32, device=values.device)
                    left_sq = torch.zeros_like(dot)
                    right_sq = torch.zeros_like(dot)
                    for start in range(0, values.size(1), chunk_size):
                        stop = min(start + chunk_size, values.size(1))
                        valid = mask[start:stop]
                        if not bool(valid.any()):
                            continue
                        a = values[left, start:stop][valid].float()
                        b = values[right, start:stop][valid].float()
                        dot = dot + (a * b).sum()
                        left_sq = left_sq + a.square().sum()
                        right_sq = right_sq + b.square().sum()
                    denom = left_sq.sqrt().clamp_min(1.0e-12) * right_sq.sqrt().clamp_min(1.0e-12)
                    result[f"{left}_{right}"] = float((dot / denom).cpu())
            return result

        return {
            "coefficients": detail["coefficients"],
            "G_expert_rms": per_expert_rms(g),
            "G_pairwise_cosine": pairwise(g),
            "E_expert_rms": per_expert_rms(e),
            "E_pairwise_cosine": pairwise(e),
            "expert_residual_rms": per_expert_rms(residual),
        }

    @staticmethod
    def _zero_context_fallback(private_response: torch.Tensor, collaborative_response: torch.Tensor,
                               context: torch.Tensor) -> torch.Tensor:
        nonzero = context.abs().sum(dim=-1) > 0
        return torch.where(nonzero[:, None], collaborative_response, private_response)

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        self._last_train_v13_diagnostics = {}
        self._last_val_v13_diagnostics = {}

    def collect_epoch_diagnostics(self) -> dict[str, float | int]:
        """Return scalar-only filter and route traces for the current epoch."""
        result: dict[str, float | int] = {"diagnostic_epoch": self._current_epoch}
        result["v13_variant_joint_view"] = int(self.v13_variant == "joint_view")
        result["v13_variant_sender_private"] = int(self.v13_variant == "sender_private")
        result["sender_private_ratio"] = self.sender_private_ratio
        for scope, values in (("train", self._last_train_v13_diagnostics),
                              ("val", self._last_val_v13_diagnostics)):
            result.update(self._prefix_diagnostics(f"{scope}_v13", values))
        if self.condition.startswith("M_"):
            theta = self.filter_bank.theta.detach().cpu()
            alpha_p = self.filter_bank.coefficients(3, torch.float32).detach().cpu()
            alpha_s = self.filter_bank.coefficients(1, torch.float32).detach().cpu()
        else:
            theta = self.single_filter.theta.detach().cpu().reshape(1, 4)
            alpha_p = self.single_filter.coefficients(3, torch.float32).detach().cpu().reshape(1, 4)
            alpha_s = self.single_filter.coefficients(1, torch.float32).detach().cpu().reshape(1, 4)
        for expert in range(theta.size(0)):
            for order in range(4):
                result[f"theta_e{expert}_l{order}"] = float(theta[expert, order])
                result[f"alpha_physical_e{expert}_l{order}"] = float(alpha_p[expert, order])
                result[f"alpha_semantic_e{expert}_l{order}"] = float(alpha_s[expert, order])
        for view, snapshot in self._last_route_snapshot.items():
            if snapshot is None:
                continue
            route = self._route_summary(snapshot["route"], snapshot["support"])
            for expert, value in enumerate(route["dense_importance"]):
                result[f"route_{view}_importance_e{expert}"] = float(value)
            for expert, value in enumerate(route["top2_selection_share"]):
                result[f"route_{view}_selection_e{expert}"] = float(value)
            result[f"route_{view}_entropy"] = float(route["dense_entropy"])
        latest_balance = float(self._last_balance_raw_tensor.cpu()) if self._last_balance_raw_tensor is not None else 0.0
        result["balance_loss_raw"] = latest_balance
        result["aux_loss_scaled"] = float(self.balance_weight * latest_balance)
        train_balance = float(self._last_train_balance_raw_tensor.cpu()) if self._last_train_balance_raw_tensor is not None else 0.0
        result["train_balance_loss_raw"] = train_balance
        result["train_aux_loss_scaled"] = float(self.balance_weight * train_balance)
        return result

    def _early_cross(
        self,
        receiver: str,
        receiver_id: int,
        source_state: torch.Tensor,
        projection: nn.Module,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        support: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = projection(source_state)
        raw = chunked_weighted_spmm(projected, edge_index, edge_weight, self.physical_chunk_size)
        raw = raw * support.to(raw.dtype).unsqueeze(-1)
        raw_rms = self._rms(raw, support)
        target_rms = self._rms(reference, support)
        if float(raw_rms.cpu()) <= self.eps:
            scale = raw_rms.new_zeros(())
            calibrated = torch.zeros_like(raw)
        else:
            scale = (target_rms / (raw_rms + self.eps)).detach()
            calibrated = raw * scale.to(raw.dtype)
        return raw, calibrated, scale

    @staticmethod
    def _sequential_from_parts(module: nn.Sequential, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Evaluate a Sequential MLP without materializing its wide concatenated input."""
        first = module[0]
        if not isinstance(first, nn.Linear):
            return module(torch.cat(parts, dim=-1))
        offset = 0
        pre_activation = None
        for part in parts:
            width = part.size(-1)
            contribution = F.linear(part, first.weight[:, offset:offset + width], bias=None)
            pre_activation = contribution if pre_activation is None else pre_activation + contribution
            offset += width
        if offset != first.in_features:
            raise ValueError(f"MLP input parts have width {offset}; expected {first.in_features}")
        if first.bias is not None:
            pre_activation = pre_activation + first.bias
        value = pre_activation
        for layer in list(module.children())[1:]:
            value = layer(value)
        return value

    def _late(
        self,
        target: str,
        target_id: int,
        source: str,
        h_source: torch.Tensor,
        u_private_source: torch.Tensor,
        u_collaborative_target: torch.Tensor,
        *,
        collect: bool,
    ):
        projection_name = "visual_to_text" if source == "visual" else "text_to_visual"
        phi = self.cross_projections[projection_name]
        edge_index = getattr(self, f"semantic_{source}_edge_index").to(u_collaborative_target.device)
        support_sem = self._support(edge_index, u_collaborative_target.size(0), u_collaborative_target.device)
        query = self.query_projections[target](F.layer_norm(u_collaborative_target, (self.hidden_dim,)))
        key = self.key_projections[source](F.layer_norm(u_private_source, (self.hidden_dim,)))
        value = phi(0.7 * h_source + 0.3 * u_private_source)
        c_sem, attn = receiver_sparse_attention(query, key, value, edge_index, self.cross_edge_chunk_size)
        c_sem = c_sem * support_sem.to(c_sem.dtype).unsqueeze(-1)
        if self.same_node_enabled:
            c_self = phi(u_private_source)
            support_self = torch.ones_like(support_sem)
        else:
            c_self = torch.zeros_like(c_sem)
            support_self = torch.zeros_like(support_sem)
        cosine = F.cosine_similarity(c_sem.detach(), c_self.detach(), dim=-1, eps=1.0e-8)
        selector_parts = (
            u_collaborative_target, c_sem, c_self, cosine[:, None],
            support_sem.to(c_sem.dtype)[:, None], support_self.to(c_sem.dtype)[:, None],
        )
        logits = self._sequential_from_parts(self.source_selector, selector_parts).clone()
        logits[~support_sem, 0] = -torch.inf
        logits[~support_self, 1] = -torch.inf
        source_weights = torch.softmax(logits, dim=-1)
        any_support = support_sem | support_self
        none_only = torch.zeros_like(source_weights)
        none_only[:, 2] = 1.0
        source_weights = torch.where(any_support[:, None], source_weights, none_only)
        p_accept = source_weights[:, 0] + source_weights[:, 1]
        candidate_cond = (
            source_weights[:, 0:1] * c_sem + source_weights[:, 1:2] * c_self
        ) / p_accept.clamp_min(1.0e-8)[:, None]
        candidate_cond = candidate_cond * any_support.to(candidate_cond.dtype).unsqueeze(-1)
        interaction_parts = (
            u_collaborative_target, candidate_cond,
            u_collaborative_target * candidate_cond,
            (u_collaborative_target - candidate_cond).abs(),
        )
        gate = torch.sigmoid(self._sequential_from_parts(self.channel_gate, interaction_parts))
        unbounded = self.cross_lambda[target_id] * p_accept[:, None] * gate * candidate_cond
        unbounded = unbounded * any_support.to(unbounded.dtype).unsqueeze(-1)
        ref_norm = u_collaborative_target.float().norm(dim=-1, keepdim=True)
        delta_norm = unbounded.float().norm(dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(delta_norm), self.tau[target_id] * ref_norm / (delta_norm + self.eps))
        controlled = unbounded * scale.to(unbounded.dtype)
        if self.training and self.late_cross_dropout > 0:
            keep = torch.rand((controlled.size(0), 1), device=controlled.device) >= self.late_cross_dropout
            controlled = controlled * keep.to(controlled.dtype)
        controlled = controlled * any_support.to(controlled.dtype).unsqueeze(-1)
        completed = u_collaborative_target + controlled
        detail = {
            "late_delta_rms": self._diagnostic_scalar(self._rms(controlled), collect),
            "late_delta_ratio": self._diagnostic_scalar(
                self._rms(controlled) / self._rms(u_collaborative_target).clamp_min(self.eps), collect
            ),
        }
        if collect:
            detail.update({
                "semantic_support_nodes": int(support_sem.sum().detach().cpu()),
                "same_node_support_nodes": int(support_self.sum().detach().cpu()),
                "p_accept_mean": float(p_accept.detach().mean().cpu()),
                "selector_share": source_weights.detach().mean(dim=0).cpu().tolist(),
                "candidate_rms": float(self._rms(candidate_cond, any_support).cpu()),
                "gate_mean": float(gate.detach().mean().cpu()),
                "unbounded_rms": float(self._rms(unbounded).cpu()),
                "relative_cap_fraction": float((scale.detach() < 1.0).float().mean().cpu()),
                "attention_entropy": float((-(attn.detach().clamp_min(1e-30) * attn.detach().clamp_min(1e-30).log()).mean()).cpu()) if attn.numel() else 0.0,
            })
        return completed, detail

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, return_details: bool = False):
        if edge_index is None:
            raise ValueError("MR-CGE V1.2 requires a global physical edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [nodes,{self.input_dim}]")
        if self.num_nodes and x.size(0) != self.num_nodes:
            raise RuntimeError("MR-CGE V1.2 requires the cached global node order")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("MR-CGE V1.2 requires a validated V7B.1 semantic cache")
        raw = {"text": x[:, :self.text_dim], "visual": x[:, self.text_dim:self.input_dim]}
        h = {m: self.projectors[m](raw[m]) for m in _MODALITIES}
        physical_edges, physical_weight, physical_support, loop_count = self._physical_operator(
            edge_index, x.size(0), h["text"].dtype
        )
        bases: dict[str, dict[str, list[torch.Tensor]]] = {m: {} for m in _MODALITIES}
        supports: dict[str, dict[str, torch.Tensor]] = {m: {} for m in _MODALITIES}
        for m in _MODALITIES:
            bases[m]["physical"] = self._basis(
                h[m], physical_edges, physical_weight, physical_support,
                self.physical_hops, self.physical_chunk_size,
            )
            sem_edges = getattr(self, f"semantic_{m}_edge_index").to(x.device)
            sem_weight = getattr(self, f"semantic_{m}_weight").to(device=x.device, dtype=h[m].dtype)
            sem_support = self._support(sem_edges, x.size(0), x.device)
            bases[m]["semantic"] = self._basis(
                h[m], sem_edges, sem_weight, sem_support,
                self.semantic_hops, self.semantic_chunk_size,
            )
            supports[m]["physical"] = physical_support
            supports[m]["semantic"] = sem_support

        private: dict[str, torch.Tensor] = {}
        private_strengths: dict[str, torch.Tensor] = {}
        responses: dict[str, dict[str, torch.Tensor]] = {m: {} for m in _MODALITIES}
        route_details: dict[str, Any] = {}
        filter_details: dict[str, Any] = {}
        balances: list[torch.Tensor] = []
        paired_rng_states: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        zeros = {m: torch.zeros_like(h[m]) for m in _MODALITIES}
        for modality_id, m in enumerate(_MODALITIES):
            for relation_id, relation in enumerate(("physical", "semantic")):
                if relation == "physical":
                    paired_rng_states[m] = capture_rng_state(x.device)
                response, route, balance, detail = self._filter(
                    bases[m][relation], supports[m][relation], h[m], relation_id, modality_id,
                    zeros[m], collect=return_details
                )
                responses[m][relation] = response
                balances.append(balance)
                view_name = f"private_{m}_{relation}"
                route_details[view_name] = route
                filter_details[view_name] = detail
            d_physical = self.rho[modality_id] * responses[m]["physical"]
            d_semantic = self.gamma[modality_id] * responses[m]["semantic"]
            if self.v13_variant == "joint_view":
                strengths = self.joint_view(h[m], d_physical, d_semantic)
                private_strengths[m] = strengths
                private[m] = h[m] + strengths[:, 0:1] * d_physical + strengths[:, 1:2] * d_semantic
            else:
                private[m] = h[m] + d_physical + d_semantic

        collaborative: dict[str, torch.Tensor] = {}
        collaborative_strengths: dict[str, torch.Tensor] = {}
        early_details: dict[str, Any] = {}
        cross_contexts: dict[str, torch.Tensor] = {}
        mechanism_forward: dict[str, float | int | torch.Tensor] = {
            "rho_text": self._diagnostic_scalar(self.rho[0], return_details),
            "rho_visual": self._diagnostic_scalar(self.rho[1], return_details),
            "gamma_text": self._diagnostic_scalar(self.gamma[0], return_details),
            "gamma_visual": self._diagnostic_scalar(self.gamma[1], return_details),
            "sender_private_ratio": self.sender_private_ratio,
        }
        collab_routes: dict[str, Any] = {}
        for modality_id, m in enumerate(_MODALITIES):
            other = "visual" if m == "text" else "text"
            projection_name = "visual_to_text" if m == "text" else "text_to_visual"
            sender_state = h[other]
            if self.v13_variant == "sender_private" and self.sender_private_ratio > 0.0:
                sender_state = ((1.0 - self.sender_private_ratio) * h[other]
                                + self.sender_private_ratio * private[other])
            raw_cross, cross_hat, calibration_scale = self._early_cross(
                m, modality_id, sender_state, self.cross_projections[projection_name],
                physical_edges, physical_weight, physical_support, bases[m]["physical"][1]
            )
            cross_contexts[m] = cross_hat
            collab_basis = list(bases[m]["physical"])
            collab_basis[1] = (collab_basis[1] + self.beta[modality_id] * cross_hat) * physical_support.to(h[m].dtype).unsqueeze(-1)
            with replay_rng_state(paired_rng_states[m], x.device):
                r_collab, route, balance, detail = self._filter(
                    collab_basis, physical_support, h[m], 0, modality_id, cross_hat, collect=return_details
                )
            nonzero_context = cross_hat.abs().sum(dim=-1) > 0
            r_collab = self._zero_context_fallback(responses[m]["physical"], r_collab, cross_hat)
            if self.v13_variant == "joint_view":
                d_physical_collab = self.rho[modality_id] * r_collab
                d_semantic_private = self.gamma[modality_id] * responses[m]["semantic"]
                strengths_collab = self.joint_view(h[m], d_physical_collab, d_semantic_private)
                collaborative_strengths[m] = strengths_collab
                collaborative[m] = private[m] + self.rho[modality_id] * (
                    strengths_collab[:, 0:1] * r_collab
                    - private_strengths[m][:, 0:1] * responses[m]["physical"]
                )
            else:
                collaborative[m] = private[m] + self.rho[modality_id] * (r_collab - responses[m]["physical"])
            balances.append(balance)
            collab_routes[m] = route
            filter_details[f"collaborative_{m}_physical"] = detail
            neighbor_rms = self._rms(bases[m]["physical"][1], physical_support).clamp_min(self.eps)
            mechanism_forward[f"early_cross_raw_rms_{m}"] = self._diagnostic_scalar(
                self._rms(raw_cross, physical_support), return_details)
            mechanism_forward[f"early_cross_calibrated_rms_{m}"] = self._diagnostic_scalar(
                self._rms(cross_hat, physical_support), return_details)
            mechanism_forward[f"early_cross_to_neighbor_rms_{m}"] = self._diagnostic_scalar(
                self._rms(cross_hat, physical_support) / neighbor_rms, return_details)
            mechanism_forward[f"physical_private_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(self.rho[modality_id] * responses[m]["physical"], physical_support), return_details)
            mechanism_forward[f"semantic_private_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(self.gamma[modality_id] * responses[m]["semantic"], supports[m]["semantic"]), return_details)
            mechanism_forward[f"physical_collaborative_contribution_rms_{m}"] = self._diagnostic_scalar(
                self._rms(self.rho[modality_id] * r_collab, physical_support), return_details)
            if self.v13_variant == "joint_view":
                support_p = physical_support
                support_s = supports[m]["semantic"]
                for scope, strength_values in (("private", private_strengths[m]),
                                               ("collaborative", collaborative_strengths[m])):
                    for index, relation, support in ((0, "sP", support_p), (1, "sS", support_s)):
                        stats = self._strength_stats(strength_values[:, index], support)
                        for field, value in stats.items():
                            mechanism_forward[f"{scope}_{relation}_{field}_{m}"] = self._diagnostic_scalar(
                                value, return_details)
                if bool(physical_support.any()):
                    change = (collaborative_strengths[m][:, 0] - private_strengths[m][:, 0])[physical_support]
                    mechanism_forward[f"collaborative_vs_private_sP_abs_mean_{m}"] = self._diagnostic_scalar(
                        change.abs().mean(), return_details)
                    mechanism_forward[f"collaborative_vs_private_sP_signed_mean_{m}"] = self._diagnostic_scalar(
                        change.mean(), return_details)
                else:
                    zero = h[m].new_zeros(())
                    mechanism_forward[f"collaborative_vs_private_sP_abs_mean_{m}"] = self._diagnostic_scalar(
                        zero, return_details)
                    mechanism_forward[f"collaborative_vs_private_sP_signed_mean_{m}"] = self._diagnostic_scalar(
                        zero, return_details)
            if return_details:
                early_details[m] = {
                    "raw_rms": float(self._rms(raw_cross, physical_support).cpu()),
                    "calibrated_rms": float(self._rms(cross_hat, physical_support).cpu()),
                    "scale": float(calibration_scale.detach().cpu()),
                    "beta": float(self.beta[modality_id].detach().cpu()),
                    "nonzero_nodes": int(nonzero_context.sum().detach().cpu()),
                    "private_response_rms": float(self._rms(responses[m]["physical"], physical_support).cpu()),
                    "collaborative_response_rms": float(self._rms(r_collab, physical_support).cpu()),
                    "sender_state_rms": float(self._rms(sender_state).cpu()),
                    "sender_private_ratio": self.sender_private_ratio if self.v13_variant == "sender_private" else 0.0,
                    "early_cross_to_neighbor_rms": mechanism_forward[f"early_cross_to_neighbor_rms_{m}"],
                }

        final: dict[str, torch.Tensor] = {}
        late_details: dict[str, Any] = {}
        for target_id, target in enumerate(_MODALITIES):
            source = "visual" if target == "text" else "text"
            final[target], late_details[target] = self._late(
                target, target_id, source, h[source], private[source], collaborative[target], collect=return_details
            )
            mechanism_forward[f"late_residual_ratio_{target}"] = float(late_details[target]["late_delta_ratio"])

        if self.training:
            self._last_train_v13_diagnostics = dict(mechanism_forward)
        else:
            self._last_val_v13_diagnostics = dict(mechanism_forward)

        if self._capture_state_audit:
            self._last_state_audit = {
                "private": private,
                "collaborative": collaborative,
                "responses": responses,
                "sender_states": {},
                "early_cross": cross_contexts,
            }
            # Sender tensors and calibrated Cross retain autograd for gradient-path tests.
            self._last_state_audit["sender_states"] = {
                "text": ((1.0 - self.sender_private_ratio) * h["text"] + self.sender_private_ratio * private["text"])
                if self.v13_variant == "sender_private" and self.sender_private_ratio > 0.0 else h["text"],
                "visual": ((1.0 - self.sender_private_ratio) * h["visual"] + self.sender_private_ratio * private["visual"])
                if self.v13_variant == "sender_private" and self.sender_private_ratio > 0.0 else h["visual"],
            }

        balance = torch.stack(balances).mean() if self.condition.startswith("M_") else x.new_zeros(())
        aux_loss = self.balance_weight * balance
        if self.training:
            self._last_train_balance_raw_tensor = balance.detach()
        else:
            self._last_balance_raw_tensor = balance.detach()
        route_snapshots: dict[str, Any] = {}
        for view_name, route in route_details.items():
            modality, relation = view_name.rsplit("_", 2)[1:]
            if route is not None:
                route_snapshots[view_name] = {
                    "route": self._compact_route_state(route, supports[modality][relation]),
                    "support": supports[modality][relation].detach(),
                }
        for modality, route in collab_routes.items():
            if route is not None:
                route_snapshots[f"collaborative_{modality}_physical"] = {
                    "route": self._compact_route_state(route, physical_support),
                    "support": physical_support.detach(),
                }
        self._last_route_snapshot = route_snapshots
        fused = torch.cat((final["text"], final["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused) + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
        )
        info: dict[str, Any] = {
            "condition": self.condition,
            "variant": self.v13_variant,
            "v13_variant": self.v13_variant,
            "active_nodes": physical_support.detach(),
            "input_self_loops_removed": loop_count,
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "unimodal_logits": {m: self.unimodal_heads[m](private[m]) for m in _MODALITIES},
            "semantic_candidate_fingerprint": self.semantic_candidate_fingerprint,
            "semantic_v7b_fingerprint": self.semantic_v7b_fingerprint,
            "test_evaluation": False,
        }
        if return_details:
            filter_summaries = filter_details
            route_summaries = {}
            for key, value in route_details.items():
                modality, relation = key.rsplit("_", 2)[1:]
                route_summaries[key] = self._route_summary(value, supports[modality][relation])
            collaborative_route_summaries = {
                m: self._route_summary(collab_routes[m], physical_support) for m in _MODALITIES
            }
            route_changes = {}
            if self.condition.startswith("M_"):
                for m in _MODALITIES:
                    before, after = route_details[f"private_{m}_physical"], collab_routes[m]
                    route_changes[m] = {
                        "dense_importance_delta": (after["importance"] - before["importance"]).detach().cpu().tolist(),
                        "top2_selection_share_delta": (after["share"] - before["share"]).detach().cpu().tolist(),
                        "node_top2_changed_fraction": float((after["selected"][physical_support] != before["selected"][physical_support]).any(dim=-1).float().mean().cpu()) if bool(physical_support.any()) else 0.0,
                    }
            info["details"] = {
                "physical_support_nodes": int(physical_support.sum().detach().cpu()),
                "semantic_support_nodes": {m: int(supports[m]["semantic"].sum().detach().cpu()) for m in _MODALITIES},
                "private_rms": {m: float(self._rms(private[m]).cpu()) for m in _MODALITIES},
                "physical_response_rms": {m: float(self._rms(responses[m]["physical"], physical_support).cpu()) for m in _MODALITIES},
                "semantic_response_rms": {m: float(self._rms(responses[m]["semantic"], supports[m]["semantic"]).cpu()) for m in _MODALITIES},
                "early_physical_cross": early_details,
                "late": late_details,
                "routes": route_summaries,
                "collaborative_routes": collaborative_route_summaries,
                "private_collaborative_route_changes": route_changes,
                "filters": filter_summaries,
                "theta_actual": self.filter_bank.theta.detach().cpu().tolist() if self.condition.startswith("M_") else self.single_filter.theta.detach().cpu().tolist(),
                "rho": self.rho.detach().cpu().tolist(),
                "gamma": self.gamma.detach().cpu().tolist(),
                "beta": self.beta.detach().cpu().tolist(),
                "balance_loss": float(balance.detach().cpu()),
                "aux_loss": float(aux_loss.detach().cpu()),
                "shared_expert_objects": len({id(e) for e in self.filter_bank.experts}) if self.condition.startswith("M_") else 0,
                "actual_experts_executed_per_filter_call": 4 if self.condition.startswith("M_") else 1,
                "fusion_input_rms": float(self._rms(fused).cpu()),
                "fused_output_rms": float(self._rms(z).cpu()),
                "v13_mechanism": dict(mechanism_forward),
            }
        self._last_diagnostics = info.get("details", {})
        return z, None, None, aux_loss, info

    def capture_gradient_diagnostics(self) -> dict[str, Any]:
        names = [
            "projectors.text.linear1.weight", "projectors.visual.linear1.weight",
            "cross_projections.visual_to_text.0.weight", "cross_projections.text_to_visual.0.weight",
            "query_projections.text.weight", "query_projections.visual.weight",
            "key_projections.text.weight", "key_projections.visual.weight",
            "source_selector.2.weight", "channel_gate.2.weight", "fusion_linear1.weight",
            "unimodal_heads.text.weight", "unimodal_heads.visual.weight",
            "rho_raw", "gamma_raw", "beta_raw", "cross_lambda_raw", "tau_raw",
        ]
        if self.condition.startswith("M_"):
            names += ["filter_bank.theta", "router.net.3.weight"]
            for expert_id in range(self.num_experts):
                names += [
                    f"filter_bank.experts.{expert_id}.net.0.weight",
                    f"filter_bank.experts.{expert_id}.net.3.weight",
                ]
        if self.v13_variant == "joint_view":
            names += ["joint_view.input.weight", "joint_view.output.weight", "joint_view.output.bias"]
        params = dict(self.named_parameters())
        result = {}
        for name in names:
            param = params[name]
            grad = param.grad
            norm = float(grad.detach().float().norm().cpu()) if grad is not None else 0.0
            result[name] = {"gradient_present": grad is not None, "gradient_finite": bool(grad is None or torch.isfinite(grad).all()), "gradient_norm": norm, "gradient_nonzero": norm > 0}
        self._last_gradient_diagnostics = result
        return result


__all__ = ["Model", "CONDITIONS", "SharedExpertBank", "GraphAwareRouter",
           "JointViewContribution", "receiver_sparse_attention"]
