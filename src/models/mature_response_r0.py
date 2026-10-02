from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = (
    "smooth_base",
    "bank_generic",
    "bank_crossmoe_generic",
    "bank_crossmoe_protected",
)
NUM_EXPERTS = 3
TOP_K = 2
PROMPT_DIM = 32


def remove_self_messages(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    src, dst = edge_index.long()
    keep = src != dst
    return edge_index[:, keep], src[keep], dst[keep]


def incoming_degree(dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.long, device=dst.device)
    if dst.numel():
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.long))
    return degree


def self_anchored_low_high(
    intrinsic: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree: torch.Tensor,
    edge_chunk_size: int = 100_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Lraw=(I+sum incoming I)/(d+1), Hraw=I-Lraw, and neighbor mean."""
    if edge_chunk_size < 1:
        raise ValueError("edge_chunk_size must be positive")
    n, width = intrinsic.shape
    neighbor_sum = intrinsic.new_zeros((n, width))
    for begin in range(0, int(src.numel()), edge_chunk_size):
        end = min(begin + edge_chunk_size, int(src.numel()))
        neighbor_sum = neighbor_sum.index_add(0, dst[begin:end], intrinsic[src[begin:end]])
    denom = degree.to(intrinsic.dtype).unsqueeze(-1)
    low_raw = (intrinsic + neighbor_sum) / (denom + 1.0)
    high_raw = intrinsic - low_raw
    neighbor_mean = neighbor_sum / denom.clamp_min(1.0)
    return low_raw, high_raw, neighbor_mean


def pairwise_prompt_cosines(prompts: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    normalized = F.normalize(prompts, dim=-1, eps=eps)
    cosine = normalized @ normalized.transpose(0, 1)
    keep = ~torch.eye(prompts.size(0), dtype=torch.bool, device=prompts.device)
    return cosine[keep]


def prompt_orthogonality_loss(prompts: torch.Tensor) -> torch.Tensor:
    """Mean exp(cosine) over ordered off-diagonal prompt pairs."""
    cosine = pairwise_prompt_cosines(prompts)
    return cosine.exp().mean()


class _Projector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    """One-step mature response bank with node-level cross-modal vector experts."""

    def __init__(self, cfg: Any, data_info: dict[str, int]):
        super().__init__()
        model_cfg = cfg.model
        self.variant = str(model_cfg.get("variant", "smooth_base")).strip().lower()
        if self.variant not in VARIANTS:
            raise ValueError(f"model.variant must be one of {VARIANTS}, got {self.variant!r}")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        self.edge_chunk_size = int(model_cfg.get("edge_chunk_size", 100_000))
        self.orth_weight = float(model_cfg.get("orth_weight", 1e-3))
        self.dropout = float(model_cfg.get("dropout", 0.2))
        if self.hidden_dim != 128:
            raise ValueError("R0 fixes hidden_dim=128")
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        if self.orth_weight != 1e-3:
            raise ValueError("R0 fixes orth_weight=1e-3")

        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim + self.visual_dim != int(data_info["input_dim"]):
            raise ValueError("Text and Visual dimensions must cover the joint input")
        if min(self.text_dim, self.visual_dim) < 1:
            raise ValueError("R0 requires nonempty Text and Visual feature slices")

        # Frozen construction order: semantic projectors; response transforms and
        # norms; final fusion; CrossMoE; generic blocks; protected attention/FFN.
        self.proj_t = _Projector(self.text_dim, self.hidden_dim, self.dropout)
        self.proj_v = _Projector(self.visual_dim, self.hidden_dim, self.dropout)
        self.low_transforms = nn.ModuleList(
            [nn.Linear(128, 128, bias=False) for _ in range(2)]
        )
        self.high_transforms = nn.ModuleList(
            [nn.Linear(128, 128, bias=False) for _ in range(2)]
        )
        for layer in (*self.low_transforms, *self.high_transforms):
            nn.init.xavier_uniform_(layer.weight)
        self.low_response_norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
        self.high_response_norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
        self.smooth_residual_norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
        self.fusion = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(256, 128), nn.LayerNorm(128),
        )

        self.cross_prompts = nn.ParameterList(
            [nn.Parameter(torch.empty(NUM_EXPERTS, PROMPT_DIM)) for _ in range(2)]
        )
        for prompts in self.cross_prompts:
            nn.init.normal_(prompts, mean=0.0, std=0.02)

        self.cross_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(416, 128), nn.GELU(), nn.Dropout(self.dropout),
                    nn.Linear(128, 128),
                )
                for _ in range(NUM_EXPERTS)
            ])
            for _ in range(2)
        ])
        self.routers = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, NUM_EXPERTS))
            for _ in range(2)
        ])
        for router in self.routers:
            nn.init.normal_(router[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(router[-1].bias)

        self.generic_composers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 103), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(103, 128)
            )
            for _ in range(2)
        ])
        self.protected_attention = nn.ModuleList([
            nn.MultiheadAttention(128, 4, dropout=self.dropout, batch_first=True)
            for _ in range(2)
        ])
        # Both composer types use the same per-modality residual and post-FFN
        # parameters, making the first-stage capacity comparison transparent.
        self.response_residual_norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
        self.post_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 256), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(256, 128)
            )
            for _ in range(2)
        ])
        self.final_residual_norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(2)])
        self.out_dim = 128

    def split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError(f"expected joint Text/Visual features, got {tuple(x.shape)}")
        return x[:, :self.text_dim], x[:, self.text_dim:]

    def _cross_response(
        self,
        intrinsic: list[torch.Tensor],
        low: list[torch.Tensor],
        high: list[torch.Tensor],
        routing_override: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        return_diagnostics: bool,
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        outputs, diagnostics = [], []
        for target_modality in range(2):
            other = 1 - target_modality
            other_state = torch.cat([intrinsic[other], low[other], high[other]], dim=-1)
            router_input = torch.cat([intrinsic[target_modality], other_state], dim=-1)
            logits = self.routers[target_modality](router_input)
            selected_logits, selected_indices = logits.topk(TOP_K, dim=-1, sorted=True)
            selected_weights = torch.softmax(selected_logits, dim=-1)
            if routing_override is not None:
                selected_indices = routing_override[target_modality][0].to(selected_indices.device)
                selected_weights = routing_override[target_modality][1].to(
                    device=selected_weights.device, dtype=selected_weights.dtype
                )
            expert_outputs = []
            for expert_index, expert in enumerate(self.cross_experts[target_modality]):
                prompt = self.cross_prompts[target_modality][expert_index].expand(other_state.size(0), -1)
                expert_input = torch.cat([other_state, prompt], dim=-1)
                expert_outputs.append(expert(expert_input))
            expert_stack = torch.stack(expert_outputs, dim=1)
            selected_outputs = expert_stack.gather(
                1, selected_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )
            cross = (selected_outputs * selected_weights.unsqueeze(-1)).sum(dim=1)
            outputs.append(cross)
            if return_diagnostics:
                diagnostics.append({
                    "router_logits": logits,
                    "top_indices": selected_indices,
                    "top_weights": selected_weights,
                    "expert_outputs": expert_stack,
                })
        return outputs, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        response_overrides: dict[str, Any] | None = None,
        routing_override: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ):
        if edge_index is None:
            raise ValueError("R0 requires the original physical edge_index")
        x_t, x_v = self.split_modalities(x)
        intrinsic = [self.proj_t(x_t), self.proj_v(x_v)]
        _, src, dst = remove_self_messages(edge_index)
        degree = incoming_degree(dst, int(x.size(0)))

        raw_low, raw_high, neighbor_mean = [], [], []
        low, high, smooth_delta = [], [], []
        for modality in range(2):
            lraw, hraw, nmean = self_anchored_low_high(
                intrinsic[modality], src, dst, degree, self.edge_chunk_size
            )
            raw_low.append(lraw); raw_high.append(hraw); neighbor_mean.append(nmean)
            low.append(self.low_response_norms[modality](self.low_transforms[modality](lraw)))
            high.append(self.high_response_norms[modality](self.high_transforms[modality](hraw)))

            # Match M0/N1 uniform propagation: transformed incoming-neighbor mean
            # plus the intrinsic residual. For d=0, the mean is exactly zero.
            smooth_delta.append(self.low_transforms[modality](nmean))

        override = response_overrides or {}
        if "L" in override:
            low = [override["L"][m].to(device=low[m].device, dtype=low[m].dtype) for m in range(2)]
        if "H" in override:
            high = [override["H"][m].to(device=high[m].device, dtype=high[m].dtype) for m in range(2)]

        cross = [intrinsic[0].new_zeros(intrinsic[0].shape), intrinsic[1].new_zeros(intrinsic[1].shape)]
        moe_info: list[dict[str, torch.Tensor]] = []
        orth_loss = x.new_zeros(())
        cross_active = self.variant in {"bank_crossmoe_generic", "bank_crossmoe_protected"}
        if cross_active:
            cross, moe_info = self._cross_response(
                intrinsic, low, high, routing_override, return_diagnostics
            )
            orth_loss = sum(prompt_orthogonality_loss(prompt) for prompt in self.cross_prompts)
        if "X" in override:
            cross = [override["X"][m].to(device=cross[m].device, dtype=cross[m].dtype) for m in range(2)]

        structural, attention, composition_response, composer_inputs = [], [], [], []
        protected_queries, protected_tokens, pre_ffn = [], [], []
        if self.variant == "smooth_base":
            for modality in range(2):
                structural.append(self.smooth_residual_norms[modality](intrinsic[modality] + smooth_delta[modality]))
                attention.append(None)
                composition_response.append(smooth_delta[modality])
                composer_inputs.append(None); protected_queries.append(None); protected_tokens.append(None)
                pre_ffn.append(None)
        else:
            for modality in range(2):
                if self.variant == "bank_crossmoe_protected":
                    query = intrinsic[modality].unsqueeze(1)
                    tokens = torch.stack([low[modality], high[modality], cross[modality]], dim=1)
                    response, weights = self.protected_attention[modality](
                        query, tokens, tokens, need_weights=True, average_attn_weights=False
                    )
                    response = response.squeeze(1)
                    attention.append(weights.squeeze(2))
                    protected_queries.append(query)
                    protected_tokens.append(tokens)
                    composer_inputs.append(torch.cat(
                        [intrinsic[modality], low[modality], high[modality], cross[modality]], dim=-1
                    ))
                else:
                    x_cross = cross[modality] if cross_active else torch.zeros_like(cross[modality])
                    composer_input = torch.cat(
                        [intrinsic[modality], low[modality], high[modality], x_cross], dim=-1
                    )
                    response = self.generic_composers[modality](composer_input)
                    attention.append(None)
                    protected_queries.append(None); protected_tokens.append(None)
                    composer_inputs.append(composer_input)
                u = self.response_residual_norms[modality](intrinsic[modality] + response)
                structural.append(self.final_residual_norms[modality](u + self.post_ffns[modality](u)))
                composition_response.append(response)
                pre_ffn.append(u)

        z = self.fusion(torch.cat([intrinsic[0], intrinsic[1], structural[0], structural[1]], dim=-1))
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
            "structural": structural,
            "composition_response": composition_response,
            "composer_inputs": composer_inputs,
            "protected_queries": protected_queries,
            "protected_tokens": protected_tokens,
            "pre_ffn": pre_ffn,
            "degree": degree,
            "src": src,
            "dst": dst,
            "orth_loss": orth_loss,
            "cross_active": cross_active,
            "moe": moe_info,
            "attention": attention,
        }
        if return_diagnostics:
            return z, intrinsic, structural, orth_loss, aux
        # The common NC task expects an auxiliary loss even when it is zero.
        return z, intrinsic, structural, orth_loss, aux


def parameter_counts(model: Model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    generic_first = sum(parameter.numel() for module in model.generic_composers for parameter in module.parameters())
    protected_first = sum(parameter.numel() for module in model.protected_attention for parameter in module.parameters())
    common_second = sum(
        parameter.numel()
        for modules in (model.response_residual_norms, model.post_ffns, model.final_residual_norms)
        for module in modules for parameter in module.parameters()
    )
    return {
        "model_total": int(total),
        "model_trainable": int(trainable),
        "generic_first_stage_both_modalities": int(generic_first),
        "protected_mha_both_modalities": int(protected_first),
        "common_post_composer_both_modalities": int(common_second),
        "generic_active_composer_both_modalities": int(generic_first + common_second),
        "protected_active_composer_both_modalities": int(protected_first + common_second),
    }


def copy_n1_smooth_common_weights(source: nn.Module, target: Model) -> None:
    """Map the N1 SmoothOnly common path into R0 for forward compatibility."""
    with torch.no_grad():
        target.proj_t.load_state_dict(source.proj_t.state_dict())
        target.proj_v.load_state_dict(source.proj_v.state_dict())
        for modality in range(2):
            target.low_transforms[modality].load_state_dict(source.w_s[modality].state_dict())
            target.smooth_residual_norms[modality].load_state_dict(source.residual_norms[modality].state_dict())
        target.fusion.load_state_dict(source.fusion.state_dict())


def copy_m0_uniform_common_weights(source: nn.Module, target: Model) -> None:
    with torch.no_grad():
        target.proj_t.load_state_dict(source.proj_t.state_dict())
        target.proj_v.load_state_dict(source.proj_v.state_dict())
        for modality in range(2):
            target.low_transforms[modality].load_state_dict(source.w0[modality].state_dict())
            target.smooth_residual_norms[modality].load_state_dict(source.residual_norms[modality].state_dict())
        target.fusion.load_state_dict(source.fusion.state_dict())
