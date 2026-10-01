from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.adaptive_prop_m0 import (
    _Projector,
    fixed_degree_mean,
    incoming_degree,
    remove_self_messages,
)


VARIANTS = {
    "smooth_only",
    "static_strength",
    "same_state_strength",
    "cross_state_strength",
}
MODALITIES = ("text", "visual")
FUNCTIONS = ("smooth", "absdiff", "product")
MASTER_BLOCKS = (
    "h0_text", "h0_visual", "smooth_text", "absdiff_text", "product_text",
    "smooth_visual", "absdiff_visual", "product_visual",
)
BLOCK_DIM = 128
MASTER_DIM = len(MASTER_BLOCKS) * BLOCK_DIM
SOFTPLUS_INV_01 = -2.2521684610440906


def _feature_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=-1, eps=eps)


class Model(nn.Module):
    """Recipient-conditioned whole-neighborhood strength screen (N1)."""

    def __init__(self, cfg: Any, data_info: dict[str, int]):
        super().__init__()
        model_cfg = cfg.model
        self.variant = str(model_cfg.get("variant", "smooth_only")).strip().lower()
        if self.variant not in VARIANTS:
            raise ValueError(f"model.variant must be one of {sorted(VARIANTS)}, got {self.variant!r}")
        self.hidden_dim = int(model_cfg.get("hidden_dim", 128))
        if self.hidden_dim != BLOCK_DIM:
            raise ValueError(f"N1 fixes hidden_dim={BLOCK_DIM}, got {self.hidden_dim}")
        self.edge_chunk_size = int(model_cfg.get("edge_chunk_size", 100000))
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if self.text_dim < 1 or self.visual_dim < 1:
            raise ValueError("N1 requires non-empty Text and Visual feature slices")
        if self.text_dim + self.visual_dim != int(data_info["input_dim"]):
            raise ValueError("Text/Visual feature dimensions do not cover data.x")

        dropout = float(model_cfg.get("dropout", 0.2))
        # Preserve M0-UNI's common parameter construction and initialization order.
        self.proj_t = _Projector(self.text_dim, self.hidden_dim, dropout)
        self.proj_v = _Projector(self.visual_dim, self.hidden_dim, dropout)
        self.w_s = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2)]
        )
        for layer in self.w_s:
            nn.init.xavier_uniform_(layer.weight)
        self.fusion = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.residual_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(2)])

        # Added modules are created only after every M0-compatible common module.
        self.w_d = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2)]
        )
        self.w_p = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2)]
        )
        for layer in list(self.w_d) + list(self.w_p):
            nn.init.xavier_uniform_(layer.weight)
        self.strength_mixers = nn.ModuleList(
            [nn.Sequential(nn.Linear(MASTER_DIM, 64), nn.GELU(), nn.Linear(64, 2)) for _ in range(2)]
        )
        for mixer in self.strength_mixers:
            first, second = mixer[0], mixer[2]
            nn.init.zeros_(first.bias)
            nn.init.normal_(second.weight, mean=0.0, std=1e-3)
            nn.init.constant_(second.bias, SOFTPLUS_INV_01)
        self.out_dim = self.hidden_dim

    def split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError(
                f"expected data.x with [Text({self.text_dim}), Visual({self.visual_dim})], "
                f"got {tuple(x.shape)}"
            )
        return x[:, : self.text_dim], x[:, self.text_dim :]

    def _master_state(self, h0, contexts) -> torch.Tensor:
        return torch.cat(
            [h0[0], h0[1], contexts[0][0], contexts[0][1], contexts[0][2],
             contexts[1][0], contexts[1][1], contexts[1][2]],
            dim=-1,
        )

    @staticmethod
    def _mask_for_variant(master: torch.Tensor, variant: str, modality: int) -> torch.Tensor:
        if variant == "static_strength":
            return torch.zeros_like(master)
        if variant == "same_state_strength":
            mask = torch.zeros_like(master)
            blocks = (0, 2, 3, 4) if modality == 0 else (1, 5, 6, 7)
            for block in blocks:
                begin = block * BLOCK_DIM
                mask[:, begin : begin + BLOCK_DIM] = master[:, begin : begin + BLOCK_DIM]
            return mask
        if variant in {"cross_state_strength", "smooth_only"}:
            return master
        raise ValueError(f"unknown variant {variant!r}")

    def _contexts(self, h0, src: torch.Tensor, dst: torch.Tensor, degree: torch.Tensor):
        num_nodes = h0[0].size(0)
        contexts = [[h0[0].new_zeros((num_nodes, self.hidden_dim)) for _ in range(3)] for _ in range(2)]
        # Keep both Smooth reductions in the same modality-major order as M0 UNI.
        # CUDA index_add uses atomic accumulation, so interleaving unrelated
        # channel kernels can change the last few floating-point bits.
        for modality, h in enumerate(h0):
            for begin in range(0, src.numel(), self.edge_chunk_size):
                end = min(begin + self.edge_chunk_size, src.numel())
                edge_src, edge_dst = src[begin:end], dst[begin:end]
                message = self.w_s[modality](h[edge_src])
                contexts[modality][0] = contexts[modality][0].index_add(0, edge_dst, message)
        for begin in range(0, src.numel(), self.edge_chunk_size):
            end = min(begin + self.edge_chunk_size, src.numel())
            edge_src, edge_dst = src[begin:end], dst[begin:end]
            for modality, h in enumerate(h0):
                h_i, h_j = h[edge_dst], h[edge_src]
                messages = (self.w_d[modality](torch.abs(h_j - h_i)),
                            self.w_p[modality](h_i * h_j))
                for channel, message in enumerate(messages, start=1):
                    contexts[modality][channel] = contexts[modality][channel].index_add(
                        0, edge_dst, message
                    )
        for modality in range(2):
            contexts[modality] = [fixed_degree_mean(value, degree) for value in contexts[modality]]
        return contexts

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        beta_overrides: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_diagnostics: bool = False,
    ):
        if edge_index is None:
            raise ValueError("N1 requires the physical edge_index")
        x_t, x_v = self.split_modalities(x)
        h0 = [self.proj_t(x_t), self.proj_v(x_v)]
        edge_index_nonself, src, dst = remove_self_messages(edge_index)
        degree = incoming_degree(dst, int(x.size(0)))
        contexts = self._contexts(h0, src, dst, degree)
        master = self._master_state(h0, contexts)
        if master.size(1) != MASTER_DIM:
            raise AssertionError(f"master_dim={master.size(1)} != {MASTER_DIM}")

        betas = []
        for modality in range(2):
            mixer_input = self._mask_for_variant(master, self.variant, modality)
            raw = self.strength_mixers[modality](mixer_input)
            beta = F.softplus(raw)
            betas.append(beta)
        if beta_overrides is not None:
            if len(beta_overrides) != 2 or any(value.shape != (x.size(0), 2) for value in beta_overrides):
                raise ValueError("beta_overrides must contain Text and Visual tensors of shape [N, 2]")
            betas = [value.to(device=x.device, dtype=x.dtype) for value in beta_overrides]
        if self.variant == "smooth_only" and beta_overrides is None:
            betas = [value.new_zeros(value.shape) for value in betas]

        updates = []
        for modality in range(2):
            c_s, c_d, c_p = contexts[modality]
            beta_d, beta_p = betas[modality].unbind(dim=-1)
            if self.variant == "smooth_only" and beta_overrides is None:
                updates.append(c_s)
            else:
                updates.append(c_s + beta_d.unsqueeze(-1) * c_d + beta_p.unsqueeze(-1) * c_p)
        h_tilde = [self.residual_norms[m](h0[m] + updates[m]) for m in range(2)]
        z = self.fusion(torch.cat([h0[0], h0[1], h_tilde[0], h_tilde[1]], dim=-1))
        aux_info: dict[str, Any] = {
            "num_nonself_messages": int(src.numel()),
            "degree": degree.detach(),
        }
        if return_diagnostics:
            aux_info.update({
                "h0": tuple(value.detach() for value in h0),
                "contexts": tuple(tuple(value.detach() for value in pair) for pair in contexts),
                "h_tilde": tuple(value.detach() for value in h_tilde),
                "betas": tuple(value.detach() for value in betas),
                "edge_index_nonself": edge_index_nonself.detach(),
                "master_dim": int(master.size(1)),
            })
        return z, None, None, z.new_zeros(()), aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device | str | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self(x.to(device), edge_index.to(device))
        return z.detach().cpu()


def common_parameter_names(model: Model) -> tuple[str, ...]:
    prefixes = ("proj_t.", "proj_v.", "w_s.", "fusion.", "residual_norms.")
    return tuple(name for name, _ in model.named_parameters() if name.startswith(prefixes))


def parameter_counts(model: Model) -> dict[str, int]:
    groups = {
        "projectors": ("proj_t.", "proj_v."),
        "smooth": ("w_s.",),
        "alternative_functions": ("w_d.", "w_p."),
        "strength_mixers": ("strength_mixers.",),
        "residual_fusion": ("fusion.", "residual_norms."),
    }
    counts = {key: sum(p.numel() for name, p in model.named_parameters() if name.startswith(prefix))
              for key, prefix in groups.items()}
    counts["model_total"] = sum(p.numel() for p in model.parameters())
    return counts


def copy_m0_uniform_common_weights(m0_model: nn.Module, n1_model: Model) -> None:
    """Map M0-UNI common weights into N1's Smooth path."""
    source, target = m0_model.state_dict(), n1_model.state_dict()
    pairs = []
    for prefix in ("proj_t.", "proj_v.", "fusion.", "residual_norms."):
        pairs.extend((name, name) for name in source if name.startswith(prefix))
    pairs.extend((f"w0.{m}.weight", f"w_s.{m}.weight") for m in range(2))
    with torch.no_grad():
        for old, new in pairs:
            if old not in source or new not in target or source[old].shape != target[new].shape:
                raise AssertionError(f"cannot map M0 UNI parameter {old} to N1 {new}")
            target[new].copy_(source[old])
    n1_model.load_state_dict(target, strict=True)
