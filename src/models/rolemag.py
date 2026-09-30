from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .rolemag_components.core import RoleMAGCore


class _RoleMAGBackbone(nn.Module):
    """RoleMAG's shared GCN backbone from the source implementation."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"backbone_num_layers must be >= 1, got {num_layers}")
        self.convs = nn.ModuleList(
            [GCNConv(input_dim if layer == 0 else hidden_dim, hidden_dim) for layer in range(num_layers)]
        )
        self.dropout = float(dropout)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index.to(x.device))
            if layer_idx != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Model(nn.Module):
    """RoleMAG adapted to baseline's frozen-feature NC/LP model contract.

    baseline stores concatenated features as [text, visual]; RoleMAG's core
    consumes separate visual/text tensors, so this wrapper performs that
    explicit split and leaves the task runners, splits, samplers, and metrics
    unchanged.
    """

    def __init__(self, cfg, data_info: dict):
        super().__init__()
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError(
                "rolemag requires positive text_dim and visual_dim; "
                f"got text_dim={self.text_dim}, visual_dim={self.visual_dim}"
            )
        if self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "rolemag expects baseline's concatenated [text, visual] features: "
                f"input_dim={self.input_dim}, text_dim+visual_dim={self.text_dim + self.visual_dim}"
            )

        hidden_dim = int(cfg.model.get("hidden_dim", 256))
        dropout = float(cfg.model.get("dropout", 0.2))
        backbone_num_layers = int(cfg.model.get("backbone_num_layers", 3))
        self.role_residual_weight = float(cfg.model.get("role_residual_init", 0.1))
        self.total_epochs = int(cfg.task.get("epochs", 1))

        self.backbone = _RoleMAGBackbone(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=backbone_num_layers,
            dropout=dropout,
        )
        self.core = RoleMAGCore(
            v_feat_dim=self.visual_dim,
            t_feat_dim=self.text_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            router_hidden_dim=int(cfg.model.get("router_hidden_dim", 64)),
            topk_complementary=int(cfg.model.get("topk_complementary", 16)),
            num_queries=int(cfg.model.get("num_queries", 4)),
            lambda_bias=float(cfg.model.get("lambda_bias", 1.0)),
            eps=float(cfg.model.get("eps", 1e-8)),
            polyh2_init=tuple(cfg.model.get("polyh2_init", [1.0, -0.5, 0.5])),
            lambda_evi=float(cfg.model.get("lambda_evi", 0.0)),
            out_edge_ratio=float(cfg.model.get("out_edge_ratio", 0.0)),
            eta_kl=float(cfg.model.get("eta_kl", 1.0)),
            lambda_bal=float(cfg.model.get("lambda_bal", 0.0)),
            lambda_qca=float(cfg.model.get("lambda_qca", 0.0)),
            tau=float(cfg.model.get("tau", 0.07)),
            shared_num_layers=int(cfg.model.get("shared_num_layers", 3)),
            shared_edge_floor=float(cfg.model.get("shared_edge_floor", 1.0)),
            shared_fusion_weight=float(cfg.model.get("shared_fusion_weight", 1.0)),
            beta_temperature=float(cfg.model.get("beta_temperature", 1.5)),
            confidence_scale=float(cfg.model.get("confidence_scale", 1.25)),
            role_prior_strength=float(cfg.model.get("role_prior_strength", 0.2)),
            beta_floor=float(cfg.model.get("beta_floor", 0.5)),
            beta_cap=float(cfg.model.get("beta_cap", 12.0)),
            lambda_role_entropy=float(cfg.model.get("lambda_role_entropy", 0.0)),
            lambda_gate_entropy=float(cfg.model.get("lambda_gate_entropy", 0.0)),
            lambda_u_center=float(cfg.model.get("lambda_u_center", 0.0)),
            u_center_target=float(cfg.model.get("u_center_target", 0.35)),
            disable_router=bool(cfg.model.get("disable_router", False)),
            disable_complementary=bool(cfg.model.get("disable_complementary", False)),
            disable_directionality=bool(cfg.model.get("disable_directionality", False)),
            disable_topk=bool(cfg.model.get("disable_topk", False)),
            disable_attention_bias=bool(cfg.model.get("disable_attention_bias", False)),
            disable_heterophily=bool(cfg.model.get("disable_heterophily", False)),
            disable_gating=bool(cfg.model.get("disable_gating", False)),
            mix_hetero_into_shared=bool(cfg.model.get("mix_hetero_into_shared", False)),
            comp_warmup_epochs=int(cfg.model.get("comp_warmup_epochs", 0)),
            comp_ramp_epochs=int(cfg.model.get("comp_ramp_epochs", 1)),
            hetero_warmup_epochs=int(cfg.model.get("hetero_warmup_epochs", 0)),
            hetero_ramp_epochs=int(cfg.model.get("hetero_ramp_epochs", 1)),
            comp_schedule_scale=float(cfg.model.get("comp_schedule_scale", 1.0)),
            hetero_schedule_scale=float(cfg.model.get("hetero_schedule_scale", 1.0)),
        )
        self.out_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.backbone.reset_parameters()
        self.core.reset_parameters()

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        self.core.set_training_progress(epoch=int(epoch), total_epochs=int(total_epochs))

    def set_epoch(self, epoch: int) -> None:
        # baseline task loops count epochs from 1; RoleMAG's reference schedule
        # counts from 0. Preserve the source warmup/ramp boundary.
        self.set_training_progress(epoch=max(int(epoch) - 1, 0), total_epochs=self.total_epochs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        if edge_index is None:
            raise ValueError("rolemag requires edge_index")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"rolemag expected x with shape [num_nodes, {self.input_dim}], got {tuple(x.shape)}"
            )

        text_feat = x[:, : self.text_dim]
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim]
        text_mask = torch.isfinite(text_feat).all(dim=1) & (torch.nan_to_num(text_feat).abs().sum(dim=1) > 0)
        visual_mask = torch.isfinite(visual_feat).all(dim=1) & (torch.nan_to_num(visual_feat).abs().sum(dim=1) > 0)
        text_feat = torch.nan_to_num(text_feat) * text_mask.unsqueeze(-1).to(dtype=x.dtype)
        visual_feat = torch.nan_to_num(visual_feat) * visual_mask.unsqueeze(-1).to(dtype=x.dtype)

        # RoleMAG's source order is [visual, text], while baseline loaders use
        # [text, visual]. Keep each modality-specific pathway aligned.
        backbone_input = torch.cat((visual_feat, text_feat), dim=-1)
        backbone_h = self.backbone(backbone_input, edge_index)
        role_h, _routing_info, aux_loss = self.core(
            v_feat=visual_feat,
            t_feat=text_feat,
            edge_index=edge_index,
            v_mask=visual_mask,
            t_mask=text_mask,
        )
        z = backbone_h + self.role_residual_weight * role_h
        return z, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Run the exact graph-wide RoleMAG pass and return CPU embeddings.

        Role routing uses edge pairs, neighborhood structure, and incoming
        top-k selection across the supplied graph. A layerwise decomposition
        would change those quantities, so this model uses an exact full-graph
        inference pass even when the framework's layerwise option is selected.
        """
        del batch_size
        if edge_index is None:
            raise ValueError("rolemag requires edge_index")
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()
