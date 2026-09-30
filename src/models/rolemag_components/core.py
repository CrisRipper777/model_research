import torch
import torch.nn as nn
import torch.nn.functional as F

from .edge_features import RoleMAGEdgeFeatureBuilder
from .experts import RoleMAGComplementaryExpert, RoleMAGPolyH2Expert, RoleMAGSharedExpert
from .losses import dirichlet_kl
from .router import RoleMAGFactorizedRouter


class RoleMAGCore(nn.Module):
    def __init__(
        self,
        v_feat_dim,
        t_feat_dim,
        hidden_dim,
        dropout=0.2,
        router_hidden_dim=64,
        topk_complementary=16,
        num_queries=4,
        lambda_bias=1.0,
        eps=1e-8,
        polyh2_init=(1.0, -0.5, 0.5),
        lambda_evi=0.0,
        out_edge_ratio=1.0,
        eta_kl=1.0,
        lambda_bal=0.0,
        lambda_qca=0.0,
        tau=0.07,
        shared_num_layers=3,
        shared_edge_floor=1.0,
        shared_fusion_weight=1.0,
        beta_temperature=1.0,
        confidence_scale=1.0,
        role_prior_strength=0.35,
        beta_floor=0.25,
        beta_cap=25.0,
        lambda_role_entropy=0.0,
        lambda_gate_entropy=0.0,
        lambda_u_center=0.0,
        u_center_target=0.35,
        disable_router=False,
        disable_complementary=False,
        disable_directionality=False,
        disable_topk=False,
        disable_attention_bias=False,
        disable_heterophily=False,
        disable_gating=False,
        mix_hetero_into_shared=False,
        comp_warmup_epochs=0,
        comp_ramp_epochs=1,
        hetero_warmup_epochs=0,
        hetero_ramp_epochs=1,
        comp_schedule_scale=1.0,
        hetero_schedule_scale=1.0,
    ):
        super().__init__()
        self.v_feat_dim = v_feat_dim
        self.t_feat_dim = t_feat_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.eps = float(eps)
        self.lambda_evi = float(lambda_evi)
        self.out_edge_ratio = float(out_edge_ratio)
        self.eta_kl = float(eta_kl)
        self.lambda_bal = float(lambda_bal)
        self.lambda_qca = float(lambda_qca)
        self.tau = float(tau)
        self.shared_fusion_weight = float(shared_fusion_weight)
        self.lambda_role_entropy = float(lambda_role_entropy)
        self.lambda_gate_entropy = float(lambda_gate_entropy)
        self.lambda_u_center = float(lambda_u_center)
        self.u_center_target = float(u_center_target)
        self.disable_router = bool(disable_router)
        self.disable_complementary = bool(disable_complementary)
        self.disable_directionality = bool(disable_directionality)
        self.disable_topk = bool(disable_topk)
        self.disable_attention_bias = bool(disable_attention_bias)
        self.disable_heterophily = bool(disable_heterophily)
        self.disable_gating = bool(disable_gating)
        self.mix_hetero_into_shared = bool(mix_hetero_into_shared)
        self.comp_warmup_epochs = int(comp_warmup_epochs)
        self.comp_ramp_epochs = max(int(comp_ramp_epochs), 1)
        self.hetero_warmup_epochs = int(hetero_warmup_epochs)
        self.hetero_ramp_epochs = max(int(hetero_ramp_epochs), 1)
        self.comp_schedule_scale = float(comp_schedule_scale)
        self.hetero_schedule_scale = float(hetero_schedule_scale)
        self.current_comp_scale = self.comp_schedule_scale
        self.current_hetero_scale = self.hetero_schedule_scale

        self.v_proj = nn.Linear(v_feat_dim, hidden_dim)
        self.t_proj = nn.Linear(t_feat_dim, hidden_dim)
        self.input_skip = nn.Linear(v_feat_dim + t_feat_dim + 2, hidden_dim)

        self.base_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_builder = RoleMAGEdgeFeatureBuilder(eps=eps)
        self.router = RoleMAGFactorizedRouter(
            sem_dim=7,
            str_dim=7,
            hidden_dim=router_hidden_dim,
            eps=eps,
            beta_temperature=beta_temperature,
            confidence_scale=confidence_scale,
            role_prior_strength=role_prior_strength,
            beta_floor=beta_floor,
            beta_cap=beta_cap,
            disable_router=disable_router,
            disable_directionality=disable_directionality,
        )

        self.shared_expert = RoleMAGSharedExpert(
            input_dim=v_feat_dim + t_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=shared_num_layers,
            edge_floor=shared_edge_floor,
        )
        self.comp_expert = RoleMAGComplementaryExpert(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            topk=topk_complementary,
            lambda_bias=lambda_bias,
            dropout=dropout,
            eps=eps,
            disable_topk=disable_topk,
            disable_attention_bias=disable_attention_bias,
        )
        self.hetero_expert = RoleMAGPolyH2Expert(
            hidden_dim=hidden_dim,
            dropout=dropout,
            init_theta=polyh2_init,
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # The unified baseline protocol trains one task head over the fused
        # embedding. Keep QCA projections only when that optional model loss
        # is explicitly enabled, avoiding unused trainable parameters.
        if self.lambda_qca > 0.0:
            self.qca_proj_image = nn.Linear(hidden_dim, hidden_dim)
            self.qca_proj_text = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.qca_proj_image = None
            self.qca_proj_text = None

    def _resolve_schedule_scale(self, epoch, warmup_epochs, ramp_epochs, peak_scale):
        if epoch < warmup_epochs:
            return 0.0
        progress = (epoch - warmup_epochs + 1) / max(ramp_epochs, 1)
        return float(max(0.0, min(1.0, progress)) * peak_scale)

    def set_training_progress(self, epoch, total_epochs):
        del total_epochs
        self.current_comp_scale = self._resolve_schedule_scale(
            epoch=epoch,
            warmup_epochs=self.comp_warmup_epochs,
            ramp_epochs=self.comp_ramp_epochs,
            peak_scale=self.comp_schedule_scale,
        )
        self.current_hetero_scale = self._resolve_schedule_scale(
            epoch=epoch,
            warmup_epochs=self.hetero_warmup_epochs,
            ramp_epochs=self.hetero_ramp_epochs,
            peak_scale=self.hetero_schedule_scale,
        )

    def _aggregate_node_role_mass(self, edge_index, route, num_nodes, dtype):
        dst = edge_index[1]
        role_mass = torch.zeros(num_nodes, 3, device=dst.device, dtype=dtype)
        stacked = torch.stack([route["w_shared"], route["w_comp"], route["w_hetero"]], dim=-1).to(dtype=dtype)
        role_mass.index_add_(0, dst, stacked)
        normalizer = role_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return role_mass / normalizer

    def _sample_non_edges(self, edge_index, num_nodes, num_samples, device):
        if num_samples <= 0 or num_nodes <= 1:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        src_e = edge_index[0].to(torch.long)
        dst_e = edge_index[1].to(torch.long)
        edge_key = src_e * num_nodes + dst_e
        edge_key = torch.unique(edge_key)

        sampled_src = []
        sampled_dst = []
        sampled_key = torch.empty(0, dtype=torch.long, device=device)
        remaining = int(num_samples)
        attempts = 0
        max_attempts = 10

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            candidate_n = max(remaining * 4, 128)
            cand_src = torch.randint(0, num_nodes, (candidate_n,), device=device)
            cand_dst = torch.randint(0, num_nodes, (candidate_n,), device=device)
            valid = cand_src != cand_dst
            if not valid.any():
                continue

            cand_src = cand_src[valid]
            cand_dst = cand_dst[valid]
            cand_key = cand_src * num_nodes + cand_dst

            in_edge = torch.isin(cand_key, edge_key)
            in_sample = torch.isin(cand_key, sampled_key) if sampled_key.numel() > 0 else torch.zeros_like(in_edge)
            keep = ~(in_edge | in_sample)
            if not keep.any():
                continue

            kept_src = cand_src[keep]
            kept_dst = cand_dst[keep]
            kept_key = cand_key[keep]

            sort_idx = torch.argsort(kept_key)
            kept_key_sorted = kept_key[sort_idx]
            kept_src_sorted = kept_src[sort_idx]
            kept_dst_sorted = kept_dst[sort_idx]
            keep_first = torch.ones_like(kept_key_sorted, dtype=torch.bool)
            keep_first[1:] = kept_key_sorted[1:] != kept_key_sorted[:-1]
            kept_key = kept_key_sorted[keep_first]
            kept_src = kept_src_sorted[keep_first]
            kept_dst = kept_dst_sorted[keep_first]

            take = min(remaining, kept_src.numel())
            if take <= 0:
                continue

            sampled_src.append(kept_src[:take])
            sampled_dst.append(kept_dst[:take])
            sampled_key = torch.cat([sampled_key, kept_key[:take]], dim=0)
            remaining -= take

        if not sampled_src:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        src = torch.cat(sampled_src, dim=0)
        dst = torch.cat(sampled_dst, dim=0)
        return src, dst

    def _compute_evi_conf_loss(self, c_in, c_out):
        if c_in.numel() == 0 or c_out.numel() == 0:
            return c_in.new_tensor(0.0 if c_in.numel() > 0 else 0.0)
        in_term = -torch.log(c_in.clamp_min(self.eps)).mean()
        out_term = -torch.log((1.0 - c_out).clamp_min(self.eps)).mean()
        return in_term + out_term

    def _compute_evi_kl_loss(self, alpha_out):
        if alpha_out.numel() == 0:
            return alpha_out.new_tensor(0.0)
        alpha_target = torch.ones_like(alpha_out)
        return dirichlet_kl(alpha_out, alpha_target, eps=self.eps).mean()

    def _compute_evi_loss(self, h_v, h_t, edge_index, v_mask, t_mask, route_in):
        if self.lambda_evi <= 0.0:
            return h_v.new_tensor(0.0)

        num_edges = edge_index.size(1)
        num_nodes = h_v.size(0)
        num_out_edges = int(round(max(0.0, self.out_edge_ratio) * float(num_edges)))
        if num_out_edges <= 0:
            return h_v.new_tensor(0.0)

        out_src, out_dst = self._sample_non_edges(edge_index, num_nodes, num_out_edges, h_v.device)
        if out_src.numel() == 0:
            return h_v.new_tensor(0.0)

        out_edge_index = torch.stack([out_src, out_dst], dim=0)
        out_sem, out_str = self.edge_builder(
            h_v,
            h_t,
            out_edge_index,
            v_mask,
            t_mask,
            structure_edge_index=edge_index,
        )
        route_out = self.router(out_sem, out_str)

        loss_conf = self._compute_evi_conf_loss(route_in["c"], route_out["c"])
        loss_kl = self._compute_evi_kl_loss(route_out["alpha"])
        return self.lambda_evi * (loss_conf + self.eta_kl * loss_kl)

    def _compute_bal_loss_from_pi(self, pi):
        if pi.numel() == 0:
            return pi.new_tensor(0.0)
        importance = pi.sum(dim=0)
        mean_imp = importance.mean()
        std_imp = importance.std(unbiased=False)
        cv = std_imp / (mean_imp + self.eps)
        return cv.pow(2)

    def _compute_bal_loss(self, route):
        if self.lambda_bal <= 0.0:
            return route["pi"].new_tensor(0.0)
        return self.lambda_bal * self._compute_bal_loss_from_pi(route["pi"])

    def _compute_role_entropy_loss(self, route):
        if self.lambda_role_entropy <= 0.0:
            return route["pi"].new_tensor(0.0)
        pi = route["pi"].clamp_min(self.eps)
        entropy = -(pi * torch.log(pi)).sum(dim=-1).mean()
        return -self.lambda_role_entropy * entropy

    def _compute_gate_entropy_loss(self, gate):
        if self.lambda_gate_entropy <= 0.0:
            return gate.new_tensor(0.0)
        gate = gate.clamp_min(self.eps)
        entropy = -(gate * torch.log(gate)).sum(dim=-1).mean()
        return -self.lambda_gate_entropy * entropy

    def _compute_u_center_loss(self, route):
        if self.lambda_u_center <= 0.0:
            return route["u"].new_tensor(0.0)
        u_mean = route["u"].mean()
        return self.lambda_u_center * (u_mean - self.u_center_target).pow(2)

    def _info_nce_loss(self, anchor, target):
        if anchor.size(0) < 2:
            return anchor.new_tensor(0.0)
        anchor = F.normalize(anchor, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        temp = max(self.tau, self.eps)
        logits = torch.matmul(anchor, target.transpose(0, 1)) / temp
        labels = torch.arange(anchor.size(0), device=anchor.device)
        return F.cross_entropy(logits, labels)

    def _compute_qca_loss(self, comp_info, h_v, h_t, v_mask, t_mask):
        if self.lambda_qca <= 0.0:
            return h_v.new_tensor(0.0)
        paired = v_mask & t_mask
        index = torch.nonzero(paired, as_tuple=False).flatten()
        if index.numel() < 2:
            return h_v.new_tensor(0.0)

        z_t2i = self.qca_proj_image(comp_info["z_t2i"][index])
        z_i2t = self.qca_proj_text(comp_info["z_i2t"][index])
        v_target = self.qca_proj_image(h_v[index])
        t_target = self.qca_proj_text(h_t[index])

        loss_t2i = self._info_nce_loss(z_t2i, v_target)
        loss_i2t = self._info_nce_loss(z_i2t, t_target)
        return self.lambda_qca * 0.5 * (loss_t2i + loss_i2t)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        gate_out = self.gate[-1]
        if isinstance(gate_out, nn.Linear) and gate_out.bias is not None:
            gate_out.bias.data.copy_(torch.tensor([2.0, 0.0, 0.0], device=gate_out.bias.device))
        self.current_comp_scale = self.comp_schedule_scale
        self.current_hetero_scale = self.hetero_schedule_scale

    def forward(self, v_feat, t_feat, edge_index, v_mask=None, t_mask=None):
        if v_mask is None:
            v_mask = torch.ones(v_feat.size(0), dtype=torch.bool, device=v_feat.device)
        if t_mask is None:
            t_mask = torch.ones(t_feat.size(0), dtype=torch.bool, device=t_feat.device)

        h_v = self.v_proj(v_feat)
        h_t = self.t_proj(t_feat)

        h_v = h_v * v_mask.unsqueeze(-1).float()
        h_t = h_t * t_mask.unsqueeze(-1).float()

        mask_feature = torch.stack([v_mask.float(), t_mask.float()], dim=-1)
        raw_feature = torch.cat(
            [
                v_feat * v_mask.unsqueeze(-1).float(),
                t_feat * t_mask.unsqueeze(-1).float(),
                mask_feature,
            ],
            dim=-1,
        )
        raw_feature_shared = raw_feature[:, : self.v_feat_dim + self.t_feat_dim]
        h_base = self.input_skip(raw_feature) + self.base_fuse(torch.cat([h_v, h_t, mask_feature], dim=-1))

        v_sem, v_str = self.edge_builder(h_v, h_t, edge_index, v_mask, t_mask)
        route = self.router(v_sem, v_str)
        route = {key: value.clone() if torch.is_tensor(value) else value for key, value in route.items()}
        if self.disable_complementary:
            route["w_comp"] = torch.zeros_like(route["w_comp"])
            route["w_t2i"] = torch.zeros_like(route["w_t2i"])
            route["w_i2t"] = torch.zeros_like(route["w_i2t"])
        if self.disable_heterophily:
            route["w_hetero"] = torch.zeros_like(route["w_hetero"])

        shared_weight = route["w_shared"] + route["w_hetero"] if self.mix_hetero_into_shared else route["w_shared"]

        h_s = self.shared_expert(h_base, raw_feature_shared, edge_index, shared_weight)
        if self.disable_complementary:
            h_c = torch.zeros_like(h_base)
            comp_info = {"z_t2i": torch.zeros_like(h_base), "z_i2t": torch.zeros_like(h_base)}
        else:
            h_c, comp_info = self.comp_expert(
                h_t,
                h_v,
                edge_index,
                route["w_t2i"],
                route["w_i2t"],
                w_comp=route["w_comp"],
                return_aux=True,
            )
        h_h = torch.zeros_like(h_base) if self.disable_heterophily else self.hetero_expert(h_base, edge_index, route["w_hetero"])

        gate_input = torch.cat([h_base, h_s, h_c, h_h], dim=-1)
        gate = (
            torch.full((h_base.size(0), 3), 1.0 / 3.0, device=h_base.device, dtype=h_base.dtype)
            if self.disable_gating
            else torch.softmax(self.gate(gate_input), dim=-1)
        )
        node_role_mass = self._aggregate_node_role_mass(edge_index, route, h_base.size(0), h_base.dtype)

        h = (
            h_base
            + self.shared_fusion_weight * h_s
            + self.current_comp_scale * gate[:, [1]] * h_c
            + self.current_hetero_scale * gate[:, [2]] * h_h
        )

        routing_info = {
            "pi": route["pi"],
            "beta": route["beta"],
            "c": route["c"],
            "u": 1.0 - route["c"],
            "shared_weight": shared_weight,
            "comp_weight": route["w_comp"],
            "hetero_weight": route["w_hetero"],
            "w_t2i": route["w_t2i"],
            "w_i2t": route["w_i2t"],
            "d_t2i": route["d_t2i"],
            "d_i2t": route["d_i2t"],
            "schedule_comp_scale": h.new_tensor(self.current_comp_scale),
            "schedule_hetero_scale": h.new_tensor(self.current_hetero_scale),
            "node_role_mass": node_role_mass,
            "expert_outputs": {
                "shared": h_s,
                "comp": h_c,
                "hetero": h_h,
                "gate": gate,
            },
        }

        internal_loss = h.new_tensor(0.0)
        if self.training:
            loss_evi = self._compute_evi_loss(h_v, h_t, edge_index, v_mask, t_mask, route)
            loss_bal = self._compute_bal_loss(route)
            loss_role_entropy = self._compute_role_entropy_loss(route)
            loss_gate_entropy = self._compute_gate_entropy_loss(gate)
            loss_u_center = self._compute_u_center_loss(routing_info)
            loss_qca = h.new_tensor(0.0) if self.disable_complementary else self._compute_qca_loss(comp_info, h_v, h_t, v_mask, t_mask)
            internal_loss = loss_evi + loss_bal + loss_role_entropy + loss_gate_entropy + loss_u_center + loss_qca

        return h, routing_info, internal_loss
