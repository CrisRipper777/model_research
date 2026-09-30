import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleMAGFactorizedRouter(nn.Module):
    def __init__(
        self,
        sem_dim,
        str_dim,
        hidden_dim,
        eps=1e-8,
        beta_temperature=1.0,
        confidence_scale=1.0,
        role_prior_strength=0.35,
        beta_floor=0.25,
        beta_cap=25.0,
        disable_router=False,
        disable_directionality=False,
    ):
        super().__init__()
        self.eps = eps
        self.beta_temperature = float(beta_temperature)
        self.confidence_scale = float(confidence_scale)
        self.role_prior_strength = float(role_prior_strength)
        self.beta_floor = float(beta_floor)
        self.beta_cap = float(beta_cap)
        self.disable_router = bool(disable_router)
        self.disable_directionality = bool(disable_directionality)

        self.g_t = nn.Sequential(nn.Linear(sem_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.g_i = nn.Sequential(nn.Linear(sem_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.g_sem = nn.Sequential(nn.Linear(sem_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.g_str = nn.Sequential(nn.Linear(str_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _compute_structural_role_prior(self, v_sem, v_str):
        s_t = v_sem[:, 0]
        s_i = v_sem[:, 1]
        delta_ti = v_sem[:, 2]

        observed = v_str[:, 0]
        common_count = v_str[:, 1]
        jaccard = v_str[:, 2]
        aa = v_str[:, 3]

        mean_sim = 0.5 * (s_t + s_i)
        sim_agreement = (1.0 - 0.5 * delta_ti).clamp(0.0, 1.0)
        sim_high = (0.5 * (mean_sim + 1.0)).clamp(0.0, 1.0)
        sim_low = 1.0 - sim_high

        structural_support = torch.sigmoid(2.0 * observed + 0.25 * common_count + 0.5 * jaccard + 0.1 * aa)

        prior_shared = structural_support * sim_high * sim_agreement
        prior_hetero = structural_support * sim_low * sim_agreement
        prior_comp = structural_support * (1.0 - sim_agreement) + 0.25 * delta_ti.clamp_min(0.0)

        prior = torch.stack([prior_shared, prior_comp, prior_hetero], dim=-1).clamp_min(self.eps)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return prior

    def forward(self, v_sem, v_str):
        rho_t = torch.sigmoid(self.g_t(v_sem)).squeeze(-1)
        rho_i = torch.sigmoid(self.g_i(v_sem)).squeeze(-1)

        pi_s = rho_t * rho_i
        pi_h = (1.0 - rho_t) * (1.0 - rho_i)
        pi_c = rho_t * (1.0 - rho_i) + (1.0 - rho_t) * rho_i
        pi = torch.stack([pi_s, pi_c, pi_h], dim=-1)
        if self.disable_router:
            pi = torch.full_like(pi, 1.0 / 3.0)
        elif self.role_prior_strength > 0.0:
            prior = self._compute_structural_role_prior(v_sem, v_str)
            mix = min(max(self.role_prior_strength, 0.0), 1.0)
            pi = (1.0 - mix) * pi + mix * prior
            pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        s_sem = F.softplus(self.g_sem(v_sem)).squeeze(-1) + self.eps
        s_str = F.softplus(self.g_str(v_str)).squeeze(-1) + self.eps
        beta = (s_sem * s_str) / max(self.beta_temperature, self.eps)
        beta = beta.clamp(min=self.beta_floor, max=self.beta_cap)

        evidence = beta.unsqueeze(-1) * pi
        alpha = 1.0 + evidence

        calibrated_beta = self.confidence_scale * beta
        c = calibrated_beta / (3.0 + calibrated_beta)
        role_weight = c.unsqueeze(-1) * pi

        d_t2i_unnorm = rho_t * (1.0 - rho_i)
        d_i2t_unnorm = (1.0 - rho_t) * rho_i
        denom = d_t2i_unnorm + d_i2t_unnorm + self.eps

        d_t2i = d_t2i_unnorm / denom
        d_i2t = 1.0 - d_t2i
        if self.disable_directionality:
            d_t2i = torch.full_like(d_t2i, 0.5)
            d_i2t = torch.full_like(d_i2t, 0.5)

        w_shared = role_weight[:, 0]
        w_comp = role_weight[:, 1]
        w_hetero = role_weight[:, 2]
        w_t2i = w_comp * d_t2i
        w_i2t = w_comp * d_i2t

        return {
            "rho_t": rho_t,
            "rho_i": rho_i,
            "pi": pi,
            "beta": beta,
            "calibrated_beta": calibrated_beta,
            "evidence": evidence,
            "alpha": alpha,
            "c": c,
            "w_shared": w_shared,
            "w_comp": w_comp,
            "w_hetero": w_hetero,
            "w_t2i": w_t2i,
            "w_i2t": w_i2t,
            "d_t2i": d_t2i,
            "d_i2t": d_i2t,
        }
