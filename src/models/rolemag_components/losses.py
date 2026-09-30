import torch
import torch.nn.functional as F


def dirichlet_kl(alpha_p, alpha_q, eps=1e-8):
    """KL(Dir(alpha_p) || Dir(alpha_q)) for batched concentration parameters."""
    alpha_p = alpha_p.clamp_min(eps)
    alpha_q = alpha_q.clamp_min(eps)

    sum_p = alpha_p.sum(dim=-1, keepdim=True)
    sum_q = alpha_q.sum(dim=-1, keepdim=True)

    ln_b_p = torch.lgamma(alpha_p).sum(dim=-1) - torch.lgamma(sum_p).squeeze(-1)
    ln_b_q = torch.lgamma(alpha_q).sum(dim=-1) - torch.lgamma(sum_q).squeeze(-1)

    digamma_diff = torch.digamma(alpha_p) - torch.digamma(sum_p)
    kl = ln_b_q - ln_b_p + ((alpha_p - alpha_q) * digamma_diff).sum(dim=-1)
    return kl


def pn_rkl_loss(alpha_pred, pi_detached, beta_in, eps=1e-8):
    """Reverse-KL style edge target with stopgrad argmax(pi)."""
    num_classes = alpha_pred.size(-1)
    target_class = pi_detached.argmax(dim=-1)
    one_hot = F.one_hot(target_class, num_classes=num_classes).float()
    alpha_target = torch.ones_like(alpha_pred) + float(beta_in) * one_hot
    return dirichlet_kl(alpha_pred, alpha_target, eps=eps).mean()
