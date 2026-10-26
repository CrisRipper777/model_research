"""Numerically explicit, blockwise diagnostics for small spans of node states."""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import torch


def _select(value: torch.Tensor, support: torch.Tensor | None) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"expected [nodes, features], got {tuple(value.shape)}")
    return value if support is None else value[support]


def _block_slices(n: int, block_size: int):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    for start in range(0, n, block_size):
        yield slice(start, min(start + block_size, n))


def pairwise_gram(
    values: Iterable[torch.Tensor],
    support: torch.Tensor | None = None,
    *,
    block_size: int = 8192,
) -> torch.Tensor:
    """FP64 Frobenius Gram, accumulated in node blocks; input tensors stay put."""
    columns = list(values)
    if not columns:
        return torch.empty((0, 0), dtype=torch.float64)
    if any(x.ndim != 2 or x.shape != columns[0].shape for x in columns):
        raise ValueError("all values must have the same [nodes, features] shape")
    if support is not None and (support.ndim != 1 or support.numel() != columns[0].size(0)):
        raise ValueError("support must be a [nodes] mask")
    k = len(columns)
    gram = torch.zeros((k, k), dtype=torch.float64, device=columns[0].device)
    for sl in _block_slices(columns[0].size(0), block_size):
        mask = support[sl] if support is not None else None
        block = torch.stack([_select(x[sl], mask).reshape(-1).to(torch.float64) for x in columns])
        if block.size(1):
            gram.add_(block @ block.T)
    return (gram + gram.T) * 0.5


def vector_norms(
    values: Iterable[torch.Tensor], support: torch.Tensor | None = None, *, block_size: int = 8192
) -> torch.Tensor:
    gram = pairwise_gram(values, support, block_size=block_size)
    return gram.diagonal().clamp_min(0).sqrt()


def global_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    support: torch.Tensor | None = None,
    *,
    block_size: int = 8192,
    eps: float = 1.0e-30,
) -> dict:
    gram = pairwise_gram((left, right), support, block_size=block_size)
    a, b = float(gram[0, 0]), float(gram[1, 1])
    if a <= eps or b <= eps:
        return {"value": None, "status": "zero_norm_operand", "left_norm": math.sqrt(max(a, 0.0)),
                "right_norm": math.sqrt(max(b, 0.0))}
    return {"value": float(gram[0, 1] / math.sqrt(a * b)), "status": "ok",
            "left_norm": math.sqrt(a), "right_norm": math.sqrt(b)}


def rms(
    value: torch.Tensor, support: torch.Tensor | None = None, *, block_size: int = 8192
) -> dict:
    selected_count = int(support.sum().item()) if support is not None else value.size(0)
    if selected_count == 0 or value.size(1) == 0:
        return {"value": None, "status": "empty_support", "node_count": selected_count}
    norm_sq = float(pairwise_gram((value,), support, block_size=block_size)[0, 0])
    count = selected_count * value.size(1)
    return {"value": math.sqrt(max(norm_sq, 0.0) / count), "status": "ok", "node_count": selected_count}


def node_cosines(
    left: torch.Tensor,
    right: torch.Tensor,
    support: torch.Tensor,
    *,
    near_zero: float = 1.0e-12,
) -> dict:
    n_supported = int(support.sum().item())
    if n_supported == 0:
        return {"status": "empty_support", "valid_nodes": 0, "zero_or_near_zero_nodes": 0,
                "p10": None, "median": None, "p90": None}
    result = []
    skipped = 0
    # Bound temporary GPU memory regardless of graph size.
    for sl in _block_slices(left.size(0), 8192):
        active = support[sl]
        if not bool(active.any()):
            continue
        a, b = left[sl][active].float(), right[sl][active].float()
        na, nb = a.norm(dim=-1), b.norm(dim=-1)
        valid = (na > near_zero) & (nb > near_zero)
        skipped += int((~valid).sum().item())
        if bool(valid.any()):
            cos = ((a[valid] * b[valid]).sum(dim=-1) / (na[valid] * nb[valid])).clamp(-1.0, 1.0)
            result.append(cos.detach().cpu())
    if not result:
        return {"status": "no_nonzero_pairs", "valid_nodes": 0, "zero_or_near_zero_nodes": skipped,
                "p10": None, "median": None, "p90": None}
    cos = torch.cat(result)
    q = torch.quantile(cos, torch.tensor([0.1, 0.5, 0.9]))
    return {"status": "ok", "valid_nodes": int(cos.numel()),
            "zero_or_near_zero_nodes": skipped, "zero_or_near_zero_fraction": skipped / n_supported,
            "p10": float(q[0]), "median": float(q[1]), "p90": float(q[2])}


def _spectrum_from_gram(gram: torch.Tensor, ambient_dim: int) -> dict:
    if gram.numel() == 0:
        return {"singular_values": [], "effective_rank": 0.0, "rank": 0,
                "rank_tolerance": 0.0, "condition_number": None, "status": "empty_basis"}
    symmetric = (gram + gram.T) * 0.5
    eig, _ = torch.linalg.eigh(symmetric)
    largest = max(float(eig[-1]), 0.0)
    eig_tol = max(ambient_dim, gram.size(0)) * torch.finfo(torch.float64).eps * max(largest, 1.0)
    min_eig = float(eig[0])
    if min_eig < -max(eig_tol * 16, 1.0e-13):
        return {"singular_values": [], "effective_rank": None, "rank": None,
                "rank_tolerance": None, "condition_number": None, "status": "invalid_negative_gram_eigenvalue",
                "minimum_gram_eigenvalue": min_eig}
    singular = eig.clamp_min(0).flip(0).sqrt()
    sigma_max = float(singular[0]) if singular.numel() else 0.0
    # A Gram eigensolve loses roughly half the digits of a direct tall-matrix SVD.
    tol = math.sqrt(max(ambient_dim, gram.size(0)) * torch.finfo(torch.float64).eps) * sigma_max
    retained = singular[singular > tol] if sigma_max > 0 else singular[:0]
    rank = int(retained.numel())
    if rank:
        probabilities = retained / retained.sum()
        effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
        condition = float(retained[0] / retained[-1])
    else:
        effective_rank, condition = 0.0, None
    return {"singular_values": [float(x) for x in singular], "effective_rank": effective_rank,
            "rank": rank, "rank_tolerance": tol, "condition_number": condition,
            "status": "ok" if rank else "zero_rank", "minimum_gram_eigenvalue": min_eig,
            "gram_eigenvalue_tolerance": eig_tol}


def basis_spectrum(
    values: Iterable[torch.Tensor], support: torch.Tensor | None = None, *, block_size: int = 8192
) -> dict:
    columns = list(values)
    if not columns:
        return _spectrum_from_gram(torch.empty((0, 0), dtype=torch.float64), 0)
    gram = pairwise_gram(columns, support, block_size=block_size)
    norms = gram.diagonal().clamp_min(0).sqrt()
    nonzero = norms > 0
    scales = torch.where(nonzero, 1.0 / norms, torch.zeros_like(norms))
    normalized_gram = gram * scales[:, None] * scales[None, :]
    ambient_dim = (int(support.sum().item()) if support is not None else columns[0].size(0)) * columns[0].size(1)
    result = _spectrum_from_gram(normalized_gram, ambient_dim)
    result.update({"normalization_scales": [float(x) for x in scales],
                   "zero_columns": int((~nonzero).sum().item()),
                   "normalized_gram": normalized_gram.cpu().tolist(),
                   "support_coordinate_count": ambient_dim})
    return result


def span_novelty(
    basis: Iterable[torch.Tensor],
    target: torch.Tensor,
    support: torch.Tensor | None = None,
    *,
    block_size: int = 8192,
) -> dict:
    """Novelty by a scaled FP64 Gram pseudoinverse and direct blocked residual."""
    columns = list(basis)
    if any(x.shape != target.shape for x in columns):
        raise ValueError("basis and target must share shape")
    if not columns:
        target_rms = rms(target, support, block_size=block_size)
        return {"novelty": 1.0 if target_rms["value"] not in (None, 0.0) else 0.0,
                "target_norm": 0.0 if target_rms["value"] is None else target_rms["value"],
                "status": "empty_basis" if target_rms["value"] else "zero_target_undefined_ratio",
                "residual_norm": None if target_rms["value"] is None else target_rms["value"],
                "spectrum": _spectrum_from_gram(torch.empty((0, 0), dtype=torch.float64), 0)}

    base_gram = pairwise_gram(columns, support, block_size=block_size)
    base_norms = base_gram.diagonal().clamp_min(0).sqrt()
    nonzero = base_norms > 0
    indices = nonzero.nonzero().flatten().tolist()
    if not indices:
        target_sq = float(pairwise_gram((target,), support, block_size=block_size)[0, 0])
        target_norm = math.sqrt(max(target_sq, 0.0))
        return {"novelty": 1.0 if target_norm > 0 else 0.0, "target_norm": target_norm,
                "residual_norm": target_norm, "status": "zero_basis" if target_norm else "zero_target_undefined_ratio",
                "spectrum": _spectrum_from_gram(torch.empty((0, 0), dtype=torch.float64), 0),
                "normalization_scales": [0.0] * len(columns), "used_pseudoinverse": False,
                "reconstruction_relative_error": 0.0}

    active_columns = [columns[i] for i in indices]
    scales = base_norms[indices].reciprocal()
    gram = base_gram[indices][:, indices] * scales[:, None] * scales[None, :]
    ambient = (int(support.sum().item()) if support is not None else target.size(0)) * target.size(1)
    spectrum = _spectrum_from_gram(gram, ambient)
    if spectrum["status"].startswith("invalid"):
        return {"novelty": None, "target_norm": None, "residual_norm": None,
                "status": spectrum["status"], "spectrum": spectrum,
                "normalization_scales": [float(1.0 / x) if x > 0 else 0.0 for x in base_norms]}
    target_sq = float(pairwise_gram((target,), support, block_size=block_size)[0, 0])
    target_norm = math.sqrt(max(target_sq, 0.0))
    norm_scale_all = [float(1.0 / x) if x > 0 else 0.0 for x in base_norms]
    if target_norm == 0.0:
        return {"novelty": 0.0, "target_norm": 0.0, "residual_norm": 0.0,
                "status": "zero_target_undefined_ratio", "spectrum": spectrum,
                "normalization_scales": norm_scale_all, "used_pseudoinverse": spectrum["rank"] < len(columns),
                "reconstruction_relative_error": 0.0}

    q = pairwise_gram([*active_columns, target], support, block_size=block_size)[:-1, -1]
    q = q * scales / target_norm
    eigvals, eigvecs = torch.linalg.eigh((gram + gram.T) * 0.5)
    eig_tol = max(ambient, gram.size(0)) * torch.finfo(torch.float64).eps * max(float(eigvals[-1]), 1.0)
    keep = eigvals > eig_tol
    if bool(keep.any()):
        coeff = eigvecs[:, keep] @ ((eigvecs[:, keep].T @ q) / eigvals[keep])
    else:
        coeff = torch.zeros_like(q)
    residual_sq = torch.zeros((), dtype=torch.float64, device=target.device)
    projected_sq = torch.zeros_like(residual_sq)
    for sl in _block_slices(target.size(0), block_size):
        y = _select(target[sl], support[sl] if support is not None else None).to(torch.float64).reshape(-1) / target_norm
        v = torch.stack([
            _select(col[sl], support[sl] if support is not None else None).to(torch.float64).reshape(-1) * scale
            for col, scale in zip(active_columns, scales)
        ])
        residual_block = y - coeff @ v
        residual_sq += residual_block.square().sum()
        projected_sq += (coeff @ v).square().sum()
    rel_residual = float(residual_sq.clamp_min(0).sqrt())
    reconstruction_error = abs((rel_residual ** 2 + float(projected_sq)) - 1.0)
    return {"novelty": rel_residual, "target_norm": target_norm, "residual_norm": rel_residual * target_norm,
            "status": "ok" if math.isfinite(rel_residual) and reconstruction_error <= 1.0e-6 else "numerical_instability",
            "spectrum": spectrum, "normalization_scales": norm_scale_all,
            "used_pseudoinverse": spectrum["rank"] < len(columns),
            "reconstruction_relative_error": reconstruction_error,
            "projection_coefficients_scaled_basis": [float(x) for x in coeff],
            "gram_eigenvalue_tolerance": eig_tol}


def all_pairwise_cosines(values: Iterable[torch.Tensor], support: torch.Tensor, *, block_size=8192) -> dict:
    columns = list(values)
    gram = pairwise_gram(columns, support, block_size=block_size)
    norms = gram.diagonal().clamp_min(0).sqrt()
    result = {}
    for i, j in itertools.combinations_with_replacement(range(len(columns)), 2):
        if float(norms[i]) == 0 or float(norms[j]) == 0:
            result[f"{i}:{j}"] = {"value": None, "status": "zero_norm_operand"}
        else:
            result[f"{i}:{j}"] = {"value": float(gram[i, j] / (norms[i] * norms[j])), "status": "ok"}
    return {"gram": gram.cpu().tolist(), "normalized_gram": (gram / norms[:, None].clamp_min(1e-300) / norms[None, :].clamp_min(1e-300)).cpu().tolist(),
            "cosines": result, "norms": [float(x) for x in norms]}
