import math

import torch

from scripts.mrcge_v13_propagation_audit_math import (
    basis_spectrum,
    global_cosine,
    pairwise_gram,
    span_novelty,
)


def test_span_member_has_near_zero_residual():
    torch.manual_seed(5)
    a, b = torch.randn(19, 7, dtype=torch.float64), torch.randn(19, 7, dtype=torch.float64)
    target = 0.3 * a - 2.1 * b
    result = span_novelty([a, b], target, block_size=4)
    assert result["status"] == "ok"
    assert result["novelty"] < 1.0e-8


def test_independent_orthogonal_component_is_retained():
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    result = span_novelty([a], target, block_size=2)
    assert result["status"] == "ok"
    assert math.isclose(result["novelty"], 1.0, abs_tol=1.0e-12)


def test_repeated_basis_columns_use_pseudoinverse():
    a = torch.randn(11, 5, dtype=torch.float64)
    result = span_novelty([a, a.clone()], 2.0 * a, block_size=3)
    assert result["status"] == "ok"
    assert result["spectrum"]["rank"] == 1
    assert result["used_pseudoinverse"]
    assert result["novelty"] < 1.0e-8


def test_fully_rank_deficient_basis_is_finite():
    a = torch.randn(7, 4, dtype=torch.float64)
    result = span_novelty([a, -3.0 * a, torch.zeros_like(a)], a, block_size=2)
    assert result["status"] == "ok"
    assert result["spectrum"]["rank"] == 1
    assert math.isfinite(result["novelty"])
    assert result["novelty"] < 1.0e-8


def test_zero_target_has_explicit_status_and_no_nan():
    a = torch.randn(9, 3, dtype=torch.float64)
    result = span_novelty([a], torch.zeros_like(a), block_size=4)
    assert result["status"] == "zero_target_undefined_ratio"
    assert result["novelty"] == 0.0
    assert math.isfinite(result["novelty"])


def test_nonzero_basis_scaling_does_not_change_projection():
    torch.manual_seed(11)
    a, b, target = (torch.randn(13, 6, dtype=torch.float64) for _ in range(3))
    first = span_novelty([a, b], target, block_size=5)
    second = span_novelty([17.0 * a, -0.002 * b], target, block_size=5)
    assert math.isclose(first["novelty"], second["novelty"], rel_tol=1.0e-10, abs_tol=1.0e-12)


def test_fp64_gram_projection_matches_direct_lstsq_reference():
    torch.manual_seed(17)
    basis = [torch.randn(17, 8, dtype=torch.float64) for _ in range(4)]
    target = torch.randn(17, 8, dtype=torch.float64)
    result = span_novelty(basis, target, block_size=6)
    design = torch.stack([value.reshape(-1) for value in basis], dim=1)
    y = target.reshape(-1)
    reference_coeff = torch.linalg.lstsq(design, y).solution
    reference = float((y - design @ reference_coeff).norm() / y.norm())
    assert result["status"] == "ok"
    assert math.isclose(result["novelty"], reference, rel_tol=1.0e-9, abs_tol=1.0e-10)


def test_block_sizes_preserve_gram_and_novelty():
    torch.manual_seed(23)
    basis = [torch.randn(21, 9, dtype=torch.float64) for _ in range(3)]
    target = torch.randn(21, 9, dtype=torch.float64)
    gram_small = pairwise_gram([*basis, target], block_size=2)
    gram_large = pairwise_gram([*basis, target], block_size=21)
    result_small = span_novelty(basis, target, block_size=2)
    result_large = span_novelty(basis, target, block_size=21)
    assert torch.allclose(gram_small, gram_large, rtol=1.0e-12, atol=1.0e-12)
    assert math.isclose(result_small["novelty"], result_large["novelty"], rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_spectrum_and_cosine_expose_degeneracy():
    a = torch.randn(8, 3, dtype=torch.float64)
    spectrum = basis_spectrum([a, a, torch.zeros_like(a)], block_size=3)
    cosine = global_cosine(torch.zeros_like(a), a, block_size=2)
    assert spectrum["rank"] == 1
    assert spectrum["zero_columns"] == 1
    assert cosine["value"] is None and cosine["status"] == "zero_norm_operand"
