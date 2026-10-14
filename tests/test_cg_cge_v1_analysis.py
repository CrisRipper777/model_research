from __future__ import annotations

import pytest

from scripts.analyze_cg_cge_v1 import paired_summary


def test_paired_summary_uses_sample_standard_deviation_and_sign_counts():
    result = paired_summary([0.1, -0.2, 0.3])
    assert result["n"] == 3
    assert result["mean_difference"] == pytest.approx(0.0666666666667)
    assert result["sample_sd"] == pytest.approx(0.251661147842)
    assert result["positive_seeds"] == 2
    assert result["negative_seeds"] == 1
    assert result["zero_seeds"] == 0


def test_paired_summary_rejects_empty_or_nonfinite_values():
    with pytest.raises(ValueError, match="non-empty finite"):
        paired_summary([])
    with pytest.raises(ValueError, match="non-empty finite"):
        paired_summary([float("nan")])
