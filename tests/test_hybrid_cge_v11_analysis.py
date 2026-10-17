from __future__ import annotations

import math

import pytest

from scripts.analyze_hybrid_cge_v11 import fmt, stats, strict_reference


def test_paired_summary_uses_signed_candidate_minus_reference_values():
    result = stats([0.02, -0.01, 0.03])
    assert result["n"] == 3
    assert result["positive_seeds"] == 2
    assert result["negative_seeds"] == 1
    assert result["zero_seeds"] == 0
    assert result["mean"] == pytest.approx(0.0133333333)
    assert fmt([0.01, 0.02, 0.03], 100) == "2.000 ± 1.000"


def test_paired_summary_rejects_nonfinite_and_empty_input():
    with pytest.raises(ValueError):
        stats([])
    with pytest.raises(ValueError):
        stats([0.1, math.nan])


def test_historical_pairing_requires_all_three_seed_fingerprints():
    preflight = {"datasets": {"Movies": {"seeds": {str(seed): {
        "historical_reference_match": {"A2_cross_source": {"eligible": seed != 44}}
    } for seed in (42, 43, 44)}}}}
    assert strict_reference(preflight, "Movies", "View-CGE A2") is False
    preflight["datasets"]["Movies"]["seeds"]["44"]["historical_reference_match"]["A2_cross_source"]["eligible"] = True
    assert strict_reference(preflight, "Movies", "View-CGE A2") is True
