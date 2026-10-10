"""
Tests for probabilistic weather model, Student-t distribution, and tail filters.
"""

import pytest
from src.weather_model import (
    student_t_cdf, gaussian_cdf, lead_scaled_sigma,
    compute_bucket_prob, tail_risk_filter
)


def test_heavy_tail_property():
    """At 3-sigma, Student-t (nu=5) tail probability must be > 2x Gaussian tail probability."""
    mu, sigma = 70.0, 3.0
    val = mu + 3 * sigma  # 79.0

    p_tail_gauss = 1.0 - gaussian_cdf(val, mu, sigma)
    p_tail_student = 1.0 - student_t_cdf(val, mu, sigma, df=5.0)

    assert p_tail_student > 2.0 * p_tail_gauss
    print(f"ok: at 3-sigma, student-t tail ({p_tail_student:.4f}) is > 2x gauss ({p_tail_gauss:.4f})")


def test_lead_scaling():
    base_sigma = 2.0
    s1 = lead_scaled_sigma(base_sigma, lead=1)
    s2 = lead_scaled_sigma(base_sigma, lead=2)
    s3 = lead_scaled_sigma(base_sigma, lead=3)

    assert s1 == base_sigma
    assert s2 > s1
    assert s3 > s2


def test_bucket_width_handling():
    """Single point bucket '32' with bucket_width=1.0 must cover [31.5, 32.5]."""
    p = compute_bucket_prob(
        lo=32.0, hi=32.0, fc=32.0, bias=0.0, scale=1.0,
        precision=0.1, bucket_width=1.0, dist_type="student_t", df=5.0
    )
    # Integrating over 1 unit around center should be substantial (> 0.25)
    assert p > 0.25


def test_tail_risk_filter():
    # Extreme NO at 0.98 where model says NO=0.99 (YES=0.01). Implied YES is 0.02.
    # 0.01 * 1.4 = 0.014 < 0.02 -> allowed
    assert tail_risk_filter(side="NO", p_model=0.99, price=0.94, min_edge=0.03, tail_safety_multiplier=1.4) is True

    # Extreme NO where model says NO=0.985 (YES=0.015). Implied YES is 0.02.
    # 0.015 * 1.4 = 0.021 >= 0.02 -> BLOCKED by tail safety filter
    assert tail_risk_filter(side="NO", p_model=0.985, price=0.98, min_edge=0.001, tail_safety_multiplier=1.4) is False
