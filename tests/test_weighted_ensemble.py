"""
Unit tests for dynamic multi-model weighted ensemble blending,
lead-time dependent variance smearing, bucket probability normalization,
and heavy-tail tail risk filtering.
"""

import math
import pytest
from src.weather_model import (
    ensemble_bucket_prob,
    weighted_ensemble_bucket_prob,
    compute_effective_smear,
    normalize_bucket_probabilities,
    compute_bucket_prob,
    tail_risk_filter,
)


def test_multi_model_weight_normalization_50_50():
    """Verify multi-model weight blending with 50/50 weights."""
    models = {
        "ecmwf": [68.0, 70.0, 72.0],
        "gfs": [74.0, 76.0, 78.0],
    }
    p_ecmwf = ensemble_bucket_prob(models["ecmwf"], lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0)
    p_gfs = ensemble_bucket_prob(models["gfs"], lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0)
    expected_blend = 0.5 * p_ecmwf + 0.5 * p_gfs

    # 1. Explicit normalized weights (0.5, 0.5)
    p_blend_explicit = weighted_ensemble_bucket_prob(
        model_forecasts=models,
        weights={"ecmwf": 0.5, "gfs": 0.5},
        lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0, lead_days=0,
    )
    assert math.isclose(p_blend_explicit, expected_blend, rel_tol=1e-7)

    # 2. Unnormalized integer weights (50, 50)
    p_blend_unnorm = weighted_ensemble_bucket_prob(
        model_forecasts=models,
        weights={"ecmwf": 50.0, "gfs": 50.0},
        lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0, lead_days=0,
    )
    assert math.isclose(p_blend_unnorm, expected_blend, rel_tol=1e-7)

    # 3. None / empty weights (defaults to equal weighting 0.5, 0.5)
    p_blend_default = weighted_ensemble_bucket_prob(
        model_forecasts=models,
        weights=None,
        lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0, lead_days=0,
    )
    assert math.isclose(p_blend_default, expected_blend, rel_tol=1e-7)


def test_multi_model_weight_normalization_70_30():
    """Verify multi-model weight blending with 70/30 weights and missing model normalization."""
    models = {
        "ecmwf": [68.0, 70.0, 72.0],
        "gfs": [74.0, 76.0, 78.0],
    }
    p_ecmwf = ensemble_bucket_prob(models["ecmwf"], lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0)
    p_gfs = ensemble_bucket_prob(models["gfs"], lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0)
    expected_70_30 = 0.7 * p_ecmwf + 0.3 * p_gfs

    p_blend = weighted_ensemble_bucket_prob(
        model_forecasts=models,
        weights={"ecmwf": 0.7, "gfs": 0.3},
        lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0, lead_days=0,
    )
    assert math.isclose(p_blend, expected_70_30, rel_tol=1e-7)

    # Normalization when extra model not in model_forecasts is passed in weights dictionary
    weights_with_missing_model = {"ecmwf": 0.7, "gfs": 0.3, "icon_seamless": 1.0}
    p_blend_missing = weighted_ensemble_bucket_prob(
        model_forecasts=models,
        weights=weights_with_missing_model,
        lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0, lead_days=0,
    )
    assert math.isclose(p_blend_missing, expected_70_30, rel_tol=1e-7)


def test_single_model_fallback():
    """Verify seamless fallback when only 1 model or a raw member list is supplied."""
    members = [68.0, 70.0, 72.0]
    expected_p = ensemble_bucket_prob(members, lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0)

    # 1. Single model in dict
    p_dict = weighted_ensemble_bucket_prob(
        model_forecasts={"ecmwf": members},
        weights={"ecmwf": 0.2},  # Should normalize to 1.0 seamlessly
        lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0, lead_days=0,
    )
    assert math.isclose(p_dict, expected_p, rel_tol=1e-7)

    # 2. Raw member list passed directly
    p_list = weighted_ensemble_bucket_prob(
        model_forecasts=members,
        weights=None,
        lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0, lead_days=0,
    )
    assert math.isclose(p_list, expected_p, rel_tol=1e-7)

    # 3. Empty input gracefully returns 0.0
    assert weighted_ensemble_bucket_prob({}, {}, lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0) == 0.0
    assert weighted_ensemble_bucket_prob([], None, lo=69.0, hi=71.0, bias=0.0, smear=1.5, precision=1.0, bucket_width=2.0) == 0.0


def test_lead_time_effective_smear_expansion():
    """Verify effective smear expands with lead time: smear * sqrt(1 + 0.15 * lead_days)."""
    base_smear = 1.5
    s0 = compute_effective_smear(base_smear, lead_days=0)
    s1 = compute_effective_smear(base_smear, lead_days=1)
    s2 = compute_effective_smear(base_smear, lead_days=2)
    s3 = compute_effective_smear(base_smear, lead_days=3)

    assert s0 == base_smear
    assert math.isclose(s1, base_smear * math.sqrt(1.0 + 0.15 * 1), rel_tol=1e-7)
    assert math.isclose(s2, base_smear * math.sqrt(1.0 + 0.15 * 2), rel_tol=1e-7)
    assert s0 < s1 < s2 < s3

    # Negative lead_days is clamped to 0
    assert compute_effective_smear(base_smear, lead_days=-1) == base_smear

    # Verify probability density impact:
    # Near forecast center (70.0), lead_days=0 is sharper -> higher probability than lead_days=2
    members = [70.0]
    p_center_lead0 = weighted_ensemble_bucket_prob(
        members, lo=69.5, hi=70.5, bias=0.0, smear=base_smear, precision=1.0, bucket_width=1.0, lead_days=0
    )
    p_center_lead2 = weighted_ensemble_bucket_prob(
        members, lo=69.5, hi=70.5, bias=0.0, smear=base_smear, precision=1.0, bucket_width=1.0, lead_days=2
    )
    assert p_center_lead0 > p_center_lead2

    # In far tail (78.0), broadened variance gives higher probability at lead_days=2 than lead_days=0
    p_tail_lead0 = weighted_ensemble_bucket_prob(
        members, lo=77.5, hi=78.5, bias=0.0, smear=base_smear, precision=1.0, bucket_width=1.0, lead_days=0
    )
    p_tail_lead2 = weighted_ensemble_bucket_prob(
        members, lo=77.5, hi=78.5, bias=0.0, smear=base_smear, precision=1.0, bucket_width=1.0, lead_days=2
    )
    assert p_tail_lead2 > p_tail_lead0


def test_probability_conservation_across_partitioned_buckets():
    """Verify strict bucket probability normalization across contiguous partitions covering (-inf, +inf)."""
    models = {
        "ecmwf": [62.0, 63.5, 64.0],
        "gfs": [63.0, 65.0, 66.0],
    }
    weights = {"ecmwf": 0.6, "gfs": 0.4}

    # Range buckets partition: (-inf, 60], [61, 62], [63, 64], [65, 66], [67, +inf)
    range_buckets = [(None, 60.0), (61.0, 62.0), (63.0, 64.0), (65.0, 66.0), (67.0, None)]

    for lead in [0, 1, 3]:
        probs = [
            weighted_ensemble_bucket_prob(
                model_forecasts=models,
                weights=weights,
                lo=lo,
                hi=hi,
                bias=0.0,
                smear=1.5,
                precision=1.0,
                bucket_width=2.0,
                lead_days=lead,
            )
            for lo, hi in range_buckets
        ]
        total = sum(probs)
        assert math.isclose(total, 1.0, abs_tol=1e-7), f"Sum {total} != 1.0 at lead {lead}"

    # Point buckets partition: (-inf, 60], [61, 61], [62, 62], [63, 63], [64, 64], [65, +inf)
    point_buckets = [(None, 60.0), (61.0, 61.0), (62.0, 62.0), (63.0, 63.0), (64.0, 64.0), (65.0, None)]
    point_probs = [
        weighted_ensemble_bucket_prob(
            model_forecasts=models,
            weights=weights,
            lo=lo,
            hi=hi,
            bias=0.0,
            smear=1.5,
            precision=1.0,
            bucket_width=1.0,
            lead_days=2,
        )
        for lo, hi in point_buckets
    ]
    assert math.isclose(sum(point_probs), 1.0, abs_tol=1e-7)

    # Test normalize_bucket_probabilities helper function
    raw_list = [0.2, 0.3, 0.4]
    norm_list = normalize_bucket_probabilities(raw_list)
    assert math.isclose(sum(norm_list), 1.0, abs_tol=1e-9)

    raw_dict = {"b1": 2.0, "b2": 3.0, "b3": 5.0}
    norm_dict = normalize_bucket_probabilities(raw_dict)
    assert math.isclose(sum(norm_dict.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(norm_dict["b1"], 0.2, abs_tol=1e-9)


def test_backward_compatibility_ensemble_bucket_prob():
    """Verify existing ensemble_bucket_prob continues to function identical to before."""
    members = [71.0, 72.0, 73.0, 74.0]
    p_orig = ensemble_bucket_prob(
        members=members, lo=72.0, hi=73.0, bias=-0.5, smear=1.2, precision=1.0, bucket_width=2.0
    )
    # Re-evaluating must be strictly repeatable
    p_repeat = ensemble_bucket_prob(
        members=members, lo=72.0, hi=73.0, bias=-0.5, smear=1.2, precision=1.0, bucket_width=2.0
    )
    assert p_orig == p_repeat

    # Calling weighted_ensemble_bucket_prob on members with lead_days=0 produces identical probability
    p_weighted = weighted_ensemble_bucket_prob(
        model_forecasts=members, lo=72.0, hi=73.0, bias=-0.5, smear=1.2, precision=1.0, bucket_width=2.0, lead_days=0
    )
    assert math.isclose(p_orig, p_weighted, abs_tol=1e-9)


def test_heavy_tail_tail_risk_filter():
    """Verify tail_risk_filter guards against long-shot tail buckets with deceptive edge."""
    # 1. Backward compatibility: NO selling filter at high price
    assert tail_risk_filter(side="NO", p_model=0.99, price=0.94, min_edge=0.03, tail_safety_multiplier=1.4) is True
    assert tail_risk_filter(side="NO", p_model=0.985, price=0.98, min_edge=0.001, tail_safety_multiplier=1.4) is False

    # 2. Leg A: Legitimate cheap YES convexity trade with real edge
    # price = 0.08, p_model = 0.15 (1.87x price > 1.4x), edge = 0.15 - 0.08 - 0.0037 = 0.0663 >= 0.04
    assert tail_risk_filter(side="YES", p_model=0.15, price=0.08, min_edge=0.04, tail_safety_multiplier=1.4) is True

    # 3. Deceptive long-shot tail bucket: model probability below min_tail_prob (0.05)
    # price = 0.01, p_model = 0.035 (edge = 0.035 - 0.01 - 0.0005 = 0.0245 >= 0.02)
    # Pure Student-t polynomial tail noise -> BLOCKED
    assert tail_risk_filter(side="YES", p_model=0.035, price=0.01, min_edge=0.02, tail_safety_multiplier=1.4) is False

    # 4. Deceptive long-shot tail bucket: insufficient safety multiplier over ask
    # price = 0.07, p_model = 0.085 -> ratio = 1.21 < 1.4x multiplier -> BLOCKED
    assert tail_risk_filter(side="YES", p_model=0.085, price=0.07, min_edge=0.005, tail_safety_multiplier=1.4) is False

    # 5. Normal mid-range contract: passes cleanly
    assert tail_risk_filter(side="YES", p_model=0.60, price=0.50, min_edge=0.04, tail_safety_multiplier=1.4) is True
