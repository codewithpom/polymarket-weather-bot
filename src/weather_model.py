"""
Fine-tuned probabilistic model with Student-t heavy tail distribution,
lead-time scaling, and asymmetric tail-risk filtering.
"""

from __future__ import annotations
import math
from scipy import stats
from src.fees import taker_fee

DEFAULT_DF = 5.0          # Degrees of freedom for Student-t error distribution
LEAD_SCALE_ALPHA = 0.65   # sigma(L) = sigma_1 * sqrt(1 + alpha * (L - 1))


def student_t_cdf(x: float, mu: float, scale: float, df: float = DEFAULT_DF) -> float:
    """Cumulative distribution function for non-standardized Student-t."""
    if scale <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / scale
    return float(stats.t.cdf(z, df=df))


def gaussian_cdf(x: float, mu: float, sigma: float) -> float:
    """Cumulative distribution function for Gaussian."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def lead_scaled_sigma(base_sigma: float, lead: int = 1, alpha: float = LEAD_SCALE_ALPHA) -> float:
    """Scales forecast error standard deviation with lead time (days)."""
    if lead <= 1:
        return base_sigma
    return base_sigma * math.sqrt(1.0 + alpha * (lead - 1))


def compute_bucket_prob(
    lo: float | None,
    hi: float | None,
    fc: float,
    bias: float,
    scale: float,
    precision: float = 1.0,
    bucket_width: float | None = None,
    dist_type: str = "student_t",
    df: float = DEFAULT_DF,
) -> float:
    """
    Computes P(actual temp in [lo, hi]) given forecast `fc`, `bias`, and `scale`.
    Uses Student-t by default to prevent catastrophic thin-tail underestimation.
    """
    mu = fc + bias
    cdf_fn = (lambda x: student_t_cdf(x, mu, scale, df=df)) if dist_type == "student_t" \
             else (lambda x: gaussian_cdf(x, mu, scale))

    if lo is not None and hi is not None and lo == hi:
        half_width = (bucket_width / 2.0) if bucket_width is not None else (precision / 2.0)
        eff_lo = lo - half_width
        eff_hi = hi + half_width
    else:
        half_step = precision / 2.0
        eff_lo = (lo - half_step) if lo is not None else None
        eff_hi = (hi + half_step) if hi is not None else None

    p_hi = cdf_fn(eff_hi) if eff_hi is not None else 1.0
    p_lo = cdf_fn(eff_lo) if eff_lo is not None else 0.0
    return max(0.0, min(1.0, p_hi - p_lo))


def ensemble_bucket_prob(
    members: list[float],
    lo: float | None,
    hi: float | None,
    bias: float,
    smear: float,
    precision: float = 1.0,
    bucket_width: float | None = None,
    dist_type: str = "student_t",
    df: float = DEFAULT_DF,
) -> float:
    """Computes bucket probability by integrating across all ensemble members."""
    if not members:
        return 0.0
    probs = [
        compute_bucket_prob(
            lo=lo, hi=hi, fc=m, bias=bias, scale=smear,
            precision=precision, bucket_width=bucket_width,
            dist_type=dist_type, df=df,
        )
        for m in members
    ]
    return sum(probs) / len(probs)


def evaluate_edge_and_ev(
    side: str,
    p_true: float,
    market_price: float,
    category: str = "weather",
) -> tuple[float, float]:
    """Returns (edge_after_fee, ev_per_share)."""
    fee = taker_fee(1.0, market_price, category)
    edge = (p_true - market_price) - fee
    ev = (1.0 * p_true) - market_price - fee
    return edge, ev


def tail_risk_filter(
    side: str,
    p_model: float,
    price: float,
    min_edge: float = 0.04,
    tail_safety_multiplier: float = 1.4,
) -> bool:
    """
    Asymmetric Tail Risk Filter:
    Guarantees that NO selling on high-price contracts (>0.90) requires a minimum
    safety margin against leptokurtic weather extremes.
    """
    fee = taker_fee(1.0, price, "weather")
    edge = (p_model - price) - fee
    if edge < min_edge:
        return False

    if side == "NO" and price >= 0.90:
        implied_yes = 1.0 - price
        model_yes = 1.0 - p_model
        if model_yes * tail_safety_multiplier >= implied_yes:
            return False

    return True
