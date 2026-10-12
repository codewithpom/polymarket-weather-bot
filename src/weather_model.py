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
LEAD_SMEAR_RATE = 0.15    # Growth rate per lead day for effective smear


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


def compute_effective_smear(smear: float, lead_days: int = 0, rate: float = LEAD_SMEAR_RATE) -> float:
    """
    Computes effective smear variance scaling with lead time:
    effective_smear = smear * sqrt(1.0 + rate * max(0, lead_days))
    Ensures same-day trades (lead_days=0) have the sharpest probability density,
    while 2-day+ forecasts appropriately broaden variance to prevent overconfidence.
    """
    return smear * math.sqrt(1.0 + rate * max(0, lead_days))


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


def weighted_ensemble_bucket_prob(
    model_forecasts: dict[str, list[float]] | list[float],
    weights: dict[str, float] | None = None,
    lo: float | None = None,
    hi: float | None = None,
    bias: float | dict[str, float] = 0.0,
    smear: float = 1.0,
    precision: float = 1.0,
    bucket_width: float | None = None,
    lead_days: int = 0,
    dist_type: str = "student_t",
    df: float = DEFAULT_DF,
) -> float:
    """
    Computes dynamic multi-model weighted ensemble blended bucket probability
    with lead-time dependent variance smearing.

    - Normalizes weights across models present in model_forecasts.
    - If only one model or raw member list is supplied, falls back seamlessly
      to standard single-model distribution.
    - Forecast uncertainty expands with lead time:
      effective_smear = smear * sqrt(1.0 + 0.15 * max(0, lead_days))
    - Across contiguous mutually exclusive buckets covering (-inf, +inf),
      probabilities strictly conserve and sum to 1.0.
    """
    eff_smear = compute_effective_smear(smear, lead_days)

    # Fallback: raw member list passed directly
    if isinstance(model_forecasts, (list, tuple)):
        b = bias if isinstance(bias, (int, float)) else 0.0
        return ensemble_bucket_prob(
            members=list(model_forecasts),
            lo=lo,
            hi=hi,
            bias=float(b),
            smear=eff_smear,
            precision=precision,
            bucket_width=bucket_width,
            dist_type=dist_type,
            df=df,
        )

    if not isinstance(model_forecasts, dict) or not model_forecasts:
        return 0.0

    # Filter for models that have non-empty forecast member lists
    active_models = {k: v for k, v in model_forecasts.items() if v}
    if not active_models:
        return 0.0

    # Seamless single-model fallback
    if len(active_models) == 1:
        model_name = next(iter(active_models.keys()))
        m_bias = bias.get(model_name, 0.0) if isinstance(bias, dict) else float(bias)
        return ensemble_bucket_prob(
            members=active_models[model_name],
            lo=lo,
            hi=hi,
            bias=m_bias,
            smear=eff_smear,
            precision=precision,
            bucket_width=bucket_width,
            dist_type=dist_type,
            df=df,
        )

    # Multi-model weight normalization across models present
    if weights is None or not isinstance(weights, dict) or not weights:
        norm_weights = {k: 1.0 / len(active_models) for k in active_models}
    else:
        raw_weights = {k: max(0.0, float(weights.get(k, 0.0))) for k in active_models}
        total_w = sum(raw_weights.values())
        if total_w > 0.0:
            norm_weights = {k: w / total_w for k, w in raw_weights.items()}
        else:
            norm_weights = {k: 1.0 / len(active_models) for k in active_models}

    total_prob = 0.0
    for model_name, w in norm_weights.items():
        if w <= 0.0:
            continue
        m_bias = bias.get(model_name, 0.0) if isinstance(bias, dict) else float(bias)
        m_prob = ensemble_bucket_prob(
            members=active_models[model_name],
            lo=lo,
            hi=hi,
            bias=m_bias,
            smear=eff_smear,
            precision=precision,
            bucket_width=bucket_width,
            dist_type=dist_type,
            df=df,
        )
        total_prob += w * m_prob

    return max(0.0, min(1.0, total_prob))


def normalize_bucket_probabilities(
    bucket_probs: list[float] | dict[any, float]
) -> list[float] | dict[any, float]:
    """
    Strict Bucket Probability Normalization:
    Ensures that across all contiguous mutually exclusive buckets covering
    (-inf, +inf) or discrete partitions, probabilities sum strictly to 1.0.
    Handles list and dict structures while guarding against non-positive sums.
    """
    if isinstance(bucket_probs, dict):
        total = sum(bucket_probs.values())
        if total <= 0:
            n = len(bucket_probs)
            return {k: (1.0 / n if n > 0 else 0.0) for k in bucket_probs}
        return {k: v / total for k, v in bucket_probs.items()}
    elif isinstance(bucket_probs, list):
        total = sum(bucket_probs)
        if total <= 0:
            n = len(bucket_probs)
            return [(1.0 / n if n > 0 else 0.0) for _ in bucket_probs]
        return [v / total for v in bucket_probs]
    raise TypeError(f"Unsupported type for bucket_probs: {type(bucket_probs)}")


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
    min_tail_prob: float = 0.05,
    min_weighted_edge: float | None = None,
) -> bool:
    """
    Asymmetric Heavy-Tail Risk Filter:
    1. Guarantees minimum edge after taker fees:
       edge = (p_model - price) - fee >= min_edge
    2. Leg B (NO Selling on high-price contracts, price >= 0.90):
       Guarantees a minimum safety margin against leptokurtic weather extremes:
       model_yes * tail_safety_multiplier < implied_yes
    3. Leg A (YES Buying on long-shot tail buckets, price <= 0.15 or p_model <= 0.15):
       - Prevents buying extreme tail buckets driven purely by Student-t distribution artifacts:
         requires p_model >= min_tail_prob (default 0.05).
       - Enforces asymmetric tail safety margin:
         p_model >= price * tail_safety_multiplier.
       - Accounts for weighted probability edge:
         weighted_edge = edge * p_model >= min_weighted_edge
         to prevent committing capital on low-probability deceptive edges.
    """
    fee = taker_fee(1.0, price, "weather")
    edge = (p_model - price) - fee
    if edge < min_edge:
        return False

    eff_min_weighted = min_weighted_edge if min_weighted_edge is not None else (min_edge * min_tail_prob)

    if side == "NO" and price >= 0.90:
        implied_yes = 1.0 - price
        model_yes = 1.0 - p_model
        if model_yes * tail_safety_multiplier >= implied_yes:
            return False

    if side == "YES" and (price <= 0.15 or p_model <= 0.15):
        if p_model < min_tail_prob:
            return False
        if p_model < price * tail_safety_multiplier:
            return False
        weighted_edge = edge * p_model
        if weighted_edge < eff_min_weighted:
            return False

    return True
