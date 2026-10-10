"""
Polymarket dynamic taker fee model.

fee_usdc = shares * base_rate * p * (1 - p)
Peaks at p=0.50, decays towards 0 at extremes. Makers pay ZERO and earn rebates.
"""

from __future__ import annotations

BASE_RATES = {
    "weather": 0.05,     # peak 1.25% at p=0.50
    "crypto": 0.07,      # peak 1.75%
    "politics": 0.04,
    "sports": 0.05,
    "geopolitics": 0.0,
}


def taker_fee(shares: float, price: float, category: str = "weather") -> float:
    """Taker fee in USDC for buying/selling `shares` at `price`."""
    rate = BASE_RATES.get(category, 0.05)
    return shares * rate * price * (1.0 - price)


def taker_fee_pct_of_notional(price: float, category: str = "weather") -> float:
    """Fee as % of notional (shares * price)."""
    if price <= 0:
        return 0.0
    rate = BASE_RATES.get(category, 0.05)
    return rate * (1.0 - price)


def breakeven_edge(price: float, category: str = "weather") -> float:
    """Minimum probability edge needed per share for a taker BUY to be EV positive."""
    rate = BASE_RATES.get(category, 0.05)
    return rate * price * (1.0 - price)
