"""
Institutional Risk Manager.

Enforces:
1. Categorical Multi-Bucket Exposure: Caps correlated legs on a single city-day at 5%.
2. Single-Trade Cap: 2% of bankroll.
3. Daily Drawdown Circuit Breaker: Auto-halts if daily loss reaches 5%.
4. Three-Lock Live Gate: Mechanical fail-closed protection before real execution.
"""

from __future__ import annotations
import json
import os
import sqlite3
import time
from src.config import CONFIG_FILE, HALT_FILE, DEFAULT_RISK_CONFIG, FORCE_LIVE


class RiskBlocked(RuntimeError):
    """Raised when an order violates risk limits."""


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
    else:
        cfg = dict(DEFAULT_RISK_CONFIG)
        save_config(cfg)

    # Invariant validations
    if cfg["per_trade_cap_pct"] > cfg["per_market_cap_pct"]:
        raise ValueError("per_trade_cap cannot exceed per_market_cap")
    if cfg["per_market_cap_pct"] > cfg["daily_loss_limit_pct"]:
        raise ValueError("per_market_cap cannot exceed daily_loss_limit")
    return cfg


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def halted() -> bool:
    return os.path.exists(HALT_FILE)


def trip_halt(reason: str) -> None:
    with open(HALT_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {reason}\n")
    print(f"!! HALT TRIPPED: {reason}")


def position_size(model_prob: float, price: float, cfg: dict) -> float:
    """Computes quarter-Kelly position size with single-trade cap."""
    if price <= 0 or price >= 1.0 or model_prob <= price:
        return 0.0

    b = (1.0 - price) / price
    p = model_prob
    q = 1.0 - p
    f_star = (p * b - q) / b
    if f_star <= 0:
        return 0.0

    f_scaled = f_star * cfg["kelly_fraction"]
    f_capped = min(f_scaled, cfg["per_trade_cap_pct"])
    stake = f_capped * cfg["bankroll"]
    if stake < cfg["min_stake"]:
        return 0.0
    return round(stake, 2)


def active_market_exposure(con: sqlite3.Connection, market_key: str, strategy: str) -> float:
    """Sum of notional dollars deployed on unresolved trades for this city-day."""
    row = con.execute("""
        SELECT COALESCE(SUM(size * price), 0.0) FROM paper_trades
        WHERE strategy = ? AND market LIKE ? AND outcome IS NULL
    """, (strategy, f"{market_key}%")).fetchone()
    return float(row[0])


def check_daily_loss(con: sqlite3.Connection, cfg: dict, strategy: str) -> None:
    """Trips HALT if net realized loss over trailing 24h exceeds daily_loss_limit."""
    cutoff = time.time() - 86400.0
    row = con.execute("""
        SELECT COALESCE(SUM(pnl), 0.0) FROM paper_trades
        WHERE strategy = ? AND graded_ts >= ?
    """, (strategy, cutoff)).fetchone()
    trailing_pnl = float(row[0])
    max_loss = -(cfg["bankroll"] * cfg["daily_loss_limit_pct"])
    if trailing_pnl <= max_loss:
        trip_halt(f"daily loss {trailing_pnl:.2f} <= limit {max_loss:.2f}")


def approve(
    con: sqlite3.Connection,
    cfg: dict,
    market_key: str,
    model_prob: float,
    price: float,
    strategy: str,
) -> float:
    """
    Validates all risk constraints and returns approved dollar stake.
    Raises RiskBlocked if any limit is violated.
    """
    if halted():
        raise RiskBlocked(f"Trading halted ({HALT_FILE})")

    check_daily_loss(con, cfg, strategy)
    if halted():
        raise RiskBlocked("Daily loss limit exceeded")

    base_stake = position_size(model_prob, price, cfg)
    if base_stake <= 0:
        raise RiskBlocked("No positive edge or below min_stake")

    market_cap = cfg["bankroll"] * cfg["per_market_cap_pct"]
    current_exp = active_market_exposure(con, market_key, strategy)
    remaining = market_cap - current_exp
    if remaining <= 0:
        raise RiskBlocked(f"Market cap full for {market_key} (${current_exp:.2f}/${market_cap:.2f})")

    trade_cap = cfg["bankroll"] * cfg["per_trade_cap_pct"]
    approved_stake = min(base_stake, remaining, trade_cap)
    if approved_stake < cfg["min_stake"]:
        raise RiskBlocked(f"Approved stake ${approved_stake:.2f} < min_stake ${cfg['min_stake']}")

    return round(approved_stake, 2)


def require_live(con: sqlite3.Connection, strategy: str) -> None:
    """
    Three-Lock Pre-Order Safety Gate:
    1. LIVE=1 must be set in environment.
    2. FORCE_LIVE=1 or T6 paper-trading gate passed.
    3. HALT file must NOT exist.
    """
    from src.config import LIVE
    import src.paper as paper

    if halted():
        raise PermissionError(f"Refusing live order: HALT file exists ({HALT_FILE})")

    if not LIVE:
        raise PermissionError("LIVE=0: live orders disabled in environment")

    paper.require_live_ok(con, strategy)
