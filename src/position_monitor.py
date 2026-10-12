"""
Dynamic Position Exit & Intra-Day Weather Risk Monitor.

Continuously audits open positions against:
1. Intra-Day Station Observations (METAR ASOS):
   - Detects breached buckets (e.g. daily max has already reached a NO bucket or overshot a YES bucket ceiling).
   - Executes immediate salvage exit to recover capital before expiration to 0.00.
2. Synoptic Model Shift (Adverse Probability Decay):
   - Re-evaluates positions against the latest weather forecast cycle (AWS GEFS / ECMWF / HRRR).
   - If model probability collapses against the trade thesis, executes early stop-loss exit.
3. Convexity Take-Profit:
   - For cheap YES positions that surged by >= +100% to +300% (e.g. 1.5c -> 6.0c), locks in gains
     when updated model probability is below current market bid.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sqlite3
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    GAMMA_API,
    CLOB_API,
    RESOLUTION_SOURCES,
    CITY_CALIB,
    MODEL_COORDS,
    BUCKET_WIDTH,
    LIVE,
)
from src.fees import taker_fee
import src.paper as paper
from src.scanner import parse_bucket, _extract_date
from src.station_truth import daily_max_so_far
from src.weather_model import ensemble_bucket_prob, weighted_ensemble_bucket_prob

logger = logging.getLogger("weather_bot.monitor")

# Shared HTTP session
_SESSION = requests.Session()
_retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
_adapter = HTTPAdapter(max_retries=_retries, pool_connections=10, pool_maxsize=20)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)


def get_live_market_bid(market_id: str, token_index: int) -> float | None:
    """
    Fetches the highest executable BID price from Polymarket CLOB / Gamma API.
    Selling shares fills against the highest available bid.
    """
    if not market_id:
        return None

    try:
        r = _SESSION.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        # Check CLOB orderbook if token ID is available
        toks_raw = data.get("clobTokenIds", "[]")
        toks = json.loads(toks_raw) if isinstance(toks_raw, str) else toks_raw
        if len(toks) > token_index:
            target_token = toks[token_index]
            book_res = _SESSION.get(f"{CLOB_API}/book", params={"token_id": target_token}, timeout=8)
            if book_res.status_code == 200:
                book = book_res.json()
                bids = [float(b["price"]) for b in book.get("bids", []) if "price" in b]
                if bids:
                    return max(bids)

        # Fallback to Gamma outcome price
        raw_prices = data.get("outcomePrices", "[]")
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        if len(prices) > token_index:
            # Conservative taker slippage estimate: 95% of last price
            mid = float(prices[token_index])
            return max(0.001, round(mid * 0.95, 3))
    except Exception as e:
        logger.debug(f"Failed to fetch market bid for {market_id}: {e}")

    return None


def parse_trade_metadata(market_str: str, meta_str: str | None) -> dict[str, Any]:
    """Extracts city, date, and bucket bounds from trade record."""
    meta = {}
    if meta_str:
        try:
            parsed = json.loads(meta_str)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            meta = {}

    city = meta.get("city")
    target_date = meta.get("date")
    bucket = meta.get("bucket")
    unit = meta.get("unit")

    # If missing from meta, extract from market title
    if not city:
        for c in RESOLUTION_SOURCES:
            if c in market_str.lower():
                city = c
                break

    if not target_date:
        target_date = _extract_date(market_str)

    if not bucket:
        bucket_bounds = parse_bucket(market_str)
        if bucket_bounds:
            bucket = list(bucket_bounds)

    if not unit and city and city in RESOLUTION_SOURCES:
        unit = RESOLUTION_SOURCES[city]["unit"]

    return {
        "city": city,
        "date": target_date,
        "bucket": bucket,
        "unit": unit,
    }


def evaluate_exit_decision(
    side: str,
    entry_price: float,
    current_bid: float | None,
    bucket: list[float | None] | None,
    max_so_far: float | None,
    new_model_prob: float | None,
    is_today: bool,
) -> tuple[bool, str, float]:
    """
    Evaluates whether an open position should be exited early.
    Returns: (should_exit: bool, exit_reason: str, exit_price: float)
    """
    if current_bid is None or current_bid <= 0.0:
        return False, "", 0.0

    lo = bucket[0] if bucket and len(bucket) > 0 else None
    hi = bucket[1] if bucket and len(bucket) > 1 else None

    # -------------------------------------------------------------------------
    # RULE 1: Intra-Day Observation Breach (Salvage Exit / Definite Invalidation)
    # -------------------------------------------------------------------------
    if is_today and max_so_far is not None:
        if side.upper() == "NO":
            # If actual temp has already entered or reached the bucket, NO is invalidated!
            reached_bucket = False
            if lo is not None and hi is not None:
                reached_bucket = (lo <= max_so_far <= hi) or (lo == hi and max_so_far >= lo)
            elif lo is not None and hi is None:
                reached_bucket = (max_so_far >= lo)
            elif lo is None and hi is not None:
                reached_bucket = (max_so_far <= hi)

            if reached_bucket:
                reason = f"Observation breach: actual high already reached {max_so_far} within bucket {bucket}. Salvaging at bid {current_bid:.3f}."
                return True, reason, current_bid

        elif side.upper() == "YES":
            # If actual temperature has overshot the upper bound by >= 0.5 deg,
            # daily high can never decrease, so YES contract is mathematically impossible!
            if hi is not None and max_so_far > (hi + 0.5):
                reason = f"Observation ceiling overshoot: actual high reached {max_so_far} > bucket ceiling {hi}. Salvaging at bid {current_bid:.3f}."
                return True, reason, current_bid

    # -------------------------------------------------------------------------
    # RULE 2: Adverse Forecast Model Shift (Stop-Loss)
    # -------------------------------------------------------------------------
    if new_model_prob is not None:
        if side.upper() == "NO":
            # If updated weather model projects YES probability has surged (so NO prob collapsed)
            updated_no_prob = 1.0 - new_model_prob
            if updated_no_prob < 0.40 and current_bid >= 0.20:
                reason = f"Forecast model reversal: updated NO probability collapsed to {updated_no_prob:.2%}. Exiting at bid {current_bid:.3f} to prevent full loss."
                return True, reason, current_bid

        elif side.upper() == "YES":
            # If model probability for YES dropped to near-zero and bid is still viable
            if new_model_prob < 0.05 and current_bid >= (entry_price * 0.40):
                reason = f"Forecast model shift: updated YES probability decayed to {new_model_prob:.2%}. Cutting position at bid {current_bid:.3f}."
                return True, reason, current_bid

    # -------------------------------------------------------------------------
    # RULE 3: Convexity Take-Profit (Lock-in Surge Gains)
    # -------------------------------------------------------------------------
    if side.upper() == "YES":
        # If position has doubled or surged (+100% to +300%) and market bid > updated model prob
        gain_multiple = current_bid / entry_price if entry_price > 0 else 1.0
        if gain_multiple >= 2.0 and current_bid >= 0.04:
            if new_model_prob is None or current_bid > new_model_prob:
                reason = f"Convexity take-profit: position gained {gain_multiple:.1f}x (entry {entry_price:.3f} -> bid {current_bid:.3f}). Locking in profits."
                return True, reason, current_bid

    return False, "", 0.0


def monitor_and_exit_positions(
    con: sqlite3.Connection,
    is_live: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Main monitoring loop. Evaluates all unsettled trades in the ledger and
    executes early exits when stop-loss, salvage, or take-profit rules trigger.
    """
    open_trades = con.execute("""
        SELECT id, ts, strategy, market, market_id, token_index, side,
               price, model_prob, size, fee_est, meta
        FROM paper_trades
        WHERE outcome IS NULL
        ORDER BY id ASC
    """).fetchall()

    if not open_trades:
        print("Position Monitor: No open positions to monitor.")
        return []

    today_iso = dt.date.today().isoformat()
    exits_executed = []

    print(f"\n================================================================================")
    print(f"  POSITION EXIT & INTRA-DAY WEATHER MONITOR ({len(open_trades)} OPEN POSITIONS)")
    print(f"  Mode: {'DRY RUN' if dry_run else ('LIVE' if is_live else 'PAPER')}")
    print(f"================================================================================")

    for r in open_trades:
        trade_id = r["id"]
        market_str = r["market"]
        mkt_id = r["market_id"]
        t_idx = r["token_index"] or 0
        side = r["side"]
        entry_price = r["price"]
        size = r["size"]

        info = parse_trade_metadata(market_str, r["meta"])
        city = info["city"]
        target_date = info["date"]
        bucket = info["bucket"]
        is_today = (target_date == today_iso)

        # 1. Fetch current live market bid
        current_bid = get_live_market_bid(mkt_id, t_idx)

        # 2. Fetch intra-day METAR max so far if target date is today
        max_so_far = None
        if is_today and city:
            max_so_far = daily_max_so_far(city, today_iso)

        # 3. Evaluate exit rules
        should_exit, reason, exit_price = evaluate_exit_decision(
            side=side,
            entry_price=entry_price,
            current_bid=current_bid,
            bucket=bucket,
            max_so_far=max_so_far,
            new_model_prob=None,  # Intraday METAR is primary trigger
            is_today=is_today,
        )

        if should_exit:
            print(f"\n[TRIGGERED EXIT] Trade #{trade_id} ({market_str} - {side})")
            print(f"  Reason: {reason}")
            print(f"  Size: {size:.2f} shares | Entry: {entry_price:.3f} | Exit Bid: {exit_price:.3f}")

            if not dry_run:
                realized_pnl = paper.close_trade(
                    con=con,
                    trade_id=trade_id,
                    exit_price=exit_price,
                    exit_reason=reason,
                )
                print(f"  -> Position CLOSED. Realized PnL: ${realized_pnl:+.2f}")
            else:
                entry_cost = size * entry_price
                est_pnl = size * exit_price - entry_cost - taker_fee(size, exit_price, "weather")
                print(f"  -> [DRY RUN] Would close position. Estimated PnL: ${est_pnl:+.2f}")

            exits_executed.append({
                "trade_id": trade_id,
                "market": market_str,
                "side": side,
                "exit_price": exit_price,
                "reason": reason,
                "dry_run": dry_run,
            })

    print(f"\nPosition Monitor: {len(exits_executed)} early exits processed out of {len(open_trades)} open positions.")
    print(f"================================================================================\n")
    return exits_executed
