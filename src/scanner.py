"""
Market Scanner, Ensemble Evaluator, and Batch Grader.
"""

from __future__ import annotations
import datetime as dt
import json
import re
import requests
import sqlite3

from src.config import (
    GAMMA_API, CLOB_API, ENSEMBLE_API, ENSEMBLE_MODELS,
    RESOLUTION_SOURCES, CITY_CALIB, MODEL_COORDS, BUCKET_WIDTH,
    MIN_EDGE_AFTER_FEE
)
from src.fees import taker_fee
from src.station_truth import daily_max_range, daily_max, NoStationData
from src.weather_model import ensemble_bucket_prob, tail_risk_filter
import src.risk as risk
import src.paper as paper
import src.executor as executor

STRATEGY = "weather_extreme"

BUCKET_RE = re.compile(
    r"(?:(-?\d+)\s*(?:°|deg)?\s*[CF]?\s*or (?:below|lower|less))|"
    r"(?:(-?\d+)\s*(?:°|deg)?\s*[CF]?\s*or (?:above|higher|more))|"
    r"(?:(-?\d+)\s*(?:°|deg)?\s*[CF]?\s*[-–to]+\s*(-?\d+))|"
    r"(?:^\s*(-?\d+)\s*(?:°|deg)?\s*[CF]?\s*$)",
    re.I,
)


def parse_bucket(outcome_title: str) -> tuple[float | None, float | None] | None:
    """Parses bucket label to (lo, hi) bounds in native units."""
    m = BUCKET_RE.search(outcome_title)
    if not m:
        return None
    if m.group(1):
        return (None, float(m.group(1)))
    if m.group(2):
        return (float(m.group(2)), None)
    if m.group(3):
        return (float(m.group(3)), float(m.group(4)))
    return (float(m.group(5)), float(m.group(5)))


def _extract_date(title: str) -> str | None:
    m = re.search(r"on ([A-Z][a-z]+ \d{1,2})", title)
    if not m:
        return None
    year = dt.date.today().year
    try:
        d = dt.datetime.strptime(f"{m.group(1)} {year}", "%B %d %Y").date()
        return d.isoformat()
    except ValueError:
        return None


def fetch_temp_events() -> list[dict]:
    """Queries Gamma API for active HIGHEST temperature events."""
    out, offset = [], 0
    while True:
        try:
            r = requests.get(f"{GAMMA_API}/events", params={
                "closed": "false", "limit": 100, "offset": offset,
                "tag_slug": "weather",
            }, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            out += [e for e in batch if re.search(r"highest temperature", e.get("title", ""), re.I)]
            offset += 100
            if len(batch) < 100:
                break
        except Exception as e:
            print(f"Error fetching gamma events: {e}")
            break
    return out


def best_prices(token_id: str) -> tuple[float | None, float | None]:
    try:
        r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=15)
        if r.status_code != 200:
            return None, None
        book = r.json()
        bid = max((float(x["price"]) for x in book.get("bids", [])), default=None)
        ask = min((float(x["price"]) for x in book.get("asks", [])), default=None)
        return bid, ask
    except Exception:
        return None, None


def ensemble_daily_max(lat: float, lon: float, date: str, unit: str) -> list[float]:
    """Retrieves all ensemble members daily max for date."""
    r = requests.get(ENSEMBLE_API, params={
        "latitude": lat, "longitude": lon,
        "models": ENSEMBLE_MODELS,
        "hourly": "temperature_2m",
        "temperature_unit": unit,
        "start_date": date, "end_date": date,
        "timezone": "auto",
    }, timeout=30)
    r.raise_for_status()
    hourly = r.json().get("hourly", {})
    maxes = []
    for key, vals in hourly.items():
        if key.startswith("temperature_2m") and vals and any(v is not None for v in vals):
            maxes.append(max(v for v in vals if v is not None))
    return maxes


def scan_and_execute(con: sqlite3.Connection, is_live: bool = False) -> int:
    """Scans active temperature markets and executes approved trades."""
    if risk.halted():
        print(f"Trading HALTED. Clear {risk.HALT_FILE} before scanning.")
        return 0

    cfg = risk.load_config()
    events = fetch_temp_events()
    print(f"Found {len(events)} active highest temperature events.")
    trades_executed = 0

    for event in events:
        title = event.get("title", "")
        city = next((c for c in RESOLUTION_SOURCES if c in title.lower()), None)
        if not city:
            continue

        bias, smear = CITY_CALIB[city]
        lat, lon = MODEL_COORDS.get(city)
        src = RESOLUTION_SOURCES[city]
        width = BUCKET_WIDTH[city]
        unit = "celsius" if src["unit"] == "C" else "fahrenheit"
        mdate = _extract_date(title)
        if not mdate:
            continue

        try:
            members = ensemble_daily_max(lat, lon, mdate, unit)
        except Exception as e:
            print(f"[skip] {title}: ensemble error ({e})")
            continue

        for mkt in event.get("markets", []):
            bucket = parse_bucket(mkt.get("groupItemTitle") or mkt.get("question", ""))
            if not bucket:
                continue

            model_p = ensemble_bucket_prob(
                members=members, lo=bucket[0], hi=bucket[1],
                bias=bias, smear=smear, precision=src["precision"],
                bucket_width=width, dist_type="student_t", df=5.0
            )

            toks = json.loads(mkt.get("clobTokenIds", "[]"))
            if len(toks) != 2:
                continue
            yes_bid, yes_ask = best_prices(toks[0])

            for side, price, true_p in (
                ("YES", yes_ask, model_p),
                ("NO", (1.0 - yes_bid) if yes_bid else None, 1.0 - model_p),
            ):
                if price is None or not (0.01 < price < 0.99):
                    continue
                fee = taker_fee(1.0, price, "weather")
                edge = true_p - price - fee

                if edge >= MIN_EDGE_AFTER_FEE and tail_risk_filter(side, true_p, price, min_edge=MIN_EDGE_AFTER_FEE):
                    market_key = f"{city} {mdate}"
                    try:
                        stake = risk.approve(con, cfg, market_key, true_p, price, STRATEGY)
                    except risk.RiskBlocked as e:
                        continue

                    print(f"-> TRADE: {title} | {mkt.get('groupItemTitle')} | {side} @ {price:.2f} "
                          f"model={true_p:.3f} edge={edge:+.3f} stake=${stake:.2f}")

                    meta = {
                        "city": city, "date": mdate, "bucket": list(bucket),
                        "unit": src["unit"], "edge": edge
                    }
                    executor.execute_order(
                        con=con,
                        strategy=STRATEGY,
                        market_title=f"{city} {mdate} {mkt.get('groupItemTitle')}",
                        side=side,
                        price=price,
                        model_prob=true_p,
                        stake=stake,
                        market_id=str(mkt.get("id") or ""),
                        token_id=toks[0] if side == "YES" else toks[1],
                        token_index=0 if side == "YES" else 1,
                        meta=meta,
                        is_live=is_live,
                    )
                    trades_executed += 1

    return trades_executed


def bucket_hit(actual: float, lo: float | None, hi: float | None) -> bool:
    return (lo is None or actual >= lo) and (hi is None or actual <= hi)


def grade_settled_trades(con: sqlite3.Connection, verbose: bool = True) -> int:
    """Batch grades settled trades from official station truth."""
    rows = con.execute("""
        SELECT id, side, meta FROM paper_trades
        WHERE strategy=? AND outcome IS NULL
    """, (STRATEGY,)).fetchall()
    if not rows:
        return 0

    dates_by_city: dict[str, set[str]] = {}
    today_iso = dt.date.today().isoformat()
    for r in rows:
        meta = json.loads(r["meta"] or "{}")
        city, date = meta.get("city"), meta.get("date")
        if city and date and date < today_iso:
            dates_by_city.setdefault(city, set()).add(date)

    cached_obs: dict[str, dict[str, float]] = {}
    for city, dates in dates_by_city.items():
        s = min(dt.date.fromisoformat(d) for d in dates)
        e = max(dt.date.fromisoformat(d) for d in dates)
        try:
            cached_obs[city] = daily_max_range(city, s, e)
        except Exception as e_range:
            if verbose:
                print(f"  [truth batch] {city}: {e_range}")

    graded = 0
    for r in rows:
        meta = json.loads(r["meta"] or "{}")
        city, date, bucket = meta.get("city"), meta.get("date"), meta.get("bucket")
        if not (city and date and bucket) or date >= today_iso:
            continue

        actual = cached_obs.get(city, {}).get(date)
        if actual is None:
            try:
                actual = daily_max(city, date)
            except (NoStationData, Exception):
                continue

        yes_won = bucket_hit(actual, bucket[0], bucket[1])
        outcome = int(yes_won if r["side"] == "YES" else not yes_won)
        paper.grade(con, r["id"], outcome)
        graded += 1
        if verbose:
            print(f"  graded #{r['id']} {city} {date} actual={actual} bucket={bucket} {r['side']} -> {'WIN' if outcome else 'LOSS'}")

    if graded:
        paper.check_rolling_halt(con, STRATEGY)
        risk.check_daily_loss(con, risk.load_config(), STRATEGY)
    return graded
