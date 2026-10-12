"""
Station ground-truth scraper for official settlement resolution stations.

Sources:
- METAR (Iowa State Mesonet ASOS archive): LGA, MIA, EGLC, RKSI, ZBAA, KORD, LAX, SFO, LFPG, RJTT.
- HKO (Hong Kong Observatory open data extract): HKO.
"""

from __future__ import annotations
import csv
import datetime as dt
import io
import time
import requests
from src.config import RESOLUTION_SOURCES, MESONET_API, HKO_API


class NoStationData(RuntimeError):
    """Station published nothing for that date — never substitute a fallback."""


def _mesonet_get(params: dict, timeout: int = 90) -> str:
    delay = 3.0
    for attempt in range(5):
        r = requests.get(MESONET_API, params=params, timeout=timeout)
        if r.status_code == 429:
            if attempt == 4:
                r.raise_for_status()
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return r.text
    raise RuntimeError("Mesonet unreachable")


def _metar_obs(src: dict, date: dt.date) -> list[float]:
    nxt = date + dt.timedelta(days=1)
    field = "tmpc" if src["unit"] == "C" else "tmpf"
    text = _mesonet_get({
        "station": src["station"], "network": src["network"], "data": field,
        "year1": date.year, "month1": date.month, "day1": date.day,
        "year2": nxt.year, "month2": nxt.month, "day2": nxt.day,
        "tz": src["tz"], "format": "onlycomma", "latlon": "no",
        "missing": "null", "trace": "null", "direct": "no",
        "report_type": ["3", "4"],
    })
    obs = []
    for row in csv.DictReader(io.StringIO(text)):
        stamp, val = row.get("valid", ""), row.get(field)
        if not stamp.startswith(date.isoformat()):
            continue
        if val in (None, "", "null", "M"):
            continue
        try:
            obs.append(float(val))
        except ValueError:
            continue
    return obs


# In-memory month cache for HKO to prevent duplicate monthly downloads
_HKO_CACHE: dict[tuple[int, int], dict[str, float]] = {}


def _get_hko_month(year: int, month: int) -> dict[str, float]:
    key = (year, month)
    if key in _HKO_CACHE:
        return _HKO_CACHE[key]
    r = requests.get(HKO_API, params={
        "dataType": "CLMMAXT", "rformat": "csv", "station": "HKO",
        "year": year, "month": month,
    }, timeout=60)
    r.raise_for_status()
    out = {}
    for line in r.text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if len(parts) >= 5 and parts[4] not in ("C", ""):
            continue
        try:
            date_str = f"{y:04d}-{m:02d}-{d:02d}"
            out[date_str] = float(parts[3])
        except ValueError:
            pass
    _HKO_CACHE[key] = out
    return out


def daily_max(city: str, date: dt.date | str) -> float:
    """Highest temperature recorded at the resolution station in native units."""
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    src = RESOLUTION_SOURCES[city]

    if src["adapter"] == "hko":
        m_dict = _get_hko_month(date.year, date.month)
        d_str = date.isoformat()
        if d_str not in m_dict:
            raise NoStationData(f"hong kong {date}: no HKO data published")
        raw = m_dict[d_str]
    else:
        obs = _metar_obs(src, date)
        if not obs:
            raise NoStationData(f"{city} {date}: no obs from {src['station']}")
        raw = max(obs)

    step = src["precision"]
    return round(round(raw / step) * step, 1)


def daily_max_range(city: str, start: dt.date, end: dt.date) -> dict[str, float]:
    """{date_iso: daily_max} over [start, end] inclusive in native units."""
    src = RESOLUTION_SOURCES[city]
    step = src["precision"]

    if src["adapter"] == "hko":
        out = {}
        cur = dt.date(start.year, start.month, 1)
        while cur <= end:
            m_dict = _get_hko_month(cur.year, cur.month)
            for d_str, v in m_dict.items():
                d_obj = dt.date.fromisoformat(d_str)
                if start <= d_obj <= end:
                    out[d_str] = round(round(v / step) * step, 1)
            cur = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        return out

    nxt = end + dt.timedelta(days=1)
    field = "tmpc" if src["unit"] == "C" else "tmpf"
    text = _mesonet_get({
        "station": src["station"], "network": src["network"], "data": field,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": nxt.year, "month2": nxt.month, "day2": nxt.day,
        "tz": src["tz"], "format": "onlycomma", "latlon": "no",
        "missing": "null", "trace": "null", "direct": "no",
        "report_type": ["3", "4"],
    })

    by_day: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(text)):
        stamp, val = row.get("valid", ""), row.get(field)
        if val in (None, "", "null", "M") or len(stamp) < 10:
            continue
        day = stamp[:10]
        try:
            t = float(val)
        except ValueError:
            continue
        by_day[day] = max(by_day.get(day, t), t)

    return {
        d: round(round(v / step) * step, 1)
        for d, v in by_day.items()
        if start.isoformat() <= d <= end.isoformat()
    }


def station_coords(city: str) -> tuple[float, float]:
    src = RESOLUTION_SOURCES[city]
    return src["lat"], src["lon"]
