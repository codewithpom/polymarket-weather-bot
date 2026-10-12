"""
AWS Direct ECMWF Open Data Weather Ingestion Engine.
Direct byte-range slice ingestion from AWS Open Data ECMWF IFS 0.25° S3 bucket (ecmwf-forecasts).
Provides high-resolution global medium-range forecasts for global settlement hubs.
"""

from __future__ import annotations
import concurrent.futures
import datetime as dt
import json
import re
import threading
import time
from zoneinfo import ZoneInfo
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import eccodes

from src.config import (
    AWS_ECMWF_S3_BASE,
    RESOLUTION_SOURCES,
    MODEL_COORDS,
)

# Global eccodes thread lock (shared across ingestion modules)
try:
    from src.aws_weather import _ECCODES_LOCK
except ImportError:
    _ECCODES_LOCK = threading.Lock()

# Reusable HTTP session with connection pooling and retries
_SESSION = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retries, pool_connections=20, pool_maxsize=30)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)
# Custom user-agent prevents AWS S3 anonymous 503 rate limit
_SESSION.headers.update({"User-Agent": "PolymarketWeatherBot/1.0"})

# In-memory caches
_ECMWF_DAILY_MAX_CACHE: dict[tuple[str, str, str], float] = {}
_ECMWF_IDX_CACHE: dict[str, tuple[int, int]] = {}
_LATEST_ECMWF_CYCLE_CACHE: tuple[str, str] | None = None
_LATEST_ECMWF_CYCLE_TIMESTAMP: float = 0.0
_CYCLE_CACHE_TTL = 300.0  # 5 minutes


def clear_ecmwf_caches() -> None:
    """Clears in-memory caches (useful in testing and reset)."""
    global _LATEST_ECMWF_CYCLE_CACHE, _LATEST_ECMWF_CYCLE_TIMESTAMP
    _ECMWF_DAILY_MAX_CACHE.clear()
    _ECMWF_IDX_CACHE.clear()
    _LATEST_ECMWF_CYCLE_CACHE = None
    _LATEST_ECMWF_CYCLE_TIMESTAMP = 0.0


def get_latest_ecmwf_cycle(
    s3_base: str = AWS_ECMWF_S3_BASE,
    timeout: int = 5,
) -> tuple[str, str]:
    """
    Finds latest available ECMWF cycle (date YYYYMMDD and cycle '00', '06', '12', '18').
    """
    global _LATEST_ECMWF_CYCLE_CACHE, _LATEST_ECMWF_CYCLE_TIMESTAMP
    now_ts = time.time()
    if _LATEST_ECMWF_CYCLE_CACHE and (now_ts - _LATEST_ECMWF_CYCLE_TIMESTAMP < _CYCLE_CACHE_TTL):
        return _LATEST_ECMWF_CYCLE_CACHE

    now_utc = dt.datetime.now(dt.timezone.utc)
    for days_back in range(3):
        check_date = (now_utc - dt.timedelta(days=days_back)).date()
        date_str = check_date.strftime("%Y%m%d")
        for cycle in ("18", "12", "06", "00"):
            if days_back == 0 and int(cycle) > now_utc.hour:
                continue
            test_url = f"{s3_base}/{date_str}/{cycle}z/ifs/0p25/oper/{date_str}{cycle}0000-0h-oper-fc.index"
            try:
                resp = _SESSION.head(test_url, timeout=timeout)
                if resp.status_code == 200:
                    result = (date_str, cycle)
                    _LATEST_ECMWF_CYCLE_CACHE = result
                    _LATEST_ECMWF_CYCLE_TIMESTAMP = now_ts
                    return result
            except Exception:
                continue

    raise RuntimeError("No available ECMWF IFS cycle found on S3")


def extract_ecmwf_2t_byte_range(index_text: str) -> tuple[int, int] | None:
    """
    Parses the ECMWF .index file (JSON lines) to find 2m temperature ('2t')
    and extracts byte range (offset, offset + length - 1).
    Also supports wgrib2 style text fallback.
    """
    lines = [line.strip() for line in index_text.strip().splitlines() if line.strip()]
    for line in lines:
        if '"2t"' in line or '"t2m"' in line:
            try:
                data = json.loads(line)
                if data.get("param") in ("2t", "t2m"):
                    offset = int(data.get("_offset", 0))
                    length = int(data.get("_length", 0))
                    if length > 0:
                        return (offset, offset + length - 1)
                    return (offset, -1)
            except Exception:
                m_offset = re.search(r'"_offset":\s*(\d+)', line)
                m_length = re.search(r'"_length":\s*(\d+)', line)
                if m_offset and m_length:
                    offset = int(m_offset.group(1))
                    length = int(m_length.group(1))
                    return (offset, offset + length - 1)

    # wgrib2 style fallback
    for i, line in enumerate(lines):
        if ":2t:" in line or ":TMP:2 m above ground:" in line or ":2T:" in line:
            parts = line.split(":")
            if len(parts) > 1 and parts[1].isdigit():
                start = int(parts[1])
                if i + 1 < len(lines):
                    next_parts = lines[i + 1].split(":")
                    if len(next_parts) > 1 and next_parts[1].isdigit():
                        return (start, int(next_parts[1]) - 1)
                return (start, -1)
    return None


def fetch_ecmwf_hourly_temp(
    date_str: str,
    cycle: str,
    step: int,
    lat: float,
    lon: float,
    unit: str = "celsius",
    timeout: int = 15,
    s3_base: str = AWS_ECMWF_S3_BASE,
) -> float:
    """
    Fetches the 2m temperature byte-slice for ECMWF IFS 0.25° at (lat, lon).
    Uses ecCodes under _ECCODES_LOCK to extract temperature at the nearest grid point.
    Converts Kelvin to requested unit (Celsius, Fahrenheit, or Kelvin).
    """
    prefix = f"{s3_base}/{date_str}/{cycle}z/ifs/0p25/oper"
    filename_base = f"{date_str}{cycle}0000-{step}h-oper-fc"
    idx_url = f"{prefix}/{filename_base}.index"
    grib_url = f"{prefix}/{filename_base}.grib2"

    if idx_url in _ECMWF_IDX_CACHE:
        start, end = _ECMWF_IDX_CACHE[idx_url]
    else:
        r_idx = _SESSION.get(idx_url, timeout=timeout)
        r_idx.raise_for_status()
        byte_range = extract_ecmwf_2t_byte_range(r_idx.text)
        if not byte_range:
            raise ValueError(f"2t (2m temperature) not found in ECMWF index: {idx_url}")
        start, end = byte_range
        _ECMWF_IDX_CACHE[idx_url] = (start, end)

    range_val = f"bytes={start}-{end}" if end >= 0 else f"bytes={start}-"
    headers = {"Range": range_val}
    r_grib = _SESSION.get(grib_url, headers=headers, timeout=timeout)
    r_grib.raise_for_status()

    norm_lon = (360.0 + lon) % 360.0
    with _ECCODES_LOCK:
        gh = eccodes.codes_new_from_message(r_grib.content)
        try:
            nearest = eccodes.codes_grib_find_nearest(gh, lat, norm_lon, is_lsm=False, npoints=1)
            val_k = float(nearest[0].value)
        finally:
            eccodes.codes_release(gh)

    unit_clean = unit.strip().lower()
    if unit_clean in ("fahrenheit", "f"):
        return (val_k - 273.15) * 9.0 / 5.0 + 32.0
    elif unit_clean in ("kelvin", "k"):
        return val_k
    else:
        return val_k - 273.15


def aws_ecmwf_daily_max(
    city: str,
    date: str,
    unit: str = "celsius",
    cycle_override: tuple[str, str] | None = None,
    s3_base: str = AWS_ECMWF_S3_BASE,
) -> float:
    """
    Computes daily maximum temperature for a global hub city on target date (YYYY-MM-DD)
    using ECMWF IFS 0.25° oper slices from AWS S3.
    Caches result to avoid redundant network requests.
    """
    city_lower = city.strip().lower().replace("_", " ")
    cache_key = (city_lower, date, unit.lower())
    if cache_key in _ECMWF_DAILY_MAX_CACHE:
        return _ECMWF_DAILY_MAX_CACHE[cache_key]

    if city_lower in RESOLUTION_SOURCES:
        info = RESOLUTION_SOURCES[city_lower]
        lat, lon = info["lat"], info["lon"]
        tz_name = info["tz"]
    elif city_lower in MODEL_COORDS:
        lat, lon = MODEL_COORDS[city_lower]
        tz_name = RESOLUTION_SOURCES.get(city_lower, {}).get("tz", "UTC")
    else:
        raise ValueError(f"City '{city}' not recognized for ECMWF ingestion.")

    tz = ZoneInfo(tz_name)
    cycle_date, cycle_hour = cycle_override or get_latest_ecmwf_cycle(s3_base=s3_base)
    cycle_dt = dt.datetime.strptime(f"{cycle_date}{cycle_hour}", "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    target_date = dt.date.fromisoformat(date)

    # 3-hourly steps from 0 to 144 covering target date
    forecast_steps: list[int] = []
    for step in range(0, 145, 3):
        valid_dt = cycle_dt + dt.timedelta(hours=step)
        local_dt = valid_dt.astimezone(tz)
        if local_dt.date() == target_date:
            forecast_steps.append(step)
        elif local_dt.date() > target_date:
            break

    if not forecast_steps:
        raise ValueError(f"No ECMWF forecast steps found for {city} on {date} from cycle {cycle_date}_{cycle_hour}z")

    def _fetch_step(s: int) -> float:
        return fetch_ecmwf_hourly_temp(
            date_str=cycle_date,
            cycle=cycle_hour,
            step=s,
            lat=lat,
            lon=lon,
            unit=unit,
            s3_base=s3_base,
        )

    max_workers = min(12, len(forecast_steps))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        hourly_temps = list(executor.map(_fetch_step, forecast_steps))

    daily_max = max(hourly_temps)
    _ECMWF_DAILY_MAX_CACHE[cache_key] = daily_max
    return daily_max


def aws_ecmwf_daily_max_members(
    city: str,
    date: str,
    unit: str = "celsius",
    cycle_override: tuple[str, str] | None = None,
) -> list[float]:
    """Convenience helper returning daily max as single-member list for ensemble blending."""
    return [aws_ecmwf_daily_max(city=city, date=date, unit=unit, cycle_override=cycle_override)]
