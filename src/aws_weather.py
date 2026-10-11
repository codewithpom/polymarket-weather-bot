"""
AWS Direct NOAA GEFS Weather Ingestion Engine.
Direct byte-range slice ingestion from AWS Open Data NOAA GEFS S3 bucket.
"""

from __future__ import annotations
import concurrent.futures
import datetime as dt
import threading
import time
from zoneinfo import ZoneInfo
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import eccodes

from src.config import (
    AWS_GEFS_S3_BASE,
    AWS_GEFS_MEMBERS,
    MODEL_COORDS,
    RESOLUTION_SOURCES,
)

# Global eccodes thread lock (eccodes C library definitions are not thread-safe across threads)
_ECCODES_LOCK = threading.Lock()

# Reusable HTTP session with pooling and retries
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

# In-memory caches to avoid redundant queries during a scan
_DAILY_MAX_CACHE: dict[tuple[str, str, str, int], list[float]] = {}
_IDX_CACHE: dict[str, tuple[int, int]] = {}
_LATEST_CYCLE_CACHE: tuple[str, str] | None = None
_LATEST_CYCLE_TIMESTAMP: float = 0.0
_CYCLE_CACHE_TTL = 300.0  # 5 minutes


def clear_caches() -> None:
    """Clears in-memory caches (useful in testing and reset)."""
    global _LATEST_CYCLE_CACHE, _LATEST_CYCLE_TIMESTAMP
    _DAILY_MAX_CACHE.clear()
    _IDX_CACHE.clear()
    _LATEST_CYCLE_CACHE = None
    _LATEST_CYCLE_TIMESTAMP = 0.0


def get_latest_gefs_cycle(s3_base: str = AWS_GEFS_S3_BASE, timeout: int = 5) -> tuple[str, str]:
    """
    Finds the latest available NOAA GEFS cycle (date YYYYMMDD and cycle 00/06/12/18).
    """
    global _LATEST_CYCLE_CACHE, _LATEST_CYCLE_TIMESTAMP
    now_ts = time.time()
    if _LATEST_CYCLE_CACHE and (now_ts - _LATEST_CYCLE_TIMESTAMP < _CYCLE_CACHE_TTL):
        return _LATEST_CYCLE_CACHE

    now_utc = dt.datetime.now(dt.timezone.utc)
    # Check today and previous 2 days in reverse chronological order
    for days_back in range(3):
        check_date = (now_utc - dt.timedelta(days=days_back)).date()
        date_str = check_date.strftime("%Y%m%d")
        for cycle in ("18", "12", "06", "00"):
            # Avoid querying future cycles on the current UTC day
            if days_back == 0 and int(cycle) > now_utc.hour:
                continue

            test_url = f"{s3_base}/gefs.{date_str}/{cycle}/atmos/pgrb2ap5/gec00.t{cycle}z.pgrb2a.0p50.f000.idx"
            try:
                resp = _SESSION.head(test_url, timeout=timeout)
                if resp.status_code == 200:
                    result = (date_str, cycle)
                    _LATEST_CYCLE_CACHE = result
                    _LATEST_CYCLE_TIMESTAMP = now_ts
                    return result
            except Exception:
                continue

    raise RuntimeError("No available NOAA GEFS cycle found on S3")


def extract_tmp2m_byte_range(idx_text: str) -> tuple[int, int] | None:
    """
    Parses the .idx text to find ':TMP:2 m above ground:' and computes the byte range (start, end).
    """
    lines = [line.strip() for line in idx_text.strip().splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if ":TMP:2 m above ground:" in line:
            parts = line.split(":")
            if len(parts) > 1 and parts[1].isdigit():
                start = int(parts[1])
                # Find end offset from subsequent line
                if i + 1 < len(lines):
                    next_parts = lines[i + 1].split(":")
                    if len(next_parts) > 1 and next_parts[1].isdigit():
                        end = int(next_parts[1]) - 1
                        return (start, end)
                # If no next line, open-ended range
                return (start, -1)
    return None


def fetch_member_hourly_temp(
    date_str: str,
    cycle: str,
    member: int,
    forecast_hour: int,
    lat: float,
    lon: float,
    unit: str = "celsius",
    timeout: int = 15,
) -> float:
    """
    Uses requests with Range: bytes={start}-{end} to download the ~240KB GRIB2 slice.
    Uses eccodes (codes_new_from_message, codes_grib_find_nearest, codes_release)
    to extract 2m temperature at (lat, (360 + lon) % 360).
    Converts Kelvin to Celsius (K - 273.15) or Fahrenheit ((K - 273.15) * 9/5 + 32).
    """
    mem_str = "gec00" if member == 0 else f"gep{member:02d}"
    idx_filename = f"{mem_str}.t{cycle}z.pgrb2a.0p50.f{forecast_hour:03d}.idx"
    grib_filename = f"{mem_str}.t{cycle}z.pgrb2a.0p50.f{forecast_hour:03d}"
    base_url = f"{AWS_GEFS_S3_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2ap5"
    idx_url = f"{base_url}/{idx_filename}"
    grib_url = f"{base_url}/{grib_filename}"

    # Check idx cache
    if idx_url in _IDX_CACHE:
        start, end = _IDX_CACHE[idx_url]
    else:
        r_idx = _SESSION.get(idx_url, timeout=timeout)
        r_idx.raise_for_status()
        byte_range = extract_tmp2m_byte_range(r_idx.text)
        if not byte_range:
            raise ValueError(f"TMP 2m not found in idx: {idx_url}")
        start, end = byte_range
        _IDX_CACHE[idx_url] = (start, end)

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


def aws_gefs_daily_max(
    city: str,
    date: str,
    unit: str,
    num_members: int = AWS_GEFS_MEMBERS,
) -> list[float]:
    """
    For target date (YYYY-MM-DD), finds the forecast hours covering the local day.
    Fetches the hourly temperatures for each ensemble member across those forecast hours,
    and computes the daily maximum for each member.
    Returns list[float] of daily max values for all members (e.g. 30 members).
    Includes an in-memory cache to avoid redundant network queries during a scan.
    """
    city_lower = city.strip().lower()
    cache_key = (city_lower, date, unit.lower(), num_members)
    if cache_key in _DAILY_MAX_CACHE:
        return _DAILY_MAX_CACHE[cache_key]

    if city_lower not in RESOLUTION_SOURCES or city_lower not in MODEL_COORDS:
        raise ValueError(f"City '{city}' not recognized in config.")

    lat, lon = MODEL_COORDS[city_lower]
    tz_name = RESOLUTION_SOURCES[city_lower]["tz"]
    tz = ZoneInfo(tz_name)

    cycle_date, cycle_hour = get_latest_gefs_cycle()
    cycle_dt = dt.datetime.strptime(f"{cycle_date}{cycle_hour}", "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    target_date = dt.date.fromisoformat(date)

    # Find forecast hours covering the local target date
    forecast_hours: list[int] = []
    for fhr in range(0, 385, 3):
        valid_dt = cycle_dt + dt.timedelta(hours=fhr)
        local_dt = valid_dt.astimezone(tz)
        if local_dt.date() == target_date:
            forecast_hours.append(fhr)
        elif local_dt.date() > target_date:
            break

    if not forecast_hours:
        raise ValueError(f"No forecast hours found for {city} on {date} from cycle {cycle_date}_{cycle_hour}")

    # Determine member list (1..num_members for perturbation members)
    if num_members <= 30:
        members = list(range(1, num_members + 1))
    else:
        members = [0] + list(range(1, num_members))

    def fetch_member_max(mem: int) -> float:
        temps = [
            fetch_member_hourly_temp(
                date_str=cycle_date,
                cycle=cycle_hour,
                member=mem,
                forecast_hour=fhr,
                lat=lat,
                lon=lon,
                unit=unit,
            )
            for fhr in forecast_hours
        ]
        return max(temps)

    max_workers = min(16, len(members))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        member_maxes = list(executor.map(fetch_member_max, members))

    _DAILY_MAX_CACHE[cache_key] = member_maxes
    return member_maxes
