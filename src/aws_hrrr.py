"""
AWS Direct NOAA HRRR Weather Ingestion Engine.
Direct byte-range slice ingestion from AWS Open Data NOAA HRRR S3 bucket (noaa-hrrr-bdp-pds).
Provides high-resolution (3km CONUS) short-range forecasts for US cities.
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
    AWS_HRRR_S3_BASE,
    RESOLUTION_SOURCES,
    MODEL_COORDS,
)

# Global eccodes thread lock (shared across ingestion modules)
try:
    from src.aws_weather import _ECCODES_LOCK
except ImportError:
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
_SESSION.headers.update({"User-Agent": "PolymarketWeatherBot/1.0"})

# In-memory caches
_HRRR_DAILY_MAX_CACHE: dict[tuple[str, str, str], float] = {}
_HRRR_IDX_CACHE: dict[str, tuple[int, int]] = {}
_LATEST_HRRR_CYCLE_CACHE: tuple[str, int] | None = None
_LATEST_HRRR_CYCLE_TIMESTAMP: float = 0.0
_CYCLE_CACHE_TTL = 300.0  # 5 minutes

# US city mapping & coordinate aliases for HRRR
HRRR_US_CITIES: dict[str, dict] = {
    "nyc": {"lat": 40.7772, "lon": -73.8726, "tz": "America/New_York", "unit": "F"},
    "miami": {"lat": 25.7932, "lon": -80.2906, "tz": "America/New_York", "unit": "F"},
    "chicago": {"lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago", "unit": "F"},
    "los angeles": {"lat": 33.9425, "lon": -118.4081, "tz": "America/Los_Angeles", "unit": "F"},
    "san francisco": {"lat": 37.6213, "lon": -122.3790, "tz": "America/Los_Angeles", "unit": "F"},
}

_CITY_ALIASES: dict[str, str] = {
    "new york": "nyc",
    "new_york": "nyc",
    "la": "los angeles",
    "los_angeles": "los angeles",
    "sf": "san francisco",
    "san_francisco": "san francisco",
}


def clear_hrrr_caches() -> None:
    """Clears in-memory caches (useful in testing and reset)."""
    global _LATEST_HRRR_CYCLE_CACHE, _LATEST_HRRR_CYCLE_TIMESTAMP
    _HRRR_DAILY_MAX_CACHE.clear()
    _HRRR_IDX_CACHE.clear()
    _LATEST_HRRR_CYCLE_CACHE = None
    _LATEST_HRRR_CYCLE_TIMESTAMP = 0.0


def normalize_us_city(city: str) -> str:
    """Normalizes city string and resolves aliases."""
    c = city.strip().lower().replace("_", " ")
    return _CITY_ALIASES.get(c, c)


def is_us_city(city: str) -> bool:
    """Returns True if the city is supported by HRRR CONUS."""
    norm = normalize_us_city(city)
    return norm in HRRR_US_CITIES or norm in RESOLUTION_SOURCES and RESOLUTION_SOURCES[norm].get("network", "").endswith("_ASOS") and not RESOLUTION_SOURCES[norm].get("network", "").startswith(("GB", "KR", "CN", "FR", "JP"))


def get_latest_hrrr_cycle(
    s3_base: str = AWS_HRRR_S3_BASE,
    timeout: int = 5,
) -> tuple[str, int]:
    """
    Finds the latest available NOAA HRRR cycle (date YYYYMMDD and cycle hour 0..23).
    HRRR runs every hour. Data is usually available within ~1-2 hours of cycle time.
    """
    global _LATEST_HRRR_CYCLE_CACHE, _LATEST_HRRR_CYCLE_TIMESTAMP
    now_ts = time.time()
    if _LATEST_HRRR_CYCLE_CACHE and (now_ts - _LATEST_HRRR_CYCLE_TIMESTAMP < _CYCLE_CACHE_TTL):
        return _LATEST_HRRR_CYCLE_CACHE

    now_utc = dt.datetime.now(dt.timezone.utc)
    # Check backwards from current UTC hour up to 24 hours back
    for hours_back in range(1, 25):
        check_dt = now_utc - dt.timedelta(hours=hours_back)
        date_str = check_dt.strftime("%Y%m%d")
        cycle = check_dt.hour
        test_url = f"{s3_base}/hrrr.{date_str}/conus/hrrr.t{cycle:02d}z.wrfsfcf00.grib2.idx"
        try:
            resp = _SESSION.head(test_url, timeout=timeout)
            if resp.status_code == 200:
                result = (date_str, cycle)
                _LATEST_HRRR_CYCLE_CACHE = result
                _LATEST_HRRR_CYCLE_TIMESTAMP = now_ts
                return result
        except Exception:
            continue

    raise RuntimeError("No available NOAA HRRR cycle found on S3")


def extract_hrrr_tmp2m_byte_range(idx_text: str) -> tuple[int, int] | None:
    """
    Parses the .idx text to locate ':TMP:2 m above ground:' and calculates byte range (start, end).
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


def fetch_hrrr_hourly_temp(
    date_str: str,
    cycle: int,
    forecast_hour: int,
    lat: float,
    lon: float,
    unit: str = "fahrenheit",
    timeout: int = 15,
    s3_base: str = AWS_HRRR_S3_BASE,
) -> float:
    """
    Downloads the 2m temperature byte-slice from HRRR wrfsfcf GRIB2 file on S3.
    Uses ecCodes with _ECCODES_LOCK to extract temperature at (lat, lon).
    Converts Kelvin to requested native unit (Fahrenheit, Celsius, or Kelvin).
    """
    filename_base = f"hrrr.t{cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
    base_url = f"{s3_base}/hrrr.{date_str}/conus"
    idx_url = f"{base_url}/{filename_base}.idx"
    grib_url = f"{base_url}/{filename_base}"

    if idx_url in _HRRR_IDX_CACHE:
        start, end = _HRRR_IDX_CACHE[idx_url]
    else:
        r_idx = _SESSION.get(idx_url, timeout=timeout)
        r_idx.raise_for_status()
        byte_range = extract_hrrr_tmp2m_byte_range(r_idx.text)
        if not byte_range:
            raise ValueError(f"TMP:2 m above ground not found in HRRR idx: {idx_url}")
        start, end = byte_range
        _HRRR_IDX_CACHE[idx_url] = (start, end)

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


def aws_hrrr_daily_max(
    city: str,
    date: str,
    unit: str = "fahrenheit",
    cycle_override: tuple[str, int] | None = None,
    s3_base: str = AWS_HRRR_S3_BASE,
) -> float:
    """
    Computes daily maximum temperature for a US city on target date (YYYY-MM-DD)
    using NOAA HRRR hourly forecast slices from AWS S3.
    Caches result to avoid redundant fetches.
    """
    city_norm = normalize_us_city(city)
    if not is_us_city(city_norm):
        raise ValueError(f"City '{city}' not recognized for HRRR US ingestion.")

    cache_key = (city_norm, date, unit.lower())
    if cache_key in _HRRR_DAILY_MAX_CACHE:
        return _HRRR_DAILY_MAX_CACHE[cache_key]

    if city_norm in HRRR_US_CITIES:
        info = HRRR_US_CITIES[city_norm]
        lat, lon = info["lat"], info["lon"]
        tz_name = info["tz"]
    elif city_norm in RESOLUTION_SOURCES:
        info = RESOLUTION_SOURCES[city_norm]
        lat, lon = info["lat"], info["lon"]
        tz_name = info["tz"]
    elif city_norm in MODEL_COORDS:
        lat, lon = MODEL_COORDS[city_norm]
        tz_name = "America/New_York"
    else:
        raise ValueError(f"City '{city}' not recognized for HRRR US ingestion.")

    tz = ZoneInfo(tz_name)
    cycle_date, cycle_hour = cycle_override or get_latest_hrrr_cycle(s3_base=s3_base)
    cycle_dt = dt.datetime.strptime(f"{cycle_date}{cycle_hour:02d}", "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    target_date = dt.date.fromisoformat(date)

    # Synoptic cycles (00, 06, 12, 18) forecast up to 48 hours; other hourly cycles forecast to 18 hours
    max_fhr = 48 if cycle_hour in (0, 6, 12, 18) else 18
    forecast_hours: list[int] = []
    for fhr in range(0, max_fhr + 1):
        valid_dt = cycle_dt + dt.timedelta(hours=fhr)
        local_dt = valid_dt.astimezone(tz)
        if local_dt.date() == target_date:
            forecast_hours.append(fhr)
        elif local_dt.date() > target_date:
            break

    if not forecast_hours:
        raise ValueError(f"No HRRR forecast hours found for {city} on {date} from cycle {cycle_date}_{cycle_hour:02d}")

    def _fetch_hour(fhr: int) -> float:
        return fetch_hrrr_hourly_temp(
            date_str=cycle_date,
            cycle=cycle_hour,
            forecast_hour=fhr,
            lat=lat,
            lon=lon,
            unit=unit,
            s3_base=s3_base,
        )

    max_workers = min(12, len(forecast_hours))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        hourly_temps = list(executor.map(_fetch_hour, forecast_hours))

    daily_max = max(hourly_temps)
    _HRRR_DAILY_MAX_CACHE[cache_key] = daily_max
    return daily_max


def aws_hrrr_daily_max_members(
    city: str,
    date: str,
    unit: str = "fahrenheit",
    cycle_override: tuple[str, int] | None = None,
) -> list[float]:
    """Convenience helper returning daily max as single-member list for ensemble blending."""
    return [aws_hrrr_daily_max(city=city, date=date, unit=unit, cycle_override=cycle_override)]
