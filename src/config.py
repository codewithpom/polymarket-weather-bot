"""
Production Configuration for Polymarket Weather Prediction Strategy.
"""

from __future__ import annotations
import os
import pathlib
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Root paths
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "paper.db"
HALT_FILE = DATA_DIR / "HALT"
CONFIG_FILE = DATA_DIR / "risk_config.json"

# Execution flags
LIVE = int(os.getenv("LIVE", "0"))
FORCE_LIVE = int(os.getenv("FORCE_LIVE", "0"))
MIN_EDGE_AFTER_FEE = float(os.getenv("MIN_EDGE_AFTER_FEE", "0.04"))
USE_AWS_DIRECT = int(os.getenv("USE_AWS_DIRECT", "1"))
AWS_GEFS_S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
AWS_HRRR_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
AWS_ECMWF_S3_BASE = "https://ecmwf-forecasts.s3.amazonaws.com"
AWS_GEFS_MEMBERS = int(os.getenv("AWS_GEFS_MEMBERS", "30"))

# API Endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"
PREVRUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MESONET_API = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
HKO_API = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"

# Multi-model ensemble components: ~100+ global members
ENSEMBLE_MODELS = "icon_seamless,gfs_seamless,ecmwf_ifs025,gem_global"

# Official settlement stations from Polymarket resolution rules
RESOLUTION_SOURCES = {
    "nyc": {
        "adapter": "metar", "station": "LGA", "network": "NY_ASOS", "icao": "KLGA",
        "lat": 40.7772, "lon": -73.8726, "tz": "America/New_York",
        "unit": "F", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA",
    },
    "miami": {
        "adapter": "metar", "station": "MIA", "network": "FL_ASOS", "icao": "KMIA",
        "lat": 25.7932, "lon": -80.2906, "tz": "America/New_York",
        "unit": "F", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/us/fl/miami/KMIA",
    },
    "london": {
        "adapter": "metar", "station": "EGLC", "network": "GB__ASOS", "icao": "EGLC",
        "lat": 51.5053, "lon": 0.0553, "tz": "Europe/London",
        "unit": "C", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/gb/london/EGLC",
    },
    "seoul": {
        "adapter": "metar", "station": "RKSI", "network": "KR__ASOS", "icao": "RKSI",
        "lat": 37.4692, "lon": 126.4505, "tz": "Asia/Seoul",
        "unit": "C", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/kr/incheon/RKSI",
    },
    "beijing": {
        "adapter": "metar", "station": "ZBAA", "network": "CN__ASOS", "icao": "ZBAA",
        "lat": 40.0801, "lon": 116.5846, "tz": "Asia/Shanghai",
        "unit": "C", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/cn/beijing/ZBAA",
    },
    "hong kong": {
        "adapter": "hko", "station": "HKO",
        "lat": 22.3019, "lon": 114.1741, "tz": "Asia/Hong_Kong",
        "unit": "C", "precision": 0.1,
        "source_url": "https://www.weather.gov.hk/en/cis/climat.htm",
    },
    "chicago": {
        "adapter": "metar", "station": "KORD", "network": "IL_ASOS", "icao": "KORD",
        "lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago",
        "unit": "F", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/us/il/chicago/KORD",
    },
    "los angeles": {
        "adapter": "metar", "station": "LAX", "network": "CA_ASOS", "icao": "KLAX",
        "lat": 33.9425, "lon": -118.4081, "tz": "America/Los_Angeles",
        "unit": "F", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/us/ca/los-angeles/KLAX",
    },
    "san francisco": {
        "adapter": "metar", "station": "SFO", "network": "CA_ASOS", "icao": "KSFO",
        "lat": 37.6213, "lon": -122.3790, "tz": "America/Los_Angeles",
        "unit": "F", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/us/ca/san-francisco/KSFO",
    },
    "paris": {
        "adapter": "metar", "station": "LFPG", "network": "FR__ASOS", "icao": "LFPG",
        "lat": 49.0097, "lon": 2.5479, "tz": "Europe/Paris",
        "unit": "C", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/fr/roissy-en-france/LFPG",
    },
    "tokyo": {
        "adapter": "metar", "station": "RJTT", "network": "JP__ASOS", "icao": "RJTT",
        "lat": 35.5494, "lon": 139.7798, "tz": "Asia/Tokyo",
        "unit": "C", "precision": 1.0,
        "source_url": "https://www.wunderground.com/history/daily/jp/ota/RJTT",
    },
}

# Gated / Blocked cities excluded from real-time scanning and trading
# Hong Kong: HKO published data has 30-day monthly lag; cannot settle same-day
GATED_CITIES = {"hong kong"}

# Empirical walk-forward calibration parameters (bias, smear) in native units
CITY_CALIB = {
    "nyc": (-0.5, 1.5),        # LGA  F  bias=-0.45, stdev=2.55 (+14.4% Brier skill)
    "miami": (+2.3, 1.4),      # MIA  F  bias=+2.30, stdev=2.37 (+5.6% skill)
    "london": (+0.0, 0.7),     # EGLC C  bias=+0.04, stdev=1.11 (+21.0% skill, best)
    "beijing": (+1.0, 0.8),    # ZBAA C  bias=+0.99, stdev=1.34 (+10.9% skill)
    "hong kong": (+1.6, 0.7),  # HKO  C  bias=+1.59, stdev=1.22
    "seoul": (+0.4, 0.8),      # Gimpo/Inland cell bias=+0.38, stdev=1.31 (+19.4% skill)
    "chicago": (-0.5, 1.5),
    "los angeles": (+0.2, 1.3),
    "san francisco": (+0.1, 1.4),
    "paris": (+0.2, 0.8),
    "tokyo": (+0.3, 0.9),
}

# Terrestrial model query coordinates (inland Gimpo for Seoul eliminates water grid bias)
MODEL_COORDS = {
    "nyc": (40.7772, -73.8726),
    "miami": (25.7932, -80.2906),
    "london": (51.5053, 0.0553),
    "beijing": (40.0801, 116.5846),
    "hong kong": (22.3019, 114.1741),
    "seoul": (37.56, 126.79),
    "chicago": (41.9742, -87.9073),
    "los angeles": (33.9425, -118.4081),
    "san francisco": (37.6213, -122.3790),
    "paris": (49.0097, 2.5479),
    "tokyo": (35.5494, 139.7798),
}

# Contract bucket width per city: 2.0 for Fahrenheit, 1.0 for Celsius
BUCKET_WIDTH = {
    "nyc": 2.0, "miami": 2.0, "london": 1.0,
    "beijing": 1.0, "hong kong": 1.0, "seoul": 1.0,
    "chicago": 2.0, "los angeles": 2.0, "san francisco": 2.0,
    "paris": 1.0, "tokyo": 1.0,
}

# Institutional risk limits
DEFAULT_RISK_CONFIG = {
    "bankroll": float(os.getenv("MAX_BANKROLL", "1000.0")),
    "kelly_fraction": 0.25,        # Quarter-Kelly
    "per_trade_cap_pct": 0.02,     # Max 2% per single bucket
    "per_market_cap_pct": 0.05,    # Max 5% per city-day correlated set
    "daily_loss_limit_pct": 0.05,  # Max 5% daily drawdown circuit breaker
    "min_stake": 1.0,              # Minimum order size ($1.00)
}
