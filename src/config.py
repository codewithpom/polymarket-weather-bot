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
}

# Empirical walk-forward calibration parameters (bias, smear) in native units
CITY_CALIB = {
    "nyc": (-0.5, 1.5),        # LGA  F  bias=-0.45, stdev=2.55 (+14.4% Brier skill)
    "miami": (+2.3, 1.4),      # MIA  F  bias=+2.30, stdev=2.37 (+5.6% skill)
    "london": (+0.0, 0.7),     # EGLC C  bias=+0.04, stdev=1.11 (+21.0% skill, best)
    "beijing": (+1.0, 0.8),    # ZBAA C  bias=+0.99, stdev=1.34 (+10.9% skill)
    "hong kong": (+1.6, 0.7),  # HKO  C  bias=+1.59, stdev=1.22
    "seoul": (+0.4, 0.8),      # Gimpo/Inland cell bias=+0.38, stdev=1.31 (+19.4% skill)
}

# Terrestrial model query coordinates (inland Gimpo for Seoul eliminates water grid bias)
MODEL_COORDS = {
    "nyc": (40.7772, -73.8726),
    "miami": (25.7932, -80.2906),
    "london": (51.5053, 0.0553),
    "beijing": (40.0801, 116.5846),
    "hong kong": (22.3019, 114.1741),
    "seoul": (37.56, 126.79),
}

# Contract bucket width per city: 2.0 for Fahrenheit, 1.0 for Celsius
BUCKET_WIDTH = {
    "nyc": 2.0, "miami": 2.0, "london": 1.0,
    "beijing": 1.0, "hong kong": 1.0, "seoul": 1.0
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
