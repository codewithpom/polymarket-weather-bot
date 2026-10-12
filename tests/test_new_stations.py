"""
Unit tests for the 5 newly added high-liquidity global settlement stations:
Chicago (KORD), Los Angeles (LAX), San Francisco (SFO), Paris (LFPG), Tokyo (RJTT).
"""

from __future__ import annotations
import datetime as dt
from unittest.mock import patch, MagicMock
import pytest

from src.config import (
    RESOLUTION_SOURCES,
    CITY_CALIB,
    MODEL_COORDS,
    BUCKET_WIDTH,
    GATED_CITIES,
)
from src.station_truth import daily_max, daily_max_range, station_coords, NoStationData


NEW_CITIES = ["chicago", "los angeles", "san francisco", "paris", "tokyo"]

EXPECTED_CONFIG = {
    "chicago": {
        "adapter": "metar", "station": "KORD", "network": "IL_ASOS", "icao": "KORD",
        "lat": 41.9742, "lon": -87.9073, "tz": "America/Chicago",
        "unit": "F", "precision": 1.0, "bucket_width": 2.0,
        "calib": (-0.5, 1.5),
    },
    "los angeles": {
        "adapter": "metar", "station": "LAX", "network": "CA_ASOS", "icao": "KLAX",
        "lat": 33.9425, "lon": -118.4081, "tz": "America/Los_Angeles",
        "unit": "F", "precision": 1.0, "bucket_width": 2.0,
        "calib": (0.2, 1.3),
    },
    "san francisco": {
        "adapter": "metar", "station": "SFO", "network": "CA_ASOS", "icao": "KSFO",
        "lat": 37.6213, "lon": -122.3790, "tz": "America/Los_Angeles",
        "unit": "F", "precision": 1.0, "bucket_width": 2.0,
        "calib": (0.1, 1.4),
    },
    "paris": {
        "adapter": "metar", "station": "LFPG", "network": "FR__ASOS", "icao": "LFPG",
        "lat": 49.0097, "lon": 2.5479, "tz": "Europe/Paris",
        "unit": "C", "precision": 1.0, "bucket_width": 1.0,
        "calib": (0.2, 0.8),
    },
    "tokyo": {
        "adapter": "metar", "station": "RJTT", "network": "JP__ASOS", "icao": "RJTT",
        "lat": 35.5494, "lon": 139.7798, "tz": "Asia/Tokyo",
        "unit": "C", "precision": 1.0, "bucket_width": 1.0,
        "calib": (0.3, 0.9),
    },
}


# ===========================================================================
# 1. Configuration Validation Tests
# ===========================================================================

@pytest.mark.parametrize("city", NEW_CITIES)
def test_resolution_sources_config(city: str):
    """Verify RESOLUTION_SOURCES entries have correct metadata and types."""
    assert city in RESOLUTION_SOURCES
    src = RESOLUTION_SOURCES[city]
    expected = EXPECTED_CONFIG[city]

    assert src["adapter"] == expected["adapter"]
    assert src["station"] == expected["station"]
    assert src["network"] == expected["network"]
    assert src["icao"] == expected["icao"]
    assert src["unit"] == expected["unit"]
    assert src["precision"] == expected["precision"]
    assert src["tz"] == expected["tz"]
    assert abs(src["lat"] - expected["lat"]) < 1e-4
    assert abs(src["lon"] - expected["lon"]) < 1e-4
    assert "source_url" in src


def test_gated_cities_integrity():
    """Verify GATED_CITIES retains only Hong Kong and none of the new cities."""
    assert GATED_CITIES == {"hong kong"}
    for city in NEW_CITIES:
        assert city not in GATED_CITIES


@pytest.mark.parametrize("city", NEW_CITIES)
def test_city_calib_config(city: str):
    """Verify CITY_CALIB bias and smear parameters match empirical specifications."""
    assert city in CITY_CALIB
    bias, smear = CITY_CALIB[city]
    expected_bias, expected_smear = EXPECTED_CONFIG[city]["calib"]
    assert abs(bias - expected_bias) < 1e-4
    assert abs(smear - expected_smear) < 1e-4


@pytest.mark.parametrize("city", NEW_CITIES)
def test_model_coords_config(city: str):
    """Verify MODEL_COORDS matches terrestrial station coordinates."""
    assert city in MODEL_COORDS
    lat, lon = MODEL_COORDS[city]
    expected_lat = EXPECTED_CONFIG[city]["lat"]
    expected_lon = EXPECTED_CONFIG[city]["lon"]
    assert abs(lat - expected_lat) < 1e-4
    assert abs(lon - expected_lon) < 1e-4

    # station_coords helper must return identical coords
    s_lat, s_lon = station_coords(city)
    assert abs(s_lat - lat) < 1e-4
    assert abs(s_lon - lon) < 1e-4


@pytest.mark.parametrize("city", NEW_CITIES)
def test_bucket_width_config(city: str):
    """Verify contract bucket widths: 2.0 for Fahrenheit, 1.0 for Celsius."""
    assert city in BUCKET_WIDTH
    expected_bw = EXPECTED_CONFIG[city]["bucket_width"]
    assert BUCKET_WIDTH[city] == expected_bw


# ===========================================================================
# 2. Station Ground Truth daily_max Unit Tests
# ===========================================================================

@pytest.mark.parametrize("city", NEW_CITIES)
@patch("src.station_truth._mesonet_get")
def test_daily_max_success(mock_mesonet: MagicMock, city: str):
    """Verify daily_max parses METAR CSV, filters by date, and rounds to precision."""
    src = RESOLUTION_SOURCES[city]
    field = "tmpc" if src["unit"] == "C" else "tmpf"
    target_date = dt.date(2026, 10, 8)

    # Mock response containing multiple observations, invalid/null rows, and out-of-date row
    mock_csv = (
        f"station,valid,{field}\n"
        f"{src['station']},2026-10-07 23:55,50.0\n"
        f"{src['station']},2026-10-08 00:50,60.2\n"
        f"{src['station']},2026-10-08 06:00,null\n"
        f"{src['station']},2026-10-08 12:00,M\n"
        f"{src['station']},2026-10-08 15:30,75.4\n"
        f"{src['station']},2026-10-08 18:00,72.1\n"
        f"{src['station']},2026-10-09 00:15,80.0\n"
    )
    mock_mesonet.return_value = mock_csv

    # Test with dt.date object
    result_date = daily_max(city, target_date)
    # 75.4 rounded to step 1.0 is 75.0
    assert result_date == 75.0

    # Verify _mesonet_get called with correct query parameters
    mock_mesonet.assert_called_once()
    called_params = mock_mesonet.call_args[0][0]
    assert called_params["station"] == src["station"]
    assert called_params["network"] == src["network"]
    assert called_params["data"] == field
    assert called_params["tz"] == src["tz"]
    assert called_params["year1"] == 2026
    assert called_params["month1"] == 10
    assert called_params["day1"] == 8

    # Test with string date
    mock_mesonet.reset_mock()
    mock_mesonet.return_value = mock_csv
    result_str = daily_max(city, "2026-10-08")
    assert result_str == 75.0


@pytest.mark.parametrize("city", NEW_CITIES)
@patch("src.station_truth._mesonet_get")
def test_daily_max_no_station_data(mock_mesonet: MagicMock, city: str):
    """Verify daily_max raises NoStationData when no observations are returned for that date."""
    src = RESOLUTION_SOURCES[city]
    field = "tmpc" if src["unit"] == "C" else "tmpf"

    # CSV with only nulls and headers
    mock_csv = (
        f"station,valid,{field}\n"
        f"{src['station']},2026-10-08 00:50,null\n"
        f"{src['station']},2026-10-08 06:00,M\n"
    )
    mock_mesonet.return_value = mock_csv

    with pytest.raises(NoStationData):
        daily_max(city, dt.date(2026, 10, 8))


# ===========================================================================
# 3. Station Ground Truth daily_max_range Unit Tests
# ===========================================================================

@pytest.mark.parametrize("city", NEW_CITIES)
@patch("src.station_truth._mesonet_get")
def test_daily_max_range_success(mock_mesonet: MagicMock, city: str):
    """Verify daily_max_range aggregates daily maximums over date range."""
    src = RESOLUTION_SOURCES[city]
    field = "tmpc" if src["unit"] == "C" else "tmpf"

    start_date = dt.date(2026, 10, 8)
    end_date = dt.date(2026, 10, 10)

    mock_csv = (
        f"station,valid,{field}\n"
        f"{src['station']},2026-10-07 23:00,55.0\n"
        f"{src['station']},2026-10-08 12:00,70.2\n"
        f"{src['station']},2026-10-08 16:00,74.8\n"
        f"{src['station']},2026-10-09 11:00,68.4\n"
        f"{src['station']},2026-10-09 15:00,71.0\n"
        f"{src['station']},2026-10-10 14:00,76.1\n"
        f"{src['station']},2026-10-11 02:00,82.0\n"
    )
    mock_mesonet.return_value = mock_csv

    range_result = daily_max_range(city, start_date, end_date)

    assert range_result == {
        "2026-10-08": 75.0,  # 74.8 rounded to 1.0
        "2026-10-09": 71.0,  # 71.0
        "2026-10-10": 76.0,  # 76.1 rounded to 1.0
    }

    mock_mesonet.assert_called_once()
    called_params = mock_mesonet.call_args[0][0]
    assert called_params["station"] == src["station"]
    assert called_params["network"] == src["network"]
    assert called_params["data"] == field
    assert called_params["year1"] == 2026
    assert called_params["month1"] == 10
    assert called_params["day1"] == 8
    assert called_params["year2"] == 2026
    assert called_params["month2"] == 10
    assert called_params["day2"] == 11


@pytest.mark.parametrize("city", NEW_CITIES)
@patch("src.station_truth._mesonet_get")
def test_daily_max_range_empty(mock_mesonet: MagicMock, city: str):
    """Verify daily_max_range returns empty dict when no observations match."""
    src = RESOLUTION_SOURCES[city]
    field = "tmpc" if src["unit"] == "C" else "tmpf"
    mock_mesonet.return_value = f"station,valid,{field}\n"

    range_result = daily_max_range(city, dt.date(2026, 10, 8), dt.date(2026, 10, 9))
    assert range_result == {}


# ===========================================================================
# 4. Scanner Market Title Parsing Matching
# ===========================================================================

@pytest.mark.parametrize("city,title", [
    ("chicago", "Highest temperature in Chicago on October 15?"),
    ("los angeles", "Highest temperature in Los Angeles on October 15?"),
    ("san francisco", "Highest temperature in San Francisco on October 15?"),
    ("paris", "Highest temperature in Paris on October 15?"),
    ("tokyo", "Highest temperature in Tokyo on October 15?"),
])
def test_scanner_city_resolution(city: str, title: str):
    """Verify title substring matching detects the exact city name."""
    matched = next((c for c in RESOLUTION_SOURCES if c in title.lower()), None)
    assert matched == city
