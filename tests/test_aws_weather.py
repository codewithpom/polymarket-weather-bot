"""
Tests for AWS direct NOAA GEFS weather ingestion engine and scanner fallback integration.
"""

from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from collections import namedtuple

import src.aws_weather as aws_weather
from src.aws_weather import (
    extract_tmp2m_byte_range,
    fetch_member_hourly_temp,
    aws_gefs_daily_max,
    get_latest_gefs_cycle,
    clear_caches,
)
import src.scanner as scanner

# Sample .idx text representing NOAA GEFS wgrib2 index format
SAMPLE_IDX = """
1:0:d=2026101100:HGT:10 mb:3 hour fcst:ENS=+1
2:210315:d=2026101100:TMP:10 mb:3 hour fcst:ENS=+1
62:11150000:d=2026101100:HGT:2 m above ground:3 hour fcst:ENS=+1
63:11312146:d=2026101100:TMP:2 m above ground:3 hour fcst:ENS=+1
64:11553590:d=2026101100:RH:2 m above ground:3 hour fcst:ENS=+1
65:11700000:d=2026101100:UGRD:10 m above ground:3 hour fcst:ENS=+1
"""

SAMPLE_IDX_NO_TMP2M = """
1:0:d=2026101100:HGT:10 mb:3 hour fcst:ENS=+1
2:210315:d=2026101100:TMP:10 mb:3 hour fcst:ENS=+1
3:335914:d=2026101100:RH:10 mb:3 hour fcst:ENS=+1
"""

SAMPLE_IDX_LAST_LINE = """
1:0:d=2026101100:HGT:10 mb:3 hour fcst:ENS=+1
2:500000:d=2026101100:TMP:2 m above ground:3 hour fcst:ENS=+1
"""


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure clean caches and state before each test."""
    clear_caches()
    scanner._ENSEMBLE_CACHE.clear()
    yield
    clear_caches()
    scanner._ENSEMBLE_CACHE.clear()


# ---------------------------------------------------------------------------
# 1. Byte Range Parsing Tests
# ---------------------------------------------------------------------------

def test_extract_tmp2m_byte_range_success():
    byte_range = extract_tmp2m_byte_range(SAMPLE_IDX)
    assert byte_range is not None
    start, end = byte_range
    assert start == 11312146
    assert end == 11553589  # 11553590 - 1
    # Slice size should be exact span
    assert (end - start + 1) == 241444


def test_extract_tmp2m_byte_range_missing():
    assert extract_tmp2m_byte_range(SAMPLE_IDX_NO_TMP2M) is None
    assert extract_tmp2m_byte_range("") is None
    assert extract_tmp2m_byte_range("   \n  \n") is None


def test_extract_tmp2m_byte_range_last_line():
    byte_range = extract_tmp2m_byte_range(SAMPLE_IDX_LAST_LINE)
    assert byte_range is not None
    start, end = byte_range
    assert start == 500000
    assert end == -1


# ---------------------------------------------------------------------------
# 2. Kelvin Conversion & Extraction Tests
# ---------------------------------------------------------------------------

def test_kelvin_conversion():
    # 273.15 K == 0.0 C == 32.0 F
    k_freezing = 273.15
    c_freezing = k_freezing - 273.15
    f_freezing = (k_freezing - 273.15) * 9.0 / 5.0 + 32.0
    assert c_freezing == pytest.approx(0.0)
    assert f_freezing == pytest.approx(32.0)

    # 373.15 K == 100.0 C == 212.0 F
    k_boiling = 373.15
    c_boiling = k_boiling - 273.15
    f_boiling = (k_boiling - 273.15) * 9.0 / 5.0 + 32.0
    assert c_boiling == pytest.approx(100.0)
    assert f_boiling == pytest.approx(212.0)

    # 300.0 K == 26.85 C == 80.33 F
    k_warm = 300.0
    c_warm = k_warm - 273.15
    f_warm = (k_warm - 273.15) * 9.0 / 5.0 + 32.0
    assert c_warm == pytest.approx(26.85)
    assert f_warm == pytest.approx(80.33)


@patch("src.aws_weather._SESSION.get")
@patch("src.aws_weather.eccodes")
def test_fetch_member_hourly_temp_celsius_and_fahrenheit(mock_eccodes, mock_get):
    # Mock idx response
    mock_idx_resp = MagicMock()
    mock_idx_resp.text = SAMPLE_IDX
    mock_idx_resp.status_code = 200

    # Mock grib slice response
    mock_grib_resp = MagicMock()
    mock_grib_resp.content = b"GRIB_DUMMY_BINARY_DATA"
    mock_grib_resp.status_code = 206

    mock_get.side_effect = [mock_idx_resp, mock_grib_resp, mock_idx_resp, mock_grib_resp]

    # Mock eccodes handle & nearest point
    mock_eccodes.codes_new_from_message.return_value = 12345
    NearestPoint = namedtuple("NearestPoint", ["value", "lat", "lon"])
    mock_eccodes.codes_grib_find_nearest.return_value = (NearestPoint(value=300.0, lat=40.7, lon=286.1),)

    # Fetch Celsius
    temp_c = fetch_member_hourly_temp(
        date_str="20261011",
        cycle="00",
        member=1,
        forecast_hour=3,
        lat=40.7772,
        lon=-73.8726,
        unit="celsius",
    )
    assert temp_c == pytest.approx(300.0 - 273.15)
    mock_eccodes.codes_release.assert_called_with(12345)

    # Verify Range header sent
    grib_call_headers = mock_get.call_args_list[1][1]["headers"]
    assert grib_call_headers["Range"] == "bytes=11312146-11553589"

    # Verify longitude normalization: -73.8726 -> (360 - 73.8726) = 286.1274
    mock_eccodes.codes_grib_find_nearest.assert_called_with(
        12345, 40.7772, pytest.approx(286.1274), is_lsm=False, npoints=1
    )

    # Fetch Fahrenheit
    temp_f = fetch_member_hourly_temp(
        date_str="20261011",
        cycle="00",
        member=1,
        forecast_hour=3,
        lat=40.7772,
        lon=-73.8726,
        unit="fahrenheit",
    )
    assert temp_f == pytest.approx((300.0 - 273.15) * 9.0 / 5.0 + 32.0)


# ---------------------------------------------------------------------------
# 3. Latest Cycle Lookup Tests
# ---------------------------------------------------------------------------

@patch("src.aws_weather._SESSION.head")
def test_get_latest_gefs_cycle_success(mock_head):
    def head_side_effect(url, **kwargs):
        resp = MagicMock()
        # Simulate 12z available, 18z unavailable
        if "12/atmos" in url:
            resp.status_code = 200
        else:
            resp.status_code = 404
        return resp

    mock_head.side_effect = head_side_effect
    date_str, cycle = get_latest_gefs_cycle()
    assert cycle == "12"
    assert len(date_str) == 8


@patch("src.aws_weather._SESSION.head")
def test_get_latest_gefs_cycle_all_fail(mock_head):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_head.return_value = mock_resp

    with pytest.raises(RuntimeError, match="No available NOAA GEFS cycle found"):
        get_latest_gefs_cycle()


# ---------------------------------------------------------------------------
# 4. Daily Max Calculation & Caching Tests
# ---------------------------------------------------------------------------

@patch("src.aws_weather.get_latest_gefs_cycle")
@patch("src.aws_weather.fetch_member_hourly_temp")
def test_aws_gefs_daily_max(mock_fetch_hourly, mock_get_cycle):
    mock_get_cycle.return_value = ("20261011", "00")

    # Simulate 8 hourly temperatures per member; return member * 2.0 + (fhr % 5)
    def fake_hourly(date_str, cycle, member, forecast_hour, lat, lon, unit):
        return float(member * 2.0 + (forecast_hour % 5))

    mock_fetch_hourly.side_effect = fake_hourly

    # Compute daily max for 5 members
    results = aws_gefs_daily_max(city="nyc", date="2026-10-11", unit="fahrenheit", num_members=5)
    assert len(results) == 5
    # For member 1: max of (1*2 + (fhr % 5)) -> 2 + 4 = 6.0 (or similar)
    assert results[0] > 0
    # Higher members have higher temps in our fake function
    assert results[4] > results[0]

    # Verify Cache: Second call should NOT call fetch_member_hourly_temp again
    call_count_before = mock_fetch_hourly.call_count
    cached_results = aws_gefs_daily_max(city="nyc", date="2026-10-11", unit="fahrenheit", num_members=5)
    assert cached_results == results
    assert mock_fetch_hourly.call_count == call_count_before


def test_aws_gefs_daily_max_unknown_city():
    with pytest.raises(ValueError, match="not recognized"):
        aws_gefs_daily_max(city="atlantis", date="2026-10-11", unit="fahrenheit")


# ---------------------------------------------------------------------------
# 5. Scanner Fallback to Open-Meteo Integration Tests
# ---------------------------------------------------------------------------

@patch("src.scanner.USE_AWS_DIRECT", 1)
@patch("src.scanner.aws_gefs_daily_max")
def test_ensemble_daily_max_uses_aws_when_enabled(mock_aws_daily_max):
    expected_members = [68.0, 69.5, 71.0, 70.2]
    mock_aws_daily_max.return_value = expected_members

    result = scanner.ensemble_daily_max(
        lat=40.7772,
        lon=-73.8726,
        date="2026-10-12",
        unit="fahrenheit",
        city="nyc",
    )
    assert result == expected_members
    mock_aws_daily_max.assert_called_once_with("nyc", "2026-10-12", "fahrenheit")


@patch("src.scanner.USE_AWS_DIRECT", 1)
@patch("src.scanner.aws_gefs_daily_max")
@patch("src.scanner._SESSION.get")
def test_ensemble_daily_max_fallback_on_aws_exception(mock_get, mock_aws_daily_max, capsys):
    # AWS raises an exception
    mock_aws_daily_max.side_effect = RuntimeError("AWS S3 Connection Timeout")

    # Open-Meteo returns valid mock ensemble response
    mock_open_meteo_resp = MagicMock()
    mock_open_meteo_resp.status_code = 200
    mock_open_meteo_resp.json.return_value = {
        "hourly": {
            "temperature_2m_member01": [60.0, 65.0, 62.0],
            "temperature_2m_member02": [61.0, 66.5, 63.0],
        }
    }
    mock_get.return_value = mock_open_meteo_resp

    result = scanner.ensemble_daily_max(
        lat=40.7772,
        lon=-73.8726,
        date="2026-10-12",
        unit="fahrenheit",
        city="nyc",
    )

    # Should have fallen back to Open-Meteo maxes: [65.0, 66.5]
    assert result == [65.0, 66.5]

    # Verify fallback message was logged
    captured = capsys.readouterr()
    assert "[AWS Direct] Fallback to Open-Meteo: AWS S3 Connection Timeout" in captured.out


@patch("src.scanner.USE_AWS_DIRECT", 1)
@patch("src.scanner.aws_gefs_daily_max")
@patch("src.scanner._SESSION.get")
def test_ensemble_daily_max_fallback_on_empty_aws_data(mock_get, mock_aws_daily_max, capsys):
    # AWS returns empty list
    mock_aws_daily_max.return_value = []

    mock_open_meteo_resp = MagicMock()
    mock_open_meteo_resp.status_code = 200
    mock_open_meteo_resp.json.return_value = {
        "hourly": {
            "temperature_2m_member01": [55.0, 58.0],
        }
    }
    mock_get.return_value = mock_open_meteo_resp

    result = scanner.ensemble_daily_max(
        lat=40.7772,
        lon=-73.8726,
        date="2026-10-12",
        unit="fahrenheit",
        city="nyc",
    )

    assert result == [58.0]
    captured = capsys.readouterr()
    assert "[AWS Direct] Fallback to Open-Meteo: no members returned" in captured.out


@patch("src.scanner.USE_AWS_DIRECT", 0)
@patch("src.scanner.aws_gefs_daily_max")
@patch("src.scanner._SESSION.get")
def test_ensemble_daily_max_aws_disabled(mock_get, mock_aws_daily_max):
    mock_open_meteo_resp = MagicMock()
    mock_open_meteo_resp.status_code = 200
    mock_open_meteo_resp.json.return_value = {
        "hourly": {
            "temperature_2m_member01": [50.0, 52.0],
        }
    }
    mock_get.return_value = mock_open_meteo_resp

    result = scanner.ensemble_daily_max(
        lat=40.7772,
        lon=-73.8726,
        date="2026-10-12",
        unit="fahrenheit",
        city="nyc",
    )

    assert result == [52.0]
    mock_aws_daily_max.assert_not_called()
