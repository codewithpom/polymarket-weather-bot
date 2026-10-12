"""
Unit tests for AWS Direct NOAA HRRR, ECMWF Open Data, and Unified Weather Routing.
Tests index parsing, byte range calculation, Kelvin conversion, caching, and fallback behavior.
"""

from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from collections import namedtuple

import src.aws_weather as aws_weather
from src.aws_weather import (
    aws_unified_daily_max,
    fallback_open_meteo_daily_max,
    clear_caches,
)
import src.aws_hrrr as aws_hrrr
from src.aws_hrrr import (
    extract_hrrr_tmp2m_byte_range,
    fetch_hrrr_hourly_temp,
    aws_hrrr_daily_max,
    aws_hrrr_daily_max_members,
    get_latest_hrrr_cycle,
    clear_hrrr_caches,
    is_us_city,
    normalize_us_city,
)
import src.aws_ecmwf as aws_ecmwf
from src.aws_ecmwf import (
    extract_ecmwf_2t_byte_range,
    fetch_ecmwf_hourly_temp,
    aws_ecmwf_daily_max,
    aws_ecmwf_daily_max_members,
    get_latest_ecmwf_cycle,
    clear_ecmwf_caches,
)

# Sample HRRR wgrib2 index
SAMPLE_HRRR_IDX = """
69:37484379:d=2026101206:SNOWC:surface:anl:
70:37534620:d=2026101206:SNOD:surface:anl:
71:37607839:d=2026101206:TMP:2 m above ground:anl:
72:38791710:d=2026101206:POT:2 m above ground:anl:
73:39900270:d=2026101206:SPFH:2 m above ground:anl:
"""

SAMPLE_HRRR_IDX_NO_TMP = """
1:0:d=2026101206:HGT:10 mb:anl:
2:210315:d=2026101206:TMP:10 mb:anl:
"""

SAMPLE_HRRR_IDX_LAST_LINE = """
1:0:d=2026101206:HGT:10 mb:anl:
2:50000000:d=2026101206:TMP:2 m above ground:anl:
"""

# Sample ECMWF JSON-lines index
SAMPLE_ECMWF_INDEX = """
{"domain": "g", "date": "20261012", "time": "0000", "step": "0", "param": "z", "_offset": 0, "_length": 896851}
{"domain": "g", "date": "20261012", "time": "0000", "step": "0", "param": "msl", "_offset": 896851, "_length": 524500}
{"domain": "g", "date": "20261012", "time": "0000", "step": "0", "param": "2t", "_offset": 32339444, "_length": 657277}
{"domain": "g", "date": "20261012", "time": "0000", "step": "0", "param": "2d", "_offset": 32996721, "_length": 650000}
"""

SAMPLE_ECMWF_INDEX_NO_2T = """
{"domain": "g", "date": "20261012", "time": "0000", "step": "0", "param": "z", "_offset": 0, "_length": 896851}
{"domain": "g", "date": "20261012", "time": "0000", "step": "0", "param": "msl", "_offset": 896851, "_length": 524500}
"""


@pytest.fixture(autouse=True)
def reset_all_state():
    """Ensure clean caches across all engines before and after each test."""
    clear_caches()
    clear_hrrr_caches()
    clear_ecmwf_caches()
    yield
    clear_caches()
    clear_hrrr_caches()
    clear_ecmwf_caches()


# ===========================================================================
# 1. NOAA HRRR Byte Range Parsing Tests
# ===========================================================================

def test_extract_hrrr_tmp2m_byte_range_success():
    byte_range = extract_hrrr_tmp2m_byte_range(SAMPLE_HRRR_IDX)
    assert byte_range is not None
    start, end = byte_range
    assert start == 37607839
    assert end == 38791709  # 38791710 - 1
    # Verify exact slice span: ~1.18 MB
    assert (end - start + 1) == 1183871


def test_extract_hrrr_tmp2m_byte_range_missing():
    assert extract_hrrr_tmp2m_byte_range(SAMPLE_HRRR_IDX_NO_TMP) is None
    assert extract_hrrr_tmp2m_byte_range("") is None
    assert extract_hrrr_tmp2m_byte_range("   \n \n") is None


def test_extract_hrrr_tmp2m_byte_range_last_line():
    byte_range = extract_hrrr_tmp2m_byte_range(SAMPLE_HRRR_IDX_LAST_LINE)
    assert byte_range is not None
    start, end = byte_range
    assert start == 50000000
    assert end == -1


# ===========================================================================
# 2. NOAA HRRR Temperature Extraction & Unit Conversion Tests
# ===========================================================================

def test_hrrr_kelvin_conversion():
    # 273.15 K == 0.0 C == 32.0 F
    val_k = 273.15
    c_val = val_k - 273.15
    f_val = (val_k - 273.15) * 9.0 / 5.0 + 32.0
    assert c_val == pytest.approx(0.0)
    assert f_val == pytest.approx(32.0)

    # 300.0 K == 26.85 C == 80.33 F
    val_k = 300.0
    c_val = val_k - 273.15
    f_val = (val_k - 273.15) * 9.0 / 5.0 + 32.0
    assert c_val == pytest.approx(26.85)
    assert f_val == pytest.approx(80.33)


@patch("src.aws_hrrr._SESSION.get")
@patch("src.aws_hrrr.eccodes")
def test_fetch_hrrr_hourly_temp(mock_eccodes, mock_get):
    mock_idx_resp = MagicMock()
    mock_idx_resp.text = SAMPLE_HRRR_IDX
    mock_idx_resp.status_code = 200

    mock_grib_resp = MagicMock()
    mock_grib_resp.content = b"HRRR_GRIB_BINARY_DATA"
    mock_grib_resp.status_code = 206

    mock_get.side_effect = [mock_idx_resp, mock_grib_resp, mock_idx_resp, mock_grib_resp, mock_idx_resp, mock_grib_resp]

    # Mock eccodes handle & nearest point
    mock_eccodes.codes_new_from_message.return_value = 88888
    NearestPoint = namedtuple("NearestPoint", ["value", "lat", "lon"])
    mock_eccodes.codes_grib_find_nearest.return_value = (NearestPoint(value=295.15, lat=40.77, lon=286.13),)

    # Fetch Fahrenheit (295.15 K -> 22.0 C -> 71.6 F)
    temp_f = fetch_hrrr_hourly_temp(
        date_str="20261012",
        cycle=6,
        forecast_hour=0,
        lat=40.7772,
        lon=-73.8726,
        unit="fahrenheit",
    )
    assert temp_f == pytest.approx(71.6)
    mock_eccodes.codes_release.assert_called_with(88888)

    # Verify Range header sent
    grib_headers = mock_get.call_args_list[1][1]["headers"]
    assert grib_headers["Range"] == "bytes=37607839-38791709"

    # Fetch Celsius (295.15 K -> 22.0 C)
    temp_c = fetch_hrrr_hourly_temp(
        date_str="20261012",
        cycle=6,
        forecast_hour=0,
        lat=40.7772,
        lon=-73.8726,
        unit="celsius",
    )
    assert temp_c == pytest.approx(22.0)

    # Fetch Kelvin
    temp_k = fetch_hrrr_hourly_temp(
        date_str="20261012",
        cycle=6,
        forecast_hour=0,
        lat=40.7772,
        lon=-73.8726,
        unit="kelvin",
    )
    assert temp_k == pytest.approx(295.15)


# ===========================================================================
# 3. NOAA HRRR Daily Max & US Cities Tests
# ===========================================================================

@patch("src.aws_hrrr.get_latest_hrrr_cycle")
@patch("src.aws_hrrr.fetch_hrrr_hourly_temp")
def test_aws_hrrr_daily_max_all_us_cities(mock_fetch_hourly, mock_get_cycle):
    mock_get_cycle.return_value = ("20261012", 6)

    # Simulate hourly temps: base + (fhr % 6)
    def fake_hourly(date_str, cycle, forecast_hour, lat, lon, unit, s3_base):
        return 70.0 + float(forecast_hour % 6)

    mock_fetch_hourly.side_effect = fake_hourly

    # Test all 5 US cities specified in requirements
    for city in ["nyc", "miami", "chicago", "los angeles", "san francisco"]:
        val = aws_hrrr_daily_max(city=city, date="2026-10-12", unit="fahrenheit")
        assert val >= 70.0
        assert is_us_city(city)

    # Test aliases
    assert normalize_us_city("new york") == "nyc"
    assert normalize_us_city("la") == "los angeles"
    assert normalize_us_city("sf") == "san francisco"


@patch("src.aws_hrrr.get_latest_hrrr_cycle")
@patch("src.aws_hrrr.fetch_hrrr_hourly_temp")
def test_aws_hrrr_daily_max_caching(mock_fetch_hourly, mock_get_cycle):
    mock_get_cycle.return_value = ("20261012", 6)
    mock_fetch_hourly.return_value = 75.0

    val1 = aws_hrrr_daily_max(city="nyc", date="2026-10-12", unit="fahrenheit")
    assert val1 == 75.0
    initial_call_count = mock_fetch_hourly.call_count
    assert initial_call_count > 0

    # Second call should hit in-memory cache
    val2 = aws_hrrr_daily_max(city="nyc", date="2026-10-12", unit="fahrenheit")
    assert val2 == 75.0
    assert mock_fetch_hourly.call_count == initial_call_count


def test_aws_hrrr_daily_max_unsupported_city():
    with pytest.raises(ValueError, match="not recognized"):
        aws_hrrr_daily_max(city="tokyo", date="2026-10-12", unit="fahrenheit")


# ===========================================================================
# 4. ECMWF Open Data Byte Range Parsing Tests
# ===========================================================================

def test_extract_ecmwf_2t_byte_range_success():
    byte_range = extract_ecmwf_2t_byte_range(SAMPLE_ECMWF_INDEX)
    assert byte_range is not None
    start, end = byte_range
    assert start == 32339444
    assert end == 32339444 + 657277 - 1
    assert (end - start + 1) == 657277


def test_extract_ecmwf_2t_byte_range_missing():
    assert extract_ecmwf_2t_byte_range(SAMPLE_ECMWF_INDEX_NO_2T) is None
    assert extract_ecmwf_2t_byte_range("") is None


def test_extract_ecmwf_2t_byte_range_wgrib_fallback():
    wgrib_idx = "10:1000000:d=2026101200:2t:surface:3 hour fcst:\n11:2000000:d=2026101200:msl:surface:3 hour fcst:"
    byte_range = extract_ecmwf_2t_byte_range(wgrib_idx)
    assert byte_range == (1000000, 1999999)


# ===========================================================================
# 5. ECMWF Open Data Hourly Temp & Conversion Tests
# ===========================================================================

@patch("src.aws_ecmwf._SESSION.get")
@patch("src.aws_ecmwf.eccodes")
def test_fetch_ecmwf_hourly_temp(mock_eccodes, mock_get):
    mock_idx_resp = MagicMock()
    mock_idx_resp.text = SAMPLE_ECMWF_INDEX
    mock_idx_resp.status_code = 200

    mock_grib_resp = MagicMock()
    mock_grib_resp.content = b"ECMWF_GRIB_DATA"
    mock_grib_resp.status_code = 206

    mock_get.side_effect = [mock_idx_resp, mock_grib_resp, mock_idx_resp, mock_grib_resp]

    mock_eccodes.codes_new_from_message.return_value = 99999
    NearestPoint = namedtuple("NearestPoint", ["value", "lat", "lon"])
    mock_eccodes.codes_grib_find_nearest.return_value = (NearestPoint(value=288.15, lat=51.5, lon=0.05),)

    # Celsius: 288.15 K -> 15.0 C
    temp_c = fetch_ecmwf_hourly_temp(
        date_str="20261012",
        cycle="00",
        step=3,
        lat=51.5053,
        lon=0.0553,
        unit="celsius",
    )
    assert temp_c == pytest.approx(15.0)

    # Fahrenheit: 15.0 C -> 59.0 F
    temp_f = fetch_ecmwf_hourly_temp(
        date_str="20261012",
        cycle="00",
        step=3,
        lat=51.5053,
        lon=0.0553,
        unit="fahrenheit",
    )
    assert temp_f == pytest.approx(59.0)


# ===========================================================================
# 6. ECMWF Daily Max & Global Hubs Tests
# ===========================================================================

@patch("src.aws_ecmwf.get_latest_ecmwf_cycle")
@patch("src.aws_ecmwf.fetch_ecmwf_hourly_temp")
def test_aws_ecmwf_daily_max_global_hubs(mock_fetch_hourly, mock_get_cycle):
    mock_get_cycle.return_value = ("20261012", "00")
    mock_fetch_hourly.return_value = 18.5

    for hub in ["london", "seoul", "beijing", "hong kong", "paris", "tokyo"]:
        val = aws_ecmwf_daily_max(city=hub, date="2026-10-12", unit="celsius")
        assert val == 18.5


@patch("src.aws_ecmwf.get_latest_ecmwf_cycle")
@patch("src.aws_ecmwf.fetch_ecmwf_hourly_temp")
def test_aws_ecmwf_daily_max_caching(mock_fetch_hourly, mock_get_cycle):
    mock_get_cycle.return_value = ("20261012", "00")
    mock_fetch_hourly.return_value = 21.0

    val1 = aws_ecmwf_daily_max(city="london", date="2026-10-12", unit="celsius")
    assert val1 == 21.0
    calls_before = mock_fetch_hourly.call_count

    val2 = aws_ecmwf_daily_max(city="london", date="2026-10-12", unit="celsius")
    assert val2 == 21.0
    assert mock_fetch_hourly.call_count == calls_before


# ===========================================================================
# 7. Unified Weather Routing Tests
# ===========================================================================

@patch("src.aws_weather.aws_hrrr_daily_max")
@patch("src.aws_weather.aws_ecmwf_daily_max")
@patch("src.aws_weather.aws_gefs_daily_max")
def test_unified_routing_us_short_range_queries_hrrr(mock_gefs, mock_ecmwf, mock_hrrr):
    mock_hrrr.return_value = 72.5
    mock_ecmwf.return_value = 71.0
    mock_gefs.return_value = [70.0, 71.5, 72.0]

    # US city, lead time 5 hours (<24h)
    members = aws_unified_daily_max(
        city="nyc",
        date="2026-10-12",
        unit="fahrenheit",
        lead_hours=5.0,
        prefer_hrrr_short_range=True,
    )

    # HRRR should be queried and present in members
    mock_hrrr.assert_called_once_with(city="nyc", date="2026-10-12", unit="fahrenheit")
    mock_ecmwf.assert_called_once_with(city="nyc", date="2026-10-12", unit="fahrenheit")
    mock_gefs.assert_called_once_with(city="nyc", date="2026-10-12", unit="fahrenheit", num_members=30)
    assert 72.5 in members
    assert len(members) == 5  # 1 HRRR + 1 ECMWF + 3 GEFS


@patch("src.aws_weather.aws_hrrr_daily_max")
@patch("src.aws_weather.aws_ecmwf_daily_max")
@patch("src.aws_weather.aws_gefs_daily_max")
def test_unified_routing_global_hub_queries_gefs_ecmwf(mock_gefs, mock_ecmwf, mock_hrrr):
    mock_ecmwf.return_value = 16.0
    mock_gefs.return_value = [15.5, 16.2]

    # Global city (London), lead time 12 hours
    members = aws_unified_daily_max(
        city="london",
        date="2026-10-12",
        unit="celsius",
        lead_hours=12.0,
    )

    # HRRR should NOT be queried for non-US city
    mock_hrrr.assert_not_called()
    mock_ecmwf.assert_called_once_with(city="london", date="2026-10-12", unit="celsius")
    mock_gefs.assert_called_once_with(city="london", date="2026-10-12", unit="celsius", num_members=30)
    assert members == [16.0, 15.5, 16.2]


@patch("src.aws_weather.aws_hrrr_daily_max")
@patch("src.aws_weather.aws_ecmwf_daily_max")
@patch("src.aws_weather.aws_gefs_daily_max")
@patch("src.aws_weather.fallback_open_meteo_daily_max")
def test_unified_routing_fallback_on_aws_exception(mock_om, mock_gefs, mock_ecmwf, mock_hrrr, capsys):
    # AWS models raise exceptions (e.g. S3 timeout / HTTP 503)
    mock_hrrr.side_effect = RuntimeError("HRRR S3 timeout")
    mock_ecmwf.side_effect = RuntimeError("ECMWF 503 SlowDown")
    mock_gefs.side_effect = RuntimeError("GEFS bucket unavailable")

    mock_om.return_value = [68.0, 69.0, 70.0]

    members = aws_unified_daily_max(
        city="nyc",
        date="2026-10-12",
        unit="fahrenheit",
        lead_hours=6.0,
        fallback_to_open_meteo=True,
    )

    # Should have called fallback and returned Open-Meteo members
    mock_om.assert_called_once()
    assert members == [68.0, 69.0, 70.0]

    captured = capsys.readouterr()
    assert "Fallback to Open-Meteo ensemble" in captured.out


@patch("src.aws_weather.aws_ecmwf_daily_max")
@patch("src.aws_weather.aws_gefs_daily_max")
@patch("src.aws_weather.fallback_open_meteo_daily_max")
def test_unified_routing_fallback_when_data_not_published(mock_om, mock_gefs, mock_ecmwf):
    mock_ecmwf.side_effect = RuntimeError("No available ECMWF IFS cycle found on S3")
    mock_gefs.side_effect = RuntimeError("No available NOAA GEFS cycle found on S3")

    mock_om.return_value = [14.0, 15.0]

    members = aws_unified_daily_max(
        city="london",
        date="2026-10-12",
        unit="celsius",
        fallback_to_open_meteo=True,
    )

    mock_om.assert_called_once()
    assert members == [14.0, 15.0]


@patch("src.aws_weather.aws_ecmwf_daily_max")
@patch("src.aws_weather.aws_gefs_daily_max")
@patch("src.aws_weather.fallback_open_meteo_daily_max")
def test_unified_routing_all_fail_returns_empty_safely(mock_om, mock_gefs, mock_ecmwf):
    mock_ecmwf.side_effect = RuntimeError("ECMWF fail")
    mock_gefs.side_effect = RuntimeError("GEFS fail")
    mock_om.side_effect = RuntimeError("Open-Meteo fail")

    # Should not raise exception; returns empty list gracefully
    members = aws_unified_daily_max(
        city="london",
        date="2026-10-12",
        unit="celsius",
        fallback_to_open_meteo=True,
    )
    assert members == []
