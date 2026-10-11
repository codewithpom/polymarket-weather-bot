#!/usr/bin/env python3
"""
Benchmarking CLI Tool: AWS S3 Direct GEFS Ingestion vs Open-Meteo Multi-Model Ensemble API.

Compares:
1. Ingestion latency and query response time (ms).
2. Ensemble distribution summary statistics (min, max, mean, standard deviation, spread).
3. Evaluated Polymarket contract bucket probabilities and pricing differences.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import pathlib
import sys
import time
from typing import Any

# Ensure UTF-8 output encoding for terminals where supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure repository root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    RESOLUTION_SOURCES,
    MODEL_COORDS,
    CITY_CALIB,
    BUCKET_WIDTH,
)
from src.scanner import ensemble_daily_max, _ENSEMBLE_CACHE
from src.weather_model import ensemble_bucket_prob


def _get_aws_module():
    """Dynamically imports src.aws_weather if available."""
    try:
        import src.aws_weather as aws_weather
        return aws_weather
    except ImportError:
        return None


def calculate_statistics(members: list[float]) -> dict[str, float]:
    """Computes summary statistics for a given ensemble sample."""
    if not members:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "spread": 0.0,
        }
    n = len(members)
    mean_val = sum(members) / n
    variance = sum((x - mean_val) ** 2 for x in members) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance)
    min_val = min(members)
    max_val = max(members)
    spread_val = max_val - min_val

    return {
        "count": float(n),
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "spread": spread_val,
    }


def generate_sample_buckets(
    center_temp: float,
    width: float,
    num_buckets: int = 5,
    unit: str = "F",
) -> list[tuple[float | None, float | None, str]]:
    """
    Generates representative Polymarket temperature contract buckets centered
    around the estimated mean.
    """
    center_aligned = round(center_temp / width) * width
    buckets: list[tuple[float | None, float | None, str]] = []

    half = num_buckets // 2
    # Lower tail bucket (e.g. <= center - half*width)
    tail_lo_val = center_aligned - (half * width)
    buckets.append((None, tail_lo_val, f"<= {tail_lo_val:.1f} deg{unit}"))

    # Interior buckets
    for i in range(-half, half):
        lo = center_aligned + (i * width)
        hi = lo + width
        buckets.append((lo, hi, f"{lo:.1f} to {hi:.1f} deg{unit}"))

    # Upper tail bucket
    tail_hi_val = center_aligned + (half * width)
    buckets.append((tail_hi_val, None, f">= {tail_hi_val:.1f} deg{unit}"))

    return buckets


def run_openmeteo_benchmark(
    lat: float,
    lon: float,
    date: str,
    unit: str,
    flush_cache: bool = True,
) -> tuple[list[float], float, str | None]:
    """Measures latency and fetches Open-Meteo multi-model ensemble."""
    if flush_cache:
        cache_key = (round(lat, 4), round(lon, 4), date, unit)
        _ENSEMBLE_CACHE.pop(cache_key, None)

    t_start = time.perf_counter()
    try:
        members = ensemble_daily_max(lat, lon, date, unit, timeout=45)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return members, latency_ms, None
    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return [], latency_ms, str(exc)


def run_aws_benchmark(
    city: str,
    date: str,
    unit: str,
    members: int,
    mock: bool = False,
) -> tuple[list[float], float, str | None]:
    """Measures latency and fetches AWS S3 GEFS ensemble."""
    if mock:
        t_start = time.perf_counter()
        time.sleep(0.05)  # simulate rapid direct S3 pull
        # Synthetic GEFS ensemble centered at realistic city temperature
        base = 66.5 if unit.lower().startswith("f") else 16.5
        import random
        random.seed(42)
        mock_members = [
            round(base + random.gauss(0.0, 2.2), 2)
            for _ in range(members)
        ]
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return mock_members, latency_ms, None

    aws_mod = _get_aws_module()
    if aws_mod is None or not hasattr(aws_mod, "aws_gefs_daily_max"):
        return [], 0.0, "AWS direct module (src/aws_weather.py) is not available"

    # Clear internal cache if present
    if hasattr(aws_mod, "clear_caches"):
        aws_mod.clear_caches()
    elif hasattr(aws_mod, "_DAILY_MAX_CACHE") and isinstance(aws_mod._DAILY_MAX_CACHE, dict):
        aws_mod._DAILY_MAX_CACHE.clear()

    t_start = time.perf_counter()
    try:
        members_res = aws_mod.aws_gefs_daily_max(
            city=city,
            date=date,
            unit=unit,
            num_members=members,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return members_res, latency_ms, None
    except Exception as exc:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return [], latency_ms, str(exc)


def print_comparison_table(
    city: str,
    date: str,
    unit_symbol: str,
    bias: float,
    smear: float,
    precision: float,
    width: float,
    aws_members: list[float],
    aws_lat_ms: float,
    aws_err: str | None,
    om_members: list[float],
    om_lat_ms: float,
    om_err: str | None,
) -> None:
    """Pretty prints the formatted benchmarking report in the console."""
    sep_double = "=" * 94
    sep_single = "-" * 94

    print("\n" + sep_double)
    print("   WEATHER DATA INGESTION BENCHMARK: AWS S3 DIRECT (GEFS) vs OPEN-METEO ENSEMBLE   ")
    print(sep_double)
    icao_str = RESOLUTION_SOURCES[city].get("icao", "HKO")
    print(f"  Target City:        {city.upper():<12} Station: {RESOLUTION_SOURCES[city]['station']} ({icao_str})")
    print(f"  Target Date:        {date:<12} Coordinates: {MODEL_COORDS[city][0]:.4f} N, {MODEL_COORDS[city][1]:.4f} E")
    print(f"  Unit:               deg {unit_symbol}         Calibration: Bias={bias:+.2f}, Smear={smear:.2f}")
    print(f"  Bucket Width:       {width:.1f} deg {unit_symbol}     Precision: {precision:.1f}")
    print(sep_single)

    # 1. Latency & Ingestion Speed Table
    print("\n[1] LATENCY & THROUGHPUT COMPARISON:")
    print("+" + "-" * 26 + "+" + "-" * 14 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+")
    print(f"| {'Engine':<24} | {'Status':<12} | {'Latency (ms)':<14} | {'Members Pulled':<14} | {'Speed / Advantage':<14} |")
    print("+" + "-" * 26 + "+" + "-" * 14 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+")

    aws_status = "SUCCESS" if not aws_err else "ERROR"
    om_status = "SUCCESS" if not om_err else "ERROR"

    if aws_status == "SUCCESS" and om_status == "SUCCESS" and aws_lat_ms > 0:
        ratio = om_lat_ms / aws_lat_ms
        if ratio >= 1.0:
            speed_adv = f"{ratio:.2f}x FASTER"
        else:
            speed_adv = f"{1.0 / ratio:.2f}x SLOWER"
    elif aws_status != "SUCCESS":
        speed_adv = "N/A (Failed)"
    else:
        speed_adv = "Baseline"

    aws_mem_str = f"{len(aws_members)} mem"
    om_mem_str = f"{len(om_members)} mem"

    print(f"| {'AWS S3 Direct (NOAA GEFS)':<24} | {aws_status:<12} | {aws_lat_ms:>11.1f} ms | {aws_mem_str:>14} | {speed_adv:<16} |")
    print(f"| {'Open-Meteo Multi-Model':<24} | {om_status:<12} | {om_lat_ms:>11.1f} ms | {om_mem_str:>14} | {'Baseline':<16} |")
    print("+" + "-" * 26 + "+" + "-" * 14 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+")

    if aws_err:
        print(f"  * AWS Error Detail: {aws_err}")
    if om_err:
        print(f"  * Open-Meteo Error Detail: {om_err}")

    # 2. Ensemble Summary Statistics Table
    aws_stats = calculate_statistics(aws_members)
    om_stats = calculate_statistics(om_members)

    print("\n[2] ENSEMBLE SUMMARY STATISTICS:")
    print("+" + "-" * 26 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 10 + "+")
    print(f"| {'Metric':<24} | {'AWS S3 Direct':<16} | {'Open-Meteo':<16} | {'Delta (AWS - OM)':<16} | {'% Diff':<8} |")
    print("+" + "-" * 26 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 10 + "+")

    metrics = [
        ("Member Count", "count", False),
        ("Min Temperature", "min", True),
        ("Max Temperature", "max", True),
        ("Mean Temperature (mean)", "mean", True),
        ("Std Deviation (std)", "std", True),
        ("Spread (Max - Min)", "spread", True),
    ]

    for label, key, is_temp in metrics:
        v_aws = aws_stats[key]
        v_om = om_stats[key]
        diff = v_aws - v_om
        pct_diff = (diff / v_om * 100.0) if v_om != 0 else 0.0

        if is_temp:
            aws_str = f"{v_aws:>6.2f} deg{unit_symbol}"
            om_str = f"{v_om:>6.2f} deg{unit_symbol}"
            diff_str = f"{diff:>+6.2f} deg{unit_symbol}"
        else:
            aws_str = f"{int(v_aws):>6}"
            om_str = f"{int(v_om):>6}"
            diff_str = f"{int(diff):>+6}"

        pct_str = f"{pct_diff:>+6.1f}%"
        print(f"| {label:<24} | {aws_str:>16} | {om_str:>16} | {diff_str:>16} | {pct_str:>8} |")

    print("+" + "-" * 26 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 10 + "+")

    # 3. Probability Difference for Sample Temperature Buckets
    print("\n[3] POLYMARKET CONTRACT BUCKET PROBABILITY COMPARISON:")
    center_est = om_stats["mean"] if om_stats["count"] > 0 else (aws_stats["mean"] if aws_stats["count"] > 0 else 65.0)
    buckets = generate_sample_buckets(center_est, width, num_buckets=5, unit=unit_symbol)

    print("+" + "-" * 24 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+")
    print(f"| {'Bucket Interval':<22} | {'P(AWS S3 GEFS)':<14} | {'P(Open-Meteo)':<14} | {'Prob Delta':<14} | {'Abs Delta':<14} |")
    print("+" + "-" * 24 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+")

    max_abs_delta = 0.0
    for lo, hi, label in buckets:
        p_aws = ensemble_bucket_prob(
            members=aws_members,
            lo=lo,
            hi=hi,
            bias=bias,
            smear=smear,
            precision=precision,
            bucket_width=width,
            dist_type="student_t",
            df=5.0,
        ) if aws_members else 0.0

        p_om = ensemble_bucket_prob(
            members=om_members,
            lo=lo,
            hi=hi,
            bias=bias,
            smear=smear,
            precision=precision,
            bucket_width=width,
            dist_type="student_t",
            df=5.0,
        ) if om_members else 0.0

        delta = p_aws - p_om
        abs_delta = abs(delta)
        if abs_delta > max_abs_delta:
            max_abs_delta = abs_delta

        aws_p_str = f"{p_aws * 100:>6.2f}%"
        om_p_str = f"{p_om * 100:>6.2f}%"
        delta_str = f"{delta * 100:>+6.2f}%"
        abs_str = f"{abs_delta * 100:>6.2f}%"

        print(f"| {label:<22} | {aws_p_str:>14} | {om_p_str:>14} | {delta_str:>14} | {abs_str:>14} |")

    print("+" + "-" * 24 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+" + "-" * 16 + "+")

    # 4. Benchmarking Takeaway & Conclusion
    print("\n[4] STRATEGIC TAKEAWAYS:")
    if aws_status == "SUCCESS" and om_status == "SUCCESS":
        lat_diff = om_lat_ms - aws_lat_ms
        if lat_diff > 0:
            print(f"  * Latency Edge: AWS Direct is {lat_diff:.1f}ms faster ({om_lat_ms / aws_lat_ms:.2f}x speedup).")
        else:
            print(f"  * Latency Delta: Open-Meteo is {-lat_diff:.1f}ms faster than AWS Direct.")
        print(f"  * Ensemble Agreement: Mean temp difference between models is {abs(aws_stats['mean'] - om_stats['mean']):.2f} deg{unit_symbol}.")
        print(f"  * Maximum Bucket Probability Delta: {max_abs_delta * 100:.2f}%.")
        if max_abs_delta < 0.05:
            print("  * Calibration Status: HIGH CONSISTENCY - Both models align within +/-5% probability envelope.")
        else:
            print("  * Calibration Status: DIVERGENCE DETECTED - Significant model alpha exists between GEFS and multi-model.")
    elif aws_status != "SUCCESS":
        print(f"  * AWS Direct Ingestion encountered an issue: {aws_err}")
        print("  * Production scanner will automatically fallback to Open-Meteo multi-model ensemble.")
    print(sep_double + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tool comparing AWS S3 direct NOAA GEFS pull vs Open-Meteo ensemble API."
    )
    tomorrow_iso = (dt.date.today() + dt.timedelta(days=1)).isoformat()

    parser.add_argument(
        "--city",
        type=str,
        default="nyc",
        help="City code to benchmark (e.g., nyc, miami, london, seoul, beijing, hong kong). Default: nyc",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=tomorrow_iso,
        help=f"Target forecast date in YYYY-MM-DD format. Default: tomorrow ({tomorrow_iso})",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=30,
        help="Number of GEFS ensemble members to request (default: 30).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run benchmark with simulated/mock AWS data for offline validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city = args.city.strip().lower()

    if city not in RESOLUTION_SOURCES:
        valid_cities = ", ".join(RESOLUTION_SOURCES.keys())
        print(f"Error: Unknown city '{args.city}'. Available cities: {valid_cities}", file=sys.stderr)
        sys.exit(1)

    src = RESOLUTION_SOURCES[city]
    lat, lon = MODEL_COORDS[city]
    unit_str = "celsius" if src["unit"] == "C" else "fahrenheit"
    unit_symbol = src["unit"]
    bias, smear = CITY_CALIB[city]
    precision = src["precision"]
    width = BUCKET_WIDTH[city]

    print(f"Starting weather ingestion benchmark for {city.upper()} on {args.date}...")

    # Run Open-Meteo pull
    print("-> Querying Open-Meteo multi-model ensemble API...")
    om_members, om_lat_ms, om_err = run_openmeteo_benchmark(lat, lon, args.date, unit_str)

    # Run AWS S3 Direct pull
    print(f"-> Querying AWS S3 NOAA GEFS direct ({args.members} members)..." + (" [MOCK]" if args.mock else ""))
    aws_members, aws_lat_ms, aws_err = run_aws_benchmark(
        city=city,
        date=args.date,
        unit=unit_str,
        members=args.members,
        mock=args.mock,
    )

    # Output formatted report
    print_comparison_table(
        city=city,
        date=args.date,
        unit_symbol=unit_symbol,
        bias=bias,
        smear=smear,
        precision=precision,
        width=width,
        aws_members=aws_members,
        aws_lat_ms=aws_lat_ms,
        aws_err=aws_err,
        om_members=om_members,
        om_lat_ms=om_lat_ms,
        om_err=om_err,
    )


if __name__ == "__main__":
    main()
