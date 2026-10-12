import sqlite3
import pytest
from src.paper import log_trade, close_trade, connect
from src.position_monitor import (
    evaluate_exit_decision,
    parse_trade_metadata,
    monitor_and_exit_positions,
)


def test_close_trade_pnl_and_metadata():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            strategy TEXT NOT NULL,
            market TEXT NOT NULL,
            market_id TEXT,
            token_index INTEGER,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            model_prob REAL NOT NULL,
            size REAL NOT NULL,
            fee_est REAL NOT NULL,
            meta TEXT,
            outcome INTEGER,
            graded_ts REAL,
            pnl REAL
        )
    """)

    # Log a trade: bought 100 shares @ $0.10 ($10 cost, fee 0.045)
    tid = log_trade(
        con=con,
        strategy="weather_extreme",
        market="london 2026-10-12 16C",
        side="YES",
        price=0.10,
        model_prob=0.25,
        size=100.0,
        fee_est=0.045,
    )

    # Close trade early @ $0.30 ($30 revenue)
    # Entry cost = $10.00, entry fee = $0.045
    # Exit revenue = $30.00, exit fee = 100 * 0.05 * 0.3 * 0.7 = $1.05
    # Realized PnL = $30.00 - $10.00 - 0.045 - 1.05 = $18.905
    pnl = close_trade(con, tid, exit_price=0.30, exit_reason="Take profit surge")

    assert round(pnl, 2) == 18.90

    row = con.execute("SELECT * FROM paper_trades WHERE id=?", (tid,)).fetchone()
    assert row["outcome"] == 1
    assert row["graded_ts"] is not None
    assert round(row["pnl"], 2) == 18.90
    assert "Take profit surge" in row["meta"]


def test_evaluate_exit_observation_breach_no():
    # NO contract on bucket [90.0, 91.0] deg F
    # Observation reached 90.5 deg F -> contract breached!
    should_exit, reason, price = evaluate_exit_decision(
        side="NO",
        entry_price=0.40,
        current_bid=0.15,
        bucket=[90.0, 91.0],
        max_so_far=90.5,
        new_model_prob=None,
        is_today=True,
    )
    assert should_exit is True
    assert "Observation breach" in reason
    assert price == 0.15


def test_evaluate_exit_observation_overshoot_yes():
    # YES contract on bucket [60.0, 61.0] deg F
    # Observation reached 63.0 deg F -> ceiling overshot, daily max cannot decrease!
    should_exit, reason, price = evaluate_exit_decision(
        side="YES",
        entry_price=0.20,
        current_bid=0.02,
        bucket=[60.0, 61.0],
        max_so_far=63.0,
        new_model_prob=None,
        is_today=True,
    )
    assert should_exit is True
    assert "Observation ceiling overshoot" in reason
    assert price == 0.02


def test_evaluate_exit_forecast_reversal_no():
    # NO contract entered at 0.70. New model probability projects YES is 70% (NO is 30%)
    should_exit, reason, price = evaluate_exit_decision(
        side="NO",
        entry_price=0.70,
        current_bid=0.35,
        bucket=[25.0, 25.0],
        max_so_far=None,
        new_model_prob=0.70,  # NO prob is now 30%
        is_today=False,
    )
    assert should_exit is True
    assert "Forecast model reversal" in reason
    assert price == 0.35


def test_evaluate_exit_convexity_take_profit():
    # YES contract bought at 0.015, current bid is 0.060 (4x multiple)
    should_exit, reason, price = evaluate_exit_decision(
        side="YES",
        entry_price=0.015,
        current_bid=0.060,
        bucket=[16.0, 16.0],
        max_so_far=None,
        new_model_prob=0.05,
        is_today=False,
    )
    assert should_exit is True
    assert "Convexity take-profit" in reason
    assert price == 0.060


def test_parse_trade_metadata():
    market = "seoul 2026-10-12 25C"
    meta_json = '{"city": "seoul", "date": "2026-10-12", "bucket": [25.0, 25.0], "unit": "C"}'
    info = parse_trade_metadata(market, meta_json)
    assert info["city"] == "seoul"
    assert info["date"] == "2026-10-12"
    assert info["bucket"] == [25.0, 25.0]
    assert info["unit"] == "C"


def test_monitor_dry_run_execution():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            strategy TEXT NOT NULL,
            market TEXT NOT NULL,
            market_id TEXT,
            token_index INTEGER,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            model_prob REAL NOT NULL,
            size REAL NOT NULL,
            fee_est REAL NOT NULL,
            meta TEXT,
            outcome INTEGER,
            graded_ts REAL,
            pnl REAL
        )
    """)
    tid = log_trade(
        con=con,
        strategy="weather_extreme",
        market="nyc 2026-10-12 60-61F",
        side="NO",
        price=0.50,
        model_prob=0.80,
        size=20.0,
        fee_est=0.10,
        market_id="mock_mkt_1",
        meta='{"city": "nyc", "date": "2026-10-12", "bucket": [60.0, 61.0]}',
    )

    # In dry-run mode, trade remains open (outcome is NULL)
    exits = monitor_and_exit_positions(con, is_live=False, dry_run=True)
    row = con.execute("SELECT outcome FROM paper_trades WHERE id=?", (tid,)).fetchone()
    assert row["outcome"] is None
