"""
Unit tests for Supabase sync functionality.
"""

import json
import sqlite3
import tempfile
import pathlib
from unittest.mock import patch, MagicMock

import pytest
from src.supabase_sync import (
    _format_timestamp,
    _parse_meta,
    read_paper_trades,
    sync_trades,
    get_credentials,
)


def test_format_timestamp():
    # Unix epoch
    ts_epoch = 1728600000.0
    formatted = _format_timestamp(ts_epoch)
    assert formatted is not None
    assert "2024-" in formatted or "2026-" in formatted or "+00:00" in formatted

    # None and empty
    assert _format_timestamp(None) is None
    assert _format_timestamp("") is None

    # String ISO
    iso_str = "2026-10-11T12:00:00Z"
    assert _format_timestamp(iso_str) == iso_str


def test_parse_meta():
    assert _parse_meta(None) is None
    assert _parse_meta({"city": "miami"}) == {"city": "miami"}
    assert _parse_meta('{"city": "london", "temp": 14}') == {"city": "london", "temp": 14}
    assert _parse_meta("invalid-json") == {"raw": "invalid-json"}


def test_read_paper_trades_and_sync_mock(tmp_path):
    db_file = tmp_path / "test_paper.db"
    con = sqlite3.connect(str(db_file))
    with con:
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
        con.execute("""
            INSERT INTO paper_trades (
                ts, strategy, market, market_id, token_index, side,
                price, model_prob, size, fee_est, meta, outcome, graded_ts, pnl
            ) VALUES (
                1791661078.0, 'weather_extreme', 'miami test', '12345', 1, 'NO',
                0.05, 0.70, 100.0, 0.50, '{"city": "miami"}', 1, 1791670000.0, 94.50
            )
        """)
    con.close()

    rows = read_paper_trades(db_file)
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["strategy"] == "weather_extreme"
    assert rows[0]["market_id"] == "12345"
    assert rows[0]["meta"] == {"city": "miami"}
    assert rows[0]["outcome"] == 1
    assert rows[0]["pnl"] == 94.50
    assert rows[0]["ts"] is not None

    # Mock requests.post and get_credentials
    with patch("src.supabase_sync.get_credentials", return_value=("https://test.supabase.co", "test-key")):
        with patch("requests.post") as mock_post:
            mock_res = MagicMock()
            mock_res.raise_for_status.return_value = None
            mock_post.return_value = mock_res

            synced = sync_trades(db_path=db_file)
            assert synced == 1
            assert mock_post.called

            args, kwargs = mock_post.call_args
            assert args[0] == "https://test.supabase.co/rest/v1/weather_trades?on_conflict=id"
            headers = kwargs["headers"]
            assert headers["apikey"] == "test-key"
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Prefer"] == "resolution=merge-duplicates,return=minimal"
            payload = json.loads(kwargs["data"])
            assert len(payload) == 1
            assert payload[0]["id"] == 1
            assert payload[0]["market"] == "miami test"
