"""
Supabase Synchronization for Polymarket Weather Bot.

Mirrors paper and live trade records from SQLite (paper.db) to the
Supabase `public.weather_trades` table via PostgREST for dashboard
and real-time monitoring.
"""

from __future__ import annotations
import datetime
import json
import logging
import os
import pathlib
import sqlite3
import threading
from typing import Any, Optional

import requests
from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger("supabase_sync")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "data" / "paper.db"
FALLBACK_ENV_PATH = pathlib.Path(r"C:\Users\Mr\Desktop\files (5)\.env")


def get_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Resolve SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY).
    Priority:
      1. os.environ
      2. polymarket-weather-bot/.env
      3. C:\\Users\\Mr\\Desktop\\files (5)\\.env
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        bot_env = ROOT_DIR / ".env"
        if bot_env.is_file():
            vals = dotenv_values(bot_env)
            url = url or vals.get("SUPABASE_URL")
            key = key or vals.get("SUPABASE_SERVICE_KEY") or vals.get("SUPABASE_ANON_KEY")

    if not url or not key:
        if FALLBACK_ENV_PATH.is_file():
            vals = dotenv_values(FALLBACK_ENV_PATH)
            url = url or vals.get("SUPABASE_URL")
            key = key or vals.get("SUPABASE_SERVICE_KEY") or vals.get("SUPABASE_ANON_KEY")

    return url, key


def _format_timestamp(val: Any) -> Optional[str]:
    """Convert float/int unix epoch or string to ISO 8601 string."""
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        try:
            dt = datetime.datetime.fromtimestamp(float(val), tz=datetime.timezone.utc)
            return dt.isoformat()
        except Exception:
            return None
    if isinstance(val, str):
        try:
            f = float(val)
            dt = datetime.datetime.fromtimestamp(f, tz=datetime.timezone.utc)
            return dt.isoformat()
        except ValueError:
            return val
    return None


def _parse_meta(meta_val: Any) -> Optional[dict]:
    """Safely parse metadata into a dict for jsonb columns."""
    if meta_val is None:
        return None
    if isinstance(meta_val, dict):
        return meta_val
    if isinstance(meta_val, str):
        try:
            return json.loads(meta_val)
        except Exception:
            return {"raw": meta_val}
    return {"raw": str(meta_val)}


def read_paper_trades(db_path: pathlib.Path | str = DEFAULT_DB_PATH) -> list[dict]:
    """Read all rows from paper_trades in SQLite database."""
    db_file = pathlib.Path(db_path)
    if not db_file.exists():
        logger.warning(f"Database file does not exist: {db_file}")
        return []

    con = sqlite3.connect(str(db_file))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'")
        if not cur.fetchone():
            logger.warning(f"Table 'paper_trades' not found in {db_file}")
            return []

        rows = cur.execute("SELECT * FROM paper_trades ORDER BY id ASC").fetchall()
        trades = []
        for r in rows:
            trade = {
                "id": int(r["id"]),
                "ts": _format_timestamp(r["ts"]),
                "strategy": str(r["strategy"]),
                "market": str(r["market"]),
                "market_id": str(r["market_id"]) if r["market_id"] is not None else None,
                "token_index": int(r["token_index"]) if r["token_index"] is not None else 0,
                "side": str(r["side"]),
                "price": float(r["price"]),
                "model_prob": float(r["model_prob"]),
                "size": float(r["size"]),
                "fee_est": float(r["fee_est"]),
                "meta": _parse_meta(r["meta"]),
                "outcome": int(r["outcome"]) if r["outcome"] is not None else None,
                "graded_ts": _format_timestamp(r["graded_ts"]) if r["graded_ts"] is not None else None,
                "pnl": float(r["pnl"]) if r["pnl"] is not None else None,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            trades.append(trade)
        return trades
    finally:
        con.close()


def sync_trades(db_path: pathlib.Path | str = DEFAULT_DB_PATH, batch_size: int = 100) -> int:
    """
    Synchronously upsert all paper trades from SQLite to Supabase weather_trades table.
    Returns the total number of trades synced.
    """
    url, key = get_credentials()
    if not url or not key:
        logger.info("Supabase credentials not configured. Skipping sync.")
        return 0

    trades = read_paper_trades(db_path)
    if not trades:
        logger.info("No trades to sync.")
        return 0

    endpoint = f"{url.rstrip('/')}/rest/v1/weather_trades?on_conflict=id"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    synced_count = 0
    total = len(trades)

    for i in range(0, total, batch_size):
        batch = trades[i : i + batch_size]
        try:
            res = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(batch),
                timeout=15,
            )
            res.raise_for_status()
            synced_count += len(batch)
        except Exception as e:
            logger.error(f"Failed to sync batch {i}-{i+len(batch)}: {e}")
            raise

    logger.info(f"Successfully synced {synced_count}/{total} trades to Supabase.")
    return synced_count


def sync_trades_async(db_path: pathlib.Path | str = DEFAULT_DB_PATH) -> threading.Thread:
    """Trigger trade synchronization in a background daemon thread."""
    def _worker():
        try:
            sync_trades(db_path)
        except Exception as e:
            logger.warning(f"Background Supabase sync encountered an error: {e}")

    t = threading.Thread(target=_worker, name="supabase-sync-worker", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    count = sync_trades()
    print(f"Sync complete. Total trades synced: {count}")
