"""
Paper Trading Ledger, Brier Verification Gate, and Rolling Halt Monitor.
"""

from __future__ import annotations
import json
import sqlite3
import time
from src.config import DB_PATH, HALT_FILE, FORCE_LIVE
from src.fees import taker_fee

MIN_TRADES = 50
MIN_DAYS = 14
ROLLING_WINDOW = 100
DAY_SECONDS = 86400.0


def connect(path=DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    with con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
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
    return con


def log_trade(
    con: sqlite3.Connection,
    strategy: str,
    market: str,
    side: str,
    price: float,
    model_prob: float,
    size: float,
    fee_est: float,
    market_id: str = "",
    token_index: int = 0,
    meta: dict | None = None,
    ts: float | None = None,
) -> int:
    if not (0.0 < price < 1.0):
        raise ValueError(f"Price must be in (0, 1), got {price}")
    if not (0.0 <= model_prob <= 1.0):
        raise ValueError(f"model_prob must be in [0, 1], got {model_prob}")
    if size <= 0:
        raise ValueError(f"Size must be > 0, got {size}")

    now = ts if ts is not None else time.time()
    meta_json = json.dumps(meta or {})
    with con:
        cur = con.execute("""
            INSERT INTO paper_trades (
                ts, strategy, market, market_id, token_index, side,
                price, model_prob, size, fee_est, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, strategy, market, market_id, token_index, side,
              price, model_prob, size, fee_est, meta_json))
        return int(cur.lastrowid)


def grade(con: sqlite3.Connection, trade_id: int, outcome: int, graded_ts: float | None = None) -> float:
    """
    Settle trade with binary outcome (1 = win, 0 = loss).
    pnl = size * outcome - size * price - fee_est
    """
    if outcome not in (0, 1):
        raise ValueError("Outcome must be 0 or 1")
    row = con.execute("SELECT outcome, price, size, fee_est FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
    if not row:
        raise KeyError(f"Trade {trade_id} not found")
    if row["outcome"] is not None:
        raise RuntimeError(f"Trade {trade_id} already settled with outcome {row['outcome']}")

    pnl = row["size"] * outcome - row["size"] * row["price"] - row["fee_est"]
    now = graded_ts if graded_ts is not None else time.time()
    with con:
        con.execute("UPDATE paper_trades SET outcome=?, graded_ts=?, pnl=? WHERE id=?",
                    (outcome, now, pnl, trade_id))
    return pnl


def close_trade(
    con: sqlite3.Connection,
    trade_id: int,
    exit_price: float,
    exit_reason: str,
    exit_ts: float | None = None,
) -> float:
    """
    Closes an open position early on the CLOB.
    Realizes PnL = size * exit_price - size * entry_price - entry_fee - exit_fee.
    Marks outcome as 1 (if profitable) or 0 (if loss) to release active market risk,
    and records exit details in meta and pnl.
    """
    if not (0.0 <= exit_price <= 1.0):
        raise ValueError(f"Exit price must be in [0, 1], got {exit_price}")

    row = con.execute("SELECT outcome, price, size, fee_est, meta FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
    if not row:
        raise KeyError(f"Trade {trade_id} not found")
    if row["outcome"] is not None:
        raise RuntimeError(f"Trade {trade_id} already settled with outcome {row['outcome']}")

    now = exit_ts if exit_ts is not None else time.time()
    size = row["size"]
    entry_price = row["price"]
    entry_fee = row["fee_est"]
    exit_fee = taker_fee(size, exit_price, "weather")

    realized_pnl = size * exit_price - size * entry_price - entry_fee - exit_fee
    binary_outcome = 1 if realized_pnl > 0.0 else 0

    try:
        meta_dict = json.loads(row["meta"]) if row["meta"] else {}
    except Exception:
        meta_dict = {}
    meta_dict["exit"] = {
        "exit_price": exit_price,
        "exit_fee": exit_fee,
        "exit_reason": exit_reason,
        "exit_ts": now,
        "early_exit": True,
    }

    with con:
        con.execute("""
            UPDATE paper_trades
            SET outcome=?, graded_ts=?, pnl=?, meta=?
            WHERE id=?
        """, (binary_outcome, now, realized_pnl, json.dumps(meta_dict), trade_id))

    return realized_pnl


def brier_score(con: sqlite3.Connection, strategy: str) -> tuple[float, float, int]:
    """Returns (model_brier, market_brier, count)."""
    rows = con.execute("""
        SELECT model_prob, price, outcome FROM paper_trades
        WHERE strategy=? AND outcome IS NOT NULL
    """, (strategy,)).fetchall()
    if not rows:
        return 0.0, 0.0, 0
    m_sq = sum((r["model_prob"] - r["outcome"]) ** 2 for r in rows)
    mk_sq = sum((r["price"] - r["outcome"]) ** 2 for r in rows)
    n = len(rows)
    return m_sq / n, mk_sq / n, n


def stats(con: sqlite3.Connection, strategy: str) -> dict:
    rows = con.execute("""
        SELECT ts, price, model_prob, size, fee_est, outcome, pnl
        FROM paper_trades WHERE strategy=? AND outcome IS NOT NULL
        ORDER BY ts ASC
    """, (strategy,)).fetchall()
    n = len(rows)
    if not n:
        return {"n": 0, "pnl": 0.0, "hit_rate": 0.0, "days": 0.0,
                "model_brier": 0.0, "market_brier": 0.0, "edge": 0.0}

    total_pnl = sum(r["pnl"] for r in rows)
    wins = sum(1 for r in rows if r["outcome"] == 1)
    days = (rows[-1]["ts"] - rows[0]["ts"]) / DAY_SECONDS if n > 1 else 0.0
    mb, mkb, _ = brier_score(con, strategy)
    return {
        "n": n, "pnl": total_pnl, "hit_rate": wins / n, "days": days,
        "model_brier": mb, "market_brier": mkb, "edge": mkb - mb,
    }


def require_live_ok(con: sqlite3.Connection, strategy: str) -> None:
    """T6 Gatekeeper: Raises PermissionError unless model is verified."""
    if FORCE_LIVE:
        print(f"!! FORCE_LIVE=1 — bypassing T6 gate for {strategy}. Trading ungated on purpose.")
        return

    st = stats(con, strategy)
    if st["n"] < MIN_TRADES:
        raise PermissionError(f"T6 Gate BLOCKED: only {st['n']}/{MIN_TRADES} paper trades")
    if st["days"] < MIN_DAYS:
        raise PermissionError(f"T6 Gate BLOCKED: only {st['days']:.1f}/{MIN_DAYS} trading days")
    if st["model_brier"] >= st["market_brier"]:
        raise PermissionError(f"T6 Gate BLOCKED: model Brier {st['model_brier']:.4f} >= market {st['market_brier']:.4f}")
    if st["pnl"] <= 0:
        raise PermissionError(f"T6 Gate BLOCKED: paper PnL ${st['pnl']:.2f} <= 0")


def check_rolling_halt(con: sqlite3.Connection, strategy: str) -> bool:
    """Trips HALT if model Brier >= market Brier over rolling 100 trades."""
    rows = con.execute(f"""
        SELECT model_prob, price, outcome FROM paper_trades
        WHERE strategy=? AND outcome IS NOT NULL
        ORDER BY graded_ts DESC LIMIT {ROLLING_WINDOW}
    """, (strategy,)).fetchall()
    if len(rows) < ROLLING_WINDOW:
        return False
    mb = sum((r["model_prob"] - r["outcome"]) ** 2 for r in rows) / len(rows)
    mkb = sum((r["price"] - r["outcome"]) ** 2 for r in rows) / len(rows)
    if mb >= mkb:
        with open(HALT_FILE, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] "
                    f"model Brier {mb:.4f} >= market {mkb:.4f} over {len(rows)} trades\n")
        print(f"!! HALT {strategy}: model Brier {mb:.4f} >= market {mkb:.4f} over {len(rows)} trades")
        return True
    return False


def report(con: sqlite3.Connection) -> None:
    strats = [r[0] for r in con.execute("SELECT DISTINCT strategy FROM paper_trades").fetchall()]
    if not strats:
        print("No paper trades logged.")
        return
    print(f"\n{'Strategy':<18}{'N':>5}{'Model Br':>11}{'Market Br':>11}{'Edge':>9}{'PnL':>11}{'Hit Rate':>10}{'Days':>8}  Gate Status")
    print("-" * 95)
    for s in strats:
        st = stats(con, s)
        try:
            require_live_ok(con, s)
            gate = "PASSED (Ready for live)"
        except PermissionError as e:
            gate = str(e).replace("T6 Gate BLOCKED: ", "BLOCKED: ")

        print(f"{s:<18}{st['n']:>5}{st['model_brier']:>11.4f}{st['market_brier']:>11.4f}"
              f"{st['edge']:>+9.4f}{st['pnl']:>+10.2f}${st['hit_rate']:>9.1%}{st['days']:>7.1f}d  {gate}")
    print("-" * 95)
