"""
Tests for Risk Manager, Kelly sizing, and exposure caps.
"""

import contextlib
import os
import shutil
import tempfile
import time
import pytest
import src.paper as paper
import src.risk as risk


@contextlib.contextmanager
def tmprisk():
    tmp = tempfile.mkdtemp()
    saved_halt = risk.HALT_FILE
    saved_cfg = risk.CONFIG_FILE
    risk.HALT_FILE = os.path.join(tmp, "HALT")
    risk.CONFIG_FILE = os.path.join(tmp, "cfg.json")
    con = paper.connect(os.path.join(tmp, "t.db"))
    try:
        yield con, risk.load_config()
    finally:
        con.close()
        risk.HALT_FILE = saved_halt
        risk.CONFIG_FILE = saved_cfg
        shutil.rmtree(tmp, ignore_errors=True)


def test_kelly_sizing():
    cfg = {"bankroll": 1000.0, "kelly_fraction": 0.25,
           "per_trade_cap_pct": 0.05, "min_stake": 1.0}
    # Edge: p=0.9, price=0.5 -> f* = 0.8 -> 0.8 * 0.25 = 0.2 -> capped at 0.05 = $50
    assert risk.position_size(0.9, 0.5, cfg) == 50.0
    # No edge
    assert risk.position_size(0.5, 0.5, cfg) == 0.0


def test_per_trade_cap():
    with tmprisk() as (con, cfg):
        # Quarter-Kelly wants $200, but per-trade cap is 2% of $1000 = $20
        stake = risk.approve(con, cfg, "nyc 2026-07-20", 0.99, 0.5, "s")
        assert stake == 20.0


def test_per_market_cap():
    with tmprisk() as (con, cfg):
        # Per-market cap is 5% = $50
        # First leg takes $20
        s1 = risk.approve(con, cfg, "nyc 2026-07-20", 0.99, 0.5, "s")
        paper.log_trade(con, "s", "nyc 2026-07-20 80F", "YES", 0.5, 0.99, s1 / 0.5, 0.05)
        # Second leg takes $20 ($40 total)
        s2 = risk.approve(con, cfg, "nyc 2026-07-20", 0.99, 0.5, "s")
        paper.log_trade(con, "s", "nyc 2026-07-20 82F", "YES", 0.5, 0.99, s2 / 0.5, 0.05)
        # Third leg only gets remaining $10
        s3 = risk.approve(con, cfg, "nyc 2026-07-20", 0.99, 0.5, "s")
        assert s3 == 10.0


def test_daily_loss_trips_halt():
    with tmprisk() as (con, cfg):
        # Daily loss limit is 5% of $1000 = $50
        # Loss of $60 trips halt
        tid = paper.log_trade(con, "s", "m", "YES", 0.5, 0.9, 120.0, 0.0, ts=time.time())
        paper.grade(con, tid, 0)  # loses 120 * 0.5 = -$60
        risk.check_daily_loss(con, cfg, "s")
        assert risk.halted() is True
