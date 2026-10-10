"""
Tests for T6 Paper Trading Gate and Fail-Closed Safeguards.
"""

import contextlib
import os
import shutil
import tempfile
import time
import pytest
import src.paper as paper

DAY = 86400.0


@contextlib.contextmanager
def tmpcon():
    tmp = tempfile.mkdtemp()
    con = paper.connect(os.path.join(tmp, "t.db"))
    saved_halt = paper.HALT_FILE
    paper.HALT_FILE = os.path.join(tmp, "HALT")
    try:
        yield con
    finally:
        con.close()
        paper.HALT_FILE = saved_halt
        shutil.rmtree(tmp, ignore_errors=True)


def _fill(con, strategy, n, model_prob, price, outcome, days=20, size=10.0, fee=0.05):
    t0 = time.time() - days * DAY
    for i in range(n):
        tid = paper.log_trade(con, strategy, f"m{i}", "YES", price, model_prob,
                              size, fee, ts=t0 + i * (days * DAY / max(n - 1, 1)))
        paper.grade(con, tid, outcome)


def test_pnl_math():
    with tmpcon() as con:
        tid = paper.log_trade(con, "s", "m", "YES", 0.40, 0.9, 10.0, 0.05)
        assert abs(paper.grade(con, tid, 1) - 5.95) < 1e-9
        tid = paper.log_trade(con, "s", "m", "YES", 0.40, 0.9, 10.0, 0.05)
        assert abs(paper.grade(con, tid, 0) - (-4.05)) < 1e-9


def test_no_regrade():
    with tmpcon() as con:
        tid = paper.log_trade(con, "s", "m", "YES", 0.4, 0.9, 10.0, 0.05)
        paper.grade(con, tid, 1)
        with pytest.raises(RuntimeError):
            paper.grade(con, tid, 0)


def test_brier_math():
    with tmpcon() as con:
        _fill(con, "s", 10, model_prob=0.9, price=0.5, outcome=1, days=20)
        mb, mkb, n = paper.brier_score(con, "s")
        assert abs(mb - 0.01) < 1e-9
        assert abs(mkb - 0.25) < 1e-9


def test_gate_blocks_under_50_trades():
    with tmpcon() as con:
        _fill(con, "s", 49, model_prob=0.99, price=0.5, outcome=1, days=20)
        with pytest.raises(PermissionError, match="only 49/50"):
            paper.require_live_ok(con, "s")


def test_gate_blocks_under_14_days():
    with tmpcon() as con:
        _fill(con, "s", 60, model_prob=0.99, price=0.5, outcome=1, days=10)
        with pytest.raises(PermissionError, match="trading days"):
            paper.require_live_ok(con, "s")


def test_gate_blocks_uninformed_profitable():
    with tmpcon() as con:
        # Every trade wins, but model assigned 10% vs market 50%
        _fill(con, "s", 60, model_prob=0.1, price=0.5, outcome=1, days=20)
        with pytest.raises(PermissionError, match="model Brier"):
            paper.require_live_ok(con, "s")


def test_gate_blocks_informed_unprofitable():
    with tmpcon() as con:
        _fill(con, "s", 60, model_prob=0.1, price=0.5, outcome=0, days=20, fee=100.0)
        with pytest.raises(PermissionError, match="paper PnL"):
            paper.require_live_ok(con, "s")


def test_gate_passes_clean_model():
    with tmpcon() as con:
        _fill(con, "s", 60, model_prob=0.9, price=0.5, outcome=1, days=20)
        paper.require_live_ok(con, "s")


def test_rolling_halt():
    with tmpcon() as con:
        _fill(con, "s", 100, model_prob=0.1, price=0.5, outcome=1, days=20)
        assert paper.check_rolling_halt(con, "s") is True
        assert os.path.exists(paper.HALT_FILE)
