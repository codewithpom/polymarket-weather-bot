"""
Main CLI Entry Point for Polymarket Weather Trading Bot.

Commands:
  python src/main.py scan [--live]   # Scan active markets and execute approved trades
  python src/main.py grade           # Batch-grade settled trades against station truth
  python src/main.py report          # Print paper performance & T6 gate status
  python src/main.py selftest        # Execute offline self-test suite
"""

from __future__ import annotations
import argparse
import pathlib
import sys

# Ensure repository root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.paper as paper
from src.scanner import scan_and_execute, grade_settled_trades
from src.config import LIVE, FORCE_LIVE


def run_selftest() -> bool:
    print("Running system offline self-tests...")
    import pytest
    res = pytest.main(["tests", "-v"])
    return res == 0


def main():
    parser = argparse.ArgumentParser(description="Polymarket Weather Bot CLI")
    subparsers = parser.add_subparsers(dest="command")

    scan_parser = subparsers.add_parser("scan", help="Scan markets and trade")
    scan_parser.add_argument("--live", action="store_true", help="Attempt live execution")

    subparsers.add_parser("grade", help="Grade settled paper trades")
    subparsers.add_parser("report", help="Print performance report")
    subparsers.add_parser("selftest", help="Run offline unit tests")

    args = parser.parse_args()

    if args.command == "selftest":
        success = run_selftest()
        sys.exit(0 if success else 1)

    con = paper.connect()

    if args.command == "scan":
        is_live = bool(args.live or LIVE)
        print(f"Starting market scan (Mode: {'LIVE' if is_live else 'PAPER'})...")
        count = scan_and_execute(con, is_live=is_live)
        print(f"Scan complete. Trades executed: {count}")
    elif args.command == "grade":
        print("Grading settled trades from station truth...")
        count = grade_settled_trades(con)
        print(f"Grading complete. Trades settled: {count}")
        paper.report(con)
    elif args.command == "report":
        paper.report(con)
    else:
        # Default behavior: grade, scan, report
        print("Default pipeline: Grade -> Scan -> Report")
        g_count = grade_settled_trades(con)
        print(f"Settled {g_count} trades.")
        s_count = scan_and_execute(con, is_live=bool(LIVE))
        print(f"Executed {s_count} trades.")
        paper.report(con)

    con.close()


if __name__ == "__main__":
    main()
