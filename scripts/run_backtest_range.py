from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.daily_trend_backtest import run_backtest
from app.config.loader import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parity backtest for a date range.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional output path for backtest.json (defaults to logs/backtests/<run_id>/backtest.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cfg = load_config()
    out_path: Optional[str] = args.out_path
    run_backtest(cfg, start_date=args.start_date, end_date=args.end_date, out_path=out_path)


if __name__ == "__main__":
    main()
