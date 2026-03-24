from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.reporting.live_parity import build_live_parity_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a live-vs-backtest parity tracker report.")
    parser.add_argument("--config", default="config.json", help="Config path to use for the parity backtest.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional backtest run directory. Defaults to logs/backtests/parity_<start>_<end>_tracker.",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/parity_tracking",
        help="Directory for parity tracker artifacts.",
    )
    parser.add_argument(
        "--checkpoint-start-date",
        default=None,
        help="Optional checkpoint start date to record separately from the compared range.",
    )
    parser.add_argument(
        "--planned-live-start-date",
        default=None,
        help="Optional planned live-capital start date. Defaults to checkpoint start date + 21 days.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path("logs/backtests") / f"parity_{args.start_date}_to_{args.end_date}_tracker"
    )
    report = build_live_parity_report(
        config_path=Path(args.config),
        start_date=args.start_date,
        end_date=args.end_date,
        run_dir=run_dir,
        out_dir=Path(args.out_dir),
        checkpoint_start_date=args.checkpoint_start_date,
        planned_live_start_date=args.planned_live_start_date,
    )
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["artifacts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

