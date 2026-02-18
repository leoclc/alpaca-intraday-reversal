import argparse
import logging
import os
import sys
from pathlib import Path

# Allow `python scripts/run_backtest_cli.py ...` without requiring PYTHONPATH setup.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.backtest.daily_trend_backtest import run_backtest
from app.config.loader import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Run backtest with an explicit config path and output directory.")
    ap.add_argument("--config", required=True, help="Path to config JSON (e.g. configs/candidate_*.json)")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--run-dir", required=True, help="Output directory (e.g. logs/backtests/my_run)")
    args = ap.parse_args()

    cfg_path = str(Path(args.config).resolve())
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = str((run_dir / "backtest.json").resolve())
    log_path = run_dir / "run.log"

    # Ensure the main loader path resolution stays consistent with other entrypoints.
    os.environ["APP_CONFIG_PATH"] = cfg_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
        ],
    )

    cfg = load_config()
    try:
        run_backtest(cfg, start_date=args.start_date, end_date=args.end_date, out_path=out_path)
    except Exception:
        logging.exception("[BACKTEST] failed run_dir=%s", str(run_dir))
        raise
    logging.info("[BACKTEST] finished run_dir=%s", str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
