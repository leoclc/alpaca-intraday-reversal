from __future__ import annotations

import datetime as dt
import logging
import time

from app.config.loader import load_config
from app.live.runner import run_flatten, run_live
from app.utils.time import et_now, parse_time_hhmm
from app.watchlist.daily_strategy_builder import build_watchlist


def _sleep_until(target: dt.datetime) -> None:
    while True:
        now = et_now()
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        time.sleep(min(30.0, max(1.0, remaining)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cfg = load_config()
    params = cfg.get("daily_trend_reversal") or {}

    watchlist_time_str = str(params.get("watchlist_time_et") or "03:00")
    entry_time_str = str(params.get("entry_time_et") or params.get("entry_start_et") or "09:35")
    entry_time = parse_time_hhmm(entry_time_str)
    watchlist_time = parse_time_hhmm(watchlist_time_str)
    now = et_now()
    watchlist_dt = dt.datetime.combine(now.date(), watchlist_time).replace(tzinfo=now.tzinfo)
    if now < watchlist_dt:
        logging.info("[LIVE] waiting for watchlist_time_et=%s", watchlist_time_str)
        _sleep_until(watchlist_dt)
    logging.info("[LIVE] building watchlist for today")
    build_watchlist(cfg)

    entry_dt = dt.datetime.combine(et_now().date(), entry_time).replace(tzinfo=et_now().tzinfo)
    if et_now() < entry_dt:
        logging.info("[LIVE] waiting for entry_time_et=%s", entry_time_str)
        _sleep_until(entry_dt)
    logging.info("[LIVE] placing orders")
    run_live(cfg)

    if bool(params.get("intraday_only", False)):
        session_close_str = str(params.get("session_close_et") or "16:00")
        flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)
        close_time = parse_time_hhmm(session_close_str)
        close_dt = dt.datetime.combine(et_now().date(), close_time).replace(tzinfo=et_now().tzinfo)
        flatten_dt = close_dt - dt.timedelta(minutes=flatten_buffer)
        if et_now() < flatten_dt:
            logging.info("[LIVE] waiting to flatten at %s ET", flatten_dt.time().strftime("%H:%M"))
            _sleep_until(flatten_dt)
        logging.info("[LIVE] flattening positions (intraday_only)")
        run_flatten(cfg)


if __name__ == "__main__":
    main()
