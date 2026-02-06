from __future__ import annotations

import datetime as dt
import logging
import time

from app.config.loader import load_config
from app.brokers.alpaca import AlpacaBroker
from app.live.minute_report import log_account_summary, log_minute_report
from app.live.runner import run_flatten, run_live
from app.utils.time import ET_TZ, et_now, parse_time_hhmm
from app.watchlist.daily_strategy_builder import build_watchlist
from app.watchlist.storage import read_watchlist, expected_watchlist_date_str


def _sleep_until(target: dt.datetime) -> None:
    while True:
        now = et_now()
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        time.sleep(min(30.0, max(1.0, remaining)))


def _schedule_for_day(cfg: dict, day: dt.date) -> tuple[dt.datetime, list[dt.datetime], dt.datetime]:
    params = cfg.get("daily_trend_reversal") or {}
    watchlist_time_str = str(params.get("watchlist_time_et") or "03:00")
    entry_time_str = str(params.get("entry_time_et") or params.get("entry_start_et") or "09:35")
    entry_times_raw = params.get("entry_times_et")
    if isinstance(entry_times_raw, list) and entry_times_raw:
        entry_time_strs = [str(t) for t in entry_times_raw if t]
    else:
        entry_time_strs = [entry_time_str]
    entry_time_strs = sorted(entry_time_strs, key=lambda t: parse_time_hhmm(t))
    session_close_str = str(params.get("session_close_et") or "16:00")
    flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)

    watchlist_time = parse_time_hhmm(watchlist_time_str)
    entry_times = [parse_time_hhmm(t) for t in entry_time_strs]
    close_time = parse_time_hhmm(session_close_str)

    base = dt.datetime.combine(day, watchlist_time).replace(tzinfo=ET_TZ)
    entry_dts = [dt.datetime.combine(day, t).replace(tzinfo=ET_TZ) for t in entry_times]
    close_dt = dt.datetime.combine(day, close_time).replace(tzinfo=ET_TZ)
    flatten_dt = close_dt - dt.timedelta(minutes=flatten_buffer)
    return base, entry_dts, flatten_dt


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cfg = load_config()
    params = cfg.get("daily_trend_reversal") or {}
    watchlist_time_str = str(params.get("watchlist_time_et") or "03:00")
    entry_time_str = str(params.get("entry_time_et") or params.get("entry_start_et") or "09:35")
    entry_times_raw = params.get("entry_times_et")
    if isinstance(entry_times_raw, list) and entry_times_raw:
        entry_time_strs = [str(t) for t in entry_times_raw if t]
    else:
        entry_time_strs = [entry_time_str]
    entry_time_strs = sorted(entry_time_strs, key=lambda t: parse_time_hhmm(t))
    intraday_only = bool(params.get("intraday_only", False))
    minute_report_enabled = bool(params.get("minute_report_enabled", True))
    minute_report_interval = int(params.get("minute_report_interval_sec") or 60)

    last_watchlist_date: dt.date | None = None
    last_entry_date: dt.date | None = None
    last_flatten_date: dt.date | None = None
    plans: list = []
    entry_time_map: dict[str, str] = {}

    while True:
        now = et_now()
        today = now.date()
        watchlist_dt, entry_dts, flatten_dt = _schedule_for_day(cfg, today)
        last_entry_dt = entry_dts[-1] if entry_dts else None
        if today.weekday() >= 5:
            logging.info("[LIVE] weekend detected; skipping %s", today.isoformat())
            tomorrow = today + dt.timedelta(days=1)
            next_watchlist_dt, _, _ = _schedule_for_day(cfg, tomorrow)
            _sleep_until(next_watchlist_dt)
            continue

        # If we're already past entry time, skip the day entirely.
        if last_entry_dt and now >= last_entry_dt and last_entry_date != today:
            logging.info("[LIVE] entry_time_et already passed; skipping trading for %s", today.isoformat())
            last_watchlist_date = today
            last_entry_date = today
            last_flatten_date = today
            tomorrow = today + dt.timedelta(days=1)
            next_watchlist_dt, _, _ = _schedule_for_day(cfg, tomorrow)
            _sleep_until(next_watchlist_dt)
            continue

        if last_watchlist_date != today:
            if now < watchlist_dt:
                logging.info("[LIVE] waiting for watchlist_time_et=%s", watchlist_time_str)
                _sleep_until(watchlist_dt)
            else:
                logging.info("[LIVE] watchlist_time_et=%s already passed; building now", watchlist_time_str)
            broker = AlpacaBroker(cfg)
            log_account_summary(broker)
            logging.info("[LIVE] building watchlist for today")
            build_watchlist(cfg)
            wl = read_watchlist(expected_watchlist_date_str(), cfg)
            entry_time_map = {}
            for row in wl.get("watchlist") or []:
                sym = str(row.get("symbol") or "").upper()
                if not sym:
                    continue
                entry_time_map[sym] = str(row.get("entry_time_et") or entry_time_strs[0])
            last_watchlist_date = today

        if last_entry_date != today:
            traded_symbols: set[str] = set()
            plans = []
            for entry_time_str, entry_dt in zip(entry_time_strs, entry_dts):
                if et_now() < entry_dt:
                    logging.info("[LIVE] waiting for entry_time_et=%s", entry_time_str)
                    _sleep_until(entry_dt)
                else:
                    logging.info("[LIVE] entry_time_et=%s already passed; placing orders now", entry_time_str)
                logging.info("[LIVE] placing orders for entry_time_et=%s", entry_time_str)
                allowed_symbols = None
                if entry_time_map:
                    allowed_symbols = {s for s, t in entry_time_map.items() if t == entry_time_str}
                    if traded_symbols:
                        allowed_symbols -= traded_symbols
                    if not allowed_symbols:
                        continue
                placed, new_plans = run_live(
                    cfg,
                    return_plans=True,
                    entry_time_et=entry_time_str,
                    symbols_allow=allowed_symbols,
                )
                for resp in placed or []:
                    sym = str(resp.get("symbol") or "").upper()
                    if sym:
                        traded_symbols.add(sym)
                for plan in new_plans or []:
                    sym = str(getattr(plan, "symbol", "") or "").upper()
                    if sym:
                        traded_symbols.add(sym)
                    plans.append(plan)
            last_entry_date = today

        if last_entry_date == today and minute_report_enabled:
            report_until = flatten_dt if intraday_only else _schedule_for_day(cfg, today)[2]
            while et_now() < report_until:
                broker = AlpacaBroker(cfg)
                try:
                    log_minute_report(cfg, broker, plans)
                except Exception as exc:
                    logging.error("[LIVE] minute report failed: %s", exc)
                time.sleep(max(5, minute_report_interval))

        if intraday_only and last_flatten_date != today:
            if et_now() < flatten_dt:
                logging.info("[LIVE] waiting to flatten at %s ET", flatten_dt.time().strftime("%H:%M"))
                _sleep_until(flatten_dt)
            logging.info("[LIVE] flattening positions (intraday_only)")
            run_flatten(cfg)
            last_flatten_date = today

        # Sleep until next day's watchlist time.
        tomorrow = today + dt.timedelta(days=1)
        next_watchlist_dt, _, _ = _schedule_for_day(cfg, tomorrow)
        logging.info("[LIVE] day complete; sleeping until %s ET", next_watchlist_dt.isoformat())
        _sleep_until(next_watchlist_dt)


if __name__ == "__main__":
    main()
