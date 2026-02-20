from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.data.alpaca_intraday_store import filter_intraday_bars_until, get_intraday_bars
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.execution.daily_execution_model import simulate_exit
from app.market.filters import market_filter_decision
from app.strategies.daily_trend_reversal import build_trade, generate_signal_for_date
from app.strategies.types import TradeResult
from app.utils.time import iter_trading_days, parse_time_hhmm
from app.watchlist.day_filter import day_filter_decision
from app.watchlist.storage import read_watchlist

_DEFAULT_RUN_ID: Optional[str] = None


def _add_minutes(time_str: str, minutes: int) -> str:
    if not time_str or minutes <= 0:
        return time_str
    try:
        base = parse_time_hhmm(time_str)
        shifted = dt.datetime.combine(dt.date.today(), base) + dt.timedelta(minutes=minutes)
        return shifted.strftime("%H:%M")
    except Exception:
        return time_str


def run_replay(
    cfg: Dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_store: Optional[AlpacaOHLCStore] = None,
    run_id: Optional[str] = None,
) -> List[TradeResult]:
    replay_cfg = cfg.get("replay") or {}
    start = start_date or replay_cfg.get("start_date")
    end = end_date or replay_cfg.get("end_date") or start
    if not start or not end:
        raise ValueError("Replay requires start_date and end_date")
    data_store = data_store or AlpacaOHLCStore(cfg=cfg)
    emit_details = bool(replay_cfg.get("emit_daily_details", False))
    details_root: Optional[Path] = None
    if emit_details:
        logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
        run_id = run_id or str(replay_cfg.get("run_id") or "")
        global _DEFAULT_RUN_ID
        if not run_id:
            if _DEFAULT_RUN_ID is None:
                _DEFAULT_RUN_ID = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = _DEFAULT_RUN_ID
        details_root = logs_dir / "replay_runs" / str(run_id)
        details_root.mkdir(parents=True, exist_ok=True)
    trades: List[TradeResult] = []
    for day in iter_trading_days(start, end):
        date_str = day.isoformat()
        skip, info = market_filter_decision(date_str, cfg, data_store)
        if skip:
            logging.info("[REPLAY] market filter skip date=%s info=%s", date_str, info)
            continue
        wl = read_watchlist(date_str, cfg)
        wl_rows = wl.get("watchlist") or []
        skip_day, day_info = day_filter_decision(wl_rows, cfg, meta=wl.get("meta"))
        if skip_day:
            logging.info("[REPLAY] day_filter skip date=%s info=%s", date_str, day_info)
            continue
        symbol_entry_time = {
            str(r.get("symbol") or "").upper(): str(r.get("entry_time_et") or "")
            for r in wl_rows
            if r.get("symbol")
        }
        symbol_overrides = {
            str(r.get("symbol") or "").upper(): (r.get("param_overrides") or {})
            for r in wl_rows
            if r.get("symbol")
        }
        symbol_watchlist_stats = {}
        for idx, row in enumerate(wl_rows, start=1):
            sym = str((row or {}).get("symbol") or "").upper()
            if not sym:
                continue
            symbol_watchlist_stats[sym] = {
                "rank": idx,
                "avgR": row.get("avgR"),
                "avgR_stderr": row.get("avgR_stderr"),
                "win_rate": row.get("win_rate"),
                "profit_factor": row.get("profit_factor"),
                "trades_count": row.get("trades_count"),
                "total_pnl_pct": row.get("total_pnl_pct"),
            }
        symbols = list(symbol_entry_time.keys())
        day_trades = 0
        skip_counts = {
            "no_signal": 0,
            "no_intraday": 0,
            "no_intraday_before_cutoff": 0,
            "no_plan": 0,
            "no_exit": 0,
        }
        # Keep sample no-plan diagnostics small so normal full-year runs don't spam logs.
        debug_limit = int((replay_cfg.get("debug_no_plan_limit") or 0) or 0)
        debug_emitted = 0
        day_details: List[Dict] = []
        if not symbols:
            logging.info("[REPLAY] date=%s watchlist empty or missing", date_str)
            continue
        params = cfg.get("daily_trend_reversal") or {}
        entry_times_raw = params.get("entry_times_et")
        if isinstance(entry_times_raw, list) and entry_times_raw:
            entry_times = [str(t) for t in entry_times_raw if t]
        else:
            entry_times = [str(params.get("entry_time_et") or "09:35")]
        # Watchlist may include per-symbol entry-time overrides (from the daily builder). Make sure we
        # fetch enough intraday minutes to cover the latest entry time in the watchlist too.
        wl_entry_times = [str(t) for t in symbol_entry_time.values() if t]
        entry_times_for_fetch = sorted({t for t in (entry_times + wl_entry_times) if t})
        intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
        early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
        time_stop_minutes = int(params.get("time_stop_minutes") or 0)
        intraday_only = bool(params.get("intraday_only", False))
        confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
        confirm_minutes = int(params.get("confirm_minutes") or 0)
        apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0
        minutes_needed = 0
        if early_range_minutes > 0:
            minutes_needed = max(minutes_needed, early_range_minutes)
        use_intraday_entry = bool(params.get("use_intraday_entry", False))
        max_entry_minutes = 0
        try:
            session_open_et = str(params.get("session_open_et") or "09:30")
            open_time = parse_time_hhmm(session_open_et)
            # Precompute the flatten cutoff (intraday_only) in "minutes from open".
            flatten_minutes_from_open = None
            if intraday_only:
                try:
                    session_close_et = str(params.get("session_close_et") or "16:00")
                    flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)
                    open_dt = dt.datetime.combine(dt.date.today(), open_time)
                    close_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_close_et))
                    flatten_dt = close_dt - dt.timedelta(minutes=max(0, flatten_buffer))
                    flatten_minutes_from_open = int((flatten_dt - open_dt).total_seconds() / 60)
                    flatten_minutes_from_open = max(1, flatten_minutes_from_open)
                except Exception:
                    flatten_minutes_from_open = None

            confirm_pad = confirm_minutes if apply_confirm else 0
            for entry_time_et in entry_times_for_fetch:
                entry_time = parse_time_hhmm(entry_time_et)
                entry_minutes_raw = int(
                    (dt.datetime.combine(dt.date.today(), entry_time) - dt.datetime.combine(dt.date.today(), open_time)).total_seconds()
                    / 60
                )
                entry_minutes_raw = max(0, entry_minutes_raw)

                # Ensure we can reference the last completed bar before the entry timestamp.
                max_entry_minutes = max(max_entry_minutes, max(1, entry_minutes_raw + 1))

                # Confirmation needs bars through (entry + confirm) to evaluate [entry, cutoff).
                if apply_confirm:
                    minutes_needed = max(minutes_needed, max(1, entry_minutes_raw + confirm_pad + 1))

                # Exit simulation needs intraday bars through the effective cutoff (time-stop and/or flatten).
                cutoff_minutes = None
                if time_stop_minutes > 0:
                    cutoff_minutes = entry_minutes_raw + confirm_pad + time_stop_minutes
                if intraday_only and flatten_minutes_from_open is not None:
                    cutoff_minutes = flatten_minutes_from_open if cutoff_minutes is None else min(cutoff_minutes, flatten_minutes_from_open)
                if cutoff_minutes is not None and cutoff_minutes > 0:
                    # +1 to safely include the last completed bar before the cutoff even if the data API treats
                    # end timestamps as exclusive.
                    minutes_needed = max(minutes_needed, cutoff_minutes + 1)
        except Exception:
            max_entry_minutes = max(max_entry_minutes, 1)
        if use_intraday_entry:
            minutes_needed = max(minutes_needed, max_entry_minutes)
        for symbol in symbols:
            signal = generate_signal_for_date(symbol, date_str, cfg, data_store)
            if not signal:
                skip_counts["no_signal"] += 1
                continue
            bars_intraday = None
            if minutes_needed > 0:
                bars_intraday = get_intraday_bars(symbol, date_str, minutes_needed, cfg=cfg, allow_fetch=True)
                if not bars_intraday:
                    skip_counts["no_intraday"] += 1
                    continue
            entry_time_override = symbol_entry_time.get(symbol) or None
            entry_time_cutoff = entry_time_override or (entry_times[0] if entry_times else "09:35")
            if apply_confirm:
                entry_time_cutoff = _add_minutes(entry_time_cutoff, confirm_minutes)
            bars_intraday_entry = bars_intraday
            if bars_intraday and entry_time_cutoff:
                bars_intraday_entry = filter_intraday_bars_until(
                    bars_intraday,
                    date_str,
                    entry_time_cutoff,
                )
                if not bars_intraday_entry:
                    skip_counts["no_intraday_before_cutoff"] += 1
            plan = build_trade(
                signal,
                cfg,
                data_store,
                context="replay",
                bars_intraday=bars_intraday_entry,
                entry_time_override=entry_time_override,
                param_overrides=symbol_overrides.get(symbol) or None,
            )
            if not plan:
                skip_counts["no_plan"] += 1
                if debug_limit > 0 and debug_emitted < debug_limit:
                    debug_emitted += 1
                    first_ts = None
                    last_ts = None
                    first_ts_entry = None
                    last_ts_entry = None
                    try:
                        if bars_intraday:
                            first_ts = bars_intraday[0].get("timestamp")
                            last_ts = bars_intraday[-1].get("timestamp")
                        if bars_intraday_entry:
                            first_ts_entry = bars_intraday_entry[0].get("timestamp")
                            last_ts_entry = bars_intraday_entry[-1].get("timestamp")
                    except Exception:
                        pass
                    logging.info(
                        "[REPLAY_NO_PLAN] date=%s symbol=%s entry_time_override=%s cutoff=%s intraday_bars=%s intraday_first=%s intraday_last=%s entry_bars=%s entry_first=%s entry_last=%s",
                        date_str,
                        symbol,
                        entry_time_override,
                        entry_time_cutoff,
                        (len(bars_intraday) if bars_intraday else 0),
                        first_ts,
                        last_ts,
                        (len(bars_intraday_entry) if bars_intraday_entry else 0),
                        first_ts_entry,
                        last_ts_entry,
                    )
                continue
            # Persist watchlist scoring context used for this symbol so sizing can be quality-aware
            # without introducing lookahead (stats are from the rolling window ending at D-1).
            plan.watchlist_stats = symbol_watchlist_stats.get(symbol) or None
            bars = data_store.get_daily_bars(symbol, None, None, cfg=cfg, allow_fetch=True)
            exit_info = simulate_exit(plan, "daily", bars, bars_intraday, cfg)
            if not exit_info:
                skip_counts["no_exit"] += 1
                continue
            direction_mult = 1.0 if plan.direction == "long" else -1.0
            pnl = (float(exit_info["exit_price"]) - plan.entry_price) * direction_mult
            pnl_pct = (pnl / plan.entry_price) * 100.0
            r_multiple = pnl / plan.stop_distance
            if details_root is not None:
                day_details.append(
                    {
                        "symbol": symbol,
                        "param_overrides": getattr(plan, "param_overrides", None),
                        "signal_date": signal.signal_date,
                        "direction": signal.direction,
                        "trend_state": signal.trend_state,
                        "return_pct": signal.return_pct,
                        "entry_date": plan.entry_date,
                        "entry_time_et": plan.entry_time_et,
                        "entry_price": plan.entry_price,
                        "entry_price_mode": getattr(plan, "entry_price_mode", None),
                        "stop_price": plan.stop_price,
                        "target_price": plan.target_price,
                        "time_exit_date": plan.time_exit_date,
                        "stop_distance": plan.stop_distance,
                        "target_rr": plan.target_rr,
                        "target_mode": getattr(plan, "target_mode", None),
                        "target_window_avg_pct": getattr(plan, "target_window_avg_pct", None),
                        "target_window_mult": getattr(plan, "target_window_mult", None),
                        "target_window_minutes": getattr(plan, "target_window_minutes", None),
                        "target_window_samples": getattr(plan, "target_window_samples", None),
                        "gap_bps": getattr(plan, "gap_bps", None),
                        "early_pullback_bps": getattr(plan, "early_pullback_bps", None),
                        "early_reversal_bps": getattr(plan, "early_reversal_bps", None),
                        "confirm_move_bps": getattr(plan, "confirm_move_bps", None),
                        "confirm_minutes": getattr(plan, "confirm_minutes", None),
                        "confirm_hit_bps": getattr(plan, "confirm_hit_bps", None),
                        "signal_return_pct": getattr(plan, "signal_return_pct", None),
                        "signal_return_atr": getattr(plan, "signal_return_atr", None),
                        "atr": getattr(plan, "atr", None),
                        "watchlist_rank": (plan.watchlist_stats or {}).get("rank") if getattr(plan, "watchlist_stats", None) else None,
                        "watchlist_avgR": (plan.watchlist_stats or {}).get("avgR") if getattr(plan, "watchlist_stats", None) else None,
                        "watchlist_avgR_stderr": (plan.watchlist_stats or {}).get("avgR_stderr")
                        if getattr(plan, "watchlist_stats", None)
                        else None,
                        "watchlist_win_rate": (plan.watchlist_stats or {}).get("win_rate")
                        if getattr(plan, "watchlist_stats", None)
                        else None,
                        "watchlist_profit_factor": (plan.watchlist_stats or {}).get("profit_factor")
                        if getattr(plan, "watchlist_stats", None)
                        else None,
                        "watchlist_trades_count": (plan.watchlist_stats or {}).get("trades_count")
                        if getattr(plan, "watchlist_stats", None)
                        else None,
                        "watchlist_total_pnl_pct": (plan.watchlist_stats or {}).get("total_pnl_pct")
                        if getattr(plan, "watchlist_stats", None)
                        else None,
                        "exit_date": str(exit_info["exit_date"]),
                        "exit_price": float(exit_info["exit_price"]),
                        "exit_reason": str(exit_info["exit_reason"]),
                        "exit_ts": exit_info.get("exit_ts"),
                        "stop_hit_ts": exit_info.get("stop_hit_ts"),
                        "target_hit_ts": exit_info.get("target_hit_ts"),
                        "pnl_pct": pnl_pct,
                        "r_multiple": r_multiple,
                        "mfe_pct": exit_info.get("mfe_pct"),
                        "mae_pct": exit_info.get("mae_pct"),
                        "mfe_r": exit_info.get("mfe_r"),
                        "mae_r": exit_info.get("mae_r"),
                        "mfe_r_full": exit_info.get("mfe_r_full"),
                        "mae_r_full": exit_info.get("mae_r_full"),
                        "mfe_r_before_stop": exit_info.get("mfe_r_before_stop"),
                        "mae_r_to_target": exit_info.get("mae_r_to_target"),
                    }
                )
            trades.append(
                TradeResult(
                    plan=plan,
                    exit_date=str(exit_info["exit_date"]),
                    exit_price=float(exit_info["exit_price"]),
                    exit_reason=str(exit_info["exit_reason"]),
                    pnl_pct=pnl_pct,
                    r_multiple=r_multiple,
                    mfe_pct=exit_info.get("mfe_pct"),
                    mae_pct=exit_info.get("mae_pct"),
                    mfe_r=exit_info.get("mfe_r"),
                    mae_r=exit_info.get("mae_r"),
                    exit_ts=exit_info.get("exit_ts"),
                    stop_hit_ts=exit_info.get("stop_hit_ts"),
                    target_hit_ts=exit_info.get("target_hit_ts"),
                    mfe_r_full=exit_info.get("mfe_r_full"),
                    mae_r_full=exit_info.get("mae_r_full"),
                    mfe_r_before_stop=exit_info.get("mfe_r_before_stop"),
                    mae_r_to_target=exit_info.get("mae_r_to_target"),
                )
            )
            day_trades += 1
        logging.info("[REPLAY] date=%s day_trades=%s total_trades=%s", date_str, day_trades, len(trades))
        logging.info(
            "[REPLAY_SUMMARY] date=%s symbols=%s traded=%s no_signal=%s no_intraday=%s no_intraday_before_cutoff=%s no_plan=%s no_exit=%s",
            date_str,
            len(symbols),
            day_trades,
            skip_counts["no_signal"],
            skip_counts["no_intraday"],
            skip_counts["no_intraday_before_cutoff"],
            skip_counts["no_plan"],
            skip_counts["no_exit"],
        )
        if details_root is not None:
            out_path = details_root / f"{date_str}.json"
            out_path.write_text(json.dumps(day_details, indent=2), encoding="utf-8")
    return trades
