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
from app.strategies.daily_trend_reversal import build_trade, generate_signal_for_date, resolve_entry_times
from app.strategies.types import TradeResult
from app.utils.time import iter_trading_days, parse_time_hhmm
from app.watchlist.day_filter import day_filter_decision
from app.watchlist.prioritization import sort_symbols_by_watchlist_priority
from app.watchlist.storage import read_watchlist

_DEFAULT_RUN_ID: Optional[str] = None


def _watchlist_stats_from_row(row: Dict, rank: int) -> Dict:
    stats = {"rank": int(rank)}
    if not isinstance(row, dict):
        return stats
    excluded = {"symbol", "direction", "entry_time_et", "param_overrides", "reasons"}
    for key, value in row.items():
        if str(key) in excluded:
            continue
        stats[str(key)] = value
    return stats


def _add_minutes(time_str: str, minutes: int) -> str:
    if not time_str or minutes <= 0:
        return time_str
    try:
        base = parse_time_hhmm(time_str)
        shifted = dt.datetime.combine(dt.date.today(), base) + dt.timedelta(minutes=minutes)
        return shifted.strftime("%H:%M")
    except Exception:
        return time_str


def _merged_symbol_params(cfg: Dict, overrides: Optional[Dict]) -> Dict:
    params = dict(cfg.get("daily_trend_reversal") or {})
    if isinstance(overrides, dict) and overrides:
        params.update(overrides)
    return params


def _intraday_minutes_needed(params: Dict, entry_time_et: str) -> int:
    intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
    early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
    max_early_pullback_bps = float(params.get("max_early_pullback_bps") or 0.0)
    min_early_reversal_bps = float(params.get("min_early_reversal_bps") or 0.0)
    requires_early_data = (
        intraday_filter_enabled
        and early_range_minutes > 0
        and (max_early_pullback_bps > 0 or min_early_reversal_bps > 0)
    )

    time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    intraday_only = bool(params.get("intraday_only", False))
    confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
    confirm_minutes = int(params.get("confirm_minutes") or 0)
    apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0
    confirm_pad = confirm_minutes if apply_confirm else 0
    use_intraday_entry = bool(params.get("use_intraday_entry", False))

    minutes_needed = 0
    session_open_et = str(params.get("session_open_et") or "09:30")
    entry_minutes_raw = 0
    try:
        open_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_open_et))
        entry_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(str(entry_time_et or "")))
        entry_minutes_raw = int((entry_dt - open_dt).total_seconds() / 60)
        entry_minutes_raw = max(0, entry_minutes_raw)
    except Exception:
        entry_minutes_raw = 0

    if requires_early_data:
        minutes_needed = max(minutes_needed, early_range_minutes)
    if use_intraday_entry:
        minutes_needed = max(minutes_needed, max(1, entry_minutes_raw + 1))
    if apply_confirm:
        minutes_needed = max(minutes_needed, max(1, entry_minutes_raw + confirm_pad + 1))

    cutoff_minutes = None
    if time_stop_minutes > 0:
        cutoff_minutes = entry_minutes_raw + confirm_pad + time_stop_minutes
    if intraday_only:
        try:
            session_close_et = str(params.get("session_close_et") or "16:00")
            flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)
            open_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_open_et))
            close_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_close_et))
            flatten_dt = close_dt - dt.timedelta(minutes=max(0, flatten_buffer))
            flatten_minutes = max(1, int((flatten_dt - open_dt).total_seconds() / 60))
            cutoff_minutes = flatten_minutes if cutoff_minutes is None else min(cutoff_minutes, flatten_minutes)
        except Exception:
            pass
    if cutoff_minutes is not None and cutoff_minutes > 0:
        minutes_needed = max(minutes_needed, cutoff_minutes + 1)
    return int(max(0, minutes_needed))


def _entry_cutoff_time(params: Dict, entry_time_et: str) -> str:
    confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
    confirm_minutes = int(params.get("confirm_minutes") or 0)
    if confirm_move_bps > 0 and confirm_minutes > 0:
        return _add_minutes(entry_time_et, confirm_minutes)
    return entry_time_et


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
            symbol_watchlist_stats[sym] = _watchlist_stats_from_row(row, idx)
        symbols = list(symbol_entry_time.keys())
        symbols = sort_symbols_by_watchlist_priority(symbols, symbol_watchlist_stats)
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
        watch_cfg = cfg.get("watchlist") or {}
        entry_time_mode = str(watch_cfg.get("entry_time_mode") or "fixed").lower().strip()
        scan_first_valid_mode = entry_time_mode in {"scan_first_valid", "dynamic_first_valid", "scan"}
        entry_time_sort_mode = str(watch_cfg.get("entry_time_sort_mode") or "asc").lower().strip()
        entry_times = resolve_entry_times(params, sort_mode=entry_time_sort_mode)
        for symbol in symbols:
            signal = generate_signal_for_date(symbol, date_str, cfg, data_store)
            if not signal:
                skip_counts["no_signal"] += 1
                continue
            symbol_override = symbol_overrides.get(symbol) or {}
            symbol_params = _merged_symbol_params(cfg, symbol_override)
            entry_time_override = symbol_entry_time.get(symbol) or None
            scan_times_for_symbol = list(entry_times) if scan_first_valid_mode else []
            if scan_first_valid_mode and scan_times_for_symbol and entry_time_override:
                if entry_time_override in scan_times_for_symbol:
                    scan_times_for_symbol = scan_times_for_symbol[scan_times_for_symbol.index(entry_time_override) :]
                else:
                    try:
                        start_t = parse_time_hhmm(str(entry_time_override))
                        filtered = [t for t in scan_times_for_symbol if parse_time_hhmm(t) >= start_t]
                        if filtered:
                            scan_times_for_symbol = filtered
                    except Exception:
                        pass
            if not scan_times_for_symbol and entry_time_override:
                scan_times_for_symbol = [entry_time_override]
            if not scan_times_for_symbol:
                scan_times_for_symbol = [str(symbol_params.get("entry_time_et") or "09:35")]
            minutes_needed = 0
            for t in scan_times_for_symbol:
                minutes_needed = max(minutes_needed, _intraday_minutes_needed(symbol_params, t))
            bars_intraday = None
            if minutes_needed > 0:
                bars_intraday = get_intraday_bars(symbol, date_str, minutes_needed, cfg=cfg, allow_fetch=True)
                if not bars_intraday:
                    skip_counts["no_intraday"] += 1
                    continue
            plan = None
            first_missing_intraday_cutoff = False
            intraday_cutoff_cache: Dict[str, List[Dict]] = {}
            if scan_first_valid_mode:
                for scan_time in scan_times_for_symbol:
                    entry_time_cutoff = _entry_cutoff_time(symbol_params, scan_time)
                    bars_intraday_entry = bars_intraday
                    if bars_intraday and entry_time_cutoff:
                        if entry_time_cutoff not in intraday_cutoff_cache:
                            intraday_cutoff_cache[entry_time_cutoff] = filter_intraday_bars_until(
                                bars_intraday,
                                date_str,
                                entry_time_cutoff,
                            )
                        bars_intraday_entry = intraday_cutoff_cache.get(entry_time_cutoff) or []
                        if not bars_intraday_entry:
                            first_missing_intraday_cutoff = True
                            continue
                    plan = build_trade(
                        signal,
                        cfg,
                        data_store,
                        context="replay",
                        bars_intraday=bars_intraday_entry,
                        entry_time_override=scan_time,
                        param_overrides=symbol_override or None,
                    )
                    if plan:
                        break
            else:
                entry_time_for_symbol = (
                    entry_time_override
                    or (entry_times[0] if entry_times else str(symbol_params.get("entry_time_et") or "09:35"))
                )
                entry_time_cutoff = _entry_cutoff_time(symbol_params, entry_time_for_symbol)
                bars_intraday_entry = bars_intraday
                if bars_intraday and entry_time_cutoff:
                    bars_intraday_entry = filter_intraday_bars_until(
                        bars_intraday,
                        date_str,
                        entry_time_cutoff,
                    )
                    if not bars_intraday_entry:
                        first_missing_intraday_cutoff = True
                plan = build_trade(
                    signal,
                    cfg,
                    data_store,
                    context="replay",
                    bars_intraday=bars_intraday_entry,
                    entry_time_override=entry_time_override,
                    param_overrides=symbol_override or None,
                )
            if first_missing_intraday_cutoff and plan is None:
                skip_counts["no_intraday_before_cutoff"] += 1
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
                        if locals().get("bars_intraday_entry"):
                            first_ts_entry = bars_intraday_entry[0].get("timestamp")
                            last_ts_entry = bars_intraday_entry[-1].get("timestamp")
                    except Exception:
                        pass
                    logging.info(
                        "[REPLAY_NO_PLAN] date=%s symbol=%s entry_time_override=%s cutoff=%s intraday_bars=%s intraday_first=%s intraday_last=%s entry_bars=%s entry_first=%s entry_last=%s",
                        date_str,
                        symbol,
                        (entry_time_override if not scan_first_valid_mode else "scan_first_valid"),
                        (entry_time_cutoff if "entry_time_cutoff" in locals() else None),
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
                wl_stats = (
                    plan.watchlist_stats
                    if isinstance(getattr(plan, "watchlist_stats", None), dict)
                    else {}
                )
                detail = {
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
                    "watchlist_rank": wl_stats.get("rank"),
                    "watchlist_avgR": wl_stats.get("avgR"),
                    "watchlist_avgR_stderr": wl_stats.get("avgR_stderr"),
                    "watchlist_win_rate": wl_stats.get("win_rate"),
                    "watchlist_profit_factor": wl_stats.get("profit_factor"),
                    "watchlist_trades_count": wl_stats.get("trades_count"),
                    "watchlist_total_pnl_pct": wl_stats.get("total_pnl_pct"),
                    "watchlist_stats": wl_stats,
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
                for key, value in wl_stats.items():
                    prefixed = f"watchlist_{str(key)}"
                    if prefixed not in detail:
                        detail[prefixed] = value
                day_details.append(detail)
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
