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
from app.strategies.daily_trend_reversal import build_trade, generate_signals
from app.strategies.types import TradeResult
from app.utils.time import iter_trading_days, parse_time_hhmm
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
        intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
        early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
        time_stop_minutes = int(params.get("time_stop_minutes") or 0)
        confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
        confirm_minutes = int(params.get("confirm_minutes") or 0)
        apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0
        minutes_needed = 0
        if early_range_minutes > 0:
            minutes_needed = max(minutes_needed, early_range_minutes)
        max_entry_minutes = 0
        try:
            session_open_et = str(params.get("session_open_et") or "09:30")
            open_time = parse_time_hhmm(session_open_et)
            for entry_time_et in entry_times:
                entry_time = parse_time_hhmm(entry_time_et)
                entry_minutes = int(
                    (dt.datetime.combine(dt.date.today(), entry_time) - dt.datetime.combine(dt.date.today(), open_time)).total_seconds()
                    / 60
                )
                entry_minutes = max(1, entry_minutes + 1)
                max_entry_minutes = max(max_entry_minutes, entry_minutes)
        except Exception:
            max_entry_minutes = max(max_entry_minutes, 1)
        if bool(params.get("use_intraday_entry", False)):
            minutes_needed = max(minutes_needed, max_entry_minutes)
        if time_stop_minutes > 0:
            minutes_needed = max(minutes_needed, max_entry_minutes + time_stop_minutes)
        if apply_confirm:
            minutes_needed = max(minutes_needed, max_entry_minutes + confirm_minutes)
        for symbol in symbols:
            signals = generate_signals([symbol], date_str, date_str, cfg, data_store)
            if not signals:
                skip_counts["no_signal"] += 1
                continue
            signal = signals[0]
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
                        "confirm_move_bps": getattr(plan, "confirm_move_bps", None),
                        "confirm_minutes": getattr(plan, "confirm_minutes", None),
                        "confirm_hit_bps": getattr(plan, "confirm_hit_bps", None),
                        "signal_return_pct": getattr(plan, "signal_return_pct", None),
                        "signal_return_atr": getattr(plan, "signal_return_atr", None),
                        "atr": getattr(plan, "atr", None),
                        "exit_date": str(exit_info["exit_date"]),
                        "exit_price": float(exit_info["exit_price"]),
                        "exit_reason": str(exit_info["exit_reason"]),
                        "pnl_pct": pnl_pct,
                        "r_multiple": r_multiple,
                        "mfe_pct": exit_info.get("mfe_pct"),
                        "mae_pct": exit_info.get("mae_pct"),
                        "mfe_r": exit_info.get("mfe_r"),
                        "mae_r": exit_info.get("mae_r"),
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
