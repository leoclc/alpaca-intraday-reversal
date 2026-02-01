from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.data.alpaca_intraday_store import get_intraday_bars
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.execution.daily_execution_model import simulate_exit
from app.strategies.daily_trend_reversal import build_trade, generate_signals
from app.strategies.types import TradeResult
from app.utils.time import iter_trading_days
from app.watchlist.storage import read_watchlist

_DEFAULT_RUN_ID: Optional[str] = None


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
        wl = read_watchlist(date_str, cfg)
        symbols = [str(r.get("symbol") or "").upper() for r in wl.get("watchlist") or [] if r.get("symbol")]
        day_trades = 0
        day_details: List[Dict] = []
        if not symbols:
            logging.info("[REPLAY] date=%s watchlist empty or missing", date_str)
            continue
        params = cfg.get("daily_trend_reversal") or {}
        intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
        early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
        time_stop_minutes = int(params.get("time_stop_minutes") or 0)
        minutes_needed = max(early_range_minutes, time_stop_minutes)
        for symbol in symbols:
            signals = generate_signals([symbol], date_str, date_str, cfg, data_store)
            if not signals:
                continue
            signal = signals[0]
            bars_intraday = None
            if minutes_needed > 0:
                bars_intraday = get_intraday_bars(symbol, date_str, minutes_needed, cfg=cfg, allow_fetch=True)
            plan = build_trade(signal, cfg, data_store, context="replay", bars_intraday=bars_intraday)
            if not plan:
                continue
            bars = data_store.get_daily_bars(symbol, None, None, cfg=cfg, allow_fetch=True)
            exit_info = simulate_exit(plan, "daily", bars, bars_intraday, cfg)
            if not exit_info:
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
                        "stop_price": plan.stop_price,
                        "target_price": plan.target_price,
                        "time_exit_date": plan.time_exit_date,
                        "stop_distance": plan.stop_distance,
                        "target_rr": plan.target_rr,
                        "exit_date": str(exit_info["exit_date"]),
                        "exit_price": float(exit_info["exit_price"]),
                        "exit_reason": str(exit_info["exit_reason"]),
                        "pnl_pct": pnl_pct,
                        "r_multiple": r_multiple,
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
                )
            )
            day_trades += 1
        logging.info("[REPLAY] date=%s day_trades=%s total_trades=%s", date_str, day_trades, len(trades))
        if details_root is not None:
            out_path = details_root / f"{date_str}.json"
            out_path.write_text(json.dumps(day_details, indent=2), encoding="utf-8")
    return trades
