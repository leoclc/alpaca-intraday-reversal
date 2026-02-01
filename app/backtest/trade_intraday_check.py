from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.loader import load_config
from app.data.alpaca_intraday_store import get_intraday_bars
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.execution.daily_execution_model import simulate_exit
from app.strategies.types import TradePlan
from app.utils.time import ensure_date, parse_time_hhmm


def _session_minutes(cfg: Dict) -> int:
    params = cfg.get("daily_trend_reversal") or {}
    open_time = parse_time_hhmm(str(params.get("session_open_et") or "09:30"))
    close_time = parse_time_hhmm(str(params.get("session_close_et") or "16:00"))
    base = dt.date.today()
    start = dt.datetime.combine(base, open_time)
    end = dt.datetime.combine(base, close_time)
    minutes = int((end - start).total_seconds() / 60)
    return max(1, minutes)


def _minutes_needed(cfg: Dict) -> int:
    params = cfg.get("daily_trend_reversal") or {}
    time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    if time_stop_minutes > 0:
        return time_stop_minutes
    return _session_minutes(cfg)


def _build_trade_plan(plan_dict: Dict[str, Any]) -> Optional[TradePlan]:
    try:
        return TradePlan(
            symbol=str(plan_dict.get("symbol") or ""),
            direction=str(plan_dict.get("direction") or ""),
            signal_date=str(plan_dict.get("signal_date") or plan_dict.get("entry_date") or ""),
            entry_date=str(plan_dict.get("entry_date") or ""),
            entry_time_et=str(plan_dict.get("entry_time_et") or "09:35"),
            entry_price=float(plan_dict.get("entry_price") or 0.0),
            stop_price=float(plan_dict.get("stop_price") or 0.0),
            target_price=float(plan_dict.get("target_price") or 0.0),
            time_exit_date=str(plan_dict.get("time_exit_date") or plan_dict.get("entry_date") or ""),
            stop_distance=float(plan_dict.get("stop_distance") or 0.0),
            target_rr=float(plan_dict.get("target_rr") or 0.0),
        )
    except Exception:
        return None


def run_intraday_check(
    backtest_path: str,
    cfg: Optional[Dict] = None,
    out_path: Optional[str] = None,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    payload = json.loads(Path(backtest_path).read_text(encoding="utf-8"))
    trades = payload.get("trades") or []
    params = cfg.get("daily_trend_reversal") or {}
    minutes_needed = _minutes_needed(cfg)

    data_store = AlpacaOHLCStore(cfg=cfg)
    mismatches: List[Dict[str, Any]] = []
    missing_intraday: List[Dict[str, Any]] = []
    matched = 0
    checked = 0
    reason_mismatch = 0
    price_mismatch = 0

    for trade in trades:
        plan_dict = trade.get("plan") or {}
        plan = _build_trade_plan(plan_dict)
        if not plan or not plan.symbol or not plan.entry_date:
            continue
        entry_date = ensure_date(plan.entry_date).isoformat()
        bars_intraday = get_intraday_bars(plan.symbol, entry_date, minutes_needed, cfg=cfg, allow_fetch=True)
        if not bars_intraday:
            missing_intraday.append({"symbol": plan.symbol, "date": entry_date})
            continue
        bars_daily = data_store.get_daily_bars(plan.symbol, None, None, cfg=cfg, allow_fetch=True)
        if not bars_daily:
            missing_intraday.append({"symbol": plan.symbol, "date": entry_date, "reason": "missing_daily"})
            continue
        exit_info = simulate_exit(plan, "daily", bars_daily, bars_intraday, cfg)
        if not exit_info:
            continue
        checked += 1
        actual_reason = str(trade.get("exit_reason") or "")
        actual_price = float(trade.get("exit_price") or 0.0)
        new_reason = str(exit_info.get("exit_reason") or "")
        new_price = float(exit_info.get("exit_price") or 0.0)
        same_reason = actual_reason == new_reason
        same_price = abs(actual_price - new_price) <= 1e-6
        if same_reason and same_price:
            matched += 1
            continue
        if not same_reason:
            reason_mismatch += 1
        if not same_price:
            price_mismatch += 1
        mismatches.append(
            {
                "symbol": plan.symbol,
                "date": entry_date,
                "actual_reason": actual_reason,
                "actual_price": actual_price,
                "recalc_reason": new_reason,
                "recalc_price": new_price,
            }
        )
        if sample_limit and len(mismatches) >= sample_limit:
            break

    summary = {
        "total_trades": len(trades),
        "checked": checked,
        "matched": matched,
        "reason_mismatch": reason_mismatch,
        "price_mismatch": price_mismatch,
        "missing_intraday": len(missing_intraday),
        "minutes_checked": minutes_needed,
        "time_stop_minutes": int(params.get("time_stop_minutes") or 0),
    }
    result = {"summary": summary, "mismatches": mismatches, "missing_intraday": missing_intraday}
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify backtest exits using intraday minute bars.")
    parser.add_argument("backtest_path", help="Path to backtest.json")
    parser.add_argument("--out", dest="out_path", default=None, help="Write results to JSON file")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit mismatch rows")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    run_intraday_check(args.backtest_path, load_config(), out_path=args.out_path, sample_limit=args.sample_limit)
