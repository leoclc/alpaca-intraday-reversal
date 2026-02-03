from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from app.utils.time import ensure_date, ensure_et, parse_time_hhmm


def _find_bar_index(bars: List[Dict[str, Any]], date_str: str) -> Optional[int]:
    for idx, bar in enumerate(bars):
        if str(bar.get("date")) == date_str:
            return idx
    return None


def _parse_intraday_ts(bar: Dict[str, Any]) -> Optional[dt.datetime]:
    for key in ("timestamp", "datetime", "time", "date"):
        if key in bar and bar[key]:
            try:
                return dt.datetime.fromisoformat(str(bar[key]))
            except Exception:
                continue
    return None


def compute_sma(bars: List[Dict[str, Any]], period: int, end_index: int) -> Optional[float]:
    if period <= 0 or end_index < 0:
        return None
    start = end_index - period + 1
    if start < 0:
        return None
    try:
        closes = [float(bars[i]["close"]) for i in range(start, end_index + 1)]
    except Exception:
        return None
    if not closes:
        return None
    return sum(closes) / float(len(closes))


def compute_atr_daily(bars: List[Dict[str, Any]], period: int, end_index: int) -> Optional[float]:
    if period <= 0 or end_index <= 0:
        return None
    start = end_index - period + 1
    if start < 1:
        return None
    trs: List[float] = []
    for i in range(start, end_index + 1):
        try:
            high = float(bars[i]["high"])
            low = float(bars[i]["low"])
            prev_close = float(bars[i - 1]["close"])
        except Exception:
            return None
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return None
    return sum(trs) / float(len(trs))


def simulate_entry(
    signal,
    entry_time_et: str,
    mode: str,
    bars_daily: List[Dict[str, Any]],
    bars_intraday: Optional[List[Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    entry_date = str(signal.signal_date)
    idx = _find_bar_index(bars_daily, entry_date)
    source_date = entry_date
    bar = None
    if idx is None:
        for i, row in enumerate(bars_daily):
            if str(row.get("date")) < entry_date:
                idx = i
                bar = row
        if idx is None or bar is None:
            return None
        source_date = str(bar.get("date"))
    else:
        bar = bars_daily[idx]
    entry_price_mode = str((cfg.get("daily_trend_reversal") or {}).get("entry_price_mode") or "open").lower()
    use_intraday_entry = bool((cfg.get("daily_trend_reversal") or {}).get("use_intraday_entry", False))
    entry_price: Optional[float] = None
    if use_intraday_entry and bars_intraday:
        entry_time = parse_time_hhmm(entry_time_et)
        entry_dt = ensure_et(dt.datetime.combine(ensure_date(entry_date), entry_time))
        for row in bars_intraday:
            ts = _parse_intraday_ts(row)
            if not ts:
                continue
            ts = ensure_et(ts)
            if ts >= entry_dt:
                if entry_price_mode == "close":
                    entry_price = float(row.get("close") or row.get("c") or 0.0)
                else:
                    entry_price = float(row.get("open") or row.get("o") or 0.0)
                break
    if entry_price is None and source_date == entry_date:
        if entry_price_mode == "close":
            entry_price = float(bar["close"])
        else:
            entry_price = float(bar["open"])
    elif entry_price is None:
        # No daily bar for entry_date yet (live). Prefer intraday price if available.
        if bars_intraday:
            entry_time = parse_time_hhmm(entry_time_et)
            entry_dt = ensure_et(dt.datetime.combine(ensure_date(entry_date), entry_time))
            for row in bars_intraday:
                ts = _parse_intraday_ts(row)
                if not ts:
                    continue
                ts = ensure_et(ts)
                if ts >= entry_dt:
                    if entry_price_mode == "close":
                        entry_price = float(row.get("close") or row.get("c") or 0.0)
                    else:
                        entry_price = float(row.get("open") or row.get("o") or 0.0)
                    break
        if entry_price is None:
            # Fallback to previous close
            entry_price = float(bar["close"])
    entry_time = parse_time_hhmm(entry_time_et)
    entry_dt = ensure_date(entry_date)
    return {
        "entry_price": entry_price,
        "entry_date": entry_date,
        "entry_source_date": source_date,
        "entry_time_et": entry_time,
        "entry_index": idx,
        "entry_dt": entry_dt,
    }


def simulate_exit(
    trade_plan,
    mode: str,
    bars_daily: List[Dict[str, Any]],
    bars_intraday: Optional[List[Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    entry_idx = _find_bar_index(bars_daily, trade_plan.entry_date)
    if entry_idx is None:
        return None
    params = cfg.get("daily_trend_reversal") or {}
    time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    if bars_intraday and time_stop_minutes > 0:
        entry_time = parse_time_hhmm(trade_plan.entry_time_et)
        entry_dt = ensure_et(dt.datetime.combine(ensure_date(trade_plan.entry_date), entry_time))
        cutoff = entry_dt + dt.timedelta(minutes=time_stop_minutes)
        stop_first = bool(params.get("stop_first_when_both", True))
        direction = str(trade_plan.direction).lower()
        last_bar = None
        for bar in bars_intraday:
            ts = _parse_intraday_ts(bar)
            if not ts:
                continue
            ts = ensure_et(ts)
            if ts < entry_dt:
                continue
            if ts > cutoff:
                break
            last_bar = bar
            high = float(bar.get("high") or bar.get("h") or bar.get("High") or 0.0)
            low = float(bar.get("low") or bar.get("l") or bar.get("Low") or 0.0)
            if direction == "long":
                hit_stop = low <= trade_plan.stop_price
                hit_target = high >= trade_plan.target_price
            else:
                hit_stop = high >= trade_plan.stop_price
                hit_target = low <= trade_plan.target_price
            if hit_stop and hit_target:
                reason = "stop" if stop_first else "target"
                price = trade_plan.stop_price if stop_first else trade_plan.target_price
                return {"exit_date": trade_plan.entry_date, "exit_price": price, "exit_reason": reason}
            if hit_stop:
                return {"exit_date": trade_plan.entry_date, "exit_price": trade_plan.stop_price, "exit_reason": "stop"}
            if hit_target:
                return {
                    "exit_date": trade_plan.entry_date,
                    "exit_price": trade_plan.target_price,
                    "exit_reason": "target",
                }
        if last_bar:
            close_price = float(last_bar.get("close") or last_bar.get("c") or last_bar.get("Close") or 0.0)
            return {"exit_date": trade_plan.entry_date, "exit_price": close_price, "exit_reason": "time_stop"}
    intraday_only = bool((cfg.get("daily_trend_reversal") or {}).get("intraday_only", False))
    if intraday_only:
        exit_idx = entry_idx
    else:
        exit_idx = _find_bar_index(bars_daily, trade_plan.time_exit_date)
        if exit_idx is None:
            exit_idx = len(bars_daily) - 1
    stop_first = bool((cfg.get("daily_trend_reversal") or {}).get("stop_first_when_both", True))
    direction = str(trade_plan.direction).lower()
    for i in range(entry_idx, exit_idx + 1):
        bar = bars_daily[i]
        high = float(bar["high"])
        low = float(bar["low"])
        if direction == "long":
            hit_stop = low <= trade_plan.stop_price
            hit_target = high >= trade_plan.target_price
            if hit_stop and hit_target:
                reason = "stop" if stop_first else "target"
                price = trade_plan.stop_price if stop_first else trade_plan.target_price
                return {"exit_date": bar["date"], "exit_price": price, "exit_reason": reason}
            if hit_stop:
                return {"exit_date": bar["date"], "exit_price": trade_plan.stop_price, "exit_reason": "stop"}
            if hit_target:
                return {"exit_date": bar["date"], "exit_price": trade_plan.target_price, "exit_reason": "target"}
        else:
            hit_stop = high >= trade_plan.stop_price
            hit_target = low <= trade_plan.target_price
            if hit_stop and hit_target:
                reason = "stop" if stop_first else "target"
                price = trade_plan.stop_price if stop_first else trade_plan.target_price
                return {"exit_date": bar["date"], "exit_price": price, "exit_reason": reason}
            if hit_stop:
                return {"exit_date": bar["date"], "exit_price": trade_plan.stop_price, "exit_reason": "stop"}
            if hit_target:
                return {"exit_date": bar["date"], "exit_price": trade_plan.target_price, "exit_reason": "target"}
    last_bar = bars_daily[exit_idx]
    return {
        "exit_date": last_bar["date"],
        "exit_price": float(last_bar["close"]),
        "exit_reason": "eod_flat" if intraday_only else "time_exit",
    }
