from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from app.utils.time import ensure_date, ensure_et, parse_time_hhmm

_SMA_CACHE: Dict[tuple, Optional[float]] = {}
_ATR_CACHE: Dict[tuple, Optional[float]] = {}


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
    cache_key = (id(bars), int(period), int(end_index))
    if cache_key in _SMA_CACHE:
        return _SMA_CACHE[cache_key]
    if period <= 0 or end_index < 0:
        _SMA_CACHE[cache_key] = None
        return None
    start = end_index - period + 1
    if start < 0:
        _SMA_CACHE[cache_key] = None
        return None
    try:
        closes = [float(bars[i]["close"]) for i in range(start, end_index + 1)]
    except Exception:
        _SMA_CACHE[cache_key] = None
        return None
    if not closes:
        _SMA_CACHE[cache_key] = None
        return None
    val = sum(closes) / float(len(closes))
    _SMA_CACHE[cache_key] = val
    return val


def compute_atr_daily(bars: List[Dict[str, Any]], period: int, end_index: int) -> Optional[float]:
    cache_key = (id(bars), int(period), int(end_index))
    if cache_key in _ATR_CACHE:
        return _ATR_CACHE[cache_key]
    if period <= 0 or end_index <= 0:
        _ATR_CACHE[cache_key] = None
        return None
    start = end_index - period + 1
    if start < 1:
        _ATR_CACHE[cache_key] = None
        return None
    trs: List[float] = []
    for i in range(start, end_index + 1):
        try:
            high = float(bars[i]["high"])
            low = float(bars[i]["low"])
            prev_close = float(bars[i - 1]["close"])
        except Exception:
            _ATR_CACHE[cache_key] = None
            return None
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        _ATR_CACHE[cache_key] = None
        return None
    val = sum(trs) / float(len(trs))
    _ATR_CACHE[cache_key] = val
    return val


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
    intraday_only = bool(params.get("intraday_only", False))
    time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    require_intraday_exit = bool(params.get("require_intraday_exit", intraday_only or time_stop_minutes > 0))

    entry_time = parse_time_hhmm(trade_plan.entry_time_et)
    entry_dt = ensure_et(dt.datetime.combine(ensure_date(trade_plan.entry_date), entry_time))

    time_stop_cutoff: Optional[dt.datetime] = None
    if time_stop_minutes > 0:
        time_stop_cutoff = entry_dt + dt.timedelta(minutes=time_stop_minutes)

    flatten_dt: Optional[dt.datetime] = None
    if intraday_only:
        session_close_et = str(params.get("session_close_et") or "16:00")
        flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)
        close_dt = ensure_et(dt.datetime.combine(ensure_date(trade_plan.entry_date), parse_time_hhmm(session_close_et)))
        flatten_dt = close_dt - dt.timedelta(minutes=max(0, flatten_buffer))

    cutoff: Optional[dt.datetime] = time_stop_cutoff
    if flatten_dt is not None:
        cutoff = flatten_dt if cutoff is None else min(cutoff, flatten_dt)

    # Intraday exit simulation: resolves stop/target ordering without falling back to daily OHLC.
    # If intraday is required but unavailable, skip the trade (no "guessing" from daily bars).
    if cutoff is not None and require_intraday_exit:
        if not bars_intraday:
            return None
        stop_first = bool(params.get("stop_first_when_both", True))
        direction = str(trade_plan.direction).lower()
        mfe_val = 0.0
        mae_val = 0.0
        last_bar = None
        last_ts = None
        for bar in bars_intraday:
            ts = _parse_intraday_ts(bar)
            if not ts:
                continue
            ts = ensure_et(ts)
            if ts < entry_dt:
                continue
            # Strict parity: at the cutoff timestamp you do not yet have the in-progress bar.
            if ts >= cutoff:
                break
            last_bar = bar
            last_ts = ts
            high = float(bar.get("high") or bar.get("h") or bar.get("High") or 0.0)
            low = float(bar.get("low") or bar.get("l") or bar.get("Low") or 0.0)
            if direction == "long":
                mfe_val = max(mfe_val, high - trade_plan.entry_price)
                mae_val = min(mae_val, low - trade_plan.entry_price)
                hit_stop = low <= trade_plan.stop_price
                hit_target = high >= trade_plan.target_price
            else:
                mfe_val = max(mfe_val, trade_plan.entry_price - low)
                mae_val = min(mae_val, trade_plan.entry_price - high)
                hit_stop = high >= trade_plan.stop_price
                hit_target = low <= trade_plan.target_price
            mfe_pct = (mfe_val / trade_plan.entry_price) * 100.0 if trade_plan.entry_price else None
            mae_pct = (mae_val / trade_plan.entry_price) * 100.0 if trade_plan.entry_price else None
            mfe_r = (mfe_val / trade_plan.stop_distance) if trade_plan.stop_distance else None
            mae_r = (mae_val / trade_plan.stop_distance) if trade_plan.stop_distance else None
            if hit_stop and hit_target:
                reason = "stop" if stop_first else "target"
                price = trade_plan.stop_price if stop_first else trade_plan.target_price
                return {
                    "exit_date": trade_plan.entry_date,
                    "exit_price": price,
                    "exit_reason": reason,
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "exit_ts": last_ts.isoformat() if last_ts else None,
                }
            if hit_stop:
                return {
                    "exit_date": trade_plan.entry_date,
                    "exit_price": trade_plan.stop_price,
                    "exit_reason": "stop",
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "exit_ts": last_ts.isoformat() if last_ts else None,
                }
            if hit_target:
                return {
                    "exit_date": trade_plan.entry_date,
                    "exit_price": trade_plan.target_price,
                    "exit_reason": "target",
                    "mfe_pct": mfe_pct,
                    "mae_pct": mae_pct,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "exit_ts": last_ts.isoformat() if last_ts else None,
                }
        if not last_bar:
            return None
        close_price = float(last_bar.get("close") or last_bar.get("c") or last_bar.get("Close") or 0.0)
        mfe_pct = (mfe_val / trade_plan.entry_price) * 100.0 if trade_plan.entry_price else None
        mae_pct = (mae_val / trade_plan.entry_price) * 100.0 if trade_plan.entry_price else None
        mfe_r = (mfe_val / trade_plan.stop_distance) if trade_plan.stop_distance else None
        mae_r = (mae_val / trade_plan.stop_distance) if trade_plan.stop_distance else None
        exit_reason = "time_stop"
        if intraday_only and flatten_dt is not None and cutoff == flatten_dt:
            if time_stop_cutoff is None or flatten_dt <= time_stop_cutoff:
                exit_reason = "eod_flat"
        return {
            "exit_date": trade_plan.entry_date,
            "exit_price": close_price,
            "exit_reason": exit_reason,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "exit_ts": last_ts.isoformat() if last_ts else None,
        }

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
