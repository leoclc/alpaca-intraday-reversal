from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional

from app.data.alpaca_intraday_store import get_intraday_bars
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.execution.daily_execution_model import compute_atr_daily, compute_sma, simulate_entry
from app.strategies.types import Signal, TradePlan
from app.utils.time import ensure_date, ensure_et, parse_time_hhmm

_TARGET_CACHE: Dict[tuple, Optional[Dict[str, float]]] = {}
_TARGET_DAY_CACHE: Dict[tuple, Dict[str, Optional[float]]] = {}


def _prior_trading_days(end_date: dt.date, count: int) -> List[dt.date]:
    days: List[dt.date] = []
    cur = end_date - dt.timedelta(days=1)
    while len(days) < count and cur.year >= 1970:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= dt.timedelta(days=1)
    return days


def _entry_price_mode_base(entry_price_mode_used: Optional[str]) -> str:
    mode = str(entry_price_mode_used or "open").lower()
    if "close" in mode:
        return "close"
    return "open"


def _minutes_needed_for_window(entry_time_et: str, window_minutes: int, session_open_et: str) -> int:
    try:
        entry_time = parse_time_hhmm(entry_time_et)
        open_time = parse_time_hhmm(session_open_et)
        entry_minutes = int(
            (dt.datetime.combine(dt.date.today(), entry_time) - dt.datetime.combine(dt.date.today(), open_time)).total_seconds()
            / 60
        )
        entry_minutes = max(0, entry_minutes)
    except Exception:
        entry_minutes = 0
    return max(1, entry_minutes + max(1, window_minutes))


def _compute_symbol_window_mfe_for_day(
    symbol: str,
    day: dt.date,
    entry_time_et: str,
    direction: str,
    entry_price_mode_used: str,
    window_minutes: int,
    cfg: Dict,
) -> Optional[float]:
    session_open_et = str((cfg.get("daily_trend_reversal") or {}).get("session_open_et") or "09:30")
    minutes_needed = _minutes_needed_for_window(entry_time_et, window_minutes, session_open_et)
    bars = get_intraday_bars(symbol, day, minutes_needed, cfg=cfg, allow_fetch=True)
    if not bars:
        return None

    entry_price = None
    price_mode = _entry_price_mode_base(entry_price_mode_used)
    entry_dt = ensure_et(dt.datetime.combine(day, parse_time_hhmm(entry_time_et)))
    window_end = entry_dt + dt.timedelta(minutes=window_minutes)
    max_high = None
    min_low = None
    for row in bars:
        ts_raw = row.get("timestamp") or row.get("datetime") or row.get("time") or row.get("date")
        if not ts_raw:
            continue
        try:
            ts = ensure_et(dt.datetime.fromisoformat(str(ts_raw)))
        except Exception:
            continue
        if ts < entry_dt:
            continue
        if ts > window_end:
            break
        if entry_price is None:
            if price_mode == "close":
                entry_price = float(row.get("close") or row.get("c") or 0.0)
            else:
                entry_price = float(row.get("open") or row.get("o") or 0.0)
        high = float(row.get("high") or row.get("h") or 0.0)
        low = float(row.get("low") or row.get("l") or 0.0)
        if max_high is None or high > max_high:
            max_high = high
        if min_low is None or low < min_low:
            min_low = low

    if entry_price is None or entry_price <= 0 or max_high is None or min_low is None:
        return None
    if direction == "long":
        return (max_high - entry_price) / entry_price * 100.0
    return (entry_price - min_low) / entry_price * 100.0


def _compute_symbol_window_avg(
    symbol: str,
    signal_date: str,
    entry_time_et: str,
    direction: str,
    entry_price_mode_used: str,
    window_minutes: int,
    lookback_days: int,
    min_samples: int,
    cfg: Dict,
) -> Optional[Dict[str, float]]:
    if window_minutes <= 0 or lookback_days <= 0 or min_samples <= 0:
        return None
    symbol_u = str(symbol).upper()
    direction_u = str(direction).lower()
    entry_mode = str(entry_price_mode_used)
    cache_key = (
        symbol_u,
        str(signal_date),
        str(entry_time_et),
        direction_u,
        entry_mode,
        int(window_minutes),
        int(lookback_days),
    )
    if cache_key in _TARGET_CACHE:
        return _TARGET_CACHE[cache_key]
    target_date = ensure_date(signal_date)
    days = _prior_trading_days(target_date, lookback_days)
    if not days:
        _TARGET_CACHE[cache_key] = None
        return None
    session_open_et = str((cfg.get("daily_trend_reversal") or {}).get("session_open_et") or "09:30")
    day_cache_key = (
        symbol_u,
        str(entry_time_et),
        direction_u,
        _entry_price_mode_base(entry_mode),
        int(window_minutes),
        session_open_et,
    )
    day_cache = _TARGET_DAY_CACHE.setdefault(day_cache_key, {})
    samples: List[float] = []
    for day in days:
        day_str = day.isoformat()
        if day_str not in day_cache:
            day_cache[day_str] = _compute_symbol_window_mfe_for_day(
                symbol_u,
                day,
                entry_time_et,
                direction_u,
                entry_mode,
                window_minutes,
                cfg,
            )
        mfe = day_cache.get(day_str)
        if mfe is None:
            continue
        if mfe > 0:
            samples.append(mfe)
    if len(samples) < min_samples:
        _TARGET_CACHE[cache_key] = None
        return None
    avg = sum(samples) / float(len(samples))
    result = {"avg_mfe_pct": avg, "samples": float(len(samples))}
    _TARGET_CACHE[cache_key] = result
    return result


def generate_signals(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    cfg: Dict,
    data_store: AlpacaOHLCStore,
) -> List[Signal]:
    out: List[Signal] = []
    symbols_list = [str(s).upper() for s in symbols if s]
    if not symbols_list:
        return out
    start = ensure_date(start_date)
    end = ensure_date(end_date)
    params = cfg.get("daily_trend_reversal") or {}
    trend_ma_days = int(params.get("trend_ma_days") or 200)
    reversal_mode = str(params.get("reversal_mode") or "").lower().strip()
    reversal_threshold_pct = float(params.get("reversal_threshold_pct") or 0.0)
    reversal_threshold_atr = float(params.get("reversal_threshold_atr") or 0.0)
    atr_period = int(params.get("atr_period") or 14)
    reversal_lookback_days = int(params.get("reversal_lookback_days") or 1)
    reversal_quantile = float(params.get("reversal_quantile") or 0.2)
    reversal_quantile_lookback_days = int(params.get("reversal_quantile_lookback_days") or 60)
    trend_fast_len = int(params.get("trend_fast_len") or 0)
    trend_slow_len = int(params.get("trend_slow_len") or 0)
    trend_min_slope_bps = float(params.get("trend_min_slope_bps") or 0.0)
    trend_min_distance_atr = float(params.get("trend_min_distance_atr") or 0.0)
    volume_lookback_days = int(params.get("volume_lookback_days") or 0)
    volume_min_ratio = float(params.get("volume_min_ratio") or 0.0)
    if not reversal_mode:
        reversal_mode = "atr" if reversal_threshold_atr > 0 else "pct"
    for symbol in symbols_list:
        bars = data_store.get_daily_bars(symbol, None, None, cfg=cfg, allow_fetch=True)
        if not bars or len(bars) < trend_ma_days + 2:
            continue
        for idx, bar in enumerate(bars):
            bar_date = ensure_date(str(bar.get("date")))
            if bar_date < start or bar_date > end:
                continue
            prev_idx = idx - 1
            prev2_idx = idx - 2
            if prev_idx < trend_ma_days - 1 or prev2_idx < 0:
                continue
            sma = compute_sma(bars, trend_ma_days, prev_idx)
            if sma is None:
                continue
            close_prev = float(bars[prev_idx]["close"])
            trend_state = "uptrend" if close_prev > sma else "downtrend" if close_prev < sma else "flat"
            if trend_fast_len and trend_slow_len:
                fast = compute_sma(bars, trend_fast_len, prev_idx)
                slow = compute_sma(bars, trend_slow_len, prev_idx)
                fast_prev = compute_sma(bars, trend_fast_len, prev_idx - 1)
                atr = compute_atr_daily(bars, atr_period, prev_idx)
                if fast is None or slow is None or fast_prev is None or atr is None:
                    continue
                slope_bps = ((fast - fast_prev) / fast_prev) * 10000.0 if fast_prev else 0.0
                if fast > slow and slope_bps >= trend_min_slope_bps and close_prev >= slow + trend_min_distance_atr * atr:
                    trend_state = "uptrend"
                elif fast < slow and slope_bps <= -trend_min_slope_bps and close_prev <= slow - trend_min_distance_atr * atr:
                    trend_state = "downtrend"
                else:
                    trend_state = "flat"
            if trend_state == "flat":
                continue
            lookback_days = max(1, reversal_lookback_days)
            prevn_idx = prev_idx - lookback_days
            if prevn_idx < 0:
                continue
            close_prevn = float(bars[prevn_idx]["close"])
            if close_prevn == 0:
                continue
            return_pct = ((close_prev / close_prevn) - 1.0) * 100.0
            if volume_lookback_days > 0 and volume_min_ratio > 0:
                vol_start = max(0, prev_idx - volume_lookback_days + 1)
                if prev_idx - vol_start + 1 < volume_lookback_days:
                    continue
                vols = []
                for i in range(vol_start, prev_idx + 1):
                    try:
                        vols.append(float(bars[i]["volume"]))
                    except Exception:
                        vols.append(0.0)
                vols.sort()
                median_vol = vols[len(vols) // 2] if vols else 0.0
                vol_prev = float(bars[prev_idx]["volume"]) if bars[prev_idx].get("volume") is not None else 0.0
                if median_vol <= 0 or vol_prev < (median_vol * volume_min_ratio):
                    continue
            direction = None
            if reversal_mode == "quantile":
                q = min(0.49, max(0.01, reversal_quantile))
                lb = max(2, reversal_quantile_lookback_days)
                start_idx = prev_idx - lb
                if start_idx < 1:
                    continue
                returns = []
                for i in range(start_idx + 1, prev_idx + 1):
                    try:
                        c1 = float(bars[i]["close"])
                        c0 = float(bars[i - 1]["close"])
                    except Exception:
                        continue
                    if c0 == 0:
                        continue
                    returns.append(((c1 / c0) - 1.0) * 100.0)
                if len(returns) < max(5, lb // 2):
                    continue
                returns.sort()
                low_idx = int((len(returns) - 1) * q)
                high_idx = int((len(returns) - 1) * (1.0 - q))
                low_thr = returns[low_idx]
                high_thr = returns[high_idx]
                if trend_state == "uptrend" and return_pct <= low_thr:
                    direction = "long"
                elif trend_state == "downtrend" and return_pct >= high_thr:
                    direction = "short"
            elif reversal_threshold_atr > 0:
                atr = compute_atr_daily(bars, atr_period, prev_idx)
                if not atr or atr <= 0:
                    continue
                return_atr = (close_prev - close_prevn) / atr
                if trend_state == "uptrend" and return_atr <= -reversal_threshold_atr:
                    direction = "long"
                elif trend_state == "downtrend" and return_atr >= reversal_threshold_atr:
                    direction = "short"
            else:
                if trend_state == "uptrend" and return_pct <= -reversal_threshold_pct:
                    direction = "long"
                elif trend_state == "downtrend" and return_pct >= reversal_threshold_pct:
                    direction = "short"
            if not direction:
                continue
            out.append(
                Signal(
                    symbol=symbol,
                    signal_date=str(bar.get("date")),
                    direction=direction,
                    trend_state=trend_state,
                    return_pct=return_pct,
                )
            )
    return out


def generate_signal_for_date(
    symbol: str,
    target_date: str,
    cfg: Dict,
    data_store: AlpacaOHLCStore,
) -> Optional[Signal]:
    sym = str(symbol).upper()
    if not sym:
        return None
    params = cfg.get("daily_trend_reversal") or {}
    trend_ma_days = int(params.get("trend_ma_days") or 200)
    reversal_mode = str(params.get("reversal_mode") or "").lower().strip()
    reversal_threshold_pct = float(params.get("reversal_threshold_pct") or 0.0)
    reversal_threshold_atr = float(params.get("reversal_threshold_atr") or 0.0)
    atr_period = int(params.get("atr_period") or 14)
    reversal_lookback_days = int(params.get("reversal_lookback_days") or 1)
    reversal_quantile = float(params.get("reversal_quantile") or 0.2)
    reversal_quantile_lookback_days = int(params.get("reversal_quantile_lookback_days") or 60)
    trend_fast_len = int(params.get("trend_fast_len") or 0)
    trend_slow_len = int(params.get("trend_slow_len") or 0)
    trend_min_slope_bps = float(params.get("trend_min_slope_bps") or 0.0)
    trend_min_distance_atr = float(params.get("trend_min_distance_atr") or 0.0)
    volume_lookback_days = int(params.get("volume_lookback_days") or 0)
    volume_min_ratio = float(params.get("volume_min_ratio") or 0.0)
    if not reversal_mode:
        reversal_mode = "atr" if reversal_threshold_atr > 0 else "pct"

    bars = data_store.get_daily_bars(sym, None, None, cfg=cfg, allow_fetch=True)
    if not bars or len(bars) < trend_ma_days + 2:
        return None
    tgt = ensure_date(target_date)
    prev_idx = None
    for idx, bar in enumerate(bars):
        bar_date = ensure_date(str(bar.get("date")))
        if bar_date < tgt:
            prev_idx = idx
    if prev_idx is None:
        return None
    if prev_idx < trend_ma_days - 1 or prev_idx - 1 < 0:
        return None
    sma = compute_sma(bars, trend_ma_days, prev_idx)
    if sma is None:
        return None
    close_prev = float(bars[prev_idx]["close"])
    trend_state = "uptrend" if close_prev > sma else "downtrend" if close_prev < sma else "flat"
    if trend_fast_len and trend_slow_len:
        fast = compute_sma(bars, trend_fast_len, prev_idx)
        slow = compute_sma(bars, trend_slow_len, prev_idx)
        fast_prev = compute_sma(bars, trend_fast_len, prev_idx - 1)
        atr = compute_atr_daily(bars, atr_period, prev_idx)
        if fast is None or slow is None or fast_prev is None or atr is None:
            return None
        slope_bps = ((fast - fast_prev) / fast_prev) * 10000.0 if fast_prev else 0.0
        if fast > slow and slope_bps >= trend_min_slope_bps and close_prev >= slow + trend_min_distance_atr * atr:
            trend_state = "uptrend"
        elif fast < slow and slope_bps <= -trend_min_slope_bps and close_prev <= slow - trend_min_distance_atr * atr:
            trend_state = "downtrend"
        else:
            trend_state = "flat"
    if trend_state == "flat":
        return None

    lookback_days = max(1, reversal_lookback_days)
    prevn_idx = prev_idx - lookback_days
    if prevn_idx < 0:
        return None
    close_prevn = float(bars[prevn_idx]["close"])
    if close_prevn == 0:
        return None
    return_pct = ((close_prev / close_prevn) - 1.0) * 100.0

    if volume_lookback_days > 0 and volume_min_ratio > 0:
        vol_start = max(0, prev_idx - volume_lookback_days + 1)
        if prev_idx - vol_start + 1 < volume_lookback_days:
            return None
        vols = []
        for i in range(vol_start, prev_idx + 1):
            try:
                vols.append(float(bars[i]["volume"]))
            except Exception:
                vols.append(0.0)
        vols.sort()
        median_vol = vols[len(vols) // 2] if vols else 0.0
        vol_prev = float(bars[prev_idx]["volume"]) if bars[prev_idx].get("volume") is not None else 0.0
        if median_vol <= 0 or vol_prev < (median_vol * volume_min_ratio):
            return None

    direction = None
    if reversal_mode == "quantile":
        q = min(0.49, max(0.01, reversal_quantile))
        lb = max(2, reversal_quantile_lookback_days)
        start_idx = prev_idx - lb
        if start_idx < 1:
            return None
        returns = []
        for i in range(start_idx + 1, prev_idx + 1):
            try:
                c1 = float(bars[i]["close"])
                c0 = float(bars[i - 1]["close"])
            except Exception:
                continue
            if c0 == 0:
                continue
            returns.append(((c1 / c0) - 1.0) * 100.0)
        if len(returns) < max(5, lb // 2):
            return None
        returns.sort()
        low_idx = int((len(returns) - 1) * q)
        high_idx = int((len(returns) - 1) * (1.0 - q))
        low_thr = returns[low_idx]
        high_thr = returns[high_idx]
        if trend_state == "uptrend" and return_pct <= low_thr:
            direction = "long"
        elif trend_state == "downtrend" and return_pct >= high_thr:
            direction = "short"
    elif reversal_threshold_atr > 0:
        atr = compute_atr_daily(bars, atr_period, prev_idx)
        if not atr or atr <= 0:
            return None
        return_atr = (close_prev - close_prevn) / atr
        if trend_state == "uptrend" and return_atr <= -reversal_threshold_atr:
            direction = "long"
        elif trend_state == "downtrend" and return_atr >= reversal_threshold_atr:
            direction = "short"
    else:
        if trend_state == "uptrend" and return_pct <= -reversal_threshold_pct:
            direction = "long"
        elif trend_state == "downtrend" and return_pct >= reversal_threshold_pct:
            direction = "short"
    if not direction:
        return None
    return Signal(
        symbol=sym,
        signal_date=str(tgt),
        direction=direction,
        trend_state=trend_state,
        return_pct=return_pct,
    )


def build_trade(
    signal: Signal,
    cfg: Dict,
    data_store: AlpacaOHLCStore,
    context: str = "replay",
    bars_intraday: Optional[List[Dict]] = None,
    entry_time_override: Optional[str] = None,
    param_overrides: Optional[Dict] = None,
) -> Optional[TradePlan]:
    params = dict(cfg.get("daily_trend_reversal") or {})
    if param_overrides:
        params.update(param_overrides)
    entry_time_et = str(params.get("entry_time_et") or "09:35")
    entry_start_et = str(params.get("entry_start_et") or entry_time_et)
    entry_end_et = str(params.get("entry_end_et") or entry_time_et)
    entry_times_raw = params.get("entry_times_et")
    if entry_time_override:
        entry_times = [str(entry_time_override)]
    elif isinstance(entry_times_raw, list) and entry_times_raw:
        entry_times = [str(t) for t in entry_times_raw if t]
    else:
        entry_times = [entry_time_et]
    try:
        entry_times = sorted(entry_times, key=lambda t: parse_time_hhmm(t))
    except Exception:
        pass
    atr_period = int(params.get("atr_period") or 14)
    stop_mode = str(params.get("stop_mode") or "atr").lower()
    stop_atr_mult = float(params.get("stop_atr_mult") or 1.0)
    stop_pct = float(params.get("stop_pct") or 1.0)
    # Prefer `target_rr` (used by sweeps/watchlist grids). Keep `target_r` as a legacy alias.
    target_rr = float(params.get("target_rr") or params.get("target_r") or 1.5)
    target_mode = str(params.get("target_mode") or "rr").lower()
    target_window_minutes = int(params.get("target_window_minutes") or params.get("time_stop_minutes") or 0)
    target_window_lookback_days = int(params.get("target_window_lookback_days") or 60)
    target_window_mult = float(params.get("target_window_mult") or 0.8)
    target_window_min_samples = int(params.get("target_window_min_samples") or 15)
    target_window_min_pct = float(params.get("target_window_min_pct") or 0.1)
    target_window_max_pct = float(params.get("target_window_max_pct") or 5.0)
    min_rr = float(params.get("min_rr") or 0.0)
    stop_r = float(params.get("stop_r") or 1.0)
    time_exit_days = int(params.get("time_exit_days") or 1)
    intraday_only = bool(params.get("intraday_only", False))
    intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
    intraday_filter_apply_watchlist = bool(params.get("intraday_filter_apply_in_watchlist", True))
    intraday_filter_require = bool(params.get("intraday_filter_require_bars", False))
    confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
    confirm_minutes = int(params.get("confirm_minutes") or 0)
    confirm_apply_in_watchlist = bool(params.get("confirm_apply_in_watchlist", True))
    confirm_entry_price_mode = str(params.get("confirm_entry_price_mode") or "close").lower()
    min_gap_bps_long = float(params.get("min_gap_bps_long") or 0.0)
    min_gap_bps_short = float(params.get("min_gap_bps_short") or 0.0)
    reversal_lookback_days = int(params.get("reversal_lookback_days") or 1)
    early_range_minutes = int(params.get("early_range_minutes") or 0)
    max_early_pullback_bps = float(params.get("max_early_pullback_bps") or 0.0)
    session_open_et = str(params.get("session_open_et") or "09:30")
    use_intraday_entry = bool(params.get("use_intraday_entry", False))
    intraday_entry_in_watchlist = bool(params.get("intraday_entry_in_watchlist", False))
    bars = data_store.get_daily_bars(signal.symbol, None, None, cfg=cfg, allow_fetch=True)
    if not bars:
        return None
    direction = signal.direction.lower()
    apply_gap_filter = (min_gap_bps_long > 0) or (min_gap_bps_short > 0)
    apply_early_filter = intraday_filter_enabled
    if context == "watchlist" and not intraday_filter_apply_watchlist:
        apply_early_filter = False
    apply_intraday_entry = use_intraday_entry and (context != "watchlist" or intraday_entry_in_watchlist)
    apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0
    if context == "watchlist" and not confirm_apply_in_watchlist:
        apply_confirm = False
    minutes_needed = 0
    if apply_early_filter and early_range_minutes > 0 and max_early_pullback_bps > 0:
        minutes_needed = early_range_minutes
    max_entry_minutes = 0
    for t in entry_times:
        try:
            entry_time = parse_time_hhmm(t)
            open_time = parse_time_hhmm(session_open_et)
            entry_minutes = int(
                (dt.datetime.combine(dt.date.today(), entry_time) - dt.datetime.combine(dt.date.today(), open_time)).total_seconds() / 60
            )
            entry_minutes = max(1, entry_minutes + 1)
            max_entry_minutes = max(max_entry_minutes, entry_minutes)
        except Exception:
            max_entry_minutes = max(max_entry_minutes, 1)
    if apply_intraday_entry:
        minutes_needed = max(minutes_needed, max_entry_minutes)
    if apply_confirm:
        minutes_needed = max(minutes_needed, max_entry_minutes + confirm_minutes)
    if minutes_needed > 0 and bars_intraday is None:
        bars_intraday = get_intraday_bars(signal.symbol, signal.signal_date, minutes_needed, cfg=cfg, allow_fetch=True)
    def _intraday_entry_price(
        bars_intraday_local: List[Dict],
        entry_time_str: str,
        entry_price_mode: str,
        entry_date_str: str,
    ) -> Optional[float]:
        entry_dt = ensure_et(dt.datetime.combine(ensure_date(entry_date_str), parse_time_hhmm(entry_time_str)))
        # Parity with live: we only have completed minute bars strictly BEFORE entry_dt (see filter_intraday_bars_until).
        # So we use the last available completed bar before entry_dt as the entry-price reference.
        best_row: Optional[Dict] = None
        best_ts: Optional[dt.datetime] = None
        for row in bars_intraday_local:
            ts_raw = row.get("timestamp") or row.get("datetime") or row.get("time") or row.get("date")
            if not ts_raw:
                continue
            try:
                ts = ensure_et(dt.datetime.fromisoformat(str(ts_raw)))
            except Exception:
                continue
            if ts >= entry_dt:
                # bars are time-sorted; once we pass entry_dt there is nothing else to consider.
                break
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_row = row
        if not best_row:
            return None
        if entry_price_mode == "close":
            return float(best_row.get("close") or best_row.get("c") or 0.0)
        return float(best_row.get("open") or best_row.get("o") or 0.0)

    def _confirm_entry(
        bars_intraday_local: List[Dict],
        entry_time_str: str,
        entry_date_str: str,
        entry_price_local: float,
        direction_local: str,
    ) -> Optional[Dict[str, object]]:
        if not bars_intraday_local or entry_price_local <= 0 or confirm_minutes <= 0 or confirm_move_bps <= 0:
            return None
        entry_dt = ensure_et(dt.datetime.combine(ensure_date(entry_date_str), parse_time_hhmm(entry_time_str)))
        cutoff_dt = entry_dt + dt.timedelta(minutes=confirm_minutes)
        for row in bars_intraday_local:
            ts_raw = row.get("timestamp") or row.get("datetime") or row.get("time") or row.get("date")
            if not ts_raw:
                continue
            try:
                ts = ensure_et(dt.datetime.fromisoformat(str(ts_raw)))
            except Exception:
                continue
            if ts < entry_dt:
                continue
            if ts > cutoff_dt:
                break
            high = float(row.get("high") or row.get("h") or 0.0)
            low = float(row.get("low") or row.get("l") or 0.0)
            if direction_local == "long":
                hit_bps = ((high - entry_price_local) / entry_price_local) * 10000.0
                if hit_bps >= confirm_move_bps:
                    confirm_price = float(row.get("close") or row.get("c") or 0.0) if confirm_entry_price_mode == "close" else float(row.get("open") or row.get("o") or 0.0)
                    return {"entry_price": confirm_price, "entry_time_et": ts.strftime("%H:%M"), "confirm_hit_bps": hit_bps}
            else:
                hit_bps = ((entry_price_local - low) / entry_price_local) * 10000.0
                if hit_bps >= confirm_move_bps:
                    confirm_price = float(row.get("close") or row.get("c") or 0.0) if confirm_entry_price_mode == "close" else float(row.get("open") or row.get("o") or 0.0)
                    return {"entry_price": confirm_price, "entry_time_et": ts.strftime("%H:%M"), "confirm_hit_bps": hit_bps}
        return None

    signal_return_pct = float(signal.return_pct)
    for entry_time_et in entry_times:
        if entry_start_et and entry_end_et:
            if not (entry_start_et <= entry_time_et <= entry_end_et):
                continue
        entry_info = simulate_entry(signal, entry_time_et, "daily", bars, None, cfg)
        if not entry_info:
            continue
        entry_price = float(entry_info["entry_price"])
        entry_date = str(entry_info["entry_date"])
        entry_idx = entry_info["entry_index"]
        signal_return_atr = None
        signal_atr = None
        if entry_idx is not None:
            prev_idx = entry_idx - 1
            prevn_idx = prev_idx - max(1, reversal_lookback_days)
            if prev_idx >= 0 and prevn_idx >= 0:
                try:
                    close_prev = float(bars[prev_idx]["close"])
                    close_prevn = float(bars[prevn_idx]["close"])
                except Exception:
                    close_prev = None
                    close_prevn = None
                atr_for_signal = compute_atr_daily(bars, atr_period, prev_idx)
                signal_atr = atr_for_signal
                if atr_for_signal and close_prev is not None and close_prevn is not None:
                    signal_return_atr = (close_prev - close_prevn) / atr_for_signal
        if apply_intraday_entry:
            if not bars_intraday:
                continue
            entry_price_mode = str((cfg.get("daily_trend_reversal") or {}).get("entry_price_mode") or "open").lower()
            intraday_price = _intraday_entry_price(bars_intraday, entry_time_et, entry_price_mode, signal.signal_date)
            if intraday_price is None:
                continue
            entry_price = float(intraday_price)
            entry_date = str(signal.signal_date)
        confirm_hit_bps = None
        if apply_confirm:
            if not bars_intraday:
                continue
            confirm = _confirm_entry(bars_intraday, entry_time_et, signal.signal_date, entry_price, direction)
            if not confirm:
                continue
            entry_price = float(confirm["entry_price"])
            entry_time_et = str(confirm["entry_time_et"])
            confirm_hit_bps = float(confirm["confirm_hit_bps"]) if confirm.get("confirm_hit_bps") is not None else None
        if apply_confirm:
            entry_price_mode_used = f"confirm_{confirm_entry_price_mode}"
        elif apply_intraday_entry:
            entry_price_mode_used = f"intraday_{entry_price_mode}"
        else:
            entry_price_mode_used = f"daily_{str((cfg.get('daily_trend_reversal') or {}).get('entry_price_mode') or 'open').lower()}"
        gap_bps = None
        if entry_idx is not None and entry_idx > 0:
            try:
                prev_close = float(bars[entry_idx - 1]["close"])
                open_price = float(bars[entry_idx]["open"])
            except Exception:
                prev_close = 0.0
                open_price = 0.0
            if prev_close > 0 and open_price > 0:
                gap_bps = ((open_price - prev_close) / prev_close) * 10000.0
                if apply_gap_filter:
                    if direction == "long" and min_gap_bps_long > 0 and gap_bps < min_gap_bps_long:
                        continue
                    if direction == "short" and min_gap_bps_short > 0 and gap_bps > -min_gap_bps_short:
                        continue
        early_pullback_bps = None
        if apply_early_filter and early_range_minutes > 0 and max_early_pullback_bps > 0:
            if not bars_intraday and intraday_filter_require:
                continue
            if bars_intraday:
                slice_end = min(len(bars_intraday), early_range_minutes)
                window = bars_intraday[:slice_end]
                lows = [float(b.get("low") or b.get("l") or 0.0) for b in window]
                highs = [float(b.get("high") or b.get("h") or 0.0) for b in window]
                min_low = min(lows) if lows else None
                max_high = max(highs) if highs else None
                if direction == "long" and min_low is not None and entry_price > 0:
                    pullback_bps = ((entry_price - min_low) / entry_price) * 10000.0
                    early_pullback_bps = pullback_bps
                    if pullback_bps >= max_early_pullback_bps:
                        continue
                if direction == "short" and max_high is not None and entry_price > 0:
                    pullback_bps = ((max_high - entry_price) / entry_price) * 10000.0
                    early_pullback_bps = pullback_bps
                    if pullback_bps >= max_early_pullback_bps:
                        continue
        atr_end_idx = entry_idx - 1
        if entry_info.get("entry_source_date") != entry_date:
            atr_end_idx = entry_idx
        atr = compute_atr_daily(bars, atr_period, atr_end_idx)
        target_mode_used = target_mode
        target_window_avg_pct = None
        target_window_samples = None
        target_distance = None
        if target_mode == "symbol_window_avg" and target_window_minutes > 0:
            avg_info = _compute_symbol_window_avg(
                signal.symbol,
                signal.signal_date,
                entry_time_et,
                direction,
                entry_price_mode_used,
                target_window_minutes,
                target_window_lookback_days,
                target_window_min_samples,
                cfg,
            )
            if avg_info:
                target_window_avg_pct = float(avg_info.get("avg_mfe_pct") or 0.0)
                target_window_samples = int(avg_info.get("samples") or 0)
                target_distance_pct = target_window_avg_pct * target_window_mult
                if target_window_min_pct > 0:
                    target_distance_pct = max(target_distance_pct, target_window_min_pct)
                if target_window_max_pct > 0:
                    target_distance_pct = min(target_distance_pct, target_window_max_pct)
                target_distance = entry_price * (target_distance_pct / 100.0)
                if target_distance <= 0:
                    target_mode_used = "rr"
            else:
                target_mode_used = "rr"
        stop_distance = None
        if stop_mode != "target_rr":
            if stop_mode == "atr":
                if atr is None:
                    continue
                stop_distance = atr * stop_atr_mult * stop_r
            else:
                stop_distance = entry_price * (stop_pct / 100.0) * stop_r
            if stop_distance <= 0:
                continue

        if target_mode_used == "symbol_window_avg" and target_distance:
            if stop_mode == "target_rr":
                stop_distance = target_distance / max(target_rr, 0.0001)
                if stop_distance <= 0:
                    continue
            target_rr = target_distance / stop_distance
            if direction == "long":
                stop_price = entry_price - stop_distance
                target_price = entry_price + target_distance
            else:
                stop_price = entry_price + stop_distance
                target_price = entry_price - target_distance
        else:
            if stop_distance is None:
                if atr is None:
                    continue
                stop_distance = atr * stop_atr_mult * stop_r
                if stop_distance <= 0:
                    continue
            if direction == "long":
                stop_price = entry_price - stop_distance
                target_price = entry_price + stop_distance * target_rr
            else:
                stop_price = entry_price + stop_distance
                target_price = entry_price - stop_distance * target_rr
        if min_rr > 0:
            effective_rr = target_rr
            if target_distance is not None and stop_distance > 0:
                effective_rr = target_distance / stop_distance
            if effective_rr < min_rr:
                continue
        if intraday_only:
            time_exit_idx = entry_idx
        else:
            time_exit_idx = min(entry_idx + time_exit_days, len(bars) - 1)
        time_exit_date = str(bars[time_exit_idx]["date"])
        return TradePlan(
            symbol=signal.symbol,
            direction=direction,
            signal_date=signal.signal_date,
            entry_date=entry_date,
            entry_time_et=entry_time_et,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            time_exit_date=time_exit_date,
            stop_distance=stop_distance,
            target_rr=target_rr,
            target_mode=target_mode_used,
            target_window_avg_pct=target_window_avg_pct,
            target_window_mult=target_window_mult if target_mode_used == "symbol_window_avg" else None,
            target_window_minutes=target_window_minutes if target_mode_used == "symbol_window_avg" else None,
            target_window_samples=target_window_samples,
            gap_bps=gap_bps,
            early_pullback_bps=early_pullback_bps,
            confirm_move_bps=confirm_move_bps if apply_confirm else None,
            confirm_minutes=confirm_minutes if apply_confirm else None,
            confirm_hit_bps=confirm_hit_bps,
            signal_return_pct=signal_return_pct,
            signal_return_atr=signal_return_atr,
            atr=signal_atr,
            entry_price_mode=entry_price_mode_used,
        )
    return None
