from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from app.data.alpaca_intraday_store import get_intraday_bars
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.execution.daily_execution_model import compute_atr_daily, compute_sma, simulate_entry
from app.strategies.types import Signal, TradePlan
from app.utils.time import ensure_date


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


def build_trade(
    signal: Signal,
    cfg: Dict,
    data_store: AlpacaOHLCStore,
    context: str = "replay",
    bars_intraday: Optional[List[Dict]] = None,
) -> Optional[TradePlan]:
    params = cfg.get("daily_trend_reversal") or {}
    entry_time_et = str(params.get("entry_time_et") or "09:35")
    entry_start_et = str(params.get("entry_start_et") or entry_time_et)
    entry_end_et = str(params.get("entry_end_et") or entry_time_et)
    atr_period = int(params.get("atr_period") or 14)
    stop_mode = str(params.get("stop_mode") or "atr").lower()
    stop_atr_mult = float(params.get("stop_atr_mult") or 1.0)
    stop_pct = float(params.get("stop_pct") or 1.0)
    target_rr = float(params.get("target_r") or params.get("target_rr") or 1.5)
    stop_r = float(params.get("stop_r") or 1.0)
    time_exit_days = int(params.get("time_exit_days") or 1)
    intraday_only = bool(params.get("intraday_only", False))
    intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
    intraday_filter_apply_watchlist = bool(params.get("intraday_filter_apply_in_watchlist", True))
    intraday_filter_require = bool(params.get("intraday_filter_require_bars", False))
    min_gap_bps_long = float(params.get("min_gap_bps_long") or 0.0)
    min_gap_bps_short = float(params.get("min_gap_bps_short") or 0.0)
    early_range_minutes = int(params.get("early_range_minutes") or 0)
    max_early_pullback_bps = float(params.get("max_early_pullback_bps") or 0.0)
    bars = data_store.get_daily_bars(signal.symbol, None, None, cfg=cfg, allow_fetch=True)
    if not bars:
        return None
    if entry_start_et and entry_end_et:
        if not (entry_start_et <= entry_time_et <= entry_end_et):
            return None
    entry_info = simulate_entry(signal, entry_time_et, "daily", bars, None, cfg)
    if not entry_info:
        return None
    entry_price = float(entry_info["entry_price"])
    entry_date = str(entry_info["entry_date"])
    entry_idx = entry_info["entry_index"]
    direction = signal.direction.lower()
    apply_gap_filter = (min_gap_bps_long > 0) or (min_gap_bps_short > 0)
    apply_early_filter = intraday_filter_enabled
    if context == "watchlist" and not intraday_filter_apply_watchlist:
        apply_early_filter = False
    minutes_needed = 0
    if apply_early_filter and early_range_minutes > 0 and max_early_pullback_bps > 0:
        minutes_needed = early_range_minutes
    if minutes_needed > 0 and bars_intraday is None:
        bars_intraday = get_intraday_bars(signal.symbol, entry_date, minutes_needed, cfg=cfg, allow_fetch=True)
    if apply_gap_filter:
        if entry_idx is not None and entry_idx > 0:
            try:
                prev_close = float(bars[entry_idx - 1]["close"])
                open_price = float(bars[entry_idx]["open"])
            except Exception:
                prev_close = 0.0
                open_price = 0.0
            if prev_close > 0 and open_price > 0:
                gap_bps = ((open_price - prev_close) / prev_close) * 10000.0
                if direction == "long" and min_gap_bps_long > 0 and gap_bps < min_gap_bps_long:
                    return None
                if direction == "short" and min_gap_bps_short > 0 and gap_bps > -min_gap_bps_short:
                    return None
    if apply_early_filter and early_range_minutes > 0 and max_early_pullback_bps > 0:
        if not bars_intraday and intraday_filter_require:
            return None
        if bars_intraday:
                slice_end = min(len(bars_intraday), early_range_minutes)
                window = bars_intraday[:slice_end]
                lows = [float(b.get("low") or b.get("l") or 0.0) for b in window]
                highs = [float(b.get("high") or b.get("h") or 0.0) for b in window]
                min_low = min(lows) if lows else None
                max_high = max(highs) if highs else None
                if direction == "long" and min_low is not None and entry_price > 0:
                    pullback_bps = ((entry_price - min_low) / entry_price) * 10000.0
                    if pullback_bps >= max_early_pullback_bps:
                        return None
                if direction == "short" and max_high is not None and entry_price > 0:
                    pullback_bps = ((max_high - entry_price) / entry_price) * 10000.0
                    if pullback_bps >= max_early_pullback_bps:
                        return None
    atr_end_idx = entry_idx - 1
    if entry_info.get("entry_source_date") != entry_date:
        atr_end_idx = entry_idx
    atr = compute_atr_daily(bars, atr_period, atr_end_idx)
    if stop_mode == "atr":
        if atr is None:
            return None
        stop_distance = atr * stop_atr_mult * stop_r
    else:
        stop_distance = entry_price * (stop_pct / 100.0) * stop_r
    if stop_distance <= 0:
        return None
    if direction == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + stop_distance * target_rr
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - stop_distance * target_rr
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
    )
