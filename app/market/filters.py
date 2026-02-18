from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple

from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.execution.daily_execution_model import compute_atr_daily, compute_sma
from app.utils.time import ensure_date


def market_filter_decision(
    date_str: str,
    cfg: Dict,
    data_store: AlpacaOHLCStore,
) -> Tuple[bool, Dict]:
    mf = cfg.get("market_filters") or {}
    if not bool(mf.get("enabled", False)):
        return False, {}

    symbol = str(mf.get("symbol") or "SPY").upper()
    # Fetch a bounded history window so symbols not already cached on disk (e.g. SPY) still
    # have enough bars for ATR/SMA computations (otherwise Alpaca may return only recent bars).
    try:
        tgt = ensure_date(date_str)
    except Exception:
        return False, {"reason": "invalid_date", "date": date_str, "symbol": symbol}
    atr_period = int(mf.get("atr_period") or 14)
    trend_days = int(mf.get("trend_ma_days") or 200)
    # Use calendar days with a weekend buffer; trading-day exactness isn't required here.
    lookback_days = max(atr_period + 10, trend_days + 10, 90)
    start = (tgt - dt.timedelta(days=int(lookback_days * 3))).isoformat()
    bars = data_store.get_daily_bars(symbol, start, date_str, cfg=cfg, allow_fetch=True)
    if not bars:
        return False, {"reason": "no_market_bars"}

    idx = None
    for i, bar in enumerate(bars):
        if ensure_date(str(bar.get("date"))) == tgt:
            idx = i
            break
    if idx is None or idx - 1 < 0:
        return False, {"reason": "insufficient_market_history"}

    prev = bars[idx - 1]
    try:
        prev_close = float(prev.get("close") or 0.0)
    except Exception:
        prev_close = 0.0
    try:
        open_today = float(bars[idx].get("open") or 0.0)
    except Exception:
        open_today = 0.0

    reasons = []
    details: Dict[str, float | str] = {"symbol": symbol, "date": date_str}

    if prev_close > 0 and open_today > 0 and bool(mf.get("use_gap", True)):
        gap_bps = ((open_today - prev_close) / prev_close) * 10000.0
        details["gap_bps"] = gap_bps
        gap_max = float(mf.get("gap_bps_max") or 0.0)
        if gap_max > 0 and abs(gap_bps) > gap_max:
            reasons.append("gap")

    if bool(mf.get("use_atr", True)):
        atr_period = int(mf.get("atr_period") or 14)
        atr = compute_atr_daily(bars, atr_period, idx - 1)
        if atr and prev_close > 0:
            atr_pct = (atr / prev_close) * 100.0
            details["atr_pct"] = atr_pct
            atr_max = float(mf.get("atr_max_pct") or 0.0)
            if atr_max > 0 and atr_pct > atr_max:
                reasons.append("atr")

    if bool(mf.get("use_trend", False)):
        trend_days = int(mf.get("trend_ma_days") or 200)
        sma = compute_sma(bars, trend_days, idx - 1)
        if sma is not None and prev_close > 0:
            details["trend_ma"] = sma
            if prev_close < sma:
                reasons.append("trend")

    if reasons:
        return True, {"reasons": reasons, **details}
    return False, details
