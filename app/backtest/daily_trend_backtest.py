from __future__ import annotations

import datetime as dt
import json
import logging
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

from app.config.loader import load_config
from app.data.alpaca_quote_store import resolve_quote_for_timestamp
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.market.filters import market_filter_decision
from app.portfolio.sizing import compute_qty_with_guards, estimate_slot_target_from_stats
from app.replay.daily_strategy_replay import run_replay
from app.utils.time import ensure_et, iter_trading_days, parse_time_hhmm
from app.watchlist.daily_strategy_builder import build_watchlist
from app.watchlist.node_assets import read_asset_universe_snapshot, resolve_asset_universe_symbols

_FILL_MODEL_LOGGED = False
_FILL_MODEL_DEBUG_LOGS = 0
_QUOTE_MODEL_LOGGED = False
_QUOTE_MODEL_DEBUG_LOGS = 0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _adverse_fill_price(price: float, *, side: str, bps: float) -> float:
    px = _safe_float(price, default=0.0)
    if px <= 0:
        return px
    bps_eff = max(0.0, _safe_float(bps, default=0.0))
    if bps_eff <= 0:
        return px
    mult = 1.0 + (bps_eff / 10000.0) if str(side).lower() == "buy" else 1.0 - (bps_eff / 10000.0)
    return max(0.0001, px * mult)


def _summarize(trades) -> Dict[str, float]:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avgR": 0.0, "total_pnl_pct": 0.0}
    wins = [t for t in trades if t.r_multiple > 0]
    win_rate = len(wins) / float(len(trades))
    avg_r = sum(t.r_multiple for t in trades) / float(len(trades))
    total_pnl = sum(t.pnl_pct for t in trades)
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avgR": avg_r,
        "total_pnl_pct": total_pnl,
    }


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)

    def _p(pct: float) -> float:
        if not vals:
            return 0.0
        idx = int(round((len(vals) - 1) * pct))
        idx = max(0, min(len(vals) - 1, idx))
        return float(vals[idx])

    return {
        "p10": _p(0.10),
        "p25": _p(0.25),
        "p50": _p(0.50),
        "p75": _p(0.75),
        "p90": _p(0.90),
    }


def _build_daily_lookup(
    data_store: AlpacaOHLCStore,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    cfg: Dict,
) -> Dict[str, Dict[str, Dict]]:
    bars_map = data_store.get_daily_bars_bulk(symbols, start_date, end_date, cfg=cfg, allow_fetch=False)
    lookup: Dict[str, Dict[str, Dict]] = {}
    for sym, bars in bars_map.items():
        day_map: Dict[str, Dict] = {}
        for bar in bars or []:
            if bar.get("date"):
                day_map[str(bar["date"])] = bar
        lookup[sym] = day_map
    return lookup


def _attach_move_stats(record: Dict, daily_lookup: Dict[str, Dict[str, Dict]]) -> None:
    symbol = str(record.get("symbol") or "").upper()
    entry_date = str(record.get("entry_date") or "")
    entry_price = float(record.get("entry_price") or 0.0)
    stop_distance = float(record.get("stop_distance") or 0.0)
    direction = str(record.get("direction") or "long").lower()
    day_bar = daily_lookup.get(symbol, {}).get(entry_date)
    if not day_bar or entry_price <= 0:
        record["day_high"] = None
        record["day_low"] = None
        record["day_mfe_pct"] = None
        record["day_mae_pct"] = None
        record["day_mfe_r"] = None
        record["day_mae_r"] = None
        return
    high = float(day_bar.get("high") or 0.0)
    low = float(day_bar.get("low") or 0.0)
    if direction == "long":
        mfe = (high - entry_price) / entry_price * 100.0
        mae = (low - entry_price) / entry_price * 100.0
    else:
        mfe = (entry_price - low) / entry_price * 100.0
        mae = (entry_price - high) / entry_price * 100.0
    record["day_high"] = high
    record["day_low"] = low
    record["day_mfe_pct"] = mfe
    record["day_mae_pct"] = mae
    if stop_distance > 0:
        record["day_mfe_r"] = (mfe / 100.0) * entry_price / stop_distance
        record["day_mae_r"] = (mae / 100.0) * entry_price / stop_distance
    else:
        record["day_mfe_r"] = None
        record["day_mae_r"] = None


def _apply_portfolio_sizing(
    day_trades: List,
    equity: float,
    cfg: Dict,
) -> Tuple[List, List[Dict], float]:
    params = cfg.get("daily_trend_reversal") or {}
    intraday_only = bool(params.get("intraday_only", False))
    session_close_et = str(params.get("session_close_et") or "16:00")
    fill_model_enabled = bool(params.get("backtest_fill_model_enabled", False))
    fill_half_spread_bps = max(0.0, _safe_float(params.get("backtest_half_spread_bps"), default=0.0))
    fill_entry_slippage_bps = max(0.0, _safe_float(params.get("backtest_entry_slippage_bps"), default=0.0))
    fill_stop_slippage_bps = max(0.0, _safe_float(params.get("backtest_stop_slippage_bps"), default=0.0))
    fill_target_slippage_bps = max(0.0, _safe_float(params.get("backtest_target_slippage_bps"), default=0.0))
    fill_time_slippage_bps = max(0.0, _safe_float(params.get("backtest_time_slippage_bps"), default=0.0))
    fill_debug = bool(params.get("backtest_fill_debug", False))
    fill_debug_max_logs = max(0, _safe_int(params.get("backtest_fill_debug_max_logs"), default=40))
    quote_fill_enabled = bool(params.get("backtest_quote_fill_enabled", False))
    quote_lookback_seconds = max(1, _safe_int(params.get("backtest_quote_lookback_seconds"), default=120))
    quote_forward_seconds = max(0, _safe_int(params.get("backtest_quote_forward_seconds"), default=10))
    quote_max_age_seconds = max(1, _safe_int(params.get("backtest_quote_max_age_seconds"), default=300))
    quote_fallback_to_bps = bool(params.get("backtest_quote_fallback_to_bps", True))
    quote_target_limit_protect = bool(params.get("backtest_quote_target_limit_protect", True))
    quote_require_both_sides = bool(params.get("backtest_quote_require_both_sides", True))
    quote_max_spread_bps = max(0.0, _safe_float(params.get("backtest_quote_max_spread_bps"), default=80.0))
    quote_max_deviation_bps = max(0.0, _safe_float(params.get("backtest_quote_max_deviation_bps"), default=120.0))
    quote_max_spread_bps_stop = max(
        quote_max_spread_bps,
        _safe_float(params.get("backtest_quote_max_spread_bps_stop"), default=quote_max_spread_bps),
    )
    quote_max_deviation_bps_stop = max(
        quote_max_deviation_bps,
        _safe_float(params.get("backtest_quote_max_deviation_bps_stop"), default=quote_max_deviation_bps),
    )
    quote_debug = bool(params.get("backtest_quote_debug", False))
    quote_debug_max_logs = max(0, _safe_int(params.get("backtest_quote_debug_max_logs"), default=40))
    slot_distribution_enabled = bool(params.get("slot_distribution_enabled", False))
    watch_cfg = cfg.get("watchlist") or {}
    slot_lookback_days = float(watch_cfg.get("lookback_days") or 252)
    slot_min_slots = int(params.get("slot_distribution_min_slots") or 1)
    slot_max_slots = int(params.get("slot_distribution_max_slots") or 0)
    used_notional = 0.0
    accepted: List = []
    sized_records: List[Dict] = []

    if fill_model_enabled:
        global _FILL_MODEL_LOGGED
        if not _FILL_MODEL_LOGGED:
            logging.info(
                "[BACKTEST_FILL] enabled=1 half_spread_bps=%.3f entry_slip_bps=%.3f stop_slip_bps=%.3f target_slip_bps=%.3f time_slip_bps=%.3f debug=%s max_debug_logs=%s",
                fill_half_spread_bps,
                fill_entry_slippage_bps,
                fill_stop_slippage_bps,
                fill_target_slippage_bps,
                fill_time_slippage_bps,
                fill_debug,
                fill_debug_max_logs,
            )
            _FILL_MODEL_LOGGED = True
    if quote_fill_enabled:
        global _QUOTE_MODEL_LOGGED
        if not _QUOTE_MODEL_LOGGED:
            logging.info(
                "[BACKTEST_QUOTE_FILL] enabled=1 lookback_s=%s forward_s=%s max_age_s=%s require_both_sides=%s max_spread_bps=%.2f max_deviation_bps=%.2f max_spread_bps_stop=%.2f max_deviation_bps_stop=%.2f fallback_to_bps=%s target_limit_protect=%s debug=%s max_debug_logs=%s",
                quote_lookback_seconds,
                quote_forward_seconds,
                quote_max_age_seconds,
                quote_require_both_sides,
                quote_max_spread_bps,
                quote_max_deviation_bps,
                quote_max_spread_bps_stop,
                quote_max_deviation_bps_stop,
                quote_fallback_to_bps,
                quote_target_limit_protect,
                quote_debug,
                quote_debug_max_logs,
            )
            _QUOTE_MODEL_LOGGED = True

    def _parse_ts_iso(val: object) -> Optional[dt.datetime]:
        if val is None:
            return None
        ts = str(val).strip()
        if not ts:
            return None
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        try:
            return ensure_et(dt.datetime.fromisoformat(ts))
        except Exception:
            return None

    def _entry_dt(plan_obj) -> Optional[dt.datetime]:
        try:
            return ensure_et(
                dt.datetime.combine(
                    dt.date.fromisoformat(str(plan_obj.entry_date)),
                    parse_time_hhmm(str(plan_obj.entry_time_et or "09:35")),
                )
            )
        except Exception:
            return None

    def _fallback_exit_dt(trade_obj, entry_dt_obj: dt.datetime) -> dt.datetime:
        try:
            exit_day = dt.date.fromisoformat(str(getattr(trade_obj, "exit_date", "") or entry_dt_obj.date().isoformat()))
            return ensure_et(dt.datetime.combine(exit_day, parse_time_hhmm(session_close_et)))
        except Exception:
            return entry_dt_obj

    def _entry_side(direction: str) -> str:
        return "buy" if str(direction).lower() == "long" else "sell"

    def _exit_side(direction: str) -> str:
        return "sell" if str(direction).lower() == "long" else "buy"

    def _exit_slippage_bps(exit_reason: str) -> Tuple[float, str]:
        reason = str(exit_reason or "").lower()
        if reason == "stop":
            return fill_stop_slippage_bps, "stop"
        if reason == "target":
            return fill_target_slippage_bps, "target"
        return fill_time_slippage_bps, "time"

    def _side_price_bps(raw_price: float, fill_price: float, side: str) -> float:
        raw_px = _safe_float(raw_price, default=0.0)
        fill_px = _safe_float(fill_price, default=0.0)
        if raw_px <= 0 or fill_px <= 0:
            return 0.0
        if str(side).lower() == "buy":
            return ((fill_px / raw_px) - 1.0) * 10000.0
        return ((raw_px - fill_px) / raw_px) * 10000.0

    def _resolve_quote_leg(
        symbol: str,
        target_ts: Optional[dt.datetime],
        side: str,
        raw_reference_price: float,
        *,
        max_spread_bps: float,
        max_deviation_bps: float,
    ) -> Dict[str, Any]:
        if not quote_fill_enabled:
            return {"ok": False, "reason": "quote_disabled"}
        if target_ts is None:
            return {"ok": False, "reason": "missing_ts"}
        quote = resolve_quote_for_timestamp(
            symbol,
            target_ts,
            cfg,
            lookback_seconds=quote_lookback_seconds,
            forward_seconds=quote_forward_seconds,
            allow_fetch=True,
        )
        if not quote:
            return {"ok": False, "reason": "no_quote"}
        quote_ts = _parse_ts_iso(quote.get("timestamp"))
        if quote_ts is None:
            return {"ok": False, "reason": "bad_quote_ts"}
        age_seconds = abs((quote_ts - ensure_et(target_ts)).total_seconds())
        if age_seconds > float(quote_max_age_seconds):
            return {
                "ok": False,
                "reason": "quote_too_old",
                "quote_ts": quote.get("timestamp"),
                "age_seconds": age_seconds,
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "selection": quote.get("selected"),
            }
        bid = quote.get("bid")
        ask = quote.get("ask")
        bid = None if bid is None else _safe_float(bid, default=0.0)
        ask = None if ask is None else _safe_float(ask, default=0.0)
        if quote_require_both_sides and ((bid is None or bid <= 0) or (ask is None or ask <= 0)):
            return {
                "ok": False,
                "reason": "missing_bid_or_ask",
                "quote_ts": quote.get("timestamp"),
                "bid": bid,
                "ask": ask,
                "selection": quote.get("selected"),
            }
        spread_bps = None
        if bid is not None and bid > 0 and ask is not None and ask > 0:
            if ask < bid:
                return {
                    "ok": False,
                    "reason": "crossed_quote",
                    "quote_ts": quote.get("timestamp"),
                    "bid": bid,
                    "ask": ask,
                    "selection": quote.get("selected"),
                }
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid) * 10000.0 if mid > 0 else None
            if spread_bps is not None and spread_bps > float(max_spread_bps):
                return {
                    "ok": False,
                    "reason": "spread_too_wide",
                    "quote_ts": quote.get("timestamp"),
                    "bid": bid,
                    "ask": ask,
                    "spread_bps": spread_bps,
                    "selection": quote.get("selected"),
                }
        px = None
        px_src = None
        side_l = str(side).lower()
        if side_l == "buy":
            if ask is not None and ask > 0:
                px = ask
                px_src = "ask"
            elif bid is not None and bid > 0:
                px = bid
                px_src = "bid_fallback"
        else:
            if bid is not None and bid > 0:
                px = bid
                px_src = "bid"
            elif ask is not None and ask > 0:
                px = ask
                px_src = "ask_fallback"
        if px is None or px <= 0:
            return {
                "ok": False,
                "reason": "no_side_price",
                "quote_ts": quote.get("timestamp"),
                "bid": bid,
                "ask": ask,
                "spread_bps": spread_bps,
                "selection": quote.get("selected"),
            }
        raw_ref = _safe_float(raw_reference_price, default=0.0)
        deviation_bps = (abs(px - raw_ref) / raw_ref * 10000.0) if raw_ref > 0 else None
        if deviation_bps is not None and deviation_bps > float(max_deviation_bps):
            return {
                "ok": False,
                "reason": "deviation_too_large",
                "quote_ts": quote.get("timestamp"),
                "bid": bid,
                "ask": ask,
                "spread_bps": spread_bps,
                "deviation_bps": deviation_bps,
                "selection": quote.get("selected"),
            }
        return {
            "ok": True,
            "price": px,
            "price_source": px_src,
            "quote_ts": quote.get("timestamp"),
            "selection": quote.get("selected"),
            "age_seconds": age_seconds,
            "spread_bps": spread_bps,
            "deviation_bps": deviation_bps,
            "bid": bid,
            "ask": ask,
        }

    def _target_limit_bound(direction: str, exit_reason: str, exit_price: float, target_price: float) -> Tuple[float, bool]:
        if (not quote_target_limit_protect) or str(exit_reason or "").lower() != "target":
            return exit_price, False
        d = str(direction or "").lower()
        px = _safe_float(exit_price, default=0.0)
        tgt = _safe_float(target_price, default=0.0)
        if px <= 0 or tgt <= 0:
            return exit_price, False
        if d == "long":
            bounded = max(px, tgt)
        else:
            bounded = min(px, tgt)
        return bounded, (abs(bounded - px) > 1e-12)

    def _compute_fill_meta(trade_obj, plan_obj, qty: int) -> Dict[str, Any]:
        raw_entry = _safe_float(getattr(plan_obj, "entry_price", None), default=0.0)
        raw_exit = _safe_float(getattr(trade_obj, "exit_price", None), default=0.0)
        direction = str(getattr(plan_obj, "direction", "long")).lower()
        direction_mult = 1.0 if direction == "long" else -1.0
        stop_distance = max(0.0, _safe_float(getattr(plan_obj, "stop_distance", None), default=0.0))

        entry_side = _entry_side(direction)
        exit_side = _exit_side(direction)
        exit_slip_bps, exit_bucket = _exit_slippage_bps(str(getattr(trade_obj, "exit_reason", "") or ""))
        entry_bps_model = fill_half_spread_bps + fill_entry_slippage_bps
        exit_bps_model = fill_half_spread_bps + exit_slip_bps
        use_bps_base = bool(fill_model_enabled or (quote_fill_enabled and quote_fallback_to_bps))
        entry_bps = entry_bps_model if use_bps_base else 0.0
        exit_bps = exit_bps_model if use_bps_base else 0.0
        entry_fill = _adverse_fill_price(raw_entry, side=entry_side, bps=entry_bps) if use_bps_base else raw_entry
        exit_fill = _adverse_fill_price(raw_exit, side=exit_side, bps=exit_bps) if use_bps_base else raw_exit
        entry_fill_mode = "bps" if use_bps_base else "raw"
        exit_fill_mode = "bps" if use_bps_base else "raw"
        entry_quote_meta: Dict[str, Any] = {"ok": False, "reason": "quote_disabled"}
        exit_quote_meta: Dict[str, Any] = {"ok": False, "reason": "quote_disabled"}
        entry_ts = _entry_dt(plan_obj)
        exit_ts = _parse_ts_iso(getattr(trade_obj, "exit_ts", None))
        if exit_ts is None and entry_ts is not None:
            exit_ts = _fallback_exit_dt(trade_obj, entry_ts)

        if quote_fill_enabled:
            entry_quote_meta = _resolve_quote_leg(
                str(getattr(plan_obj, "symbol", "") or ""),
                entry_ts,
                entry_side,
                raw_entry,
                max_spread_bps=quote_max_spread_bps,
                max_deviation_bps=quote_max_deviation_bps,
            )
            exit_reason_l = str(getattr(trade_obj, "exit_reason", "") or "").lower()
            exit_spread_cap = quote_max_spread_bps_stop if exit_reason_l == "stop" else quote_max_spread_bps
            exit_dev_cap = quote_max_deviation_bps_stop if exit_reason_l == "stop" else quote_max_deviation_bps
            exit_quote_meta = _resolve_quote_leg(
                str(getattr(plan_obj, "symbol", "") or ""),
                exit_ts,
                exit_side,
                raw_exit,
                max_spread_bps=exit_spread_cap,
                max_deviation_bps=exit_dev_cap,
            )
            if bool(entry_quote_meta.get("ok")):
                entry_fill = _safe_float(entry_quote_meta.get("price"), default=entry_fill)
                entry_fill_mode = "quote"
            elif not quote_fallback_to_bps:
                entry_fill = raw_entry
                entry_fill_mode = "raw_quote_missing"
            if bool(exit_quote_meta.get("ok")):
                exit_fill = _safe_float(exit_quote_meta.get("price"), default=exit_fill)
                exit_fill_mode = "quote"
            elif not quote_fallback_to_bps:
                exit_fill = raw_exit
                exit_fill_mode = "raw_quote_missing"
            exit_fill, target_limit_bound_applied = _target_limit_bound(
                direction,
                str(getattr(trade_obj, "exit_reason", "") or ""),
                exit_fill,
                _safe_float(getattr(plan_obj, "target_price", None), default=0.0),
            )
        else:
            target_limit_bound_applied = False
            exit_spread_cap = quote_max_spread_bps
            exit_dev_cap = quote_max_deviation_bps
        raw_pnl_per_share = (raw_exit - raw_entry) * direction_mult
        fill_pnl_per_share = (exit_fill - entry_fill) * direction_mult
        raw_pnl_pct = (raw_pnl_per_share / raw_entry * 100.0) if raw_entry > 0 else 0.0
        fill_pnl_pct = (fill_pnl_per_share / entry_fill * 100.0) if entry_fill > 0 else 0.0
        raw_r = (raw_pnl_per_share / stop_distance) if stop_distance > 0 else 0.0
        fill_r = (fill_pnl_per_share / stop_distance) if stop_distance > 0 else 0.0
        fill_cost_per_share = raw_pnl_per_share - fill_pnl_per_share
        fill_cost_total = fill_cost_per_share * max(0, int(qty))
        stop_price = _safe_float(getattr(plan_obj, "stop_price", None), default=0.0)
        stop_distance_fill = abs(entry_fill - stop_price) if stop_price > 0 else None
        fill_entry_bps_eff = _side_price_bps(raw_entry, entry_fill, entry_side)
        fill_exit_bps_eff = _side_price_bps(raw_exit, exit_fill, exit_side)

        return {
            "fill_model_enabled": fill_model_enabled,
            "quote_fill_enabled": quote_fill_enabled,
            "quote_fallback_to_bps": quote_fallback_to_bps,
            "entry_fill_mode": entry_fill_mode,
            "exit_fill_mode": exit_fill_mode,
            "entry_quote_ok": bool(entry_quote_meta.get("ok")),
            "exit_quote_ok": bool(exit_quote_meta.get("ok")),
            "entry_quote_spread_cap_bps": quote_max_spread_bps if quote_fill_enabled else None,
            "entry_quote_deviation_cap_bps": quote_max_deviation_bps if quote_fill_enabled else None,
            "exit_quote_spread_cap_bps": exit_spread_cap if quote_fill_enabled else None,
            "exit_quote_deviation_cap_bps": exit_dev_cap if quote_fill_enabled else None,
            "entry_quote_reason": entry_quote_meta.get("reason"),
            "exit_quote_reason": exit_quote_meta.get("reason"),
            "entry_quote_ts": entry_quote_meta.get("quote_ts"),
            "exit_quote_ts": exit_quote_meta.get("quote_ts"),
            "entry_quote_selection": entry_quote_meta.get("selection"),
            "exit_quote_selection": exit_quote_meta.get("selection"),
            "entry_quote_age_seconds": entry_quote_meta.get("age_seconds"),
            "exit_quote_age_seconds": exit_quote_meta.get("age_seconds"),
            "entry_quote_spread_bps": entry_quote_meta.get("spread_bps"),
            "exit_quote_spread_bps": exit_quote_meta.get("spread_bps"),
            "entry_quote_deviation_bps": entry_quote_meta.get("deviation_bps"),
            "exit_quote_deviation_bps": exit_quote_meta.get("deviation_bps"),
            "entry_quote_bid": entry_quote_meta.get("bid"),
            "entry_quote_ask": entry_quote_meta.get("ask"),
            "exit_quote_bid": exit_quote_meta.get("bid"),
            "exit_quote_ask": exit_quote_meta.get("ask"),
            "entry_quote_price_source": entry_quote_meta.get("price_source"),
            "exit_quote_price_source": exit_quote_meta.get("price_source"),
            "target_limit_bound_applied": target_limit_bound_applied,
            "entry_side": entry_side,
            "exit_side": exit_side,
            "fill_exit_reason_bucket": exit_bucket,
            "raw_entry_price": raw_entry,
            "raw_exit_price": raw_exit,
            "entry_fill_price": entry_fill,
            "exit_fill_price": exit_fill,
            "fill_half_spread_bps": fill_half_spread_bps if use_bps_base else 0.0,
            "fill_entry_slippage_bps": fill_entry_slippage_bps if use_bps_base else 0.0,
            "fill_exit_slippage_bps": exit_slip_bps if use_bps_base else 0.0,
            "fill_entry_bps": fill_entry_bps_eff,
            "fill_exit_bps": fill_exit_bps_eff,
            "fill_entry_bps_model": entry_bps_model if use_bps_base else 0.0,
            "fill_exit_bps_model": exit_bps_model if use_bps_base else 0.0,
            "raw_pnl_per_share": raw_pnl_per_share,
            "pnl_per_share": fill_pnl_per_share,
            "raw_pnl_pct": raw_pnl_pct,
            "pnl_pct": fill_pnl_pct,
            "raw_r_multiple": raw_r,
            "r_multiple": fill_r,
            "fill_cost_per_share": fill_cost_per_share,
            "fill_cost_total": fill_cost_total,
            "stop_distance_fill": stop_distance_fill,
        }

    def _apply_fill_to_trade(trade_obj, meta: Dict[str, Any]) -> None:
        trade_obj.raw_exit_price = meta.get("raw_exit_price")
        trade_obj.raw_pnl_pct = meta.get("raw_pnl_pct")
        trade_obj.raw_r_multiple = meta.get("raw_r_multiple")
        trade_obj.entry_fill_price = meta.get("entry_fill_price")
        trade_obj.exit_fill_price = meta.get("exit_fill_price")
        trade_obj.entry_fill_mode = meta.get("entry_fill_mode")
        trade_obj.exit_fill_mode = meta.get("exit_fill_mode")
        trade_obj.quote_fill_enabled = bool(meta.get("quote_fill_enabled"))
        trade_obj.entry_quote_ok = bool(meta.get("entry_quote_ok"))
        trade_obj.exit_quote_ok = bool(meta.get("exit_quote_ok"))
        trade_obj.entry_quote_spread_cap_bps = meta.get("entry_quote_spread_cap_bps")
        trade_obj.entry_quote_deviation_cap_bps = meta.get("entry_quote_deviation_cap_bps")
        trade_obj.exit_quote_spread_cap_bps = meta.get("exit_quote_spread_cap_bps")
        trade_obj.exit_quote_deviation_cap_bps = meta.get("exit_quote_deviation_cap_bps")
        trade_obj.entry_quote_reason = meta.get("entry_quote_reason")
        trade_obj.exit_quote_reason = meta.get("exit_quote_reason")
        trade_obj.entry_quote_ts = meta.get("entry_quote_ts")
        trade_obj.exit_quote_ts = meta.get("exit_quote_ts")
        trade_obj.entry_quote_selection = meta.get("entry_quote_selection")
        trade_obj.exit_quote_selection = meta.get("exit_quote_selection")
        trade_obj.entry_quote_age_seconds = meta.get("entry_quote_age_seconds")
        trade_obj.exit_quote_age_seconds = meta.get("exit_quote_age_seconds")
        trade_obj.entry_quote_spread_bps = meta.get("entry_quote_spread_bps")
        trade_obj.exit_quote_spread_bps = meta.get("exit_quote_spread_bps")
        trade_obj.entry_quote_deviation_bps = meta.get("entry_quote_deviation_bps")
        trade_obj.exit_quote_deviation_bps = meta.get("exit_quote_deviation_bps")
        trade_obj.entry_quote_bid = meta.get("entry_quote_bid")
        trade_obj.entry_quote_ask = meta.get("entry_quote_ask")
        trade_obj.exit_quote_bid = meta.get("exit_quote_bid")
        trade_obj.exit_quote_ask = meta.get("exit_quote_ask")
        trade_obj.entry_quote_price_source = meta.get("entry_quote_price_source")
        trade_obj.exit_quote_price_source = meta.get("exit_quote_price_source")
        trade_obj.target_limit_bound_applied = bool(meta.get("target_limit_bound_applied"))
        trade_obj.fill_entry_bps = meta.get("fill_entry_bps")
        trade_obj.fill_exit_bps = meta.get("fill_exit_bps")
        trade_obj.fill_entry_bps_model = meta.get("fill_entry_bps_model")
        trade_obj.fill_exit_bps_model = meta.get("fill_exit_bps_model")
        trade_obj.fill_half_spread_bps = meta.get("fill_half_spread_bps")
        trade_obj.fill_entry_slippage_bps = meta.get("fill_entry_slippage_bps")
        trade_obj.fill_exit_slippage_bps = meta.get("fill_exit_slippage_bps")
        trade_obj.fill_cost_per_share = meta.get("fill_cost_per_share")
        trade_obj.fill_cost_total = meta.get("fill_cost_total")
        trade_obj.stop_distance_fill = meta.get("stop_distance_fill")
        trade_obj.fill_model_enabled = bool(meta.get("fill_model_enabled"))
        trade_obj.fill_exit_reason_bucket = meta.get("fill_exit_reason_bucket")
        trade_obj.exit_price = _safe_float(meta.get("exit_fill_price"), default=_safe_float(trade_obj.exit_price, default=0.0))
        trade_obj.pnl_pct = _safe_float(meta.get("pnl_pct"), default=_safe_float(trade_obj.pnl_pct, default=0.0))
        trade_obj.r_multiple = _safe_float(meta.get("r_multiple"), default=_safe_float(trade_obj.r_multiple, default=0.0))

    def _log_fill_debug(plan_obj, trade_obj, qty: int, meta: Dict[str, Any]) -> None:
        if not (fill_model_enabled and fill_debug):
            return
        global _FILL_MODEL_DEBUG_LOGS
        if _FILL_MODEL_DEBUG_LOGS >= fill_debug_max_logs:
            return
        _FILL_MODEL_DEBUG_LOGS += 1
        logging.info(
            "[BACKTEST_FILL_DEBUG] symbol=%s date=%s side=%s reason=%s qty=%s entry_mode=%s exit_mode=%s raw_entry=%.5f fill_entry=%.5f raw_exit=%.5f fill_exit=%.5f raw_r=%.4f fill_r=%.4f cost_ps=%.5f cost_total=%.2f entry_bps=%.2f exit_bps=%.2f model_entry_bps=%.2f model_exit_bps=%.2f",
            str(getattr(plan_obj, "symbol", "") or ""),
            str(getattr(plan_obj, "entry_date", "") or ""),
            str(getattr(plan_obj, "direction", "") or ""),
            str(getattr(trade_obj, "exit_reason", "") or ""),
            int(max(0, int(qty))),
            str(meta.get("entry_fill_mode") or ""),
            str(meta.get("exit_fill_mode") or ""),
            _safe_float(meta.get("raw_entry_price"), default=0.0),
            _safe_float(meta.get("entry_fill_price"), default=0.0),
            _safe_float(meta.get("raw_exit_price"), default=0.0),
            _safe_float(meta.get("exit_fill_price"), default=0.0),
            _safe_float(meta.get("raw_r_multiple"), default=0.0),
            _safe_float(meta.get("r_multiple"), default=0.0),
            _safe_float(meta.get("fill_cost_per_share"), default=0.0),
            _safe_float(meta.get("fill_cost_total"), default=0.0),
            _safe_float(meta.get("fill_entry_bps"), default=0.0),
            _safe_float(meta.get("fill_exit_bps"), default=0.0),
            _safe_float(meta.get("fill_entry_bps_model"), default=0.0),
            _safe_float(meta.get("fill_exit_bps_model"), default=0.0),
        )

    def _log_quote_debug(plan_obj, trade_obj, qty: int, meta: Dict[str, Any]) -> None:
        if not (quote_fill_enabled and quote_debug):
            return
        global _QUOTE_MODEL_DEBUG_LOGS
        if _QUOTE_MODEL_DEBUG_LOGS >= quote_debug_max_logs:
            return
        _QUOTE_MODEL_DEBUG_LOGS += 1
        logging.info(
            "[BACKTEST_QUOTE_DEBUG] symbol=%s date=%s reason=%s qty=%s entry_ok=%s entry_mode=%s entry_qts=%s entry_sel=%s entry_bid=%.5f entry_ask=%.5f entry_src=%s entry_age=%.2fs entry_spread_bps=%.2f entry_dev_bps=%.2f exit_ok=%s exit_mode=%s exit_qts=%s exit_sel=%s exit_bid=%.5f exit_ask=%.5f exit_src=%s exit_age=%.2fs exit_spread_bps=%.2f exit_dev_bps=%.2f target_limit_bound=%s fallback_to_bps=%s",
            str(getattr(plan_obj, "symbol", "") or ""),
            str(getattr(plan_obj, "entry_date", "") or ""),
            str(getattr(trade_obj, "exit_reason", "") or ""),
            int(max(0, int(qty))),
            bool(meta.get("entry_quote_ok")),
            str(meta.get("entry_fill_mode") or ""),
            str(meta.get("entry_quote_ts") or ""),
            str(meta.get("entry_quote_selection") or ""),
            _safe_float(meta.get("entry_quote_bid"), default=0.0),
            _safe_float(meta.get("entry_quote_ask"), default=0.0),
            str(meta.get("entry_quote_price_source") or meta.get("entry_quote_reason") or ""),
            _safe_float(meta.get("entry_quote_age_seconds"), default=0.0),
            _safe_float(meta.get("entry_quote_spread_bps"), default=0.0),
            _safe_float(meta.get("entry_quote_deviation_bps"), default=0.0),
            bool(meta.get("exit_quote_ok")),
            str(meta.get("exit_fill_mode") or ""),
            str(meta.get("exit_quote_ts") or ""),
            str(meta.get("exit_quote_selection") or ""),
            _safe_float(meta.get("exit_quote_bid"), default=0.0),
            _safe_float(meta.get("exit_quote_ask"), default=0.0),
            str(meta.get("exit_quote_price_source") or meta.get("exit_quote_reason") or ""),
            _safe_float(meta.get("exit_quote_age_seconds"), default=0.0),
            _safe_float(meta.get("exit_quote_spread_bps"), default=0.0),
            _safe_float(meta.get("exit_quote_deviation_bps"), default=0.0),
            bool(meta.get("target_limit_bound_applied")),
            quote_fallback_to_bps,
        )

    def _build_sized_record(
        trade_obj,
        plan_obj,
        qty: int,
        state: Dict,
        pnl_total: float,
        notional: float,
        capacity_notional: float,
        fill_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        fill_meta = fill_meta or {}
        wl_stats = (
            getattr(plan_obj, "watchlist_stats", None)
            if isinstance(getattr(plan_obj, "watchlist_stats", None), dict)
            else {}
        )
        quality_state = state.get("quality_sizing") if isinstance(state, dict) else {}
        record = {
            "symbol": plan_obj.symbol,
            "param_overrides": getattr(plan_obj, "param_overrides", None),
            "direction": plan_obj.direction,
            "entry_date": plan_obj.entry_date,
            "entry_time_et": plan_obj.entry_time_et,
            "entry_price": plan_obj.entry_price,
            "entry_price_mode": getattr(plan_obj, "entry_price_mode", None),
            "stop_price": plan_obj.stop_price,
            "target_price": plan_obj.target_price,
            "stop_distance": plan_obj.stop_distance,
            "raw_entry_price": fill_meta.get("raw_entry_price", plan_obj.entry_price),
            "raw_exit_price": fill_meta.get("raw_exit_price", getattr(trade_obj, "exit_price", None)),
            "entry_fill_price": fill_meta.get("entry_fill_price", plan_obj.entry_price),
            "exit_fill_price": fill_meta.get("exit_fill_price", getattr(trade_obj, "exit_price", None)),
            "stop_distance_fill": fill_meta.get("stop_distance_fill"),
            "fill_model_enabled": bool(fill_meta.get("fill_model_enabled", False)),
            "quote_fill_enabled": bool(fill_meta.get("quote_fill_enabled", False)),
            "entry_fill_mode": fill_meta.get("entry_fill_mode"),
            "exit_fill_mode": fill_meta.get("exit_fill_mode"),
            "fill_half_spread_bps": fill_meta.get("fill_half_spread_bps", 0.0),
            "fill_entry_slippage_bps": fill_meta.get("fill_entry_slippage_bps", 0.0),
            "fill_exit_slippage_bps": fill_meta.get("fill_exit_slippage_bps", 0.0),
            "fill_entry_bps": fill_meta.get("fill_entry_bps", 0.0),
            "fill_exit_bps": fill_meta.get("fill_exit_bps", 0.0),
            "fill_entry_bps_model": fill_meta.get("fill_entry_bps_model", 0.0),
            "fill_exit_bps_model": fill_meta.get("fill_exit_bps_model", 0.0),
            "fill_exit_reason_bucket": fill_meta.get("fill_exit_reason_bucket"),
            "fill_cost_per_share": fill_meta.get("fill_cost_per_share", 0.0),
            "fill_cost_total": fill_meta.get("fill_cost_total", 0.0),
            "entry_quote_ok": bool(fill_meta.get("entry_quote_ok", False)),
            "exit_quote_ok": bool(fill_meta.get("exit_quote_ok", False)),
            "entry_quote_spread_cap_bps": fill_meta.get("entry_quote_spread_cap_bps"),
            "entry_quote_deviation_cap_bps": fill_meta.get("entry_quote_deviation_cap_bps"),
            "exit_quote_spread_cap_bps": fill_meta.get("exit_quote_spread_cap_bps"),
            "exit_quote_deviation_cap_bps": fill_meta.get("exit_quote_deviation_cap_bps"),
            "entry_quote_reason": fill_meta.get("entry_quote_reason"),
            "exit_quote_reason": fill_meta.get("exit_quote_reason"),
            "entry_quote_ts": fill_meta.get("entry_quote_ts"),
            "exit_quote_ts": fill_meta.get("exit_quote_ts"),
            "entry_quote_selection": fill_meta.get("entry_quote_selection"),
            "exit_quote_selection": fill_meta.get("exit_quote_selection"),
            "entry_quote_age_seconds": fill_meta.get("entry_quote_age_seconds"),
            "exit_quote_age_seconds": fill_meta.get("exit_quote_age_seconds"),
            "entry_quote_spread_bps": fill_meta.get("entry_quote_spread_bps"),
            "exit_quote_spread_bps": fill_meta.get("exit_quote_spread_bps"),
            "entry_quote_deviation_bps": fill_meta.get("entry_quote_deviation_bps"),
            "exit_quote_deviation_bps": fill_meta.get("exit_quote_deviation_bps"),
            "entry_quote_bid": fill_meta.get("entry_quote_bid"),
            "entry_quote_ask": fill_meta.get("entry_quote_ask"),
            "exit_quote_bid": fill_meta.get("exit_quote_bid"),
            "exit_quote_ask": fill_meta.get("exit_quote_ask"),
            "entry_quote_price_source": fill_meta.get("entry_quote_price_source"),
            "exit_quote_price_source": fill_meta.get("exit_quote_price_source"),
            "target_limit_bound_applied": bool(fill_meta.get("target_limit_bound_applied", False)),
            "raw_pnl_pct": fill_meta.get("raw_pnl_pct", getattr(trade_obj, "pnl_pct", None)),
            "raw_r_multiple": fill_meta.get("raw_r_multiple", getattr(trade_obj, "r_multiple", None)),
            "target_mode": getattr(plan_obj, "target_mode", None),
            "target_window_avg_pct": getattr(plan_obj, "target_window_avg_pct", None),
            "target_window_mult": getattr(plan_obj, "target_window_mult", None),
            "target_window_minutes": getattr(plan_obj, "target_window_minutes", None),
            "target_window_samples": getattr(plan_obj, "target_window_samples", None),
            "gap_bps": getattr(plan_obj, "gap_bps", None),
            "early_pullback_bps": getattr(plan_obj, "early_pullback_bps", None),
            "early_reversal_bps": getattr(plan_obj, "early_reversal_bps", None),
            "confirm_move_bps": getattr(plan_obj, "confirm_move_bps", None),
            "confirm_minutes": getattr(plan_obj, "confirm_minutes", None),
            "confirm_hit_bps": getattr(plan_obj, "confirm_hit_bps", None),
            "signal_return_pct": getattr(plan_obj, "signal_return_pct", None),
            "signal_return_atr": getattr(plan_obj, "signal_return_atr", None),
            "atr": getattr(plan_obj, "atr", None),
            "watchlist_rank": wl_stats.get("rank"),
            "watchlist_avgR": wl_stats.get("avgR"),
            "watchlist_avgR_stderr": wl_stats.get("avgR_stderr"),
            "watchlist_win_rate": wl_stats.get("win_rate"),
            "watchlist_profit_factor": wl_stats.get("profit_factor"),
            "watchlist_trades_count": wl_stats.get("trades_count"),
            "watchlist_total_pnl_pct": wl_stats.get("total_pnl_pct"),
            "watchlist_stats": wl_stats,
            "exit_date": trade_obj.exit_date,
            "exit_price": trade_obj.exit_price,
            "exit_reason": trade_obj.exit_reason,
            "exit_ts": getattr(trade_obj, "exit_ts", None),
            "stop_hit_ts": getattr(trade_obj, "stop_hit_ts", None),
            "target_hit_ts": getattr(trade_obj, "target_hit_ts", None),
            "qty": qty,
            "pnl_total": pnl_total,
            "pnl_pct": trade_obj.pnl_pct,
            "r_multiple": trade_obj.r_multiple,
            "quality_risk_mult": (quality_state or {}).get("risk_multiplier"),
            "quality_score": (quality_state or {}).get("score"),
            "mfe_pct": getattr(trade_obj, "mfe_pct", None),
            "mae_pct": getattr(trade_obj, "mae_pct", None),
            "mfe_r": getattr(trade_obj, "mfe_r", None),
            "mae_r": getattr(trade_obj, "mae_r", None),
            "mfe_r_full": getattr(trade_obj, "mfe_r_full", None),
            "mae_r_full": getattr(trade_obj, "mae_r_full", None),
            "mfe_r_before_stop": getattr(trade_obj, "mfe_r_before_stop", None),
            "mae_r_to_target": getattr(trade_obj, "mae_r_to_target", None),
            "equity_before": None,
            "equity_after": None,
            "notional": notional,
            "capacity_notional": capacity_notional,
            "sizing_state": state,
        }
        for key, value in wl_stats.items():
            prefixed = f"watchlist_{str(key)}"
            if prefixed not in record:
                record[prefixed] = value
        return record

    if not intraday_only:
        # Legacy path for multi-day holds: realized PnL is applied in sequence.
        open_positions = 0
        for idx, trade in enumerate(day_trades):
            plan = getattr(trade, "plan", None)
            if plan is None:
                continue
            slot_target = None
            if slot_distribution_enabled:
                stats_rows = []
                for pending in day_trades[idx:]:
                    p = getattr(pending, "plan", None)
                    s = getattr(p, "watchlist_stats", None) if p is not None else None
                    if isinstance(s, dict):
                        stats_rows.append(s)
                target, _ = estimate_slot_target_from_stats(
                    stats_rows,
                    slot_lookback_days,
                    min_slots=slot_min_slots,
                    max_slots=slot_max_slots,
                    cap_by_candidates=True,
                )
                if target > 0:
                    slot_target = target
            qty, state = compute_qty_with_guards(
                plan,
                equity,
                used_notional,
                cfg,
                open_positions=open_positions,
                slot_target_override=slot_target,
            )
            if qty <= 0:
                continue
            fill_meta = _compute_fill_meta(trade, plan, qty)
            _apply_fill_to_trade(trade, fill_meta)
            pnl_total = _safe_float(fill_meta.get("pnl_per_share"), default=0.0) * qty
            equity_before = equity
            equity = equity + pnl_total
            entry_fill_price = _safe_float(fill_meta.get("entry_fill_price"), default=_safe_float(plan.entry_price, default=0.0))
            notional = entry_fill_price * qty
            capacity_qty = int((state or {}).get("capacity_qty") or qty)
            capacity_notional = entry_fill_price * capacity_qty
            used_notional += capacity_notional
            open_positions += 1
            accepted.append(trade)
            rec = _build_sized_record(
                trade,
                plan,
                qty,
                state,
                pnl_total,
                notional,
                capacity_notional,
                fill_meta=fill_meta,
            )
            rec["equity_before"] = equity_before
            rec["equity_after"] = equity
            sized_records.append(rec)
            _log_fill_debug(plan, trade, qty, fill_meta)
            _log_quote_debug(plan, trade, qty, fill_meta)
        return accepted, sized_records, equity

    work: List[Dict] = []
    for seq, trade in enumerate(day_trades):
        plan = getattr(trade, "plan", None)
        if plan is None:
            continue
        entry_dt_obj = _entry_dt(plan)
        if entry_dt_obj is None:
            continue
        exit_dt_obj = _parse_ts_iso(getattr(trade, "exit_ts", None))
        if exit_dt_obj is None:
            exit_dt_obj = _fallback_exit_dt(trade, entry_dt_obj)
        work.append(
            {
                "seq": seq,
                "trade": trade,
                "plan": plan,
                "entry_dt": entry_dt_obj,
                "exit_dt": exit_dt_obj,
            }
        )
    work.sort(key=lambda r: (r["entry_dt"], r["seq"]))

    open_positions_state: List[Dict] = []

    def _realize_until(cutoff_dt: dt.datetime, include_equal: bool = False) -> None:
        nonlocal equity, used_notional, open_positions_state
        if not open_positions_state:
            return
        still_open: List[Dict] = []
        for pos in open_positions_state:
            should_close = pos["exit_dt"] <= cutoff_dt if include_equal else pos["exit_dt"] < cutoff_dt
            if should_close:
                rec = pos["record"]
                eq_before = equity
                equity += float(pos["pnl_total"])
                rec["equity_before"] = eq_before
                rec["equity_after"] = equity
                used_notional = max(0.0, used_notional - float(pos["capacity_notional"]))
            else:
                still_open.append(pos)
        open_positions_state = still_open

    for row_idx, row in enumerate(work):
        trade = row["trade"]
        plan = row["plan"]
        entry_dt_obj = row["entry_dt"]
        _realize_until(entry_dt_obj)
        slot_target = None
        if slot_distribution_enabled:
            stats_rows = []
            for pending in work[row_idx:]:
                s = getattr(pending.get("plan"), "watchlist_stats", None)
                if isinstance(s, dict):
                    stats_rows.append(s)
            target, _ = estimate_slot_target_from_stats(
                stats_rows,
                slot_lookback_days,
                min_slots=slot_min_slots,
                max_slots=slot_max_slots,
                cap_by_candidates=True,
            )
            if target > 0:
                slot_target = target
        qty, state = compute_qty_with_guards(
            plan,
            equity,
            used_notional,
            cfg,
            open_positions=len(open_positions_state),
            slot_target_override=slot_target,
        )
        if qty <= 0:
            continue
        fill_meta = _compute_fill_meta(trade, plan, qty)
        _apply_fill_to_trade(trade, fill_meta)
        pnl_total = _safe_float(fill_meta.get("pnl_per_share"), default=0.0) * qty
        entry_fill_price = _safe_float(fill_meta.get("entry_fill_price"), default=_safe_float(plan.entry_price, default=0.0))
        notional = entry_fill_price * qty
        capacity_qty = int((state or {}).get("capacity_qty") or qty)
        capacity_notional = entry_fill_price * capacity_qty
        used_notional += capacity_notional

        accepted.append(trade)
        rec = _build_sized_record(
            trade,
            plan,
            qty,
            state,
            pnl_total,
            notional,
            capacity_notional,
            fill_meta=fill_meta,
        )
        sized_records.append(rec)
        _log_fill_debug(plan, trade, qty, fill_meta)
        _log_quote_debug(plan, trade, qty, fill_meta)
        open_positions_state.append(
            {
                "exit_dt": row["exit_dt"],
                "capacity_notional": capacity_notional,
                "pnl_total": pnl_total,
                "record": rec,
            }
        )

    if open_positions_state:
        cutoff_points = sorted({pos["exit_dt"] for pos in open_positions_state})
        for cutoff in cutoff_points:
            _realize_until(cutoff, include_equal=True)
    return accepted, sized_records, equity


def run_backtest(
    cfg: Optional[Dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[Dict[str, float], List]:
    cfg = cfg or load_config()
    global _FILL_MODEL_LOGGED, _FILL_MODEL_DEBUG_LOGS, _QUOTE_MODEL_LOGGED, _QUOTE_MODEL_DEBUG_LOGS
    _FILL_MODEL_LOGGED = False
    _FILL_MODEL_DEBUG_LOGS = 0
    _QUOTE_MODEL_LOGGED = False
    _QUOTE_MODEL_DEBUG_LOGS = 0
    rep = cfg.get("replay") or {}
    start = start_date or rep.get("start_date")
    end = end_date or rep.get("end_date") or start
    if not start or not end:
        raise ValueError("start_date/end_date required (or set replay.start_date/end_date in config.json)")

    symbols, universe_source = resolve_asset_universe_symbols(cfg, target_date=start, allow_fetch=True)
    logging.info(
        "[BACKTEST] asset universe date=%s source=%s size=%s",
        start,
        universe_source,
        len(symbols),
    )

    data_store = AlpacaOHLCStore(cfg=cfg)
    # Prefetch historical bars once so daily scans don't hit Alpaca repeatedly.
    params = cfg.get("daily_trend_reversal") or {}
    trend_ma_days = int(params.get("trend_ma_days") or 200)
    atr_period = int(params.get("atr_period") or 14)
    pad_days = max(trend_ma_days, atr_period) * 2 + 10
    start_dt = dt.date.fromisoformat(start)
    prefetch_start = (start_dt - dt.timedelta(days=pad_days)).isoformat()
    logging.info("[BACKTEST] prefetching daily bars from %s to %s", prefetch_start, end)
    data_store.get_daily_bars_bulk(symbols, prefetch_start, end, cfg=cfg, allow_fetch=True)
    all_trades: List = []
    sized_trades: List[Dict] = []
    equity_curve: List[Dict] = []
    skip_count = 0
    skip_reasons: Dict[str, int] = {}
    params = cfg.get("daily_trend_reversal") or {}
    starting_equity = float(params.get("starting_equity") or 100000.0)
    equity = starting_equity
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    days_all = list(iter_trading_days(start, end))
    # iter_trading_days() only skips weekends. For parity (and speed), skip market holidays too by
    # filtering against a reference symbol's daily bars (market holidays are known ahead of time in live).
    calendar_cfg = cfg.get("market_calendar") or {}
    calendar_symbol = str(calendar_cfg.get("symbol") or "SPY").upper()
    try:
        cal_bars = data_store.get_daily_bars(calendar_symbol, prefetch_start, end, cfg=cfg, allow_fetch=True) or []
        cal_days = {str(b.get("date")) for b in cal_bars if b.get("date")}
        days = [d for d in days_all if d.isoformat() in cal_days]
        if len(days) != len(days_all):
            logging.info(
                "[BACKTEST] calendar filter symbol=%s trading_days=%s skipped_weekdays=%s",
                calendar_symbol,
                len(days),
                (len(days_all) - len(days)),
            )
    except Exception:
        days = days_all
    # Prepare output directory early for incremental flush.
    if out_path:
        out_file = Path(out_path)
        if out_file.suffix == "":
            out_file = out_file / run_id / "backtest.json"
    else:
        logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
        out_file = logs_dir / "backtests" / run_id / "backtest.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    trades_ndjson = out_file.parent / "backtest_trades.ndjson"
    monthly_csv = out_file.parent / "backtest_monthly.csv"
    monthly_jsonl = out_file.parent / "backtest_monthly.jsonl"
    if not monthly_csv.exists():
        with monthly_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "month",
                    "trades",
                    "win_rate",
                    "avgR",
                    "total_pnl_pct",
                    "total_pnl_dollars",
                    "start_equity",
                    "end_equity",
                    "stop",
                    "target",
                    "time_stop",
                    "eod_flat",
                    "time_exit",
                ],
            )
            writer.writeheader()
    daily_lookup = _build_daily_lookup(data_store, symbols, prefetch_start, end, cfg)
    month_key = None
    month_start_equity = equity
    month_trades: List[Dict] = []
    month_exit_reasons: Dict[str, int] = {}
    for idx, day in enumerate(days):
        date_str = day.isoformat()
        current_month = f"{day.year:04d}-{day.month:02d}"
        if month_key is None:
            month_key = current_month
            month_start_equity = equity
        symbols_for_day = symbols
        day_snapshot_symbols, _day_meta = read_asset_universe_snapshot(cfg, date_str)
        if day_snapshot_symbols:
            symbols_for_day = day_snapshot_symbols
            logging.info(
                "[BACKTEST] using universe snapshot date=%s size=%s",
                date_str,
                len(symbols_for_day),
            )
        logging.info("[BACKTEST] build watchlist date=%s symbols=%s", date_str, len(symbols_for_day))
        _ = build_watchlist(cfg, target_date=date_str, symbols=symbols_for_day, data_store=data_store, run_id=run_id)
        skip, info = market_filter_decision(date_str, cfg, data_store)
        if skip:
            skip_count += 1
            if isinstance(info, dict):
                reasons = info.get("reasons")
                if isinstance(reasons, list) and reasons:
                    for reason in reasons:
                        skip_reasons[str(reason)] = skip_reasons.get(str(reason), 0) + 1
                else:
                    reason = info.get("reason")
                    if reason:
                        reason_key = str(reason)
                        skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1
            logging.info("[BACKTEST] market filter skip date=%s info=%s", date_str, info)
            equity_curve.append({"date": date_str, "equity": equity})
            continue
        day_trades = run_replay(cfg, start_date=date_str, end_date=date_str, data_store=data_store, run_id=run_id)
        if day_trades:
            accepted, sized_records, equity = _apply_portfolio_sizing(day_trades, equity, cfg)
            if accepted:
                all_trades.extend(accepted)
            if sized_records:
                for rec in sized_records:
                    _attach_move_stats(rec, daily_lookup)
                sized_trades.extend(sized_records)
                month_trades.extend(sized_records)
                for rec in sized_records:
                    reason = str(rec.get("exit_reason") or "")
                    if reason:
                        month_exit_reasons[reason] = month_exit_reasons.get(reason, 0) + 1
                # Append per-trade details immediately so we don't wait for full run.
                with trades_ndjson.open("a", encoding="utf-8") as handle:
                    for rec in sized_records:
                        handle.write(json.dumps(rec) + "\n")
        equity_curve.append({"date": date_str, "equity": equity})
        logging.info("[BACKTEST] date=%s trades=%s total=%s equity=%.2f", date_str, len(day_trades), len(all_trades), equity)
        next_month = None
        if idx + 1 < len(days):
            next_day = days[idx + 1]
            next_month = f"{next_day.year:04d}-{next_day.month:02d}"
        if next_month != current_month:
            # Flush month summary.
            wins = [t for t in month_trades if float(t.get("r_multiple") or 0.0) > 0]
            trades_count = len(month_trades)
            win_rate = len(wins) / float(trades_count) if trades_count else 0.0
            avg_r = sum(float(t.get("r_multiple") or 0.0) for t in month_trades) / float(trades_count) if trades_count else 0.0
            total_pnl_pct = sum(float(t.get("pnl_pct") or 0.0) for t in month_trades)
            total_pnl_dollars = sum(float(t.get("pnl_total") or 0.0) for t in month_trades)
            month_row = {
                "month": current_month,
                "trades": trades_count,
                "win_rate": win_rate,
                "avgR": avg_r,
                "total_pnl_pct": total_pnl_pct,
                "total_pnl_dollars": total_pnl_dollars,
                "start_equity": month_start_equity,
                "end_equity": equity,
                "stop": month_exit_reasons.get("stop", 0),
                "target": month_exit_reasons.get("target", 0),
                "time_stop": month_exit_reasons.get("time_stop", 0),
                "eod_flat": month_exit_reasons.get("eod_flat", 0),
                "time_exit": month_exit_reasons.get("time_exit", 0),
            }
            with monthly_csv.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "month",
                        "trades",
                        "win_rate",
                        "avgR",
                        "total_pnl_pct",
                        "total_pnl_dollars",
                        "start_equity",
                        "end_equity",
                        "stop",
                        "target",
                        "time_stop",
                        "eod_flat",
                        "time_exit",
                    ],
                )
                writer.writerow(month_row)
            with monthly_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(month_row) + "\n")
            month_key = next_month
            month_start_equity = equity
            month_trades = []
            month_exit_reasons = {}

    leverage = float(params.get("leverage") or 4.0)
    # Aggregate notional per day for margin usage metrics.
    notional_by_day: Dict[str, float] = {}
    for rec in sized_trades:
        day = rec.get("entry_date")
        if not day:
            continue
        notional_by_day[day] = notional_by_day.get(day, 0.0) + float(rec.get("notional") or 0.0)

    daily_metrics: List[Dict] = []
    peak = 0.0
    max_drawdown_pct = 0.0
    max_drawdown_date = None
    max_daily_drop_pct = 0.0
    max_daily_drop_date = None
    max_day_margin_usage_pct = 0.0
    max_day_margin_usage_date = None
    prev_equity = None
    for row in equity_curve:
        date_str = row["date"]
        equity_val = float(row["equity"])
        if equity_val > peak:
            peak = equity_val
        drawdown_pct = ((equity_val - peak) / peak) * 100.0 if peak > 0 else 0.0
        if drawdown_pct < max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
            max_drawdown_date = date_str
        daily_return_pct = 0.0
        if prev_equity is not None and prev_equity > 0:
            daily_return_pct = ((equity_val - prev_equity) / prev_equity) * 100.0
            if daily_return_pct < max_daily_drop_pct:
                max_daily_drop_pct = daily_return_pct
                max_daily_drop_date = date_str
        prev_equity = equity_val
        day_notional = notional_by_day.get(date_str, 0.0)
        margin_usage_pct = ((day_notional / (equity_val * leverage)) * 100.0) if equity_val > 0 else 0.0
        if margin_usage_pct > max_day_margin_usage_pct:
            max_day_margin_usage_pct = margin_usage_pct
            max_day_margin_usage_date = date_str
        daily_metrics.append(
            {
                "date": date_str,
                "equity": round(equity_val, 6),
                "daily_return_pct": round(daily_return_pct, 6),
                "drawdown_pct": round(drawdown_pct, 6),
                "margin_usage_pct": round(margin_usage_pct, 6),
            }
        )

    summary = _summarize(all_trades)
    if out_file:
        exit_counts: Dict[str, int] = {}
        entry_stats: Dict[str, Dict[str, float]] = {}
        entry_trades: Dict[str, int] = {}
        entry_wins: Dict[str, int] = {}
        entry_r_sum: Dict[str, float] = {}
        direction_stats: Dict[str, Dict[str, float]] = {}
        direction_trades: Dict[str, int] = {}
        direction_wins: Dict[str, int] = {}
        direction_r_sum: Dict[str, float] = {}
        mfe_vals: List[float] = []
        mae_vals: List[float] = []
        gap_vals: List[float] = []
        pullback_vals: List[float] = []
        confirm_hit_vals: List[float] = []
        signal_return_vals: List[float] = []
        signal_return_atr_vals: List[float] = []
        atr_vals: List[float] = []
        symbol_stats: Dict[str, Dict[str, float]] = {}
        for rec in sized_trades:
            sym = str(rec.get("symbol") or "").upper()
            if not sym:
                continue
            stat = symbol_stats.setdefault(
                sym,
                {
                    "trades": 0,
                    "wins": 0,
                    "avgR": 0.0,
                    "total_pnl_pct": 0.0,
                    "avg_mfe_pct": 0.0,
                    "avg_mae_pct": 0.0,
                },
            )
            stat["trades"] += 1
            if float(rec.get("r_multiple") or 0.0) > 0:
                stat["wins"] += 1
            stat["avgR"] += float(rec.get("r_multiple") or 0.0)
            stat["total_pnl_pct"] += float(rec.get("pnl_pct") or 0.0)
            mfe_val = rec.get("mfe_pct")
            if mfe_val is None:
                mfe_val = rec.get("day_mfe_pct")
            mae_val = rec.get("mae_pct")
            if mae_val is None:
                mae_val = rec.get("day_mae_pct")
            if mfe_val is not None:
                stat["avg_mfe_pct"] += float(mfe_val or 0.0)
            if mae_val is not None:
                stat["avg_mae_pct"] += float(mae_val or 0.0)
            reason = str(rec.get("exit_reason") or "")
            if reason:
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
            et = str(rec.get("entry_time_et") or "")
            if et:
                entry_trades[et] = entry_trades.get(et, 0) + 1
                entry_r_sum[et] = entry_r_sum.get(et, 0.0) + float(rec.get("r_multiple") or 0.0)
                if float(rec.get("r_multiple") or 0.0) > 0:
                    entry_wins[et] = entry_wins.get(et, 0) + 1
            direction = str(rec.get("direction") or "")
            if direction:
                direction_trades[direction] = direction_trades.get(direction, 0) + 1
                direction_r_sum[direction] = direction_r_sum.get(direction, 0.0) + float(rec.get("r_multiple") or 0.0)
                if float(rec.get("r_multiple") or 0.0) > 0:
                    direction_wins[direction] = direction_wins.get(direction, 0) + 1
            if rec.get("mfe_pct") is not None:
                mfe_vals.append(float(rec.get("mfe_pct") or 0.0))
            elif rec.get("day_mfe_pct") is not None:
                mfe_vals.append(float(rec.get("day_mfe_pct") or 0.0))
            if rec.get("mae_pct") is not None:
                mae_vals.append(float(rec.get("mae_pct") or 0.0))
            elif rec.get("day_mae_pct") is not None:
                mae_vals.append(float(rec.get("day_mae_pct") or 0.0))
            if rec.get("gap_bps") is not None:
                gap_vals.append(float(rec.get("gap_bps") or 0.0))
            if rec.get("early_pullback_bps") is not None:
                pullback_vals.append(float(rec.get("early_pullback_bps") or 0.0))
            if rec.get("confirm_hit_bps") is not None:
                confirm_hit_vals.append(float(rec.get("confirm_hit_bps") or 0.0))
            if rec.get("signal_return_pct") is not None:
                signal_return_vals.append(float(rec.get("signal_return_pct") or 0.0))
            if rec.get("signal_return_atr") is not None:
                signal_return_atr_vals.append(float(rec.get("signal_return_atr") or 0.0))
            if rec.get("atr") is not None:
                atr_vals.append(float(rec.get("atr") or 0.0))
        for sym, stat in symbol_stats.items():
            trades_count = int(stat["trades"])
            if trades_count <= 0:
                continue
            stat["win_rate"] = stat["wins"] / float(trades_count)
            stat["avgR"] = stat["avgR"] / float(trades_count)
            stat["avg_mfe_pct"] = stat["avg_mfe_pct"] / float(trades_count)
            stat["avg_mae_pct"] = stat["avg_mae_pct"] / float(trades_count)
        for et, cnt in entry_trades.items():
            wins = entry_wins.get(et, 0)
            entry_stats[et] = {
                "trades": cnt,
                "win_rate": (wins / float(cnt)) if cnt else 0.0,
                "avgR": (entry_r_sum.get(et, 0.0) / float(cnt)) if cnt else 0.0,
            }
        for direction, cnt in direction_trades.items():
            wins = direction_wins.get(direction, 0)
            direction_stats[direction] = {
                "trades": cnt,
                "win_rate": (wins / float(cnt)) if cnt else 0.0,
                "avgR": (direction_r_sum.get(direction, 0.0) / float(cnt)) if cnt else 0.0,
            }
        trades_payload = []
        for t in all_trades:
            plan = getattr(t, "plan", None)
            plan_dict = plan.__dict__ if plan is not None else None
            trades_payload.append(
                {
                    "plan": plan_dict,
                    "exit_date": t.exit_date,
                    "exit_price": t.exit_price,
                    "raw_entry_price": plan.entry_price if plan is not None else None,
                    "raw_exit_price": getattr(t, "raw_exit_price", None),
                    "entry_fill_price": getattr(t, "entry_fill_price", None),
                    "exit_fill_price": getattr(t, "exit_fill_price", None),
                    "exit_reason": t.exit_reason,
                    "pnl_pct": t.pnl_pct,
                    "raw_pnl_pct": getattr(t, "raw_pnl_pct", None),
                    "r_multiple": t.r_multiple,
                    "raw_r_multiple": getattr(t, "raw_r_multiple", None),
                    "fill_model_enabled": getattr(t, "fill_model_enabled", None),
                    "quote_fill_enabled": getattr(t, "quote_fill_enabled", None),
                    "entry_fill_mode": getattr(t, "entry_fill_mode", None),
                    "exit_fill_mode": getattr(t, "exit_fill_mode", None),
                    "entry_quote_ok": getattr(t, "entry_quote_ok", None),
                    "exit_quote_ok": getattr(t, "exit_quote_ok", None),
                    "entry_quote_spread_cap_bps": getattr(t, "entry_quote_spread_cap_bps", None),
                    "entry_quote_deviation_cap_bps": getattr(t, "entry_quote_deviation_cap_bps", None),
                    "exit_quote_spread_cap_bps": getattr(t, "exit_quote_spread_cap_bps", None),
                    "exit_quote_deviation_cap_bps": getattr(t, "exit_quote_deviation_cap_bps", None),
                    "entry_quote_reason": getattr(t, "entry_quote_reason", None),
                    "exit_quote_reason": getattr(t, "exit_quote_reason", None),
                    "entry_quote_ts": getattr(t, "entry_quote_ts", None),
                    "exit_quote_ts": getattr(t, "exit_quote_ts", None),
                    "entry_quote_selection": getattr(t, "entry_quote_selection", None),
                    "exit_quote_selection": getattr(t, "exit_quote_selection", None),
                    "entry_quote_age_seconds": getattr(t, "entry_quote_age_seconds", None),
                    "exit_quote_age_seconds": getattr(t, "exit_quote_age_seconds", None),
                    "entry_quote_spread_bps": getattr(t, "entry_quote_spread_bps", None),
                    "exit_quote_spread_bps": getattr(t, "exit_quote_spread_bps", None),
                    "entry_quote_deviation_bps": getattr(t, "entry_quote_deviation_bps", None),
                    "exit_quote_deviation_bps": getattr(t, "exit_quote_deviation_bps", None),
                    "entry_quote_bid": getattr(t, "entry_quote_bid", None),
                    "entry_quote_ask": getattr(t, "entry_quote_ask", None),
                    "exit_quote_bid": getattr(t, "exit_quote_bid", None),
                    "exit_quote_ask": getattr(t, "exit_quote_ask", None),
                    "entry_quote_price_source": getattr(t, "entry_quote_price_source", None),
                    "exit_quote_price_source": getattr(t, "exit_quote_price_source", None),
                    "target_limit_bound_applied": getattr(t, "target_limit_bound_applied", None),
                    "fill_exit_reason_bucket": getattr(t, "fill_exit_reason_bucket", None),
                    "fill_entry_bps": getattr(t, "fill_entry_bps", None),
                    "fill_exit_bps": getattr(t, "fill_exit_bps", None),
                    "fill_entry_bps_model": getattr(t, "fill_entry_bps_model", None),
                    "fill_exit_bps_model": getattr(t, "fill_exit_bps_model", None),
                    "fill_half_spread_bps": getattr(t, "fill_half_spread_bps", None),
                    "fill_entry_slippage_bps": getattr(t, "fill_entry_slippage_bps", None),
                    "fill_exit_slippage_bps": getattr(t, "fill_exit_slippage_bps", None),
                    "fill_cost_per_share": getattr(t, "fill_cost_per_share", None),
                    "fill_cost_total": getattr(t, "fill_cost_total", None),
                    "stop_distance_fill": getattr(t, "stop_distance_fill", None),
                    "mfe_pct": getattr(t, "mfe_pct", None),
                    "mae_pct": getattr(t, "mae_pct", None),
                    "mfe_r": getattr(t, "mfe_r", None),
                    "mae_r": getattr(t, "mae_r", None),
                    "gap_bps": getattr(plan, "gap_bps", None) if plan else None,
                    "early_pullback_bps": getattr(plan, "early_pullback_bps", None) if plan else None,
                    "early_reversal_bps": getattr(plan, "early_reversal_bps", None) if plan else None,
                    "target_mode": getattr(plan, "target_mode", None) if plan else None,
                    "target_window_avg_pct": getattr(plan, "target_window_avg_pct", None) if plan else None,
                    "target_window_mult": getattr(plan, "target_window_mult", None) if plan else None,
                    "target_window_minutes": getattr(plan, "target_window_minutes", None) if plan else None,
                    "target_window_samples": getattr(plan, "target_window_samples", None) if plan else None,
                    "confirm_move_bps": getattr(plan, "confirm_move_bps", None) if plan else None,
                    "confirm_minutes": getattr(plan, "confirm_minutes", None) if plan else None,
                    "confirm_hit_bps": getattr(plan, "confirm_hit_bps", None) if plan else None,
                    "signal_return_pct": getattr(plan, "signal_return_pct", None) if plan else None,
                    "signal_return_atr": getattr(plan, "signal_return_atr", None) if plan else None,
                    "atr": getattr(plan, "atr", None) if plan else None,
                }
            )
        equity_return_pct = ((equity / starting_equity) - 1.0) * 100.0 if starting_equity > 0 else 0.0
        payload = {
            "summary": {
                **summary,
                "starting_equity": starting_equity,
                "ending_equity": equity,
                "total_pnl_dollars": equity - starting_equity,
                "total_pnl_pct_equity": equity_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "max_drawdown_date": max_drawdown_date,
                "max_daily_drop_pct": max_daily_drop_pct,
                "max_daily_drop_date": max_daily_drop_date,
                "max_day_margin_usage_pct": max_day_margin_usage_pct,
                "max_day_margin_usage_date": max_day_margin_usage_date,
                "market_filter_skips": skip_count,
                "market_filter_skip_reasons": skip_reasons,
            },
            "trades": trades_payload,
            "sized_trades": sized_trades,
            "equity_curve": equity_curve,
            "daily_metrics": daily_metrics,
            "symbol_stats": symbol_stats,
        }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Write a daily metrics CSV alongside the backtest JSON.
        daily_csv = out_file.parent / "backtest_daily.csv"
        with daily_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["date", "equity", "daily_return_pct", "drawdown_pct", "margin_usage_pct"],
            )
            writer.writeheader()
            for row in daily_metrics:
                writer.writerow(row)
        metrics_json = out_file.parent / "backtest_metrics.json"
        metrics_json.write_text(
            json.dumps(
                {
                    "max_drawdown_pct": max_drawdown_pct,
                    "max_drawdown_date": max_drawdown_date,
                    "max_daily_drop_pct": max_daily_drop_pct,
                    "max_daily_drop_date": max_daily_drop_date,
                    "max_day_margin_usage_pct": max_day_margin_usage_pct,
                    "max_day_margin_usage_date": max_day_margin_usage_date,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tuning_report = out_file.parent / "tuning_report.json"
        tuning_report.write_text(
            json.dumps(
                {
                    "summary": payload.get("summary") or {},
                    "exit_counts": exit_counts,
                    "entry_time_stats": entry_stats,
                    "direction_stats": direction_stats,
                    "mfe_pct_percentiles": _percentiles(mfe_vals),
                    "mae_pct_percentiles": _percentiles(mae_vals),
                    "gap_bps_percentiles": _percentiles(gap_vals),
                    "early_pullback_bps_percentiles": _percentiles(pullback_vals),
                    "confirm_hit_bps_percentiles": _percentiles(confirm_hit_vals),
                    "signal_return_pct_percentiles": _percentiles(signal_return_vals),
                    "signal_return_atr_percentiles": _percentiles(signal_return_atr_vals),
                    "atr_percentiles": _percentiles(atr_vals),
                    "config_snapshot": {
                        "daily_trend_reversal": cfg.get("daily_trend_reversal") or {},
                        "watchlist": cfg.get("watchlist") or {},
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logging.info("[BACKTEST] wrote %s", out_file)
    return summary, all_trades
