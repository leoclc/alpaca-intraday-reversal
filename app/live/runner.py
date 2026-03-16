from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Dict, List, Optional

from app.brokers.alpaca import AlpacaBroker
from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.data.alpaca_intraday_store import (
    filter_intraday_bars_until,
    get_intraday_bars,
    get_latest_intraday_prices,
)
from app.market.filters import market_filter_decision
from app.portfolio.sizing import (
    compute_qty_with_guards,
    estimate_slot_target_from_stats,
    resolve_effective_margin_buffer,
)
from app.strategies.daily_trend_reversal import build_trade, generate_signal_for_date
from app.utils.time import ensure_et, et_now, parse_time_hhmm
from app.watchlist.day_filter import day_filter_decision
from app.watchlist.prioritization import sort_symbols_by_watchlist_priority
from app.watchlist.storage import expected_watchlist_date_str, read_watchlist


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


def _merged_symbol_params(cfg: Dict, overrides: Optional[Dict]) -> Dict:
    params = dict(cfg.get("daily_trend_reversal") or {})
    if isinstance(overrides, dict) and overrides:
        params.update(overrides)
    return params


def _estimate_slot_target_for_live_pass(
    cfg: Dict,
    wl_rows: List[Dict],
    symbols: List[str],
    entry_time_et: Optional[str],
    scan_first_valid_mode: bool,
) -> tuple[Optional[int], Dict]:
    params = cfg.get("daily_trend_reversal") or {}
    if not bool(params.get("slot_distribution_enabled", False)):
        return None, {"enabled": False}
    watch_cfg = cfg.get("watchlist") or {}
    lookback_days = float(watch_cfg.get("lookback_days") or 252)
    slot_min_slots = int(params.get("slot_distribution_min_slots") or 1)
    slot_max_slots = int(params.get("slot_distribution_max_slots") or 0)
    symbols_set = {str(s or "").upper() for s in (symbols or []) if s}
    current_et = str(entry_time_et or "").strip()
    rows_eligible: List[Dict] = []
    for row in wl_rows or []:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").upper()
        if not sym or sym not in symbols_set:
            continue
        if scan_first_valid_mode and current_et:
            row_et = str(row.get("entry_time_et") or "").strip()
            if row_et:
                try:
                    if parse_time_hhmm(row_et) > parse_time_hhmm(current_et):
                        continue
                except Exception:
                    pass
        rows_eligible.append(row)
    slot_target, meta = estimate_slot_target_from_stats(
        rows_eligible,
        lookback_days,
        min_slots=slot_min_slots,
        max_slots=slot_max_slots,
        cap_by_candidates=True,
    )
    meta = dict(meta or {})
    meta["enabled"] = True
    meta["eligible_symbols"] = len(rows_eligible)
    meta["entry_time_et"] = current_et or None
    return (slot_target if slot_target > 0 else None), meta


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
        try:
            entry_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(entry_time_et))
            return (entry_dt + dt.timedelta(minutes=confirm_minutes)).strftime("%H:%M")
        except Exception:
            return entry_time_et
    return entry_time_et


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_float_opt(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _asset_maintenance_margin_pct(
    broker: Optional[AlpacaBroker],
    symbol: Optional[str],
    asset_margin_cache: Optional[Dict[str, Optional[float]]],
) -> Optional[float]:
    sym = str(symbol or "").upper().strip()
    if not sym or broker is None or not broker.ready():
        return None
    cache = asset_margin_cache if isinstance(asset_margin_cache, dict) else None
    if cache is not None and sym in cache and cache[sym] is not None:
        return cache[sym]
    mmr_pct: Optional[float] = None
    try:
        asset = broker.get_asset(sym) or {}
        mmr_pct = _safe_float_opt(asset.get("maintenance_margin_requirement"))
        if mmr_pct is not None and mmr_pct < 0:
            mmr_pct = None
    except Exception:
        mmr_pct = None
    # Cache only successful values so transient API failures don't poison the cache
    # with `None` and silently disable MMR-based guards for the rest of the session.
    if cache is not None and mmr_pct is not None:
        cache[sym] = mmr_pct
    elif cache is not None and sym in cache and cache[sym] is None:
        cache.pop(sym, None)
    return mmr_pct


def _bp_short_mmr_multiplier(
    cfg: Dict,
    broker: Optional[AlpacaBroker],
    symbol: Optional[str],
    asset_margin_cache: Optional[Dict[str, Optional[float]]],
) -> float:
    params = cfg.get("daily_trend_reversal") or {}
    if not bool(params.get("buying_power_short_mmr_enabled", False)):
        return 1.0
    mmr_pct = _asset_maintenance_margin_pct(broker, symbol, asset_margin_cache)
    if mmr_pct is None:
        return 1.0
    weight = max(0.0, min(float(params.get("buying_power_short_mmr_weight") or 0.0), 2.0))
    floor_pct = max(0.0, float(params.get("buying_power_short_mmr_floor_pct") or 0.0))
    cap_pct = max(floor_pct, float(params.get("buying_power_short_mmr_cap_pct") or 50.0))
    eff_pct = _clamp(mmr_pct * weight, floor_pct, cap_pct)
    return 1.0 + (eff_pct / 100.0)


def _bp_effective_price(
    cfg: Dict,
    side: str,
    ref_price: float,
    *,
    broker: Optional[AlpacaBroker] = None,
    symbol: Optional[str] = None,
    asset_margin_cache: Optional[Dict[str, Optional[float]]] = None,
) -> float:
    px = max(0.0, float(ref_price or 0.0))
    if px <= 0:
        return 0.0
    params = cfg.get("daily_trend_reversal") or {}
    if not bool(params.get("buying_power_model_enabled", True)):
        return px
    long_markup = max(0.0, min(float(params.get("buying_power_long_open_markup") or 0.0), 0.5))
    short_markup = max(0.0, min(float(params.get("buying_power_short_open_markup") or 0.03), 0.5))
    short_margin_mult = max(1.0, min(float(params.get("buying_power_short_margin_multiplier") or 1.2), 3.0))
    side_norm = str(side or "").strip().lower()
    if side_norm == "sell":
        mmr_mult = _bp_short_mmr_multiplier(cfg, broker, symbol, asset_margin_cache)
        return px * (1.0 + short_markup) * short_margin_mult * mmr_mult
    return px * (1.0 + long_markup)


def _compute_reanchored_exit_prices(plan: Any, fill_price: float) -> tuple[float, float, float, float]:
    fill_px = max(0.0, float(fill_price or 0.0))
    if fill_px <= 0:
        raise ValueError("fill price must be positive")

    raw_entry = _safe_float(getattr(plan, "entry_price", None), fill_px)
    raw_stop = _safe_float(getattr(plan, "stop_price", None), 0.0)
    raw_target = _safe_float(getattr(plan, "target_price", None), 0.0)
    stop_offset = abs(raw_entry - raw_stop)
    target_offset = abs(raw_target - raw_entry)
    if stop_offset <= 0:
        stop_offset = max(0.0, _safe_float(getattr(plan, "stop_distance", None), 0.0))
    if target_offset <= 0 and stop_offset > 0:
        rr = max(0.0, _safe_float(getattr(plan, "target_rr", None), 0.0))
        target_offset = stop_offset * rr if rr > 0 else stop_offset
    if stop_offset <= 0 or target_offset <= 0:
        raise ValueError("invalid stop/target offsets for reanchoring")

    direction = str(getattr(plan, "direction", "long") or "long").lower()
    if direction == "long":
        stop_price = fill_px - stop_offset
        target_price = fill_px + target_offset
    else:
        stop_price = fill_px + stop_offset
        target_price = fill_px - target_offset
    return stop_price, target_price, stop_offset, target_offset


def _wait_for_entry_fill(
    broker: AlpacaBroker,
    order_id: str,
    expected_qty: int,
    timeout_sec: float,
    poll_sec: float,
    cancel_unfilled: bool,
) -> Optional[Dict[str, Any]]:
    oid = str(order_id or "").strip()
    if not oid:
        return None
    expected = max(1, int(expected_qty))
    timeout = max(0.5, float(timeout_sec))
    poll = max(0.05, float(poll_sec))
    deadline = time.time() + timeout
    last_order: Optional[Dict[str, Any]] = None

    while True:
        try:
            order = broker.get_order(oid) or {}
        except Exception:
            order = {}
        if order:
            last_order = order
            status = str(order.get("status") or "").lower().strip()
            filled_qty = int(_safe_float(order.get("filled_qty"), 0.0) or 0)
            fill_price = _safe_float(order.get("filled_avg_price"), 0.0)
            if filled_qty > 0 and fill_price > 0:
                full = status == "filled" or filled_qty >= expected
                if full:
                    return {
                        "order": order,
                        "status": status,
                        "filled_qty": filled_qty,
                        "fill_price": fill_price,
                        "full": True,
                        "timed_out": False,
                    }
            if status in {"canceled", "expired", "rejected"}:
                break

        if time.time() >= deadline:
            break
        time.sleep(poll)

    if cancel_unfilled:
        try:
            broker.cancel_order(oid)
        except Exception:
            pass
        try:
            refreshed = broker.get_order(oid) or {}
            if refreshed:
                last_order = refreshed
        except Exception:
            pass

    if not isinstance(last_order, dict):
        return None
    status = str(last_order.get("status") or "").lower().strip()
    filled_qty = int(_safe_float(last_order.get("filled_qty"), 0.0) or 0)
    fill_price = _safe_float(last_order.get("filled_avg_price"), 0.0)
    if filled_qty <= 0 or fill_price <= 0:
        return None
    return {
        "order": last_order,
        "status": status,
        "filled_qty": filled_qty,
        "fill_price": fill_price,
        "full": status == "filled" or filled_qty >= expected,
        "timed_out": True,
    }


def _position_fill_snapshot(broker: AlpacaBroker, symbol: str) -> Optional[Dict[str, Any]]:
    sym = str(symbol or "").upper().strip()
    if not sym:
        return None
    try:
        positions = broker.list_positions() or []
    except Exception:
        return None
    for pos in positions:
        if str((pos or {}).get("symbol") or "").upper().strip() != sym:
            continue
        qty_abs = abs(int(_safe_float((pos or {}).get("qty"), 0.0) or 0))
        avg_entry = _safe_float((pos or {}).get("avg_entry_price"), 0.0)
        if qty_abs > 0 and avg_entry > 0:
            return {
                "filled_qty": qty_abs,
                "fill_price": avg_entry,
                "source": "position_snapshot",
            }
    return None


def _position_snapshot_map(broker: AlpacaBroker) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        positions = broker.list_positions() or []
    except Exception:
        return out
    for pos in positions:
        sym = str((pos or {}).get("symbol") or "").upper().strip()
        if not sym:
            continue
        out[sym] = pos or {}
    return out


def _open_exit_coverage_map(broker: AlpacaBroker) -> Dict[tuple[str, str], Dict[str, bool]]:
    coverage: Dict[tuple[str, str], Dict[str, bool]] = {}
    try:
        orders = broker.list_orders(status="open", limit=500) or []
    except Exception:
        return coverage
    stop_types = {"stop", "stop_limit", "trailing_stop"}
    for order in orders:
        sym = str((order or {}).get("symbol") or "").upper().strip()
        side = str((order or {}).get("side") or "").lower().strip()
        otype = str((order or {}).get("type") or "").lower().strip()
        if not sym or side not in {"buy", "sell"}:
            continue
        key = (sym, side)
        state = coverage.setdefault(key, {"has_limit": False, "has_stop": False})
        if otype == "limit":
            state["has_limit"] = True
        if otype in stop_types:
            state["has_stop"] = True
    return coverage


def _repair_pending_reanchor_exits(
    cfg: Dict[str, Any],
    broker: AlpacaBroker,
    pending: Dict[str, Dict[str, Any]],
    *,
    debug: bool = False,
) -> None:
    if not pending or not broker.ready():
        return
    positions_by_symbol = _position_snapshot_map(broker)
    if not positions_by_symbol:
        return
    coverage = _open_exit_coverage_map(broker)
    for symbol, meta in list((pending or {}).items()):
        sym = str(symbol or "").upper().strip()
        if not sym:
            continue
        pos = positions_by_symbol.get(sym)
        if not pos:
            continue
        side_raw = str((pos or {}).get("side") or "").lower().strip()
        qty_raw = _safe_float((pos or {}).get("qty"), 0.0)
        is_long = (side_raw == "long") or (qty_raw > 0)
        close_side = "sell" if is_long else "buy"
        cov = coverage.get((sym, close_side), {"has_limit": False, "has_stop": False})
        if cov.get("has_limit") and cov.get("has_stop"):
            if debug:
                logging.info("[LIVE_DEBUG] reanchor_repair_skip symbol=%s reason=covered", sym)
            continue

        qty_abs = int(abs(qty_raw))
        avg_entry = _safe_float((pos or {}).get("avg_entry_price"), 0.0)
        plan = (meta or {}).get("plan")
        tif = str((meta or {}).get("tif") or "day").lower()
        if qty_abs <= 0 or avg_entry <= 0 or plan is None:
            if debug:
                logging.info(
                    "[LIVE_DEBUG] reanchor_repair_skip symbol=%s reason=invalid_snapshot qty=%s avg_entry=%s has_plan=%s",
                    sym,
                    qty_abs,
                    avg_entry,
                    bool(plan),
                )
            continue
        try:
            stop_price, target_price, stop_offset, target_offset = _compute_reanchored_exit_prices(plan, avg_entry)
            resp = broker.submit_oco_exit_order(
                symbol=sym,
                side=close_side,
                qty=qty_abs,
                take_profit=target_price,
                stop_loss=stop_price,
                base_price=avg_entry,
                tif=tif,
            )
            logging.info(
                "[LIVE] reanchor_repair_exit_placed symbol=%s qty=%s close_side=%s entry=%.6f stop=%.6f target=%.6f stop_offset=%.6f target_offset=%.6f order_id=%s",
                sym,
                qty_abs,
                close_side,
                avg_entry,
                stop_price,
                target_price,
                stop_offset,
                target_offset,
                (resp or {}).get("id"),
            )
            pending.pop(sym, None)
        except Exception as exc:
            logging.error("[LIVE] reanchor_repair_exit_failed symbol=%s error=%s", sym, exc)


def _passes_live_quality_gate(cfg: Dict, stats: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    params = cfg.get("daily_trend_reversal") or {}
    if not bool(params.get("live_quality_gate_enabled", False)):
        return True, None
    if not isinstance(stats, dict):
        return False, "missing_watchlist_stats"

    min_selection_score = _safe_float_opt(params.get("live_quality_min_selection_score"))
    min_avg_r = _safe_float_opt(params.get("live_quality_min_avgR"))
    min_trades = int(params.get("live_quality_min_trades_count") or 0)
    max_rank = int(params.get("live_quality_max_rank") or 0)
    min_positive_month_rate = _safe_float_opt(params.get("live_quality_min_positive_month_rate"))

    rank = int(stats.get("rank") or 0)
    if max_rank > 0 and rank > 0 and rank > max_rank:
        return False, f"rank>{max_rank}"

    selection_score = _safe_float_opt(stats.get("selection_score"))
    if min_selection_score is not None and selection_score is not None and selection_score < min_selection_score:
        return False, f"selection_score<{min_selection_score:.4f}"

    avg_r = _safe_float_opt(stats.get("avgR"))
    if min_avg_r is not None and avg_r is not None and avg_r < min_avg_r:
        return False, f"avgR<{min_avg_r:.4f}"

    trades_count = int(stats.get("trades_count") or 0)
    if min_trades > 0 and trades_count < min_trades:
        return False, f"trades_count<{min_trades}"

    positive_month_rate = _safe_float_opt(stats.get("positive_month_rate"))
    if (
        min_positive_month_rate is not None
        and positive_month_rate is not None
        and positive_month_rate < min_positive_month_rate
    ):
        return False, f"positive_month_rate<{min_positive_month_rate:.4f}"

    return True, None


def _account_buying_power(acct: Optional[Dict[str, Any]]) -> tuple[float, bool]:
    if not isinstance(acct, dict):
        return 0.0, False
    primary_candidates: List[float] = []
    for key in ("effective_buying_power", "daytrading_buying_power", "buying_power"):
        if key not in acct or acct.get(key) is None:
            continue
        try:
            val = max(0.0, float(acct.get(key) or 0.0))
            if val > 0:
                primary_candidates.append(val)
        except Exception:
            continue
    if primary_candidates:
        # Prefer intraday/effective BP fields that Alpaca applies to active session checks.
        return min(primary_candidates), True
    fallback_candidates: List[float] = []
    for key in ("regt_buying_power",):
        if key not in acct or acct.get(key) is None:
            continue
        try:
            val = max(0.0, float(acct.get(key) or 0.0))
            if val > 0:
                fallback_candidates.append(val)
        except Exception:
            continue
    if fallback_candidates:
        return min(fallback_candidates), True
    return 0.0, False


def _account_non_marginable_buying_power(acct: Optional[Dict[str, Any]]) -> tuple[float, bool]:
    if not isinstance(acct, dict):
        return 0.0, False
    try:
        val = max(0.0, float(acct.get("non_marginable_buying_power") or 0.0))
        if val > 0:
            return val, True
    except Exception:
        pass
    return 0.0, False


def _seed_runtime_exposure(
    cfg: Dict,
    broker: AlpacaBroker,
    *,
    asset_margin_cache: Optional[Dict[str, Optional[float]]] = None,
    emit_debug: bool = True,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[float], bool, Optional[float], bool]:
    params = cfg.get("daily_trend_reversal") or {}
    if not broker.ready():
        return None, None, None, None, None, False, None, False
    try:
        acct = broker.get_account() or {}
        equity_seed = _safe_float(acct.get("equity"), 0.0)
        leverage_cfg = params.get("leverage")
        try:
            leverage = float(leverage_cfg) if leverage_cfg is not None else _safe_float(acct.get("multiplier"), 4.0)
        except Exception:
            leverage = 4.0
        buying_power, bp_present = _account_buying_power(acct)
        non_marginable_buying_power, non_marginable_bp_present = _account_non_marginable_buying_power(acct)
        max_margin_usage = float(params.get("max_margin_usage") or 0.70)
        allowed_total_seed = max(0.0, equity_seed * leverage)
        if bp_present:
            allowed_total_seed = min(allowed_total_seed, buying_power)
        allowed_total_seed *= max(0.0, min(max_margin_usage, 1.0))

        used_positions = 0.0
        positions = broker.list_positions() or []
        for pos in positions:
            qty_abs = abs(_safe_float(pos.get("qty"), 0.0))
            mv = abs(_safe_float(pos.get("market_value"), 0.0))
            if mv <= 0 and qty_abs > 0:
                px = abs(_safe_float(pos.get("current_price"), 0.0))
                if px <= 0:
                    px = abs(_safe_float(pos.get("avg_entry_price"), 0.0))
                mv = qty_abs * px
            side_raw = str(pos.get("side") or "").strip().lower()
            is_short = side_raw == "short" or _safe_float(pos.get("qty"), 0.0) < 0
            bp_side = "sell" if is_short else "buy"
            symbol = str(pos.get("symbol") or "").upper()
            if mv > 0 and qty_abs > 0:
                mv = _bp_effective_price(
                    cfg,
                    bp_side,
                    mv / qty_abs,
                    broker=broker,
                    symbol=symbol,
                    asset_margin_cache=asset_margin_cache,
                ) * qty_abs
            used_positions += mv
        open_order_notional, open_order_count = _estimate_open_entry_order_exposure(
            cfg,
            broker,
            asset_margin_cache=asset_margin_cache,
        )
        used_notional_runtime = used_positions + open_order_notional
        open_positions_runtime = len(positions) + open_order_count
        if emit_debug:
            logging.info(
                "[LIVE_DEBUG] exposure_seed equity=%.2f allowed_total=%.2f used_positions=%.2f used_open_orders=%.2f used_total=%.2f open_positions=%s open_entry_orders=%s buying_power=%.2f bp_present=%s non_marginable_bp=%.2f non_marginable_bp_present=%s",
                equity_seed,
                allowed_total_seed,
                used_positions,
                open_order_notional,
                used_notional_runtime,
                len(positions),
                open_order_count,
                buying_power,
                bp_present,
                non_marginable_buying_power,
                non_marginable_bp_present,
            )
        return (
            equity_seed,
            allowed_total_seed,
            used_notional_runtime,
            open_positions_runtime,
            buying_power,
            bp_present,
            non_marginable_buying_power,
            non_marginable_bp_present,
        )
    except Exception as exc:
        if emit_debug:
            logging.info("[LIVE_DEBUG] exposure_seed_failed error=%s", exc)
        return None, None, None, None, None, False, None, False


def _estimate_open_entry_order_exposure(
    cfg: Dict,
    broker: AlpacaBroker,
    *,
    asset_margin_cache: Optional[Dict[str, Optional[float]]] = None,
) -> tuple[float, int]:
    if not broker.ready():
        return 0.0, 0
    try:
        orders = broker.list_orders(status="open", limit=500) or []
    except Exception:
        return 0.0, 0
    unresolved: List[tuple[str, float, float, str]] = []
    symbols_needing_px: List[str] = []
    exposure = 0.0
    count = 0
    for order in orders:
        if not isinstance(order, dict):
            continue
        position_intent = str(order.get("position_intent") or "").strip().lower()
        if "open" not in position_intent or "close" in position_intent:
            continue
        qty_total = _safe_float(order.get("qty"), 0.0)
        filled_qty = _safe_float(order.get("filled_qty"), 0.0)
        remaining_qty = max(0.0, qty_total - filled_qty)
        if remaining_qty <= 0:
            continue
        notional = _safe_float(order.get("notional"), 0.0)
        if notional > 0:
            exposure += abs(notional)
            count += 1
            continue
        order_side = str(order.get("side") or "").strip().lower()
        symbol = str(order.get("symbol") or "").upper()
        limit_price = _safe_float(order.get("limit_price"), 0.0)
        if limit_price > 0:
            exposure += remaining_qty * _bp_effective_price(
                cfg,
                order_side,
                limit_price,
                broker=broker,
                symbol=symbol,
                asset_margin_cache=asset_margin_cache,
            )
            count += 1
            continue
        if not symbol:
            continue
        stop_price = _safe_float(order.get("stop_price"), 0.0)
        unresolved.append((symbol, remaining_qty, stop_price, order_side))
        symbols_needing_px.append(symbol)
    px_map: Dict[str, float] = {}
    if symbols_needing_px:
        try:
            px_map = get_latest_intraday_prices(sorted(set(symbols_needing_px)), cfg=cfg, lookback_minutes=5)
        except Exception:
            px_map = {}
    for symbol, remaining_qty, stop_price, order_side in unresolved:
        px = _safe_float(px_map.get(symbol), 0.0)
        if px <= 0 and stop_price > 0:
            px = stop_price
        if px > 0:
            exposure += remaining_qty * _bp_effective_price(
                cfg,
                order_side,
                px,
                broker=broker,
                symbol=symbol,
                asset_margin_cache=asset_margin_cache,
            )
            count += 1
    return exposure, count


def _compute_qty(
    cfg: Dict,
    plan,
    broker: AlpacaBroker,
    *,
    slot_target: Optional[int] = None,
    equity_override: Optional[float] = None,
    allowed_total_override: Optional[float] = None,
    used_notional_override: Optional[float] = None,
    open_positions_override: Optional[int] = None,
    day_start_equity: Optional[float] = None,
) -> tuple[int, Dict, float, float, int]:
    params = cfg.get("daily_trend_reversal") or {}
    fixed_qty = params.get("fixed_qty")
    if fixed_qty:
        return int(fixed_qty), {"fixed_qty": int(fixed_qty)}, 0.0, 0.0, 0
    risk_per_trade = float(params.get("risk_per_trade") or 0.0)
    if risk_per_trade <= 0:
        return 1, {"risk_per_trade": risk_per_trade}, 0.0, 0.0, 0
    acct = None
    if equity_override is None or allowed_total_override is None:
        acct = broker.get_account() if broker.ready() else None
        if not acct:
            return 1, {"account": "missing"}, 0.0, 0.0, 0
    if equity_override is not None:
        equity = _safe_float(equity_override, 0.0)
    else:
        try:
            equity = float((acct or {}).get("equity") or 0.0)
        except Exception:
            return 1, {"equity": "parse_error"}, 0.0, 0.0, 0

    leverage_cfg = params.get("leverage")
    max_margin_usage = float(params.get("max_margin_usage") or 0.70)
    if allowed_total_override is not None:
        allowed_total = max(0.0, _safe_float(allowed_total_override, 0.0))
    else:
        try:
            leverage = float(leverage_cfg) if leverage_cfg is not None else float((acct or {}).get("multiplier") or 4.0)
        except Exception:
            leverage = 4.0
        try:
            buying_power = float((acct or {}).get("buying_power") or 0.0)
        except Exception:
            buying_power = 0.0
        allowed_total = max(0.0, equity * leverage)
        if buying_power > 0:
            allowed_total = min(allowed_total, buying_power)
        allowed_total *= max(0.0, min(max_margin_usage, 1.0))

    if used_notional_override is not None:
        used_notional = max(0.0, _safe_float(used_notional_override, 0.0))
        open_positions = max(0, int(open_positions_override or 0))
    else:
        used_notional = 0.0
        positions = []
        try:
            positions = broker.list_positions() or []
            for pos in positions:
                try:
                    mv = float(pos.get("market_value") or 0.0)
                except Exception:
                    mv = 0.0
                used_notional += abs(mv)
        except Exception:
            used_notional = 0.0
            positions = []
        open_positions = len(positions)
    if open_positions_override is not None:
        open_positions = max(0, int(open_positions_override))

    qty, state = compute_qty_with_guards(
        plan,
        equity,
        used_notional,
        cfg,
        allowed_total_override=allowed_total,
        open_positions=open_positions,
        slot_target_override=slot_target,
        day_start_equity=day_start_equity,
    )
    return qty, state, allowed_total, used_notional, open_positions


def flatten_intraday_positions_if_needed(cfg: Dict, broker: AlpacaBroker) -> List[Dict]:
    params = cfg.get("daily_trend_reversal") or {}
    if not bool(params.get("intraday_only", False)):
        return []
    session_close_et = str(params.get("session_close_et") or "16:00")
    flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)
    now = et_now()
    close_time = parse_time_hhmm(session_close_et)
    close_dt = dt.datetime.combine(now.date(), close_time).replace(tzinfo=now.tzinfo)
    flatten_dt = close_dt - dt.timedelta(minutes=flatten_buffer)
    if now < flatten_dt:
        return []
    if not broker.ready():
        logging.error("[LIVE] Alpaca credentials missing; cannot flatten positions")
        return []
    try:
        positions = broker.list_positions()
    except Exception as exc:
        logging.error("[LIVE] failed to list positions error=%s", exc)
        return []
    closed: List[Dict] = []
    for pos in positions:
        symbol = str(pos.get("symbol") or "")
        if not symbol:
            continue
        try:
            resp = broker.close_position(symbol)
            closed.append(resp)
            logging.info("[LIVE] eod_flat symbol=%s", symbol)
        except Exception as exc:
            logging.error("[LIVE] failed to close symbol=%s error=%s", symbol, exc)
    return closed


def _parse_iso_ts(val: Optional[str]) -> Optional[dt.datetime]:
    if not val:
        return None
    ts = str(val).strip()
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(ts)
    except Exception:
        return None
    return ensure_et(parsed)


def enforce_time_stop(cfg: Dict, broker: AlpacaBroker) -> List[str]:
    params = cfg.get("daily_trend_reversal") or {}
    base_time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    tgt = expected_watchlist_date_str(None)
    wl = read_watchlist(tgt, cfg)
    wl_rows = wl.get("watchlist") or []
    symbol_overrides = {
        str(r.get("symbol") or "").upper(): (r.get("param_overrides") or {})
        for r in wl_rows
        if r.get("symbol")
    }
    has_time_stop_override = False
    for ov in symbol_overrides.values():
        try:
            if int((ov or {}).get("time_stop_minutes") or 0) > 0:
                has_time_stop_override = True
                break
        except Exception:
            continue
    if base_time_stop_minutes <= 0 and not has_time_stop_override:
        return []
    if not broker.ready():
        logging.error("[LIVE] Alpaca credentials missing; cannot enforce time stop")
        return []
    try:
        positions = broker.list_positions() or []
    except Exception as exc:
        logging.error("[LIVE] time stop failed to list positions error=%s", exc)
        return []
    if not positions:
        return []
    symbols = [str(p.get("symbol") or "").upper() for p in positions if p.get("symbol")]
    if not symbols:
        return []
    now = et_now()
    session_open_et = str(params.get("session_open_et") or "09:30")
    open_time = parse_time_hhmm(session_open_et)
    after_dt = ensure_et(dt.datetime.combine(now.date(), open_time) - dt.timedelta(minutes=1))
    try:
        orders = broker.list_orders(status="filled", symbols=symbols, after=after_dt.isoformat())
    except Exception as exc:
        logging.error("[LIVE] time stop failed to list orders error=%s", exc)
        return []
    latest_fill: Dict[str, dt.datetime] = {}
    for order in orders or []:
        sym = str(order.get("symbol") or "").upper()
        filled_at = _parse_iso_ts(order.get("filled_at") or order.get("updated_at") or order.get("submitted_at"))
        if not sym or not filled_at:
            continue
        prev = latest_fill.get(sym)
        if not prev or filled_at > prev:
            latest_fill[sym] = filled_at
    closed: List[str] = []
    for pos in positions:
        sym = str(pos.get("symbol") or "").upper()
        if not sym:
            continue
        entry_dt = latest_fill.get(sym)
        if not entry_dt:
            continue
        symbol_params = _merged_symbol_params(cfg, symbol_overrides.get(sym) or None)
        try:
            time_stop_minutes = int(symbol_params.get("time_stop_minutes") or 0)
        except Exception:
            time_stop_minutes = 0
        if time_stop_minutes <= 0:
            continue
        cutoff = entry_dt + dt.timedelta(minutes=time_stop_minutes)
        if now < cutoff:
            continue
        try:
            broker.cancel_orders(symbol=sym)
        except Exception as exc:
            logging.error("[LIVE] time stop cancel orders failed symbol=%s error=%s", sym, exc)
        try:
            broker.close_position(sym)
            closed.append(sym)
            logging.info(
                "[LIVE] time_stop close symbol=%s minutes=%s entry=%s cutoff=%s now=%s",
                sym,
                time_stop_minutes,
                entry_dt,
                cutoff,
                now,
            )
        except Exception as exc:
            logging.error("[LIVE] time stop close failed symbol=%s error=%s", sym, exc)
    return closed


def run_live(
    cfg: Optional[Dict] = None,
    target_date: Optional[str] = None,
    return_plans: bool = False,
    entry_time_et: Optional[str] = None,
    symbols_allow: Optional[set] = None,
) -> List[Dict] | tuple[List[Dict], List]:
    cfg = cfg or load_config()
    tgt = expected_watchlist_date_str(target_date)
    wl = read_watchlist(tgt, cfg)
    data_store = AlpacaOHLCStore(cfg=cfg)
    skip, info = market_filter_decision(tgt, cfg, data_store)
    params = cfg.get("daily_trend_reversal") or {}
    watch_cfg = cfg.get("watchlist") or {}
    entry_time_mode = str(watch_cfg.get("entry_time_mode") or "fixed").lower().strip()
    scan_first_valid_mode = entry_time_mode in {"scan_first_valid", "dynamic_first_valid", "scan"}
    debug = bool(params.get("live_order_debug") or params.get("live_verbose"))

    def _log_debug(msg: str, *args) -> None:
        if debug:
            logging.info("[LIVE_DEBUG] " + msg, *args)

    if skip:
        logging.info("[LIVE] market filter skip date=%s info=%s", tgt, info)
        return ([], []) if return_plans else []
    wl_rows = wl.get("watchlist") or []
    symbols = [str(r.get("symbol") or "").upper() for r in wl_rows if r.get("symbol")]
    skip_day, day_info = day_filter_decision(wl_rows, cfg, meta=wl.get("meta"))
    if skip_day:
        logging.info("[LIVE] day_filter skip date=%s info=%s", tgt, day_info)
        return ([], []) if return_plans else []
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
    symbol_watchlist_stats: Dict[str, Dict] = {}
    for idx, row in enumerate(wl_rows, start=1):
        sym = str((row or {}).get("symbol") or "").upper()
        if not sym:
            continue
        symbol_watchlist_stats[sym] = _watchlist_stats_from_row(row, idx)
    if symbols_allow is not None:
        symbols = [s for s in symbols if s in symbols_allow]
    symbols = sort_symbols_by_watchlist_priority(symbols, symbol_watchlist_stats)
    if debug and symbols:
        preview = []
        for s in symbols[:12]:
            st = symbol_watchlist_stats.get(s) or {}
            preview.append(
                {
                    "symbol": s,
                    "rank": st.get("rank"),
                    "selection_score": st.get("selection_score"),
                    "avgR": st.get("avgR"),
                    "trades_count": st.get("trades_count"),
                }
            )
        _log_debug("execution_priority_top=%s", preview)
    slot_target, slot_meta = _estimate_slot_target_for_live_pass(
        cfg,
        wl_rows,
        symbols,
        entry_time_et,
        scan_first_valid_mode,
    )
    entry_type = str(params.get("entry_order_type") or "market").lower()
    tif = str(params.get("order_tif") or "day").lower()
    use_brackets = bool(params.get("use_brackets", True))
    live_reanchor_brackets_on_fill = bool(params.get("live_reanchor_brackets_on_fill", True))
    live_reanchor_fill_timeout_sec = max(0.5, float(params.get("live_reanchor_fill_timeout_sec") or 30.0))
    live_reanchor_fill_poll_sec = max(0.05, float(params.get("live_reanchor_fill_poll_sec") or 0.25))
    live_reanchor_cancel_unfilled_entry = bool(params.get("live_reanchor_cancel_unfilled_entry", True))
    _log_debug("watchlist date=%s symbols=%s", tgt, len(symbols))
    _log_debug(
        "entry_type=%s tif=%s intraday_filter=%s early_range_minutes=%s time_stop_minutes=%s entry_time_override=%s reanchor_brackets_on_fill=%s",
        entry_type,
        tif,
        bool(params.get("intraday_filter_enabled", False)),
        params.get("early_range_minutes"),
        params.get("time_stop_minutes"),
        entry_time_et,
        live_reanchor_brackets_on_fill,
    )
    _log_debug(
        "slot_distribution enabled=%s slot_target=%s meta=%s",
        bool((cfg.get("daily_trend_reversal") or {}).get("slot_distribution_enabled", False)),
        slot_target,
        slot_meta,
    )
    if not symbols:
        logging.warning("[LIVE] no watchlist entries for date=%s", tgt)
        return ([], []) if return_plans else []
    broker = AlpacaBroker(cfg)
    equity_seed: Optional[float] = None
    allowed_total_seed: Optional[float] = None
    used_notional_runtime: Optional[float] = None
    open_positions_runtime: Optional[int] = None
    live_buying_power: Optional[float] = None
    live_buying_power_present = False
    live_non_marginable_buying_power: Optional[float] = None
    live_non_marginable_buying_power_present = False
    day_start_equity_seed: Optional[float] = None
    asset_margin_cache: Dict[str, Optional[float]] = {}
    if broker.ready():
        (
            equity_seed,
            allowed_total_seed,
            used_notional_runtime,
            open_positions_runtime,
            live_buying_power,
            live_buying_power_present,
            live_non_marginable_buying_power,
            live_non_marginable_buying_power_present,
        ) = _seed_runtime_exposure(
            cfg,
            broker,
            asset_margin_cache=asset_margin_cache,
            emit_debug=debug,
        )
        if equity_seed is not None and equity_seed > 0:
            day_start_equity_seed = float(equity_seed)
    placed: List[Dict] = []
    plans: List = []
    pending_reanchor_exits: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        _log_debug("symbol=%s start", symbol)
        wl_stats = symbol_watchlist_stats.get(symbol) or {}
        quality_ok, quality_reason = _passes_live_quality_gate(cfg, wl_stats)
        if not quality_ok:
            _log_debug(
                "symbol=%s skip_live_quality_gate reason=%s rank=%s selection_score=%s avgR=%s trades=%s",
                symbol,
                quality_reason,
                wl_stats.get("rank"),
                wl_stats.get("selection_score"),
                wl_stats.get("avgR"),
                wl_stats.get("trades_count"),
            )
            continue
        signal = generate_signal_for_date(symbol, tgt, cfg, data_store)
        if not signal:
            _log_debug("symbol=%s no_signal", symbol)
            continue
        if scan_first_valid_mode and entry_time_et:
            symbol_scan_start = str(symbol_entry_time.get(symbol) or "")
            if symbol_scan_start:
                try:
                    if parse_time_hhmm(str(entry_time_et)) < parse_time_hhmm(symbol_scan_start):
                        _log_debug(
                            "symbol=%s skip_scan_start current=%s start=%s",
                            symbol,
                            entry_time_et,
                            symbol_scan_start,
                        )
                        continue
                except Exception:
                    pass
        _log_debug(
            "symbol=%s signal direction=%s trend=%s return_pct=%.4f",
            symbol,
            signal.direction,
            signal.trend_state,
            signal.return_pct,
        )
        symbol_override = symbol_overrides.get(symbol) or {}
        symbol_params = _merged_symbol_params(cfg, symbol_override)
        symbol_entry_time_override = (
            entry_time_et
            or symbol_entry_time.get(symbol)
            or str(symbol_params.get("entry_time_et") or "09:35")
        )
        minutes_needed = _intraday_minutes_needed(symbol_params, symbol_entry_time_override)
        bars_intraday = None
        if minutes_needed > 0:
            bars_intraday = get_intraday_bars(symbol, tgt, minutes_needed, cfg=cfg, allow_fetch=True)
            _log_debug(
                "symbol=%s intraday_bars=%s minutes_needed=%s",
                symbol,
                len(bars_intraday) if bars_intraday else 0,
                minutes_needed,
            )
            if bars_intraday:
                _log_debug(
                    "symbol=%s intraday_first=%s intraday_last=%s",
                    symbol,
                    bars_intraday[0].get("timestamp") if bars_intraday else None,
                    bars_intraday[-1].get("timestamp") if bars_intraday else None,
                )
        bars_intraday_entry = bars_intraday
        confirm_move_bps = float(symbol_params.get("confirm_move_bps") or 0.0)
        confirm_minutes = int(symbol_params.get("confirm_minutes") or 0)
        entry_time_cutoff = _entry_cutoff_time(symbol_params, symbol_entry_time_override)
        # If confirmation is enabled, ensure we have enough intraday bars to evaluate it.
        # Otherwise a run at (e.g.) 09:35 will incorrectly treat "not yet" as "failed confirmation".
        if confirm_move_bps > 0 and confirm_minutes > 0:
            try:
                entry_time_str = symbol_entry_time_override
                entry_dt = ensure_et(dt.datetime.combine(dt.date.fromisoformat(tgt), parse_time_hhmm(entry_time_str)))
                cutoff_dt = entry_dt + dt.timedelta(minutes=confirm_minutes)
                if bars_intraday:
                    # Parity with replay/backtests: only include bars strictly before the cutoff time.
                    bars_intraday_entry = filter_intraday_bars_until(bars_intraday, tgt, entry_time_cutoff)
                required_last = cutoff_dt - dt.timedelta(minutes=1)
                last_ts = None
                if bars_intraday_entry:
                    last_ts = _parse_iso_ts(str(bars_intraday_entry[-1].get("timestamp") or ""))
                if last_ts is None or last_ts < required_last:
                    _log_debug(
                        "symbol=%s confirm_pending last_bar=%s required_last=%s (run after cutoff to evaluate confirm)",
                        symbol,
                        last_ts,
                        required_last,
                    )
                    continue
            except Exception:
                # If anything about the timestamps is weird, just fall back to the existing behavior.
                pass
        plan = build_trade(
                signal,
                cfg,
                data_store,
                context="live",
                bars_intraday=bars_intraday_entry,
                entry_time_override=symbol_entry_time_override,
                param_overrides=symbol_override or None,
            )
        if not plan:
            _log_debug("symbol=%s no_plan", symbol)
            continue
        plan.watchlist_stats = symbol_watchlist_stats.get(symbol) or None
        plans.append(plan)
        _log_debug(
            "symbol=%s plan entry_time=%s entry=%.4f stop=%.4f target=%.4f stop_dist=%.4f rr=%.2f",
            symbol,
            plan.entry_time_et,
            plan.entry_price,
            plan.stop_price,
            plan.target_price,
            plan.stop_distance,
            plan.target_rr,
        )
        if (
            getattr(plan, "signal_return_atr", None) is not None
            or getattr(plan, "open_noise_bps", None) is not None
        ):
            _log_debug(
                "symbol=%s quality signal_return_atr=%s open_noise_bps=%s open_noise_atr=%s open_noise_stop_ratio=%s stop_to_open_noise_ratio=%s",
                symbol,
                getattr(plan, "signal_return_atr", None),
                getattr(plan, "open_noise_bps", None),
                getattr(plan, "open_noise_atr", None),
                getattr(plan, "open_noise_stop_ratio", None),
                getattr(plan, "stop_to_open_noise_ratio", None),
            )
        base_price = None
        if entry_type == "market":
            latest = get_latest_intraday_prices([symbol], cfg=cfg, lookback_minutes=1)
            base_price = latest.get(symbol)
            if base_price is None:
                base_price = plan.entry_price
            _log_debug("symbol=%s base_price=%s", symbol, base_price)
        if broker.ready():
            try:
                acct_now = broker.get_account() or {}
                equity_now = _safe_float(acct_now.get("equity"), 0.0)
                bp_now, bp_now_present = _account_buying_power(acct_now)
                non_marginable_bp_now, non_marginable_bp_now_present = _account_non_marginable_buying_power(acct_now)
                if equity_now > 0:
                    equity_seed = equity_now
                    if day_start_equity_seed is None:
                        day_start_equity_seed = equity_now
                if bp_now_present:
                    max_margin_usage = float(params.get("max_margin_usage") or 0.70)
                    bp_budget = bp_now * max(0.0, min(max_margin_usage, 1.0))
                    if used_notional_runtime is not None:
                        allowed_total_seed = max(0.0, used_notional_runtime + bp_budget)
                    else:
                        allowed_total_seed = bp_budget
                    live_buying_power = bp_now
                    live_buying_power_present = True
                    live_non_marginable_buying_power = non_marginable_bp_now
                    live_non_marginable_buying_power_present = non_marginable_bp_now_present
                    _log_debug(
                        "symbol=%s broker_bp_snapshot equity=%.2f buying_power=%.2f non_marginable_bp=%.2f bp_budget=%.2f allowed_total_seed=%.2f used_notional_runtime=%s",
                        symbol,
                        equity_seed or 0.0,
                        bp_now,
                        non_marginable_bp_now,
                        bp_budget,
                        allowed_total_seed or 0.0,
                        used_notional_runtime,
                    )
            except Exception as exc:
                _log_debug("symbol=%s broker_bp_snapshot_failed error=%s", symbol, exc)
        qty, state, allowed_total, used_notional, open_positions = _compute_qty(
            cfg,
            plan,
            broker,
            slot_target=slot_target,
            equity_override=equity_seed,
            allowed_total_override=allowed_total_seed,
            used_notional_override=used_notional_runtime,
            open_positions_override=open_positions_runtime,
            day_start_equity=day_start_equity_seed,
        )
        _log_debug(
            "symbol=%s qty=%s allowed_total=%.2f used_notional=%.2f open_positions=%s state=%s",
            symbol,
            qty,
            allowed_total,
            used_notional,
            open_positions,
            state,
        )
        if qty > 0 and live_buying_power_present:
            ref_px = _safe_float(base_price, 0.0) if entry_type == "market" else _safe_float(plan.entry_price, 0.0)
            if ref_px > 0:
                open_side = "buy" if plan.direction == "long" else "sell"
                bp_buffer = resolve_effective_margin_buffer(params)
                live_bp_buffer_override = _safe_float_opt(params.get("live_buying_power_buffer"))
                if live_bp_buffer_override is not None:
                    bp_buffer = max(bp_buffer, max(0.0, min(live_bp_buffer_override, 0.9)))
                bp_budget = max(0.0, (live_buying_power or 0.0) * (1.0 - bp_buffer))
                live_non_marginable_bp_cap_enabled = bool(params.get("live_non_marginable_bp_cap_enabled", True))
                live_non_marginable_mmr_threshold_pct = max(
                    0.0,
                    float(params.get("live_non_marginable_mmr_threshold_pct") or 100.0),
                )
                live_non_marginable_bp_cap_when_mmr_missing = bool(
                    params.get("live_non_marginable_bp_cap_when_mmr_missing", True)
                )
                mmr_pct = _asset_maintenance_margin_pct(
                    broker,
                    symbol,
                    asset_margin_cache,
                )
                cap_non_marginable_reason: Optional[str] = None
                if mmr_pct is not None and mmr_pct >= live_non_marginable_mmr_threshold_pct:
                    cap_non_marginable_reason = "mmr_threshold"
                elif (
                    mmr_pct is None
                    and open_side == "sell"
                    and live_non_marginable_bp_cap_when_mmr_missing
                ):
                    cap_non_marginable_reason = "mmr_missing_short"
                if (
                    live_non_marginable_bp_cap_enabled
                    and live_non_marginable_buying_power_present
                    and (live_non_marginable_buying_power or 0.0) > 0.0
                    and cap_non_marginable_reason is not None
                ):
                    non_marginable_bp_budget = max(
                        0.0,
                        (live_non_marginable_buying_power or 0.0) * (1.0 - bp_buffer),
                    )
                    if non_marginable_bp_budget < bp_budget:
                        _log_debug(
                            "symbol=%s bp_budget_capped_non_marginable reason=%s mmr_pct=%s threshold=%.2f bp_budget=%.2f non_marginable_bp_budget=%.2f",
                            symbol,
                            cap_non_marginable_reason,
                            "None" if mmr_pct is None else f"{mmr_pct:.2f}",
                            live_non_marginable_mmr_threshold_pct,
                            bp_budget,
                            non_marginable_bp_budget,
                        )
                        bp_budget = non_marginable_bp_budget
                bp_unit_price = _bp_effective_price(
                    cfg,
                    open_side,
                    ref_px,
                    broker=broker,
                    symbol=symbol,
                    asset_margin_cache=asset_margin_cache,
                )
                bp_qty_cap = int(bp_budget // bp_unit_price) if bp_budget > 0 and bp_unit_price > 0 else 0
                if bp_qty_cap < qty:
                    _log_debug(
                        "symbol=%s qty_capped_by_broker_bp qty=%s cap=%s buying_power=%.2f bp_buffer=%.3f ref_px=%.4f bp_unit_px=%.4f side=%s",
                        symbol,
                        qty,
                        bp_qty_cap,
                        live_buying_power or 0.0,
                        bp_buffer,
                        ref_px,
                        bp_unit_price,
                        open_side,
                    )
                    qty = max(0, bp_qty_cap)
                    if isinstance(state, dict):
                        state["broker_buying_power"] = live_buying_power
                        state["broker_bp_qty_cap"] = bp_qty_cap
        if qty <= 0:
            continue
        if not broker.ready():
            logging.error("[LIVE] Alpaca credentials missing; cannot place order for %s", symbol)
            continue
        side = "buy" if plan.direction == "long" else "sell"
        if use_brackets and entry_type == "market" and live_reanchor_brackets_on_fill:
            pending_reanchor_exits[str(symbol).upper()] = {"plan": plan, "tif": tif}

        def _submit_reanchored_market_with_qty(order_qty: int) -> Dict[str, Any]:
            # Submit market entry first, then anchor exits to actual fill.
            entry_payload = {
                "symbol": symbol,
                "side": side,
                "qty": order_qty,
                "type": "market",
                "time_in_force": tif,
            }
            entry_resp = broker.submit_order(entry_payload)
            entry_order_id = str((entry_resp or {}).get("id") or "").strip()
            fill_info = _wait_for_entry_fill(
                broker,
                entry_order_id,
                expected_qty=order_qty,
                timeout_sec=live_reanchor_fill_timeout_sec,
                poll_sec=live_reanchor_fill_poll_sec,
                cancel_unfilled=live_reanchor_cancel_unfilled_entry,
            )
            if not fill_info:
                # Fallback for API timing gaps: infer fill from live position snapshot.
                pos_fill = _position_fill_snapshot(broker, symbol)
                if pos_fill:
                    fallback_order = {}
                    try:
                        fallback_order = broker.get_order(entry_order_id) or {}
                    except Exception:
                        fallback_order = {}
                    fill_info = {
                        "order": fallback_order,
                        "status": "position_snapshot",
                        "filled_qty": int(pos_fill.get("filled_qty") or 0),
                        "fill_price": _safe_float(pos_fill.get("fill_price"), 0.0),
                        "full": False,
                        "timed_out": True,
                        "source": "position_snapshot",
                    }
                    _log_debug(
                        "symbol=%s reanchor_fill_fallback source=position_snapshot order_id=%s filled_qty=%s fill_price=%.6f",
                        symbol,
                        entry_order_id,
                        fill_info.get("filled_qty"),
                        _safe_float(fill_info.get("fill_price"), 0.0),
                    )
                else:
                    latest_status = None
                    latest_filled_qty = None
                    latest_fill_px = None
                    try:
                        latest = broker.get_order(entry_order_id) or {}
                        latest_status = latest.get("status")
                        latest_filled_qty = latest.get("filled_qty")
                        latest_fill_px = latest.get("filled_avg_price")
                    except Exception:
                        pass
                    raise RuntimeError(
                        "entry fill unavailable for symbol=%s order_id=%s status=%s filled_qty=%s fill_px=%s"
                        % (symbol, entry_order_id, latest_status, latest_filled_qty, latest_fill_px)
                    )
            fill_qty = int(fill_info.get("filled_qty") or 0)
            fill_price = _safe_float(fill_info.get("fill_price"), 0.0)
            if fill_qty <= 0 or fill_price <= 0:
                raise RuntimeError(f"invalid entry fill for symbol={symbol} order_id={entry_order_id}")
            if bool(fill_info.get("timed_out")):
                _log_debug(
                    "symbol=%s reanchor_entry_fill_timed_out order_id=%s status=%s filled_qty=%s fill_price=%.6f",
                    symbol,
                    entry_order_id,
                    fill_info.get("status"),
                    fill_qty,
                    fill_price,
                )
            if fill_qty < order_qty:
                _log_debug(
                    "symbol=%s reanchor_entry_partial_fill order_id=%s requested_qty=%s filled_qty=%s",
                    symbol,
                    entry_order_id,
                    order_qty,
                    fill_qty,
                )

            stop_price_anchored, target_price_anchored, stop_offset, target_offset = _compute_reanchored_exit_prices(
                plan,
                fill_price,
            )
            close_side = "sell" if side == "buy" else "buy"
            try:
                exit_resp = broker.submit_oco_exit_order(
                    symbol=symbol,
                    side=close_side,
                    qty=fill_qty,
                    take_profit=target_price_anchored,
                    stop_loss=stop_price_anchored,
                    base_price=fill_price,
                    tif=tif,
                )
            except Exception:
                # Never leave a fresh fill without a protective exit order.
                try:
                    broker.close_position(symbol)
                    logging.error("[LIVE] emergency close submitted symbol=%s after exit-order failure", symbol)
                except Exception as close_exc:
                    logging.error("[LIVE] emergency close failed symbol=%s error=%s", symbol, close_exc)
                raise

            _log_debug(
                "symbol=%s reanchor_exits_on_fill entry_fill=%.6f stop=%.6f target=%.6f stop_offset=%.6f target_offset=%.6f close_side=%s",
                symbol,
                fill_price,
                stop_price_anchored,
                target_price_anchored,
                stop_offset,
                target_offset,
                close_side,
            )
            combined = dict(entry_resp or {})
            combined["entry_order"] = entry_resp
            combined["exit_order"] = exit_resp
            combined["filled_qty_used"] = fill_qty
            combined["filled_avg_price_used"] = fill_price
            combined["reanchored_stop_price"] = stop_price_anchored
            combined["reanchored_target_price"] = target_price_anchored
            combined["reanchor_enabled"] = True
            return combined

        def _submit_with_qty(order_qty: int) -> Dict[str, Any]:
            if use_brackets and entry_type == "market" and live_reanchor_brackets_on_fill:
                return _submit_reanchored_market_with_qty(order_qty)
            if use_brackets:
                return broker.submit_bracket_order(
                    symbol=symbol,
                    side=side,
                    qty=order_qty,
                    entry_type=entry_type,
                    entry_price=plan.entry_price if entry_type == "limit" else None,
                    base_price=base_price,
                    take_profit=plan.target_price,
                    stop_loss=plan.stop_price,
                    tif=tif,
                )
            payload = {
                "symbol": symbol,
                "side": side,
                "qty": order_qty,
                "type": entry_type,
                "time_in_force": tif,
            }
            if entry_type == "limit":
                payload["limit_price"] = plan.entry_price
            return broker.submit_order(payload)

        def _record_success(resp_obj: Dict[str, Any], placed_qty_val: int) -> None:
            nonlocal used_notional_runtime, open_positions_runtime, pending_reanchor_exits
            effective_qty = int(_safe_float((resp_obj or {}).get("filled_qty_used"), float(placed_qty_val)) or 0)
            if effective_qty <= 0:
                effective_qty = int(max(0, placed_qty_val))
            placed.append(resp_obj)
            logging.info("[LIVE] order placed symbol=%s side=%s qty=%s", symbol, side, effective_qty)
            _log_debug("symbol=%s order_response=%s", symbol, resp_obj)
            pending_reanchor_exits.pop(str(symbol).upper(), None)
            if used_notional_runtime is not None and open_positions_runtime is not None:
                reserve_qty = int((state or {}).get("capacity_qty") or effective_qty)
                if (resp_obj or {}).get("filled_qty_used") is not None:
                    reserve_qty = min(max(0, reserve_qty), max(0, effective_qty))
                reserve_unit_price = _safe_float((state or {}).get("capacity_notional_per_share"), 0.0)
                reserve_px = _safe_float((resp_obj or {}).get("filled_avg_price_used"), 0.0)
                if entry_type == "market":
                    reserve_px = max(reserve_px, _safe_float(base_price, 0.0))
                if reserve_px <= 0:
                    reserve_px = _safe_float(getattr(plan, "entry_price", 0.0), 0.0)
                reserve_side = "buy" if plan.direction == "long" else "sell"
                reserve_unit_price_live = _bp_effective_price(
                    cfg,
                    reserve_side,
                    reserve_px,
                    broker=broker,
                    symbol=symbol,
                    asset_margin_cache=asset_margin_cache,
                )
                if reserve_unit_price <= 0:
                    reserve_unit_price = reserve_unit_price_live
                else:
                    reserve_unit_price = max(reserve_unit_price, reserve_unit_price_live)
                reserve_notional = max(0.0, reserve_qty * reserve_unit_price)
                used_notional_runtime += reserve_notional
                open_positions_runtime += 1
                _log_debug(
                    "symbol=%s reserve_notional=%.2f reserve_qty=%s reserve_px=%.4f reserve_unit_price=%.4f used_notional_runtime=%.2f open_positions_runtime=%s",
                    symbol,
                    reserve_notional,
                    reserve_qty,
                    reserve_px,
                    reserve_unit_price,
                    used_notional_runtime,
                    open_positions_runtime,
                )
        try:
            resp = _submit_with_qty(qty)
            _record_success(resp, qty)
        except Exception as exc:
            logging.error("[LIVE] order failed symbol=%s error=%s", symbol, exc)
            resp = getattr(exc, "response", None)
            status_code = None
            body_text = ""
            if resp is not None:
                try:
                    status_code = resp.status_code
                    body_text = str(resp.text or "")
                    logging.error("[LIVE] order failed symbol=%s status=%s body=%s", symbol, resp.status_code, resp.text)
                except Exception:
                    pass
            body_lower = body_text.lower()
            if status_code in (403, 422) and "buying power" in body_lower:
                (
                    equity_seed,
                    allowed_total_seed,
                    used_notional_runtime,
                    open_positions_runtime,
                    live_buying_power,
                    live_buying_power_present,
                    live_non_marginable_buying_power,
                    live_non_marginable_buying_power_present,
                ) = _seed_runtime_exposure(
                    cfg,
                    broker,
                    asset_margin_cache=asset_margin_cache,
                    emit_debug=False,
                )
                _log_debug(
                    "symbol=%s exposure_resync_after_bp_reject equity=%.2f allowed_total=%.2f used_notional=%.2f open_positions=%s buying_power=%.2f bp_present=%s non_marginable_bp=%.2f non_marginable_bp_present=%s",
                    symbol,
                    equity_seed or 0.0,
                    allowed_total_seed or 0.0,
                    used_notional_runtime or 0.0,
                    open_positions_runtime or 0,
                    live_buying_power or 0.0,
                    live_buying_power_present,
                    live_non_marginable_buying_power or 0.0,
                    live_non_marginable_buying_power_present,
                )
    _repair_pending_reanchor_exits(
        cfg,
        broker,
        pending_reanchor_exits,
        debug=debug,
    )
    flatten_intraday_positions_if_needed(cfg, broker)
    return (placed, plans) if return_plans else placed


def run_flatten(cfg: Optional[Dict] = None) -> List[Dict]:
    cfg = cfg or load_config()
    broker = AlpacaBroker(cfg)
    return flatten_intraday_positions_if_needed(cfg, broker)
