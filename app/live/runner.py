from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional

from app.brokers.alpaca import AlpacaBroker
from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.data.alpaca_intraday_store import (
    filter_intraday_bars_until,
    get_intraday_bars,
    get_latest_intraday_prices,
)
from app.market.filters import market_filter_decision
from app.portfolio.sizing import compute_qty_with_guards
from app.strategies.daily_trend_reversal import build_trade, generate_signal_for_date
from app.utils.time import ensure_et, et_now, parse_time_hhmm
from app.watchlist.day_filter import day_filter_decision
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


def _compute_qty(cfg: Dict, plan, broker: AlpacaBroker) -> tuple[int, Dict, float, float, int]:
    params = cfg.get("daily_trend_reversal") or {}
    fixed_qty = params.get("fixed_qty")
    if fixed_qty:
        return int(fixed_qty), {"fixed_qty": int(fixed_qty)}, 0.0, 0.0, 0
    risk_per_trade = float(params.get("risk_per_trade") or 0.0)
    if risk_per_trade <= 0:
        return 1, {"risk_per_trade": risk_per_trade}, 0.0, 0.0, 0
    acct = broker.get_account() if broker.ready() else None
    if not acct:
        return 1, {"account": "missing"}, 0.0, 0.0, 0
    try:
        equity = float(acct.get("equity") or 0.0)
    except Exception:
        return 1, {"equity": "parse_error"}, 0.0, 0.0, 0

    leverage_cfg = params.get("leverage")
    max_margin_usage = float(params.get("max_margin_usage") or 0.70)
    try:
        leverage = float(leverage_cfg) if leverage_cfg is not None else float(acct.get("multiplier") or 4.0)
    except Exception:
        leverage = 4.0
    try:
        buying_power = float(acct.get("buying_power") or 0.0)
    except Exception:
        buying_power = 0.0
    allowed_total = max(0.0, equity * leverage)
    if buying_power > 0:
        allowed_total = min(allowed_total, buying_power)
    allowed_total *= max(0.0, min(max_margin_usage, 1.0))

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

    qty, state = compute_qty_with_guards(
        plan,
        equity,
        used_notional,
        cfg,
        allowed_total_override=allowed_total,
        open_positions=len(positions),
    )
    return qty, state, allowed_total, used_notional, len(positions)


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
    entry_type = str(params.get("entry_order_type") or "market").lower()
    tif = str(params.get("order_tif") or "day").lower()
    use_brackets = bool(params.get("use_brackets", True))
    _log_debug("watchlist date=%s symbols=%s", tgt, len(symbols))
    _log_debug(
        "entry_type=%s tif=%s intraday_filter=%s early_range_minutes=%s time_stop_minutes=%s entry_time_override=%s",
        entry_type,
        tif,
        bool(params.get("intraday_filter_enabled", False)),
        params.get("early_range_minutes"),
        params.get("time_stop_minutes"),
        entry_time_et,
    )
    if not symbols:
        logging.warning("[LIVE] no watchlist entries for date=%s", tgt)
        return ([], []) if return_plans else []
    broker = AlpacaBroker(cfg)
    placed: List[Dict] = []
    plans: List = []
    for symbol in symbols:
        _log_debug("symbol=%s start", symbol)
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
        base_price = None
        if entry_type == "market":
            latest = get_latest_intraday_prices([symbol], cfg=cfg, lookback_minutes=1)
            base_price = latest.get(symbol)
            if base_price is None:
                base_price = plan.entry_price
            _log_debug("symbol=%s base_price=%s", symbol, base_price)
        qty, state, allowed_total, used_notional, open_positions = _compute_qty(cfg, plan, broker)
        _log_debug(
            "symbol=%s qty=%s allowed_total=%.2f used_notional=%.2f open_positions=%s state=%s",
            symbol,
            qty,
            allowed_total,
            used_notional,
            open_positions,
            state,
        )
        if qty <= 0:
            continue
        if not broker.ready():
            logging.error("[LIVE] Alpaca credentials missing; cannot place order for %s", symbol)
            continue
        side = "buy" if plan.direction == "long" else "sell"
        try:
            if use_brackets:
                resp = broker.submit_bracket_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    entry_type=entry_type,
                    entry_price=plan.entry_price if entry_type == "limit" else None,
                    base_price=base_price,
                    take_profit=plan.target_price,
                    stop_loss=plan.stop_price,
                    tif=tif,
                )
            else:
                payload = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "type": entry_type,
                    "time_in_force": tif,
                }
                if entry_type == "limit":
                    payload["limit_price"] = plan.entry_price
                resp = broker.submit_order(payload)
            placed.append(resp)
            logging.info("[LIVE] order placed symbol=%s side=%s qty=%s", symbol, side, qty)
            _log_debug("symbol=%s order_response=%s", symbol, resp)
        except Exception as exc:
            logging.error("[LIVE] order failed symbol=%s error=%s", symbol, exc)
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    logging.error("[LIVE] order failed symbol=%s status=%s body=%s", symbol, resp.status_code, resp.text)
                except Exception:
                    pass
    flatten_intraday_positions_if_needed(cfg, broker)
    return (placed, plans) if return_plans else placed


def run_flatten(cfg: Optional[Dict] = None) -> List[Dict]:
    cfg = cfg or load_config()
    broker = AlpacaBroker(cfg)
    return flatten_intraday_positions_if_needed(cfg, broker)
