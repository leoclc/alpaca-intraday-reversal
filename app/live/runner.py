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
from app.watchlist.storage import expected_watchlist_date_str, read_watchlist


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

    allowed_total *= max(0.0, min(max_margin_usage, 1.0))
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
    time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    if time_stop_minutes <= 0:
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
            logging.info("[LIVE] time_stop close symbol=%s entry=%s cutoff=%s now=%s", sym, entry_dt, cutoff, now)
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
    debug = bool(params.get("live_order_debug") or params.get("live_verbose"))

    def _log_debug(msg: str, *args) -> None:
        if debug:
            logging.info("[LIVE_DEBUG] " + msg, *args)

    if skip:
        logging.info("[LIVE] market filter skip date=%s info=%s", tgt, info)
        return ([], []) if return_plans else []
    symbols = [str(r.get("symbol") or "").upper() for r in wl.get("watchlist") or [] if r.get("symbol")]
    symbol_overrides = {
        str(r.get("symbol") or "").upper(): (r.get("param_overrides") or {})
        for r in wl.get("watchlist") or []
        if r.get("symbol")
    }
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
        _log_debug(
            "symbol=%s signal direction=%s trend=%s return_pct=%.4f",
            symbol,
            signal.direction,
            signal.trend_state,
            signal.return_pct,
        )
        params = cfg.get("daily_trend_reversal") or {}
        intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
        early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
        time_stop_minutes = int(params.get("time_stop_minutes") or 0)
        confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
        confirm_minutes = int(params.get("confirm_minutes") or 0)
        minutes_needed = max(early_range_minutes, time_stop_minutes)
        try:
            entry_time_str = entry_time_et or str(params.get("entry_time_et") or "09:35")
            session_open_et = str(params.get("session_open_et") or "09:30")
            entry_time = parse_time_hhmm(entry_time_str)
            open_time = parse_time_hhmm(session_open_et)
            entry_minutes = int(
                (dt.datetime.combine(dt.date.today(), entry_time) - dt.datetime.combine(dt.date.today(), open_time)).total_seconds()
                / 60
            )
            entry_minutes = max(1, entry_minutes + 1)
            if bool(params.get("use_intraday_entry", False)):
                minutes_needed = max(minutes_needed, entry_minutes)
            if confirm_move_bps > 0 and confirm_minutes > 0:
                minutes_needed = max(minutes_needed, entry_minutes + confirm_minutes)
        except Exception:
            minutes_needed = max(minutes_needed, 1)
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
        # If confirmation is enabled, ensure we have enough intraday bars to evaluate it.
        # Otherwise a run at (e.g.) 09:35 will incorrectly treat "not yet" as "failed confirmation".
        if confirm_move_bps > 0 and confirm_minutes > 0:
            try:
                entry_time_str = entry_time_et or str(params.get("entry_time_et") or "09:35")
                entry_dt = ensure_et(dt.datetime.combine(dt.date.fromisoformat(tgt), parse_time_hhmm(entry_time_str)))
                cutoff_dt = entry_dt + dt.timedelta(minutes=confirm_minutes)
                cutoff_time_et = cutoff_dt.strftime("%H:%M")
                if bars_intraday:
                    # Parity with replay/backtests: only include bars strictly before the cutoff time.
                    bars_intraday_entry = filter_intraday_bars_until(bars_intraday, tgt, cutoff_time_et)
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
            entry_time_override=entry_time_et,
            param_overrides=symbol_overrides.get(symbol) or None,
        )
        if not plan:
            _log_debug("symbol=%s no_plan", symbol)
            continue
        plans.append(plan)
        _log_debug(
            "symbol=%s plan entry=%.4f stop=%.4f target=%.4f stop_dist=%.4f rr=%.2f",
            symbol,
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
