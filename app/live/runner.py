from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional

from app.brokers.alpaca import AlpacaBroker
from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.data.alpaca_intraday_store import get_intraday_bars
from app.portfolio.sizing import compute_qty_with_guards
from app.strategies.daily_trend_reversal import build_trade, generate_signal_for_date
from app.utils.time import et_now, parse_time_hhmm
from app.watchlist.storage import expected_watchlist_date_str, read_watchlist


def _compute_qty(cfg: Dict, plan, broker: AlpacaBroker) -> int:
    params = cfg.get("daily_trend_reversal") or {}
    fixed_qty = params.get("fixed_qty")
    if fixed_qty:
        return int(fixed_qty)
    risk_per_trade = float(params.get("risk_per_trade") or 0.0)
    if risk_per_trade <= 0:
        return 1
    acct = broker.get_account() if broker.ready() else None
    if not acct:
        return 1
    try:
        equity = float(acct.get("equity") or 0.0)
    except Exception:
        return 1

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
    qty, _ = compute_qty_with_guards(
        plan,
        equity,
        used_notional,
        cfg,
        allowed_total_override=allowed_total,
        open_positions=len(positions),
    )
    return qty


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


def run_live(
    cfg: Optional[Dict] = None,
    target_date: Optional[str] = None,
    return_plans: bool = False,
) -> List[Dict] | tuple[List[Dict], List]:
    cfg = cfg or load_config()
    tgt = expected_watchlist_date_str(target_date)
    wl = read_watchlist(tgt, cfg)
    symbols = [str(r.get("symbol") or "").upper() for r in wl.get("watchlist") or [] if r.get("symbol")]
    if not symbols:
        logging.warning("[LIVE] no watchlist entries for date=%s", tgt)
        return ([], []) if return_plans else []
    data_store = AlpacaOHLCStore(cfg=cfg)
    broker = AlpacaBroker(cfg)
    params = cfg.get("daily_trend_reversal") or {}
    entry_type = str(params.get("entry_order_type") or "market").lower()
    tif = str(params.get("order_tif") or "day").lower()
    use_brackets = bool(params.get("use_brackets", True))
    placed: List[Dict] = []
    plans: List = []
    for symbol in symbols:
        signal = generate_signal_for_date(symbol, tgt, cfg, data_store)
        if not signal:
            continue
        params = cfg.get("daily_trend_reversal") or {}
        intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
        early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
        time_stop_minutes = int(params.get("time_stop_minutes") or 0)
        minutes_needed = max(early_range_minutes, time_stop_minutes)
        bars_intraday = None
        if minutes_needed > 0:
            bars_intraday = get_intraday_bars(symbol, tgt, minutes_needed, cfg=cfg, allow_fetch=True)
        plan = build_trade(signal, cfg, data_store, context="live", bars_intraday=bars_intraday)
        if not plan:
            continue
        plans.append(plan)
        qty = _compute_qty(cfg, plan, broker)
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
        except Exception as exc:
            logging.error("[LIVE] order failed symbol=%s error=%s", symbol, exc)
    flatten_intraday_positions_if_needed(cfg, broker)
    return (placed, plans) if return_plans else placed


def run_flatten(cfg: Optional[Dict] = None) -> List[Dict]:
    cfg = cfg or load_config()
    broker = AlpacaBroker(cfg)
    return flatten_intraday_positions_if_needed(cfg, broker)
