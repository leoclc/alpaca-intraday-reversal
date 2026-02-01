from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from app.brokers.alpaca import AlpacaBroker
from app.data.alpaca_intraday_store import get_latest_intraday_prices
from app.strategies.types import TradePlan
from app.utils.time import et_now


def log_account_summary(broker: AlpacaBroker) -> None:
    try:
        acct = broker.get_account()
    except Exception as exc:
        logging.error("[LIVE] account summary failed: %s", exc)
        return
    try:
        positions = broker.list_positions() or []
    except Exception:
        positions = []
    equity = acct.get("equity")
    buying_power = acct.get("buying_power")
    multiplier = acct.get("multiplier")
    cash = acct.get("cash")
    logging.info(
        "[LIVE] account equity=%s buying_power=%s multiplier=%s cash=%s open_positions=%s",
        equity,
        buying_power,
        multiplier,
        cash,
        len(positions),
    )


def _pnl_for_position(
    pos: Dict,
    current_price: Optional[float],
) -> Tuple[float, float]:
    try:
        qty = float(pos.get("qty") or pos.get("quantity") or 0.0)
    except Exception:
        qty = 0.0
    try:
        entry = float(pos.get("avg_entry_price") or pos.get("entry_price") or 0.0)
    except Exception:
        entry = 0.0
    if not current_price or entry <= 0 or qty == 0:
        return 0.0, 0.0
    side = str(pos.get("side") or pos.get("direction") or "").lower()
    if side == "short" or (pos.get("side") == "sell"):
        pnl_per = entry - current_price
    else:
        pnl_per = current_price - entry
    total = pnl_per * qty
    pct = (pnl_per / entry) * 100.0
    return total, pct


def log_minute_report(
    cfg: Dict,
    broker: AlpacaBroker,
    plans: Iterable[TradePlan],
    *,
    max_symbols: Optional[int] = None,
    lookback_minutes: Optional[int] = None,
) -> None:
    params = cfg.get("daily_trend_reversal") or {}
    max_symbols = int(max_symbols or params.get("minute_report_max_symbols") or 20)
    lookback_minutes = int(lookback_minutes or params.get("minute_report_lookback_minutes") or 5)
    now = et_now()
    try:
        positions = broker.list_positions() or []
    except Exception:
        positions = []

    plan_list = list(plans)
    symbols: List[str] = []
    for p in positions:
        sym = str(p.get("symbol") or "").upper()
        if sym:
            symbols.append(sym)
    for plan in plan_list:
        sym = str(plan.symbol or "").upper()
        if sym:
            symbols.append(sym)
    symbols = list(dict.fromkeys(symbols))

    prices = get_latest_intraday_prices(symbols, cfg=cfg, lookback_minutes=lookback_minutes)

    logging.info("=" * 60)
    logging.info("[MINUTE REPORT] %s ET", now.strftime("%H:%M:%S"))
    logging.info("=" * 60)

    if positions:
        logging.info("ACTIVE POSITIONS (%s):", len(positions))
        total_unreal = 0.0
        for pos in positions:
            sym = str(pos.get("symbol") or "").upper()
            cur = prices.get(sym)
            total, pct = _pnl_for_position(pos, cur)
            total_unreal += total
            logging.info(
                "  %s qty=%s entry=%s current=%s pnl=%+.2f (%+.2f%%)",
                sym,
                pos.get("qty") or pos.get("quantity"),
                pos.get("avg_entry_price") or pos.get("entry_price"),
                f"{cur:.2f}" if cur is not None else "n/a",
                total,
                pct,
            )
        logging.info("  TOTAL UNREALIZED PNL: %+0.2f", total_unreal)
    else:
        logging.info("ACTIVE POSITIONS: none")

    if plan_list:
        rows = []
        for plan in plan_list:
            sym = str(plan.symbol or "").upper()
            cur = prices.get(sym)
            if cur is None or plan.entry_price <= 0:
                continue
            dist_pct = abs((cur - plan.entry_price) / plan.entry_price) * 100.0
            rows.append(
                {
                    "symbol": sym,
                    "direction": plan.direction,
                    "entry": plan.entry_price,
                    "current": cur,
                    "dist_pct": dist_pct,
                }
            )
        rows.sort(key=lambda r: r["dist_pct"])
        rows = rows[: max_symbols]
        logging.info("WATCHLIST DISTANCE (top %s):", max_symbols)
        for row in rows:
            logging.info(
                "  %s %s entry=%.2f current=%.2f dist=%.2f%%",
                row["symbol"],
                row["direction"],
                row["entry"],
                row["current"],
                row["dist_pct"],
            )
