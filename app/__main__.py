from __future__ import annotations

import datetime as dt
import logging
import time

from app.config.loader import load_config
from app.brokers.alpaca import AlpacaBroker
from app.live.minute_report import log_account_summary, log_minute_report
from app.live.runner import enforce_time_stop, run_flatten, run_live
from app.strategies.daily_trend_reversal import resolve_entry_times
from app.utils.time import ensure_et, et_now, parse_time_hhmm
from app.watchlist.daily_strategy_builder import build_watchlist
from app.watchlist.storage import read_watchlist, expected_watchlist_date_str


def _sleep_until(target: dt.datetime) -> None:
    while True:
        now = et_now()
        if now >= target:
            return
        remaining = (target - now).total_seconds()
        time.sleep(min(30.0, max(1.0, remaining)))


def _ordered_entry_time_strs(cfg: dict) -> list[str]:
    params = cfg.get("daily_trend_reversal") or {}
    watch_cfg = cfg.get("watchlist") or {}
    sort_mode = str(watch_cfg.get("entry_time_sort_mode") or "asc").lower().strip()
    return resolve_entry_times(params, sort_mode=sort_mode)


def _schedule_for_day(cfg: dict, day: dt.date) -> tuple[dt.datetime, list[dt.datetime], dt.datetime]:
    params = cfg.get("daily_trend_reversal") or {}
    watchlist_time_str = str(params.get("watchlist_time_et") or "03:00")
    confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
    confirm_minutes = int(params.get("confirm_minutes") or 0)
    apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0
    entry_time_strs = _ordered_entry_time_strs(cfg)
    session_close_str = str(params.get("session_close_et") or "16:00")
    flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)

    watchlist_time = parse_time_hhmm(watchlist_time_str)
    entry_times = [parse_time_hhmm(t) for t in entry_time_strs]
    close_time = parse_time_hhmm(session_close_str)

    # NOTE: ET_TZ is a pytz timezone in our environment; we must localize (ensure_et)
    # rather than doing .replace(tzinfo=ET_TZ), otherwise DST/LMT offsets are wrong.
    base = ensure_et(dt.datetime.combine(day, watchlist_time))
    entry_dts = [
        ensure_et(
            dt.datetime.combine(day, t)
            + (dt.timedelta(minutes=confirm_minutes) if apply_confirm else dt.timedelta())
        )
        for t in entry_times
    ]
    close_dt = ensure_et(dt.datetime.combine(day, close_time))
    flatten_dt = close_dt - dt.timedelta(minutes=flatten_buffer)
    return base, entry_dts, flatten_dt


def _parse_iso_ts(value: object) -> dt.datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    return ensure_et(parsed)


def _safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _close_reason(order: dict) -> str:
    order_type = str(order.get("type") or order.get("order_type") or "").lower().strip()
    if order_type in {"stop", "stop_limit", "trailing_stop"}:
        return "stop"
    if order_type == "limit":
        return "target_or_limit"
    if order_type == "market":
        return "market_close"
    return order_type or "close"


def _log_new_position_closes(
    cfg: dict,
    broker: AlpacaBroker,
    day: dt.date,
    seen_order_ids: set[str],
) -> None:
    params = cfg.get("daily_trend_reversal") or {}
    session_open_et = str(params.get("session_open_et") or "09:30")
    open_dt = ensure_et(dt.datetime.combine(day, parse_time_hhmm(session_open_et)))
    after_dt = open_dt - dt.timedelta(minutes=1)
    try:
        orders = broker.list_orders(status="closed", after=after_dt.isoformat(), limit=500) or []
    except Exception as exc:
        logging.error("[LIVE] close monitor failed to list orders: %s", exc)
        return

    def _order_sort_key(order: dict) -> str:
        return str(order.get("filled_at") or order.get("updated_at") or order.get("submitted_at") or "")

    sorted_orders = sorted(orders, key=_order_sort_key)

    # Build an intraday open-lot ledger from broker orders so each close log can include
    # realized PnL (USD and %) based on actual filled open/close prices.
    open_lots: dict[str, list[dict[str, float | str]]] = {}
    close_metrics: dict[str, dict[str, float | None]] = {}
    for order in sorted_orders:
        if str(order.get("status") or "").lower().strip() != "filled":
            continue
        order_id = str(order.get("id") or "")
        symbol = str(order.get("symbol") or "").upper().strip()
        side = str(order.get("side") or "").lower().strip()
        qty_val = _safe_float(order.get("filled_qty") or order.get("qty"))
        fill_val = _safe_float(order.get("filled_avg_price"))
        if not order_id or not symbol or side not in {"buy", "sell"} or not qty_val or not fill_val:
            continue

        intent = str(order.get("position_intent") or "").lower().strip()
        is_open = "open" in intent
        is_close = "close" in intent
        if not is_open and not is_close:
            # Fallback for APIs that omit position_intent on parent bracket entries.
            order_class = str(order.get("order_class") or "").lower().strip()
            if order_class == "bracket":
                is_open = True

        if is_open:
            lots = open_lots.setdefault(symbol, [])
            lots.append({"side": side, "qty": float(qty_val), "price": float(fill_val)})
            continue

        if not is_close:
            continue

        lots = open_lots.get(symbol) or []
        remaining = float(qty_val)
        matched_qty = 0.0
        realized = 0.0
        cost_basis = 0.0
        while remaining > 1e-9 and lots:
            match_idx = None
            for i, lot in enumerate(lots):
                lot_side = str(lot.get("side") or "").lower().strip()
                lot_qty = float(lot.get("qty") or 0.0)
                if lot_qty > 1e-9 and lot_side in {"buy", "sell"} and lot_side != side:
                    match_idx = i
                    break
            if match_idx is None:
                break

            lot = lots[match_idx]
            lot_qty = float(lot.get("qty") or 0.0)
            lot_price = float(lot.get("price") or 0.0)
            take_qty = min(remaining, lot_qty)
            if take_qty <= 0:
                break

            if side == "sell":
                pnl_per_share = float(fill_val) - lot_price
            else:
                pnl_per_share = lot_price - float(fill_val)
            realized += pnl_per_share * take_qty
            cost_basis += lot_price * take_qty
            matched_qty += take_qty
            remaining -= take_qty
            lot["qty"] = lot_qty - take_qty
            if float(lot.get("qty") or 0.0) <= 1e-9:
                lots.pop(match_idx)

        close_metrics[order_id] = {
            "entry_price": ((cost_basis / matched_qty) if matched_qty > 0 else None),
            "pnl_usd": (realized if matched_qty > 0 else None),
            "pnl_pct": ((realized / cost_basis) * 100.0 if cost_basis > 0 else None),
            "matched_qty": (matched_qty if matched_qty > 0 else None),
            "unmatched_qty": (remaining if remaining > 1e-9 else None),
        }

    for order in sorted_orders:
        order_id = str(order.get("id") or "")
        if not order_id or order_id in seen_order_ids:
            continue
        seen_order_ids.add(order_id)

        if str(order.get("status") or "").lower().strip() != "filled":
            continue

        position_intent = str(order.get("position_intent") or "").lower().strip()
        if "close" not in position_intent:
            continue

        symbol = str(order.get("symbol") or "").upper()
        qty = str(order.get("filled_qty") or order.get("qty") or "")
        fill = str(order.get("filled_avg_price") or "")
        side = str(order.get("side") or "").lower().strip()
        reason = _close_reason(order)
        metrics = close_metrics.get(order_id) or {}
        entry_price = metrics.get("entry_price")
        pnl_usd = metrics.get("pnl_usd")
        pnl_pct = metrics.get("pnl_pct")
        pnl_str = "n/a" if pnl_usd is None else f"{pnl_usd:+.2f}"
        pnl_pct_str = "n/a" if pnl_pct is None else f"{pnl_pct:+.2f}%"
        entry_str = "n/a" if entry_price is None else f"{entry_price:.6f}"
        ts = _parse_iso_ts(order.get("filled_at") or order.get("updated_at") or order.get("submitted_at"))
        ts_str = ts.strftime("%H:%M:%S ET") if ts else "n/a"
        logging.info(
            "[LIVE] position_closed symbol=%s reason=%s side=%s qty=%s entry=%s exit=%s pnl=%s pnl_pct=%s at=%s order_id=%s",
            symbol,
            reason,
            side,
            qty,
            entry_str,
            fill,
            pnl_str,
            pnl_pct_str,
            ts_str,
            order_id[:8],
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cfg = load_config()
    params = cfg.get("daily_trend_reversal") or {}
    watchlist_time_str = str(params.get("watchlist_time_et") or "03:00")
    entry_time_strs = _ordered_entry_time_strs(cfg)
    intraday_only = bool(params.get("intraday_only", False))
    minute_report_enabled = bool(params.get("minute_report_enabled", False))
    close_report_enabled = bool(params.get("position_close_report_enabled", True))
    monitor_interval = int(
        params.get("position_close_poll_interval_sec")
        or params.get("minute_report_interval_sec")
        or 30
    )
    watch_cfg = cfg.get("watchlist") or {}
    entry_time_mode = str(watch_cfg.get("entry_time_mode") or "fixed").lower().strip()
    scan_first_valid_mode = entry_time_mode in {"scan_first_valid", "dynamic_first_valid", "scan"}

    last_watchlist_date: dt.date | None = None
    last_entry_date: dt.date | None = None
    last_flatten_date: dt.date | None = None
    close_monitor_date: dt.date | None = None
    seen_close_order_ids: set[str] = set()
    plans: list = []
    entry_time_map: dict[str, str] = {}

    while True:
        now = et_now()
        today = now.date()
        watchlist_dt, entry_dts, flatten_dt = _schedule_for_day(cfg, today)
        last_entry_dt = entry_dts[-1] if entry_dts else None
        if today.weekday() >= 5:
            logging.info("[LIVE] weekend detected; skipping %s", today.isoformat())
            tomorrow = today + dt.timedelta(days=1)
            next_watchlist_dt, _, _ = _schedule_for_day(cfg, tomorrow)
            _sleep_until(next_watchlist_dt)
            continue

        # If we're already past entry time, skip the day entirely.
        if last_entry_dt and now >= last_entry_dt and last_entry_date != today:
            logging.info("[LIVE] entry_time_et already passed; skipping trading for %s", today.isoformat())
            last_watchlist_date = today
            last_entry_date = today
            last_flatten_date = today
            tomorrow = today + dt.timedelta(days=1)
            next_watchlist_dt, _, _ = _schedule_for_day(cfg, tomorrow)
            _sleep_until(next_watchlist_dt)
            continue

        if last_watchlist_date != today:
            if now < watchlist_dt:
                logging.info("[LIVE] waiting for watchlist_time_et=%s", watchlist_time_str)
                _sleep_until(watchlist_dt)
            else:
                logging.info("[LIVE] watchlist_time_et=%s already passed; building now", watchlist_time_str)
            broker = AlpacaBroker(cfg)
            log_account_summary(broker)
            logging.info("[LIVE] building watchlist for today")
            build_watchlist(cfg)
            wl = read_watchlist(expected_watchlist_date_str(), cfg)
            entry_time_map = {}
            for row in wl.get("watchlist") or []:
                sym = str(row.get("symbol") or "").upper()
                if not sym:
                    continue
                entry_time_map[sym] = str(row.get("entry_time_et") or entry_time_strs[0])
            last_watchlist_date = today

        if last_entry_date != today:
            traded_symbols: set[str] = set()
            plans = []
            for entry_time_str, entry_dt in zip(entry_time_strs, entry_dts):
                if et_now() < entry_dt:
                    logging.info("[LIVE] waiting for entry_time_et=%s", entry_time_str)
                    _sleep_until(entry_dt)
                else:
                    logging.info("[LIVE] entry_time_et=%s already passed; placing orders now", entry_time_str)
                logging.info("[LIVE] placing orders for entry_time_et=%s", entry_time_str)
                allowed_symbols = None
                if entry_time_map:
                    if scan_first_valid_mode:
                        allowed_symbols = set()
                        for sym, start_time in entry_time_map.items():
                            st = str(start_time or (entry_time_strs[0] if entry_time_strs else entry_time_str))
                            try:
                                if parse_time_hhmm(entry_time_str) >= parse_time_hhmm(st):
                                    allowed_symbols.add(sym)
                            except Exception:
                                if st == entry_time_str:
                                    allowed_symbols.add(sym)
                    else:
                        allowed_symbols = {s for s, t in entry_time_map.items() if t == entry_time_str}
                    if traded_symbols:
                        allowed_symbols -= traded_symbols
                    if not allowed_symbols:
                        continue
                placed, new_plans = run_live(
                    cfg,
                    return_plans=True,
                    entry_time_et=entry_time_str,
                    symbols_allow=allowed_symbols,
                )
                for resp in placed or []:
                    sym = str(resp.get("symbol") or "").upper()
                    if sym:
                        traded_symbols.add(sym)
                for plan in new_plans or []:
                    sym = str(getattr(plan, "symbol", "") or "").upper()
                    if sym:
                        traded_symbols.add(sym)
                    plans.append(plan)
            last_entry_date = today

        if last_entry_date == today:
            report_until = flatten_dt if intraday_only else _schedule_for_day(cfg, today)[2]
            if close_monitor_date != today:
                seen_close_order_ids.clear()
                close_monitor_date = today
            while et_now() < report_until:
                broker = AlpacaBroker(cfg)
                try:
                    enforce_time_stop(cfg, broker)
                    if close_report_enabled:
                        _log_new_position_closes(cfg, broker, today, seen_close_order_ids)
                    if minute_report_enabled:
                        log_minute_report(cfg, broker, plans)
                except Exception as exc:
                    logging.error("[LIVE] post-entry monitor failed: %s", exc)
                time.sleep(max(5, monitor_interval))

        if intraday_only and last_flatten_date != today:
            if et_now() < flatten_dt:
                logging.info("[LIVE] waiting to flatten at %s ET", flatten_dt.time().strftime("%H:%M"))
                _sleep_until(flatten_dt)
            logging.info("[LIVE] flattening positions (intraday_only)")
            run_flatten(cfg)
            last_flatten_date = today

        # Sleep until next day's watchlist time.
        tomorrow = today + dt.timedelta(days=1)
        next_watchlist_dt, _, _ = _schedule_for_day(cfg, tomorrow)
        logging.info("[LIVE] day complete; sleeping until %s ET", next_watchlist_dt.isoformat())
        _sleep_until(next_watchlist_dt)


if __name__ == "__main__":
    main()
