from __future__ import annotations

import datetime as dt
import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.market.filters import market_filter_decision
from app.portfolio.sizing import compute_qty_with_guards
from app.replay.daily_strategy_replay import run_replay
from app.utils.time import iter_trading_days
from app.watchlist.daily_strategy_builder import build_watchlist
from app.watchlist.node_assets import fetch_asset_symbols, resolve_watchlist_asset_filters, resolve_watchlist_builder_base


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
    used_notional = 0.0
    open_positions = 0
    accepted: List = []
    sized_records: List[Dict] = []
    for trade in day_trades:
        plan = getattr(trade, "plan", None)
        if plan is None:
            continue
        qty, state = compute_qty_with_guards(
            plan,
            equity,
            used_notional,
            cfg,
            open_positions=open_positions,
        )
        if qty <= 0:
            continue
        direction_mult = 1.0 if plan.direction == "long" else -1.0
        pnl_per_share = (float(trade.exit_price) - plan.entry_price) * direction_mult
        pnl_total = pnl_per_share * qty
        equity_before = equity
        equity = equity + pnl_total
        notional = plan.entry_price * qty
        capacity_qty = int((state or {}).get("capacity_qty") or qty)
        capacity_notional = plan.entry_price * capacity_qty
        used_notional += capacity_notional
        open_positions += 1
        wl_stats = getattr(plan, "watchlist_stats", None) if isinstance(getattr(plan, "watchlist_stats", None), dict) else {}
        quality_state = state.get("quality_sizing") if isinstance(state, dict) else {}
        accepted.append(trade)
        sized_records.append(
            {
                "symbol": plan.symbol,
                "param_overrides": getattr(plan, "param_overrides", None),
                "direction": plan.direction,
                "entry_date": plan.entry_date,
                "entry_time_et": plan.entry_time_et,
                "entry_price": plan.entry_price,
                "entry_price_mode": getattr(plan, "entry_price_mode", None),
                "stop_price": plan.stop_price,
                "target_price": plan.target_price,
                "stop_distance": plan.stop_distance,
                "target_mode": getattr(plan, "target_mode", None),
                "target_window_avg_pct": getattr(plan, "target_window_avg_pct", None),
                "target_window_mult": getattr(plan, "target_window_mult", None),
                "target_window_minutes": getattr(plan, "target_window_minutes", None),
                "target_window_samples": getattr(plan, "target_window_samples", None),
                "gap_bps": getattr(plan, "gap_bps", None),
                "early_pullback_bps": getattr(plan, "early_pullback_bps", None),
                "early_reversal_bps": getattr(plan, "early_reversal_bps", None),
                "confirm_move_bps": getattr(plan, "confirm_move_bps", None),
                "confirm_minutes": getattr(plan, "confirm_minutes", None),
                "confirm_hit_bps": getattr(plan, "confirm_hit_bps", None),
                "signal_return_pct": getattr(plan, "signal_return_pct", None),
                "signal_return_atr": getattr(plan, "signal_return_atr", None),
                "atr": getattr(plan, "atr", None),
                "watchlist_rank": wl_stats.get("rank"),
                "watchlist_avgR": wl_stats.get("avgR"),
                "watchlist_avgR_stderr": wl_stats.get("avgR_stderr"),
                "watchlist_win_rate": wl_stats.get("win_rate"),
                "watchlist_profit_factor": wl_stats.get("profit_factor"),
                "watchlist_trades_count": wl_stats.get("trades_count"),
                "watchlist_total_pnl_pct": wl_stats.get("total_pnl_pct"),
                "exit_date": trade.exit_date,
                "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason,
                "exit_ts": getattr(trade, "exit_ts", None),
                "stop_hit_ts": getattr(trade, "stop_hit_ts", None),
                "target_hit_ts": getattr(trade, "target_hit_ts", None),
                "qty": qty,
                "pnl_total": pnl_total,
                "pnl_pct": trade.pnl_pct,
                "r_multiple": trade.r_multiple,
                "quality_risk_mult": (quality_state or {}).get("risk_multiplier"),
                "quality_score": (quality_state or {}).get("score"),
                "mfe_pct": getattr(trade, "mfe_pct", None),
                "mae_pct": getattr(trade, "mae_pct", None),
                "mfe_r": getattr(trade, "mfe_r", None),
                "mae_r": getattr(trade, "mae_r", None),
                "mfe_r_full": getattr(trade, "mfe_r_full", None),
                "mae_r_full": getattr(trade, "mae_r_full", None),
                "mfe_r_before_stop": getattr(trade, "mfe_r_before_stop", None),
                "mae_r_to_target": getattr(trade, "mae_r_to_target", None),
                "equity_before": equity_before,
                "equity_after": equity,
                "notional": notional,
                "capacity_notional": capacity_notional,
                "sizing_state": state,
            }
        )
    return accepted, sized_records, equity


def run_backtest(
    cfg: Optional[Dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[Dict[str, float], List]:
    cfg = cfg or load_config()
    rep = cfg.get("replay") or {}
    start = start_date or rep.get("start_date")
    end = end_date or rep.get("end_date") or start
    if not start or not end:
        raise ValueError("start_date/end_date required (or set replay.start_date/end_date in config.json)")

    base_url = resolve_watchlist_builder_base(cfg)
    asset_filters = resolve_watchlist_asset_filters(cfg) or {}
    logging.info("[BACKTEST] fetching asset universe via node base=%s filters=%s", base_url, asset_filters)
    symbols = fetch_asset_symbols(base_url=base_url, **asset_filters)
    logging.info("[BACKTEST] asset universe size=%s", len(symbols))

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
        logging.info("[BACKTEST] build watchlist date=%s", date_str)
        _ = build_watchlist(cfg, target_date=date_str, symbols=symbols, data_store=data_store, run_id=run_id)
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
                    "exit_reason": t.exit_reason,
                    "pnl_pct": t.pnl_pct,
                    "r_multiple": t.r_multiple,
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
