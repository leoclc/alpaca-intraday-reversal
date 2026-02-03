from __future__ import annotations

import datetime as dt
import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
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
        used_notional += notional
        open_positions += 1
        accepted.append(trade)
        sized_records.append(
            {
                "symbol": plan.symbol,
                "direction": plan.direction,
                "entry_date": plan.entry_date,
                "entry_time_et": plan.entry_time_et,
                "entry_price": plan.entry_price,
                "stop_price": plan.stop_price,
                "target_price": plan.target_price,
                "exit_date": trade.exit_date,
                "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason,
                "qty": qty,
                "pnl_total": pnl_total,
                "pnl_pct": trade.pnl_pct,
                "r_multiple": trade.r_multiple,
                "equity_before": equity_before,
                "equity_after": equity,
                "notional": notional,
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
    params = cfg.get("daily_trend_reversal") or {}
    starting_equity = float(params.get("starting_equity") or 100000.0)
    equity = starting_equity
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for day in iter_trading_days(start, end):
        date_str = day.isoformat()
        logging.info("[BACKTEST] build watchlist date=%s", date_str)
        _ = build_watchlist(cfg, target_date=date_str, symbols=symbols, data_store=data_store, run_id=run_id)
        day_trades = run_replay(cfg, start_date=date_str, end_date=date_str, data_store=data_store, run_id=run_id)
        if day_trades:
            accepted, sized_records, equity = _apply_portfolio_sizing(day_trades, equity, cfg)
            if accepted:
                all_trades.extend(accepted)
            if sized_records:
                sized_trades.extend(sized_records)
        equity_curve.append({"date": date_str, "equity": equity})
        logging.info("[BACKTEST] date=%s trades=%s total=%s equity=%.2f", date_str, len(day_trades), len(all_trades), equity)

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
    out_file: Optional[Path] = None
    if out_path:
        out_file = Path(out_path)
        if out_file.suffix == "":
            run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = out_file / run_id / "backtest.json"
    else:
        run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
        out_file = logs_dir / "backtests" / run_id / "backtest.json"
    if out_file:
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
            },
            "trades": trades_payload,
            "sized_trades": sized_trades,
            "equity_curve": equity_curve,
            "daily_metrics": daily_metrics,
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
        logging.info("[BACKTEST] wrote %s", out_file)
    return summary, all_trades
