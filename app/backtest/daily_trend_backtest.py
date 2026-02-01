from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
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
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for day in iter_trading_days(start, end):
        date_str = day.isoformat()
        logging.info("[BACKTEST] build watchlist date=%s", date_str)
        _ = build_watchlist(cfg, target_date=date_str, symbols=symbols, data_store=data_store, run_id=run_id)
        day_trades = run_replay(cfg, start_date=date_str, end_date=date_str, data_store=data_store, run_id=run_id)
        if day_trades:
            all_trades.extend(day_trades)
        logging.info("[BACKTEST] date=%s trades=%s total=%s", date_str, len(day_trades), len(all_trades))

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
        payload = {"summary": summary, "trades": trades_payload}
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info("[BACKTEST] wrote %s", out_file)
    return summary, all_trades
