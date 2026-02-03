from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.replay.daily_strategy_replay import run_replay
from app.utils.time import iter_trading_days
from app.watchlist.daily_strategy_builder import build_watchlist
from app.watchlist.node_assets import fetch_asset_symbols, resolve_watchlist_asset_filters, resolve_watchlist_builder_base


def _default_grid() -> Dict[str, List[float]]:
    return {
        "rr_target": [1.2, 1.5, 1.8],
        "stop_bps": [30, 40, 50],
        "max_hold_minutes": [30, 60, 90],
    }


def _summarize(trades) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avgR": 0.0,
            "profit_factor": 0.0,
            "compounded": 0.0,
            "max_drawdown": 0.0,
        }
    wins = [t for t in trades if t.r_multiple > 0]
    win_rate = len(wins) / float(len(trades))
    avg_r = sum(t.r_multiple for t in trades) / float(len(trades))
    gross_profit = sum(t.r_multiple for t in trades if t.r_multiple > 0)
    gross_loss = abs(sum(t.r_multiple for t in trades if t.r_multiple < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        equity *= 1.0 + (t.pnl_pct / 100.0)
        if equity > peak:
            peak = equity
        drawdown = (equity / peak) - 1.0
        if drawdown < max_dd:
            max_dd = drawdown
    compounded = equity - 1.0
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avgR": avg_r,
        "profit_factor": profit_factor,
        "compounded": compounded,
        "max_drawdown": max_dd,
    }


def _grid_product(grid: Dict[str, Iterable[float]]) -> Iterable[Dict[str, float]]:
    keys = list(grid.keys())
    if not keys:
        return []
    combos = [{}]
    for key in keys:
        values = list(grid[key])
        new_combos = []
        for combo in combos:
            for value in values:
                next_combo = dict(combo)
                next_combo[key] = value
                new_combos.append(next_combo)
        combos = new_combos
    return combos


def _extract_row(row: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    return {key: row.get(key) for key in keys}


def _run_combo_chunk(
    worker_id: int,
    combos: List[Dict[str, float]],
    cfg: Dict,
    start: str,
    end: str,
    symbols: List[str],
    out_path: Path,
    fieldnames: List[str],
) -> List[Dict[str, float]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    params = cfg.get("daily_trend_reversal") or {}
    trend_ma_days = int(params.get("trend_ma_days") or 200)
    atr_period = int(params.get("atr_period") or 14)
    pad_days = max(trend_ma_days, atr_period) * 2 + 10
    start_dt = dt.date.fromisoformat(start)
    prefetch_start = (start_dt - dt.timedelta(days=pad_days)).isoformat()
    data_store = AlpacaOHLCStore(cfg=cfg)
    cfg["watchlists_dir"] = str(out_path.parent / f"watchlists_worker{worker_id}")
    logging.info("[SWEEP] worker=%s prefetching daily bars from %s to %s", worker_id, prefetch_start, end)
    data_store.get_daily_bars_bulk(symbols, prefetch_start, end, cfg=cfg, allow_fetch=True)

    rows: List[Dict[str, float]] = []
    total_combos = len(combos)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        for idx, combo in enumerate(combos, start=1):
            logging.info("[SWEEP] worker=%s combo %s/%s start params=%s", worker_id, idx, total_combos, combo)
            cfg_combo = deepcopy(cfg)
            combo_params = cfg_combo.setdefault("daily_trend_reversal", {})
            for key, value in combo.items():
                if key == "rr_target":
                    combo_params["target_r"] = float(value)
                elif key == "stop_bps":
                    combo_params["stop_mode"] = "pct"
                    combo_params["stop_pct"] = float(value) / 100.0
                elif key == "max_hold_minutes":
                    combo_params["time_stop_minutes"] = int(value)
                else:
                    combo_params[key] = value
            all_trades = []
            replay_cfg = dict(cfg_combo.get("replay") or {})
            replay_cfg["emit_daily_details"] = False
            cfg_combo["replay"] = replay_cfg
            for day in iter_trading_days(start, end):
                date_str = day.isoformat()
                _ = build_watchlist(cfg_combo, target_date=date_str, symbols=symbols, data_store=data_store)
                day_trades = run_replay(cfg_combo, start_date=date_str, end_date=date_str, data_store=data_store)
                if day_trades:
                    all_trades.extend(day_trades)
            stats = _summarize(all_trades)
            row = {**combo, **stats}
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            logging.info(
                "[SWEEP] worker=%s combo %s/%s done trades=%s avgR=%.4f",
                worker_id,
                idx,
                total_combos,
                stats["trades"],
                stats["avgR"],
            )
    return rows


def run_param_sweep(
    cfg: Optional[Dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_path: Optional[str] = None,
    grid: Optional[Dict[str, Iterable[float]]] = None,
    workers: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = cfg or load_config()
    rep = cfg.get("replay") or {}
    start = start_date or rep.get("start_date")
    end = end_date or rep.get("end_date") or start
    if not start or not end:
        raise ValueError("start_date/end_date required (or set replay.start_date/end_date in config.json)")

    base_url = resolve_watchlist_builder_base(cfg)
    asset_filters = resolve_watchlist_asset_filters(cfg) or {}
    logging.info("[SWEEP] fetching asset universe via node base=%s filters=%s", base_url, asset_filters)
    symbols = fetch_asset_symbols(base_url=base_url, **asset_filters)
    logging.info("[SWEEP] asset universe size=%s", len(symbols))

    data_store = AlpacaOHLCStore(cfg=cfg)
    params = cfg.get("daily_trend_reversal") or {}
    trend_ma_days = int(params.get("trend_ma_days") or 200)
    atr_period = int(params.get("atr_period") or 14)
    pad_days = max(trend_ma_days, atr_period) * 2 + 10
    start_dt = dt.date.fromisoformat(start)
    prefetch_start = (start_dt - dt.timedelta(days=pad_days)).isoformat()
    logging.info("[SWEEP] prefetching daily bars from %s to %s", prefetch_start, end)
    data_store.get_daily_bars_bulk(symbols, prefetch_start, end, cfg=cfg, allow_fetch=True)

    grid = grid or _default_grid()
    combos = list(_grid_product(grid))
    if not combos:
        raise ValueError("parameter grid is empty")

    logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
    run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = logs_dir / "backtests" / str(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path) if out_path else out_dir / "param_sweep_2025_daily_trend.csv"
    logging.info("[SWEEP] run_id=%s out_dir=%s", run_id, out_dir)
    # Write config snapshot for reproducibility/debugging
    snapshot_path = out_dir / "config_snapshot.json"
    snapshot_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    # Emit a restore helper so it's easy to revert after experiments.
    restore_script = out_dir / "restore_config.ps1"
    restore_script.write_text(
        "Copy-Item -Force .\\logs\\backtests\\{run_id}\\config_snapshot.json .\\config.json\n".format(run_id=run_id),
        encoding="utf-8",
    )
    logging.info(
        "[SWEEP] config daily_trend_reversal=%s watchlist=%s",
        cfg.get("daily_trend_reversal"),
        cfg.get("watchlist"),
    )

    fieldnames = list(grid.keys()) + ["trades", "win_rate", "avgR", "profit_factor", "max_drawdown", "compounded"]
    rows: List[Dict[str, float]] = []
    workers = int(workers or 1)
    workers = max(1, min(workers, len(combos)))
    if workers == 1:
        rows = _run_combo_chunk(1, combos, cfg, start, end, symbols, out_path, fieldnames)
    else:
        chunks = [combos[i::workers] for i in range(workers)]
        worker_paths = [out_dir / f"param_sweep_2025_daily_trend.worker{i + 1}.csv" for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(
                        _run_combo_chunk,
                        i + 1,
                        chunk,
                        cfg,
                        start,
                        end,
                        symbols,
                        worker_paths[i],
                        fieldnames,
                    )
                )
            for fut in as_completed(futures):
                rows.extend(fut.result())
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    logging.info("[SWEEP] wrote %s", out_path)
    if rows:
        filtered = [row for row in rows if float(row.get("trades") or 0) >= 500]
        filtered.sort(key=lambda r: (float(r.get("avgR") or 0.0), float(r.get("compounded") or 0.0)), reverse=True)
        top_rows = filtered[:10]
        if top_rows:
            logging.info("[SWEEP] top rows (avgR then compounded), trades>=500:")
            for row in top_rows:
                logging.info("[SWEEP] %s", _extract_row(row, fieldnames))
    return out_path
