from __future__ import annotations

import datetime as dt
import time
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.data.alpaca_intraday_store import filter_intraday_bars_until, get_intraday_bars
from app.execution.daily_execution_model import simulate_exit
from app.strategies.daily_trend_reversal import build_trade, generate_signal_for_date, generate_signals
from app.strategies.types import TradeResult
from app.utils.time import ensure_date, parse_time_hhmm
from app.watchlist.node_assets import fetch_asset_symbols, resolve_watchlist_asset_filters, resolve_watchlist_builder_base
from app.watchlist.storage import expected_watchlist_date_str, write_watchlist


def _lookback_start_date(target_date: dt.date, lookback_days: int) -> dt.date:
    if lookback_days <= 0:
        return target_date
    count = 0
    cur = target_date - dt.timedelta(days=1)
    while count < lookback_days:
        if cur.weekday() < 5:
            count += 1
            if count >= lookback_days:
                return cur
        cur -= dt.timedelta(days=1)
    return cur


def _expand_param_grid(grid: Dict) -> List[Dict]:
    if not isinstance(grid, dict) or not grid:
        return [{}]
    keys = []
    values = []
    for key, vals in grid.items():
        if vals is None:
            continue
        if not isinstance(vals, list):
            vals = [vals]
        if not vals:
            continue
        keys.append(key)
        values.append(vals)
    if not keys:
        return [{}]
    combos: List[Dict] = [{}]
    for key, vals in zip(keys, values):
        next_combos: List[Dict] = []
        for combo in combos:
            for val in vals:
                new_combo = dict(combo)
                new_combo[key] = val
                next_combos.append(new_combo)
        combos = next_combos
    return combos


def _compute_stats(trades: List[TradeResult]) -> Dict[str, float]:
    trades_count = len(trades)
    if trades_count == 0:
        return {
            "trades_count": 0,
            "win_rate": 0.0,
            "avgR": 0.0,
            "profit_factor": 0.0,
            "total_pnl_pct": 0.0,
        }
    wins = [t for t in trades if t.r_multiple > 0]
    win_rate = len(wins) / float(trades_count)
    avg_r = sum(t.r_multiple for t in trades) / float(trades_count)
    gross_profit = sum(t.r_multiple for t in trades if t.r_multiple > 0)
    gross_loss = abs(sum(t.r_multiple for t in trades if t.r_multiple < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    total_pnl_pct = sum(t.pnl_pct for t in trades)
    return {
        "trades_count": trades_count,
        "win_rate": win_rate,
        "avgR": avg_r,
        "profit_factor": profit_factor,
        "total_pnl_pct": total_pnl_pct,
    }


def _add_minutes(time_str: str, minutes: int) -> str:
    if not time_str or minutes <= 0:
        return time_str
    try:
        base_time = parse_time_hhmm(time_str)
        shifted = dt.datetime.combine(dt.date.today(), base_time) + dt.timedelta(minutes=minutes)
        return shifted.strftime("%H:%M")
    except Exception:
        return time_str


def build_watchlist(
    cfg: Dict,
    target_date: Optional[str] = None,
    symbols: Optional[Iterable[str]] = None,
    data_store: Optional[AlpacaOHLCStore] = None,
    run_id: Optional[str] = None,
) -> List[Dict]:
    data_store = data_store or AlpacaOHLCStore(cfg=cfg)
    watch_cfg = cfg.get("watchlist") or {}
    lookback_days = int(watch_cfg.get("lookback_days") or 90)
    min_trades = int(watch_cfg.get("minTrades") or 0)
    reject_negative_pnl = bool(watch_cfg.get("reject_negative_pnl", False))
    min_profit_factor = float(watch_cfg.get("minProfitFactor") or 0.0)
    min_win_rate = float(watch_cfg.get("minWinRate") or 0.0)
    entry_time_rank_by = str(watch_cfg.get("entry_time_rank_by") or "avgR").lower()
    param_rank_by = str(watch_cfg.get("param_rank_by") or entry_time_rank_by).lower()
    try:
        min_avg_r_raw = watch_cfg.get("minAvgR")
        min_avg_r = float(min_avg_r_raw) if min_avg_r_raw is not None else None
    except Exception:
        min_avg_r = None
    top_k = int(watch_cfg.get("top_k") or 0)
    top_k_rank_by = str(watch_cfg.get("top_k_rank_by") or "total_pnl_pct").lower()
    report_enabled = bool(watch_cfg.get("report_enabled", False) or cfg.get("watchlist_report_enabled", False))
    param_grid = _expand_param_grid(watch_cfg.get("param_grid") or {})
    progress_interval_sec = int(watch_cfg.get("progress_interval_sec") or 60)
    tgt = expected_watchlist_date_str(target_date)
    symbols_list = [str(s).upper() for s in (symbols or []) if s]
    watchlist_source = str(cfg.get("watchlist_source") or "node").lower()
    if not symbols_list and watchlist_source == "node":
        asset_filters = resolve_watchlist_asset_filters(cfg) or {}
        base_url = resolve_watchlist_builder_base(cfg)
        symbols_list = fetch_asset_symbols(base_url=base_url, **asset_filters)
    elif not symbols_list:
        symbols_list = [str(s).upper() for s in (cfg.get("symbols") or cfg.get("watchlist_symbols") or []) if s]

    funnel = {
        "scanned_symbols": 0,
        "signals_found": 0,
        "trades_simulated": 0,
        "symbols_passing_filters": 0,
        "watchlist_size": 0,
    }
    reject_counts = {
        "min_trades": 0,
        "neg_pnl": 0,
        "min_pf": 0,
        "min_avg_r": 0,
        "min_win_rate": 0,
    }
    report_rows: List[Dict] = []
    pass_trades_only = 0
    pass_trades_and_pnl = 0
    pass_trades_and_avg_r = 0
    pass_trades_and_pf = 0
    pass_trades_and_win_rate = 0
    pnl_samples: List[float] = []
    pf_samples: List[float] = []
    trades_samples: List[int] = []
    watchlist: List[Dict] = []

    params = cfg.get("daily_trend_reversal") or {}
    entry_time_et = str(params.get("entry_time_et") or "09:35")
    entry_times_raw = params.get("entry_times_et")
    if isinstance(entry_times_raw, list) and entry_times_raw:
        entry_times = [str(t) for t in entry_times_raw if t]
    else:
        entry_times = [entry_time_et]
    try:
        entry_times = sorted(entry_times, key=lambda t: parse_time_hhmm(t))
    except Exception:
        pass
    intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
    early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
    time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
    confirm_minutes = int(params.get("confirm_minutes") or 0)
    confirm_apply_in_watchlist = bool(params.get("confirm_apply_in_watchlist", True))
    apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0 and confirm_apply_in_watchlist
    use_intraday_entry = bool(params.get("use_intraday_entry", False))
    session_open_et = str(params.get("session_open_et") or "09:30")
    max_entry_minutes = 0
    for t in entry_times:
        try:
            entry_time = parse_time_hhmm(t)
            open_time = parse_time_hhmm(session_open_et)
            entry_minutes = int(
                (dt.datetime.combine(dt.date.today(), entry_time) - dt.datetime.combine(dt.date.today(), open_time)).total_seconds()
                / 60
            )
            entry_minutes = max(1, entry_minutes + 1)
            max_entry_minutes = max(max_entry_minutes, entry_minutes)
        except Exception:
            max_entry_minutes = max(max_entry_minutes, 1)
    minutes_needed_base = 0
    if early_range_minutes > 0:
        minutes_needed_base = max(minutes_needed_base, early_range_minutes)
    if use_intraday_entry:
        minutes_needed_base = max(minutes_needed_base, max_entry_minutes)
    if time_stop_minutes > 0:
        minutes_needed_base = max(minutes_needed_base, max_entry_minutes + time_stop_minutes)
    if apply_confirm:
        minutes_needed_base = max(minutes_needed_base, max_entry_minutes + confirm_minutes)

    tgt_date = ensure_date(tgt)
    lookback_start = _lookback_start_date(tgt_date, lookback_days)
    bars_map = data_store.get_daily_bars_bulk(
        symbols_list,
        lookback_start.isoformat(),
        tgt_date.isoformat(),
        cfg=cfg,
        allow_fetch=True,
    )
    insufficient_history: List[str] = []
    start_ts = time.time()
    last_log_ts = start_ts
    total_symbols = len(symbols_list)
    grid_size = max(1, len(param_grid))
    for symbol_idx, symbol in enumerate(symbols_list, start=1):
        bars = bars_map.get(symbol, [])
        if not bars:
            continue
        end_idx = None
        for idx, bar in enumerate(bars):
            if ensure_date(str(bar.get("date"))) < tgt_date:
                end_idx = idx
        if end_idx is None:
            continue
        if end_idx - lookback_days + 1 < 0:
            available_days = end_idx + 1
            insufficient_history.append(f"{symbol}:{available_days}")
            start_idx = 0
        else:
            start_idx = end_idx - lookback_days + 1
        start_date = str(bars[start_idx]["date"])
        end_date = str(bars[end_idx]["date"])

        # Only include symbols that have a signal for the target trading day.
        # This keeps parity with live: the watchlist for day D is the set of symbols we would actually
        # attempt to trade on day D (signal uses daily data through D-1 close only).
        signal_today = generate_signal_for_date(symbol, tgt, cfg, data_store)
        if signal_today is None:
            continue

        funnel["scanned_symbols"] += 1
        signals = generate_signals([symbol], start_date, end_date, cfg, data_store)
        funnel["signals_found"] += len(signals)
        trades_by_candidate: Dict[tuple, List[TradeResult]] = {}
        for entry_time in entry_times:
            for grid_idx in range(len(param_grid)):
                trades_by_candidate[(entry_time, grid_idx)] = []
        for signal in signals:
            bars_intraday = None
            if minutes_needed_base > 0:
                bars_intraday = get_intraday_bars(symbol, signal.signal_date, minutes_needed_base, cfg=cfg, allow_fetch=True)
                if not bars_intraday:
                    continue
            for entry_time in entry_times:
                for grid_idx, overrides in enumerate(param_grid):
                    bars_intraday_entry = bars_intraday
                    if bars_intraday and entry_time:
                        cutoff_time = entry_time
                        if apply_confirm:
                            cutoff_time = _add_minutes(entry_time, confirm_minutes)
                        bars_intraday_entry = filter_intraday_bars_until(
                            bars_intraday,
                            signal.signal_date,
                            cutoff_time,
                        )
                    plan = build_trade(
                        signal,
                        cfg,
                        data_store,
                        context="watchlist",
                        bars_intraday=bars_intraday_entry,
                        entry_time_override=entry_time,
                        param_overrides=overrides,
                    )
                    if not plan:
                        continue
                    exit_info = simulate_exit(plan, "daily", bars, bars_intraday, cfg)
                    if not exit_info:
                        continue
                    direction_mult = 1.0 if plan.direction == "long" else -1.0
                    pnl = (float(exit_info["exit_price"]) - plan.entry_price) * direction_mult
                    pnl_pct = (pnl / plan.entry_price) * 100.0
                    r_multiple = pnl / plan.stop_distance
                    trades_by_candidate[(entry_time, grid_idx)].append(
                        TradeResult(
                            plan=plan,
                            exit_date=str(exit_info["exit_date"]),
                            exit_price=float(exit_info["exit_price"]),
                            exit_reason=str(exit_info["exit_reason"]),
                            pnl_pct=pnl_pct,
                            r_multiple=r_multiple,
                        )
                    )
        total_trades_sim = sum(len(t) for t in trades_by_candidate.values())
        funnel["trades_simulated"] += total_trades_sim
        # Pick best entry time + param combo per symbol.
        best_time = entry_times[0]
        best_params = param_grid[0] if param_grid else {}
        best_stats = _compute_stats(trades_by_candidate.get((best_time, 0), []))
        def _score(stats: Dict[str, float]) -> float:
            if param_rank_by in ("total_pnl_pct", "pnl", "total_pnl"):
                return float(stats.get("total_pnl_pct") or 0.0)
            if param_rank_by in ("profit_factor", "pf"):
                return float(stats.get("profit_factor") or 0.0)
            return float(stats.get("avgR") or 0.0)
        for entry_time in entry_times:
            for grid_idx, overrides in enumerate(param_grid):
                stats = _compute_stats(trades_by_candidate.get((entry_time, grid_idx), []))
                if _score(stats) > _score(best_stats):
                    best_time = entry_time
                    best_params = overrides
                    best_stats = stats
        stats = best_stats
        trades_samples.append(int(stats["trades_count"]))
        pnl_samples.append(float(stats["total_pnl_pct"]))
        pf_samples.append(float(stats["profit_factor"]))
        reasons: List[str] = []
        if stats["trades_count"] < min_trades:
            reject_counts["min_trades"] += 1
            reasons.append("min_trades")
        if reject_negative_pnl and stats["total_pnl_pct"] <= 0:
            reject_counts["neg_pnl"] += 1
            reasons.append("reject_negative_pnl")
        if min_avg_r is not None and stats["avgR"] <= min_avg_r:
            reject_counts["min_avg_r"] += 1
            reasons.append("min_avg_r")
        if min_profit_factor > 0 and stats["profit_factor"] < min_profit_factor:
            reject_counts["min_pf"] += 1
            reasons.append("min_profit_factor")
        if min_win_rate > 0 and stats["win_rate"] < min_win_rate:
            reject_counts["min_win_rate"] += 1
            reasons.append("min_win_rate")
        if stats["trades_count"] >= min_trades:
            pass_trades_only += 1
            if (not reject_negative_pnl) or stats["total_pnl_pct"] > 0:
                pass_trades_and_pnl += 1
            if min_avg_r is None or stats["avgR"] > min_avg_r:
                pass_trades_and_avg_r += 1
            if min_profit_factor <= 0 or stats["profit_factor"] >= min_profit_factor:
                pass_trades_and_pf += 1
            if min_win_rate <= 0 or stats["win_rate"] >= min_win_rate:
                pass_trades_and_win_rate += 1
        if report_enabled:
            report_rows.append(
                {
                    "symbol": symbol,
                    "entry_time_et": best_time,
                    "param_overrides": best_params,
                    **stats,
                    "reasons": reasons,
                }
            )
        if reasons:
            now_ts = time.time()
            if progress_interval_sec > 0 and (now_ts - last_log_ts) >= progress_interval_sec:
                elapsed = int(now_ts - start_ts)
                logging.info(
                    "[WATCHLIST_PROGRESS] date=%s symbols=%s/%s grid=%s signals=%s trades=%s passed=%s elapsed=%ss",
                    tgt,
                    symbol_idx,
                    total_symbols,
                    grid_size,
                    funnel["signals_found"],
                    funnel["trades_simulated"],
                    funnel["symbols_passing_filters"],
                    elapsed,
                )
                last_log_ts = now_ts
            continue
        funnel["symbols_passing_filters"] += 1
        watchlist.append({"symbol": symbol, "entry_time_et": best_time, "param_overrides": best_params, **stats})
        now_ts = time.time()
        if progress_interval_sec > 0 and (now_ts - last_log_ts) >= progress_interval_sec:
            elapsed = int(now_ts - start_ts)
            logging.info(
                "[WATCHLIST_PROGRESS] date=%s symbols=%s/%s grid=%s signals=%s trades=%s passed=%s elapsed=%ss",
                tgt,
                symbol_idx,
                total_symbols,
                grid_size,
                funnel["signals_found"],
                funnel["trades_simulated"],
                funnel["symbols_passing_filters"],
                elapsed,
            )
            last_log_ts = now_ts

    if watchlist and top_k > 0:
        def _rank_key(row: Dict) -> float:
            if top_k_rank_by in ("avgR", "avgr"):
                return float(row.get("avgR") or 0.0)
            if top_k_rank_by in ("profit_factor", "pf"):
                return float(row.get("profit_factor") or 0.0)
            if top_k_rank_by in ("win_rate", "winrate"):
                return float(row.get("win_rate") or 0.0)
            return float(row.get("total_pnl_pct") or 0.0)
        watchlist.sort(key=_rank_key, reverse=True)
        watchlist = watchlist[:top_k]

    funnel["watchlist_size"] = len(watchlist)
    logging.info(
        "[WATCHLIST_FUNNEL] date=%s scanned=%s signals=%s trades=%s passed=%s watchlist=%s",
        tgt,
        funnel["scanned_symbols"],
        funnel["signals_found"],
        funnel["trades_simulated"],
        funnel["symbols_passing_filters"],
        funnel["watchlist_size"],
    )
    if funnel["symbols_passing_filters"] == 0 and funnel["scanned_symbols"] > 0:
        logging.info(
            "[WATCHLIST_FILTERS] date=%s reject_min_trades=%s reject_neg_pnl=%s reject_min_avg_r=%s reject_min_pf=%s",
            tgt,
            reject_counts["min_trades"],
            reject_counts["neg_pnl"],
            reject_counts["min_avg_r"],
            reject_counts["min_pf"],
        )
        if reject_counts["min_win_rate"] > 0:
            logging.info("[WATCHLIST_FILTERS] date=%s reject_min_win_rate=%s", tgt, reject_counts["min_win_rate"])
        if trades_samples:
            trades_samples.sort()
            trades_mean = sum(trades_samples) / float(len(trades_samples))
            logging.info(
                "[WATCHLIST_STATS] trades_count mean=%.2f p50=%s p90=%s max=%s",
                trades_mean,
                trades_samples[len(trades_samples) // 2],
                trades_samples[int(len(trades_samples) * 0.9) - 1],
                trades_samples[-1],
            )
        if pnl_samples:
            pnl_samples.sort()
            pnl_mean = sum(pnl_samples) / float(len(pnl_samples))
            logging.info(
                "[WATCHLIST_STATS] total_pnl_pct mean=%.2f p50=%.2f p90=%.2f max=%.2f",
                pnl_mean,
                pnl_samples[len(pnl_samples) // 2],
                pnl_samples[int(len(pnl_samples) * 0.9) - 1],
                pnl_samples[-1],
            )
        if pf_samples:
            pf_samples.sort()
            pf_mean = sum(pf_samples) / float(len(pf_samples))
            logging.info(
                "[WATCHLIST_STATS] profit_factor mean=%.2f p50=%.2f p90=%.2f max=%.2f",
                pf_mean,
                pf_samples[len(pf_samples) // 2],
                pf_samples[int(len(pf_samples) * 0.9) - 1],
                pf_samples[-1],
            )
    if watchlist:
        write_watchlist(watchlist, cfg, date_str=tgt)
    else:
        logging.warning("[WATCHLIST] empty watchlist date=%s; no fallback applied", tgt)
        # Overwrite any stale watchlist for this date so replay can't pick up old symbols.
        write_watchlist([], cfg, date_str=tgt)
    if watchlist:
        entry_time_counts: Dict[str, int] = {}
        for row in watchlist:
            et = str(row.get("entry_time_et") or "")
            entry_time_counts[et] = entry_time_counts.get(et, 0) + 1
        logging.info("[WATCHLIST_ENTRY_TIMES] date=%s %s", tgt, entry_time_counts)
    if report_enabled:
        logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
        if run_id:
            report_dir = logs_dir / "backtests" / str(run_id) / "watchlist_reports"
        else:
            report_dir = logs_dir / "watchlist_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"verify_report_{tgt}.json"
        report_payload = {
            "date": tgt,
            "lookback_start": lookback_start.isoformat(),
            "lookback_days": lookback_days,
            "filters": {
                "minTrades": min_trades,
                "reject_negative_pnl": reject_negative_pnl,
                "minProfitFactor": min_profit_factor,
                "minAvgR": min_avg_r,
                "minWinRate": min_win_rate,
                "top_k": top_k,
                "entry_time_rank_by": entry_time_rank_by,
                "top_k_rank_by": top_k_rank_by,
                "param_rank_by": param_rank_by,
                "param_grid": watch_cfg.get("param_grid") or {},
            },
            "summary": {
                **funnel,
                **reject_counts,
                "pass_trades_only": pass_trades_only,
                "pass_trades_and_pnl": pass_trades_and_pnl,
                "pass_trades_and_avg_r": pass_trades_and_avg_r,
                "pass_trades_and_pf": pass_trades_and_pf,
                "pass_trades_and_win_rate": pass_trades_and_win_rate,
            },
            "symbols": report_rows,
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    # Muted insufficient history spam; keep list in case we want to surface later.
    return watchlist
