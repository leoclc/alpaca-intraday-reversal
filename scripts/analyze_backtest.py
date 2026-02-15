from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * p))
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _iso_month(date_str: str) -> str:
    s = str(date_str or "")
    return s[:7] if len(s) >= 7 else s


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


@dataclass(frozen=True)
class WatchlistRow:
    symbol: str
    rank: int
    avgR: Optional[float]
    win_rate: Optional[float]
    profit_factor: Optional[float]
    trades_count: Optional[int]
    total_pnl_pct: Optional[float]


class WatchlistIndex:
    def __init__(self, watchlists_dir: Path) -> None:
        self._dir = watchlists_dir
        self._cache: Dict[str, Dict[str, WatchlistRow]] = {}

    def _load_date(self, date_str: str) -> Dict[str, WatchlistRow]:
        date = str(date_str or "")
        if date in self._cache:
            return self._cache[date]
        path = self._dir / f"{date}.json"
        if not path.exists():
            self._cache[date] = {}
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._cache[date] = {}
            return {}
        wl = payload.get("watchlist") or []
        out: Dict[str, WatchlistRow] = {}
        for idx, row in enumerate(wl, start=1):
            sym = str((row or {}).get("symbol") or "").upper()
            if not sym:
                continue
            out[sym] = WatchlistRow(
                symbol=sym,
                rank=idx,
                avgR=(None if row.get("avgR") is None else _safe_float(row.get("avgR"), default=0.0)),
                win_rate=(None if row.get("win_rate") is None else _safe_float(row.get("win_rate"), default=0.0)),
                profit_factor=(None if row.get("profit_factor") is None else _safe_float(row.get("profit_factor"), default=0.0)),
                trades_count=(None if row.get("trades_count") is None else _safe_int(row.get("trades_count"), default=0)),
                total_pnl_pct=(None if row.get("total_pnl_pct") is None else _safe_float(row.get("total_pnl_pct"), default=0.0)),
            )
        self._cache[date] = out
        return out

    def lookup(self, date_str: str, symbol: str) -> Optional[WatchlistRow]:
        sym = str(symbol or "").upper()
        if not sym:
            return None
        return self._load_date(date_str).get(sym)


def _enrich_trade(t: Dict[str, Any]) -> Dict[str, Any]:
    # Keep original fields; just add derived keys.
    ep = _safe_float(t.get("entry_price"), default=0.0)
    sd = _safe_float(t.get("stop_distance"), default=0.0)
    tp = _safe_float(t.get("target_price"), default=0.0)
    atr = _safe_float(t.get("atr"), default=0.0)
    direction = str(t.get("direction") or "").lower()
    gap_bps = _safe_float(t.get("gap_bps"), default=0.0)
    pb = t.get("early_pullback_bps")
    target_r = abs(tp - ep) / sd if sd > 0 else None
    stop_atr = sd / atr if atr > 0 else None
    gap_fav_bps = gap_bps if direction == "short" else -gap_bps
    out = dict(t)
    out["_month"] = _iso_month(str(t.get("entry_date") or ""))
    out["_target_r"] = target_r
    out["_stop_atr"] = stop_atr
    out["_gap_fav_bps"] = gap_fav_bps
    out["_early_pullback_bps"] = _safe_float(pb, default=0.0) if pb is not None else None
    # Loss classification for stopouts: did the day ever reach the target distance after entry?
    if str(t.get("exit_reason") or "") == "stop" and target_r is not None and t.get("day_mfe_r") is not None:
        try:
            out["_stop_shakeout_day_hit_target"] = bool(_safe_float(t.get("day_mfe_r")) >= float(target_r))
        except Exception:
            out["_stop_shakeout_day_hit_target"] = None
    else:
        out["_stop_shakeout_day_hit_target"] = None
    return out


def _agg_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    if n <= 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avgR": 0.0,
            "pnl_total": 0.0,
            "stops": 0,
            "targets": 0,
            "time_stop": 0,
        }
    wins = sum(1 for t in trades if _safe_float(t.get("r_multiple")) > 0)
    avg_r = sum(_safe_float(t.get("r_multiple")) for t in trades) / float(n)
    pnl_total = sum(_safe_float(t.get("pnl_total")) for t in trades)
    stops = sum(1 for t in trades if str(t.get("exit_reason") or "") == "stop")
    targets = sum(1 for t in trades if str(t.get("exit_reason") or "") == "target")
    time_stop = sum(1 for t in trades if str(t.get("exit_reason") or "") == "time_stop")
    return {
        "trades": n,
        "win_rate": wins / float(n),
        "avgR": avg_r,
        "pnl_total": pnl_total,
        "stops": stops,
        "targets": targets,
        "time_stop": time_stop,
    }


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _format_pct(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}%"
    except Exception:
        return ""


def analyze(run_dir: Path, *, watchlists_dir: Optional[Path]) -> Path:
    trades_path = run_dir / "backtest_trades.ndjson"
    if not trades_path.exists():
        raise FileNotFoundError(f"missing {trades_path}")

    daily_path = run_dir / "backtest_daily.csv"
    daily_rows = _read_csv_rows(daily_path) if daily_path.exists() else []

    trades_raw = _read_ndjson(trades_path)
    trades = [_enrich_trade(t) for t in trades_raw]

    wl_index = WatchlistIndex(watchlists_dir) if watchlists_dir else None
    if wl_index is not None:
        for t in trades:
            date = str(t.get("entry_date") or "")
            sym = str(t.get("symbol") or "")
            row = wl_index.lookup(date, sym)
            if row is None:
                continue
            t["_watchlist_rank"] = row.rank
            t["_watchlist_avgR"] = row.avgR
            t["_watchlist_win_rate"] = row.win_rate
            t["_watchlist_profit_factor"] = row.profit_factor
            t["_watchlist_trades_count"] = row.trades_count
            t["_watchlist_total_pnl_pct"] = row.total_pnl_pct

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Trade table (trade-by-trade).
    trade_fields = [
        "entry_date",
        "symbol",
        "direction",
        "exit_reason",
        "r_multiple",
        "pnl_total",
        "equity_before",
        "equity_after",
        "stop_distance",
        "atr",
        "target_price",
        "entry_price",
        "gap_bps",
        "early_pullback_bps",
        "mfe_r",
        "mae_r",
        "day_mfe_r",
        "day_mae_r",
        "_target_r",
        "_stop_atr",
        "_gap_fav_bps",
        "_stop_shakeout_day_hit_target",
        "_watchlist_rank",
        "_watchlist_avgR",
        "_watchlist_win_rate",
        "_watchlist_profit_factor",
        "_watchlist_trades_count",
        "_watchlist_total_pnl_pct",
    ]
    _write_csv(analysis_dir / "trades_enriched.csv", trade_fields, trades)

    # Month-by-month summary.
    by_month: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        by_month[str(t.get("_month") or "")].append(t)
    month_rows: List[Dict[str, Any]] = []

    # Month equity return from daily curve (preferred), falling back to trade pnl.
    month_equity: Dict[str, Dict[str, Any]] = {}
    if daily_rows:
        by_month_daily: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in daily_rows:
            by_month_daily[_iso_month(row.get("date") or "")].append(row)
        for m, rs in by_month_daily.items():
            if not rs:
                continue
            start_eq = _safe_float(rs[0].get("equity"))
            end_eq = _safe_float(rs[-1].get("equity"))
            month_equity[m] = {
                "start_equity": start_eq,
                "end_equity": end_eq,
                "equity_change": end_eq - start_eq,
                "equity_return_pct": ((end_eq - start_eq) / start_eq * 100.0) if start_eq > 0 else None,
            }

    for m in sorted(by_month.keys()):
        ts = by_month[m]
        a = _agg_trades(ts)
        # stop shakeouts share (stopouts that would have hit target sometime during the day)
        stopouts = [t for t in ts if str(t.get("exit_reason") or "") == "stop"]
        shake = [t for t in stopouts if t.get("_stop_shakeout_day_hit_target") is True]
        shake_share = (len(shake) / float(len(stopouts))) if stopouts else 0.0
        gap_pos = [t for t in ts if _safe_float(t.get("_gap_fav_bps")) >= 0]
        stop_atr_vals = [float(t["_stop_atr"]) for t in ts if t.get("_stop_atr") is not None]
        eq = month_equity.get(m) or {}
        month_rows.append(
            {
                "month": m,
                "trades": a["trades"],
                "win_rate": a["win_rate"],
                "avgR": a["avgR"],
                "pnl_total": a["pnl_total"],
                "equity_return_pct": eq.get("equity_return_pct"),
                "stops": a["stops"],
                "targets": a["targets"],
                "time_stop": a["time_stop"],
                "stop_atr_mean": _mean(stop_atr_vals),
                "gap_fav_pos_share": (len(gap_pos) / float(len(ts))) if ts else 0.0,
                "stop_shakeout_share": shake_share,
            }
        )
    _write_csv(
        analysis_dir / "month_summary.csv",
        [
            "month",
            "trades",
            "win_rate",
            "avgR",
            "pnl_total",
            "equity_return_pct",
            "stops",
            "targets",
            "time_stop",
            "stop_atr_mean",
            "gap_fav_pos_share",
            "stop_shakeout_share",
        ],
        month_rows,
    )

    # Rank bins summary.
    rank_rows: List[Dict[str, Any]] = []
    if any(t.get("_watchlist_rank") is not None for t in trades):
        bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30)]
        for lo, hi in bins:
            ts = [t for t in trades if t.get("_watchlist_rank") is not None and lo <= int(t["_watchlist_rank"]) <= hi]
            a = _agg_trades(ts)
            rank_rows.append(
                {
                    "rank_bin": f"{lo}-{hi}",
                    "trades": a["trades"],
                    "win_rate": a["win_rate"],
                    "avgR": a["avgR"],
                    "pnl_total": a["pnl_total"],
                    "stops": a["stops"],
                    "targets": a["targets"],
                }
            )
        _write_csv(
            analysis_dir / "rank_bins.csv",
            ["rank_bin", "trades", "win_rate", "avgR", "pnl_total", "stops", "targets"],
            rank_rows,
        )

    # Symbol summary.
    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        sym = str(t.get("symbol") or "").upper()
        if sym:
            by_sym[sym].append(t)
    symbol_rows: List[Dict[str, Any]] = []
    for sym, ts in by_sym.items():
        a = _agg_trades(ts)
        symbol_rows.append(
            {
                "symbol": sym,
                "trades": a["trades"],
                "win_rate": a["win_rate"],
                "avgR": a["avgR"],
                "pnl_total": a["pnl_total"],
                "stops": a["stops"],
                "targets": a["targets"],
            }
        )
    symbol_rows.sort(key=lambda r: float(r.get("pnl_total") or 0.0))
    _write_csv(
        analysis_dir / "symbols.csv",
        ["symbol", "trades", "win_rate", "avgR", "pnl_total", "stops", "targets"],
        symbol_rows,
    )

    # Core diagnostics for report.
    overall = _agg_trades(trades)
    stopouts = [t for t in trades if str(t.get("exit_reason") or "") == "stop"]
    stop_mfe_vals = [_safe_float(t.get("mfe_r")) for t in stopouts if t.get("mfe_r") is not None]
    stop_mfe_zero_share = (sum(1 for v in stop_mfe_vals if v <= 1e-9) / float(len(stopouts))) if stopouts else 0.0
    shake_share = (
        (sum(1 for t in stopouts if t.get("_stop_shakeout_day_hit_target") is True) / float(len(stopouts))) if stopouts else 0.0
    )

    stop_atr_vals_all = [float(t["_stop_atr"]) for t in trades if t.get("_stop_atr") is not None]
    gap_fav_vals = [float(t["_gap_fav_bps"]) for t in trades if t.get("_gap_fav_bps") is not None]

    # Worst/best daily returns from daily curve
    worst_days: List[Dict[str, Any]] = []
    best_days: List[Dict[str, Any]] = []
    if daily_rows:
        day_rows = []
        for row in daily_rows:
            date = str(row.get("date") or "")
            dr = _safe_float(row.get("daily_return_pct"))
            if dr == 0.0:
                continue
            day_rows.append((date, dr))
        day_rows.sort(key=lambda x: x[1])
        for date, dr in day_rows[:12]:
            ts = [t for t in trades if str(t.get("entry_date") or "") == date]
            c = Counter(str(t.get("exit_reason") or "") for t in ts)
            worst_days.append(
                {
                    "date": date,
                    "daily_return_pct": dr,
                    "trades": len(ts),
                    "stops": c.get("stop", 0),
                    "targets": c.get("target", 0),
                    "time_stop": c.get("time_stop", 0),
                    "pnl_total": sum(_safe_float(t.get("pnl_total")) for t in ts),
                }
            )
        day_rows.sort(key=lambda x: x[1], reverse=True)
        for date, dr in day_rows[:12]:
            ts = [t for t in trades if str(t.get("entry_date") or "") == date]
            c = Counter(str(t.get("exit_reason") or "") for t in ts)
            best_days.append(
                {
                    "date": date,
                    "daily_return_pct": dr,
                    "trades": len(ts),
                    "stops": c.get("stop", 0),
                    "targets": c.get("target", 0),
                    "time_stop": c.get("time_stop", 0),
                    "pnl_total": sum(_safe_float(t.get("pnl_total")) for t in ts),
                }
            )
        _write_csv(
            analysis_dir / "worst_days.csv",
            ["date", "daily_return_pct", "trades", "stops", "targets", "time_stop", "pnl_total"],
            worst_days,
        )
        _write_csv(
            analysis_dir / "best_days.csv",
            ["date", "daily_return_pct", "trades", "stops", "targets", "time_stop", "pnl_total"],
            best_days,
        )

    # Emit a compact markdown report.
    report_path = analysis_dir / "report.md"
    lines: List[str] = []
    lines.append(f"# Backtest Diagnostics\n")
    lines.append(f"- run_dir: `{run_dir}`\n")
    lines.append(f"- trades_file: `{trades_path}`\n")
    if daily_rows:
        lines.append(f"- daily_curve: `{daily_path}`\n")
    if watchlists_dir:
        lines.append(f"- watchlists_dir: `{watchlists_dir}`\n")

    lines.append("\n## Overall\n")
    lines.append(
        f"- trades={overall['trades']} win_rate={overall['win_rate']:.3f} avgR={overall['avgR']:.3f} "
        f"pnl_total=${overall['pnl_total']:.2f} stops={overall['stops']} targets={overall['targets']} time_stop={overall['time_stop']}\n"
    )
    lines.append(f"- stopouts: {len(stopouts)}  stopout_mfe_r==0 share: {stop_mfe_zero_share:.3f}\n")
    lines.append(f"- stopouts that would have hit target later in the day (shakeouts): share={shake_share:.3f}\n")
    lines.append(
        f"- stop_atr (stop_distance / daily_ATR): p25={_format_float(_pct(stop_atr_vals_all,0.25))} "
        f"p50={_format_float(_pct(stop_atr_vals_all,0.50))} p75={_format_float(_pct(stop_atr_vals_all,0.75))}\n"
    )
    lines.append(
        f"- gap_fav_bps (gap aligned with mean-reversion direction): p25={_format_float(_pct(gap_fav_vals,0.25),1)} "
        f"p50={_format_float(_pct(gap_fav_vals,0.50),1)} p75={_format_float(_pct(gap_fav_vals,0.75),1)}\n"
    )

    lines.append("\n## Month By Month\n")
    lines.append("See `month_summary.csv` for full table.\n")
    # Include only the two worst months by equity return (when available) else by pnl_total.
    month_rows_sorted = list(month_rows)
    month_rows_sorted.sort(
        key=lambda r: (float(r.get("equity_return_pct")) if r.get("equity_return_pct") is not None else 0.0)
    )
    worst_months = month_rows_sorted[:3]
    lines.append("\nWorst months:\n")
    for r in worst_months:
        lines.append(
            f"- {r['month']}: equity_return={_format_pct(r.get('equity_return_pct'))} trades={r['trades']} "
            f"win_rate={float(r['win_rate']):.3f} avgR={float(r['avgR']):.3f} pnl_total=${float(r['pnl_total']):.2f} "
            f"stop_atr_mean={_format_float(r.get('stop_atr_mean'))} stop_shakeout_share={float(r['stop_shakeout_share']):.3f}\n"
        )

    if worst_days:
        lines.append("\n## Worst Days (By Daily Return)\n")
        lines.append("See `worst_days.csv` for full table.\n")
        for r in worst_days[:12]:
            lines.append(
                f"- {r['date']}: daily_return={_format_pct(r['daily_return_pct'])} trades={r['trades']} "
                f"stops={r['stops']} targets={r['targets']} pnl_total=${float(r['pnl_total']):.2f}\n"
            )

    if rank_rows:
        lines.append("\n## Watchlist Rank Bins\n")
        lines.append("See `rank_bins.csv`.\n")
        for r in rank_rows:
            lines.append(
                f"- rank {r['rank_bin']}: trades={r['trades']} win_rate={float(r['win_rate']):.3f} "
                f"avgR={float(r['avgR']):.3f} pnl_total=${float(r['pnl_total']):.2f}\n"
            )

    # Put actionable diagnostics at the end.
    lines.append("\n## Notes For Tuning\n")
    lines.append("- A large fraction of stopouts have `mfe_r==0` but the day later reaches the target distance.\n")
    lines.append("  That pattern is consistent with entering too early (before reversal starts) and/or needing entry confirmation.\n")
    lines.append("- Trades with very small stop_atr tend to have worse expectancy; consider enforcing a minimum stop distance vs ATR.\n")
    lines.append("- Gap alignment matters materially; consider direction-aware gap filters (avoid shorts after gap-down, avoid longs after gap-up).\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


def _resolve_latest_run_dir(repo_root: Path) -> Optional[Path]:
    base = repo_root / "logs" / "backtests"
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    # Prefer timestamp-like folder names; else fall back to last write time.
    def _key(p: Path) -> Tuple[int, float]:
        name = p.name
        is_ts = int(len(name) == 15 and name[:8].isdigit() and name[8] == "_" and name[9:].isdigit())
        return (is_ts, p.stat().st_mtime)

    dirs.sort(key=_key, reverse=True)
    return dirs[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a backtest run folder and emit diagnostics artifacts.")
    ap.add_argument("--run-dir", type=str, default="", help="Backtest run directory (e.g. logs/backtests/20260214_091312)")
    ap.add_argument(
        "--watchlists-dir",
        type=str,
        default="watchlists",
        help="Directory containing daily watchlist JSON files (default: watchlists/)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None:
        resolved = _resolve_latest_run_dir(repo_root)
        if resolved is None:
            raise SystemExit("could not resolve latest run dir (missing logs/backtests)")
        run_dir = resolved
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()

    watchlists_dir = Path(args.watchlists_dir) if args.watchlists_dir else None
    if watchlists_dir is not None and not watchlists_dir.is_absolute():
        watchlists_dir = (repo_root / watchlists_dir).resolve()
    if watchlists_dir is not None and not watchlists_dir.exists():
        watchlists_dir = None

    report_path = analyze(run_dir, watchlists_dir=watchlists_dir)
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

