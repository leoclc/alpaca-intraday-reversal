from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import hashlib
import os
import time
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.data.alpaca_intraday_store import filter_intraday_bars_until, get_intraday_bars
from app.execution.daily_execution_model import simulate_exit
from app.strategies.daily_trend_reversal import build_trade, generate_signal_for_date, generate_signals_cached
from app.utils.time import ensure_date, parse_time_hhmm
from app.watchlist.day_filter import summarize_watchlist_rows
from app.watchlist.node_assets import resolve_asset_universe_symbols
from app.watchlist.storage import expected_watchlist_date_str, write_watchlist


@dataclass(frozen=True)
class TradeLite:
    pnl_pct: float
    r_multiple: float


@dataclass
class CandidateAcc:
    trades_count: int = 0
    wins: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0
    sum_pnl_pct: float = 0.0
    gross_profit_r: float = 0.0
    gross_loss_r: float = 0.0  # abs(sum(neg r))
    month_pnl_pct: Dict[str, float] = field(default_factory=dict)
    month_trades: Dict[str, int] = field(default_factory=dict)


@dataclass
class SymbolRollingState:
    # All signals for this symbol keyed by signal_date (YYYY-MM-DD).
    signals_by_date: Dict[str, object]
    # The set of signal_dates currently included in the rolling lookback window.
    window_signal_dates: set[str]
    # Rolling aggregates for each candidate (entry_time, grid_idx) over window_signal_dates.
    acc_by_candidate: Dict[Tuple[str, int], CandidateAcc]
    window_start_date: str = ""
    window_end_date: str = ""


# Per-symbol rolling cache so backtest watchlists don't rescan 252 days of signals on every trading day.
# Keyed by (symbol, direction_key) where direction_key is either "all" (non-directional history)
# or the specific direction ("long"/"short") when directional_history_only=True.
_ROLLING_STATE: Dict[Tuple[str, str], SymbolRollingState] = {}

# Cache signals_by_date per symbol so direction-specific rolling states can share it without recomputing.
_SIGNALS_BY_SYMBOL: Dict[str, Dict[str, object]] = {}


# Cache of per-signal trade outcomes used for watchlist scoring. This is safe (no lookahead) because keys include
# the signal date; the caller still decides which dates are eligible for a given target day.
_TRADE_LITE_CACHE: Dict[Tuple[str, str, str, str, str, str], Optional[TradeLite]] = {}
_TRADE_LITE_DISK_LOADED: set[Tuple[str, str, str, str]] = set()

# Bump this whenever the trade-lite computation logic changes so we don't reuse stale caches.
_TRADE_LITE_CACHE_VERSION = 5


def _trade_lite_namespace(cfg: Dict) -> str:
    """Namespace trade-lite caches by strategy params so we don't mix outcomes across configs."""
    params = cfg.get("daily_trend_reversal") or {}
    # Sizing-only controls do not change simulated entry/exit outcomes for watchlist scoring.
    # Excluding them keeps trade-lite caches reusable while tuning portfolio risk overlays.
    if isinstance(params, dict):
        params = {k: v for k, v in params.items() if not str(k).startswith("quality_sizing_")}
    try:
        raw = json.dumps({"v": _TRADE_LITE_CACHE_VERSION, "params": params}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        raw = f"v={_TRADE_LITE_CACHE_VERSION}|{str(params)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _trade_lite_cache_dir(cfg: Dict) -> Path:
    base = cfg.get("trade_lite_cache_dir") or "cache/trade_lites"
    path = Path(str(base))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _trade_lite_disk_path(cfg: Dict, *, namespace: str, symbol: str, signal_date: str, direction: str) -> Path:
    sym = str(symbol).upper().strip()
    direc = str(direction).lower().strip() or "x"
    ns = str(namespace)
    # One file per (namespace, symbol, signal_date, direction) storing all candidates for that signal.
    return _trade_lite_cache_dir(cfg) / ns / sym / f"{str(signal_date)}_{direc}.json"


def _load_trade_lite_disk_cache(path: Path) -> Dict[str, Dict[str, Optional[TradeLite]]]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        return {}
    out: Dict[str, Dict[str, Optional[TradeLite]]] = {}
    for entry_time, mapping in candidates.items():
        if not isinstance(mapping, dict):
            continue
        et = str(entry_time or "")
        if not et:
            continue
        inner: Dict[str, Optional[TradeLite]] = {}
        for overrides_key, val in mapping.items():
            ok = str(overrides_key)
            if val is None:
                inner[ok] = None
                continue
            if isinstance(val, list) and len(val) == 2:
                try:
                    inner[ok] = TradeLite(pnl_pct=float(val[0]), r_multiple=float(val[1]))
                except Exception:
                    inner[ok] = None
                continue
            if isinstance(val, dict):
                try:
                    inner[ok] = TradeLite(pnl_pct=float(val.get("pnl_pct") or 0.0), r_multiple=float(val.get("r_multiple") or 0.0))
                except Exception:
                    inner[ok] = None
                continue
        if inner:
            out[et] = inner
    return out


def _save_trade_lite_disk_cache(
    path: Path,
    *,
    namespace: str,
    symbol: str,
    signal_date: str,
    direction: str,
    updates: Dict[str, Dict[str, Optional[TradeLite]]],
) -> None:
    if not updates:
        return
    try:
        existing: Dict[str, object] = {}
        if path.exists() and path.stat().st_size > 0:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        if not isinstance(existing, dict):
            existing = {}

        candidates: Dict[str, Dict[str, object]] = {}
        if isinstance(existing.get("candidates"), dict):
            for k, v in existing.get("candidates", {}).items():  # type: ignore[assignment]
                if isinstance(v, dict):
                    candidates[str(k)] = {str(ok): v[ok] for ok in v}

        for entry_time, mapping in updates.items():
            et = str(entry_time or "")
            if not et or not isinstance(mapping, dict):
                continue
            row = candidates.get(et) or {}
            for overrides_key, lite in mapping.items():
                ok = str(overrides_key)
                if lite is None:
                    row[ok] = None
                else:
                    row[ok] = [float(lite.pnl_pct), float(lite.r_multiple)]
            candidates[et] = row

        existing["meta"] = {
            "namespace": str(namespace),
            "symbol": str(symbol).upper(),
            "signal_date": str(signal_date),
            "direction": str(direction).lower(),
        }
        existing["candidates"] = candidates
        payload = json.dumps(existing, separators=(",", ":"), sort_keys=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        tmp.write_text(payload, encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            if "tmp" in locals() and tmp.exists():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _overrides_key(overrides: Dict) -> str:
    try:
        return json.dumps(overrides or {}, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(overrides or {})


def _trade_cache_key(
    namespace: str, symbol: str, signal_date: str, direction: str, entry_time: str, overrides: Dict
) -> Tuple[str, str, str, str, str, str]:
    try:
        ok = _overrides_key(overrides)
    except Exception:
        ok = str(overrides or {})
    return (
        str(namespace),
        str(symbol).upper(),
        str(signal_date),
        str(direction).lower(),
        str(entry_time),
        ok,
    )


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


def _normalize_symbol_param_overrides(raw: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        symbol = str(key or "").upper().strip()
        if not symbol or not isinstance(value, dict) or not value:
            continue
        out[symbol] = dict(value)
    return out


def _load_symbol_param_overrides(watch_cfg: Dict) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    path_raw = watch_cfg.get("symbol_param_overrides_path")
    if path_raw:
        try:
            payload = json.loads(Path(str(path_raw)).read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("symbol_param_overrides"), dict):
                out.update(_normalize_symbol_param_overrides(payload.get("symbol_param_overrides")))
            else:
                out.update(_normalize_symbol_param_overrides(payload))
        except Exception:
            logging.warning("[WATCHLIST] failed to load symbol_param_overrides_path=%s", str(path_raw))
    out.update(_normalize_symbol_param_overrides(watch_cfg.get("symbol_param_overrides")))
    return out


def _compute_stats(trades: List[TradeLite]) -> Dict[str, float]:
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


def _month_key(date_str: str) -> str:
    s = str(date_str or "")
    return s[:7] if len(s) >= 7 else ""


def _acc_add(acc: CandidateAcc, lite: TradeLite, signal_date: str) -> None:
    acc.trades_count += 1
    acc.sum_r += float(lite.r_multiple)
    r = float(lite.r_multiple)
    acc.sum_r2 += r * r
    acc.sum_pnl_pct += float(lite.pnl_pct)
    if r > 0:
        acc.wins += 1
        acc.gross_profit_r += r
    elif r < 0:
        acc.gross_loss_r += abs(r)
    month = _month_key(signal_date)
    if month:
        acc.month_pnl_pct[month] = float(acc.month_pnl_pct.get(month) or 0.0) + float(lite.pnl_pct)
        acc.month_trades[month] = int(acc.month_trades.get(month) or 0) + 1


def _acc_sub(acc: CandidateAcc, lite: TradeLite, signal_date: str) -> None:
    acc.trades_count -= 1
    acc.sum_r -= float(lite.r_multiple)
    r = float(lite.r_multiple)
    acc.sum_r2 -= r * r
    acc.sum_pnl_pct -= float(lite.pnl_pct)
    if r > 0:
        acc.wins -= 1
        acc.gross_profit_r -= r
    elif r < 0:
        acc.gross_loss_r -= abs(r)
    month = _month_key(signal_date)
    if month:
        next_pnl = float(acc.month_pnl_pct.get(month) or 0.0) - float(lite.pnl_pct)
        next_n = int(acc.month_trades.get(month) or 0) - 1
        if next_n <= 0:
            acc.month_trades.pop(month, None)
            acc.month_pnl_pct.pop(month, None)
        else:
            acc.month_trades[month] = next_n
            acc.month_pnl_pct[month] = next_pnl


def _longest_negative_month_streak(values: List[float]) -> int:
    best = 0
    run = 0
    for v in values:
        if float(v) < 0:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _stats_from_acc(acc: CandidateAcc) -> Dict[str, float]:
    import math

    if acc.trades_count <= 0:
        return {
            "trades_count": 0,
            "win_rate": 0.0,
            "avgR": 0.0,
            "stdR": 0.0,
            "avgR_stderr": 0.0,
            "profit_factor": 0.0,
            "total_pnl_pct": 0.0,
            "months_count": 0,
            "positive_months": 0,
            "negative_months": 0,
            "positive_month_rate": 0.0,
            "worst_month_pnl_pct": 0.0,
            "max_monthly_drawdown_pct": 0.0,
            "longest_negative_month_streak": 0,
        }
    win_rate = float(acc.wins) / float(acc.trades_count) if acc.trades_count > 0 else 0.0
    n = float(acc.trades_count)
    avg_r = float(acc.sum_r) / n if n > 0 else 0.0
    # Population std is fine for ranking; we primarily want a monotonic penalty for small-N and volatile series.
    var_r = max((float(acc.sum_r2) / n) - (avg_r * avg_r), 0.0) if n > 0 else 0.0
    std_r = math.sqrt(var_r)
    avg_r_stderr = (std_r / math.sqrt(n)) if n > 1 else std_r
    if acc.gross_loss_r > 0:
        profit_factor = float(acc.gross_profit_r) / float(acc.gross_loss_r)
    else:
        profit_factor = float(acc.gross_profit_r) if acc.gross_profit_r > 0 else 0.0
    month_keys = sorted([m for m, n in acc.month_trades.items() if int(n or 0) > 0 and str(m)])
    month_pnls = [float(acc.month_pnl_pct.get(m) or 0.0) for m in month_keys]
    months_count = len(month_pnls)
    positive_months = sum(1 for v in month_pnls if v > 0)
    negative_months = sum(1 for v in month_pnls if v < 0)
    positive_month_rate = (positive_months / float(months_count)) if months_count > 0 else 0.0
    worst_month_pnl_pct = min(month_pnls) if month_pnls else 0.0
    cum_pnl = 0.0
    peak_pnl = 0.0
    max_monthly_drawdown_pct = 0.0
    for pnl in month_pnls:
        cum_pnl += float(pnl)
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        drawdown = peak_pnl - cum_pnl
        if drawdown > max_monthly_drawdown_pct:
            max_monthly_drawdown_pct = drawdown

    return {
        "trades_count": int(acc.trades_count),
        "win_rate": win_rate,
        "avgR": avg_r,
        "stdR": std_r,
        "avgR_stderr": avg_r_stderr,
        "profit_factor": profit_factor,
        "total_pnl_pct": float(acc.sum_pnl_pct),
        "months_count": months_count,
        "positive_months": positive_months,
        "negative_months": negative_months,
        "positive_month_rate": positive_month_rate,
        "worst_month_pnl_pct": worst_month_pnl_pct,
        "max_monthly_drawdown_pct": max_monthly_drawdown_pct,
        "longest_negative_month_streak": _longest_negative_month_streak(month_pnls),
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


def _trade_lites_for_signal(
    symbol: str,
    signal: object,
    *,
    trade_ns: str,
    cfg: Dict,
    data_store: AlpacaOHLCStore,
    bars_daily: List[Dict],
    entry_times: List[str],
    param_grid: List[Dict],
    minutes_needed_base: int,
    apply_confirm: bool,
    confirm_minutes: int,
) -> Dict[Tuple[str, int], Optional[TradeLite]]:
    # Return per-candidate trade outcome for a given signal_date, computing missing candidates and
    # caching results in _TRADE_LITE_CACHE. Keys are (entry_time, grid_idx).
    sym = str(symbol).upper()
    signal_date = str(getattr(signal, "signal_date", "") or "")
    direction = str(getattr(signal, "direction", "") or "").lower()

    disk_key = (str(trade_ns), sym, signal_date, direction)
    if disk_key not in _TRADE_LITE_DISK_LOADED and signal_date and direction:
        path = _trade_lite_disk_path(cfg, namespace=trade_ns, symbol=sym, signal_date=signal_date, direction=direction)
        loaded = _load_trade_lite_disk_cache(path)
        if loaded:
            for entry_time, mapping in loaded.items():
                for ok, lite in mapping.items():
                    cache_key = (str(trade_ns), sym, signal_date, direction, str(entry_time), str(ok))
                    _TRADE_LITE_CACHE.setdefault(cache_key, lite)
        _TRADE_LITE_DISK_LOADED.add(disk_key)

    cached_by_candidate: Dict[Tuple[str, int], Optional[TradeLite]] = {}
    any_missing = False
    for entry_time in entry_times:
        for grid_idx, overrides in enumerate(param_grid):
            cache_key = _trade_cache_key(trade_ns, sym, signal_date, direction, entry_time, overrides)
            if cache_key in _TRADE_LITE_CACHE:
                cached_by_candidate[(entry_time, grid_idx)] = _TRADE_LITE_CACHE[cache_key]
            else:
                any_missing = True
    if not any_missing:
        return cached_by_candidate

    bars_intraday = None
    if minutes_needed_base > 0:
        bars_intraday = get_intraday_bars(sym, signal_date, minutes_needed_base, cfg=cfg, allow_fetch=True)
        if not bars_intraday:
            # Preserve existing semantics: treat as "no trade" for this signal (do not cache),
            # allowing future attempts if data becomes available.
            return cached_by_candidate

    intraday_entry_cache: Dict[str, List[dict]] = {}
    for entry_time in entry_times:
        cutoff_time = entry_time
        if apply_confirm:
            cutoff_time = _add_minutes(entry_time, confirm_minutes)
        if bars_intraday and entry_time and entry_time not in intraday_entry_cache:
            intraday_entry_cache[entry_time] = filter_intraday_bars_until(
                bars_intraday,
                signal_date,
                cutoff_time,
            )
        for grid_idx, overrides in enumerate(param_grid):
            cache_key = _trade_cache_key(trade_ns, sym, signal_date, direction, entry_time, overrides)
            if cache_key in _TRADE_LITE_CACHE:
                cached_by_candidate[(entry_time, grid_idx)] = _TRADE_LITE_CACHE[cache_key]
                continue
            bars_intraday_entry = bars_intraday
            if bars_intraday and entry_time:
                bars_intraday_entry = intraday_entry_cache.get(entry_time) or []
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
                _TRADE_LITE_CACHE[cache_key] = None
                cached_by_candidate[(entry_time, grid_idx)] = None
                continue
            exit_info = simulate_exit(plan, "daily", bars_daily, bars_intraday, cfg)
            if not exit_info:
                _TRADE_LITE_CACHE[cache_key] = None
                cached_by_candidate[(entry_time, grid_idx)] = None
                continue
            direction_mult = 1.0 if plan.direction == "long" else -1.0
            pnl = (float(exit_info["exit_price"]) - plan.entry_price) * direction_mult
            pnl_pct = (pnl / plan.entry_price) * 100.0
            r_multiple = pnl / plan.stop_distance
            lite = TradeLite(pnl_pct=pnl_pct, r_multiple=r_multiple)
            _TRADE_LITE_CACHE[cache_key] = lite
            cached_by_candidate[(entry_time, grid_idx)] = lite

    # Persist newly computed candidates so subsequent runs can resume quickly without recomputing.
    if bars_intraday and signal_date and direction:
        updates: Dict[str, Dict[str, Optional[TradeLite]]] = {}
        for entry_time in entry_times:
            inner: Dict[str, Optional[TradeLite]] = {}
            for overrides in param_grid:
                ok = _overrides_key(overrides)
                ck = (str(trade_ns), sym, signal_date, direction, str(entry_time), ok)
                if ck in _TRADE_LITE_CACHE:
                    inner[ok] = _TRADE_LITE_CACHE[ck]
            if inner:
                updates[str(entry_time)] = inner
        if updates:
            path = _trade_lite_disk_path(cfg, namespace=trade_ns, symbol=sym, signal_date=signal_date, direction=direction)
            _save_trade_lite_disk_cache(
                path,
                namespace=trade_ns,
                symbol=sym,
                signal_date=signal_date,
                direction=direction,
                updates=updates,
            )
    return cached_by_candidate


def build_watchlist(
    cfg: Dict,
    target_date: Optional[str] = None,
    symbols: Optional[Iterable[str]] = None,
    data_store: Optional[AlpacaOHLCStore] = None,
    run_id: Optional[str] = None,
) -> List[Dict]:
    data_store = data_store or AlpacaOHLCStore(cfg=cfg)
    watch_cfg = cfg.get("watchlist") or {}
    trade_ns = _trade_lite_namespace(cfg)
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
    try:
        min_months_count_raw = watch_cfg.get("min_months_count")
        min_months_count = int(min_months_count_raw) if min_months_count_raw is not None else 0
    except Exception:
        min_months_count = 0
    try:
        min_positive_month_rate_raw = watch_cfg.get("min_positive_month_rate")
        min_positive_month_rate = (
            float(min_positive_month_rate_raw) if min_positive_month_rate_raw is not None else None
        )
    except Exception:
        min_positive_month_rate = None
    try:
        max_negative_months_raw = watch_cfg.get("max_negative_months")
        max_negative_months = int(max_negative_months_raw) if max_negative_months_raw is not None else None
    except Exception:
        max_negative_months = None
    try:
        max_monthly_drawdown_pct_raw = watch_cfg.get("max_monthly_drawdown_pct")
        max_monthly_drawdown_pct = (
            float(max_monthly_drawdown_pct_raw) if max_monthly_drawdown_pct_raw is not None else None
        )
    except Exception:
        max_monthly_drawdown_pct = None
    try:
        max_longest_negative_month_streak_raw = watch_cfg.get("max_longest_negative_month_streak")
        max_longest_negative_month_streak = (
            int(max_longest_negative_month_streak_raw)
            if max_longest_negative_month_streak_raw is not None
            else None
        )
    except Exception:
        max_longest_negative_month_streak = None
    try:
        min_worst_month_pnl_pct_raw = watch_cfg.get("min_worst_month_pnl_pct")
        min_worst_month_pnl_pct = (
            float(min_worst_month_pnl_pct_raw) if min_worst_month_pnl_pct_raw is not None else None
        )
    except Exception:
        min_worst_month_pnl_pct = None
    top_k = int(watch_cfg.get("top_k") or 0)
    top_k_rank_by = str(watch_cfg.get("top_k_rank_by") or "total_pnl_pct").lower()
    directional_history_only = bool(watch_cfg.get("directional_history_only", False))
    report_enabled = bool(watch_cfg.get("report_enabled", False) or cfg.get("watchlist_report_enabled", False))
    try:
        rank_lcb_z = float(watch_cfg.get("rank_lcb_z") or 1.0)
    except Exception:
        rank_lcb_z = 1.0
    param_grid = _expand_param_grid(watch_cfg.get("param_grid") or {})
    # Optional explicit per-symbol parameter overrides (e.g., target/stop knobs for specific symbols).
    # These are applied in the watchlist builder so ranking/filtering uses the same params replay will execute.
    symbol_param_overrides = _load_symbol_param_overrides(watch_cfg)
    symbol_param_override_mode = str(watch_cfg.get("symbol_param_override_mode") or "merge").lower().strip()
    if symbol_param_override_mode not in {"merge", "replace"}:
        symbol_param_override_mode = "merge"
    grid_idx_by_overrides_key: Dict[str, int] = {}
    for idx, overrides in enumerate(param_grid):
        ok = _overrides_key(overrides)
        if ok not in grid_idx_by_overrides_key:
            grid_idx_by_overrides_key[ok] = idx
    progress_interval_sec = int(watch_cfg.get("progress_interval_sec") or 60)
    tgt = expected_watchlist_date_str(target_date)
    symbols_list = [str(s).upper() for s in (symbols or []) if s]
    watchlist_source = str(cfg.get("watchlist_source") or "node").lower()
    if not symbols_list and watchlist_source == "node":
        symbols_list, universe_source = resolve_asset_universe_symbols(cfg, target_date=tgt, allow_fetch=True)
        logging.info(
            "[WATCHLIST] universe date=%s source=%s size=%s",
            tgt,
            universe_source,
            len(symbols_list),
        )
    elif not symbols_list:
        symbols_list = [str(s).upper() for s in (cfg.get("symbols") or cfg.get("watchlist_symbols") or []) if s]
    # Deterministic symbol scan order keeps tie-break behavior stable across runs.
    symbols_list = sorted({str(s).upper() for s in symbols_list if s})

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
        "min_months_count": 0,
        "min_positive_month_rate": 0,
        "max_negative_months": 0,
        "max_monthly_drawdown_pct": 0,
        "max_longest_negative_month_streak": 0,
        "min_worst_month_pnl_pct": 0,
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
    intraday_only = bool(params.get("intraday_only", False))
    confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
    confirm_minutes = int(params.get("confirm_minutes") or 0)
    confirm_apply_in_watchlist = bool(params.get("confirm_apply_in_watchlist", True))
    apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0 and confirm_apply_in_watchlist
    use_intraday_entry = bool(params.get("use_intraday_entry", False))
    session_open_et = str(params.get("session_open_et") or "09:30")
    max_entry_minutes = 0
    minutes_needed_base = 0
    if early_range_minutes > 0:
        minutes_needed_base = max(minutes_needed_base, early_range_minutes)
    try:
        open_time = parse_time_hhmm(session_open_et)
        open_dt = dt.datetime.combine(dt.date.today(), open_time)
        flatten_minutes_from_open = None
        if intraday_only:
            try:
                session_close_et = str(params.get("session_close_et") or "16:00")
                flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)
                close_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_close_et))
                flatten_dt = close_dt - dt.timedelta(minutes=max(0, flatten_buffer))
                flatten_minutes_from_open = int((flatten_dt - open_dt).total_seconds() / 60)
                flatten_minutes_from_open = max(1, flatten_minutes_from_open)
            except Exception:
                flatten_minutes_from_open = None

        confirm_pad = confirm_minutes if apply_confirm else 0
        for t in entry_times:
            try:
                entry_time = parse_time_hhmm(t)
            except Exception:
                continue
            entry_minutes_raw = int((dt.datetime.combine(dt.date.today(), entry_time) - open_dt).total_seconds() / 60)
            entry_minutes_raw = max(0, entry_minutes_raw)

            # Ensure we can reference the last completed bar before entry_dt.
            max_entry_minutes = max(max_entry_minutes, max(1, entry_minutes_raw + 1))

            # Confirmation evaluates [entry_dt, cutoff_dt). Ensure we have bars up to cutoff.
            if apply_confirm:
                minutes_needed_base = max(minutes_needed_base, max(1, entry_minutes_raw + confirm_pad + 1))

            cutoff_minutes = None
            if time_stop_minutes > 0:
                cutoff_minutes = entry_minutes_raw + confirm_pad + time_stop_minutes
            if intraday_only and flatten_minutes_from_open is not None:
                cutoff_minutes = flatten_minutes_from_open if cutoff_minutes is None else min(cutoff_minutes, flatten_minutes_from_open)
            if cutoff_minutes is not None and cutoff_minutes > 0:
                # +1 to be robust if the data API treats end timestamps as exclusive.
                minutes_needed_base = max(minutes_needed_base, cutoff_minutes + 1)
    except Exception:
        max_entry_minutes = max(max_entry_minutes, 1)
    if use_intraday_entry:
        minutes_needed_base = max(minutes_needed_base, max_entry_minutes)

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
    symbol_overrides_used = 0
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

        signals_by_date = _SIGNALS_BY_SYMBOL.get(symbol)
        if signals_by_date is None:
            signals_all = generate_signals_cached(symbol, cfg, data_store)
            if not signals_all:
                continue
            signals_by_date = {str(s.signal_date): s for s in signals_all}
            _SIGNALS_BY_SYMBOL[symbol] = signals_by_date

        # Only include symbols that have a signal for the target trading day.
        # This keeps parity with live: the watchlist for day D is the set of symbols we would actually
        # attempt to trade on day D (signal uses daily data through D-1 close only).
        signal_today = signals_by_date.get(str(tgt))
        if signal_today is None:
            # Support building a watchlist for the "next" trading day before a daily bar exists for that date.
            signal_today = generate_signal_for_date(symbol, tgt, cfg, data_store)
        if signal_today is None:
            continue

        signal_today_direction = ""
        direction_key = "all"
        if directional_history_only:
            signal_today_direction = str(getattr(signal_today, "direction", "") or "").lower()
            direction_key = signal_today_direction or "x"

        state = _ROLLING_STATE.get((symbol, direction_key))
        if state is None:
            state = SymbolRollingState(signals_by_date=signals_by_date, window_signal_dates=set(), acc_by_candidate={})
            _ROLLING_STATE[(symbol, direction_key)] = state

        funnel["scanned_symbols"] += 1

        # Signal dates eligible for scoring are only through D-1 (end_date).
        if directional_history_only:
            new_window_dates = {
                d
                for d, sig in signals_by_date.items()
                if start_date <= d <= end_date and str(getattr(sig, "direction", "") or "").lower() == signal_today_direction
            }
        else:
            new_window_dates = {d for d in signals_by_date.keys() if start_date <= d <= end_date}
        funnel["signals_found"] += len(new_window_dates)

        # Initialize rolling accumulators once per symbol.
        if not state.acc_by_candidate:
            for entry_time in entry_times:
                for grid_idx in range(len(param_grid)):
                    state.acc_by_candidate[(entry_time, grid_idx)] = CandidateAcc()

        # If callers move backwards in time (or window jumps earlier), rebuild from scratch for safety.
        if state.window_start_date and (start_date < state.window_start_date or end_date < state.window_end_date):
            state.window_signal_dates = set()
            for k in list(state.acc_by_candidate.keys()):
                state.acc_by_candidate[k] = CandidateAcc()

        removed_dates = state.window_signal_dates - new_window_dates
        added_dates = new_window_dates - state.window_signal_dates

        # Remove dropped signal dates using only cached outcomes (do not fetch intraday here).
        for d in sorted(removed_dates):
            sig = signals_by_date.get(d)
            if sig is None:
                continue
            for entry_time in entry_times:
                for grid_idx, overrides in enumerate(param_grid):
                    cache_key = _trade_cache_key(
                        trade_ns,
                        symbol,
                        d,
                        str(getattr(sig, "direction", "") or "").lower(),
                        entry_time,
                        overrides,
                    )
                    lite = _TRADE_LITE_CACHE.get(cache_key)
                    if lite is not None:
                        _acc_sub(state.acc_by_candidate[(entry_time, grid_idx)], lite, d)

        # Add newly included signal dates (compute missing candidates as needed).
        for d in sorted(added_dates):
            sig = signals_by_date.get(d)
            if sig is None:
                continue
            lites = _trade_lites_for_signal(
                symbol,
                sig,
                trade_ns=trade_ns,
                cfg=cfg,
                data_store=data_store,
                bars_daily=bars,
                entry_times=entry_times,
                param_grid=param_grid,
                minutes_needed_base=minutes_needed_base,
                apply_confirm=apply_confirm,
                confirm_minutes=confirm_minutes,
            )
            for entry_time in entry_times:
                for grid_idx in range(len(param_grid)):
                    lite = lites.get((entry_time, grid_idx))
                    if lite is not None:
                        _acc_add(state.acc_by_candidate[(entry_time, grid_idx)], lite, d)

        state.window_signal_dates = new_window_dates
        state.window_start_date = start_date
        state.window_end_date = end_date

        total_trades_sim = sum(acc.trades_count for acc in state.acc_by_candidate.values())
        funnel["trades_simulated"] += int(total_trades_sim)

        # Pick best entry time + param combo per symbol.
        # Important: when minTrades is configured, prefer candidates that already satisfy it.
        # This avoids selecting a high-score/low-sample candidate and rejecting the symbol later.
        best_time = entry_times[0]
        best_params = param_grid[0] if param_grid else {}
        best_stats = _stats_from_acc(state.acc_by_candidate.get((best_time, 0), CandidateAcc()))

        def _score(stats: Dict[str, float]) -> float:
            if param_rank_by in ("total_pnl_pct", "pnl", "total_pnl"):
                return float(stats.get("total_pnl_pct") or 0.0)
            if param_rank_by in ("profit_factor", "pf"):
                return float(stats.get("profit_factor") or 0.0)
            if param_rank_by in ("avgr_lcb", "avgr-lcb", "avgR_lcb", "avgR-lcb", "lcb"):
                base = float(stats.get("avgR") or 0.0)
                se = float(stats.get("avgR_stderr") or 0.0)
                return base - (rank_lcb_z * se)
            return float(stats.get("avgR") or 0.0)

        candidates: List[Tuple[str, Dict, Dict[str, float]]] = []
        for entry_time in entry_times:
            for grid_idx, overrides in enumerate(param_grid):
                stats = _stats_from_acc(state.acc_by_candidate.get((entry_time, grid_idx), CandidateAcc()))
                candidates.append((entry_time, overrides, stats))

        eligible = candidates
        if min_trades > 0:
            with_min_trades = [c for c in candidates if int(c[2].get("trades_count") or 0) >= min_trades]
            if with_min_trades:
                eligible = with_min_trades

        for entry_time, overrides, stats in eligible:
            if _score(stats) > _score(best_stats):
                best_time = entry_time
                best_params = overrides
                best_stats = stats
        param_override_source = "grid_best"
        symbol_override = symbol_param_overrides.get(symbol)
        if symbol_override:
            if symbol_param_override_mode == "replace":
                forced_params = dict(symbol_override)
                param_override_source = "symbol_override_replace"
            else:
                forced_params = dict(best_params or {})
                forced_params.update(symbol_override)
                param_override_source = "symbol_override_merge"

            forced_key = _overrides_key(forced_params)
            forced_grid_idx = grid_idx_by_overrides_key.get(forced_key)
            forced_candidates: List[Tuple[str, Dict, Dict[str, float]]] = []
            if forced_grid_idx is not None:
                for entry_time in entry_times:
                    forced_stats = _stats_from_acc(state.acc_by_candidate.get((entry_time, forced_grid_idx), CandidateAcc()))
                    forced_candidates.append((entry_time, forced_params, forced_stats))
            else:
                forced_acc_by_time: Dict[str, CandidateAcc] = {entry_time: CandidateAcc() for entry_time in entry_times}
                for d in sorted(state.window_signal_dates):
                    sig = signals_by_date.get(d)
                    if sig is None:
                        continue
                    lites = _trade_lites_for_signal(
                        symbol,
                        sig,
                        trade_ns=trade_ns,
                        cfg=cfg,
                        data_store=data_store,
                        bars_daily=bars,
                        entry_times=entry_times,
                        param_grid=[forced_params],
                        minutes_needed_base=minutes_needed_base,
                        apply_confirm=apply_confirm,
                        confirm_minutes=confirm_minutes,
                    )
                    for entry_time in entry_times:
                        lite = lites.get((entry_time, 0))
                        if lite is not None:
                            _acc_add(forced_acc_by_time[entry_time], lite, d)
                for entry_time in entry_times:
                    forced_candidates.append((entry_time, forced_params, _stats_from_acc(forced_acc_by_time[entry_time])))

            forced_eligible = forced_candidates
            if min_trades > 0:
                forced_with_min_trades = [c for c in forced_candidates if int(c[2].get("trades_count") or 0) >= min_trades]
                if forced_with_min_trades:
                    forced_eligible = forced_with_min_trades
            if forced_eligible:
                best_time = forced_eligible[0][0]
                best_params = forced_params
                best_stats = forced_eligible[0][2]
                for entry_time, _, forced_stats in forced_eligible[1:]:
                    if _score(forced_stats) > _score(best_stats):
                        best_time = entry_time
                        best_stats = forced_stats
                symbol_overrides_used += 1
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
        if min_months_count > 0 and int(stats.get("months_count") or 0) < min_months_count:
            reject_counts["min_months_count"] += 1
            reasons.append("min_months_count")
        if min_positive_month_rate is not None and float(stats.get("positive_month_rate") or 0.0) < min_positive_month_rate:
            reject_counts["min_positive_month_rate"] += 1
            reasons.append("min_positive_month_rate")
        if max_negative_months is not None and int(stats.get("negative_months") or 0) > max_negative_months:
            reject_counts["max_negative_months"] += 1
            reasons.append("max_negative_months")
        if (
            max_monthly_drawdown_pct is not None
            and float(stats.get("max_monthly_drawdown_pct") or 0.0) > max_monthly_drawdown_pct
        ):
            reject_counts["max_monthly_drawdown_pct"] += 1
            reasons.append("max_monthly_drawdown_pct")
        if (
            max_longest_negative_month_streak is not None
            and int(stats.get("longest_negative_month_streak") or 0) > max_longest_negative_month_streak
        ):
            reject_counts["max_longest_negative_month_streak"] += 1
            reasons.append("max_longest_negative_month_streak")
        if (
            min_worst_month_pnl_pct is not None
            and float(stats.get("worst_month_pnl_pct") or 0.0) < min_worst_month_pnl_pct
        ):
            reject_counts["min_worst_month_pnl_pct"] += 1
            reasons.append("min_worst_month_pnl_pct")
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
                    "direction": str(getattr(signal_today, "direction", "") or "").lower(),
                    "entry_time_et": best_time,
                    "param_overrides": best_params,
                    "param_override_source": param_override_source,
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
        watchlist.append(
            {
                "symbol": symbol,
                "direction": str(getattr(signal_today, "direction", "") or "").lower(),
                "entry_time_et": best_time,
                "param_overrides": best_params,
                "param_override_source": param_override_source,
                **stats,
            }
        )
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
            if top_k_rank_by in ("avgr_lcb", "avgr-lcb", "avgR_lcb", "avgR-lcb", "lcb"):
                base = float(row.get("avgR") or 0.0)
                se = float(row.get("avgR_stderr") or 0.0)
                return base - (rank_lcb_z * se)
            if top_k_rank_by in ("profit_factor", "pf"):
                return float(row.get("profit_factor") or 0.0)
            if top_k_rank_by in ("win_rate", "winrate"):
                return float(row.get("win_rate") or 0.0)
            return float(row.get("total_pnl_pct") or 0.0)
        # Deterministic tie-break by symbol prevents run-to-run ordering drift when scores tie.
        watchlist.sort(key=lambda row: (-float(_rank_key(row)), str(row.get("symbol") or "")))
        watchlist = watchlist[:top_k]

    symbol_overrides_in_watchlist = sum(
        1
        for row in watchlist
        if str(row.get("param_override_source") or "").startswith("symbol_override")
    )

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
        if reject_counts["min_months_count"] > 0:
            logging.info("[WATCHLIST_FILTERS] date=%s reject_min_months_count=%s", tgt, reject_counts["min_months_count"])
        if reject_counts["min_positive_month_rate"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_min_positive_month_rate=%s",
                tgt,
                reject_counts["min_positive_month_rate"],
            )
        if reject_counts["max_negative_months"] > 0:
            logging.info("[WATCHLIST_FILTERS] date=%s reject_max_negative_months=%s", tgt, reject_counts["max_negative_months"])
        if reject_counts["max_monthly_drawdown_pct"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_max_monthly_drawdown_pct=%s",
                tgt,
                reject_counts["max_monthly_drawdown_pct"],
            )
        if reject_counts["max_longest_negative_month_streak"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_max_longest_negative_month_streak=%s",
                tgt,
                reject_counts["max_longest_negative_month_streak"],
            )
        if reject_counts["min_worst_month_pnl_pct"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_min_worst_month_pnl_pct=%s",
                tgt,
                reject_counts["min_worst_month_pnl_pct"],
            )
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
    entry_time_counts: Dict[str, int] = {}
    if watchlist:
        for row in watchlist:
            et = str(row.get("entry_time_et") or "")
            entry_time_counts[et] = entry_time_counts.get(et, 0) + 1

    watchlist_meta = {
        "funnel": dict(funnel),
        "reject_counts": dict(reject_counts),
        "selected_summary": summarize_watchlist_rows(watchlist),
        "entry_time_counts": entry_time_counts,
        "symbol_param_overrides_used_pre_top_k": int(symbol_overrides_used),
        "symbol_param_overrides_used_in_watchlist": int(symbol_overrides_in_watchlist),
        "filters": {
            "minTrades": min_trades,
            "reject_negative_pnl": reject_negative_pnl,
            "minProfitFactor": min_profit_factor,
            "minAvgR": min_avg_r,
            "minWinRate": min_win_rate,
            "min_months_count": min_months_count,
            "min_positive_month_rate": min_positive_month_rate,
            "max_negative_months": max_negative_months,
            "max_monthly_drawdown_pct": max_monthly_drawdown_pct,
            "max_longest_negative_month_streak": max_longest_negative_month_streak,
            "min_worst_month_pnl_pct": min_worst_month_pnl_pct,
            "top_k": top_k,
            "entry_time_rank_by": entry_time_rank_by,
            "top_k_rank_by": top_k_rank_by,
            "param_rank_by": param_rank_by,
            "directional_history_only": directional_history_only,
            "symbol_param_override_mode": symbol_param_override_mode,
            "symbol_param_overrides_count": len(symbol_param_overrides),
        },
    }

    if watchlist:
        write_watchlist(watchlist, cfg, date_str=tgt, meta=watchlist_meta)
    else:
        logging.warning("[WATCHLIST] empty watchlist date=%s; no fallback applied", tgt)
        # Overwrite any stale watchlist for this date so replay can't pick up old symbols.
        write_watchlist([], cfg, date_str=tgt, meta=watchlist_meta)
    if watchlist:
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
                "min_months_count": min_months_count,
                "min_positive_month_rate": min_positive_month_rate,
                "max_negative_months": max_negative_months,
                "max_monthly_drawdown_pct": max_monthly_drawdown_pct,
                "max_longest_negative_month_streak": max_longest_negative_month_streak,
                "min_worst_month_pnl_pct": min_worst_month_pnl_pct,
                "top_k": top_k,
                "entry_time_rank_by": entry_time_rank_by,
                "top_k_rank_by": top_k_rank_by,
                "param_rank_by": param_rank_by,
                "param_grid": watch_cfg.get("param_grid") or {},
                "directional_history_only": directional_history_only,
                "symbol_param_override_mode": symbol_param_override_mode,
                "symbol_param_overrides_count": len(symbol_param_overrides),
            },
            "summary": {
                **funnel,
                **reject_counts,
                "symbol_overrides_used_pre_top_k": int(symbol_overrides_used),
                "symbol_overrides_used_in_watchlist": int(symbol_overrides_in_watchlist),
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
