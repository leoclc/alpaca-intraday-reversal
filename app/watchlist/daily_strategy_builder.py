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
from app.strategies.daily_trend_reversal import (
    build_trade,
    generate_signal_for_date,
    generate_signals_cached,
    resolve_entry_times,
)
from app.utils.time import ensure_date, parse_time_hhmm
from app.watchlist.day_filter import summarize_watchlist_rows
from app.watchlist.node_assets import resolve_asset_universe_symbols
from app.watchlist.storage import expected_watchlist_date_str, freeze_watchlist_snapshot, write_watchlist


@dataclass(frozen=True)
class TradeLite:
    pnl_pct: float
    r_multiple: float
    exit_reason: str = ""
    selected_entry_time_et: str = ""
    mfe_r_before_stop: float | None = None
    mae_r_to_target: float | None = None
    mfe_r_full: float | None = None
    mae_r_full: float | None = None
    target_r: float | None = None
    target_hit: bool = False


@dataclass
class CandidateAcc:
    trades_count: int = 0
    wins: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0
    sum_pnl_pct: float = 0.0
    gross_profit_r: float = 0.0
    gross_loss_r: float = 0.0  # abs(sum(neg r))
    stop_count: int = 0
    target_count: int = 0
    cutoff_count: int = 0
    stop_flipable_any_count: int = 0
    stop_flipable_050_count: int = 0
    stop_flipable_100_count: int = 0
    stop_no_progress_count: int = 0
    stop_near_target_count: int = 0
    cutoff_no_progress_count: int = 0
    cutoff_near_target_count: int = 0
    cutoff_mfe_r_sum: float = 0.0
    cutoff_mfe_r_count: int = 0
    cutoff_abs_mae_r_sum: float = 0.0
    cutoff_abs_mae_r_count: int = 0
    stop_reach_target_with_wider_stop_count: int = 0
    stop_mult_needed_sum: float = 0.0
    stop_mult_needed_count: int = 0
    target_r_sum: float = 0.0
    target_r_count: int = 0
    month_pnl_pct: Dict[str, float] = field(default_factory=dict)
    month_trades: Dict[str, int] = field(default_factory=dict)
    selected_entry_time_counts: Dict[str, int] = field(default_factory=dict)


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
_TRADE_LITE_CACHE_VERSION = 10

_CUTOFF_EXIT_REASONS = {"time_stop", "eod_flat", "time_exit"}
_SCAN_FIRST_VALID_LABEL = "__scan_first_valid__"


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
            if isinstance(val, list) and len(val) >= 6:
                try:
                    inner[ok] = TradeLite(
                        pnl_pct=float(val[0]),
                        r_multiple=float(val[1]),
                        exit_reason=str(val[2] or ""),
                        selected_entry_time_et=(str(val[9]) if len(val) > 9 and val[9] is not None else ""),
                        mfe_r_before_stop=(None if val[3] is None else float(val[3])),
                        mae_r_to_target=(None if val[4] is None else float(val[4])),
                        mfe_r_full=(None if len(val) <= 6 or val[6] is None else float(val[6])),
                        mae_r_full=(None if len(val) <= 8 or val[8] is None else float(val[8])),
                        target_r=(None if len(val) <= 7 or val[7] is None else float(val[7])),
                        target_hit=bool(val[5]),
                    )
                except Exception:
                    inner[ok] = None
                continue
            if isinstance(val, dict):
                try:
                    inner[ok] = TradeLite(
                        pnl_pct=float(val.get("pnl_pct") or 0.0),
                        r_multiple=float(val.get("r_multiple") or 0.0),
                        exit_reason=str(val.get("exit_reason") or ""),
                        selected_entry_time_et=str(val.get("selected_entry_time_et") or ""),
                        mfe_r_before_stop=(
                            None
                            if val.get("mfe_r_before_stop") is None
                            else float(val.get("mfe_r_before_stop"))
                        ),
                        mae_r_to_target=(
                            None
                            if val.get("mae_r_to_target") is None
                            else float(val.get("mae_r_to_target"))
                        ),
                        mfe_r_full=(
                            None
                            if val.get("mfe_r_full") is None
                            else float(val.get("mfe_r_full"))
                        ),
                        mae_r_full=(
                            None
                            if val.get("mae_r_full") is None
                            else float(val.get("mae_r_full"))
                        ),
                        target_r=(
                            None
                            if val.get("target_r") is None
                            else float(val.get("target_r"))
                        ),
                        target_hit=bool(val.get("target_hit") or False),
                    )
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
                    row[ok] = {
                        "pnl_pct": float(lite.pnl_pct),
                        "r_multiple": float(lite.r_multiple),
                        "exit_reason": str(lite.exit_reason or ""),
                        "selected_entry_time_et": str(lite.selected_entry_time_et or ""),
                        "mfe_r_before_stop": (None if lite.mfe_r_before_stop is None else float(lite.mfe_r_before_stop)),
                        "mae_r_to_target": (None if lite.mae_r_to_target is None else float(lite.mae_r_to_target)),
                        "mfe_r_full": (None if lite.mfe_r_full is None else float(lite.mfe_r_full)),
                        "mae_r_full": (None if lite.mae_r_full is None else float(lite.mae_r_full)),
                        "target_r": (None if lite.target_r is None else float(lite.target_r)),
                        "target_hit": bool(lite.target_hit),
                    }
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


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        low, high = high, low
    return max(low, min(high, value))


def _as_float_list(raw: Any) -> List[float]:
    out: List[float] = []
    if isinstance(raw, list):
        src = raw
    elif raw is None:
        src = []
    else:
        src = [raw]
    for v in src:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _as_int_list(raw: Any) -> List[int]:
    out: List[int] = []
    if isinstance(raw, list):
        src = raw
    elif raw is None:
        src = []
    else:
        src = [raw]
    for v in src:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _auto_symbol_param_candidates(
    base_params: Dict[str, Any],
    stats: Dict[str, float],
    strategy_params: Dict[str, Any],
    auto_cfg: Dict[str, Any],
) -> Tuple[List[Tuple[Dict[str, Any], str]], Dict[str, Any]]:
    cfg = auto_cfg if isinstance(auto_cfg, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    meta: Dict[str, Any] = {
        "enabled": enabled,
        "considered": False,
        "reason": "",
        "candidate_count": 0,
    }
    if not enabled:
        meta["reason"] = "disabled"
        return [], meta

    trades_count = int(stats.get("trades_count") or 0)
    stop_count = int(stats.get("stop_count") or 0)
    cutoff_count = int(stats.get("cutoff_count") or 0)
    min_trades = max(1, int(cfg.get("min_trades") or 12))
    min_stop_count = max(1, int(cfg.get("min_stop_count") or 6))
    time_stop_target_fit_enabled = bool(cfg.get("time_stop_target_fit_enabled", True))
    time_stop_target_fit_min_cutoff_count = max(1, int(cfg.get("time_stop_target_fit_min_cutoff_count") or 8))
    time_stop_stop_fit_enabled = bool(cfg.get("time_stop_stop_fit_enabled", True))
    time_stop_stop_fit_min_cutoff_count = max(1, int(cfg.get("time_stop_stop_fit_min_cutoff_count") or 8))
    can_use_stop_based_tuning = stop_count >= min_stop_count
    can_use_time_stop_fit_tuning = (
        (time_stop_target_fit_enabled and cutoff_count >= time_stop_target_fit_min_cutoff_count)
        or (time_stop_stop_fit_enabled and cutoff_count >= time_stop_stop_fit_min_cutoff_count)
    )
    if trades_count < min_trades or (not can_use_stop_based_tuning and not can_use_time_stop_fit_tuning):
        meta["reason"] = "insufficient_samples"
        return [], meta
    meta["considered"] = True

    target_rr_min = max(0.05, _safe_float(cfg.get("target_rr_min"), 0.6))
    target_rr_max = max(target_rr_min, _safe_float(cfg.get("target_rr_max"), 2.5))
    stop_r_max = max(0.1, _safe_float(cfg.get("stop_r_max"), 1.8))
    target_window_mult_min = max(0.05, _safe_float(cfg.get("target_window_mult_min"), 0.2))
    target_window_mult_max = max(target_window_mult_min, _safe_float(cfg.get("target_window_mult_max"), 1.5))

    target_rr_base = _safe_float(
        base_params.get("target_rr"),
        _safe_float(base_params.get("target_r"), _safe_float(strategy_params.get("target_rr"), _safe_float(strategy_params.get("target_r"), 1.5))),
    )
    stop_r_base = _safe_float(base_params.get("stop_r"), _safe_float(strategy_params.get("stop_r"), 1.0))
    stop_atr_mult_base = _safe_float(
        base_params.get("stop_atr_mult"),
        _safe_float(strategy_params.get("stop_atr_mult"), 1.0),
    )
    target_window_mult_base = _safe_float(
        base_params.get("target_window_mult"),
        _safe_float(strategy_params.get("target_window_mult"), 0.8),
    )
    stop_mode = str(base_params.get("stop_mode") or strategy_params.get("stop_mode") or "").lower().strip()
    target_mode = str(base_params.get("target_mode") or strategy_params.get("target_mode") or "rr").lower().strip()
    stop_rate = _safe_float(stats.get("stop_rate"), 0.0)
    stop_flip_share_050 = _safe_float(stats.get("stop_flip_share_050"), 0.0)
    stop_flip_share_100 = _safe_float(stats.get("stop_flip_share_100"), 0.0)
    stop_wider_share = _safe_float(stats.get("stop_reach_target_with_wider_stop_share"), 0.0)
    avg_stop_mult_needed = _safe_float(stats.get("avg_stop_mult_needed_for_target"), 0.0)
    cutoff_rate = _safe_float(stats.get("cutoff_rate"), 0.0)
    cutoff_avg_mfe_r = _safe_float(stats.get("cutoff_avg_mfe_r"), 0.0)
    cutoff_avg_abs_mae_r = _safe_float(stats.get("cutoff_avg_abs_mae_r"), 0.0)
    stop_no_progress_share = _safe_float(stats.get("stop_no_progress_share"), 0.0)
    cutoff_no_progress_share = _safe_float(stats.get("cutoff_no_progress_share"), 0.0)
    cutoff_near_target_share = _safe_float(stats.get("cutoff_near_target_share"), 0.0)
    loser_no_progress_share = _safe_float(stats.get("loser_no_progress_share"), 0.0)

    max_early_pullback_base = _safe_float(
        base_params.get("max_early_pullback_bps"),
        _safe_float(strategy_params.get("max_early_pullback_bps"), 0.0),
    )
    min_early_reversal_base = _safe_float(
        base_params.get("min_early_reversal_bps"),
        _safe_float(strategy_params.get("min_early_reversal_bps"), 0.0),
    )
    confirm_move_bps_base = _safe_float(
        base_params.get("confirm_move_bps"),
        _safe_float(strategy_params.get("confirm_move_bps"), 0.0),
    )
    try:
        confirm_minutes_base = int(base_params.get("confirm_minutes") or strategy_params.get("confirm_minutes") or 0)
    except Exception:
        confirm_minutes_base = 0
    try:
        time_stop_minutes_base = int(base_params.get("time_stop_minutes") or strategy_params.get("time_stop_minutes") or 0)
    except Exception:
        time_stop_minutes_base = 0

    min_stop_flip_share_050 = _safe_float(cfg.get("min_stop_flip_share_050"), 0.35)
    min_stop_flip_share_100 = _safe_float(cfg.get("min_stop_flip_share_100"), 0.15)
    min_stop_rate_for_stop_flip_tuning = _safe_float(cfg.get("min_stop_rate_for_stop_flip_tuning"), 0.35)
    target_rr_reduce_mult = _safe_float(cfg.get("target_rr_reduce_mult"), 0.85)
    target_rr_widen_mult = _safe_float(cfg.get("target_rr_widen_mult"), 0.85)
    target_rr_widen_buffer = _safe_float(cfg.get("target_rr_widen_buffer"), 1.05)
    target_window_mult_reduce_mult = _safe_float(cfg.get("target_window_mult_reduce_mult"), 0.85)
    cutoff_rate_threshold = _safe_float(cfg.get("cutoff_rate_threshold"), 0.45)
    cutoff_target_rr_reduce_mult = _safe_float(cfg.get("cutoff_target_rr_reduce_mult"), 0.90)
    cutoff_target_window_mult_reduce_mult = _safe_float(cfg.get("cutoff_target_window_mult_reduce_mult"), 0.90)

    min_wider_stop_share = _safe_float(cfg.get("min_wider_stop_share"), 0.20)
    min_stop_rate_for_wider_stop_tuning = _safe_float(cfg.get("min_stop_rate_for_wider_stop_tuning"), 0.30)
    stop_r_widen_mult = _safe_float(cfg.get("stop_r_widen_mult"), 1.15)
    stop_r_widen_buffer = _safe_float(cfg.get("stop_r_widen_buffer"), 1.05)
    max_candidates = max(1, int(cfg.get("max_candidates") or 4))

    entry_quality_min_stop_rate = _safe_float(cfg.get("entry_quality_min_stop_rate"), 0.30)
    entry_quality_min_stop_no_progress_share = _safe_float(cfg.get("entry_quality_min_stop_no_progress_share"), 0.35)
    entry_quality_min_loser_no_progress_share = _safe_float(cfg.get("entry_quality_min_loser_no_progress_share"), 0.30)
    entry_quality_min_cutoff_no_progress_share = _safe_float(cfg.get("entry_quality_min_cutoff_no_progress_share"), 0.45)
    entry_quality_enable_early_filter_tuning = bool(cfg.get("entry_quality_enable_early_filter_tuning", True))
    entry_quality_enable_confirm_candidates = bool(cfg.get("entry_quality_enable_confirm_candidates", True))
    entry_quality_min_early_reversal_values = sorted({
        v for v in _as_float_list(cfg.get("entry_quality_min_early_reversal_bps_values")) if v > 0
    })
    if not entry_quality_min_early_reversal_values:
        entry_quality_min_early_reversal_values = [18.0, 20.0, 25.0]
    entry_quality_max_early_pullback_values = sorted({
        v for v in _as_float_list(cfg.get("entry_quality_max_early_pullback_bps_values")) if v > 0
    }, reverse=True)
    if not entry_quality_max_early_pullback_values:
        entry_quality_max_early_pullback_values = [35.0, 30.0, 25.0]
    entry_quality_confirm_move_values = sorted({
        v for v in _as_float_list(cfg.get("entry_quality_confirm_move_bps_values")) if v > 0
    })
    if not entry_quality_confirm_move_values:
        entry_quality_confirm_move_values = [2.0, 3.0]
    entry_quality_confirm_minutes_values = sorted({
        v for v in _as_int_list(cfg.get("entry_quality_confirm_minutes_values")) if v > 0
    })
    if not entry_quality_confirm_minutes_values:
        entry_quality_confirm_minutes_values = [2]
    entry_quality_min_cutoff_near_target_share = _safe_float(cfg.get("entry_quality_min_cutoff_near_target_share"), 0.40)
    time_stop_expand_mult = _safe_float(cfg.get("time_stop_expand_mult"), 1.25)
    time_stop_max = max(1, int(cfg.get("time_stop_max") or 90))
    entry_quality_enable_time_extension = bool(cfg.get("entry_quality_enable_time_extension", False))
    time_stop_target_fit_min_cutoff_rate = _safe_float(cfg.get("time_stop_target_fit_min_cutoff_rate"), 0.40)
    time_stop_target_fit_mult = _safe_float(cfg.get("time_stop_target_fit_mult"), 1.15)
    time_stop_target_fit_floor = max(0.05, _safe_float(cfg.get("time_stop_target_fit_floor"), target_rr_min))
    time_stop_target_fit_min_delta = max(0.0, _safe_float(cfg.get("time_stop_target_fit_min_delta"), 0.05))
    time_stop_target_fit_enable_time_extension = bool(cfg.get("time_stop_target_fit_enable_time_extension", False))
    time_stop_target_fit_time_expand_cap = max(1.05, _safe_float(cfg.get("time_stop_target_fit_time_expand_cap"), 2.5))
    time_stop_target_fit_time_min_delta_minutes = max(
        1,
        int(cfg.get("time_stop_target_fit_time_min_delta_minutes") or 5),
    )
    time_stop_stop_fit_min_cutoff_rate = _safe_float(cfg.get("time_stop_stop_fit_min_cutoff_rate"), 0.40)
    time_stop_stop_fit_mult = max(1.0, _safe_float(cfg.get("time_stop_stop_fit_mult"), 1.35))
    time_stop_stop_fit_atr_mult_min = max(0.01, _safe_float(cfg.get("time_stop_stop_fit_atr_mult_min"), 0.20))
    time_stop_stop_fit_atr_mult_max = max(
        time_stop_stop_fit_atr_mult_min,
        _safe_float(cfg.get("time_stop_stop_fit_atr_mult_max"), stop_atr_mult_base),
    )
    time_stop_stop_fit_min_delta = max(0.0, _safe_float(cfg.get("time_stop_stop_fit_min_delta"), 0.05))

    base_key = _overrides_key(base_params)
    seen_keys = {base_key}
    candidates: List[Tuple[Dict[str, Any], str]] = []
    target_candidate: Optional[Dict[str, Any]] = None
    stop_candidate: Optional[Dict[str, Any]] = None
    target_reason = ""
    stop_reason = ""
    fit_target_candidate: Optional[Dict[str, Any]] = None
    fit_target_reason = ""
    fit_stop_candidate: Optional[Dict[str, Any]] = None
    fit_stop_reason = ""

    def _add_candidate(params: Dict[str, Any], reason: str) -> None:
        if len(candidates) >= max_candidates:
            return
        key = _overrides_key(params)
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append((params, reason))

    if (
        stop_rate >= min_stop_rate_for_stop_flip_tuning
        and (stop_flip_share_050 >= min_stop_flip_share_050 or stop_flip_share_100 >= min_stop_flip_share_100)
    ):
        if stop_mode == "target_rr" and target_mode == "symbol_window_avg":
            tuned_target_mult = _clamp(
                target_window_mult_base * target_window_mult_reduce_mult,
                target_window_mult_min,
                target_window_mult_max,
            )
            if tuned_target_mult < (target_window_mult_base - 1e-9):
                target_candidate = dict(base_params)
                target_candidate["target_window_mult"] = round(float(tuned_target_mult), 4)
                target_reason = "reduce_target_window_mult_stop_flip"
                _add_candidate(target_candidate, target_reason)
        else:
            tuned_target_rr = _clamp(target_rr_base * target_rr_reduce_mult, target_rr_min, target_rr_max)
            if tuned_target_rr < (target_rr_base - 1e-9):
                target_candidate = dict(base_params)
                target_candidate["target_rr"] = round(float(tuned_target_rr), 4)
                target_reason = "reduce_target_rr_stop_flip"
                _add_candidate(target_candidate, target_reason)

    if stop_rate >= min_stop_rate_for_wider_stop_tuning and stop_wider_share >= min_wider_stop_share:
        if stop_mode == "target_rr":
            widened_rr = target_rr_base * target_rr_widen_mult
            if avg_stop_mult_needed > 1.0:
                widened_rr = min(
                    widened_rr,
                    target_rr_base / max(1.0, (avg_stop_mult_needed * target_rr_widen_buffer)),
                )
            widened_rr = _clamp(widened_rr, target_rr_min, target_rr_max)
            if widened_rr < (target_rr_base - 1e-9):
                stop_candidate = dict(base_params)
                stop_candidate["target_rr"] = round(float(widened_rr), 4)
                stop_reason = "widen_stop_via_target_rr"
                _add_candidate(stop_candidate, stop_reason)
        else:
            widened_stop_r = stop_r_base * stop_r_widen_mult
            if avg_stop_mult_needed > 1.0:
                widened_stop_r = max(widened_stop_r, stop_r_base * avg_stop_mult_needed * stop_r_widen_buffer)
            widened_stop_r = _clamp(widened_stop_r, stop_r_base, stop_r_max)
            if widened_stop_r > (stop_r_base + 1e-9):
                stop_candidate = dict(base_params)
                stop_candidate["stop_r"] = round(float(widened_stop_r), 4)
                stop_reason = "widen_stop_r_recover_target"
                _add_candidate(stop_candidate, stop_reason)

    if cutoff_rate >= cutoff_rate_threshold:
        if stop_mode == "target_rr" and target_mode == "symbol_window_avg":
            tuned_cutoff_mult = _clamp(
                target_window_mult_base * cutoff_target_window_mult_reduce_mult,
                target_window_mult_min,
                target_window_mult_max,
            )
            if tuned_cutoff_mult < (target_window_mult_base - 1e-9):
                cutoff_candidate = dict(base_params)
                cutoff_candidate["target_window_mult"] = round(float(tuned_cutoff_mult), 4)
                _add_candidate(cutoff_candidate, "reduce_target_window_mult_cutoff")
        else:
            tuned_cutoff_rr = _clamp(target_rr_base * cutoff_target_rr_reduce_mult, target_rr_min, target_rr_max)
            if tuned_cutoff_rr < (target_rr_base - 1e-9):
                cutoff_candidate = dict(base_params)
                cutoff_candidate["target_rr"] = round(float(tuned_cutoff_rr), 4)
                _add_candidate(cutoff_candidate, "reduce_target_rr_cutoff")

    # Keep targets realistic for short time-stop windows by calibrating to observed
    # favorable move achieved before cutoff (in R) for this symbol/candidate.
    if (
        time_stop_target_fit_enabled
        and stop_mode == "target_rr"
        and time_stop_minutes_base > 0
        and cutoff_count >= time_stop_target_fit_min_cutoff_count
        and cutoff_rate >= time_stop_target_fit_min_cutoff_rate
        and cutoff_avg_mfe_r > 0
    ):
        fitted_target_rr = _clamp(
            max(time_stop_target_fit_floor, cutoff_avg_mfe_r * max(0.1, time_stop_target_fit_mult)),
            target_rr_min,
            target_rr_max,
        )
        if (target_rr_base - fitted_target_rr) >= time_stop_target_fit_min_delta:
            tuned = dict(base_params)
            tuned["target_rr"] = round(float(fitted_target_rr), 4)
            fit_target_reason = "fit_target_rr_to_cutoff_mfe"
            fit_target_candidate = tuned
            _add_candidate(tuned, fit_target_reason)
        if time_stop_target_fit_enable_time_extension:
            # If observed movement before cutoff is much smaller than target_r, try extending
            # time_stop to better match the movement horizon instead of only shrinking target.
            denom = max(1e-6, cutoff_avg_mfe_r * max(0.1, time_stop_target_fit_mult))
            needed_mult = target_rr_base / denom
            if needed_mult > 1.05:
                time_mult = min(time_stop_target_fit_time_expand_cap, max(1.05, needed_mult))
                widened_time_stop = int(round(float(time_stop_minutes_base) * float(time_mult)))
                widened_time_stop = min(widened_time_stop, time_stop_max)
                if widened_time_stop >= (time_stop_minutes_base + time_stop_target_fit_time_min_delta_minutes):
                    tuned = dict(base_params)
                    tuned["time_stop_minutes"] = int(widened_time_stop)
                    _add_candidate(tuned, "extend_time_stop_target_fit")

    # Keep stop distance realistic for short time-stop windows by calibrating
    # ATR stop size to observed adverse movement before cutoff.
    if (
        time_stop_stop_fit_enabled
        and target_mode == "rr"
        and time_stop_minutes_base > 0
        and cutoff_count >= time_stop_stop_fit_min_cutoff_count
        and cutoff_rate >= time_stop_stop_fit_min_cutoff_rate
        and cutoff_avg_abs_mae_r > 0
        and stop_atr_mult_base > 0
    ):
        fitted_stop_atr_mult = stop_atr_mult_base * cutoff_avg_abs_mae_r * time_stop_stop_fit_mult
        fitted_stop_atr_mult = _clamp(
            fitted_stop_atr_mult,
            time_stop_stop_fit_atr_mult_min,
            time_stop_stop_fit_atr_mult_max,
        )
        if (stop_atr_mult_base - fitted_stop_atr_mult) >= time_stop_stop_fit_min_delta:
            tuned = dict(base_params)
            tuned["stop_atr_mult"] = round(float(fitted_stop_atr_mult), 4)
            fit_stop_reason = "fit_stop_atr_to_cutoff_mae"
            fit_stop_candidate = tuned
            _add_candidate(tuned, fit_stop_reason)

    # Evaluate the joint fit explicitly: in practice, a symbol can need
    # both a reachable target and a cutoff-aware stop size.
    if fit_target_candidate is not None and fit_stop_candidate is not None:
        fit_combo = dict(base_params)
        if "target_rr" in fit_target_candidate:
            fit_combo["target_rr"] = fit_target_candidate.get("target_rr")
        if "time_stop_minutes" in fit_target_candidate:
            fit_combo["time_stop_minutes"] = fit_target_candidate.get("time_stop_minutes")
        if "stop_atr_mult" in fit_stop_candidate:
            fit_combo["stop_atr_mult"] = fit_stop_candidate.get("stop_atr_mult")
        combo_reason = "+".join([p for p in [fit_target_reason, fit_stop_reason] if p]) or "fit_target_and_stop_to_cutoff"
        _add_candidate(fit_combo, combo_reason)

    entry_quality_issue = (
        (stop_rate >= entry_quality_min_stop_rate)
        and (
            stop_no_progress_share >= entry_quality_min_stop_no_progress_share
            or loser_no_progress_share >= entry_quality_min_loser_no_progress_share
        )
    ) or (
        cutoff_rate >= cutoff_rate_threshold
        and cutoff_no_progress_share >= entry_quality_min_cutoff_no_progress_share
    )
    if entry_quality_issue:
        if entry_quality_enable_early_filter_tuning:
            tighter_pullback_vals = [
                v for v in entry_quality_max_early_pullback_values if (max_early_pullback_base <= 0 or v < max_early_pullback_base)
            ]
            stronger_reversal_vals = [v for v in entry_quality_min_early_reversal_values if v > min_early_reversal_base]
            for val in tighter_pullback_vals:
                tuned = dict(base_params)
                tuned["max_early_pullback_bps"] = round(float(val), 4)
                _add_candidate(tuned, "tighten_early_pullback_no_progress")
            for val in stronger_reversal_vals:
                tuned = dict(base_params)
                tuned["min_early_reversal_bps"] = round(float(val), 4)
                _add_candidate(tuned, "raise_early_reversal_no_progress")
            if tighter_pullback_vals and stronger_reversal_vals:
                combo = dict(base_params)
                combo["max_early_pullback_bps"] = round(float(tighter_pullback_vals[0]), 4)
                combo["min_early_reversal_bps"] = round(float(stronger_reversal_vals[0]), 4)
                _add_candidate(combo, "tighten_entry_quality_no_progress")

        if entry_quality_enable_confirm_candidates:
            for move_bps in entry_quality_confirm_move_values:
                for mins in entry_quality_confirm_minutes_values:
                    is_stricter_than_base = (
                        confirm_move_bps_base <= 0
                        or confirm_minutes_base <= 0
                        or move_bps > confirm_move_bps_base
                        or mins > confirm_minutes_base
                    )
                    if not is_stricter_than_base:
                        continue
                    tuned = dict(base_params)
                    tuned["confirm_move_bps"] = round(float(move_bps), 4)
                    tuned["confirm_minutes"] = int(mins)
                    tuned["confirm_apply_in_watchlist"] = True
                    _add_candidate(tuned, "add_confirm_no_progress")

        if (
            entry_quality_enable_time_extension
            and time_stop_minutes_base > 0
            and cutoff_near_target_share >= entry_quality_min_cutoff_near_target_share
        ):
            widened_time_stop = int(round(float(time_stop_minutes_base) * float(time_stop_expand_mult)))
            widened_time_stop = max(time_stop_minutes_base + 1, widened_time_stop)
            widened_time_stop = min(widened_time_stop, time_stop_max)
            if widened_time_stop > time_stop_minutes_base:
                tuned = dict(base_params)
                tuned["time_stop_minutes"] = int(widened_time_stop)
                _add_candidate(tuned, "extend_time_stop_cutoff_near_target")

    if target_candidate is not None and stop_candidate is not None and len(candidates) < max_candidates:
        combo = dict(base_params)
        for key in ("target_rr", "target_window_mult", "stop_r"):
            if key in target_candidate:
                combo[key] = target_candidate.get(key)
            if key in stop_candidate:
                combo[key] = stop_candidate.get(key)
        combo_reason = "+".join([p for p in [target_reason, stop_reason] if p])
        _add_candidate(combo, combo_reason or "combined_auto_tuning")

    meta["candidate_count"] = len(candidates)
    if not candidates:
        meta["reason"] = "no_adjustment_triggered"
    return candidates, meta


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
    selected_entry_time = str(lite.selected_entry_time_et or "")
    if selected_entry_time:
        acc.selected_entry_time_counts[selected_entry_time] = int(
            acc.selected_entry_time_counts.get(selected_entry_time) or 0
        ) + 1
    acc.sum_r += float(lite.r_multiple)
    r = float(lite.r_multiple)
    acc.sum_r2 += r * r
    acc.sum_pnl_pct += float(lite.pnl_pct)
    if lite.target_r is not None:
        try:
            target_r_val = float(lite.target_r)
            if target_r_val > 0:
                acc.target_r_sum += target_r_val
                acc.target_r_count += 1
        except Exception:
            pass
    if r > 0:
        acc.wins += 1
        acc.gross_profit_r += r
    elif r < 0:
        acc.gross_loss_r += abs(r)
    exit_reason = str(lite.exit_reason or "").lower()
    target_r = max(0.0, float(lite.target_r or 0.0))
    near_target_thresh = (0.75 * target_r) if target_r > 0 else 0.75
    if exit_reason == "stop":
        acc.stop_count += 1
        if lite.mfe_r_before_stop is not None:
            mfe_val = float(lite.mfe_r_before_stop)
            if mfe_val <= 1e-9:
                acc.stop_no_progress_count += 1
            if near_target_thresh > 0 and mfe_val >= near_target_thresh:
                acc.stop_near_target_count += 1
            if mfe_val > 0:
                acc.stop_flipable_any_count += 1
            if mfe_val >= 0.50:
                acc.stop_flipable_050_count += 1
            if mfe_val >= 1.00:
                acc.stop_flipable_100_count += 1
        if lite.target_hit and lite.mae_r_to_target is not None and float(lite.mae_r_to_target) < 0:
            needed_mult = -float(lite.mae_r_to_target)
            acc.stop_reach_target_with_wider_stop_count += 1
            if needed_mult > 0:
                acc.stop_mult_needed_sum += needed_mult
                acc.stop_mult_needed_count += 1
    elif exit_reason == "target":
        acc.target_count += 1
    elif exit_reason in _CUTOFF_EXIT_REASONS:
        acc.cutoff_count += 1
        if lite.mfe_r_full is not None:
            cutoff_mfe = float(lite.mfe_r_full)
            acc.cutoff_mfe_r_sum += cutoff_mfe
            acc.cutoff_mfe_r_count += 1
            if cutoff_mfe <= 1e-9:
                acc.cutoff_no_progress_count += 1
            if near_target_thresh > 0 and cutoff_mfe >= near_target_thresh:
                acc.cutoff_near_target_count += 1
        if lite.mae_r_full is not None:
            cutoff_abs_mae = abs(float(lite.mae_r_full))
            acc.cutoff_abs_mae_r_sum += cutoff_abs_mae
            acc.cutoff_abs_mae_r_count += 1
    month = _month_key(signal_date)
    if month:
        acc.month_pnl_pct[month] = float(acc.month_pnl_pct.get(month) or 0.0) + float(lite.pnl_pct)
        acc.month_trades[month] = int(acc.month_trades.get(month) or 0) + 1


def _acc_sub(acc: CandidateAcc, lite: TradeLite, signal_date: str) -> None:
    acc.trades_count -= 1
    selected_entry_time = str(lite.selected_entry_time_et or "")
    if selected_entry_time:
        next_n = int(acc.selected_entry_time_counts.get(selected_entry_time) or 0) - 1
        if next_n <= 0:
            acc.selected_entry_time_counts.pop(selected_entry_time, None)
        else:
            acc.selected_entry_time_counts[selected_entry_time] = next_n
    acc.sum_r -= float(lite.r_multiple)
    r = float(lite.r_multiple)
    acc.sum_r2 -= r * r
    acc.sum_pnl_pct -= float(lite.pnl_pct)
    if lite.target_r is not None:
        try:
            target_r_val = float(lite.target_r)
            if target_r_val > 0:
                acc.target_r_sum -= target_r_val
                acc.target_r_count -= 1
        except Exception:
            pass
    if r > 0:
        acc.wins -= 1
        acc.gross_profit_r -= r
    elif r < 0:
        acc.gross_loss_r -= abs(r)
    exit_reason = str(lite.exit_reason or "").lower()
    target_r = max(0.0, float(lite.target_r or 0.0))
    near_target_thresh = (0.75 * target_r) if target_r > 0 else 0.75
    if exit_reason == "stop":
        acc.stop_count -= 1
        if lite.mfe_r_before_stop is not None:
            mfe_val = float(lite.mfe_r_before_stop)
            if mfe_val <= 1e-9:
                acc.stop_no_progress_count -= 1
            if near_target_thresh > 0 and mfe_val >= near_target_thresh:
                acc.stop_near_target_count -= 1
            if mfe_val > 0:
                acc.stop_flipable_any_count -= 1
            if mfe_val >= 0.50:
                acc.stop_flipable_050_count -= 1
            if mfe_val >= 1.00:
                acc.stop_flipable_100_count -= 1
        if lite.target_hit and lite.mae_r_to_target is not None and float(lite.mae_r_to_target) < 0:
            needed_mult = -float(lite.mae_r_to_target)
            acc.stop_reach_target_with_wider_stop_count -= 1
            if needed_mult > 0:
                acc.stop_mult_needed_sum -= needed_mult
                acc.stop_mult_needed_count -= 1
    elif exit_reason == "target":
        acc.target_count -= 1
    elif exit_reason in _CUTOFF_EXIT_REASONS:
        acc.cutoff_count -= 1
        if lite.mfe_r_full is not None:
            cutoff_mfe = float(lite.mfe_r_full)
            acc.cutoff_mfe_r_sum -= cutoff_mfe
            acc.cutoff_mfe_r_count -= 1
            if cutoff_mfe <= 1e-9:
                acc.cutoff_no_progress_count -= 1
            if near_target_thresh > 0 and cutoff_mfe >= near_target_thresh:
                acc.cutoff_near_target_count -= 1
        if lite.mae_r_full is not None:
            cutoff_abs_mae = abs(float(lite.mae_r_full))
            acc.cutoff_abs_mae_r_sum -= cutoff_abs_mae
            acc.cutoff_abs_mae_r_count -= 1
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
            "stop_count": 0,
            "target_count": 0,
            "cutoff_count": 0,
            "stop_rate": 0.0,
            "target_rate": 0.0,
            "cutoff_rate": 0.0,
            "cutoff_avg_mfe_r": 0.0,
            "cutoff_avg_abs_mae_r": 0.0,
            "avg_target_r": 0.0,
            "cutoff_target_fit_ratio": 0.0,
            "stop_flip_share_any": 0.0,
            "stop_flip_share_050": 0.0,
            "stop_flip_share_100": 0.0,
            "stop_no_progress_share": 0.0,
            "stop_near_target_share": 0.0,
            "cutoff_no_progress_share": 0.0,
            "cutoff_near_target_share": 0.0,
            "stop_reach_target_with_wider_stop_share": 0.0,
            "avg_stop_mult_needed_for_target": 0.0,
            "loser_count": 0,
            "loser_no_progress_share": 0.0,
            "loser_near_target_share": 0.0,
            "months_count": 0,
            "positive_months": 0,
            "negative_months": 0,
            "positive_month_rate": 0.0,
            "worst_month_pnl_pct": 0.0,
            "max_monthly_drawdown_pct": 0.0,
            "longest_negative_month_streak": 0,
            "scan_selected_entry_time_counts": {},
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
    stop_count = max(0, int(acc.stop_count))
    target_count = max(0, int(acc.target_count))
    cutoff_count = max(0, int(acc.cutoff_count))
    stop_rate = (stop_count / float(acc.trades_count)) if acc.trades_count > 0 else 0.0
    target_rate = (target_count / float(acc.trades_count)) if acc.trades_count > 0 else 0.0
    cutoff_rate = (cutoff_count / float(acc.trades_count)) if acc.trades_count > 0 else 0.0
    stop_flip_share_any = (max(0, int(acc.stop_flipable_any_count)) / float(stop_count)) if stop_count > 0 else 0.0
    stop_flip_share_050 = (max(0, int(acc.stop_flipable_050_count)) / float(stop_count)) if stop_count > 0 else 0.0
    stop_flip_share_100 = (max(0, int(acc.stop_flipable_100_count)) / float(stop_count)) if stop_count > 0 else 0.0
    stop_no_progress_share = (max(0, int(acc.stop_no_progress_count)) / float(stop_count)) if stop_count > 0 else 0.0
    stop_near_target_share = (max(0, int(acc.stop_near_target_count)) / float(stop_count)) if stop_count > 0 else 0.0
    cutoff_no_progress_share = (max(0, int(acc.cutoff_no_progress_count)) / float(cutoff_count)) if cutoff_count > 0 else 0.0
    cutoff_near_target_share = (max(0, int(acc.cutoff_near_target_count)) / float(cutoff_count)) if cutoff_count > 0 else 0.0
    cutoff_avg_mfe_r = (
        float(acc.cutoff_mfe_r_sum) / float(max(1, int(acc.cutoff_mfe_r_count)))
    ) if int(acc.cutoff_mfe_r_count) > 0 else 0.0
    cutoff_avg_abs_mae_r = (
        float(acc.cutoff_abs_mae_r_sum) / float(max(1, int(acc.cutoff_abs_mae_r_count)))
    ) if int(acc.cutoff_abs_mae_r_count) > 0 else 0.0
    avg_target_r = (
        float(acc.target_r_sum) / float(max(1, int(acc.target_r_count)))
    ) if int(acc.target_r_count) > 0 else 0.0
    cutoff_target_fit_ratio = (cutoff_avg_mfe_r / avg_target_r) if avg_target_r > 0 else 0.0
    stop_reach_target_with_wider_stop_share = (
        max(0, int(acc.stop_reach_target_with_wider_stop_count)) / float(stop_count)
    ) if stop_count > 0 else 0.0
    avg_stop_mult_needed_for_target = (
        float(acc.stop_mult_needed_sum) / float(max(1, int(acc.stop_mult_needed_count)))
    ) if int(acc.stop_mult_needed_count) > 0 else 0.0
    loser_count = max(0, int(acc.trades_count) - int(acc.wins))
    loser_no_progress_share = (
        float(max(0, int(acc.stop_no_progress_count) + int(acc.cutoff_no_progress_count))) / float(loser_count)
    ) if loser_count > 0 else 0.0
    loser_near_target_share = (
        float(max(0, int(acc.stop_near_target_count) + int(acc.cutoff_near_target_count))) / float(loser_count)
    ) if loser_count > 0 else 0.0
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
        "stop_count": stop_count,
        "target_count": target_count,
        "cutoff_count": cutoff_count,
        "stop_rate": stop_rate,
        "target_rate": target_rate,
        "cutoff_rate": cutoff_rate,
        "cutoff_avg_mfe_r": cutoff_avg_mfe_r,
        "cutoff_avg_abs_mae_r": cutoff_avg_abs_mae_r,
        "avg_target_r": avg_target_r,
        "cutoff_target_fit_ratio": cutoff_target_fit_ratio,
        "stop_flip_share_any": stop_flip_share_any,
        "stop_flip_share_050": stop_flip_share_050,
        "stop_flip_share_100": stop_flip_share_100,
        "stop_no_progress_share": stop_no_progress_share,
        "stop_near_target_share": stop_near_target_share,
        "cutoff_no_progress_share": cutoff_no_progress_share,
        "cutoff_near_target_share": cutoff_near_target_share,
        "stop_reach_target_with_wider_stop_share": stop_reach_target_with_wider_stop_share,
        "avg_stop_mult_needed_for_target": avg_stop_mult_needed_for_target,
        "loser_count": loser_count,
        "loser_no_progress_share": loser_no_progress_share,
        "loser_near_target_share": loser_near_target_share,
        "months_count": months_count,
        "positive_months": positive_months,
        "negative_months": negative_months,
        "positive_month_rate": positive_month_rate,
        "worst_month_pnl_pct": worst_month_pnl_pct,
        "max_monthly_drawdown_pct": max_monthly_drawdown_pct,
        "longest_negative_month_streak": _longest_negative_month_streak(month_pnls),
        "scan_selected_entry_time_counts": dict(sorted((acc.selected_entry_time_counts or {}).items())),
    }


def _recent_candidate_stats(
    *,
    state: SymbolRollingState,
    signals_by_date: Dict[str, object],
    trade_ns: str,
    symbol: str,
    entry_time: str,
    overrides: Dict[str, Any],
    end_date: str,
    lookback_days: int,
) -> Dict[str, float]:
    if lookback_days <= 0 or not state.window_signal_dates:
        return _stats_from_acc(CandidateAcc())
    end_dt = ensure_date(end_date)
    start_dt = end_dt - dt.timedelta(days=max(0, lookback_days - 1))
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()
    sym = str(symbol).upper()
    overrides_key = _overrides_key(overrides or {})
    acc = CandidateAcc()
    for signal_date in sorted(state.window_signal_dates):
        if signal_date < start_str or signal_date > end_str:
            continue
        sig = signals_by_date.get(signal_date)
        if sig is None:
            continue
        direction = str(getattr(sig, "direction", "") or "").lower()
        cache_key = (str(trade_ns), sym, signal_date, direction, str(entry_time), overrides_key)
        lite = _TRADE_LITE_CACHE.get(cache_key)
        if lite is None:
            continue
        _acc_add(acc, lite, signal_date)
    return _stats_from_acc(acc)


def _add_minutes(time_str: str, minutes: int) -> str:
    if not time_str or minutes <= 0:
        return time_str
    try:
        base_time = parse_time_hhmm(time_str)
        shifted = dt.datetime.combine(dt.date.today(), base_time) + dt.timedelta(minutes=minutes)
        return shifted.strftime("%H:%M")
    except Exception:
        return time_str


def _flatten_minutes_from_open(params: Dict[str, Any], *, session_open_et: str) -> Optional[int]:
    try:
        if not bool(params.get("intraday_only", False)):
            return None
        session_close_et = str(params.get("session_close_et") or "16:00")
        flatten_buffer = int(params.get("flatten_buffer_minutes") or 0)
        open_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_open_et))
        close_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_close_et))
        flatten_dt = close_dt - dt.timedelta(minutes=max(0, flatten_buffer))
        flatten_minutes = int((flatten_dt - open_dt).total_seconds() / 60)
        return max(1, flatten_minutes)
    except Exception:
        return None


def _candidate_intraday_requirements(
    strategy_params: Dict[str, Any],
    overrides: Dict[str, Any],
    *,
    entry_time_et: str,
    context: str,
    direction: str,
) -> Dict[str, Any]:
    params = dict(strategy_params or {})
    if isinstance(overrides, dict) and overrides:
        params.update(overrides)

    session_open_et = str(params.get("session_open_et") or "09:30")
    intraday_filter_enabled = bool(params.get("intraday_filter_enabled", False))
    intraday_filter_apply_watchlist = bool(params.get("intraday_filter_apply_in_watchlist", True))
    apply_early_filter = intraday_filter_enabled and (context != "watchlist" or intraday_filter_apply_watchlist)
    early_range_minutes = int(params.get("early_range_minutes") or 0) if intraday_filter_enabled else 0
    max_early_pullback_bps = float(params.get("max_early_pullback_bps") or 0.0)
    min_early_reversal_bps = float(params.get("min_early_reversal_bps") or 0.0)
    try:
        min_early_reversal_bps_long = (
            float(params["min_early_reversal_bps_long"])
            if params.get("min_early_reversal_bps_long") is not None
            else None
        )
    except Exception:
        min_early_reversal_bps_long = None
    try:
        min_early_reversal_bps_short = (
            float(params["min_early_reversal_bps_short"])
            if params.get("min_early_reversal_bps_short") is not None
            else None
        )
    except Exception:
        min_early_reversal_bps_short = None
    if min_early_reversal_bps_long is None:
        min_early_reversal_bps_long = min_early_reversal_bps
    if min_early_reversal_bps_short is None:
        min_early_reversal_bps_short = min_early_reversal_bps
    min_early_reversal_dir = min_early_reversal_bps_long if str(direction).lower() == "long" else min_early_reversal_bps_short
    requires_early_data = (
        apply_early_filter
        and early_range_minutes > 0
        and (
            max_early_pullback_bps > 0
            or (min_early_reversal_dir is not None and float(min_early_reversal_dir) > 0)
        )
    )

    use_intraday_entry = bool(params.get("use_intraday_entry", False))
    intraday_entry_in_watchlist = bool(params.get("intraday_entry_in_watchlist", False))
    apply_intraday_entry = use_intraday_entry and (context != "watchlist" or intraday_entry_in_watchlist)

    confirm_move_bps = float(params.get("confirm_move_bps") or 0.0)
    confirm_minutes = int(params.get("confirm_minutes") or 0)
    confirm_apply_in_watchlist = bool(params.get("confirm_apply_in_watchlist", True))
    apply_confirm = confirm_move_bps > 0 and confirm_minutes > 0 and (context != "watchlist" or confirm_apply_in_watchlist)
    confirm_pad = confirm_minutes if apply_confirm else 0

    time_stop_minutes = int(params.get("time_stop_minutes") or 0)
    intraday_only = bool(params.get("intraday_only", False))
    flatten_minutes = _flatten_minutes_from_open(params, session_open_et=session_open_et)

    minutes_needed = 0
    entry_minutes_raw = 0
    try:
        open_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(session_open_et))
        entry_dt = dt.datetime.combine(dt.date.today(), parse_time_hhmm(str(entry_time_et or "")))
        entry_minutes_raw = int((entry_dt - open_dt).total_seconds() / 60)
        entry_minutes_raw = max(0, entry_minutes_raw)
    except Exception:
        entry_minutes_raw = 0

    if requires_early_data:
        minutes_needed = max(minutes_needed, early_range_minutes)
    if apply_intraday_entry:
        minutes_needed = max(minutes_needed, max(1, entry_minutes_raw + 1))
    if apply_confirm:
        minutes_needed = max(minutes_needed, max(1, entry_minutes_raw + confirm_pad + 1))

    cutoff_minutes: Optional[int] = None
    if time_stop_minutes > 0:
        cutoff_minutes = entry_minutes_raw + confirm_pad + time_stop_minutes
    if intraday_only and flatten_minutes is not None:
        cutoff_minutes = flatten_minutes if cutoff_minutes is None else min(cutoff_minutes, flatten_minutes)
    if cutoff_minutes is not None and cutoff_minutes > 0:
        minutes_needed = max(minutes_needed, cutoff_minutes + 1)

    entry_cutoff_time = _add_minutes(str(entry_time_et or ""), confirm_pad) if apply_confirm else str(entry_time_et or "")
    return {
        "minutes_needed": int(max(0, minutes_needed)),
        "entry_cutoff_time": str(entry_cutoff_time or ""),
    }


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
    scan_entry_times: Optional[List[str]] = None,
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
    strategy_params = cfg.get("daily_trend_reversal") or {}
    candidate_requirements: Dict[Tuple[str, int], Dict[str, Any]] = {}
    minutes_needed = int(max(0, minutes_needed_base))
    scan_times_clean: List[str] = []
    if isinstance(scan_entry_times, list):
        scan_times_clean = [str(t) for t in scan_entry_times if t]
    is_scan_mode = bool(scan_times_clean)

    def _latest_cutoff_time(reqs: List[Dict[str, Any]]) -> str:
        if not reqs:
            return ""
        times = [str(r.get("entry_cutoff_time") or "") for r in reqs if str(r.get("entry_cutoff_time") or "")]
        if not times:
            return ""
        try:
            return sorted(times, key=lambda t: parse_time_hhmm(t))[-1]
        except Exception:
            return times[-1]

    def _scan_times_for_entry(entry_time_label: str) -> List[str]:
        if not is_scan_mode:
            return []
        if entry_time_label == _SCAN_FIRST_VALID_LABEL:
            return list(scan_times_clean)
        et = str(entry_time_label or "")
        if not et:
            return list(scan_times_clean)
        if et in scan_times_clean:
            idx = scan_times_clean.index(et)
            return list(scan_times_clean[idx:])
        try:
            et_parsed = parse_time_hhmm(et)
            filtered = [t for t in scan_times_clean if parse_time_hhmm(t) >= et_parsed]
            return filtered if filtered else list(scan_times_clean)
        except Exception:
            return list(scan_times_clean)

    for entry_time in entry_times:
        for grid_idx, overrides in enumerate(param_grid):
            cache_key = _trade_cache_key(trade_ns, sym, signal_date, direction, entry_time, overrides)
            if cache_key in _TRADE_LITE_CACHE:
                cached_by_candidate[(entry_time, grid_idx)] = _TRADE_LITE_CACHE[cache_key]
            else:
                any_missing = True
                scan_times = _scan_times_for_entry(entry_time)
                if scan_times:
                    reqs = [
                        _candidate_intraday_requirements(
                            strategy_params,
                            overrides,
                            entry_time_et=scan_time,
                            context="watchlist",
                            direction=direction,
                        )
                        for scan_time in scan_times
                    ]
                    req = {
                        "minutes_needed": max((int(r.get("minutes_needed") or 0) for r in reqs), default=0),
                        "entry_cutoff_time": _latest_cutoff_time(reqs),
                        "scan_mode": True,
                        "scan_times": list(scan_times),
                    }
                else:
                    req = _candidate_intraday_requirements(
                        strategy_params,
                        overrides,
                        entry_time_et=entry_time,
                        context="watchlist",
                        direction=direction,
                    )
                candidate_requirements[(entry_time, grid_idx)] = req
                minutes_needed = max(minutes_needed, int(req.get("minutes_needed") or 0))
    if not any_missing:
        return cached_by_candidate

    bars_intraday = None
    if minutes_needed > 0:
        bars_intraday = get_intraday_bars(sym, signal_date, minutes_needed, cfg=cfg, allow_fetch=True)
        if not bars_intraday:
            # Preserve existing semantics: treat as "no trade" for this signal (do not cache),
            # allowing future attempts if data becomes available.
            return cached_by_candidate

    intraday_entry_cache: Dict[str, List[dict]] = {}
    for entry_time in entry_times:
        for grid_idx, overrides in enumerate(param_grid):
            cache_key = _trade_cache_key(trade_ns, sym, signal_date, direction, entry_time, overrides)
            if cache_key in _TRADE_LITE_CACHE:
                cached_by_candidate[(entry_time, grid_idx)] = _TRADE_LITE_CACHE[cache_key]
                continue
            req = candidate_requirements.get((entry_time, grid_idx))
            if req is None:
                scan_times = _scan_times_for_entry(entry_time)
                if scan_times:
                    reqs = [
                        _candidate_intraday_requirements(
                            strategy_params,
                            overrides,
                            entry_time_et=scan_time,
                            context="watchlist",
                            direction=direction,
                        )
                        for scan_time in scan_times
                    ]
                    req = {
                        "minutes_needed": max((int(r.get("minutes_needed") or 0) for r in reqs), default=0),
                        "entry_cutoff_time": _latest_cutoff_time(reqs),
                        "scan_mode": True,
                        "scan_times": list(scan_times),
                    }
                else:
                    req = _candidate_intraday_requirements(
                        strategy_params,
                        overrides,
                        entry_time_et=entry_time,
                        context="watchlist",
                        direction=direction,
                    )
                candidate_requirements[(entry_time, grid_idx)] = req
            cutoff_time = str(req.get("entry_cutoff_time") or entry_time or "")
            bars_intraday_entry = bars_intraday
            if bars_intraday and cutoff_time:
                if cutoff_time not in intraday_entry_cache:
                    intraday_entry_cache[cutoff_time] = filter_intraday_bars_until(
                        bars_intraday,
                        signal_date,
                        cutoff_time,
                    )
                bars_intraday_entry = intraday_entry_cache.get(cutoff_time) or []
            plan = None
            if bool(req.get("scan_mode")) and is_scan_mode:
                req_scan_times = req.get("scan_times")
                if not isinstance(req_scan_times, list) or not req_scan_times:
                    req_scan_times = _scan_times_for_entry(entry_time)
                for scan_time in req_scan_times:
                    plan = build_trade(
                        signal,
                        cfg,
                        data_store,
                        context="watchlist",
                        bars_intraday=bars_intraday_entry,
                        entry_time_override=scan_time,
                        param_overrides=overrides,
                    )
                    if plan:
                        break
            else:
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
            lite = TradeLite(
                pnl_pct=pnl_pct,
                r_multiple=r_multiple,
                exit_reason=str(exit_info.get("exit_reason") or ""),
                selected_entry_time_et=str(getattr(plan, "entry_time_et", "") or ""),
                mfe_r_before_stop=(
                    None if exit_info.get("mfe_r_before_stop") is None else float(exit_info.get("mfe_r_before_stop"))
                ),
                mae_r_to_target=(
                    None if exit_info.get("mae_r_to_target") is None else float(exit_info.get("mae_r_to_target"))
                ),
                mfe_r_full=(None if exit_info.get("mfe_r_full") is None else float(exit_info.get("mfe_r_full"))),
                mae_r_full=(None if exit_info.get("mae_r_full") is None else float(exit_info.get("mae_r_full"))),
                target_r=(
                    (abs(float(getattr(plan, "target_price", 0.0)) - float(getattr(plan, "entry_price", 0.0))) / float(getattr(plan, "stop_distance", 0.0)))
                    if float(getattr(plan, "stop_distance", 0.0)) > 0
                    else None
                ),
                target_hit=bool(exit_info.get("target_hit_ts") is not None),
            )
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
    try:
        max_cutoff_rate_raw = watch_cfg.get("maxCutoffRate")
        if max_cutoff_rate_raw is None:
            max_cutoff_rate_raw = watch_cfg.get("max_cutoff_rate")
        max_cutoff_rate = float(max_cutoff_rate_raw) if max_cutoff_rate_raw is not None else None
    except Exception:
        max_cutoff_rate = None
    try:
        max_stop_rate_raw = watch_cfg.get("maxStopRate")
        if max_stop_rate_raw is None:
            max_stop_rate_raw = watch_cfg.get("max_stop_rate")
        max_stop_rate = float(max_stop_rate_raw) if max_stop_rate_raw is not None else None
    except Exception:
        max_stop_rate = None
    try:
        min_cutoff_avg_mfe_r_raw = watch_cfg.get("min_cutoff_avg_mfe_r")
        min_cutoff_avg_mfe_r = (
            float(min_cutoff_avg_mfe_r_raw) if min_cutoff_avg_mfe_r_raw is not None else None
        )
    except Exception:
        min_cutoff_avg_mfe_r = None
    try:
        min_cutoff_target_fit_ratio_raw = watch_cfg.get("min_cutoff_target_fit_ratio")
        min_cutoff_target_fit_ratio = (
            float(min_cutoff_target_fit_ratio_raw) if min_cutoff_target_fit_ratio_raw is not None else None
        )
    except Exception:
        min_cutoff_target_fit_ratio = None
    try:
        max_loser_no_progress_share_raw = watch_cfg.get("max_loser_no_progress_share")
        max_loser_no_progress_share = (
            float(max_loser_no_progress_share_raw) if max_loser_no_progress_share_raw is not None else None
        )
    except Exception:
        max_loser_no_progress_share = None
    try:
        max_stop_no_progress_share_raw = watch_cfg.get("max_stop_no_progress_share")
        max_stop_no_progress_share = (
            float(max_stop_no_progress_share_raw) if max_stop_no_progress_share_raw is not None else None
        )
    except Exception:
        max_stop_no_progress_share = None
    top_k = int(watch_cfg.get("top_k") or 0)
    top_k_rank_by = str(watch_cfg.get("top_k_rank_by") or "total_pnl_pct").lower()
    directional_history_only = bool(watch_cfg.get("directional_history_only", False))
    report_enabled = bool(watch_cfg.get("report_enabled", False) or cfg.get("watchlist_report_enabled", False))
    try:
        rank_lcb_z = float(watch_cfg.get("rank_lcb_z") or 1.0)
    except Exception:
        rank_lcb_z = 1.0
    try:
        quality_score_trades_ref = max(1, int(watch_cfg.get("entry_quality_score_trades_ref") or 20))
    except Exception:
        quality_score_trades_ref = 20
    try:
        quality_score_weight_lcb = float(watch_cfg.get("entry_quality_score_weight_lcb") or 1.0)
    except Exception:
        quality_score_weight_lcb = 1.0
    try:
        quality_score_weight_target_rate = float(watch_cfg.get("entry_quality_score_weight_target_rate") or 0.20)
    except Exception:
        quality_score_weight_target_rate = 0.20
    try:
        quality_score_weight_stop_rate = float(watch_cfg.get("entry_quality_score_weight_stop_rate") or 0.12)
    except Exception:
        quality_score_weight_stop_rate = 0.12
    try:
        quality_score_weight_cutoff_rate = float(watch_cfg.get("entry_quality_score_weight_cutoff_rate") or 0.08)
    except Exception:
        quality_score_weight_cutoff_rate = 0.08
    try:
        quality_score_weight_stop_no_progress = float(
            watch_cfg.get("entry_quality_score_weight_stop_no_progress") or 0.25
        )
    except Exception:
        quality_score_weight_stop_no_progress = 0.25
    try:
        quality_score_weight_cutoff_no_progress = float(
            watch_cfg.get("entry_quality_score_weight_cutoff_no_progress") or 0.12
        )
    except Exception:
        quality_score_weight_cutoff_no_progress = 0.12
    try:
        quality_score_weight_loser_no_progress = float(
            watch_cfg.get("entry_quality_score_weight_loser_no_progress") or 0.10
        )
    except Exception:
        quality_score_weight_loser_no_progress = 0.10
    try:
        quality_score_min_target_fit_ratio = max(
            0.0,
            float(watch_cfg.get("entry_quality_score_min_target_fit_ratio") or 0.80),
        )
    except Exception:
        quality_score_min_target_fit_ratio = 0.80
    try:
        quality_score_weight_target_fit_shortfall = float(
            watch_cfg.get("entry_quality_score_weight_target_fit_shortfall") or 0.20
        )
    except Exception:
        quality_score_weight_target_fit_shortfall = 0.20
    try:
        scan_start_min_selected_trades = int(watch_cfg.get("scan_start_min_selected_trades") or 0)
    except Exception:
        scan_start_min_selected_trades = 0
    scan_start_min_selected_trades = max(0, scan_start_min_selected_trades)
    try:
        scan_start_min_selected_share = float(watch_cfg.get("scan_start_min_selected_share") or 0.0)
    except Exception:
        scan_start_min_selected_share = 0.0
    scan_start_min_selected_share = max(0.0, min(1.0, scan_start_min_selected_share))
    scan_block_selected_entry_times: set[str] = set()
    raw_blocked_entry_times = watch_cfg.get("scan_block_selected_entry_times")
    if isinstance(raw_blocked_entry_times, list):
        for t in raw_blocked_entry_times:
            ts = str(t or "").strip()
            if ts:
                scan_block_selected_entry_times.add(ts)
    elif raw_blocked_entry_times:
        ts = str(raw_blocked_entry_times).strip()
        if ts:
            scan_block_selected_entry_times.add(ts)
    try:
        scan_max_blocked_selected_share = float(watch_cfg.get("scan_max_blocked_selected_share") or 0.0)
    except Exception:
        scan_max_blocked_selected_share = 0.0
    scan_max_blocked_selected_share = max(0.0, min(1.0, scan_max_blocked_selected_share))
    param_grid = _expand_param_grid(watch_cfg.get("param_grid") or {})
    # Optional explicit per-symbol parameter overrides (e.g., target/stop knobs for specific symbols).
    # These are applied in the watchlist builder so ranking/filtering uses the same params replay will execute.
    symbol_param_overrides = _load_symbol_param_overrides(watch_cfg)
    symbol_param_override_mode = str(watch_cfg.get("symbol_param_override_mode") or "merge").lower().strip()
    if symbol_param_override_mode not in {"merge", "replace"}:
        symbol_param_override_mode = "merge"
    auto_symbol_tuning_cfg = watch_cfg.get("auto_symbol_tuning") or {}
    auto_symbol_tuning_enabled = bool(auto_symbol_tuning_cfg.get("enabled", False))
    auto_symbol_tuning_allow_with_symbol_override = bool(
        auto_symbol_tuning_cfg.get("allow_with_symbol_override", False)
    )
    auto_symbol_tuning_require_improvement = bool(auto_symbol_tuning_cfg.get("require_improvement", True))
    try:
        auto_symbol_tuning_min_score_improvement = float(auto_symbol_tuning_cfg.get("min_score_improvement") or 0.0)
    except Exception:
        auto_symbol_tuning_min_score_improvement = 0.0
    dynamic_wf_cfg = watch_cfg.get("dynamic_walk_forward") or {}
    dynamic_wf_enabled = bool(dynamic_wf_cfg.get("enabled", False))
    try:
        dynamic_wf_lookback_days = int(dynamic_wf_cfg.get("lookback_days") or 63)
    except Exception:
        dynamic_wf_lookback_days = 63
    dynamic_wf_lookback_days = max(1, dynamic_wf_lookback_days)
    try:
        dynamic_wf_min_trades = int(dynamic_wf_cfg.get("min_trades") or 0)
    except Exception:
        dynamic_wf_min_trades = 0
    dynamic_wf_skip_if_insufficient_trades = bool(
        dynamic_wf_cfg.get("skip_if_insufficient_trades", False)
    )
    try:
        dynamic_wf_min_total_pnl_pct_raw = dynamic_wf_cfg.get("min_total_pnl_pct")
        dynamic_wf_min_total_pnl_pct = (
            float(dynamic_wf_min_total_pnl_pct_raw) if dynamic_wf_min_total_pnl_pct_raw is not None else None
        )
    except Exception:
        dynamic_wf_min_total_pnl_pct = None
    try:
        dynamic_wf_min_avg_r_raw = dynamic_wf_cfg.get("min_avg_r")
        dynamic_wf_min_avg_r = float(dynamic_wf_min_avg_r_raw) if dynamic_wf_min_avg_r_raw is not None else None
    except Exception:
        dynamic_wf_min_avg_r = None
    try:
        dynamic_wf_min_win_rate_raw = dynamic_wf_cfg.get("min_win_rate")
        dynamic_wf_min_win_rate = (
            float(dynamic_wf_min_win_rate_raw) if dynamic_wf_min_win_rate_raw is not None else None
        )
    except Exception:
        dynamic_wf_min_win_rate = None
    try:
        dynamic_wf_max_stop_rate_raw = dynamic_wf_cfg.get("max_stop_rate")
        dynamic_wf_max_stop_rate = (
            float(dynamic_wf_max_stop_rate_raw) if dynamic_wf_max_stop_rate_raw is not None else None
        )
    except Exception:
        dynamic_wf_max_stop_rate = None
    try:
        dynamic_wf_max_cutoff_no_progress_raw = dynamic_wf_cfg.get("max_cutoff_no_progress_share")
        dynamic_wf_max_cutoff_no_progress_share = (
            float(dynamic_wf_max_cutoff_no_progress_raw)
            if dynamic_wf_max_cutoff_no_progress_raw is not None
            else None
        )
    except Exception:
        dynamic_wf_max_cutoff_no_progress_share = None
    try:
        dynamic_wf_max_loser_no_progress_raw = dynamic_wf_cfg.get("max_loser_no_progress_share")
        dynamic_wf_max_loser_no_progress_share = (
            float(dynamic_wf_max_loser_no_progress_raw)
            if dynamic_wf_max_loser_no_progress_raw is not None
            else None
        )
    except Exception:
        dynamic_wf_max_loser_no_progress_share = None
    try:
        dynamic_wf_max_stop_no_progress_raw = dynamic_wf_cfg.get("max_stop_no_progress_share")
        dynamic_wf_max_stop_no_progress_share = (
            float(dynamic_wf_max_stop_no_progress_raw)
            if dynamic_wf_max_stop_no_progress_raw is not None
            else None
        )
    except Exception:
        dynamic_wf_max_stop_no_progress_share = None
    try:
        dynamic_wf_max_cutoff_rate_raw = dynamic_wf_cfg.get("max_cutoff_rate")
        dynamic_wf_max_cutoff_rate = (
            float(dynamic_wf_max_cutoff_rate_raw) if dynamic_wf_max_cutoff_rate_raw is not None else None
        )
    except Exception:
        dynamic_wf_max_cutoff_rate = None
    try:
        dynamic_wf_min_cutoff_avg_mfe_r_raw = dynamic_wf_cfg.get("min_cutoff_avg_mfe_r")
        dynamic_wf_min_cutoff_avg_mfe_r = (
            float(dynamic_wf_min_cutoff_avg_mfe_r_raw)
            if dynamic_wf_min_cutoff_avg_mfe_r_raw is not None
            else None
        )
    except Exception:
        dynamic_wf_min_cutoff_avg_mfe_r = None
    try:
        dynamic_wf_min_cutoff_target_fit_ratio_raw = dynamic_wf_cfg.get("min_cutoff_target_fit_ratio")
        dynamic_wf_min_cutoff_target_fit_ratio = (
            float(dynamic_wf_min_cutoff_target_fit_ratio_raw)
            if dynamic_wf_min_cutoff_target_fit_ratio_raw is not None
            else None
        )
    except Exception:
        dynamic_wf_min_cutoff_target_fit_ratio = None
    dynamic_wf_log_details = bool(dynamic_wf_cfg.get("log_details", False))
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
        "max_stop_rate": 0,
        "max_cutoff_rate": 0,
        "min_cutoff_avg_mfe_r": 0,
        "min_cutoff_target_fit_ratio": 0,
        "max_loser_no_progress_share": 0,
        "max_stop_no_progress_share": 0,
        "wf_min_trades": 0,
        "wf_min_total_pnl_pct": 0,
        "wf_min_avg_r": 0,
        "wf_min_win_rate": 0,
        "wf_max_stop_rate": 0,
        "wf_max_cutoff_rate": 0,
        "wf_max_cutoff_no_progress_share": 0,
        "wf_max_loser_no_progress_share": 0,
        "wf_max_stop_no_progress_share": 0,
        "wf_min_cutoff_avg_mfe_r": 0,
        "wf_min_cutoff_target_fit_ratio": 0,
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
    entry_times_all = resolve_entry_times(params, sort_mode="preserve")
    entry_time_mode = str(watch_cfg.get("entry_time_mode") or "fixed").lower().strip()
    scan_first_valid_mode = entry_time_mode in {"scan_first_valid", "dynamic_first_valid", "scan"}
    scan_first_valid_candidate_mode = str(
        watch_cfg.get("scan_first_valid_candidate_mode") or "single"
    ).lower().strip()
    scan_first_valid_per_start_mode = scan_first_valid_candidate_mode in {"per_start", "per_start_time", "start_time"}
    entry_time_sort_mode = str(watch_cfg.get("entry_time_sort_mode") or "asc").lower().strip()
    if entry_time_sort_mode in {"asc", "sorted", "time_asc"}:
        try:
            entry_times_all = sorted(entry_times_all, key=lambda t: parse_time_hhmm(t))
        except Exception:
            pass
    elif entry_time_sort_mode in {"desc", "time_desc", "reverse"}:
        try:
            entry_times_all = sorted(entry_times_all, key=lambda t: parse_time_hhmm(t), reverse=True)
        except Exception:
            entry_times_all = list(reversed(entry_times_all))
    if scan_first_valid_mode and scan_first_valid_per_start_mode:
        # Score one candidate per scan start time (e.g., 09:35..09:38), each candidate scans forward
        # from its own start time to the end of entry_times_et.
        entry_times = list(entry_times_all)
    else:
        entry_times = [_SCAN_FIRST_VALID_LABEL] if scan_first_valid_mode else list(entry_times_all)
    entry_times_for_minutes = list(entry_times_all) if scan_first_valid_mode else list(entry_times)
    target_mode = str(params.get("target_mode") or "rr").lower().strip()
    if scan_first_valid_mode and target_mode == "symbol_window_avg":
        try:
            slow_threshold = int(watch_cfg.get("scan_first_valid_slow_time_threshold") or 12)
        except Exception:
            slow_threshold = 12
        slow_threshold = max(1, slow_threshold)
        entry_time_count = len(entry_times_all)
        if entry_time_count >= slow_threshold:
            msg = (
                "[WATCHLIST_PERF] slow_combo entry_time_mode=scan_first_valid "
                f"target_mode=symbol_window_avg entry_times={entry_time_count} "
                f"(threshold={slow_threshold}); this can be very slow. "
                "For fast backtests use target_mode=rr (or reduce entry_times_et)."
            )
            logging.warning(msg)
            if bool(watch_cfg.get("scan_first_valid_fail_on_slow_target_mode", False)):
                raise RuntimeError(msg)
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
        for t in entry_times_for_minutes:
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
    auto_tuning_considered = 0
    auto_tuning_applied = 0
    auto_tuning_skipped_due_symbol_override = 0
    dynamic_wf_considered_symbols = 0
    dynamic_wf_suppressed_symbols = 0
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
                scan_entry_times=(entry_times_all if scan_first_valid_mode else None),
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
        def _score(stats: Dict[str, float]) -> float:
            avg_r = float(stats.get("avgR") or 0.0)
            se = float(stats.get("avgR_stderr") or 0.0)
            lcb = avg_r - (rank_lcb_z * se)
            if param_rank_by in ("total_pnl_pct", "pnl", "total_pnl"):
                return float(stats.get("total_pnl_pct") or 0.0)
            if param_rank_by in ("profit_factor", "pf"):
                return float(stats.get("profit_factor") or 0.0)
            if param_rank_by in ("avgr_lcb", "avgr-lcb", "avgR_lcb", "avgR-lcb", "lcb"):
                return lcb
            if param_rank_by in ("entry_quality", "entry_quality_score", "quality", "quality_score"):
                trades = max(0, int(stats.get("trades_count") or 0))
                target_rate = float(stats.get("target_rate") or 0.0)
                stop_rate = float(stats.get("stop_rate") or 0.0)
                cutoff_rate = float(stats.get("cutoff_rate") or 0.0)
                stop_no_progress_share = float(stats.get("stop_no_progress_share") or 0.0)
                cutoff_no_progress_share = float(stats.get("cutoff_no_progress_share") or 0.0)
                loser_no_progress_share = float(stats.get("loser_no_progress_share") or 0.0)
                target_fit_ratio = float(stats.get("cutoff_target_fit_ratio") or 0.0)
                fit_shortfall = max(0.0, quality_score_min_target_fit_ratio - target_fit_ratio)
                fit_shortfall_norm = (
                    min(1.0, fit_shortfall / max(1e-6, quality_score_min_target_fit_ratio))
                    if quality_score_min_target_fit_ratio > 0
                    else 0.0
                )
                quality_raw = (
                    (quality_score_weight_lcb * lcb)
                    + (quality_score_weight_target_rate * target_rate)
                    - (quality_score_weight_stop_rate * stop_rate)
                    - (quality_score_weight_cutoff_rate * cutoff_rate)
                    - (quality_score_weight_stop_no_progress * stop_no_progress_share)
                    - (quality_score_weight_cutoff_no_progress * cutoff_no_progress_share)
                    - (quality_score_weight_loser_no_progress * loser_no_progress_share)
                    - (quality_score_weight_target_fit_shortfall * fit_shortfall_norm)
                )
                sample_weight = min(1.0, float(trades) / float(max(1, quality_score_trades_ref)))
                # Shrink toward LCB for low-sample candidates to reduce lucky-minute overfitting.
                return (sample_weight * quality_raw) + ((1.0 - sample_weight) * (quality_score_weight_lcb * lcb))
            return avg_r

        def _eligible_candidates(rows: List[Tuple[str, Dict[str, Any], Dict[str, float]]]) -> List[Tuple[str, Dict[str, Any], Dict[str, float]]]:
            if (
                scan_first_valid_mode
                and scan_first_valid_per_start_mode
                and (
                    scan_start_min_selected_trades > 0
                    or scan_start_min_selected_share > 0.0
                    or bool(scan_block_selected_entry_times)
                )
            ):
                scan_filtered: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []
                for row in rows:
                    entry_time, _, stats = row
                    if str(entry_time) == _SCAN_FIRST_VALID_LABEL:
                        scan_filtered.append(row)
                        continue
                    counts = stats.get("scan_selected_entry_time_counts")
                    counts_map = counts if isinstance(counts, dict) else {}
                    selected_count = int(counts_map.get(str(entry_time)) or 0)
                    trades_n = max(0, int(stats.get("trades_count") or 0))
                    if scan_block_selected_entry_times:
                        if str(entry_time) in scan_block_selected_entry_times:
                            continue
                        blocked_selected = sum(int(counts_map.get(t) or 0) for t in scan_block_selected_entry_times)
                        if blocked_selected > 0:
                            if scan_max_blocked_selected_share <= 0.0:
                                continue
                            blocked_share = (float(blocked_selected) / float(trades_n)) if trades_n > 0 else 0.0
                            if blocked_share > scan_max_blocked_selected_share:
                                continue
                    if scan_start_min_selected_trades > 0 and selected_count < scan_start_min_selected_trades:
                        continue
                    if scan_start_min_selected_share > 0.0 and trades_n > 0:
                        selected_share = float(selected_count) / float(trades_n)
                        if selected_share < scan_start_min_selected_share:
                            continue
                    scan_filtered.append(row)
                rows = scan_filtered
            if min_trades <= 0:
                return rows
            with_min_trades = [c for c in rows if int(c[2].get("trades_count") or 0) >= min_trades]
            return with_min_trades if with_min_trades else rows

        def _pick_best_candidate(rows: List[Tuple[str, Dict[str, Any], Dict[str, float]]]) -> Optional[Tuple[str, Dict[str, Any], Dict[str, float]]]:
            if not rows:
                return None
            best_row = rows[0]
            best_score = _score(rows[0][2])
            for row in rows[1:]:
                row_score = _score(row[2])
                if row_score > best_score:
                    best_row = row
                    best_score = row_score
            return best_row

        def _evaluate_forced_params(forced_params: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], Dict[str, float]]]:
            forced_key = _overrides_key(forced_params)
            forced_grid_idx = grid_idx_by_overrides_key.get(forced_key)
            forced_candidates: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []
            if forced_grid_idx is not None:
                for entry_time in entry_times:
                    forced_stats = _stats_from_acc(state.acc_by_candidate.get((entry_time, forced_grid_idx), CandidateAcc()))
                    forced_candidates.append((entry_time, forced_params, forced_stats))
                return forced_candidates

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
                    scan_entry_times=(entry_times_all if scan_first_valid_mode else None),
                )
                for entry_time in entry_times:
                    lite = lites.get((entry_time, 0))
                    if lite is not None:
                        _acc_add(forced_acc_by_time[entry_time], lite, d)
            for entry_time in entry_times:
                forced_candidates.append((entry_time, forced_params, _stats_from_acc(forced_acc_by_time[entry_time])))
            return forced_candidates

        candidates: List[Tuple[str, Dict[str, Any], Dict[str, float]]] = []
        for entry_time in entry_times:
            for grid_idx, overrides in enumerate(param_grid):
                stats = _stats_from_acc(state.acc_by_candidate.get((entry_time, grid_idx), CandidateAcc()))
                candidates.append((entry_time, overrides, stats))
        if not candidates:
            continue

        best_candidate = _pick_best_candidate(_eligible_candidates(candidates))
        if best_candidate is None:
            continue
        best_time, best_params, best_stats = best_candidate
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
            forced_best = _pick_best_candidate(_eligible_candidates(_evaluate_forced_params(forced_params)))
            if forced_best is not None:
                best_time, best_params, best_stats = forced_best
                symbol_overrides_used += 1

        auto_tuning_applied_symbol = False
        auto_tuning_reason = ""
        auto_tuning_score_delta = 0.0
        auto_tuning_candidate_count = 0
        can_apply_auto_tuning = auto_symbol_tuning_enabled and (
            auto_symbol_tuning_allow_with_symbol_override or not bool(symbol_override)
        )
        if can_apply_auto_tuning:
            auto_candidates, auto_meta = _auto_symbol_param_candidates(
                dict(best_params or {}),
                best_stats,
                params,
                auto_symbol_tuning_cfg,
            )
            auto_tuning_candidate_count = int(auto_meta.get("candidate_count") or 0)
            if bool(auto_meta.get("considered")):
                auto_tuning_considered += 1
            if auto_candidates:
                baseline_score = _score(best_stats)
                best_auto_candidate: Optional[Tuple[str, Dict[str, Any], Dict[str, float]]] = None
                best_auto_reason = ""
                for auto_params, reason in auto_candidates:
                    auto_best = _pick_best_candidate(_eligible_candidates(_evaluate_forced_params(auto_params)))
                    if auto_best is None:
                        continue
                    if best_auto_candidate is None or _score(auto_best[2]) > _score(best_auto_candidate[2]):
                        best_auto_candidate = auto_best
                        best_auto_reason = reason
                if best_auto_candidate is not None:
                    best_auto_score = _score(best_auto_candidate[2])
                    auto_tuning_score_delta = best_auto_score - baseline_score
                    should_apply_auto = (
                        (not auto_symbol_tuning_require_improvement)
                        or (auto_tuning_score_delta >= auto_symbol_tuning_min_score_improvement)
                    )
                    if should_apply_auto:
                        best_time, best_params, best_stats = best_auto_candidate
                        auto_tuning_applied_symbol = True
                        auto_tuning_reason = best_auto_reason
                        auto_tuning_applied += 1
                        if str(param_override_source).startswith("symbol_override"):
                            param_override_source = "symbol_override_auto_tune"
                        else:
                            param_override_source = "auto_symbol_tuning"
        elif auto_symbol_tuning_enabled and bool(symbol_override) and (not auto_symbol_tuning_allow_with_symbol_override):
            auto_tuning_skipped_due_symbol_override += 1

        stats = best_stats
        recent_stats: Optional[Dict[str, float]] = None
        wf_recent_reasons: List[str] = []
        if dynamic_wf_enabled:
            dynamic_wf_considered_symbols += 1
            recent_stats = _recent_candidate_stats(
                state=state,
                signals_by_date=signals_by_date,
                trade_ns=trade_ns,
                symbol=symbol,
                entry_time=best_time,
                overrides=best_params or {},
                end_date=end_date,
                lookback_days=dynamic_wf_lookback_days,
            )
            recent_trades_count = int(recent_stats.get("trades_count") or 0)
            has_min_wf_trades = (dynamic_wf_min_trades <= 0) or (recent_trades_count >= dynamic_wf_min_trades)
            if dynamic_wf_min_trades > 0 and (not has_min_wf_trades):
                if dynamic_wf_skip_if_insufficient_trades:
                    if dynamic_wf_log_details:
                        logging.info(
                            "[WATCHLIST_WF_SKIP] date=%s symbol=%s reason=insufficient_trades recent_trades=%s min_trades=%s",
                            tgt,
                            symbol,
                            recent_trades_count,
                            dynamic_wf_min_trades,
                        )
                else:
                    reject_counts["wf_min_trades"] += 1
                    wf_recent_reasons.append("wf_min_trades")
            apply_wf_thresholds = has_min_wf_trades or (not dynamic_wf_skip_if_insufficient_trades)
            if apply_wf_thresholds and (
                dynamic_wf_min_total_pnl_pct is not None
                and float(recent_stats.get("total_pnl_pct") or 0.0) < dynamic_wf_min_total_pnl_pct
            ):
                reject_counts["wf_min_total_pnl_pct"] += 1
                wf_recent_reasons.append("wf_min_total_pnl_pct")
            if apply_wf_thresholds and (
                dynamic_wf_min_avg_r is not None and float(recent_stats.get("avgR") or 0.0) < dynamic_wf_min_avg_r
            ):
                reject_counts["wf_min_avg_r"] += 1
                wf_recent_reasons.append("wf_min_avg_r")
            if apply_wf_thresholds and (
                dynamic_wf_min_win_rate is not None
                and float(recent_stats.get("win_rate") or 0.0) < dynamic_wf_min_win_rate
            ):
                reject_counts["wf_min_win_rate"] += 1
                wf_recent_reasons.append("wf_min_win_rate")
            if apply_wf_thresholds and (
                dynamic_wf_max_stop_rate is not None
                and float(recent_stats.get("stop_rate") or 0.0) > dynamic_wf_max_stop_rate
            ):
                reject_counts["wf_max_stop_rate"] += 1
                wf_recent_reasons.append("wf_max_stop_rate")
            if apply_wf_thresholds and (
                dynamic_wf_max_cutoff_no_progress_share is not None
                and float(recent_stats.get("cutoff_no_progress_share") or 0.0) > dynamic_wf_max_cutoff_no_progress_share
            ):
                reject_counts["wf_max_cutoff_no_progress_share"] += 1
                wf_recent_reasons.append("wf_max_cutoff_no_progress_share")
            if apply_wf_thresholds and (
                dynamic_wf_max_loser_no_progress_share is not None
                and float(recent_stats.get("loser_no_progress_share") or 0.0) > dynamic_wf_max_loser_no_progress_share
            ):
                reject_counts["wf_max_loser_no_progress_share"] += 1
                wf_recent_reasons.append("wf_max_loser_no_progress_share")
            if apply_wf_thresholds and (
                dynamic_wf_max_stop_no_progress_share is not None
                and float(recent_stats.get("stop_no_progress_share") or 0.0) > dynamic_wf_max_stop_no_progress_share
            ):
                reject_counts["wf_max_stop_no_progress_share"] += 1
                wf_recent_reasons.append("wf_max_stop_no_progress_share")
            if apply_wf_thresholds and (
                dynamic_wf_max_cutoff_rate is not None
                and float(recent_stats.get("cutoff_rate") or 0.0) > dynamic_wf_max_cutoff_rate
            ):
                reject_counts["wf_max_cutoff_rate"] += 1
                wf_recent_reasons.append("wf_max_cutoff_rate")
            if apply_wf_thresholds and (
                dynamic_wf_min_cutoff_avg_mfe_r is not None
                and float(recent_stats.get("cutoff_avg_mfe_r") or 0.0) < dynamic_wf_min_cutoff_avg_mfe_r
            ):
                reject_counts["wf_min_cutoff_avg_mfe_r"] += 1
                wf_recent_reasons.append("wf_min_cutoff_avg_mfe_r")
            if apply_wf_thresholds and (
                dynamic_wf_min_cutoff_target_fit_ratio is not None
                and float(recent_stats.get("cutoff_target_fit_ratio") or 0.0) < dynamic_wf_min_cutoff_target_fit_ratio
            ):
                reject_counts["wf_min_cutoff_target_fit_ratio"] += 1
                wf_recent_reasons.append("wf_min_cutoff_target_fit_ratio")
            if dynamic_wf_log_details and wf_recent_reasons:
                logging.info(
                    "[WATCHLIST_WF_SUPPRESS] date=%s symbol=%s reasons=%s recent_trades=%s recent_avgR=%.4f recent_win_rate=%.4f recent_total_pnl_pct=%.4f",
                    tgt,
                    symbol,
                    ",".join(wf_recent_reasons),
                    int(recent_stats.get("trades_count") or 0),
                    float(recent_stats.get("avgR") or 0.0),
                    float(recent_stats.get("win_rate") or 0.0),
                    float(recent_stats.get("total_pnl_pct") or 0.0),
                )
            if wf_recent_reasons:
                dynamic_wf_suppressed_symbols += 1
        trades_samples.append(int(stats["trades_count"]))
        pnl_samples.append(float(stats["total_pnl_pct"]))
        pf_samples.append(float(stats["profit_factor"]))
        reasons: List[str] = []
        if wf_recent_reasons:
            reasons.extend(wf_recent_reasons)
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
        if max_stop_rate is not None and float(stats.get("stop_rate") or 0.0) > max_stop_rate:
            reject_counts["max_stop_rate"] += 1
            reasons.append("max_stop_rate")
        if max_cutoff_rate is not None and float(stats.get("cutoff_rate") or 0.0) > max_cutoff_rate:
            reject_counts["max_cutoff_rate"] += 1
            reasons.append("max_cutoff_rate")
        if (
            min_cutoff_avg_mfe_r is not None
            and float(stats.get("cutoff_avg_mfe_r") or 0.0) < min_cutoff_avg_mfe_r
        ):
            reject_counts["min_cutoff_avg_mfe_r"] += 1
            reasons.append("min_cutoff_avg_mfe_r")
        if (
            min_cutoff_target_fit_ratio is not None
            and float(stats.get("cutoff_target_fit_ratio") or 0.0) < min_cutoff_target_fit_ratio
        ):
            reject_counts["min_cutoff_target_fit_ratio"] += 1
            reasons.append("min_cutoff_target_fit_ratio")
        if (
            max_loser_no_progress_share is not None
            and float(stats.get("loser_no_progress_share") or 0.0) > max_loser_no_progress_share
        ):
            reject_counts["max_loser_no_progress_share"] += 1
            reasons.append("max_loser_no_progress_share")
        if (
            max_stop_no_progress_share is not None
            and float(stats.get("stop_no_progress_share") or 0.0) > max_stop_no_progress_share
        ):
            reject_counts["max_stop_no_progress_share"] += 1
            reasons.append("max_stop_no_progress_share")
        wf_fields: Dict[str, Any] = {}
        if dynamic_wf_enabled:
            recent_stats = recent_stats or {}
            wf_fields = {
                "wf_recent_lookback_days": int(dynamic_wf_lookback_days),
                "wf_recent_trades_count": int(recent_stats.get("trades_count") or 0),
                "wf_recent_win_rate": float(recent_stats.get("win_rate") or 0.0),
                "wf_recent_avgR": float(recent_stats.get("avgR") or 0.0),
                "wf_recent_profit_factor": float(recent_stats.get("profit_factor") or 0.0),
                "wf_recent_total_pnl_pct": float(recent_stats.get("total_pnl_pct") or 0.0),
                "wf_recent_stop_rate": float(recent_stats.get("stop_rate") or 0.0),
                "wf_recent_stop_no_progress_share": float(recent_stats.get("stop_no_progress_share") or 0.0),
                "wf_recent_cutoff_rate": float(recent_stats.get("cutoff_rate") or 0.0),
                "wf_recent_cutoff_avg_mfe_r": float(recent_stats.get("cutoff_avg_mfe_r") or 0.0),
                "wf_recent_cutoff_target_fit_ratio": float(recent_stats.get("cutoff_target_fit_ratio") or 0.0),
                "wf_recent_cutoff_no_progress_share": float(recent_stats.get("cutoff_no_progress_share") or 0.0),
                "wf_recent_loser_no_progress_share": float(recent_stats.get("loser_no_progress_share") or 0.0),
                "wf_recent_suppressed": bool(len(wf_recent_reasons) > 0),
                "wf_recent_reasons": list(wf_recent_reasons),
            }
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
            row_entry_time = (
                (entry_times_all[0] if entry_times_all else "09:30")
                if (scan_first_valid_mode and best_time == _SCAN_FIRST_VALID_LABEL)
                else best_time
            )
            selection_score = _score(stats)
            report_rows.append(
                {
                    "symbol": symbol,
                    "direction": str(getattr(signal_today, "direction", "") or "").lower(),
                    "entry_time_et": row_entry_time,
                    "param_overrides": best_params,
                    "param_override_source": param_override_source,
                    "auto_tuning_applied": auto_tuning_applied_symbol,
                    "auto_tuning_reason": auto_tuning_reason,
                    "auto_tuning_score_delta": auto_tuning_score_delta,
                    "auto_tuning_candidate_count": auto_tuning_candidate_count,
                    "entry_time_mode": ("scan_first_valid" if scan_first_valid_mode else "fixed"),
                    "selection_score_mode": param_rank_by,
                    "selection_score": float(selection_score),
                    **wf_fields,
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
                "entry_time_et": (
                    (entry_times_all[0] if entry_times_all else "09:30")
                    if (scan_first_valid_mode and best_time == _SCAN_FIRST_VALID_LABEL)
                    else best_time
                ),
                "param_overrides": best_params,
                "param_override_source": param_override_source,
                "auto_tuning_applied": auto_tuning_applied_symbol,
                "auto_tuning_reason": auto_tuning_reason,
                "auto_tuning_score_delta": auto_tuning_score_delta,
                "auto_tuning_candidate_count": auto_tuning_candidate_count,
                "entry_time_mode": ("scan_first_valid" if scan_first_valid_mode else "fixed"),
                "selection_score_mode": param_rank_by,
                "selection_score": float(_score(stats)),
                **wf_fields,
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
            if top_k_rank_by in ("entry_quality", "entry_quality_score", "quality", "quality_score"):
                return float(row.get("selection_score") or 0.0)
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
    auto_tuning_in_watchlist = sum(1 for row in watchlist if bool(row.get("auto_tuning_applied")))

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
    if dynamic_wf_enabled and dynamic_wf_considered_symbols > 0:
        logging.info(
            "[WATCHLIST_WF_FILTERS] date=%s considered=%s suppressed=%s reject_min_trades=%s reject_min_total_pnl_pct=%s reject_min_avg_r=%s reject_min_win_rate=%s reject_max_stop_rate=%s reject_max_cutoff_rate=%s reject_max_cutoff_no_progress_share=%s reject_max_loser_no_progress_share=%s reject_max_stop_no_progress_share=%s reject_min_cutoff_avg_mfe_r=%s reject_min_cutoff_target_fit_ratio=%s",
            tgt,
            dynamic_wf_considered_symbols,
            dynamic_wf_suppressed_symbols,
            reject_counts["wf_min_trades"],
            reject_counts["wf_min_total_pnl_pct"],
            reject_counts["wf_min_avg_r"],
            reject_counts["wf_min_win_rate"],
            reject_counts["wf_max_stop_rate"],
            reject_counts["wf_max_cutoff_rate"],
            reject_counts["wf_max_cutoff_no_progress_share"],
            reject_counts["wf_max_loser_no_progress_share"],
            reject_counts["wf_max_stop_no_progress_share"],
            reject_counts["wf_min_cutoff_avg_mfe_r"],
            reject_counts["wf_min_cutoff_target_fit_ratio"],
        )
    if funnel["symbols_passing_filters"] == 0 and funnel["scanned_symbols"] > 0:
        logging.info(
            "[WATCHLIST_FILTERS] date=%s reject_min_trades=%s reject_neg_pnl=%s reject_min_avg_r=%s reject_min_pf=%s reject_max_stop_rate=%s reject_max_cutoff_rate=%s reject_min_cutoff_avg_mfe_r=%s reject_min_cutoff_target_fit_ratio=%s reject_max_loser_no_progress_share=%s reject_max_stop_no_progress_share=%s",
            tgt,
            reject_counts["min_trades"],
            reject_counts["neg_pnl"],
            reject_counts["min_avg_r"],
            reject_counts["min_pf"],
            reject_counts["max_stop_rate"],
            reject_counts["max_cutoff_rate"],
            reject_counts["min_cutoff_avg_mfe_r"],
            reject_counts["min_cutoff_target_fit_ratio"],
            reject_counts["max_loser_no_progress_share"],
            reject_counts["max_stop_no_progress_share"],
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
        if reject_counts["max_stop_rate"] > 0:
            logging.info("[WATCHLIST_FILTERS] date=%s reject_max_stop_rate=%s", tgt, reject_counts["max_stop_rate"])
        if reject_counts["max_cutoff_rate"] > 0:
            logging.info("[WATCHLIST_FILTERS] date=%s reject_max_cutoff_rate=%s", tgt, reject_counts["max_cutoff_rate"])
        if reject_counts["min_cutoff_avg_mfe_r"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_min_cutoff_avg_mfe_r=%s",
                tgt,
                reject_counts["min_cutoff_avg_mfe_r"],
            )
        if reject_counts["min_cutoff_target_fit_ratio"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_min_cutoff_target_fit_ratio=%s",
                tgt,
                reject_counts["min_cutoff_target_fit_ratio"],
            )
        if reject_counts["max_loser_no_progress_share"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_max_loser_no_progress_share=%s",
                tgt,
                reject_counts["max_loser_no_progress_share"],
            )
        if reject_counts["max_stop_no_progress_share"] > 0:
            logging.info(
                "[WATCHLIST_FILTERS] date=%s reject_max_stop_no_progress_share=%s",
                tgt,
                reject_counts["max_stop_no_progress_share"],
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
        def _avg_key(key: str) -> float:
            vals: List[float] = []
            for row in watchlist:
                try:
                    vals.append(float(row.get(key) or 0.0))
                except Exception:
                    continue
            if not vals:
                return 0.0
            return float(sum(vals) / float(len(vals)))
        logging.info(
            "[WATCHLIST_CAUSE] date=%s stop_no_progress=%.3f stop_near_target=%.3f cutoff_no_progress=%.3f cutoff_near_target=%.3f loser_no_progress=%.3f loser_near_target=%.3f",
            tgt,
            _avg_key("stop_no_progress_share"),
            _avg_key("stop_near_target_share"),
            _avg_key("cutoff_no_progress_share"),
            _avg_key("cutoff_near_target_share"),
            _avg_key("loser_no_progress_share"),
            _avg_key("loser_near_target_share"),
        )

    watchlist_meta = {
        "funnel": dict(funnel),
        "reject_counts": dict(reject_counts),
        "selected_summary": summarize_watchlist_rows(watchlist),
        "entry_time_counts": entry_time_counts,
        "symbol_param_overrides_used_pre_top_k": int(symbol_overrides_used),
        "symbol_param_overrides_used_in_watchlist": int(symbol_overrides_in_watchlist),
        "auto_symbol_tuning_considered": int(auto_tuning_considered),
        "auto_symbol_tuning_applied": int(auto_tuning_applied),
        "auto_symbol_tuning_applied_in_watchlist": int(auto_tuning_in_watchlist),
        "auto_symbol_tuning_skipped_due_symbol_override": int(auto_tuning_skipped_due_symbol_override),
        "dynamic_walk_forward_considered": int(dynamic_wf_considered_symbols),
        "dynamic_walk_forward_suppressed": int(dynamic_wf_suppressed_symbols),
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
            "maxStopRate": max_stop_rate,
            "maxCutoffRate": max_cutoff_rate,
            "min_cutoff_avg_mfe_r": min_cutoff_avg_mfe_r,
            "min_cutoff_target_fit_ratio": min_cutoff_target_fit_ratio,
            "max_loser_no_progress_share": max_loser_no_progress_share,
            "max_stop_no_progress_share": max_stop_no_progress_share,
            "top_k": top_k,
            "entry_time_rank_by": entry_time_rank_by,
            "top_k_rank_by": top_k_rank_by,
            "param_rank_by": param_rank_by,
            "entry_quality_score_trades_ref": quality_score_trades_ref,
            "entry_quality_score_weight_lcb": quality_score_weight_lcb,
            "entry_quality_score_weight_target_rate": quality_score_weight_target_rate,
            "entry_quality_score_weight_stop_rate": quality_score_weight_stop_rate,
            "entry_quality_score_weight_cutoff_rate": quality_score_weight_cutoff_rate,
            "entry_quality_score_weight_stop_no_progress": quality_score_weight_stop_no_progress,
            "entry_quality_score_weight_cutoff_no_progress": quality_score_weight_cutoff_no_progress,
            "entry_quality_score_weight_loser_no_progress": quality_score_weight_loser_no_progress,
            "entry_quality_score_min_target_fit_ratio": quality_score_min_target_fit_ratio,
            "entry_quality_score_weight_target_fit_shortfall": quality_score_weight_target_fit_shortfall,
            "directional_history_only": directional_history_only,
            "entry_time_mode": ("scan_first_valid" if scan_first_valid_mode else "fixed"),
            "scan_first_valid_candidate_mode": scan_first_valid_candidate_mode,
            "symbol_param_override_mode": symbol_param_override_mode,
            "symbol_param_overrides_count": len(symbol_param_overrides),
            "auto_symbol_tuning": auto_symbol_tuning_cfg,
            "dynamic_walk_forward": dynamic_wf_cfg,
        },
    }

    payload = {"date": tgt, "watchlist": watchlist, "meta": watchlist_meta}
    if watchlist:
        write_watchlist(watchlist, cfg, date_str=tgt, meta=watchlist_meta)
    else:
        logging.warning("[WATCHLIST] empty watchlist date=%s; no fallback applied", tgt)
        # Overwrite any stale watchlist for this date so replay can't pick up old symbols.
        write_watchlist([], cfg, date_str=tgt, meta=watchlist_meta)
    if run_id is None and bool((cfg.get("watchlist") or {}).get("freeze_live_snapshot_enabled", True)):
        freeze_watchlist_snapshot(payload, cfg, date_str=tgt, overwrite=False)
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
                "maxCutoffRate": max_cutoff_rate,
                "min_cutoff_avg_mfe_r": min_cutoff_avg_mfe_r,
                "min_cutoff_target_fit_ratio": min_cutoff_target_fit_ratio,
                "max_loser_no_progress_share": max_loser_no_progress_share,
                "max_stop_no_progress_share": max_stop_no_progress_share,
                "top_k": top_k,
                "entry_time_rank_by": entry_time_rank_by,
                "top_k_rank_by": top_k_rank_by,
                "param_rank_by": param_rank_by,
                "entry_quality_score_trades_ref": quality_score_trades_ref,
                "entry_quality_score_weight_lcb": quality_score_weight_lcb,
                "entry_quality_score_weight_target_rate": quality_score_weight_target_rate,
                "entry_quality_score_weight_stop_rate": quality_score_weight_stop_rate,
                "entry_quality_score_weight_cutoff_rate": quality_score_weight_cutoff_rate,
                "entry_quality_score_weight_stop_no_progress": quality_score_weight_stop_no_progress,
                "entry_quality_score_weight_cutoff_no_progress": quality_score_weight_cutoff_no_progress,
                "entry_quality_score_weight_loser_no_progress": quality_score_weight_loser_no_progress,
                "entry_quality_score_min_target_fit_ratio": quality_score_min_target_fit_ratio,
                "entry_quality_score_weight_target_fit_shortfall": quality_score_weight_target_fit_shortfall,
                "param_grid": watch_cfg.get("param_grid") or {},
                "directional_history_only": directional_history_only,
                "entry_time_mode": ("scan_first_valid" if scan_first_valid_mode else "fixed"),
                "scan_first_valid_candidate_mode": scan_first_valid_candidate_mode,
                "symbol_param_override_mode": symbol_param_override_mode,
                "symbol_param_overrides_count": len(symbol_param_overrides),
                "auto_symbol_tuning": auto_symbol_tuning_cfg,
                "dynamic_walk_forward": dynamic_wf_cfg,
            },
            "summary": {
                **funnel,
                **reject_counts,
                "symbol_overrides_used_pre_top_k": int(symbol_overrides_used),
                "symbol_overrides_used_in_watchlist": int(symbol_overrides_in_watchlist),
                "auto_symbol_tuning_considered": int(auto_tuning_considered),
                "auto_symbol_tuning_applied": int(auto_tuning_applied),
                "auto_symbol_tuning_applied_in_watchlist": int(auto_tuning_in_watchlist),
                "auto_symbol_tuning_skipped_due_symbol_override": int(auto_tuning_skipped_due_symbol_override),
                "dynamic_walk_forward_considered": int(dynamic_wf_considered_symbols),
                "dynamic_walk_forward_suppressed": int(dynamic_wf_suppressed_symbols),
                "pass_trades_only": pass_trades_only,
                "pass_trades_and_pnl": pass_trades_and_pnl,
                "pass_trades_and_avg_r": pass_trades_and_avg_r,
                "pass_trades_and_pf": pass_trades_and_pf,
                "pass_trades_and_win_rate": pass_trades_and_win_rate,
            },
            "symbols": report_rows,
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        overrides_path = report_dir / f"overrides_{tgt}.json"
        overrides_payload = {
            "date": tgt,
            "watchlist_size": len(watchlist),
            "rows": [
                {
                    "symbol": str(row.get("symbol") or ""),
                    "direction": str(row.get("direction") or ""),
                    "entry_time_et": str(row.get("entry_time_et") or ""),
                    "param_overrides": (row.get("param_overrides") or {}),
                    "param_override_source": str(row.get("param_override_source") or ""),
                    "auto_tuning_applied": bool(row.get("auto_tuning_applied")),
                    "auto_tuning_reason": str(row.get("auto_tuning_reason") or ""),
                    "auto_tuning_score_delta": float(row.get("auto_tuning_score_delta") or 0.0),
                    "auto_tuning_candidate_count": int(row.get("auto_tuning_candidate_count") or 0),
                }
                for row in watchlist
            ],
        }
        overrides_path.write_text(json.dumps(overrides_payload, indent=2), encoding="utf-8")
    # Muted insufficient history spam; keep list in case we want to surface later.
    return watchlist
