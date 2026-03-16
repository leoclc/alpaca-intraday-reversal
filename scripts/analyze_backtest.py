from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_CUTOFF_EXIT_REASONS = {"time_stop", "eod_flat", "time_exit"}
_TARGET_R_GRID = [0.25, 0.50, 0.75, 1.00, 1.20]
_STOP_MULT_GRID = [1.10, 1.25, 1.50, 2.00]


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


def _stable_json(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        try:
            return str(value)
        except Exception:
            return ""


def _longest_negative_streak(values: List[float]) -> int:
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
    open_noise_abs = t.get("open_noise_abs")
    open_noise_stop_ratio = t.get("open_noise_stop_ratio")
    stop_to_open_noise_ratio = t.get("stop_to_open_noise_ratio")
    target_r = abs(tp - ep) / sd if sd > 0 else None
    stop_atr = sd / atr if atr > 0 else None
    gap_fav_bps = gap_bps if direction == "short" else -gap_bps
    out = dict(t)
    out["_month"] = _iso_month(str(t.get("entry_date") or ""))
    out["_target_r"] = target_r
    out["_stop_atr"] = stop_atr
    out["_gap_fav_bps"] = gap_fav_bps
    out["_early_pullback_bps"] = _safe_float(pb, default=0.0) if pb is not None else None
    if open_noise_abs is not None:
        out["_open_noise_abs"] = _safe_float(open_noise_abs, default=0.0)
    if t.get("open_noise_bps") is not None:
        out["_open_noise_bps"] = _safe_float(t.get("open_noise_bps"), default=0.0)
    if t.get("open_noise_atr") is not None:
        out["_open_noise_atr"] = _safe_float(t.get("open_noise_atr"), default=0.0)
    if open_noise_stop_ratio is not None:
        out["_open_noise_stop_ratio"] = _safe_float(open_noise_stop_ratio, default=0.0)
    elif open_noise_abs is not None and sd > 0:
        out["_open_noise_stop_ratio"] = _safe_float(open_noise_abs, default=0.0) / sd
    if stop_to_open_noise_ratio is not None:
        out["_stop_to_open_noise_ratio"] = _safe_float(stop_to_open_noise_ratio, default=0.0)
    elif open_noise_abs is not None and _safe_float(open_noise_abs, default=0.0) > 0 and sd > 0:
        out["_stop_to_open_noise_ratio"] = sd / _safe_float(open_noise_abs, default=0.0)
    po = t.get("param_overrides")
    out["param_overrides_json"] = _stable_json(po) if isinstance(po, dict) and po else ""
    out["_fill_r_delta"] = None
    if t.get("raw_r_multiple") is not None and t.get("r_multiple") is not None:
        out["_fill_r_delta"] = _safe_float(t.get("r_multiple")) - _safe_float(t.get("raw_r_multiple"))
    out["_fill_pnl_pct_delta"] = None
    if t.get("raw_pnl_pct") is not None and t.get("pnl_pct") is not None:
        out["_fill_pnl_pct_delta"] = _safe_float(t.get("pnl_pct")) - _safe_float(t.get("raw_pnl_pct"))
    out["_fill_cost_bps_notional"] = None
    notional = _safe_float(t.get("notional"), default=0.0)
    if t.get("fill_cost_total") is not None and notional > 0:
        out["_fill_cost_bps_notional"] = (_safe_float(t.get("fill_cost_total")) / notional) * 10000.0
    # Prefer persisted watchlist stats from the trade record when available.
    wl_stats = t.get("watchlist_stats") if isinstance(t.get("watchlist_stats"), dict) else {}

    def _wl_raw(key: str):
        pref = f"watchlist_{key}"
        if t.get(pref) is not None:
            return t.get(pref)
        return wl_stats.get(key)

    if _wl_raw("rank") is not None:
        out["_watchlist_rank"] = _safe_int(_wl_raw("rank"), default=0)
    if _wl_raw("avgR") is not None:
        out["_watchlist_avgR"] = _safe_float(_wl_raw("avgR"), default=0.0)
    if _wl_raw("avgR_stderr") is not None:
        out["_watchlist_avgR_stderr"] = _safe_float(_wl_raw("avgR_stderr"), default=0.0)
    if _wl_raw("win_rate") is not None:
        out["_watchlist_win_rate"] = _safe_float(_wl_raw("win_rate"), default=0.0)
    if _wl_raw("profit_factor") is not None:
        out["_watchlist_profit_factor"] = _safe_float(_wl_raw("profit_factor"), default=0.0)
    if _wl_raw("trades_count") is not None:
        out["_watchlist_trades_count"] = _safe_int(_wl_raw("trades_count"), default=0)
    if _wl_raw("total_pnl_pct") is not None:
        out["_watchlist_total_pnl_pct"] = _safe_float(_wl_raw("total_pnl_pct"), default=0.0)
    if _wl_raw("stop_count") is not None:
        out["_watchlist_stop_count"] = _safe_int(_wl_raw("stop_count"), default=0)
    if _wl_raw("target_count") is not None:
        out["_watchlist_target_count"] = _safe_int(_wl_raw("target_count"), default=0)
    if _wl_raw("cutoff_count") is not None:
        out["_watchlist_cutoff_count"] = _safe_int(_wl_raw("cutoff_count"), default=0)
    if _wl_raw("stop_rate") is not None:
        out["_watchlist_stop_rate"] = _safe_float(_wl_raw("stop_rate"), default=0.0)
    if _wl_raw("target_rate") is not None:
        out["_watchlist_target_rate"] = _safe_float(_wl_raw("target_rate"), default=0.0)
    if _wl_raw("cutoff_rate") is not None:
        out["_watchlist_cutoff_rate"] = _safe_float(_wl_raw("cutoff_rate"), default=0.0)
    if _wl_raw("cutoff_avg_mfe_r") is not None:
        out["_watchlist_cutoff_avg_mfe_r"] = _safe_float(_wl_raw("cutoff_avg_mfe_r"), default=0.0)
    if _wl_raw("cutoff_avg_abs_mae_r") is not None:
        out["_watchlist_cutoff_avg_abs_mae_r"] = _safe_float(_wl_raw("cutoff_avg_abs_mae_r"), default=0.0)
    if _wl_raw("avg_target_r") is not None:
        out["_watchlist_avg_target_r"] = _safe_float(_wl_raw("avg_target_r"), default=0.0)
    if _wl_raw("cutoff_target_fit_ratio") is not None:
        out["_watchlist_cutoff_target_fit_ratio"] = _safe_float(_wl_raw("cutoff_target_fit_ratio"), default=0.0)
    if _wl_raw("stop_flip_share_any") is not None:
        out["_watchlist_stop_flip_share_any"] = _safe_float(_wl_raw("stop_flip_share_any"), default=0.0)
    if _wl_raw("stop_flip_share_050") is not None:
        out["_watchlist_stop_flip_share_050"] = _safe_float(_wl_raw("stop_flip_share_050"), default=0.0)
    if _wl_raw("stop_flip_share_100") is not None:
        out["_watchlist_stop_flip_share_100"] = _safe_float(_wl_raw("stop_flip_share_100"), default=0.0)
    if _wl_raw("stop_no_progress_share") is not None:
        out["_watchlist_stop_no_progress_share"] = _safe_float(_wl_raw("stop_no_progress_share"), default=0.0)
    if _wl_raw("stop_near_target_share") is not None:
        out["_watchlist_stop_near_target_share"] = _safe_float(_wl_raw("stop_near_target_share"), default=0.0)
    if _wl_raw("cutoff_no_progress_share") is not None:
        out["_watchlist_cutoff_no_progress_share"] = _safe_float(_wl_raw("cutoff_no_progress_share"), default=0.0)
    if _wl_raw("cutoff_near_target_share") is not None:
        out["_watchlist_cutoff_near_target_share"] = _safe_float(_wl_raw("cutoff_near_target_share"), default=0.0)
    if _wl_raw("stop_reach_target_with_wider_stop_share") is not None:
        out["_watchlist_stop_reach_target_with_wider_stop_share"] = _safe_float(
            _wl_raw("stop_reach_target_with_wider_stop_share"),
            default=0.0,
        )
    if _wl_raw("avg_stop_mult_needed_for_target") is not None:
        out["_watchlist_avg_stop_mult_needed_for_target"] = _safe_float(
            _wl_raw("avg_stop_mult_needed_for_target"),
            default=0.0,
        )
    if _wl_raw("loser_count") is not None:
        out["_watchlist_loser_count"] = _safe_int(_wl_raw("loser_count"), default=0)
    if _wl_raw("loser_no_progress_share") is not None:
        out["_watchlist_loser_no_progress_share"] = _safe_float(_wl_raw("loser_no_progress_share"), default=0.0)
    if _wl_raw("loser_near_target_share") is not None:
        out["_watchlist_loser_near_target_share"] = _safe_float(_wl_raw("loser_near_target_share"), default=0.0)
    if _wl_raw("positive_month_rate") is not None:
        out["_watchlist_positive_month_rate"] = _safe_float(_wl_raw("positive_month_rate"), default=0.0)
    if _wl_raw("max_monthly_drawdown_pct") is not None:
        out["_watchlist_max_monthly_drawdown_pct"] = _safe_float(_wl_raw("max_monthly_drawdown_pct"), default=0.0)
    if _wl_raw("param_override_source") is not None:
        out["_watchlist_param_override_source"] = str(_wl_raw("param_override_source") or "")
    if _wl_raw("auto_tuning_applied") is not None:
        out["_watchlist_auto_tuning_applied"] = bool(_wl_raw("auto_tuning_applied"))
    if _wl_raw("auto_tuning_reason") is not None:
        out["_watchlist_auto_tuning_reason"] = str(_wl_raw("auto_tuning_reason") or "")
    if _wl_raw("auto_tuning_score_delta") is not None:
        out["_watchlist_auto_tuning_score_delta"] = _safe_float(_wl_raw("auto_tuning_score_delta"), default=0.0)
    if _wl_raw("auto_tuning_candidate_count") is not None:
        out["_watchlist_auto_tuning_candidate_count"] = _safe_int(_wl_raw("auto_tuning_candidate_count"), default=0)
    if _wl_raw("wf_recent_lookback_days") is not None:
        out["_watchlist_wf_recent_lookback_days"] = _safe_int(_wl_raw("wf_recent_lookback_days"), default=0)
    if _wl_raw("wf_recent_trades_count") is not None:
        out["_watchlist_wf_recent_trades_count"] = _safe_int(_wl_raw("wf_recent_trades_count"), default=0)
    if _wl_raw("wf_recent_win_rate") is not None:
        out["_watchlist_wf_recent_win_rate"] = _safe_float(_wl_raw("wf_recent_win_rate"), default=0.0)
    if _wl_raw("wf_recent_avgR") is not None:
        out["_watchlist_wf_recent_avgR"] = _safe_float(_wl_raw("wf_recent_avgR"), default=0.0)
    if _wl_raw("wf_recent_profit_factor") is not None:
        out["_watchlist_wf_recent_profit_factor"] = _safe_float(_wl_raw("wf_recent_profit_factor"), default=0.0)
    if _wl_raw("wf_recent_total_pnl_pct") is not None:
        out["_watchlist_wf_recent_total_pnl_pct"] = _safe_float(_wl_raw("wf_recent_total_pnl_pct"), default=0.0)
    if _wl_raw("wf_recent_stop_rate") is not None:
        out["_watchlist_wf_recent_stop_rate"] = _safe_float(_wl_raw("wf_recent_stop_rate"), default=0.0)
    if _wl_raw("wf_recent_stop_no_progress_share") is not None:
        out["_watchlist_wf_recent_stop_no_progress_share"] = _safe_float(
            _wl_raw("wf_recent_stop_no_progress_share"),
            default=0.0,
        )
    if _wl_raw("wf_recent_cutoff_rate") is not None:
        out["_watchlist_wf_recent_cutoff_rate"] = _safe_float(_wl_raw("wf_recent_cutoff_rate"), default=0.0)
    if _wl_raw("wf_recent_cutoff_avg_mfe_r") is not None:
        out["_watchlist_wf_recent_cutoff_avg_mfe_r"] = _safe_float(_wl_raw("wf_recent_cutoff_avg_mfe_r"), default=0.0)
    if _wl_raw("wf_recent_cutoff_no_progress_share") is not None:
        out["_watchlist_wf_recent_cutoff_no_progress_share"] = _safe_float(
            _wl_raw("wf_recent_cutoff_no_progress_share"),
            default=0.0,
        )
    if _wl_raw("wf_recent_loser_no_progress_share") is not None:
        out["_watchlist_wf_recent_loser_no_progress_share"] = _safe_float(
            _wl_raw("wf_recent_loser_no_progress_share"),
            default=0.0,
        )
    if _wl_raw("wf_recent_suppressed") is not None:
        out["_watchlist_wf_recent_suppressed"] = bool(_wl_raw("wf_recent_suppressed"))
    if _wl_raw("wf_recent_reasons") is not None:
        out["_watchlist_wf_recent_reasons"] = _stable_json(_wl_raw("wf_recent_reasons"))
    if t.get("quality_risk_mult") is not None:
        out["_quality_risk_mult"] = _safe_float(t.get("quality_risk_mult"), default=0.0)
    if t.get("quality_score") is not None:
        out["_quality_score"] = _safe_float(t.get("quality_score"), default=0.0)
    # Loss classification for stopouts: did the day ever reach the target distance after entry?
    if str(t.get("exit_reason") or "") == "stop" and target_r is not None and t.get("day_mfe_r") is not None:
        try:
            out["_stop_shakeout_day_hit_target"] = bool(_safe_float(t.get("day_mfe_r")) >= float(target_r))
        except Exception:
            out["_stop_shakeout_day_hit_target"] = None
    else:
        out["_stop_shakeout_day_hit_target"] = None

    # "What would make this trade win" (safe, bar-based statements only).
    exit_reason = str(t.get("exit_reason") or "")
    r_mult = _safe_float(t.get("r_multiple"), default=0.0)
    mfe_r_before_stop = None if t.get("mfe_r_before_stop") is None else _safe_float(t.get("mfe_r_before_stop"))
    mae_r_to_target = None if t.get("mae_r_to_target") is None else _safe_float(t.get("mae_r_to_target"))
    mfe_r_full = None if t.get("mfe_r_full") is None else _safe_float(t.get("mfe_r_full"))

    out["_flip_target_r_max_before_stop"] = None
    if exit_reason == "stop" and mfe_r_before_stop is not None:
        out["_flip_target_r_max_before_stop"] = mfe_r_before_stop

    out["_flip_stop_mult_needed_for_original_target"] = None
    if (
        exit_reason == "stop"
        and t.get("target_hit_ts") is not None
        and mae_r_to_target is not None
        and mae_r_to_target < 0
    ):
        out["_flip_stop_mult_needed_for_original_target"] = -mae_r_to_target

    out["_flip_target_r_max_by_cutoff"] = None
    if exit_reason in {"time_stop", "eod_flat", "time_exit"} and r_mult <= 0 and mfe_r_full is not None:
        out["_flip_target_r_max_by_cutoff"] = mfe_r_full

    # Root-cause bucketing for losing/flat outcomes.
    out["_loss_bucket"] = "win_or_flat"
    out["_entry_quality_issue"] = 0
    out["_bracket_issue"] = 0
    out["_target_issue"] = 0
    if r_mult <= 0:
        if exit_reason == "stop":
            near_thresh = (0.75 * float(target_r)) if (target_r is not None and float(target_r) > 0) else 0.75
            near_target_before_stop = (
                mfe_r_before_stop is not None
                and near_thresh > 0
                and float(mfe_r_before_stop) >= near_thresh
            )
            recoverable_with_wider_stop = out["_flip_stop_mult_needed_for_original_target"] is not None
            if mfe_r_before_stop is not None and float(mfe_r_before_stop) <= 1e-9:
                out["_loss_bucket"] = "stop_no_progress"
                out["_entry_quality_issue"] = 1
            elif near_target_before_stop and recoverable_with_wider_stop:
                out["_loss_bucket"] = "stop_near_target_recoverable"
                out["_bracket_issue"] = 1
            elif near_target_before_stop:
                out["_loss_bucket"] = "stop_near_target"
                out["_bracket_issue"] = 1
            elif recoverable_with_wider_stop:
                out["_loss_bucket"] = "stop_recoverable_with_wider_stop"
                out["_bracket_issue"] = 1
            else:
                out["_loss_bucket"] = "stop_midpath_reversal"
        elif exit_reason in _CUTOFF_EXIT_REASONS:
            near_thresh = (0.75 * float(target_r)) if (target_r is not None and float(target_r) > 0) else 0.75
            cutoff_near_target = (
                mfe_r_full is not None
                and near_thresh > 0
                and float(mfe_r_full) >= near_thresh
            )
            if mfe_r_full is not None and float(mfe_r_full) <= 1e-9:
                out["_loss_bucket"] = "cutoff_no_progress"
                out["_entry_quality_issue"] = 1
            elif cutoff_near_target:
                out["_loss_bucket"] = "cutoff_near_target_time_limited"
                out["_target_issue"] = 1
            elif out["_flip_target_r_max_by_cutoff"] is not None:
                out["_loss_bucket"] = "cutoff_partial_progress"
                out["_target_issue"] = 1
            else:
                out["_loss_bucket"] = "cutoff_no_edge"
        else:
            out["_loss_bucket"] = f"loss_{exit_reason or 'other'}"
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
            if t.get("_watchlist_rank") is None:
                t["_watchlist_rank"] = row.rank
            if t.get("_watchlist_avgR") is None:
                t["_watchlist_avgR"] = row.avgR
            if t.get("_watchlist_win_rate") is None:
                t["_watchlist_win_rate"] = row.win_rate
            if t.get("_watchlist_profit_factor") is None:
                t["_watchlist_profit_factor"] = row.profit_factor
            if t.get("_watchlist_trades_count") is None:
                t["_watchlist_trades_count"] = row.trades_count
            if t.get("_watchlist_total_pnl_pct") is None:
                t["_watchlist_total_pnl_pct"] = row.total_pnl_pct

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    watchlist_funnel_rows: List[Dict[str, Any]] = []
    watchlist_funnel_summary: Optional[Dict[str, Any]] = None
    watchlist_reject_top: List[Tuple[str, int]] = []
    if watchlists_dir and watchlists_dir.exists():
        reject_keys = [
            "min_trades",
            "neg_pnl",
            "min_pf",
            "min_avg_r",
            "min_win_rate",
            "min_months_count",
            "min_positive_month_rate",
            "max_negative_months",
            "max_monthly_drawdown_pct",
            "max_longest_negative_month_streak",
            "min_worst_month_pnl_pct",
        ]
        dates = sorted({str(t.get("entry_date") or "") for t in trades if str(t.get("entry_date") or "")})
        min_date = dates[0] if dates else ""
        max_date = dates[-1] if dates else ""
        for path in sorted(watchlists_dir.glob("*.json")):
            date = str(path.stem or "")
            if min_date and date < min_date:
                continue
            if max_date and date > max_date:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = payload.get("meta") if isinstance(payload, dict) else {}
            if not isinstance(meta, dict):
                continue
            funnel = meta.get("funnel") if isinstance(meta.get("funnel"), dict) else {}
            reject_counts = meta.get("reject_counts") if isinstance(meta.get("reject_counts"), dict) else {}
            filters = meta.get("filters") if isinstance(meta.get("filters"), dict) else {}
            row: Dict[str, Any] = {
                "date": date,
                "scanned_symbols": _safe_int(funnel.get("scanned_symbols"), default=0),
                "signals_found": _safe_int(funnel.get("signals_found"), default=0),
                "trades_simulated": _safe_int(funnel.get("trades_simulated"), default=0),
                "symbols_passing_filters": _safe_int(funnel.get("symbols_passing_filters"), default=0),
                "watchlist_size": _safe_int(funnel.get("watchlist_size"), default=0),
                "minTrades": _safe_int(filters.get("minTrades"), default=0),
                "top_k": _safe_int(filters.get("top_k"), default=0),
            }
            for k in reject_keys:
                row[f"reject_{k}"] = _safe_int(reject_counts.get(k), default=0)
            row["bound_by_top_k"] = 1 if (row["top_k"] > 0 and row["symbols_passing_filters"] > row["watchlist_size"]) else 0
            watchlist_funnel_rows.append(row)

        if watchlist_funnel_rows:
            daily_fields = [
                "date",
                "scanned_symbols",
                "signals_found",
                "trades_simulated",
                "symbols_passing_filters",
                "watchlist_size",
                "minTrades",
                "top_k",
                "bound_by_top_k",
            ] + [f"reject_{k}" for k in reject_keys]
            _write_csv(analysis_dir / "watchlist_funnel_daily.csv", daily_fields, watchlist_funnel_rows)

            n_days = len(watchlist_funnel_rows)
            avg_scanned = sum(_safe_float(r.get("scanned_symbols"), default=0.0) for r in watchlist_funnel_rows) / float(n_days)
            avg_passed = sum(_safe_float(r.get("symbols_passing_filters"), default=0.0) for r in watchlist_funnel_rows) / float(n_days)
            avg_watchlist = sum(_safe_float(r.get("watchlist_size"), default=0.0) for r in watchlist_funnel_rows) / float(n_days)
            top_k_bound_days = sum(int(_safe_int(r.get("bound_by_top_k"), default=0) > 0) for r in watchlist_funnel_rows)
            summary: Dict[str, Any] = {
                "days": n_days,
                "avg_scanned_symbols": avg_scanned,
                "avg_symbols_passing_filters": avg_passed,
                "avg_watchlist_size": avg_watchlist,
                "days_bound_by_top_k": top_k_bound_days,
                "bound_by_top_k_share": (top_k_bound_days / float(n_days)) if n_days > 0 else 0.0,
            }
            for k in reject_keys:
                summary[f"total_reject_{k}"] = sum(_safe_int(r.get(f"reject_{k}"), default=0) for r in watchlist_funnel_rows)
            _write_csv(analysis_dir / "watchlist_funnel_summary.csv", list(summary.keys()), [summary])
            watchlist_funnel_summary = summary
            watchlist_reject_top = sorted(
                [(k, int(summary.get(f"total_reject_{k}") or 0)) for k in reject_keys],
                key=lambda kv: kv[1],
                reverse=True,
            )

    # Trade table (trade-by-trade).
    po_keys = sorted(
        {
            str(k)
            for t in trades
            for k in ((t.get("param_overrides") or {}).keys() if isinstance(t.get("param_overrides"), dict) else [])
        }
    )
    if po_keys:
        for t in trades:
            po = t.get("param_overrides")
            po_map = po if isinstance(po, dict) else {}
            for key in po_keys:
                t[f"po__{key}"] = po_map.get(key)

    trade_fields = [
        "entry_date",
        "symbol",
        "direction",
        "entry_time_et",
        "entry_price_mode",
        "param_overrides_json",
        "exit_reason",
        "exit_ts",
        "stop_hit_ts",
        "target_hit_ts",
        "r_multiple",
        "pnl_total",
        "equity_before",
        "equity_after",
        "stop_distance",
        "stop_distance_fill",
        "atr",
        "target_price",
        "entry_price",
        "raw_entry_price",
        "entry_fill_price",
        "raw_exit_price",
        "exit_fill_price",
        "fill_model_enabled",
        "quote_fill_enabled",
        "entry_fill_mode",
        "exit_fill_mode",
        "entry_quote_ok",
        "exit_quote_ok",
        "entry_quote_spread_cap_bps",
        "entry_quote_deviation_cap_bps",
        "exit_quote_spread_cap_bps",
        "exit_quote_deviation_cap_bps",
        "entry_quote_reason",
        "exit_quote_reason",
        "entry_quote_ts",
        "exit_quote_ts",
        "entry_quote_selection",
        "exit_quote_selection",
        "entry_quote_age_seconds",
        "exit_quote_age_seconds",
        "entry_quote_spread_bps",
        "exit_quote_spread_bps",
        "entry_quote_deviation_bps",
        "exit_quote_deviation_bps",
        "entry_quote_bid",
        "entry_quote_ask",
        "exit_quote_bid",
        "exit_quote_ask",
        "entry_quote_price_source",
        "exit_quote_price_source",
        "target_limit_bound_applied",
        "fill_exit_reason_bucket",
        "fill_half_spread_bps",
        "fill_entry_slippage_bps",
        "fill_exit_slippage_bps",
        "fill_entry_bps",
        "fill_exit_bps",
        "fill_entry_bps_model",
        "fill_exit_bps_model",
        "fill_cost_per_share",
        "fill_cost_total",
        "raw_pnl_pct",
        "raw_r_multiple",
        "gap_bps",
        "early_pullback_bps",
        "early_reversal_bps",
        "open_noise_abs",
        "open_noise_bps",
        "open_noise_atr",
        "open_noise_stop_ratio",
        "stop_to_open_noise_ratio",
        "open_noise_window_minutes",
        "confirm_move_bps",
        "confirm_minutes",
        "confirm_hit_bps",
        "mfe_r",
        "mae_r",
        "mfe_r_full",
        "mae_r_full",
        "mfe_r_before_stop",
        "mae_r_to_target",
        "day_mfe_r",
        "day_mae_r",
        "_target_r",
        "_stop_atr",
        "_gap_fav_bps",
        "_open_noise_abs",
        "_open_noise_bps",
        "_open_noise_atr",
        "_open_noise_stop_ratio",
        "_stop_to_open_noise_ratio",
        "_fill_r_delta",
        "_fill_pnl_pct_delta",
        "_fill_cost_bps_notional",
        "_stop_shakeout_day_hit_target",
        "_flip_target_r_max_before_stop",
        "_flip_stop_mult_needed_for_original_target",
        "_flip_target_r_max_by_cutoff",
        "_watchlist_rank",
        "_watchlist_avgR",
        "_watchlist_avgR_stderr",
        "_watchlist_win_rate",
        "_watchlist_profit_factor",
        "_watchlist_trades_count",
        "_watchlist_total_pnl_pct",
        "_watchlist_stop_count",
        "_watchlist_target_count",
        "_watchlist_cutoff_count",
        "_watchlist_stop_rate",
        "_watchlist_target_rate",
        "_watchlist_cutoff_rate",
        "_watchlist_cutoff_avg_mfe_r",
        "_watchlist_cutoff_avg_abs_mae_r",
        "_watchlist_stop_flip_share_any",
        "_watchlist_stop_flip_share_050",
        "_watchlist_stop_flip_share_100",
        "_watchlist_stop_no_progress_share",
        "_watchlist_stop_near_target_share",
        "_watchlist_cutoff_no_progress_share",
        "_watchlist_cutoff_near_target_share",
        "_watchlist_stop_reach_target_with_wider_stop_share",
        "_watchlist_avg_stop_mult_needed_for_target",
        "_watchlist_loser_count",
        "_watchlist_loser_no_progress_share",
        "_watchlist_loser_near_target_share",
        "_watchlist_positive_month_rate",
        "_watchlist_max_monthly_drawdown_pct",
        "_watchlist_param_override_source",
        "_watchlist_auto_tuning_applied",
        "_watchlist_auto_tuning_reason",
        "_watchlist_auto_tuning_score_delta",
        "_watchlist_auto_tuning_candidate_count",
        "_watchlist_wf_recent_lookback_days",
        "_watchlist_wf_recent_trades_count",
        "_watchlist_wf_recent_win_rate",
        "_watchlist_wf_recent_avgR",
        "_watchlist_wf_recent_profit_factor",
        "_watchlist_wf_recent_total_pnl_pct",
        "_watchlist_wf_recent_stop_rate",
        "_watchlist_wf_recent_stop_no_progress_share",
        "_watchlist_wf_recent_cutoff_rate",
        "_watchlist_wf_recent_cutoff_avg_mfe_r",
        "_watchlist_wf_recent_cutoff_no_progress_share",
        "_watchlist_wf_recent_loser_no_progress_share",
        "_watchlist_wf_recent_suppressed",
        "_watchlist_wf_recent_reasons",
        "_quality_risk_mult",
        "_quality_score",
        "_loss_bucket",
        "_entry_quality_issue",
        "_bracket_issue",
        "_target_issue",
    ]
    trade_fields.extend([f"po__{k}" for k in po_keys])
    _write_csv(analysis_dir / "trades_enriched.csv", trade_fields, trades)

    # Month-by-month summary.
    by_month: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        by_month[str(t.get("_month") or "")].append(t)
    month_rows: List[Dict[str, Any]] = []

    # Month equity return from backtest_monthly.csv (preferred; exact), then daily curve fallback.
    month_equity: Dict[str, Dict[str, Any]] = {}
    monthly_bt_path = run_dir / "backtest_monthly.csv"
    monthly_bt_rows = _read_csv_rows(monthly_bt_path) if monthly_bt_path.exists() else []
    if monthly_bt_rows:
        for row in monthly_bt_rows:
            m = _iso_month(row.get("month") or "")
            if not m:
                continue
            start_eq = _safe_float(row.get("start_equity"))
            end_eq = _safe_float(row.get("end_equity"))
            month_equity[m] = {
                "start_equity": start_eq,
                "end_equity": end_eq,
                "equity_change": end_eq - start_eq,
                "equity_return_pct": ((end_eq - start_eq) / start_eq * 100.0) if start_eq > 0 else None,
            }
    elif daily_rows:
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
                "start_equity": eq.get("start_equity"),
                "end_equity": eq.get("end_equity"),
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
            "start_equity",
            "end_equity",
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

    # Entry-quality surfaces (historical, no lookahead): which minute entries work better.
    entry_time_rows: List[Dict[str, Any]] = []
    entry_time_symbol_rows: List[Dict[str, Any]] = []

    def _entry_quality_stats(ts: List[Dict[str, Any]]) -> Dict[str, Any]:
        a = _agg_trades(ts)
        n = int(a.get("trades") or 0)
        if n <= 0:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "avgR": 0.0,
                "pnl_total": 0.0,
                "stop_rate": 0.0,
                "target_rate": 0.0,
                "cutoff_rate": 0.0,
                "stop_no_progress_share": 0.0,
                "cutoff_no_progress_share": 0.0,
                "stop_shakeout_share": 0.0,
                "avg_target_r": None,
                "avg_stop_atr": None,
                "avg_mfe_r_full": None,
                "avg_abs_mae_r_full": None,
                "mfe_to_mae_ratio": None,
            }
        stop_count = int(a.get("stops") or 0)
        target_count = int(a.get("targets") or 0)
        cutoff_count = sum(1 for t in ts if str(t.get("exit_reason") or "") in _CUTOFF_EXIT_REASONS)
        stop_no_prog = sum(
            1
            for t in ts
            if str(t.get("exit_reason") or "") == "stop"
            and t.get("mfe_r_before_stop") is not None
            and _safe_float(t.get("mfe_r_before_stop")) <= 1e-9
        )
        cutoff_no_prog = sum(
            1
            for t in ts
            if str(t.get("exit_reason") or "") in _CUTOFF_EXIT_REASONS
            and t.get("mfe_r_full") is not None
            and _safe_float(t.get("mfe_r_full")) <= 1e-9
        )
        stop_shakeout_count = sum(
            1
            for t in ts
            if str(t.get("exit_reason") or "") == "stop"
            and t.get("_stop_shakeout_day_hit_target") is True
        )
        avg_target_r = _mean([_safe_float(t.get("_target_r")) for t in ts if t.get("_target_r") is not None])
        avg_stop_atr = _mean([_safe_float(t.get("_stop_atr")) for t in ts if t.get("_stop_atr") is not None])
        avg_mfe_r_full = _mean([_safe_float(t.get("mfe_r_full")) for t in ts if t.get("mfe_r_full") is not None])
        avg_abs_mae_r_full = _mean(
            [abs(_safe_float(t.get("mae_r_full"))) for t in ts if t.get("mae_r_full") is not None]
        )
        mfe_to_mae_ratio = None
        if avg_mfe_r_full is not None and avg_abs_mae_r_full is not None and avg_abs_mae_r_full > 1e-9:
            mfe_to_mae_ratio = float(avg_mfe_r_full) / float(avg_abs_mae_r_full)
        return {
            "trades": n,
            "win_rate": a["win_rate"],
            "avgR": a["avgR"],
            "pnl_total": a["pnl_total"],
            "stop_rate": stop_count / float(n),
            "target_rate": target_count / float(n),
            "cutoff_rate": cutoff_count / float(n),
            "stop_no_progress_share": (stop_no_prog / float(stop_count)) if stop_count > 0 else 0.0,
            "cutoff_no_progress_share": (cutoff_no_prog / float(cutoff_count)) if cutoff_count > 0 else 0.0,
            "stop_shakeout_share": (stop_shakeout_count / float(stop_count)) if stop_count > 0 else 0.0,
            "avg_target_r": avg_target_r,
            "avg_stop_atr": avg_stop_atr,
            "avg_mfe_r_full": avg_mfe_r_full,
            "avg_abs_mae_r_full": avg_abs_mae_r_full,
            "mfe_to_mae_ratio": mfe_to_mae_ratio,
        }

    by_entry_time: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_symbol_entry_time: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        et = str(t.get("entry_time_et") or "")
        sym = str(t.get("symbol") or "").upper()
        if et:
            by_entry_time[et].append(t)
            if sym:
                by_symbol_entry_time[(sym, et)].append(t)

    for et, ts in by_entry_time.items():
        row = {"entry_time_et": et}
        row.update(_entry_quality_stats(ts))
        entry_time_rows.append(row)
    entry_time_rows.sort(key=lambda r: str(r.get("entry_time_et") or ""))
    if entry_time_rows:
        _write_csv(
            analysis_dir / "entry_time_quality.csv",
            [
                "entry_time_et",
                "trades",
                "win_rate",
                "avgR",
                "pnl_total",
                "stop_rate",
                "target_rate",
                "cutoff_rate",
                "stop_no_progress_share",
                "cutoff_no_progress_share",
                "stop_shakeout_share",
                "avg_target_r",
                "avg_stop_atr",
                "avg_mfe_r_full",
                "avg_abs_mae_r_full",
                "mfe_to_mae_ratio",
            ],
            entry_time_rows,
        )

    for (sym, et), ts in by_symbol_entry_time.items():
        row = {
            "symbol": sym,
            "entry_time_et": et,
        }
        row.update(_entry_quality_stats(ts))
        entry_time_symbol_rows.append(row)
    entry_time_symbol_rows.sort(key=lambda r: (str(r.get("symbol") or ""), -int(r.get("trades") or 0), str(r.get("entry_time_et") or "")))
    if entry_time_symbol_rows:
        _write_csv(
            analysis_dir / "entry_time_symbol_quality.csv",
            [
                "symbol",
                "entry_time_et",
                "trades",
                "win_rate",
                "avgR",
                "pnl_total",
                "stop_rate",
                "target_rate",
                "cutoff_rate",
                "stop_no_progress_share",
                "cutoff_no_progress_share",
                "stop_shakeout_share",
                "avg_target_r",
                "avg_stop_atr",
                "avg_mfe_r_full",
                "avg_abs_mae_r_full",
                "mfe_to_mae_ratio",
            ],
            entry_time_symbol_rows,
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

    # Per-symbol equity-curve proxies (cumulative PnL and cumulative R).
    symbol_month_rows: List[Dict[str, Any]] = []
    symbol_curve_rows: List[Dict[str, Any]] = []
    for sym, ts in by_sym.items():
        # Trade-by-trade curve (sorted by entry date/time).
        ts_sorted = sorted(ts, key=lambda t: (str(t.get("entry_date") or ""), str(t.get("entry_time_et") or "")))
        cum_pnl = 0.0
        cum_r = 0.0
        for t in ts_sorted:
            pnl = _safe_float(t.get("pnl_total"))
            r = _safe_float(t.get("r_multiple"))
            cum_pnl += pnl
            cum_r += r
            symbol_curve_rows.append(
                {
                    "symbol": sym,
                    "entry_date": str(t.get("entry_date") or ""),
                    "entry_time_et": str(t.get("entry_time_et") or ""),
                    "exit_reason": str(t.get("exit_reason") or ""),
                    "r_multiple": r,
                    "pnl_total": pnl,
                    "cum_pnl_total": cum_pnl,
                    "cum_r": cum_r,
                    "param_overrides_json": str(t.get("param_overrides_json") or ""),
                }
            )

        # Month-by-month curve.
        by_sym_month: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in ts:
            by_sym_month[str(t.get("_month") or "")].append(t)
        cum_month_pnl = 0.0
        for m in sorted(by_sym_month.keys()):
            a = _agg_trades(by_sym_month[m])
            cum_month_pnl += float(a.get("pnl_total") or 0.0)
            symbol_month_rows.append(
                {
                    "symbol": sym,
                    "month": m,
                    "trades": a["trades"],
                    "win_rate": a["win_rate"],
                    "avgR": a["avgR"],
                    "pnl_total": a["pnl_total"],
                    "cum_pnl_total": cum_month_pnl,
                    "stops": a["stops"],
                    "targets": a["targets"],
                    "time_stop": a["time_stop"],
                }
            )
    if symbol_month_rows:
        _write_csv(
            analysis_dir / "symbol_monthly.csv",
            ["symbol", "month", "trades", "win_rate", "avgR", "pnl_total", "cum_pnl_total", "stops", "targets", "time_stop"],
            symbol_month_rows,
        )
    if symbol_curve_rows:
        _write_csv(
            analysis_dir / "symbol_equity_curve.csv",
            [
                "symbol",
                "entry_date",
                "entry_time_et",
                "exit_reason",
                "r_multiple",
                "pnl_total",
                "cum_pnl_total",
                "cum_r",
                "param_overrides_json",
            ],
            symbol_curve_rows,
        )

    # Per-symbol month-over-month stability (is the symbol curve steadily rising or choppy?).
    symbol_stability_rows: List[Dict[str, Any]] = []
    for sym, ts in by_sym.items():
        by_sym_month: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in ts:
            by_sym_month[str(t.get("_month") or "")].append(t)
        months_sorted = sorted([m for m in by_sym_month.keys() if m])
        if not months_sorted:
            continue
        month_pnls: List[float] = []
        cum_pnl = 0.0
        peak_pnl = 0.0
        max_monthly_drawdown = 0.0
        for m in months_sorted:
            a = _agg_trades(by_sym_month[m])
            pnl = float(a.get("pnl_total") or 0.0)
            month_pnls.append(pnl)
            cum_pnl += pnl
            if cum_pnl > peak_pnl:
                peak_pnl = cum_pnl
            drawdown = peak_pnl - cum_pnl
            if drawdown > max_monthly_drawdown:
                max_monthly_drawdown = drawdown
        positive_months = sum(1 for v in month_pnls if v > 0)
        non_negative_months = sum(1 for v in month_pnls if v >= 0)
        negative_months = sum(1 for v in month_pnls if v < 0)
        symbol_stability_rows.append(
            {
                "symbol": sym,
                "months": len(month_pnls),
                "positive_months": positive_months,
                "non_negative_months": non_negative_months,
                "negative_months": negative_months,
                "monthly_up_rate": (positive_months / float(len(month_pnls))) if month_pnls else 0.0,
                "cum_pnl_end": sum(month_pnls),
                "worst_month_pnl": min(month_pnls) if month_pnls else 0.0,
                "best_month_pnl": max(month_pnls) if month_pnls else 0.0,
                "max_monthly_drawdown": max_monthly_drawdown,
                "longest_negative_streak": _longest_negative_streak(month_pnls),
            }
        )
    if symbol_stability_rows:
        symbol_stability_rows.sort(
            key=lambda r: (
                -float(r.get("monthly_up_rate") or 0.0),
                -float(r.get("cum_pnl_end") or 0.0),
                float(r.get("max_monthly_drawdown") or 0.0),
            )
        )
        _write_csv(
            analysis_dir / "symbol_stability.csv",
            [
                "symbol",
                "months",
                "positive_months",
                "non_negative_months",
                "negative_months",
                "monthly_up_rate",
                "cum_pnl_end",
                "worst_month_pnl",
                "best_month_pnl",
                "max_monthly_drawdown",
                "longest_negative_streak",
            ],
            symbol_stability_rows,
        )

    # Per-symbol parameter-group summary (uses persisted param_overrides from trade records).
    by_param: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        sym = str(t.get("symbol") or "").upper()
        key = str(t.get("param_overrides_json") or "")
        if sym:
            by_param[(sym, key)].append(t)
    param_rows: List[Dict[str, Any]] = []
    for (sym, key), ts in by_param.items():
        a = _agg_trades(ts)
        param_rows.append(
            {
                "symbol": sym,
                "param_overrides_json": key,
                "trades": a["trades"],
                "win_rate": a["win_rate"],
                "avgR": a["avgR"],
                "pnl_total": a["pnl_total"],
                "stops": a["stops"],
                "targets": a["targets"],
                "time_stop": a["time_stop"],
            }
        )
    if param_rows:
        param_rows.sort(key=lambda r: float(r.get("pnl_total") or 0.0))
        _write_csv(
            analysis_dir / "symbol_param_groups.csv",
            ["symbol", "param_overrides_json", "trades", "win_rate", "avgR", "pnl_total", "stops", "targets", "time_stop"],
            param_rows,
        )

    # Per-symbol "what would make trades win" summaries (non-directional).
    symbol_flip_summary_rows: List[Dict[str, Any]] = []
    symbol_flip_target_grid_rows: List[Dict[str, Any]] = []
    symbol_flip_stop_grid_rows: List[Dict[str, Any]] = []
    for sym, ts in by_sym.items():
        n = len(ts)
        if n <= 0:
            continue
        base_total_r = sum(_safe_float(t.get("r_multiple")) for t in ts)
        base_win_rate = sum(1 for t in ts if _safe_float(t.get("r_multiple")) > 0) / float(n)
        base_avg_r = base_total_r / float(n)
        base_pnl_total = sum(_safe_float(t.get("pnl_total")) for t in ts)
        stopouts_sym = [t for t in ts if str(t.get("exit_reason") or "") == "stop"]
        cutoff_losers_sym = [
            t
            for t in ts
            if str(t.get("exit_reason") or "") in _CUTOFF_EXIT_REASONS and _safe_float(t.get("r_multiple")) <= 0
        ]
        stop_mfe_before_vals = [
            _safe_float(t.get("_flip_target_r_max_before_stop"))
            for t in stopouts_sym
            if t.get("_flip_target_r_max_before_stop") is not None
        ]
        stop_mult_needed_vals = [
            _safe_float(t.get("_flip_stop_mult_needed_for_original_target"))
            for t in stopouts_sym
            if t.get("_flip_stop_mult_needed_for_original_target") is not None
            and _safe_float(t.get("_flip_stop_mult_needed_for_original_target")) > 0
        ]
        cutoff_mfe_vals = [
            _safe_float(t.get("_flip_target_r_max_by_cutoff"))
            for t in cutoff_losers_sym
            if t.get("_flip_target_r_max_by_cutoff") is not None
        ]

        hard_nonflippable_stopouts = 0
        for t in stopouts_sym:
            pre = _safe_float(t.get("_flip_target_r_max_before_stop"), default=0.0)
            need = _safe_float(t.get("_flip_stop_mult_needed_for_original_target"), default=0.0)
            if pre <= 0 and need <= 0:
                hard_nonflippable_stopouts += 1

        symbol_flip_summary_rows.append(
            {
                "symbol": sym,
                "trades": n,
                "pnl_total": base_pnl_total,
                "avgR": base_avg_r,
                "stopouts": len(stopouts_sym),
                "cutoff_losers": len(cutoff_losers_sym),
                "stopouts_flipable_any_target": sum(1 for v in stop_mfe_before_vals if v > 0),
                "stopouts_flipable_target_025R": sum(1 for v in stop_mfe_before_vals if v >= 0.25),
                "stopouts_flipable_target_050R": sum(1 for v in stop_mfe_before_vals if v >= 0.50),
                "stopouts_flipable_target_075R": sum(1 for v in stop_mfe_before_vals if v >= 0.75),
                "stopouts_flipable_target_100R": sum(1 for v in stop_mfe_before_vals if v >= 1.00),
                "stopouts_reach_original_target_with_wider_stop": len(stop_mult_needed_vals),
                "stop_mult_needed_p50": _pct(stop_mult_needed_vals, 0.50),
                "stop_mult_needed_p75": _pct(stop_mult_needed_vals, 0.75),
                "cutoff_losers_flipable_target_025R": sum(1 for v in cutoff_mfe_vals if v >= 0.25),
                "cutoff_losers_flipable_target_050R": sum(1 for v in cutoff_mfe_vals if v >= 0.50),
                "cutoff_losers_flipable_target_075R": sum(1 for v in cutoff_mfe_vals if v >= 0.75),
                "hard_nonflippable_stopouts": hard_nonflippable_stopouts,
            }
        )

        for target_r in _TARGET_R_GRID:
            stopouts_flipped = 0
            cutoff_flipped = 0
            cf_total_r = 0.0
            cf_wins = 0
            for t in ts:
                r_orig = _safe_float(t.get("r_multiple"))
                r_cf = r_orig
                exit_reason = str(t.get("exit_reason") or "")
                if exit_reason == "stop":
                    max_pre = _safe_float(t.get("_flip_target_r_max_before_stop"), default=-1.0)
                    if max_pre >= target_r:
                        r_cf = target_r
                        stopouts_flipped += 1
                elif exit_reason in _CUTOFF_EXIT_REASONS and r_orig <= 0:
                    max_cutoff = _safe_float(t.get("_flip_target_r_max_by_cutoff"), default=-1.0)
                    if max_cutoff >= target_r:
                        r_cf = target_r
                        cutoff_flipped += 1
                cf_total_r += r_cf
                if r_cf > 0:
                    cf_wins += 1
            cf_avg_r = cf_total_r / float(n)
            cf_win_rate = cf_wins / float(n)
            symbol_flip_target_grid_rows.append(
                {
                    "symbol": sym,
                    "target_r": target_r,
                    "trades": n,
                    "stopouts": len(stopouts_sym),
                    "stopouts_flipped": stopouts_flipped,
                    "stopouts_flip_share": (stopouts_flipped / float(len(stopouts_sym))) if stopouts_sym else 0.0,
                    "cutoff_losers": len(cutoff_losers_sym),
                    "cutoff_flipped": cutoff_flipped,
                    "cutoff_flip_share": (cutoff_flipped / float(len(cutoff_losers_sym))) if cutoff_losers_sym else 0.0,
                    "base_total_r": base_total_r,
                    "cf_total_r": cf_total_r,
                    "delta_total_r": cf_total_r - base_total_r,
                    "base_avgR": base_avg_r,
                    "cf_avgR": cf_avg_r,
                    "delta_avgR": cf_avg_r - base_avg_r,
                    "base_win_rate": base_win_rate,
                    "cf_win_rate": cf_win_rate,
                }
            )

        for stop_mult in _STOP_MULT_GRID:
            stopouts_flipped = 0
            cf_total_r = 0.0
            cf_wins = 0
            for t in ts:
                r_orig = _safe_float(t.get("r_multiple"))
                r_cf = r_orig
                if str(t.get("exit_reason") or "") == "stop":
                    need = _safe_float(t.get("_flip_stop_mult_needed_for_original_target"), default=0.0)
                    target_r = _safe_float(t.get("_target_r"), default=0.0)
                    if need > 0 and target_r > 0 and need <= stop_mult:
                        r_cf = target_r
                        stopouts_flipped += 1
                cf_total_r += r_cf
                if r_cf > 0:
                    cf_wins += 1
            cf_avg_r = cf_total_r / float(n)
            cf_win_rate = cf_wins / float(n)
            symbol_flip_stop_grid_rows.append(
                {
                    "symbol": sym,
                    "stop_mult": stop_mult,
                    "trades": n,
                    "stopouts": len(stopouts_sym),
                    "stopouts_flipped": stopouts_flipped,
                    "stopouts_flip_share": (stopouts_flipped / float(len(stopouts_sym))) if stopouts_sym else 0.0,
                    "base_total_r": base_total_r,
                    "cf_total_r": cf_total_r,
                    "delta_total_r": cf_total_r - base_total_r,
                    "base_avgR": base_avg_r,
                    "cf_avgR": cf_avg_r,
                    "delta_avgR": cf_avg_r - base_avg_r,
                    "base_win_rate": base_win_rate,
                    "cf_win_rate": cf_win_rate,
                }
            )

    if symbol_flip_summary_rows:
        symbol_flip_summary_rows.sort(key=lambda r: float(r.get("pnl_total") or 0.0))
        _write_csv(
            analysis_dir / "symbol_flip_summary.csv",
            [
                "symbol",
                "trades",
                "pnl_total",
                "avgR",
                "stopouts",
                "cutoff_losers",
                "stopouts_flipable_any_target",
                "stopouts_flipable_target_025R",
                "stopouts_flipable_target_050R",
                "stopouts_flipable_target_075R",
                "stopouts_flipable_target_100R",
                "stopouts_reach_original_target_with_wider_stop",
                "stop_mult_needed_p50",
                "stop_mult_needed_p75",
                "cutoff_losers_flipable_target_025R",
                "cutoff_losers_flipable_target_050R",
                "cutoff_losers_flipable_target_075R",
                "hard_nonflippable_stopouts",
            ],
            symbol_flip_summary_rows,
        )
    if symbol_flip_target_grid_rows:
        symbol_flip_target_grid_rows.sort(key=lambda r: (str(r.get("symbol") or ""), float(r.get("target_r") or 0.0)))
        _write_csv(
            analysis_dir / "symbol_flip_target_grid.csv",
            [
                "symbol",
                "target_r",
                "trades",
                "stopouts",
                "stopouts_flipped",
                "stopouts_flip_share",
                "cutoff_losers",
                "cutoff_flipped",
                "cutoff_flip_share",
                "base_total_r",
                "cf_total_r",
                "delta_total_r",
                "base_avgR",
                "cf_avgR",
                "delta_avgR",
                "base_win_rate",
                "cf_win_rate",
            ],
            symbol_flip_target_grid_rows,
        )
    if symbol_flip_stop_grid_rows:
        symbol_flip_stop_grid_rows.sort(key=lambda r: (str(r.get("symbol") or ""), float(r.get("stop_mult") or 0.0)))
        _write_csv(
            analysis_dir / "symbol_flip_stop_grid.csv",
            [
                "symbol",
                "stop_mult",
                "trades",
                "stopouts",
                "stopouts_flipped",
                "stopouts_flip_share",
                "base_total_r",
                "cf_total_r",
                "delta_total_r",
                "base_avgR",
                "cf_avgR",
                "delta_avgR",
                "base_win_rate",
                "cf_win_rate",
            ],
            symbol_flip_stop_grid_rows,
        )

    # Loss attribution buckets: entry quality vs bracket/target geometry.
    loss_bucket_rows: List[Dict[str, Any]] = []
    symbol_loss_rows: List[Dict[str, Any]] = []
    by_loss_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        if _safe_float(t.get("r_multiple"), default=0.0) <= 0:
            by_loss_bucket[str(t.get("_loss_bucket") or "")].append(t)
    total_losers = sum(len(v) for v in by_loss_bucket.values())
    for bucket, ts in by_loss_bucket.items():
        n = len(ts)
        loss_bucket_rows.append(
            {
                "loss_bucket": bucket,
                "trades": n,
                "share_of_losers": (n / float(total_losers)) if total_losers > 0 else 0.0,
                "avgR": _mean([_safe_float(t.get("r_multiple")) for t in ts]) or 0.0,
                "pnl_total": sum(_safe_float(t.get("pnl_total")) for t in ts),
            }
        )
    if loss_bucket_rows:
        loss_bucket_rows.sort(key=lambda r: float(r.get("trades") or 0), reverse=True)
        _write_csv(
            analysis_dir / "loss_bucket_summary.csv",
            ["loss_bucket", "trades", "share_of_losers", "avgR", "pnl_total"],
            loss_bucket_rows,
        )

    for sym, ts in by_sym.items():
        losses = [t for t in ts if _safe_float(t.get("r_multiple"), default=0.0) <= 0]
        if not losses:
            continue
        c = Counter(str(t.get("_loss_bucket") or "") for t in losses)
        n = len(losses)
        top_bucket, top_count = c.most_common(1)[0]
        symbol_loss_rows.append(
            {
                "symbol": sym,
                "trades": len(ts),
                "losses": n,
                "pnl_total": sum(_safe_float(t.get("pnl_total")) for t in ts),
                "avgR": _mean([_safe_float(t.get("r_multiple")) for t in ts]) or 0.0,
                "entry_issue_share": sum(_safe_int(t.get("_entry_quality_issue"), default=0) for t in losses) / float(n),
                "bracket_issue_share": sum(_safe_int(t.get("_bracket_issue"), default=0) for t in losses) / float(n),
                "target_issue_share": sum(_safe_int(t.get("_target_issue"), default=0) for t in losses) / float(n),
                "stop_no_progress": c.get("stop_no_progress", 0),
                "stop_midpath_reversal": c.get("stop_midpath_reversal", 0),
                "stop_recoverable_with_wider_stop": c.get("stop_recoverable_with_wider_stop", 0),
                "stop_near_target": c.get("stop_near_target", 0),
                "stop_near_target_recoverable": c.get("stop_near_target_recoverable", 0),
                "cutoff_no_progress": c.get("cutoff_no_progress", 0),
                "cutoff_partial_progress": c.get("cutoff_partial_progress", 0),
                "cutoff_near_target_time_limited": c.get("cutoff_near_target_time_limited", 0),
                "cutoff_no_edge": c.get("cutoff_no_edge", 0),
                "top_loss_bucket": top_bucket,
                "top_loss_bucket_share": top_count / float(n),
            }
        )
    if symbol_loss_rows:
        symbol_loss_rows.sort(key=lambda r: float(r.get("pnl_total") or 0.0))
        _write_csv(
            analysis_dir / "symbol_loss_attribution.csv",
            [
                "symbol",
                "trades",
                "losses",
                "pnl_total",
                "avgR",
                "entry_issue_share",
                "bracket_issue_share",
                "target_issue_share",
                "stop_no_progress",
                "stop_midpath_reversal",
                "stop_recoverable_with_wider_stop",
                "stop_near_target",
                "stop_near_target_recoverable",
                "cutoff_no_progress",
                "cutoff_partial_progress",
                "cutoff_near_target_time_limited",
                "cutoff_no_edge",
                "top_loss_bucket",
                "top_loss_bucket_share",
            ],
            symbol_loss_rows,
        )

    # Trade-by-trade "flip" table (safe thresholds).
    flip_rows: List[Dict[str, Any]] = []
    for t in trades:
        if (
            t.get("_flip_target_r_max_before_stop") is None
            and t.get("_flip_stop_mult_needed_for_original_target") is None
            and t.get("_flip_target_r_max_by_cutoff") is None
        ):
            continue
        flip_rows.append(
            {
                "entry_date": t.get("entry_date"),
                "symbol": t.get("symbol"),
                "direction": t.get("direction"),
                "entry_time_et": t.get("entry_time_et"),
                "param_overrides_json": t.get("param_overrides_json"),
                "exit_reason": t.get("exit_reason"),
                "exit_ts": t.get("exit_ts"),
                "stop_hit_ts": t.get("stop_hit_ts"),
                "target_hit_ts": t.get("target_hit_ts"),
                "r_multiple": t.get("r_multiple"),
                "pnl_total": t.get("pnl_total"),
                "_target_r": t.get("_target_r"),
                "mfe_r_before_stop": t.get("mfe_r_before_stop"),
                "mae_r_to_target": t.get("mae_r_to_target"),
                "mfe_r_full": t.get("mfe_r_full"),
                "mae_r_full": t.get("mae_r_full"),
                "_flip_target_r_max_before_stop": t.get("_flip_target_r_max_before_stop"),
                "_flip_stop_mult_needed_for_original_target": t.get("_flip_stop_mult_needed_for_original_target"),
                "_flip_target_r_max_by_cutoff": t.get("_flip_target_r_max_by_cutoff"),
            }
        )
    if flip_rows:
        _write_csv(
            analysis_dir / "trade_flip.csv",
            [
                "entry_date",
                "symbol",
                "direction",
                "entry_time_et",
                "param_overrides_json",
                "exit_reason",
                "exit_ts",
                "stop_hit_ts",
                "target_hit_ts",
                "r_multiple",
                "pnl_total",
                "_target_r",
                "mfe_r_before_stop",
                "mae_r_to_target",
                "mfe_r_full",
                "mae_r_full",
                "_flip_target_r_max_before_stop",
                "_flip_stop_mult_needed_for_original_target",
                "_flip_target_r_max_by_cutoff",
            ],
            flip_rows,
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
    if watchlist_funnel_summary is not None:
        lines.append("\n## Watchlist Funnel\n")
        lines.append("See `watchlist_funnel_daily.csv` and `watchlist_funnel_summary.csv`.\n")
        lines.append(
            f"- days={int(watchlist_funnel_summary.get('days') or 0)} "
            f"avg_scanned={_format_float(watchlist_funnel_summary.get('avg_scanned_symbols'))} "
            f"avg_passed={_format_float(watchlist_funnel_summary.get('avg_symbols_passing_filters'))} "
            f"avg_watchlist={_format_float(watchlist_funnel_summary.get('avg_watchlist_size'))}\n"
        )
        lines.append(
            f"- top_k binding days: {int(watchlist_funnel_summary.get('days_bound_by_top_k') or 0)}/"
            f"{int(watchlist_funnel_summary.get('days') or 0)} "
            f"({float(watchlist_funnel_summary.get('bound_by_top_k_share') or 0.0):.3f})\n"
        )
        top_reject_nonzero = [r for r in watchlist_reject_top if int(r[1]) > 0]
        if top_reject_nonzero:
            lines.append("- largest reject drivers:\n")
            for key, cnt in top_reject_nonzero[:5]:
                lines.append(f"  {key}={int(cnt)}\n")

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

    if entry_time_rows:
        lines.append("\n## Entry-Time Quality\n")
        lines.append("See `entry_time_quality.csv` and `entry_time_symbol_quality.csv`.\n")
        qualified_entry_times = [r for r in entry_time_rows if int(r.get("trades") or 0) >= 20]
        if not qualified_entry_times:
            qualified_entry_times = list(entry_time_rows)
        best_entry_times = sorted(
            qualified_entry_times,
            key=lambda r: (float(r.get("avgR") or 0.0), float(r.get("win_rate") or 0.0)),
            reverse=True,
        )[:5]
        worst_entry_times = sorted(
            qualified_entry_times,
            key=lambda r: (float(r.get("avgR") or 0.0), float(r.get("win_rate") or 0.0)),
        )[:5]
        if best_entry_times:
            lines.append("\nBest entry minutes:\n")
            for r in best_entry_times:
                lines.append(
                    f"- {r['entry_time_et']}: trades={int(r['trades'])} win_rate={float(r['win_rate']):.3f} "
                    f"avgR={float(r['avgR']):.3f} stop_rate={float(r['stop_rate']):.3f} "
                    f"cutoff_rate={float(r['cutoff_rate']):.3f}\n"
                )
        if worst_entry_times:
            lines.append("\nWeakest entry minutes:\n")
            for r in worst_entry_times:
                lines.append(
                    f"- {r['entry_time_et']}: trades={int(r['trades'])} win_rate={float(r['win_rate']):.3f} "
                    f"avgR={float(r['avgR']):.3f} stop_rate={float(r['stop_rate']):.3f} "
                    f"cutoff_rate={float(r['cutoff_rate']):.3f}\n"
                )

    if symbol_stability_rows:
        lines.append("\n## Symbol Stability\n")
        lines.append("See `symbol_stability.csv`, `symbol_monthly.csv`, and `symbol_equity_curve.csv`.\n")
        unstable = sorted(
            symbol_stability_rows,
            key=lambda r: (
                -float(r.get("negative_months") or 0.0),
                -float(r.get("max_monthly_drawdown") or 0.0),
                float(r.get("cum_pnl_end") or 0.0),
            ),
        )
        lines.append("\nMost unstable symbols:\n")
        for r in unstable[:8]:
            lines.append(
                f"- {r['symbol']}: months={int(r['months'])} pos={int(r['positive_months'])} neg={int(r['negative_months'])} "
                f"cum_pnl=${float(r['cum_pnl_end']):.2f} max_monthly_dd=${float(r['max_monthly_drawdown']):.2f} "
                f"longest_neg_streak={int(r['longest_negative_streak'])}\n"
            )

    if symbol_flip_summary_rows:
        lines.append("\n## Flipability Snapshot\n")
        lines.append(
            "See `trade_flip.csv`, `symbol_flip_summary.csv`, `symbol_flip_target_grid.csv`, and `symbol_flip_stop_grid.csv`.\n"
        )
        t050 = [
            r
            for r in symbol_flip_target_grid_rows
            if abs(float(r.get("target_r") or 0.0) - 0.50) < 1e-9 and float(r.get("delta_total_r") or 0.0) > 0
        ]
        t050.sort(key=lambda r: float(r.get("delta_total_r") or 0.0), reverse=True)
        if t050:
            lines.append("\nLargest upside if target were 0.50R (counterfactual):\n")
            for r in t050[:8]:
                lines.append(
                    f"- {r['symbol']}: delta_total_r={float(r['delta_total_r']):.3f} "
                    f"stop_flips={int(r['stopouts_flipped'])}/{int(r['stopouts'])} "
                    f"cutoff_flips={int(r['cutoff_flipped'])}/{int(r['cutoff_losers'])}\n"
                )
        s150 = [
            r
            for r in symbol_flip_stop_grid_rows
            if abs(float(r.get("stop_mult") or 0.0) - 1.50) < 1e-9 and float(r.get("delta_total_r") or 0.0) > 0
        ]
        s150.sort(key=lambda r: float(r.get("delta_total_r") or 0.0), reverse=True)
        if s150:
            lines.append("\nLargest upside if stop were widened to 1.50x (counterfactual):\n")
            for r in s150[:8]:
                lines.append(
                    f"- {r['symbol']}: delta_total_r={float(r['delta_total_r']):.3f} "
                    f"flipped_stopouts={int(r['stopouts_flipped'])}/{int(r['stopouts'])}\n"
                )
    if loss_bucket_rows:
        lines.append("\n## Loss Attribution\n")
        lines.append("See `loss_bucket_summary.csv` and `symbol_loss_attribution.csv`.\n")
        for row in loss_bucket_rows[:8]:
            lines.append(
                f"- {row['loss_bucket']}: trades={int(row['trades'])} share={float(row['share_of_losers']):.3f} "
                f"avgR={float(row['avgR']):.3f} pnl_total=${float(row['pnl_total']):.2f}\n"
            )
        if symbol_loss_rows:
            weakest = sorted(symbol_loss_rows, key=lambda r: float(r.get("pnl_total") or 0.0))[:6]
            lines.append("\nWorst symbols by loss composition:\n")
            for r in weakest:
                lines.append(
                    f"- {r['symbol']}: pnl_total=${float(r['pnl_total']):.2f} "
                    f"entry_issue_share={float(r['entry_issue_share']):.3f} "
                    f"bracket_issue_share={float(r['bracket_issue_share']):.3f} "
                    f"target_issue_share={float(r['target_issue_share']):.3f} "
                    f"top_bucket={r['top_loss_bucket']}\n"
                )

    # Put actionable diagnostics at the end.
    lines.append("\n## Notes For Tuning\n")
    lines.append("- A large fraction of stopouts have `mfe_r==0` but the day later reaches the target distance.\n")
    lines.append("  That pattern is consistent with entering too early (before reversal starts) and/or needing entry confirmation.\n")
    lines.append("- Trades with very small stop_atr tend to have worse expectancy; consider enforcing a minimum stop distance vs ATR.\n")
    lines.append("- Use `symbol_stability.csv` to cap/remove symbols with repeated negative-month streaks and high monthly drawdown.\n")
    lines.append("- Use `symbol_flip_target_grid.csv` and `symbol_flip_stop_grid.csv` for per-symbol tuning (target/stop), not one-size-fits-all settings.\n")

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
