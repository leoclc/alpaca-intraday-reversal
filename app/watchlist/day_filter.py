from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def summarize_watchlist_rows(rows: List[Dict]) -> Dict[str, float]:
    size = len(rows or [])
    if size <= 0:
        return {
            "size": 0,
            "avg_selected_avgR": 0.0,
            "avg_selected_win_rate": 0.0,
            "avg_selected_profit_factor": 0.0,
            "avg_selected_total_pnl_pct": 0.0,
            "long_count": 0,
            "short_count": 0,
        }
    avg_r_vals = [_safe_float(r.get("avgR")) for r in rows]
    win_vals = [_safe_float(r.get("win_rate")) for r in rows]
    pf_vals = [_safe_float(r.get("profit_factor")) for r in rows]
    pnl_vals = [_safe_float(r.get("total_pnl_pct")) for r in rows]
    long_count = sum(1 for r in rows if str(r.get("direction") or "").lower() == "long")
    short_count = sum(1 for r in rows if str(r.get("direction") or "").lower() == "short")
    return {
        "size": size,
        "avg_selected_avgR": (sum(avg_r_vals) / float(size)) if size > 0 else 0.0,
        "avg_selected_win_rate": (sum(win_vals) / float(size)) if size > 0 else 0.0,
        "avg_selected_profit_factor": (sum(pf_vals) / float(size)) if size > 0 else 0.0,
        "avg_selected_total_pnl_pct": (sum(pnl_vals) / float(size)) if size > 0 else 0.0,
        "long_count": long_count,
        "short_count": short_count,
    }


def day_filter_decision(rows: List[Dict], cfg: Dict, meta: Optional[Dict] = None) -> Tuple[bool, Dict]:
    watch_cfg = (cfg or {}).get("watchlist") or {}
    enabled = bool(watch_cfg.get("day_kill_switch_enabled", False))
    if not enabled:
        return False, {}

    thresholds = {
        "day_min_watchlist_size": _safe_int(watch_cfg.get("day_min_watchlist_size"), 0),
        "day_min_selected_avgR": watch_cfg.get("day_min_selected_avgR"),
        "day_min_selected_win_rate": watch_cfg.get("day_min_selected_win_rate"),
        "day_min_selected_profit_factor": watch_cfg.get("day_min_selected_profit_factor"),
        "day_min_selected_total_pnl_pct": watch_cfg.get("day_min_selected_total_pnl_pct"),
        "day_min_long_count": _safe_int(watch_cfg.get("day_min_long_count"), 0),
        "day_min_short_count": _safe_int(watch_cfg.get("day_min_short_count"), 0),
    }
    summary = summarize_watchlist_rows(rows)
    # If the builder wrote a selected summary into watchlist metadata, prefer it.
    if isinstance(meta, dict):
        selected = meta.get("selected_summary")
        if isinstance(selected, dict):
            summary = {
                "size": _safe_int(selected.get("size"), summary.get("size", 0)),
                "avg_selected_avgR": _safe_float(selected.get("avg_selected_avgR"), summary.get("avg_selected_avgR", 0.0)),
                "avg_selected_win_rate": _safe_float(
                    selected.get("avg_selected_win_rate"), summary.get("avg_selected_win_rate", 0.0)
                ),
                "avg_selected_profit_factor": _safe_float(
                    selected.get("avg_selected_profit_factor"), summary.get("avg_selected_profit_factor", 0.0)
                ),
                "avg_selected_total_pnl_pct": _safe_float(
                    selected.get("avg_selected_total_pnl_pct"), summary.get("avg_selected_total_pnl_pct", 0.0)
                ),
                "long_count": _safe_int(selected.get("long_count"), summary.get("long_count", 0)),
                "short_count": _safe_int(selected.get("short_count"), summary.get("short_count", 0)),
            }

    reasons: List[str] = []
    if summary["size"] < thresholds["day_min_watchlist_size"]:
        reasons.append("min_watchlist_size")

    try:
        th = thresholds["day_min_selected_avgR"]
        if th is not None and summary["avg_selected_avgR"] < float(th):
            reasons.append("min_selected_avgR")
    except Exception:
        pass
    try:
        th = thresholds["day_min_selected_win_rate"]
        if th is not None and summary["avg_selected_win_rate"] < float(th):
            reasons.append("min_selected_win_rate")
    except Exception:
        pass
    try:
        th = thresholds["day_min_selected_profit_factor"]
        if th is not None and summary["avg_selected_profit_factor"] < float(th):
            reasons.append("min_selected_profit_factor")
    except Exception:
        pass
    try:
        th = thresholds["day_min_selected_total_pnl_pct"]
        if th is not None and summary["avg_selected_total_pnl_pct"] < float(th):
            reasons.append("min_selected_total_pnl_pct")
    except Exception:
        pass

    if summary["long_count"] < thresholds["day_min_long_count"]:
        reasons.append("min_long_count")
    if summary["short_count"] < thresholds["day_min_short_count"]:
        reasons.append("min_short_count")

    info = {"enabled": enabled, "reasons": reasons, "summary": summary, "thresholds": thresholds}
    return len(reasons) > 0, info

