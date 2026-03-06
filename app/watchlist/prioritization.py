from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def _safe_float(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def symbol_priority_tuple(symbol: str, stats: Dict) -> Tuple[float, int, float, float, str]:
    # Higher selection_score is better; lower rank number is better.
    selection_score = _safe_float((stats or {}).get("selection_score"), float("-inf"))
    rank = _safe_int((stats or {}).get("rank"), 10**9)
    avg_r = _safe_float((stats or {}).get("avgR"), float("-inf"))
    trades_count = _safe_float((stats or {}).get("trades_count"), 0.0)
    return (-selection_score, rank, -avg_r, -trades_count, str(symbol or ""))


def sort_symbols_by_watchlist_priority(symbols: Iterable[str], stats_by_symbol: Dict[str, Dict]) -> List[str]:
    return sorted(
        [str(s or "").upper() for s in symbols if s],
        key=lambda s: symbol_priority_tuple(s, stats_by_symbol.get(s) or {}),
    )

