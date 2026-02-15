from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _pct_move_to_atr(move_pct: float, entry_price: float, atr: float) -> Optional[float]:
    if entry_price <= 0 or atr <= 0:
        return None
    return (move_pct / 100.0) * entry_price / atr


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _load_trades(backtest_path: str) -> List[Dict]:
    path = Path(backtest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("sized_trades") or []


def _trade_stats_for_combo(
    trades: List[Dict],
    stop_atr_mult: float,
    target_rr: float,
    stop_first_when_both: bool = True,
) -> Dict[str, float]:
    count = 0
    wins = 0
    r_sum = 0.0
    target_hits = 0
    stop_hits = 0
    time_exits = 0
    for trade in trades:
        entry_price = _safe_float(trade.get("entry_price")) or 0.0
        atr = _safe_float(trade.get("atr"))
        if atr is None:
            continue
        mfe_pct = _safe_float(trade.get("mfe_pct"))
        if mfe_pct is None:
            mfe_pct = _safe_float(trade.get("day_mfe_pct"))
        mae_pct = _safe_float(trade.get("mae_pct"))
        if mae_pct is None:
            mae_pct = _safe_float(trade.get("day_mae_pct"))
        if mfe_pct is None or mae_pct is None:
            continue
        mfe_atr = _pct_move_to_atr(mfe_pct, entry_price, atr)
        mae_atr = _pct_move_to_atr(mae_pct, entry_price, atr)
        if mfe_atr is None or mae_atr is None:
            continue
        mfe_r = mfe_atr / stop_atr_mult
        mae_r = mae_atr / stop_atr_mult
        hit_target = mfe_r >= target_rr
        hit_stop = mae_r <= -1.0
        if hit_target and hit_stop:
            reason = "stop" if stop_first_when_both else "target"
        elif hit_target:
            reason = "target"
        elif hit_stop:
            reason = "stop"
        else:
            reason = "time_stop"
        if reason == "target":
            r = target_rr
            target_hits += 1
        elif reason == "stop":
            r = -1.0
            stop_hits += 1
        else:
            pnl_pct = _safe_float(trade.get("pnl_pct")) or 0.0
            pnl_per_share = entry_price * (pnl_pct / 100.0)
            r = pnl_per_share / (stop_atr_mult * atr) if atr > 0 else 0.0
            time_exits += 1
        r_sum += r
        if r > 0:
            wins += 1
        count += 1
    if count == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avgR": 0.0,
            "target_hit_rate": 0.0,
            "stop_hit_rate": 0.0,
            "time_stop_rate": 0.0,
        }
    return {
        "trades": count,
        "win_rate": wins / float(count),
        "avgR": r_sum / float(count),
        "target_hit_rate": target_hits / float(count),
        "stop_hit_rate": stop_hits / float(count),
        "time_stop_rate": time_exits / float(count),
    }


def run_target_sweep(
    backtest_path: str,
    out_path: Optional[str] = None,
    stop_atr_mult_grid: Optional[Iterable[float]] = None,
    target_rr_grid: Optional[Iterable[float]] = None,
    stop_first_when_both: bool = True,
) -> Path:
    trades = _load_trades(backtest_path)
    stop_atr_mult_grid = stop_atr_mult_grid or [0.4, 0.5, 0.6, 0.7]
    target_rr_grid = target_rr_grid or [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    out_file = Path(out_path) if out_path else Path(backtest_path).with_name("target_sweep.csv")
    rows: List[Dict[str, float]] = []
    for stop_mult in stop_atr_mult_grid:
        for target_rr in target_rr_grid:
            stats = _trade_stats_for_combo(trades, float(stop_mult), float(target_rr), stop_first_when_both)
            rows.append(
                {
                    "stop_atr_mult": float(stop_mult),
                    "target_rr": float(target_rr),
                    **stats,
                }
            )
    # Write CSV
    header = [
        "stop_atr_mult",
        "target_rr",
        "trades",
        "win_rate",
        "avgR",
        "target_hit_rate",
        "stop_hit_rate",
        "time_stop_rate",
    ]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(h, "")) for h in header) + "\n")
    return out_file

