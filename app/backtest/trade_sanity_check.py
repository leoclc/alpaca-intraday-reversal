from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config.loader import load_config
from app.data.alpaca_ohlc_store import AlpacaOHLCStore


def _within(price: float, low: float, high: float, tol_bps: float) -> bool:
    tol = abs(price) * (tol_bps / 10000.0)
    return (low - tol) <= price <= (high + tol)


def _fetch_bar(
    data_store: AlpacaOHLCStore,
    symbol: str,
    date_str: str,
    cfg: Dict[str, Any],
    cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    key = (symbol, date_str)
    if key in cache:
        return cache[key]
    bars = data_store.get_daily_bars(symbol, date_str, date_str, cfg=cfg, allow_fetch=True)
    bar = bars[0] if bars else None
    cache[key] = bar
    return bar


def run_sanity_check(
    backtest_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    out_path: Optional[str] = None,
    tolerance_bps: float = 0.0,
) -> Dict[str, Any]:
    cfg = cfg or load_config()
    payload = json.loads(Path(backtest_path).read_text(encoding="utf-8"))
    trades = payload.get("trades") or []
    data_store = AlpacaOHLCStore(cfg=cfg)
    cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
    violations: List[Dict[str, Any]] = []
    missing_bars: List[Dict[str, Any]] = []
    checked = 0

    for trade in trades:
        plan = trade.get("plan") or {}
        symbol = str(plan.get("symbol") or "").upper()
        entry_date = str(plan.get("entry_date") or "")
        exit_date = str(trade.get("exit_date") or "")
        if not symbol or not entry_date or not exit_date:
            continue
        entry_bar = _fetch_bar(data_store, symbol, entry_date, cfg, cache)
        exit_bar = _fetch_bar(data_store, symbol, exit_date, cfg, cache)
        if not entry_bar or not exit_bar:
            missing_bars.append({"symbol": symbol, "entry_date": entry_date, "exit_date": exit_date})
            continue

        entry_low = float(entry_bar.get("low") or 0.0)
        entry_high = float(entry_bar.get("high") or 0.0)
        exit_low = float(exit_bar.get("low") or 0.0)
        exit_high = float(exit_bar.get("high") or 0.0)

        entry_price = float(plan.get("entry_price") or 0.0)
        stop_price = float(plan.get("stop_price") or 0.0)
        target_price = float(plan.get("target_price") or 0.0)
        exit_price = float(trade.get("exit_price") or 0.0)

        checked += 1
        issues: List[str] = []
        if entry_price and not _within(entry_price, entry_low, entry_high, tolerance_bps):
            issues.append("entry_price_outside_entry_bar")
        if entry_date == exit_date:
            if stop_price and not _within(stop_price, entry_low, entry_high, tolerance_bps):
                issues.append("stop_price_outside_entry_bar")
            if target_price and not _within(target_price, entry_low, entry_high, tolerance_bps):
                issues.append("target_price_outside_entry_bar")
            if exit_price and not _within(exit_price, entry_low, entry_high, tolerance_bps):
                issues.append("exit_price_outside_entry_bar")
        else:
            if exit_price and not _within(exit_price, exit_low, exit_high, tolerance_bps):
                issues.append("exit_price_outside_exit_bar")

        if issues:
            violations.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_bar": {"low": entry_low, "high": entry_high},
                    "exit_bar": {"low": exit_low, "high": exit_high},
                    "prices": {
                        "entry": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "exit": exit_price,
                    },
                    "issues": issues,
                }
            )

    summary = {
        "trades": len(trades),
        "checked": checked,
        "missing_bars": len(missing_bars),
        "violations": len(violations),
    }
    output = {"summary": summary, "violations": violations, "missing_bars": missing_bars}
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output

