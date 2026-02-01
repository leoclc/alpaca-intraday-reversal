from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    symbol: str
    signal_date: str
    direction: str
    trend_state: str
    return_pct: float


@dataclass
class TradePlan:
    symbol: str
    direction: str
    signal_date: str
    entry_date: str
    entry_time_et: str
    entry_price: float
    stop_price: float
    target_price: float
    time_exit_date: str
    stop_distance: float
    target_rr: float


@dataclass
class TradeResult:
    plan: TradePlan
    exit_date: str
    exit_price: float
    exit_reason: str
    pnl_pct: float
    r_multiple: float
