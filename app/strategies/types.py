from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    target_mode: str | None = None
    target_window_avg_pct: float | None = None
    target_window_mult: float | None = None
    target_window_minutes: int | None = None
    target_window_samples: int | None = None
    gap_bps: float | None = None
    early_pullback_bps: float | None = None
    early_reversal_bps: float | None = None
    open_noise_abs: float | None = None
    open_noise_bps: float | None = None
    open_noise_atr: float | None = None
    open_noise_stop_ratio: float | None = None
    stop_to_open_noise_ratio: float | None = None
    open_noise_window_minutes: int | None = None
    confirm_move_bps: float | None = None
    confirm_minutes: int | None = None
    confirm_hit_bps: float | None = None
    signal_return_pct: float | None = None
    signal_return_atr: float | None = None
    atr: float | None = None
    entry_price_mode: str | None = None
    # Per-symbol watchlist overrides used when constructing this plan (e.g. stop/target/filters).
    # Persisted so backtests can trace each trade to the exact parameter set that selected it.
    param_overrides: dict[str, Any] | None = None
    # Optional watchlist scoring stats used to qualify per-symbol sizing at execution time.
    watchlist_stats: dict[str, Any] | None = None


@dataclass
class TradeResult:
    plan: TradePlan
    exit_date: str
    exit_price: float
    exit_reason: str
    pnl_pct: float
    r_multiple: float
    mfe_pct: float | None = None
    mae_pct: float | None = None
    mfe_r: float | None = None
    mae_r: float | None = None
    exit_ts: str | None = None
    stop_hit_ts: str | None = None
    target_hit_ts: str | None = None
    mfe_r_full: float | None = None
    mae_r_full: float | None = None
    mfe_r_before_stop: float | None = None
    mae_r_to_target: float | None = None
