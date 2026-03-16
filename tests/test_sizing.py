from __future__ import annotations

from copy import deepcopy

from app.config.defaults import DEFAULT_CONFIG
from app.portfolio.sizing import compute_qty_with_guards
from app.strategies.types import TradePlan


def _base_cfg() -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["daily_trend_reversal"].update(
        {
            "risk_per_trade": 0.02,
            "leverage": 4.0,
            "max_margin_usage": 1.0,
            "margin_safety_buffer": 0.0,
            "required_free_margin_buffer": 0.0,
            "per_trade_max_pct_available": 1.0,
            "min_overlap_order_notional": 1000.0,
            "quality_sizing_enabled": False,
        }
    )
    return cfg


def _plan() -> TradePlan:
    return TradePlan(
        symbol="TEST",
        direction="long",
        signal_date="2025-01-02",
        entry_date="2025-01-03",
        entry_time_et="09:35",
        entry_price=50.0,
        stop_price=49.0,
        target_price=51.5,
        time_exit_date="2025-01-04",
        stop_distance=1.0,
        target_rr=1.5,
    )


def test_overlap_notional_floor_rejects_tiny_followup_trade():
    qty, state = compute_qty_with_guards(
        _plan(),
        equity=30_000.0,
        used_notional=119_500.0,
        cfg=_base_cfg(),
        open_positions=1,
    )

    assert qty == 0
    assert state["reject_reason"] == "min_overlap_order_notional"
    assert state["final_order_notional"] == 500.0


def test_overlap_notional_floor_allows_trade_after_first_position_is_closed():
    qty, state = compute_qty_with_guards(
        _plan(),
        equity=30_000.0,
        used_notional=119_500.0,
        cfg=_base_cfg(),
        open_positions=0,
    )

    assert qty == 10
    assert "reject_reason" not in state
    assert state["final_order_notional"] == 500.0


def test_overlap_notional_floor_keeps_meaningful_overlap_trade():
    qty, state = compute_qty_with_guards(
        _plan(),
        equity=30_000.0,
        used_notional=118_000.0,
        cfg=_base_cfg(),
        open_positions=1,
    )

    assert qty == 40
    assert "reject_reason" not in state
    assert state["final_order_notional"] == 2_000.0
