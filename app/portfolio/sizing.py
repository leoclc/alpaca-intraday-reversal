from __future__ import annotations

from typing import Dict, Optional, Tuple

from app.strategies.types import TradePlan


def compute_qty_with_guards(
    plan: TradePlan,
    equity: float,
    used_notional: float,
    cfg: Dict,
    *,
    allowed_total_override: Optional[float] = None,
    open_positions: int = 0,
) -> Tuple[int, Dict]:
    params = cfg.get("daily_trend_reversal") or {}
    max_positions = int(params.get("max_positions") or 0)
    risk_per_trade = float(params.get("risk_per_trade") or 0.0)
    leverage = float(params.get("leverage") or 1.0)
    max_margin_usage = float(params.get("max_margin_usage") or 1.0)
    margin_safety_buffer = float(params.get("margin_safety_buffer") or 0.0)
    per_trade_max_pct_available = float(params.get("per_trade_max_pct_available") or 1.0)
    equal_split = bool(params.get("equal_split_across_max_slots", False))
    min_af_abs = float(params.get("min_available_funds_abs") or 0.0)
    min_af_ratio = float(params.get("min_available_funds_ratio_of_netliq") or 0.0)

    state = {
        "equity": equity,
        "used_notional": used_notional,
        "risk_per_trade": risk_per_trade,
        "leverage": leverage,
        "max_margin_usage": max_margin_usage,
        "margin_safety_buffer": margin_safety_buffer,
        "per_trade_max_pct_available": per_trade_max_pct_available,
        "equal_split_across_max_slots": equal_split,
        "max_positions": max_positions,
        "open_positions": open_positions,
    }

    if max_positions > 0 and open_positions >= max_positions:
        state["reject_reason"] = "max_positions"
        return 0, state

    if equity <= 0:
        state["reject_reason"] = "equity_zero"
        return 0, state

    allowed_total = (
        float(allowed_total_override)
        if allowed_total_override is not None
        else max(0.0, equity * leverage) * max(0.0, min(max_margin_usage, 1.0))
    )
    available = max(0.0, allowed_total - used_notional)
    state["allowed_total"] = allowed_total
    state["available"] = available

    if min_af_abs > 0.0 and available < min_af_abs:
        state["reject_reason"] = "min_available_funds_abs"
        return 0, state
    if min_af_ratio > 0.0 and (available / max(1e-9, equity)) < min_af_ratio:
        state["reject_reason"] = "min_available_funds_ratio"
        return 0, state

    if risk_per_trade <= 0:
        state["reject_reason"] = "risk_per_trade_zero"
        return 0, state

    risk_amt = max(0.0, equity * risk_per_trade)
    state["risk_amount"] = risk_amt
    if plan.stop_distance <= 0:
        state["reject_reason"] = "stop_distance_zero"
        return 0, state

    risk_qty = int(risk_amt // plan.stop_distance) if risk_amt > 0 else 0
    state["risk_qty"] = risk_qty
    if risk_qty <= 0:
        state["reject_reason"] = "risk_qty_zero"
        return 0, state

    msb = max(0.0, min(margin_safety_buffer, 0.9))
    net_avail = available * (1.0 - msb)
    headroom_cap = max(0.0, allowed_total * (1.0 - msb) - used_notional)
    net_avail = min(net_avail, headroom_cap)
    state["net_available"] = net_avail
    state["headroom_cap"] = headroom_cap

    if equal_split and max_positions > 0:
        eq_cap = allowed_total / max_positions
        state["equal_split_cap"] = eq_cap
        net_avail = min(net_avail, eq_cap)

    if per_trade_max_pct_available < 1.0:
        cap_pct = max(0.0, min(per_trade_max_pct_available, 1.0))
        per_trade_cap = available * (1.0 - msb) * cap_pct
        state["per_trade_cap"] = per_trade_cap
        net_avail = min(net_avail, per_trade_cap)

    state["net_available_final"] = net_avail
    budget_qty = int(net_avail // max(1e-9, float(plan.entry_price))) if net_avail > 0 else 0
    state["budget_qty"] = budget_qty

    qty = min(risk_qty, budget_qty)
    state["final_qty"] = qty
    if qty <= 0:
        state["reject_reason"] = "budget_zero"
        return 0, state

    return qty, state
