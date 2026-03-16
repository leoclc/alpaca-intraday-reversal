from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from app.strategies.types import TradePlan


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float_opt(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int_opt(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def resolve_effective_margin_buffer(params: Dict[str, Any]) -> float:
    p = params or {}
    configured = _safe_float_opt(p.get("margin_safety_buffer"))
    required = _safe_float_opt(p.get("required_free_margin_buffer"))
    if required is None:
        required = _safe_float_opt(p.get("alpaca_required_free_margin_buffer"))
    if required is None:
        required = 0.30
    cfg_buffer = max(0.0, min(configured if configured is not None else 0.0, 0.9))
    req_buffer = max(0.0, min(required, 0.9))
    return max(cfg_buffer, req_buffer)


def estimate_slot_target_from_stats(
    stats_rows: list[Dict[str, Any]],
    lookback_days: float,
    *,
    min_slots: int = 1,
    max_slots: int = 0,
    cap_by_candidates: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    rows = [r for r in (stats_rows or []) if isinstance(r, dict)]
    n_candidates = len(rows)
    if n_candidates <= 0:
        return 0, {"rows": 0, "expected_trades_per_day": 0.0}

    lb_days = float(lookback_days or 0.0)
    if lb_days <= 0:
        return 0, {"rows": n_candidates, "expected_trades_per_day": 0.0, "reason": "lookback_days_zero"}

    expected = 0.0
    used_rows = 0
    for row in rows:
        trades_count = _safe_int_opt(row.get("trades_count"))
        if trades_count is None:
            trades_count = _safe_int_opt(row.get("watchlist_trades_count"))
        if trades_count is None or trades_count <= 0:
            continue
        # One symbol can only be entered once per day in this strategy, so cap contribution at 1.
        expected += min(1.0, float(trades_count) / lb_days)
        used_rows += 1

    if expected <= 0:
        return 0, {
            "rows": n_candidates,
            "rows_with_trades_count": used_rows,
            "expected_trades_per_day": 0.0,
            "reason": "no_valid_trades_count",
        }

    target = int(math.ceil(expected))
    min_slots_eff = max(1, int(min_slots or 1))
    target = max(min_slots_eff, target)
    if max_slots and int(max_slots) > 0:
        target = min(target, int(max_slots))
    if cap_by_candidates:
        target = min(target, n_candidates)
    target = max(0, target)

    return target, {
        "rows": n_candidates,
        "rows_with_trades_count": used_rows,
        "expected_trades_per_day": expected,
        "slot_target": target,
        "lookback_days": lb_days,
        "min_slots": min_slots_eff,
        "max_slots": int(max_slots or 0),
    }


def _quality_risk_multiplier(plan: TradePlan, cfg: Dict) -> Tuple[float, Dict[str, Any]]:
    params = cfg.get("daily_trend_reversal") or {}
    enabled = bool(params.get("quality_sizing_enabled", False))
    meta: Dict[str, Any] = {"enabled": enabled, "applied": False}
    if not enabled:
        return 1.0, meta

    stats = getattr(plan, "watchlist_stats", None)
    if not isinstance(stats, dict):
        meta["reason"] = "missing_watchlist_stats"
        return 1.0, meta

    try:
        min_mult = float(params.get("quality_sizing_min_mult") or 0.4)
        max_mult = float(params.get("quality_sizing_max_mult") or 1.2)
        if max_mult < min_mult:
            min_mult, max_mult = max_mult, min_mult

        avg_floor = float(params.get("quality_sizing_avgR_floor") or 0.0)
        avg_cap = float(params.get("quality_sizing_avgR_cap") or 0.6)
        pf_floor = float(params.get("quality_sizing_pf_floor") or 1.0)
        pf_cap = float(params.get("quality_sizing_pf_cap") or 3.0)
        wr_floor = float(params.get("quality_sizing_win_rate_floor") or 0.45)
        wr_cap = float(params.get("quality_sizing_win_rate_cap") or 0.75)
        trades_ref = max(1, int(params.get("quality_sizing_trades_ref") or 20))
        lcb_z = float(params.get("quality_sizing_lcb_z") or 0.0)
        stop_rate_floor = float(params.get("quality_sizing_stop_rate_floor") or 0.20)
        stop_rate_cap = float(params.get("quality_sizing_stop_rate_cap") or 0.80)
        stop_flip_floor = float(params.get("quality_sizing_stop_flip_share_floor") or 0.15)
        stop_flip_cap = float(params.get("quality_sizing_stop_flip_share_cap") or 0.70)
        positive_month_floor = float(params.get("quality_sizing_positive_month_rate_floor") or 0.45)
        positive_month_cap = float(params.get("quality_sizing_positive_month_rate_cap") or 0.85)
        stdr_floor = float(params.get("quality_sizing_stdR_floor") or 0.2)
        stdr_cap = float(params.get("quality_sizing_stdR_cap") or 1.2)
        worst_month_floor = float(params.get("quality_sizing_worst_month_pnl_pct_floor") or -15.0)
        worst_month_cap = float(params.get("quality_sizing_worst_month_pnl_pct_cap") or 0.0)
        stop_no_progress_floor = float(params.get("quality_sizing_stop_no_progress_share_floor") or 0.2)
        stop_no_progress_cap = float(params.get("quality_sizing_stop_no_progress_share_cap") or 0.75)

        w_avg = float(params.get("quality_sizing_weight_avgR") or 0.5)
        w_pf = float(params.get("quality_sizing_weight_pf") or 0.2)
        w_wr = float(params.get("quality_sizing_weight_win_rate") or 0.2)
        w_n = float(params.get("quality_sizing_weight_trades") or 0.1)
        w_stop = float(params.get("quality_sizing_weight_stop_rate") or 0.0)
        w_stop_flip = float(params.get("quality_sizing_weight_stop_flip_share") or 0.0)
        w_month = float(params.get("quality_sizing_weight_positive_month_rate") or 0.0)
        w_stdr = float(params.get("quality_sizing_weight_stdR") or 0.0)
        w_worst_month = float(params.get("quality_sizing_weight_worst_month_pnl_pct") or 0.0)
        w_stop_no_progress = float(params.get("quality_sizing_weight_stop_no_progress_share") or 0.0)
        if w_avg < 0:
            w_avg = 0.0
        if w_pf < 0:
            w_pf = 0.0
        if w_wr < 0:
            w_wr = 0.0
        if w_n < 0:
            w_n = 0.0
        if w_stop < 0:
            w_stop = 0.0
        if w_stop_flip < 0:
            w_stop_flip = 0.0
        if w_month < 0:
            w_month = 0.0
        if w_stdr < 0:
            w_stdr = 0.0
        if w_worst_month < 0:
            w_worst_month = 0.0
        if w_stop_no_progress < 0:
            w_stop_no_progress = 0.0

        avg_r = _safe_float_opt(stats.get("avgR"))
        avg_r_stderr = _safe_float_opt(stats.get("avgR_stderr"))
        profit_factor = _safe_float_opt(stats.get("profit_factor"))
        win_rate = _safe_float_opt(stats.get("win_rate"))
        trades_count = _safe_float_opt(stats.get("trades_count"))
        stop_rate = _safe_float_opt(stats.get("stop_rate"))
        stop_flip_share = _safe_float_opt(stats.get("stop_flip_share_050"))
        stop_flip_share_100 = _safe_float_opt(stats.get("stop_flip_share_100"))
        positive_month_rate = _safe_float_opt(stats.get("positive_month_rate"))
        std_r = _safe_float_opt(stats.get("stdR"))
        worst_month_pnl_pct = _safe_float_opt(stats.get("worst_month_pnl_pct"))
        max_monthly_drawdown_pct = _safe_float_opt(stats.get("max_monthly_drawdown_pct"))
        stop_no_progress_share = _safe_float_opt(stats.get("stop_no_progress_share"))

        avg_eff = avg_r
        if avg_r is not None and avg_r_stderr is not None and lcb_z > 0:
            avg_eff = avg_r - (lcb_z * max(0.0, avg_r_stderr))

        def _score(v: Optional[float], lo: float, hi: float, *, neutral: float = 0.5) -> float:
            if v is None:
                return neutral
            if hi <= lo:
                return 1.0 if v > lo else 0.0
            return _clamp((v - lo) / (hi - lo), 0.0, 1.0)

        score_avg = _score(avg_eff, avg_floor, avg_cap)
        score_pf = _score(profit_factor, pf_floor, pf_cap)
        score_wr = _score(win_rate, wr_floor, wr_cap)
        score_n = _score(trades_count, 0.0, float(trades_ref))
        score_stop = 1.0 - _score(stop_rate, stop_rate_floor, stop_rate_cap)
        score_stop_flip = 1.0 - _score(stop_flip_share, stop_flip_floor, stop_flip_cap)
        stop_flip_100_floor = float(params.get("quality_sizing_stop_flip_share_100_floor") or stop_flip_floor)
        stop_flip_100_cap = float(params.get("quality_sizing_stop_flip_share_100_cap") or max(stop_flip_cap, stop_flip_100_floor))
        score_stop_flip_100 = 1.0 - _score(stop_flip_share_100, stop_flip_100_floor, stop_flip_100_cap)
        score_month = _score(positive_month_rate, positive_month_floor, positive_month_cap)
        score_stdr = 1.0 - _score(std_r, stdr_floor, stdr_cap)
        score_worst_month = _score(worst_month_pnl_pct, worst_month_floor, worst_month_cap)
        max_mdd_floor = float(params.get("quality_sizing_max_monthly_drawdown_pct_floor") or 2.0)
        max_mdd_cap = float(params.get("quality_sizing_max_monthly_drawdown_pct_cap") or 12.0)
        score_max_monthly_drawdown = 1.0 - _score(max_monthly_drawdown_pct, max_mdd_floor, max_mdd_cap)
        score_stop_no_progress = 1.0 - _score(stop_no_progress_share, stop_no_progress_floor, stop_no_progress_cap)

        w_stop_flip_100 = float(params.get("quality_sizing_weight_stop_flip_share_100") or 0.0)
        w_max_monthly_drawdown = float(params.get("quality_sizing_weight_max_monthly_drawdown_pct") or 0.0)
        if w_stop_flip_100 < 0:
            w_stop_flip_100 = 0.0
        if w_max_monthly_drawdown < 0:
            w_max_monthly_drawdown = 0.0
        w_sum = (
            w_avg
            + w_pf
            + w_wr
            + w_n
            + w_stop
            + w_stop_flip
            + w_stop_flip_100
            + w_month
            + w_stdr
            + w_worst_month
            + w_max_monthly_drawdown
            + w_stop_no_progress
        )
        if w_sum <= 0:
            w_avg, w_pf, w_wr, w_n = 0.5, 0.2, 0.2, 0.1
            w_stop, w_stop_flip, w_stop_flip_100, w_month = 0.0, 0.0, 0.0, 0.0
            w_stdr, w_worst_month, w_max_monthly_drawdown, w_stop_no_progress = 0.0, 0.0, 0.0, 0.0
            w_sum = 1.0

        score = (
            (w_avg * score_avg)
            + (w_pf * score_pf)
            + (w_wr * score_wr)
            + (w_n * score_n)
            + (w_stop * score_stop)
            + (w_stop_flip * score_stop_flip)
            + (w_stop_flip_100 * score_stop_flip_100)
            + (w_month * score_month)
            + (w_stdr * score_stdr)
            + (w_worst_month * score_worst_month)
            + (w_max_monthly_drawdown * score_max_monthly_drawdown)
            + (w_stop_no_progress * score_stop_no_progress)
        ) / w_sum
        risk_multiplier = _clamp(min_mult + (max_mult - min_mult) * score, min_mult, max_mult)

        meta.update(
            {
                "applied": True,
                "rank": stats.get("rank"),
                "avgR": avg_r,
                "avgR_stderr": avg_r_stderr,
                "avgR_effective": avg_eff,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "trades_count": trades_count,
                "stop_rate": stop_rate,
                "stop_flip_share_050": stop_flip_share,
                "stop_flip_share_100": stop_flip_share_100,
                "positive_month_rate": positive_month_rate,
                "stdR": std_r,
                "worst_month_pnl_pct": worst_month_pnl_pct,
                "max_monthly_drawdown_pct": max_monthly_drawdown_pct,
                "stop_no_progress_share": stop_no_progress_share,
                "score_avg": score_avg,
                "score_pf": score_pf,
                "score_win_rate": score_wr,
                "score_trades": score_n,
                "score_stop_rate": score_stop,
                "score_stop_flip_share": score_stop_flip,
                "score_stop_flip_share_100": score_stop_flip_100,
                "score_positive_month_rate": score_month,
                "score_stdR": score_stdr,
                "score_worst_month_pnl_pct": score_worst_month,
                "score_max_monthly_drawdown_pct": score_max_monthly_drawdown,
                "score_stop_no_progress_share": score_stop_no_progress,
                "score": score,
                "risk_multiplier": risk_multiplier,
                "min_mult": min_mult,
                "max_mult": max_mult,
                "weight_avgR": w_avg,
                "weight_pf": w_pf,
                "weight_win_rate": w_wr,
                "weight_trades": w_n,
                "weight_stop_rate": w_stop,
                "weight_stop_flip_share": w_stop_flip,
                "weight_stop_flip_share_100": w_stop_flip_100,
                "weight_positive_month_rate": w_month,
                "weight_stdR": w_stdr,
                "weight_worst_month_pnl_pct": w_worst_month,
                "weight_max_monthly_drawdown_pct": w_max_monthly_drawdown,
                "weight_stop_no_progress_share": w_stop_no_progress,
            }
        )
        return risk_multiplier, meta
    except Exception:
        meta["reason"] = "quality_sizing_error"
        return 1.0, meta


def compute_qty_with_guards(
    plan: TradePlan,
    equity: float,
    used_notional: float,
    cfg: Dict,
    *,
    allowed_total_override: Optional[float] = None,
    open_positions: int = 0,
    slot_target_override: Optional[int] = None,
    day_start_equity: Optional[float] = None,
) -> Tuple[int, Dict]:
    params = cfg.get("daily_trend_reversal") or {}
    max_positions_cfg = int(params.get("max_positions") or 0)
    risk_per_trade = float(params.get("risk_per_trade") or 0.0)
    leverage = float(params.get("leverage") or 1.0)
    max_margin_usage = float(params.get("max_margin_usage") or 1.0)
    margin_safety_buffer = float(params.get("margin_safety_buffer") or 0.0)
    required_free_margin_buffer = _safe_float_opt(params.get("required_free_margin_buffer"))
    if required_free_margin_buffer is None:
        required_free_margin_buffer = _safe_float_opt(params.get("alpaca_required_free_margin_buffer"))
    if required_free_margin_buffer is None:
        required_free_margin_buffer = 0.30
    effective_margin_safety_buffer = resolve_effective_margin_buffer(params)
    per_trade_max_pct_available = float(params.get("per_trade_max_pct_available") or 1.0)
    min_overlap_order_notional = max(0.0, float(params.get("min_overlap_order_notional") or 0.0))
    equal_split_cfg = bool(params.get("equal_split_across_max_slots", False))
    min_af_abs = float(params.get("min_available_funds_abs") or 0.0)
    min_af_ratio = float(params.get("min_available_funds_ratio_of_netliq") or 0.0)
    slot_distribution_enabled = bool(params.get("slot_distribution_enabled", False))
    slot_min_slots = max(1, int(params.get("slot_distribution_min_slots") or 1))
    slot_max_slots = max(0, int(params.get("slot_distribution_max_slots") or 0))
    slot_default_slots = max(0, int(params.get("slot_distribution_default_slots") or 0))
    sizing_stop_pct_floor = max(0.0, float(params.get("sizing_stop_pct_floor") or 0.0))
    sizing_stop_abs_floor = max(0.0, float(params.get("sizing_stop_abs_floor") or 0.0))
    tight_stop_exposure_cap_enabled = bool(params.get("tight_stop_exposure_cap_enabled", False))
    tight_stop_exposure_full_stop_pct = max(0.0, float(params.get("tight_stop_exposure_full_stop_pct") or 0.0))
    tight_stop_exposure_min_factor = _clamp(float(params.get("tight_stop_exposure_min_factor") or 0.0), 0.0, 1.0)
    tight_stop_exposure_apply_if_rank_at_least = int(params.get("tight_stop_exposure_apply_if_rank_at_least") or 0)
    tight_stop_exposure_apply_if_score_below = _safe_float_opt(params.get("tight_stop_exposure_apply_if_score_below"))
    tight_stop_exposure_activate_equity_mult = max(0.0, float(params.get("tight_stop_exposure_activate_equity_mult") or 0.0))
    buying_power_model_enabled = bool(params.get("buying_power_model_enabled", True))
    buying_power_long_open_markup = max(0.0, min(float(params.get("buying_power_long_open_markup") or 0.0), 0.5))
    buying_power_short_open_markup = max(0.0, min(float(params.get("buying_power_short_open_markup") or 0.03), 0.5))
    buying_power_short_margin_multiplier = max(
        1.0,
        min(float(params.get("buying_power_short_margin_multiplier") or 1.2), 3.0),
    )
    equity_taper_enabled = bool(params.get("equity_risk_taper_enabled", False))
    equity_taper_start_mult = max(0.0, float(params.get("equity_risk_taper_start_mult") or 0.0))
    equity_taper_full_mult = max(equity_taper_start_mult, float(params.get("equity_risk_taper_full_mult") or 0.0))
    equity_taper_max_reduction = max(0.0, min(1.0, float(params.get("equity_risk_taper_max_reduction") or 0.0)))
    starting_equity_cfg = max(0.0, float(params.get("starting_equity") or 0.0))

    slot_target = 0
    if slot_distribution_enabled:
        if slot_target_override is not None and int(slot_target_override) > 0:
            slot_target = int(slot_target_override)
        else:
            slot_target = slot_default_slots
        if slot_target > 0:
            slot_target = max(slot_min_slots, slot_target)
            if slot_max_slots > 0:
                slot_target = min(slot_target, slot_max_slots)
    effective_max_positions = max_positions_cfg
    effective_equal_split = equal_split_cfg
    if slot_distribution_enabled and slot_target > 0:
        if max_positions_cfg > 0:
            effective_max_positions = min(max_positions_cfg, slot_target)
        else:
            effective_max_positions = slot_target
        effective_equal_split = True

    state = {
        "equity": equity,
        "day_start_equity": day_start_equity,
        "used_notional": used_notional,
        "risk_per_trade": risk_per_trade,
        "leverage": leverage,
        "max_margin_usage": max_margin_usage,
        "margin_safety_buffer": margin_safety_buffer,
        "required_free_margin_buffer": required_free_margin_buffer,
        "margin_safety_buffer_effective": effective_margin_safety_buffer,
        "per_trade_max_pct_available": per_trade_max_pct_available,
        "min_overlap_order_notional": min_overlap_order_notional,
        "equal_split_across_max_slots": equal_split_cfg,
        "max_positions": max_positions_cfg,
        "slot_distribution_enabled": slot_distribution_enabled,
        "slot_target": slot_target,
        "slot_target_override": slot_target_override,
        "slot_distribution_min_slots": slot_min_slots,
        "slot_distribution_max_slots": slot_max_slots,
        "effective_equal_split": effective_equal_split,
        "effective_max_positions": effective_max_positions,
        "open_positions": open_positions,
        "sizing_stop_pct_floor": sizing_stop_pct_floor,
        "sizing_stop_abs_floor": sizing_stop_abs_floor,
        "tight_stop_exposure_cap_enabled": tight_stop_exposure_cap_enabled,
        "tight_stop_exposure_full_stop_pct": tight_stop_exposure_full_stop_pct,
        "tight_stop_exposure_min_factor": tight_stop_exposure_min_factor,
        "tight_stop_exposure_apply_if_rank_at_least": tight_stop_exposure_apply_if_rank_at_least,
        "tight_stop_exposure_apply_if_score_below": tight_stop_exposure_apply_if_score_below,
        "tight_stop_exposure_activate_equity_mult": tight_stop_exposure_activate_equity_mult,
        "buying_power_model_enabled": buying_power_model_enabled,
        "buying_power_long_open_markup": buying_power_long_open_markup,
        "buying_power_short_open_markup": buying_power_short_open_markup,
        "buying_power_short_margin_multiplier": buying_power_short_margin_multiplier,
        "equity_risk_taper_enabled": equity_taper_enabled,
        "equity_risk_taper_start_mult": equity_taper_start_mult,
        "equity_risk_taper_full_mult": equity_taper_full_mult,
        "equity_risk_taper_max_reduction": equity_taper_max_reduction,
        "starting_equity_cfg": starting_equity_cfg,
    }

    if effective_max_positions > 0 and open_positions >= effective_max_positions:
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

    quality_mult, quality_state = _quality_risk_multiplier(plan, cfg)
    risk_amt_base = max(0.0, equity * risk_per_trade)
    risk_amt = risk_amt_base
    state["risk_amount_base"] = risk_amt_base
    state["risk_amount_equity_taper_scale"] = 1.0
    state["equity_risk_taper_progress"] = 0.0
    state["equity_risk_taper_start_equity"] = None
    state["equity_risk_taper_full_equity"] = None
    if (
        equity_taper_enabled
        and equity_taper_max_reduction > 0.0
        and starting_equity_cfg > 0.0
        and equity_taper_start_mult > 0.0
    ):
        taper_start_equity = starting_equity_cfg * equity_taper_start_mult
        taper_full_equity = starting_equity_cfg * max(equity_taper_full_mult, equity_taper_start_mult)
        state["equity_risk_taper_start_equity"] = taper_start_equity
        state["equity_risk_taper_full_equity"] = taper_full_equity
        taper_progress = 0.0
        if equity > taper_start_equity:
            if taper_full_equity <= taper_start_equity:
                taper_progress = 1.0
            else:
                taper_progress = _clamp(
                    (equity - taper_start_equity) / (taper_full_equity - taper_start_equity),
                    0.0,
                    1.0,
                )
        taper_scale = max(0.0, 1.0 - (equity_taper_max_reduction * taper_progress))
        risk_amt *= taper_scale
        state["equity_risk_taper_progress"] = taper_progress
        state["risk_amount_equity_taper_scale"] = taper_scale
    state["risk_amount_daily_scale"] = 1.0
    state["daily_drawdown_pct"] = 0.0
    dd_scale_enabled = bool(params.get("daily_drawdown_risk_scale_enabled", False))
    dd_max_reduction = max(0.0, min(1.0, float(params.get("daily_drawdown_risk_max_reduction") or 0.0)))
    dd_trigger_pct = max(0.0, float(params.get("daily_drawdown_risk_trigger_pct") or 0.0))
    dd_full_pct = max(dd_trigger_pct, float(params.get("daily_drawdown_risk_full_pct") or max(2.0, dd_trigger_pct)))
    if dd_scale_enabled and dd_max_reduction > 0.0:
        start_eq = _safe_float_opt(day_start_equity)
        if start_eq is not None and start_eq > 0.0:
            dd_pct = max(0.0, ((start_eq - equity) / start_eq) * 100.0)
            state["daily_drawdown_pct"] = dd_pct
            if dd_pct > dd_trigger_pct:
                if dd_full_pct <= dd_trigger_pct:
                    progress = 1.0
                else:
                    progress = _clamp((dd_pct - dd_trigger_pct) / (dd_full_pct - dd_trigger_pct), 0.0, 1.0)
                dd_scale = max(0.0, 1.0 - (dd_max_reduction * progress))
                risk_amt = risk_amt_base * dd_scale
                state["risk_amount_daily_scale"] = dd_scale
    open_noise_risk_scale_enabled = bool(params.get("open_noise_risk_scale_enabled", False))
    open_noise_risk_scale_ratio_start = max(0.0, float(params.get("open_noise_risk_scale_ratio_start") or 0.0))
    open_noise_risk_scale_ratio_full = max(
        open_noise_risk_scale_ratio_start,
        float(params.get("open_noise_risk_scale_ratio_full") or open_noise_risk_scale_ratio_start),
    )
    open_noise_risk_scale_min_factor = _clamp(
        float(params.get("open_noise_risk_scale_min_factor") or 1.0),
        0.05,
        1.0,
    )
    state["open_noise_risk_scale_enabled"] = open_noise_risk_scale_enabled
    state["open_noise_risk_scale_ratio_start"] = open_noise_risk_scale_ratio_start
    state["open_noise_risk_scale_ratio_full"] = open_noise_risk_scale_ratio_full
    state["open_noise_risk_scale_min_factor"] = open_noise_risk_scale_min_factor
    atr_risk_scale_enabled = bool(params.get("atr_risk_scale_enabled", False))
    atr_risk_scale_pct_start = max(0.0, float(params.get("atr_risk_scale_pct_start") or 0.0))
    atr_risk_scale_pct_full = max(
        atr_risk_scale_pct_start,
        float(params.get("atr_risk_scale_pct_full") or atr_risk_scale_pct_start),
    )
    atr_risk_scale_min_factor = _clamp(
        float(params.get("atr_risk_scale_min_factor") or 1.0),
        0.05,
        1.0,
    )
    state["atr_risk_scale_enabled"] = atr_risk_scale_enabled
    state["atr_risk_scale_pct_start"] = atr_risk_scale_pct_start
    state["atr_risk_scale_pct_full"] = atr_risk_scale_pct_full
    state["atr_risk_scale_min_factor"] = atr_risk_scale_min_factor
    state["risk_amount"] = risk_amt
    raw_stop_distance = float(plan.stop_distance)
    if raw_stop_distance <= 0:
        state["reject_reason"] = "stop_distance_zero"
        return 0, state
    entry_price = max(1e-9, float(plan.entry_price))
    open_side = "buy" if str(getattr(plan, "direction", "long")).lower() == "long" else "sell"
    bp_markup = buying_power_short_open_markup if open_side == "sell" else buying_power_long_open_markup
    side_mult = buying_power_short_margin_multiplier if open_side == "sell" else 1.0
    bp_price_mult = ((1.0 + bp_markup) * side_mult) if buying_power_model_enabled else 1.0
    bp_unit_price = entry_price * bp_price_mult
    state["bp_open_side"] = open_side
    state["bp_price_mult"] = bp_price_mult
    state["bp_unit_price"] = bp_unit_price
    min_stop_distance = 0.0
    if sizing_stop_pct_floor > 0.0:
        min_stop_distance = max(min_stop_distance, entry_price * (sizing_stop_pct_floor / 100.0))
    if sizing_stop_abs_floor > 0.0:
        min_stop_distance = max(min_stop_distance, sizing_stop_abs_floor)
    effective_stop_distance = max(raw_stop_distance, min_stop_distance)
    state["stop_distance_raw"] = raw_stop_distance
    state["stop_distance_min_for_sizing"] = min_stop_distance
    state["stop_distance_effective"] = effective_stop_distance
    state["stop_pct_effective"] = (effective_stop_distance / entry_price) * 100.0

    risk_qty_base = int(risk_amt // effective_stop_distance) if risk_amt > 0 else 0
    state["risk_qty_base"] = risk_qty_base
    if risk_qty_base <= 0:
        state["reject_reason"] = "risk_qty_zero"
        return 0, state

    msb = effective_margin_safety_buffer
    net_avail = available * (1.0 - msb)
    headroom_cap = max(0.0, allowed_total * (1.0 - msb) - used_notional)
    net_avail = min(net_avail, headroom_cap)
    state["net_available"] = net_avail
    state["headroom_cap"] = headroom_cap

    if effective_equal_split and effective_max_positions > 0:
        eq_cap = allowed_total / effective_max_positions
        state["equal_split_cap"] = eq_cap
        net_avail = min(net_avail, eq_cap)
    net_avail_before_per_trade_cap = net_avail
    per_trade_cap_applied = False

    if per_trade_max_pct_available < 1.0:
        cap_pct = max(0.0, min(per_trade_max_pct_available, 1.0))
        per_trade_cap = available * (1.0 - msb) * cap_pct
        state["per_trade_cap"] = per_trade_cap
        capped_avail = min(net_avail, per_trade_cap)
        per_trade_cap_applied = capped_avail < net_avail
        net_avail = capped_avail

    state["tight_stop_exposure_factor"] = 1.0
    state["tight_stop_exposure_cap"] = None
    state["tight_stop_exposure_applied"] = False
    state["tight_stop_exposure_enabled_for_trade"] = True
    if tight_stop_exposure_cap_enabled and tight_stop_exposure_full_stop_pct > 0.0:
        enabled_for_trade = True
        if tight_stop_exposure_apply_if_rank_at_least > 0:
            rank_val = quality_state.get("rank")
            try:
                rank_num = int(rank_val) if rank_val is not None else None
            except Exception:
                rank_num = None
            if rank_num is None or rank_num < tight_stop_exposure_apply_if_rank_at_least:
                enabled_for_trade = False
        if tight_stop_exposure_apply_if_score_below is not None:
            score_val = _safe_float_opt(quality_state.get("score"))
            if score_val is None or score_val >= float(tight_stop_exposure_apply_if_score_below):
                enabled_for_trade = False
        if tight_stop_exposure_activate_equity_mult > 0.0:
            base_equity_for_activation = starting_equity_cfg if starting_equity_cfg > 0.0 else equity
            activation_equity = base_equity_for_activation * tight_stop_exposure_activate_equity_mult
            state["tight_stop_exposure_activation_equity"] = activation_equity
            if equity < activation_equity:
                enabled_for_trade = False
        state["tight_stop_exposure_enabled_for_trade"] = enabled_for_trade
        if enabled_for_trade:
            # Tight stops create oversized notional under pure risk sizing; cap exposure until stop% is wide enough.
            stop_pct_eff = state.get("stop_pct_effective")
            stop_pct_val = float(stop_pct_eff) if stop_pct_eff is not None else 0.0
            exposure_factor = _clamp(
                stop_pct_val / max(1e-9, tight_stop_exposure_full_stop_pct),
                tight_stop_exposure_min_factor,
                1.0,
            )
            tight_cap = available * (1.0 - msb) * exposure_factor
            state["tight_stop_exposure_factor"] = exposure_factor
            state["tight_stop_exposure_cap"] = tight_cap
            capped_avail = min(net_avail, tight_cap)
            state["tight_stop_exposure_applied"] = capped_avail < net_avail
            net_avail = capped_avail

    state["quality_sizing"] = quality_state
    state["quality_qty_multiplier"] = quality_mult
    state["net_available_final"] = net_avail
    budget_qty = int(net_avail // bp_unit_price) if net_avail > 0 else 0
    budget_qty_pre_cap = (
        int(net_avail_before_per_trade_cap // bp_unit_price)
        if net_avail_before_per_trade_cap > 0
        else 0
    )
    state["budget_qty_pre_cap"] = budget_qty_pre_cap
    state["per_trade_cap_applied"] = per_trade_cap_applied
    state["budget_qty"] = budget_qty
    if budget_qty <= 0:
        state["reject_reason"] = "budget_zero"
        return 0, state

    quality_mult_eff = max(0.0, quality_mult) if quality_state.get("applied") else 1.0
    risk_amt_adjusted = risk_amt * quality_mult_eff
    open_noise_ratio = _safe_float_opt(getattr(plan, "open_noise_stop_ratio", None))
    open_noise_risk_scale_factor = 1.0
    state["open_noise_stop_ratio"] = open_noise_ratio
    state["open_noise_risk_scale_applied"] = False
    if (
        open_noise_risk_scale_enabled
        and open_noise_ratio is not None
        and open_noise_risk_scale_ratio_start > 0.0
    ):
        if open_noise_ratio > open_noise_risk_scale_ratio_start:
            if open_noise_risk_scale_ratio_full <= open_noise_risk_scale_ratio_start:
                progress = 1.0
            else:
                progress = _clamp(
                    (open_noise_ratio - open_noise_risk_scale_ratio_start)
                    / (open_noise_risk_scale_ratio_full - open_noise_risk_scale_ratio_start),
                    0.0,
                    1.0,
                )
            open_noise_risk_scale_factor = max(
                open_noise_risk_scale_min_factor,
                1.0 - ((1.0 - open_noise_risk_scale_min_factor) * progress),
            )
        risk_amt_adjusted *= open_noise_risk_scale_factor
    state["open_noise_risk_scale_factor"] = open_noise_risk_scale_factor
    state["open_noise_risk_scale_applied"] = open_noise_risk_scale_factor < 0.999999
    atr_risk_scale_factor = 1.0
    atr_pct = None
    atr_abs = _safe_float_opt(getattr(plan, "atr", None))
    if atr_abs is not None and entry_price > 0:
        atr_pct = (atr_abs / entry_price) * 100.0
    state["atr_pct"] = atr_pct
    state["atr_risk_scale_applied"] = False
    if (
        atr_risk_scale_enabled
        and atr_pct is not None
        and atr_risk_scale_pct_start > 0.0
    ):
        if atr_pct > atr_risk_scale_pct_start:
            if atr_risk_scale_pct_full <= atr_risk_scale_pct_start:
                progress = 1.0
            else:
                progress = _clamp(
                    (atr_pct - atr_risk_scale_pct_start)
                    / (atr_risk_scale_pct_full - atr_risk_scale_pct_start),
                    0.0,
                    1.0,
                )
            atr_risk_scale_factor = max(
                atr_risk_scale_min_factor,
                1.0 - ((1.0 - atr_risk_scale_min_factor) * progress),
            )
        risk_amt_adjusted *= atr_risk_scale_factor
    state["atr_risk_scale_factor"] = atr_risk_scale_factor
    state["atr_risk_scale_applied"] = atr_risk_scale_factor < 0.999999
    risk_qty_adjusted = int(risk_amt_adjusted // effective_stop_distance) if risk_amt_adjusted > 0 else 0
    state["risk_amount_adjusted"] = risk_amt_adjusted
    state["risk_qty_adjusted"] = risk_qty_adjusted
    state["risk_qty"] = risk_qty_adjusted

    base_qty = min(risk_qty_base, budget_qty)
    base_qty_pre_cap = min(risk_qty_base, budget_qty_pre_cap)
    state["base_qty_pre_cap"] = base_qty_pre_cap
    state["base_qty"] = base_qty
    qty = min(risk_qty_adjusted, budget_qty)
    reserve_capacity = bool(params.get("quality_sizing_reserve_capacity", True))
    reserve_pre_cap = bool(params.get("per_trade_cap_reserve_capacity", False))
    if quality_state.get("applied"):
        # Avoid silently dropping otherwise valid trades due to integer truncation.
        if base_qty > 0 and quality_mult_eff > 0 and qty <= 0:
            qty = 1
    capacity_qty = qty
    if quality_state.get("applied") and reserve_capacity and qty < base_qty:
        # If quality scaling trims size, keep capacity accounting at the unscaled base size to avoid
        # refilling the same notional with lower-quality symbols later in the same allocation pass.
        capacity_qty = base_qty
    if reserve_capacity and reserve_pre_cap and per_trade_cap_applied and base_qty_pre_cap > capacity_qty:
        # Optional: if per-trade cap forced size down, reserve pre-cap capacity so we don't
        # refill the remaining budget with lower-priority symbols in the same pass.
        capacity_qty = base_qty_pre_cap
    state["capacity_qty"] = capacity_qty
    state["capacity_notional_per_share"] = bp_unit_price
    state["final_qty"] = qty
    if qty <= 0:
        state["reject_reason"] = "budget_zero"
        return 0, state
    final_order_notional = qty * bp_unit_price
    state["final_order_notional"] = final_order_notional
    state["overlap_order_notional_reject_applied"] = False
    if open_positions > 0 and min_overlap_order_notional > 0.0 and final_order_notional < min_overlap_order_notional:
        state["overlap_order_notional_reject_applied"] = True
        state["reject_reason"] = "min_overlap_order_notional"
        return 0, state

    return qty, state
