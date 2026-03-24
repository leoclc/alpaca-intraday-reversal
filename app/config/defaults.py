from __future__ import annotations

CONFIG_FILE = "config.json"
RUNTIME_OVERRIDES_FILE = "runtime_overrides.json"


DEFAULT_CONFIG: dict = {
    "strategy": {"name": "daily_trend_reversal"},
    "daily_trend_reversal": {
        "trend_ma_days": 200,
        "reversal_mode": "atr",
        "reversal_threshold_pct": 2.0,
        "reversal_threshold_atr": 0.1,
        "reversal_lookback_days": 1,
        "reversal_quantile": 0.35,
        "reversal_quantile_lookback_days": 30,
        "entry_time_et": "09:35",
        "entry_times_et": [],
        "entry_start_et": "09:35",
        "entry_end_et": "11:30",
        "session_open_et": "09:30",
        "entry_price_mode": "open",
        "use_intraday_entry": True,
        "intraday_entry_in_watchlist": False,
        "max_positions": 0,
        "risk_per_trade": 0.02,
        "leverage": 4.0,
        "max_margin_usage": 0.75,
        "margin_safety_buffer": 0.25,
        # Enforced minimum free-BP reserve (used by both live and backtest sizing).
        "required_free_margin_buffer": 0.30,
        # Buying-power model parity (Alpaca short openings reserve extra BP).
        "buying_power_model_enabled": True,
        "buying_power_long_open_markup": 0.0,
        "buying_power_short_open_markup": 0.03,
        # Additional short-side BP multiplier (1.20 ~= +20% reserve on short notionals).
        "buying_power_short_margin_multiplier": 1.2,
        # Optional symbol-aware add-on from asset maintenance margin requirement.
        "buying_power_short_mmr_enabled": False,
        "buying_power_short_mmr_weight": 0.0,
        "buying_power_short_mmr_floor_pct": 0.0,
        "buying_power_short_mmr_cap_pct": 50.0,
        # Live guard: for symbols with very high maintenance margin requirement (e.g. 100%),
        # cap order sizing using non_marginable_buying_power to avoid broker-side BP rejects.
        "live_non_marginable_bp_cap_enabled": True,
        "live_non_marginable_mmr_threshold_pct": 100.0,
        # If MMR lookup is temporarily unavailable, conservatively cap short entries
        # against non_marginable_buying_power to prevent avoidable broker rejects.
        "live_non_marginable_bp_cap_when_mmr_missing": True,
        # Backtest parity: apply the same post-sizing execution cap used by live
        # (broker BP + non-marginable guard for high-MMR symbols).
        "backtest_live_bp_execution_cap_enabled": True,
        # Optional stricter parity mode: reseed open exposure from marked positions at each
        # entry-time bucket (can be noisier than live due bar-level approximation).
        "backtest_live_exposure_resync_enabled": False,
        # Backtest parity source for symbol maintenance margin requirement.
        "backtest_use_live_asset_mmr": True,
        # Simulated non-marginable BP pool = equity * multiplier (Alpaca-like accounts: 1.0).
        "backtest_non_marginable_buying_power_mult": 1.0,
        "per_trade_max_pct_available": 0.25,
        # When another position is already open, reject follow-up entries whose actual
        # BP-sized order notional is below this floor. If 0, disabled.
        "min_overlap_order_notional": 0.0,
        "equal_split_across_max_slots": False,
        # Optional dynamic slot distribution: split capital across expected trade count
        # estimated from watchlist trades_count / watchlist.lookback_days.
        "slot_distribution_enabled": False,
        "slot_distribution_min_slots": 1,
        # If > 0, cap dynamic slots at this value.
        "slot_distribution_max_slots": 0,
        # Fallback when enabled but expected slots can't be estimated from stats.
        "slot_distribution_default_slots": 0,
        "min_available_funds_abs": 0.0,
        "min_available_funds_ratio_of_netliq": 0.0,
        "starting_equity": 30000.0,
        "minute_report_enabled": False,
        "minute_report_interval_sec": 60,
        "position_close_report_enabled": True,
        "position_close_poll_interval_sec": 30,
        # Optional live retry after broker BP reject (off by default; prefer deterministic pre-cap sizing).
        "live_bp_retry_on_reject": False,
        # Optional live-only gate from watchlist quality stats (off by default).
        "live_quality_gate_enabled": False,
        "live_quality_min_selection_score": None,
        "live_quality_min_avgR": None,
        "live_quality_min_trades_count": 0,
        "live_quality_max_rank": 0,
        "live_quality_min_positive_month_rate": None,
        "minute_report_max_symbols": 20,
        "minute_report_lookback_minutes": 5,
        "stop_mode": "atr",
        "atr_period": 14,
        "stop_atr_mult": 1.0,
        "stop_pct": 1.0,
        "target_rr": 1.5,
        # Optional plan-level stop/target floors (off by default).
        "trade_guardrails_enabled": False,
        "trade_guardrail_min_stop_bps": 0.0,
        "trade_guardrail_min_target_bps": 0.0,
        "trade_guardrail_min_stop_abs": 0.0,
        "trade_guardrail_min_target_abs": 0.0,
        "stop_r": 1.0,
        "target_r": 1.5,
        "time_exit_days": 2,
        "time_stop_minutes": 90,
        "intraday_only": False,
        "intraday_filter_enabled": False,
        "intraday_filter_require_bars": False,
        "intraday_filter_apply_in_watchlist": False,
        # Optional entry-quality guards (all disabled by default for backwards compatibility):
        # - Signal strength in ATR units.
        # - Opening-window noise caps relative to ATR/stop distance.
        "min_signal_return_atr_abs": 0.0,
        "max_signal_return_atr_abs": 0.0,
        "open_noise_window_minutes": 0,
        "open_noise_apply_in_watchlist": True,
        "max_open_noise_atr": 0.0,
        "max_open_noise_stop_ratio": 0.0,
        "min_stop_to_open_noise_ratio": 0.0,
        # Optional sizing-only scale (no signal rejection): when opening noise is large
        # relative to stop_distance, reduce risk amount linearly.
        "open_noise_risk_scale_enabled": False,
        "open_noise_risk_scale_ratio_start": 1.0,
        "open_noise_risk_scale_ratio_full": 2.0,
        "open_noise_risk_scale_min_factor": 0.5,
        # Optional sizing-only scale (no signal rejection): when daily ATR% is elevated
        # versus entry price, reduce risk amount linearly.
        "atr_risk_scale_enabled": False,
        "atr_risk_scale_pct_start": 3.0,
        "atr_risk_scale_pct_full": 6.0,
        "atr_risk_scale_min_factor": 0.6,
        # Entry confirmation (optional). When enabled, evaluate the window after `entry_time_et`
        # (length `confirm_minutes`) and only enter if price moves in our favor by at least
        # `confirm_move_bps` at any point during that window.
        "confirm_move_bps": 0.0,
        "confirm_minutes": 0,
        "confirm_apply_in_watchlist": True,
        "confirm_entry_price_mode": "close",
        # Optional "don't chase" cap for confirmation mode: if the favorable move within the confirm
        # window exceeds this (in bps), skip the trade (to avoid entering after the move already happened).
        # If unset/0, disabled.
        "max_confirm_hit_bps": None,
        # Gap filters.
        "min_gap_bps_long": 0.0,
        "min_gap_bps_short": 0.0,
        # Favorable gap means gap aligned with our mean-reversion direction:
        # - long: gap down is favorable  => gap_fav_bps = -gap_bps
        # - short: gap up is favorable  => gap_fav_bps = +gap_bps
        # If unset/None, the filter is disabled (to preserve behavior of older configs).
        "min_gap_fav_bps_long": None,
        "min_gap_fav_bps_short": None,
        "early_range_minutes": 0,
        "max_early_pullback_bps": 0.0,
        "session_close_et": "16:00",
        "flatten_buffer_minutes": 5,
        "trend_fast_len": 0,
        "trend_slow_len": 0,
        "trend_min_slope_bps": 0,
        "trend_min_distance_atr": 0.0,
        "volume_lookback_days": 60,
        "volume_min_ratio": 0.0,
        "entry_order_type": "market",
        "order_tif": "day",
        "use_brackets": True,
        # For market entries, place exits from actual fill price (live parity with Alpaca fills).
        "live_reanchor_brackets_on_fill": True,
        "live_reanchor_fill_timeout_sec": 30.0,
        "live_reanchor_fill_poll_sec": 0.25,
        "live_reanchor_cancel_unfilled_entry": True,
        # Release live entry batches against broker time so local clock skew does not
        # submit 09:35 orders a few seconds early.
        "live_entry_use_broker_clock": True,
        "live_entry_timing_fallback_delay_sec": 5.0,
        "live_entry_timing_poll_sec": 0.25,
        # Backtest parity: re-simulate stop/target hits from simulated entry fill (not planned entry).
        "backtest_reanchor_brackets_on_fill": True,
        "backtest_reanchor_debug": False,
        "backtest_reanchor_debug_max_logs": 40,
        # SEC Section 31 regulatory fee on covered sales. This applies to our long exits
        # and short entries from the effective date onward.
        "backtest_sec_sell_fee_enabled": True,
        "backtest_sec_sell_fee_effective_date": "2026-04-04",
        "backtest_sec_sell_fee_rate_per_million": 20.60,
        # If true, backtest reuses existing watchlist files from watchlists_dir when present.
        # Useful for fast parity/sizing sweeps while keeping the signal set fixed.
        "backtest_reuse_existing_watchlists": False,
        # If a frozen live watchlist snapshot exists for a date, prefer it during backtests
        # instead of rebuilding that day's watchlist with later-available data.
        "backtest_reuse_frozen_live_watchlists": True,
        # Optional floor on stop_distance vs daily ATR (stop_distance / ATR).
        # If 0, disabled.
        "min_stop_atr": 0.0,
        # Optional filter (skip trade) when stop_distance / ATR is too small.
        # This is distinct from `min_stop_atr` (which widens the stop); this one rejects the setup.
        # If 0, disabled.
        "min_stop_atr_filter": 0.0,
        "fixed_qty": None,
    },
    "watchlist": {
        "lookback_days": 252,
        "minTrades": 30,
        "reject_negative_pnl": True,
        "minProfitFactor": 1.05,
        "minAvgR": 0.0,
        "entry_time_rank_by": "avgR",
        "top_k_rank_by": "total_pnl_pct",
        "top_k": 0,
        # Preserve the first live watchlist written each day in an immutable snapshot so
        # after-the-fact parity runs can reuse the exact morning artifact.
        "freeze_live_snapshot_enabled": True,
    },
    "watchlist_report_enabled": False,
    "watchlist_source": "node",
    "watchlist_node_base": "http://localhost:3000",
    "watchlist_asset_filters": {
        "asset_class": "us_equity",
        "status": "active",
        "price_min": 15,
        "price_max": 1000,
        "max_spread_bps_median": 10,
        "shortable": True,
        "marginable": True,
        "easy_to_borrow": True,
        "tradable": True,
    },
    "alpaca": {
        "base_url": "https://paper-api.alpaca.markets",
        "data_url": "https://data.alpaca.markets",
        "api_key_id": "",
        "api_secret_key": "",
        "data_feed": "iex",
        "bars_chunk_size": 50,
        "adjustment": "raw",
        "timeout_sec": 30,
        "max_retries": 3,
        "retry_backoff_sec": 2,
    },
    "replay": {
        "enabled": False,
        "start_date": None,
        "end_date": None,
        "emit_daily_details": False,
    },
    "market_filters": {
        "enabled": False,
        "symbol": "SPY",
        "use_atr": True,
        "atr_period": 14,
        "atr_max_pct": 2.5,
        "use_gap": True,
        "gap_bps_max": 80,
        "use_trend": False,
        "trend_ma_days": 200,
        "log_details": False,
    },
    "daily_bars_cache_dir": "alpaca-ohlc",
    "minute_bars_cache_dir": "cache/ohlc_minute",
    "minute_bars_refresh": False,
    "watchlists_dir": "watchlists",
    "frozen_watchlists_dir": "logs/watchlist_snapshots/live",
    "logs_dir": "logs",
    "symbols": [],
}
