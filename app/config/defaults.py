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
        "per_trade_max_pct_available": 0.25,
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
        "minute_report_max_symbols": 20,
        "minute_report_lookback_minutes": 5,
        "stop_mode": "atr",
        "atr_period": 14,
        "stop_atr_mult": 1.0,
        "stop_pct": 1.0,
        "target_rr": 1.5,
        "stop_r": 1.0,
        "target_r": 1.5,
        "time_exit_days": 2,
        "time_stop_minutes": 90,
        "intraday_only": False,
        "intraday_filter_enabled": False,
        "intraday_filter_require_bars": False,
        "intraday_filter_apply_in_watchlist": False,
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
    "logs_dir": "logs",
    "symbols": [],
}
