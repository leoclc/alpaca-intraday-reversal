from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.backtest.daily_trend_backtest import run_backtest
from app.config.defaults import DEFAULT_CONFIG
from app.strategies.types import TradePlan, TradeResult


class _DummyStore:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def get_daily_bars_bulk(self, symbols, start, end, cfg=None, allow_fetch=True):
        return None

    def get_daily_bars(self, symbol, start, end, cfg=None, allow_fetch=True):
        return [
            {
                "date": "2026-01-02",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
            }
        ]


def _cfg() -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["symbols"] = ["ONE", "TWO"]
    cfg["watchlist_source"] = "symbols"
    cfg["replay"] = {"start_date": "2026-01-02", "end_date": "2026-01-02"}
    cfg["logs_dir"] = "logs"
    cfg["daily_trend_reversal"].update(
        {
            "starting_equity": 30_000.0,
            "risk_per_trade": 0.04,
            "intraday_only": True,
            "backtest_quote_fill_enabled": False,
            "backtest_fill_model_enabled": False,
            "backtest_reanchor_brackets_on_fill": True,
            "backtest_live_bp_execution_cap_enabled": False,
            "backtest_use_live_asset_mmr": False,
            "quality_sizing_enabled": False,
        }
    )
    return cfg


def _plan(symbol: str, entry_time_et: str, entry_price: float) -> TradePlan:
    stop_distance = 0.1
    return TradePlan(
        symbol=symbol,
        direction="long",
        signal_date="2026-01-02",
        entry_date="2026-01-02",
        entry_time_et=entry_time_et,
        entry_price=entry_price,
        stop_price=entry_price - stop_distance,
        target_price=entry_price + stop_distance,
        time_exit_date="2026-01-02",
        stop_distance=stop_distance,
        target_rr=1.0,
    )


def test_intraday_backtest_releases_capital_on_fill_adjusted_exit_time():
    trade_one = TradeResult(
        plan=_plan("ONE", "09:35", 50.0),
        exit_date="2026-01-02",
        exit_price=49.9,
        exit_reason="stop",
        pnl_pct=-0.2,
        r_multiple=-1.0,
        exit_ts="2026-01-02T09:42:00-05:00",
        stop_hit_ts="2026-01-02T09:42:00-05:00",
    )
    trade_two = TradeResult(
        plan=_plan("TWO", "09:38", 60.0),
        exit_date="2026-01-02",
        exit_price=60.1,
        exit_reason="target",
        pnl_pct=0.1666666667,
        r_multiple=1.0,
        exit_ts="2026-01-02T09:40:00-05:00",
        target_hit_ts="2026-01-02T09:40:00-05:00",
    )

    def _simulate_exit(plan_obj, mode, bars_daily, bars_intraday, cfg):
        if plan_obj.symbol == "ONE":
            return {
                "exit_date": "2026-01-02",
                "exit_price": 49.9,
                "exit_reason": "stop",
                "exit_ts": "2026-01-02T09:36:00-05:00",
                "stop_hit_ts": "2026-01-02T09:36:00-05:00",
                "target_hit_ts": None,
            }
        return {
            "exit_date": "2026-01-02",
            "exit_price": 60.1,
            "exit_reason": "target",
            "exit_ts": "2026-01-02T09:40:00-05:00",
            "stop_hit_ts": None,
            "target_hit_ts": "2026-01-02T09:40:00-05:00",
        }

    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "backtest.json"
        with (
            patch("app.backtest.daily_trend_backtest.resolve_asset_universe_symbols", return_value=(["ONE", "TWO"], "test")),
            patch("app.backtest.daily_trend_backtest.AlpacaOHLCStore", _DummyStore),
            patch("app.backtest.daily_trend_backtest.build_watchlist", return_value=[]),
            patch("app.backtest.daily_trend_backtest.market_filter_decision", return_value=(False, {})),
            patch("app.backtest.daily_trend_backtest.read_asset_universe_snapshot", return_value=([], {})),
            patch("app.backtest.daily_trend_backtest._build_daily_lookup", return_value={}),
            patch("app.backtest.daily_trend_backtest.get_intraday_bars", return_value=[]),
            patch("app.backtest.daily_trend_backtest.simulate_exit", side_effect=_simulate_exit),
            patch("app.backtest.daily_trend_backtest.run_replay", return_value=[trade_one, trade_two]),
        ):
            summary, trades = run_backtest(
                cfg=_cfg(),
                start_date="2026-01-02",
                end_date="2026-01-02",
                out_path=str(out_path),
            )

    assert summary["trades"] == 2
    assert [trade.plan.symbol for trade in trades] == ["ONE", "TWO"]
