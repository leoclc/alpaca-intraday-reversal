from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from app.backtest.daily_trend_backtest import run_backtest
from app.config.defaults import DEFAULT_CONFIG
from app.watchlist.storage import freeze_watchlist_snapshot, frozen_watchlist_path


class _DummyStore:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def get_daily_bars_bulk(self, symbols, start, end, cfg=None, allow_fetch=True):
        out = {}
        for sym in symbols:
            out[str(sym).upper()] = [
                {
                    "date": "2026-01-02",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                }
            ]
        return out

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


def _cfg(tmp_path: Path) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["watchlists_dir"] = str(tmp_path / "watchlists")
    cfg["frozen_watchlists_dir"] = str(tmp_path / "frozen_watchlists")
    cfg["daily_bars_cache_dir"] = str(tmp_path / "daily_bars")
    cfg["replay"] = {"start_date": "2026-01-02", "end_date": "2026-01-02"}
    cfg["symbols"] = ["AAA"]
    cfg["watchlist_source"] = "symbols"
    cfg["daily_trend_reversal"]["backtest_reuse_frozen_live_watchlists"] = True
    return cfg


def test_freeze_watchlist_snapshot_preserves_first_payload(tmp_path: Path):
    cfg = _cfg(tmp_path)
    first = {"date": "2026-01-02", "watchlist": [{"symbol": "AAA"}], "meta": {"source": "first"}}
    second = {"date": "2026-01-02", "watchlist": [{"symbol": "BBB"}], "meta": {"source": "second"}}

    freeze_watchlist_snapshot(first, cfg, date_str="2026-01-02", overwrite=False)
    path = freeze_watchlist_snapshot(second, cfg, date_str="2026-01-02", overwrite=False)

    assert path == frozen_watchlist_path("2026-01-02", cfg)
    assert path.read_text(encoding="utf-8").find('"AAA"') >= 0
    assert path.read_text(encoding="utf-8").find('"BBB"') < 0


def test_backtest_prefers_frozen_live_watchlist_over_rebuild(tmp_path: Path):
    cfg = _cfg(tmp_path)
    payload = {
        "date": "2026-01-02",
        "watchlist": [{"symbol": "AAA", "direction": "long", "entry_time_et": "09:35"}],
        "meta": {"frozen": True},
    }
    freeze_watchlist_snapshot(payload, cfg, date_str="2026-01-02", overwrite=False)

    out_path = tmp_path / "backtest_output" / "backtest.json"

    with (
        patch("app.backtest.daily_trend_backtest.resolve_asset_universe_symbols", return_value=(["AAA"], "test")),
        patch("app.backtest.daily_trend_backtest.AlpacaOHLCStore", _DummyStore),
        patch("app.backtest.daily_trend_backtest.build_watchlist", side_effect=AssertionError("should not rebuild")),
        patch("app.backtest.daily_trend_backtest.market_filter_decision", return_value=(False, {})),
        patch("app.backtest.daily_trend_backtest.read_asset_universe_snapshot", return_value=([], {})),
        patch("app.backtest.daily_trend_backtest._build_daily_lookup", return_value={}),
        patch("app.backtest.daily_trend_backtest.run_replay", return_value=[]),
    ):
        summary, trades = run_backtest(
            cfg=cfg,
            start_date="2026-01-02",
            end_date="2026-01-02",
            out_path=str(out_path),
        )

    assert summary["trades"] == 0
    assert trades == []
    copied_watchlist = tmp_path / "watchlists" / "2026-01-02.json"
    assert copied_watchlist.exists()
    assert copied_watchlist.read_text(encoding="utf-8").find('"AAA"') >= 0
