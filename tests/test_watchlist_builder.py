import datetime as dt
import unittest
from pathlib import Path

from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.watchlist.daily_strategy_builder import build_watchlist
from tests.helpers import clean_dir, repo_root, seed_trending_universe


class TestWatchlistBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = repo_root() / "alpaca-ohlc"
        self.watch_dir = repo_root() / "watchlists"
        clean_dir(self.base_dir)
        clean_dir(self.watch_dir)

    def test_watchlist_non_empty(self):
        symbols = [f"SYM{i:03d}" for i in range(300)]
        last_date = seed_trending_universe(
            self.base_dir,
            symbols,
            "2024-01-02",
            120,
            dip_every=7,
            growth_pct=0.01,
            dip_multiplier=0.98,
        )
        target_date = (dt.date.fromisoformat(last_date) + dt.timedelta(days=1)).isoformat()
        cfg = {
            "daily_trend_reversal": {
                "trend_ma_days": 10,
                "reversal_threshold_pct": 2.0,
                "atr_period": 14,
                "stop_mode": "atr",
                "stop_atr_mult": 1.0,
                "target_rr": 1.5,
                "time_exit_days": 2,
            },
            "watchlist": {
                "lookback_days": 90,
                "minTrades": 8,
                "reject_negative_pnl": False,
                "minProfitFactor": 0.0,
                "top_k": 50,
            },
            "watchlist_source": "symbols",
            "daily_bars_cache_dir": str(self.base_dir),
            "watchlists_dir": str(self.watch_dir),
            "symbols": symbols,
        }
        data_store = AlpacaOHLCStore(cfg=cfg)
        watchlist = build_watchlist(cfg, target_date=target_date, data_store=data_store)
        self.assertTrue(len(watchlist) > 0)
        self.assertTrue(any(int(r.get("trades_count") or 0) >= 8 for r in watchlist))


if __name__ == "__main__":
    unittest.main()
