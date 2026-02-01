import unittest

from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.replay.daily_strategy_replay import run_replay
from app.strategies.daily_trend_reversal import build_trade, generate_signals
from app.watchlist.daily_strategy_builder import build_watchlist
from tests.helpers import clean_dir, repo_root, seed_trending_universe


class TestReplayParity(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = repo_root() / "alpaca-ohlc"
        self.watch_dir = repo_root() / "watchlists"
        clean_dir(self.base_dir)
        clean_dir(self.watch_dir)

    def test_verify_vs_replay_trade_plan_parity(self):
        symbol = "PARITY"
        last_date = seed_trending_universe(
            self.base_dir,
            [symbol],
            "2024-01-02",
            80,
            dip_every=7,
            growth_pct=0.01,
            dip_multiplier=0.98,
        )
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
                "lookback_days": 60,
                "minTrades": 1,
                "reject_negative_pnl": False,
                "minProfitFactor": 0.0,
                "top_k": 1,
            },
            "watchlist_source": "symbols",
            "daily_bars_cache_dir": str(self.base_dir),
            "watchlists_dir": str(self.watch_dir),
            "symbols": [symbol],
            "replay": {"start_date": last_date, "end_date": last_date},
        }
        data_store = AlpacaOHLCStore(cfg=cfg)
        signals_all = generate_signals([symbol], "2024-01-02", last_date, cfg, data_store)
        self.assertTrue(signals_all)
        signal_date = signals_all[-1].signal_date
        wl = build_watchlist(cfg, target_date=signal_date, data_store=data_store)
        self.assertTrue(wl)
        signals = generate_signals([symbol], signal_date, signal_date, cfg, data_store)
        self.assertTrue(signals)
        plan = build_trade(signals[0], cfg, data_store)
        self.assertIsNotNone(plan)
        trades = run_replay(cfg, start_date=signal_date, end_date=signal_date, data_store=data_store)
        self.assertTrue(trades)
        replay_plan = trades[0].plan
        self.assertEqual(plan.entry_date, replay_plan.entry_date)
        self.assertEqual(plan.stop_price, replay_plan.stop_price)
        self.assertEqual(plan.target_price, replay_plan.target_price)
        self.assertEqual(plan.time_exit_date, replay_plan.time_exit_date)


if __name__ == "__main__":
    unittest.main()
