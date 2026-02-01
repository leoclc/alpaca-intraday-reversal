import unittest
from pathlib import Path

from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.strategies.daily_trend_reversal import generate_signals
from tests.helpers import clean_dir, generate_trading_days, repo_root, write_bars_csv


class TestNoLookahead(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = repo_root() / "alpaca-ohlc"
        clean_dir(self.base_dir)

    def test_signal_uses_only_d_minus_1(self):
        symbol = "TEST"
        days = generate_trading_days("2024-01-02", 210)
        rows = []
        for i, day in enumerate(days):
            if i < 207:
                close = 100.0
            elif i == 207:
                close = 104.0  # D-2
            elif i == 208:
                close = 101.0  # D-1 (return -2.88%)
            else:
                close = 80.0  # D (would flip trend if misused)
            open_price = close
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": open_price,
                    "high": open_price + 1.0,
                    "low": open_price - 1.0,
                    "close": close,
                    "volume": 1_000_000,
                    "vwap": close,
                }
            )
        write_bars_csv(self.base_dir / f"{symbol}.csv", rows)
        cfg = {
            "daily_trend_reversal": {"trend_ma_days": 200, "reversal_threshold_pct": 2.0},
            "daily_bars_cache_dir": str(self.base_dir),
        }
        data_store = AlpacaOHLCStore(cfg=cfg)
        signal_date = days[-1].isoformat()
        signals = generate_signals([symbol], signal_date, signal_date, cfg, data_store)
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.direction, "long")
        self.assertEqual(sig.trend_state, "uptrend")
        self.assertLessEqual(sig.return_pct, -2.0)


if __name__ == "__main__":
    unittest.main()
