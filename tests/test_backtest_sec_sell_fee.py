import datetime as dt
import unittest

from app.backtest.daily_trend_backtest import _compute_sec_sell_fee


class TestBacktestSecSellFee(unittest.TestCase):
    def test_no_fee_before_effective_date(self):
        meta = _compute_sec_sell_fee(
            direction="long",
            entry_fill=100.0,
            exit_fill=101.0,
            entry_dt=dt.datetime(2026, 4, 3, 9, 35),
            exit_dt=dt.datetime(2026, 4, 3, 9, 50),
            entry_date="2026-04-03",
            exit_date="2026-04-03",
            params={},
        )
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["fee_per_share"], 0.0)

    def test_fee_applies_to_long_exit_sale(self):
        meta = _compute_sec_sell_fee(
            direction="long",
            entry_fill=100.0,
            exit_fill=101.0,
            entry_dt=dt.datetime(2026, 4, 6, 9, 35),
            exit_dt=dt.datetime(2026, 4, 6, 9, 50),
            entry_date="2026-04-06",
            exit_date="2026-04-06",
            params={},
        )
        self.assertTrue(meta["applied"])
        self.assertAlmostEqual(meta["fee_per_share"], 101.0 * (20.60 / 1_000_000.0))
        self.assertEqual(meta["sale_date"], "2026-04-06")

    def test_fee_applies_to_short_entry_sale(self):
        meta = _compute_sec_sell_fee(
            direction="short",
            entry_fill=55.5,
            exit_fill=54.0,
            entry_dt=dt.datetime(2026, 4, 6, 9, 35),
            exit_dt=dt.datetime(2026, 4, 6, 9, 50),
            entry_date="2026-04-06",
            exit_date="2026-04-06",
            params={},
        )
        self.assertTrue(meta["applied"])
        self.assertAlmostEqual(meta["fee_per_share"], 55.5 * (20.60 / 1_000_000.0))
        self.assertEqual(meta["sale_date"], "2026-04-06")


if __name__ == "__main__":
    unittest.main()
