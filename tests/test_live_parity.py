import unittest

from app.reporting.live_parity import build_daily_parity_row, parse_live_position_closed_line


class TestLiveParity(unittest.TestCase):
    def test_parse_live_position_closed_line(self):
        line = (
            "INFO:[LIVE] position_closed symbol=VZ reason=stop side=sell qty=1826 "
            "entry=50.289939 exit=50.091062 pnl=-363.15 pnl_pct=-0.40% at=09:36:29"
        )
        trade = parse_live_position_closed_line(line)
        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertEqual(trade["symbol"], "VZ")
        self.assertEqual(trade["reason"], "stop")
        self.assertAlmostEqual(trade["qty"], 1826.0)
        self.assertAlmostEqual(trade["pnl"], -363.15)

    def test_build_daily_parity_row(self):
        live_day = {
            "trade_count": 2,
            "symbols": ["AMZN", "FISV"],
            "pnl_total": -657.32,
            "failure_count": 3,
        }
        backtest_rows = [
            {"symbol": "FISV", "pnl_total": -357.90},
            {"symbol": "AMZN", "pnl_total": -9.00},
            {"symbol": "CRH", "pnl_total": 2.60},
        ]
        row = build_daily_parity_row("2026-03-17", live_day, backtest_rows)
        self.assertEqual(row["live_trade_count"], 2)
        self.assertEqual(row["backtest_trade_count"], 3)
        self.assertEqual(row["matched_symbols"], ["AMZN", "FISV"])
        self.assertEqual(row["backtest_only_symbols"], ["CRH"])
        self.assertEqual(row["live_only_symbols"], [])
        self.assertEqual(row["order_failure_count"], 3)


if __name__ == "__main__":
    unittest.main()
