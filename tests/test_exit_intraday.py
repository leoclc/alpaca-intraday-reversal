import datetime as dt
import unittest

from app.execution.daily_execution_model import simulate_exit
from app.strategies.types import TradePlan


class TestIntradayExitParity(unittest.TestCase):
    def test_time_stop_excludes_cutoff_bar(self):
        cfg = {
            "daily_trend_reversal": {
                "time_stop_minutes": 3,
                "intraday_only": False,
                "stop_first_when_both": True,
            }
        }
        bars_daily = [
            {"date": "2025-01-09", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}
        ]
        plan = TradePlan(
            symbol="EXIT",
            direction="long",
            signal_date="2025-01-09",
            entry_date="2025-01-09",
            entry_time_et="09:35",
            entry_price=100.0,
            stop_price=99.0,
            target_price=101.0,
            time_exit_date="2025-01-09",
            stop_distance=1.0,
            target_rr=1.0,
        )

        et = dt.timezone(dt.timedelta(hours=-5))

        def _bar(hh: int, mm: int, o: float, h: float, l: float, c: float) -> dict:
            ts = dt.datetime(2025, 1, 9, hh, mm, tzinfo=et).isoformat()
            return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000}

        # Cutoff is 09:38. Include a 09:38 bar with absurd values to ensure it is excluded
        # (otherwise we'd incorrectly hit stop/target inside that in-progress bar).
        bars_intraday = [
            _bar(9, 35, 100.00, 100.50, 99.50, 100.10),
            _bar(9, 36, 100.10, 100.40, 99.60, 100.15),
            _bar(9, 37, 100.15, 100.30, 99.70, 100.20),  # last completed bar before cutoff
            _bar(9, 38, 200.00, 200.00, 0.01, 200.00),  # must not be used
        ]

        exit_info = simulate_exit(plan, "daily", bars_daily, bars_intraday, cfg)
        self.assertIsNotNone(exit_info)
        assert exit_info is not None
        self.assertEqual(exit_info["exit_reason"], "time_stop")
        self.assertAlmostEqual(float(exit_info["exit_price"]), 100.20, places=6)

    def test_no_daily_fallback_when_intraday_required(self):
        cfg = {
            "daily_trend_reversal": {
                "time_stop_minutes": 3,
                "intraday_only": False,
            }
        }
        bars_daily = [
            {"date": "2025-01-09", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}
        ]
        plan = TradePlan(
            symbol="EXIT",
            direction="long",
            signal_date="2025-01-09",
            entry_date="2025-01-09",
            entry_time_et="09:35",
            entry_price=100.0,
            stop_price=99.0,
            target_price=101.0,
            time_exit_date="2025-01-09",
            stop_distance=1.0,
            target_rr=1.0,
        )
        # Intraday is required because time_stop_minutes > 0; missing intraday bars must not fall back to daily.
        self.assertIsNone(simulate_exit(plan, "daily", bars_daily, None, cfg))


if __name__ == "__main__":
    unittest.main()

