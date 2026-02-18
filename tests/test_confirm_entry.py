import datetime as dt
import unittest

from app.data.alpaca_ohlc_store import AlpacaOHLCStore
from app.strategies.daily_trend_reversal import build_trade
from app.strategies.types import Signal
from tests.helpers import clean_dir, repo_root, write_bars_csv


class TestConfirmEntry(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = repo_root() / "alpaca-ohlc"
        clean_dir(self.base_dir)

    def test_confirm_enters_at_cutoff_price(self):
        symbol = "CONFIRM"
        daily_rows = [
            {"date": "2025-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-03", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-06", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-07", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-08", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-09", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
        ]
        write_bars_csv(self.base_dir / f"{symbol}.csv", daily_rows)
        cfg = {
            "daily_trend_reversal": {
                "atr_period": 3,
                "stop_mode": "atr",
                "stop_atr_mult": 1.0,
                "target_rr": 1.5,
                "time_exit_days": 2,
                "confirm_move_bps": 10.0,
                "confirm_minutes": 3,
                "confirm_entry_price_mode": "close",
                "use_intraday_entry": False,
                "entry_time_et": "09:35",
            },
            "daily_bars_cache_dir": str(self.base_dir),
        }
        data_store = AlpacaOHLCStore(cfg=cfg)

        sig = Signal(
            symbol=symbol,
            signal_date="2025-01-09",
            direction="long",
            trend_state="uptrend",
            return_pct=-2.5,
        )

        et = dt.timezone(dt.timedelta(hours=-5))
        def _bar(hh: int, mm: int, o: float, h: float, l: float, c: float) -> dict:
            ts = dt.datetime(2025, 1, 9, hh, mm, tzinfo=et).isoformat()
            return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000}

        # Entry time is 09:35, confirm window is [09:35, 09:38). We include a 09:38 bar with absurd values to
        # ensure we do not accidentally "use" the cutoff bar (lookahead).
        bars_intraday = [
            _bar(9, 30, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 31, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 32, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 33, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 34, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 35, 100.00, 100.05, 99.95, 100.02),
            _bar(9, 36, 100.02, 100.20, 100.00, 100.15),  # confirm hit: +20 bps vs 100.00
            _bar(9, 37, 100.15, 100.16, 100.01, 100.05),  # last completed bar before cutoff (09:38)
            _bar(9, 38, 200.00, 200.00, 200.00, 200.00),  # must not be used
        ]

        plan = build_trade(
            sig,
            cfg,
            data_store,
            bars_intraday=bars_intraday,
            entry_time_override="09:35",
        )
        self.assertIsNotNone(plan)
        assert plan is not None

        # Confirmation triggers entry at the end of the window (cutoff), using the last completed bar before cutoff.
        self.assertEqual(plan.entry_time_et, "09:38")
        self.assertAlmostEqual(plan.entry_price, 100.05, places=6)
        self.assertIsNotNone(plan.confirm_hit_bps)
        self.assertGreaterEqual(float(plan.confirm_hit_bps or 0.0), 20.0)

    def test_directional_confirm_cap_applies_to_short_only(self):
        symbol = "CONFIRM_CAP"
        daily_rows = [
            {"date": "2025-01-02", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-03", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-06", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-07", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-08", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-09", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
        ]
        write_bars_csv(self.base_dir / f"{symbol}.csv", daily_rows)
        cfg = {
            "daily_trend_reversal": {
                "atr_period": 3,
                "stop_mode": "atr",
                "stop_atr_mult": 1.0,
                "target_rr": 1.5,
                "time_exit_days": 2,
                "confirm_move_bps": 10.0,
                "confirm_minutes": 3,
                "confirm_entry_price_mode": "close",
                "max_confirm_hit_bps_long": 30.0,
                "max_confirm_hit_bps_short": 15.0,
                "use_intraday_entry": False,
                "entry_time_et": "09:35",
            },
            "daily_bars_cache_dir": str(self.base_dir),
        }
        data_store = AlpacaOHLCStore(cfg=cfg)

        sig_long = Signal(
            symbol=symbol,
            signal_date="2025-01-09",
            direction="long",
            trend_state="uptrend",
            return_pct=-2.5,
        )
        sig_short = Signal(
            symbol=symbol,
            signal_date="2025-01-09",
            direction="short",
            trend_state="downtrend",
            return_pct=2.5,
        )

        et = dt.timezone(dt.timedelta(hours=-5))

        def _bar(hh: int, mm: int, o: float, h: float, l: float, c: float) -> dict:
            ts = dt.datetime(2025, 1, 9, hh, mm, tzinfo=et).isoformat()
            return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000}

        bars_intraday = [
            _bar(9, 30, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 31, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 32, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 33, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 34, 100.00, 100.01, 99.99, 100.00),
            _bar(9, 35, 100.00, 100.05, 99.95, 100.02),
            # Both long and short see a 20 bps confirm move inside the window.
            _bar(9, 36, 100.02, 100.20, 99.80, 100.10),
            _bar(9, 37, 100.10, 100.12, 99.95, 100.05),
            _bar(9, 38, 200.00, 200.00, 200.00, 200.00),
        ]

        plan_long = build_trade(
            sig_long,
            cfg,
            data_store,
            bars_intraday=bars_intraday,
            entry_time_override="09:35",
        )
        self.assertIsNotNone(plan_long)
        assert plan_long is not None
        self.assertEqual(plan_long.entry_time_et, "09:38")

        plan_short = build_trade(
            sig_short,
            cfg,
            data_store,
            bars_intraday=bars_intraday,
            entry_time_override="09:35",
        )
        # Short confirm move is ~20 bps and should be rejected by the short-only cap of 15 bps.
        self.assertIsNone(plan_short)

    def test_directional_stop_floor_applies_by_direction(self):
        symbol = "STOP_FLOOR_DIR"
        daily_rows = [
            {"date": "2025-01-02", "open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1_000_000, "vwap": 100.0},
            {"date": "2025-01-03", "open": 100.0, "high": 104.0, "low": 98.0, "close": 101.0, "volume": 1_000_000, "vwap": 101.0},
            {"date": "2025-01-06", "open": 101.0, "high": 105.0, "low": 99.0, "close": 102.0, "volume": 1_000_000, "vwap": 102.0},
            {"date": "2025-01-07", "open": 102.0, "high": 106.0, "low": 100.0, "close": 103.0, "volume": 1_000_000, "vwap": 103.0},
            {"date": "2025-01-08", "open": 103.0, "high": 107.0, "low": 101.0, "close": 104.0, "volume": 1_000_000, "vwap": 104.0},
            {"date": "2025-01-09", "open": 104.0, "high": 108.0, "low": 102.0, "close": 105.0, "volume": 1_000_000, "vwap": 105.0},
        ]
        write_bars_csv(self.base_dir / f"{symbol}.csv", daily_rows)
        cfg = {
            "daily_trend_reversal": {
                "atr_period": 3,
                "stop_mode": "atr",
                "stop_atr_mult": 0.05,
                "stop_r": 1.0,
                "target_rr": 1.5,
                "time_exit_days": 2,
                "confirm_move_bps": 0.0,
                "confirm_minutes": 0,
                "min_stop_atr_long": 0.20,
                "min_stop_atr_short": 0.05,
                "use_intraday_entry": False,
                "entry_time_et": "09:35",
            },
            "daily_bars_cache_dir": str(self.base_dir),
        }
        data_store = AlpacaOHLCStore(cfg=cfg)

        sig_long = Signal(
            symbol=symbol,
            signal_date="2025-01-09",
            direction="long",
            trend_state="uptrend",
            return_pct=-2.5,
        )
        sig_short = Signal(
            symbol=symbol,
            signal_date="2025-01-09",
            direction="short",
            trend_state="downtrend",
            return_pct=2.5,
        )

        plan_long = build_trade(sig_long, cfg, data_store, entry_time_override="09:35")
        plan_short = build_trade(sig_short, cfg, data_store, entry_time_override="09:35")
        self.assertIsNotNone(plan_long)
        self.assertIsNotNone(plan_short)
        assert plan_long is not None
        assert plan_short is not None
        self.assertIsNotNone(plan_long.atr)
        self.assertIsNotNone(plan_short.atr)
        assert plan_long.atr is not None
        assert plan_short.atr is not None

        long_stop_atr = plan_long.stop_distance / float(plan_long.atr)
        short_stop_atr = plan_short.stop_distance / float(plan_short.atr)
        # Long uses the larger directional floor; short keeps the smaller one.
        self.assertAlmostEqual(long_stop_atr, 0.20, places=6)
        self.assertAlmostEqual(short_stop_atr, 0.05, places=6)
        self.assertGreater(plan_long.stop_distance, plan_short.stop_distance)


if __name__ == "__main__":
    unittest.main()
