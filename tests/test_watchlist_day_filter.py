import unittest

from app.watchlist.day_filter import day_filter_decision, summarize_watchlist_rows


class TestWatchlistDayFilter(unittest.TestCase):
    def test_summarize_watchlist_rows_counts_and_means(self):
        rows = [
            {"symbol": "AAA", "direction": "long", "avgR": 0.2, "win_rate": 0.6, "profit_factor": 1.5, "total_pnl_pct": 2.0},
            {"symbol": "BBB", "direction": "short", "avgR": -0.1, "win_rate": 0.4, "profit_factor": 0.8, "total_pnl_pct": -1.0},
        ]
        s = summarize_watchlist_rows(rows)
        self.assertEqual(s["size"], 2)
        self.assertEqual(s["long_count"], 1)
        self.assertEqual(s["short_count"], 1)
        self.assertAlmostEqual(s["avg_selected_avgR"], 0.05, places=6)
        self.assertAlmostEqual(s["avg_selected_win_rate"], 0.5, places=6)
        self.assertAlmostEqual(s["avg_selected_profit_factor"], 1.15, places=6)

    def test_day_filter_disabled(self):
        rows = [{"symbol": "AAA", "direction": "long", "avgR": 0.2, "win_rate": 0.6, "profit_factor": 1.5}]
        cfg = {"watchlist": {"day_kill_switch_enabled": False, "day_min_watchlist_size": 10}}
        skip, info = day_filter_decision(rows, cfg)
        self.assertFalse(skip)
        self.assertEqual(info, {})

    def test_day_filter_enabled_thresholds(self):
        rows = [
            {"symbol": "AAA", "direction": "long", "avgR": -0.2, "win_rate": 0.35, "profit_factor": 0.7, "total_pnl_pct": -2.0},
            {"symbol": "BBB", "direction": "long", "avgR": -0.1, "win_rate": 0.4, "profit_factor": 0.8, "total_pnl_pct": -1.0},
        ]
        cfg = {
            "watchlist": {
                "day_kill_switch_enabled": True,
                "day_min_watchlist_size": 3,
                "day_min_selected_avgR": 0.0,
                "day_min_selected_win_rate": 0.5,
                "day_min_selected_profit_factor": 1.0,
                "day_min_long_count": 1,
                "day_min_short_count": 1,
            }
        }
        skip, info = day_filter_decision(rows, cfg)
        self.assertTrue(skip)
        reasons = set(info.get("reasons") or [])
        self.assertIn("min_watchlist_size", reasons)
        self.assertIn("min_selected_avgR", reasons)
        self.assertIn("min_selected_win_rate", reasons)
        self.assertIn("min_selected_profit_factor", reasons)
        self.assertIn("min_short_count", reasons)


if __name__ == "__main__":
    unittest.main()
