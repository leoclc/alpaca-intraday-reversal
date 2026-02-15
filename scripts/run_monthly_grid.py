from __future__ import annotations

import csv
import datetime as dt
import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.daily_trend_backtest import run_backtest
from app.config.loader import load_config


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _load_monthly(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def run_grid(
    *,
    start_date: str,
    end_date: str,
    confirm_move_bps: list[float],
    confirm_minutes: list[int],
    stop_atr_mult: list[float],
    target_rr: list[float],
    time_stop_minutes: list[int],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "grid_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "confirm_move_bps",
                "confirm_minutes",
                "stop_atr_mult",
                "target_rr",
                "time_stop_minutes",
                "months",
                "avg_monthly_pct",
                "sum_monthly_pct",
                "end_equity",
            ],
        )
        writer.writeheader()

    cfg_base = load_config()
    run_idx = 0
    for cm in confirm_move_bps:
        for cmin in confirm_minutes:
            for stop_mult in stop_atr_mult:
                for rr in target_rr:
                    for tstop in time_stop_minutes:
                        run_idx += 1
                        run_id = f"run_{run_idx:03d}"
                        out_path = out_dir / run_id / "backtest.json"
                        cfg = deepcopy(cfg_base)
                        params = cfg.setdefault("daily_trend_reversal", {})
                        params["confirm_move_bps"] = float(cm)
                        params["confirm_minutes"] = int(cmin)
                        params["confirm_apply_in_watchlist"] = True
                        params["confirm_entry_price_mode"] = "close"
                        params["stop_mode"] = "atr"
                        params["stop_atr_mult"] = float(stop_mult)
                        params["target_rr"] = float(rr)
                        params["time_stop_minutes"] = int(tstop)

                        run_backtest(
                            cfg,
                            start_date=start_date,
                            end_date=end_date,
                            out_path=str(out_path),
                        )

                        monthly_path = out_path.parent / "backtest_monthly.csv"
                        rows = _load_monthly(monthly_path)
                        pnl_vals = [_to_float(r.get("total_pnl_pct")) for r in rows]
                        avg_monthly = _mean(pnl_vals)
                        sum_monthly = sum(pnl_vals)
                        end_equity = _to_float(rows[-1].get("end_equity")) if rows else 0.0
                        with results_path.open("a", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(
                                handle,
                                fieldnames=[
                                    "run_id",
                                    "confirm_move_bps",
                                    "confirm_minutes",
                                    "stop_atr_mult",
                                    "target_rr",
                                    "time_stop_minutes",
                                    "months",
                                    "avg_monthly_pct",
                                    "sum_monthly_pct",
                                    "end_equity",
                                ],
                            )
                            writer.writerow(
                                {
                                    "run_id": run_id,
                                    "confirm_move_bps": cm,
                                    "confirm_minutes": cmin,
                                    "stop_atr_mult": stop_mult,
                                    "target_rr": rr,
                                    "time_stop_minutes": tstop,
                                    "months": len(rows),
                                    "avg_monthly_pct": round(avg_monthly, 6),
                                    "sum_monthly_pct": round(sum_monthly, 6),
                                    "end_equity": round(end_equity, 6),
                                }
                            )
    return results_path


if __name__ == "__main__":
    grid_run = dt.datetime.now().strftime("grid_monthly_%Y%m%d_%H%M%S")
    base_dir = Path("logs") / "backtests" / grid_run
    config_snapshot = base_dir / "config_snapshot.json"
    base_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot.write_text(json.dumps(load_config(), indent=2), encoding="utf-8")
    run_grid(
        start_date="2025-01-01",
        end_date="2025-03-31",
        # Center around the only combo we've seen with >0 avg monthly
        confirm_move_bps=[2.5, 3.0, 3.5, 4.0],
        confirm_minutes=[2],
        stop_atr_mult=[0.5, 0.6],
        target_rr=[1.5, 1.8, 2.0],
        time_stop_minutes=[30],
        out_dir=base_dir,
    )
