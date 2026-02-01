from __future__ import annotations

import csv
import datetime as dt
import shutil
from pathlib import Path
from typing import Iterable, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def generate_trading_days(start_date: str, num_days: int) -> List[dt.date]:
    start = dt.date.fromisoformat(start_date)
    days: List[dt.date] = []
    cur = start
    while len(days) < num_days:
        if cur.weekday() < 5:
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def write_bars_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "open", "high", "low", "close", "volume", "vwap"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def seed_trending_universe(
    base_dir: Path,
    symbols: List[str],
    start_date: str,
    num_days: int,
    dip_every: int = 7,
    growth_pct: float = 0.005,
    dip_multiplier: float = 0.975,
) -> str:
    days = generate_trading_days(start_date, num_days)
    for idx, symbol in enumerate(symbols):
        rows = []
        prev_close = 100.0 + idx * 0.05
        for i, day in enumerate(days):
            if i > 0 and dip_every > 0 and i % dip_every == 0:
                close = prev_close * dip_multiplier
            else:
                close = prev_close * (1.0 + growth_pct)
            open_price = prev_close
            high = max(open_price, close) * 1.002
            low = min(open_price, close) * 0.998
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": 1_000_000,
                    "vwap": round((high + low + close) / 3.0, 4),
                }
            )
            prev_close = close
        write_bars_csv(base_dir / f"{symbol}.csv", rows)
    return days[-1].isoformat()
