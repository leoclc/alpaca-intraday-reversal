from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.backtest.daily_trend_backtest import run_backtest
from app.config.loader import load_config

_POSITION_CLOSED_RE = re.compile(
    r"position_closed symbol=(?P<symbol>\S+) "
    r"reason=(?P<reason>\S+) "
    r"side=(?P<side>\S+) "
    r"qty=(?P<qty>[0-9.]+) "
    r"entry=(?P<entry>[-+0-9.]+) "
    r"exit=(?P<exit>[-+0-9.]+) "
    r"pnl=(?P<pnl>[-+0-9.]+) "
    r"pnl_pct=(?P<pnl_pct>[-+0-9.]+)% "
    r"at=(?P<at>\S+)"
)
_ORDER_FAILURE_RE = re.compile(
    r"(fill unavailable|insufficient buying power|pending_cancel|pending cancel|rejected|order failed|close monitor failed|connectivity_lost)",
    re.IGNORECASE,
)


def parse_live_position_closed_line(line: str) -> Optional[Dict[str, Any]]:
    match = _POSITION_CLOSED_RE.search(line)
    if not match:
        return None
    trade = dict(match.groupdict())
    trade["qty"] = float(trade["qty"])
    trade["entry"] = float(trade["entry"])
    trade["exit"] = float(trade["exit"])
    trade["pnl"] = float(trade["pnl"])
    trade["pnl_pct"] = float(trade["pnl_pct"])
    return trade


def parse_live_log(log_path: Path) -> Dict[str, Any]:
    trades: List[Dict[str, Any]] = []
    failure_lines: List[str] = []
    if not log_path.exists():
        return {
            "log_path": str(log_path),
            "trade_count": 0,
            "symbols": [],
            "pnl_total": 0.0,
            "trades": trades,
            "failure_count": 0,
            "failure_lines": failure_lines,
        }
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        trade = parse_live_position_closed_line(line)
        if trade:
            trades.append(trade)
            continue
        if _ORDER_FAILURE_RE.search(line):
            failure_lines.append(line.strip())
    return {
        "log_path": str(log_path),
        "trade_count": len(trades),
        "symbols": [str(t["symbol"]) for t in trades],
        "pnl_total": round(sum(float(t["pnl"]) for t in trades), 2),
        "trades": trades,
        "failure_count": len(failure_lines),
        "failure_lines": failure_lines,
    }


def group_backtest_trades(backtest_trades_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not backtest_trades_path.exists():
        return grouped
    for line in backtest_trades_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped[str(row["entry_date"])].append(row)
    return grouped


def build_daily_parity_row(
    date_str: str,
    live_day: Dict[str, Any],
    backtest_trades: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    backtest_rows = list(backtest_trades)
    bt_symbols = [str(row["symbol"]) for row in backtest_rows]
    live_symbols = [str(symbol) for symbol in live_day.get("symbols") or []]
    live_set = set(live_symbols)
    bt_set = set(bt_symbols)
    return {
        "date": date_str,
        "live_trade_count": int(live_day.get("trade_count") or 0),
        "live_symbols": live_symbols,
        "live_pnl_total": round(float(live_day.get("pnl_total") or 0.0), 2),
        "backtest_trade_count": len(backtest_rows),
        "backtest_symbols": bt_symbols,
        "backtest_pnl_total": round(sum(float(row.get("pnl_total") or 0.0) for row in backtest_rows), 2),
        "matched_symbols": sorted(live_set & bt_set),
        "live_only_symbols": sorted(live_set - bt_set),
        "backtest_only_symbols": sorted(bt_set - live_set),
        "order_failure_count": int(live_day.get("failure_count") or 0),
    }


def _iter_dates(start_date: str, end_date: str) -> Iterable[str]:
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    cur = start
    while cur <= end:
        yield cur.isoformat()
        cur += dt.timedelta(days=1)


def _live_log_path(logs_dir: Path, date_str: str) -> Path:
    day = dt.date.fromisoformat(date_str)
    return logs_dir / f"{day:%d-%m-%Y}.log"


def _load_config_for_parity(config_path: Path) -> Dict[str, Any]:
    prev_config_path = os.environ.get("APP_CONFIG_PATH")
    os.environ["APP_CONFIG_PATH"] = str(config_path.resolve())
    try:
        cfg = load_config()
    finally:
        if prev_config_path is None:
            os.environ.pop("APP_CONFIG_PATH", None)
        else:
            os.environ["APP_CONFIG_PATH"] = prev_config_path
    params = dict(cfg.get("daily_trend_reversal") or {})
    params["backtest_reuse_existing_watchlists"] = True
    params["backtest_reuse_frozen_live_watchlists"] = True
    cfg["daily_trend_reversal"] = params
    return cfg


def _write_daily_csv(path: Path, daily_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "live_trade_count",
                "live_symbols",
                "live_pnl_total",
                "backtest_trade_count",
                "backtest_symbols",
                "backtest_pnl_total",
                "matched_symbols",
                "live_only_symbols",
                "backtest_only_symbols",
                "order_failure_count",
            ],
        )
        writer.writeheader()
        for row in daily_rows:
            writer.writerow(
                {
                    **row,
                    "live_symbols": ";".join(row["live_symbols"]),
                    "backtest_symbols": ";".join(row["backtest_symbols"]),
                    "matched_symbols": ";".join(row["matched_symbols"]),
                    "live_only_symbols": ";".join(row["live_only_symbols"]),
                    "backtest_only_symbols": ";".join(row["backtest_only_symbols"]),
                }
            )


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        f"# Live vs Backtest Parity",
        "",
        f"- Checkpoint start date: `{report['checkpoint_start_date']}`",
        f"- Planned live-capital start date: `{report['planned_live_start_date']}`",
        f"- Compared range: `{report['start_date']}` to `{report['end_date']}`",
        f"- Live trades: `{report['summary']['live_trade_count']}` / PnL `{report['summary']['live_pnl_total']:+.2f}`",
        f"- Backtest trades: `{report['summary']['backtest_trade_count']}` / PnL `{report['summary']['backtest_pnl_total']:+.2f}`",
        "",
        "| Date | Live | Backtest | Matched | Live PnL | Backtest PnL | Failures |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["daily"]:
        live_symbols = ", ".join(row["live_symbols"]) or "-"
        bt_symbols = ", ".join(row["backtest_symbols"]) or "-"
        matched = ", ".join(row["matched_symbols"]) or "-"
        lines.append(
            f"| {row['date']} | {live_symbols} | {bt_symbols} | {matched} | "
            f"{row['live_pnl_total']:+.2f} | {row['backtest_pnl_total']:+.2f} | {row['order_failure_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_live_parity_report(
    *,
    config_path: Path,
    start_date: str,
    end_date: str,
    run_dir: Path,
    out_dir: Path,
    checkpoint_start_date: Optional[str] = None,
    planned_live_start_date: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint_date = checkpoint_start_date or start_date
    if planned_live_start_date is None:
        planned_live_start_date = (dt.date.fromisoformat(checkpoint_date) + dt.timedelta(days=21)).isoformat()

    cfg = _load_config_for_parity(config_path)
    out_path = run_dir / "backtest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_backtest(cfg, start_date=start_date, end_date=end_date, out_path=str(out_path))

    backtest_trades_by_date = group_backtest_trades(run_dir / "backtest_trades.ndjson")
    logs_dir = Path(str(cfg.get("logs_dir") or "logs")) / "live"
    daily_rows: List[Dict[str, Any]] = []
    live_trade_count = 0
    backtest_trade_count = 0
    live_pnl_total = 0.0
    backtest_pnl_total = 0.0

    for date_str in _iter_dates(start_date, end_date):
        live_day = parse_live_log(_live_log_path(logs_dir, date_str))
        row = build_daily_parity_row(date_str, live_day, backtest_trades_by_date.get(date_str, []))
        daily_rows.append(row)
        live_trade_count += row["live_trade_count"]
        backtest_trade_count += row["backtest_trade_count"]
        live_pnl_total += row["live_pnl_total"]
        backtest_pnl_total += row["backtest_pnl_total"]

    report = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "checkpoint_start_date": checkpoint_date,
        "planned_live_start_date": planned_live_start_date,
        "start_date": start_date,
        "end_date": end_date,
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "summary": {
            "live_trade_count": live_trade_count,
            "backtest_trade_count": backtest_trade_count,
            "live_pnl_total": round(live_pnl_total, 2),
            "backtest_pnl_total": round(backtest_pnl_total, 2),
        },
        "daily": daily_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"parity_{start_date}_to_{end_date}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    checkpoint_path = out_dir / f"checkpoint_{checkpoint_date}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_daily_csv(csv_path, daily_rows)
    _write_markdown(md_path, report)
    checkpoint_path.write_text(
        json.dumps(
            {
                "checkpoint_start_date": checkpoint_date,
                "planned_live_start_date": planned_live_start_date,
                "created_at": report["generated_at"],
                "initial_report_json": str(json_path.resolve()),
                "initial_report_markdown": str(md_path.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report["artifacts"] = {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    return report
