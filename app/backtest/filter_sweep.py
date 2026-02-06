from __future__ import annotations

import copy
import csv
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Dict, List

from app.backtest.daily_trend_backtest import run_backtest
from app.config.loader import load_config


def _run_variant(cfg: Dict, name: str, out_dir: Path) -> Dict:
    out_path = out_dir / name / "backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary, _ = run_backtest(cfg, out_path=str(out_path))
    payload = {
        "name": name,
        "trades": summary.get("trades"),
        "win_rate": summary.get("win_rate"),
        "avgR": summary.get("avgR"),
        "total_pnl_pct": summary.get("total_pnl_pct"),
        "market_filter_skips": summary.get("market_filter_skips"),
        "market_filter_skip_reasons": summary.get("market_filter_skip_reasons"),
        "backtest_path": str(out_path),
    }
    return payload


def run_filter_sweep(cfg: Dict | None = None, start_date: str | None = None, end_date: str | None = None) -> Path:
    cfg = cfg or load_config()
    if start_date:
        cfg.setdefault("replay", {})["start_date"] = start_date
    if end_date:
        cfg.setdefault("replay", {})["end_date"] = end_date

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
    out_dir = logs_dir / "filter_sweeps" / run_id

    variants: List[tuple[str, Dict]] = []

    base = copy.deepcopy(cfg)
    base.setdefault("market_filters", {})["enabled"] = False
    variants.append(("baseline", base))

    atr_only = copy.deepcopy(cfg)
    mf = atr_only.setdefault("market_filters", {})
    mf.update({"enabled": True, "use_atr": True, "use_gap": False, "use_trend": False})
    variants.append(("atr_filter", atr_only))

    gap_only = copy.deepcopy(cfg)
    mf = gap_only.setdefault("market_filters", {})
    mf.update({"enabled": True, "use_atr": False, "use_gap": True, "use_trend": False})
    variants.append(("gap_filter", gap_only))

    trend_only = copy.deepcopy(cfg)
    mf = trend_only.setdefault("market_filters", {})
    mf.update({"enabled": True, "use_atr": False, "use_gap": False, "use_trend": True})
    variants.append(("trend_filter", trend_only))

    rows = []
    for name, variant in variants:
        logging.info("[SWEEP] running %s", name)
        rows.append(_run_variant(variant, name, out_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "filter_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logging.info("[SWEEP] wrote %s", csv_path)
    return csv_path


def _parse_float_list(raw: str | None, fallback: List[float]) -> List[float]:
    if not raw:
        return fallback
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            continue
    return values or fallback


def run_filter_threshold_sweep(
    cfg: Dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    cfg = cfg or load_config()
    if start_date:
        cfg.setdefault("replay", {})["start_date"] = start_date
    if end_date:
        cfg.setdefault("replay", {})["end_date"] = end_date

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path(str(cfg.get("logs_dir") or "logs"))
    out_dir = logs_dir / "filter_sweeps" / run_id

    atr_list = _parse_float_list(os.getenv("FILTER_ATR_MAX_LIST"), [1.5, 2.0, 2.5])
    gap_list = _parse_float_list(os.getenv("FILTER_GAP_BPS_LIST"), [40.0, 60.0, 80.0])
    trend_list = _parse_float_list(os.getenv("FILTER_TREND_MA_LIST"), [200.0])

    variants: List[tuple[str, Dict]] = []

    base = copy.deepcopy(cfg)
    base.setdefault("market_filters", {})["enabled"] = False
    variants.append(("baseline", base))

    for atr_max in atr_list:
        variant = copy.deepcopy(cfg)
        mf = variant.setdefault("market_filters", {})
        mf.update(
            {
                "enabled": True,
                "use_atr": True,
                "use_gap": False,
                "use_trend": False,
                "atr_max_pct": atr_max,
            }
        )
        variants.append((f"atr_max_pct_{atr_max}", variant))

    for gap_max in gap_list:
        variant = copy.deepcopy(cfg)
        mf = variant.setdefault("market_filters", {})
        mf.update(
            {
                "enabled": True,
                "use_atr": False,
                "use_gap": True,
                "use_trend": False,
                "gap_bps_max": gap_max,
            }
        )
        variants.append((f"gap_bps_max_{gap_max}", variant))

    for trend_days in trend_list:
        variant = copy.deepcopy(cfg)
        mf = variant.setdefault("market_filters", {})
        mf.update(
            {
                "enabled": True,
                "use_atr": False,
                "use_gap": False,
                "use_trend": True,
                "trend_ma_days": int(trend_days),
            }
        )
        variants.append((f"trend_ma_{int(trend_days)}", variant))

    rows = []
    for name, variant in variants:
        logging.info("[SWEEP] running %s", name)
        row = _run_variant(variant, name, out_dir)
        mf = variant.get("market_filters") or {}
        row.update(
            {
                "atr_max_pct": mf.get("atr_max_pct"),
                "gap_bps_max": mf.get("gap_bps_max"),
                "trend_ma_days": mf.get("trend_ma_days") if mf.get("use_trend") else None,
            }
        )
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "filter_threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logging.info("[SWEEP] wrote %s", csv_path)
    return csv_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    start = os.getenv("FILTER_SWEEP_START")
    end = os.getenv("FILTER_SWEEP_END")
    mode = os.getenv("FILTER_SWEEP_MODE") or "basic"
    if mode == "threshold":
        run_filter_threshold_sweep(start_date=start, end_date=end)
    else:
        run_filter_sweep(start_date=start, end_date=end)
