from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import time
import requests

from app.utils.time import ensure_date


def _symbol_path(base_dir: Path, symbol: str) -> Path:
    safe = str(symbol).upper().strip()
    return base_dir / f"{safe}.csv"


def _parse_timestamp(val: Any) -> Optional[dt.datetime]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        ts = float(val)
        if ts > 1e12:
            ts = ts / 1e9
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    s = str(val)
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


class AlpacaOHLCStore:
    def __init__(self, base_dir: Optional[str] = None, cfg: Optional[dict] = None) -> None:
        base = base_dir or (cfg or {}).get("daily_bars_cache_dir") or "alpaca-ohlc"
        self.base_dir = Path(base)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_symbol_bars(self, symbol: str) -> List[Dict[str, Any]]:
        path = _symbol_path(self.base_dir, symbol)
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("date"):
                    continue
                try:
                    # Validate date before keeping the row.
                    _ = ensure_date(str(row["date"]))
                except Exception:
                    continue
                try:
                    rows.append(
                        {
                            "date": str(row["date"]),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                            "vwap": float(row["vwap"]) if row.get("vwap") not in (None, "", "nan") else None,
                        }
                    )
                except Exception:
                    continue
        rows.sort(key=lambda r: r["date"])
        return rows

    def write_symbol_bars(self, symbol: str, bars: Iterable[Dict[str, Any]]) -> Path:
        existing = {row["date"]: row for row in self.load_symbol_bars(symbol)}
        for bar in bars:
            if not bar.get("date"):
                continue
            existing[str(bar["date"])] = {
                "date": str(bar["date"]),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
                "vwap": float(bar["vwap"]) if bar.get("vwap") is not None else None,
            }
        merged = sorted(existing.values(), key=lambda r: r["date"])
        path = _symbol_path(self.base_dir, symbol)
        with path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["date", "open", "high", "low", "close", "volume", "vwap"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged:
                writer.writerow(row)
        return path

    def _dates_covered(self, bars: List[Dict[str, Any]]) -> Optional[tuple]:
        if not bars:
            return None
        try:
            return bars[0]["date"], bars[-1]["date"]
        except Exception:
            return None

    def get_daily_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cfg: Optional[dict] = None,
        allow_fetch: bool = True,
    ) -> List[Dict[str, Any]]:
        out = self.get_daily_bars_bulk([symbol], start_date, end_date, cfg=cfg, allow_fetch=allow_fetch)
        return out.get(symbol.upper(), [])

    def get_daily_bars_bulk(
        self,
        symbols: Iterable[str],
        start_date: Optional[str],
        end_date: Optional[str],
        cfg: Optional[dict] = None,
        allow_fetch: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        cfg = cfg or {}
        sym_list = [str(s).upper() for s in symbols if s]
        out: Dict[str, List[Dict[str, Any]]] = {}
        missing: List[str] = []
        for sym in sym_list:
            bars = self.load_symbol_bars(sym)
            if start_date or end_date:
                bars = _filter_bars(bars, start_date, end_date)
            out[sym] = bars
            covered = self._dates_covered(bars)
            if allow_fetch and (not covered or _range_missing(covered, start_date, end_date)):
                missing.append(sym)
        if missing and allow_fetch:
            fetched = self._fetch_bars_bulk(missing, start_date, end_date, cfg)
            for sym, bars in fetched.items():
                if bars:
                    self.write_symbol_bars(sym, bars)
                merged = self.load_symbol_bars(sym)
                out[sym] = _filter_bars(merged, start_date, end_date)
        return out

    def _fetch_bars_bulk(
        self,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        cfg: dict,
    ) -> Dict[str, List[Dict[str, Any]]]:
        alp = cfg.get("alpaca") or {}
        key_id = str(alp.get("api_key_id") or "").strip()
        secret = str(alp.get("api_secret_key") or "").strip()
        if not key_id or not secret:
            return {}
        chunk_size = int(alp.get("bars_chunk_size") or 50)
        out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            fetched = self._fetch_bars_alpaca(chunk, start_date, end_date, cfg)
            for sym, bars in fetched.items():
                out.setdefault(sym, []).extend(bars)
        return out

    def _fetch_bars_alpaca(
        self,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        cfg: dict,
    ) -> Dict[str, List[Dict[str, Any]]]:
        alp = cfg.get("alpaca") or {}
        base_url = str(alp.get("data_url") or "https://data.alpaca.markets").rstrip("/")
        url = f"{base_url}/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": str(alp.get("api_key_id") or ""),
            "APCA-API-SECRET-KEY": str(alp.get("api_secret_key") or ""),
        }
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "adjustment": str(alp.get("adjustment") or "raw"),
            "feed": str(alp.get("data_feed") or "iex"),
            "limit": 10000,
        }
        if start_date:
            params["start"] = str(start_date)
        if end_date:
            params["end"] = str(end_date)
        timeout = alp.get("timeout_sec") or 10
        max_retries = int(alp.get("max_retries") or 0)
        retry_backoff = float(alp.get("retry_backoff_sec") or 0)
        out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            attempt = 0
            while True:
                try:
                    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except Exception:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    if retry_backoff > 0:
                        time.sleep(retry_backoff * attempt)
            _parse_bars_payload(payload, out)
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        return out


def _parse_bars_payload(payload: dict, out: Dict[str, List[Dict[str, Any]]]) -> None:
    bars = payload.get("bars")
    if isinstance(bars, dict):
        for sym, rows in bars.items():
            for row in rows or []:
                _append_bar(out, sym, row)
        return
    if isinstance(bars, list):
        symbol = payload.get("symbol")
        for row in bars:
            _append_bar(out, symbol, row)


def _append_bar(out: Dict[str, List[Dict[str, Any]]], symbol: Optional[str], row: dict) -> None:
    if not symbol:
        return
    ts = _parse_timestamp(row.get("t") or row.get("timestamp"))
    if not ts:
        return
    date_str = ts.date().isoformat()
    out.setdefault(symbol.upper(), []).append(
        {
            "date": date_str,
            "open": float(row.get("o") or row.get("open") or 0),
            "high": float(row.get("h") or row.get("high") or 0),
            "low": float(row.get("l") or row.get("low") or 0),
            "close": float(row.get("c") or row.get("close") or 0),
            "volume": float(row.get("v") or row.get("volume") or 0),
            "vwap": float(row.get("vw") or row.get("vwap") or 0) if row.get("vw") or row.get("vwap") else None,
        }
    )


def _range_missing(covered: tuple, start_date: Optional[str], end_date: Optional[str]) -> bool:
    if not covered:
        return True
    covered_start, covered_end = covered
    if start_date and str(start_date) < str(covered_start):
        return True
    if end_date and str(end_date) > str(covered_end):
        return True
    return False


def _filter_bars(
    bars: List[Dict[str, Any]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    if not bars:
        return []
    if not start_date and not end_date:
        return list(bars)
    out: List[Dict[str, Any]] = []
    start = ensure_date(start_date) if start_date else None
    end = ensure_date(end_date) if end_date else None
    for bar in bars:
        try:
            bdate = ensure_date(str(bar.get("date")))
        except Exception:
            continue
        if start and bdate < start:
            continue
        if end and bdate > end:
            continue
        out.append(bar)
    return out
