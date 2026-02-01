from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.utils.time import ET_TZ, ensure_date, parse_time_hhmm

_MEM_CACHE: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}


def _cache_path(base_dir: Path, symbol: str, session_date: dt.date, minutes: int) -> Path:
    safe = str(symbol).upper().strip()
    base_dir = base_dir / safe
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{session_date.isoformat()}_{int(minutes)}m.json"


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


def _session_window(session_date: dt.date, minutes: int, cfg: dict) -> Tuple[dt.datetime, dt.datetime]:
    params = cfg.get("daily_trend_reversal") or {}
    session_open_et = str(params.get("session_open_et") or "09:30")
    start_et = dt.datetime.combine(session_date, parse_time_hhmm(session_open_et))
    if start_et.tzinfo is None:
        start_et = start_et.replace(tzinfo=ET_TZ)
    end_et = start_et + dt.timedelta(minutes=max(1, int(minutes)))
    start_utc = start_et.astimezone(dt.timezone.utc)
    end_utc = end_et.astimezone(dt.timezone.utc)
    return start_utc, end_utc


def _fetch_intraday_bars(symbol: str, session_date: dt.date, minutes: int, cfg: dict) -> List[Dict[str, Any]]:
    alp = cfg.get("alpaca") or {}
    key_id = str(alp.get("api_key_id") or "").strip()
    secret = str(alp.get("api_secret_key") or "").strip()
    if not key_id or not secret:
        return []
    base_url = str(alp.get("data_url") or "https://data.alpaca.markets").rstrip("/")
    url = f"{base_url}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
    }
    start_utc, end_utc = _session_window(session_date, minutes, cfg)
    params = {
        "symbols": str(symbol).upper(),
        "timeframe": "1Min",
        "adjustment": str(alp.get("adjustment") or "raw"),
        "feed": str(alp.get("data_feed") or "iex"),
        "limit": 10000,
        "start": start_utc.isoformat(),
        "end": end_utc.isoformat(),
    }
    timeout = alp.get("timeout_sec") or 10
    max_retries = int(alp.get("max_retries") or 0)
    retry_backoff = float(alp.get("retry_backoff_sec") or 0)
    out: List[Dict[str, Any]] = []
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
        bars_payload = payload.get("bars")
        if isinstance(bars_payload, dict):
            rows = bars_payload.get(str(symbol).upper()) or bars_payload.get(str(symbol)) or []
        elif isinstance(bars_payload, list):
            rows = bars_payload
        else:
            rows = []
        for row in rows or []:
            ts = _parse_timestamp(row.get("t") or row.get("timestamp"))
            if not ts:
                continue
            out.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": float(row.get("o") or row.get("open") or 0),
                    "high": float(row.get("h") or row.get("high") or 0),
                    "low": float(row.get("l") or row.get("low") or 0),
                    "close": float(row.get("c") or row.get("close") or 0),
                    "volume": float(row.get("v") or row.get("volume") or 0),
                }
            )
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    out.sort(key=lambda b: b.get("timestamp") or "")
    return out


def _fetch_recent_bars(
    symbols: List[str],
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    cfg: dict,
) -> Dict[str, List[Dict[str, Any]]]:
    alp = cfg.get("alpaca") or {}
    key_id = str(alp.get("api_key_id") or "").strip()
    secret = str(alp.get("api_secret_key") or "").strip()
    if not key_id or not secret:
        return {}
    base_url = str(alp.get("data_url") or "https://data.alpaca.markets").rstrip("/")
    url = f"{base_url}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
    }
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Min",
        "adjustment": str(alp.get("adjustment") or "raw"),
        "feed": str(alp.get("data_feed") or "iex"),
        "limit": 10000,
        "start": start_utc.isoformat(),
        "end": end_utc.isoformat(),
    }
    timeout = alp.get("timeout_sec") or 10
    max_retries = int(alp.get("max_retries") or 0)
    retry_backoff = float(alp.get("retry_backoff_sec") or 0)
    out: Dict[str, List[Dict[str, Any]]] = {str(s).upper(): [] for s in symbols}
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
        bars_payload = payload.get("bars")
        if isinstance(bars_payload, dict):
            for sym, rows in bars_payload.items():
                sym_key = str(sym).upper()
                for row in rows or []:
                    ts = _parse_timestamp(row.get("t") or row.get("timestamp"))
                    if not ts:
                        continue
                    out.setdefault(sym_key, []).append(
                        {
                            "timestamp": ts.isoformat(),
                            "open": float(row.get("o") or row.get("open") or 0),
                            "high": float(row.get("h") or row.get("high") or 0),
                            "low": float(row.get("l") or row.get("low") or 0),
                            "close": float(row.get("c") or row.get("close") or 0),
                            "volume": float(row.get("v") or row.get("volume") or 0),
                        }
                    )
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    for sym, rows in out.items():
        rows.sort(key=lambda b: b.get("timestamp") or "")
    return out


def get_intraday_bars(
    symbol: str,
    session_date: str | dt.date,
    minutes: int,
    cfg: Optional[dict] = None,
    allow_fetch: bool = True,
) -> List[Dict[str, Any]]:
    cfg = cfg or {}
    minutes = int(minutes or 0)
    if minutes <= 0 or not symbol:
        return []
    session_dt = ensure_date(session_date)
    cache_dir = Path(cfg.get("minute_bars_cache_dir") or "alpaca-minute")
    cache_file = _cache_path(cache_dir, symbol, session_dt, minutes)
    cache_key = (str(symbol).upper(), session_dt.isoformat(), minutes)
    if cache_key in _MEM_CACHE:
        return _MEM_CACHE[cache_key]
    refresh = bool(cfg.get("minute_bars_refresh", False))
    if cache_file.exists() and not refresh:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                _MEM_CACHE[cache_key] = cached
                return cached
        except Exception:
            pass
    if not allow_fetch:
        return []
    try:
        bars = _fetch_intraday_bars(symbol, session_dt, minutes, cfg)
    except Exception as exc:
        logging.warning("[INTRADAY] fetch failed sym=%s date=%s: %s", symbol, session_dt, exc)
        return []
    if bars:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(bars), encoding="utf-8")
        except Exception:
            pass
    _MEM_CACHE[cache_key] = bars
    return bars


def get_latest_intraday_prices(
    symbols: List[str],
    cfg: Optional[dict] = None,
    lookback_minutes: int = 5,
) -> Dict[str, float]:
    cfg = cfg or {}
    symbols = [str(s).upper() for s in symbols if s]
    if not symbols:
        return {}
    lookback_minutes = max(1, int(lookback_minutes or 1))
    end_utc = dt.datetime.now(tz=dt.timezone.utc)
    start_utc = end_utc - dt.timedelta(minutes=lookback_minutes)
    chunk_size = int((cfg.get("alpaca") or {}).get("bars_chunk_size") or 50)
    prices: Dict[str, float] = {}
    for i in range(0, len(symbols), max(1, chunk_size)):
        chunk = symbols[i : i + max(1, chunk_size)]
        try:
            bars_map = _fetch_recent_bars(chunk, start_utc, end_utc, cfg)
        except Exception as exc:
            logging.warning("[INTRADAY] latest bars fetch failed chunk=%s: %s", chunk, exc)
            continue
        for sym, rows in bars_map.items():
            if rows:
                prices[sym] = float(rows[-1].get("close") or 0.0)
    return prices
