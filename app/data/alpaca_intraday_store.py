from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.utils.time import ET_TZ, ensure_date, ensure_et, parse_time_hhmm

_MEM_CACHE: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
_MINUTE_BARS_CACHE_VERSION = 2


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
    # NOTE: ET_TZ is a pytz timezone in our environment; we must localize (ensure_et)
    # rather than doing .replace(tzinfo=ET_TZ), otherwise DST/LMT offsets are wrong.
    start_et = ensure_et(dt.datetime.combine(session_date, parse_time_hhmm(session_open_et)))
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
    alp = cfg.get("alpaca") or {}
    feed_tok = "".join(ch for ch in str(alp.get("data_feed") or "x").lower() if ch.isalnum() or ch in ("-", "_")) or "x"
    adj_tok = "".join(ch for ch in str(alp.get("adjustment") or "x").lower() if ch.isalnum() or ch in ("-", "_")) or "x"
    cache_ns = f"v{_MINUTE_BARS_CACHE_VERSION}_{feed_tok}_{adj_tok}"
    cache_dir = Path(cfg.get("minute_bars_cache_dir") or "alpaca-minute") / cache_ns
    cache_file = _cache_path(cache_dir, symbol, session_dt, minutes)
    cache_key = (str(symbol).upper(), f"{session_dt.isoformat()}|{cache_ns}", minutes)
    if cache_key in _MEM_CACHE:
        return _MEM_CACHE[cache_key]
    refresh = bool(cfg.get("minute_bars_refresh", False))

    def _has_expected_coverage(rows: List[Dict[str, Any]]) -> bool:
        params = cfg.get("daily_trend_reversal") or {}
        session_open_et = str(params.get("session_open_et") or "09:30")
        open_time = parse_time_hhmm(session_open_et)
        try:
            open_dt = ensure_et(dt.datetime.combine(session_dt, open_time))
            required_last_dt = open_dt + dt.timedelta(minutes=max(1, int(minutes)) - 1)
        except Exception:
            required_last_dt = None
        saw_open = False
        last_ts: Optional[dt.datetime] = None
        for row in rows or []:
            ts = _parse_timestamp(row.get("timestamp") or row.get("t") or row.get("time") or row.get("date"))
            if not ts:
                continue
            ts_et = ensure_et(ts)
            if ts_et.hour == open_time.hour and ts_et.minute == open_time.minute:
                saw_open = True
            if last_ts is None or ts_et > last_ts:
                last_ts = ts_et
        if not saw_open:
            return False
        if required_last_dt is None:
            return True
        if last_ts is None:
            return False
        return last_ts >= required_last_dt

    def _slice_to_minutes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        params = cfg.get("daily_trend_reversal") or {}
        session_open_et = str(params.get("session_open_et") or "09:30")
        try:
            open_dt = ensure_et(dt.datetime.combine(session_dt, parse_time_hhmm(session_open_et)))
            cutoff_dt = open_dt + dt.timedelta(minutes=max(1, int(minutes)))
            cutoff_time_et = cutoff_dt.strftime("%H:%M")
        except Exception:
            # Fallback: if timestamps are weird, just return full cached slice.
            return list(rows)
        return filter_intraday_bars_until(rows, session_dt, cutoff_time_et)

    def _find_alt_cache_file() -> Optional[Path]:
        # If an exact (symbol, date, minutes) cache doesn't exist, try reusing a cached slice with
        # >= minutes and then slicing down. This speeds up parameter sweeps where minutes_needed changes.
        if refresh:
            return None
        sym_dir = (cache_dir / str(symbol).upper().strip())
        if not sym_dir.exists():
            return None
        date_prefix = f"{session_dt.isoformat()}_"
        candidates: List[Tuple[int, Path]] = []
        try:
            for p in sym_dir.glob(f"{session_dt.isoformat()}_*m.json"):
                name = p.name
                if not name.startswith(date_prefix):
                    continue
                # Expected: YYYY-MM-DD_<minutes>m.json
                try:
                    rest = name[len(date_prefix) :]
                    if not rest.endswith("m.json"):
                        continue
                    mins_str = rest[: -len("m.json")]
                    m = int(mins_str)
                except Exception:
                    continue
                if m >= minutes:
                    candidates.append((m, p))
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    if cache_file.exists() and not refresh:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                if not _has_expected_coverage(cached):
                    # Stale cache is not an error; we will refetch below. Keep this at DEBUG to
                    # avoid spamming INFO logs during large backtests.
                    logging.debug(
                        "[INTRADAY] cache stale (coverage check failed); refetching sym=%s date=%s minutes=%s",
                        str(symbol).upper(),
                        session_dt.isoformat(),
                        minutes,
                    )
                else:
                    _MEM_CACHE[cache_key] = cached
                    return cached
        except Exception:
            pass
    # If exact cache is missing or stale, try reusing a larger cached slice and slice it down.
    if not refresh:
        alt = _find_alt_cache_file()
        if alt is not None and alt.exists():
            try:
                cached = json.loads(alt.read_text(encoding="utf-8"))
                sliced: List[Dict[str, Any]] = []
                if isinstance(cached, list) and cached:
                    sliced = _slice_to_minutes(cached)
                    if not _has_expected_coverage(sliced):
                        sliced = []
                if isinstance(cached, list) and sliced:
                    _MEM_CACHE[cache_key] = sliced
                    # Persist the sliced result so subsequent runs can hit the exact cache file directly.
                    try:
                        if sliced and not cache_file.exists():
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            cache_file.write_text(json.dumps(sliced), encoding="utf-8")
                    except Exception:
                        pass
                    return sliced
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


def filter_intraday_bars_until(
    bars: List[Dict[str, Any]],
    session_date: str | dt.date,
    entry_time_et: str,
) -> List[Dict[str, Any]]:
    if not bars:
        return []
    entry_dt = ensure_et(dt.datetime.combine(ensure_date(session_date), parse_time_hhmm(entry_time_et)))
    out: List[Dict[str, Any]] = []
    for row in bars:
        ts = _parse_timestamp(row.get("timestamp") or row.get("t") or row.get("time") or row.get("date"))
        if not ts:
            continue
        ts = ensure_et(ts)
        # Strict "known up to entry_time" parity with live:
        # at 09:35 you do NOT yet have the 09:35 bar via minute-bars APIs; you only have completed bars strictly
        # before the entry timestamp.
        if ts < entry_dt:
            out.append(row)
        else:
            break
    return out


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
