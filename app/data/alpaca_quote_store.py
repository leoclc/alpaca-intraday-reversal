from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.utils.time import ensure_et

_QUOTE_WINDOW_CACHE: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}


def _parse_timestamp(val: Any) -> Optional[dt.datetime]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        ts = float(val)
        # Alpaca sometimes returns ns timestamps.
        if ts > 1e12:
            ts = ts / 1e9
        try:
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
        except Exception:
            return None
    s = str(val).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Guard against >6 fractional second digits for fromisoformat.
    if "." in s and "+" in s:
        try:
            lhs, tz = s.split("+", 1)
            if "." in lhs:
                base, frac = lhs.split(".", 1)
                frac = frac[:6]
                s = f"{base}.{frac}+{tz}"
        except Exception:
            pass
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def _fetch_quotes_window(
    symbol: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    alp = cfg.get("alpaca") or {}
    key_id = str(alp.get("api_key_id") or "").strip()
    secret = str(alp.get("api_secret_key") or "").strip()
    if not key_id or not secret:
        return []
    base_url = str(alp.get("data_url") or "https://data.alpaca.markets").rstrip("/")
    feed = str(alp.get("data_feed") or "iex")
    timeout = int(alp.get("timeout_sec") or 10)
    max_retries = int(alp.get("max_retries") or 0)
    retry_backoff = float(alp.get("retry_backoff_sec") or 0.0)

    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
    }
    url = f"{base_url}/v2/stocks/quotes"
    params: Dict[str, Any] = {
        "symbols": str(symbol).upper(),
        "start": start_utc.astimezone(dt.timezone.utc).isoformat(),
        "end": end_utc.astimezone(dt.timezone.utc).isoformat(),
        "feed": feed,
        "sort": "asc",
        "limit": 10000,
    }
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
        quotes_payload = payload.get("quotes")
        rows: List[Dict[str, Any]] = []
        if isinstance(quotes_payload, dict):
            rows = quotes_payload.get(str(symbol).upper()) or quotes_payload.get(str(symbol)) or []
        elif isinstance(quotes_payload, list):
            rows = quotes_payload
        for row in rows or []:
            ts = _parse_timestamp(row.get("t") or row.get("timestamp"))
            if not ts:
                continue
            bid = row.get("bp")
            ask = row.get("ap")
            bid = None if bid is None else float(bid)
            ask = None if ask is None else float(ask)
            out.append(
                {
                    "timestamp": ts.astimezone(dt.timezone.utc).isoformat(),
                    "bid": bid,
                    "ask": ask,
                    "bid_size": None if row.get("bs") is None else float(row.get("bs")),
                    "ask_size": None if row.get("as") is None else float(row.get("as")),
                }
            )
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    out.sort(key=lambda r: str(r.get("timestamp") or ""))
    return out


def get_quotes_window(
    symbol: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    cfg: Dict[str, Any],
    allow_fetch: bool = True,
) -> List[Dict[str, Any]]:
    sym = str(symbol or "").upper().strip()
    if not sym:
        return []
    start = start_utc.astimezone(dt.timezone.utc)
    end = end_utc.astimezone(dt.timezone.utc)
    if end <= start:
        return []
    alp = cfg.get("alpaca") or {}
    feed = str(alp.get("data_feed") or "iex").lower()
    key = (sym, start.isoformat(), end.isoformat(), feed)
    cached = _QUOTE_WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    if not allow_fetch:
        return []
    try:
        rows = _fetch_quotes_window(sym, start, end, cfg)
    except Exception as exc:
        logging.warning(
            "[QUOTES] fetch failed symbol=%s start=%s end=%s error=%s",
            sym,
            start.isoformat(),
            end.isoformat(),
            exc,
        )
        rows = []
    _QUOTE_WINDOW_CACHE[key] = rows
    return rows


def resolve_quote_for_timestamp(
    symbol: str,
    target_ts: dt.datetime,
    cfg: Dict[str, Any],
    *,
    lookback_seconds: int = 120,
    forward_seconds: int = 10,
    allow_fetch: bool = True,
) -> Optional[Dict[str, Any]]:
    tgt = ensure_et(target_ts).astimezone(dt.timezone.utc)
    lb = max(1, int(lookback_seconds))
    fw = max(0, int(forward_seconds))
    start = tgt - dt.timedelta(seconds=lb)
    end = tgt + dt.timedelta(seconds=fw)
    rows = get_quotes_window(symbol, start, end, cfg, allow_fetch=allow_fetch)
    if not rows:
        return None

    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    for row in rows:
        ts = _parse_timestamp(row.get("timestamp"))
        if not ts:
            continue
        if ts >= tgt:
            after = dict(row)
            after["selected"] = "after_or_equal"
            break
        before = dict(row)
        before["selected"] = "before"
    if after is not None:
        return after
    if before is not None:
        return before
    return None
