from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


def resolve_watchlist_asset_filters(cfg: dict) -> Optional[Dict[str, Any]]:
    raw = cfg.get("watchlist_asset_filters") or cfg.get("node_asset_filters") or cfg.get("asset_filters")
    if isinstance(raw, dict) and raw:
        cleaned = {str(k): raw[k] for k in raw if str(k) not in {"from", "to"}}
        return cleaned if cleaned else None
    return None


def resolve_watchlist_builder_base(cfg: dict) -> str:
    candidates = [
        cfg.get("watchlist_node_base"),
        cfg.get("node_api_base"),
        cfg.get("marketscan_api_base"),
        os.getenv("WATCHLIST_NODE_BASE"),
        os.getenv("MARKETSCAN_API_BASE"),
        os.getenv("NODE_API_BASE"),
    ]
    for candidate in candidates:
        try:
            value = str(candidate or "").strip()
        except Exception:
            continue
        if value:
            return value.rstrip("/")
    return "http://localhost:3000"


def _http_get(url: str, *, params: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
    if timeout is None:
        env_t = os.getenv("AUTOWATCHLIST_HTTP_TIMEOUT")
        if env_t:
            try:
                timeout = float(env_t)
            except Exception:
                timeout = None
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json() or {}
    except Exception:
        return {}


def fetch_asset_symbols(
    *,
    base_url: str,
    status: str = "active",
    shortable: bool = True,
    easy_to_borrow: bool = True,
    price_min: Optional[float] = None,
    min_adv30_shares: Optional[float] = None,
    max_symbols: Optional[int] = None,
    **extra_params: Any,
) -> List[str]:
    def _coerce_bool(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return str(value).lower()
        return str(bool(value)).lower()

    if "status" in extra_params:
        status = str(extra_params.pop("status") or status)
    if "shortable" in extra_params:
        shortable = bool(extra_params.pop("shortable"))
    if "easy_to_borrow" in extra_params:
        easy_to_borrow = bool(extra_params.pop("easy_to_borrow"))
    if "price_min" in extra_params:
        price_min = extra_params.pop("price_min") or price_min
    if "min_adv30_shares" in extra_params:
        min_adv30_shares = extra_params.pop("min_adv30_shares") or min_adv30_shares
    if "max_symbols" in extra_params:
        max_symbols = extra_params.pop("max_symbols")

    if price_min is None:
        price_min = 15
    if min_adv30_shares is None:
        min_adv30_shares = 200000

    base_params: Dict[str, str] = {"status": status}
    base_params["shortable"] = _coerce_bool(shortable) or "true"
    base_params["easy_to_borrow"] = _coerce_bool(easy_to_borrow) or "true"
    for key, value in extra_params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            base_params[key] = _coerce_bool(value) or ""
        else:
            base_params[key] = str(value)
    base_params["price_min"] = str(price_min)
    base_params["min_adv30_shares"] = str(min_adv30_shares)

    payload = _http_get(f"{base_url}/api/assets", params=base_params)
    raw_results = []
    if isinstance(payload, dict):
        raw_results.extend(payload.get("results") or [])
    elif isinstance(payload, list):
        raw_results.extend(payload)
    if not raw_results:
        raise RuntimeError("Asset scan returned no symbols")

    limit: Optional[int] = None
    if max_symbols is not None:
        try:
            limit_candidate = int(max_symbols)
        except Exception:
            limit = None
        else:
            if limit_candidate > 0:
                limit = limit_candidate

    symbols: List[str] = []
    for asset in raw_results:
        try:
            sym = str((asset or {}).get("symbol") or "").upper()
        except Exception:
            sym = ""
        if not sym:
            continue
        symbols.append(sym)
        if limit is not None and len(symbols) >= limit:
            break
    symbols = sorted(dict.fromkeys(symbols))
    if not symbols:
        raise RuntimeError("Asset scan returned no symbols")
    return symbols
