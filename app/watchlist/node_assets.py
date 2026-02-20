from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.utils.time import et_today_date_str


def _normalize_symbols(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for val in raw:
        try:
            sym = str(val or "").upper().strip()
        except Exception:
            sym = ""
        if sym:
            out.append(sym)
    return sorted(dict.fromkeys(out))


def _asset_universe_dir(cfg: dict) -> Path:
    base = cfg.get("asset_universe_dir")
    if not base:
        logs_dir = str(cfg.get("logs_dir") or "logs")
        base = str(Path(logs_dir) / "asset_universe")
    path = Path(str(base))
    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_universe_snapshot_path(cfg: dict, target_date: Optional[str] = None) -> Path:
    date_str = str(target_date or et_today_date_str())
    return _asset_universe_dir(cfg) / f"{date_str}.json"


def read_asset_universe_snapshot(cfg: dict, target_date: Optional[str] = None) -> Tuple[List[str], Dict[str, Any]]:
    path = asset_universe_snapshot_path(cfg, target_date)
    if not path.exists() or path.stat().st_size <= 0:
        return [], {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}
    if not isinstance(payload, dict):
        return [], {}
    symbols = _normalize_symbols(payload.get("symbols"))
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return symbols, meta


def write_asset_universe_snapshot(
    cfg: dict,
    target_date: Optional[str],
    symbols: List[str],
    *,
    source: str,
    base_url: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Path:
    date_str = str(target_date or et_today_date_str())
    path = asset_universe_snapshot_path(cfg, date_str)
    payload = {
        "date": date_str,
        "symbols": _normalize_symbols(symbols),
        "meta": {
            "source": str(source or ""),
            "base_url": str(base_url or ""),
            "filters": dict(filters or {}),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def resolve_asset_universe_symbols(
    cfg: dict,
    *,
    target_date: Optional[str] = None,
    allow_fetch: bool = True,
    force_refresh: bool = False,
) -> Tuple[List[str], str]:
    date_str = str(target_date or et_today_date_str())
    watchlist_source = str(cfg.get("watchlist_source") or "node").lower()

    if not force_refresh:
        cached, _meta = read_asset_universe_snapshot(cfg, date_str)
        if cached:
            return cached, "snapshot"

    if watchlist_source != "node":
        symbols = _normalize_symbols(cfg.get("symbols") or cfg.get("watchlist_symbols") or [])
        if symbols:
            write_asset_universe_snapshot(cfg, date_str, symbols, source="config")
        return symbols, "config"

    if not allow_fetch:
        return [], "none"

    asset_filters = resolve_watchlist_asset_filters(cfg) or {}
    base_url = resolve_watchlist_builder_base(cfg)
    symbols = fetch_asset_symbols(base_url=base_url, **asset_filters)
    write_asset_universe_snapshot(
        cfg,
        date_str,
        symbols,
        source="node",
        base_url=base_url,
        filters=asset_filters,
    )
    return symbols, "node"
