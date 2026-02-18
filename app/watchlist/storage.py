from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from app.utils.time import et_today_date_str


def expected_watchlist_date_str(date_override: Optional[str] = None) -> str:
    return date_override or et_today_date_str()


def _watchlists_dir(cfg: Optional[dict] = None) -> Path:
    base = (cfg or {}).get("watchlists_dir") or "watchlists"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def watchlist_path(date_str: Optional[str] = None, cfg: Optional[dict] = None) -> Path:
    dt_str = expected_watchlist_date_str(date_str)
    return _watchlists_dir(cfg) / f"{dt_str}.json"


def write_watchlist(
    watchlist: list,
    cfg: Optional[dict] = None,
    date_str: Optional[str] = None,
    meta: Optional[dict] = None,
) -> Path:
    path = watchlist_path(date_str, cfg)
    payload = {"date": expected_watchlist_date_str(date_str), "watchlist": watchlist}
    if isinstance(meta, dict):
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_watchlist(date_str: Optional[str] = None, cfg: Optional[dict] = None) -> Dict:
    path = watchlist_path(date_str, cfg)
    if not path.exists() or path.stat().st_size <= 0:
        return {"date": None, "watchlist": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                "date": data.get("date"),
                "watchlist": data.get("watchlist") or [],
                "meta": data.get("meta") if isinstance(data.get("meta"), dict) else {},
            }
    except Exception:
        pass
    return {"date": None, "watchlist": [], "meta": {}}


def watchlist_is_current(date_str: Optional[str] = None, target_date: Optional[str] = None) -> bool:
    target = expected_watchlist_date_str(target_date)
    return bool(date_str == target)
