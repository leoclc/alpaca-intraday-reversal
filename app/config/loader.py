from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

from app.config.defaults import CONFIG_FILE, DEFAULT_CONFIG, RUNTIME_OVERRIDES_FILE

_RUNTIME_WRITE_LOCK = threading.Lock()


def _read_json_file(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_config() -> Dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    user_cfg = _read_json_file(Path(CONFIG_FILE))
    if user_cfg:
        merged.update(user_cfg)
    runtime = _read_json_file(Path(RUNTIME_OVERRIDES_FILE))
    if runtime:
        merged.update(runtime)
    try:
        raw_env = os.environ.get("APP_CFG_OVERRIDES")
        if raw_env:
            env_cfg = json.loads(raw_env)
            if isinstance(env_cfg, dict):
                merged.update(env_cfg)
    except Exception:
        pass
    return merged


def save_runtime_overrides(config: Dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        return False
    runtime_keys = {
        "last_daily_scan_at",
        "active_watchlist_date",
    }
    payload = {k: config.get(k) for k in runtime_keys if k in config}
    runtime_path = Path(RUNTIME_OVERRIDES_FILE)
    with _RUNTIME_WRITE_LOCK:
        try:
            existing = _read_json_file(runtime_path)
        except Exception:
            existing = {}
        merged = dict(existing or {})
        merged.update(payload)
        try:
            tmp = Path(f"{RUNTIME_OVERRIDES_FILE}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            os.replace(str(tmp), str(runtime_path))
            return True
        except Exception:
            try:
                if "tmp" in locals() and tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False
