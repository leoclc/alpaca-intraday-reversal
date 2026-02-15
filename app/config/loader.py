from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

from app.config.defaults import CONFIG_FILE, DEFAULT_CONFIG, RUNTIME_OVERRIDES_FILE

_RUNTIME_WRITE_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)
_LOGGED_PATHS = False
_LOCAL_CONFIG_FILE = "config.local.json"


def _repo_root() -> Path:
    # `app/config/loader.py` -> repo root is two parents up (app/ -> repo root)
    return Path(__file__).resolve().parents[2]


def _resolve_config_path() -> Path:
    # Allow explicit override, but default to the repo-local config (next to `app/`),
    # not whatever directory the user happened to run from.
    env_path = os.environ.get("APP_CONFIG_PATH") or os.environ.get("ALPACA_OHLC_CONFIG")
    if env_path:
        return Path(env_path)

    repo_default = _repo_root() / CONFIG_FILE
    if repo_default.exists():
        return repo_default

    # Fallback for unusual packaging/layouts.
    return Path(CONFIG_FILE)


def _resolve_runtime_overrides_path(config_path: Path) -> Path:
    env_path = os.environ.get("APP_RUNTIME_OVERRIDES_PATH")
    if env_path:
        return Path(env_path)

    # Keep runtime overrides next to the config for deterministic behavior.
    return config_path.parent / RUNTIME_OVERRIDES_FILE


def _resolve_local_override_paths(config_path: Path) -> List[Path]:
    # Support a repo-local `config.local.json` (ignored by git) for credentials and other
    # machine-specific overrides. Also support a `config.local.json` next to whichever
    # config file was selected via APP_CONFIG_PATH.
    repo_local = _repo_root() / _LOCAL_CONFIG_FILE
    adjacent = config_path.parent / _LOCAL_CONFIG_FILE
    # Dedupe (config_path may already be repo root config.json).
    seen = set()
    out: List[Path] = []
    for p in (repo_local, adjacent):
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


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
    global _LOGGED_PATHS

    cfg_path = _resolve_config_path()
    runtime_path = _resolve_runtime_overrides_path(cfg_path)
    local_paths = _resolve_local_override_paths(cfg_path)

    if not _LOGGED_PATHS:
        try:
            _LOG.info(
                "[CONFIG] using config_path=%s runtime_overrides_path=%s local_overrides=%s",
                cfg_path.resolve(),
                runtime_path.resolve(),
                [str(p.resolve()) for p in local_paths if p.exists()],
            )
        except Exception:
            _LOG.info(
                "[CONFIG] using config_path=%s runtime_overrides_path=%s local_overrides=%s",
                cfg_path,
                runtime_path,
                [str(p) for p in local_paths if p.exists()],
            )
        _LOGGED_PATHS = True

    merged = dict(DEFAULT_CONFIG)
    user_cfg = _read_json_file(cfg_path)
    if user_cfg:
        merged.update(user_cfg)
    # Apply repo-local / machine-local overrides after the main config.
    for p in local_paths:
        local_cfg = _read_json_file(p)
        if local_cfg:
            merged.update(local_cfg)
    runtime = _read_json_file(runtime_path)
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
    runtime_path = _resolve_runtime_overrides_path(_resolve_config_path())
    with _RUNTIME_WRITE_LOCK:
        try:
            existing = _read_json_file(runtime_path)
        except Exception:
            existing = {}
        merged = dict(existing or {})
        merged.update(payload)
        try:
            tmp = runtime_path.parent / f"{runtime_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
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
