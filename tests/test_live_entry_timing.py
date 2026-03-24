from __future__ import annotations

import datetime as dt

from app.__main__ import _entry_release_state
from app.utils.time import ensure_et


def _et(hour: int, minute: int, second: int = 0) -> dt.datetime:
    return ensure_et(dt.datetime(2026, 3, 23, hour, minute, second))


def test_entry_release_uses_broker_clock_when_available() -> None:
    ready, remaining, mode = _entry_release_state(
        _et(9, 35, 0),
        local_now=_et(9, 35, 5),
        broker_now=_et(9, 34, 55),
        fallback_delay_sec=5.0,
    )
    assert not ready
    assert mode == "broker_clock"
    assert remaining == 5.0


def test_entry_release_ready_when_broker_clock_reaches_target() -> None:
    ready, remaining, mode = _entry_release_state(
        _et(9, 35, 0),
        local_now=_et(9, 35, 5),
        broker_now=_et(9, 35, 0),
        fallback_delay_sec=5.0,
    )
    assert ready
    assert mode == "broker_clock"
    assert remaining == 0.0


def test_entry_release_falls_back_to_delayed_local_clock_when_broker_unavailable() -> None:
    ready, remaining, mode = _entry_release_state(
        _et(9, 35, 0),
        local_now=_et(9, 35, 2),
        broker_now=None,
        fallback_delay_sec=5.0,
    )
    assert not ready
    assert mode == "local_fallback"
    assert remaining == 3.0
