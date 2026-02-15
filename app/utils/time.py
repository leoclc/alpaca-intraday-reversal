from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

try:
    import pytz
except Exception:  # pragma: no cover
    pytz = None  # type: ignore


def _et_tz():
    if pytz:
        return pytz.timezone("America/New_York")
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return ZoneInfo("America/New_York")
    except Exception:
        return dt.timezone(dt.timedelta(hours=-5))


ET_TZ = _et_tz()


def et_now() -> dt.datetime:
    return dt.datetime.now(tz=ET_TZ)


def et_today_date_str() -> str:
    return et_now().date().isoformat()


def ensure_date(val: str | dt.date | dt.datetime) -> dt.date:
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    return dt.date.fromisoformat(str(val))


def ensure_et(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        # pytz requires localize to set the correct offset/DST
        if hasattr(ET_TZ, "localize"):
            try:
                return ET_TZ.localize(ts)  # type: ignore[attr-defined]
            except Exception:
                return ts.replace(tzinfo=ET_TZ)
        return ts.replace(tzinfo=ET_TZ)
    try:
        return ts.astimezone(ET_TZ)
    except Exception:
        return ts


def parse_time_hhmm(val: str) -> dt.time:
    parts = str(val).strip().split(":")
    if len(parts) < 2:
        return dt.time(9, 30)
    return dt.time(int(parts[0]), int(parts[1]))


def iter_trading_days(start_date: str | dt.date, end_date: str | dt.date) -> Iterable[dt.date]:
    start = ensure_date(start_date)
    end = ensure_date(end_date)
    cur = start
    one_day = dt.timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:
            yield cur
        cur += one_day
