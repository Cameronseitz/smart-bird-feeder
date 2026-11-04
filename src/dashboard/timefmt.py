"""Display formatting for visit timestamps (stdlib only, no Flask import).

Stored timestamps are naive local time in two shapes: the pipeline writes
"2026-10-05T16:51:23.123456" while SQLite's CURRENT_TIMESTAMP default gives
"2026-10-05 16:51:23". Both parse to the same string here.

Captions are always 12-hour (never 24-hour): `short_time` gives "4:51 PM" and
`short_datetime` appends a zero-padded MM/DD date, e.g. "4:51 PM 10/06".
"""
from datetime import datetime


def _parse(value):
    """Full ISO/SQLite timestamp -> naive datetime, or None if empty/unparseable."""
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return datetime.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _clock(dt):
    """Naive datetime -> '4:51 PM' (hour + minute, always 12-hour)."""
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def short_time(value):
    """Full ISO/SQLite timestamp -> '4:51 PM' (hour + minute only)."""
    if value is None:
        return ""
    if not str(value).strip():
        return ""
    dt = _parse(value)
    if dt is None:
        return str(value)
    return _clock(dt)


def short_datetime(value):
    """Full ISO/SQLite timestamp -> '4:51 PM 10/06' (time then MM/DD date)."""
    if value is None:
        return ""
    if not str(value).strip():
        return ""
    dt = _parse(value)
    if dt is None:
        return str(value)
    return f"{_clock(dt)} {dt.strftime('%m/%d')}"
