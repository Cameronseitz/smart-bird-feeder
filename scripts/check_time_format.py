"""Offline check for the dashboard caption timestamp format (stdlib only)."""
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dashboard import timefmt  # noqa: E402

TWENTY_FOUR_HOUR = re.compile(r"(?:^|[^0-9])(?:1[3-9]|2[0-3]):\d\d")
CLOCK_WITH_MERIDIEM = re.compile(r"^\d{1,2}:\d\d [AP]M$")

CASES = [
    ("2026-10-06T16:51:23.123456", "4:51 PM 10/06"),
    ("2026-10-06 16:51:23", "4:51 PM 10/06"),
    ("2026-10-06T00:05:00", "12:05 AM 10/06"),
    ("2026-10-06T12:00:00", "12:00 PM 10/06"),
    ("2026-10-06T23:59:00", "11:59 PM 10/06"),
    ("2026-01-05T09:07:00", "9:07 AM 01/05"),
]

failures = 0


def check(label, got, want):
    global failures
    ok = got == want
    if not ok:
        failures += 1
    print(f"{'PASS' if ok else 'FAIL'} {label}: {got!r} == {want!r}")


for raw, want in CASES:
    check(f"short_datetime({raw})", timefmt.short_datetime(raw), want)

# short_time keeps its old h:MM AM/PM output, no date
check("short_time(ISO)", timefmt.short_time("2026-10-06T16:51:23.123456"), "4:51 PM")
check("short_time(legacy)", timefmt.short_time("2026-10-06 16:51:23"), "4:51 PM")

# Empty in, empty out; garbage is returned verbatim exactly once
check("short_datetime(None)", timefmt.short_datetime(None), "")
check("short_datetime('')", timefmt.short_datetime(""), "")
check("short_time(None)", timefmt.short_time(None), "")
check("short_time('')", timefmt.short_time(""), "")
check("short_datetime('not-a-date')", timefmt.short_datetime("not-a-date"), "not-a-date")
check("short_time('not-a-date')", timefmt.short_time("not-a-date"), "not-a-date")

# Invariants: never 24-hour, and a time portion always carries AM/PM
for raw, _ in CASES:
    caption = timefmt.short_datetime(raw)
    clock, _, date = caption.rpartition(" ")
    if TWENTY_FOUR_HOUR.search(caption) or not CLOCK_WITH_MERIDIEM.match(clock) or not re.match(
        r"^\d\d/\d\d$", date
    ):
        failures += 1
        print(f"FAIL 12-hour invariant for {raw}: {caption!r}")
    else:
        print(f"PASS 12-hour invariant for {raw}: {caption!r}")

if failures:
    print(f"\n{failures} failure(s)")
    sys.exit(1)

print("\nAll time format checks passed")
