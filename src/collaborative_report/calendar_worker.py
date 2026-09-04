"""Isolated AkShare trading-calendar loader for collaborative reports."""

import json
import re
import sys
from contextlib import redirect_stdout
from datetime import date, datetime


_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _session_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _ISO_DATE_PATTERN.fullmatch(value):
        raise ValueError("AkShare session date must be YYYY-MM-DD")
    return date.fromisoformat(value)


def _load_sessions() -> list[str]:
    with redirect_stdout(sys.stderr):
        import akshare

        frame = akshare.tool_trade_date_hist_sina()
    if frame is None or "trade_date" not in frame.columns:
        raise ValueError("AkShare trade_date column is missing")

    sessions = sorted({_session_date(value) for value in frame["trade_date"]})
    if not sessions:
        raise ValueError("AkShare returned no trading dates")
    return [session.isoformat() for session in sessions]


def main() -> int:
    try:
        print(json.dumps(_load_sessions()))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
