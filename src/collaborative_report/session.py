"""Trading-session identity and delivery-window gating for reports."""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars

from .models import ReportMode


SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
_DELIVERY_WINDOWS = {
    ReportMode.PREMARKET: (time(8, 30), time(9, 25)),
    ReportMode.POSTMARKET: (time(16, 0), time(18, 30)),
}
_XSHG_CLOSE = time(15, 0)
_AKSHARE_WORKER_PATH = Path(__file__).with_name("calendar_worker.py").resolve()
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ReportSession:
    mode: ReportMode
    now_shanghai: datetime
    trading_date: date
    is_trading_day: bool
    report_key: str


def _shanghai_time(current_time: datetime | None) -> datetime:
    if current_time is None:
        return datetime.now(SHANGHAI_TIMEZONE)
    if current_time.tzinfo is None:
        return current_time.replace(tzinfo=SHANGHAI_TIMEZONE)
    return current_time.astimezone(SHANGHAI_TIMEZONE)


def _xshg_calendar():
    try:
        return exchange_calendars.get_calendar("XSHG")
    except Exception as exc:
        raise RuntimeError("trading calendar unavailable") from exc


def _akshare_session_date(value: object) -> date:
    if not isinstance(value, str) or not _ISO_DATE_PATTERN.fullmatch(value):
        raise ValueError("AkShare session date must be YYYY-MM-DD")
    return date.fromisoformat(value)


def _load_akshare_xshg_sessions() -> frozenset[date]:
    """Load validated explicit China trading dates through a bounded worker."""

    try:
        completed = subprocess.run(
            [sys.executable, str(_AKSHARE_WORKER_PATH)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        values = json.loads(completed.stdout)
        if not isinstance(values, list):
            raise ValueError("AkShare worker output must be a list")
        sessions = frozenset(_akshare_session_date(value) for value in values)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        raise RuntimeError("trading calendar unavailable") from exc
    if not sessions:
        raise RuntimeError("trading calendar unavailable")
    return sessions


@lru_cache(maxsize=2)
def _akshare_xshg_sessions_for_local_date(cache_date: date) -> frozenset[date]:
    """Cache the fallback dataset only for the current and prior Shanghai dates."""

    return _load_akshare_xshg_sessions()


def _akshare_xshg_sessions(current_time: datetime | None = None) -> frozenset[date]:
    """Return fallback sessions, refreshing automatically when Shanghai date changes."""

    return _akshare_xshg_sessions_for_local_date(_shanghai_time(current_time).date())


def _fallback_sessions_covering(target: date) -> frozenset[date]:
    sessions = _akshare_xshg_sessions()
    if not sessions or not min(sessions) <= target <= max(sessions):
        raise RuntimeError("trading calendar unavailable")
    return sessions


def _fallback_session_on_or_before(target: date) -> date:
    sessions = _fallback_sessions_covering(target)
    eligible = [session for session in sessions if session <= target]
    if not eligible:
        raise RuntimeError("trading calendar unavailable")
    return max(eligible)


def _fallback_report_data_session(
    mode: ReportMode,
    report_date: date,
    generated_at: datetime | None,
) -> date:
    coverage_target = report_date if mode is ReportMode.POSTMARKET else report_date - timedelta(days=1)
    sessions = _fallback_sessions_covering(coverage_target)
    if mode is ReportMode.POSTMARKET:
        if report_date not in sessions:
            raise RuntimeError("report date is not an XSHG session")
        session = report_date
    else:
        session = _fallback_session_on_or_before(report_date - timedelta(days=1))
    if generated_at is not None:
        session_close = datetime.combine(session, _XSHG_CLOSE, tzinfo=SHANGHAI_TIMEZONE)
        if session_close > generated_at.astimezone(SHANGHAI_TIMEZONE):
            raise RuntimeError("report data session incomplete")
    return session


def _latest_completed_session(calendar, current_time: datetime) -> date:
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("current_time must be timezone-aware")
    current_utc = current_time.astimezone(timezone.utc)
    try:
        session = calendar.date_to_session(current_utc.date(), direction="previous")
        while calendar.session_close(session).to_pydatetime() > current_utc:
            session = calendar.previous_session(session)
        return session.date()
    except Exception as exc:
        raise RuntimeError("trading calendar unavailable") from exc


def latest_completed_xshg_session(current_time: datetime) -> date:
    """Return the latest XSHG session closed by an actual aware instant."""

    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("current_time must be timezone-aware")
    try:
        return _latest_completed_session(_xshg_calendar(), current_time)
    except RuntimeError:
        now_shanghai = _shanghai_time(current_time)
        cutoff = now_shanghai.date()
        if now_shanghai.time().replace(tzinfo=None) < _XSHG_CLOSE:
            cutoff -= timedelta(days=1)
        return _fallback_session_on_or_before(cutoff)


def report_data_session(
    mode: ReportMode,
    report_date: date,
    generated_at: datetime | None = None,
) -> date:
    """Resolve the mode/date target and optionally prove it complete at generation time."""

    if not isinstance(mode, ReportMode):
        raise ValueError("invalid report mode")
    if generated_at is not None:
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if generated_at.astimezone(SHANGHAI_TIMEZONE).date() != report_date:
            raise ValueError("generated_at must match report_date in Asia/Shanghai")
    try:
        calendar = _xshg_calendar()
        session = calendar.date_to_session(report_date, direction="previous")
        if mode is ReportMode.PREMARKET and session.date() == report_date:
            session = calendar.previous_session(session)
        if mode is ReportMode.POSTMARKET and session.date() != report_date:
            raise RuntimeError("report date is not an XSHG session")
        if generated_at is not None:
            session_close = calendar.session_close(session).to_pydatetime()
            if session_close > generated_at.astimezone(timezone.utc):
                raise RuntimeError("report data session incomplete")
        return session.date()
    except RuntimeError as exc:
        if str(exc) in {"report date is not an XSHG session", "report data session incomplete"}:
            raise
        return _fallback_report_data_session(mode, report_date, generated_at)
    except Exception:
        return _fallback_report_data_session(mode, report_date, generated_at)


def build_report_session(
    mode: ReportMode,
    current_time: datetime | None = None,
    *,
    scheduled: bool = True,
) -> ReportSession:
    if not isinstance(mode, ReportMode):
        raise ValueError("invalid report mode")

    now_shanghai = _shanghai_time(current_time)
    trading_date = now_shanghai.date()
    try:
        is_trading_day = bool(_xshg_calendar().is_session(trading_date))
    except (RuntimeError, ValueError):
        is_trading_day = trading_date in _fallback_sessions_covering(trading_date)

    if scheduled and is_trading_day:
        window_start, window_end = _DELIVERY_WINDOWS[mode]
        local_time = now_shanghai.time().replace(tzinfo=None)
        if not window_start <= local_time <= window_end:
            raise RuntimeError("outside delivery window")

    return ReportSession(
        mode=mode,
        now_shanghai=now_shanghai,
        trading_date=trading_date,
        is_trading_day=is_trading_day,
        report_key=f"{trading_date.isoformat()}-{mode.value}",
    )
