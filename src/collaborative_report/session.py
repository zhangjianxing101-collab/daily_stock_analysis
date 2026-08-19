"""Trading-session identity and delivery-window gating for reports."""

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import exchange_calendars

from .models import ReportMode


SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
_DELIVERY_WINDOWS = {
    ReportMode.PREMARKET: (time(8, 30), time(9, 25)),
    ReportMode.POSTMARKET: (time(16, 0), time(18, 30)),
}


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

    return _latest_completed_session(_xshg_calendar(), current_time)


def report_data_session(
    mode: ReportMode,
    report_date: date,
    generated_at: datetime | None = None,
) -> date:
    """Resolve the completed XSHG session whose data a report may expose."""

    if not isinstance(mode, ReportMode):
        raise ValueError("invalid report mode")
    calendar = _xshg_calendar()
    if generated_at is not None:
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if generated_at.astimezone(SHANGHAI_TIMEZONE).date() != report_date:
            raise ValueError("generated_at must match report_date in Asia/Shanghai")
        return _latest_completed_session(calendar, generated_at)
    try:
        session = calendar.date_to_session(report_date, direction="previous")
        if mode is ReportMode.PREMARKET and session.date() == report_date:
            session = calendar.previous_session(session)
        return session.date()
    except Exception as exc:
        raise RuntimeError("trading calendar unavailable") from exc


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
        calendar = exchange_calendars.get_calendar("XSHG")
        is_trading_day = bool(calendar.is_session(trading_date))
    except Exception as exc:
        raise RuntimeError("trading calendar unavailable") from exc

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
