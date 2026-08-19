"""Trading-session identity and delivery-window gating for reports."""

from dataclasses import dataclass
from datetime import date, datetime, time
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
