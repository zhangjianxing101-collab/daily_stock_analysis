from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from src.collaborative_report.models import ReportMode
from src.collaborative_report.session import ReportSession, build_report_session


SHANGHAI = ZoneInfo("Asia/Shanghai")
TRADING_DATE = date(2026, 8, 19)


def build_with_calendar(
    mode: ReportMode,
    current_time: datetime,
    *,
    is_trading_day: bool = True,
    scheduled: bool = True,
) -> tuple[ReportSession, Mock]:
    calendar = Mock()
    calendar.is_session.return_value = is_trading_day
    with patch("src.collaborative_report.session.exchange_calendars.get_calendar", return_value=calendar):
        session = build_report_session(mode, current_time, scheduled=scheduled)
    return session, calendar


def test_premarket_trading_session_builds_report_identity() -> None:
    session, calendar = build_with_calendar(
        ReportMode.PREMARKET,
        datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI),
    )

    assert session == ReportSession(
        mode=ReportMode.PREMARKET,
        now_shanghai=datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI),
        trading_date=TRADING_DATE,
        is_trading_day=True,
        report_key="2026-08-19-premarket",
    )
    calendar.is_session.assert_called_once_with(TRADING_DATE)


def test_non_trading_day_returns_session_without_window_error() -> None:
    session, _ = build_with_calendar(
        ReportMode.POSTMARKET,
        datetime(2026, 8, 22, 16, 30, tzinfo=SHANGHAI),
        is_trading_day=False,
    )

    assert session.is_trading_day is False
    assert session.report_key == "2026-08-22-postmarket"


def test_non_trading_day_outside_window_still_returns_session() -> None:
    session, calendar = build_with_calendar(
        ReportMode.PREMARKET,
        datetime(2026, 8, 22, 12, 0, tzinfo=SHANGHAI),
        is_trading_day=False,
    )

    assert session.is_trading_day is False
    calendar.is_session.assert_called_once_with(date(2026, 8, 22))


def test_scheduled_premarket_outside_window_raises() -> None:
    with pytest.raises(RuntimeError, match="^outside delivery window$"):
        build_with_calendar(
            ReportMode.PREMARKET,
            datetime(2026, 8, 19, 9, 31, tzinfo=SHANGHAI),
        )


@pytest.mark.parametrize(
    ("mode", "hour", "minute"),
    [
        (ReportMode.PREMARKET, 8, 30),
        (ReportMode.PREMARKET, 9, 25),
        (ReportMode.POSTMARKET, 16, 0),
        (ReportMode.POSTMARKET, 18, 30),
    ],
)
def test_scheduled_windows_include_exact_boundaries(
    mode: ReportMode,
    hour: int,
    minute: int,
) -> None:
    session, _ = build_with_calendar(
        mode,
        datetime(2026, 8, 19, hour, minute, tzinfo=SHANGHAI),
    )

    assert session.is_trading_day is True


@pytest.mark.parametrize(
    ("mode", "current_time"),
    [
        (ReportMode.PREMARKET, datetime(2026, 8, 19, 8, 29, 59, tzinfo=SHANGHAI)),
        (ReportMode.PREMARKET, datetime(2026, 8, 19, 9, 25, 1, tzinfo=SHANGHAI)),
        (ReportMode.POSTMARKET, datetime(2026, 8, 19, 15, 59, 59, tzinfo=SHANGHAI)),
        (ReportMode.POSTMARKET, datetime(2026, 8, 19, 18, 30, 1, tzinfo=SHANGHAI)),
    ],
)
def test_scheduled_windows_reject_one_second_outside(
    mode: ReportMode,
    current_time: datetime,
) -> None:
    with pytest.raises(RuntimeError, match="^outside delivery window$"):
        build_with_calendar(mode, current_time)


@pytest.mark.parametrize(
    ("mode", "hour", "minute"),
    [
        (ReportMode.PREMARKET, 8, 29),
        (ReportMode.PREMARKET, 9, 26),
        (ReportMode.POSTMARKET, 15, 59),
        (ReportMode.POSTMARKET, 18, 31),
    ],
)
def test_scheduled_windows_reject_one_minute_outside(
    mode: ReportMode,
    hour: int,
    minute: int,
) -> None:
    with pytest.raises(RuntimeError, match="^outside delivery window$"):
        build_with_calendar(
            mode,
            datetime(2026, 8, 19, hour, minute, tzinfo=SHANGHAI),
        )


def test_naive_current_time_is_attached_to_shanghai_timezone() -> None:
    session, _ = build_with_calendar(
        ReportMode.PREMARKET,
        datetime(2026, 8, 19, 9, 0),
    )

    assert session.now_shanghai == datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI)
    assert session.now_shanghai.utcoffset().total_seconds() == 8 * 60 * 60


def test_aware_current_time_is_converted_to_shanghai_timezone() -> None:
    session, _ = build_with_calendar(
        ReportMode.PREMARKET,
        datetime(2026, 8, 19, 1, 0, tzinfo=timezone.utc),
    )

    assert session.now_shanghai == datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI)
    assert session.now_shanghai.tzinfo == SHANGHAI


def test_none_current_time_uses_current_shanghai_time() -> None:
    expected_now = datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI)
    calendar = Mock()
    calendar.is_session.return_value = True
    with (
        patch("src.collaborative_report.session.datetime") as datetime_class,
        patch("src.collaborative_report.session.exchange_calendars.get_calendar", return_value=calendar),
    ):
        datetime_class.now.return_value = expected_now
        session = build_report_session(ReportMode.PREMARKET)

    datetime_class.now.assert_called_once_with(SHANGHAI)
    assert session.now_shanghai == expected_now


def test_unscheduled_run_bypasses_window_but_still_checks_calendar() -> None:
    session, calendar = build_with_calendar(
        ReportMode.PREMARKET,
        datetime(2026, 8, 19, 12, 0, tzinfo=SHANGHAI),
        scheduled=False,
    )

    assert session.is_trading_day is True
    calendar.is_session.assert_called_once_with(TRADING_DATE)


def test_calendar_lookup_failure_is_fail_closed() -> None:
    with (
        patch(
            "src.collaborative_report.session.exchange_calendars.get_calendar",
            side_effect=LookupError("calendar missing"),
        ),
        pytest.raises(RuntimeError, match="^trading calendar unavailable$") as error,
    ):
        build_report_session(
            ReportMode.PREMARKET,
            datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI),
        )

    assert isinstance(error.value.__cause__, LookupError)


def test_calendar_session_failure_is_fail_closed() -> None:
    calendar = Mock()
    calendar.is_session.side_effect = ValueError("unsupported date")
    with (
        patch("src.collaborative_report.session.exchange_calendars.get_calendar", return_value=calendar),
        pytest.raises(RuntimeError, match="^trading calendar unavailable$") as error,
    ):
        build_report_session(
            ReportMode.POSTMARKET,
            datetime(2026, 8, 19, 16, 0, tzinfo=SHANGHAI),
        )

    assert isinstance(error.value.__cause__, ValueError)


def test_report_session_is_frozen() -> None:
    session, _ = build_with_calendar(
        ReportMode.PREMARKET,
        datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI),
    )

    with pytest.raises(FrozenInstanceError):
        session.report_key = "changed"  # type: ignore[misc]


def test_invalid_report_mode_is_rejected_before_calendar_lookup() -> None:
    with (
        patch("src.collaborative_report.session.exchange_calendars.get_calendar") as get_calendar,
        pytest.raises(ValueError, match="^invalid report mode$"),
    ):
        build_report_session(  # type: ignore[arg-type]
            "premarket",
            datetime(2026, 8, 19, 9, 0, tzinfo=SHANGHAI),
        )

    get_calendar.assert_not_called()
