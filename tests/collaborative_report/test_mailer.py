import pytest
from tenacity import wait_none

from src.collaborative_report.mailer import DEFAULT_RETRY_WAIT, DeliveryError, send_with_retry


class FakeEmailSender:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def send_html_email(self, html_content, text_content, subject, receivers=None, timeout_seconds=None):
        self.calls.append((html_content, text_content, subject, receivers, timeout_seconds))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_default_retry_waits_are_two_then_four_seconds() -> None:
    first_retry = type("RetryState", (), {"attempt_number": 1})()
    second_retry = type("RetryState", (), {"attempt_number": 2})()

    assert DEFAULT_RETRY_WAIT(first_retry) == 2
    assert DEFAULT_RETRY_WAIT(second_retry) == 4


def test_mailer_retries_twice_then_returns_structured_success() -> None:
    sender = FakeEmailSender([False, OSError("network"), True])

    result = send_with_retry(
        sender,
        html_content="<p>报告</p>",
        text_content="报告",
        subject="A股盘前日报 2026-08-19",
        receivers=["report@example.com"],
        timeout_seconds=8,
        wait_strategy=wait_none(),
    )

    assert result.sent is True
    assert result.attempts == 3
    assert result.subject == "A股盘前日报 2026-08-19"
    assert len(sender.calls) == 3
    assert sender.calls[-1][3:] == (["report@example.com"], 8)


def test_mailer_raises_final_failure_after_three_attempts() -> None:
    sender = FakeEmailSender([False, False, False])

    with pytest.raises(DeliveryError, match="after 3 attempts"):
        send_with_retry(
            sender,
            html_content="<p>报告</p>",
            text_content="报告",
            subject="A股收盘日报 2026-08-19",
            wait_strategy=wait_none(),
        )

    assert len(sender.calls) == 3


def test_mailer_does_not_retry_non_network_programming_errors() -> None:
    sender = FakeEmailSender([ValueError("invalid message")])

    with pytest.raises(ValueError, match="invalid message"):
        send_with_retry(
            sender,
            html_content="<p>报告</p>",
            text_content="报告",
            subject="A股盘前日报 2026-08-19",
            wait_strategy=wait_none(),
        )

    assert len(sender.calls) == 1
