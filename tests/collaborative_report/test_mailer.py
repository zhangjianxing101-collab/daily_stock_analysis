import pytest
from tenacity import wait_none

from src.collaborative_report.mailer import (
    DEFAULT_RETRY_WAIT,
    DeliveryInDoubtError,
    DeliveryNotAcceptedError,
    send_with_retry,
)
from src.notification_sender.email_sender import (
    EmailAuthenticationFailure,
    EmailDeliveryAmbiguous,
    EmailPermanentPreAcceptanceFailure,
    EmailTransientPreAcceptanceFailure,
)


class FakeEmailSender:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def send_html_email_strict(self, html_content, text_content, subject, receivers=None, timeout_seconds=None):
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


def test_mailer_retries_only_safe_transient_preacceptance_failures() -> None:
    sender = FakeEmailSender([
        EmailTransientPreAcceptanceFailure("connect"),
        EmailTransientPreAcceptanceFailure("starttls"),
        None,
    ])

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


def test_mailer_raises_retryable_not_accepted_after_three_safe_failures() -> None:
    sender = FakeEmailSender([EmailTransientPreAcceptanceFailure("connect")] * 3)

    with pytest.raises(DeliveryNotAcceptedError) as error:
        send_with_retry(
            sender,
            html_content="<p>报告</p>",
            text_content="报告",
            subject="A股收盘日报 2026-08-19",
            wait_strategy=wait_none(),
        )

    assert len(sender.calls) == 3
    assert error.value.retryable is True


@pytest.mark.parametrize(
    "failure",
    [EmailAuthenticationFailure("auth"), EmailPermanentPreAcceptanceFailure("message_build")],
)
def test_mailer_does_not_retry_permanent_preacceptance_failure(failure) -> None:
    sender = FakeEmailSender([failure])

    with pytest.raises(DeliveryNotAcceptedError) as error:
        send_with_retry(
            sender,
            html_content="<p>报告</p>",
            text_content="报告",
            subject="A股盘前日报 2026-08-19",
            wait_strategy=wait_none(),
        )

    assert len(sender.calls) == 1
    assert error.value.retryable is False


def test_mailer_never_retries_ambiguous_send_message_outcome() -> None:
    sender = FakeEmailSender([EmailDeliveryAmbiguous("send_message")])

    with pytest.raises(DeliveryInDoubtError):
        send_with_retry(
            sender,
            html_content="<p>报告</p>",
            text_content="报告",
            subject="A股盘前日报 2026-08-19",
            wait_strategy=wait_none(),
        )

    assert len(sender.calls) == 1
