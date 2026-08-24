"""Bounded delivery retry for collaborative HTML reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.notification_sender.email_sender import (
    EmailAuthenticationFailure,
    EmailDeliveryAmbiguous,
    EmailPermanentPreAcceptanceFailure,
    EmailTransientPreAcceptanceFailure,
)


class DeliveryNotAcceptedError(RuntimeError):
    """SMTP definitely did not accept the message."""

    def __init__(self, *, retryable: bool, stage: str) -> None:
        super().__init__("delivery_not_accepted")
        self.retryable = retryable
        self.stage = stage


class DeliveryInDoubtError(RuntimeError):
    """SMTP acceptance may have occurred, so automatic retry is forbidden."""


@dataclass(frozen=True)
class DeliveryResult:
    sent: bool
    attempts: int
    subject: str


DEFAULT_RETRY_WAIT = wait_exponential(multiplier=2, min=2, max=4)


def send_with_retry(
    email_sender: Any,
    *,
    html_content: str,
    text_content: str,
    subject: str,
    receivers: Sequence[str] | None = None,
    timeout_seconds: float | None = None,
    wait_strategy: Any = None,
) -> DeliveryResult:
    """Send a report up to three times, waiting two then four seconds by default."""

    attempts = 0
    retrying = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_strategy if wait_strategy is not None else DEFAULT_RETRY_WAIT,
        retry=retry_if_exception_type(EmailTransientPreAcceptanceFailure),
        reraise=True,
    )
    try:
        for attempt in retrying:
            with attempt:
                attempts += 1
                try:
                    email_sender.send_html_email_strict(
                        html_content,
                        text_content,
                        subject,
                        receivers=list(receivers) if receivers is not None else None,
                        timeout_seconds=timeout_seconds,
                    )
                except EmailDeliveryAmbiguous as exc:
                    raise DeliveryInDoubtError("delivery_in_doubt") from exc
                except (EmailAuthenticationFailure, EmailPermanentPreAcceptanceFailure) as exc:
                    raise DeliveryNotAcceptedError(retryable=False, stage=exc.stage) from exc
    except EmailTransientPreAcceptanceFailure as exc:
        raise DeliveryNotAcceptedError(retryable=True, stage=exc.stage) from exc
    return DeliveryResult(sent=True, attempts=attempts, subject=subject)
