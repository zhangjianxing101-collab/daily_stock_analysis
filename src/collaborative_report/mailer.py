"""Bounded delivery retry for collaborative HTML reports."""

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from typing import Any, Sequence

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class DeliveryError(RuntimeError):
    """Raised when a collaborative report cannot be delivered."""


class _RetryableDeliveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeliveryResult:
    sent: bool
    attempts: int
    subject: str


_NETWORK_EXCEPTIONS = (smtplib.SMTPException, OSError, TimeoutError, ConnectionError)
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
        retry=retry_if_exception_type(_RetryableDeliveryError),
        reraise=True,
    )
    try:
        for attempt in retrying:
            with attempt:
                attempts += 1
                try:
                    sent = email_sender.send_html_email(
                        html_content,
                        text_content,
                        subject,
                        receivers=list(receivers) if receivers is not None else None,
                        timeout_seconds=timeout_seconds,
                    )
                except _NETWORK_EXCEPTIONS as exc:
                    raise _RetryableDeliveryError("email delivery failed") from exc
                if not sent:
                    raise _RetryableDeliveryError("email delivery failed")
    except _RetryableDeliveryError as exc:
        raise DeliveryError(f"email delivery failed after {attempts} attempts") from exc
    return DeliveryResult(sent=True, attempts=attempts, subject=subject)
