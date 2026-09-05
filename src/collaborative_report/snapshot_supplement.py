"""Bounded Tencent quote supplements for collaborative-report snapshots.

This module deliberately has no knowledge of THS credentials or report joins.
It only obtains a small, validated set of public Tencent quote fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from data_provider.akshare_fetcher import (
    _normalize_tencent_volume,
    _parse_tencent_amount,
    _to_sina_tx_symbol,
)


TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
TENCENT_TIMEOUT_SECONDS = 8.0
TENCENT_BATCH_SIZE = 100
TENCENT_MAX_CODES = 10_000
TENCENT_MAX_CONSECUTIVE_FAILURES = 3

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_A_SHARE_CODE = re.compile(r"^[0-9]{6}$")
_TENCENT_LINE = re.compile(r'(?<![A-Za-z0-9_])v_([a-z]{2}[0-9]{6})="([^"]*)";')
_SUPPORTED_PREFIXES = frozenset({
    "000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688", "689",
})
_SUPPORTED_BSE_PREFIXES = ("43", "83", "87", "88", "92")
_FRAME_COLUMNS = ("code", "name", "price", "volume_ratio", "turnover", "volume", "amount", "source_timestamp")

_WARNING_BATCH_UNAVAILABLE = "Tencent supplemental quote batch unavailable"
_WARNING_FAILURE_LIMIT = "Tencent supplemental quote requests stopped after consecutive failures"
_WARNING_ROWS_MISSING = "Tencent supplemental quote rows missing"
_PLACEHOLDER_NAMES = frozenset({"--", "null", "none", "n/a"})


@dataclass(frozen=True)
class TencentQuoteResponse:
    """Small transport response contract that avoids exposing HTTP internals."""

    status_code: int
    text: str


class TencentQuoteTransport(Protocol):
    def __call__(self, *, url: str, timeout_seconds: float) -> TencentQuoteResponse: ...


@dataclass(frozen=True)
class SnapshotSupplement:
    """Validated rows from Tencent plus explicit coverage information."""

    frame: pd.DataFrame
    observed_at: datetime
    missing_count: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")


def _requests_transport(*, url: str, timeout_seconds: float) -> TencentQuoteResponse:
    """Use Tencent's public endpoint with certificate verification and no redirects."""

    response = requests.get(
        url,
        timeout=timeout_seconds,
        verify=True,
        allow_redirects=False,
    )
    response.encoding = "gbk"
    return TencentQuoteResponse(status_code=response.status_code, text=response.text)


def _validated_codes(codes: list[str]) -> tuple[str, ...]:
    if not isinstance(codes, list):
        raise ValueError("codes must be a list")
    if len(codes) > TENCENT_MAX_CODES:
        raise ValueError("codes must contain at most 10000 items")

    normalized: list[str] = []
    for code in codes:
        if type(code) is not str or _A_SHARE_CODE.fullmatch(code) is None or not code.isascii():
            raise ValueError("codes must contain supported six-digit A-share codes")
        if code[:3] not in _SUPPORTED_PREFIXES and not code.startswith(_SUPPORTED_BSE_PREFIXES):
            raise ValueError("codes must contain supported six-digit A-share codes")
        normalized.append(code)
    if len(set(normalized)) != len(normalized):
        raise ValueError("codes must not contain duplicates")
    return tuple(normalized)


def _receipt(clock: Callable[[], datetime], previous: datetime | None = None) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("receipt clock must return a timezone-aware datetime")
    if previous is not None and value < previous:
        raise ValueError("receipt clock must be monotonic")
    return value


def _finite_nonnegative(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _positive_finite(value: str) -> float | None:
    parsed = _finite_nonnegative(value)
    return parsed if parsed is not None and parsed > 0 else None


def _source_timestamp(value: str, receipt: datetime) -> datetime | None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{14}", value) is None:
        return None
    try:
        timestamp = datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=_SHANGHAI)
    except ValueError:
        return None
    return timestamp if timestamp <= receipt else None


def _optional_volume(fields: list[str]) -> int | None:
    try:
        value = _normalize_tencent_volume(fields)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None
    return value if type(value) is int and value >= 0 else None


def _optional_amount(fields: list[str]) -> float | None:
    try:
        value = _parse_tencent_amount(fields)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None
    return value if value is not None and math.isfinite(value) and value >= 0 else None


def _parse_batch(
    text: str,
    expected: dict[str, str],
    receipt: datetime,
) -> list[dict[str, object]]:
    """Return valid rows; invalid and unsolicited provider rows are excluded."""

    if not isinstance(text, str):
        return []
    matched: dict[str, list[str]] = {}
    for symbol, payload in _TENCENT_LINE.findall(text):
        if symbol in expected:
            matched.setdefault(symbol, []).append(payload)

    rows: list[dict[str, object]] = []
    for symbol, payloads in matched.items():
        if len(payloads) != 1:
            continue
        fields = payloads[0].split("~")
        code = expected[symbol]
        if len(fields) <= 49 or fields[2] != code:
            continue
        name = fields[1].strip()
        price = _positive_finite(fields[3])
        volume_ratio = _finite_nonnegative(fields[49])
        turnover = _finite_nonnegative(fields[38])
        source_timestamp = _source_timestamp(fields[30], receipt)
        if (
            not name
            or name.lower() in _PLACEHOLDER_NAMES
            or price is None
            or volume_ratio is None
            or turnover is None
            or source_timestamp is None
        ):
            continue
        rows.append(
            {
                "code": code,
                "name": name,
                "price": price,
                "volume_ratio": volume_ratio,
                "turnover": turnover,
                "volume": _optional_volume(fields),
                "amount": _optional_amount(fields),
                "source_timestamp": source_timestamp,
            }
        )
    return rows


class TencentSnapshotSupplementClient:
    """Fetch serial Tencent quote batches without forwarding any provider credentials."""

    def __init__(
        self,
        *,
        transport: TencentQuoteTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._transport = transport or _requests_transport
        self._clock = clock or (lambda: datetime.now(_SHANGHAI))

    def fetch(self, codes: list[str]) -> SnapshotSupplement:
        requested = _validated_codes(codes)
        if not requested:
            return SnapshotSupplement(pd.DataFrame(columns=_FRAME_COLUMNS), _receipt(self._clock), 0)

        rows: list[dict[str, object]] = []
        consecutive_failures = 0
        batch_unavailable = False
        stopped_for_failures = False
        observed_at: datetime | None = None

        for offset in range(0, len(requested), TENCENT_BATCH_SIZE):
            batch = requested[offset : offset + TENCENT_BATCH_SIZE]
            expected = {_to_sina_tx_symbol(code): code for code in batch}
            url = TENCENT_QUOTE_URL + ",".join(expected)
            try:
                response = self._transport(url=url, timeout_seconds=TENCENT_TIMEOUT_SECONDS)
            except Exception:
                batch_unavailable = True
                consecutive_failures += 1
                observed_at = _receipt(self._clock, observed_at)
                if consecutive_failures >= TENCENT_MAX_CONSECUTIVE_FAILURES:
                    stopped_for_failures = True
                    break
                continue

            observed_at = _receipt(self._clock, observed_at)
            if not isinstance(response, TencentQuoteResponse) or response.status_code != 200:
                batch_unavailable = True
                consecutive_failures += 1
                if consecutive_failures >= TENCENT_MAX_CONSECUTIVE_FAILURES:
                    stopped_for_failures = True
                    break
                continue

            batch_rows = _parse_batch(response.text, expected, observed_at)
            if not batch_rows:
                batch_unavailable = True
                consecutive_failures += 1
                if consecutive_failures >= TENCENT_MAX_CONSECUTIVE_FAILURES:
                    stopped_for_failures = True
                    break
                continue

            consecutive_failures = 0
            rows.extend(batch_rows)
            # Missing coverage, rather than raw provider row counts, is public.
            # The main report layer owns all quality classification and reporting.

        frame = pd.DataFrame(rows, columns=_FRAME_COLUMNS)
        valid_codes = set(frame["code"].tolist()) if not frame.empty else set()
        missing_count = len(requested) - len(valid_codes)
        warnings: list[str] = []
        if batch_unavailable:
            warnings.append(_WARNING_BATCH_UNAVAILABLE)
        if stopped_for_failures:
            warnings.append(_WARNING_FAILURE_LIMIT)
        if missing_count:
            warnings.append(_WARNING_ROWS_MISSING)
        return SnapshotSupplement(
            frame=frame,
            observed_at=observed_at or _receipt(self._clock),
            missing_count=missing_count,
            warnings=tuple(warnings),
        )


def fetch_snapshot_supplement(
    codes: list[str],
    *,
    clock: Callable[[], datetime] | None = None,
    transport: TencentQuoteTransport | None = None,
) -> SnapshotSupplement:
    """Fetch a bounded supplemental snapshot without constructing a long-lived client."""

    return TencentSnapshotSupplementClient(clock=clock, transport=transport).fetch(codes)
