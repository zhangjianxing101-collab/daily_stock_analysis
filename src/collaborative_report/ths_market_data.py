"""Thin, typed REST client for the THS (Fuyao) market-data API.

This module validates the documented response shape before exposing a provider
payload. The next integration layer owns A-share code normalization and
market-data semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from re import fullmatch
from typing import Any, Callable, Mapping, Protocol, Sequence

import requests

from .settings import DEFAULT_THS_BASE_URL, ThsSettings


class ThsEndpoint(str, Enum):
    A_SHARE_SNAPSHOT = "/api/a-share/prices/snapshot"
    A_SHARE_HISTORICAL = "/api/a-share/prices/historical"
    FINANCIAL_INDICATORS = "/api/a-share/financials/indicators"
    SKYROCKET_LIST = "/api/a-share/special-data/skyrocket-list"
    HOT_STOCK_LIST = "/api/a-share/special-data/hot-stock-list"
    HOT_STOCK_LIST_HISTORY = "/api/a-share/special-data/hot-stock-list-history"
    HOT_STOCK_RANK_TREND = "/api/a-share/special-data/hot-stock-rank-trend"
    THS_INDEX_CATALOG = "/api/a-share-index/catalog/ths-index-list"
    THS_INDEX_CONSTITUENTS = "/api/a-share-index/constituents/ths-stock-list"
    INDEX_SNAPSHOT = "/api/a-share-index/prices/snapshot"
    INDEX_HISTORICAL = "/api/a-share-index/prices/historical"


class ThsAdjust(str, Enum):
    NONE = "none"
    FORWARD = "forward"
    BACKWARD = "backward"


class ThsHotListPeriod(str, Enum):
    DAY = "day"
    HOUR = "hour"


class ThsIndexTag(str, Enum):
    CONCEPT = "cn_concept"
    REGION = "region"
    FEATURED = "tszs"
    INDUSTRY = "industry"


@dataclass(frozen=True)
class ThsHttpResponse:
    """Minimal HTTP response contract so tests do not need a live network."""

    status_code: int
    payload: Any


class ThsHttpTransport(Protocol):
    def __call__(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        params: Mapping[str, str | int],
        timeout_seconds: float,
    ) -> ThsHttpResponse: ...


@dataclass(frozen=True)
class ThsApiResponse:
    """Validated THS API envelope without exposing request credentials."""

    data: Mapping[str, Any]
    request_id: str | None

    @property
    def timestamp_ms(self) -> int | None:
        value = self.data.get("timestamp")
        return value if type(value) is int else None

    @property
    def items(self) -> tuple[Mapping[str, Any], ...]:
        value = self.data.get("item", ())
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, Mapping))


class ThsMarketDataError(RuntimeError):
    """Base error whose public text never includes credentials or response body."""


class ThsConfigurationError(ThsMarketDataError):
    pass


_THS_NETWORK_REASONS = frozenset({
    "network_failed",
    "tls_failed",
    "proxy_failed",
    "connect_timeout",
    "read_timeout",
    "connection_failed",
    "internal_failure",
})


class ThsNetworkError(ThsMarketDataError):
    def __init__(self, reason: str = "network_failed") -> None:
        self.reason = reason if reason in _THS_NETWORK_REASONS else "network_failed"
        message = "THS client internal failure" if self.reason == "internal_failure" else "THS network request failed"
        super().__init__(message)


class ThsResponseError(ThsMarketDataError):
    pass


class ThsApiError(ThsMarketDataError):
    def __init__(self, code: int, request_id: str | None = None):
        self.code = code
        self.request_id = request_id
        detail = f"THS API request failed (code={code}"
        if request_id:
            detail += f", request_id={request_id}"
        super().__init__(detail + ")")


class ThsAuthenticationError(ThsApiError):
    pass


class ThsPermissionError(ThsApiError):
    pass


class ThsRateLimitError(ThsApiError):
    pass


class ThsDataUnavailableError(ThsApiError):
    pass


_RETRYABLE_API_CODES = frozenset({4001, 5001, 5002, 5003})
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _requests_transport(
    *,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, str | int],
    timeout_seconds: float,
) -> ThsHttpResponse:
    try:
        response = requests.get(
            url,
            headers=dict(headers),
            params=dict(params),
            timeout=timeout_seconds,
            verify=True,
            allow_redirects=False,
        )
    except requests.exceptions.SSLError:
        raise ThsNetworkError("tls_failed") from None
    except requests.exceptions.ProxyError:
        raise ThsNetworkError("proxy_failed") from None
    except requests.exceptions.ConnectTimeout:
        raise ThsNetworkError("connect_timeout") from None
    except requests.exceptions.ReadTimeout:
        raise ThsNetworkError("read_timeout") from None
    except requests.exceptions.ConnectionError:
        raise ThsNetworkError("connection_failed") from None
    except requests.Timeout:
        raise ThsNetworkError from None
    except requests.RequestException:
        raise ThsNetworkError from None

    try:
        payload = response.json()
    except ValueError:
        raise ThsResponseError("THS API returned invalid JSON") from None
    return ThsHttpResponse(status_code=response.status_code, payload=payload)


def _request_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > 128:
        return None
    if not all(character.isascii() and (character.isalnum() or character in "-_") for character in normalized):
        return None
    return normalized


def _require_non_empty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _require_timestamp(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer timestamp")
    return value


def _is_timestamp(value: object) -> bool:
    return type(value) is int and value >= 0


def _require_item_list(data: Mapping[str, Any]) -> None:
    items = data.get("item")
    if not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items):
        raise ThsResponseError("THS API returned an invalid success payload")


def _require_timestamped_items(data: Mapping[str, Any], *, nullable_timestamp: bool = False) -> None:
    timestamp = data.get("timestamp")
    if not _is_timestamp(timestamp) and not (nullable_timestamp and timestamp is None):
        raise ThsResponseError("THS API returned an invalid success payload")
    _require_item_list(data)


def _require_hot_list_history(data: Mapping[str, Any]) -> None:
    day = data.get("date")
    if not isinstance(day, str) or fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day) is None:
        raise ThsResponseError("THS API returned an invalid success payload")
    if not _is_timestamp(data.get("date_ms")):
        raise ThsResponseError("THS API returned an invalid success payload")
    _require_item_list(data)


def _require_financial_indicators(data: Mapping[str, Any]) -> None:
    thscode = data.get("thscode")
    report = data.get("report")
    abilities = data.get("abilities")
    if (
        not isinstance(thscode, str)
        or not thscode.strip()
        or not isinstance(report, str)
        or fullmatch(r"[0-9]{4}-[1-4]", report) is None
        or not isinstance(abilities, list)
        or not abilities
    ):
        raise ThsResponseError("THS API returned an invalid success payload")
    for ability in abilities:
        if not isinstance(ability, Mapping):
            raise ThsResponseError("THS API returned an invalid success payload")
        ability_name = ability.get("ability")
        indicators = ability.get("indicators")
        if not isinstance(ability_name, str) or not ability_name.strip() or not isinstance(indicators, list):
            raise ThsResponseError("THS API returned an invalid success payload")
        for indicator in indicators:
            if not isinstance(indicator, Mapping):
                raise ThsResponseError("THS API returned an invalid success payload")
            index_id = indicator.get("index_id")
            value = indicator.get("value")
            if (
                not isinstance(index_id, str)
                or not index_id.strip()
                or not isinstance(value, (str, type(None)))
            ):
                raise ThsResponseError("THS API returned an invalid success payload")


def _validate_success_data(endpoint: ThsEndpoint, data: Mapping[str, Any]) -> None:
    if endpoint is ThsEndpoint.FINANCIAL_INDICATORS:
        _require_financial_indicators(data)
        return
    if endpoint is ThsEndpoint.HOT_STOCK_LIST_HISTORY:
        _require_hot_list_history(data)
        return
    _require_timestamped_items(data, nullable_timestamp=endpoint is ThsEndpoint.A_SHARE_SNAPSHOT)


class ThsMarketDataClient:
    """Read-only THS client with bounded retries and a test-injectable transport."""

    def __init__(
        self,
        settings: ThsSettings | None = None,
        *,
        transport: ThsHttpTransport = _requests_transport,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings or ThsSettings.from_env()
        self._transport = transport
        self._sleep = sleep

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def a_share_snapshot(
        self, thscodes: Sequence[str] | None = None, *, limit: int = 100, offset: int = 0
    ) -> ThsApiResponse:
        if thscodes is not None:
            params: dict[str, str | int] = {"thscodes": self._join_codes(thscodes)}
        else:
            if not 1 <= limit <= 1000 or offset < 0:
                raise ValueError("snapshot pagination is invalid")
            params = {"limit": limit, "offset": offset}
        return self._get(ThsEndpoint.A_SHARE_SNAPSHOT, params)

    def a_share_historical(
        self,
        thscode: str,
        *,
        start_ms: int,
        end_ms: int,
        adjust: ThsAdjust = ThsAdjust.FORWARD,
        offset: int = 0,
    ) -> ThsApiResponse:
        return self._historical(
            ThsEndpoint.A_SHARE_HISTORICAL,
            thscode,
            start_ms=start_ms,
            end_ms=end_ms,
            extra={"adjust": ThsAdjust(adjust).value, "offset": self._offset(offset)},
        )

    def financial_indicators(self, thscode: str, report: str) -> ThsApiResponse:
        report = _require_non_empty("report", report)
        if fullmatch(r"[0-9]{4}-[1-4]", report) is None:
            raise ValueError("report must have the format YYYY-1 through YYYY-4")
        return self._get(
            ThsEndpoint.FINANCIAL_INDICATORS,
            {"thscode": _require_non_empty("thscode", thscode), "report": report},
        )

    def skyrocket_list(self, period: ThsHotListPeriod = ThsHotListPeriod.DAY) -> ThsApiResponse:
        return self._get(ThsEndpoint.SKYROCKET_LIST, {"period": ThsHotListPeriod(period).value})

    def hot_stock_list(self, period: ThsHotListPeriod = ThsHotListPeriod.DAY) -> ThsApiResponse:
        return self._get(ThsEndpoint.HOT_STOCK_LIST, {"period": ThsHotListPeriod(period).value})

    def hot_stock_list_history(self, day: str) -> ThsApiResponse:
        return self._get(ThsEndpoint.HOT_STOCK_LIST_HISTORY, {"date": _require_non_empty("day", day)})

    def hot_stock_rank_trend(self, thscode: str, *, start_date: str, end_date: str) -> ThsApiResponse:
        return self._get(
            ThsEndpoint.HOT_STOCK_RANK_TREND,
            {
                "thscode": _require_non_empty("thscode", thscode),
                "start_date": _require_non_empty("start_date", start_date),
                "end_date": _require_non_empty("end_date", end_date),
            },
        )

    def ths_index_catalog(self, tag: ThsIndexTag = ThsIndexTag.CONCEPT) -> ThsApiResponse:
        return self._get(ThsEndpoint.THS_INDEX_CATALOG, {"tag": ThsIndexTag(tag).value})

    def ths_index_constituents(self, thscode: str) -> ThsApiResponse:
        return self._get(ThsEndpoint.THS_INDEX_CONSTITUENTS, {"thscode": _require_non_empty("thscode", thscode)})

    def index_snapshot(self, thscodes: Sequence[str]) -> ThsApiResponse:
        return self._get(ThsEndpoint.INDEX_SNAPSHOT, {"thscodes": self._join_codes(thscodes)})

    def index_historical(self, thscode: str, *, start_ms: int, end_ms: int) -> ThsApiResponse:
        return self._historical(ThsEndpoint.INDEX_HISTORICAL, thscode, start_ms=start_ms, end_ms=end_ms, extra={})

    def _historical(
        self,
        endpoint: ThsEndpoint,
        thscode: str,
        *,
        start_ms: int,
        end_ms: int,
        extra: Mapping[str, str | int],
    ) -> ThsApiResponse:
        start_ms = _require_timestamp("start_ms", start_ms)
        end_ms = _require_timestamp("end_ms", end_ms)
        if end_ms < start_ms:
            raise ValueError("end_ms must not be before start_ms")
        params: dict[str, str | int] = {
            "thscode": _require_non_empty("thscode", thscode),
            "interval": "1d",
            "start": start_ms,
            "end": end_ms,
        }
        params.update(extra)
        return self._get(endpoint, params)

    @staticmethod
    def _offset(value: int) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("offset must be a non-negative integer")
        return value

    @staticmethod
    def _join_codes(thscodes: Sequence[str]) -> str:
        if isinstance(thscodes, str):
            raise ValueError("thscodes must be a sequence of codes")
        normalized = tuple(_require_non_empty("thscode", code) for code in thscodes)
        if not normalized:
            raise ValueError("thscodes must not be empty")
        return ",".join(normalized)

    def _get(self, endpoint: ThsEndpoint, params: Mapping[str, str | int]) -> ThsApiResponse:
        if (
            not self._settings.enabled
            or not self._settings.api_key
            or self._settings.base_url != DEFAULT_THS_BASE_URL
        ):
            raise ThsConfigurationError("THS data provider is not configured")

        url = f"{self._settings.base_url}{endpoint.value}"
        headers = {"X-api-key": self._settings.api_key, "Accept": "application/json"}
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = self._transport(
                    url=url,
                    headers=headers,
                    params=params,
                    timeout_seconds=self._settings.timeout_seconds,
                )
                result = self._parse_response(endpoint, response)
            except ThsNetworkError as exc:
                if attempt == self._settings.max_retries:
                    raise ThsNetworkError(exc.reason) from None
                self._backoff(attempt)
                continue
            except (TimeoutError, ConnectionError):
                if attempt == self._settings.max_retries:
                    raise ThsNetworkError from None
                self._backoff(attempt)
                continue
            except ThsApiError as exc:
                if (
                    exc.code not in _RETRYABLE_API_CODES
                    and exc.code not in _RETRYABLE_HTTP_STATUSES
                ) or attempt == self._settings.max_retries:
                    raise
                self._backoff(attempt)
                continue
            except ThsResponseError:
                raise
            except Exception:
                if attempt == self._settings.max_retries:
                    raise ThsNetworkError("internal_failure") from None
                self._backoff(attempt)
                continue
            else:
                return result
        raise AssertionError("retry loop must return or raise")

    def _parse_response(self, endpoint: ThsEndpoint, response: ThsHttpResponse) -> ThsApiResponse:
        if type(response.status_code) is not int:
            raise ThsResponseError("THS API returned an invalid HTTP status")
        if response.status_code not in range(200, 300):
            if response.status_code in _RETRYABLE_HTTP_STATUSES:
                raise ThsApiError(response.status_code)
            raise ThsResponseError(f"THS HTTP request failed (status={response.status_code})")
        if not isinstance(response.payload, Mapping):
            raise ThsResponseError("THS API returned an invalid response envelope")

        code = response.payload.get("code")
        request_id = _request_id(response.payload.get("request_id"))
        if type(code) is not int:
            raise ThsResponseError("THS API returned an invalid response envelope")
        if code != 0:
            if code == 2001:
                raise ThsAuthenticationError(code, request_id)
            if code == 2003:
                raise ThsPermissionError(code, request_id)
            if code == 4001:
                raise ThsRateLimitError(code, request_id)
            if code in {3001, 3002, 3004}:
                raise ThsDataUnavailableError(code, request_id)
            raise ThsApiError(code, request_id)

        data = response.payload.get("data")
        if not isinstance(data, Mapping):
            raise ThsResponseError("THS API returned an invalid success payload")
        _validate_success_data(endpoint, data)
        return ThsApiResponse(data=data, request_id=request_id)

    def _backoff(self, attempt: int) -> None:
        self._sleep(0.1 * (attempt + 1))
