import os
from collections.abc import Mapping
from typing import Callable
from unittest.mock import patch

import pytest

from src.collaborative_report.settings import ThsSettings
from src.collaborative_report.ths_market_data import (
    ThsAdjust,
    ThsApiError,
    ThsAuthenticationError,
    ThsConfigurationError,
    ThsHttpResponse,
    ThsHotListPeriod,
    ThsIndexTag,
    ThsMarketDataClient,
    ThsNetworkError,
    ThsResponseError,
)


TEST_API_KEY = "test-key-that-must-not-appear-in-errors"


def settings(**overrides: str) -> ThsSettings:
    with patch.dict(os.environ, {"THS_API_KEY": TEST_API_KEY, **overrides}, clear=True):
        return ThsSettings.from_env()


def success(data: Mapping[str, object] | None = None) -> ThsHttpResponse:
    return ThsHttpResponse(
        200,
        {"code": 0, "message": "success", "request_id": "request-123", "data": data or {"item": []}},
    )


def test_settings_disable_provider_without_a_key_and_never_echo_invalid_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        configured = ThsSettings.from_env()

    assert configured.api_key is None
    assert not configured.enabled

    secret = "invalid-key-content"
    with patch.dict(os.environ, {"THS_TIMEOUT_SECONDS": secret}, clear=True), pytest.raises(ValueError) as error:
        ThsSettings.from_env()

    assert secret not in str(error.value)


def test_settings_repr_does_not_include_api_key() -> None:
    assert TEST_API_KEY not in repr(settings())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"THS_ENABLED": "sometimes"}, "THS_ENABLED must be a boolean"),
        ({"THS_BASE_URL": "http://example.test"}, "THS_BASE_URL must be an HTTPS origin"),
        ({"THS_TIMEOUT_SECONDS": "0"}, "THS_TIMEOUT_SECONDS must be greater than 0 and at most 60"),
        ({"THS_MAX_RETRIES": "4"}, "THS_MAX_RETRIES must be between 0 and 3"),
    ],
)
def test_settings_validate_provider_controls(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        settings(**overrides)


def test_snapshot_uses_api_key_header_and_parses_envelope() -> None:
    calls: list[dict[str, object]] = []

    def transport(**kwargs: object) -> ThsHttpResponse:
        calls.append(kwargs)
        return success({"timestamp": 1, "item": [{"thscode": "600000.SH", "last_price": 10.0}]})

    response = ThsMarketDataClient(settings(), transport=transport).a_share_snapshot(["600000.SH"])

    assert response.timestamp_ms == 1
    assert response.items == ({"thscode": "600000.SH", "last_price": 10.0},)
    assert calls == [
        {
            "url": "https://fuyao.aicubes.cn/api/a-share/prices/snapshot",
            "headers": {"X-api-key": TEST_API_KEY, "Accept": "application/json"},
            "params": {"thscodes": "600000.SH"},
            "timeout_seconds": 10.0,
        }
    ]


@pytest.mark.parametrize(
    ("operation", "path", "params"),
    [
        (
            lambda client: client.a_share_historical("600000.SH", start_ms=1, end_ms=2, adjust=ThsAdjust.NONE),
            "/api/a-share/prices/historical",
            {"thscode": "600000.SH", "interval": "1d", "start": 1, "end": 2, "adjust": "none", "offset": 0},
        ),
        (
            lambda client: client.financial_indicators("600000.SH", "2026-1"),
            "/api/a-share/financials/indicators",
            {"thscode": "600000.SH", "report": "2026-1"},
        ),
        (
            lambda client: client.skyrocket_list(ThsHotListPeriod.HOUR),
            "/api/a-share/special-data/skyrocket-list",
            {"period": "hour"},
        ),
        (lambda client: client.hot_stock_list(), "/api/a-share/special-data/hot-stock-list", {"period": "day"}),
        (
            lambda client: client.hot_stock_list_history("2026-08-24"),
            "/api/a-share/special-data/hot-stock-list-history",
            {"date": "2026-08-24"},
        ),
        (
            lambda client: client.hot_stock_rank_trend(
                "600000.SH", start_date="2026-08-01", end_date="2026-08-24"
            ),
            "/api/a-share/special-data/hot-stock-rank-trend",
            {"thscode": "600000.SH", "start_date": "2026-08-01", "end_date": "2026-08-24"},
        ),
        (
            lambda client: client.ths_index_catalog(ThsIndexTag.INDUSTRY),
            "/api/a-share-index/catalog/ths-index-list",
            {"tag": "industry"},
        ),
        (
            lambda client: client.ths_index_constituents("886042.TI"),
            "/api/a-share-index/constituents/ths-stock-list",
            {"thscode": "886042.TI"},
        ),
        (
            lambda client: client.index_snapshot(["000001.SH", "886042.TI"]),
            "/api/a-share-index/prices/snapshot",
            {"thscodes": "000001.SH,886042.TI"},
        ),
        (
            lambda client: client.index_historical("886042.TI", start_ms=1, end_ms=2),
            "/api/a-share-index/prices/historical",
            {"thscode": "886042.TI", "interval": "1d", "start": 1, "end": 2},
        ),
    ],
)
def test_client_exposes_all_task_one_endpoints(
    operation: Callable[[ThsMarketDataClient], object], path: str, params: dict[str, object]
) -> None:
    calls: list[dict[str, object]] = []

    def transport(**kwargs: object) -> ThsHttpResponse:
        calls.append(kwargs)
        return success()

    operation(ThsMarketDataClient(settings(), transport=transport))

    assert calls[0]["url"] == f"https://fuyao.aicubes.cn{path}"
    assert calls[0]["params"] == params


def test_client_requires_configuration_without_attempting_transport() -> None:
    calls: list[object] = []
    client = ThsMarketDataClient(
        ThsSettings(None, False, "https://fuyao.aicubes.cn", 10, 2),
        transport=lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ThsConfigurationError, match="^THS data provider is not configured$"):
        client.hot_stock_list()

    assert calls == []


def test_authentication_error_is_not_retried_and_does_not_leak_secret() -> None:
    calls = 0

    def transport(**kwargs: object) -> ThsHttpResponse:
        nonlocal calls
        calls += 1
        return ThsHttpResponse(
            200,
            {"code": 2001, "message": TEST_API_KEY, "request_id": "safe-id", "data": None},
        )

    with pytest.raises(ThsAuthenticationError) as error:
        ThsMarketDataClient(settings(), transport=transport).hot_stock_list()

    assert calls == 1
    assert TEST_API_KEY not in str(error.value)
    assert "safe-id" in str(error.value)


def test_retryable_api_error_retries_with_injected_sleep() -> None:
    outcomes = iter(
        [
            ThsHttpResponse(
                200,
                {"code": 4001, "message": "slow down", "request_id": "first", "data": None},
            ),
            success({"item": [{"rank": 1}]}),
        ]
    )
    pauses: list[float] = []
    client = ThsMarketDataClient(settings(THS_MAX_RETRIES="1"), transport=lambda **kwargs: next(outcomes), sleep=pauses.append)

    result = client.hot_stock_list()

    assert result.items == ({"rank": 1},)
    assert pauses == [0.1]


def test_timeout_retries_then_raises_safe_network_error() -> None:
    pauses: list[float] = []

    def timeout(**kwargs: object) -> ThsHttpResponse:
        raise TimeoutError

    with pytest.raises(ThsNetworkError, match="^THS network request failed$"):
        ThsMarketDataClient(settings(THS_MAX_RETRIES="1"), transport=timeout, sleep=pauses.append).hot_stock_list()

    assert pauses == [0.1]


def test_retryable_http_status_retries() -> None:
    outcomes = iter([ThsHttpResponse(503, {"detail": "upstream unavailable"}), success()])
    pauses: list[float] = []

    result = ThsMarketDataClient(
        settings(THS_MAX_RETRIES="1"), transport=lambda **kwargs: next(outcomes), sleep=pauses.append
    ).hot_stock_list()

    assert result.items == ()
    assert pauses == [0.1]


def test_unexpected_transport_error_is_sanitized() -> None:
    def transport(**kwargs: object) -> ThsHttpResponse:
        raise RuntimeError(TEST_API_KEY)

    with pytest.raises(ThsNetworkError) as error:
        ThsMarketDataClient(settings(THS_MAX_RETRIES="0"), transport=transport).hot_stock_list()

    assert TEST_API_KEY not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        ThsHttpResponse(200, []),
        ThsHttpResponse(200, {"code": "0", "data": {}}),
        ThsHttpResponse(200, {"code": 0, "data": None}),
        ThsHttpResponse(403, {"detail": "forbidden"}),
    ],
)
def test_malformed_or_non_retryable_http_responses_are_safe(response: ThsHttpResponse) -> None:
    with pytest.raises(ThsResponseError) as error:
        ThsMarketDataClient(settings(), transport=lambda **kwargs: response).hot_stock_list()

    assert TEST_API_KEY not in str(error.value)


def test_bad_inputs_fail_before_a_network_request() -> None:
    client = ThsMarketDataClient(settings(), transport=lambda **kwargs: pytest.fail("transport should not run"))

    with pytest.raises(ValueError, match="^thscodes must not be empty$"):
        client.index_snapshot([])
    with pytest.raises(ValueError, match="^thscodes must be a sequence of codes$"):
        client.index_snapshot("000001.SH")
    with pytest.raises(ValueError, match="^end_ms must not be before start_ms$"):
        client.index_historical("886042.TI", start_ms=2, end_ms=1)
    with pytest.raises(ValueError, match="^report must have the format YYYY-1 through YYYY-4$"):
        client.financial_indicators("600000.SH", "2026-5")
