import os
import traceback
from collections.abc import Mapping
from typing import Callable
from unittest.mock import patch

import pytest

import src.collaborative_report.ths_market_data as ths_market_data
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
        {
            "code": 0,
            "message": "success",
            "request_id": "request-123",
            "data": data if data is not None else {"timestamp": 1, "item": []},
        },
    )


def valid_response_data(url: str) -> Mapping[str, object]:
    if url.endswith("/api/a-share/financials/indicators"):
        return {
            "thscode": "600000.SH",
            "report": "2026-1",
            "abilities": [
                {
                    "ability": "growth",
                    "indicators": [{"index_id": "net_profit_yoy_growth_ratio", "value": "1.2"}],
                }
            ],
        }
    if url.endswith("/api/a-share/special-data/hot-stock-list-history"):
        return {"date": "2026-08-24", "date_ms": 1, "item": []}
    return {"timestamp": 1, "item": []}


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
        ({"THS_TIMEOUT_SECONDS": "0"}, "THS_TIMEOUT_SECONDS must be greater than 0 and at most 60"),
        ({"THS_MAX_RETRIES": "4"}, "THS_MAX_RETRIES must be between 0 and 3"),
    ],
)
def test_settings_validate_provider_controls(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        settings(**overrides)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://fuyao.aicubes.cn",
        "https://attacker.example",
        "https://fuyao.aicubes.cn/",
        "https://fuyao.aicubes.cn/api",
        "https://fuyao.aicubes.cn?target=attacker.example",
        "https://user@fuyao.aicubes.cn",
    ],
)
def test_settings_only_accepts_the_exact_official_ths_origin(base_url: str) -> None:
    with pytest.raises(ValueError, match="^THS_BASE_URL must be exactly https://fuyao.aicubes.cn$"):
        settings(THS_BASE_URL=base_url)


def test_requests_transport_disables_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Response:
        status_code = 302

        @staticmethod
        def json() -> object:
            return {"code": 0, "data": {"timestamp": 1, "item": []}}

    def get(*args: object, **kwargs: object) -> Response:
        calls.append({"args": args, "kwargs": kwargs})
        return Response()

    monkeypatch.setattr(ths_market_data.requests, "get", get)

    ths_market_data._requests_transport(
        url="https://fuyao.aicubes.cn/api/a-share/prices/snapshot",
        headers={"X-api-key": TEST_API_KEY},
        params={"thscodes": "600000.SH"},
        timeout_seconds=10,
    )

    assert calls[0]["kwargs"] == {
        "headers": {"X-api-key": TEST_API_KEY},
        "params": {"thscodes": "600000.SH"},
        "timeout": 10,
        "verify": True,
        "allow_redirects": False,
    }


@pytest.mark.parametrize(
    ("request_error", "reason"),
    [
        (ths_market_data.requests.exceptions.SSLError, "tls_failed"),
        (ths_market_data.requests.exceptions.ProxyError, "proxy_failed"),
        (ths_market_data.requests.exceptions.ConnectTimeout, "connect_timeout"),
        (ths_market_data.requests.exceptions.ReadTimeout, "read_timeout"),
        (ths_market_data.requests.exceptions.ConnectionError, "connection_failed"),
    ],
)
def test_requests_transport_maps_network_errors_without_leaking_provider_text(
    monkeypatch: pytest.MonkeyPatch,
    request_error: type[Exception],
    reason: str,
) -> None:
    secret = "transport-secret-token"

    def get(*args: object, **kwargs: object) -> object:
        raise request_error(secret)

    monkeypatch.setattr(ths_market_data.requests, "get", get)

    with pytest.raises(ThsNetworkError) as error:
        ths_market_data._requests_transport(
            url="https://fuyao.aicubes.cn/api/a-share/prices/snapshot",
            headers={"X-api-key": TEST_API_KEY},
            params={"thscodes": "600000.SH"},
            timeout_seconds=10,
        )

    assert error.value.reason == reason
    assert secret not in str(error.value)
    assert secret not in "".join(traceback.format_exception(error.type, error.value, error.tb))


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
        return success(valid_response_data(str(kwargs["url"])))

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


def test_client_rejects_a_non_official_origin_before_sending_the_api_key() -> None:
    client = ThsMarketDataClient(
        ThsSettings(TEST_API_KEY, True, "https://attacker.example", 10, 0),
        transport=lambda **kwargs: pytest.fail("transport should not run"),
    )

    with pytest.raises(ThsConfigurationError, match="^THS data provider is not configured$"):
        client.hot_stock_list()


@pytest.mark.parametrize("api_key", ["contains space", "contains\nnewline", "contains-\u2603"])
def test_client_rejects_unencodable_header_credentials_without_sending_them(api_key: str) -> None:
    transport = lambda **kwargs: pytest.fail("transport should not run")

    with pytest.raises(ThsNetworkError) as error:
        ThsMarketDataClient(settings(THS_API_KEY=api_key), transport=transport).hot_stock_list()

    assert error.value.reason == "credential_encoding_failed"
    assert api_key not in str(error.value)


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
            success({"timestamp": 1, "item": [{"rank": 1}]}),
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

    with pytest.raises(ThsNetworkError, match="^THS network request failed$") as error:
        ThsMarketDataClient(settings(THS_MAX_RETRIES="1"), transport=timeout, sleep=pauses.append).hot_stock_list()

    assert error.value.reason == "network_failed"
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

    assert error.value.reason == "internal_failure"
    assert str(error.value) == "THS client internal failure"
    assert TEST_API_KEY not in str(error.value)
    formatted_traceback = "".join(traceback.format_exception(error.type, error.value, error.tb))
    assert TEST_API_KEY not in formatted_traceback


@pytest.mark.parametrize(
    ("error_factory", "reason"),
    [
        (
            lambda secret: UnicodeEncodeError("ascii", secret, 0, 1, "not encodable"),
            "credential_encoding_failed",
        ),
        (TypeError, "internal_type_error"),
        (ValueError, "internal_value_error"),
        (AttributeError, "internal_attribute_error"),
    ],
)
def test_internal_transport_errors_have_fixed_redacted_reasons(
    error_factory: Callable[[str], Exception], reason: str
) -> None:
    secret = "internal-transport-secret"

    def transport(**kwargs: object) -> ThsHttpResponse:
        raise error_factory(secret)

    with pytest.raises(ThsNetworkError) as error:
        ThsMarketDataClient(settings(THS_MAX_RETRIES="0"), transport=transport).hot_stock_list()

    assert error.value.reason == reason
    assert secret not in str(error.value)
    assert secret not in "".join(traceback.format_exception(error.type, error.value, error.tb))


def test_internal_reason_uses_the_existing_retry_policy() -> None:
    calls = 0
    pauses: list[float] = []

    def transport(**kwargs: object) -> ThsHttpResponse:
        nonlocal calls
        calls += 1
        raise ValueError("internal-transport-secret")

    with pytest.raises(ThsNetworkError) as error:
        ThsMarketDataClient(
            settings(THS_MAX_RETRIES="1"), transport=transport, sleep=pauses.append
        ).hot_stock_list()

    assert error.value.reason == "internal_value_error"
    assert calls == 2
    assert pauses == [0.1]


def test_network_reason_survives_the_existing_retry_policy() -> None:
    calls = 0
    pauses: list[float] = []

    def transport(**kwargs: object) -> ThsHttpResponse:
        nonlocal calls
        calls += 1
        raise ThsNetworkError("read_timeout")

    with pytest.raises(ThsNetworkError) as error:
        ThsMarketDataClient(
            settings(THS_MAX_RETRIES="1"), transport=transport, sleep=pauses.append
        ).hot_stock_list()

    assert error.value.reason == "read_timeout"
    assert calls == 2
    assert pauses == [0.1]


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


@pytest.mark.parametrize(
    ("operation", "data"),
    [
        (lambda client: client.a_share_historical("600000.SH", start_ms=1, end_ms=2), {"item": []}),
        (lambda client: client.hot_stock_list(), {"timestamp": 1, "item": ["not-an-object"]}),
        (lambda client: client.hot_stock_list_history("2026-08-24"), {"timestamp": 1, "item": []}),
        (
            lambda client: client.financial_indicators("600000.SH", "2026-1"),
            {"thscode": "600000.SH", "report": "2026-1", "abilities": [{"ability": "growth"}]},
        ),
        (
            lambda client: client.financial_indicators("600000.SH", "2026-1"),
            {"thscode": "600000.SH", "report": "2026-1", "abilities": []},
        ),
    ],
)
def test_success_payload_shapes_are_validated_by_endpoint(
    operation: Callable[[ThsMarketDataClient], object], data: Mapping[str, object]
) -> None:
    with pytest.raises(ThsResponseError, match="^THS API returned an invalid success payload$"):
        operation(ThsMarketDataClient(settings(), transport=lambda **kwargs: success(data)))


def test_financial_indicator_payload_uses_its_documented_non_list_shape() -> None:
    response = ThsMarketDataClient(
        settings(),
        transport=lambda **kwargs: success(valid_response_data("/api/a-share/financials/indicators")),
    ).financial_indicators("600000.SH", "2026-1")

    assert response.items == ()
    assert response.data["abilities"] == [
        {
            "ability": "growth",
            "indicators": [{"index_id": "net_profit_yoy_growth_ratio", "value": "1.2"}],
        }
    ]


def test_bad_inputs_fail_before_a_network_request() -> None:
    client = ThsMarketDataClient(settings(), transport=lambda **kwargs: pytest.fail("transport should not run"))

    with pytest.raises(ValueError, match="^thscodes must not be empty$"):
        client.index_snapshot([])
    with pytest.raises(ValueError, match="^thscodes must not be empty$"):
        client.a_share_snapshot([])
    with pytest.raises(ValueError, match="^thscodes must be a sequence of codes$"):
        client.index_snapshot("000001.SH")
    with pytest.raises(ValueError, match="^end_ms must not be before start_ms$"):
        client.index_historical("886042.TI", start_ms=2, end_ms=1)
    with pytest.raises(ValueError, match="^report must have the format YYYY-1 through YYYY-4$"):
        client.financial_indicators("600000.SH", "2026-5")


def test_none_stock_codes_requests_a_share_snapshot_pagination() -> None:
    calls: list[dict[str, object]] = []

    def transport(**kwargs: object) -> ThsHttpResponse:
        calls.append(kwargs)
        return success({"timestamp": None, "item": []})

    ThsMarketDataClient(settings(), transport=transport).a_share_snapshot(None, limit=50, offset=10)

    assert calls[0]["params"] == {"limit": 50, "offset": 10}
