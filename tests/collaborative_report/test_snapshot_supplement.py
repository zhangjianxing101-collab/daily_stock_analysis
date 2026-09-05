from datetime import datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

import src.collaborative_report.snapshot_supplement as snapshot_supplement
from src.collaborative_report.snapshot_supplement import (
    TENCENT_BATCH_SIZE,
    TENCENT_MAX_CONSECUTIVE_FAILURES,
    TencentQuoteResponse,
    TencentSnapshotSupplementClient,
)


RECEIPT = datetime(2026, 9, 4, 15, 1, tzinfo=ZoneInfo("Asia/Shanghai"))


def quote(symbol: str, code: str, **overrides: str) -> str:
    fields = [""] * 50
    fields[1] = "Example"
    fields[2] = code
    fields[3] = "10.25"
    fields[6] = "123"
    fields[30] = "20260904150000"
    fields[35] = "10.25/123/4567"
    fields[38] = "2.5"
    fields[49] = "1.2"
    for index, value in overrides.items():
        fields[int(index)] = value
    return f'v_{symbol}="{"~".join(fields)}";'


def test_fetch_validates_bound_identity_and_optional_metrics() -> None:
    calls: list[dict[str, object]] = []

    def transport(**kwargs: object) -> TencentQuoteResponse:
        calls.append(kwargs)
        return TencentQuoteResponse(200, quote("sh600000", "600000"))

    result = TencentSnapshotSupplementClient(transport=transport, clock=lambda: RECEIPT).fetch(["600000"])

    assert calls == [{"url": "https://qt.gtimg.cn/q=sh600000", "timeout_seconds": 8.0}]
    assert result.observed_at == RECEIPT
    assert result.missing_count == 0
    assert result.warnings == ()
    assert result.frame.to_dict("records") == [
        {
            "code": "600000",
            "name": "Example",
            "price": 10.25,
            "volume_ratio": 1.2,
            "turnover": 2.5,
            "volume": 12300,
            "amount": 4567.0,
            "source_timestamp": datetime(2026, 9, 4, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        quote("sh600000", "600001"),
        quote("sh600000", "600000") + "\n" + quote("sh600000", "600000"),
    ],
)
def test_fetch_excludes_identity_mismatches_and_duplicate_response_rows(payload: str) -> None:
    result = TencentSnapshotSupplementClient(
        transport=lambda **kwargs: TencentQuoteResponse(200, payload), clock=lambda: RECEIPT
    ).fetch(["600000"])

    assert result.frame.empty
    assert result.missing_count == 1
    assert result.warnings == (
        "Tencent supplemental quote batch unavailable",
        "Tencent supplemental quote rows missing",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"30": "20260904150101"},
        {"30": "not-a-timestamp"},
        {"1": ""},
        {"1": " -- "},
        {"1": "NULL"},
        {"1": "none"},
        {"1": "n/a"},
        {"3": "nan"},
        {"38": "-1"},
        {"49": "infinity"},
    ],
)
def test_fetch_excludes_invalid_source_times_and_required_values(overrides: dict[str, str]) -> None:
    result = TencentSnapshotSupplementClient(
        transport=lambda **kwargs: TencentQuoteResponse(200, quote("sh600000", "600000", **overrides)),
        clock=lambda: RECEIPT,
    ).fetch(["600000"])

    assert result.frame.empty
    assert result.missing_count == 1


def test_fetch_ignores_unsolicited_rows_without_counting_them() -> None:
    payload = "\n".join([quote("sh600000", "600000"), quote("sz000001", "000001")])
    result = TencentSnapshotSupplementClient(
        transport=lambda **kwargs: TencentQuoteResponse(200, payload), clock=lambda: RECEIPT
    ).fetch(["600000"])

    assert result.frame["code"].tolist() == ["600000"]
    assert result.missing_count == 0


def test_fetch_parses_concatenated_tencent_assignments() -> None:
    payload = quote("sh600000", "600000") + quote("sz000001", "000001")
    result = TencentSnapshotSupplementClient(
        transport=lambda **kwargs: TencentQuoteResponse(200, payload), clock=lambda: RECEIPT
    ).fetch(["600000", "000001"])

    assert result.frame["code"].tolist() == ["600000", "000001"]
    assert result.missing_count == 0


def test_fetch_batches_serially_and_stops_after_three_consecutive_failures() -> None:
    calls: list[dict[str, object]] = []
    secret = "ths-api-key=never-forward-this"

    def transport(**kwargs: object) -> TencentQuoteResponse:
        calls.append(kwargs)
        raise RuntimeError(secret)

    codes = [f"600{number:03d}" for number in range(TENCENT_BATCH_SIZE * 4)]
    result = TencentSnapshotSupplementClient(transport=transport, clock=lambda: RECEIPT).fetch(codes)

    assert len(calls) == TENCENT_MAX_CONSECUTIVE_FAILURES
    assert all(call["timeout_seconds"] == 8.0 for call in calls)
    assert all(call["url"].startswith("https://qt.gtimg.cn/q=") for call in calls)
    assert all(secret not in str(call) for call in calls)
    assert result.frame.empty
    assert result.missing_count == len(codes)
    assert result.warnings == (
        "Tencent supplemental quote batch unavailable",
        "Tencent supplemental quote requests stopped after consecutive failures",
        "Tencent supplemental quote rows missing",
    )


@pytest.mark.parametrize("payload", ["", 'v_sh600000="malformed";'])
def test_fetch_stops_after_three_empty_or_wholly_invalid_http_200_batches(payload: str) -> None:
    calls: list[dict[str, object]] = []

    def transport(**kwargs: object) -> TencentQuoteResponse:
        calls.append(kwargs)
        return TencentQuoteResponse(200, payload)

    codes = [f"600{number:03d}" for number in range(TENCENT_BATCH_SIZE * 4)]
    result = TencentSnapshotSupplementClient(transport=transport, clock=lambda: RECEIPT).fetch(codes)

    assert len(calls) == TENCENT_MAX_CONSECUTIVE_FAILURES
    assert result.frame.empty
    assert result.missing_count == len(codes)
    assert result.warnings == (
        "Tencent supplemental quote batch unavailable",
        "Tencent supplemental quote requests stopped after consecutive failures",
        "Tencent supplemental quote rows missing",
    )


def test_fetch_rejects_a_receipt_clock_that_moves_backward_between_batches() -> None:
    receipts = iter([RECEIPT, RECEIPT - timedelta(seconds=1)])
    payload = "\n".join([quote("sh600000", "600000"), quote("sh600100", "600100")])
    client = TencentSnapshotSupplementClient(
        transport=lambda **kwargs: TencentQuoteResponse(200, payload), clock=lambda: next(receipts)
    )

    with pytest.raises(ValueError, match="^receipt clock must be monotonic$"):
        client.fetch([f"600{number:03d}" for number in range(TENCENT_BATCH_SIZE + 1)])


@pytest.mark.parametrize(
    "codes",
    [
        ("600000",),
        ["600000", "600000"],
        ["600000.SH"],
        ["900901"],
        ["999999"],
        ["\uff16\uff10\uff10\uff10\uff10\uff10"],
        ["600000"] * 10_001,
    ],
)
def test_fetch_rejects_non_list_duplicate_or_unsupported_codes(codes: object) -> None:
    client = TencentSnapshotSupplementClient(transport=lambda **kwargs: pytest.fail("transport should not run"))

    with pytest.raises(ValueError):
        client.fetch(codes)  # type: ignore[arg-type]


def test_requests_transport_enforces_tls_timeout_and_no_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Response:
        status_code = 200
        encoding: str | None = None
        text = 'v_sh600000="";'

    def get(*args: object, **kwargs: object) -> Response:
        calls.append({"args": args, "kwargs": kwargs})
        return Response()

    monkeypatch.setattr(snapshot_supplement.requests, "get", get)
    response = snapshot_supplement._requests_transport(
        url="https://qt.gtimg.cn/q=sh600000", timeout_seconds=8.0
    )

    assert response.status_code == 200
    assert calls == [{
        "args": ("https://qt.gtimg.cn/q=sh600000",),
        "kwargs": {"timeout": 8.0, "verify": True, "allow_redirects": False},
    }]


def test_stale_source_timestamp_is_retained_for_the_main_quality_gate() -> None:
    stale = (RECEIPT - timedelta(days=10)).strftime("%Y%m%d%H%M%S")
    result = TencentSnapshotSupplementClient(
        transport=lambda **kwargs: TencentQuoteResponse(200, quote("sh600000", "600000", **{"30": stale})),
        clock=lambda: RECEIPT,
    ).fetch(["600000"])

    assert result.frame.loc[0, "source_timestamp"] == RECEIPT - timedelta(days=10)
