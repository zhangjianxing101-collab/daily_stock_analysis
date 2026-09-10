import traceback
from datetime import date, datetime, timedelta, timezone
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.collaborative_report.market_data import (
    MarketDataGateway,
    MarketDataset,
    normalize_a_share_snapshot,
    normalize_a_share_thscode,
    read_a_share_snapshot_archive,
    validate_daily_bars,
    write_a_share_snapshot_archive,
)
from src.collaborative_report.ths_market_data import ThsApiResponse, ThsNetworkError, ThsResponseError


SESSION = date(2026, 8, 19)
OBSERVED_AT = datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc)


def formatted_traceback(caught: pytest.ExceptionInfo[ValueError]) -> str:
    return "".join(traceback.format_exception(caught.type, caught.value, caught.tb))


def daily_bars(rows: int = 60, *, end: str = "2026-08-19") -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=rows)
    values = np.arange(rows, dtype=float) + 10.0
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": values,
            "High": values + 2.0,
            "Low": values - 1.0,
            "Close": values + 1.0,
            "Volume": np.arange(rows, dtype=float) + 1_000.0,
        }
    )


def ths_response(items: list[dict[str, object]], timestamp_ms: int | None = None) -> ThsApiResponse:
    if timestamp_ms is None:
        timestamp_ms = int((OBSERVED_AT - timedelta(hours=1)).timestamp() * 1000)
    return ThsApiResponse({"timestamp": timestamp_ms, "item": items}, None)


def ths_snapshot_item(code: str = "600000") -> dict[str, object]:
    return {
        "thscode": f"{code}.SH",
        "ticker": code,
        "last_price": 10.0,
        "price_change_ratio_pct": 1.2,
        "volume": 1000,
        "turnover": 10_000.0,
    }


def ths_bar_items(rows: int = 60) -> list[dict[str, object]]:
    frame = daily_bars(rows)
    return [
        {
            "date_ms": int(pd.Timestamp(row.Date, tz="Asia/Shanghai").timestamp() * 1000),
            "open_price": row.Open,
            "high_price": row.High,
            "low_price": row.Low,
            "close_price": row.Close,
            "volume": row.Volume,
        }
        for row in frame.itertuples(index=False)
    ]


def test_completed_session_snapshot_archive_round_trip(tmp_path) -> None:
    source_timestamp = datetime(2026, 8, 19, 15, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
    frame = normalize_a_share_snapshot(pd.DataFrame([{
        "代码": "600000", "名称": "浦发银行", "最新价": 10.0, "涨跌幅": 1.2,
        "量比": 1.1, "换手率": 2.0, "成交额": 100_000_000,
        "成交量": 10_000_000, "总市值": 300_000_000_000,
    }]))
    path = tmp_path / "market-snapshot.json"
    write_a_share_snapshot_archive(
        path,
        MarketDataset(frame, "fixture", OBSERVED_AT, (), source_timestamp),
        expected_session=SESSION,
    )

    loaded = read_a_share_snapshot_archive(
        path,
        expected_session=SESSION,
        observed_at=OBSERVED_AT + timedelta(hours=1),
    )

    assert loaded.source == "archive:fixture"
    assert loaded.source_timestamp == source_timestamp
    assert loaded.frame.to_dict(orient="records") == frame.to_dict(orient="records")
    assert loaded.frame.attrs["screening_complete_count"] == 1


def test_snapshot_archive_rejects_wrong_session(tmp_path) -> None:
    path = tmp_path / "market-snapshot.json"
    frame = normalize_a_share_snapshot(pd.DataFrame([{"代码": "600000", "最新价": 10.0}]))
    write_a_share_snapshot_archive(
        path,
        MarketDataset(
            frame,
            "fixture",
            OBSERVED_AT,
            (),
            datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        ),
        expected_session=SESSION,
    )

    with pytest.raises(ValueError, match="snapshot archive session invalid"):
        read_a_share_snapshot_archive(
            path,
            expected_session=date(2026, 8, 18),
            observed_at=OBSERVED_AT,
        )


@pytest.mark.parametrize(
    ("code", "thscode"),
    [
        ("600000", "600000.SH"),
        ("688001", "688001.SH"),
        ("000001", "000001.SZ"),
        ("300001", "300001.SZ"),
        ("430047", "430047.BJ"),
        ("830001", "830001.BJ"),
        ("920001", "920001.BJ"),
    ],
)
def test_normalize_a_share_thscode_uses_only_documented_exchange_ranges(code: str, thscode: str) -> None:
    assert normalize_a_share_thscode(code) == thscode


@pytest.mark.parametrize("code", ["900001", "400001", "60000", "600000.SH", True])
def test_normalize_a_share_thscode_rejects_unknown_or_non_six_digit_codes(code: object) -> None:
    with pytest.raises(ValueError):
        normalize_a_share_thscode(code)


def test_ths_snapshot_is_preferred_and_preserves_source_timestamp() -> None:
    client = Mock()
    client.a_share_snapshot.return_value = ths_response([ths_snapshot_item()])
    fallback = Mock()
    gateway = MarketDataGateway(ths_client=client, snapshot_fetcher=fallback, clock=lambda: OBSERVED_AT)

    result = gateway.get_a_share_snapshot(["600000"])

    client.a_share_snapshot.assert_called_once_with(("600000.SH",))
    fallback.assert_not_called()
    assert result.source == "ths.fuyao.a_share_snapshot"
    assert result.source_timestamp == datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert result.frame.loc[0, "price"] == 10.0


@pytest.mark.parametrize("activity", [0, None])
def test_ths_snapshot_quarantines_missing_unquoted_rows(activity) -> None:
    client = Mock()
    client.a_share_snapshot.return_value = ths_response([
        ths_snapshot_item("600000"),
        {**ths_snapshot_item("600001"), "last_price": None, "volume": activity, "turnover": activity},
    ])
    data = MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT).get_a_share_snapshot(["600000", "600001"])
    assert data.frame["code"].tolist() == ["600000"]
    assert data.frame.attrs["quarantined_row_count"] == 1
    assert data.frame.attrs["provider_row_count"] == 2
    assert "ths_snapshot_unquoted_rows_excluded" in data.warnings


def test_ths_snapshot_does_not_hide_missing_price_with_active_trading() -> None:
    client = Mock()
    client.a_share_snapshot.return_value = ths_response([
        ths_snapshot_item("600000"), {**ths_snapshot_item("600001"), "last_price": None},
    ])
    with pytest.raises(ValueError, match="THS snapshot contains unsafe prices"):
        MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT).get_a_share_snapshot(["600000", "600001"])


@pytest.mark.parametrize("bad", [
    {"last_price": "invalid", "volume": None, "turnover": None},
    {"last_price": None, "volume": "invalid", "turnover": None},
])
def test_ths_quarantine_does_not_hide_malformed_values(bad) -> None:
    client = Mock()
    client.a_share_snapshot.return_value = ths_response([
        ths_snapshot_item("600000"), {**ths_snapshot_item("600001"), **bad},
    ])
    with pytest.raises(ValueError):
        MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT).get_a_share_snapshot(["600000", "600001"])


@pytest.mark.parametrize("fault", [None, "future", "stale", "price", "missing", "duplicate", "unknown", "boundary", "above_boundary"])
def test_ths_supplement_requires_independent_time_identity_and_price_agreement(fault) -> None:
    stamp = datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    record = {"code": "600000", "name": "Stock", "price": 10.0,
              "volume_ratio": 1.2, "turnover": 2.3, "source_timestamp": stamp}
    if fault == "future":
        record["source_timestamp"] = OBSERVED_AT + timedelta(seconds=1)
    elif fault == "stale":
        record["source_timestamp"] = stamp - timedelta(days=1)
    elif fault == "price":
        record["price"] = 11.0
    elif fault == "missing":
        record["volume_ratio"] = np.nan
    elif fault == "unknown":
        record["code"] = "600001"
    elif fault == "boundary":
        record["price"] = 10.01
    elif fault == "above_boundary":
        record["price"] = 10.01001
    records = [record, record] if fault == "duplicate" else [record]
    fetcher = Mock(return_value=SimpleNamespace(frame=pd.DataFrame(records), observed_at=OBSERVED_AT))
    client = Mock()
    client.a_share_snapshot.return_value = ths_response([ths_snapshot_item()])
    data = MarketDataGateway(ths_client=client, snapshot_supplement_fetcher=fetcher,
                             clock=lambda: OBSERVED_AT).get_a_share_snapshot(["600000"])
    fetcher.assert_called_once_with(("600000",))
    if fault in (None, "boundary"):
        assert data.frame.loc[0, "name"] == "Stock"
        assert data.frame.loc[0, "turnover"] == 2.3
        assert data.frame.loc[0, "amount"] == 10000.0
        assert data.source.endswith("+tencent")
        assert data.frame.attrs["screening_complete_count"] == 1
    else:
        assert pd.isna(data.frame.loc[0, "name"])
        assert "snapshot_screening_fields_incomplete" in data.warnings
        assert data.frame.attrs["screening_complete_count"] == 0


def test_premarket_supplement_uses_last_completed_session_not_ths_response_time() -> None:
    observed = datetime(2026, 8, 20, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    prior_close = datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    client = Mock()
    client.a_share_snapshot.return_value = ThsApiResponse(
        {
            "timestamp": int(observed.timestamp() * 1000),
            "item": [ths_snapshot_item()],
        },
        None,
    )
    supplement = Mock(return_value=SimpleNamespace(
        frame=pd.DataFrame([{
            "code": "600000",
            "name": "Stock",
            "price": 10.0,
            "volume_ratio": 1.2,
            "turnover": 2.3,
            "source_timestamp": prior_close,
        }]),
        observed_at=observed,
    ))

    result = MarketDataGateway(
        ths_client=client,
        snapshot_supplement_fetcher=supplement,
        clock=lambda: observed,
    ).get_a_share_snapshot(["600000"])

    assert result.source_timestamp == prior_close
    assert result.frame.attrs["screening_complete_count"] == 1
    assert result.source.endswith("+tencent")


def test_premarket_current_day_quote_proves_prior_close_before_xshg_open() -> None:
    observed = datetime(2026, 8, 20, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    prior_close = datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    client = Mock()
    client.a_share_snapshot.return_value = ThsApiResponse(
        {
            "timestamp": int(observed.timestamp() * 1000),
            "item": [ths_snapshot_item()],
        },
        None,
    )
    supplement = Mock(return_value=SimpleNamespace(
        frame=pd.DataFrame([{
            "code": "600000",
            "name": "Stock",
            "price": 10.0,
            "volume_ratio": 1.2,
            "turnover": 2.3,
            "source_timestamp": observed,
        }]),
        observed_at=observed,
    ))

    result = MarketDataGateway(
        ths_client=client,
        snapshot_supplement_fetcher=supplement,
        clock=lambda: observed,
    ).get_a_share_snapshot(["600000"])

    assert result.source_timestamp == prior_close
    assert result.frame.attrs["screening_complete_count"] == 1
    assert result.source.endswith("+tencent")


def test_current_day_supplement_is_rejected_after_xshg_open() -> None:
    observed = datetime(2026, 8, 20, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    client = Mock()
    client.a_share_snapshot.return_value = ThsApiResponse(
        {
            "timestamp": int(observed.timestamp() * 1000),
            "item": [ths_snapshot_item()],
        },
        None,
    )
    supplement = Mock(return_value=SimpleNamespace(
        frame=pd.DataFrame([{
            "code": "600000",
            "name": "Stock",
            "price": 10.0,
            "volume_ratio": 1.2,
            "turnover": 2.3,
            "source_timestamp": observed,
        }]),
        observed_at=observed,
    ))

    result = MarketDataGateway(
        ths_client=client,
        snapshot_supplement_fetcher=supplement,
        clock=lambda: observed,
    ).get_a_share_snapshot(["600000"])

    assert result.frame.attrs["screening_complete_count"] == 0
    assert not result.source.endswith("+tencent")
    assert "snapshot_screening_fields_incomplete" in result.warnings


def test_ths_supplement_skips_unsupported_code_without_losing_supported_coverage() -> None:
    stamp = datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    fetcher = Mock(return_value=SimpleNamespace(
        frame=pd.DataFrame([{
            "code": "600000", "name": "Stock", "price": 10.0,
            "volume_ratio": 1.2, "turnover": 2.3, "source_timestamp": stamp,
        }]),
        observed_at=OBSERVED_AT,
    ))
    client = Mock()
    client.a_share_snapshot.return_value = ThsApiResponse(
        {
            "timestamp": int(stamp.timestamp() * 1000),
            "total": 2,
            "item": [ths_snapshot_item("600000"), ths_snapshot_item("302001")],
        },
        None,
    )

    data = MarketDataGateway(
        ths_client=client,
        snapshot_supplement_fetcher=fetcher,
        clock=lambda: OBSERVED_AT,
    ).get_a_share_snapshot()

    fetcher.assert_called_once_with(("600000",))
    assert data.frame.attrs["screening_complete_count"] == 1
    assert data.frame.set_index("code").loc["600000", "name"] == "Stock"
    assert pd.isna(data.frame.set_index("code").loc["302001", "name"])
    assert "snapshot_screening_fields_incomplete" in data.warnings


def test_ths_full_snapshot_preserves_oldest_page_timestamp() -> None:
    client = Mock()
    older = OBSERVED_AT - timedelta(hours=2)
    newer = OBSERVED_AT - timedelta(hours=1)
    client.a_share_snapshot.side_effect = [
        ThsApiResponse({"timestamp": int(older.timestamp() * 1000), "total": 2,
                        "item": [ths_snapshot_item("600000")]}, None),
        ThsApiResponse({"timestamp": int(newer.timestamp() * 1000), "total": 2,
                        "item": [ths_snapshot_item("600001")]}, None),
    ]
    data = MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT).get_a_share_snapshot()
    assert data.source_timestamp == older


def test_ths_snapshot_rejects_mixed_session_pages_even_before_market_close() -> None:
    yesterday = datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    today = datetime(2026, 8, 20, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    client = Mock()
    client.a_share_snapshot.side_effect = [
        ThsApiResponse({"timestamp": int(yesterday.timestamp() * 1000), "total": 2,
                        "item": [ths_snapshot_item("600000")]}, None),
        ThsApiResponse({"timestamp": int(today.timestamp() * 1000), "total": 2,
                        "item": [ths_snapshot_item("600001")]}, None),
    ]
    with pytest.raises(ValueError, match="THS snapshot pages have mixed sessions"):
        MarketDataGateway(ths_client=client, clock=lambda: today + timedelta(minutes=10)).get_a_share_snapshot()


def test_ths_full_snapshot_reads_all_pages_before_returning_data() -> None:
    client = Mock()
    timestamp_ms = int((OBSERVED_AT - timedelta(hours=1)).timestamp() * 1000)
    client.a_share_snapshot.side_effect = [
        ThsApiResponse({"timestamp": timestamp_ms, "total": 2, "item": [ths_snapshot_item("600000")]}, None),
        ThsApiResponse(
            {"timestamp": timestamp_ms, "total": 2, "item": [{**ths_snapshot_item("000001"), "thscode": "000001.SZ"}]},
            None,
        ),
    ]
    gateway = MarketDataGateway(ths_client=client, snapshot_fetcher=Mock(), clock=lambda: OBSERVED_AT)

    result = gateway.get_a_share_snapshot()

    assert result.frame["code"].tolist() == ["600000", "000001"]
    assert client.a_share_snapshot.call_args_list == [
        call(None, limit=1000, offset=0),
        call(None, limit=1000, offset=1),
    ]


@pytest.mark.parametrize("kind", ["snapshot", "daily", "hot_list", "index"])
def test_ths_validates_source_time_at_response_receipt(kind: str) -> None:
    client = Mock()
    source_time = OBSERVED_AT + timedelta(seconds=1)
    received_at = OBSERVED_AT + timedelta(seconds=2)
    response = ths_response(
        ths_bar_items() if kind in {"daily", "index"} else [ths_snapshot_item()],
        int(source_time.timestamp() * 1000),
    )
    client.a_share_snapshot.return_value = response
    client.a_share_historical.return_value = response
    client.hot_stock_list.return_value = response
    client.index_historical.return_value = response
    gateway = MarketDataGateway(ths_client=client, clock=Mock(side_effect=[OBSERVED_AT, received_at]))
    if kind == "snapshot":
        result = gateway.get_a_share_snapshot(["600000"])
    elif kind == "daily":
        result = gateway.get_daily_bars("600000", SESSION)
    elif kind == "index":
        result = gateway.get_ths_index_bars("886042.TI", SESSION)
    else:
        result = gateway.get_ths_hot_stock_list()
    assert result.observed_at == received_at
    assert result.source_timestamp == source_time


def test_ths_pagination_rejects_future_page_before_requesting_next_page() -> None:
    client = Mock()
    response = ThsApiResponse({
        "timestamp": int((OBSERVED_AT + timedelta(seconds=3)).timestamp() * 1000),
        "total": 2,
        "item": [ths_snapshot_item()],
    }, None)
    client.a_share_snapshot.return_value = response
    gateway = MarketDataGateway(
        ths_client=client,
        clock=Mock(side_effect=[OBSERVED_AT, OBSERVED_AT + timedelta(seconds=1)]),
    )
    with pytest.raises(ValueError, match="THS source timestamp is in the future"):
        gateway.get_a_share_snapshot()
    assert client.a_share_snapshot.call_count == 1


@pytest.mark.parametrize("received_at", [OBSERVED_AT - timedelta(seconds=1), OBSERVED_AT.replace(tzinfo=None)])
def test_ths_rejects_invalid_receipt_clock(received_at: datetime) -> None:
    client = Mock()
    client.a_share_snapshot.return_value = ths_response([ths_snapshot_item()])
    gateway = MarketDataGateway(ths_client=client, clock=Mock(side_effect=[OBSERVED_AT, received_at]))
    with pytest.raises(ValueError, match="provider acquisition clock invalid"):
        gateway.get_a_share_snapshot(["600000"])


def test_ths_snapshot_recoverable_failure_uses_existing_source_with_warning(caplog) -> None:
    client = Mock()
    client.a_share_snapshot.side_effect = ThsNetworkError("https://private.example/?token=private-key")
    fallback = Mock(return_value=pd.DataFrame({"代码": ["000001"], "名称": ["A"], "最新价": [10]}))
    gateway = MarketDataGateway(ths_client=client, snapshot_fetcher=fallback, clock=lambda: OBSERVED_AT)

    result = gateway.get_a_share_snapshot()

    fallback.assert_called_once_with()
    assert result.source == "akshare.stock_zh_a_spot_em"
    assert result.warnings == ("THS unavailable; existing snapshot source used", "snapshot source timestamp unavailable")
    assert "ths_network_failed" in caplog.text
    assert "private-key" not in caplog.text
    assert "private.example" not in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        ths_response([{"ticker": "600000"}]),
        ths_response([{**ths_snapshot_item(), "last_price": 0}]),
        ths_response([{**ths_snapshot_item(), "turnover": -1}]),
    ],
)
def test_ths_snapshot_quality_failure_never_falls_back(response: ThsApiResponse) -> None:
    client = Mock()
    client.a_share_snapshot.return_value = response
    fallback = Mock(return_value=pd.DataFrame({"代码": ["000001"], "最新价": [10]}))
    gateway = MarketDataGateway(ths_client=client, snapshot_fetcher=fallback, clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match="^THS snapshot"):
        gateway.get_a_share_snapshot()

    fallback.assert_not_called()


def test_ths_malformed_response_never_falls_back() -> None:
    client = Mock()
    client.a_share_snapshot.side_effect = ThsResponseError("THS API returned an invalid success payload")
    fallback = Mock()
    gateway = MarketDataGateway(ths_client=client, snapshot_fetcher=fallback, clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match="^THS snapshot data invalid$"):
        gateway.get_a_share_snapshot()

    fallback.assert_not_called()


def test_ths_daily_bars_are_preferred_and_normalized() -> None:
    client = Mock()
    client.a_share_historical.return_value = ths_response(ths_bar_items())
    fallback = Mock()
    gateway = MarketDataGateway(ths_client=client, daily_fetcher=fallback, clock=lambda: OBSERVED_AT)

    result = gateway.get_daily_bars("600000", expected_session=SESSION, days=160)

    assert client.a_share_historical.call_args.args[0] == "600000.SH"
    assert client.a_share_historical.call_args.kwargs["start_ms"] < client.a_share_historical.call_args.kwargs["end_ms"]
    fallback.assert_not_called()
    assert result.source == "ths.fuyao.a_share_historical"
    assert result.frame.iloc[-1]["date"].date() == SESSION


def test_ths_daily_bars_recoverable_failure_uses_existing_source_with_warning() -> None:
    client = Mock()
    client.a_share_historical.side_effect = ThsNetworkError("THS network request failed")
    fallback = Mock(return_value=(daily_bars(), "fallback"))
    gateway = MarketDataGateway(ths_client=client, daily_fetcher=fallback, clock=lambda: OBSERVED_AT)

    result = gateway.get_daily_bars("600000", expected_session=SESSION)

    fallback.assert_called_once_with("600000", days=160)
    assert result.source == "fallback"
    assert result.warnings == ("THS unavailable; existing daily source used",)


def test_ths_stale_daily_bars_use_existing_source_with_warning() -> None:
    client = Mock()
    client.a_share_historical.return_value = ths_response(ths_bar_items(61)[:-1])
    fallback = Mock(return_value=(daily_bars(), "fallback"))
    gateway = MarketDataGateway(ths_client=client, daily_fetcher=fallback, clock=lambda: OBSERVED_AT)

    result = gateway.get_daily_bars("600000", expected_session=SESSION)

    fallback.assert_called_once_with("600000", days=160)
    assert result.source == "fallback"
    assert result.warnings == ("THS daily bars stale; existing daily source used",)


def test_ths_daily_bars_quality_failure_never_falls_back() -> None:
    client = Mock()
    client.a_share_historical.return_value = ths_response([{**ths_bar_items()[0], "close_price": 0}])
    fallback = Mock()
    gateway = MarketDataGateway(ths_client=client, daily_fetcher=fallback, clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match="^daily bars"):
        gateway.get_daily_bars("600000", expected_session=SESSION)

    fallback.assert_not_called()


def test_ths_specialty_helpers_return_market_datasets_without_report_integration() -> None:
    client = Mock()
    client.financial_indicators.return_value = ThsApiResponse(
        {
            "thscode": "600000.SH",
            "report": "2026-1",
            "abilities": [{"ability": "growth", "indicators": [{"index_id": "profit_yoy", "value": "1.2"}]}],
        },
        None,
    )
    client.hot_stock_list.return_value = ths_response([{"rank": 1, "thscode": "600000.SH"}])
    client.skyrocket_list.return_value = ths_response([{"rank": 1, "thscode": "600000.SH"}])
    client.ths_index_catalog.return_value = ths_response([{"thscode": "886001.TI"}])
    client.ths_index_constituents.return_value = ths_response([{"thscode": "600000.SH"}])
    client.index_snapshot.return_value = ths_response([{"thscode": "000001.SH", "last_price": 1.0}])
    client.index_historical.return_value = ths_response(ths_bar_items())
    gateway = MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT)

    datasets = [
        gateway.get_ths_financial_indicators("600000", "2026-1"),
        gateway.get_ths_hot_stock_list(),
        gateway.get_ths_skyrocket_list(),
        gateway.get_ths_index_catalog(),
        gateway.get_ths_index_constituents("886001.TI"),
        gateway.get_ths_index_snapshot(["000001.SH"]),
        gateway.get_ths_index_bars("000001.SH", SESSION),
    ]

    assert [dataset.source for dataset in datasets] == [
        "ths.fuyao.financial_indicators",
        "ths.fuyao.hot_stock_list",
        "ths.fuyao.skyrocket_list",
        "ths.fuyao.index_catalog",
        "ths.fuyao.index_constituents",
        "ths.fuyao.index_snapshot",
        "ths.fuyao.index_historical",
    ]
    assert datasets[0].frame.to_dict("records") == [
        {"thscode": "600000.SH", "report": "2026-1", "ability": "growth", "index_id": "profit_yoy", "value": "1.2"}
    ]


def test_normalize_a_share_snapshot_maps_chinese_columns_and_numeric_values() -> None:
    raw = pd.DataFrame(
        {
            "代码": ["000001"],
            "名称": ["平安银行"],
            "最新价": ["12.34"],
            "涨跌幅": ["1.25%"],
            "量比": ["1.8"],
            "换手率": ["2.2%"],
            "成交额": ["1,234.5"],
            "成交量": ["12,345"],
            "总市值": ["9,876.5"],
        }
    )

    result = normalize_a_share_snapshot(raw)

    assert list(result.columns) == [
        "code",
        "name",
        "price",
        "change_pct",
        "volume_ratio",
        "turnover",
        "amount",
        "volume",
        "total_mv",
    ]
    assert result.iloc[0].to_dict() == {
        "code": "000001",
        "name": "平安银行",
        "price": 12.34,
        "change_pct": 1.25,
        "volume_ratio": 1.8,
        "turnover": 2.2,
        "amount": 1234.5,
        "volume": 12345.0,
        "total_mv": 9876.5,
    }


def test_normalize_a_share_snapshot_preserves_leading_zero_and_does_not_mutate_input() -> None:
    raw = pd.DataFrame({"股票代码": ["000001", "bad"], "股票简称": ["A", "B"], "现价": [1, 2]})
    before = raw.copy(deep=True)

    result = normalize_a_share_snapshot(raw)

    pd.testing.assert_frame_equal(raw, before)
    assert result["code"].tolist() == ["000001"]
    assert pd.isna(result.iloc[0]["change_pct"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[:59], "daily bars insufficient"),
        (lambda frame: pd.concat([frame.iloc[:-1], frame.iloc[[-2]]], ignore_index=True), "daily bars duplicate dates"),
        (lambda frame: frame.assign(Date=[*frame["Date"].iloc[:-1], "not-a-date"]), "daily bars invalid dates"),
        (lambda frame: frame.assign(Date=[*frame["Date"].iloc[:-1], "2026-08-20"]), "daily bars future dates"),
        (lambda frame: frame.assign(Date=pd.bdate_range(end="2026-08-18", periods=len(frame))), "daily bars stale"),
    ],
)
def test_validate_daily_bars_rejects_bad_date_series(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        validate_daily_bars(mutate(daily_bars()), SESSION)


def test_validate_daily_bars_rejects_non_monotonic_dates() -> None:
    frame = daily_bars()
    frame.loc[[10, 11], "Date"] = frame.loc[[11, 10], "Date"].to_numpy()

    with pytest.raises(ValueError, match="^daily bars non-monotonic dates$"):
        validate_daily_bars(frame, SESSION)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Open", np.nan, "daily bars invalid ohlc"),
        ("High", np.inf, "daily bars invalid ohlc"),
        ("Low", 0, "daily bars invalid ohlc"),
        ("Close", -1, "daily bars invalid ohlc"),
        ("Volume", -1, "daily bars invalid volume"),
        ("Volume", np.inf, "daily bars invalid volume"),
    ],
)
def test_validate_daily_bars_rejects_invalid_ohlcv(column: str, value: float, message: str) -> None:
    frame = daily_bars()
    frame.loc[10, column] = value

    with pytest.raises(ValueError, match=f"^{message}$"):
        validate_daily_bars(frame, SESSION)


@pytest.mark.parametrize(
    "updates",
    [
        {"High": 8.0, "Low": 9.0},
        {"High": 10.0, "Open": 11.0},
        {"High": 10.0, "Close": 11.0},
        {"Low": 12.0, "Open": 11.0},
        {"Low": 12.0, "Close": 11.0},
    ],
)
def test_validate_daily_bars_rejects_impossible_ohlc(updates: dict[str, float]) -> None:
    frame = daily_bars()
    for column, value in updates.items():
        frame.loc[10, column] = value

    with pytest.raises(ValueError, match="^daily bars invalid ohlc$"):
        validate_daily_bars(frame, SESSION)


def test_validate_daily_bars_normalizes_chinese_columns_and_returns_new_ascending_frame() -> None:
    original = daily_bars().rename(
        columns={
            "Date": "日期",
            "Open": "开盘",
            "High": "最高",
            "Low": "最低",
            "Close": "收盘",
            "Volume": "成交量",
        }
    )
    original = original.iloc[::-1]

    result = validate_daily_bars(original, SESSION)

    assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert result["date"].is_monotonic_increasing
    assert result is not original
    assert list(original.columns)[0] == "日期"


def test_get_a_share_snapshot_calls_injected_provider_once() -> None:
    provider = Mock(return_value=pd.DataFrame({"代码": ["000001"], "名称": ["A"], "最新价": [10]}))
    gateway = MarketDataGateway(snapshot_fetcher=provider, clock=lambda: OBSERVED_AT)

    result = gateway.get_a_share_snapshot()

    provider.assert_called_once_with()
    assert result.source == "akshare.stock_zh_a_spot_em"
    assert result.observed_at == OBSERVED_AT
    assert result.frame["code"].tolist() == ["000001"]
    assert result.source_timestamp is None


def test_get_a_share_snapshot_carries_authoritative_provider_timestamp() -> None:
    raw = pd.DataFrame({"代码": ["000001"], "名称": ["A"], "最新价": [10]})
    raw.attrs["source_timestamp"] = "2026-08-19T15:00:00+08:00"
    gateway = MarketDataGateway(snapshot_fetcher=lambda: raw, clock=lambda: OBSERVED_AT)

    result = gateway.get_a_share_snapshot()

    assert result.source_timestamp == datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_get_a_share_snapshot_rejects_untrustworthy_provider_timestamp() -> None:
    raw = pd.DataFrame({"代码": ["000001"], "名称": ["A"], "最新价": [10]})
    raw.attrs["source_timestamp"] = "not-a-time"
    gateway = MarketDataGateway(snapshot_fetcher=lambda: raw, clock=lambda: OBSERVED_AT)

    result = gateway.get_a_share_snapshot()

    assert result.source_timestamp is None
    assert "snapshot source timestamp unavailable" in result.warnings


def test_get_a_share_snapshot_rejects_empty_provider_response() -> None:
    gateway = MarketDataGateway(snapshot_fetcher=lambda: pd.DataFrame())

    with pytest.raises(ValueError, match="^snapshot provider returned empty data$"):
        gateway.get_a_share_snapshot()


def test_get_a_share_snapshot_wraps_provider_failure() -> None:
    failure = RuntimeError("https://feed.invalid/?token=secret")
    gateway = MarketDataGateway(snapshot_fetcher=Mock(side_effect=failure))

    with pytest.raises(ValueError, match="^snapshot provider unavailable$") as caught:
        gateway.get_a_share_snapshot()

    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert "https://feed.invalid" not in formatted_traceback(caught)


def test_get_daily_bars_calls_injected_manager_and_preserves_source() -> None:
    fetcher = Mock(return_value=(daily_bars(), "test-source"))
    gateway = MarketDataGateway(daily_fetcher=fetcher, clock=lambda: OBSERVED_AT)

    result = gateway.get_daily_bars("000001", expected_session=SESSION, days=160)

    fetcher.assert_called_once_with("000001", days=160)
    assert result.source == "test-source"
    assert result.frame.iloc[-1]["date"].date() == SESSION


def test_get_daily_bars_rejects_empty_provider_response() -> None:
    gateway = MarketDataGateway(daily_fetcher=lambda code, *, days: (pd.DataFrame(), "empty"))

    with pytest.raises(ValueError, match="^daily provider returned empty data$"):
        gateway.get_daily_bars("000001", expected_session=SESSION)


def test_get_daily_bars_wraps_provider_failure() -> None:
    failure = RuntimeError("https://feed.invalid/?token=secret")
    gateway = MarketDataGateway(daily_fetcher=Mock(side_effect=failure))

    with pytest.raises(ValueError, match="^daily provider unavailable$") as caught:
        gateway.get_daily_bars("000001", expected_session=SESSION)

    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert "https://feed.invalid" not in formatted_traceback(caught)


def test_get_leading_sector_codes_calls_top_boards_and_returns_partial_map() -> None:
    boards = Mock(
        return_value=pd.DataFrame(
            {"板块名称": ["弱板块", "强板块", "次强板块"], "涨跌幅": [1.0, 5.0, 3.0]}
        )
    )
    members = Mock(
        side_effect=lambda name: {
            "强板块": pd.DataFrame({"代码": ["000001", "600000"]}),
            "次强板块": pd.DataFrame({"代码": ["300001"]}),
        }[name]
    )
    gateway = MarketDataGateway(sector_names_fetcher=boards, sector_members_fetcher=members, clock=lambda: OBSERVED_AT)

    result = gateway.get_leading_sector_codes(limit=2)

    boards.assert_called_once_with()
    assert members.call_args_list == [call("强板块"), call("次强板块")]
    assert result.frame.set_index("code")["sector"].to_dict() == {
        "000001": "强板块",
        "600000": "强板块",
        "300001": "次强板块",
    }
    assert result.source == "akshare.industry_boards"


def test_get_leading_sector_codes_warns_without_leaking_partial_failure() -> None:
    boards = lambda: pd.DataFrame({"板块名称": ["强板块", "次强板块"], "涨跌幅": [5.0, 3.0]})

    def members(name: str) -> pd.DataFrame:
        if name == "强板块":
            raise RuntimeError("secret token and provider details")
        return pd.DataFrame({"代码": ["300001"]})

    gateway = MarketDataGateway(sector_names_fetcher=boards, sector_members_fetcher=members, clock=lambda: OBSERVED_AT)

    result = gateway.get_leading_sector_codes(limit=2)

    assert result.frame.to_dict("records") == [{"code": "300001", "sector": "次强板块"}]
    assert result.warnings == ("sector constituents unavailable: 强板块",)
    assert "secret" not in " ".join(result.warnings)


def test_get_leading_sector_codes_wraps_sector_list_failure() -> None:
    failure = RuntimeError("https://feed.invalid/?token=secret")
    gateway = MarketDataGateway(
        sector_names_fetcher=Mock(side_effect=failure),
        sector_members_fetcher=Mock(),
    )

    with pytest.raises(ValueError, match="^sector provider unavailable$") as caught:
        gateway.get_leading_sector_codes()

    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert "https://feed.invalid" not in formatted_traceback(caught)


def test_get_limit_counts_uses_completed_session_pools() -> None:
    up = Mock(return_value=pd.DataFrame({"代码": ["600000", "000001"]}))
    down = Mock(return_value=pd.DataFrame({"代码": ["300001"]}))
    gateway = MarketDataGateway(
        limit_up_fetcher=up,
        limit_down_fetcher=down,
        clock=lambda: OBSERVED_AT,
    )

    result = gateway.get_limit_counts(SESSION)

    up.assert_called_once_with(date="20260819")
    down.assert_called_once_with(date="20260819")
    assert result.frame.to_dict("records") == [{"limit_up_count": 2, "limit_down_count": 1}]
    assert result.source == "akshare.eastmoney_limit_pools"
    assert result.source_timestamp == datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_get_limit_counts_rejects_incomplete_session() -> None:
    before_close = datetime(2026, 8, 19, 14, 59, tzinfo=ZoneInfo("Asia/Shanghai"))
    gateway = MarketDataGateway(
        limit_up_fetcher=lambda **kwargs: pd.DataFrame(),
        limit_down_fetcher=lambda **kwargs: pd.DataFrame(),
        clock=lambda: before_close,
    )

    with pytest.raises(ValueError, match="^limit pool session incomplete$"):
        gateway.get_limit_counts(SESSION)


def test_get_sector_snapshot_normalizes_industry_fields_and_preserves_raw_input() -> None:
    raw = pd.DataFrame(
        {
            "板块名称": ["  半导体  "],
            "涨跌幅": ["2.5%"],
            "上涨家数": ["10"],
            "下跌家数": ["2"],
            "换手率": ["3.4%"],
            "成交额": ["1,234"],
            "领涨股票": ["  芯片股  "],
            "领涨股票代码": ["000001"],
            "领涨股票-涨跌幅": ["5.6%"],
        }
    )
    raw.attrs["source_timestamp"] = "2026-08-19 15:00:00"
    original = raw.copy(deep=True)
    original.attrs = raw.attrs.copy()

    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert list(result.frame.columns) == [
        "sector_type", "name", "change_pct", "advance_count", "decline_count", "turnover_rate", "amount",
        "leader_name", "leader_code", "leader_change_pct",
    ]
    assert result.frame.to_dict("records") == [{
        "sector_type": "industry", "name": "半导体", "change_pct": 2.5, "advance_count": 10.0,
        "decline_count": 2.0, "turnover_rate": 3.4, "amount": 1234.0, "leader_name": "芯片股",
        "leader_code": "000001", "leader_change_pct": 5.6,
    }]
    assert result.source == "akshare.eastmoney_industry_boards"
    assert result.source_timestamp == datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    pd.testing.assert_frame_equal(raw, original)
    assert raw.attrs == original.attrs


def test_get_sector_snapshot_normalizes_concept_english_aliases_and_optional_na() -> None:
    raw = pd.DataFrame({"sector": ["AI"], "change_pct": ["1.2"], "leader_code": [" "], "amount": ["bad"]})

    result = MarketDataGateway(concept_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("concept")

    assert result.frame.loc[0, "sector_type"] == "concept"
    assert result.frame.loc[0, "name"] == "AI"
    assert result.frame.loc[0, "change_pct"] == 1.2
    for column in ("advance_count", "decline_count", "turnover_rate", "amount", "leader_name", "leader_code", "leader_change_pct"):
        assert pd.isna(result.frame.loc[0, column])
    assert result.warnings == ("snapshot source timestamp unavailable",)


def test_get_sector_snapshot_normalizes_leader_codes() -> None:
    raw = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "change_pct": [1, 2, 3, 4],
            "leader_code": [1, 600000.0, "invalid", True],
        }
    )

    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.frame["leader_code"].tolist()[:2] == ["000001", "600000"]
    assert result.frame["leader_code"].isna().tolist() == [False, False, True, True]


def test_get_sector_snapshot_has_deterministic_nullable_dtypes_across_sector_types() -> None:
    industry_raw = pd.DataFrame(
        {"name": ["Industry"], "change_pct": [1], "advance_count": [2], "amount": [100]}
    )
    concept_raw = pd.DataFrame(
        {"name": ["Concept"], "change_pct": [2.5], "decline_count": [1], "turnover_rate": [3]}
    )
    gateway = MarketDataGateway(
        industry_sector_fetcher=lambda: industry_raw,
        concept_sector_fetcher=lambda: concept_raw,
        clock=lambda: OBSERVED_AT,
    )

    industry = gateway.get_sector_snapshot("industry").frame
    concept = gateway.get_sector_snapshot("concept").frame
    expected_dtypes = {
        "sector_type": "string",
        "name": "string",
        "change_pct": "Float64",
        "advance_count": "Int64",
        "decline_count": "Int64",
        "turnover_rate": "Float64",
        "amount": "Float64",
        "leader_name": "string",
        "leader_code": "string",
        "leader_change_pct": "Float64",
    }
    assert industry.dtypes.astype(str).to_dict() == expected_dtypes
    assert concept.dtypes.astype(str).to_dict() == expected_dtypes

    combined = pd.concat([industry, concept], ignore_index=True)
    assert combined.dtypes.astype(str).to_dict() == expected_dtypes
    assert combined["change_pct"].sum() == 3.5
    assert combined["advance_count"].sum() == 2


def test_get_sector_snapshot_uses_correct_default_akshare_fetchers(monkeypatch) -> None:
    industry = Mock(return_value=pd.DataFrame({"name": ["Industry"], "change_pct": [1]}))
    concept = Mock(return_value=pd.DataFrame({"name": ["Concept"], "change_pct": [2]}))
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_board_industry_name_em=industry,
        stock_board_concept_name_em=concept,
    ))
    gateway = MarketDataGateway(clock=lambda: OBSERVED_AT)

    assert gateway.get_sector_snapshot("industry").frame.loc[0, "name"] == "Industry"
    assert gateway.get_sector_snapshot("concept").frame.loc[0, "name"] == "Concept"
    industry.assert_called_once_with()
    concept.assert_called_once_with()


@pytest.mark.parametrize(
    ("sector_type", "tag"),
    [("industry", "industry"), ("concept", "cn_concept")],
)
def test_get_sector_snapshot_uses_ths_catalog_and_batched_index_quotes(sector_type, tag) -> None:
    catalog_rows = [
        {"thscode": f"88{index:04d}.TI", "name": f"Sector {index}"}
        for index in range(205)
    ]
    client = Mock()
    client.ths_index_catalog.return_value = ths_response(catalog_rows)

    def snapshot(codes):
        return ths_response([
            {
                "thscode": code,
                "price_change_ratio_pct": index / 10,
                "turnover": 1_000 + index,
            }
            for index, code in enumerate(codes)
        ])

    client.index_snapshot.side_effect = snapshot
    result = MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT).get_sector_snapshot(sector_type)

    assert client.ths_index_catalog.call_args.args[0].value == tag
    assert [len(item.args[0]) for item in client.index_snapshot.call_args_list] == [100, 100, 5]
    assert len(result.frame) == 205
    assert result.frame["sector_type"].unique().tolist() == [sector_type]
    assert result.frame.loc[0, "name"] == "Sector 0"
    assert result.frame.loc[0, "change_pct"] == 0
    assert result.frame.loc[0, "amount"] == 1_000
    assert result.source == "ths.fuyao.index_snapshot"
    assert result.source_timestamp == datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_ths_sector_snapshot_derives_breadth_and_leader_from_cached_full_snapshot() -> None:
    stamp = datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    client = Mock()
    client.a_share_snapshot.return_value = ThsApiResponse(
        {
            "timestamp": int(stamp.timestamp() * 1000),
            "total": 2,
            "item": [
                ths_snapshot_item("600000"),
                {**ths_snapshot_item("600001"), "price_change_ratio_pct": -2.0},
            ],
        },
        None,
    )
    client.ths_index_catalog.return_value = ths_response([
        {"thscode": "881001.TI", "name": "Industry A"},
    ])
    client.index_snapshot.return_value = ths_response([
        {"thscode": "881001.TI", "price_change_ratio_pct": 1.2, "turnover": 10_000},
    ])
    client.ths_index_constituents.return_value = ths_response([
        {"thscode": "600000.SH"}, {"thscode": "600001.SH"},
    ])
    supplement = Mock(return_value=SimpleNamespace(
        frame=pd.DataFrame([
            {
                "code": "600000", "name": "Leader", "price": 10.0,
                "volume_ratio": 1.2, "turnover": 2.3, "source_timestamp": stamp,
            },
            {
                "code": "600001", "name": "Decliner", "price": 10.0,
                "volume_ratio": 1.1, "turnover": 2.0, "source_timestamp": stamp,
            },
        ]),
        observed_at=OBSERVED_AT,
    ))
    gateway = MarketDataGateway(
        ths_client=client,
        snapshot_supplement_fetcher=supplement,
        clock=lambda: OBSERVED_AT,
    )

    gateway.get_a_share_snapshot()
    result = gateway.get_sector_snapshot("industry")

    row = result.frame.iloc[0]
    assert row["advance_count"] == 1
    assert row["decline_count"] == 1
    assert row["leader_name"] == "Leader"
    assert row["leader_code"] == "600000"
    assert row["leader_change_pct"] == 1.2
    leading = gateway.get_leading_sector_codes()
    assert leading.source == "ths.fuyao.index_constituents"
    assert leading.frame.to_dict("records") == [
        {"code": "600000", "sector": "Industry A"},
        {"code": "600001", "sector": "Industry A"},
    ]
    client.ths_index_constituents.assert_called_once_with("881001.TI")


def test_get_sector_snapshot_marks_partial_ths_catalog_coverage() -> None:
    client = Mock()
    client.ths_index_catalog.return_value = ths_response([
        {"thscode": "881001.TI", "name": "Industry A"},
        {"thscode": "881002.TI", "name": "Industry B"},
    ])
    client.index_snapshot.return_value = ths_response([
        {"thscode": "881001.TI", "price_change_ratio_pct": 1.2, "turnover": 10_000},
    ])

    result = MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.frame["name"].tolist() == ["Industry A"]
    assert result.warnings == ("THS sector snapshot rows missing: 1",)


def test_get_sector_snapshot_falls_back_to_akshare_when_ths_is_unavailable(monkeypatch) -> None:
    client = Mock()
    client.ths_index_catalog.side_effect = ThsNetworkError()
    industry = Mock(return_value=pd.DataFrame({"name": ["Industry"], "change_pct": [1]}))
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_board_industry_name_em=industry))

    result = MarketDataGateway(ths_client=client, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.source == "akshare.eastmoney_industry_boards"
    assert result.frame["name"].tolist() == ["Industry"]
    industry.assert_called_once_with()


def test_get_sector_snapshot_rejects_invalid_type_and_isolates_provider_failures() -> None:
    industry = Mock(side_effect=RuntimeError("https://feed.invalid/?token=secret"))
    concept = Mock(return_value=pd.DataFrame({"name": ["Concept"], "change_pct": [1]}))
    gateway = MarketDataGateway(industry_sector_fetcher=industry, concept_sector_fetcher=concept, clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match="^sector type invalid$"):
        gateway.get_sector_snapshot("other")
    with pytest.raises(ValueError, match="^industry sector provider unavailable$") as caught:
        gateway.get_sector_snapshot("industry")
    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert gateway.get_sector_snapshot("concept").frame.loc[0, "name"] == "Concept"


@pytest.mark.parametrize("raw", [None, pd.DataFrame(), pd.DataFrame({"name": ["A"]}), pd.DataFrame({"change_pct": [1]})])
def test_get_sector_snapshot_rejects_empty_and_missing_required_data(raw) -> None:
    gateway = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match="^industry sector provider returned (empty|invalid) data$"):
        gateway.get_sector_snapshot("industry")


def test_get_sector_snapshot_excludes_invalid_change_rows_with_count_and_rejects_all_invalid() -> None:
    raw = pd.DataFrame({"name": ["Valid", "Blank", "Bad", "Infinite"], "change_pct": [1, 2, "bad", np.inf]})
    raw.loc[1, "name"] = " "
    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.frame["name"].tolist() == ["Valid"]
    assert result.warnings == ("sector snapshot rows excluded: 3", "snapshot source timestamp unavailable")
    all_invalid = pd.DataFrame({"name": [" ", "Bad"], "change_pct": [1, "bad"]})
    with pytest.raises(ValueError, match="^industry sector provider returned invalid data$"):
        MarketDataGateway(industry_sector_fetcher=lambda: all_invalid, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")


@pytest.mark.parametrize(
    "column, value",
    [
        ("advance_count", -1), ("advance_count", 1.5), ("decline_count", -1), ("decline_count", 1.5),
        ("turnover_rate", -0.1), ("amount", -1), ("turnover_rate", np.inf), ("amount", np.inf),
    ],
)
def test_get_sector_snapshot_rejects_invalid_optional_finite_values(column: str, value: object) -> None:
    raw = pd.DataFrame({"name": ["A"], "change_pct": [1], column: [value]})
    with pytest.raises(ValueError, match="^industry sector provider returned invalid data$"):
        MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")


def test_get_sector_snapshot_rejects_duplicate_names() -> None:
    raw = pd.DataFrame({"name": [" A ", "A"], "change_pct": [1, 2]})
    with pytest.raises(ValueError, match="^industry sector provider returned invalid data$"):
        MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")


def test_get_sector_snapshot_sanitizes_duplicate_columns() -> None:
    raw = pd.DataFrame([["A", "B", 1]], columns=["name", "name", "change_pct"])

    with pytest.raises(ValueError, match="^industry sector provider returned invalid data$") as caught:
        MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert caught.value.__cause__ is None


def test_get_sector_snapshot_sanitizes_object_conversion_failure() -> None:
    class UnsafeText:
        def __str__(self) -> str:
            raise RuntimeError("https://feed.invalid/?token=secret")

    raw = pd.DataFrame({"name": [UnsafeText()], "change_pct": [1]})

    with pytest.raises(ValueError, match="^industry sector provider returned invalid data$") as caught:
        MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert "feed.invalid" not in formatted_traceback(caught)


def test_get_sector_snapshot_handles_source_timestamps_and_postmarket_freshness_boundary() -> None:
    raw = pd.DataFrame({"name": ["A"], "change_pct": [1]})
    raw.attrs["quote_timestamp"] = "2026-08-19 15:00:00"
    boundary = datetime(2026, 8, 19, 19, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert (
        MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: boundary)
        .get_sector_snapshot("industry").source_timestamp
        == datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    with pytest.raises(ValueError, match="^industry sector snapshot stale$"):
        MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: boundary + timedelta(seconds=1)).get_sector_snapshot("industry")
    raw.attrs["data_timestamp"] = "2026-08-19 19:01:00+08:00"
    raw.attrs.pop("quote_timestamp")
    with pytest.raises(ValueError, match="^industry sector source timestamp is in the future$"):
        MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: boundary).get_sector_snapshot("industry")


def test_get_sector_snapshot_freshness_uses_equivalent_instants_across_timezones() -> None:
    raw = pd.DataFrame({"name": ["A"], "change_pct": [1]})
    raw.attrs["source_timestamp"] = "2026-08-19T15:00:00+08:00"
    boundary_utc = datetime(2026, 8, 19, 11, 0, tzinfo=timezone.utc)

    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: boundary_utc).get_sector_snapshot("industry")

    assert result.observed_at == boundary_utc
    with pytest.raises(ValueError, match="^industry sector snapshot stale$"):
        MarketDataGateway(
            industry_sector_fetcher=lambda: raw,
            clock=lambda: boundary_utc + timedelta(seconds=1),
        ).get_sector_snapshot("industry")


def test_get_sector_snapshot_marks_invalid_source_timestamp_unavailable() -> None:
    raw = pd.DataFrame({"name": ["A"], "change_pct": [1]})
    raw.attrs["source_timestamp"] = "not-a-timestamp"

    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.source_timestamp is None
    assert result.warnings == ("snapshot source timestamp unavailable",)


@pytest.mark.parametrize(
    "value",
    [
        pd.Series(["2026-08-19T15:00:00+08:00"]),
        np.array(["2026-08-19T15:00:00+08:00", "2026-08-19T15:01:00+08:00"]),
        ["2026-08-19T15:00:00+08:00"],
    ],
)
def test_get_sector_snapshot_marks_non_scalar_source_timestamp_unavailable(value: object) -> None:
    raw = pd.DataFrame({"name": ["A"], "change_pct": [1]})
    raw.attrs["source_timestamp"] = value

    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.source_timestamp is None
    assert result.warnings == ("snapshot source timestamp unavailable",)


def test_get_sector_snapshot_marks_pd_na_source_timestamp_unavailable() -> None:
    raw = pd.DataFrame({"name": ["A"], "change_pct": [1]})
    raw.attrs["source_timestamp"] = pd.NA

    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.source_timestamp is None
    assert result.warnings == ("snapshot source timestamp unavailable",)


def test_get_sector_snapshot_skips_missing_higher_priority_timestamp_alias() -> None:
    raw = pd.DataFrame({"name": ["A"], "change_pct": [1]})
    raw.attrs["source_timestamp"] = pd.NA
    raw.attrs["quote_timestamp"] = "2026-08-19T15:00:00+08:00"

    result = MarketDataGateway(industry_sector_fetcher=lambda: raw, clock=lambda: OBSERVED_AT).get_sector_snapshot("industry")

    assert result.source_timestamp == datetime(2026, 8, 19, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert result.warnings == ()


@pytest.mark.parametrize("sector_type", [[], {}, set()])
def test_get_sector_snapshot_rejects_unhashable_sector_type(sector_type: object) -> None:
    with pytest.raises(ValueError, match="^sector type invalid$"):
        MarketDataGateway().get_sector_snapshot(sector_type)


def global_download_frame(*, multi_index: bool) -> pd.DataFrame:
    symbols = ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F"]
    dates = pd.to_datetime(["2026-08-17", "2026-08-18"])
    closes = {symbol: [100.0, 102.0] for symbol in symbols}
    if not multi_index:
        return pd.DataFrame(closes, index=dates)
    columns = pd.MultiIndex.from_product([["Close"], symbols])
    return pd.DataFrame(np.array(list(closes.values())).T, index=dates, columns=columns)


@pytest.mark.parametrize("multi_index", [False, True])
def test_get_global_snapshot_supports_simple_and_multiindex_closes(multi_index: bool) -> None:
    download = Mock(return_value=global_download_frame(multi_index=multi_index))
    gateway = MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT)

    result = gateway.get_global_snapshot()

    download.assert_called_once_with(
        ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F", "CNY=X"],
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        timeout=15,
    )
    assert result.frame["symbol"].tolist() == ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F"]
    assert result.frame["change_pct"].tolist() == pytest.approx([2.0] * 6)
    assert result.observed_at == OBSERVED_AT


def test_get_global_snapshot_includes_optional_usd_cny_when_available() -> None:
    raw = global_download_frame(multi_index=False)
    raw["CNY=X"] = [7.10, 7.20]

    result = MarketDataGateway(
        yfinance_download=Mock(return_value=raw), clock=lambda: OBSERVED_AT,
    ).get_global_snapshot()

    fx = result.frame.loc[result.frame["symbol"] == "CNY=X"].iloc[0]
    assert fx["close"] == pytest.approx(7.20)
    assert fx["change_pct"] == pytest.approx((7.20 / 7.10 - 1) * 100)


def test_get_global_snapshot_rejects_empty_provider_response() -> None:
    gateway = MarketDataGateway(yfinance_download=lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="^global provider returned empty data$"):
        gateway.get_global_snapshot()


def test_get_global_snapshot_wraps_yahoo_failure() -> None:
    failure = RuntimeError("https://query.invalid/?crumb=secret")
    gateway = MarketDataGateway(yfinance_download=Mock(side_effect=failure), clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match="^global provider unavailable$") as caught:
        gateway.get_global_snapshot()

    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert "https://query.invalid" not in formatted_traceback(caught)


def test_get_global_snapshot_ignores_same_day_incomplete_close() -> None:
    frame = global_download_frame(multi_index=False)
    frame.loc[pd.Timestamp("2026-08-19")] = [999.0] * len(frame.columns)
    gateway = MarketDataGateway(yfinance_download=lambda *args, **kwargs: frame, clock=lambda: OBSERVED_AT)

    result = gateway.get_global_snapshot()

    assert result.frame["close"].tolist() == [102.0] * 6
    assert result.frame["as_of_date"].tolist() == [date(2026, 8, 18)] * 6


def test_get_global_snapshot_uses_exchange_session_closes() -> None:
    frame = global_download_frame(multi_index=False)
    frame.loc[pd.Timestamp("2026-08-19")] = [104.0] * len(frame.columns)
    frame = frame.sort_index()

    before_equity_close = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: datetime(2026, 8, 19, 19, 59, tzinfo=timezone.utc),
    ).get_global_snapshot()
    after_equity_close = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: datetime(2026, 8, 19, 20, 1, tzinfo=timezone.utc),
    ).get_global_snapshot()
    after_futures_close = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: datetime(2026, 8, 19, 22, 1, tzinfo=timezone.utc),
    ).get_global_snapshot()

    assert before_equity_close.frame.set_index("symbol")["close"].to_dict() == {
        symbol: 102.0 for symbol in ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F"]
    }
    after_equity = after_equity_close.frame.set_index("symbol")["close"].to_dict()
    assert after_equity["^GSPC"] == 104.0
    assert after_equity["^IXIC"] == 104.0
    assert after_equity["^DJI"] == 104.0
    assert after_equity["GC=F"] == 102.0
    assert after_equity["HG=F"] == 102.0
    assert after_equity["CL=F"] == 102.0
    assert after_futures_close.frame["close"].tolist() == [104.0] * 6


def test_get_global_snapshot_is_invariant_to_observation_timezone() -> None:
    frame = global_download_frame(multi_index=False)
    frame.loc[pd.Timestamp("2026-08-19")] = [104.0] * len(frame.columns)
    instant_utc = datetime(2026, 8, 19, 22, 1, tzinfo=timezone.utc)
    instant_shanghai = instant_utc.astimezone(ZoneInfo("Asia/Shanghai"))

    utc_result = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: instant_utc,
    ).get_global_snapshot()
    shanghai_result = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: instant_shanghai,
    ).get_global_snapshot()

    pd.testing.assert_frame_equal(utc_result.frame, shanghai_result.frame)
    assert utc_result.observed_at == shanghai_result.observed_at


def dated_global_frame(dates: list[str]) -> pd.DataFrame:
    symbols = ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F"]
    return pd.DataFrame(
        {symbol: np.arange(len(dates), dtype=float) + 100.0 for symbol in symbols},
        index=pd.to_datetime(dates),
    )


def test_get_global_snapshot_honors_xnys_early_close() -> None:
    frame = dated_global_frame(["2026-11-24", "2026-11-25", "2026-11-27"])
    observed = datetime(2026, 11, 27, 14, 0, tzinfo=ZoneInfo("America/New_York"))

    result = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: observed,
    ).get_global_snapshot()

    as_of = result.frame.set_index("symbol")["as_of_date"].to_dict()
    assert as_of["^GSPC"] == date(2026, 11, 27)
    assert as_of["^IXIC"] == date(2026, 11, 27)
    assert as_of["^DJI"] == date(2026, 11, 27)


def test_get_global_snapshot_uses_prior_sessions_on_holiday_and_weekend() -> None:
    frame = dated_global_frame(["2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27", "2026-11-28"])
    thanksgiving = datetime(2026, 11, 26, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    weekend = datetime(2026, 11, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))

    holiday_result = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: thanksgiving,
    ).get_global_snapshot()
    weekend_result = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: frame,
        clock=lambda: weekend,
    ).get_global_snapshot()

    holiday_dates = holiday_result.frame.set_index("symbol")["as_of_date"].to_dict()
    assert holiday_dates["^GSPC"] == date(2026, 11, 25)
    assert holiday_dates["GC=F"] == date(2026, 11, 26)
    assert set(weekend_result.frame["as_of_date"]) == {date(2026, 11, 27)}


def test_get_global_snapshot_sanitizes_calendar_failure() -> None:
    failure = RuntimeError("https://calendar.invalid/?token=secret")
    gateway = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: global_download_frame(multi_index=False),
        clock=lambda: OBSERVED_AT,
    )

    with (
        patch("src.collaborative_report.market_data.exchange_calendars.get_calendar", side_effect=failure),
        pytest.raises(ValueError, match="^market calendar unavailable$") as caught,
    ):
        gateway.get_global_snapshot()

    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert "https://calendar.invalid" not in formatted_traceback(caught)


def test_get_gold_bars_calls_yfinance_and_normalizes_ohlcv() -> None:
    download = Mock(return_value=daily_bars(end="2026-08-18"))
    gateway = MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT)

    result = gateway.get_gold_bars()

    download.assert_called_once_with(
        "GC=F",
        period="10y",
        end="2026-08-19",
        interval="1d",
        auto_adjust=False,
        progress=False,
        timeout=15,
    )
    assert result == MarketDataset(
        frame=result.frame,
        source="yfinance:GC=F",
        observed_at=OBSERVED_AT,
    )
    assert list(result.frame.columns) == ["date", "open", "high", "low", "close", "volume"]


def test_get_gold_bars_rejects_empty_provider_response() -> None:
    gateway = MarketDataGateway(yfinance_download=lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="^gold provider returned empty data$"):
        gateway.get_gold_bars()


def test_get_gold_bars_wraps_yahoo_failure() -> None:
    failure = RuntimeError("https://query.invalid/?crumb=secret")
    gateway = MarketDataGateway(yfinance_download=Mock(side_effect=failure), clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match="^gold provider unavailable$") as caught:
        gateway.get_gold_bars()

    assert caught.value.__cause__ is None
    assert "secret" not in formatted_traceback(caught)
    assert "https://query.invalid" not in formatted_traceback(caught)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[:59], "daily bars insufficient"),
        (lambda frame: pd.concat([frame.iloc[:-1], frame.iloc[[-2]]], ignore_index=True), "daily bars duplicate dates"),
        (
            lambda frame: frame.assign(
                Date=[*frame["Date"].iloc[:10], frame.loc[11, "Date"], frame.loc[10, "Date"], *frame["Date"].iloc[12:]]
            ),
            "daily bars non-monotonic dates",
        ),
        (lambda frame: frame.assign(High=frame["Low"] - 1.0), "daily bars invalid ohlc"),
        (lambda frame: frame.assign(Volume=-1.0), "daily bars invalid volume"),
    ],
)
def test_get_gold_bars_rejects_invalid_history(mutate, message: str) -> None:
    frame = daily_bars(end="2026-08-18")
    gateway = MarketDataGateway(yfinance_download=lambda *args, **kwargs: mutate(frame), clock=lambda: OBSERVED_AT)

    with pytest.raises(ValueError, match=f"^{message}$"):
        gateway.get_gold_bars()


def test_get_gold_bars_rejects_incomplete_row_before_futures_cutoff() -> None:
    gateway = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: daily_bars(end="2026-08-19"),
        clock=lambda: datetime(2026, 8, 19, 21, 59, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="^daily bars future dates$"):
        gateway.get_gold_bars()


def test_get_gold_bars_accepts_local_date_after_futures_cutoff() -> None:
    download = Mock(return_value=daily_bars(end="2026-08-19"))
    gateway = MarketDataGateway(
        yfinance_download=download,
        clock=lambda: datetime(2026, 8, 19, 22, 1, tzinfo=timezone.utc),
    )

    result = gateway.get_gold_bars()

    assert download.call_args.kwargs["end"] == "2026-08-20"
    assert result.frame.iloc[-1]["date"].date() == date(2026, 8, 19)


def test_get_gold_bars_does_not_download_when_calendar_is_unavailable() -> None:
    download = Mock()
    gateway = MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT)

    with (
        patch("src.collaborative_report.market_data.exchange_calendars.get_calendar", side_effect=RuntimeError),
        pytest.raises(ValueError, match="^market calendar unavailable$"),
    ):
        gateway.get_gold_bars()

    download.assert_not_called()


def test_get_gold_bars_rejects_stale_history(caplog) -> None:
    gateway = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: daily_bars(end="2026-07-31"),
        clock=lambda: OBSERVED_AT,
    )

    with pytest.raises(ValueError, match="^daily bars stale$"):
        gateway.get_gold_bars()
    assert "actual=2026-07-31" in caplog.text
    assert "expected=" in caplog.text


@pytest.mark.parametrize("first", ["stale", "empty", "timeout", "connection"])
def test_gold_history_bounded_retry_recovers_only_complete_series(first) -> None:
    complete = daily_bars(end="2026-08-18")
    first_result = {
        "stale": daily_bars(end="2026-08-17"), "empty": pd.DataFrame(),
        "timeout": TimeoutError(), "connection": ConnectionError(),
    }[first]
    download = Mock(side_effect=[first_result, complete])
    result = MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT).get_gold_bars()
    assert download.call_count == 2
    assert download.call_args_list[0].kwargs["end"] == download.call_args_list[1].kwargs["end"] == "2026-08-19"
    assert download.call_args_list[1].kwargs["start"] == "2016-08-18"
    assert "period" not in download.call_args_list[1].kwargs
    assert result.frame.iloc[-1]["date"].date() == date(2026, 8, 18)
    assert result.warnings == ("gold history recovered after bounded retry",)


def test_gold_history_retry_does_not_change_analysis_window() -> None:
    complete = daily_bars(end="2026-08-18")
    padding = complete.iloc[[0]].assign(Date=pd.Timestamp("2016-08-18"))
    padded = pd.concat([padding, complete], ignore_index=True)
    primary = MarketDataGateway(yfinance_download=Mock(return_value=complete), clock=lambda: OBSERVED_AT).get_gold_bars()
    retry = MarketDataGateway(
        yfinance_download=Mock(side_effect=[pd.DataFrame(), padded]), clock=lambda: OBSERVED_AT,
    ).get_gold_bars()
    pd.testing.assert_frame_equal(primary.frame, retry.frame)


def test_gold_history_retry_validates_padding_before_removing_it() -> None:
    complete = daily_bars(end="2026-08-18")
    invalid_padding = complete.iloc[[0]].assign(Date=pd.Timestamp("2016-08-18"), Volume=-1)
    download = Mock(side_effect=[pd.DataFrame(), pd.concat([invalid_padding, complete], ignore_index=True)])
    with pytest.raises(ValueError, match="daily bars invalid volume"):
        MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT).get_gold_bars()
    assert download.call_count == 2


def test_gold_history_stale_retry_remains_unavailable() -> None:
    download = Mock(return_value=daily_bars(end="2026-08-17"))
    with pytest.raises(ValueError, match="daily bars stale"):
        MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT).get_gold_bars()
    assert download.call_count == 2


@pytest.mark.parametrize("fault", ["volume", "future", "insufficient"])
def test_gold_history_integrity_failure_never_triggers_retry(fault) -> None:
    frame = daily_bars(end="2026-08-18")
    if fault == "volume":
        frame = frame.assign(Volume=-1)
    elif fault == "future":
        frame = daily_bars(end="2026-08-19")
    else:
        frame = frame.iloc[:59]
    download = Mock(return_value=frame)
    with pytest.raises(ValueError):
        MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT).get_gold_bars()
    download.assert_called_once()


def test_market_dataset_requires_timezone_aware_observation() -> None:
    with pytest.raises(ValueError, match="^observed_at must be timezone-aware$"):
        MarketDataset(pd.DataFrame(), "test", datetime(2026, 8, 19))
