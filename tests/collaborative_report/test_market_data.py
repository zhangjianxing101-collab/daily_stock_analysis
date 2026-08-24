import traceback
from datetime import date, datetime, timedelta, timezone
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
    validate_daily_bars,
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


def test_ths_snapshot_recoverable_failure_uses_existing_source_with_warning() -> None:
    client = Mock()
    client.a_share_snapshot.side_effect = ThsNetworkError("THS network request failed")
    fallback = Mock(return_value=pd.DataFrame({"代码": ["000001"], "名称": ["A"], "最新价": [10]}))
    gateway = MarketDataGateway(ths_client=client, snapshot_fetcher=fallback, clock=lambda: OBSERVED_AT)

    result = gateway.get_a_share_snapshot()

    fallback.assert_called_once_with()
    assert result.source == "akshare.stock_zh_a_spot_em"
    assert result.warnings == ("THS unavailable; existing snapshot source used", "snapshot source timestamp unavailable")


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
        ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F"],
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        timeout=15,
    )
    assert result.frame["symbol"].tolist() == ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F"]
    assert result.frame["change_pct"].tolist() == pytest.approx([2.0] * 6)
    assert result.observed_at == OBSERVED_AT


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
    gateway = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: daily_bars(end="2026-08-19"),
        clock=lambda: datetime(2026, 8, 19, 22, 1, tzinfo=timezone.utc),
    )

    result = gateway.get_gold_bars()

    assert result.frame.iloc[-1]["date"].date() == date(2026, 8, 19)


def test_get_gold_bars_rejects_stale_history() -> None:
    gateway = MarketDataGateway(
        yfinance_download=lambda *args, **kwargs: daily_bars(end="2026-07-31"),
        clock=lambda: OBSERVED_AT,
    )

    with pytest.raises(ValueError, match="^daily bars stale$"):
        gateway.get_gold_bars()


def test_market_dataset_requires_timezone_aware_observation() -> None:
    with pytest.raises(ValueError, match="^observed_at must be timezone-aware$"):
        MarketDataset(pd.DataFrame(), "test", datetime(2026, 8, 19))
