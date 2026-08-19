from datetime import date, datetime, timezone
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

from src.collaborative_report.market_data import (
    MarketDataGateway,
    MarketDataset,
    normalize_a_share_snapshot,
    validate_daily_bars,
)


SESSION = date(2026, 8, 19)
OBSERVED_AT = datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc)


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


def test_get_a_share_snapshot_rejects_empty_provider_response() -> None:
    gateway = MarketDataGateway(snapshot_fetcher=lambda: pd.DataFrame())

    with pytest.raises(ValueError, match="^snapshot provider returned empty data$"):
        gateway.get_a_share_snapshot()


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
    )
    assert result.frame["symbol"].tolist() == ["^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F"]
    assert result.frame["change_pct"].tolist() == pytest.approx([2.0] * 6)
    assert result.observed_at == OBSERVED_AT


def test_get_global_snapshot_rejects_empty_provider_response() -> None:
    gateway = MarketDataGateway(yfinance_download=lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="^global provider returned empty data$"):
        gateway.get_global_snapshot()


def test_get_global_snapshot_ignores_same_day_incomplete_close() -> None:
    frame = global_download_frame(multi_index=False)
    frame.loc[pd.Timestamp("2026-08-19")] = [999.0] * len(frame.columns)
    gateway = MarketDataGateway(yfinance_download=lambda *args, **kwargs: frame, clock=lambda: OBSERVED_AT)

    result = gateway.get_global_snapshot()

    assert result.frame["close"].tolist() == [102.0] * 6
    assert result.frame["as_of_date"].tolist() == [date(2026, 8, 18)] * 6


def test_get_gold_bars_calls_yfinance_and_normalizes_ohlcv() -> None:
    download = Mock(return_value=daily_bars())
    gateway = MarketDataGateway(yfinance_download=download, clock=lambda: OBSERVED_AT)

    result = gateway.get_gold_bars()

    download.assert_called_once_with("GC=F", period="10y", interval="1d", auto_adjust=False, progress=False)
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


def test_market_dataset_requires_timezone_aware_observation() -> None:
    with pytest.raises(ValueError, match="^observed_at must be timezone-aware$"):
        MarketDataset(pd.DataFrame(), "test", datetime(2026, 8, 19))
