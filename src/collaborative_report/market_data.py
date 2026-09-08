"""Normalized, injectable market-data access for collaborative reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import logging
from typing import Any, Callable, Mapping, Sequence

import exchange_calendars
import numpy as np
import pandas as pd

from .ths_market_data import (
    ThsAuthenticationError,
    ThsConfigurationError,
    ThsDataUnavailableError,
    ThsHotListPeriod,
    ThsIndexTag,
    ThsMarketDataClient,
    ThsNetworkError,
    ThsPermissionError,
    ThsRateLimitError,
    ThsResponseError,
)


_SNAPSHOT_COLUMNS = (
    "code",
    "name",
    "price",
    "change_pct",
    "volume_ratio",
    "turnover",
    "amount",
    "volume",
    "total_mv",
)
_SNAPSHOT_ALIASES = {
    "code": ("code", "代码", "股票代码", "证券代码"),
    "name": ("name", "名称", "股票简称", "证券简称"),
    "price": ("price", "最新价", "现价", "最新"),
    "change_pct": ("change_pct", "涨跌幅", "涨幅"),
    "volume_ratio": ("volume_ratio", "量比"),
    "turnover": ("turnover", "换手率", "换手"),
    "amount": ("amount", "成交额"),
    "volume": ("volume", "成交量"),
    "total_mv": ("total_mv", "总市值"),
}
_SECTOR_SNAPSHOT_COLUMNS = (
    "sector_type",
    "name",
    "change_pct",
    "advance_count",
    "decline_count",
    "turnover_rate",
    "amount",
    "leader_name",
    "leader_code",
    "leader_change_pct",
)
_SECTOR_SNAPSHOT_ALIASES = {
    "name": ("板块名称", "名称", "name", "sector"),
    "change_pct": ("涨跌幅", "change_pct", "涨幅"),
    "advance_count": ("上涨家数", "advance_count"),
    "decline_count": ("下跌家数", "decline_count"),
    "turnover_rate": ("换手率", "turnover_rate", "turnover"),
    "amount": ("成交额", "amount"),
    "leader_name": ("领涨股票", "领涨股", "leader_name"),
    "leader_code": ("领涨股票代码", "领涨股代码", "leader_code", "code"),
    "leader_change_pct": ("领涨股票-涨跌幅", "领涨股涨跌幅", "leader_change_pct"),
}
_SECTOR_POSTMARKET_FRESHNESS = timedelta(hours=4)
_THS_SECTOR_BATCH_SIZE = 100
_BAR_ALIASES = {
    "date": ("date", "Date", "日期", "时间"),
    "open": ("open", "Open", "开盘"),
    "high": ("high", "High", "最高"),
    "low": ("low", "Low", "最低"),
    "close": ("close", "Close", "收盘"),
    "volume": ("volume", "Volume", "成交量"),
}
_GLOBAL_SYMBOLS = ("^GSPC", "^IXIC", "^DJI", "GC=F", "HG=F", "CL=F")
_SYMBOL_CALENDARS = {
    "^GSPC": "XNYS",
    "^IXIC": "XNYS",
    "^DJI": "XNYS",
    "GC=F": "CMES",
    "HG=F": "CMES",
    "CL=F": "CMES",
}
_THS_SOURCE = "ths.fuyao"
logger = logging.getLogger(__name__)
_THS_FAILURE_CODES = {
    ThsAuthenticationError: "ths_authentication_failed",
    ThsConfigurationError: "ths_configuration_missing",
    ThsDataUnavailableError: "ths_data_unavailable",
    ThsNetworkError: "ths_network_failed",
    ThsPermissionError: "ths_permission_denied",
    ThsRateLimitError: "ths_rate_limited",
}
_THS_NETWORK_FAILURE_CODES = {
    reason: f"ths_{reason}" for reason in (
        "network_failed", "tls_failed", "proxy_failed", "connect_timeout",
        "read_timeout", "connection_failed", "internal_failure",
        "credential_encoding_failed", "internal_type_error", "internal_value_error", "internal_attribute_error",
    )
}
_RECOVERABLE_THS_ERRORS = (
    ThsAuthenticationError,
    ThsConfigurationError,
    ThsDataUnavailableError,
    ThsNetworkError,
    ThsPermissionError,
    ThsRateLimitError,
)


@dataclass(frozen=True)
class MarketDataset:
    frame: pd.DataFrame
    source: str
    observed_at: datetime
    warnings: tuple[str, ...] = ()
    source_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.source_timestamp is not None and (
            self.source_timestamp.tzinfo is None or self.source_timestamp.utcoffset() is None
        ):
            raise ValueError("source_timestamp must be timezone-aware")


def _snapshot_source_timestamp(raw: pd.DataFrame) -> tuple[datetime | None, tuple[str, ...]]:
    unavailable = (None, ("snapshot source timestamp unavailable",))
    value: object | None = None
    for key in ("source_timestamp", "quote_timestamp", "data_timestamp"):
        candidate = raw.attrs.get(key)
        if candidate is None:
            continue
        if not pd.api.types.is_scalar(candidate):
            return unavailable
        try:
            missing = pd.isna(candidate)
            if not isinstance(missing, (bool, np.bool_)):
                return unavailable
            if bool(missing):
                continue
        except Exception:
            return unavailable
        value = candidate
        break
    if value is None:
        return unavailable
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Shanghai")
        return timestamp.to_pydatetime(), ()
    except Exception:
        return unavailable


def _matching_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> object | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    return None


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "-": pd.NA, "--": pd.NA, "—": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def _canonical_code(value: object) -> str | None:
    if pd.isna(value) or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        text = str(int(value)).zfill(6)
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        text = str(int(value)).zfill(6)
    else:
        text = str(value).strip()
    if len(text) == 6 and text.isascii() and text.isdigit():
        return text
    return None


def _optional_canonical_code(value: object) -> str | None:
    try:
        return _canonical_code(value)
    except Exception:
        return None


def normalize_a_share_thscode(code: object) -> str:
    """Return a documented THS A-share code without guessing the exchange."""

    normalized = _canonical_code(code)
    if normalized is None:
        raise ValueError("A-share code must contain exactly six digits")

    prefix = normalized[:3]
    if prefix in {"600", "601", "603", "605", "688", "689"}:
        return f"{normalized}.SH"
    if prefix in {"000", "001", "002", "003", "300", "301"}:
        return f"{normalized}.SZ"
    if prefix.startswith(("43", "83", "87", "88", "92")):
        return f"{normalized}.BJ"
    raise ValueError("A-share code exchange is unknown")


def _ths_timestamp(timestamp_ms: int | None, observed_at: datetime) -> tuple[datetime | None, tuple[str, ...]]:
    if timestamp_ms is None:
        return None, ("THS source timestamp unavailable",)
    try:
        timestamp = pd.Timestamp(timestamp_ms, unit="ms", tz="UTC").tz_convert("Asia/Shanghai").to_pydatetime()
    except (OverflowError, TypeError, ValueError):
        raise ValueError("THS source timestamp invalid") from None
    if timestamp > observed_at:
        raise ValueError("THS source timestamp is in the future")
    return timestamp, ()


def _ths_items_frame(items: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([dict(item) for item in items])


def _normalize_ths_snapshot(items: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    raw = _ths_items_frame(items)
    required = {"thscode", "ticker", "last_price", "price_change_ratio_pct", "volume", "turnover"}
    if raw.empty:
        raise ThsDataUnavailableError(3002)
    if not required.issubset(raw.columns):
        raise ValueError("THS snapshot missing required fields")

    result = pd.DataFrame(
        {
            "code": raw["ticker"].map(_canonical_code),
            "name": pd.Series(pd.NA, index=raw.index, dtype="string"),
            "price": _numeric(raw["last_price"]),
            "change_pct": _numeric(raw["price_change_ratio_pct"]),
            "volume_ratio": pd.Series(pd.NA, index=raw.index, dtype="Float64"),
            "turnover": pd.Series(pd.NA, index=raw.index, dtype="Float64"),
            "amount": _numeric(raw["turnover"]),
            "volume": _numeric(raw["volume"]),
            "total_mv": pd.Series(pd.NA, index=raw.index, dtype="Float64"),
        }
    )
    if result["code"].isna().any():
        raise ValueError("THS snapshot contains unsafe prices")
    if result["code"].duplicated().any():
        raise ValueError("THS snapshot contains duplicate codes")
    unquoted = (raw["last_price"].isna() & (
        (raw["volume"].isna() | result["volume"].eq(0))
        & (raw["turnover"].isna() | result["amount"].eq(0))
    )).fillna(False)
    excluded = int(unquoted.sum())
    result = result.loc[~unquoted].copy()
    if result.empty or not result["price"].notna().all():
        raise ValueError("THS snapshot contains unsafe prices")
    if not np.isfinite(result["price"].to_numpy(dtype=float)).all() or (result["price"] <= 0).any():
        raise ValueError("THS snapshot contains unsafe prices")
    for column in ("change_pct", "amount", "volume"):
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (column in {"amount", "volume"} and (values < 0).any()):
            raise ValueError("THS snapshot contains invalid values")
    result = result.loc[:, _SNAPSHOT_COLUMNS].reset_index(drop=True)
    result.attrs.update(provider_row_count=len(raw), quarantined_row_count=excluded)
    return result


def _normalize_ths_daily_bars(items: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    raw = _ths_items_frame(items)
    required = {"date_ms", "open_price", "high_price", "low_price", "close_price", "volume"}
    if raw.empty:
        raise ThsDataUnavailableError(3002)
    if not required.issubset(raw.columns):
        raise ValueError("THS daily bars missing required fields")
    try:
        dates = pd.to_datetime(raw["date_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("THS daily bars invalid dates") from None
    return pd.DataFrame(
        {
            "date": dates,
            "open": raw["open_price"],
            "high": raw["high_price"],
            "low": raw["low_price"],
            "close": raw["close_price"],
            "volume": raw["volume"],
        }
    )


def normalize_a_share_snapshot(raw: pd.DataFrame) -> pd.DataFrame:
    """Map common AkShare snapshot columns without changing ``raw``."""

    if not isinstance(raw, pd.DataFrame):
        raise ValueError("snapshot data invalid")
    source = raw.copy(deep=True)
    result = pd.DataFrame(index=source.index)
    for canonical, aliases in _SNAPSHOT_ALIASES.items():
        column = _matching_column(source, aliases)
        result[canonical] = source[column].copy() if column is not None else pd.NA

    result["code"] = result["code"].map(_canonical_code)
    result = result.loc[result["code"].notna()].copy()
    result["name"] = result["name"].astype("string")
    for column in _SNAPSHOT_COLUMNS[2:]:
        result[column] = _numeric(result[column])
    return result.loc[:, _SNAPSHOT_COLUMNS].reset_index(drop=True)


def _normalize_sector_snapshot(raw: pd.DataFrame, sector_type: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Normalize a board list while retaining unavailable optional fields as nulls."""

    if not isinstance(raw, pd.DataFrame):
        raise ValueError(f"{sector_type} sector provider returned invalid data")
    source = raw.copy(deep=True)
    name_column = _matching_column(source, _SECTOR_SNAPSHOT_ALIASES["name"])
    change_column = _matching_column(source, _SECTOR_SNAPSHOT_ALIASES["change_pct"])
    if name_column is None or change_column is None:
        raise ValueError(f"{sector_type} sector provider returned invalid data")

    result = pd.DataFrame(index=source.index)
    result["sector_type"] = pd.Series(sector_type, index=source.index, dtype="string")
    result["name"] = source[name_column].astype("string").str.strip().replace("", pd.NA)
    result["change_pct"] = _numeric(source[change_column]).astype("Float64")
    for column in ("advance_count", "decline_count", "turnover_rate", "amount", "leader_change_pct"):
        source_column = _matching_column(source, _SECTOR_SNAPSHOT_ALIASES[column])
        result[column] = (
            _numeric(source[source_column]).astype("Float64")
            if source_column is not None
            else pd.Series(pd.NA, index=source.index, dtype="Float64")
        )
    leader_name_column = _matching_column(source, _SECTOR_SNAPSHOT_ALIASES["leader_name"])
    result["leader_name"] = (
        source[leader_name_column].astype("string").str.strip().replace("", pd.NA)
        if leader_name_column is not None
        else pd.Series(pd.NA, index=source.index, dtype="string")
    )
    leader_code_column = _matching_column(source, _SECTOR_SNAPSHOT_ALIASES["leader_code"])
    result["leader_code"] = (
        source[leader_code_column].map(_optional_canonical_code).astype("string")
        if leader_code_column is not None
        else pd.Series(pd.NA, index=source.index, dtype="string")
    )

    valid = result["name"].notna() & result["change_pct"].notna()
    valid &= np.isfinite(result["change_pct"].to_numpy(dtype=float, na_value=np.nan))
    excluded = int((~valid).sum())
    result = result.loc[valid].copy()
    if result.empty:
        raise ValueError(f"{sector_type} sector provider returned invalid data")

    for column in ("advance_count", "decline_count", "turnover_rate", "amount", "leader_change_pct"):
        values = result[column]
        supplied = values.notna()
        if supplied.any() and not np.isfinite(values.loc[supplied].to_numpy(dtype=float)).all():
            raise ValueError(f"{sector_type} sector provider returned invalid data")
    for column in ("advance_count", "decline_count"):
        values = result[column].dropna()
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{sector_type} sector provider returned invalid data")
        result[column] = result[column].astype("Int64")
    if (result["turnover_rate"].dropna() < 0).any() or (result["amount"].dropna() < 0).any():
        raise ValueError(f"{sector_type} sector provider returned invalid data")
    if result.duplicated(subset=["sector_type", "name"]).any():
        raise ValueError(f"{sector_type} sector provider returned invalid data")

    warnings = (f"sector snapshot rows excluded: {excluded}",) if excluded else ()
    return result.loc[:, _SECTOR_SNAPSHOT_COLUMNS].reset_index(drop=True), warnings


def _flatten_yfinance_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy(deep=True)
    columns = frame.columns
    for level in range(columns.nlevels):
        labels = {str(value) for value in columns.get_level_values(level)}
        if "Close" in labels and {"Open", "High", "Low"}.issubset(labels):
            flattened = frame.copy(deep=True)
            flattened.columns = columns.get_level_values(level)
            return flattened
    return frame.copy(deep=True)


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed: list[pd.Timestamp | pd.NaT] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                parsed.append(pd.NaT)
                continue
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_localize(None)
            parsed.append(timestamp.normalize())
        except (TypeError, ValueError, OverflowError):
            parsed.append(pd.NaT)
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns]")


def _normalize_bar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("daily bars invalid")
    source = _flatten_yfinance_bars(frame)
    if _matching_column(source, _BAR_ALIASES["date"]) is None:
        source = source.copy(deep=True)
        source["date"] = source.index

    result = pd.DataFrame(index=source.index)
    for canonical, aliases in _BAR_ALIASES.items():
        column = _matching_column(source, aliases)
        if column is None:
            raise ValueError("daily bars missing columns")
        result[canonical] = source[column].copy()
    result["date"] = _parse_dates(result["date"])
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = _numeric(result[column])
    return result.reset_index(drop=True)


def _validate_bar_values(frame: pd.DataFrame) -> None:
    ohlc = frame.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
        raise ValueError("daily bars invalid ohlc")
    open_values = frame["open"].to_numpy(dtype=float)
    high_values = frame["high"].to_numpy(dtype=float)
    low_values = frame["low"].to_numpy(dtype=float)
    close_values = frame["close"].to_numpy(dtype=float)
    if (
        (high_values < low_values).any()
        or (high_values < np.maximum(open_values, close_values)).any()
        or (low_values > np.minimum(open_values, close_values)).any()
    ):
        raise ValueError("daily bars invalid ohlc")
    volume = frame["volume"].to_numpy(dtype=float)
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("daily bars invalid volume")


def _validate_bar_series(
    frame: pd.DataFrame,
    *,
    min_rows: int,
    latest_allowed: date,
    required_latest: date | None = None,
) -> pd.DataFrame:
    normalized = _normalize_bar_columns(frame)
    if len(normalized) < min_rows:
        raise ValueError("daily bars insufficient")
    if normalized["date"].isna().any():
        raise ValueError("daily bars invalid dates")
    if normalized["date"].duplicated().any():
        raise ValueError("daily bars duplicate dates")
    dates = normalized["date"]
    if not (dates.is_monotonic_increasing or dates.is_monotonic_decreasing):
        raise ValueError("daily bars non-monotonic dates")
    if (dates > pd.Timestamp(latest_allowed)).any():
        raise ValueError("daily bars future dates")
    _validate_bar_values(normalized)
    ascending = normalized.sort_values("date", kind="stable").reset_index(drop=True)
    if required_latest is not None and ascending.iloc[-1]["date"].date() != required_latest:
        logger.warning(
            "Daily bars stale: expected=%s actual=%s",
            required_latest.isoformat(), ascending.iloc[-1]["date"].date().isoformat(),
        )
        raise ValueError("daily bars stale")
    return ascending


def validate_daily_bars(frame: pd.DataFrame, expected_session: date, *, min_rows: int = 60) -> pd.DataFrame:
    """Validate a complete daily series and return a new ascending frame."""

    return _validate_bar_series(
        frame,
        min_rows=min_rows,
        latest_allowed=expected_session,
        required_latest=expected_session,
    )


def _latest_completed_session_date(symbol: str, observed_at: datetime) -> date:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    try:
        observed_utc = pd.Timestamp(observed_at).tz_convert("UTC")
        calendar = exchange_calendars.get_calendar(_SYMBOL_CALENDARS[symbol])
        session = calendar.date_to_session(observed_utc.date(), direction="previous")
        while calendar.session_close(session) > observed_utc:
            session = calendar.previous_session(session)
        return session.date()
    except Exception:
        raise ValueError("market calendar unavailable") from None


class MarketDataGateway:
    """Provider boundary whose callables can be replaced for offline operation."""

    def __init__(
        self,
        *,
        snapshot_fetcher: Callable[[], pd.DataFrame] | None = None,
        daily_fetcher: Callable[..., tuple[pd.DataFrame, str]] | None = None,
        sector_names_fetcher: Callable[[], pd.DataFrame] | None = None,
        sector_members_fetcher: Callable[[str], pd.DataFrame] | None = None,
        industry_sector_fetcher: Callable[[], pd.DataFrame] | None = None,
        concept_sector_fetcher: Callable[[], pd.DataFrame] | None = None,
        yfinance_download: Callable[..., pd.DataFrame] | None = None,
        ths_client: ThsMarketDataClient | None = None,
        snapshot_supplement_fetcher: Callable[[Sequence[str]], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._snapshot_fetcher = snapshot_fetcher
        self._daily_fetcher = daily_fetcher
        self._sector_names_fetcher = sector_names_fetcher
        self._sector_members_fetcher = sector_members_fetcher
        self._industry_sector_fetcher = industry_sector_fetcher
        self._concept_sector_fetcher = concept_sector_fetcher
        self._yfinance_download = yfinance_download
        self._ths_client = ths_client
        self._snapshot_supplement_fetcher = snapshot_supplement_fetcher
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _observed_at(self) -> datetime:
        return self._clock()

    def _received_at(self, requested_at: datetime) -> datetime:
        received_at = self._observed_at()
        if (
            requested_at.tzinfo is None or requested_at.utcoffset() is None
            or received_at.tzinfo is None or received_at.utcoffset() is None
            or received_at < requested_at
        ):
            raise ValueError("provider acquisition clock invalid")
        return received_at

    def _ths_snapshot(self, codes: Sequence[str] | None, observed_at: datetime) -> MarketDataset:
        client = self._require_ths_client()
        thscodes = None if codes is None else tuple(normalize_a_share_thscode(code) for code in codes)
        if thscodes is not None:
            response = client.a_share_snapshot(thscodes)
            items = response.items
            observed_at = self._received_at(observed_at)
            source_timestamp, warnings = _ths_timestamp(response.timestamp_ms, observed_at)
        else:
            items, observed_at, source_timestamp, warnings = self._ths_all_snapshot_pages(client, observed_at)
        frame = _normalize_ths_snapshot(items)
        if frame.attrs.get("quarantined_row_count", 0):
            warnings = (*warnings, "ths_snapshot_unquoted_rows_excluded")
        source = _THS_SOURCE + ".a_share_snapshot"
        if self._snapshot_supplement_fetcher is not None:
            frame, observed_at, source_timestamp, supplement_warnings = self._enrich_snapshot(
                frame, observed_at, source_timestamp,
            )
            warnings = (*warnings, *supplement_warnings)
            if frame.attrs.get("screening_complete_count", 0):
                source += "+tencent"
        return MarketDataset(
            frame,
            source,
            observed_at,
            warnings,
            source_timestamp,
        )

    def _enrich_snapshot(self, frame, observed_at, source_timestamp):
        result = frame.copy(deep=True)
        result.attrs["screening_complete_count"] = 0
        warning = "snapshot_screening_fields_incomplete"
        supported_codes: list[str] = []
        for code in result["code"]:
            try:
                normalize_a_share_thscode(code)
            except ValueError:
                continue
            supported_codes.append(code)
        try:
            supplement = self._snapshot_supplement_fetcher(tuple(supported_codes))
        except Exception:
            return result, self._received_at(observed_at), source_timestamp, (warning,)
        received_at = self._received_at(observed_at)
        try:
            if (supplement.observed_at.tzinfo is None or supplement.observed_at.utcoffset() is None
                    or not observed_at <= supplement.observed_at <= received_at):
                raise ValueError("supplement acquisition time invalid")
            records = supplement.frame
            required = {"code", "name", "price", "volume_ratio", "turnover", "source_timestamp"}
            if not required.issubset(records.columns) or records["code"].duplicated().any():
                raise ValueError("supplement fields invalid")
        except Exception:
            return result, received_at, source_timestamp, (warning,)
        # The THS snapshot timestamp is the response assembly time, not the
        # quote's completed-session identity. Prove that identity independently.
        from .session import latest_completed_xshg_session

        exchange_tz = "Asia/Shanghai"
        try:
            expected = latest_completed_xshg_session(observed_at)
        except Exception:
            return result, received_at, source_timestamp, (warning,)
        indexed = records.set_index("code")
        timestamps = []
        for index, row in result.iterrows():
            code = row["code"]
            if code not in indexed.index:
                continue
            other = indexed.loc[code]
            try:
                stamp = pd.Timestamp(other["source_timestamp"])
                if pd.isna(stamp) or stamp.tzinfo is None:
                    continue
                local = stamp.tz_convert(exchange_tz)
                if local.date() != expected or local.hour < 15 or stamp > supplement.observed_at:
                    continue
                name = other["name"]
                price, ratio, turnover = (float(other[key]) for key in ("price", "volume_ratio", "turnover"))
                if (not isinstance(name, str) or not name.strip()
                        or not np.isfinite([price, ratio, turnover]).all()
                        or price <= 0 or ratio < 0 or turnover < 0
                        or abs(price - float(row["price"])) > 0.01 + 1e-9):
                    continue
                result.loc[index, ["name", "volume_ratio", "turnover"]] = [name, ratio, turnover]
                timestamps.append(stamp.to_pydatetime())
            except (TypeError, ValueError, OverflowError):
                continue
        result.attrs["screening_complete_count"] = len(timestamps)
        if timestamps:
            source_timestamp = (
                min(timestamps)
                if source_timestamp is None
                else min(source_timestamp, *timestamps)
            )
            result.attrs["supplement_source"] = "tencent"
            result.attrs["supplement_source_timestamp"] = min(timestamps).isoformat()
        warnings = (warning,) if len(timestamps) < len(result) else ()
        return result, received_at, source_timestamp, warnings

    def _ths_all_snapshot_pages(
        self, client: ThsMarketDataClient, observed_at: datetime
    ) -> tuple[tuple[Mapping[str, Any], ...], datetime, datetime | None, tuple[str, ...]]:
        offset = 0
        items: list[Mapping[str, Any]] = []
        response: Any | None = None
        expected_total: int | None = None
        timestamps: list[datetime | None] = []
        warnings: list[str] = []
        while expected_total is None or len(items) < expected_total:
            response = client.a_share_snapshot(None, limit=1000, offset=offset)
            observed_at = self._received_at(observed_at)
            total = response.data.get("total")
            if type(total) is not int or total < 0:
                raise ValueError("THS snapshot total invalid")
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise ValueError("THS snapshot total changed during pagination")
            timestamp, page_warnings = _ths_timestamp(response.timestamp_ms, observed_at)
            timestamps.append(timestamp)
            warnings.extend(page_warnings)
            page_items = response.items
            if not page_items:
                if expected_total == 0 and not items:
                    raise ThsDataUnavailableError(3002)
                raise ValueError("THS snapshot pagination incomplete")
            items.extend(page_items)
            if len(items) > expected_total:
                raise ValueError("THS snapshot pagination invalid")
            offset += len(page_items)
        if response is None:
            raise AssertionError("THS snapshot pagination must return a response")
        session_dates = {pd.Timestamp(stamp).tz_convert("Asia/Shanghai").date()
                         for stamp in timestamps if stamp is not None}
        if len(session_dates) > 1:
            raise ValueError("THS snapshot pages have mixed sessions")
        source_timestamp = min(timestamps) if all(stamp is not None for stamp in timestamps) else None
        return tuple(items), observed_at, source_timestamp, tuple(dict.fromkeys(warnings))

    def get_a_share_snapshot(self, codes: Sequence[str] | None = None) -> MarketDataset:
        observed_at = self._observed_at()
        fallback_warnings: tuple[str, ...] = ()
        try:
            return self._ths_snapshot(codes, observed_at)
        except ThsResponseError:
            raise ValueError("THS snapshot data invalid") from None
        except _RECOVERABLE_THS_ERRORS as exc:
            code = _THS_FAILURE_CODES.get(type(exc), "ths_unavailable")
            if isinstance(exc, ThsNetworkError):
                code = _THS_NETWORK_FAILURE_CODES.get(exc.reason, "ths_network_failed")
            logger.warning("A-share primary source: %s", code)
            fallback_warnings = ("THS unavailable; existing snapshot source used",)
        try:
            if self._snapshot_fetcher is None:
                import akshare

                raw = akshare.stock_zh_a_spot_em()
            else:
                raw = self._snapshot_fetcher()
        except Exception:
            raise ValueError("snapshot provider unavailable") from None
        if raw is None or raw.empty:
            raise ValueError("snapshot provider returned empty data")
        normalized = normalize_a_share_snapshot(raw)
        if normalized.empty:
            raise ValueError("snapshot provider returned empty data")
        source_timestamp, warnings = _snapshot_source_timestamp(raw)
        return MarketDataset(
            normalized,
            "akshare.stock_zh_a_spot_em",
            self._received_at(observed_at),
            fallback_warnings + warnings,
            source_timestamp,
        )

    def _ths_daily_bars(self, code: str, expected_session: date, days: int, observed_at: datetime) -> MarketDataset:
        if type(days) is not int or days <= 0:
            raise ValueError("days must be a positive integer")
        thscode = normalize_a_share_thscode(code)
        start = datetime.combine(expected_session - timedelta(days=days * 2), datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(expected_session + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(milliseconds=1)
        response = self._require_ths_client().a_share_historical(
            thscode,
            start_ms=int(start.timestamp() * 1000),
            end_ms=int(end.timestamp() * 1000),
        )
        observed_at = self._received_at(observed_at)
        source_timestamp, warnings = _ths_timestamp(response.timestamp_ms, observed_at)
        normalized = validate_daily_bars(_normalize_ths_daily_bars(response.items), expected_session)
        return MarketDataset(normalized, _THS_SOURCE + ".a_share_historical", observed_at, warnings, source_timestamp)

    def get_daily_bars(self, code: str, expected_session: date, *, days: int = 160) -> MarketDataset:
        observed_at = self._observed_at()
        fallback_warnings: tuple[str, ...] = ()
        try:
            return self._ths_daily_bars(code, expected_session, days, observed_at)
        except ThsResponseError:
            raise ValueError("THS daily bars invalid") from None
        except ValueError as exc:
            if str(exc) != "daily bars stale":
                raise
            fallback_warnings = ("THS daily bars stale; existing daily source used",)
        except _RECOVERABLE_THS_ERRORS:
            fallback_warnings = ("THS unavailable; existing daily source used",)
        try:
            if self._daily_fetcher is None:
                from data_provider.base import DataFetcherManager

                raw, source = DataFetcherManager().get_daily_data(
                    code,
                    days=days,
                    validator=lambda frame: validate_daily_bars(frame, expected_session),
                )
            else:
                raw, source = self._daily_fetcher(code, days=days)
        except Exception:
            raise ValueError("daily provider unavailable") from None
        if raw is None or raw.empty:
            raise ValueError("daily provider returned empty data")
        normalized = validate_daily_bars(raw, expected_session)
        return MarketDataset(normalized, str(source), self._received_at(observed_at), fallback_warnings)

    def _require_ths_client(self) -> ThsMarketDataClient:
        if self._ths_client is None:
            raise ThsConfigurationError("THS data provider is not configured")
        return self._ths_client

    def _ths_dataset(self, source: str, response: Any, observed_at: datetime) -> MarketDataset:
        observed_at = self._received_at(observed_at)
        source_timestamp, warnings = _ths_timestamp(response.timestamp_ms, observed_at)
        return MarketDataset(_ths_items_frame(response.items), source, observed_at, warnings, source_timestamp)

    def get_ths_financial_indicators(self, code: str, report: str) -> MarketDataset:
        response = self._require_ths_client().financial_indicators(normalize_a_share_thscode(code), report)
        records = [
            {
                "thscode": response.data["thscode"],
                "report": response.data["report"],
                "ability": ability["ability"],
                **indicator,
            }
            for ability in response.data["abilities"]
            for indicator in ability["indicators"]
        ]
        return MarketDataset(
            pd.DataFrame(records),
            _THS_SOURCE + ".financial_indicators",
            self._observed_at(),
            ("THS source timestamp unavailable",),
        )

    def get_ths_hot_stock_list(self, period: ThsHotListPeriod = ThsHotListPeriod.DAY) -> MarketDataset:
        observed_at = self._observed_at()
        return self._ths_dataset(_THS_SOURCE + ".hot_stock_list", self._require_ths_client().hot_stock_list(period), observed_at)

    def get_ths_skyrocket_list(self, period: ThsHotListPeriod = ThsHotListPeriod.DAY) -> MarketDataset:
        observed_at = self._observed_at()
        return self._ths_dataset(_THS_SOURCE + ".skyrocket_list", self._require_ths_client().skyrocket_list(period), observed_at)

    def get_ths_index_catalog(self, tag: ThsIndexTag = ThsIndexTag.CONCEPT) -> MarketDataset:
        observed_at = self._observed_at()
        return self._ths_dataset(_THS_SOURCE + ".index_catalog", self._require_ths_client().ths_index_catalog(tag), observed_at)

    def get_ths_index_constituents(self, thscode: str) -> MarketDataset:
        observed_at = self._observed_at()
        return self._ths_dataset(
            _THS_SOURCE + ".index_constituents", self._require_ths_client().ths_index_constituents(thscode), observed_at
        )

    def get_ths_index_snapshot(self, thscodes: Sequence[str]) -> MarketDataset:
        observed_at = self._observed_at()
        return self._ths_dataset(_THS_SOURCE + ".index_snapshot", self._require_ths_client().index_snapshot(thscodes), observed_at)

    def get_ths_index_bars(self, thscode: str, expected_session: date, *, days: int = 160) -> MarketDataset:
        if type(days) is not int or days <= 0:
            raise ValueError("days must be a positive integer")
        observed_at = self._observed_at()
        start = datetime.combine(expected_session - timedelta(days=days * 2), datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(expected_session + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - timedelta(milliseconds=1)
        response = self._require_ths_client().index_historical(
            thscode,
            start_ms=int(start.timestamp() * 1000),
            end_ms=int(end.timestamp() * 1000),
        )
        observed_at = self._received_at(observed_at)
        source_timestamp, warnings = _ths_timestamp(response.timestamp_ms, observed_at)
        normalized = validate_daily_bars(_normalize_ths_daily_bars(response.items), expected_session)
        return MarketDataset(normalized, _THS_SOURCE + ".index_historical", observed_at, warnings, source_timestamp)

    def get_leading_sector_codes(self, limit: int = 10) -> MarketDataset:
        try:
            if self._sector_names_fetcher is None or self._sector_members_fetcher is None:
                import akshare

                names_fetcher = self._sector_names_fetcher or akshare.stock_board_industry_name_em
                members_fetcher = self._sector_members_fetcher or akshare.stock_board_industry_cons_em
            else:
                names_fetcher = self._sector_names_fetcher
                members_fetcher = self._sector_members_fetcher
            boards = names_fetcher()
        except Exception:
            raise ValueError("sector provider unavailable") from None
        if boards is None or boards.empty:
            raise ValueError("sector provider returned empty data")
        name_column = _matching_column(boards, ("板块名称", "名称", "name", "sector"))
        change_column = _matching_column(boards, ("涨跌幅", "change_pct", "涨幅"))
        if name_column is None or change_column is None:
            raise ValueError("sector provider returned invalid data")
        ranked = pd.DataFrame({"name": boards[name_column].astype("string"), "change": _numeric(boards[change_column])})
        ranked = ranked.dropna(subset=["name", "change"]).sort_values("change", ascending=False).head(max(limit, 0))

        records: list[dict[str, str]] = []
        seen: set[str] = set()
        warnings: list[str] = []
        for sector in ranked["name"].tolist():
            sector_name = str(sector)
            try:
                members = members_fetcher(sector_name)
                if members is None or members.empty:
                    raise ValueError
                code_column = _matching_column(members, ("代码", "股票代码", "code"))
                if code_column is None:
                    raise ValueError
                codes = members[code_column].map(_canonical_code).dropna()
                for code in codes:
                    if code not in seen:
                        records.append({"code": code, "sector": sector_name})
                        seen.add(code)
            except Exception:
                warnings.append(f"sector constituents unavailable: {sector_name}")

        frame = pd.DataFrame(records, columns=["code", "sector"])
        return MarketDataset(frame, "akshare.industry_boards", self._observed_at(), tuple(warnings))

    def get_sector_snapshot(self, sector_type: str) -> MarketDataset:
        if not isinstance(sector_type, str) or sector_type not in {"industry", "concept"}:
            raise ValueError("sector type invalid")
        configured_fetcher = (
            self._industry_sector_fetcher if sector_type == "industry" else self._concept_sector_fetcher
        )
        if configured_fetcher is None and self._ths_client is not None:
            try:
                return self._ths_sector_snapshot(sector_type)
            except ThsResponseError:
                raise ValueError(f"{sector_type} sector provider returned invalid data") from None
            except _RECOVERABLE_THS_ERRORS:
                logger.warning("%s sector primary source unavailable", sector_type)
        requested_at = self._observed_at()
        fetcher = configured_fetcher
        try:
            if fetcher is None:
                import akshare

                fetcher = (
                    akshare.stock_board_industry_name_em
                    if sector_type == "industry"
                    else akshare.stock_board_concept_name_em
                )
            raw = fetcher()
        except Exception:
            raise ValueError(f"{sector_type} sector provider unavailable") from None
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            raise ValueError(f"{sector_type} sector provider returned empty data")
        if not isinstance(raw, pd.DataFrame):
            raise ValueError(f"{sector_type} sector provider returned invalid data")

        try:
            normalized, normalization_warnings = _normalize_sector_snapshot(raw, sector_type)
        except Exception:
            raise ValueError(f"{sector_type} sector provider returned invalid data") from None
        received_at = self._received_at(requested_at)
        source_timestamp, timestamp_warnings = _snapshot_source_timestamp(raw)
        if source_timestamp is not None:
            if source_timestamp > received_at:
                raise ValueError(f"{sector_type} sector source timestamp is in the future")
            received_local = pd.Timestamp(received_at).tz_convert("Asia/Shanghai")
            if (
                received_local.time() >= datetime.min.time().replace(hour=15)
                and source_timestamp < received_at - _SECTOR_POSTMARKET_FRESHNESS
            ):
                raise ValueError(f"{sector_type} sector snapshot stale")
        return MarketDataset(
            normalized,
            f"akshare.eastmoney_{sector_type}_boards",
            received_at,
            tuple(dict.fromkeys((*normalization_warnings, *timestamp_warnings))),
            source_timestamp,
        )

    def _ths_sector_snapshot(self, sector_type: str) -> MarketDataset:
        """Build a sector board from the documented THS catalog and quote APIs."""

        requested_at = self._observed_at()
        client = self._require_ths_client()
        tag = ThsIndexTag.INDUSTRY if sector_type == "industry" else ThsIndexTag.CONCEPT
        catalog = client.ths_index_catalog(tag)
        received_at = self._received_at(requested_at)
        catalog_timestamp, catalog_warnings = _ths_timestamp(catalog.timestamp_ms, received_at)

        names: dict[str, str] = {}
        for item in catalog.items:
            thscode = item.get("thscode")
            name = item.get("name")
            if (
                not isinstance(thscode, str)
                or not thscode.strip()
                or not isinstance(name, str)
                or not name.strip()
            ):
                continue
            normalized_code = thscode.strip().upper()
            normalized_name = name.strip()
            if normalized_code in names or normalized_name in names.values():
                raise ThsResponseError("THS API returned duplicate sector catalog rows")
            names[normalized_code] = normalized_name
        if not names:
            raise ThsDataUnavailableError(3002)

        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        snapshot_timestamps: list[datetime] = []
        codes = tuple(names)
        for start in range(0, len(codes), _THS_SECTOR_BATCH_SIZE):
            response = client.index_snapshot(codes[start:start + _THS_SECTOR_BATCH_SIZE])
            received_at = self._received_at(received_at)
            response_timestamp, _ = _ths_timestamp(response.timestamp_ms, received_at)
            if response_timestamp is not None:
                snapshot_timestamps.append(response_timestamp)
            for item in response.items:
                thscode = item.get("thscode")
                if not isinstance(thscode, str):
                    continue
                normalized_code = thscode.strip().upper()
                if normalized_code not in names or normalized_code in seen:
                    continue
                seen.add(normalized_code)
                rows.append({
                    "name": names[normalized_code],
                    "change_pct": item.get("price_change_ratio_pct"),
                    "amount": item.get("turnover"),
                })
        if not rows:
            raise ThsDataUnavailableError(3002)

        raw = pd.DataFrame(rows)
        normalized, normalization_warnings = _normalize_sector_snapshot(raw, sector_type)
        missing = len(names) - len(seen)
        warnings = list(catalog_warnings)
        warnings.extend(normalization_warnings)
        if missing:
            warnings.append(f"THS sector snapshot rows missing: {missing}")
        source_timestamp = min(snapshot_timestamps) if snapshot_timestamps else catalog_timestamp
        return MarketDataset(
            normalized,
            _THS_SOURCE + ".index_snapshot",
            received_at,
            tuple(dict.fromkeys(warnings)),
            source_timestamp,
        )

    def _download(self, *args, **kwargs) -> pd.DataFrame:
        if self._yfinance_download is None:
            import yfinance

            return yfinance.download(*args, **kwargs)
        return self._yfinance_download(*args, **kwargs)

    @staticmethod
    def _global_closes(raw: pd.DataFrame) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            for level in range(raw.columns.nlevels):
                if "Close" in raw.columns.get_level_values(level):
                    return raw.xs("Close", axis=1, level=level, drop_level=True).copy()
            raise ValueError("global provider returned invalid data")
        if all(symbol in raw.columns for symbol in _GLOBAL_SYMBOLS):
            return raw.loc[:, _GLOBAL_SYMBOLS].copy()
        raise ValueError("global provider returned invalid data")

    def get_global_snapshot(self) -> MarketDataset:
        observed_at = self._observed_at()
        try:
            raw = self._download(
                list(_GLOBAL_SYMBOLS),
                period="5d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=15,
            )
        except Exception:
            raise ValueError("global provider unavailable") from None
        if raw is None or raw.empty:
            raise ValueError("global provider returned empty data")
        closes = self._global_closes(raw)
        records: list[dict[str, object]] = []
        warnings: list[str] = []
        for symbol in _GLOBAL_SYMBOLS:
            if symbol not in closes.columns:
                warnings.append(f"global close unavailable: {symbol}")
                continue
            values = pd.to_numeric(closes[symbol], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            latest_completed = _latest_completed_session_date(symbol, observed_at)
            completed = [pd.Timestamp(index).date() <= latest_completed for index in values.index]
            values = values.loc[completed]
            values = values.sort_index()
            if len(values) < 2:
                warnings.append(f"global close unavailable: {symbol}")
                continue
            previous_close = float(values.iloc[-2])
            close = float(values.iloc[-1])
            if previous_close <= 0 or close <= 0:
                warnings.append(f"global close unavailable: {symbol}")
                continue
            records.append(
                {
                    "symbol": symbol,
                    "close": close,
                    "previous_close": previous_close,
                    "change_pct": (close / previous_close - 1.0) * 100.0,
                    "as_of_date": pd.Timestamp(values.index[-1]).date(),
                }
            )
        if not records:
            raise ValueError("global provider returned invalid data")
        return MarketDataset(pd.DataFrame(records), "yfinance", observed_at, tuple(warnings))

    def get_gold_bars(self) -> MarketDataset:
        observed_at = self._observed_at()
        latest_completed = _latest_completed_session_date("GC=F", observed_at)
        end = latest_completed + timedelta(days=1)
        canonical_start = (pd.Timestamp(end) - pd.DateOffset(years=10)).date()
        for attempt in range(2):
            window = {"period": "10y"} if attempt == 0 else {
                "start": (canonical_start - timedelta(days=1)).isoformat(),
            }
            failure = None
            try:
                raw = self._download(
                    "GC=F", **window, end=end.isoformat(), interval="1d",
                    auto_adjust=False, progress=False, timeout=15,
                )
            except (TimeoutError, ConnectionError):
                failure = "gold provider unavailable"
                raw = None
            except Exception:
                raise ValueError("gold provider unavailable") from None
            observed_at = self._received_at(observed_at)
            if raw is None or raw.empty:
                failure = failure or "gold provider returned empty data"
            else:
                try:
                    normalized = _validate_bar_series(
                        raw, min_rows=60, latest_allowed=latest_completed,
                        required_latest=latest_completed,
                    )
                except ValueError as exc:
                    if str(exc) != "daily bars stale":
                        raise
                    failure = "daily bars stale"
                else:
                    # Validate all rows before removing deliberate leading retry padding.
                    normalized = _validate_bar_series(
                        normalized.loc[normalized["date"] >= pd.Timestamp(canonical_start)],
                        min_rows=60, latest_allowed=latest_completed, required_latest=latest_completed,
                    )
                    warnings = ("gold history recovered after bounded retry",) if attempt else ()
                    return MarketDataset(normalized, "yfinance:GC=F", observed_at, warnings)
            if attempt:
                raise ValueError(failure) from None
            logger.warning("Gold history: bounded_retry")
        raise AssertionError("gold retry must return or raise")
