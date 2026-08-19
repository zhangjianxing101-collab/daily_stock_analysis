"""Normalized, injectable market-data access for collaborative reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable

import exchange_calendars
import numpy as np
import pandas as pd


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


@dataclass(frozen=True)
class MarketDataset:
    frame: pd.DataFrame
    source: str
    observed_at: datetime
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")


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
        yfinance_download: Callable[..., pd.DataFrame] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._snapshot_fetcher = snapshot_fetcher
        self._daily_fetcher = daily_fetcher
        self._sector_names_fetcher = sector_names_fetcher
        self._sector_members_fetcher = sector_members_fetcher
        self._yfinance_download = yfinance_download
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _observed_at(self) -> datetime:
        return self._clock()

    def get_a_share_snapshot(self) -> MarketDataset:
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
        return MarketDataset(normalized, "akshare.stock_zh_a_spot_em", self._observed_at())

    def get_daily_bars(self, code: str, expected_session: date, *, days: int = 160) -> MarketDataset:
        try:
            if self._daily_fetcher is None:
                from data_provider.base import DataFetcherManager

                raw, source = DataFetcherManager().get_daily_data(code, days=days)
            else:
                raw, source = self._daily_fetcher(code, days=days)
        except Exception:
            raise ValueError("daily provider unavailable") from None
        if raw is None or raw.empty:
            raise ValueError("daily provider returned empty data")
        normalized = validate_daily_bars(raw, expected_session)
        return MarketDataset(normalized, str(source), self._observed_at())

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
        try:
            raw = self._download(
                "GC=F",
                period="10y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=15,
            )
        except Exception:
            raise ValueError("gold provider unavailable") from None
        if raw is None or raw.empty:
            raise ValueError("gold provider returned empty data")
        latest_completed = _latest_completed_session_date("GC=F", observed_at)
        normalized = _validate_bar_series(
            raw,
            min_rows=60,
            latest_allowed=latest_completed,
            required_latest=latest_completed,
        )
        return MarketDataset(normalized, "yfinance:GC=F", observed_at)
