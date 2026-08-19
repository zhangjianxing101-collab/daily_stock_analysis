"""Deterministic aggressive A-share screening for collaborative reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd

from .market_data import MarketDataset
from .models import Candidate


_SNAPSHOT_REQUIRED = ("code", "name", "price", "change_pct", "volume_ratio", "turnover", "amount", "volume")
_SHORT_CORE_RULES = frozenset(("ma5>ma10>ma20", "close_breaks_20d_high"))
_SWING_CORE_RULE = "ma20>ma50且close>ma20"
_BAR_ALIASES = {
    "date": ("date", "Date", "日期", "时间"),
    "open": ("open", "Open", "开盘"),
    "high": ("high", "High", "最高"),
    "low": ("low", "Low", "最低"),
    "close": ("close", "Close", "收盘"),
    "volume": ("volume", "Volume", "成交量"),
}


@dataclass(frozen=True)
class ScreeningResult:
    short_term: tuple[Candidate, ...]
    swing: tuple[Candidate, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Indicators:
    close: float
    ma5: float
    ma10: float
    ma20: float
    ma50: float
    prior_high20: float
    return5: float
    return20: float
    atr14: float
    volume_expansion: float


@dataclass(frozen=True)
class _RankedCandidate:
    candidate: Candidate
    amount: float
    inaccessible: bool


def _at_boundary(value: float, boundary: float) -> bool:
    return bool(np.isclose(value, boundary, rtol=0, atol=1e-10))


def _snapshot_frame(snapshot: MarketDataset | pd.DataFrame) -> pd.DataFrame:
    frame = snapshot.frame if isinstance(snapshot, MarketDataset) else snapshot
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("snapshot must be a DataFrame or MarketDataset")
    return frame.copy(deep=True)


def _eligible_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    if any(column not in frame.columns for column in _SNAPSHOT_REQUIRED):
        return frame.iloc[0:0].copy()

    eligible = frame.copy(deep=True)
    code = eligible["code"].astype("string")
    name = eligible["name"].astype("string")
    ascii_code = code.map(lambda value: value.isascii(), na_action="ignore").fillna(False)
    valid = code.str.fullmatch(r"[0-9]{6}", na=False) & ascii_code
    valid &= (
        name.notna()
        & ~name.str.contains("st", case=False, regex=False, na=True)
        & ~name.str.contains("退", regex=False, na=True)
    )

    for column in ("price", "change_pct", "volume_ratio", "turnover", "amount"):
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
        valid &= np.isfinite(eligible[column])
    valid &= (eligible["price"] > 0) & (eligible["amount"] >= 50_000_000)

    eligible["volume"] = pd.to_numeric(eligible["volume"], errors="coerce")
    valid &= np.isfinite(eligible["volume"]) & (eligible["volume"] > 0)

    eligible = eligible.loc[valid].copy()
    eligible["code"] = code.loc[valid].astype(str)
    eligible["name"] = name.loc[valid].astype(str)
    return eligible


def prefilter_universe(snapshot: MarketDataset | pd.DataFrame, limit: int) -> pd.DataFrame:
    """Return an eligible, deterministic high-activity subset without mutating input."""

    frame = _eligible_snapshot(_snapshot_frame(snapshot))
    if frame.empty or limit <= 0:
        return frame.iloc[0:0].reset_index(drop=True)

    activity = pd.DataFrame(index=frame.index)
    activity["amount"] = frame["amount"]
    activity["turnover"] = frame["turnover"]
    activity["volume_ratio"] = frame["volume_ratio"]
    activity["change"] = frame["change_pct"].abs()
    composite = activity.rank(method="average", pct=True, ascending=True).mean(axis=1)
    ranked = frame.assign(_prefilter_score=composite)
    ranked = ranked.sort_values(["_prefilter_score", "code"], ascending=[False, True], kind="stable")
    ranked = ranked.drop_duplicates(subset="code", keep="first").head(limit)
    return ranked.drop(columns="_prefilter_score").reset_index(drop=True)


def _history_frame(history: MarketDataset | pd.DataFrame) -> pd.DataFrame:
    frame = history.frame if isinstance(history, MarketDataset) else history
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("history invalid")
    source = frame.copy(deep=True)
    if not any(alias in source.columns for alias in _BAR_ALIASES["date"]):
        source["date"] = source.index

    normalized = pd.DataFrame(index=source.index)
    for canonical, aliases in _BAR_ALIASES.items():
        column = next((alias for alias in aliases if alias in source.columns), None)
        if column is None:
            raise ValueError("history invalid")
        normalized[canonical] = source[column].copy()

    if len(normalized) < 60:
        raise ValueError("history invalid")
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if normalized["date"].isna().any() or normalized["date"].duplicated().any():
        raise ValueError("history invalid")
    if not normalized["date"].is_monotonic_increasing:
        raise ValueError("history invalid")

    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    values = normalized[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("history invalid")
    if (normalized[["open", "high", "low", "close"]] <= 0).any(axis=None) or (normalized["volume"] < 0).any():
        raise ValueError("history invalid")
    if (
        (normalized["high"] < normalized["low"]).any()
        or (normalized["high"] < normalized[["open", "close"]].max(axis=1)).any()
        or (normalized["low"] > normalized[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("history invalid")
    return normalized.reset_index(drop=True)


def _indicators(frame: pd.DataFrame) -> _Indicators:
    close = frame["close"]
    current_close = float(close.iloc[-1])
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    previous_volume = float(frame["volume"].iloc[-6:-1].mean())
    values = _Indicators(
        close=current_close,
        ma5=float(close.iloc[-5:].mean()),
        ma10=float(close.iloc[-10:].mean()),
        ma20=float(close.iloc[-20:].mean()),
        ma50=float(close.iloc[-50:].mean()),
        prior_high20=float(frame["high"].iloc[-21:-1].max()),
        return5=(current_close / float(close.iloc[-6]) - 1.0) * 100.0,
        return20=(current_close / float(close.iloc[-21]) - 1.0) * 100.0,
        atr14=float(true_range.iloc[-14:].mean()),
        volume_expansion=float(frame["volume"].iloc[-1]) / previous_volume if previous_volume > 0 else np.nan,
    )
    if not np.isfinite(tuple(values.__dict__.values())).all() or values.atr14 <= 0:
        raise ArithmeticError("indicators incomplete")
    return values


def _metadata(
    history: MarketDataset | pd.DataFrame,
    fallback_observed_at: datetime | None,
) -> tuple[datetime, str]:
    if isinstance(history, MarketDataset):
        return history.observed_at, history.source
    if fallback_observed_at is None:
        raise ValueError("observed_at is required for DataFrame histories")
    return fallback_observed_at, "validated_history"


def _short_rules(indicators: _Indicators, turnover: float, leading: bool) -> tuple[float, tuple[str, ...]]:
    score = 0.0
    rules: list[str] = []
    conditions = (
        (indicators.ma5 > indicators.ma10 > indicators.ma20, 25, "ma5>ma10>ma20"),
        (indicators.close > indicators.prior_high20, 25, "close_breaks_20d_high"),
        (indicators.volume_expansion >= 1.5, 20, "volume_expansion>=1.5"),
        (
            indicators.return5 > 0
            and not _at_boundary(indicators.return5, 0)
            and indicators.return5 < 15
            and not _at_boundary(indicators.return5, 15),
            15,
            "0<return_5d<15%",
        ),
        (leading, 10, "leading_sector"),
        (1 <= turnover <= 12, 5, "1<=turnover<=12"),
    )
    for matched, points, rule in conditions:
        if matched:
            score += points
            rules.append(rule)
    return score, tuple(rules)


def _swing_rules(indicators: _Indicators, turnover: float, leading: bool) -> tuple[float, tuple[str, ...]]:
    score = 0.0
    rules: list[str] = []
    above_ma20 = (indicators.close / indicators.ma20 - 1.0) * 100.0
    return20_in_range = (
        (indicators.return20 > 3 or _at_boundary(indicators.return20, 3))
        and (indicators.return20 < 25 or _at_boundary(indicators.return20, 25))
    )
    above_ma20_in_range = (
        (above_ma20 > 0 or _at_boundary(above_ma20, 0))
        and (above_ma20 < 8 or _at_boundary(above_ma20, 8))
    )
    conditions = (
        (indicators.ma20 > indicators.ma50 and indicators.close > indicators.ma20, 30, "ma20>ma50且close>ma20"),
        (return20_in_range, 20, "3%<=return_20d<=25%"),
        (above_ma20_in_range, 20, "0%<=close_above_ma20<=8%"),
        (indicators.volume_expansion >= 1.2, 15, "volume_expansion>=1.2"),
        (leading, 10, "leading_sector"),
        (0.5 <= turnover <= 8, 5, "0.5<=turnover<=8"),
    )
    for matched, points, rule in conditions:
        if matched:
            score += points
            rules.append(rule)
    return score, tuple(rules)


def _trigger(horizon: str, rules: tuple[str, ...], inaccessible: bool) -> str:
    if inaccessible:
        return "观望：涨停附近，等待恢复可交易"
    if horizon == "short":
        if "close_breaks_20d_high" in rules and "volume_expansion>=1.5" in rules:
            return "放量突破前20日高点"
        if "close_breaks_20d_high" in rules:
            return "突破前20日高点"
        if "ma5>ma10>ma20" in rules:
            return "MA5高于MA10和MA20"
        return "短线动量条件跟踪"
    return "站稳MA20且MA20高于MA50" if "ma20>ma50且close>ma20" in rules else "回到MA20上方后跟踪"


def _candidate(
    row: pd.Series,
    indicators: _Indicators,
    *,
    horizon: str,
    score: float,
    rules: tuple[str, ...],
    observed_at: datetime,
    source: str,
    inaccessible: bool,
) -> Candidate:
    warning = "涨停附近不可交易，仅供观望" if inaccessible else ""
    return Candidate(
        code=str(row["code"]),
        name=str(row["name"]),
        horizon="1-5个交易日" if horizon == "short" else "1-4周",
        score=score,
        close=indicators.close,
        trigger=_trigger(horizon, rules, inaccessible),
        stop_price=round(indicators.close - 2 * indicators.atr14, 2),
        target_price=round(indicators.close + 3 * indicators.atr14, 2),
        matched_rules=rules,
        observed_at=observed_at,
        source=source,
        warning=warning,
    )


def _select(pool: list[_RankedCandidate], limit: int) -> tuple[Candidate, ...]:
    if limit <= 0:
        return ()
    ordered = sorted(pool, key=lambda item: (-item.candidate.score, -item.amount, item.candidate.code))
    actionable: list[Candidate] = []
    watch: list[Candidate] = []
    seen: set[str] = set()
    for item in ordered:
        if item.candidate.code in seen:
            continue
        seen.add(item.candidate.code)
        (watch if item.inaccessible else actionable).append(item.candidate)
    return tuple((actionable + watch)[:limit])


def screen_aggressive(
    snapshot: MarketDataset | pd.DataFrame,
    histories: Mapping[str, MarketDataset | pd.DataFrame],
    leading_sectors: Mapping[str, str],
    *,
    short_limit: int = 5,
    swing_limit: int = 5,
    prefilter_limit: int = 120,
    observed_at: datetime | None = None,
) -> ScreeningResult:
    """Screen liquid A-shares with fixed technical rules and no AI calls."""

    if observed_at is not None and (observed_at.tzinfo is None or observed_at.utcoffset() is None):
        raise ValueError("observed_at must be timezone-aware")
    if observed_at is None and any(isinstance(history, pd.DataFrame) for history in histories.values()):
        raise ValueError("observed_at is required for DataFrame histories")

    universe = prefilter_universe(snapshot, prefilter_limit)
    short_pool: list[_RankedCandidate] = []
    swing_pool: list[_RankedCandidate] = []
    warnings: list[str] = []

    for _, row in universe.iterrows():
        code = str(row["code"])
        history = histories.get(code)
        if history is None:
            warnings.append(f"{code}: history unavailable")
            continue
        try:
            frame = _history_frame(history)
        except Exception:
            warnings.append(f"{code}: history invalid")
            continue
        try:
            indicators = _indicators(frame)
        except Exception:
            warnings.append(f"{code}: indicators incomplete")
            continue

        stop_price = indicators.close - 2 * indicators.atr14
        if not np.isfinite(stop_price) or stop_price <= 0:
            warnings.append(f"{code}: risk bounds invalid")
            continue

        candidate_time, source = _metadata(history, observed_at)
        turnover = float(row["turnover"])
        leading = code in leading_sectors
        inaccessible = float(row["change_pct"]) >= 9.8
        short_score, short_rules = _short_rules(indicators, turnover, leading)
        swing_score, swing_rules = _swing_rules(indicators, turnover, leading)
        if short_score >= 45 and _SHORT_CORE_RULES.intersection(short_rules):
            short = _candidate(
                row,
                indicators,
                horizon="short",
                score=short_score,
                rules=short_rules,
                observed_at=candidate_time,
                source=source,
                inaccessible=inaccessible,
            )
            short_pool.append(_RankedCandidate(short, float(row["amount"]), inaccessible))
        if swing_score >= 50 and _SWING_CORE_RULE in swing_rules:
            swing = _candidate(
                row,
                indicators,
                horizon="swing",
                score=swing_score,
                rules=swing_rules,
                observed_at=candidate_time,
                source=source,
                inaccessible=inaccessible,
            )
            swing_pool.append(_RankedCandidate(swing, float(row["amount"]), inaccessible))
        if inaccessible:
            warnings.append(f"{code}: inaccessible upper limit, watch only")

    return ScreeningResult(
        short_term=_select(short_pool, short_limit),
        swing=_select(swing_pool, swing_limit),
        warnings=tuple(warnings),
    )
