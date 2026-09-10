"""Deterministic, offline sector-strength analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import pandas as pd


_SECTOR_TYPES = frozenset(("industry", "concept"))
_STATE_KEYS = frozenset(("rank", "change_pct", "breadth_pct", "activity_percentile", "universe_size"))


@dataclass(frozen=True)
class SectorRow:
    sector_type: str
    name: str
    rank: int
    change_pct: float
    breadth_pct: float | None
    activity_percentile: float | None
    leader_name: str | None
    leader_code: str | None
    leader_change_pct: float | None
    rotation: str
    persistence: str
    crowding_risk: str


@dataclass(frozen=True)
class SectorAnalysis:
    strongest: tuple[SectorRow, ...]
    weakest: tuple[SectorRow, ...]
    watch: tuple[SectorRow, ...]
    valid_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectorClassification:
    rotation: str
    persistence: str
    crowding_risk: str


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _optional_number(value: object, label: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    return _finite_number(value, label)


def _optional_text(value: object, label: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string when present")
    return value.strip()


def _state_number(state: Mapping[str, object], key: str, *, required: bool = False) -> float | None:
    if key not in state:
        if required:
            raise ValueError(f"current {key} is required")
        return None
    value = _optional_number(state[key], key)
    if required and value is None:
        raise ValueError(f"current {key} is required")
    return value


def _validated_state(state: Mapping[str, object], *, current: bool) -> dict[str, float | None]:
    if not isinstance(state, Mapping):
        raise ValueError("sector state must be a mapping")
    label = "current" if current else "previous"
    required = frozenset(("rank", "change_pct")) if current else frozenset()
    values: dict[str, float | None] = {}
    for key in _STATE_KEYS:
        values[key] = _state_number(state, key, required=key in required)
    rank = values["rank"]
    universe_size = values["universe_size"]
    if rank is not None and (rank <= 0 or not rank.is_integer()):
        raise ValueError(f"{label} rank must be a positive integer")
    if universe_size is not None and (universe_size <= 0 or not universe_size.is_integer()):
        raise ValueError(f"{label} universe_size must be a positive integer")
    if rank is not None and universe_size is not None and rank > universe_size:
        raise ValueError(f"{label} rank must not exceed universe_size")
    for key in ("breadth_pct", "activity_percentile"):
        value = values[key]
        if value is not None and not 0 <= value <= 100:
            raise ValueError(f"{label} {key} must be between 0 and 100")
    return values


def _top_twenty(state: Mapping[str, float | None]) -> bool:
    return state["rank"] is not None and state["rank"] <= 20


def _crowding(state: Mapping[str, float | None]) -> str:
    breadth = state["breadth_pct"]
    activity = state["activity_percentile"]
    if breadth is None or activity is None:
        return "unavailable"
    if _top_twenty(state) and activity >= 90 and breadth < 50:
        return "high"
    if activity >= 75 and breadth >= 50:
        return "medium"
    return "low"


def classify_sector(current: Mapping[str, object], previous: Mapping[str, object] | None) -> SectorClassification:
    """Classify one current sector against its prior normalized observation."""

    current_state = _validated_state(current, current=True)
    previous_state = None if previous is None else _validated_state(previous, current=False)
    if previous is None:
        rotation = "first_observation"
    elif current_state["change_pct"] > 0 and (
        (current_state["breadth_pct"] is not None and current_state["breadth_pct"] < 50)
        or (
            current_state["activity_percentile"] is not None
            and previous_state["activity_percentile"] is not None
            and current_state["activity_percentile"] < previous_state["activity_percentile"]
        )
    ):
        rotation = "diverging"
    elif _top_twenty(previous_state) and (
        current_state["change_pct"] < 0
        or (
            current_state["universe_size"] is not None
            and current_state["rank"] > current_state["universe_size"] / 2
        )
    ):
        rotation = "retreating"
    elif (
        _top_twenty(current_state)
        and _top_twenty(previous_state)
        and current_state["rank"] <= previous_state["rank"] - 5
        and (
            current_state["activity_percentile"] is None
            or previous_state["activity_percentile"] is None
            or current_state["activity_percentile"] >= previous_state["activity_percentile"]
        )
    ):
        rotation = "accelerating"
    elif previous_state["rank"] is None or previous_state["rank"] > 20:
        rotation = "new_start" if _top_twenty(current_state) else "continuing"
    else:
        rotation = "continuing"

    breadth = current_state["breadth_pct"]
    activity = current_state["activity_percentile"]
    if previous_state is None or breadth is None or activity is None or previous_state["activity_percentile"] is None:
        persistence = "unavailable"
    elif breadth < 50 or activity < previous_state["activity_percentile"] or rotation == "retreating":
        persistence = "low"
    elif _top_twenty(current_state) and _top_twenty(previous_state) and breadth >= 55 and activity >= previous_state["activity_percentile"]:
        persistence = "high"
    elif _top_twenty(current_state) and (
        _top_twenty(previous_state) or breadth >= 55 or activity >= previous_state["activity_percentile"]
    ):
        persistence = "medium"
    else:
        persistence = "low"
    return SectorClassification(rotation=rotation, persistence=persistence, crowding_risk=_crowding(current_state))


def _validate_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a DataFrame")
    required = {"sector_type", "name", "change_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {', '.join(sorted(missing))}")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for record in frame.to_dict("records"):
        sector_type = record["sector_type"]
        name = record["name"]
        if sector_type not in _SECTOR_TYPES:
            raise ValueError("sector_type must be industry or concept")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty string")
        normalized_name = name.strip()
        key = (sector_type, normalized_name)
        if key in seen:
            raise ValueError("duplicate sector_type and name")
        seen.add(key)
        change_pct = _finite_number(record["change_pct"], "change_pct")
        advance = _optional_number(record.get("advance_count"), "advance_count")
        decline = _optional_number(record.get("decline_count"), "decline_count")
        if advance is not None and (advance < 0 or not advance.is_integer()):
            raise ValueError("advance_count must be a non-negative integer")
        if decline is not None and (decline < 0 or not decline.is_integer()):
            raise ValueError("decline_count must be a non-negative integer")
        denominator = None if advance is None or decline is None else advance + decline
        breadth_pct = None if denominator is None or denominator == 0 else advance / denominator * 100
        turnover_rate = _optional_number(record.get("turnover_rate"), "turnover_rate")
        if turnover_rate is not None and turnover_rate < 0:
            raise ValueError("turnover_rate must be non-negative")
        amount = _optional_number(record.get("amount"), "amount")
        if amount is not None and amount < 0:
            raise ValueError("amount must be non-negative")
        rows.append({
            "sector_type": sector_type,
            "name": normalized_name,
            "change_pct": change_pct,
            "breadth_pct": breadth_pct,
            "turnover_rate": turnover_rate if turnover_rate is not None else amount,
            "leader_name": _optional_text(record.get("leader_name"), "leader_name"),
            "leader_code": _optional_text(record.get("leader_code"), "leader_code"),
            "leader_change_pct": _optional_number(record.get("leader_change_pct"), "leader_change_pct"),
        })
    return rows


def _activity_percentiles(rows: list[dict[str, object]]) -> None:
    for sector_type in _SECTOR_TYPES:
        group = [row for row in rows if row["sector_type"] == sector_type and row["turnover_rate"] is not None]
        group.sort(key=lambda row: (float(row["turnover_rate"]), str(row["name"])))
        count = len(group)
        if count == 1:
            group[0]["activity_percentile"] = 100.0
            continue
        start = 0
        while start < count:
            end = start + 1
            while end < count and group[end]["turnover_rate"] == group[start]["turnover_rate"]:
                end += 1
            percentile = (start + end - 1) / 2 / (count - 1) * 100
            for row in group[start:end]:
                row["activity_percentile"] = percentile
            start = end
    for row in rows:
        row.setdefault("activity_percentile", None)


def analyze_sectors(
    frame: pd.DataFrame,
    *,
    previous: Mapping[tuple[str, str], Mapping[str, object]] | tuple[()] | None,
    observed_at: datetime,
    limit: int = 10,
) -> SectorAnalysis:
    """Validate and rank a sector frame without mutating its source data."""

    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if previous != () and previous is not None and not isinstance(previous, Mapping):
        raise ValueError("previous must be a mapping, an empty tuple, or None")
    no_trustworthy_prior = previous is None or previous == ()
    rows = _validate_frame(frame)
    _activity_percentiles(rows)
    rows.sort(key=lambda row: (-float(row["change_pct"]), str(row["name"]), str(row["sector_type"])))
    universe_size = len(rows)
    output: list[SectorRow] = []
    for index, row in enumerate(rows, start=1):
        state = {
            "rank": index,
            "change_pct": row["change_pct"],
            "breadth_pct": row["breadth_pct"],
            "activity_percentile": row["activity_percentile"],
            "universe_size": universe_size,
        }
        prior = None if no_trustworthy_prior else previous.get((str(row["sector_type"]), str(row["name"])), {})
        classification = classify_sector(state, prior)
        output.append(SectorRow(
            sector_type=str(row["sector_type"]), name=str(row["name"]), rank=index,
            change_pct=float(row["change_pct"]), breadth_pct=row["breadth_pct"],
            activity_percentile=row["activity_percentile"], leader_name=row["leader_name"],
            leader_code=row["leader_code"], leader_change_pct=row["leader_change_pct"],
            rotation=classification.rotation, persistence=classification.persistence,
            crowding_risk=classification.crowding_risk,
        ))
    strongest = tuple(output[:limit])
    strongest_keys = {(row.sector_type, row.name) for row in strongest}
    weakest = tuple(sorted(
        (row for row in output if (row.sector_type, row.name) not in strongest_keys),
        key=lambda row: (row.change_pct, row.name, row.sector_type),
    )[:limit])
    watch = tuple(row for row in strongest if row.persistence in {"high", "medium"})
    return SectorAnalysis(strongest=strongest, weakest=weakest, watch=watch, valid_count=len(output))
