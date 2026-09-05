from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import get_type_hints

import pandas as pd
import pytest

from src.collaborative_report.sector_analysis import (
    SectorAnalysis,
    SectorClassification,
    SectorRow,
    analyze_sectors,
    classify_sector,
)


OBSERVED_AT = datetime(2026, 9, 6, 9, 30, tzinfo=timezone.utc)


def sector_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def classification_state(**overrides: object) -> dict[str, object]:
    return {
        "rank": 10,
        "change_pct": 1.0,
        "breadth_pct": 55.0,
        "activity_percentile": 60.0,
        "universe_size": 40,
        **overrides,
    }


def test_public_contracts_are_frozen_and_exact() -> None:
    assert tuple(get_type_hints(SectorRow)) == (
        "sector_type", "name", "rank", "change_pct", "breadth_pct", "activity_percentile",
        "leader_name", "leader_code", "leader_change_pct", "rotation", "persistence", "crowding_risk",
    )
    assert tuple(get_type_hints(SectorAnalysis)) == ("strongest", "weakest", "watch", "valid_count", "warnings")
    assert tuple(get_type_hints(SectorClassification)) == ("rotation", "persistence", "crowding_risk")
    result = SectorClassification("continuing", "medium", "low")
    with pytest.raises(FrozenInstanceError):
        result.rotation = "retreating"  # type: ignore[misc]


def test_analyze_sectors_ranks_ties_derives_evidence_and_keeps_buckets_disjoint() -> None:
    frame = sector_frame([
        {"sector_type": "industry", "name": "Beta", "change_pct": 3, "advance_count": 3, "decline_count": 1, "turnover_rate": 20, "leader_name": "B", "leader_code": "2", "leader_change_pct": 5},
        {"sector_type": "industry", "name": "Alpha", "change_pct": 3, "advance_count": 1, "decline_count": 1, "turnover_rate": 10},
        {"sector_type": "industry", "name": "Gamma", "change_pct": 1, "advance_count": 0, "decline_count": 2, "turnover_rate": 30},
        {"sector_type": "concept", "name": "Delta", "change_pct": -2, "advance_count": 2, "decline_count": 0, "turnover_rate": 50},
    ])

    result = analyze_sectors(frame, previous={}, observed_at=OBSERVED_AT, limit=2)

    assert [(row.name, row.rank) for row in result.strongest] == [("Alpha", 1), ("Beta", 2)]
    assert [(row.name, row.rank) for row in result.weakest] == [("Delta", 4), ("Gamma", 3)]
    assert {row.name for row in result.strongest}.isdisjoint(row.name for row in result.weakest)
    beta = result.strongest[1]
    assert beta.breadth_pct == 75.0
    assert beta.activity_percentile == 50.0
    assert beta.leader_name == "B"
    assert beta.leader_change_pct == 5.0
    assert result.valid_count == 4


def test_analyze_sectors_fills_strongest_first_in_small_universe_and_does_not_mutate_input() -> None:
    frame = sector_frame([
        {"sector_type": "industry", "name": "A", "change_pct": 2},
        {"sector_type": "concept", "name": "B", "change_pct": 1},
        {"sector_type": "industry", "name": "C", "change_pct": -1},
    ])
    original = frame.copy(deep=True)

    result = analyze_sectors(frame, previous={}, observed_at=OBSERVED_AT, limit=2)

    assert [row.name for row in result.strongest] == ["A", "B"]
    assert [row.name for row in result.weakest] == ["C"]
    pd.testing.assert_frame_equal(frame, original)


def test_analyze_sectors_watch_uses_strongest_rows_with_durable_persistence() -> None:
    frame = sector_frame([
        {"sector_type": "industry", "name": "A", "change_pct": 2, "advance_count": 6, "decline_count": 4, "turnover_rate": 20},
        {"sector_type": "industry", "name": "B", "change_pct": 1, "advance_count": 2, "decline_count": 8, "turnover_rate": 10},
    ])
    previous = {
        ("industry", "A"): classification_state(rank=15, activity_percentile=50),
        ("industry", "B"): classification_state(rank=15, activity_percentile=60),
    }

    result = analyze_sectors(frame, previous=previous, observed_at=OBSERVED_AT)

    assert [row.name for row in result.watch] == ["A"]
    assert result.strongest[0].persistence == "high"


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [
        (classification_state(), None, "first_observation"),
        (classification_state(rank=20), {}, "new_start"),
        (classification_state(rank=10, breadth_pct=55, activity_percentile=60), classification_state(rank=15, activity_percentile=60), "accelerating"),
        (classification_state(rank=10, breadth_pct=49, activity_percentile=60), classification_state(rank=15, activity_percentile=60), "diverging"),
        (classification_state(rank=10, breadth_pct=55, activity_percentile=50), classification_state(rank=15, activity_percentile=60), "diverging"),
        (classification_state(rank=25, change_pct=-1, universe_size=40), classification_state(rank=10), "retreating"),
        (classification_state(rank=10, change_pct=-1), classification_state(rank=10), "retreating"),
        (classification_state(rank=10, change_pct=0), classification_state(rank=12), "continuing"),
    ],
)
def test_classify_sector_covers_all_rotation_states(
    current: dict[str, object], previous: dict[str, object] | None, expected: str
) -> None:
    assert classify_sector(current, previous).rotation == expected


@pytest.mark.parametrize(
    ("current", "previous", "persistence", "crowding"),
    [
        (classification_state(breadth_pct=None), classification_state(), "unavailable", "unavailable"),
        (classification_state(breadth_pct=55, activity_percentile=60), classification_state(rank=10, activity_percentile=60), "high", "low"),
        (classification_state(breadth_pct=55, activity_percentile=60), classification_state(rank=25, breadth_pct=40, activity_percentile=55), "medium", "low"),
        (classification_state(rank=1, breadth_pct=49, activity_percentile=90), classification_state(), "low", "high"),
        (classification_state(rank=2, breadth_pct=50, activity_percentile=75), classification_state(), "medium", "medium"),
        (classification_state(rank=2, breadth_pct=50, activity_percentile=74.999), classification_state(), "medium", "low"),
    ],
)
def test_classify_sector_persistence_and_crowding_boundaries(
    current: dict[str, object], previous: dict[str, object], persistence: str, crowding: str
) -> None:
    result = classify_sector(current, previous)
    assert result.persistence == persistence
    assert result.crowding_risk == crowding


@pytest.mark.parametrize(
    "frame",
    [
        sector_frame([{"name": "A", "change_pct": 1}]),
        sector_frame([{"sector_type": "other", "name": "A", "change_pct": 1}]),
        sector_frame([{"sector_type": "industry", "name": "", "change_pct": 1}]),
        sector_frame([{"sector_type": "industry", "name": "A", "change_pct": float("nan")}]),
        sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1}, {"sector_type": "industry", "name": "A", "change_pct": 2}]),
        sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1, "advance_count": 1, "decline_count": -1}]),
        sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1, "turnover_rate": float("inf")}]),
        sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1, "leader_name": ""}]),
    ],
)
def test_analyze_sectors_rejects_invalid_frames(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        analyze_sectors(frame, previous={}, observed_at=OBSERVED_AT)


@pytest.mark.parametrize("observed_at", [datetime(2026, 9, 6), "2026-09-06"])
@pytest.mark.parametrize("limit", [0, -1, 1.0, True])
def test_analyze_sectors_rejects_invalid_control_inputs(observed_at: object, limit: object) -> None:
    frame = sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1}])
    with pytest.raises(ValueError):
        analyze_sectors(frame, previous={}, observed_at=observed_at, limit=limit)  # type: ignore[arg-type]
