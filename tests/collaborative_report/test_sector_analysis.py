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
    assert get_type_hints(SectorRow) == {
        "sector_type": str, "name": str, "rank": int, "change_pct": float,
        "breadth_pct": float | None, "activity_percentile": float | None,
        "leader_name": str | None, "leader_code": str | None, "leader_change_pct": float | None,
        "rotation": str, "persistence": str, "crowding_risk": str,
    }
    assert get_type_hints(SectorAnalysis) == {
        "strongest": tuple[SectorRow, ...], "weakest": tuple[SectorRow, ...],
        "watch": tuple[SectorRow, ...], "valid_count": int, "warnings": tuple[str, ...],
    }
    assert get_type_hints(SectorClassification) == {
        "rotation": str, "persistence": str, "crowding_risk": str,
    }
    result = SectorClassification("continuing", "medium", "low")
    with pytest.raises(FrozenInstanceError):
        result.rotation = "retreating"  # type: ignore[misc]
    row = SectorRow("industry", "A", 1, 1.0, None, None, None, None, None, "first_observation", "unavailable", "unavailable")
    analysis = SectorAnalysis((row,), (), (), 1)
    with pytest.raises(FrozenInstanceError):
        row.rank = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        analysis.valid_count = 2  # type: ignore[misc]


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


def test_analyze_sectors_is_permutation_invariant_for_mixed_type_ties_at_bucket_boundary() -> None:
    rows = [
        {"sector_type": "industry", "name": "Shared", "change_pct": 1},
        {"sector_type": "concept", "name": "Shared", "change_pct": 1},
        {"sector_type": "concept", "name": "Alpha", "change_pct": 2},
        {"sector_type": "industry", "name": "Zulu", "change_pct": 0},
    ]

    first = analyze_sectors(sector_frame(rows), previous=(), observed_at=OBSERVED_AT, limit=2)
    second = analyze_sectors(sector_frame(list(reversed(rows))), previous=(), observed_at=OBSERVED_AT, limit=2)

    expected_strongest = [("concept", "Alpha"), ("concept", "Shared")]
    expected_weakest = [("industry", "Zulu"), ("industry", "Shared")]
    assert [(row.sector_type, row.name) for row in first.strongest] == expected_strongest
    assert [(row.sector_type, row.name) for row in first.weakest] == expected_weakest
    assert first.strongest == second.strongest
    assert first.weakest == second.weakest


def test_activity_percentiles_are_tie_aware_and_single_member_groups_are_one_hundred() -> None:
    frame = sector_frame([
        {"sector_type": "industry", "name": "A", "change_pct": 4, "turnover_rate": 10},
        {"sector_type": "industry", "name": "B", "change_pct": 3, "turnover_rate": 20},
        {"sector_type": "industry", "name": "C", "change_pct": 2, "turnover_rate": 20},
        {"sector_type": "industry", "name": "D", "change_pct": 1, "turnover_rate": 30},
        {"sector_type": "concept", "name": "Only", "change_pct": 0, "turnover_rate": 5},
    ])

    result = analyze_sectors(frame, previous=(), observed_at=OBSERVED_AT)
    activity = {row.name: row.activity_percentile for row in (*result.strongest, *result.weakest)}

    assert activity == {"A": 0.0, "B": 50.0, "C": 50.0, "D": 100.0, "Only": 100.0}


def test_analyze_sectors_distinguishes_no_snapshot_from_missing_prior_sector_and_never_mutates_prior() -> None:
    frame = sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1}])
    previous = {("industry", "Other"): classification_state()}
    original_previous = {key: value.copy() for key, value in previous.items()}

    no_snapshot = analyze_sectors(frame, previous=(), observed_at=OBSERVED_AT)
    missing_sector = analyze_sectors(frame, previous=previous, observed_at=OBSERVED_AT)

    assert no_snapshot.strongest[0].rotation == "first_observation"
    assert missing_sector.strongest[0].rotation == "new_start"
    assert previous == original_previous


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
        (classification_state(rank=25, breadth_pct=49, universe_size=40), classification_state(rank=10), "diverging"),
        (classification_state(rank=25, change_pct=-1, universe_size=40), classification_state(rank=10), "retreating"),
        (classification_state(rank=10, change_pct=-1), classification_state(rank=10), "retreating"),
        (classification_state(rank=5, change_pct=-1), classification_state(rank=10), "retreating"),
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
        (classification_state(rank=2, breadth_pct=49, activity_percentile=90), classification_state(), "low", "high"),
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
        sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1, "advance_count": 0.5}]),
        sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1, "decline_count": 0.5}]),
    ],
)
def test_analyze_sectors_rejects_invalid_frames(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        analyze_sectors(frame, previous={}, observed_at=OBSERVED_AT)


def test_analyze_sectors_rejects_huge_integer_changes_as_value_errors() -> None:
    frame = pd.DataFrame(
        {"sector_type": ["industry"], "name": ["A"], "change_pct": [10**10000]},
        dtype=object,
    )

    with pytest.raises(ValueError):
        analyze_sectors(frame, previous={}, observed_at=OBSERVED_AT)


@pytest.mark.parametrize("counts", [{"advance_count": 3}, {"decline_count": 2}])
def test_analyze_sectors_treats_partial_breadth_counts_as_nullable(counts: dict[str, int]) -> None:
    frame = sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1, **counts}])

    result = analyze_sectors(frame, previous=(), observed_at=OBSERVED_AT)

    assert result.strongest[0].breadth_pct is None


def test_analyze_sectors_accepts_integral_float_breadth_counts() -> None:
    frame = sector_frame([
        {"sector_type": "industry", "name": "A", "change_pct": 1, "advance_count": 3.0, "decline_count": 2},
    ])

    result = analyze_sectors(frame, previous=(), observed_at=OBSERVED_AT)

    assert result.strongest[0].breadth_pct == 60.0


def test_missing_activity_is_unavailable_for_persistence_and_crowding() -> None:
    current = classification_state(activity_percentile=None)

    result = classify_sector(current, classification_state())

    assert result.persistence == "unavailable"
    assert result.crowding_risk == "unavailable"


def test_classify_sector_accepts_current_state_with_only_rank_and_change() -> None:
    result = classify_sector({"rank": 10, "change_pct": 1}, {})

    assert result == SectorClassification("new_start", "unavailable", "unavailable")


def test_negative_current_change_retreats_without_universe_size() -> None:
    result = classify_sector(
        {"rank": 10, "change_pct": -1},
        classification_state(rank=10),
    )

    assert result.rotation == "retreating"


def test_non_top_twenty_narrow_high_activity_sector_has_low_crowding_risk() -> None:
    result = classify_sector(
        classification_state(rank=21, breadth_pct=49, activity_percentile=90),
        classification_state(),
    )

    assert result.crowding_risk == "low"


@pytest.mark.parametrize("observed_at", [datetime(2026, 9, 6), "2026-09-06"])
def test_analyze_sectors_rejects_invalid_observed_at_with_valid_limit(observed_at: object) -> None:
    frame = sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1}])
    with pytest.raises(ValueError):
        analyze_sectors(frame, previous={}, observed_at=observed_at, limit=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [0, -1, 1.0, True])
def test_analyze_sectors_rejects_invalid_limit_with_valid_observed_at(limit: object) -> None:
    frame = sector_frame([{"sector_type": "industry", "name": "A", "change_pct": 1}])
    with pytest.raises(ValueError):
        analyze_sectors(frame, previous={}, observed_at=OBSERVED_AT, limit=limit)  # type: ignore[arg-type]
