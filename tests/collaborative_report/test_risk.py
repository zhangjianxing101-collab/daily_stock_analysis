import math
from dataclasses import FrozenInstanceError
from typing import get_type_hints

import pytest

from src.collaborative_report.models import Position
from src.collaborative_report.risk import (
    PositionRisk,
    evaluate_position,
    pause_new_entries,
    suggested_board_lots,
)


def test_evaluate_position_returns_exact_values_for_sample_position() -> None:
    result = evaluate_position(Position(code="600000", quantity=500, cost_price=10), 12, 20_000)

    assert result == PositionRisk(
        code="600000",
        quantity=500,
        cost_price=10.0,
        current_price=12.0,
        cost_value=5_000.0,
        market_value=6_000.0,
        unrealized_pnl=1_000.0,
        unrealized_return=0.2,
        cost_concentration=0.25,
        market_concentration=0.3,
        concentration_label="正常",
        warnings=(),
    )


def test_position_risk_is_frozen_and_has_the_public_contract() -> None:
    result = evaluate_position(Position(code="600000", quantity=100, cost_price=10), 10, 20_000)

    assert tuple(get_type_hints(PositionRisk)) == (
        "code",
        "quantity",
        "cost_price",
        "current_price",
        "cost_value",
        "market_value",
        "unrealized_pnl",
        "unrealized_return",
        "cost_concentration",
        "market_concentration",
        "concentration_label",
        "warnings",
    )
    assert get_type_hints(PositionRisk)["warnings"] == tuple[str, ...]
    with pytest.raises(FrozenInstanceError):
        result.quantity = 200  # type: ignore[misc]


@pytest.mark.parametrize(
    ("current_price", "expected_label", "has_warning"),
    [
        (8.0, "正常", False),
        (8.0000000001, "集中度偏高", True),
        (12.0, "集中度偏高", True),
        (12.0000000001, "高风险集中", True),
    ],
)
def test_concentration_uses_inclusive_forty_and_sixty_percent_boundaries(
    current_price: float, expected_label: str, has_warning: bool
) -> None:
    result = evaluate_position(Position(code="600000", quantity=1_000, cost_price=8), current_price, 20_000)

    assert result.concentration_label == expected_label
    assert bool(result.warnings) is has_warning


def test_concentration_warning_is_fixed_for_both_non_normal_labels() -> None:
    elevated = evaluate_position(Position(code="600000", quantity=1_000, cost_price=10), 10, 20_000)
    high = evaluate_position(Position(code="600000", quantity=1_000, cost_price=10), 13, 20_000)

    assert elevated.warnings == high.warnings
    assert len(elevated.warnings) == 1


@pytest.mark.parametrize("current_price", [0, -1, float("nan"), float("inf"), float("-inf"), "12"])
def test_evaluate_position_rejects_invalid_current_price(current_price: object) -> None:
    with pytest.raises(ValueError):
        evaluate_position(Position(code="600000", quantity=500, cost_price=10), current_price, 20_000)  # type: ignore[arg-type]


@pytest.mark.parametrize("capital", [0, -1, float("nan"), float("inf"), float("-inf"), "20000"])
def test_evaluate_position_rejects_invalid_capital(capital: object) -> None:
    with pytest.raises(ValueError):
        evaluate_position(Position(code="600000", quantity=500, cost_price=10), 12, capital)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "position",
    [
        Position(code="600000", quantity=0, cost_price=10),
        Position(code="600000", quantity=-100, cost_price=10),
        Position(code="600000", quantity=100.0, cost_price=10),  # type: ignore[arg-type]
        Position(code="600000", quantity=True, cost_price=10),  # type: ignore[arg-type]
        Position(code="600000", quantity=100, cost_price=0),
        Position(code="600000", quantity=100, cost_price=float("nan")),
        Position(code="600000", quantity=100, cost_price=float("inf")),
    ],
)
def test_evaluate_position_defends_against_malformed_positions(position: Position) -> None:
    with pytest.raises(ValueError):
        evaluate_position(position, 12, 20_000)


def test_suggested_board_lots_matches_exact_example() -> None:
    assert suggested_board_lots(20, 19, capital=20_000, available_cash=10_000) == 400


def test_suggested_board_lots_returns_zero_for_invalid_price_geometry_or_capacity() -> None:
    assert suggested_board_lots(20, 20, capital=20_000, available_cash=10_000) == 0
    assert suggested_board_lots(20, 21, capital=20_000, available_cash=10_000) == 0
    assert suggested_board_lots(20, 0, capital=20_000, available_cash=10_000) == 0
    assert suggested_board_lots(20, -1, capital=20_000, available_cash=10_000) == 0
    assert suggested_board_lots(20, 19, capital=20_000, available_cash=1_999) == 0


def test_suggested_board_lots_obeys_cash_cap_and_lot_rounding() -> None:
    assert suggested_board_lots(20, 19, capital=20_000, available_cash=7_999) == 300
    assert suggested_board_lots(10, 9.5, capital=20_000, available_cash=3_550) == 300
    assert suggested_board_lots(10, 9, capital=20_000, available_cash=20_000, lot_size=300) == 300


def test_suggested_board_lots_never_crosses_floating_point_budget_edges() -> None:
    shares = suggested_board_lots(10.1, 9.1, capital=20_000, available_cash=100_000)

    assert shares % 100 == 0
    assert shares * (10.1 - 9.1) <= 20_000 * 0.02
    assert shares * 10.1 <= 100_000


def test_suggested_board_lots_uses_cash_cap_when_risk_capacity_overflows() -> None:
    stop_price = math.nextafter(1.0, 0.0)

    assert suggested_board_lots(1.0, stop_price, capital=1e308, available_cash=100) == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entry_price": 0},
        {"entry_price": -1},
        {"entry_price": float("nan")},
        {"entry_price": float("inf")},
        {"stop_price": float("nan")},
        {"stop_price": float("inf")},
        {"capital": 0},
        {"capital": -1},
        {"capital": float("nan")},
        {"available_cash": -1},
        {"available_cash": float("inf")},
        {"risk_fraction": 0},
        {"risk_fraction": -0.01},
        {"risk_fraction": 0.020001},
        {"risk_fraction": float("nan")},
        {"lot_size": 0},
        {"lot_size": -100},
        {"lot_size": 100.0},
        {"lot_size": True},
    ],
)
def test_suggested_board_lots_rejects_malformed_inputs(kwargs: dict[str, object]) -> None:
    arguments = {
        "entry_price": 20,
        "stop_price": 19,
        "capital": 20_000,
        "available_cash": 10_000,
        **kwargs,
    }
    with pytest.raises(ValueError):
        suggested_board_lots(**arguments)  # type: ignore[arg-type]


def test_pause_new_entries_has_exact_boundaries() -> None:
    assert pause_new_entries(0.10, 2) is False
    assert pause_new_entries(0.1000000001, 2) is True
    assert pause_new_entries(0, 2) is False
    assert pause_new_entries(0, 3) is True


@pytest.mark.parametrize(
    ("max_drawdown", "consecutive_losses"),
    [
        (-0.01, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        (0, -1),
        (0, 1.0),
        (0, True),
    ],
)
def test_pause_new_entries_rejects_invalid_inputs(max_drawdown: object, consecutive_losses: object) -> None:
    with pytest.raises(ValueError):
        pause_new_entries(max_drawdown, consecutive_losses)  # type: ignore[arg-type]
