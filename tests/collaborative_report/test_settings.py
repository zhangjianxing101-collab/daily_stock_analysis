import os
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import MappingProxyType
from typing import get_type_hints
from unittest.mock import patch

import pytest

from src.collaborative_report.models import Candidate, ModuleResult, Position, ReportMode
from src.collaborative_report.settings import CollaborativeSettings


VALID_PORTFOLIO = '[{"code":"600000","quantity":500,"cost_price":10.25}]'


def load_settings(**overrides: str) -> CollaborativeSettings:
    environment = {"COLLAB_PORTFOLIO_JSON": VALID_PORTFOLIO, **overrides}
    with patch.dict(os.environ, environment, clear=True):
        return CollaborativeSettings.from_env()


def test_from_env_parses_required_portfolio_and_defaults() -> None:
    settings = load_settings(COLLAB_CAPITAL_CNY="20000")

    assert settings.capital_cny == 20000
    assert settings.positions[0].code == "600000"
    assert settings.positions[0].quantity == 500
    assert settings.positions[0].cost_price == 10.25
    assert settings.risk_fraction == 0.02
    assert settings.short_limit == 5
    assert settings.swing_limit == 5
    assert settings.screen_prefilter == 120


def test_capital_defaults_to_twenty_thousand() -> None:
    assert load_settings().capital_cny == 20000


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "[]",
        '[{"code":"bad"}]',
    ],
)
def test_invalid_required_portfolio_payload_raises_value_error(payload: str) -> None:
    with pytest.raises(ValueError):
        load_settings(COLLAB_PORTFOLIO_JSON=payload)


@pytest.mark.parametrize("payload", [None, "", "{}", "null"])
def test_portfolio_must_be_a_required_non_empty_json_list(payload: str | None) -> None:
    environment = {} if payload is None else {"COLLAB_PORTFOLIO_JSON": payload}
    with patch.dict(os.environ, environment, clear=True), pytest.raises(ValueError):
        CollaborativeSettings.from_env()


@pytest.mark.parametrize(
    "position",
    [
        {"code": "60000", "quantity": 500, "cost_price": 10.25},
        {"code": "6000000", "quantity": 500, "cost_price": 10.25},
        {"code": "60000A", "quantity": 500, "cost_price": 10.25},
        {"code": 600000, "quantity": 500, "cost_price": 10.25},
        {"code": "600000", "quantity": 0, "cost_price": 10.25},
        {"code": "600000", "quantity": -100, "cost_price": 10.25},
        {"code": "600000", "quantity": 150, "cost_price": 10.25},
        {"code": "600000", "quantity": 500.0, "cost_price": 10.25},
        {"code": "600000", "quantity": 500, "cost_price": 0},
        {"code": "600000", "quantity": 500, "cost_price": -1},
        {"code": "600000", "quantity": 500, "cost_price": "10.25"},
    ],
)
def test_portfolio_position_validation(position: dict[str, object]) -> None:
    import json

    with pytest.raises(ValueError):
        load_settings(COLLAB_PORTFOLIO_JSON=json.dumps([position]))


@pytest.mark.parametrize("capital", ["0", "-1", "not-a-number"])
def test_capital_must_be_positive_number(capital: str) -> None:
    with pytest.raises(ValueError):
        load_settings(COLLAB_CAPITAL_CNY=capital)


@pytest.mark.parametrize("risk_fraction", ["0", "-0.01", "0.020001", "not-a-number"])
def test_risk_fraction_must_be_greater_than_zero_and_at_most_two_percent(risk_fraction: str) -> None:
    with pytest.raises(ValueError):
        load_settings(COLLAB_RISK_FRACTION=risk_fraction)


@pytest.mark.parametrize("risk_fraction", ["0.000001", "0.02"])
def test_risk_fraction_accepts_boundaries(risk_fraction: str) -> None:
    assert load_settings(COLLAB_RISK_FRACTION=risk_fraction).risk_fraction == float(risk_fraction)


@pytest.mark.parametrize("variable", ["COLLAB_SHORT_LIMIT", "COLLAB_SWING_LIMIT"])
@pytest.mark.parametrize("value", ["0", "11", "1.5", "not-a-number"])
def test_candidate_limits_must_be_inclusive_integers_from_one_to_ten(variable: str, value: str) -> None:
    with pytest.raises(ValueError):
        load_settings(**{variable: value})


@pytest.mark.parametrize("variable", ["COLLAB_SHORT_LIMIT", "COLLAB_SWING_LIMIT"])
@pytest.mark.parametrize("value", ["1", "10"])
def test_candidate_limits_accept_boundaries(variable: str, value: str) -> None:
    settings = load_settings(**{variable: value})
    assert getattr(settings, variable.removeprefix("COLLAB_").lower()) == int(value)


@pytest.mark.parametrize("value", ["19", "301", "20.5", "not-a-number"])
def test_screen_prefilter_must_be_inclusive_integer_from_twenty_to_three_hundred(value: str) -> None:
    with pytest.raises(ValueError):
        load_settings(COLLAB_SCREEN_PREFILTER=value)


@pytest.mark.parametrize("value", ["20", "300"])
def test_screen_prefilter_accepts_boundaries(value: str) -> None:
    assert load_settings(COLLAB_SCREEN_PREFILTER=value).screen_prefilter == int(value)


def test_settings_do_not_include_email_or_ai_configuration() -> None:
    settings = load_settings(EMAIL_TO="analyst@example.com", OPENAI_API_KEY="secret")

    assert not hasattr(settings, "email_to")
    assert not hasattr(settings, "openai_api_key")


def test_validation_errors_do_not_echo_environment_values() -> None:
    secret_payload = "invalid-secret-portfolio-content"

    with pytest.raises(ValueError) as error:
        load_settings(COLLAB_PORTFOLIO_JSON=secret_payload)

    assert secret_payload not in str(error.value)


def test_shared_models_are_frozen_and_expose_the_public_contracts() -> None:
    observed_at = datetime(2026, 8, 19, tzinfo=timezone.utc)
    position = Position(code="600000", quantity=500, cost_price=10.25)
    candidate = Candidate(
        code="600000",
        name="Pudong Development Bank",
        horizon="short",
        score=88.5,
        close=10.5,
        trigger="break above 10.60 with volume confirmation",
        stop_price=10.0,
        target_price=11.4,
        matched_rules=("volume_breakout",),
        observed_at=observed_at,
        source="market_feed",
    )
    payload = MappingProxyType({"candidate": candidate})
    result = ModuleResult(name="screening", status="ok", observed_at=observed_at, payload=payload)

    assert ReportMode.PREMARKET.value == "premarket"
    assert ReportMode.POSTMARKET.value == "postmarket"
    assert get_type_hints(Candidate)["trigger"] is str
    assert candidate.warning == ""
    assert result.warnings == ()
    with pytest.raises(FrozenInstanceError):
        position.quantity = 600  # type: ignore[misc]
