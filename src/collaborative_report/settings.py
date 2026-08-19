"""Environment settings owned by the collaborative report workflow."""

import json
import math
import os
from dataclasses import dataclass
from typing import Any

from .models import Position


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_position(value: Any) -> Position:
    if not isinstance(value, dict):
        raise ValueError("COLLAB_PORTFOLIO_JSON entries must be objects")

    code = value.get("code")
    quantity = value.get("quantity")
    cost_price = value.get("cost_price")
    if not isinstance(code, str) or len(code) != 6 or not code.isascii() or not code.isdigit():
        raise ValueError("portfolio code must contain exactly six digits")
    if type(quantity) is not int or quantity <= 0 or quantity % 100 != 0:
        raise ValueError("portfolio quantity must be a positive multiple of 100")
    if type(cost_price) not in (int, float) or not math.isfinite(cost_price) or cost_price <= 0:
        raise ValueError("portfolio cost_price must be a positive finite number")

    return Position(code=code, quantity=quantity, cost_price=float(cost_price))


def _parse_portfolio() -> tuple[Position, ...]:
    raw_value = os.environ.get("COLLAB_PORTFOLIO_JSON")
    if not raw_value:
        raise ValueError("COLLAB_PORTFOLIO_JSON is required")
    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("COLLAB_PORTFOLIO_JSON must be valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("COLLAB_PORTFOLIO_JSON must be a non-empty list")
    return tuple(_parse_position(item) for item in payload)


@dataclass(frozen=True)
class CollaborativeSettings:
    capital_cny: float
    portfolio: tuple[Position, ...]
    risk_fraction: float
    short_limit: int
    swing_limit: int
    screen_prefilter: int

    @classmethod
    def from_env(cls) -> "CollaborativeSettings":
        capital_cny = _env_float("COLLAB_CAPITAL_CNY", 20000)
        if capital_cny <= 0:
            raise ValueError("COLLAB_CAPITAL_CNY must be positive")

        risk_fraction = _env_float("COLLAB_RISK_FRACTION", 0.02)
        if not 0 < risk_fraction <= 0.02:
            raise ValueError("COLLAB_RISK_FRACTION must be greater than 0 and at most 0.02")

        short_limit = _env_int("COLLAB_SHORT_LIMIT", 5)
        if not 1 <= short_limit <= 10:
            raise ValueError("COLLAB_SHORT_LIMIT must be between 1 and 10")

        swing_limit = _env_int("COLLAB_SWING_LIMIT", 5)
        if not 1 <= swing_limit <= 10:
            raise ValueError("COLLAB_SWING_LIMIT must be between 1 and 10")

        screen_prefilter = _env_int("COLLAB_SCREEN_PREFILTER", 120)
        if not 20 <= screen_prefilter <= 300:
            raise ValueError("COLLAB_SCREEN_PREFILTER must be between 20 and 300")

        return cls(
            capital_cny=capital_cny,
            portfolio=_parse_portfolio(),
            risk_fraction=risk_fraction,
            short_limit=short_limit,
            swing_limit=swing_limit,
            screen_prefilter=screen_prefilter,
        )
