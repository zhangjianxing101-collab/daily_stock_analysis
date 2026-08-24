"""Environment settings owned by the collaborative report workflow."""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .models import Position


DEFAULT_THS_BASE_URL = "https://fuyao.aicubes.cn"
DEFAULT_THS_TIMEOUT_SECONDS = 10.0
DEFAULT_THS_MAX_RETRIES = 2


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


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


def _ths_base_url() -> str:
    raw_value = os.environ.get("THS_BASE_URL", DEFAULT_THS_BASE_URL).strip().rstrip("/")
    parsed = urlparse(raw_value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
        raise ValueError("THS_BASE_URL must be an HTTPS origin")
    return raw_value


@dataclass(frozen=True)
class ThsSettings:
    """Runtime-only settings for the THS REST data provider."""

    api_key: str | None = field(repr=False)
    enabled: bool
    base_url: str
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_env(cls) -> "ThsSettings":
        api_key = os.environ.get("THS_API_KEY", "").strip() or None
        provider_enabled = _env_bool("THS_ENABLED", True)

        timeout_seconds = _env_float("THS_TIMEOUT_SECONDS", DEFAULT_THS_TIMEOUT_SECONDS)
        if not 0 < timeout_seconds <= 60:
            raise ValueError("THS_TIMEOUT_SECONDS must be greater than 0 and at most 60")

        max_retries = _env_int("THS_MAX_RETRIES", DEFAULT_THS_MAX_RETRIES)
        if not 0 <= max_retries <= 3:
            raise ValueError("THS_MAX_RETRIES must be between 0 and 3")

        return cls(
            api_key=api_key,
            enabled=provider_enabled and api_key is not None,
            base_url=_ths_base_url(),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )


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
    if type(cost_price) not in (int, float):
        raise ValueError("portfolio cost_price must be a positive finite number")
    try:
        normalized_cost_price = float(cost_price)
    except (OverflowError, ValueError) as exc:
        raise ValueError("portfolio cost_price must be a positive finite number") from exc
    if not math.isfinite(normalized_cost_price) or normalized_cost_price <= 0:
        raise ValueError("portfolio cost_price must be a positive finite number")

    return Position(code=code, quantity=quantity, cost_price=normalized_cost_price)


def _parse_positions() -> tuple[Position, ...]:
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
    positions: tuple[Position, ...]
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
            positions=_parse_positions(),
            risk_fraction=risk_fraction,
            short_limit=short_limit,
            swing_limit=swing_limit,
            screen_prefilter=screen_prefilter,
        )
