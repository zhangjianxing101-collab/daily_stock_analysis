"""Deterministic portfolio risk controls for collaborative reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext

from .models import Position


_CONCENTRATION_WARNING = "单只持仓集中度超过40%，请控制仓位"


@dataclass(frozen=True)
class PositionRisk:
    code: str
    quantity: int
    cost_price: float
    current_price: float
    cost_value: float
    market_value: float
    unrealized_pnl: float
    unrealized_return: float
    cost_concentration: float
    market_concentration: float
    concentration_label: str
    warnings: tuple[str, ...]


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _finite_calculation(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _floor_decimal_ratio(numerator: Decimal, denominator: Decimal) -> int:
    numerator_value, numerator_scale = numerator.as_integer_ratio()
    denominator_value, denominator_scale = denominator.as_integer_ratio()
    return (numerator_value * denominator_scale) // (numerator_scale * denominator_value)


def evaluate_position(position: Position, current_price: float, capital: float) -> PositionRisk:
    """Calculate valuation, return, and concentration for one position."""

    if type(position.quantity) is not int or position.quantity <= 0:
        raise ValueError("position quantity must be a positive integer")
    cost_price = _finite_number(position.cost_price, "position cost_price")
    normalized_current_price = _finite_number(current_price, "current_price")
    normalized_capital = _finite_number(capital, "capital")
    if cost_price <= 0:
        raise ValueError("position cost_price must be positive")
    if normalized_current_price <= 0:
        raise ValueError("current_price must be positive")
    if normalized_capital <= 0:
        raise ValueError("capital must be positive")

    try:
        quantity = Decimal(position.quantity)
        decimal_cost_price = _decimal(position.cost_price)
        decimal_current_price = _decimal(current_price)
        decimal_capital = _decimal(capital)
        precision = max(50, len(str(position.quantity)) + 40)
        with localcontext() as context:
            context.prec = precision
            decimal_cost_value = quantity * decimal_cost_price
            decimal_market_value = quantity * decimal_current_price
            decimal_unrealized_pnl = decimal_market_value - decimal_cost_value
            decimal_unrealized_return = decimal_unrealized_pnl / decimal_cost_value
            decimal_cost_concentration = decimal_cost_value / decimal_capital
            decimal_market_concentration = decimal_market_value / decimal_capital
            concentration_value = max(decimal_cost_value, decimal_market_value)
            normal_limit = decimal_capital * Decimal("0.40")
            elevated_limit = decimal_capital * Decimal("0.60")

        cost_value = _finite_calculation(float(decimal_cost_value), "cost_value")
        market_value = _finite_calculation(float(decimal_market_value), "market_value")
        unrealized_pnl = _finite_calculation(float(decimal_unrealized_pnl), "unrealized_pnl")
        unrealized_return = _finite_calculation(float(decimal_unrealized_return), "unrealized_return")
        cost_concentration = _finite_calculation(float(decimal_cost_concentration), "cost_concentration")
        market_concentration = _finite_calculation(float(decimal_market_concentration), "market_concentration")
    except (DecimalException, OverflowError) as exc:
        raise ValueError("position risk calculations must be finite") from exc

    if concentration_value <= normal_limit:
        concentration_label = "正常"
        warnings: tuple[str, ...] = ()
    elif concentration_value <= elevated_limit:
        concentration_label = "集中度偏高"
        warnings = (_CONCENTRATION_WARNING,)
    else:
        concentration_label = "高风险集中"
        warnings = (_CONCENTRATION_WARNING,)

    return PositionRisk(
        code=position.code,
        quantity=position.quantity,
        cost_price=cost_price,
        current_price=normalized_current_price,
        cost_value=cost_value,
        market_value=market_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_return=unrealized_return,
        cost_concentration=cost_concentration,
        market_concentration=market_concentration,
        concentration_label=concentration_label,
        warnings=warnings,
    )


def suggested_board_lots(
    entry_price: float,
    stop_price: float,
    *,
    capital: float,
    available_cash: float,
    risk_fraction: float = 0.02,
    lot_size: int = 100,
) -> int:
    """Return a whole-lot share count bounded by stop risk and cash."""

    normalized_entry = _finite_number(entry_price, "entry_price")
    normalized_stop = _finite_number(stop_price, "stop_price")
    normalized_capital = _finite_number(capital, "capital")
    normalized_cash = _finite_number(available_cash, "available_cash")
    normalized_risk_fraction = _finite_number(risk_fraction, "risk_fraction")

    if normalized_entry <= 0:
        raise ValueError("entry_price must be positive")
    if normalized_capital <= 0:
        raise ValueError("capital must be positive")
    if normalized_cash < 0:
        raise ValueError("available_cash must be non-negative")
    if not 0 < normalized_risk_fraction <= 0.02:
        raise ValueError("risk_fraction must be greater than zero and at most 0.02")
    if type(lot_size) is not int or lot_size <= 0:
        raise ValueError("lot_size must be a positive integer")
    decimal_entry = _decimal(entry_price)
    decimal_stop = _decimal(stop_price)
    decimal_capital = _decimal(capital)
    decimal_cash = _decimal(available_cash)
    decimal_risk_fraction = _decimal(risk_fraction)
    if decimal_stop <= 0 or decimal_stop >= decimal_entry:
        return 0

    with localcontext() as context:
        context.prec = 50
        risk_per_share = decimal_entry - decimal_stop
        risk_budget = decimal_capital * decimal_risk_fraction
    risk_shares = _floor_decimal_ratio(risk_budget, risk_per_share)
    cash_shares = _floor_decimal_ratio(decimal_cash, decimal_entry)
    shares = (min(risk_shares, cash_shares) // lot_size) * lot_size
    return shares


def pause_new_entries(max_drawdown: float, consecutive_losses: int) -> bool:
    """Return whether portfolio loss controls require pausing new entries."""

    normalized_drawdown = _finite_number(max_drawdown, "max_drawdown")
    if normalized_drawdown < 0:
        raise ValueError("max_drawdown must be non-negative")
    if type(consecutive_losses) is not int or consecutive_losses < 0:
        raise ValueError("consecutive_losses must be a non-negative integer")
    return normalized_drawdown > 0.10 or consecutive_losses >= 3
