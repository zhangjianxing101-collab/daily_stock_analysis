"""Deterministic, no-lookahead backtests for collaborative report strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trade:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_fraction: float
    exit_reason: str


@dataclass(frozen=True)
class BacktestSummary:
    strategy: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_equity: float
    total_return: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    consecutive_losses: int
    trades: tuple[Trade, ...]


_REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _validate_parameters(
    *,
    capital: object,
    horizon: object | None,
    fee_rate: object,
    sell_tax: object,
    slippage: object,
    risk_fraction: object,
    stop_fraction: object,
) -> tuple[float, int | None, float, float, float, float, float]:
    normalized_capital = _finite_number(capital, "capital")
    normalized_fee = _finite_number(fee_rate, "fee_rate")
    normalized_tax = _finite_number(sell_tax, "sell_tax")
    normalized_slippage = _finite_number(slippage, "slippage")
    normalized_risk = _finite_number(risk_fraction, "risk_fraction")
    normalized_stop = _finite_number(stop_fraction, "stop_fraction")
    if normalized_capital <= 0:
        raise ValueError("capital must be positive")
    if not 0 <= normalized_fee < 1 or not 0 <= normalized_tax < 1 or normalized_fee + normalized_tax >= 1:
        raise ValueError("cost rates must be at least zero and less than one")
    if not 0 <= normalized_slippage < 1:
        raise ValueError("slippage must be at least zero and less than one")
    if not 0 < normalized_risk <= 0.02:
        raise ValueError("risk_fraction must be greater than zero and at most 0.02")
    if not 0 < normalized_stop < 1:
        raise ValueError("stop_fraction must be greater than zero and less than one")
    if horizon is not None and (type(horizon) is not int or horizon <= 0):
        raise ValueError("horizon must be a positive integer")
    return (
        normalized_capital,
        horizon,
        normalized_fee,
        normalized_tax,
        normalized_slippage,
        normalized_risk,
        normalized_stop,
    )


def _validate_bars(bars: pd.DataFrame, *, min_rows: int = 1) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame):
        raise ValueError("bars must be a DataFrame")
    if any(column not in bars.columns for column in _REQUIRED_COLUMNS):
        raise ValueError("bars missing required columns")
    if len(bars) < min_rows:
        raise ValueError("bars insufficient")

    frame = bars.loc[:, _REQUIRED_COLUMNS].copy(deep=True).reset_index(drop=True)
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    except (TypeError, ValueError, OverflowError):
        raise ValueError("bars contain invalid dates") from None
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("bars contain invalid or duplicate dates")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("bars must be strictly ascending")

    numeric_columns = ("open", "high", "low", "close", "volume")
    for column in numeric_columns:
        native_values = frame[column].map(
            lambda value: not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
        )
        if not native_values.all():
            raise ValueError("bars numeric values must be native int or float")
        frame[column] = frame[column].astype(float)
    values = frame.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("bars contain non-finite values")
    if (frame.loc[:, ("open", "high", "low", "close")] <= 0).to_numpy().any() or (frame["volume"] < 0).any():
        raise ValueError("bars contain impossible values")
    if (
        (frame["high"] < frame["low"]).any()
        or (frame["high"] < frame[["open", "close"]].max(axis=1)).any()
        or (frame["low"] > frame[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("bars contain impossible ohlc values")
    return frame


def _position_size(
    *,
    current_equity: float,
    available_cash: float,
    entry_price: float,
    fee_rate: float,
    sell_tax: float,
    slippage: float,
    risk_fraction: float,
    stop_fraction: float,
    lot_size: int | None,
) -> float:
    decimal_entry = Decimal(str(entry_price))
    decimal_fee = Decimal(str(fee_rate))
    decimal_tax = Decimal(str(sell_tax))
    decimal_slippage = Decimal(str(slippage))
    one = Decimal("1")
    entry_outflow = decimal_entry * (one + decimal_fee)
    stop_execution_price = decimal_entry * (one - Decimal(str(stop_fraction))) * (one - decimal_slippage)
    stop_proceeds = stop_execution_price * (one - decimal_fee - decimal_tax)
    planned_loss_per_unit = entry_outflow - stop_proceeds
    risk_capacity = Decimal(str(current_equity)) * Decimal(str(risk_fraction)) / planned_loss_per_unit
    cash_capacity = Decimal(str(available_cash)) / entry_outflow
    capacity = min(risk_capacity, cash_capacity)
    if lot_size is None:
        shares = float(capacity)
        while Decimal(str(shares)) > capacity:
            shares = math.nextafter(shares, 0.0)
        return max(shares, 0.0)
    whole_units = int(capacity.to_integral_value(rounding=ROUND_FLOOR))
    return float((whole_units // lot_size) * lot_size)


def _drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    maximum = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _worst_loss_streak(trades: list[Trade]) -> int:
    current = 0
    worst = 0
    for trade in trades:
        if trade.pnl < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def _exact_ratio_in_band(
    numerators: pd.Series,
    denominators: pd.Series,
    lower: Decimal,
    upper: Decimal,
    *,
    lower_inclusive: bool = True,
) -> pd.Series:
    matches: list[bool] = []
    one = Decimal("1")
    for numerator, denominator in zip(numerators, denominators):
        if pd.isna(numerator) or pd.isna(denominator):
            matches.append(False)
            continue
        ratio = Decimal(str(numerator)) / Decimal(str(denominator)) - one
        lower_matches = lower <= ratio if lower_inclusive else lower < ratio
        matches.append(lower_matches and ratio <= upper)
    return pd.Series(matches, index=numerators.index, dtype=bool)


def _run_backtest(
    frame: pd.DataFrame,
    entry_signals: pd.Series,
    *,
    strategy: str,
    capital: float,
    horizon: int | None,
    fee_rate: float,
    sell_tax: float,
    slippage: float,
    risk_fraction: float,
    stop_fraction: float,
    lot_size: int | None,
    close_exit_reason: Callable[[int], str | None] | None = None,
) -> BacktestSummary:
    cash = capital
    trades: list[Trade] = []
    equity_curve = [capital]
    cursor = 0
    row_count = len(frame)

    while cursor < row_count - 1:
        signal_indices = np.flatnonzero(entry_signals.iloc[cursor : row_count - 1].to_numpy(dtype=bool))
        if not len(signal_indices):
            break
        signal_index = cursor + int(signal_indices[0])
        entry_index = signal_index + 1
        entry_price = float(frame.iloc[entry_index]["open"]) * (1 + slippage)
        shares = _position_size(
            current_equity=cash,
            available_cash=cash,
            entry_price=entry_price,
            fee_rate=fee_rate,
            sell_tax=sell_tax,
            slippage=slippage,
            risk_fraction=risk_fraction,
            stop_fraction=stop_fraction,
            lot_size=lot_size,
        )
        if shares <= 0:
            cursor = entry_index
            continue

        buy_cost = shares * entry_price * (1 + fee_rate)
        cash -= buy_cost
        stop_price = entry_price * (1 - stop_fraction)
        exit_index: int | None = None
        exit_price = 0.0
        exit_reason = ""
        pending_reason: str | None = None

        for index in range(entry_index, row_count):
            row = frame.iloc[index]
            if pending_reason is not None:
                exit_index = index
                exit_price = float(row["open"]) * (1 - slippage)
                exit_reason = pending_reason
            elif float(row["open"]) <= stop_price:
                exit_index = index
                exit_price = float(row["open"]) * (1 - slippage)
                exit_reason = "stop_gap"
            elif float(row["low"]) <= stop_price:
                exit_index = index
                exit_price = stop_price * (1 - slippage)
                exit_reason = "stop"
            elif horizon is not None and index - entry_index + 1 >= horizon:
                exit_index = index
                exit_price = float(row["close"]) * (1 - slippage)
                exit_reason = "horizon"

            if exit_index is not None:
                proceeds = shares * exit_price * (1 - fee_rate - sell_tax)
                cash += proceeds
                equity_curve.append(cash)
                pnl = proceeds - buy_cost
                trades.append(
                    Trade(
                        signal_date=frame.iloc[signal_index]["date"],
                        entry_date=frame.iloc[entry_index]["date"],
                        exit_date=frame.iloc[exit_index]["date"],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        shares=shares,
                        pnl=pnl,
                        return_fraction=pnl / buy_cost,
                        exit_reason=exit_reason,
                    )
                )
                break

            equity_curve.append(cash + shares * float(row["close"]))
            if close_exit_reason is not None:
                pending_reason = close_exit_reason(index)

        if exit_index is None:
            exit_index = row_count - 1
            exit_price = float(frame.iloc[-1]["close"]) * (1 - slippage)
            proceeds = shares * exit_price * (1 - fee_rate - sell_tax)
            cash += proceeds
            equity_curve.append(cash)
            pnl = proceeds - buy_cost
            trades.append(
                Trade(
                    signal_date=frame.iloc[signal_index]["date"],
                    entry_date=frame.iloc[entry_index]["date"],
                    exit_date=frame.iloc[-1]["date"],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    pnl=pnl,
                    return_fraction=pnl / buy_cost,
                    exit_reason="final",
                )
            )
        cursor = exit_index

    wins = sum(trade.pnl > 0 for trade in trades)
    trade_count = len(trades)
    return BacktestSummary(
        strategy=strategy,
        start_date=frame.iloc[0]["date"],
        end_date=frame.iloc[-1]["date"],
        initial_capital=capital,
        final_equity=cash,
        total_return=cash / capital - 1,
        max_drawdown=_drawdown(equity_curve),
        trade_count=trade_count,
        win_rate=wins / trade_count if trade_count else 0.0,
        consecutive_losses=_worst_loss_streak(trades),
        trades=tuple(trades),
    )


def backtest_breakout(
    bars: pd.DataFrame,
    capital: float = 20_000,
    horizon: int = 5,
    fee_rate: float = 0.0003,
    sell_tax: float = 0.0005,
    slippage: float = 0.001,
    risk_fraction: float = 0.02,
    stop_fraction: float = 0.03,
) -> BacktestSummary:
    """Backtest MA5/10/20 prior-high breakouts using next-session fills."""

    params = _validate_parameters(
        capital=capital,
        horizon=horizon,
        fee_rate=fee_rate,
        sell_tax=sell_tax,
        slippage=slippage,
        risk_fraction=risk_fraction,
        stop_fraction=stop_fraction,
    )
    normalized_capital, normalized_horizon, fee_rate, sell_tax, slippage, risk_fraction, stop_fraction = params
    frame = _validate_bars(bars)
    close = frame["close"]
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    prior_high = close.rolling(20).max().shift(1)
    signals = ((ma5 > ma10) & (ma10 > ma20) & (close > prior_high)).fillna(False)
    return _run_backtest(
        frame,
        signals,
        strategy="breakout",
        capital=normalized_capital,
        horizon=normalized_horizon,
        fee_rate=fee_rate,
        sell_tax=sell_tax,
        slippage=slippage,
        risk_fraction=risk_fraction,
        stop_fraction=stop_fraction,
        lot_size=100,
    )


def backtest_swing(
    bars: pd.DataFrame,
    capital: float = 20_000,
    horizon: int = 20,
    fee_rate: float = 0.0003,
    sell_tax: float = 0.0005,
    slippage: float = 0.001,
    risk_fraction: float = 0.02,
    stop_fraction: float = 0.03,
) -> BacktestSummary:
    """Backtest the MA20/50 swing trend rule using next-session fills."""

    params = _validate_parameters(
        capital=capital,
        horizon=horizon,
        fee_rate=fee_rate,
        sell_tax=sell_tax,
        slippage=slippage,
        risk_fraction=risk_fraction,
        stop_fraction=stop_fraction,
    )
    normalized_capital, normalized_horizon, fee_rate, sell_tax, slippage, risk_fraction, stop_fraction = params
    frame = _validate_bars(bars)
    close = frame["close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    return_in_band = _exact_ratio_in_band(close, close.shift(20), Decimal("0.03"), Decimal("0.25"))
    bias_in_band = _exact_ratio_in_band(
        close,
        ma20,
        Decimal("0"),
        Decimal("0.08"),
        lower_inclusive=False,
    )
    signals = (
        (ma20 > ma50)
        & (close > ma20)
        & return_in_band
        & bias_in_band
    ).fillna(False)
    return _run_backtest(
        frame,
        signals,
        strategy="swing",
        capital=normalized_capital,
        horizon=normalized_horizon,
        fee_rate=fee_rate,
        sell_tax=sell_tax,
        slippage=slippage,
        risk_fraction=risk_fraction,
        stop_fraction=stop_fraction,
        lot_size=100,
    )
