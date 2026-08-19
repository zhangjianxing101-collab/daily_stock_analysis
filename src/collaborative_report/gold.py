"""Deterministic 20/50-day moving-average analysis for gold."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd

from .backtest import BacktestSummary, _run_backtest, _validate_bars, _validate_parameters


@dataclass(frozen=True)
class GoldResult:
    direction: str
    signal: str
    risk_level: str
    watch_only: bool
    latest_close: float
    fast_ma: float
    slow_ma: float
    backtest: BacktestSummary
    risk_checks: Mapping[str, float | int]


def analyze_gold(
    bars: pd.DataFrame,
    capital: float = 20_000,
    fast: int = 20,
    slow: int = 50,
    fee_rate: float = 0.0003,
    sell_tax: float = 0.0005,
    slippage: float = 0.001,
    risk_fraction: float = 0.02,
    stop_fraction: float = 0.03,
) -> GoldResult:
    """Return the current gold regime and its no-lookahead strategy backtest."""

    if type(fast) is not int or type(slow) is not int or fast <= 1 or slow <= fast:
        raise ValueError("moving-average windows must satisfy 1 < fast < slow")
    params = _validate_parameters(
        capital=capital,
        horizon=None,
        fee_rate=fee_rate,
        sell_tax=sell_tax,
        slippage=slippage,
        risk_fraction=risk_fraction,
        stop_fraction=stop_fraction,
    )
    normalized_capital, _, fee_rate, sell_tax, slippage, risk_fraction, stop_fraction = params
    frame = _validate_bars(bars, min_rows=slow)
    close = frame["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    crosses_above = (close > fast_ma) & (close.shift(1) <= fast_ma.shift(1))
    crosses_below = (close < fast_ma) & (close.shift(1) >= fast_ma.shift(1))
    entries = (crosses_above & (fast_ma > slow_ma)).fillna(False)

    def close_exit_reason(index: int) -> str | None:
        if bool(crosses_below.iloc[index]):
            return "cross_below"
        if bool(fast_ma.iloc[index] < slow_ma.iloc[index]):
            return "fast_below_slow"
        return None

    backtest = _run_backtest(
        frame,
        entries,
        strategy="gold_ma20_ma50",
        capital=normalized_capital,
        horizon=None,
        fee_rate=fee_rate,
        sell_tax=sell_tax,
        slippage=slippage,
        risk_fraction=risk_fraction,
        stop_fraction=stop_fraction,
        lot_size=None,
        close_exit_reason=close_exit_reason,
    )

    latest_close = float(close.iloc[-1])
    latest_fast = float(fast_ma.iloc[-1])
    latest_slow = float(slow_ma.iloc[-1])
    if latest_close > latest_fast > latest_slow:
        direction = "bullish"
    elif latest_close < latest_fast < latest_slow:
        direction = "bearish"
    else:
        direction = "neutral"
    if bool(entries.iloc[-1]):
        signal = "buy"
    elif bool(crosses_below.iloc[-1]) or latest_fast < latest_slow:
        signal = "sell"
    else:
        signal = "hold"

    if backtest.max_drawdown > 0.10 or backtest.consecutive_losses >= 3:
        risk_level = "high"
    elif backtest.max_drawdown > 0.06 or backtest.consecutive_losses == 2:
        risk_level = "medium"
    else:
        risk_level = "low"
    watch_only = risk_level == "high" or backtest.trade_count < 10
    risk_checks: Mapping[str, float | int] = MappingProxyType(
        {
            "single_trade_risk_limit": risk_fraction,
            "stop_loss": stop_fraction,
            "drawdown_pause": 0.10,
            "consecutive_loss_pause": 3,
            "minimum_trade_sample": 10,
        }
    )
    return GoldResult(
        direction=direction,
        signal=signal,
        risk_level=risk_level,
        watch_only=watch_only,
        latest_close=latest_close,
        fast_ma=latest_fast,
        slow_ma=latest_slow,
        backtest=backtest,
        risk_checks=risk_checks,
    )
