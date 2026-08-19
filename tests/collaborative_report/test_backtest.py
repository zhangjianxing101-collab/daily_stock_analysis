import inspect
from dataclasses import FrozenInstanceError
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

import src.collaborative_report.backtest as backtest_module
from src.collaborative_report.backtest import BacktestSummary, Trade, _run_backtest, backtest_breakout, backtest_swing


def breakout_bars(*, tail: list[tuple[float, float, float, float]] | None = None) -> pd.DataFrame:
    closes = [10.0] * 20 + [12.0]
    rows = [(close, close + 0.2, close - 0.2, close) for close in closes]
    rows.extend(tail or [(10.0, 11.2, 9.9, 11.0), (11.0, 11.2, 10.8, 11.0)])
    dates = pd.bdate_range("2026-01-02", periods=len(rows))
    return pd.DataFrame(
        {
            "date": dates,
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": 100.0,
        }
    )


def bars_from_closes(closes: list[float] | np.ndarray) -> pd.DataFrame:
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=len(values)),
            "open": values,
            "high": values + 0.2,
            "low": values - 0.2,
            "close": values,
            "volume": 100.0,
        }
    )


def swing_boundary_bars(*, latest_close: float, return_base: float, ma20: float) -> pd.DataFrame:
    prior_closes = [ma20] * 18 + [20 * ma20 - latest_close - 18 * ma20]
    closes = [50.0] * 30 + [return_base] + prior_closes + [latest_close]
    return bars_from_closes(closes)


def _captured_swing_signal(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> bool:
    captured: dict[str, pd.Series] = {}

    def capture(_frame: pd.DataFrame, signals: pd.Series, **_kwargs) -> BacktestSummary:
        captured["signals"] = signals
        return BacktestSummary(
            strategy="swing",
            start_date=_frame.iloc[0]["date"],
            end_date=_frame.iloc[-1]["date"],
            initial_capital=20_000,
            final_equity=20_000,
            total_return=0,
            max_drawdown=0,
            trade_count=0,
            win_rate=0,
            consecutive_losses=0,
            trades=(),
        )

    monkeypatch.setattr(backtest_module, "_run_backtest", capture)
    backtest_swing(frame)
    return bool(captured["signals"].iloc[-1])


def test_exact_public_api_signatures_and_defaults() -> None:
    expected_backtest_defaults = {
        "bars": inspect.Parameter.empty,
        "capital": 20_000,
        "horizon": 5,
        "fee_rate": 0.0003,
        "sell_tax": 0.0005,
        "slippage": 0.001,
        "risk_fraction": 0.02,
        "stop_fraction": 0.03,
    }
    expected_swing_defaults = {**expected_backtest_defaults, "horizon": 20}

    for function, expected in (
        (backtest_breakout, expected_backtest_defaults),
        (backtest_swing, expected_swing_defaults),
    ):
        signature = inspect.signature(function)
        assert list(signature.parameters) == list(expected)
        assert {name: parameter.default for name, parameter in signature.parameters.items()} == expected
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in signature.parameters.values()
        )
        assert get_type_hints(function)["return"] is BacktestSummary


def test_public_contracts_are_frozen_and_ordered() -> None:
    assert tuple(get_type_hints(Trade)) == (
        "signal_date",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "shares",
        "pnl",
        "return_fraction",
        "exit_reason",
    )
    assert tuple(get_type_hints(BacktestSummary)) == (
        "strategy",
        "start_date",
        "end_date",
        "initial_capital",
        "final_equity",
        "total_return",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "consecutive_losses",
        "trades",
    )

    trade = backtest_breakout(breakout_bars(), horizon=1).trades[0]
    with pytest.raises(FrozenInstanceError):
        trade.shares = 0  # type: ignore[misc]
    summary = backtest_breakout(breakout_bars(), horizon=1)
    with pytest.raises(FrozenInstanceError):
        summary.trade_count = 0  # type: ignore[misc]


def test_breakout_enters_next_open_and_applies_exact_costs() -> None:
    summary = backtest_breakout(breakout_bars(), horizon=2)
    trade = summary.trades[0]

    assert trade.signal_date == pd.Timestamp("2026-01-30")
    assert trade.entry_date == pd.Timestamp("2026-02-02")
    assert trade.exit_date == pd.Timestamp("2026-02-03")
    assert trade.entry_price == pytest.approx(10.0 * 1.001)
    assert trade.exit_price == pytest.approx(11.0 * 0.999)
    assert trade.shares == 1_300
    buy_cost = trade.entry_price * trade.shares * 1.0003
    sell_proceeds = trade.exit_price * trade.shares * (1 - 0.0003 - 0.0005)
    assert trade.pnl == pytest.approx(sell_proceeds - buy_cost)
    assert trade.return_fraction == pytest.approx(trade.pnl / buy_cost)
    assert trade.exit_reason == "horizon"


def test_gap_stop_uses_adverse_open_and_intraday_stop_uses_stop_price() -> None:
    gap = breakout_bars(tail=[(10.0, 10.2, 9.9, 10.0), (9.0, 9.4, 8.8, 9.2)])
    intraday = breakout_bars(tail=[(10.0, 10.2, 9.9, 10.0), (10.0, 10.1, 9.5, 9.8)])

    gap_trade = backtest_breakout(gap, horizon=5, fee_rate=0, sell_tax=0, slippage=0).trades[0]
    intraday_trade = backtest_breakout(intraday, horizon=5, fee_rate=0, sell_tax=0, slippage=0).trades[0]

    assert (gap_trade.exit_price, gap_trade.exit_reason) == (9.0, "stop_gap")
    assert (intraday_trade.exit_price, intraday_trade.exit_reason) == (9.7, "stop")


def test_open_trade_is_liquidated_on_final_close_with_costs() -> None:
    frame = breakout_bars(tail=[(10.0, 10.6, 9.9, 10.5)])

    trade = backtest_breakout(frame, horizon=5).trades[0]

    assert trade.exit_date == frame.iloc[-1]["date"]
    assert trade.exit_price == pytest.approx(10.5 * 0.999)
    assert trade.exit_reason == "final"


def test_cash_constrained_sizing_reserves_entry_fee_and_rounds_to_board_lot() -> None:
    summary = backtest_breakout(
        breakout_bars(),
        capital=1_500,
        horizon=1,
        fee_rate=0.001,
        sell_tax=0,
        slippage=0,
        risk_fraction=0.02,
        stop_fraction=0.001,
    )

    trade = summary.trades[0]
    assert trade.shares == 100
    assert trade.shares * trade.entry_price * 1.001 <= 1_500
    assert 200 * trade.entry_price * 1.001 > 1_500


def test_mark_to_market_drawdown_includes_interim_adverse_close_before_horizon() -> None:
    frame = bars_from_closes([10.0, 10.0, 8.0, 10.0])
    frame.loc[:, "low"] = [9.8, 9.8, 7.8, 9.8]
    signals = pd.Series([True, False, False, False])

    summary = _run_backtest(
        frame,
        signals,
        strategy="test",
        capital=20_000,
        horizon=3,
        fee_rate=0,
        sell_tax=0,
        slippage=0,
        risk_fraction=0.02,
        stop_fraction=0.5,
        lot_size=None,
    )

    assert summary.final_equity == 20_000
    assert summary.total_return == 0
    assert summary.max_drawdown == pytest.approx(0.008)


def test_future_bar_mutation_cannot_change_trade_or_equity_through_fixed_prefix() -> None:
    frame = breakout_bars(
        tail=[
            (10.0, 10.7, 9.9, 10.5),
            (10.5, 10.7, 10.0, 10.2),
            (10.2, 11.2, 10.1, 11.0),
            *[(10.0, 10.2, 9.8, 10.0)] * 12,
        ]
    )
    prefix_end = 23
    mutated = frame.copy(deep=True)
    suffix = mutated.index > prefix_end
    mutated.loc[suffix, "open"] = np.linspace(30.0, 40.0, suffix.sum())
    mutated.loc[suffix, "close"] = mutated.loc[suffix, "open"] + 1.0
    mutated.loc[suffix, "high"] = mutated.loc[suffix, "close"] + 0.2
    mutated.loc[suffix, "low"] = mutated.loc[suffix, "open"] - 0.2

    original = backtest_breakout(frame, horizon=3, fee_rate=0, sell_tax=0, slippage=0)
    changed_future = backtest_breakout(mutated, horizon=3, fee_rate=0, sell_tax=0, slippage=0)

    assert original.trades[0] == changed_future.trades[0]
    assert original.trades[0].exit_date == frame.iloc[prefix_end]["date"]
    original_prefix_equity = original.initial_capital + sum(
        trade.pnl for trade in original.trades if trade.exit_date <= frame.iloc[prefix_end]["date"]
    )
    changed_prefix_equity = changed_future.initial_capital + sum(
        trade.pnl for trade in changed_future.trades if trade.exit_date <= frame.iloc[prefix_end]["date"]
    )
    assert changed_prefix_equity == pytest.approx(original_prefix_equity)


def test_overlapping_entry_signals_are_suppressed_until_position_exits() -> None:
    frame = bars_from_closes([10.0, 10.0, 10.1, 10.2, 10.3])
    signals = pd.Series([True, True, True, True, False])

    summary = _run_backtest(
        frame,
        signals,
        strategy="test",
        capital=20_000,
        horizon=4,
        fee_rate=0,
        sell_tax=0,
        slippage=0,
        risk_fraction=0.02,
        stop_fraction=0.03,
        lot_size=100,
    )

    assert summary.trade_count == 1
    assert summary.trades[0].signal_date == frame.iloc[0]["date"]
    assert summary.trades[0].entry_date == frame.iloc[1]["date"]
    assert summary.trades[0].exit_date == frame.iloc[4]["date"]
    assert summary.trades[0].exit_reason == "horizon"


def test_swing_uses_current_and_prior_data_only() -> None:
    closes = np.concatenate([np.linspace(8.0, 10.0, 50), np.linspace(10.1, 11.0, 20), [50.0]])
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=len(closes)),
            "open": closes,
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes,
            "volume": 100.0,
        }
    )

    before_future_spike = backtest_swing(frame.iloc[:-1], horizon=20)
    after_future_spike = backtest_swing(frame, horizon=20)

    assert before_future_spike.trades
    assert after_future_spike.trades[0].signal_date == before_future_spike.trades[0].signal_date
    assert after_future_spike.trades[0].entry_date == before_future_spike.trades[0].entry_date


@pytest.mark.parametrize("latest_close", [103.0, 125.0])
def test_swing_return_boundaries_are_exact_and_inclusive(
    monkeypatch: pytest.MonkeyPatch, latest_close: float
) -> None:
    assert _captured_swing_signal(
        monkeypatch,
        swing_boundary_bars(latest_close=latest_close, return_base=100.0, ma20=latest_close / 1.04),
    ) is True


@pytest.mark.parametrize("latest_close", [102.99999999999, 125.00000000001])
def test_swing_return_values_just_outside_exact_boundaries_are_rejected(
    monkeypatch: pytest.MonkeyPatch, latest_close: float
) -> None:
    assert _captured_swing_signal(
        monkeypatch,
        swing_boundary_bars(latest_close=latest_close, return_base=100.0, ma20=latest_close / 1.04),
    ) is False


@pytest.mark.parametrize("latest_close", [100.0, 108.0])
def test_swing_bias_boundaries_are_exact_and_inclusive(
    monkeypatch: pytest.MonkeyPatch, latest_close: float
) -> None:
    assert _captured_swing_signal(
        monkeypatch,
        swing_boundary_bars(latest_close=latest_close, return_base=90.0, ma20=100.0),
    ) is True


@pytest.mark.parametrize("latest_close", [99.99999999999, 108.00000000001])
def test_swing_bias_values_just_outside_exact_boundaries_are_rejected(
    monkeypatch: pytest.MonkeyPatch, latest_close: float
) -> None:
    assert _captured_swing_signal(
        monkeypatch,
        swing_boundary_bars(latest_close=latest_close, return_base=90.0, ma20=100.0),
    ) is False


def test_summary_tracks_mark_to_market_drawdown_and_worst_loss_streak() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-02", periods=6),
            "open": [10.0] * 6,
            "high": [10.2] * 6,
            "low": [9.75] * 6,
            "close": [10.0, 9.8, 10.0, 9.8, 10.0, 9.8],
            "volume": [100.0] * 6,
        }
    )
    signals = pd.Series([True, False, True, False, True, False])

    summary = _run_backtest(
        frame,
        signals,
        strategy="test",
        capital=20_000,
        horizon=1,
        fee_rate=0,
        sell_tax=0,
        slippage=0,
        risk_fraction=0.02,
        stop_fraction=0.03,
        lot_size=100,
    )

    assert summary.trade_count == 3
    assert summary.consecutive_losses == 3
    assert summary.win_rate == 0
    assert summary.final_equity == pytest.approx(19_220)
    assert summary.max_drawdown == pytest.approx(0.039)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.iloc[::-1],
        lambda frame: pd.concat([frame, frame.iloc[[-1]]], ignore_index=True),
        lambda frame: frame.assign(close=np.inf),
        lambda frame: frame.assign(high=frame["low"] - 1),
        lambda frame: frame.drop(columns="volume"),
    ],
)
def test_backtests_reject_invalid_bars(mutate) -> None:
    with pytest.raises(ValueError):
        backtest_breakout(mutate(breakout_bars()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capital": 0},
        {"horizon": 0},
        {"fee_rate": -0.1},
        {"sell_tax": float("nan")},
        {"slippage": 1},
        {"risk_fraction": 0.03},
        {"stop_fraction": 0},
    ],
)
def test_backtests_reject_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        backtest_breakout(breakout_bars(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bars", "kwargs"),
    [
        (lambda: breakout_bars().iloc[::-1], {}),
        (lambda: breakout_bars().assign(low=np.nan), {}),
        (lambda: breakout_bars().drop(columns="open"), {}),
        (breakout_bars, {"horizon": True}),
        (breakout_bars, {"capital": float("inf")}),
        (breakout_bars, {"risk_fraction": 0}),
    ],
)
def test_swing_rejects_invalid_bars_and_parameters(bars, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        backtest_swing(bars(), **kwargs)  # type: ignore[arg-type]
