from dataclasses import FrozenInstanceError
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

import src.collaborative_report.gold as gold_module
from src.collaborative_report.backtest import BacktestSummary
from src.collaborative_report.gold import GoldResult, analyze_gold


def gold_bars(*, tail: list[tuple[float, float, float, float]] | None = None) -> pd.DataFrame:
    closes = [10.0] * 30 + [11.0] * 19 + [10.0, 12.0]
    rows = [(close, close + 0.2, close - 0.2, close) for close in closes]
    rows.extend(tail or [(12.0, 12.4, 11.9, 12.2)])
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=len(rows)),
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": 100.0,
        }
    )


def test_gold_contract_and_exact_risk_checks() -> None:
    result = analyze_gold(gold_bars())

    assert tuple(get_type_hints(GoldResult)) == (
        "direction",
        "signal",
        "risk_level",
        "watch_only",
        "latest_close",
        "fast_ma",
        "slow_ma",
        "backtest",
        "risk_checks",
    )
    assert result.risk_checks == {
        "single_trade_risk_limit": 0.02,
        "stop_loss": 0.03,
        "drawdown_pause": 0.10,
        "consecutive_loss_pause": 3,
        "minimum_trade_sample": 10,
    }
    with pytest.raises(FrozenInstanceError):
        result.signal = "hold"  # type: ignore[misc]


def test_gold_crosses_above_twenty_only_when_twenty_is_above_fifty() -> None:
    result = analyze_gold(gold_bars(), fee_rate=0, sell_tax=0, slippage=0)

    assert result.backtest.trade_count == 1
    assert result.backtest.trades[0].signal_date == pd.Timestamp("2025-03-13")
    assert result.backtest.trades[0].entry_date == pd.Timestamp("2025-03-14")
    assert result.backtest.trades[0].shares == pytest.approx(20_000 * 0.02 / (12.0 * 0.03))

    downtrend = gold_bars()
    downtrend.loc[:, "close"] = np.linspace(20.0, 10.0, len(downtrend))
    downtrend.loc[:, "open"] = downtrend["close"]
    downtrend.loc[:, "high"] = downtrend["close"] + 0.2
    downtrend.loc[:, "low"] = downtrend["close"] - 0.2
    assert analyze_gold(downtrend).backtest.trade_count == 0


def test_gold_three_percent_gap_and_intraday_stops() -> None:
    gap = gold_bars(tail=[(12.0, 12.2, 11.9, 12.0), (11.0, 11.3, 10.9, 11.2)])
    intraday = gold_bars(tail=[(12.0, 12.2, 11.9, 12.0), (12.0, 12.1, 11.5, 11.8)])

    gap_trade = analyze_gold(gap, fee_rate=0, sell_tax=0, slippage=0).backtest.trades[0]
    intraday_trade = analyze_gold(intraday, fee_rate=0, sell_tax=0, slippage=0).backtest.trades[0]

    assert (gap_trade.exit_price, gap_trade.exit_reason) == (11.0, "stop_gap")
    assert (intraday_trade.exit_price, intraday_trade.exit_reason) == (pytest.approx(11.64), "stop")


def test_current_gold_direction_signal_and_watch_only_sample_rule() -> None:
    result = analyze_gold(gold_bars())

    assert result.direction == "bullish"
    assert result.signal == "hold"
    assert result.risk_level == "low"
    assert result.watch_only is True


@pytest.mark.parametrize(
    ("drawdown", "losses", "trades", "risk_level", "watch_only"),
    [
        (0.06, 1, 10, "low", False),
        (0.0600001, 1, 10, "medium", False),
        (0.10, 2, 10, "medium", False),
        (0.1000001, 0, 10, "high", True),
        (0.0, 3, 10, "high", True),
        (0.0, 0, 9, "low", True),
    ],
)
def test_gold_risk_and_pause_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    drawdown: float,
    losses: int,
    trades: int,
    risk_level: str,
    watch_only: bool,
) -> None:
    frame = gold_bars()
    summary = BacktestSummary(
        strategy="gold_ma20_ma50",
        start_date=frame.iloc[0]["date"],
        end_date=frame.iloc[-1]["date"],
        initial_capital=20_000,
        final_equity=20_000,
        total_return=0,
        max_drawdown=drawdown,
        trade_count=trades,
        win_rate=0,
        consecutive_losses=losses,
        trades=(),
    )
    monkeypatch.setattr(gold_module, "_run_backtest", lambda *args, **kwargs: summary)

    result = analyze_gold(frame)

    assert result.risk_level == risk_level
    assert result.watch_only is watch_only


def test_gold_rejects_insufficient_or_invalid_bars() -> None:
    with pytest.raises(ValueError):
        analyze_gold(gold_bars().iloc[:49])

    duplicated = pd.concat([gold_bars(), gold_bars().iloc[[-1]]], ignore_index=True)
    with pytest.raises(ValueError):
        analyze_gold(duplicated)
