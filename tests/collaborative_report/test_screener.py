from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.collaborative_report.market_data import MarketDataset, normalize_a_share_snapshot
from src.collaborative_report.screener import ScreeningResult, prefilter_universe, screen_aggressive


OBSERVED_AT = datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc)


def snapshot_row(
    code: str,
    name: str = "测试股份",
    *,
    price: float = 12.0,
    change_pct: float = 3.0,
    volume_ratio: float = 1.8,
    turnover: float = 3.0,
    amount: float = 100_000_000.0,
    volume: float = 1_000_000.0,
) -> dict[str, object]:
    return {
        "code": code,
        "name": name,
        "price": price,
        "change_pct": change_pct,
        "volume_ratio": volume_ratio,
        "turnover": turnover,
        "amount": amount,
        "total_mv": 2_000_000_000.0,
        "volume": volume,
    }


def bars(
    kind: str = "breakout",
    *,
    current_volume: float | None = None,
    gap: bool = False,
) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-08-19", periods=60)
    if kind == "breakout":
        closes = np.concatenate([np.linspace(9.0, 10.3, 40), np.linspace(10.4, 11.0, 19), [12.0]])
        volumes = np.full(60, 100.0)
        volumes[-1] = 200.0 if current_volume is None else current_volume
    elif kind == "swing":
        closes = np.linspace(10.0, 13.0, 60)
        volumes = np.full(60, 100.0)
        volumes[-1] = 120.0 if current_volume is None else current_volume
    elif kind == "downtrend":
        closes = np.linspace(20.0, 10.0, 60)
        volumes = np.full(60, 100.0)
        volumes[-1] = 100.0 if current_volume is None else current_volume
    else:
        closes = np.full(60, 10.0)
        volumes = np.full(60, 100.0)
        volumes[-1] = 100.0 if current_volume is None else current_volume

    opens = closes - 0.05
    highs = closes + 0.15
    lows = closes - 0.15
    if gap:
        opens[-1] = closes[-2] + 5.0
        closes[-1] = opens[-1] + 0.1
        highs[-1] = closes[-1] + 0.1
        lows[-1] = opens[-1] - 0.1
    return pd.DataFrame(
        {"date": dates, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def test_breakout_has_exact_short_score_rules_and_risk_prices() -> None:
    snapshot = pd.DataFrame([snapshot_row("000001")])

    result = screen_aggressive(
        snapshot,
        {"000001": bars()},
        {"000001": "强势板块"},
        observed_at=OBSERVED_AT,
    )

    candidate = result.short_term[0]
    assert candidate.score == 100
    assert candidate.matched_rules == (
        "ma5>ma10>ma20",
        "close_breaks_20d_high",
        "volume_expansion>=1.5",
        "0<return_5d<15%",
        "leading_sector",
        "1<=turnover<=12",
    )
    assert candidate.horizon == "1-5个交易日"
    assert candidate.close == 12.0
    assert candidate.trigger == "放量突破前20日高点"
    assert candidate.stop_price == 11.28
    assert candidate.target_price == 13.08


def test_swing_has_exact_score_and_rules() -> None:
    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000002", price=13.0)]),
        {"000002": bars("swing")},
        {"000002": "强势板块"},
        observed_at=OBSERVED_AT,
    )

    candidate = result.swing[0]
    assert candidate.score == 100
    assert candidate.matched_rules == (
        "ma20>ma50且close>ma20",
        "3%<=return_20d<=25%",
        "0%<=close_above_ma20<=8%",
        "volume_expansion>=1.2",
        "leading_sector",
        "0.5<=turnover<=8",
    )
    assert candidate.horizon == "1-4周"
    assert candidate.trigger == "站稳MA20且MA20高于MA50"


@pytest.mark.parametrize(
    ("turnover", "short_points", "swing_points"),
    [(0.5, 0, 5), (1.0, 5, 5), (8.0, 5, 5), (12.0, 5, 0), (12.01, 0, 0)],
)
def test_turnover_score_boundaries(turnover: float, short_points: int, swing_points: int) -> None:
    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000003", price=13.0, turnover=turnover)]),
        {"000003": bars("swing")},
        {"000003": "强势板块"},
        observed_at=OBSERVED_AT,
    )

    assert result.short_term[0].score == 50 + short_points
    assert result.swing[0].score == 95 + swing_points


def test_return_and_volume_boundaries_are_inclusive_only_where_specified() -> None:
    frame = bars("flat", current_volume=150.0)
    frame.loc[frame.index[-6], "close"] = 10.0
    frame.loc[frame.index[-1], ["open", "close", "high", "low"]] = [10.95, 11.0, 11.1, 10.9]
    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000004", price=11.0)]),
        {"000004": frame},
        {},
        observed_at=OBSERVED_AT,
    )

    assert "volume_expansion>=1.5" in result.short_term[0].matched_rules
    assert "0<return_5d<15%" in result.short_term[0].matched_rules

    frame.loc[frame.index[-1], ["open", "close", "high", "low"]] = [11.45, 11.5, 11.6, 11.4]
    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000004", price=11.5)]),
        {"000004": frame},
        {},
        observed_at=OBSERVED_AT,
    )
    assert "0<return_5d<15%" not in result.short_term[0].matched_rules


@pytest.mark.parametrize("return_pct", [3.0, 25.0])
def test_swing_return_boundaries_are_inclusive(return_pct: float) -> None:
    frame = bars("flat", current_volume=120.0)
    current = 10.0 * (1 + return_pct / 100)
    frame.loc[frame.index[-1], ["open", "close", "high", "low"]] = [
        current - 0.05,
        current,
        current + 0.15,
        current - 0.15,
    ]

    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000004", price=current)]),
        {"000004": frame},
        {},
        observed_at=OBSERVED_AT,
    )

    assert "3%<=return_20d<=25%" in result.swing[0].matched_rules


def test_prior_high_excludes_current_bar() -> None:
    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000005")]),
        {"000005": bars()},
        {},
        observed_at=OBSERVED_AT,
    )

    assert "close_breaks_20d_high" in result.short_term[0].matched_rules


def test_normalized_akshare_snapshot_integrates_with_prefilter_and_screening() -> None:
    normalized = normalize_a_share_snapshot(
        pd.DataFrame(
            {
                "代码": ["000007"],
                "名称": ["测试股份"],
                "最新价": ["12.0"],
                "涨跌幅": ["3.0%"],
                "量比": ["1.8"],
                "换手率": ["3.0%"],
                "成交额": ["100,000,000"],
                "成交量": ["1,000,000"],
                "总市值": ["2,000,000,000"],
            }
        )
    )

    assert prefilter_universe(normalized, 5)["code"].tolist() == ["000007"]
    result = screen_aggressive(normalized, {"000007": bars()}, {}, observed_at=OBSERVED_AT)
    assert [candidate.code for candidate in result.short_term] == ["000007"]


def test_zero_score_history_is_omitted_from_both_pools() -> None:
    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000008", price=10.0, turnover=20.0)]),
        {"000008": bars("downtrend")},
        {},
        observed_at=OBSERVED_AT,
    )

    assert result.short_term == ()
    assert result.swing == ()


def test_weak_universe_does_not_fill_requested_limits() -> None:
    no_core = bars("flat", current_volume=200.0)
    no_core.loc[no_core.index[:40], ["open", "high", "low", "close"]] = [10.95, 11.15, 10.85, 11.0]
    no_core.loc[no_core.index[-10:-6], ["open", "high", "low", "close"]] = [11.95, 12.15, 11.85, 12.0]
    no_core.loc[no_core.index[-1], ["open", "high", "low", "close"]] = [10.45, 10.65, 10.35, 10.5]
    snapshot = pd.DataFrame(
        [
            snapshot_row("000080", price=12.0),
            snapshot_row("000081", price=13.0, turnover=0.5),
            snapshot_row("000082", price=10.5),
        ]
    )
    histories = {"000080": bars(), "000081": bars("swing"), "000082": no_core}

    result = screen_aggressive(
        snapshot,
        histories,
        {"000082": "强势板块"},
        short_limit=5,
        swing_limit=5,
        observed_at=OBSERVED_AT,
    )

    assert [candidate.code for candidate in result.short_term] == ["000080"]
    assert [candidate.code for candidate in result.swing] == ["000081", "000080"]


def test_atr_uses_previous_close_for_gap_true_range() -> None:
    frame = bars("flat", gap=True)
    result = screen_aggressive(
        pd.DataFrame([snapshot_row("000006", price=float(frame.iloc[-1]["close"]))]),
        {"000006": frame},
        {},
        observed_at=OBSERVED_AT,
    )

    candidate = result.short_term[0]
    expected_atr = (13 * 0.3 + 5.2) / 14
    assert candidate.stop_price == round(candidate.close - 2 * expected_atr, 2)
    assert candidate.target_price == round(candidate.close + 3 * expected_atr, 2)


def test_ineligible_snapshot_rows_are_excluded() -> None:
    rows = [
        snapshot_row("000010", volume=0),
        snapshot_row("000011", "ST风险"),
        snapshot_row("000012", "即将退市"),
        snapshot_row("000013", amount=49_999_999),
        snapshot_row("ABC123"),
        snapshot_row("000014", price=0),
        snapshot_row("000015", volume_ratio=np.nan),
        snapshot_row("000016"),
    ]
    result = screen_aggressive(
        pd.DataFrame(rows),
        {row["code"]: bars() for row in rows},
        {},
        observed_at=OBSERVED_AT,
    )

    assert [candidate.code for candidate in result.short_term] == ["000016"]


def test_snapshot_missing_volume_column_is_ineligible() -> None:
    snapshot = pd.DataFrame([snapshot_row("000017")]).drop(columns="volume")

    result = screen_aggressive(snapshot, {"000017": bars()}, {}, observed_at=OBSERVED_AT)

    assert result.short_term == ()
    assert result.swing == ()


@pytest.mark.parametrize("volume", [np.nan, 0.0, -1.0])
def test_nonfinite_or_nonpositive_snapshot_volume_is_ineligible(volume: float) -> None:
    snapshot = pd.DataFrame([snapshot_row("000018", volume=volume)])

    result = screen_aggressive(snapshot, {"000018": bars()}, {}, observed_at=OBSERVED_AT)

    assert result.short_term == ()
    assert result.swing == ()


def test_limit_up_is_never_actionable_and_only_fills_unused_capacity() -> None:
    snapshot = pd.DataFrame(
        [
            snapshot_row("000020", amount=200_000_000),
            snapshot_row("000021", change_pct=9.8, amount=300_000_000),
        ]
    )
    histories = {"000020": bars(), "000021": bars()}

    filled = screen_aggressive(snapshot, histories, {}, short_limit=1, swing_limit=1, observed_at=OBSERVED_AT)
    assert filled.short_term[0].code == "000020"
    assert "观望" not in filled.short_term[0].trigger

    fallback = screen_aggressive(
        snapshot.iloc[[1]],
        {"000021": bars()},
        {},
        observed_at=OBSERVED_AT,
    )
    assert fallback.short_term[0].trigger.startswith("观望")
    assert fallback.short_term[0].warning
    assert fallback.warnings


def test_sort_ties_limits_and_deduplicates_by_amount_then_code() -> None:
    snapshot = pd.DataFrame(
        [
            snapshot_row("000032", amount=200_000_000),
            snapshot_row("000031", amount=200_000_000),
            snapshot_row("000030", amount=300_000_000),
            snapshot_row("000030", amount=100_000_000),
        ]
    )
    histories = {code: bars() for code in ("000030", "000031", "000032")}

    result = screen_aggressive(snapshot, histories, {}, short_limit=2, swing_limit=1, observed_at=OBSERVED_AT)

    assert [candidate.code for candidate in result.short_term] == ["000030", "000031"]
    assert [candidate.code for candidate in result.swing] == ["000030"]


def test_missing_and_bad_histories_degrade_to_sanitized_warnings() -> None:
    snapshot = pd.DataFrame([snapshot_row("000040"), snapshot_row("000041"), snapshot_row("000042")])
    bad_atr = bars("flat")
    bad_atr.loc[:, ["open", "high", "low", "close"]] = 10.0
    incomplete = bars().iloc[:59]

    result = screen_aggressive(
        snapshot,
        {"000041": incomplete, "000042": bad_atr},
        {},
        observed_at=OBSERVED_AT,
    )

    assert result.short_term == ()
    assert result.swing == ()
    assert result.warnings == (
        "000040: history unavailable",
        "000041: history invalid",
        "000042: indicators incomplete",
    )


def test_market_dataset_metadata_and_plain_frame_metadata() -> None:
    history_time = datetime(2026, 8, 19, 7, 30, tzinfo=timezone.utc)
    snapshot = MarketDataset(pd.DataFrame([snapshot_row("000050"), snapshot_row("000051")]), "snapshot", OBSERVED_AT)
    histories = {
        "000050": MarketDataset(bars(), "provider-a", history_time),
        "000051": bars(),
    }

    result = screen_aggressive(snapshot, histories, {}, observed_at=OBSERVED_AT)
    candidates = {candidate.code: candidate for candidate in result.short_term}

    assert candidates["000050"].source == "provider-a"
    assert candidates["000050"].observed_at == history_time
    assert candidates["000051"].source == "validated_history"
    assert candidates["000051"].observed_at == OBSERVED_AT
    assert all(candidate.observed_at.utcoffset() is not None for candidate in candidates.values())


def test_plain_history_requires_explicit_observed_at() -> None:
    with pytest.raises(ValueError, match="^observed_at is required for DataFrame histories$"):
        screen_aggressive(pd.DataFrame([snapshot_row("000052")]), {"000052": bars()}, {})


def test_plain_history_is_repeatable_with_explicit_observed_at() -> None:
    snapshot = pd.DataFrame([snapshot_row("000053")])
    histories = {"000053": bars()}

    first = screen_aggressive(snapshot, histories, {}, observed_at=OBSERVED_AT)
    second = screen_aggressive(snapshot, histories, {}, observed_at=OBSERVED_AT)

    assert first == second
    assert first.short_term[0].observed_at == OBSERVED_AT


def test_market_dataset_history_uses_its_timestamp_without_observed_at_argument() -> None:
    history_time = datetime(2026, 8, 19, 7, 45, tzinfo=timezone.utc)
    history = MarketDataset(bars(), "provider-b", history_time)

    result = screen_aggressive(pd.DataFrame([snapshot_row("000054")]), {"000054": history}, {})

    assert result.short_term[0].observed_at == history_time
    assert result.short_term[0].source == "provider-b"


def test_prefilter_is_stable_capped_canonical_and_does_not_mutate() -> None:
    original = pd.DataFrame(
        [
            snapshot_row("000061", amount=100_000_000, turnover=2, volume_ratio=2, change_pct=-2),
            snapshot_row("000060", amount=100_000_000, turnover=2, volume_ratio=2, change_pct=2),
            snapshot_row("000062", amount=90_000_000, turnover=1, volume_ratio=1, change_pct=1),
        ]
    )
    before = original.copy(deep=True)

    selected = prefilter_universe(original, 2)

    pd.testing.assert_frame_equal(original, before)
    assert selected["code"].tolist() == ["000060", "000061"]
    assert len(selected) == 2
    assert list(selected.columns) == list(original.columns)


def test_screening_does_not_mutate_inputs_and_result_is_frozen() -> None:
    snapshot = pd.DataFrame([snapshot_row("000070")])
    history = bars()
    snapshot_before = snapshot.copy(deep=True)
    history_before = history.copy(deep=True)

    result = screen_aggressive(snapshot, {"000070": history}, {}, observed_at=OBSERVED_AT)

    pd.testing.assert_frame_equal(snapshot, snapshot_before)
    pd.testing.assert_frame_equal(history, history_before)
    assert isinstance(result, ScreeningResult)
    with pytest.raises(FrozenInstanceError):
        result.short_term = ()  # type: ignore[misc]


def test_naive_observed_at_is_rejected() -> None:
    with pytest.raises(ValueError, match="observed_at must be timezone-aware"):
        screen_aggressive(pd.DataFrame(), {}, {}, observed_at=datetime(2026, 8, 19))
