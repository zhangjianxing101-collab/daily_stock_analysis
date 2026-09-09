import json
import runpy
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.collaborative_report.ai_bridge import enrich_codes
from src.collaborative_report.backtest import backtest_breakout, backtest_swing
from src.collaborative_report.gold import analyze_gold
from src.collaborative_report.market_data import MarketDataGateway, MarketDataset
from src.collaborative_report.models import Candidate, ModuleResult, Position, ReportMode
from src.collaborative_report.report import RenderedReport, render_report
from src.collaborative_report.risk import evaluate_position, suggested_board_lots
from src.collaborative_report.runner import (
    EXIT_FAILURE,
    EXIT_SUCCESS,
    SNAPSHOT_DAILY_CLOSE_TOLERANCE,
    FinalState,
    DeliveryState,
    DeliveryStateError,
    LocalDeliveryLedger,
    RunnerDependencies,
    classify_prior_candidates,
    default_dependencies,
    publish_report_artifacts,
    run_report,
    write_report_artifacts,
)
from src.collaborative_report.runner import (
    _PRIOR_SECTOR_MANIFEST_MAX_BYTES,
    _decision_summary,
    _report_delivery_blockers,
    _load_prior_sector_state,
    _market_payload,
    _production_mail_sender,
    _prior_sector_recap_modules,
    _redacted_manifest,
    _rerank_sector_ties,
    _sector_state,
    _trusted_sector_snapshot_timestamp,
)
from src.collaborative_report.sector_analysis import SectorAnalysis, SectorRow, analyze_sectors
from src.collaborative_report.screener import ScreeningResult, screen_aggressive
from src.collaborative_report.session import ReportSession, build_report_session, report_data_session
from src.collaborative_report.settings import CollaborativeSettings
from src.collaborative_report.ths_market_data import ThsApiResponse, ThsMarketDataClient


SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 19, 16, 30, tzinfo=SHANGHAI)


def test_sector_source_may_follow_report_start_when_received_later() -> None:
    report_start = datetime(2026, 8, 19, 15, 2, tzinfo=SHANGHAI)
    source = datetime(2026, 8, 19, 15, 5, tzinfo=SHANGHAI)
    received = datetime(2026, 8, 19, 15, 6, tzinfo=SHANGHAI)

    assert _trusted_sector_snapshot_timestamp(source, received, report_start) == source
PORTFOLIO_CODE = "600000"
CANDIDATE_CODE = "600001"
SECTOR_STATE_KEYS = (
    "sector_type", "name", "rank", "change_pct", "breadth_pct", "activity_percentile",
    "universe_size", "rotation", "persistence", "crowding_risk", "source_timestamp",
)


class InvalidOffsetTimezone(tzinfo):
    def utcoffset(self, dt):
        raise ValueError("invalid offset")

    def dst(self, dt):
        return None


def candidate(code: str = CANDIDATE_CODE, *, observed_at: datetime = NOW) -> Candidate:
    return Candidate(
        code=code,
        name="示例股份",
        horizon="1-5个交易日",
        score=85,
        close=10.5,
        trigger="放量突破前20日高点",
        stop_price=9.8,
        target_price=12,
        matched_rules=("close_breaks_20d_high",),
        observed_at=observed_at,
        source="fixture",
    )


def bars(*, high: float = 10.4, low: float = 10.0, close: float = 10.2) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-08-19", periods=80)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [10.1] * 80,
            "high": [10.3] * 80,
            "low": [9.9] * 80,
            "close": [10.1] * 80,
            "volume": [1_000_000] * 80,
        }
    )
    frame.loc[79, ["high", "low", "close"]] = [high, low, close]
    return frame


def dataset(frame: pd.DataFrame, source: str = "fixture") -> MarketDataset:
    return MarketDataset(frame=frame, source=source, observed_at=NOW)


def finalized_attempt(output_dir: Path, report_key: str = "2026-08-19-premarket"):
    return write_report_artifacts(
        output_dir,
        report_key=report_key,
        rendered=RenderedReport("subject", "html", "text"),
        manifest={"report_key": report_key, "final_state": "sent"},
    )


class FakeGateway:
    def __init__(self) -> None:
        self.snapshot = dataset(
            pd.DataFrame(
                [
                    {
                        "code": PORTFOLIO_CODE,
                        "name": "持仓股份",
                        "price": 10.2,
                        "change_pct": 1.0,
                        "volume_ratio": 1.2,
                        "turnover": 2.0,
                        "amount": 100_000_000,
                        "volume": 1_000_000,
                        "total_mv": 5_000_000_000,
                    },
                    {
                        "code": CANDIDATE_CODE,
                        "name": "示例股份",
                        "price": 10.5,
                        "change_pct": 2.0,
                        "volume_ratio": 1.8,
                        "turnover": 3.0,
                        "amount": 200_000_000,
                        "volume": 2_000_000,
                        "total_mv": 6_000_000_000,
                    },
                ]
            )
        )
        self.snapshot = replace(
            self.snapshot,
            source_timestamp=datetime(2026, 8, 19, 15, 5, tzinfo=SHANGHAI),
        )
        self.histories = {
            PORTFOLIO_CODE: dataset(bars()),
            CANDIDATE_CODE: dataset(bars(high=10.6, close=10.5)),
        }
        self.sector_snapshots = {
            "industry": replace(
                dataset(pd.DataFrame([
                    {
                        "sector_type": "industry", "name": "示例板块", "change_pct": 2.0,
                        "advance_count": 8, "decline_count": 2, "turnover_rate": 3.0,
                        "leader_name": "示例股份", "leader_code": CANDIDATE_CODE,
                        "leader_change_pct": 2.0,
                    },
                ]), "fixture.industry"),
                source_timestamp=datetime(2026, 8, 19, 15, 5, tzinfo=SHANGHAI),
            ),
            "concept": replace(
                dataset(pd.DataFrame([
                    {
                        "sector_type": "concept", "name": "示例概念", "change_pct": 1.0,
                        "advance_count": 6, "decline_count": 4, "turnover_rate": 2.0,
                        "leader_name": "示例股份", "leader_code": CANDIDATE_CODE,
                        "leader_change_pct": 2.0,
                    },
                ]), "fixture.concept"),
                source_timestamp=datetime(2026, 8, 19, 15, 5, tzinfo=SHANGHAI),
            ),
        }

    def get_a_share_snapshot(self):
        return self.snapshot

    def get_daily_bars(self, code, expected_session, *, days=160):
        return self.histories[code]

    def get_leading_sector_codes(self, limit=10):
        return dataset(pd.DataFrame([{"code": CANDIDATE_CODE, "sector": "示例板块"}]))

    def get_sector_snapshot(self, sector_type):
        return self.sector_snapshots[sector_type]

    def get_global_snapshot(self):
        return dataset(pd.DataFrame([{"symbol": "^GSPC", "change_pct": 0.5}]))

    def get_gold_bars(self):
        return dataset(bars())


@pytest.fixture
def settings() -> CollaborativeSettings:
    return CollaborativeSettings(
        capital_cny=20_000,
        positions=(Position(PORTFOLIO_CODE, 100, 10.0),),
        risk_fraction=0.02,
        short_limit=5,
        swing_limit=5,
        screen_prefilter=20,
        position_sizing=True,
    )


@pytest.fixture
def deps(settings) -> RunnerDependencies:
    gateway = FakeGateway()
    mailer = Mock(return_value=True)
    def render(mode, report_date, **kwargs):
        prefix = f"{kwargs['subject_prefix']} " if kwargs.get("subject_prefix") else ""
        return RenderedReport(f"{prefix}A股收盘日报 2026-08-19", "<html>private</html>", "private")

    renderer = Mock(side_effect=render)
    return RunnerDependencies(
        settings_loader=lambda: settings,
        session_builder=lambda mode, current_time, scheduled: ReportSession(
            mode=mode,
            now_shanghai=NOW,
            trading_date=date(2026, 8, 19),
            is_trading_day=True,
            report_key=f"2026-08-19-{mode.value}",
        ),
        clock=lambda: NOW,
        gateway=gateway,
        screener=lambda *args, **kwargs: ScreeningResult((candidate(),), ()),
        risk_evaluator=lambda position, price, capital: {"状态": "正常"},
        short_backtest=lambda frame, **kwargs: {"strategy": "short", "trade_count": 3},
        swing_backtest=lambda frame, **kwargs: {"strategy": "swing", "trade_count": 4},
        gold_analyzer=lambda frame, **kwargs: {"direction": "neutral"},
        ai_enricher=lambda codes, **kwargs: ModuleResult("ai", "ok", NOW, {"count": len(tuple(codes))}),
        renderer=renderer,
        mail_sender=mailer,
        data_session_resolver=lambda mode, report_date, generated_at: report_date,
    )


def test_non_trading_day_is_successful_skip_without_data_or_mail(tmp_path, deps) -> None:
    gateway = Mock()
    local = replace(
        deps,
        gateway=gateway,
        session_builder=lambda mode, current_time, scheduled: ReportSession(
            mode, NOW, NOW.date(), False, f"2026-08-19-{mode.value}"
        ),
    )

    result = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)

    assert result.exit_code == EXIT_SUCCESS
    assert result.final_state is FinalState.NON_TRADING_DAY_SKIP
    gateway.get_a_share_snapshot.assert_not_called()
    deps.mail_sender.assert_not_called()


@pytest.mark.parametrize("message", ["trading calendar unavailable", "outside delivery window"])
def test_calendar_and_delayed_scheduled_run_are_hard_failures(tmp_path, deps, message) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError(message)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, session_builder=fail),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code in {"calendar_unavailable", "outside_delivery_window"}
    deps.mail_sender.assert_not_called()


def test_incomplete_postmarket_data_has_a_distinct_safe_error_code(tmp_path, deps) -> None:
    def incomplete(*args, **kwargs):
        raise RuntimeError("report data session incomplete")

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, data_session_resolver=incomplete),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "report_data_incomplete"
    deps.mail_sender.assert_not_called()


def test_calendar_worker_timeout_stops_production_before_data_and_mail(tmp_path, deps) -> None:
    from src.collaborative_report import session as sessions

    gateway = Mock()
    local = replace(deps, session_builder=build_report_session, gateway=gateway)
    sessions._akshare_xshg_sessions_for_local_date.cache_clear()
    try:
        with (
            patch.object(sessions, "_xshg_calendar", side_effect=RuntimeError("unavailable")),
            patch.object(sessions.subprocess, "run", side_effect=subprocess.TimeoutExpired("calendar", 30)),
        ):
            result = run_report(ReportMode.POSTMARKET, deps=local, output_dir=tmp_path)
    finally:
        sessions._akshare_xshg_sessions_for_local_date.cache_clear()

    assert result.exit_code == EXIT_FAILURE
    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "calendar_unavailable"
    gateway.get_a_share_snapshot.assert_not_called()
    deps.mail_sender.assert_not_called()


def test_incomplete_supplement_marks_market_and_screening_partial(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot.frame.attrs.update(quarantined_row_count=1, screening_complete_count=1)
    gateway.snapshot = replace(gateway.snapshot, warnings=("snapshot_screening_fields_incomplete",))
    result = run_report(ReportMode.PREMARKET, deps=replace(deps, gateway=gateway),
                        force=True, preview_only=True, output_dir=tmp_path)
    assert result.modules["market"].status == "partial"
    assert result.modules["market"].payload["无报价剔除记录"] == 1
    assert result.modules["market"].payload["选股字段齐全记录"] == 1
    assert result.modules["screening"].status == "partial"
    deps.mail_sender.assert_not_called()


def test_force_bypasses_only_window_gate(tmp_path, deps) -> None:
    session_builder = Mock(side_effect=deps.session_builder)
    stale_gateway = FakeGateway()
    stale_gateway.get_daily_bars = Mock(side_effect=ValueError("provider secret token=abc"))

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, session_builder=session_builder, gateway=stale_gateway),
        force=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.error_code == "report_content_incomplete"
    session_builder.assert_called_once_with(ReportMode.PREMARKET, NOW, scheduled=False)
    assert result.modules["screening"].status == "unavailable"
    assert "provider secret" not in json.dumps(result.to_public_dict(), ensure_ascii=False)


def test_force_still_applies_risk_and_board_lot_sizing(tmp_path, deps) -> None:
    risk_evaluator = Mock(return_value={"状态": "正常"})
    sizing_evaluator = Mock(return_value=100)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(
            deps,
            risk_evaluator=risk_evaluator,
            sizing_evaluator=sizing_evaluator,
        ),
        force=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_SUCCESS
    risk_evaluator.assert_called_once()
    sizing_evaluator.assert_called_once_with(
        10.5,
        9.8,
        capital=20_000,
        available_cash=18_980,
        risk_fraction=0.02,
    )


def test_position_sizing_is_skipped_when_report_scope_disables_it(tmp_path, deps, settings) -> None:
    sizing_evaluator = Mock(return_value=100)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(
            deps,
            settings_loader=lambda: replace(settings, position_sizing=False),
            sizing_evaluator=sizing_evaluator,
        ),
        force=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_SUCCESS
    assert result.modules["sizing"].status == "skipped"
    assert result.modules["sizing"].payload == {"状态": "已按当前报告范围关闭，不依据持仓分配资金"}
    sizing_evaluator.assert_not_called()


def test_quarantined_held_snapshot_row_suppresses_all_new_position_sizing(tmp_path, deps) -> None:
    primary = Mock()
    primary.a_share_snapshot.return_value = ThsApiResponse(
        {
            "timestamp": int(datetime(2026, 8, 19, 15, 5, tzinfo=SHANGHAI).timestamp() * 1000),
            "total": 2,
            "item": [
                {
                    "thscode": "600000.SH",
                    "ticker": PORTFOLIO_CODE,
                    "last_price": None,
                    "price_change_ratio_pct": 0.0,
                    "volume": 0,
                    "turnover": 0,
                },
                {
                    "thscode": "600001.SH",
                    "ticker": CANDIDATE_CODE,
                    "last_price": 10.5,
                    "price_change_ratio_pct": 1.0,
                    "volume": 1_000_000,
                    "turnover": 100_000_000,
                },
            ],
        },
        None,
    )
    gateway = FakeGateway()
    snapshot_gateway = MarketDataGateway(ths_client=primary, clock=lambda: NOW)
    gateway.get_a_share_snapshot = snapshot_gateway.get_a_share_snapshot
    sizing_evaluator = Mock(return_value=100)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway, sizing_evaluator=sizing_evaluator),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.modules["market"].payload["无报价剔除记录"] == 1
    assert result.modules["portfolio"].status == "unavailable"
    assert result.modules["sizing"].status == "unavailable"
    assert result.modules["sizing"].warnings == ("持仓估值不可用，未提供仓位建议",)
    sizing_evaluator.assert_not_called()


def test_empty_portfolio_keeps_full_capital_available_for_sizing(tmp_path, deps, settings) -> None:
    sizing_evaluator = Mock(return_value=100)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(
            deps,
            settings_loader=lambda: replace(settings, positions=()),
            sizing_evaluator=sizing_evaluator,
        ),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.modules["portfolio"].status == "ok"
    sizing_evaluator.assert_called_once_with(
        10.5,
        9.8,
        capital=20_000,
        available_cash=20_000,
        risk_fraction=0.02,
    )


def test_force_does_not_accept_stale_snapshot_or_create_action_levels(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot = MarketDataset(
        gateway.snapshot.frame,
        gateway.snapshot.source,
        datetime(2026, 8, 18, 16, 30, tzinfo=SHANGHAI),
    )
    sizing_evaluator = Mock(return_value=100)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway, sizing_evaluator=sizing_evaluator),
        force=True,
        output_dir=tmp_path,
    )

    assert result.modules["market"].status == "unavailable"
    assert result.modules["screening"].status == "unavailable"
    assert not result.short_term_candidates
    sizing_evaluator.assert_not_called()


def test_material_snapshot_history_conflict_forces_watch_and_suppresses_risk_and_sizing(
    tmp_path, deps, settings
) -> None:
    conflict_settings = replace(settings, positions=(Position(CANDIDATE_CODE, 100, 10.0),))
    gateway = FakeGateway()
    gateway.snapshot.frame.loc[gateway.snapshot.frame["code"] == CANDIDATE_CODE, "price"] = 11.0
    gateway.snapshot = MarketDataset(
        gateway.snapshot.frame,
        gateway.snapshot.source,
        NOW,
        (),
        datetime(2026, 8, 18, 15, 0, tzinfo=SHANGHAI),
    )
    risk_evaluator = Mock(return_value={"状态": "正常"})
    sizing_evaluator = Mock(return_value=100)

    ai_enricher = Mock(return_value=ModuleResult("ai", "ok", NOW, {}))
    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(
            deps,
            settings_loader=lambda: conflict_settings,
            gateway=gateway,
            risk_evaluator=risk_evaluator,
            sizing_evaluator=sizing_evaluator,
            ai_enricher=ai_enricher,
        ),
        output_dir=tmp_path,
    )

    assert SNAPSHOT_DAILY_CLOSE_TOLERANCE == 0.01
    assert "snapshot_daily_close_conflict" in result.modules["market"].warnings
    assert result.modules["market"].status == "partial"
    assert deps.renderer.call_args.kwargs["short_term_candidates"][0].warning == "价格来源冲突，仅供观望"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "snapshot_daily_close_conflict" in manifest["warning_codes"]
    risk_evaluator.assert_not_called()
    sizing_evaluator.assert_not_called()
    assert CANDIDATE_CODE not in ai_enricher.call_args.args[0]
    deps.mail_sender.assert_not_called()


def test_snapshot_history_price_within_tolerance_remains_actionable(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot.frame.loc[gateway.snapshot.frame["code"] == CANDIDATE_CODE, "price"] = 10.59
    sizing_evaluator = Mock(return_value=100)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway, sizing_evaluator=sizing_evaluator),
        output_dir=tmp_path,
    )

    assert "snapshot_daily_close_conflict" not in result.modules["market"].warnings
    assert result.short_term_candidates[0].warning == ""
    sizing_evaluator.assert_called_once()


def test_postmarket_early_intraday_source_timestamp_is_untrusted_and_suppressed(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot = replace(
        gateway.snapshot,
        source_timestamp=datetime(2026, 8, 19, 14, 59, tzinfo=SHANGHAI),
    )
    ai_enricher = Mock(return_value=ModuleResult("ai", "ok", NOW, {}))
    sizing_evaluator = Mock(return_value=100)

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(
            deps,
            gateway=gateway,
            ai_enricher=ai_enricher,
            sizing_evaluator=sizing_evaluator,
        ),
        output_dir=tmp_path,
    )

    assert "snapshot_timestamp_untrusted" in result.modules["market"].warnings
    assert deps.renderer.call_args.kwargs["short_term_candidates"][0].warning == "快照权威性不足，仅供观望"
    assert CANDIDATE_CODE not in ai_enricher.call_args.args[0]
    sizing_evaluator.assert_not_called()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_timestamps"]["market"] == "unavailable"
    deps.mail_sender.assert_not_called()


def test_postmarket_stale_prior_session_source_timestamp_is_untrusted(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot = replace(
        gateway.snapshot,
        source_timestamp=datetime(2026, 8, 18, 15, 5, tzinfo=SHANGHAI),
    )

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        output_dir=tmp_path,
    )

    assert result.modules["market"].status == "partial"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_timestamps"]["market"] == "unavailable"


def test_akshare_shape_without_timestamp_attrs_never_uses_snapshot_features_actionably(
    tmp_path, deps
) -> None:
    raw = pd.DataFrame(
        {
            "代码": [CANDIDATE_CODE],
            "名称": ["示例股份"],
            "最新价": [10.5],
            "涨跌幅": [9.9],
            "量比": [8.8],
            "换手率": [7.7],
            "成交额": [999_999_999],
            "成交量": [9_999_999],
            "总市值": [6_000_000_000],
        }
    )
    gateway = FakeGateway()
    gateway.snapshot = MarketDataGateway(
        snapshot_fetcher=lambda: raw,
        clock=lambda: NOW,
    ).get_a_share_snapshot()
    ai_enricher = Mock(return_value=ModuleResult("ai", "ok", NOW, {}))
    risk_evaluator = Mock(return_value={"状态": "正常"})

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(
            deps,
            gateway=gateway,
            ai_enricher=ai_enricher,
            risk_evaluator=risk_evaluator,
        ),
        output_dir=tmp_path,
    )

    assert result.modules["market"].status == "partial"
    assert "快照来源时间不可用，所有建议仅供观察" in result.modules["market"].warnings
    assert deps.renderer.call_args.kwargs["short_term_candidates"][0].warning == "快照权威性不足，仅供观望"
    assert CANDIDATE_CODE not in ai_enricher.call_args.args[0]
    risk_evaluator.assert_not_called()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_timestamps"]["market"] == "unavailable"
    deps.mail_sender.assert_not_called()


def test_inconsistent_session_identity_fails_closed(tmp_path, deps) -> None:
    local = replace(
        deps,
        session_builder=lambda mode, current_time, scheduled: ReportSession(
            ReportMode.POSTMARKET,
            NOW,
            NOW.date(),
            True,
            "unrelated-key",
        ),
    )

    result = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)

    assert result.exit_code == EXIT_FAILURE
    assert result.error_code == "report_identity_invalid"
    deps.mail_sender.assert_not_called()


def test_external_duplicate_marker_skips_production_before_data(tmp_path, deps) -> None:
    gateway = Mock()
    local = replace(deps, gateway=gateway)

    result = run_report(
        ReportMode.PREMARKET,
        deps=local,
        already_sent=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_SUCCESS
    assert result.final_state is FinalState.DUPLICATE_SKIP
    gateway.get_a_share_snapshot.assert_not_called()
    deps.mail_sender.assert_not_called()


def test_ai_gold_and_screening_failures_block_incomplete_delivery(tmp_path, deps) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("smtp_password=do-not-leak")

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, screener=fail, gold_analyzer=fail, ai_enricher=fail),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "report_content_incomplete"
    assert result.modules["screening"].status == "unavailable"
    assert result.modules["gold"].status == "unavailable"
    assert result.modules["ai"].status == "unavailable"
    assert "AI分析暂不可用" in result.modules["ai"].warnings
    assert not result.short_term_candidates
    deps.mail_sender.assert_not_called()


def test_screening_failure_analyzes_only_portfolio_codes(tmp_path, deps) -> None:
    ai_enricher = Mock(return_value=ModuleResult("ai", "ok", NOW, {}))

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(
            deps,
            screener=Mock(side_effect=RuntimeError("screen unavailable")),
            ai_enricher=ai_enricher,
        ),
        output_dir=tmp_path,
    )

    assert result.modules["screening"].status == "unavailable"
    assert not result.short_term_candidates
    ai_enricher.assert_called_once_with((PORTFOLIO_CODE,), observed_at=NOW)


def test_ai_failure_preserves_deterministic_candidates_and_delivery(tmp_path, deps) -> None:
    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, ai_enricher=Mock(side_effect=RuntimeError("AI key leaked"))),
        output_dir=tmp_path,
    )

    assert result.modules["ai"].status == "unavailable"
    assert result.short_term_candidates == (candidate(),)
    assert deps.renderer.call_args.kwargs["short_term_candidates"] == (candidate(),)
    deps.mail_sender.assert_called_once()


def test_ths_evidence_reorders_only_existing_candidates_and_is_reported(tmp_path, deps) -> None:
    class EvidenceGateway(FakeGateway):
        def get_ths_hot_stock_list(self):
            return dataset(pd.DataFrame([{"ticker": CANDIDATE_CODE}]), "ths.fuyao.hot_stock_list")

        def get_ths_index_catalog(self, tag):
            return dataset(pd.DataFrame([{"thscode": "886042.TI"}]), "ths.fuyao.index_catalog")

        def get_ths_index_snapshot(self, thscodes):
            assert thscodes == ("886042.TI",)
            return dataset(
                pd.DataFrame([{"price_change_ratio_pct": 1.0}]),
                "ths.fuyao.index_snapshot",
            )

        def get_ths_financial_indicators(self, code, report):
            assert code == CANDIDATE_CODE
            assert report == "2026-2"
            return dataset(
                pd.DataFrame([{"index_id": "net_profit_yoy_growth_ratio", "value": "1.2"}]),
                "ths.fuyao.financial_indicators",
            )

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=EvidenceGateway()),
        output_dir=tmp_path,
    )

    selected = result.short_term_candidates[0]
    assert selected.code == CANDIDATE_CODE
    assert selected.score == 90
    assert selected.stop_price == candidate().stop_price
    assert selected.target_price == candidate().target_price
    assert selected.matched_rules == (
        "close_breaks_20d_high",
        "THS热榜证据",
        "THS基本面证据",
        "THS指数环境证据",
    )
    assert result.modules["ths_market_evidence"].status == "ok"
    assert result.modules["ths_market_evidence"].payload["数据源"] == "ths.fuyao.hot_stock_list"
    assert result.modules["ths_financial_evidence"].payload["正向增长证据数"] == 1


def test_real_gold_result_projects_without_losing_immutable_risk_checks(tmp_path, deps) -> None:
    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gold_analyzer=analyze_gold),
        output_dir=tmp_path,
    )

    assert result.modules["gold"].status == "ok"
    assert result.modules["gold"].payload["risk_checks"]["single_trade_risk_limit"] == 0.02


@pytest.mark.parametrize("message,expected", [
    ("daily bars stale", "daily_bars_stale"),
    ("daily bars invalid ohlc", "daily_bars_invalid_ohlc"),
    ("THS source timestamp is in the future", "ths_timestamp_future"),
    ("THS source timestamp invalid", "ths_timestamp_invalid"),
    ("THS snapshot missing required fields", "ths_snapshot_fields_missing"),
    ("THS snapshot contains unsafe prices", "ths_snapshot_prices_invalid"),
    ("THS snapshot contains duplicate codes", "ths_snapshot_duplicate_codes"),
    ("THS snapshot contains invalid values", "ths_snapshot_values_invalid"),
    ("provider acquisition clock invalid", "provider_clock_invalid"),
    ("https://private.invalid/?token=secret", "gold_data_failed"),
])
def test_gold_failure_diagnostics_are_allowlisted(tmp_path, deps, message, expected) -> None:
    from src.collaborative_report.runner import _warning_codes

    deps.gateway.get_gold_bars = Mock(side_effect=ValueError(message))
    result = run_report(ReportMode.POSTMARKET, deps=deps, preview_only=True, output_dir=tmp_path)

    assert expected in result.modules["gold"].warnings
    assert expected in _warning_codes(result.modules)
    assert "private.invalid" not in repr(result.modules["gold"])
    assert "secret" not in repr(result.modules["gold"])
    deps.mail_sender.assert_not_called()


def test_gold_analysis_failure_has_distinct_safe_diagnostic(tmp_path, deps) -> None:
    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gold_analyzer=Mock(side_effect=RuntimeError("private details"))),
        preview_only=True,
        output_dir=tmp_path,
    )
    assert "gold_analysis_failed" in result.modules["gold"].warnings
    assert "private details" not in repr(result.modules["gold"])


def test_data_failure_warns_watch_only_and_never_fabricates_success(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.get_a_share_snapshot = Mock(side_effect=ValueError("API_KEY=secret"))

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.error_code == "report_content_incomplete"
    assert result.modules["market"].status == "unavailable"
    assert "数据不足，建议观望" in result.modules["market"].warnings
    assert result.modules["portfolio"].status == "unavailable"
    assert not result.short_term_candidates
    assert "secret" not in json.dumps(result.to_public_dict(), ensure_ascii=False)
    deps.mail_sender.assert_not_called()


def test_renderer_receives_session_generation_time(tmp_path, deps) -> None:
    run_report(ReportMode.POSTMARKET, deps=deps, output_dir=tmp_path)

    assert deps.renderer.call_args.kwargs["generated_at"] == NOW
    assert deps.renderer.call_args.args[:2] == (ReportMode.POSTMARKET, date(2026, 8, 19))


def test_advancing_clock_uses_one_generation_time_for_real_renderer_and_manifest(tmp_path, deps) -> None:
    generated_at = NOW + timedelta(minutes=2)
    renderer = Mock(wraps=render_report)
    clock = Mock(side_effect=(
        NOW,
        NOW + timedelta(minutes=1),
        generated_at,
        NOW + timedelta(minutes=3),
        NOW + timedelta(minutes=4),
    ))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, clock=clock, renderer=renderer),
        output_dir=tmp_path,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert renderer.call_args.kwargs["generated_at"] == generated_at
    assert manifest["generated_at"] == generated_at.isoformat()


def test_snapshot_receipt_time_and_authority_age_guards_reject_future_data() -> None:
    from src.collaborative_report.runner import (
        _snapshot_fetch_is_current,
        _snapshot_source_is_authoritative,
    )

    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    gateway = FakeGateway()
    assert not _snapshot_fetch_is_current(
        replace(gateway.snapshot, observed_at=NOW + timedelta(seconds=1)),
        session,
        checked_at=NOW,
    )
    assert not _snapshot_source_is_authoritative(
        replace(gateway.snapshot, source_timestamp=NOW + timedelta(seconds=1)),
        session,
        expected_session=NOW.date(),
        checked_at=NOW,
    )
    assert not _snapshot_source_is_authoritative(
        replace(
            gateway.snapshot,
            observed_at=NOW,
            source_timestamp=NOW + timedelta(seconds=1),
        ),
        session,
        expected_session=NOW.date(),
        checked_at=NOW + timedelta(seconds=2),
    )

    close_snapshot = replace(gateway.snapshot, source_timestamp=datetime(2026, 8, 19, 15, 0, tzinfo=SHANGHAI))
    boundary = datetime(2026, 8, 19, 23, 0, tzinfo=SHANGHAI)
    assert _snapshot_source_is_authoritative(
        close_snapshot, session, expected_session=NOW.date(), checked_at=boundary
    )
    assert not _snapshot_source_is_authoritative(
        close_snapshot,
        session,
        expected_session=NOW.date(),
        checked_at=boundary + timedelta(seconds=1),
    )

    premarket = ReportSession(
        ReportMode.PREMARKET,
        datetime(2026, 8, 23, 15, 0, tzinfo=SHANGHAI),
        date(2026, 8, 23),
        True,
        "2026-08-23-premarket",
    )
    prior_close = replace(close_snapshot, source_timestamp=datetime(2026, 8, 19, 15, 0, tzinfo=SHANGHAI))
    premarket_boundary = datetime(2026, 8, 23, 15, 0, tzinfo=SHANGHAI)
    assert _snapshot_source_is_authoritative(
        prior_close,
        premarket,
        expected_session=date(2026, 8, 19),
        checked_at=premarket_boundary,
    )
    assert not _snapshot_source_is_authoritative(
        prior_close,
        premarket,
        expected_session=date(2026, 8, 19),
        checked_at=premarket_boundary + timedelta(seconds=1),
    )


def test_future_snapshot_source_is_never_promoted_after_its_receipt_time(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot = replace(
        gateway.snapshot,
        source_timestamp=NOW + timedelta(minutes=1),
    )
    clock = Mock(side_effect=(
        NOW,
        NOW,
        NOW + timedelta(minutes=2),
        NOW + timedelta(minutes=3),
        NOW + timedelta(minutes=4),
    ))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, clock=clock, gateway=gateway),
        output_dir=tmp_path,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.modules["market"].status == "partial"
    assert manifest["source_timestamps"]["market"] == "unavailable"
    assert result.error_code == "report_content_incomplete"
    deps.mail_sender.assert_not_called()


@pytest.mark.parametrize("clock_values", [
    (NOW, NOW + timedelta(days=1)),
    (NOW, NOW, NOW, NOW + timedelta(days=1)),
    (NOW, NOW - timedelta(seconds=1)),
    (datetime(2026, 8, 19, 16, 30),),
])
def test_invalid_advancing_clock_stops_before_delivery_claim(tmp_path, deps, clock_values) -> None:
    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, clock=Mock(side_effect=clock_values)),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.error_code == "report_clock_invalid"
    assert LocalDeliveryLedger(tmp_path).record("2026-08-19-postmarket") is None
    deps.mail_sender.assert_not_called()


def test_authoritative_snapshot_expiry_stops_before_delivery_claim(tmp_path, deps) -> None:
    start = datetime(2026, 8, 19, 22, 50, tzinfo=SHANGHAI)
    gateway = FakeGateway()
    gateway.snapshot = replace(
        gateway.snapshot,
        source_timestamp=datetime(2026, 8, 19, 15, 0, tzinfo=SHANGHAI),
    )
    clock = Mock(side_effect=(
        start,
        start,
        start,
        start,
        datetime(2026, 8, 19, 23, 0, 1, tzinfo=SHANGHAI),
    ))
    session_builder = lambda mode, current_time, scheduled: ReportSession(
        mode,
        start,
        start.date(),
        True,
        f"{start.date().isoformat()}-{mode.value}",
    )

    result = run_report(
        ReportMode.POSTMARKET,
        force=True,
        deps=replace(deps, clock=clock, gateway=gateway, session_builder=session_builder),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.error_code == "snapshot_source_expired"
    assert LocalDeliveryLedger(tmp_path).record("2026-08-19-postmarket") is None
    deps.mail_sender.assert_not_called()


def test_authoritative_snapshot_expiry_stops_before_rendering(tmp_path, deps) -> None:
    start = datetime(2026, 8, 19, 22, 50, tzinfo=SHANGHAI)
    gateway = FakeGateway()
    gateway.snapshot = replace(
        gateway.snapshot,
        source_timestamp=datetime(2026, 8, 19, 15, 0, tzinfo=SHANGHAI),
    )
    renderer = Mock(wraps=deps.renderer)
    clock = Mock(side_effect=(
        start,
        start,
        datetime(2026, 8, 19, 23, 0, 1, tzinfo=SHANGHAI),
    ))
    session_builder = lambda mode, current_time, scheduled: ReportSession(
        mode,
        start,
        start.date(),
        True,
        f"{start.date().isoformat()}-{mode.value}",
    )

    result = run_report(
        ReportMode.POSTMARKET,
        force=True,
        deps=replace(
            deps,
            clock=clock,
            gateway=gateway,
            renderer=renderer,
            session_builder=session_builder,
        ),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.error_code == "snapshot_source_expired"
    assert result.html_path is None
    assert LocalDeliveryLedger(tmp_path).record("2026-08-19-postmarket") is None
    renderer.assert_not_called()
    deps.mail_sender.assert_not_called()


def test_postmarket_classifies_trigger_invalidated_watch_and_stale() -> None:
    prior = (
        {"code": "600001", "name": "触发", "trigger_price": 10.5, "stop_price": 9.8},
        {"code": "600002", "name": "双触发", "trigger_price": 10.5, "stop_price": 9.8},
        {"code": "600003", "name": "观察", "trigger_price": 10.5, "stop_price": 9.8},
        {"code": "600004", "name": "陈旧", "trigger_price": 10.5, "stop_price": 9.8},
    )
    histories = {
        "600001": dataset(bars(high=10.6, low=10.0)),
        "600002": dataset(bars(high=10.6, low=9.7)),
        "600003": dataset(bars(high=10.4, low=10.0)),
        "600004": dataset(bars()).__class__(bars().iloc[:-1], "fixture", NOW),
    }

    rows = classify_prior_candidates(prior, histories, expected_session=date(2026, 8, 19))

    assert {row["code"]: row["status"] for row in rows} == {
        "600001": "触发",
        "600002": "失效",
        "600003": "继续观察",
        "600004": "失效",
    }


def test_postmarket_ingests_prior_manifest_and_classifies_all_statuses(tmp_path, deps) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "premarket",
                "trading_date": "2026-08-19",
                "report_key": "2026-08-19-premarket",
                "generated_at": "2026-08-19T09:00:00+08:00",
                "final_state": "sent",
                "test_email": False,
                "candidate_state": [
                    {"code": "600001", "name": "触发", "trigger_price": 10.5, "stop_price": 9.8},
                    {"code": "600002", "name": "失效", "trigger_price": 10.5, "stop_price": 9.8},
                    {"code": "600003", "name": "观察", "trigger_price": 10.5, "stop_price": 9.8},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    gateway = FakeGateway()
    gateway.histories.update(
        {
            "600001": dataset(bars(high=10.6, low=10.0)),
            "600002": dataset(bars(high=10.6, low=9.7)),
            "600003": dataset(bars(high=10.4, low=10.0)),
        }
    )

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        prior_report=prior,
        output_dir=tmp_path / "out",
    )

    assert [row["status"] for row in result.morning_candidates] == ["触发", "失效", "继续观察"]
    assert deps.renderer.call_args.kwargs["morning_candidates"] == result.morning_candidates
    assert result.modules["morning_candidates"].status == "ok"


@pytest.mark.parametrize(
    "overrides",
    [
        {"trading_date": "2026-08-18", "report_key": "2026-08-18-premarket"},
        {"report_key": "2026-08-19-postmarket"},
        {"generated_at": "2026-08-19T09:00:00"},
        {"generated_at": "not-a-timestamp"},
        {"generated_at": "2026-08-18T15:30:00+00:00"},
    ],
)
def test_prior_state_wrong_session_identity_or_timestamp_degrades_without_guessing(
    tmp_path, deps, overrides
) -> None:
    payload = {
        "schema_version": 1,
        "mode": "premarket",
        "trading_date": "2026-08-19",
        "report_key": "2026-08-19-premarket",
        "generated_at": "2026-08-19T09:00:00+08:00",
        "final_state": "sent",
        "test_email": False,
        "candidate_state": [
            {"code": "600002", "name": "旧候选", "trigger_price": 10.5, "stop_price": 9.8}
        ],
    }
    payload.update(overrides)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(payload), encoding="utf-8")
    gateway = FakeGateway()
    gateway.get_daily_bars = Mock(wraps=gateway.get_daily_bars)

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        prior_report=prior,
        output_dir=tmp_path / "out",
    )

    assert result.modules["morning_candidates"].status == "unavailable"
    assert result.morning_candidates == ()
    assert all(call.args[0] != "600002" for call in gateway.get_daily_bars.call_args_list)


def test_prior_candidate_history_is_fetched_once_before_screening(tmp_path, deps) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "premarket",
                "trading_date": "2026-08-19",
                "report_key": "2026-08-19-premarket",
                "generated_at": "2026-08-19T01:00:00+00:00",
                "final_state": "sent",
                "test_email": False,
                "candidate_state": [
                    {"code": "600002", "name": "旧候选", "trigger_price": 10.5, "stop_price": 9.8}
                ],
            }
        ),
        encoding="utf-8",
    )
    gateway = FakeGateway()
    gateway.histories["600002"] = dataset(bars(high=10.6, low=10.0))
    gateway.get_daily_bars = Mock(wraps=gateway.get_daily_bars)
    screener = Mock(return_value=ScreeningResult((candidate(),), ()))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway, screener=screener),
        prior_report=prior,
        output_dir=tmp_path / "out",
    )

    histories_at_screen = screener.call_args.args[1]
    assert "600002" in histories_at_screen
    assert [call.args[0] for call in gateway.get_daily_bars.call_args_list].count("600002") == 1
    assert result.morning_candidates[0]["status"] == "触发"


@pytest.mark.parametrize("content", [None, "not json", '{"schema_version":99}'])
def test_missing_or_corrupt_prior_state_is_degradable(tmp_path, deps, content) -> None:
    prior = None
    if content is not None:
        prior = tmp_path / "prior.json"
        prior.write_text(content, encoding="utf-8")

    result = run_report(
        ReportMode.POSTMARKET,
        deps=deps,
        prior_report=prior,
        output_dir=tmp_path / "out",
    )

    assert result.exit_code == EXIT_SUCCESS
    assert result.modules["morning_candidates"].status == "unavailable"
    assert "早盘候选状态不可用" in result.modules["morning_candidates"].warnings


@pytest.mark.parametrize(
    ("final_state", "test_email"),
    [("prepared", False), ("hard_failure", False), ("test_sent", True), ("sent", True)],
)
def test_prior_state_must_be_successful_production_delivery(tmp_path, deps, final_state, test_email) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "premarket",
                "trading_date": "2026-08-19",
                "report_key": "2026-08-19-premarket",
                "generated_at": "2026-08-19T09:00:00+08:00",
                "final_state": final_state,
                "test_email": test_email,
                "candidate_state": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_report(
        ReportMode.POSTMARKET, deps=deps, prior_report=prior, output_dir=tmp_path / "out"
    )

    assert result.modules["morning_candidates"].status == "unavailable"


def test_premarket_manifest_persists_only_nonportfolio_candidate_state(tmp_path, deps, settings) -> None:
    local = replace(
        deps,
        screener=lambda *args, **kwargs: ScreeningResult(
            (candidate(PORTFOLIO_CODE), candidate(CANDIDATE_CODE)), ()
        ),
    )

    result = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, ensure_ascii=False)

    assert manifest["report_key"] == "2026-08-19-premarket"
    assert manifest["candidate_state"] == [
        {
            "code": CANDIDATE_CODE,
            "name": "示例股份",
            "trigger_price": 10.5,
            "stop_price": 9.8,
        }
    ]
    for private in (
        PORTFOLIO_CODE,
        "report@example.com",
        "smtp-password",
        '"quantity"',
        '"cost_price"',
        '"capital_cny"',
    ):
        assert private not in serialized


def test_artifacts_are_atomic_and_private_outputs_are_separate(tmp_path, monkeypatch) -> None:
    replacements = []
    original_replace = Path.replace

    def tracked_replace(self, target):
        replacements.append((self, Path(target)))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", tracked_replace)
    paths = write_report_artifacts(
        tmp_path,
        report_key="2026-08-19-premarket",
        rendered=RenderedReport("subject", "<p>private</p>", "private text"),
        manifest={"report_key": "2026-08-19-premarket", "final_state": "sent"},
    )

    report_root = tmp_path / "production" / "2026-08-19-premarket"
    assert paths.html_path.parent.parent == report_root / "attempts"
    assert paths.html_path.read_text(encoding="utf-8") == "<p>private</p>"
    assert paths.text_path.read_text(encoding="utf-8") == "private text"
    assert json.loads(paths.manifest_path.read_text(encoding="utf-8"))["report_key"] == "2026-08-19-premarket"
    assert replacements
    assert stat.S_IMODE(paths.html_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.text_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.html_path.parent.stat().st_mode) == 0o700
    assert not (report_root / "current.json").exists()
    publish_report_artifacts(
        tmp_path,
        report_key="2026-08-19-premarket",
        paths=paths,
        test_email=False,
    )
    pointer = json.loads((report_root / "current.json").read_text(encoding="utf-8"))
    assert pointer["attempt"] == paths.html_path.parent.name
    assert not list(tmp_path.rglob("*.staging"))


def test_artifact_publish_failure_cleans_staging_and_sends_nothing(tmp_path, deps) -> None:
    def fail_publish(*args, **kwargs):
        raise OSError("publish failed")

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, artifact_writer=fail_publish),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    deps.mail_sender.assert_not_called()
    assert not list(tmp_path.rglob("*.staging"))


def test_atomic_pointer_publish_failure_keeps_previous_attempt_coherent(
    tmp_path, monkeypatch
) -> None:
    rendered = RenderedReport("subject", "old", "old")
    first = write_report_artifacts(
        tmp_path,
        report_key="2026-08-19-premarket",
        rendered=rendered,
        manifest={"report_key": "2026-08-19-premarket", "final_state": "sent"},
    )
    publish_report_artifacts(
        tmp_path,
        report_key="2026-08-19-premarket",
        paths=first,
        test_email=False,
    )
    original_replace = Path.replace

    def fail_pointer_publish(self, target):
        if target.name == "current.json":
            raise OSError("publish failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_pointer_publish)
    with pytest.raises(OSError, match="publish failure"):
        second = write_report_artifacts(
            tmp_path,
            report_key="2026-08-19-premarket",
            rendered=RenderedReport("subject", "new", "new"),
            manifest={"report_key": "2026-08-19-premarket", "final_state": "sent"},
        )
        publish_report_artifacts(
            tmp_path,
            report_key="2026-08-19-premarket",
            paths=second,
            test_email=False,
        )

    assert first.html_path.read_text(encoding="utf-8") == "old"
    pointer = json.loads(
        (tmp_path / "production" / "2026-08-19-premarket" / "current.json").read_text(
            encoding="utf-8"
        )
    )
    assert pointer["attempt"] == first.html_path.parent.name
    assert not list(tmp_path.rglob("*.staging"))


def test_artifact_writer_cleans_only_valid_abandoned_staging_directories(tmp_path) -> None:
    report_root = tmp_path / "production" / "2026-08-19-premarket"
    stale = report_root / f".staging-{'a' * 32}"
    unrelated = report_root / ".staging-user-data"
    stale.mkdir(parents=True)
    unrelated.mkdir()

    write_report_artifacts(
        tmp_path,
        report_key="2026-08-19-premarket",
        rendered=RenderedReport("subject", "html", "text"),
        manifest={},
    )

    assert not stale.exists()
    assert unrelated.exists()


def test_local_ledger_skips_second_production_run_without_boolean(tmp_path, deps) -> None:
    first = run_report(ReportMode.PREMARKET, deps=deps, output_dir=tmp_path)
    second = run_report(ReportMode.PREMARKET, deps=deps, output_dir=tmp_path)

    assert first.final_state is FinalState.SENT
    assert second.final_state is FinalState.DUPLICATE_SKIP
    assert deps.mail_sender.call_count == 1
    marker = tmp_path / ".delivery-ledger" / "2026-08-19-premarket.json"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert json.loads(marker.read_text(encoding="utf-8"))["state"] == "sent"


def test_nontrading_and_duplicate_skip_do_not_load_portfolio_settings(tmp_path, deps) -> None:
    settings_loader = Mock(side_effect=AssertionError("must not load"))
    duplicate_deps = replace(deps, settings_loader=settings_loader)
    ledger = LocalDeliveryLedger(tmp_path)
    attempt = finalized_attempt(tmp_path)
    claim = ledger.claim("2026-08-19-premarket", NOW, attempt_id=attempt.attempt_id)
    ledger.begin_sending("2026-08-19-premarket", claim, NOW)
    ledger.mark_sent(
        "2026-08-19-premarket", claim, NOW, attempt_id=attempt.attempt_id
    )

    duplicate = run_report(ReportMode.PREMARKET, deps=duplicate_deps, output_dir=tmp_path)
    holiday = run_report(
        ReportMode.PREMARKET,
        deps=replace(
            duplicate_deps,
            session_builder=lambda mode, current_time, scheduled: ReportSession(
                mode, NOW, NOW.date(), False, f"2026-08-19-{mode.value}"
            ),
        ),
        output_dir=tmp_path / "holiday",
    )

    assert duplicate.final_state is FinalState.DUPLICATE_SKIP
    assert holiday.final_state is FinalState.NON_TRADING_DAY_SKIP
    settings_loader.assert_not_called()


def test_test_email_ignores_production_dedupe_and_does_not_write_marker(tmp_path, deps) -> None:
    ledger = LocalDeliveryLedger(tmp_path)
    attempt = finalized_attempt(tmp_path)
    claim = ledger.claim("2026-08-19-premarket", NOW, attempt_id=attempt.attempt_id)
    ledger.begin_sending("2026-08-19-premarket", claim, NOW)
    ledger.mark_sent(
        "2026-08-19-premarket", claim, NOW, attempt_id=attempt.attempt_id
    )

    result = run_report(
        ReportMode.PREMARKET,
        deps=deps,
        already_sent=True,
        test_email=True,
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.TEST_SENT
    assert result.html_path.parent.parent == tmp_path / "test" / "2026-08-19-premarket" / "attempts"
    assert deps.mail_sender.call_count == 1


def test_ambiguous_mail_failure_remains_non_resendable_and_never_marks_sent(tmp_path, deps) -> None:
    mail_sender = Mock(side_effect=RuntimeError("smtp failed"))
    local = replace(deps, mail_sender=mail_sender)

    first = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
    second = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)

    assert first.final_state is FinalState.OPERATOR_ACTION_REQUIRED
    assert second.final_state is FinalState.OPERATOR_ACTION_REQUIRED
    assert second.exit_code == EXIT_FAILURE
    assert mail_sender.call_count == 1
    assert LocalDeliveryLedger(tmp_path).status("2026-08-19-premarket") is DeliveryState.IN_DOUBT


def test_definitive_preacceptance_failure_is_retryable(tmp_path, deps) -> None:
    from src.collaborative_report.runner import _DeliveryNotAcceptedError

    mail_sender = Mock(side_effect=[_DeliveryNotAcceptedError("not accepted"), True])
    local = replace(deps, mail_sender=mail_sender)

    first = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
    second = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)

    assert first.final_state is FinalState.HARD_FAILURE
    assert second.final_state is FinalState.SENT
    assert mail_sender.call_count == 2
    assert LocalDeliveryLedger(tmp_path).status("2026-08-19-premarket") is DeliveryState.SENT


def test_final_attempt_failure_after_send_requires_reconciliation_and_prevents_resend(tmp_path, deps) -> None:
    artifact_finalizer = Mock(side_effect=OSError("post-send manifest failed"))
    local = replace(deps, artifact_finalizer=artifact_finalizer)

    first = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
    second = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)

    assert first.final_state is FinalState.OPERATOR_ACTION_REQUIRED
    assert first.exit_code == EXIT_FAILURE
    assert second.final_state is FinalState.OPERATOR_ACTION_REQUIRED
    assert deps.mail_sender.call_count == 1


def test_ledger_failure_after_send_returns_sent_with_sanitized_partial_status(tmp_path, deps) -> None:
    class FailingFinalizeLedger(LocalDeliveryLedger):
        def mark_sent(self, report_key, claim_id, sent_at, *, attempt_id):
            raise OSError("private marker path")

    ledger = FailingFinalizeLedger(tmp_path)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, ledger_factory=lambda output: ledger),
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.OPERATOR_ACTION_REQUIRED
    assert result.exit_code == EXIT_FAILURE
    assert "private marker path" not in json.dumps(result.to_public_dict())
    assert ledger.status("2026-08-19-premarket") is DeliveryState.IN_DOUBT

    second = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, ledger_factory=lambda output: ledger),
        output_dir=tmp_path,
    )
    assert second.final_state is FinalState.OPERATOR_ACTION_REQUIRED
    assert second.exit_code == EXIT_FAILURE
    assert deps.mail_sender.call_count == 1


def test_mail_configuration_failure_is_hard_at_delivery_boundary_without_marker(tmp_path, deps) -> None:
    from src.collaborative_report.runner import _DeliveryConfigurationError

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(
            deps,
            mail_sender=Mock(side_effect=_DeliveryConfigurationError("sender@example.com auth")),
        ),
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "configuration_invalid"
    assert LocalDeliveryLedger(tmp_path).status("2026-08-19-premarket") is DeliveryState.FAILED
    assert "sender@example.com" not in json.dumps(result.to_public_dict())


def test_concurrent_runner_calls_send_at_most_once(tmp_path, deps) -> None:
    sending = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def mail_sender(rendered, *, test_email=False):
        nonlocal calls
        with calls_lock:
            calls += 1
        sending.set()
        assert release.wait(timeout=5)
        return True

    local = replace(deps, mail_sender=mail_sender)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_report, ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
        assert sending.wait(timeout=5)
        second = executor.submit(run_report, ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
        second_result = second.result(timeout=5)
        release.set()
        first_result = first.result(timeout=5)

    assert calls == 1
    assert first_result.final_state is FinalState.SENT
    assert second_result.final_state is FinalState.DUPLICATE_SKIP


def test_losing_concurrent_run_cannot_replace_winner_final_pointer(tmp_path, deps) -> None:
    first_preparing = threading.Event()
    second_preparing = threading.Event()
    release_loser = threading.Event()
    calls = 0
    lock = threading.Lock()

    def delayed_prepare(*args, **kwargs):
        nonlocal calls
        with lock:
            calls += 1
            ordinal = calls
        if ordinal == 1:
            first_preparing.set()
            assert second_preparing.wait(timeout=5)
        elif ordinal == 2:
            second_preparing.set()
            assert release_loser.wait(timeout=5)
        return write_report_artifacts(*args, **kwargs)

    local = replace(
        deps,
        artifact_writer=delayed_prepare,
        artifact_finalizer=write_report_artifacts,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(run_report, ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
        assert first_preparing.wait(timeout=5)
        loser = executor.submit(run_report, ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
        winner_result = winner.result(timeout=5)
        pointer_path = tmp_path / "production" / "2026-08-19-premarket" / "current.json"
        winner_pointer = pointer_path.read_bytes()
        release_loser.set()
        loser_result = loser.result(timeout=5)

    assert winner_result.final_state is FinalState.SENT
    assert loser_result.final_state is FinalState.DUPLICATE_SKIP
    assert pointer_path.read_bytes() == winner_pointer
    assert json.loads(winner_pointer)["final_state"] == "sent"
    assert deps.mail_sender.call_count == 1


def test_sent_preflight_repairs_missing_pointer_after_publish_failure_without_resend(
    tmp_path, deps
) -> None:
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("pointer publish failed")
        return publish_report_artifacts(*args, **kwargs)

    settings_loader = Mock(side_effect=deps.settings_loader)
    local = replace(
        deps,
        settings_loader=settings_loader,
        artifact_publisher=fail_once,
    )

    first = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)
    pointer_path = tmp_path / "production" / "2026-08-19-premarket" / "current.json"
    assert first.final_state is FinalState.SENT
    assert not pointer_path.exists()

    second = run_report(ReportMode.PREMARKET, deps=local, output_dir=tmp_path)

    assert second.final_state is FinalState.DUPLICATE_SKIP
    assert json.loads(pointer_path.read_text(encoding="utf-8"))["final_state"] == "sent"
    assert deps.mail_sender.call_count == 1
    assert settings_loader.call_count == 1


def test_corrupt_delivery_state_fails_closed_without_settings_or_send(tmp_path, deps) -> None:
    marker = tmp_path / ".delivery-ledger" / "2026-08-19-premarket.json"
    marker.parent.mkdir(mode=0o700)
    marker.write_text("{not-json", encoding="utf-8")
    settings_loader = Mock(side_effect=AssertionError("must not load"))

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, settings_loader=settings_loader),
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "delivery_state_unavailable"
    settings_loader.assert_not_called()
    deps.mail_sender.assert_not_called()


def test_unreadable_delivery_state_fails_closed_without_leaking_error(tmp_path, deps, monkeypatch) -> None:
    ledger = LocalDeliveryLedger(tmp_path)
    claim = ledger.claim("2026-08-19-premarket", NOW)
    marker = tmp_path / ".delivery-ledger" / "2026-08-19-premarket.json"
    original_read_text = Path.read_text

    def deny_state_read(self, *args, **kwargs):
        if self == marker:
            raise PermissionError("sender@example.com token=private")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_state_read)
    result = run_report(ReportMode.PREMARKET, deps=deps, output_dir=tmp_path)

    assert claim
    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "delivery_state_unavailable"
    assert "sender@example.com" not in json.dumps(result.to_public_dict())
    deps.mail_sender.assert_not_called()


def test_delivery_state_requires_manual_reconciliation_before_retry(tmp_path) -> None:
    ledger = LocalDeliveryLedger(tmp_path)
    claim = ledger.claim("2026-08-19-premarket", NOW)
    ledger.begin_sending("2026-08-19-premarket", claim, NOW)
    ledger.mark_in_doubt("2026-08-19-premarket", claim, NOW)

    assert ledger.claim("2026-08-19-premarket", NOW) is None
    ledger.reconcile("2026-08-19-premarket", DeliveryState.FAILED, NOW)
    assert ledger.claim("2026-08-19-premarket", NOW)


@pytest.mark.parametrize(
    "report_key",
    [
        "../2026-08-19-premarket",
        "/tmp/2026-08-19-premarket",
        "2026-08-19/premarket",
        "2026-08-19-./premarket",
        "2026-8-19-premarket",
        "2026-02-30-premarket",
        "2026-08-19-PREMARKET",
        "2026-08-19-premarket.json",
    ],
)
def test_public_artifact_and_ledger_helpers_reject_noncanonical_report_keys(tmp_path, report_key) -> None:
    ledger = LocalDeliveryLedger(tmp_path)
    with pytest.raises((ValueError, DeliveryStateError)):
        ledger.status(report_key)
    with pytest.raises(ValueError):
        write_report_artifacts(
            tmp_path,
            report_key=report_key,
            rendered=RenderedReport("subject", "html", "text"),
            manifest={},
        )


def test_artifact_and_ledger_boundaries_reject_symlink_escape(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "2026-08-19-premarket").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="escapes output root"):
        write_report_artifacts(
            tmp_path,
            report_key="2026-08-19-premarket",
            rendered=RenderedReport("subject", "html", "text"),
            manifest={},
        )

    ledger_root = tmp_path / "ledger-root"
    ledger_root.mkdir()
    (ledger_root / ".delivery-ledger").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes output root"):
        LocalDeliveryLedger(ledger_root)


def test_test_email_prefix_does_not_change_report_identity(tmp_path, deps) -> None:
    result = run_report(ReportMode.PREMARKET, deps=deps, test_email=True, output_dir=tmp_path)

    assert result.final_state is FinalState.TEST_SENT
    assert result.report_key == "2026-08-19-premarket"
    assert deps.renderer.call_args.kwargs["subject_prefix"] == "测试"
    sent_report = deps.mail_sender.call_args.args[0]
    assert sent_report.subject.startswith("测试 ")


def test_incomplete_test_report_is_written_but_never_emailed(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot = replace(gateway.snapshot, source_timestamp=None)
    gateway.get_daily_bars = Mock(side_effect=ValueError("daily bars stale"))

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        test_email=True,
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "report_content_incomplete"
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert result.modules["delivery_readiness"].payload["reason_codes"] == (
        "market_breadth_unavailable",
        "market_source_untrusted",
        "screening_unavailable",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_state"] == "hard_failure"
    assert manifest["module_statuses"]["delivery_readiness"] == "unavailable"
    deps.mail_sender.assert_not_called()


def test_incomplete_preview_remains_available_for_diagnostics(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot = replace(gateway.snapshot, source_timestamp=None)
    gateway.get_daily_bars = Mock(side_effect=ValueError("daily bars stale"))

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.PREVIEWED
    assert result.html_path is not None and result.html_path.exists()
    assert result.modules["delivery_readiness"].status == "unavailable"
    assert "delivery_readiness" in deps.renderer.call_args.kwargs["modules"]
    deps.mail_sender.assert_not_called()


def test_postmarket_delivery_requires_both_sector_rankings() -> None:
    modules = {
        "market": ModuleResult("market", "ok", NOW, {"上涨占比": 60.0}),
        "screening": ModuleResult("screening", "ok", NOW, {"短线候选数": 0, "波段候选数": 0}),
        "industry_sectors": ModuleResult("industry_sectors", "ok", NOW, {"strongest": [{}]}),
        "concept_sectors": ModuleResult("concept_sectors", "unavailable", NOW, {}),
    }

    assert _report_delivery_blockers(ReportMode.POSTMARKET, modules) == (
        "concept_sectors_unavailable",
    )


def test_smtp_failure_is_fatal_with_sanitized_result(tmp_path, deps) -> None:
    mail_sender = Mock(side_effect=RuntimeError("receiver@example.com password=hunter2"))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, mail_sender=mail_sender),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.final_state is FinalState.OPERATOR_ACTION_REQUIRED
    assert result.error_code == "delivery_reconciliation_required"
    assert result.manifest_path.exists()
    assert "hunter2" not in json.dumps(result.to_public_dict())


def test_settings_privacy_failure_is_closed_without_secret_in_result(tmp_path, deps) -> None:
    def fail_settings():
        raise ValueError("COLLAB_PORTFOLIO_JSON contains 600000 at cost 10.0")

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, settings_loader=fail_settings),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.error_code == "configuration_invalid"
    assert PORTFOLIO_CODE not in json.dumps(result.to_public_dict())


def test_cash_portfolio_is_available_without_position_risk_evaluation(tmp_path, deps, settings) -> None:
    cash_settings = replace(settings, positions=())
    risk = Mock()

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, settings_loader=lambda: cash_settings, risk_evaluator=risk),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.modules["portfolio"].status == "ok"
    assert result.modules["portfolio"].payload == {"status": "当前无持仓"}
    risk.assert_not_called()


def test_portfolio_status_is_partial_when_only_some_positions_succeed(tmp_path, deps, settings) -> None:
    second_code = "600002"
    mixed = replace(settings, positions=(*settings.positions, Position(second_code, 100, 10.0)))
    gateway = FakeGateway()
    extra = gateway.snapshot.frame.iloc[[0]].copy()
    extra["code"] = second_code
    gateway.snapshot = MarketDataset(
        pd.concat([gateway.snapshot.frame, extra], ignore_index=True),
        "fixture",
        NOW,
        (),
        gateway.snapshot.source_timestamp,
    )
    gateway.histories[second_code] = dataset(bars())
    risk = Mock(side_effect=[{"状态": "正常"}, ValueError("bad")])

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, settings_loader=lambda: mixed, gateway=gateway, risk_evaluator=risk),
        output_dir=tmp_path,
    )

    assert result.modules["portfolio"].status == "partial"


def test_redacted_manifest_has_no_private_body_hashes(tmp_path, deps) -> None:
    result = run_report(ReportMode.PREMARKET, deps=deps, output_dir=tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert "checksums" not in manifest


def sector_row(**overrides: object) -> SectorRow:
    values: dict[str, object] = {
        "sector_type": "industry",
        "name": "半导体",
        "rank": 1,
        "change_pct": 3.2,
        "breadth_pct": 65.0,
        "activity_percentile": 80.0,
        "leader_name": "私密龙头",
        "leader_code": "600000",
        "leader_change_pct": 9.9,
        "rotation": "accelerating",
        "persistence": "high",
        "crowding_risk": "medium",
    }
    values.update(overrides)
    return SectorRow(**values)  # type: ignore[arg-type]


def sector_analysis(*rows: SectorRow, valid_count: int | None = None) -> SectorAnalysis:
    universe_size = len(rows) if valid_count is None else valid_count
    return SectorAnalysis(strongest=rows, weakest=(), watch=(), valid_count=universe_size)


def sector_state_rows(
    sector_type: str,
    count: int,
    *,
    universe_size: int | None = None,
    source_timestamp: str = "2026-08-18T15:30:00+08:00",
) -> list[dict[str, object]]:
    base = serialized_sector_state()[0]
    size = count if universe_size is None else universe_size
    return [
        dict(
            base,
            sector_type=sector_type,
            name=f"{sector_type}-{rank:02d}",
            rank=rank,
            universe_size=size,
            source_timestamp=source_timestamp,
        )
        for rank in range(1, count + 1)
    ]


def prior_sector_manifest(*, state: list[dict[str, object]], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "postmarket",
        "final_state": "sent",
        "test_email": False,
        "trading_date": "2026-08-18",
        "report_key": "2026-08-18-postmarket",
        "generated_at": "2026-08-18T16:30:00+08:00",
        "sector_state": state,
    }
    payload.update(overrides)
    return payload


def test_premarket_uses_verified_prior_close_sector_recap(tmp_path, deps) -> None:
    prior = tmp_path / "prior-sector.json"
    prior.write_text(
        json.dumps(prior_sector_manifest(state=[
            *sector_state_rows("industry", 2),
            *sector_state_rows("concept", 2),
        ])),
        encoding="utf-8",
    )

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, renderer=render_report),
        force=True,
        preview_only=True,
        prior_sector_report=prior,
        output_dir=tmp_path / "report",
    )

    assert result.modules["industry_sectors"].status == "partial"
    assert result.modules["concept_sectors"].payload["strongest"][0]["name"] == "concept-01"
    report = result.html_path.read_text(encoding="utf-8")
    assert "行业板块" in report
    assert "概念板块" in report
    assert "盘前沿用最近一次已验证收盘板块状态" in report


def test_prior_sector_recap_rejects_no_types() -> None:
    assert _prior_sector_recap_modules(()) == {}


def serialized_sector_state() -> list[dict[str, object]]:
    return _sector_state(
        {"industry": sector_analysis(sector_row())},
        {"industry": datetime(2026, 8, 18, 15, 30, tzinfo=SHANGHAI)},
    )


def current_serialized_sector_state() -> list[dict[str, object]]:
    return _sector_state(
        {"industry": sector_analysis(sector_row())},
        {"industry": datetime(2026, 8, 19, 15, 30, tzinfo=SHANGHAI)},
    )


def test_load_prior_sector_state_returns_sanitized_valid_state(tmp_path) -> None:
    raw_state = serialized_sector_state()
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_sector_manifest(state=raw_state)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    loaded = _load_prior_sector_state(path, session)

    assert loaded == tuple(raw_state)
    assert loaded[0] is not raw_state[0]
    raw_state[0]["name"] = "篡改"
    assert loaded[0]["name"] == "半导体"


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda tmp_path: None,
        lambda tmp_path: tmp_path / "missing.json",
        lambda tmp_path: tmp_path,
        lambda tmp_path: (tmp_path / "bad.json"),
    ],
)
def test_load_prior_sector_state_rejects_unavailable_files(tmp_path, path_factory) -> None:
    path = path_factory(tmp_path)
    if path is not None and path.name == "bad.json":
        path.write_bytes(b"{\\xff")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize("contents", ["{", "[]", '"not a manifest"'])
def test_load_prior_sector_state_rejects_corrupt_or_non_mapping_roots(tmp_path, contents) -> None:
    path = tmp_path / "prior.json"
    path.write_text(contents, encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


def test_load_prior_sector_state_converts_deep_json_recursion_to_fixed_error(tmp_path) -> None:
    depth = 10_000
    content = '{"nested":' + "[" * depth + "0" + "]" * depth + "}"
    with pytest.raises(RecursionError):
        json.loads(content)
    path = tmp_path / "recursive.json"
    path.write_text(content, encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$") as error:
        _load_prior_sector_state(path, session)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_load_prior_sector_state_rejects_oversized_manifest_without_full_read(tmp_path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (_PRIOR_SECTOR_MANIFEST_MAX_BYTES + 1))
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$") as error:
        _load_prior_sector_state(path, session)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("duplicate_key", ["final_state", "name"])
def test_load_prior_sector_state_rejects_duplicate_json_object_keys(tmp_path, duplicate_key) -> None:
    content = json.dumps(prior_sector_manifest(state=serialized_sector_state()), ensure_ascii=False)
    if duplicate_key == "final_state":
        content = content.replace(
            '"final_state": "sent"',
            '"final_state": "sent", "final_state": "prepared"',
        )
    else:
        content = content.replace(
            '"name": "半导体"',
            '"name": "半导体", "name": "重复名称"',
        )
    path = tmp_path / "duplicate-key.json"
    path.write_text(content, encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 999},
        {"mode": "premarket"},
        {"final_state": "prepared"},
        {"test_email": True},
        {"trading_date": "2026-08-19"},
        {"trading_date": "2026-8-18"},
        {"report_key": "2026-08-18-premarket"},
        {"report_key": "2026-8-18-postmarket"},
        {"generated_at": "2026-08-18T16:30:00"},
        {"generated_at": "2026-08-20T16:30:00+08:00"},
        {"sector_state": {}},
    ],
)
def test_load_prior_sector_state_rejects_invalid_manifest_contract(tmp_path, overrides) -> None:
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_sector_manifest(state=serialized_sector_state(), **overrides)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_load_prior_sector_state_requires_exact_integer_schema_version(tmp_path, schema_version) -> None:
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_sector_manifest(
        state=serialized_sector_state(), schema_version=schema_version,
    )), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


def test_load_prior_sector_state_rejects_generated_at_on_the_wrong_local_date(tmp_path) -> None:
    path = tmp_path / "prior.json"
    state = serialized_sector_state()
    state[0]["source_timestamp"] = "2026-08-17T15:30:00+08:00"
    path.write_text(json.dumps(prior_sector_manifest(
        state=state,
        generated_at="2026-08-17T16:30:00+08:00",
    )), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize("trading_date", [date(2026, 8, 22), date(2026, 10, 1)])
def test_load_prior_sector_state_rejects_non_xshg_session_manifest(tmp_path, trading_date) -> None:
    date_text = trading_date.isoformat()
    payload = prior_sector_manifest(
        state=[],
        trading_date=date_text,
        report_key=f"{date_text}-postmarket",
        generated_at=f"{date_text}T16:30:00+08:00",
    )
    path = tmp_path / "non-session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    now = datetime(2026, 10, 12, 16, 30, tzinfo=SHANGHAI)
    session = ReportSession(ReportMode.POSTMARKET, now, now.date(), True, "2026-10-12-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.update({"extra": "private"}),
        lambda item: item.pop("name"),
        lambda item: item.update({"sector_type": "other"}),
        lambda item: item.update({"name": "  "}),
        lambda item: item.update({"rank": 0}),
        lambda item: item.update({"rank": True}),
        lambda item: item.update({"rank": 21}),
        lambda item: item.update({"universe_size": 0}),
        lambda item: item.update({"rank": 2, "universe_size": 1}),
        lambda item: item.update({"change_pct": float("nan")}),
        lambda item: item.update({"change_pct": True}),
        lambda item: item.update({"breadth_pct": 101}),
        lambda item: item.update({"activity_percentile": -1}),
        lambda item: item.update({"rotation": "unknown"}),
        lambda item: item.update({"persistence": "unknown"}),
        lambda item: item.update({"crowding_risk": "unknown"}),
        lambda item: item.update({"source_timestamp": "2026-08-18T15:30:00"}),
        lambda item: item.update({"source_timestamp": "2026-08-20T15:30:00+08:00"}),
    ],
)
def test_load_prior_sector_state_rejects_invalid_items(tmp_path, mutate) -> None:
    state = serialized_sector_state()
    mutate(state[0])
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_sector_manifest(state=state)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize("missing_key", SECTOR_STATE_KEYS)
def test_load_prior_sector_state_rejects_every_missing_item_key(tmp_path, missing_key) -> None:
    state = serialized_sector_state()
    state[0].pop(missing_key)
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_sector_manifest(state=state)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


def test_load_prior_sector_state_rejects_duplicates(tmp_path) -> None:
    state = serialized_sector_state()
    state.append(dict(state[0]))
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_sector_manifest(state=state)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[1].update(rank=1),
        lambda rows: rows.pop(1),
        lambda rows: rows[1].update(universe_size=4),
        lambda rows: rows[1].update(source_timestamp="2026-08-18T15:31:00+08:00"),
    ],
    ids=["duplicate-ranks", "non-contiguous-ranks", "universe-size", "source-timestamp"],
)
def test_load_prior_sector_state_rejects_inconsistent_type_collections(tmp_path, mutation) -> None:
    state = sector_state_rows("industry", 3)
    mutation(state)
    path = tmp_path / "inconsistent.json"
    path.write_text(json.dumps(prior_sector_manifest(state=state)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize("limit_case", ["over-20-for-type", "over-40-total"])
def test_load_prior_sector_state_rejects_collection_row_limits(tmp_path, limit_case) -> None:
    industry = sector_state_rows("industry", 20)
    if limit_case == "over-20-for-type":
        state = [*industry, dict(industry[-1], name="industry-extra")]
    else:
        concept = sector_state_rows("concept", 20)
        state = [*industry, *concept, {"invalid": "must be capped before item validation"}]
    path = tmp_path / "too-many.json"
    path.write_text(json.dumps(prior_sector_manifest(state=state)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_prior_sector_state_loader_failure_degrades_to_first_observation(tmp_path, failure) -> None:
    path = tmp_path / "prior.json"
    if failure == "corrupt":
        path.write_text("{private provider payload", encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    try:
        prior_rows = _load_prior_sector_state(path, session)
    except ValueError as exc:
        assert str(exc) == "prior sector state unavailable"
        previous = ()
    else:
        previous = {(row["sector_type"], row["name"]): row for row in prior_rows}
    analysis = analyze_sectors(
        pd.DataFrame([{"sector_type": "industry", "name": "半导体", "change_pct": 1.0}]),
        previous=previous, observed_at=NOW,
    )
    assert analysis.strongest[0].rotation == "first_observation"


def test_sector_state_is_deterministic_private_and_top_twenty_only() -> None:
    industry_rows = tuple(sector_row(name=f"I-{rank:02d}", rank=rank) for rank in range(1, 22))
    analyses = {
        "concept": sector_analysis(sector_row(sector_type="concept", name="A", rank=2), sector_row(sector_type="concept", name="B", rank=1)),
        "industry": sector_analysis(*industry_rows, valid_count=21),
    }
    timestamps = {
        "concept": datetime(2026, 8, 18, 15, 30, tzinfo=SHANGHAI),
        "industry": datetime(2026, 8, 18, 15, 31, tzinfo=SHANGHAI),
    }

    state = _sector_state(analyses, timestamps)

    assert [(row["sector_type"], row["rank"], row["name"]) for row in state] == [
        ("concept", 1, "B"),
        ("concept", 2, "A"),
        *(("industry", rank, f"I-{rank:02d}") for rank in range(1, 21)),
    ]
    assert set(state[0]) == set(SECTOR_STATE_KEYS)
    assert "私密龙头" not in json.dumps(state, ensure_ascii=False)


def twenty_five_industry_sectors() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sector_type": "industry",
            "name": f"行业-{index:02d}",
            "change_pct": float(26 - index),
        }
        for index in range(1, 26)
    ])


def test_sector_state_persists_complete_top_twenty_from_real_analysis() -> None:
    analysis = analyze_sectors(
        twenty_five_industry_sectors(),
        previous=(),
        observed_at=NOW,
        limit=20,
    )

    state = _sector_state({"industry": analysis}, {"industry": NOW})

    assert len(state) == 20
    assert [row["rank"] for row in state] == list(range(1, 21))
    assert {row["universe_size"] for row in state} == {25}


def test_sector_state_rejects_incomplete_default_limit_analysis_for_large_universe() -> None:
    analysis = analyze_sectors(
        twenty_five_industry_sectors(),
        previous=(),
        observed_at=NOW,
    )

    assert len(analysis.strongest) == 10
    with pytest.raises(ValueError, match="^sector state invalid$"):
        _sector_state({"industry": analysis}, {"industry": NOW})


@pytest.mark.parametrize(
    "untrustworthy",
    [None, "2026-08-18T15:30:00+08:00", datetime(2026, 8, 18, 15, 30),
     datetime(2026, 8, 18, 15, 30, tzinfo=InvalidOffsetTimezone())],
)
def test_sector_state_omits_only_type_with_untrustworthy_timestamp(untrustworthy) -> None:
    analyses = {
        "industry": sector_analysis(sector_row()),
        "concept": sector_analysis(sector_row(sector_type="concept", name="机器人")),
    }
    timestamps = {"industry": untrustworthy, "concept": NOW}

    state = _sector_state(analyses, timestamps)  # type: ignore[arg-type]

    assert [(row["sector_type"], row["name"]) for row in state] == [("concept", "机器人")]


def test_sector_state_omits_missing_timestamp_and_rejects_other_invalid_inputs() -> None:
    analysis = sector_analysis(sector_row())
    assert _sector_state({"industry": analysis}, {}) == []
    with pytest.raises(ValueError, match="^sector state invalid$"):
        _sector_state({"industry": sector_analysis(sector_row(change_pct=float("inf")))}, {"industry": NOW})


@pytest.mark.parametrize(
    "rows,valid_count",
    [
        ((sector_row(name="A", rank=1), sector_row(name="B", rank=1)), 2),
        ((sector_row(name="A", rank=1), sector_row(name="C", rank=3)), 3),
    ],
    ids=["duplicate-ranks", "non-contiguous-ranks"],
)
def test_sector_state_rejects_invalid_rank_collections(rows, valid_count) -> None:
    with pytest.raises(ValueError, match="^sector state invalid$"):
        _sector_state(
            {"industry": sector_analysis(*rows, valid_count=valid_count)},
            {"industry": NOW},
        )


def test_sector_state_does_not_mutate_inputs_and_excludes_privacy_sentinels() -> None:
    privacy_sentinels = (
        "leader_name_private", "leader_code_private", "provider_payload_private", "raw_text_private",
        "https://private.example", "credential_private", "token_private", "private@example.com",
        "quantity_private", "cost_price_private", "capital_cny_private",
    )
    analysis = SectorAnalysis(
        strongest=(sector_row(leader_name=privacy_sentinels[0], leader_code=privacy_sentinels[1]),),
        weakest=(), watch=(), valid_count=1, warnings=privacy_sentinels[2:],
    )
    analyses = {"industry": analysis}
    timestamps = {"industry": NOW}
    analyses_before = dict(analyses)
    timestamps_before = dict(timestamps)

    state = _sector_state(analyses, timestamps)

    assert analyses == analyses_before
    assert timestamps == timestamps_before
    manifest = _redacted_manifest(
        ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket"),
        {}, RenderedReport("subject", "html", "text"), final_state="sent", candidates=(),
        morning_candidates=(), portfolio_codes=set(), test_email=False,
        market_source_timestamp=NOW.isoformat(), sector_state=state,
    )
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert all(sentinel not in serialized for sentinel in privacy_sentinels)


def test_redacted_manifest_postmarket_includes_valid_sector_state_and_premarket_omits_it() -> None:
    postmarket = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    premarket = ReportSession(ReportMode.PREMARKET, NOW, NOW.date(), True, "2026-08-19-premarket")
    common = dict(
        modules={}, rendered=RenderedReport("subject", "html", "text"), final_state="sent", candidates=(),
        morning_candidates=(), portfolio_codes=set(), test_email=False, market_source_timestamp=NOW.isoformat(),
    )
    state = current_serialized_sector_state()

    manifest = _redacted_manifest(postmarket, sector_state=state, **common)
    assert manifest["sector_state"] == state
    assert manifest["sector_state"] is not state
    assert manifest["sector_state"][0] is not state[0]
    state[0]["name"] = "篡改"
    assert manifest["sector_state"][0]["name"] == "半导体"
    assert "sector_state" not in _redacted_manifest(premarket, sector_state=state, **common)
    assert _redacted_manifest(postmarket, **common)["sector_state"] == []
    with pytest.raises(ValueError, match="^sector state invalid$"):
        _redacted_manifest(postmarket, sector_state=[{"private": "payload"}], **common)


def test_redacted_manifest_rejects_duplicate_normalized_sector_identity() -> None:
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    state = current_serialized_sector_state()
    duplicate = dict(state[0], name=" 半导体 ")
    common = dict(
        modules={}, rendered=RenderedReport("subject", "html", "text"), final_state="sent", candidates=(),
        morning_candidates=(), portfolio_codes=set(), test_email=False, market_source_timestamp=NOW.isoformat(),
    )

    with pytest.raises(ValueError, match="^sector state invalid$"):
        _redacted_manifest(session, sector_state=[state[0], duplicate], **common)


def test_redacted_manifest_canonicalizes_direct_caller_sector_state_order() -> None:
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    industry = sector_state_rows("industry", 2, source_timestamp="2026-08-19T15:30:00+08:00")
    concept = sector_state_rows("concept", 2, source_timestamp="2026-08-19T15:31:00+08:00")

    manifest = _redacted_manifest(
        session, {}, RenderedReport("subject", "html", "text"), final_state="sent", candidates=(),
        morning_candidates=(), portfolio_codes=set(), test_email=False,
        market_source_timestamp=NOW.isoformat(),
        sector_state=[industry[1], concept[1], industry[0], concept[0]],
    )

    assert [(row["sector_type"], row["rank"], row["name"]) for row in manifest["sector_state"]] == [
        ("concept", 1, "concept-01"),
        ("concept", 2, "concept-02"),
        ("industry", 1, "industry-01"),
        ("industry", 2, "industry-02"),
    ]


def test_same_sector_name_is_allowed_across_industry_and_concept(tmp_path) -> None:
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")
    analyses = {
        "industry": sector_analysis(sector_row(name="共同名称")),
        "concept": sector_analysis(sector_row(sector_type="concept", name="共同名称")),
    }
    current_timestamp = datetime(2026, 8, 19, 15, 30, tzinfo=SHANGHAI)
    state = _sector_state(analyses, {"industry": current_timestamp, "concept": current_timestamp})
    manifest = _redacted_manifest(
        session, {}, RenderedReport("subject", "html", "text"), final_state="sent", candidates=(),
        morning_candidates=(), portfolio_codes=set(), test_email=False,
        market_source_timestamp=NOW.isoformat(), sector_state=state,
    )

    assert [(row["sector_type"], row["name"]) for row in manifest["sector_state"]] == [
        ("concept", "共同名称"), ("industry", "共同名称"),
    ]
    prior_timestamp = datetime(2026, 8, 18, 15, 30, tzinfo=SHANGHAI)
    prior_state = _sector_state(analyses, {"industry": prior_timestamp, "concept": prior_timestamp})
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_sector_manifest(state=prior_state)), encoding="utf-8")
    loaded = _load_prior_sector_state(path, session)
    assert [(row["sector_type"], row["name"]) for row in loaded] == [
        ("concept", "共同名称"), ("industry", "共同名称"),
    ]


def test_default_dependencies_bind_production_collaborative_modules_without_running_them() -> None:
    clock = Mock(return_value=NOW)

    production = default_dependencies(clock=clock)

    assert production.settings_loader == CollaborativeSettings.from_env
    assert production.session_builder is build_report_session
    assert production.data_session_resolver is report_data_session
    assert production.screener is screen_aggressive
    assert production.risk_evaluator is evaluate_position
    assert production.sizing_evaluator is suggested_board_lots
    assert production.short_backtest is backtest_breakout
    assert production.swing_backtest is backtest_swing
    assert production.gold_analyzer is analyze_gold
    assert production.ai_enricher is enrich_codes
    assert production.renderer is render_report
    assert isinstance(production.gateway, MarketDataGateway)
    assert isinstance(production.gateway._ths_client, ThsMarketDataClient)
    assert production.gateway._clock is clock
    clock.assert_not_called()


def test_production_mail_uses_sender_fallback_when_receivers_empty() -> None:
    config = type(
        "Config",
        (),
        {"email_sender": "sender@example.com", "email_password": "auth", "email_receivers": []},
    )()
    email_sender = Mock()
    rendered = RenderedReport("subject", "<p>body</p>", "body")

    with (
        patch("src.config.get_config", return_value=config),
        patch("src.notification_sender.email_sender.EmailSender", return_value=email_sender),
        patch("src.collaborative_report.runner.send_with_retry", return_value=True) as send,
    ):
        assert _production_mail_sender(rendered) is True

    send.assert_called_once_with(
        email_sender,
        html_content=rendered.html,
        text_content=rendered.text,
        subject=rendered.subject,
        receivers=None,
    )


@pytest.mark.parametrize(("sender", "password"), [("", "auth"), ("sender@example.com", "")])
def test_production_mail_rejects_missing_sender_or_password(sender, password) -> None:
    from src.collaborative_report.runner import _DeliveryConfigurationError

    config = type(
        "Config",
        (),
        {"email_sender": sender, "email_password": password, "email_receivers": []},
    )()

    with (
        patch("src.config.get_config", return_value=config),
        patch("src.notification_sender.email_sender.EmailSender") as email_sender,
        pytest.raises(_DeliveryConfigurationError, match="email configuration unavailable"),
    ):
        _production_mail_sender(RenderedReport("subject", "html", "text"))

    email_sender.assert_not_called()


def test_script_import_has_no_side_effect(monkeypatch) -> None:
    cli_main = Mock()
    monkeypatch.setattr("src.collaborative_report.cli.main", cli_main)
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_collaborative_report.py"

    namespace = runpy.run_path(str(script), run_name="collaborative_report_entrypoint_test")

    assert namespace["main"] is cli_main
    cli_main.assert_not_called()


def test_candidate_sector_context_defaults_are_backward_compatible() -> None:
    item = candidate()

    assert item.industry_sector == ""
    assert item.concept_sectors == ()
    assert item.sector_rotation == ""
    assert item.sector_persistence == ""


def test_postmarket_runs_independent_sector_modules_and_persists_safe_state(tmp_path, deps) -> None:
    result = run_report(
        ReportMode.POSTMARKET,
        deps=deps,
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    industry = result.modules["industry_sectors"]
    concept = result.modules["concept_sectors"]
    assert industry.status == "partial"
    assert concept.status == "partial"
    assert set(industry.payload) == {"valid_count", "strongest", "weakest", "watch"}
    assert industry.payload["strongest"][0]["leader_code"] == CANDIDATE_CODE
    selected = result.short_term_candidates[0]
    assert selected.industry_sector == "示例板块"
    assert selected.concept_sectors == ("示例概念",)
    assert selected.sector_rotation == "first_observation"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["module_statuses"]["industry_sectors"] == "partial"
    assert {row["sector_type"] for row in manifest["sector_state"]} == {"industry", "concept"}
    deps.mail_sender.assert_not_called()


def test_postmarket_sector_source_failure_does_not_suppress_other_module(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.get_sector_snapshot = Mock(side_effect=lambda kind: (
        (_ for _ in ()).throw(RuntimeError("token=secret")) if kind == "industry"
        else gateway.sector_snapshots[kind]
    ))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.modules["industry_sectors"].status == "unavailable"
    assert result.modules["concept_sectors"].status == "partial"
    serialized = json.dumps(result.modules["industry_sectors"].payload, ensure_ascii=False)
    assert "secret" not in serialized
    assert all("secret" not in warning for warning in result.modules["industry_sectors"].warnings)


def test_premarket_never_fetches_postmarket_sector_modules(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.get_sector_snapshot = Mock(wraps=gateway.get_sector_snapshot)

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert "industry_sectors" not in result.modules
    assert "concept_sectors" not in result.modules
    gateway.get_sector_snapshot.assert_not_called()


def test_market_overview_has_safe_breadth_semantics_and_decision_summary(tmp_path, deps) -> None:
    result = run_report(
        ReportMode.POSTMARKET,
        deps=deps,
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    market = result.modules["market"].payload
    assert set((
        "股票数量", "上涨家数", "下跌家数", "平盘家数", "上涨占比", "成交额", "市场温度",
        "涨停家数", "跌停家数", "市场风格",
    )).issubset(market)
    assert market["上涨家数"] == 2
    assert market["上涨占比"] == 100.0
    assert market["市场温度"] == "偏热"
    summary = result.modules["decision_summary"]
    assert set(summary.payload) == {
        "今日方向判断", "策略信号", "风险等级", "是否建议观望", "操作建议", "板块共振", "关键风险",
    }
    assert "人工确认" in summary.payload["操作建议"]
    assert deps.mail_sender.assert_not_called() is None


def test_untrusted_market_breadth_is_unavailable_and_forces_watch_summary(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.snapshot = replace(gateway.snapshot, source_timestamp=None)

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    market = result.modules["market"].payload
    assert market["上涨家数"] == "不可用"
    assert market["成交额"] == "不可用"
    assert market["市场温度"] == "不可用"
    assert result.modules["decision_summary"].payload["今日方向判断"] == "数据不足，建议观望"
    assert result.modules["decision_summary"].payload["是否建议观望"] is True


def test_postmarket_valid_prior_sector_state_classifies_transition(tmp_path, deps) -> None:
    previous_at = datetime(2026, 8, 18, 15, 5, tzinfo=SHANGHAI)
    prior_analyses = {
        sector_type: analyze_sectors(
            snapshot.frame.assign(change_pct=1.0), previous=(), observed_at=previous_at, limit=20,
        )
        for sector_type, snapshot in FakeGateway().sector_snapshots.items()
    }
    prior = tmp_path / "prior-sector.json"
    prior.write_text(json.dumps(prior_sector_manifest(
        state=_sector_state(prior_analyses, {"industry": previous_at, "concept": previous_at}),
    )), encoding="utf-8")

    result = run_report(
        ReportMode.POSTMARKET,
        deps=deps,
        force=True,
        preview_only=True,
        prior_sector_report=prior,
        output_dir=tmp_path / "out",
    )

    row = result.modules["industry_sectors"].payload["strongest"][0]
    assert row["rotation"] != "first_observation"
    assert "板块历史状态不可用，按首次观察处理" not in result.modules["industry_sectors"].warnings


def test_missing_sector_source_timestamp_marks_partial_and_omits_only_that_state(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.sector_snapshots["industry"] = replace(gateway.sector_snapshots["industry"], source_timestamp=None)

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.modules["industry_sectors"].status == "partial"
    assert "板块来源时间不可用，未持久化状态" in result.modules["industry_sectors"].warnings
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert {row["sector_type"] for row in manifest["sector_state"]} == {"concept"}


@pytest.mark.parametrize(
    ("changes", "temperature"),
    [([1.0, 1.0, 1.0, -1.0, -1.0], "偏热"), ([1.0, 1.0, -1.0, -1.0, -1.0], "偏冷")],
)
def test_market_payload_temperature_boundaries_and_invalid_amounts(changes, temperature) -> None:
    market = _market_payload(dataset(pd.DataFrame({
        "change_pct": changes,
        "amount": [float("nan"), -1, float("inf"), -2, float("nan")],
    })), authoritative=True)

    assert market["上涨占比"] in {40.0, 60.0}
    assert market["市场温度"] == temperature
    assert market["成交额"] == "不可用"
    unavailable = _market_payload(dataset(pd.DataFrame({"change_pct": changes})), authoritative=False)
    assert unavailable["上涨家数"] == "不可用"
    assert unavailable["市场温度"] == "不可用"


def test_sector_context_reranks_only_equal_scores_without_changing_candidates() -> None:
    low = replace(candidate("600001"), score=88, sector_persistence="low", sector_rotation="continuing")
    high = replace(candidate("600002"), score=88, sector_persistence="high", sector_rotation="retreating")
    fixed = replace(candidate("600003"), score=89, sector_persistence="", sector_rotation="")

    ranked = _rerank_sector_ties((fixed, low, high))

    assert [item.code for item in ranked] == ["600003", "600002", "600001"]
    assert {item.code for item in ranked} == {"600001", "600002", "600003"}
    assert [item.score for item in ranked] == [89, 88, 88]


def test_sector_data_never_creates_candidates_and_missing_price_forces_watch(tmp_path, deps, settings) -> None:
    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(
            deps,
            screener=lambda *args, **kwargs: ScreeningResult((), ()),
            settings_loader=lambda: replace(settings, positions=(Position("600099", 100, 10.0),)),
        ),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.short_term_candidates == ()
    assert result.swing_candidates == ()
    summary = result.modules["decision_summary"].payload
    assert summary["风险等级"] == "高"
    assert summary["是否建议观望"] is True
    assert result.modules["sizing"].status == "unavailable"
    deps.mail_sender.assert_not_called()


def test_unavailable_market_keeps_the_complete_safe_overview_schema(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.get_a_share_snapshot = Mock(side_effect=RuntimeError("provider-token=private"))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    payload = result.modules["market"].payload
    expected = {
        "股票数量", "上涨家数", "下跌家数", "平盘家数", "上涨占比", "成交额", "市场温度",
        "涨停家数", "跌停家数", "市场风格",
    }
    assert set(payload) == expected
    assert set(payload.values()) == {"不可用"}
    assert all("private" not in warning for warning in result.modules["market"].warnings)


@pytest.mark.parametrize("degraded_module", ["market", "industry_sectors", "concept_sectors"])
def test_partial_core_modules_force_conservative_decision(degraded_module) -> None:
    modules = {
        "market": ModuleResult("market", "ok", NOW, {"上涨占比": 70.0}),
        "portfolio": ModuleResult("portfolio", "ok", NOW, {}),
        "screening": ModuleResult("screening", "ok", NOW, {}),
        "industry_sectors": ModuleResult("industry_sectors", "ok", NOW, {"strongest": []}),
        "concept_sectors": ModuleResult("concept_sectors", "ok", NOW, {"strongest": []}),
    }
    module = modules[degraded_module]
    modules[degraded_module] = ModuleResult(module.name, "partial", module.observed_at, module.payload)

    summary = _decision_summary(modules, NOW)

    assert summary.payload["今日方向判断"] == "偏强"
    assert summary.payload["是否建议观望"] is True
    assert summary.payload["风险等级"] == "高"
    assert summary.payload["策略信号"] == "观望"


def test_concept_sector_failure_does_not_suppress_industry_module(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.get_sector_snapshot = Mock(side_effect=lambda kind: (
        (_ for _ in ()).throw(RuntimeError("private-concept-token")) if kind == "concept"
        else gateway.sector_snapshots[kind]
    ))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.modules["industry_sectors"].status == "partial"
    assert result.modules["concept_sectors"].status == "unavailable"
    assert all("private" not in warning for warning in result.modules["concept_sectors"].warnings)


@pytest.mark.parametrize("available_type", ["industry", "concept"])
def test_prior_sector_availability_is_tracked_per_type(tmp_path, deps, available_type) -> None:
    previous_at = datetime(2026, 8, 18, 15, 5, tzinfo=SHANGHAI)
    gateway = FakeGateway()
    analysis = analyze_sectors(
        gateway.sector_snapshots[available_type].frame.assign(change_pct=1.0),
        previous=(),
        observed_at=previous_at,
        limit=20,
    )
    prior = tmp_path / f"prior-{available_type}.json"
    prior.write_text(json.dumps(prior_sector_manifest(
        state=_sector_state({available_type: analysis}, {available_type: previous_at}),
    )), encoding="utf-8")

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        prior_sector_report=prior,
        output_dir=tmp_path / "out",
    )

    missing_type = "concept" if available_type == "industry" else "industry"
    available = result.modules[f"{available_type}_sectors"]
    missing = result.modules[f"{missing_type}_sectors"]
    assert available.payload["strongest"][0]["rotation"] != "first_observation"
    assert "板块历史状态不可用，按首次观察处理" not in available.warnings
    assert missing.payload["strongest"][0]["rotation"] == "first_observation"
    assert "板块历史状态不可用，按首次观察处理" in missing.warnings


def test_future_sector_source_timestamp_degrades_without_blocking_preview(tmp_path, deps) -> None:
    gateway = FakeGateway()
    future = NOW + timedelta(minutes=1)
    gateway.sector_snapshots["industry"] = replace(
        gateway.sector_snapshots["industry"], source_timestamp=future,
    )

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.PREVIEWED
    assert result.modules["industry_sectors"].status == "partial"
    assert "板块来源时间不可用，未持久化状态" in result.modules["industry_sectors"].warnings
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_timestamps"]["industry_sectors"] == "unavailable"
    assert manifest["source_timestamps"]["concept_sectors"] == NOW.replace(hour=15, minute=5).isoformat()
    assert {row["sector_type"] for row in manifest["sector_state"]} == {"concept"}


@pytest.mark.parametrize(
    "amounts",
    [
        [100.0, None],
        [100.0, "not-a-number"],
        [100.0, -1.0],
        [float("1.7e308"), float("1.7e308")],
    ],
    ids=["missing", "unparseable", "negative", "overflow"],
)
def test_market_turnover_requires_complete_finite_amount_evidence(amounts) -> None:
    market = _market_payload(dataset(pd.DataFrame({
        "change_pct": [1.0, -1.0],
        "amount": amounts,
    })), authoritative=True)

    assert market["成交额"] == "不可用"


@pytest.mark.parametrize("failed_type", ["industry", "concept"])
def test_empty_sector_snapshot_does_not_discard_healthy_type_state(tmp_path, deps, failed_type) -> None:
    gateway = FakeGateway()
    gateway.sector_snapshots[failed_type] = replace(
        gateway.sector_snapshots[failed_type],
        frame=pd.DataFrame(columns=["sector_type", "name", "change_pct"]),
    )

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    healthy_type = "concept" if failed_type == "industry" else "industry"
    assert result.modules[f"{failed_type}_sectors"].status == "unavailable"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert {row["sector_type"] for row in manifest["sector_state"]} == {healthy_type}


def test_sector_type_mismatch_is_unavailable_without_discarding_healthy_state(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.sector_snapshots["industry"] = replace(
        gateway.sector_snapshots["industry"],
        frame=gateway.sector_snapshots["industry"].frame.assign(sector_type="concept"),
    )

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.modules["industry_sectors"].status == "unavailable"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert {row["sector_type"] for row in manifest["sector_state"]} == {"concept"}


def test_weak_market_forces_watch_and_high_risk_when_modules_are_usable() -> None:
    modules = {
        "market": ModuleResult("market", "ok", NOW, {"上涨占比": 30.0}),
        "portfolio": ModuleResult("portfolio", "ok", NOW, {}),
        "screening": ModuleResult("screening", "ok", NOW, {}),
        "industry_sectors": ModuleResult(
            "industry_sectors", "ok", NOW,
            {"strongest": [{"sector_type": "industry", "name": "行业", "rank": 1, "change_pct": 0.0}]},
        ),
        "concept_sectors": ModuleResult(
            "concept_sectors", "ok", NOW,
            {"strongest": [{"sector_type": "concept", "name": "概念", "rank": 1, "change_pct": 0.0}]},
        ),
    }

    summary = _decision_summary(modules, NOW)

    assert summary.payload["今日方向判断"] == "偏弱"
    assert summary.payload["是否建议观望"] is True
    assert summary.payload["风险等级"] == "高"


def test_strong_market_weak_sectors_keep_direction_but_force_conservative_risk() -> None:
    modules = {
        "market": ModuleResult("market", "ok", NOW, {"上涨占比": 70.0}),
        "portfolio": ModuleResult("portfolio", "ok", NOW, {}),
        "screening": ModuleResult("screening", "ok", NOW, {}),
        "industry_sectors": ModuleResult(
            "industry_sectors", "ok", NOW,
            {"strongest": [{"sector_type": "industry", "name": "行业", "rank": 1, "change_pct": -1.0}]},
        ),
        "concept_sectors": ModuleResult(
            "concept_sectors", "ok", NOW,
            {"strongest": [{"sector_type": "concept", "name": "概念", "rank": 1, "change_pct": -0.5}]},
        ),
    }

    summary = _decision_summary(modules, NOW)

    assert summary.payload["今日方向判断"] == "偏强"
    assert summary.payload["是否建议观望"] is True
    assert summary.payload["风险等级"] == "高"
    assert "板块方向偏弱" in summary.payload["关键风险"]
    assert "市场与板块方向冲突" in summary.payload["关键风险"]


@pytest.mark.parametrize(
    "changes",
    [
        [1.0, None],
        [1.0, "not-a-number"],
        [1.0, float("nan")],
        [1.0, float("inf")],
    ],
    ids=["missing", "unparseable", "nan", "infinite"],
)
def test_market_breadth_requires_complete_finite_change_evidence(changes) -> None:
    market = _market_payload(dataset(pd.DataFrame({
        "change_pct": changes,
        "amount": [100.0, 200.0],
    })), authoritative=True)

    for key in ("上涨家数", "下跌家数", "平盘家数", "上涨占比", "市场温度"):
        assert market[key] == "不可用"
    assert market["股票数量"] == 2
    assert market["成交额"] == 300.0


def test_zero_row_market_has_unavailable_breadth_and_turnover() -> None:
    market = _market_payload(dataset(pd.DataFrame(columns=["change_pct", "amount"])), authoritative=True)

    assert market["股票数量"] == 0
    assert market["成交额"] == "不可用"
    for key in ("上涨家数", "下跌家数", "平盘家数", "上涨占比", "市场温度"):
        assert market[key] == "不可用"


def test_complete_market_breadth_counts_are_internally_consistent() -> None:
    market = _market_payload(dataset(pd.DataFrame({
        "change_pct": [1.0, -1.0, 0.0, 2.0],
        "amount": [10.0, 20.0, 30.0, 40.0],
    })), authoritative=True)

    assert market["股票数量"] == 4
    assert market["上涨家数"] + market["下跌家数"] + market["平盘家数"] == 4
    assert market["上涨家数"] == 2
    assert market["下跌家数"] == 1
    assert market["平盘家数"] == 1
    assert market["上涨占比"] == 50.0
    assert market["市场温度"] == "中性"
    assert market["成交额"] == 100.0


def test_incomplete_authoritative_breadth_marks_market_partial_and_summary_watch(tmp_path, deps) -> None:
    gateway = FakeGateway()
    frame = gateway.snapshot.frame.copy()
    frame.loc[frame.index[-1], "change_pct"] = None
    gateway.snapshot = replace(gateway.snapshot, frame=frame)

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    market = result.modules["market"]
    assert market.status == "partial"
    assert market.payload["上涨占比"] == "不可用"
    assert market.payload["成交额"] == 300_000_000.0
    summary = result.modules["decision_summary"].payload
    assert summary["今日方向判断"] == "数据不足，建议观望"
    assert summary["是否建议观望"] is True
    assert summary["风险等级"] == "高"


def test_prior_sector_loader_rejects_source_older_than_manifest_date(tmp_path) -> None:
    state = serialized_sector_state()
    state[0]["source_timestamp"] = "2020-01-02T15:30:00+08:00"
    path = tmp_path / "stale-prior-sector.json"
    path.write_text(json.dumps(prior_sector_manifest(state=state)), encoding="utf-8")
    session = ReportSession(ReportMode.POSTMARKET, NOW, NOW.date(), True, "2026-08-19-postmarket")

    with pytest.raises(ValueError, match="^prior sector state unavailable$"):
        _load_prior_sector_state(path, session)


def test_stale_current_sector_source_isolated_from_manifest_state(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.sector_snapshots["industry"] = replace(
        gateway.sector_snapshots["industry"],
        source_timestamp=datetime(2026, 8, 18, 15, 5, tzinfo=SHANGHAI),
    )

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        preview_only=True,
        output_dir=tmp_path,
    )

    assert result.final_state is FinalState.PREVIEWED
    assert result.modules["industry_sectors"].status == "partial"
    assert "板块来源时间不可用，未持久化状态" in result.modules["industry_sectors"].warnings
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_timestamps"]["industry_sectors"] == "unavailable"
    assert {row["sector_type"] for row in manifest["sector_state"]} == {"concept"}
