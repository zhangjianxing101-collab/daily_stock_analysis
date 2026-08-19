import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.collaborative_report.gold import analyze_gold
from src.collaborative_report.market_data import MarketDataset
from src.collaborative_report.models import Candidate, ModuleResult, Position, ReportMode
from src.collaborative_report.report import RenderedReport
from src.collaborative_report.runner import (
    EXIT_FAILURE,
    EXIT_SUCCESS,
    FinalState,
    RunnerDependencies,
    classify_prior_candidates,
    run_report,
    write_report_artifacts,
)
from src.collaborative_report.screener import ScreeningResult
from src.collaborative_report.session import ReportSession
from src.collaborative_report.settings import CollaborativeSettings


SHANGHAI = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 8, 19, 16, 30, tzinfo=SHANGHAI)
PORTFOLIO_CODE = "600000"
CANDIDATE_CODE = "600001"


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
        self.histories = {
            PORTFOLIO_CODE: dataset(bars()),
            CANDIDATE_CODE: dataset(bars()),
        }

    def get_a_share_snapshot(self):
        return self.snapshot

    def get_daily_bars(self, code, expected_session, *, days=160):
        return self.histories[code]

    def get_leading_sector_codes(self, limit=10):
        return dataset(pd.DataFrame([{"code": CANDIDATE_CODE, "sector": "示例板块"}]))

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

    assert result.exit_code == EXIT_SUCCESS
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


def test_duplicate_marker_skips_before_data_and_test_email(tmp_path, deps) -> None:
    gateway = Mock()
    local = replace(deps, gateway=gateway)

    result = run_report(
        ReportMode.PREMARKET,
        deps=local,
        already_sent=True,
        test_email=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_SUCCESS
    assert result.final_state is FinalState.DUPLICATE_SKIP
    gateway.get_a_share_snapshot.assert_not_called()
    deps.mail_sender.assert_not_called()


def test_ai_gold_and_screening_failures_degrade_and_still_send(tmp_path, deps) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("smtp_password=do-not-leak")

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, screener=fail, gold_analyzer=fail, ai_enricher=fail),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_SUCCESS
    assert result.final_state is FinalState.SENT
    assert result.modules["screening"].status == "unavailable"
    assert result.modules["gold"].status == "unavailable"
    assert result.modules["ai"].status == "unavailable"
    assert "AI分析暂不可用" in result.modules["ai"].warnings
    assert not result.short_term_candidates
    deps.mail_sender.assert_called_once()


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


def test_real_gold_result_projects_without_losing_immutable_risk_checks(tmp_path, deps) -> None:
    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gold_analyzer=analyze_gold),
        output_dir=tmp_path,
    )

    assert result.modules["gold"].status == "ok"
    assert result.modules["gold"].payload["risk_checks"]["single_trade_risk_limit"] == 0.02


def test_data_failure_warns_watch_only_and_never_fabricates_success(tmp_path, deps) -> None:
    gateway = FakeGateway()
    gateway.get_a_share_snapshot = Mock(side_effect=ValueError("API_KEY=secret"))

    result = run_report(
        ReportMode.PREMARKET,
        deps=replace(deps, gateway=gateway),
        force=True,
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_SUCCESS
    assert result.modules["market"].status == "unavailable"
    assert "数据不足，建议观望" in result.modules["market"].warnings
    assert result.modules["portfolio"].status == "unavailable"
    assert not result.short_term_candidates
    assert "secret" not in json.dumps(result.to_public_dict(), ensure_ascii=False)


def test_renderer_receives_session_generation_time(tmp_path, deps) -> None:
    run_report(ReportMode.POSTMARKET, deps=deps, output_dir=tmp_path)

    assert deps.renderer.call_args.kwargs["generated_at"] == NOW
    assert deps.renderer.call_args.args[:2] == (ReportMode.POSTMARKET, date(2026, 8, 19))


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
        manifest={"report_key": "2026-08-19-premarket"},
    )

    assert paths.html_path.read_text(encoding="utf-8") == "<p>private</p>"
    assert paths.text_path.read_text(encoding="utf-8") == "private text"
    assert json.loads(paths.manifest_path.read_text(encoding="utf-8"))["report_key"] == "2026-08-19-premarket"
    assert len(replacements) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_test_email_prefix_does_not_change_report_identity(tmp_path, deps) -> None:
    result = run_report(ReportMode.PREMARKET, deps=deps, test_email=True, output_dir=tmp_path)

    assert result.final_state is FinalState.TEST_SENT
    assert result.report_key == "2026-08-19-premarket"
    assert deps.renderer.call_args.kwargs["subject_prefix"] == "测试"
    sent_report = deps.mail_sender.call_args.args[0]
    assert sent_report.subject.startswith("测试 ")


def test_smtp_failure_is_fatal_with_sanitized_result(tmp_path, deps) -> None:
    mail_sender = Mock(side_effect=RuntimeError("receiver@example.com password=hunter2"))

    result = run_report(
        ReportMode.POSTMARKET,
        deps=replace(deps, mail_sender=mail_sender),
        output_dir=tmp_path,
    )

    assert result.exit_code == EXIT_FAILURE
    assert result.final_state is FinalState.HARD_FAILURE
    assert result.error_code == "delivery_failed"
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
