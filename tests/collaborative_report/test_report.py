from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.collaborative_report.models import Candidate, ModuleResult, ReportMode
from src.collaborative_report.report import (
    build_subject,
    evaluate_candidate_actionability,
    render_report,
)


SHANGHAI = ZoneInfo("Asia/Shanghai")
OBSERVED_AT = datetime(2026, 8, 19, 8, 55, tzinfo=SHANGHAI)


def module(name: str, payload: dict, *warnings: str, status: str = "ok") -> ModuleResult:
    return ModuleResult(name, status, OBSERVED_AT, payload, warnings)


def candidate(
    *,
    warning: str = "",
    name: str = "示例股份",
    observed_at: datetime = OBSERVED_AT,
    industry_sector: str = "",
    concept_sectors: tuple[str, ...] = (),
    sector_rotation: str = "",
    sector_persistence: str = "",
) -> Candidate:
    return Candidate(
        code="600001",
        name=name,
        horizon="1-5个交易日",
        score=85.0,
        close=10.5,
        trigger="放量突破10.60",
        stop_price=9.8,
        target_price=12.0,
        matched_rules=("ma5>ma10>ma20", "volume_expansion"),
        observed_at=observed_at,
        source="synthetic",
        warning=warning,
        industry_sector=industry_sector,
        concept_sectors=concept_sectors,
        sector_rotation=sector_rotation,
        sector_persistence=sector_persistence,
    )


def test_premarket_report_contains_required_sections_levels_warnings_and_disclaimer() -> None:
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={
            "global": module("global", {"标普500": "+0.5%", "原油": "偏强"}, "隔夜行情延迟"),
            "gold": module("gold", {"方向": "中性", "风险": "低"}),
            "portfolio": module("portfolio", {"状态": "集中度正常", "计划风险": "2%"}),
        },
        short_term_candidates=(candidate(),),
        swing_candidates=(candidate(name="波段股份"),),
    )

    assert rendered.subject == "A股盘前日报 2026-08-19"
    for expected in (
        "全球与黄金背景",
        "持仓状态",
        "短线候选池",
        "波段候选池",
        "触发条件",
        "止损",
        "目标",
        "2026-08-19 08:55 CST",
        "隔夜行情延迟",
        "人工确认后操作",
        "不承诺收益",
        "不自动下单",
    ):
        assert expected in rendered.html
        assert expected in rendered.text


def test_postmarket_report_contains_required_sections_and_optional_morning_status() -> None:
    rendered = render_report(
        ReportMode.POSTMARKET,
        date(2026, 8, 19),
        modules={
            "market": module("market", {"上涨家数": 3100, "下跌家数": 1800}),
            "portfolio": module("portfolio", {"浮动盈亏": "+120.00"}),
            "backtests": module("backtests", {"短线胜率": "55%", "最大回撤": "8%"}, "样本有限"),
        },
        morning_candidates=({"code": "600001", "name": "示例股份", "status": "继续观察"},),
        short_term_candidates=(candidate(),),
        swing_candidates=(candidate(name="波段股份"),),
    )

    assert rendered.subject == "A股收盘日报 2026-08-19"
    for expected in ("市场宽度", "早盘候选跟踪", "继续观察", "下一交易日短线池", "回测摘要", "样本有限"):
        assert expected in rendered.html
        assert expected in rendered.text


def test_all_supplied_module_warnings_render_once_in_html_and_text() -> None:
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={
            "global": module("global", {"状态": "正常"}, "全球模块警告"),
            "ai": module("ai", {"结论": "仅使用确定性分析"}, "AI分析暂不可用", status="unavailable"),
            "future_module": module("future_module", {"新字段": "保留展示"}, "未来模块警告"),
        },
    )

    for warning in ("全球模块警告", "AI分析暂不可用", "未来模块警告"):
        assert rendered.html.count(warning) == 1
        assert rendered.text.count(warning) == 1
    assert "AI分析" in rendered.html
    assert "仅使用确定性分析" in rendered.html
    assert "future_module" in rendered.text
    assert "保留展示" in rendered.text


def test_ths_evidence_sections_show_source_time_and_manual_confirmation_boundary() -> None:
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={
            "ths_market_evidence": module(
                "ths_market_evidence",
                {"数据源": "ths.fuyao.hot_stock_list", "采集时间": OBSERVED_AT.isoformat(), "热榜命中数": 1},
                "同花顺行业指数快照不可用，未参与候选排序",
                status="partial",
            ),
            "ths_financial_evidence": module(
                "ths_financial_evidence",
                {"报告期": "2026-2", "覆盖数": 1, "正向增长证据数": 1},
            ),
        },
    )

    for output in (rendered.html, rendered.text):
        assert "同花顺市场证据" in output
        assert "同花顺基本面证据" in output
        assert "ths.fuyao.hot_stock_list" in output
        assert "同花顺行业指数快照不可用，未参与候选排序" in output
        assert "人工确认后操作 / 不承诺收益 / 不自动下单" in output


def test_non_ok_module_status_is_explicit_without_duplicating_warning() -> None:
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={
            "ai": module(
                "ai",
                {"结论": "确定性结果"},
                "AI固定警告",
                "AI固定警告",
                status="unavailable",
            ),
        },
    )

    for output in (rendered.html, rendered.text):
        assert "模块状态：unavailable" in output
        assert output.count("AI固定警告") == 1


def test_postmarket_without_morning_rows_renders_unavailable_status() -> None:
    rendered = render_report(
        ReportMode.POSTMARKET,
        date(2026, 8, 19),
        modules={},
        morning_candidates=(),
    )

    for output in (rendered.html, rendered.text):
        assert "早盘候选跟踪" in output
        assert "暂无早盘候选记录，状态不可用" in output


def test_report_autoescapes_user_supplied_values() -> None:
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={"global": module("global", {"消息": '<script>alert("x")</script>'})},
        short_term_candidates=(candidate(name="<b>危险名称</b>"),),
    )

    assert "<script>" not in rendered.html
    assert "&lt;script&gt;" in rendered.html
    assert "<b>危险名称</b>" not in rendered.html
    assert "&lt;b&gt;危险名称&lt;/b&gt;" in rendered.html


def test_stale_or_non_actionable_candidate_suppresses_all_price_levels() -> None:
    stale = candidate(warning="数据陈旧，仅供观察")
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={},
        short_term_candidates=(stale,),
    )

    assert "仅供观察" in rendered.html
    assert "放量突破10.60" not in rendered.html
    assert "9.80" not in rendered.html
    assert "12.00" not in rendered.html
    assert "放量突破10.60" not in rendered.text
    assert "9.80" not in rendered.text
    assert "12.00" not in rendered.text


def test_friday_close_candidate_is_actionable_in_monday_premarket_report() -> None:
    report_date = date(2026, 8, 17)
    generated_at = datetime(2026, 8, 17, 9, 0, tzinfo=SHANGHAI)
    friday_close = candidate(observed_at=datetime(2026, 8, 14, 15, 5, tzinfo=SHANGHAI))

    policy = evaluate_candidate_actionability(
        friday_close,
        mode=ReportMode.PREMARKET,
        report_date=report_date,
        generated_at=generated_at,
    )
    rendered = render_report(
        ReportMode.PREMARKET,
        report_date,
        modules={},
        short_term_candidates=(friday_close,),
        generated_at=generated_at,
    )

    assert policy.actionable is True
    for output in (rendered.html, rendered.text):
        assert "放量突破10.60" in output
        assert "9.80" in output
        assert "12.00" in output


def test_last_open_session_survives_multi_day_xshg_holiday() -> None:
    policy = evaluate_candidate_actionability(
        candidate(observed_at=datetime(2025, 9, 30, 15, 5, tzinfo=SHANGHAI)),
        mode=ReportMode.PREMARKET,
        report_date=date(2025, 10, 9),
        generated_at=datetime(2025, 10, 9, 9, 0, tzinfo=SHANGHAI),
    )

    assert policy.actionable is True


def test_candidate_older_than_expected_market_session_is_suppressed() -> None:
    old_candidate = candidate(observed_at=datetime(2025, 9, 29, 15, 5, tzinfo=SHANGHAI))
    generated_at = datetime(2025, 10, 9, 9, 0, tzinfo=SHANGHAI)
    policy = evaluate_candidate_actionability(
        old_candidate,
        mode=ReportMode.PREMARKET,
        report_date=date(2025, 10, 9),
        generated_at=generated_at,
    )
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2025, 10, 9),
        modules={},
        short_term_candidates=(old_candidate,),
        generated_at=generated_at,
    )

    assert policy.actionable is False
    assert policy.reason == "数据交易日2025-09-29早于预期交易日2025-09-30，仅供观察"
    for output in (rendered.html, rendered.text):
        assert policy.reason in output
        assert "放量突破10.60" not in output
        assert "9.80" not in output
        assert "12.00" not in output


def test_cross_timezone_equivalent_instants_are_not_future() -> None:
    generated_at = datetime(2026, 8, 17, 9, 0, tzinfo=SHANGHAI)
    same_instant_utc = candidate(observed_at=datetime(2026, 8, 17, 1, 0, tzinfo=timezone.utc))

    policy = evaluate_candidate_actionability(
        same_instant_utc,
        mode=ReportMode.PREMARKET,
        report_date=date(2026, 8, 17),
        generated_at=generated_at,
    )

    assert policy.actionable is True


def test_actual_premarket_generation_time_accepts_post_nine_observation() -> None:
    generated_at = datetime(2026, 8, 17, 9, 20, tzinfo=SHANGHAI)
    observed = candidate(observed_at=datetime(2026, 8, 17, 9, 10, tzinfo=SHANGHAI))

    policy = evaluate_candidate_actionability(
        observed,
        mode=ReportMode.PREMARKET,
        report_date=date(2026, 8, 17),
        generated_at=generated_at,
    )

    assert policy.actionable is True


def test_late_premarket_generation_keeps_prior_session_as_target() -> None:
    report_date = date(2026, 8, 17)
    generated_at = datetime(2026, 8, 17, 16, 30, tzinfo=SHANGHAI)
    friday = candidate(
        name="周五数据",
        observed_at=datetime(2026, 8, 14, 15, 5, tzinfo=SHANGHAI),
    )
    monday = candidate(
        name="周一收盘数据",
        observed_at=datetime(2026, 8, 17, 15, 5, tzinfo=SHANGHAI),
    )

    friday_policy = evaluate_candidate_actionability(
        friday,
        mode=ReportMode.PREMARKET,
        report_date=report_date,
        generated_at=generated_at,
    )
    monday_policy = evaluate_candidate_actionability(
        monday,
        mode=ReportMode.PREMARKET,
        report_date=report_date,
        generated_at=generated_at,
    )

    assert friday_policy.actionable is True
    assert monday_policy.actionable is False
    assert monday_policy.reason == "数据交易日晚于报告会话，仅供观察"


def test_postmarket_generation_before_target_close_is_rejected_without_candidates() -> None:
    with pytest.raises(RuntimeError, match="^report data session incomplete$"):
        render_report(
            ReportMode.POSTMARKET,
            date(2026, 8, 17),
            modules={},
            generated_at=datetime(2026, 8, 17, 9, 0, tzinfo=SHANGHAI),
        )


def test_naive_generated_at_is_rejected_before_candidate_iteration() -> None:
    with pytest.raises(ValueError, match="^generated_at must be timezone-aware$"):
        render_report(
            ReportMode.PREMARKET,
            date(2026, 8, 17),
            modules={},
            generated_at=datetime(2026, 8, 17, 9, 0),
        )


def test_generated_at_local_date_mismatch_is_rejected_without_candidates() -> None:
    with pytest.raises(
        ValueError,
        match="^generated_at must match report_date in Asia/Shanghai$",
    ):
        render_report(
            ReportMode.PREMARKET,
            date(2026, 8, 17),
            modules={},
            generated_at=datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc),
        )


def test_omitted_generated_at_uses_session_identity_without_inventing_time() -> None:
    observed_after_nine = candidate(observed_at=datetime(2026, 8, 17, 9, 10, tzinfo=SHANGHAI))

    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 17),
        modules={},
        short_term_candidates=(observed_after_nine,),
    )

    for output in (rendered.html, rendered.text):
        assert "数据时间晚于报告生成时间" not in output
        assert "放量突破10.60" in output


def test_calendar_failure_suppresses_levels() -> None:
    with (
        patch(
            "src.collaborative_report.session.exchange_calendars.get_calendar",
            side_effect=RuntimeError("calendar down"),
        ),
        patch(
            "src.collaborative_report.session._akshare_xshg_sessions",
            side_effect=RuntimeError("trading calendar unavailable"),
        ),
    ):
        rendered = render_report(
            ReportMode.PREMARKET,
            date(2026, 8, 17),
            modules={},
            short_term_candidates=(
                candidate(observed_at=datetime(2026, 8, 17, 8, 55, tzinfo=SHANGHAI)),
            ),
            generated_at=datetime(2026, 8, 17, 9, 0, tzinfo=SHANGHAI),
        )

    for output in (rendered.html, rendered.text):
        assert "交易日历不可用，仅供观察" in output
        assert "放量突破10.60" not in output


def test_primary_calendar_failure_with_valid_fallback_preserves_levels() -> None:
    with (
        patch(
            "src.collaborative_report.session.exchange_calendars.get_calendar",
            side_effect=RuntimeError("calendar down"),
        ),
        patch(
            "src.collaborative_report.session._akshare_xshg_sessions",
            return_value=frozenset({date(2026, 8, 14), date(2026, 8, 17)}),
        ),
    ):
        rendered = render_report(
            ReportMode.PREMARKET,
            date(2026, 8, 17),
            modules={},
            short_term_candidates=(
                candidate(observed_at=datetime(2026, 8, 17, 8, 55, tzinfo=SHANGHAI)),
            ),
            generated_at=datetime(2026, 8, 17, 9, 0, tzinfo=SHANGHAI),
        )

    for output in (rendered.html, rendered.text):
        assert "交易日历不可用，仅供观察" not in output
        assert "放量突破10.60" in output


def test_html_and_text_share_complete_disclaimer_content() -> None:
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={},
    )

    for statement in (
        "人工确认后操作 / 不承诺收益 / 不自动下单",
        "所有价格均为分析参考，需核验数据时效与市场状态。",
        "盘前参考价不代表成交价。",
    ):
        assert rendered.html.count(statement) == 1
        assert rendered.text.count(statement) == 1


def test_subject_prefix_is_explicit_and_optional() -> None:
    assert build_subject(ReportMode.PREMARKET, date(2026, 8, 19)) == "A股盘前日报 2026-08-19"
    assert build_subject(ReportMode.POSTMARKET, date(2026, 8, 19), prefix="测试") == "测试 A股收盘日报 2026-08-19"


def _sector_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sector_type": "industry",
        "rank": 1,
        "name": "有色金属",
        "change_pct": 2.5,
        "breadth_pct": 60.0,
        "activity_percentile": 75.0,
        "leader_name": "示例股份",
        "leader_code": "600000",
        "leader_change_pct": 7.1,
        "rotation": "continuing",
        "persistence": "high",
        "crowding_risk": "medium",
    }
    row.update(overrides)
    return row


def test_postmarket_decision_summary_precedes_market_and_sector_evidence() -> None:
    rendered = render_report(
        ReportMode.POSTMARKET,
        date(2026, 8, 19),
        modules={
            "market": module("market", {"上涨家数": 3100}),
            "industry_sectors": module(
                "industry_sectors",
                {"valid_count": 1, "strongest": [_sector_row()], "weakest": [], "watch": []},
            ),
            "decision_summary": module(
                "decision_summary",
                {"今日方向判断": "偏强", "策略信号": "顺势关注", "风险等级": "中", "是否建议观望": False},
            ),
        },
    )

    for output in (rendered.html, rendered.text):
        assert output.index("决策摘要") < output.index("市场宽度") < output.index("行业板块")


def test_sector_tables_have_html_text_parity_and_complete_columns() -> None:
    industry = _sector_row()
    concept = _sector_row(
        sector_type="concept",
        rank=2,
        name="人工智能",
        change_pct=-1.25,
        rotation="retreating",
        persistence="low",
        crowding_risk="high",
    )
    rendered = render_report(
        ReportMode.POSTMARKET,
        date(2026, 8, 19),
        modules={
            "industry_sectors": module(
                "industry_sectors",
                {"valid_count": 24, "strongest": [industry], "weakest": [], "watch": [industry]},
            ),
            "concept_sectors": module(
                "concept_sectors",
                {"valid_count": 30, "strongest": [], "weakest": [concept], "watch": []},
            ),
        },
    )

    for value in ("有色金属", "2.50%", "60.0%", "75.0%", "延续", "高", "中", "人工智能", "-1.25%", "退潮"):
        assert value in rendered.html
        assert value in rendered.text
    for heading in ("排名", "板块", "涨跌幅", "宽度", "活跃度", "领涨标的", "轮动", "持续性", "拥挤风险"):
        assert heading in rendered.html
        assert heading in rendered.text
    assert rendered.html.count('class="sector-table-wrap"') == 4
    assert "有效板块数：24" in rendered.html
    assert "下一交易日观察" in rendered.text


def test_unavailable_sector_fields_are_not_rendered_as_zero() -> None:
    row = _sector_row(
        breadth_pct=None,
        activity_percentile=None,
        leader_name=None,
        leader_code=None,
        leader_change_pct=None,
        rotation="first_observation",
        persistence="unavailable",
        crowding_risk="unavailable",
    )
    rendered = render_report(
        ReportMode.POSTMARKET,
        date(2026, 8, 19),
        modules={
            "industry_sectors": module(
                "industry_sectors",
                {"valid_count": 1, "strongest": [row], "weakest": [], "watch": []},
                "板块历史状态不可用，按首次观察处理",
                status="partial",
            ),
        },
    )

    for output in (rendered.html, rendered.text):
        sector_section = output.split("行业板块", 1)[1]
        assert "首次观察" in sector_section
        assert "不可用" in sector_section
        assert "0.0%" not in sector_section
        assert output.count("板块历史状态不可用，按首次观察处理") == 1


def test_candidate_sector_context_is_preserved_in_html_and_text() -> None:
    rendered = render_report(
        ReportMode.PREMARKET,
        date(2026, 8, 19),
        modules={},
        short_term_candidates=(candidate(
            industry_sector="有色金属",
            concept_sectors=("黄金概念", "稀缺资源"),
            sector_rotation="accelerating",
            sector_persistence="high",
        ),),
    )

    for output in (rendered.html, rendered.text):
        for value in ("行业：有色金属", "概念：黄金概念、稀缺资源", "轮动：加速", "持续性：高"):
            assert value in output
