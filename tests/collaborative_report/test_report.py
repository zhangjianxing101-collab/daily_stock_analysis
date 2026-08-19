from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.collaborative_report.models import Candidate, ModuleResult, ReportMode
from src.collaborative_report.report import build_subject, render_report


SHANGHAI = ZoneInfo("Asia/Shanghai")
OBSERVED_AT = datetime(2026, 8, 19, 8, 55, tzinfo=SHANGHAI)


def module(name: str, payload: dict, *warnings: str, status: str = "ok") -> ModuleResult:
    return ModuleResult(name, status, OBSERVED_AT, payload, warnings)


def candidate(*, warning: str = "", name: str = "示例股份") -> Candidate:
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
        observed_at=OBSERVED_AT,
        source="synthetic",
        warning=warning,
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


def test_subject_prefix_is_explicit_and_optional() -> None:
    assert build_subject(ReportMode.PREMARKET, date(2026, 8, 19)) == "A股盘前日报 2026-08-19"
    assert build_subject(ReportMode.POSTMARKET, date(2026, 8, 19), prefix="测试") == "测试 A股收盘日报 2026-08-19"
