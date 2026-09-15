import json
from datetime import date
from pathlib import Path

import pytest

from scripts.send_archived_collaborative_report import (
    _featured_codes,
    _latest_artifact,
    _named_artifact,
    _named_report_artifact,
    _validated_report,
)


def test_latest_artifact_selects_newest_valid_prior_preview() -> None:
    metadata = [{"artifacts": [
        {
            "id": 10, "name": "test-report-2026-09-08-postmarket",
            "expired": False, "created_at": "2026-09-08T09:00:00Z",
        },
        {
            "id": 11, "name": "test-report-2026-09-09-postmarket",
            "expired": False, "created_at": "2026-09-09T09:00:00Z",
        },
        {
            "id": 12, "name": "test-report-2026-09-10-postmarket",
            "expired": False, "created_at": "2026-09-10T09:00:00Z",
        },
    ]}]

    assert _latest_artifact(metadata, before=date(2026, 9, 10)) == (
        11, "test-report-2026-09-09-postmarket",
    )


def test_named_artifact_selects_exact_newest_archive() -> None:
    metadata = [{"artifacts": [
        {"id": 10, "name": "market-snapshot-2026-09-14", "expired": False, "created_at": "2026-09-14T08:00:00Z"},
        {"id": 11, "name": "market-snapshot-2026-09-14", "expired": False, "created_at": "2026-09-14T09:00:00Z"},
        {"id": 12, "name": "market-snapshot-2026-09-15", "expired": False, "created_at": "2026-09-15T09:00:00Z"},
    ]}]

    assert _named_artifact(metadata, "market-snapshot-2026-09-14") == (
        11, "market-snapshot-2026-09-14",
    )


def test_named_report_artifact_accepts_production_report_name() -> None:
    metadata = [{"artifacts": [
        {
            "id": 10, "name": "test-report-2026-09-14-postmarket",
            "expired": False, "created_at": "2026-09-14T08:00:00Z",
        },
        {
            "id": 11, "name": "report-2026-09-14-postmarket",
            "expired": False, "created_at": "2026-09-14T09:00:00Z",
        },
        {
            "id": 12, "name": "report-2026-09-15-postmarket",
            "expired": False, "created_at": "2026-09-15T09:00:00Z",
        },
    ]}]

    assert _named_report_artifact(metadata, date(2026, 9, 14)) == (
        11, "report-2026-09-14-postmarket",
    )


def test_featured_codes_are_bounded_and_deduplicated() -> None:
    text = "\n".join((
        "", "板块龙头精选观察（最多5只）", "600001 甲", "600002 乙", "600001 甲",
        "600003 丙", "600004 丁", "600005 戊", "600006 己", "下一交易日短线池",
    ))

    assert _featured_codes(text) == ("600001", "600002", "600003", "600004", "600005")


def test_validated_report_requires_substantive_preview(tmp_path: Path) -> None:
    attempt = tmp_path / "attempts" / "one"
    attempt.mkdir(parents=True)
    report_key = "2026-09-09-postmarket"
    (attempt / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "report_key": report_key,
        "mode": "postmarket",
        "trading_date": "2026-09-09",
        "generated_at": "2026-09-09T16:30:00+08:00",
        "final_state": "previewed",
        "test_email": False,
        "module_statuses": {"ai": "ok", "market": "ok"},
    }), encoding="utf-8")
    sections = "\n".join((
        "决策摘要", "市场宽度", "市场风格：均衡", "行业板块", "概念板块", "市场新闻与事件线索",
        "下一交易日短线池", "下一交易日波段池",
        "600001 示例一（1-5个交易日，评分 90）",
        "600002 示例二（1-4周，评分 80）",
        "600003 示例三（1-4周，评分 70）",
        "AI分析",
        "600001 示例一：结论: 仅供研究观察",
        "600002 示例二：建议: 仅供研究观察",
        "600003 示例三：风险: 需人工确认",
        "资金分配",
    ))
    (attempt / "report.txt").write_text(sections + "\n" + "内容" * 2_000, encoding="utf-8")
    (attempt / "report.html").write_text("<body>" + "内容" * 3_000 + "</body>", encoding="utf-8")

    key, report = _validated_report(tmp_path, "report-2026-09-09-postmarket")

    assert key == report_key
    assert report.subject == "补发核验｜A股收盘日报 2026-09-09"
    assert "市值补全及AI补全时间" in report.text


def test_validated_report_rejects_thin_content(tmp_path: Path) -> None:
    attempt = tmp_path / "attempts" / "one"
    attempt.mkdir(parents=True)
    (attempt / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "report_key": "2026-09-09-postmarket",
        "mode": "postmarket",
        "trading_date": "2026-09-09",
        "generated_at": "2026-09-09T16:30:00+08:00",
        "final_state": "previewed",
        "test_email": False,
    }), encoding="utf-8")
    (attempt / "report.txt").write_text("空报告", encoding="utf-8")
    (attempt / "report.html").write_text("<body>空报告</body>", encoding="utf-8")

    with pytest.raises(ValueError, match="archived report incomplete"):
        _validated_report(tmp_path, "test-report-2026-09-09-postmarket")


@pytest.mark.parametrize("statuses,style,blocked", [
    ({"ai": "unavailable", "market": "partial"}, "均衡", False),
    ({"ai": "ok", "market": "partial", "delivery_readiness": "unavailable"}, "均衡", True),
    ({"ai": "ok", "market": "partial"}, "不可用", False),
    ({"ai": "ok", "market": "ok"}, "均衡", False),
])
def test_validated_report_rejects_incomplete_core_sections(
    tmp_path: Path, statuses: dict[str, str], style: str, blocked: bool,
) -> None:
    attempt = tmp_path / "attempts" / "one"
    attempt.mkdir(parents=True)
    (attempt / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "report_key": "2026-09-09-postmarket",
        "mode": "postmarket",
        "trading_date": "2026-09-09",
        "generated_at": "2026-09-09T16:30:00+08:00",
        "final_state": "previewed",
        "test_email": False,
        "module_statuses": statuses,
    }), encoding="utf-8")
    text = "\n".join((
        "决策摘要", "市场宽度", f"市场风格：{style}", "行业板块", "概念板块",
        "市场新闻与事件线索", "下一交易日短线池", "下一交易日波段池",
        "600001 示例一（1-5个交易日，评分 90）",
        "600002 示例二（1-4周，评分 80）",
        "600003 示例三（1-4周，评分 70）",
        "status：blocked" if blocked else "status：ready",
        "内容" * 2_000,
    ))
    (attempt / "report.txt").write_text(text, encoding="utf-8")
    (attempt / "report.html").write_text("<body>" + "内容" * 3_000 + "</body>", encoding="utf-8")

    with pytest.raises(ValueError, match="archived report incomplete"):
        _validated_report(tmp_path, "test-report-2026-09-09-postmarket")
