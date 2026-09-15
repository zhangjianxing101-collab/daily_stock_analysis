import json
from datetime import date
from pathlib import Path

import pytest

from scripts.send_archived_collaborative_report import (
    _archived_quantitative_fallback,
    _artifact_run_id,
    _download_artifact,
    _featured_codes,
    _latest_artifact,
    _merge_archived_ai_analysis,
    _named_artifact,
    _named_report_artifact,
    _validated_report,
)


_ARCHIVED_ANALYSIS_TEXT = """
筛选状态
短线1：300569 天能重工；收盘 5.22；评分 100.0；行业 风电零部件；规则 ma5>ma10>ma20、leading_sector
短线2：300850 新强联；收盘 30.09；评分 100.0；行业 风电零部件；规则 ma5>ma10>ma20、leading_sector
波段1：301630 同宇新材；收盘 197.57；评分 95.0；行业 电子化学品；规则 ma20>ma50且close>ma20
回测摘要
300569 天能重工｜短线：区间 2025-10-31 至 2026-09-15；交易 8 次；胜率 25.00%；累计收益 -0.23%；最大回撤 9.35%；最大连续亏损 3 次；期末权益 ¥19,954.10
300850 新强联｜短线：区间 2025-10-31 至 2026-09-15；交易 6 次；胜率 66.67%；累计收益 3.15%；最大回撤 3.32%；最大连续亏损 1 次；期末权益 ¥20,629.07
301630 同宇新材｜波段：区间 2025-10-31 至 2026-09-15；交易 0 次；胜率 0.00%；累计收益 0.00%；最大回撤 0.00%；最大连续亏损 0 次；期末权益 ¥20,000.00
"""


def test_archived_quantitative_fallback_uses_screening_and_matching_backtest() -> None:
    output = _archived_quantitative_fallback(
        _ARCHIVED_ANALYSIS_TEXT, ("300569", "300850", "301630")
    )

    assert output["300569"]["name"] == "天能重工"
    assert "累计收益-0.23%" in output["300569"]["risk_warning"]
    assert "回测风险不达标" in output["300569"]["operation_advice"]
    assert output["300850"]["confidence_level"] == "中"
    assert "回测样本不足" in output["301630"]["operation_advice"]
    assert "非AI模型结论" in output["301630"]["conclusion"]


def test_merge_archived_ai_analysis_preserves_live_results_and_fills_missing() -> None:
    merged, fallback_codes = _merge_archived_ai_analysis(
        _ARCHIVED_ANALYSIS_TEXT,
        ("300569", "300850", "301630"),
        {"300850": {"conclusion": "真实AI结论", "operation_advice": "继续观察"}},
    )

    assert merged["300850"]["conclusion"] == "真实AI结论"
    assert "量化规则回退" in merged["300569"]["conclusion"]
    assert fallback_codes == ("300569", "301630")


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


def test_artifact_run_id_reads_originating_workflow_run() -> None:
    metadata = [{"artifacts": [{
        "id": 11,
        "name": "report-2026-09-15-postmarket",
        "expired": False,
        "created_at": "2026-09-15T09:00:00Z",
        "workflow_run": {"id": 34952185768},
    }]}]

    assert _artifact_run_id(metadata, 11) == 34952185768


def test_artifact_run_id_rejects_missing_workflow_run() -> None:
    metadata = [{"artifacts": [{
        "id": 11,
        "name": "report-2026-09-15-postmarket",
        "expired": False,
        "created_at": "2026-09-15T09:00:00Z",
    }]}]

    with pytest.raises(ValueError, match="artifact run unavailable"):
        _artifact_run_id(metadata, 11)


def test_download_artifact_uses_exact_run_and_name(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr("scripts.send_archived_collaborative_report.subprocess.run", fake_run)

    destination = tmp_path / "artifact"
    _download_artifact("owner/repo", 123, "report-2026-09-15-postmarket", destination)

    assert calls[0][0] == [
        "gh", "run", "download", "123", "--repo", "owner/repo",
        "--name", "report-2026-09-15-postmarket", "--dir", str(destination),
    ]
    assert destination.is_dir()


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
    assert "市值补全、AI补全或量化规则回退时间" in report.text


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
