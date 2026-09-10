import json
from datetime import date
from pathlib import Path

import pytest

from scripts.send_archived_collaborative_report import _latest_artifact, _validated_report


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
    }), encoding="utf-8")
    sections = "\n".join((
        "决策摘要", "市场宽度", "行业板块", "概念板块", "市场新闻与事件线索",
        "下一交易日短线池", "下一交易日波段池",
        "600001 示例一（1-5个交易日，评分 90）",
        "600002 示例二（1-4周，评分 80）",
        "600003 示例三（1-4周，评分 70）",
    ))
    (attempt / "report.txt").write_text(sections + "\n" + "内容" * 2_000, encoding="utf-8")
    (attempt / "report.html").write_text("<body>" + "内容" * 3_000 + "</body>", encoding="utf-8")

    key, report = _validated_report(tmp_path, "test-report-2026-09-09-postmarket")

    assert key == report_key
    assert report.subject == "补发核验｜A股收盘日报 2026-09-09"
    assert "未使用当前行情重算历史结果" in report.text


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
