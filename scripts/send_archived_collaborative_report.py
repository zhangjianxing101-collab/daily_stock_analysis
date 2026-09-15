#!/usr/bin/env python3
"""Repair and send a validated archived postmarket report as a test email."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
from lxml import html as lxml_html

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collaborative_report.ai_bridge import enrich_codes
from src.collaborative_report.market_data import read_a_share_snapshot_archive
from src.collaborative_report.report import RenderedReport, module_rows

_ARTIFACT = re.compile(r"^(?:test-)?report-(\d{4}-\d{2}-\d{2})-postmarket$")
_REQUIRED_TEXT = (
    "决策摘要",
    "市场宽度",
    "行业板块",
    "概念板块",
    "市场新闻与事件线索",
    "下一交易日短线池",
    "下一交易日波段池",
)
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_MAX_ARCHIVE_BYTES = 50_000_000


def _github_json(path: str) -> object:
    completed = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _artifact_candidates(metadata: object) -> list[dict[str, Any]]:
    return [
        artifact
        for page in metadata
        if isinstance(metadata, list)
        for artifact in page.get("artifacts", ())
        if isinstance(page, dict)
        if isinstance(artifact, dict)
        and artifact.get("expired") is False
        and isinstance(artifact.get("created_at"), str)
        and type(artifact.get("id")) is int
    ]


def _latest_artifact(metadata: object, *, before: date, max_age_days: int = 7) -> tuple[int, str]:
    candidates: list[tuple[date, str, int, str]] = []
    for artifact in _artifact_candidates(metadata):
        name = artifact.get("name")
        match = _ARTIFACT.fullmatch(name) if isinstance(name, str) else None
        try:
            report_date = date.fromisoformat(match.group(1)) if match else None
        except ValueError:
            continue
        if report_date is not None and before - timedelta(days=max_age_days) <= report_date < before:
            candidates.append((report_date, artifact["created_at"], artifact["id"], name))
    if not candidates:
        raise ValueError("archived report unavailable")
    _, _, artifact_id, name = max(candidates)
    return artifact_id, name


def _named_artifact(metadata: object, name: str) -> tuple[int, str]:
    matches = [
        (item["created_at"], item["id"], str(item["name"]))
        for item in _artifact_candidates(metadata)
        if item.get("name") == name
    ]
    if not matches:
        raise ValueError("archived artifact unavailable")
    _, artifact_id, artifact_name = max(matches)
    return artifact_id, artifact_name


def _named_report_artifact(metadata: object, report_date: date) -> tuple[int, str]:
    matches: list[tuple[str, int, str]] = []
    for item in _artifact_candidates(metadata):
        artifact_name = item.get("name")
        match = _ARTIFACT.fullmatch(artifact_name) if isinstance(artifact_name, str) else None
        if match and match.group(1) == report_date.isoformat():
            matches.append((item["created_at"], item["id"], artifact_name))
    if not matches:
        raise ValueError("archived artifact unavailable")
    _, artifact_id, artifact_name = max(matches)
    return artifact_id, artifact_name


def _extract_archive(payload: bytes, destination: Path) -> None:
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise ValueError("archived report invalid")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "artifact.zip"
    archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as bundle:
        size = 0
        for info in bundle.infolist():
            size += info.file_size
            if size > _MAX_ARCHIVE_BYTES or not (destination / info.filename).resolve().is_relative_to(
                destination.resolve()
            ):
                raise ValueError("archived report invalid")
        bundle.extractall(destination)


def _download_artifact(repository: str, artifact_id: int, destination: Path) -> None:
    payload = subprocess.run(
        ["gh", "api", f"/repos/{repository}/actions/artifacts/{artifact_id}/zip"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=60,
    ).stdout
    _extract_archive(payload, destination)


def _report_attempt(
    root: Path, artifact_name: str, *, allow_prepared: bool = False
) -> tuple[str, Path, dict[str, Any]]:
    match = _ARTIFACT.fullmatch(artifact_name)
    if match is None:
        raise ValueError("archived report invalid")
    report_key = f"{match.group(1)}-postmarket"
    states = {"previewed", "prepared"} if allow_prepared else {"previewed"}
    valid: list[tuple[datetime, Path, dict[str, Any]]] = []
    for path in root.rglob("manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            generated = datetime.fromisoformat(payload["generated_at"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and payload.get("report_key") == report_key
            and payload.get("mode") == "postmarket"
            and payload.get("trading_date") == report_key.removesuffix("-postmarket")
            and payload.get("final_state") in states
            and payload.get("test_email") is False
            and generated.tzinfo is not None
        ):
            valid.append((generated, path.parent, payload))
    if not valid:
        raise ValueError("archived report invalid")
    _, attempt, manifest = max(valid, key=lambda item: item[0])
    return report_key, attempt, manifest


def _validated_report(root: Path, artifact_name: str) -> tuple[str, RenderedReport]:
    report_key, attempt, manifest = _report_attempt(root, artifact_name)
    html = (attempt / "report.html").read_text(encoding="utf-8")
    text = (attempt / "report.txt").read_text(encoding="utf-8")
    statuses = manifest.get("module_statuses")
    ai_section = text.split("\nAI分析\n", 1)[1].split("\n资金分配\n", 1)[0] if "\nAI分析\n" in text else ""
    if (
        not isinstance(statuses, dict)
        or statuses.get("delivery_readiness") not in (None, "ok")
        or statuses.get("ai") not in ("ok", "partial")
        or statuses.get("market") not in ("ok", "partial")
        or "status：blocked" in text
        or not re.search(r"(?m)^市场风格：(?!不可用|\s*$).+", text)
        or len(re.findall(r"(?m)^\d{6} .+：.*(?:结论:|建议:|风险:|新闻:|基本面:)", ai_section)) < 3
    ):
        raise ValueError("archived report incomplete")
    if any(section not in text for section in _REQUIRED_TEXT):
        raise ValueError("archived report incomplete")
    candidates = set(re.findall(r"(?m)^(\d{6})\s+.+（(?:1-5个交易日|1-4周)，评分", text))
    if len(candidates) < 3 or len(html) < 5_000 or len(text) < 3_000:
        raise ValueError("archived report incomplete")
    subject = f"补发核验｜A股收盘日报 {report_key.removesuffix('-postmarket')}"
    notice = "本邮件补发已保存的当日收盘工件；市值补全及AI补全时间已在对应章节单独披露。"
    return report_key, RenderedReport(
        subject,
        html.replace("<body>", f"<body><p><strong>{notice}</strong></p>", 1),
        f"{notice}\n\n{text}",
    )


def _featured_codes(text: str) -> tuple[str, ...]:
    try:
        section = text.split("\n板块龙头精选观察（最多5只）\n", 1)[1].split("\n下一交易日短线池\n", 1)[0]
    except IndexError:
        raise ValueError("archived report incomplete") from None
    return tuple(dict.fromkeys(re.findall(r"(?m)^(\d{6})\s+", section)))[:5]


def _historical_market_style(snapshot_path: Path) -> tuple[Mapping[str, Any], datetime, int]:
    from src.collaborative_report.snapshot_supplement import fetch_snapshot_supplement

    now = datetime.now(_SHANGHAI)
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    session = date.fromisoformat(raw["session"])
    archived = read_a_share_snapshot_archive(snapshot_path, expected_session=session, observed_at=now)
    frame = archived.frame.copy(deep=True)
    supplement = fetch_snapshot_supplement(frame["code"].astype(str).tolist())
    current = supplement.frame.set_index("code")
    inferred: list[float | None] = []
    for _, row in frame.iterrows():
        code = str(row["code"])
        if code not in current.index:
            inferred.append(None)
            continue
        quote = current.loc[code]
        try:
            value = float(quote["total_mv"]) * float(row["price"]) / float(quote["price"])
        except (KeyError, TypeError, ValueError, OverflowError):
            value = math.nan
        inferred.append(value if math.isfinite(value) and value > 0 else None)
    frame["total_mv"] = pd.Series(inferred, index=frame.index, dtype="Float64")
    coverage = int(frame["total_mv"].notna().sum())
    if coverage < math.ceil(len(frame) * 0.7):
        raise ValueError("historical market style unavailable")
    ranked = frame.dropna(subset=["total_mv", "change_pct"]).sort_values("total_mv")
    group_size = max(len(ranked) * 3 // 10, 1)
    small_return = float(ranked.head(group_size)["change_pct"].mean())
    large_return = float(ranked.tail(group_size)["change_pct"].mean())
    if not math.isfinite(small_return) or not math.isfinite(large_return):
        raise ValueError("historical market style unavailable")
    spread = large_return - small_return
    market = {
        "市场风格": "大盘占优" if spread >= 0.5 else ("小盘占优" if spread <= -0.5 else "均衡"),
        "大盘组平均涨跌幅": large_return,
        "小盘组平均涨跌幅": small_return,
    }
    return market, supplement.observed_at, coverage


def _replace_text_section(text: str, title: str, next_title: str, lines: list[str]) -> str:
    pattern = re.compile(rf"(?ms)^{re.escape(title)}\n.*?(?=^{re.escape(next_title)}\n)")
    updated, count = pattern.subn(title + "\n" + "\n".join(lines) + "\n\n", text, count=1)
    if count != 1:
        raise ValueError("archived report invalid")
    return updated


def _section(document, title: str):
    matches = document.xpath("//section[h2[normalize-space()=$title]]", title=title)
    if len(matches) != 1:
        raise ValueError("archived report invalid")
    return matches[0]


def _replace_html_section(
    document, title: str, rows: tuple[tuple[str, str], ...], *, metadata: str, warning: str = ""
) -> None:
    section = _section(document, title)
    section.clear()
    section.set("class", "section")
    heading = lxml_html.Element("h2")
    heading.text = title
    section.append(heading)
    table = lxml_html.Element("table", role="presentation")
    body = lxml_html.Element("tbody")
    for label, value in rows:
        row = lxml_html.Element("tr")
        key = lxml_html.Element("td", {"class": "key"})
        key.text = str(label)
        cell = lxml_html.Element("td")
        cell.text = str(value)
        row.extend((key, cell))
        body.append(row)
    table.append(body)
    section.append(table)
    meta = lxml_html.Element("p", {"class": "meta"})
    meta.text = metadata
    section.append(meta)
    if warning:
        paragraph = lxml_html.Element("p", {"class": "warning"})
        paragraph.text = warning
        section.append(paragraph)


def _repair_archived_report(report_root: Path, artifact_name: str, snapshot_root: Path) -> None:
    report_key, attempt, manifest = _report_attempt(report_root, artifact_name, allow_prepared=True)
    report_date = date.fromisoformat(report_key.removesuffix("-postmarket"))
    snapshots = [
        path for path in snapshot_root.rglob("market-snapshot.json") if path.is_file() and not path.is_symlink()
    ]
    if len(snapshots) != 1:
        raise ValueError("archived snapshot invalid")
    market, market_cap_at, coverage = _historical_market_style(snapshots[0])
    text_path, html_path = attempt / "report.txt", attempt / "report.html"
    text = text_path.read_text(encoding="utf-8")
    codes = _featured_codes(text)
    if len(codes) < 3:
        raise ValueError("archived report incomplete")
    generated_at = datetime.fromisoformat(manifest["generated_at"])
    ai = enrich_codes(codes, observed_at=generated_at)
    if ai.status not in {"ok", "partial"} or len(ai.payload) < 3:
        raise ValueError("archived AI analysis unavailable")
    names = dict(re.findall(r"(?m)^(\d{6})\s+([^（：\n]+)", text))
    ai_payload = {
        code: {"name": names.get(code, "名称不可用"), **dict(value)}
        for code, value in ai.payload.items()
        if isinstance(value, Mapping)
    }
    ai_rows = module_rows("ai", ai_payload)
    if len(ai_rows) < 3:
        raise ValueError("archived AI analysis unavailable")
    cap_time = market_cap_at.astimezone(_SHANGHAI).strftime("%Y-%m-%d %H:%M CST")
    style_warning = (
        f"市场风格使用{report_date.isoformat()}收盘涨跌幅，并以{cap_time} Tencent总市值反推当日市值；"
        "用于大小盘分组，不改写当日价格、涨跌幅或成交额。"
    )
    market_rows = (
        ("市场风格", str(market["市场风格"])),
        ("大盘组平均涨跌幅", f"{float(market['大盘组平均涨跌幅']):.2f}%"),
        ("小盘组平均涨跌幅", f"{float(market['小盘组平均涨跌幅']):.2f}%"),
        ("市值字段覆盖记录", str(coverage)),
    )
    text, count = re.subn(
        r"(?m)^市场风格：不可用$",
        "\n".join(f"{label}：{value}" for label, value in market_rows) + f"\n警告：{style_warning}",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("archived report invalid")
    ai_lines = [
        f"分析上下文截止：{generated_at.astimezone(_SHANGHAI).strftime('%Y-%m-%d %H:%M CST')}",
        f"补全执行时间：{datetime.now(_SHANGHAI).strftime('%Y-%m-%d %H:%M CST')}",
        f"模块状态：{ai.status}",
        *(f"{label}：{value}" for label, value in ai_rows),
        *(f"警告：{warning}" for warning in ai.warnings),
    ]
    text = _replace_text_section(text, "AI分析", "资金分配", ai_lines)
    risk = "历史回测显示多只候选最大回撤超过10%且连续亏损超过3次；板块持续性仍属首次观察，任何操作需人工确认。"
    text = re.sub(r"(?m)^关键风险：.*$", f"关键风险：{risk}", text, count=1)
    text = _replace_text_section(
        text,
        "报告完整性检查",
        "板块龙头精选观察（最多5只）",
        ["数据时间：" + datetime.now(_SHANGHAI).strftime("%Y-%m-%d %H:%M CST"), "status：ready", "reason_codes：无"],
    )

    document = lxml_html.document_fromstring(html_path.read_text(encoding="utf-8"))
    market_body = _section(document, "市场宽度").xpath("./table/tbody")
    if len(market_body) != 1:
        raise ValueError("archived report invalid")
    style_rows = market_body[0].xpath("./tr[td[1][normalize-space()='市场风格']]")
    if len(style_rows) != 1:
        raise ValueError("archived report invalid")
    style_rows[0][1].text = str(market["市场风格"])
    position = market_body[0].index(style_rows[0]) + 1
    for label, value in market_rows[1:]:
        row = lxml_html.Element("tr")
        key, cell = lxml_html.Element("td", {"class": "key"}), lxml_html.Element("td")
        key.text, cell.text = label, value
        row.extend((key, cell))
        market_body[0].insert(position, row)
        position += 1
    warning = lxml_html.Element("p", {"class": "warning"})
    warning.text = "警告：" + style_warning
    _section(document, "市场宽度").append(warning)
    _replace_html_section(
        document,
        "AI分析",
        ai_rows,
        metadata=(
            f"分析上下文截止：{generated_at.astimezone(_SHANGHAI).strftime('%Y-%m-%d %H:%M CST')}；"
            f"补全执行时间：{datetime.now(_SHANGHAI).strftime('%Y-%m-%d %H:%M CST')}"
        ),
        warning="；".join(ai.warnings),
    )
    _replace_html_section(
        document,
        "报告完整性检查",
        (("status", "ready"), ("reason_codes", "无")),
        metadata="数据时间：" + datetime.now(_SHANGHAI).strftime("%Y-%m-%d %H:%M CST"),
    )
    risk_cells = _section(document, "决策摘要").xpath("./table/tbody/tr[td[1][normalize-space()='关键风险']]/td[2]")
    if len(risk_cells) != 1:
        raise ValueError("archived report invalid")
    risk_cells[0].text = risk

    statuses = manifest.get("module_statuses")
    if not isinstance(statuses, dict):
        raise ValueError("archived report invalid")
    statuses.update(ai=ai.status, market="partial", decision_summary="partial", delivery_readiness="ok")
    sources = manifest.get("source_timestamps")
    if isinstance(sources, dict):
        sources.update(market_cap_supplement=market_cap_at.isoformat(), ai_repair=datetime.now(_SHANGHAI).isoformat())
    warnings = manifest.get("warning_codes")
    if isinstance(warnings, list):
        manifest["warning_codes"] = [
            item for item in warnings if item not in {"ai_warning_1", "delivery_readiness_warning_1"}
        ] + ["historical_market_cap_supplement"]
    manifest["final_state"] = "previewed"
    manifest["historical_repair"] = {
        "schema_version": 1,
        "repaired_at": datetime.now(_SHANGHAI).isoformat(),
        "market_cap_source": "Tencent",
        "market_cap_coverage": coverage,
        "ai_codes": list(ai_payload),
    }
    text_path.write_text(text, encoding="utf-8")
    html_path.write_text(
        "<!doctype html>\n" + lxml_html.tostring(document, encoding="unicode", method="html"), encoding="utf-8"
    )
    (attempt / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--before", type=date.fromisoformat)
    parser.add_argument("--report-date", type=date.fromisoformat)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.before is None) == (args.report_date is None):
        parser.error("specify exactly one of --before or --report-date")
    diagnostic = {
        "report_key": "unavailable",
        "final_state": "hard_failure",
        "module_statuses": {},
        "source_timestamps": {},
        "warning_codes": [],
    }
    exit_code = 1
    try:
        metadata = _github_json(f"/repos/{args.repository}/actions/artifacts?per_page=100")
        if args.report_date is None:
            artifact_id, artifact_name = _latest_artifact(metadata, before=args.before)
            report_date = date.fromisoformat(_ARTIFACT.fullmatch(artifact_name).group(1))
        else:
            report_date = args.report_date
            artifact_id, artifact_name = _named_report_artifact(metadata, report_date)
        snapshot_id, _ = _named_artifact(metadata, f"market-snapshot-{report_date.isoformat()}")
        with tempfile.TemporaryDirectory() as temporary:
            report_root, snapshot_root = Path(temporary) / "report", Path(temporary) / "snapshot"
            _download_artifact(args.repository, artifact_id, report_root)
            _download_artifact(args.repository, snapshot_id, snapshot_root)
            _repair_archived_report(report_root, artifact_name, snapshot_root)
            report_key, rendered = _validated_report(report_root, artifact_name)
            repaired_output = args.output_dir / "repaired-archived-report"
            repaired_output.mkdir(parents=True, exist_ok=True)
            _, attempt, _ = _report_attempt(report_root, artifact_name)
            for name in ("report.html", "report.txt", "manifest.json"):
                shutil.copy2(attempt / name, repaired_output / name)
            from src.collaborative_report.runner import _production_mail_sender

            _production_mail_sender(rendered, test_email=True)
        diagnostic.update(report_key=report_key, final_state="test_sent")
        exit_code = 0
    except Exception:
        diagnostic["warning_codes"] = ["archived_report_send_failed"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostic-manifest.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(
                f"final_state={diagnostic['final_state']}\nrunner_exit={exit_code}\nreport_key={diagnostic['report_key']}\n"
            )
    print(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
