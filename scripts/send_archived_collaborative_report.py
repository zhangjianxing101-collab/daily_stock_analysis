#!/usr/bin/env python3
"""Send the latest validated prior postmarket preview without recomputing history."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collaborative_report.report import RenderedReport
from src.collaborative_report.runner import _production_mail_sender


_ARTIFACT = re.compile(r"^test-report-(\d{4}-\d{2}-\d{2})-postmarket$")
_REQUIRED_TEXT = (
    "决策摘要", "市场宽度", "行业板块", "概念板块",
    "市场新闻与事件线索", "下一交易日短线池", "下一交易日波段池",
)


def _github_json(path: str) -> object:
    completed = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _latest_artifact(metadata: object, *, before: date, max_age_days: int = 7) -> tuple[int, str]:
    candidates: list[tuple[date, str, int, str]] = []
    for page in metadata if isinstance(metadata, list) else ():
        for artifact in page.get("artifacts", ()) if isinstance(page, dict) else ():
            name = artifact.get("name") if isinstance(artifact, dict) else None
            match = _ARTIFACT.fullmatch(name) if isinstance(name, str) else None
            try:
                report_date = date.fromisoformat(match.group(1)) if match else None
                artifact_id = int(artifact["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                report_date is None
                or not before - timedelta(days=max_age_days) <= report_date < before
                or artifact.get("expired") is not False
                or not isinstance(artifact.get("created_at"), str)
            ):
                continue
            candidates.append((report_date, artifact["created_at"], artifact_id, name))
    if not candidates:
        raise ValueError("archived report unavailable")
    _, _, artifact_id, name = max(candidates)
    return artifact_id, name


def _extract_archive(payload: bytes, destination: Path) -> None:
    archive = destination / "artifact.zip"
    archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("archived report invalid")
        bundle.extractall(destination)


def _validated_report(root: Path, artifact_name: str) -> tuple[str, RenderedReport]:
    report_key = artifact_name.removeprefix("test-report-")
    valid: list[tuple[datetime, Path, dict[str, object]]] = []
    for manifest_path in root.rglob("manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            generated = datetime.fromisoformat(payload["generated_at"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and payload.get("report_key") == report_key
            and payload.get("mode") == "postmarket"
            and payload.get("trading_date") == report_key.removesuffix("-postmarket")
            and payload.get("final_state") == "previewed"
            and payload.get("test_email") is False
        ):
            valid.append((generated, manifest_path.parent, payload))
    if not valid:
        raise ValueError("archived report invalid")
    _, attempt, _ = max(valid, key=lambda item: item[0])
    html = (attempt / "report.html").read_text(encoding="utf-8")
    text = (attempt / "report.txt").read_text(encoding="utf-8")
    if any(section not in text for section in _REQUIRED_TEXT):
        raise ValueError("archived report incomplete")
    candidate_codes = set(re.findall(r"(?m)^(\d{6})\s+.+（(?:1-5个交易日|1-4周)，评分", text))
    if len(candidate_codes) < 3 or len(html) < 5_000 or len(text) < 3_000:
        raise ValueError("archived report incomplete")
    subject = f"补发核验｜A股收盘日报 {report_key.removesuffix('-postmarket')}"
    notice = "本邮件补发已保存的当日收盘工件，未使用当前行情重算历史结果。"
    rendered = RenderedReport(
        subject,
        html.replace("<body>", f"<body><p><strong>{notice}</strong></p>", 1),
        f"{notice}\n\n{text}",
    )
    return report_key, rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--before", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    diagnostic = {
        "report_key": "unavailable", "final_state": "hard_failure",
        "module_statuses": {}, "source_timestamps": {}, "warning_codes": [],
    }
    exit_code = 1
    try:
        metadata = _github_json(f"/repos/{args.repository}/actions/artifacts?per_page=100")
        artifact_id, artifact_name = _latest_artifact(metadata, before=args.before)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = subprocess.run(
                ["gh", "api", f"/repos/{args.repository}/actions/artifacts/{artifact_id}/zip"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=60,
            ).stdout
            _extract_archive(downloaded, root)
            report_key, rendered = _validated_report(root, artifact_name)
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
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"final_state={diagnostic['final_state']}\n")
            handle.write(f"runner_exit={exit_code}\n")
            handle.write(f"report_key={diagnostic['report_key']}\n")
    print(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
