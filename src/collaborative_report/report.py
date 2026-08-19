"""Compose collaborative A-share reports as HTML and plain text."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import Candidate, ModuleResult, ReportMode
from .session import latest_completed_xshg_session, report_data_session


_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
_ENVIRONMENT = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"], default=True),
    trim_blocks=True,
    lstrip_blocks=True,
)
_DISCLAIMER_LINES = (
    "人工确认后操作 / 不承诺收益 / 不自动下单",
    "所有价格均为分析参考，需核验数据时效与市场状态。",
    "盘前参考价不代表成交价。",
)
_NO_MORNING_STATUS = "暂无早盘候选记录，状态不可用"
_MODULE_TITLES = {
    "global": "全球与黄金背景",
    "gold": "黄金量化背景",
    "portfolio": "持仓状态",
    "market": "市场宽度",
    "backtests": "回测摘要",
    "ai": "AI分析",
    "screening": "筛选状态",
}


@dataclass(frozen=True)
class RenderedReport:
    subject: str
    html: str
    text: str


@dataclass(frozen=True)
class CandidateActionability:
    """Report-boundary decision controlling whether price levels may be shown."""

    actionable: bool
    reason: str = ""


@dataclass(frozen=True)
class _ModuleView:
    title: str
    status: str
    observed_at: str
    rows: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _CandidateView:
    code: str
    name: str
    horizon: str
    score: str
    close: str
    trigger: str
    stop: str
    target: str
    rules: str
    observed_at: str
    source: str
    actionable: bool
    warning: str


def build_subject(mode: ReportMode, report_date: date, *, prefix: str | None = None) -> str:
    """Build the production subject, leaving an explicit prefix hook for callers."""

    normalized_mode = ReportMode(mode)
    label = "盘前" if normalized_mode is ReportMode.PREMARKET else "收盘"
    subject = f"A股{label}日报 {report_date.isoformat()}"
    return f"{prefix.strip()} {subject}" if prefix and prefix.strip() else subject


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M %Z")


def _display_value(value: Any) -> str:
    if value is None:
        return "暂无"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if is_dataclass(value) and not isinstance(value, type):
        return "；".join(f"{key}: {_display_value(item)}" for key, item in asdict(value).items() if key != "trades")
    if isinstance(value, Mapping):
        return "；".join(f"{key}: {_display_value(item)}" for key, item in value.items())
    if isinstance(value, (tuple, list, set)):
        return "、".join(_display_value(item) for item in value) or "暂无"
    return str(value)


def _module_view(title: str, result: ModuleResult | None) -> _ModuleView:
    if result is None:
        return _ModuleView(title, "unavailable", "暂无", (), ("数据暂不可用",))
    rows = tuple((str(key), _display_value(value)) for key, value in result.payload.items())
    warnings = tuple(dict.fromkeys(str(warning) for warning in result.warnings))
    return _ModuleView(title, result.status, _format_timestamp(result.observed_at), rows, warnings)


def _module_sections(
    modules: Mapping[str, ModuleResult], primary_keys: Sequence[str]
) -> tuple[_ModuleView, ...]:
    sections = [
        _module_view(_MODULE_TITLES[key], modules.get(key)) for key in primary_keys
    ]
    for key, result in modules.items():
        if key in primary_keys:
            continue
        title = _MODULE_TITLES.get(key, result.name or key)
        sections.append(_module_view(title, result))
    return tuple(sections)


def evaluate_candidate_actionability(
    candidate: Candidate,
    *,
    mode: ReportMode,
    report_date: date,
    generated_at: datetime | None = None,
) -> CandidateActionability:
    """Expose levels only when candidate data belongs to the report's completed XSHG session."""

    if not isinstance(mode, ReportMode):
        raise ValueError("invalid report mode")
    try:
        target_session = report_data_session(mode, report_date, generated_at)
    except RuntimeError as exc:
        if str(exc) == "trading calendar unavailable":
            return CandidateActionability(False, "交易日历不可用，仅供观察")
        raise
    return _evaluate_candidate_for_session(
        candidate,
        target_session=target_session,
        generated_at=generated_at,
    )


def _evaluate_candidate_for_session(
    candidate: Candidate,
    *,
    target_session: date | None,
    generated_at: datetime | None,
) -> CandidateActionability:
    observed_at = candidate.observed_at
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        return CandidateActionability(False, "数据时间缺少时区，仅供观察")
    if generated_at is not None and observed_at > generated_at:
        return CandidateActionability(False, "数据时间晚于报告生成时间，仅供观察")
    if target_session is None:
        return CandidateActionability(False, "交易日历不可用，仅供观察")
    try:
        observed_session = latest_completed_xshg_session(observed_at)
    except RuntimeError:
        return CandidateActionability(False, "交易日历不可用，仅供观察")
    if observed_session < target_session:
        return CandidateActionability(
            False,
            f"数据交易日{observed_session.isoformat()}早于预期交易日{target_session.isoformat()}，仅供观察",
        )
    if observed_session > target_session:
        return CandidateActionability(False, "数据交易日晚于报告会话，仅供观察")
    if candidate.warning.strip():
        return CandidateActionability(False, candidate.warning.strip())
    trigger = candidate.trigger.strip()
    if not trigger or any(marker in trigger.lower() for marker in ("观望", "仅供观察", "watch only", "stale")):
        return CandidateActionability(False, "仅供观察")
    return CandidateActionability(True)


def _candidate_view(
    candidate: Candidate,
    target_session: date | None,
    generated_at: datetime | None,
) -> _CandidateView:
    policy = _evaluate_candidate_for_session(
        candidate,
        target_session=target_session,
        generated_at=generated_at,
    )
    return _CandidateView(
        code=candidate.code,
        name=candidate.name,
        horizon=candidate.horizon,
        score=f"{candidate.score:.1f}",
        close=f"{candidate.close:.2f}",
        trigger=candidate.trigger if policy.actionable else "已抑制",
        stop=f"{candidate.stop_price:.2f}" if policy.actionable else "已抑制",
        target=f"{candidate.target_price:.2f}" if policy.actionable else "已抑制",
        rules="、".join(candidate.matched_rules),
        observed_at=_format_timestamp(candidate.observed_at),
        source=candidate.source,
        actionable=policy.actionable,
        warning=policy.reason,
    )


def _morning_rows(items: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(item.get("code", "")),
            str(item.get("name", "")),
            str(item.get("status", "继续观察")),
        )
        for item in items
    )


def _plain_text(
    mode: ReportMode,
    report_date: date,
    sections: Sequence[_ModuleView],
    short_title: str,
    short_candidates: Sequence[_CandidateView],
    swing_title: str,
    swing_candidates: Sequence[_CandidateView],
    morning_rows: Sequence[tuple[str, str, str]],
) -> str:
    lines = [build_subject(mode, report_date), ""]
    for section in sections:
        lines.extend((section.title, f"数据时间：{section.observed_at}"))
        if section.status != "ok":
            lines.append(f"模块状态：{section.status}")
        lines.extend(f"{label}：{value}" for label, value in section.rows)
        lines.extend(f"警告：{warning}" for warning in section.warnings)
        lines.append("")
    if mode is ReportMode.POSTMARKET:
        lines.append("早盘候选跟踪")
        if morning_rows:
            lines.extend(f"{code} {name}：{status}" for code, name, status in morning_rows)
        else:
            lines.append(_NO_MORNING_STATUS)
        lines.append("")
    for title, candidates in ((short_title, short_candidates), (swing_title, swing_candidates)):
        lines.append(title)
        if not candidates:
            lines.append("暂无候选")
        for item in candidates:
            lines.append(f"{item.code} {item.name}（{item.horizon}，评分 {item.score}）")
            lines.append(f"状态：{'等待人工确认' if item.actionable else '仅供观察'}")
            lines.append(f"触发条件：{item.trigger}；止损：{item.stop}；目标：{item.target}")
            lines.append(f"匹配规则：{item.rules}；数据时间：{item.observed_at}；来源：{item.source}")
            if item.warning:
                lines.append(f"警告：{item.warning}")
        lines.append("")
    lines.extend(_DISCLAIMER_LINES)
    return "\n".join(lines)


def render_report(
    mode: ReportMode,
    report_date: date,
    *,
    modules: Mapping[str, ModuleResult],
    short_term_candidates: Sequence[Candidate] = (),
    swing_candidates: Sequence[Candidate] = (),
    morning_candidates: Sequence[Mapping[str, Any]] = (),
    subject_prefix: str | None = None,
    generated_at: datetime | None = None,
) -> RenderedReport:
    """Render one of the two collaborative report modes from structured results."""

    normalized_mode = ReportMode(mode)
    try:
        target_session = report_data_session(normalized_mode, report_date, generated_at)
    except RuntimeError as exc:
        if str(exc) != "trading calendar unavailable":
            raise
        target_session = None
    if normalized_mode is ReportMode.PREMARKET:
        sections = _module_sections(modules, ("global", "gold", "portfolio"))
        short_title, swing_title = "短线候选池", "波段候选池"
    else:
        sections = _module_sections(modules, ("market", "portfolio", "backtests", "gold"))
        short_title, swing_title = "下一交易日短线池", "下一交易日波段池"

    short_views = tuple(
        _candidate_view(item, target_session, generated_at)
        for item in short_term_candidates
    )
    swing_views = tuple(
        _candidate_view(item, target_session, generated_at)
        for item in swing_candidates
    )
    morning_rows = _morning_rows(morning_candidates)
    subject = build_subject(normalized_mode, report_date, prefix=subject_prefix)
    template = _ENVIRONMENT.get_template("collaborative_report.html.j2")
    html = template.render(
        mode=normalized_mode.value,
        subject=subject,
        report_date=report_date.isoformat(),
        sections=sections,
        short_title=short_title,
        swing_title=swing_title,
        short_candidates=short_views,
        swing_candidates=swing_views,
        morning_rows=morning_rows,
        no_morning_status=_NO_MORNING_STATUS,
        disclaimer_lines=_DISCLAIMER_LINES,
    )
    text = _plain_text(
        normalized_mode,
        report_date,
        sections,
        short_title,
        short_views,
        swing_title,
        swing_views,
        morning_rows,
    )
    return RenderedReport(subject=subject, html=html, text=text)
