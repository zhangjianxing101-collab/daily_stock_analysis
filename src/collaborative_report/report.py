"""Compose collaborative A-share reports as HTML and plain text."""

from __future__ import annotations

import math
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
    "ths_market_evidence": "同花顺市场证据",
    "ths_financial_evidence": "同花顺基本面证据",
    "decision_summary": "决策摘要",
}
_SECTOR_TITLES = {
    "industry_sectors": "行业板块",
    "concept_sectors": "概念板块",
}
_ROTATION_LABELS = {
    "first_observation": "首次观察",
    "new_start": "新启动",
    "continuing": "延续",
    "accelerating": "加速",
    "diverging": "分化",
    "retreating": "退潮",
    "unavailable": "不可用",
}
_LEVEL_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
    "unavailable": "不可用",
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
    sector_context: tuple[str, ...]


@dataclass(frozen=True)
class _SectorView:
    rank: str
    name: str
    change: str
    breadth: str
    activity: str
    leader: str
    leader_change: str
    rotation: str
    persistence: str
    crowding: str


@dataclass(frozen=True)
class _SectorModuleView:
    title: str
    status: str
    observed_at: str
    coverage: str
    strongest: tuple[_SectorView, ...]
    weakest: tuple[_SectorView, ...]
    watch: tuple[_SectorView, ...]
    rotation_summary: str
    warnings: tuple[str, ...]


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
    modules: Mapping[str, ModuleResult],
    primary_keys: Sequence[str],
    *,
    excluded_keys: Sequence[str] = (),
    include_remaining: bool = True,
) -> tuple[_ModuleView, ...]:
    sections = [
        _module_view(_MODULE_TITLES[key], modules.get(key)) for key in primary_keys
    ]
    if not include_remaining:
        return tuple(sections)
    excluded = set(primary_keys) | set(excluded_keys)
    for key, result in modules.items():
        if key in excluded:
            continue
        title = _MODULE_TITLES.get(key, result.name or key)
        sections.append(_module_view(title, result))
    return tuple(sections)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _format_percent(value: Any, digits: int) -> str:
    normalized = _finite_number(value)
    return "不可用" if normalized is None else f"{normalized:.{digits}f}%"


def _sector_row_view(row: Any, *, expected_type: str) -> _SectorView | None:
    if not isinstance(row, Mapping):
        return None
    required = {
        "sector_type", "rank", "name", "change_pct", "breadth_pct",
        "activity_percentile", "leader_name", "leader_code", "leader_change_pct",
        "rotation", "persistence", "crowding_risk",
    }
    if not required.issubset(row) or row.get("sector_type") != expected_type:
        return None
    rank = row.get("rank")
    name = row.get("name")
    change = _finite_number(row.get("change_pct"))
    rotation = row.get("rotation")
    persistence = row.get("persistence")
    crowding = row.get("crowding_risk")
    if (
        type(rank) is not int or rank <= 0
        or not isinstance(name, str) or not name.strip()
        or change is None
        or rotation not in _ROTATION_LABELS
        or persistence not in _LEVEL_LABELS
        or crowding not in _LEVEL_LABELS
    ):
        return None
    leader_name = row.get("leader_name")
    leader_code = row.get("leader_code")
    leader_parts = [
        value.strip()
        for value in (leader_name, leader_code)
        if isinstance(value, str) and value.strip()
    ]
    return _SectorView(
        rank=str(rank),
        name=name.strip(),
        change=f"{change:.2f}%",
        breadth=_format_percent(row.get("breadth_pct"), 1),
        activity=_format_percent(row.get("activity_percentile"), 1),
        leader="（".join(leader_parts) + ("）" if len(leader_parts) == 2 else "") if leader_parts else "不可用",
        leader_change=_format_percent(row.get("leader_change_pct"), 2),
        rotation=_ROTATION_LABELS[str(rotation)],
        persistence=_LEVEL_LABELS[str(persistence)],
        crowding=_LEVEL_LABELS[str(crowding)],
    )


def _sector_rows_view(value: Any, *, expected_type: str) -> tuple[_SectorView, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(
        view
        for row in value
        for view in (_sector_row_view(row, expected_type=expected_type),)
        if view is not None
    )


def _sector_module_view(key: str, result: ModuleResult) -> _SectorModuleView:
    expected_type = key.removesuffix("_sectors")
    payload = result.payload if isinstance(result.payload, Mapping) else {}
    strongest = _sector_rows_view(payload.get("strongest"), expected_type=expected_type)
    weakest = _sector_rows_view(payload.get("weakest"), expected_type=expected_type)
    watch = _sector_rows_view(payload.get("watch"), expected_type=expected_type)
    valid_count = payload.get("valid_count")
    coverage = (
        f"有效板块数：{valid_count}；强榜：{len(strongest)}；弱榜：{len(weakest)}"
        if type(valid_count) is int and valid_count >= 0
        else f"有效板块数：不可用；强榜：{len(strongest)}；弱榜：{len(weakest)}"
    )
    rotation_counts: dict[str, int] = {}
    for row in strongest:
        rotation_counts[row.rotation] = rotation_counts.get(row.rotation, 0) + 1
    rotation_summary = "、".join(
        f"{label} {count}"
        for label, count in rotation_counts.items()
    ) or "不可用"
    return _SectorModuleView(
        title=_SECTOR_TITLES[key],
        status=result.status,
        observed_at=_format_timestamp(result.observed_at),
        coverage=coverage,
        strongest=strongest,
        weakest=weakest,
        watch=watch,
        rotation_summary=rotation_summary,
        warnings=tuple(dict.fromkeys(str(warning) for warning in result.warnings)),
    )


def _sector_module_views(modules: Mapping[str, ModuleResult]) -> tuple[_SectorModuleView, ...]:
    return tuple(
        _sector_module_view(key, modules[key])
        for key in ("industry_sectors", "concept_sectors")
        if key in modules
    )


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
    sector_context: list[str] = []
    if candidate.industry_sector.strip():
        sector_context.append(f"行业：{candidate.industry_sector.strip()}")
    concepts = tuple(item.strip() for item in candidate.concept_sectors if item.strip())
    if concepts:
        sector_context.append(f"概念：{'、'.join(concepts)}")
    if candidate.sector_rotation in _ROTATION_LABELS:
        sector_context.append(f"轮动：{_ROTATION_LABELS[candidate.sector_rotation]}")
    if candidate.sector_persistence in _LEVEL_LABELS:
        sector_context.append(f"持续性：{_LEVEL_LABELS[candidate.sector_persistence]}")
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
        sector_context=tuple(sector_context),
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
    leading_sections: Sequence[_ModuleView],
    sector_sections: Sequence[_SectorModuleView],
    trailing_sections: Sequence[_ModuleView],
    short_title: str,
    short_candidates: Sequence[_CandidateView],
    swing_title: str,
    swing_candidates: Sequence[_CandidateView],
    morning_rows: Sequence[tuple[str, str, str]],
) -> str:
    lines = [build_subject(mode, report_date), ""]

    def append_module(section: _ModuleView) -> None:
        lines.extend((section.title, f"数据时间：{section.observed_at}"))
        if section.status != "ok":
            lines.append(f"模块状态：{section.status}")
        lines.extend(f"{label}：{value}" for label, value in section.rows)
        lines.extend(f"警告：{warning}" for warning in section.warnings)
        lines.append("")

    def append_sector_table(title: str, rows: Sequence[_SectorView]) -> None:
        lines.append(title)
        lines.append("排名 | 板块 | 涨跌幅 | 宽度 | 活跃度 | 领涨标的 | 领涨涨跌幅 | 轮动 | 持续性 | 拥挤风险")
        if not rows:
            lines.append("暂无可用数据")
        for row in rows:
            lines.append(
                " | ".join((
                    row.rank, row.name, row.change, row.breadth, row.activity,
                    row.leader, row.leader_change, row.rotation, row.persistence, row.crowding,
                ))
            )

    for section in leading_sections:
        append_module(section)
    for section in sector_sections:
        lines.extend((section.title, f"数据时间：{section.observed_at}", section.coverage))
        if section.status != "ok":
            lines.append(f"模块状态：{section.status}")
        append_sector_table("强势榜", section.strongest)
        append_sector_table("弱势榜", section.weakest)
        lines.append(f"轮动摘要：{section.rotation_summary}")
        lines.append("下一交易日观察")
        if section.watch:
            lines.extend(
                f"{row.name}：轮动 {row.rotation}；持续性 {row.persistence}；拥挤风险 {row.crowding}"
                for row in section.watch
            )
        else:
            lines.append("暂无基于持续性的观察项")
        lines.extend(f"警告：{warning}" for warning in section.warnings)
        lines.append("")
    for section in trailing_sections:
        append_module(section)
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
            if item.sector_context:
                lines.append("；".join(item.sector_context))
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
        leading_sections = _module_sections(modules, ("global", "gold", "portfolio"))
        sector_sections: tuple[_SectorModuleView, ...] = ()
        trailing_sections: tuple[_ModuleView, ...] = ()
        short_title, swing_title = "短线候选池", "波段候选池"
    else:
        special_keys = ("decision_summary", "market", "industry_sectors", "concept_sectors")
        leading_sections = _module_sections(
            modules,
            ("decision_summary", "market"),
            excluded_keys=("industry_sectors", "concept_sectors", "portfolio", "backtests", "gold"),
            include_remaining=False,
        )
        sector_sections = _sector_module_views(modules)
        trailing_sections = _module_sections(
            modules,
            ("portfolio", "backtests", "gold"),
            excluded_keys=special_keys,
        )
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
        leading_sections=leading_sections,
        sector_sections=sector_sections,
        trailing_sections=trailing_sections,
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
        leading_sections,
        sector_sections,
        trailing_sections,
        short_title,
        short_views,
        swing_title,
        swing_views,
        morning_rows,
    )
    return RenderedReport(subject=subject, html=html, text=text)
