"""Conservative market-news clues for collaborative A-share reports."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from .models import ModuleResult


_FINANCIAL_TEMPLATES = (
    "orz-cls",
    "orz-eastmoney",
    "orz-sina-finance",
    "orz-xueqiu",
)
_MACRO_TEMPLATE = "orz-baidu"
_MAX_ITEMS_PER_SOURCE = 2
_MAX_AGE = timedelta(days=3)
_MAX_FUTURE_SKEW = timedelta(minutes=15)
_EVIDENCE_WARNING = "聚合新闻仅作事件线索，关键事实及发布时间需核验原始来源"


def _production_service():
    from src.config import get_config
    from src.services.intelligence_service import IntelligenceService

    return IntelligenceService(config=get_config())


def _parse_published_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _usable_item(item: Any, *, observed_at: datetime) -> dict[str, str] | None:
    if not isinstance(item, Mapping):
        return None
    title = str(item.get("title") or "").strip()
    source = str(item.get("source") or "").strip()
    url = str(item.get("url") or "").strip()
    published_at = _parse_published_at(item.get("published_at"))
    observed_utc = observed_at.astimezone(timezone.utc)
    if (
        not title
        or not source
        or published_at is None
        or published_at < observed_utc - _MAX_AGE
        or published_at > observed_utc + _MAX_FUTURE_SKEW
    ):
        return None
    return {
        "标题": title[:300],
        "来源": source[:80],
        "聚合时间": published_at.isoformat(),
        "链接": url[:1000] if url.startswith(("http://", "https://")) else "不可用",
        "证据用途": "事件线索，未独立核验",
    }


def fetch_market_news(
    *,
    observed_at: datetime,
    service_factory: Callable[[], Any] = _production_service,
) -> ModuleResult:
    """Fetch diversified CN-market clues without treating them as verified facts."""

    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")

    try:
        service = service_factory()
        templates_payload = service.list_source_templates(market="cn", source_type="newsnow")
        templates = templates_payload.get("items", ()) if isinstance(templates_payload, Mapping) else ()
    except Exception:
        return ModuleResult("news", "unavailable", observed_at, {}, ("市场新闻线索暂不可用",))

    by_id = {
        str(item.get("template_id")): item
        for item in templates
        if isinstance(item, Mapping)
    }
    selected: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    available_sources: list[str] = []
    failed_sources = 0
    macro_count = 0

    for template_id in (*_FINANCIAL_TEMPLATES, _MACRO_TEMPLATE):
        template = by_id.get(template_id)
        if template is None:
            failed_sources += 1
            continue
        try:
            response = service.test_source(dict(template))
            raw_items = response.get("sample_items", ()) if isinstance(response, Mapping) else ()
            if not isinstance(raw_items, (tuple, list)):
                raise ValueError
            source_items = [
                normalized
                for item in raw_items
                for normalized in (_usable_item(item, observed_at=observed_at),)
                if normalized is not None
            ]
            available_sources.append(str(template.get("name") or template_id))
        except Exception:
            failed_sources += 1
            continue

        if template_id == _MACRO_TEMPLATE:
            macro_count = len(source_items)
            continue
        for item in source_items[:_MAX_ITEMS_PER_SOURCE]:
            title_key = item["标题"].casefold()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            selected.append(item)

    payload: dict[str, Any] = {
        "可用来源": available_sources,
        "财经线索数": len(selected),
        "宏观舆情线索数": macro_count,
    }
    payload.update({f"线索{index}": item for index, item in enumerate(selected, 1)})
    warnings = [_EVIDENCE_WARNING]
    if failed_sources:
        warnings.append(f"部分新闻源不可用（{failed_sources}个）")
    if not selected:
        warnings.append("未获取到可用财经新闻线索")
    status = "ok" if selected and not failed_sources else ("partial" if selected else "unavailable")
    return ModuleResult("news", status, observed_at, payload, tuple(warnings))
