"""Safe adapter from collaborative reports to the stock-analysis pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from .models import ModuleResult


_EMPTY_PAYLOAD = MappingProxyType({})
_PROJECTED_FIELDS = (
    "action",
    "action_label",
    "operation_advice",
    "confidence_level",
    "news_summary",
    "risk_warning",
    "fundamental_analysis",
    "current_price",
    "change_pct",
    "data_sources",
    "success",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _normalize_codes(codes: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in codes:
        if not isinstance(value, str):
            continue
        code = value.strip()
        if len(code) != 6 or not code.isascii() or not code.isdigit() or code in seen:
            continue
        seen.add(code)
        normalized.append(code)
    return normalized


def _project(result: Any) -> Mapping[str, Any]:
    conclusion_getter = getattr(result, "get_core_conclusion", None)
    conclusion = conclusion_getter() if callable(conclusion_getter) else getattr(result, "analysis_summary", None)
    values = {"conclusion": _freeze(conclusion)}
    values.update({field: _freeze(getattr(result, field, None)) for field in _PROJECTED_FIELDS})
    return MappingProxyType(values)


def create_pipeline():
    """Construct the production pipeline without loading it during module import."""

    from src.config import get_config
    from src.core.pipeline import StockAnalysisPipeline

    return StockAnalysisPipeline(config=get_config())


def enrich_codes(
    codes: Iterable[str],
    pipeline=None,
    *,
    observed_at=None,
) -> ModuleResult:
    """Run AI analysis for valid codes and return a sanitized immutable projection."""

    timestamp = _utc_now() if observed_at is None else observed_at
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")

    unique_codes = _normalize_codes(codes)
    if not unique_codes:
        return ModuleResult(
            name="ai",
            status="skipped",
            observed_at=timestamp,
            payload=_EMPTY_PAYLOAD,
            warnings=("ai_no_valid_codes",),
        )

    try:
        active_pipeline = create_pipeline() if pipeline is None else pipeline
        results = active_pipeline.run(
            stock_codes=unique_codes,
            send_notification=False,
            merge_notification=False,
        )
        requested = set(unique_codes)
        successful: dict[str, Mapping[str, Any]] = {}
        unsuccessful: set[str] = set()
        saw_none = False
        for result in results:
            if result is None:
                saw_none = True
                continue
            code = getattr(result, "code", None)
            if code not in requested or code in successful:
                continue
            if not bool(getattr(result, "success", False)):
                unsuccessful.add(code)
                continue
            successful[code] = _project(result)

        ordered_payload = MappingProxyType({code: successful[code] for code in unique_codes if code in successful})
        warnings: list[str] = ["ai_result_none"] if saw_none else []
        for code in unique_codes:
            if code in successful:
                continue
            warning_code = "ai_result_unsuccessful" if code in unsuccessful else "ai_result_missing"
            warnings.append(f"{code}:{warning_code}")
        status = "ok" if len(successful) == len(unique_codes) else "partial"
        return ModuleResult(
            name="ai",
            status=status,
            observed_at=timestamp,
            payload=ordered_payload,
            warnings=tuple(warnings),
        )
    except Exception as exc:
        return ModuleResult(
            name="ai",
            status="unavailable",
            observed_at=timestamp,
            payload=_EMPTY_PAYLOAD,
            warnings=(f"AI分析暂不可用（{type(exc).__name__}）",),
        )
