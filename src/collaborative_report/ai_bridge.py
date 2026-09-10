"""Safe adapter from collaborative reports to the stock-analysis pipeline."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from .models import ModuleResult


_EMPTY_PAYLOAD = MappingProxyType({})
DEFAULT_AI_CHILD_TIMEOUT_SECONDS = 300.0
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


class IsolatedPipelineError(RuntimeError):
    """Categorical production child failure without provider detail."""


def _json_safe_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))


def _isolated_pipeline_child(connection, codes, observed_at, pipeline_factory) -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        logging.disable(logging.CRITICAL)
        pipeline = pipeline_factory()
        results = pipeline.run(
            stock_codes=list(codes),
            send_notification=False,
            merge_notification=False,
            current_time=observed_at,
            save_report=False,
        )
        projected: list[dict[str, Any]] = []
        for result in results:
            if result is None:
                continue
            code = getattr(result, "code", None)
            if not isinstance(code, str):
                continue
            success = bool(getattr(result, "success", False))
            item: dict[str, Any] = {"code": code, "success": success, "projection": None}
            if success:
                try:
                    item["projection"] = _json_safe_projection(_project(result))
                except Exception:
                    item["projection"] = None
            projected.append(item)
        connection.send({"ok": True, "results": projected})
    except BaseException:
        try:
            connection.send({"ok": False, "error": "ai_child_failed"})
        except Exception:
            pass
    finally:
        connection.close()
        os.close(devnull)


def _run_isolated_pipeline(
    codes: list[str],
    observed_at: datetime,
    *,
    pipeline_factory=create_pipeline,
    timeout_seconds: float = DEFAULT_AI_CHILD_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run production AI in a spawned process with all child output discarded."""

    multiprocessing.freeze_support()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_pipeline_child,
        args=(sender, tuple(codes), observed_at, pipeline_factory),
        name="collaborative-ai-worker",
    )
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                raise IsolatedPipelineError("ai_child_timeout")
            raise IsolatedPipelineError("ai_child_failed")
        try:
            payload = receiver.recv()
        except (EOFError, OSError):
            raise IsolatedPipelineError("ai_child_failed") from None
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise IsolatedPipelineError("ai_child_failed")
        results = payload.get("results")
        if not isinstance(results, list):
            raise IsolatedPipelineError("ai_child_failed")
        return results
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)


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
        if pipeline is None:
            results = _run_isolated_pipeline(unique_codes, timestamp)
        else:
            results = pipeline.run(
                stock_codes=unique_codes,
                send_notification=False,
                merge_notification=False,
                current_time=timestamp,
                save_report=False,
            )
        requested = set(unique_codes)
        successful: dict[str, Mapping[str, Any]] = {}
        unsuccessful: set[str] = set()
        projection_failed: set[str] = set()
        for result in results:
            if result is None:
                continue
            isolated_result = isinstance(result, dict) and set(result) == {
                "code", "projection", "success"
            }
            code = result.get("code") if isolated_result else getattr(result, "code", None)
            if code not in requested or code in successful:
                continue
            try:
                success = result.get("success") if isolated_result else getattr(result, "success", False)
                if not bool(success):
                    unsuccessful.add(code)
                    continue
                if isolated_result:
                    projection = result.get("projection")
                    if not isinstance(projection, dict):
                        raise ValueError
                    successful[code] = _freeze(projection)
                else:
                    successful[code] = _project(result)
            except Exception:
                projection_failed.add(code)

        ordered_payload = MappingProxyType({code: successful[code] for code in unique_codes if code in successful})
        warnings: list[str] = []
        for code in unique_codes:
            if code in successful:
                continue
            if code in projection_failed:
                warning_code = "ai_projection_failed"
            elif code in unsuccessful:
                warning_code = "ai_result_unsuccessful"
            else:
                warning_code = "ai_result_missing"
            warnings.append(f"{warning_code}:{code}")
        status = "ok" if len(successful) == len(unique_codes) else "partial"
        return ModuleResult(
            name="ai",
            status=status,
            observed_at=timestamp,
            payload=ordered_payload,
            warnings=tuple(warnings),
        )
    except Exception as exc:
        failure = str(exc) if isinstance(exc, IsolatedPipelineError) else type(exc).__name__
        return ModuleResult(
            name="ai",
            status="unavailable",
            observed_at=timestamp,
            payload=_EMPTY_PAYLOAD,
            warnings=(f"AI分析暂不可用（{failure}）",),
        )
