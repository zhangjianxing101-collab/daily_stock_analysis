import importlib
import inspect
import sys
from datetime import datetime, timezone
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


OBSERVED_AT = datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc)


def result(code: str, *, success: bool = True, conclusion: str | None = None, **overrides):
    values = {
        "code": code,
        "analysis_summary": f"summary {code}",
        "action": "hold",
        "action_label": "持有",
        "operation_advice": "继续观察",
        "confidence_level": "中",
        "news_summary": "news",
        "risk_warning": "risk",
        "fundamental_analysis": "fundamental",
        "current_price": 12.3,
        "change_pct": 1.2,
        "data_sources": "market,news",
        "success": success,
        "dashboard": {"secret": "must not leak"},
        "raw_response": "raw provider response",
        "model_used": "provider/model",
    }
    values.update(overrides)
    item = SimpleNamespace(**values)
    item.get_core_conclusion = Mock(return_value=conclusion or values["analysis_summary"])
    return item


def test_enrich_codes_exact_signature_deduplicates_and_calls_pipeline_once() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    pipeline = Mock()
    pipeline.run.return_value = [result("000002"), result("000001")]

    output = enrich_codes(
        [" 000002 ", "000001", "000002", "bad", 600000, "１２３４５６", "12345", "1234567"],
        pipeline,
        observed_at=OBSERVED_AT,
    )

    assert list(inspect.signature(enrich_codes).parameters) == ["codes", "pipeline", "observed_at"]
    assert inspect.signature(enrich_codes).parameters["pipeline"].default is None
    assert inspect.signature(enrich_codes).parameters["observed_at"].kind is inspect.Parameter.KEYWORD_ONLY
    pipeline.run.assert_called_once_with(
        stock_codes=["000002", "000001"],
        send_notification=False,
        merge_notification=False,
        current_time=OBSERVED_AT,
        save_report=False,
    )
    assert output.name == "ai"
    assert output.status == "ok"
    assert output.observed_at == OBSERVED_AT
    assert list(output.payload) == ["000002", "000001"]
    assert output.warnings == ()


def test_success_projection_is_allowlisted_and_uses_core_conclusion() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    analysis = result("000001", conclusion="dashboard conclusion")
    pipeline = Mock()
    pipeline.run.return_value = [analysis]

    projected = enrich_codes(["000001"], pipeline, observed_at=OBSERVED_AT).payload["000001"]

    assert projected == {
        "conclusion": "dashboard conclusion",
        "action": "hold",
        "action_label": "持有",
        "operation_advice": "继续观察",
        "confidence_level": "中",
        "news_summary": "news",
        "risk_warning": "risk",
        "fundamental_analysis": "fundamental",
        "current_price": 12.3,
        "change_pct": 1.2,
        "data_sources": "market,news",
        "success": True,
    }
    assert "dashboard" not in projected
    assert "raw_response" not in projected
    assert "model_used" not in projected
    analysis.get_core_conclusion.assert_called_once_with()


def test_analysis_summary_is_used_when_core_conclusion_method_is_absent() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    analysis = result("000001")
    del analysis.get_core_conclusion
    pipeline = Mock()
    pipeline.run.return_value = [analysis]

    projected = enrich_codes(["000001"], pipeline, observed_at=OBSERVED_AT).payload["000001"]

    assert projected["conclusion"] == "summary 000001"


def test_unrequested_results_are_discarded_and_first_successful_duplicate_wins() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    failed = result("000001", success=False, analysis_summary="failed")
    first = result("000001", analysis_summary="first", conclusion="first")
    duplicate = result("000001", analysis_summary="second", conclusion="second")
    pipeline = Mock()
    pipeline.run.return_value = [result("999999"), failed, first, duplicate]

    output = enrich_codes(["000001"], pipeline, observed_at=OBSERVED_AT)

    assert output.status == "ok"
    assert output.payload["000001"]["conclusion"] == "first"
    assert output.warnings == ()
    duplicate.get_core_conclusion.assert_not_called()


def test_mock_only_unsuccessful_and_production_missing_results_have_distinct_warnings() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    pipeline = Mock()
    pipeline.run.return_value = [
        result("000001"),
        None,
        result("000002", success=False, error_message="secret provider URL"),
    ]

    output = enrich_codes(["000001", "000002", "000003"], pipeline, observed_at=OBSERVED_AT)

    assert output.status == "partial"
    assert list(output.payload) == ["000001"]
    assert output.warnings == (
        "ai_result_unsuccessful:000002",
        "ai_result_missing:000003",
    )
    assert "secret" not in " ".join(output.warnings)
    assert "URL" not in " ".join(output.warnings)


def test_no_successful_results_are_partial_without_constructing_production_pipeline() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    pipeline = Mock()
    pipeline.run.return_value = []

    output = enrich_codes(["000001"], pipeline, observed_at=OBSERVED_AT)

    assert output.status == "partial"
    assert output.payload == {}
    assert output.warnings == ("ai_result_missing:000001",)


def test_projection_failures_are_isolated_per_result_without_error_details() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    conclusion_failure = result("000001")
    conclusion_failure.get_core_conclusion.side_effect = RuntimeError("secret conclusion failure")

    class MalformedResult:
        code = "000002"
        success = True
        analysis_summary = "malformed"

        def get_core_conclusion(self):
            return self.analysis_summary

        @property
        def action(self):
            raise ValueError("provider payload malformed secret=token")

    pipeline = Mock()
    pipeline.run.return_value = [conclusion_failure, MalformedResult(), result("000003")]

    output = enrich_codes(["000001", "000002", "000003"], pipeline, observed_at=OBSERVED_AT)

    assert output.status == "partial"
    assert list(output.payload) == ["000003"]
    assert output.warnings == (
        "ai_projection_failed:000001",
        "ai_projection_failed:000002",
    )
    serialized = repr(output)
    assert "RuntimeError" not in serialized
    assert "ValueError" not in serialized
    assert "secret" not in serialized
    assert "provider payload" not in serialized


def test_pipeline_exception_is_sanitized_in_full_result_report() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    pipeline = Mock()
    pipeline.run.side_effect = RuntimeError("provider unavailable secret=https://token.example")

    output = enrich_codes(["000001"], pipeline, observed_at=OBSERVED_AT)
    report = repr(
        {
            "name": output.name,
            "status": output.status,
            "observed_at": output.observed_at,
            "payload": dict(output.payload),
            "warnings": output.warnings,
        }
    )

    assert output.status == "unavailable"
    assert output.payload == {}
    assert output.warnings == ("AI分析暂不可用（RuntimeError）",)
    serialized = repr(output) + report
    assert "provider unavailable" not in serialized
    assert "secret" not in serialized
    assert "token.example" not in serialized


def test_empty_valid_codes_skip_with_immutable_payload_and_fixed_warning(monkeypatch) -> None:
    import src.collaborative_report.ai_bridge as ai_bridge

    factory = Mock()
    monkeypatch.setattr(ai_bridge, "create_pipeline", factory)

    output = ai_bridge.enrich_codes(["", "  ", "abcdef", 123456], observed_at=OBSERVED_AT)

    assert output.name == "ai"
    assert output.status == "skipped"
    assert output.payload == {}
    assert isinstance(output.payload, MappingProxyType)
    assert output.warnings == ("ai_no_valid_codes",)
    factory.assert_not_called()


def test_candidate_objects_are_not_accepted_or_mutated() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes
    from src.collaborative_report.models import Candidate

    candidate = Candidate(
        code="000001",
        name="candidate",
        horizon="short",
        score=88.0,
        close=10.0,
        trigger="breakout",
        stop_price=9.0,
        target_price=12.0,
        matched_rules=("rule",),
        observed_at=OBSERVED_AT,
        source="test",
    )
    pipeline = Mock()
    pipeline.run.return_value = [result("000001", current_price=99.0)]

    output = enrich_codes([candidate, "000001"], pipeline, observed_at=OBSERVED_AT)  # type: ignore[list-item]

    assert (candidate.close, candidate.score) == (10.0, 88.0)
    assert output.payload["000001"]["current_price"] == 99.0
    pipeline.run.assert_called_once_with(
        stock_codes=["000001"],
        send_notification=False,
        merge_notification=False,
        current_time=OBSERVED_AT,
        save_report=False,
    )


def test_default_clock_is_aware_and_supplied_naive_time_is_rejected(monkeypatch) -> None:
    import src.collaborative_report.ai_bridge as ai_bridge

    clock_time = datetime(2026, 8, 19, 9, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(ai_bridge, "_utc_now", Mock(return_value=clock_time))
    pipeline = Mock()
    pipeline.run.return_value = []

    output = ai_bridge.enrich_codes([], pipeline)

    assert output.observed_at == clock_time
    assert output.observed_at.utcoffset() is not None
    with pytest.raises(ValueError, match="^observed_at must be timezone-aware$"):
        ai_bridge.enrich_codes([], pipeline, observed_at=datetime(2026, 8, 19, 9, 30))


def test_payload_and_nested_values_are_immutable() -> None:
    from src.collaborative_report.ai_bridge import enrich_codes

    pipeline = Mock()
    pipeline.run.return_value = [
        result("000001", data_sources={"quotes": ["primary", {"fallback": ["secondary"]}]})
    ]

    output = enrich_codes(["000001"], pipeline, observed_at=OBSERVED_AT)

    with pytest.raises(TypeError):
        output.payload["000002"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        output.payload["000001"]["action"] = "buy"  # type: ignore[index]
    sources = output.payload["000001"]["data_sources"]
    assert isinstance(sources, MappingProxyType)
    assert sources["quotes"][0] == "primary"
    assert isinstance(sources["quotes"], tuple)
    assert isinstance(sources["quotes"][1], MappingProxyType)
    with pytest.raises(TypeError):
        sources["quotes"][1]["fallback"] = ()


def test_create_pipeline_imports_dependencies_lazily(monkeypatch) -> None:
    sys.modules.pop("src.collaborative_report.ai_bridge", None)
    monkeypatch.delitem(sys.modules, "src.config", raising=False)
    monkeypatch.delitem(sys.modules, "src.core.pipeline", raising=False)

    ai_bridge = importlib.import_module("src.collaborative_report.ai_bridge")

    assert "src.config" not in sys.modules
    assert "src.core.pipeline" not in sys.modules

    config = object()
    pipeline = object()
    get_config = Mock(return_value=config)
    constructor = Mock(return_value=pipeline)
    config_module = ModuleType("src.config")
    config_module.get_config = get_config
    pipeline_module = ModuleType("src.core.pipeline")
    pipeline_module.StockAnalysisPipeline = constructor
    monkeypatch.setitem(sys.modules, "src.config", config_module)
    monkeypatch.setitem(sys.modules, "src.core.pipeline", pipeline_module)

    assert ai_bridge.create_pipeline() is pipeline
    get_config.assert_called_once_with()
    constructor.assert_called_once_with(config=config)
