"""Total orchestration, privacy boundaries, and artifacts for daily reports."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .ai_bridge import enrich_codes
from .backtest import backtest_breakout, backtest_swing
from .gold import analyze_gold
from .mailer import DeliveryInDoubtError, DeliveryNotAcceptedError, send_with_retry
from .market_data import MarketDataGateway, MarketDataset
from .models import Candidate, ModuleResult, Position, ReportMode
from .report import RenderedReport, render_report
from .risk import evaluate_position, suggested_board_lots
from .screener import ScreeningResult, prefilter_universe, rank_with_ths_evidence, screen_aggressive
from .session import SHANGHAI_TIMEZONE, ReportSession, build_report_session, report_data_session
from .settings import CollaborativeSettings, ThsSettings
from .ths_market_data import ThsIndexTag, ThsMarketDataClient


# Process status is intentionally binary for GitHub Actions. Detailed outcomes live
# in FinalState so successful skips do not fail a scheduled workflow.
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
MANIFEST_SCHEMA_VERSION = 1
SNAPSHOT_DAILY_CLOSE_TOLERANCE = 0.01
_TEST_SUBJECT_PREFIX = "测试"
_PRICE_CONFLICT_WARNING_CODE = "snapshot_daily_close_conflict"
_PRICE_CONFLICT_CANDIDATE_WARNING = "价格来源冲突，仅供观望"
_UNTRUSTED_SNAPSHOT_CANDIDATE_WARNING = "快照权威性不足，仅供观望"
_SNAPSHOT_AUTHORITY_WARNING_CODE = "snapshot_timestamp_untrusted"
_SNAPSHOT_UNAVAILABLE_OBSERVATION_WARNING = "快照来源时间不可用，所有建议仅供观察"
_REPORT_KEY_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})-(premarket|postmarket)")
_POSTMARKET_MAX_SOURCE_AGE = timedelta(hours=4)
_PREMARKET_MAX_SOURCE_AGE = timedelta(days=4)
_DATA_FAILURE_CODES = {
    "stale snapshot": "snapshot_acquisition_time_invalid",
    "THS source timestamp is in the future": "ths_timestamp_future",
    "THS source timestamp invalid": "ths_timestamp_invalid",
    "THS snapshot data invalid": "ths_snapshot_invalid",
    "THS snapshot missing required fields": "ths_snapshot_fields_missing",
    "THS snapshot contains unsafe prices": "ths_snapshot_prices_invalid",
    "THS snapshot contains duplicate codes": "ths_snapshot_duplicate_codes",
    "THS snapshot contains invalid values": "ths_snapshot_values_invalid",
    "snapshot provider unavailable": "snapshot_provider_unavailable",
    "snapshot provider returned empty data": "snapshot_provider_empty",
    "provider acquisition clock invalid": "provider_clock_invalid",
    "market calendar unavailable": "market_calendar_unavailable",
    "gold provider unavailable": "gold_provider_unavailable",
    "gold provider returned empty data": "gold_provider_empty",
    "daily bars insufficient": "daily_bars_insufficient",
    "daily bars invalid dates": "daily_bars_invalid_dates",
    "daily bars duplicate dates": "daily_bars_duplicate_dates",
    "daily bars non-monotonic dates": "daily_bars_unordered",
    "daily bars future dates": "daily_bars_future_dates",
    "daily bars invalid ohlc": "daily_bars_invalid_ohlc",
    "daily bars invalid volume": "daily_bars_invalid_volume",
    "daily bars missing columns": "daily_bars_missing_columns",
    "daily bars stale": "daily_bars_stale",
}
_SAFE_DATA_FAILURE_CODES = frozenset(_DATA_FAILURE_CODES.values()) | {
    "snapshot_data_failed", "gold_data_failed", "gold_analysis_failed",
    "ths_snapshot_unquoted_rows_excluded", "snapshot_screening_fields_incomplete",
}


class FinalState(str, Enum):
    SENT = "sent"
    TEST_SENT = "test_sent"
    PREVIEWED = "previewed"
    DUPLICATE_SKIP = "duplicate_skip"
    NON_TRADING_DAY_SKIP = "non_trading_day_skip"
    OPERATOR_ACTION_REQUIRED = "operator_action_required"
    HARD_FAILURE = "hard_failure"


class _DeliveryConfigurationError(DeliveryNotAcceptedError):
    def __init__(self, _message: str = "") -> None:
        super().__init__(retryable=False, stage="configuration")
        self.args = ("email configuration unavailable",)


class _DeliveryNotAcceptedError(DeliveryNotAcceptedError):
    """Backward-compatible injected safe-preacceptance failure for runner tests."""

    def __init__(self, _message: str = "") -> None:
        super().__init__(retryable=True, stage="preacceptance")


class DeliveryStateError(RuntimeError):
    """Raised when local delivery ownership cannot be established safely."""


class DeliveryState(str, Enum):
    CLAIMED = "claimed"
    SENDING = "sending"
    SENT = "sent"
    IN_DOUBT = "in_doubt"
    FAILED = "failed"


@dataclass(frozen=True)
class DeliveryRecord:
    report_key: str
    state: DeliveryState
    claim_id: str
    transitioned_at: datetime
    attempt_id: str | None


@dataclass(frozen=True)
class ArtifactPaths:
    html_path: Path
    text_path: Path
    manifest_path: Path
    index_path: Path | None = None
    attempt_id: str | None = None


def _canonical_report_key(report_key: str) -> str:
    if not isinstance(report_key, str) or _REPORT_KEY_PATTERN.fullmatch(report_key) is None:
        raise ValueError("invalid report key")
    date_text, mode = report_key.rsplit("-", 1)
    try:
        parsed = date.fromisoformat(date_text)
    except ValueError:
        raise ValueError("invalid report key") from None
    if f"{parsed.isoformat()}-{mode}" != report_key:
        raise ValueError("invalid report key")
    return report_key


def _contained_path(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve(strict=False)
    candidate = root.joinpath(*parts)
    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError("path escapes output root")
    return candidate


class LocalDeliveryLedger:
    """Cross-process production-delivery claims keyed by report identity."""

    def __init__(self, output_dir: Path | str) -> None:
        self.root = Path(output_dir)
        self.directory = _contained_path(self.root, ".delivery-ledger")

    def _path(self, report_key: str) -> Path:
        key = _canonical_report_key(report_key)
        return _contained_path(self.root, ".delivery-ledger", f"{key}.json")

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        lock_path = _contained_path(self.root, ".delivery-ledger", ".lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read(self, report_key: str) -> Mapping[str, Any] | None:
        path = self._path(report_key)
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            raise DeliveryStateError("delivery state unavailable") from exc
        try:
            payload = json.loads(content)
            state = DeliveryState(payload["state"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise DeliveryStateError("delivery state unavailable") from exc
        if set(payload) != {"attempt_id", "claim_id", "report_key", "state", "transitioned_at"}:
            raise DeliveryStateError("delivery state unavailable")
        if payload.get("report_key") != report_key:
            raise DeliveryStateError("delivery state unavailable")
        claim_id = payload.get("claim_id")
        if not isinstance(claim_id, str) or re.fullmatch(r"[0-9a-f]{32}", claim_id) is None:
            raise DeliveryStateError("delivery state unavailable")
        try:
            transitioned_at = datetime.fromisoformat(payload["transitioned_at"])
        except (TypeError, ValueError):
            raise DeliveryStateError("delivery state unavailable") from None
        if transitioned_at.tzinfo is None or transitioned_at.utcoffset() is None:
            raise DeliveryStateError("delivery state unavailable")
        attempt_id = payload.get("attempt_id")
        if attempt_id is not None and (
            not isinstance(attempt_id, str) or re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None
        ):
            raise DeliveryStateError("delivery state unavailable")
        if state is DeliveryState.SENT and attempt_id is None:
            raise DeliveryStateError("delivery state unavailable")
        return payload

    def status(self, report_key: str) -> DeliveryState | None:
        record = self.record(report_key)
        return record.state if record else None

    def record(self, report_key: str) -> DeliveryRecord | None:
        with self._locked():
            payload = self._read(report_key)
        if payload is None:
            return None
        return DeliveryRecord(
            report_key=str(payload["report_key"]),
            state=DeliveryState(payload["state"]),
            claim_id=str(payload["claim_id"]),
            transitioned_at=datetime.fromisoformat(str(payload["transitioned_at"])),
            attempt_id=payload.get("attempt_id"),
        )

    def _write(
        self,
        report_key: str,
        state: DeliveryState,
        claim_id: str,
        transitioned_at: datetime,
        attempt_id: str | None,
    ) -> None:
        if transitioned_at.tzinfo is None or transitioned_at.utcoffset() is None:
            raise DeliveryStateError("delivery state timestamp unavailable")
        payload = {
            "claim_id": claim_id,
            "attempt_id": attempt_id,
            "report_key": report_key,
            "state": state.value,
            "transitioned_at": transitioned_at.isoformat(),
        }
        _atomic_write(
            self._path(report_key),
            json.dumps(payload, sort_keys=True) + "\n",
            mode=0o600,
        )

    def claim(
        self,
        report_key: str,
        claimed_at: datetime,
        *,
        attempt_id: str | None = None,
    ) -> str | None:
        key = _canonical_report_key(report_key)
        with self._locked():
            payload = self._read(key)
            if payload is not None and DeliveryState(payload["state"]) is not DeliveryState.FAILED:
                return None
            claim_id = uuid.uuid4().hex
            self._write(key, DeliveryState.CLAIMED, claim_id, claimed_at, attempt_id)
            return claim_id

    def _transition(
        self,
        report_key: str,
        claim_id: str,
        expected: set[DeliveryState],
        target: DeliveryState,
        transitioned_at: datetime,
        attempt_id: str | None = None,
    ) -> None:
        key = _canonical_report_key(report_key)
        with self._locked():
            payload = self._read(key)
            if (
                payload is None
                or payload.get("claim_id") != claim_id
                or DeliveryState(payload["state"]) not in expected
            ):
                raise DeliveryStateError("delivery state transition rejected")
            selected_attempt = attempt_id if attempt_id is not None else payload.get("attempt_id")
            if target is DeliveryState.SENT and selected_attempt is None:
                raise DeliveryStateError("sent state requires finalized attempt")
            self._write(key, target, claim_id, transitioned_at, selected_attempt)

    def begin_sending(self, report_key: str, claim_id: str, at: datetime) -> None:
        self._transition(report_key, claim_id, {DeliveryState.CLAIMED}, DeliveryState.SENDING, at)

    def mark_sent(self, report_key: str, claim_id: str, at: datetime, *, attempt_id: str) -> None:
        self._transition(
            report_key,
            claim_id,
            {DeliveryState.SENDING},
            DeliveryState.SENT,
            at,
            attempt_id,
        )

    def mark_in_doubt(
        self,
        report_key: str,
        claim_id: str,
        at: datetime,
        *,
        attempt_id: str | None = None,
    ) -> None:
        self._transition(
            report_key,
            claim_id,
            {DeliveryState.SENDING},
            DeliveryState.IN_DOUBT,
            at,
            attempt_id,
        )

    def mark_failed(self, report_key: str, claim_id: str, at: datetime) -> None:
        self._transition(
            report_key,
            claim_id,
            {DeliveryState.CLAIMED, DeliveryState.SENDING},
            DeliveryState.FAILED,
            at,
        )

    def reconcile(
        self,
        report_key: str,
        target: DeliveryState,
        at: datetime,
        *,
        attempt_id: str | None = None,
    ) -> None:
        if target not in {DeliveryState.FAILED, DeliveryState.SENT}:
            raise DeliveryStateError("invalid reconciliation target")
        key = _canonical_report_key(report_key)
        with self._locked():
            payload = self._read(key)
            if payload is None or DeliveryState(payload["state"]) not in {
                DeliveryState.CLAIMED,
                DeliveryState.SENDING,
                DeliveryState.IN_DOUBT,
            }:
                raise DeliveryStateError("delivery state reconciliation rejected")
            selected_attempt = attempt_id if attempt_id is not None else payload.get("attempt_id")
            if target is DeliveryState.SENT and selected_attempt is None:
                raise DeliveryStateError("sent state requires finalized attempt")
            action = "reconcile_sent" if target is DeliveryState.SENT else "reconcile_failed"
            self._append_audit(key, action, at)
            self._write(key, target, str(payload["claim_id"]), at, selected_attempt)

    def _append_audit(self, report_key: str, action: str, at: datetime) -> None:
        if at.tzinfo is None or at.utcoffset() is None:
            raise DeliveryStateError("audit timestamp unavailable")
        audit_path = _contained_path(self.root, ".delivery-ledger", "audit.jsonl")
        descriptor = os.open(audit_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            line = json.dumps(
                {"action": action, "report_key": report_key, "timestamp": at.isoformat()},
                sort_keys=True,
            ) + "\n"
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(audit_path.parent)


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    final_state: FinalState
    report_key: str = "unavailable"
    modules: Mapping[str, ModuleResult] = field(default_factory=dict)
    html_path: Path | None = None
    text_path: Path | None = None
    manifest_path: Path | None = None
    short_term_candidates: tuple[Candidate, ...] = ()
    swing_candidates: tuple[Candidate, ...] = ()
    morning_candidates: tuple[Mapping[str, Any], ...] = ()
    error_code: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """Return the only fields safe for CLI and workflow logs."""

        payload = {
            "report_key": self.report_key,
            "module_statuses": {name: result.status for name, result in self.modules.items()},
            "artifact_path": str(self.manifest_path) if self.manifest_path else None,
            "final_state": self.final_state.value,
        }
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        return payload


@dataclass(frozen=True)
class RunnerDependencies:
    settings_loader: Callable[[], CollaborativeSettings]
    session_builder: Callable[..., ReportSession]
    clock: Callable[[], datetime]
    gateway: Any
    screener: Callable[..., ScreeningResult]
    risk_evaluator: Callable[..., Any]
    short_backtest: Callable[..., Any]
    swing_backtest: Callable[..., Any]
    gold_analyzer: Callable[..., Any]
    ai_enricher: Callable[..., ModuleResult]
    renderer: Callable[..., RenderedReport]
    mail_sender: Callable[..., Any]
    data_session_resolver: Callable[..., date] = report_data_session
    sizing_evaluator: Callable[..., int] = suggested_board_lots
    ledger_factory: Callable[[Path | str], Any] = LocalDeliveryLedger
    artifact_writer: Callable[..., ArtifactPaths] | None = None
    artifact_finalizer: Callable[..., ArtifactPaths] | None = None
    artifact_publisher: Callable[..., None] | None = None


def _production_mail_sender(rendered: RenderedReport, *, test_email: bool = False) -> Any:
    from src.config import get_config
    from src.notification_sender.email_sender import EmailSender

    try:
        config = get_config()
        sender_address = str(getattr(config, "email_sender", "") or "").strip()
        password = str(getattr(config, "email_password", "") or "").strip()
        if not sender_address or not password:
            raise ValueError
        receivers = tuple(getattr(config, "email_receivers", ()) or ())
        sender = EmailSender(config)
    except Exception:
        raise _DeliveryConfigurationError("email configuration unavailable") from None
    return send_with_retry(
        sender,
        html_content=rendered.html,
        text_content=rendered.text,
        subject=rendered.subject,
        receivers=receivers or None,
    )


def default_dependencies(*, clock: Callable[[], datetime] | None = None) -> RunnerDependencies:
    from .snapshot_supplement import fetch_snapshot_supplement

    active_clock = clock or (lambda: datetime.now().astimezone())
    return RunnerDependencies(
        settings_loader=CollaborativeSettings.from_env,
        session_builder=build_report_session,
        clock=active_clock,
        gateway=MarketDataGateway(
            ths_client=ThsMarketDataClient(ThsSettings.from_env()),
            snapshot_supplement_fetcher=lambda codes: fetch_snapshot_supplement(list(codes), clock=active_clock),
            clock=active_clock,
        ),
        screener=screen_aggressive,
        risk_evaluator=evaluate_position,
        short_backtest=backtest_breakout,
        swing_backtest=backtest_swing,
        gold_analyzer=analyze_gold,
        ai_enricher=enrich_codes,
        renderer=render_report,
        mail_sender=_production_mail_sender,
        sizing_evaluator=suggested_board_lots,
    )


def _unavailable(name: str, observed_at: datetime, warning: str) -> ModuleResult:
    return ModuleResult(name=name, status="unavailable", observed_at=observed_at, payload={}, warnings=(warning,))


def _data_unavailable(
    name: str, observed_at: datetime, warning: str, error: Exception, fallback: str,
) -> ModuleResult:
    code = _DATA_FAILURE_CODES.get(str(error), fallback)
    return ModuleResult(name, "unavailable", observed_at, {}, (warning, code))


def _module(name: str, observed_at: datetime, payload: Mapping[str, Any], *warnings: str) -> ModuleResult:
    return ModuleResult(name=name, status="ok", observed_at=observed_at, payload=payload, warnings=tuple(warnings))


def _as_payload(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _as_payload(getattr(value, item.name))
            for item in fields(value)
            if item.name != "trades"
        }
    if isinstance(value, Mapping):
        return {str(key): _as_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _global_payload(dataset: MarketDataset) -> Mapping[str, Any]:
    if dataset.frame.empty:
        return {}
    rows = dataset.frame.to_dict(orient="records")
    return {str(row.get("symbol", index)): _as_payload(row) for index, row in enumerate(rows)}


def _market_payload(dataset: MarketDataset, *, authoritative: bool) -> Mapping[str, Any]:
    frame = dataset.frame
    coverage = {}
    if frame.attrs.get("quarantined_row_count", 0):
        coverage["无报价剔除记录"] = int(frame.attrs["quarantined_row_count"])
    if "screening_complete_count" in frame.attrs:
        coverage["选股字段齐全记录"] = int(frame.attrs["screening_complete_count"])
    if not authoritative:
        return {"股票数量": int(len(frame)), **coverage}
    changes = pd.to_numeric(frame.get("change_pct", pd.Series(dtype=float)), errors="coerce")
    return {
        "股票数量": int(len(frame)),
        "上涨家数": int((changes > 0).sum()),
        "下跌家数": int((changes < 0).sum()),
        **coverage,
    }


def _dataset_source_payload(dataset: MarketDataset) -> Mapping[str, Any]:
    return {
        "数据源": dataset.source,
        "采集时间": dataset.observed_at.isoformat(),
        "来源时间": dataset.source_timestamp.isoformat() if dataset.source_timestamp else "unavailable",
        "记录数": int(len(dataset.frame)),
    }


def _canonical_evidence_code(value: object) -> str | None:
    text = str(value).strip().upper()
    if len(text) >= 6 and text[:6].isdigit() and (len(text) == 6 or text[6] == "."):
        return text[:6]
    return None


def _positive_financial_indicator(dataset: MarketDataset) -> bool:
    """Recognize only documented profitability growth fields, never infer from blanks."""

    required = {"index_id", "value"}
    if not required.issubset(dataset.frame.columns):
        return False
    supported = {"net_profit_yoy_growth_ratio", "net_profit_yoy_growth"}
    for _, row in dataset.frame.iterrows():
        if str(row.get("index_id", "")).strip() not in supported:
            continue
        try:
            value = float(str(row.get("value", "")).replace("%", "").strip())
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value > 0:
            return True
    return False


def _index_environment_support(dataset: MarketDataset) -> tuple[bool, int, int]:
    for column in ("price_change_ratio_pct", "change_pct", "pct_chg", "涨跌幅"):
        if column not in dataset.frame:
            continue
        values = pd.to_numeric(dataset.frame[column], errors="coerce")
        values = values[np.isfinite(values)]
        if not values.empty:
            positive = int((values > 0).sum())
            return positive * 2 > len(values), positive, int(len(values))
    return False, 0, 0


def _latest_completed_financial_report(day: date) -> str:
    quarter = (day.month - 1) // 3
    if quarter == 0:
        return f"{day.year - 1}-4"
    return f"{day.year}-{quarter}"


def _ths_evidence(
    gateway: Any,
    candidate_codes: Sequence[str],
    *,
    trading_date: date,
    observed_at: datetime,
) -> tuple[Mapping[str, Mapping[str, Any]], ModuleResult, ModuleResult]:
    """Collect advisory THS evidence without allowing it to generate candidates."""

    evidence: dict[str, dict[str, Any]] = {code: {} for code in candidate_codes}
    market_payload: dict[str, Any] = {"候选覆盖数": len(candidate_codes)}
    market_warnings: list[str] = []
    market_sources = 0
    financial_payload: dict[str, Any] = {"报告期": _latest_completed_financial_report(trading_date)}
    financial_warnings: list[str] = []

    try:
        hot = gateway.get_ths_hot_stock_list()
        hot_codes = {
            code
            for column in ("ticker", "code", "thscode")
            if column in hot.frame
            for code in (hot.frame[column].map(_canonical_evidence_code).dropna().tolist())
        }
        for code in candidate_codes:
            evidence[code]["hot_list"] = code in hot_codes
        market_payload.update(_dataset_source_payload(hot))
        market_payload["热榜命中数"] = sum(code in hot_codes for code in candidate_codes)
        market_sources += 1
    except Exception:
        market_warnings.append("同花顺热榜数据不可用，未参与候选排序")

    try:
        catalog = gateway.get_ths_index_catalog(ThsIndexTag.INDUSTRY)
        market_payload["行业指数目录"] = int(len(catalog.frame))
        market_payload["指数数据源"] = catalog.source
        market_payload["指数采集时间"] = catalog.observed_at.isoformat()
        market_sources += 1
        if "thscode" not in catalog.frame:
            raise ValueError("THS index catalog missing codes")
        index_codes = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in catalog.frame["thscode"]
                if isinstance(value, str) and value.strip()
            )
        )[:10]
        if not index_codes:
            raise ValueError("THS index catalog empty")
        index_snapshot = gateway.get_ths_index_snapshot(index_codes)
        index_support, positive, sample_count = _index_environment_support(index_snapshot)
        if sample_count == 0:
            raise ValueError("THS index snapshot missing change data")
        market_payload["行业指数样本数"] = sample_count
        market_payload["正向行业指数数"] = positive
        market_payload["指数环境"] = "偏强" if index_support else "中性或偏弱"
        market_payload["指数快照数据源"] = index_snapshot.source
        market_payload["指数快照采集时间"] = index_snapshot.observed_at.isoformat()
        for code in candidate_codes:
            evidence[code]["index_support"] = index_support
    except Exception:
        market_warnings.append("同花顺行业指数快照不可用，未参与候选排序")

    report = _latest_completed_financial_report(trading_date)
    available = 0
    positive = 0
    for code in candidate_codes:
        try:
            financials = gateway.get_ths_financial_indicators(code, report)
            available += 1
            if available == 1:
                financial_payload.update(
                    {
                        "数据源": financials.source,
                        "采集时间": financials.observed_at.isoformat(),
                        "来源时间": financials.source_timestamp.isoformat()
                        if financials.source_timestamp
                        else "unavailable",
                    }
                )
            is_positive = _positive_financial_indicator(financials)
            evidence[code]["financial_positive"] = is_positive
            positive += int(is_positive)
        except Exception:
            financial_warnings.append(f"{code}: 同花顺财务指标不可用，未参与候选排序")
    financial_payload.update({"覆盖数": available, "正向增长证据数": positive})

    market_status = "ok" if not market_warnings else ("partial" if market_sources else "unavailable")
    financial_status = "ok" if available == len(candidate_codes) else ("partial" if available else "unavailable")
    return (
        evidence,
        ModuleResult(
            "ths_market_evidence",
            market_status,
            observed_at,
            market_payload,
            tuple(dict.fromkeys(market_warnings)),
        ),
        ModuleResult(
            "ths_financial_evidence",
            financial_status,
            observed_at,
            financial_payload,
            tuple(dict.fromkeys(financial_warnings)) or (() if available else ("同花顺财务数据不可用，未参与候选排序",)),
        ),
    )


def _snapshot_fetch_is_current(
    dataset: MarketDataset,
    session: ReportSession,
    *,
    checked_at: datetime | None = None,
) -> bool:
    observed_at = dataset.observed_at
    checked = checked_at if checked_at is not None else session.now_shanghai
    if (
        observed_at.tzinfo is None
        or observed_at.utcoffset() is None
        or checked.tzinfo is None
        or checked.utcoffset() is None
        or session.now_shanghai.tzinfo is None
        or session.now_shanghai.utcoffset() is None
    ):
        return False
    observed_local = observed_at.astimezone(SHANGHAI_TIMEZONE)
    checked_local = checked.astimezone(SHANGHAI_TIMEZONE)
    session_start = session.now_shanghai.astimezone(SHANGHAI_TIMEZONE)
    return (
        observed_local <= checked_local
        and observed_local.date() == session.trading_date
        and checked_local.date() == session.trading_date
        and checked_local >= session_start
    )


def _snapshot_source_is_authoritative(
    dataset: MarketDataset,
    session: ReportSession,
    *,
    expected_session: date,
    checked_at: datetime | None = None,
) -> bool:
    source = dataset.source_timestamp
    observed_at = dataset.observed_at
    checked = checked_at if checked_at is not None else session.now_shanghai
    if (
        source is None
        or source.tzinfo is None
        or source.utcoffset() is None
        or observed_at.tzinfo is None
        or observed_at.utcoffset() is None
        or checked.tzinfo is None
        or checked.utcoffset() is None
        or session.now_shanghai.tzinfo is None
        or session.now_shanghai.utcoffset() is None
    ):
        return False
    local = source.astimezone(SHANGHAI_TIMEZONE)
    observed_local = observed_at.astimezone(SHANGHAI_TIMEZONE)
    checked_local = checked.astimezone(SHANGHAI_TIMEZONE)
    session_start = session.now_shanghai.astimezone(SHANGHAI_TIMEZONE)
    if local > observed_local or local > checked_local:
        return False
    if checked_local.date() != session.trading_date or checked_local < session_start:
        return False
    if local.date() != expected_session or local.time() < time(15, 0):
        return False
    maximum_age = (
        _POSTMARKET_MAX_SOURCE_AGE
        if session.mode is ReportMode.POSTMARKET
        else _PREMARKET_MAX_SOURCE_AGE
    )
    return checked_local - local <= maximum_age


def _portfolio_prices(snapshot: MarketDataset, positions: Sequence[Position]) -> dict[str, float]:
    frame = snapshot.frame
    if "code" not in frame or "price" not in frame:
        return {}
    prices: dict[str, float] = {}
    requested = {position.code for position in positions}
    for _, row in frame.iterrows():
        code = str(row["code"])
        if code not in requested or code in prices:
            continue
        try:
            price = float(row["price"])
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(price) and price > 0:
            prices[code] = price
    return prices


def _snapshot_history_conflicts(
    snapshot: MarketDataset,
    histories: Mapping[str, MarketDataset],
    *,
    expected_session: date,
) -> set[str]:
    if "code" not in snapshot.frame or "price" not in snapshot.frame:
        return set()
    conflicts: set[str] = set()
    for _, row in snapshot.frame.iterrows():
        code = str(row["code"])
        history = histories.get(code)
        bar = _last_session_bar(history, expected_session) if history is not None else None
        if bar is None:
            continue
        try:
            snapshot_price = float(row["price"])
            history_close = float(bar["close"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) and value > 0 for value in (snapshot_price, history_close)):
            continue
        relative_difference = abs(snapshot_price - history_close) / history_close
        if relative_difference > SNAPSHOT_DAILY_CLOSE_TOLERANCE:
            conflicts.add(code)
    return conflicts


def _untrusted_snapshot_codes(
    snapshot: MarketDataset,
    *,
    source_is_authoritative: bool,
    requested_codes: Sequence[str] = (),
) -> set[str]:
    if source_is_authoritative:
        return set()
    codes = {str(code) for code in snapshot.frame.get("code", ())}
    codes.update(str(code) for code in requested_codes)
    return codes


def _suppress_unactionable_candidates(
    candidates: Sequence[Candidate], conflict_codes: set[str], untrusted_codes: set[str]
) -> tuple[Candidate, ...]:
    projected: list[Candidate] = []
    for item in candidates:
        if item.code in conflict_codes:
            projected.append(
                replace(
                    item,
                    trigger="观望：价格来源冲突",
                    warning=_PRICE_CONFLICT_CANDIDATE_WARNING,
                )
            )
        elif item.code in untrusted_codes:
            projected.append(
                replace(
                    item,
                    trigger="观望：快照权威性不足",
                    warning=_UNTRUSTED_SNAPSHOT_CANDIDATE_WARNING,
                )
            )
        else:
            projected.append(item)
    return tuple(projected)


def _last_session_bar(dataset: MarketDataset, expected_session: date) -> pd.Series | None:
    frame = dataset.frame
    if not isinstance(frame, pd.DataFrame) or frame.empty or "date" not in frame:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce")
    matching = frame.loc[dates.dt.date == expected_session]
    if len(matching) != 1:
        return None
    return matching.iloc[0]


def classify_prior_candidates(
    candidates: Sequence[Mapping[str, Any]],
    histories: Mapping[str, MarketDataset],
    *,
    expected_session: date,
) -> tuple[Mapping[str, Any], ...]:
    """Classify prior candidates conservatively from one completed daily bar."""

    rows: list[Mapping[str, Any]] = []
    for item in candidates:
        code = str(item.get("code", ""))
        name = str(item.get("name", ""))
        try:
            trigger = float(item["trigger_price"])
            stop = float(item["stop_price"])
            if not all(math.isfinite(value) and value > 0 for value in (trigger, stop)):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            rows.append({"code": code, "name": name, "status": "失效"})
            continue
        current = histories.get(code)
        bar = _last_session_bar(current, expected_session) if current is not None else None
        if bar is None:
            status = "失效"
        else:
            try:
                high = float(bar["high"])
                low = float(bar["low"])
                close = float(bar["close"])
                valid = all(math.isfinite(value) and value > 0 for value in (high, low, close))
            except (KeyError, TypeError, ValueError, OverflowError):
                valid = False
            if not valid or low <= stop:
                status = "失效"
            elif high >= trigger:
                status = "触发"
            else:
                status = "继续观察"
        rows.append({"code": code, "name": name, "status": status})
    return tuple(rows)


def _load_prior_state(path: Path | None, session: ReportSession) -> tuple[Mapping[str, Any], ...]:
    if path is None:
        raise ValueError("prior state unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("prior state unavailable") from None
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("prior state unavailable")
    expected_date = session.trading_date.isoformat()
    if (
        payload.get("mode") != ReportMode.PREMARKET.value
        or payload.get("trading_date") != expected_date
        or payload.get("report_key") != f"{expected_date}-{ReportMode.PREMARKET.value}"
        or payload.get("final_state") != FinalState.SENT.value
        or payload.get("test_email") is not False
        or not isinstance(payload.get("candidate_state"), list)
    ):
        raise ValueError("prior state unavailable")
    generated_at_raw = payload.get("generated_at")
    if not isinstance(generated_at_raw, str):
        raise ValueError("prior state unavailable")
    try:
        generated_at = datetime.fromisoformat(generated_at_raw)
    except ValueError:
        raise ValueError("prior state unavailable") from None
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("prior state unavailable")
    if generated_at.astimezone(SHANGHAI_TIMEZONE).date() != session.trading_date:
        raise ValueError("prior state unavailable")
    if generated_at > session.now_shanghai:
        raise ValueError("prior state unavailable")
    rows: list[Mapping[str, Any]] = []
    for item in payload["candidate_state"]:
        if not isinstance(item, dict):
            raise ValueError("prior state unavailable")
        code = item.get("code")
        if not isinstance(code, str) or not code.isascii() or not re.fullmatch(r"\d{6}", code):
            raise ValueError("prior state unavailable")
        rows.append(item)
    return tuple(rows)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_file_lock(path: Path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_write(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_report_artifacts(
    output_dir: Path | str,
    *,
    report_key: str,
    rendered: RenderedReport,
    manifest: Mapping[str, Any],
    test_email: bool = False,
) -> ArtifactPaths:
    """Publish an immutable attempt and atomically point at the coherent set."""

    key = _canonical_report_key(report_key)
    root = Path(output_dir)
    channel = "test" if test_email else "production"
    parent = _contained_path(root, channel)
    report_root = _contained_path(root, channel, key)
    attempts = _contained_path(root, channel, key, "attempts")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.chmod(0o700)
    report_root.mkdir(mode=0o700, exist_ok=True)
    report_root.chmod(0o700)
    attempts.mkdir(mode=0o700, exist_ok=True)
    attempts.chmod(0o700)
    _fsync_directory(parent)
    lock_path = _contained_path(root, channel, key, ".publish.lock")
    with _exclusive_file_lock(lock_path):
        return _write_report_artifact_attempt(
            root,
            channel=channel,
            report_key=key,
            report_root=report_root,
            attempts=attempts,
            rendered=rendered,
            manifest=manifest,
        )


def _write_report_artifact_attempt(
    root: Path,
    *,
    channel: str,
    report_key: str,
    report_root: Path,
    attempts: Path,
    rendered: RenderedReport,
    manifest: Mapping[str, Any],
) -> ArtifactPaths:
    for child in report_root.iterdir():
        if (
            child.is_dir()
            and not child.is_symlink()
            and re.fullmatch(r"\.staging-[0-9a-f]{32}", child.name)
        ):
            shutil.rmtree(child)
    attempt_id = uuid.uuid4().hex
    staging = _contained_path(root, channel, report_key, f".staging-{attempt_id}")
    attempt = _contained_path(root, channel, report_key, "attempts", attempt_id)
    index_path = _contained_path(root, channel, report_key, "current.json")
    try:
        staging.mkdir(mode=0o700)
        _atomic_write(staging / "report.html", rendered.html)
        _atomic_write(staging / "report.txt", rendered.text)
        _atomic_write(
            staging / "manifest.json",
            json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        _fsync_directory(staging)
        staging.replace(attempt)
        _fsync_directory(report_root)
        _fsync_directory(attempts)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ArtifactPaths(
        attempt / "report.html",
        attempt / "report.txt",
        attempt / "manifest.json",
        index_path,
        attempt_id,
    )


def _artifact_paths_for_attempt(
    output_dir: Path | str,
    *,
    report_key: str,
    attempt_id: str,
    test_email: bool,
) -> ArtifactPaths:
    key = _canonical_report_key(report_key)
    if re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise ValueError("invalid artifact attempt")
    root = Path(output_dir)
    channel = "test" if test_email else "production"
    attempt = _contained_path(root, channel, key, "attempts", attempt_id)
    paths = ArtifactPaths(
        attempt / "report.html",
        attempt / "report.txt",
        attempt / "manifest.json",
        _contained_path(root, channel, key, "current.json"),
        attempt_id,
    )
    if not all(path.is_file() for path in (paths.html_path, paths.text_path, paths.manifest_path)):
        raise ValueError("artifact attempt unavailable")
    return paths


def publish_report_artifacts(
    output_dir: Path | str,
    *,
    report_key: str,
    paths: ArtifactPaths,
    test_email: bool = False,
) -> None:
    """Atomically publish a validated immutable attempt as the canonical pointer."""

    if paths.attempt_id is None:
        raise ValueError("artifact attempt unavailable")
    expected = _artifact_paths_for_attempt(
        output_dir,
        report_key=report_key,
        attempt_id=paths.attempt_id,
        test_email=test_email,
    )
    if (paths.html_path, paths.text_path, paths.manifest_path) != (
        expected.html_path,
        expected.text_path,
        expected.manifest_path,
    ):
        raise ValueError("artifact path mismatch")
    try:
        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("artifact manifest unavailable") from None
    final_state = manifest.get("final_state") if isinstance(manifest, dict) else None
    if manifest.get("report_key") != report_key or final_state not in {
        FinalState.SENT.value,
        FinalState.TEST_SENT.value,
        FinalState.PREVIEWED.value,
        "prepared",
    }:
        raise ValueError("artifact manifest identity invalid")
    pointer = {
        "attempt": paths.attempt_id,
        "final_state": final_state,
        "html": f"attempts/{paths.attempt_id}/report.html",
        "manifest": f"attempts/{paths.attempt_id}/manifest.json",
        "text": f"attempts/{paths.attempt_id}/report.txt",
    }
    _atomic_write(
        expected.index_path,
        json.dumps(pointer, sort_keys=True) + "\n",
        mode=0o600,
    )


def finalize_prepared_artifact(
    output_dir: Path | str,
    *,
    report_key: str,
    attempt_id: str,
    final_state: FinalState,
    test_email: bool = False,
) -> ArtifactPaths:
    """Create a new immutable finalized attempt from a prepared attempt."""

    prepared = _artifact_paths_for_attempt(
        output_dir,
        report_key=report_key,
        attempt_id=attempt_id,
        test_email=test_email,
    )
    try:
        manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("artifact manifest unavailable") from None
    if not isinstance(manifest, dict) or manifest.get("report_key") != report_key:
        raise ValueError("artifact manifest identity invalid")
    manifest["final_state"] = final_state.value
    return write_report_artifacts(
        output_dir,
        report_key=report_key,
        rendered=RenderedReport(
            subject="reconciled report",
            html=prepared.html_path.read_text(encoding="utf-8"),
            text=prepared.text_path.read_text(encoding="utf-8"),
        ),
        manifest=manifest,
        test_email=test_email,
    )


def _warning_codes(modules: Mapping[str, ModuleResult]) -> list[str]:
    codes: list[str] = []
    for name, result in modules.items():
        for index, warning in enumerate(result.warnings, start=1):
            codes.append(
                warning
                if warning == _PRICE_CONFLICT_WARNING_CODE or warning in _SAFE_DATA_FAILURE_CODES
                else f"{name}_warning_{index}"
            )
    return codes


def _candidate_state(candidates: Sequence[Candidate], portfolio_codes: set[str]) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        if (
            item.code in portfolio_codes
            or item.code in seen
            or item.warning in {
                _PRICE_CONFLICT_CANDIDATE_WARNING,
                _UNTRUSTED_SNAPSHOT_CANDIDATE_WARNING,
            }
        ):
            continue
        seen.add(item.code)
        state.append(
            {
                "code": item.code,
                "name": item.name,
                "trigger_price": item.close,
                "stop_price": item.stop_price,
            }
        )
    return state


def _redacted_manifest(
    session: ReportSession,
    modules: Mapping[str, ModuleResult],
    rendered: RenderedReport,
    *,
    final_state: str,
    candidates: Sequence[Candidate],
    morning_candidates: Sequence[Mapping[str, Any]],
    portfolio_codes: set[str],
    test_email: bool,
    market_source_timestamp: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = generated_at if generated_at is not None else session.now_shanghai
    source_timestamps = {
        name: result.observed_at.isoformat() for name, result in modules.items()
    }
    if "market" in modules:
        source_timestamps["market"] = market_source_timestamp
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "report_key": session.report_key,
        "mode": session.mode.value,
        "trading_date": session.trading_date.isoformat(),
        "generated_at": generated.isoformat(),
        "final_state": final_state,
        "test_email": test_email,
        "module_statuses": {name: result.status for name, result in modules.items()},
        "source_timestamps": source_timestamps,
        "warning_codes": _warning_codes(modules),
    }
    if session.mode is ReportMode.PREMARKET:
        manifest["candidate_state"] = _candidate_state(candidates, portfolio_codes)
    else:
        manifest["morning_candidate_statuses"] = [
            {"code": row.get("code"), "status": row.get("status")}
            for row in morning_candidates
            if row.get("code") not in portfolio_codes
        ]
    serialized = json.dumps(manifest, ensure_ascii=False)
    if any(code in serialized for code in portfolio_codes):
        raise RuntimeError("redaction validation failed")
    return manifest


def _failure(error_code: str, *, report_key: str = "unavailable", modules=None, paths=None) -> RunResult:
    return RunResult(
        exit_code=EXIT_FAILURE,
        final_state=FinalState.HARD_FAILURE,
        report_key=report_key,
        modules=modules or {},
        html_path=paths.html_path if paths else None,
        text_path=paths.text_path if paths else None,
        manifest_path=paths.manifest_path if paths else None,
        error_code=error_code,
    )


def _operator_required(
    error_code: str,
    *,
    report_key: str,
    modules=None,
    paths=None,
) -> RunResult:
    return RunResult(
        exit_code=EXIT_FAILURE,
        final_state=FinalState.OPERATOR_ACTION_REQUIRED,
        report_key=report_key,
        modules=modules or {},
        html_path=paths.html_path if paths else None,
        text_path=paths.text_path if paths else None,
        manifest_path=paths.manifest_path if paths else None,
        error_code=error_code,
    )


def _session_identity_is_valid(session: ReportSession, mode: ReportMode) -> bool:
    timestamp = session.now_shanghai
    return (
        session.mode is mode
        and timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        and timestamp.date() == session.trading_date
        and session.report_key == f"{session.trading_date.isoformat()}-{mode.value}"
    )


def _clock_checkpoint(
    session: ReportSession,
    checked_at: datetime,
    *,
    previous: datetime,
) -> datetime | None:
    """Return a monotonic in-session Shanghai checkpoint, or fail closed."""

    if (
        checked_at.tzinfo is None
        or checked_at.utcoffset() is None
        or previous.tzinfo is None
        or previous.utcoffset() is None
    ):
        return None
    current = checked_at.astimezone(SHANGHAI_TIMEZONE)
    prior = previous.astimezone(SHANGHAI_TIMEZONE)
    if current.date() != session.trading_date or current < prior:
        return None
    return current


def run_report(
    mode: ReportMode,
    *,
    deps: RunnerDependencies | None = None,
    force: bool = False,
    test_email: bool = False,
    preview_only: bool = False,
    already_sent: bool = False,
    prior_report: Path | str | None = None,
    output_dir: Path | str = "reports/collaborative",
) -> RunResult:
    """Run one report without leaking third-party exception text to its result."""

    if preview_only and test_email:
        return _failure("preview_test_email_conflict")
    normalized_mode = ReportMode(mode)
    active = deps or default_dependencies()
    try:
        now = active.clock()
    except Exception:
        return _failure("report_clock_invalid")
    try:
        session = active.session_builder(normalized_mode, now, scheduled=not force)
    except Exception as exc:
        code = "outside_delivery_window" if str(exc) == "outside delivery window" else "calendar_unavailable"
        return _failure(code)
    if not _session_identity_is_valid(session, normalized_mode):
        return _failure("report_identity_invalid")
    checkpoint = _clock_checkpoint(session, now, previous=session.now_shanghai)
    if checkpoint is None:
        return _failure("report_clock_invalid", report_key=session.report_key)
    if not session.is_trading_day:
        return RunResult(EXIT_SUCCESS, FinalState.NON_TRADING_DAY_SKIP, session.report_key, {})
    ledger = None
    if not test_email and not preview_only:
        if already_sent:
            return RunResult(EXIT_SUCCESS, FinalState.DUPLICATE_SKIP, session.report_key, {})
        try:
            ledger = active.ledger_factory(output_dir)
            delivery_record = ledger.record(session.report_key)
        except Exception:
            return _failure("delivery_state_unavailable", report_key=session.report_key)
        if delivery_record is not None and delivery_record.state is DeliveryState.SENT:
            try:
                recovered_paths = _artifact_paths_for_attempt(
                    output_dir,
                    report_key=session.report_key,
                    attempt_id=delivery_record.attempt_id,
                    test_email=False,
                )
                (active.artifact_publisher or publish_report_artifacts)(
                    output_dir,
                    report_key=session.report_key,
                    paths=recovered_paths,
                    test_email=False,
                )
            except Exception:
                return _operator_required(
                    "sent_artifact_repair_required",
                    report_key=session.report_key,
                )
            return RunResult(
                EXIT_SUCCESS,
                FinalState.DUPLICATE_SKIP,
                session.report_key,
                {},
                recovered_paths.html_path,
                recovered_paths.text_path,
                recovered_paths.manifest_path,
            )
        if delivery_record is not None and delivery_record.state in {
            DeliveryState.CLAIMED,
            DeliveryState.SENDING,
        }:
            return RunResult(EXIT_SUCCESS, FinalState.DUPLICATE_SKIP, session.report_key, {})
        if delivery_record is not None and delivery_record.state is DeliveryState.IN_DOUBT:
            return _operator_required(
                "delivery_reconciliation_required",
                report_key=session.report_key,
            )

    try:
        settings = active.settings_loader()
    except Exception:
        return _failure("configuration_invalid", report_key=session.report_key)

    artifact_writer = active.artifact_writer or write_report_artifacts
    artifact_finalizer = active.artifact_finalizer or write_report_artifacts
    artifact_publisher = active.artifact_publisher or publish_report_artifacts
    try:
        expected_session = active.data_session_resolver(
            normalized_mode,
            session.trading_date,
            session.now_shanghai,
        )
    except Exception as exc:
        code = "report_data_incomplete" if str(exc) == "report data session incomplete" else "calendar_unavailable"
        return _failure(code, report_key=session.report_key)

    modules: dict[str, ModuleResult] = {}
    prior_candidates: tuple[Mapping[str, Any], ...] = ()
    if normalized_mode is ReportMode.POSTMARKET:
        try:
            prior_candidates = _load_prior_state(
                Path(prior_report) if prior_report is not None else None,
                session,
            )
        except Exception:
            modules["morning_candidates"] = _unavailable(
                "morning_candidates", session.now_shanghai, "早盘候选状态不可用"
            )

    try:
        global_data = active.gateway.get_global_snapshot()
        modules["global"] = _module(
            "global", global_data.observed_at, _global_payload(global_data), *global_data.warnings
        )
    except Exception:
        modules["global"] = _unavailable("global", session.now_shanghai, "全球市场数据暂不可用")

    snapshot: MarketDataset | None
    snapshot_source_is_authoritative = False
    market_source_timestamp = "unavailable"
    try:
        snapshot = active.gateway.get_a_share_snapshot()
    except Exception as exc:
        snapshot = None
        modules["market"] = _data_unavailable(
            "market", session.now_shanghai, "数据不足，建议观望", exc, "snapshot_data_failed",
        )
    else:
        try:
            checkpoint = _clock_checkpoint(session, active.clock(), previous=checkpoint)
        except Exception:
            return _failure("report_clock_invalid", report_key=session.report_key, modules=modules)
        if checkpoint is None:
            return _failure("report_clock_invalid", report_key=session.report_key, modules=modules)
        try:
            if not _snapshot_fetch_is_current(snapshot, session, checked_at=checkpoint):
                raise ValueError("stale snapshot")
            snapshot_source_is_authoritative = _snapshot_source_is_authoritative(
                snapshot,
                session,
                expected_session=expected_session,
                checked_at=checkpoint,
            )
            if snapshot_source_is_authoritative and snapshot.source_timestamp is not None:
                market_source_timestamp = snapshot.source_timestamp.isoformat()
            market_warnings = snapshot.warnings
            if not snapshot_source_is_authoritative:
                market_warnings = (*market_warnings, _SNAPSHOT_AUTHORITY_WARNING_CODE)
                if snapshot.source_timestamp is None:
                    market_warnings = (*market_warnings, _SNAPSHOT_UNAVAILABLE_OBSERVATION_WARNING)
            modules["market"] = ModuleResult(
                "market",
                "ok" if snapshot_source_is_authoritative and not snapshot.warnings else "partial",
                snapshot.source_timestamp or snapshot.observed_at,
                _market_payload(snapshot, authoritative=snapshot_source_is_authoritative),
                tuple(dict.fromkeys(market_warnings)),
            )
        except Exception as exc:
            snapshot = None
            modules["market"] = _data_unavailable(
                "market", session.now_shanghai, "数据不足，建议观望", exc, "snapshot_data_failed",
            )

    portfolio_codes = {position.code for position in settings.positions}
    prices = _portfolio_prices(snapshot, settings.positions) if snapshot is not None else {}

    prefiltered_codes: list[str] = []
    if snapshot is not None:
        try:
            prefiltered = prefilter_universe(snapshot, settings.screen_prefilter)
            prefiltered_codes = [str(code) for code in prefiltered.get("code", ())]
        except Exception:
            prefiltered_codes = []
    prior_codes = tuple(str(item["code"]) for item in prior_candidates)
    requested_codes = tuple(dict.fromkeys((*prefiltered_codes, *portfolio_codes, *prior_codes)))
    histories: dict[str, MarketDataset] = {}
    history_failures = 0
    for code in requested_codes:
        try:
            histories[code] = active.gateway.get_daily_bars(code, expected_session)
        except Exception:
            history_failures += 1

    conflict_codes = (
        _snapshot_history_conflicts(snapshot, histories, expected_session=expected_session)
        if snapshot is not None
        else set()
    )
    untrusted_codes = (
        _untrusted_snapshot_codes(
            snapshot,
            source_is_authoritative=snapshot_source_is_authoritative,
            requested_codes=requested_codes,
        )
        if snapshot is not None
        else set()
    )
    suppressed_codes = conflict_codes | untrusted_codes
    if conflict_codes:
        market = modules["market"]
        modules["market"] = ModuleResult(
            market.name,
            "partial",
            market.observed_at,
            market.payload,
            (*market.warnings, _PRICE_CONFLICT_WARNING_CODE),
        )
        prices = {code: price for code, price in prices.items() if code not in conflict_codes}
    if untrusted_codes:
        market = modules["market"]
        modules["market"] = ModuleResult(
            market.name,
            "partial",
            market.observed_at,
            market.payload,
            tuple(dict.fromkeys((*market.warnings, _SNAPSHOT_AUTHORITY_WARNING_CODE))),
        )
        prices = {code: price for code, price in prices.items() if code not in untrusted_codes}

    leading: dict[str, str] = {}
    try:
        sectors = active.gateway.get_leading_sector_codes()
        if {"code", "sector"}.issubset(sectors.frame.columns):
            leading = {
                str(row["code"]): str(row["sector"])
                for _, row in sectors.frame.iterrows()
            }
    except Exception:
        pass

    screening = ScreeningResult((), ())
    if snapshot is None or not histories:
        modules["screening"] = _unavailable("screening", session.now_shanghai, "数据不足，建议观望")
    else:
        try:
            screening = active.screener(
                snapshot,
                histories,
                leading,
                short_limit=settings.short_limit,
                swing_limit=settings.swing_limit,
                prefilter_limit=settings.screen_prefilter,
                observed_at=session.now_shanghai,
            )
            screening = ScreeningResult(
                _suppress_unactionable_candidates(
                    screening.short_term, conflict_codes, untrusted_codes
                ),
                _suppress_unactionable_candidates(
                    screening.swing, conflict_codes, untrusted_codes
                ),
                screening.warnings,
            )
            incomplete_fields = "snapshot_screening_fields_incomplete" in snapshot.warnings
            status = "partial" if history_failures or screening.warnings or incomplete_fields else "ok"
            modules["screening"] = ModuleResult(
                "screening",
                status,
                session.now_shanghai,
                {"短线候选数": len(screening.short_term), "波段候选数": len(screening.swing)},
                screening.warnings,
            )
        except Exception:
            screening = ScreeningResult((), ())
            modules["screening"] = _unavailable(
                "screening", session.now_shanghai, "筛选暂不可用，仅分析持仓"
            )

    evidence_codes = tuple(dict.fromkeys(item.code for item in (*screening.short_term, *screening.swing)))
    if evidence_codes:
        evidence, market_evidence, financial_evidence = _ths_evidence(
            active.gateway,
            evidence_codes,
            trading_date=session.trading_date,
            observed_at=session.now_shanghai,
        )
        modules["ths_market_evidence"] = market_evidence
        modules["ths_financial_evidence"] = financial_evidence
        screening = ScreeningResult(
            rank_with_ths_evidence(screening.short_term, evidence),
            rank_with_ths_evidence(screening.swing, evidence),
            screening.warnings,
        )
    else:
        modules["ths_market_evidence"] = _unavailable(
            "ths_market_evidence", session.now_shanghai, "无技术候选，同花顺市场证据未参与排序"
        )
        modules["ths_financial_evidence"] = _unavailable(
            "ths_financial_evidence", session.now_shanghai, "无技术候选，同花顺财务证据未参与排序"
        )

    backtest_codes = tuple(
        dict.fromkeys(candidate.code for candidate in (*screening.short_term, *screening.swing))
    )
    if not backtest_codes and modules["screening"].status == "unavailable":
        backtest_codes = tuple(portfolio_codes)
    backtest_payload: dict[str, Any] = {}
    backtest_warnings: list[str] = []
    for code in backtest_codes:
        history = histories.get(code)
        if history is None:
            backtest_warnings.append("策略历史数据不足")
            continue
        try:
            backtest_payload[code] = {
                "short": _as_payload(active.short_backtest(history.frame, capital=settings.capital_cny)),
                "swing": _as_payload(active.swing_backtest(history.frame, capital=settings.capital_cny)),
            }
        except Exception:
            backtest_warnings.append("策略回测暂不可用")
    modules["backtests"] = ModuleResult(
        "backtests",
        "ok" if backtest_payload and not backtest_warnings else ("partial" if backtest_payload else "unavailable"),
        session.now_shanghai,
        backtest_payload,
        tuple(dict.fromkeys(backtest_warnings)) or (() if backtest_payload else ("策略回测暂不可用",)),
    )

    gold_failure_stage = "gold_data_failed"
    try:
        gold_data = active.gateway.get_gold_bars()
        gold_failure_stage = "gold_analysis_failed"
        gold = active.gold_analyzer(gold_data.frame, capital=settings.capital_cny)
        modules["gold"] = _module("gold", gold_data.observed_at, _as_payload(gold), *gold_data.warnings)
    except Exception as exc:
        modules["gold"] = _data_unavailable(
            "gold", session.now_shanghai, "黄金模块暂不可用", exc, gold_failure_stage,
        )

    ai_codes = tuple(
        code
        for code in dict.fromkeys(
            (*portfolio_codes, *(item.code for item in screening.short_term), *(item.code for item in screening.swing))
        )
        if code not in suppressed_codes
        and not next(
            (
                item.warning.strip()
                for item in (*screening.short_term, *screening.swing)
                if item.code == code
            ),
            "",
        )
    )
    try:
        modules["ai"] = active.ai_enricher(ai_codes, observed_at=session.now_shanghai)
    except Exception:
        modules["ai"] = _unavailable("ai", session.now_shanghai, "AI分析暂不可用")

    portfolio_payload: dict[str, Any] = {}
    portfolio_warnings: list[str] = []
    for position in settings.positions:
        price = prices.get(position.code)
        if price is None:
            portfolio_warnings.append("持仓价格不可用")
            continue
        try:
            portfolio_payload[position.code] = _as_payload(
                active.risk_evaluator(position, price, settings.capital_cny)
            )
        except Exception:
            portfolio_warnings.append("持仓风险计算不可用")
    if not settings.positions:
        modules["portfolio"] = ModuleResult(
            "portfolio",
            "ok",
            session.now_shanghai,
            {"status": "当前无持仓"},
            ("当前无持仓",),
        )
    else:
        modules["portfolio"] = ModuleResult(
            "portfolio",
            (
                "ok"
                if len(portfolio_payload) == len(settings.positions)
                else ("partial" if portfolio_payload else "unavailable")
            ),
            session.now_shanghai,
            portfolio_payload,
            tuple(dict.fromkeys(portfolio_warnings)) or (() if portfolio_payload else ("数据不足，建议观望",)),
        )

    portfolio_valuations_complete = all(position.code in prices for position in settings.positions)
    available_cash: float | None = None
    if portfolio_valuations_complete:
        current_market_value = sum(prices[position.code] * position.quantity for position in settings.positions)
        available_cash = max(settings.capital_cny - current_market_value, 0.0)
    sizing_payload: dict[str, int] = {}
    sizing_warnings: list[str] = []
    if not portfolio_valuations_complete:
        sizing_warnings.append("持仓估值不可用，未提供仓位建议")
    for item in (*screening.short_term, *screening.swing):
        if not portfolio_valuations_complete:
            continue
        if item.warning.strip():
            sizing_warnings.append("候选不可操作，未提供仓位建议")
            continue
        try:
            sizing_payload[item.code] = active.sizing_evaluator(
                item.close,
                item.stop_price,
                capital=settings.capital_cny,
                available_cash=available_cash,
                risk_fraction=settings.risk_fraction,
            )
        except Exception:
            sizing_warnings.append("候选仓位计算不可用")
    modules["sizing"] = ModuleResult(
        "sizing",
        "ok" if sizing_payload and not sizing_warnings else ("partial" if sizing_payload else "unavailable"),
        session.now_shanghai,
        sizing_payload,
        tuple(dict.fromkeys(sizing_warnings)) or (() if sizing_payload else ("暂无可计算候选",)),
    )

    morning_candidates: tuple[Mapping[str, Any], ...] = ()
    if normalized_mode is ReportMode.POSTMARKET and prior_candidates:
        morning_candidates = classify_prior_candidates(
            prior_candidates,
            histories,
            expected_session=expected_session,
        )
        modules["morning_candidates"] = _module(
            "morning_candidates", session.now_shanghai, {"候选数": len(morning_candidates)}
        )

    try:
        generated_at = _clock_checkpoint(session, active.clock(), previous=checkpoint)
    except Exception:
        return _failure("report_clock_invalid", report_key=session.report_key, modules=modules)
    if generated_at is None:
        return _failure("report_clock_invalid", report_key=session.report_key, modules=modules)
    checkpoint = generated_at
    if snapshot_source_is_authoritative and not _snapshot_source_is_authoritative(
        snapshot,
        session,
        expected_session=expected_session,
        checked_at=generated_at,
    ):
        return _failure("snapshot_source_expired", report_key=session.report_key, modules=modules)
    try:
        rendered = active.renderer(
            normalized_mode,
            session.trading_date,
            modules=modules,
            short_term_candidates=screening.short_term,
            swing_candidates=screening.swing,
            morning_candidates=morning_candidates,
            subject_prefix=_TEST_SUBJECT_PREFIX if test_email else None,
            generated_at=generated_at,
        )
    except Exception:
        return _failure("report_render_failed", report_key=session.report_key, modules=modules)
    try:
        checkpoint = _clock_checkpoint(session, active.clock(), previous=checkpoint)
    except Exception:
        return _failure("report_clock_invalid", report_key=session.report_key, modules=modules)
    if checkpoint is None:
        return _failure("report_clock_invalid", report_key=session.report_key, modules=modules)

    final_state = FinalState.PREVIEWED if preview_only else (FinalState.TEST_SENT if test_email else FinalState.SENT)
    try:
        manifest = _redacted_manifest(
            session,
            modules,
            rendered,
            final_state="prepared",
            candidates=(*screening.short_term, *screening.swing),
            morning_candidates=morning_candidates,
            portfolio_codes=portfolio_codes,
            test_email=test_email,
            market_source_timestamp=market_source_timestamp,
            generated_at=generated_at,
        )
        paths = artifact_writer(
            output_dir,
            report_key=session.report_key,
            rendered=rendered,
            manifest=manifest,
            test_email=test_email or preview_only,
        )
    except Exception:
        return _failure("artifact_write_failed", report_key=session.report_key, modules=modules)
    try:
        checkpoint = _clock_checkpoint(session, active.clock(), previous=checkpoint)
    except Exception:
        return _failure("report_clock_invalid", report_key=session.report_key, modules=modules, paths=paths)
    if checkpoint is None:
        return _failure("report_clock_invalid", report_key=session.report_key, modules=modules, paths=paths)
    if snapshot_source_is_authoritative and not _snapshot_source_is_authoritative(
        snapshot,
        session,
        expected_session=expected_session,
        checked_at=checkpoint,
    ):
        return _failure("snapshot_source_expired", report_key=session.report_key, modules=modules, paths=paths)

    if preview_only:
        manifest = dict(
            manifest,
            final_state=final_state.value,
            module_statuses={name: result.status for name, result in modules.items()},
            warning_codes=_warning_codes(modules),
        )
        try:
            final_paths = artifact_finalizer(
                output_dir,
                report_key=session.report_key,
                rendered=rendered,
                manifest=manifest,
                test_email=True,
            )
        except Exception:
            return _failure("artifact_finalize_failed", report_key=session.report_key, modules=modules, paths=paths)
        return RunResult(
            EXIT_SUCCESS, final_state, session.report_key, modules,
            final_paths.html_path, final_paths.text_path, final_paths.manifest_path,
            screening.short_term, screening.swing, morning_candidates,
        )

    claim_id: str | None = None
    if not test_email:
        try:
            claim_id = ledger.claim(
                session.report_key,
                checkpoint,
                attempt_id=paths.attempt_id,
            )
            if claim_id is None:
                current = ledger.record(session.report_key)
                if current is not None and current.state is DeliveryState.SENT:
                    return RunResult(EXIT_SUCCESS, FinalState.DUPLICATE_SKIP, session.report_key, {})
                return _operator_required(
                    "delivery_reconciliation_required",
                    report_key=session.report_key,
                )
            ledger.begin_sending(session.report_key, claim_id, checkpoint)
        except Exception:
            return _failure(
                "delivery_state_unavailable",
                report_key=session.report_key,
                modules=modules,
                paths=paths,
            )

    try:
        delivery_result = active.mail_sender(rendered, test_email=test_email)
        if delivery_result is False:
            raise RuntimeError("delivery result unavailable")
    except DeliveryNotAcceptedError as exc:
        if claim_id is not None:
            try:
                ledger.mark_failed(session.report_key, claim_id, session.now_shanghai)
            except Exception:
                pass
        error_code = "configuration_invalid" if isinstance(exc, _DeliveryConfigurationError) else "delivery_failed"
        return _failure(error_code, report_key=session.report_key, modules=modules, paths=paths)
    except DeliveryInDoubtError:
        if claim_id is not None:
            try:
                ledger.mark_in_doubt(
                    session.report_key,
                    claim_id,
                    session.now_shanghai,
                    attempt_id=paths.attempt_id,
                )
            except Exception:
                pass
        return _operator_required(
            "delivery_reconciliation_required",
            report_key=session.report_key,
            modules=modules,
            paths=paths,
        )
    except Exception:
        if claim_id is not None:
            try:
                ledger.mark_in_doubt(
                    session.report_key,
                    claim_id,
                    session.now_shanghai,
                    attempt_id=paths.attempt_id,
                )
            except Exception:
                pass
        return _operator_required(
            "delivery_reconciliation_required",
            report_key=session.report_key,
            modules=modules,
            paths=paths,
        )

    manifest = dict(
        manifest,
        final_state=final_state.value,
        module_statuses={name: result.status for name, result in modules.items()},
        warning_codes=_warning_codes(modules),
    )
    try:
        final_paths = artifact_finalizer(
            output_dir,
            report_key=session.report_key,
            rendered=rendered,
            manifest=manifest,
            test_email=test_email,
        )
    except Exception:
        if claim_id is not None:
            try:
                ledger.mark_in_doubt(
                    session.report_key,
                    claim_id,
                    session.now_shanghai,
                    attempt_id=paths.attempt_id,
                )
            except Exception:
                pass
            return _operator_required(
                "delivery_reconciliation_required",
                report_key=session.report_key,
                modules=modules,
                paths=paths,
            )
        modules["delivery_state"] = ModuleResult(
            "delivery_state", "partial", session.now_shanghai, {}, ("manifest_finalize_failed",)
        )
        final_paths = paths

    if claim_id is not None:
        try:
            ledger.mark_sent(
                session.report_key,
                claim_id,
                session.now_shanghai,
                attempt_id=final_paths.attempt_id,
            )
        except Exception:
            try:
                ledger.mark_in_doubt(
                    session.report_key,
                    claim_id,
                    session.now_shanghai,
                    attempt_id=final_paths.attempt_id,
                )
            except Exception:
                pass
            return _operator_required(
                "delivery_reconciliation_required",
                report_key=session.report_key,
                modules=modules,
                paths=final_paths,
            )

    try:
        artifact_publisher(
            output_dir,
            report_key=session.report_key,
            paths=final_paths,
            test_email=test_email,
        )
    except Exception:
        modules["delivery_state"] = ModuleResult(
            "delivery_state", "partial", session.now_shanghai, {}, ("artifact_pointer_update_failed",)
        )
    paths = final_paths
    return RunResult(
        EXIT_SUCCESS,
        final_state,
        session.report_key,
        modules,
        paths.html_path,
        paths.text_path,
        paths.manifest_path,
        screening.short_term,
        screening.swing,
        morning_candidates,
    )
