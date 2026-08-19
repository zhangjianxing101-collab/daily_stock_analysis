"""Total orchestration, privacy boundaries, and artifacts for daily reports."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .ai_bridge import enrich_codes
from .backtest import backtest_breakout, backtest_swing
from .gold import analyze_gold
from .mailer import send_with_retry
from .market_data import MarketDataGateway, MarketDataset
from .models import Candidate, ModuleResult, Position, ReportMode
from .report import RenderedReport, render_report
from .risk import evaluate_position, suggested_board_lots
from .screener import ScreeningResult, prefilter_universe, screen_aggressive
from .session import SHANGHAI_TIMEZONE, ReportSession, build_report_session, report_data_session
from .settings import CollaborativeSettings


# Process status is intentionally binary for GitHub Actions. Detailed outcomes live
# in FinalState so successful skips do not fail a scheduled workflow.
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
MANIFEST_SCHEMA_VERSION = 1
SNAPSHOT_DAILY_CLOSE_TOLERANCE = 0.01
_TEST_SUBJECT_PREFIX = "测试"
_PRICE_CONFLICT_WARNING_CODE = "snapshot_daily_close_conflict"
_PRICE_CONFLICT_CANDIDATE_WARNING = "价格来源冲突，仅供观望"


class FinalState(str, Enum):
    SENT = "sent"
    TEST_SENT = "test_sent"
    DUPLICATE_SKIP = "duplicate_skip"
    NON_TRADING_DAY_SKIP = "non_trading_day_skip"
    HARD_FAILURE = "hard_failure"


class _DeliveryConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactPaths:
    html_path: Path
    text_path: Path
    manifest_path: Path


class LocalDeliveryLedger:
    """Atomic local production-delivery markers keyed by report identity."""

    def __init__(self, output_dir: Path | str) -> None:
        self.directory = Path(output_dir) / ".delivery-ledger"

    def _path(self, report_key: str) -> Path:
        return self.directory / f"{report_key}.json"

    def is_sent(self, report_key: str) -> bool:
        try:
            payload = json.loads(self._path(report_key).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return payload == {"report_key": report_key, "production_sent": True}

    def mark_sent(self, report_key: str, sent_at: datetime) -> None:
        del sent_at
        _atomic_write(
            self._path(report_key),
            json.dumps({"report_key": report_key, "production_sent": True}, sort_keys=True) + "\n",
            mode=0o600,
        )


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

        return {
            "report_key": self.report_key,
            "module_statuses": {name: result.status for name, result in self.modules.items()},
            "artifact_path": str(self.manifest_path) if self.manifest_path else None,
            "final_state": self.final_state.value,
        }


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
    manifest_writer: Callable[[Path, str], None] | None = None


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
    active_clock = clock or (lambda: datetime.now().astimezone())
    return RunnerDependencies(
        settings_loader=CollaborativeSettings.from_env,
        session_builder=build_report_session,
        clock=active_clock,
        gateway=MarketDataGateway(clock=active_clock),
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


def _market_payload(dataset: MarketDataset) -> Mapping[str, Any]:
    frame = dataset.frame
    changes = pd.to_numeric(frame.get("change_pct", pd.Series(dtype=float)), errors="coerce")
    return {
        "股票数量": int(len(frame)),
        "上涨家数": int((changes > 0).sum()),
        "下跌家数": int((changes < 0).sum()),
    }


def _snapshot_is_fresh(dataset: MarketDataset, session: ReportSession) -> bool:
    observed_at = dataset.observed_at
    return (
        observed_at <= session.now_shanghai
        and observed_at.astimezone(session.now_shanghai.tzinfo).date() == session.trading_date
    )


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
    histories: Mapping[str, MarketDataset],
    *,
    expected_session: date,
    session: ReportSession,
    conflict_codes: set[str],
) -> set[str]:
    source_timestamp = snapshot.source_timestamp
    source_is_current = bool(
        source_timestamp is not None
        and source_timestamp <= session.now_shanghai
        and source_timestamp.astimezone(SHANGHAI_TIMEZONE).date() == session.trading_date
    )
    if source_is_current:
        return set(conflict_codes)
    codes = {str(code) for code in snapshot.frame.get("code", ())}
    corroborated = {
        code
        for code in codes
        if code in histories
        and _last_session_bar(histories[code], expected_session) is not None
        and code not in conflict_codes
    }
    return codes - corroborated


def _suppress_conflicting_candidates(
    candidates: Sequence[Candidate], conflict_codes: set[str]
) -> tuple[Candidate, ...]:
    return tuple(
        replace(
            item,
            trigger="观望：价格来源冲突",
            warning=_PRICE_CONFLICT_CANDIDATE_WARNING,
        )
        if item.code in conflict_codes
        else item
        for item in candidates
    )


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


def _atomic_write(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
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
    """Stage and atomically publish one coherent report-key artifact directory."""

    root = Path(output_dir)
    parent = root / ("test" if test_email else "production")
    parent.mkdir(parents=True, exist_ok=True)
    parent.chmod(0o700)
    final = parent / report_key
    staging = parent / f".{report_key}.{uuid.uuid4().hex}.staging"
    backup = parent / f".{report_key}.{uuid.uuid4().hex}.backup"
    try:
        staging.mkdir(mode=0o700)
        _atomic_write(staging / "report.html", rendered.html)
        _atomic_write(staging / "report.txt", rendered.text)
        _atomic_write(
            staging / "manifest.json",
            json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        if final.exists():
            final.replace(backup)
        staging.replace(final)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not final.exists():
            backup.replace(final)
        raise
    return ArtifactPaths(final / "report.html", final / "report.txt", final / "manifest.json")


def _warning_codes(modules: Mapping[str, ModuleResult]) -> list[str]:
    codes: list[str] = []
    for name, result in modules.items():
        for index, warning in enumerate(result.warnings, start=1):
            codes.append(
                warning
                if warning == _PRICE_CONFLICT_WARNING_CODE
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
            or item.warning == _PRICE_CONFLICT_CANDIDATE_WARNING
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
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "report_key": session.report_key,
        "mode": session.mode.value,
        "trading_date": session.trading_date.isoformat(),
        "generated_at": session.now_shanghai.isoformat(),
        "final_state": final_state,
        "test_email": test_email,
        "module_statuses": {name: result.status for name, result in modules.items()},
        "source_timestamps": {name: result.observed_at.isoformat() for name, result in modules.items()},
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


def _session_identity_is_valid(session: ReportSession, mode: ReportMode) -> bool:
    timestamp = session.now_shanghai
    return (
        session.mode is mode
        and timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
        and timestamp.date() == session.trading_date
        and session.report_key == f"{session.trading_date.isoformat()}-{mode.value}"
    )


def run_report(
    mode: ReportMode,
    *,
    deps: RunnerDependencies | None = None,
    force: bool = False,
    test_email: bool = False,
    already_sent: bool = False,
    prior_report: Path | str | None = None,
    output_dir: Path | str = "reports/collaborative",
) -> RunResult:
    """Run one report without leaking third-party exception text to its result."""

    normalized_mode = ReportMode(mode)
    active = deps or default_dependencies()
    now = active.clock()
    try:
        session = active.session_builder(normalized_mode, now, scheduled=not force)
    except Exception as exc:
        code = "outside_delivery_window" if str(exc) == "outside delivery window" else "calendar_unavailable"
        return _failure(code)
    if not _session_identity_is_valid(session, normalized_mode):
        return _failure("report_identity_invalid")
    if not session.is_trading_day:
        return RunResult(EXIT_SUCCESS, FinalState.NON_TRADING_DAY_SKIP, session.report_key, {})
    ledger = active.ledger_factory(output_dir)
    if not test_email and (already_sent or ledger.is_sent(session.report_key)):
        return RunResult(EXIT_SUCCESS, FinalState.DUPLICATE_SKIP, session.report_key, {})

    try:
        settings = active.settings_loader()
    except Exception:
        return _failure("configuration_invalid", report_key=session.report_key)

    artifact_writer = active.artifact_writer or write_report_artifacts
    manifest_writer = active.manifest_writer or _atomic_write
    try:
        expected_session = active.data_session_resolver(
            normalized_mode,
            session.trading_date,
            session.now_shanghai,
        )
    except Exception:
        return _failure("calendar_unavailable", report_key=session.report_key)

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
    try:
        snapshot = active.gateway.get_a_share_snapshot()
        if not _snapshot_is_fresh(snapshot, session):
            raise ValueError("stale snapshot")
        modules["market"] = _module("market", snapshot.observed_at, _market_payload(snapshot), *snapshot.warnings)
    except Exception:
        snapshot = None
        modules["market"] = _unavailable("market", session.now_shanghai, "数据不足，建议观望")

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
            histories,
            expected_session=expected_session,
            session=session,
            conflict_codes=conflict_codes,
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
            (*market.warnings, "snapshot_timestamp_untrusted"),
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
                _suppress_conflicting_candidates(screening.short_term, suppressed_codes),
                _suppress_conflicting_candidates(screening.swing, suppressed_codes),
                screening.warnings,
            )
            status = "partial" if history_failures or screening.warnings else "ok"
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

    try:
        gold_data = active.gateway.get_gold_bars()
        gold = active.gold_analyzer(gold_data.frame, capital=settings.capital_cny)
        modules["gold"] = _module("gold", gold_data.observed_at, _as_payload(gold), *gold_data.warnings)
    except Exception:
        modules["gold"] = _unavailable("gold", session.now_shanghai, "黄金模块暂不可用")

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

    current_market_value = sum(
        price * position.quantity
        for position in settings.positions
        if (price := prices.get(position.code))
    )
    available_cash = max(settings.capital_cny - current_market_value, 0.0)
    sizing_payload: dict[str, int] = {}
    sizing_warnings: list[str] = []
    for item in (*screening.short_term, *screening.swing):
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
        rendered = active.renderer(
            normalized_mode,
            session.trading_date,
            modules=modules,
            short_term_candidates=screening.short_term,
            swing_candidates=screening.swing,
            morning_candidates=morning_candidates,
            subject_prefix=_TEST_SUBJECT_PREFIX if test_email else None,
            generated_at=session.now_shanghai,
        )
    except Exception:
        return _failure("report_render_failed", report_key=session.report_key, modules=modules)

    final_state = FinalState.TEST_SENT if test_email else FinalState.SENT
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
        )
        paths = artifact_writer(
            output_dir,
            report_key=session.report_key,
            rendered=rendered,
            manifest=manifest,
            test_email=test_email,
        )
    except Exception:
        return _failure("artifact_write_failed", report_key=session.report_key, modules=modules)

    try:
        active.mail_sender(rendered, test_email=test_email)
    except _DeliveryConfigurationError:
        return _failure("configuration_invalid", report_key=session.report_key, modules=modules, paths=paths)
    except Exception:
        failure_manifest = dict(manifest, final_state=FinalState.HARD_FAILURE.value, error_code="delivery_failed")
        try:
            _atomic_write(
                paths.manifest_path,
                json.dumps(failure_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            )
        except Exception:
            pass
        return _failure("delivery_failed", report_key=session.report_key, modules=modules, paths=paths)

    if not test_email:
        try:
            ledger.mark_sent(session.report_key, session.now_shanghai)
        except Exception:
            modules["delivery_state"] = ModuleResult(
                "delivery_state", "partial", session.now_shanghai, {}, ("delivery_ledger_update_failed",)
            )

    manifest = dict(
        manifest,
        final_state=final_state.value,
        module_statuses={name: result.status for name, result in modules.items()},
        warning_codes=_warning_codes(modules),
    )
    try:
        manifest_writer(
            paths.manifest_path,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
    except Exception:
        modules["delivery_state"] = ModuleResult(
            "delivery_state", "partial", session.now_shanghai, {}, ("manifest_finalize_failed",)
        )
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
