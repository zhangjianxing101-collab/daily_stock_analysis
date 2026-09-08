"""Command-line interface for collaborative report orchestration."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .market_data import MarketDataGateway, MarketDataset
from .models import ReportMode
from .runner import (
    EXIT_FAILURE,
    EXIT_SUCCESS,
    DeliveryState,
    FinalState,
    LocalDeliveryLedger,
    RunResult,
    finalize_prepared_artifact,
    publish_report_artifacts,
    run_report,
)
from .settings import ThsSettings
from .ths_market_data import ThsHotListPeriod, ThsIndexTag, ThsMarketDataClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and deliver an A-share collaborative daily report")
    parser.add_argument("--mode", choices=tuple(mode.value for mode in ReportMode))
    parser.add_argument("--force", action="store_true", help="Bypass only the scheduled delivery window")
    parser.add_argument("--test-email", action="store_true", help="Prefix the delivered subject as test-only")
    parser.add_argument("--no-send", action="store_true", help="Generate artifacts only; never send email or update delivery state")
    parser.add_argument("--already-sent", action="store_true", help="Skip a completed production report identity")
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--prior-sector-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--ledger-status", metavar="REPORT_KEY")
    actions.add_argument("--reconcile-sent", metavar="REPORT_KEY")
    actions.add_argument("--reconcile-failed", metavar="REPORT_KEY")
    commands = parser.add_subparsers(dest="command")
    query = commands.add_parser("query", help="Read normalized market data without generating a trade")
    query_actions = query.add_subparsers(dest="query_command", required=True)

    quote = query_actions.add_parser("quote", help="Read one or more A-share quotes")
    quote.add_argument("codes", nargs="+", help="Six-digit A-share codes")

    bars = query_actions.add_parser("bars", help="Read A-share daily bars")
    bars.add_argument("code", help="Six-digit A-share code")
    bars.add_argument("--expected-session", type=date.fromisoformat, default=date.today())
    bars.add_argument("--days", type=int, default=160)

    financials = query_actions.add_parser("financials", help="Read THS financial indicators")
    financials.add_argument("code", help="Six-digit A-share code")
    financials.add_argument("--report", required=True, help="Financial period in YYYY-1 through YYYY-4")

    hot_list = query_actions.add_parser("hot-list", help="Read the THS hot-stock list")
    hot_list.add_argument("--period", choices=tuple(item.value for item in ThsHotListPeriod), default="day")

    index = query_actions.add_parser("index", help="Read THS industry or concept index data")
    index_actions = index.add_subparsers(dest="index_command", required=True)
    catalog = index_actions.add_parser("catalog", help="Read the index catalog")
    catalog.add_argument("--tag", choices=tuple(item.value for item in ThsIndexTag), default=ThsIndexTag.INDUSTRY.value)
    constituents = index_actions.add_parser("constituents", help="Read one index's constituents")
    constituents.add_argument("thscode")
    index_quote = index_actions.add_parser("quote", help="Read one or more index quotes")
    index_quote.add_argument("thscodes", nargs="+")
    index_bars = index_actions.add_parser("bars", help="Read index daily bars")
    index_bars.add_argument("thscode")
    index_bars.add_argument("--expected-session", type=date.fromisoformat, default=date.today())
    index_bars.add_argument("--days", type=int, default=160)
    return parser


def _default_query_gateway() -> MarketDataGateway:
    return MarketDataGateway(ths_client=ThsMarketDataClient(ThsSettings.from_env()))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    try:
        if value != value:
            return None
    except Exception:
        return str(value)
    return str(value)


def _dataset_payload(dataset: MarketDataset) -> dict[str, Any]:
    records = json.loads(dataset.frame.to_json(orient="records", date_format="iso"))
    return {
        "source": dataset.source,
        "observed_at": dataset.observed_at.isoformat(),
        "source_timestamp": dataset.source_timestamp.isoformat() if dataset.source_timestamp else None,
        "warnings": list(dataset.warnings),
        "data": _json_safe(records),
    }


def _run_query(args: argparse.Namespace, gateway: MarketDataGateway) -> dict[str, Any]:
    if args.query_command == "quote":
        return _dataset_payload(gateway.get_a_share_snapshot(args.codes))
    if args.query_command == "bars":
        return _dataset_payload(gateway.get_daily_bars(args.code, args.expected_session, days=args.days))
    if args.query_command == "financials":
        return _dataset_payload(gateway.get_ths_financial_indicators(args.code, args.report))
    if args.query_command == "hot-list":
        return _dataset_payload(gateway.get_ths_hot_stock_list(ThsHotListPeriod(args.period)))
    if args.query_command != "index":
        raise ValueError("unsupported query")
    if args.index_command == "catalog":
        return _dataset_payload(gateway.get_ths_index_catalog(ThsIndexTag(args.tag)))
    if args.index_command == "constituents":
        return _dataset_payload(gateway.get_ths_index_constituents(args.thscode))
    if args.index_command == "quote":
        return _dataset_payload(gateway.get_ths_index_snapshot(args.thscodes))
    if args.index_command == "bars":
        return _dataset_payload(gateway.get_ths_index_bars(args.thscode, args.expected_session, days=args.days))
    raise ValueError("unsupported index query")


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., RunResult] = run_report,
    clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    query_gateway_factory: Callable[[], MarketDataGateway] = _default_query_gateway,
) -> int:
    """Parse arguments, print only redacted status, and return a process code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "query":
        try:
            print(json.dumps(_run_query(args, query_gateway_factory()), ensure_ascii=False, sort_keys=True))
            return EXIT_SUCCESS
        except Exception:
            print(json.dumps({"error": "query_unavailable", "query": args.query_command}, sort_keys=True))
            return EXIT_FAILURE
    if args.output_dir is None:
        parser.error("--output-dir is required for report and ledger actions")
    operator_key = args.ledger_status or args.reconcile_sent or args.reconcile_failed
    if operator_key:
        if (
            args.mode is not None
            or args.force
            or args.test_email
            or args.no_send
            or args.already_sent
            or args.prior_report
            or args.prior_sector_report
        ):
            parser.error("ledger operator actions cannot be combined with report-run arguments")
        ledger = LocalDeliveryLedger(args.output_dir)
        try:
            record = ledger.record(operator_key)
            if args.ledger_status:
                print(
                    json.dumps(
                        {"report_key": operator_key, "state": record.state.value if record else "absent"},
                        sort_keys=True,
                    )
                )
                return EXIT_SUCCESS
            if record is None or record.state is not DeliveryState.IN_DOUBT:
                raise ValueError("invalid transition")
            now = clock()
            if args.reconcile_failed:
                ledger.reconcile(operator_key, DeliveryState.FAILED, now)
                state = DeliveryState.FAILED
                action = "reconcile_failed"
            else:
                if record.attempt_id is None:
                    raise ValueError("invalid transition")
                finalized = finalize_prepared_artifact(
                    args.output_dir,
                    report_key=operator_key,
                    attempt_id=record.attempt_id,
                    final_state=FinalState.SENT,
                )
                ledger.reconcile(
                    operator_key,
                    DeliveryState.SENT,
                    now,
                    attempt_id=finalized.attempt_id,
                )
                publish_report_artifacts(
                    args.output_dir,
                    report_key=operator_key,
                    paths=finalized,
                )
                state = DeliveryState.SENT
                action = "reconcile_sent"
            print(json.dumps({"action": action, "report_key": operator_key, "state": state.value}, sort_keys=True))
            return EXIT_SUCCESS
        except Exception:
            print(json.dumps({"error": "invalid_transition", "report_key": operator_key}, sort_keys=True))
            return EXIT_FAILURE
    if args.mode is None:
        parser.error("--mode is required for report runs")
    mode = ReportMode(args.mode)
    if mode is ReportMode.PREMARKET and args.prior_report is not None:
        parser.error("--prior-report is valid only for postmarket mode")
    if mode is ReportMode.PREMARKET and args.prior_sector_report is not None:
        parser.error("--prior-sector-report is valid only for postmarket mode")
    result = runner(
        mode,
        force=args.force,
        test_email=args.test_email,
        preview_only=args.no_send,
        already_sent=args.already_sent,
        prior_report=args.prior_report,
        prior_sector_report=args.prior_sector_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_public_dict(), ensure_ascii=False, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
