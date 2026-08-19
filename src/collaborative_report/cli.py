"""Command-line interface for collaborative report orchestration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and deliver an A-share collaborative daily report")
    parser.add_argument("--mode", choices=tuple(mode.value for mode in ReportMode))
    parser.add_argument("--force", action="store_true", help="Bypass only the scheduled delivery window")
    parser.add_argument("--test-email", action="store_true", help="Prefix the delivered subject as test-only")
    parser.add_argument("--already-sent", action="store_true", help="Skip a completed production report identity")
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--ledger-status", metavar="REPORT_KEY")
    actions.add_argument("--reconcile-sent", metavar="REPORT_KEY")
    actions.add_argument("--reconcile-failed", metavar="REPORT_KEY")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., RunResult] = run_report,
    clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> int:
    """Parse arguments, print only redacted status, and return a process code."""

    parser = _parser()
    args = parser.parse_args(argv)
    operator_key = args.ledger_status or args.reconcile_sent or args.reconcile_failed
    if operator_key:
        if args.mode is not None or args.force or args.test_email or args.already_sent or args.prior_report:
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
    result = runner(
        mode,
        force=args.force,
        test_email=args.test_email,
        already_sent=args.already_sent,
        prior_report=args.prior_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_public_dict(), ensure_ascii=False, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
