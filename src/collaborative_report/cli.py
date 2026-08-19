"""Command-line interface for collaborative report orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from .models import ReportMode
from .runner import RunResult, run_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and deliver an A-share collaborative daily report")
    parser.add_argument("--mode", required=True, choices=tuple(mode.value for mode in ReportMode))
    parser.add_argument("--force", action="store_true", help="Bypass only the scheduled delivery window")
    parser.add_argument("--test-email", action="store_true", help="Prefix the delivered subject as test-only")
    parser.add_argument("--already-sent", action="store_true", help="Skip a completed production report identity")
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., RunResult] = run_report,
) -> int:
    """Parse arguments, print only redacted status, and return a process code."""

    parser = _parser()
    args = parser.parse_args(argv)
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
