import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from src.collaborative_report.cli import main
from src.collaborative_report.models import ReportMode
from src.collaborative_report.report import RenderedReport
from src.collaborative_report.runner import (
    EXIT_FAILURE,
    EXIT_SUCCESS,
    DeliveryState,
    FinalState,
    LocalDeliveryLedger,
    RunResult,
    write_report_artifacts,
)


NOW = datetime(2026, 8, 19, 17, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def result(exit_code: int, state: FinalState) -> RunResult:
    return RunResult(
        exit_code=exit_code,
        final_state=state,
        report_key="2026-08-19-premarket",
        modules={},
    )


def test_cli_passes_all_task_9_arguments_and_returns_success(tmp_path, capsys) -> None:
    runner = Mock(return_value=result(EXIT_SUCCESS, FinalState.DUPLICATE_SKIP))
    prior = tmp_path / "prior.json"

    exit_code = main(
        [
            "--mode",
            "postmarket",
            "--force",
            "--test-email",
            "--already-sent",
            "--prior-report",
            str(prior),
            "--output-dir",
            str(tmp_path),
        ],
        runner=runner,
    )

    assert exit_code == EXIT_SUCCESS
    runner.assert_called_once_with(
        ReportMode.POSTMARKET,
        force=True,
        test_email=True,
        already_sent=True,
        prior_report=prior,
        output_dir=tmp_path,
    )
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "report_key": "2026-08-19-premarket",
        "module_statuses": {},
        "artifact_path": None,
        "final_state": "duplicate_skip",
    }


@pytest.mark.parametrize(
    ("state", "exit_code"),
    [
        (FinalState.SENT, EXIT_SUCCESS),
        (FinalState.TEST_SENT, EXIT_SUCCESS),
        (FinalState.NON_TRADING_DAY_SKIP, EXIT_SUCCESS),
        (FinalState.DUPLICATE_SKIP, EXIT_SUCCESS),
        (FinalState.OPERATOR_ACTION_REQUIRED, EXIT_FAILURE),
        (FinalState.HARD_FAILURE, EXIT_FAILURE),
    ],
)
def test_cli_returns_runner_exit_code(state, exit_code, tmp_path) -> None:
    runner = Mock(return_value=result(exit_code, state))

    assert main(["--mode", "premarket", "--output-dir", str(tmp_path)], runner=runner) == exit_code


def test_cli_rejects_premarket_prior_report_without_calling_runner(tmp_path) -> None:
    runner = Mock()

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--mode",
                "premarket",
                "--prior-report",
                str(tmp_path / "prior.json"),
                "--output-dir",
                str(tmp_path),
            ],
            runner=runner,
        )

    assert error.value.code == 2
    runner.assert_not_called()


def test_repository_entrypoint_help_executes_without_running_report() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_collaborative_report.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0
    assert "--mode {premarket,postmarket}" in completed.stdout
    assert "--already-sent" in completed.stdout
    assert "--ledger-status" in completed.stdout
    assert "--reconcile-sent" in completed.stdout
    assert "--reconcile-failed" in completed.stdout
    assert completed.stderr == ""


def test_cli_status_and_failed_reconciliation_are_private_and_audited(tmp_path, capsys) -> None:
    ledger = LocalDeliveryLedger(tmp_path)
    claim = ledger.claim("2026-08-19-premarket", NOW, attempt_id="a" * 32)
    ledger.begin_sending("2026-08-19-premarket", claim, NOW)
    ledger.mark_in_doubt("2026-08-19-premarket", claim, NOW, attempt_id="a" * 32)

    assert main(
        ["--ledger-status", "2026-08-19-premarket", "--output-dir", str(tmp_path)],
        clock=lambda: NOW,
    ) == EXIT_SUCCESS
    assert json.loads(capsys.readouterr().out) == {
        "report_key": "2026-08-19-premarket",
        "state": "in_doubt",
    }

    assert main(
        ["--reconcile-failed", "2026-08-19-premarket", "--output-dir", str(tmp_path)],
        clock=lambda: NOW,
    ) == EXIT_SUCCESS
    output = json.loads(capsys.readouterr().out)
    assert output == {"action": "reconcile_failed", "report_key": "2026-08-19-premarket", "state": "failed"}
    audit = (tmp_path / ".delivery-ledger" / "audit.jsonl").read_text(encoding="utf-8")
    assert json.loads(audit) == {
        "action": "reconcile_failed",
        "report_key": "2026-08-19-premarket",
        "timestamp": NOW.isoformat(),
    }
    assert "@" not in audit


def test_cli_reconcile_sent_finalizes_attempt_and_invalid_transition_fails(tmp_path, capsys) -> None:
    prepared = write_report_artifacts(
        tmp_path,
        report_key="2026-08-19-premarket",
        rendered=RenderedReport("subject", "html", "text"),
        manifest={"report_key": "2026-08-19-premarket", "final_state": "prepared"},
    )
    ledger = LocalDeliveryLedger(tmp_path)
    claim = ledger.claim("2026-08-19-premarket", NOW, attempt_id=prepared.attempt_id)
    ledger.begin_sending("2026-08-19-premarket", claim, NOW)
    ledger.mark_in_doubt(
        "2026-08-19-premarket", claim, NOW, attempt_id=prepared.attempt_id
    )

    assert main(
        ["--reconcile-sent", "2026-08-19-premarket", "--output-dir", str(tmp_path)],
        clock=lambda: NOW,
    ) == EXIT_SUCCESS
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "sent"
    pointer = json.loads(
        (tmp_path / "production" / "2026-08-19-premarket" / "current.json").read_text(
            encoding="utf-8"
        )
    )
    assert pointer["final_state"] == "sent"

    assert main(
        ["--reconcile-failed", "2026-08-19-premarket", "--output-dir", str(tmp_path)],
        clock=lambda: NOW,
    ) == EXIT_FAILURE
    assert json.loads(capsys.readouterr().out)["error"] == "invalid_transition"
