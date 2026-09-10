import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.collaborative_report.cli import main
from src.collaborative_report.market_data import MarketDataset
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
    prior_sector = tmp_path / "prior-sector.json"

    exit_code = main(
        [
            "--mode",
            "postmarket",
            "--force",
            "--test-email",
            "--already-sent",
            "--prior-report",
            str(prior),
            "--prior-sector-report",
            str(prior_sector),
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
        preview_only=False,
        already_sent=True,
        prior_report=prior,
        prior_sector_report=prior_sector,
        prior_market_snapshot=None,
        market_snapshot_output=None,
        output_dir=tmp_path,
    )
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "report_key": "2026-08-19-premarket",
        "module_statuses": {},
        "artifact_path": None,
        "final_state": "duplicate_skip",
    }


def test_cli_includes_safe_runner_error_code(tmp_path, capsys) -> None:
    runner = Mock(
        return_value=RunResult(
            exit_code=EXIT_FAILURE,
            final_state=FinalState.HARD_FAILURE,
            report_key="2026-08-19-premarket",
            error_code="configuration_invalid",
        )
    )

    assert main(["--mode", "premarket", "--output-dir", str(tmp_path)], runner=runner) == EXIT_FAILURE

    assert json.loads(capsys.readouterr().out)["error_code"] == "configuration_invalid"


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


def test_cli_rejects_premarket_prior_sector_report_without_calling_runner(tmp_path) -> None:
    runner = Mock()

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--mode",
                "premarket",
                "--prior-sector-report",
                str(tmp_path / "prior-sector.json"),
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
    assert "--prior-sector-report" in completed.stdout
    assert "--prior-market-snapshot" in completed.stdout
    assert "--market-snapshot-output" in completed.stdout
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


class QueryGateway:
    def __init__(self) -> None:
        self.dataset = MarketDataset(
            pd.DataFrame([{"code": "600000", "price": 10.0}]),
            "ths.fuyao.a_share_snapshot",
            NOW,
        )

    def get_a_share_snapshot(self, codes):
        assert codes == ["600000"]
        return self.dataset

    def get_daily_bars(self, code, expected_session, *, days=160):
        return self.dataset

    def get_ths_financial_indicators(self, code, report):
        return self.dataset

    def get_ths_hot_stock_list(self, period):
        return self.dataset

    def get_ths_index_catalog(self, tag):
        return self.dataset

    def get_ths_index_constituents(self, thscode):
        return self.dataset

    def get_ths_index_snapshot(self, thscodes):
        return self.dataset

    def get_ths_index_bars(self, thscode, expected_session, *, days=160):
        return self.dataset


@pytest.mark.parametrize(
    "argv",
    [
        ["query", "quote", "600000"],
        ["query", "bars", "600000", "--expected-session", "2026-08-19"],
        ["query", "financials", "600000", "--report", "2026-2"],
        ["query", "hot-list", "--period", "day"],
        ["query", "index", "catalog", "--tag", "industry"],
        ["query", "index", "constituents", "886042.TI"],
        ["query", "index", "quote", "886042.TI"],
        ["query", "index", "bars", "886042.TI", "--expected-session", "2026-08-19"],
    ],
)
def test_cli_query_subcommands_emit_normalized_json_without_report_or_mail(argv, capsys) -> None:
    gateway = QueryGateway()

    assert main(argv, query_gateway_factory=lambda: gateway) == EXIT_SUCCESS

    assert json.loads(capsys.readouterr().out) == {
        "data": [{"code": "600000", "price": 10.0}],
        "observed_at": NOW.isoformat(),
        "source": "ths.fuyao.a_share_snapshot",
        "source_timestamp": None,
        "warnings": [],
    }


def test_cli_query_failure_is_redacted_and_returns_failure(capsys) -> None:
    secret = "not-for-output"
    gateway = Mock()
    gateway.get_a_share_snapshot.side_effect = RuntimeError(secret)

    assert main(["query", "quote", "600000"], query_gateway_factory=lambda: gateway) == EXIT_FAILURE

    output = capsys.readouterr().out
    assert json.loads(output) == {"error": "query_unavailable", "query": "quote"}
    assert secret not in output
