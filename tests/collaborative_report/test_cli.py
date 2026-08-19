from pathlib import Path
from unittest.mock import Mock

import pytest

from src.collaborative_report.cli import main
from src.collaborative_report.models import ReportMode
from src.collaborative_report.runner import EXIT_FAILURE, EXIT_SUCCESS, FinalState, RunResult


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
    assert "2026-08-19-premarket" in output
    assert "duplicate_skip" in output


@pytest.mark.parametrize(
    ("state", "exit_code"),
    [
        (FinalState.SENT, EXIT_SUCCESS),
        (FinalState.TEST_SENT, EXIT_SUCCESS),
        (FinalState.NON_TRADING_DAY_SKIP, EXIT_SUCCESS),
        (FinalState.DUPLICATE_SKIP, EXIT_SUCCESS),
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
