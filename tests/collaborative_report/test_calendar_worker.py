import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.collaborative_report import session as session_module


WORKER_PATH = Path(__file__).parents[2] / "src" / "collaborative_report" / "calendar_worker.py"


def _worker_environment(fake_module_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = [str(fake_module_dir)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return environment


def _write_fake_akshare(module_dir: Path, *, columns: list[str], dates: list[str]) -> None:
    module_dir.mkdir()
    module_dir.joinpath("akshare.py").write_text(
        "\n".join(
            [
                "class FakeFrame:",
                f"    columns = {columns!r}",
                "",
                "    def __getitem__(self, name):",
                "        return [",
                *[f"            {value!r}," for value in dates],
                "        ]",
                "",
                "def tool_trade_date_hist_sina():",
                "    print('provider diagnostic on stdout', flush=True)",
                "    return FakeFrame()",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_worker(fake_module_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(WORKER_PATH)],
        env=_worker_environment(fake_module_dir),
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )


def test_worker_returns_sorted_unique_dates_and_keeps_provider_stdout_out_of_json(tmp_path: Path) -> None:
    _write_fake_akshare(
        tmp_path / "provider",
        columns=["trade_date"],
        dates=["2026-08-20", "2026-08-18", "2026-08-20"],
    )

    result = _run_worker(tmp_path / "provider")

    assert result.returncode == 0
    assert json.loads(result.stdout) == ["2026-08-18", "2026-08-20"]
    assert "provider diagnostic on stdout" not in result.stdout
    assert "provider diagnostic on stdout" in result.stderr


@pytest.mark.parametrize(
    ("columns", "dates"),
    [
        ([], ["2026-08-19"]),
        (["trade_date"], []),
        (["trade_date"], ["2026/08/19"]),
    ],
    ids=["missing-column", "empty-dates", "malformed-date"],
)
def test_worker_fails_for_invalid_provider_data(
    tmp_path: Path,
    columns: list[str],
    dates: list[str],
) -> None:
    provider_dir = tmp_path / "provider"
    _write_fake_akshare(provider_dir, columns=columns, dates=dates)

    result = _run_worker(provider_dir)

    assert result.returncode != 0
    assert result.stdout == ""


def test_parent_translates_real_worker_timeout_and_reaps_child(tmp_path: Path, monkeypatch) -> None:
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    provider_dir.joinpath("akshare.py").write_text(
        "import time\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", _worker_environment(provider_dir)["PYTHONPATH"])
    real_run = session_module.subprocess.run
    real_popen = session_module.subprocess.Popen
    children = []

    def track_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def run_with_short_timeout(*args, **kwargs):
        kwargs["timeout"] = 0.5
        return real_run(*args, **kwargs)

    monkeypatch.setattr(session_module.subprocess, "run", run_with_short_timeout)
    monkeypatch.setattr(session_module.subprocess, "Popen", track_child)

    with pytest.raises(RuntimeError, match="^trading calendar unavailable$"):
        session_module._load_akshare_xshg_sessions()

    assert len(children) == 1
    assert children[0].returncode is not None
    assert children[0].poll() is not None
