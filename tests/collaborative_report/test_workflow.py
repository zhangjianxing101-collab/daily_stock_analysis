"""Contracts for the private collaborative-report GitHub Actions workflow."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/01-collaborative-report.yml"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
DOCS_PATH = REPO_ROOT / "docs/collaborative-report.md"


def _workflow() -> dict:
    # BaseLoader keeps GitHub's YAML 1.2-style "on" key as text under PyYAML.
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _steps(workflow: dict) -> list[dict]:
    return workflow["jobs"]["deliver"]["steps"]


def _step(workflow: dict, name: str) -> dict:
    return next(step for step in _steps(workflow) if step.get("name") == name)


def _upload_steps(workflow: dict) -> list[dict]:
    return [step for step in _steps(workflow) if step.get("uses") == "actions/upload-artifact@v6"]


def test_workflow_trigger_permissions_concurrency_and_toolchain_contract() -> None:
    workflow = _workflow()
    trigger = workflow["on"]

    assert [item["cron"] for item in trigger["schedule"]] == ["0 1 * * 1-5", "30 8 * * 1-5"]
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert inputs["mode"] == {
        "description": "Report mode",
        "type": "choice",
        "options": ["premarket", "postmarket"],
        "default": "premarket",
        "required": "false",
    }
    assert inputs["force"]["type"] == "boolean"
    assert inputs["force"]["default"] == "false"
    assert inputs["test_email"]["type"] == "boolean"
    assert inputs["test_email"]["default"] == "false"
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"] == {
        "group": "collaborative-report-${{ github.event.schedule || inputs.mode }}",
        "cancel-in-progress": "false",
    }

    steps = _steps(workflow)
    assert any(step.get("uses") == "actions/checkout@v5" for step in steps)
    assert any(step.get("uses") == "actions/setup-python@v6" for step in steps)
    assert any(step.get("uses") == "actions/download-artifact@v7" for step in steps)
    assert all(step.get("uses") == "actions/upload-artifact@v6" for step in _upload_steps(workflow))
    assert _step(workflow, "Set up Python")["with"]["python-version"] == "3.11"
    assert "pip install -r requirements.txt" in _step(workflow, "Install dependencies")["run"]


def test_workflow_validates_context_uses_safe_argument_arrays_and_preserves_runner_failures() -> None:
    workflow = _workflow()
    context = _step(workflow, "Resolve report context")
    runner = _step(workflow, "Run collaborative report")
    final_gate = _step(workflow, "Fail when the runner failed")
    all_run_content = "\n".join(str(step.get("run", "")) for step in _steps(workflow))

    assert context["env"] == {
        "EVENT_NAME": "${{ github.event_name }}",
        "MANUAL_MODE": "${{ inputs.mode }}",
        "SCHEDULE": "${{ github.event.schedule }}",
        "FORCE_INPUT": "${{ inputs.force }}",
        "TEST_EMAIL_INPUT": "${{ inputs.test_email }}",
    }
    assert '"0 1 * * 1-5") mode="premarket"' in context["run"]
    assert '"30 8 * * 1-5") mode="postmarket"' in context["run"]
    assert "premarket|postmarket" in context["run"]
    assert "date +%F" in context["run"]
    assert "args=(" in runner["run"]
    assert 'args+=(--mode "$MODE")' in runner["run"]
    assert 'args+=(--output-dir "$OUTPUT_DIR")' in runner["run"]
    assert '"${args[@]}"' in runner["run"]
    assert "set +e" in runner["run"]
    assert "runner_exit=$?" in runner["run"]
    assert 'exit "$RUNNER_EXIT"' in final_gate["run"]
    assert "inputs." not in all_run_content
    assert "github.event." not in all_run_content


def test_workflow_uses_external_duplicate_check_and_redacted_postmarket_prior_state() -> None:
    workflow = _workflow()
    duplicate_check = _step(workflow, "Check external sent marker")
    prior_download = _step(workflow, "Download prior premarket report")
    prior_extract = _step(workflow, "Extract prior candidate state")
    runner = _step(workflow, "Run collaborative report")

    assert "gh api" in duplicate_check["run"]
    assert "/actions/artifacts?name=sent-$REPORT_KEY" in duplicate_check["run"]
    assert "prior_run_id" in duplicate_check["run"]
    assert "^[0-9]+$" in duplicate_check["run"]
    assert "--already-sent" in runner["run"]
    assert duplicate_check["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert prior_download["uses"] == "actions/download-artifact@v7"
    assert prior_download["with"]["name"] == "${{ steps.context.outputs.prior_report_artifact }}"
    assert prior_download["with"]["path"] == ".prior-report-download"
    assert prior_download["with"]["github-token"] == "${{ github.token }}"
    assert prior_download["with"]["repository"] == "${{ github.repository }}"
    assert prior_download["with"]["run-id"] == "${{ steps.duplicate.outputs.prior_run_id }}"
    assert prior_download["continue-on-error"] == "true"
    assert "candidate_state" in prior_extract["run"]
    assert "manifest.json" in prior_extract["run"]
    assert ".prior-report.json" in prior_extract["run"]
    assert "prior_report_warning=prior_premarket_report_unavailable" in prior_extract["run"]
    assert "--prior-report" in runner["run"]
    assert "curl " not in all_run_content(workflow)
    assert "gh api" not in prior_extract["run"]


def all_run_content(workflow: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(workflow))


def test_workflow_maps_secrets_and_variables_without_literal_personal_data() -> None:
    workflow = _workflow()
    env = _step(workflow, "Run collaborative report")["env"]
    secret_values = {
        "EMAIL_SENDER",
        "EMAIL_PASSWORD",
        "EMAIL_RECEIVERS",
        "COLLAB_PORTFOLIO_JSON",
        "ANSPIRE_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEYS",
        "AIHUBMIX_KEY",
        "OPENAI_API_KEY",
        "OPENAI_API_KEYS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEYS",
        "LITELLM_API_KEY",
        "LLM_PRIMARY_API_KEY",
        "LLM_PRIMARY_API_KEYS",
        "LLM_GEMINI_API_KEY",
        "LLM_GEMINI_API_KEYS",
        "LLM_DEEPSEEK_API_KEY",
        "LLM_DEEPSEEK_API_KEYS",
        "LLM_ANSPIRE_API_KEY",
        "LLM_ANSPIRE_API_KEYS",
        "LLM_OPENAI_API_KEY",
        "LLM_OPENAI_API_KEYS",
        "LLM_ANTHROPIC_API_KEY",
        "LLM_ANTHROPIC_API_KEYS",
    }
    for key in secret_values:
        assert env[key] == f"${{{{ secrets.{key} }}}}"

    assert env["COLLAB_CAPITAL_CNY"] == "${{ vars.COLLAB_CAPITAL_CNY || '20000' }}"
    assert env["COLLAB_RISK_FRACTION"] == "${{ vars.COLLAB_RISK_FRACTION || '0.02' }}"
    assert env["COLLAB_SHORT_LIMIT"] == "${{ vars.COLLAB_SHORT_LIMIT || '5' }}"
    assert env["COLLAB_SWING_LIMIT"] == "${{ vars.COLLAB_SWING_LIMIT || '5' }}"
    assert env["COLLAB_SCREEN_PREFILTER"] == "${{ vars.COLLAB_SCREEN_PREFILTER || '120' }}"
    for key, value in env.items():
        if "MODEL" in key:
            assert "secrets." not in value
            assert "vars." in value

    source = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "@qq.com" not in source.lower()
    assert "@gmail.com" not in source.lower()
    assert '"code"' not in source
    assert "cost_price" not in source
    assert "sk-" not in source.lower()


def test_workflow_artifacts_are_private_redacted_short_lived_and_marker_is_strictly_sent_only() -> None:
    workflow = _workflow()
    uploads = _upload_steps(workflow)
    by_name = {step["name"]: step for step in uploads}

    assert set(by_name) == {"Upload private report", "Upload diagnostic manifest", "Upload sent marker"}
    assert by_name["Upload private report"]["with"]["name"] == "report-${{ steps.context.outputs.report_key }}"
    assert by_name["Upload diagnostic manifest"]["with"]["name"] == "diagnostic-${{ steps.context.outputs.report_key }}"
    assert by_name["Upload sent marker"]["with"]["name"] == "sent-${{ steps.context.outputs.report_key }}"
    assert all(step["with"]["retention-days"] == "7" for step in uploads)
    assert "steps.runner.outputs.final_state == 'sent'" in by_name["Upload sent marker"]["if"]
    assert "steps.context.outputs.test_email == 'false'" in by_name["Upload sent marker"]["if"]
    assert "steps.runner.outputs.runner_exit == '0'" in by_name["Upload sent marker"]["if"]
    assert "report.html" not in _step(workflow, "Write safe workflow summary")["run"]
    assert "report.txt" not in _step(workflow, "Write safe workflow summary")["run"]


def test_env_example_and_documentation_are_synthetic_and_cover_secure_operation() -> None:
    env_example = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    documentation = DOCS_PATH.read_text(encoding="utf-8")

    for key in (
        "COLLAB_PORTFOLIO_JSON",
        "COLLAB_CAPITAL_CNY=20000",
        "COLLAB_RISK_FRACTION=0.02",
        "COLLAB_SHORT_LIMIT=5",
        "COLLAB_SWING_LIMIT=5",
        "COLLAB_SCREEN_PREFILTER=120",
    ):
        assert key in env_example
    assert "example.invalid" in env_example
    assert "@qq.com" not in env_example.lower()
    assert 'COLLAB_PORTFOLIO_JSON=[{"code":"000000","quantity":100,"cost_price":10.00}]' in env_example

    for phrase in (
        "QQ",
        "SMTP",
        "authorization code",
        "same address",
        "GitHub Secrets",
        "Repository variables",
        "20,000",
        "test email",
        "premarket",
        "postmarket",
        "private",
        "seven days",
        "duplicate",
        "reconcile",
        "rotate",
        "login password",
    ):
        assert phrase.casefold() in documentation.casefold()
