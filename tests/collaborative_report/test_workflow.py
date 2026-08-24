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


def _production_prior_candidates(artifacts: list[dict], artifact_name: str) -> list[dict]:
    """Model the metadata gate before the workflow downloads a prior report."""

    candidates = [
        artifact
        for artifact in artifacts
        if artifact.get("name") == artifact_name
        and artifact.get("expired") is False
        and str(artifact.get("id", "")).isdigit()
        and str(artifact.get("workflow_run", {}).get("id", "")).isdigit()
    ]
    return sorted(candidates, key=lambda item: (str(item["created_at"]), int(item["id"])), reverse=True)


def _is_valid_prior_manifest(payload: object, report_key: str) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("schema_version") == 1
        and payload.get("report_key") == report_key
        and payload.get("mode") == "premarket"
        and payload.get("trading_date") == report_key.removesuffix("-premarket")
        and payload.get("final_state") == "sent"
        and payload.get("test_email") is False
        and isinstance(payload.get("candidate_state"), list)
    )


def _select_valid_prior_artifact(artifacts: list[dict], manifests: dict[int, object], report_key: str) -> int | None:
    for artifact in _production_prior_candidates(artifacts, f"report-{report_key}"):
        if _is_valid_prior_manifest(manifests.get(int(artifact["id"])), report_key):
            return int(artifact["id"])
    return None


def _external_delivery_gate(
    *,
    test_email: bool,
    sent_query_ok: bool,
    in_doubt_query_ok: bool,
    claim_query_ok: bool,
    sent_exists: bool = False,
    in_doubt_exists: bool = False,
    claim_exists: bool = False,
    reconcile_sent: bool = False,
    claim_upload_ok: bool = True,
) -> dict[str, object]:
    """Model the workflow's production-only cross-run delivery barrier."""

    if test_email:
        return {"action": "deliver", "already_sent": False, "runner_runs": True, "diagnostic_only": False}
    if not sent_query_ok:
        return {
            "action": "blocked",
            "already_sent": False,
            "runner_runs": False,
            "diagnostic_only": True,
            "reason": "actions_marker_query_unavailable",
        }
    if sent_exists:
        return {"action": "deliver", "already_sent": True, "runner_runs": True, "diagnostic_only": False}
    if not in_doubt_query_ok:
        return {
            "action": "blocked",
            "already_sent": False,
            "runner_runs": False,
            "diagnostic_only": True,
            "reason": "actions_marker_query_unavailable",
        }
    if in_doubt_exists and reconcile_sent:
        return {"action": "reconcile", "already_sent": False, "runner_runs": False, "diagnostic_only": True}
    if in_doubt_exists:
        return {
            "action": "blocked",
            "already_sent": False,
            "runner_runs": False,
            "diagnostic_only": True,
            "reason": "reconcile_required",
        }
    if not claim_query_ok:
        return {
            "action": "blocked",
            "already_sent": False,
            "runner_runs": False,
            "diagnostic_only": True,
            "reason": "actions_marker_query_unavailable",
        }
    if claim_exists:
        return {
            "action": "blocked",
            "already_sent": False,
            "runner_runs": False,
            "diagnostic_only": True,
            "reason": "claim_reconcile_required",
        }
    if not claim_upload_ok:
        return {
            "action": "blocked",
            "already_sent": False,
            "runner_runs": False,
            "diagnostic_only": True,
            "reason": "claim_upload_failed",
        }
    return {"action": "deliver", "already_sent": False, "runner_runs": True, "diagnostic_only": False}


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
    assert inputs["reconcile_sent"]["type"] == "boolean"
    assert inputs["reconcile_sent"]["default"] == "false"
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"] == {
        "group": "collaborative-report-${{ github.event_name == 'schedule' && (github.event.schedule == '0 1 * * 1-5' && 'premarket' || github.event.schedule == '30 8 * * 1-5' && 'postmarket' || 'invalid-schedule') || inputs.mode }}",
        "cancel-in-progress": "false",
    }

    steps = _steps(workflow)
    assert any(step.get("uses") == "actions/checkout@v5" for step in steps)
    assert any(step.get("uses") == "actions/setup-python@v6" for step in steps)
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
        "RECONCILE_SENT_INPUT": "${{ inputs.reconcile_sent }}",
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
    prior_extract = _step(workflow, "Extract prior candidate state")
    runner = _step(workflow, "Run collaborative report")

    assert "gh api" in duplicate_check["run"]
    assert "/actions/artifacts?name=sent-$REPORT_KEY" in duplicate_check["run"]
    assert "--paginate --slurp" in duplicate_check["run"]
    assert "created_at" in duplicate_check["run"]
    assert "prior-artifact-candidates.tsv" in duplicate_check["run"]
    assert "--already-sent" in runner["run"]
    assert duplicate_check["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "gh run download" in prior_extract["run"]
    assert "--name \"$PRIOR_REPORT_KEY\"" in prior_extract["run"]
    assert "--repo \"$GITHUB_REPOSITORY\"" in prior_extract["run"]
    assert prior_extract["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "continue" in prior_extract["run"]
    assert "^[0-9]+$" in prior_extract["run"]
    assert "candidate_state" in prior_extract["run"]
    assert "manifest.json" in prior_extract["run"]
    assert ".prior-report.json" in prior_extract["run"]
    assert "prior_report_warning=prior_premarket_report_unavailable" in prior_extract["run"]
    assert "--prior-report" in runner["run"]
    assert "curl " not in all_run_content(workflow)


def test_external_marker_handoff_is_production_only_and_sent_marker_is_strict() -> None:
    workflow = _workflow()
    duplicate_check = _step(workflow, "Check external sent marker")
    runner = _step(workflow, "Run collaborative report")
    sent_marker = _step(workflow, "Upload sent marker")

    assert 'if [ "$TEST_EMAIL" = "false" ]' in duplicate_check["run"]
    assert 'if [ "$ALREADY_SENT" = "true" ]' in runner["run"]
    assert 'args+=(--already-sent)' in runner["run"]
    assert "final_state == \"sent\"" in runner["run"]
    assert 'os.environ["TEST_EMAIL"] == "false"' in runner["run"]
    assert "steps.runner.outputs.final_state == 'sent'" in sent_marker["if"]
    assert "steps.context.outputs.test_email == 'false'" in sent_marker["if"]
    assert "steps.runner.outputs.runner_exit == '0'" in sent_marker["if"]


def test_actions_marker_query_outage_blocks_production_before_runner_and_keeps_only_diagnostics() -> None:
    workflow = _workflow()
    duplicate_check = _step(workflow, "Check external sent marker")
    runner = _step(workflow, "Run collaborative report")
    private_report = _step(workflow, "Upload private report")
    diagnostic = _step(workflow, "Upload diagnostic manifest")
    sent_marker = _step(workflow, "Upload sent marker")

    blocked = _external_delivery_gate(
        test_email=False,
        sent_query_ok=False,
        in_doubt_query_ok=True,
        claim_query_ok=True,
    )
    assert blocked == {
        "action": "blocked",
        "already_sent": False,
        "runner_runs": False,
        "diagnostic_only": True,
        "reason": "actions_marker_query_unavailable",
    }
    test_email = _external_delivery_gate(
        test_email=True,
        sent_query_ok=False,
        in_doubt_query_ok=False,
        claim_query_ok=False,
    )
    assert test_email["runner_runs"] is True
    assert "in-doubt-$REPORT_KEY" in duplicate_check["run"]
    assert "actions_marker_query_unavailable" in duplicate_check["run"]
    assert "steps.duplicate.outputs.delivery_action == 'deliver'" in runner["if"]
    assert "steps.claim.outcome == 'success'" in runner["if"]
    assert "steps.runner.outputs.report_available" in private_report["if"]
    assert diagnostic["if"] == "${{ always() }}"
    assert "steps.duplicate.outputs.delivery_action" not in sent_marker["if"]


def test_in_doubt_marker_blocks_delivery_and_manual_reconcile_never_calls_runner() -> None:
    workflow = _workflow()
    duplicate_check = _step(workflow, "Check external sent marker")
    runner = _step(workflow, "Run collaborative report")
    reconcile = _step(workflow, "Reconcile verified in-doubt delivery")
    uploads = {step["name"]: step for step in _upload_steps(workflow)}

    blocked = _external_delivery_gate(
        test_email=False,
        sent_query_ok=True,
        in_doubt_query_ok=True,
        claim_query_ok=True,
        in_doubt_exists=True,
    )
    assert blocked["action"] == "blocked"
    assert blocked["reason"] == "reconcile_required"
    stale_in_doubt = _external_delivery_gate(
        test_email=False,
        sent_query_ok=True,
        in_doubt_query_ok=True,
        claim_query_ok=True,
        sent_exists=True,
        in_doubt_exists=True,
    )
    assert stale_in_doubt["action"] == "deliver"
    assert stale_in_doubt["already_sent"] is True
    reconcile_action = _external_delivery_gate(
        test_email=False,
        sent_query_ok=True,
        in_doubt_query_ok=True,
        claim_query_ok=True,
        in_doubt_exists=True,
        reconcile_sent=True,
    )
    assert reconcile_action["action"] == "reconcile"
    assert reconcile_action["runner_runs"] is False
    assert "in_doubt_exists" in duplicate_check["run"]
    assert "reconcile_required" in duplicate_check["run"]
    decision_block = duplicate_check["run"].split(
        'if [ "$delivery_action" = "deliver" ] && [ "$TEST_EMAIL" = "false" ]; then',
        maxsplit=1,
    )[1]
    assert decision_block.index('if [ "$sent_exists" = "true" ]') < decision_block.index(
        'if [ "$in_doubt_exists" = "true" ]'
    )
    assert "operator_action_required" in runner["run"]
    assert "in-doubt-${{ steps.context.outputs.report_key }}" == uploads["Upload in-doubt marker"]["with"]["name"]
    assert "steps.runner.outputs.needs_reconcile == 'true'" in uploads["Upload in-doubt marker"]["if"]
    assert "steps.context.outputs.test_email == 'false'" in uploads["Upload in-doubt marker"]["if"]
    assert "steps.duplicate.outputs.delivery_action == 'reconcile'" in reconcile["if"]
    assert "run_collaborative_report.py" not in reconcile["run"]
    assert "sent-marker.json" in reconcile["run"]


def test_production_claim_is_mandatory_before_runner_and_blocks_crash_retries() -> None:
    workflow = _workflow()
    duplicate_check = _step(workflow, "Check external sent marker")
    claim = _step(workflow, "Upload production delivery claim")
    claim_failure = _step(workflow, "Record claim upload failure")
    runner = _step(workflow, "Run collaborative report")

    post_smtp_crash = _external_delivery_gate(
        test_email=False,
        sent_query_ok=True,
        in_doubt_query_ok=True,
        claim_query_ok=True,
        claim_exists=True,
    )
    assert post_smtp_crash == {
        "action": "blocked",
        "already_sent": False,
        "runner_runs": False,
        "diagnostic_only": True,
        "reason": "claim_reconcile_required",
    }
    claim_upload_failure = _external_delivery_gate(
        test_email=False,
        sent_query_ok=True,
        in_doubt_query_ok=True,
        claim_query_ok=True,
        claim_upload_ok=False,
    )
    assert claim_upload_failure["runner_runs"] is False
    assert claim_upload_failure["reason"] == "claim_upload_failed"
    sent_wins = _external_delivery_gate(
        test_email=False,
        sent_query_ok=True,
        in_doubt_query_ok=False,
        claim_query_ok=False,
        sent_exists=True,
    )
    assert sent_wins["already_sent"] is True
    in_doubt_wins = _external_delivery_gate(
        test_email=False,
        sent_query_ok=True,
        in_doubt_query_ok=True,
        claim_query_ok=True,
        in_doubt_exists=True,
        claim_exists=True,
    )
    assert in_doubt_wins["reason"] == "reconcile_required"
    test_email = _external_delivery_gate(
        test_email=True,
        sent_query_ok=False,
        in_doubt_query_ok=False,
        claim_query_ok=False,
        claim_exists=True,
        claim_upload_ok=False,
    )
    assert test_email["runner_runs"] is True
    assert "/actions/artifacts?name=claim-$REPORT_KEY" in duplicate_check["run"]
    assert duplicate_check["run"].index("sent-$REPORT_KEY") < duplicate_check["run"].index("in-doubt-$REPORT_KEY")
    assert duplicate_check["run"].index("in-doubt-$REPORT_KEY") < duplicate_check["run"].index("claim-$REPORT_KEY")
    assert 'if [ "$delivery_action" = "deliver" ] && [ "$sent_exists" = "false" ]; then' in duplicate_check["run"]
    assert 'if [ "$delivery_action" = "deliver" ] && [ "$sent_exists" = "false" ] && [ "$in_doubt_exists" = "false" ]; then' in duplicate_check["run"]
    assert claim["with"]["name"] == "claim-${{ steps.context.outputs.report_key }}"
    assert claim["with"]["path"] == ".workflow-artifacts/claim-marker.json"
    assert claim["with"]["include-hidden-files"] == "true"
    assert claim["with"]["retention-days"] == "7"
    assert "steps.context.outputs.test_email == 'false'" in claim["if"]
    assert _steps(workflow).index(claim) < _steps(workflow).index(runner)
    assert "steps.claim.outcome == 'success'" in runner["if"]
    assert "claim_upload_failed" in claim_failure["run"]
    assert "failure()" in claim_failure["if"]
    assert 'if [ "$block_reason" = "reconcile_required" ] || [ "$block_reason" = "claim_reconcile_required" ]; then' in duplicate_check["run"]
    assert '"state": "claimed"' in _step(workflow, "Prepare production delivery claim")["run"]
    assert '"timestamp"' in _step(workflow, "Prepare production delivery claim")["run"]
    assert "EMAIL_" not in _step(workflow, "Prepare production delivery claim")["run"]


def test_newer_test_artifact_cannot_obscure_an_older_production_prior_report() -> None:
    report_key = "2026-08-20-premarket"
    artifacts = [
        {"id": 99, "name": f"test-report-{report_key}", "expired": False, "created_at": "2026-08-20T02:00:00Z", "workflow_run": {"id": 999}},
        {"id": 98, "name": f"report-{report_key}", "expired": False, "created_at": "2026-08-20T01:00:00Z", "workflow_run": {"id": 998}},
    ]
    manifests = {98: {"schema_version": 1, "report_key": report_key, "mode": "premarket", "trading_date": "2026-08-20", "final_state": "sent", "test_email": False, "candidate_state": []}}

    assert _select_valid_prior_artifact(artifacts, manifests, report_key) == 98


def test_invalid_newest_production_artifact_falls_back_to_older_valid_prior_report() -> None:
    report_key = "2026-08-20-premarket"
    artifacts = [
        {"id": 99, "name": f"report-{report_key}", "expired": False, "created_at": "2026-08-20T02:00:00Z", "workflow_run": {"id": 999}},
        {"id": 98, "name": f"report-{report_key}", "expired": False, "created_at": "2026-08-20T01:00:00Z", "workflow_run": {"id": 998}},
    ]
    manifests = {
        99: {"schema_version": 1, "report_key": report_key, "mode": "premarket", "trading_date": "2026-08-20", "final_state": "test_sent", "test_email": True, "candidate_state": []},
        98: {"schema_version": 1, "report_key": report_key, "mode": "premarket", "trading_date": "2026-08-20", "final_state": "sent", "test_email": False, "candidate_state": []},
    }

    assert _select_valid_prior_artifact(artifacts, manifests, report_key) == 98


def test_unavailable_or_invalid_prior_candidates_leave_the_fixed_warning_path() -> None:
    report_key = "2026-08-20-premarket"
    artifacts = [{"id": 99, "name": f"report-{report_key}", "expired": False, "created_at": "2026-08-20T02:00:00Z", "workflow_run": {"id": 999}}]

    assert _select_valid_prior_artifact(artifacts, {99: {}}, report_key) is None
    assert "prior_premarket_report_unavailable" in _step(_workflow(), "Extract prior candidate state")["run"]


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
        "THS_API_KEY",
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

    assert set(by_name) == {
        "Upload private report",
        "Upload diagnostic manifest",
        "Upload sent marker",
        "Upload in-doubt marker",
        "Upload production delivery claim",
    }
    assert by_name["Upload private report"]["with"]["name"] == "${{ steps.runner.outputs.report_artifact_name }}"
    assert by_name["Upload diagnostic manifest"]["with"]["name"] == "diagnostic-${{ steps.context.outputs.report_key }}"
    assert by_name["Upload sent marker"]["with"]["name"] == "sent-${{ steps.context.outputs.report_key }}"
    assert by_name["Upload production delivery claim"]["with"]["name"] == "claim-${{ steps.context.outputs.report_key }}"
    assert all(step["with"]["retention-days"] == "7" for step in uploads)
    assert "steps.runner.outputs.final_state == 'sent'" in by_name["Upload sent marker"]["if"]
    assert "steps.context.outputs.test_email == 'false'" in by_name["Upload sent marker"]["if"]
    assert "steps.runner.outputs.runner_exit == '0'" in by_name["Upload sent marker"]["if"]
    assert "report.html" not in _step(workflow, "Write safe workflow summary")["run"]
    assert "report.txt" not in _step(workflow, "Write safe workflow summary")["run"]
    runner = _step(workflow, "Run collaborative report")["run"]
    assert "'test-report' if channel == 'test' else 'report'" in runner


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
    assert "local delivery ledger remains the source of truth" not in documentation
    assert "cross-run duplicate guard" in documentation
    assert "Within a single run" in documentation
    assert "test-report-<report-key>" in documentation
    assert "reconcile_sent=true" in documentation
    assert "verify QQ delivery directly" in documentation
    assert "claim-<report-key>" in documentation
    assert "confirming no message was delivered" in documentation
    assert "Do not auto-clear or auto-retry claims" in documentation
    assert "does not resolve GitHub cloud markers" in documentation
    assert "THS_API_KEY" in documentation
    assert "query quote" in documentation
    assert "never create a candidate" in documentation

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
