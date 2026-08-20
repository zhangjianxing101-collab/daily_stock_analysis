# Collaborative A-share Report Delivery

This workflow delivers a private A-share report at 09:00 and 16:30 China Standard Time on weekdays. The `premarket` run prepares the morning candidates; the `postmarket` run reviews the same-date morning candidate state when its private report artifact is available.

## QQ SMTP setup

Enable QQ SMTP/POP3 service in QQ Mail settings and generate an SMTP authorization code. Configure the same address as both `EMAIL_SENDER` and `EMAIL_RECEIVERS` when the report is for one mailbox. Use the authorization code as `EMAIL_PASSWORD`; never place the QQ login password in configuration, source control, a workflow summary, or chat.

Add the following encrypted GitHub Secrets under **Settings -> Secrets and variables -> Actions**:

- `EMAIL_SENDER`, `EMAIL_PASSWORD`, and `EMAIL_RECEIVERS`
- `COLLAB_PORTFOLIO_JSON`
- Any AI provider keys used by the report, such as `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`

Do not place sender/receiver addresses, portfolio JSON, authorization codes, or AI keys in repository variables. The `.env.example` values are synthetic placeholders only.

## Repository variables

Set non-secret Repository variables for the delivery policy and model choices. Set `COLLAB_CAPITAL_CNY` to `20000` for a CNY 20,000 baseline, `COLLAB_RISK_FRACTION` to `0.02`, `COLLAB_SHORT_LIMIT` and `COLLAB_SWING_LIMIT` to `5`, and `COLLAB_SCREEN_PREFILTER` to `120`. Model names are repository variables; the workflow has safe defaults when a model variable is omitted.

## Manual operation

Use **Run workflow** for a manual `premarket` or `postmarket` report. The workflow serializes a scheduled and manual run of the same mode, while preserving `cancel-in-progress: false`. For a manual test email, select `premarket`, set `force` to true when outside its normal window, and set `test_email` to true. A test email is marked as a test and never creates a production delivery marker.

Use a production manual run only after verifying the secrets and variables. A normal sent report creates `sent-<report-key>`; a later fresh GitHub job uses that durable marker as the cross-run duplicate guard and passes `--already-sent` to the runner as an external completed identity. Before any production runner call, the workflow must query both the sent and `in-doubt-<report-key>` markers. If either query is unavailable, it fails closed before mail delivery and writes only the redacted diagnostic artifact. Test email bypasses production marker lookup. Within a single run or a locally durable output directory, the runner's local delivery ledger is the delivery state machine and a second barrier; it is ephemeral on GitHub-hosted runners and is not cross-run state.

An ambiguous QQ delivery or `operator_action_required` result writes a private seven-day `in-doubt-<report-key>` marker and blocks later production delivery for that report identity, even when a sent marker also exists. First verify QQ delivery directly. If delivery is confirmed, manually dispatch the workflow with the same mode, `reconcile_sent=true`, and `test_email=false`. This narrowly defined action writes a sent marker and audit-safe diagnostic only; it never runs the report runner or sends email. After that reconciliation run succeeds and its sent marker is visible, an authorized repository operator must delete only the matching `in-doubt-<report-key>` artifact in the Actions UI and record the QQ verification. Do not remove an in-doubt marker before that confirmation, and do not use reconciliation when QQ delivery cannot be verified. The local CLI reconciliation command does not resolve GitHub cloud markers.

## Privacy and retention

The private production report artifact is named `report-<report-key>` and can contain portfolio analysis; a test email uses the separate `test-report-<report-key>` name and cannot obscure production prior-report lookup. Both are available only to repository users with Actions artifact access. Postmarket checks production artifacts newest-first, validates the redacted premarket state, and skips invalid candidates until it finds a valid one; otherwise it records `prior_premarket_report_unavailable`. The diagnostic artifact is redacted and contains only report state, module statuses, source timestamps, and warning codes. Workflow logs and the concise summary do not print the report HTML, report text, mailbox values, portfolio values, credentials, or raw model responses. All artifacts expire after seven days.

If an authorization code, mailbox, portfolio value, or AI key leaks, revoke or rotate it immediately in its provider, replace the encrypted GitHub Secret, audit workflow runs and repository history, and then send a new manual test email. Never reuse a QQ login password as an SMTP authorization code.
