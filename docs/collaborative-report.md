# Collaborative A-share Report Delivery

This workflow delivers a private A-share report at 09:00 and 16:30 China Standard Time on weekdays. The `premarket` run prepares the morning candidates; the `postmarket` run reviews the same-date morning candidate state when its private report artifact is available.

## QQ SMTP setup

Enable QQ SMTP/POP3 service in QQ Mail settings and generate an SMTP authorization code. Configure the same address as both `EMAIL_SENDER` and `EMAIL_RECEIVERS` when the report is for one mailbox. Use the authorization code as `EMAIL_PASSWORD`; never place the QQ login password in configuration, source control, a workflow summary, or chat.

Add the following encrypted GitHub Secrets under **Settings -> Secrets and variables -> Actions**:

- `EMAIL_SENDER`, `EMAIL_PASSWORD`, and `EMAIL_RECEIVERS`
- `COLLAB_PORTFOLIO_JSON`
- `THS_API_KEY` for the optional 同花顺/福耀 market-data source
- Any AI provider keys used by the report, such as `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`

For existing GitHub setups that saved the market-data key as `THS`, the workflow
uses that encrypted secret when `THS_API_KEY` is absent or empty. If both are set,
`THS_API_KEY` takes precedence. No key is copied into repository variables or
logs. Local runs continue to use the canonical `THS_API_KEY` environment variable.

Do not place sender/receiver addresses, portfolio JSON, authorization codes, or AI keys in repository variables. The `.env.example` values are synthetic placeholders only.

## 同花顺数据与人工查询

When `THS_API_KEY` is configured, the collaborative report prefers 同花顺 data for A-share snapshots and daily bars, then falls back to the existing provider only for recoverable provider failures. Financial indicators, hot lists, and industry-index catalog data are shown as evidence with their source and collection time. They can only re-rank candidates that already passed the deterministic technical rules; they never create a candidate, change stop/target levels, place an order, or remove the requirement for manual confirmation. Missing, stale, invalid, or conflicting provider data is marked as unavailable or observation-only.

Use the read-only local query interface for an explicit connectivity or data check. It prints normalized JSON and never prints `THS_API_KEY` or request headers:

```bash
python scripts/run_collaborative_report.py query quote 600000
python scripts/run_collaborative_report.py query bars 600000 --expected-session 2026-08-21
python scripts/run_collaborative_report.py query financials 600000 --report 2026-2
python scripts/run_collaborative_report.py query hot-list --period day
python scripts/run_collaborative_report.py query index catalog --tag industry
python scripts/run_collaborative_report.py query index constituents 886042.TI
python scripts/run_collaborative_report.py query index quote 886042.TI
python scripts/run_collaborative_report.py query index bars 886042.TI --expected-session 2026-08-21
```

Queries are read-only data retrieval, not trading commands. Review returned data quality and confirm every investment decision manually.

## Snapshot completeness

THS rows without a quoted price are excluded only when volume and traded amount
are also genuinely missing or zero. Malformed text, negative values, duplicate
codes, or missing prices alongside trading activity still fail validation. The
report discloses excluded-row and screening-field coverage counts.
Snapshot pages with different source-session dates are rejected before aggregation.
When any held position lacks a validated price, the report cannot determine available
cash and suppresses all new-position sizing. An explicitly empty portfolio still
uses the configured capital baseline.

Production snapshots supplement THS names, volume ratios and turnover percentages
from the project's public Tencent quote source before screening the full universe.
THS traded amount is never interpreted as turnover percentage. Supplements require
exact identities, independent same-session timestamps at or after 15:00, no future
timestamps, finite metrics, and price agreement within CNY 0.01. Rejected fields
remain unavailable. Codes outside the supplemental provider's supported exchange
prefixes do not block enrichment for the supported universe. Partial coverage marks screening as partial. For conservative
freshness checks, the combined dataset uses the oldest accepted source timestamp;
one expired source can therefore invalidate the combined snapshot.

Preview runs additionally produce bounded `provider-quality` diagnostics containing
only counts and parsed dates. Raw responses, stderr and credentials are not uploaded.

## Gold history completeness

Empty or stale gold responses, or recognized connection/time-out failures, allow
one alternate-range retry. It requests one extra leading calendar day with the
same exclusive end. The entire response is validated before removing the padding
and revalidating the canonical ten-year window. Responses are never spliced;
integrity failures do not trigger recovery, and a second failure stays unavailable.

Gold history downloads end exclusively on the day after the latest completed
futures session. This avoids requesting the ongoing session's incomplete daily
bar. Returned data must still pass all existing date, price, volume, sample-size,
and freshness checks; a stale series or a provider response beyond the cutoff
remains unavailable rather than being silently accepted.

## Acquisition and generation times

Reports invalidated by clock or source-expiry checks are excluded from the
workflow's report upload; their sanitized diagnostic manifest remains available.
Provider logs expose only fixed THS failure categories and validated expected/actual
bar dates, never exception text or response bodies.
The workflow includes these allowlisted lines in `provider_diagnostics`. THS
transport categories distinguish TLS, proxy, connect/read timeout, connection,
and internal-client failures. TLS verification stays enabled, redirects remain
disabled, and retry limits are unchanged.

Source timestamps are checked against the time a response is received, not the
time its request began. Each snapshot page is checked before another page can
be requested. The report keeps its original trading-date identity but uses its
actual final generation time for rendering and the manifest. Clock rollback,
date rollover, genuinely future observations, and data expiring during a run
remain safe failures rather than permission to reuse yesterday's signals.

Market and gold failures include fixed diagnostic codes for known validation
errors and generic stage codes for other exceptions. Raw provider errors,
credentials, request URLs and response bodies are not copied into the report
or diagnostic manifest.

## Repository variables

Set non-secret Repository variables for the delivery policy and model choices. Set `COLLAB_CAPITAL_CNY` to `20000` for a CNY 20,000 baseline, `COLLAB_RISK_FRACTION` to `0.02`, `COLLAB_SHORT_LIMIT` and `COLLAB_SWING_LIMIT` to `5`, and `COLLAB_SCREEN_PREFILTER` to `120`. Model names are repository variables; the workflow has safe defaults when a model variable is omitted.

## Manual operation

Use **Run workflow** for a manual `premarket` or `postmarket` report. The workflow serializes a scheduled and manual run of the same mode, while preserving `cancel-in-progress: false`. For a manual test email, select `premarket`, set `force` to true when outside its normal window, and set `test_email` to true. A test email is marked as a test and never creates a production delivery marker.

Use a production manual run only after verifying the secrets and variables. A normal sent report creates `sent-<report-key>`; a later fresh GitHub job uses that durable marker as the cross-run duplicate guard and passes `--already-sent` to the runner as an external completed identity. Before any production runner call, the workflow queries unexpired `sent-<report-key>`, `in-doubt-<report-key>`, and `claim-<report-key>` artifacts in that order. A sent marker wins; otherwise an in-doubt marker or unresolved claim fails closed before mail delivery and writes only the redacted diagnostic artifact. Test email bypasses all production marker lookup and never creates a production marker. Within a single run or a locally durable output directory, the runner's local delivery ledger is the delivery state machine and a second barrier; it is ephemeral on GitHub-hosted runners and is not cross-run state.

After the production preflight succeeds, the workflow uploads a private seven-day `claim-<report-key>` artifact before invoking the runner or SMTP. Its minimal body contains only the report identity, `claimed` state, and timestamp. The upload is mandatory: upload failure is a hard failure and the runner does not start. The claim remains after a pre-send crash or a process loss after SMTP acceptance, so a later run cannot silently resend.

An ambiguous QQ delivery or `operator_action_required` result writes a private seven-day `in-doubt-<report-key>` marker and blocks later production delivery when no sent marker exists. First verify QQ delivery directly. If delivery is confirmed, manually dispatch the workflow with the same mode, `reconcile_sent=true`, and `test_email=false`. This narrowly defined action writes a sent marker and audit-safe diagnostic only; it never runs the report runner or sends email. The local CLI reconciliation command does not resolve GitHub cloud markers.

For an unresolved claim, check QQ mailbox delivery and relevant delivery logs first. Only after confirming no message was delivered may an authorized repository operator manually delete the corresponding `claim-<report-key>` artifact in the Actions UI, record that verification, and manually re-run the same mode. If delivery occurred or remains uncertain, preserve the claim as the no-send barrier; when an ambiguous runner result is available, its `in-doubt-<report-key>` marker is the durable replacement. Do not auto-clear or auto-retry claims.

## Privacy and retention

The private production report artifact is named `report-<report-key>` and can contain portfolio analysis; a test email uses the separate `test-report-<report-key>` name and cannot obscure production prior-report lookup. Both are available only to repository users with Actions artifact access. Postmarket checks production artifacts newest-first, validates the redacted premarket state, and skips invalid candidates until it finds a valid one; otherwise it records `prior_premarket_report_unavailable`. The diagnostic artifact is redacted and contains only report state, module statuses, source timestamps, and warning codes. Workflow logs and the concise summary do not print the report HTML, report text, mailbox values, portfolio values, credentials, or raw model responses. All artifacts expire after seven days.

If an authorization code, mailbox, portfolio value, or AI key leaks, revoke or rotate it immediately in its provider, replace the encrypted GitHub Secret, audit workflow runs and repository history, and then send a new manual test email. Never reuse a QQ login password as an SMTP authorization code.
