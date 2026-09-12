# Collaborative A-share Report Delivery

This workflow delivers one private A-share postmarket report at 16:30 China Standard Time on weekdays, with a conditional outlook for the next trading session and independently validated industry and concept analysis. The separate daily-analysis workflow remains available manually but no longer runs on its former 18:00 schedule, avoiding a second scheduled email.

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
When optional position sizing is enabled, any held position without a validated
price suppresses all new-position sizing. Position sizing is disabled by default
for the current market-and-sector research scope.

Production snapshots supplement THS names, volume ratios and turnover percentages
from the project's public Tencent quote source before screening the full universe.
THS traded amount is never interpreted as turnover percentage. Supplements require
exact identities, no future timestamps, finite metrics, and price agreement within
CNY 0.01. A completed-session quote must be stamped at or after 15:00. On an XSHG
trading day before 09:30, an exactly matching current-day quote may prove the prior
session close; the report records that prior close as the data-session identity.
Current-day quotes at or after 09:30 cannot be used to backfill a premarket report. Rejected fields
remain unavailable. Codes outside the supplemental provider's supported exchange
prefixes do not block enrichment for the supported universe. Partial coverage marks screening as partial. For conservative
freshness checks, the combined dataset uses the oldest accepted source timestamp;
one expired source can therefore invalidate the combined snapshot.

After an authoritative postmarket run, the workflow stores the normalized completed-session
A-share snapshot as a private `market-snapshot-YYYY-MM-DD` artifact for seven days. The
archive contains public normalized quote fields and quality counts, never portfolio, mailbox,
credential, or AI-response data. Historical premarket archive loading remains available in
the local runner for compatibility but is not part of scheduled delivery.

Preview runs additionally produce bounded `provider-quality` diagnostics containing
only counts and parsed dates. Raw responses, stderr and credentials are not uploaded.

The market overview derives style only when every quoted stock has a finite,
positive total-market-cap value. It compares the equal-weight returns of the
largest and smallest 30% of the verified snapshot and reports both group averages
beside `大盘占优`, `小盘占优`, or `均衡`. A 0.5 percentage-point spread is required
before declaring either size group dominant.

Post-close limit-up and limit-down counts come from AKShare's documented
Eastmoney daily pools (`stock_zt_pool_em` and `stock_zt_pool_dtgc_em`) for the
report's exact completed session. The report records the source and session time.
It does not infer these counts from percentage-change thresholds because board,
ST, IPO, and price-rounding rules make that approximation unsafe. A provider
failure, malformed response, or session before 15:00 leaves the counts unavailable
and marks the market module partial.

## Industry and concept analysis

The postmarket report reads industry and concept board snapshots independently,
preferring the configured THS catalog and index quote APIs and using the AkShare
Eastmoney adapters only as a recoverable fallback. A failure in one board family
does not suppress the other. Each accepted row requires a unique non-empty board name and a finite daily
change. Breadth is calculated from advancing and declining constituent counts, and
activity is the percentile rank of turnover rate within the same board family.
Leader identity and leader change are supporting evidence. Missing optional evidence
remains `不可用`; it is never converted to zero. Invalid timestamps, duplicate names,
malformed values and rows without a valid change are rejected or disclosed as partial
coverage before any conclusion is produced.

Industry and concept universes are ranked separately by daily change, with board name
as the deterministic tie-breaker. The report shows at most 10 strongest and 10 weakest
boards from each universe, without overlap. The next-session watch list contains only
strong boards whose persistence is `high` or `medium`; it is observation evidence, not
an instruction to trade.

Rotation labels use the current rank, change, breadth and the latest trustworthy prior
postmarket state:

- `first_observation`: no trustworthy prior state exists.
- `new_start`: the board enters the top 20 from outside the prior top 20.
- `accelerating`: it remains in the top 20, improves by at least five ranks and activity does not fall when both activity values are available.
- `diverging`: price change is positive while breadth is below 50%, or activity declines from the prior observation.
- `retreating`: a prior top-20 board turns negative or falls below the current universe midpoint.
- `continuing`: none of the preceding transition rules applies.

Persistence is unavailable without prior activity, current breadth or current activity.
It is `low` when breadth is below 50%, activity falls, or rotation is retreating. It is
`high` when current and prior ranks are both top 20, breadth is at least 55%, and
activity does not fall. Other qualifying current top-20 observations are `medium`;
remaining observations are `low`. Crowding risk is `high` for a top-20 board with
activity at or above the 90th percentile and breadth below 50%, `medium` when activity
is at or above the 75th percentile and breadth is at least 50%, and otherwise `low`.
Crowding is unavailable when breadth or activity is missing.

For rotation comparison, the workflow considers at most five unexpired production
artifacts named `report-YYYY-MM-DD-postmarket`, newest earlier trading date first. It
accepts only a completed `sent`, non-test postmarket manifest whose identity and date
match, then passes a reduced file containing only sector state and required provenance.
Preview, test, failed, current-day, future-dated and malformed artifacts are ignored.
If no valid state is available, both board families remain usable but are labeled as a
first observation and the fixed `prior_postmarket_sector_state_unavailable` warning is
recorded.

Sector context can annotate deterministic technical candidates with an industry,
matched concepts, rotation and persistence. It may reorder candidates only within an
already equal technical/THS score. It cannot create a candidate, change a score,
override data-quality suppression, alter entry/stop/target levels, size a position, or
remove the requirement for manual confirmation.

The report also shows a deduplicated featured watchlist of at most five candidates.
Only candidates carrying explicit sector-leader or sector-membership evidence are
eligible. If fewer than three qualify, the report shows the smaller verified set
instead of filling the list with unsupported names. The full short-term and swing
pools remain visible as the audit trail.

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

Set non-secret Repository variables for the delivery policy and model choices.
`COLLAB_SHORT_LIMIT` and `COLLAB_SWING_LIMIT` default to `5`, and
`COLLAB_SCREEN_PREFILTER` defaults to `120`. `COLLAB_POSITION_SIZING` defaults to
`false`; leave it disabled when the report should analyze only the market, sectors,
and candidates. `COLLAB_CAPITAL_CNY` and `COLLAB_RISK_FRACTION` apply only when
position sizing is explicitly enabled; the default reference capital is CNY 20,000.
Model names are repository variables; the
workflow has safe defaults when a model variable is omitted.

The report's market-news section uses enabled ORZ/NewsNow China-market templates
as discovery-only clues. It diversifies financial sources, keeps Baidu items in a
separate macro-attention count, rejects stale, undated, or future-dated items, and
does not add these clues to technical scores. Aggregator timestamps and repeated
landing-page links are not treated as proof; material facts must be checked against
the original publisher or an official disclosure before any manual decision.

## Manual operation

Use **Run workflow** for a manual `postmarket` report. The workflow serializes scheduled and manual runs while preserving `cancel-in-progress: false`. For a manual test email, set `force` to true when outside its normal window and set `test_email` to true. A test email is marked as a test and never creates a production delivery marker.

Use a production manual run only after verifying the secrets and variables. A normal sent report creates `sent-<report-key>`; a later fresh GitHub job uses that durable marker as the cross-run duplicate guard and passes `--already-sent` to the runner as an external completed identity. Before any production runner call, the workflow queries unexpired `sent-<report-key>`, `in-doubt-<report-key>`, and `claim-<report-key>` artifacts in that order. A sent marker wins; otherwise an in-doubt marker or unresolved claim fails closed before mail delivery and writes only the redacted diagnostic artifact. Test email bypasses all production marker lookup and never creates a production marker. Within a single run or a locally durable output directory, the runner's local delivery ledger is the delivery state machine and a second barrier; it is ephemeral on GitHub-hosted runners and is not cross-run state.

After the production preflight succeeds, the workflow uploads a private seven-day `claim-<report-key>` artifact before invoking the runner or SMTP. Its minimal body contains only the report identity, `claimed` state, and timestamp. The upload is mandatory: upload failure is a hard failure and the runner does not start. The claim remains after a pre-send crash or a process loss after SMTP acceptance, so a later run cannot silently resend.

An ambiguous QQ delivery or `operator_action_required` result writes a private seven-day `in-doubt-<report-key>` marker and blocks later production delivery when no sent marker exists. First verify QQ delivery directly. If delivery is confirmed, manually dispatch the workflow with the same mode, `reconcile_sent=true`, and `test_email=false`. This narrowly defined action writes a sent marker and audit-safe diagnostic only; it never runs the report runner or sends email. The local CLI reconciliation command does not resolve GitHub cloud markers.

For an unresolved claim, check QQ mailbox delivery and relevant delivery logs first. Only after confirming no message was delivered may an authorized repository operator manually delete the corresponding `claim-<report-key>` artifact in the Actions UI, record that verification, and manually re-run the same mode. If delivery occurred or remains uncertain, preserve the claim as the no-send barrier; when an ambiguous runner result is available, its `in-doubt-<report-key>` marker is the durable replacement. Do not auto-clear or auto-retry claims.

## Privacy and retention

The private production report artifact is named `report-<report-key>` and can contain portfolio analysis; a test email uses the separate `test-report-<report-key>` name. Both are available only to repository users with Actions artifact access. Earlier postmarket sector lookup is bounded and writes only the validated sector-state subset to the runner. The diagnostic artifact is redacted and contains only report state, module statuses, source timestamps, and warning codes. Workflow logs and the concise summary do not print the report HTML, report text, mailbox values, portfolio values, credentials, remote errors, or raw model responses. All artifacts expire after seven days.

If an authorization code, mailbox, portfolio value, or AI key leaks, revoke or rotate it immediately in its provider, replace the encrypted GitHub Secret, audit workflow runs and repository history, and then send a new manual test email. Never reuse a QQ login password as an SMTP authorization code.
