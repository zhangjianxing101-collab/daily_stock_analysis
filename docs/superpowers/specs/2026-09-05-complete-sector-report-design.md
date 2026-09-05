# Complete Sector Report Design

## Goal

Expand the post-market collaborative report into a complete decision-first daily report. The first screen must answer market direction, risk level, action stance and whether to wait. The detailed body must add industry and concept-sector analysis while retaining market breadth, candidates, portfolio, sizing, risk checks, gold and global context.

The report remains analysis-only. It does not promise returns or place orders, and every trade-related output requires manual confirmation.

## Confirmed Product Decisions

- Use the hybrid decision layout: concise conclusions first, complete evidence afterward.
- Analyze both industry and concept sectors.
- Show the strongest 10 and weakest 10 sectors for each type.
- Include leaders, breadth, activity, rotation stage, persistence and crowding risk.
- Continue producing the report when one source or module fails. Show the reason, source time and coverage instead of filling missing values with zero or stale data.
- Scope this feature to the post-market report. Premarket reports may reference the most recent completed post-market sector state but do not run an intraday sector ranking.

## Report Structure

The post-market report is ordered as follows:

1. Decision summary: market direction, risk level, action stance, wait/watch decision and key risks.
2. Market overview: major indices, advancing/declining/flat counts, limit-up/limit-down counts when available, turnover, market temperature and style.
3. Sector analysis: industry and concept strongest/weakest tables, rotation interpretation, persistence, crowding and next-session watch list.
4. Trading and risk: short-term candidates, swing candidates, entry/stop/target conditions, portfolio valuation, position sizing and risk-rule results.
5. External context: global indices, gold, copper and oil.
6. Evidence and source health: timestamps, providers, coverage counts and degraded modules.

Email text and HTML use the same module payload. HTML may use responsive tables, but it must not omit rows or evidence that exist in the text report.

## Components

### Sector Snapshot Gateway

Add a bounded gateway that retrieves industry and concept board snapshots independently through the existing AkShare/Eastmoney integration style. The two board types must not share a failure boundary.

Normalize available provider fields into a stable frame:

- `sector_type`: `industry` or `concept`
- `name`: non-empty board name
- `change_pct`: finite daily percentage change
- `advance_count` and `decline_count`: non-negative counts when provided
- `turnover_rate` or `amount`: finite, non-negative activity evidence when provided
- `leader_name`, `leader_code` and `leader_change_pct`: optional leader evidence
- `source_timestamp`, `observed_at` and `source`

Identity, duplicate, numeric and timestamp checks occur before analysis. A row missing `change_pct` cannot enter strongest/weakest rankings. Missing optional evidence does not invalidate another valid row; it lowers coverage and leaves dependent conclusions unavailable.

The existing THS index catalog may cross-check identities when a stable mapping exists. It must not silently replace a conflicting board identity or convert THS traded amount into turnover percentage.

### Sector Analyzer

The analyzer receives a validated current snapshot and an optional previous completed post-market sector state. It has no network or rendering responsibilities.

For each sector type it produces:

- strongest 10, sorted by `change_pct` descending;
- weakest 10, sorted by `change_pct` ascending;
- deterministic tie-breaking by normalized board name;
- breadth percentage when advance and decline counts are both available;
- activity percentile within the same sector type;
- leader evidence;
- rotation stage and persistence classification;
- crowding risk and next-session watch reasons.

The strongest and weakest sets cannot contain the same sector. If fewer than 20 valid sectors exist, the strongest set is filled first, and the weakest set uses the remaining rows. The report discloses the valid-row count and requested coverage.

### Rotation And Persistence

Rotation is based on the current and immediately preceding completed post-market states:

- `first_observation`: no trustworthy previous state exists;
- `new_start`: current top 20, absent from the previous top 20;
- `continuing`: top 20 in both states with no acceleration or divergence condition;
- `accelerating`: top 20 in both states, rank improves by at least five places, and activity does not deteriorate when activity evidence exists;
- `diverging`: positive performance but breadth is below 50%, or required breadth/activity evidence conflicts with the headline move;
- `retreating`: previous top 20 and now negative or outside the current top half.

Persistence is evidence-based:

- `high`: current and previous top 20, breadth at least 55%, and activity is stable or improving;
- `medium`: current top 20 and at least one of previous strength, broad participation or improving activity is confirmed;
- `low`: headline strength is narrow, activity is deteriorating, or the sector is retreating;
- `unavailable`: the previous state or required evidence is missing.

Crowding risk is `high` only when a top-ranked sector has extreme activity but participation is narrow; `medium` when activity is elevated without narrow participation; otherwise `low`. If activity or breadth is unavailable, crowding risk is `unavailable`.

These classifications are descriptive evidence. They cannot independently create a stock candidate. They can annotate or re-rank a stock that already passes deterministic technical screening.

### State Persistence

Post-market preview and delivery bundles include a sanitized sector state containing only normalized sector metrics, report key, source timestamps and classifications. The workflow retrieves the most recent completed post-market state using the existing prior-artifact pattern.

State is accepted only when:

- its report key is an earlier trading session;
- its schema and sector identities validate;
- source timestamps are timezone-aware and no later than the report generation checkpoint;
- it came from a completed report artifact rather than an in-doubt or failed delivery.

Missing or rejected state produces `first_observation`; it never blocks the current report.

### Report Integration

Add independent `industry_sectors` and `concept_sectors` module results. Each module reports `ok`, `partial` or `unavailable`, includes its own warnings and preserves its own source timestamp.

The decision summary derives a market direction only when market breadth is usable. Sector evidence may strengthen or weaken the explanation but cannot override unavailable core market data. The summary's watch decision remains governed by existing portfolio and risk rules.

Candidate rows include the matched industry/concept sector, rotation stage and persistence when available. Missing sector membership does not remove an otherwise valid candidate.

## Failure And Degradation Rules

- Industry failure does not suppress concept analysis; concept failure does not suppress industry analysis.
- Invalid or stale sector rows are excluded and counted.
- A provider exception is converted to a fixed diagnostic code. Raw response text, URLs, credentials and exception messages do not enter reports or artifacts.
- Missing optional fields render as `不可用`; they are never represented as zero.
- Missing previous state renders rotation as `首次观察` and persistence as `不可用`.
- If all sector data is unavailable, the report still contains market, portfolio, candidate, risk, gold and global modules with a clear sector warning.
- A partially populated report is still sent under the user's selected policy, provided existing delivery and expiry gates pass.

## Presentation

The decision summary remains compact and appears before detailed tables. Industry and concept sections each contain:

- strongest 10 table;
- weakest 10 table;
- a short rotation summary;
- next-session watch list with reasons;
- source time and valid/total coverage.

Table columns are rank, sector, change, breadth, activity, leader, leader change, rotation stage, persistence and crowding risk. On narrow screens, tables scroll horizontally inside their section without shrinking text or hiding columns. The text report presents the same fields as readable lines.

## Verification

Required deterministic tests cover:

- normalization, identities, duplicates, invalid numbers and timestamps;
- independent industry/concept failures;
- strongest/weakest ranking, tie-breaking and non-overlap with fewer than 20 rows;
- every rotation transition and persistence/crowding boundary;
- missing previous state and missing optional evidence;
- report module statuses, summary behavior and candidate annotations;
- sector-state schema, prior-state rejection and workflow artifact handling;
- HTML/text parity for all sector rows and source metadata;
- existing no-email preview behavior and prohibition on automatic trading.

Before merge, run the collaborative-report suite, the repository CI gates and a no-email post-market preview on a trading day. If no trading-day preview is available, the PR must state that limitation and must not claim live report completion.

## Out Of Scope

- Intraday sector monitoring or alerts
- Automatic trading or order routing
- User-configurable sector formulas in this iteration
- Long-horizon sector backtesting beyond the immediately previous completed report state
- Fabricated values or implicit reuse of stale sector data
