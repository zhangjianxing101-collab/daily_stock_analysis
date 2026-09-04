# China Trading Calendar Fallback Design

## Goal

Keep A-share reports operational after the bundled XSHG calendar data ends,
without treating an unverified weekday as a trading day.

## Design

The local `exchange_calendars` XSHG calendar remains the primary source. When it
cannot be initialized or queried for a date, the runner loads the AkShare China
trading-date dataset and accepts only dates explicitly returned by that source.
Missing, malformed, or stale fallback data remains a hard failure: the dataset
must bracket the target date with its earliest and latest explicit sessions
before it can classify the target as a non-trading day. The fallback dataset is cached only per
Shanghai local date, so a long-running process refreshes it on the next day.

AkShare runs in a short-lived subprocess with a 30-second timeout, including
import and decoding time. A timed-out worker is killed and reaped; no background
thread is left waiting on the network. Invalid worker output and worker failures
follow the same safe calendar-unavailable path and cannot trigger email delivery.

For post-market reports, the fallback also requires the known 15:00 Shanghai
close to have passed. The runner reports `report_data_incomplete` separately
from `calendar_unavailable` so an early manual run can be retried later without
changing configuration.

## Verification

Session tests cover fallback trading-day resolution, both coverage bounds,
non-trading days, cache rollover, invalid worker output, and the post-close
completeness guard. Runner tests exercise the real calendar-resolution path
with a timed-out worker and verify that neither data collection nor email runs.
Worker tests use a local fake provider to exercise subprocess execution without
depending on live network availability.

An online loader smoke check on 2026-09-04 returned 8,797 explicit sessions from
1990-12-19 through 2026-12-31. This verifies calendar access only, not stock quote
freshness, GitHub Actions execution, or email delivery.
