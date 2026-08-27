# China Trading Calendar Fallback Design

## Goal

Keep A-share reports operational after the bundled XSHG calendar data ends,
without treating an unverified weekday as a trading day.

## Design

The local `exchange_calendars` XSHG calendar remains the primary source. When it
cannot be initialized or queried for a date, the runner loads the existing
AkShare China trading-date dataset and accepts only dates explicitly returned by
that source. Missing or malformed fallback data remains a hard failure.

For post-market reports, the fallback also requires the known 15:00 Shanghai
close to have passed. The runner reports `report_data_incomplete` separately
from `calendar_unavailable` so an early manual run can be retried later without
changing configuration.

## Verification

Session tests cover fallback trading-day resolution, non-trading days, and the
post-close completeness guard. Runner tests cover the distinct safe error code.
