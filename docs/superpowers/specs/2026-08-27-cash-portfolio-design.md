# Cash Portfolio Support Design

## Goal

Allow a user with no current positions to generate collaborative A-share market
reports without inventing a holding or disabling market analysis.

## Design

`COLLAB_PORTFOLIO_JSON` accepts an empty JSON array as a valid cash portfolio.
The setting remains required, so an omitted, malformed, or non-array value still
fails closed. The portfolio module reports an explicit no-position status. Market
screening, risk limits, backtests, candidate generation, and no-auto-ordering
controls remain unchanged.

## Verification

Settings tests cover `[]`; runner tests verify the no-position portfolio module
is available and does not invoke a position risk evaluator.
