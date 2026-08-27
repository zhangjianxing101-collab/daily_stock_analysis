# Collaborative Report Workflow Diagnostics Design

## Goal

Make a failed collaborative A-share report actionable without exposing secrets,
portfolio data, raw provider responses, or exception text.

## Design

The report runner will include its existing safe `error_code` in its public JSON
result. The GitHub Actions workflow will copy that code into its diagnostic
manifest when no report manifest exists.

Before invoking the runner, the workflow will run a minimal import preflight.
It emits only one of these fixed codes: `runtime_dependency_unavailable`,
`runtime_python_incompatible`, or `runtime_bootstrap_failed`. It does not write
or upload the original exception text.

Normal report failures continue to use their existing codes, such as
`configuration_invalid` and provider-specific module warnings. Production
delivery, sent-marker, reconciliation, and email behavior are unchanged.

## Verification

Workflow contract tests will assert that the safe error code is retained in the
runner output and diagnostic manifest, and that raw stderr is not uploaded.
