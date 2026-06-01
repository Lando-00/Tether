# Phase 9.7 W3-A — Research heartbeat latency

## Change
- `src/tether/protocol/orchestration/notebook.py`: lowered `_HEARTBEAT_INTERVAL_SEC` from `5.0` to `2.0` and refreshed the surrounding rationale comment to note the Phase 9.7 W3-A cooperative-cancel review.

## Non-changes (per scope)
- Did NOT add `ResearchSettings.progress_heartbeat_sec`.
- Tests reference the module-level constant only (monkeypatched); no test asserted the literal `5.0`, so no test updates required.
- Did not touch the intentional dirty files (`src/tether/config/default.yml`, `body.json`).

## Verification
`pytest -q tests/unit/protocol/orchestration/test_notebook_progress_heartbeat.py tests/unit/protocol/orchestration/test_notebook_external_cancel.py tests/unit/protocol/orchestration/test_notebook_synth_cancel_grace.py`
→ 9 passed, 1 skipped.

## Status
- Fully done.
- No blockers / open questions.
