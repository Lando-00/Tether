# Design Notes

In-flight or completed design notes that are not (yet) ratified ADRs. Use this section for
proposals, exploration write-ups, and design docs that may evolve before locking into an ADR.

## Contents

| Doc | Status | Summary |
|---|---|---|
| [`telemetry-and-observability-plan.md`](./telemetry-and-observability-plan.md) | Plan — not implemented | Whether OpenTelemetry is worth it for a local-first single-user app, a `VERBOSE`/`TRACE` log-level ladder, and the deferred `async_span` (M4) seam. Recommends a local turn-timeline first, OTel opt-in only. |
| [`fact-based-orchestration-default.md`](./fact-based-orchestration-default.md) | Implemented | Turn triage + `AutoOrchestrator`: the fact-based (Notebook) loop is now the default mode, answering conversational turns directly and researching only when a turn needs external evidence. |
