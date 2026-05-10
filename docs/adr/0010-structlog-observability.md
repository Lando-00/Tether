# ADR-0010: `structlog` + `RequestId` middleware + optional OpenTelemetry adapter

- **Status**: Accepted (Phase 7 of refactor)
- **Date**: 2026-05 (Phase 7)
- **Synthesis citation**: §3.7, §4 Phase 7 (steps 67–76), §11 R4

## Context

The pre-refactor logging story was 39 ad-hoc `print()` sites and 5 INFO log calls that
dumped full prompts/args/results — a privacy issue for a single-user app that handles
personal data (and a future blocker once connectors carry WhatsApp/Gmail content).
Correlation across `request_id` → `session_id` → `turn_id` → `tool_call_id` was impossible
because none of those IDs were threaded into log records consistently. There was no audit
trail for tool execution. Operators (read: the user) had no debug endpoint to replay a
turn's events.

## Decision

Adopt **`structlog` + stdlib bridge** with explicit correlation flow, opt-in OpenTelemetry,
and a redacted audit log:

- **`structlog` + stdlib bridge** in `src/tether/runtime/logging.py::configure_logging
  (settings)`. One small dep; `contextvars`-based bind; flat JSON-line output. Replaces
  `core/logging.py`.
- **Correlation flow**: `request_id` → `session_id` → `turn_id` → `tool_call_id`. Generated
  at the highest layer that knows them; bound via
  `structlog.contextvars.bind_contextvars()` in middleware / orchestrator / tool-runner.
- **`RequestId` FastAPI middleware** assigns/propagates `X-Request-ID`, binds the value to
  contextvars for the duration of the request.
- **Redaction**: one regex set in `src/tether/core/log_redaction.py`, used by both the
  `structlog` processor AND tests. Masks API keys, emails, FS paths, phone numbers;
  truncates content fields >200 chars at INFO. `caplog`-based tests assert no PII or API
  keys leak at INFO.
- **Sink**: stderr default; optional rotating JSONL at `platformdirs.user_log_dir(
  "Tether")` (per §11 R4 — moved off CWD-relative `data/logs/`). CWD path is still a
  valid explicit override.
- **Tool spans**: `tool.start`, `tool.end`, `tool.error` with `args_redacted` +
  `args_size_bytes`. **Provider streaming spans** wrap MLC stream calls; provider accepts
  caller `request_id`.
- **Audit log**: `tool_audit` table (B5 columns: `args_sha256`, `capabilities`, `status`,
  `error_kind`, `duration_ms`, `correlation_id`). Append-only; stores `args_sha256` (not
  raw args) by default; `Settings.security.audit_log.store_args=true` opt-in for debug.
  Cancelled tool calls record `status='cancelled'` / `error_kind='cancelled'` per §11 R3.
- **Debug endpoint**: `GET /api/v1/debug/turns/{session_id}/{turn_id}` returns the redacted
  timeline from the `turn_timeline` view + `tool_calls` (cf. ADR-0008).
- **Replace 39 `print()` sites** with `logger.*`; demote 5 leaky INFO sites to DEBUG +
  redaction-only.
- **OpenTelemetry**: opt-in via `Settings.observability.otel.enabled=false` default. When
  on, manual spans for `chat.turn`, `provider.stream`, `tool.execute`, plus FastAPI +
  `httpx` auto-instrumentation. Skip SQLite auto-instrumentation.
- **No Sentry / external error reporter by default**. Seam exists for users who want it.
- **`thinking_text` redaction** happens at log time only, not at write time (B5 OQ #4
  resolved). Persisted thinking is a `messages.role='thinking'` row (cf. ADR-0008).

## Consequences

### Positive
- Every log line carries the full correlation chain — bisecting a stuck turn becomes a
  one-`grep` task.
- PII redaction is enforced by tests, not by reviewer vigilance.
- Audit log gives the user a tamper-evident ledger of every tool call without leaking
  raw arguments.
- OTel adapter present but off — operators who want spans get them with one config flip.

### Negative
- `structlog` is one more dependency. We accept it: the alternative is reinventing it on
  top of `logging`.
- Contextvars binding requires discipline: a missed `bind_contextvars` in a new code
  path = missing correlation IDs in those logs.

### Trade-offs accepted
- Default-off `store_args=false` means debug repro of tool-arg-bug requires either an
  opt-in flip or a local debugger session. Privacy wins.

## Alternatives considered

- **Stdlib `logging` + custom JSONFormatter** — rejected: contextvars binding, processor
  pipeline, and key remapping are exactly what `structlog` already solves.
- **Always-on OTel** — rejected: adds startup cost, dependency surface, and a network
  exporter for a local-first app. Opt-in is correct.
- **Sentry default** — rejected: external error reporter for a single-user app violates
  the "no telemetry" baseline.

## References

- `files/investigations/_synthesis.md` §3.7, §4 Phase 7 steps 67–76, §11 R3, R4
- `src/tether/runtime/logging.py`, `src/tether/core/log_redaction.py`
- `src/tether/adapters/http/middleware/{request_id,redaction}.py`
- ADR-0008 (`tool_audit` schema, `turn_timeline` view), ADR-0006 (correlation IDs on wire)
