# ADR-0006: Wire protocol v2 (NDJSON) + parser strategy (Seam D)

- **Status**: Accepted (Phase 5 of refactor)
- **Date**: 2026-05 (Phase 5)
- **Synthesis digest**: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)

## Context

The pre-refactor wire format had several issues: `data: {…}` envelopes around top-level
fields, three different terminal events for tool flows (`tool_started` / `tool_completed` /
`tool_error`), a `tool_marker_detected` event that leaked parser internals, a `done` event
that fired twice per turn (synthesis §6 bug #3), and an `info` event with no schema.
`Engine.chat()` returned `str | List[Dict]` chunks rather than typed objects. The parser was
a global singleton from `Settings.providers.parser`, even though MLC-on-Adreno needs
marker-based parsing while a future OpenAI-compatible provider would want native
tool-call deltas.

## Decision

Adopt **Wire protocol v2** with a Pydantic v2 discriminated union, NDJSON default, and
provider-owned parsers (Seam D):

- **Two-vocabulary architecture**: internal `ParserEvent` (frozen dataclasses, hot path) →
  external `WireEvent` (Pydantic v2 discriminated union). Translation only at
  `Orchestrator._wire(parser_event)`.
- **Wire shape**: top-level fields, no `data: {…}` envelope. `protocol_version: "1.0"` on
  every event. Event types: `message_start`, `message_stop`, `text_delta`, `thinking_delta`,
  `tool_call`, `tool_result`, `error`, `loop_limit_reached`, `hw_reset`. **Cancellation has
  NO `Cancelled` event** (§11 R1) — it surfaces as `MessageStop(stop_reason="cancelled" |
  "client_disconnect")`, optionally preceded by an info-level `Error`.
- **Naming**: `tool_call` (model emitted intent) + `tool_result` (server executed). The old
  `tool_started` / `tool_completed` / `tool_error` triplet is removed. `text` →
  `text_delta`. `done` → `message_stop` (one terminal event per turn, fixes bug #3).
- **Versioning**: embedded in event + `X-Tether-Protocol-Version: 1.0` header.
- **Transport**: NDJSON default; SSE via `Accept` content negotiation (same events, two
  formatters).
- **Library-mode types**: `Engine.chat()` yields **typed `WireEvent` Python objects**, not
  bytes. HTTP transport wrappers turn them into NDJSON / SSE bytes.
- **Discoverability**: `GET /api/v1/protocol/schema` returns Pydantic-derived JSON Schema;
  `GET /api/v1/protocol/example` returns canonical NDJSON recording. Build-time artifact
  `docs/specs/events.schema.json` (Phase 8 step 91) freshness-checked in CI.
- **Parser strategy (Seam D)**: each provider creates its own parser via
  `Provider.create_parser() -> StreamParser`. The orchestrator no longer holds a global
  parser. `MLCProvider` ships `SlidingParser` (marker-based `<<function_call>>`); future
  `NexaProvider` ships an OpenAI-style tool-call-delta parser. The provider-scoped
  `system_prompt` (`providers.mlc.args.system_prompt`) carries the marker-format steering
  with the provider that needs it.
- **Cutover via dual emit + content negotiation** (§11 R18): step 54a ships v2 as opt-in
  via `Accept: application/x-ndjson; version=1.0`; step 54b updates clients/CLI/goldens;
  step 54c flips the default. Legacy names are removed in a follow-up minor version.

## Consequences

### Positive
- Library consumers receive typed events — IDE autocomplete, mypy enforcement,
  no JSON parsing at the boundary.
- Parser becomes a per-turn instance, eliminating cross-turn state leakage and supporting
  heterogeneous provider parsers without refactor.
- One terminal event per turn closes a long-standing replay-correctness bug.
- `/api/v1/protocol/schema` makes the wire self-describing — third-party clients (and AI
  agents) can validate without consulting humans.

### Negative
- Naming cutover is a wire-visible breaking change. Mitigated by dual-emit content
  negotiation + golden-stream tests + CLI client update in the same PR.
- Provider authors must write a parser if their model speaks something other than the
  default marker format.

### Trade-offs accepted
- Two parser implementations (sliding marker + future native-delta) is acceptable cost for
  the heterogeneity Seam D is designed for.

## Alternatives considered

- **Keep one global parser, switch by provider capability flag** — rejected: the parser
  state-machines for marker vs native are fundamentally different; flags hide the seam.
- **OpenAI Chat Completions wire format verbatim** — rejected: SSE-only, no thinking_delta
  semantics, no protocol_version field; we'd reinvent it anyway.
- **Keep `done`/`text`/`tool_started` legacy names** — rejected: the per-turn double-`done`
  bug cannot be fixed without renaming, and `tool_started`/`tool_completed`/`tool_error`
  conflate model intent with server execution.

## References

- Synthesis digest: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
- `src/tether/protocol/wire/events.py` (`WireEvent` discriminated union)
- `src/tether/protocol/parsers/{events.py, sliding.py}`
- `src/tether/providers/base.py::create_parser`
- `docs/specs/events.schema.json` (planned, Phase 8 step 91)
