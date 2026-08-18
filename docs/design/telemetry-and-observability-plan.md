# Telemetry & Observability Plan (OTel + log levels)

> **Status**: PLAN ONLY — nothing here is implemented. This document exists to
> decide *whether* to invest in OpenTelemetry, and if so, exactly what to build.
> Supersedes nothing; extends [ADR-0010](../adr/0010-structlog-observability.md).
>
> **Date**: 2026-07 · **Scope**: `src/tether/observability/`, `src/tether/core/logging.py`,
> `src/tether/config/settings.py` (`observability.*`)

---

## 1. Executive answer: is OTel worth it here?

**Yes — but not the way it is currently wired, and not as the primary tool.**

The recommendation is a **three-tier split**:

| Tier | Mechanism | Default | Who it serves |
|---|---|---|---|
| 1 | Structured logs with real levels (structlog, already shipped) | **on** | the single user, day to day |
| 2 | **In-process turn timeline** (local, no OTel dependency) | **on** | "why was that turn slow / wrong?" |
| 3 | OTel traces + GenAI semantic conventions | **off, opt-in** | power users with a collector |

The reasoning, honestly stated: Tether is a **single-user, local-first** app. The
classic OTel value proposition — correlating a request across many services owned
by many teams — does not apply. Nobody is running a Jaeger instance next to their
laptop to debug their own chat client. Making OTel the centrepiece would be
cargo-culting distributed-systems tooling into a desktop application.

**But** two things genuinely argue *for* OTel, and they are why the answer is not
a flat "no":

1. **This is an LLM app, and the ecosystem standardised on `gen_ai.*`.** The
   OpenTelemetry GenAI semantic conventions define spans and attributes for LLM
   inference, tool execution, and agent workflows (`gen_ai.system`,
   `gen_ai.operation.name`, `gen_ai.request.model`, token counts). As of 2026
   they are marked **"development"** rather than stable, but the core span and
   attribute set is widely shipped and practically stable. Adopting that
   *vocabulary* costs nothing even if we never export a single span, and it makes
   Tether legible to any LLM observability tool (Phoenix, Langfuse, Honeycomb)
   the moment someone wants one.
2. **The seam is already half-built and currently misleading.** `otel_adapter.py`
   exists, is gated behind an `experimental_acknowledged` flag, and emits
   *zero-duration point spans at `*.end` time only* — no parent/child, no
   waterfall. Leaving it in this state is worse than either finishing it or
   deleting it, because it looks like tracing and is not.

**The cost that decides the default.** The OTel Python SDK adds roughly **50 MB
RSS** when tracing is enabled, and CPU overhead ranging from a few percent to
much worse under heavy auto-instrumentation. On a Snapdragon X Elite laptop
where an NPU model already dominates the memory budget, spending 50 MB on
telemetry for an audience of one is a bad trade. **Hence: off by default,
forever.** That is not a temporary state pending maturity; it is the correct
end state for this product.

### What we would lose by doing nothing

Today, debugging "why did that turn take 40 seconds?" means grepping JSON lines
and mentally reconstructing an ordering. There is no duration tree. The
`async_span` (M4) module that `docs/architecture.md` refers to **does not exist**
— every "span" site is a bare `logger.info(...)` call. That is the real gap, and
notably **it is fixable without OTel at all**. Tier 2 below is the highest
value-per-effort work in this document.

---

## 2. Where we actually are (verified, not assumed)

| Claim | Reality |
|---|---|
| `structlog` + stdlib bridge | ✅ shipped, `core/logging.py` |
| Redaction on logs *and* tracebacks | ✅ `RedactingFilter` + `_RedactingFormatter` |
| Correlation IDs via contextvars | ✅ `request_id` bound by middleware |
| `runtime/spans.py::async_span` (M4) | ❌ **does not exist** — documented as DEFERRED in ADR-0010 |
| OTel adapter | ⚠️ exists, experimental-gated, **point spans only** |
| Real parent/child tracing | ❌ none |
| Metrics (token/s, TTFT, tool latency) | ❌ none |

### Two configuration bugs found while surveying

1. **Dead setting.** `observability.log_level` (`Literal["DEBUG","INFO","WARNING","ERROR"]`)
   is declared in `Settings` but **never read anywhere**. `configure_logging()`
   reads `observability.logs.level` instead. Two competing knobs, one of which
   silently does nothing. Anyone setting `log_level: DEBUG` today gets no effect.
2. **Untyped sibling.** `observability.logs.level` is a bare `str` with no
   validation, so a typo (`"INFOO"`) silently falls back to `INFO` via
   `getattr(logging, name, logging.INFO)`.

Both must be fixed as part of the levels work in §3 — they are the direct cause
of the "I want proper levels" ask.

---

## 3. Tier 1 — Log levels done properly

The user asked for `debug`, `info`, `verbose`, etc. Python's stdlib has five
levels and no `VERBOSE`, so we define the mapping explicitly rather than
inventing a parallel scheme.

### Proposed level ladder

| Name | Numeric | Meaning in Tether | Example events |
|---|---|---|---|
| `ERROR` | 40 | The turn failed | provider unreachable, tool crashed |
| `WARNING` | 30 | Degraded but continuing | tool-loop exhausted, malformed marker, provider fell back |
| `INFO` | 20 | One line per significant lifecycle step | turn start/stop, tool call + result, model switch |
| `VERBOSE` | 15 | **new** — per-phase and per-decision detail | notebook phase transitions, parser state changes, fact accepted/rejected |
| `DEBUG` | 10 | Developer-level internals | full history assembly, prompt construction, engine cache keys |
| `TRACE` | 5 | **new** — firehose | every provider chunk, every parser feed |

`VERBOSE` and `TRACE` are registered via `logging.addLevelName()` plus a
`structlog` filtering bound logger. This is a well-trodden pattern; the only
discipline required is that the custom levels are registered **before**
`configure_logging()` builds the filtering wrapper.

**Why add levels at all?** Today the practical choice is INFO (too quiet to debug
a bad research turn) or DEBUG (a wall of noise). The notebook orchestrator in
particular has five phases per iteration and up to 20 iterations — that is
exactly the "I want detail but not the firehose" case `VERBOSE` serves.

### Work items (T1)

- **T1.1** Delete the dead `observability.log_level`, or make it the single
  canonical field and delete `logs.level`. **Recommendation: keep `logs.level`**
  (it is the one actually wired) and remove `log_level`, since `logs.*` already
  groups file/console/format. Removing a `StrictModel` field is a breaking config
  change for anyone who set it — but it never did anything, so behaviour cannot
  regress. Note it in the changelog.
- **T1.2** Type `logs.level` as a `Literal[...]` including the new names so a
  typo fails loudly at boot instead of silently degrading.
- **T1.3** Register `VERBOSE`/`TRACE` and expose `logger.verbose(...)` /
  `logger.trace(...)` helpers.
- **T1.4** Audit existing call sites and demote/promote to the new ladder. The
  concrete candidates found: `notebook.phase_start` → VERBOSE;
  `provider.stream.chunk` (already sampled via `provider_chunk_log_sample`) →
  TRACE, at which point the sampling knob can arguably be retired.
- **T1.5** Wire the CLI `--debug` flag to request a server-side level rather than
  only changing client-side rendering.

**Risk**: low. **Value**: high, immediate, benefits the single user daily.

---

## 4. Tier 2 — `async_span` and a local turn timeline (the real win)

This is the part worth building first, and it **has no OTel dependency**.

### T2.1 — Implement `runtime/spans.py::async_span`

The contract ADR-0010 already promised:

```python
@asynccontextmanager
async def async_span(name: str, **attrs) -> AsyncIterator[Span]:
    """Time a block, bind span_id/parent_span_id to contextvars,
    emit <name>.start and <name>.end (+ duration_ms), and record
    exceptions as <name>.error before re-raising."""
```

Key design points:

- **Parent/child via contextvars**, not a global stack — the orchestrator is
  async and concurrent, so a `ContextVar[str | None]` holding the current
  `span_id` is the only correct mechanism. This is also precisely what makes a
  future OTel bridge trivial.
- **Nothing is exported.** Spans are structlog events. Zero new dependencies.
- **Cheap when disabled**: if the effective level is above the span's level,
  skip timing entirely.

### T2.2 — Instrument the real hierarchy

```
chat.turn
├── provider.stream          (per tool-loop iteration)
├── tool.run                 (per tool call)
└── notebook.phase           (research mode: plan/explore/extract/refine/synthesize)
    └── provider.stream
```

Instrumentation sites already exist as bare log calls and only need wrapping:
`chatty.py` (`provider.stream.start` at :795), `tool_runner.py`, and
`notebook.py` (`notebook.phase_start`).

### T2.3 — Surface the timeline locally

There is already a `turn_timeline` **SQL view** and a `tool_audit` table
(ADR-0008) plus a debug endpoint contract in ADR-0010
(`GET /api/v1/debug/turns/{session_id}/{turn_id}`). Extend the timeline with
span durations and render it in the CLI as an indented waterfall:

```
chat.turn                      4820ms
  provider.stream              1180ms
  tool.run  web_search          890ms  (error: no BRAVE_API_KEY)
  provider.stream              2650ms
```

**This gives the user 90% of the value of tracing, locally, with no collector,
no 50 MB, and no network.** For a single-user app, that is the right answer.

---

## 5. Tier 3 — OTel, done correctly, still off by default

Only after T2 exists, because the OTel layer becomes a thin **bridge** over
`async_span` rather than a parallel implementation.

### T3.1 — Replace the point-span adapter

Delete the structlog-event→span translation. Instead, `async_span` optionally
opens a real OTel span on enter and closes it on exit, with correct parenting
from the same contextvar. This removes the `experimental_acknowledged` gate's
reason to exist.

### T3.2 — Adopt GenAI semantic conventions

Map Tether concepts onto the standard so third-party tools understand us:

| Tether | `gen_ai.*` |
|---|---|
| provider kind (`geniex`/`mlc`/`ollama`) | `gen_ai.system` |
| chat turn | `gen_ai.operation.name = "chat"` |
| model id | `gen_ai.request.model` |
| tool call | `gen_ai.operation.name = "execute_tool"`, `gen_ai.tool.name` |
| token counts | `gen_ai.usage.input_tokens` / `.output_tokens` |

**Caveat to document**: these conventions are "development" status, so pin the
convention version in an attribute and expect churn. Also note GenieX reports
**zeroed usage in streaming mode**, so token attributes will be absent or zero
on the default provider — do not build dashboards that assume them.

### T3.3 — Privacy is non-negotiable

Prompts and completions are **personal data** in this app (connectors carry
WhatsApp content). The existing `redact_text` pass over span attributes must be
preserved, and `gen_ai` "capture message content" options must default to
**off**. An opt-in `observability.otel.capture_content` flag, documented as
"sends your prompts to your collector", is the only acceptable shape.

### T3.4 — Metrics (optional, lowest priority)

TTFT, tokens/sec, tool latency histograms, and tool-loop-exhaustion counters are
genuinely interesting for small-model tuning — which matters given the move to
smaller models. But they are only worth it if someone is actually looking at a
dashboard. Defer until requested.

---

## 6. Recommended sequencing

| Step | Work | Depends on | Value |
|---|---|---|---|
| 1 | T1.1–T1.2 fix the level config bugs | — | high / trivial |
| 2 | T1.3–T1.5 VERBOSE/TRACE ladder | 1 | high |
| 3 | T2.1 `async_span` | 2 | high |
| 4 | T2.2 instrument the hierarchy | 3 | high |
| 5 | T2.3 local waterfall view | 4 | high |
| 6 | T3.1 real OTel bridge | 3 | medium (few users) |
| 7 | T3.2 GenAI conventions | 6 | medium |
| 8 | T3.4 metrics | 6 | low, on demand |

Steps 1–5 need **no new dependencies** and should be treated as the actual
project. Steps 6–8 are opt-in polish.

## 7. Explicit non-goals

- **No always-on OTel.** 50 MB RSS and CPU overhead for an audience of one.
- **No external error reporter** (Sentry et al.) — ADR-0010 already rejected
  this and nothing has changed.
- **No auto-instrumentation of SQLite.** Noise; the store is not the bottleneck.
- **No prompt/completion content leaving the machine by default.** Ever.

## 8. Open questions for the maintainer

1. **Remove `observability.log_level` outright, or alias it to `logs.level`?**
   Removal is cleaner; aliasing is kinder to any existing config. It has never
   worked, so removal is safe on behaviour grounds.
2. **Is `TRACE` worth having** given `provider_chunk_log_sample` already exists?
   Possibly `TRACE` replaces that knob entirely.
3. **Do we want the debug endpoint from ADR-0010 built** (`/debug/turns/...`), or
   is a CLI-only waterfall enough? CLI-only is less surface area.
4. **Should `async_span` land before or after the notebook-orchestrator work?**
   Recommendation: **after**, so the new orchestrator is instrumented once, in
   its final shape, rather than being rewritten twice.
