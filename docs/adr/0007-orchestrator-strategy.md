# ADR-0007: Orchestrator strategy ABC (Seam B) — `ChattyAgent` default

- **Status**: Accepted (Phase 5 of refactor)
- **Date**: 2026-05 (Phase 5)
- **Synthesis citation**: §3.5, §12.3 (Seam B), §4 Phase 5 (steps 56–58 + §12.7 new todos)

## Context

A6's earlier proposal kept the orchestrator as a single concrete class on the grounds of
"no abstraction without ≥2 plausible impls in 12 months." The research-briefing integration
(§12) surfaced a second concrete impl: a `NotebookOrchestrator` implementing the
research-mode atomic-fact context-management pattern from
`docs/research/06_context_strategies.md`. With two real impls, the ABC earns its weight.
At the same time, the orchestrator must not become a sans-IO state machine — that would
re-fragment cancellation/shutdown ownership. Loop-limit and tool-error policies must be
explicit, configurable enums, not implicit behavior.

## Decision

Adopt an **`Orchestrator` ABC** (Seam B) with `ChattyAgentOrchestrator` as the default
strategy and a stub `NotebookOrchestrator` reserved for research mode:

- **`Orchestrator` ABC** in `src/tether/core/interfaces.py`:
  ```python
  class Orchestrator(ABC):
      @abstractmethod
      async def run(self, *, session_id, prompt, model_name,
                    tools: Mapping[str, Tool],
                    cancel_token: Optional[CancelToken] = None,
                    ) -> AsyncIterator[WireEvent]: ...
  ```
- **`ChattyAgentOrchestrator`** (`src/tether/protocol/orchestration/chatty.py`): renamed
  from today's ReAct loop. Thin stateful class with named seams (`_seed_history`,
  `_run_one_turn_until_tool_or_end`, `_dispatch_tools`, `_persist_partial`,
  `_classify_outcome`).
- **`NotebookOrchestrator`** (`src/tether/protocol/orchestration/notebook.py`): stub, raises
  `NotImplementedError` from `run()`. Implementation lands in a follow-up session.
- **`OrchestratorConfig`** frozen dataclass carries `max_tool_loops`, `tool_timeout_sec`,
  `save_thinking`, `include_thinking_in_history`, plus three policy enums (below). Built
  once at the composition root; the orchestrator does NOT call `load_settings()`.
- **Cancellation contract**: `try/finally` wraps the body. On cancel: close provider stream
  → cancel in-flight tool task with 250 ms grace → persist partial assistant text (200 ms
  write timeout) → finalize parser → emit `MessageStop(stop_reason="cancelled" |
  "client_disconnect")` (per ADR-0006). `CancelToken` Protocol (§11 R7) lets the API stay
  loop-implementation-agnostic.
- **Loop-limit policy**: `EMIT_LIMIT_EVENT` default — orchestrator emits
  `loop_limit_reached` and the client renders truncation. Alternative
  `RUN_FINAL_TURN_NO_TOOLS` available; one-line config flip if user disagrees.
- **Tool-error policy**: `FEED_BACK_TO_MODEL` default. Tool errors become a
  `tool_result(status="error", ...)` and the loop continues. Old `BREAK_LOOP` available
  for tests. This is a wire-visible behavior change vs pre-refactor.
- **Strategy registry on `Settings.orchestrator`**: `{default: "chat", registry: {chat:
  <impl>, research: <impl>}}`. Engine resolves per-request. `mode: Literal["chat","research"]
  = "chat"` field on `/api/v1/chat/stream`; `"research"` returns 501; unknown returns 400.
  No `mode="auto"` in v1.
- **`HistoryStrategy` Protocol dropped** for now (§11 R13): orchestrator calls a private
  `_compact_history_if_needed()` no-op. Swap for an injectable strategy if/when compaction
  earns its weight.
- **`GenerationService` dissolves** into `Engine` + `Orchestrator` (no third class).

## Consequences

### Positive
- Two distinct conversation patterns (chat ReAct vs research notebook) can coexist behind
  one Engine API; clients select via a request field.
- Cancellation is centralized in one `try/finally`; no scattered cleanup.
- Policies are explicit enums, not magic strings or substring greps.

### Negative
- The ABC adds one indirection to read for newcomers. Mitigated by clear naming
  (`ChattyAgentOrchestrator`).
- Wire-visible policy change (`FEED_BACK_TO_MODEL`) requires golden-stream test updates
  and a documented change for downstream clients.

### Trade-offs accepted
- We pay one extra abstraction to keep Notebook-mode pluggable; without the ≥2-impl rule
  this would be premature, but Seam B has it.

## Alternatives considered

- **Keep concrete Orchestrator** — rejected post-§12 once Notebook mode became concrete.
- **Mode strings + `if/elif` inside one orchestrator** — rejected: the two flows do not
  share enough structure (atomic-fact accumulation vs ReAct tool loop) to be an internal
  switch.
- **Sans-IO state machine** — rejected per §3.5: refactors cancellation/shutdown ownership
  out of one place; a thin stateful class wins on clarity.

## References

- `files/investigations/_synthesis.md` §3.5, §4 Phase 5, §11 R7, R13, §12.3, §12.7
- `docs/research/06_context_strategies.md` (Notebook context strategy)
- `src/tether/core/interfaces.py::Orchestrator`
- `src/tether/protocol/orchestration/{chatty.py, notebook.py}`
