# Developer scripts

Diagnostic and debugging utilities. Not part of the installed `tether`
package — invoke directly from a checkout.

- `cli_chat.py` — interactive chat REPL against the running service
- `run_debug.{py,bat,ps1}` — local launcher with debug logging
- `show_tool_schemas.py` — print all registered tool JSON schemas

## Post-refactor smoke tests

These scripts exercise the live MLC provider and validate the end-to-end
flow. They each load a real GPU model so the first run pays a one-time
OpenCL kernel JIT cost (~30 s engine init + ~60–180 s first prefill).

- `library_smoke.py` — Full library-mode smoke: builds `Engine` from
  settings, creates a session, streams one short turn through the
  chatty orchestrator + parser + MLC provider, validates the v2
  envelope.

- `tool_call_smoke.py` — Full tool-call round-trip smoke: asks the
  time tool, validates `ToolCall` + `ToolResult(status='ok')` events
  appear in the stream and the model produces a natural-language
  answer from the result.

- `provider_warm_then_tools.py` — Calls `MLCProvider.stream()` directly
  (skipping the orchestrator) twice in one process: first no-tools,
  then with one tool schema. Useful when investigating provider-level
  issues without the parser/orchestrator in the loop.

- `mlc_tools_isolated.py` — Calls `AsyncMLCEngine.chat.completions.create()`
  directly (skipping Tether entirely). Useful as a "control" when the
  higher layers misbehave.

## Resolved: ≥3-tool deadlock (was blocking p3-p6 of the test plan)

Passing `tools=[…]` + `tool_choice="auto"` to MLC's CodeLinaro build
with the Qwen3 `use_function_calling` conv template **deadlocked the
async stream at ≥3 schemas** — the generator opened but
`async for response in stream_generator` yielded zero chunks, with the
process at ~4 % CPU (not JIT-compiling).

**Resolution (commit `<the fix commit>`)**: `MLCProvider` now defaults
to `marker_only_tools=True`, suppressing `tools=`/`tool_choice=` at
the engine boundary. Tether's `SlidingParser` detects tool calls from
the `<<function_call>>` text marker, so MLC's structured-tool path
was unused regardless — dropping it loses nothing and unblocks
multi-tool generation. Diagnostic-mode `marker_only_tools=False`
re-enables the native path for upstream comparison work.

The fix is covered by `tests/unit/test_mlc_marker_only_tools.py`
(4 tests) and proven end-to-end by `tool_call_smoke.py`, which
produces a full `ToolCall` → `ToolResult` → text-reply round-trip.
