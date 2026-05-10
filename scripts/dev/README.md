# Developer scripts

Diagnostic and debugging utilities. Not part of the installed `tether`
package — invoke directly from a checkout.

- `cli_chat.py` — interactive chat REPL against the running service
- `run_debug.{py,bat,ps1}` — local launcher with debug logging
- `show_tool_schemas.py` — print all registered tool JSON schemas

## Post-refactor smoke tests

These scripts exercise the live MLC provider and validate the end-to-end
flow. They each load a real GPU model so the first run pays a one-time
OpenCL kernel JIT cost (~30 s engine init + ~60–150 s first prefill).

- `library_smoke.py` — Full library-mode smoke: builds `Engine` from
  settings, creates a session, streams one short turn through the
  chatty orchestrator + parser + MLC provider, validates v2 envelope.
  **Currently times out at 600 s when ≥3 tools are enabled** — see
  "Known issues" below.

- `provider_warm_then_tools.py` — Calls `MLCProvider.stream()` directly
  (skipping the orchestrator) twice in one process: first no-tools,
  then with one tool schema. Proves the provider wrapper works.

- `mlc_tools_isolated.py` — Calls `AsyncMLCEngine.chat.completions.create()`
  directly (skipping Tether entirely). Proves MLC itself works with
  tools enabled. Useful as a "control" when the higher layers misbehave.

## Known issues

**`Engine.chat()` deadlocks with ≥3 tool schemas (post-refactor).**
With Qwen3-4B-q4f16_1-MLC on Adreno, `MLCProvider.stream()` opens but
the `async for response in stream_generator` never yields a single
chunk when `tools=[…3+ schemas…]` is passed to MLC.
Symptoms: `chunks_emitted: 0` after 600 s timeout, process at ~4% CPU
(idle, not JIT-compiling). One-tool calls complete in 90 s; the same
shape via `provider_warm_then_tools.py` yields chunks correctly.
Triage scripts above isolate it as upstream of the orchestrator
(`library_smoke.py` reproduces; `mlc_tools_isolated.py` does not).
Mitigation while investigating: disable `web_search` and one other
tool in `tether/config/default.yml::tools.enabled`.
