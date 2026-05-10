# Tether — AI Agent Map

> If you're an AI agent (Copilot CLI, Codex, Claude Code, Cursor, etc.) operating
> on this repo, start here.

## TL;DR

- **What**: Single-user, local-first FastAPI service for streaming chat completions
  from on-device LLMs (MLC-LLM on Snapdragon X Elite Adreno GPU) with
  function calling and persistent sessions.
- **Architecture map**: [`docs/architecture.md`](./docs/architecture.md) — start here for the layered overview, request lifecycle, and extension seams.
- **Binding contract**: `files/investigations/_synthesis.md` (in session-state) — the locked refactor plan; never relitigate decisions ratified there.
- **Live execution status**: `RESUME.md` (in session-state) — current phase, in-flight todos, last completed work.

## Operating rules

1. **`legacy/` and `llm_service/` are frozen** — never modify. Reference only.
2. **`src/tether/`** is the canonical package; `tether_service.*` is a deprecation
   alias kept for one release cycle.
3. **Hardware backend** is the Adreno X1 **GPU** via OpenCL. Not the NPU. (NPU
   is reserved for the future `NexaProvider`.)
4. **GC-disabled in MLC shutdown daemon thread** is **load-bearing**. Don't touch
   `_terminate_bounded` or `daemon_thread_call` without reading
   [`docs/runbooks/shutdown-hang-fix-summary.md`](./docs/runbooks/shutdown-hang-fix-summary.md) first.
5. **Python entry point** is always `C:\ProgramData\miniconda3\envs\mlc-venv2\python.exe`
   (the conda env is `mlc-venv2`; native ARM64 Python doesn't have the MLC wheels).
6. **No forbidden features**: autonomous reply, plugin SDK, multi-agent routing,
   scheduled jobs, voice, Canvas, push-based notification.

## Conventions enforced by tests/lint

- All settings are typed Pydantic `StrictModel` (extra=forbid, frozen).
- All tools subclass `BaseTool` and use `@tool(name=...)` from `tether.tools.registration`.
- All connectors subclass `Connector` (ABC) and tools they expose carry the
  mandatory `{connector_id}_` prefix.
- All wire-protocol events are typed Pydantic models in
  `tether.protocol.events` (v2 NDJSON: `message_start`, `text_delta`,
  `tool_call`, `tool_result`, `message_stop`).

## Sub-agent prompt templates

If you're delegating work to sub-agents (impl + code-review + rubber-duck pattern), reusable prompt scaffolds live in `.github/agents/`:

- `.github/agents/implementation.agent.md` — impl-agent template
- `.github/agents/code-review.agent.md` — code-review-agent template
- `.github/agents/README.md` — usage guide

## Where things live

| You want… | Look at… |
|---|---|
| Architecture overview + diagrams | [`docs/architecture.md`](./docs/architecture.md) |
| ADRs (locked decisions) | [`docs/adr/`](./docs/adr/) |
| Operational runbooks | [`docs/runbooks/`](./docs/runbooks/) |
| Wire protocol / API specs | [`docs/specs/`](./docs/specs/) (post Phase 8 step 91) |
| Tool implementations | [`src/tether/tools/`](./src/tether/tools/) |
| Connector contract (ABC) | [`src/tether/core/interfaces.py`](./src/tether/core/interfaces.py) and [`src/tether/connectors/`](./src/tether/connectors/) |
| Provider implementations | [`src/tether/providers/`](./src/tether/providers/) |
| Configuration loader | [`src/tether/config/`](./src/tether/config/) |
| HTTP entry point | [`src/tether/app/http/api.py`](./src/tether/app/http/api.py) and [`src/tether/app/__main__.py`](./src/tether/app/__main__.py) |

## Quick commands

```powershell
# Run server
C:\ProgramData\miniconda3\envs\mlc-venv2\python.exe -m tether.app

# Run tests (default-on markers only)
C:\ProgramData\miniconda3\envs\mlc-venv2\python.exe -m pytest -q

# Reload editable install after pyproject.toml changes
C:\ProgramData\miniconda3\envs\mlc-venv2\python.exe -m pip install -e . --no-deps --quiet
```
