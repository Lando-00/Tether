# Tether Domain Context

Tether is a single-user, local-first FastAPI service that exposes streaming
chat completions from local model providers, persistent sessions, function
calling, and a connector framework.

## Core vocabulary

- **Engine**: the library-first composition root used by HTTP, CLI, and other
  adapters.
- **ModelProvider**: the provider boundary implemented by GenieX, MLC, Ollama,
  dummy, and the Nexa compatibility stub.
- **GenieX**: the committed default provider, running as an external
  OpenAI-compatible server on the Snapdragon X Elite NPU. Tether is its client.
- **Tool**: a typed operation registered through the `@tool` decorator and
  executed through `ToolRunner`.
- **Connector**: an integration boundary for inbound data and guarded outbound
  actions; connector tools use the `{connector_id}_` prefix.
- **Orchestrator**: the strategy that drives a chat turn, currently Chatty or
  Notebook.
- **Wire protocol v2**: the typed NDJSON event stream
  (`message_start`, `text_delta`, `tool_call`, `tool_result`, `message_stop`).
- **Session store**: SQLite-backed persistence for chat history and inbox
  events, with tool-call audit records.

## Non-negotiable boundaries

- Preserve the library-first `Engine` composition root.
- Treat `src/tether/` as canonical; `tether_service.*` is a deprecation alias.
- Keep MLC shutdown GC-disabled behaviour intact.
- Keep outbound URL allowlisting and the CSRF, CORS, and TrustedHost middleware
  ordering intact.
- Tether never starts, stops, downloads for, or upgrades the external GenieX
  server.
- Do not add autonomous reply, plugin SDK, multi-agent routing, scheduled-job,
  voice, Canvas, or push-notification features without reopening the locked
  project direction.

## Known config gotcha

- `ObservabilitySettings.log_level` (the top-level `observability.log_level`
  field) is currently a dead/unused field — nothing in `src/tether/` reads it.
  The runtime logging level is actually controlled by
  `observability.logs.level` (`LogsSettings.level`). This is a statement of
  current behavior, not a recommendation to change the code.

## Source of truth

- Architecture and request lifecycle: `docs/architecture.md`
- Locked decisions and current deferrals:
  `docs/refactor/synthesis-2026-05.md`
- Detailed decisions: `docs/adr/`
- Operational procedures: `docs/runbooks/`
- Agent navigation and commands: `AGENTS.md`

When this file conflicts with an ADR or the refactor synthesis, the ADR or
locked synthesis wins; update this context when the vocabulary or boundaries
change.
