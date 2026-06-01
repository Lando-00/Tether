# Tether

> Single-user, local-first FastAPI service for streaming chat completions
> from on-device LLMs with function calling and persistent sessions.

Tether runs quantised language models locally via [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
on Qualcomm Adreno/OpenCL hardware. It exposes a streaming NDJSON (or SSE) API with
tool-use (function calling), SQLite-backed session history, and a config-driven
Model-Context-Protocol (MCP) architecture. No data leaves your machine.

## Status

Active development. See [`docs/refactor/synthesis-2026-05.md`](./docs/refactor/synthesis-2026-05.md)
for the current phase digest (locked decisions, open deferrals, in-flight work)
and [`AGENTS.md`](./AGENTS.md) for AI-agent navigation.

## Quickstart

### Prerequisites

| Requirement | Notes |
|---|---|
| Snapdragon X Elite | Adreno 740 GPU, OpenCL backend — required for MLC/Adreno runtime |
| Python 3.12 (x64 under Prism) | conda env `mlc-venv2` (current) or `tether` (fresh; see `scripts/setup_fresh_env.ps1`) |
| MLC-LLM runtime | Qualcomm CodeLinaro `2025.06.r1` wheels (see `environment-mlc-venv2.yml` or `environment-tether.yml`) |
| Compiled MLC model | Set `TETHER_MODELS_DIR` to parent directory (defaults to `./models/`) |

### Install

```powershell
conda activate mlc-venv2
pip install -e ".[server,cli,brave]"
```

> **Note:** Qualcomm CodeLinaro MLC-LLM wheels must be installed separately **before**
> the above step. See `environment-mlc-venv2.yml` for version pins and wheel download instructions.

For a clean fresh-env validation pass, see `docs/runbooks/fresh-env-setup.md` and run `scripts/setup_fresh_env.ps1`.

For web-search tool support, copy `.env.example` to `.env` and set `BRAVE_API_KEY`.

### Run

```powershell
python -m tether.app
# Starts on http://127.0.0.1:8080

# Or, after pip install -e ".[server]":
tether-server
```

Interactive API docs: `http://127.0.0.1:8080/docs`

### First request

```powershell
# 1. Create a session
$sid = (Invoke-RestMethod -Method Post http://127.0.0.1:8080/api/v1/sessions).session_id

# 2. Stream a reply (NDJSON — default)
curl -N -X POST http://127.0.0.1:8080/api/v1/chat/stream `
  -H "Content-Type: application/json" `
  -d "{`"session_id`":`"$sid`",`"prompt`":`"What time is it?`"}"
```

The response is a stream of newline-delimited JSON objects. Each has a `type` field:

| Event type      | Meaning                                            |
|-----------------|----------------------------------------------------|
| `message_start` | Turn begins; lists available tools                 |
| `text_delta`    | Partial text token from the model                  |
| `tool_call`     | Model invoked a tool (name + arguments)            |
| `tool_result`   | Tool returned; `status` is `"ok"` or `"error"`     |
| `message_stop`  | Turn complete                                      |

For SSE framing send `Accept: text/event-stream`. See
[`docs/architecture.md`](./docs/architecture.md#2-request-lifecycle-a-single-chat-turn)
for the full event vocabulary.

### CLI

The `tether-cli` console script is the interactive front-end:

```powershell
tether-cli                               # default: connects to http://127.0.0.1:8080/api/v1
tether-cli --api-url http://host:port/api/v1   # custom server URL
tether-cli -m Qwen3-4B-q4f16_1-MLC       # skip the model picker
tether-cli --debug                       # show NDJSON event stream
```

Inside the chat loop, slash commands:

| Command     | Effect |
|-------------|--------|
| `\tools`    | List registered tools (from `GET /api/v1/tools`) |
| `\models`   | Switch model mid-chat (no restart needed) |
| `\menu`     | Back to session management (new / resume / delete) |
| `\thinking` | Toggle the `thinking_delta` (model's `<think>` block) rendering |
| `\exit` · `\quit` | End the chat |

A minimalist backup (plain-`requests`, dependency-light) lives at `scripts/dev/cli_chat.py` — reference impl of the v2 NDJSON wire protocol.

## API Reference (quick)

All routes are under `/api/v1`.

| Method   | Path                              | Description                      |
|----------|-----------------------------------|----------------------------------|
| `POST`   | `/sessions`                       | Create a new session             |
| `GET`    | `/sessions`                       | List all sessions                |
| `GET`    | `/sessions/{id}/messages`         | Retrieve session history         |
| `DELETE` | `/sessions/{id}`                  | Delete one session               |
| `DELETE` | `/sessions`                       | Delete all sessions              |
| `POST`   | `/chat/stream`                    | Stream a chat completion         |
| `GET`    | `/models`                         | List loaded models               |
| `GET`    | `/tools`                          | List registered tools + schemas  |
| `GET`    | `/healthz` · `/readyz`            | Liveness / readiness probes      |

Request body for `/chat/stream`: `{"session_id": "...", "prompt": "..."}`.
`model_name` is optional; defaults to the first model in `default.yml`.

### Research mode

Tether ships an opt-in research mode ([ADR-0020](./docs/adr/0020-notebook-orchestrator-algorithm.md)). When enabled, `mode="research"` on `/api/v1/chat/stream` runs a Hanov-style Plan→Search→Extract→Refine→Synthesize loop with structured fact extraction and a final synthesized answer.

Enable it in `src/tether/config/default.yml`:

```yaml
orchestrator:
  registry:
    chat: "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator"
    research: "tether.protocol.orchestration.notebook.NotebookOrchestrator"
tools:
  enabled:
    - web_search  # required by research mode
```

Set `BRAVE_API_KEY`; `web_search` uses the Brave Search API.

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s","prompt":"Latest research on...", "model_name":"qwen3-4b", "mode":"research"}'
```

Recommended loop bounds depend on how interactive you want the turn to feel:

| Profile | `max_facts` | `max_iterations` | Use when |
|---|---:|---:|---|
| Interactive | 10 | 5 | quick local experiments and CLI/manual smoke tests |
| Default | 40 | 20 | deeper answers when multi-minute latency is acceptable |
| Deep batch | 80 | 40 | overnight/batch research after raising timeouts deliberately |

Set these under `orchestrator.research` in `src/tether/config/default.yml`.
Phase-progress events (`notebook_phase_progress`) emit every ~2 seconds during long
plan/extract/synthesize calls so clients can show progress during MLC cold-loads
or slow searches. If the research loop finishes with zero gathered facts, Tether
emits `notebook_no_facts` before synthesis; synthesis still runs so the model can
plainly say it did not gather enough evidence.

Research synthesis strips normal `<think>...</think>` blocks from user-visible
`text_delta` events. A long hidden preamble before a bare-leading `</think>`
remains a documented model-template edge case because buffering enough to solve it
regresses first-token streaming latency; see `fu-research-thinkstripper-long-bare-leading`.

Research mode is single-tool for v1 (`web_search` only). Multi-tool research is a future seam.

## Where to go next

| You want…                         | Read…                                             |
|-----------------------------------|---------------------------------------------------|
| Architecture overview             | [`docs/architecture.md`](./docs/architecture.md)  |
| Design decision records           | [`docs/adr/`](./docs/adr/)                        |
| Operational runbooks              | [`docs/runbooks/`](./docs/runbooks/)              |
| AI-agent navigation               | [`AGENTS.md`](./AGENTS.md)                        |
| Locked decisions / phase digest   | [`docs/refactor/synthesis-2026-05.md`](./docs/refactor/synthesis-2026-05.md) |

## Development

```powershell
conda activate mlc-venv2

# Run the full test suite
python -m pytest -q

# Type-check
mypy src/tether/

# Lint
ruff check src/tether/
```

### Built-in tools

| Tool name    | Description                            | Config required      |
|--------------|----------------------------------------|----------------------|
| `time`       | Current date and time                  | none                 |
| `weather`    | Current conditions via open-meteo      | none                 |
| `forecast`   | Multi-day forecast via open-meteo      | none                 |
| `web_search` | Brave Search API web search            | `BRAVE_API_KEY` env  |

To add a custom tool: inherit `BaseTool` in `src/tether/tools/`, register it under
`tools.registry` and `tools.enabled` in `src/tether/config/default.yml`, restart.
See [`docs/architecture.md`](./docs/architecture.md) for the tool-calling flow.

For AI coding agent conventions see [`AGENTS.md`](./AGENTS.md) and
`.github/copilot-instructions.md`.

### Connectors

Connectors extend Tether with personal-data integrations (inbound read + outbound send).
They register their own tools in the `ToolRegistry` under a mandatory `{connector_id}_` prefix
so they appear as regular tools in the chat loop.

#### WhatsApp connector

The WhatsApp connector links a **personal WhatsApp account** (WhatsApp Web protocol, powered
by [neonize](https://github.com/krypton-byte/neonize)) to Tether's connector framework. Once
linked, nine `whatsapp_*` tools become available in every chat session. Enable it with:

```powershell
pip install -e ".[whatsapp]"
# Then uncomment the `connectors:` block in src/tether/config/default.yml
```

**Pairing flow**: run `tether-cli connect whatsapp` — a QR code is printed to the terminal;
scan it in WhatsApp → Settings → Linked Devices. The connector moves to `running` state and
tools activate immediately. To unlink: `tether-cli logout whatsapp` (deletes local credentials,
does not affect your WhatsApp account).

#### WhatsApp tools

| Tool | Description |
|---|---|
| `whatsapp_prepare_send` | Build a text draft (no send) |
| `whatsapp_confirm_send` | Dispatch a draft after explicit user confirmation |
| `whatsapp_list_unread` | List unread inbound events from Tether's local inbox |
| `whatsapp_get_thread` | Retrieve recent messages with a peer (reads local inbox since first connect) |
| `whatsapp_inbox_mark_seen` | Mark Tether-local inbox events as seen (does not affect WhatsApp UI) |
| `whatsapp_mark_platform_read` | Send WhatsApp read receipts (blue checkmarks visible to other party) |
| `whatsapp_send_media` | Build a media draft from a local file path (image/video/audio/document) |
| `whatsapp_get_contacts` | Search contact list by display name or E.164 phone number |
| `whatsapp_resolve_contact` | Resolve a name / E.164 / JID to canonical WhatsApp JID |

Send tools follow a **draft + confirm** two-phase pattern: `*_prepare_send` / `*_send_media`
return a draft id without touching the network; `whatsapp_confirm_send` only dispatches after
the user has explicitly affirmed in their last message (gated by `ConfirmIntentClassifier`).
Accidental sends are structurally impossible.

Design details: [ADR-0018](./docs/adr/0018-whatsapp-connector-library-and-adapter.md)
(neonize backend, `WhatsAppClientAdapter` seam) and
[ADR-0019](./docs/adr/0019-confirm-intent-classifier-seam.md) (`ConfirmIntentClassifier` ABC).

## License

MIT — see `LICENSE`.

## Acknowledgments

Local inference powered by [MLC-LLM](https://github.com/mlc-ai/mlc-llm) and the
Qualcomm CodeLinaro Adreno OpenCL backend.
