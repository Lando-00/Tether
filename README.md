# Tether

> Single-user, local-first FastAPI service for streaming chat completions
> from local model providers with function calling and persistent sessions.

Tether uses a provider registry for local inference. The committed default is
**GenieX**, a separately managed OpenAI-compatible server that runs on the
Snapdragon X Elite NPU. Tether also ships providers for
[MLC-LLM](https://github.com/mlc-ai/mlc-llm) on Adreno/OpenCL, Ollama, and a
dependency-free dummy backend. It exposes a streaming NDJSON (or SSE) API with
tool use, SQLite-backed session history, and a config-driven
Model-Context-Protocol (MCP) architecture.

## Status

Active development. See [`docs/refactor/synthesis-2026-05.md`](./docs/refactor/synthesis-2026-05.md)
for the current phase digest (locked decisions, open deferrals, in-flight work)
and [`AGENTS.md`](./AGENTS.md) for AI-agent navigation.

## Quickstart

### Prerequisites

| Requirement | Notes |
|---|---|
| Windows on ARM64, Snapdragon X Elite | Target host class (`X1E80100`) |
| Python 3.12 | Use a native ARM64 python.org venv for GenieX; MLC requires an x64 conda env under Prism |
| GenieX server | Required by the committed default provider; Tether connects to it but never manages it |
| Provider-specific model files | GenieX convention: `./models/geniex` is a junction to an out-of-repo store; MLC uses `TETHER_MODELS_DIR` (default `./models/`) |

### Install

```powershell
# Default GenieX path: create a native ARM64 venv outside the clone
<arm64-python-3.12>\python.exe -m venv <venv-path>
& <venv-path>\Scripts\Activate.ps1

# Install without uvicorn[standard]: httptools has no win_arm64 wheel
<venv-path>\Scripts\python.exe -m pip install -e ".[cli,brave,dev]" `
  "fastapi>=0.117.0,<0.118.0" "uvicorn>=0.37.0,<0.38.0" `
  watchfiles websockets pyyaml python-dotenv colorama
```

Choose an environment for the provider you are developing:

- **Native ARM64 / GenieX-only**: use a stdlib venv, not conda. The installed
  x64 Miniconda cannot create native ARM64 environments. Uvicorn uses its
  pure-Python `h11` parser in this setup.
- **x64 conda under Prism / MLC**: install the Qualcomm CodeLinaro
  `cp312-cp312-win_amd64` wheels separately before the editable install.

`tzdata` is declared on Windows because Windows has no system timezone
database. Pre-existing environments created before that dependency was added
may need `python -m pip install tzdata`.

See [`docs/runbooks/geniex-provider.md`](./docs/runbooks/geniex-provider.md) and
[`docs/runbooks/fresh-env-setup.md`](./docs/runbooks/fresh-env-setup.md).

For a clean MLC-capable validation environment, see
[`docs/runbooks/fresh-env-setup.md`](./docs/runbooks/fresh-env-setup.md) and
run `scripts/setup_fresh_env.ps1`.

For web-search tool support, copy `.env.example` to `.env` and set `BRAVE_API_KEY`.

### Run

Once installed, one command brings up the whole stack — the GenieX server, the
Tether HTTP service, and the CLI:

```powershell
.\tether.ps1
```

It reuses anything already running, waits for each service to become healthy,
drops you into the CLI, and stops only what it started when you leave.

```powershell
.\tether.ps1 status   # health of both services
.\tether.ps1 stop     # stop what the launcher started
.\tether.ps1 logs     # tail background service logs
.\tether.ps1 -NoCli   # headless: services only
```

If your venv lives outside the clone, copy `tether.local.example.ps1` to
`tether.local.ps1` (gitignored) and set `$Python` once. Full details in
[`docs/runbooks/one-command-launch.md`](./docs/runbooks/one-command-launch.md).

To run just the HTTP service:

```powershell
python -m tether.app
# Starts on http://127.0.0.1:8080

# Or, after the provider-appropriate editable install:
tether-server
```

Interactive API docs: `http://127.0.0.1:8080/docs`

### First request

```powershell
# 1. Create a session
$sid = (Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8080/api/v1/sessions `
  -ContentType "application/json" `
  -Body '{}').session_id

# 2. Stream a reply (NDJSON — default)
curl -N -X POST http://127.0.0.1:8080/api/v1/chat/stream `
  -H "Content-Type: application/json" `
  -d "{`"session_id`":`"$sid`",`"model_name`":`"bartowski/Qwen_Qwen3-8B-GGUF:Q4_0`",`"prompt`":`"What time is it?`"}"
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

The `tether-cli` console script is the interactive front-end. It expects the
HTTP service to already be running — use [`.\tether.ps1`](./docs/runbooks/one-command-launch.md)
to start both together.

```powershell
tether-cli                               # default: connects to http://127.0.0.1:8080/api/v1
tether-cli --api-url http://host:port/api/v1   # custom server URL
tether-cli -m Qwen3-4B-q4f16_1-MLC       # skip the model picker
tether-cli --debug                       # show NDJSON event stream
tether-cli --mode research               # start in research-mode orchestrator
```

Inside the chat loop, slash commands:

| Command     | Effect |
|-------------|--------|
| `\tools`    | List registered tools with on/off state (from `GET /api/v1/tools`) |
| `\tools on\|off <name>` | Enable/disable a tool for later turns |
| `\models`   | Switch model mid-chat (no restart needed) |
| `\mode`     | Toggle chat/research orchestrator mode |
| `\chat` · `\research` | Switch directly to a specific orchestrator mode |
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
| `GET`    | `/tools`                          | List registered tools + schemas + `enabled` |
| `POST`   | `/tools/{name}/enabled`           | Enable/disable a tool at runtime |
| `GET`    | `/healthz` · `/readyz`            | Liveness / readiness probes      |

Request body for `/chat/stream`:
`{"session_id": "...", "model_name": "...", "prompt": "..."}`.
`model_name` is required; omitting it returns HTTP 422 with a `missing`
validation error.

All mutating requests must send `Content-Type: application/json`; otherwise
the CSRF-hardening middleware returns HTTP 415 `unsupported_media_type`.
Endpoints with no meaningful payload, including `POST /sessions`, should send
an empty JSON object (`{}`).

### Orchestrator modes

`POST /api/v1/chat/stream` accepts an optional `mode`. **Omit it** to use the
server's configured default (`orchestrator.default`, which ships as `auto`).

| `mode` | Orchestrator | Behaviour |
|---|---|---|
| `auto` *(default)* | `AutoOrchestrator` | Triages each turn: answers conversationally when no external evidence is needed, and runs the full research loop when it is. |
| `chat` | `ChattyAgentOrchestrator` | Legacy tool loop. Appends raw tool results to history. Explicit opt-out. |
| `research` | `NotebookOrchestrator` | Always researches; never downgraded by triage. |

**Why `auto` is the default.** Tether primarily runs small models, and the
fact-based loop keeps each LLM call bounded to `(question, notebook, current tool
result)` instead of accumulating raw tool output — roughly 1.6 k vs 6 k tokens
after five searches ([ADR-0020](./docs/adr/0020-notebook-orchestrator-algorithm.md)).
Triage is what makes that safe as a default: small talk, creative requests and
back-references ("what did I just say?") are answered directly and never reach a
search backend. See
[`docs/design/fact-based-orchestration-default.md`](./docs/design/fact-based-orchestration-default.md).

### Research mode

Research runs a Hanov-style Plan→Search→Extract→Refine→Synthesize loop with
structured fact extraction and a final synthesized answer. It is reached either
by `mode="research"` (always researches) or automatically by `auto` when a turn
needs external evidence.

It is registered by default and requires the `web_search` tool. The shipped
`src/tether/config/default.yml` already contains:

```yaml
orchestrator:
  default: "auto"
  registry:
    auto: "tether.protocol.orchestration.notebook.AutoOrchestrator"
    chat: "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator"
    research: "tether.protocol.orchestration.notebook.NotebookOrchestrator"
tools:
  enabled:
    - web_search  # required by research mode
```

Comment out the `research:` line to disable the mode. The engine fails loud at
boot if research is registered while `web_search` is not enabled.

Set `BRAVE_API_KEY`; `web_search` uses the Brave Search API. Without it, research
turns gather no facts and `notebook_no_facts` carries a `note` explaining why.

```bash
curl -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s","prompt":"Latest research on...", "model_name":"bartowski/Qwen_Qwen3-8B-GGUF:Q4_0", "mode":"research"}'
```

Research preserves the exact text entered in session history. A short,
whole-message correction ending in `*` (for example, `Ireland*`) is applied
transiently to recent context; it does not rewrite that stored input. If the
correction is ambiguous, or a planned query appears to substitute a
near-spelling entity, the turn asks a clarifying question before any model
search or Brave request. The v2 stream is
`message_start` → `notebook_clarification_requested` →
`message_stop(complete)`; the CLI renders this as a clarification panel.

Narrow finite-decimal arithmetic such as `what is 25 + 50` is computed
locally with `Decimal`, not sent to Brave. It becomes a high-confidence
notebook fact with `source_kind: local_deterministic`; a calculation-only
turn skips planning, search, and extraction. In `tether-cli --debug`,
notebook fact output includes this source kind.

Recommended loop bounds depend on how interactive you want the turn to feel:

| Profile | `max_facts` | `max_iterations` | Use when |
|---|---:|---:|---|
| Interactive | 10 | 5 | quick local experiments and CLI/manual smoke tests |
| Default | 40 | 20 | deeper answers when multi-minute latency is acceptable |
| Deep batch | 80 | 40 | overnight/batch research after raising timeouts deliberately |

Set these under `orchestrator.research` in `src/tether/config/default.yml`.
`planner_model`, `extractor_model`, and `synthesizer_model` optionally select
per-phase models. `synth_assume_open_think_models` is an exact list of model
IDs whose synthesis stream begins inside a hidden `<think>` block. The
Pydantic default is empty; the shipped configuration lists the two local
Qwen3-4B IDs. Add an ID only when its template has that behavior: listed
models suppress the hidden preamble and fail the turn with non-fatal
`UnclosedThinkBlock` if it never closes. Non-listed models keep normal
first-token streaming. Accepted limitation: a long hidden preamble from a
non-listed model can still reach the stream before a bare leading close marker
is recognized.

Phase-progress events (`notebook_phase_progress`) emit every ~2 seconds during long
plan/extract/synthesize calls so clients can show progress during MLC cold-loads
or slow searches. If the research loop finishes with zero gathered facts, Tether
emits `notebook_no_facts` before synthesis; synthesis still runs so the model can
plainly say it did not gather enough evidence. If synthesis produces no visible
text at all, the turn ends with a non-fatal `EmptySynthesis` error rather than
reporting success with an empty answer.

`/readyz` also reports `operational_health.notebook_cleanup`, a bounded
diagnostic view of abandoned async-generator cleanup. It is informational
only: it never changes `ready`, resets hardware, or touches MLC shutdown.
For legacy v0 NDJSON, notebook progress events are suppressed and a
clarification is sent as one text event.

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
# Activate the selected environment, or set this to its interpreter:
$python = "<python-executable>"

# Lint
& $python -m ruff check src/tether tests

# Type-check
& $python -m mypy src/tether

# Default-on tests, then docs drift tests
& $python -m pytest -q
& $python -m pytest -q -m docs tests/docs

# Build distributions
& $python -m build
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

#### Enabling and disabling tools at runtime

Tools can be switched on and off without a restart — orchestrators are built per
turn, so a toggle takes effect on the next message.

```powershell
# List tools and their state
Invoke-RestMethod http://127.0.0.1:8080/api/v1/tools | Select-Object name, enabled

# Disable one
Invoke-RestMethod -Method Post http://127.0.0.1:8080/api/v1/tools/web_search/enabled `
  -ContentType 'application/json' -Body '{"enabled":false}'
```

In the CLI: `\tools` lists them with an on/off column, `\tools off <name>` and
`\tools on <name>` toggle.

**Disabling is a context-budget feature, not just a permission.** A disabled tool:

1. disappears from the tool roster shown to the model,
2. cannot be dispatched, and
3. has its past calls and results dropped from the model-facing history
   (`get_history(exclude_tools=...)`) — the rows stay in the database, only the
   model's view is pruned.

All three matter on small models. Marker-only providers (GenieX, and MLC with
`marker_only_tools`) ignore the `tools=` parameter entirely, so the prompt is the
*only* place the model learns which tools exist. The roster is therefore rebuilt
from the enabled set on every turn rather than hard-coded in the system prompt —
otherwise a disabled tool stays advertised and a small model keeps calling it
until the tool loop is exhausted.

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

Local inference integrations include GenieX, [MLC-LLM](https://github.com/mlc-ai/mlc-llm),
and Ollama.
