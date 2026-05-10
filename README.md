# Tether

> Single-user, local-first FastAPI service for streaming chat completions
> from on-device LLMs with function calling and persistent sessions.

Tether runs quantised language models locally via [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
on Qualcomm Adreno/OpenCL hardware. It exposes a streaming NDJSON (or SSE) API with
tool-use (function calling), SQLite-backed session history, and a config-driven
Model-Context-Protocol (MCP) architecture. No data leaves your machine.

## Status

Active development — Phase 8 refactor underway. See [`RESUME.md`](./RESUME.md) for the
current work-in-progress state and [`AGENTS.md`](./AGENTS.md) for AI-agent navigation.

## Quickstart

### Prerequisites

| Requirement | Notes |
|---|---|
| Snapdragon X Elite | Adreno 740 GPU, OpenCL backend — required for MLC/Adreno runtime |
| Python 3.12 | via `conda` environment named `mlc-venv2` |
| MLC-LLM runtime | Qualcomm CodeLinaro wheels (see `environment.yml`) |
| Compiled MLC model | Set `TETHER_MODELS_DIR` to parent directory (falls back to `dist/`) |

### Install

```powershell
conda activate mlc-venv2
pip install -e ".[server,cli,brave]"
```

> **Note:** Qualcomm CodeLinaro MLC-LLM wheels must be installed separately **before**
> the above step. See `environment.yml` for version pins and wheel download instructions.

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

```powershell
tether-cli   # interactive session (installed by pip install -e ".[cli]")
```

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
| `GET`    | `/health`                         | Liveness / readiness probe       |

Request body for `/chat/stream`: `{"session_id": "...", "prompt": "..."}`.
`model_name` is optional; defaults to the first model in `default.yml`.

## Where to go next

| You want…                         | Read…                                             |
|-----------------------------------|---------------------------------------------------|
| Architecture overview             | [`docs/architecture.md`](./docs/architecture.md)  |
| Design decision records           | [`docs/adr/`](./docs/adr/)                        |
| Operational runbooks              | [`docs/runbooks/`](./docs/runbooks/)              |
| AI-agent navigation               | [`AGENTS.md`](./AGENTS.md)                        |
| Current task state / resume point | [`RESUME.md`](./RESUME.md)                        |

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

## License

MIT — see `LICENSE`.

## Acknowledgments

Local inference powered by [MLC-LLM](https://github.com/mlc-ai/mlc-llm) and the
Qualcomm CodeLinaro Adreno OpenCL backend.
