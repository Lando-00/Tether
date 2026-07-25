# GenieX External Provider — Operator Runbook

> **Status**: Implemented as an opt-in provider; live hardware validation is separate.
> MLC remains the default GPU provider. GenieX is opt-in via configuration.

---

## Overview

GenieX is a **separately managed** OpenAI-compatible inference server that runs
Qualcomm-optimized GGUF models on the NPU (Hexagon HTP) via the GenieX CLI.

**Tether never starts, stops, downloads models for, or upgrades GenieX.**
The server must be launched manually by the operator before Tether can use it.

### Relationship to retired Direct Genie

The earlier `GenieProvider` (ADR-0021, `genie-t2t-run.exe` subprocess) is
**retired**. The new direction is an external HTTP server (`geniex serve`) that
exposes an OpenAI-compatible `/v1` API. Tether connects to it as any other
HTTP model backend.

---

## Server Contract

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/` | GET | Health check — returns `200 "GenieX-CLI is running"` |
| `/v1/models` | GET | Lists known models (note: listed ≠ usable; see caveats) |
| `/v1/chat/completions` | POST | Chat completions (stream and non-stream) |

**Base URL convention**: `http://<host>:<port>` (default: `http://127.0.0.1:18181`).
All model endpoints live under `/v1`.

### Initial Supported Model

```
unsloth/Qwen3-1.7B-GGUF:Q4_0
```

Model IDs follow the pattern `org/repo:quantization` and must be specified
exactly as returned by `/v1/models`.

### Tool Calling

GenieX does **not** expose native tool-calling APIs (`tools`, `tool_choice`
parameters are not supported). Tether uses **marker-only tools**: the system
prompt instructs the model to emit `<<function_call>> {...}` text which
`SlidingParser` detects in the token stream — identical to MLC's approach.

### Key Request Fields

| Field | Placement | Notes |
|-------|-----------|-------|
| `enable_think` | **Top-level** (not nested) | Set `false` to suppress `<think>` blocks |
| `max_tokens` | Top-level | Respected |
| `temperature` | Top-level | Respected |
| `stream` | Top-level | `true` for SSE streaming |

### Response Metadata Caveats

- `id` — always `""` (empty string; use client-side correlation)
- `model` — always `""` in responses (never echoes requested model)
- `created` — always `0`
- `finish_reason` — always `""` in streaming (detect end via `data:[DONE]`)
- `usage` — **zeros in streaming mode**; populated only in non-stream responses
- SSE framing: `data:{json}\n\n` (no space after `data:`)

### Degradation: 503 Behavior

When the GenieX server is **not running**, the provider receives a TCP
connection-refused error (`httpx.ConnectError`). The provider should:

1. Map this to HTTP 503 (Service Unavailable) in Tether's response.
2. Include a clear error message: the external GenieX server is not reachable.
3. **Not** attempt to start the server automatically.
4. **Not** retry indefinitely — fail fast after a brief connect timeout.

---

## Starting the Server

Launch manually using the dev helper:

```powershell
.\scripts\dev\run_geniex_server.ps1 -DataDir $env:GENIEX_DATA_DIR
```

Or directly:

```powershell
$env:GENIEX_DATA_DIR = "<your-data-directory>"
geniex serve --host 127.0.0.1 --port 18181 --compute npu --keep-alive 5m
```

Verify health:

```powershell
Invoke-RestMethod http://127.0.0.1:18181/v1/
# Expected: "GenieX-CLI is running"
```

---

## Tether Configuration

GenieX is configured via `model_registry` + `default_model_provider` in
`src/tether/config/default.yml`. The provider is **opt-in**; MLC remains default.

```yaml
providers:
  default_model_provider: "mlc"          # ← MLC stays default
  model_registry:
    mlc:
      impl: "tether.providers.mlc.provider.MLCProvider"
      args:
        device: "auto"
        max_tokens: 1024
        marker_only_tools: true
    # geniex:                             # ← Uncomment to enable
    #   impl: "tether.providers.geniex.provider.GenieXProvider"
    #   args:
    #     base_url: "http://127.0.0.1:18181"  # Server root, without /v1
    #     model_id: "unsloth/Qwen3-1.7B-GGUF:Q4_0"
    #     connect_timeout_seconds: 2
    #     timeout_seconds: 30
```

The provider always sends top-level `enable_think: false` and never sends
native `tools`, `tool_choice`, or `functions` fields. These are provider
contract guarantees, not configuration options.

Environment variables used by the helper and live tests:

| Variable | Purpose | Default |
|----------|---------|---------|
| `GENIEX_BASE_URL` | Live-test server base URL | *(none — required for live tests)* |
| `GENIEX_DATA_DIR` | Server data/model directory | *(none — required)* |

For Tether configuration overrides, use the standard nested setting:
`TETHER__PROVIDERS__MODEL_REGISTRY__GENIEX__ARGS__BASE_URL`.

---

## Running Hardware-Gated Tests

Live provider tests require a running GenieX server and are gated behind the
`hardware` marker:

```powershell
# Start server in one terminal
.\scripts\dev\run_geniex_server.ps1 -DataDir $env:GENIEX_DATA_DIR

# Run tests in another terminal
python -m pytest -m hardware tests/hardware/ -v
```

Tests verify:
- Warm-up health and model validation
- Typed and legacy SSE streaming
- Mid-stream generator closure
- Marker-only prompt generation smoke coverage
- Provider metadata and lifecycle

Mocked unit/integration tests separately verify exact request fields, SSE
framing, transport errors, degraded startup, and unavailable-provider 503
routing without contacting a real server.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `httpx.ConnectError` / connection refused | Server not running | Start GenieX manually |
| 400 "model not found in local cache" | Model not downloaded | `geniex pull <model-id>` |
| 500 "quantization not found" | Model listed but not cached | Re-pull with correct quantization |
| Thinking blocks in output | `enable_think` nested or missing | Ensure `enable_think: false` is top-level |
| Empty `model` in response | Known server behavior | Use request model ID; don't trust response |

---

## Design Principles

1. **Tether is a client only** — it connects to GenieX but never manages it.
2. **Fail-fast** — if the server is unreachable, surface 503 immediately.
3. **Marker-only tools** — no native tool protocol; reuse `SlidingParser`.
4. **Config-driven opt-in** — switch providers by changing `default_model_provider`.
5. **No hard-coded paths** — use environment variables for all machine-specific paths.
