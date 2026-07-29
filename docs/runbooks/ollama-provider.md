# Ollama provider runbook

Tether's `OllamaProvider` (ADR-0022) lets you route chat requests to an
[Ollama](https://ollama.com) server running on any machine reachable over HTTP —
most commonly a GPU workstation on your LAN.

**Why Ollama?**
GPU workstations with discrete NVIDIA/AMD cards run Ollama with CUDA/ROCm
acceleration. Tether's MLC provider targets the Snapdragon Adreno NPU/GPU and
doesn't ship x86-64 wheels. Rather than duplicating inference stacks, `OllamaProvider`
delegates to Ollama over its HTTP API so you can use both providers from a single
Tether server instance (ADR-0021 multi-provider registry).

**Recommended hardware**: RTX 4060 Ti 16 GB or comparable GPU.

---

## Table of contents

1. [Overview](#overview)
2. [Installing Ollama on the GPU PC](#installing-ollama-on-the-gpu-pc)
3. [Making Ollama listen on LAN](#making-ollama-listen-on-lan)
4. [Firewall setup](#firewall-setup)
5. [Verifying connectivity](#verifying-connectivity)
6. [Pulling models for 16 GB VRAM](#pulling-models-for-16-gb-vram)
7. [Tether YAML config](#tether-yaml-config)
8. [CLI usage](#cli-usage)
9. [Troubleshooting](#troubleshooting)
10. [Running live tests](#running-live-tests)
11. [Future enhancements](#future-enhancements)

---

## Overview

`OllamaProvider` wraps two Ollama HTTP surfaces behind a single class:

| `api_surface` | Endpoint | Protocol | Features |
|---|---|---|---|
| `"native"` (default) | `POST /api/chat` | NDJSON streaming | `think: true` for reasoning, parsed tool-call dicts |
| `"openai_compat"` | `POST /v1/chat/completions` | SSE streaming | OpenAI-compatible; useful when fronting Ollama with a reverse proxy or talking to vLLM / llama.cpp / LiteLLM |

The surface is selected at construction time via the `api_surface` YAML arg. Both
modes support tool calling; only `"native"` supports the thinking channel
(`thinking_models` whitelist). If `thinking_models` is non-empty and
`api_surface="openai_compat"`, the provider logs a WARNING at startup and never
sends `think: true`.

---

## Installing Ollama on the GPU PC

### Option A — Windows native (recommended)

1. Download the installer from <https://ollama.com/download> and run it.
2. Ollama installs as a Windows service and starts automatically.
3. Verify it is running:
   ```powershell
   curl http://localhost:11434/api/version
   # → {"version":"0.x.y",...}
   ```

### Option B — Docker (Linux / WSL2)

On Linux:
```bash
# Requires NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
docker run -d \
  --gpus=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

On Windows with WSL2 the Docker approach works but the native Windows installer is
simpler and avoids GPU passthrough complexity. Use Docker only if you are already
managing the machine via containers.

---

## Making Ollama listen on LAN

By default Ollama binds only to `127.0.0.1:11434`. To accept connections from your
Snapdragon laptop you need to set `OLLAMA_HOST=0.0.0.0:11434` **before** starting
the Ollama server process.

### Windows (permanent — survives reboots)

```powershell
# Run once in an elevated PowerShell prompt:
setx OLLAMA_HOST "0.0.0.0:11434" /M   # /M = machine-wide (all users)
# Then restart the Ollama service:
Restart-Service Ollama -ErrorAction SilentlyContinue
# If Ollama isn't installed as a service, restart it from the system tray.
```

### Windows (session only — quick test)

```powershell
$env:OLLAMA_HOST = "0.0.0.0:11434"
ollama serve   # keep this terminal open
```

### Linux / macOS

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

> **Security warning — read before continuing.**
>
> Setting `OLLAMA_HOST=0.0.0.0` exposes the Ollama API to **every device on your
> network** (and beyond, if the machine has a public IP). There is no built-in
> authentication on the native API.
>
> Mitigations:
> - **Firewall rule** (§ Firewall setup below): limit port 11434 to your laptop's
>   IP only. Simplest option for a home LAN.
> - **Reverse proxy + Bearer token**: put Nginx or Caddy in front of Ollama, require
>   an `Authorization: Bearer <token>` header, set `api_surface: "openai_compat"` in
>   Tether YAML, and put the token in the env var named by `api_key_env`.
> - **WireGuard / Tailscale**: if you already have a VPN, bind Ollama to the VPN
>   interface IP instead of `0.0.0.0`.

---

## Firewall setup

### Windows (GPU PC)

Open PowerShell **as Administrator** on the GPU PC:

```powershell
# Replace 192.168.0.0/16 with your actual LAN subnet, or with your laptop's
# specific IP (e.g. 192.168.1.42) for tighter control.
New-NetFirewallRule `
  -DisplayName "Ollama LAN" `
  -Direction Inbound `
  -LocalPort 11434 `
  -Protocol TCP `
  -Action Allow `
  -RemoteAddress "192.168.0.0/16"
```

To use a single IP instead of a subnet:

```powershell
New-NetFirewallRule `
  -DisplayName "Ollama - Tether laptop only" `
  -Direction Inbound `
  -LocalPort 11434 `
  -Protocol TCP `
  -Action Allow `
  -RemoteAddress "192.168.1.42"   # ← your laptop's LAN IP
```

### Linux (GPU PC)

```bash
# UFW
sudo ufw allow from 192.168.1.0/24 to any port 11434 proto tcp

# firewalld
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port protocol="tcp" port="11434" accept'
sudo firewall-cmd --reload
```

---

## Verifying connectivity

From your Snapdragon laptop:

```powershell
# Replace <gpu-pc-ip> with the actual LAN IP of the GPU PC.
curl http://<gpu-pc-ip>:11434/api/version
```

Expected response:
```json
{"version":"0.7.1"}
```

If this fails:

| Symptom | Likely cause |
|---|---|
| `Connection refused` | Ollama not running; or `OLLAMA_HOST` still `127.0.0.1` |
| `Connection timed out` | Firewall blocking the port |
| `No route to host` | Different network segment; check routing |

---

## Pulling models for 16 GB VRAM

On the GPU PC (SSH in, or open a terminal directly):

```bash
# Start here — general use, tool calling, thinking mode
ollama pull qwen3:8b        # ~5.2 GB download, ~5.2 GB VRAM at q4

# Larger thinking model — still fits at q4
ollama pull qwen3:14b       # ~9.0 GB download, ~9.0 GB VRAM

# Fast fallback for short prompts or quick tests
ollama pull llama3.2:3b     # ~2.0 GB download, ~2.0 GB VRAM

# Reasoning-focused alternative
ollama pull deepseek-r1:8b  # ~5.0 GB download, ~5.0 GB VRAM
```

**Models to avoid on 16 GB VRAM** (at default q4 quantisation):

| Model | Approx VRAM | Status |
|---|---|---|
| `qwen3:32b` | ~20 GB | Exceeds 16 GB — partial CPU offload only |
| `llama3.3:70b` | ~43 GB | Far too large |
| `qwen3:14b` + another model | ~18 GB | Can co-load with small context; risky |

List what is currently pulled:
```bash
ollama list
```

---

## Tether YAML config

### `api_surface: "native"` — recommended default

```yaml
providers:
  model_registry:

    # Native Ollama API — NDJSON streaming, /api/chat.
    # Supports think:true for reasoning models.
    ollama-gpu:
      impl: "tether.providers.ollama.provider.OllamaProvider"
      args:
        base_url: "http://192.168.1.50:11434"  # ← LAN IP of GPU PC
        api_surface: "native"                   # NDJSON /api/chat (default)
        models:
          - "qwen3:8b"
          - "qwen3:14b"
          - "llama3.2:3b"
          - "deepseek-r1:8b"
        default_model: "qwen3:8b"
        thinking_models:                        # enables think:true for these models
          - "qwen3:8b"
          - "qwen3:14b"
          - "deepseek-r1:8b"
        context_windows:
          "qwen3:8b": 32768
          "qwen3:14b": 32768
          "llama3.2:3b": 131072
          "deepseek-r1:8b": 32768
        discover_at_startup: false              # set true to auto-discover pulled models
        timeout_seconds: 600                    # read timeout — LLM generation is slow
        connect_timeout_seconds: 10             # LAN should connect in <1s; 10s is generous
        keep_alive: null                        # null = Ollama default (5 min idle unload)
        default_options: {}                     # e.g. {temperature: 0.7, top_p: 0.9}

  default_model_provider: "ollama-gpu"
```

When you use `model_registry`, remove the existing `model:` (singular) block — the
two shapes are mutually exclusive (raises `ConfigError` if both are set).

### `api_surface: "openai_compat"` — for reverse-proxy or cross-compatible setups

```yaml
providers:
  model_registry:

    # OpenAI-compatible surface — SSE streaming, /v1/chat/completions.
    # Use when Ollama is behind a Bearer-auth reverse proxy, or when
    # talking to a vLLM / llama.cpp-server / LiteLLM endpoint that
    # speaks the OpenAI API.
    ollama-openai:
      impl: "tether.providers.ollama.provider.OllamaProvider"
      args:
        base_url: "http://192.168.1.50:11434/v1"  # note the /v1 suffix
        api_surface: "openai_compat"
        api_key_env: "OLLAMA_API_KEY"              # env var holding the Bearer token
        models:
          - "qwen3:8b"
        default_model: "qwen3:8b"
        thinking_models: []                        # MUST be empty — openai_compat
                                                   # ignores think:true; provider
                                                   # logs WARNING if non-empty
        timeout_seconds: 600
        connect_timeout_seconds: 10

  default_model_provider: "ollama-openai"
```

Set the API key (if your reverse proxy requires one):
```powershell
$env:OLLAMA_API_KEY = "your-bearer-token"
```

### Running both surfaces simultaneously

You can expose both the MLC on-device model **and** the Ollama GPU model from a
single Tether server:

```yaml
providers:
  model_registry:
    mlc-local:
      impl: "tether.providers.mlc.provider.MLCProvider"
      args:
        device: "auto"
        max_tokens: 1024
        marker_only_tools: true

    ollama-gpu:
      impl: "tether.providers.ollama.provider.OllamaProvider"
      args:
        base_url: "http://192.168.1.50:11434"
        api_surface: "native"
        models: ["qwen3:8b", "qwen3:14b"]
        default_model: "qwen3:8b"
        thinking_models: ["qwen3:8b", "qwen3:14b"]
        context_windows:
          "qwen3:8b": 32768
          "qwen3:14b": 32768
        timeout_seconds: 600

  default_model_provider: "mlc-local"
```

If the GPU PC is unreachable at startup, Tether enters degraded mode: the
`ollama-gpu` provider is marked unhealthy in `/api/v1/readyz`, but `mlc-local`
continues serving requests normally (ADR-0021 `_provider_start_failures`).

### Hardcoding vs environment variable for `base_url`

The Tether YAML loader does not currently support `${VAR}` interpolation.
Options:

1. **Hardcode the IP** — simplest. Change the IP in YAML when the GPU PC gets a new
   lease.
2. **DHCP reservation** — assign a static DHCP lease to the GPU PC's MAC address in
   your router settings. Then the IP is stable and hardcoding is safe.
3. **Hostname** — if your router supports mDNS / local DNS, use
   `http://gpu-pc.local:11434` instead of an IP address.

---

## CLI usage

### Starting with a specific provider

```powershell
# Long form
tether-cli --provider ollama-gpu --model qwen3:8b

# Short form
tether-cli -P ollama-gpu -m qwen3:8b
```

When `--provider` is omitted the server uses `default_model_provider` from YAML.

### REPL slash commands

Inside the `tether-cli` interactive session:

| Command | Effect |
|---|---|
| `\providers` | Lists all configured providers with health status (from `/api/v1/readyz`) |
| `\models` | Shows model selector with provider column; sorted by `(provider_id, id)` |
| `\thinking` | Toggles display of thinking / reasoning content (for models in `thinking_models`) |

### Routing a single request via curl

```bash
# Route to Ollama GPU provider explicitly:
curl -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "prompt": "Explain quantum entanglement in simple terms.",
    "model_name": "qwen3:8b",
    "provider_id": "ollama-gpu"
  }'
```

Omit `provider_id` to use `default_model_provider`. An unknown `provider_id` returns
HTTP 422; a known but unhealthy provider returns HTTP 503.

### Inspecting provider health

```bash
curl http://localhost:8080/api/v1/readyz | python -m json.tool
```

Relevant fields in the response:

```json
{
  "providers": {
    "ollama-gpu": {
      "healthy": true,
      "kind": "ollama",
      "source": "ollama+native://http://192.168.1.50:11434",
      "error": null
    }
  },
  "default_provider_id": "ollama-gpu",
  "provider": true
}
```

`"healthy": false` with a non-null `"error"` means Ollama was unreachable at startup.
Restart Tether after fixing the connectivity issue to re-probe.

### Model details

```bash
curl http://localhost:8080/api/v1/models/details | python -m json.tool
```

Each row includes:
- `"provider_id"` — which backend owns the model
- `"thinking_channel"` — whether `think: true` is enabled for this model
- `"supports_reasoning_effort"` — always `false` for `OllamaProvider` in Phase 1
- `"context_window"` — from `context_windows` config, or 4096 default

---

## Troubleshooting

### Error table

| Error / symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: could not reach Ollama at http://...:11434; is the server running?` | Ollama not running, `OLLAMA_HOST` not `0.0.0.0`, or firewall blocking | Step through: (1) `curl http://<gpu-ip>:11434/api/version` from GPU PC itself; (2) set `OLLAMA_HOST=0.0.0.0:11434` and restart Ollama; (3) check firewall rule on GPU PC |
| `RuntimeError: model 'X' not pulled on Ollama; run ollama pull X` | Model name in `models:` not pulled on the GPU PC | SSH / RDP to GPU PC; run `ollama pull X` |
| HTTP 503 on `/api/v1/chat/stream` with "Provider 'ollama-gpu' unavailable" | Warm-up failed at Tether startup | Check `/api/v1/readyz` for the specific error; restart Tether after fixing connectivity |
| `RuntimeError: Ollama timed out after 600s; increase timeout_seconds in provider config` | Model load + long generation exceeded timeout | Raise `timeout_seconds` in YAML; or pre-warm the model with a tiny request before heavy use |
| First request after a few minutes of idle is slow (~20-30s) | Ollama's default `keep_alive` (5 min) unloaded the model | Set `keep_alive: "60m"` in provider args to keep the model in VRAM; or pre-warm via a scheduled tiny request |
| Thinking / reasoning content not appearing in responses | Model not in `thinking_models` whitelist; or `api_surface: "openai_compat"` (which ignores `think: true`) | Add model to `thinking_models:`; switch to `api_surface: "native"` |
| Tool calls dropped silently; model describes what it would do instead of calling | Ollama version too old (< 0.2.8) for the model's tool-call schema; or model doesn't support structured tools | Run `ollama --version`; update Ollama; confirm model supports tools with `ollama show <model>` |
| `ValueError: OllamaProvider: invalid base_url '...'` at startup | `base_url` failed the outbound URL allowlist check (ADR-0011) | Ensure the URL uses `http://` or `https://`; check `security.outbound_allowlist` in settings if customised |
| `WARNING ollama.thinking_models_ignored_for_openai_compat` | `thinking_models` non-empty with `api_surface: "openai_compat"` | Clear `thinking_models: []` for `openai_compat` entries, or switch to `api_surface: "native"` |
| Connection works from GPU PC but not from laptop | `OLLAMA_HOST` is `127.0.0.1` | See [Making Ollama listen on LAN](#making-ollama-listen-on-lan) |
| Provider shows `healthy: false` immediately after fixing connectivity | Tether cached the failure from startup | Restart Tether; `warm_up` re-probes on each startup |

### Checking Ollama logs on the GPU PC

**Windows** (Ollama native service):
```powershell
# Ollama writes to the Windows Event Log and also to:
Get-Content "$env:LOCALAPPDATA\Ollama\server.log" -Tail 100
```

**Linux / Docker**:
```bash
journalctl -u ollama -n 100 --no-pager   # systemd
docker logs ollama --tail 100            # Docker
```

### Testing the Ollama API directly

```bash
# List pulled models
curl http://<gpu-ip>:11434/api/tags

# Simple non-streaming chat (no Tether involved)
curl -X POST http://<gpu-ip>:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Say hi"}],"stream":false}'

# Check tool-calling support
curl -X POST http://<gpu-ip>:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:8b",
    "messages": [{"role":"user","content":"What is 2+2? Use the calculator tool."}],
    "tools": [{"type":"function","function":{"name":"calculator","description":"Compute arithmetic","parameters":{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}}}],
    "stream": false
  }'
```

---

## Running live tests

The repo ships a live integration test suite gated behind `@pytest.mark.ollama` and
the `OLLAMA_BASE_URL` environment variable. These tests are **skipped by default** in
`pytest -q`.

### Prerequisites

- GPU PC running Ollama with at least one model pulled (default: `qwen3:8b`)
- LAN connectivity verified via `curl http://<gpu-ip>:11434/api/version`

### Running the tests

```powershell
# Point at your GPU PC
$env:OLLAMA_BASE_URL = "http://192.168.1.50:11434"

# Optional: override the test model (default: qwen3:8b)
$env:OLLAMA_LIVE_MODEL = "llama3.2:3b"

# Run all live Ollama tests
C:\ProgramData\miniconda3\envs\mlc-venv2\python.exe -m pytest -m ollama -v

# Or target the file directly
C:\ProgramData\miniconda3\envs\mlc-venv2\python.exe -m pytest tests/hardware/test_ollama_live.py -v
```

### What the tests cover

| Test | Surface | What it checks |
|---|---|---|
| `test_live_version` | Native | `GET /api/version` returns a non-empty dict |
| `test_live_simple_chat_native` | Native | Stream "Say hi" → at least one text chunk within 60s |
| `test_live_simple_chat_openai_compat` | OpenAI compat | Same prompt via `/v1/chat/completions` SSE |
| `test_live_tool_call` | Native | Fake tool schema → at least one `tool_call` chunk emitted |
| `test_live_thinking_output` | Native | Thinking chunk emitted when model is in `thinking_models`; skipped otherwise |
| `test_live_cancellation_closes_stream` | Native | Cancel token set after 500ms → generator exits within 2s |
| `test_live_model_not_found_returns_actionable_error` | Native | Non-existent model → `RuntimeError` with `"ollama pull"` in message |

---

## Future enhancements

The following are out of scope for Phase 13 and tracked as follow-up items:

- **`\pull <model>` slash command** — trigger `ollama pull <model>` from the Tether
  CLI without opening a separate terminal on the GPU PC.
- **`keep_alive` forwarded into `/api/chat` options** — currently accepted by the
  provider constructor but not forwarded as a top-level key in the request body.
  Planned for Phase 14.
- **Reasoning effort mapping** — `OllamaProvider` advertises
  `supports_reasoning_effort=False` in Phase 1. Waiting for Ollama to expose a
  structured reasoning-budget field before wiring this up.
- **Periodic re-probe of remote provider health** — currently `warm_up` is called
  once at Tether startup. A background probe loop would surface GPU PC connectivity
  loss without requiring a Tether restart.
- **`discover_at_startup: true` + `\pull` combo** — when `discover_at_startup` is
  true and a new model is pulled on the GPU PC, a `/refresh` API call could merge
  the newly discovered model without a restart.
