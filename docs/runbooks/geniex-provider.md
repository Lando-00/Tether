# GenieX External Provider — Operator Runbook

> **Status**: Implemented, live-hardware validated, and selected by the
> committed configuration as the default model provider.

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

### Models

The shipped default is:

```
bartowski/Qwen_Qwen3-8B-GGUF:Q4_0     # ~4.5 GB on disk
```

A smaller fallback is also useful to keep cached:

```
unsloth/Qwen3-1.7B-GGUF:Q4_0          # ~1.0 GB on disk
```

Model IDs follow the pattern `org/repo:quantization` and must be specified
exactly as returned by `/v1/models`.

Pull a model with:

```powershell
geniex --skip-update --data-dir <model-store>\geniex pull "<org>/<repo>:Q4_0" --model-type llm
```

#### Q4_0 is not a preference — it is the NPU requirement

Only **`Q4_0`** is accelerated by the Hexagon NPU in llama.cpp. `Q8_0`, `F16`
and the mixed K-quants (`Q4_K_M`, `Q5_K_M`, …) all fall back to GPU or CPU.

This has a practical consequence when shopping for models: **many popular GGUF
repos ship no `Q4_0` at all.** `unsloth/Qwen3-8B-GGUF`, `Qwen/Qwen3-8B-GGUF` and
`ggml-org/Qwen3-8B-GGUF` are all `Q4_K_M`-only, which is why the default points
at the `bartowski` repack. Always confirm a `*-Q4_0.gguf` asset exists before
pulling, e.g.:

```powershell
(Invoke-RestMethod "https://huggingface.co/api/models/<org>/<repo>/tree/main") |
  Where-Object { $_.path -match 'Q4_0\.gguf$' } |
  Select-Object path, @{n='GB';e={[math]::Round($_.lfs.size/1GB,2)}}
```

#### Memory: do not keep two large models resident

Measured on a 16 GB Snapdragon X Elite: the 8B is stable on its own
(~1.3 s/response warm, 5/5 clean runs), but **loading the 1.7B and the 8B into
the same server process crashed it repeatedly.** The failure is not a clean
error — output first degrades into token repetition (`quickly quickly
quickly …`), then the server dies and subsequent requests fail with
connection-refused.

Guidance:

* Keep one large model in play at a time. Switching models within the
  `--keepalive` window is what triggers the dual-load.
* A shorter `--keepalive` releases the previous model sooner.
* `context_window` in Tether's config must not exceed the server's `--nctx`
  (default 4096); raising it costs KV-cache memory on top of the weights.
* This machine already runs at ~20 GB committed against 15.6 GB physical, so a
  12B/14B `Q4_0` (6.4 GB / 8.0 GB) is likely to be worse, not better.

### Tool Calling

GenieX does **not** expose native tool-calling APIs (`tools`, `tool_choice`
parameters are not supported). Tether uses **marker-only tools**: the system
prompt instructs the model to emit `<<function_call>> {...}` text which
`SlidingParser` detects in the token stream — identical to MLC's approach.

### Key Request Fields

| Field | Placement | Notes |
|-------|-----------|-------|
| `enable_think` | **Top-level** (not nested) | Set `false` to suppress `<think>` blocks |
| `max_tokens` | Top-level | Currently ignored by GenieX; an 8-token request produced 243 tokens in validation |
| `temperature` | Top-level | Respected |
| `top_p`, `top_k`, `min_p`, `presence_penalty` | Top-level | Accepted (HTTP 200); Tether does not currently send them — see "Degenerate output" below |
| `stream` | Top-level | `true` for SSE streaming |

### Response Metadata Caveats

- `id` — always `""` (empty string; use client-side correlation)
- `model` — always `""` in responses (never echoes requested model)
- `created` — always `0`
- `finish_reason` — always `""` in streaming (detect end via `data:[DONE]`)
- `usage` — **zeros in streaming mode**; populated only in non-stream responses
- SSE framing: `data:{json}\n\n` (no space after `data:`)
- `max_tokens` — accepted but not enforced by the validated GenieX release

### Readiness and external-server health

`GET /api/v1/readyz` is **not** a liveness probe for the GenieX server.
Readiness answers whether Tether itself can serve requests. Provider health
reporting is deliberately cheap, synchronous, and network-free, and GenieX
falls back to its configured model id before the first successful inventory
read. Consequently, `/readyz` can report:

```json
{"ready": true, "providers": {"geniex": {"healthy": true}}}
```

while the external server is stopped. This is correct by design.
`providers.<id>.healthy` flips to `false` only after an inventory read has
actually failed. Probe GenieX itself with `GET <base_url>/v1/`.

### Unreachable-server behavior

Unreachability surfaces differently depending on when Tether learns about it:

| Timing | Client-visible result |
|---|---|
| Provider is already known unhealthy during routing | Request fails before streaming with HTTP 503: `Provider '<id>' is currently unavailable…` |
| Provider was healthy, but the server dies after startup | Headers and `message_start` have already been sent with HTTP 200; the stream emits `error(error_type="TransientProviderError", is_fatal=false)`, then `message_stop(stop_reason="error")` |

HTTP 503 is impossible after response headers have been sent. Streaming
clients must handle typed `error` events rather than checking only the HTTP
status. The measured connection-refused path failed in **2.05 seconds**,
matching `connect_timeout_seconds: 2`. The current error text says
`GenieX server timeout` even when the underlying transport error is connection
refused.

Tether does not attempt to start GenieX automatically or retry indefinitely.

---

## Starting the Server

### Model-store convention

The repository refers only to `./models/geniex`. The actual GenieX data store
should live outside the clone, for example `<model-store>\geniex`, with
`aihub\` and `models\<org>\<repo>\` beneath it. Link that store into the
repository at `models\geniex`. This keeps large, shared model assets out of
Git while giving scripts and documentation one stable project-relative path.
The repository's `.gitignore` excludes `/models/`.

Create the recommended privilege-free directory junction from the project
root:

```powershell
New-Item -ItemType Junction -Path 'models\geniex' -Target '<model-store>\geniex'
Get-Item .\models\geniex | Select-Object FullName, LinkType, Target
git check-ignore -v models/geniex
```

Expected: `LinkType` is `Junction`, traversal through `models\geniex` reaches
the external store, and Git reports the `/models/` ignore rule. A directory
symbolic link is an alternative for contributors with Developer Mode enabled
or an elevated shell; `New-Item -ItemType SymbolicLink` otherwise fails.

### Environment choice

- Use a **native ARM64 Python 3.12 stdlib venv** for GenieX-only Tether work.
  GenieX runs out of process, so this environment does not need the MLC wheels.
- Use an **x64 Python 3.12 environment under Prism** when developing or testing
  the MLC provider. Qualcomm's CodeLinaro MLC artifacts are published only as
  `cp312-cp312-win_amd64` wheels and cannot be installed into native ARM64
  Python.

The installed Miniconda is x64 (`platform: win-64`,
`__archspec=x86_64`) and cannot create a native ARM64 environment. Create the
GenieX environment with the ARM64 Python 3.12 interpreter from python.org,
outside the repository:

```powershell
<arm64-python-3.12>\python.exe -m venv <venv-path>
& <venv-path>\Scripts\Activate.ps1
```

The validated interpreter reports
`3.12.10 [MSC v.1943 64 bit (ARM64)]` and `platform.machine() == "ARM64"`.
For automation, call `<venv-path>\Scripts\python.exe` directly because
activation does not persist across one-shot shell invocations.

`uvicorn[standard]` is not installable in a bare Windows ARM64 venv:
`httptools` has no `win_arm64` wheel and attempts a Visual C++ source build.
Install the server dependencies without `httptools`:

```powershell
<venv-path>\Scripts\python.exe -m pip install -e ".[cli,brave,dev]" `
  "fastapi>=0.117.0,<0.118.0" "uvicorn>=0.37.0,<0.38.0" `
  watchfiles websockets pyyaml python-dotenv colorama
```

Uvicorn then uses its pure-Python `h11` HTTP parser. That performance
difference is immaterial for this single-user local service. `tzdata` is a
declared Windows dependency because Windows has no system IANA timezone
database; a pre-existing environment created before that dependency was added
may need `python -m pip install tzdata`.

### GenieX CLI

The GenieX CLI installs to `%LOCALAPPDATA%\GenieX CLI\geniex.exe` and is
normally exposed on `PATH` as `geniex`. The validated release is **v0.3.17**,
with Hexagon HTP v73 and v81 backends. Confirm the active executable path
before starting the server:

```powershell
Get-Command geniex
```

Launch manually using the dev helper:

```powershell
.\scripts\dev\run_geniex_server.ps1 `
  -DataDir (Resolve-Path .\models\geniex).Path `
  -BindHost 127.0.0.1
```

The helper is foreground-only and blocks its terminal; run it in a dedicated
window or PowerShell background job.

To start GenieX *and* the Tether service *and* the CLI in one step, use the
root launcher instead — it reads the base URL from settings, waits for
`GET <base>/v1/`, and adopts an already-running server rather than starting a
second one. See [`one-command-launch.md`](./one-command-launch.md):

```powershell
.\tether.ps1
```

Or invoke GenieX directly:

```powershell
geniex --skip-update --data-dir <model-store>\geniex serve `
  --host 127.0.0.1:18181 --compute npu --keepalive 300
```

The names belong to different layers:

- `GENIEX_DATADIR` is the environment variable read by the **GenieX CLI**
  itself (no second underscore).
- `-DataDir` is the parameter accepted by
  `scripts/dev/run_geniex_server.ps1`; the helper passes it to GenieX's global
  `--data-dir` flag.
- `GENIEX_DATA_DIR` is only the helper script's optional convenience fallback.
  It is not the environment variable consumed by GenieX.

Verify health:

```powershell
Invoke-RestMethod http://127.0.0.1:18181/v1/
# Expected: "GenieX-CLI is running"
Invoke-RestMethod http://127.0.0.1:18181/v1/models
# Expected model: bartowski/Qwen_Qwen3-8B-GGUF:Q4_0
```

The project default bind is `127.0.0.1:18181`; `GET /v1/` is the canonical
health check. The validated store already contains the advertised model, so
no `geniex pull` is required for this baseline.

Startup may emit a benign `SoX is not installed` warning. A
`pkg_resources is deprecated` `UserWarning` may also surface from yoyo; this
is expected and is why Tether retains the `setuptools<81` compatibility
ceiling.

---

## Tether Configuration

GenieX is configured via `model_registry` + `default_model_provider` in
`src/tether/config/default.yml` and is the committed default. MLC remains in
the registry as an alternative provider.

```yaml
providers:
  default_model_provider: "geniex"
  model_registry:
    geniex:
      impl: "tether.providers.geniex.provider.GenieXProvider"
      args:
        base_url: "http://127.0.0.1:18181"  # Server root, without /v1
        model_id: "bartowski/Qwen_Qwen3-8B-GGUF:Q4_0"
        connect_timeout_seconds: 2
        timeout_seconds: 600
    mlc:
      impl: "tether.providers.mlc.provider.MLCProvider"
      args:
        device: "auto"
        max_tokens: 1024
        marker_only_tools: true
```

The provider always sends top-level `enable_think: false` and never sends
native `tools`, `tool_choice`, or `functions` fields. These are provider
contract guarantees, not configuration options.

### Selecting GenieX from the CLI

Choose the provider explicitly, then provide a model that it advertises:

```powershell
tether-cli --provider geniex --model "bartowski/Qwen_Qwen3-8B-GGUF:Q4_0"
```

Use `\providers` to inspect provider health and `\models` to choose a model
within the current provider. Tether rejects a model that belongs to another
provider rather than silently switching backends. API callers may omit
`provider_id` only when exactly one healthy provider advertises the model;
duplicate model IDs require an explicit provider selection.

Environment variables used by GenieX, the helper, and live tests:

| Variable | Purpose | Default |
|----------|---------|---------|
| `GENIEX_BASE_URL` | Live-test server base URL | *(none — required for live tests)* |
| `GENIEX_MODEL_ID` | Live-test model id | *(none — required for live tests)* |
| `GENIEX_DATADIR` | GenieX CLI data/model directory | GenieX-managed default |
| `GENIEX_DATA_DIR` | Optional `run_geniex_server.ps1` convenience fallback | *(none)* |

For Tether configuration overrides, use the standard nested setting:
`TETHER__PROVIDERS__MODEL_REGISTRY__GENIEX__ARGS__BASE_URL`.

---

## Running Hardware-Gated Tests

Live provider tests require a running GenieX server and are gated behind both
the `hardware` and `geniex` markers. Both are excluded by default. Set both
environment variables; with only `GENIEX_BASE_URL`, the tests skip silently:

```powershell
# Start server in one terminal
.\scripts\dev\run_geniex_server.ps1 -DataDir (Resolve-Path .\models\geniex).Path

# Run tests in another terminal
$env:GENIEX_BASE_URL = "http://127.0.0.1:18181"
$env:GENIEX_MODEL_ID = "bartowski/Qwen_Qwen3-8B-GGUF:Q4_0"
python -m pytest -m "hardware and geniex" tests/hardware/test_geniex_live.py -v
```

Tests verify:
- Warm-up health and model validation
- Typed and legacy SSE streaming
- Mid-stream generator closure
- Marker-only prompt generation smoke coverage
- Provider metadata and lifecycle

The shipped configuration passes **10** live GenieX tests. End-to-end NPU
validation also produced this successful turn:

```text
message_start
→ tool_call(time, {timezone: Europe/Dublin, format: human})
→ tool_result(status: ok)
→ text_delta…
→ message_stop(complete)
```

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
| Output exceeds requested `max_tokens` | GenieX currently ignores the field | Bound client timeouts/output handling; do not rely on this field as a hard cap |
| `SoX is not installed` at startup | Optional audio utility absent | Benign for text inference |
| `pkg_resources is deprecated` from yoyo | Expected compatibility warning | Keep `setuptools<81`; do not remove the ceiling |
| Innocuous prompt returns no text and `tool_loop_exhausted` | Model repeatedly retries failed `web_search`; commonly triggered when `BRAVE_API_KEY` is absent | Set `BRAVE_API_KEY`, or disable research mode and `web_search` together |

### Degenerate output (collapsed generation)

**Symptom.** A reply starts as normal prose, then locks onto a short sequence
and repeats it, usually cycling between a few such sequences:

```
Yes, using an alias like `cddev终端全端looplooploop…缆缆缆港口港口港口…衣衣衣衣…注意注意…
```

Left alone it keeps going, so the operator ends up interrupting the client.

**Rate.** Roughly **8% of turns** on `Qwen3-8B:Q4_0`, measured by replaying one
fixed conversation 69 times.

**What it is not.** All three of the obvious explanations were tested and
ruled out:

| Hypothesis | Test | Result |
|---|---|---|
| Context overflow | Failing conversation was ~600 tokens against a 4096 window | Ruled out |
| Unconstrained sampling tail | 3 request variants, interleaved round-robin | 2/23 without `top_p`/`top_k`/`min_p`/`presence_penalty` vs 1/23 with — indistinguishable |
| `enable_think: false` mishandling | Arm with the flag omitted | Same rate (and 4-6x slower, since `<think>` blocks are then generated) |

Failures were spread evenly through the run and appeared in every arm, so this
is nondeterministic corruption inside the inference stack. **No request
parameter avoids it.** Do not spend time re-tuning sampling; that ground has
been covered.

> Because sampling parameters make no measurable difference, Tether does not
> send `top_p` / `top_k` / `min_p` / `presence_penalty`. The server accepts
> them (HTTP 200) if you want to experiment.

**Mitigation.** Tether detects the collapse and aborts the stream with a
`TransientProviderError` rather than relaying garbage
(`tether.providers.geniex.degeneracy`). The turn surfaces as an `error` event
and **retrying usually succeeds**. Detection keys purely on repetition
periodicity, never on script or language, so replies in Chinese, Korean or
Russian are unaffected.

Tether also bounds total output client-side, because the validated GenieX
release accepts `max_tokens` but does not enforce it.

Look for these in the server log:

```
geniex.stream.degenerate    # collapse detected, stream aborted
geniex.stream.output_cap    # client-side output bound reached
```

**If the rate climbs well above ~8%**, suspect the host rather than the model:
free memory (an 8B at Q4_0 needs roughly 5 GB resident, and this class of
machine has 16 GB), and confirm only one model is resident — GenieX becomes
unstable with two.

### Silent context eviction (system prompt lost in long chats)

**Symptom.** Nothing visible. Long conversations quietly stop using tools, and
the model appears to "forget" how it is supposed to behave. The request still
returns HTTP 200 with fluent text, so nothing upstream notices.

**Cause.** GenieX does not reject an over-long prompt — it evicts the front of
the context and carries on. Measured with a sentinel instruction planted in the
system prompt (`--nctx 4096`, `Qwen3-8B:Q4_0`):

| `usage.prompt_tokens` | System prompt |
|---|---|
| 277 | honoured |
| 2329 | honoured |
| 4489 | **lost** |
| 6757 | **lost** |

The front is where the system prompt lives. Because GenieX ignores `tools=`,
that system prompt is the **only** statement of the `<<function_call>>`
convention — so once it is evicted, the model still sees the per-turn tool
roster appended at the end but no longer knows the syntax for calling anything.
Tool calling degrades in exactly the long conversations where it is most useful.

> A prompt of ~9000 estimated tokens still answered a simple question
> correctly, so overflow is not a hard cliff and cannot be detected by watching
> for garbage output. Only the sentinel test reveals it.

**Mitigation.** Tether trims the *middle* of the conversation before sending,
so the provider never has to evict the front
(`tether.protocol.orchestration.context_budget`). System messages and the most
recent turns are always kept; the oldest non-system turns are dropped first.

Look for this in the server log:

```
Context budget: dropped N old message(s) to fit 4096-token window
```

Raising `--nctx` on the server and `context_window` in `default.yml` together
buys more room, at the cost of KV-cache memory. They must be raised together:
`context_window` is what Tether budgets against.

### Web-search configuration trap

The shipped configuration enables `web_search`. Without `BRAVE_API_KEY`, a
model can repeat the same failed tool call until
`limits.max_tool_loops` (default **5**) is exhausted. This was observed with
Qwen3-1.7B on the prompt `Write one short sentence about the sea.`: five
identical failing searches, no text, then
`message_stop(stop_reason="tool_loop_exhausted")`.

This is a general repeated-failed-tool-call pattern, not a GenieX-specific
failure, although a small model is especially prone to it. Two changes since
have largely defused it — the per-turn tool roster (so a disabled tool stops
being advertised) and the move to an 8B default, which now answers that prompt
directly without reaching for a tool. The underlying trap still applies to any
sufficiently small model. Do **not** remove
only `web_search` from `tools.enabled`: research mode is registered by
default, and the engine fails at boot if research is present without
`web_search` (ADR-0020 §D6). The valid fixes are:

1. Set `BRAVE_API_KEY`; or
2. Disable the research orchestrator and `web_search` together.

---

## Design Principles

1. **Tether is a client only** — it connects to GenieX but never manages it.
2. **Per-request degradation** — surface 503 before streaming when
   unavailability is already known; otherwise emit a typed in-stream error.
3. **Marker-only tools** — no native tool protocol; reuse `SlidingParser`.
4. **Config-driven selection** — GenieX is the committed default; callers can
   select another registered provider explicitly.
5. **No hard-coded paths** — use environment variables for all machine-specific paths.
