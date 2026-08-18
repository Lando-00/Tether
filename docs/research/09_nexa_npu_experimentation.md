# NexaSDK NPU Experimentation — Findings (May 2026)

> **Working-directory artifacts:** were staged under an off-repo
> `<workspace>\nexa\` directory while the refactor was in flight; the
> headline doc and the provider skeleton are now committed under
> `docs/research/`. Disk-only artifacts (1.2 GB Granite NPU model,
> 88 MB Nexa CLI install) stay off-repo; the cleanup recipe at the
> bottom of this doc covers them.
> **Hardware:** Surface Pro 11, Snapdragon X Elite, 16 GB.
> **Goal:** Validate NPU execution as a future `NexaProvider` for Tether
> *while the refactor is in flight*, so when ABCs land we have working
> template code and head-to-head numbers ready.
> **Status:** ⚠️ **NPU path blocked by upstream license-server outage.**
> Architecture work delivered. Live NPU benchmarks not possible today.

## TL;DR

1. ✅ Native ARM64 Nexa CLI v0.2.73 installs cleanly on our hardware
   and works at the command-routing level (`pull`, `list`, `version`).
2. ✅ `NexaAI/Granite-4.0-h-350M-NPU` (1.2 GB) downloaded fine —
   HF/S3 model storage is healthy.
3. ❌ **Every NPU inference attempt fails identically with
   `License validation failed: fail to request license host for
   activation` after a 21.5 s timeout.** Confirmed via direct network
   test: `lic.nexa.ai:443` (54.200.56.5) is unreachable from our
   network. Same on multiple networks per upstream issue reports.
4. ❌ Python SDK (`pip install nexaai`) install also broken — different
   symptom (S3 403 on the bridge zip), same root cause (Qualcomm acquired
   Nexa in March 2026; infrastructure migration in progress as of
   2026-05-10).
5. 🟢 **Architecture deliverable IS ready** — the Tether refactor team
   already shipped a `NexaProvider` stub at
   `tether_service/providers/nexa/provider.py` with the right
   `kind`/`capabilities`. Our HTTP-targeted skeleton at
   [`reference/nexa_provider_skeleton.py`](./reference/nexa_provider_skeleton.py)
   shows the methods.

## What works today

### Install — Native ARM64 v0.2.73

Downloaded from
<https://github.com/qualcomm/nexa-sdk/releases/tag/v0.2.73>:

* `nexa-cli_windows_arm64.exe` (94 MB, signed by "Nexa AI, Inc.")
* Inno Setup installer; **silent flag is `/VERYSILENT`** (NOT NSIS-style
  `/S` — that hangs).
* Default install location: `%LOCALAPPDATA%\Nexa CLI\`.
* No PATH modification — invoke via full path or add the install dir to
  `$env:Path`.

**Native ARM64 is preferred** for our hardware, but the v0.2.73 release
also ships `nexa-cli_windows_x86_64.exe` (154 MB) for x64 Prism
emulation if needed.

```powershell
# Install
.\nexa-cli_windows_arm64.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG=install.log

# Verify
$env:Path = "$env:LOCALAPPDATA\Nexa CLI;$env:Path"
nexa version
# NexaSDK Bridge Version: v1.0.45-rc1
# NexaSDK CLI Version:    v0.2.73
```

### Model download — works for at least some HF-hosted models

```powershell
# Set license (persistent config; env var also works for non-NPU models)
nexa config set license 'key/eyJhY2NvdW50...'   # README default

# Pull — works
nexa --skip-update pull NexaAI/Granite-4.0-h-350M-NPU
# 1.2 GB downloaded in ~55 s, cached at:
#   %USERPROFILE%\.cache\nexa.ai\nexa_sdk\models\NexaAI\

# nexa list — works
nexa --skip-update list
# ┌───────────────────────────────┬─────────┬────────┬──────┬────────┐
# │ NAME                          │ SIZE    │ PLUGIN │ TYPE │ QUANTS │
# ├───────────────────────────────┼─────────┼────────┼──────┼────────┤
# │ NexaAI/Granite-4.0-h-350M-NPU │ 1.2 GiB │ npu    │ llm  │ N/A    │
# └───────────────────────────────┴─────────┴────────┴──────┴────────┘
```

### Cache layout (for cleanup later)

```
%USERPROFILE%\.cache\nexa.ai\nexa_sdk\models\<NexaAI\Model-Name>\
%LOCALAPPDATA%\Nexa CLI\                         (88 MB, install)
%LOCALAPPDATA%\Nexa CLI\npu\                     (NPU plugin DLLs:
                                                  granite-nano-sdk.dll,
                                                  granite4-sdk.dll, etc.)
```

## What's blocked

### NPU inference — license-server outage

Every `nexa infer` for an NPU model ends with:

```
🌎loading model... [60+ frames of progress]
Error: SDKError(Invalid license)
[ERROR] License validation failed: fail to request license host for activation
```

…after a **21.5-second timeout**. We tried both:

1. `$env:NEXA_TOKEN = "key/..."` (env var path)
2. `nexa config set license 'key/...'` (persistent config path)

Both fail identically — the runtime still calls a hardcoded activation
host. Per **Goribesh's packet-trace diagnosis on issue #1072
(2026-05-07)**:

> nexa-cli.exe is sending TCP SYN to **54.200.56.5:443** (an AWS
> us-west-2 IP) and the host never responds. Connection stays in
> SynSent state for ~21s before the SDK times out.
>
> This appears to be a decommissioned activation host. The README
> explicitly states "previous token validation service has been
> deprecated", but the v0.2.73 binary still hardcodes a call to it on
> first NPU model use.

Confirmed on our hardware: `Test-NetConnection lic.nexa.ai -Port 443`
returns `TcpTestSucceeded: False`. Resolves to the same
`54.200.56.5`.

### Python SDK — `pip install nexaai` broken

`pip install nexaai` (currently 1.0.44, sdist-only) builds the wheel by
calling out to `nexa-model-hub-bucket.s3.us-west-1.amazonaws.com` for a
`nexasdk-bridge.zip`, which **returns 403 Forbidden for every platform
key** (windows_arm64, windows_x86_64, linux_*, macos_arm64). Tested
versions 1.0.41 → 1.0.44, all four platform paths — all 403. See issues
[#1069](https://github.com/qualcomm/nexa-sdk/issues/1069) and
[#1071](https://github.com/qualcomm/nexa-sdk/issues/1071).

### Some non-NPU GGUF pulls hang

`nexa pull ggml-org/Qwen3-0.6B-GGUF` (and a couple of `NexaAI/*-GGUF`
names referenced in the v0.2.73-era docs) hang indefinitely in the
"download manifest" phase. Pulls of NPU model names work; pulls of GGUF
names listed in the *current* (2026-04+) README don't, suggesting the
v0.2.73 CLI's manifest endpoint is also part of the migration.

## Root cause and recovery path

**Qualcomm acquired Nexa AI in March 2026** ([qualcomm/nexa-sdk
discussion #1058](https://github.com/qualcomm/nexa-sdk/discussions/1058)).
Infrastructure is migrating from `nexa.ai`/`nexa4ai.com` to
Qualcomm-owned hosts. Per [`iwr-redmond` on
issue #1073](https://github.com/qualcomm/nexa-sdk/issues/1073) (2026-05-10):

> Nexa was acquired by Qualcomm in March and infrastructure is currently
> being migrated. Have you tried the default license in the Readme file
> while we wait for the migration to complete?

We tried. Default license is read by the runtime, but the runtime then
calls a deprecated activation host that no longer responds. **No
user-side workaround exists**; this requires Qualcomm to either:

1. Ship a new CLI build that skips the dead activation host, or
2. Bring up an activation host at the old IP, or
3. Provide an offline activation path.

### Issues to watch for resolution

| # | Title | Opened | Status (2026-05-10) |
|---|-------|--------|---------------------|
| [#1068](https://github.com/qualcomm/nexa-sdk/issues/1068) | "Qualcomm NPU license not available" | 2026-04-24 | closed-completed (but not resolved) |
| [#1069](https://github.com/qualcomm/nexa-sdk/issues/1069) | "Unable to install nexaai sdk" (Python SDK 403) | 2026-04-24 | open |
| [#1071](https://github.com/qualcomm/nexa-sdk/issues/1071) | "Windows ARM64 installation completely broken" | 2026-05-02 | open |
| [#1072](https://github.com/qualcomm/nexa-sdk/issues/1072) | "NEXA_TOKEN does not work: SDKError(Invalid license)" | 2026-05-03 | open |
| [#1073](https://github.com/qualcomm/nexa-sdk/issues/1073) | "Windows ARM64 NPU activation fails: lic.nexa.ai unreachable" | 2026-05-08 | open |

When ALL of #1069, #1071, #1072, #1073 are closed, this experiment is
worth re-running.

## Architecture deliverables

### Tether's repo already has a `NexaProvider` stub

The refactor team shipped a stub at
`tether_service/providers/nexa/provider.py` with the right
`ProviderCapabilities`:

```python
class NexaProvider(ModelProvider):
    @property
    def kind(self) -> str: return "nexa"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools_native=True,        # OpenAI-style tools API
            tools_marker=False,
            thinking_channel=False,
            cancel_inflight=True,
            multi_model=True,
            warm_up_required=False,   # Server-side cold-start
        )
```

All concrete methods raise `NotImplementedError`. The `ProviderCapabilities`
shape was designed in the refactor briefing
(`docs/REFACTOR_BRIEFING.md` §2 Seam A) explicitly to support
`NexaProvider` without further type changes — confirmed by the
NexaProvider docstring in `tether_service/providers/types.py`.

### Our HTTP-targeting skeleton

[`reference/nexa_provider_skeleton.py`](./reference/nexa_provider_skeleton.py)
fills in the methods against the **OpenAI-compatible HTTP server**
(`nexa serve`). Key choices:

* **Targets HTTP, not the broken Python SDK.** Cleaner abstraction,
  works around the `pip install nexaai` blocker, and the same
  HTTP-boundary code works whether the runtime is local or remote.
* **`httpx.AsyncClient`** for streaming SSE responses. No
  Nexa-specific Python deps — the skeleton is import-safe even when
  `nexaai` can't be installed.
* **`stream_typed`** parses OpenAI-style SSE chunks and emits
  `ProviderText` for `delta.content` and `ProviderToolCall` for
  `delta.tool_calls`, matching the contract Tether's orchestrator
  consumes (Phase 5+).
* **`cancel_token`** is honored at chunk boundaries; closing the
  `httpx` response stream when cancelled aborts the upstream request
  cleanly — matches `cancel_inflight=True` capability.
* **Tool calls coalesced per `index`** — OpenAI splits arguments
  across many SSE events; the skeleton accumulates and emits a single
  `ProviderToolCall` event per call when `finish_reason="tool_calls"`.

When the upstream issues are resolved, this skeleton needs ≤ 50 lines
of changes to:

1. Replace the `ModelProvider = object` placeholder with the real
   import (file moves into the Tether repo).
2. Validate `tools` payload shape against what `nexa serve` actually
   accepts (untested today since the server can't be exercised).
3. Add a small subprocess manager if Tether owns the `nexa serve`
   lifecycle, or a config-driven URL if it's externally managed.

## What's NOT in this round

Listed for the next session, when the license server is back:

- ❌ Live `nexa infer` smoke on Granite-4.0-h-350M-NPU
- ❌ Live `nexa infer` on OmniNeural-4B
- ❌ Head-to-head benchmark vs Adreno Qwen3-4B-q4f16_1
- ❌ Function-calling SSE protocol shape from a live run
- ❌ Multi-turn KV reuse semantics (`reset()` / `save_kv_cache`)
  from a live run
- ❌ Validation that our HTTP skeleton actually parses real Nexa SSE

## Recheck recipe (for future-us)

When issues #1069/#1072/#1073 close:

```powershell
# 1. Verify license server reachable
Test-NetConnection lic.nexa.ai -Port 443
# Expect: TcpTestSucceeded: True

# 2. Set license (persistent config — survives session restarts)
$env:Path = "$env:LOCALAPPDATA\Nexa CLI;$env:Path"
nexa config set license 'key/eyJ...'   # README default

# 3. Smoke test (the one we couldn't run)
nexa --skip-update infer NexaAI/Granite-4.0-h-350M-NPU `
  --max-tokens 20 -p "Reply with: OK"
# If license host is back, this returns text instead of timing out.

# 4. If smoke passes, run the head-to-head bench
#    (adapt scripts/bench_model.py to drive nexa serve over HTTP —
#     TODO when unblocked)
```

## Disk-only artifacts (NOT committed)

Currently retained on disk for instant recheck:

- `<workspace>\nexa-cli_windows_arm64.exe` (94 MB) — installer
- `<workspace>\nexaai-1.0.44.tar.gz` (73 KB) — Python sdist (broken)
- `%LOCALAPPDATA%\Nexa CLI\` (~88 MB, installed CLI)
- `%USERPROFILE%\.cache\nexa.ai\nexa_sdk\models\NexaAI\Granite-4.0-h-350M-NPU\`
  (~1.2 GB, NPU model)

Total: ~1.4 GB. Drop with:

```powershell
# Uninstall CLI
& "$env:LOCALAPPDATA\Nexa CLI\unins000.exe" /VERYSILENT
# Delete cached models
Remove-Item -Recurse "$env:USERPROFILE\.cache\nexa.ai"
# Delete installer cache (whatever workspace path is in use)
```

## Move-back when refactor settles

Tether's refactor has already merged the `NexaProvider` *stub* into
`main`. When the license outage is resolved and we want to ship the
real implementation:

1. Pull this skeleton's `stream_typed` body and helpers into
   `tether_service/providers/nexa/provider.py`, replacing the
   `NotImplementedError` body of each method.
2. Restore the real imports
   (`from tether_service.providers.types import …`).
3. Add unit tests under `tests/unit/providers/test_nexa_provider.py`
   that mock the SSE stream — no live server needed.
4. Add an integration test under
   `tests/integration/test_nexa_provider_live.py` gated by a
   `NEXA_LIVE=1` env var so CI doesn't depend on the server.
5. Write a `tools/run_nexa_serve.py` sidecar manager (or document
   external management) that Tether's `Engine` can use.
6. Re-run the bench from `scripts/bench_model.py` adapted for HTTP and
   produce head-to-head numbers vs the existing MLC Adreno baseline.

## Sources

- [qualcomm/nexa-sdk README](https://github.com/qualcomm/nexa-sdk) — install + token
- [v0.2.73 release](https://github.com/qualcomm/nexa-sdk/releases/tag/v0.2.73) — working ARM64 / x64 installers
- [discussion #1058](https://github.com/qualcomm/nexa-sdk/discussions/1058) — Qualcomm acquisition (March 2026)
- Issues: [#1068](https://github.com/qualcomm/nexa-sdk/issues/1068),
  [#1069](https://github.com/qualcomm/nexa-sdk/issues/1069),
  [#1071](https://github.com/qualcomm/nexa-sdk/issues/1071),
  [#1072](https://github.com/qualcomm/nexa-sdk/issues/1072),
  [#1073](https://github.com/qualcomm/nexa-sdk/issues/1073)
- [NexaSDK Python API Reference](https://docs.nexa.ai/en/nexa-sdk-python/api-reference) — ground truth for the eventual SDK shape
