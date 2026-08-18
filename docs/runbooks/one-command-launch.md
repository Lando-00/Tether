# One-Command Launch — `tether.ps1`

`tether.ps1` at the repository root brings the whole local stack up with a
single command: the **GenieX** inference server, the **Tether HTTP service**,
and the **Tether CLI**, in that order, with a health gate between each step.

```powershell
.\tether.ps1
```

That is the whole happy path. Everything below is detail.

---

## What it does

1. **Resolves the interpreter** and verifies `import tether` works, so a wrong
   or unbuilt environment fails with instructions instead of a traceback from
   somewhere deep in startup.
2. **Reads live settings** by shelling out to `scripts/dev/launch_config.py`,
   which loads the real `Settings` object. Ports, provider ids and the GenieX
   base URL therefore come from `src/tether/config/default.yml` plus any
   `TETHER__*` environment overrides — the launcher has no hardcoded ports and
   cannot drift from config.
3. **Starts GenieX** (unless already healthy) and waits for `GET <base>/v1/`.
4. **Starts the Tether service** (unless already listening) and waits for
   `GET /api/v1/readyz`.
5. **Runs the CLI** in the foreground of the current window, pointed at the
   API base URL it just verified.
6. **Tears down only what it started** when the CLI exits.

### Reuse, never duplicate

Each service is probed before being started. An already-running GenieX is
adopted and explicitly **not** stopped on exit — the operator owns its
lifecycle (see [`geniex-provider.md`](./geniex-provider.md)). The same applies
to an already-running Tether service. This makes it safe to run `.\tether.ps1`
repeatedly, or to leave GenieX warm across many CLI sessions.

---

## Commands

| Command | Effect |
|---|---|
| `.\tether.ps1` | `up` (default): start what's missing, then chat |
| `.\tether.ps1 status` | Report health of both services and any launcher-managed pids. Exit code 0 only when Tether reports ready |
| `.\tether.ps1 stop` | Stop the services this launcher started |
| `.\tether.ps1 logs` | Tail the background service logs (`-Follow` to stream) |

`tether.cmd` is a shim that forwards to the same script, so `tether status`
works from `cmd.exe`, Explorer, or a Run box.

---

## Common options

```powershell
# Run GenieX on CPU with a smaller model
.\tether.ps1 -Compute cpu -Model "unsloth/Qwen3-1.7B-GGUF:Q4_0"

# Headless: bring the stack up and leave it running
.\tether.ps1 -NoCli

# GenieX is already managed elsewhere (or the provider is Ollama/dummy)
.\tether.ps1 -NoGeniex

# Watch each service in its own console window instead of log files
.\tether.ps1 -ShowWindows

# Keep the services alive after leaving the CLI
.\tether.ps1 -KeepRunning

# Route to a specific provider / orchestrator mode, or pass CLI flags through
.\tether.ps1 -Provider geniex -Mode research
.\tether.ps1 -CliArgs "--reasoning-effort","high"
```

Run `Get-Help .\tether.ps1 -Full` for the complete parameter list.

---

## Resolution order

The launcher discovers everything it needs, in this order, and stops at the
first hit:

| Thing | Order |
|---|---|
| Python | `-Python` → `$env:TETHER_PYTHON` → active `$env:VIRTUAL_ENV` → `.venv\Scripts\python.exe` → `.venv-geniex\Scripts\python.exe` → `python` on `PATH` |
| GenieX exe | `-GenieXPath` → `geniex` on `PATH` → `%LOCALAPPDATA%\GenieX CLI\geniex.exe` |
| GenieX data dir | `-DataDir` → `$env:GENIEX_DATADIR` → `.\models\geniex` |

### Machine-local defaults

A common setup keeps the venv outside the clone. Rather than typing `-Python`
every time, copy the example config:

```powershell
Copy-Item tether.local.example.ps1 tether.local.ps1
```

`tether.local.ps1` is gitignored and dot-sourced before any resolution runs, so
it can default any launcher parameter that was not passed on the command line.

---

## Logs and state

Without `-ShowWindows`, each background service is started hidden with stdout
and stderr redirected to `.tether-run\logs\<service>.{out,err}.log`. The
launcher records the pids it started in `.tether-run\state.json`, which is what
`stop` and `status` read. Both live under the gitignored `.tether-run\`
directory.

> Windows does not flush a file's size into its directory entry while a process
> still holds it open, so a live log reports 0 bytes to `Get-ChildItem`. The
> `logs` command reads the files instead of trusting `Length`.

---

## Troubleshooting

**"cannot import 'tether'"** — the resolved interpreter is not the Tether
environment. Pass `-Python`, set `$env:TETHER_PYTHON`, or create
`tether.local.ps1`.

**"GenieX did not become healthy"** — the launcher prints the tail of both
GenieX logs and stops what it started. Re-run with `-ShowWindows` to watch the
server start in its own window.

**Tether reports "up, not ready — HTTP 503"** — the HTTP service is listening
but a provider is degraded. This is expected when GenieX is down; `/readyz` is
deliberately not a GenieX liveness probe. Probe GenieX directly at
`<base_url>/v1/`.

**Ctrl+C killed the console before teardown ran** — pids survive in
`.tether-run\state.json`; run `.\tether.ps1 stop`.

**First message after a cold start is slow** — GenieX loads the model on the
first request, and an 8B generates at roughly 8 tok/s on this hardware. The
launcher deliberately does not send a warm-up prompt, because GenieX ignores
`max_tokens` and a throwaway warm-up would generate a full response.

---

## See also

- [`geniex-provider.md`](./geniex-provider.md) — GenieX operation, the Q4_0
  quantization rule, and the out-of-repo model store
- [`fresh-env-setup.md`](./fresh-env-setup.md) — creating the environment the
  launcher resolves
