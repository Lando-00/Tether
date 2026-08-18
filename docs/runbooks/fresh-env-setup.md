# Windows Python environments — setup runbook

Per the locked decision recorded in
[`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
§1 and ADR-0014: `mlc-venv2` was the environment used during the refactor;
`tether` is the x64/Prism MLC validation environment. A separate native ARM64
stdlib venv is the validated environment for default-provider GenieX work.

The native ARM64 venv and the x64 conda environments can coexist because they
live at separate paths and serve different provider workflows.

## Choose the environment for the provider

Tether supports two useful Windows-on-ARM64 development shapes:

| Environment | Use it for | Why |
|---|---|---|
| Native ARM64 Python 3.12 stdlib venv | Default GenieX provider, HTTP/API/connector work that does not import MLC | GenieX runs out of process; use the ARM64 python.org interpreter |
| x64 Python 3.12 conda env under Prism | MLC provider development and hardware tests | The CodeLinaro wheels are published only as `cp312-cp312-win_amd64` |

The installed Miniconda reports `platform: win-64` and
`__archspec=x86_64`. It can create only x64 environments; neither
`environment-tether.yml` nor `scripts/setup_fresh_env.ps1` can produce a
native ARM64 environment. Those files remain correct only for the MLC path.

## Native ARM64 venv for GenieX

Create the venv outside the clone using the ARM64 Python 3.12 interpreter from
python.org:

```powershell
<arm64-python-3.12>\python.exe -m venv <venv-path>
& <venv-path>\Scripts\Activate.ps1
```

The validated interpreter identifies as
`3.12.10 [MSC v.1943 64 bit (ARM64)]`, with
`platform.machine() == "ARM64"`. Keeping the venv outside the repository
avoids adding environment files or ignore rules to the clone.

### Install on Windows ARM64

Do not install the `server` extra in this environment.
`uvicorn[standard]` pulls in `httptools`, which has no `win_arm64` wheel and
attempts a source build requiring Microsoft Visual C++ 14 or newer.
`httptools` is the only missing wheel from that extra; install the remaining
server dependencies explicitly:

```powershell
<venv-path>\Scripts\python.exe -m pip install -e ".[cli,brave,dev]" `
  "fastapi>=0.117.0,<0.118.0" "uvicorn>=0.37.0,<0.38.0" `
  watchfiles websockets pyyaml python-dotenv colorama
```

Uvicorn falls back to the pure-Python `h11` HTTP parser rather than the C
`httptools` parser. This is not operationally significant for a single-user
local service.

`tzdata` is a declared Windows dependency. Windows has no system IANA
timezone database, and without the package a bare venv fails timezone-tool
tests with `Unknown timezone: UTC`. A pre-existing environment created before
the dependency was declared may need:

```powershell
<venv-path>\Scripts\python.exe -m pip install tzdata
```

Activation affects only the current process. Scripts and one-shot shell calls
should invoke `<venv-path>\Scripts\python.exe` directly.

### Verified ARM64 baseline

The native ARM64 venv passes:

- **2002 passed, 20 skipped, 69 deselected**
- Docs drift gate: **9 passed**
- `ruff check src/tether tests`
- `mypy src/tether` across 119 source files

The expected skips are nine tests requiring `mlc_llm`, seven OpenTelemetry
tests plus one related optional-OTel case (install `tether[otel]` to enable
them), and three POSIX/SQLite back-compatibility cases. The historically
observed Windows
`test_tool_contract` “no current event loop” failures did not occur in this
ARM64 venv; that does not rewrite the different historical x64/conda
baseline.

## x64/Prism conda environment for MLC

### Prerequisites

| Requirement | Notes |
|---|---|
| Snapdragon X Elite (Windows on ARM64) | The same hardware Tether targets in production. ADR-0014. |
| Miniconda / Mambaforge | A common Miniconda base is `%USERPROFILE%\miniconda3`; `conda` may not be on `PATH` in a fresh shell |
| Anaconda channel terms accepted | A new contributor may need to accept Anaconda's Terms of Service for configured default channels before environment creation succeeds |
| Working `mlc-venv2` env | Reference baseline; the fresh env must match its test count. |
| CodeLinaro `2025.06.r1` wheels (5.4 MB + 41.9 MB) | Local-first under `%USERPROFILE%\Downloads\`; JFrog fallback is wired into the bootstrap script. |
| `BRAVE_API_KEY` in `.env` (optional) | Only if running `pytest -m network`. |

### TL;DR — single command

```powershell
# From the project root:
.\scripts\setup_fresh_env.ps1
```

The script is idempotent: re-run after any failure.

### Finding conda in a fresh shell

Do not assume `conda` is already on `PATH`. With the common Miniconda layout,
invoke it directly:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" --version
```

Alternatively, initialize the current PowerShell process:

```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
conda activate tether
```

Activation changes only the current process. It does not persist across
separate one-shot shell invocations. Automation should therefore call the
environment interpreter directly, for example
`<conda-base>\envs\tether\python.exe`, rather than depending on a previous
`conda activate`.

### What the bootstrap script does

1. **Pre-flight** — verifies `conda` on PATH and host arch.
2. **Wheels** — checks for the two CodeLinaro wheels under `%USERPROFILE%\Downloads\`; downloads from CodeLinaro JFrog if missing.
3. **Conda env** — creates `tether` from `environment-tether.yml` with `CONDA_SUBDIR=win-64` so x64 Python (not native ARM64) is selected. Pins `subdir = win-64` inside the env so subsequent `conda install` calls stay x64.
4. **MLC wheels** — installs TVM wheel first (mlc_llm depends on TVM symbols at import time), then MLC wheel.
5. **Tether** — editable install with extras `[server,cli,brave,otel,dev]`.
6. **Smoke tests** — `import tether`, `from tether import Engine`, `tether is tether_service` (alias identity), `import mlc_llm`.

### After bootstrap — manual verification

Replace `<envpy>` with the path printed by the bootstrap script, or activate
the environment and use `$env:CONDA_PREFIX\python.exe`.

#### Default test suite (required)

```powershell
& <envpy> -m pytest -q
```

Expected: all default-on tests pass. Test counts change as the repository grows.

#### Docs drift gate

```powershell
& <envpy> -m pytest -m docs tests/docs/ -q
```

Expected: all docs drift tests pass.

#### Real Brave Search (network)

Pre-condition: `.env` has a valid `BRAVE_API_KEY`. Get a free key at <https://api-dashboard.search.brave.com/>.

```powershell
& <envpy> -m pytest -m network -q
```

#### Server smoke test

```powershell
& <envpy> -m tether.app
# In another shell:
Invoke-WebRequest http://127.0.0.1:8080/healthz
# Expected: 200 OK
# Then Ctrl+C the server — should exit cleanly within ~5s (force-exit budget).
```

#### CLI console scripts

```powershell
& <envpy> -m pip show tether
# Then re-launch a shell so the env's Scripts/ is on PATH, or call directly:
& "$($envpy | Split-Path)\Scripts\tether-server.exe" --help
& "$($envpy | Split-Path)\Scripts\tether-cli.exe" --help
```

Both should resolve and exit 0.

## Troubleshooting

### Symptom: `pip install` of MLC wheel fails with "unsupported platform"

Cause: env got native ARM64 Python instead of x64 under Prism.

Fix:
```powershell
conda env remove -n tether
$env:CONDA_SUBDIR = "win-64"
conda env create -f environment-tether.yml
```

The bootstrap script does this automatically; this manual fix is for the rare case where you bypassed it.

### Symptom: `import mlc_llm` fails with `ImportError: DLL load failed while importing _ssl`

**Root cause** — a stale base-conda `libcrypto-3-x64.dll` masks the env's own copy when TVM iterates `%PATH%`.

In detail:

1. `tvm._ffi.libinfo.get_dll_directories()` (inside the CodeLinaro TVM wheel) iterates every entry of the parent shell's `%PATH%` and feeds each one to `os.add_dll_directory()` as part of `import tvm`. This is hard-coded behaviour upstream — we cannot change it.
2. When the parent shell has the **base conda env** on `%PATH%` (for example `<conda-base>\Library\bin`), TVM prepends that directory to the per-process DLL search list.
3. On the validation host, the base conda env had an older conda-forge OpenSSL, so `<conda-base>\Library\bin\libcrypto-3-x64.dll` exported only the older symbol set.
4. The tether env itself ships conda-forge openssl 3.5.3 (`Library\bin\libcrypto-3-x64.dll` is **7.3 MB**), and CPython's `_ssl.pyd` is linked against the 3.5 symbol set.
5. After `import tvm`, Windows resolves `libcrypto-3-x64.dll` (a *bare-name* dependency of `_ssl.pyd`) to the **3.0.17** copy first, then the import of `_ssl` aborts with `ImportError: DLL load failed while importing _ssl: The specified procedure could not be found.` because a 3.5-only symbol is missing.

mlc-venv2 does not exhibit the bug only because at the time of validation it was the active conda env and `conda activate` puts `…\envs\mlc-venv2\Library\bin` (3.5.3 openssl) *before* the base env on `%PATH%`. As soon as the tether env is invoked via its full python path from any other shell, the precondition collapses and the bug surfaces. Both envs have **byte-identical** `libtvm.dll` and `libcrypto-3-x64.dll` files — the difference is purely DLL search ordering.

**Real fix landed in this repo** — a `.pth` file pre-imports `ssl` during Python's site initialization, *before* any user code (including `import tvm`) has a chance to mutate the DLL search list. Once `_ssl.pyd` is bound to the correct `libcrypto-3-x64.dll`, Windows caches that binding for the process lifetime and every later `import ssl` (transitively via `mlc_llm → fastapi → anyio → ssl`, `openai → httpx → ssl`, etc.) reuses it.

- File: [`src/_tether_dll_fix.pth`](../../src/_tether_dll_fix.pth) — single-line `import ssl` plus a long comment explaining the rationale.
- Wired into the wheel via `[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml` so the file lands directly in `site-packages/_tether_dll_fix.pth` for both regular and editable installs.

Earlier strategies that **did not** work (kept here for the next person who tries them):

| Attempt | Why it failed |
|---|---|
| `os.add_dll_directory(<env>/Library/bin)` *before* `import tvm` | TVM's later `os.add_dll_directory()` calls prepend to the search list, so they win regardless of pre-existing entries. |
| Prepend `<env>/Library/bin` to `%PATH%` before `import tvm` | Same — TVM iterates `%PATH%` in order but each `add_dll_directory` call moves that entry to the head; the last `add` wins. |
| Adding a shim in `tether.providers.mlc.provider.__init__` | Runs too late — `import mlc_llm` (which is what triggers `import tvm`) has already corrupted the search path before the provider module loads. |

**Verification** (run after `conda env remove -n tether; conda env create -f environment-tether.yml; .\scripts\setup_fresh_env.ps1`):

```powershell
& <envpy> -c "import mlc_llm; print('mlc_llm:', mlc_llm.__version__)"
# Expected: mlc_llm: 0.1.dev0
& <envpy> -m pytest -q --ignore=tests/integration/test_cancellation_contract.py
# Expected: all selected tests pass.
```

Tracked as `fu-fresh-env-mlc-llm-import` — **resolved** as of this fix.

Note: a second fresh-env gap surfaced during the same validation — `trio` was missing from the env, which collapsed ~60 anyio-based tests from `[asyncio,trio]` to `[asyncio]` only. `scripts/setup_fresh_env.ps1` now installs `trio` after the editable install as a stopgap; the proper pin will move to `pyproject.toml` as part of `fu-pin-fastapi-pydantic-versions`.

### Dependency pins that are deliberate

- Keep `setuptools<81`: yoyo 8.x imports `pkg_resources`.
- Keep `neonize==0.3.17.post0`: ADR-0018 D1 records the adapter and
  binary-compatibility reasons for the exact pin.

### Symptom: docs drift gate (`pytest -m docs`) fails on `test_openapi_no_drift`

Cause: FastAPI/Pydantic versions differ between envs. Pydantic V2 adds `ValidationError` schema fields in minor releases.

**Status (Phase 8 close, ResolvedBy `fu-pin-fastapi-pydantic-versions`)**: `pyproject.toml` now pins `pydantic>=2.11.0,<2.12.0`, `pydantic-settings>=2.14.0,<2.15.0`, `fastapi>=0.117.0,<0.118.0`, `uvicorn[standard]>=0.37.0,<0.38.0` to the `mlc-venv2` baseline. The drift gate now passes 4/4 in both envs. When intentionally bumping these versions, regenerate `docs/specs/openapi.json` via `python -m scripts.docs.generate` and commit alongside the pin change.

### Symptom: `import mlc_llm` succeeds but `Engine` fails with "no such model"

Cause: `models/` is empty. The fresh env doesn't carry over your downloaded models.

Fix: either (a) set `TETHER_MODELS_DIR` to point at your existing model dir, or (b) copy/symlink at least one MLC model into `models/`. See [`../architecture.md`](../architecture.md#5-data-layout).

### Symptom: pytest collects 0 tests

Cause: editable install wasn't re-run after the env was created.

Fix:
```powershell
& <envpy> -m pip install -e ".[server,cli,brave,otel,dev]"
```

### Symptom: `BRAVE_API_KEY not set` during `pytest -m network`

Cause: `.env` missing or key not set. The `conftest.py` uses python-dotenv at session start.

Fix: copy `.env.example` to `.env` and fill in `BRAVE_API_KEY`. Re-run.

## Regenerating `requirements.txt`

The repo's `requirements.txt` is a pinned lock file derived from `pyproject.toml` extras `[server,cli,brave,otel,dev]`. Regenerate from inside the fresh `tether` env so the resolver matches what the env actually sees:

```powershell
& <envpy> -m pip install pip-tools
& <envpy> -m piptools compile pyproject.toml `
    --extra server --extra cli --extra brave --extra otel --extra dev `
    -o requirements.txt
```

Commit `requirements.txt` alongside any pyproject changes.

## References

- ADR-0014 — Pin Qualcomm CodeLinaro `2025.06.r1` runtime
- ADR-0016 — MLC isolation rule (no imports outside factory)
- [`../architecture.md`](../architecture.md#9-frozen--locked-decisions-dont-relitigate) — locked decisions
- `docs/research/05_mlc_llm_versioning.md` — wheel catalog (auto-generated)
- `scripts/setup_fresh_env.ps1` — idempotent bootstrap
- `environment-tether.yml` + `environment-mlc-venv2.yml` — conda env specs
