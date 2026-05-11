# Fresh `tether` conda env — setup runbook

Per the locked decision in `_synthesis.md` §3: `mlc-venv2` is the env used during the refactor; `tether` is a fresh validation env created at the end. This runbook walks through creating and validating that fresh env.

Both envs coexist — `mlc-venv2` keeps working through the validation period. Drop it later.

## Prerequisites

| Requirement | Notes |
|---|---|
| Snapdragon X Elite (Windows on ARM64) | The same hardware Tether targets in production. ADR-0014. |
| Miniconda / Mambaforge with `conda` on PATH | Verify: `conda --version` |
| Working `mlc-venv2` env | Reference baseline; the fresh env must match its test count. |
| CodeLinaro `2025.06.r1` wheels (5.4 MB + 41.9 MB) | Local-first at `C:\Users\lovan\Downloads\`, JFrog fallback wired into the bootstrap script. |
| `BRAVE_API_KEY` in `.env` (optional) | Only if running `pytest -m network`. |

## TL;DR — single command

```powershell
cd D:\Dev\Tether
.\scripts\setup_fresh_env.ps1
```

The script is idempotent: re-run after any failure.

## What the bootstrap script does

1. **Pre-flight** — verifies `conda` on PATH and host arch.
2. **Wheels** — checks for the two CodeLinaro wheels under `C:\Users\lovan\Downloads\`; downloads from CodeLinaro JFrog if missing.
3. **Conda env** — creates `tether` from `environment-tether.yml` with `CONDA_SUBDIR=win-64` so x64 Python (not native ARM64) is selected. Pins `subdir = win-64` inside the env so subsequent `conda install` calls stay x64.
4. **MLC wheels** — installs TVM wheel first (mlc_llm depends on TVM symbols at import time), then MLC wheel.
5. **Tether** — editable install with extras `[server,cli,brave,otel,dev]`.
6. **Smoke tests** — `import tether`, `from tether import Engine`, `tether is tether_service` (alias identity), `import mlc_llm`.

## After bootstrap — manual verification

Replace `<envpy>` with the path printed by the bootstrap script (typically `C:\ProgramData\miniconda3\envs\tether\python.exe`).

### Default test suite (required)

```powershell
& <envpy> -m pytest -q
```

Expected: **1343 passed, 4 skipped, 49 deselected** (matches `mlc-venv2` baseline).

### Docs drift gate

```powershell
& <envpy> -m pytest -m docs tests/docs/ -q
```

Expected: **4 passed**.

### Real Brave Search (network)

Pre-condition: `.env` has a valid `BRAVE_API_KEY`. Get a free key at <https://api-dashboard.search.brave.com/>.

```powershell
& <envpy> -m pytest -m network -q
```

### Server smoke test

```powershell
& <envpy> -m tether.app
# In another shell:
Invoke-WebRequest http://127.0.0.1:8080/health
# Expected: 200 OK
# Then Ctrl+C the server — should exit cleanly within ~5s (force-exit budget).
```

### CLI console scripts

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

Cause: `import tvm.relax` (transitively pulled in by `mlc_llm.__init__`) loads `libtvm.dll` which alters the DLL search path in a way that prevents Python's `_ssl.pyd` from loading correctly afterwards. Reproduces with `python -c "import tvm.relax; import ssl"` even when `import ssl` alone works.

The fresh `tether` env hits this; `mlc-venv2` does not. Both envs have identical `libtvm.dll` bytes, openssl 3.5.3, and Python 3.12.11 — but `tether` exhibits the symptom and `mlc-venv2` does not. The difference is unidentified (possibly accumulated `os.add_dll_directory()` state from earlier installs in `mlc-venv2`'s history).

**Workarounds**:
- For non-runtime workflows (tests that don't exercise the real MLC engine, server smoke that doesn't load a model): the Tether package still works — `import tether`, `from tether import Engine`, and 1295/1306 tests pass per the validation log. Tests that fail to collect are MLC-runtime-dependent.
- For real chat sessions: use `mlc-venv2`. This is an env-fragility issue with the CodeLinaro `2025.06.r1` TVM wheel on certain conda-forge package combinations; fixing it likely requires either rebuilding the wheels OR pinning a specific TVM wheel + DLL search order strategy. Tracked as `fu-fresh-env-mlc-llm-import` (future work).

### Symptom: docs drift gate (`pytest -m docs`) fails on `test_openapi_no_drift`

Cause: FastAPI/Pydantic versions differ between envs. Pydantic V2 adds `ValidationError` schema fields in minor releases.

**Status (Phase 8 close, ResolvedBy `fu-pin-fastapi-pydantic-versions`)**: `pyproject.toml` now pins `pydantic>=2.11.0,<2.12.0`, `pydantic-settings>=2.14.0,<2.15.0`, `fastapi>=0.117.0,<0.118.0`, `uvicorn[standard]>=0.37.0,<0.38.0` to the `mlc-venv2` baseline. The drift gate now passes 4/4 in both envs. When intentionally bumping these versions, regenerate `docs/specs/openapi.json` via `python -m scripts.docs.generate` and commit alongside the pin change.

### Symptom: `import mlc_llm` succeeds but `Engine` fails with "no such model"

Cause: `models/` is empty. The fresh env doesn't carry over your downloaded models.

Fix: either (a) set `TETHER_MODELS_DIR` to point at your existing model dir, or (b) copy/symlink at least one MLC model into `models/`. See `docs/architecture.md` §5.

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
- `docs/architecture.md` §9 — Locked decisions
- `docs/research/05_mlc_llm_versioning.md` — wheel catalog (auto-generated)
- `scripts/setup_fresh_env.ps1` — idempotent bootstrap
- `environment-tether.yml` + `environment-mlc-venv2.yml` — conda env specs
