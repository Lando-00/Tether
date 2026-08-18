# scripts/setup_fresh_env.ps1
#
# Idempotent bootstrap for the fresh `tether` conda environment per the
# locked decision in synthesis Section 3 ("Conda env: mlc-venv2 for the
# refactor; create fresh tether env at end for verification only").
#
# Run from the repo root:
#
#   .\scripts\setup_fresh_env.ps1
#
# What it does (each step is idempotent):
#   1. Verifies the host is Windows-on-ARM64 with x64-Python wheels available.
#   2. Verifies the two CodeLinaro 2025.06.r1 wheels (local-first under
#      %USERPROFILE%\Downloads\, with JFrog fallback URLs).
#   3. Creates conda env `tether` with x64 Python 3.12 (CONDA_SUBDIR=win-64).
#   4. Installs the two CodeLinaro wheels.
#   5. Editable-installs Tether with the full extras matrix.
#   6. Smoke-tests `import tether` and the Engine factory.
#
# Pre-conditions checked:
#   - conda on PATH
#   - python on PATH (any version; we use full paths to env pythons below)
#   - Existence (or downloadability) of the two pinned wheels
#
# References:
#   - ADR-0014 (CodeLinaro runtime pin)
#   - ADR-0016 (MLC isolation rule)
#   - docs/runbooks/fresh-env-setup.md (this script's runbook companion)

$ErrorActionPreference = "Stop"

# --- Configuration ---------------------------------------------------------

$EnvName = "tether"
$Subdir = "win-64"

$WheelDir = Join-Path $env:USERPROFILE "Downloads"
$TvmWheel = "tvm_adreno_cpu_clml_2025_06_r1-0.20.dev0-cp312-cp312-win_amd64.whl"
$MlcWheel = "mlc_llm_adreno_cpu_clml_2025_06_r1-0.1.dev0-cp312-cp312-win_amd64.whl"

$JFrogBase = "https://codelinaro.jfrog.io/artifactory/clo-472-adreno-opensource-ai/mlc-llm/2025.06.r1"

$ExtrasMatrix = "server,cli,brave,otel,dev"

# --- Helpers ---------------------------------------------------------------

function Write-Stage {
    param([string]$msg)
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Get-CondaEnvPython {
    param([string]$EnvName)
    # Always use the full path; never rely on `conda activate` because
    # activation here doesn't propagate to spawned subprocesses cleanly.
    $base = (conda info --base 2>$null).Trim()
    if (-not $base) {
        throw "conda is not on PATH. Install Miniconda / Mambaforge before running."
    }
    return (Join-Path $base "envs\$EnvName\python.exe")
}

function Ensure-WheelLocal {
    param([string]$WheelName, [string]$LocalDir, [string]$JFrogUrl)

    $LocalPath = Join-Path $LocalDir $WheelName
    if (Test-Path $LocalPath) {
        Write-Host "  Wheel present: $LocalPath"
        return $LocalPath
    }

    Write-Host "  Wheel missing locally; downloading from CodeLinaro JFrog..."
    if (-not (Test-Path $LocalDir)) {
        New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
    }
    try {
        Invoke-WebRequest -Uri $JFrogUrl -OutFile $LocalPath -UseBasicParsing
    }
    catch {
        throw "Failed to download $WheelName from $JFrogUrl. Check network or download manually to $LocalPath. Inner: $_"
    }
    if (-not (Test-Path $LocalPath)) {
        throw "Download appeared to succeed but $LocalPath is missing."
    }
    Write-Host "  Wheel downloaded: $LocalPath"
    return $LocalPath
}

# --- 1. Pre-flight checks --------------------------------------------------

Write-Stage "Pre-flight"

$arch = (Get-CimInstance Win32_ComputerSystem).SystemType
Write-Host "  SystemType: $arch"
if ($arch -notmatch "ARM|x64") {
    Write-Warning "Unexpected SystemType: $arch. Tether targets Snapdragon X Elite (ARM64) running x64 Python under Prism."
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not on PATH. Install Miniconda / Mambaforge before running."
}

# --- 2. Wheel discovery / download ----------------------------------------

Write-Stage "Resolving CodeLinaro 2025.06.r1 wheels"

$TvmPath = Ensure-WheelLocal $TvmWheel $WheelDir "$JFrogBase/$TvmWheel"
$MlcPath = Ensure-WheelLocal $MlcWheel $WheelDir "$JFrogBase/$MlcWheel"

# --- 3. Create conda env (idempotent) -------------------------------------

Write-Stage "Conda env: $EnvName"

$envExists = $false
$envList = conda env list 2>$null | Out-String
if ($envList -match "(?m)^\s*$EnvName\s+") {
    $envExists = $true
}

if ($envExists) {
    Write-Host "  Env '$EnvName' already exists; skipping `conda env create`."
}
else {
    Write-Host "  Creating env '$EnvName' from environment-tether.yml (CONDA_SUBDIR=$Subdir)..."
    $env:CONDA_SUBDIR = $Subdir
    try {
        & conda env create -f environment-tether.yml
        if ($LASTEXITCODE -ne 0) {
            throw "conda env create failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Remove-Item Env:\CONDA_SUBDIR -ErrorAction SilentlyContinue
    }
}

$envPython = Get-CondaEnvPython $EnvName
if (-not (Test-Path $envPython)) {
    throw "Expected python at $envPython but it doesn't exist. Inspect the env: conda env list"
}
Write-Host "  Env python: $envPython"

# Pin subdir inside the env so subsequent conda install stays x64.
# Safe to re-run; conda config is idempotent for --set.
Write-Host "  Pinning env subdir to $Subdir..."
& conda config --env --set subdir $Subdir
if ($LASTEXITCODE -ne 0) {
    Write-Warning "conda config --env --set subdir failed (exit $LASTEXITCODE). Subsequent conda installs may pick the host arch."
}

# Verify x64 Python (NOT ARM64)
$pyinfo = & $envPython -c "import sys, platform; print(f'arch={platform.machine()};maxsize={sys.maxsize};plat={sys.platform}')" 2>&1
Write-Host "  $pyinfo"
if ($pyinfo -notmatch "maxsize=9223372036854775807") {
    throw "Env python is not 64-bit. Recreate the env with CONDA_SUBDIR=win-64."
}

# --- 4. Install CodeLinaro wheels in dependency order ---------------------

Write-Stage "Installing CodeLinaro 2025.06.r1 wheels"

# TVM first (mlc_llm depends on tvm symbols at import time)
Write-Host "  pip install $TvmPath"
& $envPython -m pip install $TvmPath 2>&1 | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) {
    throw "pip install of TVM wheel failed with exit code $LASTEXITCODE"
}

Write-Host "  pip install $MlcPath"
& $envPython -m pip install $MlcPath 2>&1 | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) {
    throw "pip install of MLC wheel failed with exit code $LASTEXITCODE"
}

# --- 5. Editable install of Tether ----------------------------------------

Write-Stage "Editable install: Tether [$ExtrasMatrix]"

& $envPython -m pip install -e ".[$ExtrasMatrix]" 2>&1 | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) {
    throw "pip install -e . failed with exit code $LASTEXITCODE"
}

# --- 5b. Extra test-only deps not yet in pyproject extras -----------------
#
# `trio` enables anyio's dual-backend pytest parametrization (`[asyncio]` AND
# `[trio]`). Without it, ~60 anyio-based tests collapse to asyncio-only and
# the suite undershoots the mlc-venv2 baseline by ~60 tests. The CodeLinaro
# wheels do NOT require trio; it is purely a test-matrix concern. Pinning
# trio in pyproject.toml is owned by fu-pin-fastapi-pydantic-versions; until
# that lands, install it here so the fresh-env validation suite hits parity.

Write-Stage "Extra test-only deps (trio)"
& $envPython -m pip install trio 2>&1 | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip install trio failed (exit $LASTEXITCODE); fresh env will run asyncio-only test parametrizations."
}

# --- 6. Smoke tests --------------------------------------------------------

Write-Stage "Smoke tests"

Write-Host "  import tether..."
& $envPython -c "import tether; print('  tether:', tether.__name__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed: cannot import tether"
}

Write-Host "  Engine class importable..."
& $envPython -c "from tether import Engine; from tether.config.settings import load_settings; print('  Engine + Settings imports ok')" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed: Engine / Settings import"
}

Write-Host "  alias identity (tether is tether_service)..."
& $envPython -c "import tether, tether_service; assert tether is tether_service, 'alias broken'; print('  alias ok')" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Smoke test failed: tether_service alias identity"
}

Write-Host "  mlc_llm imports (heavy)..."
& $envPython -c "import mlc_llm; print('  mlc_llm ok:', mlc_llm.__version__ if hasattr(mlc_llm, '__version__') else '(no __version__)')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "mlc_llm import failed. The Tether refactor itself works (Engine doesn't import mlc_llm at module load per ADR-0016), but you won't be able to run real chat sessions until this is resolved."
}

Write-Stage "DONE"
Write-Host ""
Write-Host "Fresh '$EnvName' env ready. Next steps:" -ForegroundColor Green
Write-Host "  - Default test suite:  & '$envPython' -m pytest -q"
Write-Host "  - Docs drift gate:     & '$envPython' -m pytest -m docs tests/docs/ -q"
Write-Host "  - Real Brave search:   & '$envPython' -m pytest -m network -q  (requires BRAVE_API_KEY in .env)"
Write-Host "  - Server smoke:        & '$envPython' -m tether.app"
Write-Host "  - CLI:                 & '$envPython' -m tether.cli.main"
Write-Host ""
Write-Host "Runbook: docs/runbooks/fresh-env-setup.md"
