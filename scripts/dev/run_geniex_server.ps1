<#
.SYNOPSIS
    Launches a GenieX server in foreground for local development.

.DESCRIPTION
    Foreground-only helper that starts `geniex serve` with explicit parameters.
    The server is operator-managed: this script does NOT detach, auto-start on
    boot, pull models, mutate persistent environment, or kill processes.

    Prerequisites:
      - GenieX CLI installed and on PATH (or provide -GenieXPath)
      - Model already downloaded (e.g. `geniex pull unsloth/Qwen3-1.7B-GGUF:Q4_0`)
      - Data directory exists

.PARAMETER DataDir
    Path to the GenieX data directory containing downloaded models.
    Falls back to $env:GENIEX_DATA_DIR if not supplied.

.PARAMETER Host
    Listen address. Default: 127.0.0.1

.PARAMETER Port
    Listen port. Default: 18181

.PARAMETER Compute
    Compute backend. Default: npu
    Valid values: npu, cpu

.PARAMETER KeepAlive
    Keep-alive duration string (e.g. "5m", "0" to disable).
    Default: 5m

.PARAMETER GenieXPath
    Path to the geniex executable. Default: "geniex" (assumes on PATH).

.EXAMPLE
    .\run_geniex_server.ps1 -DataDir "$env:USERPROFILE\.geniex"
    .\run_geniex_server.ps1 -DataDir (Join-Path $HOME ".geniex") -Port 18182 -Compute cpu
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$DataDir,

    [Parameter()]
    [string]$Host = "127.0.0.1",

    [Parameter()]
    [int]$Port = 18181,

    [Parameter()]
    [ValidateSet("npu", "cpu")]
    [string]$Compute = "npu",

    [Parameter()]
    [string]$KeepAlive = "5m",

    [Parameter()]
    [string]$GenieXPath = "geniex"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Resolve data directory ---
if (-not $DataDir) {
    $DataDir = $env:GENIEX_DATA_DIR
}
if (-not $DataDir) {
    Write-Error @"
DataDir is required. Supply one of:
  -DataDir <path>              (parameter)
  `$env:GENIEX_DATA_DIR        (environment variable)
"@
    exit 1
}

if (-not (Test-Path $DataDir -PathType Container)) {
    Write-Error "DataDir does not exist or is not a directory: $DataDir"
    exit 1
}

# --- Validate geniex command ---
$geniexCmd = Get-Command $GenieXPath -ErrorAction SilentlyContinue
if (-not $geniexCmd) {
    Write-Error @"
GenieX CLI not found: '$GenieXPath'
Ensure it is installed and on PATH, or pass -GenieXPath <full-path>.
"@
    exit 1
}

Write-Host "GenieX server (foreground)" -ForegroundColor Cyan
Write-Host "  Executable : $($geniexCmd.Source)"
Write-Host "  DataDir    : $DataDir"
Write-Host "  Listen     : http://${Host}:${Port}"
Write-Host "  Compute    : $Compute"
Write-Host "  KeepAlive  : $KeepAlive"
Write-Host ""
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

# --- Launch in foreground (blocking) ---
# Use process-scoped env for GENIEX_DATA_DIR so we don't mutate the user's
# persistent environment.  The original value is restored on script exit.
$originalDataDir = $env:GENIEX_DATA_DIR
try {
    $env:GENIEX_DATA_DIR = $DataDir

    & $GenieXPath serve `
        --host $Host `
        --port $Port `
        --compute $Compute `
        --keep-alive $KeepAlive
}
finally {
    $env:GENIEX_DATA_DIR = $originalDataDir
}
