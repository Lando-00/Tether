<#
.SYNOPSIS
    One command to bring up the whole Tether stack: GenieX -> Tether server -> CLI.

.DESCRIPTION
    Starts the operator-managed GenieX inference server, then the Tether HTTP
    service, waits for each to come up, and finally drops you into the
    interactive Tether CLI in this window.

    Services that are already running are detected and reused, never restarted
    or duplicated, and never stopped on exit -- so attaching to a long-lived
    GenieX server is safe. Only what this invocation started gets torn down.

    Ports, provider ids and the GenieX base URL are read from the real Tether
    settings (src/tether/config/default.yml plus any TETHER__* overrides) via
    scripts/dev/launch_config.py, so nothing here can drift from config.

.PARAMETER Command
    up      (default) Start whatever is missing, then run the CLI.
    status  Report health of GenieX and the Tether server, then exit.
    stop    Stop services previously started by this launcher, then exit.
    logs    Tail the background service logs, then exit.

.PARAMETER Python
    Python interpreter to run Tether with. Resolution order:
      -Python  ->  $env:TETHER_PYTHON  ->  active $env:VIRTUAL_ENV  ->
      .venv\Scripts\python.exe  ->  .venv-geniex\Scripts\python.exe  ->
      `python` on PATH.
    Machine-specific defaults belong in tether.local.ps1 (gitignored); copy
    tether.local.example.ps1 to get started.

.PARAMETER GenieXPath
    Path to geniex.exe. Defaults to `geniex` on PATH, then the standard
    install location under %LOCALAPPDATA%.

.PARAMETER DataDir
    GenieX data/model directory. Defaults to $env:GENIEX_DATADIR, then
    .\models\geniex if present.

.PARAMETER Compute
    GenieX compute backend: npu (default) or cpu.

.PARAMETER KeepAliveSeconds
    GenieX model keep-alive in seconds. 0 disables unloading between requests.

.PARAMETER Provider
    Provider id passed through to the CLI (--provider).

.PARAMETER Model
    Model id passed through to the CLI (--model).

.PARAMETER Mode
    Orchestrator mode passed through to the CLI (--mode): chat or research.
    Omit to let the server pick via orchestrator.default.

.PARAMETER CliArgs
    Extra arguments appended verbatim to the CLI invocation.

.PARAMETER TimeoutSeconds
    How long to wait for each service to come up. Default 120.

.PARAMETER NoGeniex
    Do not start GenieX (it is managed elsewhere, or the active provider is
    not GenieX).

.PARAMETER NoServer
    Do not start the Tether HTTP service.

.PARAMETER NoCli
    Start the services and exit without launching the CLI. Implies -KeepRunning.

.PARAMETER KeepRunning
    Leave started services running after the CLI exits.

.PARAMETER ShowWindows
    Give each background service its own visible console window instead of
    redirecting it to a log file.

.PARAMETER CliDebug
    Pass --debug to the CLI (shows raw stream events).

.PARAMETER Follow
    With `logs`, tail the most recently written log file live.

.EXAMPLE
    .\tether.ps1
    Bring everything up and start chatting.

.EXAMPLE
    .\tether.ps1 -Compute cpu -Model "unsloth/Qwen3-1.7B-GGUF:Q4_0"
    Run GenieX on CPU with a smaller model.

.EXAMPLE
    .\tether.ps1 -NoCli ; .\tether.ps1 status ; .\tether.ps1 stop
    Run the stack headless, inspect it, then tear it down.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("up", "status", "stop", "logs")]
    [string]$Command = "up",

    [string]$Python,
    [string]$GenieXPath,
    [string]$DataDir,

    [ValidateSet("npu", "cpu")]
    [string]$Compute = "npu",

    [ValidateRange(0, [int]::MaxValue)]
    [int]$KeepAliveSeconds = 300,

    [string]$Provider,
    [string]$Model,

    [ValidateSet("chat", "research")]
    [string]$Mode,

    [string[]]$CliArgs = @(),

    [ValidateRange(5, 3600)]
    [int]$TimeoutSeconds = 120,

    [switch]$NoGeniex,
    [switch]$NoServer,
    [switch]$NoCli,
    [switch]$KeepRunning,
    [switch]$ShowWindows,
    [switch]$CliDebug,
    [switch]$Follow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$RunDir = Join-Path $RepoRoot ".tether-run"
$LogDir = Join-Path $RunDir "logs"
$StateFile = Join-Path $RunDir "state.json"

# ---------------------------------------------------------------- output ----

function Write-Step { param([string]$Text) Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Ok { param([string]$Text) Write-Host "  ok  $Text" -ForegroundColor Green }
function Write-Note { param([string]$Text) Write-Host "      $Text" -ForegroundColor DarkGray }
function Write-Bad { param([string]$Text) Write-Host "  x   $Text" -ForegroundColor Red }

function Stop-WithError {
    param([string]$Message, [string[]]$Hints = @())
    Write-Bad $Message
    foreach ($h in $Hints) { Write-Host "      $h" -ForegroundColor Yellow }
    exit 1
}

# ------------------------------------------------------- local overrides ----

# tether.local.ps1 is gitignored and dot-sourced before any resolution, so a
# machine can pin its interpreter / data dir without touching tracked files.
$LocalConfig = Join-Path $RepoRoot "tether.local.ps1"
if (Test-Path $LocalConfig) {
    Write-Verbose "Loading local launcher config: $LocalConfig"
    . $LocalConfig
}

# ------------------------------------------------------------ resolution ----

function Resolve-PythonPath {
    $candidates = @()
    if ($Python) { $candidates += $Python }
    if ($env:TETHER_PYTHON) { $candidates += $env:TETHER_PYTHON }
    if ($env:VIRTUAL_ENV) { $candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe") }
    $candidates += (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    $candidates += (Join-Path $RepoRoot ".venv-geniex\Scripts\python.exe")

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c -PathType Leaf)) { return (Resolve-Path $c).Path }
    }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

function Assert-TetherImportable {
    param([string]$PythonExe)
    & $PythonExe -c "import tether" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "Python at '$PythonExe' cannot import 'tether'." @(
            "Create/activate the Tether environment, then:",
            "  <python> -m pip install -e `".[cli,dev]`"",
            "Or point the launcher at the right interpreter:",
            "  .\tether.ps1 -Python C:\path\to\venv\Scripts\python.exe",
            "  (or set `$env:TETHER_PYTHON, or create tether.local.ps1)"
        )
    }
}

function Resolve-GenieXPath {
    $candidates = @()
    if ($GenieXPath) { $candidates += $GenieXPath }
    $onPath = Get-Command geniex -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
    if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA "GenieX CLI\geniex.exe") }

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c -PathType Leaf)) { return (Resolve-Path $c).Path }
    }
    return $null
}

function Resolve-DataDir {
    $candidates = @()
    if ($DataDir) { $candidates += $DataDir }
    if ($env:GENIEX_DATADIR) { $candidates += $env:GENIEX_DATADIR }
    $candidates += (Join-Path $RepoRoot "models\geniex")

    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c -PathType Container)) { return (Resolve-Path $c).Path }
    }
    return $null
}

function Get-LaunchConfig {
    param([string]$PythonExe)
    $raw = & $PythonExe -m scripts.dev.launch_config 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        Stop-WithError "Could not read Tether settings via scripts/dev/launch_config.py." @(
            "Run it directly to see the underlying error:",
            "  $PythonExe -m scripts.dev.launch_config"
        )
    }
    return ($raw | Out-String | ConvertFrom-Json)
}

# ---------------------------------------------------------------- health ----

function Get-EndpointStatus {
    <#
      Returns the HTTP status code, or 0 when nothing is listening.
      A 4xx/5xx still means the service is up, which matters for /readyz:
      it answers 503 while a provider is degraded even though the server
      itself is perfectly fine.
    #>
    param([string]$Url, [int]$TimeoutSec = 3)
    try {
        $r = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing -ErrorAction Stop
        return [int]$r.StatusCode
    }
    catch {
        try {
            $response = $_.Exception.Response
            if ($response) { return [int]$response.StatusCode }
        }
        catch { }
        return 0
    }
}

function Test-ServiceUp {
    param([string]$Url, [int]$TimeoutSec = 3)
    return ((Get-EndpointStatus -Url $Url -TimeoutSec $TimeoutSec) -gt 0)
}

function Wait-ForService {
    param(
        [string]$Name,
        [string]$Url,
        [int]$TimeoutSec,
        [System.Diagnostics.Process]$Process
    )
    # A carriage-return spinner only makes sense on a live console; when stdout
    # is redirected (CI, `| Tee-Object`) it would smear across the log.
    $interactive = -not [Console]::IsOutputRedirected
    $blank = "`r" + (" " * 62) + "`r"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $spin = @('|', '/', '-', '\')
    $i = 0
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if ($Process -and $Process.HasExited) {
            if ($interactive) { Write-Host $blank -NoNewline }
            Write-Bad "$Name exited during startup (exit code $($Process.ExitCode))."
            return $false
        }
        if (Test-ServiceUp -Url $Url) {
            if ($interactive) { Write-Host $blank -NoNewline }
            return $true
        }
        if ($interactive) {
            Write-Host ("`r      {0} waiting for {1} ... {2}s " -f $spin[$i % 4], $Name, [int]$sw.Elapsed.TotalSeconds) -NoNewline
        }
        $i++
        Start-Sleep -Milliseconds 500
    }
    if ($interactive) { Write-Host $blank -NoNewline }
    return $false
}

# ----------------------------------------------------------------- state ----

function Get-State {
    if (-not (Test-Path $StateFile)) { return @{} }
    try {
        $obj = Get-Content $StateFile -Raw | ConvertFrom-Json
        $h = @{}
        foreach ($p in $obj.PSObject.Properties) { $h[$p.Name] = $p.Value }
        return $h
    }
    catch { return @{} }
}

function Save-State {
    param([hashtable]$State)
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    ($State | ConvertTo-Json -Depth 5) | Set-Content -Path $StateFile -Encoding UTF8
}

function Remove-StateEntry {
    param([string]$Key)
    $state = Get-State
    if ($state.ContainsKey($Key)) {
        $state.Remove($Key)
        Save-State $state
    }
}

# ------------------------------------------------------------- processes ----

function Start-BackgroundService {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    $common = @{
        FilePath         = $FilePath
        ArgumentList     = $Arguments
        WorkingDirectory = $WorkingDirectory
        PassThru         = $true
    }
    if ($ShowWindows) {
        $proc = Start-Process @common -WindowStyle Normal
        $outLog = $null
        $errLog = $null
    }
    else {
        $outLog = Join-Path $LogDir "$Name.out.log"
        $errLog = Join-Path $LogDir "$Name.err.log"
        # Truncate first so a failed start is never diagnosed against a
        # stale tail from a previous run.
        Set-Content -Path $outLog -Value "" -NoNewline -ErrorAction SilentlyContinue
        Set-Content -Path $errLog -Value "" -NoNewline -ErrorAction SilentlyContinue
        $proc = Start-Process @common -WindowStyle Hidden `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    }

    $state = Get-State
    $state[$Name] = @{
        pid     = $proc.Id
        started = (Get-Date).ToString("o")
        out_log = $outLog
        err_log = $errLog
    }
    Save-State $state
    return $proc
}

function Get-DescendantPids {
    param([int]$ProcessId)
    $result = @()
    try {
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction Stop
    }
    catch { return $result }
    foreach ($child in $children) {
        $result += [int]$child.ProcessId
        $result += Get-DescendantPids -ProcessId ([int]$child.ProcessId)
    }
    return $result
}

function Stop-ServiceTree {
    param([string]$Name, [int]$ProcessId)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Note "$Name (pid $ProcessId) already gone."
        Remove-StateEntry $Name
        return
    }
    # Children first: a dying parent would otherwise orphan them.
    foreach ($childPid in (Get-DescendantPids -ProcessId $ProcessId)) {
        Stop-Process -Id $childPid -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    Write-Ok "$Name stopped (pid $ProcessId)."
    Remove-StateEntry $Name
}

function Show-Tail {
    param([string]$Path, [int]$Lines = 20)
    if ($Path -and (Test-Path $Path)) {
        $content = Get-Content $Path -Tail $Lines -ErrorAction SilentlyContinue
        if ($content) {
            Write-Host "      --- $Path (last $Lines lines) ---" -ForegroundColor DarkGray
            $content | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
        }
    }
}

# -------------------------------------------------------------- commands ----

function Invoke-Status {
    param($Config)
    Write-Step "Status"

    if ($Config.geniex) {
        $code = Get-EndpointStatus -Url $Config.geniex.health_url
        $label = "GenieX  $($Config.geniex.base_url)"
        if ($code -eq 200) { Write-Ok "$label  [running]" }
        elseif ($code -gt 0) { Write-Bad "$label  [responding HTTP $code]" }
        else { Write-Bad "$label  [down]" }
        if ($Config.geniex.model_id) { Write-Note "model: $($Config.geniex.model_id)" }
    }
    else {
        Write-Note "GenieX  [not configured in providers.model_registry]"
    }

    $serverCode = Get-EndpointStatus -Url $Config.readyz_url
    if ($serverCode -eq 200) { Write-Ok "Tether  $($Config.origin)  [ready]" }
    elseif ($serverCode -gt 0) { Write-Bad "Tether  $($Config.origin)  [up, not ready - HTTP $serverCode]" }
    else { Write-Bad "Tether  $($Config.origin)  [down]" }

    $state = Get-State
    if ($state.Keys.Count -gt 0) {
        Write-Host ""
        Write-Note "Launcher-managed processes:"
        foreach ($k in $state.Keys) {
            $entry = $state[$k]
            $alive = [bool](Get-Process -Id $entry.pid -ErrorAction SilentlyContinue)
            $flag = if ($alive) { "alive" } else { "dead" }
            Write-Note ("{0,-14} pid {1}  [{2}]" -f $k, $entry.pid, $flag)
        }
    }
    return ($serverCode -eq 200)
}

function Invoke-Stop {
    Write-Step "Stopping launcher-managed services"
    $state = Get-State
    if ($state.Keys.Count -eq 0) {
        Write-Note "Nothing recorded in $StateFile."
        return
    }
    foreach ($k in @($state.Keys)) {
        Stop-ServiceTree -Name $k -ProcessId ([int]$state[$k].pid)
    }
}

function Invoke-Logs {
    Write-Step "Service logs"
    if (-not (Test-Path $LogDir)) {
        Write-Note "No logs yet ($LogDir does not exist)."
        Write-Note "Logs are only written when services run without -ShowWindows."
        return
    }
    $files = @(Get-ChildItem $LogDir -Filter *.log)
    if ($files.Count -eq 0) {
        Write-Note "No log files in $LogDir."
        return
    }

    if ($Follow) {
        # -Wait can only follow one file, so pick the busiest: the stderr log
        # of the most recently started service (uvicorn/structlog write there).
        $target = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Note "Following $($target.FullName)  (Ctrl+C to stop)"
        Write-Host ""
        Get-Content $target.FullName -Tail 30 -Wait
        return
    }

    # NOTE: do not filter on .Length -- Windows does not flush the directory
    # entry's size while a process still holds the file open, so a live log
    # reports 0 bytes. Read it instead and judge by what comes back.
    $printed = $false
    foreach ($f in $files) {
        $content = Get-Content $f.FullName -Tail 30 -ErrorAction SilentlyContinue
        if ($content) {
            Write-Host "      --- $($f.FullName) (last 30 lines) ---" -ForegroundColor DarkGray
            $content | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
            Write-Host ""
            $printed = $true
        }
    }
    if (-not $printed) { Write-Note "Log files are empty." }
    Write-Note "Full logs: $LogDir   (live tail: .\tether.ps1 logs -Follow)"
}

# ------------------------------------------------------------------ main ----

Push-Location $RepoRoot
try {
    if ($Command -eq "stop") { Invoke-Stop; exit 0 }
    if ($Command -eq "logs") { Invoke-Logs; exit 0 }

    $pythonExe = Resolve-PythonPath
    if (-not $pythonExe) {
        Stop-WithError "No Python interpreter found." @(
            "Pass one with -Python, set `$env:TETHER_PYTHON, or create a .venv in the repo."
        )
    }
    Assert-TetherImportable -PythonExe $pythonExe
    $config = Get-LaunchConfig -PythonExe $pythonExe

    if ($Command -eq "status") {
        $ready = Invoke-Status -Config $config
        exit $(if ($ready) { 0 } else { 1 })
    }

    Write-Host ""
    Write-Host "  Tether" -ForegroundColor Magenta -NoNewline
    Write-Host " - one-command launcher" -ForegroundColor DarkGray
    Write-Host ""
    Write-Note "python   : $pythonExe"
    Write-Note "provider : $($config.default_provider)"
    Write-Host ""

    $startedAnything = $false

    # --- 1. GenieX inference server --------------------------------------
    if ($NoGeniex) {
        Write-Step "GenieX skipped (-NoGeniex)"
    }
    elseif (-not $config.geniex) {
        Write-Step "GenieX not configured - skipping"
    }
    else {
        Write-Step "GenieX inference server"
        if ((Get-EndpointStatus -Url $config.geniex.health_url) -eq 200) {
            Write-Ok "already running at $($config.geniex.base_url) - reusing it"
            Write-Note "(started elsewhere, so this launcher will not stop it)"
        }
        else {
            $geniexExe = Resolve-GenieXPath
            if (-not $geniexExe) {
                Stop-WithError "GenieX CLI not found." @(
                    "Install it, or pass -GenieXPath <full path to geniex.exe>.",
                    "Skip GenieX entirely with -NoGeniex.",
                    "See docs/runbooks/geniex-provider.md"
                )
            }
            $resolvedDataDir = Resolve-DataDir
            if (-not $resolvedDataDir) {
                Stop-WithError "GenieX data directory not found." @(
                    "Pass -DataDir <path>, set `$env:GENIEX_DATADIR,",
                    "or create the models\geniex junction described in",
                    "docs/runbooks/geniex-provider.md"
                )
            }

            $uri = [Uri]$config.geniex.base_url
            $listen = "$($uri.Host):$($uri.Port)"
            Write-Note "exe      : $geniexExe"
            Write-Note "data dir : $resolvedDataDir"
            Write-Note "listen   : $listen  ($Compute, keepalive ${KeepAliveSeconds}s)"

            $geniexArgs = @(
                "--skip-update", "--data-dir", $resolvedDataDir, "serve",
                "--host", $listen,
                "--compute", $Compute,
                "--keepalive", "$KeepAliveSeconds"
            )
            $proc = Start-BackgroundService -Name "geniex" -FilePath $geniexExe `
                -Arguments $geniexArgs -WorkingDirectory $RepoRoot
            $startedAnything = $true

            if (Wait-ForService -Name "GenieX" -Url $config.geniex.health_url `
                    -TimeoutSec $TimeoutSeconds -Process $proc) {
                Write-Ok "GenieX ready at $($config.geniex.base_url)  (pid $($proc.Id))"
            }
            else {
                Write-Bad "GenieX did not become healthy within ${TimeoutSeconds}s."
                Show-Tail -Path (Join-Path $LogDir "geniex.err.log")
                Show-Tail -Path (Join-Path $LogDir "geniex.out.log")
                Invoke-Stop
                Stop-WithError "Aborting." @("Retry with -ShowWindows to watch GenieX start.")
            }
        }
    }

    # --- 2. Tether HTTP service ------------------------------------------
    if ($NoServer) {
        Write-Step "Tether HTTP service skipped (-NoServer)"
    }
    else {
        Write-Step "Tether HTTP service"
        $code = Get-EndpointStatus -Url $config.readyz_url
        if ($code -gt 0) {
            Write-Ok "already running at $($config.origin) - reusing it"
            if ($code -ne 200) { Write-Note "note: /readyz returned HTTP $code (degraded provider?)" }
        }
        else {
            Write-Note "listen   : $($config.origin)"
            $proc = Start-BackgroundService -Name "tether-server" -FilePath $pythonExe `
                -Arguments @("-m", "tether.app") -WorkingDirectory $RepoRoot
            $startedAnything = $true

            if (Wait-ForService -Name "Tether" -Url $config.readyz_url `
                    -TimeoutSec $TimeoutSeconds -Process $proc) {
                Write-Ok "Tether ready at $($config.origin)  (pid $($proc.Id))"
            }
            else {
                Write-Bad "Tether did not come up within ${TimeoutSeconds}s."
                Show-Tail -Path (Join-Path $LogDir "tether-server.err.log")
                Show-Tail -Path (Join-Path $LogDir "tether-server.out.log")
                Invoke-Stop
                Stop-WithError "Aborting."
            }
        }
    }

    if (-not $ShowWindows -and $startedAnything) {
        Write-Note "logs     : $LogDir   (tail them with: .\tether.ps1 logs)"
    }

    # --- 3. CLI -----------------------------------------------------------
    if ($NoCli) {
        Write-Host ""
        Write-Ok "Stack is up. Services left running (-NoCli)."
        Write-Note "Chat now  : .\tether.ps1"
        Write-Note "Inspect   : .\tether.ps1 status"
        Write-Note "Shut down : .\tether.ps1 stop"
        exit 0
    }

    Write-Step "Tether CLI"
    if ($startedAnything -and -not $KeepRunning) {
        Write-Note "Services started here stop when you leave the CLI (-KeepRunning to keep them)."
    }
    Write-Host ""

    $invocation = @("-m", "tether.cli.main", "--api-url", $config.api_base_url)
    if ($Provider) { $invocation += @("--provider", $Provider) }
    if ($Model) { $invocation += @("--model", $Model) }
    if ($Mode) { $invocation += @("--mode", $Mode) }
    if ($CliDebug) { $invocation += "--debug" }
    if ($CliArgs) { $invocation += $CliArgs }

    try {
        & $pythonExe @invocation
    }
    finally {
        if ($startedAnything -and -not $KeepRunning) {
            Write-Host ""
            Invoke-Stop
        }
        elseif ($startedAnything) {
            Write-Host ""
            Write-Note "Services left running (-KeepRunning). Stop with: .\tether.ps1 stop"
        }
    }
}
finally {
    Pop-Location
}
