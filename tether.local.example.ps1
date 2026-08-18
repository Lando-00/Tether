# Machine-local defaults for .\tether.ps1
#
# Copy this file to `tether.local.ps1` (gitignored) and edit. The launcher
# dot-sources it before resolving anything, so any launcher parameter that was
# not passed explicitly on the command line can be defaulted here.
#
#   Copy-Item tether.local.example.ps1 tether.local.ps1
#
# Only set what you actually need -- every value below has a working default.

# Python interpreter used for the Tether server and CLI. Set this when your
# environment lives outside the repo (the launcher otherwise looks for
# .venv\Scripts\python.exe, then .venv-geniex, then `python` on PATH).
# $Python = "C:\path\to\venvs\tether-geniex\Scripts\python.exe"

# GenieX CLI executable, if it is not on PATH and not in the default
# %LOCALAPPDATA%\GenieX CLI\ install location.
# $GenieXPath = "C:\path\to\geniex.exe"

# GenieX data/model directory. Defaults to $env:GENIEX_DATADIR, then the
# .\models\geniex junction. Point it at your shared model store to skip the
# junction entirely.
# $DataDir = "C:\path\to\models\geniex"

# Compute backend for GenieX: "npu" (default) or "cpu".
# $Compute = "npu"

# Seconds GenieX keeps a model resident between requests. 0 unloads eagerly.
# $KeepAliveSeconds = 300

# Give each background service its own console window instead of log files.
# $ShowWindows = $true
