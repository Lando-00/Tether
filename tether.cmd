@echo off
REM Thin shim so `tether` works from cmd.exe, Explorer, or a Run box.
REM All arguments are forwarded verbatim to tether.ps1:
REM     tether
REM     tether status
REM     tether -Compute cpu -NoCli
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tether.ps1" %*
