@echo off
setlocal

chcp 65001 >nul 2>nul
cd /d "%~dp0"

if "%SPRITEPIPE_HOST%"=="" set "SPRITEPIPE_HOST=127.0.0.1"
if "%SPRITEPIPE_PORT%"=="" set "SPRITEPIPE_PORT=7865"
if "%PYTHONUTF8%"=="" set "PYTHONUTF8=1"

where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv was not found on PATH.
  echo Install uv first, then run this file again:
  echo https://docs.astral.sh/uv/getting-started/installation/
  call :maybe_pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$hostName=$env:SPRITEPIPE_HOST; $port=[int]$env:SPRITEPIPE_PORT; try { $tcp=[Net.Sockets.TcpClient]::new(); $task=$tcp.ConnectAsync($hostName, $port); if ($task.Wait(500) -and $tcp.Connected) { $tcp.Close(); if ($env:SPRITEPIPE_NO_OPEN -ne '1') { Start-Process ('http://' + $hostName + ':' + $port + '/') }; exit 2 }; $tcp.Close(); exit 0 } catch { exit 0 }"
if errorlevel 2 (
  echo Sprite Motif web GUI already appears to be running.
  echo URL: http://%SPRITEPIPE_HOST%:%SPRITEPIPE_PORT%/
  call :maybe_pause
  exit /b 0
)

echo Starting Sprite Motif Pipeline web GUI...
echo URL: http://%SPRITEPIPE_HOST%:%SPRITEPIPE_PORT%/
echo Press Ctrl+C in this window to stop the server.
echo.

uv run --with-editable . spritepipe-gui --host "%SPRITEPIPE_HOST%" --port "%SPRITEPIPE_PORT%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo [ERROR] GUI exited with code %EXIT_CODE%.
) else (
  echo GUI stopped.
)
call :maybe_pause
exit /b %EXIT_CODE%

:maybe_pause
if /I not "%SPRITEPIPE_NO_PAUSE%"=="1" pause
exit /b 0
